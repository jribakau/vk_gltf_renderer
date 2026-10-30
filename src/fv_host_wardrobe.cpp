/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// FvHost: what a figure wears (wear, wardrobe.*) and its dForce cloth.
//

#include "fv_host_impl.hpp"

//--------------------------------------------------------------------------------------------------
// Wardrobe

// Topic "wardrobe/<prop>": {figure, items, collision: {enabled, busy, corrected}}; `collision` is the
// collider's state (meshes it holds out of the body now), when the figure has one.
void FvHost::Impl::publishWardrobe(Loaded& f)
{
  Json items = Json::object();
  for(const fv::WornItem& w : f.fig->items)
  {
    const fv::ClothState cloth = f.fig->clothOf(w);
    items["i:" + std::to_string(w.id)] = {{"label", w.label},
                                          {"kind", w.kind},
                                          {"rel", w.rel},
                                          {"visible", w.visible},
                                          {"own", w.rel.empty()},
                                          {"channels", w.channels.size()},
                                          {"collision", !w.collision.empty()},
                                          {"cloth", {{"able", cloth.able}, {"native", cloth.native}, {"on", cloth.on}, {"strength", cloth.strength}}}};
    // its scene meshes, as the materials topic's items list theirs: which surfaces are this item's
    items["i:" + std::to_string(w.id)]["meshes"] = w.meshes;
    // the colour preset on it, null for its own; absent, as it was worn
    if(const auto c = f.colors.find(w.id); c != f.colors.end())
    {
      items["i:" + std::to_string(w.id)]["color"] = c->second.rel.empty() ? Json(nullptr) : Json(c->second.rel);
      if(c->second.automatic)
        items["i:" + std::to_string(w.id)]["colorAuto"] = true;
    }
  }
  Json topic = {{"figure", f.id}, {"items", items}};
  if(f.collider)
  {
    f.collisionVersion = f.collider->version();
    topic["collision"] = {{"enabled", f.collider->enabled()}, {"busy", f.collider->busy()}, {"corrected", f.collider->corrected()}};
  }
  protocol->events().snapshot("wardrobe/" + f.id, topic);
}

// wardrobe.collision {prop, on}: the worn conforming meshes are pushed out of the posed body (true)
// or shown as they conform (false).
Json FvHost::Impl::wardrobeCollision(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->collider)
    throw std::runtime_error("no figure with a body is selected");
  figure->collider->setEnabled(args.value("on", true));
  publishWardrobe(*figure);
  return {{"ok", true}};
}

Json FvHost::Impl::wear(const Json& args)
{
  HostSettings s;
  {
    std::lock_guard lock(settingsMutex);
    s = settings;
  }
  std::string rel;
  if(args.contains("rel"))
    rel = args["rel"].get<std::string>();
  else
  {
    const std::string path = args.at("path").get<std::string>();
    const auto        lib  = libraryFor(s, pathOf(path));
    if(!lib)
      throw std::runtime_error(path + " is not in a DAZ content library (no data folder above it)");
    rel = lib->second;
  }
  std::shared_ptr<Loaded> target = onApp([this, args] { return figureArg(args); });
  if(!target || !target->fig->pose)
    throw std::runtime_error("Select a figure first: clothing and hair go on a figure with a skeleton.");
  WearRequest request{rel, "", {}};
  request.step = true;
  if(const auto it = args.find("iray"); it != args.end() && it->is_array())
    for(const Json& c : *it)
      if(c.is_string())
        request.iray.push_back(c.get<std::string>());
  return {{"job", startWear(target, {std::move(request)})}};
}

// Puts wearables on a figure: all of them are loaded and bound side by side on workers against one
// copy of the figure, then merged on the application thread in request order, and the renderer's
// scene is composed again once at the end. A prepare reads only that copy, so the order of the
// merges (and the keys they give) is the order of the requests.
uint64_t FvHost::Impl::startWear(std::shared_ptr<Loaded> target, std::vector<WearRequest> requests, std::function<void()> beforeBuild, std::function<void()> after,
                                 std::shared_future<void> commitAfter, WearLooks looks)
{
  HostSettings s;
  std::shared_ptr<fv::dson::Library> lib;
  {
    std::lock_guard lock(settingsMutex);
    s   = settings;
    lib = target->lib;
  }
  const uint64_t    jobId = ++jobs;
  const std::string label = "Wearing " + utf8Of(pathOf(requests.front().rel).stem());
  job(jobId, {{"label", label}, {"state", "running"}});
  const fv::mat::SkinMode skin = skinMode.load();
  loaders.spawn([this, s, requests = std::move(requests), target, lib, jobId, label, skin, beforeBuild = std::move(beforeBuild), after = std::move(after),
                 commitAfter, looks = std::move(looks)] {
    // A batch that commits after something else (a restore after the figure's add-ons) prepares at
    // once and takes the wear lock only to commit; any other batch holds it throughout.
    std::unique_lock serial(wearMutex, std::defer_lock);
    if(!commitAfter.valid())
      serial.lock();
    bool any  = false;
    auto fail = [&](const std::string& what) {
      notice(what, "error");
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", what}});
    };
    // A character lists material presets with its add-ons, but one preset often covers several (an
    // "all add-ons" preset on the eyebrows holds the eyes' and lashes' surfaces too), as DAZ applies
    // them to the whole figure: every add-on takes all of them, each surface where its node is.
    std::vector<std::string> addonPresets;
    for(const WearRequest& r : requests)
      if(!r.name.empty())
        for(const std::string& p : r.presets)
          if(std::find(addonPresets.begin(), addonPresets.end(), p) == addonPresets.end())
            addonPresets.push_back(p);
    std::mutex            presetMutex;  // the prepares run side by side
    std::set<std::string> presetMissing, presetMatched;
    auto applyPreset = [&](const std::string& presetRel, fv::SceneGraph& scene, bool report) {
      try
      {
        if(auto doc = target->fig->store().load(presetRel))
        {
          const fv::dson::MaterialPresetResult r = fv::dson::applyMaterialPreset(doc->json, scene, nullptr, fv::dson::definitionsIn(doc->json, target->fig->store()));
          if(report)
          {
            std::lock_guard lock(presetMutex);
            presetMissing.insert(r.missing.begin(), r.missing.end());
            presetMatched.insert(r.matched.begin(), r.matched.end());
          }
        }
      }
      catch(const std::exception& e)
      {
        notice(presetRel + ": " + e.what());
      }
    };
    using Clock          = std::chrono::steady_clock;
    const auto msSince   = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    // textures, the way the figure's were made: cooked where the cache has them (the rest cooked in
    // place once it is worn), or the sources; several at once
    const auto cook = [&](const std::vector<fv::build::TextureRequest>& requestsOf, std::vector<fv::build::TextureRequest>& uncooked) {
      auto found = resolveTextures(s, lib.get(), requestsOf, target->texturesFlipped, uncooked);
      prefetchImages(requestsOf, found);
      return found;
    };
    const auto tBatch    = Clock::now();
    // what is put on: an add-on whose node the figure already has is not put on twice
    // (by value: a timed-out onApp drops its task, but one already running finishes after this moves on)
    std::vector<const WearRequest*> todo;
    std::shared_ptr<fv::Figure::WearInputs> inputs;
    double                                  msBase = 0;
    try
    {
      std::vector<std::string> names;
      for(const WearRequest& r : requests)
        names.push_back(r.name);
      const auto worn = onApp([this, target, names] {
        std::vector<char> out(names.size(), 0);
        if(inScene(target))
          for(size_t i = 0; i < names.size(); i++)
            out[i] = !names[i].empty()
                     && std::any_of(target->fig->items.begin(), target->fig->items.end(),
                                    [&](const fv::WornItem& item) { return item.nodeNames.count(names[i]) > 0; });
        return out;
      });
      std::set<std::string> taken;
      for(size_t i = 0; i < requests.size(); i++)
        if(!worn[i] && (requests[i].name.empty() || taken.insert(requests[i].name).second))
          todo.push_back(&requests[i]);
      // the figure as it is now, copied on the application thread once for every prepare of the
      // batch: dials move its joint centres meanwhile, and commits grow it
      if(!todo.empty())
      {
        fv::timing::Scope t("wearBase (render thread)");
        const auto        tBase = Clock::now();
        inputs                  = std::make_shared<fv::Figure::WearInputs>(onApp([target] { return target->fig->wearInputs(); }));
        msBase                  = msSince(tBase);
      }
    }
    catch(const std::exception& e)
    {
      todo.clear();
      notice(e.what(), "error");
    }

    // every request prepared side by side, each on a thread of its own (a prepare waits on pool jobs,
    // so not on the pool): loaded, fitted and bound, its presets, strands and textures
    struct Slot
    {
      std::shared_ptr<fv::Figure::PreparedWear>                               prepared;
      std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
      std::vector<fv::build::TextureRequest>                                  uncooked;  // shown with their fallbacks
      double                                                                  msPrep = 0, msStrands = 0, msWorker = 0;
      std::string                                                             error;
      // the Iray preset it took as it went on (WearRequest::iray), and its document
      std::string                                autoColor;
      std::shared_ptr<const fv::dson::Doc>       autoColorDoc;
    };
    std::vector<Slot> slots(todo.size());
    auto              prepare = [&](size_t i) {
      const WearRequest& request = *todo[i];
      Slot&              slot    = slots[i];
      const auto         tWear   = Clock::now();
      try
      {
        if(!accepting)
          return;
        job(jobId, {{"label", utf8Of(pathOf(request.rel).stem()) + ": loading"}, {"state", "running"}});
        const auto tPrep = Clock::now();
        slot.prepared    = target->fig->prepareWear(request.rel, s.subdivision, *inputs, target->texturesFlipped);
        slot.msPrep      = msSince(tPrep);
        // the material presets that go with it, applied to the wearable before its textures are looked up
        fv::SceneGraph& preparedScene = fv::Figure::preparedScene(*slot.prepared);
        {
          fv::timing::Scope t("material presets");
          if(const std::string brows = defaultBrowMaterial(preparedScene); !brows.empty())
            applyPreset(brows, preparedScene, false);
          for(const std::string& presetRel : request.name.empty() ? request.presets : addonPresets)
            applyPreset(presetRel, preparedScene, true);
          // as the character came: "Character's own" eyes are these (Genesis 9's are an add-on's)
          fv::dson::keepBaseline(preparedScene);
          // a 3Delight-era item takes the installed Iray preset that keeps its look, over its
          // baseline: its textures are cooked once, and its colour "own" takes it off again
          if(!request.iray.empty())
          {
            std::vector<std::shared_ptr<const fv::dson::Doc>> docs;
            std::vector<const fv::dson::Json*>                candidates;
            for(const std::string& c : request.iray)
            {
              std::shared_ptr<const fv::dson::Doc> doc;
              try
              {
                doc = target->fig->store().load(c);
              }
              catch(const std::exception&)
              {
              }
              docs.push_back(doc);
              candidates.push_back(doc ? &doc->json : nullptr);
            }
            if(const int k = fv::dson::chooseIrayPreset(preparedScene, candidates); k >= 0)
            {
              std::set<int> all;
              for(size_t m = 0; m < preparedScene.materials.size(); m++)
                all.insert(int(m));
              fv::dson::choosePreset(preparedScene, all, &docs[size_t(k)]->json, {},
                                     fv::dson::definitionsIn(docs[size_t(k)]->json, target->fig->store()));
              slot.autoColor    = request.iray[size_t(k)];
              slot.autoColorDoc = docs[size_t(k)];
            }
          }
        }
        // a strand hair's direction is measured here, so the composition on the render thread finds it
        const auto tStrands = Clock::now();
        {
          fv::timing::Scope t("strand directions");
          fv::build::measureStrands(preparedScene, target->texturesFlipped);
        }
        slot.msStrands = msSince(tStrands);
        // the wearable's textures
        job(jobId, {{"label", label + ": Textures"}, {"state", "running"}});
        fv::timing::Scope t("textures");
        const auto        requestsOf = fv::build::textureRequests(preparedScene, skin);
        const auto        found      = cook(requestsOf, slot.uncooked);
        for(size_t k = 0; k < requestsOf.size(); k++)
          slot.images[fv::build::requestKey(requestsOf[k])] = found[k];
      }
      catch(const std::exception& e)
      {
        slot.prepared.reset();
        slot.error = e.what();
      }
      slot.msWorker = msSince(tWear);
    };
    {
      std::vector<std::thread> threads;
      for(size_t i = 1; i < todo.size(); i++)
        threads.emplace_back(prepare, i);
      if(!todo.empty())
        prepare(0);
      for(auto& t : threads)
        t.join();
    }

    // committed in request order, after what this batch waits for
    if(commitAfter.valid())
    {
      while(commitAfter.wait_for(100ms) != std::future_status::ready)
        if(!accepting)
          return;
      serial.lock();
    }
    for(size_t i = 0; i < todo.size(); i++)
    {
      const WearRequest& request = *todo[i];
      Slot&              slot    = slots[i];
      if(!accepting)
        return;
      if(!slot.prepared)
      {
        if(!slot.error.empty())
        {
          const std::string problem = utf8Of(pathOf(request.rel).stem()) + ": " + slot.error;
          notice(problem, "error");
          if(looks.problems)
            post([problems = looks.problems, problem] { problems->push_back(problem); });
        }
        continue;
      }
      auto done = std::make_shared<std::promise<bool>>();
      auto ok   = done->get_future();
      post([this, target, prepared = slot.prepared, rel = request.rel, ids = request.ids, keys = request.keys, only = request.only, step = request.step, label,
            images = std::move(slot.images), uncooked = std::move(slot.uncooked), done, autoColor = slot.autoColor,
            autoColorDoc = slot.autoColorDoc, problems = looks.problems]() mutable {
        if(!inScene(target))
        {
          done->set_value(false);
          return;
        }
        if(const std::string refused = renderer->refuseGeometry(geometryBytes(fv::Figure::preparedScene(*prepared))); !refused.empty())
        {
          const std::string problem = utf8Of(pathOf(rel).stem()) + " wasn't added: " + refused;
          notice(problem, "error");
          if(problems)
            problems->push_back(problem);
          done->set_value(false);
          return;
        }
        try
        {
          // what it may replace (the same garment again), as it was, for the step's undo
          std::map<int, Json> had;
          if(step)
            for(const fv::WornItem& w : target->fig->items)
              had[w.id] = itemRecord(*target, w);
          fv::WearResult r = target->fig->commitWear(prepared, rel, ids, keys, only);
          if(only)
          {
            // one item of the file put back: the others are worn already, or were not
            std::vector<int> kept;
            for(int id : r.items)
              if(std::find(ids.begin(), ids.end(), id) != ids.end())
                kept.push_back(id);
              else
                target->fig->removeItem(id);
            r.items = std::move(kept);
          }
          if(step && !r.items.empty())
          {
            Json        undo = Json::array(), redo = Json::array();
            std::string what;
            for(int id : r.items)
              if(const fv::WornItem* w = wornItem(*target->fig, id))
              {
                undo.push_back({{"cmd", "wardrobe.remove"}, {"args", {{"prop", target->id}, {"item", id}}}});
                redo.push_back({{"cmd", "wardrobe.restore"}, {"args", {{"prop", target->id}, {"item", id}, {"record", itemRecord(*target, *w)}}}});
                if(what.empty())
                  what = w->label.empty() ? utf8Of(pathOf(rel).stem()) : w->label;
              }
            for(int id : r.replaced)
              if(had.count(id))
                undo.push_back({{"cmd", "wardrobe.restore"}, {"args", {{"prop", target->id}, {"item", id}, {"record", had[id]}}}});
            journal.record({.redo = redo, .undo = undo, .label = "Wear " + what, .at = std::chrono::steady_clock::now()});
            publishHistory();
          }
          for(auto& [key, image] : images)
            target->images.insert_or_assign(key, std::move(image));
          if(!uncooked.empty())
            cookInPlace(target, std::move(uncooked));
          // the Iray preset it took is its colour: undo, a scene file and "own" treat it as one
          if(!autoColor.empty() && autoColorDoc)
          {
            const fv::SceneGraph& scene   = target->fig->loaded.scene;
            const std::set<int>   targets = fv::dson::materialPresetTargets(autoColorDoc->json, scene);
            for(int id : r.items)
              for(const fv::WornItem& w : target->fig->items)
                if(w.id == id)
                {
                  std::set<int> owned;
                  for(int m : w.meshes)
                    if(m >= 0 && m < int(scene.meshes.size()))
                      for(int mi : scene.meshes[size_t(m)].materials)
                        if(targets.count(mi))
                          owned.insert(mi);
                  if(!owned.empty())
                    target->colors[id] = {autoColor, owned, true};
                }
            publishWardrobe(*target);
            notice(utf8Of(pathOf(rel).stem()) + " is a 3Delight item: it took its installed Iray material, " + utf8Of(pathOf(autoColor).stem()) + ".", "info");
          }
          for(const std::string& w : r.warnings)
            notice(w);
          for(const std::string& n : r.notes)
            notice(n, "info");
          if(!r.missingBones.empty())
            notice(utf8Of(pathOf(rel).stem()) + ": this figure has no " + r.missingBones.front()
                   + (r.missingBones.size() > 1 ? " and " + std::to_string(r.missingBones.size() - 1) + " other bones" : "") + " to hang it from.");
          if(r.fit < 0.5)
            notice(utf8Of(pathOf(rel).stem())
                   + ": most of its bones found no match on this figure, so it was probably made for another figure and will not follow the pose.");
          done->set_value(true);
        }
        catch(const std::exception& e)
        {
          notice(e.what(), "error");
          done->set_value(false);
        }
      });
      // the application thread merges it; a renderer that is closing never will
      const auto tCommit = Clock::now();
      {
        fv::timing::Scope timeCommit("commit (render thread)");
        while(ok.wait_for(100ms) != std::future_status::ready)
          if(!accepting)
            return;
      }
      any = ok.get() || any;
      if(fv::timing::enabled())
      {
        const double msCommit = msSince(tCommit);
        char         buf[256];
        std::snprintf(buf, sizeof buf, "wear summary: worker %.0f ms (prepare %.0f, strands %.0f), render thread %.0f ms (wearBase %.0f, commit %.0f)",
                      slot.msWorker, slot.msPrep, slot.msStrands, msBase + msCommit, msBase, msCommit);
        fv::timing::line(buf);
        fv::timing::report("wear " + utf8Of(pathOf(request.rel).stem()), msSince(tBatch));
      }
    }
    // surfaces no worn item had; those of the figure itself took the character's preset at load
    for(const std::string& s : presetMatched)
      presetMissing.erase(s);
    if(!presetMissing.empty())
    {
      const auto unknown = onApp([target, names = presetMissing] {
        std::vector<std::string> out;
        for(const std::string& n : names)
          if(std::none_of(target->fig->loaded.scene.materials.begin(), target->fig->loaded.scene.materials.end(),
                          [&](const fv::Material& m) { return m.name == n; }))
            out.push_back(n);
        return out;
      });
      if(!unknown.empty())
      {
        std::string list;
        for(size_t i = 0; i < unknown.size() && i < 6; i++)
          list += (i ? ", " : "") + unknown[i];
        if(unknown.size() > 6)
          list += " and " + std::to_string(unknown.size() - 6) + " more";
        notice("Material presets name surfaces nothing worn has: " + list, "info");
      }
    }
    // `after` runs whatever happened, so a restore waiting on this wear still finishes
    if(!any)
    {
      fail("Nothing was put on.");
      post([looks, after] {
        if(looks.shape)
          looks.shape();
        if(looks.surfaces)
          looks.surfaces();
        if(after)
          after();
      });
      return;
    }
    // A restore's surfaces go on now, held from the scene that is about to go, and the textures
    // they ask for are made here; its dials and pose go on, settled, just before the build. The
    // composition then builds the figure as the file has it: nothing is lowered, uploaded or
    // imported again on the application thread after it.
    std::vector<fv::build::TextureRequest>              lookRequests;
    std::vector<std::optional<fv::build::TextureImage>> lookImages;
    if(looks.surfaces)
    {
      fv::timing::Scope t("restore surfaces and their textures");
      try
      {
        lookRequests = onApp([this, target, surfaces = looks.surfaces] {
          std::vector<fv::build::TextureRequest> missing;
          if(!inScene(target))
            return missing;
          if(const auto p = findProp(target->id); p && p->materials)
            p->materials->hold();
          surfaces();
          for(fv::build::TextureRequest& r : fv::build::textureRequests(target->fig->loaded.scene, skinMode.load()))
            if(!target->images.count(fv::build::requestKey(r)))
              missing.push_back(std::move(r));
          return missing;
        });
        // as the materials would have baked them, side by side
        lookImages.resize(lookRequests.size());
        std::atomic<size_t>      next{0};
        std::vector<std::thread> workers;
        for(int w = 0; w < std::min<int>(8, int(lookRequests.size())); w++)
          workers.emplace_back([&] {
            for(size_t k; (k = next++) < lookRequests.size();)
              lookImages[k] = bakeTexture(lib, lookRequests[k], target->texturesFlipped);
          });
        for(auto& w : workers)
          w.join();
        prefetchImages(lookRequests, lookImages);
      }
      catch(const std::exception& e)
      {
        notice(e.what(), "error");
      }
    }
    post([this, target, jobId, label, beforeBuild, after, looks, lookRequests = std::move(lookRequests), lookImages = std::move(lookImages)]() mutable {
      const auto releaseHeld = [&](bool refresh) {
        if(const auto p = findProp(target->id); p && p->figure == target && p->materials)
          p->materials->release(refresh);
      };
      if(!inScene(target))
      {
        job(jobId, {{"label", label}, {"state", "failed"}, {"error", "the figure left the scene"}});
        if(after)
          after();
        return;
      }
      for(size_t k = 0; k < lookRequests.size() && k < lookImages.size(); k++)
        target->images.insert_or_assign(fv::build::requestKey(lookRequests[k]), std::move(lookImages[k]));
      if(looks.shape)
      {
        fv::timing::Scope t("restore dials and pose, settled");
        {
          fv::timing::Scope ts("set", 5);
          looks.shape();
        }
        {
          fv::timing::Scope ts("settle (their files)", 5);
          target->fig->settle();
        }
        fv::timing::Scope tp("pump (into the meshes)", 5);
        target->fig->pump();  // into the meshes and binds the build reads; the composition drains the rest
      }
      // the composition's attach takes the surfaces from the build
      releaseHeld(false);
      if(beforeBuild)
        beforeBuild();
      requestCompose(label, [this, jobId, label, after] {
        job(jobId, {{"label", label}, {"state", "done"}});
        if(after)
          after();
      }, target.get());
    });
  });
  return jobId;
}

Json FvHost::Impl::itemRecord(const Loaded& f, const fv::WornItem& w) const
{
  Json rec = {{"rel", w.rel}, {"piece", pieceOfKey(w.key)}, {"key", w.key}, {"label", w.label}, {"visible", w.visible},
              {"cloth", {{"on", w.cloth.on ? Json(*w.cloth.on) : Json()}, {"strength", w.cloth.strength}}}};
  // its dials as they are set: they leave with it
  Json dials = Json::object();
  for(const std::string& name : w.channels)
    if(const fv::morph::Channel* c = f.fig->morphs->channel(name); c && !c->hidden && c->value != c->loaded)
      dials[name] = c->value;
  if(!dials.empty())
    rec["dials"] = std::move(dials);
  if(const auto c = f.colors.find(w.id); c != f.colors.end())
    rec["color"] = c->second.rel.empty() ? Json() : Json(c->second.rel);
  // its surfaces' edits, in the order its meshes have them
  Json                  edits = Json::array();
  const auto            p     = findProp(f.id);
  const fv::SceneGraph& scene = f.fig->loaded.scene;
  std::set<int>         seen;
  for(int m : w.meshes)
    if(m >= 0 && m < int(scene.meshes.size()))
      for(int mi : scene.meshes[size_t(m)].materials)
        if(seen.insert(mi).second)
          edits.push_back(p && p->materials ? p->materials->editsOf(mi) : Json::object());
  rec["materials"] = edits;
  return rec;
}

Json FvHost::Impl::wardrobeRestore(const Json& args)
{
  const auto p = figureProp(args);
  if(!p || !p->figure->fig->pose)
    throw std::runtime_error("no such figure");
  std::shared_ptr<Loaded> target = p->figure;
  const int               id     = itemId(args);
  const Json&             rec    = args.at("record");
  if(wornItem(*target->fig, id))
    throw std::runtime_error("i:" + std::to_string(id) + " is worn already");
  WearRequest request{rec.at("rel").get<std::string>(), "", {}};
  request.ids.assign(size_t(std::max(rec.value("piece", 0), 0)) + 1, 0);
  request.ids.back() = id;
  request.keys.assign(request.ids.size(), "");
  request.keys.back() = rec.value("key", std::string());
  request.only        = true;
  // as it was: shown or hidden and its cloth before the build, its colour and edits once its surfaces are in
  auto before = [this, target, id, rec] {
    if(!wornItem(*target->fig, id))
      return;
    target->fig->setItemVisible(id, rec.value("visible", true));
    const Json&         c = rec.value("cloth", Json::object());
    std::optional<bool> on;
    if(c.contains("on") && c["on"].is_boolean())
      on = c["on"].get<bool>();
    target->fig->setItemCloth(id, on, c.value("strength", 1.f));
    std::vector<std::pair<std::string, double>> dials;
    const Json saved = rec.value("dials", Json::object());
    for(const auto& [name, value] : saved.items())
      if(target->fig->morphs->channel(name))
        dials.emplace_back(name, value.get<double>());
    if(!dials.empty())
      target->fig->morphs->setMany(dials);
  };
  auto after = [this, target, id, rec] {
    const auto q = findProp(target->id);
    const fv::WornItem* w = wornItem(*target->fig, id);
    if(!q || q->figure != target || !w)
      return;
    if(rec.contains("color") && (rec["color"].is_string() || rec["color"].is_null()))
    {
      const std::string rel = rec["color"].is_string() ? rec["color"].get<std::string>() : std::string();
      std::shared_ptr<const fv::dson::Doc> doc;
      if(rel.empty() || (doc = target->fig->store().load(rel)))
        applyItemColor(*q, {id}, rel, doc ? &doc->json : nullptr, true);
    }
    if(q->materials)
    {
      const fv::SceneGraph& scene = target->fig->loaded.scene;
      std::vector<int>      mats;
      std::set<int>         seen;
      for(int m : w->meshes)
        if(m >= 0 && m < int(scene.meshes.size()))
          for(int mi : scene.meshes[size_t(m)].materials)
            if(seen.insert(mi).second)
              mats.push_back(mi);
      const Json& edits = rec.value("materials", Json::array());
      for(size_t k = 0; k < mats.size() && k < edits.size(); k++)
        if(edits[k].is_object() && !edits[k].empty())
          q->materials->set({{"material", mats[k]}, {"values", edits[k]}});
    }
    publishWardrobe(*target);
  };
  return {{"job", startWear(target, {std::move(request)}, before, after)}, {"prop", target->id}};
}

// Taken off, an item is hidden in the renderer's scene and stays there, buffers and BLAS, until the
// next composition (a wear, a geograft shown or hidden, a prop added) leaves it out. A geograft gives
// the body its polygons back, which changes the body's indices: that still composes the scene again.
Json FvHost::Impl::wardrobeRemove(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const int           id   = itemId(args);
  const fv::WornItem* item = wornItem(*figure->fig, id);
  if(!item)
    throw std::runtime_error("no such item");
  const bool geograft = !item->hiddenPolys.empty();
  const auto objects  = item->objects;
  figure->fig->removeItem(id);
  // shoes give the feet their pose back
  if(figure->fig->pose)
    protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
  if(geograft)
  {
    requestCompose("Taking off", {}, figure.get());
    return {{"ok", true}};
  }
  showItem(*figure, objects, false);
  if(figure->collider)
    figure->collider->poke();
  publishWardrobe(*figure);
  if(figure->dforce)
    figure->dforce->poke();
  return {{"ok", true}};
}

// wardrobe.cloth {prop, item, on?: bool | null, strength?}: dForce on one item (figure.hpp,
// clothSettings): true drapes it, at the defaults when its file has no simulation settings, false
// holds it, null leaves it to the file; `strength` scales the weights. Turning an item on turns the
// figure's dForce on, so the choice shows.
Json FvHost::Impl::wardrobeCloth(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const int           id   = itemId(args);
  const fv::WornItem* item = wornItem(*figure->fig, id);
  if(!item)
    throw std::runtime_error("no such item");
  std::optional<bool> on = item->cloth.on;
  if(args.contains("on"))
    on = args["on"].is_boolean() ? std::optional<bool>(args["on"].get<bool>()) : std::nullopt;
  const float strength = args.value("strength", item->cloth.strength);
  const auto  changed  = figure->fig->setItemCloth(id, on, strength);
  if(figure->dforce)
  {
    if(on && *on && !figure->dforce->enabled())
      figure->dforce->setEnabled(true);
    figure->dforce->changed(changed, false);
    figure->dforce->poke();
  }
  if(figure->collider)
    figure->collider->poke();
  publishWardrobe(*figure);
  if(figure->dforce)
    publishDforce(*figure, true);
  return {{"ok", true}};
}

Json FvHost::Impl::wardrobeVisible(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const int           id      = itemId(args);
  const bool          visible = args.at("visible").get<bool>();
  const fv::WornItem* item    = wornItem(*figure->fig, id);
  if(!item)
    throw std::runtime_error("no such item");
  figure->fig->setItemVisible(id, visible);
  if(!item->hiddenPolys.empty())
  {
    requestCompose("Showing", {}, figure.get());
    return {{"ok", true}};
  }
  showItem(*figure, item->objects, visible);
  if(figure->collider)
    figure->collider->poke();
  publishWardrobe(*figure);
  if(figure->dforce)
    figure->dforce->poke();
  return {{"ok", true}};
}

// An item's nodes shown or hidden in the renderer's scene as they are, without composing it again
// (taken off, they go at the next composition's feed).
void FvHost::Impl::showItem(const Loaded& f, const std::vector<std::pair<int, std::string>>& objects, bool visible)
{
  if(!f.feed || !renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  for(const auto& [node, bone] : objects)
    if(const rs::EntityId e = f.feed->entityOf(node); scene.entity(e))
      scene.setVisible(e, visible);
}

//--------------------------------------------------------------------------------------------------
// dForce

// Topic "dforce/<prop>": {figure, enabled, mode, active, busy, draped}, when it changes.
void FvHost::Impl::publishDforce(Loaded& f, bool force)
{
  if(!f.dforce)
    return;
  const fv::dforce::Dforce& d = *f.dforce;
  if(!force && f.dforceVersion == d.version())
    return;
  f.dforceVersion = d.version();
  protocol->events().snapshot("dforce/" + f.id, {{"figure", f.id},
                                                 {"enabled", d.enabled()},
                                                 {"mode", d.mode() == fv::dforce::Mode::eLive ? "live" : d.mode() == fv::dforce::Mode::eLiveGpu ? "gpu" : "settle"},
                                                 {"active", d.active()},
                                                 {"busy", d.busy()},
                                                 {"draped", d.draped()}});
}

// dforce.enable {prop, on}: cloth drapes (true) or shows the conforming shape (false);
// dforce.mode {prop, mode}: "settle" (once the pose stops changing), "live" (every frame) or "gpu"
// (every frame, on the experimental GPU solver, in a build with FV_DFORCE_GPU);
// dforce.resimulate {prop} settles the cloth now.
Json FvHost::Impl::dforceCommand(const std::string& cmd, const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->dforce)
    throw std::runtime_error("no figure with dynamic clothing is selected");
  if(cmd == "enable")
  {
    figure->dforce->setEnabled(args.value("on", true));
    // a mesh the simulation takes over (or gives back) changes hands with the collider
    if(figure->collider)
      figure->collider->poke();
  }
  else if(cmd == "mode")
    figure->dforce->setMode(dforceMode(args.value("mode", "settle")));
  else
    figure->dforce->resimulate();
  publishDforce(*figure, true);
  return {{"ok", true}};
}
