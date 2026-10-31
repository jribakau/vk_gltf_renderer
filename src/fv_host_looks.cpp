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
// FvHost: pose and gizmo edits, presets, and the eyes, colours, skin and makeup chosen for a figure.
//

#include "fv_host_impl.hpp"

namespace {

// The shader definitions a preset's surfaces name, from the figure's store; none without a preset.
fv::dson::MaterialDefinitions definitionsOf(const fv::Figure& fig, const fv::dson::Json* doc)
{
  return doc ? fv::dson::definitionsIn(*doc, fig.store()) : fv::dson::MaterialDefinitions{};
}

}  // namespace

std::pair<std::shared_ptr<Loaded>, const fv::pose::PoseBone*> FvHost::Impl::boneAt(rs::EntityId entity) const
{
  for(const auto& p : props)
  {
    const auto& f = p->figure;
    if(!f || !f->fig->pose)
      continue;
    const auto it = f->nodeOf.find(entity);
    if(it == f->nodeOf.end())
      continue;
    const auto* pb = f->fig->pose->boneForNode(it->second);
    return pb && !pb->axes.empty() ? std::pair{f, pb} : std::pair{nullptr, nullptr};
  }
  return {nullptr, nullptr};
}

void FvHost::Impl::applyGizmoEdit()
{
  if(!gizmoEdit)
    return;
  const GizmoEdit edit = *std::exchange(gizmoEdit, std::nullopt);
  if(!inScene(edit.figure) || !edit.figure->fig->pose)
    return;
  // nodes, correctives and rebound skins go out with this frame's pump(); the drag is one step
  const Json args = {{"prop", edit.figure->id}, {"bone", edit.bone}, {"gesture", {{"id", "gizmo:" + std::to_string(edit.gesture)}}}};
  journaled("pose.drag", args, [&] {
    edit.figure->fig->pose->setLocalTransform(edit.bone, edit.rotation, edit.position);
    patchPose(*edit.figure, {edit.bone});
    return Json();
  });
}

void FvHost::Impl::patchPose(const Loaded& f, const std::vector<std::string>& boneIds)
{
  if(!f.fig->pose)
    return;
  Json bones = Json::object();
  for(const std::string& id : boneIds)
    if(const auto* pb = f.fig->pose->get(id))
    {
      Json axes = Json::object();
      for(const auto& a : pb->axes)
        axes[axisKey(a)] = {{"value", a.value}};
      bones[pb->id] = {{"axes", axes}};
    }
  protocol->events().patch("pose/" + f.id, {{"changed", f.fig->pose->changed()}, {"bones", bones}});
}

void FvHost::Impl::applyPreset(const fv::dson::Json&                             doc,
                               const std::string&                                kind,
                               uint64_t                                          jobId,
                               const std::string&                                rel,
                               const std::optional<fv::dson::EnvironmentPreset>& environment,
                               const std::string&                                environmentPath,
                               const std::string&                                target)
{
  const std::string name = fileStem(rel);
  if(environment)
  {
    // as applyEnvironmentPreset (src/lib/actions.ts): the dome lights the scene, so the sun goes off
    const std::string hdrName = utf8Of(pathOf(normRel(environment->image)).filename());
    if(environmentPath.empty())
      notice(name + ": its environment image " + environment->image + " isn't in the content library.", "error");
    else
    {
      Json env = {{"source", "hdr"},
                  {"path", environmentPath},
                  {"intensity", environment->intensity.value_or(1.0)},
                  {"rotation", fv::lighting::presetRotation(environment->rotation.value_or(0.0))}};
      if(environment->drawDome.value_or(false))
        env["background"] = "environment";
      const Json patch = {{"env", env}, {"sun", {{"enabled", false}}}};
      journaled("lighting.set", patch, [&] { return lightingSet(patch); });
      notice("Applied lighting preset " + name + " (" + hdrName + "). The sun light is off; turn it back on in the Light tab.", "info");
    }
    job(jobId, {{"state", "done"}, {"preset", kind}});
    return;
  }
  const auto prop   = figureProp(target.empty() ? Json::object() : Json{{"prop", target}});
  const auto figure = prop ? prop->figure : nullptr;
  if(kind == "pose preset" || kind == "shape preset" || kind == "property preset")
  {
    if(!figure || !figure->fig->pose)
      notice(name + " is a DAZ " + kind + ". Select a DAZ figure first, then open it.");
    else
      // a step of the history: the figure's look before and after
      journaled("preset", {{"prop", prop->id}, {"name", name}}, [&] {
        fv::PosePresetResult r = figure->fig->applyPosePreset(doc);
        protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
        // a property preset can also set surfaces ("#materials/<surface>:?<channel>" urls)
        const fv::dson::MaterialPresetResult surfaces = fv::dson::applyMaterialPreset(doc, figure->fig->loaded.scene, nullptr, definitionsOf(*figure->fig, &doc));
        if(!surfaces.changed.empty())
        {
          if(prop->materials)
            prop->materials->surfacesChanged(surfaces.changed);
          auto& presets = figure->materialPresets;
          presets.erase(std::remove(presets.begin(), presets.end(), rel), presets.end());
          presets.push_back(rel);
          presetChangedEyes(*prop, rel, surfaces.changed);
          r.applied += int(surfaces.changed.size());
        }
        // it showed or hid worn items (a geograft's Apply and Remove presets)
        if(!r.itemsShown.empty())
          requestCompose("Showing", {}, figure.get());
        if(!r.applied)
          notice(name + ": nothing in this " + kind + " matches " + prop->name + ".");
        else if(!r.skipped.empty())
          notice("Applied " + name + ". " + std::to_string(r.skipped.size()) + " of its controls aren't on this figure and were skipped.", "info");
        else
          notice("Applied " + name + " to " + prop->name, "info");
        return Json();
      });
  }
  else if(kind == "material preset" || kind == "layered image preset")
  {
    if(!figure)
      notice(name + " is a DAZ " + kind + ". Select a DAZ figure first, then open the preset.");
    else if(figure->addonsPending)
    {
      // Genesis 9's eyes, mouth and lashes are add-ons still going on: the preset waits for their surfaces
      figure->afterAddons.push_back([this, doc, kind, jobId, rel, id = prop->id] { applyPreset(doc, kind, jobId, rel, std::nullopt, {}, id); });
      return;
    }
    else if(fv::dson::isEyePreset(doc, figure->fig->loaded.scene))
    {
      // eyes swap cleanly, as the Material tab's chooser does
      try
      {
        journaled("preset", {{"prop", prop->id}, {"name", name}}, [&] { return applyEyes(*prop, rel, &doc, false); });
        notice("Eyes " + name + " on " + prop->name + ".", "info");
      }
      catch(const std::exception& e)
      {
        notice(e.what());
      }
    }
    else
      journaled("preset", {{"prop", prop->id}, {"name", name}}, [&] {
        const fv::dson::MaterialPresetResult r = fv::dson::applyMaterialPreset(doc, figure->fig->loaded.scene, nullptr, definitionsOf(*figure->fig, &doc));
        // not composed yet, the composition lowers the surfaces as they are now
        if(prop->materials)
          prop->materials->surfacesChanged(r.changed);
        if(r.changed.empty())
          notice(name + ": none of its surfaces match " + prop->name + ".");
        else
        {
          auto& presets = figure->materialPresets;
          presets.erase(std::remove(presets.begin(), presets.end(), rel), presets.end());
          presets.push_back(rel);
          presetChangedEyes(*prop, rel, r.changed);
          notice("Applied material preset " + name + " to " + std::to_string(r.changed.size()) + " surfaces.", "info");
        }
        return Json();
      });
  }
  else
    notice(name + " is a DAZ " + kind + " and isn't supported in the native viewport yet.");
  job(jobId, {{"state", "done"}, {"preset", kind}});
}

// The eyes `rel` on a figure, `doc` its preset (null, with `rel` "", for the character's own): the eye
// surfaces back to the character's own, then the preset's eye part over them, so nothing of the eyes
// before stays. Unless `keepEdits`, the panel's edits of the eye surfaces go too (an iris colour sets
// the iris's mean, so kept it would make every pair look the same). Throws, changing nothing, when
// the figure has no eyes or nothing of the preset lands on them.
Json FvHost::Impl::applyEyes(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&             f     = *prop.figure;
  fv::SceneGraph&     scene = f.fig->loaded.scene;
  const std::set<int> eyes  = fv::dson::eyeSurfaces(scene);
  if(eyes.empty())
    throw std::runtime_error(prop.name + " has no eyes to change.");
  if(doc)
  {
    const std::set<int> targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return eyes.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + prop.name + "'s eyes.");
  }
  const fv::dson::EyesResult r       = fv::dson::chooseEyes(scene, doc, f.eyesOwned, definitionsOf(*f.fig, doc));
  std::set<int>              changed = r.changed;
  if(!keepEdits)
  {
    std::set<int> cleared = eyes;
    cleared.insert(f.eyesOwned.begin(), f.eyesOwned.end());
    for(int i : cleared)
      if(i >= 0 && i < int(scene.materials.size()) && !scene.materials[size_t(i)].edits.empty())
      {
        scene.materials[size_t(i)].edits = {};
        changed.insert(i);
      }
  }
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  // the scene file replays it as its eyes, not as a preset
  std::set<int> replaced = changed;
  replaced.insert(f.eyesOwned.begin(), f.eyesOwned.end());
  replaced.insert(r.owned.begin(), r.owned.end());
  choiceReplaced(f, rel, replaced);
  f.eyes      = rel;
  f.eyesOwned = r.owned;
  f.eyesPreset.clear();
  publishEyes(prop);
  return {{"eyes", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

namespace {

// The surfaces a worn item's meshes use.
std::set<int> itemSurfaces(const fv::SceneGraph& scene, const fv::WornItem& w)
{
  std::set<int> out;
  for(int m : w.meshes)
    if(m >= 0 && m < int(scene.meshes.size()))
      for(int mi : scene.meshes[size_t(m)].materials)
        if(mi >= 0 && mi < int(scene.materials.size()))
          out.insert(mi);
  return out;
}

}  // namespace

// The colour `rel` on worn items, `doc` its preset (null, with `rel` "", for the items as they were
// worn): the items' surfaces back to how they were worn, then the preset's part on them, so nothing
// of the colour before stays and nothing else it sets is touched. Several items go as one (a hair
// worn as a cap, a base and its curls), so a preset that colours only some of them still applies.
// Unless `keepEdits`, the panel's edits of those surfaces go too (a recolour sets the mean, so kept
// it would make every colour look the same). Throws, changing nothing, when nothing of the preset
// lands on the items.
Json FvHost::Impl::applyItemColor(Prop& prop, const std::vector<int>& items, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&                          f     = *prop.figure;
  fv::SceneGraph&                  scene = f.fig->loaded.scene;
  std::vector<const fv::WornItem*> worn;
  std::set<int>                    surfaces, previous;
  for(int item : items)
  {
    const fv::WornItem* w = wornItem(*f.fig, item);
    if(!w)
      throw std::runtime_error(prop.name + " doesn't wear that item any more.");
    worn.push_back(w);
    const std::set<int> own = itemSurfaces(scene, *w);
    surfaces.insert(own.begin(), own.end());
    if(const auto was = f.colors.find(item); was != f.colors.end())
      previous.insert(was->second.owned.begin(), was->second.owned.end());
  }
  const std::string label = worn.empty() ? std::string("The item") : worn.front()->label;
  if(surfaces.empty())
    throw std::runtime_error(label + " has no surfaces to colour.");
  std::set<int> targets;
  if(doc)
  {
    targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return surfaces.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + label + "'s.");
  }
  const fv::dson::PresetChoice r       = fv::dson::choosePreset(scene, surfaces, doc, previous, definitionsOf(*f.fig, doc));
  std::set<int>                changed = r.changed;
  // the preset's colours replace the edits, all but the tiling, the fabric and the sheen (no colours)
  if(!keepEdits)
    for(int i : surfaces)
      if(fv::mat::Edits& was = scene.materials[size_t(i)].edits; !was.empty())
      {
        fv::mat::Edits kept;
        kept.uvScaleU       = was.uvScaleU;
        kept.uvScaleV       = was.uvScaleV;
        kept.uvOffsetU      = was.uvOffsetU;
        kept.uvOffsetV      = was.uvOffsetV;
        kept.uvRotation     = was.uvRotation;
        kept.fabric         = was.fabric;
        kept.fabricScale    = was.fabricScale;
        kept.fabricStrength = was.fabricStrength;
        kept.sheenColor     = was.sheenColor;
        kept.sheenRoughness = was.sheenRoughness;
        was                 = kept;
        changed.insert(i);
      }
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  // each item owns what the choice changed on its own surfaces
  for(const fv::WornItem* w : worn)
  {
    const std::set<int> own = itemSurfaces(scene, *w);
    std::set<int>       owned;
    std::set_intersection(own.begin(), own.end(), r.owned.begin(), r.owned.end(), std::inserter(owned, owned.end()));
    f.colors[w->id] = {rel, owned};
  }
  // the scene file replays it as the items' colour, not as a preset; one that also set other
  // surfaces (a character's skin and hair) stays a preset for those
  if(std::includes(surfaces.begin(), surfaces.end(), targets.begin(), targets.end()))
    f.materialPresets.erase(std::remove(f.materialPresets.begin(), f.materialPresets.end(), rel), f.materialPresets.end());
  publishWardrobe(f);
  return {{"color", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

// wardrobe.color {prop, item | [item...], rel}: the preset is read on a worker and applied on the
// application thread. The job fails, changing nothing, for an item the figure no longer wears or a
// preset none of whose surfaces are the items'.
Json FvHost::Impl::wardrobeColor(const Json& args)
{
  if(!args.contains("rel") || !(args["rel"].is_string() || args["rel"].is_null()))
    throw std::runtime_error("wardrobe.color wants rel: a library path, or null for the item as it was worn");
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const std::vector<int> items = itemIds(args);
  if(items.empty())
    throw std::runtime_error("wardrobe.color wants an item");
  for(int item : items)
    if(!wornItem(*figure->fig, item))
      throw std::runtime_error("no such item");
  const fv::WornItem* w     = wornItem(*figure->fig, items.front());
  const std::string   rel   = args["rel"].is_string() ? args["rel"].get<std::string>() : std::string();
  const uint64_t      jobId = ++jobs;
  const std::string   label = w->label + ": " + (rel.empty() ? std::string("its own colour") : fileStem(rel));
  job(jobId, {{"label", label}, {"state", "running"}});
  const auto fail  = [this, jobId, label](const std::string& why) { job(jobId, {{"label", label}, {"state", "failed"}, {"error", why}}); };
  const auto apply = [this, figure, items, rel, jobId, label, fail](const std::optional<fv::dson::Json>& doc) {
    const auto p = findProp(figure->id);
    if(!p || p->figure != figure)
      return fail("the figure left the scene");
    try
    {
      Json done     = journaled("wardrobe.color", {{"prop", p->id}, {"label", label}}, [&] { return applyItemColor(*p, items, rel, doc ? &*doc : nullptr, false); });
      done["label"] = label;
      done["state"] = "done";
      job(jobId, done);
    }
    catch(const std::exception& e)
    {
      fail(e.what());
    }
  };
  if(rel.empty())
  {
    apply(std::nullopt);
    return {{"job", jobId}};
  }
  loaders.spawn([this, figure, rel, apply, fail] {
    try
    {
      auto doc = figure->fig->store().load(rel);
      if(!doc)
        throw std::runtime_error(fileStem(rel) + " isn't in the content library.");
      const auto preset = fv::dson::describePreset(doc->json);
      if(!preset || !preset->setsMaterials)
        throw std::runtime_error(fileStem(rel) + " is not a material preset.");
      post([apply, body = doc->json] { apply(body); });
    }
    catch(const std::exception& e)
    {
      fail(e.what());
    }
  });
  return {{"job", jobId}};
}

void FvHost::Impl::presetChangedEyes(Prop& prop, const std::string& rel, const std::set<int>& changed)
{
  Loaded& f = *prop.figure;
  // a worn item the preset reached has it for its colour now: undo and a scene file put that back
  bool colors = false;
  for(const fv::WornItem& w : f.fig->items)
  {
    const std::set<int> surfaces = itemSurfaces(f.fig->loaded.scene, w);
    std::set<int>       owned;
    std::set_intersection(surfaces.begin(), surfaces.end(), changed.begin(), changed.end(), std::inserter(owned, owned.end()));
    if(owned.empty())
      continue;
    f.colors[w.id] = {rel, owned};
    colors         = true;
  }
  if(colors)
    publishWardrobe(f);
  // Of more than half of the skin or the eyes, the preset is them now. Of less (a lash colour, a nail
  // polish, a tattoo), it lies over what was chosen, which stays chosen, and goes back after it.
  const auto reach = [&](const std::set<int>& part) {
    const auto touched = std::count_if(part.begin(), part.end(), [&](int i) { return changed.count(i) > 0; });
    return touched == 0 ? 0 : touched * 2 > std::ptrdiff_t(part.size()) ? 2 : 1;
  };
  const int skin = reach(fv::dson::skinSurfaces(f.fig->loaded.scene, wornSurfaces(f)));
  const int eyes = reach(fv::dson::eyeSurfaces(f.fig->loaded.scene));
  if(skin == 2)
  {
    f.skin.reset();
    f.skinOwned.clear();
    f.skinPreset = rel;
    publishSkin(prop);
  }
  if(eyes == 2)
  {
    f.eyes.reset();
    f.eyesOwned.clear();
    f.eyesPreset = rel;
    publishEyes(prop);
  }
  std::erase_if(f.presetsOver, [&](const Loaded::PresetOver& o) { return o.rel == rel; });
  if((skin == 1 && f.skin) || (eyes == 1 && f.eyes))
  {
    std::erase(f.materialPresets, rel);
    f.presetsOver.push_back({rel, changed});
  }
}

void FvHost::Impl::choiceReplaced(Loaded& f, const std::string& rel, const std::set<int>& surfaces)
{
  std::erase(f.materialPresets, rel);
  std::vector<Loaded::PresetOver> kept;
  for(Loaded::PresetOver& o : f.presetsOver)
    if(o.rel == rel)
      continue;
    else if(std::any_of(o.surfaces.begin(), o.surfaces.end(), [&](int i) { return surfaces.count(i) > 0; }))
      f.materialPresets.push_back(o.rel);
    else
      kept.push_back(std::move(o));
  f.presetsOver = std::move(kept);
}

// "materials/<prop>".eyes: {rel, preset, own, ready, surfaces} (fv_host.hpp material.eyes)
void FvHost::Impl::publishEyes(Prop& p)
{
  if(!p.figure || !p.materials)
    return;
  const Loaded& f   = *p.figure;
  Json          own = Json::array();
  for(const fv::dson::PostLoadAddon& a : f.fig->addons)
    for(const std::string& rel : a.presets)
      if(std::find(own.begin(), own.end(), Json(rel)) == own.end())
        own.push_back(rel);
  Json surfaces = Json::array();
  for(int i : fv::dson::eyeSurfaces(f.fig->loaded.scene))
    surfaces.push_back("m:" + std::to_string(i));
  p.materials->setEyes({{"rel", f.eyes && !f.eyes->empty() ? Json(*f.eyes) : Json(nullptr)},
                        {"preset", f.eyesPreset.empty() ? Json(nullptr) : Json(f.eyesPreset)},
                        {"own", own},
                        {"ready", !f.addonsPending},
                        {"surfaces", surfaces}});
}

// A preset chosen for part of a figure (material.eyes, material.skin, material.makeup) {prop, rel}:
// read on a worker, and applied by `apply` on the application thread once the figure's add-ons
// (Genesis 9's eyes and mouth) are on. The job fails, changing nothing, for a prop that is not a
// figure, or when `apply` throws.
Json FvHost::Impl::choiceJob(const Json& args, const std::string& command, const std::string& part, ChoiceApply apply)
{
  if(!args.contains("rel") || !(args["rel"].is_string() || args["rel"].is_null()))
    throw std::runtime_error(command + " wants rel: a library path, or null for the character's own");
  const std::string rel   = args["rel"].is_string() ? args["rel"].get<std::string>() : std::string();
  const uint64_t    jobId = ++jobs;
  const std::string label = part + ": " + (rel.empty() ? std::string("the character's own") : fileStem(rel));
  job(jobId, {{"label", label}, {"state", "running"}});
  const auto fail = [this, jobId, label](const std::string& why) { job(jobId, {{"label", label}, {"state", "failed"}, {"error", why}}); };
  const auto prop = figureProp(args);
  if(!prop)
  {
    fail("Select a figure first.");
    return {{"job", jobId}};
  }
  std::shared_ptr<Loaded> target = prop->figure;
  // on the application thread, the add-ons on
  const auto done = [this, target, rel, jobId, label, fail, apply, command](const std::optional<fv::dson::Json>& doc) {
    const auto p = findProp(target->id);
    if(!p || p->figure != target)
      return fail("the figure left the scene");
    try
    {
      // a step of the history: the look before and after
      Json out     = journaled(command, {{"prop", p->id}, {"label", label}}, [&] { return apply(*p, rel, doc ? &*doc : nullptr); });
      out["label"] = label;
      out["state"] = "done";
      job(jobId, out);
    }
    catch(const std::exception& e)
    {
      fail(e.what());
    }
  };
  auto choose = [this, target, rel, done, fail] {
    if(rel.empty())
      return done(std::nullopt);
    loaders.spawn([this, target, rel, done, fail] {
      try
      {
        auto doc = target->fig->store().load(rel);
        if(!doc)
          throw std::runtime_error(fileStem(rel) + " isn't in the content library.");
        const auto preset = fv::dson::describePreset(doc->json);
        if(!preset || !preset->setsMaterials)
          throw std::runtime_error(fileStem(rel) + " is not a material preset.");
        post([done, body = doc->json] { done(body); });
      }
      catch(const std::exception& e)
      {
        fail(e.what());
      }
    });
  };
  if(target->addonsPending)
    target->afterAddons.push_back(std::move(choose));
  else
    choose();
  return {{"job", jobId}};
}

// material.eyes {prop, rel}: fails for a figure with no eyes, or a preset none of whose surfaces are its eyes.
Json FvHost::Impl::materialEyes(const Json& args)
{
  return choiceJob(args, "material.eyes", "Eyes", [this](Prop& p, const std::string& rel, const fv::dson::Json* doc) { return applyEyes(p, rel, doc, false); });
}

// material.skin {prop, rel}: fails for a preset none of whose surfaces are the character's skin.
Json FvHost::Impl::materialSkin(const Json& args)
{
  return choiceJob(args, "material.skin", "Skin", [this](Prop& p, const std::string& rel, const fv::dson::Json* doc) { return applySkin(p, rel, doc, false); });
}

// material.makeup {prop, rel}: as material.skin.
Json FvHost::Impl::materialMakeup(const Json& args)
{
  return choiceJob(args, "material.makeup", "Makeup", [this](Prop& p, const std::string& rel, const fv::dson::Json* doc) { return applyMakeup(p, rel, doc, false); });
}

namespace {

// A colour the panel gave a surface: what a skin or makeup choice takes away, as a kept tone would
// make every skin the same. Roughness and the like stay. True when there was one.
bool clearColours(fv::mat::Edits& e, bool makeupOnly)
{
  const bool had = e.makeupColor || e.makeupWeight || (!makeupOnly && (e.color || e.tone || e.toneStrength || e.recolor || e.recolorStrength));
  e.makeupColor.reset();
  e.makeupWeight.reset();
  if(!makeupOnly)
  {
    e.color.reset();
    e.tone.reset();
    e.toneStrength.reset();
    e.recolor.reset();
    e.recolorStrength.reset();
  }
  return had;
}

}  // namespace

// The surfaces of a figure's worn clothing, props and hair, which its skin leaves alone; a hair's
// brows and lashes are the character's.
std::set<int> FvHost::Impl::wornSurfaces(const Loaded& f)
{
  const fv::SceneGraph& scene = f.fig->loaded.scene;
  std::set<int>         out;
  for(const fv::WornItem& w : f.fig->items)
  {
    if(w.kind == "Anatomy")
      continue;
    for(int i : itemSurfaces(scene, w))
    {
      const fv::Material& m = scene.materials[size_t(i)];
      std::string         name = m.name;
      std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
      if(w.kind == "Hair" && (m.surface.kind == fv::mat::Kind::eLashes || name.find("brow") != std::string::npos))
        continue;
      out.insert(i);
    }
  }
  return out;
}

// The skin `rel` on a figure, `doc` its preset (null, with `rel` "", for the character's own): the
// skin surfaces back to the character's own, then the preset's part on them, so nothing of the skin
// before stays; the makeup laid on it goes with it. Unless `keepEdits`, the panel's colours on them
// go too (a tone sets the skin's mean). Throws, changing nothing, when nothing of the preset lands
// on the skin.
Json FvHost::Impl::applySkin(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&             f        = *prop.figure;
  fv::SceneGraph&     scene    = f.fig->loaded.scene;
  const std::set<int> surfaces = fv::dson::skinSurfaces(scene, wornSurfaces(f));
  if(surfaces.empty())
    throw std::runtime_error(prop.name + " has no skin to change.");
  if(doc)
  {
    const std::set<int> targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return surfaces.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + prop.name + "'s skin.");
  }
  const fv::dson::PresetChoice r       = fv::dson::choosePreset(scene, surfaces, doc, f.skinOwned, definitionsOf(*f.fig, doc));
  std::set<int>                changed = r.changed;
  if(!keepEdits)
  {
    std::set<int> cleared = surfaces;
    cleared.insert(f.skinOwned.begin(), f.skinOwned.end());
    for(int i : cleared)
      if(i >= 0 && i < int(scene.materials.size()) && clearColours(scene.materials[size_t(i)].edits, false))
        changed.insert(i);
  }
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  std::set<int> replaced = changed;
  replaced.insert(f.skinOwned.begin(), f.skinOwned.end());
  replaced.insert(r.owned.begin(), r.owned.end());
  choiceReplaced(f, rel, replaced);
  f.skin      = rel;
  f.skinOwned = r.owned;
  f.skinPreset.clear();
  // back to the baseline, the makeup is the skin's own
  f.makeup.reset();
  f.makeupLayer.clear();
  publishSkin(prop);
  return {{"skin", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

// The makeup `rel` over the skin on now (null `doc`, `rel` "": the skin's own makeup): the channels
// the last makeup set go back, then the preset's part on the skin goes over. Unless `keepEdits`, the
// panel's makeup colours and weights go too. Throws, changing nothing, when nothing of the preset
// lands on the skin.
Json FvHost::Impl::applyMakeup(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&             f        = *prop.figure;
  fv::SceneGraph&     scene    = f.fig->loaded.scene;
  const std::set<int> surfaces = fv::dson::skinSurfaces(scene, wornSurfaces(f));
  if(doc)
  {
    const std::set<int> targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return surfaces.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + prop.name + "'s skin.");
  }
  const fv::dson::LayerChoice r       = fv::dson::chooseLayer(scene, surfaces, doc, f.makeupLayer, definitionsOf(*f.fig, doc));
  std::set<int>               changed = r.changed;
  if(!keepEdits)
    for(int i : surfaces)
      if(clearColours(scene.materials[size_t(i)].edits, true))
        changed.insert(i);
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  f.makeup      = rel;
  f.makeupLayer = r.layer;
  f.materialPresets.erase(std::remove(f.materialPresets.begin(), f.materialPresets.end(), rel), f.materialPresets.end());
  publishSkin(prop);
  return {{"makeup", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

// "materials/<prop>".skin: {rel, preset, makeup, ready, surfaces} (fv_host.hpp material.skin)
void FvHost::Impl::publishSkin(Prop& p)
{
  if(!p.figure || !p.materials)
    return;
  const Loaded& f        = *p.figure;
  Json          surfaces = Json::array();
  for(int i : fv::dson::skinSurfaces(f.fig->loaded.scene, wornSurfaces(f)))
    surfaces.push_back("m:" + std::to_string(i));
  p.materials->setSkin({{"rel", f.skin && !f.skin->empty() ? Json(*f.skin) : Json(nullptr)},
                        {"preset", f.skinPreset.empty() ? Json(nullptr) : Json(f.skinPreset)},
                        {"makeup", f.makeup && !f.makeup->empty() ? Json(*f.makeup) : Json(nullptr)},
                        {"ready", !f.addonsPending},
                        {"surfaces", surfaces}});
}
