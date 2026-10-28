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
// FvHost: imported models, clips, GLB export, and the scene file (scene.state, scene.restore).
//

#include "fv_host_impl.hpp"

//--------------------------------------------------------------------------------------------------
// Imported models and animation

Json FvHost::Impl::importModel(const std::string& path, std::function<void(std::shared_ptr<Prop>)> then)
{
  const uint64_t    jobId      = ++jobs;
  const uint64_t    generation = loadGeneration;
  const std::string label      = "Loading " + utf8Of(pathOf(path).filename());
  job(jobId, {{"label", label}, {"state", "running"}});
  loaders.spawn([this, path, jobId, label, generation, then = std::move(then)] {
    auto        model = std::make_shared<tinygltf::Model>();
    std::string error;
    if(!fv::composer::readModelFile(pathOf(path), *model, error))
    {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", error}});
      notice("Could not open " + utf8Of(pathOf(path).filename()) + ": " + error, "error");
      return;
    }
    post([this, model, path, jobId, label, generation, then] {
      if(generation != loadGeneration)
      {
        job(jobId, {{"label", label}, {"state", "done"}});
        return;
      }
      std::string ext = pathOf(path).extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::toupper(c)); });
      auto p    = addProp(PropKind::eModel, utf8Of(pathOf(path).stem()));
      p->model  = model;
      p->path   = path;
      p->format = ext == ".OBJ" ? "OBJ" : "glTF";
      requestCompose("Adding " + p->name, [this, p, jobId, label, then] {
        if(!findProp(p->id))
        {
          job(jobId, {{"label", label}, {"state", "done"}});
          return;
        }
        job(jobId, {{"label", label}, {"state", "done"}, {"prop", p->id}, {"figure", p->id}});
        selectProp(p->id);
        if(then)
          then(p);
      });
    });
  });
  return {{"job", jobId}};
}

int FvHost::Impl::clipNumber(const std::string& prop, int index)
{
  const auto [it, added] = clipNumbers.try_emplace({prop, index}, nextClip);
  if(added)
    nextClip++;
  return it->second;
}

rs::ClipId FvHost::Impl::clipOf(int number) const
{
  for(const auto& [key, n] : clipNumbers)
    if(n == number)
      if(const auto p = findProp(key.first); p && key.second >= 0 && key.second < int(p->imported.clips.size()))
        return p->imported.clips[size_t(key.second)];
  return {};
}

std::vector<int> FvHost::Impl::clipOrder() const
{
  std::vector<int> out;
  for(const auto& p : props)
    for(int i = 0; i < int(p->imported.clips.size()); i++)
      if(!p->imported.clips[size_t(i)].isNone())
        if(const auto it = clipNumbers.find({p->id, i}); it != clipNumbers.end())
          out.push_back(it->second);
  return out;
}

// The clips in the scene by their numbers, which stay theirs while their prop is here (D4), each with
// the prop it came with, in display order; and where playback is (`current` null without clips).
Json FvHost::Impl::animationTopic()
{
  Json                    clips = Json::object();
  const AnimationControl& ctrl  = renderer->editAnimationControl();
  Json                    out   = {{"clips", clips}, {"current", nullptr},  {"playing", false},
                                   {"time", 0.0},    {"speed", ctrl.speed}, {"loop", ctrl.loop}};
  if(!renderer->feedsHostScene())
    return out;
  const rs::RetainedScene& scene = renderer->hostScene();
  int                      order = 0;
  for(const auto& p : props)
    for(int i = 0; i < int(p->imported.clips.size()); i++)
    {
      const rs::ClipRecord* clip = scene.clip(p->imported.clips[size_t(i)]);
      if(!clip)
        continue;
      const std::array<float, 2> range  = clip->range();
      const int                  number = clipNumber(p->id, i);
      clips[std::to_string(number)] = {{"name", clip->name.empty() ? "Clip " + std::to_string(order + 1) : clip->name},
                                       {"duration", std::max(0.f, range[1] - range[0])},
                                       {"order", order++},
                                       {"prop", p->id}};
    }
  out["clips"] = clips;
  if(currentClip >= 0 && clips.contains(std::to_string(currentClip)))
  {
    out["current"] = currentClip;
    out["playing"] = ctrl.play;
    out["time"]    = std::round(renderer->hostClipTime() * 1000.f) / 1000.0;
  }
  return out;
}

void FvHost::Impl::publishAnimation(bool force)
{
  const auto now = std::chrono::steady_clock::now();
  if(!force && now - animationAt < 100ms)
    return;
  animationAt = now;
  Json topic  = animationTopic();
  if(!force && topic == lastAnimation)
    return;
  lastAnimation = topic;
  protocol->events().snapshot("animation", std::move(topic));
}

Json FvHost::Impl::animationCommand(const std::string& cmd, const Json& args)
{
  AnimationControl&      ctrl  = renderer->editAnimationControl();
  const std::vector<int> order = clipOrder();
  if(order.empty())
    throw std::runtime_error("nothing in the scene has animations");
  // the clip a command acts on: the current one, else the first
  const auto current = [&] {
    if(clipOf(currentClip).isNone())
    {
      currentClip = order.front();
      renderer->setHostClip(clipOf(currentClip));
    }
  };
  if(cmd == "select")
  {
    const int clip = args.at("clip").get<int>();
    if(clipOf(clip).isNone())
      throw std::runtime_error("no such clip");
    currentClip = clip;
    renderer->setHostClip(clipOf(clip));
    ctrl.reset   = true;
    ctrl.play    = args.value("play", true);
    ctrl.runOnce = !ctrl.play;
  }
  else if(cmd == "play")
  {
    current();
    // playing from the end starts over
    if(renderer->hostClipTime() >= renderer->hostClipDuration() - 1e-3f && !ctrl.loop)
      ctrl.reset = true;
    ctrl.play = true;
  }
  else if(cmd == "pause")
    ctrl.play = false;
  else if(cmd == "stop")
  {
    ctrl.play  = false;
    ctrl.reset = true;
  }
  else if(cmd == "seek")
  {
    current();
    renderer->seekHostClip(args.at("time").get<float>());
  }
  else if(cmd == "speed")
    ctrl.speed = std::clamp(args.at("speed").get<float>(), 0.f, 100.f);
  else if(cmd == "loop")
    ctrl.loop = args.at("loop").get<bool>();
  publishAnimation(true);
  return {{"ok", true}};
}

// A prop as a GLB for other tools, as it stands. A figure: pose, dials, worn items and material
// edits; the glTF is built here (the scene belongs to this thread), composing the textures, which
// takes the time, and writing the file run on a worker. A model: its file's contents with its
// material edits. {scene: true}: everything in the scene, as the renderer has it, but the sun.
// The file's model with its material edits laid over its own materials, as FvMaterials::writeModel
// lays them over the scene's: its textures stay the file's own.
tinygltf::Model FvHost::Impl::editedModel(const Prop& p) const
{
  tinygltf::Model copy = *p.model;
  for(size_t i = 0; p.materials && i < copy.materials.size(); i++)
    if(const fv::mat::Edits* edits = p.materials->modelEdits(int(i)))
      copy.materials[i] = fv::model::editedMaterial(copy.materials[i], *edits);
  for(const char* ext : fv::model::kEditedExtensions)
    if(std::any_of(copy.materials.begin(), copy.materials.end(), [&](const tinygltf::Material& m) { return m.extensions.count(ext) > 0; })
       && std::find(copy.extensionsUsed.begin(), copy.extensionsUsed.end(), ext) == copy.extensionsUsed.end())
      copy.extensionsUsed.push_back(ext);
  return copy;
}

// The scene as a file (D21: what the renderer draws, FV extensions and all), built fresh from the
// props rather than copied from the renderer's scene: figures built again with the live options (so
// their dialled vertices and binds agree, and what was taken off is not there, while a hidden item
// stays, marked hidden); models with their material edits; cameras at their current lens; lights
// at their own intensity, a hidden one's root marked hidden; no sun of the host's; the sky. What the
// composer can't carry over is stripped, with a notice. The figures are built here, on the app
// thread: a worker would race the dials, which change their meshes in place on this thread.
Json FvHost::Impl::exportScene(const std::string& path)
{
  if(props.empty())
    throw std::runtime_error("The scene is empty: there is nothing to export.");
  fv::timing::Scope                                  timeExport("export scene");
  std::vector<std::optional<fv::build::BuildResult>> builds(props.size());
  std::vector<std::optional<tinygltf::Model>>        models(props.size());
  std::vector<fv::composer::Part>                    parts(props.size());
  for(size_t i = 0; i < props.size(); i++)
  {
    const Prop&              p     = *props[i];
    const tinygltf::Model*   model = nullptr;
    if(Loaded* f = p.figure.get())
    {
      builds[i] = fv::build::buildGltf(f->fig->loaded.scene, f->fig->loaded.root, buildOptionsOf(*f, skinMode.load()));
      model     = &builds[i]->model;
    }
    else if(p.model)
      model = &models[i].emplace(editedModel(p));
    parts[i] = partOf(i, model, true);
  }
  std::vector<fv::composer::Placed> placed;
  tinygltf::Model                   scene = fv::composer::compose(parts, placed);
  for(const std::string& stripped : fv::composer::stripForFile(scene))
    notice("Export: " + stripped, "warning");
  renderer->writeEnvironmentSky(scene, pathOf(path));
  // tinygltf writes extensionsUsed as it is: what the sky added, and nothing stripped
  tinygltf::utils::syncExtensionsUsed(scene);
  std::string error;
  if(!fv::composer::packForGlb(scene, error))
    throw std::runtime_error(error);
  tinygltf::TinyGLTF writer;
  if(!writer.WriteGltfSceneToFile(&scene, utf8Of(pathOf(path)), false, true, false, true))
    throw std::runtime_error("could not write " + path);
  notice("Exported " + utf8Of(pathOf(path).filename()), "info");
  return {{"ok", true}};
}

Json FvHost::Impl::exportGlb(const Json& args)
{
  const std::string path = args.at("path").get<std::string>();
  if(args.value("scene", false))
    return exportScene(path);
  const auto prop = propArg(args);
  if(!prop || (!prop->figure && !prop->model))
    throw std::runtime_error("Select a figure or a model to export.");
  if(prop->model)
  {
    tinygltf::Model copy = editedModel(*prop);
    std::string     error;
    if(!fv::composer::packForGlb(copy, error))
      throw std::runtime_error(error);
    tinygltf::TinyGLTF writer;
    if(!writer.WriteGltfSceneToFile(&copy, utf8Of(pathOf(path)), false, true, false, true))
      throw std::runtime_error("could not write " + path);
    notice("Exported " + utf8Of(pathOf(path).filename()), "info");
    return {{"ok", true}};
  }
  const auto&                        figure = prop->figure;
  std::shared_ptr<fv::dson::Library> lib;
  {
    std::lock_guard lock(settingsMutex);
    lib = figure->lib;
  }
  auto              prepared = std::make_shared<fv::build::GlbExport>(fv::build::prepareGlb(figure->fig->loaded.scene, figure->fig->loaded.root));
  const uint64_t    jobId    = ++jobs;
  const std::string label    = "Exporting " + utf8Of(pathOf(path).filename());
  job(jobId, {{"label", label}, {"state", "running"}});
  loaders.spawn([this, prepared, lib, path, jobId, label] {
    try
    {
      fv::build::ExportOptions options;
      options.find = [lib](const std::string& rel) -> std::optional<std::filesystem::path> {
        if(!lib)
          return std::nullopt;
        std::string r = fv::dson::decodeUri(rel);
        std::replace(r.begin(), r.end(), '\\', '/');
        const size_t at = r.find_first_not_of('/');
        return lib->find(at == std::string::npos ? std::string() : r.substr(at));
      };
      const auto notes = fv::build::finishGlb(*prepared, pathOf(path), options);
      for(size_t i = 0; i < notes.size() && i < 5; i++)
        notice("Export: " + notes[i]);
      if(notes.size() > 5)
        notice("Export: " + std::to_string(notes.size() - 5) + " more textures could not be made");
      job(jobId, {{"label", label}, {"state", "done"}});
      notice("Exported " + utf8Of(pathOf(path).filename()) + (notes.empty() ? "" : " with " + std::to_string(notes.size()) + " textures replaced by neutral ones"), "info");
    }
    catch(const std::exception& e)
    {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", e.what()}});
      notice(std::string("Export failed: ") + e.what(), "error");
    }
  });
  return {{"job", jobId}};
}

//--------------------------------------------------------------------------------------------------
// Scene state

// What the host owns of a prop in a scene file: how a figure was opened, what it wears, its dials,
// its pose where it differs from how it loaded, and the edits of its materials; for a model, its
// file and material edits. The app adds the prop's place, cameras, lights and lighting from the
// scene topic and its own settings.
Json FvHost::Impl::sceneState(const Json& args)
{
  const auto prop = propArg(args);
  if(!prop)
    throw std::runtime_error("no such prop");
  if(prop->model)
    return {{"source", {{"path", utf8Of(pathOf(prop->path))}}},
            {"worn", Json::array()},
            {"morphs", Json::object()},
            {"pose", Json::array()},
            {"materials", prop->materials ? prop->materials->editState() : Json::object()}};
  if(!prop->figure)
    return Json::object();
  fv::Figure& fig  = *prop->figure->fig;
  Json        worn = Json::array();
  for(const fv::WornItem& w : fig.items)
    if(!w.rel.empty())
    {
      Json entry = {{"rel", w.rel}, {"visible", w.visible}, {"label", w.label}};
      // dForce on the item as the user set it, when they did
      if(w.cloth.on || w.cloth.strength != 1.f)
        entry["cloth"] = {{"on", w.cloth.on ? Json(*w.cloth.on) : Json(nullptr)}, {"strength", std::round(w.cloth.strength * 1e3) / 1e3}};
      // the colour chosen for it, null for its own
      if(const auto c = prop->figure->colors.find(w.id); c != prop->figure->colors.end())
        entry["color"] = c->second.rel.empty() ? Json(nullptr) : Json(c->second.rel);
      worn.push_back(std::move(entry));
    }
  Json morphs = Json::object();
  for(const fv::morph::Channel& c : fig.morphs->channels())
    if(!c.hidden && c.value != 0)
      morphs[c.name] = std::round(c.value * 1e5) / 1e5;
  Json pose = Json::array();
  if(fig.pose)
    for(const auto& pb : fig.pose->bones())
      for(const auto& a : pb.axes)
        if(a.value != a.initial)
          pose.push_back({{"bone", pb.id}, {"kind", fv::pose::kindName(a.kind)}, {"axis", a.axis}, {"value", std::round(a.value * 1e5) / 1e5}});
  Json state = {{"source", prop->figure->source},
                {"worn", worn},
                {"morphs", morphs},
                {"pose", pose},
                {"materialPresets", prop->figure->materialPresets},
                {"materials", prop->materials ? prop->materials->editState() : Json::object()}};
  // the eyes chosen, null for the character's own; none when nothing was chosen
  if(prop->figure->eyes)
    state["eyes"] = prop->figure->eyes->empty() ? Json(nullptr) : Json(*prop->figure->eyes);
  // the same for the skin and the makeup
  if(prop->figure->skin)
    state["skin"] = prop->figure->skin->empty() ? Json(nullptr) : Json(*prop->figure->skin);
  if(prop->figure->makeup)
    state["makeup"] = prop->figure->makeup->empty() ? Json(nullptr) : Json(*prop->figure->makeup);
  if(prop->figure->collider)
    state["collision"] = prop->figure->collider->enabled();
  if(prop->figure->dforce)
  {
    state["dforce"]     = prop->figure->dforce->enabled();
    state["dforceMode"] = prop->figure->dforce->mode() == fv::dforce::Mode::eLive    ? "live" :
                          prop->figure->dforce->mode() == fv::dforce::Mode::eLiveGpu ? "gpu" :
                                                                                       "settle";
  }
  return state;
}

// Puts a scene file's state on a prop: for a figure, what it wears first (items before dials, since
// an item brings dials of its own), then dials, pose, material presets and material edits, and
// dForce last, so it drapes the finished figure. The figure's own add-ons are already going on
// with their presets; they are waited for, not worn again.
Json FvHost::Impl::sceneRestore(const Json& args)
{
  const auto prop = propArg(args);
  if(!prop)
    throw std::runtime_error("no such prop");
  // a model takes back its material edits; it has no wardrobe, dials or pose
  if(!prop->figure)
  {
    std::vector<std::string> problems;
    if(prop->materials && args.contains("materials") && args["materials"].is_object())
      problems = prop->materials->restoreEdits(args["materials"]);
    for(size_t i = 0; i < problems.size() && i < 5; i++)
      notice(problems[i]);
    return {{"ok", true}};
  }
  std::shared_ptr<Loaded> target = prop->figure;
  const uint64_t          jobId  = ++jobs;
  job(jobId, {{"label", "Restoring " + prop->name}, {"state", "running"}});
  const Json worn  = args.value("worn", Json::array());
  // the worn files start reading now: they are put on only after the figure's add-ons
  {
    std::set<std::string> started;
    for(const Json& w : worn)
      if(w.is_object() && w.contains("rel") && w["rel"].is_string() && started.insert(relKey(w["rel"].get<std::string>())).second)
        target->fig->prefetchWear(w["rel"].get<std::string>());
  }
  // What the file sets on the figure, in three steps: its dials and pose, its surfaces (presets,
  // skin, eyes, makeup, colours, edits) and dForce, and the end, which reports what could not be
  // put back. A restore that wears something does the first two before its composition, so the
  // figure is built once as the file has it (wears.looks); one that wears nothing does all three now.
  auto problemList = std::make_shared<std::vector<std::string>>();
  auto shape       = [this, target, args, problemList] {
    const auto prop = findProp(target->id);
    if(!prop || prop->figure != target)
      return;
    std::vector<std::string>& problems = *problemList;
    fv::Figure&               fig      = *target->fig;
    if(args.contains("morphs") && args["morphs"].is_object())
    {
      std::vector<std::pair<std::string, double>> dials;
      int                                         missing = 0;
      for(auto it = args["morphs"].begin(); it != args["morphs"].end(); ++it)
        if(it.value().is_number() && fig.morphs->channel(it.key()))
          dials.emplace_back(it.key(), it.value().get<double>());
        else
          missing++;
      if(!dials.empty())
        fig.morphs->setMany(dials);
      if(missing)
        problems.push_back(std::to_string(missing) + (missing == 1 ? " morph dial" : " morph dials") + " not found");
    }
    if(args.contains("pose") && args["pose"].is_array() && !args["pose"].empty())
    {
      if(!fig.pose)
        problems.push_back("the pose was skipped: this figure has no skeleton");
      else
      {
        std::vector<fv::pose::ChannelValue> values;
        for(const Json& c : args["pose"])
        {
          fv::pose::ChannelKind kind;
          if(fv::pose::parseKind(c.value("kind", ""), kind))
            values.push_back({c.value("bone", ""), kind, c.value("axis", ""), c.value("value", 0.0)});
        }
        fig.pose->setMany(values);
        protocol->events().snapshot("pose/" + target->id, poseSnapshot(*target));
      }
    }
  };
  auto surfaces = [this, target, args, problemList] {
    const auto prop = findProp(target->id);
    if(!prop || prop->figure != target)
      return;
    std::vector<std::string>& problems = *problemList;
    fv::Figure&               fig      = *target->fig;
    // each step that costs something is timed
    std::optional<fv::timing::Scope> step;
    step.emplace("restore: material presets", 5);
    if(args.contains("materialPresets") && args["materialPresets"].is_array())
      for(const Json& p : args["materialPresets"])
      {
        const std::string rel = p.is_string() ? p.get<std::string>() : "";
        if(rel.empty())
          continue;
        try
        {
          auto doc = fig.store().load(rel);
          if(!doc)
          {
            problems.push_back("material preset " + fileStem(rel) + " isn't in the content library");
            continue;
          }
          // eyes replace the eyes before them instead of stacking on them (older files list each one tried)
          if(fv::dson::isEyePreset(doc->json, fig.loaded.scene))
          {
            applyEyes(*prop, rel, &doc->json, true);
            continue;
          }
          const fv::dson::MaterialPresetResult r = fv::dson::applyMaterialPreset(doc->json, fig.loaded.scene);
          if(prop->materials)
            prop->materials->surfacesChanged(r.changed);
          if(!r.changed.empty())
          {
            target->materialPresets.push_back(rel);
            presetChangedEyes(*prop, rel, r.changed);
          }
        }
        catch(const std::exception& e)
        {
          problems.push_back("material preset " + fileStem(rel) + ": " + e.what());
        }
      }
    // the skin, eyes and makeup chosen, over the presets and under the edits (a tone, an iris colour
    // and a lipstick's are edits); the skin first, as it takes the makeup off
    const auto restoreChoice = [&](const char* key, const char* what, auto apply) {
      if(!args.contains(key) || !(args[key].is_string() || args[key].is_null()))
        return;
      const std::string rel = args[key].is_string() ? args[key].get<std::string>() : std::string();
      try
      {
        std::shared_ptr<const fv::dson::Doc> doc;
        if(!rel.empty() && !(doc = fig.store().load(rel)))
          problems.push_back(std::string(what) + " " + fileStem(rel) + " isn't in the content library");
        else
          (this->*apply)(*prop, rel, doc ? &doc->json : nullptr, true);
      }
      catch(const std::exception& e)
      {
        problems.push_back(std::string(key) + ": " + e.what());
      }
    };
    step.reset();
    step.emplace("restore: skin", 5);
    restoreChoice("skin", "the skin", &Impl::applySkin);
    step.reset();
    step.emplace("restore: eyes", 5);
    // the eyes chosen, over the presets and under the edits (an iris colour is an edit)
    if(args.contains("eyes") && (args["eyes"].is_string() || args["eyes"].is_null()))
    {
      const std::string rel = args["eyes"].is_string() ? args["eyes"].get<std::string>() : std::string();
      try
      {
        std::shared_ptr<const fv::dson::Doc> doc;
        if(!rel.empty() && !(doc = fig.store().load(rel)))
          problems.push_back("the eyes " + fileStem(rel) + " aren't in the content library");
        else
          applyEyes(*prop, rel, doc ? &doc->json : nullptr, true);
      }
      catch(const std::exception& e)
      {
        problems.push_back(std::string("eyes: ") + e.what());
      }
    }
    step.reset();
    step.emplace("restore: makeup", 5);
    restoreChoice("makeup", "the makeup", &Impl::applyMakeup);
    step.reset();
    step.emplace("restore: item colours", 5);
    // each worn item's colour, over the presets and under the edits (a hair's recolour is an edit)
    for(const Json& w : args.value("worn", Json::array()))
    {
      if(!w.is_object() || !w.contains("color") || !(w["color"].is_string() || w["color"].is_null()))
        continue;
      const std::string rel = w["color"].is_string() ? w["color"].get<std::string>() : std::string();
      for(const fv::WornItem& i : fig.items)
        if(relKey(i.rel) == relKey(w.value("rel", "")) && (w.value("label", "").empty() || i.label == w.value("label", "")))
          try
          {
            std::shared_ptr<const fv::dson::Doc> doc;
            if(!rel.empty() && !(doc = fig.store().load(rel)))
              problems.push_back("the colour " + fileStem(rel) + " of " + i.label + " isn't in the content library");
            else
              applyItemColor(*prop, {i.id}, rel, doc ? &doc->json : nullptr, true);
          }
          catch(const std::exception& e)
          {
            problems.push_back(i.label + ": " + e.what());
          }
    }
    step.reset();
    step.emplace("restore: edits", 5);
    if(prop->materials && args.contains("materials") && args["materials"].is_object())
      for(const std::string& p : prop->materials->restoreEdits(args["materials"]))
        problems.push_back(p);
    step.reset();
    step.emplace("restore: cloth", 5);
    // dForce on each item as the file set it; the items are worn by now
    bool clothChanged = false;
    for(const Json& w : args.value("worn", Json::array()))
    {
      if(!w.is_object() || !w.contains("cloth") || !w["cloth"].is_object())
        continue;
      const Json&         c = w["cloth"];
      std::optional<bool> on;
      if(c.contains("on") && c["on"].is_boolean())
        on = c["on"].get<bool>();
      for(const fv::WornItem& i : fig.items)
        if(relKey(i.rel) == relKey(w.value("rel", "")) && (w.value("label", "").empty() || i.label == w.value("label", "")))
          if(!fig.setItemCloth(i.id, on, c.value("strength", 1.f)).empty())
            clothChanged = true;
    }
    if(clothChanged)
    {
      if(target->dforce)
        target->dforce->poke();
      publishWardrobe(*target);
    }
    if(target->collider && args.contains("collision") && args["collision"].is_boolean())
      target->collider->setEnabled(args["collision"].get<bool>());
    if(target->dforce && args.contains("dforceMode") && args["dforceMode"].is_string())
    {
      // a mode this build lacks (the GPU solver) falls back to live on the CPU, and the job still ends
      try
      {
        target->dforce->setMode(dforceMode(args["dforceMode"].get<std::string>()));
      }
      catch(const std::exception& e)
      {
        target->dforce->setMode(fv::dforce::Mode::eLive);
        problems.push_back(std::string("dForce: ") + e.what() + "; simulating live on the CPU");
      }
    }
    if(target->dforce && args.contains("dforce") && args["dforce"].is_boolean())
    {
      target->dforce->setEnabled(args["dforce"].get<bool>());
      if(target->collider)
        target->collider->poke();
    }
  };
  auto finish = [this, target, jobId, problemList] {
    const auto prop = findProp(target->id);
    if(!prop || prop->figure != target)
    {
      job(jobId, {{"state", "failed"}, {"error", "the figure left the scene"}});
      return;
    }
    const std::vector<std::string>& problems = *problemList;
    for(size_t i = 0; i < problems.size() && i < 5; i++)
      notice(prop->name + ": " + problems[i]);
    if(problems.size() > 5)
      notice(std::to_string(problems.size() - 5) + " more problems restoring " + prop->name);
    job(jobId, {{"state", "done"}});
  };
  auto apply = [shape, surfaces, finish] {
    shape();
    surfaces();
    finish();
  };
  // one file can put on several items (an outfit's "Full Load"): each is listed, the file is worn once;
  // the figure's add-ons are put on by its load
  std::vector<WearRequest> requests;
  {
    std::set<std::string> skip;
    for(const fv::dson::PostLoadAddon& a : target->fig->addons)
      skip.insert(relKey(a.file));
    if(worn.is_array())
      for(const Json& w : worn)
      {
        const std::string rel = w.value("rel", "");
        if(!rel.empty() && skip.insert(relKey(rel)).second)
          requests.push_back({rel, "", {}});
      }
  }
  auto wear = [this, target, worn, apply, shape, surfaces, finish, requests](std::shared_future<void> commitAfter) mutable {
    auto hidden = [target, worn] {
      std::vector<int> ids;
      for(const Json& w : worn)
        if(!w.value("visible", true))
          for(const fv::WornItem& i : target->fig->items)
            if(relKey(i.rel) == relKey(w.value("rel", "")) && (w.value("label", "").empty() || i.label == w.value("label", "")))
            {
              target->fig->setItemVisible(i.id, false);
              ids.push_back(i.id);
            }
      return ids;
    };
    auto hide = [hidden] { hidden(); };
    if(requests.empty())
    {
      // no composition follows: the renderer's scene hides them now, as wardrobe.visible does
      bool graft = false;
      for(int id : hidden())
        if(const fv::WornItem* item = wornItem(*target->fig, id))
        {
          if(!item->hiddenPolys.empty())
            graft = true;
          else
            showItem(*target, item->objects, false);
        }
      if(graft)
        requestCompose("Showing", {}, target.get());
      publishWardrobe(*target);
      apply();
      return;
    }
    startWear(target, std::move(requests), hide, finish, std::move(commitAfter), {surfaces, shape});
  };
  if(target->addonsPending && !requests.empty())
  {
    // the worn items are loaded and bound now, beside the add-ons, and put on after them: the
    // add-ons come first, so the keys are as before and the material steps find their surfaces
    auto addonsOn = std::make_shared<std::promise<void>>();
    target->afterAddons.push_back([addonsOn] { addonsOn->set_value(); });
    wear(addonsOn->get_future().share());
  }
  else if(target->addonsPending)
    target->afterAddons.push_back([wear]() mutable { wear({}); });
  else
    wear({});
  return {{"job", jobId}};
}
