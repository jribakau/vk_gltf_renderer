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
// FvHost: the edit history (daz-viewer docs/command-journal-plan.md, C1). Each edit the journal
// knows reads what it is about to change (its scope) as the ops that would put it back, applies,
// and reads it again: the two are the step's undo and redo. Undo and redo apply those ops through
// applyOp, the same functions the commands use.
//

#include "fv_host_impl.hpp"

namespace {

Json op(const char* cmd, Json args)
{
  return {{"cmd", cmd}, {"args", std::move(args)}};
}

Json vec3(const glm::vec3& v)
{
  return Json::array({v.x, v.y, v.z});
}

// A merge patch's keys with its values dropped: what it touches, not what it sets
Json shapeOf(const Json& patch)
{
  if(!patch.is_object())
    return nullptr;
  Json out = Json::object();
  for(auto it = patch.begin(); it != patch.end(); ++it)
    out[it.key()] = shapeOf(it.value());
  return out;
}

// `state` cut to `shape`: where the shape ends, the state's value is taken whole
Json project(const Json& state, const Json& shape)
{
  Json out = Json::object();
  for(auto it = shape.begin(); it != shape.end(); ++it)
  {
    const Json had = state.is_object() ? state.value(it.key(), Json()) : Json();
    out[it.key()]  = it.value().is_object() && had.is_object() ? project(had, it.value()) : had;
  }
  return out;
}

// material.set's `material`: "m:N", N, or a list of them
std::vector<int> materialsNamed(const Json& v)
{
  std::vector<int> out;
  const auto       one = [&](const Json& m) {
    if(m.is_number_integer())
      out.push_back(m.get<int>());
    else if(m.is_string() && m.get<std::string>().rfind("m:", 0) == 0)
      out.push_back(std::atoi(m.get<std::string>().c_str() + 2));
  };
  if(v.is_array())
    for(const Json& m : v)
      one(m);
  else
    one(v);
  return out;
}

const char* dforceModeName(fv::dforce::Mode m)
{
  return m == fv::dforce::Mode::eLive ? "live" : m == fv::dforce::Mode::eLiveGpu ? "gpu" : "settle";
}

}  // namespace

Json FvHost::Impl::journalScope(const std::string& cmd, const Json& args) const
{
  if(cmd == "prop.set" || cmd == "camera.set")
  {
    const auto p = args.contains("prop") && args["prop"].is_string() ? findProp(args["prop"].get<std::string>()) : nullptr;
    if(!p)
      return nullptr;
    Json fields = Json::array();
    if(cmd == "prop.set")
      for(const char* f : {"parent", "name", "visible", "transform", "order"})
        if(args.contains(f))
          fields.push_back(f);
    // a new parent keeps the prop where it is in the world: its transform changes with it
    if(args.contains("parent") && !args.contains("transform"))
      fields.push_back("transform");
    if((args.contains("camera") || cmd == "camera.set") && p->kind == PropKind::eCamera)
      fields.push_back("camera");
    if(args.contains("light") && p->kind == PropKind::eLight)
      fields.push_back("light");
    return {{"kind", "prop"}, {"prop", p->id}, {"fields", fields}};
  }
  if(cmd.rfind("morph.", 0) == 0 || cmd.rfind("pose.", 0) == 0)
  {
    const auto p = figureProp(args);
    if(!p || !p->figure)
      return nullptr;
    const fv::Figure& fig = *p->figure->fig;
    if(cmd == "morph.set" || cmd == "morph.reset")
    {
      if(!fig.morphs)
        return nullptr;
      Json dials = Json::array();
      if(cmd == "morph.set")
      {
        if(const auto it = args.find("values"); it != args.end() && it->is_object())
          for(auto v = it->begin(); v != it->end(); ++v)
            if(v.value().is_number())
              dials.push_back(v.key());
      }
      else
        for(const fv::morph::Channel& c : fig.morphs->channels())
          if(c.value != 0)
            dials.push_back(c.name);
      return {{"kind", "morphs"}, {"prop", p->id}, {"dials", dials}};
    }
    if(!fig.pose)
      return nullptr;
    Json keys = Json::array();
    if(cmd == "pose.drag")
    {
      // the gizmo turns (and moves) one bone: all its channels
      if(const fv::pose::PoseBone* pb = fig.pose->get(args.value("bone", "")))
        for(const fv::pose::Axis& a : pb->axes)
          keys.push_back(pb->id + "/" + axisKey(a));
    }
    else if(cmd == "pose.set")
    {
      if(const auto it = args.find("values"); it != args.end() && it->is_object())
        for(auto v = it->begin(); v != it->end(); ++v)
          keys.push_back(v.key());
    }
    else
      for(const fv::pose::PoseBone& pb : fig.pose->bones())
        if(cmd == "pose.reset" || fv::pose::isEyeBone(pb.name))
          for(const fv::pose::Axis& a : pb.axes)
            if(cmd == "pose.eyes" || a.value != a.initial)
              keys.push_back(pb.id + "/" + axisKey(a));
    return {{"kind", "pose"}, {"prop", p->id}, {"keys", keys}};
  }
  if(cmd == "material.set" || cmd == "material.reset")
  {
    const auto p = propArg(args);
    if(!p || !p->materials)
      return nullptr;
    std::vector<int> index;
    if(args.contains("material"))
      index = materialsNamed(args["material"]);
    else
      for(size_t i = 0; i < p->materials->count(); i++)
        if(!p->materials->editsOf(int(i)).empty())
          index.push_back(int(i));
    return {{"kind", "materials"}, {"prop", p->id}, {"materials", index}};
  }
  if(cmd.rfind("wardrobe.", 0) == 0 || cmd.rfind("dforce.", 0) == 0)
  {
    const auto p = figureProp(args);
    if(!p || !p->figure)
      return nullptr;
    if(cmd == "wardrobe.visible" || cmd == "wardrobe.cloth")
      return {{"kind", cmd == "wardrobe.visible" ? "visible" : "cloth"}, {"prop", p->id}, {"item", itemIdOf(args.at("item"))}};
    return {{"kind", cmd == "wardrobe.collision" ? "collision" : "dforce"}, {"prop", p->id}};
  }
  if(cmd == "lighting.set")
    return {{"kind", "lighting"}, {"shape", shapeOf(args)}};
  if(cmd == "scene.order")
    return {{"kind", "order"}};
  if(cmd == "view.use")
    return {{"kind", "view"}};
  return nullptr;
}

Json FvHost::Impl::journalState(const Json& scope) const
{
  Json              ops  = Json::array();
  const std::string kind = scope.value("kind", "");
  if(kind == "lighting")
    ops.push_back(op("lighting.set", project(fv::lighting::toJson(lighting), scope["shape"])));
  else if(kind == "order")
  {
    Json order = Json::array();
    for(const auto& p : props)
      order.push_back(p->id);
    ops.push_back(op("scene.order", {{"order", order}}));
  }
  else if(kind == "view")
    ops.push_back(op("view.use", {{"camera", viewCamera.empty() ? Json() : Json(viewCamera)}}));
  if(!scope.contains("prop"))
    return ops;
  const std::string id = scope["prop"].get<std::string>();
  const auto        p  = findProp(id);
  if(!p)
    return ops;
  if(kind == "prop")
  {
    Json set = {{"prop", id}};
    for(const Json& f : scope["fields"])
    {
      const std::string field = f.get<std::string>();
      if(field == "parent")  // first: a new parent moves the transform, which comes after
        ops.push_back(op("prop.set", {{"prop", id}, {"parent", p->parent.empty() ? Json() : Json(p->parent)}}));
      else if(field == "name")
        set["name"] = p->name;
      else if(field == "visible")
        set["visible"] = p->visible;
      else if(field == "order")
        set["order"] = int(std::find(props.begin(), props.end(), p) - props.begin());
      else if(field == "transform")
      {
        const glm::quat& r = p->transform.rotation;
        set["transform"]   = {{"t", vec3(p->transform.translation)}, {"r", {r.x, r.y, r.z, r.w}}, {"s", vec3(p->transform.scale)}};
        if(p->euler)
          set["transform"]["e"] = vec3(*p->euler);
      }
      else if(field == "camera")
      {
        set["camera"]                   = FvView::lensJson(p->camera.lens);
        set["camera"]["locked"]         = p->camera.locked;
        set["camera"]["targetDistance"] = p->camera.targetDistance;
      }
      else if(field == "light")
        set["light"] = {{"type", p->light.type},     {"color", p->light.color}, {"intensity", p->light.intensity},
                        {"radius", p->light.radius}, {"inner", p->light.inner}, {"outer", p->light.outer}};
    }
    if(set.size() > 1)
      ops.push_back(op("prop.set", set));
  }
  else if(kind == "materials" && p->materials)
    for(const Json& m : scope["materials"])
    {
      // its edits from none: what it had, and nothing it did not
      ops.push_back(op("material.reset", {{"prop", id}, {"material", m}}));
      if(Json values = p->materials->editsOf(m.get<int>()); !values.empty())
        ops.push_back(op("material.set", {{"prop", id}, {"material", m}, {"values", std::move(values)}}));
    }
  if(!p->figure)
    return ops;
  const Loaded& f = *p->figure;
  if(kind == "morphs" && f.fig->morphs)
  {
    Json values = Json::object();
    for(const Json& d : scope["dials"])
      values[d.get<std::string>()] = f.fig->morphs->get(d.get<std::string>());
    ops.push_back(op("morph.set", {{"prop", id}, {"values", values}}));
  }
  else if(kind == "pose" && f.fig->pose)
  {
    Json values = Json::object();
    for(const Json& k : scope["keys"])
    {
      // "bone/kind/axis": the bone id may itself hold no slash
      const std::string key = k.get<std::string>();
      const size_t      a   = key.rfind('/');
      const size_t      b   = a == std::string::npos || a == 0 ? std::string::npos : key.rfind('/', a - 1);
      if(b == std::string::npos)
        continue;
      if(const auto v = f.fig->pose->posed(key.substr(0, b), key.substr(b + 1, a - b - 1), key.substr(a + 1)))
        values[key] = *v;
    }
    ops.push_back(op("pose.set", {{"prop", id}, {"values", values}}));
  }
  else if(kind == "visible" || kind == "cloth")
  {
    const int           item = scope["item"].get<int>();
    const fv::WornItem* w    = wornItem(*f.fig, item);
    if(!w)
      return ops;
    if(kind == "visible")
      ops.push_back(op("wardrobe.visible", {{"prop", id}, {"item", item}, {"visible", w->visible}}));
    else
    {
      // turning an item's cloth on turns the figure's dForce on: that goes back after it
      ops.push_back(op("wardrobe.cloth", {{"prop", id}, {"item", item}, {"on", w->cloth.on ? Json(*w->cloth.on) : Json()}, {"strength", w->cloth.strength}}));
      if(f.dforce)
        ops.push_back(op("dforce.enable", {{"prop", id}, {"on", f.dforce->enabled()}}));
    }
  }
  else if(kind == "collision" && f.collider)
    ops.push_back(op("wardrobe.collision", {{"prop", id}, {"on", f.collider->enabled()}}));
  else if(kind == "dforce" && f.dforce)
  {
    ops.push_back(op("dforce.enable", {{"prop", id}, {"on", f.dforce->enabled()}}));
    ops.push_back(op("dforce.mode", {{"prop", id}, {"mode", dforceModeName(f.dforce->mode())}}));
  }
  return ops;
}

// As the app's own undo names its steps (src/lib/undo/history.svelte.ts)
std::string FvHost::Impl::journalLabel(const std::string& cmd, const Json& args, const Json& scope) const
{
  const auto        p    = scope.contains("prop") ? findProp(scope["prop"].get<std::string>()) : nullptr;
  const std::string name = p ? p->name : std::string();
  const auto        on   = [&] { return std::string(args.value("on", true) ? "on" : "off"); };
  if(cmd == "prop.set")
  {
    if(args.contains("name"))
      return "Rename " + name;
    if(args.contains("visible"))
      return (args["visible"].get<bool>() ? "Show " : "Hide ") + name;
    if(args.contains("parent"))
      return "Change what " + name + " rides on";
    if(const auto t = args.find("transform"); t != args.end() && t->is_object())
      return (t->size() != 1 || t->contains("t") ? "Move " : t->contains("s") ? "Scale " : "Rotate ") + name;
    if(args.contains("camera"))
      return "Lens of " + name;
    if(args.contains("light"))
      return "Light " + name;
    return "Reorder " + name;
  }
  if(cmd == "camera.set")
    return "Lens of " + name;
  if(cmd == "morph.set")
  {
    const Json& dials = scope["dials"];
    if(dials.size() != 1)
      return std::to_string(dials.size()) + " morphs";
    const std::string         dial = dials[0].get<std::string>();
    const fv::morph::Channel* c    = p && p->figure && p->figure->fig->morphs ? p->figure->fig->morphs->channel(dial) : nullptr;
    return "Morph " + (c && !c->label.empty() ? c->label : dial);
  }
  if(cmd == "morph.reset")
    return "Reset morphs";
  if(cmd == "pose.set" || cmd == "pose.drag")
  {
    // a whole-pose edit sent as one pose.set (Mirror pose) names itself
    if(args.contains("label") && args["label"].is_string())
      return args["label"].get<std::string>();
    const Json& keys = scope["keys"];
    std::string bone = keys.empty() ? std::string() : keys[0].get<std::string>();
    if(const size_t a = bone.rfind('/'); a != std::string::npos && a > 0)
      bone = bone.substr(0, bone.rfind('/', a - 1));
    const fv::pose::PoseBone* pb = p && p->figure && p->figure->fig->pose ? p->figure->fig->pose->get(bone) : nullptr;
    return "Pose " + (pb && !pb->label.empty() ? pb->label : bone);
  }
  if(cmd == "pose.reset")
    return "Reset pose";
  if(cmd == "pose.eyes")
    return "Eyes on camera";
  if(cmd == "material.set")
  {
    const Json&       ms    = scope["materials"];
    const std::string first = p && p->materials && !ms.empty() ? p->materials->nameOf(ms[0].get<int>()) : "surface";
    return ms.size() > 1 ? "Material " + first + " and " + std::to_string(ms.size() - 1) + " more" : "Material " + first;
  }
  if(cmd == "material.reset")
    return args.contains("material") ? "Reset material" : "Reset all materials";
  if(cmd == "wardrobe.visible" || cmd == "wardrobe.cloth")
  {
    const fv::WornItem* w     = p && p->figure ? wornItem(*p->figure->fig, scope["item"].get<int>()) : nullptr;
    const std::string   label = w ? (w->label.empty() ? w->rel : w->label) : "item";
    if(cmd == "wardrobe.visible")
      return (args.value("visible", true) ? "Show " : "Hide ") + label;
    if(args.contains("on"))
      return "Cloth " + (args["on"].is_boolean() ? on() : std::string("as its file has it")) + ": " + label;
    return "Cloth strength: " + label;
  }
  if(cmd == "wardrobe.collision")
    return "Collision " + on();
  if(cmd == "dforce.enable")
    return "dForce " + on();
  if(cmd == "dforce.mode")
  {
    const std::string mode = args.value("mode", "settle");
    return "dForce " + (mode == "gpu" ? std::string("live on the GPU") : mode);
  }
  if(cmd == "lighting.set")
    return "Lighting";
  if(cmd == "scene.order")
    return "Reorder";
  if(cmd == "view.use")
  {
    const auto camera = args.contains("camera") && args["camera"].is_string() ? findProp(args["camera"].get<std::string>()) : nullptr;
    return camera ? "Look through " + camera->name : "Free view";
  }
  return cmd;
}

Json FvHost::Impl::journaled(const std::string& cmd, const Json& args, const std::function<Json()>& apply)
{
  if(replaying)
    return apply();
  const Json scope = journalScope(cmd, args);
  if(scope.is_null())
    return apply();
  const Json         before = journalState(scope);
  Json               result;
  std::exception_ptr failed;
  try
  {
    result = apply();
  }
  catch(...)
  {
    // what it changed before it threw is still a step, or the history would not match the scene
    failed = std::current_exception();
  }
  const Json after = journalState(scope);
  if(before == after)
  {
    if(failed)
      std::rethrow_exception(failed);
    return result;
  }
  // a slider's or a drag's edits merge into one step: the same edit of the same things, back to back
  bool merges = cmd == "morph.set" || cmd == "material.set" || cmd == "lighting.set" || cmd == "camera.set"
                || (cmd == "pose.set" && !args.contains("label"));
  if(cmd == "prop.set")
    merges = !args.contains("name") && !args.contains("visible") && !args.contains("parent") && !args.contains("order");
  Json shape = shapeOf(args);
  if(shape.is_object())
    for(const char* k : {"prop", "gesture"})
      shape.erase(k);
  std::string key = merges ? cmd + "|" + scope.dump() + "|" + shape.dump() : std::string();
  // a gesture ({id, end?}: a viewport drag, a slider's drag) is one step however long it takes
  const Json gesture = args.value("gesture", Json());
  if(gesture.is_object() && gesture.contains("id"))
    key = cmd + "|" + scope.dump() + "|gesture " + gesture["id"].dump();
  journal.record({.redo    = after,
                  .undo    = before,
                  .label   = journalLabel(cmd, args, scope),
                  .key     = key,
                  .at      = std::chrono::steady_clock::now(),
                  .gesture = gesture.is_object() && gesture.contains("id"),
                  .closes  = gesture.is_object() && gesture.value("end", false)});
  publishHistory();
  if(failed)
    std::rethrow_exception(failed);
  return result;
}

Json FvHost::Impl::edit(const std::string& cmd, const Json& args)
{
  return journaled(cmd, args, [&] { return applyOp({{"cmd", cmd}, {"args", args}}); });
}

Json FvHost::Impl::applyOp(const Json& o)
{
  const std::string cmd  = o.at("cmd").get<std::string>();
  const Json&       args = o.at("args");
  if(cmd == "prop.set")
    return propSet(args);
  if(cmd == "morph.set")
  {
    auto figure = figureArg(args);
    if(!figure || !figure->fig->morphs)
      throw std::runtime_error("no such figure");
    std::vector<std::pair<std::string, double>> values;
    for(auto it = args.at("values").begin(); it != args.at("values").end(); ++it)
      if(it.value().is_number())
        values.emplace_back(it.key(), it.value().get<double>());
    figure->fig->morphs->setMany(values);
    return {{"ok", true}};
  }
  if(cmd == "pose.set")
    return poseSet(args);
  if(cmd == "material.set" || cmd == "material.reset")
  {
    auto p = propArg(args);
    if(!p || !p->materials)
      throw std::runtime_error("Select a figure or a model: its surfaces are edited here.");
    return cmd == "material.set" ? p->materials->set(args) : p->materials->reset(args);
  }
  if(cmd == "wardrobe.visible")
    return wardrobeVisible(args);
  if(cmd == "wardrobe.collision")
    return wardrobeCollision(args);
  if(cmd == "wardrobe.cloth")
    return wardrobeCloth(args);
  if(cmd == "dforce.enable" || cmd == "dforce.mode")
    return dforceCommand(cmd.substr(std::string("dforce.").size()), args);
  if(cmd == "lighting.set")
    return lightingSet(args);
  if(cmd == "scene.order")
    return sceneOrder(args);
  if(cmd == "view.use")
    return viewUse(args);
  throw std::runtime_error("the history can't apply " + cmd);
}

Json FvHost::Impl::historyStep(bool undo)
{
  // dials sent before the undo land first, as a step of their own
  applyPendingMorphs();
  const auto step = undo ? journal.takeUndo() : journal.takeRedo();
  if(!step)
    return {{"label", nullptr}};
  Json problems = Json::array();
  replaying     = true;
  for(const Json& o : undo ? step->undo : step->redo)
    try
    {
      applyOp(o);
    }
    catch(const std::exception& e)
    {
      // a prop that is gone (removing one is no step yet): the rest of the step still applies
      problems.push_back(o.value("cmd", "") + ": " + e.what());
    }
  replaying = false;
  publishHistory();
  Json out = {{"label", step->label}};
  if(!problems.empty())
    out["problems"] = problems;
  return out;
}

void FvHost::Impl::publishHistory()
{
  protocol->events().snapshot("history", journal.topic());
}

// every prop's place at once, as a scene file opens: one composition, not one per prop that moves
Json FvHost::Impl::sceneOrder(const Json& args)
{
  std::vector<std::shared_ptr<Prop>> next;
  for(const Json& id : args.at("order"))
    if(id.is_string())
      if(auto p = findProp(id.get<std::string>()); p && std::find(next.begin(), next.end(), p) == next.end())
        next.push_back(p);
  // what the list leaves out keeps its order, after what it names
  for(const auto& p : props)
    if(std::find(next.begin(), next.end(), p) == next.end())
      next.push_back(p);
  if(next != props)
  {
    props = std::move(next);
    requestCompose("Reordering");
    publishScene();
  }
  return {{"ok", true}};
}
