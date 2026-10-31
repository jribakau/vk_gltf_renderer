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

glm::vec3 vec3From(const Json& v)
{
  if(!v.is_array() || v.size() < 3)
    throw std::runtime_error("expected [x, y, z]");
  return {v[0].get<float>(), v[1].get<float>(), v[2].get<float>()};
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
  // a prop there or not: what puts it back, or takes it away
  if(cmd == "prop.remove" || cmd == "prop.restore")
    return args.contains("prop") && args["prop"].is_string() ? Json{{"kind", "props"}, {"ids", Json::array({args["prop"]})}} : Json();
  if(cmd == "scene.clear")
  {
    Json ids = Json::array();
    for(const auto& p : props)
      ids.push_back(p->id);
    return {{"kind", "props"}, {"ids", ids}};
  }
  if(cmd == "prop.set" || cmd == "camera.set")
  {
    const auto p = args.contains("prop") && args["prop"].is_string() ? findProp(args["prop"].get<std::string>()) : nullptr;
    if(!p)
      return nullptr;
    Json fields = Json::array();
    if(cmd == "prop.set")
      for(const char* f : {"parent", "name", "visible", "transform", "order"})
        if(args.contains(f) || (f == std::string_view("parent") && args.contains("bone")))
          fields.push_back(f);
    // a new parent (or bone) keeps the prop where it is in the world: its transform changes with it
    if((args.contains("parent") || args.contains("bone")) && !args.contains("transform"))
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
          if(c.value != c.rest)
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
  // what rewrites a figure's surfaces: its whole look before and after
  if(cmd == "preset" || cmd == "material.skin" || cmd == "material.eyes" || cmd == "material.makeup" || cmd == "material.tattoo" || cmd == "wardrobe.color"
     || cmd == "figure.look")
  {
    const auto p = figureProp(args);
    return p && p->figure ? Json{{"kind", "look"}, {"prop", p->id}} : Json();
  }
  if(cmd == "wardrobe.remove" || cmd == "wardrobe.restore")
  {
    // an item worn or not: what puts it back on, or takes it off
    const auto p = figureProp(args);
    if(!p || !p->figure || !args.contains("item"))
      return nullptr;
    return {{"kind", "items"}, {"prop", p->id}, {"items", Json::array({itemIdOf(args["item"])})}};
  }
  if(cmd.rfind("wardrobe.", 0) == 0 || cmd.rfind("dforce.", 0) == 0)
  {
    const auto p = figureProp(args);
    if(!p || !p->figure)
      return nullptr;
    if(cmd == "wardrobe.visible" || cmd == "wardrobe.cloth" || cmd == "wardrobe.tight")
      return {{"kind", cmd == "wardrobe.visible" ? "visible" : cmd == "wardrobe.cloth" ? "cloth" : "tight"}, {"prop", p->id}, {"item", itemIdOf(args.at("item"))}};
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

Json FvHost::Impl::journalState(const Json& scope)
{
  Json              ops  = Json::array();
  const std::string kind = scope.value("kind", "");
  if(kind == "props")
  {
    // each one back as it is now, or away; then what rode on them back on, and the view through one
    std::set<std::string> ids;
    for(const Json& id : scope["ids"])
      ids.insert(id.get<std::string>());
    for(const Json& id : scope["ids"])
      if(const auto p = findProp(id.get<std::string>()))
        ops.push_back(op("prop.restore", {{"prop", p->id}, {"record", propRecord(*p)}}));
      else
        ops.push_back(op("prop.remove", {{"prop", id}}));
    for(const auto& r : props)
      if(!r->parent.empty() && ids.count(r->parent))
      {
        const glm::quat& q = r->transform.rotation;
        Json             t = {{"t", vec3(r->transform.translation)}, {"r", {q.x, q.y, q.z, q.w}}, {"s", vec3(r->transform.scale)}};
        if(r->euler)
          t["e"] = vec3(*r->euler);
        ops.push_back(op("prop.set", {{"prop", r->id}, {"parent", r->parent}, {"bone", r->bone.empty() ? Json() : Json(r->bone)}}));
        ops.push_back(op("prop.set", {{"prop", r->id}, {"transform", t}}));
      }
    if(ids.count(viewCamera))
      ops.push_back(op("view.use", {{"camera", viewCamera}}));
    return ops;
  }
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
        ops.push_back(op("prop.set", {{"prop", id}, {"parent", p->parent.empty() ? Json() : Json(p->parent)}, {"bone", p->bone.empty() ? Json() : Json(p->bone)}}));
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
  else if(kind == "tight")
  {
    if(const fv::WornItem* w = wornItem(*f.fig, scope["item"].get<int>()))
      ops.push_back(op("wardrobe.tight", {{"prop", id}, {"item", scope["item"]}, {"amount", w->tight}}));
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
      ops.push_back(op("wardrobe.cloth", {{"prop", id}, {"item", item}, {"on", w->cloth.on ? Json(*w->cloth.on) : Json()}, {"strength", w->cloth.strength}, {"volume", w->cloth.volume}}));
      if(f.dforce)
        ops.push_back(op("dforce.enable", {{"prop", id}, {"on", f.dforce->enabled()}}));
    }
  }
  else if(kind == "look")
    ops.push_back(op("figure.look", {{"prop", id}, {"look", keepLook(f)}, {"state", sceneState({{"prop", id}})}}));
  else if(kind == "items")
    for(const Json& item : scope["items"])
      if(const fv::WornItem* w = wornItem(*f.fig, item.get<int>()))
        ops.push_back(op("wardrobe.restore", {{"prop", id}, {"item", w->id}, {"record", itemRecord(f, *w)}}));
      else
        ops.push_back(op("wardrobe.remove", {{"prop", id}, {"item", item}}));
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
  // the prop the scope names, else the one the command does (a removal's scope names a list)
  const auto        p    = findProp(scope.contains("prop") ? scope["prop"].get<std::string>() : args.value("prop", std::string()));
  const std::string name = p ? p->name : std::string();
  const auto        on   = [&] { return std::string(args.value("on", true) ? "on" : "off"); };
  if(cmd == "prop.set")
  {
    if(args.contains("name"))
      return "Rename " + name;
    if(args.contains("visible"))
      return (args["visible"].get<bool>() ? "Show " : "Hide ") + name;
    if(args.contains("parent") || args.contains("bone"))
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
  if(cmd == "prop.remove")
    return "Remove " + name;
  if(cmd == "preset")
    return "Apply " + args.value("name", std::string("preset"));
  if(cmd == "material.skin" || cmd == "material.eyes" || cmd == "material.makeup" || cmd == "material.tattoo" || cmd == "wardrobe.color")
    return args.value("label", cmd);
  if(cmd == "wardrobe.remove")
  {
    const fv::WornItem* w = p && p->figure && args.contains("item") ? wornItem(*p->figure->fig, itemIdOf(args["item"])) : nullptr;
    return "Take off " + (w ? (w->label.empty() ? w->rel : w->label) : std::string("item"));
  }
  if(cmd == "scene.clear")
    return "Clear the scene";
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
  if(cmd == "wardrobe.tight")
  {
    const fv::WornItem* w = p && p->figure ? wornItem(*p->figure->fig, scope["item"].get<int>()) : nullptr;
    return "Skin tight: " + (w ? (w->label.empty() ? w->rel : w->label) : std::string("item"));
  }
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
  const std::string  label  = journalLabel(cmd, args, scope);
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
  // a kept look is new each time it is read: what it was read from says whether anything changed
  const auto unkept = [](Json ops) {
    for(Json& o : ops)
      if(o.value("cmd", "") == "figure.look")
        o["args"].erase("look");
    return ops;
  };
  if(unkept(before) == unkept(after))
  {
    if(failed)
      std::rethrow_exception(failed);
    return result;
  }
  // a slider's or a drag's edits merge into one step: the same edit of the same things, back to back
  bool merges = cmd == "morph.set" || cmd == "material.set" || cmd == "lighting.set" || cmd == "camera.set"
                || (cmd == "pose.set" && !args.contains("label"));
  if(cmd == "prop.set")
    merges = !args.contains("name") && !args.contains("visible") && !args.contains("parent") && !args.contains("bone") && !args.contains("order");
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
                  .label   = label,
                  .key     = key,
                  .at      = std::chrono::steady_clock::now(),
                  .gesture = gesture.is_object() && gesture.contains("id"),
                  .closes  = gesture.is_object() && gesture.value("end", false)});
  publishHistory();
  pruneParked();
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
  if(cmd == "wardrobe.tight")
    return wardrobeTight(args);
  if(cmd == "dforce.enable" || cmd == "dforce.mode")
    return dforceCommand(cmd.substr(std::string("dforce.").size()), args);
  if(cmd == "lighting.set")
    return lightingSet(args);
  if(cmd == "scene.order")
    return sceneOrder(args);
  if(cmd == "view.use")
    return viewUse(args);
  if(cmd == "prop.remove")
    return propRemove(args);
  if(cmd == "prop.restore")
    return propRestore(args);
  if(cmd == "wardrobe.remove")
    return wardrobeRemove(args);
  if(cmd == "wardrobe.restore")
    return wardrobeRestore(args);
  if(cmd == "figure.look")
    return applyLook(args);
  if(cmd == "view.free")
  {
    // the free view as a kept scene had it, not framed again once its props are in
    if(auto manip = renderer->getCameraManipulator())
    {
      const auto at = [&](const char* k) { return glm::dvec3(vec3From(args.at(k))); };
      manip->setLookat(at("position"), at("target"), at("up"), true);
    }
    frameIn = 0;
    return {{"ok", true}};
  }
  throw std::runtime_error("the history can't apply " + cmd);
}

Json FvHost::Impl::historyStep(bool undo)
{
  if(historyRun)
    throw std::runtime_error("the last undo or redo is still loading");
  if(journal.grouping())
    throw std::runtime_error("a scene file is still opening");
  // dials sent before the undo land first, as a step of their own
  applyPendingMorphs();
  const auto step = undo ? journal.takeUndo() : journal.takeRedo();
  if(!step)
    return {{"label", nullptr}};
  return startRun({.undo     = undo,
                   .serial   = step->serial,
                   .label    = step->label,
                   .ops      = undo ? step->undo : step->redo,
                   .jobLabel = (undo ? "Undoing " : "Redoing ") + step->label});
}

Json FvHost::Impl::startRun(HistoryRun run)
{
  historyRun = std::move(run);
  if(auto done = runHistory())
    return *done;
  // a restore loads: the rest applies once it is in, under a job
  historyRun->job = ++jobs;
  job(historyRun->job, {{"label", historyRun->jobLabel}, {"state", "running"}});
  publishHistory();
  return {{"label", historyRun->label}, {"job", historyRun->job}};
}

std::optional<Json> FvHost::Impl::runHistory()
{
  HistoryRun& run = *historyRun;
  replaying       = true;
  while(run.next < run.ops.size())
  {
    const Json        o     = run.ops[run.next++];
    const std::string cmd   = o.value("cmd", "");
    const Json        scope = journalScope(cmd, o.value("args", Json::object()));
    // what it is about to change: the way back, ahead of what the ops before it read
    const Json before = scope.is_null() ? Json::array() : journalState(scope);
    try
    {
      const Json result = applyOp(o);
      run.opposite.insert(run.opposite.begin(), before.begin(), before.end());
      if(result.is_object() && result.contains("job"))
      {
        run.waitJob  = result["job"].get<uint64_t>();
        run.waitProp = o["args"].value("prop", "");
        replaying    = false;
        return std::nullopt;
      }
    }
    catch(const std::exception& e)
    {
      run.problems.push_back(cmd + ": " + e.what());
    }
  }
  replaying = false;
  journal.rewrite(run.serial, !run.undo, run.opposite);
  Json out = {{"label", run.label}};
  if(!run.problems.empty())
    out["problems"] = run.problems;
  if(run.job)
    job(run.job, run.problems.empty() ? Json{{"state", "done"}} : Json{{"state", "done"}, {"problems", run.problems}});
  historyRun.reset();
  publishHistory();
  pruneParked();
  return out;
}

void FvHost::Impl::pollHistory()
{
  if(!historyRun || !historyRun->waitJob)
    return;
  Json ended;
  {
    std::lock_guard lock(keptMutex);
    const auto it = endedJobs.find(historyRun->waitJob);
    if(it == endedJobs.end())
      return;
    ended = it->second;
  }
  if(ended.value("state", "") == "failed")
    historyRun->problems.push_back("restoring " + historyRun->waitProp + ": " + ended.value("error", std::string("it did not load")));
  historyRun->waitJob = 0;
  runHistory();
}

void FvHost::Impl::publishHistory()
{
  Json topic    = journal.topic();
  topic["busy"] = historyRun.has_value() || journal.grouping();
  protocol->events().snapshot("history", topic);
  // a step recorded, undone or redone: the scene the crash copy keeps changed
  noteRecovery();
}

void FvHost::Impl::noteRecovery()
{
  recoveryDirtyAt = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void FvHost::Impl::writeRecovery()
{
  int64_t       at  = recoveryDirtyAt.load();
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if(recoveryPath.empty() || !at || now - at < std::chrono::nanoseconds(kRecoveryQuiet).count() || !recoveryDirtyAt.compare_exchange_strong(at, 0))
    return;
  fv::timing::Scope t("recovery copy", 5);
  Json ids = Json::array();
  for(const auto& p : props)
    ids.push_back(p->id);
  Json ops = journalState({{"kind", "props"}, {"ids", ids}});
  ops.push_back(op("scene.order", {{"order", ids}}));
  ops.push_back(op("lighting.set", fv::lighting::toJson(lighting)));
  ops.push_back(op("view.use", {{"camera", viewCamera.empty() ? Json() : Json(viewCamera)}}));
  if(auto manip = renderer->getCameraManipulator(); manip && viewCamera.empty())
  {
    glm::dvec3 eye, center, up;
    manip->getLookat(eye, center, up);
    ops.push_back(op("view.free", {{"position", vec3(eye)}, {"target", vec3(center)}, {"up", vec3(up)}}));
  }
  const auto  wall = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  // a name that isn't UTF-8 (an OBJ's, a file's) is no reason to lose the copy
  std::string text = Json{{"version", 1}, {"props", props.size()}, {"at", wall}, {"ops", ops}}.dump(-1, ' ', false, Json::error_handler_t::replace);
  loaders.spawn([this, path = recoveryPath, text = std::move(text), n = ++recoveryWrites] {
    std::lock_guard lock(recoveryMutex);
    if(n <= recoveryWritten)
      return;
    recoveryWritten = n;
    // written whole, then put in place: a crash mid-write leaves the copy before
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
      std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      out << text;
      if(!out.good())
      {
        LOGW("[fv] recovery copy: could not write %s\n", tmp.string().c_str());
        return;
      }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if(ec)
      LOGW("[fv] recovery copy: %s\n", ec.message().c_str());
  });
}

Json FvHost::Impl::recoveryFile(const Json& args)
{
  const Json path = args.value("path", Json());
  if(path.is_string() && !path.get<std::string>().empty())
  {
    recoveryPath = pathOf(path.get<std::string>());
    // the copy there didn't all come back: it goes aside before anything is written over it
    if(const Json keepAs = args.value("keepAs", Json()); keepAs.is_string())
    {
      std::lock_guard lock(recoveryMutex);
      std::error_code ec;
      std::filesystem::rename(recoveryPath, pathOf(keepAs.get<std::string>()), ec);
      if(ec)
        LOGW("[fv] recovery copy: could not keep the last one aside: %s\n", ec.message().c_str());
    }
    // kept at the next frame
    recoveryDirtyAt = 1;
    return {{"ok", true}};
  }
  // none kept from now on: a write on its way is dropped and the copy goes (the scene closed as meant)
  const std::filesystem::path old = std::exchange(recoveryPath, {});
  recoveryDirtyAt                 = 0;
  std::lock_guard lock(recoveryMutex);
  recoveryWritten = std::max(recoveryWritten, recoveryWrites);
  if(!old.empty())
  {
    std::error_code ec;
    std::filesystem::remove(old, ec);
    std::filesystem::remove(std::filesystem::path(old) += ".tmp", ec);
  }
  return {{"ok", true}};
}

// A kept scene as written, or nothing when there is none or it can't be read
static Json readKept(const std::filesystem::path& path)
{
  std::ifstream in(path, std::ios::binary);
  if(!in)
    return nullptr;
  Json kept = Json::parse(in, nullptr, false);
  if(kept.is_discarded() || !kept.is_object() || kept.value("version", 0) != 1 || !kept.contains("ops") || !kept["ops"].is_array())
    return nullptr;
  return kept;
}

Json FvHost::Impl::recoveryPeek(const Json& args)
{
  const std::filesystem::path path = pathOf(args.at("path").get<std::string>());
  const Json                  kept = readKept(path);
  std::error_code             ec;
  if(kept.is_null() && std::filesystem::exists(path, ec))
    return {{"props", 0}, {"unreadable", true}};
  return kept.is_null() ? Json{{"props", 0}} : Json{{"props", kept.value("props", 0)}, {"at", kept.value("at", Json())}};
}

Json FvHost::Impl::recoveryForget(const Json& args)
{
  std::error_code ec;
  std::filesystem::remove(pathOf(args.at("path").get<std::string>()), ec);
  return {{"ok", true}};
}

Json FvHost::Impl::recoverScene(const Json& args)
{
  if(historyRun)
    throw std::runtime_error("the last undo or redo is still loading");
  if(journal.grouping())
    throw std::runtime_error("a scene file is still opening");
  if(!props.empty())
    throw std::runtime_error("the scene is not empty");
  const Json kept = readKept(pathOf(args.at("path").get<std::string>()));
  if(kept.is_null())
    throw std::runtime_error("no scene was kept there, or it can't be read");
  return startRun({.label = "Recover the scene", .ops = kept["ops"], .jobLabel = "Recovering the scene"});
}

void FvHost::Impl::recordAdded(const std::shared_ptr<Prop>& p, const std::string& label)
{
  if(replaying || !p || !findProp(p->id))
    return;
  journal.record({.redo  = journalState({{"kind", "props"}, {"ids", Json::array({p->id})}}),
                  .undo  = Json::array({op("prop.remove", {{"prop", p->id}})}),
                  .label = label,
                  .at    = std::chrono::steady_clock::now()});
  publishHistory();
  pruneParked();
}

void FvHost::Impl::pruneParked()
{
  std::set<std::string> named;
  std::set<uint64_t>    looks;
  const auto            note = [&](const Json& o) {
    if(o.value("cmd", "") == "prop.restore")
      named.insert(o["args"].value("prop", ""));
    else if(o.value("cmd", "") == "figure.look")
      looks.insert(o["args"].value("look", uint64_t(0)));
  };
  journal.forEachOp(note);
  // an undo that waits on a load holds its ops and the way back it read so far
  if(historyRun)
    for(const Json* ops : {&historyRun->ops, &historyRun->opposite})
      for(const Json& o : *ops)
        note(o);
  std::erase_if(keptLooks, [&](const auto& kept) { return !looks.count(kept.first); });
  std::erase_if(parked, [&](const auto& kept) { return !named.count(kept.first); });
  // the oldest go first: their undo loads them again
  while(parked.size() > kParked)
    parked.erase(parked.begin());
}

uint64_t FvHost::Impl::keepLook(const Loaded& f)
{
  auto l             = std::make_shared<FigureLook>();
  l->materials       = f.fig->loaded.scene.materials;
  l->materialPresets = f.materialPresets;
  l->presetsOver     = f.presetsOver;
  l->eyes            = f.eyes;
  l->eyesOwned       = f.eyesOwned;
  l->eyesPreset      = f.eyesPreset;
  l->skin            = f.skin;
  l->skinOwned       = f.skinOwned;
  l->skinPreset      = f.skinPreset;
  l->makeup          = f.makeup;
  l->makeupLayer     = f.makeupLayer;
  l->tattoos         = f.tattoos;
  l->tattooLayer     = f.tattooLayer;
  l->colors          = f.colors;
  if(f.fig->morphs)
    for(const fv::morph::Channel& c : f.fig->morphs->channels())
      if(!c.hidden)
        l->dials.emplace_back(c.name, c.value);
  if(f.fig->pose)
    for(const fv::pose::PoseBone& pb : f.fig->pose->bones())
      for(const fv::pose::Axis& a : pb.axes)
        if(const auto v = f.fig->pose->posed(pb.id, fv::pose::kindName(a.kind), a.axis))
          l->pose.push_back({pb.id, a.kind, a.axis, *v});
  for(const fv::WornItem& w : f.fig->items)
    l->visible[w.id] = w.visible;
  keptLooks[++lookCount] = std::move(l);
  return lookCount;
}

Json FvHost::Impl::applyLook(const Json& args)
{
  const auto p = findProp(args.at("prop").get<std::string>());
  if(!p || !p->figure)
    throw std::runtime_error("no such figure");
  const auto kept = keptLooks.find(args.at("look").get<uint64_t>());
  if(kept == keptLooks.end())
    throw std::runtime_error("its look is no longer kept");
  const FigureLook& l     = *kept->second;
  Loaded&           f     = *p->figure;
  fv::SceneGraph&   scene = f.fig->loaded.scene;
  // the surfaces as they were, lowered again (a wear since appended its own after them)
  std::set<int> changed;
  for(size_t i = 0; i < l.materials.size() && i < scene.materials.size(); i++)
  {
    scene.materials[i] = l.materials[i];
    changed.insert(int(i));
  }
  if(p->materials)
    p->materials->surfacesChanged(changed);
  f.materialPresets = l.materialPresets;
  f.presetsOver     = l.presetsOver;
  f.eyes            = l.eyes;
  f.eyesOwned       = l.eyesOwned;
  f.eyesPreset      = l.eyesPreset;
  f.skin            = l.skin;
  f.skinOwned       = l.skinOwned;
  f.skinPreset      = l.skinPreset;
  f.makeup          = l.makeup;
  f.makeupLayer     = l.makeupLayer;
  f.tattoos         = l.tattoos;
  f.tattooLayer     = l.tattooLayer;
  f.colors          = l.colors;
  // the dials and the pose where they differ
  if(f.fig->morphs)
  {
    std::vector<std::pair<std::string, double>> dials;
    for(const auto& [name, value] : l.dials)
      if(f.fig->morphs->get(name) != value)
        dials.emplace_back(name, value);
    if(!dials.empty())
      f.fig->morphs->setMany(dials);
  }
  if(f.fig->pose)
  {
    std::vector<fv::pose::ChannelValue> values;
    for(const fv::pose::ChannelValue& v : l.pose)
      if(f.fig->pose->posed(v.bone, fv::pose::kindName(v.kind), v.axis) != v.value)
        values.push_back(v);
    if(!values.empty())
    {
      f.fig->pose->setMany(values);
      protocol->events().snapshot("pose/" + f.id, poseSnapshot(f));
    }
  }
  for(const auto& [item, visible] : l.visible)
    if(const fv::WornItem* w = wornItem(*f.fig, item); w && w->visible != visible)
      wardrobeVisible({{"prop", p->id}, {"item", item}, {"visible", visible}});
  publishEyes(*p);
  publishSkin(*p);
  publishWardrobe(f);
  return {{"ok", true}};
}

Json FvHost::Impl::propRecord(const Prop& p)
{
  const glm::quat& r   = p.transform.rotation;
  Json             rec = {{"kind", kindName(p.kind)},
                          {"name", p.name},
                          {"visible", p.visible},
                          {"parent", p.parent.empty() ? Json() : Json(p.parent)},
                          {"bone", p.bone.empty() ? Json() : Json(p.bone)},
                          {"order", int(std::find_if(props.begin(), props.end(), [&](const auto& q) { return q.get() == &p; }) - props.begin())},
                          {"transform", {{"t", vec3(p.transform.translation)}, {"r", {r.x, r.y, r.z, r.w}}, {"s", vec3(p.transform.scale)}}}};
  if(p.euler)
    rec["transform"]["e"] = vec3(*p.euler);
  if(p.kind == PropKind::eCamera)
  {
    rec["camera"]                   = FvView::lensJson(p.camera.lens);
    rec["camera"]["locked"]         = p.camera.locked;
    rec["camera"]["targetDistance"] = p.camera.targetDistance;
  }
  if(p.kind == PropKind::eLight)
    rec["light"] = {{"type", p.light.type},     {"color", p.light.color}, {"intensity", p.light.intensity},
                    {"radius", p.light.radius}, {"inner", p.light.inner}, {"outer", p.light.outer}};
  // a figure's or a model's part of a scene file, for when it loads again
  if(p.figure || p.model)
    rec["state"] = sceneState({{"prop", p.id}});
  return rec;
}

Json FvHost::Impl::propRemove(const Json& args)
{
  auto p = findProp(args.at("prop").get<std::string>());
  if(!p)
    throw std::runtime_error("no such prop");
  // kept, for its undo
  std::erase_if(parked, [&](const auto& kept) { return kept.first == p->id; });
  parked.emplace_back(p->id, p);
  removeProp(p);
  publishScene();
  requestCompose("Removing " + p->name);
  return {{"ok", true}};
}

void FvHost::Impl::placeFrom(Prop& p, const Json& rec)
{
  p.name    = rec.value("name", p.name);
  p.visible = rec.value("visible", true);
  if(const Json& t = rec.value("transform", Json::object()); t.contains("t") && t.contains("r") && t.contains("s"))
  {
    p.transform.translation = vec3From(t["t"]);
    p.transform.rotation    = glm::quat(t["r"][3].get<float>(), t["r"][0].get<float>(), t["r"][1].get<float>(), t["r"][2].get<float>());
    p.transform.scale       = vec3From(t["s"]);
    p.euler                 = t.contains("e") ? std::optional<glm::vec3>(vec3From(t["e"])) : std::nullopt;
  }
  // under its parent when that is back already; else what puts it back on follows
  const std::string parent = rec.contains("parent") && rec["parent"].is_string() ? rec["parent"].get<std::string>() : "";
  p.parent                 = findProp(parent) ? parent : "";
  p.bone                   = !p.parent.empty() && rec.contains("bone") && rec["bone"].is_string() ? rec["bone"].get<std::string>() : "";
  if(p.kind == PropKind::eCamera && rec.contains("camera"))
  {
    p.camera.lens           = FvView::lensOf(rec["camera"], p.camera.lens);
    p.camera.locked         = rec["camera"].value("locked", false);
    p.camera.targetDistance = rec["camera"].value("targetDistance", p.camera.targetDistance);
  }
  if(p.kind == PropKind::eLight && rec.contains("light"))
  {
    const Json& l    = rec["light"];
    p.light.type      = l.value("type", p.light.type);
    p.light.color     = l.value("color", p.light.color);
    p.light.intensity = l.value("intensity", p.light.intensity);
    p.light.radius    = l.value("radius", p.light.radius);
    p.light.inner     = l.value("inner", p.light.inner);
    p.light.outer     = l.value("outer", p.light.outer);
  }
  // its place in the scene's order
  const auto self = std::find_if(props.begin(), props.end(), [&](const auto& q) { return q.get() == &p; });
  if(self != props.end())
  {
    auto keep = *self;
    props.erase(self);
    props.insert(props.begin() + std::clamp(rec.value("order", int(props.size())), 0, int(props.size())), std::move(keep));
  }
}

Json FvHost::Impl::propRestore(const Json& args)
{
  const std::string id  = args.at("prop").get<std::string>();
  const Json&       rec = args.at("record");
  if(findProp(id))
    throw std::runtime_error(id + " is in the scene already");
  std::shared_ptr<Prop> p;
  if(const auto kept = std::find_if(parked.begin(), parked.end(), [&](const auto& k) { return k.first == id; }); kept != parked.end())
  {
    // the same objects: composed again, its figure's state as it was
    p = kept->second;
    parked.erase(kept);
    props.push_back(p);
    propCount = std::max(propCount, std::atoi(id.c_str() + 2));
  }
  else
  {
    const std::string kind = rec.value("kind", "");
    if(kind == "camera" || kind == "light")
      p = addProp(kind == "camera" ? PropKind::eCamera : PropKind::eLight, rec.value("name", ""), id);
    else
    {
      // loaded again from where it came from, under its id, and its scene-file part put back
      const Json state  = rec.value("state", Json::object());
      const Json source = state.value("source", Json::object());
      const auto then   = [this, rec, state, kind](std::shared_ptr<Prop> q) {
        placeFrom(*q, rec);
        applyTransform(*q);
        applyVisible(*q);
        requestCompose("Restoring " + q->name);
        if(kind == "model")
        {
          if(q->materials && state.contains("materials"))
            q->materials->restoreEdits(state["materials"]);
        }
        else
        {
          Json restore    = state;
          restore["prop"] = q->id;
          const Json r    = sceneRestore(restore);
          // the history's run waits on the restore (its wears) next
          if(historyRun && historyRun->waitProp == q->id && r.contains("job"))
            historyRun->waitJob = r["job"].get<uint64_t>();
        }
        publishScene();
      };
      if(kind == "model")
        return importModel(source.value("path", ""), then, id);
      HostSettings s;
      {
        std::lock_guard lock(settingsMutex);
        s = settings;
      }
      std::optional<std::pair<std::vector<std::filesystem::path>, std::string>> library;
      if(source.contains("rel"))
        library = std::make_pair(s.roots, source["rel"].get<std::string>());
      else if(source.contains("path"))
        library = libraryFor(s, pathOf(source["path"].get<std::string>()));
      if(!library)
        throw std::runtime_error(rec.value("name", id) + " can't be loaded again: where it came from isn't known");
      return loadFigure(s, library->first, library->second, rec.value("name", ""), source, "", true, then, {}, {}, id);
    }
  }
  placeFrom(*p, rec);
  if(p->figure)
  {
    std::lock_guard lock(morphMutex);
    morphFigures.insert(p->id);
  }
  if(p->figure)
  {
    // its topics went with it: published again as new
    p->figure->morphList        = -1;
    p->figure->morphsSeen       = -1;
    p->figure->dforceVersion    = -1;
    p->figure->collisionVersion = -1;
  }
  requestCompose("Restoring " + p->name);
  publishScene();
  return {{"prop", id}};
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
