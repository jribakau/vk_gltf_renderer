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
// Placing props by what they take up and what is under them (fv_host.hpp's prop.drop and prop.aim,
// view.set's solo): a prop's bounds, and rays against the scene, from the triangle sets the contact
// pass keeps already (a DAZ object's posed meshes, a model's meshes), so a figure that sits is boxed
// as it sits, not as it was made. A set is indexed the first time a ray needs it, on the pool. And
// the nodes inside a prop (node.set): a set's wall hidden, a lamp in it moved.
//

#include "fv_host_impl.hpp"

#include <figure/physics/collide/surface_index.hpp>

namespace {

using Bounds = std::array<glm::vec3, 2>;

Bounds emptyBounds()
{
  return {glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest())};
}

bool isEmpty(const Bounds& b)
{
  return b[0].x > b[1].x;
}

void addPoint(Bounds& b, const glm::vec3& p)
{
  b[0] = glm::min(b[0], p);
  b[1] = glm::max(b[1], p);
}

// a box's corners carried by a matrix
void addBox(Bounds& b, const glm::vec3& lo, const glm::vec3& hi, const glm::mat4& m)
{
  for(int c = 0; c < 8; c++)
    addPoint(b, glm::vec3(m * glm::vec4(c & 1 ? hi.x : lo.x, c & 2 ? hi.y : lo.y, c & 4 ? hi.z : lo.z, 1.f)));
}

// Whether the ray from o along d crosses the box ahead of o.
bool crosses(const fv::physics::Box& box, const glm::vec3& o, const glm::vec3& d)
{
  double t0 = 0, t1 = std::numeric_limits<double>::max();
  for(int a = 0; a < 3; a++)
  {
    if(std::abs(d[a]) < 1e-12)
    {
      if(o[a] < box.min[a] || o[a] > box.max[a])
        return false;
      continue;
    }
    double n = (box.min[a] - o[a]) / d[a], f = (box.max[a] - o[a]) / d[a];
    if(n > f)
      std::swap(n, f);
    t0 = std::max(t0, n);
    t1 = std::min(t1, f);
    if(t0 > t1)
      return false;
  }
  return true;
}

}  // namespace

std::vector<FvHost::Impl::PlacedSet> FvHost::Impl::placedSets(const Prop& p)
{
  std::vector<PlacedSet> out;
  if(!renderer->feedsHostScene())
    return out;
  rs::RetainedScene& scene = renderer->hostScene();
  if(p.model)
  {
    // a model's skinned meshes are left out, as the contact pass leaves them: their triangles are not
    // where the file has them
    const tinygltf::Model& model = *p.model;
    for(size_t node = 0; node < model.nodes.size() && node < p.imported.entities.size(); node++)
    {
      const tinygltf::Node& n = model.nodes[node];
      const rs::EntityId    e = p.imported.entities[node];
      if(n.mesh >= 0 && n.skin < 0 && scene.entity(e) && scene.shown(e))
        out.push_back({contactMesh(p.model, n.mesh), scene.world(e)});
    }
  }
  else if(p.figure && p.figure->feed)
  {
    const rs::EntityId root = p.figure->feed->entityOf(p.figure->fig->loaded.root);
    if(scene.entity(root))
    {
      const glm::mat4 world = scene.world(root);
      for(const auto& set : contactSets(*p.figure))
        out.push_back({set, world});
    }
  }
  return out;
}

// What a prop takes up in the world: its posed triangles, else what it draws; a camera or a light is
// where it stands.
std::optional<FvHost::Impl::Bounds> FvHost::Impl::ownBounds(const Prop& p)
{
  Bounds b = emptyBounds();
  for(const PlacedSet& s : placedSets(p))
  {
    const fv::physics::Box& box = s.set->bounds();
    if(!box.empty())
      addBox(b, glm::vec3(box.min[0], box.min[1], box.min[2]), glm::vec3(box.max[0], box.max[1], box.max[2]), s.toWorld);
  }
  if(isEmpty(b) && (p.figure || p.model) && renderer->feedsHostScene() && renderer->hostScene().entity(p.root))
  {
    rs::RetainedScene& scene = renderer->hostScene();
    for(const rs::DrawItem& item : scene.drawItems())
      if(item.shown && scene.isWithin(item.entity, p.root))
        if(const std::array<glm::vec3, 2> mb = scene.mesh(item.mesh)->bounds(); mb[0].x <= mb[1].x)
          addBox(b, mb[0], mb[1], item.world);
  }
  if(isEmpty(b) && (p.kind == PropKind::eCamera || p.kind == PropKind::eLight))
    addPoint(b, glm::vec3(worldOf(p)[3]));
  return isEmpty(b) ? std::nullopt : std::optional<Bounds>(b);
}

std::optional<FvHost::Impl::Bounds> FvHost::Impl::propBounds(const Prop& p)
{
  Bounds b = emptyBounds();
  const auto add = [&](const Prop& q) {
    if(const auto own = ownBounds(q))
    {
      addPoint(b, (*own)[0]);
      addPoint(b, (*own)[1]);
    }
  };
  add(p);
  // with what rides on it: a group is what it carries
  for(const auto& q : props)
    if(ridesOn(*q, p.id) && shown(*q))
      add(*q);
  if(isEmpty(b))
    addPoint(b, glm::vec3(worldOf(p)[3]));
  return b;
}

std::optional<FvHost::Impl::Bounds> FvHost::Impl::selectionBounds(const std::vector<std::shared_ptr<Prop>>& ps)
{
  Bounds b = emptyBounds();
  for(const auto& p : ps)
    if(const auto pb = propBounds(*p))
    {
      addPoint(b, (*pb)[0]);
      addPoint(b, (*pb)[1]);
    }
  return isEmpty(b) ? std::nullopt : std::optional<Bounds>(b);
}

// The props a placing command moves: those it names ({props: [id]} or {prop}), else the selection;
// of those, only what rides on none of the others, which moves with what carries it.
std::vector<std::shared_ptr<Prop>> FvHost::Impl::movingArg(const Json& args) const
{
  std::vector<std::string> ids;
  if(args.contains("props") && args["props"].is_array())
  {
    for(const Json& id : args["props"])
      if(id.is_string())
        ids.push_back(id.get<std::string>());
  }
  else if(args.contains("prop") && args["prop"].is_string())
    ids.push_back(args["prop"].get<std::string>());
  else if(view)
    for(const std::string& id : view->selectedProps())
      ids.push_back(id);
  std::vector<std::shared_ptr<Prop>> out;
  for(const std::string& id : ids)
  {
    auto p = findProp(id);
    if(!p)
      throw std::runtime_error("no such prop");
    if(std::none_of(ids.begin(), ids.end(), [&](const std::string& other) { return other != id && ridesOn(*p, other); })
       && std::find(out.begin(), out.end(), p) == out.end())
      out.push_back(p);
  }
  return out;
}

std::optional<FvHost::Impl::SceneHit> FvHost::Impl::rayThrough(const std::vector<PlacedSet>& sets, const glm::vec3& o, const glm::vec3& d, bool madeOnly)
{
  std::optional<SceneHit> best;
  double                  bestT = std::numeric_limits<double>::max();
  for(const PlacedSet& s : sets)
  {
    if(madeOnly && !s.set->made())
      continue;
    const glm::mat4 inv = glm::inverse(s.toWorld);
    const glm::vec3 os  = glm::vec3(inv * glm::vec4(o, 1.f));
    const glm::vec3 ds  = glm::mat3(inv) * d;  // t stays the world's: os + t ds is o + t d
    if(!crosses(s.set->bounds(), os, ds))
      continue;
    const fv::physics::TriangleSet& set = s.set->get();
    if(set.index().empty())
      continue;
    fv::physics::Stamps& stamps = fv::physics::threadStamps();
    const double         od[3]  = {os.x, os.y, os.z};
    const double         dd[3]  = {ds.x, ds.y, ds.z};
    const auto           hit    = set.index().firstHit(od, dd, bestT, stamps.stamp, stamps.next(set.index().triangles()));
    if(hit.tri < 0 || hit.t >= bestT)
      continue;
    const auto   P = set.positions();
    const auto   T = set.tris();
    const size_t t = size_t(hit.tri);
    const glm::vec3 a(P[T[t * 3] * 3ull], P[T[t * 3] * 3ull + 1], P[T[t * 3] * 3ull + 2]);
    const glm::vec3 b(P[T[t * 3 + 1] * 3ull], P[T[t * 3 + 1] * 3ull + 1], P[T[t * 3 + 1] * 3ull + 2]);
    const glm::vec3 c(P[T[t * 3 + 2] * 3ull], P[T[t * 3 + 2] * 3ull + 1], P[T[t * 3 + 2] * 3ull + 2]);
    glm::vec3       n = glm::transpose(glm::inverse(glm::mat3(s.toWorld))) * glm::cross(b - a, c - a);
    n                 = glm::length(n) > 1e-20f ? glm::normalize(n) : -d;
    if(glm::dot(n, d) > 0)
      n = -n;
    bestT = hit.t;
    best  = SceneHit{o + float(hit.t) * d, n};
  }
  return best;
}

float FvHost::Impl::groundHeight() const
{
  return lighting.ground.autoHeight ? autoGround : float(lighting.ground.height);
}

// prop.drop: what it names (or the selection) comes down, or up out of what it is sunk in, until its
// lowest point rests on what is under it, else on the ground. Rays go straight down from above its
// top, through its middle and four points toward its corners, and the highest surface any of them
// meets is what it rests on. A set too big to have been indexed yet is on the pool first: the answer
// is then a job.
Json FvHost::Impl::propDrop(const Json& args)
{
  const auto moving = movingArg(args);
  if(moving.empty())
    throw std::runtime_error("nothing is selected to drop");
  const auto box = selectionBounds(moving);
  if(!box)
    throw std::runtime_error("nothing selected takes up room to drop");
  // what it can rest on: everything shown but what moves and what rides on it
  std::vector<PlacedSet> under;
  for(const auto& q : props)
  {
    if(!shown(*q) || std::any_of(moving.begin(), moving.end(), [&](const auto& m) { return m == q || ridesOn(*q, m->id); }))
      continue;
    for(PlacedSet& s : placedSets(*q))
      under.push_back(std::move(s));
  }
  const glm::vec3 lo = (*box)[0], hi = (*box)[1];
  const glm::vec3 mid = (lo + hi) * 0.5f;
  const float     top = hi.y + std::max(0.01f, 0.01f * (hi.y - lo.y));
  std::vector<glm::vec3> starts{{mid.x, top, mid.z}};
  for(const glm::vec2 k : {glm::vec2(-1, -1), glm::vec2(1, -1), glm::vec2(-1, 1), glm::vec2(1, 1)})
    starts.push_back({mid.x + k.x * 0.4f * (hi.x - lo.x), top, mid.z + k.y * 0.4f * (hi.z - lo.z)});
  const glm::vec3 down(0, -1, 0);
  // a set those rays cross that isn't indexed yet is indexed on the pool, then the drop runs again
  if(!args.value("indexed", false))
  {
    std::vector<std::shared_ptr<const fv::physics::SharedTriangles>> unmade;
    for(const PlacedSet& s : under)
      if(!s.set->made())
      {
        const glm::mat4 inv = glm::inverse(s.toWorld);
        if(std::any_of(starts.begin(), starts.end(), [&](const glm::vec3& o) {
             return crosses(s.set->bounds(), glm::vec3(inv * glm::vec4(o, 1.f)), glm::mat3(inv) * down);
           }))
          unmade.push_back(s.set);
      }
    if(!unmade.empty())
    {
      const uint64_t    jobId = ++jobs;
      const std::string label = moving.size() == 1 ? "Dropping " + moving[0]->name : "Dropping " + std::to_string(moving.size()) + " props";
      job(jobId, {{"label", label}, {"state", "running"}});
      Json again        = args;
      again["indexed"]  = true;
      if(!again.contains("props") && !again.contains("prop"))
      {
        again["props"] = Json::array();
        for(const auto& m : moving)
          again["props"].push_back(m->id);
      }
      loaders.spawn([this, unmade, again, jobId, label] {
        for(const auto& s : unmade)
          s->get();
        post([this, again, jobId, label] {
          try
          {
            propDrop(again);
            job(jobId, {{"label", label}, {"state", "done"}});
          }
          catch(const std::exception& e)
          {
            job(jobId, {{"label", label}, {"state", "failed"}, {"error", problemOf(e)}});
            notice(problemOf(e), "error");
          }
        });
      });
      return {{"job", jobId}};
    }
  }
  std::optional<float> rest;
  for(const glm::vec3& o : starts)
    if(const auto hit = rayThrough(under, o, down, false))
      rest = std::max(rest.value_or(hit->at.y), hit->at.y);
  const float dy = rest.value_or(groundHeight()) - lo.y;
  if(std::abs(dy) < 1e-5f)
    return {{"ok", true}, {"moved", 0.0}};
  moveBy(moving, glm::vec3(0, dy, 0), moving.size() == 1 ? "Drop " + moving[0]->name : "Drop " + std::to_string(moving.size()) + " props");
  return {{"ok", true}, {"moved", dy}};
}

// Each moves by `delta` in the world (through its parent's space), as one step of the history.
void FvHost::Impl::moveBy(const std::vector<std::shared_ptr<Prop>>& moving, const glm::vec3& delta, const std::string& label)
{
  const auto set = [&](const Prop& p) {
    const glm::vec3 local = glm::inverse(glm::mat3(parentWorld(p))) * delta;
    const glm::vec3 t     = p.transform.translation + local;
    edit("prop.set", {{"prop", p.id}, {"transform", {{"t", {t.x, t.y, t.z}}}}, {"label", label}});
  };
  if(moving.size() == 1)
    return set(*moving[0]);
  journal.begin(label);
  try
  {
    for(const auto& p : moving)
      set(*p);
  }
  catch(...)
  {
    journal.end();
    publishHistory();
    throw;
  }
  journal.end();
  publishHistory();
}

// prop.aim: a light, a camera or anything turned so its -Z looks at a prop's middle or at a point.
Json FvHost::Impl::propAim(const Json& args)
{
  const auto p = findProp(args.at("prop").get<std::string>());
  if(!p)
    throw std::runtime_error("no such prop");
  const Json& at = args.at("at");
  glm::vec3   target;
  std::string what;
  if(at.is_string())
  {
    const auto t = findProp(at.get<std::string>());
    if(!t)
      throw std::runtime_error("no such prop to aim at");
    if(t == p)
      throw std::runtime_error(p->name + " can't aim at itself");
    const auto b = propBounds(*t);
    target       = b ? ((*b)[0] + (*b)[1]) * 0.5f : glm::vec3(worldOf(*t)[3]);
    what         = t->name;
  }
  else if(at.is_array() && at.size() == 3)
  {
    target = {at[0].get<float>(), at[1].get<float>(), at[2].get<float>()};
    what   = "a point";
  }
  else
    throw std::runtime_error("aim at a prop's id or at [x, y, z]");
  const glm::vec3 from = glm::vec3(worldOf(*p)[3]);
  glm::vec3       f    = target - from;
  if(glm::length(f) < 1e-5f)
    throw std::runtime_error(p->name + " stands where it would look");
  f            = glm::normalize(f);
  const glm::vec3 up = std::abs(f.y) > 0.999f ? glm::vec3(0, 0, -1) : glm::vec3(0, 1, 0);
  const glm::quat world = glm::quatLookAt(f, up);  // its -Z along f
  // into the space of what it rides on
  glm::vec3 scale, translation, skew;
  glm::vec4 perspective;
  glm::quat parent;
  glm::decompose(parentWorld(*p), scale, parent, translation, skew, perspective);
  const glm::quat r = glm::normalize(glm::inverse(parent) * world);
  return edit("prop.set", {{"prop", p->id}, {"transform", {{"r", {r.x, r.y, r.z, r.w}}}}, {"label", "Aim " + p->name + " at " + what}});
}

// ---- isolate (view.set's solo)

bool FvHost::Impl::shownNow(const Prop& p) const
{
  if(!p.visible)
    return false;
  bool any = false;
  for(const std::string& id : solo)
    if(const auto s = findProp(id))
    {
      any = true;
      // it, what rides on it, and what carries it (else the cascade would hide it with its carrier)
      if(s.get() == &p || ridesOn(p, id) || ridesOn(*s, p.id))
        return true;
    }
  return !any;
}

void FvHost::Impl::setSolo(const std::vector<std::string>& ids)
{
  solo = {ids.begin(), ids.end()};
  for(const auto& p : props)
    applyVisible(*p);
  publishScene();
}

// ---- a node inside a prop (node.set)

namespace {

// what a DAZ figure's scene graph counts in: centimetres under its root; a model, its file's metres
double unitOf(const Prop& p)
{
  return p.figure ? kCentimetre : 1.0;
}

Json transformJson(const rs::Transform& t, double unit)
{
  const glm::vec3 e = glm::degrees(glm::eulerAngles(t.rotation));
  return {{"t", {t.translation.x * unit, t.translation.y * unit, t.translation.z * unit}},
          {"r", {t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w}},
          {"e", {e.x, e.y, e.z}},
          {"s", {t.scale.x, t.scale.y, t.scale.z}}};
}

glm::vec3 vec3Of(const Json& v)
{
  if(!v.is_array() || v.size() != 3)
    throw std::runtime_error("expected [x, y, z]");
  return {v[0].get<float>(), v[1].get<float>(), v[2].get<float>()};
}

}  // namespace

rs::EntityId FvHost::Impl::nodeEntity(const Prop& p, int node) const
{
  if(p.figure && p.figure->feed && node >= 0 && node < int(p.figure->fig->loaded.scene.nodes.size()))
    return p.figure->feed->entityOf(node);
  if(p.model && node >= 0 && node < int(p.imported.entities.size()))
    return p.imported.entities[size_t(node)];
  return {};
}

std::string FvHost::Impl::nodeName(const Prop& p, int node) const
{
  if(p.figure && node >= 0 && node < int(p.figure->fig->loaded.scene.nodes.size()))
    if(const std::string& name = p.figure->fig->loaded.scene.nodes[size_t(node)].name; !name.empty())
      return name;
  if(p.model && node >= 0 && node < int(p.model->nodes.size()))
    if(const std::string& name = p.model->nodes[size_t(node)].name; !name.empty())
      return name;
  return "node " + std::to_string(node);
}

// What isn't the Scene tab's to change: the figure's root (the prop's own Transform) and what it wears
// (the Wardrobe's). A posable bone and a skinned mesh can be hidden but not moved here (nodeState's
// `moves`): the Pose tab moves the one, and the other goes where its bones are.
std::string FvHost::Impl::nodeRefusal(const Prop& p, int node) const
{
  if(nodeEntity(p, node).isNone())
    return "no such node";
  if(p.figure)
  {
    const fv::Figure&     fig   = *p.figure->fig;
    const fv::SceneGraph& scene = fig.loaded.scene;
    if(node == fig.loaded.root)
      return p.name + "'s root moves with it: use its Transform";
    for(const fv::WornItem& w : fig.items)
    {
      if(w.rel.empty())
        continue;  // the figure's own parts
      for(int at = node; at >= 0; at = scene.nodes[size_t(at)].parent)
        if(std::any_of(w.objects.begin(), w.objects.end(), [&](const auto& o) { return o.first == at; })
           || (scene.nodes[size_t(at)].mesh >= 0 && std::find(w.meshes.begin(), w.meshes.end(), scene.nodes[size_t(at)].mesh) != w.meshes.end()))
          return nodeName(p, node) + " is part of " + w.label + ": the Wardrobe shows and hides it";
    }
  }
  return {};
}

Json FvHost::Impl::nodeState(const Prop& p, int node)
{
  if(!nodeRefusal(p, node).empty() || !renderer->feedsHostScene())
    return Json();
  rs::RetainedScene& scene = renderer->hostScene();
  const rs::Entity*  e     = scene.entity(nodeEntity(p, node));
  if(!e)
    return Json();
  Json out       = transformJson(e->local, unitOf(p));
  out["visible"] = e->visible;
  out["edited"]  = p.nodeEdits.count(node) > 0;
  bool moves     = true;
  if(p.figure)
  {
    const fv::SceneGraph& graph = p.figure->fig->loaded.scene;
    const fv::Node&       n     = graph.nodes[size_t(node)];
    if(p.figure->fig->pose)
      for(const auto& pb : p.figure->fig->pose->bones())
        if(pb.bone == node && !pb.axes.empty())
          moves = false;
    if(n.mesh >= 0 && n.mesh < int(graph.meshes.size()) && graph.meshes[size_t(n.mesh)].skinned())
      moves = false;
  }
  else if(p.model && node < int(p.model->nodes.size()) && p.model->nodes[size_t(node)].skin >= 0)
    moves = false;
  out["moves"] = moves;
  return out;
}

// node.set {prop, node, visible?: bool | null, transform?: {t (metres), e (degrees) | r, s} | null}: a
// node shown or hidden, moved, or (null) back as its file has it.
Json FvHost::Impl::nodeSet(const Json& args)
{
  const auto p = findProp(args.at("prop").get<std::string>());
  if(!p)
    throw std::runtime_error("no such prop");
  if(!p->figure && !p->model)
    throw std::runtime_error(p->name + " has no nodes of its own");
  const int node = args.at("node").get<int>();
  if(const std::string why = nodeRefusal(*p, node); !why.empty())
    throw std::runtime_error(why);
  if(!renderer->feedsHostScene())
    throw std::runtime_error("the scene isn't fed yet");
  const Json         state  = nodeState(*p, node);
  rs::RetainedScene& scene  = renderer->hostScene();
  const rs::Entity*  entity = scene.entity(nodeEntity(*p, node));
  Prop::NodeEdit     edit   = p->nodeEdits.count(node) ? p->nodeEdits[node] : Prop::NodeEdit{};
  // a model's node goes back to what its file had
  if(p->model && !edit.baseLocal)
  {
    edit.baseLocal   = entity->local;
    edit.baseVisible = entity->visible;
  }
  if(args.contains("visible"))
    edit.visible = args["visible"].is_boolean() ? std::optional<bool>(args["visible"].get<bool>()) : std::nullopt;
  if(args.contains("transform"))
  {
    const Json& t = args["transform"];
    if(t.is_null())
      edit.local.reset();
    else
    {
      if(!state.value("moves", false))
        throw std::runtime_error(nodeName(*p, node) + " goes where its bones are: pose it in the Pose tab");
      rs::Transform local = edit.local ? *edit.local : entity->local;
      if(t.contains("t"))
        local.translation = vec3Of(t["t"]) / float(unitOf(*p));
      if(t.contains("r") && t["r"].is_array() && t["r"].size() == 4)
        local.rotation = glm::normalize(glm::quat(t["r"][3].get<float>(), t["r"][0].get<float>(), t["r"][1].get<float>(), t["r"][2].get<float>()));
      else if(t.contains("e"))
        local.rotation = glm::quat(glm::radians(vec3Of(t["e"])));
      if(t.contains("s"))
      {
        local.scale = vec3Of(t["s"]);
        for(int i = 0; i < 3; i++)
          if(std::abs(local.scale[i]) < 1e-4f)
            local.scale[i] = 1e-4f;
      }
      edit.local = local;
    }
  }
  p->nodeEdits[node] = edit;
  applyNodeEdit(*p, node);
  if(!edit.visible && !edit.local)
    p->nodeEdits.erase(node);
  // the eye in the Nodes list, and the selected node's fields
  protocol->events().patch("nodes/" + p->id, {{"nodes", {{std::to_string(node), {{"hidden", edit.visible == std::optional<bool>(false) ? Json(true) : Json()}}}}}});
  if(view)
    view->refreshSelection();
  return {{"ok", true}};
}

void FvHost::Impl::applyNodeEdit(Prop& p, int node)
{
  const auto it = p.nodeEdits.find(node);
  if(p.figure && p.figure->feed)
  {
    const fv::Figure& fig = *p.figure->fig;
    p.figure->feed->setNodeOverride(fig.loaded.scene, fig.loaded.root, node, it != p.nodeEdits.end() ? it->second.local : std::nullopt,
                                    it != p.nodeEdits.end() ? it->second.visible : std::nullopt);
    return;
  }
  if(!p.model || it == p.nodeEdits.end() || !renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  const rs::EntityId e     = nodeEntity(p, node);
  if(!scene.entity(e))
    return;
  const Prop::NodeEdit& edit  = it->second;
  const rs::Transform   local = edit.local ? *edit.local : edit.baseLocal.value_or(scene.entity(e)->local);
  if(!(scene.entity(e)->local == local))
    scene.setLocal(e, local);
  if(const bool shown = edit.visible.value_or(edit.baseVisible); scene.entity(e)->visible != shown)
    scene.setVisible(e, shown);
}

void FvHost::Impl::applyNodeEdits(Prop& p)
{
  for(const auto& [node, edit] : p.nodeEdits)
    applyNodeEdit(p, node);
}

Json FvHost::Impl::nodeEditJson(const Prop& p, int node) const
{
  Json out = Json::object();
  if(const auto it = p.nodeEdits.find(node); it != p.nodeEdits.end())
  {
    if(it->second.visible)
      out["visible"] = *it->second.visible;
    if(it->second.local)
    {
      Json t = transformJson(*it->second.local, unitOf(p));
      t.erase("e");
      out["transform"] = std::move(t);
    }
  }
  return out;
}

Json FvHost::Impl::nodeEditsJson(const Prop& p) const
{
  Json out = Json::object();
  for(const auto& [node, edit] : p.nodeEdits)
    if(Json e = nodeEditJson(p, node); !e.empty())
      out[std::to_string(node)] = std::move(e);
  return out;
}

// A scene file's or a record's node edits put back, each as node.set does; one that can't be now (a
// node its file no longer has) is a notice
void FvHost::Impl::restoreNodeEdits(Prop& p, const Json& nodes)
{
  if(!nodes.is_object())
    return;
  for(auto it = nodes.begin(); it != nodes.end(); ++it)
  {
    Json args = it.value();
    if(!args.is_object())
      continue;
    args["prop"] = p.id;
    args["node"] = std::atoi(it.key().c_str());
    try
    {
      nodeSet(args);
    }
    catch(const std::exception& e)
    {
      notice(p.name + ": " + problemOf(e));
    }
  }
}
