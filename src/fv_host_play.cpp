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
// Play mode: walking about the scene in first person and flying a ship in it (the controllers are
// fv_play.hpp's). The app holds the keys and the mouse (pointer lock) and sends them as `play` input
// events; each frame here steps the controller in charge, moves the ship it flies, opens its canopy,
// folds its gear and lights its engines, and puts the camera where the walker, the pilot or the chase
// camera looks from.
//
// A ship is any glTF model with a node named fv_seat (the pilot's eye). Its other markers are optional:
// fv_board (where one climbs in), fv_exit (where one stands after getting out), fv_com (what it turns
// about), fv_pad_* (what it lands on). Its moving parts are found by name too: `canopy` turns about
// its x axis to open, gear_* about theirs to fold, and *_glow shows while the engines run.
//

#include "fv_host_impl.hpp"

#include <cstring>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "fv_backdrop.hpp"
#include "fv_play.hpp"
#include "fv_world.hpp"

namespace game = fv::play;
using Clock    = std::chrono::steady_clock;

namespace {

constexpr double kBoardReach    = 3.8;   // metres from a ship's boarding point that F gets in
constexpr double kExitSpeed     = 4.0;   // m/s: slower than this to get out ...
constexpr double kExitHeight    = 3.0;   // ... and its pads no higher over the ground
constexpr double kEnterSeconds  = 1.2;
constexpr double kExitSeconds   = 1.0;
constexpr double kCanopyOpen    = -0.9;  // radians about its x axis: the front up
constexpr double kCanopyNear    = 9.0;   // a parked ship opens to someone on foot this close
constexpr double kCanopySeconds = 0.9;   // to open or shut
constexpr double kGearFold      = 1.66;  // radians about their x axes: back under the hull
constexpr double kGearSeconds   = 1.4;
constexpr double kGearDownUnder = 6.0;   // metres over the ground: the gear comes down under this
constexpr double kStepLongest   = 1.0 / 60.0;
constexpr double kLensWalk = 18.0, kLensFly = 20.0;  // focal lengths, mm: a game's wide view
constexpr double kSpawnAhead = 16.0;                 // metres ahead of the camera a ship is put down
constexpr double kNoticeSeconds = 2.5;

// A model's node in the model's frame (its scene's root), its parents' turns and moves with it
glm::dmat4 nodeMatrix(const tinygltf::Model& model, int node)
{
  std::vector<int> parent(model.nodes.size(), -1);
  for(size_t i = 0; i < model.nodes.size(); i++)
    for(const int c : model.nodes[i].children)
      if(c >= 0 && size_t(c) < parent.size())
        parent[size_t(c)] = int(i);
  glm::dmat4 m(1.0);
  for(int at = node, steps = 0; at >= 0 && steps <= int(model.nodes.size()); at = parent[size_t(at)], steps++)
  {
    const tinygltf::Node& n = model.nodes[size_t(at)];
    glm::dmat4            local(1.0);
    if(n.matrix.size() == 16)
      local = glm::make_mat4(n.matrix.data());
    else
    {
      if(n.translation.size() == 3)
        local = glm::translate(local, glm::dvec3(n.translation[0], n.translation[1], n.translation[2]));
      if(n.rotation.size() == 4)
        local *= glm::mat4_cast(glm::dquat(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]));
      if(n.scale.size() == 3)
        local = glm::scale(local, glm::dvec3(n.scale[0], n.scale[1], n.scale[2]));
    }
    m = local * m;
  }
  return m;
}

rs::Transform nodeLocal(const tinygltf::Node& n)
{
  rs::Transform t;
  if(n.translation.size() == 3)
    t.translation = glm::vec3(float(n.translation[0]), float(n.translation[1]), float(n.translation[2]));
  if(n.rotation.size() == 4)
    t.rotation = glm::quat(float(n.rotation[3]), float(n.rotation[0]), float(n.rotation[1]), float(n.rotation[2]));
  if(n.scale.size() == 3)
    t.scale = glm::vec3(float(n.scale[0]), float(n.scale[1]), float(n.scale[2]));
  return t;
}

bool startsWith(const std::string& s, const char* prefix)
{
  return s.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string& s, const char* suffix)
{
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Its uniform scale in the world (a ship scaled by the panel is that much bigger)
double scaleOf(const glm::dmat4& world)
{
  return std::max(1e-6, glm::length(glm::dvec3(world[0])));
}

template <class F>
void substeps(double dt, F&& f)
{
  const int n = std::max(1, int(std::ceil(dt / kStepLongest)));
  for(int i = 0; i < n; i++)
    f(dt / n);
}

double toward(double value, double target, double rate, double dt)
{
  return value < target ? std::min(target, value + rate * dt) : std::max(target, value - rate * dt);
}

}  // namespace

struct FvHost::Impl::PlayState
{
  enum class Mode
  {
    eWalk,
    eEnter,
    eFly,
    eExit
  };
  // A ship among the props: its markers and moving parts, from its file
  struct Rig
  {
    const tinygltf::Model*       model = nullptr;
    glm::dvec3                   seat{0.0}, board{0.0}, exit{0.0}, com{0.0};  // the model's frame
    game::ShipShape              shape;                                      // about its centre, unscaled
    double                       length = 10.0;
    int                          canopy = -1;
    std::vector<int>             gear, glow;
    std::map<int, rs::Transform> base;  // the moving parts as the file has them
    double                       canopyAngle = 0.0, gearAngle = 0.0;
    bool                         engines  = false;
    bool                         settling = false;  // nobody aboard and not at rest yet: it comes down and stops
    game::Ship                   loose;             // ... as it does
  };

  bool               active = false;
  Mode               mode   = Mode::eWalk;
  game::Walker       walker;
  game::Ship         ship;
  game::Chase        chase;
  bool               cockpit = false;
  std::string        shipId;  // the ship flown, or being got into or out of
  double             transition = 0.0;
  game::Pose         from;  // where getting in or out began
  game::Pose         view;  // the camera last frame
  uint32_t           keys = 0;
  int                uses = 0, views = 0, jumps = 0;
  glm::dvec2         look{0.0};
  Clock::time_point  last{}, heard{};
  std::map<std::string, Rig> rigs;
  // models read and found to be no ship, while they live (a freed one's address may come back as a ship)
  std::map<const tinygltf::Model*, std::weak_ptr<const tinygltf::Model>> plain;
  std::map<std::string, fv::composer::Trs> startedAt;  // the ships' places before play moved them
  FvView::Lens       lens;                             // the view's before play
  glm::dvec2         clip{0.0};
  std::string        notice;
  Clock::time_point  noticeUntil{};
  Json               published;
  Clock::time_point  publishedAt{};
  std::set<const fv::physics::SharedTriangles*> indexing;

  // What play asks of the host, as static members: they name the host's own types
  static std::optional<Rig>             rigOf(const tinygltf::Model& model);
  static glm::dvec3                     upAt(const Impl& s, const glm::dvec3& p);
  static std::optional<game::Surface>   groundUnder(Impl& s, const std::vector<PlacedSet>& sets, const glm::dvec3& from,
                                                    const glm::dvec3& up, double reach);
  static std::vector<PlacedSet>         playSets(Impl& s, const std::string& skip);
  static game::World                    playWorld(Impl& s, const std::vector<PlacedSet>& sets);
  static game::Ship                     shipOf(const Impl& s, const Prop& p, const Rig& rig);
  static void                           putShip(Impl& s, Prop& p, const Rig& rig, const game::Ship& ship);
  static game::ShipShape                shapeIn(const Rig& rig, double size);
  static void                           poseRig(Impl& s, const Prop& p, const Rig& rig);
  static void                           findRigs(Impl& s, PlayState& P);
};

// A ship's markers and parts, if the model is one
std::optional<FvHost::Impl::PlayState::Rig> FvHost::Impl::PlayState::rigOf(const tinygltf::Model& model)
{
  if(std::none_of(model.nodes.begin(), model.nodes.end(), [](const tinygltf::Node& n) { return n.name == "fv_seat"; }))
    return std::nullopt;
  Rig rig;
  rig.model     = &model;
  bool hasBoard = false, hasExit = false, hasCom = false;
  std::vector<glm::dvec3> pads;
  for(size_t i = 0; i < model.nodes.size(); i++)
  {
    const std::string& name = model.nodes[i].name;
    const auto         at   = [&] { return glm::dvec3(nodeMatrix(model, int(i))[3]); };
    if(name == "fv_seat")
      rig.seat = at();
    else if(name == "fv_board")
      hasBoard = true, rig.board = at();
    else if(name == "fv_exit")
      hasExit = true, rig.exit = at();
    else if(name == "fv_com")
      hasCom = true, rig.com = at();
    else if(startsWith(name, "fv_pad"))
      pads.push_back(at());
    else if(name == "canopy")
      rig.canopy = int(i);
    else if(startsWith(name, "gear_") && model.nodes[i].mesh < 0)
      rig.gear.push_back(int(i));
    else if(endsWith(name, "_glow"))
      rig.glow.push_back(int(i));
  }
  for(const int n : rig.gear)
    rig.base[n] = nodeLocal(model.nodes[size_t(n)]);
  if(rig.canopy >= 0)
    rig.base[rig.canopy] = nodeLocal(model.nodes[size_t(rig.canopy)]);
  // its size from what its meshes take up
  glm::dvec3 lo(1e30), hi(-1e30);
  for(size_t i = 0; i < model.nodes.size(); i++)
  {
    const int mesh = model.nodes[i].mesh;
    if(mesh < 0 || size_t(mesh) >= model.meshes.size())
      continue;
    const glm::dmat4 m = nodeMatrix(model, int(i));
    for(const tinygltf::Primitive& prim : model.meshes[size_t(mesh)].primitives)
    {
      const auto it = prim.attributes.find("POSITION");
      if(it == prim.attributes.end() || it->second < 0 || size_t(it->second) >= model.accessors.size())
        continue;
      const tinygltf::Accessor& a = model.accessors[size_t(it->second)];
      if(a.minValues.size() < 3 || a.maxValues.size() < 3)
        continue;
      for(int c = 0; c < 8; c++)
      {
        const glm::dvec3 p(c & 1 ? a.maxValues[0] : a.minValues[0], c & 2 ? a.maxValues[1] : a.minValues[1], c & 4 ? a.maxValues[2] : a.minValues[2]);
        const glm::dvec3 w = glm::dvec3(m * glm::dvec4(p, 1.0));
        lo                 = glm::min(lo, w);
        hi                 = glm::max(hi, w);
      }
    }
  }
  if(lo.x > hi.x)
    lo = hi = rig.seat;
  const glm::dvec3 mid = (lo + hi) * 0.5;
  if(!hasCom)
    rig.com = mid;
  if(!hasBoard)
    rig.board = {hi.x, rig.seat.y - 0.3, rig.seat.z};
  if(!hasExit)
    rig.exit = {hi.x + 0.8, lo.y, rig.seat.z};
  if(pads.empty())  // none marked: the corners of its underside
    pads = {{lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {mid.x, lo.y, hi.z}};
  rig.length = std::max(hi.z - lo.z, 1.0);
  for(const glm::dvec3& p : pads)
    rig.shape.pads.push_back(p - rig.com);
  for(const glm::dvec3& p : {glm::dvec3(mid.x, mid.y, hi.z), glm::dvec3(mid.x, mid.y, lo.z), glm::dvec3(hi.x, rig.com.y, mid.z),
                             glm::dvec3(lo.x, rig.com.y, mid.z), glm::dvec3(mid.x, hi.y, rig.seat.z)})
    rig.shape.hull.push_back(p - rig.com);
  rig.shape.radius = 0.5 * glm::length(hi - lo);
  return rig;
}


bool FvHost::Impl::playing() const
{
  return play && play->active;
}

// Which way is up at a point: the scene's +y, a planet's own vertical away from where the figures stand
glm::dvec3 FvHost::Impl::PlayState::upAt(const Impl& s, const glm::dvec3& p)
{
  if(s.planetBuilt && s.planetBuilt->world && s.backdropStands())
    return fv::world::viewUp(*s.planetBuilt->world, p - glm::dvec3(0.0, double(s.groundHeight()), 0.0));
  return {0.0, 1.0, 0.0};
}

// The highest ground straight under `from` within `reach`: the floor, the backdrop's land or the
// planet's, or a prop's surface
std::optional<game::Surface> FvHost::Impl::PlayState::groundUnder(Impl& s, const std::vector<PlacedSet>& sets, const glm::dvec3& from, const glm::dvec3& up,
                                                    double reach)
{
  std::optional<game::Surface> best;
  double                       bestRise = 0.0;
  const auto                   consider = [&](const glm::dvec3& at, const glm::dvec3& normal) {
    const double rise = glm::dot(at - from, up);  // under `from`: up to 0
    if(rise > 1e-6 || rise < -reach || (best && rise <= bestRise))
      return;
    best     = game::Surface{at, normal};
    bestRise = rise;
  };
  const double floor = double(s.groundHeight());
  if(s.backdropStands())
  {
    if(fv::backdrop::isPlanet(s.backdropSpec))
    {
      if(s.planetBuilt && s.planetBuilt->world)
        consider(from - up * fv::world::heightOver(*s.planetBuilt->world, from - glm::dvec3(0.0, floor, 0.0)), up);
    }
    else
    {
      const auto   at = [&](double x, double z) { return floor + double(fv::backdrop::groundAt(s.backdropSpec, float(x), float(z))); };
      const double e  = 0.5;
      const glm::dvec3 n = glm::normalize(glm::dvec3(at(from.x - e, from.z) - at(from.x + e, from.z), 2.0 * e, at(from.x, from.z - e) - at(from.x, from.z + e)));
      consider({from.x, at(from.x, from.z), from.z}, n);
    }
  }
  else if(s.lighting.ground.mode != "off")
    consider({from.x, floor, from.z}, {0.0, 1.0, 0.0});
  if(const auto hit = Impl::rayThrough(sets, glm::vec3(from), glm::vec3(-up), true))
    consider(glm::dvec3(hit->at), glm::dvec3(hit->normal));
  return best;
}

// What the controllers collide with: every shown prop's surfaces but `skip`'s (a ship doesn't meet
// itself) and people's, whose poses move under them. A set not indexed yet is indexed on the pool and
// met once it is.
std::vector<FvHost::Impl::PlacedSet> FvHost::Impl::PlayState::playSets(Impl& s, const std::string& skip)
{
  std::vector<PlacedSet>                                           out;
  std::vector<std::shared_ptr<const fv::physics::SharedTriangles>> unmade;
  for(const auto& q : s.props)
  {
    if(!s.shown(*q) || q->kind == PropKind::eCharacter || q->kind == PropKind::eCreature
       || (!skip.empty() && (q->id == skip || s.ridesOn(*q, skip))))
      continue;
    for(PlacedSet& set : s.placedSets(*q))
    {
      if(!set.set->made() && s.play->indexing.insert(set.set.get()).second)
        unmade.push_back(set.set);
      out.push_back(std::move(set));
    }
  }
  if(!unmade.empty())
    s.loaders.spawn([unmade] {
      for(const auto& set : unmade)
        set->get();
    });
  return out;
}

game::World FvHost::Impl::PlayState::playWorld(Impl& s, const std::vector<PlacedSet>& sets)
{
  game::World w;
  w.up     = [&s](const glm::dvec3& p) { return upAt(s, p); };
  w.ground = [&s, &sets](const glm::dvec3& from, const glm::dvec3& up, double reach) { return groundUnder(s, sets, from, up, reach); };
  w.ray    = [&sets](const glm::dvec3& from, const glm::dvec3& dir, double reach) -> std::optional<game::Surface> {
    const auto hit = Impl::rayThrough(sets, glm::vec3(from), glm::vec3(dir), true);
    if(!hit || glm::dot(glm::dvec3(hit->at) - from, dir) > reach)
      return std::nullopt;
    return game::Surface{glm::dvec3(hit->at), glm::dvec3(hit->normal)};
  };
  return w;
}

// The ship as the controller holds it, from where its prop stands
game::Ship FvHost::Impl::PlayState::shipOf(const Impl& s, const Prop& p, const Rig& rig)
{
  const glm::dmat4 world = glm::dmat4(s.worldOf(p));
  glm::dvec3       scale, t, skew;
  glm::dquat       r;
  glm::dvec4       perspective;
  glm::decompose(world, scale, r, t, skew, perspective);
  game::Ship ship;
  ship.orientation = glm::normalize(r);
  ship.com         = glm::dvec3(world * glm::dvec4(rig.com, 1.0));
  return ship;
}

// Its prop where the controller has it: no step of the history (play.stop makes one)
void FvHost::Impl::PlayState::putShip(Impl& s, Prop& p, const Rig& rig, const game::Ship& ship)
{
  const double     size   = scaleOf(glm::dmat4(s.worldOf(p)));
  const glm::dmat3 turn   = glm::mat3_cast(ship.orientation);
  const glm::dvec3 origin = ship.com - turn * (rig.com * size);
  const glm::dmat4 world  = glm::translate(glm::dmat4(1.0), origin) * glm::dmat4(turn) * glm::scale(glm::dmat4(1.0), glm::dvec3(size));
  const glm::dmat4 local  = glm::inverse(glm::dmat4(s.parentWorld(p))) * world;
  glm::dvec3       scale, t, skew;
  glm::dquat       r;
  glm::dvec4       perspective;
  glm::decompose(local, scale, r, t, skew, perspective);
  s.play->startedAt.try_emplace(p.id, p.transform);
  p.transform = {glm::vec3(t), glm::quat(glm::normalize(r)), glm::vec3(scale)};
  p.euler.reset();
  s.applyTransform(p);
  s.movedProps.insert(p.id);
}

// Its shape in the world's metres, about its centre
game::ShipShape FvHost::Impl::PlayState::shapeIn(const Rig& rig, double size)
{
  game::ShipShape shape = rig.shape;
  for(glm::dvec3& p : shape.pads)
    p *= size;
  for(glm::dvec3& p : shape.hull)
    p *= size;
  shape.radius *= size;
  return shape;
}

// Its canopy, gear and engines as the rig has them now
void FvHost::Impl::PlayState::poseRig(Impl& s, const Prop& p, const Rig& rig)
{
  rs::RetainedScene& scene = s.renderer->hostScene();
  const auto         turn  = [&](int node, double angle) {
    const rs::EntityId e = s.nodeEntity(p, node);
    const rs::Entity*  entity = scene.entity(e);
    const auto         base   = rig.base.find(node);
    if(!entity || base == rig.base.end())
      return;
    rs::Transform local = base->second;
    local.rotation      = local.rotation * glm::angleAxis(float(angle), glm::vec3(1.f, 0.f, 0.f));
    if(!(entity->local == local))
      scene.setLocal(e, local);
  };
  if(rig.canopy >= 0)
    turn(rig.canopy, rig.canopyAngle);
  for(const int g : rig.gear)
    turn(g, rig.gearAngle);
  for(const int g : rig.glow)
  {
    const rs::EntityId e = s.nodeEntity(p, g);
    if(const rs::Entity* entity = scene.entity(e); entity && entity->visible != rig.engines)
      scene.setVisible(e, rig.engines);
  }
}

// Ships come and go with the props: a model with a seat is one
void FvHost::Impl::PlayState::findRigs(Impl& s, PlayState& P)
{
  for(auto it = P.rigs.begin(); it != P.rigs.end();)
  {
    const auto p = s.findProp(it->first);
    it           = p && p->model.get() == it->second.model ? std::next(it) : P.rigs.erase(it);
  }
  for(const auto& p : s.props)
  {
    if(!p->model || P.rigs.count(p->id))
      continue;
    if(const auto it = P.plain.find(p->model.get()); it != P.plain.end() && !it->second.expired())
      continue;  // read once: a model is no ship the next frame either
    if(auto rig = rigOf(*p->model))
      P.rigs.emplace(p->id, std::move(*rig));
    else
      P.plain[p->model.get()] = p->model;
  }
}

Json FvHost::Impl::playStart(const Json& args)
{
  auto manip = renderer->getCameraManipulator();
  if(!manip || !renderer->feedsHostScene())
    throw std::runtime_error("the view isn't ready yet");
  if(!play)
    play = std::make_shared<PlayState>();
  PlayState& P = *play;
  if(!viewCamera.empty())
    viewUse({{"camera", nullptr}});  // play moves the free view
  if(!P.active)
  {
    P        = PlayState{};
    P.active = true;
    P.heard  = Clock::now();
    P.clip   = manip->getClipPlanes();
    if(view)
    {
      P.lens               = view->lens();
      FvView::Lens wide    = P.lens;
      wide.focalLength     = kLensWalk;
      wide.dof             = false;
      view->setLens(wide);
    }
    glm::dvec3 eye, center, up;
    manip->getLookat(eye, center, up);
    const auto sets  = PlayState::playSets(*this, "");
    const auto world = PlayState::playWorld(*this, sets);
    const glm::dvec3 ahead = glm::length(center - eye) > 1e-9 ? glm::normalize(center - eye) : glm::dvec3(0.0, 0.0, -1.0);
    P.walker               = game::walkerAt({eye, ahead, up}, world, 2000.0);
    P.view                 = game::walkerView(P.walker, world);
  }
  PlayState::findRigs(*this, P);
  // a ship to fly, when the scene has none: down on the ground ahead, its side (the way in) turned to
  // the camera
  const std::string path = args.value("ship", std::string());
  if(P.rigs.empty() && !path.empty())
  {
    const auto       sets  = PlayState::playSets(*this, "");
    const glm::dvec3 eye   = P.view.eye;
    const glm::dvec3 up    = PlayState::upAt(*this, eye);
    glm::dvec3       ahead = P.view.forward - up * glm::dot(P.view.forward, up);
    ahead                  = glm::length(ahead) > 1e-6 ? glm::normalize(ahead) : glm::dvec3(0.0, 0.0, -1.0);
    glm::dvec3 at          = eye + ahead * kSpawnAhead;
    if(const auto g = PlayState::groundUnder(*this, sets, at + up * 50.0, up, 2000.0))
      at = g->at;
    else
      at -= up * game::WalkerTuning{}.eyeHeight;
    const glm::dvec3 x = -ahead, z = glm::cross(x, up);
    const glm::quat  q = glm::quat_cast(glm::mat3(glm::dmat3(x, up, z)));
    importModel(path,
                [this](std::shared_ptr<Prop> p) {
                  recordAdded(p, "Add " + p->name);
                  if(view)
                    view->deselect(p->id);
                  renderer->selectEntity(rs::EntityId{});
                },
                {}, {{"transform", {{"t", {at.x, at.y, at.z}}, {"r", {q.x, q.y, q.z, q.w}}}}});
  }
  publishPlay(true);
  return {{"ok", true}};
}

Json FvHost::Impl::playStop()
{
  if(!playing())
    return {{"ok", true}};
  PlayState& P = *play;
  P.active     = false;
  // the ships stay where they were left, a step of the history each, and their parts as their files have them
  for(auto& [id, rig] : P.rigs)
    if(const auto p = findProp(id))
    {
      rig.canopyAngle = rig.gearAngle = 0.0;
      rig.engines                     = false;
      PlayState::poseRig(*this, *p, rig);
    }
  for(const auto& [id, before] : P.startedAt)
    if(const auto p = findProp(id))
    {
      const fv::composer::Trs now = p->transform;
      p->transform                = before;
      applyTransform(*p);
      try
      {
        edit("prop.set", {{"prop", id},
                          {"transform",
                           {{"t", {now.translation.x, now.translation.y, now.translation.z}},
                            {"r", {now.rotation.x, now.rotation.y, now.rotation.z, now.rotation.w}},
                            {"s", {now.scale.x, now.scale.y, now.scale.z}}}},
                          {"label", "Fly " + p->name}});
      }
      catch(const std::exception& e)
      {
        notice(p->name + ": " + problemOf(e));
      }
    }
  P.startedAt.clear();
  if(view)
    view->setLens(P.lens);
  if(auto manip = renderer->getCameraManipulator())
  {
    const glm::dvec2 clip = manip->getClipPlanes();
    manip->setClipPlanes({P.clip.x > 0 ? P.clip.x : clip.x, std::max(clip.y, P.clip.y)});
    // the view stays where play left it, upright and looking a few metres ahead to orbit about
    const glm::dvec3 up = PlayState::upAt(*this, P.view.eye);
    manip->setLookat(P.view.eye, P.view.eye + P.view.forward * 3.0, up, true);
  }
  publishPlay(true);
  return {{"ok", true}};
}

void FvHost::Impl::playInput(const Json& e)
{
  if(!playing())
    return;
  PlayState&     P       = *play;
  const uint32_t keys    = e.value("k", uint32_t(0));
  const uint32_t pressed = keys & ~P.keys;
  P.keys                 = keys;
  P.heard                = Clock::now();
  P.uses += (pressed & game::kUse) ? 1 : 0;
  P.views += (pressed & game::kView) ? 1 : 0;
  P.jumps += (pressed & game::kUp) ? 1 : 0;
  P.look += glm::dvec2(e.value("dx", 0.0), e.value("dy", 0.0));
  if(const double wheel = e.value("wheel", 0.0); wheel != 0.0)
    P.chase.zoom = std::clamp(P.chase.zoom * std::exp(wheel * 0.001), 0.5, 3.0);
}

void FvHost::Impl::applyPlay()
{
  if(!playing() || !renderer->feedsHostScene())
    return;
  auto manip = renderer->getCameraManipulator();
  if(!manip)
    return;
  PlayState&   P   = *play;
  const auto   now = Clock::now();
  const double dt  = P.last == Clock::time_point{} ? 0.0 : std::min(std::chrono::duration<double>(now - P.last).count(), 0.1);
  P.last           = now;
  // the app sends the keys each of its frames: silence is a release that got lost
  if(now - P.heard > std::chrono::milliseconds(500))
    P.keys = 0;
  const int        uses = std::exchange(P.uses, 0), views = std::exchange(P.views, 0), jumps = std::exchange(P.jumps, 0);
  const glm::dvec2 look = std::exchange(P.look, glm::dvec2(0.0));

  PlayState::findRigs(*this, P);
  const auto       sets  = PlayState::playSets(*this, "");
  const game::World world = PlayState::playWorld(*this, sets);
  std::shared_ptr<Prop> flown = P.shipId.empty() ? nullptr : findProp(P.shipId);
  PlayState::Rig*       rig   = flown && P.rigs.count(P.shipId) ? &P.rigs.at(P.shipId) : nullptr;
  if(P.mode != PlayState::Mode::eWalk && !rig)
  {
    // the ship went (an undo, a delete): on foot where the camera is
    P.walker = game::walkerAt(P.view, world, 2000.0);
    P.mode   = PlayState::Mode::eWalk;
    P.shipId.clear();
  }
  const double size     = flown ? scaleOf(glm::dmat4(worldOf(*flown))) : 1.0;
  std::string  prompt;
  const auto   seatPose = [&] { return game::cockpitView(P.ship, (rig->seat - rig->com) * size); };
  const auto   say      = [&](const std::string& text) {
    P.notice      = text;
    P.noticeUntil = now + std::chrono::milliseconds(int(kNoticeSeconds * 1000));
  };

  switch(P.mode)
  {
    case PlayState::Mode::eWalk: {
      P.walker.keys = P.keys;
      P.walker.look += look;
      P.walker.jumps += jumps;
      substeps(dt, [&](double h) { game::stepWalker(P.walker, world, h); });
      P.view = game::walkerView(P.walker, world);
      // the nearest ship whose way in is within reach
      std::string closest;
      double      nearest = kBoardReach;
      for(const auto& [id, r] : P.rigs)
        if(const auto p = findProp(id); p && shown(*p))
          if(const double d = glm::distance(P.view.eye, glm::dvec3(glm::dmat4(worldOf(*p)) * glm::dvec4(r.board, 1.0))); d < nearest)
          {
            nearest = d;
            closest = id;
          }
      if(!closest.empty())
        prompt = "F  Get in";
      if(uses > 0 && !closest.empty())
      {
        P.shipId     = closest;
        flown        = findProp(closest);
        rig          = &P.rigs.at(closest);
        P.ship       = PlayState::shipOf(*this, *flown, *rig);
        P.chase      = {.zoom = P.chase.zoom};
        P.mode       = PlayState::Mode::eEnter;
        P.transition = 0.0;
        P.from       = P.view;
        rig->settling = false;
      }
      break;
    }
    case PlayState::Mode::eEnter: {
      P.transition += dt;
      P.view = game::blend(P.from, game::cockpitView(P.ship, (rig->seat - rig->com) * scaleOf(glm::dmat4(worldOf(*flown)))), P.transition / kEnterSeconds);
      if(P.transition >= kEnterSeconds)
      {
        P.mode         = PlayState::Mode::eFly;
        P.ship.throttle = 0.0;
        P.ship.aim      = glm::dvec2(0.0);
        rig->engines    = true;
        if(view)
        {
          FvView::Lens wide = view->lens();
          wide.focalLength  = kLensFly;
          view->setLens(wide);
        }
      }
      break;
    }
    case PlayState::Mode::eFly: {
      if(views > 0)
        P.cockpit = !P.cockpit;
      P.ship.keys = P.keys;
      P.ship.aim += look;
      const auto            own   = PlayState::playSets(*this, P.shipId);
      const game::World     its   = PlayState::playWorld(*this, own);
      const game::ShipShape shape = PlayState::shapeIn(*rig, size);
      substeps(dt, [&](double h) { game::stepShip(P.ship, shape, its, h, true); });
      PlayState::putShip(*this, *flown, *rig, P.ship);
      P.view = P.cockpit ? seatPose() : game::chaseView(P.chase, P.ship, rig->length * size, dt);
      const bool canLeave = glm::length(P.ship.velocity) < kExitSpeed && P.ship.height < kExitHeight;
      if(canLeave)
        prompt = "F  Get out";
      if(uses > 0)
      {
        if(!canLeave)
          say("Slow down and come down low to get out");
        else
        {
          // out on its boarding side, standing on what is there, facing it
          const glm::dmat4 placed = glm::dmat4(worldOf(*flown));
          const glm::dvec3 out    = glm::dvec3(placed * glm::dvec4(rig->exit, 1.0));
          const glm::dvec3 up     = PlayState::upAt(*this, out);
          glm::dvec3       facing = P.ship.com - out;
          facing -= up * glm::dot(facing, up);
          const game::WalkerTuning t;
          P.walker = game::walkerAt({out + up * (t.eyeHeight + 0.5), glm::length(facing) > 1e-6 ? glm::normalize(facing) : glm::dvec3(0.0, 0.0, -1.0), up}, world, 20.0);
          P.from       = P.view;
          P.transition = 0.0;
          P.mode       = PlayState::Mode::eExit;
          rig->engines = false;
          rig->settling = true;
          rig->loose    = P.ship;
          rig->loose.throttle = 0.0;
        }
      }
      break;
    }
    case PlayState::Mode::eExit: {
      P.transition += dt;
      P.view = game::blend(P.from, game::walkerView(P.walker, world), P.transition / kExitSeconds);
      if(P.transition >= kExitSeconds)
      {
        P.mode = PlayState::Mode::eWalk;
        P.shipId.clear();
        if(view)
        {
          FvView::Lens wide = view->lens();
          wide.focalLength  = kLensWalk;
          view->setLens(wide);
        }
      }
      break;
    }
  }

  // ships nobody flies: one left in the air comes down onto its pads and stops; then it stands
  for(auto& [id, r] : P.rigs)
  {
    const auto p = findProp(id);
    if(!p)
      continue;
    const bool flying = id == P.shipId && (P.mode == PlayState::Mode::eFly || P.mode == PlayState::Mode::eEnter);
    if(r.settling && !flying)
    {
      const double          sz    = scaleOf(glm::dmat4(worldOf(*p)));
      const auto            own   = PlayState::playSets(*this, id);
      const game::World     its   = PlayState::playWorld(*this, own);
      const game::ShipShape shape = PlayState::shapeIn(r, sz);
      substeps(dt, [&](double h) { game::stepShip(r.loose, shape, its, h, false); });
      PlayState::putShip(*this, *p, r, r.loose);
      if(r.loose.touching && glm::length(r.loose.velocity) < 0.02)
        r.settling = false;
    }
    // its parts: the gear down near the ground, the canopy open to someone close or getting in or out
    const bool   inside = id == P.shipId;
    const double height = flying ? P.ship.height : 0.0;
    const bool   gearUp = flying && P.mode == PlayState::Mode::eFly && height > kGearDownUnder;
    bool         open   = false;
    if(inside)
      open = (P.mode == PlayState::Mode::eEnter && P.transition < 0.75 * kEnterSeconds) || P.mode == PlayState::Mode::eExit;
    else if(P.mode == PlayState::Mode::eWalk)
      open = glm::distance(P.view.eye, glm::dvec3(glm::dmat4(worldOf(*p)) * glm::dvec4(r.board, 1.0))) < kCanopyNear;
    r.canopyAngle = toward(r.canopyAngle, open ? kCanopyOpen : 0.0, std::abs(kCanopyOpen) / kCanopySeconds, dt);
    r.gearAngle   = toward(r.gearAngle, gearUp ? kGearFold : 0.0, kGearFold / kGearSeconds, dt);
    PlayState::poseRig(*this, *p, r);
  }

  // the camera: a near plane close enough for the cockpit, the far one the scene's
  const glm::dvec2 clip = manip->getClipPlanes();
  if(clip.x > 0.03)
    manip->setClipPlanes({0.02, clip.y});
  const double ahead = P.mode == PlayState::Mode::eWalk ? 10.0 : 30.0;
  manip->setLookat(P.view.eye, P.view.eye + P.view.forward * ahead, P.view.up, true);
  if(now < P.noticeUntil)
    prompt = P.notice;

  const bool  fly   = P.mode == PlayState::Mode::eFly;
  const char* modes[] = {"walk", "enter", "fly", "exit"};
  Json        state = {{"active", true},
                       {"mode", modes[int(P.mode)]},
                       {"prompt", prompt},
                       {"ship", P.shipId.empty() ? Json() : Json(P.shipId)},
                       {"ships", P.rigs.size()},
                       {"view", P.cockpit ? "cockpit" : "chase"},
                       {"speed", fly ? std::round(glm::length(P.ship.velocity) * 10.0) / 10.0 : 0.0},
                       {"height", fly ? std::round(P.ship.height) : 0.0},
                       {"throttle", fly ? std::round(P.ship.throttle / game::ShipTuning{}.cruise * 100.0) / 100.0 : 0.0},
                       {"boost", fly && (P.ship.keys & game::kFast) != 0}};
  if(state != P.published && now - P.publishedAt > std::chrono::milliseconds(100))
  {
    P.published   = state;
    P.publishedAt = now;
    protocol->events().snapshot("play", state);
  }
}

void FvHost::Impl::publishPlay(bool force)
{
  if(playing() && !force)
    return;
  Json state = playing() ? Json{{"active", true}, {"mode", "walk"}, {"prompt", ""}, {"ships", play->rigs.size()}} : Json{{"active", false}};
  if(play)
  {
    play->published   = state;
    play->publishedAt = Clock::now();
  }
  protocol->events().snapshot("play", state);
}
