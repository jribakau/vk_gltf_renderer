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
// camera looks from. On foot the torch (L) is a spot light the host keeps in the walker's hand, aimed
// at what the view looks at, and B puts a ship down ahead: the nearest in the scene, else the app's
// own (play.start's `ship`). While playing, the view's exposure adapts as the walker's eye does
// (HostRenderer::setPlayEye), so a night is seen by the moon and the torch's pool by its light; X (or
// play.start's `eye` false) leaves the exposure as the Camera tab has it. What the
// walker and the ship meet are the props' triangles, the planet's places, and the backdrop's trees
// (their trunks, found from each tree's mesh) and stones (their own triangles) near them.
//
// A ship is any glTF model with a node named fv_seat (the pilot's eye). Its other markers are optional:
// fv_board (where one climbs in), fv_exit (where one stands after getting out), fv_com (what it turns
// about), fv_pad_* (what it lands on). Its moving parts are found by name too: `canopy` turns about
// its x axis to open, gear_* about theirs to fold, and *_glow shows while the engines run.
//
// A car is one with fv_seat (the driver's eye) and four mesh-less pivots wheel_fl, wheel_fr, wheel_rl and
// wheel_rr at its wheels' middles, each wheel's meshes under its pivot and its axle along the pivot's x;
// a steering_wheel pivot turns about its z, and fv_board, fv_exit and fv_com are as a ship's. N puts one
// down ahead: the nearest in the scene, else the next of the app's (play.start's `cars`). It is driven
// in first person from its seat or with the chase camera behind it (V), and one nobody drives settles
// onto its wheels and sleeps.
//

#include "fv_host_impl.hpp"

#include <cstring>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "fv_backdrop.hpp"
#include "fv_play.hpp"
#include "fv_world.hpp"
#include "fv_world_plants.hpp"

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
constexpr double kCarAhead   = 8.0;                  // and a car
constexpr double kStopSpeed  = 2.0;                  // m/s: slower than this a car can be got out of
constexpr double kSteeringRatio = 15.0;              // the steering wheel's turn for the road wheels'
constexpr double kNoticeSeconds = 2.5;
constexpr double kSolidsNear    = 14.0;  // metres round the walker that the backdrop's trees and stones are met
constexpr double kShipWaits     = 4.0;   // seconds B waits for a ship it asked to load before asking again
// The torch: a bright LED torch's beam, some 11,000 cd in its middle (a spot light's units: 20 lights
// a figure 2.5 m away about as the built-in studio does), a little warmer than white, from a reflector
// a couple of centimetres across
constexpr double kTorchStrength = 0.6;
constexpr double kTorchInner    = 0.12;  // radians from its axis: full strength inside
constexpr double kTorchOuter    = 0.45;  // and none outside
constexpr double kTorchRadius   = 0.015;
constexpr double kTorchSight    = 40.0;  // metres ahead the beam is aimed at what the view looks at, at most

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

// Each node's parent (-1: a root)
std::vector<int> parentsOf(const tinygltf::Model& model)
{
  std::vector<int> parent(model.nodes.size(), -1);
  for(size_t i = 0; i < model.nodes.size(); i++)
    for(const int c : model.nodes[i].children)
      if(c >= 0 && size_t(c) < parent.size())
        parent[size_t(c)] = int(i);
  return parent;
}

// What a node's mesh takes up in the model's frame, grown into `lo` and `hi` (its accessors' boxes)
void growByMesh(const tinygltf::Model& model, int node, glm::dvec3& lo, glm::dvec3& hi)
{
  const int mesh = model.nodes[size_t(node)].mesh;
  if(mesh < 0 || size_t(mesh) >= model.meshes.size())
    return;
  const glm::dmat4 m = nodeMatrix(model, node);
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

// The torch as a light: a spot shining along its -Z
tinygltf::Light torchPayload()
{
  tinygltf::Light light;
  light.name                = "Torch";
  light.type                = "spot";
  light.color               = {1.0, 0.96, 0.9};
  light.intensity           = kTorchStrength;
  light.spot.innerConeAngle = kTorchInner;
  light.spot.outerConeAngle = kTorchOuter;
  tinygltf::Value::Object extras;
  extras["radius"] = tinygltf::Value(kTorchRadius);
  light.extras     = tinygltf::Value(extras);
  return light;
}

}  // namespace

struct FvHost::Impl::PlayState
{
  enum class Mode
  {
    eWalk,
    eEnter,
    eFly,
    eExit,
    eDrive
  };
  // A ship or a car among the props: its markers and moving parts, from its file
  struct Rig
  {
    enum class Kind
    {
      eShip,
      eCar
    };
    Kind                         kind  = Kind::eShip;
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
    // a car's: its wheels' pivots (front left, front right, rear left, rear right) and its steering wheel's;
    // each pivot's parent's frame's turn of the car's up and down (a model unit's), and which way its x
    // runs along the car's left (a wheel rolls forward about the car's left)
    std::array<int, 4>        wheels{-1, -1, -1, -1};
    int                       steeringWheel = -1;
    std::array<glm::dvec3, 4> upInParent{}, downInParent{};
    std::array<float, 4>      spinSign{1.f, 1.f, 1.f, 1.f};
    game::CarShape            car;              // about its centre, unscaled
    game::Car                 drive;            // where it is and how it goes, the world's
    bool                      placed = false;   // `drive` holds it (else it is read from the prop when needed)
  };

  bool               active = false;
  Mode               mode   = Mode::eWalk;
  game::Walker       walker;
  game::Ship         ship;
  game::Chase        chase;
  bool               cockpit = false;
  std::string        shipId;  // the ship flown, or being got into or out of
  double             transition = 0.0;
  double             range      = 0.0;  // what full throttle is now, m/s (the HUD's)
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
  std::string        shipPath;     // the GLB B puts down when the scene has no ship (play.start's `ship`)
  Clock::time_point  shipAsked{};  // when B last asked for it to load
  std::vector<std::string> carPaths;  // the GLBs N puts down in turn when the scene has no car (play.start's `cars`)
  size_t             carNext = 0;
  Clock::time_point  carAsked{};
  int                torches = 0, brings = 0, eyes = 0, carBrings = 0;  // presses of L, B, X and N not yet taken
  bool               eye = true;  // the view's exposure adapts as the walker's eye does (X; play.start's `eye`)
  game::Torch        torch;
  rs::EntityId       torchEntity;
  rs::LightId        torchLight;
  bool               torchShown = false;
  rs::Transform      torchAt;

  // What play asks of the host, as static members: they name the host's own types
  static std::optional<Rig>             rigOf(const tinygltf::Model& model);
  static glm::dvec3                     upAt(const Impl& s, const glm::dvec3& p);
  static std::optional<game::Surface>   groundUnder(Impl& s, const std::vector<PlacedSet>& sets, const glm::dvec3& from,
                                                    const glm::dvec3& up, double reach, bool slopes = false);
  static std::vector<PlacedSet>         playSets(Impl& s, const std::string& skip, const glm::dvec3& around, double within = kSolidsNear);
  static void                           plantSets(Impl& s, const glm::dvec3& around, double within, std::vector<PlacedSet>& out);
  static std::string                    bringShip(Impl& s, PlayState& P, const Clock::time_point& now);
  static std::string                    bringCar(Impl& s, PlayState& P, const Clock::time_point& now);
  static void                           showTorch(Impl& s, PlayState& P, bool on, const game::World& world, double dt);
  static void                           dropTorch(Impl& s, PlayState& P);
  static game::World                    playWorld(Impl& s, const std::vector<PlacedSet>& sets, bool slopes = false);
  static game::Ship                     shipOf(const Impl& s, const Prop& p, const Rig& rig);
  static void                           putRig(Impl& s, Prop& p, const Rig& rig, const glm::dvec3& com, const glm::dquat& orientation);
  static game::ShipShape                shapeIn(const Rig& rig, double size);
  static game::CarShape                 carShapeIn(const Rig& rig, double size);
  static void                           placeCar(const Impl& s, const Prop& p, Rig& rig);
  static void                           stepLooseCar(Impl& s, Prop& p, Rig& rig, double dt);
  static void                           poseRig(Impl& s, const Prop& p, const Rig& rig);
  static void                           findRigs(Impl& s, PlayState& P);
  static size_t                         countOf(const PlayState& P, Rig::Kind kind);
};

// A ship's or a car's markers and parts, if the model is one
std::optional<FvHost::Impl::PlayState::Rig> FvHost::Impl::PlayState::rigOf(const tinygltf::Model& model)
{
  if(std::none_of(model.nodes.begin(), model.nodes.end(), [](const tinygltf::Node& n) { return n.name == "fv_seat"; }))
    return std::nullopt;
  static constexpr const char* kWheels[4] = {"wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"};
  Rig rig;
  rig.model     = &model;
  bool hasBoard = false, hasExit = false, hasCom = false;
  std::vector<glm::dvec3> pads;
  for(size_t i = 0; i < model.nodes.size(); i++)
  {
    const std::string& name  = model.nodes[i].name;
    const bool         pivot = model.nodes[i].mesh < 0;
    const auto         at    = [&] { return glm::dvec3(nodeMatrix(model, int(i))[3]); };
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
    else if(startsWith(name, "gear_") && pivot)
      rig.gear.push_back(int(i));
    else if(endsWith(name, "_glow"))
      rig.glow.push_back(int(i));
    else if(name == "steering_wheel" && pivot)
      rig.steeringWheel = int(i);
    else
      for(int w = 0; w < 4; w++)
        if(name == kWheels[w] && pivot)
          rig.wheels[size_t(w)] = int(i);
  }
  const bool car = std::all_of(rig.wheels.begin(), rig.wheels.end(), [](int n) { return n >= 0; });
  if(!car && std::any_of(rig.wheels.begin(), rig.wheels.end(), [](int n) { return n >= 0; }))
    return std::nullopt;  // a car's wheels, but not all four of them
  // its size from what its meshes take up; a car's body's apart from its wheels'
  const std::vector<int> parent  = parentsOf(model);
  const auto             wheelOf = [&](int node) {
    for(int at = node, steps = 0; at >= 0 && steps <= int(model.nodes.size()); at = parent[size_t(at)], steps++)
      for(int w = 0; w < 4; w++)
        if(at == rig.wheels[size_t(w)])
          return w;
    return -1;
  };
  glm::dvec3                lo(1e30), hi(-1e30);
  std::array<glm::dvec3, 4> wheelLo, wheelHi;
  wheelLo.fill(glm::dvec3(1e30));
  wheelHi.fill(glm::dvec3(-1e30));
  for(size_t i = 0; i < model.nodes.size(); i++)
    if(const int w = car ? wheelOf(int(i)) : -1; w >= 0)
      growByMesh(model, int(i), wheelLo[size_t(w)], wheelHi[size_t(w)]);
    else
      growByMesh(model, int(i), lo, hi);
  if(car)
  {
    rig.kind   = Rig::Kind::eCar;
    rig.canopy = -1;
    rig.gear.clear();
    std::array<game::CarWheel, 4> wheels;
    glm::dvec3                    hubs(0.0);
    double                        tyres = 1e30;
    const bool                    body  = lo.x <= hi.x;  // a body with no mesh of its own is its wheels' box
    for(size_t w = 0; w < 4; w++)
    {
      const int        node = rig.wheels[w];
      const glm::dmat4 m    = nodeMatrix(model, node);
      wheels[w].hub         = glm::dvec3(m[3]);
      if(wheelLo[w].x <= wheelHi[w].x)
        wheels[w].radius = 0.5 * std::max(wheelHi[w].y - wheelLo[w].y, wheelHi[w].z - wheelLo[w].z);
      hubs += 0.25 * wheels[w].hub;
      tyres = std::min(tyres, wheels[w].hub.y - wheels[w].radius);
      if(!body)
      {
        lo = glm::min(lo, wheels[w].hub - glm::dvec3(wheels[w].radius));
        hi = glm::max(hi, wheels[w].hub + glm::dvec3(wheels[w].radius));
      }
      rig.base[node] = nodeLocal(model.nodes[size_t(node)]);
      // the car's up and down in the frame the pivot is turned in (its parent's), and its x along the car's left
      const int        above = parent[size_t(node)];
      const glm::dmat3 back  = glm::inverse(above >= 0 ? glm::dmat3(nodeMatrix(model, above)) : glm::dmat3(1.0));
      rig.upInParent[w]      = glm::normalize(back * glm::dvec3(0.0, 1.0, 0.0));
      rig.downInParent[w]    = back * glm::dvec3(0.0, -1.0, 0.0);
      rig.spinSign[w]        = (glm::dmat3(m) * glm::dvec3(1.0, 0.0, 0.0)).x >= 0.0 ? 1.f : -1.f;
    }
    if(rig.steeringWheel >= 0)
      rig.base[rig.steeringWheel] = nodeLocal(model.nodes[size_t(rig.steeringWheel)]);
    // unmarked: its centre of mass between the wheels, a little nearer the front, at 1.6 tyres' radii up;
    // the way in at the driver's door, out a step past it on the ground
    const double front = 0.5 * (wheels[0].hub.z + wheels[1].hub.z), rear = 0.5 * (wheels[2].hub.z + wheels[3].hub.z);
    if(!hasCom)
      rig.com = {hubs.x, hubs.y + 0.6 * wheels[0].radius, rear + 0.52 * (front - rear)};
    const double side = rig.seat.x >= hubs.x ? 1.0 : -1.0;  // the driver's side: +x, the car's left
    const double door = side > 0.0 ? hi.x : lo.x;
    if(!hasBoard)
      rig.board = {door + side * 0.3, rig.seat.y - 0.3, rig.seat.z};
    if(!hasExit)
      rig.exit = {door + side * 0.9, tyres, rig.seat.z};
    rig.length = std::max(hi.z - lo.z, 1.0);
    for(game::CarWheel& w : wheels)
      w.hub -= rig.com;
    rig.car          = game::carShape(wheels, lo - rig.com, hi - rig.com);
    rig.shape.radius = rig.car.radius;
    return rig;
  }
  for(const int n : rig.gear)
    rig.base[n] = nodeLocal(model.nodes[size_t(n)]);
  if(rig.canopy >= 0)
    rig.base[rig.canopy] = nodeLocal(model.nodes[size_t(rig.canopy)]);
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


// B: a ship down on the ground ahead, its side (the way in) turned to the walker: the nearest the scene
// has, which comes down onto its pads there, else the app's own, loaded. What went wrong, to say.
std::string FvHost::Impl::PlayState::bringShip(Impl& s, PlayState& P, const Clock::time_point& now)
{
  const glm::dvec3 eye   = P.view.eye;
  const glm::dvec3 up    = upAt(s, eye);
  glm::dvec3       ahead = P.view.forward - up * glm::dot(P.view.forward, up);
  ahead                  = glm::length(ahead) > 1e-6 ? glm::normalize(ahead) : glm::dvec3(0.0, 0.0, -1.0);
  std::shared_ptr<Prop> ship;
  double                nearest = 1e300;
  for(const auto& [id, rig] : P.rigs)
    if(const auto p = s.findProp(id); p && s.shown(*p) && rig.kind == Rig::Kind::eShip)
      if(const double d = glm::distance(eye, glm::dvec3(glm::dmat4(s.worldOf(*p))[3])); d < nearest)
      {
        nearest = d;
        ship    = p;
      }
  if(!ship && P.shipPath.empty())
    return "No ship to put down";
  if(!ship && now - P.shipAsked < std::chrono::milliseconds(int(kShipWaits * 1000.0)))
    return "";  // the one asked for is on its way
  const double size   = ship ? scaleOf(glm::dmat4(s.worldOf(*ship))) : 1.0;
  const double length = ship ? P.rigs.at(ship->id).length * size : 0.0;
  glm::dvec3   at     = eye + ahead * std::max(kSpawnAhead, 1.2 * length);
  const auto   sets   = playSets(s, ship ? ship->id : std::string(), at, 20.0);
  if(const auto g = groundUnder(s, sets, at + up * 50.0, up, 2000.0))
    at = g->at;
  else
    at -= up * game::WalkerTuning{}.eyeHeight;
  const glm::dvec3 x = -ahead, z = glm::cross(x, up);
  const glm::dmat3 turn(x, up, z);
  if(ship)
  {
    // a little over the ground, to come down onto its pads however the ground lies
    Rig&       rig = P.rigs.at(ship->id);
    game::Ship put;
    put.orientation = glm::normalize(glm::quat_cast(turn));
    put.com         = at + up * 0.3 + turn * (rig.com * size);
    putRig(s, *ship, rig, put.com, put.orientation);
    rig.loose    = put;
    rig.settling = true;
    return "";
  }
  P.shipAsked     = now;
  const glm::quat q = glm::quat_cast(glm::mat3(turn));
  s.importModel(P.shipPath,
                [&s](std::shared_ptr<Prop> p) {
                  s.recordAdded(p, "Add " + p->name);
                  if(s.view)
                    s.view->deselect(p->id);
                  s.renderer->selectEntity(rs::EntityId{});
                },
                {}, {{"transform", {{"t", {at.x, at.y, at.z}}, {"r", {q.x, q.y, q.z, q.w}}}}});
  return "";
}

// N: a car on the ground ahead, its driver's door turned to the walker: the nearest the scene has, which
// comes down onto its wheels there, else the next of the app's, loaded. What went wrong, to say.
std::string FvHost::Impl::PlayState::bringCar(Impl& s, PlayState& P, const Clock::time_point& now)
{
  const glm::dvec3 eye   = P.view.eye;
  const glm::dvec3 up    = upAt(s, eye);
  glm::dvec3       ahead = P.view.forward - up * glm::dot(P.view.forward, up);
  ahead                  = glm::length(ahead) > 1e-6 ? glm::normalize(ahead) : glm::dvec3(0.0, 0.0, -1.0);
  std::shared_ptr<Prop> car;
  double                nearest = 1e300;
  for(const auto& [id, rig] : P.rigs)
    if(const auto p = s.findProp(id); p && s.shown(*p) && rig.kind == Rig::Kind::eCar)
      if(const double d = glm::distance(eye, glm::dvec3(glm::dmat4(s.worldOf(*p))[3])); d < nearest)
      {
        nearest = d;
        car     = p;
      }
  if(!car && P.carPaths.empty())
    return "No car to put down";
  if(!car && now - P.carAsked < std::chrono::milliseconds(int(kShipWaits * 1000.0)))
    return "";  // the one asked for is on its way
  const double size   = car ? scaleOf(glm::dmat4(s.worldOf(*car))) : 1.0;
  const Rig*   rig    = car ? &P.rigs.at(car->id) : nullptr;
  const double length = rig ? rig->length * size : 5.0;
  glm::dvec3   at     = eye + ahead * std::max(kCarAhead, 1.5 * length);
  const auto   sets   = playSets(s, car ? car->id : std::string(), at, 20.0);
  if(const auto g = groundUnder(s, sets, at + up * 50.0, up, 2000.0))
    at = g->at;
  else
    at -= up * game::WalkerTuning{}.eyeHeight;
  // its side to the walker: the driver's (+x is a car's left, where a left-hand drive's door is)
  const double     side = rig && rig->seat.x < rig->com.x ? -1.0 : 1.0;
  const glm::dvec3 x = -ahead * side, z = glm::cross(x, up);
  const glm::dmat3 turn(x, up, z);
  if(car)
  {
    Rig& placed = P.rigs.at(car->id);
    // a little over the ground, to come down onto its wheels however the ground lies
    placed.drive    = game::carAt(at + up * 0.3 + turn * (placed.com * size), glm::normalize(glm::quat_cast(turn)), carShapeIn(placed, size));
    placed.placed   = true;
    placed.settling = true;
    putRig(s, *car, placed, placed.drive.com, placed.drive.orientation);
    return "";
  }
  P.carAsked        = now;
  const glm::quat q = glm::quat_cast(glm::mat3(turn));
  // over a backdrop's uneven ground a little up, to come down onto its wheels; on the studio floor on it,
  // as the floor's "auto" height is the scene's lowest point, and a car put down over it would lift it
  const glm::dvec3 put = at + up * (s.backdropStands() ? 0.3 : 0.0);
  s.importModel(P.carPaths[P.carNext++ % P.carPaths.size()],
                [&s](std::shared_ptr<Prop> p) {
                  s.recordAdded(p, "Add " + p->name);
                  if(s.view)
                    s.view->deselect(p->id);
                  s.renderer->selectEntity(rs::EntityId{});
                },
                {}, {{"transform", {{"t", {put.x, put.y, put.z}}, {"r", {q.x, q.y, q.z, q.w}}}}});
  return "";
}

// The torch in hand where the view is, or put away
void FvHost::Impl::PlayState::showTorch(Impl& s, PlayState& P, bool on, const game::World& world, double dt)
{
  if(!s.renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = s.renderer->hostScene();
  const bool         have  = scene.entity(P.torchEntity) && scene.light(P.torchLight);
  if(!on)
  {
    if(have && P.torchShown)
      scene.setLightShown(P.torchLight, false);
    P.torchShown   = false;
    P.torch.placed = false;  // taken out again it is in hand at once, not swinging in from where it was
    return;
  }
  const game::Pose    pose = game::torchView(P.torch, P.view, game::sightDistance(world, P.view, kTorchSight), dt);
  const rs::Transform local{.translation = glm::vec3(pose.eye), .rotation = glm::quat(glm::quatLookAt(pose.forward, pose.up))};
  if(!have)
  {
    P.torchEntity = scene.createEntity("Torch", {0, -1});
    scene.setLocal(P.torchEntity, local);
    P.torchLight = scene.addLight({.owner = {0, -1}, .payload = torchPayload()});
    scene.setEntityLight(P.torchEntity, P.torchLight);
    scene.release(P.torchLight);
  }
  else
  {
    if(!P.torchShown)
      scene.setLightShown(P.torchLight, true);
    if(!(P.torchAt == local))
      scene.setLocal(P.torchEntity, local);
  }
  P.torchShown = true;
  P.torchAt    = local;
}

void FvHost::Impl::PlayState::dropTorch(Impl& s, PlayState& P)
{
  if(s.renderer->feedsHostScene() && s.renderer->hostScene().entity(P.torchEntity))
    s.renderer->hostScene().destroyEntity(P.torchEntity);
  P.torchEntity  = {};
  P.torchLight   = {};
  P.torchShown   = false;
  P.torch.placed = false;
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
// planet's, or a prop's surface. A planet's ground is faced as it lies with `slopes` (two more asks of
// its height, for a car's wheels), else straight up (all on foot or in a ship need).
std::optional<game::Surface> FvHost::Impl::PlayState::groundUnder(Impl& s, const std::vector<PlacedSet>& sets, const glm::dvec3& from, const glm::dvec3& up,
                                                    double reach, bool slopes)
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
      {
        const fv::world::World& world  = *s.planetBuilt->world;
        const glm::dvec3        lift(0.0, floor, 0.0);
        const auto              under  = [&](const glm::dvec3& p) { return p - up * fv::world::heightOver(world, p - lift); };
        const glm::dvec3        ground = under(from);
        glm::dvec3              normal = up;
        if(slopes)
        {
          // the ground's own slope from two points of it some way off (about the size of what it is drawn with)
          constexpr double kApart = 0.75;
          const glm::dvec3 across = glm::length(glm::cross(up, glm::dvec3(1.0, 0.0, 0.0))) > 0.1 ? glm::normalize(glm::cross(up, glm::dvec3(1.0, 0.0, 0.0)))
                                                                                                   : glm::normalize(glm::cross(up, glm::dvec3(0.0, 0.0, 1.0)));
          const glm::dvec3 along = glm::cross(across, up);
          glm::dvec3       n     = glm::cross(under(from + along * kApart) - ground, under(from + across * kApart) - ground);
          if(glm::length(n) > 1e-9)
          {
            n      = glm::normalize(n);
            normal = glm::dot(n, up) < 0.0 ? -n : n;
          }
        }
        consider(ground, normal);
      }
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
// itself) and people's, whose poses move under them; the planet's places round `around` (their walls
// are met as a prop's are) and its bridges (their decks stood and driven on, over the river), and the
// backdrop's trees and stones within `within` of it. A set not indexed yet is indexed on the pool and
// met once it is.
std::vector<FvHost::Impl::PlacedSet> FvHost::Impl::PlayState::playSets(Impl& s, const std::string& skip, const glm::dvec3& around, double within)
{
  std::vector<PlacedSet> out;
  for(const auto& q : s.props)
  {
    if(!s.shown(*q) || q->kind == PropKind::eCharacter || q->kind == PropKind::eCreature
       || (!skip.empty() && (q->id == skip || s.ridesOn(*q, skip))))
      continue;
    for(PlacedSet& set : s.placedSets(*q))
      out.push_back(std::move(set));
  }
  s.placeSets(around, 60.0, out);
  plantSets(s, around, within, out);
  std::vector<std::shared_ptr<const fv::physics::SharedTriangles>> unmade;
  for(const PlacedSet& set : out)
    if(!set.set->made() && s.play->indexing.insert(set.set.get()).second)
      unmade.push_back(set.set);
  if(!unmade.empty())
    s.loaders.spawn([unmade] {
      for(const auto& set : unmade)
        set->get();
    });
  return out;
}

// The backdrop's plants that stand within `within` of `around` and are met: a tree (and what else grows
// tall) by its trunk, a stone, a stump or a log by its own triangles; shrubs, grass and the sea
// floor's are walked through. A flat backdrop's where its variants have them; a planet's in each quad
// drawn near.
void FvHost::Impl::PlayState::plantSets(Impl& s, const glm::dvec3& around, double within, std::vector<PlacedSet>& out)
{
  if(!s.backdropBuilt || !s.backdropStands() || !s.renderer->feedsHostScene())
    return;
  rs::RetainedScene&   scene = s.renderer->hostScene();
  const BackdropBuilt& built = *s.backdropBuilt;
  if(!scene.entity(built.root))
    return;
  const glm::mat4 root  = scene.world(built.root);
  const glm::vec3 at    = glm::vec3(glm::inverse(root) * glm::vec4(glm::vec3(around), 1.0F));  // in the root's frame
  const auto      solid = [&](std::string_view role, const BackdropBuilt::Variant& v) -> std::shared_ptr<const fv::physics::SharedTriangles> {
    if(!v.model || v.mesh < 0)
      return nullptr;
    if(fv::world::isTall(role))
      return s.trunkMesh(v.model, v.mesh, v.base);
    if(role == "rock" || role == "boulder" || role == "stump" || role == "log")
      return s.contactMesh(v.model, v.mesh);
    return nullptr;
  };
  // each copy's place (its variant's own turn and scale under where it stands), about `about`
  const auto meet = [&](const std::shared_ptr<const fv::physics::SharedTriangles>& set, const glm::mat4& toRoot, const glm::vec3& about,
                        const std::vector<glm::mat4>& places) {
    for(const glm::mat4& place : places)
      if(glm::distance(about + glm::vec3(place[3]), at) <= float(within))
        out.push_back({set, toRoot * place});
  };
  if(fv::backdrop::isPlanet(s.backdropSpec))
  {
    if(!s.planetBuilt)
      return;
    for(const auto& [quad, chunk] : s.planetBuilt->shown)
    {
      if(!chunk.stood || glm::distance(glm::vec3(chunk.middle), at) > float(chunk.round + within))
        continue;
      const glm::mat4 toRoot = root * glm::translate(glm::mat4(1.0F), glm::vec3(chunk.middle));
      for(const PlanetBuilt::Chunk::Stood& stood : *chunk.stood)
      {
        if(stood.role < 0 || size_t(stood.role) >= fv::world::kStandingRoles.size())
          continue;
        // its own models, or where the kit has none, those it is drawn with (a riverside tree, a leafy one)
        const std::string_view role  = fv::world::kStandingRoles[size_t(stood.role)];
        auto                   found = built.variants.find(std::string(role));
        if(found == built.variants.end())
          found = built.variants.find(std::string(fv::world::drawnAs(role)));
        if(found == built.variants.end() || stood.variant >= found->second.size())
          continue;
        if(const auto set = solid(role, found->second[stood.variant]))
          meet(set, toRoot, glm::vec3(chunk.middle), stood.places);
      }
    }
    return;
  }
  for(const auto& [role, variants] : built.variants)
    for(const BackdropBuilt::Variant& v : variants)
      if(!v.resting.empty())
        if(const auto set = solid(role, v))
          meet(set, root, glm::vec3(0.0F), v.resting);
}

game::World FvHost::Impl::PlayState::playWorld(Impl& s, const std::vector<PlacedSet>& sets, bool slopes)
{
  game::World w;
  w.up     = [&s](const glm::dvec3& p) { return upAt(s, p); };
  w.ground = [&s, &sets, slopes](const glm::dvec3& from, const glm::dvec3& up, double reach) { return groundUnder(s, sets, from, up, reach, slopes); };
  w.ray    = [&sets](const glm::dvec3& from, const glm::dvec3& dir, double reach) -> std::optional<game::Surface> {
    const auto hit = Impl::rayThrough(sets, glm::vec3(from), glm::vec3(dir), true);
    if(!hit || glm::dot(glm::dvec3(hit->at) - from, dir) > reach)
      return std::nullopt;
    return game::Surface{glm::dvec3(hit->at), glm::dvec3(hit->normal)};
  };
  // the fastest full throttle goes: a planet's radius in ten seconds, a backdrop's land in twenty
  if(s.backdropStands())
    w.speedCap = fv::backdrop::isPlanet(s.backdropSpec) ? 100.0 * s.backdropSpec.radius : 400.0;
  else
    w.speedCap = 250.0;
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

// Its prop where the controller has its centre (`com`) and turn: no step of the history (play.stop makes one)
void FvHost::Impl::PlayState::putRig(Impl& s, Prop& p, const Rig& rig, const glm::dvec3& com, const glm::dquat& orientation)
{
  const double     size   = scaleOf(glm::dmat4(s.worldOf(p)));
  const glm::dmat3 turn   = glm::mat3_cast(orientation);
  const glm::dvec3 origin = com - turn * (rig.com * size);
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

// A car's shape in the world's metres, about its centre
game::CarShape FvHost::Impl::PlayState::carShapeIn(const Rig& rig, double size)
{
  game::CarShape shape = rig.car;
  for(game::CarWheel& w : shape.wheels)
  {
    w.hub *= size;
    w.radius *= size;
  }
  for(glm::dvec3& p : shape.hull)
    p *= size;
  shape.half *= size;
  shape.radius *= size;
  return shape;
}

// A car the controller doesn't hold yet: where its prop stands, its suspension as its file has it
void FvHost::Impl::PlayState::placeCar(const Impl& s, const Prop& p, Rig& rig)
{
  if(rig.placed)
    return;
  const game::Ship at = shipOf(s, p, rig);  // (its centre and its turn, as a ship's)
  rig.drive           = game::carAt(at.com, at.orientation, carShapeIn(rig, scaleOf(glm::dmat4(s.worldOf(p)))));
  rig.placed          = true;
}

// A car nobody drives, while it settles: down onto its wheels, its brakes on, till it sleeps
void FvHost::Impl::PlayState::stepLooseCar(Impl& s, Prop& p, Rig& rig, double dt)
{
  placeCar(s, p, rig);
  const game::CarShape shape = carShapeIn(rig, scaleOf(glm::dmat4(s.worldOf(p))));
  const auto           own   = playSets(s, p.id, rig.drive.com, kSolidsNear + 2.0 * shape.radius);
  const game::World    its   = playWorld(s, own, true);
  substeps(dt, [&](double h) { game::stepCar(rig.drive, shape, its, h, false); });
  putRig(s, p, rig, rig.drive.com, rig.drive.orientation);
  if(rig.drive.asleep)
    rig.settling = false;
}

size_t FvHost::Impl::PlayState::countOf(const PlayState& P, Rig::Kind kind)
{
  return size_t(std::count_if(P.rigs.begin(), P.rigs.end(), [kind](const auto& r) { return r.second.kind == kind; }));
}

// Its canopy, gear and engines as the rig has them now; a car's wheels where its suspension has them,
// steered and turned as they roll, and its steering wheel turned with them
void FvHost::Impl::PlayState::poseRig(Impl& s, const Prop& p, const Rig& rig)
{
  rs::RetainedScene& scene = s.renderer->hostScene();
  const auto         set   = [&](int node, const rs::Transform& local) {
    const rs::EntityId e      = s.nodeEntity(p, node);
    const rs::Entity*  entity = scene.entity(e);
    if(entity && !(entity->local == local))
      scene.setLocal(e, local);
  };
  if(rig.kind == Rig::Kind::eCar)
  {
    const double size = scaleOf(glm::dmat4(s.worldOf(p)));
    const double rest = game::CarTuning{}.travelUp;
    for(size_t w = 0; w < 4; w++)
    {
      const auto base = rig.base.find(rig.wheels[w]);
      if(base == rig.base.end())
        continue;
      const game::CarWheelState& state = rig.drive.wheels[w];
      rs::Transform              local = base->second;
      local.translation += glm::vec3(rig.downInParent[w] * ((state.length - rest) / size));
      local.rotation = glm::angleAxis(float(state.steer), glm::vec3(rig.upInParent[w])) * local.rotation
                       * glm::angleAxis(rig.spinSign[w] * float(state.spin), glm::vec3(1.f, 0.f, 0.f));
      set(rig.wheels[w], local);
    }
    if(const auto base = rig.base.find(rig.steeringWheel); base != rig.base.end())
    {
      rs::Transform local = base->second;
      local.rotation      = local.rotation * glm::angleAxis(float(-rig.drive.steer * kSteeringRatio), glm::vec3(0.f, 0.f, 1.f));
      set(rig.steeringWheel, local);
    }
  }
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

// Ships and cars come and go with the props: a model with a seat is one. A car found settles onto its
// wheels where it was put.
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
    {
      rig->settling = rig->kind == Rig::Kind::eCar;
      P.rigs.emplace(p->id, std::move(*rig));
    }
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
    P.eye = args.value("eye", true);
    renderer->setPlayEye(P.eye);  // the view's exposure is the walker's eye's while playing, unless it is put off
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
    const auto sets  = PlayState::playSets(*this, "", eye);
    const auto world = PlayState::playWorld(*this, sets);
    const glm::dvec3 ahead = glm::length(center - eye) > 1e-9 ? glm::normalize(center - eye) : glm::dvec3(0.0, 0.0, -1.0);
    P.walker               = game::walkerAt({eye, ahead, up}, world, 2000.0);
    P.view                 = game::walkerView(P.walker, world);
  }
  PlayState::findRigs(*this, P);
  // the ship B puts down when the scene has none, and the cars N does in turn
  P.shipPath = args.value("ship", std::string());
  P.carPaths.clear();
  if(const auto cars = args.find("cars"); cars != args.end() && cars->is_array())
    for(const Json& path : *cars)
      if(path.is_string() && !path.get<std::string>().empty())
        P.carPaths.push_back(path.get<std::string>());
  publishPlay(true);
  return {{"ok", true}};
}

void FvHost::Impl::playMoveTo(const glm::dvec3& eye, const glm::dvec3& ahead, const glm::dvec3& up)
{
  PlayState& P = *play;
  if(P.mode != PlayState::Mode::eWalk)
    throw std::runtime_error("get out first: Go to moves the one walking");
  const auto sets  = PlayState::playSets(*this, "", eye);
  const auto world = PlayState::playWorld(*this, sets);
  P.walker         = game::walkerAt({eye, ahead, up}, world, 2000.0);
  P.view           = game::walkerView(P.walker, world);
}

Json FvHost::Impl::playStop()
{
  if(!playing())
    return {{"ok", true}};
  PlayState& P = *play;
  P.active     = false;
  PlayState::dropTorch(*this, P);
  renderer->setPlayEye(false);
  // the ships and cars stay where they were left, a step of the history each, and their parts as their
  // files have them
  for(auto& [id, rig] : P.rigs)
    if(const auto p = findProp(id))
    {
      rig.canopyAngle = rig.gearAngle = 0.0;
      rig.engines                     = false;
      rig.drive.steer                 = 0.0;
      for(game::CarWheelState& w : rig.drive.wheels)
        w = game::CarWheelState{.length = game::CarTuning{}.travelUp};
      PlayState::poseRig(*this, *p, rig);
    }
  for(const auto& [id, before] : P.startedAt)
    if(const auto p = findProp(id))
    {
      const fv::composer::Trs now  = p->transform;
      const auto              rig  = P.rigs.find(id);
      const bool              car  = rig != P.rigs.end() && rig->second.kind == PlayState::Rig::Kind::eCar;
      p->transform                 = before;
      applyTransform(*p);
      try
      {
        edit("prop.set", {{"prop", id},
                          {"transform",
                           {{"t", {now.translation.x, now.translation.y, now.translation.z}},
                            {"r", {now.rotation.x, now.rotation.y, now.rotation.z, now.rotation.w}},
                            {"s", {now.scale.x, now.scale.y, now.scale.z}}}},
                          {"label", (car ? "Drive " : "Fly ") + p->name}});
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
  P.torches += (pressed & game::kTorch) ? 1 : 0;
  P.brings += (pressed & game::kShipHere) ? 1 : 0;
  P.carBrings += (pressed & game::kCarHere) ? 1 : 0;
  P.eyes += (pressed & game::kEye) ? 1 : 0;
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
  const int        torches = std::exchange(P.torches, 0), brings = std::exchange(P.brings, 0), eyes = std::exchange(P.eyes, 0);
  const int        carBrings = std::exchange(P.carBrings, 0);
  const glm::dvec2 look      = std::exchange(P.look, glm::dvec2(0.0));

  PlayState::findRigs(*this, P);
  std::shared_ptr<Prop> flown = P.shipId.empty() ? nullptr : findProp(P.shipId);
  PlayState::Rig*       rig   = flown && P.rigs.count(P.shipId) ? &P.rigs.at(P.shipId) : nullptr;
  const auto            isCar = [&] { return rig && rig->kind == PlayState::Rig::Kind::eCar; };
  // what is met near whoever is in charge: on foot, in the ship or at the wheel
  const bool            aboard = rig && P.mode != PlayState::Mode::eWalk && P.mode != PlayState::Mode::eExit;
  const auto            sets   = PlayState::playSets(*this, "", !aboard ? P.walker.feet : isCar() ? rig->drive.com : P.ship.com);
  const game::World     world  = PlayState::playWorld(*this, sets);
  if(P.mode != PlayState::Mode::eWalk && !rig)
  {
    // the ship or the car went (an undo, a delete): on foot where the camera is
    P.walker = game::walkerAt(P.view, world, 2000.0);
    P.mode   = PlayState::Mode::eWalk;
    P.shipId.clear();
  }
  const double size     = flown ? scaleOf(glm::dmat4(worldOf(*flown))) : 1.0;
  std::string  prompt;
  const auto   seatPose = [&] {
    return isCar() ? game::cockpitView(game::bodyOf(rig->drive), (rig->seat - rig->com) * size, rig->drive.gaze)
                   : game::cockpitView(P.ship, (rig->seat - rig->com) * size);
  };
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
      // the nearest ship or car whose way in is within reach
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
      if(brings > 0)
        if(const std::string why = PlayState::bringShip(*this, P, now); !why.empty())
          say(why);
      if(carBrings > 0)
        if(const std::string why = PlayState::bringCar(*this, P, now); !why.empty())
          say(why);
      if(uses > 0 && !closest.empty())
      {
        P.shipId     = closest;
        flown        = findProp(closest);
        rig          = &P.rigs.at(closest);
        P.chase      = {.zoom = P.chase.zoom};
        P.mode       = PlayState::Mode::eEnter;
        P.transition = 0.0;
        P.from       = P.view;
        if(isCar())
        {
          // it keeps how it rests; the driver looks ahead
          PlayState::placeCar(*this, *flown, *rig);
          rig->drive.gaze = rig->drive.look = glm::dvec2(0.0);
        }
        else
        {
          P.ship        = PlayState::shipOf(*this, *flown, *rig);
          rig->settling = false;
        }
      }
      break;
    }
    case PlayState::Mode::eEnter: {
      P.transition += dt;
      P.view = game::blend(P.from, seatPose(), P.transition / kEnterSeconds);
      if(P.transition >= kEnterSeconds)
      {
        P.mode          = isCar() ? PlayState::Mode::eDrive : PlayState::Mode::eFly;
        P.ship.throttle = 0.0;
        P.ship.aim      = glm::dvec2(0.0);
        rig->engines    = true;
        rig->settling   = false;
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
      const game::ShipShape shape = PlayState::shapeIn(*rig, size);
      const auto            own   = PlayState::playSets(*this, P.shipId, P.ship.com, kSolidsNear + 2.0 * shape.radius + 0.5 * glm::length(P.ship.velocity));
      const game::World     its   = PlayState::playWorld(*this, own);
      substeps(dt, [&](double h) { game::stepShip(P.ship, shape, its, h, true); });
      P.range = game::shipSpeedRange(P.ship, its.speedCap);
      PlayState::putRig(*this, *flown, *rig, P.ship.com, P.ship.orientation);
      P.view = P.cockpit ? seatPose() : game::chaseView(P.chase, P.ship, rig->length * size, dt);
      const bool canLeave = glm::length(P.ship.velocity) < kExitSpeed && P.ship.height >= 0.0 && P.ship.height < kExitHeight;
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
    case PlayState::Mode::eDrive: {
      if(views > 0)
        P.cockpit = !P.cockpit;
      game::Car& car = rig->drive;
      car.keys       = P.keys;
      car.look += look;
      const game::CarShape shape = PlayState::carShapeIn(*rig, size);
      const auto           own   = PlayState::playSets(*this, P.shipId, car.com, kSolidsNear + 2.0 * shape.radius + 0.5 * glm::length(car.velocity));
      const game::World    its   = PlayState::playWorld(*this, own, true);
      substeps(dt, [&](double h) { game::stepCar(car, shape, its, h, true); });
      PlayState::putRig(*this, *flown, *rig, car.com, car.orientation);
      // from the driver's seat, or behind it, the chase kept in front of what stands between and over the ground
      const glm::dvec3 up = PlayState::upAt(*this, car.com);
      P.view = P.cockpit ? seatPose()
                         : game::clearView(game::chaseView(P.chase, game::chaseBody(car, up), rig->length * size, dt, game::kCarChase),
                                           car.com + up * (0.5 * size), its);
      const bool canLeave = std::abs(game::carSpeed(car)) < kStopSpeed;
      if(canLeave)
        prompt = "F  Get out";
      if(uses > 0)
      {
        if(!canLeave)
          say("Stop to get out");
        else
        {
          // out at the driver's door, standing on what is there, facing the car; at the other door when
          // that side is against something
          const glm::dmat4 placed = glm::dmat4(worldOf(*flown));
          glm::dvec3       out    = glm::dvec3(placed * glm::dvec4(rig->exit, 1.0));
          const glm::dvec3 middle = glm::dvec3(placed * glm::dvec4(rig->com, 1.0));
          if(const glm::dvec3 way = out - middle; glm::length(way) > 1e-6)
            if(its.ray(middle, glm::normalize(way), glm::length(way) + 0.3))
            {
              const glm::dvec3 mirrored = rig->exit + glm::dvec3(2.0 * (rig->com.x - rig->exit.x), 0.0, 0.0);
              out                       = glm::dvec3(placed * glm::dvec4(mirrored, 1.0));
            }
          const glm::dvec3 standUp = PlayState::upAt(*this, out);
          glm::dvec3       facing  = car.com - out;
          facing -= standUp * glm::dot(facing, standUp);
          const game::WalkerTuning t;
          P.walker = game::walkerAt({out + standUp * (t.eyeHeight + 0.5), glm::length(facing) > 1e-6 ? glm::normalize(facing) : glm::dvec3(0.0, 0.0, -1.0), standUp},
                                    world, 20.0);
          P.from        = P.view;
          P.transition  = 0.0;
          P.mode        = PlayState::Mode::eExit;
          rig->engines  = false;
          rig->settling = true;
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

  // ships nobody flies: one left in the air comes down onto its pads and stops; then it stands. Cars
  // nobody drives settle onto their wheels and sleep.
  for(auto& [id, r] : P.rigs)
  {
    const auto p = findProp(id);
    if(!p)
      continue;
    if(r.kind == PlayState::Rig::Kind::eCar)
    {
      const bool driving = id == P.shipId && (P.mode == PlayState::Mode::eDrive || P.mode == PlayState::Mode::eEnter);
      if(r.settling && !driving)
        PlayState::stepLooseCar(*this, *p, r, dt);
      if(r.placed)
        PlayState::poseRig(*this, *p, r);
      continue;
    }
    const bool flying = id == P.shipId && (P.mode == PlayState::Mode::eFly || P.mode == PlayState::Mode::eEnter);
    if(r.settling && !flying)
    {
      const double          sz    = scaleOf(glm::dmat4(worldOf(*p)));
      const game::ShipShape shape = PlayState::shapeIn(r, sz);
      const auto            own   = PlayState::playSets(*this, id, r.loose.com, kSolidsNear + 2.0 * shape.radius);
      const game::World     its   = PlayState::playWorld(*this, own);
      substeps(dt, [&](double h) { game::stepShip(r.loose, shape, its, h, false); });
      PlayState::putRig(*this, *p, r, r.loose.com, r.loose.orientation);
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

  // the torch, in hand on foot (L): put away in the ship, out again on getting out if it was
  if(torches % 2 == 1)
    P.torch.on = !P.torch.on;
  PlayState::showTorch(*this, P, P.torch.on && (P.mode == PlayState::Mode::eWalk || P.mode == PlayState::Mode::eExit), world, dt);
  // the eye (X): the view's exposure adapting as it does, on foot and in the ship, or as the Camera tab has it
  if(eyes % 2 == 1)
  {
    P.eye = !P.eye;
    renderer->setPlayEye(P.eye);
  }

  // the camera: a near plane close enough for the cockpit, the far one the scene's
  const glm::dvec2 clip = manip->getClipPlanes();
  if(clip.x > 0.03)
    manip->setClipPlanes({0.02, clip.y});
  const double ahead = P.mode == PlayState::Mode::eWalk ? 10.0 : 30.0;
  manip->setLookat(P.view.eye, P.view.eye + P.view.forward * ahead, P.view.up, true);
  if(now < P.noticeUntil)
    prompt = P.notice;

  const bool  fly     = P.mode == PlayState::Mode::eFly;
  const bool  driving = P.mode == PlayState::Mode::eDrive && isCar();
  const char* modes[] = {"walk", "enter", "fly", "exit", "drive"};
  const Json  vehicle = !rig || P.mode == PlayState::Mode::eWalk ? Json() : Json(isCar() ? "car" : "ship");
  double      speed   = 0.0;
  if(fly)
    speed = glm::length(P.ship.velocity);
  else if(driving)
    speed = std::abs(game::carSpeed(rig->drive));
  Json state = {{"active", true},
                {"mode", modes[int(P.mode)]},
                {"prompt", prompt},
                {"ship", P.shipId.empty() ? Json() : Json(P.shipId)},
                {"vehicle", vehicle},
                {"ships", PlayState::countOf(P, PlayState::Rig::Kind::eShip)},
                {"cars", PlayState::countOf(P, PlayState::Rig::Kind::eCar)},
                {"view", P.cockpit ? "cockpit" : "chase"},
                {"speed", std::round(speed * 10.0) / 10.0},
                {"height", fly ? std::round(std::max(0.0, P.ship.height)) : 0.0},
                {"throttle", fly ? std::round(P.ship.throttle * 100.0) / 100.0 : 0.0},
                {"range", fly ? std::round(P.range) : 0.0},
                {"boost", (fly && (P.ship.keys & game::kFast) != 0) || (driving && (rig->drive.keys & game::kFast) != 0)},
                {"gear", driving ? Json(rig->drive.gear < 0 ? "R" : "D") : Json()},
                {"handbrake", driving && rig->drive.handbrake},
                {"torch", P.torch.on},
                {"eye", P.eye}};
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
  Json state = playing() ? Json{{"active", true},
                                {"mode", "walk"},
                                {"prompt", ""},
                                {"ships", PlayState::countOf(*play, PlayState::Rig::Kind::eShip)},
                                {"cars", PlayState::countOf(*play, PlayState::Rig::Kind::eCar)},
                                {"torch", play->torch.on},
                                {"eye", play->eye}}
                         : Json{{"active", false}};
  if(play)
  {
    play->published   = state;
    play->publishedAt = Clock::now();
  }
  protocol->events().snapshot("play", state);
}
