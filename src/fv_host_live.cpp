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
// FvHost: live edits on the application thread: node transforms, vertex uploads, rebinds, morphs,
// poses and viewport input.
//

#include "fv_host_impl.hpp"

// The figure's moved nodes, fed: their transforms (a pose step), or the whole figure when a node is new
void FvHost::Impl::applyNodes(Loaded& f, const std::set<int>& moved)
{
  if(moved.empty() || !f.feed)
    return;
  const std::vector<int> nodes(moved.begin(), moved.end());
  if(!f.feed->syncTransforms(f.fig->loaded.scene, f.fig->loaded.root, nodes))
    f.feed->sync(f.fig->loaded.scene, f.fig->loaded.root);
}

// A mesh whose vertices moved in place (a dial, a fit, a drape), fed with the tangents it had: the
// frame loop makes them once it stops moving
void FvHost::Impl::applyMesh(Loaded& f, const fv::morph::MeshChange& change)
{
  if(!f.feed || change.mesh < 0 || change.mesh >= int(f.fig->loaded.scene.meshes.size())
     || !f.fig->loaded.scene.meshes[change.mesh].geo)
    return;
  const fv::mesh::BuiltMesh& geo   = *f.fig->loaded.scene.meshes[change.mesh].geo;
  const size_t              count = (change.cage ? geo.cagePositions.size() : geo.position.size()) / 3;
  if(!change.all && change.indices.empty())
    return;
  dialTiming.addUpload(change.all ? count : change.indices.size(), change.all);
  const int  mesh = change.mesh;
  const bool fed  = change.cage ? f.feed->syncCages(f.fig->loaded.scene, {&mesh, 1})
                                : f.feed->syncVertices(f.fig->loaded.scene, {&mesh, 1}, true);
  if(!fed)
    f.feed->sync(f.fig->loaded.scene, f.fig->loaded.root);
}

// Meshes bound again to their bones (a rebind): their skins' inverse binds
void FvHost::Impl::applyRebind(Loaded& f, const std::vector<int>& meshes)
{
  if(!f.feed || meshes.empty())
    return;
  if(!f.feed->syncSkins(f.fig->loaded.scene, meshes))
    f.feed->sync(f.fig->loaded.scene, f.fig->loaded.root);
}

void FvHost::Impl::prefetchImages(const std::vector<fv::build::TextureRequest>& requests, const std::vector<std::optional<fv::build::TextureImage>>& found)
{
  fv::timing::Scope        t("decode textures");
  std::atomic<size_t>      next{0};
  std::vector<std::thread> workers;
  for(int w = 0; w < std::min<int>(8, int(found.size())); w++)
    workers.emplace_back([&] {
      for(size_t i; (i = next++) < found.size();)
        if(found[i] && !found[i]->uri.empty())
          renderer->prefetchImage(pathOf(found[i]->uri), requests[i].srgb);
    });
  for(auto& w : workers)
    w.join();
}

void FvHost::Impl::applyPendingMorphs()
{
  std::map<std::string, std::map<std::string, double>> pending;
  std::map<std::string, Json>                          gestures;
  std::map<std::pair<std::string, std::string>, Json>  tight;
  {
    std::lock_guard lock(morphMutex);
    pending.swap(pendingMorphs);
    gestures.swap(pendingGestures);
    tight.swap(pendingTight);
  }
  for(const auto& [key, args] : tight)
  {
    // the item may have been taken off since
    try
    {
      edit("wardrobe.tight", args);
    }
    catch(const std::exception& e)
    {
      LOGW("[fv] wardrobe.tight: %s\n", e.what());
    }
  }
  for(const auto& [id, dials] : pending)
  {
    const auto p = findProp(id);
    if(!p || !p->figure || !p->figure->fig->morphs || dials.empty())
      continue;
    std::vector<std::pair<std::string, double>> values(dials.begin(), dials.end());
    // a step of the history: what a frame's dials were before, read here where they land
    Json args = {{"prop", id}, {"values", Json::object()}};
    for(const auto& [name, value] : dials)
      args["values"][name] = value;
    if(const auto g = gestures.find(id); g != gestures.end())
      args["gesture"] = g->second;
    journaled("morph.set", args, [&] {
      const auto t0 = std::chrono::steady_clock::now();
      p->figure->fig->morphs->setMany(values);
      dialTiming.add(DialTiming::eSetMany, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      return Json();
    });
  }
}

void FvHost::Impl::publishMorphs(Loaded& f)
{
  if(!f.fig->morphs)
    return;
  fv::morph::MorphSet& morphs      = *f.fig->morphs;
  const bool           listChanged = morphs.listVersion() != f.morphList;
  if(!listChanged && f.morphsSeen == morphs.version())
    return;
  f.morphsSeen = morphs.version();
  auto& events = protocol->events();
  if(listChanged)
  {
    Json channels = Json::object();
    f.published.clear();
    int order = 0;
    for(const fv::morph::Channel& ch : morphs.channels())
    {
      if(ch.hidden)
        continue;
      const double w = morphs.weightOf(ch.name);
      channels[ch.name] = {{"label", ch.label}, {"group", ch.group}, {"value", ch.value}, {"min", ch.min},
                           {"max", ch.max},     {"weight", w},       {"order", order++}};
      f.published[ch.name] = {ch.value, w};
    }
    f.morphList = morphs.listVersion();
    events.snapshot("morphs/" + f.id, {{"figure", f.id}, {"channels", channels}});
    return;
  }
  Json channels = Json::object();
  for(const fv::morph::Channel& ch : morphs.channels())
  {
    if(ch.hidden)
      continue;
    const double w  = morphs.weightOf(ch.name);
    const auto   it = f.published.find(ch.name);
    if(it != f.published.end() && it->second.first == ch.value && it->second.second == w)
      continue;
    channels[ch.name]    = {{"value", ch.value}, {"weight", w}};
    f.published[ch.name] = {ch.value, w};
  }
  if(!channels.empty())
    events.patch("morphs/" + f.id, {{"channels", channels}});
}

void FvHost::Impl::publishBusy(Loaded& f)
{
  const std::string b = f.fig->busy();
  if(!b.empty())
  {
    if(!f.busyOn)
    {
      f.busyJob = ++jobs;
      f.busyOn  = true;
      job(f.busyJob, {{"label", b}, {"state", "running"}});
    }
  }
  else if(f.busyOn)
  {
    job(f.busyJob, {{"state", "done"}});
    f.busyOn = false;
  }
}

Json FvHost::Impl::poseSet(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->fig->pose)
    throw std::runtime_error("no such figure");
  std::vector<fv::pose::ChannelValue> values;
  for(auto it = args.at("values").begin(); it != args.at("values").end(); ++it)
  {
    // "bone/kind/axis": the bone id may itself hold no slash
    const std::string key  = it.key();
    const size_t      a    = key.rfind('/');
    const size_t      k    = a == std::string::npos ? std::string::npos : key.rfind('/', a - 1);
    fv::pose::ChannelKind kind;
    if(k == std::string::npos || !fv::pose::parseKind(key.substr(k + 1, a - k - 1), kind) || !it.value().is_number())
      continue;
    values.push_back({key.substr(0, k), kind, key.substr(a + 1), it.value().get<double>()});
  }
  std::vector<std::string> bones;
  for(const auto& v : values)
    if(std::find(bones.begin(), bones.end(), v.bone) == bones.end())
      bones.push_back(v.bone);
  // nodes, corrective morphs and rebound skins go out with this frame's pump()
  {
    const auto t0 = std::chrono::steady_clock::now();
    figure->fig->pose->setMany(values);
    dialTiming.add(DialTiming::ePoseSet, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  // the values as the rig took them (clamped), for every panel showing the pose
  patchPose(*figure, bones);
  return {{"changed", figure->fig->pose->changed()}};
}

void FvHost::Impl::frameCamera()
{
  auto manip = renderer->getCameraManipulator();
  if(!renderer->feedsHostScene() || !manip)
    return;
  const nvutils::Bbox box  = renderer->visibleSceneBounds();
  const VkExtent2D    size = renderer->renderSize();
  if(box.isEmpty())
    return;  // nothing drawn to frame: the view stays
  manip->fit(box.min(), box.max(), true, false, size.height ? double(size.width) / size.height : 1.0);
}

// Input events as the frame has them. The app maps a pointer to the frame size it asked for last
// and tags each event with it (`w`, `h`) and with its batch's number (`seq`): a batch mapped to
// another size (a /size still in flight) is scaled to this frame; batches are put in order; and a
// batch that arrived after a later one (two POSTs on two server threads) keeps its presses,
// releases and wheel but loses its moves, whose positions are stale.
void FvHost::Impl::normalizeInput(std::vector<Json>& events)
{
  const auto seqOf = [](const Json& e) { return e.value("seq", uint64_t(0)); };
  std::stable_sort(events.begin(), events.end(), [&](const Json& a, const Json& b) { return seqOf(a) < seqOf(b); });
  const VkExtent2D  size = renderer->renderSize();
  uint64_t          newest = lastInputSeq;
  std::vector<Json> kept;
  kept.reserve(events.size());
  for(Json& e : events)
  {
    const uint64_t seq = seqOf(e);
    if(seq && seq < lastInputSeq && e.value("t", "") == "move")
      continue;
    newest = std::max(newest, seq);
    const double w = e.value("w", 0.0), h = e.value("h", 0.0);
    if(w > 0 && h > 0 && size.width && size.height && e.contains("x") && e.contains("y")
       && (uint32_t(std::lround(w)) != size.width || uint32_t(std::lround(h)) != size.height))
    {
      e["x"] = e["x"].get<double>() * size.width / w;
      e["y"] = e["y"].get<double>() * size.height / h;
    }
    kept.push_back(std::move(e));
  }
  lastInputSeq = newest;
  events.swap(kept);
}

namespace {

// The wheel scales the distance to the point of interest by e^(dy * this): a browser notch (100) is
// about 10%, the same at any frame size, and a touchpad's small steps add up to as much.
constexpr double kWheelPerPixel = 0.001;
// The closest the wheel brings the eye to the point of interest, in scene units (metres).
constexpr double kMinViewDistance = 0.01;
// Flying crosses the scene's diagonal (at least this many metres) in kFlySecondsAcross; Shift is
// kFlyFast times faster, and each wheel notch with the right button held kFlyWheelStep times.
constexpr double kMinFlySize       = 1.0;
constexpr double kFlySecondsAcross = 2.0;
constexpr double kFlyFast          = 4.0;
constexpr double kFlyWheelStep     = 1.25;
// The app sends the held keys each of its frames: silence this long is a release that got lost.
constexpr auto kFlyTimeout = std::chrono::milliseconds(250);
// The right button's look: a half turn across the frame's width, at the same rate up and down; it
// follows the hand with this time constant (applyLook).
constexpr double kLookTurnsPerWidth = 0.5;
constexpr double kLookSmoothing     = 0.025;

struct ViewFrame
{
  glm::dvec3 forward, right, up;
  double     distance;
};

// The view's axes; nullopt when the eye sits on the point of interest.
std::optional<ViewFrame> viewFrame(const glm::dvec3& eye, const glm::dvec3& center, const glm::dvec3& up)
{
  const double distance = glm::length(center - eye);
  if(distance < 1e-9)
    return std::nullopt;
  const glm::dvec3 forward = (center - eye) / distance;
  glm::dvec3       right   = glm::cross(forward, up);
  // looking straight along up (the manipulator keeps a drag off it, a camera.goto may not)
  right = glm::length(right) < 1e-9 ? glm::dvec3(1, 0, 0) : glm::normalize(right);
  return ViewFrame{forward, right, glm::cross(right, forward), distance};
}

}  // namespace

void FvHost::Impl::applyInput(const std::vector<Json>& events)
{
  auto manip = renderer->getCameraManipulator();
  if(!manip)
    return;
  // looking through a locked camera, the view stays put
  const auto camera = findProp(viewCamera);
  const bool locked = camera && camera->camera.locked;
  for(const Json& e : events)
  {
    const std::string type = e.value("t", "");
    glm::dvec3        flyMove{0.0};
    if(type == "fly" && e.contains("m") && e["m"].is_array() && e["m"].size() >= 3)
      flyMove = {e["m"][0].get<double>(), e["m"][1].get<double>(), e["m"][2].get<double>()};
    // the wheel with the right button held sets the flying speed, which moves nothing
    const bool moves = (type == "move" && (lmb || mmb || rmb)) || (type == "wheel" && !rmb) || flyMove != glm::dvec3(0.0);
    if(locked && moves)
    {
      if(!lockedNoticed)
        notice(camera->name + " is locked: unlock it in the Scene tab to move it, or go back to the free view.", "info");
      lockedNoticed = true;
      continue;
    }
    const double x = e.value("x", mouseX), y = e.value("y", mouseY);
    // left orbits, middle pans, right looks around and flies with the movement keys (applyFly), the
    // wheel dollies; the modifiers pick the left and middle drags' variants from camera_controls'
    // table, whose manipulator this is
    nvutils::CameraManipulator::Inputs inputs{.lmb = lmb, .mmb = mmb, .shift = e.value("shift", false), .ctrl = e.value("ctrl", false),
                                              .alt = e.value("alt", false)};
    if(type == "down" || type == "up")
    {
      // each button on its own: the app reports a second button pressed or released during a drag
      const int  b    = e.value("b", 0);
      const bool down = type == "down";
      (b == 0 ? lmb : b == 2 ? rmb : mmb) = down;
      if(down)
      {
        mouseX = x;
        mouseY = y;
        manip->setMousePosition({x, y});
      }
      else
      {
        if(b == 2)
        {
          fly.move = {};
          // the rest of the look is the gesture's, which ends here
          turnLook(std::exchange(lookPending, glm::dvec2(0.0)));
        }
        if(!lmb && !mmb && !rmb)
          endViewGesture();
      }
    }
    else if(type == "move")
    {
      manip->setMousePosition({mouseX, mouseY});
      if(rmb)
        lookPending += glm::dvec2(x - mouseX, y - mouseY);
      else if(lmb || mmb)
        manip->mouseMove({x, y}, inputs);
      mouseX = x;
      mouseY = y;
    }
    else if(type == "wheel")
    {
      if(rmb)
        fly.scale = std::clamp(fly.scale * std::pow(kFlyWheelStep, -e.value("dy", 0.0) / 100.0), 1.0 / 64, 64.0);
      else
        wheelDolly(e);
    }
    else if(type == "fly")
    {
      fly.move  = flyMove;
      fly.fast  = e.value("fast", false);
      fly.heard = std::chrono::steady_clock::now();
    }
    else if(type == "leave" && (lmb || mmb || rmb))
    {
      lmb = mmb = rmb = false;
      fly.move        = {};
      turnLook(std::exchange(lookPending, glm::dvec2(0.0)));
      endViewGesture();
    }
  }
}

// The wheel dollies toward the point of interest, scaling the distance to it, and stops short of it.
// With Shift it dollies toward the cursor instead: the point under it at the depth of the point of
// interest stays under it, and the point of interest moves along. The manipulator's own wheel is
// not used: its step goes with the square of a step over the frame's width (a touchpad's small
// steps barely move it, a 4K frame half as much as a 2K one), and its Shift and Alt variants are
// the ImGui viewport's (a pick at ImGui's cursor; a field of view the lens sets back).
void FvHost::Impl::wheelDolly(const Json& e)
{
  auto         manip = renderer->getCameraManipulator();
  const double dy    = e.value("dy", 0.0);
  if(!manip || dy == 0)
    return;
  if(manip->getProjectionType() == nvutils::CameraManipulator::Orthographic)
  {
    // a browser wheel notch is about 100 (down positive); ImGui's is 1 (up positive)
    manip->wheel(-dy / 100.0 * nvapp::ElementCamera::wheelSpeed, {});
    return;
  }
  glm::dvec3 eye, center, up;
  manip->getLookat(eye, center, up);
  const std::optional<ViewFrame> frame = viewFrame(eye, center, up);
  if(!frame)
    return;
  const double distance = frame->distance;
  const double factor   = std::max(distance * std::exp(dy * kWheelPerPixel), std::min(distance, kMinViewDistance)) / distance;
  if(factor == 1.0)
    return;
  glm::dvec3       pivot = center;
  const glm::uvec2 size  = manip->getWindowSize();
  if(e.value("shift", false) && size.x > 0 && size.y > 0 && e.contains("x") && e.contains("y"))
  {
    const double halfHeight = std::tan(glm::radians(manip->getFov()) * 0.5) * distance;
    const double halfWidth  = halfHeight * double(size.x) / double(size.y);
    pivot += frame->right * ((2.0 * e["x"].get<double>() / size.x - 1.0) * halfWidth)
             + frame->up * ((1.0 - 2.0 * e["y"].get<double>() / size.y) * halfHeight);
  }
  manip->setLookat(pivot + (eye - pivot) * factor, pivot + (center - pivot) * factor, up, true);
}

// The right button's look, spread over the frames. The app sends its moves once per its own frame,
// and they land unevenly among the renderer's (two batches in one frame, none in the next): applied
// as they came, a turn of the whole view stutters. Each frame takes a share of what is left to turn
// that grows with the frame's time, so the view follows the hand a few tens of milliseconds behind,
// evenly. Once a frame, after the input.
void FvHost::Impl::applyLook()
{
  const auto now = std::chrono::steady_clock::now();
  // the time since the last frame, which a still spell doesn't count; a long frame not all of it
  const double seconds = lookPending == glm::dvec2(0.0) ? 0.0 : std::min(std::chrono::duration<double>(now - lookLast).count(), 0.1);
  lookLast             = now;
  if(seconds <= 0)
    return;
  glm::dvec2 step = lookPending * (1.0 - std::exp(-seconds / kLookSmoothing));
  // the last fraction of a pixel at once, so the turn ends
  if(glm::length(lookPending - step) < 0.05)
    step = lookPending;
  lookPending -= step;
  turnLook(step);
}

void FvHost::Impl::turnLook(const glm::dvec2& pixels)
{
  auto manip = renderer->getCameraManipulator();
  if(!manip || pixels == glm::dvec2(0.0))
    return;
  // the manipulator turns a full turn per window width across and per window height up and down:
  // both scaled to kLookTurnsPerWidth of the width
  const glm::dvec2 size(manip->getWindowSize());
  manip->setMousePosition({0, 0});
  manip->motion({pixels.x * kLookTurnsPerWidth, pixels.y * kLookTurnsPerWidth * size.y / size.x},
                nvutils::CameraManipulator::Actions::LookAround);
}

// Flying: while the right button is held, the movement keys move the view at a steady speed for the
// time since the last frame: W and S along where it looks, A and D to its sides, E and Q along the
// world's up. Once a frame, after the input.
void FvHost::Impl::applyFly()
{
  const auto now    = std::chrono::steady_clock::now();
  auto       manip  = renderer->getCameraManipulator();
  const auto camera = findProp(viewCamera);
  if(!manip || !rmb || fly.move == glm::dvec3(0.0) || now - fly.heard > kFlyTimeout || (camera && camera->camera.locked))
  {
    fly.flying = false;
    return;
  }
  if(!fly.flying)
  {
    // a flight starts: its speed from the scene's size, and its time from this frame
    const nvutils::Bbox box = renderer->visibleSceneBounds();
    fly.speed               = std::max(box.isEmpty() ? 0.0 : 2.0 * box.radius(), kMinFlySize) / kFlySecondsAcross;
    fly.last                = now;
    fly.flying              = true;
  }
  // a frame that took long (a load, a compose) moves as far as a short one, not a jump
  const double seconds = std::min(std::chrono::duration<double>(now - fly.last).count(), 0.1);
  fly.last             = now;
  glm::dvec3 eye, center, up;
  manip->getLookat(eye, center, up);
  const std::optional<ViewFrame> frame = viewFrame(eye, center, up);
  if(!frame || seconds <= 0)
    return;
  glm::dvec3 direction = frame->right * fly.move.x + glm::normalize(up) * fly.move.y + frame->forward * fly.move.z;
  // diagonally no faster than straight
  if(glm::length(direction) > 1.0)
    direction = glm::normalize(direction);
  const glm::dvec3 step = direction * (fly.speed * fly.scale * (fly.fast ? kFlyFast : 1.0) * seconds);
  manip->setLookat(eye + step, center + step, up, true);
}
