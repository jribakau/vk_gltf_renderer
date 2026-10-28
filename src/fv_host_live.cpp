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

void FvHost::Impl::applyNodes(Loaded& f, const std::set<int>& moved)
{
  if(moved.empty())
    return;
  std::vector<GltfRenderer::NodeTransform> transforms;
  transforms.reserve(moved.size());
  const fv::SceneGraph& scene = f.fig->loaded.scene;
  for(int n : moved)
  {
    const int g = n >= 0 && n < int(f.nodeOf.size()) ? f.nodeOf[n] : -1;
    if(g < 0 || n >= int(scene.nodes.size()))
      continue;
    const fv::Node& node = scene.nodes[n];
    transforms.push_back({g, glm::vec3(node.position.x, node.position.y, node.position.z),
                          glm::quat(float(node.quaternion.w), float(node.quaternion.x), float(node.quaternion.y), float(node.quaternion.z)),
                          glm::vec3(node.scale.x, node.scale.y, node.scale.z)});
  }
  if(!transforms.empty())
    renderer->setNodeTransforms(transforms);
}

void FvHost::Impl::applyMesh(Loaded& f, const fv::morph::MeshChange& change)
{
  if(change.mesh < 0 || change.mesh >= int(f.fig->loaded.scene.meshes.size()) || change.mesh >= int(f.primsOf.size()) || f.primsOf[change.mesh].empty())
    return;
  const auto& geo = f.fig->loaded.scene.meshes[change.mesh].geo;
  if(!geo)
    return;
  const size_t count = geo->position.size() / 3;
  const bool   normals = geo->normal.size() == geo->position.size();
  // glm::vec3 is the tightly packed 12-byte attribute the renderer stages
  static_assert(sizeof(glm::vec3) == 12, "vertex uploads assume a packed vec3");

  GltfRenderer::VertexUpload upload;
  if(change.all)
  {
    upload.positions.resize(count);
    if(count)
      std::memcpy(upload.positions.data(), geo->position.data(), count * sizeof(glm::vec3));
    if(normals)
    {
      upload.normals.resize(count);
      if(count)
        std::memcpy(upload.normals.data(), geo->normal.data(), count * sizeof(glm::vec3));
    }
  }
  else
  {
    if(change.indices.empty())
      return;
    upload.indices.reserve(change.indices.size());
    upload.positions.reserve(change.indices.size());
    if(normals)
      upload.normals.reserve(change.indices.size());
    for(uint32_t v : change.indices)
    {
      if(v >= count)
        continue;
      const float* p = geo->position.data() + size_t(v) * 3;
      upload.indices.push_back(v);
      upload.positions.emplace_back(p[0], p[1], p[2]);
      if(normals)
      {
        const float* nrm = geo->normal.data() + size_t(v) * 3;
        upload.normals.emplace_back(nrm[0], nrm[1], nrm[2]);
      }
    }
    if(upload.indices.empty())
      return;
  }
  dialTiming.addUpload(change.all ? count : upload.indices.size(), change.all);
  const auto& prims = f.primsOf[change.mesh];
  for(size_t i = 0; i < prims.size(); i++)
  {
    GltfRenderer::VertexUpload one = i + 1 == prims.size() ? std::move(upload) : upload;
    one.renderPrimID               = prims[i];
    renderer->queueVertexUpload(std::move(one));
  }
}

void FvHost::Impl::applyRebind(Loaded& f, const std::vector<int>& meshes)
{
  for(int mesh : meshes)
  {
    if(mesh < 0 || mesh >= int(f.skinOf.size()) || mesh >= int(f.fig->loaded.scene.meshes.size()))
      continue;
    const int skin = f.skinOf[mesh];
    if(skin < 0)
      continue;
    const fv::Mesh&          m   = f.fig->loaded.scene.meshes[mesh];
    const std::vector<float> ibm = fv::build::inverseBindMatrices(m);
    if(ibm.size() != m.bones.size() * 16)
      continue;
    static_assert(sizeof(glm::mat4) == 64, "inverse bind matrices are packed mat4");
    std::vector<glm::mat4> mats(m.bones.size());
    std::memcpy(mats.data(), ibm.data(), ibm.size() * sizeof(float));
    renderer->setInverseBindMatrices(skin, mats);
  }
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
  {
    std::lock_guard lock(morphMutex);
    pending.swap(pendingMorphs);
  }
  for(const auto& [id, dials] : pending)
  {
    const auto p = findProp(id);
    if(!p || !p->figure || !p->figure->fig->morphs || dials.empty())
      continue;
    std::vector<std::pair<std::string, double>> values(dials.begin(), dials.end());
    const auto                                  t0 = std::chrono::steady_clock::now();
    p->figure->fig->morphs->setMany(values);
    dialTiming.add(DialTiming::eSetMany, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
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
  nvvkgltf::Scene* scene = renderer->editScene();
  auto             manip = renderer->getCameraManipulator();
  if(!scene || !manip)
    return;
  const nvutils::Bbox box  = renderer->visibleSceneBounds();
  const VkExtent2D    size = renderer->renderSize();
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
    if(locked && ((type == "move" && (lmb || mmb)) || type == "wheel"))
    {
      if(!lockedNoticed)
        notice(camera->name + " is locked: unlock it in the Scene tab to move it, or go back to the free view.", "info");
      lockedNoticed = true;
      continue;
    }
    const double      x = e.value("x", mouseX), y = e.value("y", mouseY);
    // the old viewer's mapping: left orbits, middle and right pan, the wheel dollies
    nvutils::CameraManipulator::Inputs inputs{.lmb = lmb, .mmb = mmb, .shift = e.value("shift", false), .ctrl = e.value("ctrl", false),
                                              .alt = e.value("alt", false)};
    if(type == "down")
    {
      const int b = e.value("b", 0);
      if(b == 0)
        lmb = true;
      else
        mmb = true;
      mouseX = x;
      mouseY = y;
      manip->setMousePosition({x, y});
    }
    else if(type == "up")
    {
      const int b = e.value("b", 0);
      if(b == 0)
        lmb = false;
      else
        mmb = false;
    }
    else if(type == "move")
    {
      if(lmb || mmb)
      {
        manip->setMousePosition({mouseX, mouseY});
        manip->mouseMove({x, y}, inputs);
      }
      mouseX = x;
      mouseY = y;
    }
    else if(type == "wheel")
    {
      // a browser wheel notch is about 100 (down positive); ImGui's is 1 (up positive)
      manip->wheel(-e.value("dy", 0.0) / 100.0 * nvapp::ElementCamera::wheelSpeed, inputs);
    }
    else if(type == "leave")
      lmb = mmb = false;
  }
}
