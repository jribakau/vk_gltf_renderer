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

#include "fv_view.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include <glm/glm.hpp>
#include <nvutils/camera_manipulator.hpp>

#include <figure/scene/graph.hpp>

#include "renderer.hpp"
#include "shaders/shaderio.h"

namespace {

double fovForFocalLength(double mm, double aspect)
{
  mm                = std::max(mm, 1.0);
  const double film = 35.0 / std::max(aspect, 1.0);
  return 2.0 * std::atan(film / 2.0 / mm) * (180.0 / 3.14159265358979323846);
}

// Camera direction from the target. Figures face +Z, so front looks along +Z. A hair off the pole
// keeps a top or bottom view from looking straight along the up axis.
const std::unordered_map<std::string, glm::dvec3>& viewDirections()
{
  static const std::unordered_map<std::string, glm::dvec3> dirs{
      {"front", {0, 0, 1}}, {"back", {0, 0, -1}}, {"right", {1, 0, 0}}, {"left", {-1, 0, 0}},
      {"top", {0, 1, 1e-4}}, {"bottom", {0, -1, 1e-4}}, {"threeQuarter", {0.35, 0.18, 1}},
  };
  return dirs;
}

fv::Json vec3(const glm::dvec3& v)
{
  return fv::Json::array({v.x, v.y, v.z});
}

glm::dvec3 vec3Of(const fv::Json& v)
{
  if(!v.is_array() || v.size() < 3)
    throw std::runtime_error("expected a [x, y, z] vector");
  return {v[0].get<double>(), v[1].get<double>(), v[2].get<double>()};
}

}  // namespace

FvView::FvView(GltfRenderer* renderer, fv::EventHub& events, fv::CommandRegistry& commands, std::function<fv::Json(std::function<fv::Json()>)> onApp)
    : m_renderer(renderer)
    , m_events(&events)
    , m_onApp(std::move(onApp))
{
  m_renderer->setPickObserver([this](int renderNode, const glm::vec3& hit) {
    if(!m_pickBones || renderNode < 0 || !m_bones.nearest)
      return false;
    const int node = m_bones.nearest(hit);
    const int gltf = node >= 0 && node < int(m_gltfOf.size()) ? m_gltfOf[node] : -1;
    if(gltf < 0)
      return false;
    m_renderer->selectSceneNodeFromScript(gltf);
    return true;
  });
  commands.add("select", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      if(!args.contains("node") || args["node"].is_null())
        m_renderer->selectSceneNodeFromScript(-1);
      else
      {
        const int node = args["node"].get<int>();
        const int gltf = node >= 0 && node < int(m_gltfOf.size()) ? m_gltfOf[node] : -1;
        if(gltf < 0)
          throw std::runtime_error("no such node");
        m_renderer->selectSceneNodeFromScript(gltf);
      }
      publishSelection(true);
      return fv::Json{{"ok", true}};
    });
  });
  commands.add("camera.view", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      auto        manip = m_renderer->getCameraManipulator();
      const auto& dirs  = viewDirections();
      const auto  it    = dirs.find(args.value("name", ""));
      if(!manip || it == dirs.end())
        throw std::runtime_error("view is front, back, left, right, top, bottom or threeQuarter");
      glm::dvec3 eye, center, up;
      manip->getLookat(eye, center, up);
      double dist = glm::length(eye - center);
      if(dist < 1e-4)
        dist = 1;
      const glm::dvec3 dir = glm::normalize(it->second);
      manip->setLookat(center + dir * dist, center, {0, 1, 0}, true);
      return fv::Json{{"ok", true}};
    });
  });
  commands.add("camera.frame", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      m_renderer->fitCamera(!args.value("selected", false));
      return fv::Json{{"ok", true}};
    });
  });
  commands.add("camera.set", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      if(args.contains("focalLength"))
        m_focal = args["focalLength"].get<double>();
      if(args.contains("dof"))
        m_dof = args["dof"].get<bool>();
      if(args.contains("fStop"))
        m_fStop = args["fStop"].get<double>();
      if(args.contains("focusMode"))
        m_focus = args["focusMode"].get<std::string>();
      if(args.contains("focusDistance"))
        m_focusDistance = args["focusDistance"].get<double>();
      m_aspect = -1;
      applyLens();
      applyDof();
      return fv::Json{{"ok", true}};
    });
  });
  commands.add("camera.get", [this](const fv::Json&) {
    return m_onApp([this] {
      auto manip = m_renderer->getCameraManipulator();
      if(!manip)
        throw std::runtime_error("no camera");
      glm::dvec3 eye, center, up;
      manip->getLookat(eye, center, up);
      return fv::Json{{"position", vec3(eye)}, {"target", vec3(center)}, {"up", vec3(up)}, {"focalLength", m_focal}};
    });
  });
  commands.add("camera.goto", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      auto manip = m_renderer->getCameraManipulator();
      if(!manip)
        throw std::runtime_error("no camera");
      if(args.contains("focalLength"))
        m_focal = args["focalLength"].get<double>();
      manip->setLookat(vec3Of(args.at("position")), vec3Of(args.at("target")), {0, 1, 0}, true);
      m_aspect = -1;
      applyLens();
      return fv::Json{{"ok", true}};
    });
  });
  commands.add("view.set", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      if(args.contains("display"))
      {
        const std::string display = args["display"].get<std::string>();
        if(display == "clay")
        {
          m_renderer->setVisualization(shaderio::Visualization::eClay);
          m_renderer->setWireframe(false);
        }
        else if(display == "normals")
        {
          m_renderer->setVisualization(shaderio::Visualization::eNormalShading);
          m_renderer->setWireframe(false);
        }
        else if(display == "wireframe")
        {
          m_renderer->setVisualization(shaderio::Visualization::eRendered);
          m_renderer->setWireframe(true);
        }
        else if(display == "shaded")
        {
          m_renderer->setVisualization(shaderio::Visualization::eRendered);
          m_renderer->setWireframe(false);
        }
        else
          throw std::runtime_error("display is shaded, clay, wireframe or normals");
      }
      if(args.contains("grid"))
        m_renderer->setShowGrid(args["grid"].get<bool>());
      if(args.contains("gizmo"))
        m_renderer->setShowAxis(args["gizmo"].get<bool>());
      if(args.contains("pick"))
      {
        const std::string pick = args["pick"].get<std::string>();
        if(pick != "bone" && pick != "mesh")
          throw std::runtime_error("pick is bone or mesh");
        m_pickBones = pick == "bone";
        m_renderer->setShowGizmo(m_pickBones);
      }
      return fv::Json{{"ok", true}};
    });
  });
}

void FvView::setFigure(const std::string& id, const std::string& name, const fv::SceneGraph& scene, int root,
                       const std::vector<int>& nodeOf, int morphs, Bones posable)
{
  m_id     = id;
  m_gltfOf = nodeOf;
  m_bones  = std::move(posable);
  m_figureOf.clear();
  for(int i = 0; i < int(nodeOf.size()); i++)
    if(nodeOf[i] >= 0)
      m_figureOf[nodeOf[i]] = i;
  m_nodes.assign(scene.nodes.size(), {});
  fv::Json nodes = fv::Json::object();
  int      order = 0, bones = 0, triangles = 0;
  const std::function<void(int, int)> walk = [&](int node, int depth) {
    if(node < 0 || node >= int(scene.nodes.size()))
      return;
    const fv::Node& n = scene.nodes[node];
    std::string     kind = "group";
    if(n.kind == fv::NodeKind::eBone)
    {
      kind = "bone";
      bones++;
    }
    else if(n.kind == fv::NodeKind::eMesh)
      kind = n.mesh >= 0 && n.mesh < int(scene.meshes.size()) && scene.meshes[n.mesh].skinned() ? "skinned" : "mesh";
    m_nodes[node] = {n.name.empty() ? kind : n.name, kind, n.kind == fv::NodeKind::eMesh ? n.mesh : -1};
    nodes[std::to_string(node)] = {{"name", m_nodes[node].name}, {"parent", n.parent}, {"kind", kind}, {"order", order++}, {"depth", depth}};
    for(int child : n.children)
      walk(child, depth + 1);
  };
  if(root >= 0)
    walk(root, 0);
  for(const fv::Mesh& mesh : scene.meshes)
    if(mesh.geo)
      triangles += int((mesh.geo->fullIndex.empty() ? mesh.geo->renderTris.size() : mesh.geo->fullIndex.size()) / 3);
  m_events->snapshot("scene", {{"figure", id},
                               {"name", name},
                               {"figures", {{id, {{"name", name}}}}},
                               {"nodes", nodes},
                               {"stats", {{"meshes", scene.meshes.size()},
                                          {"triangles", triangles},
                                          {"materials", scene.materials.size()},
                                          {"bones", bones},
                                          {"morphs", morphs}}}});
  m_lastGltf = -2;
  publishSelection(true);
}

void FvView::clearFigure()
{
  m_id.clear();
  m_gltfOf.clear();
  m_figureOf.clear();
  m_nodes.clear();
  m_bones    = {};
  m_lastGltf = -1;
  m_events->snapshot("scene", {{"figures", fv::Json::object()}});
  m_events->snapshot("selection", {{"node", nullptr}});
}

std::vector<fv::Json> FvView::onInput(const std::vector<fv::Json>& events)
{
  std::vector<fv::Json> rest;
  rest.reserve(events.size());
  for(const fv::Json& e : events)
  {
    const std::string type = e.value("t", "");
    const double      x = e.value("x", m_x0), y = e.value("y", m_y0);
    const bool        left     = e.value("b", 0) == 0;
    const bool        pressed  = type == "down" && left;
    const bool        released = (type == "up" && left) || (type == "leave" && m_lmb);
    if(pressed || released)
      m_lmb = pressed;
    // a press on a handle, its drag and its release are the gizmo's: no orbit, no pick
    if((type == "down" || type == "up" || type == "move" || type == "leave")
       && m_renderer->gizmoInput({float(x), float(y)}, m_lmb, pressed, released))
    {
      m_click = false;
      continue;
    }
    rest.push_back(e);
    if(pressed)
    {
      m_click = true;
      m_x0    = x;
      m_y0    = y;
    }
    else if(type == "up" && e.value("b", 0) == 0)
    {
      const bool click = m_click && std::hypot(x - m_x0, y - m_y0) < 4;
      m_click          = false;
      if(!click)
        continue;
      const VkExtent2D size = m_renderer->renderSize();
      if(size.width == 0 || size.height == 0)
        continue;
      m_renderer->pickAt({float(x / size.width), float(y / size.height)}, false);
    }
    else if(type == "leave")
      m_click = false;
  }
  return rest;
}

void FvView::onFrame()
{
  applyLens();
  publishSelection(false);
  publishCamera();
}

void FvView::publishCamera()
{
  auto manip = m_renderer->getCameraManipulator();
  const auto now = std::chrono::steady_clock::now();
  if(!manip || now - m_cameraAt < std::chrono::milliseconds(33))
    return;
  glm::dvec3 eye, center, up;
  manip->getLookat(eye, center, up);
  fv::Json camera{{"position", vec3(eye)}, {"target", vec3(center)}, {"up", vec3(up)}, {"focalLength", m_focal}};
  if(camera == m_camera)
    return;
  m_camera   = std::move(camera);
  m_cameraAt = now;
  m_events->snapshot("camera", m_camera);
}

void FvView::applyLens()
{
  auto manip = m_renderer->getCameraManipulator();
  if(!manip || m_focal <= 0)
    return;
  const glm::uvec2 size = manip->getWindowSize();
  if(size.y == 0)
    return;
  // the renderer sets up its own camera for a loaded scene, so check the camera's fov, not just the aspect
  const double aspect = double(size.x) / double(size.y);
  const double fov    = fovForFocalLength(m_focal, aspect);
  if(std::abs(aspect - m_aspect) < 1e-4 && std::abs(manip->getFov() - fov) < 1e-4)
    return;
  manip->setFov(fov);
  m_aspect = aspect;
}

void FvView::applyDof()
{
  float aperture = 0;
  if(m_dof && m_fStop > 0)
  {
    const double radius = (m_focal / (2.0 * m_fStop)) * 1e-3;
    aperture            = float(radius * radius);
  }
  m_renderer->setDepthOfField(float(std::max(m_focusDistance, 1e-3)), aperture, m_focus != "manual");
}

void FvView::publishSelection(bool force)
{
  const int gltf = m_renderer->selectedNodeIndex();
  if(!force && gltf == m_lastGltf)
    return;
  m_lastGltf = gltf;
  const auto it = m_figureOf.find(gltf);
  if(gltf < 0 || it == m_figureOf.end() || it->second < 0 || it->second >= int(m_nodes.size()))
  {
    m_events->snapshot("selection", {{"node", nullptr}});
    return;
  }
  const NodeRec& n   = m_nodes[it->second];
  fv::Json       sel = {{"figure", m_id}, {"node", it->second}, {"name", n.name}, {"mesh", n.mesh}, {"kind", n.kind}};
  if(const auto bone = m_bones.idOf.find(it->second); bone != m_bones.idOf.end())
    sel["bone"] = bone->second;
  m_events->snapshot("selection", sel);
}
