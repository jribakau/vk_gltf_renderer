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

#include <tinygltf/tiny_gltf.h>

#include <algorithm>
#include <cmath>
#include <set>
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
    if(renderNode < 0 || m_pick == "mesh")
      return false;
    nvvkgltf::Scene* scene = m_renderer->editScene();
    if(!scene || !scene->valid() || renderNode >= int(scene->getRenderNodes().size()))
      return false;
    const auto [prop, node] = locate(scene->getRenderNodes()[size_t(renderNode)].refNodeID);
    if(prop < 0)
      return false;
    const PropNodes& p = m_props[size_t(prop)];
    if(m_pick == "bone")
    {
      // the posed bone nearest the hit, of the figure hit; a prop without bones is selected whole
      const int bone = p.bones.nearest ? p.bones.nearest(hit) : -1;
      const int gltf = bone >= 0 && bone < int(p.gltfOf.size()) ? p.gltfOf[size_t(bone)] : -1;
      m_renderer->selectSceneNodeFromScript(gltf >= 0 ? gltf : p.root);
      return true;
    }
    m_renderer->selectSceneNodeFromScript(p.root);
    return true;
  });
  commands.add("select", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      const std::string prop = args.contains("prop") && args["prop"].is_string() ? args["prop"].get<std::string>() : "";
      if(prop.empty())
        m_renderer->selectSceneNodeFromScript(-1);
      else
      {
        const auto it = std::find_if(m_props.begin(), m_props.end(), [&](const PropNodes& p) { return p.id == prop; });
        if(it == m_props.end())
          throw std::runtime_error("no such prop");
        int gltf = it->root;
        if(args.contains("node") && !args["node"].is_null())
        {
          const int node = args["node"].get<int>();
          gltf           = node >= 0 && node < int(it->gltfOf.size()) ? it->gltfOf[size_t(node)] : -1;
          if(gltf < 0)
            throw std::runtime_error("no such node");
        }
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
      if(args.contains("prop") && args["prop"].is_string())
      {
        // the prop's subtree, by selecting it for the fit and putting the selection back
        const auto it = std::find_if(m_props.begin(), m_props.end(), [&](const PropNodes& p) { return p.id == args["prop"].get<std::string>(); });
        if(it == m_props.end())
          throw std::runtime_error("no such prop");
        const int was = m_renderer->selectedNodeIndex();
        m_renderer->selectSceneNodeFromScript(it->root);
        m_renderer->fitCamera(false);
        m_renderer->selectSceneNodeFromScript(was);
      }
      else
        m_renderer->fitCamera(!args.value("selected", false));
      return fv::Json{{"ok", true}};
    });
  });
  commands.add("camera.set", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      setLens(lensOf(args, m_lens));
      if(m_onLens)
        m_onLens(m_lens);
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
      return fv::Json{{"position", vec3(eye)}, {"target", vec3(center)}, {"up", vec3(up)}, {"focalLength", m_lens.focalLength}};
    });
  });
  commands.add("camera.goto", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      auto manip = m_renderer->getCameraManipulator();
      if(!manip)
        throw std::runtime_error("no camera");
      if(args.contains("focalLength"))
      {
        Lens lens        = m_lens;
        lens.focalLength = args["focalLength"].get<double>();
        setLens(lens);
        if(m_onLens)
          m_onLens(m_lens);
      }
      const glm::dvec3 up = args.contains("up") ? vec3Of(args["up"]) : glm::dvec3(0, 1, 0);
      manip->setLookat(vec3Of(args.at("position")), vec3Of(args.at("target")), up, true);
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
        if(pick != "object" && pick != "bone" && pick != "mesh")
          throw std::runtime_error("pick is object, mesh or bone");
        m_pick = pick;
        // the gizmo moves props and poses bones; a mesh pick only shows what was hit
        m_renderer->setShowGizmo(m_pick != "mesh");
      }
      if(args.contains("helpers"))
        m_helpers = args["helpers"].get<bool>();
      if(args.contains("transform"))
      {
        const std::string t = args["transform"].get<std::string>();
        if(t != "move" && t != "rotate" && t != "scale")
          throw std::runtime_error("transform is move, rotate or scale");
        m_transform = t;
      }
      return fv::Json{{"ok", true}};
    });
  });
  m_renderer->setShowGizmo(true);
}

fv::Json FvView::describeFigure(PropNodes& out, const fv::SceneGraph& scene, int root, const std::vector<int>& nodeOf, int morphs)
{
  out.gltfOf = nodeOf;
  out.nodes.assign(scene.nodes.size(), {});
  fv::Json nodes = fv::Json::object();
  int      order = 0, bones = 0, triangles = 0;
  const std::function<void(int, int)> walk = [&](int node, int depth) {
    if(node < 0 || node >= int(scene.nodes.size()))
      return;
    const fv::Node& n    = scene.nodes[node];
    std::string     kind = "group";
    if(n.kind == fv::NodeKind::eBone)
    {
      kind = "bone";
      bones++;
    }
    else if(n.kind == fv::NodeKind::eMesh)
      kind = n.mesh >= 0 && n.mesh < int(scene.meshes.size()) && scene.meshes[n.mesh].skinned() ? "skinned" : "mesh";
    out.nodes[node] = {n.name.empty() ? kind : n.name, kind, n.kind == fv::NodeKind::eMesh ? n.mesh : -1};
    nodes[std::to_string(node)] = {{"name", out.nodes[node].name}, {"parent", n.parent}, {"kind", kind}, {"order", order++}, {"depth", depth}};
    for(int child : n.children)
      walk(child, depth + 1);
  };
  if(root >= 0)
    walk(root, 0);
  for(const fv::Mesh& mesh : scene.meshes)
    if(mesh.geo)
      triangles += int((mesh.geo->fullIndex.empty() ? mesh.geo->renderTris.size() : mesh.geo->fullIndex.size()) / 3);
  return {{"nodes", nodes},
          {"stats", {{"meshes", scene.meshes.size()}, {"triangles", triangles}, {"materials", scene.materials.size()}, {"bones", bones}, {"morphs", morphs}}}};
}

fv::Json FvView::describeModel(PropNodes& out, const tinygltf::Model& model, int root, int firstNode, int lastNode, int meshBase)
{
  const int count = std::max(0, lastNode - firstNode);
  out.gltfOf.resize(size_t(count));
  for(int i = 0; i < count; i++)
    out.gltfOf[size_t(i)] = firstNode + i;
  out.bones = {};
  // parents as topic nodes: the prop root above the model's roots is no topic node
  std::vector<int> parent(size_t(count), -1);
  for(int i = 0; i < count; i++)
    for(int c : model.nodes[size_t(firstNode + i)].children)
      if(c >= firstNode && c < lastNode)
        parent[size_t(c - firstNode)] = i;
  out.nodes.assign(size_t(count), {});
  fv::Json          nodes     = fv::Json::object();
  int               order     = 0;
  size_t            triangles = 0;
  std::set<int>     meshes;
  std::set<int>     materials;
  const std::function<void(int, int)> walk = [&](int node, int depth) {
    if(node < 0 || node >= count || depth > 512)
      return;
    const tinygltf::Node& n     = model.nodes[size_t(firstNode + node)];
    std::string           kind  = n.mesh >= 0 ? (n.skin >= 0 ? "skinned" : "mesh") : n.camera >= 0 ? "camera" : n.light >= 0 ? "light" : "group";
    const std::string     label = n.name.empty() ? (n.mesh >= 0 && n.mesh < int(model.meshes.size()) && !model.meshes[size_t(n.mesh)].name.empty() ? model.meshes[size_t(n.mesh)].name : kind) : n.name;
    out.nodes[size_t(node)]     = {label, kind, n.mesh >= 0 ? n.mesh - meshBase : -1};
    nodes[std::to_string(node)] = {{"name", label}, {"parent", parent[size_t(node)]}, {"kind", kind}, {"order", order++}, {"depth", depth}};
    if(n.mesh >= 0 && n.mesh < int(model.meshes.size()) && meshes.insert(n.mesh).second)
      for(const tinygltf::Primitive& p : model.meshes[size_t(n.mesh)].primitives)
      {
        materials.insert(p.material);
        const int acc = p.indices >= 0 ? p.indices : (p.attributes.count("POSITION") ? p.attributes.at("POSITION") : -1);
        if(acc >= 0 && acc < int(model.accessors.size()))
          triangles += model.accessors[size_t(acc)].count / 3;
      }
    for(int c : n.children)
      walk(c - firstNode, depth + 1);
  };
  if(root >= 0 && root < int(model.nodes.size()))
    for(int c : model.nodes[size_t(root)].children)
      if(c >= firstNode && c < lastNode)
        walk(c - firstNode, 0);
  materials.erase(-1);
  return {{"nodes", nodes},
          {"stats", {{"meshes", meshes.size()}, {"triangles", triangles}, {"materials", materials.size()}, {"bones", 0}, {"morphs", 0}}}};
}

void FvView::setProps(std::vector<PropNodes> props)
{
  m_props = std::move(props);
  m_where.clear();
  for(int p = 0; p < int(m_props.size()); p++)
  {
    m_where[m_props[size_t(p)].root] = {p, -1};
    const auto& gltfOf = m_props[size_t(p)].gltfOf;
    for(int i = 0; i < int(gltfOf.size()); i++)
      if(gltfOf[size_t(i)] >= 0)
        m_where[gltfOf[size_t(i)]] = {p, i};
  }
  m_lastGltf = -2;
  publishSelection(true);
}

const FvView::PropNodes* FvView::propOfRoot(int gltf) const
{
  const auto it = m_where.find(gltf);
  return it == m_where.end() || it->second.second != -1 ? nullptr : &m_props[size_t(it->second.first)];
}

std::pair<int, int> FvView::locate(int gltf) const
{
  const auto it = m_where.find(gltf);
  return it == m_where.end() ? std::pair{-1, -1} : it->second;
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
      // a camera or light marker under the pointer is the pick; they have nothing the ray could hit
      if(m_helpers && m_helperPick)
        if(const std::string prop = m_helperPick({float(x), float(y)}); !prop.empty())
        {
          const auto it = std::find_if(m_props.begin(), m_props.end(), [&](const PropNodes& p) { return p.id == prop; });
          if(it != m_props.end())
          {
            m_renderer->selectSceneNodeFromScript(it->root);
            continue;
          }
        }
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
  fv::Json camera{{"position", vec3(eye)}, {"target", vec3(center)}, {"up", vec3(up)}, {"focalLength", m_lens.focalLength}};
  camera.update(m_cameraExtra);
  if(camera == m_camera)
    return;
  m_camera   = std::move(camera);
  m_cameraAt = now;
  m_events->snapshot("camera", m_camera);
}

void FvView::setCameraExtra(fv::Json extra)
{
  m_cameraExtra = extra.is_object() ? std::move(extra) : fv::Json::object();
  m_camera      = nullptr;
  m_cameraAt    = {};
}

fv::Json FvView::lensJson(const Lens& lens)
{
  return {{"focalLength", lens.focalLength}, {"dof", lens.dof}, {"fStop", lens.fStop}, {"focusMode", lens.focusMode}, {"focusDistance", lens.focusDistance}};
}

FvView::Lens FvView::lensOf(const fv::Json& args, Lens base)
{
  if(args.contains("focalLength"))
    base.focalLength = std::clamp(args["focalLength"].get<double>(), 1.0, 2000.0);
  if(args.contains("dof"))
    base.dof = args["dof"].get<bool>();
  if(args.contains("fStop"))
    base.fStop = std::max(args["fStop"].get<double>(), 0.1);
  if(args.contains("focusMode"))
  {
    base.focusMode = args["focusMode"].get<std::string>();
    if(base.focusMode != "target" && base.focusMode != "manual")
      throw std::runtime_error("focusMode is target or manual");
  }
  if(args.contains("focusDistance"))
    base.focusDistance = std::max(args["focusDistance"].get<double>(), 1e-3);
  return base;
}

void FvView::setLens(const Lens& lens)
{
  m_lens   = lens;
  m_aspect = -1;
  applyLens();
  applyDof();
}

void FvView::applyLens()
{
  auto manip = m_renderer->getCameraManipulator();
  if(!manip || m_lens.focalLength <= 0)
    return;
  const glm::uvec2 size = manip->getWindowSize();
  if(size.y == 0)
    return;
  // the renderer sets up its own camera for a loaded scene, so check the camera's fov, not just the aspect
  const double aspect = double(size.x) / double(size.y);
  const double fov    = fovForFocalLength(m_lens.focalLength, aspect);
  if(std::abs(aspect - m_aspect) < 1e-4 && std::abs(manip->getFov() - fov) < 1e-4)
    return;
  manip->setFov(fov);
  m_aspect = aspect;
}

void FvView::applyDof()
{
  float aperture = 0;
  if(m_lens.dof && m_lens.fStop > 0)
  {
    const double radius = (m_lens.focalLength / (2.0 * m_lens.fStop)) * 1e-3;
    aperture            = float(radius * radius);
  }
  m_renderer->setDepthOfField(float(std::max(m_lens.focusDistance, 1e-3)), aperture, m_lens.focusMode != "manual");
}

void FvView::deselect(const std::string& prop)
{
  if(prop.empty() || m_selectedProp != prop)
    return;
  m_renderer->selectSceneNodeFromScript(-1);
  publishSelection(true);
}

void FvView::publishSelection(bool force)
{
  const int gltf = m_renderer->selectedNodeIndex();
  if(!force && gltf == m_lastGltf)
    return;
  m_lastGltf                 = gltf;
  const auto [prop, node] = locate(gltf);
  if(gltf < 0 || prop < 0)
  {
    m_selectedProp.clear();
    m_events->snapshot("selection", {{"prop", nullptr}, {"node", nullptr}});
    return;
  }
  const PropNodes& p = m_props[size_t(prop)];
  m_selectedProp     = p.id;
  if(node < 0 || node >= int(p.nodes.size()))
  {
    m_events->snapshot("selection", {{"prop", p.id}, {"node", nullptr}});
    return;
  }
  const NodeRec& n   = p.nodes[size_t(node)];
  fv::Json       sel = {{"prop", p.id}, {"node", node}, {"name", n.name}, {"mesh", n.mesh}, {"kind", n.kind}};
  if(const auto bone = p.bones.idOf.find(node); bone != p.bones.idOf.end())
    sel["bone"] = bone->second;
  m_events->snapshot("selection", sel);
}
