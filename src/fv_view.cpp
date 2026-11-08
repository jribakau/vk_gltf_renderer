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
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <glm/glm.hpp>
#include <nvutils/camera_manipulator.hpp>

#include <figure/scene/graph.hpp>

#include "host_renderer.hpp"
#include "pixel_time.hpp"
#include "shaders/shaderio.h"

double fovForFrame(double mm, double viewAspect, double frameAspect)
{
  mm = std::max(mm, 1.0);
  // tangents of the half fields: the long side's is 17.5 mm over the focal length
  const auto halfY = [&](double aspect) { return 17.5 / mm / std::max(aspect, 1.0); };
  double     tanY  = halfY(viewAspect);
  if(frameAspect > 0 && viewAspect > 0)
  {
    const double frameY = halfY(frameAspect), frameX = frameY * frameAspect;
    tanY                = std::max(frameY, frameX / viewAspect) * (1 + kFrameMargin);
  }
  return 2.0 * std::atan(tanY) * (180.0 / 3.14159265358979323846);
}

namespace {

// Windows' default double-click time: a second click sooner, in the same place, is a double-click's
constexpr auto kDoubleClickTime = std::chrono::milliseconds(500);


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

FvView::FvView(HostRenderer* renderer, fv::EventHub& events, fv::CommandRegistry& commands, std::function<fv::Json(std::function<fv::Json()>)> onApp)
    : m_renderer(renderer)
    , m_events(&events)
    , m_onApp(std::move(onApp))
{
  m_renderer->setPickObserver([this](const std::optional<HostRenderer::InstancePick>& pick, const glm::vec3& hit) {
    const fv::HostAddress at = pick ? m_map.pick(pick->entity, pick->material) : fv::HostAddress{};
    // a Ctrl+click puts a prop in the selection or takes it out, and on nothing changes nothing
    if(m_clickCtrl && m_pick == "object")
    {
      if(at.prop >= 0)
        toggleMember(m_props[size_t(at.prop)].id);
      return true;
    }
    notePick(at);
    // a mesh pick selects the node hit, which the renderer does
    if(at.prop < 0 || m_pick == "mesh")
      return false;
    const PropNodes& p = m_props[size_t(at.prop)];
    if(m_pick == "bone")
    {
      // the posed bone nearest the hit, of the figure hit; a prop without bones is selected whole
      const int bone = p.bones.nearest ? p.bones.nearest(hit) : -1;
      const rs::EntityId entity = bone >= 0 && bone < int(p.entityOf.size()) ? p.entityOf[size_t(bone)] : rs::EntityId{};
      m_renderer->selectEntity(entity.isNone() ? p.root : entity);
      return true;
    }
    m_selectedProps = {p.id};
    m_renderer->selectEntity(p.root);
    return true;
  });
  m_renderer->setInspectObserver([this](const HostRenderer::PixelReadout& readout) { publishInspect(readout); });
  m_renderer->setPixelTimeObserver([this](const HostRenderer::PixelTimes& times) { publishPixelTimes(times); });
  // The pixel readout: what is at a pixel of the view (x, y and the frame's w, h, as the input events
  // give them; without w, h the render's own pixels) and how bright it is, published as `inspect` a
  // frame or two later and again as the still refines; x null stops it
  commands.add("view.inspect", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      if(!args.contains("x") || args["x"].is_null())
      {
        m_inspectAt.reset();
        m_renderer->inspectAt(std::nullopt);
        m_events->snapshot("inspect", nullptr);
        return fv::Json{{"ok", true}};
      }
      if(!args["x"].is_number() || !args.contains("y") || !args["y"].is_number())
        throw std::runtime_error("x and y are the pixel's place in the view, as input events give it; x null stops");
      const VkExtent2D size  = m_renderer->renderSize();
      const double     w     = args.value("w", 0.0) > 0 ? args.value("w", 0.0) : double(size.width);
      const double     h     = args.value("h", 0.0) > 0 ? args.value("h", 0.0) : double(size.height);
      if(!(w > 0 && h > 0))
        throw std::runtime_error("the view has no size yet");
      const glm::dvec2 at(args["x"].get<double>(), args["y"].get<double>());
      if(!(at.x >= 0 && at.y >= 0 && at.x <= w && at.y <= h))
        throw std::runtime_error("x and y lie outside the view");
      m_inspectAt = at;
      m_renderer->inspectAt(glm::vec2(float(at.x / w), float(at.y / h)));
      return fv::Json{{"ok", true}};
    });
  });
  commands.add("select", [this](const fv::Json& args) {
    return m_onApp([this, args] {
      // several: the first is the primary
      if(args.contains("props") && args["props"].is_array())
      {
        std::vector<std::string> ids;
        for(const fv::Json& id : args["props"])
          if(id.is_string() && propNodes(id.get<std::string>()) && std::find(ids.begin(), ids.end(), id.get<std::string>()) == ids.end())
            ids.push_back(id.get<std::string>());
        m_selectedProps = ids;
        m_renderer->selectEntity(ids.empty() ? rs::EntityId{} : propNodes(ids[0])->root);
        m_surfaceProp.clear();
        m_surface.clear();
        publishSelection(true);
        return fv::Json{{"ok", true}};
      }
      const std::string prop = args.contains("prop") && args["prop"].is_string() ? args["prop"].get<std::string>() : "";
      if(!prop.empty() && (args.value("toggle", false) || args.value("add", false)))
      {
        if(!propNodes(prop))
          throw std::runtime_error("no such prop");
        if(args.value("toggle", false) || std::find(m_selectedProps.begin(), m_selectedProps.end(), prop) == m_selectedProps.end())
          toggleMember(prop);
        return fv::Json{{"ok", true}};
      }
      m_selectedProps = prop.empty() ? std::vector<std::string>{} : std::vector<std::string>{prop};
      if(prop.empty())
        m_renderer->selectEntity({});
      else
      {
        const auto it = std::find_if(m_props.begin(), m_props.end(), [&](const PropNodes& p) { return p.id == prop; });
        if(it == m_props.end())
          throw std::runtime_error("no such prop");
        rs::EntityId entity = it->root;
        if(args.contains("node") && !args["node"].is_null())
        {
          const int node = args["node"].get<int>();
          entity         = node >= 0 && node < int(it->entityOf.size()) ? it->entityOf[size_t(node)] : rs::EntityId{};
          if(entity.isNone())
            throw std::runtime_error("no such node");
        }
        m_renderer->selectEntity(entity);
      }
      // chosen in a panel, not clicked: no surface
      m_surfaceProp.clear();
      m_surface.clear();
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
        const rs::EntityId was = m_renderer->selectedEntity();
        m_renderer->selectEntity(it->root);
        m_renderer->fitCamera(false);
        m_renderer->selectEntity(was);
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
  // The wipe against a kept frame: `keep` keeps the image as it is now, `at` puts the divider at a share
  // of the view's width from its left edge, the kept frame left of it; `at` null ends the wipe
  commands.add("view.wipe", [this](const fv::Json& args) {
    std::optional<std::optional<float>> at;
    if(args.contains("at"))
    {
      if(args["at"].is_null())
        at = std::optional<float>();
      else if(!args["at"].is_number())
        throw std::runtime_error("at is a share of the view's width, or null");
      else
        at = std::optional<float>(float(args["at"].get<double>()));
    }
    const bool keep = args.value("keep", false);
    return m_onApp([this, keep, at] {
      if(keep)
        m_renderer->keepWipeFrame();
      if(at)
        m_renderer->setWipe(*at);
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
        else if(display == "shaded" || display == "exposure")
        {
          m_renderer->setVisualization(shaderio::Visualization::eRendered);
          m_renderer->setWireframe(false);
        }
        // which material, which part of a prop, how large the triangles are, where a bump turns away
        else if(display == "material" || display == "object" || display == "density" || display == "facing")
        {
          m_renderer->setVisualization(display == "material" ? shaderio::Visualization::eMaterialId :
                                       display == "object"   ? shaderio::Visualization::eObjectId :
                                       display == "density"  ? shaderio::Visualization::eTriangleSize :
                                                               shaderio::Visualization::eBackFacingNormal);
          m_renderer->setWireframe(false);
        }
        // how long each pixel took, the paths as the shaded image's
        else if(display == "time")
        {
          if(!m_renderer->pixelTimeAvailable())
            throw std::runtime_error("this GPU has no shader clock (VK_KHR_shader_clock), so there is no Time view");
          m_renderer->setVisualization(shaderio::Visualization::eRendered);
          m_renderer->setWireframe(false);
        }
        else
          throw std::runtime_error("display is shaded, clay, wireframe, normals, exposure, material, object, density, facing or time");
        m_renderer->setFalseColor(display == "exposure");
        m_renderer->setPixelTime(display == "time");
        // the last breakdown goes with the view, so the next one starts empty
        if(m_pixelTime && display != "time")
          m_events->snapshot("time", nullptr);
        m_pixelTime = display == "time";
      }
      if(args.contains("grid"))
        m_renderer->setShowGrid(args["grid"].get<bool>());
      if(args.contains("gizmo"))
        m_renderer->setShowAxis(args["gizmo"].get<bool>());
      if(args.contains("pick"))
      {
        const std::string pick = args["pick"].get<std::string>();
        if(pick != "object" && pick != "bone" && pick != "mesh" && pick != "pull")
          throw std::runtime_error("pick is object, mesh, bone or pull");
        m_pick = pick;
        // the gizmo moves props and poses bones; a mesh pick only shows what was hit, and a pull's drag
        // is on the body
        m_renderer->setShowGizmo(m_pick != "mesh" && m_pick != "pull");
      }
      if(args.contains("select"))
        m_select = args["select"].get<bool>();
      if(args.contains("helpers"))
        m_helpers = args["helpers"].get<bool>();
      if(args.contains("frameAspect"))
      {
        const double aspect = args["frameAspect"].is_null() ? 0.0 : args["frameAspect"].get<double>();
        if(!(aspect >= 0) || aspect > 10)
          throw std::runtime_error("frameAspect is a width over height up to 10, or 0 for none");
        if(aspect != m_frameAspect)
        {
          m_frameAspect = aspect;
          m_aspect      = -1;
          applyLens();
        }
      }
      if(args.contains("transform"))
      {
        const std::string t = args["transform"].get<std::string>();
        if(t != "move" && t != "rotate" && t != "scale")
          throw std::runtime_error("transform is move, rotate or scale");
        m_transform = t;
      }
      // the gizmo's axes, the object's own or its parent's, and its snapping (null: none)
      if(args.contains("space") || args.contains("snap"))
      {
        if(args.contains("space"))
        {
          const std::string space = args["space"].get<std::string>();
          if(space != "world" && space != "local")
            throw std::runtime_error("space is world or local");
          m_gizmo.local = space == "local";
        }
        if(args.contains("snap"))
        {
          const fv::Json& snap = args["snap"];
          m_gizmo.snap         = snap.is_object();
          if(snap.is_object())
          {
            const auto step = [&](const char* key, float& into, float most) {
              if(snap.contains(key))
              {
                const float v = snap[key].get<float>();
                if(!(v > 0) || v > most)
                  throw std::runtime_error(std::string("snap.") + key + " is a step above 0, up to " + std::to_string(int(most)));
                into = v;
              }
            };
            step("move", m_gizmo.move, 100.f);
            step("rotate", m_gizmo.rotate, 180.f);
            step("scale", m_gizmo.scale, 10.f);
          }
          else if(!snap.is_null())
            throw std::runtime_error("snap is {move, rotate, scale} or null");
        }
        m_renderer->setGizmoOptions(m_gizmo);
      }
      // Isolate: a prop, a list of them, or null for all
      if(args.contains("solo"))
      {
        std::vector<std::string> ids;
        const fv::Json&          solo = args["solo"];
        if(solo.is_string())
          ids.push_back(solo.get<std::string>());
        else if(solo.is_array())
        {
          for(const fv::Json& id : solo)
            if(id.is_string())
              ids.push_back(id.get<std::string>());
        }
        else if(!solo.is_null())
          throw std::runtime_error("solo is a prop, a list of props, or null");
        if(m_solo)
          m_solo(ids);
      }
      return fv::Json{{"ok", true}};
    });
  });
  m_renderer->setShowGizmo(true);
}

void FvView::setProps(std::vector<PropNodes> props)
{
  m_props = std::move(props);
  m_map.set(m_props);
  // what went leaves the selection
  std::erase_if(m_selectedProps, [&](const std::string& id) { return !propNodes(id); });
  publishSelection(true);
}

const FvView::PropNodes* FvView::propNodes(const std::string& prop) const
{
  const auto it = std::find_if(m_props.begin(), m_props.end(), [&](const PropNodes& p) { return p.id == prop; });
  return it == m_props.end() ? nullptr : &*it;
}

void FvView::toggleMember(const std::string& prop)
{
  const PropNodes* p = propNodes(prop);
  if(!p)
    return;
  if(const auto it = std::find(m_selectedProps.begin(), m_selectedProps.end(), prop); it != m_selectedProps.end())
    m_selectedProps.erase(it);
  else
    m_selectedProps.insert(m_selectedProps.begin(), prop);
  // the primary: what came in, else the first left
  const PropNodes* primary = m_selectedProps.empty() ? nullptr : propNodes(m_selectedProps.front());
  m_renderer->selectEntity(primary ? primary->root : rs::EntityId{});
  m_surfaceProp.clear();
  m_surface.clear();
  publishSelection(true);
}

const FvView::PropNodes* FvView::propOfRoot(rs::EntityId entity) const
{
  const int prop = m_map.propOfRoot(entity);
  return prop < 0 ? nullptr : &m_props[size_t(prop)];
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
    // the Pull tool: a press on a figure in Physics mode holds it until the release; the camera
    // doesn't move meanwhile
    if(m_pulling)
    {
      if(type == "move")
        m_pull("move", {float(x), float(y)});
      else if(released)
      {
        m_pull("up", {float(x), float(y)});
        m_pulling = false;
      }
      continue;
    }
    if(pressed && m_pick == "pull" && m_pull && m_pull("down", {float(x), float(y)}))
    {
      m_pulling = true;
      m_click   = false;
      continue;
    }
    // a press on a handle, its drag and its release are the gizmo's: no orbit, no pick
    if((type == "down" || type == "up" || type == "move" || type == "leave")
       && m_renderer->gizmoInput({float(x), float(y)}, m_lmb, pressed, released))
    {
      m_click      = false;
      m_gizmoPress = m_gizmoPress || pressed;
      continue;
    }
    rest.push_back(e);
    if(pressed)
    {
      m_click      = true;
      m_gizmoPress = false;
      m_x0         = x;
      m_y0         = y;
    }
    else if(type == "dblclick")
    {
      // not after a double-click on a gizmo handle
      const VkExtent2D size = m_renderer->renderSize();
      if(!std::exchange(m_gizmoPress, false) && size.width && size.height)
        m_renderer->pickAt({float(x / size.width), float(y / size.height)}, true);
    }
    else if(type == "up" && e.value("b", 0) == 0)
    {
      const bool click = m_click && std::hypot(x - m_x0, y - m_y0) < 4;
      m_click          = false;
      if(!click || !m_select)
        continue;
      // the second click of a double-click picks nothing: the first one selected, and picking again
      // would let go of a mesh it selected (the renderer's toggle); a third click picks again
      const auto now    = std::chrono::steady_clock::now();
      const bool second = now - m_lastClick < kDoubleClickTime && std::hypot(x - m_lastX, y - m_lastY) < 4;
      m_lastClick       = second ? std::chrono::steady_clock::time_point{} : now;
      m_lastX           = x;
      m_lastY           = y;
      if(second)
        continue;
      const VkExtent2D size = m_renderer->renderSize();
      if(size.width == 0 || size.height == 0)
        continue;
      m_clickCtrl = e.value("ctrl", false);
      // a camera or light marker under the pointer is the pick; they have nothing the ray could hit
      if(m_helpers && m_helperPick)
        if(const std::string prop = m_helperPick({float(x), float(y)}); !prop.empty())
        {
          const auto it = std::find_if(m_props.begin(), m_props.end(), [&](const PropNodes& p) { return p.id == prop; });
          if(it != m_props.end())
          {
            if(m_clickCtrl && m_pick == "object")
              toggleMember(prop);
            else
            {
              notePick({});
              m_selectedProps = {prop};
              m_renderer->selectEntity(it->root);
            }
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
  // the throttle counts from this look, so an unchanged camera is built and compared once per
  // period, not every frame
  m_cameraAt = now;
  glm::dvec3 eye, center, up;
  manip->getLookat(eye, center, up);
  fv::Json camera{{"position", vec3(eye)}, {"target", vec3(center)}, {"up", vec3(up)}, {"focalLength", m_lens.focalLength}};
  camera.update(m_cameraExtra);
  if(camera == m_camera)
    return;
  m_camera = std::move(camera);
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
  return {{"focalLength", lens.focalLength}, {"dof", lens.dof}, {"fStop", lens.fStop}, {"focusMode", lens.focusMode},
          {"focusDistance", lens.focusDistance}, {"blades", lens.blades}, {"bladeRotation", lens.bladeRotation}};
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
  if(args.contains("blades"))
  {
    base.blades = args["blades"].get<int>();
    if(base.blades != 0 && (base.blades < 3 || base.blades > 16))
      throw std::runtime_error("blades is 0 (round) or 3 to 16");
  }
  if(args.contains("bladeRotation"))
    base.bladeRotation = args["bladeRotation"].get<double>();
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
  const double fov    = fovForFrame(m_lens.focalLength, aspect, m_frameAspect);
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
  m_renderer->setDepthOfField(float(std::max(m_lens.focusDistance, 1e-3)), aperture, m_lens.focusMode != "manual",
                              m_lens.blades, float(glm::radians(m_lens.bladeRotation)));
}

void FvView::deselect(const std::string& prop)
{
  if(prop.empty() || std::find(m_selectedProps.begin(), m_selectedProps.end(), prop) == m_selectedProps.end())
    return;
  if(m_selectedProp != prop)
  {
    std::erase(m_selectedProps, prop);
    publishSelection(true);
    return;
  }
  std::erase(m_selectedProps, prop);
  const PropNodes* primary = m_selectedProps.empty() ? nullptr : propNodes(m_selectedProps.front());
  m_renderer->selectEntity(primary ? primary->root : rs::EntityId{});
  publishSelection(true);
}

void FvView::notePick(const fv::HostAddress& at)
{
  m_surfaceProp.clear();
  m_surface.clear();
  // published again even when the node is the same: another surface of it
  m_publishAgain = true;
  if(at.surface < 0)
    return;
  m_surfaceProp = m_props[size_t(at.prop)].id;
  m_surface     = "m:" + std::to_string(at.surface);
}

void FvView::publishInspect(const HostRenderer::PixelReadout& readout)
{
  if(!m_inspectAt)
    return;
  const shaderio::PixelInspect& d   = readout.data;
  const auto                    rgb = [](const glm::vec4& v) { return fv::Json::array({v.x, v.y, v.z}); };
  fv::Json j = {{"x", m_inspectAt->x},
                {"y", m_inspectAt->y},
                {"radiance", rgb(d.radiance)},
                {"samples", int(d.radiance.w)},
                {"exposed", readout.exposed},
                {"moving", d.ids.w != 0}};
  if(!(d.position.w > 0.0F))
  {
    j["hit"] = nullptr;
    m_events->snapshot("inspect", std::move(j));
    return;
  }
  j["hit"]                 = d.ids.x == -2 ? "ground" : d.ids.x == -3 ? "water" : "mesh";
  j["distance"]            = d.position.w;
  j["position"]            = rgb(d.position);
  j["normal"]              = rgb(d.normal);
  j["front"]               = d.normal.w > 0.5F;
  j["baseColor"]           = rgb(d.baseColor);
  j["opacity"]             = d.baseColor.w;
  j["metallic"]            = d.surface.x;
  j["roughness"]           = d.surface.y;
  j["ior"]                 = d.surface.z;
  j["specular"]            = d.surface.w;
  j["transmission"]        = d.emission.w;
  j["emission"]            = rgb(d.emission);
  j["clearcoat"]           = d.layers.x;
  j["clearcoatRoughness"]  = d.layers.y;
  j["sheen"]               = d.layers.z;
  j["thickness"]           = d.layers.w;
  j["diffuseTransmission"] = d.medium.w;
  if(d.medium.x > 0.0F || d.medium.y > 0.0F || d.medium.z > 0.0F)
    j["subsurface"] = rgb(d.medium);
  if(d.ids.x >= 0)
    j["uv"] = fv::Json::array({d.uv.x, d.uv.y});
  if(readout.pick)
  {
    const fv::HostAddress at = m_map.pick(readout.pick->entity, readout.pick->material);
    if(at.prop >= 0 && at.prop < int(m_props.size()))
    {
      const PropNodes& p = m_props[size_t(at.prop)];
      j["prop"]          = p.id;
      if(at.node >= 0 && at.node < int(p.nodes.size()))
      {
        j["node"] = at.node;
        j["name"] = p.nodes[size_t(at.node)].label;
      }
      if(at.surface >= 0)
        j["surface"] = "m:" + std::to_string(at.surface);
    }
  }
  m_events->snapshot("inspect", std::move(j));
}

void FvView::publishPixelTimes(const HostRenderer::PixelTimes& times)
{
  if(!m_pixelTime)
    return;
  // What each line is: the sky, the ground, the water, a node of a prop, or a mesh the view can't name
  std::map<std::string, fv::Json>  what;
  std::vector<pixel_time::Group> groups;
  for(const HostRenderer::PixelTimes::Part& part : times.parts)
  {
    std::string key;
    fv::Json    line;
    switch(part.kind)
    {
      case HostRenderer::PixelTimes::Kind::eSky:
        key  = "sky";
        line = {{"what", "sky"}};
        break;
      case HostRenderer::PixelTimes::Kind::eGround:
        key  = "ground";
        line = {{"what", "ground"}};
        break;
      case HostRenderer::PixelTimes::Kind::eWater:
        key  = "water";
        line = {{"what", "water"}};
        break;
      case HostRenderer::PixelTimes::Kind::eMesh:
      {
        key  = "mesh";
        line = {{"what", "mesh"}};
        if(!part.pick)
          break;
        const fv::HostAddress at = m_map.pick(part.pick->entity, part.pick->material);
        if(at.prop < 0 || at.prop >= int(m_props.size()))
          break;
        const PropNodes& p = m_props[size_t(at.prop)];
        key                = p.id;
        line               = {{"what", "mesh"}, {"prop", p.id}};
        if(at.node >= 0 && at.node < int(p.nodes.size()))
        {
          key += "/" + std::to_string(at.node);
          line["node"] = at.node;
          line["name"] = p.nodes[size_t(at.node)].label;
        }
        break;
      }
    }
    what[key] = line;
    groups.push_back({.key = key, .ticks = part.ticks, .samples = part.samples});
  }
  fv::Json parts = fv::Json::array();
  for(const pixel_time::Share& share : pixel_time::shares(groups, kPixelTimeLines, "rest"))
  {
    fv::Json line     = share.key == "rest" && !what.contains("rest") ? fv::Json{{"what", "rest"}} : what[share.key];
    line["share"]     = std::round(share.share * 1000.0) / 1000.0;
    line["cost"]      = std::round(share.cost * 100.0) / 100.0;
    parts.push_back(std::move(line));
  }
  // the frame's average a sample, and what it is of: the clock's ticks (nanoseconds on NVIDIA), or a
  // count the debug pixelTimeCount asked for
  const double perSample = times.samples > 0 ? double(times.ticks) / double(times.samples) : 0.0;
  m_events->snapshot("time", fv::Json{{"parts", std::move(parts)},
                                      {"samples", times.samples},
                                      {"perSample", std::round(perSample * 100.0) / 100.0},
                                      {"counting", pixel_time::countingName(times.counting)}});
}

void FvView::publishSelection(bool force)
{
  const rs::EntityId entity = m_renderer->selectedEntity();
  if(!force && !m_publishAgain && entity == m_lastEntity)
    return;
  m_lastEntity               = entity;
  m_publishAgain             = false;
  const fv::HostAddress at   = m_map.locate(entity);
  const int             prop = at.prop, node = at.node;
  if(prop < 0)
  {
    m_selectedProp.clear();
    m_selectedProps.clear();
    m_events->snapshot("selection", {{"prop", nullptr}, {"props", fv::Json::array()}, {"node", nullptr}});
    return;
  }
  const PropNodes& p = m_props[size_t(prop)];
  m_selectedProp     = p.id;
  // the primary first; a selection made some other way (the host's, a mesh pick) is one again
  if(const auto it = std::find(m_selectedProps.begin(), m_selectedProps.end(), p.id); it == m_selectedProps.end())
    m_selectedProps = {p.id};
  else
    std::rotate(m_selectedProps.begin(), it, it + 1);
  const auto withSurface = [&](fv::Json sel) {
    if(m_surfaceProp == p.id && !m_surface.empty())
      sel["surface"] = m_surface;
    sel["props"] = m_selectedProps;
    return sel;
  };
  if(node < 0 || node >= int(p.nodes.size()))
  {
    m_events->snapshot("selection", withSurface({{"prop", p.id}, {"node", nullptr}}));
    return;
  }
  const NodeRec& n   = p.nodes[size_t(node)];
  fv::Json       sel = {{"prop", p.id}, {"node", node}, {"name", n.name}, {"mesh", n.mesh}, {"kind", n.kind}};
  if(const auto bone = p.bones.idOf.find(node); bone != p.bones.idOf.end())
    sel["bone"] = bone->second;
  else if(m_nodeState)
    if(fv::Json state = m_nodeState(p.id, node); !state.is_null())
      sel["state"] = std::move(state);
  m_events->snapshot("selection", withSurface(std::move(sel)));
}
