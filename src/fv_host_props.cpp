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
// FvHost: the scene's props, their topic and helpers, the view, and composing the scene.
//

#include "fv_host_impl.hpp"

//--------------------------------------------------------------------------------------------------
// Props

std::shared_ptr<Prop> FvHost::Impl::findProp(const std::string& id) const
{
  for(const auto& p : props)
    if(p->id == id)
      return p;
  return nullptr;
}

std::shared_ptr<Prop> FvHost::Impl::propArg(const Json& args) const
{
  for(const char* key : {"prop", "figure"})
    if(args.contains(key) && args[key].is_string())
    {
      auto p = findProp(args[key].get<std::string>());
      if(!p)
        throw std::runtime_error("no such prop");
      return p;
    }
  return view ? findProp(view->selectedProp()) : nullptr;
}

std::shared_ptr<Prop> FvHost::Impl::figureProp(const Json& args) const
{
  for(const char* key : {"prop", "figure"})
    if(args.contains(key) && args[key].is_string())
    {
      auto p = findProp(args[key].get<std::string>());
      return p && p->figure ? p : nullptr;
    }
  if(view)
    if(auto p = findProp(view->selectedProp()); p && p->figure)
      return p;
  std::shared_ptr<Prop> only;
  for(const auto& p : props)
    if(p->figure)
    {
      if(only)
        return nullptr;
      only = p;
    }
  return only;
}

std::shared_ptr<Loaded> FvHost::Impl::figureArg(const Json& args) const
{
  auto p = figureProp(args);
  return p ? p->figure : nullptr;
}

bool FvHost::Impl::inScene(const std::shared_ptr<Loaded>& f) const
{
  return f && std::any_of(props.begin(), props.end(), [&](const auto& p) { return p->figure == f; });
}

std::shared_ptr<Prop> FvHost::Impl::addProp(PropKind kind, const std::string& name, const std::string& id)
{
  auto p = std::make_shared<Prop>();
  // an id is never given out twice: one that comes back keeps the count past it
  if(!id.empty())
    propCount = std::max(propCount, std::atoi(id.c_str() + 2));
  p->id   = id.empty() ? "p:" + std::to_string(++propCount) : id;
  p->kind = kind;
  p->name = uniqueName(name.empty() ? kindName(kind) : name, nullptr);
  props.push_back(p);
  return p;
}

// A name the scene doesn't have yet: "Camera", "Camera 2", ... (`self` keeps its own)
std::string FvHost::Impl::uniqueName(const std::string& name, const Prop* self) const
{
  std::string unique = name;
  for(int n = 2; std::any_of(props.begin(), props.end(), [&](const auto& q) { return q.get() != self && q->name == unique; }); n++)
    unique = name + " " + std::to_string(n);
  return unique;
}

// It shows: it and everything it rides on are visible.
bool FvHost::Impl::shown(const Prop& p) const
{
  const Prop* at = &p;
  for(int steps = 0; at && steps <= int(props.size()); steps++)
  {
    if(!at->visible)
      return false;
    at = at->parent.empty() ? nullptr : findProp(at->parent).get();
  }
  return true;
}

// `q` rides on the prop `id`, however far up.
bool FvHost::Impl::ridesOn(const Prop& q, const std::string& id) const
{
  const Prop* at = q.parent.empty() ? nullptr : findProp(q.parent).get();
  for(int steps = 0; at && steps <= int(props.size()); steps++)
  {
    if(at->id == id)
      return true;
    at = at->parent.empty() ? nullptr : findProp(at->parent).get();
  }
  return false;
}

std::unique_ptr<FvMaterials> FvHost::Impl::makeMaterials(const std::shared_ptr<Prop>& p)
{
  std::weak_ptr<Prop> weak = p;
  return std::make_unique<FvMaterials>(FvMaterials::Hooks{
      .renderer = renderer,
      .events   = &protocol->events(),
      .post     = [this](std::function<void()> f) { post(std::move(f)); },
      .notice   = [this](const std::string& text, const char* level) { notice(text, level); },
      .nextJob  = [this] { return ++jobs; },
      .job      = [this](uint64_t id, const Json& state) { job(id, state); },
      .texture = [this, weak](const fv::build::TextureRequest& req, bool flipped) -> std::optional<fv::build::TextureImage> {
        std::shared_ptr<fv::dson::Library> lib;
        {
          std::lock_guard lock(settingsMutex);
          if(auto p = weak.lock(); p && p->figure)
            lib = p->figure->lib;
        }
        return bakeTexture(lib, req, flipped);
      },
      .remember =
          [weak](const fv::build::TextureRequest& req, const fv::build::TextureImage& image) {
            if(auto p = weak.lock(); p && p->figure)
              p->figure->images.insert_or_assign(fv::build::requestKey(req), image);
          },
      .resolves =
          [weak](const fv::build::TextureRequest& req) {
            const auto p = weak.lock();
            if(!p || !p->figure)
              return false;
            const auto it = p->figure->images.find(fv::build::requestKey(req));
            return it != p->figure->images.end() && it->second.has_value();
          },
      .writeFigure =
          [weak](std::vector<std::pair<int, fv::mat::LowerOptions>> materials) {
            const auto p = weak.lock();
            if(!p || !p->figure || !p->figure->feed)
              return;
            Loaded& f = *p->figure;
            if(!f.feed->syncMaterials(f.fig->loaded.scene, materials))
              f.feed->sync(f.fig->loaded.scene, f.fig->loaded.root);
          },
      .writeModel =
          [this, weak](int index, fv::mat::Material material) {
            const auto p = weak.lock();
            if(!p || !renderer->feedsHostScene() || index < 0 || index >= int(p->imported.materials.size()))
              return;
            if(const rs::MaterialId id = p->imported.materials[size_t(index)]; renderer->hostScene().material(id))
              renderer->hostScene().setMaterial(id, std::move(material));
          },
  });
}

std::optional<fv::build::TextureImage> FvHost::Impl::bakeTexture(const std::shared_ptr<fv::dson::Library>& lib, const fv::build::TextureRequest& req, bool flipped)
{
  HostSettings st;
  {
    std::lock_guard lock(settingsMutex);
    st = settings;
  }
  // a figure loaded from its sources (Original textures, or not cooked yet) bakes from them too
  const bool cooker = !st.texServer.empty() && !st.rootId.empty();
  if(cooker && (flipped || !st.originalTextures))
    return cookedTexture(st, req, flipped, false).image;
  return lib ? fallbackTexture(st, *lib, req, cooker) : std::nullopt;
}

// Out of the scene: its topics go, a busy line or a load waiting on it ends. The scene is composed again.
void FvHost::Impl::removeProp(const std::shared_ptr<Prop>& p)
{
  if(p->id == viewCamera)
    viewUse({{"camera", nullptr}});
  auto& events = protocol->events();
  if(p->materials)
    p->materials->detach();
  for(const char* topic : {"nodes/", "pose/", "morphs/", "wardrobe/", "dforce/"})
    events.drop(topic + p->id);
  if(p->figure && p->figure->busyOn)
    job(p->figure->busyJob, {{"state", "done"}});
  {
    std::lock_guard lock(morphMutex);
    morphFigures.erase(p->id);
    pendingMorphs.erase(p->id);
  }
  if(gizmoEdit && gizmoEdit->figure == p->figure)
    gizmoEdit.reset();
  // the selection goes with it now, not when the composition runs: a second Delete has nothing to remove
  if(view)
    view->deselect(p->id);
  // what rode on it stays where it was in the world
  for(const auto& q : props)
    if(q->parent == p->id)
    {
      const glm::mat4 world = worldOf(*q);
      q->parent.clear();
      q->bone.clear();
      glm::vec3 skew;
      glm::vec4 perspective;
      glm::decompose(world, q->transform.scale, q->transform.rotation, q->transform.translation, skew, perspective);
      q->euler.reset();
    }
  props.erase(std::remove(props.begin(), props.end(), p), props.end());
  // hidden now (with what rides on it, until the composition puts that back); gone from the scene
  // when the composition runs, from the GPU at its retire frame
  if(renderer->feedsHostScene() && renderer->hostScene().entity(p->root))
    renderer->hostScene().setVisible(p->root, false);
  if(p->figure)
    p->figure->feed = nullptr;  // the prop feed lets go of it at the composition
}

glm::mat4 FvHost::Impl::worldOf(const Prop& p) const
{
  glm::mat4   m(1.f);
  const Prop* at = &p;
  for(int steps = 0; at && steps <= int(props.size()); steps++)
  {
    const glm::mat4 local = glm::translate(glm::mat4(1.f), at->transform.translation) * glm::mat4_cast(at->transform.rotation)
                            * glm::scale(glm::mat4(1.f), at->transform.scale);
    m              = local * m;
    const Prop* up = at->parent.empty() ? nullptr : findProp(at->parent).get();
    if(up)
      m = boneOffset(*up, at->bone) * m;
    at = up;
  }
  return m;
}

glm::mat4 FvHost::Impl::parentWorld(const Prop& p) const
{
  return parentWorld(p.parent, p.bone);
}

glm::mat4 FvHost::Impl::parentWorld(const std::string& parent, const std::string& bone) const
{
  const auto up = parent.empty() ? nullptr : findProp(parent);
  return up ? worldOf(*up) * boneOffset(*up, bone) : glm::mat4(1.f);
}

int FvHost::Impl::boneNode(const Prop& figure, const std::string& bone) const
{
  if(bone.empty() || !figure.figure || !figure.figure->fig || !figure.figure->fig->pose)
    return -1;
  const fv::pose::PoseBone* pb = figure.figure->fig->pose->get(bone);
  return pb ? pb->bone : -1;
}

glm::mat4 FvHost::Impl::boneOffset(const Prop& figure, const std::string& bone) const
{
  // the pose is in the scene's entities already: the bone's from there, as the drawn figure has it
  const int node = boneNode(figure, bone);
  if(node < 0 || !figure.figure->feed || !renderer->feedsHostScene())
    return glm::mat4(1.f);
  rs::RetainedScene& s = renderer->hostScene();
  const rs::EntityId e = figure.figure->feed->entityOf(node);
  if(!s.entity(e) || !s.entity(figure.root))
    return glm::mat4(1.f);
  return glm::inverse(s.world(figure.root)) * s.world(e);
}

void FvHost::Impl::applyTransform(const Prop& p)
{
  if(!renderer->feedsHostScene() || !renderer->hostScene().entity(p.root))
    return;
  renderer->hostScene().setLocal(
      p.root, {.translation = p.transform.translation, .rotation = p.transform.rotation, .scale = p.transform.scale});
}

// A prop under a hidden one stays hidden, whatever its own flag, and a light hidden shines with
// nothing: the scene's visibility cascades so (RetainedScene::shown).
void FvHost::Impl::applyVisible(const Prop& p)
{
  if(!renderer->feedsHostScene() || !renderer->hostScene().entity(p.root))
    return;
  renderer->hostScene().setVisible(p.root, p.visible);
}

tinygltf::Light FvHost::Impl::lightPayload(const Prop& p) const
{
  tinygltf::Light l;
  l.name                = p.name;
  l.type                = p.light.type;
  const glm::vec3 c     = fv::lighting::linearColor(p.light.color);
  l.color               = {c.x, c.y, c.z};
  l.intensity           = p.light.intensity;
  l.spot.innerConeAngle = p.light.inner;
  l.spot.outerConeAngle = p.light.outer;
  tinygltf::Value::Object extras;
  extras["radius"] = tinygltf::Value(p.light.radius);
  l.extras         = tinygltf::Value(extras);
  return l;
}

namespace {

Json vec3Json(const glm::vec3& v)
{
  return Json::array({v.x, v.y, v.z});
}

glm::vec3 vec3Arg(const Json& v)
{
  if(v.is_number())
    return glm::vec3(v.get<float>());
  if(!v.is_array() || v.size() < 3)
    throw std::runtime_error("expected [x, y, z]");
  return {v[0].get<float>(), v[1].get<float>(), v[2].get<float>()};
}

// Degrees about X, then Y, then Z, as the panel shows them.
glm::quat eulerQuat(const glm::vec3& deg)
{
  return glm::quat(glm::radians(deg));
}

glm::vec3 quatEuler(const glm::quat& q)
{
  return glm::degrees(glm::eulerAngles(q));
}

}  // namespace

Json FvHost::Impl::propJson(const Prop& p) const
{
  const int order = int(std::find_if(props.begin(), props.end(), [&](const auto& q) { return q.get() == &p; }) - props.begin());
  Json      out   = {{"kind", kindName(p.kind)},
                     {"name", p.name},
                     {"parent", p.parent.empty() ? Json() : Json(p.parent)},
                     {"bone", p.bone.empty() ? Json() : Json(p.bone)},
                     {"visible", p.visible},
                     {"order", order},
                     {"transform",
                      {{"t", vec3Json(p.transform.translation)},
                       {"r", Json::array({p.transform.rotation.x, p.transform.rotation.y, p.transform.rotation.z, p.transform.rotation.w})},
                       {"e", vec3Json(p.euler ? *p.euler : quatEuler(p.transform.rotation))},
                       {"s", vec3Json(p.transform.scale)}}}};
  if(!p.stats.empty())
    out["stats"] = p.stats;
  if(p.figure)
  {
    out["source"]     = figureSource(*p.figure);
    out["generation"] = figureGeneration(*p.figure);
    out["posable"]    = bool(p.figure->fig->pose);
    out["format"]     = "DAZ";
  }
  if(p.model)
  {
    out["source"] = p.path;
    out["format"] = p.format;
  }
  if(p.kind == PropKind::eCamera)
  {
    Json camera              = FvView::lensJson(p.camera.lens);
    camera["locked"]         = p.camera.locked;
    camera["targetDistance"] = p.camera.targetDistance;
    out["camera"]            = camera;
  }
  if(p.kind == PropKind::eLight)
    out["light"] = {{"type", p.light.type},           {"color", p.light.color}, {"intensity", p.light.intensity},
                    {"radius", p.light.radius},       {"inner", p.light.inner}, {"outer", p.light.outer}};
  return out;
}

namespace {

// Where a prop's marker is, in the world: a camera's eye, a light's position.
glm::vec3 anchorOf(const glm::mat4& world)
{
  return glm::vec3(world[3]);
}

}  // namespace

// What has no mesh shows as lines: a camera as its frustum, a short pyramid along its -Z with a
// triangle on the side that is up; a point light as a star, a spot as its cone, a directional light
// as an arrow along its -Z. The selected one is orange; the camera looked through isn't drawn.
void FvHost::Impl::drawHelpers()
{
  std::vector<shaderio::lines::LineVertex> lines;
  auto manip = renderer->getCameraManipulator();
  if(view && view->helpersShown() && manip)
  {
    const glm::vec2 size(renderer->renderSize().width, renderer->renderSize().height);
    const float     aspect   = size.y > 0 ? size.x / size.y : 1.5f;
    const std::string selected = view->selectedProp();
    // markers keep their size on screen: scaled with the distance from the eye
    const glm::vec3 eye = glm::vec3(manip->getEye());
    for(const auto& p : props)
    {
      if(!p->visible || (p->kind != PropKind::eCamera && p->kind != PropKind::eLight) || p->id == viewCamera)
        continue;
      const glm::mat4 world = worldOf(*p);
      const glm::vec3 at    = anchorOf(world);
      const glm::vec3 x     = glm::normalize(glm::vec3(world[0])), y = glm::normalize(glm::vec3(world[1])), z = glm::normalize(glm::vec3(world[2]));
      const float     scale = std::clamp(glm::length(at - eye) * 0.06f, 0.05f, 2.f);
      glm::vec4       color = p->kind == PropKind::eCamera ? glm::vec4(0.85f, 0.88f, 0.92f, 0.9f) : glm::vec4(fv::lighting::linearColor(p->light.color), 0.95f);
      if(p->kind == PropKind::eLight)
        color = glm::vec4(glm::pow(glm::vec3(color), glm::vec3(1.f / 2.2f)), color.w);  // drawn over the tonemapped image
      if(p->id == selected)
        color = {1.f, 0.62f, 0.2f, 1.f};
      const uint32_t c    = LineHelperVk::pack(color);
      const auto     line = [&](const glm::vec3& a, const glm::vec3& b) {
        lines.push_back({a, c});
        lines.push_back({b, c});
      };
      if(p->kind == PropKind::eCamera)
      {
        const double    film = 35.0 / std::max(double(aspect), 1.0);
        const float     h    = float(film / 2.0 / std::max(p->camera.lens.focalLength, 1.0));  // tan of half the vertical field
        const float     d    = 1.2f * scale;
        const glm::vec3 centre = at - z * d;
        const glm::vec3 u = y * (h * d), r = x * (h * d * aspect);
        const glm::vec3 corners[4] = {centre - r - u, centre + r - u, centre + r + u, centre - r + u};
        for(int i = 0; i < 4; i++)
        {
          line(at, corners[i]);
          line(corners[i], corners[(i + 1) % 4]);
        }
        // which way is up
        const glm::vec3 top = centre + u * 1.6f;
        line(centre - r * 0.4f + u * 1.1f, top);
        line(top, centre + r * 0.4f + u * 1.1f);
        line(centre + r * 0.4f + u * 1.1f, centre - r * 0.4f + u * 1.1f);
      }
      else if(p->light.type == "point")
      {
        const float k = 0.35f * scale;
        line(at - x * k, at + x * k);
        line(at - y * k, at + y * k);
        line(at - z * k, at + z * k);
        const glm::vec3 diag[4] = {glm::normalize(x + y + z), glm::normalize(x - y + z), glm::normalize(-x + y + z), glm::normalize(-x - y + z)};
        for(const glm::vec3& v : diag)
          line(at - v * (k * 0.6f), at + v * (k * 0.6f));
      }
      else
      {
        const glm::vec3 dir = -z;
        if(p->light.type == "spot")
        {
          const float     length = 1.6f * scale;
          const float     radius = length * float(std::tan(std::clamp(p->light.outer, 0.01, 1.5)));
          const glm::vec3 base   = at + dir * length;
          glm::vec3       prev   = base + x * radius;
          for(int i = 1; i <= 24; i++)
          {
            const float     a    = float(i) / 24.f * 6.2831853f;
            const glm::vec3 next = base + (x * std::cos(a) + y * std::sin(a)) * radius;
            line(prev, next);
            if(i % 6 == 0)
              line(at, next);
            prev = next;
          }
        }
        else
        {
          const float length = 1.6f * scale, head = 0.3f * scale;
          const glm::vec3 tip = at + dir * length;
          line(at, tip);
          line(tip, tip - dir * head + x * head * 0.6f);
          line(tip, tip - dir * head - x * head * 0.6f);
          line(tip, tip - dir * head + y * head * 0.6f);
          line(tip, tip - dir * head - y * head * 0.6f);
          glm::vec3 prev = at + x * head;
          for(int i = 1; i <= 16; i++)
          {
            const float     a    = float(i) / 16.f * 6.2831853f;
            const glm::vec3 next = at + (x * std::cos(a) + y * std::sin(a)) * head;
            line(prev, next);
            prev = next;
          }
        }
      }
    }
  }
  renderer->setHelperLines(std::move(lines));
}

// The camera or light whose marker is nearest a pixel, within 14 of them: its anchor projected as
// the view sees it.
std::string FvHost::Impl::helperAt(const glm::vec2& px) const
{
  auto manip = renderer->getCameraManipulator();
  if(!manip)
    return {};
  const VkExtent2D extent = renderer->renderSize();
  const glm::mat4  viewProj = manip->getPerspectiveMatrix() * manip->getViewMatrix();
  std::string      best;
  float            nearest = 14.f;
  for(const auto& p : props)
  {
    if(!p->visible || (p->kind != PropKind::eCamera && p->kind != PropKind::eLight) || p->id == viewCamera)
      continue;
    const glm::vec4 clip = viewProj * glm::vec4(anchorOf(worldOf(*p)), 1.f);
    if(clip.w <= 1e-6f)
      continue;
    const glm::vec2 at = (glm::vec2(clip) / clip.w * 0.5f + 0.5f) * glm::vec2(float(extent.width), float(extent.height));
    const float     d  = glm::length(at - px);
    if(d < nearest)
    {
      nearest = d;
      best    = p->id;
    }
  }
  return best;
}

// Topic "scene": {props: {id: ...}, order: [id]}.
void FvHost::Impl::publishScene()
{
  Json all   = Json::object();
  Json order = Json::array();
  for(const auto& p : props)
  {
    all[p->id] = propJson(*p);
    order.push_back(p->id);
  }
  protocol->events().snapshot("scene", {{"props", all}, {"order", order}, {"view", viewCamera.empty() ? Json() : Json(viewCamera)}});
  movedProps.clear();
}

// The view looks through a camera prop: from where it stands, along its -Z, its orbit target
// targetDistance ahead.
void FvHost::Impl::viewFromCamera(const Prop& p)
{
  auto manip = renderer->getCameraManipulator();
  if(!manip)
    return;
  const glm::mat4  world   = worldOf(p);
  const glm::dvec3 eye     = glm::dvec3(world[3]);
  const glm::dvec3 forward = -glm::normalize(glm::dvec3(world[2]));
  const glm::dvec3 up      = glm::normalize(glm::dvec3(world[1]));
  manip->setLookat(eye, eye + forward * p.camera.targetDistance, up, true);
  viewWorld = world;
}

// Orbiting, panning and zooming through an unlocked camera move it: where the view went is written
// back to the camera prop. A locked camera takes the view back instead.
void FvHost::Impl::cameraFromView(bool dragEnds)
{
  const auto p     = findProp(viewCamera);
  auto       manip = renderer->getCameraManipulator();
  if(!p || !manip)
    return;
  glm::dvec3 eye, center, up;
  manip->getLookat(eye, center, up);
  const glm::mat4 world = worldOf(*p);
  // the camera moved (its parent, the gizmo, prop.set): the view goes with it
  for(int c = 0; c < 4; c++)
    if(glm::length(glm::vec4(world[c]) - glm::vec4(viewWorld[c])) > 1e-5f)
    {
      viewFromCamera(*p);
      return;
    }
  const double     distance = glm::length(center - eye);
  const glm::dvec3 forward  = (center - eye) / std::max(distance, 1e-9);
  // the view's up need not be square to where it looks; the camera's is
  const glm::dvec3 square = glm::normalize(up - glm::dot(up, forward) * forward);
  const bool       same   = glm::length(glm::dvec3(world[3]) - eye) < 1e-5 && glm::length(-glm::normalize(glm::dvec3(world[2])) - forward) < 1e-5
                    && glm::length(glm::normalize(glm::dvec3(world[1])) - square) < 1e-4 && std::abs(distance - p->camera.targetDistance) < 1e-5;
  if(same)
    return;
  if(p->camera.locked)
  {
    viewFromCamera(*p);
    return;
  }
  // a step of the history: a drag one from press to release, the wheel's steps merged as a slider's
  Json args = {{"prop", p->id}, {"transform", Json::object()}, {"camera", Json::object()}};
  if(lmb || mmb || rmb || dragEnds)
    args["gesture"] = {{"id", "view:" + std::to_string(viewGesture)}, {"end", dragEnds}};
  journaled("prop.set", args, [&] {
    const glm::mat4 view   = glm::lookAt(glm::vec3(eye), glm::vec3(center), glm::vec3(up));
    const glm::mat4 local = glm::inverse(parentWorld(*p)) * glm::inverse(view);
    glm::vec3       scale, skew;
    glm::vec4       perspective;
    glm::decompose(local, scale, p->transform.rotation, p->transform.translation, skew, perspective);
    p->transform.scale       = glm::vec3(1.f);
    p->euler.reset();
    p->camera.targetDistance = std::max(distance, 1e-3);
    applyTransform(*p);
    viewWorld = worldOf(*p);
    movedProps.insert(p->id);
    return Json();
  });
}

void FvHost::Impl::endViewGesture()
{
  // the last moves of the drag came in the batch that releases it: they are its step's end
  if(!viewCamera.empty() && frameIn == 0)
    cameraFromView(true);
  viewGesture++;
}

void FvHost::Impl::publishView()
{
  if(view)
  {
    Json extra = {{"view", viewCamera.empty() ? Json() : Json(viewCamera)}};
    if(!viewCamera.empty())
      extra["free"] = {{"position", Json::array({freeView.eye.x, freeView.eye.y, freeView.eye.z})},
                       {"target", Json::array({freeView.center.x, freeView.center.y, freeView.center.z})},
                       {"up", Json::array({freeView.up.x, freeView.up.y, freeView.up.z})}};
    view->setCameraExtra(extra);
  }
  protocol->events().patch("scene", {{"view", viewCamera.empty() ? Json() : Json(viewCamera)}});
}

// view.use {camera}: the viewport looks through a camera prop, with its lens; {camera: null} goes
// back to the free view where it was left.
Json FvHost::Impl::viewUse(const Json& args)
{
  const std::string id = args.contains("camera") && args["camera"].is_string() ? args["camera"].get<std::string>() : "";
  if(id == viewCamera)
    return {{"view", id.empty() ? Json() : Json(id)}};
  std::shared_ptr<Prop> camera;
  if(!id.empty())
  {
    camera = findProp(id);
    if(!camera || camera->kind != PropKind::eCamera)
      throw std::runtime_error("no such camera");
  }
  auto manip = renderer->getCameraManipulator();
  if(viewCamera.empty() && manip)
  {
    manip->getLookat(freeView.eye, freeView.center, freeView.up);
    if(view)
      freeView.lens = view->lens();
  }
  viewCamera    = id;
  lockedNoticed = false;
  if(camera)
  {
    if(view)
      view->setLens(camera->camera.lens);
    viewFromCamera(*camera);
  }
  else
  {
    if(manip)
      manip->setLookat(freeView.eye, freeView.center, freeView.up, true);
    if(view)
      view->setLens(freeView.lens);
  }
  publishView();
  return {{"view", id.empty() ? Json() : Json(id)}};
}

void FvHost::Impl::selectProp(const std::string& id)
{
  const auto p = findProp(id);
  if(p && !p->root.isNone())
    renderer->selectEntity(p->root);
}

void FvHost::Impl::requestCompose(const std::string& label, std::function<void()> after, Loaded* settle)
{
  if(!pendingCompose)
    pendingCompose.emplace();
  pendingCompose->labels.push_back(label);
  if(after)
    pendingCompose->after.push_back(std::move(after));
  if(settle)
    pendingCompose->settle.insert(settle);
}

// Prop i as a part of the composed scene, `model` what goes under its root (its figure's build or
// its file). A light the scene shows hidden draws at intensity 0; in an export (`forFile`) it keeps
// its intensity, and its root, hidden like any hidden prop's, says it is off.
fv::composer::Part FvHost::Impl::partOf(size_t i, const tinygltf::Model* model, bool forFile) const
{
  const Prop&        p = *props[i];
  fv::composer::Part part;
  part.id        = p.id;
  part.name      = p.name;
  part.transform = p.transform;
  part.visible   = p.visible;
  if(!p.bone.empty())
  {
    // on a bone: at the scene's root where the pose has it, the file being the scene as it is now
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(worldOf(p), part.transform.scale, part.transform.rotation, part.transform.translation, skew, perspective);
  }
  else
    for(size_t j = 0; j < props.size(); j++)
      if(!p.parent.empty() && props[j]->id == p.parent)
        part.parent = int(j);
  if(model)
    part.model = model;
  else if(p.kind == PropKind::eCamera)
    part.camera = fv::composer::perspectiveCamera(p.camera.lens.focalLength);
  else if(p.kind == PropKind::eLight)
  {
    tinygltf::Light l = lightPayload(p);
    l.intensity       = forFile || shown(p) ? p.light.intensity : 0.0;
    part.light        = l;
  }
  return part;
}

void FvHost::Impl::releaseFed()
{
  propFeed.reset();
  if(renderer->feedsHostScene() && renderer->hostScene().entity(sunEntity))
    renderer->hostScene().destroyEntity(sunEntity);
  sunEntity = {};
  sunLight  = {};
  for(const auto& p : props)
  {
    if(p->figure)
    {
      p->figure->feed = nullptr;
      p->figure->nodeOf.clear();
    }
    p->root        = {};
    p->imported    = {};
    p->lightRecord = {};
  }
}

fv::PropSpec FvHost::Impl::specOf(const Prop& p, bool figureChanged) const
{
  fv::PropSpec s{.id = p.id, .number = uint32_t(std::atoi(p.id.c_str() + 2)), .name = p.name, .parent = p.parent, .visible = p.visible};
  if(const auto up = p.parent.empty() ? nullptr : findProp(p.parent))
    s.parentNode = boneNode(*up, p.bone);
  s.transform = {.translation = p.transform.translation, .rotation = p.transform.rotation, .scale = p.transform.scale};
  if(Loaded* f = p.figure.get())
  {
    s.figure        = &f->fig->loaded.scene;
    s.figureRoot    = f->fig->loaded.root;
    s.figureOptions = buildOptionsOf(*f, skinMode.load());
    s.figureChanged = figureChanged;
    bool gpuRefine  = false;
    {
      std::lock_guard lock(settingsMutex);
      gpuRefine           = settings.figureGpuRefine;
      s.figureDazSkinning = settings.figureDazSkinning;
      s.figureDazOrder    = settings.figureDazOrder;
    }
    if(gpuRefine && f->fig->morphs)
      s.figureCageOf = [morphs = f->fig->morphs.get()](int mesh) { return morphs->morphedCage(mesh); };
  }
  else if(p.model)
  {
    s.model = p.model;
    if(!p.path.empty())
      s.imageSearchPaths.push_back(pathOf(p.path).parent_path());
  }
  else if(p.kind == PropKind::eCamera)
    s.camera = fv::composer::perspectiveCamera(p.camera.lens.focalLength);
  else if(p.kind == PropKind::eLight)
    s.light = lightPayload(p);
  return s;
}

// The props' changes since the last composition, fed into the renderer's scene in place (P5): a prop
// that came is fed, one that went goes (its instances retire at their frame), a figure whose
// wardrobe changed is fed again, and the rest keep their handles, so the selection, the clip that
// plays and the camera stay. The first composition makes the scene resident (a scene replace);
// after it there is no device wait.
void FvHost::Impl::compose()
{
  ComposeRequest request = std::move(*pendingCompose);
  pendingCompose.reset();
  std::string label;
  for(const std::string& l : request.labels)
    if(label.find(l) == std::string::npos)
      label += (label.empty() ? "" : ", ") + l;
  nvutils::ScopedTimer timer("[fv] compose: " + label + "\n");
  fv::timing::Scope    timeCompose("compose " + label);
  bool       finished = false;
  const auto finish   = [&] {
    finished = true;
    for(auto& f : request.after)
      f();
  };
  // a throw part way still ends what waits on this composition (load jobs, selections), each on its own
  struct Unwind
  {
    std::function<void()> f;
    ~Unwind() { f(); }
  } unwind{[&] {
    if(finished)
      return;
    finished = true;
    for(auto& f : request.after)
      try
      {
        f();
      }
      catch(const std::exception& e)
      {
        LOGW("[fv] compose: %s\n", e.what());
      }
  }};

  // the first thing in an empty scene is framed; after that the camera stays, and so does a camera looked through
  const bool hasGeometry = std::any_of(props.begin(), props.end(), [](const auto& p) { return p->figure || p->model; });
  const bool frame       = hasGeometry && !composedAny && viewCamera.empty() && !viewKept;
  composedAny            = hasGeometry;
  viewKept               = viewKept && !hasGeometry;

  rs::RetainedScene& scene = renderer->hostScene();
  if(!propFeed)
    propFeed = std::make_unique<fv::PropFeed>(scene);
  fv::PropFeed::Result fed;
  {
    fv::timing::Scope         t("feed");
    std::vector<fv::PropSpec> specs;
    for(const auto& p : props)
      specs.push_back(specOf(*p, p->figure && request.settle.count(p->figure.get())));
    try
    {
      fed = propFeed->sync(specs);
    }
    catch(const std::exception& e)
    {
      notice(std::string("The scene could not be fed: ") + e.what(), "error");
    }
    if(!scene.entity(sunEntity))
      feedSun(scene);
  }
  for(const auto& [id, text] : fed.notices)
    if(const auto p = findProp(id))
      notice(p->name + ": " + text, "warning");
  const std::set<std::string> added(fed.added.begin(), fed.added.end());
  for(const auto& p : props)
  {
    const fv::FedProp* f = propFeed->fed(p->id);
    p->root              = f ? f->root : rs::EntityId{};
    p->lightRecord       = f ? f->light : rs::LightId{};
    if(p->model && f && added.count(p->id))
    {
      p->imported = f->imported;
      // its clips' numbers, theirs from now on
      for(int c = 0; c < int(p->imported.clips.size()); c++)
        if(!p->imported.clips[size_t(c)].isNone())
          clipNumber(p->id, c);
    }
    if(p->figure)
    {
      p->figure->feed = f && f->figure ? f->figure.get() : nullptr;
      // a dial moves only the cages of the meshes its feed has the GPU refine (figureGpuRefine)
      if(p->figure->fig->morphs)
        p->figure->fig->morphs->cageOnly = [l = p->figure.get()](int mesh) { return l->feed && l->feed->refines(mesh); };
    }
  }
  // the first composition makes the scene resident, waited for; after it each frame's sync takes the changes
  if(!renderer->drawsRetainedScene())
  {
    nvutils::ScopedTimer t("upload");
    fv::timing::Scope    ts("upload");
    renderer->replaceHostScene();
    appliedParams.erase("infinitePlaneDistance");
    appliedParams.erase("ptFireflyClamp");  // the replace recalibrated it (calibrateFireflyClamp); ours goes again
    applyLighting(false);
  }
  // "auto" ground: the scene's lowest drawn point, again at each structural change (D6), from the
  // positions as fed (morphed, unskinned: D7)
  {
    const std::array<glm::vec3, 2> bounds = scene.bounds();
    const float                    ground = bounds[0].y <= bounds[1].y ? bounds[0].y : 0.f;
    if(ground != autoGround)
    {
      autoGround = ground;
      appliedParams.erase("infinitePlaneDistance");
      applyLighting(false);
    }
  }

  // what the host keeps of each prop's place in the scene, and the topics of the props that changed
  fv::timing::Scope              timeAttach("attach");
  std::vector<FvView::PropNodes> views;
  for(size_t i = 0; i < props.size(); i++)
  {
    Prop&             p       = *props[i];
    const bool        changed = added.count(p.id) || (p.figure && request.settle.count(p.figure.get()));
    FvView::PropNodes nodes;
    nodes.id   = p.id;
    nodes.root = p.root;
    if(p.figure && p.figure->feed)
    {
      Loaded&                   f     = *p.figure;
      const fv::SceneGraph&     graph = f.fig->loaded.scene;
      std::vector<rs::EntityId> entityOf(graph.nodes.size());
      f.nodeOf.clear();
      for(int n = 0; n < int(graph.nodes.size()); n++)
      {
        entityOf[size_t(n)] = f.feed->entityOf(n);
        if(!entityOf[size_t(n)].isNone())
          f.nodeOf[entityOf[size_t(n)]] = n;
      }
      for(int m = 0; m < int(graph.materials.size()); m++)
        nodes.materials.push_back(f.feed->materialOf(m));
      if(changed)
      {
        // no drain(): its changes go to the frame's pump, which feeds what moved since; a drain here
        // rebinds the skins after the feed took them, and drops that
        if(!p.materials)
          p.materials = makeMaterials(props[i]);
        if(added.count(p.id))
          p.materials->attach(p.id, &f.fig->loaded.scene, f.feed->lowering(), f.texturesFlipped, skinMode.load(), f.feed->takeKept());
        else
          p.materials->extend(f.feed->lowering(), f.feed->takeKept());
        publishEyes(p);
        publishSkin(p);
      }
      Json described = FvView::describeFigure(nodes, graph, f.fig->loaded.root, std::move(entityOf),
                                              f.fig->morphs ? f.fig->morphs->size() : 0);
      if(f.fig->pose)
      {
        for(const auto& pb : f.fig->pose->bones())
          if(!pb.axes.empty())
            nodes.bones.idOf[pb.bone] = pb.id;
        // a world hit, in the figure's own space under its prop root
        std::weak_ptr<Prop> weak = props[i];
        nodes.bones.nearest      = [this, weak](const glm::vec3& hit) {
          const auto p = weak.lock();
          if(!p || !p->figure || !p->figure->fig->pose)
            return -1;
          const glm::vec3 local = glm::vec3(glm::inverse(worldOf(*p)) * glm::vec4(hit, 1.f));
          const auto*     pb    = p->figure->fig->pose->nearestBone({local.x, local.y, local.z});
          return pb ? pb->bone : -1;
        };
      }
      if(changed)
      {
        p.stats = described["stats"];
        protocol->events().snapshot("nodes/" + p.id, {{"prop", p.id}, {"nodes", described["nodes"]}});
      }
    }
    else if(p.figure)
    {
      // its feed failed, so none of it is in this scene
      if(p.materials)
        p.materials->detach();
    }
    else if(p.model)
    {
      if(!p.materials)
        p.materials = makeMaterials(props[i]);
      if(changed)
        p.materials->attachModel(p.id, *p.model);
      nodes.materials = p.imported.materials;
      nodes.materials.resize(p.model->materials.size());
      Json described = FvView::describeModel(nodes, *p.model, p.imported.entities);
      if(changed)
      {
        p.stats = described["stats"];
        protocol->events().snapshot("nodes/" + p.id, {{"prop", p.id}, {"nodes", described["nodes"]}});
      }
    }
    views.push_back(std::move(nodes));
  }
  if(view)
    view->setProps(std::move(views));
  for(const auto& p : props)
    if(p->figure && (added.count(p->id) || request.settle.count(p->figure.get())))
    {
      publishFigure(*p);
      if(p->figure->dforce && request.settle.count(p->figure.get()))
        p->figure->dforce->poke();
      if(p->figure->collider && request.settle.count(p->figure.get()))
        p->figure->collider->poke();
    }
  // the clip that plays plays on; when its prop went, the first clip there is, if any
  if(clipOf(currentClip).isNone())
  {
    const std::vector<int> order = clipOrder();
    currentClip                  = order.empty() ? -1 : order.front();
    renderer->setHostClip(clipOf(currentClip));
    if(currentClip < 0)
      renderer->editAnimationControl().play = false;
  }
  publishScene();
  publishAnimation(true);
  if(frame)
    frameIn = 2;
  finish();
}

Json FvHost::Impl::propAdd(const Json& args)
{
  if(args.contains("path") || args.contains("rel"))
  {
    Json asProp  = args;
    asProp["as"] = "prop";
    return load(asProp);
  }
  const std::string kind = args.value("kind", "");
  if(kind != "camera" && kind != "light")
    throw std::runtime_error("prop.add wants a path, a rel, or kind camera or light");
  const std::string type = args.value("type", "point");
  if(kind == "light" && type != "point" && type != "spot" && type != "directional")
    throw std::runtime_error("a light is point, spot or directional");
  const std::string label = kind == "camera" ? "Camera" : type == "point" ? "Point light" : type == "spot" ? "Spot light" : "Directional light";
  auto              manip = renderer->getCameraManipulator();
  auto              p     = addProp(kind == "camera" ? PropKind::eCamera : PropKind::eLight, args.value("name", label));
  if(kind == "camera")
  {
    p->camera.lens = view ? view->lens() : FvView::Lens{};
    if(args.contains("camera"))
      p->camera.lens = FvView::lensOf(args["camera"], p->camera.lens);
    // where the view is: a camera made from it frames what it shows
    if(manip)
    {
      glm::dvec3 eye, center, up;
      manip->getLookat(eye, center, up);
      const glm::mat4 look         = glm::inverse(glm::lookAt(glm::vec3(eye), glm::vec3(center), glm::vec3(up)));
      p->transform.translation     = glm::vec3(eye);
      p->transform.rotation        = glm::quat_cast(glm::mat3(look));
      p->camera.targetDistance     = std::max(glm::length(center - eye), 1e-3);
    }
  }
  else
  {
    p->light.type = type;
    if(type == "directional")
      p->light.intensity = 3;
    // above the orbit target, a little towards the camera, aimed down at it
    if(manip)
    {
      glm::dvec3 eye, center, up;
      manip->getLookat(eye, center, up);
      const glm::vec3 target(center);
      const glm::vec3 at = target + glm::vec3(0.f, 1.5f, 0.f) + glm::normalize(glm::vec3(eye - center)) * 1.f;
      p->transform.translation = at;
      p->transform.rotation    = glm::quat_cast(glm::mat3(glm::inverse(glm::lookAt(at, target, glm::vec3(0, 1, 0)))));
    }
  }
  if(args.contains("transform"))
    propSet({{"prop", p->id}, {"transform", args["transform"]}});
  placeProp(*p, args.value("place", Json()));
  requestCompose("Adding " + p->name, [this, id = p->id] { selectProp(id); });
  return {{"prop", p->id}};
}

void FvHost::Impl::placeProp(const Prop& p, const Json& place)
{
  if(!place.is_object() || place.empty())
    return;
  Json set    = place;
  set["prop"] = p.id;
  propSet(set);
}

// What the panels change about a prop: a name, whether it shows, where it is, what it rides on, a
// camera's lens and lock, a light's colour and strength. Only a new parent composes the scene again.
Json FvHost::Impl::propSet(const Json& args)
{
  auto p = findProp(args.at("prop").get<std::string>());
  if(!p)
    throw std::runtime_error("no such prop");
  if(args.contains("name") && args["name"].is_string() && !args["name"].get<std::string>().empty())
    p->name = args["name"].get<std::string>();
  if(args.contains("transform"))
  {
    const Json& t = args["transform"];
    if(t.contains("t"))
      p->transform.translation = vec3Arg(t["t"]);
    const bool quat = t.contains("r") && t["r"].is_array() && t["r"].size() == 4;
    if(quat)
    {
      // the rotation itself; degrees with it (a scene file) are kept for the panel when they say the same
      p->transform.rotation = glm::normalize(glm::quat(t["r"][3].get<float>(), t["r"][0].get<float>(), t["r"][1].get<float>(), t["r"][2].get<float>()));
      p->euler.reset();
      if(t.contains("e"))
      {
        const glm::vec3 e = vec3Arg(t["e"]);
        if(std::abs(glm::dot(eulerQuat(e), p->transform.rotation)) > 1.f - 1e-6f)
          p->euler = e;
      }
    }
    else if(t.contains("e"))
    {
      p->euler              = vec3Arg(t["e"]);
      p->transform.rotation = eulerQuat(*p->euler);
    }
    if(t.contains("s"))
    {
      p->transform.scale = vec3Arg(t["s"]);
      for(int i = 0; i < 3; i++)
        if(std::abs(p->transform.scale[i]) < 1e-4f)
          p->transform.scale[i] = 1e-4f;
    }
    applyTransform(*p);
    if(p->id == viewCamera)
      viewFromCamera(*p);
  }
  if(args.contains("visible"))
  {
    p->visible = args["visible"].get<bool>();
    applyVisible(*p);
  }
  if(args.contains("camera") && p->kind == PropKind::eCamera)
  {
    const Json& c  = args["camera"];
    p->camera.lens = FvView::lensOf(c, p->camera.lens);
    if(c.contains("locked"))
      p->camera.locked = c["locked"].get<bool>();
    if(c.contains("targetDistance"))
      p->camera.targetDistance = std::max(c["targetDistance"].get<double>(), 1e-3);
    if(p->id == viewCamera && view)
    {
      view->setLens(p->camera.lens);
      viewFromCamera(*p);
    }
  }
  if(args.contains("light") && p->kind == PropKind::eLight)
  {
    const Json& l = args["light"];
    if(l.contains("color"))
    {
      fv::lighting::linearColor(l["color"].get<std::string>());  // throws on a bad colour
      p->light.color = l["color"].get<std::string>();
    }
    if(l.contains("intensity"))
      p->light.intensity = std::max(l["intensity"].get<double>(), 0.0);
    if(l.contains("radius"))
      p->light.radius = std::max(l["radius"].get<double>(), 0.0);
    if(l.contains("inner"))
      p->light.inner = std::clamp(l["inner"].get<double>(), 0.0, 1.5707);
    if(l.contains("outer"))
      p->light.outer = std::clamp(l["outer"].get<double>(), 0.0, 1.5707);
    p->light.inner = std::min(p->light.inner, p->light.outer);
    if(l.contains("type") && l["type"].get<std::string>() != p->light.type)
    {
      const std::string type = l["type"].get<std::string>();
      if(type != "point" && type != "spot" && type != "directional")
        throw std::runtime_error("a light is point, spot or directional");
      p->light.type = type;
      requestCompose("Changing " + p->name);
    }
    if(renderer->feedsHostScene() && renderer->hostScene().light(p->lightRecord))
      renderer->hostScene().setLight(p->lightRecord, lightPayload(*p));
  }
  if(args.contains("parent") || args.contains("bone"))
  {
    // a new parent rides on the figure itself unless a bone comes with it; a bone alone is another
    // of the same figure's
    const std::string parent = !args.contains("parent") ? p->parent : args["parent"].is_string() ? args["parent"].get<std::string>() : "";
    std::string       bone   = args.contains("bone") && args["bone"].is_string() ? args["bone"].get<std::string>() : "";
    auto              at     = parent.empty() ? nullptr : findProp(parent);
    if(!parent.empty() && !at)
      throw std::runtime_error("no such parent");
    if(!bone.empty())
    {
      if(!at)
        throw std::runtime_error("a bone needs a figure to ride on");
      if(boneNode(*at, bone) < 0)
        throw std::runtime_error(at->name + " has no bone " + bone);
      bone = at->figure->fig->pose->get(bone)->id;  // the id, whichever it was named by
    }
    if(parent != p->parent || bone != p->bone)
    {
      // it stays where it is in the world, now relative to its new parent
      for(Prop* up = at.get(); up; up = up->parent.empty() ? nullptr : findProp(up->parent).get())
        if(up == p.get())
          throw std::runtime_error(p->name + " can't ride on something that rides on it");
      const glm::mat4 local = glm::inverse(parentWorld(parent, bone)) * worldOf(*p);
      glm::vec3       skew;
      glm::vec4       perspective;
      glm::decompose(local, p->transform.scale, p->transform.rotation, p->transform.translation, skew, perspective);
      p->euler.reset();
      p->parent = parent;
      p->bone   = bone;
      if(p->id == viewCamera)
        viewFromCamera(*p);
      requestCompose("Parenting " + p->name);
    }
  }
  if(args.contains("order") && args["order"].is_number_integer())
  {
    // a scene file sends every prop's place; only one that moves needs the scene composed again
    const int to   = std::clamp(args["order"].get<int>(), 0, int(props.size()) - 1);
    const int from = int(std::find(props.begin(), props.end(), p) - props.begin());
    if(from != to)
    {
      props.erase(props.begin() + from);
      props.insert(props.begin() + to, p);
      requestCompose("Reordering");
    }
  }
  publishScene();
  return {{"ok", true}};
}

// A copy beside the original: a camera or light as it is, a model sharing the file's data with its
// material edits, a figure loaded again from its source with its state put back.
Json FvHost::Impl::propDuplicate(const Json& args)
{
  auto src = findProp(args.at("prop").get<std::string>());
  if(!src)
    throw std::runtime_error("no such prop");
  const auto place = [this, src](const std::shared_ptr<Prop>& p) {
    p->transform = src->transform;
    p->euler     = src->euler;
    p->visible   = src->visible;
    p->parent    = src->parent;
    p->bone      = src->bone;
  };
  if(src->figure)
  {
    const Json state = sceneState({{"prop", src->id}});
    HostSettings s;
    {
      std::lock_guard lock(settingsMutex);
      s = settings;
    }
    std::optional<std::pair<std::vector<std::filesystem::path>, std::string>> library;
    const Json& source = src->figure->source;
    if(source.contains("rel"))
      library = std::make_pair(s.roots, source["rel"].get<std::string>());
    else if(source.contains("path"))
      library = libraryFor(s, pathOf(source["path"].get<std::string>()));
    if(!library)
      throw std::runtime_error(src->name + " can't be loaded again: where it came from isn't known");
    return loadFigure(s, library->first, library->second, src->name, source, "", true, [this, state, place, name = src->name](std::shared_ptr<Prop> p) {
      place(p);
      p->name = uniqueName(name, p.get());
      applyTransform(*p);
      applyVisible(*p);
      // it was composed at the scene's root: under its parent it needs composing again
      if(!p->parent.empty())
        requestCompose("Duplicating " + p->name);
      Json restore = state;
      restore["prop"] = p->id;
      sceneRestore(restore);
      publishScene();
      recordAdded(p, "Duplicate " + name);
    });
  }
  auto p    = addProp(src->kind, src->name);
  p->model  = src->model;
  p->path   = src->path;
  p->format = src->format;
  p->camera = src->camera;
  p->light  = src->light;
  place(p);
  Json edits = src->materials ? src->materials->editState() : Json::object();
  requestCompose("Duplicating " + src->name, [this, id = p->id, edits] {
    const auto p = findProp(id);
    if(!p)
      return;
    if(p->materials && !edits.empty())
      p->materials->restoreEdits(edits);
    selectProp(id);
  });
  recordAdded(p, "Duplicate " + src->name);
  return {{"prop", p->id}};
}

Json FvHost::Impl::sceneClear()
{
  loadGeneration++;  // loads still running drop their result
  for(auto p : std::vector(props))
    removeProp(p);
  requestCompose("Clearing the scene");
  return {{"ok", true}};
}
