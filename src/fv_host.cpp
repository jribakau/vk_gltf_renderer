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

// httplib pulls in winsock2; it has to come before anything that includes windows.h.
#include <httplib.h>

#include "fv_host.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <nvapp/elem_camera.hpp>
#include <nvutils/logger.hpp>

#include <figure/gltf/gltf_build.hpp>
#include <figure/dson/figure_loader.hpp>
#include <figure/pose/pose_rig.hpp>

#include "fv_protocol.hpp"
#include "renderer.hpp"

using fv::Json;
using namespace std::chrono_literals;

namespace {

// DAZ scenes are in centimetres; the renderer's world is in metres.
constexpr double kCentimetre = 0.01;

struct HostSettings
{
  std::vector<std::filesystem::path> roots;
  std::string                        rootId;
  std::string                        texServer;  // http://127.0.0.1:<port>/<token>
  int                                maxTextureSize = 4096;
  int                                subdivision    = 0;
};

// A loaded figure and what the renderer made of it. Lives on the application thread.
struct Figure
{
  std::string                        id;
  fv::dson::LoadResult               result;
  std::unique_ptr<fv::pose::PoseRig> rig;
  std::vector<int>                   nodeOf;  // scene node -> renderer (glTF) node
};

std::filesystem::path pathOf(const std::string& utf8)
{
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

std::string utf8Of(const std::filesystem::path& p)
{
  const std::u8string s = p.generic_u8string();
  return std::string(s.begin(), s.end());
}

// encodeURIComponent
std::string encodeComponent(std::string_view s)
{
  static const char* hex = "0123456789ABCDEF";
  std::string        out;
  for(unsigned char c : s)
  {
    if(std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '!' || c == '~' || c == '*' || c == '\'' || c == '(' || c == ')')
      out += char(c);
    else
    {
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 15];
    }
  }
  return out;
}

// A library-relative path, its segments encoded (src/lib/ipc.ts encodeRel).
std::string encodeRel(std::string_view rel)
{
  std::string out, segment;
  auto        flush = [&] {
    if(!segment.empty())
      out += (out.empty() ? "" : "/") + encodeComponent(segment);
    segment.clear();
  };
  for(char c : rel)
  {
    if(c == '/' || c == '\\')
      flush();
    else
      segment += c;
  }
  flush();
  return out;
}

// A DSON image path as the library has it: decoded, forward slashes, no leading slash.
std::string normRel(const std::string& path)
{
  std::string rel = fv::dson::decodeUri(path);
  std::replace(rel.begin(), rel.end(), '\\', '/');
  const size_t at = rel.find_first_not_of('/');
  return at == std::string::npos ? std::string{} : rel.substr(at);
}

// encodeURI for an absolute path used as an image URI (the renderer percent-decodes URIs).
std::string uriOfPath(const std::string& path)
{
  std::string out;
  for(unsigned char c : path)
  {
    if(c == '%' || c == ' ' || c == '#' || c == '?' || c >= 0x80)
    {
      static const char* hex = "0123456789ABCDEF";
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 15];
    }
    else
      out += char(c);
  }
  return out;
}

std::string mimeOf(const std::string& path)
{
  std::string ext = path.substr(path.find_last_of('.') + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if(ext == "png")
    return "image/png";
  if(ext == "ktx2")
    return "image/ktx2";
  if(ext == "webp")
    return "image/webp";
  return "image/jpeg";
}

// Where a cooked texture is: the texture server cooks it (BC7 KTX2, flipped for the DSON UVs,
// cutout folded into alpha) and answers with the file's path.
std::optional<fv::build::TextureImage> cookedTexture(const HostSettings& s, const fv::build::TextureRequest& req)
{
  // http://127.0.0.1:<port>/<token>
  const std::string& base = s.texServer;
  const size_t       hostAt = base.find("://");
  const size_t       pathAt = base.find('/', hostAt == std::string::npos ? 0 : hostAt + 3);
  if(hostAt == std::string::npos || pathAt == std::string::npos)
    return std::nullopt;
  httplib::Client cli(base.substr(0, pathAt));
  cli.set_read_timeout(120s);
  std::string url = base.substr(pathAt) + "/" + s.rootId + "/" + encodeRel(normRel(req.rel)) + "?srgb=" + (req.srgb ? "1" : "0")
                    + "&flip=1&max=" + std::to_string(s.maxTextureSize);
  if(req.normal)
    url += "&nm=1";
  if(!req.alphaRel.empty())
    url += "&alpha=" + encodeComponent(s.rootId + "/" + normRel(req.alphaRel));
  url += "&path=1";
  auto res = cli.Get(url);
  if(!res || res->status != 200)
  {
    LOGW("texture %s: %s\n", req.rel.c_str(), res ? (std::to_string(res->status) + " " + res->body).c_str() : "no answer");
    return std::nullopt;
  }
  return fv::build::TextureImage{uriOfPath(res->body), "image/ktx2", true};
}

// The source image as it is, for a renderer without a texture server (tests, a standalone run).
std::optional<fv::build::TextureImage> sourceTexture(const fv::dson::Library& lib, const fv::build::TextureRequest& req)
{
  auto path = lib.find(normRel(req.rel));
  if(!path)
    return std::nullopt;
  const std::string p = utf8Of(*path);
  return fv::build::TextureImage{uriOfPath(p), mimeOf(p), false};
}

// The library folder a .duf sits in: one of the configured roots, else the nearest folder above it
// with a "data" folder (as src/lib/actions.ts finds it).
std::optional<std::pair<std::vector<std::filesystem::path>, std::string>> libraryFor(const HostSettings& s, const std::filesystem::path& file)
{
  std::error_code ec;
  const auto      abs = std::filesystem::weakly_canonical(file, ec);
  for(const auto& root : s.roots)
  {
    const auto r   = std::filesystem::weakly_canonical(root, ec);
    const auto rel = abs.lexically_relative(r);
    if(!rel.empty() && *rel.begin() != "..")
    {
      // the file's own library first, then the other content folders for whatever it references
      std::vector<std::filesystem::path> roots{root};
      for(const auto& other : s.roots)
        if(other != root)
          roots.push_back(other);
      return std::make_pair(roots, utf8Of(rel));
    }
  }
  auto dir = abs.parent_path();
  for(int i = 0; i < 8 && !dir.empty(); i++, dir = dir.parent_path())
  {
    if(std::filesystem::is_directory(dir / "data", ec))
    {
      std::vector<std::filesystem::path> roots{dir};
      roots.insert(roots.end(), s.roots.begin(), s.roots.end());
      return std::make_pair(roots, utf8Of(abs.lexically_relative(dir)));
    }
    if(dir == dir.root_path())
      break;
  }
  return std::nullopt;
}

std::string axisKey(const fv::pose::Axis& a)
{
  return std::string(fv::pose::kindName(a.kind)) + "/" + a.axis;
}

// The pose topic: bones by id, each with its axes by "kind/axis".
Json poseSnapshot(const Figure& f)
{
  Json bones = Json::object();
  int  order = 0;
  for(const auto& pb : f.rig->bones())
  {
    if(pb.axes.empty())
      continue;
    Json axes = Json::object();
    int  i    = 0;
    for(const auto& a : pb.axes)
      axes[axisKey(a)] = {{"label", a.label}, {"kind", fv::pose::kindName(a.kind)}, {"axis", a.axis}, {"min", a.min}, {"max", a.max},
                          {"value", a.value}, {"initial", a.initial}, {"order", i++}};
    bones[pb.id] = {{"name", pb.name}, {"label", pb.label}, {"group", pb.group}, {"order", order++}, {"axes", axes}};
  }
  return {{"figure", f.id}, {"changed", f.rig->changed()}, {"bones", bones}};
}

}  // namespace

struct FvHost::Impl
{
  GltfRenderer*               renderer = nullptr;
  std::shared_ptr<FvProtocol> protocol;
  nvapp::Application*         app = nullptr;

  std::mutex settingsMutex;
  HostSettings   settings;

  // work for the application thread, run in onUpdate
  std::mutex                        queueMutex;
  std::deque<std::function<void()>> queue;
  std::atomic<bool>                 accepting{true};

  // viewport input, applied in onUpdate
  std::mutex        inputMutex;
  std::vector<Json> input;
  double            mouseX = 0, mouseY = 0;
  bool              lmb = false, mmb = false;

  // application thread only
  std::unique_ptr<Figure> figure;
  int                     figures = 0;
  // frames until the camera frames a newly loaded figure (0: none pending)
  int frameIn = 0;

  std::mutex               loadMutex;
  std::vector<std::thread> loaders;
  std::atomic<uint64_t>    loadGeneration{0};
  std::atomic<uint64_t>    jobs{0};
  std::atomic<uint64_t>    notices{0};

  // Runs `f` on the application thread and waits for it (from an HTTP thread).
  template <class F>
  auto onApp(F f) -> decltype(f())
  {
    using R     = decltype(f());
    auto task   = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto result = task->get_future();
    {
      std::lock_guard lock(queueMutex);
      if(!accepting)
        throw std::runtime_error("the renderer is stopping");
      queue.emplace_back([task] { (*task)(); });
    }
    if(result.wait_for(30s) != std::future_status::ready)
      throw std::runtime_error("timed out waiting for the renderer");
    return result.get();
  }

  void post(std::function<void()> f)
  {
    std::lock_guard lock(queueMutex);
    if(accepting)
      queue.push_back(std::move(f));
  }

  void notice(const std::string& text, const char* level = "warning")
  {
    protocol->events().patch("notices", {{std::to_string(++notices), {{"text", text}, {"level", level}}}});
  }

  void job(uint64_t id, const Json& state) { protocol->events().patch("jobs", {{std::to_string(id), state}}); }

  Json load(const Json& args);
  void install(std::unique_ptr<Figure> f, fv::build::BuildResult built, uint64_t generation, uint64_t jobId);
  Json poseSet(const Json& args);
  void applyMoved(const std::vector<int>& moved);
  void frameCamera();
  void applyInput(const std::vector<Json>& events);
};

Json FvHost::Impl::load(const Json& args)
{
  HostSettings s;
  {
    std::lock_guard lock(settingsMutex);
    s = settings;
  }
  // an absolute path, or a path in the content folders (a library item)
  std::optional<std::pair<std::vector<std::filesystem::path>, std::string>> library;
  std::string                                                              name;
  if(args.contains("rel"))
  {
    const std::string rel = args["rel"].get<std::string>();
    library               = std::make_pair(s.roots, rel);
    name                  = rel;
  }
  else
  {
    const std::string path = args.at("path").get<std::string>();
    library                = libraryFor(s, pathOf(path));
    if(!library)
      throw std::runtime_error(path + " is not in a DAZ content library (no data folder above it)");
    name = path;
  }
  const uint64_t    generation = ++loadGeneration;
  const uint64_t    jobId      = ++jobs;
  const std::string label      = "Loading " + utf8Of(pathOf(name).filename());
  job(jobId, {{"label", label}, {"state", "running"}});

  std::lock_guard lock(loadMutex);
  loaders.emplace_back([this, s, roots = library->first, rel = library->second, generation, jobId, label] {
    try
    {
      const auto t0  = std::chrono::steady_clock::now();
      auto       lib = std::make_shared<fv::dson::Library>(roots);
      fv::dson::Store store(lib);
      auto            f = std::make_unique<Figure>();
      f->result         = fv::dson::loadDson(store, rel, {.subdivision = s.subdivision});
      if(generation != loadGeneration)
        return;  // a newer load replaced this one
      for(const auto& w : f->result.warnings)
        notice(w);
      // the figure's skeleton poses; worn items' skeletons follow it by bone name
      if(f->result.mainRig >= 0)
      {
        std::vector<const fv::dson::Rig*> followers;
        for(size_t r = 0; r < f->result.rigs.size(); r++)
          if(int(r) != f->result.mainRig)
            followers.push_back(&f->result.rigs[r]);
        f->rig = std::make_unique<fv::pose::PoseRig>(f->result.scene, f->result.rigs[f->result.mainRig], followers);
      }
      // centimetres to metres, after the skins were bound (as loaders/index.ts scales)
      auto& root = f->result.scene.nodes[f->result.root];
      root.scale = root.scale * kCentimetre;

      // textures: every one the build asks for, resolved in parallel
      const bool cooked = !s.texServer.empty() && !s.rootId.empty();
      std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
      {
        const auto                                                requests = fv::build::textureRequests(f->result.scene);
        std::atomic<size_t>                                       next{0};
        std::vector<std::optional<fv::build::TextureImage>>      found(requests.size());
        std::vector<std::thread>                                  workers;
        for(int w = 0; w < std::min<int>(8, int(requests.size())); w++)
          workers.emplace_back([&] {
            for(size_t i; (i = next++) < requests.size();)
              found[i] = cooked ? cookedTexture(s, requests[i]) : sourceTexture(*lib, requests[i]);
          });
        for(auto& w : workers)
          w.join();
        for(size_t i = 0; i < requests.size(); i++)
          images[fv::build::requestKey(requests[i])] = found[i];
      }
      fv::build::BuildOptions options;
      options.texturesFlipped = cooked;
      options.texture         = [&](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
        auto it = images.find(fv::build::requestKey(req));
        return it == images.end() ? std::nullopt : it->second;
      };
      auto built = fv::build::buildGltf(f->result.scene, f->result.root, options);
      LOGI("[fv] %s built in %.2f s: %zu nodes, %zu meshes, %zu materials, %zu images, %zu bytes of geometry\n", rel.c_str(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), built.model.nodes.size(), built.model.meshes.size(),
           built.model.materials.size(), built.model.images.size(), built.model.buffers.empty() ? size_t(0) : built.model.buffers[0].data.size());
      // the renderer takes it on the application thread
      auto shared = std::make_shared<std::pair<std::unique_ptr<Figure>, fv::build::BuildResult>>(std::move(f), std::move(built));
      post([this, shared, generation, jobId] { install(std::move(shared->first), std::move(shared->second), generation, jobId); });
    }
    catch(const std::exception& e)
    {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", e.what()}});
      notice(e.what(), "error");
    }
  });
  return {{"job", jobId}};
}

void FvHost::Impl::install(std::unique_ptr<Figure> f, fv::build::BuildResult built, uint64_t generation, uint64_t jobId)
{
  if(generation != loadGeneration)
    return;
  f->nodeOf = std::move(built.nodeOf);
  if(!renderer->loadModel(std::move(built.model)))
  {
    job(jobId, {{"state", "failed"}, {"error", "the renderer could not take the scene"}});
    return;
  }
  f->id  = "fig:" + std::to_string(++figures);
  figure = std::move(f);
  // the renderer sets its own camera up over the next frames; frame the figure after that
  frameIn = 2;
  auto& events = protocol->events();
  events.snapshot("scene", {{"figures", {{figure->id, {{"name", figure->result.name}}}}}});
  if(figure->rig)
    events.snapshot("pose/" + figure->id, poseSnapshot(*figure));
  job(jobId, {{"state", "done"}, {"figure", figure->id}});
}

void FvHost::Impl::applyMoved(const std::vector<int>& moved)
{
  std::vector<GltfRenderer::NodeTransform> transforms;
  transforms.reserve(moved.size());
  for(int n : moved)
  {
    const int g = n >= 0 && n < int(figure->nodeOf.size()) ? figure->nodeOf[n] : -1;
    if(g < 0)
      continue;
    const fv::Node& node = figure->result.scene.nodes[n];
    transforms.push_back({g, glm::vec3(node.position.x, node.position.y, node.position.z),
                          glm::quat(float(node.quaternion.w), float(node.quaternion.x), float(node.quaternion.y), float(node.quaternion.z)),
                          glm::vec3(node.scale.x, node.scale.y, node.scale.z)});
  }
  renderer->setNodeTransforms(transforms);
}

Json FvHost::Impl::poseSet(const Json& args)
{
  if(!figure || !figure->rig || args.value("figure", figure->id) != figure->id)
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
  applyMoved(figure->rig->setMany(values));
  // the values as the rig took them (clamped), for every panel showing the pose
  Json bones = Json::object();
  for(const auto& v : values)
    if(const auto* pb = figure->rig->get(v.bone))
      for(const auto& a : pb->axes)
        if(a.kind == v.kind && a.axis == v.axis)
          bones[pb->id]["axes"][axisKey(a)]["value"] = a.value;
  protocol->events().patch("pose/" + figure->id, {{"changed", figure->rig->changed()}, {"bones", bones}});
  return {{"changed", figure->rig->changed()}};
}

void FvHost::Impl::frameCamera()
{
  nvvkgltf::Scene* scene = renderer->editScene();
  auto             manip = renderer->getCameraManipulator();
  if(!scene || !manip)
    return;
  const nvutils::Bbox box  = scene->getSceneBounds();
  const VkExtent2D    size = renderer->renderSize();
  manip->fit(box.min(), box.max(), true, false, size.height ? double(size.width) / size.height : 1.0);
}

void FvHost::Impl::applyInput(const std::vector<Json>& events)
{
  auto manip = renderer->getCameraManipulator();
  if(!manip)
    return;
  for(const Json& e : events)
  {
    const std::string type = e.value("t", "");
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

//--------------------------------------------------------------------------------------------------

FvHost::FvHost(GltfRenderer* renderer, std::shared_ptr<FvProtocol> protocol)
    : m_impl(std::make_unique<Impl>())
{
  Impl& s     = *m_impl;
  s.renderer  = renderer;
  s.protocol  = std::move(protocol);
  auto& cmds  = s.protocol->commands();
  Impl* impl  = m_impl.get();

  cmds.add("settings.apply", [impl](const Json& args) {
    std::lock_guard lock(impl->settingsMutex);
    HostSettings&       st = impl->settings;
    if(args.contains("libraryRoots"))
    {
      st.roots.clear();
      for(const auto& r : args["libraryRoots"])
        st.roots.push_back(pathOf(r.get<std::string>()));
    }
    st.rootId         = args.value("rootId", st.rootId);
    st.texServer      = args.value("texServer", st.texServer);
    st.maxTextureSize = args.value("maxTextureSize", st.maxTextureSize);
    st.subdivision    = args.value("subdivision", st.subdivision);
    return Json{{"ok", true}};
  });
  cmds.add("load", [impl](const Json& args) { return impl->load(args); });
  cmds.add("pose.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->poseSet(args); }); });
  cmds.add("pose.reset", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      if(!impl->figure || !impl->figure->rig)
        throw std::runtime_error("no figure loaded");
      impl->applyMoved(impl->figure->rig->reset());
      impl->protocol->events().snapshot("pose/" + impl->figure->id, poseSnapshot(*impl->figure));
      return Json{{"changed", 0}};
    });
  });
  cmds.add("render.mode", [impl](const Json& args) {
    const std::string mode = args.at("mode").get<std::string>();
    if(mode != "raster" && mode != "rtx")
      throw std::runtime_error("mode is raster or rtx");
    return impl->onApp([impl, mode] {
      impl->renderer->setRenderSystem(mode == "rtx" ? RenderingMode::ePathtracer : RenderingMode::eRasterizer);
      impl->protocol->events().snapshot("render", {{"mode", mode}});
      return Json{{"mode", mode}};
    });
  });
  cmds.add("camera.frame", [impl](const Json&) {
    return impl->onApp([impl] {
      impl->frameCamera();
      return Json{{"ok", true}};
    });
  });
  // a clean exit (DLSS/NGX shut down), which the host asks for before it ends the process
  cmds.add("session.shutdown", [impl](const Json&) {
    impl->post([impl] {
      if(impl->app)
        impl->app->close();
    });
    return Json{{"ok", true}};
  });
  s.protocol->setInputSink([impl](const Json& events) {
    std::lock_guard lock(impl->inputMutex);
    for(const auto& e : events)
      impl->input.push_back(e);
  });
}

FvHost::~FvHost() = default;

void FvHost::onAttach(nvapp::Application* app)
{
  m_impl->app = app;
  m_impl->protocol->events().snapshot(
      "render", {{"mode", m_impl->renderer->renderSystem() == RenderingMode::ePathtracer ? "rtx" : "raster"}});
}

void FvHost::onDetach()
{
  Impl& s = *m_impl;
  {
    std::lock_guard lock(s.queueMutex);
    s.accepting = false;
  }
  s.loadGeneration++;  // loads still running drop their result
  std::lock_guard lock(s.loadMutex);
  for(auto& t : s.loaders)
    if(t.joinable())
      t.join();
  s.loaders.clear();
  s.queue.clear();
  s.figure.reset();
}

void FvHost::onUpdate()
{
  Impl&                             s = *m_impl;
  std::deque<std::function<void()>> work;
  {
    std::lock_guard lock(s.queueMutex);
    work.swap(s.queue);
  }
  for(auto& f : work)
    f();
  if(s.frameIn > 0 && --s.frameIn == 0)
    s.frameCamera();
  std::vector<Json> events;
  {
    std::lock_guard lock(s.inputMutex);
    events.swap(s.input);
  }
  if(!events.empty())
    s.applyInput(events);
}
