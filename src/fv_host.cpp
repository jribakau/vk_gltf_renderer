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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvmcp/parameter_tools.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_registry.hpp>

#include <figure/dson/material_preset.hpp>
#include <figure/dson/presets.hpp>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>

#include "env_image_loader.hpp"
#include "fv_lighting.hpp"
#include "fv_materials.hpp"
#include "fv_protocol.hpp"
#include "fv_view.hpp"
#include "gltf_environment_sky.hpp"
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
struct Loaded
{
  std::string                 id;
  std::unique_ptr<fv::Figure> fig;
  std::vector<int>            nodeOf;   // scene node -> glTF node
  std::unordered_map<int, int> sceneOf;  // glTF node -> scene node
  std::vector<int>            meshOf;   // scene mesh -> glTF mesh
  std::vector<int>            skinOf;   // scene mesh -> glTF skin
  std::vector<std::vector<int>> primsOf;  // glTF mesh -> render primitives
  // morph topic: -1 until the channel list has been published, then the value and weight last sent
  int                                                          morphList = -1;
  std::unordered_map<std::string, std::pair<double, double>> published;
  uint64_t                                                     busyJob = 0;
  bool                                                         busyOn  = false;
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

// A number as the cooker parses it, five decimals at most.
std::string fmtNumber(double v)
{
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.5f", v);
  std::string out = buf;
  while(out.size() > 1 && out.back() == '0')
    out.pop_back();
  if(!out.empty() && out.back() == '.')
    out.pop_back();
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

// Whether the renderer reads this image itself (else only the cooker can: .tif, .bmp, ...).
bool readable(const std::string& path)
{
  std::string ext = path.substr(path.find_last_of('.') + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "webp";
}

struct CookedTexture
{
  std::optional<fv::build::TextureImage> image;
  bool                                   uncooked = false;  // asked cachedOnly, and not cooked yet
};

// Where a cooked texture is (BC7 KTX2, cutout folded into alpha, rows bottom-up for the DSON UVs
// when `flip`). The texture server cooks it first, which takes seconds per 4K map; `cachedOnly`
// answers at once instead, `uncooked` when it isn't cooked.
CookedTexture cookedTexture(const HostSettings& s, const fv::build::TextureRequest& req, bool flip, bool cachedOnly, bool quiet = false)
{
  // http://127.0.0.1:<port>/<token>
  const std::string& base = s.texServer;
  const size_t       hostAt = base.find("://");
  const size_t       pathAt = base.find('/', hostAt == std::string::npos ? 0 : hostAt + 3);
  if(hostAt == std::string::npos || pathAt == std::string::npos)
    return {};
  httplib::Client cli(base.substr(0, pathAt));
  cli.set_read_timeout(cachedOnly ? 30s : 600s);
  std::string url = base.substr(pathAt) + "/" + s.rootId + "/" + encodeRel(normRel(req.rel)) + "?srgb=" + (req.srgb ? "1" : "0")
                    + "&flip=" + (flip ? "1" : "0") + "&max=" + std::to_string(s.maxTextureSize);
  if(req.normal)
    url += "&nm=1";
  if(!req.channels.empty())
    url += "&ch=" + req.channels;
  // a source in a query value: <rootId>/<path>, percent-encoded
  const auto source = [&](const std::string& rel) { return encodeComponent(s.rootId + "/" + normRel(rel)); };
  if(!req.alphaRel.empty())
    url += "&alpha=" + source(req.alphaRel);
  if(!req.bumpRel.empty())
    url += "&bump=" + source(req.bumpRel) + "&bs=" + fmtNumber(req.bumpStrength);
  if(req.makeup)
  {
    // mk=<base 1|0>|<colour source>|<weight source>|<weight>|<r,g,b tint>|<r,g,b factor>, percent-encoded as a whole
    const fv::mat::MakeupRequest& m = *req.makeup;
    const auto                    rgb = [](const std::array<float, 3>& c) { return fmtNumber(c[0]) + "," + fmtNumber(c[1]) + "," + fmtNumber(c[2]); };
    const std::string             mk = std::string(m.baseIsSource ? "1" : "0") + "|" + (m.colorRel.empty() ? "-" : source(m.colorRel)) + "|"
                           + (m.weightRel.empty() ? "-" : source(m.weightRel)) + "|" + fmtNumber(m.weightFactor) + "|" + rgb(m.tint) + "|" + rgb(m.factor);
    url += "&mk=" + encodeComponent(mk);
  }
  url += cachedOnly ? "&path=1&cached=1" : "&path=1";
  auto res = cli.Get(url);
  if(res && res->status == 202 && cachedOnly)
    return {.uncooked = true};
  if(!res || res->status != 200)
  {
    if(!quiet)
      LOGW("texture %s: %s\n", req.rel.c_str(), res ? (std::to_string(res->status) + " " + res->body).c_str() : "no answer");
    return {};
  }
  return {.image = fv::build::TextureImage{uriOfPath(res->body), "image/ktx2", flip}};
}

// Cooks textures for a later load, on a thread of its own: a figure shows with its source images
// rather than wait for these.
void cookLater(const HostSettings& s, std::vector<fv::build::TextureRequest> requests)
{
  std::thread([s, requests = std::move(requests)] {
    std::atomic<size_t>      next{0};
    std::vector<std::thread> workers;
    for(int w = 0; w < std::min<int>(4, int(requests.size())); w++)
      workers.emplace_back([&] {
        for(size_t i; (i = next++) < requests.size();)
          cookedTexture(s, requests[i], true, false, true);
      });
    for(auto& w : workers)
      w.join();
  }).detach();
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

// A texture without waiting for the cooker: the source image as it is, where that shows the right
// thing. A normal map keeps its green (the material tells the renderer to flip it) and loses the
// bump blended onto it; a height map and a roughness map need the cooker, so the surface goes
// without them until a later load finds them cooked. A cutout alone changes the picture too much
// for that: it and makeup (and formats the renderer cannot read) are cooked, unflipped, and waited for.
std::optional<fv::build::TextureImage> fallbackTexture(const HostSettings& s, const fv::dson::Library& lib, const fv::build::TextureRequest& req, bool cooker)
{
  if(cooker && (!req.alphaRel.empty() || req.makeup || !readable(req.rel)))
    return cookedTexture(s, req, false, false).image;
  if(req.channels == "mr" || req.channels == "bn")
    return std::nullopt;
  auto image = sourceTexture(lib, req);
  if(image && req.channels == "ng")
    image->flipGreen = true;
  return image;
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
Json poseSnapshot(const Loaded& f)
{
  Json bones = Json::object();
  int  order = 0;
  for(const auto& pb : f.fig->pose->bones())
  {
    if(pb.axes.empty())
      continue;
    Json axes = Json::object();
    int  i    = 0;
    for(const auto& a : pb.axes)
      axes[axisKey(a)] = {{"label", a.label}, {"kind", fv::pose::kindName(a.kind)}, {"axis", a.axis}, {"min", a.min}, {"max", a.max},
                          {"value", a.value}, {"initial", a.initial}, {"order", i++}};
    bones[pb.id] = {{"name", pb.name}, {"label", pb.label}, {"group", pb.group}, {"order", order++}, {"node", pb.bone}, {"axes", axes}};
  }
  return {{"figure", f.id}, {"changed", f.fig->pose->changed()}, {"eyes", f.fig->pose->hasEyes()}, {"bones", bones}};
}

std::string fileStem(const std::string& rel)
{
  const size_t slash = rel.find_last_of("/\\");
  std::string  name  = slash == std::string::npos ? rel : rel.substr(slash + 1);
  const size_t dot   = name.find_last_of('.');
  if(dot != std::string::npos)
    name.resize(dot);
  return name;
}

constexpr const char* kStudioName = "Studio (built-in)";
constexpr uint32_t    kStudioWidth = 512, kStudioHeight = 256;

// The registered parameters nvmcp can read and write: values, not triggers or custom parsers.
nvmcp::detail::ParameterList exposedParameters(const nvutils::ParameterRegistry* registry)
{
  nvmcp::detail::ParameterList out;
  if(!registry)
    return out;
  for(const nvutils::ParameterBase* p : registry->getParameters())
    if(p->destination.raw && p->type != nvutils::ParameterBase::Type::BOOL8_TRIGGER
       && p->type != nvutils::ParameterBase::Type::CUSTOM && p->type != nvutils::ParameterBase::Type::INVALID)
      out.push_back(p);
  return out;
}

Json toolJson(const nvmcp::ToolResult& r)
{
  if(r.isError)
    throw std::runtime_error(r.json);
  return Json::parse(r.json);
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
  std::unique_ptr<Loaded> figure;
  int                     figures = 0;
  // morph.set merges here (any thread); onUpdate applies it as one setMany
  std::mutex                     morphMutex;
  std::string                    figureId;
  std::map<std::string, double>  pendingMorphs;
  bool                           morphTouched = false;
  // frames until the camera frames a newly loaded figure (0: none pending)
  int frameIn = 0;
  // the gizmo's latest drag on a bone: the rig takes it once a frame, however many events came
  struct GizmoEdit
  {
    std::string             bone;
    fv::Quat                rotation;
    std::optional<fv::Vec3> position;
  };
  std::optional<GizmoEdit> gizmoEdit;

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
  void install(std::unique_ptr<fv::Figure> f, fv::build::BuildResult built, uint64_t generation, uint64_t jobId, bool texturesFlipped,
               fv::mat::SkinMode skin, std::shared_ptr<fv::dson::Library> lib);
  Json poseSet(const Json& args);
  void applyNodes(const std::set<int>& moved);
  void applyMesh(const fv::morph::MeshChange& change);
  void applyRebind(const std::vector<int>& meshes);
  void applyPendingMorphs();
  void publishMorphs();
  void publishBusy();
  void frameCamera();
  void applyInput(const std::vector<Json>& events);
  void patchPose(const std::vector<std::string>& boneIds);
  // the main figure's posed bone at a glTF node, or nullptr
  const fv::pose::PoseBone* boneAt(int gltf) const;
  void                      applyGizmoEdit();
  void applyPreset(const fv::dson::Json&                             doc,
                   const std::string&                                kind,
                   uint64_t                                          jobId,
                   const std::string&                                rel,
                   const std::optional<fv::dson::EnvironmentPreset>& environment,
                   const std::string&                                environmentPath);

  // ---- lighting (fv_lighting.hpp), application thread only
  nvutils::ParameterRegistry* registry = nullptr;
  fv::lighting::State         lighting;
  bool                        lightingStarted = false;
  Json                        appliedParams   = Json::object();  // what the renderer was last given
  Json                        appliedSun;                        // the sun light as last written
  Json                        publishedLighting;
  // the environment image in the renderer ("studio", "hdr:<path>"), and the one asked for
  std::string loadedEnv;
  std::string wantedEnv;
  uint64_t    envRequest = 0;
  uint64_t    envJob     = 0;
  bool        envLoading = false;
  std::string envName    = kStudioName;
  glm::uvec2  envSize{kStudioWidth, kStudioHeight};
  // the host's sun in the renderer's model, and the ground height the renderer chose at load
  int   sunNode    = -1;
  int   sunLight   = -1;
  float autoGround = 0.0F;

  Json lightingTopic() const;
  Json lightingSet(const Json& patch);
  void applyLighting(bool everything);
  void requestEnvironment();
  void applySun(bool force);
  void addSun(tinygltf::Model& model);
  void setParams(const Json& values);
  void publishLighting();

  std::unique_ptr<FvView> view;
  // the materials of the loaded figure (fv_materials.hpp), and how skin is shaded for the render mode
  std::unique_ptr<FvMaterials>       materials;
  std::atomic<fv::mat::SkinMode>     skinMode{fv::mat::SkinMode::eRaster};
  std::shared_ptr<fv::dson::Library> libraryOfFigure;  // the loaded figure's, for source textures (settingsMutex)
  // figure loads only: a preset must not take a generation, or it would cancel a figure load
  std::mutex figureLoads;
  uint64_t   latestFigureSeq = 0;
  std::atomic<uint64_t> requestSeq{0};
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
  const uint64_t    seq   = ++requestSeq;
  const uint64_t    jobId = ++jobs;
  const std::string label = "Loading " + utf8Of(pathOf(name).filename());
  job(jobId, {{"label", label}, {"state", "running"}});

  std::lock_guard lock(loadMutex);
  loaders.emplace_back([this, s, roots = library->first, rel = library->second, seq, jobId, label] {
    try
    {
      auto lib   = std::make_shared<fv::dson::Library>(roots);
      auto store = std::make_shared<fv::dson::Store>(lib);
      // a preset is applied to the loaded figure; it must not replace it or cancel a figure load
      if(auto doc = store->load(rel))
        if(auto preset = fv::dson::describePreset(doc->json))
        {
          // an environment preset's HDR, found through the preset's own library
          std::string envPath;
          if(preset->environment)
            if(auto found = lib->find(normRel(preset->environment->image)))
              envPath = utf8Of(*found);
          post([this, body = doc->json, kind = preset->kind, jobId, rel, env = preset->environment, envPath] {
            applyPreset(body, kind, jobId, rel, env, envPath);
          });
          return;
        }
      uint64_t generation = 0;
      {
        std::lock_guard gate(figureLoads);
        if(seq < latestFigureSeq)
        {
          job(jobId, {{"label", label}, {"state", "done"}});
          return;
        }
        latestFigureSeq = seq;
        generation      = ++loadGeneration;
      }
      const auto t0 = std::chrono::steady_clock::now();
      auto       f = fv::Figure::load(store, rel, {.subdivision = s.subdivision}, [this, jobId, label](const std::string& step) {
        job(jobId, {{"label", label + ": " + step}, {"state", "running"}});
      });
      if(generation != loadGeneration)
      {
        job(jobId, {{"label", label}, {"state", "done"}});
        return;  // a newer load replaced this one
      }
      for(const auto& w : f->takeWarnings())
        notice(w);
      // centimetres to metres, after the skins were bound (as loaders/index.ts scales). A later
      // rebind reads bone and mesh worlds that both carry this scale, so it cancels back out.
      auto& root = f->loaded.scene.nodes[f->loaded.root];
      root.scale = root.scale * kCentimetre;

      // textures: every one the build asks for, resolved in parallel. The cooked ones when all are
      // cooked; the build flips the UVs for every texture or none, so otherwise the source images,
      // with cutouts (only the cooker folds them into alpha) and .tif and such cooked unflipped.
      // the path tracer gets the skin's volume; the rasterizer would show it see-through
      const fv::mat::SkinMode skin = skinMode.load();
      job(jobId, {{"label", label + ": Textures"}, {"state", "running"}});
      bool cooked = !s.texServer.empty() && !s.rootId.empty();
      std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
      {
        const auto                                          requests = fv::build::textureRequests(f->loaded.scene, skin);
        std::vector<std::optional<fv::build::TextureImage>> found(requests.size());
        std::atomic<bool>                                   uncooked{false};
        const auto resolve = [&](const std::function<std::optional<fv::build::TextureImage>(const fv::build::TextureRequest&)>& one) {
          std::atomic<size_t>      next{0};
          std::vector<std::thread> workers;
          for(int w = 0; w < std::min<int>(8, int(requests.size())); w++)
            workers.emplace_back([&] {
              for(size_t i; (i = next++) < requests.size();)
                found[i] = one(requests[i]);
            });
          for(auto& w : workers)
            w.join();
        };
        if(cooked)
          resolve([&](const fv::build::TextureRequest& req) {
            CookedTexture c = cookedTexture(s, req, true, true);
            if(c.uncooked)
              uncooked = true;
            return c.image;
          });
        if(!cooked || uncooked)
        {
          resolve([&](const fv::build::TextureRequest& req) { return fallbackTexture(s, *lib, req, cooked); });
          if(cooked)
            cookLater(s, requests);
          cooked = false;
        }
        for(size_t i = 0; i < requests.size(); i++)
          images[fv::build::requestKey(requests[i])] = found[i];
      }
      fv::build::BuildOptions options;
      options.texturesFlipped = cooked;
      options.skin            = skin;
      options.texture         = [&](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
        auto it = images.find(fv::build::requestKey(req));
        return it == images.end() ? std::nullopt : it->second;
      };
      auto built = fv::build::buildGltf(f->loaded.scene, f->loaded.root, options);
      LOGI("[fv] %s built in %.2f s: %zu nodes, %zu meshes, %zu materials, %zu images, %zu bytes of geometry\n", rel.c_str(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), built.model.nodes.size(), built.model.meshes.size(),
           built.model.materials.size(), built.model.images.size(), built.model.buffers.empty() ? size_t(0) : built.model.buffers[0].data.size());
      // the renderer takes it on the application thread
      auto shared = std::make_shared<std::pair<std::unique_ptr<fv::Figure>, fv::build::BuildResult>>(std::move(f), std::move(built));
      const bool flipped = cooked;
      post([this, shared, generation, jobId, flipped, skin, lib] {
        install(std::move(shared->first), std::move(shared->second), generation, jobId, flipped, skin, lib);
      });
    }
    catch(const std::exception& e)
    {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", e.what()}});
      notice(e.what(), "error");
    }
  });
  return {{"job", jobId}};
}

void FvHost::Impl::install(std::unique_ptr<fv::Figure> fig, fv::build::BuildResult built, uint64_t generation, uint64_t jobId, bool texturesFlipped,
                           fv::mat::SkinMode skin, std::shared_ptr<fv::dson::Library> lib)
{
  if(generation != loadGeneration)
    return;
  auto f     = std::make_unique<Loaded>();
  f->fig     = std::move(fig);
  f->nodeOf  = std::move(built.nodeOf);
  for(int i = 0; i < int(f->nodeOf.size()); i++)
    if(f->nodeOf[i] >= 0)
      f->sceneOf[f->nodeOf[i]] = i;
  f->meshOf  = std::move(built.meshOf);
  f->skinOf  = std::move(built.skinOf);
  addSun(built.model);
  if(!renderer->loadModel(std::move(built.model)))
  {
    sunNode = sunLight = -1;
    job(jobId, {{"state", "failed"}, {"error", "the renderer could not take the scene"}});
    return;
  }
  // the renderer put the ground at the scene's lowest point; "auto" keeps that
  for(const nvutils::ParameterBase* p : exposedParameters(registry))
    if(p->info.name == "infinitePlaneDistance" && p->type == nvutils::ParameterBase::Type::FLOAT32)
      autoGround = *static_cast<const float*>(p->destination.raw);
  appliedParams.erase("infinitePlaneDistance");
  applyLighting(false);
  int meshes = 0;
  for(int m : f->meshOf)
    meshes = std::max(meshes, m + 1);
  f->primsOf.resize(meshes);
  for(int i = 0; i < meshes; i++)
    f->primsOf[i] = renderer->renderPrimitivesOfMesh(i);
  // the glTF already holds the scene's dials and binds; don't upload them again
  f->fig->drain();
  if(figure && figure->busyOn)
    job(figure->busyJob, {{"state", "done"}});
  f->id  = "fig:" + std::to_string(++figures);
  figure = std::move(f);
  {
    std::lock_guard lock(settingsMutex);
    libraryOfFigure = std::move(lib);
  }
  materials->attach(figure->id, &figure->fig->loaded.scene, built, texturesFlipped, skin);
  {
    std::lock_guard lock(morphMutex);
    figureId = figure->id;
    pendingMorphs.clear();
  }
  morphTouched = true;
  // the renderer sets its own camera up over the next frames; frame the figure after that
  frameIn = 2;
  // the pose first: a panel that sees the figure in "scene" finds its pose already there
  if(figure->fig->pose)
    protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
  if(view)
  {
    FvView::Bones bones;
    if(figure->fig->pose)
    {
      for(const auto& pb : figure->fig->pose->bones())
        if(!pb.axes.empty())
          bones.idOf[pb.bone] = pb.id;
      bones.nearest = [this](const glm::vec3& hit) {
        const auto* pb = figure && figure->fig->pose ? figure->fig->pose->nearestBone({hit.x, hit.y, hit.z}) : nullptr;
        return pb ? pb->bone : -1;
      };
    }
    view->setFigure(figure->id, figure->fig->loaded.name, figure->fig->loaded.scene, figure->fig->loaded.root, figure->nodeOf,
                    figure->fig->morphs ? figure->fig->morphs->size() : 0, std::move(bones));
  }
  job(jobId, {{"state", "done"}, {"figure", figure->id}});
}

const fv::pose::PoseBone* FvHost::Impl::boneAt(int gltf) const
{
  if(!figure || !figure->fig->pose)
    return nullptr;
  const auto it = figure->sceneOf.find(gltf);
  const auto* pb = it == figure->sceneOf.end() ? nullptr : figure->fig->pose->boneForNode(it->second);
  return pb && !pb->axes.empty() ? pb : nullptr;
}

void FvHost::Impl::applyGizmoEdit()
{
  if(!gizmoEdit)
    return;
  const GizmoEdit edit = *std::exchange(gizmoEdit, std::nullopt);
  if(!figure || !figure->fig->pose)
    return;
  // nodes, correctives and rebound skins go out with this frame's pump()
  figure->fig->pose->setLocalTransform(edit.bone, edit.rotation, edit.position);
  morphTouched = true;
  patchPose({edit.bone});
}

void FvHost::Impl::patchPose(const std::vector<std::string>& boneIds)
{
  if(!figure || !figure->fig->pose)
    return;
  Json bones = Json::object();
  for(const std::string& id : boneIds)
    if(const auto* pb = figure->fig->pose->get(id))
    {
      Json axes = Json::object();
      for(const auto& a : pb->axes)
        axes[axisKey(a)] = {{"value", a.value}};
      bones[pb->id] = {{"axes", axes}};
    }
  protocol->events().patch("pose/" + figure->id, {{"changed", figure->fig->pose->changed()}, {"bones", bones}});
}

void FvHost::Impl::applyPreset(const fv::dson::Json&                             doc,
                               const std::string&                                kind,
                               uint64_t                                          jobId,
                               const std::string&                                rel,
                               const std::optional<fv::dson::EnvironmentPreset>& environment,
                               const std::string&                                environmentPath)
{
  const std::string name = fileStem(rel);
  if(environment)
  {
    // as applyEnvironmentPreset (src/lib/actions.ts): the dome lights the scene, so the sun goes off
    const std::string hdrName = utf8Of(pathOf(normRel(environment->image)).filename());
    if(environmentPath.empty())
      notice(name + ": its environment image " + environment->image + " isn't in the content library.", "error");
    else
    {
      Json env = {{"source", "hdr"},
                  {"path", environmentPath},
                  {"intensity", environment->intensity.value_or(1.0)},
                  {"rotation", fv::lighting::presetRotation(environment->rotation.value_or(0.0))}};
      if(environment->drawDome.value_or(false))
        env["background"] = "environment";
      lightingSet({{"env", env}, {"sun", {{"enabled", false}}}});
      notice("Applied lighting preset " + name + " (" + hdrName + "). The sun light is off; turn it back on in the Light tab.", "info");
    }
    job(jobId, {{"state", "done"}, {"preset", kind}});
    return;
  }
  if(kind == "pose preset")
  {
    if(!figure || !figure->fig->pose)
      notice(name + " is a DAZ pose preset. Load a DAZ figure first, then open the pose.");
    else
    {
      const fv::PosePresetResult r = figure->fig->applyPosePreset(doc);
      morphTouched                 = true;
      protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
      if(!r.applied)
        notice(name + ": nothing in this pose matches the loaded figure.");
      else if(!r.skipped.empty())
        notice("Applied pose " + name + ". " + std::to_string(r.skipped.size()) + " pose controls aren't supported yet and were skipped.", "info");
      else
        notice("Applied pose " + name, "info");
    }
  }
  else if(kind == "material preset")
  {
    if(!figure)
      notice(name + " is a DAZ material preset. Load a DAZ figure first, then open the preset.");
    else
    {
      const fv::dson::MaterialPresetResult r = fv::dson::applyMaterialPreset(doc, figure->fig->loaded.scene);
      materials->surfacesChanged(r.changed);
      if(r.changed.empty())
        notice(name + ": none of its surfaces match the loaded figure.");
      else
        notice("Applied material preset " + name + " to " + std::to_string(r.changed.size()) + " surfaces.", "info");
    }
  }
  else
    notice(name + " is a DAZ " + kind + " and isn't supported in the native viewport yet.");
  job(jobId, {{"state", "done"}, {"preset", kind}});
}

Json FvHost::Impl::lightingTopic() const
{
  Json t           = fv::lighting::toJson(lighting);
  t["env"]["name"]    = envName;
  t["env"]["size"]    = {envSize.x, envSize.y};
  t["env"]["loading"] = envLoading;
  return t;
}

void FvHost::Impl::publishLighting()
{
  const Json now = lightingTopic();
  if(publishedLighting.is_null())
    protocol->events().snapshot("lighting", now);
  else if(Json diff = fv::lighting::mergeDiff(publishedLighting, now); !diff.empty())
    protocol->events().patch("lighting", diff);
  publishedLighting = now;
}

Json FvHost::Impl::lightingSet(const Json& patch)
{
  fv::lighting::State next = fv::lighting::applyPatch(lighting, patch);  // throws, leaving everything as it was
  if(next.env.source == "hdr" && next.env.path.empty())
    throw std::invalid_argument("lighting: env.path is required for an hdr environment");
  lighting = std::move(next);
  applyLighting(false);
  return lightingTopic();
}

// Writes what changed since the last call (everything, the first time) to the renderer: its
// parameters, the sun light, and the environment image, which loads as a job.
void FvHost::Impl::applyLighting(bool everything)
{
  if(everything)
  {
    appliedParams = Json::object();
    appliedSun    = nullptr;
  }
  requestEnvironment();

  Json params                     = fv::lighting::rendererParams(lighting, envSize);
  params["infinitePlaneDistance"] = lighting.ground.autoHeight ? double(autoGround) : lighting.ground.height;
  Json changed                    = Json::object();
  for(auto it = params.begin(); it != params.end(); ++it)
    if(!appliedParams.contains(it.key()) || appliedParams[it.key()] != it.value())
      changed[it.key()] = it.value();
  if(!changed.empty())
  {
    setParams(changed);
    appliedParams.update(changed);
  }
  applySun(everything);
  publishLighting();
}

// The environment image the state asks for, if the renderer doesn't have it: the studio is built
// here, an HDR decodes on a worker. Each new choice bumps the request, so a load that finishes
// after a newer one is dropped.
void FvHost::Impl::requestEnvironment()
{
  const fv::lighting::Env& e    = lighting.env;
  const std::string        want = e.source == "studio" ? "studio" : e.source == "hdr" ? "hdr:" + e.path : "";
  if(want == wantedEnv)
    return;
  wantedEnv           = want;
  const uint64_t req  = ++envRequest;
  const bool     wasLoading = envLoading;
  envLoading          = false;
  if(wasLoading && envJob)
    job(envJob, {{"state", "done"}});
  if(want.empty() || want == loadedEnv)
  {
    // a sky, a plain colour, none, or the image already there: the envSystem parameter does it
    if(!want.empty())
      appliedParams.erase("envSystem");
    return;
  }
  if(want == "studio")
  {
    EnvImage img;
    img.pixels = fv::lighting::studioImage(kStudioWidth, kStudioHeight);
    img.size   = {kStudioWidth, kStudioHeight};
    if(renderer->loadHdrEnvironment(std::move(img), {}))
    {
      loadedEnv = want;
      envName   = kStudioName;
      envSize   = {kStudioWidth, kStudioHeight};
    }
    return;
  }
  const std::string path  = e.path;
  const std::string label = "Loading " + utf8Of(pathOf(path).filename());
  envLoading              = true;
  envJob                  = ++jobs;
  job(envJob, {{"label", label}, {"state", "running"}});
  std::lock_guard lock(loadMutex);
  loaders.emplace_back([this, path, want, req, jobId = envJob, label] {
    auto image = std::make_shared<EnvImage>(loadEnvImage(pathOf(path)));
    post([this, image, path, want, req, jobId, label] {
      if(req != envRequest)
        return;  // another environment was chosen meanwhile, and closed this job
      envLoading = false;
      const glm::uvec2 size{image->size.width, image->size.height};
      if(!image->valid() || !renderer->loadHdrEnvironment(std::move(*image), pathOf(path)))
      {
        job(jobId, {{"label", label}, {"state", "failed"}, {"error", "could not read " + path}});
        notice("Could not read the environment " + path, "error");
        // the panel shows what is on screen: the HDR that was there before, else the studio
        if(loadedEnv.rfind("hdr:", 0) == 0)
        {
          wantedEnv         = loadedEnv;
          lighting.env.path = loadedEnv.substr(4);
          publishLighting();
        }
        else
        {
          lighting.env.source = "studio";
          lighting.env.path.clear();
          applyLighting(false);
        }
        return;
      }
      loadedEnv = want;
      envName   = utf8Of(pathOf(path).filename());
      envSize   = size;
      job(jobId, {{"label", label}, {"state", "done"}});
      appliedParams.erase("envSystem");
      applyLighting(false);  // hdrBlur depends on the image's size
    });
  });
}

// The sun light's colour, strength and size, and its node's rotation, when they changed.
void FvHost::Impl::applySun(bool force)
{
  nvvkgltf::Scene* scene = renderer->editScene();
  if(!scene || sunLight < 0 || sunNode < 0 || sunLight >= int(scene->getModel().lights.size())
     || sunNode >= int(scene->getModel().nodes.size()))
    return;
  const fv::lighting::Sun& s     = lighting.sun;
  const glm::vec3          color = fv::lighting::linearColor(s.color);
  const Json               light = {{"color", {color.x, color.y, color.z}},
                                    {"intensity", s.enabled ? s.intensity : 0.0},
                                    {"radius", fv::lighting::sunRadiusKm(s.softness)}};
  const Json               aim   = {s.azimuth, s.elevation};
  if(force || appliedSun.is_null() || appliedSun["light"] != light)
  {
    tinygltf::Light& l = scene->editor().getLightForEdit(sunLight);
    l.color            = {color.x, color.y, color.z};
    l.intensity        = light["intensity"].get<double>();
    tinygltf::Value::Object extras = l.extras.IsObject() ? l.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
    extras["radius"]               = tinygltf::Value(light["radius"].get<double>());
    l.extras                       = tinygltf::Value(extras);
    scene->markLightDirty(sunLight);
  }
  if(force || appliedSun.is_null() || appliedSun["aim"] != aim)
  {
    const GltfRenderer::NodeTransform t{.node = sunNode, .rotation = fv::lighting::sunRotation(s.azimuth, s.elevation)};
    renderer->setNodeTransforms({&t, 1});
  }
  appliedSun = {{"light", light}, {"aim", aim}};
}

// A directional "Sun" on a root node of a model about to be loaded, marked as the sky's sun so a
// physical sky follows it. Written with the current state, so the load shows it as it is.
void FvHost::Impl::addSun(tinygltf::Model& model)
{
  const fv::lighting::Sun& s     = lighting.sun;
  const glm::vec3          color = fv::lighting::linearColor(s.color);
  tinygltf::Light          light;
  light.name      = "Sun";
  light.type      = "directional";
  light.color     = {color.x, color.y, color.z};
  light.intensity = s.enabled ? s.intensity : 0.0;
  tinygltf::Value::Object lightExtras;
  lightExtras["radius"] = tinygltf::Value(fv::lighting::sunRadiusKm(s.softness));
  light.extras          = tinygltf::Value(lightExtras);
  model.lights.push_back(light);
  sunLight = int(model.lights.size()) - 1;

  tinygltf::Node node;
  node.name  = "Sun";
  node.light = sunLight;
  const glm::quat r = fv::lighting::sunRotation(s.azimuth, s.elevation);
  node.rotation     = {r.x, r.y, r.z, r.w};
  tinygltf::Value::Object nodeExtras;
  nodeExtras["fvId"] = tinygltf::Value(std::string("l:sun"));
  node.extras        = tinygltf::Value(nodeExtras);
  gltf_environment_sky::setSkySunMarker(node, gltf_environment_sky::kSkySunOwnerScene);
  model.nodes.push_back(node);
  sunNode = int(model.nodes.size()) - 1;

  if(model.scenes.empty())
    model.scenes.emplace_back();
  const int scene = model.defaultScene >= 0 && model.defaultScene < int(model.scenes.size()) ? model.defaultScene : 0;
  model.scenes[scene].nodes.push_back(sunNode);
  if(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), "KHR_lights_punctual") == model.extensionsUsed.end())
    model.extensionsUsed.push_back("KHR_lights_punctual");
  appliedSun = {{"light", {{"color", {color.x, color.y, color.z}}, {"intensity", light.intensity}, {"radius", fv::lighting::sunRadiusKm(s.softness)}}},
                {"aim", {s.azimuth, s.elevation}}};
}

// A batch of renderer parameters, validated whole before any is applied; callbacks fire as over MCP.
void FvHost::Impl::setParams(const Json& values)
{
  if(!registry || values.empty())
    return;
  const auto params = exposedParameters(registry);
  toolJson(nvmcp::detail::setParameters(params, Json{{"values", values}}.dump()));
}

void FvHost::Impl::applyNodes(const std::set<int>& moved)
{
  if(moved.empty() || !figure)
    return;
  std::vector<GltfRenderer::NodeTransform> transforms;
  transforms.reserve(moved.size());
  const fv::SceneGraph& scene = figure->fig->loaded.scene;
  for(int n : moved)
  {
    const int g = n >= 0 && n < int(figure->nodeOf.size()) ? figure->nodeOf[n] : -1;
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

void FvHost::Impl::applyMesh(const fv::morph::MeshChange& change)
{
  if(!figure || change.mesh < 0 || change.mesh >= int(figure->fig->loaded.scene.meshes.size()))
    return;
  const int gltfMesh = change.mesh < int(figure->meshOf.size()) ? figure->meshOf[change.mesh] : -1;
  if(gltfMesh < 0 || gltfMesh >= int(figure->primsOf.size()) || figure->primsOf[gltfMesh].empty())
    return;
  const auto& geo = figure->fig->loaded.scene.meshes[change.mesh].geo;
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
  const auto& prims = figure->primsOf[gltfMesh];
  for(size_t i = 0; i < prims.size(); i++)
  {
    GltfRenderer::VertexUpload one = i + 1 == prims.size() ? std::move(upload) : upload;
    one.renderPrimID               = prims[i];
    renderer->queueVertexUpload(std::move(one));
  }
}

void FvHost::Impl::applyRebind(const std::vector<int>& meshes)
{
  if(!figure)
    return;
  for(int mesh : meshes)
  {
    if(mesh < 0 || mesh >= int(figure->skinOf.size()) || mesh >= int(figure->fig->loaded.scene.meshes.size()))
      continue;
    const int skin = figure->skinOf[mesh];
    if(skin < 0)
      continue;
    const fv::Mesh&          m   = figure->fig->loaded.scene.meshes[mesh];
    const std::vector<float> ibm = fv::build::inverseBindMatrices(m);
    if(ibm.size() != m.bones.size() * 16)
      continue;
    static_assert(sizeof(glm::mat4) == 64, "inverse bind matrices are packed mat4");
    std::vector<glm::mat4> mats(m.bones.size());
    std::memcpy(mats.data(), ibm.data(), ibm.size() * sizeof(float));
    renderer->setInverseBindMatrices(skin, mats);
  }
}

void FvHost::Impl::applyPendingMorphs()
{
  std::vector<std::pair<std::string, double>> values;
  {
    std::lock_guard lock(morphMutex);
    if(pendingMorphs.empty())
      return;
    values.reserve(pendingMorphs.size());
    for(const auto& [name, value] : pendingMorphs)
      values.emplace_back(name, value);
    pendingMorphs.clear();
  }
  if(!figure || !figure->fig->morphs)
    return;
  figure->fig->morphs->setMany(values);
  morphTouched = true;
}

void FvHost::Impl::publishMorphs()
{
  if(!figure || !figure->fig->morphs)
    return;
  fv::morph::MorphSet& morphs = *figure->fig->morphs;
  const bool           listChanged = morphs.listVersion() != figure->morphList;
  if(!listChanged && !morphTouched)
    return;
  morphTouched = false;
  auto& events = protocol->events();
  if(listChanged)
  {
    Json channels = Json::object();
    figure->published.clear();
    int order = 0;
    for(const fv::morph::Channel& ch : morphs.channels())
    {
      if(ch.hidden)
        continue;
      const double w = morphs.weightOf(ch.name);
      channels[ch.name] = {{"label", ch.label}, {"group", ch.group}, {"value", ch.value}, {"min", ch.min},
                           {"max", ch.max},     {"weight", w},       {"order", order++}};
      figure->published[ch.name] = {ch.value, w};
    }
    figure->morphList = morphs.listVersion();
    events.snapshot("morphs/" + figure->id, {{"figure", figure->id}, {"channels", channels}});
    return;
  }
  Json channels = Json::object();
  for(const fv::morph::Channel& ch : morphs.channels())
  {
    if(ch.hidden)
      continue;
    const double w  = morphs.weightOf(ch.name);
    const auto   it = figure->published.find(ch.name);
    if(it != figure->published.end() && it->second.first == ch.value && it->second.second == w)
      continue;
    channels[ch.name]          = {{"value", ch.value}, {"weight", w}};
    figure->published[ch.name] = {ch.value, w};
  }
  if(!channels.empty())
    events.patch("morphs/" + figure->id, {{"channels", channels}});
}

void FvHost::Impl::publishBusy()
{
  if(!figure)
    return;
  const std::string b = figure->fig->busy();
  if(!b.empty())
  {
    if(!figure->busyOn)
    {
      figure->busyJob = ++jobs;
      figure->busyOn  = true;
      job(figure->busyJob, {{"label", b}, {"state", "running"}});
    }
  }
  else if(figure->busyOn)
  {
    job(figure->busyJob, {{"state", "done"}});
    figure->busyOn = false;
  }
}

Json FvHost::Impl::poseSet(const Json& args)
{
  if(!figure || !figure->fig->pose || args.value("figure", figure->id) != figure->id)
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
  figure->fig->pose->setMany(values);
  morphTouched = true;
  // the values as the rig took them (clamped), for every panel showing the pose
  patchPose(bones);
  return {{"changed", figure->fig->pose->changed()}};
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

FvHost::FvHost(GltfRenderer* renderer, std::shared_ptr<FvProtocol> protocol, nvutils::ParameterRegistry* registry)
    : m_impl(std::make_unique<Impl>())
{
  Impl& s     = *m_impl;
  s.renderer  = renderer;
  s.protocol  = std::move(protocol);
  s.registry  = registry;
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
  s.materials = std::make_unique<FvMaterials>(FvMaterials::Hooks{
      .renderer = renderer,
      .events   = &s.protocol->events(),
      .post     = [impl](std::function<void()> f) { impl->post(std::move(f)); },
      .notice   = [impl](const std::string& text, const char* level) { impl->notice(text, level); },
      .nextJob  = [impl] { return ++impl->jobs; },
      .job      = [impl](uint64_t id, const Json& state) { impl->job(id, state); },
      .texture =
          [impl](const fv::build::TextureRequest& req, bool flipped) -> std::optional<fv::build::TextureImage> {
        HostSettings                       st;
        std::shared_ptr<fv::dson::Library> lib;
        {
          std::lock_guard lock(impl->settingsMutex);
          st  = impl->settings;
          lib = impl->libraryOfFigure;
        }
        if(!st.texServer.empty() && !st.rootId.empty())
          return cookedTexture(st, req, flipped, false).image;
        return lib ? fallbackTexture(st, *lib, req, false) : std::nullopt;
      },
  });
  cmds.add("load", [impl](const Json& args) { return impl->load(args); });
  cmds.add("material.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materials->set(args); }); });
  cmds.add("material.reset", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materials->reset(args); }); });
  cmds.add("pose.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->poseSet(args); }); });
  cmds.add("pose.eyes", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      if(!impl->figure || !impl->figure->fig->pose || args.value("figure", impl->figure->id) != impl->figure->id)
        throw std::runtime_error("no such figure");
      auto manip = impl->renderer->getCameraManipulator();
      if(!manip)
        throw std::runtime_error("no camera");
      const glm::dvec3 eye = manip->getEye();
      const int        n   = impl->figure->fig->pose->aimEyes({eye.x, eye.y, eye.z});
      std::vector<std::string> bones;
      for(const auto& pb : impl->figure->fig->pose->bones())
        if(fv::pose::isEyeBone(pb.name))
          bones.push_back(pb.id);
      impl->morphTouched = true;
      impl->patchPose(bones);
      return Json{{"aimed", n}};
    });
  });
  cmds.add("pose.reset", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      if(!impl->figure || !impl->figure->fig->pose || args.value("figure", impl->figure->id) != impl->figure->id)
        throw std::runtime_error("no figure loaded");
      impl->figure->fig->pose->reset();
      impl->morphTouched = true;
      impl->protocol->events().snapshot("pose/" + impl->figure->id, poseSnapshot(*impl->figure));
      return Json{{"changed", 0}};
    });
  });
  // answered at once: the values merge with whatever else arrived this frame and apply in onUpdate
  cmds.add("morph.set", [impl](const Json& args) {
    std::lock_guard lock(impl->morphMutex);
    const std::string fig = args.value("figure", impl->figureId);
    if(impl->figureId.empty() || fig != impl->figureId)
      throw std::runtime_error("no such figure");
    const Json* values = args.contains("values") ? &args["values"] : nullptr;
    if(!values || !values->is_object())
      throw std::runtime_error("morph.set wants values: {name: value}");
    for(auto it = values->begin(); it != values->end(); ++it)
      if(it.value().is_number())
        impl->pendingMorphs[it.key()] = it.value().get<double>();
    return Json{{"ok", true}};
  });
  cmds.add("morph.reset", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      if(!impl->figure || !impl->figure->fig->morphs || args.value("figure", impl->figure->id) != impl->figure->id)
        throw std::runtime_error("no figure loaded");
      {
        std::lock_guard lock(impl->morphMutex);
        impl->pendingMorphs.clear();
      }
      impl->figure->fig->morphs->resetAll();
      impl->morphTouched = true;
      return Json{{"ok", true}};
    });
  });
  cmds.add("render.mode", [impl](const Json& args) {
    const std::string mode = args.at("mode").get<std::string>();
    if(mode != "raster" && mode != "rtx")
      throw std::runtime_error("mode is raster or rtx");
    return impl->onApp([impl, mode] {
      impl->renderer->setRenderSystem(mode == "rtx" ? RenderingMode::ePathtracer : RenderingMode::eRasterizer);
      impl->skinMode = mode == "rtx" ? fv::mat::SkinMode::eVolume : fv::mat::SkinMode::eRaster;
      impl->materials->setSkinMode(impl->skinMode);
      impl->protocol->events().snapshot("render", {{"mode", mode}});
      return Json{{"mode", mode}};
    });
  });
  cmds.add("lighting.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->lightingSet(args); }); });
  // any registered renderer parameter, read or written as nvmcp's parameter tools do
  cmds.add("params.get", [impl](const Json& args) {
    return impl->onApp([impl, args] { return toolJson(nvmcp::detail::getParameters(exposedParameters(impl->registry), args.dump())); });
  });
  cmds.add("params.set", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      return toolJson(nvmcp::detail::setParameters(exposedParameters(impl->registry), args.dump()));
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
  s.view = std::make_unique<FvView>(renderer, s.protocol->events(), s.protocol->commands(), [impl](std::function<Json()> f) {
    return impl->onApp(std::move(f));
  });
  s.protocol->setInputSink([impl](const Json& events) {
    std::lock_guard lock(impl->inputMutex);
    for(const auto& e : events)
      impl->input.push_back(e);
  });
  // the gizmo poses the main figure's bones through the rig, so limits and formulas apply
  auto hasKind = [](const fv::pose::PoseBone& pb, fv::pose::ChannelKind kind) {
    return std::any_of(pb.axes.begin(), pb.axes.end(), [kind](const fv::pose::Axis& a) { return a.kind == kind; });
  };
  renderer->setGizmoHooks({
      .attach =
          [impl, hasKind](int gltf) {
            const auto* pb    = impl->boneAt(gltf);
            uint32_t    flags = 0;
            if(pb && hasKind(*pb, fv::pose::ChannelKind::eRotation))
              flags |= TransformHelperVk::ShowRotation;
            if(pb && hasKind(*pb, fv::pose::ChannelKind::eTranslation))
              flags |= TransformHelperVk::ShowTranslation;
            return flags;
          },
      .edit =
          [impl, hasKind](int gltf, const glm::vec3& t, const glm::quat& r, bool done) {
            // the release repeats the last move's transform; a click on a handle with no drag has only the release
            const auto* pb = done ? nullptr : impl->boneAt(gltf);
            if(!pb)
              return;
            impl->gizmoEdit = Impl::GizmoEdit{pb->id, fv::Quat{r.x, r.y, r.z, r.w},
                                              hasKind(*pb, fv::pose::ChannelKind::eTranslation) ?
                                                  std::optional<fv::Vec3>(fv::Vec3{t.x, t.y, t.z}) :
                                                  std::nullopt};
          },
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
  s.renderer->setGizmoHooks({});
  s.renderer->setPickObserver({});
  if(s.view)
    s.view->clearFigure();
  s.materials->detach();
  s.figure.reset();
  {
    std::lock_guard morphLock(s.morphMutex);
    s.figureId.clear();
    s.pendingMorphs.clear();
  }
}

void FvHost::onUpdate()
{
  Impl& s = *m_impl;
  // the default look until the app sends its own: the renderer's remembered settings don't apply here
  if(!s.lightingStarted)
  {
    s.lightingStarted = true;
    try
    {
      s.applyLighting(true);
    }
    catch(const std::exception& e)
    {
      LOGW("[fv] lighting: %s\n", e.what());
    }
  }
  std::deque<std::function<void()>> work;
  {
    std::lock_guard lock(s.queueMutex);
    work.swap(s.queue);
  }
  for(auto& f : work)
  {
    try
    {
      f();
    }
    catch(const std::exception& e)
    {
      s.notice(e.what(), "error");
    }
  }
  // input first: a gizmo drag poses this frame, and what the gizmo didn't take moves the camera
  std::vector<Json> events;
  {
    std::lock_guard lock(s.inputMutex);
    events.swap(s.input);
  }
  if(!events.empty())
    s.applyInput(s.view ? s.view->onInput(events) : events);
  s.applyGizmoEdit();
  s.applyPendingMorphs();
  if(s.figure)
  {
    const fv::Figure::Changes changes = s.figure->fig->pump();
    s.applyNodes(changes.nodes);
    for(const fv::morph::MeshChange& mesh : changes.meshes)
      s.applyMesh(mesh);
    s.applyRebind(changes.rebound);
    if(!changes.empty())
      s.morphTouched = true;
    for(const std::string& w : s.figure->fig->takeWarnings())
      s.notice(w);
    s.publishBusy();
    s.publishMorphs();
  }
  if(s.frameIn > 0 && --s.frameIn == 0)
    s.frameCamera();
  if(s.view)
    s.view->onFrame();
}
