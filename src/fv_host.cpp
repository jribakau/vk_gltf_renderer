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
#include <iterator>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <source_location>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvmcp/parameter_tools.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvutils/timers.hpp>

#include <figure/dforce/dforce.hpp>
#include <figure/wardrobe/collider.hpp>
#ifdef FV_DFORCE_GPU
#include "fv_dforce_gpu.hpp"
#endif
#include <figure/dson/eye_materials.hpp>
#include <figure/dson/skin_materials.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/dson/presets.hpp>
#include <figure/figure.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/gltf/export_glb.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/io/timing.hpp>

#include "env_image_loader.hpp"
#include "fv_compose.hpp"
#include "fv_lighting.hpp"
#include "fv_materials.hpp"
#include "fv_protocol.hpp"
#include "fv_textures.hpp"
#include "fv_view.hpp"
#include "fv_workers.hpp"
#include "gltf_environment_sky.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include "gizmo_lines_vk.hpp"
#include "gltf_scene_editor.hpp"
#include "renderer.hpp"
#include "tinygltf_utils.hpp"
#include "ui_animation.hpp"

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
  bool originalTextures = false;  // the source images, not the BC7 cache: only what the renderer cannot read is cooked
};

// A loaded figure and what the renderer made of it. Lives on the application thread.
struct Loaded
{
  std::string                 id;  // its prop's
  std::unique_ptr<fv::Figure> fig;
  // drapes the worn dynamic surfaces (it works on `fig`, so it goes first); null without a body
  std::unique_ptr<fv::dforce::Dforce> dforce;
  int                                 dforceVersion = -1;
  // keeps the worn conforming meshes out of the posed body (figure/wardrobe/collider.hpp); null without a body
  std::unique_ptr<fv::wardrobe::Collider> collider;
  int                                     collisionVersion = -1;
  // where it is in the renderer's scene, set when the scene is composed (fv_compose.hpp)
  std::vector<int>              nodeOf;   // scene node -> glTF node
  std::unordered_map<int, int>  sceneOf;  // glTF node -> scene node
  std::vector<int>              meshOf;   // scene mesh -> glTF mesh
  std::vector<int>              skinOf;   // scene mesh -> glTF skin
  std::vector<std::vector<int>> primsOf;  // scene mesh -> the render primitives owning its vertex sets (one per pose)
  // the glTF its load built on the worker, which the next composition takes instead of building it again
  std::optional<fv::build::BuildResult> prebuilt;
  // morph topic: -1 until the channel list has been published, then the value and weight last sent
  int                                                          morphList = -1;
  std::unordered_map<std::string, std::pair<double, double>> published;
  int                                                          morphsSeen = -1;  // MorphSet::version() last published
  uint64_t                                                     busyJob = 0;
  bool                                                         busyOn  = false;
  // the textures the glTF was built with, by request key, and whether their rows run bottom-up:
  // a wardrobe change builds the glTF again with these and the new items' own
  std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
  bool                                                                    texturesFlipped = false;
  // its DAZ library, for source textures and what it wears (the settings mutex)
  std::shared_ptr<fv::dson::Library> lib;
  // how it was opened ({rel} or {path}), for a scene file
  Json source;
  // material presets opened onto it, library paths in order, for a scene file
  std::vector<std::string> materialPresets;
  // the eyes chosen (material.eyes, or an eye-only material preset): its library path, "" for the
  // character's own; set only once chosen, so a scene file without a choice replays its presets alone
  std::optional<std::string> eyes;
  std::set<int>              eyesOwned;   // what those eyes changed
  std::string                eyesPreset;  // a mixed material preset that set the eye surfaces since
  // worn item id -> the colour preset chosen for it (wardrobe.color): its library path, "" for the
  // item as it was worn, and what that changed, for the next choice
  struct ItemColor
  {
    std::string   rel;
    std::set<int> owned;
  };
  std::map<int, ItemColor> colors;
  // the skin chosen (material.skin), as `eyes`, with what it changed and a mixed material preset
  // that set skin surfaces since
  std::optional<std::string> skin;
  std::set<int>              skinOwned;
  std::string                skinPreset;
  // the makeup chosen (material.makeup), "" for the skin's own, and the channels it set, which the
  // next choice takes off; a skin choice takes it off with the rest
  std::optional<std::string> makeup;
  fv::dson::PresetLayer      makeupLayer;
  // the add-ons its file lists are still going on; a restore waits for them, as it would otherwise
  // wear them a second time without their presets
  bool                               addonsPending = false;
  std::vector<std::function<void()>> afterAddons;
};

enum class PropKind
{
  eFigure,
  eModel,
  eCamera,
  eLight
};

const char* kindName(PropKind k)
{
  switch(k)
  {
    case PropKind::eFigure:
      return "figure";
    case PropKind::eModel:
      return "model";
    case PropKind::eCamera:
      return "camera";
    case PropKind::eLight:
      return "light";
  }
  return "model";
}

// A camera prop: a lens of its own, and how far ahead its orbit target is.
struct CameraProp
{
  FvView::Lens lens{.focalLength = 50};
  bool         locked         = false;
  double       targetDistance = 3;
};

// A light prop (KHR_lights_punctual): colour in sRGB as the colour pickers hold it, intensity in
// candela (point, spot) or lux (directional), the radius its shadows soften with, a spot's cones.
struct LightProp
{
  std::string type      = "point";
  std::string color     = "#ffffff";
  double      intensity = 20;
  double      radius    = 0.05;
  double      inner     = 0;
  double      outer     = 0.785398;
};

// One thing in the scene. Its prop root (fv_compose.hpp) carries `transform` and `visible`; under
// it is a figure, a model read from a file, a camera or a light.
struct Prop
{
  std::string             id, name;
  PropKind                kind = PropKind::eModel;
  fv::composer::Trs        transform;
  std::optional<glm::vec3> euler;  // the rotation as the panel typed it, degrees XYZ; a gizmo turn drops it
  bool                    visible = true;
  std::string             parent;  // a prop id, "" at the scene's root
  std::shared_ptr<Loaded> figure;
  // a model: as its file holds it (shared by duplicates) and where it came from
  std::shared_ptr<const tinygltf::Model> model;
  std::string                            path;
  std::string                            format;
  CameraProp                             camera;
  LightProp                              light;
  std::unique_ptr<FvMaterials>           materials;  // a figure's or a model's
  fv::composer::Placed                    placed;     // where the last composition put it
  Json                                   stats = Json::object();
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

// normRel, compared as the library compares paths: without case
std::string relKey(const std::string& path)
{
  std::string key = normRel(path);
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return key;
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

// Texture server requests in flight. A cook can take minutes (the read timeout is 600 s), so a
// stopping renderer cuts them off instead of joining workers that wait on one; a request that starts
// afterwards is not sent.
class TextureCalls
{
public:
  bool add(httplib::Client* cli)
  {
    std::lock_guard lock(m_mutex);
    if(m_stopped)
      return false;
    m_live.insert(cli);
    return true;
  }
  void remove(httplib::Client* cli)
  {
    std::lock_guard lock(m_mutex);
    m_live.erase(cli);
  }
  bool stopped()
  {
    std::lock_guard lock(m_mutex);
    return m_stopped;
  }
  // Client::stop is the one call that is safe while another thread is inside a request: it shuts
  // the socket down and the request fails at once.
  void stopAll()
  {
    std::lock_guard lock(m_mutex);
    m_stopped = true;
    for(httplib::Client* cli : m_live)
      cli->stop();
  }

private:
  std::mutex                 m_mutex;
  std::set<httplib::Client*> m_live;
  bool                       m_stopped = false;
};

TextureCalls& textureCalls()
{
  static TextureCalls calls;
  return calls;
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
  if(req.bumpHeight > 0)
    url += "&bh=" + fmtNumber(req.bumpHeight);
  if(req.makeup)
  {
    // mk=<base 1|0>|<colour source>|<weight source>|<weight>|<r,g,b tint>|<r,g,b factor>, percent-encoded as a whole
    const fv::mat::MakeupRequest& m = *req.makeup;
    const auto                    rgb = [](const std::array<float, 3>& c) { return fmtNumber(c[0]) + "," + fmtNumber(c[1]) + "," + fmtNumber(c[2]); };
    const std::string             mk = std::string(m.baseIsSource ? "1" : "0") + "|" + (m.colorRel.empty() ? "-" : source(m.colorRel)) + "|"
                           + (m.weightRel.empty() ? "-" : source(m.weightRel)) + "|" + fmtNumber(m.weightFactor) + "|" + rgb(m.tint) + "|" + rgb(m.factor);
    url += "&mk=" + encodeComponent(mk);
  }
  // ir=<iris r,g,b | ->|<strength>|<eye | iris>|<x,y,r;...>|<sclera r,g,b | ->, percent-encoded as a whole
  if(req.iris)
    url += "&ir=" + encodeComponent(fv::mat::irisQuery(*req.iris));
  // rc=<r,g,b>|<strength>[|<reference source>], percent-encoded as a whole
  if(req.recolor)
    url += "&rc=" + encodeComponent(fv::mat::recolorQuery(*req.recolor) + (req.recolor->reference.empty() ? "" : "|" + source(req.recolor->reference)));
  // ly=<source>;<source>..., percent-encoded as a whole
  if(!req.layers.empty())
  {
    std::string ly;
    for(const std::string& layer : req.layers)
      ly += (ly.empty() ? "" : ";") + source(layer);
    url += "&ly=" + encodeComponent(ly);
  }
  url += cachedOnly ? "&path=1&cached=1" : "&path=1";
  if(!textureCalls().add(&cli))
    return {};
  auto res = cli.Get(url);
  textureCalls().remove(&cli);
  if(res && res->status == 202 && cachedOnly)
    return {.uncooked = true};
  if(!res || res->status != 200)
  {
    if(!quiet && !textureCalls().stopped())
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
// bump blended onto it. A roughness map (its grey into glTF's channels) and a height map (turned
// into a normal map) are composed here: without them skin loses its sheen and its pores. A cutout
// alone changes the picture too much: it, makeup and an iris colour (and formats the renderer cannot
// read) are cooked, unflipped, and waited for; without a cooker, or with Original textures (only
// what the renderer cannot read is cooked then), they are composed here too (fv_textures.hpp).
std::optional<fv::build::TextureImage> fallbackTexture(const HostSettings& s, const fv::dson::Library& lib, const fv::build::TextureRequest& req, bool cooker)
{
  using fv::textures::Fallback;
  const bool     derived = req.channels == "mr" || req.channels == "bn";
  const bool     laid    = !req.alphaRel.empty() || req.makeup || req.iris || req.recolor || !req.layers.empty();
  // an iris or a recolour only the cooker makes (composeImage has neither): cooked under Original textures too
  const Fallback route =
      cooker && (req.iris || req.recolor) ? Fallback::eCook : fv::textures::fallbackOf(req.rel, derived, laid, cooker, s.originalTextures);
  if(route == Fallback::eCook)
    return cookedTexture(s, req, false, false).image;
  // composed into a PNG beside the others made this way (composeImage recolours the iris)
  if(route == Fallback::eCompose)
  {
    std::error_code ec;
    const auto      dir = std::filesystem::temp_directory_path(ec) / "fv-composed";
    // named by the request and by where its sources are, their size and their time (FNV-1a, the
    // same from build to build): a texture edited in place, or the same relative path in another
    // library, composes again instead of showing the old picture
    uint64_t   h   = 1469598103934665603ULL;
    const auto mix = [&h](const std::string& s) {
      for(unsigned char c : s)
      {
        h ^= c;
        h *= 1099511628211ULL;
      }
      h ^= '|';
      h *= 1099511628211ULL;
    };
    mix(fv::build::requestKey(req));
    {
      std::vector<std::string> sources = {req.rel, req.alphaRel, req.bumpRel};
      sources.insert(sources.end(), req.layers.begin(), req.layers.end());
      if(req.makeup)
        sources.insert(sources.end(), {req.makeup->colorRel, req.makeup->weightRel});
      for(const std::string& rel : sources)
        if(!rel.empty())
          if(auto src = lib.find(normRel(rel)))
          {
            std::error_code e2;
            mix(utf8Of(*src));
            mix(std::to_string(std::filesystem::file_size(*src, e2)));
            mix(std::to_string(std::filesystem::last_write_time(*src, e2).time_since_epoch().count()));
          }
    }
    char name[32];
    std::snprintf(name, sizeof name, "%016llx", static_cast<unsigned long long>(h));
    const auto file = dir / (std::string(name) + ".png");
    if(!std::filesystem::exists(file, ec))
    {
      fv::build::ExportOptions o;
      o.find     = [&](const std::string& rel) { return lib.find(normRel(rel)); };
      o.bottomUp = false;
      std::string why;
      const auto  png = fv::build::composeImage(req, o, &why);
      if(!png)
      {
        LOGW("texture %s: %s\n", req.rel.c_str(), why.c_str());
        return derived ? std::nullopt : sourceTexture(lib, req);
      }
      std::filesystem::create_directories(dir, ec);
      const auto    tmp = file.string() + "." + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
      std::ofstream out(std::filesystem::path(tmp), std::ios::binary);
      out.write(reinterpret_cast<const char*>(png->data()), std::streamsize(png->size()));
      out.close();
      if(!out)
      {
        LOGW("texture %s: could not write %s\n", req.rel.c_str(), tmp.c_str());
        std::filesystem::remove(tmp, ec);
        return derived ? std::nullopt : sourceTexture(lib, req);
      }
      std::filesystem::rename(tmp, file, ec);
      if(ec)
      {
        LOGW("texture %s: could not place %s: %s\n", req.rel.c_str(), utf8Of(file).c_str(), ec.message().c_str());
        std::filesystem::remove(tmp, ec);
        return derived ? std::nullopt : sourceTexture(lib, req);
      }
    }
    return fv::build::TextureImage{uriOfPath(utf8Of(file)), "image/png", false};
  }
  if(route == Fallback::eNone)
    return std::nullopt;
  auto image = sourceTexture(lib, req);
  if(image && req.channels == "ng")
    image->flipGreen = true;
  return image;
}

// Genesis 9's brows ship a grey placeholder surface; DAZ's post-load script (Remove Duplicate
// Eyebrows.dse, encrypted) lays the default brow material over it, which a character's own preset
// may then change. The preset for the brows a wearable holds, or none.
std::string defaultBrowMaterial(const fv::SceneGraph& scene)
{
  static const std::string dir = "data/DAZ 3D/Genesis 9/Base/Tools/Script Loads/Eyebrows/";
  for(const fv::Node& n : scene.nodes)
  {
    if(n.dsonNode.rfind("G9EyebrowCards", 0) == 0)
      return dir + "Card Eyebrows MAT.duf";
    if(n.dsonNode.rfind("G9EyebrowFibers", 0) == 0)
      return dir + "Fiber Eyebrows MAT.duf";
  }
  return {};
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

// A drag's cost on the application thread, summed over a second and logged once (`[fv-time]`): a
// dial's morph apply (the formula graph, the deltas and the re-welded normals; the phases inside are
// fv::timing::Accum lines), a pose step, the node transforms, the vertex uploads it queues for the
// renderer (and how many vertices, how many of them whole meshes) and the rebinds. Phase 6 of
// docs/pipeline-roadmap.md reads these.
struct DialTiming
{
  enum Kind
  {
    eSetMany,
    eApplyMesh,
    ePoseSet,
    eApplyNodes,
    eApplyRebind,
    eCount
  };
  struct Sum
  {
    int    n     = 0;
    double ms    = 0;
    double maxMs = 0;
  };
  std::array<Sum, eCount>               sums{};
  std::chrono::steady_clock::time_point since{};
  size_t                                vertices    = 0;  // uploaded this second, meshes summed
  int                                   uploads     = 0;
  int                                   fullUploads = 0;  // of them whole meshes (`all`)

  void add(Kind k, double ms)
  {
    if(since == std::chrono::steady_clock::time_point{})
      since = std::chrono::steady_clock::now();
    Sum& s = sums[size_t(k)];
    s.n++;
    s.ms += ms;
    s.maxMs = std::max(s.maxMs, ms);
  }

  void addUpload(size_t vertexCount, bool all)
  {
    if(since == std::chrono::steady_clock::time_point{})
      since = std::chrono::steady_clock::now();
    vertices += vertexCount;
    uploads++;
    fullUploads += all ? 1 : 0;
  }

  // The line to log once a second has passed since the first sample, else empty; clears the sums.
  std::string report(std::chrono::steady_clock::time_point now)
  {
    if(since == std::chrono::steady_clock::time_point{} || now - since < std::chrono::seconds(1))
      return {};
    const auto mean = [](const Sum& s) { return s.n ? s.ms / s.n : 0.0; };
    const Sum& a    = sums[eSetMany];
    const Sum& b    = sums[eApplyMesh];
    const Sum& p    = sums[ePoseSet];
    const Sum& nd   = sums[eApplyNodes];
    const Sum& rb   = sums[eApplyRebind];
    char       line[512];
    snprintf(line, sizeof line,
             "[fv-time] dial: %d steps, setMany %.1f ms mean, %.1f max; %d pose steps, poseSet %.1f ms mean, %.1f max; "
             "%d frames of mesh changes, applyMesh %.1f ms mean, %.1f max (%d uploads, %d whole meshes, %zu vertices); "
             "applyNodes %.1f ms mean over %d, applyRebind %.1f ms mean over %d",
             a.n, mean(a), a.maxMs, p.n, mean(p), p.maxMs, b.n, mean(b), b.maxMs, uploads, fullUploads, vertices, mean(nd), nd.n, mean(rb), rb.n);
    *this = DialTiming{};
    return line;
  }
};

}  // namespace

std::string figureSource(const Loaded& figure);
std::string figureGeneration(const Loaded& figure);

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
  uint64_t          lastInputSeq = 0;  // the newest batch applied (normalizeInput)
  double            mouseX = 0, mouseY = 0;
  bool              lmb = false, mmb = false;

  // application thread only: the scene's props in order; prop root i of the renderer's scene is props[i]'s
  std::vector<std::shared_ptr<Prop>> props;
  int                                propCount = 0;
  // morph.set merges here (any thread); onUpdate applies it as one setMany per figure
  std::mutex                                           morphMutex;
  std::set<std::string>                                morphFigures;  // the figures morph.set may name
  std::map<std::string, std::map<std::string, double>> pendingMorphs;
  // frames until the camera frames the scene (0: none pending)
  int frameIn = 0;
  // the gizmo's latest drag on a bone: the rig takes it once a frame, however many events came
  struct GizmoEdit
  {
    std::shared_ptr<Loaded> figure;
    std::string             bone;
    fv::Quat                rotation;
    std::optional<fv::Vec3> position;
  };
  std::optional<GizmoEdit> gizmoEdit;
  // props the gizmo or prop.set moved this frame, published once in onUpdate
  std::set<std::string> movedProps;
  // the camera prop the viewport looks through ("" for the free view), and where the free view was
  // left meanwhile, with its lens
  std::string viewCamera;
  struct FreeView
  {
    glm::dvec3   eye{0, 1, 3}, center{0, 1, 0}, up{0, 1, 0};
    FvView::Lens lens;
  };
  FreeView freeView;
  bool     lockedNoticed = false;
  // where the camera looked through stood in the world when the view last matched it: when it is
  // somewhere else now (its parent moved, the gizmo, prop.set), the view follows it
  glm::mat4 viewWorld{1.f};

  // A change that needs the scene composed again (fv_compose.hpp): collected over a frame and run
  // once in onUpdate. `after` runs once it is composed (or failed to be).
  struct ComposeRequest
  {
    std::vector<std::string>           labels;
    std::vector<std::function<void()>> after;
    std::set<Loaded*>                  settle;  // figures whose cloth settles again: what they wear changed
  };
  std::optional<ComposeRequest> pendingCompose;
  bool                          composedAny = false;  // the last composition had a figure or model in it

  // loads, wears, exports and environment decodes, each on a thread of its own
  FvWorkers loaders;
  // bumped by scene.clear and on shutdown: a load still running drops its result
  std::atomic<uint64_t>    loadGeneration{0};
  std::atomic<uint64_t>    jobs{0};
  std::atomic<uint64_t>    notices{0};

  // Runs `f` on the application thread and waits for it (from an HTTP thread). A task that has not
  // started within 30 s is dropped and the caller told so; one that is running is waited for, however
  // long, so no command is reported failed and then applied (a retry would apply it twice).
  template <class F>
  auto onApp(F f, std::source_location where = std::source_location::current()) -> decltype(f())
  {
    using R     = decltype(f());
    auto task   = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto state  = std::make_shared<std::atomic<int>>(0);  // 0 queued, 1 running, 2 dropped
    auto result = task->get_future();
    {
      std::lock_guard lock(queueMutex);
      if(!accepting)
        throw std::runtime_error("the renderer is stopping");
      queue.emplace_back([task, state, line = where.line()] {
        int queued = 0;
        if(!state->compare_exchange_strong(queued, 1))
          return;
        // one that holds the application thread up shows in the timing log, by the line that sent it
        const auto start = std::chrono::steady_clock::now();
        (*task)();
        if(fv::timing::enabled())
          fv::timing::report("command run, sent at fv_host.cpp:" + std::to_string(line),
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), 50);
      });
    }
    if(result.wait_for(30s) != std::future_status::ready)
    {
      int queued = 0;
      if(state->compare_exchange_strong(queued, 2))
        throw std::runtime_error("timed out waiting for the renderer; the command was dropped");
      result.wait();  // started meanwhile: its result is the truth
    }
    return result.get();
  }

  // No more work for the application thread: what is queued is dropped, and every HTTP thread
  // waiting on it (onApp) wakes with an error instead of after its timeout. Called when the stream
  // stops (before its server joins the threads those requests hold) and on detach.
  void stopAccepting()
  {
    std::deque<std::function<void()>> dropped;
    {
      std::lock_guard lock(queueMutex);
      accepting = false;
      dropped.swap(queue);
    }
    dropped.clear();
  }

  std::string uniqueName(const std::string& name, const Prop* self) const;
  bool        shown(const Prop& p) const;
  bool        ridesOn(const Prop& q, const std::string& id) const;
  void        applyLightShown(const Prop& p);

  void post(std::function<void()> f, std::source_location where = std::source_location::current())
  {
    // one that holds the application thread up shows in the timing log, by the line that posted it
    auto timed = [f = std::move(f), line = where.line()] {
      const auto start = std::chrono::steady_clock::now();
      f();
      if(fv::timing::enabled())
        fv::timing::report("posted at fv_host.cpp:" + std::to_string(line),
                           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), 50);
    };
    std::lock_guard lock(queueMutex);
    if(accepting)
      queue.push_back(std::move(timed));
  }

  // The jobs and notices topics keep the latest few finished jobs and notices, not every one of the
  // session: each patch merges the whole topic on both ends, and a reconnect resends it. A finished
  // job stays a while, so a client that waits on it after the answer still sees how it ended.
  static constexpr size_t kKeptJobs    = 32;
  static constexpr size_t kKeptNotices = 20;
  std::mutex              keptMutex;
  std::deque<uint64_t>    finishedJobs;
  std::deque<uint64_t>    shownNotices;

  void notice(const std::string& text, const char* level = "warning")
  {
    const uint64_t id    = ++notices;
    Json           patch = {{std::to_string(id), {{"text", text}, {"level", level}}}};
    {
      std::lock_guard lock(keptMutex);
      shownNotices.push_back(id);
      while(shownNotices.size() > kKeptNotices)
      {
        patch[std::to_string(shownNotices.front())] = nullptr;
        shownNotices.pop_front();
      }
    }
    protocol->events().patch("notices", patch);
  }

  void job(uint64_t id, const Json& state)
  {
    Json patch = {{std::to_string(id), state}};
    if(state.contains("state") && state["state"] != "running")
    {
      std::lock_guard lock(keptMutex);
      if(std::find(finishedJobs.begin(), finishedJobs.end(), id) == finishedJobs.end())
        finishedJobs.push_back(id);
      while(finishedJobs.size() > kKeptJobs)
      {
        patch[std::to_string(finishedJobs.front())] = nullptr;
        finishedJobs.pop_front();
      }
    }
    protocol->events().patch("jobs", patch);
  }

  // ---- props (application thread)
  std::shared_ptr<Prop> findProp(const std::string& id) const;
  // The prop a command names ({prop}, or {figure} as panels used to send it), else the selected one.
  std::shared_ptr<Prop> propArg(const Json& args) const;
  // The figure a command is for: the one it names, else the selected prop when that is a figure,
  // else the only figure in the scene. Null when there is none of those.
  std::shared_ptr<Prop> figureProp(const Json& args) const;
  std::shared_ptr<Loaded> figureArg(const Json& args) const;
  bool                    inScene(const std::shared_ptr<Loaded>& f) const;
  std::shared_ptr<Prop>   addProp(PropKind kind, const std::string& name);
  void                    removeProp(const std::shared_ptr<Prop>& p);
  std::unique_ptr<FvMaterials> makeMaterials(const std::shared_ptr<Prop>& p);
  glm::mat4               worldOf(const Prop& p) const;
  void                    applyTransform(const Prop& p);
  void                    applyVisible(const Prop& p);
  Json                    propJson(const Prop& p) const;
  void                    publishScene();
  // camera frustums and light markers, drawn over the image (GltfRenderer::setHelperLines), and the
  // one at a pixel
  void        drawHelpers();
  std::string helperAt(const glm::vec2& px) const;
  void                    selectProp(const std::string& id);
  void                    requestCompose(const std::string& label, std::function<void()> after = {}, Loaded* settle = nullptr);
  void                    compose();
  Json                    propAdd(const Json& args);
  Json                    propSet(const Json& args);
  Json                    propDuplicate(const Json& args);
  Json                    sceneClear();
  // the view: looking through a camera prop, or the free view
  Json viewUse(const Json& args);
  void viewFromCamera(const Prop& p);
  void cameraFromView();
  void publishView();

  Json load(const Json& args);
  // A figure or prop from a DAZ library, as a new prop; `then` runs once it is in the scene.
  // `prefetch`: library files the figure will wear (a scene file's worn list), read on the pool from
  // the start of the load so its restore finds them in the figure's store
  Json loadFigure(const HostSettings& s, std::vector<std::filesystem::path> roots, std::string rel, const std::string& name, Json source,
                  std::string target, bool asProp, std::function<void(std::shared_ptr<Prop>)> then = {}, std::vector<std::string> prefetch = {});
  void install(std::unique_ptr<fv::Figure> f, fv::build::BuildResult built, uint64_t generation, uint64_t jobId, bool texturesFlipped,
               fv::mat::SkinMode skin, std::shared_ptr<fv::dson::Library> lib,
               std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images, Json source,
               std::function<void(std::shared_ptr<Prop>)> then);
  // wardrobe: what a figure wears (fv_host.hpp commands wear, wardrobe.remove, wardrobe.visible)
  struct WearRequest
  {
    std::string              rel;
    std::string              name;     // the DSON node name an add-on has, so it is not put on twice
    std::vector<std::string> presets;  // material presets applied to it first
  };
  Json     wear(const Json& args);
  // `beforeBuild` runs on the application thread once everything is on, before the scene is built
  // again; `after` once it has been. With `commitAfter`, the requests are prepared at once and put
  // on once it is ready (a restore's worn items after the figure's add-ons).
  uint64_t startWear(std::shared_ptr<Loaded> target, std::vector<WearRequest> requests, std::function<void()> beforeBuild = {}, std::function<void()> after = {},
                     std::shared_future<void> commitAfter = {});
  // glTF, GLB and OBJ files, read on a worker, as a new prop
  Json importModel(const std::string& path, std::function<void(std::shared_ptr<Prop>)> then = {});
  Json animationTopic();
  void publishAnimation(bool force);
  Json animationCommand(const std::string& cmd, const Json& args);
  Json lastAnimation;
  std::chrono::steady_clock::time_point animationAt{};
  Json exportGlb(const Json& args);
  Json sceneState(const Json& args);
  Json sceneRestore(const Json& args);
  std::mutex wearMutex;
  Json wardrobeRemove(const Json& args);
  Json wardrobeVisible(const Json& args);
  Json wardrobeCollision(const Json& args);
  Json wardrobeCloth(const Json& args);
  void showItem(const Loaded& f, const std::vector<std::pair<int, std::string>>& objects, bool visible);
  void publishDforce(Loaded& f, bool force);
  Json dforceCommand(const std::string& cmd, const Json& args);
  void publishFigure(Prop& p);
  void publishWardrobe(Loaded& f);
  std::optional<std::array<glm::dvec3, 3>> cameraRestore;
  Json poseSet(const Json& args);
  // the images a load or a wear resolved, decoded on this worker's threads so the reload takes them ready
  void prefetchImages(const std::vector<fv::build::TextureRequest>& requests, const std::vector<std::optional<fv::build::TextureImage>>& found);
  void applyNodes(Loaded& f, const std::set<int>& moved);
  void applyMesh(Loaded& f, const fv::morph::MeshChange& change);
  void applyRebind(Loaded& f, const std::vector<int>& meshes);
  void applyPendingMorphs();
  void publishMorphs(Loaded& f);
  void publishBusy(Loaded& f);
  void frameCamera();
  void normalizeInput(std::vector<Json>& events);
  void applyInput(const std::vector<Json>& events);
  void patchPose(const Loaded& f, const std::vector<std::string>& boneIds);
  // a figure's posed bone at a glTF node, or {nullptr, nullptr}
  std::pair<std::shared_ptr<Loaded>, const fv::pose::PoseBone*> boneAt(int gltf) const;
  void                      applyGizmoEdit();
  void applyPreset(const fv::dson::Json&                             doc,
                   const std::string&                                kind,
                   uint64_t                                          jobId,
                   const std::string&                                rel,
                   const std::optional<fv::dson::EnvironmentPreset>& environment,
                   const std::string&                                environmentPath,
                   const std::string&                                target);
  // eyes (fv_host.hpp material.eyes): `rel` "" and no `doc` for the character's own
  Json materialEyes(const Json& args);
  Json applyEyes(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits);
  // a material preset that is not a pair of eyes changed `changed`: if that reached the eyes, they are its now
  void presetChangedEyes(Prop& prop, const std::string& rel, const std::set<int>& changed);
  // a worn item's colour (fv_host.hpp wardrobe.color): `rel` "" and no `doc` for the item as it was worn
  Json wardrobeColor(const Json& args);
  Json applyItemColor(Prop& prop, const std::vector<int>& items, const std::string& rel, const fv::dson::Json* doc, bool keepEdits);
  void publishEyes(Prop& p);
  // skin and makeup (fv_host.hpp material.skin, material.makeup): `rel` "" and no `doc` for the
  // character's own
  Json materialSkin(const Json& args);
  Json materialMakeup(const Json& args);
  Json applySkin(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits);
  Json applyMakeup(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits);
  void publishSkin(Prop& p);
  std::set<int> wornSurfaces(const Loaded& f);
  // a preset chosen for part of a figure, read on a worker and applied by `apply` once the add-ons are on
  using ChoiceApply = std::function<Json(Prop&, const std::string& rel, const fv::dson::Json* doc)>;
  Json choiceJob(const Json& args, const std::string& command, const std::string& part, ChoiceApply apply);

  // ---- lighting (fv_lighting.hpp), application thread only
  nvutils::ParameterRegistry* registry = nullptr;
  fv::lighting::State         lighting;
  bool                        lightingStarted = false;
  Json                        appliedParams   = Json::object();  // what the renderer was last given
  DialTiming                  dialTiming;
  // when the last frame's update ended: the renderer's own work in between shows when it is long
  std::optional<std::chrono::steady_clock::time_point> lastUpdateEnd;
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
  // the loaded HDR's sun, toward it in the image's frame (fv::lighting::environmentSun); none for
  // the studio and for an image without one
  std::optional<glm::vec3> envSun;
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
  void fillSun(tinygltf::Light& light, tinygltf::Node& node, int lightIndex);
  void setParams(const Json& values);
  void publishLighting();

  std::unique_ptr<FvView> view;
  // how skin is shaded for the render mode
  std::atomic<fv::mat::SkinMode> skinMode{fv::mat::SkinMode::eRaster};
  // a texture a material bake needs, from the cooker or a library (worker threads)
  std::optional<fv::build::TextureImage> bakeTexture(const std::shared_ptr<fv::dson::Library>& lib, const fv::build::TextureRequest& req, bool flipped);
};

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

std::shared_ptr<Prop> FvHost::Impl::addProp(PropKind kind, const std::string& name)
{
  auto p  = std::make_shared<Prop>();
  p->id   = "p:" + std::to_string(++propCount);
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
      .texture =
          [this, weak](const fv::build::TextureRequest& req, bool flipped) -> std::optional<fv::build::TextureImage> {
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
      glm::vec3 skew;
      glm::vec4 perspective;
      glm::decompose(world, q->transform.scale, q->transform.rotation, q->transform.translation, skew, perspective);
      q->euler.reset();
    }
  props.erase(std::remove(props.begin(), props.end(), p), props.end());
  // hidden now; gone from the GPU when the composition runs
  if(p->placed.root >= 0)
  {
    Prop hidden    = {};
    hidden.placed  = p->placed;
    hidden.visible = false;
    applyVisible(hidden);
  }
}

glm::mat4 FvHost::Impl::worldOf(const Prop& p) const
{
  glm::mat4   m(1.f);
  const Prop* at = &p;
  for(int steps = 0; at && steps <= int(props.size()); steps++)
  {
    const glm::mat4 local = glm::translate(glm::mat4(1.f), at->transform.translation) * glm::mat4_cast(at->transform.rotation)
                            * glm::scale(glm::mat4(1.f), at->transform.scale);
    m = local * m;
    at = at->parent.empty() ? nullptr : findProp(at->parent).get();
  }
  return m;
}

void FvHost::Impl::applyTransform(const Prop& p)
{
  if(p.placed.root < 0)
    return;
  const GltfRenderer::NodeTransform t{p.placed.root, p.transform.translation, p.transform.rotation, p.transform.scale};
  renderer->setNodeTransforms({&t, 1});
}

void FvHost::Impl::applyVisible(const Prop& p)
{
  nvvkgltf::Scene* scene = renderer->editScene();
  if(!scene || !scene->valid() || p.placed.root < 0 || p.placed.root >= int(scene->getModel().nodes.size()))
    return;
  tinygltf::utils::setNodeVisibility(scene->getModel().nodes[size_t(p.placed.root)], {p.visible});
  // from the prop at the top of what it rides on: a prop under a hidden one stays hidden, whatever its own flag
  const Prop* top = &p;
  for(int steps = 0; !top->parent.empty() && steps <= int(props.size()); steps++)
  {
    const auto up = findProp(top->parent);
    if(!up || up->placed.root < 0)
      break;
    top = up.get();
  }
  scene->editor().updateVisibility(top->placed.root);
  // a light shows nothing, but lights; hidden, or riding on something hidden, it goes dark
  applyLightShown(p);
  for(const auto& q : props)
    if(q.get() != &p && ridesOn(*q, p.id))
      applyLightShown(*q);
}

void FvHost::Impl::applyLightShown(const Prop& p)
{
  nvvkgltf::Scene* scene = renderer->editScene();
  if(!scene || !scene->valid() || p.placed.light < 0 || p.placed.light >= int(scene->getModel().lights.size()))
    return;
  scene->editor().getLightForEdit(p.placed.light).intensity = shown(p) ? p.light.intensity : 0.0;
  scene->markLightDirty(p.placed.light);
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
void FvHost::Impl::cameraFromView()
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
  const glm::mat4 view   = glm::lookAt(glm::vec3(eye), glm::vec3(center), glm::vec3(up));
  const auto      parent = p->parent.empty() ? nullptr : findProp(p->parent);
  const glm::mat4 local  = (parent ? glm::inverse(worldOf(*parent)) : glm::mat4(1.f)) * glm::inverse(view);
  glm::vec3       scale, skew;
  glm::vec4       perspective;
  glm::decompose(local, scale, p->transform.rotation, p->transform.translation, skew, perspective);
  p->transform.scale       = glm::vec3(1.f);
  p->euler.reset();
  p->camera.targetDistance = std::max(distance, 1e-3);
  applyTransform(*p);
  viewWorld = worldOf(*p);
  movedProps.insert(p->id);
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
  if(p && p->placed.root >= 0)
    renderer->selectSceneNodeFromScript(p->placed.root);
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

// The scene built again from its props and handed to the renderer in place of the old one: every
// figure's glTF from its scene graph (its pose, dials, worn items and edits are in the graph, so it
// comes back as it was), each on a thread of its own; every model as its file had it, with its
// material edits laid over again. The camera stays where it was, unless the scene was empty.
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

  // figures built again, in parallel: each fv::Figure on one thread, and this one waits
  std::vector<std::optional<fv::build::BuildResult>> builds(props.size());
  {
    nvutils::ScopedTimer             t("buildGltf");
    std::vector<std::future<void>>   work;
    const fv::mat::SkinMode          skin = skinMode.load();
    for(size_t i = 0; i < props.size(); i++)
    {
      Loaded* f = props[i]->figure.get();
      if(!f)
        continue;
      if(f->prebuilt)
      {
        builds[i] = std::move(*f->prebuilt);
        f->prebuilt.reset();
        continue;
      }
      work.push_back(std::async(std::launch::async, [f, skin, &out = builds[i]] {
        fv::build::BuildOptions options;
        options.texturesFlipped = f->texturesFlipped;
        options.skin            = skin;
        options.texture         = [f](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
          auto it = f->images.find(fv::build::requestKey(req));
          return it == f->images.end() ? std::nullopt : it->second;
        };
        options.walkThrough = fv::walkThroughMaterials(f->fig->loaded.scene, f->fig->items);
        out = fv::build::buildGltf(f->fig->loaded.scene, f->fig->loaded.root, options);
      }));
    }
    fv::timing::Scope timeWait("buildGltf wait");
    for(auto& w : work)
    {
      try
      {
        w.get();
      }
      catch(const std::exception& e)
      {
        notice(std::string("A figure could not be built: ") + e.what(), "error");
      }
    }
  }

  std::vector<fv::composer::Part> parts(props.size());
  for(size_t i = 0; i < props.size(); i++)
  {
    const Prop&        p    = *props[i];
    fv::composer::Part& part = parts[i];
    part.id                 = p.id;
    part.name               = p.name;
    part.transform          = p.transform;
    part.visible            = p.visible;
    for(size_t j = 0; j < props.size(); j++)
      if(!p.parent.empty() && props[j]->id == p.parent)
        part.parent = int(j);
    if(p.figure && builds[i])
      part.model = &builds[i]->model;
    else if(p.model)
      part.model = p.model.get();
    else if(p.kind == PropKind::eCamera)
      part.camera = fv::composer::perspectiveCamera(p.camera.lens.focalLength);
    else if(p.kind == PropKind::eLight)
    {
      tinygltf::Light l;
      l.name                   = p.name;
      l.type                   = p.light.type;
      const glm::vec3 c        = fv::lighting::linearColor(p.light.color);
      l.color                  = {c.x, c.y, c.z};
      l.intensity              = shown(p) ? p.light.intensity : 0.0;
      l.spot.innerConeAngle    = p.light.inner;
      l.spot.outerConeAngle    = p.light.outer;
      tinygltf::Value::Object extras;
      extras["radius"] = tinygltf::Value(p.light.radius);
      l.extras         = tinygltf::Value(extras);
      part.light       = l;
    }
  }
  std::vector<fv::composer::Placed> placed;
  tinygltf::Model                  composed = [&] {
    fv::timing::Scope timeMerge("merge");
    tinygltf::Model   m = fv::composer::compose(parts, placed);
    addSun(m);
    return m;
  }();

  const bool hasGeometry = std::any_of(props.begin(), props.end(), [](const auto& p) { return p->figure || p->model; });
  if(auto manip = renderer->getCameraManipulator())
  {
    glm::dvec3 eye, center, up;
    manip->getLookat(eye, center, up);
    cameraRestore = std::array<glm::dvec3, 3>{eye, center, up};
  }
  // the first thing in an empty scene is framed; after that the camera stays, and so does a camera looked through
  if(hasGeometry && !composedAny && viewCamera.empty())
    cameraRestore.reset();
  composedAny = hasGeometry;
  const int selected = renderer->selectedNodeIndex();
  const std::string selectedProp = view ? view->selectedProp() : "";
  // the clip playing, its speed and loop: loadModel starts the animation over from the first clip
  struct Clip
  {
    std::string prop, name;
    float       speed = 1.f;
    bool        loop = true, play = false;
  };
  std::optional<Clip> clip;
  {
    const Json a = animationTopic();
    const int  c = a.value("current", -1);
    if(c >= 0 && a["clips"].contains(std::to_string(c)))
    {
      const Json&             info = a["clips"][std::to_string(c)];
      const AnimationControl& ctrl = renderer->editAnimationControl();
      clip = Clip{info["prop"].is_string() ? info["prop"].get<std::string>() : "", info.value("name", ""), ctrl.speed, ctrl.loop, ctrl.play};
    }
  }
  const bool loaded = [&] {
    nvutils::ScopedTimer t("loadModel");
    fv::timing::Scope    ts("loadModel");
    return renderer->loadModel(std::move(composed));
  }();
  if(!loaded)
  {
    sunNode = sunLight = -1;
    notice(label + ": the renderer could not take the scene.", "error");
    finish();
    return;
  }
  // the renderer put the ground at the scene's lowest point; "auto" keeps that
  for(const nvutils::ParameterBase* p : exposedParameters(registry))
    if(p->info.name == "infinitePlaneDistance" && p->type == nvutils::ParameterBase::Type::FLOAT32)
      autoGround = *static_cast<const float*>(p->destination.raw);
  appliedParams.erase("infinitePlaneDistance");
  appliedParams.erase("ptFireflyClamp");  // the load recalibrated it (calibrateFireflyClamp); ours goes again
  applyLighting(false);

  fv::timing::Scope              timeAttach("attach");
  std::vector<FvView::PropNodes> views;
  nvvkgltf::Scene*               scn = renderer->editScene();
  for(size_t i = 0; i < props.size(); i++)
  {
    Prop& p  = *props[i];
    p.placed = placed[i];
    const nvvkgltf::IndexRemapping& base = p.placed.base;
    const FvMaterials::Placement    at{base.materials, base.textures, base.samplers, base.meshes};
    FvView::PropNodes               nodes;
    nodes.id   = p.id;
    nodes.root = p.placed.root;
    if(p.figure && builds[i])
    {
      Loaded&                 f     = *p.figure;
      fv::build::BuildResult& built = *builds[i];
      f.nodeOf                      = std::move(built.nodeOf);
      for(int& n : f.nodeOf)
        if(n >= 0)
          n += base.nodes;
      f.sceneOf.clear();
      for(int n = 0; n < int(f.nodeOf.size()); n++)
        if(f.nodeOf[n] >= 0)
          f.sceneOf[f.nodeOf[n]] = n;
      f.meshOf = std::move(built.meshOf);
      for(int& m : f.meshOf)
        if(m >= 0)
          m += base.meshes;
      f.skinOf = std::move(built.skinOf);
      for(int& k : f.skinOf)
        if(k >= 0)
          k += base.skins;
      // one entry per vertex set: the mesh's material groups read one set of buffers, so a dial's
      // vertices go up once, not once per group
      f.primsOf.assign(f.meshOf.size(), {});
      for(size_t m = 0; m < f.meshOf.size(); m++)
        if(f.meshOf[m] >= 0)
          f.primsOf[m] = renderer->vertexSetOwnersOfMesh(f.meshOf[m]);
      // the glTF already holds the scene's dials and binds; don't upload them again
      f.fig->drain();
      if(!p.materials)
        p.materials = makeMaterials(props[i]);
      p.materials->attach(p.id, &f.fig->loaded.scene, built, f.texturesFlipped, skinMode.load(), at);
      publishEyes(p);
      publishSkin(p);
      nodes.materials     = at.materials;
      nodes.materialCount = int(f.fig->loaded.scene.materials.size());
      Json described = FvView::describeFigure(nodes, f.fig->loaded.scene, f.fig->loaded.root, f.nodeOf, f.fig->morphs ? f.fig->morphs->size() : 0);
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
      p.stats = described["stats"];
      protocol->events().snapshot("nodes/" + p.id, {{"prop", p.id}, {"nodes", described["nodes"]}});
    }
    else if(p.figure)
    {
      // its build failed, so none of it is in this scene: nothing may address the scene before
      Loaded& f = *p.figure;
      std::fill(f.nodeOf.begin(), f.nodeOf.end(), -1);
      std::fill(f.meshOf.begin(), f.meshOf.end(), -1);
      std::fill(f.skinOf.begin(), f.skinOf.end(), -1);
      f.sceneOf.clear();
      f.primsOf.assign(f.meshOf.size(), {});
      if(p.materials)
        p.materials->detach();
    }
    else if(p.model && scn && scn->valid())
    {
      if(!p.materials)
        p.materials = makeMaterials(props[i]);
      p.materials->attachModel(p.id, at, int(p.model->materials.size()), int(p.model->meshes.size()));
      nodes.materials     = at.materials;
      nodes.materialCount = int(p.model->materials.size());
      Json described = FvView::describeModel(nodes, scn->getModel(), p.placed.root, p.placed.firstNode, p.placed.lastNode, base.meshes);
      p.stats        = described["stats"];
      protocol->events().snapshot("nodes/" + p.id, {{"prop", p.id}, {"nodes", described["nodes"]}});
    }
    views.push_back(std::move(nodes));
  }
  if(view)
    view->setProps(std::move(views));
  for(const auto& p : props)
    if(p->figure)
    {
      publishFigure(*p);
      if(p->figure->dforce && request.settle.count(p->figure.get()))
        p->figure->dforce->poke();
      if(p->figure->collider && request.settle.count(p->figure.get()))
        p->figure->collider->poke();
    }
  // the selection was the node of a prop; it is again where that prop is now
  if(const auto p = findProp(selectedProp))
    renderer->selectSceneNodeFromScript(selected >= 0 && p->placed.root >= 0 ? p->placed.root : -1);
  if(clip)
  {
    // the same clip of the same prop, if it is still in the scene
    AnimationControl& ctrl  = renderer->editAnimationControl();
    const Json        a     = animationTopic();
    bool              found = false;
    for(auto it = a["clips"].begin(); it != a["clips"].end() && !found; ++it)
      if((it.value()["prop"].is_string() ? it.value()["prop"].get<std::string>() : "") == clip->prop && it.value().value("name", "") == clip->name)
      {
        ctrl.currentAnimation = std::stoi(it.key());
        ctrl.reset            = true;
        ctrl.speed            = clip->speed;
        ctrl.loop             = clip->loop;
        ctrl.play             = clip->play;
        ctrl.runOnce          = !clip->play;
        found                 = true;
      }
    if(!found)
      ctrl.play = false;
  }
  publishScene();
  publishAnimation(true);
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
  requestCompose("Adding " + p->name, [this, id = p->id] { selectProp(id); });
  return {{"prop", p->id}};
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
    nvvkgltf::Scene* scene = renderer->editScene();
    if(scene && scene->valid() && p->placed.light >= 0 && p->placed.light < int(scene->getModel().lights.size()))
    {
      tinygltf::Light& light       = scene->editor().getLightForEdit(p->placed.light);
      const glm::vec3  c           = fv::lighting::linearColor(p->light.color);
      light.color                  = {c.x, c.y, c.z};
      light.intensity              = shown(*p) ? p->light.intensity : 0.0;
      light.spot.innerConeAngle    = p->light.inner;
      light.spot.outerConeAngle    = p->light.outer;
      tinygltf::Value::Object extras = light.extras.IsObject() ? light.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
      extras["radius"]               = tinygltf::Value(p->light.radius);
      light.extras                   = tinygltf::Value(extras);
      scene->markLightDirty(p->placed.light);
    }
  }
  if(args.contains("parent"))
  {
    const std::string parent = args["parent"].is_string() ? args["parent"].get<std::string>() : "";
    if(parent != p->parent)
    {
      // it stays where it is in the world, now relative to its new parent
      auto at = parent.empty() ? nullptr : findProp(parent);
      if(!parent.empty() && !at)
        throw std::runtime_error("no such parent");
      for(Prop* up = at.get(); up; up = up->parent.empty() ? nullptr : findProp(up->parent).get())
        if(up == p.get())
          throw std::runtime_error(p->name + " can't ride on something that rides on it");
      const glm::mat4 world = worldOf(*p);
      const glm::mat4 local = at ? glm::inverse(worldOf(*at)) * world : world;
      glm::vec3       skew;
      glm::vec4       perspective;
      glm::decompose(local, p->transform.scale, p->transform.rotation, p->transform.translation, skew, perspective);
      p->euler.reset();
      p->parent = parent;
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
  return {{"prop", p->id}};
}

Json FvHost::Impl::sceneClear()
{
  loadGeneration++;  // loads still running drop their result
  for(auto p : std::vector(props))
    removeProp(p);
  propCount = 0;
  requestCompose("Clearing the scene");
  return {{"ok", true}};
}

Json FvHost::Impl::load(const Json& args)
{
  HostSettings s;
  {
    std::lock_guard lock(settingsMutex);
    s = settings;
  }
  if(args.contains("path"))
  {
    std::string ext = pathOf(args["path"].get<std::string>()).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if(ext == ".gltf" || ext == ".glb" || ext == ".obj")
      return importModel(args["path"].get<std::string>());
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
  // how it was opened, for a scene file
  const Json source = args.contains("rel") ? Json{{"rel", args["rel"]}} : Json{{"path", args["path"]}};
  // a wearable goes on the figure named or selected, unless it is to be a prop of its own
  const bool asProp = args.value("as", "") == "prop" || args.value("replace", false);
  std::string target = args.contains("prop") && args["prop"].is_string() ? args["prop"].get<std::string>() : "";
  std::vector<std::string> prefetch;
  if(args.contains("prefetch") && args["prefetch"].is_array())
    for(const Json& r : args["prefetch"])
      if(r.is_string())
        prefetch.push_back(r.get<std::string>());
  return loadFigure(s, library->first, library->second, fileStem(name), source, target, asProp, {}, std::move(prefetch));
}

Json FvHost::Impl::loadFigure(const HostSettings& s, std::vector<std::filesystem::path> roots, std::string rel, const std::string& name, Json source,
                              std::string target, bool asProp, std::function<void(std::shared_ptr<Prop>)> then, std::vector<std::string> prefetch)
{
  const uint64_t    jobId      = ++jobs;
  const uint64_t    generation = loadGeneration;
  const std::string label      = "Loading " + utf8Of(pathOf(rel).filename());
  job(jobId, {{"label", label}, {"state", "running"}});

  loaders.spawn([this, s, roots = std::move(roots), rel = std::move(rel), jobId, label, asProp, source = std::move(source), target = std::move(target),
                 generation, then = std::move(then), prefetch = std::move(prefetch)] {
    try
    {
      auto lib   = std::make_shared<fv::dson::Library>(roots);
      auto store = std::make_shared<fv::dson::Store>(lib);
      for(const std::string& wear : prefetch)
        fv::dson::prefetchScene(store, wear);
      // a preset is applied to a figure in the scene
      auto doc = store->load(rel);
      if(doc)
        if(auto preset = fv::dson::describePreset(doc->json))
        {
          // an environment preset's HDR, found through the preset's own library
          std::string envPath;
          if(preset->environment)
            if(auto found = lib->find(normRel(preset->environment->image)))
              envPath = utf8Of(*found);
          post([this, body = doc->json, kind = preset->kind, jobId, rel, env = preset->environment, envPath, target] {
            applyPreset(body, kind, jobId, rel, env, envPath, target);
          });
          return;
        }
      // clothing, hair and anatomy go on a figure in the scene, as in DAZ Studio
      if(!asProp && doc && fv::dson::isWearable(doc->json))
      {
        const std::string wearer = onApp([this, target] {
          const auto p = figureProp(target.empty() ? Json::object() : Json{{"prop", target}});
          return p && p->figure->fig->pose ? p->id : std::string();
        });
        if(!wearer.empty())
        {
          // done once it is on: the app waits on this job to lay a colour preset over the piece,
          // which finds none of its surfaces while the wear is still loading
          const auto worn = onApp([this, wearer] { return figureArg(Json{{"prop", wearer}}); });
          startWear(worn, {{rel, "", {}}}, {}, [this, jobId, label] { job(jobId, {{"label", label}, {"state", "done"}}); });
          return;
        }
        if(onApp([this] { return std::count_if(props.begin(), props.end(), [](const auto& p) { return p->figure && p->figure->fig->pose; }) > 1; }))
          throw std::runtime_error("Select the figure to put " + fileStem(rel) + " on.");
      }
      const auto t0 = std::chrono::steady_clock::now();
      auto       f = fv::Figure::load(store, rel, {.subdivision = s.subdivision}, [this, jobId, label](const std::string& step) {
        job(jobId, {{"label", label + ": " + step}, {"state", "running"}});
      });
      if(generation != loadGeneration)
      {
        job(jobId, {{"label", label}, {"state", "done"}});
        return;  // the scene was cleared meanwhile
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
      // Original textures skips the cooked set: the sources, cutouts composed here.
      // the path tracer gets the skin's volume; the rasterizer would show it see-through
      const fv::mat::SkinMode skin = skinMode.load();
      job(jobId, {{"label", label + ": Textures"}, {"state", "running"}});
      const bool cooker = !s.texServer.empty() && !s.rootId.empty();
      bool       cooked = cooker && !s.originalTextures;
      std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
      {
        fv::timing::Scope                                   timeTextures("textures");
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
          resolve([&](const fv::build::TextureRequest& req) { return fallbackTexture(s, *lib, req, cooker); });
          if(cooked)
            cookLater(s, requests);
          cooked = false;
        }
        prefetchImages(requests, found);
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
      options.walkThrough = fv::walkThroughMaterials(f->loaded.scene, f->items);
      auto built = [&] {
        fv::timing::Scope t("buildGltf");
        return fv::build::buildGltf(f->loaded.scene, f->loaded.root, options);
      }();
      LOGI("[fv] %s built in %.2f s: %zu nodes, %zu meshes, %zu materials, %zu images, %zu bytes of geometry\n", rel.c_str(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), built.model.nodes.size(), built.model.meshes.size(),
           built.model.materials.size(), built.model.images.size(), built.model.buffers.empty() ? size_t(0) : built.model.buffers[0].data.size());
      // the renderer takes it on the application thread
      auto shared = std::make_shared<std::pair<std::unique_ptr<fv::Figure>, fv::build::BuildResult>>(std::move(f), std::move(built));
      const bool flipped = cooked;
      post([this, shared, generation, jobId, flipped, skin, lib, images = std::move(images), source, then]() mutable {
        install(std::move(shared->first), std::move(shared->second), generation, jobId, flipped, skin, lib, std::move(images), std::move(source), then);
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
                           fv::mat::SkinMode skin, std::shared_ptr<fv::dson::Library> lib,
                           std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images, Json source,
                           std::function<void(std::shared_ptr<Prop>)> then)
{
  // the scene was cleared meanwhile: this one is dropped, and its job ends
  if(generation != loadGeneration)
  {
    job(jobId, {{"state", "done"}});
    return;
  }
  auto f             = std::make_shared<Loaded>();
  f->fig             = std::move(fig);
  f->images          = std::move(images);
  f->source          = std::move(source);
  f->texturesFlipped = texturesFlipped;
  f->lib             = std::move(lib);
  // the render mode may have changed while it loaded: built for another, it is built again
  if(skin == skinMode.load())
    f->prebuilt = std::move(built);
  // its add-ons go on once it is in the scene; until then the eyes it will have aren't there to choose
  f->addonsPending = !f->fig->addons.empty() && f->fig->pose;
  // "Mirel.duf" reads as "Mirel"
  auto p    = addProp(PropKind::eFigure, fileStem(f->fig->loaded.name));
  f->id     = p->id;
  p->figure = f;
  if(f->fig->morphs && !f->fig->loaded.morphs.mainMeshes.empty())
  {
    const int body = f->fig->loaded.morphs.meshes[f->fig->loaded.morphs.mainMeshes[0]].mesh;
    f->dforce      = std::make_unique<fv::dforce::Dforce>(f->fig->loaded.scene, *f->fig->morphs, body, f->fig->loaded.root);
    f->dforce->playing = [this] { return renderer->editAnimationControl().play; };
#ifdef FV_DFORCE_GPU
    // the experimental GPU solver, for mode "gpu": made and stepped on the application thread
    f->dforce->makeGpuSim = [this](std::vector<fv::dforce::SettleMesh> meshes, const fv::dforce::SettleOptions& opts)
        -> std::shared_ptr<fv::dforce::ClothBackend> {
      if(!app)
        return nullptr;
      auto sim = std::make_shared<DforceGpu>(&renderer->allocator(), app, std::move(meshes), opts);
      if(!sim->valid())
      {
        notice("dForce: the GPU solver could not start; simulating on the CPU");
        return nullptr;
      }
      return sim;
    };
#endif
    // the figure's shown clothing collides in the simulation (a belt, a bra, shoes); its anatomy,
    // hair and props do not
    f->dforce->collidesWith = [fig = f->fig.get()](int mesh) {
      for(const fv::WornItem& w : fig->items)
        if(w.visible && w.kind == "Clothing" && std::find(w.meshes.begin(), w.meshes.end(), mesh) != w.meshes.end())
          return true;
      return false;
    };
    // off until the Wardrobe toggle (or a scene file) turns it on: a settle costs seconds on a dressed figure
    f->dforce->setEnabled(false);
    // on from the start: a pass is a fraction of a second on the pool, and it is what keeps a garment
    // fitted to another shape or generation out of the body
    f->collider = std::make_unique<fv::wardrobe::Collider>(f->fig->loaded.scene, *f->fig->morphs, body, f->fig->loaded.root);
    f->collider->settings = [fig = f->fig.get()] {
      std::map<int, fv::dson::Smoothing> all;
      for(const fv::WornItem& w : fig->items)
        if(w.visible)
          all.insert(w.collision.begin(), w.collision.end());
      return all;
    };
    f->collider->wornOrder = [fig = f->fig.get()] {
      std::vector<int> order;
      for(const fv::WornItem& w : fig->items)
        order.insert(order.end(), w.meshes.begin(), w.meshes.end());
      return order;
    };
    f->collider->owned = [raw = f.get()](int mesh) {
      const fv::Mesh& m = raw->fig->loaded.scene.meshes[size_t(mesh)];
      return raw->dforce && raw->dforce->enabled() && m.dforce && m.dforce->simulated();
    };
  }
  {
    std::lock_guard lock(morphMutex);
    morphFigures.insert(p->id);
  }
  requestCompose("Adding " + p->name, [this, p, f, jobId, then] {
    if(!inScene(f))
    {
      job(jobId, {{"state", "done"}});
      return;
    }
    job(jobId, {{"state", "done"}, {"prop", p->id}, {"figure", p->id}});
    selectProp(p->id);
    // Genesis 9 leaves its eyes, mouth and lashes to add-ons the scene file lists
    if(!f->fig->addons.empty() && f->fig->pose)
    {
      std::vector<WearRequest> requests;
      for(const fv::dson::PostLoadAddon& a : f->fig->addons)
        requests.push_back({a.file, a.name, a.presets});
      f->addonsPending = true;
      startWear(f, std::move(requests), {}, [this, f] {
        f->addonsPending = false;
        if(const auto p = findProp(f->id); p && p->figure == f)
        {
          publishEyes(*p);
          publishSkin(*p);
        }
        for(const auto& after : std::exchange(f->afterAddons, {}))
          after();
      });
    }
    if(then)
      then(p);
  });
}

// The library path the figure was opened from, a rel when it came from the library.
std::string figureSource(const Loaded& figure)
{
  if(!figure.source.is_object())
    return {};
  if(figure.source.contains("rel"))
    return figure.source.value("rel", "");
  if(figure.source.contains("path"))
    return figure.source.value("path", "");
  return {};
}

// "Genesis 9", plus " Female" or " Male" when the path says. Bones cover a file whose path never does.
std::string figureGeneration(const Loaded& figure)
{
  const std::string source = figureSource(figure);
  std::vector<std::string> bones;
  if(figure.fig && figure.fig->pose)
    for(const auto& pb : figure.fig->pose->bones())
      if(!pb.name.empty())
        bones.push_back(pb.name);
  const auto gen = fv::wardrobe::identifyGeneration(source, bones, {});
  if(!gen)
    return {};
  std::string label = fv::wardrobe::label(*gen);
  const std::string gender = fv::wardrobe::genderHint(source);
  if(!gender.empty())
  {
    label.push_back(' ');
    label.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(gender[0]))));
    label.append(gender, 1, std::string::npos);
  }
  return label;
}

// What the panels learn when a figure is composed into the scene: its pose and what it wears.
void FvHost::Impl::publishFigure(Prop& p)
{
  if(p.figure->fig->pose)
    protocol->events().snapshot("pose/" + p.id, poseSnapshot(*p.figure));
  publishWardrobe(*p.figure);
  publishDforce(*p.figure, true);
}

//--------------------------------------------------------------------------------------------------
// Wardrobe

// Topic "wardrobe/<prop>": {figure, items, collision: {enabled, busy, corrected}}; `collision` is the
// collider's state (meshes it holds out of the body now), when the figure has one.
void FvHost::Impl::publishWardrobe(Loaded& f)
{
  Json items = Json::object();
  for(const fv::WornItem& w : f.fig->items)
  {
    const fv::ClothState cloth = f.fig->clothOf(w);
    items["i:" + std::to_string(w.id)] = {{"label", w.label},
                                          {"kind", w.kind},
                                          {"rel", w.rel},
                                          {"visible", w.visible},
                                          {"own", w.rel.empty()},
                                          {"channels", w.channels.size()},
                                          {"collision", !w.collision.empty()},
                                          {"cloth", {{"able", cloth.able}, {"native", cloth.native}, {"on", cloth.on}, {"strength", cloth.strength}}}};
    // its scene meshes, as the materials topic's items list theirs: which surfaces are this item's
    items["i:" + std::to_string(w.id)]["meshes"] = w.meshes;
    // the colour preset on it, null for its own; absent, as it was worn
    if(const auto c = f.colors.find(w.id); c != f.colors.end())
      items["i:" + std::to_string(w.id)]["color"] = c->second.rel.empty() ? Json(nullptr) : Json(c->second.rel);
  }
  Json topic = {{"figure", f.id}, {"items", items}};
  if(f.collider)
  {
    f.collisionVersion = f.collider->version();
    topic["collision"] = {{"enabled", f.collider->enabled()}, {"busy", f.collider->busy()}, {"corrected", f.collider->corrected()}};
  }
  protocol->events().snapshot("wardrobe/" + f.id, topic);
}

// wardrobe.collision {prop, on}: the worn conforming meshes are pushed out of the posed body (true)
// or shown as they conform (false).
Json FvHost::Impl::wardrobeCollision(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->collider)
    throw std::runtime_error("no figure with a body is selected");
  figure->collider->setEnabled(args.value("on", true));
  publishWardrobe(*figure);
  return {{"ok", true}};
}

Json FvHost::Impl::wear(const Json& args)
{
  HostSettings s;
  {
    std::lock_guard lock(settingsMutex);
    s = settings;
  }
  std::string rel;
  if(args.contains("rel"))
    rel = args["rel"].get<std::string>();
  else
  {
    const std::string path = args.at("path").get<std::string>();
    const auto        lib  = libraryFor(s, pathOf(path));
    if(!lib)
      throw std::runtime_error(path + " is not in a DAZ content library (no data folder above it)");
    rel = lib->second;
  }
  std::shared_ptr<Loaded> target = onApp([this, args] { return figureArg(args); });
  if(!target || !target->fig->pose)
    throw std::runtime_error("Select a figure first: clothing and hair go on a figure with a skeleton.");
  return {{"job", startWear(target, {{rel, "", {}}})}};
}

// Puts wearables on a figure: all of them are loaded and bound side by side on workers against one
// copy of the figure, then merged on the application thread in request order, and the renderer's
// scene is composed again once at the end. A prepare reads only that copy, so the order of the
// merges (and the keys they give) is the order of the requests.
uint64_t FvHost::Impl::startWear(std::shared_ptr<Loaded> target, std::vector<WearRequest> requests, std::function<void()> beforeBuild, std::function<void()> after,
                                 std::shared_future<void> commitAfter)
{
  HostSettings s;
  std::shared_ptr<fv::dson::Library> lib;
  {
    std::lock_guard lock(settingsMutex);
    s   = settings;
    lib = target->lib;
  }
  const uint64_t    jobId = ++jobs;
  const std::string label = "Wearing " + utf8Of(pathOf(requests.front().rel).stem());
  job(jobId, {{"label", label}, {"state", "running"}});
  const fv::mat::SkinMode skin = skinMode.load();
  loaders.spawn([this, s, requests = std::move(requests), target, lib, jobId, label, skin, beforeBuild = std::move(beforeBuild), after = std::move(after),
                 commitAfter] {
    // A batch that commits after something else (a restore after the figure's add-ons) prepares at
    // once and takes the wear lock only to commit; any other batch holds it throughout.
    std::unique_lock serial(wearMutex, std::defer_lock);
    if(!commitAfter.valid())
      serial.lock();
    bool any  = false;
    auto fail = [&](const std::string& what) {
      notice(what, "error");
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", what}});
    };
    // A character lists material presets with its add-ons, but one preset often covers several (an
    // "all add-ons" preset on the eyebrows holds the eyes' and lashes' surfaces too), as DAZ applies
    // them to the whole figure: every add-on takes all of them, each surface where its node is.
    std::vector<std::string> addonPresets;
    for(const WearRequest& r : requests)
      if(!r.name.empty())
        for(const std::string& p : r.presets)
          if(std::find(addonPresets.begin(), addonPresets.end(), p) == addonPresets.end())
            addonPresets.push_back(p);
    std::mutex            presetMutex;  // the prepares run side by side
    std::set<std::string> presetMissing, presetMatched;
    auto applyPreset = [&](const std::string& presetRel, fv::SceneGraph& scene, bool report) {
      try
      {
        if(auto doc = target->fig->store().load(presetRel))
        {
          const fv::dson::MaterialPresetResult r = fv::dson::applyMaterialPreset(doc->json, scene);
          if(report)
          {
            std::lock_guard lock(presetMutex);
            presetMissing.insert(r.missing.begin(), r.missing.end());
            presetMatched.insert(r.matched.begin(), r.matched.end());
          }
        }
      }
      catch(const std::exception& e)
      {
        notice(presetRel + ": " + e.what());
      }
    };
    using Clock          = std::chrono::steady_clock;
    const auto msSince   = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    const auto tBatch    = Clock::now();
    // what is put on: an add-on whose node the figure already has is not put on twice
    // (by value: a timed-out onApp drops its task, but one already running finishes after this moves on)
    std::vector<const WearRequest*> todo;
    std::shared_ptr<fv::Figure::WearInputs> inputs;
    double                                  msBase = 0;
    try
    {
      std::vector<std::string> names;
      for(const WearRequest& r : requests)
        names.push_back(r.name);
      const auto worn = onApp([this, target, names] {
        std::vector<char> out(names.size(), 0);
        if(inScene(target))
          for(size_t i = 0; i < names.size(); i++)
            out[i] = !names[i].empty()
                     && std::any_of(target->fig->items.begin(), target->fig->items.end(),
                                    [&](const fv::WornItem& item) { return item.nodeNames.count(names[i]) > 0; });
        return out;
      });
      std::set<std::string> taken;
      for(size_t i = 0; i < requests.size(); i++)
        if(!worn[i] && (requests[i].name.empty() || taken.insert(requests[i].name).second))
          todo.push_back(&requests[i]);
      // the figure as it is now, copied on the application thread once for every prepare of the
      // batch: dials move its joint centres meanwhile, and commits grow it
      if(!todo.empty())
      {
        fv::timing::Scope t("wearBase (render thread)");
        const auto        tBase = Clock::now();
        inputs                  = std::make_shared<fv::Figure::WearInputs>(onApp([target] { return target->fig->wearInputs(); }));
        msBase                  = msSince(tBase);
      }
    }
    catch(const std::exception& e)
    {
      todo.clear();
      notice(e.what(), "error");
    }

    // every request prepared side by side, each on a thread of its own (a prepare waits on pool jobs,
    // so not on the pool): loaded, fitted and bound, its presets, strands and textures
    struct Slot
    {
      std::shared_ptr<fv::Figure::PreparedWear>                               prepared;
      std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
      double                                                                  msPrep = 0, msStrands = 0, msWorker = 0;
      std::string                                                             error;
    };
    std::vector<Slot> slots(todo.size());
    auto              prepare = [&](size_t i) {
      const WearRequest& request = *todo[i];
      Slot&              slot    = slots[i];
      const auto         tWear   = Clock::now();
      try
      {
        if(!accepting)
          return;
        job(jobId, {{"label", utf8Of(pathOf(request.rel).stem()) + ": loading"}, {"state", "running"}});
        const auto tPrep = Clock::now();
        slot.prepared    = target->fig->prepareWear(request.rel, s.subdivision, *inputs, target->texturesFlipped);
        slot.msPrep      = msSince(tPrep);
        // the material presets that go with it, applied to the wearable before its textures are looked up
        fv::SceneGraph& preparedScene = fv::Figure::preparedScene(*slot.prepared);
        {
          fv::timing::Scope t("material presets");
          if(const std::string brows = defaultBrowMaterial(preparedScene); !brows.empty())
            applyPreset(brows, preparedScene, false);
          for(const std::string& presetRel : request.name.empty() ? request.presets : addonPresets)
            applyPreset(presetRel, preparedScene, true);
          // as the character came: "Character's own" eyes are these (Genesis 9's are an add-on's)
          fv::dson::keepBaseline(preparedScene);
        }
        // a strand hair's direction is measured here, so the composition on the render thread finds it
        const auto tStrands = Clock::now();
        {
          fv::timing::Scope t("strand directions");
          fv::build::measureStrands(preparedScene, target->texturesFlipped);
        }
        slot.msStrands = msSince(tStrands);
        // the wearable's textures, the way the figure's were made: bottom-up cooked, or the sources
        job(jobId, {{"label", label + ": Textures"}, {"state", "running"}});
        fv::timing::Scope t("textures");
        const auto        requestsOf = fv::build::textureRequests(preparedScene, skin);
        std::vector<std::optional<fv::build::TextureImage>> found(requestsOf.size());
        std::atomic<size_t>                                 next{0};
        std::vector<std::thread>                            workers;
        const bool                                          cooker = !s.texServer.empty() && !s.rootId.empty();
        for(int w = 0; w < std::min<int>(8, int(requestsOf.size())); w++)
          workers.emplace_back([&] {
            for(size_t k; (k = next++) < requestsOf.size();)
              found[k] = target->texturesFlipped ? (cooker ? cookedTexture(s, requestsOf[k], true, false).image : std::nullopt) :
                                                   (lib ? fallbackTexture(s, *lib, requestsOf[k], cooker) : std::nullopt);
          });
        for(auto& w : workers)
          w.join();
        prefetchImages(requestsOf, found);
        for(size_t k = 0; k < requestsOf.size(); k++)
          slot.images[fv::build::requestKey(requestsOf[k])] = found[k];
      }
      catch(const std::exception& e)
      {
        slot.prepared.reset();
        slot.error = e.what();
      }
      slot.msWorker = msSince(tWear);
    };
    {
      std::vector<std::thread> threads;
      for(size_t i = 1; i < todo.size(); i++)
        threads.emplace_back(prepare, i);
      if(!todo.empty())
        prepare(0);
      for(auto& t : threads)
        t.join();
    }

    // committed in request order, after what this batch waits for
    if(commitAfter.valid())
    {
      while(commitAfter.wait_for(100ms) != std::future_status::ready)
        if(!accepting)
          return;
      serial.lock();
    }
    for(size_t i = 0; i < todo.size(); i++)
    {
      const WearRequest& request = *todo[i];
      Slot&              slot    = slots[i];
      if(!accepting)
        return;
      if(!slot.prepared)
      {
        if(!slot.error.empty())
          notice(utf8Of(pathOf(request.rel).stem()) + ": " + slot.error, "error");
        continue;
      }
      auto done = std::make_shared<std::promise<bool>>();
      auto ok   = done->get_future();
      post([this, target, prepared = slot.prepared, rel = request.rel, label, images = std::move(slot.images), done]() mutable {
        if(!inScene(target))
        {
          done->set_value(false);
          return;
        }
        try
        {
          const fv::WearResult r = target->fig->commitWear(prepared, rel);
          for(auto& [key, image] : images)
            target->images.insert_or_assign(key, std::move(image));
          for(const std::string& w : r.warnings)
            notice(w);
          for(const std::string& n : r.notes)
            notice(n, "info");
          if(!r.missingBones.empty())
            notice(utf8Of(pathOf(rel).stem()) + ": this figure has no " + r.missingBones.front()
                   + (r.missingBones.size() > 1 ? " and " + std::to_string(r.missingBones.size() - 1) + " other bones" : "") + " to hang it from.");
          if(r.fit < 0.5)
            notice(utf8Of(pathOf(rel).stem())
                   + ": most of its bones found no match on this figure, so it was probably made for another figure and will not follow the pose.");
          done->set_value(true);
        }
        catch(const std::exception& e)
        {
          notice(e.what(), "error");
          done->set_value(false);
        }
      });
      // the application thread merges it; a renderer that is closing never will
      const auto tCommit = Clock::now();
      {
        fv::timing::Scope timeCommit("commit (render thread)");
        while(ok.wait_for(100ms) != std::future_status::ready)
          if(!accepting)
            return;
      }
      any = ok.get() || any;
      if(fv::timing::enabled())
      {
        const double msCommit = msSince(tCommit);
        char         buf[256];
        std::snprintf(buf, sizeof buf, "wear summary: worker %.0f ms (prepare %.0f, strands %.0f), render thread %.0f ms (wearBase %.0f, commit %.0f)",
                      slot.msWorker, slot.msPrep, slot.msStrands, msBase + msCommit, msBase, msCommit);
        fv::timing::line(buf);
        fv::timing::report("wear " + utf8Of(pathOf(request.rel).stem()), msSince(tBatch));
      }
    }
    // surfaces no worn item had; those of the figure itself took the character's preset at load
    for(const std::string& s : presetMatched)
      presetMissing.erase(s);
    if(!presetMissing.empty())
    {
      const auto unknown = onApp([target, names = presetMissing] {
        std::vector<std::string> out;
        for(const std::string& n : names)
          if(std::none_of(target->fig->loaded.scene.materials.begin(), target->fig->loaded.scene.materials.end(),
                          [&](const fv::Material& m) { return m.name == n; }))
            out.push_back(n);
        return out;
      });
      if(!unknown.empty())
      {
        std::string list;
        for(size_t i = 0; i < unknown.size() && i < 6; i++)
          list += (i ? ", " : "") + unknown[i];
        if(unknown.size() > 6)
          list += " and " + std::to_string(unknown.size() - 6) + " more";
        notice("Material presets name surfaces nothing worn has: " + list, "info");
      }
    }
    // `after` runs whatever happened, so a restore waiting on this wear still finishes
    if(!any)
    {
      fail("Nothing was put on.");
      if(after)
        post(after);
      return;
    }
    post([this, target, jobId, label, beforeBuild, after] {
      if(!inScene(target))
      {
        job(jobId, {{"label", label}, {"state", "failed"}, {"error", "the figure left the scene"}});
        if(after)
          after();
        return;
      }
      if(beforeBuild)
        beforeBuild();
      requestCompose(label, [this, jobId, label, after] {
        job(jobId, {{"label", label}, {"state", "done"}});
        if(after)
          after();
      }, target.get());
    });
  });
  return jobId;
}

namespace {
int itemIdOf(const Json& v)
{
  if(v.is_number_integer())
    return v.get<int>();
  const std::string s = v.get<std::string>();
  return std::atoi(s.c_str() + (s.rfind("i:", 0) == 0 ? 2 : 0));
}

int itemId(const Json& args)
{
  return itemIdOf(args.at("item"));
}

// `item` as one id or a list of them ("i:3", 3, ["i:3", "i:4"]).
std::vector<int> itemIds(const Json& args)
{
  const Json&      v = args.at("item");
  std::vector<int> out;
  if(v.is_array())
    for(const Json& x : v)
      out.push_back(itemIdOf(x));
  else
    out.push_back(itemIdOf(v));
  return out;
}
}  // namespace

namespace {
const fv::WornItem* wornItem(const fv::Figure& fig, int id)
{
  const auto it = std::find_if(fig.items.begin(), fig.items.end(), [&](const fv::WornItem& i) { return i.id == id; });
  return it == fig.items.end() ? nullptr : &*it;
}
}  // namespace

// Taken off, an item is hidden in the renderer's scene and stays there, buffers and BLAS, until the
// next composition (a wear, a geograft shown or hidden, a prop added) leaves it out. A geograft gives
// the body its polygons back, which changes the body's indices: that still composes the scene again.
Json FvHost::Impl::wardrobeRemove(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const int           id   = itemId(args);
  const fv::WornItem* item = wornItem(*figure->fig, id);
  if(!item)
    throw std::runtime_error("no such item");
  const bool geograft = !item->hiddenPolys.empty();
  const auto objects  = item->objects;
  figure->fig->removeItem(id);
  // shoes give the feet their pose back
  if(figure->fig->pose)
    protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
  if(geograft)
  {
    requestCompose("Taking off", {}, figure.get());
    return {{"ok", true}};
  }
  showItem(*figure, objects, false);
  if(figure->collider)
    figure->collider->poke();
  publishWardrobe(*figure);
  if(figure->dforce)
    figure->dforce->poke();
  return {{"ok", true}};
}

// wardrobe.cloth {prop, item, on?: bool | null, strength?}: dForce on one item (figure.hpp,
// clothSettings): true drapes it, at the defaults when its file has no simulation settings, false
// holds it, null leaves it to the file; `strength` scales the weights. Turning an item on turns the
// figure's dForce on, so the choice shows.
Json FvHost::Impl::wardrobeCloth(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const int           id   = itemId(args);
  const fv::WornItem* item = wornItem(*figure->fig, id);
  if(!item)
    throw std::runtime_error("no such item");
  std::optional<bool> on = item->cloth.on;
  if(args.contains("on"))
    on = args["on"].is_boolean() ? std::optional<bool>(args["on"].get<bool>()) : std::nullopt;
  const float strength = args.value("strength", item->cloth.strength);
  const auto  changed  = figure->fig->setItemCloth(id, on, strength);
  if(figure->dforce)
  {
    if(on && *on && !figure->dforce->enabled())
      figure->dforce->setEnabled(true);
    figure->dforce->changed(changed, false);
    figure->dforce->poke();
  }
  if(figure->collider)
    figure->collider->poke();
  publishWardrobe(*figure);
  if(figure->dforce)
    publishDforce(*figure, true);
  return {{"ok", true}};
}

Json FvHost::Impl::wardrobeVisible(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const int           id      = itemId(args);
  const bool          visible = args.at("visible").get<bool>();
  const fv::WornItem* item    = wornItem(*figure->fig, id);
  if(!item)
    throw std::runtime_error("no such item");
  figure->fig->setItemVisible(id, visible);
  if(!item->hiddenPolys.empty())
  {
    requestCompose("Showing", {}, figure.get());
    return {{"ok", true}};
  }
  showItem(*figure, item->objects, visible);
  if(figure->collider)
    figure->collider->poke();
  publishWardrobe(*figure);
  if(figure->dforce)
    figure->dforce->poke();
  return {{"ok", true}};
}

// An item's nodes shown or hidden in the renderer's scene as they are (KHR_node_visibility, as
// buildGltf writes it), without composing it again.
void FvHost::Impl::showItem(const Loaded& f, const std::vector<std::pair<int, std::string>>& objects, bool visible)
{
  nvvkgltf::Scene* scene = renderer->editScene();
  if(!scene || !scene->valid())
    return;
  tinygltf::Model& model = scene->getModel();
  for(const auto& [node, bone] : objects)
  {
    const int g = node >= 0 && node < int(f.nodeOf.size()) ? f.nodeOf[node] : -1;
    if(g < 0 || g >= int(model.nodes.size()))
      continue;
    tinygltf::utils::setNodeVisibility(model.nodes[g], {visible});
    scene->editor().updateVisibility(g);
  }
}

//--------------------------------------------------------------------------------------------------
// dForce

// A dForce mode by its name; "gpu" needs the build to have the GPU solver.
static fv::dforce::Mode dforceMode(const std::string& name)
{
  if(name == "live")
    return fv::dforce::Mode::eLive;
  if(name == "gpu")
  {
#ifdef FV_DFORCE_GPU
    return fv::dforce::Mode::eLiveGpu;
#else
    throw std::runtime_error("this renderer was built without the GPU cloth solver (FV_DFORCE_GPU)");
#endif
  }
  return fv::dforce::Mode::eSettle;
}

// Topic "dforce/<prop>": {figure, enabled, mode, active, busy, draped}, when it changes.
void FvHost::Impl::publishDforce(Loaded& f, bool force)
{
  if(!f.dforce)
    return;
  const fv::dforce::Dforce& d = *f.dforce;
  if(!force && f.dforceVersion == d.version())
    return;
  f.dforceVersion = d.version();
  protocol->events().snapshot("dforce/" + f.id, {{"figure", f.id},
                                                 {"enabled", d.enabled()},
                                                 {"mode", d.mode() == fv::dforce::Mode::eLive ? "live" : d.mode() == fv::dforce::Mode::eLiveGpu ? "gpu" : "settle"},
                                                 {"active", d.active()},
                                                 {"busy", d.busy()},
                                                 {"draped", d.draped()}});
}

// dforce.enable {prop, on}: cloth drapes (true) or shows the conforming shape (false);
// dforce.mode {prop, mode}: "settle" (once the pose stops changing), "live" (every frame) or "gpu"
// (every frame, on the experimental GPU solver, in a build with FV_DFORCE_GPU);
// dforce.resimulate {prop} settles the cloth now.
Json FvHost::Impl::dforceCommand(const std::string& cmd, const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->dforce)
    throw std::runtime_error("no figure with dynamic clothing is selected");
  if(cmd == "enable")
  {
    figure->dforce->setEnabled(args.value("on", true));
    // a mesh the simulation takes over (or gives back) changes hands with the collider
    if(figure->collider)
      figure->collider->poke();
  }
  else if(cmd == "mode")
    figure->dforce->setMode(dforceMode(args.value("mode", "settle")));
  else
    figure->dforce->resimulate();
  publishDforce(*figure, true);
  return {{"ok", true}};
}

//--------------------------------------------------------------------------------------------------
// Imported models and animation

Json FvHost::Impl::importModel(const std::string& path, std::function<void(std::shared_ptr<Prop>)> then)
{
  const uint64_t    jobId      = ++jobs;
  const uint64_t    generation = loadGeneration;
  const std::string label      = "Loading " + utf8Of(pathOf(path).filename());
  job(jobId, {{"label", label}, {"state", "running"}});
  loaders.spawn([this, path, jobId, label, generation, then = std::move(then)] {
    auto        model = std::make_shared<tinygltf::Model>();
    std::string error;
    if(!fv::composer::readModelFile(pathOf(path), *model, error))
    {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", error}});
      notice("Could not open " + utf8Of(pathOf(path).filename()) + ": " + error, "error");
      return;
    }
    post([this, model, path, jobId, label, generation, then] {
      if(generation != loadGeneration)
      {
        job(jobId, {{"label", label}, {"state", "done"}});
        return;
      }
      std::string ext = pathOf(path).extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::toupper(c)); });
      auto p    = addProp(PropKind::eModel, utf8Of(pathOf(path).stem()));
      p->model  = model;
      p->path   = path;
      p->format = ext == ".OBJ" ? "OBJ" : "glTF";
      requestCompose("Adding " + p->name, [this, p, jobId, label, then] {
        if(!findProp(p->id))
        {
          job(jobId, {{"label", label}, {"state", "done"}});
          return;
        }
        job(jobId, {{"label", label}, {"state", "done"}, {"prop", p->id}, {"figure", p->id}});
        selectProp(p->id);
        if(then)
          then(p);
      });
    });
  });
  return {{"job", jobId}};
}

// The clips in the scene, each with the prop whose nodes it moves, and where playback is.
Json FvHost::Impl::animationTopic()
{
  Json                    clips = Json::object();
  nvvkgltf::Scene*        scn   = renderer->editScene();
  const AnimationControl& ctrl  = renderer->editAnimationControl();
  Json                    out   = {{"clips", clips}, {"current", -1}, {"playing", false}, {"time", 0.0}, {"speed", ctrl.speed}, {"loop", ctrl.loop}};
  if(!scn || !scn->valid() || !scn->animation().hasAnimation())
    return out;
  std::vector<fv::composer::Placed> placed;
  for(const auto& p : props)
    placed.push_back(p->placed);
  const tinygltf::Model& model = scn->getModel();
  const int              n     = scn->animation().getNumAnimations();
  for(int i = 0; i < n; i++)
  {
    const nvvkgltf::AnimationInfo& info = scn->animation().getAnimationInfo(i);
    Json                           clip = {{"name", info.name.empty() ? "Clip " + std::to_string(i + 1) : info.name}, {"duration", std::max(0.f, info.end - info.start)}, {"order", i}, {"prop", nullptr}};
    if(i < int(model.animations.size()) && !model.animations[size_t(i)].channels.empty())
    {
      const int part = fv::composer::partOfNode(placed, model.animations[size_t(i)].channels[0].target_node);
      if(part >= 0)
        clip["prop"] = props[size_t(part)]->id;
    }
    clips[std::to_string(i)] = clip;
  }
  out["clips"]   = clips;
  out["current"] = ctrl.currentAnimation >= 0 && ctrl.currentAnimation < n ? ctrl.currentAnimation : -1;
  out["playing"] = ctrl.play;
  if(out["current"] >= 0)
  {
    const nvvkgltf::AnimationInfo& info = scn->animation().getAnimationInfo(out["current"].get<int>());
    out["time"]                         = std::round(std::max(0.f, info.currentTime - info.start) * 1000.f) / 1000.0;
  }
  return out;
}

void FvHost::Impl::publishAnimation(bool force)
{
  const auto now = std::chrono::steady_clock::now();
  if(!force && now - animationAt < 100ms)
    return;
  animationAt = now;
  Json topic  = animationTopic();
  if(!force && topic == lastAnimation)
    return;
  lastAnimation = topic;
  protocol->events().snapshot("animation", std::move(topic));
}

Json FvHost::Impl::animationCommand(const std::string& cmd, const Json& args)
{
  nvvkgltf::Scene*  scn  = renderer->editScene();
  AnimationControl& ctrl = renderer->editAnimationControl();
  if(!scn || !scn->valid() || !scn->animation().hasAnimation())
    throw std::runtime_error("nothing in the scene has animations");
  const int n = scn->animation().getNumAnimations();
  if(cmd == "select")
  {
    const int clip = args.at("clip").get<int>();
    if(clip < 0 || clip >= n)
      throw std::runtime_error("no such clip");
    ctrl.currentAnimation = clip;
    ctrl.reset            = true;
    ctrl.play             = args.value("play", true);
    ctrl.runOnce          = !ctrl.play;
  }
  else if(cmd == "play")
  {
    if(ctrl.currentAnimation < 0 || ctrl.currentAnimation >= n)
      ctrl.currentAnimation = 0;
    const nvvkgltf::AnimationInfo& info = scn->animation().getAnimationInfo(ctrl.currentAnimation);
    // playing from the end starts over
    if(info.currentTime >= info.end - 1e-3f && !ctrl.loop)
      ctrl.reset = true;
    ctrl.play = true;
  }
  else if(cmd == "pause")
    ctrl.play = false;
  else if(cmd == "stop")
  {
    ctrl.play  = false;
    ctrl.reset = true;
  }
  else if(cmd == "seek")
  {
    if(ctrl.currentAnimation < 0 || ctrl.currentAnimation >= n)
      ctrl.currentAnimation = 0;
    ctrl.scrubTo(scn->animation().getAnimationInfo(ctrl.currentAnimation).start + args.at("time").get<float>(), scn);
  }
  else if(cmd == "speed")
    ctrl.speed = std::clamp(args.at("speed").get<float>(), 0.f, 100.f);
  else if(cmd == "loop")
    ctrl.loop = args.at("loop").get<bool>();
  publishAnimation(true);
  return {{"ok", true}};
}

// A prop as a GLB for other tools, as it stands. A figure: pose, dials, worn items and material
// edits; the glTF is built here (the scene belongs to this thread), composing the textures, which
// takes the time, and writing the file run on a worker. A model: its file's contents with its
// material edits. {scene: true}: everything in the scene, as the renderer has it, but the sun.
Json FvHost::Impl::exportGlb(const Json& args)
{
  const std::string path = args.at("path").get<std::string>();
  if(args.value("scene", false))
  {
    nvvkgltf::Scene* scn = renderer->editScene();
    if(!scn || !scn->valid() || props.empty())
      throw std::runtime_error("The scene is empty: there is nothing to export.");
    // without the sun the host added: the file describes the scene, not this session's lighting
    const int node = sunNode, light = sunLight;
    const auto withoutSun = [node, light](tinygltf::Model& m) {
      if(node < 0 || node >= int(m.nodes.size()))
        return;
      for(tinygltf::Scene& scene : m.scenes)
        scene.nodes.erase(std::remove(scene.nodes.begin(), scene.nodes.end(), node), scene.nodes.end());
      // appended last, so it goes without renumbering anything; else it stays, emptied and unreferenced
      if(node == int(m.nodes.size()) - 1)
        m.nodes.pop_back();
      else
        m.nodes[size_t(node)] = tinygltf::Node{};
      if(light >= 0 && light == int(m.lights.size()) - 1)
        m.lights.pop_back();
    };
    if(!renderer->save(pathOf(path), true, withoutSun))
      throw std::runtime_error("the renderer could not write " + path);
    notice("Exported " + utf8Of(pathOf(path).filename()), "info");
    return {{"ok", true}};
  }
  const auto prop = propArg(args);
  if(!prop || (!prop->figure && !prop->model))
    throw std::runtime_error("Select a figure or a model to export.");
  if(prop->model)
  {
    // the file's model with the factors material edits change (FvMaterials::writeModel) as the
    // scene shows them; its textures stay the file's own
    tinygltf::Model  copy = *prop->model;
    nvvkgltf::Scene* scn  = renderer->editScene();
    const auto&      base = prop->placed.base;
    for(size_t i = 0; scn && scn->valid() && i < copy.materials.size() && base.materials + i < scn->getModel().materials.size(); i++)
    {
      const tinygltf::Material& live      = scn->getModel().materials[base.materials + i];
      tinygltf::Material&       m         = copy.materials[i];
      m.pbrMetallicRoughness.baseColorFactor = live.pbrMetallicRoughness.baseColorFactor;
      m.pbrMetallicRoughness.roughnessFactor = live.pbrMetallicRoughness.roughnessFactor;
      m.pbrMetallicRoughness.metallicFactor  = live.pbrMetallicRoughness.metallicFactor;
      m.alphaMode                            = live.alphaMode;
      m.doubleSided                          = live.doubleSided;
      m.emissiveFactor                       = live.emissiveFactor;
      if(m.normalTexture.index >= 0)
        m.normalTexture.scale = live.normalTexture.scale;
      for(const char* ext : {"KHR_materials_emissive_strength", "KHR_materials_specular", "KHR_materials_clearcoat"})
      {
        const auto it = live.extensions.find(ext);
        if(it == live.extensions.end() || !it->second.IsObject())
          continue;
        tinygltf::Value::Object o = m.extensions.count(ext) && m.extensions[ext].IsObject() ? m.extensions[ext].Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
        for(const auto& [key, value] : it->second.Get<tinygltf::Value::Object>())
          if(value.IsNumber())
            o[key] = value;
        m.extensions[ext] = tinygltf::Value(o);
        if(std::find(copy.extensionsUsed.begin(), copy.extensionsUsed.end(), ext) == copy.extensionsUsed.end())
          copy.extensionsUsed.push_back(ext);
      }
    }
    std::string error;
    if(!fv::composer::packForGlb(copy, error))
      throw std::runtime_error(error);
    tinygltf::TinyGLTF writer;
    if(!writer.WriteGltfSceneToFile(&copy, utf8Of(pathOf(path)), false, true, false, true))
      throw std::runtime_error("could not write " + path);
    notice("Exported " + utf8Of(pathOf(path).filename()), "info");
    return {{"ok", true}};
  }
  const auto&                        figure = prop->figure;
  std::shared_ptr<fv::dson::Library> lib;
  {
    std::lock_guard lock(settingsMutex);
    lib = figure->lib;
  }
  auto              prepared = std::make_shared<fv::build::GlbExport>(fv::build::prepareGlb(figure->fig->loaded.scene, figure->fig->loaded.root));
  const uint64_t    jobId    = ++jobs;
  const std::string label    = "Exporting " + utf8Of(pathOf(path).filename());
  job(jobId, {{"label", label}, {"state", "running"}});
  loaders.spawn([this, prepared, lib, path, jobId, label] {
    try
    {
      fv::build::ExportOptions options;
      options.find = [lib](const std::string& rel) -> std::optional<std::filesystem::path> {
        if(!lib)
          return std::nullopt;
        std::string r = fv::dson::decodeUri(rel);
        std::replace(r.begin(), r.end(), '\\', '/');
        const size_t at = r.find_first_not_of('/');
        return lib->find(at == std::string::npos ? std::string() : r.substr(at));
      };
      const auto notes = fv::build::finishGlb(*prepared, pathOf(path), options);
      for(size_t i = 0; i < notes.size() && i < 5; i++)
        notice("Export: " + notes[i]);
      if(notes.size() > 5)
        notice("Export: " + std::to_string(notes.size() - 5) + " more textures could not be made");
      job(jobId, {{"label", label}, {"state", "done"}});
      notice("Exported " + utf8Of(pathOf(path).filename()) + (notes.empty() ? "" : " with " + std::to_string(notes.size()) + " textures replaced by neutral ones"), "info");
    }
    catch(const std::exception& e)
    {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", e.what()}});
      notice(std::string("Export failed: ") + e.what(), "error");
    }
  });
  return {{"job", jobId}};
}

//--------------------------------------------------------------------------------------------------
// Scene state

// What the host owns of a prop in a scene file: how a figure was opened, what it wears, its dials,
// its pose where it differs from how it loaded, and the edits of its materials; for a model, its
// file and material edits. The app adds the prop's place, cameras, lights and lighting from the
// scene topic and its own settings.
Json FvHost::Impl::sceneState(const Json& args)
{
  const auto prop = propArg(args);
  if(!prop)
    throw std::runtime_error("no such prop");
  if(prop->model)
    return {{"source", {{"path", utf8Of(pathOf(prop->path))}}},
            {"worn", Json::array()},
            {"morphs", Json::object()},
            {"pose", Json::array()},
            {"materials", prop->materials ? prop->materials->editState() : Json::object()}};
  if(!prop->figure)
    return Json::object();
  fv::Figure& fig  = *prop->figure->fig;
  Json        worn = Json::array();
  for(const fv::WornItem& w : fig.items)
    if(!w.rel.empty())
    {
      Json entry = {{"rel", w.rel}, {"visible", w.visible}, {"label", w.label}};
      // dForce on the item as the user set it, when they did
      if(w.cloth.on || w.cloth.strength != 1.f)
        entry["cloth"] = {{"on", w.cloth.on ? Json(*w.cloth.on) : Json(nullptr)}, {"strength", std::round(w.cloth.strength * 1e3) / 1e3}};
      // the colour chosen for it, null for its own
      if(const auto c = prop->figure->colors.find(w.id); c != prop->figure->colors.end())
        entry["color"] = c->second.rel.empty() ? Json(nullptr) : Json(c->second.rel);
      worn.push_back(std::move(entry));
    }
  Json morphs = Json::object();
  for(const fv::morph::Channel& c : fig.morphs->channels())
    if(!c.hidden && c.value != 0)
      morphs[c.name] = std::round(c.value * 1e5) / 1e5;
  Json pose = Json::array();
  if(fig.pose)
    for(const auto& pb : fig.pose->bones())
      for(const auto& a : pb.axes)
        if(a.value != a.initial)
          pose.push_back({{"bone", pb.id}, {"kind", fv::pose::kindName(a.kind)}, {"axis", a.axis}, {"value", std::round(a.value * 1e5) / 1e5}});
  Json state = {{"source", prop->figure->source},
                {"worn", worn},
                {"morphs", morphs},
                {"pose", pose},
                {"materialPresets", prop->figure->materialPresets},
                {"materials", prop->materials ? prop->materials->editState() : Json::object()}};
  // the eyes chosen, null for the character's own; none when nothing was chosen
  if(prop->figure->eyes)
    state["eyes"] = prop->figure->eyes->empty() ? Json(nullptr) : Json(*prop->figure->eyes);
  // the same for the skin and the makeup
  if(prop->figure->skin)
    state["skin"] = prop->figure->skin->empty() ? Json(nullptr) : Json(*prop->figure->skin);
  if(prop->figure->makeup)
    state["makeup"] = prop->figure->makeup->empty() ? Json(nullptr) : Json(*prop->figure->makeup);
  if(prop->figure->collider)
    state["collision"] = prop->figure->collider->enabled();
  if(prop->figure->dforce)
  {
    state["dforce"]     = prop->figure->dforce->enabled();
    state["dforceMode"] = prop->figure->dforce->mode() == fv::dforce::Mode::eLive    ? "live" :
                          prop->figure->dforce->mode() == fv::dforce::Mode::eLiveGpu ? "gpu" :
                                                                                       "settle";
  }
  return state;
}

// Puts a scene file's state on a prop: for a figure, what it wears first (items before dials, since
// an item brings dials of its own), then dials, pose, material presets and material edits, and
// dForce last, so it drapes the finished figure. The figure's own add-ons are already going on
// with their presets; they are waited for, not worn again.
Json FvHost::Impl::sceneRestore(const Json& args)
{
  const auto prop = propArg(args);
  if(!prop)
    throw std::runtime_error("no such prop");
  // a model takes back its material edits; it has no wardrobe, dials or pose
  if(!prop->figure)
  {
    std::vector<std::string> problems;
    if(prop->materials && args.contains("materials") && args["materials"].is_object())
      problems = prop->materials->restoreEdits(args["materials"]);
    for(size_t i = 0; i < problems.size() && i < 5; i++)
      notice(problems[i]);
    return {{"ok", true}};
  }
  std::shared_ptr<Loaded> target = prop->figure;
  const uint64_t          jobId  = ++jobs;
  job(jobId, {{"label", "Restoring " + prop->name}, {"state", "running"}});
  const Json worn  = args.value("worn", Json::array());
  // the worn files start reading now: they are put on only after the figure's add-ons
  {
    std::set<std::string> started;
    for(const Json& w : worn)
      if(w.is_object() && w.contains("rel") && w["rel"].is_string() && started.insert(relKey(w["rel"].get<std::string>())).second)
        target->fig->prefetchWear(w["rel"].get<std::string>());
  }
  auto       apply = [this, target, args, jobId] {
    const auto prop = findProp(target->id);
    if(!prop || prop->figure != target)
    {
      job(jobId, {{"state", "failed"}, {"error", "the figure left the scene"}});
      return;
    }
    std::vector<std::string> problems;
    fv::Figure&              fig = *target->fig;
    if(args.contains("morphs") && args["morphs"].is_object())
    {
      std::vector<std::pair<std::string, double>> dials;
      int                                         missing = 0;
      for(auto it = args["morphs"].begin(); it != args["morphs"].end(); ++it)
        if(it.value().is_number() && fig.morphs->channel(it.key()))
          dials.emplace_back(it.key(), it.value().get<double>());
        else
          missing++;
      if(!dials.empty())
        fig.morphs->setMany(dials);
      if(missing)
        problems.push_back(std::to_string(missing) + (missing == 1 ? " morph dial" : " morph dials") + " not found");
    }
    if(args.contains("pose") && args["pose"].is_array() && !args["pose"].empty())
    {
      if(!fig.pose)
        problems.push_back("the pose was skipped: this figure has no skeleton");
      else
      {
        std::vector<fv::pose::ChannelValue> values;
        for(const Json& c : args["pose"])
        {
          fv::pose::ChannelKind kind;
          if(fv::pose::parseKind(c.value("kind", ""), kind))
            values.push_back({c.value("bone", ""), kind, c.value("axis", ""), c.value("value", 0.0)});
        }
        fig.pose->setMany(values);
        protocol->events().snapshot("pose/" + target->id, poseSnapshot(*target));
      }
    }
    if(args.contains("materialPresets") && args["materialPresets"].is_array())
      for(const Json& p : args["materialPresets"])
      {
        const std::string rel = p.is_string() ? p.get<std::string>() : "";
        if(rel.empty())
          continue;
        try
        {
          auto doc = fig.store().load(rel);
          if(!doc)
          {
            problems.push_back("material preset " + fileStem(rel) + " isn't in the content library");
            continue;
          }
          // eyes replace the eyes before them instead of stacking on them (older files list each one tried)
          if(fv::dson::isEyePreset(doc->json, fig.loaded.scene))
          {
            applyEyes(*prop, rel, &doc->json, true);
            continue;
          }
          const fv::dson::MaterialPresetResult r = fv::dson::applyMaterialPreset(doc->json, fig.loaded.scene);
          if(prop->materials)
            prop->materials->surfacesChanged(r.changed);
          if(!r.changed.empty())
          {
            target->materialPresets.push_back(rel);
            presetChangedEyes(*prop, rel, r.changed);
          }
        }
        catch(const std::exception& e)
        {
          problems.push_back("material preset " + fileStem(rel) + ": " + e.what());
        }
      }
    // the skin, eyes and makeup chosen, over the presets and under the edits (a tone, an iris colour
    // and a lipstick's are edits); the skin first, as it takes the makeup off
    const auto restoreChoice = [&](const char* key, const char* what, auto apply) {
      if(!args.contains(key) || !(args[key].is_string() || args[key].is_null()))
        return;
      const std::string rel = args[key].is_string() ? args[key].get<std::string>() : std::string();
      try
      {
        std::shared_ptr<const fv::dson::Doc> doc;
        if(!rel.empty() && !(doc = fig.store().load(rel)))
          problems.push_back(std::string(what) + " " + fileStem(rel) + " isn't in the content library");
        else
          (this->*apply)(*prop, rel, doc ? &doc->json : nullptr, true);
      }
      catch(const std::exception& e)
      {
        problems.push_back(std::string(key) + ": " + e.what());
      }
    };
    restoreChoice("skin", "the skin", &Impl::applySkin);
    // the eyes chosen, over the presets and under the edits (an iris colour is an edit)
    if(args.contains("eyes") && (args["eyes"].is_string() || args["eyes"].is_null()))
    {
      const std::string rel = args["eyes"].is_string() ? args["eyes"].get<std::string>() : std::string();
      try
      {
        std::shared_ptr<const fv::dson::Doc> doc;
        if(!rel.empty() && !(doc = fig.store().load(rel)))
          problems.push_back("the eyes " + fileStem(rel) + " aren't in the content library");
        else
          applyEyes(*prop, rel, doc ? &doc->json : nullptr, true);
      }
      catch(const std::exception& e)
      {
        problems.push_back(std::string("eyes: ") + e.what());
      }
    }
    restoreChoice("makeup", "the makeup", &Impl::applyMakeup);
    // each worn item's colour, over the presets and under the edits (a hair's recolour is an edit)
    for(const Json& w : args.value("worn", Json::array()))
    {
      if(!w.is_object() || !w.contains("color") || !(w["color"].is_string() || w["color"].is_null()))
        continue;
      const std::string rel = w["color"].is_string() ? w["color"].get<std::string>() : std::string();
      for(const fv::WornItem& i : fig.items)
        if(relKey(i.rel) == relKey(w.value("rel", "")) && (w.value("label", "").empty() || i.label == w.value("label", "")))
          try
          {
            std::shared_ptr<const fv::dson::Doc> doc;
            if(!rel.empty() && !(doc = fig.store().load(rel)))
              problems.push_back("the colour " + fileStem(rel) + " of " + i.label + " isn't in the content library");
            else
              applyItemColor(*prop, {i.id}, rel, doc ? &doc->json : nullptr, true);
          }
          catch(const std::exception& e)
          {
            problems.push_back(i.label + ": " + e.what());
          }
    }
    if(prop->materials && args.contains("materials") && args["materials"].is_object())
      for(const std::string& p : prop->materials->restoreEdits(args["materials"]))
        problems.push_back(p);
    // dForce on each item as the file set it; the items are worn by now
    bool clothChanged = false;
    for(const Json& w : args.value("worn", Json::array()))
    {
      if(!w.is_object() || !w.contains("cloth") || !w["cloth"].is_object())
        continue;
      const Json&         c = w["cloth"];
      std::optional<bool> on;
      if(c.contains("on") && c["on"].is_boolean())
        on = c["on"].get<bool>();
      for(const fv::WornItem& i : fig.items)
        if(relKey(i.rel) == relKey(w.value("rel", "")) && (w.value("label", "").empty() || i.label == w.value("label", "")))
          if(!fig.setItemCloth(i.id, on, c.value("strength", 1.f)).empty())
            clothChanged = true;
    }
    if(clothChanged)
    {
      if(target->dforce)
        target->dforce->poke();
      publishWardrobe(*target);
    }
    if(target->collider && args.contains("collision") && args["collision"].is_boolean())
      target->collider->setEnabled(args["collision"].get<bool>());
    if(target->dforce && args.contains("dforceMode") && args["dforceMode"].is_string())
    {
      // a mode this build lacks (the GPU solver) falls back to live on the CPU, and the job still ends
      try
      {
        target->dforce->setMode(dforceMode(args["dforceMode"].get<std::string>()));
      }
      catch(const std::exception& e)
      {
        target->dforce->setMode(fv::dforce::Mode::eLive);
        problems.push_back(std::string("dForce: ") + e.what() + "; simulating live on the CPU");
      }
    }
    if(target->dforce && args.contains("dforce") && args["dforce"].is_boolean())
    {
      target->dforce->setEnabled(args["dforce"].get<bool>());
      if(target->collider)
        target->collider->poke();
    }
    for(size_t i = 0; i < problems.size() && i < 5; i++)
      notice(prop->name + ": " + problems[i]);
    if(problems.size() > 5)
      notice(std::to_string(problems.size() - 5) + " more problems restoring " + prop->name);
    job(jobId, {{"state", "done"}});
  };
  // one file can put on several items (an outfit's "Full Load"): each is listed, the file is worn once;
  // the figure's add-ons are put on by its load
  std::vector<WearRequest> requests;
  {
    std::set<std::string> skip;
    for(const fv::dson::PostLoadAddon& a : target->fig->addons)
      skip.insert(relKey(a.file));
    if(worn.is_array())
      for(const Json& w : worn)
      {
        const std::string rel = w.value("rel", "");
        if(!rel.empty() && skip.insert(relKey(rel)).second)
          requests.push_back({rel, "", {}});
      }
  }
  auto wear = [this, target, worn, apply, requests](std::shared_future<void> commitAfter) mutable {
    auto hidden = [target, worn] {
      std::vector<int> ids;
      for(const Json& w : worn)
        if(!w.value("visible", true))
          for(const fv::WornItem& i : target->fig->items)
            if(relKey(i.rel) == relKey(w.value("rel", "")) && (w.value("label", "").empty() || i.label == w.value("label", "")))
            {
              target->fig->setItemVisible(i.id, false);
              ids.push_back(i.id);
            }
      return ids;
    };
    auto hide = [hidden] { hidden(); };
    if(requests.empty())
    {
      // no composition follows: the renderer's scene hides them now, as wardrobe.visible does
      bool graft = false;
      for(int id : hidden())
        if(const fv::WornItem* item = wornItem(*target->fig, id))
        {
          if(!item->hiddenPolys.empty())
            graft = true;
          else
            showItem(*target, item->objects, false);
        }
      if(graft)
        requestCompose("Showing", {}, target.get());
      publishWardrobe(*target);
      apply();
      return;
    }
    startWear(target, std::move(requests), hide, apply, std::move(commitAfter));
  };
  if(target->addonsPending && !requests.empty())
  {
    // the worn items are loaded and bound now, beside the add-ons, and put on after them: the
    // add-ons come first, so the keys are as before and the material steps find their surfaces
    auto addonsOn = std::make_shared<std::promise<void>>();
    target->afterAddons.push_back([addonsOn] { addonsOn->set_value(); });
    wear(addonsOn->get_future().share());
  }
  else if(target->addonsPending)
    target->afterAddons.push_back([wear]() mutable { wear({}); });
  else
    wear({});
  return {{"job", jobId}};
}

std::pair<std::shared_ptr<Loaded>, const fv::pose::PoseBone*> FvHost::Impl::boneAt(int gltf) const
{
  for(const auto& p : props)
  {
    const auto& f = p->figure;
    if(!f || !f->fig->pose)
      continue;
    const auto it = f->sceneOf.find(gltf);
    if(it == f->sceneOf.end())
      continue;
    const auto* pb = f->fig->pose->boneForNode(it->second);
    return pb && !pb->axes.empty() ? std::pair{f, pb} : std::pair{nullptr, nullptr};
  }
  return {nullptr, nullptr};
}

void FvHost::Impl::applyGizmoEdit()
{
  if(!gizmoEdit)
    return;
  const GizmoEdit edit = *std::exchange(gizmoEdit, std::nullopt);
  if(!inScene(edit.figure) || !edit.figure->fig->pose)
    return;
  // nodes, correctives and rebound skins go out with this frame's pump()
  edit.figure->fig->pose->setLocalTransform(edit.bone, edit.rotation, edit.position);
  patchPose(*edit.figure, {edit.bone});
}

void FvHost::Impl::patchPose(const Loaded& f, const std::vector<std::string>& boneIds)
{
  if(!f.fig->pose)
    return;
  Json bones = Json::object();
  for(const std::string& id : boneIds)
    if(const auto* pb = f.fig->pose->get(id))
    {
      Json axes = Json::object();
      for(const auto& a : pb->axes)
        axes[axisKey(a)] = {{"value", a.value}};
      bones[pb->id] = {{"axes", axes}};
    }
  protocol->events().patch("pose/" + f.id, {{"changed", f.fig->pose->changed()}, {"bones", bones}});
}

void FvHost::Impl::applyPreset(const fv::dson::Json&                             doc,
                               const std::string&                                kind,
                               uint64_t                                          jobId,
                               const std::string&                                rel,
                               const std::optional<fv::dson::EnvironmentPreset>& environment,
                               const std::string&                                environmentPath,
                               const std::string&                                target)
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
  const auto prop   = figureProp(target.empty() ? Json::object() : Json{{"prop", target}});
  const auto figure = prop ? prop->figure : nullptr;
  if(kind == "pose preset" || kind == "shape preset" || kind == "property preset")
  {
    if(!figure || !figure->fig->pose)
      notice(name + " is a DAZ " + kind + ". Select a DAZ figure first, then open it.");
    else
    {
      fv::PosePresetResult r = figure->fig->applyPosePreset(doc);
      protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
      // a property preset can also set surfaces ("#materials/<surface>:?<channel>" urls)
      const fv::dson::MaterialPresetResult surfaces = fv::dson::applyMaterialPreset(doc, figure->fig->loaded.scene);
      if(!surfaces.changed.empty())
      {
        if(prop->materials)
          prop->materials->surfacesChanged(surfaces.changed);
        auto& presets = figure->materialPresets;
        presets.erase(std::remove(presets.begin(), presets.end(), rel), presets.end());
        presets.push_back(rel);
        presetChangedEyes(*prop, rel, surfaces.changed);
        r.applied += int(surfaces.changed.size());
      }
      // it showed or hid worn items (a geograft's Apply and Remove presets)
      if(!r.itemsShown.empty())
        requestCompose("Showing", {}, figure.get());
      if(!r.applied)
        notice(name + ": nothing in this " + kind + " matches " + prop->name + ".");
      else if(!r.skipped.empty())
        notice("Applied " + name + ". " + std::to_string(r.skipped.size()) + " of its controls aren't on this figure and were skipped.", "info");
      else
        notice("Applied " + name + " to " + prop->name, "info");
    }
  }
  else if(kind == "material preset")
  {
    if(!figure)
      notice(name + " is a DAZ material preset. Select a DAZ figure first, then open the preset.");
    else if(figure->addonsPending)
    {
      // Genesis 9's eyes, mouth and lashes are add-ons still going on: the preset waits for their surfaces
      figure->afterAddons.push_back([this, doc, kind, jobId, rel, id = prop->id] { applyPreset(doc, kind, jobId, rel, std::nullopt, {}, id); });
      return;
    }
    else if(fv::dson::isEyePreset(doc, figure->fig->loaded.scene))
    {
      // eyes swap cleanly, as the Material tab's chooser does
      try
      {
        applyEyes(*prop, rel, &doc, false);
        notice("Eyes " + name + " on " + prop->name + ".", "info");
      }
      catch(const std::exception& e)
      {
        notice(e.what());
      }
    }
    else
    {
      const fv::dson::MaterialPresetResult r = fv::dson::applyMaterialPreset(doc, figure->fig->loaded.scene);
      // not composed yet, the composition lowers the surfaces as they are now
      if(prop->materials)
        prop->materials->surfacesChanged(r.changed);
      if(r.changed.empty())
        notice(name + ": none of its surfaces match " + prop->name + ".");
      else
      {
        auto& presets = figure->materialPresets;
        presets.erase(std::remove(presets.begin(), presets.end(), rel), presets.end());
        presets.push_back(rel);
        presetChangedEyes(*prop, rel, r.changed);
        notice("Applied material preset " + name + " to " + std::to_string(r.changed.size()) + " surfaces.", "info");
      }
    }
  }
  else
    notice(name + " is a DAZ " + kind + " and isn't supported in the native viewport yet.");
  job(jobId, {{"state", "done"}, {"preset", kind}});
}

// The eyes `rel` on a figure, `doc` its preset (null, with `rel` "", for the character's own): the eye
// surfaces back to the character's own, then the preset's eye part over them, so nothing of the eyes
// before stays. Unless `keepEdits`, the panel's edits of the eye surfaces go too (an iris colour sets
// the iris's mean, so kept it would make every pair look the same). Throws, changing nothing, when
// the figure has no eyes or nothing of the preset lands on them.
Json FvHost::Impl::applyEyes(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&             f     = *prop.figure;
  fv::SceneGraph&     scene = f.fig->loaded.scene;
  const std::set<int> eyes  = fv::dson::eyeSurfaces(scene);
  if(eyes.empty())
    throw std::runtime_error(prop.name + " has no eyes to change.");
  if(doc)
  {
    const std::set<int> targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return eyes.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + prop.name + "'s eyes.");
  }
  const fv::dson::EyesResult r       = fv::dson::chooseEyes(scene, doc, f.eyesOwned);
  std::set<int>              changed = r.changed;
  if(!keepEdits)
  {
    std::set<int> cleared = eyes;
    cleared.insert(f.eyesOwned.begin(), f.eyesOwned.end());
    for(int i : cleared)
      if(i >= 0 && i < int(scene.materials.size()) && !scene.materials[size_t(i)].edits.empty())
      {
        scene.materials[size_t(i)].edits = {};
        changed.insert(i);
      }
  }
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  f.eyes      = rel;
  f.eyesOwned = r.owned;
  f.eyesPreset.clear();
  // the scene file replays it as its eyes, not as a preset
  f.materialPresets.erase(std::remove(f.materialPresets.begin(), f.materialPresets.end(), rel), f.materialPresets.end());
  publishEyes(prop);
  return {{"eyes", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

namespace {

// The surfaces a worn item's meshes use.
std::set<int> itemSurfaces(const fv::SceneGraph& scene, const fv::WornItem& w)
{
  std::set<int> out;
  for(int m : w.meshes)
    if(m >= 0 && m < int(scene.meshes.size()))
      for(int mi : scene.meshes[size_t(m)].materials)
        if(mi >= 0 && mi < int(scene.materials.size()))
          out.insert(mi);
  return out;
}

}  // namespace

// The colour `rel` on worn items, `doc` its preset (null, with `rel` "", for the items as they were
// worn): the items' surfaces back to how they were worn, then the preset's part on them, so nothing
// of the colour before stays and nothing else it sets is touched. Several items go as one (a hair
// worn as a cap, a base and its curls), so a preset that colours only some of them still applies.
// Unless `keepEdits`, the panel's edits of those surfaces go too (a recolour sets the mean, so kept
// it would make every colour look the same). Throws, changing nothing, when nothing of the preset
// lands on the items.
Json FvHost::Impl::applyItemColor(Prop& prop, const std::vector<int>& items, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&                          f     = *prop.figure;
  fv::SceneGraph&                  scene = f.fig->loaded.scene;
  std::vector<const fv::WornItem*> worn;
  std::set<int>                    surfaces, previous;
  for(int item : items)
  {
    const fv::WornItem* w = wornItem(*f.fig, item);
    if(!w)
      throw std::runtime_error(prop.name + " doesn't wear that item any more.");
    worn.push_back(w);
    const std::set<int> own = itemSurfaces(scene, *w);
    surfaces.insert(own.begin(), own.end());
    if(const auto was = f.colors.find(item); was != f.colors.end())
      previous.insert(was->second.owned.begin(), was->second.owned.end());
  }
  const std::string label = worn.empty() ? std::string("The item") : worn.front()->label;
  if(surfaces.empty())
    throw std::runtime_error(label + " has no surfaces to colour.");
  std::set<int> targets;
  if(doc)
  {
    targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return surfaces.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + label + "'s.");
  }
  const fv::dson::PresetChoice r       = fv::dson::choosePreset(scene, surfaces, doc, previous);
  std::set<int>                changed = r.changed;
  if(!keepEdits)
    for(int i : surfaces)
      if(!scene.materials[size_t(i)].edits.empty())
      {
        scene.materials[size_t(i)].edits = {};
        changed.insert(i);
      }
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  // each item owns what the choice changed on its own surfaces
  for(const fv::WornItem* w : worn)
  {
    const std::set<int> own = itemSurfaces(scene, *w);
    std::set<int>       owned;
    std::set_intersection(own.begin(), own.end(), r.owned.begin(), r.owned.end(), std::inserter(owned, owned.end()));
    f.colors[w->id] = {rel, owned};
  }
  // the scene file replays it as the items' colour, not as a preset; one that also set other
  // surfaces (a character's skin and hair) stays a preset for those
  if(std::includes(surfaces.begin(), surfaces.end(), targets.begin(), targets.end()))
    f.materialPresets.erase(std::remove(f.materialPresets.begin(), f.materialPresets.end(), rel), f.materialPresets.end());
  publishWardrobe(f);
  return {{"color", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

// wardrobe.color {prop, item | [item...], rel}: the preset is read on a worker and applied on the
// application thread. The job fails, changing nothing, for an item the figure no longer wears or a
// preset none of whose surfaces are the items'.
Json FvHost::Impl::wardrobeColor(const Json& args)
{
  if(!args.contains("rel") || !(args["rel"].is_string() || args["rel"].is_null()))
    throw std::runtime_error("wardrobe.color wants rel: a library path, or null for the item as it was worn");
  auto figure = figureArg(args);
  if(!figure)
    throw std::runtime_error("no such figure");
  const std::vector<int> items = itemIds(args);
  if(items.empty())
    throw std::runtime_error("wardrobe.color wants an item");
  for(int item : items)
    if(!wornItem(*figure->fig, item))
      throw std::runtime_error("no such item");
  const fv::WornItem* w     = wornItem(*figure->fig, items.front());
  const std::string   rel   = args["rel"].is_string() ? args["rel"].get<std::string>() : std::string();
  const uint64_t      jobId = ++jobs;
  const std::string   label = w->label + ": " + (rel.empty() ? std::string("its own colour") : fileStem(rel));
  job(jobId, {{"label", label}, {"state", "running"}});
  const auto fail  = [this, jobId, label](const std::string& why) { job(jobId, {{"label", label}, {"state", "failed"}, {"error", why}}); };
  const auto apply = [this, figure, items, rel, jobId, label, fail](const std::optional<fv::dson::Json>& doc) {
    const auto p = findProp(figure->id);
    if(!p || p->figure != figure)
      return fail("the figure left the scene");
    try
    {
      Json done     = applyItemColor(*p, items, rel, doc ? &*doc : nullptr, false);
      done["label"] = label;
      done["state"] = "done";
      job(jobId, done);
    }
    catch(const std::exception& e)
    {
      fail(e.what());
    }
  };
  if(rel.empty())
  {
    apply(std::nullopt);
    return {{"job", jobId}};
  }
  loaders.spawn([this, figure, rel, apply, fail] {
    try
    {
      auto doc = figure->fig->store().load(rel);
      if(!doc)
        throw std::runtime_error(fileStem(rel) + " isn't in the content library.");
      const auto preset = fv::dson::describePreset(doc->json);
      if(!preset || preset->kind != "material preset")
        throw std::runtime_error(fileStem(rel) + " is not a material preset.");
      post([apply, body = doc->json] { apply(body); });
    }
    catch(const std::exception& e)
    {
      fail(e.what());
    }
  });
  return {{"job", jobId}};
}

void FvHost::Impl::presetChangedEyes(Prop& prop, const std::string& rel, const std::set<int>& changed)
{
  Loaded& f = *prop.figure;
  // a worn item the preset reached has it for its colour now: undo and a scene file put that back
  bool colors = false;
  for(const fv::WornItem& w : f.fig->items)
  {
    const std::set<int> surfaces = itemSurfaces(f.fig->loaded.scene, w);
    std::set<int>       owned;
    std::set_intersection(surfaces.begin(), surfaces.end(), changed.begin(), changed.end(), std::inserter(owned, owned.end()));
    if(owned.empty())
      continue;
    f.colors[w.id] = {rel, owned};
    colors         = true;
  }
  if(colors)
    publishWardrobe(f);
  const std::set<int> skin = fv::dson::skinSurfaces(f.fig->loaded.scene, wornSurfaces(f));
  if(std::any_of(changed.begin(), changed.end(), [&](int i) { return skin.count(i) > 0; }))
  {
    f.skin.reset();
    f.skinOwned.clear();
    f.skinPreset = rel;
    publishSkin(prop);
  }
  const std::set<int> eyes = fv::dson::eyeSurfaces(f.fig->loaded.scene);
  if(std::none_of(changed.begin(), changed.end(), [&](int i) { return eyes.count(i) > 0; }))
    return;
  f.eyes.reset();
  f.eyesOwned.clear();
  f.eyesPreset = rel;
  publishEyes(prop);
}

// "materials/<prop>".eyes: {rel, preset, own, ready, surfaces} (fv_host.hpp material.eyes)
void FvHost::Impl::publishEyes(Prop& p)
{
  if(!p.figure || !p.materials)
    return;
  const Loaded& f   = *p.figure;
  Json          own = Json::array();
  for(const fv::dson::PostLoadAddon& a : f.fig->addons)
    for(const std::string& rel : a.presets)
      if(std::find(own.begin(), own.end(), Json(rel)) == own.end())
        own.push_back(rel);
  Json surfaces = Json::array();
  for(int i : fv::dson::eyeSurfaces(f.fig->loaded.scene))
    surfaces.push_back("m:" + std::to_string(i));
  p.materials->setEyes({{"rel", f.eyes && !f.eyes->empty() ? Json(*f.eyes) : Json(nullptr)},
                        {"preset", f.eyesPreset.empty() ? Json(nullptr) : Json(f.eyesPreset)},
                        {"own", own},
                        {"ready", !f.addonsPending},
                        {"surfaces", surfaces}});
}

// A preset chosen for part of a figure (material.eyes, material.skin, material.makeup) {prop, rel}:
// read on a worker, and applied by `apply` on the application thread once the figure's add-ons
// (Genesis 9's eyes and mouth) are on. The job fails, changing nothing, for a prop that is not a
// figure, or when `apply` throws.
Json FvHost::Impl::choiceJob(const Json& args, const std::string& command, const std::string& part, ChoiceApply apply)
{
  if(!args.contains("rel") || !(args["rel"].is_string() || args["rel"].is_null()))
    throw std::runtime_error(command + " wants rel: a library path, or null for the character's own");
  const std::string rel   = args["rel"].is_string() ? args["rel"].get<std::string>() : std::string();
  const uint64_t    jobId = ++jobs;
  const std::string label = part + ": " + (rel.empty() ? std::string("the character's own") : fileStem(rel));
  job(jobId, {{"label", label}, {"state", "running"}});
  const auto fail = [this, jobId, label](const std::string& why) { job(jobId, {{"label", label}, {"state", "failed"}, {"error", why}}); };
  const auto prop = figureProp(args);
  if(!prop)
  {
    fail("Select a figure first.");
    return {{"job", jobId}};
  }
  std::shared_ptr<Loaded> target = prop->figure;
  // on the application thread, the add-ons on
  const auto done = [this, target, rel, jobId, label, fail, apply](const std::optional<fv::dson::Json>& doc) {
    const auto p = findProp(target->id);
    if(!p || p->figure != target)
      return fail("the figure left the scene");
    try
    {
      Json out     = apply(*p, rel, doc ? &*doc : nullptr);
      out["label"] = label;
      out["state"] = "done";
      job(jobId, out);
    }
    catch(const std::exception& e)
    {
      fail(e.what());
    }
  };
  auto choose = [this, target, rel, done, fail] {
    if(rel.empty())
      return done(std::nullopt);
    loaders.spawn([this, target, rel, done, fail] {
      try
      {
        auto doc = target->fig->store().load(rel);
        if(!doc)
          throw std::runtime_error(fileStem(rel) + " isn't in the content library.");
        const auto preset = fv::dson::describePreset(doc->json);
        if(!preset || preset->kind != "material preset")
          throw std::runtime_error(fileStem(rel) + " is not a material preset.");
        post([done, body = doc->json] { done(body); });
      }
      catch(const std::exception& e)
      {
        fail(e.what());
      }
    });
  };
  if(target->addonsPending)
    target->afterAddons.push_back(std::move(choose));
  else
    choose();
  return {{"job", jobId}};
}

// material.eyes {prop, rel}: fails for a figure with no eyes, or a preset none of whose surfaces are its eyes.
Json FvHost::Impl::materialEyes(const Json& args)
{
  return choiceJob(args, "material.eyes", "Eyes", [this](Prop& p, const std::string& rel, const fv::dson::Json* doc) { return applyEyes(p, rel, doc, false); });
}

// material.skin {prop, rel}: fails for a preset none of whose surfaces are the character's skin.
Json FvHost::Impl::materialSkin(const Json& args)
{
  return choiceJob(args, "material.skin", "Skin", [this](Prop& p, const std::string& rel, const fv::dson::Json* doc) { return applySkin(p, rel, doc, false); });
}

// material.makeup {prop, rel}: as material.skin.
Json FvHost::Impl::materialMakeup(const Json& args)
{
  return choiceJob(args, "material.makeup", "Makeup", [this](Prop& p, const std::string& rel, const fv::dson::Json* doc) { return applyMakeup(p, rel, doc, false); });
}

namespace {

// A colour the panel gave a surface: what a skin or makeup choice takes away, as a kept tone would
// make every skin the same. Roughness and the like stay. True when there was one.
bool clearColours(fv::mat::Edits& e, bool makeupOnly)
{
  const bool had = e.makeupColor || e.makeupWeight || (!makeupOnly && (e.color || e.tone || e.toneStrength || e.recolor || e.recolorStrength));
  e.makeupColor.reset();
  e.makeupWeight.reset();
  if(!makeupOnly)
  {
    e.color.reset();
    e.tone.reset();
    e.toneStrength.reset();
    e.recolor.reset();
    e.recolorStrength.reset();
  }
  return had;
}

}  // namespace

// The surfaces of a figure's worn clothing, props and hair, which its skin leaves alone; a hair's
// brows and lashes are the character's.
std::set<int> FvHost::Impl::wornSurfaces(const Loaded& f)
{
  const fv::SceneGraph& scene = f.fig->loaded.scene;
  std::set<int>         out;
  for(const fv::WornItem& w : f.fig->items)
  {
    if(w.kind == "Anatomy")
      continue;
    for(int i : itemSurfaces(scene, w))
    {
      const fv::Material& m = scene.materials[size_t(i)];
      std::string         name = m.name;
      std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
      if(w.kind == "Hair" && (m.surface.kind == fv::mat::Kind::eLashes || name.find("brow") != std::string::npos))
        continue;
      out.insert(i);
    }
  }
  return out;
}

// The skin `rel` on a figure, `doc` its preset (null, with `rel` "", for the character's own): the
// skin surfaces back to the character's own, then the preset's part on them, so nothing of the skin
// before stays; the makeup laid on it goes with it. Unless `keepEdits`, the panel's colours on them
// go too (a tone sets the skin's mean). Throws, changing nothing, when nothing of the preset lands
// on the skin.
Json FvHost::Impl::applySkin(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&             f        = *prop.figure;
  fv::SceneGraph&     scene    = f.fig->loaded.scene;
  const std::set<int> surfaces = fv::dson::skinSurfaces(scene, wornSurfaces(f));
  if(surfaces.empty())
    throw std::runtime_error(prop.name + " has no skin to change.");
  if(doc)
  {
    const std::set<int> targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return surfaces.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + prop.name + "'s skin.");
  }
  const fv::dson::PresetChoice r       = fv::dson::choosePreset(scene, surfaces, doc, f.skinOwned);
  std::set<int>                changed = r.changed;
  if(!keepEdits)
  {
    std::set<int> cleared = surfaces;
    cleared.insert(f.skinOwned.begin(), f.skinOwned.end());
    for(int i : cleared)
      if(i >= 0 && i < int(scene.materials.size()) && clearColours(scene.materials[size_t(i)].edits, false))
        changed.insert(i);
  }
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  f.skin      = rel;
  f.skinOwned = r.owned;
  f.skinPreset.clear();
  // back to the baseline, the makeup is the skin's own
  f.makeup.reset();
  f.makeupLayer.clear();
  f.materialPresets.erase(std::remove(f.materialPresets.begin(), f.materialPresets.end(), rel), f.materialPresets.end());
  publishSkin(prop);
  return {{"skin", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

// The makeup `rel` over the skin on now (null `doc`, `rel` "": the skin's own makeup): the channels
// the last makeup set go back, then the preset's part on the skin goes over. Unless `keepEdits`, the
// panel's makeup colours and weights go too. Throws, changing nothing, when nothing of the preset
// lands on the skin.
Json FvHost::Impl::applyMakeup(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits)
{
  Loaded&             f        = *prop.figure;
  fv::SceneGraph&     scene    = f.fig->loaded.scene;
  const std::set<int> surfaces = fv::dson::skinSurfaces(scene, wornSurfaces(f));
  if(doc)
  {
    const std::set<int> targets = fv::dson::materialPresetTargets(*doc, scene);
    if(std::none_of(targets.begin(), targets.end(), [&](int i) { return surfaces.count(i) > 0; }))
      throw std::runtime_error("None of " + fileStem(rel) + "'s surfaces are " + prop.name + "'s skin.");
  }
  const fv::dson::LayerChoice r       = fv::dson::chooseLayer(scene, surfaces, doc, f.makeupLayer);
  std::set<int>               changed = r.changed;
  if(!keepEdits)
    for(int i : surfaces)
      if(clearColours(scene.materials[size_t(i)].edits, true))
        changed.insert(i);
  if(prop.materials)
    prop.materials->surfacesChanged(changed);
  f.makeup      = rel;
  f.makeupLayer = r.layer;
  f.materialPresets.erase(std::remove(f.materialPresets.begin(), f.materialPresets.end(), rel), f.materialPresets.end());
  publishSkin(prop);
  return {{"makeup", rel.empty() ? Json(nullptr) : Json(rel)}, {"changed", changed.size()}};
}

// "materials/<prop>".skin: {rel, preset, makeup, ready, surfaces} (fv_host.hpp material.skin)
void FvHost::Impl::publishSkin(Prop& p)
{
  if(!p.figure || !p.materials)
    return;
  const Loaded& f        = *p.figure;
  Json          surfaces = Json::array();
  for(int i : fv::dson::skinSurfaces(f.fig->loaded.scene, wornSurfaces(f)))
    surfaces.push_back("m:" + std::to_string(i));
  p.materials->setSkin({{"rel", f.skin && !f.skin->empty() ? Json(*f.skin) : Json(nullptr)},
                        {"preset", f.skinPreset.empty() ? Json(nullptr) : Json(f.skinPreset)},
                        {"makeup", f.makeup && !f.makeup->empty() ? Json(*f.makeup) : Json(nullptr)},
                        {"ready", !f.addonsPending},
                        {"surfaces", surfaces}});
}

Json FvHost::Impl::lightingTopic() const
{
  Json t           = fv::lighting::toJson(lighting);
  t["env"]["name"]    = envName;
  t["env"]["size"]    = {envSize.x, envSize.y};
  t["env"]["loading"] = envLoading;
  // where the HDR's own sun is, as the panel's sun angles at the current rotation
  if(envSun && lighting.env.source == "hdr")
  {
    const glm::dvec2 a = fv::lighting::environmentSunAngles(*envSun, lighting.env.rotation);
    t["env"]["sun"]    = {{"azimuth", std::round(a.x * 10.0) / 10.0}, {"elevation", std::round(a.y * 10.0) / 10.0}};
  }
  else
    t["env"]["sun"] = nullptr;
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
  fv::timing::Scope t("lighting.set", 20);
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
  {
    fv::timing::Scope t("requestEnvironment", 20);
    requestEnvironment();
  }

  Json params                     = fv::lighting::rendererParams(lighting, envSize);
  params["infinitePlaneDistance"] = lighting.ground.autoHeight ? double(autoGround) : lighting.ground.height;
  if(const auto still = appliedParams.find("ptFireflyClampStill"); still != appliedParams.end() && still->is_number())
    params["ptFireflyClampStill"] = fv::lighting::stillFireflyClamp(params["ptFireflyClampStill"].get<double>(), still->get<double>());
  Json changed                    = Json::object();
  for(auto it = params.begin(); it != params.end(); ++it)
    if(!appliedParams.contains(it.key()) || appliedParams[it.key()] != it.value())
      changed[it.key()] = it.value();
  if(!changed.empty())
  {
    std::string keys;
    for(auto it = changed.begin(); it != changed.end(); ++it)
      keys += (keys.empty() ? "" : ",") + it.key();
    fv::timing::Scope t("setParams " + keys, 20);
    setParams(changed);
    appliedParams.update(changed);
  }
  {
    fv::timing::Scope t("applySun", 20);
    applySun(everything);
  }
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
      envSun.reset();
      envName   = kStudioName;
      envSize   = {kStudioWidth, kStudioHeight};
      appliedParams.erase("ptFireflyClamp");  // an environment load recalibrates it; applyLighting sends ours
    }
    return;
  }
  const std::string path  = e.path;
  const std::string label = "Loading " + utf8Of(pathOf(path).filename());
  envLoading              = true;
  envJob                  = ++jobs;
  job(envJob, {{"label", label}, {"state", "running"}});
  loaders.spawn([this, path, want, req, jobId = envJob, label] {
    auto image = std::make_shared<EnvImage>(loadEnvImage(pathOf(path)));
    // before the renderer takes the pixels: it writes the sampling pdf into alpha
    const std::optional<glm::vec3> sun =
        image->valid() ? fv::lighting::environmentSun(image->pixels.data(), image->size.width, image->size.height) : std::nullopt;
    post([this, image, path, want, req, jobId, label, sun] {
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
      envSun    = sun;
      envName   = utf8Of(pathOf(path).filename());
      envSize   = size;
      job(jobId, {{"label", label}, {"state", "done"}});
      appliedParams.erase("envSystem");
      appliedParams.erase("ptFireflyClamp");  // the load recalibrated it
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

// The "Sun" light and its node as the current state has them: directional, marked as the sky's sun
// so a physical sky follows it.
void FvHost::Impl::fillSun(tinygltf::Light& light, tinygltf::Node& node, int lightIndex)
{
  const fv::lighting::Sun& s     = lighting.sun;
  const glm::vec3          color = fv::lighting::linearColor(s.color);
  light.name      = "Sun";
  light.type      = "directional";
  light.color     = {color.x, color.y, color.z};
  light.intensity = s.enabled ? s.intensity : 0.0;
  tinygltf::Value::Object lightExtras;
  lightExtras["radius"] = tinygltf::Value(fv::lighting::sunRadiusKm(s.softness));
  light.extras          = tinygltf::Value(lightExtras);

  node.name         = "Sun";
  node.light        = lightIndex;
  const glm::quat r = fv::lighting::sunRotation(s.azimuth, s.elevation);
  node.rotation     = {r.x, r.y, r.z, r.w};
  tinygltf::Value::Object nodeExtras = node.extras.IsObject() ? node.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
  nodeExtras["fvId"]                 = tinygltf::Value(std::string("l:sun"));
  node.extras                        = tinygltf::Value(nodeExtras);
  gltf_environment_sky::setSkySunMarker(node, gltf_environment_sky::kSkySunOwnerScene);
}

// The sun on a root node of a model about to be loaded, written with the current state, so the load
// shows it as it is.
void FvHost::Impl::addSun(tinygltf::Model& model)
{
  tinygltf::Light light;
  tinygltf::Node  node;
  fillSun(light, node, int(model.lights.size()));
  model.lights.push_back(light);
  sunLight = int(model.lights.size()) - 1;
  model.nodes.push_back(node);
  sunNode = int(model.nodes.size()) - 1;

  if(model.scenes.empty())
    model.scenes.emplace_back();
  const int scene = model.defaultScene >= 0 && model.defaultScene < int(model.scenes.size()) ? model.defaultScene : 0;
  model.scenes[scene].nodes.push_back(sunNode);
  if(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), "KHR_lights_punctual") == model.extensionsUsed.end())
    model.extensionsUsed.push_back("KHR_lights_punctual");
  const fv::lighting::Sun& s     = lighting.sun;
  const glm::vec3          color = fv::lighting::linearColor(s.color);
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
  // the figure library's phase timers (loads, wears, glTF builds) go to the log
  fv::timing::setSink([](const std::string& line) { LOGI("%s\n", line.c_str()); });
  // what this build has beyond the protocol, so the app offers only that
#ifdef FV_DFORCE_GPU
  s.protocol->setFeatures({{"dforceGpu", true}});
#else
  s.protocol->setFeatures({{"dforceGpu", false}});
#endif

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
    st.originalTextures = args.value("originalTextures", st.originalTextures);
    return Json{{"ok", true}};
  });
  // load works out on a worker what the file is, and comes to the application thread for the rest
  cmds.add("load", [impl](const Json& args) { return impl->load(args); });
  cmds.add("prop.add", [impl](const Json& args) {
    if(args.contains("path") || args.contains("rel"))
    {
      Json asProp  = args;
      asProp["as"] = "prop";
      return impl->load(asProp);
    }
    return impl->onApp([impl, args] { return impl->propAdd(args); });
  });
  cmds.add("prop.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->propSet(args); }); });
  cmds.add("prop.remove", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      auto p = impl->findProp(args.at("prop").get<std::string>());
      if(!p)
        throw std::runtime_error("no such prop");
      impl->removeProp(p);
      impl->publishScene();
      impl->requestCompose("Removing " + p->name);
      return Json{{"ok", true}};
    });
  });
  cmds.add("prop.duplicate", [impl](const Json& args) {
    // a figure loads again, which waits on workers: answered with its job
    return impl->onApp([impl, args] { return impl->propDuplicate(args); });
  });
  cmds.add("scene.clear", [impl](const Json&) { return impl->onApp([impl] { return impl->sceneClear(); }); });
  // every prop's place at once, as a scene file opens: one composition, not one per prop that moves
  cmds.add("scene.order", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      std::vector<std::shared_ptr<Prop>> next;
      for(const Json& id : args.at("order"))
        if(id.is_string())
          if(auto p = impl->findProp(id.get<std::string>()); p && std::find(next.begin(), next.end(), p) == next.end())
            next.push_back(p);
      // what the list leaves out keeps its order, after what it names
      for(const auto& p : impl->props)
        if(std::find(next.begin(), next.end(), p) == next.end())
          next.push_back(p);
      if(next != impl->props)
      {
        impl->props = std::move(next);
        impl->requestCompose("Reordering");
        impl->publishScene();
      }
      return Json{{"ok", true}};
    });
  });
  cmds.add("view.use", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      fv::timing::Scope t("view.use", 20);
      return impl->viewUse(args);
    });
  });
  for(const char* name : {"select", "play", "pause", "stop", "seek", "speed", "loop"})
    cmds.add(std::string("anim.") + name, [impl, cmd = std::string(name)](const Json& args) {
      return impl->onApp([impl, cmd, args] { return impl->animationCommand(cmd, args); });
    });
  cmds.add("export.glb", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->exportGlb(args); }); });
  cmds.add("scene.state", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->sceneState(args); }); });
  cmds.add("scene.restore", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->sceneRestore(args); }); });
  cmds.add("wear", [impl](const Json& args) { return impl->wear(args); });
  cmds.add("wardrobe.remove", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->wardrobeRemove(args); }); });
  cmds.add("wardrobe.visible", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->wardrobeVisible(args); }); });
  cmds.add("wardrobe.collision", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->wardrobeCollision(args); }); });
  cmds.add("wardrobe.cloth", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->wardrobeCloth(args); }); });
  cmds.add("dforce.enable", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->dforceCommand("enable", args); }); });
  cmds.add("dforce.mode", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->dforceCommand("mode", args); }); });
  cmds.add("dforce.resimulate", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->dforceCommand("resimulate", args); }); });
  const auto materialsOf = [impl](const Json& args) -> FvMaterials& {
    auto p = impl->propArg(args);
    if(!p || !p->materials)
      throw std::runtime_error("Select a figure or a model: its surfaces are edited here.");
    return *p->materials;
  };
  cmds.add("material.set", [impl, materialsOf](const Json& args) { return impl->onApp([materialsOf, args] { return materialsOf(args).set(args); }); });
  cmds.add("material.reset", [impl, materialsOf](const Json& args) { return impl->onApp([materialsOf, args] { return materialsOf(args).reset(args); }); });
  cmds.add("material.eyes", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materialEyes(args); }); });
  cmds.add("wardrobe.color", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->wardrobeColor(args); }); });
  cmds.add("material.skin", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materialSkin(args); }); });
  cmds.add("material.makeup", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materialMakeup(args); }); });
  cmds.add("pose.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->poseSet(args); }); });
  cmds.add("pose.eyes", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      const auto p = impl->figureProp(args);
      if(!p || !p->figure->fig->pose)
        throw std::runtime_error("no such figure");
      auto manip = impl->renderer->getCameraManipulator();
      if(!manip)
        throw std::runtime_error("no camera");
      // the eye in the figure's own space, under its prop root
      const glm::vec3 eye = glm::vec3(glm::inverse(impl->worldOf(*p)) * glm::vec4(glm::vec3(manip->getEye()), 1.f));
      const int       n   = p->figure->fig->pose->aimEyes({eye.x, eye.y, eye.z});
      std::vector<std::string> bones;
      for(const auto& pb : p->figure->fig->pose->bones())
        if(fv::pose::isEyeBone(pb.name))
          bones.push_back(pb.id);
      impl->patchPose(*p->figure, bones);
      return Json{{"aimed", n}};
    });
  });
  cmds.add("pose.reset", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      auto figure = impl->figureArg(args);
      if(!figure || !figure->fig->pose)
        throw std::runtime_error("no such figure");
      figure->fig->pose->reset();
      impl->protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
      return Json{{"changed", 0}};
    });
  });
  // answered at once: the values merge with whatever else arrived this frame and apply in onUpdate
  cmds.add("morph.set", [impl](const Json& args) {
    std::lock_guard   lock(impl->morphMutex);
    const std::string fig = args.contains("prop") ? args.value("prop", "") : args.value("figure", "");
    if(!impl->morphFigures.count(fig))
      throw std::runtime_error("no such figure");
    const Json* values = args.contains("values") ? &args["values"] : nullptr;
    if(!values || !values->is_object())
      throw std::runtime_error("morph.set wants values: {name: value}");
    for(auto it = values->begin(); it != values->end(); ++it)
      if(it.value().is_number())
        impl->pendingMorphs[fig][it.key()] = it.value().get<double>();
    return Json{{"ok", true}};
  });
  cmds.add("morph.reset", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      auto figure = impl->figureArg(args);
      if(!figure || !figure->fig->morphs)
        throw std::runtime_error("no such figure");
      {
        std::lock_guard lock(impl->morphMutex);
        impl->pendingMorphs.erase(figure->id);
      }
      figure->fig->morphs->resetAll();
      return Json{{"ok", true}};
    });
  });
  cmds.add("render.mode", [impl](const Json& args) {
    const std::string mode = args.at("mode").get<std::string>();
    if(mode != "raster" && mode != "rtx")
      throw std::runtime_error("mode is raster or rtx");
    // RTX skin: a volume of measured human skin (the default), the file's Iray volume, or
    // thin-walled scattering at the surface
    const std::string skin = args.value("skin", std::string("measured"));
    if(skin != "volume" && skin != "thin" && skin != "measured")
      throw std::runtime_error("skin is volume, thin or measured");
    const fv::mat::SkinMode rtxSkin = skin == "thin"       ? fv::mat::SkinMode::eThin :
                                      skin == "measured" ? fv::mat::SkinMode::eMeasured :
                                                           fv::mat::SkinMode::eVolume;
    return impl->onApp([impl, mode, rtxSkin] {
      impl->renderer->setRenderSystem(mode == "rtx" ? RenderingMode::ePathtracer : RenderingMode::eRasterizer);
      impl->skinMode = mode == "rtx" ? rtxSkin : fv::mat::SkinMode::eRaster;
      for(const auto& p : impl->props)
        if(p->figure && p->materials)
          p->materials->setSkinMode(impl->skinMode);
      impl->protocol->events().snapshot("render", {{"mode", mode}});
      return Json{{"mode", mode}};
    });
  });
  // whether the renderer is still refining a still, for an image export that waits for it
  cmds.add("render.status", [impl](const Json&) {
    return impl->onApp([impl] {
      const VkExtent2D size = impl->renderer->renderSize();
      return Json{{"refining", impl->renderer->isRefiningStill()},
                  {"still", impl->renderer->isStill()},
                  {"converged", impl->renderer->isStillConverged()},
                  {"complete", impl->renderer->accumulationComplete()},
                  {"rtx", impl->renderer->renderSystem() == RenderingMode::ePathtracer},
                  // the frame's size: a /size lands before the next frame, and until then the rest is the old frame's
                  {"width", size.width},
                  {"height", size.height}};
    });
  });
  // GPU memory, for the Export dialog: the budget (VK_EXT_memory_budget; `known` false without it)
  // and what a pixel of the frame costs in the mode drawn, so a size is judged before it is asked for
  cmds.add("render.budget", [impl](const Json&) {
    return impl->onApp([impl] {
      const auto       b    = impl->renderer->deviceMemoryBudget();
      const VkExtent2D size = impl->renderer->renderSize();
      return Json{{"known", b.known},   {"budgetBytes", b.budget},
                  {"usageBytes", b.usage}, {"bytesPerPixel", impl->renderer->frameBytesPerPixel()},
                  {"width", size.width},   {"height", size.height}};
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
  // the app's own timings (an open's phases, measured where the user waits), into the timing log
  cmds.add("app.timing", [](const Json& args) {
    if(args.contains("lines") && args["lines"].is_array())
      for(const Json& l : args["lines"])
        if(l.is_string())
          fv::timing::line("app: " + l.get<std::string>());
    return Json::object();
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
  s.view->onHelperPick([impl](const glm::vec2& px) { return impl->helperAt(px); });
  // camera.set edits the lens of the view in use: the camera looked through keeps it, else the free view
  s.view->onLensChange([impl](const FvView::Lens& lens) {
    if(const auto p = impl->findProp(impl->viewCamera))
    {
      p->camera.lens = lens;
      impl->publishScene();
    }
    else
      impl->freeView.lens = lens;
  });
  s.protocol->setInputSink([impl](const Json& events) {
    std::lock_guard lock(impl->inputMutex);
    for(const auto& e : events)
      impl->input.push_back(e);
  });
  // the stream stops before this element detaches: its server joins the HTTP threads, which must not wait on this queue
  s.protocol->setStopHook([impl] { impl->stopAccepting(); });
  // The gizmo moves, turns and scales a selected prop (object picking), and poses a figure's bones
  // through its rig (bone picking), so limits and formulas apply.
  auto hasKind = [](const fv::pose::PoseBone& pb, fv::pose::ChannelKind kind) {
    return std::any_of(pb.axes.begin(), pb.axes.end(), [kind](const fv::pose::Axis& a) { return a.kind == kind; });
  };
  renderer->setGizmoHooks({
      .attach =
          [impl, hasKind](int gltf) -> uint32_t {
            if(!impl->view)
              return 0;
            const std::string& pick = impl->view->pickMode();
            if(pick == "object")
            {
              if(!impl->view->propOfRoot(gltf))
                return 0;
              const std::string& t = impl->view->transformMode();
              return t == "rotate" ? TransformHelperVk::ShowRotation : t == "scale" ? TransformHelperVk::ShowScale : TransformHelperVk::ShowTranslation;
            }
            if(pick != "bone")
              return 0;
            const auto [figure, pb] = impl->boneAt(gltf);
            uint32_t flags          = 0;
            if(pb && hasKind(*pb, fv::pose::ChannelKind::eRotation))
              flags |= TransformHelperVk::ShowRotation;
            if(pb && hasKind(*pb, fv::pose::ChannelKind::eTranslation))
              flags |= TransformHelperVk::ShowTranslation;
            return flags;
          },
      .edit =
          [impl, hasKind](int gltf, const glm::vec3& t, const glm::quat& r, const glm::vec3& scale, bool done) {
            if(const FvView::PropNodes* nodes = impl->view ? impl->view->propOfRoot(gltf) : nullptr)
            {
              auto p = impl->findProp(nodes->id);
              if(!p)
                return;
              p->transform = {t, glm::normalize(r), p->kind == PropKind::eCamera ? glm::vec3(1.f) : scale};
              p->euler.reset();
              impl->applyTransform(*p);
              impl->movedProps.insert(p->id);
              if(p->id == impl->viewCamera)
                impl->viewFromCamera(*p);
              return;
            }
            // the release repeats the last move's transform; a click on a handle with no drag has only the release
            const auto [figure, pb] = done ? std::pair<std::shared_ptr<Loaded>, const fv::pose::PoseBone*>{} : impl->boneAt(gltf);
            if(!pb)
              return;
            impl->gizmoEdit = Impl::GizmoEdit{figure, pb->id, fv::Quat{r.x, r.y, r.z, r.w},
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
  m_impl->publishScene();
}

void FvHost::onDetach()
{
  Impl& s = *m_impl;
  s.stopAccepting();
  s.protocol->setStopHook({});
  s.loadGeneration++;  // loads still running drop their result
  textureCalls().stopAll();  // and don't wait out a cook
  s.loaders.joinAll();
  s.queue.clear();
  s.renderer->setGizmoHooks({});
  s.renderer->setPickObserver({});
  if(s.view)
    s.view->setProps({});
  for(const auto& p : s.props)
    if(p->materials)
      p->materials->detach();
  s.props.clear();
  s.pendingCompose.reset();
  {
    std::lock_guard morphLock(s.morphMutex);
    s.morphFigures.clear();
    s.pendingMorphs.clear();
  }
}

void FvHost::onUpdate()
{
  Impl& s = *m_impl;
  if(s.lastUpdateEnd && fv::timing::enabled())
    fv::timing::report("frame: renderer between updates", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *s.lastUpdateEnd).count(), 100);
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
  // a phase of the frame that holds the application thread up shows in the timing log
  std::optional<fv::timing::Scope> phase;
  const auto timePhase = [&phase](const char* name) {
    phase.reset();
    phase.emplace(name, 20);
  };
  timePhase("frame: queued work");
  for(auto& f : work)
  {
    try
    {
      fv::timing::Scope task("frame: one queued task", 100);
      f();
    }
    catch(const std::exception& e)
    {
      s.notice(e.what(), "error");
    }
  }
  timePhase("frame: compose");
  // what this frame's work changed in the scene's structure, composed once
  if(s.pendingCompose)
  {
    try
    {
      s.compose();
    }
    catch(const std::exception& e)
    {
      s.notice(std::string("The scene could not be built: ") + e.what(), "error");
    }
  }
  timePhase("frame: input and morphs");
  // input first: a gizmo drag poses or moves this frame, and what the gizmo didn't take moves the camera
  std::vector<Json> events;
  {
    std::lock_guard lock(s.inputMutex);
    events.swap(s.input);
  }
  if(!events.empty())
  {
    s.normalizeInput(events);
    s.applyInput(s.view ? s.view->onInput(events) : events);
  }
  s.applyGizmoEdit();
  // not while a composition settles: the renderer moves the view to a scene camera for a frame or two
  if(!s.viewCamera.empty() && s.frameIn == 0)
    s.cameraFromView();
  s.applyPendingMorphs();
  for(const auto& p : s.props)
  {
    if(!p->figure)
      continue;
    Loaded& f = *p->figure;
    timePhase("frame: figure pump");
    fv::Figure::Changes changes = f.fig->pump();
    // this frame's edits first, then a settle or a collision pass that finished writes its result
    // and is drained on its own: its vertices never hide an edit to the same mesh, and are no
    // reason to run again
    std::vector<int> moved;
    for(const fv::morph::MeshChange& m : changes.meshes)
      moved.push_back(m.mesh);
    const auto drainInto = [&] {
      fv::Figure::Changes more = f.fig->drain();
      changes.nodes.insert(more.nodes.begin(), more.nodes.end());
      changes.meshes.insert(changes.meshes.end(), std::make_move_iterator(more.meshes.begin()), std::make_move_iterator(more.meshes.end()));
      changes.rebound.insert(changes.rebound.end(), more.rebound.begin(), more.rebound.end());
    };
    timePhase("frame: dForce and collision");
    if(f.dforce)
    {
      if(!changes.nodes.empty() || !moved.empty())
        f.dforce->changed(moved, !changes.nodes.empty());
      if(f.dforce->pump())
        drainInto();
      s.publishDforce(f, false);
    }
    if(f.collider)
    {
      if(!changes.nodes.empty() || !moved.empty())
        f.collider->changed(moved, !changes.nodes.empty());
      if(f.collider->pump())
        drainInto();
      if(f.collisionVersion != f.collider->version())
        s.publishWardrobe(f);
    }
    timePhase("frame: figure changes applied");
    if(!changes.nodes.empty())
    {
      const auto t0 = std::chrono::steady_clock::now();
      s.applyNodes(f, changes.nodes);
      s.dialTiming.add(DialTiming::eApplyNodes, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    if(!changes.meshes.empty())
    {
      const auto t0 = std::chrono::steady_clock::now();
      for(const fv::morph::MeshChange& mesh : changes.meshes)
        s.applyMesh(f, mesh);
      s.dialTiming.add(DialTiming::eApplyMesh, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    if(!changes.rebound.empty())
    {
      const auto t0 = std::chrono::steady_clock::now();
      s.applyRebind(f, changes.rebound);
      s.dialTiming.add(DialTiming::eApplyRebind, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    for(const std::string& w : f.fig->takeWarnings())
      s.notice(p->name + ": " + w);
    s.publishBusy(f);
    s.publishMorphs(f);
  }
  timePhase("frame: the rest");
  if(fv::timing::enabled())
    if(const std::string line = s.dialTiming.report(std::chrono::steady_clock::now()); !line.empty())
    {
      LOGI("%s\n", line.c_str());
      for(const std::string& phase : fv::timing::takeSummary())
        LOGI("%s\n", phase.c_str());
    }
  // props the gizmo moved: one patch a frame
  if(!s.movedProps.empty())
  {
    Json moved = Json::object();
    for(const std::string& id : s.movedProps)
      if(const auto p = s.findProp(id))
      {
        const Json now = s.propJson(*p);
        moved[id]      = {{"transform", now["transform"]}};
        if(now.contains("camera"))
          moved[id]["camera"] = now["camera"];
      }
    s.movedProps.clear();
    s.protocol->events().patch("scene", {{"props", moved}});
  }
  s.publishAnimation(false);
  if(s.frameIn > 0 && --s.frameIn == 0)
  {
    if(s.cameraRestore)
    {
      if(auto manip = s.renderer->getCameraManipulator())
        manip->setLookat((*s.cameraRestore)[0], (*s.cameraRestore)[1], (*s.cameraRestore)[2], true);
      s.cameraRestore.reset();
    }
    else
      s.frameCamera();
  }
  s.drawHelpers();
  if(s.view)
    s.view->onFrame();
  phase.reset();
  s.lastUpdateEnd = std::chrono::steady_clock::now();
}
