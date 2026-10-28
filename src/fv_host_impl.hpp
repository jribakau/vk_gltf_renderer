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

#pragma once

//
// FvHost's implementation, shared by its source files (fv_host.cpp, fv_host_support.cpp and the
// fv_host_<area>.cpp files): the host's state, FvHost::Impl, and the helpers more than one of them
// uses, in namespace fvhost. Include it first: httplib has to come before windows.h.
//


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
#include "fv_model_material.hpp"
#include "fv_parameter_filter.hpp"
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

namespace fvhost {


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
    // the installed Iray preset a 3Delight-era item took as it was worn (WearRequest::iray), not a
    // colour chosen: the Wardrobe marks it
    bool automatic = false;
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

struct CookedTexture
{
  std::optional<fv::build::TextureImage> image;
  bool                                   uncooked = false;  // asked cachedOnly, and not cooked yet
};

constexpr const char* kStudioName = "Studio (built-in)";
constexpr uint32_t    kStudioWidth = 512, kStudioHeight = 256;

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

// ---- helpers (fv_host_support.cpp)

const char* kindName(PropKind k);

std::filesystem::path pathOf(const std::string& utf8);

std::string utf8Of(const std::filesystem::path& p);

// encodeURIComponent
std::string encodeComponent(std::string_view s);

// A number as the cooker parses it, five decimals at most.
std::string fmtNumber(double v);

// A library-relative path, its segments encoded (src/lib/ipc.ts encodeRel).
std::string encodeRel(std::string_view rel);

// A DSON image path as the library has it: decoded, forward slashes, no leading slash.
std::string normRel(const std::string& path);

// normRel, compared as the library compares paths: without case
std::string relKey(const std::string& path);

// encodeURI for an absolute path used as an image URI (the renderer percent-decodes URIs).
std::string uriOfPath(const std::string& path);

std::string mimeOf(const std::string& path);

TextureCalls& textureCalls();

// Where a cooked texture is (BC7 KTX2, cutout folded into alpha, rows bottom-up for the DSON UVs
// when `flip`). The texture server cooks it first, which takes seconds per 4K map; `cachedOnly`
// answers at once instead, `uncooked` when it isn't cooked.
CookedTexture cookedTexture(const HostSettings& s, const fv::build::TextureRequest& req, bool flip, bool cachedOnly, bool quiet = false);

// Cooks textures for a later load, on a thread of its own: a figure shows with its source images
// rather than wait for these.
void cookLater(const HostSettings& s, std::vector<fv::build::TextureRequest> requests);

// The source image as it is, for a renderer without a texture server (tests, a standalone run).
std::optional<fv::build::TextureImage> sourceTexture(const fv::dson::Library& lib, const fv::build::TextureRequest& req);

// A texture without waiting for the cooker: the source image as it is, where that shows the right
// thing. A normal map keeps its green (the material tells the renderer to flip it) and loses the
// bump blended onto it. A roughness map (its grey into glTF's channels) and a height map (turned
// into a normal map) are composed here: without them skin loses its sheen and its pores. A cutout
// alone changes the picture too much: it, makeup and an iris colour (and formats the renderer cannot
// read) are cooked, unflipped, and waited for; without a cooker, or with Original textures (only
// what the renderer cannot read is cooked then), they are composed here too (fv_textures.hpp).
std::optional<fv::build::TextureImage> fallbackTexture(const HostSettings& s, const fv::dson::Library& lib, const fv::build::TextureRequest& req, bool cooker);

// Genesis 9's brows ship a grey placeholder surface; DAZ's post-load script (Remove Duplicate
// Eyebrows.dse, encrypted) lays the default brow material over it, which a character's own preset
// may then change. The preset for the brows a wearable holds, or none.
std::string defaultBrowMaterial(const fv::SceneGraph& scene);

// The library folder a .duf sits in: one of the configured roots, else the nearest folder above it
// with a "data" folder (as src/lib/actions.ts finds it).
std::optional<std::pair<std::vector<std::filesystem::path>, std::string>> libraryFor(const HostSettings& s, const std::filesystem::path& file);

std::string axisKey(const fv::pose::Axis& a);

// The pose topic: bones by id, each with its axes by "kind/axis".
Json poseSnapshot(const Loaded& f);

std::string fileStem(const std::string& rel);

// The registered parameters nvmcp can read and write: values, not triggers or custom parsers, and
// none that loads or saves the scene behind the host (fv::hostExposesParameter).
nvmcp::detail::ParameterList exposedParameters(const nvutils::ParameterRegistry* registry);

Json toolJson(const nvmcp::ToolResult& r);

// How a figure is built for the scene: its textures as resolved (cooked, or baked for an edit),
// flipped as they arrive, the skin for the renderer, the surfaces light walks through.
fv::build::BuildOptions buildOptionsOf(const Loaded& f, fv::mat::SkinMode skin);

int itemIdOf(const Json& v);

int itemId(const Json& args);

// `item` as one id or a list of them ("i:3", 3, ["i:3", "i:4"]).
std::vector<int> itemIds(const Json& args);

const fv::WornItem* wornItem(const fv::Figure& fig, int id);

// A dForce mode by its name; "gpu" needs the build to have the GPU solver.
fv::dforce::Mode dforceMode(const std::string& name);

}  // namespace fvhost

using namespace fvhost;

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

  // "<file>:<line>" of a call site, for the timing log (the host is several files)
  static std::string sourceAt(const std::source_location& where)
  {
    const std::string_view file = where.file_name();
    const size_t           cut  = file.find_last_of("/\\");
    return std::string(cut == std::string_view::npos ? file : file.substr(cut + 1)) + ":" + std::to_string(where.line());
  }

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
      queue.emplace_back([task, state, at = sourceAt(where)] {
        int queued = 0;
        if(!state->compare_exchange_strong(queued, 1))
          return;
        // one that holds the application thread up shows in the timing log, by the line that sent it
        const auto start = std::chrono::steady_clock::now();
        (*task)();
        if(fv::timing::enabled())
          fv::timing::report("command run, sent at " + at,
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
  fv::composer::Part partOf(size_t i, const tinygltf::Model* model, bool forFile) const;
  // a model prop's file with its material edits laid over its own materials
  tinygltf::Model editedModel(const Prop& p) const;
  // export.glb {scene: true}: the scene as a file, built fresh from the props
  Json exportScene(const std::string& path);
  bool        ridesOn(const Prop& q, const std::string& id) const;
  void        applyLightShown(const Prop& p);

  void post(std::function<void()> f, std::source_location where = std::source_location::current())
  {
    // one that holds the application thread up shows in the timing log, by the line that posted it
    auto timed = [f = std::move(f), at = sourceAt(where)] {
      const auto start = std::chrono::steady_clock::now();
      f();
      if(fv::timing::enabled())
        fv::timing::report("posted at " + at,
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
  // `iray`: installed Iray presets a 3Delight-era wearable may take as it goes on (WearRequest::iray)
  Json loadFigure(const HostSettings& s, std::vector<std::filesystem::path> roots, std::string rel, const std::string& name, Json source,
                  std::string target, bool asProp, std::function<void(std::shared_ptr<Prop>)> then = {}, std::vector<std::string> prefetch = {},
                  std::vector<std::string> iray = {});
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
    // installed Iray presets a 3Delight-era item may take as it goes on, the one that keeps its look
    // (fv::dson::chooseIrayPreset); recorded as its colour, so "own" takes it off again
    std::vector<std::string> iray;
  };
  Json     wear(const Json& args);
  // What a restore sets on the figure besides its wardrobe, put on before the wear's composition so
  // it is built once (both run on the application thread): `surfaces` after the commits, with the
  // materials held, and `shape` (dials and pose) right before the build.
  struct WearLooks
  {
    std::function<void()> surfaces, shape;
  };
  // `beforeBuild` runs on the application thread once everything is on, before the scene is built
  // again; `after` once it has been. With `commitAfter`, the requests are prepared at once and put
  // on once it is ready (a restore's worn items after the figure's add-ons). `looks`, a restore's.
  uint64_t startWear(std::shared_ptr<Loaded> target, std::vector<WearRequest> requests, std::function<void()> beforeBuild = {}, std::function<void()> after = {},
                     std::shared_future<void> commitAfter = {}, WearLooks looks = {});
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
