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
#include "pipeline_build_scope.hpp"

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
#include <numeric>
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

#include <figure/physics/cloth/cloth.hpp>
#include <figure/physics/contact/contact.hpp>
#include <figure/physics/rain/rain.hpp>
#include <figure/physics/rig/figure_physics.hpp>
#include <figure/physics/flesh/flesh.hpp>
#include <figure/physics/rig/secondary.hpp>
#include <figure/physics/vam/player.hpp>
#include <figure/wardrobe/collider.hpp>
#ifdef FV_PHYSICS_GPU
#include "fv_physics_gpu.hpp"
#endif
#include <figure/dson/eye_materials.hpp>
#include <figure/dson/skin_materials.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/dson/presets.hpp>
#include <figure/figure.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/gltf/export_glb.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/io/system_memory.hpp>
#include <figure/io/timing.hpp>

#include "env_image_loader.hpp"
#include "fv_compose.hpp"
#include "fv_figure_feed.hpp"
#include "fv_prop_feed.hpp"
#include "fv_journal.hpp"
#include "ram_watch.hpp"
#include "fv_lighting.hpp"
#include "fv_materials.hpp"
#include "fv_model_material.hpp"
#include "fv_parameter_filter.hpp"
#include "fv_protocol.hpp"
#include "fv_textures.hpp"
#include "fv_view.hpp"
#include "fv_workers.hpp"
#include "gltf_environment_sky.hpp"
#include "gizmo_lines_vk.hpp"
#include "host_renderer.hpp"
#include "scene_import_gltf.hpp"
#include "tinygltf_utils.hpp"
#include "animation_control.hpp"

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

// A shown dial as the morph topic lists it (publishMorphs): what the panel is told of it, as last sent.
struct ListedDial
{
  // what the list says of a Reach: cm and deg to a thousandth, as no one reads 17 digits of them
  struct Reach
  {
    double cm = 0, deg = 0;
    int    morphs = 0, bones = 0, scales = 0;

    bool operator==(const Reach&) const = default;
  };
  std::string              name, label, group, path, region, kind;
  double                   value = 0, min = 0, max = 1, weight = 0, rest = 0;
  bool                     helper = false, inert = false;
  std::optional<Reach>     reach;
  std::vector<std::string> items;  // the worn items it moves (MorphSet::itemsMoving)

  bool operator==(const ListedDial&) const = default;
};

// The shown dials of a figure's morphs, in the list's order.
std::vector<ListedDial> listedDials(const fv::morph::MorphSet& morphs);
// The topic's `channels` for them, each with its place in the order.
Json listedChannels(const std::vector<ListedDial>& dials);
// The patch to `channels` that turns `sent` into `now`, empty when they say the same; none when the
// whole list must go again: dials came, went or moved in the order, which each one's `order` says.
std::optional<Json> listedPatch(const std::vector<ListedDial>& sent, const std::vector<ListedDial>& now);
// The patch to `channels` for the shown dials whose value or weight moved since `listed` was sent
// (`listedAt`: where each is in it), which `listed` takes on; empty when none did. Every frame a dial
// moves: a dial found at its place in `listed` costs no lookup by name.
Json valuesPatch(const fv::morph::MorphSet& morphs, std::vector<ListedDial>& listed, const std::unordered_map<std::string, size_t>& listedAt);

// A DAZ pose file's keys over time played on a figure, as a clip of the "animation" topic
// (fv_host_scene.cpp): opened from the Library it waits at its end, which is the pose the file sets;
// imported it plays. Every frame it plays, its channels go through the figure's pose rig and dials
// as a pose preset's do, and are no step of the history. A VaM Timeline clip (fv_host_vam.cpp) has
// no channels: its controllers hold the figure's ragdoll in Physics mode, set from it each frame.
struct FigureAnimation
{
  int                 number = -1;  // its clip number, from Impl::nextClip
  std::string         name, rel;
  Json                source;  // how it was opened, for a scene file: {rel} in the content folders, else {path}
  fv::pose::PoseClip  clip;
  std::shared_ptr<fv::physics::vam::Player> vam;
  // a VaM clip blends in from where the body is at its next frame: it begins (not after a seek)
  bool vamBegins = true;
  float               time  = 0;
  bool                dirty = true;  // to be sampled and applied at the next frame
  // each channel's value before it came, which Rest pose and Remove put back, and as last applied
  // (NaN: to be sent), so a frame sends only what moved
  std::vector<double> before, applied;
  std::chrono::steady_clock::time_point posePublished{};
};

// A loaded figure and what the renderer made of it. Lives on the application thread.
struct Loaded
{
  std::string                 id;  // its prop's
  std::unique_ptr<fv::Figure> fig;
  // drapes the worn dynamic surfaces (it works on `fig`, so it goes first); null without a body
  std::unique_ptr<fv::physics::Cloth> dforce;
  int                                 dforceVersion = -1;
  // keeps the worn conforming meshes out of the posed body (figure/wardrobe/collider.hpp); null without a body
  std::unique_ptr<fv::wardrobe::Collider> collider;
  int                                     collisionVersion = -1;
  // dents the body where a prop presses into it (figure/physics/contact/contact.hpp); null but for a
  // person's body (fv_host_figure.cpp), not a rigged prop's. Where the figure stood at the last look,
  // as a move pokes it.
  std::unique_ptr<fv::physics::Contact> contact;
  int                                   contactVersion = -1;
  std::optional<glm::mat4>              contactPlaced;
  // rain on its skin (figure/physics/rain/rain.hpp); null but for a person, as `contact`. The feed its
  // water last went to: a feed made since gets it again
  std::unique_ptr<fv::physics::Rain> rain;
  int                                rainVersion = -1;
  const fv::FigureFeed*              rainFed     = nullptr;
  // its ragdoll and Physics mode (figure/physics/rig/figure_physics.hpp); null but for a person, as
  // `contact`
  std::unique_ptr<fv::physics::FigurePhysics> physics;
  int                                         physicsVersion = -1;
  // the chest's soft bones (figure/physics/rig/secondary.hpp), in either mode; null but for a person
  std::unique_ptr<fv::physics::Secondary> secondary;
  int                                     secondaryVersion = -1;
  // its soft flesh (figure/physics/flesh/flesh.hpp), in either mode; null but for a person
  std::unique_ptr<fv::physics::Flesh> flesh;
  int                                 fleshVersion = -1;
  // per ragdoll part, its part of the body (contact.hpp's `part`): what the flesh meets of its own limbs
  std::vector<int> limbParts;
  // without dents of its own (a DAZ prop, a chair), it is what the bodies near it meet: its shown
  // meshes posed in its root's space (a person's body alone, without its own dent), made again once
  // its pose or shape moved (`contactRev` bumps; a person's own dent doesn't)
  int                                                               contactRev    = 0;
  int                                                               contactSetsAt = -1;
  std::vector<std::shared_ptr<const fv::physics::SharedTriangles>> contactSets;
  // what it is in the renderer's scene, the prop feed's (fv_prop_feed.hpp), and the scene node each of
  // the feed's entities is; set by each composition
  fv::FigureFeed*             feed = nullptr;
  std::map<rs::EntityId, int> nodeOf;
  // morph topic: -1 until the channel list has been published, then the shown dials as last sent, in
  // the list's order, and where each is in it. A change to the channels (MorphSet::listVersion) that
  // leaves them as they were sends nothing: a dial's first drag brings its correctives in, hidden.
  int                                     morphList = -1;
  std::vector<ListedDial>                 listed;
  std::unordered_map<std::string, size_t> listedAt;
  int                                     morphsSeen = -1;  // MorphSet::version() last published
  uint64_t                                                     busyJob = 0;
  bool                                                         busyOn  = false;
  // the textures the glTF was built with, by request key, and whether their rows run bottom-up:
  // a wardrobe change builds the glTF again with these and the new items' own
  std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
  bool                                                                    texturesFlipped = false;
  // cooks that landed on a worker for its fallbacks (cookInPlace), for the application thread to take
  std::mutex                                                                    landedMutex;
  std::vector<std::pair<fv::build::TextureRequest, fv::build::TextureImage>> landed;
  // of those cooks, the ones not yet taken or given up on (application thread): a still waits for them
  int cooking = 0;
  // its DAZ library, for source textures and what it wears (the settings mutex)
  std::shared_ptr<fv::dson::Library> lib;
  // how it was opened ({rel} or {path}), for a scene file
  Json source;
  // material presets opened onto it, library paths in order, for a scene file
  std::vector<std::string> materialPresets;
  // those laid over the skin or eyes chosen, which go back after them, and what each set: a later
  // choice over those surfaces takes the preset off them, so it goes back before that choice instead
  struct PresetOver
  {
    std::string   rel;
    std::set<int> surfaces;
  };
  std::vector<PresetOver> presetsOver;
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
  // the tattoos on its skin (material.tattoo), bottom up, over the skin and makeup: each stays its own
  // to hide, fade or take off, by laying the others again. `tattooLayer` is what they set together.
  struct Tattoo
  {
    std::string                           rel;
    bool                                  on      = true;
    float                                 opacity = 1;
    bool                                  layered = false;  // a layered image preset, which can fade
    std::shared_ptr<const fv::dson::Json> doc;
  };
  std::vector<Tattoo>   tattoos;
  fv::dson::PresetLayer tattooLayer;
  // the add-ons its file lists are still going on; a restore waits for them, as it would otherwise
  // wear them a second time without their presets
  bool                               addonsPending = false;
  // wear batches started and not yet asked to be composed (startWear): any worker holds the count up
  std::atomic<int>                   wearsRunning{0};
  // one wear batch at a time commits to it, in the order they asked. Its own: a batch prepares against a
  // copy of this figure, and another figure's has nothing to wait for here
  std::mutex                         wearMutex;
  std::vector<std::function<void()>> afterAddons;
  // a DAZ animation opened onto it: one at a time
  std::optional<FigureAnimation> animation;
};

// What a prop is to the user. The DAZ kinds (character to prop) all hold a `figure` (a DAZ file's scene
// graph, dials and rig); character and creature are bodies, which wear, collide with their clothes and
// drape them; only a character has contact, Physics mode, soft bones and flesh. A group is an empty
// carrier to put props under. Decided at load (classifyDaz), and the user may change a DAZ kind
// (prop.set's kind, setKind).
enum class PropKind
{
  eCharacter,
  eCreature,
  eVehicle,
  eSet,
  eProp,
  eGroup,
  eModel,
  eCamera,
  eLight
};

// What decides a DAZ prop's kind as it loads (classifyDaz): its library path and the file's
// asset_info.type; the app's reading of the library, which only turns a prop into something it reads
// more exactly; or a kind it had before (a restore, a duplicate), which stands.
struct KindInput
{
  std::string             rel;
  std::string             assetType;
  std::optional<PropKind> hint;
  std::optional<PropKind> forced;
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
  std::string             bone;    // the parent figure's posable bone it rides on (its id), "" for the figure itself
  std::shared_ptr<Loaded> figure;
  // a model: as its file holds it (shared by duplicates) and where it came from
  std::shared_ptr<const tinygltf::Model> model;
  std::string                            path;
  std::string                            format;
  CameraProp                             camera;
  LightProp                              light;
  std::unique_ptr<FvMaterials>           materials;  // a figure's or a model's
  // what the composition made of it in the renderer's scene, as the prop feed holds it
  // (fv_prop_feed.hpp): its prop root, a model's import, a light's record
  rs::EntityId                           root;
  rs::ImportResult                       imported;
  rs::LightId                            lightRecord;
  Json                                   stats = Json::object();
  // what the user changed of nodes inside it (node.set), by topic node: its local transform (in its
  // parent's units: a DAZ figure's centimetres) and whether it shows; a model's node keeps what its
  // file had, to go back to
  struct NodeEdit
  {
    std::optional<rs::Transform> local;
    std::optional<bool>          visible;
    std::optional<rs::Transform> baseLocal;
    bool                         baseVisible = true;
  };
  std::map<int, NodeEdit> nodeEdits;
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
  // a host attaching again (another in the same process: the tests) sends them again
  void resume()
  {
    std::lock_guard lock(m_mutex);
    m_stopped = false;
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
// And a second line, of the frames those landed in ("drag frames"): the application thread's loop from
// one host update to the next, the host's update in its phases, the renderer's recording, and on how
// many the GPU had finished the frame before when this one was submitted, so it waited on the CPU.
struct DialTiming
{
  enum Kind
  {
    eSetMany,
    eApplyMesh,
    ePoseSet,
    eApplyNodes,
    eApplyRebind,
    eFrameLoop,
    eFrameHost,
    eFrameRecord,
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
  std::vector<std::pair<std::string, double>> phases;     // the host update's ms, by name, in order
  int                                   idleAtSubmit = 0;  // drag frames the GPU waited for
  uint64_t                              landed       = 0;  // samples ever, kept across reports

  void add(Kind k, double ms)
  {
    landed++;
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

  // a drag frame's update, its phases as (name, ms) with a name repeated for each figure
  void addHost(double ms, const std::vector<std::pair<const char*, double>>& phaseMs)
  {
    const uint64_t keep = landed;
    add(eFrameHost, ms);
    landed = keep;
    for(const auto& [name, t] : phaseMs)
    {
      auto it = std::find_if(phases.begin(), phases.end(), [&](const auto& p) { return p.first == name; });
      if(it == phases.end())
        it = phases.insert(phases.end(), {name, 0.0});
      it->second += t;
    }
  }
  // the frame after a drag frame was recorded and submitted: its loop, recording and wait
  void addFrame(double loopMs, double recordMs, bool gpuIdle)
  {
    const uint64_t keep = landed;
    add(eFrameLoop, loopMs);
    add(eFrameRecord, recordMs);
    landed = keep;
    idleAtSubmit += gpuIdle ? 1 : 0;
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
    std::string out = line;
    const Sum&  lp  = sums[eFrameLoop];
    const Sum&  hu  = sums[eFrameHost];
    const Sum&  rc  = sums[eFrameRecord];
    if(hu.n > 0)
    {
      snprintf(line, sizeof line,
               "\n[fv-time] drag frames: %d, loop %.1f ms mean, %.1f max; host update %.1f ms mean, %.1f max (", hu.n,
               mean(lp), lp.maxMs, mean(hu), hu.maxMs);
      out += line;
      for(size_t i = 0; i < phases.size(); i++)
      {
        const std::string& full = phases[i].first;  // "frame: <phase>"
        const size_t       c    = full.rfind(": ");
        const std::string  name = c == std::string::npos ? full : full.substr(c + 2);
        snprintf(line, sizeof line, "%s%s %.1f", i ? ", " : "", name.c_str(), phases[i].second / hu.n);
        out += line;
      }
      snprintf(line, sizeof line, "); recording %.1f ms mean, %.1f max; the GPU waited on %d of %d", mean(rc), rc.maxMs,
               idleAtSubmit, lp.n);
      out += line;
    }
    const uint64_t keep = landed;
    *this               = DialTiming{};
    landed              = keep;
    return out;
  }
};

// ---- helpers (fv_host_support.cpp)

const char* kindName(PropKind k);
// a kind's name as the protocol sends it, back; none for one it doesn't know
std::optional<PropKind> kindOf(std::string_view name);
// a DAZ file's kinds (the prop holds a figure), and of those the bodies (character, creature)
bool dazKind(PropKind k);
bool bodyKind(PropKind k);

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
// A worn item's place among its file's items, from its key: "w5" is the first, "w5.2" the third
int pieceOfKey(const std::string& key);

// encodeURI for an absolute path used as an image URI (the renderer percent-decodes URIs).
std::string uriOfPath(const std::string& path);

std::string mimeOf(const std::string& path);

TextureCalls& textureCalls();

// Where a cooked texture is (BC7 KTX2, cutout folded into alpha, rows bottom-up for the DSON UVs
// when `flip`). The texture server cooks it first, which takes seconds per 4K map; `cachedOnly`
// answers at once instead, `uncooked` when it isn't cooked.
CookedTexture cookedTexture(const HostSettings& s, const fv::build::TextureRequest& req, bool flip, bool cachedOnly, bool quiet = false);

// Cooks textures (flipped) side by side, colour maps first (a normal map cooks about ten times
// slower), and returns once all are done; `landed` is called with each as it lands, on the thread
// that cooked it. FvHost::Impl::cookInPlace runs it for a figure on a cold cache.
void cookAll(const HostSettings&                                                                  s,
             std::vector<fv::build::TextureRequest>                                               requests,
             const std::function<void(const fv::build::TextureRequest&, fv::build::TextureImage)>& landed);

// The source image as it is, for a renderer without a texture server (tests, a standalone run).
std::optional<fv::build::TextureImage> sourceTexture(const fv::dson::Library& lib, const fv::build::TextureRequest& req);

// A texture without waiting for the cooker: the source image as it is, where that shows the right
// thing. A normal map keeps its green (the material tells the renderer to flip it) and loses the
// bump blended onto it. A roughness map (its grey into glTF's channels) and a height map (turned
// into a normal map) are composed here without a cooker or with Original textures; with the cooker
// they stand in as one texel until their cooks land (a few seconds without skin's sheen and pores,
// where a still waits). A cutout
// alone changes the picture too much: it, makeup and an iris colour (and formats the renderer cannot
// read) are cooked, unflipped, and waited for; without a cooker, or with Original textures (only
// what the renderer cannot read is cooked then), they are composed here too (fv_textures.hpp).
// `flip`: the figure's images are flipped, so one cooked here is as it will stay, which `final` says.
std::optional<fv::build::TextureImage> fallbackTexture(const HostSettings& s, const fv::dson::Library& lib, const fv::build::TextureRequest& req, bool cooker, bool flip = false, bool* final = nullptr);

// A figure's textures, several resolved at once. In a figure built for the cooked images (`flipped`,
// which needs the cooker): the cooked one where the cache has it, else the fallback, its request
// added to `uncooked` (a source's rows run the other way, which its material's slot turns over). In
// one built for the sources: the fallbacks.
std::vector<std::optional<fv::build::TextureImage>> resolveTextures(const HostSettings&                           s,
                                                                    const fv::dson::Library*                      lib,
                                                                    const std::vector<fv::build::TextureRequest>& requests,
                                                                    bool                                          flipped,
                                                                    std::vector<fv::build::TextureRequest>&       uncooked);

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
// What a figure's or a model's meshes will take on the GPU, for HostRenderer::geometryShortfall: their
// buffers in the residency's formats (scene_price.hpp) and what the driver says their BLASes hold.
// Their textures aren't in it: those come down to fit as they arrive.
// A failure as a person reads it: bad_alloc's own words are "bad allocation".
std::string problemOf(const std::exception& e);
// What a prop holds of system memory: its figure's meshes, or its model's buffers
uint64_t propRamBytes(const Prop& p);
// fn(i) for each of `count`, on up to `threads` threads of their own, for work that waits on the pool or
// on a server. When they have all stopped, the first thing one threw comes out here.
void sideBySide(size_t count, int threads, const std::function<void(size_t)>& fn);

uint64_t geometryBytes(const fv::SceneGraph& scene, const HostRenderer& renderer);
uint64_t geometryBytes(const fv::SceneGraph& scene, const std::vector<int>& meshes, const HostRenderer& renderer);
uint64_t geometryBytes(const tinygltf::Model& model, const HostRenderer& renderer);

// The registered parameters nvmcp can read and write: values, not triggers or custom parsers, and
// none that loads or saves the scene behind the host (fv::hostExposesParameter).
nvmcp::detail::ParameterList exposedParameters(const nvutils::ParameterRegistry* registry);

Json toolJson(const nvmcp::ToolResult& r);

// A batch of the renderer's parameters as the host sets it (`params.set`, FvHost::Impl::setParams):
// `args` is nvmcp's, {"values": {name: value, ...}}, validated whole before any value is written, each
// parameter's callback then run. nvmcp writes every value before it runs any callback, and a callback
// may set another parameter (ptSamples turns Auto SPP off, a preset overwrites its fields), so a value
// the batch names beside it was lost to it: the app's ptAdaptiveSampling to its ptSamples, in every
// batch it sent. A named value holds: those of parameters with no callback of their own are written
// once more after the callbacks. A parameter with a callback stays as its callback left it, since it
// may clamp or consume its own value. Returns nvmcp's answer, the values as they are in the end.
Json setExposedParameters(const nvutils::ParameterRegistry* registry, const Json& args);

// How a figure is built for the scene: its textures as resolved (cooked, or baked for an edit),
// flipped as they arrive, the skin for the renderer, the surfaces light walks through.
fv::build::BuildOptions buildOptionsOf(const Loaded& f, fv::mat::SkinMode skin, bool solidGlass);

int itemIdOf(const Json& v);

int itemId(const Json& args);

// `item` as one id or a list of them ("i:3", 3, ["i:3", "i:4"]).
std::vector<int> itemIds(const Json& args);

const fv::WornItem* wornItem(const fv::Figure& fig, int id);

// A dForce mode by its name; "gpu" needs the build to have the GPU solver.
fv::physics::ClothMode dforceMode(const std::string& name);

}  // namespace fvhost

using namespace fvhost;

std::string figureSource(const Loaded& figure);
std::string figureGeneration(const Loaded& figure);
// A figure's contact settings as the scene file and the history keep them ({on, softness, bulge,
// reach}); the figure has a contact pass.
Json contactState(const Loaded& f);
// A figure's rain as the scene file and the history keep it ({on, amount, time, dried, seed}); the
// figure has rain.
Json rainState(const Loaded& f);
Json secondaryState(const Loaded& f);
Json softState(const Loaded& f);
Json physicsState(const Loaded& f);

struct FvHost::Impl
{
  HostRenderer*               renderer = nullptr;
  std::shared_ptr<FvProtocol> protocol;
  nvapp::Application*         app = nullptr;

  mutable std::mutex settingsMutex;
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
  bool              lmb = false, mmb = false, rmb = false;
  // flying: the movement keys the app holds down with the right button, as a direction in the view's
  // right, up and forward, sent again each of its frames; applyFly() moves the view once a frame
  struct Fly
  {
    glm::dvec3                            move{0.0};
    bool                                  fast  = false;
    double                                speed = 0;  // scene units a second, from the scene's size as a flight starts
    double                                scale = 1;  // the wheel's while the right button is held, kept for the next flight
    bool                                  flying = false;
    std::chrono::steady_clock::time_point heard{}, last{};
  };
  Fly fly;
  // the right button's look still to turn, in pixels, and when applyLook() last took some
  glm::dvec2                            lookPending{0.0};
  std::chrono::steady_clock::time_point lookLast{};

  // application thread only: the scene's props in order
  std::vector<std::shared_ptr<Prop>> props;
  int                                propCount = 0;
  // morph.set merges here (any thread); onUpdate applies it as one setMany per figure
  std::mutex                                           morphMutex;
  std::set<std::string>                                morphFigures;  // the figures morph.set may name
  std::map<std::string, std::map<std::string, double>> pendingMorphs;
  std::map<std::string, Json>                          pendingGestures;  // the latest `gesture` sent with them
  std::map<std::pair<std::string, std::string>, Json>  pendingTight;     // wardrobe.tight's latest, by prop and item
  // frames until the camera frames the scene (0: none pending)
  int frameIn = 0;
  // the gizmo's latest drag on a bone: the rig takes it once a frame, however many events came
  struct GizmoEdit
  {
    std::shared_ptr<Loaded> figure;
    std::string             bone;
    fv::Quat                rotation;
    std::optional<fv::Vec3> position;
    uint64_t                gesture = 0;
  };
  std::optional<GizmoEdit> gizmoEdit;
  // the history's gestures (fv_host_journal.cpp): a gizmo drag, and a drag of the view looking
  // through a camera prop, each from press to release; counted up at each release
  uint64_t gizmoGesture = 0, viewGesture = 0;
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
  bool                          viewKept    = false;  // a scene file sets the view: its first figure or model isn't framed

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
  // started within 30 s, and no pipeline build to wait for, is dropped and the caller told so; one that is running is waited for, however
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
    while(result.wait_for(30s) != std::future_status::ready)
    {
      // a pipeline build holds the thread for as long as it takes, and is no sign of a hang: the settings
      // that are waiting must not be lost to it, with the Render tab still showing them
      if(pipelineBuildsRunning().load() > 0)
        continue;
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
  // the notice that a figure drawn with DAZ's skinning exports with glTF's linear four
  void noticeLinearExport(const Loaded& f);
  // once a second: system memory short, what is only kept for later goes, and it says so
  void watchSystemMemory();
  // Worn items' detail following GPU memory: what could move, told to the renderer, and a step of it asked for
  void offerItemDetail();
  void stepItemDetail(uint64_t item, bool up);
  struct DetailItem
  {
    std::weak_ptr<Loaded> figure;
    int                   item = 0;
    fv::WearQuality       lower, higher;
  };
  std::map<uint64_t, DetailItem>        detailItems;  // as last offered, under keys never used again
  uint64_t                              detailKeys   = 0;
  bool                                  detailMaking = false;  // one at a time
  bool                                  detailNoticed = false;
  std::chrono::steady_clock::time_point detailAgainAt;  // what failed for want of system memory isn't tried until then
  rs::RamWatch                          ramWatch;
  std::chrono::steady_clock::time_point lastRamReading{};
  bool                                  ramShort = false;
  // for what can fail every frame until something changes: once in a while
  void noticeNowAndThen(const std::string& text)
  {
    const auto now = std::chrono::steady_clock::now();
    if(now - lastFrameNotice < std::chrono::seconds(10))
      return;
    lastFrameNotice = now;
    notice(text, "error");
  }
  std::chrono::steady_clock::time_point lastFrameNotice{};
  // a figure's CPU drawn vertices brought up to its morphs, for an export
  void catchUpDrawn(Loaded& f);
  bool        ridesOn(const Prop& q, const std::string& id) const;
  // a light prop's record as the prop has it, at its real intensity (hidden, its entity shines with nothing)
  tinygltf::Light lightPayload(const Prop& p) const;
  // the props in the renderer's scene, fed as deltas, and a prop as the feed takes it
  std::unique_ptr<fv::PropFeed> propFeed;
  fv::PropSpec                  specOf(const Prop& p, bool figureChanged) const;
  // what the compositions fed goes from the renderer's scene
  void releaseFed();

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
  std::map<uint64_t, Json> endedJobs;  // how each of those ended, for a history run waiting on one
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
      endedJobs[id] = state;
      // a load or a wear that ended changed the scene the crash copy keeps
      noteRecovery();
      while(finishedJobs.size() > kKeptJobs)
      {
        patch[std::to_string(finishedJobs.front())] = nullptr;
        endedJobs.erase(finishedJobs.front());
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
  // Named or selected, any DAZ prop (a rigged chair poses too); the only one in the scene counts
  // only bodies (a set beside one person leaves that person the figure).
  std::shared_ptr<Prop> figureProp(const Json& args) const;
  std::shared_ptr<Loaded> figureArg(const Json& args) const;
  // A body a command is for (wear, collision, dForce, contact, Physics mode): the one it names, which
  // must be a character or creature (else it throws, naming `what`), else the selected prop when it is
  // one, else the only body. Null when there is none of those.
  std::shared_ptr<Prop>   bodyProp(const Json& args, const char* what) const;
  std::shared_ptr<Loaded> bodyArg(const Json& args, const char* what) const;
  // A DAZ prop becomes another DAZ kind: toward a body it gains a body's machinery (buildBody), away
  // from one it loses it (dropBody); what it wears stays on
  void setKind(Prop& p, PropKind kind);
  PropKind classifyDaz(const Loaded& f, const KindInput& in) const;
  void     buildBody(Prop& p);
  void     dropBody(Loaded& f);
  bool                    inScene(const std::shared_ptr<Loaded>& f) const;
  // a new prop at the end; `id` one it had (a restore), else the next
  std::shared_ptr<Prop>   addProp(PropKind kind, const std::string& name, const std::string& id = {});
  void                    removeProp(const std::shared_ptr<Prop>& p);
  std::unique_ptr<FvMaterials> makeMaterials(const std::shared_ptr<Prop>& p);
  glm::mat4               worldOf(const Prop& p) const;
  // what its transform is relative to: its parent's world, on the bone it rides on (identity at the
  // scene's root)
  glm::mat4               parentWorld(const Prop& p) const;
  glm::mat4               parentWorld(const std::string& parent, const std::string& bone) const;
  // a figure's posable bone (an id or a name) as its node in the figure's scene graph, -1 for none
  int                     boneNode(const Prop& figure, const std::string& bone) const;
  // where the bone is in the figure's prop root, posed (identity for none, or before it is fed)
  glm::mat4               boneOffset(const Prop& figure, const std::string& bone) const;
  void                    applyTransform(const Prop& p);
  // the scene shows it: it is visible, and Isolate (view.set's solo) leaves it in
  void                    applyVisible(const Prop& p);
  bool                    shownNow(const Prop& p) const;
  // view.set's solo: only these show, with what rides on them and what carries them; empty, all
  std::set<std::string>   solo;
  void                    setSolo(const std::vector<std::string>& ids);

  // ---- placing (fv_host_place.cpp): what a prop takes up in the world, and rays against the scene
  using Bounds = std::array<glm::vec3, 2>;
  // a triangle set the contact pass keeps (contactSets, contactMesh) and where it stands
  struct PlacedSet
  {
    std::shared_ptr<const fv::physics::SharedTriangles> set;
    glm::mat4                                            toWorld{1.f};
  };
  struct SceneHit
  {
    glm::vec3 at;
    glm::vec3 normal;  // toward where the ray came from
  };
  std::vector<PlacedSet> placedSets(const Prop& p);
  // its own (a figure posed as it is, a camera or light where it stands), and with what rides on it
  std::optional<Bounds> ownBounds(const Prop& p);
  std::optional<Bounds> propBounds(const Prop& p);
  std::optional<Bounds> selectionBounds(const std::vector<std::shared_ptr<Prop>>& ps);
  // the nearest surface the ray meets; `madeOnly` skips sets not indexed yet. Safe on any thread
  static std::optional<SceneHit> rayThrough(const std::vector<PlacedSet>& sets, const glm::vec3& o, const glm::vec3& d, bool madeOnly);
  float                          groundHeight() const;
  std::vector<std::shared_ptr<Prop>> movingArg(const Json& args) const;
  void moveBy(const std::vector<std::shared_ptr<Prop>>& moving, const glm::vec3& delta, const std::string& label);
  Json propDrop(const Json& args);
  Json propAim(const Json& args);
  // prop.add {kind: "group", props: [id]}: a group at their bottom middle with them in it, one step;
  // prop.ungroup {prop}: what it carries on what carried it, and it gone, one step
  Json groupAdd(const Json& args);
  Json propUngroup(const Json& args);
  // prop.place: the selection where the pointer's ray meets the scene; the sets being indexed for it
  Json                                                propPlace(const Json& args);
  std::set<const fv::physics::SharedTriangles*>       placeIndexing;

  // ---- a node inside a prop (fv_host_place.cpp): hidden or moved, a set's wall, a car's wheel
  rs::EntityId nodeEntity(const Prop& p, int node) const;
  std::string  nodeName(const Prop& p, int node) const;
  // why it can't be edited (a bone, the root, a worn item's), "" when it can
  std::string nodeRefusal(const Prop& p, int node) const;
  Json        nodeSet(const Json& args);
  void        applyNodeEdit(Prop& p, int node);
  void        applyNodeEdits(Prop& p);
  // {visible, t, e, s, edited} as the panels show it (metres, degrees), null for one that can't be edited
  Json nodeState(const Prop& p, int node);
  // a node's edit as node.set takes it ({visible?, transform?}), and every one of a prop's
  Json nodeEditJson(const Prop& p, int node) const;
  Json nodeEditsJson(const Prop& p) const;
  void restoreNodeEdits(Prop& p, const Json& nodes);
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
  // the view moved (a drag, the wheel, camera.goto): a camera prop looked through, unless locked,
  // goes where it is now; `dragEnds` when a drag's release brought it
  void cameraFromView(bool dragEnds = false);
  void publishView();

  Json load(const Json& args);
  // A figure or prop from a DAZ library, as a new prop; `then` runs once it is in the scene.
  // `prefetch`: library files the figure will wear (a scene file's worn list), read on the pool from
  // the start of the load so its restore finds them in the figure's store
  // `iray`: installed Iray presets a 3Delight-era wearable may take as it goes on (WearRequest::iray)
  Json loadFigure(const HostSettings& s, std::vector<std::filesystem::path> roots, std::string rel, const std::string& name, Json source,
                  std::string target, bool asProp, std::function<void(std::shared_ptr<Prop>)> then = {}, std::vector<std::string> prefetch = {},
                  std::vector<std::string> iray = {}, std::string id = {}, Json place = {}, KindInput kind = {});
  void install(std::unique_ptr<fv::Figure>                                             f,
               uint64_t                                                                generation,
               uint64_t                                                                jobId,
               bool                                                                    texturesFlipped,
               fv::mat::SkinMode                                                       skin,
               std::shared_ptr<fv::dson::Library>                                      lib,
               std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images,
               std::vector<fv::build::TextureRequest>                                  uncooked,
               Json                                                                    source,
               std::function<void(std::shared_ptr<Prop>)>                              then,
               std::string                                                             id,
               Json                                                                    place,
               KindInput                                                               kind = {});
  // A figure on a cold texture cache: `requests` (shown with their fallbacks) cooked on a worker, each
  // swapped in as it lands (takeLanded), so the figure ends with every image cooked without a reload
  void cookInPlace(const std::shared_ptr<Loaded>& f, std::vector<fv::build::TextureRequest> requests);
  // The cooks that landed for `f` since the last call: its images, its feed's textures, and the
  // materials that sample them lowered again (with their edits)
  void takeLanded(Loaded& f);
  // Textures the scene's figures still have cooking (Loaded::cooking): render.status says it, and a still waits
  int texturesCooking() const;
  // some figure's drape is on its way on the GPU solver
  bool drapingOnGpu() const;
  // wardrobe: what a figure wears (fv_host.hpp commands wear, wardrobe.remove, wardrobe.visible)
  struct WearRequest
  {
    std::string              rel;
    std::string              name;     // the DSON node name an add-on has, so it is not put on twice
    std::vector<std::string> presets;  // material presets applied to it first
    // installed Iray presets a 3Delight-era item may take as it goes on, the one that keeps its look
    // (fv::dson::chooseIrayPreset); recorded as its colour, so "own" takes it off again
    std::vector<std::string> iray;
    std::vector<int>         ids;  // the ids its items had (Figure::commitWear), by their place in the file
    std::vector<std::string> keys;  // the keys they had, the same way
    bool only = false;  // its items without an id come off again at once: one item of a file put back
    bool step = false;  // the user put it on: a step of the history
    // the file's foot pose goes on (a heel's); not for a scene file's items, whose feet it has
    bool posesFeet = true;
    // its items made for another generation are fitted through the common layer where it can
    // (wardrobe::FitPath::eCommon): the item's Fit choice
    bool commonFit = false;
  };
  Json     wear(const Json& args);
  // What a restore sets on the figure besides its wardrobe, put on before the wear's composition so
  // it is built once (both run on the application thread): `surfaces` after the commits, with the
  // materials held, and `shape` (dials and pose) right before the build.
  struct WearLooks
  {
    std::function<void()> surfaces, shape;
    // what could not be put on, each named, by the time `after` runs (a restore's job reports them)
    std::shared_ptr<std::vector<std::string>> problems;
  };
  // `beforeBuild` runs on the application thread once everything is on, before the scene is built
  // again; `after` once it has been. With `commitAfter`, the requests are prepared at once and put
  // on once it is ready (a restore's worn items after the figure's add-ons). `looks`, a restore's.
  uint64_t startWear(std::shared_ptr<Loaded> target, std::vector<WearRequest> requests, std::function<void()> beforeBuild = {}, std::function<void()> after = {},
                     std::shared_future<void> commitAfter = {}, WearLooks looks = {});
  // glTF, GLB and OBJ files, read on a worker, as a new prop
  Json importModel(const std::string& path, std::function<void(std::shared_ptr<Prop>)> then = {}, std::string id = {}, Json place = {});
  // What prop.add's `place` says of a prop being made (prop.set's fields), so that it's never drawn
  // anywhere else: a scene file's name, visibility, transform, lens or light
  void placeProp(const Prop& p, const Json& place);
  Json animationTopic();
  void publishAnimation(bool force);
  Json animationCommand(const std::string& cmd, const Json& args);
  // The clips by the number the "animation" topic gives them: animation i of a model prop's file
  // keeps its number while the prop is in the scene, and a number is never given again (D4)
  std::map<std::pair<std::string, int>, int> clipNumbers;
  int                                        nextClip    = 0;
  int                                        currentClip = -1;  // the number of the clip the renderer plays
  int                                        clipNumber(const std::string& prop, int index);
  // a clip number's record in the scene, none when it isn't there or is a figure's animation
  rs::ClipId clipOf(int number) const;
  // the figure whose animation a clip number is, or null
  Loaded* figureClipOf(int number) const;
  bool    clipExists(int number) const { return !clipOf(number).isNone() || figureClipOf(number); }
  // the clips in the scene in display order: their numbers
  std::vector<int>                      clipOrder() const;
  Json lastAnimation;
  std::chrono::steady_clock::time_point animationAt{};
  // A pose file's animation onto a figure, held there and made the current clip: paused at its end
  // or playing from its start. Another the figure had goes first, putting back the pose it found.
  // Throws when nothing in it reaches the figure. `source` is how it was opened (FigureAnimation);
  // `vam`, a VaM clip's player, which moves the ragdoll instead of channels.
  void holdAnimation(Prop& prop, fv::pose::PoseClip clip, const std::string& rel, bool play, Json source,
                     std::shared_ptr<fv::physics::vam::Player> vam = nullptr);
  // the figure's animation off it, its channels back where it found them
  void dropAnimation(Loaded& f);
  void restoreBeforeAnimation(Loaded& f);
  // The current clip, a figure's animation, moved on by this frame's time and applied
  void applyFigureAnimation();
  std::chrono::steady_clock::time_point figureAnimationFrame{};
  // load {as: "animation"}: a pose file read on a worker and played on the figure named or selected.
  // `resume`, a scene file's "animation" block, puts it back as it was saved instead: at its time,
  // playing or not, with the pose it found, and no notice.
  Json loadAnimation(std::vector<std::filesystem::path> roots, std::string rel, std::string target, Json source, Json resume = nullptr);
  // load {as: "vam", path, prop?, person?, clip?, scene?}: a VaM scene (.json, or a .var package's)
  // read on a worker, one Person's Timeline clip played on the figure named or selected, in Physics
  // mode (fv_host_vam.cpp)
  Json loadVam(const Json& args);
  // a VaM clip's targets onto its figure's ragdoll, at the clip's time
  void driveVam(Loaded& f);
  // A figure's animation as a scene file keeps it (sceneState's "animation"), and a channel's key in it
  Json animationState(const Loaded& f);
  static std::string animationKey(const fv::pose::PresetTarget& t);
  Json exportGlb(const Json& args);
  Json sceneState(const Json& args);
  Json sceneRestore(const Json& args);
  Json wardrobeRemove(const Json& args);
  Json wardrobeVisible(const Json& args);
  Json wardrobeKeepFull(const Json& args);
  Json wardrobeCollision(const Json& args);
  Json wardrobeCloth(const Json& args);
  Json wardrobeTight(const Json& args);
  Json wardrobeFit(const Json& args);
  // body contact (fv_host_physics.cpp): what a figure's pass meets, the look each frame at what moved
  // near a figure, figure.contact and its topic
  // `bodies`: other people's too; `ragdoll`: for a ragdoll, which meets the people in Physics mode by
  // their capsules instead
  std::vector<fv::physics::ColliderRef>               contactColliders(const Loaded& f, bool bodies = true, bool ragdoll = false);
  // another figure near this one moves by physics: its dents are held until it rests
  bool                                                movesNear(const Loaded& f);
  // what a person's soft flesh meets while it moves: the props near them and the limbs of the
  // people near them, and with self-contact their own
  fv::physics::FleshColliders                         fleshColliders(Loaded& f);
  std::shared_ptr<const fv::physics::SharedTriangles> contactMesh(const std::shared_ptr<const tinygltf::Model>& model, int mesh);
  const std::vector<std::shared_ptr<const fv::physics::SharedTriangles>>& contactSets(Loaded& prop);
  void                                                pokeContacts();
  void                                                publishContact(Loaded& f);
  Json                                                figureContact(const Json& args);
  // rain on a person's skin (fv_host_physics.cpp): figure.rain and its topic
  void publishRain(Loaded& f);
  Json figureRain(const Json& args);
  // Physics mode (fv_host_physics.cpp): the frame's step, the Pull, physics.set and physics.restore
  void stepPhysics();
  void publishPhysics(Loaded& f);
  Json physicsOps(const Loaded& f);
  Json physicsSet(const Json& args);
  Json physicsRestore(const Json& args);
  Json physicsPin(const Json& args);
  // secondary motion: physics.secondary and the secondary/<prop> topic
  void publishSecondary(Loaded& f);
  Json physicsSecondary(const Json& args);
  // soft flesh: the frame's step once the figure is posed, physics.soft and the soft/<prop> topic
  void stepFlesh(Loaded& f);
  // its offsets onto the renderer: the body's, and those of what is worn over it
  void showFlesh(Loaded& f);
  std::vector<float> fleshItem, fleshDrawn;  // showFlesh's, kept between frames
  void publishSoft(Loaded& f);
  Json physicsSoft(const Json& args);
  bool cameraRay(const glm::vec2& px, glm::vec3& origin, glm::vec3& dir) const;
  bool pullAt(const std::string& phase, const glm::vec2& px);
  std::chrono::steady_clock::time_point physicsFrame{};
  double                                physicsElapsed = 0;  // the frame's, as stepPhysics found it (s)
  // the Pull's hold: the figure, how far along the pointer's ray it took hold, and what puts its
  // physics back as it was then (the history's step at the release)
  struct PullHold
  {
    std::string prop;
    double      depth = 0;
    Json        before;
  };
  std::optional<PullHold> pull;
  // each prop as a collider at the last look (a model, or a DAZ prop and its shape's revision): its
  // meshes' worlds and bounds (metres), whether it showed; and each model's meshes as triangle sets,
  // read on the pool the first time a pass needs them
  struct ContactSeen
  {
    std::vector<glm::mat4> worlds;
    fv::physics::Box       bounds;
    bool                   shown = false;
    int                    rev   = 0;
  };
  std::unordered_map<std::string, ContactSeen>                                                             contactSeen;
  std::map<std::pair<const tinygltf::Model*, int>, std::shared_ptr<const fv::physics::SharedTriangles>> contactMeshes;
  void showItem(const Loaded& f, const std::vector<std::pair<int, std::string>>& objects, bool visible);
  void publishDforce(Loaded& f, bool force);
  Json dforceCommand(const std::string& cmd, const Json& args);
  void publishFigure(Prop& p);
  void publishWardrobe(Loaded& f);
  Json poseSet(const Json& args);
  // the images a load or a wear resolved, decoded on this worker's threads so the reload takes them ready
  void prefetchImages(const std::vector<fv::build::TextureRequest>& requests, const std::vector<std::optional<fv::build::TextureImage>>& found);
  // Something changed that the figure's feed hasn't got yet (a node or mesh just worn): fed whole
  void feedWhole(Loaded& f);
  void applyNodes(Loaded& f, const std::set<int>& moved);
  void applyMesh(Loaded& f, const fv::morph::MeshChange& change);
  void applyRebind(Loaded& f, const std::vector<int>& meshes);
  void applyPendingMorphs();
  void publishMorphs(Loaded& f);
  void publishBusy(Loaded& f);
  void frameCamera();
  void normalizeInput(std::vector<Json>& events);
  void applyInput(const std::vector<Json>& events);
  void wheelDolly(const Json& e);
  void applyFly();
  void applyLook();
  void turnLook(const glm::dvec2& pixels);
  // a drag of the view ends: a camera prop looked through takes where the drag left the view (its
  // step ends with it)
  void endViewGesture();
  void patchPose(const Loaded& f, const std::vector<std::string>& boneIds);
  // a figure's posed bone at an entity, or {nullptr, nullptr}
  std::pair<std::shared_ptr<Loaded>, const fv::pose::PoseBone*> boneAt(rs::EntityId entity) const;
  void                      applyGizmoEdit();
  // where the gizmo turns a prop's root entity, in its own space (GizmoHooks::pivot)
  std::optional<glm::vec3> gizmoPivot(rs::EntityId entity);
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
  // a shader preset on some surfaces (fv_host.hpp material.shader): `rel` "" and no `doc` takes off what was chosen for them
  Json materialShader(const Json& args);
  Json applyShader(Prop& prop, const std::set<int>& surfaces, const std::string& rel, const fv::dson::Json* doc, bool keepEdits);
  // A skin or eyes `rel` chosen, over `surfaces`: it isn't a preset any more, and what lay over those
  // surfaces is gone from them, so goes back before the choice.
  void choiceReplaced(Loaded& f, const std::string& rel, const std::set<int>& surfaces);
  void publishEyes(Prop& p);
  // skin and makeup (fv_host.hpp material.skin, material.makeup): `rel` "" and no `doc` for the
  // character's own
  Json materialSkin(const Json& args);
  Json materialMakeup(const Json& args);
  Json applySkin(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits);
  Json applyMakeup(Prop& prop, const std::string& rel, const fv::dson::Json* doc, bool keepEdits);
  void publishSkin(Prop& p);
  // tattoos (fv_host.hpp material.tattoo)
  Json materialTattoo(const Json& args);
  Json materialTattooSet(const Json& args);
  Json materialTattooRemove(const Json& args);
  // puts one on the list, in place of an earlier go with the same preset, and lays them; throws,
  // changing nothing, when none of the preset is on the figure's skin
  Json addTattoo(Prop& prop, const std::string& rel, std::shared_ptr<const fv::dson::Json> doc);
  // whether a preset reaches the figure's skin
  bool reachesSkin(const Loaded& f, const fv::dson::Json& doc);
  // takes the stack off and, with `on`, lays those shown again; the surfaces this changed
  std::set<int> layTattoos(Loaded& f, bool on = true);
  std::set<int> wornSurfaces(const Loaded& f);
  // a preset chosen for part of a figure, read on a worker and applied by `apply` once the add-ons are on
  using ChoiceApply = std::function<Json(Prop&, const std::string& rel, const fv::dson::Json* doc)>;
  Json choiceJob(const Json& args, const std::string& command, const std::string& part, ChoiceApply apply, const std::string& own = "the character's own");

  // ---- lighting (fv_lighting.hpp), application thread only
  nvutils::ParameterRegistry* registry = nullptr;
  fv::lighting::State         lighting;
  bool                        lightingStarted = false;
  Json                        appliedParams   = Json::object();  // what the renderer was last given
  DialTiming                  dialTiming;
  // when the last frame's update ended: the renderer's own work in between shows when it is long
  std::optional<std::chrono::steady_clock::time_point> lastUpdateEnd;
  std::optional<std::chrono::steady_clock::time_point> lastUpdateStart;
  bool dragFrame = false;  // the last update landed a dial or pose step or an edit's vertices (DialTiming)
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
  // the host's sun in the renderer's scene, and the ground height the renderer chose at load
  rs::EntityId sunEntity;
  rs::LightId  sunLight;
  float        autoGround = 0.0F;
  // the physical sky's clouds as the renderer's last bake measured them, for the exposure
  fv::lighting::CloudReading cloudReading;

  // a look shown in place of the state's while the app's pointer is on its chip (lighting.preview):
  // its tone map's look and lookStrength are what the renderer grades with. Not journaled, not in
  // the topic; a lighting.set ends it
  std::optional<fv::lighting::Tonemap> lookPreview;

  Json lightingTopic() const;
  // a new reading of the clouds, after a bake: the exposure follows it. Every frame.
  void followCloudReading();
  Json lightingSet(const Json& patch);
  Json lightingPreview(const Json& args);
  void applyLighting(bool everything);
  void requestEnvironment();
  void applySun(bool force);
  // the "Sun" light as the current state has it, and its entity in `scene` (marked as the sky's sun)
  tinygltf::Light sunPayload() const;
  void            feedSun(rs::RetainedScene& scene);
  void setParams(const Json& values);
  void publishLighting();

  // ---- the edit history (fv_journal.hpp, fv_host_journal.cpp), application thread only
  fv::Journal journal;
  bool        replaying = false;  // an undo or redo applies its ops: they are no new step
  // Applies an edit as a step: what it is about to change is read first, and again after; a step is
  // recorded when the two differ, with the ops that put each back. Without a step (nothing changed,
  // or an op the journal does not know) it is only applied.
  Json journaled(const std::string& cmd, const Json& args, const std::function<Json()>& apply);
  // journaled() with the op itself as what applies
  Json edit(const std::string& cmd, const Json& args);
  // One op as a step holds it, {cmd, args}: the commands the captures write
  Json applyOp(const Json& op);
  // What an edit reads, named so it reads the same before and after ("kind" and what it names)
  Json journalScope(const std::string& cmd, const Json& args) const;
  // The ops that put the scope back as it is now
  Json journalState(const Json& scope);
  std::string journalLabel(const std::string& cmd, const Json& args, const Json& scope) const;
  Json historyStep(bool undo);
  void publishHistory();
  Json sceneOrder(const Json& args);
  // A prop that came (a load, prop.add, a duplicate) as a step: its undo takes it away
  void recordAdded(const std::shared_ptr<Prop>& p, const std::string& label);

  // ---- a figure's look, kept for a step's undo (C3): its surfaces and edits, the looks chosen, its
  // dials, pose and which items show. A preset or a choice (skin, eyes, makeup, an item's colour)
  // rewrites surfaces, which no command puts back: figure.look {prop, look} does, from the one kept.
  // A kept look goes once no step names it.
  struct FigureLook
  {
    std::vector<fv::Material>                   materials;
    std::vector<std::string>                    materialPresets;
    std::vector<Loaded::PresetOver>             presetsOver;
    std::optional<std::string>                  eyes, skin, makeup;
    std::set<int>                               eyesOwned, skinOwned;
    std::string                                 eyesPreset, skinPreset;
    fv::dson::PresetLayer                       makeupLayer;
    std::vector<Loaded::Tattoo>                 tattoos;
    fv::dson::PresetLayer                       tattooLayer;
    std::map<int, Loaded::ItemColor>            colors;
    std::vector<std::pair<std::string, double>> dials;
    std::vector<fv::pose::ChannelValue>         pose;
    std::map<int, bool>                         visible;
  };
  std::map<uint64_t, std::shared_ptr<const FigureLook>> keptLooks;
  uint64_t                                              lookCount = 0;
  uint64_t keepLook(const Loaded& f);
  Json     applyLook(const Json& args);

  // ---- worn items as steps (C3): what puts one back on as it was, and doing that
  Json itemRecord(const Loaded& f, const fv::WornItem& w) const;
  // wardrobe.restore {prop, item, record}: worn again under its id, its file's other items left off (-> {job})
  Json wardrobeRestore(const Json& args);

  // ---- structural steps (C3): a removed prop is kept, so its undo puts the same objects back
  // under its id, composed again; past kParkedBytes of them (the oldest go) or with system memory
  // short, it loads again from its record. A kept prop goes once no step names it.
  static constexpr uint64_t                            kParkedBytes = uint64_t(4) << 30;
  std::vector<std::pair<std::string, std::shared_ptr<Prop>>> parked;
  void pruneParked();
  // what puts a prop back: its place, fields, and a figure's or a model's scene-file part
  Json propRecord(const Prop& p);
  Json propRemove(const Json& args);
  // prop.restore {prop, record}: back under its id, kept or loaded again (-> {job})
  Json propRestore(const Json& args);
  void placeFrom(Prop& p, const Json& record);
  // An undo or redo applying its ops; it waits for a restore that loads, as a job
  struct HistoryRun
  {
    bool        undo = true;
    uint64_t    serial = 0;
    std::string label;
    bool        open = false;  // the step of a scene file opening
    Json        ops;
    size_t      next     = 0;
    Json        opposite = Json::array();  // what each op read before it applied: the way back
    Json        problems = Json::array();
    uint64_t    job      = 0;              // "Undoing ..." while it waits
    std::string jobLabel;
    uint64_t    waitJob  = 0;              // the load (then the restore) it waits on
    std::string waitProp;
  };
  std::optional<HistoryRun> historyRun;
  // starts a run (a step's ops, or a kept scene's): its answer once done, else {label, job} while it waits
  Json startRun(HistoryRun run);

  // ---- the scene kept on disk for a crash (C5): the ops that put every prop back under its id (as
  // scene.clear's undo has them), the order, the lighting and the view, rewritten on a worker a second
  // after the changes pause (recovery.file). scene.recover runs them as an undo runs its ops.
  static constexpr std::chrono::milliseconds kRecoveryQuiet{1000};
  std::filesystem::path recoveryPath;           // application thread; empty: none kept
  std::atomic<int64_t>  recoveryDirtyAt{0};     // steady clock, ns: the last change not kept yet; 0: none
  uint64_t              recoveryWrites = 0;     // application thread: the newest write started
  std::mutex            recoveryMutex;
  uint64_t              recoveryWritten = 0;    // under recoveryMutex: the newest write done, or dropped
  // any thread
  void noteRecovery();
  void writeRecovery();
  Json recoveryFile(const Json& args);
  Json recoveryPeek(const Json& args);
  Json recoveryForget(const Json& args);
  Json recoverScene(const Json& args);
  // applies the run's ops on from where it stopped; its answer once done, none while it waits
  std::optional<Json> runHistory();
  // a frame: the job a run waits on ended
  void pollHistory();

  std::unique_ptr<FvView> view;
  // how skin is shaded (render.mode's skin)
  std::atomic<fv::mat::SkinMode> skinMode{fv::mat::SkinMode::eMeasured};
  // render.mode's solidGlass: Iray Uber's Thin Walled off refracts as a closed solid
  std::atomic<bool> solidGlass{false};
  // a texture a material bake needs, from the cooker or a library (worker threads)
  std::optional<fv::build::TextureImage> bakeTexture(const std::shared_ptr<fv::dson::Library>& lib, const fv::build::TextureRequest& req, bool flipped);
};
