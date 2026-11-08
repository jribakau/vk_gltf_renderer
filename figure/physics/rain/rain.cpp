#include <figure/physics/rain/rain.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/physics/skin.hpp>

namespace fv::physics {

using namespace std::chrono_literals;

namespace {

// after an edit, how long to wait for the next before a pass
constexpr auto kDebounce = 250ms;
// which way is down moved enough to run again (about half a degree)
constexpr double kDownChange = 0.01;

}  // namespace

struct Rain::Job
{
  RainSettings                           settings;
  std::shared_ptr<const mesh::BuiltMesh> geo;
  std::shared_ptr<const RainSurface>     surface;  // null: made from `geo` on the pool
  std::vector<float>                     rest;
  bool                                   skinned = false;
  CageSkin                               skin;
  SkinPose                               pose;
  Mat4                                   toSpace;
  Vec3                                   down;
};

Rain::Rain(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root)
    : m_scene(scene)
    , m_morphs(morphs)
    , m_bodyMesh(bodyMesh)
    , m_root(root)
    , m_pass(std::chrono::duration_cast<std::chrono::milliseconds>(kDebounce))
{
  m_pass.waitWhile = [this] { return driven && driven(); };
}

Rain::~Rain() = default;

void Rain::set(const RainSettings& settings)
{
  const RainSettings next = clamped(settings);
  if(next == m_settings)
    return;
  m_settings = next;
  if(m_settings.on)
  {
    m_down = currentDown();
    m_pass.poke();
  }
  else
    m_pass.cancel();
  m_pass.bump();
}

void Rain::setPlacement(const Mat4& placed)
{
  m_placed = placed;
  if(!m_settings.on)
    return;
  // against what the last pass was asked for, so a slow turn adds up
  const Vec3 now = currentDown();
  if(length(now - m_down) > kDownChange)
  {
    m_down = now;
    m_pass.poke();
  }
}

void Rain::changed(const std::vector<int>& meshes, bool nodesMoved)
{
  if(m_settings.on && (nodesMoved || std::find(meshes.begin(), meshes.end(), m_bodyMesh) != meshes.end()))
    m_pass.poke();
}

Vec3 Rain::currentDown() const
{
  const auto worlds    = m_scene.worlds();
  const Mat4 rootWorld = m_root >= 0 && size_t(m_root) < worlds.size() ? worlds[size_t(m_root)] : Mat4{};
  // world down in the pass's space, through all the placement has (as the cloth's gravity)
  const Mat4 fromWorld = inverse(m_placed * rootWorld);
  const Vec3 down{-fromWorld.e[4], -fromWorld.e[5], -fromWorld.e[6]};
  return length(down) > 0 ? normalize(down) : Vec3{0, -1, 0};
}

std::shared_ptr<Rain::Job> Rain::prepare()
{
  timing::Scope t("rain: prepare", 2);
  if(m_bodyMesh < 0 || size_t(m_bodyMesh) >= m_scene.meshes.size())
    return nullptr;
  const Mesh& body = m_scene.meshes[size_t(m_bodyMesh)];
  if(!body.geo)
    return nullptr;
  auto job      = std::make_shared<Job>();
  job->settings = m_settings;
  job->geo      = body.geo;
  if(m_surface && m_surfaceOf == body.geo.get())
    job->surface = m_surface;
  std::vector<float> scratch;
  const auto         view = m_morphs.morphedCageView(m_bodyMesh, scratch);
  job->rest.assign(view.begin(), view.end());
  if(job->rest.empty())
    return nullptr;
  const auto worlds    = m_scene.worlds();
  const auto nodeOf    = physics::nodeOfMesh(m_scene);
  const Mat4 rootWorld = m_root >= 0 && size_t(m_root) < worlds.size() ? worlds[size_t(m_root)] : Mat4{};
  const int  node      = nodeOf[size_t(m_bodyMesh)];
  job->toSpace         = inverse(rootWorld) * (node >= 0 ? worlds[size_t(node)] : Mat4{});
  job->skinned         = body.skinned() && body.cageSkin.start.size() == job->rest.size() / 3 + 1;
  if(job->skinned)
  {
    job->skin = physics::cageSkin(body.cageSkin);
    job->pose = physics::currentSkin(m_scene, worlds, nodeOf, m_bodyMesh);
  }
  job->down = currentDown();
  return job;
}

bool Rain::apply(const Landed* landed)
{
  if(!landed)
  {
    // no body to rain on
    const bool had = !m_layers.empty();
    m_layers.clear();
    m_stats = {};
    return had;
  }
  if(landed->surface)
  {
    m_surface   = landed->surface;
    m_surfaceOf = landed->surfaceOf;
  }
  m_layers = landed->layers;
  m_stats  = landed->stats;
  m_pass.bump();
  return true;
}

bool Rain::pump()
{
  if(!m_settings.on)
  {
    if(m_layers.empty())
      return false;
    m_layers.clear();
    m_stats = {};
    m_pass.bump();
    return true;
  }
  return m_pass.pump(
      [this] {
        auto job = prepare();
        if(!job)
          return std::future<Landed>();
        m_down = job->down;
        return jobs::pool().submit([job] {
          timing::Scope t("rain: pass", 2);
          Landed        out;
          auto          surface = job->surface;
          if(!surface)
          {
            surface       = std::make_shared<const RainSurface>(rainSurface(*job->geo));
            out.surface   = surface;
            out.surfaceOf = job->geo.get();
          }
          const std::vector<float> posed = job->skinned ? physics::skinLocal(job->rest, job->skin, job->pose) : job->rest;
          const std::vector<float> body  = physics::transformPoints(job->toSpace, posed);
          const RainInput          in{.posed = body, .down = job->down};
          const RainResult         r = simulateRain(*surface, in, job->settings);
          out.layers                 = waterLayers(*surface, in, r);
          out.stats                  = {.beads = int(r.beads.size()), .tracks = int(r.tracks.size()), .landed = r.landed, .runs = r.runs, .dripped = r.dripped};
          return out;
        });
      },
      [this](const Landed* landed) { return apply(landed); });
}

}  // namespace fv::physics
