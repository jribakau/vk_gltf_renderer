#pragma once

// Rain on a person's skin (rain_sim.hpp says what it does): the pass that runs it. Once the pose, the
// shape, which way is down or the settings stop changing for a moment, the rain is replayed from a dry
// skin on the pool over a snapshot of the posed body, and what it left (per material group, the beads
// and tracks in its UVs) is handed to the renderer by the host. So a still converges on the water, and
// a scene file opens with the same drops.
//
// Between passes the water rides the skin, since it lies in the skin's UVs. While a clip plays no pass
// runs (the drops would jump every frame): they ride with it, and run again for the pose it stops at.
//
// One thread (the figure's owner) calls everything; the pass runs on the pool over a snapshot.

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <figure/materials/water.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/physics/pass.hpp>
#include <figure/physics/rain/rain_sim.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics {

// What the last pass left, for the topic.
struct RainStats
{
  int beads = 0, tracks = 0, landed = 0, runs = 0, dripped = 0;
};

class Rain
{
public:
  // `bodyMesh` is the figure's body; `root` its root node (the pass runs in its space, centimetres).
  Rain(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root);
  ~Rain();

  // True while a clip plays: no pass runs, and one runs once it stops.
  std::function<bool()> driven;

  const RainSettings& settings() const { return m_settings; }
  bool                enabled() const { return m_settings.on; }
  int                 bodyMesh() const { return m_bodyMesh; }
  // A pass is due or running.
  bool busy() const { return m_pass.busy(); }
  // Bumps whenever busy(), the stats or the settings change, for a status topic.
  int              version() const { return m_pass.version(); }
  const RainStats& stats() const { return m_stats; }
  // What lies on the skin, per material group of the body mesh; empty while it's off.
  const std::map<int, mat::WaterLayer>& layers() const { return m_layers; }

  // Off dries the skin at the next pump; any other change runs a pass after a pause (a slider's drag
  // shows where it settled, not every step of the way).
  void set(const RainSettings& settings);
  // Where the figure is placed in the world on top of its scene graph (the host's entity), so the pass
  // knows which way is down. A turn that matters runs a pass.
  void setPlacement(const Mat4& placed);
  // Something it reads changed: bones moved, or these meshes' vertices did (only the body counts).
  void changed(const std::vector<int>& meshes, bool nodesMoved);

  // Starts a pass that is due and takes one that finished. Call every frame. True when layers()
  // changed.
  bool pump();

private:
  struct Job;
  struct Landed
  {
    std::map<int, mat::WaterLayer>     layers;
    RainStats                          stats;
    std::shared_ptr<const RainSurface> surface;  // made on the pool when the mesh was new to it
    const void*                        surfaceOf = nullptr;  // the mesh it was made from
  };
  std::shared_ptr<Job> prepare();
  bool                 apply(const Landed* landed);
  Vec3                 currentDown() const;

  SceneGraph&      m_scene;
  morph::MorphSet& m_morphs;
  int              m_bodyMesh;
  int              m_root;
  RainSettings     m_settings;
  Mat4             m_placed;
  Vec3             m_down{0, -1, 0};  // as the last pass was asked for
  // the skin it runs on, and the mesh it was made from
  std::shared_ptr<const RainSurface> m_surface;
  const void*                        m_surfaceOf = nullptr;
  std::map<int, mat::WaterLayer>     m_layers;
  RainStats                          m_stats;
  StillPass<Landed>                  m_pass;
};

}  // namespace fv::physics
