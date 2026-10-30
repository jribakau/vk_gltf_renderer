#pragma once

// Collision between worn conforming meshes and the posed body: DAZ Studio's smoothing modifier, as
// far as its collision goes. A conforming item follows the body's bones and morphs, and still
// ends inside it where a pose bends a joint the item's weights were not painted for, or where a
// body shape swells under a garment fitted to another one (the usual case across generations).
//
// Once the pose or shape stops changing for a moment, each item the figure wears is pushed out of
// the body in pose space: bound to the closest body point (left limbs to left limbs), pushed out
// along the skin's normal to a small clearance, and the push smoothed over the garment so it reads
// as cloth stretched over the body rather than a dent. Items are layered, inside out: each is
// pushed out of the body and of every item already pushed under it, so a shirt clears the bra by
// the same clearance it clears the skin. The order is each item's clearance from the body in its
// rest shape (a bra lies closer than a shirt), and the order worn where that is a tie. The result
// is written back as a hidden morph channel per mesh ("collide:<mesh>", weight 1) holding the cage
// deltas whose skinning gives the pushed shape; the renderer sees it as any other morph, and
// turning collision off clears it.
//
// An item the user made tighter (dson::Smoothing::tight) is first pulled toward the skin, and its
// clearance comes down with the amount, from what's above to half a millimetre, so it follows the pose
// and layers over what is under it as the push does.
//
// A mesh dForce drapes is left alone (the simulation collides it); the body itself is never moved.
// Each item's settings come from its file's smoothing modifier when it carries one, else from the
// defaults the figure's wardrobe gives clothing (figure.hpp, WornItem::collision).
//
// One thread (the figure's owner) calls everything; the pass runs on the pool over a snapshot.

#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <figure/dson/modifiers.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/scene/graph.hpp>

namespace fv::wardrobe {

// The hidden channel a mesh's correction is stored as.
std::string collideChannelOf(int mesh);

class Collider
{
public:
  // `bodyMesh` is the figure's body; `root` its root node (the pass runs in its space, centimetres).
  Collider(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root);
  ~Collider();

  // The meshes to keep out of the body and each one's settings, read when a pass starts: the
  // wardrobe's per-item collision settings. A mesh that is not listed, is hidden, or is unskinned is
  // left alone.
  std::function<std::map<int, dson::Smoothing>()> settings;
  // DAZ's skinning (eight influences, the file's dual quaternion share), as the figure's drawn meshes
  // are skinned (figureDazSkinning, gpu-figures-plan.md F3); off: four, linear
  bool dazSkinning = false;
  // The meshes in the order worn, for layering two that lie equally close to the body; unset or a
  // mesh not listed: mesh order.
  std::function<std::vector<int>()> wornOrder;
  // True for a mesh a simulation owns (dForce drapes it), which is not pushed here.
  std::function<bool(int mesh)> owned;

  bool enabled() const { return m_enabled; }
  // A pass is waiting out its debounce or running.
  bool busy() const;
  // Meshes holding a correction now.
  int corrected() const { return int(m_corrected.size()); }
  // Bumps whenever busy() or the set of corrected meshes changes, for a status topic.
  int version() const { return m_version; }

  // Something the figure shows changed: bones moved, or these meshes' vertices did. A pass is due
  // once the changes stop for a moment. The host reports what it flushed before pump() writes a
  // correction, so a correction is never its own cause.
  void changed(const std::vector<int>& meshes, bool nodesMoved);
  // Schedules a pass (an item worn or taken off, its settings changed).
  void poke();
  // The same for an edit being dragged, where the fit is what's being watched: a pass as soon as the
  // one in hand ends, not after a pause in the drag.
  void pokeNow();
  // Weight 1 pushes the items out, weight 0 shows them as they conform. Turning it on runs a pass.
  void setEnabled(bool on);

  // Starts a pass that is due and writes one that finished. Call every frame. True when it wrote
  // corrections (the morph set has vertices to flush).
  bool pump();

private:
  struct Job;
  std::shared_ptr<Job> prepare() const;
  bool                 apply(const Job& job, const std::vector<std::vector<float>>& positions);

  SceneGraph&      m_scene;
  morph::MorphSet& m_morphs;
  int              m_bodyMesh;
  int              m_root;

  bool                                  m_enabled   = true;
  bool                                  m_scheduled = false;
  std::chrono::steady_clock::time_point m_due;
  uint64_t                              m_gen    = 0;
  uint64_t                              m_jobGen = 0;
  std::shared_ptr<Job>                           m_snapshot;
  std::future<std::vector<std::vector<float>>>   m_job;
  std::set<int> m_corrected;
  int           m_version = 0;
};

}  // namespace fv::wardrobe
