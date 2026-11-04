#pragma once

// Body contact: the body dents where a prop presses into it, and the flesh around the dent bulges by
// the volume pushed in (the user's design, "Body contact deformation", docs/physics-plan.md step 1).
// The sibling of the wardrobe collider with the roles swapped: that one moves clothing and never the
// body, this moves only the body.
//
// Once the pose, the shape or a prop near the body stops changing for a moment, a pass on the pool:
//  1. a body vertex inside a collider (behind the face it is closest to, or by a ray's crossings on a
//     closed piece) moves onto its surface;
//  2. those vertices make patches, joined by the cage's edges;
//  3. each patch puts the volume it pushed in back around itself, along the body's normals, fading
//     to nothing at a reach measured along the surface (a dent on one thigh leaves the other alone);
//  4. the correction is smoothed, which gives the dent its sloped rim; the volume is put back again,
//     since smoothing doesn't keep it, and what that carried into a collider is pushed out again.
//     Twice.
// The result is the hidden channel "contact:<body>", weight 1, written as the collider writes its
// own; everything else that reads the body (the collider, dForce, the renderer) sees the dent with no
// code of its own. The pass reads the body without its own channel, so it never deepens its own dent.
// While a clip plays the channel is at weight 0 and no pass runs: a dent where a hand used to be
// looks wrong.
//
// One thread (the figure's owner) calls everything; the pass runs on the pool over a snapshot.

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <figure/dforce/skin.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/physics/collide/triangle_set.hpp>
#include <figure/physics/pass.hpp>
#include <figure/physics/write.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics {

// The hidden channel a body's dent is stored as.
std::string contactChannelOf(int mesh);

struct ContactSettings
{
  bool on = true;
  // this body's share of an overlap with another figure (step 3); a prop takes none
  float softness = 0.5f;
  // the share of the pushed-in volume put back around the dent: 0 is a dimple only
  float bulge = 0.7f;
  // how far along the skin the bulge reaches at least, cm; a deep dent reaches three times its depth
  float reach = 3.f;

  bool operator==(const ContactSettings&) const = default;
};

// The pass on a snapshot, as the tests drive it.
struct ContactInput
{
  std::span<const float>    body;  // posed, xyz per cage vertex, in the root's space (cm)
  std::span<const uint32_t> tris;
  // the cage's polygon loops, for its edges; empty: the triangles' edges
  std::span<const int32_t> polyStart, polyVerts;
  std::span<const ColliderRef> colliders;
  float                        bulge = 0.7f, reach = 3.f;
};
struct ContactResult
{
  std::vector<float> positions;  // as `body`, dented
  int                patches = 0;
};
ContactResult resolveContacts(const ContactInput& in);

class Contact
{
public:
  // `bodyMesh` is the figure's body; `root` its root node (the pass runs in its space, centimetres).
  Contact(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root);
  ~Contact();

  // What the body meets, placed in its root's space; asked on the owner's thread as a pass starts.
  std::function<std::vector<ColliderRef>()> colliders;
  // True while a clip plays.
  std::function<bool()> playing;

  const ContactSettings& settings() const { return m_settings; }
  bool                   enabled() const { return m_settings.on; }
  // A pass is due or running.
  bool busy() const;
  // How many dents the last pass made.
  int patches() const { return m_patches; }
  // Bumps whenever busy(), patches() or the settings change, for a status topic.
  int version() const { return m_pass.version(); }
  // The posed body's bounds in the root's space as the last pass found it; empty before one.
  const Box& bodyBounds() const { return m_bounds; }

  // Turning it off puts the channel at weight 0; any other change runs a pass at once (a slider's
  // drag watches the dent). True when the morph set has something to flush.
  bool set(const ContactSettings& settings);
  // Something it reads changed: bones moved, or these meshes' vertices did (only the body counts).
  void changed(const std::vector<int>& meshes, bool nodesMoved);
  // Schedules a pass: a collider near the body moved, came or went.
  void poke();

  // Starts a pass that is due and writes one that finished. Call every frame. True when it wrote
  // (the morph set has vertices to flush).
  bool pump();

private:
  struct Job;
  // what a pass hands back: the cage deltas to write (the inverse skin done on the pool), and its dents
  struct Landed
  {
    std::vector<float> delta;
    int                patches = 0;
  };
  std::shared_ptr<Job> prepare();
  bool                 apply(const Landed* landed);

  SceneGraph&      m_scene;
  morph::MorphSet& m_morphs;
  int              m_bodyMesh;
  int              m_root;
  ChannelWriter    m_writer;
  ContactSettings  m_settings;
  // held at weight 0 while a clip plays
  bool                       m_held    = false;
  int                        m_patches = 0;
  Box                        m_bounds;
  StillPass<Landed>          m_pass;
};

}  // namespace fv::physics
