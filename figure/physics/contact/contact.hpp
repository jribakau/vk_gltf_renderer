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
// Another person's body is met the same way, without its own dent, and each takes its share of the
// overlap by softness (step 3), so the two passes don't chase each other. So are the body's own parts
// (step 10), each giving way by half so the two meet between: a vertex meets the skin of another part
// only when the way to it along the skin is long, so the crease of a bent knee or elbow, which DAZ's
// own corrective shapes look after, is left as it is.
// While what it meets is driven (a clip plays, another figure's ragdoll moves) the channel is at
// weight 0 and no pass runs: a dent where a hand used to be looks wrong.
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
  // the body's own parts press on one another: a kneel's calf on its thigh, an arm on the chest
  bool self = true;
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
  // per vertex, its part of the body (bodyParts): with them the parts press on one another
  std::span<const uint8_t> parts;
  // the body unposed, as `body`: the ways along the skin between parts are measured on it, since a
  // bent joint squeezes or stretches its skin; empty, on `body`
  std::span<const float> rest;
};
struct ContactResult
{
  std::vector<float> positions;  // as `body`, dented
  int                patches = 0;
  // vertices pressed into another part of the body, as the pass found them
  int pressed = 0;
};
ContactResult resolveContacts(const ContactInput& in);

// The parts of a body that press on one another. A vertex's is its strongest bone's, and a bone's its
// own or its nearest ancestor's: the collar and the bones of no limb (the hip, the spine, the
// pectorals) are the torso's, the neck and the face's bones the head's, the fingers the hand's.
namespace part {
constexpr uint8_t kTorso = 0, kHead = 1, kLUpperArm = 2, kRUpperArm = 3, kLForearm = 4, kRForearm = 5, kLHand = 6, kRHand = 7,
                  kLThigh = 8, kRThigh = 9, kLShin = 10, kRShin = 11, kLFoot = 12, kRFoot = 13;
}
uint8_t              partOfBone(const SceneGraph& scene, int node);
std::vector<uint8_t> bodyParts(const SceneGraph& scene, const Mesh& mesh);

// A figure's meshes as posed now, with their morphs and skin: xyz per cage vertex in the space of its
// root node (cm), one list per mesh asked, empty for one without a cage. How a DAZ prop (a chair) is
// met by the bodies near it.
// `without` is a hidden channel left out: a person's own dent, when it is met by another body.
std::vector<std::vector<float>> posedCages(const SceneGraph& scene, const morph::MorphSet& morphs, std::span<const int> meshes, int root,
                                           const std::string& without = {});

// A body's share of its overlap with another person's: its softness against theirs, half each when
// both are 0, all of it when the other doesn't dent (off, or null).
double overlapShare(const ContactSettings& self, const ContactSettings* other);

class Contact
{
public:
  // `bodyMesh` is the figure's body; `root` its root node (the pass runs in its space, centimetres).
  Contact(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root);
  ~Contact();

  // What the body meets, placed in its root's space; asked on the owner's thread as a pass starts.
  std::function<std::vector<ColliderRef>()> colliders;
  // True while what it meets is driven: a clip plays, or another figure near it moves on its own.
  std::function<bool()> driven;

  const ContactSettings& settings() const { return m_settings; }
  int                    bodyMesh() const { return m_bodyMesh; }
  bool                   enabled() const { return m_settings.on; }
  // A pass is due or running.
  bool busy() const;
  // How many dents the last pass made.
  int patches() const { return m_patches; }
  // The channel is at weight 0 while what it meets is driven.
  bool held() const { return m_held; }
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
  // held at weight 0 while driven
  bool                       m_held    = false;
  int                        m_patches = 0;
  Box                        m_bounds;
  // the body's parts, made the first time a pass presses them on one another
  std::vector<uint8_t>       m_parts;
  StillPass<Landed>          m_pass;
};

}  // namespace fv::physics
