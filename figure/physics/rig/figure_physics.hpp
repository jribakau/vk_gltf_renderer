#pragma once

// A figure's ragdoll and its Physics mode (docs/physics-plan.md step 2). In Physics mode the body's
// bones are rigid bodies: one per anatomical part (the hip with the pelvis, the abdomen, the chest,
// the neck, the head, each collar, upper arm, forearm, hand, thigh, shin and foot; a G8 twist bone is
// a part of its own, hinged on its one axis), each a capsule fitted to the cage vertices it moves.
// Ball joints keep each part inside its bone's DAZ limits and drive it toward the pose its channels
// say, so everything that writes channels (the Pose tab, presets, a clip, undo, the scene file) sets
// where the motors aim. What the ragdoll reaches is the rig's physical layer
// (pose::PoseBone::physical): the bones show it, the formulas read it, the channels stay the target.
//
// Gravity's torque is fed forward in each drive and the hip is held where it entered, so a pose holds
// exactly and Hold only says how far a push moves a part: at 1 nothing gives, at 0 it is limp. With
// "Let it fall" the hip goes free and nothing is fed forward: it falls as it is.
//
// The Pull is a pin that drags a point of a part along the pointer; let go, the channels take the
// reached pose. Fingers, toes, the face and the eyes keep their channels.
//
// One thread (the figure's owner) calls everything.

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <figure/morph/morph_set.hpp>
#include <figure/physics/rigid/world.hpp>
#include <figure/pose/pose_rig.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics {

struct PhysicsSettings
{
  // Physics mode
  bool on = false;
  // Hold: how hard the motors keep the pose, 0 limp to 1 rigid
  double strength = 0.8;
  bool   gravity  = true;
  // the hip goes free, and nothing holds the pose against gravity
  bool letFall = false;

  bool operator==(const PhysicsSettings&) const = default;
};

class FigurePhysics
{
public:
  // `bodyMesh` is the figure's body (its capsules are fitted to its cage; -1 or no skin: to the bones'
  // segments); `root` its root node: the ragdoll lives in its space, in centimetres.
  FigurePhysics(SceneGraph& scene, pose::PoseRig& rig, const morph::MorphSet* morphs, int bodyMesh, int root);

  // Whether the figure has a ragdoll: a hip, and parts under it.
  bool valid() const { return !m_parts.empty(); }
  const PhysicsSettings& settings() const { return m_settings; }
  // Turning Physics on starts the parts where the pose is, at rest; off folds what they reached into
  // the channels. True when the pose moved (the bones need applying).
  bool set(const PhysicsSettings& settings);

  // Gravity in the root's space (cm/s^2), the ground (none: no ground) and what else the body meets.
  void                                       setGravity(const Vec3& g);
  void                                       setGround(const std::optional<Plane>& ground);
  std::function<std::vector<ColliderRef>()> colliders;

  // The Pull: a ray in the root's space; true when it meets a part, the pull then holding that point.
  // `depth` is how far along the ray it met it, the depth a drag keeps.
  bool grab(const Vec3& origin, const Vec3& dir, double& depth);
  void drag(const Vec3& target);
  bool pulling() const { return m_pull >= 0; }
  // The pull ends: the channel values that pose the reached pose, for the caller to set as targets
  // (the motors then hold it where it is).
  std::vector<pose::ChannelValue> release();
  // Where each part is, as channel values: the reached pose (leaving, a scene file, undo).
  std::vector<pose::ChannelValue> reached() const;
  // The physical layer as a scene file keeps it: bone id/kind/axis -> value.
  std::vector<std::pair<std::string, double>> physicalValues() const;
  // Takes a physical layer back (undo, a scene file), at rest, asleep where it holds.
  void restorePhysical(const std::vector<std::pair<std::string, double>>& values);

  // A frame: simulates `seconds` while awake and writes the reached pose to the rig. True when bones
  // moved.
  bool step(double seconds);
  // Simulating: something moved it, or it hasn't come to rest.
  bool awake() const { return m_settings.on && (!m_world.asleep() || m_pull >= 0); }
  // An input changed (a prop near it moved, the figure was placed elsewhere): it simulates again.
  void wake();
  // Bumps when the settings, awake() or pulling() change, for a status topic.
  int version() const { return m_version; }

  struct Part
  {
    int poseBone = -1;  // its bone in PoseRig::bones()
    int node     = -1;
    int parent   = -1;  // the parent part
    int joint    = -1;  // its joint in world().joints; -1 for the hip
    // its centre of mass in its bone's frame (cm)
    Vec3 com;
    // the bones merged into it (its own first): the pelvis into the hip, spine2 into spine1, ...
    std::vector<int> members;
    double           subtreeMass = 0;
    // the rotation totals it was last written at (degrees), for the Euler branch
    std::array<double, 3> last{};
  };
  const std::vector<Part>& parts() const { return m_parts; }
  const RigidWorld&        world() const { return m_world; }
  // the part a pose bone is, or -1
  int partOf(int poseBone) const;

private:
  void build();
  // puts the bodies where the bones are now, still
  void placeBodies();
  // the joints' frames, boxes, targets and stiffness, from the rig as it is now
  void updateJoints();
  void feedForward();
  // allowances for overlaps in the pose now and in the target pose
  void allowOverlaps();
  bool writeBack();
  Mat4 boneWorld(int node) const;
  // the target pose's world for a part's bone, from the bones' channels
  Mat4 targetWorld(int node) const;
  std::vector<double> targets() const;

  SceneGraph&            m_scene;
  pose::PoseRig&         m_rig;
  const morph::MorphSet* m_morphs;
  int                    m_bodyMesh, m_root;
  PhysicsSettings        m_settings;
  RigidWorld             m_world;
  std::vector<Part>      m_parts;
  std::vector<int>       m_partOfNode;
  int                    m_restVersion = -1;
  int                    m_hipPin = -1, m_pull = -1;
  Vec3                   m_gravity{0, -980, 0};
  std::optional<Plane>   m_ground;
  std::vector<double>    m_lastTargets;
  double                 m_sinceInput = 0;
  int                    m_version    = 0;
  bool                   m_wasAwake   = false;
};

}  // namespace fv::physics
