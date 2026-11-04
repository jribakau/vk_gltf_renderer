#pragma once

// Secondary motion (docs/physics-plan.md step 5): the chest's soft bones (Genesis 8's lPectoral and
// rPectoral, Genesis 9's l_pectoral and r_pectoral) as bodies on sprung joints to the bone they hang
// from, so they answer that bone's motion: a step's bounce, a turn's sway, and back to rest on the
// pose their channels and formulas give. Gravity is fed forward, so at rest they are where DAZ has
// them. In either mode: the skeleton or the ragdoll moves the chest, this follows it.
//
// Bounce sets how loose they are (the spring's frequency, from 8 Hz firm to 1.5 Hz), Damping how soon
// they settle (the damping ratio). What they reach is the rig's physical layer, for these bones alone;
// once still they show their channels again and nothing is written.
//
// One thread (the figure's owner) calls everything.

#include <array>
#include <vector>

#include <figure/physics/rigid/world.hpp>
#include <figure/pose/pose_rig.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics {

struct SecondarySettings
{
  bool   on      = false;
  double bounce  = 0.5;
  double damping = 0.3;

  bool operator==(const SecondarySettings&) const = default;
};

class Secondary
{
public:
  // `root` is the figure's root node: the bodies live in its space, in centimetres.
  Secondary(SceneGraph& scene, pose::PoseRig& rig, int root);

  // Whether the figure has soft bones.
  bool                     valid() const { return !m_soft.empty(); }
  const SecondarySettings& settings() const { return m_settings; }
  // Off, the bones show their channels again. True when bones moved.
  bool set(const SecondarySettings& settings);
  // Gravity in the root's space (cm/s^2).
  void setGravity(const Vec3& g) { m_gravity = g; }

  // A frame, after whatever posed the chest: the bones they hang from as they are now, `seconds`
  // simulated, the reached pose written. True when bones moved.
  bool step(double seconds);
  bool awake() const { return m_settings.on && !m_soft.empty() && !m_world.asleep(); }
  // Bumps when the settings or awake() change, for a status topic.
  int version() const { return m_version; }

  const RigidWorld& world() const { return m_world; }
  // The soft bones (PoseRig::bones() indices), in the world's body order.
  std::vector<int> bones() const;

private:
  struct Soft
  {
    int  poseBone = -1, node = -1, parentNode = -1;
    Vec3 com;  // in its bone's frame (cm)
    // the rotation totals it was last written at (degrees), for the Euler branch
    std::array<double, 3> last{};
  };
  void build();
  Mat4 boneWorld(int node) const;
  // the bodies where the bones are now, still
  void place();
  // the joints from the rig as it is now: frames, boxes, targets, stiffness; the parents' worlds
  void aim(const std::vector<Mat4>& parents);
  bool writeBack();

  SceneGraph&       m_scene;
  pose::PoseRig&    m_rig;
  int               m_root;
  SecondarySettings m_settings;
  RigidWorld        m_world;
  std::vector<Soft> m_soft;
  std::vector<int>  m_owned;
  Vec3              m_gravity{0, -980, 0};
  // the bones they hang from at the last frame, and the targets then
  std::vector<Mat4>   m_parents;
  std::vector<double> m_targets;
  int                 m_restVersion = -1;
  int                 m_version     = 0;
  bool                m_wasAwake    = false;
  bool                m_written     = false;
};

}  // namespace fv::physics
