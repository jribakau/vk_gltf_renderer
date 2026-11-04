#pragma once

// A Timeline clip played on one of our figures in Physics mode (docs/physics-plan.md step 16). Each
// controller the clip keys, or its Pose holds still, pins the part of our body its bone is in, matched
// by anatomy (wardrobe::parseBone: Genesis 2's lHand is Genesis 9's l_hand): the hip and the pelvis
// hold the hip, which the clip then aims; the chest, the head, the hands, the knees, the feet and
// whatever else it holds are pins, sprung as VaM springs them on our masses. The toes have no bodies
// of their own and are left. Between the pins the body is held at a Hold of the clip's own, soft as
// VaM's joints are next to its holds (Options::hold), toward the figure's channels. VaM's own joint
// drives can take that Hold's place (Options::jointDrives), but on our masses they are far stiffer
// against the pins than in VaM, so they are off.
//
// A clip that begins (begin()) blends from where the body is to its targets over its first seconds,
// as VaM starts it from the Pose the clip opens on: jumped to at once, a figure standing and a clip
// kneeling fold a leg the wrong way and keep it there.
//
// Space. Timeline's positions are local to the Person's main control, which the clip's Pose places.
// On our figure the floor under that control is our figure's root, facing as the Person faces:
// Unity's left-handed metres mirrored across x into our right-handed centimetres (VaM's left hand, at
// -x, is our left at +x), and scaled by the legs' length, since VaM's bodies are Genesis 2.
// Genesis 2 rests in a T-pose with its bones along the Person's axes, so a controller's rotation is
// its bone's turn from that pose. Ours rest in an A-pose: a limb's turn is taken on from its bone
// pointing the way VaM's does at rest.

#include <map>
#include <string>
#include <vector>

#include <figure/physics/rig/figure_physics.hpp>
#include <figure/physics/vam/timeline.hpp>

namespace fv::physics::vam {

// Unity's Euler angles (degrees), as Quaternion.Euler: z, then x, then y.
Quat unityEuler(const Vec3& degrees);
// Unity's space to ours: metres to centimetres, mirrored across x.
Vec3 toOurs(const Vec3& unityMetres);
Quat toOurs(const Quat& unity);

// A VaM spring (stiffness, damper, force limit) on our part's mass, which keeps its natural frequency
// and damping ratio: each scaled by m_ours / m_vam, from VaM's units to ours. Linear: N/m, N s/m, N to
// dyne/cm, dyne s/cm, dyne. Angular: N m/rad, N m s/rad, N m to dyne cm/rad, dyne cm s/rad, dyne cm.
struct Spring
{
  double stiffness = 0, damper = 0, maxForce = 0;
};
Spring onOurMass(const Spring& vam, double vamKg, double ourGrams, bool angular);

class Player
{
public:
  // the Person's clip `clip`
  Player(Person person, size_t clip);

  // Measured on Genesis 9 with the inspected scene's ANIMATION A (VaM.theInspectedClipOnGenesis9): at
  // these, after the blend, the hands end 1.6 cm from their targets, the knees 0.9 to 1.5, the feet 2
  // and 9.7, the head 0.4; VaM's own body is 1.4 to 3.7 cm from them. VaM's joint drives on our masses
  // leave the hands 21 cm off, and 9 on the whole body's; the Pose tab's default Hold, 12.6.
  struct Options
  {
    double hold = 0.3;
    // VaM's springs onto ours by each controller's own mass (each keeps its frequency), or all by the
    // whole body's (the balance between them kept)
    bool wholeBody = false;
    // the controllers' joint drives in place of the Hold
    bool jointDrives = false;
    // how long a clip that begins takes to come in (s)
    double blendIn = 2;
  };
  Options options;

  // The clip begins at `time` with the body where `physics` has it: it blends from there. jump()
  // drops a blend (a seek jumps).
  void begin(const FigurePhysics& physics, const pose::PoseRig& rig, const SceneGraph& scene, float time);
  void jump() { m_blend.clear(); }

  const Person& person() const { return m_person; }
  const Clip&   clip() const { return m_person.clips[m_clip]; }

  // What holds the figure `physics` is the ragdoll of at `time` (s, into the clip).
  FigurePhysics::Drive at(float time, const FigurePhysics& physics, const pose::PoseRig& rig, const SceneGraph& scene);

  // Where each controller that holds the body aims, in the figure's root space, as the last at() had
  // it: the bone's origin and its turn.
  struct Target
  {
    std::string controller;
    int         part = -1;  // 0: the hip
    bool        move = false, turn = false;
    Vec3        at;
    Quat        rotation;
  };
  const std::vector<Target>& targets() const { return m_targets; }
  // VaM's legs over ours (the scale its positions take on), 1 where either has none
  double scale() const { return m_scale; }
  // The controllers that hold something and that our figure has no part for, by name.
  const std::vector<std::string>& unmatched() const { return m_unmatched; }

private:
  struct Bound
  {
    std::string  name;
    const Track* track = nullptr;  // none: held where the Pose has it
    Controller   settings;
    int          part = -1;
    bool         hip = false, pelvis = false;
    bool         move = false, turn = false;
    // our bone's turn when it points as VaM's does at rest (VaM's identity)
    Quat rest;
  };
  void bind(const FigurePhysics& physics, const pose::PoseRig& rig, const SceneGraph& scene);
  // a controller's settings: the clip's Pose, else the scene's
  Controller settingsOf(const std::string& name) const;
  const Pose& pose() const { return clip().pose ? *clip().pose : m_person.scene; }

  Person                   m_person;
  size_t                   m_clip = 0;
  std::vector<Bound>       m_bound;
  std::vector<FigurePhysics::Drive::Joint> m_joints;
  std::vector<std::string> m_unmatched;
  std::vector<Target>      m_targets;
  double                   m_scale = 1;
  // where each bound controller's part was as the clip began (the hip's under "hip"), and when
  std::map<std::string, std::pair<Vec3, Quat>> m_blend;
  float                                        m_blendFrom = 0;
  // the whole body's mass, ours (g) and VaM's controllers' (kg)
  double m_ourGrams = 0, m_vamKg = 0;
  // VaM's pelvis origin from its hip, in our space at our scale
  Vec3 m_pelvisOffset;
  int  m_restVersion = -2;
  size_t m_parts     = 0;
};

}  // namespace fv::physics::vam
