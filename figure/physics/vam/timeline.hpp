#pragma once

// Virt-A-Mate scenes and the Timeline clips in them (docs/physics-plan.md step 16), as much of them as
// playing a Person's clip on one of our figures takes. Timeline (AcidBubbles' VamTimeline, open
// source) keys a Person's controllers, each as seven curves: its position (X, Y, Z) and its rotation
// (RotX, RotY, RotZ, RotW) in the Person's own space, which is Unity's (left-handed, Y up, metres)
// and local to the Person's main control. VaM's physics body springs toward those targets. A clip's
// Pose is the Person as the clip starts: where its main control stands, each controller's settings
// (whether it holds the body, its springs, its mass) and its bones.
//
// Read as Timeline's serializer writes them (AtomAnimationSerializer), and played as its
// BezierAnimationCurve plays them. Triggers, FloatParams (morphs, plugin values), audio and the
// tracks of other atoms are not read.

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <tinygltf/json.hpp>

#include <figure/math/xform.hpp>

namespace fv::physics::vam {

// Timeline's curve types (CurveTypeValues).
enum CurveType : int
{
  eLeaveAsIs    = 0,
  eFlat         = 1,
  eLinear       = 2,
  eSmoothLocal  = 3,
  eBounce       = 4,
  eLinearFlat   = 5,
  eFlatLinear   = 6,
  eCopyPrevious = 7,
  eConstant     = 8,
  eFlatLong     = 9,
  eSmoothGlobal = 10,
};

// A key of a curve (Timeline's BezierKeyframe). `in` and `out` are the Bezier control points either
// side of it, values like `value`, which its curve type sets (Curve::compute).
struct Key
{
  float time = 0, value = 0;
  int   curve = eSmoothLocal;
  float in = 0, out = 0;
};

// One value over a clip's time.
struct Curve
{
  std::vector<Key> keys;

  // Keys at the clip's start and end where it has none (AddEdgeFramesIfMissing), as Timeline adds
  // them to a curve it reads.
  void addEdges(float length, int curveType);
  // The control points the curve types give (ComputeCurves). `loop`: the keys wrap round, the first
  // key's neighbour before it being the one before the last. Smooth (Global), which Timeline solves
  // over the whole curve, is taken as Smooth (Local).
  void compute(bool loop);
  // The value at `time` (Evaluate): a cubic Bezier between keys, or a line, or a step. A curve with
  // one key is that key's value.
  float at(float time) const;
};

// A controller's keys in a clip.
struct Track
{
  std::string controller;
  // which of the two it keys (TargetsPosition, TargetsRotation) and moves (ControlPosition, ControlRotation)
  bool                 position = true, rotation = true;
  std::array<Curve, 3> xyz;
  std::array<Curve, 4> rot;  // x, y, z, w

  Vec3 positionAt(float time) const;
  // normalized: Timeline makes a quaternion of the four curves' values as they are
  Quat rotationAt(float time) const;
};

// A FreeControllerV3's settings as a scene or a clip's Pose writes them, in VaM's units: N/m, N s/m
// and N for the position; N m/rad, N m s/rad and N m for the rotation and the joint drive; kg. A
// setting the file leaves out keeps VaM's default.
struct Controller
{
  std::string positionState = "On", rotationState = "On";
  double      mass               = 1;
  double      holdPositionSpring = 2000, holdPositionDamper = 35, holdPositionMaxForce = 1000;
  double      holdRotationSpring = 100, holdRotationDamper = 5, holdRotationMaxForce = 1000;
  // the spring its body's joint has toward VaM's own rest pose
  double jointDriveSpring = 0, jointDriveDamper = 0, jointDriveMaxForce = 0;
  // where it stands in the Person's space: metres, and Unity's Euler angles (degrees)
  std::optional<Vec3> position, rotation;

  // Whether it holds its body's position and rotation: On, Hold, Lock and Comply do (Comply as On).
  // Off and Following don't, nor what VaM links to another body (ParentLink, PhysicsLink, LookAt),
  // which isn't read.
  bool holdsPosition() const;
  bool holdsRotation() const;
};

// A Person as it stands: where its main control is (the root of its space, in the scene's), its
// controllers, and its bones' local positions (metres, in their parent bone's frame). Genesis 2's
// bones are aligned with the Person's axes at rest, in a T-pose, so those give its proportions.
struct Pose
{
  std::optional<Vec3>               rootPosition, rootRotation;  // metres; Euler degrees
  std::map<std::string, Controller> controllers;
  std::map<std::string, Vec3>       bones;
};

struct Clip
{
  std::string        name;
  float              length = 0;
  bool               loop   = true;
  std::vector<Track> tracks;  // the Person's own controllers
  // the Person as the clip starts, where the clip keeps one (Timeline applies it then)
  std::optional<Pose> pose;
};

struct Person
{
  std::string       id;
  Pose              scene;  // as the scene has it
  std::vector<Clip> clips;
};

struct Scene
{
  std::vector<Person> people;
};

// A curve's keys as Timeline's serializer writes them. From SerializeVersion 231 each is a string: a
// letter, 'A' plus 1 when a value follows and plus 2 when a curve type does, then the time, the value
// and the curve type as little-endian hex (float32, float32, one byte). What a key leaves out is the
// previous key's (before the first: 0 and Smooth (Local)), and a key at the time of the one before it
// is dropped. Older files' encodings and objects ({t, v, c, i, o}) are read too. Throws on a key it
// can't read.
std::vector<Key> decodeKeys(const nlohmann::json& keys, int version);

// The Person atoms of a VaM scene (its .json) and their Timeline clips, each clip's curves with their
// edges, the loop's last key and the rotations' signs made continuous as Timeline does before it
// plays, and their control points computed. Throws when the text isn't a VaM scene.
Scene readScene(const nlohmann::json& scene);
Scene readScene(const std::string& text);

}  // namespace fv::physics::vam
