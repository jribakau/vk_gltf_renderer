#pragma once

// Rigid bodies on the physics core (docs/physics-plan.md, "Solver"): XPBD with small fixed substeps
// (Macklin et al. 2016; Mueller et al. 2020, "Detailed rigid body simulation with extended position
// based dynamics"). What a ragdoll is made of: bodies with a capsule each, ball joints held inside
// DAZ's Euler box and driven toward a target rotation, pins that hold a body point (the hip, the
// pull) and contacts with the ground, with each other and with props.
//
// Units are the figure's: centimetres, grams, seconds, so a force is in dyne (1 N = 1e5 dyne) and a
// torque in dyne cm. Everything is in one space, the figure root's.
//
// A frame takes whole substeps of a fixed length and carries the rest of its time to the next, so the
// stiffness doesn't follow the frame rate, and the same inputs give the same steps, bit for bit.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/physics/collide/triangle_set.hpp>

namespace fv::physics {

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// A capsule: the points within `radius` of the segment a-b, in its body's frame (from the centre of
// mass).
struct Capsule
{
  Vec3   a, b;
  double radius = 0;
};

struct RigidBody
{
  // the centre of mass and the orientation, now and at the substep's start; velocities
  Vec3 x, xPrev, v, w;
  Quat q, qPrev;
  // 0: kinematic (moved by its owner, never by the solver)
  double invMass = 0;
  // the inverse inertia, diagonal in the frame `inertiaFrame` turns the body's frame into
  Vec3 invInertia;
  Quat inertiaFrame;
  Capsule shape;
  // forces and torques acting all this frame (dyne, dyne cm, world), besides gravity
  Vec3 force, torque;
  // bodies with the same group never collide (a body and its joint's other side, set by the owner)
  std::vector<int> noCollide;

  double mass() const { return invMass > 0 ? 1 / invMass : 0; }
  // I^-1 v in the world
  Vec3 applyInvInertia(const Vec3& v) const;
  Vec3 toWorld(const Vec3& local) const { return x + rotate(local, q); }
};

// Euler angles (radians) in a three order, the branch nearest `near`: (a, b, c) and
// (a + 180, 180 - b, c + 180) with b the middle axis, each any number of turns round.
std::array<double, 3> eulerNear(const Quat& q, std::string_view order, const std::array<double, 3>& near);

// A ball joint: the point `anchorA` of body a and `anchorB` of body b stay together (in their
// frames). The joint's frame on each side (`frameA`, `frameB`) gives the rotation the limits and the
// drive read: R = frameA^-1 qa^-1 qb frameB, as Euler angles in `order`. Body a may be -1, the world
// (its anchor and frame then in the world).
struct Joint
{
  int         a = -1, b = -1;
  Vec3        anchorA, anchorB;
  Quat        frameA, frameB;
  std::string order = "XYZ";
  // each Euler axis's box (radians); lo > hi: free
  std::array<double, 3> lo{1, 1, 1}, hi{0, 0, 0};
  // the drive: the rotation R it pulls toward, and how hard (compliance, rad / (dyne cm)); a
  // compliance below 0 has no drive
  Quat   target;
  double compliance = -1;
  // the drive's damping, as a time (s): its damper is driveDamping / compliance, in the solve, so it
  // reaches the whole chain the drive turns. About 2 / its frequency damps it critically.
  double driveDamping = 0;
  // joint friction: how much of the relative angular velocity goes each second (1/s), on the two
  // bodies alone (it barely reaches a chain's swing; a limp joint has nothing else)
  double damping = 0;
  // the torque the drive can give at most (dyne cm); 0: no limit
  double maxTorque = 0;
  // a torque the drive gives b and takes from a besides its spring (dyne cm, world): gravity's
  // compensation, set by the owner. It moves the spring's rest by compliance times it, so a drive
  // with no compliance gives none (it holds anyway)
  Vec3 feedForward;
  // the Euler angles where the joint was last, for the branch
  std::array<double, 3> euler{};
};

// A pin: a body's point `local` pulled to `target` (world), and its rotation to `rotation` when
// `turn`; compliant (cm/dyne, rad/(dyne cm)), with a force limit (dyne; 0: none).
struct Pin
{
  int    body = -1;
  Vec3   local, target;
  bool   move = true, turn = false;
  Quat   rotation;
  double compliance = 0, angularCompliance = 0, maxForce = 0, maxTorque = 0;
  // how much of the body's velocity (relative to the target, which stands still) goes each second
  double damping = 0;
  // a force and torque the pin gives besides its spring (the weight it carries), as a joint's
  Vec3 feedForward, feedForwardTorque;
  bool on = true;
};

// The ground: the points where dot(n, p) < offset are under it.
struct Plane
{
  Vec3   n{0, 1, 0};
  double offset = 0;
};

struct Material
{
  double staticFriction = 0.8, dynamicFriction = 0.6;
  // a torque limit against rolling, as a share of the normal force times the radius
  double rolling = 0.05;
};

struct WorldSettings
{
  // the substep length (s) and the most a frame simulates (s): a hitch slows the motion, not kicks it
  double h = 1.0 / 600, maxFrame = 1.0 / 20;
  // Newton steps of the joints and the pins each substep, solved exactly over the tree they make
  int iterations = 2;
  Vec3   gravity{0, -980, 0};
  // velocity caps (cm/s, rad/s): a safety net, not damping
  double maxSpeed = 3000, maxSpin = 50;
  // asleep below these (cm/s at any capsule's end, rad/s) for `restTime` of simulated time
  double restSpeed = 0.3, restSpin = 0.3 * kDeg, restTime = 0.2;
  Material material;
};

class RigidWorld
{
public:
  WorldSettings settings;

  std::vector<RigidBody> bodies;
  std::vector<Joint>     joints;
  std::vector<Pin>       pins;
  std::vector<Plane>     planes;
  // props, in their own space with the matrix into this world (step 1's colliders)
  std::vector<ColliderRef> props;
  // pairs of bodies that overlapped as they came in, and by how much they may still (cm); a pair is
  // allowed what it had, and it only shrinks as they part
  struct Allowance
  {
    int    a = -1, b = -1;
    double depth = 0;
  };
  std::vector<Allowance> allowances;

  // Simulates `seconds` (clamped to settings.maxFrame) in whole substeps, carrying the rest: how many
  // it took.
  int step(double seconds);
  // One substep of settings.h.
  void substep();
  // The time carried to the next frame.
  double carried() const { return m_carry; }
  void   resetClock() { m_carry = 0; }

  // Whether the bodies moved less than the rest speed for the rest time; reset by wake().
  bool asleep() const { return m_asleep; }
  void wake();
  // Asleep now, still (a forced sleep, or a pose taken back at rest).
  void sleep();
  // The fastest a capsule's end moved over the last frame's substeps (cm/s).
  double lastSpeed() const { return m_lastSpeed; }

  // Every pair that overlaps now is allowed that much (entering, or after a target change).
  void allowOverlaps();
  // How deep two bodies' capsules overlap now (cm, 0 when apart).
  double overlap(int a, int b) const;

private:
  struct Contact
  {
    int    a = -1, b = -1;  // b -1: something fixed (the ground, a prop)
    Vec3   ra, rb;          // the contact points in each body's frame (b's: world, when fixed)
    Vec3   n;               // from b to a, the way a is pushed
    double depth   = 0;     // how far a must move along n
    double allowed = 0;
    double lambdaN = 0, lambdaT = 0;
  };
  void predict(double h);
  void solveAnchor(Joint& j);
  void solveDrive(Joint& j, double h);
  void solveLimits(Joint& j);
  void solvePin(Pin& p, double h);
  // The joints and the pins with no force limit, over the tree the bodies and they make, solved
  // exactly (Baraff 1996, "Linear-time dynamics using Lagrange multipliers"): one Newton step of
  // their projection. What would close a loop is marked for the sweep after it.
  void solveTree(std::vector<uint8_t>& sweptJoint, std::vector<uint8_t>& sweptPin);
  void collide();
  void solveContact(Contact& c);
  void updateVelocities(double h);
  void velocityPass(double h);

  // positional: move b's point toward a's by `c` along n (a and b may be -1, the world)
  double applyPositional(int a, int b, const Vec3& pa, const Vec3& pb, const Vec3& n, double c, double alpha, double& lambda, double maxLambda);
  // angular: turn b relative to a by `theta` about n
  double applyAngular(int a, int b, const Vec3& n, double theta, double alpha, double& lambda, double maxLambda);
  void   applyPositionalImpulse(int a, int b, const Vec3& pa, const Vec3& pb, const Vec3& p);

  std::vector<Contact> m_contacts;
  // the tree solve's multipliers this substep, six per joint and per pin
  std::vector<double>  m_lambdaJoint, m_lambdaPin;
  double               m_carry = 0, m_still = 0, m_lastSpeed = 0;
  bool                 m_asleep = false;
  double               m_h      = 1.0 / 600;
};

// The closest points of segments p1-q1 and p2-q2.
void closestSegments(const Vec3& p1, const Vec3& q1, const Vec3& p2, const Vec3& q2, Vec3& c1, Vec3& c2);

// The mass, inverse inertia and its frame of a solid capsule of density `rho` (g/cm^3).
void setCapsuleMass(RigidBody& body, double rho = 1);

}  // namespace fv::physics
