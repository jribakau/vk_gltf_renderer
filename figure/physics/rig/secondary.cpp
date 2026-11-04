#include <figure/physics/rig/secondary.hpp>

#include <algorithm>
#include <cmath>
#include <regex>

#include <figure/io/timing.hpp>

namespace fv::physics {

namespace {

using pose::ChannelKind;

constexpr double kDeg = 3.14159265358979323846 / 180;
constexpr double kPi  = 3.14159265358979323846;
// a move of the bone they hang from past this in one frame is a jump (a preset, an undo, a scene
// opening): they go with it, at rest, rather than be flung
constexpr double kJumpMove = 30;               // cm
constexpr double kJumpTurn = 45 * kDeg;        // radians
// the spring's frequency at Bounce 0 and 1 (Hz)
constexpr double kFirm = 8, kLoose = 1.5;

Vec3 originOf(const Mat4& m)
{
  return {m.e[12], m.e[13], m.e[14]};
}

bool isSoft(const std::string& name)
{
  static const std::regex soft(R"(^(l|r)_?pectoral$)", std::regex::icase);
  return std::regex_match(name, soft);
}

bool hasRotation(const pose::PoseBone& pb)
{
  return std::any_of(pb.axes.begin(), pb.axes.end(), [](const pose::Axis& a) { return a.kind == ChannelKind::eRotation; });
}

// The angle a rotation turns by (radians).
double angleOf(const Quat& q)
{
  return 2 * std::acos(std::clamp(std::abs(q.w), 0.0, 1.0));
}

}  // namespace

Secondary::Secondary(SceneGraph& scene, pose::PoseRig& rig, int root)
    : m_scene(scene)
    , m_rig(rig)
    , m_root(root)
{
  build();
}

std::vector<int> Secondary::bones() const
{
  return m_owned;
}

Mat4 Secondary::boneWorld(int node) const
{
  // in the root's space: the root node's placement and its centimetre scale divided out
  return inverse(m_scene.world(m_root)) * m_scene.world(node);
}

void Secondary::build()
{
  m_world       = RigidWorld{};
  m_restVersion = m_rig.restVersion();
  m_soft.clear();
  m_owned.clear();
  m_parents.clear();
  m_targets.clear();
  const auto& bones = m_rig.bones();
  for(size_t i = 0; i < bones.size(); i++)
  {
    const pose::PoseBone& pb = bones[i];
    if(!isSoft(pb.name) || !hasRotation(pb) || pb.bone < 0)
      continue;
    const Node& n = m_scene.nodes[size_t(pb.bone)];
    if(n.parent < 0 || !n.joint)
      continue;
    Soft s;
    s.poseBone   = int(i);
    s.node       = pb.bone;
    s.parentNode = n.parent;
    // a ball on the bone's middle, in its frame (at rest the root's, moved to the joint centre)
    const Vec3   center{n.joint->center[0], n.joint->center[1], n.joint->center[2]};
    const Vec3   end{n.joint->endPoint[0], n.joint->endPoint[1], n.joint->endPoint[2]};
    const double l = length(end - center);
    s.com          = l > 1e-3 ? (end - center) * 0.5 : Vec3{0, 0, 4};
    RigidBody b;
    b.shape     = {{}, {}, std::clamp(0.4 * l, 1.5, 8.0)};
    b.noCollide = {1};
    setCapsuleMass(b);
    Joint j;
    j.b       = int(m_world.bodies.size());
    j.order   = pb.order;
    j.anchorB = s.com * -1.0;
    m_world.bodies.push_back(b);
    m_world.joints.push_back(j);
    m_owned.push_back(int(i));
    m_soft.push_back(s);
  }
  m_world.sleep();
}

void Secondary::aim(const std::vector<Mat4>& parents)
{
  const double f     = kFirm + (kLoose - kFirm) * std::clamp(m_settings.bounce, 0.0, 1.0);
  const double omega = 2 * kPi * f;
  const double zeta  = std::clamp(m_settings.damping, 0.02, 1.0);
  for(size_t i = 0; i < m_soft.size(); i++)
  {
    const Soft&           s  = m_soft[i];
    const pose::PoseBone& pb = m_rig.bones()[size_t(s.poseBone)];
    Joint&                j  = m_world.joints[i];
    const RigidBody&      b  = m_world.bodies[i];
    // the bone it hangs from is the world's here
    j.a       = -1;
    j.frameA  = rotationOf(parents[i]) * pb.orient;
    j.frameB  = pb.orient;
    j.anchorA = parents[i].transformPoint(m_rig.targetPosition(pb));
    std::array<double, 3> total{};
    for(int a = 0; a < 3; a++)
    {
      const double      d  = pb.driven[size_t(ChannelKind::eRotation)][size_t(a)];
      const pose::Axis* ax = pb.axis(ChannelKind::eRotation, a);
      total[size_t(a)]     = pb.posed(ChannelKind::eRotation, a) + d;
      double lo = d, hi = d;
      if(ax && ax->clamped)
        lo = ax->min + d, hi = ax->max + d;
      else if(ax)
        lo = std::min(-30.0 + d, total[size_t(a)]), hi = std::max(30.0 + d, total[size_t(a)]);
      j.lo[size_t(a)] = lo * kDeg;
      j.hi[size_t(a)] = hi * kDeg;
    }
    j.target = quatFromEuler(total[0] * kDeg, total[1] * kDeg, total[2] * kDeg, pb.order);
    // the spring for the frequency asked, about the joint: the ball's own inertia and its lever
    const double m       = b.mass();
    const double ownSpin = b.invInertia.x > 0 ? 1 / b.invInertia.x : 0;
    const double inertia = std::max(1e-6, ownSpin + m * dot(s.com, s.com));
    j.compliance         = 1 / (inertia * omega * omega);
    j.driveDamping       = 2 * zeta / omega;
    j.damping            = 0;
    j.maxTorque          = 0;
    // gravity's torque about the joint, given back by the drive: at rest it is where DAZ has it
    j.feedForward = cross(b.x - j.anchorA, m_gravity * m) * -1.0;
  }
}

void Secondary::place()
{
  // where the channels put each bone under its parent as it is now
  for(size_t i = 0; i < m_soft.size(); i++)
  {
    Soft&                 s  = m_soft[i];
    const pose::PoseBone& pb = m_rig.bones()[size_t(s.poseBone)];
    const Mat4            w  = boneWorld(s.parentNode) * compose(m_rig.targetPosition(pb), m_rig.targetRotation(pb), m_scene.nodes[size_t(s.node)].scale);
    RigidBody&            b  = m_world.bodies[i];
    b.q = b.qPrev = rotationOf(w);
    b.x = b.xPrev = originOf(w) + rotate(s.com, b.q);
    b.v = b.w = {};
    for(int a = 0; a < 3; a++)
      s.last[size_t(a)] = pb.posed(ChannelKind::eRotation, a) + pb.driven[size_t(ChannelKind::eRotation)][size_t(a)];
  }
}

bool Secondary::set(const SecondarySettings& settings)
{
  const SecondarySettings old = m_settings;
  m_settings                  = settings;
  if(settings != old)
    m_version++;
  if(settings.on && !old.on && valid())
  {
    // still where the pose has them, until the chest moves
    if(m_rig.restVersion() != m_restVersion)
      build();
    place();
    m_parents.clear();
    m_world.sleep();
  }
  if(!settings.on && old.on && m_written)
  {
    m_rig.setPhysical({}, m_owned);
    m_written = false;
    return true;
  }
  return false;
}

bool Secondary::writeBack()
{
  std::vector<pose::PhysicalValue> values;
  bool                             moved = false;
  for(size_t i = 0; i < m_soft.size(); i++)
  {
    Soft&                 s      = m_soft[i];
    const pose::PoseBone& pb     = m_rig.bones()[size_t(s.poseBone)];
    const Quat            local  = rotationOf(boneWorld(s.parentNode)).inverse() * m_world.bodies[i].q;
    const auto            totals = m_rig.rotationTotals(pb, local, s.last);
    pose::PhysicalValue   v;
    v.bone = s.poseBone;
    for(int a = 0; a < 3; a++)
    {
      v.rotation[size_t(a)] = totals[size_t(a)] - pb.driven[size_t(ChannelKind::eRotation)][size_t(a)];
      moved                 = moved || std::abs(totals[size_t(a)] - s.last[size_t(a)]) > 1e-3;
    }
    s.last = totals;
    values.push_back(v);
  }
  if(!moved)
    return false;
  m_rig.setPhysical(values, m_owned);
  m_written = true;
  return true;
}

bool Secondary::step(double seconds)
{
  if(!m_settings.on || m_soft.empty())
    return false;
  fv::timing::Scope t("secondary: step", 1);
  if(m_rig.restVersion() != m_restVersion)
  {
    // formulas moved the joints: the bodies again, where the bones are
    build();
    place();
  }
  std::vector<Mat4>   parents;
  std::vector<double> targets;
  for(const Soft& s : m_soft)
  {
    parents.push_back(boneWorld(s.parentNode));
    const pose::PoseBone& pb = m_rig.bones()[size_t(s.poseBone)];
    for(int a = 0; a < 3; a++)
      targets.push_back(pb.posed(ChannelKind::eRotation, a) + pb.driven[size_t(ChannelKind::eRotation)][size_t(a)]);
  }
  // what moved since the last frame; a jump takes them along at rest
  bool moved = targets != m_targets, jumped = m_parents.size() != parents.size();
  for(size_t i = 0; i < parents.size() && !jumped; i++)
  {
    const Mat4& was = m_parents[i];
    const Mat4& now = parents[i];
    for(size_t k = 0; k < 16; k++)
      moved = moved || std::abs(was.e[k] - now.e[k]) > 1e-6;
    jumped = length(originOf(now) - originOf(was)) > kJumpMove || angleOf(rotationOf(was).inverse() * rotationOf(now)) > kJumpTurn;
  }
  bool wrote = false;
  if(jumped)
  {
    // no motion known to answer: where the pose has them, still, showing their channels
    place();
    m_world.sleep();
    if(m_written)
    {
      m_rig.setPhysical({}, m_owned);
      m_written = false;
      wrote     = true;
    }
  }
  else if(moved)
    m_world.wake();
  if(!jumped && !m_world.asleep())
  {
    m_world.settings.gravity = m_gravity;
    // the parents swept from where they were to where they are, across the frame's substeps
    aim(m_parents);
    std::vector<Vec3> fromAnchor;
    std::vector<Quat> fromFrame;
    for(const Joint& j : m_world.joints)
    {
      fromAnchor.push_back(j.anchorA);
      fromFrame.push_back(j.frameA);
    }
    aim(parents);
    std::vector<Vec3> toAnchor;
    std::vector<Quat> toFrame;
    for(const Joint& j : m_world.joints)
    {
      toAnchor.push_back(j.anchorA);
      toFrame.push_back(j.frameA);
    }
    m_world.onSubstep = [&](double t) {
      for(size_t i = 0; i < m_world.joints.size(); i++)
      {
        m_world.joints[i].anchorA = fromAnchor[i] + (toAnchor[i] - fromAnchor[i]) * t;
        m_world.joints[i].frameA  = slerp(fromFrame[i], toFrame[i], t);
      }
    };
    m_world.step(seconds);
    m_world.onSubstep = nullptr;
    if(m_world.asleep())
    {
      // still: the channels again, exactly (nothing is kept of the motion)
      if(m_written)
      {
        m_rig.setPhysical({}, m_owned);
        m_written = false;
        wrote     = true;
      }
    }
    else
      wrote = writeBack();
  }
  m_parents = std::move(parents);
  m_targets = std::move(targets);
  if(awake() != m_wasAwake)
  {
    m_wasAwake = awake();
    m_version++;
  }
  return wrote;
}

}  // namespace fv::physics
