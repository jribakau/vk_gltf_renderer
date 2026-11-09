#include <figure/physics/vam/player.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <figure/rig/spec.hpp>

namespace fv::physics::vam {

namespace {

constexpr double kDeg = 3.14159265358979323846 / 180;
// a joint damper's spring: its damping time c / k (s), long enough that it never pulls
constexpr double kNoPull = 1e4;

// A Person's body controllers and the Genesis 2 bone each holds
const std::map<std::string, std::string>& boneOf()
{
  static const std::map<std::string, std::string> table{
      {"hipControl", "hip"},           {"pelvisControl", "pelvis"},     {"abdomenControl", "abdomen"},   {"abdomen2Control", "abdomen2"},
      {"chestControl", "chest"},       {"neckControl", "neck"},         {"headControl", "head"},         {"lShoulderControl", "lCollar"},
      {"rShoulderControl", "rCollar"}, {"lArmControl", "lShldr"},       {"rArmControl", "rShldr"},       {"lElbowControl", "lForeArm"},
      {"rElbowControl", "rForeArm"},   {"lHandControl", "lHand"},       {"rHandControl", "rHand"},       {"lThighControl", "lThigh"},
      {"rThighControl", "rThigh"},     {"lKneeControl", "lShin"},       {"rKneeControl", "rShin"},       {"lFootControl", "lFoot"},
      {"rFootControl", "rFoot"},       {"lToeControl", "lToe"},         {"rToeControl", "rToe"},
  };
  return table;
}

// Which way a Genesis 2 limb bone points at rest, in Unity's space: toward the bones below it (their
// local positions summed: the hand's middle knuckle is under its carpal), else its forearm's way,
// else as a T-pose has it.
Vec3 vamRestDirection(const std::string& bone, const Pose& pose, const Pose& scene)
{
  const char side = bone[0];
  const std::string limb = bone.substr(1);
  const auto at = [&](const std::string& name) -> std::optional<Vec3> {
    if(const auto it = pose.bones.find(name); it != pose.bones.end())
      return it->second;
    if(const auto it = scene.bones.find(name); it != scene.bones.end())
      return it->second;
    return std::nullopt;
  };
  const auto chain = [&](std::initializer_list<const char*> names) -> std::optional<Vec3> {
    Vec3 sum;
    for(const char* n : names)
    {
      const auto p = at(std::string(1, side) + n);
      if(!p)
        return std::nullopt;
      sum = sum + *p;
    }
    return length(sum) > 1e-4 ? std::optional<Vec3>(normalize(sum)) : std::nullopt;
  };
  std::optional<Vec3> d;
  if(limb == "Collar")
    d = chain({"Shldr"});
  else if(limb == "Shldr")
    d = chain({"ForeArm"});
  else if(limb == "ForeArm")
    d = chain({"Hand"});
  else if(limb == "Hand")
  {
    d = chain({"Carpal1", "Mid1"});
    if(!d)
      d = chain({"Hand"});
  }
  else if(limb == "Thigh")
    d = chain({"Shin"});
  else if(limb == "Shin")
    d = chain({"Foot"});
  else if(limb == "Foot")
    d = chain({"Toe"});
  if(d)
    return *d;
  const double out = side == 'l' ? -1 : 1;
  if(limb == "Thigh" || limb == "Shin")
    return {0, -1, 0};
  if(limb == "Foot")
    return normalize(Vec3{0, -0.05, 1});
  return {out, 0, 0};
}

bool isLimb(const std::string& bone)
{
  if(bone.size() < 2 || (bone[0] != 'l' && bone[0] != 'r'))
    return false;
  const std::string limb = bone.substr(1);
  return limb == "Collar" || limb == "Shldr" || limb == "ForeArm" || limb == "Hand" || limb == "Thigh" || limb == "Shin" || limb == "Foot";
}

// a Genesis 2 bone's rest position under its parent, the clip's Pose's else the scene's
std::optional<Vec3> restOffset(const std::string& name, const Pose& pose, const Pose& scene)
{
  if(const auto it = pose.bones.find(name); it != pose.bones.end())
    return it->second;
  if(const auto it = scene.bones.find(name); it != scene.bones.end())
    return it->second;
  return std::nullopt;
}

}  // namespace

Quat unityEuler(const Vec3& degrees)
{
  // Quaternion.Euler(x, y, z) is qy * qx * qz, which three's order "YXZ" is
  return quatFromEuler(degrees.x * kDeg, degrees.y * kDeg, degrees.z * kDeg, "YXZ");
}

Vec3 toOurs(const Vec3& unity)
{
  return Vec3{-unity.x, unity.y, unity.z} * 100.0;
}

Quat toOurs(const Quat& unity)
{
  // the mirror M = diag(-1, 1, 1) takes a turn R to M R M: a turn the other way about M's image of
  // its axis
  return {unity.x, -unity.y, -unity.z, unity.w};
}

Spring onOurMass(const Spring& vam, double vamKg, double ourGrams, bool angular)
{
  const double r = vamKg > 0 && ourGrams > 0 ? ourGrams / 1000 / vamKg : 1;
  // 1 N = 1e5 dyne and 1 m = 100 cm: N/m is 1e3 dyne/cm, N m is 1e7 dyne cm
  const double perLength = angular ? 1e7 : 1e3;
  const double force     = angular ? 1e7 : 1e5;
  return {vam.stiffness * r * perLength, vam.damper * r * perLength, vam.maxForce * r * force};
}

Player::Player(Person person, size_t clip)
    : m_person(std::move(person))
    , m_clip(std::min(clip, m_person.clips.empty() ? size_t(0) : m_person.clips.size() - 1))
{
}

Controller Player::settingsOf(const std::string& name) const
{
  if(clip().pose)
    if(const auto it = clip().pose->controllers.find(name); it != clip().pose->controllers.end())
      return it->second;
  if(const auto it = m_person.scene.controllers.find(name); it != m_person.scene.controllers.end())
    return it->second;
  return {};
}

void Player::bind(const FigurePhysics& physics, const pose::PoseRig& rig, const SceneGraph& scene)
{
  m_bound.clear();
  m_joints.clear();
  m_unmatched.clear();
  m_restVersion = rig.restVersion();
  m_parts       = physics.parts().size();
  const auto& bones = rig.bones();
  // the part a Genesis 2 bone is in on our figure: a part whose own bone is in its anatomical slot,
  // else one with such a bone merged into it
  const auto partFor = [&](const std::string& g2) {
    const rig::Name want   = rig::parse(g2);
    int             merged = -1;
    for(size_t i = 0; i < physics.parts().size(); i++)
    {
      const FigurePhysics::Part& part = physics.parts()[i];
      for(size_t m = 0; m < part.members.size(); m++)
      {
        const rig::Name p = rig::parse(bones[size_t(part.members[m])].name);
        if(p.slot != want.slot || p.side != want.side)
          continue;
        if(m == 0)
          return int(i);
        if(merged < 0)
          merged = int(i);
      }
    }
    return merged;
  };
  const auto jointOf = [&](int part) -> const fv::JointData* {
    const int node = bones[size_t(physics.parts()[size_t(part)].poseBone)].bone;
    return node >= 0 && scene.nodes[size_t(node)].joint ? scene.nodes[size_t(node)].joint.get() : nullptr;
  };

  // VaM's legs and ours: the hip down to the ankle, at rest
  {
    double vam = 0;
    bool   all = true;
    for(const char* b : {"pelvis", "lThigh", "lShin", "lFoot"})
      if(const auto o = restOffset(b, pose(), m_person.scene))
        vam -= o->y;
      else
        all = all && std::string(b) == "pelvis";
    double ours = 0;
    const int hip = physics.parts().empty() ? -1 : 0, foot = partFor("lFoot");
    if(hip >= 0 && foot >= 0)
      if(const fv::JointData *a = jointOf(hip), *b = jointOf(foot); a && b)
        ours = a->center[1] - b->center[1];
    m_scale = all && vam > 0.1 && ours > 10 ? ours / (vam * 100) : 1;
    m_pelvisOffset = toOurs(restOffset("pelvis", pose(), m_person.scene).value_or(Vec3{})) * m_scale;
  }

  // the whole body's mass, ours and VaM's
  m_ourGrams = 0;
  for(const RigidBody& b : physics.world().bodies)
    m_ourGrams += b.mass();
  m_vamKg = 0;
  for(const auto& [name, g2] : boneOf())
    m_vamKg += settingsOf(name).mass;

  // what holds: every controller the clip keys, and those its Pose holds still
  std::set<std::string> names;
  for(const Track& t : clip().tracks)
    names.insert(t.controller);
  if(clip().pose)
    for(const auto& [name, c] : clip().pose->controllers)
      names.insert(name);
  for(const std::string& name : names)
  {
    const auto bone = boneOf().find(name);
    if(bone == boneOf().end())
      continue;
    Bound b;
    b.name     = name;
    b.settings = settingsOf(name);
    for(const Track& t : clip().tracks)
      if(t.controller == name)
        b.track = &t;
    const bool position = b.track ? b.track->position : clip().pose && b.settings.position.has_value();
    const bool rotation = b.track ? b.track->rotation : clip().pose && b.settings.rotation.has_value();
    // Unity's drive gives nothing at a force limit of 0
    b.move = position && b.settings.holdsPosition() && b.settings.holdPositionSpring > 0 && b.settings.holdPositionMaxForce > 0;
    b.turn = rotation && b.settings.holdsRotation() && b.settings.holdRotationSpring > 0 && b.settings.holdRotationMaxForce > 0;
    if(!b.move && !b.turn)
      continue;
    // no bodies of their own: posed by their channels, which the clip doesn't set
    if(bone->second == "lToe" || bone->second == "rToe")
      continue;
    b.hip    = bone->second == "hip";
    b.pelvis = bone->second == "pelvis";
    b.part   = b.hip || b.pelvis ? (physics.parts().empty() ? -1 : 0) : partFor(bone->second);
    if(b.part < 0 || (b.part == 0 && !b.hip && !b.pelvis))
    {
      m_unmatched.push_back(name);
      continue;
    }
    // our bone turned from its rest to point as VaM's does at rest
    if(isLimb(bone->second))
      if(const fv::JointData* j = jointOf(b.part))
      {
        const Vec3 ours = normalize(Vec3{j->endPoint[0] - j->center[0], j->endPoint[1] - j->center[1], j->endPoint[2] - j->center[2]});
        const Vec3 vam  = normalize(toOurs(vamRestDirection(bone->second, pose(), m_person.scene)));
        if(length(ours) > 0.5)
          b.rest = quatFromUnitVectors(ours, vam);
      }
    m_bound.push_back(b);
  }

  // the joint drives: each controller's, whether it holds or not, on its part's joint
  for(const auto& [name, g2] : boneOf())
  {
    if(!options.jointDrives || g2 == "hip" || g2 == "pelvis" || g2 == "lToe" || g2 == "rToe")
      continue;
    const bool known = (clip().pose && clip().pose->controllers.contains(name)) || m_person.scene.controllers.contains(name);
    const int  part  = known ? partFor(g2) : -1;
    if(part <= 0 || std::any_of(m_joints.begin(), m_joints.end(), [&](const auto& j) { return j.part == part; }))
      continue;
    const Controller           c    = settingsOf(name);
    const RigidBody&           body = physics.world().bodies[size_t(part)];
    const Spring               s    = onOurMass({c.jointDriveSpring, c.jointDriveDamper, c.jointDriveMaxForce}, options.wholeBody ? m_vamKg : c.mass,
                                                    options.wholeBody ? m_ourGrams : body.mass(), true);
    FigurePhysics::Drive::Joint j;
    j.part = part;
    if(c.jointDriveMaxForce <= 0 || (s.stiffness <= 0 && s.damper <= 0))
      // no drive: limp, but for the joint's friction
      j.damping = 2;
    else if(s.stiffness > 0)
    {
      j.compliance   = 1 / s.stiffness;
      j.driveDamping = s.damper / s.stiffness;
      j.damping      = 2;
      j.maxTorque    = s.maxForce;
    }
    else
    {
      // a damper alone: the drive's damper, in the solve, which reaches the chain the joint turns
      // (a body's own spin damped reaches little of a limb's swing), on a spring too soft to pull
      // toward any pose: it would take hours (kNoPull, c / k) to come back
      j.compliance   = kNoPull / s.damper;
      j.driveDamping = kNoPull;
      j.damping      = 2;
      j.maxTorque    = s.maxForce;
    }
    m_joints.push_back(j);
  }
}

void Player::begin(const FigurePhysics& physics, const pose::PoseRig& rig, const SceneGraph& scene, float time)
{
  if(rig.restVersion() != m_restVersion || physics.parts().size() != m_parts)
    bind(physics, rig, scene);
  m_blend.clear();
  m_blendFrom = time;
  const auto now = [&](int part) {
    const RigidBody& b = physics.world().bodies[size_t(part)];
    return std::make_pair(b.toWorld(physics.parts()[size_t(part)].com * -1.0), b.q);
  };
  if(physics.parts().empty() || options.blendIn <= 0)
    return;
  m_blend["hip"] = now(0);
  for(const Bound& b : m_bound)
    if(!b.hip && !b.pelvis)
      m_blend[b.name] = now(b.part);
}

FigurePhysics::Drive Player::at(float time, const FigurePhysics& physics, const pose::PoseRig& rig, const SceneGraph& scene)
{
  if(rig.restVersion() != m_restVersion || physics.parts().size() != m_parts)
  {
    bind(physics, rig, scene);
    m_blend.clear();
  }
  FigurePhysics::Drive d;
  m_targets.clear();
  if(physics.parts().empty())
    return d;
  d.joints = m_joints;
  d.hold   = options.hold;
  // the share of the way from where the body began, eased; a loop round or a time before it ends it
  const double into = options.blendIn > 0 ? (time - m_blendFrom) / options.blendIn : 1;
  if(into < 0 || into >= 1)
    m_blend.clear();
  const double w     = into * into * (3 - 2 * into);
  const auto   blend = [&](const std::string& name, Vec3& at, Quat& q) {
    if(const auto it = m_blend.find(name); it != m_blend.end())
    {
      at = it->second.first + (at - it->second.first) * w;
      q  = slerp(it->second.second, q, w);
    }
  };
  const float t = std::clamp(time, 0.f, clip().length);

  // the Person's space placed on our figure: the floor under its main control is our root, its
  // facing ours
  const Pose& p        = pose();
  const Vec3  rootAt   = p.rootPosition.value_or(Vec3{});
  const Quat  rootTurn = p.rootRotation ? unityEuler(*p.rootRotation) : Quat{};
  const Quat  unturn   = p.rootRotation ? unityEuler({0, p.rootRotation->y, 0}).inverse() : Quat{};
  const auto  place    = [&](const Vec3& local) {
    const Vec3 w = rootAt + rotate(local, rootTurn);
    return toOurs(rotate(Vec3{w.x - rootAt.x, w.y, w.z - rootAt.z}, unturn)) * m_scale;
  };
  const auto turn = [&](const Quat& local) { return toOurs(unturn * rootTurn * local); };
  struct Sampled
  {
    const Bound* b;
    Vec3         at;
    Quat         rotation;
  };
  std::vector<Sampled> sampled;
  for(const Bound& b : m_bound)
  {
    Vec3 local;
    Quat q;
    if(b.track)
    {
      local = b.track->positionAt(t);
      q     = b.track->rotationAt(t);
    }
    else
    {
      local = b.settings.position.value_or(Vec3{});
      q     = b.settings.rotation ? unityEuler(*b.settings.rotation) : Quat{};
    }
    sampled.push_back({&b, place(local), turn(q) * b.rest});
  }

  // the hip: where the hip controller has it, else under the pelvis controller
  const Sampled* hip = nullptr;
  const Sampled* pelvis = nullptr;
  for(const Sampled& s : sampled)
    if(s.b->hip)
      hip = &s;
    else if(s.b->pelvis)
      pelvis = &s;
  const Sampled* moves = hip && hip->b->move ? hip : pelvis && pelvis->b->move ? pelvis : nullptr;
  const Sampled* turns = hip && hip->b->turn ? hip : pelvis && pelvis->b->turn ? pelvis : moves;
  if(moves)
  {
    Quat q  = turns->rotation;
    Vec3 at = moves->b->pelvis ? moves->at - rotate(m_pelvisOffset, q) : moves->at;
    blend("hip", at, q);
    d.hip = std::make_pair(at, q);
    m_targets.push_back({moves->b->name, 0, true, true, at, q});
  }

  for(const Sampled& s : sampled)
  {
    const Bound& b = *s.b;
    if(b.hip || b.pelvis)
      continue;
    const double              grams = physics.world().bodies[size_t(b.part)].mass();
    const double              vamKg = options.wholeBody ? m_vamKg : b.settings.mass, ours = options.wholeBody ? m_ourGrams : grams;
    const Spring              lin   = onOurMass({b.settings.holdPositionSpring, b.settings.holdPositionDamper, b.settings.holdPositionMaxForce}, vamKg, ours, false);
    const Spring              ang   = onOurMass({b.settings.holdRotationSpring, b.settings.holdRotationDamper, b.settings.holdRotationMaxForce}, vamKg, ours, true);
    FigurePhysics::Drive::Pin pin;
    pin.part              = b.part;
    pin.move              = b.move;
    pin.turn              = b.turn;
    pin.at                = s.at;
    pin.rotation          = s.rotation;
    blend(b.name, pin.at, pin.rotation);
    pin.compliance        = lin.stiffness > 0 ? 1 / lin.stiffness : 0;
    pin.angularCompliance = ang.stiffness > 0 ? 1 / ang.stiffness : 0;
    // a damper on a mass slows it at c / m each second
    pin.damping   = grams > 0 ? lin.damper / grams : 0;
    pin.maxForce  = lin.maxForce;
    pin.maxTorque = ang.maxForce;
    d.pins.push_back(pin);
    m_targets.push_back({b.name, b.part, b.move, b.turn, pin.at, pin.rotation});
  }
  return d;
}

}  // namespace fv::physics::vam
