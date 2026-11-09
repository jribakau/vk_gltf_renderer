#include <figure/physics/rig/figure_physics.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <figure/physics/skin.hpp>
#include <figure/io/timing.hpp>
#include <figure/rig/spec.hpp>

namespace fv::physics {

namespace {

using pose::ChannelKind;

constexpr double kGravity = 980;

// a bone's place in the ragdoll by the skeleton spec's part column: a part of its own, merged into
// the part above, or neither
rig::Part ragdollPart(const rig::Name& name)
{
  return name.joint ? name.joint->part : rig::Part::eNone;
}

Vec3 originOf(const Mat4& m)
{
  return {m.e[12], m.e[13], m.e[14]};
}

// How far (radians) a joint gives under the torque of the load it carries, at Hold s: nothing at 1,
// a loose limb at 0.5, nothing holds at 0. The shape of dForce's tether law, its scale by eye.
double give(double s)
{
  if(s >= 0.999)
    return 0;
  s              = std::max(s, 1e-3);
  const double r = (1 - s) / s;
  return 2 * r * r;
}

bool hasRotation(const pose::PoseBone& pb)
{
  return std::any_of(pb.axes.begin(), pb.axes.end(), [](const pose::Axis& a) { return a.kind == ChannelKind::eRotation; });
}

// A capsule fitted to points: along their principal axis, as long as they reach (2% to 98%), as thick
// as their median distance from it.
std::optional<Capsule> fitCapsule(const std::vector<Vec3>& pts)
{
  if(pts.size() < 12)
    return std::nullopt;
  Vec3 mu;
  for(const Vec3& p : pts)
    mu = mu + p;
  mu = mu * (1.0 / double(pts.size()));
  double c[3][3]{};
  for(const Vec3& p : pts)
  {
    const double d[3] = {p.x - mu.x, p.y - mu.y, p.z - mu.z};
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++)
        c[i][j] += d[i] * d[j];
  }
  Vec3 v = normalize(Vec3{1, 0.7, 0.3});
  for(int it = 0; it < 60; it++)
  {
    const Vec3 n{c[0][0] * v.x + c[0][1] * v.y + c[0][2] * v.z, c[1][0] * v.x + c[1][1] * v.y + c[1][2] * v.z, c[2][0] * v.x + c[2][1] * v.y + c[2][2] * v.z};
    if(length(n) < 1e-12)
      break;
    v = normalize(n);
  }
  std::vector<double> t, r;
  t.reserve(pts.size());
  r.reserve(pts.size());
  for(const Vec3& p : pts)
  {
    const double along = dot(p - mu, v);
    t.push_back(along);
    r.push_back(length(p - mu - v * along));
  }
  std::sort(t.begin(), t.end());
  std::nth_element(r.begin(), r.begin() + long(r.size() / 2), r.end());
  const double radius = std::max(1.0, r[r.size() / 2]);
  const double lo = t[size_t(double(t.size() - 1) * 0.02)], hi = t[size_t(double(t.size() - 1) * 0.98)];
  const double half = std::max(0.0, (hi - lo) / 2 - radius);
  const Vec3   mid  = mu + v * ((hi + lo) / 2);
  return Capsule{mid - v * half, mid + v * half, radius};
}

}  // namespace

FigurePhysics::FigurePhysics(SceneGraph& scene, pose::PoseRig& rig, const morph::MorphSet* morphs, int bodyMesh, int root)
    : m_scene(scene)
    , m_rig(rig)
    , m_morphs(morphs)
    , m_bodyMesh(bodyMesh)
    , m_root(root)
{
  build();
}

int FigurePhysics::partOf(int poseBone) const
{
  for(size_t i = 0; i < m_parts.size(); i++)
    if(std::find(m_parts[i].members.begin(), m_parts[i].members.end(), poseBone) != m_parts[i].members.end())
      return int(i);
  return -1;
}

Mat4 FigurePhysics::boneWorld(int node) const
{
  // in the root's space: the root node's placement and its centimetre scale divided out
  return inverse(m_scene.world(m_root)) * m_scene.world(node);
}

Capsule FigurePhysics::restCapsule(size_t part) const
{
  // at rest a part's bone frame is the root's moved to its joint centre
  const Part&    p = m_parts[part];
  const Capsule& s = m_world.bodies[part].shape;
  const auto&    d = m_scene.nodes[size_t(p.node)].joint;
  const Vec3     o = (d ? Vec3{d->center[0], d->center[1], d->center[2]} : Vec3{}) + p.com;
  return {o + s.a, o + s.b, s.radius};
}

Capsule FigurePhysics::capsuleNow(size_t part) const
{
  const Part&    p = m_parts[part];
  const Capsule& s = m_world.bodies[part].shape;
  const Mat4     w = boneWorld(p.node);
  return {w.transformPoint(p.com + s.a), w.transformPoint(p.com + s.b), s.radius};
}

Mat4 FigurePhysics::targetWorld(int node) const
{
  // a part's own bone as its channels pose it, every other bone as it is
  const Node& n = m_scene.nodes[size_t(node)];
  if(node == m_root || n.parent < 0)
    return boneWorld(node);
  const int part = m_partOfNode[size_t(node)];
  if(part < 0)
    return boneWorld(node);
  Mat4 local = m_scene.local(node);
  if(m_parts[size_t(part)].node == node)
  {
    const pose::PoseBone& pb = m_rig.bones()[size_t(m_parts[size_t(part)].poseBone)];
    local                    = compose(m_rig.targetPosition(pb), m_rig.targetRotation(pb), n.scale);
  }
  return targetWorld(n.parent) * local;
}

void FigurePhysics::build()
{
  fv::timing::Scope t("physics: build", 1);
  m_parts.clear();
  m_world = RigidWorld{};
  m_hipPin = m_pull = -1;
  m_aimed           = false;
  m_partOfNode.assign(m_scene.nodes.size(), -1);
  m_restVersion = m_rig.restVersion();
  const auto& bones = m_rig.bones();
  std::vector<int> poseOfNode(m_scene.nodes.size(), -1);
  for(size_t i = 0; i < bones.size(); i++)
    if(bones[i].bone >= 0 && size_t(bones[i].bone) < poseOfNode.size())
      poseOfNode[size_t(bones[i].bone)] = int(i);
  const auto slotOf = [&](int poseBone) { return rig::parse(bones[size_t(poseBone)].name); };
  // a twist bone is a part when a part's bone lies under it (G8's bend and twist chains); G9's twist
  // bones hang off the side and keep their formulas
  const auto partBelow = [&](int node) {
    std::vector<int> stack(m_scene.nodes[size_t(node)].children);
    while(!stack.empty())
    {
      const int c = stack.back();
      stack.pop_back();
      if(const int pbc = poseOfNode[size_t(c)]; pbc >= 0)
      {
        const auto parsed = slotOf(pbc);
        if(ragdollPart(parsed) == rig::Part::eOwn && !parsed.twist)
          return true;
      }
      stack.insert(stack.end(), m_scene.nodes[size_t(c)].children.begin(), m_scene.nodes[size_t(c)].children.end());
    }
    return false;
  };
  const auto partAbove = [&](int node) {
    for(int n = m_scene.nodes[size_t(node)].parent; n >= 0; n = m_scene.nodes[size_t(n)].parent)
      if(m_partOfNode[size_t(n)] >= 0)
        return m_partOfNode[size_t(n)];
    return -1;
  };
  // parents before children
  std::vector<int> order;
  {
    std::vector<int> stack{m_rig.bones().empty() ? -1 : 0};
    std::vector<int> roots;
    for(size_t n = 0; n < m_scene.nodes.size(); n++)
      if(m_scene.nodes[n].parent < 0)
        roots.push_back(int(n));
    stack = roots;
    while(!stack.empty())
    {
      const int n = stack.back();
      stack.pop_back();
      if(poseOfNode[size_t(n)] >= 0)
        order.push_back(poseOfNode[size_t(n)]);
      const auto& ch = m_scene.nodes[size_t(n)].children;
      stack.insert(stack.end(), ch.rbegin(), ch.rend());
    }
  }
  for(const int i : order)
  {
    const pose::PoseBone& pb = bones[size_t(i)];
    const auto            parsed = slotOf(i);
    // the hip is the root part whatever channels it has: held, its channels are its target, and
    // falling it shows where it is on any axis
    if(pb.group == "Figure" || (!hasRotation(pb) && parsed.slot != "hip"))
      continue;
    const int  up     = partAbove(pb.bone);
    const bool part   = ragdollPart(parsed) == rig::Part::eOwn && (!parsed.twist || partBelow(pb.bone));
    if(part && (parsed.slot == "hip" ? up < 0 && m_parts.empty() : up >= 0))
    {
      Part p;
      p.poseBone = i;
      p.node     = pb.bone;
      p.parent   = up;
      p.members  = {i};
      p.group    = pb.group;
      m_partOfNode[size_t(pb.bone)] = int(m_parts.size());
      m_parts.push_back(std::move(p));
    }
    else if(ragdollPart(parsed) == rig::Part::eMerged && up >= 0 && m_partOfNode[size_t(m_scene.nodes[size_t(pb.bone)].parent)] == up)
    {
      m_parts[size_t(up)].members.push_back(i);
      m_partOfNode[size_t(pb.bone)] = up;
    }
  }
  if(m_parts.empty() || m_parts.size() < 2)
  {
    m_parts.clear();
    return;
  }
  // the bones it poses, and no others (the chest's soft bones are another's)
  m_owned.clear();
  for(const Part& p : m_parts)
    m_owned.insert(m_owned.end(), p.members.begin(), p.members.end());

  // each part's cage vertices: those whose strongest bone is it, or under it and no part
  std::vector<std::vector<Vec3>> points(m_parts.size());
  const auto                     partAt = [&](int node) {
    for(int n = node; n >= 0; n = m_scene.nodes[size_t(n)].parent)
      if(m_partOfNode[size_t(n)] >= 0)
        return m_partOfNode[size_t(n)];
    return -1;
  };
  if(m_bodyMesh >= 0 && size_t(m_bodyMesh) < m_scene.meshes.size())
  {
    const Mesh& body = m_scene.meshes[size_t(m_bodyMesh)];
    if(body.geo && body.skinned())
    {
      std::vector<float>           scratch;
      const std::span<const float> cage = m_morphs ? m_morphs->morphedCageView(m_bodyMesh, scratch) : std::span<const float>(body.geo->cagePositions);
      const mesh::SkinInput&       skin = body.cageSkin;
      const size_t                 nv   = cage.size() / 3;
      if(skin.start.size() == nv + 1)
        for(size_t v = 0; v < nv; v++)
        {
          int   best = -1;
          float w    = -1;
          for(int k = skin.start[v]; k < skin.start[v + 1]; k++)
            if(skin.weight[size_t(k)] > w)
            {
              w    = skin.weight[size_t(k)];
              best = skin.bone[size_t(k)];
            }
          if(best < 0 || size_t(best) >= body.bones.size())
            continue;
          if(const int part = partAt(body.bones[size_t(best)]); part >= 0)
            points[size_t(part)].push_back({cage[v * 3], cage[v * 3 + 1], cage[v * 3 + 2]});
        }
    }
  }
  // the parts' children, for the fallback shapes and the subtrees
  std::vector<std::vector<int>> children(m_parts.size());
  for(size_t i = 1; i < m_parts.size(); i++)
    children[size_t(m_parts[i].parent)].push_back(int(i));
  const auto centerOf = [&](int node) {
    const auto& d = m_scene.nodes[size_t(node)].joint;
    return d ? Vec3{d->center[0], d->center[1], d->center[2]} : Vec3{};
  };
  for(size_t i = 0; i < m_parts.size(); i++)
  {
    Part&      p      = m_parts[i];
    const Vec3 center = centerOf(p.node);
    std::optional<Capsule> fit = fitCapsule(points[i]);
    if(!fit && children[i].size() > 1)
    {
      // no skin to fit to, and several joints under it (the hip, the chest): across them
      std::vector<Vec3> joints{center};
      for(const int c : children[i])
        joints.push_back(centerOf(m_parts[size_t(c)].node));
      Vec3 mu;
      for(const Vec3& q : joints)
        mu = mu + q;
      mu = mu * (1.0 / double(joints.size()));
      Vec3   axis;
      double spread = 0;
      for(const Vec3& q : joints)
        for(const Vec3& o : joints)
          if(length(q - o) > spread)
          {
            spread = length(q - o);
            axis   = normalize(q - o);
          }
      const double r    = std::max(1.5, 0.35 * spread);
      const double half = std::max(0.0, spread / 2 - r);
      fit               = Capsule{mu - axis * half, mu + axis * half, r};
    }
    else if(!fit)
    {
      // no skin to fit to: from the joint to its child's, or to its end
      Vec3 end;
      if(!children[i].empty())
        end = centerOf(m_parts[size_t(children[i][0])].node);
      else if(const auto& d = m_scene.nodes[size_t(p.node)].joint)
        end = Vec3{d->endPoint[0], d->endPoint[1], d->endPoint[2]};
      const double l    = length(end - center);
      const double r    = std::max(1.5, 0.18 * l);
      const Vec3   dir  = l > 1e-6 ? (end - center) * (1 / l) : Vec3{0, 1, 0};
      const double half = std::max(0.0, l / 2 - r);
      const Vec3   mid  = (center + end) * 0.5;
      fit               = Capsule{mid - dir * half, mid + dir * half, r};
    }
    // in the bone's frame, which at rest is the root's moved to the joint centre; about its middle
    const Vec3 mid = (fit->a + fit->b) * 0.5;
    p.com          = mid - center;
    RigidBody b;
    b.shape = {fit->a - mid, fit->b - mid, fit->radius};
    setCapsuleMass(b);
    m_world.bodies.push_back(b);
  }
  // no part much lighter than the rest, nor spinning on a hair: a light part a heavy one hangs from
  // takes every correction of its joints, and a mass ratio past a few hundred whips it round
  {
    double total = 0;
    for(const RigidBody& b : m_world.bodies)
      total += b.mass();
    for(RigidBody& b : m_world.bodies)
    {
      const double was = b.mass();
      const double m   = std::max(was, 0.015 * total);
      // a solid ball 3 cm round at least
      const double least = m * 0.4 * 3 * 3;
      b.invMass          = 1 / m;
      for(double* inv : {&b.invInertia.x, &b.invInertia.y, &b.invInertia.z})
        *inv = std::min(*inv * was / m, 1 / least);
    }
  }
  // a part never meets its parent, and the subtree masses
  for(size_t i = 0; i < m_parts.size(); i++)
    if(m_parts[i].parent >= 0)
    {
      m_world.bodies[i].noCollide.push_back(m_parts[i].parent);
      m_world.bodies[size_t(m_parts[i].parent)].noCollide.push_back(int(i));
    }
  for(size_t i = m_parts.size(); i-- > 0;)
  {
    m_parts[i].subtreeMass += m_world.bodies[i].mass();
    if(m_parts[i].parent >= 0)
      m_parts[size_t(m_parts[i].parent)].subtreeMass += m_parts[i].subtreeMass;
  }
  for(size_t i = 1; i < m_parts.size(); i++)
  {
    Joint j;
    j.a             = m_parts[i].parent;
    j.b             = int(i);
    m_parts[i].joint = int(m_world.joints.size());
    m_world.joints.push_back(j);
  }
  // the hip, held where it is
  Pin hip;
  hip.body  = 0;
  hip.local = m_parts[0].com * -1.0;
  hip.turn  = true;
  m_hipPin  = int(m_world.pins.size());
  m_world.pins.push_back(hip);
  if(m_ground)
    m_world.planes = {*m_ground};
  m_otherKeys.clear();
  syncPins();
}

void FigurePhysics::placeBodies()
{
  for(size_t i = 0; i < m_parts.size(); i++)
  {
    Part&       p  = m_parts[i];
    RigidBody&  b  = m_world.bodies[i];
    const Mat4  w  = boneWorld(p.node);
    b.q            = rotationOf(w);
    b.x            = originOf(w) + rotate(p.com, b.q);
    b.xPrev        = b.x;
    b.qPrev        = b.q;
    b.v = b.w = {};
    const pose::PoseBone& pb = m_rig.bones()[size_t(p.poseBone)];
    for(int a = 0; a < 3; a++)
      p.last[size_t(a)] = pb.shown(ChannelKind::eRotation, a) + pb.driven[size_t(ChannelKind::eRotation)][size_t(a)];
    if(p.joint >= 0)
      for(int a = 0; a < 3; a++)
        m_world.joints[size_t(p.joint)].euler[size_t(a)] = p.last[size_t(a)] * kDeg;
  }
}

double FigurePhysics::strengthOf(const Part& p) const
{
  if(m_drive.hold)
    return std::clamp(*m_drive.hold, 0.0, 1.0);
  const auto it = m_settings.parts.find(p.group);
  return std::clamp(it != m_settings.parts.end() ? it->second : m_settings.strength, 0.0, 1.0);
}

void FigurePhysics::updateJoints()
{
  for(size_t i = 1; i < m_parts.size(); i++)
  {
    const Part&           p  = m_parts[i];
    // the drive that turns a part is its own group's
    const double          s     = strengthOf(p);
    const double          giveS = give(s);
    const Part&           up = m_parts[size_t(p.parent)];
    const pose::PoseBone& pb = m_rig.bones()[size_t(p.poseBone)];
    Joint&                j  = m_world.joints[size_t(p.joint)];
    const int             parentNode = m_scene.nodes[size_t(p.node)].parent;
    // the bone above it is the parent part's own or one merged into it: fixed in that part's frame
    const Mat4 wUp  = boneWorld(up.node);
    const Mat4 rel  = inverse(wUp) * boneWorld(parentNode);
    j.frameA        = rotationOf(rel) * pb.orient;
    j.frameB        = pb.orient;
    j.anchorA       = rel.transformPoint(m_rig.targetPosition(pb)) - up.com;
    j.anchorB       = p.com * -1.0;
    j.order         = pb.order;
    std::array<double, 3> total{};
    for(int a = 0; a < 3; a++)
    {
      const double        d  = pb.driven[size_t(ChannelKind::eRotation)][size_t(a)];
      const pose::Axis*   ax = pb.axis(ChannelKind::eRotation, a);
      total[size_t(a)]       = pb.posed(ChannelKind::eRotation, a) + d;
      double lo = d, hi = d;
      if(ax && ax->clamped)
        lo = ax->min + d, hi = ax->max + d;
      else if(ax)
        // a limit the figure doesn't give: about a person's reach, and never short of the pose asked for
        lo = std::min(-90.0 + d, total[size_t(a)]), hi = std::max(90.0 + d, total[size_t(a)]);
      j.lo[size_t(a)] = lo * kDeg;
      j.hi[size_t(a)] = hi * kDeg;
    }
    j.target = quatFromEuler(total[0] * kDeg, total[1] * kDeg, total[2] * kDeg, pb.order);
    // the same sag at every joint for the same Hold: the give over the torque of the load it carries
    const RigidBody& b     = m_world.bodies[i];
    const Vec3       joint = b.toWorld(j.anchorB);
    Vec3             com;
    double           mass = 0;
    for(size_t k = i; k < m_parts.size(); k++)
    {
      // the subtree: parts are in tree order, so it runs on until a part not under this one
      bool under = false;
      for(int q = int(k); q >= 0; q = m_parts[size_t(q)].parent)
        if(q == int(i))
        {
          under = true;
          break;
        }
      if(!under)
        continue;
      com = com + m_world.bodies[k].x * m_world.bodies[k].mass();
      mass += m_world.bodies[k].mass();
    }
    const double reach = mass > 0 ? std::max(5.0, length(com * (1 / mass) - joint)) : 5.0;
    if(s <= 1e-3)
      j.compliance = -1;
    else
      j.compliance = giveS / (std::max(mass, 1.0) * kGravity * reach);
    // critically damped: the drive turns the load it carries at about sqrt(g / (give reach))
    j.driveDamping = 2 * std::sqrt(std::max(giveS, 1e-4) * reach / kGravity);
    j.damping      = 2;
    j.maxTorque    = 0;
  }
  // a clip's own drives where it has them
  for(const Drive::Joint& d : m_drive.joints)
    if(d.part > 0 && size_t(d.part) < m_parts.size())
    {
      Joint& j        = m_world.joints[size_t(m_parts[size_t(d.part)].joint)];
      j.compliance    = d.compliance;
      j.driveDamping  = d.driveDamping;
      j.damping       = d.damping;
      j.maxTorque     = d.maxTorque;
    }
  // the hip: where its channels put it under the figure's root bone, or where a clip holds it
  Pin& hip = m_world.pins[size_t(m_hipPin)];
  {
    const Mat4 w     = targetWorld(m_parts[0].node);
    hip.target       = m_drive.hip ? m_drive.hip->first : originOf(w);
    hip.rotation     = m_drive.hip ? m_drive.hip->second : rotationOf(w);
    hip.on = !m_settings.letFall;
    // held hard: a soft hip would let the whole body sway on it, and the only damping that reaches
    // that sway is the hip's own, a few kilograms of the body's fifty. A pull bends the spine instead.
    hip.compliance = hip.angularCompliance = 0;
    hip.damping                            = 0;
  }
}

void FigurePhysics::feedForward()
{
  const Vec3 g    = m_world.settings.gravity;
  const bool feed = m_settings.gravity && !m_settings.letFall;
  for(size_t i = 1; i < m_parts.size(); i++)
  {
    Joint& j = m_world.joints[size_t(m_parts[i].joint)];
    j.feedForward = {};
    // a limp part has no drive to give it back
    if(!feed || strengthOf(m_parts[i]) <= 1e-3)
      continue;
    // the torque gravity puts on everything under this joint, about it, given back by the drive
    const Vec3 at = m_world.bodies[i].toWorld(j.anchorB);
    Vec3       torque;
    for(size_t k = i; k < m_parts.size(); k++)
    {
      bool under = false;
      for(int q = int(k); q >= 0; q = m_parts[size_t(q)].parent)
        if(q == int(i))
        {
          under = true;
          break;
        }
      if(under)
        torque = torque + cross(m_world.bodies[k].x - at, g * m_world.bodies[k].mass());
    }
    j.feedForward = torque * -1.0;
  }
  Pin& hip = m_world.pins[size_t(m_hipPin)];
  hip.feedForward = hip.feedForwardTorque = {};
  if(m_settings.gravity && hip.on)
  {
    // the hip carries the body's weight, and the torque of all of it about the hip
    const Vec3 x = m_world.bodies[0].x;
    Vec3       torque;
    for(const RigidBody& b : m_world.bodies)
      torque = torque + cross(b.x - x, g * b.mass());
    hip.feedForward       = g * (-m_parts[0].subtreeMass);
    hip.feedForwardTorque = feed ? torque * -1.0 : Vec3{};
  }
}

std::vector<double> FigurePhysics::targets() const
{
  std::vector<double> out;
  out.reserve(m_parts.size() * 3 + 3);
  for(const Part& p : m_parts)
  {
    const pose::PoseBone& pb = m_rig.bones()[size_t(p.poseBone)];
    for(int a = 0; a < 3; a++)
      out.push_back(pb.posed(ChannelKind::eRotation, a));
    for(const int m : p.members)
      if(m != p.poseBone)
        for(int a = 0; a < 3; a++)
          out.push_back(m_rig.bones()[size_t(m)].posed(ChannelKind::eRotation, a));
  }
  const pose::PoseBone& hip = m_rig.bones()[size_t(m_parts[0].poseBone)];
  for(int a = 0; a < 3; a++)
    out.push_back(hip.posed(ChannelKind::eTranslation, a));
  return out;
}

void FigurePhysics::allowOverlaps()
{
  // a figure that came in seated a little into a chair stays there
  m_world.props = colliders ? colliders() : std::vector<ColliderRef>{};
  m_world.allowOverlaps();
  // and what overlaps in the pose asked for, so a target inside another part doesn't fight a contact
  std::vector<RigidBody> saved = m_world.bodies;
  for(size_t i = 0; i < m_parts.size(); i++)
  {
    const Mat4 w     = targetWorld(m_parts[i].node);
    m_world.bodies[i].q = rotationOf(w);
    m_world.bodies[i].x = originOf(w) + rotate(m_parts[i].com, m_world.bodies[i].q);
  }
  for(size_t i = 0; i < m_parts.size(); i++)
    for(size_t k = i + 1; k < m_parts.size(); k++)
    {
      const RigidBody& b = m_world.bodies[i];
      if(std::find(b.noCollide.begin(), b.noCollide.end(), int(k)) != b.noCollide.end())
        continue;
      const double d = m_world.overlap(int(i), int(k));
      if(d <= 0)
        continue;
      auto it = std::find_if(m_world.allowances.begin(), m_world.allowances.end(), [&](const RigidWorld::Allowance& a) { return a.a == int(i) && a.b == int(k); });
      if(it == m_world.allowances.end())
        m_world.allowances.push_back({int(i), int(k), d});
      else
        it->depth = std::max(it->depth, d);
    }
  m_world.bodies = std::move(saved);
}

bool FigurePhysics::set(const PhysicsSettings& settings, bool fold)
{
  const PhysicsSettings old = m_settings;
  m_settings                = settings;
  if(old == settings)
    return false;
  m_version++;
  if(settings.on && !old.on)
  {
    // fitted to the shape as it is now: a morph moves the body without moving a joint
    build();
    if(m_parts.empty())
    {
      m_settings.on = false;
      return false;
    }
    if(m_ground)
      m_world.planes = {*m_ground};
    placeBodies();
    m_lastTargets = targets();
    allowOverlaps();
    wake();
    // the parts' bones now show the physical layer, where they are
    std::vector<pose::PhysicalValue> values;
    for(const Part& p : m_parts)
    {
      const pose::PoseBone& pb = m_rig.bones()[size_t(p.poseBone)];
      pose::PhysicalValue   v;
      v.bone = p.poseBone;
      for(int a = 0; a < 3; a++)
        v.rotation[size_t(a)] = pb.posed(ChannelKind::eRotation, a);
      if(p.parent < 0)
        v.translation = Vec3{pb.posed(ChannelKind::eTranslation, 0), pb.posed(ChannelKind::eTranslation, 1), pb.posed(ChannelKind::eTranslation, 2)};
      values.push_back(v);
    }
    m_rig.setPhysical(values, m_owned);
    m_wasAwake = true;
    return true;
  }
  if(!settings.on && old.on)
  {
    if(m_pull >= 0)
    {
      m_world.pins.erase(m_world.pins.begin() + m_pull);
      m_pull = -1;
    }
    m_held.clear();
    m_drive = {};
    syncPins();
    if(fold)
      m_rig.foldPhysical(m_owned);
    else
      m_rig.setPhysical({}, m_owned);
    m_world.sleep();
    m_wasAwake = false;
    return true;
  }
  wake();
  return false;
}

void FigurePhysics::setGravity(const Vec3& g)
{
  const Vec3 was = m_gravity;
  m_gravity      = g;
  if(length(was - g) > 1)
    wake();
}

void FigurePhysics::setGround(const std::optional<Plane>& ground)
{
  const bool same = ground.has_value() == m_ground.has_value() && (!ground || (length(ground->n - m_ground->n) < 1e-9 && std::abs(ground->offset - m_ground->offset) < 1e-6));
  m_ground        = ground;
  m_world.planes.clear();
  if(ground)
    m_world.planes.push_back(*ground);
  if(!same)
    wake();
}

void FigurePhysics::wake()
{
  m_world.wake();
  m_sinceInput = 0;
}

bool FigurePhysics::grab(const Vec3& origin, const Vec3& dir, double& depth)
{
  if(!m_settings.on || m_parts.empty())
    return false;
  const Vec3 d    = normalize(dir);
  int        best = -1;
  double     bestT = INFINITY;
  for(size_t i = 0; i < m_world.bodies.size(); i++)
  {
    const RigidBody& b = m_world.bodies[i];
    Vec3             onRay, onAxis;
    closestSegments(origin, origin + d * 1e5, b.toWorld(b.shape.a), b.toWorld(b.shape.b), onRay, onAxis);
    const double miss = length(onRay - onAxis);
    if(miss >= b.shape.radius)
      continue;
    const double t = dot(onRay - origin, d) - std::sqrt(b.shape.radius * b.shape.radius - miss * miss);
    if(t > 0 && t < bestT)
    {
      bestT = t;
      best  = int(i);
    }
  }
  if(best < 0)
    return false;
  if(m_pull >= 0)
    m_world.pins.erase(m_world.pins.begin() + m_pull);
  const RigidBody& b  = m_world.bodies[size_t(best)];
  const Vec3       at = origin + d * bestT;
  Pin              pull;
  pull.body     = best;
  pull.local    = rotate(at - b.x, b.q.inverse());
  pull.target   = at;
  pull.maxForce = 3 * m_parts[0].subtreeMass * kGravity;
  m_pull        = int(m_world.pins.size());
  m_world.pins.push_back(pull);
  depth = bestT;
  wake();
  m_version++;
  return true;
}

int FigurePhysics::partAt(int poseBone) const
{
  if(poseBone < 0 || size_t(poseBone) >= m_rig.bones().size())
    return -1;
  if(const int p = partOf(poseBone); p >= 0)
    return p;
  for(int n = m_rig.bones()[size_t(poseBone)].bone; n >= 0; n = m_scene.nodes[size_t(n)].parent)
    if(size_t(n) < m_partOfNode.size() && m_partOfNode[size_t(n)] >= 0)
      return m_partOfNode[size_t(n)];
  return -1;
}

Vec3 FigurePhysics::pinPoint(int part) const
{
  const Part& p = m_parts[size_t(part)];
  // the knee is a shin's top: its joint, which is its bone's origin
  if(rig::parse(m_rig.bones()[size_t(p.poseBone)].name).slot == "shin")
    return p.com * -1.0;
  return {};
}

void FigurePhysics::syncPins()
{
  if(m_hipPin < 0 || m_parts.empty())
    return;
  std::optional<Pin> pull;
  if(m_pull >= 0)
    pull = m_world.pins[size_t(m_pull)];
  m_world.pins.resize(size_t(m_hipPin) + 1);
  // stiff: a tenth of a millimetre under the figure's whole weight, a hundredth of a radian under it
  // on a 10 cm lever
  const double weight = std::max(1.0, m_parts[0].subtreeMass * kGravity);
  for(const HeldPin& h : m_held)
  {
    const int part = partOf(h.poseBone);
    if(part < 0)
      continue;
    Pin p;
    p.body              = part;
    p.local             = pinPoint(part);
    p.target            = h.at;
    p.turn              = h.turn;
    p.rotation          = h.rotation;
    p.compliance        = 0.01 / weight;
    p.angularCompliance = 0.01 / (weight * 10);
    p.maxForce          = h.maxForce > 0 ? h.maxForce : 10 * weight;
    p.maxTorque         = p.maxForce * 10;
    m_world.pins.push_back(p);
  }
  // a clip's, one for each of its pins, at its bone's origin
  m_drivePin = int(m_world.pins.size());
  for(const Drive::Pin& d : m_drive.pins)
  {
    Pin p;
    p.on = d.part >= 0 && size_t(d.part) < m_parts.size();
    if(p.on)
    {
      p.body  = d.part;
      p.local = m_parts[size_t(d.part)].com * -1.0;
    }
    m_world.pins.push_back(p);
  }
  aimDrivePins(nullptr);
  m_pull = -1;
  if(pull)
  {
    m_pull = int(m_world.pins.size());
    m_world.pins.push_back(*pull);
  }
}

bool FigurePhysics::pin(int poseBone, bool turn, double maxForce)
{
  const int part = partAt(poseBone);
  if(!m_settings.on || part < 0)
    return false;
  const RigidBody& b = m_world.bodies[size_t(part)];
  HeldPin          h;
  h.poseBone = m_parts[size_t(part)].poseBone;
  h.turn     = turn;
  h.at       = b.toWorld(pinPoint(part));
  h.rotation = b.q;
  h.maxForce = maxForce;
  const auto it = std::find_if(m_held.begin(), m_held.end(), [&](const HeldPin& p) { return p.poseBone == h.poseBone; });
  if(it != m_held.end())
    *it = h;
  else
    m_held.push_back(h);
  syncPins();
  m_version++;
  wake();
  return true;
}

bool FigurePhysics::unpin(int poseBone)
{
  const int part = partAt(poseBone);
  if(part < 0)
    return false;
  const int  own = m_parts[size_t(part)].poseBone;
  const auto it  = std::find_if(m_held.begin(), m_held.end(), [&](const HeldPin& p) { return p.poseBone == own; });
  if(it == m_held.end())
    return false;
  m_held.erase(it);
  syncPins();
  m_version++;
  wake();
  return true;
}

void FigurePhysics::restorePins(const std::vector<HeldPin>& pins)
{
  m_held.clear();
  for(const HeldPin& h : pins)
    if(partOf(h.poseBone) >= 0)
      m_held.push_back(h);
  syncPins();
  m_version++;
}

void FigurePhysics::drive(Drive d)
{
  // the same pins on the same parts: only where they aim moves, which a step sweeps to
  const bool same = d.pins.size() == m_drive.pins.size()
                    && std::equal(d.pins.begin(), d.pins.end(), m_drive.pins.begin(),
                                  [](const Drive::Pin& a, const Drive::Pin& b) { return a.part == b.part && a.move == b.move && a.turn == b.turn; });
  const bool was = !m_drive.empty();
  m_drive        = std::move(d);
  if(!m_settings.on || m_parts.empty())
    return;
  if(!same)
    syncPins();
  if(was || !m_drive.empty())
    wake();
}

void FigurePhysics::aimDrivePins(std::vector<std::pair<Vec3, Quat>>* from)
{
  if(m_drivePin < 0)
    return;
  for(size_t i = 0; i < m_drive.pins.size() && size_t(m_drivePin) + i < m_world.pins.size(); i++)
  {
    Pin&              p = m_world.pins[size_t(m_drivePin) + i];
    const Drive::Pin& d = m_drive.pins[i];
    if(from)
      from->emplace_back(p.target, p.rotation);
    p.move              = d.move;
    p.turn              = d.turn;
    p.target            = d.at;
    p.rotation          = d.rotation;
    p.compliance        = d.compliance;
    p.angularCompliance = d.angularCompliance;
    p.damping           = d.damping;
    p.maxForce          = d.maxForce;
    p.maxTorque         = d.maxTorque;
  }
}

void FigurePhysics::setOthers(std::vector<Other> others)
{
  // an allowance carried by whose it is; a figure not met before is allowed what it overlaps now
  // (two figures entering Physics in a hug stay so), and it only shrinks as they part
  const size_t                     n = m_world.bodies.size();
  std::vector<std::vector<double>> allowed(n, std::vector<double>(others.size(), 0.0));
  std::vector<std::string>         keys;
  m_world.others.clear();
  Box mine;
  for(const RigidBody& b : m_world.bodies)
    for(const Vec3& e : {b.toWorld(b.shape.a), b.toWorld(b.shape.b)})
    {
      mine.add(e.x - b.shape.radius, e.y - b.shape.radius, e.z - b.shape.radius);
      mine.add(e.x + b.shape.radius, e.y + b.shape.radius, e.z + b.shape.radius);
    }
  bool near = false;
  for(size_t g = 0; g < others.size(); g++)
  {
    const Other& o   = others[g];
    const auto   was = std::find(m_otherKeys.begin(), m_otherKeys.end(), o.key);
    const size_t old = size_t(was - m_otherKeys.begin());
    keys.push_back(o.key);
    Box theirs;
    for(RigidWorld::Kinematic k : o.capsules)
    {
      k.group = int(g);
      m_world.others.push_back(k);
      for(const Vec3& e : {k.a, k.b})
      {
        theirs.add(e.x - k.radius, e.y - k.radius, e.z - k.radius);
        theirs.add(e.x + k.radius, e.y + k.radius, e.z + k.radius);
      }
    }
    for(size_t i = 0; i < n; i++)
    {
      if(was != m_otherKeys.end() && i < m_world.othersAllowed.size() && old < m_world.othersAllowed[i].size())
      {
        allowed[i][g] = m_world.othersAllowed[i][old];
        continue;
      }
      const RigidBody& b = m_world.bodies[i];
      for(const RigidWorld::Kinematic& k : o.capsules)
      {
        Vec3 ci, ck;
        closestSegments(b.toWorld(b.shape.a), b.toWorld(b.shape.b), k.a, k.b, ci, ck);
        allowed[i][g] = std::max(allowed[i][g], b.shape.radius + k.radius - length(ci - ck));
      }
    }
    near = near || (o.moving && mine.grown(5).overlaps(theirs));
  }
  m_world.othersAllowed = std::move(allowed);
  m_otherKeys           = std::move(keys);
  // the world alone: no input of its own, so the forced sleep still comes
  if(near && m_settings.on)
    m_world.wake();
}

namespace {

// The scale of a matrix that turns and scales the same along every axis; 0 when it doesn't.
double evenScale(const Mat4& m)
{
  const Vec3   c[3] = {{m.e[0], m.e[1], m.e[2]}, {m.e[4], m.e[5], m.e[6]}, {m.e[8], m.e[9], m.e[10]}};
  const double s    = length(c[0]);
  if(s < 1e-12)
    return 0;
  for(int a = 0; a < 3; a++)
  {
    if(std::abs(length(c[a]) - s) > 1e-6 * s)
      return 0;
    if(std::abs(dot(c[a], c[(a + 1) % 3])) > 1e-6 * s * s)
      return 0;
  }
  return s;
}

}  // namespace

void stepTogether(std::span<const Together> figures, double seconds)
{
  // every ragdoll's capsules at the frame's start, in the shared world
  struct Snap
  {
    std::vector<RigidWorld::Kinematic> capsules;
    Box                                box;
    double                             scale  = 0;
    bool                               moving = false;
  };
  std::vector<Snap> snaps(figures.size());
  for(size_t i = 0; i < figures.size(); i++)
  {
    const FigurePhysics* f = figures[i].physics;
    if(!f || !f->settings().on || !f->valid())
      continue;
    Snap& s  = snaps[i];
    s.scale  = evenScale(figures[i].root);
    s.moving = f->awake();
    for(const RigidBody& b : f->world().bodies)
    {
      RigidWorld::Kinematic k;
      k.a      = figures[i].root.transformPoint(b.toWorld(b.shape.a));
      k.b      = figures[i].root.transformPoint(b.toWorld(b.shape.b));
      k.radius = b.shape.radius * s.scale;
      for(const Vec3& e : {k.a, k.b})
      {
        s.box.add(e.x - k.radius, e.y - k.radius, e.z - k.radius);
        s.box.add(e.x + k.radius, e.y + k.radius, e.z + k.radius);
      }
      s.capsules.push_back(k);
    }
  }
  // each meets the others near it, in its own root's space
  for(size_t i = 0; i < figures.size(); i++)
  {
    FigurePhysics* f = figures[i].physics;
    if(!f)
      continue;
    std::vector<FigurePhysics::Other> others;
    if(snaps[i].scale > 0)
    {
      const Mat4 toMine = inverse(figures[i].root);
      for(size_t j = 0; j < figures.size(); j++)
      {
        const Snap& o = snaps[j];
        if(j == i || o.scale <= 0 || std::abs(o.scale / snaps[i].scale - 1) > 1e-6 || !snaps[i].box.grown(10 * snaps[i].scale).overlaps(o.box))
          continue;
        FigurePhysics::Other other;
        other.key    = figures[j].key;
        other.moving = o.moving;
        for(const RigidWorld::Kinematic& k : o.capsules)
          other.capsules.push_back({toMine.transformPoint(k.a), toMine.transformPoint(k.b), k.radius / snaps[i].scale, 0});
        others.push_back(std::move(other));
      }
    }
    f->setOthers(std::move(others));
  }
  for(const Together& t : figures)
    if(t.physics)
      t.physics->step(seconds);
}

void FigurePhysics::drag(const Vec3& target)
{
  if(m_pull < 0)
    return;
  m_world.pins[size_t(m_pull)].target = target;
  wake();
}

std::vector<pose::ChannelValue> FigurePhysics::release()
{
  if(m_pull < 0)
    return {};
  m_world.pins.erase(m_world.pins.begin() + m_pull);
  m_pull = -1;
  // let go, it stays: set down at rest, or a weak Hold's motors would let the swing carry on for
  // seconds before they bring it back
  for(RigidBody& b : m_world.bodies)
    b.v = b.w = {};
  m_version++;
  wake();
  return reached();
}

std::vector<pose::ChannelValue> FigurePhysics::reached() const
{
  std::vector<pose::ChannelValue> out;
  static const char* names[3] = {"x", "y", "z"};
  for(const Part& p : m_parts)
  {
    const pose::PoseBone& pb = m_rig.bones()[size_t(p.poseBone)];
    for(int a = 0; a < 3; a++)
    {
      if(pb.axis(ChannelKind::eRotation, a))
        out.push_back({pb.name, ChannelKind::eRotation, names[a], pb.shown(ChannelKind::eRotation, a)});
      if(p.parent < 0 && pb.axis(ChannelKind::eTranslation, a))
        out.push_back({pb.name, ChannelKind::eTranslation, names[a], pb.shown(ChannelKind::eTranslation, a)});
    }
  }
  return out;
}

std::vector<std::pair<std::string, double>> FigurePhysics::physicalValues() const
{
  std::vector<std::pair<std::string, double>> out;
  static const char* names[3] = {"x", "y", "z"};
  for(const Part& p : m_parts)
  {
    const pose::PoseBone& pb = m_rig.bones()[size_t(p.poseBone)];
    if(!pb.physicalOn)
      continue;
    for(int a = 0; a < 3; a++)
    {
      out.emplace_back(pb.id + "/rotation/" + names[a], pb.physical[size_t(ChannelKind::eRotation)][size_t(a)]);
      if(p.parent < 0)
        out.emplace_back(pb.id + "/translation/" + names[a], pb.physical[size_t(ChannelKind::eTranslation)][size_t(a)]);
    }
  }
  return out;
}

void FigurePhysics::restorePhysical(const std::vector<std::pair<std::string, double>>& values)
{
  if(!m_settings.on || m_parts.empty())
    return;
  std::vector<pose::PhysicalValue> list;
  for(const Part& p : m_parts)
  {
    const pose::PoseBone& pb = m_rig.bones()[size_t(p.poseBone)];
    pose::PhysicalValue   v;
    v.bone = p.poseBone;
    Vec3 t{pb.posed(ChannelKind::eTranslation, 0), pb.posed(ChannelKind::eTranslation, 1), pb.posed(ChannelKind::eTranslation, 2)};
    for(int a = 0; a < 3; a++)
      v.rotation[size_t(a)] = pb.shown(ChannelKind::eRotation, a);
    for(const auto& [key, value] : values)
    {
      const std::string prefix = pb.id + "/";
      if(key.rfind(prefix, 0) != 0)
        continue;
      const std::string rest = key.substr(prefix.size());
      const int         a    = rest.size() > 2 ? pose::axisIndex(rest.substr(rest.rfind('/') + 1)) : -1;
      if(a < 0 || a > 2)
        continue;
      if(rest.rfind("rotation/", 0) == 0)
        v.rotation[size_t(a)] = value;
      else if(rest.rfind("translation/", 0) == 0)
        (a == 0 ? t.x : a == 1 ? t.y : t.z) = value;
    }
    if(p.parent < 0)
      v.translation = t;
    list.push_back(v);
  }
  m_rig.setPhysical(list, m_owned);
  placeBodies();
  m_lastTargets = targets();
  allowOverlaps();
  m_world.sleep();
  m_version++;
}

bool FigurePhysics::writeBack()
{
  // each part's bone as its body has it; the bones above a part's own, merged into its parent, ride
  // with that part as the scene has them
  std::vector<pose::PhysicalValue> values;
  bool                             moved = false;
  std::vector<Mat4>                bone(m_parts.size());
  for(size_t i = 0; i < m_parts.size(); i++)
  {
    const RigidBody& b = m_world.bodies[i];
    bone[i]            = compose(b.x - rotate(m_parts[i].com, b.q), b.q, {1, 1, 1});
  }
  for(size_t i = 0; i < m_parts.size(); i++)
  {
    Part&                 p          = m_parts[i];
    const pose::PoseBone& pb         = m_rig.bones()[size_t(p.poseBone)];
    const int             parentNode = m_scene.nodes[size_t(p.node)].parent;
    Mat4                  above;
    if(p.parent >= 0)
      above = bone[size_t(p.parent)] * (inverse(boneWorld(m_parts[size_t(p.parent)].node)) * boneWorld(parentNode));
    else
      above = boneWorld(parentNode);
    const Quat            local  = rotationOf(above).inverse() * m_world.bodies[i].q;
    const auto            totals = m_rig.rotationTotals(pb, local, p.last);
    pose::PhysicalValue   v;
    v.bone = p.poseBone;
    for(int a = 0; a < 3; a++)
    {
      v.rotation[size_t(a)] = totals[size_t(a)] - pb.driven[size_t(ChannelKind::eRotation)][size_t(a)];
      moved                 = moved || std::abs(totals[size_t(a)] - p.last[size_t(a)]) > 1e-3;
    }
    p.last = totals;
    if(p.parent < 0)
    {
      // the hip's translation: where its joint is under the root bone, from its rest
      const Vec3 at   = inverse(above).transformPoint(originOf(bone[i]));
      const Vec3 rest = pb.rest.empty() ? Vec3{} : pb.rest.front().second;
      const auto& d   = pb.driven[size_t(ChannelKind::eTranslation)];
      const Vec3 t{at.x - rest.x - d[0], at.y - rest.y - d[1], at.z - rest.z - d[2]};
      const auto& was = pb.physical[size_t(ChannelKind::eTranslation)];
      moved           = moved || std::abs(t.x - was[0]) > 1e-4 || std::abs(t.y - was[1]) > 1e-4 || std::abs(t.z - was[2]) > 1e-4;
      v.translation   = t;
    }
    values.push_back(v);
  }
  if(!moved)
    return false;
  m_rig.setPhysical(values, m_owned);
  return true;
}

bool FigurePhysics::step(double seconds)
{
  if(!m_settings.on || m_parts.empty())
    return false;
  fv::timing::Scope t("physics: step", 1);
  if(m_rig.restVersion() != m_restVersion)
  {
    // formulas moved the joints: the parts again, where the bones are
    build();
    placeBodies();
    allowOverlaps();
    wake();
  }
  if(std::vector<double> now = targets(); now != m_lastTargets)
  {
    m_lastTargets = std::move(now);
    allowOverlaps();
    wake();
  }
  // a clip that holds still a moment is still playing
  if(m_clip)
    wake();
  bool wrote = false;
  if(awake())
  {
    m_world.settings.gravity = m_settings.gravity ? m_gravity : Vec3{};
    // where the drives and the hip aimed at the last frame's end, swept from while a clip plays
    std::vector<Quat> from;
    Pin&              hip     = m_world.pins[size_t(m_hipPin)];
    const Vec3        hipFrom = hip.target;
    const Quat        hipTurn = hip.rotation;
    const bool sweep = m_clip && m_aimed;
    if(sweep)
      for(size_t i = 1; i < m_parts.size(); i++)
        from.push_back(m_world.joints[size_t(m_parts[i].joint)].target);
    updateJoints();
    feedForward();
    // a clip's pins: where they aimed at the last frame's end, swept from while it plays
    std::vector<std::pair<Vec3, Quat>> pinFrom;
    aimDrivePins(sweep ? &pinFrom : nullptr);
    m_aimed = true;
    if(sweep)
    {
      std::vector<Quat> to;
      for(size_t i = 1; i < m_parts.size(); i++)
        to.push_back(m_world.joints[size_t(m_parts[i].joint)].target);
      const Vec3 hipTo = hip.target;
      const Quat hipAt = hip.rotation;
      m_world.onSubstep = [this, from = std::move(from), to = std::move(to), hipFrom, hipTo, hipTurn, hipAt, pinFrom = std::move(pinFrom)](double t) {
        for(size_t i = 0; i < from.size(); i++)
          m_world.joints[size_t(m_parts[i + 1].joint)].target = slerp(from[i], to[i], t);
        Pin& h     = m_world.pins[size_t(m_hipPin)];
        h.target   = hipFrom + (hipTo - hipFrom) * t;
        h.rotation = slerp(hipTurn, hipAt, t);
        for(size_t i = 0; i < pinFrom.size(); i++)
        {
          Pin&              p = m_world.pins[size_t(m_drivePin) + i];
          const Drive::Pin& d = m_drive.pins[i];
          p.target            = pinFrom[i].first + (d.at - pinFrom[i].first) * t;
          p.rotation          = slerp(pinFrom[i].second, d.rotation, t);
        }
      };
    }
    m_world.props = colliders ? colliders() : std::vector<ColliderRef>{};
    m_world.step(seconds);
    m_world.onSubstep = nullptr;
    m_sinceInput = m_pull >= 0 ? 0 : m_sinceInput + seconds;
    // still shivering long after the last input: asleep anyway
    if(m_sinceInput > 5 && !m_world.asleep())
      m_world.sleep();
    wrote = writeBack();
  }
  if(awake() != m_wasAwake)
  {
    m_wasAwake = awake();
    m_version++;
  }
  return wrote;
}

}  // namespace fv::physics
