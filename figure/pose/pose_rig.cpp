#include <figure/pose/pose_rig.hpp>
#include <figure/io/timing.hpp>
#include <figure/wardrobe/fit/bones.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <regex>

namespace fv::pose {

const char* kindName(ChannelKind k)
{
  switch(k)
  {
    case ChannelKind::eTranslation:
      return "translation";
    case ChannelKind::eScale:
      return "scale";
    default:
      return "rotation";
  }
}

bool parseKind(const std::string& s, ChannelKind& out)
{
  if(s == "rotation")
    out = ChannelKind::eRotation;
  else if(s == "translation")
    out = ChannelKind::eTranslation;
  else if(s == "scale")
    out = ChannelKind::eScale;
  else
    return false;
  return true;
}

int axisIndex(std::string_view axis)
{
  return axis == "x" ? 0 : axis == "y" ? 1 : axis == "z" ? 2 : axis == "general" ? 3 : -1;
}

std::string groupOf(const std::string& name, bool isRoot)
{
  if(isRoot)
    return "Figure";
  // by the name with its side taken off, as the wardrobe reads it: Genesis 9's "l_eye" is an eye and
  // "l_indextoe1" a toe, not a finger
  const wardrobe::ParsedBone p = wardrobe::parseBone(name);
  const std::string&         n = p.token;
  static const std::regex    face(
      R"(^(eye|ear)|brow|lid|lip|cheek|nose|nostril|jaw|tongue|teeth|mouth|chin|squint|nasolabial|face|forehead|temple|upperfacerig|lowerfacerig|belowjaw|center|lash)");
  static const std::regex head(R"(^(head|neck|necklower|neckupper))");
  static const std::regex torso(R"(^(hip|pelvis|abdomen|chest|spine|pectoral))");
  // G8's lCollar, lShldrBend; G9's l_shoulder, l_upperarm
  static const std::regex arm(R"((collar|shldr|shoulder|upperarm|forearm|hand|carpal|thumb|index|mid|ring|pinky))");
  static const std::regex leg(R"((thigh|shin|foot|toe|metatarsal|heel))");
  if(std::regex_search(n, face))
    return "Face";
  if(std::regex_search(n, head))
    return "Head & neck";
  if(std::regex_search(n, torso))
    return "Torso";
  const std::string side = p.side == 'l' ? "Left" : p.side == 'r' ? "Right" : "";
  // the leg first: a toe's name has its finger's in it ("indextoe")
  if(!side.empty() && std::regex_search(n, leg))
    return side + " leg";
  if(!side.empty() && std::regex_search(n, arm))
    return side + " arm";
  return "Other";
}

bool isEyeBone(const std::string& name)
{
  static const std::regex eye(R"(^(l|r|left|right)[\s_-]?eye(ball)?$)", std::regex::icase);
  return std::regex_match(name, eye);
}

PoseRig::PoseRig(SceneGraph& scene, const dson::Rig& main, const std::vector<const dson::Rig*>& followers)
    : m_scene(scene)
    , m_mainRoot(main.root)
{
  static const std::set<std::string> orders{"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};
  for(const auto& [id, node] : main.bones)
  {
    const Node& bone = scene.nodes[node];
    const auto& d    = bone.joint;
    if(!d)
      continue;
    PoseBone pb;
    pb.id    = id;
    pb.name  = bone.name;
    pb.label = d->label.empty() ? bone.name : d->label;
    pb.group = groupOf(bone.name, node == main.root);
    pb.bone  = node;
    for(auto& kind : pb.slot)
      kind.fill(-1);
    auto add = [&](ChannelKind kind, const std::vector<JointChannel>& list) {
      for(const JointChannel& c : list)
        if(!c.locked && c.max > c.min)
        {
          // the first of a kind and axis is the one posed, as a search of the list finds it
          if(const int ai = axisIndex(c.axis); ai >= 0 && pb.slot[size_t(kind)][size_t(ai)] < 0)
            pb.slot[size_t(kind)][size_t(ai)] = int8_t(pb.axes.size());
          Axis a;
          static_cast<JointChannel&>(a) = c;
          a.kind                        = kind;
          a.initial                     = c.value;
          pb.axes.push_back(std::move(a));
        }
    };
    add(ChannelKind::eRotation, d->rotation);
    add(ChannelKind::eTranslation, d->translation);
    add(ChannelKind::eScale, d->scale);
    // the joint frame: orientation is an Euler in the bone's own order, as authored
    const std::string orientOrder = orders.count(d->rotationOrder) ? d->rotationOrder : "XYZ";
    pb.orient = quatFromEuler(d->orientation[0] * kDeg, d->orientation[1] * kDeg, d->orientation[2] * kDeg, orientOrder);
    pb.order  = threeOrder(d->rotationOrder).data();
    pb.rest.emplace_back(node, bone.position);
    m_restBase[node]  = bone.position;
    m_byNode[node]    = m_bones.size();
    m_writer[node]    = m_bones.size();
    m_byName[pb.name] = m_bones.size();
    m_byId[pb.id]     = m_bones.size();
    m_bones.push_back(std::move(pb));
  }
  for(const dson::Rig* f : followers)
    addFollowers(*f);
}

// A follower bone moves with the figure bone of the same name, then the cross-generation alias, or
// with the figure itself for its root (src/lib/pose/PoseRig.ts leaderOf).
std::pair<int, int> PoseRig::addFollowers(const dson::Rig& rig, const FollowPlan& plan)
{
  std::set<size_t> touched;
  int              matched = 0, total = 0;
  std::vector<std::pair<int, size_t>> added;  // follower node, leader
  for(const auto& [id, node] : rig.bones)
  {
    total++;
    size_t leader = SIZE_MAX;
    if(node == rig.root)
    {
      if(auto it = m_byName.find(m_scene.nodes[m_mainRoot].name); it != m_byName.end())
        leader = it->second;
    }
    else if(auto it = m_byName.find(m_scene.nodes[node].name); it != m_byName.end())
      leader = it->second;
    else if(plan.follow)
      if(auto alias = plan.follow->find(m_scene.nodes[node].name); alias != plan.follow->end())
        if(auto it = m_byName.find(alias->second); it != m_byName.end())
          leader = it->second;
    if(leader == SIZE_MAX)
    {
      if(plan.inherit && plan.inherit->count(m_scene.nodes[node].name))
        matched++;
      continue;
    }
    matched++;
    PoseBone& pb = m_bones[leader];
    pb.followers.push_back(node);
    m_writer[node] = leader;
    // an item fitted after the joints moved was built at the moved centres: its base is without them.
    // Those are the centres it was built at, which a dial turned while it prepared has moved since
    const Vec3 now   = localOffset(pb);
    const Vec3 built = now - (plan.builtAt ? centreMoved(pb, *plan.builtAt) : Vec3{});
    const Vec3 at    = m_scene.nodes[node].position;
    m_restBase[node] = at - built;
    pb.rest.emplace_back(node, m_restBase[node] + now);
    if(built.x != now.x || built.y != now.y || built.z != now.z)
      m_restMoved = true;
    touched.insert(leader);
    added.emplace_back(node, leader);
  }
  // the figure's joints between the bone a follower's item parent follows and its own leader
  for(const auto& [node, leader] : added)
  {
    const auto parent = m_writer.find(m_scene.nodes[size_t(node)].parent);
    if(parent == m_writer.end())
      continue;
    std::vector<size_t> chain;
    bool                reached = false;
    for(size_t x = leader, guard = 0; guard < 512; guard++)
    {
      if(x == parent->second)
      {
        reached = true;
        break;
      }
      chain.push_back(x);
      const auto up = m_byNode.find(m_scene.nodes[size_t(m_bones[x].bone)].parent);
      if(up == m_byNode.end())
        break;
      x = up->second;
    }
    // its parent's bone isn't above its leader (or is its leader): it copies its leader's turn
    if(!reached || chain.size() < 2)
      continue;
    std::reverse(chain.begin(), chain.end());
    for(const size_t x : chain)
    {
      m_onChain[x].push_back(node);
      touched.insert(x);
    }
    m_chain[node] = std::move(chain);
  }
  for(size_t i : touched)
    apply(m_bones[i]);
  return {matched, total};
}

void PoseRig::writeChain(int node)
{
  const auto chain  = m_chain.find(node);
  const auto writer = m_writer.find(node);
  if(chain == m_chain.end() || writer == m_writer.end())
    return;
  const auto restOf = [&](const PoseBone& pb, int n) {
    for(const auto& [k, rest] : pb.rest)
      if(k == n)
        return rest;
    return m_scene.nodes[size_t(n)].position;
  };
  // the chain's local transforms one after the other, and where its pivots stand at rest
  Vec3 p, p0;
  Quat q;
  Vec3 s{1, 1, 1};
  for(const size_t x : chain->second)
  {
    const PoseBone& pb = m_bones[x];
    const Node&     b  = m_scene.nodes[size_t(pb.bone)];
    p                  = p + rotate(s * b.position, q);
    q                  = q * b.quaternion;
    s                  = s * b.scale;
    p0                 = p0 + restOf(pb, pb.bone);
  }
  Node& f      = m_scene.nodes[size_t(node)];
  f.position   = restOf(m_bones[writer->second], node) + (p - p0);
  f.quaternion = q;
  f.scale      = s;
  m_moved.insert(node);
}

void PoseRig::removeFollowers(const dson::Rig& rig)
{
  std::set<int> nodes;
  for(const auto& [id, node] : rig.bones)
    nodes.insert(node);
  for(const int n : nodes)
    if(const auto chain = m_chain.find(n); chain != m_chain.end())
    {
      for(const size_t x : chain->second)
        std::erase(m_onChain[x], n);
      m_chain.erase(chain);
    }
  for(PoseBone& pb : m_bones)
  {
    pb.followers.erase(std::remove_if(pb.followers.begin(), pb.followers.end(), [&](int n) { return nodes.count(n) > 0; }), pb.followers.end());
    pb.rest.erase(std::remove_if(pb.rest.begin(), pb.rest.end(), [&](const auto& r) { return nodes.count(r.first) > 0; }), pb.rest.end());
  }
  for(int n : nodes)
  {
    m_restBase.erase(n);
    m_ownScale.erase(n);
    m_writer.erase(n);
  }
}

const PoseBone* PoseRig::get(const std::string& nameOrId) const
{
  if(auto it = m_byName.find(nameOrId); it != m_byName.end())
    return &m_bones[it->second];
  if(auto it = m_byId.find(nameOrId); it != m_byId.end())
    return &m_bones[it->second];
  return nullptr;
}

int PoseRig::indexOf(const std::string& bone, bool idFirst) const
{
  const auto& first  = idFirst ? m_byId : m_byName;
  const auto& second = idFirst ? m_byName : m_byId;
  if(auto it = first.find(bone); it != first.end())
    return int(it->second);
  if(auto it = second.find(bone); it != second.end())
    return int(it->second);
  return -1;
}

Axis* PoseRig::axis(PoseBone& pb, ChannelKind kind, const std::string& axisName)
{
  return const_cast<Axis*>(pb.axis(kind, axisIndex(axisName)));
}

std::vector<int> PoseRig::setMany(const std::vector<ChannelValue>& values)
{
  fv::timing::Accum   setTime("pose.setMany");
  std::vector<size_t> touched;
  for(const ChannelValue& v : values)
  {
    const PoseBone* found = get(v.bone);
    if(!found)
      continue;
    PoseBone& pb = m_bones[size_t(found - m_bones.data())];
    Axis*     a  = axis(pb, v.kind, v.axis);
    if(!a)
      continue;
    a->value = a->clamped ? std::clamp(v.value, a->min, a->max) : v.value;
    if(std::find(touched.begin(), touched.end(), size_t(&pb - m_bones.data())) == touched.end())
      touched.push_back(size_t(&pb - m_bones.data()));
  }
  std::vector<int> moved;
  for(size_t i : touched)
  {
    auto m = apply(m_bones[i]);
    moved.insert(moved.end(), m.begin(), m.end());
  }
  // the pose drives correctives (and possibly more): the formulas run again
  if(!touched.empty() && afterPose)
    afterPose();
  return moved;
}

std::vector<int> PoseRig::reset()
{
  std::vector<int> moved;
  bool             reposed = false;
  for(PoseBone& pb : m_bones)
  {
    bool any = false;
    for(Axis& a : pb.axes)
      if(a.value != a.initial)
      {
        a.value = a.initial;
        any     = true;
      }
    if(any)
    {
      auto m = apply(pb);
      moved.insert(moved.end(), m.begin(), m.end());
      reposed = true;
    }
  }
  if(reposed && afterPose)
    afterPose();
  return moved;
}

int PoseRig::changed() const
{
  int n = 0;
  for(const PoseBone& pb : m_bones)
    if(std::any_of(pb.axes.begin(), pb.axes.end(), [](const Axis& a) { return a.value != a.initial; }))
      n++;
  return n;
}

bool PoseRig::hasEyes() const
{
  for(const PoseBone& pb : m_bones)
    if(isEyeBone(pb.name) && std::any_of(pb.axes.begin(), pb.axes.end(), [](const Axis& a) { return a.kind == ChannelKind::eRotation; }))
      return true;
  return false;
}

std::optional<std::array<double, 3>> PoseRig::aimAngles(const PoseBone& pb, const Vec3& worldPoint) const
{
  const int parent = pb.bone >= 0 && pb.bone < int(m_scene.nodes.size()) ? m_scene.nodes[pb.bone].parent : -1;
  if(parent < 0)
    return std::nullopt;
  const Mat4 eye    = m_scene.world(pb.bone);
  const Vec3 at{eye.e[12], eye.e[13], eye.e[14]};
  Vec3       dir = worldPoint - at;
  if(dot(dir, dir) < 1e-12)
    return std::nullopt;
  dir = normalize(dir);
  // the bone quaternion rotates in the parent's frame
  dir = rotate(dir, rotationOf(m_scene.world(parent)).inverse());
  const Quat aim = quatFromUnitVectors({0, 0, 1}, dir);
  // q = orient · R · orient⁻¹, so the channel Euler is R
  const Quat                 q     = pb.orient.inverse() * aim * pb.orient;
  const std::array<double, 3> euler = eulerFromQuat(q, pb.order);
  const double               deg   = 1.0 / kDeg;
  return std::array<double, 3>{euler[0] * deg, euler[1] * deg, euler[2] * deg};
}

int PoseRig::aimEyes(const Vec3& worldPoint)
{
  std::vector<ChannelValue> values;
  int                       n = 0;
  for(const PoseBone& pb : m_bones)
  {
    if(!isEyeBone(pb.name))
      continue;
    const auto angles = aimAngles(pb, worldPoint);
    if(!angles)
      continue;
    bool wrote = false;
    for(const char* axisName : {"x", "y", "z"})
    {
      const Axis* a = axis(pb, ChannelKind::eRotation, axisName);
      if(!a)
        continue;
      const int    i   = axisName[0] == 'x' ? 0 : axisName[0] == 'y' ? 1 : 2;
      const double add = pb.driven[size_t(ChannelKind::eRotation)][size_t(i)];
      const double      total  = a->clamped ? std::clamp((*angles)[i], a->min, a->max) : (*angles)[i];
      values.push_back({pb.name, ChannelKind::eRotation, axisName, total - add});
      wrote = true;
    }
    if(wrote)
      n++;
  }
  setMany(values);
  return n;
}

std::vector<int> PoseRig::setLocalTransform(const std::string& bone, const Quat& q, const std::optional<Vec3>& position)
{
  const PoseBone* pb = get(bone);
  if(!pb)
    return {};
  auto drivenOf = [&](ChannelKind kind, const char* axisName) { return pb->driven[size_t(kind)][size_t(axisIndex(axisName))]; };
  std::vector<ChannelValue> values;
  const char*               names[3] = {"x", "y", "z"};
  // where the bone's Euler is now, formulas included
  std::array<double, 3> now{};
  for(int i = 0; i < 3; i++)
  {
    const Axis* a = axis(*pb, ChannelKind::eRotation, names[i]);
    now[i]        = (a ? a->value : 0) + drivenOf(ChannelKind::eRotation, names[i]);
  }
  // so a drag past ±90 or ±180 keeps going the same way
  const std::array<double, 3> total = rotationTotals(*pb, q, now);
  // a transform that round-trips through the renderer's floats comes back a hair off: a channel
  // that only moved that much stays as it was, so a bone isn't posed by a click
  auto push = [&](ChannelKind kind, const char* axisName, double value, double noise) {
    const Axis* a = axis(*pb, kind, axisName);
    if(a && std::abs(a->value - value) > noise)
      values.push_back({pb->name, kind, axisName, value});
  };
  for(int i = 0; i < 3; i++)
    push(ChannelKind::eRotation, names[i], total[i] - drivenOf(ChannelKind::eRotation, names[i]), 1e-3);
  if(position && !pb->rest.empty())
  {
    const Vec3   t      = *position - pb->rest.front().second;
    const double xyz[3] = {t.x, t.y, t.z};
    for(int i = 0; i < 3; i++)
      push(ChannelKind::eTranslation, names[i], xyz[i] - drivenOf(ChannelKind::eTranslation, names[i]), 1e-4);
  }
  return setMany(values);
}

std::array<double, 3> PoseRig::rotationTotals(const PoseBone& pb, const Quat& q, const std::array<double, 3>& near) const
{
  // q = orient · R · orient⁻¹, so the channel Euler is R. R has two Euler forms, (a, b, c) and
  // (a + 180, 180 - b, c + 180) with b the middle axis, each any number of turns round
  const std::array<double, 3> euler = eulerFromQuat(pb.orient.inverse() * q.normalized() * pb.orient, pb.order);
  const int                   mid   = pb.order.size() == 3 ? pb.order[1] - 'X' : 1;
  auto nearest = [&](bool other, double& cost) {
    std::array<double, 3> out{};
    cost = 0;
    for(int i = 0; i < 3; i++)
    {
      out[i] = (other ? (i == mid ? 180 - euler[i] / kDeg : euler[i] / kDeg + 180) : euler[i] / kDeg);
      out[i] += 360 * std::round((near[i] - out[i]) / 360);
      cost += (out[i] - near[i]) * (out[i] - near[i]);
    }
    return out;
  };
  double                      costA = 0, costB = 0;
  const std::array<double, 3> a     = nearest(false, costA);
  const std::array<double, 3> b     = nearest(true, costB);
  return costB < costA ? b : a;
}

Quat PoseRig::targetRotation(const PoseBone& pb) const
{
  auto v = [&](int axis) {
    const double d = pb.driven[size_t(ChannelKind::eRotation)][size_t(axis)];
    return pb.posed(ChannelKind::eRotation, axis) + d;
  };
  return pb.orient * quatFromEuler(v(0) * kDeg, v(1) * kDeg, v(2) * kDeg, pb.order) * pb.orient.inverse();
}

Vec3 PoseRig::targetPosition(const PoseBone& pb) const
{
  auto v = [&](int axis) { return pb.posed(ChannelKind::eTranslation, axis) + pb.driven[size_t(ChannelKind::eTranslation)][size_t(axis)]; };
  const Vec3 rest = pb.rest.empty() ? Vec3{} : pb.rest.front().second;
  return rest + Vec3{v(0), v(1), v(2)};
}

std::vector<int> PoseRig::setPhysical(const std::vector<PhysicalValue>& values, std::span<const int> owned)
{
  fv::timing::Accum   setTime("pose.setPhysical");
  std::vector<uint8_t> listed(m_bones.size(), 0);
  std::vector<int>     moved;
  bool                 any = false;
  const auto           repose = [&](PoseBone& pb) {
    const std::vector<int> m = apply(pb);
    moved.insert(moved.end(), m.begin(), m.end());
    any = true;
  };
  for(const PhysicalValue& v : values)
  {
    if(v.bone < 0 || size_t(v.bone) >= m_bones.size())
      continue;
    listed[size_t(v.bone)] = 1;
    PoseBone& pb           = m_bones[size_t(v.bone)];
    auto      next         = pb.physical;
    for(int i = 0; i < 3; i++)
    {
      next[size_t(ChannelKind::eRotation)][size_t(i)] = v.rotation[size_t(i)];
      // a bone held at its joint keeps its channels' translation
      next[size_t(ChannelKind::eTranslation)][size_t(i)] =
          v.translation ? (i == 0 ? v.translation->x : i == 1 ? v.translation->y : v.translation->z) : pb.posed(ChannelKind::eTranslation, i);
    }
    if(pb.physicalOn && next == pb.physical)
      continue;
    pb.physical   = next;
    pb.physicalOn = true;
    repose(pb);
  }
  std::vector<uint8_t> mine(m_bones.size(), owned.empty() ? 1 : 0);
  for(const int b : owned)
    if(b >= 0 && size_t(b) < m_bones.size())
      mine[size_t(b)] = 1;
  for(size_t i = 0; i < m_bones.size(); i++)
    if(!listed[i] && mine[i] && m_bones[i].physicalOn)
    {
      m_bones[i].physicalOn = false;
      repose(m_bones[i]);
    }
  if(any && afterPose)
    afterPose();
  return moved;
}

void PoseRig::foldPhysical(std::span<const int> owned)
{
  bool reposed = false;
  for(size_t b = 0; b < m_bones.size(); b++)
  {
    PoseBone& pb = m_bones[b];
    if(!pb.physicalOn || (!owned.empty() && std::find(owned.begin(), owned.end(), int(b)) == owned.end()))
      continue;
    bool clamped = false;
    for(Axis& a : pb.axes)
    {
      const int i = axisIndex(a.axis);
      if(a.kind == ChannelKind::eScale || i < 0 || i > 2)
        continue;
      const double v = pb.physical[size_t(a.kind)][size_t(i)];
      a.value        = a.clamped ? std::clamp(v, a.min, a.max) : v;
      clamped        = clamped || a.value != v;
    }
    pb.physicalOn = false;
    // what it showed on an axis it has no channel for (a hair off 0) goes, as it would on a clamp
    apply(pb);
    reposed = reposed || clamped;
  }
  if(reposed && afterPose)
    afterPose();
}

bool PoseRig::anyPhysical() const
{
  return std::any_of(m_bones.begin(), m_bones.end(), [](const PoseBone& pb) { return pb.physicalOn; });
}

const PoseBone* PoseRig::boneForNode(int node) const
{
  const auto it = m_byNode.find(node);
  return it == m_byNode.end() ? nullptr : &m_bones[it->second];
}

const PoseBone* PoseRig::nearestBone(const Vec3& worldPoint) const
{
  // the face rig: everything under the head (names vary by generation: lEye, l_eye, r_infraorbital...)
  std::set<int> face;
  for(const PoseBone& pb : m_bones)
  {
    std::string n = pb.name;
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if(n != "head")
      continue;
    std::vector<int> stack(m_scene.nodes[pb.bone].children);
    while(!stack.empty())
    {
      const int c = stack.back();
      stack.pop_back();
      face.insert(c);
      stack.insert(stack.end(), m_scene.nodes[c].children.begin(), m_scene.nodes[c].children.end());
    }
  }
  auto isFace  = [&](const PoseBone& pb) { return pb.group == "Face" || face.count(pb.bone) > 0; };
  auto posable = [&](const PoseBone& pb) { return !pb.axes.empty() && !isFace(pb) && pb.group != "Figure"; };
  auto toSegment = [&](const Vec3& a, const Vec3& b) {
    const Vec3   ab = b - a;
    const double l2 = dot(ab, ab);
    const double t  = l2 > 0 ? std::clamp(dot(worldPoint - a, ab) / l2, 0.0, 1.0) : 0.0;
    const Vec3   d  = worldPoint - (a + ab * t);
    return dot(d, d);
  };
  const PoseBone* best     = nullptr;
  double          bestDist = 0;
  for(const PoseBone& pb : m_bones)
  {
    if(!posable(pb) || pb.bone < 0 || pb.bone >= int(m_scene.nodes.size()))
      continue;
    const Node& node = m_scene.nodes[pb.bone];
    const Mat4  w    = m_scene.world(pb.bone);
    const Vec3  joint{w.e[12], w.e[13], w.e[14]};
    std::vector<Vec3> ends;
    for(int c : node.children)
      if(auto it = m_byNode.find(c); it != m_byNode.end() && !isFace(m_bones[it->second]))
      {
        const Mat4 cw = m_scene.world(c);
        ends.push_back({cw.e[12], cw.e[13], cw.e[14]});
      }
    // the bone's end, from its rest centre: at rest a bone's frame is the figure's, at its centre
    if(const auto& d = node.joint)
    {
      const Vec3 tail{d->endPoint[0] - d->center[0], d->endPoint[1] - d->center[1], d->endPoint[2] - d->center[2]};
      if(dot(tail, tail) > 1e-8)
        ends.push_back(w.transformPoint(tail));
    }
    double dist = ends.empty() ? toSegment(joint, joint) : -1;
    for(const Vec3& e : ends)
    {
      const double de = toSegment(joint, e);
      if(dist < 0 || de < dist)
        dist = de;
    }
    if(!best || dist < bestDist)
    {
      best     = &pb;
      bestDist = dist;
    }
  }
  return best;
}

std::set<int> PoseRig::drivenNodes() const
{
  std::set<int> out;
  for(const PoseBone& pb : m_bones)
    for(const auto& [node, rest] : pb.rest)
      out.insert(node);
  return out;
}


const Axis* PoseRig::axis(const PoseBone& pb, ChannelKind kind, const std::string& axisName) const
{
  return pb.axis(kind, axisIndex(axisName));
}

std::optional<double> PoseRig::posed(const std::string& bone, const std::string& kind, const std::string& axisName) const
{
  const int   b = indexOf(bone, true);
  ChannelKind k;
  if(b < 0 || !parseKind(kind, k))
    return std::nullopt;
  return m_bones[size_t(b)].posed(k, axisIndex(axisName));
}

bool PoseRig::hasBone(const std::string& bone) const
{
  return m_byId.count(bone) || m_byName.count(bone);
}

std::set<int> PoseRig::takeMoved()
{
  return std::exchange(m_moved, {});
}

bool PoseRig::takeRestMoved()
{
  return std::exchange(m_restMoved, false);
}

double PoseRig::total(const PoseBone& pb, ChannelKind kind, int axis, double fallback)
{
  const Axis*  a      = pb.axis(kind, axis);
  double       out    = pb.physicalOn && kind != ChannelKind::eScale ? pb.physical[size_t(kind)][size_t(axis)] : a ? a->value : fallback;
  const double driven = pb.driven[size_t(kind)][size_t(axis)];
  // nothing driven adds nothing, not even a zero's sign
  if(driven != 0)
    out += driven;
  return out;
}

// Writes a bone's channels into it and its followers; returns the bones moved.
std::vector<int> PoseRig::apply(PoseBone& pb)
{
  auto v = [&](ChannelKind kind, int axis, double fallback) { return total(pb, kind, axis, fallback); };
  const Quat e = quatFromEuler(v(ChannelKind::eRotation, 0, 0) * kDeg, v(ChannelKind::eRotation, 1, 0) * kDeg,
                               v(ChannelKind::eRotation, 2, 0) * kDeg, pb.order);
  // rotate about the joint's own axes: orient · R · orient⁻¹
  const Quat   q = pb.orient * e * pb.orient.inverse();
  const Vec3   t{v(ChannelKind::eTranslation, 0, 0), v(ChannelKind::eTranslation, 1, 0), v(ChannelKind::eTranslation, 2, 0)};
  const double g = v(ChannelKind::eScale, 3, 1);
  const Vec3   s{v(ChannelKind::eScale, 0, 1) * g, v(ChannelKind::eScale, 1, 1) * g, v(ChannelKind::eScale, 2, 1) * g};
  std::vector<int> moved;
  bool             rescaled = false;
  for(const auto& [node, rest] : pb.rest)
  {
    auto [own, added] = m_ownScale.try_emplace(node, Vec3{1, 1, 1});
    if(own->second.x != s.x || own->second.y != s.y || own->second.z != s.z)
    {
      own->second = s;
      rescaled    = true;
    }
  }
  for(const auto& [node, rest] : pb.rest)
  {
    // a chain follower is written below, from its chain
    if(m_chain.count(node))
      continue;
    Node& b       = m_scene.nodes[node];
    b.quaternion  = q;
    b.position    = rest + t;
    // DAZ's inherits_scale off: the parent's own scale moves this bone's origin but doesn't size it, while
    // what the parent inherited does. The scene's nodes multiply down the tree, so it's divided out here
    Vec3 local = s;
    if(b.joint && !b.joint->inheritsScale)
      if(const auto parent = m_ownScale.find(b.parent); parent != m_ownScale.end())
      {
        const Vec3& p = parent->second;
        local         = Vec3{p.x != 0 ? s.x / p.x : s.x, p.y != 0 ? s.y / p.y : s.y, p.z != 0 ? s.z / p.z : s.z};
      }
    b.scale = local;
    moved.push_back(node);
    m_moved.insert(node);
  }
  // a change of this bone's own scale changes what its children divide out
  if(rescaled)
  {
    std::set<size_t> children;
    for(const auto& [node, rest] : pb.rest)
      for(const int child : m_scene.nodes[node].children)
        if(const auto writer = m_writer.find(child); writer != m_writer.end() && &m_bones[writer->second] != &pb)
          children.insert(writer->second);
    for(const size_t child : children)
    {
      const std::vector<int> theirs = apply(m_bones[child]);
      moved.insert(moved.end(), theirs.begin(), theirs.end());
    }
  }
  // the followers whose chain runs through this bone
  if(const auto on = m_onChain.find(size_t(&pb - m_bones.data())); on != m_onChain.end())
    for(const int node : on->second)
    {
      writeChain(node);
      moved.push_back(node);
    }
  return moved;
}

void PoseRig::setDriven(const std::vector<DrivenValue>& values)
{
  constexpr double kNone = std::numeric_limits<double>::quiet_NaN();
  std::array<std::array<double, 4>, 3> none;
  for(auto& kind : none)
    kind.fill(kNone);
  m_nextDriven.assign(m_bones.size(), none);
  for(const DrivenValue& v : values)
    if(v.bone >= 0 && size_t(v.bone) < m_bones.size() && v.axis >= 0 && v.axis < 4)
      m_nextDriven[size_t(v.bone)][size_t(v.kind)][size_t(v.axis)] = v.value;
  for(size_t i = 0; i < m_bones.size(); i++)
  {
    PoseBone& pb      = m_bones[i];
    bool      touched = false;
    for(size_t k = 0; k < 3; k++)
      for(size_t a = 0; a < 4; a++)
      {
        double&      now  = pb.driven[k][a];
        const double next = m_nextDriven[i][k][a];
        // a channel nothing drives any more goes back to its own value; one that moves less than this stays
        if(std::isnan(next) ? now == 0 : std::abs(now - next) < 1e-6)
          continue;
        now     = std::isnan(next) ? 0 : next;
        touched = true;
      }
    if(touched)
      apply(pb);
  }
}

// How far a bone's rest position moves relative to its parent's: its centre offset minus its parent's.
Vec3 PoseRig::localOffset(const PoseBone& pb) const
{
  const int parent = pb.bone >= 0 ? m_scene.nodes[pb.bone].parent : -1;
  Vec3      up;
  if(parent >= 0)
    if(auto it = m_byNode.find(parent); it != m_byNode.end())
      up = m_bones[it->second].centerOffset;
  return pb.centerOffset - up;
}

// How far a bone's centre has moved since `builtAt` recorded it, less its parent's: what its local
// offset gained. A bone it doesn't name hasn't moved.
Vec3 PoseRig::centreMoved(const PoseBone& pb, const std::map<int, std::array<double, 3>>& builtAt) const
{
  const auto moved = [&](int node) {
    const auto  at = builtAt.find(node);
    const auto& j  = m_scene.nodes[size_t(node)].joint;
    if(at == builtAt.end() || !j)
      return Vec3{};
    return Vec3{j->center[0] - at->second[0], j->center[1] - at->second[1], j->center[2] - at->second[2]};
  };
  if(pb.bone < 0)
    return {};
  const int parent = m_scene.nodes[size_t(pb.bone)].parent;
  Vec3      up;
  if(parent >= 0 && m_byNode.count(parent))
    up = moved(parent);
  return moved(pb.bone) - up;
}

void PoseRig::setCenterOffsets(const std::vector<CenterOffset>& offsets)
{
  m_nextCenter.assign(m_bones.size(), Vec3{});
  for(const CenterOffset& o : offsets)
    if(o.bone >= 0 && size_t(o.bone) < m_bones.size())
      m_nextCenter[size_t(o.bone)] = o.offset;
  // a bone's rest position depends on its own centre and its parent's
  m_marked.assign(m_bones.size(), 0);
  bool any = false;
  for(size_t i = 0; i < m_bones.size(); i++)
  {
    PoseBone&   pb   = m_bones[i];
    const Vec3& next = m_nextCenter[i];
    if(pb.centerOffset.x == next.x && pb.centerOffset.y == next.y && pb.centerOffset.z == next.z)
      continue;
    pb.centerOffset = next;
    any             = true;
    m_marked[i]     = 1;
    for(int c : m_scene.nodes[pb.bone].children)
      if(auto child = m_byNode.find(c); child != m_byNode.end())
        m_marked[child->second] = 1;
  }
  if(!any)
    return;
  for(size_t i = 0; i < m_bones.size(); i++)
  {
    if(!m_marked[i])
      continue;
    PoseBone&  pb    = m_bones[i];
    const Vec3 local = localOffset(pb);
    for(auto& [node, rest] : pb.rest)
    {
      auto base = m_restBase.find(node);
      rest      = (base != m_restBase.end() ? base->second : rest) + local;
    }
    if(auto& d = m_scene.nodes[pb.bone].joint)
    {
      if(!d->centerBase)
        d->centerBase = d->center;
      const Vec3& o = pb.centerOffset;
      d->center     = {(*d->centerBase)[0] + o.x, (*d->centerBase)[1] + o.y, (*d->centerBase)[2] + o.z};
    }
    apply(pb);
  }
  m_restMoved = true;
  m_restVersion++;
}

std::vector<int> PoseRig::rebind()
{
  fv::timing::Accum rebindTime("pose.rebind");
  struct Saved
  {
    int  node;
    Vec3 position;
    Quat quaternion;
    Vec3 scale;
  };
  std::vector<Saved> saved;
  std::set<int>      atRest;
  for(const PoseBone& pb : m_bones)
    for(const auto& [node, rest] : pb.rest)
    {
      Node& b = m_scene.nodes[node];
      saved.push_back({node, b.position, b.quaternion, b.scale});
      atRest.insert(node);
      b.position   = rest;
      b.quaternion = Quat{};
      b.scale      = Vec3{1, 1, 1};
    }
  const std::vector<Mat4> worlds = m_scene.worlds();
  std::vector<int>        changed;
  for(size_t mi = 0; mi < m_scene.nodes.size(); mi++)
  {
    const Node& n = m_scene.nodes[mi];
    if(n.kind != NodeKind::eMesh)
      continue;
    Mesh& mesh = m_scene.meshes[size_t(n.mesh)];
    if(!mesh.skinned() || !std::any_of(mesh.bones.begin(), mesh.bones.end(), [&](int b) { return atRest.count(b) > 0; }))
      continue;
    // three skins as matrixWorld · bindMatrix⁻¹ · bone · inverse · bindMatrix ('attached' mode)
    const Mat4 m = worlds[mi] * inverse(mesh.bindMatrix);
    mesh.boneInverses.resize(mesh.bones.size());
    for(size_t i = 0; i < mesh.bones.size(); i++)
      mesh.boneInverses[i] = inverse(worlds[size_t(mesh.bones[i])]) * m;
    changed.push_back(n.mesh);
  }
  for(const Saved& s : saved)
  {
    Node& b      = m_scene.nodes[s.node];
    b.position   = s.position;
    b.quaternion = s.quaternion;
    b.scale      = s.scale;
  }
  return changed;
}

}  // namespace fv::pose
