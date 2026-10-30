#include <figure/pose/pose_rig.hpp>
#include <figure/io/timing.hpp>

#include <algorithm>
#include <cmath>
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

std::string groupOf(const std::string& name, bool isRoot)
{
  if(isRoot)
    return "Figure";
  std::string n = name;
  std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  static const std::regex face(
      R"(^(l|r)?(eye|ear)|brow|lid|lip|cheek|nose|nostril|jaw|tongue|teeth|mouth|chin|squint|nasolabial|face|forehead|temple|upperfacerig|lowerfacerig|belowjaw|center|lash)");
  static const std::regex head(R"(^(head|neck|necklower|neckupper))");
  static const std::regex torso(R"(^(hip|pelvis|abdomen|chest|spine|lpectoral|rpectoral|pectoral))");
  static const std::regex arm(R"((collar|shldr|forearm|hand|carpal|thumb|index|mid|ring|pinky))");
  static const std::regex leg(R"((thigh|shin|foot|toe|metatarsal|heel))");
  if(std::regex_search(n, face))
    return "Face";
  if(std::regex_search(n, head))
    return "Head & neck";
  if(std::regex_search(n, torso))
    return "Torso";
  const std::string side = !n.empty() && n[0] == 'l' ? "Left" : !n.empty() && n[0] == 'r' ? "Right" : "";
  if(!side.empty() && std::regex_search(n, arm))
    return side + " arm";
  if(!side.empty() && std::regex_search(n, leg))
    return side + " leg";
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
    auto add = [&](ChannelKind kind, const std::vector<JointChannel>& list) {
      for(const JointChannel& c : list)
        if(!c.locked && c.max > c.min)
        {
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
    // an item fitted after the joints moved was built at the moved centres: its base is without them
    pb.rest.emplace_back(node, m_scene.nodes[node].position);
    m_restBase[node] = m_scene.nodes[node].position - localOffset(pb);
    touched.insert(leader);
  }
  for(size_t i : touched)
    apply(m_bones[i]);
  return {matched, total};
}

void PoseRig::removeFollowers(const dson::Rig& rig)
{
  std::set<int> nodes;
  for(const auto& [id, node] : rig.bones)
    nodes.insert(node);
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

Axis* PoseRig::axis(PoseBone& pb, ChannelKind kind, const std::string& axisName)
{
  for(Axis& a : pb.axes)
    if(a.kind == kind && a.axis == axisName)
      return &a;
  return nullptr;
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
      const std::string key    = pb.id + "|rotation|" + axisName;
      const auto        driven = m_driven.find(key);
      const double      add    = driven == m_driven.end() ? 0 : driven->second;
      const int         i      = axisName[0] == 'x' ? 0 : axisName[0] == 'y' ? 1 : 2;
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
  auto drivenOf = [&](ChannelKind kind, const char* axisName) {
    const auto it = m_driven.find(pb->id + "|" + kindName(kind) + "|" + axisName);
    return it == m_driven.end() ? 0.0 : it->second;
  };
  std::vector<ChannelValue> values;
  const char*               names[3] = {"x", "y", "z"};
  // where the bone's Euler is now, formulas included
  std::array<double, 3> now{};
  for(int i = 0; i < 3; i++)
  {
    const Axis* a = axis(*pb, ChannelKind::eRotation, names[i]);
    now[i]        = (a ? a->value : 0) + drivenOf(ChannelKind::eRotation, names[i]);
  }
  // q = orient · R · orient⁻¹, so the channel Euler is R. R has two Euler forms, (a, b, c) and
  // (a + 180, 180 - b, c + 180) with b the middle axis, each any number of turns round: take the
  // one nearest where the bone is, so a drag past ±90 or ±180 keeps going the same way
  const std::array<double, 3> euler = eulerFromQuat(pb->orient.inverse() * q.normalized() * pb->orient, pb->order);
  const int                   mid   = pb->order.size() == 3 ? pb->order[1] - 'X' : 1;
  auto nearest = [&](bool other, double& cost) {
    std::array<double, 3> out{};
    cost = 0;
    for(int i = 0; i < 3; i++)
    {
      out[i] = (other ? (i == mid ? 180 - euler[i] / kDeg : euler[i] / kDeg + 180) : euler[i] / kDeg);
      out[i] += 360 * std::round((now[i] - out[i]) / 360);
      cost += (out[i] - now[i]) * (out[i] - now[i]);
    }
    return out;
  };
  double                      costA = 0, costB = 0;
  const std::array<double, 3> a     = nearest(false, costA);
  const std::array<double, 3> b     = nearest(true, costB);
  const std::array<double, 3> total = costB < costA ? b : a;
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
  for(const Axis& a : pb.axes)
    if(a.kind == kind && a.axis == axisName)
      return &a;
  return nullptr;
}

std::optional<double> PoseRig::posed(const std::string& bone, const std::string& kind, const std::string& axisName) const
{
  // by id first, then by name
  const PoseBone* pb = nullptr;
  if(auto it = m_byId.find(bone); it != m_byId.end())
    pb = &m_bones[it->second];
  else if(auto it2 = m_byName.find(bone); it2 != m_byName.end())
    pb = &m_bones[it2->second];
  ChannelKind k;
  if(!pb || !parseKind(kind, k))
    return std::nullopt;
  const Axis* a = axis(*pb, k, axisName);
  return a ? a->value : (k == ChannelKind::eScale ? 1.0 : 0.0);
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

// Writes a bone's channels into it and its followers; returns the bones moved.
std::vector<int> PoseRig::apply(PoseBone& pb)
{
  auto v = [&](ChannelKind kind, const char* axisName, double fallback) {
    const Axis* a   = axis(pb, kind, axisName);
    double      out = a ? a->value : fallback;
    if(!m_driven.empty())
      if(auto it = m_driven.find(pb.id + "|" + kindName(kind) + "|" + axisName); it != m_driven.end())
        out += it->second;
    return out;
  };
  const Quat e = quatFromEuler(v(ChannelKind::eRotation, "x", 0) * kDeg, v(ChannelKind::eRotation, "y", 0) * kDeg,
                               v(ChannelKind::eRotation, "z", 0) * kDeg, pb.order);
  // rotate about the joint's own axes: orient · R · orient⁻¹
  const Quat   q = pb.orient * e * pb.orient.inverse();
  const Vec3   t{v(ChannelKind::eTranslation, "x", 0), v(ChannelKind::eTranslation, "y", 0), v(ChannelKind::eTranslation, "z", 0)};
  const double g = v(ChannelKind::eScale, "general", 1);
  const Vec3   s{v(ChannelKind::eScale, "x", 1) * g, v(ChannelKind::eScale, "y", 1) * g, v(ChannelKind::eScale, "z", 1) * g};
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
  return moved;
}

void PoseRig::setDriven(const std::map<std::string, double>& values)
{
  std::set<size_t> touched;
  auto             mark = [&](const std::string& key) {
    if(auto it = m_byId.find(key.substr(0, key.find('|'))); it != m_byId.end())
      touched.insert(it->second);
  };
  for(const auto& [key, v] : values)
  {
    auto it = m_driven.find(key);
    if(std::abs((it == m_driven.end() ? 0.0 : it->second) - v) < 1e-6)
      continue;
    if(v == 0)
      m_driven.erase(key);
    else
      m_driven[key] = v;
    mark(key);
  }
  for(auto it = m_driven.begin(); it != m_driven.end();)
  {
    if(values.count(it->first))
    {
      ++it;
      continue;
    }
    mark(it->first);
    it = m_driven.erase(it);
  }
  for(size_t i : touched)
    apply(m_bones[i]);
}

// How far a bone's rest position moves relative to its parent's: its centre offset minus its parent's.
Vec3 PoseRig::localOffset(const PoseBone& pb) const
{
  const Vec3 zero;
  auto       offsetOf = [&](const std::string& id) {
    auto it = m_centerOffsets.find(id);
    return it == m_centerOffsets.end() ? zero : it->second;
  };
  const Vec3 own = offsetOf(pb.id);
  const int  parent = pb.bone >= 0 ? m_scene.nodes[pb.bone].parent : -1;
  Vec3       up     = zero;
  if(parent >= 0)
    if(auto it = m_byNode.find(parent); it != m_byNode.end())
      up = offsetOf(m_bones[it->second].id);
  return own - up;
}

void PoseRig::setCenterOffsets(const std::map<std::string, Vec3>& offsets)
{
  auto same = [](const Vec3& a, const Vec3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
  std::set<std::string> changed;
  for(const auto& [id, v] : offsets)
  {
    auto it = m_centerOffsets.find(id);
    if(!same(it == m_centerOffsets.end() ? Vec3{} : it->second, v))
      changed.insert(id);
  }
  for(const auto& [id, v] : m_centerOffsets)
    if(!offsets.count(id))
      changed.insert(id);
  if(changed.empty())
    return;
  m_centerOffsets = offsets;
  // a bone's rest position depends on its own centre and its parent's
  std::set<size_t> affected;
  for(const std::string& id : changed)
  {
    auto it = m_byId.find(id);
    if(it == m_byId.end())
      continue;
    affected.insert(it->second);
    for(int c : m_scene.nodes[m_bones[it->second].bone].children)
      if(auto child = m_byNode.find(c); child != m_byNode.end())
        affected.insert(child->second);
  }
  for(size_t i : affected)
  {
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
      auto       it = m_centerOffsets.find(pb.id);
      const Vec3 o  = it == m_centerOffsets.end() ? Vec3{} : it->second;
      d->center     = {(*d->centerBase)[0] + o.x, (*d->centerBase)[1] + o.y, (*d->centerBase)[2] + o.z};
    }
    apply(pb);
  }
  m_restMoved = true;
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
