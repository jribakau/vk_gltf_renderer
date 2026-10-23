#include <figure/pose/pose_rig.hpp>

#include <algorithm>
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
    m_byName[pb.name] = m_bones.size();
    m_byId[pb.id]     = m_bones.size();
    m_bones.push_back(std::move(pb));
  }
  for(const dson::Rig* f : followers)
    addFollowers(*f);
}

// A follower bone moves with the figure bone of the same name, or with the figure itself for its
// root (src/lib/pose/PoseRig.ts leaderOf; cross-generation aliases come with fitting).
void PoseRig::addFollowers(const dson::Rig& rig)
{
  std::set<size_t> touched;
  for(const auto& [id, node] : rig.bones)
  {
    size_t leader = SIZE_MAX;
    if(node == rig.root)
    {
      if(auto it = m_byName.find(m_scene.nodes[m_mainRoot].name); it != m_byName.end())
        leader = it->second;
    }
    else if(auto it = m_byName.find(m_scene.nodes[node].name); it != m_byName.end())
      leader = it->second;
    if(leader == SIZE_MAX)
      continue;
    PoseBone& pb = m_bones[leader];
    pb.followers.push_back(node);
    // an item fitted after the joints moved was built at the moved centres: its base is without them
    pb.rest.emplace_back(node, m_scene.nodes[node].position);
    m_restBase[node] = m_scene.nodes[node].position - localOffset(pb);
    touched.insert(leader);
  }
  for(size_t i : touched)
    apply(m_bones[i]);
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
  for(const auto& [node, rest] : pb.rest)
  {
    Node& b       = m_scene.nodes[node];
    b.quaternion  = q;
    b.position    = rest + t;
    b.scale       = s;
    moved.push_back(node);
    m_moved.insert(node);
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
