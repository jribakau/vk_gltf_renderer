#pragma once

// Posing for DAZ figures: per-bone rotation (degrees), translation (cm) and scale channels with the
// figure's own limits, applied the DAZ way: about each joint's orientation, in its rotation order.
// Conforming clothing has its own skeleton with the same bone names; those bones follow. A port of
// src/lib/pose/PoseRig.ts; the formulas that add to the pose (FigureDriver) join it with the morphs.

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <figure/dson/rig.hpp>
#include <figure/scene/graph.hpp>

namespace fv::pose {

enum class ChannelKind
{
  eRotation,
  eTranslation,
  eScale,
};

const char* kindName(ChannelKind k);
bool        parseKind(const std::string& s, ChannelKind& out);

struct Axis : JointChannel
{
  ChannelKind kind = ChannelKind::eRotation;
  // the value it loads with; "Reset" returns here
  double initial = 0;
};

struct PoseBone
{
  std::string id, name, label, group;
  int         bone = -1;
  // same-named bones of conforming clothing, which move with this one
  std::vector<int>  followers;
  std::vector<Axis> axes;
  Quat              orient;
  std::string       order;  // three Euler order
  // rest position of the bone and of each follower
  std::vector<std::pair<int, Vec3>> rest;
};

struct ChannelValue
{
  std::string bone;  // name or id
  ChannelKind kind;
  std::string axis;
  double      value;
};

// The panel group a bone is listed under ("Torso", "Left arm", ...), from its name.
std::string groupOf(const std::string& name, bool isRoot);

class PoseRig
{
public:
  PoseRig(SceneGraph& scene, const dson::Rig& main, const std::vector<const dson::Rig*>& followers);

  const std::vector<PoseBone>& bones() const { return m_bones; }
  const PoseBone*              get(const std::string& nameOrId) const;

  // Sets channels (clamped to their limits where the figure says so) and poses the bones. Returns
  // the bone nodes moved, followers included.
  std::vector<int> setMany(const std::vector<ChannelValue>& values);
  // Every channel back to the value it loaded with.
  std::vector<int> reset();
  // How many bones differ from how they loaded.
  int changed() const;

  // The bone nodes that are posed bones or followers (to know which scene nodes the rig drives).
  std::set<int> drivenNodes() const;

private:
  Axis*            axis(PoseBone& pb, ChannelKind kind, const std::string& axis);
  std::vector<int> apply(PoseBone& pb);
  void             addFollowers(const dson::Rig& rig);

  SceneGraph&                                  m_scene;
  int                                          m_mainRoot = -1;
  std::vector<PoseBone>                        m_bones;
  std::unordered_map<std::string, size_t>      m_byName, m_byId;
};

}  // namespace fv::pose
