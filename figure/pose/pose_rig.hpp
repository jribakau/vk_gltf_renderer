#pragma once

// Posing for DAZ figures: per-bone rotation (degrees), translation (cm) and scale channels with the
// figure's own limits, applied the DAZ way: about each joint's orientation, in its rotation order.
// Conforming clothing has its own skeleton with the same bone names; those bones follow. Formulas
// (pose/figure_driver.hpp) add to the posed values and move joint centres. A port of
// src/lib/pose/PoseRig.ts.

#include <functional>
#include <map>
#include <optional>
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
  // a bone by name, else by id
  const PoseBone* get(const std::string& nameOrId) const;

  // Sets channels (clamped to their limits where the figure says so) and poses the bones. Then
  // `afterPose` runs, so formulas that read the pose see it. Returns the bone nodes moved by the
  // pose itself; takeMoved() has everything that moved since last asked, formulas included.
  std::vector<int> setMany(const std::vector<ChannelValue>& values);
  // Every channel back to the value it loaded with.
  std::vector<int> reset();
  // How many bones differ from how they loaded.
  int changed() const;

  // The bone nodes that are posed bones or followers (to know which scene nodes the rig drives).
  std::set<int> drivenNodes() const;

  // The value a bone channel is posed at, before formulas add to it (a bone by id, else name).
  std::optional<double> posed(const std::string& bone, const std::string& kind, const std::string& axis) const;
  bool                  hasBone(const std::string& bone) const;

  // Sets what formulas add to bone channels ("bone id|kind|axis" -> amount), and moves the bones
  // whose total changed.
  void setDriven(const std::map<std::string, double>& values);
  // Moves joint centres by formula offsets (bone id -> cm in figure space, relative to the load):
  // each bone's rest position and its followers' move, and the bones are re-posed. The skins need
  // re-binding at the new centres (rebind).
  void setCenterOffsets(const std::map<std::string, Vec3>& offsets);

  // Re-binds skinned meshes to their bones' rest positions as they are now: each bone's inverse
  // bind matrix becomes the inverse of where it is at rest, so the rest pose deforms nothing and a
  // bend pivots at the new centre. Returns the meshes whose boneInverses changed.
  std::vector<int> rebind();

  // Called after every pose change, once the bones moved: the figure runs its formulas here.
  std::function<void()> afterPose;

  // The bone nodes that moved since last asked.
  std::set<int> takeMoved();
  // Whether joint centres moved since last asked (the skins want re-binding).
  bool takeRestMoved();

private:
  Axis*            axis(PoseBone& pb, ChannelKind kind, const std::string& axis);
  const Axis*      axis(const PoseBone& pb, ChannelKind kind, const std::string& axis) const;
  std::vector<int> apply(PoseBone& pb);
  void             addFollowers(const dson::Rig& rig);
  Vec3             localOffset(const PoseBone& pb) const;

  SceneGraph&                                  m_scene;
  int                                          m_mainRoot = -1;
  std::vector<PoseBone>                        m_bones;
  std::unordered_map<std::string, size_t>      m_byName, m_byId;
  std::unordered_map<int, size_t>              m_byNode;
  // formula contributions on top of the posed values, by "bone id|kind|axis"
  std::map<std::string, double>                m_driven;
  // joint centre offsets from formulas (cm, figure space), by bone id, relative to the load
  std::map<std::string, Vec3>                  m_centerOffsets;
  // each bone's rest position with no centre offsets: the main bones' and their followers'
  std::unordered_map<int, Vec3>                m_restBase;
  std::set<int>                                m_moved;
  bool                                         m_restMoved = false;
};

}  // namespace fv::pose
