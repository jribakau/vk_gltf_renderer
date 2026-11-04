#pragma once

// Posing for DAZ figures: per-bone rotation (degrees), translation (cm) and scale channels with the
// figure's own limits, applied the DAZ way: about each joint's orientation, in its rotation order.
// Conforming clothing has its own skeleton with the same bone names; those bones follow. Formulas
// (pose/figure_driver.hpp) add to the posed values and move joint centres. A port of
// src/lib/pose/PoseRig.ts.

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
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
// A channel's axis as an index: x 0, y 1, z 2, general 3; -1 for anything else.
int axisIndex(std::string_view axis);

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
  // where each kind's axis is in `axes` (by axisIndex), -1 where the bone has none
  std::array<std::array<int8_t, 4>, 3> slot{};
  Quat                                 orient;
  std::string                          order;  // three Euler order
  // rest position of the bone and of each follower
  std::vector<std::pair<int, Vec3>> rest;
  // what formulas add to its channels, by kind and axis (as `slot`), whether it has the axis or not,
  // and how far they move its joint centre (cm, figure space, relative to the load)
  std::array<std::array<double, 4>, 3> driven{};
  Vec3                                 centerOffset;
  // Physics mode (docs/physics-plan.md, "The physical layer in PoseRig"): where the ragdoll has the
  // bone, as channel values before formulas, by kind and axis, whether the bone has the axis or not.
  // While `physicalOn`, the bone is posed there; its channels' values stay what the motors aim for.
  std::array<std::array<double, 4>, 3> physical{};
  bool                                 physicalOn = false;

  const Axis* axis(ChannelKind kind, int axis) const
  {
    const int s = axis >= 0 && axis < 4 ? slot[size_t(kind)][size_t(axis)] : -1;
    return s >= 0 ? &axes[size_t(s)] : nullptr;
  }
  // a channel's posed value before formulas: 0, or 1 for a scale, where the bone has no such axis.
  // In Physics mode, the target: the history, presets and the scene file read this.
  double posed(ChannelKind kind, int axis) const
  {
    const Axis* a = this->axis(kind, axis);
    return a ? a->value : kind == ChannelKind::eScale ? 1.0 : 0.0;
  }
  // what the bone shows: the physical value in Physics mode (rotation and translation), else posed().
  // Formulas read this, so correctives follow the ragdoll.
  double shown(ChannelKind kind, int axis) const
  {
    return physicalOn && kind != ChannelKind::eScale && axis >= 0 && axis < 4 ? physical[size_t(kind)][size_t(axis)] : posed(kind, axis);
  }
};

// Where the ragdoll has a bone (PoseRig::bones() index): its rotation channels (degrees) and, for a
// free bone (the hip), its translation (cm), both before formulas.
struct PhysicalValue
{
  int                   bone = -1;
  std::array<double, 3> rotation{};
  std::optional<Vec3>   translation;
};

// What formulas add to one channel of a bone (PoseRig::bones() index, axisIndex).
struct DrivenValue
{
  int         bone = -1;
  ChannelKind kind = ChannelKind::eRotation;
  int         axis = 0;
  double      value = 0;
};
// How far formulas move a bone's joint centre (PoseRig::bones() index; cm, figure space, from the load).
struct CenterOffset
{
  int  bone = -1;
  Vec3 offset;
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

// Eyeball bones: Genesis `lEye`/`rEye`, and the "Left Eye" names older figures use. Not lids or lashes.
bool isEyeBone(const std::string& name);

class PoseRig
{
public:
  PoseRig(SceneGraph& scene, const dson::Rig& main, const std::vector<const dson::Rig*>& followers);

  const std::vector<PoseBone>& bones() const { return m_bones; }
  // a bone by name, else by id
  const PoseBone* get(const std::string& nameOrId) const;
  // where a bone is in bones(): by name, else by id, or by id first; -1 for none
  int indexOf(const std::string& bone, bool idFirst = false) const;

  // Sets channels (clamped to their limits where the figure says so) and poses the bones. Then
  // `afterPose` runs, so formulas that read the pose see it. Returns the bone nodes moved by the
  // pose itself; takeMoved() has everything that moved since last asked, formulas included.
  std::vector<int> setMany(const std::vector<ChannelValue>& values);
  // Every channel back to the value it loaded with.
  std::vector<int> reset();
  // How many bones differ from how they loaded.
  int changed() const;

  // Whether the figure has eyeball bones with a rotation channel to aim.
  bool hasEyes() const;
  // Points the eyeballs at a world-space point, such as the camera. The rotation channels become
  // the joint-frame Euler that turns each bone's rest gaze (+Z) onto that point. A formula already
  // bending an eye is left in place: the channel is set so the two together still land on the aim.
  // Returns how many eyes were aimed. `worldPoint` is in the same space as the scene graph.
  int aimEyes(const Vec3& worldPoint);

  // Poses a bone to a local rotation (in its parent's frame) and, where it has translation
  // channels, a local position: the inverse of how channels pose it. Formula offsets stay in place
  // and limits apply, so the bone may stop short of `q`. Returns the bone nodes moved, as setMany.
  std::vector<int> setLocalTransform(const std::string& bone, const Quat& q, const std::optional<Vec3>& position);
  // The posed bone a scene node is (its followers are not), or nullptr.
  const PoseBone* boneForNode(int node) const;
  // The posable bone whose segment, from its joint to its child joints and end point, passes
  // closest to a world point (a click on the body). The face rig (every bone under the head) and
  // the figure root are left out, so a click on the face picks the head.
  const PoseBone* nearestBone(const Vec3& worldPoint) const;

  // The bone nodes that are posed bones or followers (to know which scene nodes the rig drives).
  std::set<int> drivenNodes() const;

  // The value a bone channel is posed at, before formulas add to it (a bone by id, else name).
  std::optional<double> posed(const std::string& bone, const std::string& kind, const std::string& axis) const;
  bool                  hasBone(const std::string& bone) const;

  // Sets what formulas add to bone channels, all of it: a channel not in `values` gets nothing. Moves
  // the bones whose total changed.
  void setDriven(const std::vector<DrivenValue>& values);
  // Moves joint centres by formula offsets, all of them, as setDriven: each bone's rest position and
  // its followers' move, and the bones are re-posed. The skins need re-binding at the new centres
  // (rebind).
  void setCenterOffsets(const std::vector<CenterOffset>& offsets);

  // Physics mode: poses the bones in `values` where the ragdoll has them, all of it as setDriven: a
  // bone not in `values` shows its channels again. Only bones whose pose changed move; then
  // `afterPose` runs once, if any did. Returns the bone nodes moved. `owned` (bones() indices) are the
  // bones the caller poses, when it isn't the only one (the ragdoll and the chest's soft bones): of
  // the others, none is touched. Empty: all.
  std::vector<int> setPhysical(const std::vector<PhysicalValue>& values, std::span<const int> owned = {});
  // Leaving Physics mode: each bone's channels take where it is shown (clamped to their limits),
  // and the physical layer goes. The pose doesn't move. `owned` as above.
  void foldPhysical(std::span<const int> owned = {});
  // Whether any bone is posed by the ragdoll.
  bool anyPhysical() const;
  // The rotation channel totals (value plus driven, degrees, x y z) that pose a bone at local
  // rotation `q`: of q's two Euler forms, each any number of turns round, the one nearest `near`.
  std::array<double, 3> rotationTotals(const PoseBone& pb, const Quat& q, const std::array<double, 3>& near) const;
  // A bone's local transform as its channels would pose it (value plus driven), Physics mode aside:
  // where the ragdoll's motors aim.
  Quat targetRotation(const PoseBone& pb) const;
  Vec3 targetPosition(const PoseBone& pb) const;
  // Bumps whenever formulas move joint centres (setCenterOffsets): the ragdoll fits its capsules again.
  int restVersion() const { return m_restVersion; }

  // Re-binds skinned meshes to their bones' rest positions as they are now: each bone's inverse
  // bind matrix becomes the inverse of where it is at rest, so the rest pose deforms nothing and a
  // bend pivots at the new centre. Returns the meshes whose boneInverses changed.
  std::vector<int> rebind();

  // Called after every pose change, once the bones moved: the figure runs its formulas here.
  std::function<void()> afterPose;

  // A worn item's skeleton follows the figure's bones of the same name (and its root the figure
  // itself), or `plan->follow` when it was made for another generation; the bones are re-posed.
  // Bones in `plan->inherit` already move with a parent that follows, so they count as matched and
  // are not given the leader's rotation a second time. Returns {matched, total} bones.
  struct FollowPlan
  {
    const std::map<std::string, std::string>* follow  = nullptr;
    const std::set<std::string>*              inherit = nullptr;
  };
  std::pair<int, int> addFollowers(const dson::Rig& rig, const FollowPlan& plan);
  std::pair<int, int> addFollowers(const dson::Rig& rig) { return addFollowers(rig, FollowPlan{}); }
  // An item taken off: its bones stop following.
  void removeFollowers(const dson::Rig& rig);

  // The bone nodes that moved since last asked.
  std::set<int> takeMoved();
  // Whether joint centres moved since last asked (the skins want re-binding).
  bool takeRestMoved();

private:
  Axis*            axis(PoseBone& pb, ChannelKind kind, const std::string& axis);
  const Axis*      axis(const PoseBone& pb, ChannelKind kind, const std::string& axis) const;
  std::vector<int> apply(PoseBone& pb);
  // the posed value plus what formulas add, or `fallback` plus that without the axis
  static double total(const PoseBone& pb, ChannelKind kind, int axis, double fallback);
  // Rotation channels, in degrees, that turn this bone's local +Z toward `worldPoint`.
  std::optional<std::array<double, 3>> aimAngles(const PoseBone& pb, const Vec3& worldPoint) const;
  Vec3             localOffset(const PoseBone& pb) const;

  SceneGraph&                                  m_scene;
  int                                          m_mainRoot = -1;
  std::vector<PoseBone>                        m_bones;
  std::unordered_map<std::string, size_t>      m_byName, m_byId;
  std::unordered_map<int, size_t>              m_byNode;
  // setDriven's and setCenterOffsets' working space: each bone's new values (NaN: not driven), and
  // which bones changed
  std::vector<std::array<std::array<double, 4>, 3>> m_nextDriven;
  std::vector<Vec3>                                 m_nextCenter;
  std::vector<uint8_t>                              m_marked;
  // each bone's rest position with no centre offsets: the main bones' and their followers'
  std::unordered_map<int, Vec3>                m_restBase;
  // a bone node's own scale, as its channels and formulas have it, before inherits_scale is divided out
  std::unordered_map<int, Vec3>                m_ownScale;
  // the bone that writes a node: its own, or its leader for a follower
  std::unordered_map<int, size_t>              m_writer;
  std::set<int>                                m_moved;
  bool                                         m_restMoved   = false;
  int                                          m_restVersion = 0;
};

}  // namespace fv::pose
