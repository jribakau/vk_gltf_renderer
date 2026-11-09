#pragma once

// The skeleton spec (docs/wardrobe-plan.md, step 2): one table of anatomical joints that every rig's
// bones map into, and the one place that says what a joint is. A bone name is read into a side and
// a joint by its rig family's alias table (rig/aliases); what used to be spread over the wardrobe's
// slot tables, the fitter's aim joints, the limb filter, the physics parts and the Pose tab's groups is
// a column of this table, and those readers are shims over it (wardrobe::parseBone, matchBone,
// limbOfBone, pose::groupOf, the fitter's nextSlots, the ragdoll's part slots).
//
// Ids are the slot names the wardrobe has used: Genesis 8's lShldrBend and Genesis 9's l_upperarm are
// both the left "upperarm"; Genesis 9's l_shoulder is the "collar"; Skyrim's NPC Spine1 is "spine2".
// A joint has no side: the side is the bone name's. Names a rig has that the table doesn't know (a
// face rig, a skirt's bones, a hair's) keep their own token and no joint.
//
// Not recorded yet, for the steps that read them: each family's rest direction per joint (step 3
// measures it), the regions a body's weights give the glutes (no joint has them).
//
// The conventions every rig is brought to (DAZ's; test RigSpec.ABonesTurnTranslationAndScaleAreTheSpecsConventions):
// - At rest every bone's world rotation is the identity, in centimetres, Y up; a bone's axes are its
//   `orientation`. A rig with bind rotations (glTF, VRM, Mixamo) has them baked into it at import.
// - Rotation: for a rotation order "ABC" (DSON's `rotation_order`, one of the six), angles a, b, c
//   about A, B, C, and the orientation's angles o:
//     O    = R_A(o_A) R_B(o_B) R_C(o_C)            (the orientation's first axis outermost)
//     turn = O R_C(c) R_B(b) R_A(a) O^-1           (the pose's first axis innermost: the twist axis
//                                                   goes first, so a bend doesn't swing it)
//   R_k(t) the turn by t about world axis k, applied right to left. Not yet checked against a DAZ
//   Studio export of a bone with a multi-axis orientation: the step's open check.
// - Translation is in the parent's frame, not turned by O.
// - Scale is along the rest's axes (the parent's at rest), before the turn; general scale multiplies
//   each axis; a bone that doesn't inherit scale divides its parent's own out.
// - Shape formulas move `center_point` alone; `end_point` stays the base figure's, so nothing here
//   leans on end points.

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fv::rig {

// The coarse limb the wardrobe binds by (wardrobe::limbOfBone, BindFilter): one side's leg is three,
// its arm two; the side comes from the name.
enum class Limb : uint8_t
{
  eNone,
  eHead,
  eArm,
  eThigh,
  eShin,
  eFoot,
  eHand,
};

// The finer region a body map matches within (step 3): each finger and toe its own, the breast apart
// from the chest.
enum class Region : uint8_t
{
  eNone,
  ePelvis,
  eAbdomen,
  eChest,
  eBreast,
  eNeck,
  eHead,
  eCollar,
  eUpperArm,
  eForearm,
  eHand,
  eThumb,
  eIndex,
  eMid,
  eRing,
  ePinky,
  eThigh,
  eShin,
  eFoot,
  eBigToe,
  eIndexToe,
  eMidToe,
  eRingToe,
  ePinkyToe,
};

// The Pose tab's group (pose::groupOf); an arm or leg joint is grouped by its side.
enum class Group : uint8_t
{
  eTorso,
  eHeadNeck,
  eArm,
  eLeg,
};

// The ragdoll (physics::FigurePhysics): a part of its own, or merged into the part above.
enum class Part : uint8_t
{
  eNone,
  eOwn,
  eMerged,
};

// Whether a body map may hold the joint as a landmark (step 3): a limb joint along its chain; a foot
// along its length only, as its height off the floor differs between the figures; never a pectoral,
// whose joint lies 8 cm apart between Genesis 8 and 9.
enum class Landmark : uint8_t
{
  eNone,
  eFull,
  eAlongLength,
};

struct Joint
{
  std::string_view id;
  std::string_view parent;    // "" at the root
  bool             sided = false;
  // the joint to try when a figure has nothing in this one (wardrobe::matchBone): a metatarsal on a
  // figure without one follows the foot
  std::string_view fallback;
  // the joints down the limb that set its direction, nearest first (the fitter's bone transfer)
  std::array<std::string_view, 2> aims{};
  Limb     limb     = Limb::eNone;
  Region   region   = Region::eNone;
  Group    group    = Group::eTorso;
  Part     part     = Part::eNone;
  Landmark landmark = Landmark::eNone;
};

// Every joint, parents before children.
std::span<const Joint> joints();
// The joint of that id, or null.
const Joint* joint(std::string_view id);

// The rig families a skeleton's names are read as. Genesis 2, 3, 8 and 9 share one alias table, as
// their names don't collide; Skyrim's names are its own.
enum class Family : uint8_t
{
  eGenesis2,
  eGenesis3,
  eGenesis8,  // and 8.1, which shares its skeleton
  eGenesis9,
  eSkyrim,
};

// Where a family keeps a limb's twist: as a bone in the chain the user poses (Genesis 3 and 8's
// lForearmTwist), as helpers beside it that its formulas turn (Genesis 9's numbered twists, Skyrim's),
// or nowhere (Genesis 2).
enum class TwistPlacement : uint8_t
{
  eNone,
  eInChain,
  eHelpers,
};
TwistPlacement twistPlacement(Family family);

// A bone name read.
struct Name
{
  char        side = 0;  // 0, 'l' or 'r'
  std::string token;     // lowercase, side and separators removed: `lBigToe_2` -> `bigtoe2`
  std::string slot;      // the joint's id, or the token itself when the table does not know it
  bool        twist = false;
  bool        skyrim = false;  // read as a Skyrim name ("NPC L Thigh [LThg]")
  const Joint* joint = nullptr;  // null for a name the table doesn't know
};
// `lThighBend` -> left thigh, `l_toes` -> left toes, `NPC Spine1 [Spn1]` -> spine2, `lip` stays a whole
// token with no side and no joint.
Name parse(const std::string& name);

// A name's limb: its joint's, else a toe or finger joint past the table's (a fourth knuckle, a fifth
// toe) by its slot's ending.
Limb limbOf(const Name& name);

// The joint a Genesis token is (its alias, else the token itself as an id), or null: what the Pose
// tab groups by the table rather than by the name's words.
const Joint* genesisJoint(const std::string& token);

// What a skeleton's names say of its family: one family when only it has such names, else whether
// they are Genesis 3 or 8's (Bend/Twist limbs, a lower and upper chest), which share their names but
// for the heel.
struct Evidence
{
  std::optional<Family> sure;
  bool                  genesis3or8 = false;
};
Evidence identify(const std::vector<std::string>& boneNames);

}  // namespace fv::rig
