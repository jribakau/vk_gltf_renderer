// A DAZ animation made for one Genesis played on another (figure/pose/retarget.hpp): on two
// skeletons that differ as Genesis 8 and 9 do (names, a limb split into bend and twist bones or not,
// bones one has and the other hasn't, the arms' and legs' rest angles, a jaw's end placed otherwise),
// every limb turns from its own rest as the file turns it on its own skeleton, and what the file holds
// at rest stays as the figure has it.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <figure/dson/rig.hpp>
#include <figure/pose/pose_clip.hpp>
#include <figure/pose/pose_rig.hpp>
#include <figure/pose/retarget.hpp>

using fv::Vec3;
using fv::pose::ChannelKind;

namespace {

constexpr double kPi = 3.14159265358979323846;

std::string triple(const char* key, Vec3 p)
{
  return std::string("\"") + key + R"(":[{"id":"x","value":)" + std::to_string(p.x) + R"(},{"id":"y","value":)" + std::to_string(p.y) + R"(},{"id":"z","value":)"
         + std::to_string(p.z) + "}]";
}
std::string channels(const char* kind, const std::string& axes, double range)
{
  std::string s = std::string("\"") + kind + "\":[";
  for(char a : std::string("xyz"))
  {
    const bool on = axes.find(a) != std::string::npos;
    s += std::string(a == 'x' ? "" : ",") + R"({"id":")" + a + R"(","min":)" + std::to_string(on ? -range : 0) + R"(,"max":)" + std::to_string(on ? range : 0)
         + R"(,"clamped":true})";
  }
  return s + "]";
}
// `order` as DAZ gives it: its first axis is the bone's twist
std::string bone(const std::string& id, const std::string& parent, Vec3 centre, Vec3 end, const std::string& axes, const char* order, Vec3 orient = {},
                 bool moves = false)
{
  std::string s = R"({"id":")" + id + R"(","type":"bone","name":")" + id + R"(","parent":"#)" + parent + R"(",)" + triple("center_point", centre) + ","
                  + triple("end_point", end) + "," + triple("orientation", orient) + R"(,"rotation_order":")" + order + R"(",)" + channels("rotation", axes, 170);
  if(moves)
    s += "," + channels("translation", "xyz", 1000);
  return s + "}";
}

// A figure standing on y = 0 with its arms `armDown` degrees below level and its legs `legOut` out
// from straight down. Genesis 8's way: limbs bent on one bone and twisted on the next, metatarsals
// between the foot and the toes, its own names; else Genesis 9's.
std::string skeleton(bool g8, double armDown, double legOut, double hipHeight)
{
  const auto name = [&](const char* g9, const char* g8name) { return std::string(g8 ? g8name : g9); };
  std::vector<std::string> b;
  const std::string        root = g8 ? "Genesis8Female" : "Genesis9";
  b.push_back(R"({"id":")" + root + R"(","type":"figure","name":")" + root + R"(",)" + triple("center_point", {}) + "}");
  const double h = hipHeight;
  b.push_back(bone("hip", root, {0, h, 0}, {0, h - 10, 0}, "xyz", "YZX", {}, true));
  b.push_back(bone("pelvis", "hip", {0, h, 0}, {0, h - 8, 0}, "xyz", "YZX"));
  b.push_back(bone(name("spine1", "abdomenLower"), "hip", {0, h + 2, 0}, {0, h + 12, 0}, "xyz", "YZX"));
  b.push_back(bone(name("spine2", "chestUpper"), name("spine1", "abdomenLower"), {0, h + 12, 0}, {0, h + 40, 0}, "xyz", "YZX"));
  // a head with a jaw, and a pectoral, whose ends each rig places its own way: some 20° apart
  b.push_back(bone("head", name("spine2", "chestUpper"), {0, h + 48, 0}, {0, h + 60, 0}, "xyz", "YZX"));
  b.push_back(bone(name("lowerjaw", "lowerJaw"), "head", {0, h + 50, 2}, g8 ? Vec3{0, h + 46, 8} : Vec3{0, h + 48, 9}, "xyz", "XYZ"));
  b.push_back(bone(name("l_pectoral", "lPectoral"), name("spine2", "chestUpper"), {6, h + 30, 4}, g8 ? Vec3{8, h + 27, 12} : Vec3{9, h + 30, 12}, "xyz", "XYZ"));
  for(const double s : {1.0, -1.0})
  {
    const auto side = [&](const std::string& n) { return (s > 0 ? (g8 ? "l" : "l_") : (g8 ? "r" : "r_")) + n; };
    const double a  = armDown * kPi / 180, l = legOut * kPi / 180;
    const Vec3   shoulder{16 * s, h + 38, 0};
    const Vec3   elbow   = shoulder + Vec3{28 * s * std::cos(a), -28 * std::sin(a), 0};
    const Vec3   wrist   = elbow + Vec3{26 * s * std::cos(a), -26 * std::sin(a), 0};
    const Vec3   hand    = wrist + Vec3{10 * s * std::cos(a), -10 * std::sin(a), 0};
    // each arm bone's axes lie along the arm, its twist axis (x, the order's first) down it; a leg's y
    const Vec3 armOrient{0, 0, -s * armDown}, legOrient{0, 0, s * legOut};
    b.push_back(bone(side(g8 ? "Collar" : "shoulder"), name("spine2", "chestUpper"), {4 * s, h + 38, 0}, shoulder, "xyz", "XYZ"));
    if(g8)
    {
      const Vec3 mid = shoulder + (elbow - shoulder) * 0.5, fmid = elbow + (wrist - elbow) * 0.5;
      b.push_back(bone(side("ShldrBend"), side("Collar"), shoulder, mid, "yz", "XYZ", armOrient));
      b.push_back(bone(side("ShldrTwist"), side("ShldrBend"), mid, elbow, "x", "XYZ", armOrient));
      b.push_back(bone(side("ForearmBend"), side("ShldrTwist"), elbow, fmid, "y", "XZY", armOrient));
      b.push_back(bone(side("ForearmTwist"), side("ForearmBend"), fmid, wrist, "x", "XZY", armOrient));
      b.push_back(bone(side("Hand"), side("ForearmTwist"), wrist, hand, "xyz", "XYZ", armOrient));
    }
    else
    {
      b.push_back(bone(side("upperarm"), side("shoulder"), shoulder, elbow, "xyz", "XYZ", armOrient));
      // Genesis 9's twist bones hang at the side of the bend, turned by its formulas: no chain goes through them
      const Vec3 mid = shoulder + (elbow - shoulder) * 0.5;
      b.push_back(bone(side("upperarmtwist1"), side("upperarm"), shoulder, mid, "x", "XYZ", armOrient));
      b.push_back(bone(side("upperarmtwist2"), side("upperarm"), mid, elbow, "x", "XYZ", armOrient));
      b.push_back(bone(side("forearm"), side("upperarm"), elbow, wrist, "xy", "XZY", armOrient));
      b.push_back(bone(side("hand"), side("forearm"), wrist, hand, "xyz", "XYZ", armOrient));
    }
    const Vec3 hipJoint{9 * s, h - 6, 0};
    const Vec3 knee  = hipJoint + Vec3{s * 44 * std::sin(l), -44 * std::cos(l), 0};
    const Vec3 ankle = knee + Vec3{s * 42 * std::sin(l), -42 * std::cos(l), 0};
    const Vec3 ball  = ankle + Vec3{0, -6, 12}, toe = ball + Vec3{0, 0, 5};
    if(g8)
    {
      const Vec3 mid = hipJoint + (knee - hipJoint) * 0.5;
      b.push_back(bone(side("ThighBend"), "pelvis", hipJoint, mid, "xz", "YZX", legOrient));
      b.push_back(bone(side("ThighTwist"), side("ThighBend"), mid, knee, "y", "YZX", legOrient));
      b.push_back(bone(side("Shin"), side("ThighTwist"), knee, ankle, "xyz", "YZX", legOrient));
      b.push_back(bone(side("Foot"), side("Shin"), ankle, ankle + Vec3{0, -3, 6}, "xyz", "ZYX"));
      b.push_back(bone(side("Metatarsals"), side("Foot"), ankle + Vec3{0, -3, 6}, ball, "xz", "ZYX"));
      b.push_back(bone(side("Toe"), side("Metatarsals"), ball, toe, "xz", "ZYX"));
    }
    else
    {
      b.push_back(bone(side("thigh"), "pelvis", hipJoint, knee, "xyz", "YZX", legOrient));
      b.push_back(bone(side("thightwist1"), side("thigh"), hipJoint, knee, "y", "YZX", legOrient));
      b.push_back(bone(side("shin"), side("thigh"), knee, ankle, "xyz", "YZX", legOrient));
      b.push_back(bone(side("foot"), side("shin"), ankle, ball, "xyz", "ZYX"));
      b.push_back(bone(side("toes"), side("foot"), ball, toe, "xz", "ZYX"));
    }
  }
  std::string out = "[";
  for(size_t i = 0; i < b.size(); i++)
    out += (i ? "," : "") + b[i];
  return out + "]";
}

struct Figure
{
  fv::SceneGraph                     scene;
  fv::dson::Rig                      rig;
  std::unique_ptr<fv::pose::PoseRig> pose;

  Figure(bool g8, double armDown, double legOut, double hipHeight)
      : Figure(fv::dson::Json::parse(skeleton(g8, armDown, legOut, hipHeight)))
  {
  }
  explicit Figure(const fv::dson::Json& bones)
  {
    auto built = fv::dson::buildRig(scene, bones, nullptr);
    EXPECT_TRUE(built);
    rig  = *built;
    pose = std::make_unique<fv::pose::PoseRig>(scene, rig, std::vector<const fv::dson::Rig*>{});
  }
  Vec3 at(const std::string& bone) const
  {
    const fv::Mat4 w = scene.world(pose->get(bone)->bone);
    return {w.e[12], w.e[13], w.e[14]};
  }
  // which way the bone from `a`'s joint to `b`'s points
  Vec3 way(const std::string& a, const std::string& b) const { return fv::normalize(at(b) - at(a)); }
  // which way a bone points from its joint to its end
  Vec3 tip(const std::string& bone) const
  {
    const fv::Node& n = scene.nodes[size_t(pose->get(bone)->bone)];
    return turned(bone, {n.joint->endPoint[0] - n.joint->center[0], n.joint->endPoint[1] - n.joint->center[1], n.joint->endPoint[2] - n.joint->center[2]});
  }
  // `v` (unit) turned as the bone is turned from its rest (a bone at rest has no turn in the scene)
  Vec3 turned(const std::string& bone, const Vec3& v) const
  {
    const fv::Mat4 w = scene.world(pose->get(bone)->bone);
    return fv::normalize({w.e[0] * v.x + w.e[4] * v.y + w.e[8] * v.z, w.e[1] * v.x + w.e[5] * v.y + w.e[9] * v.z, w.e[2] * v.x + w.e[6] * v.y + w.e[10] * v.z});
  }
  double rotation(const std::string& bone, int axis) const { return pose->get(bone)->posed(ChannelKind::eRotation, axis); }
  void apply(const fv::pose::PoseClip& clip, float t)
  {
    std::vector<fv::pose::ChannelValue> values;
    for(const auto& c : clip.channels)
      if(c.target.kind == fv::pose::PresetTarget::Kind::eBone)
        values.push_back({c.target.bone, c.target.channel, c.target.axis, fv::pose::sample(c, t)});
    pose->setMany(values);
  }
};

double degreesBetween(const Vec3& a, const Vec3& b)
{
  return std::acos(std::clamp(fv::dot(a, b), -1.0, 1.0)) * 180 / kPi;
}

// An animation keyed by hand over a second: the left leg swinging, the knee and foot bending, the arm
// lowering, swinging and twisting with the elbow and wrist bent, the back and head turning, the hip bobbing, and the jaw
// and pectoral held at rest (as Mocap Online's walks hold Genesis 8's whole face) or the jaw opened
fv::dson::Json walkFile(bool g8, double jawOpen = 0)
{
  const auto key = [](const std::string& bone, const char* channel, std::vector<std::pair<double, double>> keys) {
    fv::dson::Json a = {{"url", "name://@selection/" + bone + ":?" + channel + "/value"}, {"keys", fv::dson::Json::array()}};
    for(const auto& [t, v] : keys)
      a["keys"].push_back({t, v});
    return a;
  };
  fv::dson::Json anims = fv::dson::Json::array();
  if(g8)
  {
    anims.push_back(key("lThighBend", "rotation/x", {{0, 0}, {0.5, -30}, {1, 10}}));
    anims.push_back(key("lThighBend", "rotation/z", {{0, 0}, {0.5, 6}, {1, -4}}));
    anims.push_back(key("lThighTwist", "rotation/y", {{0, 0}, {0.5, 12}, {1, -8}}));
    anims.push_back(key("lShin", "rotation/x", {{0, 0}, {0.5, 60}, {1, 15}}));
    anims.push_back(key("lFoot", "rotation/x", {{0, 0}, {0.5, 20}, {1, -10}}));
    anims.push_back(key("lMetatarsals", "rotation/x", {{0, 0}, {0.5, 15}, {1, 5}}));
    anims.push_back(key("lShldrBend", "rotation/z", {{0, 0}, {0.5, -25}, {1, -20}}));
    anims.push_back(key("lShldrBend", "rotation/y", {{0, 0}, {0.5, -15}, {1, 10}}));
    anims.push_back(key("lShldrTwist", "rotation/x", {{0, 0}, {0.5, 30}, {1, -20}}));
    anims.push_back(key("lForearmBend", "rotation/y", {{0, 0}, {0.5, -50}, {1, -20}}));
    anims.push_back(key("lHand", "rotation/z", {{0, 0}, {0.5, 10}, {1, -5}}));
    anims.push_back(key("abdomenLower", "rotation/y", {{0, 0}, {0.5, 8}, {1, -6}}));
    anims.push_back(key("head", "rotation/y", {{0, 0}, {0.5, 20}, {1, -10}}));
    for(const char* a : {"rotation/x", "rotation/y", "rotation/z"})
    {
      anims.push_back(key("lowerJaw", a, {{0, std::string(a) == "rotation/x" ? jawOpen : 0}}));
      anims.push_back(key("lPectoral", a, {{0, 0}}));
    }
    anims.push_back(key("hip", "translation/y", {{0, 0}, {0.5, -4}, {1, 2}}));
  }
  else
  {
    anims.push_back(key("l_thigh", "rotation/x", {{0, 0}, {0.5, -30}, {1, 10}}));
    anims.push_back(key("l_thigh", "rotation/y", {{0, 0}, {0.5, 12}, {1, -8}}));
    anims.push_back(key("l_shin", "rotation/x", {{0, 0}, {0.5, 60}, {1, 15}}));
    anims.push_back(key("l_upperarm", "rotation/z", {{0, 0}, {0.5, -25}, {1, -20}}));
    anims.push_back(key("l_upperarm", "rotation/x", {{0, 0}, {0.5, 30}, {1, -20}}));
    anims.push_back(key("l_forearm", "rotation/y", {{0, 0}, {0.5, -50}, {1, -20}}));
  }
  return {{"asset_info", {{"type", "preset_pose"}}}, {"scene", {{"animations", anims}}}};
}

}  // namespace

// The elbows of the two skeletons bend about axes 4° apart, as their arms rest 4° apart: one can't
// bend about the other's, so a 50° bend leaves the forearm this far off (degrees) and the hand makes it up
constexpr double kElbowApart = 2;

// Genesis 8's walk on a Genesis 9: every limb turns from its own rest as Genesis 8's turns from its
// rest, at every key, though the two rest 4° to 6° apart
TEST(Retarget, aGenesis8AnimationTurnsAGenesis9sLimbsAsItsOwn)
{
  Figure                   g8(true, 48, 8, 105), g9(false, 44, 2, 97);
  const Figure             rest(false, 44, 2, 97);
  const fv::dson::Json     file = walkFile(true);
  const fv::pose::PoseClip own  = fv::pose::loadPoseClip(g8.pose.get(), nullptr, file);
  const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *g8.pose, *g9.pose);
  ASSERT_FALSE(clip.channels.empty());
  EXPECT_FLOAT_EQ(clip.duration, 1.f);
  // each of Genesis 9's limbs, from a joint to the next, and Genesis 8's bone whose turn is the limb's
  struct Limb
  {
    const char* source;
    const char* from;
    const char* to;
    double      within;
  };
  const Limb limbs[] = {
      {"lThighTwist", "l_thigh", "l_shin", 0.5},
      {"lShin", "l_shin", "l_foot", 0.5},
      {"lFoot", "l_foot", "l_toes", 0.5},
      {"lShldrTwist", "l_upperarm", "l_forearm", 0.5},
      {"lForearmTwist", "l_forearm", "l_hand", kElbowApart},
      {"abdomenLower", "spine1", "spine2", 0.5},
      {"lCollar", "l_shoulder", "l_upperarm", 0.5},
  };
  for(const float t : {0.f, 0.25f, 0.5f, 0.8f, 1.f})
  {
    g8.apply(own, t);
    g9.apply(clip, t);
    for(const Limb& l : limbs)
      EXPECT_LT(degreesBetween(g8.turned(l.source, rest.way(l.from, l.to)), g9.way(l.from, l.to)), l.within) << l.from << " at " << t;
    // the last bones, by their own ends. Genesis 9 has no metatarsals: their turn reaches its toes,
    // which the file doesn't move itself
    for(const auto& [source, bone] : {std::pair{"lToe", "l_toes"}, std::pair{"lHand", "l_hand"}, std::pair{"head", "head"}})
      EXPECT_LT(degreesBetween(g8.turned(source, rest.tip(bone)), g9.tip(bone)), 0.5) << bone << " at " << t;
  }
  // the hip bobs as far for its height
  g8.apply(own, 0.5f);
  g9.apply(clip, 0.5f);
  EXPECT_NEAR(g9.pose->get("hip")->posed(ChannelKind::eTranslation, 1), -4 * 97.0 / 105, 1e-3);
}

// And Genesis 9's on a Genesis 8: the turn of a limb Genesis 9 bends and twists on one bone is split
// between Genesis 8's bend and twist bones, each on its own axes
TEST(Retarget, aGenesis9AnimationSplitsALimbsTwistOntoItsTwistBone)
{
  Figure                   g8(true, 48, 8, 105), g9(false, 44, 2, 97);
  const Figure             rest(true, 48, 8, 105);
  const fv::dson::Json     file = walkFile(false);
  const fv::pose::PoseClip own  = fv::pose::loadPoseClip(g9.pose.get(), nullptr, file);
  const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *g9.pose, *g8.pose);
  ASSERT_FALSE(clip.channels.empty());
  for(const float t : {0.25f, 0.5f, 1.f})
  {
    g9.apply(own, t);
    g8.apply(clip, t);
    EXPECT_LT(degreesBetween(g9.turned("l_thigh", rest.way("lThighBend", "lShin")), g8.way("lThighBend", "lShin")), 0.5) << t;
    EXPECT_LT(degreesBetween(g9.turned("l_shin", rest.way("lShin", "lFoot")), g8.way("lShin", "lFoot")), 0.5) << t;
    EXPECT_LT(degreesBetween(g9.turned("l_upperarm", rest.way("lShldrBend", "lForearmBend")), g8.way("lShldrBend", "lForearmBend")), 0.5) << t;
    EXPECT_LT(degreesBetween(g9.turned("l_forearm", rest.way("lForearmBend", "lHand")), g8.way("lForearmBend", "lHand")), kElbowApart) << t;
  }
  EXPECT_NE(g8.pose->get("lShldrTwist")->posed(ChannelKind::eRotation, 0), 0.0) << "the twist went to the twist bone";
}

// What a figure's bone can't take of the source's turn, the bones below make up (wardrobe plan 0n): a
// Genesis 8 elbow held to 30° under Genesis 9's 50° bend leaves the rest to the hand, and its twist bone,
// whose axis lies 8° off the forearm and which takes only its twist, leaves the hand what else it was
// given. The hand points where the file's does; it was solved as though its arm had taken all of it
// (23° off at the held bend, 4° at the twist)
TEST(Retarget, whatABoneCantTakeTheBonesBelowMakeUp)
{
  fv::dson::Json bones = fv::dson::Json::parse(skeleton(true, 48, 8, 105));
  for(fv::dson::Json& b : bones)
  {
    if(b.value("id", "") == "lForearmBend")
      for(fv::dson::Json& c : b["rotation"])
        if(c.value("id", "") == "y")
        {
          c["min"] = -30;
          c["max"] = 30;
        }
    if(b.value("id", "") == "lForearmTwist")
      for(fv::dson::Json& c : b["orientation"])
        if(c.value("id", "") == "z")
          c["value"] = c.value("value", 0.0) + 8;
  }
  Figure       g8(bones), g9(false, 44, 2, 97);
  const Figure rest(true, 48, 8, 105);
  const auto   key = [](const std::string& bone, const char* channel, std::vector<std::pair<double, double>> keys) {
    fv::dson::Json a = {{"url", "name://@selection/" + bone + ":?" + channel + "/value"}, {"keys", fv::dson::Json::array()}};
    for(const auto& [t, v] : keys)
      a["keys"].push_back({t, v});
    return a;
  };
  fv::dson::Json anims = fv::dson::Json::array();
  anims.push_back(key("l_upperarm", "rotation/z", {{0, 0}, {0.5, -25}, {1, -20}}));
  anims.push_back(key("l_forearm", "rotation/y", {{0, 0}, {0.5, -50}, {1, -20}}));
  anims.push_back(key("l_forearm", "rotation/x", {{0, 0}, {0.5, 40}, {1, -30}}));
  anims.push_back(key("l_hand", "rotation/z", {{0, 0}, {0.5, 10}, {1, -5}}));
  const fv::dson::Json     file = {{"asset_info", {{"type", "preset_pose"}}}, {"scene", {{"animations", anims}}}};
  const fv::pose::PoseClip own  = fv::pose::loadPoseClip(g9.pose.get(), nullptr, file);
  const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *g9.pose, *g8.pose);
  for(const float t : {0.5f, 1.f})
  {
    g9.apply(own, t);
    g8.apply(clip, t);
    EXPECT_GE(g8.rotation("lForearmBend", 1), -30 - 1e-9) << t;
    EXPECT_LT(degreesBetween(g9.turned("l_hand", rest.tip("lHand")), g8.tip("lHand")), 0.5) << t;
  }
}

// A face the file holds at rest stays the figure's own, though the two rigs place a jaw's and a
// pectoral's ends 20° apart (Mocap Online's walks key all of Genesis 8's face at 0); a jaw it opens
// opens as far, and the head carrying them turns with the walk
TEST(Retarget, aFaceTheFileHoldsAtRestStaysTheFiguresOwn)
{
  for(const double open : {0.0, 12.0})
  {
    Figure                   g8(true, 48, 8, 105), g9(false, 44, 2, 97);
    const fv::dson::Json     file = walkFile(true, open);
    const fv::pose::PoseClip own  = fv::pose::loadPoseClip(g8.pose.get(), nullptr, file);
    const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *g8.pose, *g9.pose);
    for(const float t : {0.f, 0.5f, 1.f})
    {
      g8.apply(own, t);
      g9.apply(clip, t);
      for(int a = 0; a < 3; a++)
      {
        EXPECT_NEAR(g9.rotation("lowerjaw", a), a == 0 ? open : 0, 1e-3) << "jaw open " << open << " at " << t;
        EXPECT_NEAR(g9.rotation("l_pectoral", a), 0, 1e-3) << t;
      }
      EXPECT_LT(degreesBetween(g8.turned("head", {0, 1, 0}), g9.turned("head", {0, 1, 0})), 0.5) << t;
    }
  }
}

// A figure standing in a T (as Genesis 2 does) to one in an A: the arms are re-aimed onto the source's,
// and the forearms and hands hang from them as there, so the arm points where the file has it; the
// legs, a build apart rather than a pose, keep the figure's own stance
TEST(Retarget, aTPosesArmsAreReAimedOntoAnAPose)
{
  Figure                   tpose(true, 0, 8, 105), g9(false, 44, 2, 97);
  const fv::dson::Json     file = walkFile(true);
  const fv::pose::PoseClip own  = fv::pose::loadPoseClip(tpose.pose.get(), nullptr, file);
  const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *tpose.pose, *g9.pose);
  for(const float t : {0.f, 0.5f, 1.f})
  {
    tpose.apply(own, t);
    g9.apply(clip, t);
    EXPECT_LT(degreesBetween(tpose.way("lShldrBend", "lForearmBend"), g9.way("l_upperarm", "l_forearm")), 0.5) << t;
    EXPECT_LT(degreesBetween(tpose.way("lForearmBend", "lHand"), g9.way("l_forearm", "l_hand")), 0.5) << t;
    EXPECT_LT(degreesBetween(tpose.tip("lHand"), g9.tip("l_hand")), 0.5) << t;
  }
  tpose.apply(own, 0);
  g9.apply(clip, 0);
  EXPECT_GT(degreesBetween(tpose.way("lThighBend", "lShin"), g9.way("l_thigh", "l_shin")), 5);
}

// A still pose (one key a channel, at 0) made for Genesis 8 poses a Genesis 9 as its own: the host
// retargets it as it does an animation, whose bones aren't the figure's either
TEST(Retarget, aStillPoseForAnotherGenesisIsRetargetedToo)
{
  Figure               g8(true, 48, 8, 105), g9(false, 44, 2, 97);
  const Figure         rest(false, 44, 2, 97);
  const fv::dson::Json file = [] {
    fv::dson::Json anims = fv::dson::Json::array();
    for(const auto& [bone, channel, value] : {std::tuple{"lThighBend", "rotation/x", -30.0}, std::tuple{"lShin", "rotation/x", 60.0},
                                              std::tuple{"lShldrBend", "rotation/z", -25.0}, std::tuple{"head", "rotation/y", 20.0}})
      anims.push_back({{"url", std::string("name://@selection/") + bone + ":?" + channel + "/value"}, {"keys", {{0, value}}}});
    return fv::dson::Json{{"asset_info", {{"type", "preset_pose"}}}, {"scene", {{"animations", anims}}}};
  }();
  EXPECT_FALSE(fv::pose::isAnimated(file));
  EXPECT_FALSE(fv::pose::namesRigBones(file, *g9.pose));
  const fv::pose::PoseClip own  = fv::pose::loadPoseClip(g8.pose.get(), nullptr, file);
  const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *g8.pose, *g9.pose);
  ASSERT_FALSE(clip.channels.empty());
  EXPECT_FLOAT_EQ(clip.duration, 0.f);
  g8.apply(own, 0);
  g9.apply(clip, 0);
  EXPECT_LT(degreesBetween(g8.turned("lThighTwist", rest.way("l_thigh", "l_shin")), g9.way("l_thigh", "l_shin")), 0.5);
  EXPECT_LT(degreesBetween(g8.turned("lShin", rest.way("l_shin", "l_foot")), g9.way("l_shin", "l_foot")), 0.5);
  EXPECT_LT(degreesBetween(g8.turned("lShldrTwist", rest.way("l_upperarm", "l_forearm")), g9.way("l_upperarm", "l_forearm")), 0.5);
  EXPECT_LT(degreesBetween(g8.turned("head", rest.tip("head")), g9.tip("head")), 0.5);
  // and it isn't the rest pose
  EXPECT_GT(degreesBetween(rest.way("l_thigh", "l_shin"), g9.way("l_thigh", "l_shin")), 20);
}

// Which figure a file was made for: by its bones' names, then by its path
TEST(Retarget, theSourceFigureIsFoundByItsBonesThenItsPath)
{
  EXPECT_EQ(fv::pose::sourceFigureFile(walkFile(false), "x.duf"), "data/DAZ 3D/Genesis 9/Base/Genesis9.dsf");
  EXPECT_EQ(fv::pose::sourceFigureFile(walkFile(true), "People/Genesis 8 Female/Poses/Walk.duf"), "data/DAZ 3D/Genesis 8/Female/Genesis8Female.dsf");
  EXPECT_EQ(fv::pose::sourceFigureFile(walkFile(true), "People/Genesis 8 Male/Poses/Mocap Online/WAC Walk 01 Relaxed Loop IP Michael 8.duf"),
            "data/DAZ 3D/Genesis 8/Male/Genesis8Male.dsf");
  EXPECT_EQ(fv::pose::sourceFigureFile(walkFile(true), "People/Genesis 3 Female/Poses/Walk.duf"), "data/DAZ 3D/Genesis 3/Female/Genesis3Female.dsf");
  EXPECT_EQ(fv::pose::sourceFigureFile({{"scene", {{"animations", fv::dson::Json::array()}}}}, "x.duf"), "");
  EXPECT_EQ(fv::pose::sourceFigureName("/data/DAZ 3D/Genesis 8/Female 8_1/Genesis8_1Female.dsf"), "Genesis 8.1 Female");
  EXPECT_EQ(fv::pose::sourceFigureName("/data/DAZ 3D/Genesis 9/Base/Genesis9.dsf"), "Genesis 9");

  Figure g9(false, 44, 2, 97);
  EXPECT_TRUE(fv::pose::namesRigBones(walkFile(false), *g9.pose));
  EXPECT_FALSE(fv::pose::namesRigBones(walkFile(true), *g9.pose));
}

namespace {

// A figure and a worn item's skeleton in one scene, the item's bones at the figure's joints (as a fit
// leaves them), each following the figure bone `follow` names.
struct Worn
{
  fv::SceneGraph                     scene;
  fv::dson::Rig                      figure, item;
  std::unique_ptr<fv::pose::PoseRig> pose;
  std::map<std::string, std::string> follow;

  Worn(const std::string& figureBones, const std::string& itemBones, std::map<std::string, std::string> plan)
      : follow(std::move(plan))
  {
    auto f = fv::dson::buildRig(scene, fv::dson::Json::parse(figureBones), nullptr);
    auto i = fv::dson::buildRig(scene, fv::dson::Json::parse(itemBones), nullptr);
    EXPECT_TRUE(f && i);
    figure = *f;
    item   = *i;
    pose   = std::make_unique<fv::pose::PoseRig>(scene, figure, std::vector<const fv::dson::Rig*>{});
    fv::pose::PoseRig::FollowPlan p;
    p.follow = &follow;
    pose->addFollowers(item, p);
  }
  fv::Mat4 world(const fv::dson::Rig& rig, const std::string& name) const { return scene.world(rig.find(scene, name)); }
};

std::string rigOf(const std::string& root, const std::vector<std::string>& bones)
{
  std::string out = R"([{"id":")" + root + R"(","type":"figure","name":")" + root + R"(",)" + triple("center_point", {}) + "}";
  for(const std::string& b : bones)
    out += "," + b;
  return out + "]";
}

void expectSameWorld(const fv::Mat4& a, const fv::Mat4& b, double within, const std::string& what)
{
  for(size_t k = 0; k < 16; k++)
    EXPECT_NEAR(a.e[k], b.e[k], within) << what << " element " << k;
}

}  // namespace

// A Genesis 9 sleeve on Genesis 8: its forearm follows lForearmBend under an upper arm that follows
// lShldrBend, past lShldrTwist, and its hand lHand past lForearmTwist. Copying each leader's own turn
// lost the twists (about 12 cm at an ankle for a thigh's); the item now takes the joints between, and
// its hand ends where the body's is within a micrometre
TEST(FollowChain, AnItemsBonesTakeTheFiguresJointsTheyHaveNone)
{
  const std::string g8 = rigOf("Genesis8Female", {bone("lShldrBend", "Genesis8Female", {16, 138, 0}, {30, 138, 0}, "yz", "XYZ"),
                                                  bone("lShldrTwist", "lShldrBend", {30, 138, 0}, {44, 138, 0}, "x", "XYZ"),
                                                  bone("lForearmBend", "lShldrTwist", {44, 138, 0}, {57, 138, 0}, "y", "XZY"),
                                                  bone("lForearmTwist", "lForearmBend", {57, 138, 0}, {70, 138, 0}, "x", "XZY"),
                                                  bone("lHand", "lForearmTwist", {70, 138, 0}, {80, 138, 0}, "xyz", "XYZ")});
  const std::string g9 = rigOf("Sleeve", {bone("l_upperarm", "Sleeve", {16, 138, 0}, {44, 138, 0}, "xyz", "XYZ"),
                                          bone("l_upperarmtwist1", "l_upperarm", {16, 138, 0}, {30, 138, 0}, "x", "XYZ"),
                                          bone("l_forearm", "l_upperarm", {44, 138, 0}, {70, 138, 0}, "xy", "XZY"),
                                          bone("l_hand", "l_forearm", {70, 138, 0}, {80, 138, 0}, "xyz", "XYZ")});
  Worn w(g8, g9, {{"l_upperarm", "lShldrBend"}, {"l_upperarmtwist1", "lShldrTwist"}, {"l_forearm", "lForearmBend"}, {"l_hand", "lHand"}});
  w.pose->setMany({{"lShldrTwist", ChannelKind::eRotation, "x", 45}, {"lForearmBend", ChannelKind::eRotation, "y", 60},
                   {"lForearmTwist", ChannelKind::eRotation, "x", -30}, {"lShldrBend", ChannelKind::eRotation, "z", -20}});
  expectSameWorld(w.world(w.item, "l_forearm"), w.world(w.figure, "lForearmBend"), 1e-6, "forearm");
  expectSameWorld(w.world(w.item, "l_hand"), w.world(w.figure, "lHand"), 1e-6, "hand");
  // the twist helper turns as the figure's twist bone does (about its own joint, on the same axis)
  const fv::Mat4 helper = w.world(w.item, "l_upperarmtwist1"), twist = w.world(w.figure, "lShldrTwist");
  for(const size_t k : {0, 1, 2, 4, 5, 6, 8, 9, 10})
    EXPECT_NEAR(helper.e[k], twist.e[k], 1e-6) << "twist helper's turn, element " << k;
  // at rest again, where it was
  w.pose->setMany({{"lShldrTwist", ChannelKind::eRotation, "x", 0}, {"lForearmBend", ChannelKind::eRotation, "y", 0},
                   {"lForearmTwist", ChannelKind::eRotation, "x", 0}, {"lShldrBend", ChannelKind::eRotation, "z", 0}});
  EXPECT_NEAR(w.world(w.item, "l_hand").e[12], 70, 1e-9);
}

// A Skyrim bodice has no bone for Genesis 8's chestUpper: its neck, under a chest that follows
// chestLower, turns and moves with chestUpper's bend, its pivot where the body's neck is
TEST(FollowChain, ASkippedJointMovesTheNextPivot)
{
  const std::string g8 = rigOf("Genesis8Female", {bone("chestLower", "Genesis8Female", {0, 120, 0}, {0, 130, 0}, "xyz", "YZX"),
                                                  bone("chestUpper", "chestLower", {0, 130, 0}, {0, 145, 0}, "xyz", "YZX"),
                                                  bone("neckLower", "chestUpper", {0, 145, 0}, {0, 150, 0}, "xyz", "YZX")});
  const std::string sky = rigOf("Bodice", {bone("NPC Spine2 [Spn2]", "Bodice", {0, 120, 0}, {0, 145, 0}, "xyz", "YZX"),
                                           bone("NPC Neck [Neck]", "NPC Spine2 [Spn2]", {0, 145, 0}, {0, 150, 0}, "xyz", "YZX")});
  Worn w(g8, sky, {{"NPC Spine2 [Spn2]", "chestLower"}, {"NPC Neck [Neck]", "neckLower"}});
  w.pose->setMany({{"chestUpper", ChannelKind::eRotation, "x", 10}, {"chestLower", ChannelKind::eRotation, "z", 5}});
  expectSameWorld(w.world(w.item, "NPC Neck [Neck]"), w.world(w.figure, "neckLower"), 1e-6, "neck");
  // the neck's pivot came forward with the bend, as the body's did
  EXPECT_GT(std::abs(w.world(w.item, "NPC Neck [Neck]").e[14]), 1);
}

// A joint the item skips that is scaled scales what hangs below it on the item too
TEST(FollowChain, ASkippedJointsScaleCarriesOn)
{
  const std::string g8  = rigOf("Genesis8Female", {bone("chestLower", "Genesis8Female", {0, 120, 0}, {0, 130, 0}, "xyz", "YZX"),
                                                   bone("chestUpper", "chestLower", {0, 130, 0}, {0, 145, 0}, "xyz", "YZX"),
                                                   bone("neckLower", "chestUpper", {0, 145, 0}, {0, 150, 0}, "xyz", "YZX")});
  std::string       big = g8;
  // chestUpper with a scale channel
  const std::string withScale = R"("rotation_order":"YZX",)";
  big.replace(big.find(withScale, big.find("\"chestUpper\"")), withScale.size(),
              withScale + R"("scale":[{"id":"x","value":1,"min":0,"max":10},{"id":"y","value":1,"min":0,"max":10},{"id":"z","value":1,"min":0,"max":10}],)");
  const std::string sky = rigOf("Bodice", {bone("chest", "Bodice", {0, 120, 0}, {0, 145, 0}, "xyz", "YZX"), bone("neck", "chest", {0, 145, 0}, {0, 150, 0}, "xyz", "YZX")});
  Worn w(big, sky, {{"chest", "chestLower"}, {"neck", "neckLower"}});
  w.pose->setMany({{"chestUpper", ChannelKind::eScale, "y", 1.5}});
  expectSameWorld(w.world(w.item, "neck"), w.world(w.figure, "neckLower"), 1e-6, "neck");
  EXPECT_NEAR(w.world(w.item, "neck").e[13], 130 + 15 * 1.5, 1e-6);
}
