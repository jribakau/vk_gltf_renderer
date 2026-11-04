// A DAZ animation made for one Genesis played on another (figure/pose/retarget.hpp): on two
// skeletons that differ as Genesis 8 and 9 do (names, a limb split into bend and twist bones or not,
// bones one has and the other hasn't, the arms' and legs' rest angles), every limb ends up pointing
// where the file's own skeleton has it.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
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
  {
    auto built = fv::dson::buildRig(scene, fv::dson::Json::parse(skeleton(g8, armDown, legOut, hipHeight)), nullptr);
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
    const fv::Mat4  w = scene.world(pose->get(bone)->bone);
    const Vec3      d{n.joint->endPoint[0] - n.joint->center[0], n.joint->endPoint[1] - n.joint->center[1], n.joint->endPoint[2] - n.joint->center[2]};
    return fv::normalize({w.e[0] * d.x + w.e[4] * d.y + w.e[8] * d.z, w.e[1] * d.x + w.e[5] * d.y + w.e[9] * d.z, w.e[2] * d.x + w.e[6] * d.y + w.e[10] * d.z});
  }
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

// An animation over a second: the left leg swinging, the knee and foot bending, the arm lowering,
// swinging and twisting with the elbow bent, the back turning, the hip bobbing
fv::dson::Json walkFile(bool g8)
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
    anims.push_back(key("abdomenLower", "rotation/y", {{0, 0}, {0.5, 8}, {1, -6}}));
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

// Genesis 8's walk on a Genesis 9: every limb points where Genesis 8's does at every key
TEST(Retarget, aGenesis8AnimationPointsAGenesis9sLimbsAsItsOwn)
{
  Figure                   g8(true, 48, 8, 105), g9(false, 44, 2, 97);
  const fv::dson::Json     file = walkFile(true);
  const fv::pose::PoseClip own  = fv::pose::loadPoseClip(g8.pose.get(), nullptr, file);
  const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *g8.pose, *g9.pose);
  ASSERT_FALSE(clip.channels.empty());
  EXPECT_FLOAT_EQ(clip.duration, 1.f);
  for(const float t : {0.f, 0.25f, 0.5f, 0.8f, 1.f})
  {
    g8.apply(own, t);
    g9.apply(clip, t);
    const std::pair<const char*, const char*> limbs[][2] = {
        {{"lThighBend", "lShin"}, {"l_thigh", "l_shin"}},
        {{"lShin", "lFoot"}, {"l_shin", "l_foot"}},
        {{"lFoot", "lMetatarsals"}, {"l_foot", "l_toes"}},
        {{"lShldrBend", "lForearmBend"}, {"l_upperarm", "l_forearm"}},
        {{"lForearmBend", "lHand"}, {"l_forearm", "l_hand"}},  // the twist turns the elbow's bend
        {{"abdomenLower", "chestUpper"}, {"spine1", "spine2"}},
        {{"lCollar", "lShldrBend"}, {"l_shoulder", "l_upperarm"}},
    };
    for(const auto& l : limbs)
      EXPECT_LT(degreesBetween(g8.way(l[0].first, l[0].second), g9.way(l[1].first, l[1].second)), 0.5) << l[1].first << " at " << t;
    // Genesis 9 has no metatarsals: their turn reaches its toes, which the file doesn't move itself
    EXPECT_LT(degreesBetween(g8.tip("lToe"), g9.tip("l_toes")), 0.5) << "toes at " << t;
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
  const fv::dson::Json     file = walkFile(false);
  const fv::pose::PoseClip own  = fv::pose::loadPoseClip(g9.pose.get(), nullptr, file);
  const fv::pose::PoseClip clip = fv::pose::retargetClip(file, *g9.pose, *g8.pose);
  ASSERT_FALSE(clip.channels.empty());
  for(const float t : {0.25f, 0.5f, 1.f})
  {
    g9.apply(own, t);
    g8.apply(clip, t);
    EXPECT_LT(degreesBetween(g9.way("l_thigh", "l_shin"), g8.way("lThighBend", "lShin")), 0.5) << t;
    EXPECT_LT(degreesBetween(g9.way("l_shin", "l_foot"), g8.way("lShin", "lFoot")), 0.5) << t;
    EXPECT_LT(degreesBetween(g9.way("l_upperarm", "l_forearm"), g8.way("lShldrBend", "lForearmBend")), 0.5) << t;
    EXPECT_LT(degreesBetween(g9.way("l_forearm", "l_hand"), g8.way("lForearmBend", "lHand")), 0.5) << t;
  }
  EXPECT_NE(g8.pose->get("lShldrTwist")->posed(ChannelKind::eRotation, 0), 0.0) << "the twist went to the twist bone";
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
