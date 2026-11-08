// A pose file's keys over time as a clip of the figure's channels (figure/pose/pose_clip.hpp): what
// counts as animated, how the keys are sampled, and that a channel is resolved as a pose preset's is.

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <figure/dson/rig.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_clip.hpp>
#include <figure/pose/pose_rig.hpp>

namespace {

const char* kNodes = R"([
  {"id": "hip", "type": "figure", "name": "hip", "center_point": [{"id": "y", "value": 0}],
   "rotation": [{"id": "x", "min": -180, "max": 180}, {"id": "y", "min": -180, "max": 180}, {"id": "z", "min": -180, "max": 180}],
   "translation": [{"id": "x"}, {"id": "y"}, {"id": "z"}]},
  {"id": "head", "type": "bone", "name": "head", "parent": "#hip", "center_point": [{"id": "y", "value": 15}],
   "rotation": [{"id": "x", "min": -180, "max": 180}, {"id": "y", "min": -180, "max": 180}, {"id": "z", "min": -180, "max": 180}]}
])";

struct Rig
{
  fv::SceneGraph                       scene;
  std::unique_ptr<fv::pose::PoseRig>   pose;
  std::unique_ptr<fv::morph::MorphSet> morphs;
};

Rig rig()
{
  Rig out;
  auto built = fv::dson::buildRig(out.scene, fv::dson::Json::parse(kNodes), nullptr);
  EXPECT_TRUE(built);
  out.pose   = std::make_unique<fv::pose::PoseRig>(out.scene, *built, std::vector<const fv::dson::Rig*>{});
  out.morphs = std::make_unique<fv::morph::MorphSet>([](int) { return static_cast<fv::mesh::BuiltMesh*>(nullptr); });
  out.morphs->addLazy("Smile", []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; });
  return out;
}

const fv::pose::PoseClip::Channel* channelOf(const fv::pose::PoseClip& clip, const std::string& boneOrDial, const std::string& axis = "")
{
  for(const auto& c : clip.channels)
    if((c.target.kind == fv::pose::PresetTarget::Kind::eBone && c.target.bone == boneOrDial && c.target.axis == axis)
       || (c.target.kind == fv::pose::PresetTarget::Kind::eDial && c.target.name == boneOrDial))
      return &c;
  return nullptr;
}

}  // namespace

TEST(PoseClip, OnlyKeysAtTwoTimesMakeAFileAnimated)
{
  using fv::dson::Json;
  EXPECT_FALSE(fv::pose::isAnimated(Json::parse(R"({"scene":{"animations":[{"url":"name://@selection/head:?rotation/y/value","keys":[[0,30]]}]}})")));
  // two keys at one time are no motion
  EXPECT_FALSE(fv::pose::isAnimated(Json::parse(R"({"scene":{"animations":[{"url":"name://@selection/head:?rotation/y/value","keys":[[0,1],[0,2]]}]}})")));
  EXPECT_FALSE(fv::pose::isAnimated(Json::parse(R"({"scene":{}})")));
  EXPECT_TRUE(fv::pose::isAnimated(Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection/head:?rotation/x/value","keys":[[0,5]]},
    {"url":"name://@selection/head:?rotation/y/value","keys":[[0,0],[0.0333333,2]]}]}})")));
}

TEST(PoseClip, KeysInterpolateLinearlyAndHoldOutsideTheirRange)
{
  Rig  g    = rig();
  auto clip = fv::pose::loadPoseClip(g.pose.get(), g.morphs.get(), fv::dson::Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection/head:?rotation/y/value","keys":[[0.5,10],[1.5,30],[2,20]]},
    {"url":"name://@selection/head:?rotation/x/value","keys":[[0,7]]}]}})"));
  EXPECT_FLOAT_EQ(clip.duration, 2.f);
  const auto* y = channelOf(clip, "head", "y");
  ASSERT_TRUE(y);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*y, 0.f), 10);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*y, 0.5f), 10);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*y, 1.f), 20);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*y, 1.75f), 25);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*y, 9.f), 20);
  // one key is that value throughout
  const auto* x = channelOf(clip, "head", "x");
  ASSERT_TRUE(x);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*x, 0.f), 7);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*x, 1.f), 7);
}

TEST(PoseClip, AConstantKeyHoldsUntilTheNextAndKeysOutOfOrderAreSorted)
{
  Rig  g    = rig();
  auto clip = fv::pose::loadPoseClip(g.pose.get(), g.morphs.get(), fv::dson::Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection/head:?rotation/y/value","keys":[[0,0,["constant"]],[1,10],[2,30]]},
    {"url":"name://@selection/head:?rotation/z/value","keys":[[1,10],[0,0]]}]}})"));
  const auto* y = channelOf(clip, "head", "y");
  ASSERT_TRUE(y);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*y, 0.9f), 0);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*y, 1.5f), 20);
  const auto* z = channelOf(clip, "head", "z");
  ASSERT_TRUE(z);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*z, 0.5f), 5);
}

TEST(PoseClip, DialsAndTheFiguresRootAreChannelsAndWhatItLacksIsSkipped)
{
  Rig  g    = rig();
  auto clip = fv::pose::loadPoseClip(g.pose.get(), g.morphs.get(), fv::dson::Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#Smile:?value/value","keys":[[0,0],[1,1]]},
    {"url":"name://@selection:?translation/y/value","keys":[[0,0],[1,-4]]},
    {"url":"name://@selection/tail:?rotation/x/value","keys":[[0,0],[1,20]]},
    {"url":"name://@selection#Frown:?value/value","keys":[[0,0],[1,0]]},
    {"url":"name://@selection/head:?extra/studio_node_channels/channels/Visible/value","keys":[[0,true],[1,false]]},
    {"url":"not a url","keys":[[0,1]]}]}})"));
  ASSERT_EQ(clip.channels.size(), 2u);
  const auto* smile = channelOf(clip, "Smile");
  ASSERT_TRUE(smile);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*smile, 0.25f), 0.25);
  const auto* root = channelOf(clip, "hip", "y");
  ASSERT_TRUE(root);
  EXPECT_EQ(root->target.channel, fv::pose::ChannelKind::eTranslation);
  EXPECT_DOUBLE_EQ(fv::pose::sample(*root, 0.5f), -2);
  // a bone it lacks that moves is reported, a dial it lacks that stays at 0 is not
  ASSERT_EQ(clip.skipped.size(), 2u);
  EXPECT_EQ(clip.skipped[0], "tail rotation/x");
  EXPECT_EQ(clip.skipped[1], "not a url");
}

// The Pose tab's groups, from the bone's name with its side taken off: Genesis 9's toes are the leg's
// though their names have the fingers' in them, its pectorals the torso's, its eyes and ears the face's
TEST(PoseRig, BonesAreGroupedByWhatTheyAre)
{
  using fv::pose::groupOf;
  EXPECT_EQ(groupOf("Genesis9", true), "Figure");
  for(const char* toe : {"l_bigtoe1", "l_indextoe1", "l_midtoe2", "l_ringtoe1", "l_pinkytoe1", "lSmallToe2", "lToe", "lMetatarsals", "lHeel"})
    EXPECT_EQ(groupOf(toe, false), "Left leg") << toe;
  EXPECT_EQ(groupOf("r_indextoe1", false), "Right leg");
  for(const char* arm : {"l_index1", "lIndex1", "l_shoulder", "lCollar", "lShldrBend", "l_upperarm", "lCarpal1", "l_thumb1", "l_hand"})
    EXPECT_EQ(groupOf(arm, false), "Left arm") << arm;
  for(const char* torso : {"l_pectoral", "rPectoral", "hip", "pelvis", "spine4", "chestUpper", "abdomenLower"})
    EXPECT_EQ(groupOf(torso, false), "Torso") << torso;
  for(const char* face : {"l_eye", "r_ear", "lEye", "rEar", "lowerJaw", "rBrowInner", "CenterBrow", "l_eyelidupper"})
    EXPECT_EQ(groupOf(face, false), "Face") << face;
  EXPECT_EQ(groupOf("neck2", false), "Head & neck");
  EXPECT_EQ(groupOf("head", false), "Head & neck");
}
