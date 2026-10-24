// Preset files are recognised before they are loaded as models, and a pose preset's last keys
// drive bones and morph dials. A port of describePreset (src/lib/loaders/dson/presets.ts) and
// PoseRig.applyPreset.

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <figure/dson/presets.hpp>
#include <figure/dson/rig.hpp>
#include <figure/figure.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_rig.hpp>

namespace {

const char* kNodes = R"([
  {"id": "hip", "type": "figure", "name": "hip", "center_point": [{"id": "y", "value": 0}],
   "rotation": [{"id": "x", "min": -180, "max": 180}, {"id": "y", "min": -180, "max": 180}, {"id": "z", "min": -180, "max": 180}]},
  {"id": "head", "type": "bone", "name": "head", "parent": "#hip", "center_point": [{"id": "y", "value": 15}],
   "rotation": [{"id": "x", "min": -180, "max": 180}, {"id": "y", "min": -180, "max": 180}, {"id": "z", "min": -180, "max": 180}]}
])";

struct Rig
{
  fv::SceneGraph                     scene;
  std::unique_ptr<fv::pose::PoseRig> pose;
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

}  // namespace

TEST(Presets, namesAPoseAndLeavesAModelAlone)
{
  auto pose = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"preset_pose"},"scene":{"nodes":[]}})"));
  ASSERT_TRUE(pose);
  EXPECT_EQ(pose->kind, "pose preset");

  auto model = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"figure"},"geometry_library":[{}],"scene":{"nodes":[]}})"));
  EXPECT_FALSE(model.has_value());
}

TEST(Presets, readsAnEnvironmentAndGuessesCamerasAndLights)
{
  auto env = fv::dson::describePreset(fv::dson::Json::parse(R"({
    "asset_info": {"type": "preset_render_settings"},
    "scene": {"extra": [{"type": "studio_render_settings", "render_options": {"render_elements": [{
      "channels": [
        {"channel": {"id": "Environment Map", "image_file": "/Runtime/a.hdr"}},
        {"channel": {"id": "Environment Intensity", "current_value": 1.5}},
        {"channel": {"id": "Dome Rotation", "value": 30}},
        {"channel": {"id": "Draw Dome", "current_value": true}}
      ]
    }]}}]}
  })"));
  ASSERT_TRUE(env);
  EXPECT_EQ(env->kind, "render settings preset");
  ASSERT_TRUE(env->environment);
  EXPECT_EQ(env->environment->image, "/Runtime/a.hdr");
  EXPECT_EQ(env->environment->intensity, 1.5);
  EXPECT_EQ(env->environment->rotation, 30);
  EXPECT_EQ(env->environment->drawDome, true);

  auto cameras = fv::dson::describePreset(fv::dson::Json::parse(R"({"scene":{"nodes":[{"id":"Camera 1","url":"/data/camera"}]}})"));
  ASSERT_TRUE(cameras);
  EXPECT_EQ(cameras->kind, "camera preset");

  auto lights = fv::dson::describePreset(fv::dson::Json::parse(R"({"scene":{"nodes":[{"extra":[{"type":"studio/light"}]}]}})"));
  ASSERT_TRUE(lights);
  EXPECT_EQ(lights->kind, "light preset");

  auto unknown = fv::dson::describePreset(fv::dson::Json::parse(R"({"scene":{"nodes":[]}})"));
  ASSERT_TRUE(unknown);
  EXPECT_EQ(unknown->kind, "preset with no geometry");
}

TEST(Presets, appliesTheThreeUrlFormsTheLastKeyAndSkipsTheRest)
{
  Rig  g = rig();
  auto doc = fv::dson::Json::parse(R"({
    "scene": {"animations": [
      {"url": "name://@selection/head:?rotation/x/value", "keys": [[0, 0], [1, 30]]},
      {"url": "name://@selection#Smile:?value/value", "keys": [[0, 0.4]]},
      {"url": "hip:/data/Fig.dsf#head?rotation/y/value", "keys": [[0, 1], [1, 12]]},
      {"url": "name://@selection/missing:?rotation/z/value", "keys": [[1, 5]]},
      {"url": "not a url", "keys": [[1, 1]]}
    ]}
  })");
  const fv::PosePresetResult r = fv::applyPosePreset(g.pose.get(), g.morphs.get(), doc);
  EXPECT_EQ(r.applied, 3);
  ASSERT_EQ(r.skipped.size(), 2u);
  EXPECT_EQ(r.skipped[0], "missing rotation/z");
  EXPECT_EQ(r.skipped[1], "not a url");
  EXPECT_EQ(g.pose->posed("head", "rotation", "x"), 30);
  EXPECT_EQ(g.pose->posed("head", "rotation", "y"), 12);
  EXPECT_EQ(g.morphs->get("Smile"), 0.4);

  const fv::PosePresetResult none = g.pose ? fv::applyPosePreset(nullptr, g.morphs.get(), doc) : fv::PosePresetResult{};
  EXPECT_EQ(none.applied, 0);
}

TEST(Presets, aShapeOrPropertyPresetReachesNamedNodesAndReportsVisibility)
{
  Rig g = rig();
  g.morphs->addLazy("Anus Down@w1", []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; });
  auto doc = fv::dson::Json::parse(R"({
    "scene": {"animations": [
      {"url": "name://Anatomy_1006#Anus%20Down:?value/value", "keys": [[0, 0.7]]},
      {"url": "name://Anatomy_1006#Smile:?value/value", "keys": [[0, 0.2]]},
      {"url": "name://Anatomy_1006/head:?rotation/x/value", "keys": [[0, 40]]},
      {"url": "name://@selection/Anatomy_1006:?extra/studio_node_channels/channels/Visible/value", "keys": [[0, false]]},
      {"url": "name://@selection:?extra/studio_node_channels/channels/Visible/value", "keys": [[0, true]]}
    ]}
  })");
  std::vector<std::pair<std::string, std::string>> asked;
  const fv::PosePresetResult r = fv::applyPosePreset(g.pose.get(), g.morphs.get(), doc, [&](const std::string& node, const std::string& id) {
    asked.emplace_back(node, id);
    return node == "Anatomy_1006" && id == "Anus Down" ? std::string("Anus Down@w1") : std::string();
  });
  // the item's dial; not the figure's own Smile, which the url doesn't name
  EXPECT_EQ(g.morphs->get("Anus Down@w1"), 0.7);
  EXPECT_EQ(g.morphs->get("Smile"), 0.0);
  ASSERT_EQ(r.skipped.size(), 1u);
  EXPECT_EQ(r.skipped[0], "Anatomy_1006 Smile");
  // another node's bone is left to its skeleton, which follows the figure's
  EXPECT_EQ(g.pose->posed("head", "rotation", "x").value_or(0), 0);
  EXPECT_EQ(r.applied, 1);
  // a boolean key: the named node hidden; the selection itself has no node to report
  ASSERT_EQ(r.visibility.size(), 1u);
  EXPECT_EQ(r.visibility[0], std::make_pair(std::string("Anatomy_1006"), false));
  EXPECT_EQ(asked.size(), 2u);
}
