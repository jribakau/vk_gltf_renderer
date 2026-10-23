// The ERC graph: what formula urls name, and how properties evaluate (stages, clamps, chains,
// splines, cycles). A port of test/pose/erc.test.ts.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <optional>
#include <string>

#include <figure/pose/erc.hpp>

using fv::dson::Json;
using fv::pose::ErcGraph;
using fv::pose::KeyInfo;

namespace {

const char* kFile = "data/Fig/Morphs/A.dsf";

Json docOf(const char* text)
{
  return Json::parse(text);
}

std::optional<double> lookup(const std::map<std::string, double>& dials, const KeyInfo& k)
{
  const std::string key = k.morph ? "m:" + k.id : "b:" + k.id + ":" + k.prop + "/" + k.axis;
  const auto        it  = dials.find(key);
  return it == dials.end() ? std::nullopt : std::optional<double>(it->second);
}

}  // namespace

TEST(FigureErc, KeyOfReadsModifiersAndBones)
{
  const auto heavy = fv::pose::keyOf("Genesis8Female:#FBMHeavy?value", kFile);
  ASSERT_TRUE(heavy);
  EXPECT_EQ(heavy->key, "m:FBMHeavy");
  EXPECT_FALSE(heavy->file);

  const auto value = fv::pose::keyOf("#FBMHeavy?value/value", kFile);
  ASSERT_TRUE(value);
  EXPECT_EQ(value->key, "m:FBMHeavy");
  EXPECT_FALSE(value->file);

  const auto spaced = fv::pose::keyOf("Genesis8Female:/data/Fig/Morphs/B%20C.dsf#B%20C?value", kFile);
  ASSERT_TRUE(spaced);
  EXPECT_EQ(spaced->key, "m:B C");
  ASSERT_TRUE(spaced->file);
  EXPECT_EQ(*spaced->file, "data/Fig/Morphs/B C.dsf");

  const auto bend = fv::pose::keyOf("lShldrBend:/data/Fig/Fig.dsf#lShldrBend?rotation/z", kFile);
  ASSERT_TRUE(bend);
  EXPECT_EQ(bend->key, "b:lShldrBend:rotation/z");

  const auto centre = fv::pose::keyOf("head:/data/Fig/Fig.dsf#head?center_point/y", kFile);
  ASSERT_TRUE(centre);
  EXPECT_EQ(centre->key, "b:head:center_point/y");

  const auto scale = fv::pose::keyOf("head:#head?scale/general", kFile);
  ASSERT_TRUE(scale);
  EXPECT_EQ(scale->key, "b:head:scale/general");
}

TEST(FigureErc, KeyOfIgnoresWhatItCannotRead)
{
  EXPECT_FALSE(fv::pose::keyOf("#x", kFile));
  EXPECT_FALSE(fv::pose::keyOf("#x?visible", kFile));
  EXPECT_FALSE(fv::pose::keyOf("head:#head?rotation/w", kFile));
}

TEST(FigureErc, BoneKeysSplitOnTheLastColon)
{
  ErcGraph    g;
  const auto  bone = g.info(g.intern("b:l:Shldr:rotation/z"));
  EXPECT_FALSE(bone.morph);
  EXPECT_EQ(bone.id, "l:Shldr");
  EXPECT_EQ(bone.prop, "rotation");
  EXPECT_EQ(bone.axis, "z");
  EXPECT_TRUE(g.info(g.intern("m:x")).morph);
}

TEST(FigureErc, SumThenMultThenClamp)
{
  ErcGraph g;
  g.addDoc(docOf(R"({
    "modifier_library": [
      {"id": "CTRL", "channel": {"value": 0}, "formulas": [
        {"output": "#Out?value", "operations": [{"op": "push", "url": "#CTRL?value"}, {"op": "push", "val": 2}, {"op": "mult"}]}
      ]},
      {"id": "Gate", "channel": {"value": true}},
      {"id": "Out", "channel": {"value": 0, "min": 0, "max": 1, "clamped": true}, "formulas": [
        {"output": "#Out?value", "stage": "mult", "operations": [{"op": "push", "url": "#Gate?value"}]}
      ]}
    ]
  })"),
          kFile);

  std::map<std::string, double> dials{{"m:CTRL", 0.3}, {"m:Out", 0.1}};
  auto                          eval = [&] {
    return g.evaluator([&](const KeyInfo& k) { return lookup(dials, k); });
  };
  {
    auto v = eval();
    EXPECT_NEAR(v("m:Out").value_or(-1), (0.1 + 0.6) * 1, 1e-9);
    EXPECT_DOUBLE_EQ(v("m:Gate").value_or(-1), 1);
  }

  dials["m:CTRL"] = 2;
  EXPECT_DOUBLE_EQ(eval()("m:Out").value_or(-1), 1);

  // the switch off: the mult stage zeroes it
  dials["m:Gate"] = 0;
  EXPECT_DOUBLE_EQ(eval()("m:Out").value_or(-1), 0);
}

TEST(FigureErc, ChainsAcrossFilesAndBones)
{
  ErcGraph g;
  g.addDoc(docOf(R"({
    "modifier_library": [
      {"id": "JCM", "channel": {"value": 0}, "formulas": [
        {"output": "#JCM?value", "operations": [
          {"op": "push", "url": "lShldr:/data/Fig/Fig.dsf#lShldr?rotation/z"},
          {"op": "push", "val": -0.011111111111111112},
          {"op": "mult"}
        ]}
      ]}
    ]
  })"),
          "data/Fig/Morphs/JCM.dsf");
  g.addDoc(docOf(R"({
    "modifier_library": [
      {"id": "Pose", "channel": {"value": 0}, "formulas": [
        {"output": "lShldr:/data/Fig/Fig.dsf#lShldr?rotation/z", "operations": [
          {"op": "push", "url": "#Pose?value"}, {"op": "push", "val": -45}, {"op": "mult"}
        ]}
      ]}
    ]
  })"),
          "data/Fig/Morphs/Pose.dsf");

  const std::map<std::string, double> posed{{"b:lShldr:rotation/z", 0}, {"m:Pose", 1}, {"m:JCM", 0}};
  auto                                v = g.evaluator([&](const KeyInfo& k) { return lookup(posed, k); });
  EXPECT_DOUBLE_EQ(v("b:lShldr:rotation/z").value_or(1), -45);
  EXPECT_NEAR(v("m:JCM").value_or(-1), 0.5, 1e-9);
  const std::string* file = g.fileOf("Pose");
  ASSERT_NE(file, nullptr);
  EXPECT_EQ(*file, "data/Fig/Morphs/Pose.dsf");
}

TEST(FigureErc, SplinesWhichALinearFoldCannot)
{
  ErcGraph g;
  g.addDoc(docOf(R"({
    "modifier_library": [
      {"id": "C", "channel": {"value": 0}, "formulas": [
        {"output": "#T?value", "operations": [
          {"op": "push", "url": "#C?value"},
          {"op": "push", "val": [0, 0, 0, 0, 0]},
          {"op": "push", "val": [0.5, 0.1, 0, 0, 0]},
          {"op": "push", "val": [1, 1, 0, 0, 0]},
          {"op": "push", "val": 3},
          {"op": "spline_linear"}
        ]}
      ]}
    ]
  })"),
          kFile);
  const std::pair<double, double> samples[] = {{0, 0}, {0.25, 0.05}, {0.5, 0.1}, {0.75, 0.55}, {1, 1}};
  for(const auto& [x, y] : samples)
  {
    auto v = g.evaluator([&](const KeyInfo& k) -> std::optional<double> {
      if(k.morph && k.id == "C")
        return x;
      if(k.morph && k.id == "T")
        return 0;
      return std::nullopt;
    });
    EXPECT_NEAR(v("m:T").value_or(-1), y, 1e-9);
  }
}

TEST(FigureErc, ACycleReadsTheBaseAndUnknownInputsAreSkipped)
{
  ErcGraph g;
  g.addDoc(docOf(R"({
    "modifier_library": [
      {"id": "A", "channel": {"value": 0}, "formulas": [
        {"output": "#B?value", "operations": [{"op": "push", "url": "#A?value"}]}
      ]},
      {"id": "B", "channel": {"value": 0}, "formulas": [
        {"output": "#A?value", "operations": [{"op": "push", "url": "#B?value"}]},
        {"output": "#B?value", "operations": [{"op": "push", "url": "#Missing?value"}]}
      ]}
    ]
  })"),
          kFile);
  const std::map<std::string, double> dials{{"m:A", 1}, {"m:B", 0}};
  auto                                v = g.evaluator([&](const KeyInfo& k) { return lookup(dials, k); });
  // A = 1 + B; B = 0 + A, where A reads as its base inside the cycle: 1
  EXPECT_DOUBLE_EQ(v("m:A").value_or(-1), 2);
  EXPECT_FALSE(v("m:Missing").has_value());
}

TEST(FigureErc, AFileRegistersOnce)
{
  ErcGraph   g;
  const Json doc = docOf(R"({
    "modifier_library": [
      {"id": "A", "channel": {"value": 0}, "formulas": [
        {"output": "#B?value", "operations": [{"op": "push", "url": "#A?value"}]}
      ]}
    ]
  })");
  g.addDoc(doc, kFile);
  g.addDoc(doc, "DATA/FIG/MORPHS/A.DSF");
  const std::map<std::string, double> dials{{"m:A", 1}, {"m:B", 0}};
  auto                                v = g.evaluator([&](const KeyInfo& k) { return lookup(dials, k); });
  EXPECT_DOUBLE_EQ(v("m:B").value_or(-1), 1);
}
