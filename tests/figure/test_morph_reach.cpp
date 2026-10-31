// The reach audit: what a dial can do, from its file's deltas and every file's formulas, run as the
// driver runs them at the two ends of its range.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <figure/dson/morphs.hpp>
#include <figure/pose/morph_reach.hpp>

using fv::morph::Reach;
using fv::pose::MorphFact;

namespace {

// A morph file by its parts, read as the figure reads one. `size` above 0 says it has deltas that far.
MorphFact fact(const std::string& id, double size = 0, const std::string& channel = R"("value": 0, "min": 0, "max": 1)", const std::string& formulas = "")
{
  std::string text = R"({"modifier_library": [{"id": ")" + id + R"(", "channel": {)" + channel + "}";
  if(!formulas.empty())
    text += R"(, "formulas": )" + formulas;
  if(size > 0)
    text += R"(, "morph": {"deltas": {"count": 1, "values": [[0, )" + std::to_string(size) + ", 0, 0]]}}";
  text += "}]}";
  MorphFact f;
  f.file = "data/Fig/Morphs/" + id + ".dsf";
  f.stem = id;
  f.info = *fv::dson::morphInfoFromText(text);
  f.size = size;
  return f;
}

std::string drive(const std::string& output, const std::string& dial, double k)
{
  return R"({"output": ")" + output + R"(", "operations": [{"op": "push", "url": "Fig:#)" + dial + R"(?value"}, {"op": "push", "val": )" + std::to_string(k)
         + R"(}, {"op": "mult"}]})";
}

const Reach& of(const fv::pose::DialReach& reach, const std::string& name)
{
  static const Reach none;
  const auto         it = reach.find(name);
  EXPECT_NE(it, reach.end()) << name;
  return it == reach.end() ? none : it->second;
}

}  // namespace

TEST(MorphSize, IsItsLongestDelta)
{
  EXPECT_DOUBLE_EQ(fv::dson::morphSizeFromText(R"({"modifier_library": [{"id": "m", "channel": {"value": 0},
    "morph": {"vertex_count": 12, "deltas": {"count": 3, "values": [[0, 1, 1, 1], [7000, 3, -4, 0], [2, 0.5, 0, 0]]}}}]})"),
                   5);
  // what's in a vertex number isn't a move
  EXPECT_DOUBLE_EQ(fv::dson::morphSizeFromText(R"({"modifier_library": [{"morph": {"deltas": {"count": 1, "values": [[9000, 0, 0, 0]]}}}]})"), 0);
}

TEST(MorphSize, NothingElseInTheFileCounts)
{
  EXPECT_DOUBLE_EQ(fv::dson::morphSizeFromText(R"({"modifier_library": [{"id": "m", "channel": {"value": 0, "values": [[0, 90, 90, 90]]}}]})"), 0);
  EXPECT_DOUBLE_EQ(fv::dson::morphSizeFromText(R"({"modifier_library": [{"morph": {"deltas": {"count": 0, "values": []}}}]})"), 0);
  EXPECT_DOUBLE_EQ(fv::dson::morphSizeFromText(R"({"modifier_library": [{"morph": {"deltas": {"count": 1, "values": [[0, 2, 0, 0]]}, "values": [[0, 90, 0, 0]]}}],
    "scene": {"values": [[0, 90, 0, 0]]}})"),
                   2);
  // cut short: as far as it reached
  EXPECT_DOUBLE_EQ(fv::dson::morphSizeFromText(R"({"modifier_library": [{"morph": {"deltas": {"count": 2, "values": [[0, 2, 0, 0], [1, 9)"), 2);
  EXPECT_DOUBLE_EQ(fv::dson::morphSizeFromText("not json"), 0);
}

TEST(MorphInfo, SaysWhatItsChannelHoldsItTo)
{
  const auto info = fv::dson::morphInfoFromText(R"({"modifier_library": [{"id": "m", "channel": {"value": 0, "min": -10, "max": 10, "clamped": true}}]})");
  ASSERT_TRUE(info);
  EXPECT_TRUE(info->clamped);
  EXPECT_DOUBLE_EQ(info->lo, -10);
  EXPECT_DOUBLE_EQ(info->hi, 10);
  EXPECT_DOUBLE_EQ(*info->min, -2);
  EXPECT_FALSE(fv::dson::morphInfoFromText(R"({"modifier_library": [{"id": "m", "channel": {"value": 0, "min": 0, "max": 1, "clamped": false}}]})")->clamped);
  EXPECT_FALSE(fv::dson::morphInfoFromText(R"({"modifier_library": [{"id": "m", "channel": {"value": 0, "clamped": true}}]})")->clamped);
}

TEST(MorphReach, AMorphReachesItsOwnSize)
{
  const auto  reach = fv::pose::auditReach({fact("Lift", 3)});
  const Reach r     = of(reach, "Lift");
  EXPECT_DOUBLE_EQ(r.cm, 3);
  EXPECT_EQ(r.morphs, 1);
  EXPECT_EQ(r.bones, 0);
  EXPECT_EQ(r.scales, 0);
  EXPECT_DOUBLE_EQ(r.deg, 0);
}

TEST(MorphReach, TheFurtherEndOfTheRangeCounts)
{
  EXPECT_DOUBLE_EQ(of(fv::pose::auditReach({fact("Both", 3, R"("value": 0, "min": -2, "max": 1)")}), "Both").cm, 6);
  // from where it rests: 1 down to 0, not 0 up to 1
  EXPECT_DOUBLE_EQ(of(fv::pose::auditReach({fact("On", 3, R"("value": 1, "min": 0, "max": 1)")}), "On").cm, 3);
  EXPECT_DOUBLE_EQ(of(fv::pose::auditReach({fact("Still", 3, R"("value": 1, "min": 1, "max": 1)")}), "Still").cm, 0);
}

TEST(MorphReach, ACoefficientScalesWhatADialDrives)
{
  const auto  reach = fv::pose::auditReach({fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Shape?value", "Ctrl", 0.25) + "]"), fact("Shape", 8)});
  const Reach r     = of(reach, "Ctrl");
  EXPECT_DOUBLE_EQ(r.cm, 2);
  EXPECT_EQ(r.morphs, 1);
  EXPECT_DOUBLE_EQ(of(reach, "Shape").cm, 8);
}

TEST(MorphReach, AChainIsFollowed)
{
  const auto  reach = fv::pose::auditReach({fact("Top", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Mid?value", "Top", 0.5) + "]"),
                                            fact("Mid", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Shape?value", "Mid", 0.5) + "]"), fact("Shape", 8)});
  const Reach r     = of(reach, "Top");
  EXPECT_DOUBLE_EQ(r.cm, 2);
  // the controller between them has no deltas: it isn't one of the morphs moved
  EXPECT_EQ(r.morphs, 1);
}

// The formula may live in the morph's own file, which says nothing on the dial's side.
TEST(MorphReach, AFormulaInTheMorphsOwnFileCounts)
{
  const auto reach = fv::pose::auditReach({fact("Dial"), fact("Shape", 8, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Shape?value", "Dial", 0.5) + "]")});
  EXPECT_DOUBLE_EQ(of(reach, "Dial").cm, 4);
}

TEST(MorphReach, ASplineGivesItsRealOutput)
{
  const std::string spline = R"([{"output": "Fig:#Shape?value", "operations": [
    {"op": "push", "url": "Fig:#Ctrl?value"},
    {"op": "push", "val": [0, 0, 0, 0, 0]},
    {"op": "push", "val": [0.5, 1, 0, 0, 0]},
    {"op": "push", "val": [1, 0.25, 0, 0, 0]},
    {"op": "push", "val": 3},
    {"op": "spline_linear"}]}])";
  const auto        reach  = fv::pose::auditReach({fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", spline), fact("Shape", 8)});
  // at the dial's end: that's far enough for no one to ask what it passes through on the way
  EXPECT_DOUBLE_EQ(of(reach, "Ctrl").cm, 2);
}

// Back at 0 by the end of its range, as Neck Swallow's splines are: it does something on the way.
TEST(MorphReach, ADialThatPeaksOnTheWayMoves)
{
  const auto peak = [](const std::string& output, const std::string& dial, const std::string& knot, const std::string& end) {
    return R"([{"output": ")" + output + R"(", "operations": [{"op": "push", "url": "Fig:#)" + dial + R"(?value"},
      {"op": "push", "val": [0, 0, 0, 0, 0]},
      {"op": "push", "val": [)" + knot + R"(, 1, 0, 0, 0]},
      {"op": "push", "val": [)" + end + R"(, 0, 0, 0, 0]},
      {"op": "push", "val": 3},
      {"op": "spline_linear"}]}])";
  };
  const auto reach = fv::pose::auditReach({fact("Skin", 0, R"("value": 0, "min": 0, "max": 1)", peak("Fig:#Shape?value", "Skin", "0.5", "1")), fact("Shape", 8)});
  EXPECT_DOUBLE_EQ(of(reach, "Skin").cm, 8);
  EXPECT_EQ(of(reach, "Skin").morphs, 1);

  // over the dial's own range, not 0 to 1, and a joint counts as skin does
  const auto turn = fv::pose::auditReach({fact("Turn", 0, R"("value": 0, "min": 0, "max": 2)", peak("head:/data/Fig/Fig.dsf#head?rotation/x", "Turn", "1.5", "2"))});
  EXPECT_DOUBLE_EQ(of(turn, "Turn").deg, 1);
}

// An alias has its dial's id and nothing else of it, and its file lists first more often than not.
TEST(MorphReach, AnAliasDoesntStandInForItsDial)
{
  const auto alias = [](const std::string& id) {
    MorphFact f = fact(id, 0, R"("type": "alias")");
    f.file      = "data/Fig/Morphs/alias_head_" + id + ".dsf";
    f.stem      = "alias_head_" + id;
    return f;
  };
  const MorphFact ctrl  = fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Shape?value", "Ctrl", 10) + "]");
  const MorphFact shape = fact("Shape", 8, R"("value": 0, "min": 0, "max": 0.5, "clamped": true)");

  // its dial's formulas and limits both count, whichever file comes first
  const auto first = fv::pose::auditReach({alias("Ctrl"), ctrl, alias("Shape"), shape});
  EXPECT_DOUBLE_EQ(of(first, "Ctrl").cm, 4);
  EXPECT_EQ(of(first, "Ctrl").morphs, 1);
  EXPECT_FALSE(first.count("alias_head_Ctrl"));
  const auto last = fv::pose::auditReach({ctrl, alias("Ctrl"), shape, alias("Shape")});
  EXPECT_DOUBLE_EQ(of(last, "Ctrl").cm, 4);

  // a dial's own rest, too: this one stands on, and goes to 0
  const auto rest = fv::pose::auditReach({alias("On"), fact("On", 3, R"("value": 1, "min": 0, "max": 1)")});
  EXPECT_DOUBLE_EQ(of(rest, "On").cm, 3);
  // one with no dial on this figure reads as unknown, as any input no file holds
  EXPECT_TRUE(fv::pose::auditReach({alias("Gone")}).empty());
}

// What a controller moves that the figure has no deltas for: a worn item's copy may hold them.
TEST(MorphReach, ADialSaysWhichEmptyDialsItMoves)
{
  const std::string formulas = "[" + drive("Fig:#Lash?value", "Ctrl", 1) + "," + drive("Fig:#Curl?value", "Ctrl", 0.5) + "," + drive("Fig:#Shape?value", "Ctrl", 1) + ","
                               + drive("Fig:#pJCMEmpty?value", "Ctrl", 1) + "," + drive("Fig:#Renamed?value", "Ctrl", 1) + ","
                               + R"({"output": "Fig:#Combo?value", "operations": [{"op": "push", "url": "Fig:#Ctrl?value"}, {"op": "push", "url": "Fig:#Other?value"}, {"op": "mult"}]})" + "]";
  MorphFact renamed = fact("Renamed");
  renamed.stem      = "Renamed File";
  const auto  reach = fv::pose::auditReach({fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", formulas), fact("Lash"), fact("Curl"), fact("Shape", 2), fact("pJCMEmpty"), renamed, fact("Combo"), fact("Other")});
  // sorted; not the morph with deltas, nor one a run of the formulas would drop for its name, nor the
  // combo, which this dial doesn't move alone, nor itself
  const std::vector<std::string> empties{"Curl", "Lash"};
  EXPECT_EQ(of(reach, "Ctrl").drives, empties);
  EXPECT_TRUE(of(reach, "Lash").drives.empty());
}

TEST(MorphReach, AClampLimitsIt)
{
  const auto reach = fv::pose::auditReach({fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Shape?value", "Ctrl", 10) + "]"),
                                           fact("Shape", 8, R"("value": 0, "min": 0, "max": 0.5, "clamped": true)")});
  EXPECT_DOUBLE_EQ(of(reach, "Ctrl").cm, 4);
  // not by a slider's limits, which are narrower than the channel's
  const auto wide = fv::pose::auditReach({fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Shape?value", "Ctrl", 5) + "]"),
                                          fact("Shape", 8, R"("value": 0, "min": 0, "max": 10, "clamped": true)")});
  EXPECT_DOUBLE_EQ(of(wide, "Ctrl").cm, 40);
}

// It reaches the morph and moves it only with the other dial: there's something there, and no number.
TEST(MorphReach, ACombosDialAloneScoresNothing)
{
  const std::string combo = R"([{"output": "Fig:#Combo?value", "operations": [
    {"op": "push", "url": "Fig:#Smile?value"}, {"op": "push", "url": "Fig:#Blink?value"}, {"op": "mult"}]}])";
  const auto        reach = fv::pose::auditReach({fact("Smile"), fact("Blink"), fact("Combo", 2, R"("value": 0, "min": 0, "max": 1)", combo)});
  const Reach       r     = of(reach, "Smile");
  EXPECT_DOUBLE_EQ(r.cm, 0);
  EXPECT_EQ(r.morphs, 1);
  EXPECT_EQ(r.scales, 0);
}

TEST(MorphReach, AMultStageOnlyScales)
{
  const std::string gate  = R"([{"output": "Fig:#Shape?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Gate?value"}]},
                                {"output": "Fig:#head?rotation/x", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Gate?value"}]}])";
  const auto        reach = fv::pose::auditReach({fact("Gate", 0, R"("value": 1, "min": 0, "max": 1)", gate), fact("Shape", 8)});
  const Reach       r     = of(reach, "Gate");
  EXPECT_EQ(r.morphs, 0);
  EXPECT_EQ(r.bones, 0);
  EXPECT_EQ(r.scales, 2);
  // what it multiplies stands at 0 at rest
  EXPECT_DOUBLE_EQ(r.cm, 0);
  EXPECT_DOUBLE_EQ(r.deg, 0);
}

// A dial reached both ways is a dial that adds; past a multiplier it's still only scaling.
TEST(MorphReach, AddingWinsOverScalingWhenADialDoesBoth)
{
  const std::string both  = "[" + drive("Fig:#Shape?value", "Dial", 1) + R"(, {"output": "Fig:#Shape?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Dial?value"}]},
                               {"output": "Fig:#Scaled?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Dial?value"}]},)"
                            + drive("Fig:#Past?value", "Scaled", 1) + "]";
  const auto        reach = fv::pose::auditReach({fact("Dial", 0, R"("value": 0, "min": 0, "max": 1)", both), fact("Shape", 2), fact("Scaled", 2), fact("Past", 2)});
  const Reach       r     = of(reach, "Dial");
  EXPECT_EQ(r.morphs, 1);
  EXPECT_EQ(r.scales, 2);
}

// A switch that rests on: its gated morph stands at what the other dial gives, and goes to 0 with it.
TEST(MorphReach, AGateThatRestsOnMovesWhatItMultiplies)
{
  const std::string gate  = R"([{"output": "Fig:#Shape?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Gate?value"}]}])";
  const auto        reach = fv::pose::auditReach({fact("Gate", 0, R"("value": 1, "min": 0, "max": 1)", gate), fact("Shape", 8, R"("value": 0.5, "min": 0, "max": 1)")});
  EXPECT_DOUBLE_EQ(of(reach, "Gate").cm, 4);
}

TEST(MorphReach, ABoneMovesByDegreesAndAJointCentreByCentimetres)
{
  const auto  reach = fv::pose::auditReach({fact("Nod", 0, R"("value": 0, "min": -1, "max": 1)",
                                                 "[" + drive("head:/data/Fig/Fig.dsf#head?rotation/x", "Nod", 30) + "," + drive("head:/data/Fig/Fig.dsf#head?rotation/y", "Nod", 10) + ","
                                                     + drive("neck:/data/Fig/Fig.dsf#neck?translation/y", "Nod", 2) + "]")});
  const Reach r     = of(reach, "Nod");
  EXPECT_DOUBLE_EQ(r.deg, 30);
  EXPECT_EQ(r.bones, 2);
  EXPECT_DOUBLE_EQ(r.cm, 0);

  const auto  shape = fv::pose::auditReach({fact("Shape", 2, R"("value": 0, "min": 0, "max": 1)",
                                                 "[" + drive("head:/data/Fig/Fig.dsf#head?center_point/x", "Shape", 3) + "," + drive("head:/data/Fig/Fig.dsf#head?center_point/y", "Shape", 4) + ","
                                                     + drive("hip:/data/Fig/Fig.dsf#hip?center_point/z", "Shape", 1) + "]")});
  const Reach s     = of(shape, "Shape");
  // the morph's move plus the furthest joint's, whose axes make one offset
  EXPECT_DOUBLE_EQ(s.cm, 7);
  EXPECT_EQ(s.bones, 0);
}

// A corrective a bone drives comes along when a dial turns that bone.
TEST(MorphReach, ACorrectiveFollowsTheBoneADialTurns)
{
  const auto  reach = fv::pose::auditReach({fact("Bend", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("arm:/data/Fig/Fig.dsf#arm?rotation/x", "Bend", 90) + "]"),
                                            fact("pJCMArm", 4, R"("value": 0, "min": 0, "max": 1)",
                                                 R"([{"output": "Fig:#pJCMArm?value", "operations": [{"op": "push", "url": "arm:/data/Fig/Fig.dsf#arm?rotation/x"}, {"op": "push", "val": 0.01}, {"op": "mult"}]}])")});
  const Reach r     = of(reach, "Bend");
  EXPECT_NEAR(r.cm, 3.6, 1e-9);
  EXPECT_DOUBLE_EQ(r.deg, 90);
  EXPECT_EQ(r.morphs, 1);
  EXPECT_EQ(r.bones, 1);
}

TEST(MorphReach, ACycleEnds)
{
  const auto reach = fv::pose::auditReach({fact("A", 2, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#B?value", "A", 1) + "]"),
                                           fact("B", 2, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#A?value", "B", 1) + "]")});
  EXPECT_EQ(of(reach, "A").morphs, 2);
}

// A file names its dial by id, a listing by the file's name, and a figure has channels of both.
TEST(MorphReach, ADialGoesByItsIdAndItsFilesName)
{
  MorphFact f = fact("Lift-0xabc", 3);
  f.stem      = "Lift";
  const auto reach = fv::pose::auditReach({f});
  EXPECT_DOUBLE_EQ(of(reach, "Lift").cm, 3);
  EXPECT_DOUBLE_EQ(of(reach, "Lift-0xabc").cm, 3);
}

TEST(MorphReach, TwoCopiesOfAFileAreOneDial)
{
  MorphFact a = fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#Shape?value", "Ctrl", 1) + "]");
  MorphFact b = a;
  b.file      = "data/Other/Morphs/Ctrl.dsf";
  b.stem      = "Ctrl Copy";
  const auto reach = fv::pose::auditReach({a, b, fact("Shape", 2)});
  EXPECT_DOUBLE_EQ(of(reach, "Ctrl").cm, 2);
  EXPECT_DOUBLE_EQ(of(reach, "Ctrl Copy").cm, 2);
}

// A dial nobody sees goes without; the morphs it stands for still count for those who drive them.
TEST(MorphReach, OnlyADialAPanelCanShowHasOne)
{
  MorphFact alias = fact("Alias", 0, R"("type": "alias")");
  MorphFact cut   = fact("Cut", 2);
  cut.info.complete = false;
  const auto reach = fv::pose::auditReach({fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", "[" + drive("Fig:#pJCMHidden?value", "Ctrl", 1) + "]"), fact("pJCMHidden", 5),
                                           fact("Helper", 2, R"("value": 0, "visible": false)"), alias, cut});
  EXPECT_DOUBLE_EQ(of(reach, "Ctrl").cm, 5);
  EXPECT_FALSE(reach.count("pJCMHidden"));
  EXPECT_FALSE(reach.count("Helper"));
  EXPECT_FALSE(reach.count("Alias"));
  EXPECT_FALSE(reach.count("Cut"));
}

// An input no file holds reads as unknown, as it does in the driver: the formula is skipped.
TEST(MorphReach, AnInputNoFileHoldsSkipsItsFormula)
{
  const std::string formula = R"([{"output": "Fig:#Shape?value", "operations": [
    {"op": "push", "url": "Fig:#Ctrl?value"}, {"op": "push", "url": "Fig:#Missing?value"}, {"op": "mult"}]}])";
  const auto        reach   = fv::pose::auditReach({fact("Ctrl", 0, R"("value": 0, "min": 0, "max": 1)", formula), fact("Shape", 2)});
  EXPECT_DOUBLE_EQ(of(reach, "Ctrl").cm, 0);
  EXPECT_EQ(of(reach, "Ctrl").morphs, 1);
}

TEST(MorphReach, NoFilesNoReach)
{
  EXPECT_TRUE(fv::pose::auditReach({}).empty());
}
