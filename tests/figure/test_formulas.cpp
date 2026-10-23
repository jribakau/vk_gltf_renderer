// A formula's stack: push / mult / add / sub / div, and the two splines. The graph that orders them
// is test_erc.cpp. Ports the operation half of src/lib/pose/formulas.ts (exercised through erc.test.ts).

#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <vector>

#include <figure/pose/formulas.hpp>

using fv::dson::Json;

namespace {

Json opsOf(const char* text)
{
  return Json::parse(text);
}

std::optional<double> run(const char* text, const std::vector<double>& inputs = {})
{
  int n = 0;
  const std::vector<fv::pose::Op> ops = fv::pose::compileOps(opsOf(text), [&](const std::string&) { return n < int(inputs.size()) ? n++ : -2; });
  return fv::pose::evaluate(ops, [&](int i) -> std::optional<double> {
    return i >= 0 && i < int(inputs.size()) ? std::optional<double>(inputs[i]) : std::nullopt;
  });
}

}  // namespace

TEST(FigureFormulas, ArithmeticIsAStack)
{
  // (3 + 4) * 2 - 1, then divided by 2: push order is left, then right, then the op
  EXPECT_DOUBLE_EQ(run(R"([{"op":"push","val":3},{"op":"push","val":4},{"op":"add"},{"op":"push","val":2},{"op":"mult"},{"op":"push","val":1},{"op":"sub"},{"op":"push","val":2},{"op":"div"}])").value_or(-1), 6.5);
}

TEST(FigureFormulas, DivisionByZeroIsZero)
{
  EXPECT_DOUBLE_EQ(run(R"([{"op":"push","val":5},{"op":"push","val":0},{"op":"div"}])").value_or(-1), 0);
}

TEST(FigureFormulas, AnUnknownOperationEvaluatesToNothing)
{
  EXPECT_FALSE(run(R"([{"op":"push","val":1},{"op":"nope"}])").has_value());
  EXPECT_FALSE(run(R"([{"op":"push","url":"#x?value"}])").has_value());  // the url could not be read
}

TEST(FigureFormulas, APropertyPushReadsTheInput)
{
  EXPECT_DOUBLE_EQ(run(R"([{"op":"push","url":"#A?value"},{"op":"push","val":2},{"op":"mult"}])", {0.3}).value_or(-1), 0.6);
}

TEST(FigureFormulas, LinearSplineHitsItsKnotsAndTheMidpoints)
{
  const char* text = R"([
    {"op":"push","val":0},
    {"op":"push","val":[0,0,0,0,0]},
    {"op":"push","val":[0.5,0.1,0,0,0]},
    {"op":"push","val":[1,1,0,0,0]},
    {"op":"push","val":3},
    {"op":"spline_linear"}
  ])";
  auto        at   = [&](double x) {
    auto doc = opsOf(text);
    doc[0]["val"] = x;
    int n = 0;
    const auto ops = fv::pose::compileOps(doc, [&](const std::string&) { return n++; });
    return fv::pose::evaluate(ops, [](int) -> std::optional<double> { return std::nullopt; });
  };
  EXPECT_NEAR(at(0).value_or(-1), 0, 1e-9);
  EXPECT_NEAR(at(0.25).value_or(-1), 0.05, 1e-9);
  EXPECT_NEAR(at(0.5).value_or(-1), 0.1, 1e-9);
  EXPECT_NEAR(at(0.75).value_or(-1), 0.55, 1e-9);
  EXPECT_NEAR(at(1).value_or(-1), 1, 1e-9);
  // outside the knots it holds the end value
  EXPECT_NEAR(at(-1).value_or(-1), 0, 1e-9);
  EXPECT_NEAR(at(2).value_or(-1), 1, 1e-9);
}

TEST(FigureFormulas, TcbSplineAgreesAtTheKnots)
{
  // zero tension, continuity and bias: the curve passes through its knots, like the linear one
  const std::vector<std::vector<double>> knots{{0, 0, 0, 0, 0}, {0.5, 0.1, 0, 0, 0}, {1, 1, 0, 0, 0}};
  std::vector<const std::vector<double>*> ptrs;
  for(const auto& k : knots)
    ptrs.push_back(&k);
  EXPECT_NEAR(fv::pose::splineTcb(0, ptrs, false), 0, 1e-9);
  EXPECT_NEAR(fv::pose::splineTcb(0.5, ptrs, false), 0.1, 1e-9);
  EXPECT_NEAR(fv::pose::splineTcb(1, ptrs, false), 1, 1e-9);
  // and it leaves the straight line between them (a linear fold of the same knots cannot)
  const double mid = fv::pose::splineTcb(0.25, ptrs, false);
  EXPECT_GT(std::abs(mid - 0.05), 1e-4);
}
