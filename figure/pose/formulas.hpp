#pragma once

// One DSON formula's operations (ERC), run as DAZ Studio does: a stack machine of push / mult / add
// / sub / div and spline_tcb (or spline_linear) over one input. What reads them, and in which
// order, is erc.hpp. A port of src/lib/pose/formulas.ts.

#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>

namespace fv::pose {

// A stack entry: a number, or a knot [x, y, tension, continuity, bias] of a spline (the push's own,
// which outlives the evaluation).
struct Item
{
  enum class Kind
  {
    eNumber,
    eKnot,
    eOther,
  };
  Kind                       kind = Kind::eOther;
  double                     num  = 0;
  const std::vector<double>* knot = nullptr;
};

struct Op
{
  enum class Kind
  {
    ePush,
    eMult,
    eAdd,
    eSub,
    eDiv,
    eSplineTcb,
    eSplineLinear,
    eUnknown,
  };
  Kind kind = Kind::eUnknown;
  // ePush of a property: the index of the input it reads (-1: pushes its value, -2: unreadable)
  int input = -1;
  // ePush of a value: a number, or a knot (`knot`), which what it pushes points at
  Item::Kind          valueKind = Item::Kind::eOther;
  double              num       = 0;
  std::vector<double> knot;
};

// Compiles an operations array; `inputOf` maps a pushed url to its input index. Operations that
// aren't understood evaluate to nothing, as in the TS.
template <class InputOf>
std::vector<Op> compileOps(const dson::Json& ops, InputOf&& inputOf)
{
  std::vector<Op> out;
  out.reserve(ops.size());
  for(const dson::Json& o : ops)
  {
    Op         op;
    const auto name = dson::getString(&o, "op");
    if(name == "push")
    {
      op.kind = Op::Kind::ePush;
      if(const dson::Json* url = dson::get(&o, "url"))
        op.input = url->is_string() ? inputOf(url->get<std::string>()) : -2;
      else if(const dson::Json* val = dson::get(&o, "val"))
      {
        if(val->is_number())
        {
          op.valueKind = Item::Kind::eNumber;
          op.num       = val->get<double>();
        }
        else if(val->is_array())
        {
          op.valueKind = Item::Kind::eKnot;
          for(const dson::Json& k : *val)
            op.knot.push_back(k.is_number() ? k.get<double>() : NAN);
        }
      }
    }
    else if(name == "mult")
      op.kind = Op::Kind::eMult;
    else if(name == "add")
      op.kind = Op::Kind::eAdd;
    else if(name == "sub")
      op.kind = Op::Kind::eSub;
    else if(name == "div")
      op.kind = Op::Kind::eDiv;
    else if(name == "spline_tcb")
      op.kind = Op::Kind::eSplineTcb;
    else if(name == "spline_linear")
      op.kind = Op::Kind::eSplineLinear;
    out.push_back(std::move(op));
  }
  return out;
}

// Kochanek–Bartels (TCB) interpolation through knots [x, y, tension, continuity, bias], or linear.
double splineTcb(double x, std::span<const std::vector<double>* const> knots, bool linear);

// Runs a formula; `read(inputIndex)` gives a pushed property's value (nullopt: unknown input).
// Allocates nothing once its thread has run a formula as deep: the formulas on a thread share one
// stack, a push's read running the formulas it reads above this one's entries, which they leave as
// they found them.
template <class Read>
std::optional<double> evaluate(const std::vector<Op>& ops, Read&& read)
{
  thread_local std::vector<Item>                       shared;
  thread_local std::vector<const std::vector<double>*> knots;
  std::vector<Item>&                                   stack = shared;
  struct Restore
  {
    std::vector<Item>& stack;
    size_t             size;
    ~Restore() { stack.resize(size); }
  } restore{stack, stack.size()};
  const auto depth = [&] { return stack.size() - restore.size; };
  // pops the top; false when it is missing or not a number
  auto popNumber = [&](double& v) {
    if(depth() == 0)
      return false;
    const bool ok = stack.back().kind == Item::Kind::eNumber;
    v             = stack.back().num;
    stack.pop_back();
    return ok;
  };
  const auto number = [](double v) { return Item{Item::Kind::eNumber, v, nullptr}; };
  for(const Op& o : ops)
  {
    switch(o.kind)
    {
      case Op::Kind::ePush:
        if(o.input == -2)
          return std::nullopt;
        if(o.input >= 0)
        {
          const std::optional<double> v = read(o.input);
          if(!v)
            return std::nullopt;
          stack.push_back(number(*v));
        }
        else
          stack.push_back(Item{o.valueKind, o.num, o.valueKind == Item::Kind::eKnot ? &o.knot : nullptr});
        break;
      case Op::Kind::eMult:
      case Op::Kind::eAdd:
      case Op::Kind::eSub:
      case Op::Kind::eDiv: {
        double     a = 0, b = 0;
        const bool okB = popNumber(b);
        const bool okA = popNumber(a);
        if(!okA || !okB)
          return std::nullopt;
        stack.push_back(number(o.kind == Op::Kind::eMult ? a * b : o.kind == Op::Kind::eAdd ? a + b : o.kind == Op::Kind::eSub ? a - b : (b == 0 ? 0 : a / b)));
        break;
      }
      case Op::Kind::eSplineTcb:
      case Op::Kind::eSplineLinear: {
        double n = 0;
        if(!popNumber(n) || !(n >= 0) || size_t(n) + 1 > depth())
          return std::nullopt;
        // a spline reads nothing, so one list of its knots does for the thread
        const size_t first = stack.size() - size_t(n);
        knots.clear();
        for(size_t i = first; i < stack.size(); i++)
        {
          if(stack[i].kind != Item::Kind::eKnot)
            return std::nullopt;
          knots.push_back(stack[i].knot);
        }
        // x sits just below the knots
        if(stack[first - 1].kind != Item::Kind::eNumber)
          return std::nullopt;
        const double y = splineTcb(stack[first - 1].num, knots, o.kind == Op::Kind::eSplineLinear);
        stack.resize(first - 1);
        stack.push_back(number(y));
        break;
      }
      default:
        return std::nullopt;
    }
  }
  if(depth() == 0 || stack.back().kind != Item::Kind::eNumber)
    return std::nullopt;
  return stack.back().num;
}

}  // namespace fv::pose
