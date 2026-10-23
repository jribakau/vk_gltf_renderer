#pragma once

// One DSON formula's operations (ERC), run as DAZ Studio does: a stack machine of push / mult / add
// / sub / div and spline_tcb (or spline_linear) over one input. What reads them, and in which
// order, is erc.hpp. A port of src/lib/pose/formulas.ts.

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>

namespace fv::pose {

// A stack entry: a number, or a knot [x, y, tension, continuity, bias] of a spline.
struct Item
{
  enum class Kind
  {
    eNumber,
    eKnot,
    eOther,
  };
  Kind                kind = Kind::eOther;
  double              num  = 0;
  std::vector<double> knot;
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
  // ePush of a property: the index of the input it reads (-1: pushes `value`, -2: unreadable)
  int  input = -1;
  Item value;
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
          op.value.kind = Item::Kind::eNumber;
          op.value.num  = val->get<double>();
        }
        else if(val->is_array())
        {
          op.value.kind = Item::Kind::eKnot;
          for(const dson::Json& k : *val)
            op.value.knot.push_back(k.is_number() ? k.get<double>() : NAN);
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
double splineTcb(double x, std::vector<const std::vector<double>*> knots, bool linear);

// Runs a formula; `read(inputIndex)` gives a pushed property's value (nullopt: unknown input).
template <class Read>
std::optional<double> evaluate(const std::vector<Op>& ops, Read&& read)
{
  std::vector<Item> stack;
  stack.reserve(8);
  // pops the top; false when it is missing or not a number
  auto popNumber = [&](double& v) {
    if(stack.empty())
      return false;
    const bool ok = stack.back().kind == Item::Kind::eNumber;
    v             = stack.back().num;
    stack.pop_back();
    return ok;
  };
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
          Item it;
          it.kind = Item::Kind::eNumber;
          it.num  = *v;
          stack.push_back(std::move(it));
        }
        else
          stack.push_back(o.value);
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
        Item it;
        it.kind = Item::Kind::eNumber;
        it.num  = o.kind == Op::Kind::eMult ? a * b : o.kind == Op::Kind::eAdd ? a + b : o.kind == Op::Kind::eSub ? a - b : (b == 0 ? 0 : a / b);
        stack.push_back(std::move(it));
        break;
      }
      case Op::Kind::eSplineTcb:
      case Op::Kind::eSplineLinear: {
        double n = 0;
        if(!popNumber(n) || !(n >= 0) || size_t(n) + 1 > stack.size())
          return std::nullopt;
        const size_t                            first = stack.size() - size_t(n);
        std::vector<const std::vector<double>*> knots;
        for(size_t i = first; i < stack.size(); i++)
        {
          if(stack[i].kind != Item::Kind::eKnot)
            return std::nullopt;
          knots.push_back(&stack[i].knot);
        }
        // x sits just below the knots
        if(stack[first - 1].kind != Item::Kind::eNumber)
          return std::nullopt;
        Item it;
        it.kind = Item::Kind::eNumber;
        it.num  = splineTcb(stack[first - 1].num, std::move(knots), o.kind == Op::Kind::eSplineLinear);
        stack.resize(first - 1);
        stack.push_back(std::move(it));
        break;
      }
      default:
        return std::nullopt;
    }
  }
  if(stack.empty() || stack.back().kind != Item::Kind::eNumber)
    return std::nullopt;
  return stack.back().num;
}

}  // namespace fv::pose
