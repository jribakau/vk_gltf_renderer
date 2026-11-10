#pragma once

// A rigid part of a fitted item (docs/wardrobe-plan.md, step 4e): a prop hung from a joint, or a part
// with no skin inside an item that has one (a pin on a beret). It is moved as the surface it sits on
// moved: its item's own mesh where it sits on that, from as made to as fitted; else the skin, from the
// source body to where the maps put it on the figure. The move is the best turn and shift between the
// two (Horn's), a shift alone where too few points say how to turn or they turn too far.

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <figure/math/xform.hpp>

namespace fv::bodymap {
struct TypeMap;
}

namespace fv::garment {

// p' = to + turn (p - from)
struct RigidMove
{
  Quat turn;
  Vec3 from, to;
  Vec3 apply(const Vec3& p) const { return to + rotate(p - from, turn); }
  Mat4 matrix() const;
};

// Turns past this, in degrees, are taken for noise: a patch of skin that bends between two bodies
// says little about how a part on it turns.
constexpr double kRigidMaxTurn = 20;

// The best move taking each of `from` to its `to`; a shift alone with fewer than four points, or a
// turn past `maxTurn` degrees (180: any turn). None with no points.
std::optional<RigidMove> bestMove(const std::vector<Vec3>& from, const std::vector<Vec3>& to, double maxTurn = kRigidMaxTurn);

// The vertices of `surface` (xyz) within `reach` cm of `part` (xyz), measured from up to `samples` of
// the part's vertices spread over it; and the least distance found from part to surface.
struct Near
{
  std::vector<size_t> vertices;
  double              least = -1;
};
Near nearVertices(std::span<const float> surface, std::span<const float> part, double reach, size_t samples = 256);

// Where vertex `v` of a source body lands on the figure's skin (`figure`, its cage positions): through
// `source`'s map onto the reference and `target`'s back off it. None where either map has no place.
std::optional<Vec3> onFigure(const bodymap::TypeMap& source, const bodymap::TypeMap& target, std::span<const float> figure, size_t v);

}  // namespace fv::garment
