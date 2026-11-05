#pragma once

// A bone's move between two poses as a screw (docs/physics-plan.md, "Kinematic sources"): what a
// kinematic source sweeps its skin matrices by over a frame, so a limb turns about its joint instead of
// shrinking along a chord. Cloth's colliders and pinned vertices move by it (physics/cloth).

#include <figure/math/xform.hpp>

namespace fv::physics {

// A bone's skin matrix taken apart, so a sweep between two poses can turn it.
struct BonePose
{
  Vec3 t, s;
  Quat r;
};
BonePose poseOf(const Mat4& m);
// The matrix at fraction `t` of the move from `from` to `to` (0 the one, 1 the other).
Mat4 sweepAt(const BonePose& from, const BonePose& to, float t);

}  // namespace fv::physics
