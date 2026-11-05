#include <figure/physics/sweep.hpp>

#include <algorithm>
#include <cmath>

namespace fv::physics {

BonePose poseOf(const Mat4& m)
{
  BonePose b;
  b.t = {m.e[12], m.e[13], m.e[14]};
  b.r = rotationOf(m);
  const Vec3   c0{m.e[0], m.e[1], m.e[2]}, c1{m.e[4], m.e[5], m.e[6]}, c2{m.e[8], m.e[9], m.e[10]};
  const double det = c0.x * (c1.y * c2.z - c1.z * c2.y) - c0.y * (c1.x * c2.z - c1.z * c2.x) + c0.z * (c1.x * c2.y - c1.y * c2.x);
  b.s = {length(c0) * (det < 0 ? -1 : 1), length(c1), length(c2)};
  return b;
}

// The move from one pose to the other is a screw: a turn about an axis through a pivot and a slide
// along that axis. A skin matrix is a turn about its joint, so the pivot is the joint, and the sweep
// turns the limb about it; interpolating translation and rotation apart would turn it about the
// mesh's origin and carry the joint off along an arc of its own.
Mat4 sweepAt(const BonePose& from, const BonePose& to, float t)
{
  if(t >= 1)
    return compose(to.t, to.r, to.s);
  if(t <= 0)
    return compose(from.t, from.r, from.s);
  const Vec3 scale = from.s + (to.s - from.s) * t;
  // the move applied after `from`: rotation rd about an axis, translation td
  Quat rd = to.r * from.r.inverse();
  if(rd.w < 0)
    rd = {-rd.x, -rd.y, -rd.z, -rd.w};
  const Vec3   td      = to.t - rotate(from.t, rd);
  const double sinHalf = std::sqrt(std::max(0.0, 1 - rd.w * rd.w));
  const Quat   ru      = slerp(Quat{}, rd, t);
  if(sinHalf < 1e-6)
    return compose(from.t + td * t, (ru * from.r).normalized(), scale);
  const Vec3   axis  = Vec3{rd.x, rd.y, rd.z} * (1 / sinHalf);
  const double along = dot(td, axis);
  const Vec3   perp  = td - axis * along;
  // the pivot: the point in the plane across the axis the turn carries by `perp`
  const double tanHalf = sinHalf / rd.w;
  const Vec3   across{axis.y * perp.z - axis.z * perp.y, axis.z * perp.x - axis.x * perp.z, axis.x * perp.y - axis.y * perp.x};
  const Vec3   pivot = perp * 0.5 + across * (0.5 / tanHalf);
  const Vec3   tu    = pivot - rotate(pivot, ru) + axis * (along * t);
  return compose(rotate(from.t, ru) + tu, (ru * from.r).normalized(), scale);
}

}  // namespace fv::physics
