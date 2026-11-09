#pragma once

// A body's smooth vertex frames, and a map's turns between two bodies' frames (docs/wardrobe-plan.md,
// step 3, "Frames"). A garment's offset from its body is kept in such a frame, so it goes over to
// another body turned as that body's surface is turned there: an offset on a T-posed Genesis 3
// sleeve lands on Genesis 9's lowered arm lowered with it.

#include <vector>

#include <figure/bodymap/clone.hpp>
#include <figure/math/xform.hpp>

namespace fv::bodymap {

// Per vertex: the unit normal of the outer skin round it, and a tangent across it whose length, 0 to
// 1, says how well it is set (both zero at a vertex on no outer skin).
struct Frames
{
  std::vector<Vec3> normal, tangent;
};

// Each vertex's frame. Its tangent is its regions' limb axis (on a border, the axes of the regions it
// holds, averaged), less its part along the normal: it lies along a limb on its sides and fades where
// the axis runs along the normal (a fingertip, the crown), or where a border's axes cancel (where the
// thigh, down, meets the pelvis, up). A region's axis runs joint to joint, from the middle of its band
// with the region above to the middle of its band with the region below (the regions' borders, where
// each rig's weights hand over, agree between rigs better than the regions' own shapes do); at a
// limb's end (a finger, the head) to the middle of its own skin, by area. The pelvis's runs up to its
// band with the abdomen, and a breast, a bump on the chest rather than a limb, takes the chest's. All
// from the body's own regions, so two bodies' frames agree where their regions do.
Frames vertexFrames(const Body& body);

// Per vertex of `source`, the turn from its frame to `onto`'s where `p` put it (`onto`'s frames
// blended there): the normal onto the other the shortest way, then about it as far as takes the
// tangent onto the other's. Where either tangent is weakly set that last twist is scaled down, and
// below a tenth left out. The identity at a vertex `p` left unmapped.
std::vector<Quat> frameTurns(const Frames& source, const Frames& onto, const Projection& p);

}  // namespace fv::bodymap
