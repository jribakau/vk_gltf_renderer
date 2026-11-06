#pragma once

// A capsule floor under the body (physics-plan step 13c). The cloth meets the body's faces with sided
// collisions: a vertex is pushed back out the side it was last found on, so one carried or swept deep
// into a limb (a late step's own skin took a dress into a swinging thigh; the collision passes come
// too late for a fast shin) is kept inside for good, and its recovery trusts only 3 cm behind the
// nearest face, beyond which that face is as likely the far side of the limb. Games put capsules
// inside the body for this (Unity, PhysX and Unreal cloth collide with them): a capsule has one
// inside, so a vertex in it is pushed out whatever side it came from, onto a floor within the faces'
// recovery depth, which then finishes the job.
//
// The floor is fitted from the body surface the cloth meets, per skin bone: the cage vertices whose
// strongest influence the bone is, along their principal axis, cut into pieces about kFloorPiece cm
// long so a tapering limb gets a floor that follows it. Each piece's capsule is as thick as it can be
// while every vertex of the body stays outside it by kFloorMargin (its ends drawn in when that makes
// it thicker: a closed end, a hand's or a foot's, binds a capsule that reaches it), and its middle must
// lie in the body. Fingers, eyes and the like give nothing (too short or too thin). Each capsule's axis
// is two vertices of a collider surface of its own, skinned to the bone alone, so both solvers pose it
// with the body and nothing else changes.

#include <optional>
#include <vector>

#include <figure/physics/cloth/surface.hpp>

namespace fv::physics::cloth {

// A piece of a bone's cluster along its axis, cm: a limb is cut into pieces about this long.
constexpr float kFloorPiece = 8.f;
// How far inside the skin a floor stays, cm.
constexpr float kFloorMargin = 0.3f;
// A capsule thinner than this, or a cluster shorter than kFloorLeast cm along its axis, gives none.
constexpr float kFloorLeastRadius = 1.5f;
constexpr float kFloorLeast       = 5.f;

// The capsule floor of `body` (a skinned collider surface: its restLocal, four influences and bones),
// as a collider surface with no faces whose vertices are the capsules' ends in the body's space, each
// skinned to its bone alone, in the job's space by the body's toSpace, posed at the body's bones; its
// `capsules`. Nothing when the body isn't skinned, or no capsule fits.
std::optional<Surface> capsuleFloor(const Surface& body);

// How many of `points` (xyz each) lie inside any of `floor`'s capsules at `at` (xyz per vertex of the
// floor, its rest or posed): for a fit's check and the tests.
size_t insideFloor(const Surface& floor, const std::vector<float>& at, const std::vector<float>& points);

}  // namespace fv::physics::cloth
