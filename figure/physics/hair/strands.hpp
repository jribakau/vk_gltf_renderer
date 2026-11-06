#pragma once

// Strand hair on the physics core (physics-plan step 14): dForce strand hair has no polygons for the
// cloth's rows, so each strand is a chain hanging from its root, of the core's follow-the-leader rows
// solved root to tip (particles::projectLeader), so it can't stretch: hundreds of thousands of strand
// points fall, meet the colliders as a distance field (physics::DistanceField) and come to rest in a
// fraction of a second on the pool. As the files' "Self Collide" is off on this hair, strands don't meet
// each other: a surface's "Collision Offset" keeps its layer that far off the body, which is what gives
// the hair its volume.
//
// What is strand hair's own stays here: only some of a strand's points are simulated, every few
// centimetres, the rest following their span, with their offset from it turned as the span turned, so
// the waves shorter than a span stay; and each strand stops once it has lain still a few frames.
//
// Not DAZ's solver. Its shape stiffnesses have no stated unit, so `kShapeScale` is a first guess.

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include <figure/physics/cloth/surface.hpp>

namespace fv::physics::hair {

// The strands of one mesh, in the job's space.
struct StrandSet
{
  // The cage skinned at the pose to hang from (xyz per point): the roots stay here.
  std::vector<float> posed;
  // Strand i is verts[start[i] .. start[i + 1]), root first
  std::vector<int32_t> start, verts;
  // Per strand. 0 stays as posed, 1 hangs freely, between: part of the way there.
  std::vector<float> strength;
  // Per strand: how far off a collider it stays, cm, and how much of its motion along one it loses
  std::vector<float> offset, friction;
  // Per strand: "Local Shape Constraint Stiffness" and its tip's, the pull back to the authored bends
  std::vector<float> shapeRoot, shapeTip;
  // How much body the hair keeps against gravity, 0 to 1: strands stand further off what they rest on
  // and hold their shape for longer, both most at the root. 0 is the file's offsets alone, which on most
  // hair leaves it flat to the head. Not a DAZ setting.
  float volume = 0;
  // Its authored shape is a style, not a pose waiting for a simulation (dson::strandsUnsettled): each
  // span leans back towards its authored direction, so curls stay curls and gravity only makes it
  // droop, and more Volume means closer to the style rather than further off the head.
  bool styled = false;
  // 0: passes through colliders ("Collide" off)
  std::vector<uint8_t> collide;
  std::array<float, 3> gravity{0, -980, 0};
};

struct StrandOptions
{
  // Simulated points this far apart along a strand, cm
  float spacing = 3;
  // Frames of 1/30 s to fall and come to rest, at most
  int maxFrames = 75;
  int substeps  = 2;
  // The largest move of any of a strand's points over a frame, cm, under which it counts as still
  float restMove = 0.05f;
  // Asked once a frame: true ends it where it is, as an edit has made it out of date.
  std::function<bool()> cancelled;
};

// Every point's position after the fall (xyz per point, the same layout as `posed`). `colliders` are
// taken at their `posed` shape.
std::vector<float> settle(const StrandSet& strands, const std::vector<cloth::Surface>& colliders, const StrandOptions& options = {});

}  // namespace fv::physics::hair
