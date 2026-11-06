#pragma once

// The cloth island's build (docs/physics-plan.md step 12): from the surfaces, the particles, the
// distance rows, the collision faces and the partition the sweeps run in, for the CPU island and the
// GPU solver alike.

#include <cstdint>
#include <vector>

#include <figure/physics/cloth/surface.hpp>
#include <figure/physics/particles/particles.hpp>

namespace fv::physics::cloth {

// The rows that hold cloth together: distances between two vertices, of four kinds.
enum class DistanceKind
{
  eStretch,
  eShear,
  eBend,
  eReach,  // pulls only, and is swept with the stretch edges
};

// Whether a stretch-only sweep takes this edge.
constexpr bool holdsLength(DistanceKind kind)
{
  return kind == DistanceKind::eStretch || kind == DistanceKind::eReach;
}

// One distance row: its two vertices, its rest length and compliance, and a bend strut's buckling
// (shorter than `ratio` of its rest length, it is `buckle` as stiff).
struct Distance
{
  int          a, b;
  float        rest, alpha;  // compliance, cm per dyne
  DistanceKind kind;
  float        ratio, buckle;
};

// A collision face: its mesh, corners, layer, gap and friction, whether its mesh's own vertices meet
// it, and whether the cloth moves it.
struct Face
{
  int   mesh, a, b, c;
  int   layer;
  float offset, friction;
  bool  self, dynamic;
};

// What a cloth island is made of, built once from the surfaces' rest shapes: every vertex of every
// surface in one index space, the distance rows, the collision faces and who moves what. Shared by the
// CPU island and the GPU solver, so both simulate the same cloth.
struct Model
{
  int                  nv = 0;
  std::vector<int>     base, meshOf;
  std::vector<float>   rest, posed, strength, dampRate, friction, offset, grav, inv;
  std::vector<int>     layer;
  std::vector<uint8_t> visible, collides, dynamic;
  std::vector<Distance> edges;  // the distance rows
  // the sweep in parallel (physics/particles): edges whose free vertices all lie in one block of the
  // vertex range run block by block on the pool, the few that cross blocks after them on one thread.
  // Each list starts with the edges that hold length, so a sweep over those alone is a prefix. The
  // block count is fixed (particles::kBlocks), so a drape is the same on any machine
  physics::particles::Blocks blocks;
  std::vector<Face>          tris;
  // faces the pose moves and faces the cloth moves
  std::vector<int> staticTris, dynamicTris, allTris;
  // each vertex's neighbours over stretch edges, and the neighbours of those, sorted: a cloth does
  // not collide with the faces around a vertex, whose planes it is in or next to
  std::vector<int> ringStart, ring, ring2Start, ring2;
  // per face, the faces of its mesh across each of its three edges (-1 at a border): a vertex held
  // on a face hands the hold to the next face when it slides across an edge
  std::vector<int> triNext;
  std::vector<int> kinematicVerts;  // every vertex a pose moves
  std::vector<int> freeVerts;       // every vertex the cloth moves
  // the free vertices of partial strength, tethered to their skinned position: each one's tether
  // compliance (cm per dyne, 0 for the rest), from kAttachSag and its strength
  std::vector<int>   softVerts;
  std::vector<float> attAlpha;
  float              maxOffset = 0, minOffset = 0.05f;

  static Model build(const std::vector<Surface>& meshes, float air, bool reachLinks = true);
};

// Moves the free vertices of `x` (the model's rest shape, usually) that start inside, or within the
// gap of, a lower layer out to the gap along the nearest face of that layer, so a fitted shape that
// overlaps the body does not start inside it. Both solvers do this once before their first step.
void separateStart(const Model& model, std::vector<float>& x);

}  // namespace fv::physics::cloth
