#pragma once

// Rain on a figure's skin, as a pure function; the pass that runs it is rain.hpp. Asked for on
// 2026-11-08 after DripSim, a Skyrim mod: rain lands where it reaches the body, the drops merge into
// beads, and a bead too heavy for its slope runs down the skin the way gravity pulls it in the figure's
// pose, leaving a wet track and now and then a small bead behind; what hangs under a chin or a fingertip
// drips off once heavy enough. Once the rain stops, it dries.
//
// A sketch of what sessile drops do, its numbers set by eye (rain_sim.cpp says which):
//  - a bead is a spherical cap meeting the skin at a fixed angle, so its volume gives its size;
//  - it sticks until gravity along the skin beats the pinning of its contact line, so it runs once its
//    contact radius squared times the sine of the slope passes a threshold, which a wet track lowers:
//    later runs follow earlier ones;
//  - running, it goes down the skin's slope as the smooth normals have it, wandering a little by a
//    noise over the body, along a crease where both sides fall towards it, leaves a film behind (a dry
//    skin takes more than a wet track) and a small bead now and then, and swallows the beads it meets;
//  - after the rain it dries by a contact radius a second, its tracks sooner.
// Everything is seeded: the same body, pose, settings and seed give the same water, so a scene file
// opens with the drops it was saved with.

#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include <figure/materials/water.hpp>
#include <figure/math/xform.hpp>

namespace fv::mesh {
struct BuiltMesh;
}

namespace fv::physics {

struct RainSettings
{
  bool     on     = false;
  float    amount = 0.6f;  // how hard it rains: 0.05 a drizzle, 1 a downpour
  float    time   = 30;    // how long it rained, s
  float    dried  = 0;     // how long since it stopped, s
  uint32_t seed   = 1;     // another seed lands other drops

  bool operator==(const RainSettings&) const = default;
};
// The ranges the settings are held to (a command, a scene file).
RainSettings clamped(RainSettings s);

// The skin the water runs on: triangles of cage vertices (the cage's polygons fanned), each with its
// corners' UVs as the mesh's file has them and its material group, and what lies across each edge.
struct RainSurface
{
  std::vector<uint32_t> tris;    // three a triangle
  std::vector<float>    uv;      // six a triangle
  std::vector<int32_t>  group;   // a triangle's material group; -1: drawn by none
  std::vector<int32_t>  across;  // three a triangle: across edge k (corner k to k + 1), 3 * triangle + its edge; -1: none

  size_t triangles() const { return group.size(); }
};
// From its parts, `across` made from the triangles' shared edges (a third triangle on an edge is left
// with none across).
RainSurface rainSurface(std::vector<uint32_t> tris, std::vector<float> uv, std::vector<int32_t> group);
// A built mesh's: the cage it was built from when it is refined (its source), else its drawn triangles,
// whose vertices are the cage's then. Empty for a refined mesh without its source.
RainSurface rainSurface(const mesh::BuiltMesh& geo);

struct RainInput
{
  std::span<const float> posed;  // xyz per cage vertex
  Vec3                   down{0, -1, 0};
  double                 unitsPerMetre = 100;  // DSON's centimetres
};

// Water on the skin, at a point of a triangle whose corners 1 and 2 weigh b1 and b2 (corner 0 the rest).
struct RainBead
{
  uint32_t tri = 0;
  float    b1 = 0, b2 = 0;
  float    radius = 0;  // m, where it meets the skin
  float    height = 0;  // m
};
// A stretch of a run's track inside one triangle, from (a1, a2) to (b1, b2).
struct RainTrack
{
  uint32_t tri = 0;
  float    a1 = 0, a2 = 0, b1 = 0, b2 = 0;
  float    radius = 0;  // m, half its width
  float    height = 0;  // m
};
struct RainResult
{
  std::vector<RainBead>  beads;
  std::vector<RainTrack> tracks;
  float                  damp    = 0;  // how wet the skin is all over, 0..1
  int                    landed  = 0;  // drops that reached the skin
  int                    runs    = 0;  // times a bead started to run
  int                    dripped = 0;  // beads that fell off
};
RainResult simulateRain(const RainSurface& surface, const RainInput& in, const RainSettings& settings);

// Per material group, what the renderer draws of it: the beads and tracks in the group's UVs (as the
// surface has them; the feed flips v when its upload does), and the damp.
std::map<int, mat::WaterLayer> waterLayers(const RainSurface& surface, const RainInput& in, const RainResult& result);

// The 2x2 (row-major) that takes an offset in a triangle's UVs to metres along the tangent and the
// bitangent as the renderer makes them there (vertex_tangents.comp.slang: the tangent is dP/du, the
// bitangent the normal crossed with it and turned by the handedness of dP/du against -dP/dv). False for
// a triangle without area in UV or in space. `p` in metres, `uv` as the triangle's corners have them.
bool uvToTangent(const std::array<Vec3, 3>& p, const std::array<std::array<float, 2>, 3>& uv, std::array<float, 4>& out);

}  // namespace fv::physics
