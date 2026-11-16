#pragma once

//-------------------------------------------------------------------------------------------------------
// What a planet's roads and bridges are drawn with near the eye (fv_world_places.hpp says where they
// are): meshes a quad of the planet's ground holds, made with its ground on the pool.
//
// - A road's surface: a ribbon along its middle line as wide as its carriageway, laid on the ground
//   as the quad draws it and a few centimetres over it (the ground is cut and filled to the road's
//   level, so it is level across, but further off its points are metres apart: the ribbon follows
//   their triangles, never under them), its edges going down into the ground. Only on quads whose
//   ground is fine enough (kSurfaceFootprint): further off a road is the paint its ground has. An
//   asphalt road's alone: a track is the ground's earth.
// - On an asphalt road, its markings: a dashed line down its middle and a line along each edge, worn,
//   some dashes gone and the edge lines broken.
// - The surface and its markings are laid on the asphalt's maps (fv_world_asphalt.hpp) edge to edge,
//   a stretch of them between two of their joints at a time: each way lays the stretches in an order
//   of its own, each either way along and either way across, cut where one gives way to the next (on
//   a joint, a crack across the road). And tinted by a weathering wider than a stretch (the vertices'
//   colours), so no run of road is another's.
// - Where a way ends (WayEnd): on into the next way, the surfaces meeting; joining another road from
//   the side, short of that road's edge; or at an open end, broken off there with soil on it and its
//   edge going down into the ground, its markings stopping short of it.
// - A bridge: its deck under the road from bank to bank and on into each (kBridgeEnds), a parapet
//   each side, piers in the water when it is long, and an abutment at each end down into the bank.
//   On whichever quad holds its middle, however coarse: a bridge is seen from further than a road's
//   surface. Over the water the road is held at the deck's level, not laid on the river's bed.
//
// A stretch of road is the quad's whose square the stretch's middle falls in: quads side by side, of
// whatever levels, draw it once between them.
//
// Vulkan-free.
//-------------------------------------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "fv_world.hpp"
#include "fv_world_asphalt.hpp"

namespace fv::world {

struct Bridge;  // fv_world_places.hpp

struct RoadMesh
{
  std::vector<glm::vec3> positions, normals;  // about the quad's centre, in the scene
  // a road's surface and paint: where on the asphalt's maps (roadUv); a bridge: by its tile (RoadTiles)
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec4> colours;  // a road's surface and paint: the asphalt's tint; a bridge: none (white)
  std::vector<uint32_t>  indices;
  bool                   empty() const { return indices.empty(); }
};

struct RoadMeshes
{
  RoadMesh surface;  // asphalt
  RoadMesh marking;  // paint on it
  RoadMesh bridge;   // concrete
  bool     empty() const { return surface.empty() && marking.empty() && bridge.empty(); }
};

// How far a repeat of a bridge's maps reaches, metres (a road's are its asphalt's: roadUv)
struct RoadTiles
{
  double bridge = 3.0;
};

// Where a point of a road's surface is on its asphalt's maps: `x` metres across from the middle line
// of a carriageway `half` wide either side (0 at one edge, 1 at the other) and `along` metres along
// the maps, a repeat a unit
inline glm::vec2 roadUv(double x, double half, double along)
{
  return glm::vec2(float(0.5 + 0.5 * x / half), float(along / kAsphaltRepeat));
}

// The coarsest ground a road's surface is laid on: metres between its points (the three finest
// levels of quads, out to some 700 m from the eye)
inline constexpr double kSurfaceFootprint = 4.1;
// How far over the ground as drawn: a few centimetres, more on coarser ground
inline double surfaceLift(double footprint)
{
  return 0.03 + 0.02 * footprint;
}

// A quad's roads and bridges, about `centre` (its ground's, ChunkMesh::centre). Empty on a planet
// with no places.
RoadMeshes roadMeshes(const World& world, const Quad& quad, const glm::dvec3& centre, const RoadTiles& tiles = {});

// One bridge as roadMeshes draws it (its deck, parapets, abutments and piers), about `centre`: what
// someone on it walks and drives on, and meets (play mode). And where the middle of its deck is in the
// scene, on its road's level.
RoadMesh   bridgeMesh(const World& world, const Bridge& bridge, const glm::dvec3& centre, const RoadTiles& tiles = {});
glm::dvec3 bridgeMiddle(const World& world, const Bridge& bridge);

}  // namespace fv::world
