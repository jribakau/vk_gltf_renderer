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

namespace fv::world {

struct RoadMesh
{
  std::vector<glm::vec3> positions, normals;  // about the quad's centre, in the scene
  std::vector<glm::vec2> uvs;                 // laid by the material's tile (RoadTiles)
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

// How far a repeat of each material's maps reaches, metres
struct RoadTiles
{
  double surface = 3.0;
  double bridge  = 3.0;
};

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

}  // namespace fv::world
