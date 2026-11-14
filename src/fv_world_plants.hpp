#pragma once

//-------------------------------------------------------------------------------------------------------
// What grows on a planet (fv_world.hpp), by its biomes (Biome::plants, grass, drift), with the kit's
// plants a meadow and a forest are planted with (fv_backdrop.hpp). Two ways of standing, as theirs:
//
// - Trees, saplings, bushes, rocks and dead wood each have a place (`standing`): those of a quad of
//   the planet's ground are drawn with it, a variant's as instancing copies. A place follows from the
//   planet alone, not from the quad: each role has a grid of its own over a cube face, a plant a
//   cell at the most, kept by how thickly the biomes there grow it, so the plants of a quad are those
//   of its four children. Far quads keep a share of them, each grown to cover what the others did;
//   further still they have none, and a wood is the colour of its ground.
// - Grass, and the flowers and ferns in drifts in it, are tiles (`grassTiles`, `grassPlace`): a square
//   of ground with its plants is one mesh, drawn wherever a square of its kind lies near where one
//   looks from. The squares are a cube face's, laid by the meadow's own rule (fv::backdrop::
//   grassTiles) on the face's plane; a tile is leant as the ground is and stretched to its square,
//   which is not quite one, a face being bent onto the sphere.
//
// Nothing stands under water or on a shore, and nothing with a place where the figures do.
//
// Vulkan-free.
//-------------------------------------------------------------------------------------------------------

#include <array>
#include <compare>
#include <span>
#include <string_view>
#include <vector>

#include "fv_backdrop.hpp"
#include "fv_world.hpp"
#include "fv_world_sea.hpp"

namespace fv::world {

// ---- what has a place ----------------------------------------------------------------------------------

// The kit's roles a planet places, in the order `standing` counts them
inline constexpr std::array<std::string_view, 7> kStandingRoles = {"tree", "grown", "sapling", "shrub", "rock", "stump", "log"};

// The plants of one variant of a role on a quad: where each stands, about `centre`
struct Stand
{
  uint8_t                role    = 0;  // of kStandingRoles
  uint32_t               variant = 0;
  std::vector<glm::mat4> places;       // on the ground, turned about the planet's vertical there and scaled
};

// What stands on a quad's ground. `variants`: how many the kit has of each role (none: the role is
// left out). `centre`: what the places are about, the quad's mesh's (ChunkMesh::centre). `density`:
// of every plant's own, as the backdrop's.
std::vector<Stand> standing(const World& world, const Quad& quad, const glm::dvec3& centre, std::span<const uint32_t> variants, double density = 1.0);

// The share of a role's plants a quad `width` metres across keeps, and how much each of those grows;
// none (0) on quads too far to show them
struct Kept
{
  float share = 0.0F;
  float grow  = 1.0F;
};
Kept keptOn(std::string_view role, double width);

// What stands on a quad's sea floor (fv_world_sea.hpp), of `meshes`: seagrass on the shallows, kelp
// where the water is cool and deep enough to hold it, corals where it is warm, sponges deeper, rocks
// anywhere. Each no taller than the water over it. A Stand's `role` is of kSeaRoles and its `variant`
// counts the role's meshes in the order they come. Only the finest quads have any: they are seen
// from near the water, or from in it.
std::vector<Stand> seaStanding(const World& world, const Quad& quad, const glm::dvec3& centre, std::span<const SeaMesh> meshes, double density = 1.0);

// ---- grass ---------------------------------------------------------------------------------------------

// One kind of tile a planet lays
struct TileKind
{
  fv::backdrop::Cover cover;
  Grass               grass = Grass::eNone;  // the grass it is, or
  std::string_view    drift;                 // the drift it is of: flower | fern
  uint8_t             level = 0;             // of the grass's three, or 0 a drift's middle and 1 its edge
};
// Every kind, for a planet of `density`: the tiles' meshes are made from these
std::vector<TileKind> tileKinds(const World& world, double density = 1.0);

// A tile: a cell of a face's grid at its kind's level
struct GrassTile
{
  uint8_t face = 0;
  uint8_t kind = 0;  // of tileKinds
  int32_t x = 0, z = 0;
  auto    operator<=>(const GrassTile&) const = default;
};

// The tiles for where one looks from (`foci`, in the scene: the figures' place and the camera),
// and what was last laid, so a view that hovers at a boundary changes nothing. `was` is kept by
// whoever asks, a face at a time.
struct GrassWas
{
  std::array<std::vector<fv::backdrop::Tile>, 6> faces;
};
// `size`: metres across the thick grass round each focus (the backdrop's `size`).
std::vector<GrassTile> grassTiles(const World& world, std::span<const glm::dvec3> foci, std::span<const TileKind> kinds, GrassWas& was, double size = 60.0);

// Where a tile lies in the scene: on the ground, leant as it is and stretched to its square
struct GrassPlace
{
  glm::dvec3 position{0.0};
  glm::quat  rotation{1.0F, 0.0F, 0.0F, 0.0F};
  glm::vec3  scale{1.0F};
  // How level the ground is there: the cosine of its lean. Under kGrassLevel it is too steep for
  // grass (bare rock by then, surfaceColour), and the tile is left out.
  float level = 1.0F;
};
inline constexpr float kGrassLevel = 0.75F;
GrassPlace grassPlace(const World& world, const GrassTile& tile, const TileKind& kind);

}  // namespace fv::world
