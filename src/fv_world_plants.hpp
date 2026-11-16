#pragma once

//-------------------------------------------------------------------------------------------------------
// What grows on a planet (fv_world.hpp), by its biomes (Biome::plants, grass, drift), with the kit's
// plants a meadow and a forest are planted with (fv_backdrop.hpp). Two ways of standing, as theirs:
//
// - Trees, saplings, bushes, rocks and dead wood each have a place (`standing`): those of a quad of
//   the planet's ground are drawn with it, a variant's as instancing copies. A place follows from the
//   planet alone, not from the quad: each role has a grid of its own over a cube face, a plant a
//   cell at the most, kept by how thickly the biomes there grow it. Each plant has a reach, how far
//   it is seen from, by its size among its kind's: a wood's largest trees for kilometres, its
//   smallest and its undergrowth for a few hundred metres. A quad holds the plants whose reach passes
//   where the quad can draw, so a plant is the same plant, in its place and of its size, whichever
//   quad holds it, and whoever draws them shows each by its own distance (`grownAt`). The trees a
//   quad does not hold stand on its ground as a canopy (`canopyLift`).
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
#include <bit>
#include <compare>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "fv_backdrop.hpp"
#include "fv_world.hpp"
#include "fv_world_sea.hpp"

namespace fv::world {

// ---- what has a place ----------------------------------------------------------------------------------

// The kit's roles a planet places, in the order `standing` counts them. A biome grows its own of
// them (World::biomes): the firs of a wood and a meadow's edge, a grassland's leafy trees, pines in
// woods and on the steppe, the dry land's trees and bushes, its bare boulders, mossy rocks where it
// is wet, and willows and alders along rivers and by lakes (`riverside`, Growth::water)
inline constexpr std::array<std::string_view, 13> kStandingRoles = {"tree",    "grown", "sapling", "shrub",   "rock",    "stump",    "log",
                                                                    "broadleaf", "pine", "drytree", "drybush", "boulder", "riverside"};
// What a role is: a tree is a wood's, seen for kilometres, and none stands on a place's floor; the
// dry land's small trees and a sapling stand on none either; a stone stands on bare rock too
constexpr bool isTree(std::string_view role)
{
  return role == "tree" || role == "grown" || role == "broadleaf" || role == "pine" || role == "riverside";
}
// The role whose models draw a role's plants where the kit has none of its own: a riverside tree is
// a leafy one (the kit's own download has no willows; kit.local.json may)
constexpr std::string_view drawnAs(std::string_view role)
{
  return role == "riverside" ? std::string_view("broadleaf") : role;
}
constexpr bool isTall(std::string_view role)
{
  return isTree(role) || role == "drytree" || role == "sapling";
}
constexpr bool isStone(std::string_view role)
{
  return role == "rock" || role == "boulder";
}

// The plants of one variant of a role on a quad: where each stands, about `centre`, and how far it is
// seen from; the furthest seen first
struct Stand
{
  uint8_t                role    = 0;  // of kStandingRoles
  uint32_t               variant = 0;
  std::vector<glm::mat4> places;       // on the ground, turned about the planet's vertical there and scaled
  std::vector<float>     reach;        // metres, one a place, falling
};

// How far a role's plants are seen from: every one from `all` metres, the largest from `most`, those
// between by their size among their kind's, as many of them at a distance as its square allows (so
// the plants seen from one place are about as many however far the largest are seen). `all` follows
// from how thickly the role grows where it grows thickest, so that those seen from one place stay
// under the role's count (kTreesSeen; kUndergrowthSeen for what else stands on land; kOthersSeen for
// the sea floor's): no nearer than `least`. Where a biome grows several trees (firs and pines in one
// wood), `standing` asks with all of theirs together, so the count holds for its trees.
struct Reach
{
  double all = 0.0, most = 0.0;
  // of a plant `rank` among its kind's by size: 0 the largest, toward 1 the smallest
  double of(double rank) const { return std::min(all / std::sqrt(std::max(rank, 1e-6)), most); }
};
inline constexpr double kTreesSeen = 200000.0, kUndergrowthSeen = 15000.0, kOthersSeen = 5000.0;
Reach reachOf(std::string_view role, double perSquareMetre);
// The reach a plant needs for a quad to hold it: the quad draws only while the eye is kSplit of its
// widths away or further (less a margin: the eye's distance to a quad is to its ground as drawn),
// and the finest quads, which draw underfoot, hold every plant
inline constexpr double kHeldMargin = 0.85;
double heldFrom(const World& world, const Quad& quad);
// How much of its size a plant shows `away` metres from the eye: none past its reach, growing to all
// of it over the last kGrowsOver of the way in, so it comes into view small
inline constexpr double kGrowsOver = 0.15;
inline float grownAt(double away, double reach)
{
  return reach > 0.0 ? float(std::clamp((1.0 - away / reach) / kGrowsOver, 0.0, 1.0)) : 0.0F;
}

// What stands on a quad's ground. `variants`: how many the kit has of each role (none: the role is
// left out). `centre`: what the places are about, the quad's mesh's (ChunkMesh::centre). `density`:
// of every plant's own, as the backdrop's.
std::vector<Stand> standing(const World& world, const Quad& quad, const glm::dvec3& centre, std::span<const uint32_t> variants, double density = 1.0);

// ---- far woods as a canopy -----------------------------------------------------------------------------

// A quad holds only the trees seen from where it draws, the largest (heldFrom), and shows each of
// them only from as near as its reach: the rest of a far wood stands on its ground mesh as a canopy,
// its surface raised as high as their crowns over the ground they stand on (groundMesh's lift). Which
// trees those are is decided once for the quad, at kCanopyAway of its widths: where its trees are on
// the whole while it draws (it draws from kSplit of its widths to about twice that, and its trees
// lie across it). Nearer, a few of them are drawn and in the canopy both; further, a few are in
// neither; on the whole, as many as stand. Where its points are a few metres apart each is a crown's
// top or the floor between, by how much of the ground those trees' crowns cover; where they are tens
// of metres apart, the wood as a whole, as high as its crowns stand on the whole and its colour as
// much as a slanting view sees of them. Where `standing` would stand no tree, none.
//
// The kit's trees, as the canopy is made of them: each role's models on the whole, at a scale of 1
// (none: the kit has none of the role, and none of its trees stand)
struct Canopy
{
  double                                   density = 1.0;  // of every plant's own, as `standing`'s
  std::array<float, kStandingRoles.size()> height{};       // metres from the foot to the top
  std::array<float, kStandingRoles.size()> crown{};        // how far its crown reaches from the trunk, metres
};
inline constexpr double kCanopyAway = 5.0;
// The lift of a quad's ground; none where the quad shows all its trees (the finest, and those near
// them) or none stand
LiftOf canopyLift(const World& world, const Quad& quad, const Canopy& canopy);

// What stands on a quad's sea floor (fv_world_sea.hpp), of `meshes`: seagrass on the shallows, kelp
// where the water is cool and deep enough to hold it, corals where it is warm, sponges deeper, rocks
// anywhere. Each no taller than the water over it. A Stand's `role` is of kSeaRoles and its `variant`
// counts the role's meshes in the order they come. Each with a reach, as the land's: they are seen
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
  // What grows on each tile's ground, as found when it was first asked for (its biome's number;
  // none: nothing grows there), so a tile asked for again takes no look at the planet: a view that
  // moves asks for the same few thousand tiles every time. Of one World: whoever keeps this clears
  // `grounds` when the planet is another.
  struct Key
  {
    uint8_t face = 0;
    float   size = 0.0F;
    int32_t x = 0, z = 0;
    bool    operator==(const Key&) const = default;
  };
  struct Hash
  {
    size_t operator()(const Key& k) const
    {
      uint64_t h = (uint64_t(uint32_t(k.x)) << 32U) | uint64_t(uint32_t(k.z));
      h ^= (uint64_t(std::bit_cast<uint32_t>(k.size)) * 31U + k.face) * 0x9E3779B97F4A7C15ULL;
      return std::hash<uint64_t>{}(h);
    }
  };
  struct Ground
  {
    int8_t   biome = -1;
    uint32_t asked = 0;  // the pass it was last asked in
  };
  std::unordered_map<Key, Ground, Hash> grounds;
  uint32_t                              pass = 0;
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
