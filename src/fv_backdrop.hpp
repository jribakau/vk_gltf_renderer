#pragma once

//-------------------------------------------------------------------------------------------------------
// The backdrop (the lighting's `backdrop` section, protocol 64; the app's Light tab › Backdrop): ground
// that rolls away from where the figures stand, planted with a kit's plants. A meadow: grass and
// flowers underfoot, saplings further out, and trees in groves that close the horizon.
//
// Everything here follows from the spec alone, so a scene file keeps only the spec: the ground's height
// (`height`), its mesh (`terrain`), and where each plant stands, by a seed. Two ways of standing:
//
// - Trees and saplings each have a place (`layers`, `scatter`): the host (fv_host_backdrop.cpp) draws a
//   variant once with its places as instancing copies (rs::MeshInstance::copies). They stand round the
//   origin and never change.
// - Grass and flowers are tiles (`grassCovers`, `patch`, `grassTiles`): a square of ground with its
//   few hundred plants is one mesh, made once, and drawn wherever a square of its kind lies. Which
//   squares are thick, thinner or sparse follows where one looks from, so there is grass under any
//   camera, and a camera's move changes a few tiles at the level boundaries, far from it, and
//   nothing near. (Each plant an instance of its own was tried first: following the camera meant
//   200,000 instances made again, a fifth of a second every time.)
//
// The plants are a kit's, in a folder the app fills (Poly Haven's CC0 models, their cutout maps put
// back): `kit.json` names each file's role (KitPlant). Which roles a kind plants, how thickly and
// where, is `layers`.
//
// Vulkan-free; distances in metres, the figures' floor at y = 0.
//-------------------------------------------------------------------------------------------------------

#include <array>
#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include <glm/glm.hpp>

namespace fv::backdrop {

struct Spec
{
  std::string kind    = "none";  // none | meadow
  int         seed    = 1;       // which meadow: the hills and where everything stands
  double      size    = 60.0;    // metres across the thick grass round where one looks from; the saplings' reach
  double      density = 1.0;     // of every layer's own, 0.25 to 2
  double      relief  = 1.0;     // how hilly the ground is beyond the clearing, 0 flat to 2
  bool        operator==(const Spec&) const = default;
};

inline bool planted(const Spec& s)
{
  return s.kind != "none";
}

// The ground stays flat (the floor's height) within this of the origin, where the figures stand, and
// rises into its hills over kClearingEdge more
inline constexpr float kClearing     = 5.0F;
inline constexpr float kClearingEdge = 25.0F;
// Where the ground ends: far enough for its hills to be the horizon's
inline constexpr float kHorizon = 900.0F;

// The ground's height over the floor at (x, z)
float height(const Spec& spec, float x, float z);

// The ground as a mesh: a disc, its rings fine underfoot and coarser with distance out to kHorizon, with
// smooth normals and UVs that tile every `tile` metres
struct Terrain
{
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  // 0..1 a vertex, in drifts some 40 m across: where the ground is drier. A tiled texture is one
  // colour from far off; the kit's two tints mixed by this give the distance its patches
  std::vector<float>    shades;
  std::vector<uint32_t> indices;
};
Terrain terrain(const Spec& spec, float tile);

// A layer of one role's plants
struct Layer
{
  std::string role;                  // the kit's role it plants: grass, flower, sapling, tree
  double      perSquareMetre = 1.0;  // where it stands thickest, before the spec's density
  float       from           = 0.0F; // none nearer the origin than this
  float       to             = 30.0F; // nor further
  // Past `thin` metres the layer thins as 1 / (1 + (d / thin)^2) and each plant grows by the root of
  // that, up to `grow` times, so distant ground keeps its cover with far fewer plants. 0: even.
  float thin = 0.0F;
  float grow = 1.0F;
  float fade = 0.0F;  // metres before `to` over which it thins out to nothing, so it has no edge
  // In drifts and groves: the share of the ground the layer stands on at all (1 everywhere), by a noise
  // `patch` metres across; and how much nearer its far edge it fills in whatever the noise (0 none),
  // so trees close the horizon
  float cover    = 1.0F;
  float patch    = 20.0F;
  float closeFar = 0.0F;
  float scaleMin = 0.8F;
  float scaleMax = 1.25F;
  float sink     = 0.0F;  // metres into the ground, times its scale: a trunk's foot on a slope
};
std::vector<Layer> layers(const Spec& spec);

// ---- Grass and flowers: tiles

// One kind of tile: a square `tile` metres across with `perSquareMetre` of its role's plants
struct Cover
{
  std::string role;  // grass | flower
  float       tile           = 2.0F;
  double      perSquareMetre = 1.0;  // the spec's density already in it
  float       scaleMin       = 0.8F;
  float       scaleMax       = 1.25F;
  float       sink           = 0.0F;  // metres into the ground: a flat tile on rolling ground would show its edge
  uint32_t    salt           = 0;     // which arrangement: two covers alike but for this are two patches
};

// The grass's three levels, the finest first: thick grass underfoot in 2 m tiles, thinner and larger
// in 4 m tiles, sparse tufts in 8 m tiles; each tile of one level is four of the next finer.
inline constexpr size_t kGrassLevels = 3;
std::array<Cover, kGrassLevels> grassCovers(const Spec& spec);
// How far from where one looks each level reaches: the last, where the grass ends
std::array<float, kGrassLevels> grassReach(const Spec& spec);
// The flowers' two tiles, a drift's middle and its edge, on the grass's middle grid; and how far they reach
std::array<Cover, 2> flowerCovers(const Spec& spec);
float                flowerReach(const Spec& spec);

// The plants of one tile of `cover`, about its centre on y = 0: per variant of the role, each
// plant's matrix in the tile
std::vector<std::vector<glm::mat4>> patch(const Spec& spec, const Cover& cover, uint32_t variants);

// A tile's cell on the grid of its level's size: it covers [x, x + 1) x [z, z + 1) tiles of it
struct Tile
{
  int32_t x     = 0;
  int32_t z     = 0;
  uint8_t level = 0;
  auto    operator<=>(const Tile&) const = default;
};

// The grass's tiles for where one looks from (`foci`, (x, z) each; none: the origin): every coarse
// cell within reach of a focus is one sparse tile, or, nearer, its four cells of the middle level,
// each of those one tile or its four thick ones. No ground is covered twice or left out. `was`, the
// tiles as last asked for: a cell that was split stays split until a little further than where it
// splits, so a view that hovers at a boundary changes nothing.
std::vector<Tile> grassTiles(const Spec& spec, std::span<const glm::vec2> foci, std::span<const Tile> was = {});
// The flowers' tiles: `level` 0 a drift's middle, 1 its edge (flowerCovers), on the middle grid
std::vector<Tile> flowerTiles(const Spec& spec, std::span<const glm::vec2> foci);

// Where a tile `size` metres across lies: its centre on the ground, sunk by `sink`, leaning as the
// ground does there and turned a quarter turn or more by its cell, so its neighbours don't repeat it
struct Place
{
  glm::vec3 position{0.0F};
  glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
};
Place tilePlace(const Spec& spec, float size, int32_t x, int32_t z, float sink);

// Where a layer's plants stand: per variant of the role (the kit's meshes for it, `variants` of them),
// each plant's world matrix: on the ground, turned about the vertical and scaled. `layer` is its index
// in layers(), so two layers of one spec don't share their places.
std::vector<std::vector<glm::mat4>> scatter(const Spec& spec, const Layer& layer, uint32_t layerIndex, uint32_t variants);

// How many plants scatter() places, about: for a notice before a heavy one is made
uint64_t expectedCount(const Spec& spec, const Layer& layer);

}  // namespace fv::backdrop
