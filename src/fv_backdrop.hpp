#pragma once

//-------------------------------------------------------------------------------------------------------
// The backdrop (the lighting's `backdrop` section, protocol 64; the app's Light tab › Backdrop): ground
// that rolls away from where the figures stand, planted with a kit's plants. Two kinds:
//
// - A meadow: grass and flowers underfoot, saplings, bushes and a few rocks further out, trees in
//   groves that close into a wood round it, and wooded hills beyond.
// - A forest: a glade a few metres across in a fir wood that runs to the horizon, with ferns in
//   drifts, sparse grass, bushes, mossy rocks, stumps and dead wood on a floor of needles.
//
// Everything here follows from the spec alone, so a scene file keeps only the spec: the ground's height
// (`height`), its mesh (`terrain`), and where each plant stands, by a seed. Two ways of standing:
//
// - Trees, saplings, bushes, rocks and dead wood each have a place (`layers`, `scatter`): the host
//   (fv_host_backdrop.cpp) draws a variant once with its places as instancing copies
//   (rs::MeshInstance::copies). They stand where the seed puts them, the trees out to the horizon,
//   and never change.
// - Grass, and what stands in drifts in it (a meadow's flowers, a forest's ferns), are tiles
//   (`grassCovers`, `drifts`, `patch`, `grassTiles`, `driftTiles`): a square of ground with its
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
  std::string kind    = "none";  // none | meadow | forest
  int         seed    = 1;       // which one: the hills and where everything stands
  double      size    = 60.0;    // metres across the thick grass round where one looks from; a meadow's saplings' reach
  double      density = 1.0;     // of every layer's own, 0.25 to 2
  double      relief  = 1.0;     // how hilly the ground is beyond the clearing, 0 flat to 2
  // How many metres of air a metre of the land counts for between the eye and what it sees (the
  // physical sky's aerial perspective): 1 as it is. The land is some hundreds of metres to its
  // horizon, and through that much clear air a far wood is as sharp and dark as a near one; a
  // landscape reads as one by the air over kilometres. Nothing is planted by it.
  double air = 6.0;
  bool   operator==(const Spec&) const = default;
};

// Whether two specs plant the same: alike in all but what only the air reads
inline bool samePlanting(Spec a, const Spec& b)
{
  a.air = b.air;
  return a == b;
}

inline bool planted(const Spec& s)
{
  return s.kind != "none";
}

// The ground stays flat (the floor's height) within this of the origin, where the figures stand, and
// rises into its hills over kClearingEdge more
inline constexpr float kClearing     = 5.0F;
inline constexpr float kClearingEdge = 25.0F;
// Where the planted land's hills have risen: the horizon of a camera on the ground
inline constexpr float kHorizon = 900.0F;
// Where the trees have thinned out to nothing: beyond, woods are a colour of the ground (`woods`)
inline constexpr float kTreesEnd = 2200.0F;
// Where the ground ends: ridges some hundreds of metres high by then, kilometres off, so a camera
// in the air has land to the haze and not a disc's edge
inline constexpr float kLandsEnd = 8000.0F;
// Over these distances from the origin what the camera sees of the land gives way to the planet the
// physical sky draws under its horizon (the renderer's landFadeFrom, landFadeTo), so the land has
// no rim: the two are lit by different sums of the same air, and meet at a line otherwise
inline constexpr float kLandFadeFrom = 0.6F * kLandsEnd;
inline constexpr float kLandFadeTo   = 0.96F * kLandsEnd;
// The ground's mesh: each ring this much further out than the last. A colour that changes over
// less than three rings is drawn as blocks, so the far ground's patches grow by three times this
inline constexpr float kMeshGrowth = 0.06F;
// Where a meadow's groves have closed into the wood round it
inline constexpr float kWoods = 260.0F;

// The ground's height over the floor at (x, z)
float height(const Spec& spec, float x, float z);

// How much of a place is wood that no tree stands for, 0 to 1: beyond where the far trees keep
// their cover (they thin, and each grows, until its growth is spent), where their layer's groves
// are. The ground is that much the colour of a wood seen from afar, so the few trees that still
// stand out there read as the edge of one.
float woods(const Spec& spec, float x, float z);

// The ground as a mesh: a disc, its rings fine underfoot and coarser with distance out to kLandsEnd, with
// smooth normals (the slope over a ring's width, so far ground is lit as its mesh is) and UVs that
// tile every `tile` metres
struct Terrain
{
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  // 0..1 a vertex, in drifts some 40 m across: where the ground is drier. A tiled texture is one
  // colour from far off; the kit's two tints mixed by this give the distance its patches
  std::vector<float>    shades;
  std::vector<float>    woods;  // `woods` a vertex
  std::vector<uint32_t> indices;
};
Terrain terrain(const Spec& spec, float tile);

// A layer of one role's plants
struct Layer
{
  std::string role;                  // the kit's role it plants: sapling, tree, shrub, rock, stump, log
  double      perSquareMetre = 1.0;  // where it stands thickest, before the spec's density
  float       from           = 0.0F; // none nearer the origin than this
  float       to             = 30.0F; // nor further
  // Past `thin` metres the layer thins as 1 / (1 + (d / thin)^2) and each plant grows by the root of
  // that, up to `grow` times, so distant ground keeps its cover with far fewer plants. 0: even.
  // `thinFrom`: the distance that counts from, so a layer that carries another on thins from where
  // that one ended.
  float thin     = 0.0F;
  float grow     = 1.0F;
  float thinFrom = 0.0F;
  float fade = 0.0F;  // metres before `to` over which it thins out to nothing, so it has no edge
  // In drifts and groves: the share of the ground the layer stands on at all (1 everywhere), by a noise
  // `patch` metres across; and how much nearer its far edge it fills in whatever the noise (0 none),
  // so trees close the horizon, or nearer its near edge, so a layer that carries a closed one on
  // opens out of it
  // `patchGrows`: past where it is `patch` already, the groves are this share of the distance across
  // (by the noise's octaves, a doubling at a time), so far groves are as large as the far ground's
  // mesh can draw (`woods`, kMeshGrowth).
  float cover      = 1.0F;
  float patch      = 20.0F;
  float closeFar   = 0.0F;
  float closeNear  = 0.0F;
  float closeSpan  = 0.0F;  // metres from the near edge over which closeNear opens; 0: three tenths of the layer
  float patchGrows = 0.0F;
  float scaleMin = 0.8F;
  float scaleMax = 1.25F;
  float sink     = 0.0F;  // metres into the ground, times its scale: a trunk's foot on a slope
};
std::vector<Layer> layers(const Spec& spec);

// ---- Grass and its drifts: tiles

// One kind of tile: a square `tile` metres across with `perSquareMetre` of its role's plants
struct Cover
{
  std::string role;  // grass | flower | fern
  float       tile           = 2.0F;
  double      perSquareMetre = 1.0;  // the spec's density already in it
  float       scaleMin       = 0.8F;
  float       scaleMax       = 1.25F;
  float       sink           = 0.0F;  // metres into the ground: a flat tile on rolling ground would show its edge
  // The share of the tile its plants stand on, in clumps a little under half the tile across, as
  // thick in them as keeps the tile's count: 1 evenly over it
  float clumps = 1.0F;
  uint32_t    salt           = 0;     // which arrangement: two covers alike but for this are two patches
};

// The grass's three levels, the finest first: thick grass underfoot in 2 m tiles, thinner and larger
// in 4 m tiles, sparse tufts in 8 m tiles; each tile of one level is four of the next finer.
inline constexpr size_t kGrassLevels = 3;
std::array<Cover, kGrassLevels> grassCovers(const Spec& spec);
// How far from where one looks each level reaches: the last, where the grass ends
std::array<float, kGrassLevels> grassReach(const Spec& spec);
// What stands in drifts in the grass, a meadow's flowers and a forest's ferns: two tiles, a drift's
// middle and its edge, on the grass's middle grid
struct Drifts
{
  std::array<Cover, 2> covers;         // the middle's, the edge's
  float                reach  = 0.0F;  // how far from where one looks they are laid
  float                across = 9.0F;  // metres across a drift, about
  float                middle = 0.2F;  // the share of the ground that is a drift's middle
  float                ground = 0.4F;  // and that is drift at all
};
Drifts drifts(const Spec& spec);

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
// The drifts' tiles: `level` 0 a drift's middle, 1 its edge (Drifts::covers), on the middle grid
std::vector<Tile> driftTiles(const Spec& spec, std::span<const glm::vec2> foci);

// Where a tile `size` metres across lies: its centre on the ground, sunk by `sink`, leaning as the
// ground does there and turned a quarter turn or more by its cell, so its neighbours don't repeat it
struct Place
{
  glm::vec3 position{0.0F};
  glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
};
Place tilePlace(const Spec& spec, float size, int32_t x, int32_t z, float sink);

// The kit's roles a spec plants, sorted: its layers', its grass's and its drifts'. What a kind doesn't
// plant needn't be held (a model's maps are a few megabytes each)
std::vector<std::string> roles(const Spec& spec);

// Where a layer's plants stand: per variant of the role (the kit's meshes for it, `variants` of them),
// each plant's world matrix: on the ground, turned about the vertical and scaled. `layer` is its index
// in layers(), so two layers of one spec don't share their places.
std::vector<std::vector<glm::mat4>> scatter(const Spec& spec, const Layer& layer, uint32_t layerIndex, uint32_t variants);

// How many plants scatter() places, about: for a notice before a heavy one is made
uint64_t expectedCount(const Spec& spec, const Layer& layer);

}  // namespace fv::backdrop
