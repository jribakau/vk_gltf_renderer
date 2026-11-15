#pragma once

//-------------------------------------------------------------------------------------------------------
// A small planet as the world (the backdrop's `planet` kind, fv_backdrop.hpp; the app's World tab): a
// sphere a few kilometres to a hundred across its radius, with land, sea and climate, that the figures
// stand on and a camera can leave until it sees the whole of it.
//
// Three parts that know nothing of each other:
//
// - Fields (`sample`): what a point of the sphere is: how high, how warm, how wet, what it reflects.
//   An ordered list of stages (World::stages), each reading and writing the sample: the continents and
//   the sea's floor, mountains, hills, the climate, the biome, the ground's fine relief, its colour.
//   A biome is a row of a table (World::biomes): another one is another row. Every stage takes the
//   size of what will draw the sample (`footprint`, metres) and adds nothing finer than that can
//   draw: detail under a mesh's vertices is drawn as triangles, not as detail.
// - The base map (BaseMap, fv_world_map.hpp): the fields of the whole planet as a cube map, baked once
//   a planet, for what a point alone cannot say: where the wind carries the sea's rain, what that rain
//   wore away, where it gathers into rivers and stands in lakes. The stages read it: the `base` stage
//   adds what the rain did to the land and finds the water near a point, `gullies` furrows the steep
//   slopes down their fall finer than the map's cells, `water` cuts the rivers' beds. A planet made
//   without one (Making::base) is the stages' alone.
// - The mesh: a cube laid on the sphere, each face a quadtree (Quad). `select` says which quads draw
//   for an eye: a quad splits into four while the eye is within about twice its width of it, so the
//   ground underfoot is a metre a vertex and the far side of the planet a kilometre. `groundMesh` and
//   `waterMesh` make a quad's triangles, with skirts down its edges: where a quad meets a coarser one
//   their edges differ, and a ray would find the crack.
//
// What grows on it is a biome's too (Biome::plants, grass, drift): fv_world_plants.hpp places it.
// What people left on it (places, the roads between them) is a catalogue's: fv_world_places.hpp
// settles it and adds a stage that makes the ground theirs (World::places, Sample::built).
//
// The ground's colour is the fields' (Sample::colour, a vertex's colour). Its grain is a map's, where
// the backdrop's kit has maps: a place is so much grass, so much of a wood's floor, of rock and of
// sand (surfaceLayers, a vertex's shares), and the renderer draws each hit with one of them
// (shaders/world_ground.h.slang).
//
// The figures stand where the planet's `site` is: a level place on land, found by the seed. The planet
// is turned and set so that place is the origin with +y up (Frame); the ground is flat there for
// kClearing metres, or as far as the scene's props reach (Spec::clearing), and eases into the land
// over kClearingEdge more. Flat as a floor is, not as the sphere is: on the plane of the scene's floor.
//
// Vulkan-free. Doubles on the sphere: a metre underfoot and the planet's far side are told apart by
// more digits than a float has. Distances in metres; the mesh's points are in the frame of the
// backdrop's root, whose origin is the figures' floor.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <compare>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include <glm/glm.hpp>

#include "fv_world_features.hpp"

namespace fv::world {

struct Spec
{
  int    seed   = 1;        // which planet
  double radius = 20000.0;  // metres, the sea's level from its centre: 2 to 100 km
  double sea    = 0.55;     // the share of its surface under water, 0 to 0.9
  double relief = 1.0;      // how high its land stands, 0 flat to 2
  // How far from where the figures stand the ground is level, metres: kClearing, or as far as what
  // stands in the scene reaches (the host's, from fv::backdrop::levelFor), up to mostLevel
  double clearing = 5.0;
  // What is put on it by hand (fv_world_features.hpp), shaping the land before the base map is baked;
  // none: the seed's own land. Two specs are the same planet with the same one (the pointer).
  std::shared_ptr<const Features> features;
  bool operator==(const Spec&) const = default;
};

// The ground is level within this of where the figures stand (Spec::clearing: further with props
// further out), and eases into the land over the next, or half the clearing when that is more
inline constexpr double kClearing     = 5.0;
inline constexpr double kClearingEdge = 25.0;
// The most a planet's ground is levelled for a scene: its site is found level and dry over a few
// hundred metres (or a tenth of a small planet's radius), and no further
inline double mostLevel(double radius)
{
  return std::max(kClearing, std::min(180.0, 0.07 * radius));
}

// What a ground is drawn with, where the backdrop's kit has its maps: grass (soil with grass on it),
// a wood's floor, rock, sand. A place is so much of each (surfaceLayers); what they leave of it is
// the ground's own colour and nothing more (snow). In the order of a vertex's shares, and of
// kGroundKinds, the kit's names for them.
enum class Soil : uint8_t
{
  eGrass = 0,
  eFloor,
  eRock,
  eSand,
  ePlain,
};
inline constexpr size_t                                      kGroundLayers = 4;
inline constexpr std::array<std::string_view, kGroundLayers> kGroundKinds{"meadow", "forest", "rock", "sand"};

// No water: a level under any ground
inline constexpr double kDry = -1.0e30;

// How many biomes a planet has at the most (a sample keeps each one's share)
inline constexpr size_t kMostBiomes = 8;

// The inland water by a point (the base map's rivers and lakes), as the `base` stage finds it and
// the stages after it use it. Levels in metres over the sea's.
struct Inland
{
  bool   river      = false;  // a river runs within reach
  double riverLevel = 0.0;    // its surface's level at its nearest point
  double riverHalf  = 0.0;    // half its width there
  double riverAway  = 0.0;    // metres from its middle line
  double bank       = 0.0;    // metres past its edge over which its banks ease into the land
  double lake       = 0.0;    // 0..1: how much of the ground round the point is a lake's
  double lakeLevel  = kDry;
  double cell       = 0.0;    // metres across a cell of the map that says so
  bool   any() const { return river || lake > 0.0; }
  // Metres to the nearest water's edge, negative in it; large with none near
  double shore() const;
};

// What is built at a point (fv_world_places.hpp): a road's surface, a place's ground
inline constexpr uint8_t kGroundCovered = 1;  // something lies on it: no grass grows through
inline constexpr uint8_t kGroundKept    = 2;  // nothing of the land's is planted on it (a yard, a verge)
inline constexpr uint8_t kGroundOpen    = 4;  // a place's own floor: no tree of the land's stands on it (a place stands in a clearing)
struct Built
{
  float     road = 0.0F;    // 0..1: how much of what draws the sample is a road's surface
  glm::vec3 colour{0.0F};   // that surface's
  bool      earth = false;  // a track: drawn as the ground's sand is; else bare of the ground's layers
  uint8_t   taken = 0;      // kGroundCovered | kGroundKept | kGroundOpen
  int32_t   site  = -1;     // the site whose pack says so
};

// A point of the sphere as a stage sees it
struct Point
{
  glm::dvec3 direction{0.0, 1.0, 0.0};  // from the planet's centre, unit, in the planet's own axes (its poles on y)
  double     footprint = 1.0;           // metres across what will draw the sample
};

struct Sample
{
  double    elevation   = 0.0;   // metres over the sea's level; under it, the sea's floor
  double    continent   = 0.0;   // how far inland, 0 at the coast, about 1 a continent's middle; negative out at sea
  float     mountain    = 0.0F;  // 0..1, how much of a mountain stands here
  float     temperature = 15.0F; // degrees C
  float     moisture    = 0.5F;  // 0 dry to 1 wet
  float     rough       = 1.0F;  // of the ground's fine relief, its biome's
  float     rock        = 0.0F;  // the share of bare rock whatever the slope
  float     sand        = 0.0F;  // the share of sand over its biome's ground: a beach, the sea's floor, a river's bed
  float     cliff       = 0.0F;  // 0..1: on a cliff's face (the `cliffs` stage), 1 at its middle
  float     steepened   = 1.0F;  // how many times as steep the tiers made the land's own slope here (by a shore)
  float     tiers       = 0.0F;  // how many tiers the land stands on here: whole on a tier, between on a face
  int       biome       = 0;     // the row of World::biomes that counts most
  std::array<float, kMostBiomes> share{};  // each row's share of the place, summing to 1 where any counts
  glm::vec3 colour{0.2F};        // what level ground reflects here, linear
  // Inland water. `water` is its surface's level here as its mesh draws it (waterLevel): over the
  // ground where a river or a lake lies, under it by their banks, kDry with none near.
  double water = kDry;
  float  shore = 1.0e9F;         // metres to the nearest inland water's edge, negative in it
  float  painted = 0.0F;         // 0..1: a river too narrow for what draws the sample, as a share of its colour
  Inland inland;                 // what the base map says of the water near
  PlacedLand placed;              // what is put here by hand (World::features), found by the first stage
  Built  built;                  // what is built here
};

// What stands on a biome's ground with a place of its own (fv_world_plants.hpp): one of the kit's roles
struct Growth
{
  std::string_view role;                   // of fv_world_plants.hpp's kStandingRoles
  double           perSquareMetre = 0.01;  // where it stands thickest, before the planet's density
  float            scaleMin = 0.8F, scaleMax = 1.25F;
  float            sink = 0.0F;            // metres into the ground, times its scale
  // In groves: the share of the ground it stands on at all, by a noise `patch` metres across
  float cover = 1.0F;
  float patch = 40.0F;
};

// A biome's ground cover, in tiles as a meadow's is (fv_backdrop.hpp)
enum class Grass : uint8_t
{
  eNone = 0,
  eMeadow,  // thick and green
  eDry,     // a third as thick, in clumps, yellowed
  eWood,    // tufts on a wood's floor
  eCold,    // sparse and brown
};

// A biome: where it is, by the climate, what its ground is and what grows on it. Where two meet
// they blend over `kBiomeSoft*`.
struct Biome
{
  std::string_view name;
  float            temperatureFrom = -100.0F, temperatureTo = 100.0F;  // degrees C
  float            moistureFrom = -1.0F, moistureTo = 2.0F;
  glm::vec3        colour{0.2F};
  float            rough = 1.0F;
  // What its ground is under its plants, where they are drawn (World::planted): `colour` is the
  // biome from afar, a wood's its canopy's, and near a wood's floor is needles. Not given: `colour`.
  glm::vec3           floor{-1.0F};
  std::vector<Growth> plants;
  Grass               grass = Grass::eNone;
  std::string_view    drift;  // what stands in drifts in its grass: flower | fern | "" none
  Soil                soil = Soil::ePlain;  // what its ground is drawn with
};

struct World;
struct Stage
{
  std::string_view name;
  void (*apply)(const World&, const Point&, Sample&) = nullptr;
};

// The whole planet's fields as a cube map: fv_world_map.hpp
struct BaseMap;
// What stands on it and the roads between: fv_world_places.hpp
struct Places;

// How a planet is made
struct Making
{
  bool     base = true;  // with its base map: the rain, what it wore away, rivers and lakes
  uint32_t map  = 0;     // cells along a face of that map; 0: by the planet's size (mapCells)
  // Set from another thread, the making stops at its next step and the planet returned is not whole:
  // for a planet nobody waits for any more
  const std::atomic<bool>* stop = nullptr;
};

// Where the planet stands in the scene: `site` (the planet's axes) is the origin, +y up
struct Frame
{
  glm::dvec3 site{0.0, 1.0, 0.0};
  double     siteElevation = 0.0;
  glm::dquat turn{1.0, 0.0, 0.0, 0.0};  // the planet's axes to the scene's
  glm::dvec3 centre{0.0};               // the planet's centre in the scene
};

struct World
{
  Spec                           spec;
  std::vector<Biome>             biomes;
  std::vector<Stage>             stages;
  std::shared_ptr<const BaseMap> base;
  Frame                          frame;
  // The continents' noise level the sea stands at, and how much that noise varies: Sample::continent
  // is the noise over that level in these
  double seaLevel  = 0.5;
  double seaSpread = 0.1;
  // What the planet reflects seen whole, linear: the sky's own ground under the air
  glm::vec3 albedo{0.1F};
  // Its plants are drawn (fv_world_plants.hpp): the ground near is what lies under them (Biome::floor)
  bool planted = false;
  // What people left on it, with the stage that makes the ground theirs (withPlaces); none: null
  std::shared_ptr<const Places> places;
};

// The planet of a spec: its sea level found, its base map baked, its stages listed, and its site
// chosen. The baking is the long part: a second or two for a planet 20 km in radius, on the pool.
World make(const Spec& spec, const Making& making = {});
// The same planet, level another distance round its site: nothing of it is made again but what is
// asked of its ground from then on
World withClearing(const World& world, double clearing);

// What the point `direction` of the planet is, for something `footprint` metres across
Sample sample(const World& world, const glm::dvec3& direction, double footprint);
// The same with one stage left out (World::stages, by name): the land as it would be without it.
// What a road's level follows, which is cut and filled through the gullies and not over each one.
Sample sampleWithout(const World& world, const glm::dvec3& direction, double footprint, std::string_view stage);

// A smooth noise over the planet, 0 to 1 and a half on average, its features `across` metres across:
// what stands in groves stands where it is high. `channel`: which noise.
double noise(const World& world, const glm::dvec3& direction, double across, uint32_t channel);

// How warm a place of the planet is at `elevation` (metres over the sea), degrees C: by its latitude
// and its height, before what the weather adds
double temperatureAt(const World& world, const glm::dvec3& direction, double elevation);

// What the ground reflects where it slopes (rise over run): rock shows through as it steepens
glm::vec3 surfaceColour(const World& world, const Sample& s, float slope);

// And what it is drawn with there: its shares of the ground's layers (kGroundKinds' order), summing
// to 1 at the most. Rock where surfaceColour has rock, sand where the sample has it, the biomes' soils
// under them.
std::array<float, kGroundLayers> surfaceLayers(const World& world, const Sample& s, float slope);

// What the sea's water is: how far light goes in it before a colour is left, per channel
inline const glm::vec3 kSeaColour{0.25F, 0.72F, 0.80F};
inline constexpr float     kSeaColourDistance = 12.0F;  // metres

// ---- the cube on the sphere ------------------------------------------------------------------------

// A square of a face's quadtree: at `level` a face is 2^level squares along its edge
struct Quad
{
  uint8_t  face  = 0;  // +x, -x, +y, -y, +z, -z
  uint8_t  level = 0;
  uint32_t x     = 0;
  uint32_t y     = 0;
  auto     operator<=>(const Quad&) const = default;
};

// The direction of a face's point (u, v), each -1 to 1 across it. Even in angle, so a face's
// squares are near enough alike in size.
glm::dvec3 direction(uint8_t face, double u, double v);
// And back: the face a direction is on, and where
struct FacePoint
{
  uint8_t face = 0;
  double  u = 0.0, v = 0.0;
};
FacePoint locate(const glm::dvec3& direction);
// Where a direction is on `face`'s own grid, carried on past its edges (u or v beyond -1 to 1 for a
// direction on a face beside it); face 255 for one behind it
FacePoint onFace(uint8_t face, const glm::dvec3& direction);
bool      contains(const Quad& quad, const glm::dvec3& direction);
// A quad's four children
std::array<Quad, 4> children(const Quad& quad);
// Metres across a quad, about
double width(const World& world, const Quad& quad);
// The level whose quads' vertices are a metre apart or less
uint8_t finestLevel(const World& world);

// ---- the planet in the scene -----------------------------------------------------------------------

// Where the planet's point `direction`, `elevation` over its sea, is in the scene
glm::dvec3 toScene(const World& world, const glm::dvec3& direction, double elevation);
// And back: the planet's direction of a scene point, and its distance from the centre
glm::dvec3 toPlanet(const World& world, const glm::dvec3& point, double* fromCentre = nullptr);
// How high a scene point is over the ground or the water under it
double heightOver(const World& world, const glm::dvec3& point);
// The water's surface over the planet's point `direction`, metres over the sea's level, for something
// `footprint` metres across: the sea's own (0), a lake's or a river's where one lies. By their banks
// the surface goes down under the ground, which hides it.
double waterLevel(const World& world, const glm::dvec3& direction, double footprint);

// The lake or the river a scene point is in: how high its surface stands over the sea's level,
// metres; kDry when the point is over dry ground, over the sea, or over water it is not under.
// What a camera's paths start in (shaders/world_sea.h.slang: no sphere says where a lake is).
double inlandWaterOver(const World& world, const glm::dvec3& point);

// A camera's "up" at `eye` (the scene's frame). By the site it is the scene's +y, which the figures
// stand along; away from it the planet's own vertical there, so the horizon stays level all the way
// round. Eased from one to the other between these shares of the radius from the origin.
inline constexpr double kUpOwnFrom = 0.01, kUpOwnTo = 0.1;
glm::dvec3 viewUp(const World& world, const glm::dvec3& eye);
// How fast a flight goes at `height` over the ground, scene units a second: its own height in two
// seconds, and no slower than `slowest`, a flight's speed among the figures.
inline constexpr double kFlyHeightsASecond = 0.5;
inline double flySpeed(double height, double slowest)
{
  return std::max(slowest, kFlyHeightsASecond * std::abs(height));
}
// The far plane a view from `eye` needs: past the planet's far side.
double farPlane(const World& world, const glm::dvec3& eye);

// ---- which quads draw ------------------------------------------------------------------------------

inline constexpr uint32_t kGroundCells = 64;  // along a ground quad's edge
// A quad splits while the eye is within this many of its widths, and stays split a little further
inline constexpr double kSplit     = 3.0;
inline constexpr double kSplitHold = 1.2;

struct Selection
{
  std::vector<Quad> leaves;  // what draws: the sphere once over, sorted
  std::vector<Quad> split;   // what split to get there, sorted
};
// An eye that moves is led: its quads are chosen for where it is and for where it will be kLead
// seconds on, the nearer of the two deciding, so a quad is asked for before the eye is at its
// distance. And none is narrower than the eye passes in kPassing seconds: it would be behind a ship
// before it drew. (Short, so that it holds only for a ship low and very fast: when it slows, every
// quad held back is wanted at once, near it.)
inline constexpr double kLead    = 0.5;
inline constexpr double kPassing = 0.2;
// The quads for an eye at `eye` (the scene's frame), going at `velocity` (scene units a second).
// `was`, the last selection: a quad that was split stays so until the eye is kSplitHold times further
// than where it splits, so an eye that hovers at a boundary changes nothing.
Selection select(const World& world, const glm::dvec3& eye, const Selection* was = nullptr, const glm::dvec3& velocity = glm::dvec3(0.0));

// ---- a quad's triangles ----------------------------------------------------------------------------

struct ChunkMesh
{
  glm::dvec3             centre{0.0};  // in the scene; the positions are about it
  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec4> colors;
  // The ground's: each vertex's shares of the layers (surfaceLayers), a byte a layer, the first the
  // lowest. A water mesh has none.
  std::vector<uint32_t> layers;
  std::vector<uint32_t> indices;
  uint32_t               surface = 0;  // the first vertices are the surface's; the rest its skirts'
  double                 lowest = 0.0, highest = 0.0;  // its elevations: under 0, the sea lies on it
  bool                   inland = false;  // a river or a lake lies on it
};
ChunkMesh groundMesh(const World& world, const Quad& quad);
// The inland water over a quad, `cells` along its edge: the lakes' and rivers' surfaces where they
// lie (waterLevel), cut at the ground's own points when it has as many cells as the ground. The sea
// has no mesh: it is the sphere at its level, which the renderer meets by itself, waves and all
// (shaders/world_sea.h.slang). Where a river runs out into it the mesh goes down under it
// (kSeaUnder), and what would lie at the sea's level is left out: no triangles at all on a quad
// without inland water (ChunkMesh::inland).
inline constexpr double kSeaMeets = 0.05;  // water this little over the sea is the sea
inline constexpr double kSeaUnder = 0.6;   // under its level, and under its swell's troughs
ChunkMesh waterMesh(const World& world, const Quad& quad, uint32_t cells = kGroundCells);

}  // namespace fv::world
