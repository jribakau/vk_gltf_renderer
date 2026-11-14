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
// - The base map (BaseMap): the fields of the whole planet as a cube map, made once a seed, for what a
//   point alone cannot say (where the rivers run, what the rain wore away). Declared; nothing fills it
//   yet.
// - The mesh: a cube laid on the sphere, each face a quadtree (Quad). `select` says which quads draw
//   for an eye: a quad splits into four while the eye is within about twice its width of it, so the
//   ground underfoot is a metre a vertex and the far side of the planet a kilometre. `groundMesh` and
//   `seaMesh` make a quad's triangles, with skirts down its edges: where a quad meets a coarser one
//   their edges differ, and a ray would find the crack.
//
// The figures stand where the planet's `site` is: a level place on land, found by the seed. The planet
// is turned and set so that place is the origin with +y up (Frame); the ground is flat there for
// kClearing metres and eases into the land over kClearingEdge more.
//
// Vulkan-free. Doubles on the sphere: a metre underfoot and the planet's far side are told apart by
// more digits than a float has. Distances in metres; the mesh's points are in the frame of the
// backdrop's root, whose origin is the figures' floor.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include <glm/glm.hpp>

namespace fv::world {

struct Spec
{
  int    seed   = 1;        // which planet
  double radius = 20000.0;  // metres, the sea's level from its centre: 2 to 100 km
  double sea    = 0.55;     // the share of its surface under water, 0 to 0.9
  double relief = 1.0;      // how high its land stands, 0 flat to 2
  bool   operator==(const Spec&) const = default;
};

// The ground is level within this of where the figures stand, and eases into the land over the next
inline constexpr double kClearing     = 5.0;
inline constexpr double kClearingEdge = 25.0;

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
  int       biome       = 0;     // the row of World::biomes that counts most
  glm::vec3 colour{0.2F};        // what level ground reflects here, linear
};

// A biome: where it is, by the climate, and what its ground is. Where two meet they blend over
// `kBiomeSoft*`.
struct Biome
{
  std::string_view name;
  float            temperatureFrom = -100.0F, temperatureTo = 100.0F;  // degrees C
  float            moistureFrom = -1.0F, moistureTo = 2.0F;
  glm::vec3        colour{0.2F};
  float            rough = 1.0F;
};

struct World;
struct Stage
{
  std::string_view name;
  void (*apply)(const World&, const Point&, Sample&) = nullptr;
};

// The whole planet's fields as a cube map, `size` cells along a face's edge: each a face after another,
// row by row. Empty until something bakes it.
struct BaseMap
{
  uint32_t                          size = 0;
  std::array<std::vector<float>, 6> elevation;  // metres, added to the stages'
  std::array<std::vector<float>, 6> flow;       // how much water runs through a cell
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
};

// The planet of a spec: its sea level found, its stages listed, and its site chosen
World make(const Spec& spec);

// What the point `direction` of the planet is, for something `footprint` metres across
Sample sample(const World& world, const glm::dvec3& direction, double footprint);

// What the ground reflects where it slopes (rise over run): rock shows through as it steepens
glm::vec3 surfaceColour(const World& world, const Sample& s, float slope);

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
// How high a scene point is over the ground or the sea under it
double heightOver(const World& world, const glm::dvec3& point);

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
inline constexpr uint32_t kSeaCells    = 16;  // along a sea quad's: a sphere needs fewer
// A quad splits while the eye is within this many of its widths, and stays split a little further
inline constexpr double kSplit     = 2.0;
inline constexpr double kSplitHold = 1.2;

struct Selection
{
  std::vector<Quad> leaves;  // what draws: the sphere once over, sorted
  std::vector<Quad> split;   // what split to get there, sorted
};
// The quads for an eye at `eye` (the scene's frame). `was`, the last selection: a quad that was split
// stays so until the eye is kSplitHold times further than where it splits, so an eye that hovers at a
// boundary changes nothing.
Selection select(const World& world, const glm::dvec3& eye, const Selection* was = nullptr);

// ---- a quad's triangles ----------------------------------------------------------------------------

struct ChunkMesh
{
  glm::dvec3             centre{0.0};  // in the scene; the positions are about it
  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec4> colors;
  std::vector<uint32_t>  indices;
  uint32_t               surface = 0;  // the first vertices are the surface's; the rest its skirts'
  double                 lowest = 0.0, highest = 0.0;  // its elevations: under 0, the sea lies on it
};
ChunkMesh groundMesh(const World& world, const Quad& quad);
// The sea over a quad: the sphere at the sea's level
ChunkMesh seaMesh(const World& world, const Quad& quad);

}  // namespace fv::world
