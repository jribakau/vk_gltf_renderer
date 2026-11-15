#pragma once

//-------------------------------------------------------------------------------------------------------
// What people left on a planet (fv_world.hpp): places, the roads between them, and single things by
// the roads and alone in the land. The planet decides where and shapes its ground for them; what
// they are is data, a catalogue the app's tools write (scripts/blender/derelict_place.py --packs), so
// another kind of place is an entry there and nothing here.
//
// - A pack (Pack) is something laid out once: the models it is made of and where their copies stand
//   about its middle, how far it reaches, where a road joins it (its gates), and the ground it covers
//   (a grid a metre a cell: where no grass grows through, where nothing is planted). A housing estate
//   is a pack; so is a lamp post.
// - A kind (PlaceKind) is the packs that are alike and the rule for where one stands: the lands it
//   likes, the steepest ground, how far from the next, and either the road that is led to it or how
//   many stand alone in a square kilometre.
// - An Along rule stands a kind beside a road every so many metres.
//
// `settle` reads the planet and a catalogue and says what stands where (Places): sites, each a pack
// turned about the vertical of a point of the sphere, and the roads, curves as the rivers are. The
// same for a seed whatever the threads. `withPlaces` gives the planet a stage that makes its ground
// theirs: level under a place as a floor is (on the plane that touches the planet at its middle,
// like the figures' own clearing), cut and filled to a road's level across its width and eased into
// the land over its verges, a ford where a road meets a stream; and the ground's colour and layers
// under a road. Sample::built says what a point has of all this, which keeps the planet's plants and
// grass off it (fv_world_plants.cpp).
//
// Nothing here draws: the host holds the packs' models and stands the sites near the eye
// (fv_host_world_places.cpp).
//
// Vulkan-free; `settle` uses the job pool.
//-------------------------------------------------------------------------------------------------------

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tinygltf/json.hpp>  // nlohmann::json

#include "fv_world.hpp"
#include "fv_world_map.hpp"

namespace fv::world {

// ---- the catalogue -------------------------------------------------------------------------------------

// In a pack's own frame: its middle the origin, +y up, metres
struct Gate
{
  glm::dvec2 at{0.0};            // x, z
  glm::dvec2 heading{0.0, 1.0};  // the way out, unit
};

struct PieceCopy
{
  glm::vec3 at{0.0F};
  float     yaw = 0.0F;  // radians about +y
};

struct PackPiece
{
  std::string            file;            // a glTF, from the catalogue's folder
  bool                   growth = false;  // what grows over the place: shown from nearer, walked through
  std::vector<PieceCopy> copies;
};

struct Pack
{
  std::string            name;
  uint32_t               kind   = 0;     // of Catalogue::kinds
  double                 radius = 5.0;   // metres from its middle that it reaches
  float                  weight = 1.0F;  // its share among its kind's
  std::vector<Gate>      gates;
  std::vector<PackPiece> pieces;
  // Its ground: `n` cells of `cell` metres along x and z, centred; a row is one z
  float                cell = 1.0F;
  int                  n    = 0;
  std::vector<uint8_t> ground;
  // kGroundCovered | kGroundKept (fv_world.hpp) at a point of its frame; 0 outside its grid
  uint8_t groundAt(double x, double z) const;
};

struct RoadKind
{
  std::string name;
  double      width = 6.0;  // metres, the carriageway
  double      verge = 3.0;  // either side, over which its level eases into the land's
  glm::vec3   colour{0.17F};
  bool        earth = false;  // a track: drawn as the ground's sand is; else bare of the ground's layers
};

struct PlaceKind
{
  std::string           name;
  std::vector<uint32_t> packs;
  int                   road    = -1;      // of Catalogue::roads, led to its gate; -1: none is
  float                 weight  = 1.0F;    // its share of the places with a road
  float                 alone   = 0.0F;    // without one: so many a square kilometre of land it likes
  float                 slope   = 0.1F;    // the steepest ground it stands on, rise over run
  float                 spacing = 1000.0F; // metres to the next place with a road
  std::vector<std::pair<std::string, float>> biomes;  // how well it likes a land, by the biome's name; unnamed: not at all
};

struct Along
{
  uint32_t kind = 0, road = 0;
  double   every  = 100.0;  // metres between the spots along the road
  double   chance = 0.5;    // that a spot has one
  double   side   = 4.0;    // metres from the road's middle, either side
  double   turn   = 0.0;    // radians more than facing the road (its +x toward it)
  double   spin   = 0.0;    // and up to this either way
  double   within = 0.0;    // only this near a place that has a road; 0: anywhere
};

struct Catalogue
{
  std::vector<Pack>      packs;
  std::vector<PlaceKind> kinds;
  std::vector<RoadKind>  roads;
  std::vector<Along>     along;
};

// The catalogue of a places.json. Throws std::runtime_error saying what is wrong with it.
std::shared_ptr<const Catalogue> readCatalogue(const nlohmann::json& places);

// ---- what stands where ---------------------------------------------------------------------------------

struct Site
{
  glm::dvec3 direction{0.0, 1.0, 0.0};  // its middle, from the planet's centre
  double     elevation = 0.0;           // its floor's, metres over the sea (a levelled one's)
  double     heading   = 0.0;           // radians it is turned about the vertical there
  uint32_t   pack      = 0;             // of the catalogue's
  bool       level     = true;          // the ground is levelled for it; else it stands on the ground as it is
};

// A stretch of a road: a curve from `a` to `c` bent toward `b`, as a river's (RiverPiece), with its
// level along it
struct RoadPiece
{
  glm::dvec3 a{0.0}, b{0.0}, c{0.0};
  float      level[3] = {0.0F, 0.0F, 0.0F};
  float      half     = 3.0F;   // half the carriageway
  float      verge    = 3.0F;
  float      start    = 0.0F;   // metres along its way where it begins
  float      length   = 0.0F;
  uint32_t   way      = 0;      // which run of road between two junctions
  uint8_t    kind     = 0;      // of the catalogue's roads
};

struct Places
{
  explicit Places(uint32_t cells) : grid(cells) {}

  std::shared_ptr<const Catalogue> catalogue;
  std::vector<Site>                sites;
  uint32_t                         joined = 0;  // the first so many sites are the places a road is led to
  std::vector<RoadPiece>           roads;
  uint32_t                         ways = 0;

  // What is near a point: by cell of `grid`, the sites whose ground reaches into it and the road
  // pieces whose verges do. An item with kRoadItem set is a piece.
  static constexpr uint32_t kRoadItem = 0x80000000U;
  CubeGrid                 grid;
  std::vector<uint32_t>    first;  // a cell's items are items[first[c]] to items[first[c + 1]]
  std::vector<uint32_t>    items;
};

// A levelled site's ground eases into the land over this, past its reach
double padEdge(double radius);

// Settles a planet: `amount` 1 as the catalogue's rules say, more or fewer by it; none at 0 (null).
// `world` is the planet without places (make's). Null when stopped.
std::shared_ptr<const Places> settle(const World& world, std::shared_ptr<const Catalogue> catalogue, double amount = 1.0,
                                     const std::atomic<bool>* stop = nullptr);

// The planet with these places on its ground (null: with none). Nothing of it is made again.
World withPlaces(const World& world, std::shared_ptr<const Places> places);

// The stage `withPlaces` adds
void built(const World& world, const Point& point, Sample& sample);

// A site's frame in the scene: its pack's frame to the scene's, the pack's middle on the ground
glm::dmat4 siteFrame(const World& world, const Site& site);
// And a copy's place in its pack's frame
glm::mat4 copyPlace(const PieceCopy& copy);

// How far a pack shows from, metres
double seenFrom(const Pack& pack);
// The sites an eye at `eye` (the scene's frame) sees, sorted. `was`, the last answer: one that was
// seen stays so until the eye is kSplitHold times further, so an eye at a boundary changes nothing.
std::vector<uint32_t> sitesSeen(const World& world, const glm::dvec3& eye, std::span<const uint32_t> was = {});

}  // namespace fv::world
