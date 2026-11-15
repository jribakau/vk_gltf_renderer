#pragma once

//-------------------------------------------------------------------------------------------------------
// What is put on a planet by hand (fv_world.hpp): a planet's seed grows its continents, ranges and
// hills from noise, and an author who wants a particular world says where some of them are instead.
// A feature is a shape on the sphere, a point or an arc of a great circle with a width, and a kind:
//
//   land, sea    the continents' measure raised or lowered there: a coast where the author wants one,
//                with the noise's own wander along it
//   range        a mountain range along the arc, of the seed's own ridges: it stands by a coast or in
//                the sea as well (an arc of islands)
//   peak         one mountain, a cone with a crater if asked
//   basin        a hollow in the land: the base map fills a deep one with a lake
//   plain        level land: the mountains and the hills stood down
//
// The features shape the land before the planet's base map is baked, so the rain falls on them, wears
// them and gathers into rivers off them as it does on the seed's own land (after Génevaux et al.,
// "Terrain Modelling from Feature Primitives", 2015: a terrain as primitives with skeletons, combined;
// here each kind acts on one of the stages' own measures, so the noise still fills them in).
//
// A planet's description as a file is a seed's numbers and these (PlanetFile, parsePlanet).
//
// Vulkan-free, as the planet is.
//-------------------------------------------------------------------------------------------------------

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace fv::world {

enum class FeatureKind : uint8_t
{
  eLand = 0,
  eSea,
  eRange,
  ePeak,
  eBasin,
  ePlain,
};

struct Feature
{
  FeatureKind kind = FeatureKind::eLand;
  // Its line: a point where the two are the same, else the arc of the great circle between them (the
  // short way). Unit directions from the planet's centre, the planet's own axes (its poles on y).
  glm::dvec3 from{0.0, 1.0, 0.0};
  glm::dvec3 to{0.0, 1.0, 0.0};
  double     width = 10000.0;  // metres from its line to where nothing is left of it
  // land, sea: how far the continents' measure moves on its line (Sample::continent: about 1 is a
  // continent's middle). range: how high it stands against a seed's own (1: as high). plain: how
  // level, 0 to 1.
  double amount = 1.0;
  double height = 0.0;  // peak: metres at its top. basin: metres deep at its middle.
  double crater = 0.0;  // peak: its crater's width against the peak's own, 0 none
  bool   operator==(const Feature&) const = default;
};

struct Features
{
  std::vector<Feature> list;
  // Where the figures stand, when the author says: used if it is dry land (else the seed's own choice)
  std::optional<glm::dvec3> site;
  // How much of the seed's own mountain ranges stand, 0 (only the placed ones) to 1
  double wild = 1.0;
  bool   operator==(const Features&) const = default;
};

// What the features make of a point, each in the measure of the stage that takes it
struct PlacedLand
{
  float continent = 0.0F;  // added to Sample::continent
  float range     = 0.0F;  // how much of a placed range stands here: 0, to the highest `amount` on its line
  float level     = 0.0F;  // 0..1: how far the mountains and hills are stood down
  float elevation = 0.0F;  // metres the peaks add and the basins take
};

// The direction of a latitude and longitude, degrees: north toward +y, longitude 0 on +x, east toward +z
glm::dvec3 directionOf(double latitude, double longitude);

// How far `direction` is from a feature's line, metres over a sphere of `radius`
double distanceFrom(const Feature& f, const glm::dvec3& direction, double radius);

// How much of a feature is at a point: 1 on its line, 0 at its width from it, smooth between
double reach(const Feature& f, const glm::dvec3& direction, double radius);

// What all of them make of a point
PlacedLand placedAt(const Features& features, const glm::dvec3& direction, double radius);

// A planet as a file: a seed's numbers, where given, and what is placed on it. JSON:
//
//   { "name": "Home", "seed": 7, "radius": 100, "sea": 0.6, "relief": 1, "wild": 0.4,
//     "site": [3, 8],
//     "features": [ { "kind": "range", "from": [35, -20], "to": [-5, 25], "width": 18, "amount": 1 },
//                   { "kind": "peak", "at": [5, -35], "width": 9, "height": 3500, "crater": 0.2 } ] }
//
// Places are [latitude, longitude] in degrees, widths and the radius in kilometres, heights in metres.
struct PlanetFile
{
  std::string           name;
  std::optional<int>    seed;
  std::optional<double> radius;  // kilometres
  std::optional<double> sea;
  std::optional<double> relief;
  Features              features;
};

// Throws std::invalid_argument saying what is wrong and where
PlanetFile parsePlanet(const std::string& json);

}  // namespace fv::world
