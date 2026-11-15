// What is put on a planet by hand (src/fv_world_features.hpp): a feature's reach from its point or
// its arc, each kind in the measure of the stage that takes it, the land they make of a seed's planet,
// a planet file read with its mistakes named, and what it says of where people lived: a region's
// amount and kinds, all of them in its middle and easing to nothing at its edge.

#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "../src/fv_world.hpp"

namespace {

namespace W = fv::world;

constexpr double kRadius = 100000.0;

W::Feature at(W::FeatureKind kind, double lat, double lon, double widthKm)
{
  W::Feature f;
  f.kind  = kind;
  f.from  = f.to = W::directionOf(lat, lon);
  f.width = widthKm * 1000.0;
  return f;
}

W::Feature along(W::FeatureKind kind, double lat0, double lon0, double lat1, double lon1, double widthKm)
{
  W::Feature f = at(kind, lat0, lon0, widthKm);
  f.to         = W::directionOf(lat1, lon1);
  return f;
}

W::Spec with(W::Spec spec, std::shared_ptr<const W::Features> features)
{
  spec.features = std::move(features);
  return spec;
}

std::string whyNot(const std::string& json)
{
  try
  {
    W::parsePlanet(json);
  }
  catch(const std::invalid_argument& e)
  {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(WorldFeatures, APlaceIsALatitudeAndALongitude)
{
  EXPECT_NEAR(glm::distance(W::directionOf(90.0, 17.0), glm::dvec3(0.0, 1.0, 0.0)), 0.0, 1e-12);
  EXPECT_NEAR(glm::distance(W::directionOf(0.0, 0.0), glm::dvec3(1.0, 0.0, 0.0)), 0.0, 1e-12);
  EXPECT_NEAR(glm::distance(W::directionOf(0.0, 90.0), glm::dvec3(0.0, 0.0, 1.0)), 0.0, 1e-12);
  EXPECT_NEAR(glm::length(W::directionOf(-37.0, 211.0)), 1.0, 1e-12);
}

// From a point a feature reaches a circle; from an arc, a band along it with round ends. Distances are
// the sphere's own, to the metre at any size.
TEST(WorldFeatures, AFeatureReachesFromItsPointOrItsArc)
{
  const W::Feature point = at(W::FeatureKind::eLand, 0.0, 0.0, 20.0);
  EXPECT_NEAR(W::distanceFrom(point, W::directionOf(0.0, 0.0), kRadius), 0.0, 1e-6);
  // a degree of a 100 km planet is 1745.33 m
  EXPECT_NEAR(W::distanceFrom(point, W::directionOf(0.0, 1.0), kRadius), 1745.329, 0.01);
  EXPECT_NEAR(W::distanceFrom(point, W::directionOf(1.0, 0.0), kRadius), 1745.329, 0.01);
  EXPECT_EQ(W::reach(point, W::directionOf(0.0, 0.0), kRadius), 1.0);
  EXPECT_NEAR(W::reach(point, W::directionOf(0.0, 10.0 / 1.745329), kRadius), 0.5, 1e-3);  // half way out
  EXPECT_EQ(W::reach(point, W::directionOf(0.0, 12.0), kRadius), 0.0);

  // along the equator from 0 to 30 degrees: beside its middle the distance is the latitude's, past an
  // end the end's
  const W::Feature arc = along(W::FeatureKind::eRange, 0.0, 0.0, 0.0, 30.0, 20.0);
  EXPECT_NEAR(W::distanceFrom(arc, W::directionOf(0.0, 15.0), kRadius), 0.0, 1e-6);
  EXPECT_NEAR(W::distanceFrom(arc, W::directionOf(3.0, 15.0), kRadius), 3.0 * 1745.329, 0.05);
  EXPECT_NEAR(W::distanceFrom(arc, W::directionOf(-3.0, 29.0), kRadius), 3.0 * 1745.329, 0.05);
  EXPECT_NEAR(W::distanceFrom(arc, W::directionOf(0.0, 34.0), kRadius), 4.0 * 1745.329, 0.05);
  EXPECT_NEAR(W::distanceFrom(arc, W::directionOf(0.0, -2.0), kRadius), 2.0 * 1745.329, 0.05);
  // the far side of the great circle is far, not on the line
  EXPECT_GT(W::distanceFrom(arc, W::directionOf(0.0, 195.0), kRadius), 0.4 * 3.14159 * kRadius);

  // and no step anywhere along a walk across it and round its end
  double was = W::reach(arc, W::directionOf(-15.0, -10.0), kRadius);
  for(int i = 1; i <= 2000; i++)
  {
    const double t   = double(i) / 2000.0;
    const double now = W::reach(arc, W::directionOf(-15.0 + 30.0 * t, -10.0 + 50.0 * t), kRadius);
    EXPECT_LT(std::abs(now - was), 0.01) << i;
    was = now;
  }
}

TEST(WorldFeatures, EachKindActsInItsOwnMeasure)
{
  W::Features features;
  W::Feature  land = at(W::FeatureKind::eLand, 10.0, 0.0, 30.0);
  land.amount      = 2.0;
  W::Feature sea   = at(W::FeatureKind::eSea, 10.0, 90.0, 30.0);
  sea.amount       = 1.5;
  W::Feature range = along(W::FeatureKind::eRange, -20.0, 0.0, -20.0, 40.0, 15.0);
  range.amount     = 0.8;
  W::Feature peak  = at(W::FeatureKind::ePeak, 50.0, 180.0, 8.0);
  peak.height      = 3000.0;
  peak.crater      = 0.2;
  W::Feature basin = at(W::FeatureKind::eBasin, -50.0, 180.0, 12.0);
  basin.height     = 400.0;
  W::Feature plain = at(W::FeatureKind::ePlain, 0.0, 270.0, 25.0);
  plain.amount     = 0.9;
  features.list    = {land, sea, range, peak, basin, plain};

  EXPECT_FLOAT_EQ(W::placedAt(features, W::directionOf(10.0, 0.0), kRadius).continent, 2.0F);
  EXPECT_FLOAT_EQ(W::placedAt(features, W::directionOf(10.0, 90.0), kRadius).continent, -1.5F);
  // a range is whole over its middle and nothing past its width
  EXPECT_FLOAT_EQ(W::placedAt(features, W::directionOf(-20.0, 20.0), kRadius).range, 0.8F);
  EXPECT_FLOAT_EQ(W::placedAt(features, W::directionOf(-18.5, 20.0), kRadius).range, 0.8F);
  EXPECT_GT(W::placedAt(features, W::directionOf(-14.0, 20.0), kRadius).range, 0.0F);
  EXPECT_LT(W::placedAt(features, W::directionOf(-14.0, 20.0), kRadius).range, 0.4F);
  EXPECT_EQ(W::placedAt(features, W::directionOf(-10.0, 20.0), kRadius).range, 0.0F);
  // a peak's rim stands over its crater's floor, and its top is under the height it would have whole
  const float floor = W::placedAt(features, W::directionOf(50.0, 180.0), kRadius).elevation;
  const float rim   = W::placedAt(features, W::directionOf(50.0 + 0.2 * 8.0 / 1.745329, 180.0), kRadius).elevation;
  EXPECT_GT(rim, floor + 300.0F);
  EXPECT_LT(rim, 3000.0F);
  EXPECT_GT(rim, 1500.0F);
  EXPECT_FLOAT_EQ(W::placedAt(features, W::directionOf(-50.0, 180.0), kRadius).elevation, -400.0F);
  EXPECT_FLOAT_EQ(W::placedAt(features, W::directionOf(0.0, 270.0), kRadius).level, 0.9F);
  // and away from all of them, nothing
  const W::PlacedLand none = W::placedAt(features, W::directionOf(-80.0, 100.0), kRadius);
  EXPECT_EQ(none.continent, 0.0F);
  EXPECT_EQ(none.range, 0.0F);
  EXPECT_EQ(none.level, 0.0F);
  EXPECT_EQ(none.elevation, 0.0F);
}

// On a seed's planet: land where it is placed in the sea, sea where it is placed on land, a range that
// stands over the land round it, level land on a plain, the figures where the author stands them, and
// the same planet everywhere else as with nothing placed.
TEST(WorldFeatures, ThePlanetIsTheSeedsWithWhatIsPlacedOnIt)
{
  const W::Spec  spec{.seed = 3, .radius = 40000.0};
  const W::World wild = W::make(spec, {.base = false});

  // a place well out at sea and one well inland, of the seed's own
  glm::dvec3 atSea(0.0), inland(0.0);
  double     lowest = 0.0, highest = 0.0;
  for(int i = 0; i < 4000; i++)
  {
    const double     y   = 0.6 - 1.2 * (double(i) + 0.5) / 4000.0;
    const double     r   = std::sqrt(1.0 - y * y);
    const glm::dvec3 d(r * std::cos(double(i) * 2.399963), y, r * std::sin(double(i) * 2.399963));
    const double     c = W::sample(wild, d, 500.0).continent;
    if(c < lowest)
    {
      lowest = c;
      atSea  = d;
    }
    if(c > highest)
    {
      highest = c;
      inland  = d;
    }
  }
  ASSERT_LT(lowest, -0.8);
  ASSERT_GT(highest, 0.8);

  auto features = std::make_shared<W::Features>();
  W::Feature land;
  land.kind   = W::FeatureKind::eLand;
  land.from   = land.to = atSea;
  land.width  = 8000.0;
  land.amount = 4.0;
  W::Feature sea = land;
  sea.kind       = W::FeatureKind::eSea;
  sea.from       = sea.to = inland;
  features->list = {land, sea};
  features->site = atSea;
  const W::World placed = W::make(with(spec, features), {.base = false});

  EXPECT_LT(W::sample(wild, atSea, 500.0).elevation, 0.0);
  EXPECT_GT(W::sample(placed, atSea, 500.0).elevation, 5.0);
  EXPECT_GT(W::sample(wild, inland, 500.0).elevation, 0.0);
  EXPECT_LT(W::sample(placed, inland, 500.0).elevation, 0.0);
  // the figures stand where the author put them, now that it is land
  EXPECT_NEAR(glm::distance(placed.frame.site, atSea), 0.0, 1e-12);
  EXPECT_GT(glm::distance(wild.frame.site, atSea), 0.01);
  // a site that is under the sea is not taken
  auto drowned  = std::make_shared<W::Features>();
  drowned->site = atSea;
  EXPECT_NEAR(glm::distance(W::make(with(spec, drowned), {.base = false}).frame.site, wild.frame.site), 0.0, 1e-12);

  // away from both the planet is the seed's, to the bit
  int same = 0;
  for(int i = 0; i < 4000; i++)
  {
    const double     y = 1.0 - 2.0 * (double(i) + 0.5) / 4000.0;
    const double     r = std::sqrt(1.0 - y * y);
    const glm::dvec3 d(r * std::cos(double(i) * 2.399963), y, r * std::sin(double(i) * 2.399963));
    if(W::reach(land, d, spec.radius) > 0.0 || W::reach(sea, d, spec.radius) > 0.0)
      continue;
    EXPECT_EQ(W::sample(placed, d, 200.0).elevation, W::sample(wild, d, 200.0).elevation) << i;
    same++;
  }
  EXPECT_GT(same, 3000);

  // A range across the plain's own land, and a plain: the range's crest stands hundreds of metres over
  // the land beside it, and the plain's land is nearer level than the seed's there
  auto shaped = std::make_shared<W::Features>();
  glm::dvec3 side = glm::normalize(glm::cross(inland, glm::dvec3(0.0, 1.0, 0.0)));
  W::Feature range;
  range.kind   = W::FeatureKind::eRange;
  range.from   = glm::normalize(inland - side * 0.25);
  range.to     = glm::normalize(inland + side * 0.25);
  range.width  = 4000.0;
  range.amount = 1.0;
  shaped->list = {range};
  shaped->wild = 0.0;
  const W::World ranged = W::make(with(spec, shaped), {.base = false});
  double         crest = 0.0, beside = 0.0;
  for(int i = 0; i <= 200; i++)
  {
    const glm::dvec3 on  = glm::normalize(inland + side * (-0.2 + 0.4 * double(i) / 200.0));
    const glm::dvec3 off = glm::normalize(on + glm::normalize(glm::cross(on, side)) * (6000.0 / spec.radius));
    crest                = std::max(crest, double(W::sample(ranged, on, 300.0).mountain));
    beside               = std::max(beside, double(W::sample(ranged, off, 300.0).mountain));
  }
  EXPECT_GT(crest, 0.3);
  EXPECT_EQ(beside, 0.0) << "with the seed's own ranges stood down, none stands beside the placed one";

  auto levelled = std::make_shared<W::Features>();
  W::Feature plain;
  plain.kind     = W::FeatureKind::ePlain;
  plain.from     = plain.to = inland;
  plain.width    = 12000.0;
  plain.amount   = 1.0;
  levelled->list = {plain};
  const W::World flat = W::make(with(spec, levelled), {.base = false});
  double         roughWild = 0.0, roughFlat = 0.0;
  for(int i = 0; i < 64; i++)
  {
    const double     a = double(i) * 0.39269908;
    const glm::dvec3 d = glm::normalize(inland + (side * std::cos(a) + glm::cross(inland, side) * std::sin(a)) * (double(1 + i % 4) * 700.0 / spec.radius));
    roughWild += std::abs(W::sample(wild, d, 100.0).elevation - W::sample(wild, inland, 100.0).elevation);
    roughFlat += std::abs(W::sample(flat, d, 100.0).elevation - W::sample(flat, inland, 100.0).elevation);
  }
  EXPECT_LT(roughFlat, 0.5 * roughWild);
}

TEST(WorldFeatures, APlanetFileIsReadAndItsMistakesNamed)
{
  const W::PlanetFile p = W::parsePlanet(R"({
    "name": "Home", "seed": 7, "radius": 100, "sea": 0.6, "relief": 1.2, "wild": 0.4, "site": [3, 8],
    "features": [
      {"kind": "range", "name": "the spine", "from": [35, -20], "to": [-5, 25], "width": 18, "amount": 1},
      {"kind": "peak", "at": [5, -35], "width": 9, "height": 3500, "crater": 0.2},
      {"kind": "basin", "at": [22, 12], "width": 14, "depth": 250},
      {"kind": "land", "at": [15, 0], "width": 110, "amount": 2.5}
    ]})");
  EXPECT_EQ(p.name, "Home");
  EXPECT_EQ(p.seed, 7);
  EXPECT_EQ(p.radius, 100.0);
  EXPECT_EQ(p.sea, 0.6);
  EXPECT_EQ(p.relief, 1.2);
  EXPECT_EQ(p.features.wild, 0.4);
  ASSERT_TRUE(p.features.site.has_value());
  EXPECT_NEAR(glm::distance(*p.features.site, W::directionOf(3.0, 8.0)), 0.0, 1e-12);
  ASSERT_EQ(p.features.list.size(), 4U);
  EXPECT_EQ(p.features.list[0].kind, W::FeatureKind::eRange);
  EXPECT_EQ(p.features.list[0].name, "the spine");
  EXPECT_EQ(p.features.list[1].name, "");
  EXPECT_EQ(p.features.list[0].width, 18000.0);
  EXPECT_NEAR(glm::distance(p.features.list[0].to, W::directionOf(-5.0, 25.0)), 0.0, 1e-12);
  EXPECT_EQ(p.features.list[1].height, 3500.0);
  EXPECT_EQ(p.features.list[1].crater, 0.2);
  EXPECT_EQ(p.features.list[1].from, p.features.list[1].to);
  EXPECT_EQ(p.features.list[2].height, 250.0);
  EXPECT_EQ(p.features.list[3].amount, 2.5);

  // nothing given is nothing said: the backdrop's own numbers stand
  const W::PlanetFile bare = W::parsePlanet("{}");
  EXPECT_FALSE(bare.seed.has_value());
  EXPECT_FALSE(bare.radius.has_value());
  EXPECT_TRUE(bare.features.list.empty());
  EXPECT_EQ(bare.features.wild, 1.0);

  EXPECT_NE(whyNot("{"), "");
  EXPECT_NE(whyNot("[]").find("an object"), std::string::npos);
  EXPECT_NE(whyNot(R"({"moons": 2})").find("moons"), std::string::npos);
  EXPECT_NE(whyNot(R"({"radius": 500})").find("radius"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "fjord", "at": [0, 0], "width": 3}]})").find("features[0].kind"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "peak", "at": [0, 0], "width": 3}]})").find("height"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "land", "at": [0, 0]}]})").find("width"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "land", "at": [0, 0], "width": 3, "depth": 4}]})").find("depth"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "range", "from": [0, 0], "width": 3}]})").find("from"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "range", "from": [0, 0], "to": [0, 180], "width": 3}]})").find("opposite"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "land", "at": [95, 0], "width": 3}]})").find("latitude"), std::string::npos);
  EXPECT_NE(whyNot(R"({"features": [{"kind": "land", "name": 3, "at": [0, 0], "width": 3}]})").find("features[0].name"), std::string::npos);
}

// ---- where people lived ------------------------------------------------------------------------------

TEST(WorldFeatures, APlanetFileSaysWherePeopleLived)
{
  const W::PlanetFile p = W::parsePlanet(R"({
    "places": [
      {"kind": "region", "name": "the farmland", "at": [3, 9], "width": 15, "amount": 3, "kinds": {"farm": 3, "hamlet": 1}},
      {"kind": "region", "from": [50, 0], "to": [40, 30], "width": 30, "amount": 0},
      {"kind": "place", "name": "Kelda", "at": [4, 10], "place": "hamlet"}
    ]})");
  ASSERT_EQ(p.settlement.regions.size(), 2U);
  const W::PlacesRegion& farm = p.settlement.regions[0];
  EXPECT_EQ(farm.name, "the farmland");
  EXPECT_EQ(farm.shape.width, 15000.0);
  EXPECT_EQ(farm.shape.from, farm.shape.to);
  EXPECT_NEAR(glm::distance(farm.shape.from, W::directionOf(3.0, 9.0)), 0.0, 1e-12);
  EXPECT_EQ(farm.amount, 3.0);
  ASSERT_EQ(farm.kinds.size(), 2U);
  for(const auto& [name, much] : farm.kinds)
    EXPECT_EQ(much, name == "farm" ? 3.0 : 1.0) << name;
  const W::PlacesRegion& empty = p.settlement.regions[1];
  EXPECT_EQ(empty.amount, 0.0);
  EXPECT_TRUE(empty.kinds.empty());
  EXPECT_NEAR(glm::distance(empty.shape.to, W::directionOf(40.0, 30.0)), 0.0, 1e-12);
  ASSERT_EQ(p.settlement.named.size(), 1U);
  EXPECT_EQ(p.settlement.named[0].name, "Kelda");
  EXPECT_EQ(p.settlement.named[0].kind, "hamlet");
  EXPECT_NEAR(glm::distance(p.settlement.named[0].at, W::directionOf(4.0, 10.0)), 0.0, 1e-12);
  // (where people lived shapes no land: the features are as they were)
  EXPECT_TRUE(p.features.list.empty());
  EXPECT_TRUE(W::parsePlanet("{}").settlement.empty());

  EXPECT_NE(whyNot(R"({"places": {}})").find("places"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "town", "at": [0, 0], "width": 3}]})").find("places[0].kind"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "region", "at": [0, 0]}]})").find("width"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "region", "width": 3}]})").find("at"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "region", "at": [0, 0], "width": 3, "amount": 20}]})").find("places[0].amount"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "region", "at": [0, 0], "width": 3, "height": 2}]})").find("height"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "region", "at": [0, 0], "width": 3, "kinds": []}]})").find("kinds"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "region", "at": [0, 0], "width": 3, "kinds": {"farm": -1}}]})").find("kinds.farm"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "place", "at": [0, 0], "place": "farm"}]})").find("name"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "place", "name": "", "at": [0, 0], "place": "farm"}]})").find("name"), std::string::npos);
  EXPECT_NE(whyNot(R"({"places": [{"kind": "place", "name": "X", "at": [0, 0], "place": "farm", "width": 2}]})").find("width"), std::string::npos);
}

TEST(WorldFeatures, ARegionIsAllThereInItsMiddleAndEasesToNothingAtItsEdge)
{
  constexpr double kDegree = 180.0 / 3.14159265358979323846;
  // a point `km` east of the region's middle
  const auto east = [](double km) { return W::directionOf(0.0, km * 1000.0 / kRadius * kDegree); };
  W::PlacesRegion more;
  more.shape  = at(W::FeatureKind::eLand, 0.0, 0.0, 10.0);
  more.amount = 3.0;
  W::Settlement s;
  s.regions = {more};
  EXPECT_DOUBLE_EQ(W::settledAmount(s, east(0.0), kRadius), 3.0);
  EXPECT_DOUBLE_EQ(W::settledAmount(s, east(10.0 * (1.0 - W::kRegionEdge) - 0.1), kRadius), 3.0);
  double was = 3.0;
  for(double km = 7.5; km < 10.0; km += 0.5)
  {
    const double now = W::settledAmount(s, east(km), kRadius);
    EXPECT_LT(now, was) << km;
    EXPECT_GT(now, 1.0) << km;
    was = now;
  }
  EXPECT_DOUBLE_EQ(W::settledAmount(s, east(10.1), kRadius), 1.0);
  EXPECT_DOUBLE_EQ(W::settledAmount({}, east(0.0), kRadius), 1.0);

  // two over a point: the one of none has the last word
  W::PlacesRegion none = more;
  none.amount          = 0.0;
  s.regions.push_back(none);
  EXPECT_DOUBLE_EQ(W::settledAmount(s, east(0.0), kRadius), 0.0);

  // its kinds: the named as much as it says, the rest not at all, and all as elsewhere past its edge;
  // a region that names none leaves the kinds as they are
  W::PlacesRegion farms = more;
  farms.kinds           = {{"farm", 2.0}};
  W::Settlement k;
  k.regions = {farms, more};
  EXPECT_DOUBLE_EQ(W::settledKind(k, "farm", east(0.0), kRadius), 2.0);
  EXPECT_DOUBLE_EQ(W::settledKind(k, "hamlet", east(0.0), kRadius), 0.0);
  EXPECT_GT(W::settledKind(k, "hamlet", east(9.0), kRadius), 0.0);
  EXPECT_DOUBLE_EQ(W::settledKind(k, "hamlet", east(10.1), kRadius), 1.0);
  k.regions = {more};
  EXPECT_DOUBLE_EQ(W::settledKind(k, "hamlet", east(0.0), kRadius), 1.0);
}
