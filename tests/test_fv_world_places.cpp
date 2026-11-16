// What people left on a planet (src/fv_world_places.hpp): a catalogue read and checked, places on level
// dry ground, the same for a seed whatever the threads, each on a floor of its own; roads between them
// that cross no lake and step nowhere; single things beside the roads and alone; and nothing of the
// planet's planted on any of it. No two sites of a pack need be alike: each has its own share of what
// may or may not stand there; a group by a road stands on the ground piece by piece; and a kind
// stands no more often than its rule lets it. A planet file's say: a region with none, more, or only
// the kinds it names, and a place by name as near its point as the ground will have it. A river too
// wide to ford is bridged, the road held clear over it; a road's surface lies on its ground as drawn,
// each stretch on one quad, and a bridge on the quad that holds its middle. Its asphalt's maps
// (src/fv_world_asphalt.hpp): a seed's every time, going on round their repeat, worn where the wheels
// went and dusty at the edges; a surface laid on them from edge to edge, each way from its own place.

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include <gtest/gtest.h>

#include <figure/jobs/pool.hpp>

#include "../src/fv_world_places.hpp"
#include "../src/fv_world_plants.hpp"
#include "../src/fv_world_roads.hpp"

namespace {

namespace W = fv::world;
using Json  = nlohmann::json;

constexpr double kPi = 3.14159265358979323846;

// A pack `radius` metres round: something `built` metres each way from its middle that covers the
// ground, kept clear three metres further; with a gate, a drive from its middle out to it along +z
Json packJson(const std::string& name, const std::string& kind, int radius, bool gate, double built)
{
  std::vector<std::string> rows;
  for(int j = 0; j < 2 * radius; j++)
  {
    std::string row;
    for(int i = 0; i < 2 * radius; i++)
    {
      const double x = double(i - radius) + 0.5, z = double(j - radius) + 0.5;
      const bool   covered = (std::abs(x) < built && std::abs(z) < built) || (gate && std::abs(x) < 2.5 && z > 0.0 && z < 0.8 * radius);
      const bool   kept    = std::abs(x) < built + 3.0 && std::abs(z) < built + 3.0;
      row += covered ? '3' : kept ? '2' : '0';
    }
    rows.push_back(row);
  }
  Json pack = {{"name", name},
               {"kind", kind},
               {"radius", radius},
               {"pieces", Json::array({{{"file", name + ".gltf"}, {"copies", Json::array({Json::array({0.0, 0.0, 0.0, 0.0})})}}})},
               {"clear", {{"cell", 1.0}, {"n", 2 * radius}, {"rows", rows}}}};
  if(gate)
    pack["gates"] = Json::array({{{"at", {0.0, 0.8 * radius}}, {"heading", {0.0, 1.0}}}});
  return pack;
}

// How many things lie about the yard that may or may not be there, and at how many of its sites each is
constexpr int    kLeft       = 120;
constexpr double kLeftChance = 0.6;

Json catalogueJson()
{
  const Json anywhere = {{"snow", 0.3}, {"tundra", 1.0}, {"desert", 1.0}, {"steppe", 1.0}, {"grassland", 1.0}, {"forest", 1.0}};
  // the yard: what always stands in it, and things left about it in a ring
  Json yard = packJson("yard", "yard", 60, true, 20.0);
  Json left = Json::array();
  for(int i = 0; i < kLeft; i++)
    left.push_back(Json::array({30.0 * std::cos(0.3 * i), 0.0, 30.0 * std::sin(0.3 * i), 0.1 * i, kLeftChance}));
  yard["pieces"].push_back({{"file", "left.gltf"}, {"copies", left}});
  // the lamp: its post, and two stones a few metres from it
  Json lamp = packJson("lamp", "lamp", 4, false, 0.0);
  lamp["pieces"].push_back({{"file", "stone.gltf"}, {"copies", Json::array({Json::array({3.0, 0.0, 0.5, 0.0}), Json::array({-2.0, 0.0, -3.0, 1.0})})}});
  return {{"version", 1},
          {"packs", Json::array({yard, packJson("stead", "stead", 40, true, 8.0), packJson("hut", "hut", 8, false, 3.0), lamp})},
          {"rules",
           {{"kinds", Json::array({{{"kind", "yard"}, {"road", "road"}, {"weight", 1.0}, {"slope", 0.1}, {"spacing", 2500}, {"biomes", anywhere}},
                                   {{"kind", "stead"}, {"road", "track"}, {"weight", 2.0}, {"slope", 0.12}, {"spacing", 1500}, {"biomes", anywhere}},
                                   {{"kind", "hut"}, {"alone", 0.3}, {"slope", 0.15}, {"biomes", anywhere}}})},
            {"roads", Json::array({{{"name", "road"}, {"width", 6.0}, {"verge", 3.0}, {"colour", {0.17, 0.17, 0.17}}, {"surface", "asphalt"}},
                                   {{"name", "track"}, {"width", 3.0}, {"verge", 1.5}, {"colour", {0.25, 0.19, 0.13}}, {"surface", "earth"}}})},
            {"along", Json::array({{{"kind", "lamp"}, {"road", "road"}, {"every", 40.0}, {"chance", 1.0}, {"side", 4.0}, {"turn", 0.0}, {"spin", 0.0}}})}}}};
}

std::shared_ptr<const W::Catalogue> catalogue()
{
  static const std::shared_ptr<const W::Catalogue> c = W::readCatalogue(catalogueJson());
  return c;
}

// The planet without its places, with them, and what they are
const W::World& bare()
{
  static const W::World w = W::make({}, {.map = 256});
  return w;
}
const W::Places& places()
{
  static const std::shared_ptr<const W::Places> p = W::settle(bare(), catalogue());
  return *p;
}
const W::World& settled()
{
  static const W::World w = W::withPlaces(bare(), W::settle(bare(), catalogue()));
  return w;
}

const W::Pack& packOf(const W::Site& s)
{
  return catalogue()->packs[s.pack];
}

void tangents(const glm::dvec3& d, glm::dvec3& a, glm::dvec3& b)
{
  a = glm::normalize(glm::cross(std::abs(d.y) < 0.9 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), d));
  b = glm::cross(d, a);
}

// The planet's point `away` metres from `d`, `angle` round
glm::dvec3 beside(const W::World& w, const glm::dvec3& d, double away, double angle)
{
  glm::dvec3 a, b;
  tangents(d, a, b);
  return glm::normalize(d + (a * std::cos(angle) + b * std::sin(angle)) * (away / w.spec.radius));
}

// A point of a site's own frame, on the planet
glm::dvec3 ofSite(const W::World& w, const W::Site& s, double x, double z)
{
  const glm::dmat4 frame = W::siteFrame(w, s);
  return W::toPlanet(w, glm::dvec3(frame * glm::dvec4(x, 0.0, z, 1.0)));
}

glm::dvec3 bent(const W::RoadPiece& p, double t)
{
  const double s = 1.0 - t;
  return glm::normalize(p.a * (s * s) + p.b * (2.0 * s * t) + p.c * (t * t));
}
double levelOf(const W::RoadPiece& p, double t)
{
  const double s = 1.0 - t;
  return double(p.level[0]) * s * s + double(p.level[1]) * 2.0 * s * t + double(p.level[2]) * t * t;
}

// Metres from `d` to the nearest road's middle line, and the point of it
double roadAway(const W::World& w, const glm::dvec3& d, glm::dvec3* on = nullptr)
{
  double best = 1e30;
  for(const W::RoadPiece& p : places().roads)
  {
    if(glm::length(p.b - d) * w.spec.radius > double(p.length) + best)
      continue;
    for(int k = 0; k <= 40; k++)
      if(const double away = glm::length(bent(p, k / 40.0) - d) * w.spec.radius; away < best)
      {
        best = away;
        if(on)
          *on = bent(p, k / 40.0);
      }
  }
  return best;
}

bool same(const W::Places& a, const W::Places& b)
{
  if(a.sites.size() != b.sites.size() || a.roads.size() != b.roads.size() || a.joined != b.joined || a.ways != b.ways || a.items != b.items)
    return false;
  for(size_t i = 0; i < a.sites.size(); i++)
    if(a.sites[i].direction != b.sites[i].direction || a.sites[i].pack != b.sites[i].pack || a.sites[i].heading != b.sites[i].heading
       || a.sites[i].elevation != b.sites[i].elevation)
      return false;
  for(size_t i = 0; i < a.roads.size(); i++)
    if(a.roads[i].a != b.roads[i].a || a.roads[i].c != b.roads[i].c || a.roads[i].level[1] != b.roads[i].level[1])
      return false;
  return true;
}

// A stretch of an asphalt road well away from water, from places and from other roads' ends: its
// piece, for the checks of what the ground is across a road
const W::RoadPiece& openRoad()
{
  static const W::RoadPiece* found = nullptr;
  if(!found)
    for(const W::RoadPiece& p : places().roads)
    {
      if(p.half < 2.9F || p.length < 30.0F)
        continue;
      const glm::dvec3 d = bent(p, 0.5);
      const W::Sample  s = W::sample(bare(), d, 2.0);
      bool             clear = s.shore > 60.0F && s.elevation > 5.0 && s.cliff < 0.01F;
      // (from what the ground is levelled for: a lamp stands by every stretch, and shapes nothing)
      for(const W::Site& site : places().sites)
        clear = clear && (!site.level || glm::length(site.direction - d) * bare().spec.radius > packOf(site).radius + W::padEdge(packOf(site).radius) + 30.0);
      // (alone on its ground: no other run of road within its verges)
      for(const W::RoadPiece& q : places().roads)
        if(q.way != p.way)
          for(int k = 0; k <= 8; k++)
            clear = clear && glm::length(bent(q, k / 8.0) - d) * bare().spec.radius > 40.0;
      if(clear)
      {
        found = &p;
        break;
      }
    }
  EXPECT_NE(found, nullptr) << "no open stretch of road on this planet";
  return *found;
}

}  // namespace

// ---- the catalogue -------------------------------------------------------------------------------------

TEST(WorldPlaces, ACatalogueIsReadAsItIsWritten)
{
  const W::Catalogue& c = *catalogue();
  ASSERT_EQ(c.packs.size(), 4u);
  ASSERT_EQ(c.kinds.size(), 4u) << "three with a rule, and the lamp's, which only stands along a road";
  EXPECT_EQ(c.kinds[c.packs[0].kind].name, "yard");
  EXPECT_EQ(c.kinds[0].road, 0);
  EXPECT_EQ(c.kinds[1].road, 1);
  EXPECT_EQ(c.kinds[2].road, -1);
  EXPECT_FLOAT_EQ(c.kinds[2].alone, 0.3F);
  EXPECT_TRUE(c.roads[1].earth);
  EXPECT_FALSE(c.roads[0].earth);
  ASSERT_EQ(c.along.size(), 1u);
  EXPECT_EQ(c.kinds[c.along[0].kind].name, "lamp");
  // a pack's ground: covered under what it is made of, kept clear round that, the land's beyond
  const W::Pack& yard = c.packs[0];
  EXPECT_EQ(yard.groundAt(0.0, 0.0), W::kGroundCovered | W::kGroundKept);
  EXPECT_EQ(yard.groundAt(21.5, 0.0), W::kGroundKept);
  EXPECT_EQ(yard.groundAt(30.0, -30.0), 0);
  EXPECT_EQ(yard.groundAt(0.0, 40.0), W::kGroundCovered | W::kGroundKept) << "its drive";
  EXPECT_EQ(yard.groundAt(0.0, -40.0), 0);
  EXPECT_EQ(yard.groundAt(500.0, 0.0), 0) << "outside it";
  ASSERT_EQ(yard.gates.size(), 1u);
  EXPECT_DOUBLE_EQ(yard.gates[0].at.y, 48.0);
  // what it is made of: a copy stands at every site unless it says at how many
  ASSERT_EQ(yard.pieces.size(), 2u);
  EXPECT_FLOAT_EQ(yard.pieces[0].copies[0].chance, 1.0F);
  ASSERT_EQ(yard.pieces[1].copies.size(), size_t(kLeft));
  EXPECT_FLOAT_EQ(yard.pieces[1].copies[7].chance, float(kLeftChance));
  EXPECT_EQ(c.kinds[0].most, 0u) << "any number of a kind, unless its rule says";
}

TEST(WorldPlaces, ACatalogueSaysWhatIsWrongWithIt)
{
  const auto wrong = [](Json j) -> std::string {
    try
    {
      W::readCatalogue(j);
    }
    catch(const std::runtime_error& e)
    {
      return e.what();
    }
    return "";
  };
  Json j = catalogueJson();
  j["version"] = 2;
  EXPECT_NE(wrong(j).find("version"), std::string::npos);
  j = catalogueJson();
  j["packs"][0]["clear"]["rows"][3] = "000";
  EXPECT_NE(wrong(j).find("yard"), std::string::npos);
  j = catalogueJson();
  j["packs"][1].erase("gates");
  EXPECT_NE(wrong(j).find("stead"), std::string::npos) << "a road is led to its kind, and it has no gate";
  j = catalogueJson();
  j["rules"]["kinds"][0]["road"] = "motorway";
  EXPECT_NE(wrong(j).find("motorway"), std::string::npos);
  j = catalogueJson();
  j["rules"]["kinds"][2]["road"] = "road";
  EXPECT_NE(wrong(j).find("hut"), std::string::npos) << "by a road or alone, not both";
  j = catalogueJson();
  j["packs"][2]["pieces"][0]["copies"][0] = Json::array({1.0, 2.0});
  EXPECT_NE(wrong(j).find("hut"), std::string::npos);
  j = catalogueJson();
  j["packs"][0]["pieces"][1]["copies"][0] = Json::array({1.0, 0.0, 2.0, 0.0, 0.5, 3.0});
  EXPECT_NE(wrong(j).find("left.gltf"), std::string::npos) << "a place, a turn and a chance, and no more";
  j = catalogueJson();
  j["rules"]["along"][0]["kind"] = "pylon";
  EXPECT_NE(wrong(j).find("pylon"), std::string::npos);
  j = catalogueJson();
  j.erase("rules");
  EXPECT_FALSE(wrong(j).empty());
  EXPECT_TRUE(wrong(catalogueJson()).empty());
}

// ---- where places stand ----------------------------------------------------------------------------------

TEST(WorldPlaces, NoMoreOfAKindStandThanItsRuleLets)
{
  const auto yards = [](const W::Places& pl) {
    size_t n = 0;
    for(uint32_t i = 0; i < pl.joined; i++)
      n += pl.catalogue->packs[pl.sites[i].pack].name == "yard" ? 1 : 0;
    return n;
  };
  ASSERT_GT(yards(places()), 3u);
  Json j = catalogueJson();
  j["rules"]["kinds"][0]["most"] = 2;
  const auto few = W::settle(bare(), W::readCatalogue(j));
  ASSERT_TRUE(few);
  EXPECT_EQ(yards(*few), 2u);
  EXPECT_GT(few->joined, 4u) << "the other kind stands as it did";
}

TEST(WorldPlaces, ASiteHasItsOwnShareOfWhatMayStandThere)
{
  const W::World&  w  = settled();
  const W::Places& pl = places();
  size_t           sites = 0, least = size_t(kLeft), most = 0, all = 0;
  for(uint32_t i = 0; i < pl.joined; i++)
  {
    if(packOf(pl.sites[i]).name != "yard")
      continue;
    sites++;
    EXPECT_EQ(W::copiesAt(w, i, 0).size(), 1u) << "what the place is always stands";
    const std::vector<W::PieceCopy> left = W::copiesAt(w, i, 1), again = W::copiesAt(w, i, 1);
    ASSERT_EQ(left.size(), again.size());
    for(size_t k = 0; k < left.size(); k++)
      EXPECT_EQ(left[k].at, again[k].at) << "the same each time it is asked";
    for(const W::PieceCopy& copy : left)
      EXPECT_FLOAT_EQ(copy.at.y, 0.0F) << "on its floor, where the pack has it";
    least = std::min(least, left.size());
    most  = std::max(most, left.size());
    all  += left.size();
  }
  ASSERT_GT(sites, 3u);
  // some sites were cleared out and some left as they were; over them all, the share the pack says
  EXPECT_LT(double(least), 0.5 * kLeft);
  EXPECT_GT(double(most), 0.7 * kLeft);
  EXPECT_LT(most, size_t(kLeft)) << "none has all of it";
  EXPECT_NEAR(double(all) / double(sites * kLeft), kLeftChance, 0.2);
}

TEST(WorldRoads, AGroupByARoadStandsOnTheGroundPieceByPiece)
{
  const W::World&  w  = settled();
  const W::Places& pl = places();
  size_t           looked = 0;
  float            most   = 0.0F;
  for(uint32_t i = pl.joined; i < pl.sites.size(); i++)
  {
    const W::Site& s = pl.sites[i];
    if(s.level || i % 7)
      continue;
    looked++;
    const glm::dmat4 frame = W::siteFrame(w, s);
    for(const W::PieceCopy& copy : W::copiesAt(w, i, 1))
    {
      // (a flat frame against a planet 20 km round: under a millimetre out, 4 m from its middle)
      EXPECT_NEAR(W::heightOver(w, glm::dvec3(frame * glm::dvec4(glm::dvec3(copy.at), 1.0))), 0.0, 0.02);
      most = std::max(most, std::abs(copy.at.y));
    }
  }
  ASSERT_GT(looked, 20u);
  EXPECT_GT(most, 0.1F) << "the ground by a road is not all at the level of the lamp";
}

TEST(WorldPlaces, ASeedsPlacesAreTheSameHoweverManyThreadsSettleIt)
{
  // (a small planet: its settling is a few thousand places asked, twice over here)
  const W::World small = W::make({.seed = 7, .radius = 5000.0}, {.map = 128});
  const auto     a     = W::settle(small, catalogue());
  ASSERT_TRUE(a);
  EXPECT_GT(a->joined, 3u);
  EXPECT_FALSE(a->roads.empty());
  const auto again = W::settle(small, catalogue());
  EXPECT_TRUE(same(*a, *again));
  std::shared_ptr<const W::Places> alone;
  {
    const fv::jobs::Serial serial;
    alone = W::settle(small, catalogue());
  }
  EXPECT_TRUE(same(*a, *alone));
  // another seed's are others
  const auto other = W::settle(W::make({.seed = 8, .radius = 5000.0}, {.map = 128}), catalogue());
  EXPECT_FALSE(same(*a, *other));
  // and none at all where none are asked for
  EXPECT_FALSE(W::settle(small, catalogue(), 0.0));
  EXPECT_FALSE(W::settle(small, nullptr));
}

TEST(WorldPlaces, MoreAreAskedForMoreStand)
{
  const W::World small = W::make({.seed = 7, .radius = 5000.0}, {.map = 128});
  const auto     few = W::settle(small, catalogue(), 0.3), many = W::settle(small, catalogue(), 2.0);
  EXPECT_LT(few->joined, many->joined);
  // (the places: what stands along the roads is as many as they are long, of their kind)
  const auto placesOf = [](const W::Places& pl) {
    return std::count_if(pl.sites.begin(), pl.sites.end(), [](const W::Site& s) { return packOf(s).name != "lamp"; });
  };
  EXPECT_LT(placesOf(*few), placesOf(*many));
}

TEST(WorldPlaces, EveryPlaceIsOnLevelDryLandClearOfTheOthersAndOfTheFigures)
{
  const W::World&  w  = bare();
  const W::Places& pl = places();
  ASSERT_GT(pl.joined, 20u);
  ASSERT_GT(pl.sites.size(), size_t(pl.joined)) << "single things too";
  const double keep = W::mostLevel(w.spec.radius) * 1.5;
  for(uint32_t i = 0; i < pl.sites.size(); i++)
  {
    const W::Site& s = pl.sites[i];
    if(!s.level)
      continue;
    const W::Pack&   pack = packOf(s);
    const W::Sample  here = W::sample(w, s.direction, 4.0);
    EXPECT_GT(here.elevation, 2.5) << i;
    for(int turn = 0; turn < 8; turn++)
    {
      const W::Sample there = W::sample(w, beside(w, s.direction, pack.radius, turn * kPi / 4.0), 4.0);
      EXPECT_GT(there.elevation, 1.0) << i;
      EXPECT_GT(there.shore, 0.0F) << i << ": in the water";
      EXPECT_LT(std::abs(there.elevation - s.elevation), 0.2 * pack.radius + 1.5) << i << ": on a slope";
    }
    EXPECT_GT(glm::length(s.direction - w.frame.site) * w.spec.radius, keep + pack.radius) << i << ": on the figures' ground";
  }
  // no two floors overlap, nor their edges
  for(uint32_t i = 0; i < pl.sites.size(); i++)
    for(uint32_t k = i + 1; k < pl.sites.size(); k++)
      if(pl.sites[i].level && pl.sites[k].level)
      {
        const double ri = packOf(pl.sites[i]).radius, rk = packOf(pl.sites[k]).radius;
        ASSERT_GT(glm::length(pl.sites[i].direction - pl.sites[k].direction) * w.spec.radius, ri + rk + W::padEdge(ri) + W::padEdge(rk)) << i << " and " << k;
      }
  // places with a road keep their distance by their kinds' rule
  for(uint32_t i = 0; i < pl.joined; i++)
    for(uint32_t k = i + 1; k < pl.joined; k++)
      ASSERT_GT(glm::length(pl.sites[i].direction - pl.sites[k].direction) * w.spec.radius, 1499.0);
}

TEST(WorldPlaces, OneStandsWithinAWalkOfTheFigures)
{
  const W::World& w = bare();
  ASSERT_FALSE(places().sites.empty());
  const double away = glm::length(places().sites[0].direction - w.frame.site) * w.spec.radius;
  EXPECT_LT(away, 900.0);
  EXPECT_GT(away, W::mostLevel(w.spec.radius) * 1.5);
}

TEST(WorldPlaces, LevellingTheFiguresPlaceFurtherMovesNothing)
{
  const W::World far = W::withClearing(bare(), W::mostLevel(bare().spec.radius));
  ASSERT_GT(far.spec.clearing, bare().spec.clearing);
  const auto again = W::settle(far, catalogue());
  EXPECT_TRUE(same(places(), *again));
  // and a planet that has its places keeps them through it
  EXPECT_EQ(W::withClearing(settled(), 120.0).places, settled().places);
}

// ---- the ground under them -------------------------------------------------------------------------------

TEST(WorldPlaces, AFloorIsLevelOnItsOwnPlaneAndEasesIntoTheLand)
{
  const W::World& w = settled();
  for(const uint32_t i : {0u, 3u, places().joined - 1})
  {
    const W::Site&   s     = places().sites[i];
    const W::Pack&   pack  = packOf(s);
    const glm::dmat4 frame = glm::inverse(W::siteFrame(w, s));
    // all of it at one height in its own frame, whatever draws it
    for(const double footprint : {1.0, 8.0, 64.0})
      for(double x = -0.7 * pack.radius; x <= 0.7 * pack.radius; x += 0.35 * pack.radius)
        for(double z = -0.7 * pack.radius; z <= 0.7 * pack.radius; z += 0.35 * pack.radius)
        {
          const glm::dvec3 d  = ofSite(w, s, x, z);
          const glm::dvec3 at = glm::dvec3(frame * glm::dvec4(W::toScene(w, d, W::sample(w, d, footprint).elevation), 1.0));
          EXPECT_NEAR(at.y, 0.0, 2e-4) << i << " at " << x << ", " << z;
          EXPECT_NEAR(at.x, x, 1e-3);
          EXPECT_NEAR(at.z, z, 1e-3);
        }
    // past its edge the land is its own again, but where a road leaves it; and between, no step
    int untouched = 0;
    for(int turn = 0; turn < 8; turn++)
    {
      const glm::dvec3 d = beside(w, s.direction, pack.radius + W::padEdge(pack.radius) + 6.0, turn * kPi / 4.0);
      untouched += W::sample(w, d, 1.0).elevation == W::sample(bare(), d, 1.0).elevation ? 1 : 0;
      double last = W::sample(w, s.direction, 1.0).elevation;
      for(double away = 1.0; away < pack.radius + W::padEdge(pack.radius) + 6.0; away += 1.0)
      {
        const double e = W::sample(w, beside(w, s.direction, away, turn * kPi / 4.0), 1.0).elevation;
        ASSERT_LT(std::abs(e - last), 1.2) << i << ", " << away << " m out: a step";
        last = e;
      }
    }
    EXPECT_GE(untouched, 6) << i;
  }
}

TEST(WorldPlaces, WithNoPlacesThePlanetIsAsItWas)
{
  const W::World none = W::withPlaces(settled(), nullptr);
  ASSERT_EQ(none.stages.size(), bare().stages.size());
  for(size_t i = 0; i < none.stages.size(); i++)
    EXPECT_EQ(none.stages[i].name, bare().stages[i].name);
  EXPECT_EQ(settled().stages.size(), bare().stages.size() + 1);
  EXPECT_EQ(W::withPlaces(settled(), settled().places).stages.size(), settled().stages.size()) << "the stage once";
  // and with them, the land away from them is what it was: most of a planet
  int same = 0, all = 0;
  for(uint32_t i = 0; i < 4000; i++)
  {
    const double     y = 1.0 - 2.0 * (double(i) + 0.5) / 4000.0, r = std::sqrt(1.0 - y * y);
    const glm::dvec3 d(r * std::cos(i * 2.4), y, r * std::sin(i * 2.4));
    const W::Sample  a = W::sample(bare(), d, 2.0), b = W::sample(settled(), d, 2.0);
    all++;
    same += a.elevation == b.elevation && a.colour == b.colour && b.built.taken == 0 && b.built.road == 0.0F ? 1 : 0;
  }
  EXPECT_GT(same, all - 40);
}

TEST(WorldPlaces, NothingIsPlantedWhereAPackKeepsClearNorGrassLaidWhereItCovers)
{
  W::World w = settled();
  w.planted  = true;
  const W::Site& s    = places().sites[0];
  const W::Pack& pack = packOf(s);
  // the pack's ground as the planet has it: under its middle covered and kept, beyond it the land's
  EXPECT_EQ(W::sample(w, ofSite(w, s, 0.0, 0.0), 1.0).built.taken, W::kGroundCovered | W::kGroundKept | W::kGroundOpen);
  EXPECT_EQ(W::sample(w, ofSite(w, s, 0.0, 0.0), 1.0).built.site, 0);
  EXPECT_EQ(W::sample(w, ofSite(w, s, pack.radius - 4.0, 0.0), 1.0).built.taken, W::kGroundOpen) << "its floor is a clearing";
  EXPECT_EQ(W::sample(w, ofSite(w, s, pack.radius + 4.0, 0.0), 1.0).built.taken, 0);
  for(double x = -pack.radius; x <= pack.radius; x += 3.7)
    for(double z = -pack.radius; z <= pack.radius; z += 3.7)
      if(x * x + z * z < 0.9 * pack.radius * pack.radius)
        if(const glm::dvec3 d = ofSite(w, s, x, z); roadAway(w, d) > 12.0)  // (but where the land's road comes in to its gate, and along its verges)
          ASSERT_EQ(W::sample(w, d, 1.0).built.taken & 3, pack.groundAt(x, z)) << x << ", " << z;
  // the finest quads over it: none of their plants on kept ground, and some on the rest of the floor
  const uint8_t         finest = W::finestLevel(w);
  const W::FacePoint    at     = W::locate(s.direction);
  const uint32_t        side   = 1U << finest;
  const uint32_t        qx = uint32_t(std::clamp((at.u + 1.0) * 0.5 * side, 0.0, side - 1.0)), qy = uint32_t(std::clamp((at.v + 1.0) * 0.5 * side, 0.0, side - 1.0));
  const std::vector<uint32_t> variants(W::kStandingRoles.size(), 2);
  size_t                plants = 0, low = 0;
  for(int dy = -1; dy <= 1; dy++)
    for(int dx = -1; dx <= 1; dx++)
    {
      const W::Quad quad{at.face, finest, uint32_t(std::clamp(int(qx) + dx, 0, int(side) - 1)), uint32_t(std::clamp(int(qy) + dy, 0, int(side) - 1))};
      const glm::dvec3 centre = W::groundMesh(w, quad).centre;
      for(const W::Stand& stand : W::standing(w, quad, centre, variants, 4.0))
        for(const glm::mat4& place : stand.places)
        {
          const glm::dvec3 d = W::toPlanet(w, centre + glm::dvec3(place[3]));
          const uint8_t    taken = W::sample(w, d, 1.0).built.taken;
          ASSERT_EQ(taken & W::kGroundKept, 0);
          // no tree on the place's floor: bushes and rocks may be
          const std::string_view role = W::kStandingRoles[stand.role];
          if(W::isTall(role))
            ASSERT_EQ(taken & W::kGroundOpen, 0) << role;
          else
            low += taken & W::kGroundOpen ? 1 : 0;
          plants++;
        }
    }
  EXPECT_GT(plants, 0u) << "the land round it is planted";
  EXPECT_GT(low, 0u) << "and bushes or rocks on its floor";
  // grass round the place's middle: no tile's middle on covered ground
  const auto       kinds = W::tileKinds(w);
  W::GrassWas      was;
  const glm::dvec3 eye = W::toScene(w, s.direction, s.elevation + 1.7);
  size_t           tiles = 0;
  for(const W::GrassTile& tile : W::grassTiles(w, {&eye, 1}, kinds, was))
  {
    const W::GrassPlace place = W::grassPlace(w, tile, kinds[tile.kind]);
    ASSERT_EQ(W::sample(w, W::toPlanet(w, place.position), 1.0).built.taken & W::kGroundCovered, 0);
    tiles++;
  }
  EXPECT_GT(tiles, 0u);
}

TEST(WorldPlaces, ASiteIsSeenFromNearAndStaysSeenALittleFurther)
{
  const W::World&  w    = settled();
  const W::Site&   s    = places().sites[0];
  const double     from = W::seenFrom(packOf(s));
  const glm::dvec3 at   = W::toScene(w, s.direction, s.elevation);
  const glm::dvec3 up   = glm::normalize(at - w.frame.centre);
  const auto       sees = [&](double away, std::span<const uint32_t> was = {}) {
    const auto seen = W::sitesSeen(w, at + up * away, was);
    return std::binary_search(seen.begin(), seen.end(), 0u);
  };
  EXPECT_TRUE(sees(0.9 * from));
  EXPECT_FALSE(sees(1.1 * from));
  const uint32_t was[] = {0u};
  EXPECT_TRUE(sees(1.1 * from, was));
  EXPECT_FALSE(sees(1.3 * from, was));
  EXPECT_TRUE(W::sitesSeen(bare(), at).empty());
  // a lamp post shows from nearer than a yard does
  EXPECT_LT(W::seenFrom(catalogue()->packs[3]), W::seenFrom(catalogue()->packs[0]));
}

// ---- the roads ---------------------------------------------------------------------------------------------

TEST(WorldRoads, ARoadLeavesAPlaceByItsGateAndLeadsToAnother)
{
  const W::World&  w  = settled();
  const W::Places& pl = places();
  ASSERT_GT(pl.roads.size(), 100u);
  // the ends of the runs of road, and which places' gates they are at
  std::map<std::tuple<double, double, double>, uint32_t> ends;  // a point of the planet: its number
  const auto node = [&](const glm::dvec3& d) {
    return ends.emplace(std::make_tuple(std::round(d.x * 1e9), std::round(d.y * 1e9), std::round(d.z * 1e9)), uint32_t(ends.size())).first->second;
  };
  std::vector<std::pair<uint32_t, uint32_t>> runs;
  for(size_t i = 0; i < pl.roads.size(); i++)
    if(i == 0 || pl.roads[i].way != pl.roads[i - 1].way)
    {
      size_t last = i;
      while(last + 1 < pl.roads.size() && pl.roads[last + 1].way == pl.roads[i].way)
        last++;
      runs.emplace_back(node(pl.roads[i].a), node(pl.roads[last].c));
    }
  std::vector<uint32_t> group(ends.size() + pl.joined);
  std::iota(group.begin(), group.end(), 0U);
  const auto root = [&](uint32_t i) {
    while(group[i] != i)
      i = group[i] = group[group[i]];
    return i;
  };
  for(const auto& [a, b] : runs)
    group[root(a)] = root(b);
  uint32_t gated = 0;
  for(uint32_t i = 0; i < pl.joined; i++)
  {
    const W::Gate&   gate = packOf(pl.sites[i]).gates[0];
    const glm::dvec3 d    = ofSite(w, pl.sites[i], gate.at.x, gate.at.y);
    for(const auto& [point, number] : ends)
      if(glm::length(glm::dvec3(std::get<0>(point), std::get<1>(point), std::get<2>(point)) * 1e-9 - d) * w.spec.radius < 0.05)
      {
        group[root(uint32_t(ends.size()) + i)] = root(number);
        gated++;
      }
  }
  EXPECT_GT(gated, pl.joined * 8 / 10) << "but for a place alone on an island, or among bluffs";
  // every place with a road is joined to another by it
  std::map<uint32_t, uint32_t> together;
  for(uint32_t i = 0; i < pl.joined; i++)
    together[root(uint32_t(ends.size()) + i)]++;
  uint32_t lonely = 0;
  for(const auto& [g, count] : together)
    lonely += count == 1 ? 1 : 0;
  // (but the one by the figures, when it is alone on its land: its road leads to them)
  EXPECT_LE(lonely, pl.joined - gated + 1);
}

TEST(WorldRoads, ItMeetsAGateOnThePlacesFloorAndRunsStraightOutOfIt)
{
  const W::World& w = settled();
  uint32_t        checked = 0;
  for(uint32_t i = 0; i < places().joined && checked < 12; i++)
  {
    const W::Site&   s    = places().sites[i];
    const W::Gate&   gate = packOf(s).gates[0];
    const glm::dvec3 d    = ofSite(w, s, gate.at.x, gate.at.y);
    if(roadAway(w, d) > 0.05)
      continue;  // (none led to it)
    checked++;
    // on the floor all the way from the gate to the floor's edge, along the gate's way out
    const glm::dmat4 frame = glm::inverse(W::siteFrame(w, s));
    for(double out = 0.0; glm::length(gate.at + gate.heading * out) < packOf(s).radius - 1.0; out += 2.0)
    {
      const glm::dvec2 at = gate.at + gate.heading * out;
      const glm::dvec3 q  = ofSite(w, s, at.x, at.y);
      const W::Sample  on = W::sample(w, q, 1.0);
      EXPECT_GT(on.built.road, 0.95F) << i << ", " << out << " m out of its gate";
      EXPECT_NEAR(glm::dvec3(frame * glm::dvec4(W::toScene(w, q, on.elevation), 1.0)).y, 0.0, 2e-4);
    }
    // and behind the gate, in the place, it is the pack's own ground
    EXPECT_LT(W::sample(w, ofSite(w, s, gate.at.x - gate.heading.x * 12.0 + gate.heading.y * 9.0, gate.at.y - gate.heading.y * 12.0 - gate.heading.x * 9.0), 1.0).built.road, 0.05F);
  }
  EXPECT_GE(checked, 8u);
}

TEST(WorldRoads, ARoadCrossesNoLakeNorTheSea)
{
  // (a stream it fords: there its ground is the stream's bed)
  size_t wet = 0, all = 0, banked = 0;
  for(const W::RoadPiece& p : places().roads)
    for(int k = 0; k <= 4; k++)
    {
      const glm::dvec3 d = bent(p, k / 4.0);
      const W::Sample  s = W::sample(settled(), d, 1.0), was = W::sample(bare(), d, 2.0);
      all++;
      wet += (s.elevation < 0.5 && s.shore > 7.0F) || (was.inland.lake > 0.5 && was.water > was.elevation) ? 1 : 0;
      banked += was.elevation < 0.0 ? 1 : 0;
    }
  EXPECT_EQ(wet, 0u) << "of " << all;
  // an inlet finer than its routes it crosses on a bank: little of it
  EXPECT_LT(banked, all / 300) << "of " << all;
}

TEST(WorldRoads, ItsLevelHasNoStepAlongItNorAtAJunction)
{
  const W::World&  w  = settled();
  const W::Places& pl = places();
  // the pieces of a run follow on, in place and in level
  for(size_t i = 1; i < pl.roads.size(); i++)
    if(pl.roads[i].way == pl.roads[i - 1].way)
    {
      ASSERT_LT(glm::length(pl.roads[i].a - pl.roads[i - 1].c) * w.spec.radius, 1e-6) << i;
      ASSERT_FLOAT_EQ(pl.roads[i].level[0], pl.roads[i - 1].level[2]) << i;
      ASSERT_NEAR(pl.roads[i].start, pl.roads[i - 1].start + pl.roads[i - 1].length, 0.01F * pl.roads[i].start + 0.01F);
    }
  // runs that meet, meet at one level
  std::map<std::tuple<double, double, double>, float> at;
  size_t                                              junctions = 0;
  for(size_t i = 0; i < pl.roads.size(); i++)
    for(const auto& [d, level] : {std::make_pair(pl.roads[i].a, pl.roads[i].level[0]), std::make_pair(pl.roads[i].c, pl.roads[i].level[2])})
    {
      const auto [where, fresh] = at.emplace(std::make_tuple(std::round(d.x * 1e9), std::round(d.y * 1e9), std::round(d.z * 1e9)), level);
      if(!fresh)
      {
        ASSERT_NEAR(where->second, level, 1e-3F);
        junctions++;
      }
    }
  EXPECT_GT(junctions, pl.roads.size() / 2) << "every piece's end is another's beginning, or a junction, or a gate";
  // and the ground along a road is the road's (dry stretches: by a stream it comes down to the
  // bank), which is seldom steeper than a road is: only where the land itself is, up a bluff
  size_t off = 0, steep = 0, walked = 0;
  double      worst = 0.0;
  std::string worstAt;
  for(size_t i = 0; i < pl.roads.size(); i += 7)
  {
    const W::RoadPiece& p = pl.roads[i];
    const int           n = std::max(2, int(p.length));
    double              last = 0.0;
    bool                had  = false;
    for(int k = 0; k <= n; k++)
    {
      const W::Sample s   = W::sample(w, bent(p, double(k) / n), 1.0);
      const bool      dry = s.shore > 12.0F;
      if(dry)
      {
        off += std::abs(s.elevation - levelOf(p, double(k) / n)) > 0.05 ? 1 : 0;
        if(const double by = std::abs(s.elevation - levelOf(p, double(k) / n)); by > worst)
        {
          worst = by;
          // (what is there, for whoever reads a failure: the other runs near, a place's floor)
          const glm::dvec3   d = bent(p, double(k) / n);
          std::ostringstream said;
          said << "piece " << i << " of run " << p.way << " at " << k << "/" << n << ": level " << levelOf(p, double(k) / n) << ", ground " << s.elevation << ", the land's "
               << W::sample(bare(), d, 1.0).elevation << "; near:";
          for(const W::RoadPiece& q : pl.roads)
            if(q.way != p.way)
              for(int j = 0; j <= 4; j++)
                if(const double away = glm::length(bent(q, j / 4.0) - d) * w.spec.radius; away < 25.0)
                {
                  said << " run " << q.way << " " << away << " m off at level " << levelOf(q, j / 4.0) << ";";
                  break;
                }
          for(uint32_t site = 0; site < pl.sites.size(); site++)
            if(pl.sites[site].level)
              if(const double away = glm::length(pl.sites[site].direction - d) * w.spec.radius - packOf(pl.sites[site]).radius; away < 60.0)
                said << " site " << site << " " << away << " m past its reach, floor " << pl.sites[site].elevation << ";";
          worstAt = said.str();
        }
        EXPECT_GT(s.elevation, 0.9);
      }
      if(dry && had)
      {
        walked++;
        steep += std::abs(s.elevation - last) > 0.2 * double(p.length) / n ? 1 : 0;
      }
      had  = dry;
      last = s.elevation;
    }
  }
  ASSERT_GT(walked, 1000u);
  // (off its own level only by a junction, where the runs that meet each have a say: a junction
  // on a bluff is warped to them all, by less than a cutting is deep)
  EXPECT_LT(off, walked / 30) << "of " << walked << " metres walked";
  EXPECT_LT(worst, 4.0) << worstAt;
  EXPECT_LT(steep, walked / 20) << "of " << walked << " metres walked";
}

TEST(WorldRoads, TheGroundIsLevelAcrossItAndEasesIntoItsVerges)
{
  const W::World&     w = settled();
  const W::RoadPiece& p = openRoad();
  const glm::dvec3    on = bent(p, 0.5), run = glm::normalize(bent(p, 0.55) - bent(p, 0.45));
  const glm::dvec3    across = glm::normalize(glm::cross(on, run));
  const double        level  = levelOf(p, 0.5);
  const auto          at     = [&](double side) { return glm::normalize(on + across * (side / w.spec.radius)); };
  for(double side = -double(p.half); side <= double(p.half); side += 0.5)
  {
    const W::Sample s = W::sample(w, at(side), 1.0);
    EXPECT_NEAR(s.elevation, level, 0.08) << side << " m across";
    if(std::abs(side) < double(p.half) - 1.0)
    {
      EXPECT_GT(s.built.road, 0.95F);
      EXPECT_EQ(s.built.taken, W::kGroundCovered | W::kGroundKept);
      EXPECT_FALSE(s.built.earth);
      // asphalt: none of the ground's layers, and the road's colour
      const auto layers = W::surfaceLayers(w, s, 0.0F);
      EXPECT_LT(std::accumulate(layers.begin(), layers.end(), 0.0F), 0.06F);
      EXPECT_NEAR(s.colour.x, 0.17F, 0.04F);
      EXPECT_NEAR(s.colour.y, s.colour.x, 1e-4F);
    }
  }
  // past its verges the land is what it was (they are wider where the road is cut into the land or
  // banked over it: a bank is no steeper than 1 in 2), and between there is no step
  for(const double side : {-1.0, 1.0})
  {
    const glm::dvec3 d = at(side * (double(p.half) + double(p.verge) + 12.5));
    EXPECT_EQ(W::sample(w, d, 1.0).elevation, W::sample(bare(), d, 1.0).elevation);
    EXPECT_EQ(W::sample(w, d, 1.0).built.road, 0.0F);
    double last = W::sample(w, at(0.0), 1.0).elevation;
    for(double out = 0.25; out < double(p.half) + double(p.verge) + 13.0; out += 0.25)
    {
      const double e = W::sample(w, at(side * out), 1.0).elevation;
      EXPECT_LT(std::abs(e - last), 0.3) << out;
      last = e;
    }
  }
}

TEST(WorldRoads, CutFromNearAndPaintedFromAfar)
{
  const W::World&     w  = settled();
  const W::RoadPiece& p  = openRoad();
  const glm::dvec3    on = bent(p, 0.5);
  // what draws it finely has its bed; what draws it coarsely has the land as it was, and a share of
  // its colour by how much of a sample it is
  EXPECT_NEAR(W::sample(w, on, 1.0).elevation, levelOf(p, 0.5), 0.08);
  EXPECT_NEAR(W::sample(w, on, 2.0).elevation, levelOf(p, 0.5), 0.08);
  EXPECT_EQ(W::sample(w, on, 64.0).elevation, W::sample(bare(), on, 64.0).elevation);
  const float far = W::sample(w, on, 64.0).built.road;
  EXPECT_GT(far, 0.02F);
  EXPECT_LT(far, 0.2F);
  EXPECT_GT(W::sample(w, on, 8.0).built.road, far);
  EXPECT_EQ(W::sample(w, on, 64.0).built.taken, W::kGroundKept) << "no tree on it from any distance, though it is too thin to count as its surface";
}

TEST(WorldRoads, ATrackIsEarth)
{
  const W::World& w = settled();
  for(const W::RoadPiece& p : places().roads)
    if(p.half < 2.0F)
    {
      const W::Sample s = W::sample(w, bent(p, 0.5), 1.0);
      if(s.built.road < 0.95F || s.shore < 30.0F)
        continue;  // (by a ford, or where a wider road joins)
      EXPECT_TRUE(s.built.earth);
      EXPECT_GT(W::surfaceLayers(w, s, 0.0F)[size_t(W::Soil::eDry)], 0.9F);  // the dry lands' stony earth
      EXPECT_GT(s.colour.x, s.colour.z * 1.3F);
      return;
    }
  FAIL() << "no track on this planet";
}

// ---- single things -------------------------------------------------------------------------------------------

TEST(WorldRoads, ThingsStandBesideItFacingIt)
{
  const W::World&  w  = settled();
  const W::Places& pl = places();
  size_t           lamps = 0;
  for(const W::Site& s : pl.sites)
  {
    if(s.level)
      continue;
    lamps++;
    EXPECT_EQ(packOf(s).name, "lamp");
    if(lamps % 9)
      continue;  // (one in nine looked at: each look goes over every road)
    glm::dvec3   on;
    const double away = roadAway(w, s.direction, &on);
    if(away < 3.6)
      continue;  // (by a junction: another road is nearer than its own)
    EXPECT_NEAR(away, 4.0, 0.6);
    // its own +x, in the scene, is toward the road
    const glm::dmat4 frame  = W::siteFrame(w, s);
    const glm::dvec3 toward = glm::normalize(W::toScene(w, on, 0.0) - W::toScene(w, s.direction, 0.0));
    EXPECT_GT(glm::dot(glm::dvec3(frame[0]), toward), 0.9);
    // and it stands on the ground as the road left it
    EXPECT_NEAR(W::heightOver(w, glm::dvec3(frame[3])), 0.0, 1e-3);
  }
  EXPECT_GT(lamps, 100u);
}

TEST(WorldRoads, AGroupStandsOnlyWhereTheGroundIsAsLevelAsItsRuleAsks)
{
  const auto beside = [](const W::Places& pl) {
    std::vector<W::Site> out;
    for(uint32_t i = pl.joined; i < pl.sites.size(); i++)
      if(!pl.sites[i].level)
        out.push_back(pl.sites[i]);
    return out;
  };
  Json j = catalogueJson();
  j["rules"]["along"][0]["slope"] = 0.03;
  const auto level = W::settle(bare(), W::readCatalogue(j));
  ASSERT_TRUE(level);
  const std::vector<W::Site> all = beside(places()), some = beside(*level);
  EXPECT_LT(some.size(), all.size()) << "not every stretch of road is on level ground";
  ASSERT_GT(some.size(), 20u);
  // the ground 0.6 of its reach out (2.4 m of the lamp's 4), along and across
  const W::World& w     = bare();
  const auto      steep = [&](const W::Site& s) {
    const double e = W::sample(w, s.direction, 4.0).elevation;
    double       most = 0.0;
    for(int k = 0; k < 4; k++)
      most = std::max(most, std::abs(W::sample(w, ::beside(w, s.direction, 2.4, 0.5 * kPi * k), 4.0).elevation - e) / 2.4);
    return most;
  };
  size_t steeper = 0;
  for(const W::Site& s : some)
    EXPECT_LT(steep(s), 0.03 + 1e-6);
  for(const W::Site& s : all)
    steeper += steep(s) > 0.03 ? 1 : 0;
  EXPECT_GT(steeper, 0u) << "without the rule's slope, some stand on steeper ground";
}

TEST(WorldPlaces, AThingAloneStandsClearOfRoadsAndPlaces)
{
  const W::World&  w  = settled();
  const W::Places& pl = places();
  size_t           huts = 0;
  for(uint32_t i = pl.joined; i < pl.sites.size(); i++)
  {
    const W::Site& s = pl.sites[i];
    if(!s.level)
      continue;
    huts++;
    EXPECT_EQ(packOf(s).name, "hut");
    if(huts % 5 == 0)
      EXPECT_GT(roadAway(w, s.direction), packOf(s).radius + W::padEdge(packOf(s).radius) + 9.0);
    // on a floor of its own
    EXPECT_NEAR(W::sample(w, s.direction, 1.0).elevation, s.elevation, 1e-6);
    EXPECT_EQ(W::sample(w, s.direction, 1.0).built.taken, W::kGroundCovered | W::kGroundKept) << "and in no clearing: the wood is up to its walls";
  }
  // about three in ten square kilometres of land that will have one (some is too steep, or wet)
  EXPECT_GT(huts, 100u);
  EXPECT_LT(huts, 1600u);
}

// Whoever stands where the figures do has a way to follow: a road from the place nearest them comes
// to within a stone's throw and ends there, clear of where they stand
TEST(WorldRoads, ARoadComesToWhereTheFiguresStand)
{
  const W::World& w = bare();
  double          nearest = 1e30;
  for(const W::RoadPiece& p : places().roads)
    for(int k = 0; k <= 4; k++)
      nearest = std::min(nearest, glm::length(bent(p, k / 4.0) - w.frame.site) * w.spec.radius);
  EXPECT_LT(nearest, 250.0);
  EXPECT_GT(nearest, 39.0);
  // and it is the first place's: one end of a run is its gate
  const W::Site&   s    = places().sites[0];
  const W::Gate&   gate = packOf(s).gates[0];
  EXPECT_LT(roadAway(settled(), ofSite(settled(), s, gate.at.x, gate.at.y)), 0.05);
}

// ---- what a planet file says of where people lived -------------------------------------------------------

namespace {

const std::string& kindOf(const W::Site& s)
{
  return catalogue()->kinds[packOf(s).kind].name;
}

double apart(const glm::dvec3& a, const glm::dvec3& b)
{
  return glm::length(a - b) * bare().spec.radius;
}

// A place of these plain places as far from the figures as there is, of the kind called `kind` (any: "")
glm::dvec3 farPlace(const std::string& kind = "")
{
  const W::Places& pl   = places();
  double           most = -1.0;
  glm::dvec3       far{0.0};
  for(uint32_t i = 1; i < pl.joined; i++)
    if((kind.empty() || kindOf(pl.sites[i]) == kind) && apart(pl.sites[i].direction, bare().frame.site) > most)
    {
      most = apart(pl.sites[i].direction, bare().frame.site);
      far  = pl.sites[i].direction;
    }
  EXPECT_GT(most, 0.0) << "no place of kind " << kind;
  return far;
}

std::shared_ptr<const W::Settlement> region(const glm::dvec3& at, double width, double amount, std::vector<std::pair<std::string, double>> kinds = {})
{
  auto s = std::make_shared<W::Settlement>();
  W::PlacesRegion r;
  r.shape.from = r.shape.to = at;
  r.shape.width             = width;
  r.amount                  = amount;
  r.kinds                   = std::move(kinds);
  s->regions.push_back(r);
  return s;
}

std::shared_ptr<const W::Places> settledBy(std::shared_ptr<const W::Settlement> plan)
{
  return W::settle(bare(), catalogue(), 1.0, nullptr, std::move(plan));
}

// Of these places, the ones with a road within `within` metres of `at`, and the things alone there
uint32_t joinedNear(const W::Places& pl, const glm::dvec3& at, double within)
{
  uint32_t n = 0;
  for(uint32_t i = 0; i < pl.joined; i++)
    n += apart(pl.sites[i].direction, at) < within ? 1 : 0;
  return n;
}
uint32_t aloneNear(const W::Places& pl, const glm::dvec3& at, double within)
{
  uint32_t n = 0;
  for(uint32_t i = pl.joined; i < pl.sites.size(); i++)
    n += kindOf(pl.sites[i]) != "lamp" && apart(pl.sites[i].direction, at) < within ? 1 : 0;
  return n;
}
double roadNear(const W::Places& pl, const glm::dvec3& at, double within)
{
  double metres = 0.0;
  for(const W::RoadPiece& p : pl.roads)
    metres += apart(p.b, at) < within ? double(p.length) : 0.0;
  return metres;
}

}  // namespace

TEST(WorldPlaces, ARegionOfNoneHasNoPlacesAndTheRoadsGoRoundIt)
{
  const glm::dvec3 at    = farPlace();
  const double     width = 3000.0, inner = width * (1.0 - W::kRegionEdge);
  ASSERT_GT(joinedNear(places(), at, inner), 0U);
  const auto pl = settledBy(region(at, width, 0.0));
  ASSERT_TRUE(pl);
  EXPECT_EQ(joinedNear(*pl, at, inner), 0U);
  EXPECT_EQ(aloneNear(*pl, at, inner), 0U);
  EXPECT_LT(roadNear(*pl, at, inner), 0.5 * roadNear(places(), at, inner) + 1.0);
  // and elsewhere there are places as there were
  EXPECT_GT(pl->joined, places().joined / 2);
  EXPECT_TRUE(pl->unplaced.empty());
}

TEST(WorldPlaces, ARegionOfMoreHasMoreAndNearerEachOther)
{
  const glm::dvec3 at    = farPlace();
  const double     width = 4000.0, inner = width * (1.0 - W::kRegionEdge);
  const auto       pl    = settledBy(region(at, width, 3.0));
  ASSERT_TRUE(pl);
  EXPECT_GT(joinedNear(*pl, at, inner), joinedNear(places(), at, inner) + 1U);
  // (nearer than the rule keeps them elsewhere)
  double nearest = 1e30;
  for(uint32_t i = 0; i < pl->joined; i++)
    for(uint32_t k = i + 1; k < pl->joined; k++)
      if(apart(pl->sites[i].direction, at) < inner && apart(pl->sites[k].direction, at) < inner)
        nearest = std::min(nearest, apart(pl->sites[i].direction, pl->sites[k].direction));
  EXPECT_LT(nearest, 1499.0);
  // the same for a seed whatever the threads
  std::shared_ptr<const W::Places> alone;
  {
    const fv::jobs::Serial serial;
    alone = settledBy(region(at, width, 3.0));
  }
  EXPECT_TRUE(same(*pl, *alone));
}

TEST(WorldPlaces, ARegionsKindsAreTheOnlyOnesThere)
{
  const glm::dvec3 at    = farPlace("yard");
  const double     width = 4000.0, inner = width * (1.0 - W::kRegionEdge);
  const auto       pl    = settledBy(region(at, width, 1.0, {{"stead", 1.0}}));
  ASSERT_TRUE(pl);
  uint32_t steads = 0;
  for(uint32_t i = 0; i < pl->sites.size(); i++)
    if(apart(pl->sites[i].direction, at) < inner && kindOf(pl->sites[i]) != "lamp")
    {
      EXPECT_EQ(kindOf(pl->sites[i]), "stead") << i;
      steads++;
    }
  EXPECT_GT(steads, 0U);
}

TEST(WorldPlaces, APlaceByNameStandsAsNearItsPointAsTheGroundWillHaveIt)
{
  const glm::dvec3 at = farPlace("stead");
  // a point of the sea, far from any shore
  glm::dvec3 sea{0.0};
  for(int lat = -60; lat <= 60 && sea == glm::dvec3(0.0); lat += 10)
    for(int lon = -180; lon < 180 && sea == glm::dvec3(0.0); lon += 10)
    {
      const glm::dvec3 d   = W::directionOf(lat, lon);
      bool             wet = W::sample(bare(), d, 50.0).elevation < -50.0;
      for(int turn = 0; turn < 8 && wet; turn++)
        wet = W::sample(bare(), beside(bare(), d, 1700.0, turn * kPi / 4.0), 50.0).elevation < -5.0;
      if(wet)
        sea = d;
    }
  ASSERT_NE(sea, glm::dvec3(0.0)) << "no open sea on this planet";

  auto plan = std::make_shared<W::Settlement>();
  plan->named = {{"Kelda", "stead", at}, {"Nowhere", "castle", at}, {"Atlantis", "yard", sea}};
  const auto pl = settledBy(plan);
  ASSERT_TRUE(pl);
  ASSERT_EQ(pl->names.size(), 1U);
  const auto& [site, name] = pl->names[0];
  EXPECT_EQ(name, "Kelda");
  ASSERT_LT(site, pl->joined) << "a place with a road";
  EXPECT_EQ(kindOf(pl->sites[site]), "stead");
  EXPECT_LT(apart(pl->sites[site].direction, at), 250.0);
  // what could not stand, each said why
  ASSERT_EQ(pl->unplaced.size(), 2U);
  EXPECT_NE(pl->unplaced[0].find("Nowhere"), std::string::npos);
  EXPECT_NE(pl->unplaced[0].find("castle"), std::string::npos);
  EXPECT_NE(pl->unplaced[1].find("Atlantis"), std::string::npos);
  EXPECT_NE(pl->unplaced[1].find("no ground"), std::string::npos);
  // and a road is led to it
  bool reached = false;
  for(const W::RoadPiece& p : pl->roads)
    reached = reached || apart(p.a, pl->sites[site].direction) < packOf(pl->sites[site]).radius + 30.0
              || apart(p.c, pl->sites[site].direction) < packOf(pl->sites[site]).radius + 30.0;
  EXPECT_TRUE(reached);
  // the same every time
  EXPECT_TRUE(same(*pl, *settledBy(plan)));
}

TEST(WorldPlaces, AnEmptyPlanAsksNothingOfTheRules)
{
  EXPECT_TRUE(same(places(), *settledBy(std::make_shared<W::Settlement>())));
}

namespace {

// The quad of `level` a direction is on
W::Quad quadOf(const glm::dvec3& d, uint8_t level)
{
  const W::FacePoint p = W::locate(d);
  const double       n = double(uint64_t(1) << level);
  return {p.face, level, uint32_t(std::clamp((p.u + 1.0) * 0.5 * n, 0.0, n - 1.0)), uint32_t(std::clamp((p.v + 1.0) * 0.5 * n, 0.0, n - 1.0))};
}

double areaOf(const W::RoadMesh& m)
{
  double area = 0.0;
  for(size_t i = 0; i + 2 < m.indices.size(); i += 3)
    area += 0.5 * glm::length(glm::cross(glm::dvec3(m.positions[m.indices[i + 1]] - m.positions[m.indices[i]]),
                                         glm::dvec3(m.positions[m.indices[i + 2]] - m.positions[m.indices[i]])));
  return area;
}

}  // namespace

TEST(WorldRoads, ARiverItDoesNotFordItCrossesByABridge)
{
  const W::World&  w  = settled();
  const W::Places& pl = places();
  ASSERT_GT(pl.bridges.size(), 10U);
  ASSERT_EQ(pl.wayBridges.size(), size_t(pl.ways) + 1);
  size_t tracks = 0, level = 0;
  for(const W::Bridge& b : pl.bridges)
  {
    ASSERT_LT(b.from, b.to);
    const W::OnWay middle = W::onWay(pl, b.way, 0.5 * (double(b.from) + double(b.to)));
    ASSERT_TRUE(middle.piece);
    const bool earth = catalogue()->roads[middle.piece->kind].earth;
    tracks += earth ? 1 : 0;
    // a river there, wider than its kind of road fords
    double widest = 0.0;
    for(double s = b.from; s <= b.to; s += 1.0)
      if(const W::Inland in = W::mapInland(*w.base, w.spec.radius, W::onWay(pl, b.way, s).direction); in.river)
        widest = std::max(widest, in.riverHalf);
    EXPECT_GT(widest, earth ? 5.0 : 1.0) << "a bridge of way " << b.way << " at " << b.from;
    // the road over it clear of the water, and level from bank to bank
    bool flat = true;
    for(double s = b.from; s <= b.to; s += 1.0)
    {
      const W::OnWay on = W::onWay(pl, b.way, s);
      EXPECT_GT(on.level, double(b.water) + 0.5) << "a bridge of way " << b.way << " at " << s;
      flat = flat && std::abs(on.level - double(b.level)) < 0.05;
    }
    level += flat ? 1 : 0;
    // the river's ground under it as it was, nothing of the road's on it, and nothing planted there
    const W::Sample s = W::sample(w, middle.direction, 1.0), was = W::sample(bare(), middle.direction, 1.0);
    EXPECT_NEAR(s.elevation, was.elevation, 0.01);
    EXPECT_LT(s.built.road, 0.01F);
    EXPECT_NE(s.built.taken & W::kGroundKept, 0);
  }
  EXPECT_GE(level, pl.bridges.size() * 9 / 10) << "(but one by a way's end, which the way's other run decides)";
  // tracks ford a stream, and bridge a river; a road bridges both
  EXPECT_GT(tracks, 0U);
  EXPECT_LT(tracks, pl.bridges.size());
}

TEST(WorldRoads, ItsSurfaceLiesOnItsGroundAndEachStretchIsDrawnOnce)
{
  const W::World&  w      = settled();
  const W::Places& pl     = places();
  const uint8_t    finest = W::finestLevel(w);
  // quads along roads (not tracks), a level up from the finest, so fine enough for a surface
  std::vector<W::Quad> quads;
  for(size_t i = 0; i < pl.roads.size() && quads.size() < 10; i += 29)
  {
    const W::RoadPiece& p = pl.roads[i];
    if(!catalogue()->roads[p.kind].earth && !W::bridgeAt(pl, p.way, double(p.start) + 0.5 * double(p.length), 40.0))
      quads.push_back(quadOf(glm::normalize(p.b), uint8_t(finest - 1)));
  }
  std::sort(quads.begin(), quads.end());
  quads.erase(std::unique(quads.begin(), quads.end()), quads.end());
  ASSERT_GE(quads.size(), 6U);
  size_t laid = 0, near = 0;
  for(const W::Quad& q : quads)
  {
    const glm::dvec3      centre = W::toScene(w, W::direction(q.face, (double(q.x) + 0.5) * 2.0 / double(1U << q.level) - 1.0, (double(q.y) + 0.5) * 2.0 / double(1U << q.level) - 1.0), 0.0);
    const W::RoadMeshes   m      = W::roadMeshes(w, q, centre);
    ASSERT_FALSE(m.surface.empty());
    EXPECT_FALSE(m.marking.empty()) << "a road's middle and edges are marked";
    EXPECT_EQ(m.surface.positions.size(), m.surface.normals.size());
    EXPECT_EQ(m.surface.positions.size(), m.surface.uvs.size());
    // what its four children have of it between them is what it has: each stretch is one quad's
    double children = 0.0, paint = 0.0;
    for(const W::Quad& c : W::children(q))
    {
      const W::RoadMeshes cm = W::roadMeshes(w, c, centre);
      children += areaOf(cm.surface);
      paint += areaOf(cm.marking);
    }
    EXPECT_NEAR(children, areaOf(m.surface), 0.05 * areaOf(m.surface));
    EXPECT_NEAR(paint, areaOf(m.marking), 0.1 * areaOf(m.marking));
    // on the ground as the quad draws it, a few centimetres over it (its points are the ground's
    // own where the ground is level across the road, as it is but by water)
    const double cell = W::width(w, q) / double(W::kGroundCells), lift = W::surfaceLift(cell);
    for(size_t v = 0; v < m.surface.positions.size(); v++)
    {
      if(v % 7 == 0 || v % 7 == 6)
        continue;  // (its edges, which go down into the ground)
      double           r = 0.0;
      const glm::dvec3 d = W::toPlanet(w, centre + glm::dvec3(m.surface.positions[v]), &r);
      const W::Sample  s = W::sample(w, d, cell);
      if(s.shore < 25.0F)
        continue;
      const double over = r - w.spec.radius - s.elevation;
      laid++;
      near += over > lift - 0.06 && over < lift + 0.12 ? 1 : 0;
      EXPECT_LT(std::abs(over - lift), 0.6) << "quad " << int(q.face) << "/" << int(q.level) << "/" << q.x << "/" << q.y;
    }
  }
  EXPECT_GT(near, laid * 95 / 100) << "of " << laid;
}

TEST(WorldRoads, ABridgeIsDrawnByTheQuadThatHoldsItsMiddleAtAnyLevel)
{
  const W::World&  w      = settled();
  const W::Places& pl     = places();
  const uint8_t    finest = W::finestLevel(w);
  ASSERT_GE(pl.bridges.size(), 8U);
  for(size_t i = 0; i < 8; i++)
  {
    const W::Bridge& b      = pl.bridges[i * pl.bridges.size() / 8];
    const glm::dvec3 middle = W::onWay(pl, b.way, 0.5 * (double(b.from) + double(b.to))).direction;
    for(const uint8_t level : {uint8_t(finest - 6), uint8_t(finest - 2), finest})
    {
      const W::Quad       q      = quadOf(middle, level);
      const glm::dvec3    centre = W::toScene(w, middle, 0.0);
      const W::RoadMeshes m      = W::roadMeshes(w, q, centre);
      ASSERT_FALSE(m.bridge.empty()) << "bridge " << i << " at level " << int(level);
      // from over the road (its parapets) down into the bank and the river's bed (of its own parts:
      // a coarse quad holds others)
      double top = -1e30, bottom = 1e30;
      for(const glm::vec3& p : m.bridge.positions)
      {
        double           r  = 0.0;
        const glm::dvec3 at = W::toPlanet(w, centre + glm::dvec3(p), &r);
        if(glm::length(at - middle) * w.spec.radius > 0.5 * double(b.to - b.from) + W::kBridgeEnds + 15.0)
          continue;
        top    = std::max(top, r - w.spec.radius);
        bottom = std::min(bottom, r - w.spec.radius);
      }
      EXPECT_GT(top, double(b.level) + 0.7);
      EXPECT_LT(top, double(b.level) + 1.5);
      EXPECT_LT(bottom, double(b.water));
      // its parent's is all of it, and its four children have it once between them
      const W::Quad parent{q.face, uint8_t(level - 1), q.x / 2, q.y / 2};
      size_t        held = 0;
      for(const W::Quad& c : W::children(parent))
        held += W::roadMeshes(w, c, centre).bridge.indices.size();
      EXPECT_EQ(held, W::roadMeshes(w, parent, centre).bridge.indices.size());
    }
  }
}

// A road's surface on its asphalt's maps: from one edge of the carriageway to the other across them,
// along them by the metre, each piece on from the last by whole repeats, and tinted from place to place
TEST(WorldRoads, ItsSurfaceIsLaidOnItsAsphaltFromEdgeToEdge)
{
  const W::World&  w      = settled();
  const W::Places& pl     = places();
  const uint8_t    finest = W::finestLevel(w);
  float            longest = 0.0F;
  for(const W::RoadPiece& p : pl.roads)
    longest = std::max(longest, p.length);
  std::vector<W::Quad> quads;
  for(size_t i = 0; i < pl.roads.size() && quads.size() < 8; i += 31)
  {
    const W::RoadPiece& p = pl.roads[i];
    if(!catalogue()->roads[p.kind].earth && !W::bridgeAt(pl, p.way, double(p.start) + 0.5 * double(p.length), 40.0))
      quads.push_back(quadOf(glm::normalize(p.b), finest));
  }
  ASSERT_GE(quads.size(), 4U);
  size_t joins = 0;
  float  least = 1e9F, most = -1e9F;
  for(const W::Quad& q : quads)
  {
    const glm::dvec3    centre = W::toScene(w, W::direction(q.face, (double(q.x) + 0.5) * 2.0 / double(1U << q.level) - 1.0, (double(q.y) + 0.5) * 2.0 / double(1U << q.level) - 1.0), 0.0);
    const W::RoadMeshes m      = W::roadMeshes(w, q, centre);
    ASSERT_FALSE(m.surface.empty());
    ASSERT_EQ(m.surface.colours.size(), m.surface.positions.size());
    ASSERT_EQ(m.marking.uvs.size(), m.marking.positions.size());
    ASSERT_EQ(m.marking.colours.size(), m.marking.positions.size());
    EXPECT_TRUE(m.bridge.colours.empty());
    // a section's seven vertices: a skirt, the carriageway's five columns from edge to edge, a skirt;
    // all of it at one place along
    ASSERT_EQ(m.surface.positions.size() % 7, 0U);
    struct Across
    {
      glm::vec3 left, middle, right;
      float     along;
    };
    std::vector<Across> sections;
    const auto atJoint = [](float v) {
      return std::any_of(W::kAsphaltJoints.begin(), W::kAsphaltJoints.end(), [&](double j) { return std::abs(double(v) * W::kAsphaltRepeat - j) < 2e-3; });
    };
    size_t ragged = 0;
    for(size_t s = 0; s < m.surface.positions.size(); s += 7)
    {
      const std::vector<glm::vec2>& uv = m.surface.uvs;
      // either way across: one edge at 0, the other at 1, its skirts past them
      const bool turned = uv[s + 1].x > 0.5F;
      EXPECT_NEAR(uv[s + 1].x, turned ? 1.0F : 0.0F, 1e-5F);
      EXPECT_NEAR(uv[s + 3].x, 0.5F, 1e-5F);
      EXPECT_NEAR(uv[s + 5].x, turned ? 0.0F : 1.0F, 1e-5F);
      EXPECT_TRUE(turned ? uv[s].x > 1.0F && uv[s + 6].x < 0.0F : uv[s].x < 0.0F && uv[s + 6].x > 1.0F);
      // on one repeat of the maps (a little past it at a broken end), all of it at one place along
      // (but at a broken end)
      float lo = uv[s].y, hi = uv[s].y;
      for(size_t k = 1; k < 7; k++)
      {
        lo = std::min(lo, uv[s + k].y);
        hi = std::max(hi, uv[s + k].y);
      }
      EXPECT_GE(lo, -0.04F);
      EXPECT_LE(hi, 1.04F);
      EXPECT_LT(hi - lo, 0.7F / float(W::kAsphaltRepeat));
      ragged += hi - lo > 1e-5F ? 1 : 0;
      sections.push_back({m.surface.positions[s + 1], m.surface.positions[s + 3], m.surface.positions[s + 5], uv[s + 3].y});
    }
    // a section twice, edge to edge: where one piece of a way gives way to the next (the same place on
    // the maps) or one stretch of the maps to another (each at one of their joints)
    for(size_t a = 0; a < sections.size(); a++)
      for(size_t b = a + 1; b < sections.size(); b++)
        if(glm::length(sections[a].middle - sections[b].middle) < 1e-3F && glm::length(sections[a].left - sections[b].left) < 1e-3F
           && glm::length(sections[a].right - sections[b].right) < 1e-3F)
        {
          const bool same = std::abs(sections[a].along - sections[b].along) < 1e-5F;
          EXPECT_TRUE(same || (atJoint(sections[a].along) && atJoint(sections[b].along))) << sections[a].along << " and " << sections[b].along;
          joins++;
        }
    (void)ragged;
    // the paint on a line: an edge's, or the middle's
    for(const glm::vec2& uv : m.marking.uvs)
    {
      const float x = std::min(uv.x, 1.0F - uv.x);
      EXPECT_TRUE((x > 0.04F && x < 0.08F) || x > 0.48F) << uv.x;
    }
    for(const glm::vec4& c : m.surface.colours)
    {
      EXPECT_GT(c.y, 0.6F);
      EXPECT_LT(c.y, 1.3F);
      EXPECT_EQ(c.w, 1.0F);
      least = std::min(least, c.y);
      most  = std::max(most, c.y);
    }
  }
  EXPECT_GT(joins, 0U) << "stretches of the maps meet on these quads";
  EXPECT_GT(most - least, 0.05F) << "the tint varies";
  (void)longest;
}

// Each end of an asphalt way: an open one stops the road square, nothing of its surface on the ground
// past it and its own edge going down into the ground; one that joins a road from the side stops at
// that road's edge, not over it
TEST(WorldRoads, AnOpenEndStopsItAndASideRoadStopsAtTheEdgeOfTheRoadItJoins)
{
  const W::World&  w      = settled();
  const W::Places& pl     = places();
  const double     R      = w.spec.radius;
  const uint8_t    finest = W::finestLevel(w);
  ASSERT_EQ(pl.wayEnds.size(), size_t(pl.ways));
  // the middle lines of the asphalt roads, as points a metre apart, by way
  std::vector<std::pair<uint32_t, glm::dvec3>> lines;
  for(const W::RoadPiece& p : pl.roads)
    if(!catalogue()->roads[p.kind].earth)
      for(int k = 0, n = std::max(2, int(p.length)); k <= n; k++)
      {
        const double t = double(k) / n, s = 1.0 - t;
        lines.emplace_back(p.way, glm::normalize(p.a * (s * s) + p.b * (2.0 * s * t) + p.c * (t * t)));
      }
  const auto awayFrom = [&](uint32_t way, const glm::dvec3& d) {
    double best = 1e30;
    for(const auto& [other, at] : lines)
      if(other != way)
        best = std::min(best, glm::length(at - d) * R);
    return best;
  };
  size_t open = 0, side = 0, ground = 0, skirted = 0, looked = 0;
  for(uint32_t way = 0; way < pl.ways; way++)
  {
    if(pl.wayPieces[way] == pl.wayPieces[way + 1] || catalogue()->roads[pl.roads[pl.wayPieces[way]].kind].earth)
      continue;
    const double length = W::wayLength(pl, way);
    for(int end = 0; end < 2; end++)
    {
      const W::WayEnd& e = pl.wayEnds[way][size_t(end)];
      const double     half = double(pl.roads[pl.wayPieces[way]].half);
      if(e.cut > 0.0F)
      {
        side++;
        // where its surface stops: at the edge of the road it joins, neither over it nor short of it
        const W::OnWay stop = W::onWay(pl, way, end ? length - double(e.cut) : double(e.cut));
        const double   away = awayFrom(way, stop.direction);
        EXPECT_GT(away, half - 0.3) << "way " << way << " end " << end;
        EXPECT_LT(away, 2.0 * half + 1.0) << "way " << way << " end " << end;
      }
      if(!e.open)
        continue;
      open++;
      EXPECT_GT(awayFrom(way, W::onWay(pl, way, end ? length : 0.0).direction), half) << "nothing of another road at an open end";
      // past it, nothing of its surface on the ground; before it, all of it
      const W::OnWay   at  = W::onWay(pl, way, end ? length : 0.0);
      const glm::dvec3 out = at.forward * (end ? 1.0 : -1.0);
      const W::Sample  past = W::sample(w, glm::normalize(at.direction + out * (2.5 / R)), 1.0);
      const W::Sample  in   = W::sample(w, W::onWay(pl, way, end ? length - 3.0 : 3.0).direction, 1.0);
      if(in.shore > 10.0F && in.built.road > 0.9F)
        ground += past.built.road < 0.05F ? 1 : 0;
      else
        ground++;  // (by water, or where a track runs on)
      // its surface's edge going down into the ground there
      if(looked < 6)
      {
        const W::Quad       q      = quadOf(W::onWay(pl, way, end ? length - 0.5 : 0.5).direction, finest);
        const glm::dvec3    centre = W::toScene(w, at.direction, 0.0);
        const W::RoadMeshes m      = W::roadMeshes(w, q, centre);
        bool                under  = false;
        for(const glm::vec3& v : m.surface.positions)
        {
          double           r = 0.0;
          const glm::dvec3 d = W::toPlanet(w, centre + glm::dvec3(v), &r);
          if(glm::length(d - at.direction) * R < 1.0)
            under = under || r - R < W::sample(w, d, 1.0).elevation - 0.02;
        }
        looked++;
        skirted += under ? 1 : 0;
      }
    }
  }
  EXPECT_GT(open, 0U);
  EXPECT_GT(side, 0U);
  EXPECT_GE(ground, open * 9 / 10) << "of " << open;
  EXPECT_GE(skirted, looked * 2 / 3) << "of " << looked;
}

// The asphalt's maps: a seed's every time, another seed's others, and each of them going on round
// its repeat as it goes on from one row to the next
TEST(WorldRoads, TheAsphaltsMapsAreASeedsAndGoOnRoundTheirRepeat)
{
  const W::AsphaltSize size{.across = 128, .grain = 64, .paint = 32};
  const W::AsphaltMaps a = W::asphaltMaps(1, size), b = W::asphaltMaps(1, size), c = W::asphaltMaps(2, size);
  const int            along = 128 * int(W::kAsphaltRepeat / W::kAsphaltWidth);
  for(const auto& [map, again, other, width, height, channels] :
      {std::tuple{&a.colour, &b.colour, &c.colour, 128, along, 4}, std::tuple{&a.normal, &b.normal, &c.normal, 128, along, 2},
       std::tuple{&a.rough, &b.rough, &c.rough, 128, along, 2}, std::tuple{&a.grain, &b.grain, &c.grain, 64, 64, 2},
       std::tuple{&a.paint, &b.paint, &c.paint, 32, along, 4}})
  {
    EXPECT_EQ(map->width, width);
    EXPECT_EQ(map->height, height);
    EXPECT_EQ(map->channels, channels);
    ASSERT_EQ(map->pixels.size(), size_t(width) * size_t(height) * size_t(channels));
    EXPECT_EQ(map->pixels, again->pixels);
    EXPECT_NE(map->pixels, other->pixels);
    // the last row on into the first, as a row into the next
    const auto step = [&](int from, int to) {
      double sum = 0.0;
      for(int i = 0; i < width * channels; i++)
        sum += std::abs(int(map->pixels[size_t(from) * size_t(width * channels) + size_t(i)]) - int(map->pixels[size_t(to) * size_t(width * channels) + size_t(i)]));
      return sum / double(width * channels);
    };
    double typical = 0.0;
    for(int j = 0; j + 1 < height; j++)
      typical += step(j, j + 1) / double(height - 1);
    EXPECT_LT(step(height - 1, 0), 2.0 * typical + 1.0) << width << " by " << height;
  }
  // the grain, across too
  double typical = 0.0, round = 0.0;
  for(int j = 0; j < 64; j++)
    for(int k = 0; k < 2; k++)
    {
      const auto at = [&](int i) { return int(a.grain.pixels[(size_t(j) * 64U + size_t(i)) * 2U + size_t(k)]); };
      for(int i = 0; i + 1 < 64; i++)
        typical += std::abs(at(i) - at(i + 1)) / (63.0 * 128.0);
      round += std::abs(at(63) - at(0)) / 128.0;
    }
  EXPECT_LT(round, 2.0 * typical + 1.0);
}

// The asphalt as an old road is: smoother in the wheels' paths than between them, browner at its
// edges than in its middle; its colours those of asphalt, stone, soil and tar; its normals leaning
// every way alike; its stones' grain in relief; and its edge lines' paint worn through in places
TEST(WorldRoads, TheAsphaltIsWornWhereTheWheelsWentAndDustyAtItsEdges)
{
  const W::AsphaltMaps a = W::asphaltMaps(1, {.across = 256, .grain = 1024, .paint = 64});  // (the grain as drawn: its stones a few texels across)
  const int            across = a.colour.width, along = a.colour.height;
  const auto           linear = [](uint8_t s) {
    const double v = double(s) / 255.0;
    return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
  };
  std::vector<double> rough(static_cast<size_t>(across), 0.0), red(static_cast<size_t>(across), 0.0), blue(static_cast<size_t>(across), 0.0);
  double              grey = 0.0, leanX = 0.0, leanY = 0.0;
  for(int j = 0; j < along; j++)
    for(int i = 0; i < across; i++)
    {
      const size_t at = size_t(j) * size_t(across) + size_t(i);
      const double r = linear(a.colour.pixels[at * 4U]), g = linear(a.colour.pixels[at * 4U + 1U]), b = linear(a.colour.pixels[at * 4U + 2U]);
      ASSERT_GT(std::min({r, g, b}), 0.005) << i << ", " << j;
      ASSERT_LT(std::max({r, g, b}), 0.5) << i << ", " << j;
      rough[size_t(i)] += double(a.rough.pixels[at * 2U + 1U]) / 255.0 / double(along);
      red[size_t(i)] += r / double(along);
      blue[size_t(i)] += b / double(along);
      grey += g / double(along * across);
      leanX += (double(a.normal.pixels[at * 2U]) / 127.5 - 1.0) / double(along * across);
      leanY += (double(a.normal.pixels[at * 2U + 1U]) / 127.5 - 1.0) / double(along * across);
    }
  const auto band = [&](const std::vector<double>& v, double x0, double x1) {
    double sum = 0.0;
    int    n   = 0;
    for(int i = 0; i < across; i++)
      if(const double x = (double(i) + 0.5) * W::kAsphaltWidth / double(across); x >= x0 && x <= x1)
      {
        sum += v[size_t(i)];
        n++;
      }
    return sum / double(std::max(n, 1));
  };
  for(const double wheel : {0.7, 2.3, 3.7, 5.3})
    for(const double between : {1.5, 4.5})
      EXPECT_LT(band(rough, wheel - 0.1, wheel + 0.1), band(rough, between - 0.1, between + 0.1) - 0.04) << wheel << " against " << between;
  const double middle = band(red, 2.0, 4.0) / band(blue, 2.0, 4.0);
  EXPECT_GT(band(red, 0.0, 0.25) / band(blue, 0.0, 0.25), middle + 0.1);
  EXPECT_GT(band(red, W::kAsphaltWidth - 0.25, W::kAsphaltWidth) / band(blue, W::kAsphaltWidth - 0.25, W::kAsphaltWidth), middle + 0.1);
  EXPECT_GT(grey, 0.06);
  EXPECT_LT(grey, 0.2);
  // its joints: a crack (or tar) from one edge to the other, where a road may go on to another stretch
  for(size_t j = 0; j + 1 < W::kAsphaltJoints.size(); j++)
  {
    const int row = int(std::lround(W::kAsphaltJoints[j] / W::kAsphaltWidth * double(across)));
    double    darkest = 1e9;
    for(int r = row - 1; r <= row; r++)
    {
      double mean = 0.0;
      for(int i = 0; i < across; i++)
        mean += linear(a.colour.pixels[(size_t((r + along) % along) * size_t(across) + size_t(i)) * 4U + 1U]) / double(across);
      darkest = std::min(darkest, mean);
    }
    EXPECT_LT(darkest, 0.75 * grey) << "the joint at " << W::kAsphaltJoints[j] << " m";
  }
  EXPECT_LT(std::abs(leanX), 0.03);
  EXPECT_LT(std::abs(leanY), 0.03);
  // the grain: stones standing out of their sand
  double spread = 0.0;
  for(size_t t = 0; t < a.grain.pixels.size(); t += 2)
    spread += std::pow(double(a.grain.pixels[t]) / 127.5 - 1.0, 2.0) / double(a.grain.pixels.size() / 2);
  EXPECT_GT(std::sqrt(spread), 0.1);
  // the left edge line's paint, 0.35 m in: whole along most of it, worn through in places
  size_t whole = 0, there = 0;
  for(int j = 0; j < along; j++)
    for(int i = 0; i < a.paint.width; i++)
      if(const double x = (double(i) + 0.5) * W::kAsphaltWidth / double(a.paint.width); x > 0.3 && x < 0.4)
      {
        there++;
        whole += linear(a.paint.pixels[(size_t(j) * size_t(a.paint.width) + size_t(i)) * 4U + 1U]) > 0.25 ? 1 : 0;
      }
  ASSERT_GT(there, 0U);
  EXPECT_GT(whole, there * 4 / 10);
  EXPECT_LT(whole, there * 98 / 100);
}

TEST(WorldRoads, ATracksSurfaceIsTheGroundsEarth)
{
  const W::World&  w      = settled();
  const W::Places& pl     = places();
  const uint8_t    finest = W::finestLevel(w);
  size_t           asked  = 0;
  for(size_t i = 0; i < pl.roads.size() && asked < 6; i += 13)
  {
    const W::RoadPiece& p = pl.roads[i];
    if(!catalogue()->roads[p.kind].earth || W::bridgeAt(pl, p.way, double(p.start) + 0.5 * double(p.length), 300.0))
      continue;
    // (none of a road's near)
    const glm::dvec3 d = glm::normalize(p.b);
    if(std::any_of(pl.roads.begin(), pl.roads.end(), [&](const W::RoadPiece& o) { return !catalogue()->roads[o.kind].earth && glm::length(o.b - d) * w.spec.radius < 400.0; }))
      continue;
    asked++;
    const W::RoadMeshes m = W::roadMeshes(w, quadOf(d, finest), W::toScene(w, d, 0.0));
    EXPECT_TRUE(m.surface.empty());
    EXPECT_TRUE(m.marking.empty());
  }
  EXPECT_GE(asked, 3U);
}

