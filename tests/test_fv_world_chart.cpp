// A map of a planet (src/fv_world_chart.hpp): a pixel is where its latitude and longitude are, also
// past ±180; the sea is drawn as sea and the land as land; the rivers and the roads over them; the
// same image every time, and none once stopped.

#include <cmath>

#include <gtest/gtest.h>

#include "../src/fv_world_chart.hpp"
#include "../src/fv_world_map.hpp"
#include "../src/fv_world_places.hpp"

namespace {

namespace W = fv::world;
using Json  = nlohmann::json;

// A small planet, and the same with places and roads on it
const W::World& small()
{
  static const W::World w = W::make({.seed = 7, .radius = 5000.0}, {.map = 128});
  return w;
}

const W::World& settled()
{
  static const W::World w = [] {
    std::vector<std::string> rows(40, std::string(40, '0'));
    for(int j = 10; j < 30; j++)
      for(int i = 10; i < 30; i++)
        rows[size_t(j)][size_t(i)] = '3';
    const Json pack = {{"name", "yard"},
                       {"kind", "yard"},
                       {"radius", 20},
                       {"pieces", Json::array({{{"file", "yard.gltf"}, {"copies", Json::array({Json::array({0.0, 0.0, 0.0, 0.0})})}}})},
                       {"gates", Json::array({{{"at", {0.0, 16.0}}, {"heading", {0.0, 1.0}}}})},
                       {"clear", {{"cell", 1.0}, {"n", 40}, {"rows", rows}}}};
    const Json anywhere = {{"snow", 0.3}, {"tundra", 1.0}, {"desert", 1.0}, {"steppe", 1.0}, {"grassland", 1.0}, {"forest", 1.0}};
    const auto catalogue =
        W::readCatalogue({{"version", 1},
                          {"packs", Json::array({pack})},
                          {"rules",
                           {{"kinds", Json::array({{{"kind", "yard"}, {"road", "road"}, {"slope", 0.15}, {"spacing", 900}, {"biomes", anywhere}}})},
                            {"roads", Json::array({{{"name", "road"}, {"width", 6.0}, {"verge", 3.0}, {"colour", {0.9, 0.1, 0.1}}, {"surface", "asphalt"}}})}}}});
    return W::withPlaces(small(), W::settle(small(), catalogue));
  }();
  return w;
}

const uint8_t* pixel(const std::vector<uint8_t>& image, const W::ChartView& v, int x, int y)
{
  return &image[(size_t(y) * size_t(v.width) + size_t(x)) * 3];
}

}  // namespace

TEST(WorldChart, APixelIsWhereItsLatitudeAndLongitudeAre)
{
  for(const glm::dvec2 ll : {glm::dvec2(0.0, 0.0), glm::dvec2(30.0, 14.0), glm::dvec2(-55.5, -110.0), glm::dvec2(80.0, 179.0)})
  {
    const glm::dvec2 back = W::latitudeLongitudeOf(W::directionOf(ll.x, ll.y));
    EXPECT_NEAR(back.x, ll.x, 1e-9);
    EXPECT_NEAR(back.y, ll.y, 1e-9);
  }
  const W::ChartView v{.west = 10.0, .east = 20.0, .south = 25.0, .north = 30.0, .width = 200, .height = 100};
  // the window's corners, and a pixel's middle there and back
  EXPECT_NEAR(glm::distance(W::chartDirection(v, 0.0, 0.0), W::directionOf(30.0, 10.0)), 0.0, 1e-12);
  EXPECT_NEAR(glm::distance(W::chartDirection(v, 200.0, 100.0), W::directionOf(25.0, 20.0)), 0.0, 1e-12);
  const glm::dvec2 p = W::chartPoint(v, W::chartDirection(v, 37.5, 81.5));
  EXPECT_NEAR(p.x, 37.5, 1e-9);
  EXPECT_NEAR(p.y, 81.5, 1e-9);
  // outside it: off its edges
  EXPECT_LT(W::chartPoint(v, W::directionOf(27.0, 5.0)).x, 0.0);
  EXPECT_GT(W::chartPoint(v, W::directionOf(20.0, 15.0)).y, 100.0);
  // a window across ±180 takes the longitude round
  const W::ChartView round{.west = 170.0, .east = 200.0, .south = -10.0, .north = 10.0, .width = 300, .height = 200};
  EXPECT_NEAR(W::chartPoint(round, W::directionOf(0.0, -175.0)).x, 150.0, 1e-9);
  EXPECT_NEAR(W::chartPoint(round, W::directionOf(0.0, 175.0)).x, 50.0, 1e-9);
}

TEST(WorldChart, TheSeaIsDrawnAsSeaAndTheLandAsLand)
{
  const W::World&      w = small();
  const W::ChartView   v{.width = 256, .height = 128, .rivers = false, .roads = false};
  const auto           image = W::chartImage(w, v);
  ASSERT_EQ(image.size(), size_t(256 * 128 * 3));
  int sea = 0, land = 0, agreed = 0;
  for(int y = 0; y < v.height; y++)
    for(int x = 0; x < v.width; x++)
    {
      const W::Sample s       = W::sample(w, W::chartDirection(v, x + 0.5, y + 0.5), W::chartPixel(w, v));
      const uint8_t*  c       = pixel(image, v, x, y);
      const bool      wet     = s.elevation < 0.0;
      const bool      drawnAs = c[2] > c[0] + 20 && c[2] > c[1];  // (the sea's blue)
      (wet ? sea : land)++;
      agreed += wet == drawnAs || (!wet && s.water > s.elevation) ? 1 : 0;
    }
  EXPECT_GT(sea, 1000);
  EXPECT_GT(land, 1000);
  EXPECT_GT(agreed, int(0.98 * v.width * v.height));
  // the same every time
  EXPECT_EQ(W::chartImage(w, v), image);
  // and none once stopped
  const std::atomic<bool> stop{true};
  EXPECT_TRUE(W::chartImage(w, v, &stop).empty());
}

TEST(WorldChart, TheRiversAndRoadsAreDrawnOverIt)
{
  const W::World& w = settled();
  ASSERT_TRUE(w.places);
  ASSERT_FALSE(w.places->roads.empty());
  ASSERT_TRUE(w.base);
  ASSERT_FALSE(w.base->pieces.empty());
  // a window round the middle of a road, a few metres a pixel
  const W::RoadPiece& road = w.places->roads[w.places->roads.size() / 2];
  const glm::dvec2    at   = W::latitudeLongitudeOf(road.b);
  const double        span = 600.0 / w.spec.radius / (3.14159265358979323846 / 180.0);  // 600 m, in degrees
  W::ChartView        v{.west = at.y - span, .east = at.y + span, .south = at.x - span, .north = at.x + span, .width = 120, .height = 120};
  v.roads           = false;
  const auto bare   = W::chartImage(w, v);
  v.roads           = true;
  const auto roads  = W::chartImage(w, v);
  const glm::dvec2 p = W::chartPoint(v, road.b);
  const uint8_t*   c = pixel(roads, v, int(p.x), int(p.y));
  EXPECT_GT(c[0], 200) << "the road's colour, a bright red";
  EXPECT_LT(c[1], 120);
  EXPECT_NE(roads, bare);

  // a river's middle drawn as river, with the rivers and not without
  const W::RiverPiece& river = w.base->pieces[w.base->pieces.size() / 2];
  const glm::dvec3     mid   = W::riverMiddleAt(river, 0.5, w.spec.radius);
  const glm::dvec2     rl    = W::latitudeLongitudeOf(mid);
  W::ChartView         r{.west = rl.y - span, .east = rl.y + span, .south = rl.x - span, .north = rl.x + span, .width = 120, .height = 120};
  r.roads                = false;
  const auto      with   = W::chartImage(w, r);
  r.rivers               = false;
  const auto      without = W::chartImage(w, r);
  const glm::dvec2 q      = W::chartPoint(r, mid);
  const uint8_t*  wet     = pixel(with, r, int(q.x), int(q.y));
  EXPECT_GT(wet[2], wet[0] + 20) << "a river's blue";
  EXPECT_NE(with, without);
}
