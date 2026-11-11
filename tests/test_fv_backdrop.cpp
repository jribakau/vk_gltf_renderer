// The backdrop's ground and planting (src/fv_backdrop.hpp): flat where the figures stand, the same for
// a seed every time, a mesh that is the height function's, and plants that stand on it where their
// layer says, as thickly as it says.

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>

#include <gtest/gtest.h>

#include "../src/fv_backdrop.hpp"

namespace {

namespace B = fv::backdrop;

B::Spec meadow(int seed = 1)
{
  return {.kind = "meadow", .seed = seed};
}

B::Spec forest(int seed = 1)
{
  return {.kind = "forest", .seed = seed};
}

// How many of a layer's plants stand between two distances of the origin, and their mean scale
std::pair<size_t, float> ring(const B::Spec& spec, const B::Layer& layer, uint32_t index, float from, float to)
{
  size_t n     = 0;
  float  scale = 0.0F;
  for(const auto& variant : B::scatter(spec, layer, index, 2))
    for(const glm::mat4& m : variant)
      if(const float d = glm::length(glm::vec2(m[3].x, m[3].z)); d >= from && d < to)
      {
        n++;
        scale += glm::length(glm::vec3(m[1]));
      }
  return {n, n ? scale / float(n) : 0.0F};
}

size_t total(const std::vector<std::vector<glm::mat4>>& placed)
{
  size_t n = 0;
  for(const auto& v : placed)
    n += v.size();
  return n;
}

}  // namespace

TEST(BackdropGround, IsFlatWhereTheFiguresStandAndRollsBeyond)
{
  const B::Spec spec = meadow();
  for(float a = 0.0F; a < 6.28F; a += 0.3F)
    for(float r : {0.0F, 1.0F, 3.0F, B::kClearing})
      EXPECT_EQ(B::height(spec, r * std::cos(a), r * std::sin(a)), 0.0F) << r;
  // beyond the clearing it is not all one height, and near it stays within a person's height
  float lo = 1e9F, hi = -1e9F;
  for(float x = -60.0F; x <= 60.0F; x += 1.5F)
    for(float z = -60.0F; z <= 60.0F; z += 1.5F)
    {
      const float h = B::height(spec, x, z);
      lo            = std::min(lo, h);
      hi            = std::max(hi, h);
    }
  EXPECT_GT(hi - lo, 0.5F);
  EXPECT_LT(hi, 6.0F);
  EXPECT_GT(lo, -3.0F);
  // the same every time, another for another seed, and none without relief or without a backdrop
  EXPECT_EQ(B::height(spec, 40.0F, -25.0F), B::height(meadow(), 40.0F, -25.0F));
  EXPECT_NE(B::height(spec, 40.0F, -25.0F), B::height(meadow(2), 40.0F, -25.0F));
  B::Spec flat = spec;
  flat.relief  = 0.0;
  EXPECT_EQ(B::height(flat, 40.0F, -25.0F), 0.0F);
  EXPECT_EQ(B::height(B::Spec{}, 40.0F, -25.0F), 0.0F);
  // twice the relief, twice the height
  B::Spec steep = spec;
  steep.relief  = 2.0;
  EXPECT_FLOAT_EQ(B::height(steep, 40.0F, -25.0F), 2.0F * B::height(spec, 40.0F, -25.0F));
  // the far ground rises: the hills are the horizon's
  float far = 0.0F;
  for(float a = 0.0F; a < 6.28F; a += 0.1F)
    far += B::height(spec, 800.0F * std::cos(a), 800.0F * std::sin(a));
  EXPECT_GT(far / 63.0F, 10.0F);
}

TEST(BackdropGround, TheMeshIsTheHeightFunctions)
{
  const B::Spec    spec = meadow(3);
  const B::Terrain t    = B::terrain(spec, 2.0F);
  ASSERT_FALSE(t.positions.empty());
  ASSERT_EQ(t.normals.size(), t.positions.size());
  ASSERT_EQ(t.uvs.size(), t.positions.size());
  ASSERT_EQ(t.shades.size(), t.positions.size());
  ASSERT_EQ(t.indices.size() % 3, 0u);
  // drier drifts and greener ground both, so the distance isn't one colour
  const auto [driest, greenest] = std::minmax_element(t.shades.begin(), t.shades.end());
  EXPECT_GE(*driest, 0.0F);
  EXPECT_LE(*greenest, 1.0F);
  EXPECT_LT(*driest, 0.1F);
  EXPECT_GT(*greenest, 0.9F);
  float reach = 0.0F;
  for(size_t i = 0; i < t.positions.size(); i++)
  {
    const glm::vec3& p = t.positions[i];
    EXPECT_FLOAT_EQ(p.y, B::height(spec, p.x, p.z));
    EXPECT_NEAR(glm::length(t.normals[i]), 1.0F, 1e-4F);
    EXPECT_GT(t.normals[i].y, 0.3F);
    EXPECT_FLOAT_EQ(t.uvs[i].x, p.x / 2.0F);
    reach = std::max(reach, glm::length(glm::vec2(p.x, p.z)));
  }
  EXPECT_NEAR(reach, B::kHorizon, 1.0F);
  // every triangle faces up, and none is degenerate
  for(size_t i = 0; i < t.indices.size(); i += 3)
  {
    ASSERT_LT(t.indices[i], t.positions.size());
    const glm::vec3 a = t.positions[t.indices[i]], b = t.positions[t.indices[i + 1]], c = t.positions[t.indices[i + 2]];
    const glm::vec3 n = glm::cross(b - a, c - a);
    ASSERT_GT(n.y, 0.0F) << "triangle " << i / 3;
  }
  // light enough to make on a click
  EXPECT_LT(t.indices.size() / 3, 100000u);
}

TEST(BackdropScatter, PlantsStandOnTheGroundWhereTheirLayerSays)
{
  const B::Spec  spec = meadow();
  const B::Layer layer{.role = "sapling", .perSquareMetre = 0.5, .from = 7.0F, .to = 40.0F, .scaleMin = 0.7F, .scaleMax = 1.6F, .sink = 0.1F};
  const auto     placed = B::scatter(spec, layer, 2, 3);
  ASSERT_EQ(placed.size(), 3u);
  for(const auto& variant : placed)
  {
    EXPECT_GT(variant.size(), 100u) << "every variant is used";
    for(const glm::mat4& m : variant)
    {
      const glm::vec3 at = glm::vec3(m[3]);
      const float     d  = glm::length(glm::vec2(at.x, at.z));
      EXPECT_GE(d, layer.from);
      EXPECT_LE(d, layer.to);
      const float scale = glm::length(glm::vec3(m[1]));  // upright: its y axis is the vertical, scaled
      EXPECT_GE(scale, 0.7F - 1e-4F);
      EXPECT_LE(scale, 1.6F + 1e-4F);
      EXPECT_NEAR(m[1].y, scale, 1e-4F);
      EXPECT_NEAR(at.y, B::height(spec, at.x, at.z) - 0.1F * scale, 1e-4F);
    }
  }
  // as many as the layer's density over its ring, about
  const double area = 3.14159265 * (40.0 * 40.0 - 7.0 * 7.0);
  EXPECT_NEAR(double(total(placed)), 0.5 * area, 0.06 * 0.5 * area);
  EXPECT_NEAR(double(B::expectedCount(spec, layer)), 0.5 * area, 0.03 * 0.5 * area);
  // the same for the same spec; other places for another seed or another layer; twice as many at twice the density
  EXPECT_EQ(B::scatter(spec, layer, 2, 3), placed);
  EXPECT_NE(B::scatter(meadow(2), layer, 2, 3), placed);
  EXPECT_NE(B::scatter(spec, layer, 3, 3), placed);
  B::Spec dense = spec;
  dense.density = 2.0;
  EXPECT_NEAR(double(total(B::scatter(dense, layer, 2, 3))), area, 0.06 * area);
  // nothing to plant: no variants, no room
  EXPECT_TRUE(B::scatter(spec, layer, 2, 0).empty());
  B::Layer none = layer;
  none.to       = none.from;
  EXPECT_EQ(total(B::scatter(spec, none, 2, 3)), 0u);
}

TEST(BackdropScatter, ALayersCoverIsItsShareOfTheGround)
{
  for(const float cover : {0.25F, 0.5F, 0.75F})
  {
    size_t placed = 0, even = 0;
    for(int seed = 1; seed <= 4; seed++)
    {
      const B::Layer patchy{.role = "flower", .perSquareMetre = 2.0, .from = 0.0F, .to = 120.0F, .cover = cover, .patch = 9.0F};
      B::Layer       whole = patchy;
      whole.cover          = 1.0F;
      placed += total(B::scatter(meadow(seed), patchy, 1, 1));
      even += total(B::scatter(meadow(seed), whole, 1, 1));
    }
    EXPECT_NEAR(double(placed) / double(even), cover, 0.08) << cover;
  }
  // and filled in toward the far edge, where trees close the horizon
  const B::Layer woods{.role = "tree", .perSquareMetre = 0.05, .from = 20.0F, .to = 260.0F, .cover = 0.3F, .patch = 70.0F, .closeFar = 0.8F};
  size_t         near = 0, far = 0;
  for(int seed = 1; seed <= 4; seed++)
    for(const auto& variant : B::scatter(meadow(seed), woods, 3, 2))
      for(const glm::mat4& m : variant)
      {
        const float d = glm::length(glm::vec2(m[3].x, m[3].z));
        near += d < 100.0F;
        far += d > 220.0F;
      }
  const double nearArea = 3.14159265 * (100.0 * 100.0 - 20.0 * 20.0), farArea = 3.14159265 * (260.0 * 260.0 - 220.0 * 220.0);
  EXPECT_GT(double(far) / farArea, 1.8 * double(near) / nearArea);
  // or toward the near edge, where a layer carries a closed wood on and opens out of it
  const B::Layer beyond{.role = "tree", .perSquareMetre = 0.05, .from = 260.0F, .to = 900.0F, .cover = 0.3F, .patch = 70.0F, .closeNear = 0.8F};
  size_t         by = 0, out = 0;
  for(int seed = 1; seed <= 4; seed++)
  {
    by += ring(meadow(seed), beyond, 3, 260.0F, 300.0F).first;
    out += ring(meadow(seed), beyond, 3, 600.0F, 800.0F).first;
  }
  const double byArea = 3.14159265 * (300.0 * 300.0 - 260.0 * 260.0), outArea = 3.14159265 * (800.0 * 800.0 - 600.0 * 600.0);
  EXPECT_GT(double(by) / byArea, 1.8 * double(out) / outArea);
}

TEST(BackdropScatter, ALayerThinsAndGrowsWithDistance)
{
  const B::Layer grass{.role = "grass", .perSquareMetre = 40.0, .from = 0.0F, .to = 40.0F, .thin = 8.0F, .grow = 2.0F, .scaleMin = 1.0F, .scaleMax = 1.0F};
  size_t         near = 0, far = 0;
  float          nearScale = 0.0F, farScale = 0.0F;
  const auto     placed    = B::scatter(meadow(), grass, 0, 4);
  // the count foreseen is the count placed, about
  EXPECT_NEAR(double(B::expectedCount(meadow(), grass)), double(total(placed)), 0.05 * double(total(placed)));
  for(const auto& variant : placed)
    for(const glm::mat4& m : variant)
    {
      const float d = glm::length(glm::vec2(m[3].x, m[3].z));
      const float s = glm::length(glm::vec3(m[1]));
      if(d < 4.0F)
        near++, nearScale += s;
      else if(d > 30.0F)
        far++, farScale += s;
    }
  const double nearDensity = double(near) / (3.14159265 * 16.0), farDensity = double(far) / (3.14159265 * (1600.0 - 900.0));
  EXPECT_NEAR(nearDensity, 40.0 / 1.125, 4.0);  // 1 / (1 + (d/8)^2) averaged over the first 4 m
  EXPECT_LT(farDensity, nearDensity / 10.0);
  EXPECT_NEAR(nearScale / float(near), 1.06F, 0.05F);
  EXPECT_FLOAT_EQ(farScale / float(far), 2.0F);  // grown to its cap

  // a layer that carries another on thins from where that one ended: as thick and as small there
  const B::Layer on{.role = "tree", .perSquareMetre = 0.05, .from = 200.0F, .to = 800.0F, .thin = 200.0F, .grow = 2.0F, .thinFrom = 200.0F, .scaleMin = 1.0F, .scaleMax = 1.0F};
  const auto [first, firstScale] = ring(meadow(), on, 0, 200.0F, 240.0F);
  const auto [last, lastScale]   = ring(meadow(), on, 0, 700.0F, 800.0F);
  EXPECT_NEAR(double(first) / (3.14159265 * (240.0 * 240.0 - 200.0 * 200.0)), 0.05, 0.006);
  EXPECT_NEAR(firstScale, 1.0F, 0.02F);
  EXPECT_LT(double(last) / (3.14159265 * (800.0 * 800.0 - 700.0 * 700.0)), 0.05 / 6.0);
  EXPECT_FLOAT_EQ(lastScale, 2.0F);
  EXPECT_NEAR(double(B::expectedCount(meadow(), on)), double(ring(meadow(), on, 0, 0.0F, 1e9F).first), 0.05 * double(B::expectedCount(meadow(), on)));
}

TEST(BackdropScatter, ALayerFadesOutBeforeItsEdge)
{
  const B::Layer hard{.role = "grass", .perSquareMetre = 20.0, .from = 0.0F, .to = 30.0F, .scaleMin = 1.0F, .scaleMax = 1.0F};
  B::Layer       soft = hard;
  soft.fade           = 10.0F;
  const auto count = [](const B::Layer& layer, float from, float to) {
    size_t n = 0;
    for(const auto& variant : B::scatter(meadow(), layer, 0, 2))
      for(const glm::mat4& m : variant)
      {
        const float d = glm::length(glm::vec2(m[3].x, m[3].z));
        n += d >= from && d < to ? 1 : 0;
      }
    return n;
  };
  // the same plants up to where the fade starts, half of them through it, next to none at the edge
  EXPECT_EQ(count(soft, 0.0F, 20.0F), count(hard, 0.0F, 20.0F));
  EXPECT_NEAR(double(count(soft, 20.0F, 30.0F)), 0.5 * double(count(hard, 20.0F, 30.0F)), 0.06 * double(count(hard, 20.0F, 30.0F)));
  EXPECT_LT(count(soft, 29.0F, 30.0F) * 20, count(hard, 29.0F, 30.0F));
  EXPECT_NEAR(double(B::expectedCount(meadow(), soft)), double(count(soft, 0.0F, 30.0F)), 0.05 * double(count(soft, 0.0F, 30.0F)));
}

TEST(BackdropLayers, AMeadowsTreesAndSaplingsHaveTheirPlaces)
{
  EXPECT_TRUE(B::layers(B::Spec{}).empty());
  const auto            l = B::layers(meadow());
  std::set<std::string> roles;
  uint64_t              plants = 0, trees = 0;
  for(const B::Layer& layer : l)
  {
    roles.insert(layer.role);
    plants += B::expectedCount(meadow(), layer);
    trees += layer.role == "tree" ? B::expectedCount(meadow(), layer) : 0;
    // trunks and saplings stay clear of where the figures stand
    EXPECT_GT(layer.from, B::kClearing);
  }
  // the grass and the flowers are tiles
  EXPECT_EQ(roles, (std::set<std::string>{"sapling", "tree", "shrub", "rock"}));
  EXPECT_GT(plants, 1000u);
  EXPECT_LT(plants, 20000u);
  // mostly trees, and they stand out to the horizon: the wood round the meadow, then the hills'
  EXPECT_GT(trees, plants / 2);
  ASSERT_GE(l.size(), 3u);
  EXPECT_FLOAT_EQ(l[1].to, B::kWoods);
  EXPECT_FLOAT_EQ(l[2].from, B::kWoods);
  EXPECT_FLOAT_EQ(l[2].to, B::kHorizon);
  // as closed where the one ends as where the other begins: no ring shows from the air
  size_t inside = 0, outside = 0;
  for(int seed = 1; seed <= 4; seed++)
  {
    inside += ring(meadow(seed), l[1], 1, B::kWoods - 40.0F, B::kWoods).first;
    outside += ring(meadow(seed), l[2], 2, B::kWoods, B::kWoods + 40.0F).first;
  }
  const double in = double(inside) / (B::kWoods * B::kWoods - (B::kWoods - 40.0) * (B::kWoods - 40.0));
  const double out = double(outside) / ((B::kWoods + 40.0) * (B::kWoods + 40.0) - B::kWoods * B::kWoods);
  EXPECT_NEAR(out / in, 1.0, 0.25);
}

TEST(BackdropLayers, AForestIsAGladeInAWoodThatRunsToTheHorizon)
{
  const auto            l = B::layers(forest());
  std::set<std::string> roles;
  uint64_t              plants = 0;
  for(const B::Layer& layer : l)
  {
    roles.insert(layer.role);
    plants += B::expectedCount(forest(), layer);
    EXPECT_GT(layer.from, B::kClearing);
  }
  EXPECT_EQ(roles, (std::set<std::string>{"tree", "sapling", "shrub", "rock", "stump", "log"}));
  // what the instancing path carries without a thought: tens of thousands
  EXPECT_GT(plants, 20000u);
  EXPECT_LT(plants, 120000u);
  // the trees: close by the glade, a trunk every few metres, and still there at the horizon, fewer
  // and larger
  ASSERT_GE(l.size(), 2u);
  ASSERT_EQ(l[0].role, "tree");
  ASSERT_EQ(l[1].role, "tree");
  EXPECT_FLOAT_EQ(l[1].from, l[0].to);
  EXPECT_FLOAT_EQ(l[1].to, B::kHorizon);
  const auto [near, nearScale] = ring(forest(), l[0], 0, 10.0F, 60.0F);
  const auto [far, farScale]   = ring(forest(), l[1], 1, 700.0F, 900.0F);
  EXPECT_GT(double(near) / (3.14159265 * (60.0 * 60.0 - 10.0 * 10.0)), 0.015);
  EXPECT_GT(far, 500u);
  EXPECT_GT(farScale, 1.8F * nearScale);
  // more of them than a meadow's, whose trees are groves
  uint64_t meadows = 0, forests = 0;
  for(const B::Layer& layer : B::layers(meadow()))
    meadows += layer.role == "tree" ? B::expectedCount(meadow(), layer) : 0;
  for(const B::Layer& layer : l)
    forests += layer.role == "tree" ? B::expectedCount(forest(), layer) : 0;
  EXPECT_GT(forests, meadows * 6 / 5);
  // its grass is tufts, its drifts are ferns and reach further than a meadow's flowers
  EXPECT_LT(B::grassCovers(forest())[0].perSquareMetre, 0.25 * B::grassCovers(meadow())[0].perSquareMetre);
  EXPECT_EQ(B::drifts(forest()).covers[0].role, "fern");
  EXPECT_EQ(B::drifts(meadow()).covers[0].role, "flower");
  EXPECT_GT(B::drifts(forest()).reach, B::drifts(meadow()).reach);
  // the same ground under either
  EXPECT_EQ(B::height(forest(), 40.0F, -25.0F), B::height(meadow(), 40.0F, -25.0F));
  // what each asks of the kit
  EXPECT_EQ(B::roles(forest()), (std::vector<std::string>{"fern", "grass", "log", "rock", "sapling", "shrub", "stump", "tree"}));
  EXPECT_EQ(B::roles(meadow()), (std::vector<std::string>{"flower", "grass", "rock", "sapling", "shrub", "tree"}));
  EXPECT_TRUE(B::roles(B::Spec{}).empty());
}

TEST(BackdropTiles, APatchIsItsCoversPlantsWithinItsTile)
{
  const B::Spec spec   = meadow();
  const auto    covers = B::grassCovers(spec);
  for(size_t level = 0; level < covers.size(); level++)
  {
    const B::Cover& cover  = covers[level];
    const auto      placed = B::patch(spec, cover, 5);
    EXPECT_NEAR(double(total(placed)), cover.perSquareMetre * double(cover.tile) * double(cover.tile), 0.1 * double(total(placed)));
    for(const auto& variant : placed)
    {
      EXPECT_FALSE(variant.empty());
      for(const glm::mat4& m : variant)
      {
        EXPECT_LE(std::abs(m[3].x), 0.5F * cover.tile);
        EXPECT_LE(std::abs(m[3].z), 0.5F * cover.tile);
        EXPECT_EQ(m[3].y, 0.0F);
        const float scale = glm::length(glm::vec3(m[1]));
        EXPECT_GE(scale, cover.scaleMin - 1e-4F);
        EXPECT_LE(scale, cover.scaleMax + 1e-4F);
      }
    }
    // each level a tile twice the size with fewer, larger plants to the square metre
    if(level > 0)
    {
      EXPECT_FLOAT_EQ(cover.tile, 2.0F * covers[level - 1].tile);
      EXPECT_LT(cover.perSquareMetre, 0.4 * covers[level - 1].perSquareMetre);
      EXPECT_GT(cover.scaleMin, covers[level - 1].scaleMin);
    }
    // the same patch every time, and another for another seed
    EXPECT_EQ(B::patch(spec, cover, 5), placed);
    EXPECT_NE(B::patch(meadow(2), cover, 5), placed);
  }
  // a forest's grass stands in clumps: the same count to the tile, on about a third of it
  {
    const B::Cover tufts = B::grassCovers(forest())[1];
    B::Cover       even  = tufts;
    even.clumps          = 1.0F;
    ASSERT_LT(tufts.clumps, 0.5F);
    const auto in = B::patch(forest(), tufts, 1), over = B::patch(forest(), even, 1);
    EXPECT_NEAR(double(total(in)), double(total(over)), 0.1 * double(total(over)));
    // by the cells of a grid a sixth of the tile across that hold a plant
    const auto cells = [&](const std::vector<std::vector<glm::mat4>>& placed) {
      std::set<std::pair<int, int>> held;
      for(const glm::mat4& m : placed[0])
        held.insert({int(std::floor(m[3].x / tufts.tile * 6.0F)), int(std::floor(m[3].z / tufts.tile * 6.0F))});
      return held.size();
    };
    EXPECT_GT(cells(over), 28u);
    EXPECT_LT(cells(in), cells(over) * 7 / 10);
    EXPECT_EQ(B::patch(forest(), tufts, 1), in);
  }
  // thicker with the density
  B::Spec thick = spec;
  thick.density = 2.0;
  // (a whole number of plants a side: within a tenth)
  const double one = double(total(B::patch(spec, covers[0], 3)));
  EXPECT_NEAR(double(total(B::patch(thick, B::grassCovers(thick)[0], 3))), 2.0 * one, 0.2 * one);
}

namespace {

// The square a tile covers: its corner and its size
std::pair<glm::vec2, float> squareOf(const B::Spec& spec, const B::Tile& t)
{
  const float size = B::grassCovers(spec)[t.level].tile;
  return {{float(t.x) * size, float(t.z) * size}, size};
}

// The share of the disc `radius` round `at` the tiles cover, by points on a fine grid, and whether
// any point of it lies in two tiles
std::pair<double, bool> covered(const B::Spec& spec, const std::vector<B::Tile>& tiles, glm::vec2 at, float radius)
{
  size_t in = 0, all = 0;
  bool   twice = false;
  for(float z = at.y - radius; z <= at.y + radius; z += 0.5F)
    for(float x = at.x - radius; x <= at.x + radius; x += 0.5F)
    {
      if(glm::distance(glm::vec2(x, z), at) > radius)
        continue;
      all++;
      int n = 0;
      for(const B::Tile& t : tiles)
      {
        const auto [corner, size] = squareOf(spec, t);
        n += x >= corner.x && x < corner.x + size && z >= corner.y && z < corner.y + size ? 1 : 0;
      }
      in += n > 0 ? 1 : 0;
      twice = twice || n > 1;
    }
  return {double(in) / double(all), twice};
}

}  // namespace

TEST(BackdropTiles, TheGrassCoversTheGroundOnceThickWhereOneLooksFrom)
{
  const B::Spec spec  = meadow();
  const auto    reach = B::grassReach(spec);
  EXPECT_LT(reach[0], reach[1]);
  EXPECT_LT(reach[1], reach[2]);
  const glm::vec2              away(210.0F, -95.0F);
  const std::vector<glm::vec2> foci{{0.0F, 0.0F}, away};
  const std::vector<B::Tile>   tiles = B::grassTiles(spec, foci);
  // no tile twice, every level there
  EXPECT_EQ(std::set<B::Tile>(tiles.begin(), tiles.end()).size(), tiles.size());
  std::array<size_t, B::kGrassLevels> count{};
  for(const B::Tile& t : tiles)
  {
    ASSERT_LT(t.level, B::kGrassLevels);
    count[t.level]++;
    // a tile's level is by how far its ground is from where one looks from
    const auto [corner, size] = squareOf(spec, t);
    const glm::vec2 centre    = corner + glm::vec2(0.5F * size);
    const float     d         = std::min(glm::length(centre), glm::distance(centre, away));
    if(t.level == 0)
      EXPECT_LT(d, reach[0] + 8.0F);
    if(t.level == 2)
      EXPECT_GT(d, reach[1] - 8.0F);
    EXPECT_LT(d, reach[2] + 8.0F);
  }
  for(size_t level = 0; level < count.size(); level++)
    EXPECT_GT(count[level], 50u) << "level " << level;
  // a few thousand tiles in all: what a move of the camera has to look through
  EXPECT_LT(tiles.size(), 6000u);
  // round each focus: all of the ground within the thick grass's reach is covered, and none twice
  for(const glm::vec2& f : foci)
  {
    const auto [share, twice] = covered(spec, tiles, f, reach[1]);
    EXPECT_DOUBLE_EQ(share, 1.0);
    EXPECT_FALSE(twice);
    // and right where one stands the tiles are the thick ones
    for(const B::Tile& t : tiles)
      if(const auto [corner, size] = squareOf(spec, t); glm::distance(corner + glm::vec2(0.5F * size), f) < 0.5F * reach[0])
        EXPECT_EQ(t.level, 0) << t.x << "," << t.z;
  }
  // no focus given: round the origin
  EXPECT_EQ(B::grassTiles(spec, {}), B::grassTiles(spec, std::vector<glm::vec2>{{0.0F, 0.0F}}));
  EXPECT_TRUE(B::grassTiles(B::Spec{}, foci).empty());
}

TEST(BackdropTiles, ASmallMoveChangesAFewTilesAndNoneNear)
{
  const B::Spec spec  = meadow();
  const auto    reach = B::grassReach(spec);
  glm::vec2     at(3.0F, 7.0F);
  std::vector<B::Tile> tiles = B::grassTiles(spec, std::vector<glm::vec2>{at});
  size_t               most  = 0;
  // a walk of 60 m, a metre a step
  for(int step = 0; step < 60; step++)
  {
    at += glm::vec2(0.8F, 0.6F);
    const std::vector<B::Tile> next = B::grassTiles(spec, std::vector<glm::vec2>{at}, tiles);
    std::vector<B::Tile>       a = tiles, b = next, changed;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(changed));
    most = std::max(most, changed.size());
    // what came or went is out where the levels meet, never by the walker
    for(const B::Tile& t : changed)
    {
      const auto [corner, size] = squareOf(spec, t);
      EXPECT_GT(glm::distance(corner + glm::vec2(0.5F * size), at), 0.6F * reach[0]) << "step " << step;
    }
    const auto [share, twice] = covered(spec, next, at, reach[0]);
    EXPECT_DOUBLE_EQ(share, 1.0);
    EXPECT_FALSE(twice);
    tiles = next;
  }
  EXPECT_GT(most, 0u);
  EXPECT_LT(most, tiles.size() / 5);
  // a step back and forth across a boundary changes nothing: a cell that split stays split a little further
  const std::vector<B::Tile> there = B::grassTiles(spec, std::vector<glm::vec2>{at + glm::vec2(0.3F, 0.0F)}, tiles);
  const std::vector<B::Tile> back  = B::grassTiles(spec, std::vector<glm::vec2>{at}, there);
  EXPECT_EQ(B::grassTiles(spec, std::vector<glm::vec2>{at + glm::vec2(0.3F, 0.0F)}, back), there);
}

TEST(BackdropTiles, FlowersAndFernsStandInDriftsNearWhereOneLooksFrom)
{
  for(const B::Spec& spec : {meadow(), forest()})
  {
    const auto      tiles = B::driftTiles(spec, std::vector<glm::vec2>{{0.0F, 0.0F}});
    const B::Drifts in    = B::drifts(spec);
    const float     size  = in.covers[0].tile;
    size_t          thick = 0;
    for(const B::Tile& t : tiles)
    {
      EXPECT_LT(t.level, 2);
      thick += t.level == 0 ? 1 : 0;
      EXPECT_LT(glm::length(glm::vec2((float(t.x) + 0.5F) * size, (float(t.z) + 0.5F) * size)), in.reach);
    }
    // over the drifts' share of the ground within reach, the middles of the drifts thick
    const double cells = 3.14159265 * double(in.reach) * double(in.reach) / double(size * size);
    EXPECT_NEAR(double(tiles.size()) / cells, in.ground, 0.2) << spec.kind;
    EXPECT_GT(thick, 0u);
    EXPECT_LT(thick, tiles.size());
    EXPECT_LT(in.covers[1].perSquareMetre, in.covers[0].perSquareMetre);
    EXPECT_LT(in.middle, in.ground);
  }
}

TEST(BackdropTiles, ATileLiesOnTheGroundLeaningAsItDoes)
{
  const B::Spec spec = meadow();
  // in the clearing: level, at the floor
  const B::Place flat = B::tilePlace(spec, 2.0F, 0, 0, 0.0F);
  EXPECT_EQ(flat.position, glm::vec3(1.0F, 0.0F, 1.0F));
  EXPECT_NEAR(glm::length(flat.rotation * glm::vec3(0.0F, 1.0F, 0.0F) - glm::vec3(0.0F, 1.0F, 0.0F)), 0.0F, 1e-5F);
  // out on the rolling ground: at the ground's height less its sink, its up the ground's normal
  bool turned = false;
  for(int i = 10; i < 40; i++)
  {
    const B::Place p = B::tilePlace(spec, 4.0F, i, -i / 2, 0.1F);
    EXPECT_FLOAT_EQ(p.position.x, (float(i) + 0.5F) * 4.0F);
    EXPECT_NEAR(p.position.y, B::height(spec, p.position.x, p.position.z) - 0.1F, 1e-4F);
    const glm::vec3 up = p.rotation * glm::vec3(0.0F, 1.0F, 0.0F);
    const float     e  = 2.0F;
    const float     dx = (B::height(spec, p.position.x + e, p.position.z) - B::height(spec, p.position.x - e, p.position.z)) / (2.0F * e);
    const float     dz = (B::height(spec, p.position.x, p.position.z + e) - B::height(spec, p.position.x, p.position.z - e)) / (2.0F * e);
    EXPECT_NEAR(glm::length(up - glm::normalize(glm::vec3(-dx, 1.0F, -dz))), 0.0F, 1e-4F);
    EXPECT_NEAR(glm::length(p.rotation), 1.0F, 1e-5F);
    // and tiles are turned differently about the vertical, so one patch doesn't read as a pattern
    const glm::vec3 forward = p.rotation * glm::vec3(0.0F, 0.0F, 1.0F);
    turned = turned || std::abs(forward.x) > 0.7F;
  }
  EXPECT_TRUE(turned);
}

