// The backdrop's ground and planting (src/fv_backdrop.hpp): flat where the figures stand, the same for
// a seed every time, a mesh that is the height function's, and plants that stand on it where their
// layer says, as thickly as it says.

#include <algorithm>
#include <cmath>
#include <set>

#include <gtest/gtest.h>

#include "../src/fv_backdrop.hpp"

namespace {

namespace B = fv::backdrop;

B::Spec meadow(int seed = 1)
{
  return {.kind = "meadow", .seed = seed};
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

TEST(BackdropLayers, AMeadowPlantsGrassFlowersSaplingsAndTrees)
{
  EXPECT_TRUE(B::layers(B::Spec{}).empty());
  const auto          l = B::layers(meadow());
  std::set<std::string> roles;
  for(const B::Layer& layer : l)
    roles.insert(layer.role);
  EXPECT_EQ(roles, (std::set<std::string>{"grass", "flower", "sapling", "tree"}));
  uint64_t plants = 0;
  for(const B::Layer& layer : l)
  {
    plants += B::expectedCount(meadow(), layer);
    // trunks and saplings stay clear of where the figures stand
    if(layer.role == "tree" || layer.role == "sapling")
      EXPECT_GT(layer.from, B::kClearing);
    else
      EXPECT_EQ(layer.from, 0.0F);
  }
  // a default meadow is a few hundred megabytes of instances at most (about 0.8 KB each)
  EXPECT_GT(plants, 50000u);
  EXPECT_LT(plants, 300000u);
  // a larger one plants further out
  B::Spec wide = meadow();
  wide.size    = 120.0;
  EXPECT_GT(B::layers(wide)[0].to, l[0].to);
}
