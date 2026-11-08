// The water table (src/scene_water.hpp): a grid per material slot over the UVs its domes reach, each
// cell naming every dome that reaches into it, and a dry or damp slot with no grid.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "scene_water.hpp"

namespace {

using fv::mat::WaterItem;
using fv::mat::WaterLayer;

// a bead at (u, v) on skin stretched `metres` a UV unit both ways, the renderer's axes (v turned)
WaterItem bead(float u, float v, float radius, float metres = 1)
{
  return {.a = {u, v}, .b = {u, v}, .toTangent = {metres, 0, 0, -metres}, .radius = radius, .height = radius * 0.7f};
}

// the refs of the cell at a UV point
std::vector<uint32_t> at(const rs::WaterTables& t, uint32_t slot, float u, float v)
{
  const shaderio::WaterGrid& g = t.grids[slot];
  const int                  x = int(std::floor((u - g.uvMin.x) * g.cellsPerUv.x)), y = int(std::floor((v - g.uvMin.y) * g.cellsPerUv.y));
  if(x < 0 || y < 0 || x >= int(g.width) || y >= int(g.height))
    return {};
  const glm::uvec2 cell = t.cells[g.firstCell + uint32_t(y) * g.width + uint32_t(x)];
  return {t.refs.begin() + cell.x, t.refs.begin() + cell.x + cell.y};
}

TEST(WaterTables, EachCellNamesTheDomesThatReachIt)
{
  // two beads 2 mm across, 10 cm apart on skin of 1 m a UV unit; a track between two points
  const WaterLayer layer{.items = {bead(0.30f, 0.40f, 1e-3f), bead(0.40f, 0.40f, 1e-3f),
                                   {.a = {0.30f, 0.50f}, .b = {0.36f, 0.50f}, .toTangent = {1, 0, 0, -1}, .radius = 1.5e-3f, .height = 3e-4f}},
                         .damp  = 0.7f};
  const std::pair<uint32_t, const WaterLayer*> layers[] = {{2, &layer}};
  const rs::WaterTables                        t      = rs::packWater(4, layers);
  ASSERT_EQ(t.grids.size(), 4u);
  ASSERT_EQ(t.items.size(), 3u);
  for(uint32_t s : {0u, 1u, 3u})
  {
    EXPECT_EQ(t.grids[s].width, 0u);
    EXPECT_EQ(t.grids[s].damp, 0.f);
  }
  const shaderio::WaterGrid& g = t.grids[2];
  EXPECT_FLOAT_EQ(g.damp, 0.7f);
  ASSERT_GT(g.width, 1u);
  // cells about 5 mm on the skin
  EXPECT_NEAR(g.cellsPerUv.x, 1 / rs::kWaterCell, 0.1 / rs::kWaterCell);
  const auto has = [](const std::vector<uint32_t>& refs, uint32_t item) { return std::find(refs.begin(), refs.end(), item) != refs.end(); };
  // a bead's middle and its halo's edge name it, and not the far bead
  EXPECT_TRUE(has(at(t, 2, 0.30f, 0.40f), 0));
  EXPECT_FALSE(has(at(t, 2, 0.30f, 0.40f), 1));
  EXPECT_TRUE(has(at(t, 2, 0.30f + 1.4e-3f, 0.40f), 0));
  EXPECT_TRUE(has(at(t, 2, 0.40f, 0.40f - 1.4e-3f), 1));
  // the track, all along it
  for(float u = 0.30f; u <= 0.36f; u += 0.005f)
    EXPECT_TRUE(has(at(t, 2, u, 0.50f), 2)) << u;
  EXPECT_TRUE(at(t, 2, 0.35f, 0.45f).empty());
  // every ref names an item
  for(uint32_t r : t.refs)
    EXPECT_LT(r, t.items.size());
}

TEST(WaterTables, ADampSlotWithoutDomesAndWhatCantBeDrawn)
{
  // no area in UV: left out; nothing else on it, so only the damp
  const WaterLayer layer{.items = {{.a = {0.5f, 0.5f}, .b = {0.5f, 0.5f}, .toTangent = {0, 0, 0, 0}, .radius = 1e-3f, .height = 1e-3f}}, .damp = 0.4f};
  const std::pair<uint32_t, const WaterLayer*> layers[] = {{0, &layer}, {5, &layer}};
  const rs::WaterTables                        t      = rs::packWater(2, layers);
  ASSERT_EQ(t.grids.size(), 2u);
  EXPECT_EQ(t.grids[0].width, 0u);
  EXPECT_FLOAT_EQ(t.grids[0].damp, 0.4f);
  EXPECT_TRUE(t.items.empty());
  EXPECT_TRUE(t.cells.empty());
}

TEST(WaterTables, AStretchedSkinGetsCellsAsWideOnIt)
{
  // 2 m a UV unit along u, 0.5 m along v: cells as many a UV unit as that many 5 mm steps
  const WaterLayer layer{.items = {bead(0.2f, 0.2f, 1e-3f), {.a = {0.3f, 0.3f}, .b = {0.3f, 0.3f}, .toTangent = {2, 0, 0, -0.5f}, .radius = 1e-3f, .height = 7e-4f}},
                         .damp  = 0};
  WaterLayer stretched = layer;
  for(WaterItem& i : stretched.items)
    i.toTangent = {2, 0, 0, -0.5f};
  const std::pair<uint32_t, const WaterLayer*> layers[] = {{0, &stretched}};
  const rs::WaterTables                        t      = rs::packWater(1, layers);
  const shaderio::WaterGrid&                   g      = t.grids[0];
  EXPECT_NEAR(g.cellsPerUv.x / g.cellsPerUv.y, 4.0, 0.5);
}

}  // namespace
