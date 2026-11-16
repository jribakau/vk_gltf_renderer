// A far cut-out read a cell of its texels at a time (shaders/cutout_cell.h.slang) compiled as C++:
// a hit reads its own texel until the footprint covers one, the rays a pixel's jitter scatters over
// its footprint read a handful of texels and not the footprint's every one, a card keeps the share
// of it that is solid (a thin pattern a mip's average would cut away too), and a card nearing or
// receding changes a little at a time.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <utility>
#include <vector>

using namespace glm;
#include "../shaders/shaderio.h"
using namespace shaderio;
#include "../shaders/cutout_cell.h.slang"

namespace {

// A square map of `n` texels a side, true where it is solid
struct Mask
{
  int               n = 0;
  std::vector<bool> solid;
  bool at(float2 texel) const
  {
    const int x = ((int(std::floor(texel.x)) % n) + n) % n, y = ((int(std::floor(texel.y)) % n) + n) % n;
    return solid[size_t(y) * size_t(n) + size_t(x)];
  }
  double share() const
  {
    size_t on = 0;
    for(const bool s : solid)
      on += s ? 1 : 0;
    return double(on) / double(solid.size());
  }
};

// The share of the map's texels the cells keep solid, a texel's centre at a time as a footprint of
// `texels` reads it
double keptShare(const Mask& m, float texels)
{
  size_t on = 0;
  for(int y = 0; y < m.n; y++)
    for(int x = 0; x < m.n; x++)
      on += m.at(cutoutCellTexel(float2(float(x) + 0.5F, float(y) + 0.5F), texels)) ? 1 : 0;
  return double(on) / double(m.n) / double(m.n);
}

}  // namespace

TEST(CutoutCell, UnderATexelAHitReadsWhereItFalls)
{
  for(const float texels : {0.0F, 0.25F, 0.99F, std::nanf("")})
  {
    const float2 at = cutoutCellTexel(float2(17.3F, -4.6F), texels);
    EXPECT_EQ(at.x, 17.3F);
    EXPECT_EQ(at.y, -4.6F);
  }
}

// What a pixel's jittered rays read: hits anywhere in a footprint-sized square, a few texels of it
// (the cells there are half the footprint or more), where each ray read its own before
TEST(CutoutCell, ThePixelsJitteredRaysReadAFewTexelsOfItsFootprint)
{
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  for(const float texels : {3.0F, 8.0F, 20.0F, 100.0F, 700.0F})
    for(int pixel = 0; pixel < 50; pixel++)
    {
      const float2                 corner(unit(rng) * 4096.0F - 2048.0F, unit(rng) * 4096.0F - 2048.0F);
      std::set<std::pair<int, int>> read;
      for(int ray = 0; ray < 64; ray++)
      {
        const float2 at = cutoutCellTexel(corner + float2(unit(rng), unit(rng)) * texels, texels);
        // a texel's centre
        EXPECT_EQ(at.x - std::floor(at.x), 0.5F);
        EXPECT_EQ(at.y - std::floor(at.y), 0.5F);
        read.insert({int(std::floor(at.x)), int(std::floor(at.y))});
      }
      EXPECT_LE(read.size(), 9U) << texels << " texels across";
    }
}

// A leafy card's solid share stays as its texels have it at every distance, where a cut-off on a
// mip's average would thin it: noise, and leaves a texel thin on a period that is no power of two
// and one that is
TEST(CutoutCell, ACardKeepsItsSolidShareFarOff)
{
  const int         n = 1024;
  std::mt19937      rng(3);
  std::bernoulli_distribution sparse(0.3);
  Mask              noise{n, std::vector<bool>(size_t(n) * n)};
  for(size_t i = 0; i < noise.solid.size(); i++)
    noise.solid[i] = sparse(rng);
  Mask thin3{n, std::vector<bool>(size_t(n) * n)}, thin4{n, std::vector<bool>(size_t(n) * n)};
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      thin3.solid[size_t(y) * n + size_t(x)] = x % 3 == 0;
      thin4.solid[size_t(y) * n + size_t(x)] = x % 4 == 0 || y % 4 == 0;
    }
  for(const Mask* m : {&noise, &thin3, &thin4})
    for(const float texels : {1.5F, 4.0F, 6.0F, 16.0F, 23.0F})
      EXPECT_NEAR(keptShare(*m, texels), m->share(), 0.04) << "share " << m->share() << ", " << texels << " texels across";
}

// A card nearing or receding: from one footprint to one a percent wider, about a hundredth of its
// texels read another, across a power of two too; and between two powers of two, the narrower
// cells give way to the wider ones a cell at a time, a wider one's quarter keeping its texel
TEST(CutoutCell, ACardNearingChangesALittleAtATime)
{
  const int  n       = 256;
  const auto changed = [&](float a, float b) {
    size_t moved = 0;
    for(int y = 0; y < n; y++)
      for(int x = 0; x < n; x++)
      {
        const float2 at(float(x) + 0.5F, float(y) + 0.5F);
        const float2 p = cutoutCellTexel(at, a), q = cutoutCellTexel(at, b);
        moved += p.x != q.x || p.y != q.y ? 1 : 0;
      }
    return double(moved) / double(n) / double(n);
  };
  for(const float texels : {1.3F, 3.0F, 7.9F, 8.0F, 12.0F, 31.8F})
    EXPECT_LT(changed(texels, texels * 1.01F), 0.03) << texels;
  // a step of a power of two: every cell has gone to the next size (cells of 4 to cells of 8, the
  // texel of a quarter of each kept)
  EXPECT_NEAR(changed(4.0F, 8.0F), 0.75, 0.05);
}
