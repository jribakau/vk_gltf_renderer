// The accumulation's Owen-scrambled Sobol points (shaders/sobol.h.slang), compiled as C++: every
// pair of a decision's four dimensions stratifies a power-of-two run of samples (a (0,2)-net),
// before and after the scramble, and a scrambled point is uniform over the decisions' seeds.

#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cstdint>
#include <set>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;

namespace {
uint reversebits(uint x)
{
  x = ((x >> 1u) & 0x55555555u) | ((x & 0x55555555u) << 1u);
  x = ((x >> 2u) & 0x33333333u) | ((x & 0x33333333u) << 2u);
  x = ((x >> 4u) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4u);
  x = ((x >> 8u) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8u);
  return (x >> 16u) | (x << 16u);
}
uint firstbitlow(uint x)
{
  uint bit = 0u;
  while(x != 0u && (x & 1u) == 0u)
  {
    x >>= 1u;
    bit++;
  }
  return bit;
}
}  // namespace

#include "../shaders/sobol.h.slang"

namespace {

// Whether points 0..2^k-1 put exactly one in every elementary interval of area 2^-k of dimensions
// (a, b): for every split of k bits into 2^i columns and 2^(k-i) rows
bool isNet(const std::vector<float4>& p, int k, int a, int b)
{
  const uint n = 1u << k;
  for(int i = 0; i <= k; i++)
  {
    std::set<uint> cells;
    for(uint s = 0; s < n; s++)
    {
      const uint x = uint(p[s][a] * float(1u << i)), y = uint(p[s][b] * float(1u << (k - i)));
      cells.insert(x * (1u << (k - i)) + y);
    }
    if(cells.size() != n)
      return false;
  }
  return true;
}

}  // namespace

TEST(Sobol, EveryPairOfDimensionsIsANet)
{
  const int k = 8;
  std::vector<float4> plain(1u << k), scrambled(1u << k);
  for(uint s = 0; s < (1u << k); s++)
  {
    for(uint d = 0; d < 4; d++)
      plain[s][d] = float(sobolBits(s, d) >> 8u) / 16777216.0F;
    scrambled[s] = sobolOwen4(s, 0x1234567u);
  }
  // Sobol's first two dimensions are a (0,2)-sequence; the scramble keeps every net it had
  EXPECT_TRUE(isNet(plain, k, 0, 1));
  EXPECT_TRUE(isNet(scrambled, k, 0, 1));
  for(int a = 0; a < 4; a++)
  {
    // each dimension alone: one point per 1/2^k
    std::set<uint> cells;
    for(uint s = 0; s < (1u << k); s++)
      cells.insert(uint(scrambled[s][a] * float(1u << k)));
    EXPECT_EQ(cells.size(), size_t(1u) << k) << a;
  }
  // a run that starts at a power of two is a net too: the samples of a later frame
  std::vector<float4> later(1u << 6);
  for(uint s = 0; s < (1u << 6); s++)
    later[s] = sobolOwen4((1u << 6) + s, 0x1234567u);
  EXPECT_TRUE(isNet(later, 6, 0, 1));
}

TEST(Sobol, ScrambledPointsAreUniformOverSeeds)
{
  // one sample index across many pixels' seeds: mean 1/2, every coordinate below 1
  double sum[4] = {0, 0, 0, 0};
  const int N   = 100000;
  for(int i = 0; i < N; i++)
  {
    const float4 u = sobolOwen4(5u, sobolHash(uint(i)));
    for(int d = 0; d < 4; d++)
    {
      ASSERT_LT(u[d], 1.0F);
      ASSERT_GE(u[d], 0.0F);
      sum[d] += u[d];
    }
  }
  for(int d = 0; d < 4; d++)
    EXPECT_NEAR(sum[d] / N, 0.5, 0.005) << d;
  // two decisions of one pixel differ
  EXPECT_NE(sobolOwen4(3u, sobolHash(sobolHashCombine(99u, 0u))).x, sobolOwen4(3u, sobolHash(sobolHashCombine(99u, 1u))).x);
}

TEST(Sobol, TheRecurrenceMatchesTheTable)
{
  // every bit of the index against its direction number in the table, as the generator matrix defines a point
  const auto reference = [](uint index, uint dim) {
    uint x = 0u;
    for(uint bit = 0u; bit < 32u; bit++)
      if((index >> bit) & 1u)
        x ^= SOBOL_DIRECTIONS[dim][bit];
    return x;
  };
  for(uint dim = 0u; dim < 4u; dim++)
  {
    for(uint i = 0u; i < 4096u; i++)
      EXPECT_EQ(sobolBits(i, dim), reference(i, dim));
    for(uint i = 1u; i < 4096u; i++)  // hashed indices reach the top bits
    {
      const uint index = sobolHash(i);
      EXPECT_EQ(sobolBits(index, dim), reference(index, dim));
    }
    EXPECT_EQ(sobolBits(0xffffffffu, dim), reference(0xffffffffu, dim));
  }
}
