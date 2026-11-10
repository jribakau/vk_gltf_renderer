// The moving view's bounce cache (shaders/radiance_cache.h.slang) compiled as C++: which pixels feed it,
// where a point's cell is kept, what a path's sample is, and how a cell's radiance follows its samples.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
#include "../shaders/radiance_cache.h.slang"
using namespace shaderio;

namespace {

constexpr float kCell  = 0.04F;
constexpr uint  kSlots = 1u << 16;
const float3    kEye(0.0F, 1.6F, 3.0F);

TEST(RadianceCache, OnePixelInSixteenUpdatesAndEachPixelDoesWithinSixteenFrames)
{
  for(int frame = 0; frame < 16; frame++)
  {
    int updating = 0;
    for(int y = 0; y < 64; y++)
      for(int x = 0; x < 64; x++)
        updating += radianceCacheUpdatePixel(x, y, frame) ? 1 : 0;
    EXPECT_EQ(updating, 64 * 64 / 16) << "frame " << frame;
  }
  for(int y = 0; y < 8; y++)
    for(int x = 0; x < 8; x++)
    {
      int times = 0;
      for(int frame = 0; frame < 16; frame++)
        times += radianceCacheUpdatePixel(x, y, frame) ? 1 : 0;
      EXPECT_EQ(times, 1) << x << "," << y;
    }
}

TEST(RadianceCache, CellsDoubleWithTheDistanceFromTheEye)
{
  const float nearEnd = kCell * RC_NEAR_CELLS;
  EXPECT_EQ(radianceCacheLevel(0.0F, kCell), 0);
  EXPECT_EQ(radianceCacheLevel(0.99F * nearEnd, kCell), 0);
  EXPECT_EQ(radianceCacheLevel(1.01F * nearEnd, kCell), 0);
  EXPECT_EQ(radianceCacheLevel(2.01F * nearEnd, kCell), 1);
  EXPECT_EQ(radianceCacheLevel(4.01F * nearEnd, kCell), 2);
  EXPECT_EQ(radianceCacheLevel(1e9F, kCell), RC_MAX_LEVEL);
  EXPECT_EQ(radianceCacheLevel(std::numeric_limits<float>::quiet_NaN(), kCell), 0);
  EXPECT_FLOAT_EQ(radianceCacheCellSize(0, kCell), kCell);
  EXPECT_FLOAT_EQ(radianceCacheCellSize(3, kCell), 8.0F * kCell);
  // a cell is between one part in RC_NEAR_CELLS and one in twice that of its distance
  for(float dist = nearEnd; dist < 500.0F; dist *= 1.37F)
  {
    const float size = radianceCacheCellSize(radianceCacheLevel(dist, kCell), kCell);
    EXPECT_LE(size, dist / RC_NEAR_CELLS * 1.0001F) << dist;
    EXPECT_GT(size, dist / (2.0F * RC_NEAR_CELLS) * 0.9999F) << dist;
  }
}

TEST(RadianceCache, TheTwoSidesOfAWallAreDifferentCells)
{
  EXPECT_EQ(radianceCacheFace(float3(1.0F, 0.0F, 0.0F)), 0);
  EXPECT_EQ(radianceCacheFace(float3(-1.0F, 0.0F, 0.0F)), 1);
  EXPECT_EQ(radianceCacheFace(float3(0.0F, 1.0F, 0.0F)), 2);
  EXPECT_EQ(radianceCacheFace(float3(0.0F, -1.0F, 0.0F)), 3);
  EXPECT_EQ(radianceCacheFace(float3(0.0F, 0.0F, 1.0F)), 4);
  EXPECT_EQ(radianceCacheFace(float3(0.0F, 0.0F, -1.0F)), 5);
  // a wall's normal with a normal map's wobble keeps its face
  EXPECT_EQ(radianceCacheFace(normalize(float3(0.1F, 0.95F, -0.2F))), 2);

  const float3           p(0.513F, 1.207F, -0.331F);
  const RadianceCacheKey front = radianceCacheKey(p, float3(0.0F, 0.0F, 1.0F), kEye, kCell, kSlots);
  const RadianceCacheKey back  = radianceCacheKey(p, float3(0.0F, 0.0F, -1.0F), kEye, kCell, kSlots);
  EXPECT_TRUE(front.bucket != back.bucket || front.fingerprint != back.fingerprint);
}

TEST(RadianceCache, APointsKeyIsItsCells)
{
  const float3 n(0.0F, 1.0F, 0.0F);
  // the middle of a cell on the floor, whatever size cells are there
  const float  size = radianceCacheCellSize(radianceCacheLevel(length(kEye), kCell), kCell);
  const float3 p(0.5F * size, 0.0F, 0.5F * size);
  ASSERT_EQ(radianceCacheCellSize(radianceCacheLevel(length(p - kEye), kCell), kCell), size);
  const RadianceCacheKey key  = radianceCacheKey(p, n, kEye, kCell, kSlots);
  EXPECT_NE(key.fingerprint, 0u);
  EXPECT_LT(key.bucket, kSlots);
  EXPECT_EQ(key.bucket % uint(RC_BUCKET), 0u);

  // anywhere in the cell: the same key
  const RadianceCacheKey same = radianceCacheKey(p + float3(0.4F * size, 0.0F, 0.4F * size), n, kEye, kCell, kSlots);
  EXPECT_EQ(same.bucket, key.bucket);
  EXPECT_EQ(same.fingerprint, key.fingerprint);
  // the cell beside it, and below the origin (negative coordinates are their own cells)
  for(const float3 step : {float3(size, 0.0F, 0.0F), float3(0.0F, 0.0F, size), float3(-size, 0.0F, 0.0F), float3(0.0F, 0.0F, -size)})
  {
    const RadianceCacheKey other = radianceCacheKey(p + step, n, kEye, kCell, kSlots);
    EXPECT_TRUE(other.bucket != key.bucket || other.fingerprint != key.fingerprint);
  }
}

TEST(RadianceCache, KeysSpreadOverTheTable)
{
  // the cells of a floor: few share a bucket, and none of those that do share a fingerprint
  std::set<std::pair<uint, uint>> keys;
  std::vector<int>                perBucket(kSlots / RC_BUCKET, 0);
  const int                       side = 100;
  for(int z = 0; z < side; z++)
    for(int x = 0; x < side; x++)
    {
      const float3           p((float(x) + 0.5F) * kCell, 0.0F, (float(z) + 0.5F) * kCell);
      const RadianceCacheKey key = radianceCacheKey(p, float3(0.0F, 1.0F, 0.0F), p + float3(0.0F, 1.0F, 0.0F), kCell, kSlots);
      keys.insert({key.bucket, key.fingerprint});
      perBucket[key.bucket / RC_BUCKET]++;
    }
  EXPECT_EQ(keys.size(), size_t(side * side));
  int crowded = 0;
  for(const int count : perBucket)
    crowded += count > RC_BUCKET ? count - RC_BUCKET : 0;
  // 10,000 cells in 16,384 buckets of four: next to none finds its bucket full
  EXPECT_LT(crowded, side * side / 500);
}

TEST(RadianceCache, AJitteredPointStaysWithinHalfACellOnItsFace)
{
  const float3 p(1.0F, 2.0F, 3.0F);
  const float  size = 0.08F;
  for(int face = 0; face < 6; face++)
    for(const float2 u : {float2(0.0F, 0.0F), float2(0.999F, 0.999F), float2(0.5F, 0.5F), float2(0.1F, 0.8F)})
    {
      const float3 q = radianceCacheJitter(p, face, size, u);
      const float3 d = q - p;
      EXPECT_EQ(d[face / 2], 0.0F) << face;  // never along the face's normal: into the wall or off it
      EXPECT_LE(std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)}), 0.5F * size + 1e-6F);
    }
  EXPECT_EQ(radianceCacheJitter(p, 2, size, float2(0.5F, 0.5F)), p);
}

TEST(RadianceCache, AVertexNearTheLastTakesAnotherBounce)
{
  const float size = 0.08F;
  EXPECT_FALSE(radianceCacheFarEnough(0.5F * size, size, 0.0F));
  EXPECT_FALSE(radianceCacheFarEnough(0.99F * size, size, 0.0F));
  EXPECT_TRUE(radianceCacheFarEnough(1.01F * size, size, 0.0F));
  EXPECT_FALSE(radianceCacheFarEnough(1.5F * size, size, 0.6F));
  EXPECT_TRUE(radianceCacheFarEnough(1.5F * size, size, 0.4F));
  EXPECT_TRUE(radianceCacheFarEnough(2.0F * size, size, 0.999F));
}

TEST(RadianceCache, ASampleIsWhatThePathGatheredAndWhatItArrivedWith)
{
  const RadianceCacheSample s = radianceCacheSample(float3(0.3F, 0.1F, 0.05F), float3(0.6F, 0.5F, 0.25F));
  ASSERT_TRUE(s.valid);
  EXPECT_EQ(s.light, float3(0.3F, 0.1F, 0.05F));
  EXPECT_EQ(s.weight, float3(0.6F, 0.5F, 0.25F));
  // a path that gathered nothing is a sample too: the cell is dark that way
  const RadianceCacheSample dark = radianceCacheSample(float3(0.0F), float3(0.8F));
  EXPECT_TRUE(dark.valid);
  EXPECT_EQ(dark.light, float3(0.0F));
  // rounding left it a hair under nothing
  EXPECT_EQ(radianceCacheSample(float3(-1e-7F, 0.0F, 0.0F), float3(0.8F)).light.x, 0.0F);
  // a path through a pure red wall says what it can: its red
  const RadianceCacheSample red = radianceCacheSample(float3(0.9F, 0.0F, 0.0F), float3(0.9F, 0.0F, 0.0F));
  EXPECT_TRUE(red.valid);
  EXPECT_EQ(red.weight, float3(0.9F, 0.0F, 0.0F));
}

TEST(RadianceCache, ASampleThatIsNoNumberIsDropped)
{
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(radianceCacheSample(float3(1.0F), float3(0.0F)).valid);  // arrived with nothing
  EXPECT_FALSE(radianceCacheSample(float3(nan, 0.0F, 0.0F), float3(0.5F)).valid);
  EXPECT_FALSE(radianceCacheSample(float3(inf, 0.0F, 0.0F), float3(0.5F)).valid);
  EXPECT_FALSE(radianceCacheSample(float3(1.0F), float3(nan)).valid);
  EXPECT_FALSE(radianceCacheSample(float3(1.0F), float3(inf)).valid);
}

// a cell fed the samples of one frame
RadianceCacheCell fed(RadianceCacheCell cell, std::initializer_list<RadianceCacheSample> samples, float maxWeight = RC_MAX_WEIGHT)
{
  float3 light(0.0F), weight(0.0F);
  for(const RadianceCacheSample& s : samples)
  {
    light += s.light;
    weight += s.weight;
  }
  return radianceCacheMerge(cell, light, weight, maxWeight);
}

TEST(RadianceCache, ACellsRadianceIsItsLightOverItsThroughput)
{
  // two paths off a surface sending back (2, 1, 0.5): one arrived white, one dim and red
  const float3      sent(2.0F, 1.0F, 0.5F);
  const float3      white(0.8F), red(0.4F, 0.02F, 0.02F);
  RadianceCacheCell cell{float3(0.0F), float3(0.0F)};
  cell = fed(cell, {radianceCacheSample(sent * white, white), radianceCacheSample(sent * red, red)});
  const float3 radiance = radianceCacheRadiance(cell);
  EXPECT_NEAR(radiance.x, sent.x, 1e-5F);
  EXPECT_NEAR(radiance.y, sent.y, 1e-5F);
  EXPECT_NEAR(radiance.z, sent.z, 1e-5F);
  EXPECT_FLOAT_EQ(radianceCacheReceived(cell.weight), 1.2F);
  EXPECT_TRUE(radianceCacheValid(cell));

  // a cell no path has reached holds nothing, and neither does a channel none arrived with
  EXPECT_EQ(radianceCacheRadiance(RadianceCacheCell{float3(0.0F), float3(0.0F)}), float3(0.0F));
  EXPECT_FALSE(radianceCacheValid(RadianceCacheCell{float3(0.0F), float3(0.0F)}));
  const RadianceCacheCell redOnly = fed({float3(0.0F), float3(0.0F)}, {radianceCacheSample(float3(1.8F, 0.0F, 0.0F), float3(0.9F, 0.0F, 0.0F))});
  EXPECT_EQ(radianceCacheRadiance(redOnly), float3(2.0F, 0.0F, 0.0F));
}

TEST(RadianceCache, ThePathsThatReadACellGetTheLightThePathsThatFedItFound)
{
  // A surface that sends back more toward where the red paths come from than toward the blue ones'. One
  // colour cannot say that, but with each sample weighed by its throughput the paths that read the cell
  // get between them, channel by channel, what the paths that fed it found; the plain mean of each
  // path's radiance would hand the blue paths the red ones' light.
  const float3              red(0.6F, 0.05F, 0.05F), blue(0.05F, 0.05F, 0.6F);
  const float3              towardRed(3.0F), towardBlue(1.0F);
  const RadianceCacheSample a = radianceCacheSample(towardRed * red, red);
  const RadianceCacheSample b = radianceCacheSample(towardBlue * blue, blue);
  const RadianceCacheCell   cell     = fed({float3(0.0F), float3(0.0F)}, {a, b});
  const float3              radiance = radianceCacheRadiance(cell);
  const float3              read     = radiance * red + radiance * blue;  // two paths like them read it
  const float3              found    = a.light + b.light;
  EXPECT_NEAR(read.x, found.x, 1e-5F);
  EXPECT_NEAR(read.y, found.y, 1e-5F);
  EXPECT_NEAR(read.z, found.z, 1e-5F);
  const float3 plain = 0.5F * (towardRed + towardBlue) * (red + blue);
  EXPECT_GT(plain.z, 1.5F * found.z);
}

TEST(RadianceCache, ADimPathsSampleCountsForLittle)
{
  // forty bright paths found the surface dark; one that arrived with a hundredth of their throughput
  // caught a lamp. Divided by its throughput it would be a hundred times its share.
  float3 light(0.0F), weight(0.0F);
  for(int i = 0; i < 40; i++)
    weight += float3(0.5F);
  light += float3(5.0F) * 0.005F;
  weight += float3(0.005F);
  const float3 radiance = radianceCacheRadiance(radianceCacheMerge({float3(0.0F), float3(0.0F)}, light, weight, RC_MAX_WEIGHT));
  EXPECT_NEAR(radiance.x, 5.0F * 0.005F / 20.005F, 1e-6F);
  EXPECT_LT(radiance.x, 0.002F);  // its plain mean: 5 / 41 = 0.12
}

TEST(RadianceCache, OldLightFadesAndFasterWhenTheSceneChanged)
{
  // a cell holding all it may under a light that has gone out: a sample a frame, off a wall of 0.5
  const auto framesToFade = [](float maxWeight) {
    RadianceCacheCell cell{float3(RC_MAX_WEIGHT), float3(RC_MAX_WEIGHT)};  // radiance 1
    int               frames = 0;
    while(radianceCacheRadiance(cell).x > 0.1F && frames < 10000)
    {
      cell = radianceCacheMerge(cell, float3(0.0F), float3(0.5F), maxWeight);
      frames++;
    }
    EXPECT_LE(radianceCacheReceived(cell.weight), maxWeight + 1e-3F);
    return frames;
  };
  const int steady  = framesToFade(RC_MAX_WEIGHT);
  const int changed = framesToFade(RC_CHANGED_WEIGHT);
  EXPECT_GT(steady, 150);
  EXPECT_LT(steady, 300);
  EXPECT_LT(changed, 20);

  // a change cuts what a cell keeps though no sample landed in it: its radiance stays, and it may still be read
  const RadianceCacheCell cut = radianceCacheMerge({float3(2.0F * RC_MAX_WEIGHT), float3(RC_MAX_WEIGHT)}, float3(0.0F), float3(0.0F), RC_CHANGED_WEIGHT);
  EXPECT_FLOAT_EQ(radianceCacheReceived(cut.weight), RC_CHANGED_WEIGHT);
  EXPECT_FLOAT_EQ(radianceCacheRadiance(cut).x, 2.0F);
  EXPECT_GE(RC_CHANGED_WEIGHT, RC_MIN_WEIGHT);
  EXPECT_TRUE(radianceCacheValid(cut));
  // a frame without samples leaves a cell under the limit as it is
  const RadianceCacheCell some{float3(3.0F), float3(1.5F)};
  const RadianceCacheCell same = radianceCacheMerge(some, float3(0.0F), float3(0.0F), RC_MAX_WEIGHT);
  EXPECT_EQ(same.light, some.light);
  EXPECT_EQ(same.weight, some.weight);
  // more in a frame than the limit allows: the frame's alone
  const RadianceCacheCell flooded = radianceCacheMerge({float3(RC_MAX_WEIGHT), float3(RC_MAX_WEIGHT)}, float3(20.0F), float3(10.0F), RC_CHANGED_WEIGHT);
  EXPECT_FLOAT_EQ(radianceCacheRadiance(flooded).x, 2.0F);
  EXPECT_FLOAT_EQ(radianceCacheReceived(flooded.weight), 10.0F);
}

TEST(RadianceCache, ACellOfNoisySamplesSettlesOnTheirMean)
{
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  RadianceCacheCell                     cell{float3(0.0F), float3(0.0F)};
  for(int frame = 0; frame < 2000; frame++)
  {
    // one to three paths a frame off walls of 0.3 to 0.9, each finding a light of 10 one time in five
    const int count = 1 + int(uniform(rng) * 3.0F);
    float3    light(0.0F), weight(0.0F);
    for(int i = 0; i < count; i++)
    {
      const float3 throughput(0.3F + 0.6F * uniform(rng));
      light += throughput * (uniform(rng) < 0.2F ? 10.0F : 0.0F);
      weight += throughput;
    }
    cell = radianceCacheMerge(cell, light, weight, RC_MAX_WEIGHT);
  }
  EXPECT_LE(radianceCacheReceived(cell.weight), RC_MAX_WEIGHT + 1e-3F);
  // the mean is 2; over some 80 samples' worth its deviation is about 0.45
  EXPECT_NEAR(radianceCacheRadiance(cell).x, 2.0F, 1.5F);
}

TEST(RadianceCache, ACellIsFoundInItsBucketAndReadAsItWasWritten)
{
  std::vector<uint>      table(size_t(kSlots) * RC_WORDS, 0u);
  const RadianceCacheKey key = radianceCacheKey(float3(0.3F, 0.0F, 0.2F), float3(0.0F, 1.0F, 0.0F), kEye, kCell, kSlots);
  EXPECT_EQ(radianceCacheFind(table.data(), key), -1);  // an empty table holds nothing (a fingerprint is never 0)

  // another cell in the bucket's first slot, this one in its third
  const uint   slot = key.bucket + 2u;
  const size_t base = size_t(slot) * RC_WORDS;
  table[size_t(key.bucket) * RC_WORDS + RC_KEY] = key.fingerprint ^ 0x5555u;
  table[base + RC_KEY]                          = key.fingerprint;
  for(int c = 0; c < 3; c++)
  {
    table[base + RC_LIGHT + c]  = radianceCacheBits(float(c + 1));
    table[base + RC_WEIGHT + c] = radianceCacheBits(4.0F * float(c + 1));
  }
  ASSERT_EQ(radianceCacheFind(table.data(), key), int(slot));
  const RadianceCacheCell cell = radianceCacheRead(table.data(), int(slot));
  EXPECT_EQ(cell.light, float3(1.0F, 2.0F, 3.0F));
  EXPECT_EQ(cell.weight, float3(4.0F, 8.0F, 12.0F));
  EXPECT_EQ(radianceCacheRadiance(cell), float3(0.25F));
  EXPECT_EQ(radianceCacheFloat(radianceCacheBits(-3.5F)), -3.5F);
  // the slot's words: the key, its age, what it keeps and the frame's sums, with none over
  EXPECT_EQ(RC_SUM_WEIGHT + 3, RC_WORDS);
}

}  // namespace
