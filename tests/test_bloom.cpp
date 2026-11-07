// Bloom's kernels (shaders/bloom.h.slang) compiled as C++: each keeps the light it spreads, the
// star's spikes go across the aperture's edges, and the whole pyramid, run here on the CPU as
// shaders/bloom.slang runs it on the GPU, moves light into the glow and the star without making or
// losing any.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/bloom.h.slang"

namespace {

constexpr float kPi = 3.14159265358979f;

// An image of linear colours, sampled as the shader's samplers do: bilinear, clamped to the edge
struct Image
{
  int               w = 0, h = 0;
  std::vector<vec3> px;
  Image(int width, int height)
      : w(width)
      , h(height)
      , px(size_t(width) * size_t(height), vec3(0.0f))
  {
  }
  vec3&       at(int x, int y) { return px[size_t(y) * size_t(w) + size_t(x)]; }
  const vec3& at(int x, int y) const { return px[size_t(y) * size_t(w) + size_t(x)]; }
  vec3        sample(vec2 uv) const
  {
    const float fx = uv.x * float(w) - 0.5f, fy = uv.y * float(h) - 0.5f;
    const int   x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
    const float tx = fx - float(x0), ty = fy - float(y0);
    const auto  texel = [&](int x, int y) { return at(std::clamp(x, 0, w - 1), std::clamp(y, 0, h - 1)); };
    return mix(mix(texel(x0, y0), texel(x0 + 1, y0), tx), mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), tx), ty);
  }
  double sum() const
  {
    double s = 0.0;
    for(const vec3& c : px)
      s += double(c.r) + double(c.g) + double(c.b);
    return s;
  }
};

vec2 centre(int x, int y, int w, int h)
{
  return vec2((float(x) + 0.5f) / float(w), (float(y) + 0.5f) / float(h));
}

Image down(const Image& src)
{
  Image dst(bloomLevelSide(src.w, 1), bloomLevelSide(src.h, 1));
  for(int y = 0; y < dst.h; y++)
    for(int x = 0; x < dst.w; x++)
    {
      vec3 s(0.0f);
      for(int i = 0; i < 13; i++)
        s += bloomDownWeight(i) * src.sample(centre(x, y, dst.w, dst.h) + bloomDownOffset(i) / vec2(src.w, src.h));
      dst.at(x, y) = s;
    }
  return dst;
}

Image up(const Image& fine, const Image& coarse, float tail)
{
  Image out(fine.w, fine.h);
  for(int y = 0; y < out.h; y++)
    for(int x = 0; x < out.w; x++)
    {
      vec3 c(0.0f);
      for(int i = 0; i < 9; i++)
        c += bloomTentWeight(i) * coarse.sample(centre(x, y, out.w, out.h) + bloomTentOffset(i) / vec2(coarse.w, coarse.h));
      out.at(x, y) = fine.at(x, y) * (1.0f - tail) + c * tail;
    }
  return out;
}

Image star(const Image& level, int blades, float rotation)
{
  Image       out(level.w, level.h);
  const int   lines = bloomStarLines(blades);
  const float step  = BLOOM_STAR_LENGTH * float(level.h) / float(BLOOM_STAR_TAPS);
  for(int y = 0; y < out.h; y++)
    for(int x = 0; x < out.w; x++)
    {
      vec3 s(0.0f);
      for(int l = 0; l < lines; l++)
      {
        const float a = bloomStarAngle(blades, rotation, l);
        const vec2  d = vec2(std::cos(a), -std::sin(a)) * step / vec2(level.w, level.h);
        for(int k = 1; k <= BLOOM_STAR_TAPS; k++)
          s += bloomStarWeight(k) * (level.sample(centre(x, y, out.w, out.h) + d * float(k)) + level.sample(centre(x, y, out.w, out.h) - d * float(k)));
      }
      out.at(x, y) = lines > 0 ? s / bloomStarNorm(lines) : vec3(0.0f);
    }
  return out;
}

// shaders/bloom.slang's passes in its order: down to `levels` levels, the star on its level, up, and
// the composite (exposure 1, no limit: the light stays where it was taken)
Image bloom(const Image& image, int levels, float share, float starShare, int blades, float rotation)
{
  std::vector<Image> d;
  d.push_back(down(image));
  for(int k = 1; k < levels; k++)
    d.push_back(down(d.back()));
  const Image s = star(d[BLOOM_STAR_LEVEL - 1], blades, rotation);
  Image       glow = d.back();
  for(int k = levels - 2; k >= 0; k--)
    glow = up(d[k], glow, BLOOM_TAIL);
  Image out(image.w, image.h);
  for(int y = 0; y < out.h; y++)
    for(int x = 0; x < out.w; x++)
    {
      const vec2 uv = centre(x, y, out.w, out.h);
      vec3       g(0.0f);
      for(int i = 0; i < 9; i++)
        g += bloomTentWeight(i) * glow.sample(uv + bloomTentOffset(i) / vec2(glow.w, glow.h));
      if(starShare > 0.0f)
        g = g * (1.0f - starShare) + s.sample(uv) * starShare;
      out.at(x, y) = image.at(x, y) * (1.0f - share) + g * share;
    }
  return out;
}

}  // namespace

// Each kernel's weights sum to one and are centred: a blur, no light made or lost, nothing moved
TEST(Bloom, TheKernelsKeepTheLight)
{
  float down = 0.0f, tent = 0.0f;
  vec2  downCentre(0.0f), tentCentre(0.0f);
  for(int i = 0; i < 13; i++)
  {
    down += bloomDownWeight(i);
    downCentre += bloomDownWeight(i) * bloomDownOffset(i);
  }
  for(int i = 0; i < 9; i++)
  {
    tent += bloomTentWeight(i);
    tentCentre += bloomTentWeight(i) * bloomTentOffset(i);
  }
  EXPECT_FLOAT_EQ(down, 1.0f);
  EXPECT_FLOAT_EQ(tent, 1.0f);
  EXPECT_NEAR(length(downCentre), 0.0f, 1e-7f);
  EXPECT_NEAR(length(tentCentre), 0.0f, 1e-7f);
  for(int blades = 3; blades <= 9; blades++)
  {
    const int lines = bloomStarLines(blades);
    float     sum   = 0.0f;
    for(int k = 1; k <= BLOOM_STAR_TAPS; k++)
      sum += 2.0f * float(lines) * bloomStarWeight(k);
    EXPECT_NEAR(sum / bloomStarNorm(lines), 1.0f, 1e-6f) << blades << " blades";
  }
}

// A round aperture has no star; an even number of blades as many spikes, an odd one twice as many
TEST(Bloom, TheStarHasASpikeAcrossEachEdge)
{
  EXPECT_EQ(bloomStarLines(0), 0);
  EXPECT_EQ(bloomStarLines(2), 0);
  EXPECT_EQ(bloomStarLines(6), 3);  // 6 spikes
  EXPECT_EQ(bloomStarLines(8), 4);  // 8 spikes
  EXPECT_EQ(bloomStarLines(5), 5);  // 10 spikes
  EXPECT_EQ(bloomStarLines(7), 7);  // 14 spikes
  for(int blades = 3; blades <= 9; blades++)
  {
    const float rotation = 0.3f;
    for(int l = 0; l < bloomStarLines(blades); l++)
    {
      // across the edge between corners l and l + 1 of sampleAperture's polygon
      const float a0 = rotation + 2.0f * kPi * float(l) / float(blades), a1 = a0 + 2.0f * kPi / float(blades);
      const vec2  edge = vec2(std::cos(a1), std::sin(a1)) - vec2(std::cos(a0), std::sin(a0));
      const float a    = bloomStarAngle(blades, rotation, l);
      EXPECT_NEAR(dot(normalize(edge), vec2(std::cos(a), std::sin(a))), 0.0f, 1e-5f) << blades << " blades, line " << l;
      // and no two lines the same
      for(int m = 0; m < l; m++)
        EXPECT_GT(std::abs(std::sin(a - bloomStarAngle(blades, rotation, m))), 1e-3f) << blades << " blades, lines " << l << ", " << m;
    }
  }
}

// The pyramid halves down to an 8-pixel side, at most eight levels: seven at the user's 2160x1395
TEST(Bloom, ThePyramidStopsAtEightPixels)
{
  EXPECT_EQ(bloomLevels(2160, 1395), 7);
  EXPECT_EQ(bloomLevels(3840, 2160), 8);
  EXPECT_EQ(bloomLevels(16, 16), 1);
  EXPECT_EQ(bloomLevels(15, 400), 0);
  EXPECT_EQ(bloomLevelSide(1395, 1), 698);
  EXPECT_EQ(bloomLevelSide(1395, 3), 175);
}

// A bright point in the middle of the frame: the pyramid moves `share` of it into a glow and a star,
// and the frame holds the same light; the glow reaches far, and the star's spikes outshine the glow
// between them
TEST(Bloom, ThePyramidMovesLightWithoutMakingAny)
{
  Image image(512, 512);
  for(int y = 254; y < 258; y++)
    for(int x = 254; x < 258; x++)
      image.at(x, y) = vec3(1000.0f, 800.0f, 600.0f);
  const Image glow = bloom(image, 5, 0.05f, 0.0f, 0, 0.0f);
  EXPECT_NEAR(glow.sum() / image.sum(), 1.0, 1e-4);
  EXPECT_NEAR(glow.at(255, 255).r, 1000.0f * 0.95f, 1000.0f * 0.02f);  // most of the point stays
  EXPECT_GT(glow.at(255 + 40, 255).r, 0.0f);                           // a long tail
  EXPECT_GT(glow.at(255 + 10, 255).r, glow.at(255 + 40, 255).r);

  const Image withStar = bloom(image, 5, 0.05f, 0.5f, 6, 0.0f);
  EXPECT_NEAR(withStar.sum() / image.sum(), 1.0, 1e-4);
  // six blades at 0: spikes across the edges at 30, 90 and 150 degrees; 0 degrees lies between two
  const int r = 24;
  EXPECT_GT(withStar.at(255, 255 - r).r, 1.5f * withStar.at(255 + r, 255).r);  // straight up: a spike
}
