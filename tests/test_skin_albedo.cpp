// What a skin map means to the random walk under it (shaders/skin_albedo.h.slang) compiled as C++.
// The reference is an isotropic half-space whose walk leaves wherever it reaches the surface, as the
// diffuse exit makes it: lit at direction cosine mu0 inside, it returns 1 - H(mu0) sqrt(1 - alpha),
// Chandrasekhar's H-function, checked here against such a walk. Chiang, Kutz and Burley's inversion
// gives the map as that albedo averaged over the directions light comes from; the face-on one gives
// it for light entering head-on, which is how a refracting entry meets skin seen face-on.

// GLM as test_bsdf_lobes.cpp, another file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/skin_albedo.h.slang"
#include "../shaders/skin_slab.h.slang"
#include "../shaders/skin_spread.h.slang"

namespace {

// Gauss-Legendre nodes and weights on [0, 1]
struct Quadrature
{
  std::vector<double> mu, weight;
};

Quadrature gaussLegendre(int n)
{
  Quadrature q;
  for(int i = 0; i < n; i++)
  {
    double x = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
    double d = 0;
    for(int k = 0; k < 100; k++)
    {
      double p0 = 1, p1 = x;
      for(int j = 2; j <= n; j++)
      {
        const double p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
        p0              = p1;
        p1              = p2;
      }
      d             = n * (x * p1 - p0) / (x * x - 1);
      const double dx = p1 / d;
      x -= dx;
      if(std::abs(dx) < 1e-15)
        break;
    }
    q.mu.push_back((x + 1) / 2);
    q.weight.push_back(1 / ((1 - x * x) * d * d));  // 2 / ((1 - x^2) P'^2), halved for [0, 1]
  }
  return q;
}

// Chandrasekhar's H-function for isotropic scattering of albedo `omega`, by its fixed point
// 1 / H(mu) = sqrt(1 - omega) + omega / 2 * integral of mu' H(mu') / (mu + mu') over [0, 1]
class HFunction
{
public:
  explicit HFunction(double omega)
      : m_omega(omega)
      , m_s(std::sqrt(std::max(1 - omega, 0.0)))
      , m_q(gaussLegendre(96))
      , m_h(m_q.mu.size(), 1.0)
  {
    for(int it = 0; it < 20000; it++)
    {
      double change = 0;
      std::vector<double> next(m_h.size());
      for(size_t i = 0; i < m_h.size(); i++)
      {
        next[i] = (*this)(m_q.mu[i]);
        change  = std::max(change, std::abs(next[i] - m_h[i]));
      }
      m_h = next;
      if(change < 1e-12)
        break;
    }
  }
  double operator()(double mu) const
  {
    double sum = 0;
    for(size_t j = 0; j < m_h.size(); j++)
      sum += m_q.weight[j] * m_q.mu[j] * m_h[j] / (mu + m_q.mu[j]);
    return 1 / (m_s + m_omega / 2 * sum);
  }
  // what the half-space returns lit at direction cosine mu0 inside it
  double albedo(double mu0) const { return 1 - m_s * (*this)(mu0); }
  // ...averaged over the directions light comes from, by their cosine
  double meanAlbedo() const
  {
    double sum = 0;
    for(size_t i = 0; i < m_h.size(); i++)
      sum += 2 * m_q.weight[i] * m_q.mu[i] * (1 - m_s * m_h[i]);
    return sum;
  }

private:
  double              m_omega, m_s;
  Quadrature          m_q;
  std::vector<double> m_h;
};

// The share of walks that leave: a unit-extinction half-space below z = 0, entered at direction
// cosine mu0, isotropic scattering of albedo omega, absorption by weight
double walkAlbedo(double omega, double mu0, int walks, uint32_t seed)
{
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double                                 total = 0;
  for(int w = 0; w < walks; w++)
  {
    double z = 0, dz = -mu0, weight = 1;
    for(;;)
    {
      z += dz * -std::log(1 - u(rng));
      if(z > 0)
      {
        total += weight;
        break;
      }
      weight *= omega;
      if(weight < 1e-3)
      {
        if(u(rng) > 0.1)
          break;
        weight *= 10;
      }
      dz = 2 * u(rng) - 1;
    }
  }
  return total / walks;
}

// What a slab of skin's medium does with light that enters its top face diffusely: the share of
// walks that leave back through the top and through the bottom. The medium is the shader's for a map
// A and a diffusion length of 1 (the direction-averaged inversion, the app's: sigma_t = 1 / sqrt(3 (1 -
// alpha))), the slab `depth` diffusion lengths thick, laterally unbounded, so only the height matters.
struct SlabWalk
{
  double reflected = 0, transmitted = 0;
};
SlabWalk walkSlab(float a, double depth, int walks, uint32_t seed)
{
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double                           omega  = multiToSingleScatterAlbedo(float3(a), 0.0f).x;
  const double                           sigmaT = 1 / std::sqrt(3 * (1 - omega));
  SlabWalk                               out;
  for(int w = 0; w < walks; w++)
  {
    double z = 0, dz = -std::sqrt(u(rng)), weight = 1;  // by the cosine, downward
    for(;;)
    {
      z += dz * -std::log(1 - u(rng)) / sigmaT;
      if(z > 0)
      {
        out.reflected += weight;
        break;
      }
      if(z < -depth)
      {
        out.transmitted += weight;
        break;
      }
      weight *= omega;
      if(weight < 1e-2)
      {
        if(u(rng) > 0.5)
          break;
        weight *= 2;
      }
      dz = 2 * u(rng) - 1;
    }
  }
  out.reflected /= walks;
  out.transmitted /= walks;
  return out;
}

// the albedo either inversion gives, one channel at a time
double faceOn(float a)
{
  return faceOnSingleScatterAlbedo(float3(a)).x;
}
double directionAveraged(float a)
{
  return multiToSingleScatterAlbedo(float3(a), 0.0f).x;
}

}  // namespace

// The reference is the walk: an H-function albedo and a random walk's agree, head-on and at a slant
TEST(SkinAlbedo, TheHFunctionIsTheWalksAlbedo)
{
  for(const double omega : {0.5, 0.9, 0.99})
    for(const double mu0 : {1.0, 0.7})
    {
      const double walked = walkAlbedo(omega, mu0, 200000, 7);
      // a walk returns 0 or a weight up to 1: its spread is under half, so 200000 of them within 0.002
      EXPECT_NEAR(walked, HFunction(omega).albedo(mu0), 0.004) << "omega " << omega << " mu0 " << mu0;
    }
}

// Skin lit head-on comes back at its map with the face-on inversion, within 0.1% over a skin's range
// and 0.15% to the ends; the inversion a diffuse entry is right for returns 5 to 21% less there
TEST(SkinAlbedo, TheFaceOnInversionReturnsTheMapHeadOn)
{
  for(float a = 0.002f; a < 0.999f; a += 0.025f)
  {
    const double back = HFunction(faceOn(a)).albedo(1.0);
    EXPECT_NEAR(back / a, 1.0, a > 0.01f && a < 0.98f ? 1e-3 : 1.5e-3) << "A " << a;
  }
  // a pale skin's red, green and blue (tests/render/skin_furnace.py): what the renderer returned
  // face-on before, 0.944, 0.880 and 0.850 of it, is this
  const double red = HFunction(directionAveraged(0.8f)).albedo(1.0) / 0.8, green = HFunction(directionAveraged(0.55f)).albedo(1.0) / 0.55,
               blue = HFunction(directionAveraged(0.45f)).albedo(1.0) / 0.45;
  EXPECT_NEAR(red, 0.949, 0.002);
  EXPECT_NEAR(green, 0.887, 0.002);
  EXPECT_NEAR(blue, 0.863, 0.002);
  EXPECT_LT(HFunction(directionAveraged(0.1f)).albedo(1.0) / 0.1, 0.8);
}

// The direction-averaged inversion does what it says: the albedo over every direction light comes
// from is the map, within its fit's 0.5% over a pale skin's range and 2.5% for a dark map, where the
// face-on one is above it
TEST(SkinAlbedo, TheDirectionAveragedInversionReturnsTheMapOnAverage)
{
  for(float a : {0.05f, 0.2f, 0.45f, 0.55f, 0.8f, 0.95f})
  {
    EXPECT_NEAR(HFunction(directionAveraged(a)).meanAlbedo() / a, 1.0, a < 0.4f ? 2.5e-2 : 5e-3) << "A " << a;
    EXPECT_GT(HFunction(faceOn(a)).meanAlbedo() / a, 1.01) << "A " << a;
  }
}

// Seen from a slant the camera's ray refracts in less steeply and the face-on skin reads above its
// map: at 60 degrees through an index of 1.4, a pale skin 3, 6 and 8% (the shader's comment)
TEST(SkinAlbedo, AtASlantFaceOnSkinReadsAboveItsMap)
{
  const double sinT = std::sin(60.0 * std::numbers::pi / 180) / 1.4, mu0 = std::sqrt(1 - sinT * sinT);
  EXPECT_NEAR(HFunction(faceOn(0.8f)).albedo(mu0) / 0.8, 1.027, 0.003);
  EXPECT_NEAR(HFunction(faceOn(0.55f)).albedo(mu0) / 0.55, 1.063, 0.003);
  EXPECT_NEAR(HFunction(faceOn(0.45f)).albedo(mu0) / 0.45, 1.079, 0.003);
  EXPECT_NEAR(HFunction(directionAveraged(0.8f)).albedo(mu0) / 0.8, 0.980, 0.003);
  EXPECT_NEAR(HFunction(directionAveraged(0.45f)).albedo(mu0) / 0.45, 0.940, 0.003);
}

// A brighter map is a medium that absorbs less, all the way: no channel turns over or leaves [0, 1]
TEST(SkinAlbedo, TheFaceOnInversionRisesWithTheMap)
{
  double last = -1;
  for(float a = 0.0f; a <= 1.0f; a += 0.001f)
  {
    const double alpha = faceOn(a);
    EXPECT_GE(alpha, last) << "A " << a;
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    last = alpha;
  }
  EXPECT_LT(faceOn(0.001f), 0.02);
  EXPECT_GT(faceOn(0.999f), 0.9999);
  // the three channels are inverted alike, each by itself
  const float3 rgb = faceOnSingleScatterAlbedo(float3(0.8f, 0.55f, 0.45f));
  EXPECT_FLOAT_EQ(rgb.y, float(faceOn(0.55f)));
  EXPECT_FLOAT_EQ(rgb.z, float(faceOn(0.45f)));
}

// The thin preview's slab (skin_slab.h.slang): nothing in the way lets everything through, depth gives
// back the map, and no depth makes or loses light it shouldn't
TEST(SkinSlab, ItRunsFromClearToTheMap)
{
  const float3 A(0.6f, 0.35f, 0.2f);
  const SkinSlab none = skinSlab(A, float3(0.0f));
  EXPECT_NEAR(none.reflectance.x, 0.0f, 1e-6f);
  EXPECT_NEAR(none.transmittance.y, 1.0f, 1e-6f);
  const SkinSlab deep = skinSlab(A, float3(60.0f));
  EXPECT_NEAR(deep.reflectance.x, 0.6f, 1e-5f);
  EXPECT_NEAR(deep.reflectance.z, 0.2f, 1e-5f);
  EXPECT_LT(deep.transmittance.x, 1e-20f);
  float3 lastR(-1.0f), lastT(2.0f);
  for(float depth = 0.0f; depth <= 12.0f; depth += 0.05f)
  {
    const SkinSlab s = skinSlab(A, float3(depth));
    for(int c = 0; c < 3; c++)
    {
      EXPECT_GE(s.reflectance[c], lastR[c] - 1e-6f) << depth;  // deeper: more comes back (to rounding)
      EXPECT_LE(s.transmittance[c], lastT[c] + 1e-6f) << depth;  // ...and less goes through
      EXPECT_LE(s.reflectance[c] + s.transmittance[c], 1.0f + 1e-6f) << depth;
      EXPECT_LE(s.reflectance[c], A[c] + 1e-6f) << depth;
    }
    lastR = s.reflectance;
    lastT = s.transmittance;
  }
}

// ...and it is the walk's: a slab of the medium the still walks through, entered diffusely, returns
// and lets through what Kubelka-Munk says, within a few percent where the glow is (a light channel,
// up to a few diffusion lengths) and low by up to a quarter for a dark channel deep in. The preview
// probes to SKIN_SLAB_REACH lengths, where the light channel lets through almost nothing.
TEST(SkinSlab, ItLetsThroughWhatTheWalkDoes)
{
  for(const float a : {0.6f, 0.35f, 0.2f})
    for(const double depth : {0.25, 0.5, 1.0, 2.0, 3.0})
    {
      const SlabWalk walk = walkSlab(a, depth, 100000, uint32_t(a * 1000 + depth * 10));
      const SkinSlab km   = skinSlab(float3(a), float3(float(depth)));
      EXPECT_NEAR(km.reflectance.x, walk.reflected, 0.02) << "A " << a << " depth " << depth;
      EXPECT_NEAR(km.transmittance.x, walk.transmitted, 0.08 * walk.transmitted + 0.016) << "A " << a << " depth " << depth;
      if(a >= 0.5f)
        EXPECT_NEAR(km.transmittance.x / walk.transmitted, 1.0, 0.06) << "depth " << depth;
      std::printf("A %.2f depth %.2f: walk R %.3f T %.4f, slab R %.3f T %.4f\n", a, depth, walk.reflected, walk.transmitted,
                  km.reflectance.x, km.transmittance.x);
    }
  const SkinSlab reach = skinSlab(float3(0.6f), float3(SKIN_SLAB_REACH));
  EXPECT_LT(reach.transmittance.x, 1e-3f);
}

namespace {

// Where the still's walk lets light out that went in straight down (a face-on camera ray refracts to
// the normal): the medium the shader makes from a map A by the face-on inversion, a unit-extinction
// half-space whose walk leaves the first time it reaches the surface (ePtVolumeDiffuseExit). The radii
// are in diffusion lengths, the scatter distance the medium is made from, each with the walk's weight.
struct SpreadExit
{
  double radius, weight;
};
std::vector<SpreadExit> walkSpread(float a, int walks, uint32_t seed)
{
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double                           omega = faceOn(a);
  const double                           d     = 1 / std::sqrt(3 * (1 - omega));  // in mean free paths
  std::vector<SpreadExit>                out;
  for(int w = 0; w < walks; w++)
  {
    double x = 0, y = 0, z = 0, dx = 0, dy = 0, dz = -1, weight = 1;
    for(;;)
    {
      const double t = -std::log(1 - u(rng));
      if(z + dz * t > 0)
      {
        const double up = -z / dz;  // to the surface
        out.push_back({std::hypot(x + dx * up, y + dy * up) / d, weight});
        break;
      }
      x += dx * t;
      y += dy * t;
      z += dz * t;
      weight *= omega;
      if(weight < 1e-3)
      {
        if(u(rng) > 0.1)
          break;
        weight *= 10;
      }
      dz                = 2 * u(rng) - 1;
      const double phi  = 2 * std::numbers::pi * u(rng);
      const double side = std::sqrt(std::max(1 - dz * dz, 0.0));
      dx                = side * std::cos(phi);
      dy                = side * std::sin(phi);
    }
  }
  return out;
}

// The largest gap between the walk's share of light out within each radius and the profile's, at the
// rate skinSpreadRate gives a unit scatter distance
double spreadGap(std::vector<SpreadExit> exits, float a)
{
  std::sort(exits.begin(), exits.end(), [](const SpreadExit& l, const SpreadExit& r) { return l.radius < r.radius; });
  double total = 0;
  for(const SpreadExit& e : exits)
    total += e.weight;
  const double k   = skinSpreadRate(float3(float(faceOn(a))), float3(1.0f)).x;
  double       sum = 0, gap = 0;
  for(const SpreadExit& e : exits)
  {
    sum += e.weight;
    gap = std::max(gap, std::abs(sum / total - double(skinSpreadUnitCdf(float(k * e.radius)))));
  }
  return gap;
}

}  // namespace

// The spread's radius is drawn from the profile it is weighed as: one draw of the unit profile is
// within a percent of its share at every radius, up to where the draw is cut (SKIN_SPREAD_MAX)
TEST(SkinSpread, ItsRadiusFollowsTheProfile)
{
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float>                    radii(200000);
  for(float& r : radii)
    r = skinSpreadUnitRadius(u(rng), u(rng));
  std::sort(radii.begin(), radii.end());
  EXPECT_LE(radii.back(), SKIN_SPREAD_MAX);
  double gap = 0;
  for(size_t i = 0; i < radii.size(); i++)
    if(radii[i] < SKIN_SPREAD_MAX)
      gap = std::max(gap, std::abs(double(i + 1) / radii.size() - double(skinSpreadUnitCdf(radii[i]))));
  EXPECT_LT(gap, 0.01);
  EXPECT_GT(skinSpreadUnitCdf(SKIN_SPREAD_MAX), 0.999f);
}

// ...and the profile is where the still's walk lets the light out: through the medium a map makes,
// the walk's share within each radius is within 0.03 of the profile's for a map up to 0.8 (the scale's
// fit to the walk is within a few percent; a lighter map's walk is more peaked than the shape)
TEST(SkinSpread, ItSpreadsAsTheWalkDoes)
{
  for(const float a : {0.1f, 0.25f, 0.4f, 0.6f, 0.8f})
  {
    const double gap = spreadGap(walkSpread(a, 60000, uint32_t(a * 1000)), a);
    EXPECT_LT(gap, a > 0.7f ? 0.035 : 0.03) << "A " << a;
    std::printf("A %.2f: k %.3f, the walk's share within a radius at most %.4f from the profile's\n", a,
                skinSpreadRate(float3(float(faceOn(a))), float3(1.0f)).x, gap);
  }
}

// Measured skin spreads red furthest and blue least, as the still's shadow edges did on 10TEST (red
// 1.5 mm past the sun's penumbra, green 0.9, blue 0.7): a pale skin map over the measured distances
TEST(SkinSpread, RedSpreadsFurthest)
{
  const float3 alpha = faceOnSingleScatterAlbedo(float3(0.6f, 0.35f, 0.25f));
  const float3 rate  = skinSpreadRate(alpha, float3(0.00367f, 0.00137f, 0.00068f));
  EXPECT_LT(rate.x, rate.y);
  EXPECT_LT(rate.y, rate.z);
  // a median radius of a millimetre or so: half of the unit profile lies within 1.7
  for(int c = 0; c < 3; c++)
  {
    EXPECT_GT(1.7f / rate[c], 0.0002f) << c;
    EXPECT_LT(1.7f / rate[c], 0.002f) << c;
  }
}

// The thin preview's skin by the view's angle (skinViewAlbedo, skin_albedo.h.slang). Hapke's H-function
// is within 1.5% of the exact over the albedos skin maps make, and as a change from head-on, which is
// how the face-on preview uses it, within 0.3% where a refraction at index 1.4 enters (a cosine inside of
// 0.70 and up; at shallower ones, which only a lower index reaches, within about 1%).
TEST(SkinViewAlbedo, HapkesHIsCloseAndItsChangeFromHeadOnCloser)
{
  for(const float a : {0.05f, 0.15f, 0.25f, 0.35f, 0.5f, 0.6f, 0.8f, 0.95f})
  {
    const double    omega = faceOn(a);
    const double    s     = std::sqrt(1 - omega);
    const HFunction H(omega);
    const double    approxHeadOn = 1 - hFunctionApprox(float3(float(omega)), 1.0f).x * s;
    for(const double mu : {0.05, 0.2, 0.4, 0.6, 0.72, 0.85, 1.0})
    {
      const double approx = hFunctionApprox(float3(float(omega)), float(mu)).x;
      EXPECT_NEAR(approx / H(mu), 1.0, 0.015) << "A " << a << " mu " << mu;
      const double exactChange  = (1 - H(mu) * s) / (1 - H(1.0) * s);
      const double approxChange = (1 - approx * s) / approxHeadOn;
      EXPECT_NEAR(approxChange / exactChange, 1.0, mu >= 0.7 ? 0.003 : 0.012) << "A " << a << " mu " << mu;
    }
  }
}

// With the face-on inversion the preview is the map head-on and rises as the view slants, a darker
// channel the most, as the still's walk did on a slab under one light (A 0.6, 0.35, 0.25, specular
// off, scratch skin_angles.py): 3.0, 6.0 and 7.2% above head-on at a 45-degree view, 6.2, 13.6 and 16.7%
// at 70, where the rough boundary lets some light in shallower than the smooth one here
TEST(SkinViewAlbedo, FaceOnItIsTheMapHeadOnAndRisesAtASlant)
{
  const float3 A     = float3(0.6f, 0.35f, 0.25f);
  const float3 alpha = faceOnSingleScatterAlbedo(A);
  const float3 head  = skinViewAlbedo(A, alpha, 1.0f, 1.4f, true);
  for(int c = 0; c < 3; c++)
    EXPECT_NEAR(head[c], A[c], 1e-5f) << c;
  const float3 at45 = skinViewAlbedo(A, alpha, std::cos(0.25f * float(std::numbers::pi)), 1.4f, true) / A;
  const float3 at70 = skinViewAlbedo(A, alpha, std::cos(70.0f * float(std::numbers::pi) / 180.0f), 1.4f, true) / A;
  const float  walk45[3] = {1.030f, 1.060f, 1.072f}, walk70[3] = {1.062f, 1.136f, 1.167f};
  for(int c = 0; c < 3; c++)
  {
    EXPECT_NEAR(at45[c], walk45[c], 0.01f) << c;
    EXPECT_NEAR(at70[c], walk70[c], 0.035f) << c;
    EXPECT_GT(at70[c], at45[c]) << c;
  }
  EXPECT_LT(at70.x, at70.y);
  EXPECT_LT(at70.y, at70.z);
}

// With the direction-averaged inversion the walk's own albedo stands: head-on below the map, as a
// refraction entry under that inversion comes back (tests/render/skin_furnace.py: 6-15% short), and
// within 1.5% of the exact H-function's
TEST(SkinViewAlbedo, DirectionAveragedItIsTheWalksOwn)
{
  for(const float a : {0.8f, 0.55f, 0.45f, 0.25f})
  {
    const double    omega = directionAveraged(a);
    const HFunction H(omega);
    const float     head  = skinViewAlbedo(float3(a), float3(float(omega)), 1.0f, 1.4f, false).x;
    EXPECT_NEAR(head / (1 - H(1.0) * std::sqrt(1 - omega)), 1.0, 0.015) << a;
    EXPECT_LT(head, 0.96f * a) << a;
    EXPECT_GT(head, 0.80f * a) << a;
  }
}

// The preview's softened diffuse sees the camera at the smooth normal's angle by the share the walk's
// spread evens out: bumps that lean either way take nearly the smooth surface's colour, not each its own
TEST(SkinViewAlbedo, TheSoftenedShareSeesTheViewAtTheSmoothNormal)
{
  const float3 A(0.6f, 0.35f, 0.25f);
  const float3 alpha = faceOnSingleScatterAlbedo(A);
  const float  eta   = 1.4f;
  const float3 none(0.0f), all(1.0f), skin(0.97f, 0.9f, 0.7f);
  const float3 atMap    = skinViewAlbedo(A, alpha, 0.4f, eta, true);
  const float3 atSmooth = skinViewAlbedo(A, alpha, 0.8f, eta, true);
  for(int c = 0; c < 3; c++)
  {
    EXPECT_NEAR(softenedSkinViewAlbedo(A, alpha, 0.4f, 0.8f, none, eta, true)[c], atMap[c], 1e-6f);
    EXPECT_NEAR(softenedSkinViewAlbedo(A, alpha, 0.4f, 0.8f, all, eta, true)[c], atSmooth[c], 1e-6f);
  }
  // two bumps at 0.6 and 0.95 about a smooth 0.8: their colours far closer together softened
  const float3 a = softenedSkinViewAlbedo(A, alpha, 0.6f, 0.8f, skin, eta, true);
  const float3 b = softenedSkinViewAlbedo(A, alpha, 0.95f, 0.8f, skin, eta, true);
  const float3 a0 = skinViewAlbedo(A, alpha, 0.6f, eta, true), b0 = skinViewAlbedo(A, alpha, 0.95f, eta, true);
  for(int c = 0; c < 3; c++)
    EXPECT_LT(std::abs(a[c] - b[c]), 0.35f * std::abs(a0[c] - b0[c])) << c;
}

// The white of an eye's spread (a scatter distance with no volume): the radius is an exponential of the
// scatter distance's mean, cut at 2.5 of it
TEST(SurfaceSpread, ItsRadiusIsAnExponentialCutShort)
{
  const float d = 0.8f;
  double      sum = 0;
  const int   n   = 100000;
  for(int i = 0; i < n; i++)
  {
    const float r = surfaceSpreadRadius(d, (float(i) + 0.5f) / float(n));
    EXPECT_GE(r, 0.0f);
    EXPECT_LE(r, SURFACE_SPREAD_CUT * d);
    sum += r;
  }
  // the mean of an exponential cut at c d: d (1 - e^-c)
  EXPECT_NEAR(sum / n, d * (1.0 - std::exp(-double(SURFACE_SPREAD_CUT))), 0.005);
}

// ...and its point clears a surface that curves up to meet its own tangent plane. A Genesis 9 iris is a
// shallow bowl, a cone that rises up to 0.07 mm a millimetre toward its rim: points moved in the tangent
// plane alone lie under it wherever they go round the cone, their shadow rays met the iris itself, and
// the eye's direct light fell by a third (measured on the mesh 11-14). Lifted, none does; nor inside a
// sphere of the eyeball's radius (12 mm), nor on a flat or an outward-curved surface.
TEST(SurfaceSpread, ItsPointClearsABowl)
{
  const float d = 0.8f;  // millimetres: the sclera's scatter distance
  std::mt19937                          rng(11);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  // height of a point over each surface (+ above it, on the side of the normal)
  const float slope = 0.07f, rho0 = 2.3f;
  const auto  cone  = [&](float3 q) { return (q.z - slope * (std::sqrt(q.x * q.x + q.y * q.y) - rho0)) / std::sqrt(1.0f + slope * slope); };
  const float R     = 12.0f;
  const auto  ball  = [&](float3 q) { return R - length(q - float3(0.0f, 0.0f, R)); };  // inside a sphere, its floor at the origin
  const auto  dome  = [&](float3 q) { return length(q - float3(0.0f, 0.0f, -R)) - R; };  // outside one, its top at the origin

  const float3 coneAt(rho0, 0.0f, 0.0f);
  const float3 coneN = normalize(float3(-slope, 0.0f, 1.0f));
  const float3 coneT(0.0f, 1.0f, 0.0f);
  const float3 coneB = cross(coneN, coneT);
  const float3 flatN(0.0f, 0.0f, 1.0f), flatT(1.0f, 0.0f, 0.0f), flatB(0.0f, 1.0f, 0.0f);

  int buriedInPlane = 0;
  const int n = 20000;
  for(int i = 0; i < n; i++)
  {
    const float r   = surfaceSpreadRadius(d, u(rng));
    const float phi = u(rng) * 6.2831853f;
    if(r < 1e-3f)
      continue;
    const float3 onCone = coneAt + surfaceSpreadOffset(coneN, coneT, coneB, r, phi);
    EXPECT_GT(cone(onCone), 0.02f * r) << r << " " << phi;
    // the same point without its lift, as it was
    if(cone(onCone - coneN * (SURFACE_SPREAD_LIFT * r)) < 0.0f)
      buriedInPlane++;
    const float3 off = surfaceSpreadOffset(flatN, flatT, flatB, r, phi);
    EXPECT_GT(ball(off), 0.0f) << r;
    EXPECT_GT(dome(off), 0.0f) << r;
    // no further from the plane than a tenth of how far out it is
    EXPECT_NEAR(off.z, SURFACE_SPREAD_LIFT * r, 1e-5f);
    EXPECT_NEAR(std::sqrt(off.x * off.x + off.y * off.y), r, 1e-4f);
  }
  EXPECT_GT(buriedInPlane, n / 4);
}
