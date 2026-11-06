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

#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/skin_albedo.h.slang"

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
