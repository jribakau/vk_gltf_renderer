// nvshaders' BSDF (bsdf_functions.h.slang, a C++/Slang polyglot) compiled as C++: next-event
// estimation's every-lobe evaluation against the single random lobe it replaces, the two sides of
// its MIS pair, and sampling against evaluation for the lobes whose upstream code was fixed with it.
// Its own executable: the header needs GLM's swizzles, which the glm target turns off.

#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
namespace shaderio {
inline vec3     lerp(vec3 a, vec3 b, float t) { return glm::mix(a, b, t); }
inline uint32_t asuint(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float asfloat(uint32_t u)
{
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline vec3  max(vec3 a, vec3 b) { return glm::max(a, b); }
inline float max(float a, double b) { return glm::max(a, float(b)); }
}  // namespace shaderio
using namespace shaderio;
#include <nvshaders/bsdf_functions.h.slang>

namespace {

struct Case
{
  const char* name;
  PbrMaterial mat;
};

PbrMaterial base(float roughness)
{
  return defaultPbrMaterial(vec3(0.8f, 0.6f, 0.4f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1));
}

// One of each lobe the sum adds up, and the combinations a DAZ figure has.
std::vector<Case> cases()
{
  std::vector<Case> out;
  out.push_back({"plastic", base(0.4f)});
  PbrMaterial metal = base(0.3f);
  metal.metallic    = 1.0f;
  out.push_back({"metal", metal});
  PbrMaterial cornea  = base(0.15f);
  cornea.transmission = 1.0f;
  cornea.ior2         = 1.376f;
  cornea.thickness    = 0.01f;
  out.push_back({"cornea", cornea});
  PbrMaterial coat        = base(0.5f);
  coat.clearcoat          = 1.0f;
  coat.clearcoatRoughness = 0.3f;
  coat.Nc                 = normalize(vec3(0.3f, 0.1f, 1.0f));  // a clearcoat normal map
  out.push_back({"clearcoat", coat});
  PbrMaterial sheen    = base(0.6f);
  sheen.sheenColor     = vec3(0.5f);
  sheen.sheenRoughness = 0.5f;
  out.push_back({"sheen", sheen});
  PbrMaterial retro     = base(0.4f);
  retro.retroreflection = 0.6f;
  out.push_back({"retroreflection", retro});
  PbrMaterial skin                   = base(0.45f);
  skin.diffuseTransmissionFactor     = 0.5f;
  skin.diffuseTransmissionColorFactor = vec3(0.9f, 0.4f, 0.3f);
  out.push_back({"diffuse transmission", skin});
  PbrMaterial film          = base(0.35f);
  film.iridescence          = 0.7f;
  film.iridescenceThickness = 0.4f;
  out.push_back({"iridescence", film});
  return out;
}

vec3 sphere(std::mt19937& rng)
{
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const float z = 1.0f - 2.0f * u(rng), phi = 6.2831853f * u(rng), r = std::sqrt(std::max(0.0f, 1.0f - z * z));
  return vec3(r * std::cos(phi), r * std::sin(phi), z);
}

vec3 viewAbove(std::mt19937& rng)
{
  vec3 v = sphere(rng);
  v.z    = std::abs(v.z) * 0.9f + 0.1f;
  return normalize(v);
}

float sum(vec3 v) { return v.x + v.y + v.z; }

}  // namespace

// bsdfEvaluateAll is bsdfEvaluate's mean over the lobe choice (xi.z), at every pair of directions.
TEST(BsdfLobes, EveryLobeIsTheMeanOfOneRandomLobe)
{
  std::mt19937 rng(7);
  for(const Case& c : cases())
    for(int pair = 0; pair < 64; pair++)
    {
      BsdfEvaluateData all;
      all.k1 = viewAbove(rng);
      all.k2 = sphere(rng);
      all.xi = vec3(0.3f, 0.6f, 0.5f);
      bsdfEvaluateAll(all, c.mat);

      const int N = 20000;
      vec3      mean(0.0f);
      for(int i = 0; i < N; i++)
      {
        BsdfEvaluateData one;
        one.k1 = all.k1;
        one.k2 = all.k2;
        one.xi = vec3(0.3f, 0.6f, (float(i) + 0.5f) / float(N));
        bsdfEvaluate(one, c.mat);
        mean += (one.bsdf_diffuse + one.bsdf_glossy) / float(N);
      }
      const float expected = sum(mean), got = sum(all.bsdf_diffuse + all.bsdf_glossy);
      EXPECT_NEAR(got, expected, 2e-3f * expected + 1e-5f) << c.name << ", pair " << pair;
    }
}

// The BSDF side of the MIS pair samples exactly as bsdfSample does and reports the pdf
// bsdfEvaluateAll gives the light side at the same directions.
TEST(BsdfLobes, TheMixtureSampleReportsTheEveryLobePdf)
{
  std::mt19937                          rng(11);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(const Case& c : cases())
    for(int i = 0; i < 2000; i++)
    {
      BsdfSampleData mixture;
      mixture.k1 = viewAbove(rng);
      mixture.xi = vec3(u(rng), u(rng), u(rng));
      BsdfSampleData single = mixture;
      bsdfSampleMixture(mixture, c.mat);
      bsdfSample(single, c.mat);
      ASSERT_EQ(mixture.event_type, single.event_type) << c.name;
      if(single.event_type == BSDF_EVENT_ABSORB)
        continue;
      EXPECT_EQ(mixture.k2, single.k2) << c.name;
      EXPECT_EQ(mixture.bsdf_over_pdf, single.bsdf_over_pdf) << c.name;
      if(mixture.pdf == DIRAC)
        continue;
      BsdfEvaluateData all;
      all.k1 = mixture.k1;
      all.k2 = mixture.k2;
      all.xi = mixture.xi;
      bsdfEvaluateAll(all, c.mat);
      EXPECT_NEAR(all.pdf, mixture.pdf, 1e-4f * mixture.pdf) << c.name;
    }
}

// What sampling carries (the mean of bsdf_over_pdf) is the BSDF's integral over directions, as
// evaluation gives it. Retroreflection sampled twice its value, rough refraction's evaluation
// lacked eta_t^2 and the clearcoat's evaluation used the base normal's frame.
TEST(BsdfLobes, SamplingCarriesWhatEvaluationIntegratesTo)
{
  std::mt19937                          rng(13);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(Case c : cases())
  {
    // uniform directions cannot integrate the cornea's narrow refraction lobe: a rougher one
    if(c.mat.transmission > 0.0f)
      c.mat.roughness = vec2(0.4f * 0.4f);
    const vec3 k1 = normalize(vec3(0.4f, 0.2f, 0.9f));
    const int  N  = 400000;
    double     carried = 0.0, integral = 0.0;
    for(int i = 0; i < N; i++)
    {
      BsdfSampleData s;
      s.k1 = k1;
      s.xi = vec3(u(rng), u(rng), u(rng));
      bsdfSample(s, c.mat);
      if(s.event_type != BSDF_EVENT_ABSORB)
        carried += sum(s.bsdf_over_pdf) / N;

      BsdfEvaluateData e;
      e.k1 = k1;
      e.k2 = sphere(rng);
      e.xi = vec3(u(rng), u(rng), u(rng));
      bsdfEvaluateAll(e, c.mat);
      integral += sum(e.bsdf_diffuse + e.bsdf_glossy) * 4.0 * 3.14159265358979 / N;
    }
    EXPECT_NEAR(carried, integral, 0.03 * integral) << c.name;
  }
}

// A reflection lobe gives nothing toward a light below its normal, which its sampler never produces.
TEST(BsdfLobes, AReflectionLobeIsDarkBelowItsNormal)
{
  PbrMaterial      metal = cases()[1].mat;
  BsdfEvaluateData e;
  e.k1 = normalize(vec3(0.5f, 0.0f, 0.8f));
  e.k2 = normalize(vec3(-0.6f, 0.0f, -0.05f));
  e.xi = vec3(0.5f);
  bsdfEvaluateAll(e, metal);
  EXPECT_EQ(sum(e.bsdf_glossy + e.bsdf_diffuse), 0.0f);
  EXPECT_EQ(e.pdf, 0.0f);
}
