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
  PbrMaterial bumped        = base(0.4f);
  bumped.clearcoat          = 0.5f;
  bumped.clearcoatRoughness = 0.3f;
  bumped.sheenColor         = vec3(0.3f);
  bumped.sheenRoughness     = 0.5f;
  bumped.N                  = normalize(vec3(0.7f, 0.0f, 1.0f));  // a normal map, 35 degrees off the smooth normal
  bumped.Nc                 = bumped.N;
  orthonormalBasis(bumped.N, bumped.T, bumped.B);
  out.push_back({"normal map", bumped});
  // The thin skin preview: a bumped diffuse lobe turned toward the smooth normal, red the most
  PbrMaterial softened      = base(0.45f);
  softened.N                = normalize(vec3(0.7f, 0.0f, 1.0f));
  softened.Nc               = softened.N;
  orthonormalBasis(softened.N, softened.T, softened.B);
  softened.diffuseSoftening = vec3(0.99f, 0.9f, 0.7f);
  out.push_back({"softened diffuse", softened});
  return out;
}

// The "normal map" case's: a view above the surface that its bent normal faces away from.
const vec3 kBehindTheBump = normalize(vec3(-1.0f, 0.0f, 0.3f));

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

// A view the normal map's normal faces away from still reflects: the specular lobes bend that
// normal back toward the view, and before they did, every specular sample was absorbed.
TEST(BsdfLobes, ABumpTiltedAwayFromTheViewStillReflects)
{
  PbrMaterial metal = cases()[1].mat;
  metal.N           = cases().back().mat.N;
  orthonormalBasis(metal.N, metal.T, metal.B);
  ASSERT_LT(dot(kBehindTheBump, metal.N), 0.0f);
  ASSERT_GT(dot(kBehindTheBump, metal.Ng), 0.0f);

  std::mt19937                          rng(17);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const int                             N       = 20000;
  float                                 carried = 0.0f;
  for(int i = 0; i < N; i++)
  {
    BsdfSampleData s;
    s.k1 = kBehindTheBump;
    s.xi = vec3(u(rng), u(rng), u(rng));
    bsdfSample(s, metal);
    if(s.event_type != BSDF_EVENT_ABSORB)
      carried += sum(s.bsdf_over_pdf) / 3.0f / N;
  }
  EXPECT_GT(carried, 0.3f);
}

// The shadow-terminator term is 1 without a bump and toward a light along the bent normal, and fades
// a bumped Lambert lobe to 0 at its shadow line.
TEST(BsdfLobes, TheTerminatorTermFadesOnlyABumpedLobe)
{
  const vec3 Ns(0.0f, 0.0f, 1.0f);
  const vec3 N = cases().back().mat.N;
  for(float cosI : {0.05f, 0.3f, 0.7f, 1.0f})
  {
    const vec3 L(std::sqrt(1.0f - cosI * cosI), 0.0f, cosI);
    EXPECT_FLOAT_EQ(shadowTerminatorTerm(L, Ns, Ns), 1.0f) << cosI;
  }
  EXPECT_FLOAT_EQ(shadowTerminatorTerm(N, N, Ns), 1.0f);

  float last = 1.0f;
  for(float cosI : {0.9f, 0.6f, 0.3f, 0.1f, 0.01f})
  {
    // around the bent normal, away from the smooth one
    const vec3  L = normalize(cosI * N + std::sqrt(1.0f - cosI * cosI) * normalize(vec3(1.0f, 0.0f, -0.7f)));
    const float g = shadowTerminatorTerm(L, N, Ns);
    EXPECT_LT(g, last) << cosI;
    EXPECT_GT(g, 0.0f) << cosI;
    last = g;
  }
  EXPECT_LT(last, 0.5f);
  EXPECT_EQ(shadowTerminatorTerm(-N, N, Ns), 0.0f);
}

// The softened diffuse lobe is, per channel, its share of a Lambert lobe about the smooth normal and
// the rest about the map's: all of it about Ns at softening 1, and the plain lobe at softening 0.
TEST(BsdfLobes, ASoftenedDiffuseLobeTurnsTowardTheSmoothNormal)
{
  PbrMaterial mat = cases().back().mat;
  const vec3  Ns  = mat.Ns;
  std::mt19937 rng(21);
  for(int i = 0; i < 200; i++)
  {
    const vec3 L = sphere(rng);
    mat.diffuseSoftening = vec3(1.0f);
    BsdfEvaluateData full{};
    full.k1 = vec3(0, 0, 1);
    full.k2 = L;
    brdf_diffuse_eval(full, mat, vec3(1.0f));
    EXPECT_NEAR(full.bsdf_diffuse.x, std::max(dot(L, Ns), 0.0f) / 3.14159265f, 1e-6f);

    mat.diffuseSoftening = vec3(0.0f);
    BsdfEvaluateData none{};
    none.k1 = vec3(0, 0, 1);
    none.k2 = L;
    brdf_diffuse_eval(none, mat, vec3(1.0f));
    const float cosN = std::max(dot(L, mat.N), 0.0f);
    EXPECT_NEAR(none.bsdf_diffuse.x, cosN / 3.14159265f * (cosN > 0.0f ? shadowTerminatorTerm(L, mat.N, Ns) : 0.0f), 1e-6f);

    // between the two, channel by channel
    mat.diffuseSoftening = vec3(0.25f, 0.5f, 0.75f);
    BsdfEvaluateData mix{};
    mix.k1 = vec3(0, 0, 1);
    mix.k2 = L;
    brdf_diffuse_eval(mix, mat, vec3(1.0f));
    for(int c = 0; c < 3; c++)
    {
      const float s = mat.diffuseSoftening[c];
      EXPECT_NEAR(mix.bsdf_diffuse[c], (1.0f - s) * none.bsdf_diffuse.x + s * full.bsdf_diffuse.x, 1e-6f) << c;
    }
  }
}
