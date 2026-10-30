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

namespace {

// The mean of what sampling `mat` from a view at cosine `NoV` carries, per event kind: reflected
// (glossy reflection) and transmitted (glossy transmission) and everything.
struct Carried
{
  vec3 reflected{0.0f}, transmitted{0.0f}, all{0.0f};
};

Carried carriedAt(const PbrMaterial& mat, float NoV, int N = 200000)
{
  std::mt19937                          rng(17);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const vec3 k1 = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
  Carried    c;
  for(int i = 0; i < N; i++)
  {
    BsdfSampleData s;
    s.k1 = k1;
    s.xi = vec3(u(rng), u(rng), u(rng));
    bsdfSample(s, mat);
    if(s.event_type == BSDF_EVENT_ABSORB)
      continue;
    const vec3 v = s.bsdf_over_pdf / float(N);
    c.all += v;
    if(s.event_type == BSDF_EVENT_GLOSSY_REFLECTION)
      c.reflected += v;
    if(s.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION)
      c.transmitted += v;
  }
  return c;
}

}  // namespace

// The albedo the lobe weights split the light by is the GGX lobe's with Schlick's Fresnel, as
// integrating the lobe gives it, over every roughness and angle: the baked table is the lobe's own
// (height-correlated Smith). The specular lobe, as the metal's adds its multiple scattering.
TEST(BsdfLobes, TheGgxAlbedoFitIsTheLobesIntegral)
{
  std::mt19937                          rng(19);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float alpha : {0.02f, 0.1f, 0.3f, 0.6f, 1.0f})
    for(float NoV : {0.1f, 0.3f, 0.6f, 1.0f})
      for(float f0 : {0.04f, 0.6f})
      {
        PbrMaterial mat = base(std::sqrt(alpha));
        const vec3  k1  = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
        const int   N   = 100000;
        double      albedo = 0.0;
        for(int i = 0; i < N; i++)
        {
          BsdfSampleData s;
          s.k1 = k1;
          s.xi = vec3(u(rng), u(rng), u(rng));
          brdf_ggx_smith_sample(s, mat, LOBE_SPECULAR_REFLECTION, vec3(f0), BSDF_FRESNEL_SCHLICK, 1.0f);
          if(s.event_type != BSDF_EVENT_ABSORB)
            albedo += s.bsdf_over_pdf.x / N;
        }
        EXPECT_NEAR(ggxSchlickAlbedo(f0, alpha, NoV), albedo, 0.01) << "alpha " << alpha << " NoV " << NoV << " f0 " << f0;
      }
}

// A dielectric over a white base reflects more toward grazing, as each microfacet's Fresnel says,
// and what it reflects comes out of the base: the whole stays within the light that arrived.
TEST(BsdfLobes, ADielectricReflectsMoreAtGrazingAndTakesItFromTheBase)
{
  PbrMaterial plastic = defaultPbrMaterial(vec3(1.0f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  float       last    = 0.0f;
  for(float NoV : {1.0f, 0.6f, 0.3f, 0.1f})
  {
    const Carried c = carriedAt(plastic, NoV);
    EXPECT_GT(c.reflected.x, last) << NoV;
    EXPECT_LE(c.all.x, 1.01f) << NoV;
    EXPECT_GT(c.all.x, 0.9f) << NoV;
    last = c.reflected.x;
  }
  // a smooth one reflects the dielectric's Fresnel at the view
  PbrMaterial smooth = defaultPbrMaterial(vec3(1.0f), 0.0f, 0.05f, vec3(0, 0, 1), vec3(0, 0, 1));
  for(float NoV : {1.0f, 0.5f, 0.2f})
    EXPECT_NEAR(carriedAt(smooth, NoV).reflected.x, ior_fresnel(1.5f, NoV), 0.01f) << NoV;
}

// glTF's metal reflects its base colour head-on and white at grazing (Schlick's from the base colour).
TEST(BsdfLobes, AMetalTurnsWhiteAtGrazing)
{
  PbrMaterial gold = defaultPbrMaterial(vec3(1.0f, 0.7f, 0.3f), 1.0f, 0.3f, vec3(0, 0, 1), vec3(0, 0, 1));
  const vec3  head = carriedAt(gold, 1.0f).all, graze = carriedAt(gold, 0.15f).all;
  EXPECT_GT(graze.z / graze.x, 1.5f * head.z / head.x);
  EXPECT_LE(graze.x, 1.01f);
}

// A white metal reflects all the light at every roughness: the multiple scattering a single-scatter
// lobe loses (30-50% at alpha 1) is given back (Turquin). A coloured one gets back less, by its
// colour. Sampling and evaluation agree on it (SamplingCarriesWhatEvaluationIntegratesTo).
TEST(BsdfLobes, ARoughWhiteMetalKeepsItsEnergy)
{
  std::mt19937                          rng(23);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float alpha : {0.1f, 0.3f, 0.6f, 1.0f})
    for(float NoV : {0.2f, 0.5f, 1.0f})
    {
      PbrMaterial white = defaultPbrMaterial(vec3(1.0f), 1.0f, std::sqrt(alpha), vec3(0, 0, 1), vec3(0, 0, 1));
      const vec3  k1    = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      const int   N     = 100000;
      double      albedo = 0.0, single = 0.0;
      for(int i = 0; i < N; i++)
      {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = vec3(u(rng), u(rng), u(rng));
        BsdfSampleData t = s;
        brdf_ggx_smith_sample(s, white, LOBE_METAL_REFLECTION, vec3(1.0f), BSDF_FRESNEL_SCHLICK, 1.0f);
        brdf_ggx_smith_sample(t, white, LOBE_SPECULAR_REFLECTION, vec3(1.0f), BSDF_FRESNEL_SCHLICK, 1.0f);
        if(s.event_type != BSDF_EVENT_ABSORB)
          albedo += s.bsdf_over_pdf.x / N;
        if(t.event_type != BSDF_EVENT_ABSORB)
          single += t.bsdf_over_pdf.x / N;
      }
      EXPECT_NEAR(albedo, 1.0, 0.015) << "alpha " << alpha << " NoV " << NoV;
      if(alpha >= 0.6f)
        EXPECT_LT(single, 0.9) << "alpha " << alpha << " NoV " << NoV;  // what it would lose
    }
  // A dark metal gets back its colour's share of the loss only
  EXPECT_NEAR(ggxMultiScatterFactor(vec3(0.1f), 1.0f, 1.0f).x, 1.0f + 0.1f * (1.0f / ggxSchlickAlbedo(1.0f, 1.0f, 1.0f) - 1.0f), 1e-5f);
}

// Glass splits each microfacet's light by its Fresnel: reflection and refraction together keep
// what the rough lobes do not lose, and the reflection grows toward grazing.
TEST(BsdfLobes, GlassReflectsAndRefractsByEachMicrofacetsFresnel)
{
  PbrMaterial glass  = defaultPbrMaterial(vec3(1.0f), 0.0f, 0.2f, vec3(0, 0, 1), vec3(0, 0, 1));
  glass.transmission = 1.0f;
  glass.thickness    = 0.01f;
  float last         = 0.0f;
  for(float NoV : {1.0f, 0.5f, 0.2f})
  {
    const Carried c = carriedAt(glass, NoV);
    EXPECT_GT(c.reflected.x, last) << NoV;
    EXPECT_LE(c.reflected.x + c.transmitted.x, 1.01f) << NoV;
    EXPECT_GT(c.reflected.x + c.transmitted.x, 0.93f) << NoV;
    last = c.reflected.x;
  }
}

// What a rough dielectric reflects is its GGX lobe weighed by each microfacet's own Fresnel, exactly:
// the albedo fit only decides how often the lobe is picked. One Fresnel at an estimated angle for the
// whole lobe was off at grazing views.
TEST(BsdfLobes, ARoughDielectricReflectsItsMicrofacetFresnel)
{
  std::mt19937                          rng(23);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float roughness : {0.4f, 0.7f})
    for(float NoV : {0.15f, 0.5f})
    {
      PbrMaterial mat = defaultPbrMaterial(vec3(1.0f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1));
      const vec3  k1  = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      const int   N   = 200000;
      double      microfacet = 0.0;
      for(int i = 0; i < N; i++)
      {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = vec3(u(rng), u(rng), u(rng));
        brdf_ggx_smith_sample(s, mat, LOBE_SPECULAR_REFLECTION, vec3(1.0f), BSDF_FRESNEL_DIELECTRIC, 1.5f);
        if(s.event_type != BSDF_EVENT_ABSORB)
          microfacet += s.bsdf_over_pdf.x / N;
      }
      EXPECT_NEAR(carriedAt(mat, NoV).reflected.x, microfacet, 0.02 * microfacet + 2e-3) << roughness << ", " << NoV;
    }
}

// The oracles of the app's docs/material-model-plan.md: what a change to the closure is read against.
namespace {

// A chi-square statistic in standard deviations (Wilson and Hilferty's normal approximation).
double chiSquareSigmas(double chi2, int dof)
{
  const double k = 2.0 / (9.0 * dof);
  return (std::cbrt(chi2 / dof) - (1.0 - k)) / std::sqrt(k);
}

struct SamplingFit
{
  double sigmas   = 0.0;
  int    dof      = 0;
  double absorbed = 0.0;  // the share of samples with no direction, and what the pdf says is left
  double missing  = 0.0;
};

// A sampler's directions against the pdf it reports: samples binned over the sphere, each bin's
// expected count from the pdf integrated over it. Bins expecting few samples are pooled. The grid's
// poles are on the Y axis, away from the views used, where lobes would fall inside one bin.
// `pdfOf(k2)` is the density, `draw(xi, k2)` a sample, false for none.
template <class PdfOf, class Draw>
SamplingFit samplingFit(const PdfOf& pdfOf, const Draw& draw, int N = 1000000)
{
  const int  kTheta = 24, kPhi = 48, kSub = 8;
  const auto cell   = [&](const vec3& d) {
    const int   t   = std::min(kTheta - 1, int((d.y * 0.5f + 0.5f) * kTheta));
    const float phi = std::atan2(d.z, d.x) / 6.2831853f + 0.5f;
    const int   p   = std::min(kPhi - 1, int(phi * kPhi));
    return t * kPhi + p;
  };

  std::vector<double> expected(size_t(kTheta * kPhi), 0.0), observed(size_t(kTheta * kPhi), 0.0);
  for(int t = 0; t < kTheta; t++)
    for(int p = 0; p < kPhi; p++)
    {
      double integral = 0.0;
      for(int a = 0; a < kSub; a++)
        for(int b = 0; b < kSub; b++)
        {
          const float y   = ((t + (a + 0.5f) / kSub) / kTheta) * 2.0f - 1.0f;
          const float phi = ((p + (b + 0.5f) / kSub) / kPhi - 0.5f) * 6.2831853f;
          const float r   = std::sqrt(std::max(0.0f, 1.0f - y * y));
          integral += pdfOf(vec3(r * std::cos(phi), y, r * std::sin(phi)));
        }
      // equal-area cells: y and phi are uniform over the sphere
      expected[size_t(t * kPhi + p)] = integral * (4.0 * 3.14159265358979 / (kTheta * kPhi * kSub * kSub)) * N;
    }

  std::mt19937                          rng(29);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  double                                absorb = 0.0;
  for(int i = 0; i < N; i++)
  {
    vec3 k2(0.0f);
    if(draw(vec3(u(rng), u(rng), u(rng)), k2))
      observed[size_t(cell(k2))] += 1.0;
    else
      absorb += 1.0;
  }

  SamplingFit fit;
  double      chi2 = 0.0, poolE = 0.0, poolO = 0.0, total = 0.0;
  for(size_t c = 0; c < expected.size(); c++)
  {
    total += expected[c];
    if(expected[c] < 20.0)
    {
      poolE += expected[c];
      poolO += observed[c];
      continue;
    }
    chi2 += (observed[c] - expected[c]) * (observed[c] - expected[c]) / expected[c];
    fit.dof++;
  }
  if(poolE >= 20.0)
  {
    chi2 += (poolO - poolE) * (poolO - poolE) / poolE;
    fit.dof++;
  }
  fit.sigmas   = chiSquareSigmas(chi2, std::max(1, fit.dof));
  fit.absorbed = absorb / N;
  fit.missing  = 1.0 - total / N;
  return fit;
}

// bsdfSample against bsdfEvaluateAll
SamplingFit samplingFit(const PbrMaterial& mat, const vec3& k1, int N = 1000000)
{
  return samplingFit(
      [&](const vec3& k2) {
        BsdfEvaluateData e;
        e.k1 = k1;
        e.k2 = k2;
        e.xi = vec3(0.5f);
        bsdfEvaluateAll(e, mat);
        return e.pdf;
      },
      [&](const vec3& xi, vec3& k2) {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = xi;
        bsdfSample(s, mat);
        k2 = s.k2;
        return s.event_type != BSDF_EVENT_ABSORB;
      },
      N);
}

}  // namespace

// The sampler draws directions by the density it reports. A mismatch is noise or bias no test of the
// mean finds: the mean of bsdf_over_pdf can come out right while MIS weighs each side wrongly. Sheen's
// flip used the xi.z that had just chosen the lobe, which at some views always flipped.
TEST(BsdfLobes, SamplesFollowTheReportedPdf)
{
  for(Case c : cases())
  {
    // a narrow lobe is finer than the bins, and the pdf's integral over a bin stops being one
    if(c.mat.transmission > 0.0f)
      c.mat.roughness = vec2(0.4f * 0.4f);
    for(const vec3& k1 : {normalize(vec3(0.0f, 0.0f, 1.0f)), normalize(vec3(0.4f, 0.2f, 0.9f)), normalize(vec3(0.9f, 0.1f, 0.3f))})
    {
      const SamplingFit fit = samplingFit(c.mat, k1);
      EXPECT_LT(fit.sigmas, 5.0) << fit.sigmas << " sigmas, " << c.name << " view " << k1.x << "," << k1.y << "," << k1.z << ", " << fit.dof << " bins";
      // what the pdf leaves out over the sphere is the share the sampler gives up on
      EXPECT_NEAR(fit.missing, fit.absorbed, 0.01) << c.name << " view " << k1.x << "," << k1.y << "," << k1.z;
    }
  }
}

// A layer over a white base gives back about what it was lit with, at every angle and roughness.
// The lobes weigh their microfacets by the exact Fresnel, and the layers below are charged by the albedo
// that gives (ggxDielectricAlbedo): charged by Schlick's, this gained 2%, and 3.7% under a coat.
TEST(BsdfLobes, ALayerOverAWhiteBaseAddsNoEnergy)
{
  const auto white = [](float roughness) { return defaultPbrMaterial(vec3(1.0f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1)); };
  for(float roughness : {0.05f, 0.2f, 0.5f, 0.9f})
    for(float NoV : {0.1f, 0.3f, 0.45f, 0.6f, 1.0f})
    {
      PbrMaterial coat        = white(roughness);
      coat.clearcoat          = 1.0f;
      coat.clearcoatRoughness = 0.2f;
      PbrMaterial sheen       = white(roughness);
      sheen.sheenColor        = vec3(1.0f);
      sheen.sheenRoughness    = 0.5f;
      PbrMaterial film        = white(roughness);
      film.iridescence        = 1.0f;
      film.iridescenceThickness = 0.4f;
      PbrMaterial all         = coat;
      all.sheenColor          = vec3(1.0f);
      all.sheenRoughness      = 0.5f;
      all.iridescence         = 1.0f;
      all.iridescenceThickness = 0.4f;
      const std::array<std::pair<const char*, PbrMaterial>, 5> layers{{{"none", white(roughness)}, {"coat", coat}, {"sheen", sheen}, {"film", film}, {"all", all}}};
      for(const auto& [name, mat] : layers)
      {
        const vec3 albedo = carriedAt(mat, NoV, 100000).all;
        EXPECT_LE(std::max({albedo.x, albedo.y, albedo.z}), 1.01f) << albedo.x << ", " << name << ", roughness " << roughness << ", NoV " << NoV;
      }
    }
}

// Where the model claims reciprocity, light and view can trade places: a lone Lambert or GGX lobe.
// It doesn't claim it for what is added over those: a lobe mixture's weights and a metal's multiple
// scattering read the view alone (computeLobeWeights, ggxMultiScatterFactor).
TEST(BsdfLobes, ALoneLobeIsReciprocal)
{
  std::mt19937 rng(31);
  PbrMaterial  mat = base(0.5f);
  mat.specular     = 0.0f;
  // eval's outputs carry the light side's cosine
  const auto lambert = [&](const vec3& k1, const vec3& k2) {
    BsdfEvaluateData e;
    e.k1 = k1;
    e.k2 = k2;
    e.xi = vec3(0.5f);
    bsdfEvaluateAll(e, mat);
    return sum(e.bsdf_diffuse + e.bsdf_glossy) / k2.z;
  };
  const auto ggx = [&](const vec3& k1, const vec3& k2) {
    BsdfEvaluateData e;
    e.k1 = k1;
    e.k2 = k2;
    e.xi = vec3(0.5f);
    brdf_ggx_smith_eval(e, mat, LOBE_SPECULAR_REFLECTION, vec3(0.6f), BSDF_FRESNEL_SCHLICK, 1.0f);
    return sum(e.bsdf_glossy) / k2.z;
  };
  for(int i = 0; i < 2000; i++)
  {
    const vec3 a = viewAbove(rng), b = viewAbove(rng);
    EXPECT_NEAR(lambert(a, b), lambert(b, a), 0.001f * lambert(a, b)) << "pair " << i;
    EXPECT_NEAR(ggx(a, b), ggx(b, a), 0.001f * ggx(a, b) + 1e-5f) << "pair " << i;
  }
}

// A coat darkens what is under it, past what it reflects: more over a dark base, and not at all over white
TEST(BsdfLobes, ACoatDarkensADarkBaseAndLeavesWhite)
{
  const auto diffuseOf = [](const PbrMaterial& mat) {
    vec3 tint = mat.baseColor;
    computeLobeWeights(mat, 1.0f, tint);
    return tint;
  };
  const auto coated = [](const vec3& colour, float roughness = 0.5f) {
    PbrMaterial mat        = defaultPbrMaterial(colour, 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1));
    mat.clearcoat          = 1.0f;
    mat.clearcoatRoughness = 0.1f;
    return mat;
  };
  EXPECT_NEAR(diffuseOf(coated(vec3(1.0f))).x, 1.0f, 1e-5f);
  // OpenPBR's (1 - K) / (1 - E K) over a rough dielectric, K about 0.58 and E the colour with the base's
  // 9% specular over it: a dark colour loses about half
  const vec3 dark = diffuseOf(coated(vec3(0.3f, 0.1f, 0.03f)));
  EXPECT_NEAR(dark.x / 0.3f, 0.53f, 0.02f);
  EXPECT_NEAR(dark.z / 0.03f, 0.45f, 0.02f);
  EXPECT_LT(dark.z / 0.03f, dark.x / 0.3f);  // it deepens the colour, too

  PbrMaterial half = coated(vec3(0.3f));
  half.clearcoat   = 0.5f;
  EXPECT_NEAR(diffuseOf(half).x / 0.3f, 0.5f * (1.0f + diffuseOf(coated(vec3(0.3f))).x / 0.3f), 1e-4f);

  PbrMaterial none = coated(vec3(0.3f));
  none.clearcoat   = 0.0f;
  EXPECT_FLOAT_EQ(diffuseOf(none).x, 0.3f);
}

// DAZ's second specular lobe is no layer; clear glass sends its light on, and from inside the surface
// nothing is under the coat
TEST(BsdfLobes, OnlyACoatOverAnOpaqueBaseDarkens)
{
  PbrMaterial lobe        = defaultPbrMaterial(vec3(0.3f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  lobe.specular2          = 1.0f;
  lobe.specular2Roughness = 0.1f;
  vec3 tint               = lobe.baseColor;
  computeLobeWeights(lobe, 1.0f, tint);
  EXPECT_FLOAT_EQ(tint.x, 0.3f);

  PbrMaterial glass        = lobe;
  glass.specular2          = 0.0f;
  glass.clearcoat          = 1.0f;
  glass.clearcoatRoughness = 0.1f;
  glass.transmission       = 1.0f;
  tint               = glass.baseColor;
  computeLobeWeights(glass, 1.0f, tint);
  EXPECT_NEAR(tint.x, 0.3f, 1e-4f);

  PbrMaterial inside = glass;
  inside.transmission = 0.0f;
  inside.ior1         = 1.5f;
  inside.ior2         = 1.0f;
  tint                = inside.baseColor;
  computeLobeWeights(inside, 1.0f, tint);
  EXPECT_FLOAT_EQ(tint.x, 0.3f);
}

// A coat's colour tints its own reflection and nothing else: what lies under it keeps the same share
TEST(BsdfLobes, ACoatsColourTintsOnlyItsReflection)
{
  PbrMaterial white        = defaultPbrMaterial(vec3(0.5f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  white.Nc                 = white.N;
  white.clearcoat          = 0.8f;
  white.clearcoatRoughness = 0.2f;
  PbrMaterial red          = white;
  red.clearcoatColor       = vec3(1.0f, 0.25f, 0.5f);

  for(float cosine : {1.0f, 0.5f, 0.15f})
  {
    vec3       ta = white.baseColor, tb = red.baseColor;
    const auto wa = computeLobeWeights(white, cosine, ta);
    const auto wb = computeLobeWeights(red, cosine, tb);
    for(int l = 0; l < LOBE_COUNT; l++)
      EXPECT_FLOAT_EQ(wa[l], wb[l]) << l;
    EXPECT_EQ(ta, tb);  // the colour under it doesn't change either
  }

  const vec3 view  = normalize(vec3(0.3f, 0.1f, 0.9f));
  const vec3 light = normalize(vec3(-0.25f, -0.1f, 0.9f));
  vec3       tint  = white.baseColor;
  BsdfEvaluateData a{}, b{};
  a.k1 = b.k1 = view;
  a.k2 = b.k2 = light;
  bsdfEvaluateLobe(a, white, LOBE_CLEARCOAT_REFLECTION, tint);
  bsdfEvaluateLobe(b, red, LOBE_CLEARCOAT_REFLECTION, tint);
  EXPECT_GT(a.bsdf_glossy.x, 0.0f);
  EXPECT_NEAR(b.bsdf_glossy.x, a.bsdf_glossy.x, 1e-5f);
  EXPECT_NEAR(b.bsdf_glossy.y, 0.25f * a.bsdf_glossy.y, 1e-5f);
  EXPECT_NEAR(b.bsdf_glossy.z, 0.5f * a.bsdf_glossy.z, 1e-5f);
  EXPECT_FLOAT_EQ(a.pdf, b.pdf);
}

// The second lobe weighs as a coat of its own would, the coat over it, and both over the rest
TEST(BsdfLobes, TheSecondLobeLiesUnderTheCoatAndOverTheRest)
{
  const PbrMaterial plain = defaultPbrMaterial(vec3(0.5f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  PbrMaterial       coat  = plain;
  coat.clearcoat          = 0.8f;
  coat.clearcoatRoughness = 0.2f;
  PbrMaterial lobe        = plain;
  lobe.specular2          = 0.8f;
  lobe.specular2Roughness = 0.2f;
  PbrMaterial both        = coat;
  both.specular2          = 0.6f;
  both.specular2Roughness = 0.4f;

  for(float cosine : {1.0f, 0.5f, 0.15f})
  {
    vec3       tint = plain.baseColor;
    const auto wc   = computeLobeWeights(coat, cosine, tint);
    tint            = plain.baseColor;
    const auto wl   = computeLobeWeights(lobe, cosine, tint);
    EXPECT_FLOAT_EQ(wl[LOBE_SPECULAR2_REFLECTION], wc[LOBE_CLEARCOAT_REFLECTION]);
    EXPECT_FLOAT_EQ(wl[LOBE_CLEARCOAT_REFLECTION], 0.0f);

    tint             = plain.baseColor;
    const auto w     = computeLobeWeights(both, cosine, tint);
    const float coatShare = w[LOBE_CLEARCOAT_REFLECTION];
    EXPECT_FLOAT_EQ(coatShare, wc[LOBE_CLEARCOAT_REFLECTION]);
    EXPECT_GT(w[LOBE_SPECULAR2_REFLECTION], 0.0f);
    EXPECT_LE(w[LOBE_SPECULAR2_REFLECTION], (1.0f - coatShare) * 0.6f + 1e-5f);
    float sum = 0.0f;
    for(int l = 0; l < LOBE_COUNT; l++)
      sum += w[l];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }
}

// Its value and its pdf, by the lobe's own roughness on the base's normal; as a coat would have them
TEST(BsdfLobes, TheSecondLobeEvaluatesAsACoatOfItsRoughness)
{
  PbrMaterial lobe        = defaultPbrMaterial(vec3(0.5f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  lobe.Nc                 = lobe.N;
  lobe.specular2          = 0.7f;
  lobe.specular2Roughness = 0.2f;
  PbrMaterial coat        = lobe;
  coat.specular2          = 0.0f;
  coat.clearcoat          = 0.7f;
  coat.clearcoatRoughness = 0.2f;

  const vec3 view  = normalize(vec3(0.3f, 0.1f, 0.9f));
  const vec3 light = normalize(vec3(-0.25f, -0.1f, 0.9f));
  vec3       tint  = lobe.baseColor;
  BsdfEvaluateData a{}, b{};
  a.k1 = b.k1 = view;
  a.k2 = b.k2 = light;
  bsdfEvaluateLobe(a, lobe, LOBE_SPECULAR2_REFLECTION, tint);
  bsdfEvaluateLobe(b, coat, LOBE_CLEARCOAT_REFLECTION, tint);
  EXPECT_GT(a.pdf, 0.0f);
  EXPECT_NEAR(a.pdf, b.pdf, 1e-3f * b.pdf);
  EXPECT_NEAR(a.bsdf_glossy.x, b.bsdf_glossy.x, 1e-3f * b.bsdf_glossy.x);
}

// The diffuse a path takes under a coat carries the darkening, whichever lobe it was drawn by
TEST(BsdfLobes, TheDarkeningReachesTheLobes)
{
  PbrMaterial plain = defaultPbrMaterial(vec3(0.3f), 0.0f, 0.6f, vec3(0, 0, 1), vec3(0, 0, 1));
  plain.specular    = 0.0f;
  PbrMaterial coat  = plain;
  coat.clearcoat          = 1.0f;
  coat.clearcoatRoughness = 0.3f;
  const float plainAlbedo = carriedAt(plain, 1.0f, 200000).all.x;
  const float coatAlbedo  = carriedAt(coat, 1.0f, 200000).all.x;
  // over 0.3, K about 0.6: 0.3 * 0.58 / 0.96 below the coat, plus the coat's own 4%
  EXPECT_LT(coatAlbedo, 0.8f * plainAlbedo);
  EXPECT_NEAR(coatAlbedo, 0.04f + 0.96f * 0.3f * 0.49f, 0.02f);
}

// Emission lies under a coat: out through what it doesn't reflect, and what it doesn't give back to the base.
// Not with no coat.
TEST(BsdfLobes, EmissionComesOutThroughACoat)
{
  const vec3 base(0.3f);
  EXPECT_EQ(bsdfCoatEmission(0.0f, 0.1f, base, 0.0f, 0.0f, 0.25f, 1.0f), vec3(1.0f));
  const vec3 straight = bsdfCoatEmission(1.0f, 0.1f, base, 0.0f, 0.0f, 0.25f, 1.0f);
  const vec3 grazing  = bsdfCoatEmission(1.0f, 0.1f, base, 0.0f, 0.0f, 0.25f, 0.2f);
  EXPECT_NEAR(straight.x, 0.96f * bsdfCoatDarkening(1.0f, base, 0.0f, 0.0f, 0.25f).x, 0.005f);
  EXPECT_LT(grazing.x, straight.x);
  // half the coverage takes less
  EXPECT_GT(bsdfCoatEmission(0.5f, 0.1f, base, 0.0f, 0.0f, 0.25f, 1.0f).x, straight.x);
}
