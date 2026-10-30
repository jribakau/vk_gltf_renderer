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
#include "../shaders/openpbr/fv_openpbr.h.slang"

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

// Phase M1's oracles (the app's docs/material-model-plan.md): what the OpenPBR closure is read against.
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
// Today's lobe mixture gains up to 2.6% (a coat at a view cosine of 0.6, 1.4% with no layer at all):
// the ceiling is what it does now, for the OpenPBR closure's albedo scaling to be read against.
TEST(BsdfLobes, ALayerOverAWhiteBaseAddsNoEnergy)
{
  const auto white = [](float roughness) { return defaultPbrMaterial(vec3(1.0f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1)); };
  for(float roughness : {0.2f, 0.5f, 0.9f})
    for(float NoV : {0.1f, 0.3f, 0.6f, 1.0f})
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
        EXPECT_LE(std::max({albedo.x, albedo.y, albedo.z}), 1.03f) << albedo.x << ", " << name << ", roughness " << roughness << ", NoV " << NoV;
      }
    }
}

// Where the model claims reciprocity, light and view can trade places: a lone Lambert or GGX lobe.
// It doesn't claim it for what is added over those: a lobe mixture's weights and a metal's multiple
// scattering read the view alone (computeLobeWeights, ggxMultiScatterFactor), and the OpenPBR
// closure's layering is to be read against that.
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

// Phase M2: the OpenPBR closure behind the path tracer's interface (shaders/openpbr/fv_openpbr.h.slang)
namespace {

constexpr float kAlpha = 0.25f;

OpenPBR_ResolvedInputs inputsOf(const PbrMaterial& mat, const vec3& V = vec3(0, 0, 1))
{
  return fvOpenPbrInputs(mat, V);
}

// The closure as the path tracer calls it, for the fit above
SamplingFit openPbrFit(const PbrMaterial& mat, const vec3& k1, int N = 1000000)
{
  const OpenPBR_PreparedBsdf prepared = fvOpenPbrPrepare(mat, k1);
  return samplingFit(
      [&](const vec3& k2) {
        BsdfEvaluateData e;
        e.k1 = k1;
        e.k2 = k2;
        e.xi = vec3(0.5f);
        fvOpenPbrEvaluate(e, prepared, mat);
        return e.pdf;
      },
      [&](const vec3& xi, vec3& k2) {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = xi;
        fvOpenPbrSample(s, prepared, mat);
        k2 = s.k2;
        return s.event_type != BSDF_EVENT_ABSORB;
      },
      N);
}

}  // namespace

// The equivalences that are exact (the KHR extensions' OpenPBR alignment): a parameter comes through
// as itself, in OpenPBR's units.
TEST(OpenPbr, TheKhrParametersThatMapExactlyDo)
{
  PbrMaterial mat = base(0.5f);
  mat.metallic    = 0.7f;
  mat.ior2        = 1.33f;
  EXPECT_EQ(inputsOf(mat).base_color, mat.baseColor);
  EXPECT_EQ(inputsOf(mat).base_metalness, 0.7f);
  EXPECT_EQ(inputsOf(mat).specular_ior, 1.33f);

  // inside a volume evaluateMaterial swaps the IORs: the material's own is still the one
  PbrMaterial inside = base(0.5f);
  inside.transmission = 1.0f;
  inside.thickness    = 0.01f;
  inside.ior1         = 1.33f;
  inside.ior2         = 1.0f;
  EXPECT_EQ(inputsOf(inside).specular_ior, 1.33f);

  PbrMaterial glass       = base(0.2f);
  glass.transmission      = 0.8f;
  glass.transmissionTint  = vec3(0.5f);
  glass.dispersion        = 2.0f;
  EXPECT_EQ(inputsOf(glass).transmission_weight, 0.8f);
  EXPECT_EQ(inputsOf(glass).transmission_color, glass.baseColor * 0.5f);
  EXPECT_EQ(inputsOf(glass).transmission_depth, 0.0f);  // the path tracer's walk holds the medium
  EXPECT_EQ(inputsOf(glass).transmission_dispersion_abbe_number, 10.0f);
  EXPECT_EQ(inputsOf(glass).transmission_dispersion_scale, 1.0f);

  PbrMaterial film          = base(0.3f);
  film.iridescence          = 0.9f;
  film.iridescenceThickness = 400.0f;  // nm
  film.iridescenceIor       = 1.3f;
  EXPECT_EQ(inputsOf(film).thin_film_weight, 0.9f);
  EXPECT_NEAR(inputsOf(film).thin_film_thickness, 0.4f, 1e-6f);  // microns
  EXPECT_EQ(inputsOf(film).thin_film_ior, 1.3f);
}

// A roughness pair comes back from OpenPBR's anisotropy as the same two alphas
TEST(OpenPbr, AnisotropyKeepsBothAlphas)
{
  for(const vec2 alpha : {vec2(0.3f, 0.3f), vec2(0.6f, 0.1f), vec2(0.04f, 0.2f), vec2(1.0f, 0.5f)})
  {
    PbrMaterial mat = base(0.5f);
    mat.roughness   = alpha;
    const auto  in  = inputsOf(mat);
    const float r2  = in.specular_roughness * in.specular_roughness;
    const float a   = in.specular_roughness_anisotropy;
    const float big = r2 * std::sqrt(2.0f / (1.0f + (1.0f - a) * (1.0f - a)));
    EXPECT_NEAR(big, std::max(alpha.x, alpha.y), 1e-5f);
    EXPECT_NEAR((1.0f - a) * big, std::min(alpha.x, alpha.y), 1e-5f);
  }
}

// The approximate ones, by what each is defined to keep
TEST(OpenPbr, TheApproximateEquivalencesKeepWhatTheyAreFor)
{
  // KHR_materials_specular's factor and colour scale F0 as OpenPBR's weight and colour do
  PbrMaterial dielectric   = base(0.4f);
  dielectric.specular      = 0.5f;
  dielectric.specularColor = vec3(0.6f, 0.7f, 0.8f);
  EXPECT_EQ(inputsOf(dielectric).specular_weight, 0.5f);
  EXPECT_EQ(inputsOf(dielectric).specular_color, dielectric.specularColor);
  // a metal has neither: its specular_color would be OpenPBR's edge tint
  dielectric.metallic = 1.0f;
  EXPECT_EQ(inputsOf(dielectric).specular_weight, 1.0f);
  EXPECT_EQ(inputsOf(dielectric).specular_color, vec3(1.0f));

  // sheen: its colour as weight and tint, whichever the brightest channel is
  PbrMaterial sheen    = base(0.5f);
  sheen.sheenColor     = vec3(0.5f, 0.25f, 0.1f);
  sheen.sheenRoughness = 0.6f;
  EXPECT_EQ(inputsOf(sheen).fuzz_weight, 0.5f);
  EXPECT_EQ(inputsOf(sheen).fuzz_color * inputsOf(sheen).fuzz_weight, sheen.sheenColor);
  EXPECT_EQ(inputsOf(sheen).fuzz_roughness, 0.6f);

  // clearcoat: OpenPBR's coat at KHR's fixed index, and the base darkens under it (decision M10)
  PbrMaterial coat        = base(0.5f);
  coat.clearcoat          = 0.75f;
  coat.clearcoatRoughness = 0.2f;
  EXPECT_EQ(inputsOf(coat).coat_weight, 0.75f);
  EXPECT_EQ(inputsOf(coat).coat_roughness, 0.2f);
  EXPECT_EQ(inputsOf(coat).coat_ior, 1.5f);
  EXPECT_EQ(inputsOf(coat).coat_darkening, 1.0f);
}

// What has an inside to the closure: only a surface that refracts. A walk's diffuse entry is a thin
// wall's diffuse transmission, its share of the diffuse bucket going through.
TEST(OpenPbr, OnlyARefractingSurfaceHasAnInside)
{
  PbrMaterial opaque = base(0.5f);
  opaque.thickness   = 0.01f;
  EXPECT_TRUE(inputsOf(opaque).geometry_thin_walled);

  PbrMaterial glass  = base(0.2f);
  glass.transmission = 1.0f;
  glass.thickness    = 0.01f;
  EXPECT_FALSE(inputsOf(glass).geometry_thin_walled);
  glass.thickness = 0.0f;
  EXPECT_TRUE(inputsOf(glass).geometry_thin_walled);

  PbrMaterial entry                    = base(0.5f);
  entry.thickness                      = 0.01f;
  entry.diffuseTransmissionFactor      = 0.75f;
  entry.diffuseTransmissionColorFactor = vec3(0.9f, 0.8f, 0.7f);
  const auto in                        = inputsOf(entry);
  EXPECT_TRUE(in.geometry_thin_walled);
  EXPECT_EQ(in.subsurface_weight, 0.75f);
  EXPECT_EQ(in.subsurface_color, entry.diffuseTransmissionColorFactor);
  EXPECT_EQ(in.subsurface_scatter_anisotropy, 1.0f);

  // the share that goes through: a white wall sends 3/4 of its light on, in its colour
  std::mt19937                          rng(37);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  entry.baseColor = vec3(1.0f);
  entry.specular  = 0.0f;
  const vec3 V(0.0f, 0.0f, 1.0f);
  const auto prepared = fvOpenPbrPrepare(entry, V);
  vec3       through(0.0f), back(0.0f);
  const int  N = 100000;
  for(int i = 0; i < N; i++)
  {
    BsdfSampleData s;
    s.k1 = V;
    s.xi = vec3(u(rng), u(rng), u(rng));
    fvOpenPbrSample(s, prepared, entry);
    if(s.event_type == BSDF_EVENT_DIFFUSE_TRANSMISSION)
      through += s.bsdf_over_pdf / float(N);
    else if(s.event_type == BSDF_EVENT_DIFFUSE_REFLECTION)
      back += s.bsdf_over_pdf / float(N);
  }
  EXPECT_NEAR(through.x, 0.75f * 0.9f, 0.02f);
  EXPECT_NEAR(through.z, 0.75f * 0.7f, 0.02f);
  EXPECT_NEAR(back.x, 0.25f, 0.02f);
}

// A view from inside refracts out: the normal the closure sees faces away from the view
TEST(OpenPbr, InsideAVolumeTheFrameFacesAway)
{
  PbrMaterial glass  = base(0.3f);
  glass.transmission = 1.0f;
  glass.thickness    = 0.01f;
  const vec3 V       = normalize(vec3(0.3f, 0.1f, 0.9f));
  EXPECT_GT(dot(inputsOf(glass, V).geometry_basis.n, V), 0.0f);
  glass.ior1 = 1.5f;
  glass.ior2 = 1.0f;
  EXPECT_LT(dot(inputsOf(glass, V).geometry_basis.n, V), 0.0f);
}

// A normal map bent past the view is still the outside
TEST(OpenPbr, ABumpBentPastTheViewIsNotTheInside)
{
  PbrMaterial bumped = base(0.4f);
  bumped.N           = normalize(vec3(0.7f, 0.0f, 1.0f));
  orthonormalBasis(bumped.N, bumped.T, bumped.B);
  const vec3 V = normalize(vec3(-1.0f, 0.0f, 0.3f));
  ASSERT_LT(dot(bumped.N, V), 0.0f);
  EXPECT_GT(dot(inputsOf(bumped, V).geometry_basis.n, V), 0.0f);
  EXPECT_TRUE(fvOpenPbrTakes(bumped));
}

// Retroreflection has no OpenPBR lobe: such a material stays on today's closure
TEST(OpenPbr, RetroreflectionStaysOnTodaysClosure)
{
  PbrMaterial retro     = base(0.4f);
  retro.retroreflection = 0.5f;
  EXPECT_FALSE(fvOpenPbrTakes(retro));
}

// The sampler draws directions by the density the closure reports (as the oracle above does for
// today's), and its weight is the value over that density.
TEST(OpenPbr, SamplesFollowTheReportedPdf)
{
  for(Case c : cases())
  {
    if(!fvOpenPbrTakes(c.mat))
      continue;
    if(c.mat.transmission > 0.0f)
      c.mat.roughness = vec2(0.4f * 0.4f);
    for(const vec3& k1 : {normalize(vec3(0.0f, 0.0f, 1.0f)), normalize(vec3(0.4f, 0.2f, 0.9f)), normalize(vec3(0.9f, 0.1f, 0.3f))})
    {
      const SamplingFit fit = openPbrFit(c.mat, k1);
      EXPECT_LT(fit.sigmas, 5.0) << fit.sigmas << " sigmas, " << c.name << " view " << k1.x << "," << k1.y << "," << k1.z << ", " << fit.dof << " bins";
      EXPECT_NEAR(fit.missing, fit.absorbed, 0.01) << c.name << " view " << k1.x << "," << k1.y << "," << k1.z;
    }
  }
}

TEST(OpenPbr, ASampleWeighsItsValueOverItsPdf)
{
  std::mt19937                          rng(43);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(Case c : cases())
  {
    if(!fvOpenPbrTakes(c.mat))
      continue;
    for(int i = 0; i < 500; i++)
    {
      const vec3 V        = viewAbove(rng);
      const auto prepared = fvOpenPbrPrepare(c.mat, V);
      BsdfSampleData s;
      s.k1 = V;
      s.xi = vec3(u(rng), u(rng), u(rng));
      fvOpenPbrSample(s, prepared, c.mat);
      if(s.event_type == BSDF_EVENT_ABSORB)
        continue;
      BsdfEvaluateData e;
      e.k1 = V;
      e.k2 = s.k2;
      e.xi = s.xi;
      fvOpenPbrEvaluate(e, prepared, c.mat);
      ASSERT_GT(e.pdf, 0.0f) << c.name;
      const float weight = sum(s.bsdf_over_pdf), ratio = sum(e.bsdf_diffuse + e.bsdf_glossy) / e.pdf;
      EXPECT_NEAR(weight, ratio, 2e-3f * std::max(weight, 1.0f)) << c.name << ", sample " << i;
    }
  }
}

// What the plan asks of the layering: a layer over a white base gives back what it was lit with.
// Today's mixture gains up to 2.6% here (BsdfLobes.ALayerOverAWhiteBaseAddsNoEnergy).
TEST(OpenPbr, ALayerOverAWhiteBaseKeepsItsEnergy)
{
  std::mt19937                          rng(47);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const auto white = [](float roughness) { return defaultPbrMaterial(vec3(1.0f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1)); };
  for(float roughness : {0.2f, 0.5f, 0.9f})
    for(float NoV : {0.1f, 0.3f, 0.6f, 1.0f})
    {
      PbrMaterial coat          = white(roughness);
      coat.clearcoat            = 1.0f;
      coat.clearcoatRoughness   = 0.2f;
      PbrMaterial sheen         = white(roughness);
      sheen.sheenColor          = vec3(1.0f);
      sheen.sheenRoughness      = 0.5f;
      PbrMaterial all           = coat;
      all.sheenColor            = vec3(1.0f);
      all.sheenRoughness        = 0.5f;
      const std::array<std::pair<const char*, PbrMaterial>, 4> layers{{{"none", white(roughness)}, {"coat", coat}, {"sheen", sheen}, {"all", all}}};
      const vec3 V(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      for(const auto& [name, mat] : layers)
      {
        const auto prepared = fvOpenPbrPrepare(mat, V);
        double     albedo   = 0.0;
        const int  N        = 100000;
        for(int i = 0; i < N; i++)
        {
          BsdfSampleData s;
          s.k1 = V;
          s.xi = vec3(u(rng), u(rng), u(rng));
          fvOpenPbrSample(s, prepared, mat);
          if(s.event_type != BSDF_EVENT_ABSORB)
            albedo += s.bsdf_over_pdf.x / N;
        }
        EXPECT_LE(albedo, 1.005) << name << ", roughness " << roughness << ", NoV " << NoV;
        EXPECT_GE(albedo, 0.95) << name << ", roughness " << roughness << ", NoV " << NoV;
      }
    }
}

// A coat darkens what is under it, and tints nothing it has no colour for: today's mixture can't
TEST(OpenPbr, ACoatDarkensADiffuseBase)
{
  std::mt19937                          rng(53);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  PbrMaterial plain = defaultPbrMaterial(vec3(0.3f), 0.0f, 0.6f, vec3(0, 0, 1), vec3(0, 0, 1));
  plain.specular    = 0.0f;
  PbrMaterial coated = plain;
  coated.clearcoat          = 1.0f;
  coated.clearcoatRoughness = 0.3f;
  const vec3 V(0.0f, 0.0f, 1.0f);
  const auto diffuseOf = [&](const PbrMaterial& mat) {
    const auto prepared = fvOpenPbrPrepare(mat, V);
    double     out      = 0.0;
    const int  N        = 200000;
    for(int i = 0; i < N; i++)
    {
      BsdfSampleData s;
      s.k1 = V;
      s.xi = vec3(u(rng), u(rng), u(rng));
      fvOpenPbrSample(s, prepared, mat);
      if(s.event_type == BSDF_EVENT_DIFFUSE_REFLECTION)
        out += s.bsdf_over_pdf.x / N;
    }
    return out;
  };
  EXPECT_LT(diffuseOf(coated), 0.9 * diffuseOf(plain));
}

// What the closure doesn't have is kept on top of it: a bumped lobe's terminator, the occlusion map
TEST(OpenPbr, TheTerminatorAndOcclusionStillApply)
{
  PbrMaterial bumped = base(0.5f);
  bumped.Ns          = vec3(0, 0, 1);
  bumped.N           = normalize(vec3(0.5f, 0.0f, 1.0f));
  orthonormalBasis(bumped.N, bumped.T, bumped.B);
  const vec3 V(0.0f, 0.0f, 1.0f);
  const auto prepared = fvOpenPbrPrepare(bumped, V);
  // a light just above the bent normal's horizon, on the side it leans away from
  const vec3       L = normalize(vec3(-0.6f, 0.0f, 0.6f));
  ASSERT_GT(dot(L, bumped.N), 0.0f);
  BsdfEvaluateData e;
  e.k1 = V;
  e.k2 = L;
  e.xi = vec3(0.5f);
  fvOpenPbrEvaluate(e, prepared, bumped);
  const vec3 fadedDiffuse = e.bsdf_diffuse;
  EXPECT_LT(shadowTerminatorTerm(L, bumped.N, bumped.Ns), 1.0f);
  const OpenPBR_DiffuseSpecular raw = openpbr_eval(prepared, L);
  EXPECT_LT(fadedDiffuse.x, raw.diffuse.x);

  PbrMaterial occluded = base(0.5f);
  occluded.occlusion   = 0.25f;
  PbrMaterial open     = base(0.5f);
  const vec3  light    = normalize(vec3(0.2f, 0.1f, 0.9f));
  BsdfEvaluateData a, b;
  a.k1 = b.k1 = V;
  a.k2 = b.k2 = light;
  a.xi = b.xi = vec3(0.5f);
  fvOpenPbrEvaluate(a, fvOpenPbrPrepare(occluded, V), occluded);
  fvOpenPbrEvaluate(b, fvOpenPbrPrepare(open, V), open);
  EXPECT_NEAR(a.bsdf_diffuse.x, 0.25f * b.bsdf_diffuse.x, 1e-5f);
  EXPECT_NEAR(a.pdf, b.pdf, 1e-5f);
}
