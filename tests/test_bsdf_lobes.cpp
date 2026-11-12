// nvshaders' BSDF (bsdf_functions.h.slang, a C++/Slang polyglot) compiled as C++: next-event
// estimation's every-lobe evaluation against the single random lobe it replaces, the two sides of
// its MIS pair, and sampling against evaluation for the lobes whose upstream code was fixed with it.
// Its own executable: the header needs GLM's swizzles, which the glm target turns off.

#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

// The ground's water surface seen from above, as waterSurfaceMaterial makes it, at GGX/Beckmann alpha
// `alpha`; `gaussian` false for the GGX lobe it had
PbrMaterial water(float alpha, bool gaussian = true)
{
  PbrMaterial mat    = defaultPbrMaterial(vec3(1.0f), 0.0f, 0.0f, vec3(0, 0, 1), vec3(0, 0, 1));
  mat.roughness      = vec2(alpha);
  mat.transmission   = 1.0f;
  mat.thickness      = 1.0f;
  mat.ior1           = 1.0f;
  mat.ior2           = 1.33f;
  mat.gaussianSlopes = gaussian;
  return mat;
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
  // Lobes picked apart from their weights (bsdfLobePick): a near-black base under its specular, and
  // under a smooth coat, whose diffuse is picked at its floor
  out.push_back({"dark glossy", defaultPbrMaterial(vec3(0.04f), 0.0f, 0.3f, vec3(0, 0, 1), vec3(0, 0, 1))});
  PbrMaterial darkCoat        = defaultPbrMaterial(vec3(0.005f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  darkCoat.clearcoat          = 1.0f;
  darkCoat.clearcoatRoughness = 0.15f;
  out.push_back({"dark coat", darkCoat});
  // A window pane drawn as one surface: thin-walled glass reflecting at both faces (plate_fresnel)
  PbrMaterial pane  = defaultPbrMaterial(vec3(0.9f), 0.0f, 0.3f, vec3(0, 0, 1), vec3(0, 0, 1));
  pane.transmission = 1.0f;
  pane.plate        = true;
  out.push_back({"pane", pane});
  PbrMaterial thinGlass = pane;
  thinGlass.plate       = false;
  out.push_back({"thin glass", thinGlass});
  // The ground's water (waterSurfaceMaterial): a medium's surface whose ripples are Gaussian slopes,
  // sampled by GGX's lobe (bsdfGaussianSlopeShare); and the same slopes on a reflection lobe alone
  out.push_back({"water", water(0.15f)});
  PbrMaterial gaussianPlastic    = base(0.4f);
  gaussianPlastic.gaussianSlopes = true;
  out.push_back({"gaussian plastic", gaussianPlastic});
  // A named metal's edge tint (OpenPBR's F82), darker than any measured metal's. Last, so the cases
  // above keep the random numbers the tests that share one stream draw for them.
  PbrMaterial namedMetal = base(0.3f);
  namedMetal.metallic    = 1.0f;
  namedMetal.edgeTint    = vec3(0.6f, 0.75f, 0.9f);
  out.push_back({"metal with an edge tint", namedMetal});
  return out;
}

PbrMaterial caseNamed(const char* name)
{
  for(const Case& c : cases())
    if(std::strcmp(c.name, name) == 0)
      return c.mat;
  ADD_FAILURE() << "no case " << name;
  return base(0.5f);
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

// Where a sample of `mat` can leave toward k2 (bsdfFaceAllows), for the cases above: with reflection
// lobes only, the face's side. The transmitting cases are flat, where each lobe's own side is the face's.
bool reachable(const PbrMaterial& mat, const vec3& k2)
{
  const bool transmits = mat.transmission > 0.0f || mat.diffuseTransmissionFactor > 0.0f;
  return transmits || dot(k2, mat.Ng) > 0.0f;
}

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

// The path tracer judges a bounce by the lobe bsdfSampleCore says took it (bsdfLobeRoughness), not by
// the material's smoothest lobe: each lobe's own roughness, 1 for the broad ones.
TEST(BsdfLobes, ALobeIsAsRoughAsItsOwnRoughness)
{
  PbrMaterial mat         = base(0.6f);
  mat.roughness           = vec2(0.04f, 0.36f);  // anisotropic: the larger alpha, perceptual 0.6
  mat.clearcoat           = 1.0f;
  mat.clearcoatRoughness  = 0.05f;
  mat.specular2           = 0.5f;
  mat.specular2Roughness  = 0.2f;
  mat.sheenColor          = vec3(0.5f);
  mat.sheenRoughness      = 0.05f;
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_CLEARCOAT_REFLECTION), 0.05f);
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_SPECULAR2_REFLECTION), 0.2f);
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_SPECULAR_REFLECTION), 0.6f);
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_METAL_REFLECTION), 0.6f);
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_SPECULAR_TRANSMISSION), 0.6f);
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_DIFFUSE_REFLECTION), 1.0f);
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_DIFFUSE_TRANSMISSION), 1.0f);
  EXPECT_FLOAT_EQ(bsdfLobeRoughness(mat, LOBE_SHEEN_REFLECTION), 1.0f);  // broad at any roughness
}

// The lobe bsdfSampleCore returns is the one the sample came from: lobes come back as often as
// bsdfLobePick picks them, each with its own kind of event. Under a sharp coat (car paint, a PBRSkin top
// coat) the base's glossy samples then read as rough, where the smoothest-lobe rule called every glossy
// sample sharp.
TEST(BsdfLobes, ASampleReturnsTheLobeThatTookIt)
{
  PbrMaterial paint        = base(0.6f);
  paint.clearcoat          = 1.0f;
  paint.clearcoatRoughness = 0.05f;
  PbrMaterial cloth        = base(0.5f);
  cloth.sheenColor         = vec3(0.6f);
  cloth.sheenRoughness     = 0.3f;
  PbrMaterial skin         = base(0.45f);
  skin.specular2           = 0.6f;
  skin.specular2Roughness  = 0.08f;
  const Case materials[]   = {{"car paint", paint}, {"sheen cloth", cloth}, {"skin's second lobe", skin}};

  std::mt19937                          rng(41);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(const Case& c : materials)
  {
    const vec3 k1   = normalize(vec3(0.4f, 0.2f, 1.0f));
    vec3       tint = c.mat.baseColor;
    const auto s    = computeLobeWeights(c.mat, dot(k1, c.mat.N), tint);
    const auto P    = bsdfLobePick(c.mat, s, tint);

    const int               N = 200000;
    std::array<int, LOBE_COUNT> picked{};
    int                     glossy = 0, roughGlossy = 0;
    for(int i = 0; i < N; i++)
    {
      BsdfSampleData d;
      d.k1           = k1;
      d.xi           = vec3(u(rng), u(rng), u(rng));
      const int lobe = bsdfSampleCore(d, c.mat, false);
      ASSERT_GE(lobe, 0) << c.name;
      ASSERT_LT(lobe, LOBE_COUNT) << c.name;
      picked[lobe]++;
      if(d.event_type == BSDF_EVENT_ABSORB)
        continue;
      if(lobe == LOBE_DIFFUSE_REFLECTION)
        EXPECT_EQ(d.event_type, BSDF_EVENT_DIFFUSE_REFLECTION) << c.name;
      else
        EXPECT_NE(d.event_type & (BSDF_EVENT_GLOSSY | BSDF_EVENT_IMPULSE), 0) << c.name << ", lobe " << lobe;
      if((d.event_type & BSDF_EVENT_GLOSSY) != 0)
      {
        glossy++;
        roughGlossy += bsdfLobeRoughness(c.mat, lobe) > 0.1f;  // DIRECT_PATH_ROUGHNESS
      }
    }
    for(int l = 0; l < LOBE_COUNT; l++)
    {
      const float p = P[l], sigma = std::sqrt(std::max(p * (1.0f - p), 1e-6f) / float(N));
      EXPECT_NEAR(float(picked[l]) / float(N), p, 5.0f * sigma + 1e-6f) << c.name << ", lobe " << l;
    }
    ASSERT_GT(glossy, 0) << c.name;
    const float roughShare = float(roughGlossy) / float(glossy);
    if(c.mat.clearcoat > 0.0f || c.mat.specular2 > 0.0f)
    {
      // A glossy lobe smoother than 0.1 beside a rougher one: under the old rule every glossy sample
      // read as smooth. Now the rougher lobe's share does not.
      EXPECT_GT(roughShare, 0.05f) << c.name;
      EXPECT_LT(roughShare, 0.95f) << c.name;
    }
    else
      EXPECT_EQ(roughShare, 1.0f) << c.name;  // the sheen and the specular, both rough
  }
}

// What sampling carries (the mean of bsdf_over_pdf) is the BSDF's integral over the directions a sample
// can leave by, as evaluation gives it. Retroreflection sampled twice its value, rough refraction's
// evaluation lacked eta_t^2 and the clearcoat's evaluation used the base normal's frame.
TEST(BsdfLobes, SamplingCarriesWhatEvaluationIntegratesTo)
{
  std::mt19937                          rng(13);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(Case c : cases())
  {
    // uniform directions cannot integrate the cornea's narrow refraction lobe: a rougher one, and
    // eight times the directions, without which water's (into 1.33, GGX or Gaussian) came out 5% off
    if(c.mat.transmission > 0.0f)
      c.mat.roughness = vec2(0.4f * 0.4f);
    const vec3 k1 = normalize(vec3(0.4f, 0.2f, 0.9f));
    const int  N  = c.mat.transmission > 0.0f ? 3200000 : 400000;
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
      if(reachable(c.mat, e.k2))
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
  metal.N           = caseNamed("normal map").N;
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
  const vec3 N = caseNamed("normal map").N;
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

namespace {

// pbrt-v4's FrComplex, typed from the Fresnel equations with Snell's law in complex numbers: the
// unpolarised reflectance of a conductor of index n + ik, from air, at incidence cosine `cosI`
double conductorReflectance(double cosI, double n, double k)
{
  using C       = std::complex<double>;
  const C eta(n, k);
  cosI          = std::clamp(cosI, 0.0, 1.0);
  const C sin2T = (1.0 - cosI * cosI) / (eta * eta);
  const C cosT  = std::sqrt(1.0 - sin2T);
  const C rParl = (eta * cosI - cosT) / (eta * cosI + cosT);
  const C rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
  return 0.5 * (std::norm(rParl) + std::norm(rPerp));
}

// Measured metals at one wavelength: gold at 550 and 450 nm (its blue, whose edge rises most) and
// chrome at 550 nm (Johnson and Christy 1972, 1974), aluminium at 550 nm (McPeak et al. 2015), and how
// far off Schlick's from their reflectance head-on is at least, somewhere
struct Conductor
{
  const char* name;
  float       n, k;
  double      schlickOff;
};
constexpr Conductor kConductors[] = {
    {"gold at 550 nm", 0.43f, 2.455f, 0.0},
    {"gold at 450 nm", 1.38f, 1.914f, 0.0},
    {"chrome at 550 nm", 3.18f, 3.33f, 0.15},
    {"aluminium at 550 nm", 0.7894f, 5.8519f, 0.05},
};

constexpr float kMu82 = 1.0f / 7.0f;  // where OpenPBR's F82 tint is defined, about 82 degrees

}  // namespace

// nvshaders' exact conductor Fresnel (fresnel_conductor, the thin film's base) is pbrt's
TEST(BsdfLobes, TheConductorFresnelIsPbrts)
{
  for(const Conductor& c : kConductors)
    for(float degrees : {0.0f, 30.0f, 60.0f, 75.0f, 82.0f, 88.0f})
    {
      const float cosI = std::cos(glm::radians(degrees));
      vec2        phaseSin, phaseCos;
      const vec2  sp = fresnel_conductor(phaseSin, phaseCos, 1.0f, c.n, c.k, cosI, 1.0f - cosI * cosI);
      EXPECT_NEAR(0.5f * (sp.x + sp.y), conductorReflectance(cosI, c.n, c.k), 2e-5) << c.name << " at " << degrees;
    }
}

// OpenPBR's F82 through a conductor's reflectance head-on and at 82 degrees follows the conductor at
// every angle, to 0.025: Schlick's from the same head-on reflectance is up to 0.2 too bright toward
// grazing (chrome), and 0.08 for aluminium. Head-on and at 82 degrees it is the conductor's.
TEST(BsdfLobes, F82FollowsAConductorWhereSchlickDoesNot)
{
  for(const Conductor& c : kConductors)
  {
    const float f0   = float(conductorReflectance(1.0, c.n, c.k));
    const float at82 = float(conductorReflectance(kMu82, c.n, c.k));
    const vec3  tint = vec3(at82 / schlickFresnel(f0, 1.0f, kMu82));
    EXPECT_NEAR(f82Fresnel(vec3(f0), tint, 1.0f).x, f0, 1e-6f) << c.name;
    EXPECT_NEAR(f82Fresnel(vec3(f0), tint, kMu82).x, at82, 1e-5f) << c.name;
    double worstF82 = 0.0, worstSchlick = 0.0;
    for(int degrees = 0; degrees < 90; degrees++)
    {
      const float  mu        = std::cos(glm::radians(float(degrees)));
      const double conductor = conductorReflectance(mu, c.n, c.k);
      worstF82               = std::max(worstF82, std::abs(f82Fresnel(vec3(f0), tint, mu).x - conductor));
      worstSchlick           = std::max(worstSchlick, std::abs(schlickFresnel(f0, 1.0f, mu) - conductor));
    }
    EXPECT_LT(worstF82, 0.025) << c.name;
    EXPECT_LE(worstF82, worstSchlick) << c.name;
    EXPECT_GE(worstSchlick, c.schlickOff) << c.name;
  }
  // a tint below white lowers only the edge's middle, and no tint is below 0
  EXPECT_LT(f82Fresnel(vec3(0.55f), vec3(0.74f), 0.5f).x, schlickFresnel(0.55f, 1.0f, 0.5f));
  EXPECT_NEAR(f82Fresnel(vec3(0.55f), vec3(0.74f), 0.0f).x, 1.0f, 1e-6f);
  EXPECT_GE(f82Fresnel(vec3(0.0f), vec3(0.0f), 0.4f).x, 0.0f);
}

// A rough gold in a white furnace reflects at most what came in: sample for sample no more than a white
// Schlick metal does, which itself gives back all of it (ARoughWhiteMetalKeepsItsEnergy). The edge tint
// takes its share off the multiple scattering too (f82AverageShare).
TEST(BsdfLobes, ARoughGoldStaysUnderOneInAFurnace)
{
  // figure/materials/metals.cpp's gold, and chrome's darker tint on it, with the least of the red each
  // keeps: chrome's tint takes gold's red at 78 degrees to about 0.76
  const vec3 goldF0(1.0f, 0.7327f, 0.3659f);
  const struct
  {
    vec3  tint;
    float red;
  } tints[] = {{vec3(0.9847f, 1.0f, 0.9804f), 0.95f}, {vec3(0.7418f, 0.748f, 0.815f), 0.7f}};
  for(const auto& [tint, red] : tints)
    for(float alpha : {0.1f, 0.3f, 0.6f, 1.0f})
      for(float NoV : {0.2f, 0.5f, 1.0f})
      {
        PbrMaterial gold  = defaultPbrMaterial(goldF0, 1.0f, std::sqrt(alpha), vec3(0, 0, 1), vec3(0, 0, 1));
        gold.edgeTint     = tint;
        PbrMaterial white = defaultPbrMaterial(vec3(1.0f), 1.0f, std::sqrt(alpha), vec3(0, 0, 1), vec3(0, 0, 1));
        std::mt19937                          rng(29);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        const vec3 k1 = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
        const int  N  = 50000;
        dvec3      carried(0.0);
        bool       neverMore = true;
        for(int i = 0; i < N; i++)
        {
          BsdfSampleData s;
          s.k1             = k1;
          s.xi             = vec3(u(rng), u(rng), u(rng));
          BsdfSampleData w = s;
          bsdfSample(s, gold);
          bsdfSample(w, white);
          if(s.event_type == BSDF_EVENT_ABSORB)
            continue;
          neverMore = neverMore && all(lessThanEqual(s.bsdf_over_pdf, w.bsdf_over_pdf * 1.000001f));
          carried += dvec3(s.bsdf_over_pdf) / double(N);
        }
        EXPECT_TRUE(neverMore) << "alpha " << alpha << " NoV " << NoV;
        for(int c = 0; c < 3; c++)
          EXPECT_LE(carried[c], 1.015) << "alpha " << alpha << " NoV " << NoV << " channel " << c;
        EXPECT_GT(carried.x, red) << "alpha " << alpha << " NoV " << NoV;  // gold's red, its F0 1
      }
}

// With no edge tint the metal lobe is today's, bit for bit: Schlick's from the base colour and
// Turquin's factor from it, as before the tint existed. F82 with a white tint is Schlick's bit for bit
// too, so a tint taken off leaves nothing behind.
TEST(BsdfLobes, AMetalWithoutAnEdgeTintIsSchlicksBitForBit)
{
  PbrMaterial metal = defaultPbrMaterial(vec3(0.95f, 0.64f, 0.54f), 1.0f, 0.4f, vec3(0, 0, 1), vec3(0, 0, 1));
  EXPECT_EQ(bsdfLobeFresnel(metal, LOBE_METAL_REFLECTION, 0.5f, metal.baseColor).fresnel, BSDF_FRESNEL_SCHLICK);
  PbrMaterial tinted = metal;
  tinted.edgeTint    = vec3(1.0f, 1.0f, 0.99f);
  EXPECT_EQ(bsdfLobeFresnel(tinted, LOBE_METAL_REFLECTION, 0.5f, tinted.baseColor).fresnel, BSDF_FRESNEL_F82);
  tinted.iridescence = 0.5f;  // the film keeps the plain base colour
  EXPECT_EQ(bsdfLobeFresnel(tinted, LOBE_METAL_REFLECTION, 0.5f, tinted.baseColor).fresnel, BSDF_FRESNEL_NONE);

  const auto same = [](vec3 a, vec3 b) { return std::memcmp(&a, &b, sizeof a) == 0; };
  std::mt19937 rng(37);
  for(int i = 0; i < 500; i++)
  {
    const vec3  k1 = viewAbove(rng), k2 = viewAbove(rng);
    const float kh = dot(k1, normalize(k1 + k2));
    // the Schlick path is the expressions it was
    EXPECT_TRUE(same(bsdfMetalMultiScatter(metal, LOBE_METAL_REFLECTION, metal.baseColor, BSDF_FRESNEL_SCHLICK, k1.z),
                     ggxMultiScatterFactor(metal.baseColor, 0.5f * (metal.roughness.x + metal.roughness.y), k1.z)));
    EXPECT_TRUE(same(bsdfMicrofacetFresnel(BSDF_FRESNEL_SCHLICK, 1.0f, metal.baseColor, metal.edgeTint, kh),
                     schlickFresnel(metal.baseColor, vec3(1.0f), kh)));
    // and F82 with a white tint gives the same bits, evaluated and sampled
    EXPECT_TRUE(same(f82Fresnel(metal.baseColor, vec3(1.0f), kh), schlickFresnel(metal.baseColor, vec3(1.0f), kh)));
    BsdfEvaluateData a, b;
    a.k1 = b.k1 = k1;
    a.k2 = b.k2 = k2;
    a.xi = b.xi = vec3(0.5f);
    brdf_ggx_smith_eval(a, metal, LOBE_METAL_REFLECTION, metal.baseColor, BSDF_FRESNEL_SCHLICK, 1.0f);
    brdf_ggx_smith_eval(b, metal, LOBE_METAL_REFLECTION, metal.baseColor, BSDF_FRESNEL_F82, 1.0f);
    EXPECT_TRUE(same(a.bsdf_glossy, b.bsdf_glossy)) << i;
    BsdfSampleData s, t;
    s.k1 = t.k1 = k1;
    s.xi = t.xi = vec3(float(i % 23) / 23.0f, float(i % 29) / 29.0f, 0.5f);
    brdf_ggx_smith_sample(s, metal, LOBE_METAL_REFLECTION, metal.baseColor, BSDF_FRESNEL_SCHLICK, 1.0f);
    brdf_ggx_smith_sample(t, metal, LOBE_METAL_REFLECTION, metal.baseColor, BSDF_FRESNEL_F82, 1.0f);
    EXPECT_TRUE(same(s.bsdf_over_pdf, t.bsdf_over_pdf)) << i;
  }
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

namespace {

// Heitz et al. 2016's microsurface random walk on a rough dielectric boundary of index `eta` (GGX
// slopes of `alpha`, uniform heights, every order of scattering, each microfacet's exact Fresnel): the
// share of the light from cosine `NoV` above it that leaves below. The ground truth the transmission
// table is baked from (utils/ggx_transmission_table.py), written again here.
double walkTransmission(double eta, double alpha, double NoV, int N, uint32_t seed)
{
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double                           inf = std::numeric_limits<double>::infinity();
  // Smith's Lambda, below the horizon -1 - Lambda(-w)
  auto lambda = [alpha](const dvec3& w) {
    if(w.z > 0.99999)
      return 0.0;
    if(w.z < -0.99999)
      return -1.0;
    const double a = w.z / (std::sqrt(std::max(1.0 - w.z * w.z, 1e-24)) * alpha);
    return 0.5 * (-1.0 + (a < 0.0 ? -1.0 : 1.0) * std::sqrt(1.0 + 1.0 / (a * a)));
  };
  auto c1 = [](double h) { return std::clamp(0.5 * (h + 1.0), 0.0, 1.0); };
  // where a ray along w from height h next meets the surface, infinity where it leaves
  auto nextHeight = [&](const dvec3& w, double h, double x) {
    if(w.z > 0.9999)
      return inf;
    if(w.z < -0.9999)
      return 2.0 * x * c1(h) - 1.0;
    if(std::abs(w.z) < 1e-4)
      return h;
    const double l  = lambda(w);
    const double g1 = w.z > 0.0 ? std::pow(c1(h), l) : 0.0;
    if(x > 1.0 - g1)
      return inf;
    return 2.0 * c1(h) / std::pow(1.0 - x, 1.0 / l) - 1.0;
  };
  // the normals visible from any direction, below the horizon too (Dupuy and Benyoub 2023's cap)
  auto visibleNormal = [&](const dvec3& wi) {
    const dvec3  s   = normalize(dvec3(wi.x * alpha, wi.y * alpha, wi.z));
    const double phi = (2.0 * u(rng) - 1.0) * 3.14159265358979;
    const double z   = (1.0 - u(rng)) * (1.0 + s.z) - s.z;
    const double r   = std::sqrt(std::clamp(1.0 - z * z, 0.0, 1.0));
    const dvec3  m   = dvec3(r * std::cos(phi), r * std::sin(phi), z) + s;
    const dvec3  n   = dvec3(m.x * alpha, m.y * alpha, m.z);
    return length(n) > 1e-12 ? normalize(n) : dvec3(0.0, 0.0, 1.0);
  };
  auto fresnel = [](double c, double e) {
    c               = std::abs(c);
    const double g2 = e * e - 1.0 + c * c;
    if(g2 < 0.0)
      return 1.0;
    const double g = std::sqrt(g2), a = (g - c) / (g + c), b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
  };
  int below = 0;
  for(int p = 0; p < N; p++)
  {
    dvec3  w       = dvec3(-std::sqrt(1.0 - NoV * NoV), 0.0, -NoV);
    double h       = 2.0;
    bool   outside = true;
    for(int order = 0; order < 1000; order++)
    {
      const double x = u(rng);
      h              = outside ? nextHeight(w, h, x) : -nextHeight(-w, -h, x);
      if(std::isinf(h))
      {
        below += outside ? 0 : 1;
        break;
      }
      const dvec3  wi = -w;
      const dvec3  m  = outside ? visibleNormal(wi) : -visibleNormal(-wi);
      const double e  = outside ? eta : 1.0 / eta;
      const double c  = dot(wi, m);
      if(u(rng) < fresnel(c, e))
        w = -wi + 2.0 * c * m;
      else
      {
        const double ct = -std::sqrt(std::max(1.0 - (1.0 - c * c) / (e * e), 0.0));
        w               = normalize(m * (c / e + ct) - wi / e);
        outside         = !outside;
      }
    }
  }
  return double(below) / N;
}

PbrMaterial roughGlass(float eta, float roughness)
{
  PbrMaterial glass  = defaultPbrMaterial(vec3(1.0f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1));
  glass.transmission = 1.0f;
  glass.thickness    = 0.01f;
  glass.ior2         = eta;
  return glass;
}

}  // namespace

// A rough boundary lets in what a random walk on its microsurface lets in, every order of scattering:
// the refraction lobe alone lost what its refracted ray's masking blocked, 7% at index 1.4, roughness
// 0.5 and 70 degrees, 20% at roughness 0.7. Between the table's points: index, roughness and angle.
TEST(BsdfLobes, ARoughBoundaryLetsInWhatItsMicrosurfaceDoes)
{
  struct At
  {
    float eta, roughness, NoV;
  };
  for(const At at : {At{1.4f, 0.5f, 0.342f}, At{1.4f, 0.45f, 0.17f}, At{1.33f, 0.3f, 0.6f}, At{1.5f, 0.7f, 0.25f},
                     At{1.45f, 0.6f, 0.9f}, At{2.0f, 0.8f, 0.5f}})
  {
    const PbrMaterial glass = roughGlass(at.eta, at.roughness);
    const double      walk  = walkTransmission(at.eta, at.roughness * at.roughness, at.NoV, 200000, 29);
    const float       lobe  = carriedAt(glass, at.NoV, 400000).transmitted.x;
    EXPECT_NEAR(lobe, walk, 0.01 * walk + 0.002) << "eta " << at.eta << " roughness " << at.roughness << " NoV " << at.NoV;
  }
}

// A specular weight below 1 (KHR_materials_specular; DAZ's skin puts 0.2-0.6 of its reflection in this
// lobe) reflects less, and lets in all the rest: the lobes add up to the light that arrived, never more.
// The share alone, the gain of a boundary with all its Fresnel, made Iray skin 3.4% brighter than the
// light at a cosine of 0.15; with no reflection, the masking alone is lost and all of it comes back.
TEST(BsdfLobes, ALowerSpecularLetsInAllItDoesNotReflect)
{
  for(float specular : {0.0f, 0.25f, 0.6f, 1.0f})
    for(float roughness : {0.5f, 0.8f})
      for(float NoV : {1.0f, 0.4f, 0.15f})
      {
        PbrMaterial glass = roughGlass(1.4f, roughness);
        glass.specular    = specular;
        const Carried c   = carriedAt(glass, NoV, 400000);
        EXPECT_LE(c.all.x, 1.005f) << specular << ", " << roughness << ", " << NoV;
        EXPECT_GE(c.all.x, specular == 0.0f ? 0.995f : 0.95f) << specular << ", " << roughness << ", " << NoV;
      }
}

// Only a medium's boundary met from outside takes the walk's light back: the table knows neither total
// internal reflection (the denser side), a thin wall's mirrored reflection, nor Gaussian slopes. A
// smooth boundary gains nothing, a rough one the more the more slanted the view.
TEST(BsdfLobes, OnlyAnEntryThroughARoughBoundaryGainsItsMaskedLight)
{
  PbrMaterial glass = roughGlass(1.4f, 0.5f);
  EXPECT_TRUE(bsdfRegainsMaskedTransmission(glass));
  PbrMaterial exit = glass;
  exit.ior1        = 1.4f;
  exit.ior2        = 1.0f;
  EXPECT_FALSE(bsdfRegainsMaskedTransmission(exit));
  PbrMaterial thin = glass;
  thin.thickness   = 0.0f;
  EXPECT_FALSE(bsdfRegainsMaskedTransmission(thin));
  EXPECT_FALSE(bsdfRegainsMaskedTransmission(water(0.15f)));

  for(float eta : {1.33f, 1.5f, 2.4f})
    for(float NoV : {1.0f, 0.5f, 0.1f})
      for(float specular : {0.0f, 0.3f, 1.0f})
        EXPECT_NEAR(ggxTransmissionGain(eta, 0.0f, NoV, vec3(specular)).x, 1.0f, 1e-5f) << eta << ", " << NoV << ", " << specular;
  float last = 0.0f;
  for(float deg : {0.0f, 45.0f, 70.0f, 80.0f})
  {
    const float gain = ggxTransmissionGain(1.4f, 0.25f, std::cos(deg * 3.14159265f / 180.0f), vec3(1.0f)).x;
    EXPECT_GE(gain, last) << deg;
    last = gain;
  }
  EXPECT_NEAR(ggxTransmissionGain(1.4f, 0.25f, std::cos(70.0f * 3.14159265f / 180.0f), vec3(1.0f)).x, 1.07f, 0.01f);
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

// A thin pane reflects at both faces: 2r / (1 + r) of one face's r per polarisation, 7.7% head-on for
// glass where one face gives 4%, more at every angle, all of it at grazing.
TEST(BsdfLobes, APaneReflectsAtBothFaces)
{
  EXPECT_NEAR(plate_fresnel(1.5f, 1.0f), 2.0f * 0.04f / 1.04f, 1e-5f);
  for(float kh = 0.02f; kh <= 1.0f; kh += 0.07f)
  {
    const float one = ior_fresnel(1.5f, kh), two = plate_fresnel(1.5f, kh);
    EXPECT_GT(two, one) << kh;
    EXPECT_LE(two, 2.0f * one) << kh;
  }
  EXPECT_GT(plate_fresnel(1.5f, 1e-4f), 0.99f);
  // a pane of a medium's wall is one interface
  PbrMaterial wall = caseNamed("pane");
  EXPECT_TRUE(bsdfIsPlate(wall));
  wall.thickness = 0.01f;
  EXPECT_FALSE(bsdfIsPlate(wall));
}

// A thin pane splits each microfacet's light by the pane's Fresnel: about twice one face's reflection
// head-on, more at every angle, and what it reflects and transmits together is what one face's film
// keeps: the pane moves light from transmission to reflection, it makes or loses none.
TEST(BsdfLobes, APaneReflectsTwiceAndKeepsItsEnergy)
{
  for(float roughness : {0.05f, 0.3f})
  {
    PbrMaterial film  = defaultPbrMaterial(vec3(1.0f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1));
    film.transmission = 1.0f;
    PbrMaterial pane  = film;
    pane.plate        = true;
    for(float NoV : {1.0f, 0.5f, 0.2f})
    {
      const Carried one = carriedAt(film, NoV), two = carriedAt(pane, NoV);
      EXPECT_GT(two.reflected.x, (NoV == 1.0f ? 1.7f : 1.05f) * one.reflected.x) << roughness << ", " << NoV;
      EXPECT_LE(two.reflected.x + two.transmitted.x, 1.01f) << roughness << ", " << NoV;
      EXPECT_NEAR(two.reflected.x + two.transmitted.x, one.reflected.x + one.transmitted.x, 0.01f) << roughness << ", " << NoV;
    }
    if(roughness == 0.05f)
      EXPECT_NEAR(carriedAt(pane, 1.0f).reflected.x, 0.0769f, 0.004f);
  }
}

// A pane's colour is a crossing's face-on; at a slant its path through the glass lengthens as 1 / cos
// of the refracted angle, so Beer-Lambert raises the colour to that power (bsdfThinTintAtSlant). 7TEST's
// car glass, 0.77 / 0.84 / 0.83, read 0.73 / 0.81 / 0.80 at 60 degrees and 0.70 toward grazing.
TEST(BsdfLobes, ATintedPaneDeepensAtASlant)
{
  const vec3  glass(0.77f, 0.84f, 0.83f);
  const float cos60 = 0.5f;
  EXPECT_NEAR(bsdfThinTintAtSlant(glass, 1.5f, 1.0f).x, 0.77f, 1e-6f);
  const float cosT60 = std::sqrt(1.0f - 0.75f / 2.25f);
  EXPECT_NEAR(bsdfThinTintAtSlant(glass, 1.5f, cos60).x, std::pow(0.77f, 1.0f / cosT60), 1e-5f);
  EXPECT_NEAR(bsdfThinTintAtSlant(glass, 1.5f, cos60).x, 0.726f, 1e-3f);
  EXPECT_NEAR(bsdfThinTintAtSlant(glass, 1.5f, 0.0f).x, 0.704f, 1e-3f);
  // either side of the pane alike; a colourless pane stays so; at an index of 1 the path stops growing
  // at the floor, short of black
  EXPECT_NEAR(bsdfThinTintAtSlant(glass, 1.0f / 1.5f, cos60).y, bsdfThinTintAtSlant(glass, 1.5f, cos60).y, 1e-6f);
  for(float c : {1.0f, 0.5f, 0.0f})
    EXPECT_EQ(bsdfThinTintAtSlant(vec3(1.0f), 1.5f, c).z, 1.0f);
  EXPECT_NEAR(bsdfThinTintAtSlant(glass, 1.0f, 0.0f).x, std::pow(0.77f, 1.0f / THIN_TINT_COS_FLOOR), 1e-6f);

  // The pane's transmission, sampled and evaluated alike: every sample carries the colour, so the
  // channels' ratio is the slant's (to the float sums' rounding, sampled; exactly, evaluated). A solid glass's colour is a crossing's whatever the angle
  // (its medium absorbs on the way), and so is a thin scatter surface's (its colour stands for a medium).
  for(float roughness : {0.05f, 0.3f})
  {
    PbrMaterial pane  = defaultPbrMaterial(glass, 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1));
    pane.transmission = 1.0f;
    pane.plate        = true;
    PbrMaterial solid = pane;
    solid.plate       = false;
    solid.thickness   = 0.01f;
    PbrMaterial thinScatter      = pane;
    thinScatter.transmissionTint = vec3(1.0f, 0.5f, 0.5f);
    for(float NoV : {1.0f, cos60, 0.2f})
    {
      const float  cosT  = std::sqrt(1.0f - (1.0f - NoV * NoV) / 2.25f);
      const float  slant = std::pow(0.84f / 0.77f, 1.0f / cosT);
      const vec3   t     = carriedAt(pane, NoV, 20000).transmitted;
      ASSERT_GT(t.x, 0.0f) << roughness << ", " << NoV;
      EXPECT_NEAR(t.y / t.x, slant, 1e-3f) << roughness << ", " << NoV;
      const vec3 s = carriedAt(solid, NoV, 20000).transmitted;
      ASSERT_GT(s.x, 0.0f) << roughness << ", " << NoV;
      EXPECT_NEAR(s.y / s.x, 0.84f / 0.77f, 1e-3f) << roughness << ", " << NoV;
      const vec3 ts = carriedAt(thinScatter, NoV, 20000).transmitted;
      ASSERT_GT(ts.x, 0.0f) << roughness << ", " << NoV;
      EXPECT_NEAR(ts.y / ts.x, 0.84f * 0.5f / 0.77f, 1e-3f) << roughness << ", " << NoV;

      // the light sample's side: straight through the pane
      BsdfEvaluateData e;
      e.k1 = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      e.k2 = -e.k1;
      e.xi = vec3(0.5f);
      bsdfEvaluate(e, pane);
      ASSERT_GT(e.bsdf_glossy.x, 0.0f) << roughness << ", " << NoV;
      EXPECT_NEAR(e.bsdf_glossy.y / e.bsdf_glossy.x, slant, 1e-4f) << roughness << ", " << NoV;
    }
  }
}

// What a pane's lobe reflects is its GGX lobe weighed by each microfacet's pane Fresnel; the albedo
// fit (ggxPlateAlbedo) is how often the lobe is picked, and what a base under it is charged. It comes
// within 12% of the integral, low at middling views on rough panes; the one-face fit
// (ggxDielectricAlbedo) is within 10% over the same grid (measured 2026-11-06). What a clear pane
// reflects and transmits is exact either way (APaneReflectsTwiceAndKeepsItsEnergy).
TEST(BsdfLobes, ThePaneAlbedoFitIsTheLobesIntegral)
{
  std::mt19937                          rng(37);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float roughness : {0.05f, 0.2f, 0.4f, 0.7f, 1.0f})
    for(float NoV : {0.05f, 0.15f, 0.5f, 1.0f})
    {
      PbrMaterial mat = defaultPbrMaterial(vec3(1.0f), 0.0f, roughness, vec3(0, 0, 1), vec3(0, 0, 1));
      const vec3  k1  = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      const int   N   = 100000;
      double      microfacet = 0.0;
      for(int i = 0; i < N; i++)
      {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = vec3(u(rng), u(rng), u(rng));
        brdf_ggx_smith_sample(s, mat, LOBE_SPECULAR_REFLECTION, vec3(1.0f), BSDF_FRESNEL_PLATE, 1.5f);
        if(s.event_type != BSDF_EVENT_ABSORB)
          microfacet += s.bsdf_over_pdf.x / N;
      }
      EXPECT_NEAR(ggxPlateAlbedo(1.5f, roughness * roughness, NoV), microfacet, 0.13 * microfacet + 2e-3) << roughness << ", " << NoV;
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

// Eto and Tokuyoshi's bound takes out of the visible normals only those that reflect below the surface
// (hvd_ggx_sample_vndf_bounded): a rough lobe loses fewer of its samples there (alpha 1 none, where
// the visible-normal sampler lost 9-41%), the directions left still follow the pdf it reports, an
// anisotropic lobe's too, and the light that comes back is the same
// (TheGgxAlbedoFitIsTheLobesIntegral, TheBoundedSamplerKeepsTheMeanWithLessVariance).
TEST(BsdfLobes, TheBoundedSamplerLosesFewerReflectionsBelowTheSurface)
{
  std::mt19937                          rng(23);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float alpha : {0.1f, 0.3f, 0.6f, 1.0f})
    for(float NoV : {0.05f, 0.2f, 0.5f})
    {
      const vec3 k = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      const int  N = 200000;
      int        lostVndf = 0, lostBounded = 0;
      for(int i = 0; i < N; i++)
      {
        const vec2 xi(u(rng), u(rng));
        const vec3 h0 = hvd_ggx_sample_vndf(k, vec2(alpha), xi);
        const vec3 h1 = hvd_ggx_sample_vndf_bounded(k, vec2(alpha), xi);
        lostVndf += (2.0f * dot(k, h0) * h0 - k).z <= 0.0f ? 1 : 0;
        lostBounded += (h1.z <= 0.0f || (2.0f * dot(k, h1) * h1 - k).z <= 0.0f) ? 1 : 0;
      }
      EXPECT_LT(lostBounded, lostVndf) << "alpha " << alpha << " NoV " << NoV;
      if(alpha >= 0.3f)  // a third of the loss at least, where there is much to lose
        EXPECT_LT(double(lostBounded), 0.67 * double(lostVndf)) << "alpha " << alpha << " NoV " << NoV;
    }

  PbrMaterial rough = base(1.0f);
  rough.metallic    = 1.0f;
  PbrMaterial aniso = rough;
  aniso.roughness   = vec2(0.2f, 0.8f);
  PbrMaterial mid   = rough;
  mid.roughness     = vec2(0.36f);
  for(const Case& c : {Case{"rough metal", rough}, Case{"anisotropic metal", aniso}, Case{"mid metal", mid}})
    for(const vec3& k1 : {normalize(vec3(0.995f, 0.0f, 0.1f)), normalize(vec3(0.6f, 0.6f, 0.2f))})
    {
      const SamplingFit fit = samplingFit(c.mat, k1);
      EXPECT_LT(fit.sigmas, 5.0) << fit.sigmas << " sigmas, " << c.name << " view " << k1.x << "," << k1.y << "," << k1.z;
      EXPECT_NEAR(fit.missing, fit.absorbed, 0.01) << c.name << " view " << k1.x << "," << k1.y << "," << k1.z;
    }
}

// The bound changes how the light is estimated, not what it is: a white rough lobe's albedo through the
// bounded sampler (brdf_ggx_smith_sample) has the mean the visible-normal sampler had (G2/G1 a sample,
// as the lobe weighed them before the bound), within the two estimates' noise, with less variance at
// every view.
TEST(BsdfLobes, TheBoundedSamplerKeepsTheMeanWithLessVariance)
{
  std::mt19937                          rng(29);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float alpha : {0.3f, 0.6f, 1.0f})
    for(float NoV : {0.1f, 0.3f, 0.7f})
    {
      PbrMaterial mat = base(std::sqrt(alpha));
      const vec3  k1  = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      const int   N   = 400000;
      double      sumB = 0.0, sumB2 = 0.0, sumV = 0.0, sumV2 = 0.0;
      for(int i = 0; i < N; i++)
      {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = vec3(u(rng), u(rng), u(rng));
        brdf_ggx_smith_sample(s, mat, LOBE_SPECULAR_REFLECTION, vec3(1.0f), BSDF_FRESNEL_SCHLICK, 1.0f);
        const double b = s.event_type != BSDF_EVENT_ABSORB ? double(s.bsdf_over_pdf.x) : 0.0;
        sumB += b;
        sumB2 += b * b;

        const vec3 h  = hvd_ggx_sample_vndf(k1, mat.roughness, vec2(s.xi));
        const vec3 k2 = 2.0f * dot(k1, h) * h - k1;
        double     v  = 0.0;
        if(h.z > 0.0f && k2.z > 0.0f)
        {
          float       G1, G2;
          const float G12 = ggx_smith_shadow_mask_correlated(G1, G2, k1, k2, mat.roughness);
          v               = G12 > 0.0f ? double(G12 / G1) : 0.0;
        }
        sumV += v;
        sumV2 += v * v;
      }
      const double meanB = sumB / N, varB = sumB2 / N - meanB * meanB;
      const double meanV = sumV / N, varV = sumV2 / N - meanV * meanV;
      const double sigma = std::sqrt((varB + varV) / N);
      EXPECT_NEAR(meanB, meanV, 4.0 * sigma) << "alpha " << alpha << " NoV " << NoV;
      // a fifth to two thirds of it (utils/bounded_vndf.py, their numpy twin: x0.18 at alpha 1, NoV 0.7)
      EXPECT_LT(varB, 0.8 * varV) << "alpha " << alpha << " NoV " << NoV;
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

// Skin under a coat, as the walk has it and as the thin preview does: the walk's transmission isn't clear
// glass, so both darken alike, on the base's diffuse, on the way into the walk by either entry, and on emission
TEST(BsdfLobes, ACoatDarkensAWalkAsItsThinPreview)
{
  const vec3 skin(0.6f, 0.35f, 0.25f);
  PbrMaterial preview        = defaultPbrMaterial(skin, 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  preview.clearcoat          = 1.0f;
  preview.clearcoatRoughness = 0.1f;
  PbrMaterial walk           = preview;
  walk.transmission          = 0.95f;
  walk.thickness             = 1.0f;
  walk.scatterCoefficient    = vec3(500.0f, 200.0f, 100.0f);
  walk.transmissionTint      = 1.0f / skin;

  const vec3 darkening = bsdfCoatDarkening(1.0f, skin, 0.0f, bsdfCoatBaseSpecular(preview.specular, preview.ior2), 0.0f, preview.roughness.x);
  EXPECT_LT(darkening.x, 0.8f);
  vec3 tintWalk = walk.baseColor, tintPreview = preview.baseColor;
  computeLobeWeights(walk, 1.0f, tintWalk);
  computeLobeWeights(preview, 1.0f, tintPreview);
  PbrMaterial uncoated = walk;
  uncoated.clearcoat   = 0.0f;
  vec3 tintUncoated    = uncoated.baseColor;
  computeLobeWeights(uncoated, 1.0f, tintUncoated);
  for(int c = 0; c < 3; c++)
  {
    EXPECT_NEAR(tintWalk[c], tintPreview[c], 1e-6f);
    EXPECT_NEAR(tintPreview[c], skin[c] * darkening[c], 1e-5f);
    // the refraction's way in carries the factor and not the skin's colour, which the medium has
    EXPECT_NEAR(bsdfLobeFresnel(walk, LOBE_SPECULAR_TRANSMISSION, 1.0f, tintWalk).tint[c]
                    / bsdfLobeFresnel(uncoated, LOBE_SPECULAR_TRANSMISSION, 1.0f, tintUncoated).tint[c],
                darkening[c], 1e-4f);
  }

  // the diffuse entry takes the transmission's share to a white lobe that doesn't read the tint
  PbrMaterial entry                    = walk;
  entry.transmission                   = 0.0f;
  entry.diffuseTransmissionFactor      = 1.0f;
  entry.diffuseTransmissionColorFactor = vec3(1.0f);
  PbrMaterial bare                     = entry;
  bare.clearcoat                       = 0.0f;
  BsdfEvaluateData coated{}, plain{};
  coated.k1 = plain.k1 = normalize(vec3(0.3f, 0.1f, 0.9f));
  coated.k2 = plain.k2 = normalize(vec3(-0.25f, -0.1f, -0.9f));
  bsdfEvaluateLobe(coated, entry, LOBE_DIFFUSE_TRANSMISSION, entry.baseColor);
  bsdfEvaluateLobe(plain, bare, LOBE_DIFFUSE_TRANSMISSION, bare.baseColor);
  ASSERT_GT(plain.bsdf_diffuse.x, 0.0f);
  EXPECT_NEAR(coated.bsdf_diffuse.x / plain.bsdf_diffuse.x, darkening.x, 1e-4f);

  EXPECT_FLOAT_EQ(bsdfCoatClearTransmission(walk), 0.0f);
  const float skinSpecular = bsdfCoatBaseSpecular(preview.specular, preview.ior2);
  EXPECT_NEAR(bsdfCoatEmission(1.0f, 0.1f, skin, 0.0f, skinSpecular, bsdfCoatClearTransmission(walk), preview.roughness.x, 1.0f).x,
              bsdfCoatEmission(1.0f, 0.1f, skin, 0.0f, skinSpecular, bsdfCoatClearTransmission(preview), preview.roughness.x, 1.0f).x, 1e-6f);

  // and clear glass keeps sending its light on
  PbrMaterial glass        = walk;
  glass.scatterCoefficient = vec3(0.0f);
  EXPECT_FLOAT_EQ(bsdfCoatClearTransmission(glass), 0.95f);
}

// The metal lobe carries a coat's darkening too, on what it reflects: its Fresnel's f0 stays the base colour,
// so a grazing view and the multiple-scattering term come out as they do uncoated, times the factor. White
// metal keeps its energy, in evaluation and in sampling alike.
TEST(BsdfLobes, ACoatDarkensADarkMetal)
{
  const auto metal = [](const vec3& colour, float coat) {
    PbrMaterial mat        = defaultPbrMaterial(colour, 1.0f, 0.4f, vec3(0, 0, 1), vec3(0, 0, 1));
    mat.Nc                 = mat.N;
    mat.clearcoat          = coat;
    mat.clearcoatRoughness = 0.1f;
    return mat;
  };
  const auto ratio = [&](const vec3& colour, const vec3& view) {
    const vec3        light  = normalize(vec3(-view.x, -view.y, view.z));
    const PbrMaterial coated = metal(colour, 1.0f), bare = metal(colour, 0.0f);
    BsdfEvaluateData  a{}, b{};
    a.k1 = b.k1 = view;
    a.k2 = b.k2 = light;
    bsdfEvaluateLobe(a, coated, LOBE_METAL_REFLECTION, coated.baseColor);
    bsdfEvaluateLobe(b, bare, LOBE_METAL_REFLECTION, bare.baseColor);
    EXPECT_GT(b.bsdf_glossy.x, 0.0f);
    return a.bsdf_glossy.x / b.bsdf_glossy.x;
  };
  const vec3  dark(0.3f, 0.2f, 0.1f);
  const float factor = bsdfCoatDarkening(1.0f, dark, 1.0f, bsdfCoatBaseSpecular(metal(dark, 1.0f).specular, 1.5f), 0.0f, metal(dark, 1.0f).roughness.x).x;
  EXPECT_LT(factor, 0.9f);
  EXPECT_NEAR(ratio(dark, normalize(vec3(0.3f, 0.1f, 0.9f))), factor, 1e-4f);
  EXPECT_NEAR(ratio(dark, normalize(vec3(0.95f, 0.0f, 0.2f))), factor, 1e-4f);
  EXPECT_NEAR(ratio(vec3(1.0f), normalize(vec3(0.3f, 0.1f, 0.9f))), 1.0f, 1e-4f);

  const PbrMaterial coated = metal(dark, 1.0f), bare = metal(dark, 0.0f);
  BsdfSampleData    a{}, b{};
  a.k1 = b.k1 = normalize(vec3(0.3f, 0.1f, 0.9f));
  a.xi = b.xi = vec4(0.3f, 0.6f, 0.5f, 0.5f);
  bsdfSampleLobe(a, coated, LOBE_METAL_REFLECTION, coated.baseColor);
  bsdfSampleLobe(b, bare, LOBE_METAL_REFLECTION, bare.baseColor);
  ASSERT_GT(b.bsdf_over_pdf.x, 0.0f);
  EXPECT_NEAR(a.bsdf_over_pdf.x / b.bsdf_over_pdf.x, factor, 1e-4f);
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

namespace {

// One microfacet of a rough dielectric interface of GGX alpha `alpha`, seen from its near side in the
// frame whose +z points to that side, drawn by its visible normals for the direction `k1`: both ways on by
// its exact Fresnel. `eta` is the far side's index over the near side's. Each way's weight is its Fresnel
// share times Smith's masking of the way out (the visible normals carry the way in's): height-correlated
// for the reflection, uncorrelated for the refraction, whose correlated form is a Beta function. A way
// that single scattering loses (a reflection under the surface, a refraction back over it) weighs 0.
struct Microfacet
{
  vec3  reflected{0.0f}, refracted{0.0f};
  float fresnel = 1.0f, reflectWeight = 0.0f, refractWeight = 0.0f;
};

Microfacet coatMicrofacet(const vec3& k1, float alpha, float eta, std::mt19937& rng)
{
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  Microfacet                            m;
  const vec3                            h  = hvd_ggx_sample_vndf(k1, vec2(alpha), vec2(u(rng), u(rng)));
  const float                           kh = dot(k1, h);
  if(h.z <= 0.0f || kh <= 0.0f)
    return m;
  m.fresnel   = ior_fresnel(eta, kh);
  m.reflected = (2.0f * kh) * h - k1;
  if(m.reflected.z > 0.0f)
  {
    float       G1, G2;
    const float G = ggx_smith_shadow_mask_correlated(G1, G2, k1, m.reflected, vec2(alpha));
    m.reflectWeight = m.fresnel * G / G1;
  }
  bool tir    = false;
  m.refracted = refract(k1, h, 1.0f / eta, kh, tir);
  if(!tir && m.refracted.z < 0.0f)
    m.refractWeight = (1.0f - m.fresnel) * smith_shadow_or_mask(m.refracted, vec2(alpha));
  return m;
}

// The ground truth for a coat: a coat of index 1.5 and GGX alpha `coatAlpha` over `base`, an uncoated
// material, stacked exactly. Guo et al. 2018's position-free walk with nothing between the layers: the
// coat reflects or lets in, then the path goes between the base and the coat's inside until it leaves
// through the coat or is absorbed. Its directional albedo at view cosine `NoV`, over N paths; the coat's
// reflection on the way in is counted for every path, not drawn. The radiance's 1/eta^2 on the way in and
// eta^2 on the way out cancel, so neither is counted. A model too: Smith interfaces, no light between
// microfacets (a rough coat loses light to that: a coat of roughness 0.4 over white keeps 0.82-0.86 of
// it, so the tables stop at 0.2; DAZ's Top Coat Roughness defaults to 0.1), no interference.
struct Stack
{
  vec3 top{0.0f};    // the coat's own reflection
  vec3 under{0.0f};  // what comes back out through it
  vec3 all() const { return top + under; }
};

Stack coatedReference(const PbrMaterial& base, float coatAlpha, float NoV, int N)
{
  std::mt19937                          rng(43);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const vec3                            view(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
  dvec3                                 top(0.0), under(0.0);
  for(int i = 0; i < N; i++)
  {
    const Microfacet entry = coatMicrofacet(view, coatAlpha, 1.5f, rng);
    top += dvec3(entry.reflectWeight);
    if(entry.refractWeight <= 0.0f)
      continue;
    vec3 throughput(entry.refractWeight), down = entry.refracted;
    for(int bounce = 0; bounce < 256; bounce++)
    {
      BsdfSampleData s;
      s.k1 = -down;
      s.xi = vec3(u(rng), u(rng), u(rng));
      bsdfSample(s, base);
      if(s.event_type == BSDF_EVENT_ABSORB || s.k2.z <= 0.0f)
        break;
      throughput *= s.bsdf_over_pdf;
      if(bounce >= 3)
      {
        const float p = std::min(1.0f, std::max({throughput.x, throughput.y, throughput.z}));
        if(u(rng) >= p)
          break;
        throughput /= p;
      }
      // the coat from inside, in the frame mirrored so that its near side is +z: one way, by its Fresnel
      const Microfacet m = coatMicrofacet(vec3(-s.k2.x, -s.k2.y, s.k2.z), coatAlpha, 1.0f / 1.5f, rng);
      if(u(rng) >= m.fresnel)
      {
        under += dvec3(throughput * (m.refractWeight / (1.0f - m.fresnel)));
        break;
      }
      if(m.reflectWeight <= 0.0f)
        break;
      throughput *= m.reflectWeight / m.fresnel;
      down = vec3(m.reflected.x, m.reflected.y, -m.reflected.z);
    }
  }
  return {vec3(top / double(N)), vec3(under / double(N))};
}

PbrMaterial withCoat(PbrMaterial mat, float roughness)
{
  mat.Nc                 = mat.N;
  mat.clearcoat          = 1.0f;
  mat.clearcoatRoughness = roughness;
  return mat;
}

// One row of a coat's table: the exact stack's albedo, our closure's, and ours over it less 1, per channel
void printCoatRow(const char* name, const PbrMaterial& base, float coatRoughness, float NoV, const vec3& ref, const vec3& ours)
{
  const vec3 error = ours / ref - vec3(1.0f);
  std::printf("  %-9s colour %.2f %.2f %.2f  coat %.2f  NoV %.2f | exact %.4f %.4f %.4f | ours %.4f %.4f %.4f | %+5.1f%% %+5.1f%% %+5.1f%%\n",
              name, base.baseColor.x, base.baseColor.y, base.baseColor.z, coatRoughness, NoV, ref.x, ref.y, ref.z, ours.x,
              ours.y, ours.z, 100.0f * error.x, 100.0f * error.y, 100.0f * error.z);
}

const float kCoatViews[] = {1.0f, 0.7f, 0.4f, 0.15f};

}  // namespace

// The reference itself: a smooth coat over white gives back all the light at every view, reflects the
// dielectric's Fresnel, and over a Lambert base is the closed form a smooth coat has. The light lets in,
// T(mu) = 1 - F(mu), meets the base of albedo E, and of what the base sends up the coat's inside reflects
// K = 1 - (1 - F_avg) / n^2 back down (F_avg the Fresnel's cosine-weighted mean seen from outside, so K
// 0.596 at n 1.5): F(mu) + T(mu) E (1 - K) / (1 - E K).
TEST(BsdfLobes, TheCoatReferenceKeepsEnergyAndIsTheSmoothCoatsClosedForm)
{
  const float  alpha = 0.05f * 0.05f;
  const float  K     = 1.0f - (1.0f - 0.0918f) / (1.5f * 1.5f);
  PbrMaterial  base  = defaultPbrMaterial(vec3(1.0f, 0.3f, 0.05f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
  base.specular      = 0.0f;
  for(float NoV : kCoatViews)
  {
    const Stack ref = coatedReference(base, alpha, NoV, 100000);
    const float F   = ior_fresnel(1.5f, NoV);
    EXPECT_NEAR(ref.top.x, F, 0.002f) << NoV;
    EXPECT_NEAR(ref.all().x, 1.0f, 0.003f) << NoV;
    for(int c = 1; c < 3; c++)
    {
      const float E = base.baseColor[c];
      EXPECT_NEAR(ref.all()[c], F + (1.0f - F) * E * (1.0f - K) / (1.0f - E * K), 0.003f) << NoV << ", channel " << c;
    }
  }
}

// A coat over a diffuse base darkens it by physics: of a base of 0.3, 0.49 of what the coat lets in comes
// back out under a smooth coat. Our closure is OpenPBR's formula with K the diffuse 0.596 and E the base's
// colour, the smooth coat's closed form: within 1.5% of the exact stack at every view and both coat
// roughnesses DAZ uses most. (E had taken a base specular of 9% that a Lambert base doesn't have: 0-7.5%
// too bright.)
TEST(BsdfLobes, ACoatDarkensADiffuseBaseAsTheLayersStackedExactly)
{
  std::printf("A coat over a Lambert base: the exact stack's albedo against our closure's\n");
  for(const vec3 colour : {vec3(0.03f, 0.15f, 0.3f), vec3(0.5f, 0.7f, 0.9f)})
  {
    PbrMaterial base = defaultPbrMaterial(colour, 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
    base.specular    = 0.0f;
    for(float coatRoughness : {0.05f, 0.2f})
      for(float NoV : kCoatViews)
      {
        const vec3 ref  = coatedReference(base, coatRoughness * coatRoughness, NoV, 50000).all();
        const vec3 ours = carriedAt(withCoat(base, coatRoughness), NoV, 50000).all;
        printCoatRow("lambert", base, coatRoughness, NoV, ref, ours);
        for(int c = 0; c < 3; c++)
        {
          EXPECT_GT(ours[c], 0.985f * ref[c]) << colour[c] << ", coat " << coatRoughness << ", NoV " << NoV;
          EXPECT_LT(ours[c], 1.02f * ref[c]) << colour[c] << ", coat " << coatRoughness << ", NoV " << NoV;
        }
      }
  }
}

// Over a base with its own reflection, a plastic's specular or a metal, our closure is further from the exact
// stack. The coat bends the light in, so the base sees the view within 42 degrees of its normal and its
// Fresnel never rises toward grazing, and what the base sends past 42 degrees stays inside by total internal
// reflection. Ours keeps the base's own rise under the coat's: from a view cosine of 0.4 to grazing, a
// coated dark plastic is up to 12% too bright and a smooth metal 33%. A rough metal's light meets the coat's
// inside nearly as a diffuse base's does, where ours takes K 0.45: up to 22% too bright at every view.
// Head-on, a smooth metal's light leaves straight through the coat, which ours darkens as if it were spread
// over the hemisphere (up to 11% too dark), and so is a dark plastic's (10%). A plastic's own specular is
// darkened as its diffuse is (it had not been: 8-15% too bright head-on, 37% at a view cosine of 0.4).
// These bounds hold today's gaps, a plastic's and a metal's apart; closing them tightens them.
TEST(BsdfLobes, ACoatOverASpecularBaseAgainstTheExactStack)
{
  std::printf("A coat over a plastic or a metal: the exact stack's albedo against our closure's\n");
  for(const vec3 colour : {vec3(0.03f, 0.15f, 0.3f), vec3(0.5f, 0.7f, 0.9f)})
  {
    const std::array<std::pair<const char*, PbrMaterial>, 3> bases{{
        {"plastic", defaultPbrMaterial(colour, 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1))},
        {"metal 0.3", defaultPbrMaterial(colour, 1.0f, 0.3f, vec3(0, 0, 1), vec3(0, 0, 1))},
        {"metal 0.7", defaultPbrMaterial(colour, 1.0f, 0.7f, vec3(0, 0, 1), vec3(0, 0, 1))},
    }};
    for(const auto& [name, base] : bases)
      for(float coatRoughness : {0.05f, 0.2f})
        for(float NoV : kCoatViews)
        {
          const vec3 ref  = coatedReference(base, coatRoughness * coatRoughness, NoV, 50000).all();
          const vec3 ours = carriedAt(withCoat(base, coatRoughness), NoV, 50000).all;
          printCoatRow(name, base, coatRoughness, NoV, ref, ours);
          const bool  plastic = base.metallic == 0.0f;
          const float least = plastic ? 0.88f : 0.85f, most = plastic ? 1.15f : 1.45f;
          for(int c = 0; c < 3; c++)
          {
            EXPECT_GT(ours[c], least * ref[c]) << name << ", " << colour[c] << ", coat " << coatRoughness << ", NoV " << NoV;
            EXPECT_LT(ours[c], most * ref[c]) << name << ", " << colour[c] << ", coat " << coatRoughness << ", NoV " << NoV;
          }
        }
  }
}

// Emission lies under a coat: out through what it doesn't reflect, and what it doesn't give back to the base.
// Not with no coat.
TEST(BsdfLobes, EmissionComesOutThroughACoat)
{
  const vec3  base(0.3f);
  const float spec = bsdfCoatBaseSpecular(1.0f, 1.5f);
  EXPECT_EQ(bsdfCoatEmission(0.0f, 0.1f, base, 0.0f, spec, 0.0f, 0.25f, 1.0f), vec3(1.0f));
  const vec3 straight = bsdfCoatEmission(1.0f, 0.1f, base, 0.0f, spec, 0.0f, 0.25f, 1.0f);
  const vec3 grazing  = bsdfCoatEmission(1.0f, 0.1f, base, 0.0f, spec, 0.0f, 0.25f, 0.2f);
  EXPECT_NEAR(straight.x, 0.96f * bsdfCoatDarkening(1.0f, base, 0.0f, spec, 0.0f, 0.25f).x, 0.005f);
  EXPECT_LT(grazing.x, straight.x);
  // half the coverage takes less
  EXPECT_GT(bsdfCoatEmission(0.5f, 0.1f, base, 0.0f, spec, 0.0f, 0.25f, 1.0f).x, straight.x);
}

// The baked sheen albedo is the sheen lobe's own integral, as its sampler carries it, over its
// roughness and the view
TEST(BsdfLobes, TheSheenAlbedoTableIsTheLobesIntegral)
{
  std::mt19937                          rng(37);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float roughness : {0.1f, 0.3f, 0.5f, 0.8f, 1.0f})
    for(float NoV : {0.1f, 0.3f, 0.6f, 1.0f})
    {
      PbrMaterial mat    = base(0.5f);
      mat.sheenColor     = vec3(1.0f);
      mat.sheenRoughness = roughness;
      const vec3 k1      = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
      const int  N       = 200000;
      double     albedo  = 0.0;
      for(int i = 0; i < N; i++)
      {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = vec3(u(rng), u(rng), u(rng));
        brdf_sheen_sample(s, mat);
        if(s.event_type != BSDF_EVENT_ABSORB)
          albedo += s.bsdf_over_pdf.x / N;
      }
      EXPECT_NEAR(sheenAlbedo(roughness, NoV), albedo, 0.01) << "roughness " << roughness << " NoV " << NoV;
    }
}

// A sheen takes from the base under it only what it reflects (KHR_materials_sheen's albedo scaling):
// white over white gives back the light it was lit with at every view, and a faint sheen leaves the base
// as it was. Charged by the view and roughness alone, a 0.05 sheen took 48% of the base at NoV 0.7 (M47).
TEST(BsdfLobes, ASheenTakesFromTheBaseOnlyWhatItReflects)
{
  for(float roughness : {0.2f, 0.5f, 0.9f})
    for(float NoV : {0.1f, 0.3f, 0.7f, 1.0f})
    {
      PbrMaterial white    = defaultPbrMaterial(vec3(1.0f), 0.0f, 0.5f, vec3(0, 0, 1), vec3(0, 0, 1));
      white.specular       = 0.0f;  // a Lambert base, which gives back all of it
      white.sheenColor     = vec3(1.0f);
      white.sheenRoughness = roughness;
      const vec3 all       = carriedAt(white, NoV, 200000).all;
      EXPECT_NEAR(all.x, 1.0f, 0.01f) << "roughness " << roughness << " NoV " << NoV;

      PbrMaterial cloth = defaultPbrMaterial(vec3(0.6f, 0.2f, 0.1f), 0.0f, 0.6f, vec3(0, 0, 1), vec3(0, 0, 1));
      PbrMaterial faint = cloth;
      faint.sheenColor     = vec3(0.05f);
      faint.sheenRoughness = roughness;
      vec3       ta = cloth.baseColor, tb = faint.baseColor;
      const auto wa = computeLobeWeights(cloth, NoV, ta);
      const auto wb = computeLobeWeights(faint, NoV, tb);
      EXPECT_GT(wb[LOBE_DIFFUSE_REFLECTION], 0.95f * wa[LOBE_DIFFUSE_REFLECTION]) << "roughness " << roughness << " NoV " << NoV;
    }
}

// A near-black base under its specular samples the specular about as often as the light it returns,
// not by its 5% weight; what each sample carries keeps the mean (SamplingCarriesWhatEvaluationIntegratesTo
// and the pdf tests run on the same material). A white one is picked by weight, as before.
TEST(BsdfLobes, ADarkGlossyBasePicksItsSpecular)
{
  std::mt19937                          rng(41);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(float NoV : {1.0f, 0.5f})
  {
    const PbrMaterial dark = defaultPbrMaterial(vec3(0.04f), 0.0f, 0.3f, vec3(0, 0, 1), vec3(0, 0, 1));
    const vec3        k1   = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
    const int         N    = 20000;
    int               glossy = 0;
    for(int i = 0; i < N; i++)
    {
      BsdfSampleData s;
      s.k1 = k1;
      s.xi = vec3(u(rng), u(rng), u(rng));
      bsdfSample(s, dark);
      glossy += s.event_type == BSDF_EVENT_GLOSSY_REFLECTION ? 1 : 0;
    }
    EXPECT_GT(float(glossy) / N, 0.4f) << "NoV " << NoV;
  }
  const PbrMaterial white = defaultPbrMaterial(vec3(1.0f), 0.0f, 0.3f, vec3(0, 0, 1), vec3(0, 0, 1));
  vec3              tint  = white.baseColor;
  const auto        s     = computeLobeWeights(white, 1.0f, tint);
  const auto        p     = bsdfLobePick(white, s, tint);
  for(int j = 0; j < LOBE_COUNT; j++)
    EXPECT_NEAR(p[j], s[j], 1e-6f) << "lobe " << j;
  // the floor: a black base is still picked a tenth as often as it weighs, and carries at most about 11x
  const PbrMaterial black = defaultPbrMaterial(vec3(0.0f), 0.0f, 0.3f, vec3(0, 0, 1), vec3(0, 0, 1));
  vec3              tb    = black.baseColor;
  const auto        sb    = computeLobeWeights(black, 1.0f, tb);
  const auto        pb    = bsdfLobePick(black, sb, tb);
  EXPECT_GT(pb[LOBE_DIFFUSE_REFLECTION], 0.08f * sb[LOBE_DIFFUSE_REFLECTION]);
  EXPECT_LE(bsdfPickShare(sb, pb, LOBE_DIFFUSE_REFLECTION), 11.0f);
}

// A bounce leaves on the side of the face its event says: a reflection on the view's side, a
// transmission across it, though a normal map tilts the lobes' normal past the face. By the shading
// normal alone, a Lambert lobe tilted 35 degrees sent (1 - cos t) / 2 = 9% of its samples through the
// face (the unit disk outside the ellipse of half-axis cos t its projection leaves above the face). Light from there still counts for next-event estimation, at a pdf of 0 (weight 1).
// The ground and water opt out (faceSided).
TEST(BsdfLobes, ABounceLeavesOnTheSideItsLobeSays)
{
  const vec3  tilted = normalize(vec3(0.7f, 0.0f, 1.0f));
  PbrMaterial bumped = defaultPbrMaterial(vec3(0.8f), 0.0f, 0.5f, tilted, vec3(0, 0, 1));
  bumped.specular    = 0.0f;  // Lambert about N
  orthonormalBasis(bumped.N, bumped.T, bumped.B);
  PbrMaterial sheet                 = bumped;  // a thin sheet that only transmits
  sheet.diffuseTransmissionFactor   = 1.0f;
  PbrMaterial loose                 = bumped;
  loose.faceSided                   = false;
  const auto sampleSides = [](const PbrMaterial& mat, int& through, int& absorbed, int& kept) {
    std::mt19937                          rng(43);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    through = absorbed = kept = 0;
    for(int i = 0; i < 100000; i++)
    {
      BsdfSampleData s;
      s.k1 = vec3(0.0f, 0.0f, 1.0f);
      s.xi = vec3(u(rng), u(rng), u(rng));
      bsdfSample(s, mat);
      if(s.event_type == BSDF_EVENT_ABSORB)
      {
        absorbed++;
        continue;
      }
      kept++;
      const bool transmits = (s.event_type & BSDF_EVENT_TRANSMISSION) != 0;
      if(transmits != (s.k2.z < 0.0f))
        through++;
    }
  };
  const float leak = 0.5f * (1.0f - dot(tilted, vec3(0, 0, 1)));
  int through, absorbed, kept;
  sampleSides(bumped, through, absorbed, kept);
  EXPECT_EQ(through, 0);
  EXPECT_NEAR(absorbed / 100000.0f, leak, 0.004f);
  sampleSides(sheet, through, absorbed, kept);
  EXPECT_EQ(through, 0);
  EXPECT_NEAR(absorbed / 100000.0f, leak, 0.004f);
  sampleSides(loose, through, absorbed, kept);
  EXPECT_NEAR(through / 100000.0f, leak, 0.004f);  // as before
  EXPECT_EQ(absorbed, 0);

  // Light below the face but above the tilted normal: the lobe's value, no BSDF sample to share it
  const vec3 below = normalize(vec3(0.9f, 0.0f, -0.1f));
  ASSERT_GT(dot(below, tilted), 0.0f);
  for(int mixture = 0; mixture < 2; mixture++)
  {
    BsdfEvaluateData e;
    e.k1 = vec3(0.0f, 0.0f, 1.0f);
    e.k2 = below;
    e.xi = vec3(0.5f);
    if(mixture)
      bsdfEvaluateAll(e, bumped);
    else
      bsdfEvaluate(e, bumped);
    EXPECT_GT(sum(e.bsdf_diffuse), 0.0f) << mixture;
    EXPECT_EQ(e.pdf, 0.0f) << mixture;
    EXPECT_TRUE(bsdfEvaluated(e)) << mixture;
    BsdfEvaluateData l = e;
    bsdfEvaluateAll(l, loose);
    EXPECT_GT(l.pdf, 0.0f) << mixture;
  }
}

// Gaussian slopes (PbrMaterial::gaussianSlopes): Beckmann's distribution of microfacets, over the slope
// plane, where a microfacet's normal is normalize(-s, 1) and D(h) dw = hvd(h) h.z^2 d^2s.
namespace {

template <class F>
double overSlopes(float alpha, const F& f)
{
  const int    n    = 600;
  const double span = 7.0 * alpha, step = 2.0 * span / n;
  double       sum  = 0.0;
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
    {
      const vec3 h = normalize(vec3(-(-span + (i + 0.5) * step), -(-span + (j + 0.5) * step), 1.0));
      sum += f(h) * step * step;
    }
  return sum;
}

}  // namespace

// Every direction is some microfacet's: D cos(theta) over the hemisphere is one
TEST(BsdfLobes, GaussianSlopesCoverTheSurfaceOnce)
{
  for(float alpha : {0.01f, 0.057f, 0.3f})
  {
    const double area = overSlopes(alpha, [&](const vec3& h) { return hvd_beckmann_eval(vec2(1.0f / alpha), h) * h.z * h.z * h.z; });
    EXPECT_NEAR(area, 1.0, 2e-3) << alpha;
  }
  // anisotropic: the same Gaussian, stretched along one axis
  const vec2 inv(1.0f / 0.05f, 1.0f / 0.12f);
  const double area = overSlopes(0.12f, [&](const vec3& h) { return hvd_beckmann_eval(inv, h) * h.z * h.z * h.z; });
  EXPECT_NEAR(area, 1.0, 2e-3);
}

// Smith's masking is what makes the microfacets a view sees as large as the surface it sees:
// G1(k) times the microfacets' area facing k, projected, is k.z (Heitz 2014). Down to a view half a
// degree off the surface, where far water is seen.
TEST(BsdfLobes, GaussianMaskingShowsTheSurfaceItsOwnSize)
{
  for(float alpha : {0.02f, 0.057f, 0.3f})
    for(float elevation : {0.5f, 1.0f, 3.0f, 10.0f, 30.0f, 60.0f})
    {
      const float  e         = radians(elevation);
      const vec3   k(std::cos(e), 0.0f, std::sin(e));
      const double projected = overSlopes(alpha, [&](const vec3& h) {
        return std::max(0.0f, dot(k, h)) * hvd_beckmann_eval(vec2(1.0f / alpha), h) * h.z * h.z;
      });
      const double seen = beckmann_shadow_or_mask(k, vec2(alpha)) * projected;
      EXPECT_NEAR(seen / k.z, 1.0, 5e-3) << "alpha " << alpha << ", " << elevation << " degrees";
    }
}

// Why the ground's water has Gaussian slopes. Seen a degree above its surface, as far water is, with the
// alpha Ripples 0.71 and Waves 0.32 average out to there: GGX's tail of steep microfacets reflected sky
// up to 18 degrees high where the water's own slopes reach 10, blurred it sideways four times as wide,
// and reflected a third less of it.
TEST(BsdfLobes, FarWaterReflectsTheSkyJustAboveTheHorizon)
{
  const float e = radians(1.0f);
  const vec3  k1(std::cos(e), 0.0f, std::sin(e));
  struct Reflection
  {
    double total = 0.0, high = 0.0, wide = 0.0;
  };
  const auto reflect = [&](const PbrMaterial& mat) {
    std::mt19937                          rng(31);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    Reflection                            r;
    const int                             N = 400000;
    for(int i = 0; i < N; i++)
    {
      BsdfSampleData s;
      s.k1 = k1;
      s.xi = vec3(u(rng), u(rng), u(rng));
      bsdfSample(s, mat);
      if(s.event_type != BSDF_EVENT_GLOSSY_REFLECTION || s.k2.z <= 0.0f)
        continue;
      const double carried = s.bsdf_over_pdf.x / N;
      r.total += carried;
      if(s.k2.z > std::sin(radians(12.0f)))
        r.high += carried;
      if(std::abs(std::atan2(s.k2.y, -s.k2.x)) > radians(1.0f))
        r.wide += carried;
    }
    return r;
  };
  const Reflection gaussian = reflect(water(0.057f)), ggx = reflect(water(0.057f, false));
  EXPECT_LT(gaussian.high / gaussian.total, 0.03) << "sky above 12 degrees";
  EXPECT_GT(ggx.high / ggx.total, 0.15);
  EXPECT_LT(gaussian.wide / gaussian.total, 0.03) << "more than a degree sideways";
  EXPECT_GT(ggx.wide / ggx.total, 0.15);
  EXPECT_GT(gaussian.total, 1.25 * ggx.total);
  EXPECT_LT(gaussian.total, 1.0);
}

// Where a path's vertex leaves the lights to its BSDF ray (shaders/smooth_vertex.h.slang): every
// weighted lobe a mirror, no diffuse, and its light sample dropped only where the BSDF ray stands in.
#include "../shaders/smooth_vertex.h.slang"

namespace {

// What smoothVertex should say, read off the BSDF's own lobe weights (computeLobeWeights) at three
// views: no weight on a diffuse lobe or the sheen, and every GGX lobe with weight at most
// SMOOTH_VERTEX_ALPHA. Retroreflection's blend is kept out whatever its roughness.
bool mirrorsByTheirWeights(const PbrMaterial& mat)
{
  for(float VdotN : {0.15f, 0.5f, 1.0f})
  {
    vec3       tint = mat.baseColor;
    const auto w    = computeLobeWeights(mat, VdotN, tint);
    if(w[LOBE_DIFFUSE_REFLECTION] > 0.0f || w[LOBE_DIFFUSE_TRANSMISSION] > 0.0f || w[LOBE_SHEEN_REFLECTION] > 0.0f)
      return false;
    const float alpha = std::max(mat.roughness.x, mat.roughness.y);
    if((w[LOBE_SPECULAR_REFLECTION] > 0.0f || w[LOBE_METAL_REFLECTION] > 0.0f || w[LOBE_SPECULAR_TRANSMISSION] > 0.0f)
       && alpha > SMOOTH_VERTEX_ALPHA)
      return false;
    if(w[LOBE_CLEARCOAT_REFLECTION] > 0.0f && mat.clearcoatRoughness * mat.clearcoatRoughness > SMOOTH_VERTEX_ALPHA)
      return false;
    if(w[LOBE_SPECULAR2_REFLECTION] > 0.0f && mat.specular2Roughness * mat.specular2Roughness > SMOOTH_VERTEX_ALPHA)
      return false;
  }
  return !(mat.retroreflection > 0.0f);
}

// A surface of perceptual `roughness`, clamped and squared as gltf_material_eval.h.slang does
PbrMaterial smoothCase(float roughness, float metallic, float transmission)
{
  PbrMaterial mat  = defaultPbrMaterial(vec3(0.9f), metallic, std::max(roughness, 0.0014142f), vec3(0, 0, 1), vec3(0, 0, 1));
  mat.transmission = transmission;
  return mat;
}

}  // namespace

// smoothVertex holds exactly where the BSDF's weights give nothing to a diffuse lobe or the sheen and
// every weighted GGX lobe is at or under SMOOTH_VERTEX_ALPHA: the bar's glass and bottles (roughness
// 0), a tear film at the lowering's cap, polished metal under a smooth coat; not the frosted glass at
// the 0.1 cap, nor 1% of diffuse under glass, nor a rough coat or second lobe over a mirror.
TEST(SmoothVertex, EveryWeightedLobeIsAMirrorAndNoneDiffuse)
{
  struct Row
  {
    const char* name;
    PbrMaterial mat;
    bool        smooth;
  };
  std::vector<Row> rows;
  rows.push_back({"clear glass, roughness 0", smoothCase(0.0f, 0.0f, 1.0f), true});
  PbrMaterial film = smoothCase(0.02f, 0.0f, 1.0f);  // the tear film's cap (figure lowering)
  film.ior2        = 1.33f;
  film.thickness   = 0.01f;
  rows.push_back({"tear film at its cap", film, true});
  PbrMaterial atThreshold = smoothCase(0.0f, 0.0f, 1.0f);
  atThreshold.roughness   = vec2(SMOOTH_VERTEX_ALPHA);
  rows.push_back({"glass at the threshold", atThreshold, true});
  rows.push_back({"glass just past it", smoothCase(0.0245f, 0.0f, 1.0f), false});
  rows.push_back({"frosted glass at the clear-crossing cap", smoothCase(0.1f, 0.0f, 1.0f), false});
  rows.push_back({"glass over 1% of diffuse", smoothCase(0.0f, 0.0f, 0.99f), false});
  PbrMaterial pane = smoothCase(0.0f, 0.0f, 1.0f);
  pane.plate       = true;
  rows.push_back({"a pane", pane, true});
  PbrMaterial sheet               = smoothCase(0.0f, 0.0f, 1.0f);
  sheet.diffuseTransmissionFactor = 1.0f;  // under transmission 1 it has no weight
  rows.push_back({"glass with an unweighted diffuse transmission", sheet, true});
  rows.push_back({"half metal, half glass", smoothCase(0.0f, 0.5f, 1.0f), true});
  rows.push_back({"polished metal", smoothCase(0.0f, 1.0f, 0.0f), true});
  rows.push_back({"metal at 0.99", smoothCase(0.0f, 0.99f, 0.0f), false});
  rows.push_back({"smooth plastic", smoothCase(0.0f, 0.0f, 0.0f), false});
  PbrMaterial coated        = smoothCase(0.0f, 1.0f, 0.0f);
  coated.clearcoat          = 1.0f;
  coated.clearcoatRoughness = 0.01f;
  rows.push_back({"metal under a smooth coat", coated, true});
  coated.clearcoatRoughness = 0.3f;
  rows.push_back({"metal under a rough coat", coated, false});
  PbrMaterial dual        = smoothCase(0.0f, 1.0f, 0.0f);
  dual.specular2          = 0.5f;
  dual.specular2Roughness = 0.3f;
  rows.push_back({"a rough second lobe over a mirror", dual, false});
  PbrMaterial velvet = smoothCase(0.0f, 1.0f, 0.0f);
  velvet.sheenColor     = vec3(0.3f);
  velvet.sheenRoughness = 0.5f;
  rows.push_back({"sheen over a mirror", velvet, false});
  PbrMaterial brushed = smoothCase(0.0f, 1.0f, 0.0f);
  brushed.roughness.x = 0.01f;
  rows.push_back({"anisotropic metal, rough across", brushed, false});
  PbrMaterial retro     = smoothCase(0.0f, 1.0f, 0.0f);
  retro.retroreflection = 0.5f;
  rows.push_back({"retroreflective mirror", retro, false});
  for(const Row& r : rows)
  {
    EXPECT_EQ(smoothVertex(r.mat), r.smooth) << r.name;
    EXPECT_EQ(mirrorsByTheirWeights(r.mat), r.smooth) << r.name << ", by the BSDF's weights";
  }
}

// A smooth vertex's light toward any direction is counted once: by its BSDF ray where that ray can go
// there (its pdf above 0), and otherwise by its light sample, kept (smoothVertexSamplesLight) and
// weighed 1 (smoothVertexLightWeight), as past the face from where a lobe leaves (bsdfFaceAllows). With
// the lobe mixture (bsdfEvaluateAll, whose pdf is the density the sampler draws by) and with one lobe
// picked at random (bsdfEvaluate, each lobe weighed against its own sampler). A punctual light keeps its
// sample at weight 1. Glass, a thin pane, a tear film, metal and coated metal, flat and with a normal
// map tilting their lobes 20 degrees past the face, from views all round.
TEST(SmoothVertex, EachDirectionsLightIsCountedOnce)
{
  std::vector<PbrMaterial> flat;
  flat.push_back(smoothCase(0.0f, 0.0f, 1.0f));
  PbrMaterial pane = smoothCase(0.0f, 0.0f, 1.0f);
  pane.plate       = true;
  flat.push_back(pane);
  PbrMaterial film = smoothCase(0.02f, 0.0f, 1.0f);
  film.ior2        = 1.33f;
  film.thickness   = 0.01f;
  flat.push_back(film);
  flat.push_back(smoothCase(0.0f, 1.0f, 0.0f));
  PbrMaterial coated        = smoothCase(0.0f, 1.0f, 0.0f);
  coated.clearcoat          = 1.0f;
  coated.clearcoatRoughness = 0.01f;
  flat.push_back(coated);

  std::mt19937 rng(53);
  int          dropped = 0, keptAlone = 0, valued = 0;
  for(int tilt = 0; tilt < 2; tilt++)
    for(PbrMaterial mat : flat)
    {
      if(tilt)
      {
        mat.N  = normalize(vec3(std::sin(radians(20.0f)), 0.0f, std::cos(radians(20.0f))));
        mat.Nc = normalize(vec3(-0.2f, 0.1f, 1.0f));
        orthonormalBasis(mat.N, mat.T, mat.B);
      }
      ASSERT_TRUE(smoothVertex(mat));
      for(int i = 0; i < 20000; i++)
      {
        const vec3 k1 = viewAbove(rng), k2 = sphere(rng);
        ASSERT_TRUE(smoothVertexSamplesLight(mat, k2, DIRAC));
        EXPECT_EQ(smoothVertexLightWeight(DIRAC, 1e6f), 1.0f);
        const bool kept = smoothVertexSamplesLight(mat, k2, 1.0f);
        dropped += kept ? 0 : 1;
        for(int lobe = 0; lobe < 4; lobe++)  // the mixture, then a random lobe at three numbers
        {
          BsdfEvaluateData e;
          e.k1 = k1;
          e.k2 = k2;
          e.xi = vec3(0.5f, 0.5f, lobe == 0 ? 0.5f : (float(lobe) - 0.5f) / 3.0f);
          if(lobe == 0)
            bsdfEvaluateAll(e, mat);
          else
            bsdfEvaluate(e, mat);
          if(!(sum(e.bsdf_diffuse + e.bsdf_glossy) > 0.0f))
            continue;
          valued++;
          const float lightWeight = kept ? smoothVertexLightWeight(1.0f, e.pdf) : 0.0f;
          // where the BSDF ray goes, its hit weighs 1 (bsdfHitMisWeight with bsdfOnly: test_light_pick.cpp)
          const float bsdfWeight = (e.pdf > 0.0f) ? 1.0f : 0.0f;
          EXPECT_EQ(lightWeight + bsdfWeight, 1.0f) << "tilt " << tilt << ", lobe " << lobe << ", k2 " << k2.x << " " << k2.y << " "
                                                    << k2.z << ", pdf " << e.pdf;
          keptAlone += (kept && !(e.pdf > 0.0f)) ? 1 : 0;
        }
      }
    }
  EXPECT_GT(dropped, 80000);  // of 200000: most of the hemisphere above the face
  EXPECT_GT(valued, 100000);
  EXPECT_GT(keptAlone, 100);  // the tilted lobes do reach past the face
  // Without a face (the ground, the water) the BSDF ray reaches everywhere
  PbrMaterial loose = smoothCase(0.0f, 0.0f, 1.0f);
  loose.faceSided   = false;
  EXPECT_FALSE(smoothVertexSamplesLight(loose, vec3(0, 0, -1), 1.0f));
  EXPECT_TRUE(smoothVertexSamplesLight(loose, vec3(0, 0, -1), DIRAC));
}

// What DLSS-RR's specular albedo guide takes of a named metal (shaders/dlss_specular_guide.h.slang):
// the share of the Schlick lobe's directional albedo its edge tint leaves, against the metal lobe's own,
// sampled with each Fresnel from the same random numbers.
#include "../shaders/dlss_specular_guide.h.slang"

namespace {

// figure/materials/metals.cpp's head-on colours and edge tints: the four the tint takes most from
struct NamedMetal
{
  const char* name;
  vec3        f0, tint;
};
const NamedMetal kTintedMetals[] = {
    {"chrome", vec3(0.5496f, 0.5561f, 0.5537f), vec3(0.7418f, 0.7480f, 0.8150f)},
    {"iron", vec3(0.5289f, 0.5136f, 0.4938f), vec3(0.7655f, 0.7669f, 0.8024f)},
    {"titanium", vec3(0.6170f, 0.5799f, 0.5424f), vec3(0.7696f, 0.7901f, 0.8211f)},
    {"aluminium", vec3(0.9067f, 0.9161f, 0.9226f), vec3(0.8891f, 0.9179f, 0.9477f)},
};

// The metal lobe's single-scattering albedo with F82 over that with Schlick's, per channel
vec3 edgeTintShareOfTheLobe(const NamedMetal& m, float alpha, float NoV, int N = 40000)
{
  PbrMaterial mat = defaultPbrMaterial(m.f0, 1.0f, std::sqrt(alpha), vec3(0, 0, 1), vec3(0, 0, 1));
  mat.edgeTint    = m.tint;
  std::mt19937                          rng(41);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const vec3 k1 = vec3(std::sqrt(1.0f - NoV * NoV), 0.0f, NoV);
  dvec3      f82(0.0), schlick(0.0);
  for(int i = 0; i < N; i++)
  {
    BsdfSampleData s;
    s.k1             = k1;
    s.xi             = vec3(u(rng), u(rng), u(rng));
    BsdfSampleData t = s;
    brdf_ggx_smith_sample(s, mat, LOBE_METAL_REFLECTION, m.f0, BSDF_FRESNEL_F82, 1.0f);
    brdf_ggx_smith_sample(t, mat, LOBE_METAL_REFLECTION, m.f0, BSDF_FRESNEL_SCHLICK, 1.0f);
    if(s.event_type != BSDF_EVENT_ABSORB)
      f82 += dvec3(s.bsdf_over_pdf);
    if(t.event_type != BSDF_EVENT_ABSORB)
      schlick += dvec3(t.bsdf_over_pdf);
  }
  return vec3(f82 / glm::max(schlick, dvec3(1e-12)));
}

}  // namespace

// The guide's share is the lobe's within 5% for every named metal, from a mirror to alpha 1 and from
// head-on to 87 degrees: Schlick's alone (a share of 1) is a third too bright on a polished chrome rim,
// where the moving view and the still would then disagree.
TEST(DlssSpecularGuide, TheEdgeTintShareIsTheLobes)
{
  double worst = 0.0, worstSchlick = 0.0;
  for(const NamedMetal& m : kTintedMetals)
    for(float alpha : {0.001f, 0.03f, 0.1f, 0.2f, 0.4f, 0.7f, 1.0f})
      for(float NoV : {1.0f, 0.7f, 0.4f, 0.2f, 0.1f, 0.05f})
      {
        const vec3 lobe  = edgeTintShareOfTheLobe(m, alpha, NoV);
        const vec3 guide = dlssEdgeTintShare(m.f0, m.tint, alpha, NoV);
        for(int c = 0; c < 3; c++)
        {
          const double err = std::abs(guide[c] - lobe[c]) / lobe[c];
          worst            = std::max(worst, err);
          worstSchlick     = std::max(worstSchlick, std::abs(1.0 - lobe[c]) / lobe[c]);
          EXPECT_LT(err, 0.05) << m.name << " alpha " << alpha << " NoV " << NoV << " channel " << c << ": guide "
                               << guide[c] << ", lobe " << lobe[c];
        }
      }
  EXPECT_GT(worstSchlick, 0.3) << "the tint should matter on chrome's rim";
  EXPECT_LT(worst, 0.05);
}

// A white tint leaves the guide Schlick's (a share of 1); a mirror takes the Fresnel's own ratio at the
// view, and from DLSS_EDGE_TINT_ROUGH_ALPHA on a lobe takes the average one, as its multiple
// scattering does (f82AverageShare).
TEST(DlssSpecularGuide, TheShareRunsFromTheViewsFresnelToTheAverage)
{
  const NamedMetal& chrome = kTintedMetals[0];
  for(float alpha : {0.0f, 0.1f, 0.5f, 1.0f})
    for(float NoV : {1.0f, 0.5f, 0.15f, -0.3f})
    {
      const vec3 white = dlssEdgeTintShare(chrome.f0, vec3(1.0f), alpha, NoV);
      EXPECT_NEAR(white.x, 1.0f, 1e-6f) << alpha << " " << NoV;
      EXPECT_NEAR(white.z, 1.0f, 1e-6f) << alpha << " " << NoV;
    }
  for(float NoV : {1.0f, 0.5f, 0.15f, 0.05f})
  {
    const vec3 mirror = dlssEdgeTintShare(chrome.f0, chrome.tint, 0.0f, NoV);
    const vec3 view   = f82Fresnel(chrome.f0, chrome.tint, NoV) / schlickFresnel(chrome.f0, vec3(1.0f), NoV);
    EXPECT_NEAR(mirror.x, view.x, 1e-6f) << NoV;
    EXPECT_NEAR(dlssEdgeTintShare(chrome.f0, chrome.tint, 0.0f, -NoV).x, view.x, 1e-6f) << "the far side, " << NoV;
    for(float alpha : {DLSS_EDGE_TINT_ROUGH_ALPHA, 1.0f})
      EXPECT_NEAR(dlssEdgeTintShare(chrome.f0, chrome.tint, alpha, NoV).x, f82AverageShare(chrome.f0, chrome.tint).x, 1e-6f)
          << alpha << " " << NoV;
  }
  // chrome's rim, polished: about three quarters of Schlick's
  EXPECT_NEAR(dlssEdgeTintShare(chrome.f0, chrome.tint, 0.0f, 0.15f).x, 0.74f, 0.01f);
}

// The see-through guides take what the pixel is made of: the surface behind clear glass as it comes
// through (tinted by the glass, dimmed by the water), and the reflecting surface's own normal, depth
// and motion where its reflection outweighs that (dlssReflectionShare, gltf_pathtrace.slang).
TEST(DlssSpecularGuide, TheSeeThroughGuidesFollowWhatMakesThePixel)
{
  const auto schlick = [](float eta, float mu) {
    const float f0 = ((eta - 1.0f) / (eta + 1.0f)) * ((eta - 1.0f) / (eta + 1.0f));
    return f0 + (1.0f - f0) * std::pow(1.0f - mu, 5.0f);
  };
  const vec3 behind(0.5f, 0.4f, 0.3f);

  // clear glass head-on: the reflection's share is its Fresnel, the albedo the blend it was
  const float glass = schlick(1.5f, 1.0f);
  EXPECT_NEAR(dlssReflectionShare(glass, vec3(1.0f)), glass, 1e-6f);
  const vec3 clear = dlssSeeThroughAlbedo(behind, vec3(1.0f), glass, vec3(1.0f));
  EXPECT_NEAR(clear.x, 0.5f + glass * 0.5f, 1e-6f);

  // 7TEST's tinted car window: what is behind it comes through tinted, and still decides the guides
  const vec3 tint(0.77f, 0.84f, 0.83f);
  const vec3 tinted = dlssSeeThroughAlbedo(behind, vec3(1.0f), glass, tint);
  EXPECT_NEAR(tinted.y, (1.0f - glass) * 0.84f * 0.4f + glass, 1e-6f);
  EXPECT_LT(dlssReflectionShare(glass, tint), DLSS_REFLECTION_KEEPS_SURFACE);
  // ...but seen edge-on its reflection does: the window keeps its own guides
  EXPECT_GT(dlssReflectionShare(schlick(1.5f, std::cos(85.0f * 3.14159265f / 180.0f)), tint), DLSS_REFLECTION_KEEPS_SURFACE);
  // a pane too dark to see through is a mirror, head-on too
  EXPECT_GT(dlssReflectionShare(glass, vec3(0.02f)), DLSS_REFLECTION_KEEPS_SURFACE);

  // the water: a metre down dims as its absorption says, nothing at its surface
  const vec3 absorption(0.29f, 0.055f, 0.011f);  // ground.h.slang's GROUND_WATER_ABSORPTION
  const vec3 metre = dlssWaterThrough(absorption, 1.0f);
  EXPECT_NEAR(metre.x, std::exp(-0.29f), 1e-6f);
  EXPECT_NEAR(dlssWaterThrough(absorption, 0.0f).z, 1.0f, 1e-6f);
  EXPECT_NEAR(dlssWaterThrough(absorption, -1.0f).x, 1.0f, 1e-6f);
  // from above the floor shows through; from a low camera the sky's reflection takes over
  const float water = schlick(1.33f, 1.0f);
  EXPECT_LT(dlssReflectionShare(water, metre), 0.05f);
  EXPECT_GT(dlssReflectionShare(schlick(1.33f, std::cos(85.0f * 3.14159265f / 180.0f)), metre), DLSS_REFLECTION_KEEPS_SURFACE);
  EXPECT_LT(dlssReflectionShare(schlick(1.33f, std::cos(60.0f * 3.14159265f / 180.0f)), metre), DLSS_REFLECTION_KEEPS_SURFACE);
}

// Under water the guide point sits where the eye sees the surface behind: through flat surfaces, on the
// line through it along their normal, which is the camera ray's point at s n0 / n per segment
// (dlssSeenScale). Built here from Snell's law alone: the ray refracted at the surface, the point that
// far down it, and where the camera ray's line meets the vertical through that point.
TEST(DlssSpecularGuide, UnderWaterTheGuideSitsWhereTheEyeSeesIt)
{
  const vec3  up(0.0f, 1.0f, 0.0f);
  const float water = 1.33f, glass = 1.5f;
  // head-on, the floor 50 cm down looks 38 cm deep
  EXPECT_NEAR(0.5f * dlssSeenScale(1.0f, 1.0f, water), 0.376f, 1e-3f);
  EXPECT_NEAR(dlssSeenDepthRatio(1.0f, 1.0f, water), 1.0f / water, 1e-6f);

  for(float degrees : {15.0f, 30.0f, 45.0f, 60.0f, 75.0f})
  {
    const float a = degrees * 3.14159265f / 180.0f;
    const vec3  d(std::sin(a), -std::cos(a), 0.0f);  // the camera ray, meeting the water at the origin
    // the floor 50 cm down
    const vec3  t     = refract(d, up, 1.0f / water);
    const float s     = 0.5f / -t.y;
    const vec3  floor = s * t;
    const float seen  = floor.x / d.x;  // the camera ray's point over it
    EXPECT_NEAR(s * dlssSeenScale(1.0f, 1.0f, water), seen, 1e-5f * seen) << degrees;
    // that point is the floor with its depth times the ratio, at this view
    EXPECT_NEAR(seen * -d.y / 0.5f, dlssSeenDepthRatio(-d.y, -t.y, water), 1e-5f) << degrees;

    // through a glass block on the floor of the water, 30 cm down and 5 cm thick: each segment at
    // its own ratio, the ratios composed refraction by refraction
    const vec3  t2     = refract(t, up, water / glass);
    const float s1     = 0.3f / -t.y, s2 = 0.05f / -t2.y;
    const vec3  bottom = s1 * t + s2 * t2;
    const float scale1 = dlssSeenScale(1.0f, 1.0f, water), scale2 = dlssSeenScale(scale1, water, glass);
    EXPECT_NEAR(s1 * scale1 + s2 * scale2, bottom.x / d.x, 1e-5f * bottom.x / d.x) << degrees;
  }

  // from under the water, looking up through its surface: what is above is seen farther
  for(float degrees : {15.0f, 30.0f, 45.0f})
  {
    const float a = degrees * 3.14159265f / 180.0f;
    const vec3  d(std::sin(a), std::cos(a), 0.0f);
    const vec3  t = refract(d, -up, water / 1.0f);
    ASSERT_GT(t.y, 0.0f) << degrees;
    const float s    = 0.5f / t.y;
    const float seen = s * t.x / d.x;
    EXPECT_NEAR(s * dlssSeenScale(1.0f, water, 1.0f), seen, 1e-5f * seen) << degrees;
    EXPECT_GT(seen, s);
  }
}
