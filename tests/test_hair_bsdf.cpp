// The fibre BSDF (shaders/hair_bsdf.h.slang) compiled as C++: Chiang's orders conserve energy in a
// clear fibre, a dark fibre reflects its R lobe at the fibre's own Fresnel and nothing else, and the
// sampler's weights average to what the evaluation integrates to.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <cstring>
#include <random>

#include <gtest/gtest.h>

using namespace glm;
#include "../shaders/gltf_scene_io.h.slang"
using namespace shaderio;
namespace shaderio {
inline vec3 lerp(vec3 a, vec3 b, float t) { return glm::mix(a, b, t); }
}  // namespace shaderio
#include "../shaders/hair_bsdf.h.slang"
#include "../figure/materials/hair_color.hpp"

namespace {

HairHit fibre(float3 sigma, float h = 0.3f)
{
  HairHit hair  = hairHitZero();
  hair.sigma    = sigma;
  hair.betaM    = 0.3f;
  hair.betaN    = 0.3f;
  hair.strand   = float3(0, 1, 0);
  hair.phiY     = float3(1, 0, 0);
  hair.phiZ     = float3(0, 0, 1);
  hair.normal   = float3(0, 0, 1);
  hair.specular = 1.0f;
  hair.h        = h;
  return hair;
}

// What the fibre sends out over the sphere for light from wo: the integral of f (f is f·cos already)
float3 sphereIntegral(const HairHit& hair, float3 wo, int n = 400000)
{
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  dvec3                               sum(0.0);
  for(int i = 0; i < n; i++)
  {
    const float  z   = 1.0f - 2.0f * u(rng);
    const float  r   = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float  phi = 6.2831853f * u(rng);
    const float3 wi(r * std::cos(phi), z, r * std::sin(phi));
    sum += dvec3(hairCombined(wo, wi, hair).f);
  }
  return float3(sum * (4.0 * 3.14159265358979 / n));
}

const float3 kWo = normalize(float3(0.2f, 0.3f, 1.0f));

}  // namespace

// A clear fibre (no absorption) loses nothing: R, TT, TRT and the rest sum to one
TEST(HairBsdf, AClearFibreConservesEnergy)
{
  const float3 albedo = sphereIntegral(fibre(float3(0.0f)), kWo);
  EXPECT_NEAR(albedo.x, 1.0f, 0.02f);
}

// A black fibre returns only what its surface reflects: the R lobe at the fibre's Fresnel, whole. The
// Lambert stand-in it had cut that to 65% and added a grey 35% of c^0.45
TEST(HairBsdf, ADarkFibreReflectsItsFresnelAndNothingElse)
{
  const HairHit hair = fibre(float3(40.0f));
  const float   sinO = dot(kWo, hair.strand);
  const float   cosO = std::sqrt(1.0f - sinO * sinO);
  const float   F    = hairFresnel(cosO * std::sqrt(1.0f - hair.h * hair.h), HAIR_ETA);
  const float3  albedo = sphereIntegral(hair, kWo);
  EXPECT_NEAR(albedo.x, F, 0.03f * F);
}

// A strand runs from root to tip whichever face of its ribbon a ray meets: a back face's frame comes
// turned to the ray (HitState::frameSign), and the fibre takes the mesh's own. So both faces answer a
// pair of directions alike, where the turned frame shifted the highlights the other way (the cuticle's
// tilt). A strand hair's root is at hairRootV; a card's isn't known, so its strands keep the
// bitangent's way, on both faces.
TEST(HairBsdf, AStrandRunsTheSameWayOnBothFacesOfARibbon)
{
  // a ribbon in the xy plane, UV v growing along +y: the renderer's bitangent points to smaller v, -y
  const float3 T(1.0f, 0.0f, 0.0f), B(0.0f, -1.0f, 0.0f), N(0.0f, 0.0f, 1.0f);
  const auto   same = [](float3 a, float3 b) { return length(a - b) < 1e-6f; };

  GltfShadeMaterial strands{};
  strands.hairFactor   = 1.0f;
  strands.hairRootSigma = float3(0.3f, 0.6f, 1.2f);
  strands.hairTipSigma  = float3(0.3f, 0.6f, 1.2f);
  strands.hairLongitudinal = 0.3f;
  strands.hairAzimuthal    = 0.3f;
  strands.hairSpecular     = 1.0f;
  strands.hairRootV        = 0.0f;  // the root at v = 0: root to tip is +y
  HairHit front, back;
  ASSERT_TRUE(makeHairHit(strands, float2(0.5f), B, T, 1.0f, N, float3(0.5f), front));
  ASSERT_TRUE(makeHairHit(strands, float2(0.5f), -B, -T, -1.0f, -N, float3(0.5f), back));
  EXPECT_TRUE(same(front.strand, float3(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(same(back.strand, front.strand));
  EXPECT_TRUE(same(back.phiY, front.phiY));
  EXPECT_TRUE(same(back.phiZ, front.phiZ));
  // the same light, seen the same way: the same response, a reflection and a transmission alike
  front.h = back.h  = 0.3f;
  const float3 lit  = normalize(float3(-0.4f, 0.5f, 0.8f));
  const float3 seen = normalize(float3(0.3f, 0.1f, -0.9f));
  for(const float3 wi : {lit, seen})
  {
    const HairResponse a = hairCombined(kWo, wi, front), b = hairCombined(kWo, wi, back);
    EXPECT_GT(a.f.x, 0.0f);
    EXPECT_FLOAT_EQ(a.f.x, b.f.x);
    EXPECT_FLOAT_EQ(a.pdf, b.pdf);
  }

  // the root at v = 1 (a figure's UVs turned over as the textures are): along the bitangent
  strands.hairRootV = 1.0f;
  ASSERT_TRUE(makeHairHit(strands, float2(0.5f), -B, -T, -1.0f, -N, float3(0.5f), back));
  EXPECT_TRUE(same(back.strand, B));

  // a card: the bitangent's way on both faces
  GltfShadeMaterial card = strands;
  card.hairFromColor     = 1.0f;
  card.hairRootV         = 0.0f;
  ASSERT_TRUE(makeHairHit(card, float2(0.5f), B, T, 1.0f, N, float3(0.5f), front));
  ASSERT_TRUE(makeHairHit(card, float2(0.5f), -B, -T, -1.0f, -N, float3(0.5f), back));
  EXPECT_TRUE(same(front.strand, B));
  EXPECT_TRUE(same(back.strand, B));
}

// The sampler's weights average to the evaluation's integral, and the pdf it reports is the one the
// evaluation gives the same direction
TEST(HairBsdf, SamplingCarriesWhatEvaluationIntegrates)
{
  const HairHit                         hair = fibre(float3(0.3f, 0.6f, 1.2f));
  std::mt19937                          rng(3);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const int                             n = 400000;
  dvec3                               sum(0.0);
  int                                   pdfMismatches = 0;
  for(int i = 0; i < n; i++)
  {
    BsdfSampleData d;
    d.k1 = kWo;
    d.xi = float3(u(rng), u(rng), u(rng));
    hairBsdfSample(d, hair, u(rng));
    if(d.event_type == BSDF_EVENT_ABSORB)
      continue;
    sum += dvec3(d.bsdf_over_pdf);
    if(std::abs(hairCombined(kWo, d.k2, hair).pdf - d.pdf) > 1e-3f * d.pdf)
      pdfMismatches++;
  }
  const float3 sampled = float3(sum / double(n));
  const float3 expected = sphereIntegral(hair, kWo);
  EXPECT_NEAR(sampled.x, expected.x, 0.02f * expected.x);
  EXPECT_NEAR(sampled.z, expected.z, 0.03f * expected.z);
  EXPECT_EQ(pdfMismatches, 0);
}

// A hair colour means one thing: the figure library's colour and absorption (hair_color.hpp, the
// DSON dyes, swatches and recolours) are the shader's (cards' colours, OmniHair's dyes, the lock
// colour the cut's tail carries)
TEST(HairBsdf, TheLibrarysHairColourIsTheShaders)
{
  for(const float b : {0.0f, 0.2f, 0.45f, 1.0f})
  {
    EXPECT_NEAR(fv::mat::hairColorFit(b), hairColorFit(b), 1e-5f) << b;
    for(const float c : {0.05f, 0.3f, 0.8f})
    {
      const float3 shader = hairColorToAbsorption(float3(c), b);
      EXPECT_NEAR(fv::mat::hairAbsorption(c, b), shader.x, 1e-4f * std::max(1.0f, shader.x)) << b << " " << c;
      HairHit hair = fibre(shader);
      hair.betaN   = b;
      EXPECT_NEAR(fv::mat::hairLockColor(shader.x, b), hairLockColor(hair).x, 1e-5f) << b << " " << c;
      EXPECT_NEAR(hairLockColor(hair).x, c, 1e-4f) << b << " " << c;
    }
  }
}
