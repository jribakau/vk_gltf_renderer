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
