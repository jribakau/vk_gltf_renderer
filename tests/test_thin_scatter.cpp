// KHR_materials_scatter's thin-walled lobe split (shaders/thin_scatter.h.slang) compiled as C++:
// the spec's (1 + g) / 2 goes through the surface and the rest back, the material's own diffuse
// lobes keep their colours, and the path tracer's preview of a closed body sends nothing through.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/thin_scatter.h.slang"

namespace {

void expectNear(float3 a, float3 b, float tolerance = 1e-6f)
{
  EXPECT_NEAR(a.x, b.x, tolerance);
  EXPECT_NEAR(a.y, b.y, tolerance);
  EXPECT_NEAR(a.z, b.z, tolerance);
}

const float3 kSkin(0.8f, 0.5f, 0.4f);
const float3 kWhite(1.0f);

}  // namespace

// The preview of a closed body (g = -1): all of the scatter back out as diffuse reflection in the
// surface's colour, none into the body
TEST(ThinScatter, AClosedPreviewSendsNothingThrough)
{
  const ThinScatterLobes lobes = thinScatterLobes(0.95f, 1.0f, -1.0f, 0.0f, kWhite, kSkin, kSkin);
  EXPECT_EQ(lobes.diffuseTransmissionFactor, 0.0f);
  EXPECT_NEAR(lobes.transmission, 0.0f, 1e-7f);
  expectNear(lobes.baseColor, kSkin);
  expectNear(lobes.transmissionTint, kWhite);
}

// A thin wall at full transmission: the bucket is all scatter, split by the anisotropy
TEST(ThinScatter, AThinWallSplitsByTheAnisotropy)
{
  const float3           ms(0.2f, 0.6f, 0.9f);
  const ThinScatterLobes lobes = thinScatterLobes(1.0f, 1.0f, 0.6f, 0.0f, kWhite, kSkin, ms);
  EXPECT_NEAR(lobes.diffuseTransmissionFactor, 0.8f, 1e-6f);
  expectNear(lobes.diffuseTransmissionColor, ms);
  expectNear(lobes.baseColor, ms);
}

// Part of the transmission stays specular: its BTDF keeps the material's own base colour, which
// the blend of the diffuse reflection would otherwise tint
TEST(ThinScatter, TheResidualTransmissionKeepsItsColour)
{
  const float3           ms(0.2f, 0.6f, 0.9f);
  const ThinScatterLobes lobes = thinScatterLobes(0.8f, 0.5f, 0.2f, 0.0f, kWhite, kSkin, ms);
  EXPECT_NEAR(lobes.transmission, 0.4f, 1e-6f);
  expectNear(lobes.transmissionTint * lobes.baseColor, kSkin, 1e-5f);
}

// The material's own diffuse transmission shares its lobe with the scatter's forward half
TEST(ThinScatter, OwnDiffuseTransmissionSharesTheForwardLobe)
{
  const float3 ms(0.2f, 0.6f, 0.9f), dtColor(0.9f, 0.3f, 0.1f);
  const float  t = 0.6f, strength = 0.5f, g = 0.4f, dt = 0.5f;
  const float  base = 1.0f - t, scatter = t * strength, forward = scatter * 0.5f * (1.0f + g);
  const ThinScatterLobes lobes = thinScatterLobes(t, strength, g, dt, dtColor, kSkin, ms);
  EXPECT_NEAR(lobes.diffuseTransmissionFactor, (base * dt + forward) / (base + scatter), 1e-6f);
  expectNear(lobes.diffuseTransmissionColor, dtColor + (ms - dtColor) * (forward / (base * dt + forward)));
}

// With half the base left untouched, each diffuse lobe averages its two shares by energy
TEST(ThinScatter, TheBaseKeepsItsShareOfTheDiffuseLobes)
{
  const float3           ms(0.2f, 0.6f, 0.9f);
  const ThinScatterLobes lobes = thinScatterLobes(0.5f, 1.0f, 0.0f, 0.0f, kWhite, kSkin, ms);
  // bucket 1 = base 0.5 + scatter 0.5, a quarter of it forward
  EXPECT_NEAR(lobes.diffuseTransmissionFactor, 0.25f, 1e-6f);
  EXPECT_NEAR(lobes.transmission, 0.0f, 1e-7f);
  expectNear(lobes.diffuseTransmissionColor, ms);
  expectNear(lobes.baseColor, kSkin + (ms - kSkin) / 3.0f);
}
