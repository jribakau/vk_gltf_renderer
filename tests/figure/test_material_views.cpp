// A material's pictures on the CPU (figure/tools/closure_probe.cpp probeViews, fv_parity's "slice"
// step): the slice's layout and what known lobes look like in it, and the two balls' lighting.

#include <cmath>

#include "materials_test_util.hpp"

namespace {

using fv::probe::MaterialViews;
using fv::probe::ViewImage;

constexpr int   kDiffuse = 0, kSpecular = 2;
constexpr float kPi      = 3.14159265358979f;

// Small pictures, few furnace samples: the tests read shapes, not converged numbers
fv::probe::ViewOptions small()
{
  fv::probe::ViewOptions v;
  v.sliceSize      = 32;
  v.ballSize       = 32;
  v.furnaceCosines = 8;
  v.furnaceSamples = 512;
  return v;
}

tinygltf::Material material(double roughness, double specular)
{
  tinygltf::Material m;
  m.pbrMetallicRoughness.baseColorFactor = {1.0, 1.0, 1.0, 1.0};
  m.pbrMetallicRoughness.metallicFactor  = 0.0;
  m.pbrMetallicRoughness.roughnessFactor = roughness;
  tinygltf::Value::Object s;
  s["specularFactor"]                    = tinygltf::Value(specular);
  m.extensions["KHR_materials_specular"] = tinygltf::Value(s);
  return m;
}

float green(const ViewImage& image, int x, int y)
{
  return image.at(x, y)[1];
}

}  // namespace

TEST(MaterialViews, ALambertSliceIsFlatAtItsAlbedoOverPi)
{
  const MaterialViews v = fv::probe::viewsOfGltf(material(1.0, 0.0), {}, small());
  EXPECT_TRUE(v.lobeUsed[kDiffuse]);
  EXPECT_FALSE(v.lobeUsed[kSpecular]);
  const int n = v.sliceTotal.width;
  for(int y = 0; y < n; y += 5)
    for(int x = 0; x < n; x += 5)
      EXPECT_NEAR(green(v.sliceTotal, x, y), 1.0f / kPi, 1e-3f) << "at " << x << ", " << y;
}

TEST(MaterialViews, AGlossyLobePeaksWhereTheHalfVectorIsTheNormalAndNarrowsWithRoughness)
{
  const MaterialViews sharp = fv::probe::viewsOfGltf(material(0.2, 1.0), {}, small());
  const MaterialViews broad = fv::probe::viewsOfGltf(material(0.6, 1.0), {}, small());
  ASSERT_TRUE(sharp.lobeUsed[kSpecular]);
  const ViewImage& s = sharp.slice[kSpecular];
  const ViewImage& b = broad.slice[kSpecular];
  const int        n = s.width, row = n / 4;  // theta_d about 22 degrees
  // along theta_h it falls from the left edge, to 45 degrees (further on the light and the view both
  // near grazing, and microfacet theory's 1 / (4 cos cos) lifts the tail of a sharp lobe again)
  for(int x = 1; x < n / 2; x++)
    EXPECT_LE(green(s, x, row), green(s, x - 1, row) * 1.0001f) << "at " << x;
  // the sharper lobe is higher at its peak and has fallen further by theta_h 30 degrees
  const int at30 = n / 3;
  EXPECT_GT(green(s, 0, row), green(b, 0, row));
  EXPECT_LT(green(s, at30, row) / green(s, 0, row), green(b, at30, row) / green(b, 0, row));
}

TEST(MaterialViews, TheLobesSlicesAddUpToTheSum)
{
  const MaterialViews v = fv::probe::viewsOfGltf(material(0.4, 1.0), {}, small());
  const int           n = v.sliceTotal.width;
  for(int y = 0; y < n; y += 7)
    for(int x = 0; x < n; x += 7)
    {
      float sum = 0.0f;
      for(int lobe = 0; lobe < fv::probe::kLobeCount; lobe++)
        sum += green(v.slice[lobe], x, y);
      EXPECT_NEAR(sum, green(v.sliceTotal, x, y), 1e-4f * std::max(1.0f, sum)) << "at " << x << ", " << y;
    }
}

TEST(MaterialViews, AWhiteLambertVanishesInTheFurnaceAndReadsOneFacingTheLight)
{
  fv::probe::ViewOptions o = small();
  o.light[0] = 0.0f;
  o.light[1] = 0.0f;
  o.light[2] = 1.0f;  // behind the eye: the ball's centre faces the light
  const MaterialViews v = fv::probe::viewsOfGltf(material(1.0, 0.0), {}, o);
  const int           n = v.furnace.width;
  for(int y = 2; y < n - 2; y += 3)
    for(int x = 2; x < n - 2; x += 3)
      EXPECT_NEAR(green(v.furnace, x, y), 1.0f, 0.02f) << "at " << x << ", " << y;
  EXPECT_NEAR(green(v.lit, n / 2, n / 2), 1.0f, 0.01f);
  // Lambert's cosine toward the rim; the backdrop off the ball
  EXPECT_LT(green(v.lit, n / 2, 1), 0.5f);
  EXPECT_EQ(green(v.lit, 0, 0), fv::probe::kLitBackdrop);
}

TEST(MaterialViews, TheSlicePictureIsOnAFixedLogScale)
{
  ViewImage image;
  image.width  = 3;
  image.height = 1;
  image.rgb    = {0.0f, 0.0f, 0.0f, 1e-3f, 1e-3f, 1e-3f, 1e2f, 1e2f, 1e2f};
  const auto px = fv::probe::slicePixels(image);
  EXPECT_EQ(px[0] + px[1] + px[2], 0);  // nothing: black
  EXPECT_EQ(px[3], 68);                 // the scale's foot: viridis' violet (#440154)
  EXPECT_EQ(px[5], 84);
  EXPECT_EQ(px[6], 253);                // its top: yellow (#fde725)
  EXPECT_EQ(px[7], 231);
}
