// Where a curved mirror shows what it reflects (shaders/mirror_image.h.slang) compiled as C++: the
// radius of a smooth mesh from its vertex normals, and the specular hit distance DLSS-RR is given on
// an eye's tear film, half that radius behind it.

// GLM as test_bsdf_lobes.cpp, another file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/mirror_image.h.slang"

namespace {

// A point of a sphere of radius `radius` around `centre`, at polar angle `theta` from +z and
// azimuth `phi`, with its outward normal
struct SpherePoint
{
  vec3 position;
  vec3 normal;
};

SpherePoint onSphere(vec3 centre, float radius, float theta, float phi, float normalSign = 1.0f)
{
  const vec3 n(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
  return {centre + radius * n, normalSign * n};
}

float radiusOf(const SpherePoint& a, const SpherePoint& b, const SpherePoint& c)
{
  return meshMirrorRadius(b.position - a.position, c.position - a.position, b.normal - a.normal, c.normal - a.normal);
}

// A small triangle of a sphere, as a cornea's mesh has them
float sphereTriangleRadius(float radius, float normalSign = 1.0f)
{
  const vec3 centre(0.1f, -0.2f, 1.5f);
  return radiusOf(onSphere(centre, radius, 0.20f, 0.3f, normalSign), onSphere(centre, radius, 0.26f, 0.5f, normalSign),
                  onSphere(centre, radius, 0.23f, 0.9f, normalSign));
}

}  // namespace

// The radius comes back from a sphere's triangle whatever its size: a G9 cornea, a child's, and
// the eyeball's own curve behind it
TEST(MirrorImage, ASphereTriangleGivesItsRadius)
{
  for(const float radius : {0.0078f, 0.006f, 0.012f, 2.0f})
    EXPECT_NEAR(sphereTriangleRadius(radius), radius, radius * 1e-3f) << radius;
}

// Normals that don't turn, or turn away from the viewer's side, have no image in front of the far
// side: 0, and the shader leaves the traced distance
TEST(MirrorImage, FlatAndConcaveSurfacesHaveNone)
{
  EXPECT_EQ(meshMirrorRadius(vec3(0.01f, 0, 0), vec3(0, 0.01f, 0), vec3(0.0f), vec3(0.0f)), 0.0f);
  // a hundred-thousandth of a radian across the triangle: flat as far as a reflection can tell
  EXPECT_EQ(meshMirrorRadius(vec3(0.01f, 0, 0), vec3(0, 0.01f, 0), vec3(1e-5f, 0, 0), vec3(0.0f)), 0.0f);
  EXPECT_EQ(sphereTriangleRadius(0.0078f, -1.0f), 0.0f);
}

// The distance RR is given: half the radius behind the mirror itself, or what is left of it behind
// the iris that replaced the mirror's depth, and never at or before that depth
TEST(MirrorImage, TheImageLiesHalfTheRadiusBehind)
{
  const float cornea = 0.0078f;
  EXPECT_FLOAT_EQ(mirrorImageDistance(cornea, 0.0f), 0.0039f);
  EXPECT_FLOAT_EQ(mirrorImageDistance(cornea, 0.003f), 0.0009f);
  EXPECT_FLOAT_EQ(mirrorImageDistance(cornea, 0.005f), 0.05f * 0.0039f);
  EXPECT_GT(mirrorImageDistance(cornea, 1.0f), 0.0f);
}

// The premise, traced: a camera half a metre from a cornea sees a distant light's catchlight where
// the normal halves the way to the camera and to the light. Moving the camera a little, its two lines
// of sight to the catchlight meet R/2 behind the surface, which is the distance the shader gives.
// In doubles: two lines 4 mrad apart meet far from where floats can say.
TEST(MirrorImage, TwoViewsOfACatchlightMeetHalfTheRadiusBehind)
{
  const double R     = 0.0078;
  const dvec3  light = normalize(dvec3(std::sin(0.17), 0.0, std::cos(0.17)));  // 10 degrees off the view
  auto catchlight    = [&](dvec3 camera) {
    dvec3 n = normalize(camera);
    for(int i = 0; i < 50; ++i)
      n = normalize(normalize(camera - R * n) + light);
    return R * n;  // the sphere is centred at the origin
  };
  const dvec3 cameraA(0.0, 0.0, 0.5);
  const dvec3 cameraB(0.002, 0.0, 0.5);
  const dvec3 pA = catchlight(cameraA);
  const dvec3 pB = catchlight(cameraB);
  const dvec3 dA = normalize(pA - cameraA);
  const dvec3 dB = normalize(pB - cameraB);

  // where the two lines of sight pass closest, along A's: its distance past the catchlight
  const dvec3  w0 = cameraA - cameraB;
  const double a = dot(dA, dA), b = dot(dA, dB), c = dot(dB, dB), d = dot(dA, w0), e = dot(dB, w0);
  const double s      = (b * e - c * d) / (a * c - b * b);
  const double behind = s - length(pA - cameraA);
  EXPECT_NEAR(behind, mirrorImageDistance(sphereTriangleRadius(float(R)), 0.0f), 0.03 * R / 2);
}
