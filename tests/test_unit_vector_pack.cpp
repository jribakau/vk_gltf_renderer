// The vertex streams' normals and tangents in 32 bits (shaders/unit_vector_pack.h.slang): how far a
// direction moves on the way through, what no direction comes back as, and that a tangent keeps its
// handedness. The shaders and the CPU (uploads, the tests' fake GPU) run this header.

#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include <nvshaders/slang_types.h>

#include "../shaders/unit_vector_pack.h.slang"

namespace {

double degreesBetween(glm::vec3 a, glm::vec3 b)
{
  const double d = glm::dot(glm::dvec3(a), glm::dvec3(b)) / (glm::length(glm::dvec3(a)) * glm::length(glm::dvec3(b)));
  return std::acos(std::clamp(d, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
}

std::vector<glm::vec3> directions(int count)
{
  std::mt19937                     rng(7);
  std::normal_distribution<float>  g;
  std::vector<glm::vec3>           out;
  for(int i = 0; i < count; i++)
  {
    glm::vec3 v(g(rng), g(rng), g(rng));
    if(glm::length(v) > 1e-3f)
      out.push_back(glm::normalize(v));
  }
  // the octahedron's corners and edges, where the fold meets
  for(glm::vec3 v : {glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1), glm::vec3(0, 0, -1),
                     glm::vec3(1, 1, 0), glm::vec3(1, -1, 0), glm::vec3(-1, 0, -1), glm::vec3(0, 1, -1), glm::vec3(1, 1, -1), glm::vec3(-1, -1, -1)})
    out.push_back(glm::normalize(v));
  return out;
}

}  // namespace

// A normal comes back within 0.005 degrees, unit length, and never as the code for none.
TEST(UnitVectorPack, ANormalComesBackWithinAFewThousandthsOfADegree)
{
  double most = 0;
  for(const glm::vec3& n : directions(200000))
  {
    const uint32_t  packed = shaderio::packUnitVector(n);
    const glm::vec3 back   = shaderio::unpackUnitVector(packed);
    ASSERT_NE(packed & 0xFFFFu, shaderio::kUnitVectorZero);
    EXPECT_NEAR(glm::length(back), 1.f, 1e-5f);
    most = std::max(most, degreesBetween(n, back));
  }
  EXPECT_LT(most, 0.005);
  // its length doesn't matter, and an axis is exact
  EXPECT_EQ(shaderio::packUnitVector(glm::vec3(0, 5, 0)), shaderio::packUnitVector(glm::vec3(0, 1, 0)));
  for(glm::vec3 axis : {glm::vec3(1, 0, 0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)})
    EXPECT_EQ(shaderio::unpackUnitVector(shaderio::packUnitVector(axis)), axis);
}

// A tangent comes back within 0.01 degrees with its handedness, whichever it is.
TEST(UnitVectorPack, ATangentKeepsItsHandedness)
{
  double most = 0;
  int    i    = 0;
  for(const glm::vec3& t : directions(200000))
  {
    const float     w      = (i++ & 1) ? -1.f : 1.f;
    const uint32_t  packed = shaderio::packTangent(glm::vec4(t, w));
    const glm::vec4 back   = shaderio::unpackTangent(packed);
    ASSERT_NE(packed & 0xFFFFu, shaderio::kUnitVectorZero);
    ASSERT_EQ(back.w, w);
    EXPECT_NEAR(glm::length(glm::vec3(back)), 1.f, 1e-5f);
    most = std::max(most, degreesBetween(t, glm::vec3(back)));
  }
  EXPECT_LT(most, 0.01);
  // a w that is only its sign, as glTF writes it, and 0 counts as right-handed
  EXPECT_EQ(shaderio::unpackTangent(shaderio::packTangent(glm::vec4(1, 0, 0, -0.5f))).w, -1.f);
  EXPECT_EQ(shaderio::unpackTangent(shaderio::packTangent(glm::vec4(1, 0, 0, 0.f))).w, 1.f);
}

// No direction (a zero vector, or one that isn't finite) comes back as zero; a tangent's keeps its sign.
TEST(UnitVectorPack, NoDirectionComesBackAsZero)
{
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  for(glm::vec3 none : {glm::vec3(0), glm::vec3(nan, 0, 0), glm::vec3(0, inf, 1), glm::vec3(1e-35f, 0, 0)})
  {
    EXPECT_EQ(shaderio::packUnitVector(none) & 0xFFFFu, shaderio::kUnitVectorZero);
    EXPECT_EQ(shaderio::unpackUnitVector(shaderio::packUnitVector(none)), glm::vec3(0));
  }
  EXPECT_EQ(shaderio::unpackTangent(shaderio::packTangent(glm::vec4(0, 0, 0, -1))), glm::vec4(0, 0, 0, -1));
  EXPECT_EQ(shaderio::unpackTangent(shaderio::packTangent(glm::vec4(0, 0, 0, 1))), glm::vec4(0, 0, 0, 1));
}
