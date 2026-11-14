// The cornea's caustic (shaders/eye_caustic.h.slang, eye_caustic_solve.h.slang) compiled as C++: the
// walk finds the ray Snell's law gives, a flat film changes nothing, a curved one focuses as a lens
// of its radius does, and over a whole iris the light it puts at each point is the light a beam traced
// through the same film from the light's side lands there.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/eye_caustic.h.slang"

namespace {

constexpr float kIor = 1.336f;  // the reduced eye's chamber (figure/materials/lower.hpp kEyeChamberIor)

// A film that is a sphere's cap over the plane z = 0, its apex `apex` above it: Genesis 9's tear film
// is 8.3 across its radius and 2.6 above a near-flat iris of radius 6.2 (millimetres here). Flat with
// radius 0: the plane z = apex.
struct CapFilm
{
  float3 point;  // the shaded point
  float  radius;
  float  apex;
  float  floor;  // the cap ends at this height: below it the film is not this surface

  float3 centre() const { return float3(0.0f, 0.0f, apex - radius); }

  EyeExit leave(float3 origin, float3 w, float4 light) const
  {
    const float3 o = point + origin;
    if(radius <= 0.0f)
    {
      if(!(w.z > 1e-6f))
        return eyeNoExit();
      return eyeFilmExit(origin, w, (apex - o.z) / w.z, float3(0.0f, 0.0f, 1.0f), kIor, light);
    }
    // from inside the sphere: the far root
    const float3 oc = o - centre();
    const float  b  = dot(oc, w);
    const float  c  = dot(oc, oc) - radius * radius;
    const float  h  = b * b - c;
    if(c > 0.0f || h < 0.0f)
      return eyeNoExit();
    const float  t = -b + std::sqrt(h);
    const float3 x = o + w * t;
    if(x.z < floor)
      return eyeNoExit();
    return eyeFilmExit(origin, w, t, (x - centre()) / radius, kIor, light);
  }
};

const CapFilm kEye{float3(0.0f), 8.3f, 2.6f, 0.0f};

}  // namespace

#define EYE_FILM CapFilm
#include "../shaders/eye_caustic_solve.h.slang"

namespace {

float3 lightFrom(float degrees)
{
  const float a = glm::radians(degrees);
  return float3(std::sin(a), 0.0f, std::cos(a));
}

// The light on the plane z = 0 at `p`, as a share of the light square to its own direction, by the
// walk: its focus times the cosine it arrives at. 0 where the walk finds no ray.
float solved(const CapFilm& shape, float3 p, float3 toLight)
{
  CapFilm film = shape;
  film.point   = p;
  const EyeCaustic c = eyeCausticSolve(film, toLight, float3(0.0f, 0.0f, 1.0f), float4(toLight, 0.0f), 0.0f);
  return c.valid ? c.focus * std::max(c.arrival.z, 0.0f) : 0.0f;
}

// The same by brute force from the light's side: a parallel beam through the cap, each ray bent into
// the chamber and followed to the plane, counted in cells of `cell` across a square of half-width
// `half`. A cell's count over what an unbent beam square to the cell would leave is its light.
struct Landed
{
  int                cells = 0;
  float              cell  = 0, half = 0;
  std::vector<float> light;
  float              at(float x, float y) const
  {
    const int i = int(std::floor((x + half) / cell)), j = int(std::floor((y + half) / cell));
    return (i >= 0 && j >= 0 && i < cells && j < cells) ? light[size_t(j) * cells + i] : 0.0f;
  }
};

Landed traced(const CapFilm& film, float3 toLight, float half, float cell, int perSide)
{
  Landed out;
  out.cell  = cell;
  out.half  = half;
  out.cells = int(std::round(2.0f * half / cell));
  out.light.assign(size_t(out.cells) * out.cells, 0.0f);
  const EyeFrame f      = eyeFrame(toLight);
  const float    reach  = 3.0f * (film.radius + half);
  const float    step   = 2.0f * reach / float(perSide);
  const float    weight = step * step / (cell * cell);  // a ray's share of the beam, in cells
  const float3   centre = film.centre();
  for(int j = 0; j < perSide; j++)
    for(int i = 0; i < perSide; i++)
    {
      const float3 o = f.a * (-reach + (i + 0.5f) * step) + f.b * (-reach + (j + 0.5f) * step) + toLight * (4.0f * reach);
      const float3 d = -toLight;
      // into the sphere from outside: the near root, on the cap
      const float3 oc = o - centre;
      const float  b  = dot(oc, d);
      const float  h  = b * b - (dot(oc, oc) - film.radius * film.radius);
      if(h < 0.0f)
        continue;
      const float3 x = o + d * (-b - std::sqrt(h));
      if(x.z < film.floor)
        continue;
      const float3 n    = (x - centre) / film.radius;
      const float3 bent = refract(d, n, 1.0f / kIor);
      if(!(bent.z < -1e-6f))
        continue;
      const float3 q = x + bent * (-x.z / bent.z);
      const int    ci = int(std::floor((q.x + half) / cell)), cj = int(std::floor((q.y + half) / cell));
      if(ci >= 0 && cj >= 0 && ci < out.cells && cj < out.cells)
        out.light[size_t(cj) * out.cells + ci] += weight;
    }
  return out;
}

}  // namespace

// Through a flat film the walk arrives on the ray Snell's law gives, and the light on the plane under
// it is what the straight sample had: the cosine of the light on the film
TEST(EyeCaustic, AFlatFilmBendsTheRayAndMovesNoLight)
{
  CapFilm flat{float3(0.3f, -0.2f, 0.0f), 0.0f, 2.6f, 0.0f};
  for(float degrees : {0.0f, 20.0f, 45.0f, 70.0f, 85.0f})
  {
    const float3     toLight = lightFrom(degrees);
    const EyeCaustic c       = eyeCausticSolve(flat, toLight, float3(0.0f, 0.0f, 1.0f), float4(toLight, 0.0f), 0.0f);
    ASSERT_TRUE(c.valid) << degrees;
    // sin(out) = n sin(in)
    const float sinIn = std::sqrt(std::max(1.0f - c.arrival.z * c.arrival.z, 0.0f));
    EXPECT_NEAR(sinIn * kIor, std::sin(glm::radians(degrees)), 3e-3f) << degrees;
    EXPECT_NEAR(c.arrival.y, 0.0f, 1e-3f);
    EXPECT_NEAR(c.focus * c.arrival.z, toLight.z, 0.01f + 0.02f * toLight.z) << degrees;
    // the shadow ray leaves from where the ray crossed the film, toward the light
    EXPECT_NEAR(c.exit.z, 2.6f, 1e-3f);
    EXPECT_NEAR(dot(c.toLight, toLight), 1.0f, 1e-5f);
  }
  // from below the film's plane no ray comes through it
  const float3 under = lightFrom(100.0f);
  const EyeCaustic dark = eyeCausticSolve(flat, under, float3(0.0f, 0.0f, 1.0f), float4(under, 0.0f), 0.0f);
  EXPECT_TRUE(dark.film);
  EXPECT_FALSE(dark.valid);
  // nor is there a film to speak of where it lies on the surface (the white of the eye), or none at all:
  // the light sample stands as it was
  const float3 above = lightFrom(20.0f);
  EXPECT_FALSE(eyeCausticSolve(flat, above, float3(0.0f, 0.0f, 1.0f), float4(above, 0.0f), 3.0f).film);
  CapFilm none = kEye;
  none.point   = float3(0.0f, 0.0f, 30.0f);  // outside the sphere
  EXPECT_FALSE(eyeCausticSolve(none, above, float3(0.0f, 0.0f, 1.0f), float4(above, 0.0f), 0.0f).film);
}

// On its axis a curved film is a lens: one refracting surface of radius R focuses at n R / (n - 1)
// behind it, so a parallel beam is (f / (f - s))^2 brighter a depth s in
TEST(EyeCaustic, ACurvedFilmFocusesAsALensOfItsRadius)
{
  const float f = kIor * kEye.radius / (kIor - 1.0f);
  for(float depth : {1.0f, 2.6f, 5.0f})
  {
    CapFilm film = kEye;
    film.apex    = depth;
    const float got = solved(film, float3(0.0f), float3(0.0f, 0.0f, 1.0f));
    const float lens = (f / (f - depth)) * (f / (f - depth));
    EXPECT_NEAR(got, lens, 0.02f * lens) << "depth " << depth;
  }
}

// A lamp far off is the same light as a direction
TEST(EyeCaustic, AFarLampIsADirection)
{
  CapFilm      film    = kEye;
  film.point           = float3(2.0f, 1.0f, 0.0f);
  const float3 toLight = lightFrom(50.0f);
  const float3     up   = float3(0.0f, 0.0f, 1.0f);
  const EyeCaustic far  = eyeCausticSolve(film, toLight, up, float4(toLight, 0.0f), 0.0f);
  const EyeCaustic lamp = eyeCausticSolve(film, toLight, up, float4(toLight * 3000.0f, 1.0f), 0.0f);
  ASSERT_TRUE(far.valid && lamp.valid);
  EXPECT_NEAR(dot(far.arrival, lamp.arrival), 1.0f, 1e-4f);
  EXPECT_NEAR(lamp.focus, far.focus, 0.02f * far.focus);
}

// Over the whole iris, the light the walk puts at a point is the light a beam traced through the film
// from the light's side lands there
TEST(EyeCaustic, TheWalkPutsTheLightWhereABeamThroughTheFilmLandsIt)
{
  const float half = 6.0f, cell = 0.5f;
  for(float degrees : {0.0f, 30.0f, 60.0f, 75.0f, 85.0f, 95.0f})
  {
    const float3 toLight = lightFrom(degrees);
    const Landed beam    = traced(kEye, toLight, half, cell, 3000);
    int          cells = 0, close = 0, missed = 0;
    double       sumBeam = 0, sumWalk = 0, straight = 0;
    for(float y = -half + cell / 2; y < half; y += cell)
      for(float x = -half + cell / 2; x < half; x += cell)
      {
        if(x * x + y * y > 5.5f * 5.5f)
          continue;
        const float b = beam.at(x, y);
        const float w = solved(kEye, float3(x, y, 0.0f), toLight);
        cells++;
        sumBeam += b;
        sumWalk += w;
        straight += std::max(toLight.z, 0.0f);
        close += std::abs(w - b) <= 0.08f * std::max(b, 0.05f) + 0.01f;
        missed += (w == 0.0f && b > 0.05f);
      }
    std::printf("[eye] light at %2.0f deg: beam %.3f, walk %.3f, straight %.3f of the light square to it; %d of %d cells within 8%%, %d missed\n",
                degrees, sumBeam / cells, sumWalk / cells, straight / cells, close, cells, missed);
    // All of it up to 30 degrees. Past that the walk gives up on a band at the edge of the cap's shadow,
    // where the ray leaves at the critical angle: a tenth of the cells, and a tenth of the light at most.
    // It never invents light.
    EXPECT_GE(close, degrees <= 30.0f ? cells : cells * 8 / 10) << degrees;
    EXPECT_LE(missed, degrees <= 30.0f ? 0 : cells / 10) << degrees;
    EXPECT_LE(sumWalk, 1.02 * sumBeam) << degrees;
    EXPECT_GE(sumWalk, 0.88 * sumBeam) << degrees;
    // and at a low light the film is most of what the iris gets: the crescent
    if(degrees >= 75.0f)
      EXPECT_GT(sumWalk, 1.3 * straight) << degrees;
  }
}

// One row across the iris, the light on the +x side: for a look at the numbers (run with
// --gtest_also_run_disabled_tests)
TEST(EyeCaustic, DISABLED_ARowAcrossTheIris)
{
  const float half = 6.0f, cell = 0.5f;
  for(float degrees : {75.0f, 85.0f, 95.0f})
  {
    const float3 toLight = lightFrom(degrees);
    const Landed beam    = traced(kEye, toLight, half, cell, 6000);
    std::printf("[eye] %2.0f deg, y = 0.25: x, beam, walk\n", degrees);
    for(float x = -5.25f; x < 5.5f; x += 0.5f)
      std::printf("   %+5.2f  %6.3f  %6.3f\n", x, beam.at(x, 0.25f), solved(kEye, float3(x, 0.25f, 0.0f), toLight));
  }
}
