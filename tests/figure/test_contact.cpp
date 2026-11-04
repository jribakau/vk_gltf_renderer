// Body contact (figure/physics/contact/contact.hpp) and the physics core it is the first user of: a
// body pressed into a prop ends on its surface, the volume pushed in comes back around the dent along
// the skin and not across a gap, what is inside an open or a closed prop, the channel's life (off,
// playback, a pass that changes nothing), and the pieces under it: the surface index against brute
// force, the volume constraint, the still pass's lifecycle and the channel writer.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <future>
#include <random>
#include <thread>

#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/physics/collide/surface_index.hpp>
#include <figure/physics/collide/triangle_set.hpp>
#include <figure/physics/contact/contact.hpp>
#include <figure/physics/pass.hpp>
#include <figure/physics/volume.hpp>
#include <figure/physics/write.hpp>

namespace {

using namespace fv;
using namespace fv::physics;

constexpr double kPi = 3.14159265358979323846;

struct Soup
{
  std::vector<float>    positions;
  std::vector<uint32_t> tris;
};

// Triangles turned so each faces away from `centre`: a convex shape's faces point out.
void outward(Soup& s, std::array<double, 3> centre)
{
  for(size_t t = 0; t + 2 < s.tris.size(); t += 3)
  {
    const float* a = &s.positions[s.tris[t] * 3];
    const float* b = &s.positions[s.tris[t + 1] * 3];
    const float* c = &s.positions[s.tris[t + 2] * 3];
    const double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const double vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const double mx = (a[0] + b[0] + c[0]) / 3 - centre[0], my = (a[1] + b[1] + c[1]) / 3 - centre[1], mz = (a[2] + b[2] + c[2]) / 3 - centre[2];
    if(nx * mx + ny * my + nz * mz < 0)
      std::swap(s.tris[t + 1], s.tris[t + 2]);
  }
}

// A closed box, eight shared corners, faces out.
Soup box(std::array<float, 3> lo, std::array<float, 3> hi)
{
  Soup s;
  for(int k = 0; k < 2; k++)
    for(int j = 0; j < 2; j++)
      for(int i = 0; i < 2; i++)
        s.positions.insert(s.positions.end(), {i ? hi[0] : lo[0], j ? hi[1] : lo[1], k ? hi[2] : lo[2]});
  const auto at = [](int i, int j, int k) { return uint32_t(i + 2 * j + 4 * k); };
  const auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) { s.tris.insert(s.tris.end(), {a, b, c, a, c, d}); };
  quad(at(0, 0, 0), at(0, 1, 0), at(0, 1, 1), at(0, 0, 1));
  quad(at(1, 0, 0), at(1, 1, 0), at(1, 1, 1), at(1, 0, 1));
  quad(at(0, 0, 0), at(1, 0, 0), at(1, 0, 1), at(0, 0, 1));
  quad(at(0, 1, 0), at(1, 1, 0), at(1, 1, 1), at(0, 1, 1));
  quad(at(0, 0, 0), at(1, 0, 0), at(1, 1, 0), at(0, 1, 0));
  quad(at(0, 0, 1), at(1, 0, 1), at(1, 1, 1), at(0, 1, 1));
  outward(s, {(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0});
  return s;
}

// A sheet along a path in xy (`along(u)` gives x, y), z from -halfWidth to halfWidth, `step` apart;
// its faces point to +y where the path runs along +x.
Soup sheet(double length, double halfWidth, double step, const std::function<std::array<double, 2>(double)>& along)
{
  Soup      s;
  const int nu = int(std::lround(length / step)), nz = int(std::lround(2 * halfWidth / step));
  for(int i = 0; i <= nu; i++)
    for(int k = 0; k <= nz; k++)
    {
      const auto p = along(i * step);
      s.positions.insert(s.positions.end(), {float(p[0]), float(p[1]), float(-halfWidth + k * step)});
    }
  const auto at = [&](int i, int k) { return uint32_t(i * (nz + 1) + k); };
  for(int i = 0; i < nu; i++)
    for(int k = 0; k < nz; k++)
      s.tris.insert(s.tris.end(), {at(i, k), at(i, k + 1), at(i + 1, k + 1), at(i, k), at(i + 1, k + 1), at(i + 1, k)});
  return s;
}

// A flat body at height y over [-half, half] in x and z, facing up.
Soup flat(double half, double step, float y = 0)
{
  return sheet(2 * half, half, step, [&](double u) { return std::array<double, 2>{u - half, y}; });
}

// A closed sphere, poles shared, faces out.
Soup sphere(double r, int rings, int segments)
{
  Soup s;
  s.positions.insert(s.positions.end(), {0.f, float(r), 0.f});
  for(int i = 1; i < rings; i++)
  {
    const double th = kPi * i / rings;
    for(int j = 0; j < segments; j++)
    {
      const double ph = 2 * kPi * j / segments;
      s.positions.insert(s.positions.end(), {float(r * std::sin(th) * std::cos(ph)), float(r * std::cos(th)), float(r * std::sin(th) * std::sin(ph))});
    }
  }
  s.positions.insert(s.positions.end(), {0.f, float(-r), 0.f});
  const uint32_t south = uint32_t(s.positions.size() / 3 - 1);
  const auto     at    = [&](int ring, int j) { return uint32_t(1 + (ring - 1) * segments + (j % segments)); };
  for(int j = 0; j < segments; j++)
  {
    s.tris.insert(s.tris.end(), {0, at(1, j), at(1, j + 1)});
    s.tris.insert(s.tris.end(), {south, at(rings - 1, j + 1), at(rings - 1, j)});
  }
  for(int i = 1; i + 1 < rings; i++)
    for(int j = 0; j < segments; j++)
      s.tris.insert(s.tris.end(), {at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j), at(i + 1, j + 1), at(i, j + 1)});
  outward(s, {0, 0, 0});
  return s;
}

// The volume a closed mesh encloses.
double enclosed(std::span<const float> p, std::span<const uint32_t> tris)
{
  double v = 0;
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
  {
    const float* a = &p[tris[t] * 3];
    const float* b = &p[tris[t + 1] * 3];
    const float* c = &p[tris[t + 2] * 3];
    v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
  }
  return v;
}

std::shared_ptr<const SharedTriangles> shared(const Soup& s)
{
  return std::make_shared<const SharedTriangles>(std::make_shared<const TriangleSet>(s.positions, s.tris));
}

ContactResult resolve(const Soup& body, const std::vector<ColliderRef>& colliders, float bulge = 0.7f, float reach = 3)
{
  ContactInput in;
  in.body      = body.positions;
  in.tris      = body.tris;
  in.colliders = colliders;
  in.bulge     = bulge;
  in.reach     = reach;
  return resolveContacts(in);
}

Mat4 turnedScaled(double degrees, double scale, Vec3 move)
{
  const double c = std::cos(degrees * kPi / 180), s = std::sin(degrees * kPi / 180);
  Mat4         m;
  m.e = {c * scale, 0, -s * scale, 0, 0, scale, 0, 0, s * scale, 0, c * scale, 0, move.x, move.y, move.z, 1};
  return m;
}

Soup carried(const Soup& s, const Mat4& m)
{
  Soup out = s;
  for(size_t i = 0; i + 2 < out.positions.size(); i += 3)
  {
    const Vec3 q         = m.transformPoint({s.positions[i], s.positions[i + 1], s.positions[i + 2]});
    out.positions[i]     = float(q.x);
    out.positions[i + 1] = float(q.y);
    out.positions[i + 2] = float(q.z);
  }
  return out;
}

}  // namespace

// Pressed 1 cm into a box from below, the skin under it ends on the box's face: no deeper in than a
// hundredth of a millimetre, no further off than the clearance. The skin around it rises.
TEST(Contact, aBodyPressedOnABoxEndsOnItsSurface)
{
  const Soup body = flat(5, 0.25);
  const Soup prop = box({-1.1f, -1.f, -1.1f}, {1.1f, 4.f, 1.1f});
  const auto r    = resolve(body, {ColliderRef{shared(prop), Mat4{}, 1}});
  EXPECT_EQ(r.patches, 1);
  float highest = 0;
  for(size_t v = 0; v < body.positions.size() / 3; v++)
  {
    const float x = body.positions[v * 3], z = body.positions[v * 3 + 2], y = r.positions[v * 3 + 1];
    if(std::fabs(x) < 1.05f && std::fabs(z) < 1.05f)
    {
      EXPECT_LE(y, -1.f + 1e-3f) << "at " << x << ", " << z;
      EXPECT_GE(y, -1.f - 0.05f - 1e-3f) << "at " << x << ", " << z;
    }
    else
      highest = std::max(highest, y);
  }
  EXPECT_GT(highest, 0.05f);
}

// Near the box's side the skin goes back into the body, along its own normal, not out through the side.
TEST(Contact, theDentGoesBackAlongTheSkinNotOutThroughTheSide)
{
  const Soup body = flat(5, 0.25);
  const Soup prop = box({-1.1f, -1.f, -1.1f}, {1.1f, 4.f, 1.1f});
  const auto r    = resolve(body, {ColliderRef{shared(prop), Mat4{}, 1}}, 0.f);
  for(size_t v = 0; v < body.positions.size() / 3; v++)
    if(std::fabs(body.positions[v * 3] - 1.f) < 1e-4f && std::fabs(body.positions[v * 3 + 2]) < 1e-4f)
    {
      EXPECT_NEAR(r.positions[v * 3], 1.f, 0.1f);
      EXPECT_LT(r.positions[v * 3 + 1], -1.f);
      return;
    }
  FAIL() << "no vertex at (1, 0, 0)";
}

// At a bulge of 1 a sphere pressed by a box keeps its volume; at 0 it loses about the cap the box cut off.
TEST(Contact, theBulgeHoldsThePushedInVolume)
{
  const Soup   ball = sphere(10, 48, 96);
  const Soup   prop = box({-6.f, 8.5f, -6.f}, {6.f, 20.f, 6.f});
  const double v0   = enclosed(ball.positions, ball.tris);
  const auto   full = resolve(ball, {ColliderRef{shared(prop), Mat4{}, 1}}, 1.f);
  const auto   none = resolve(ball, {ColliderRef{shared(prop), Mat4{}, 1}}, 0.f);
  const double lost = v0 - enclosed(none.positions, ball.tris);
  const double kept = v0 - enclosed(full.positions, ball.tris);
  // the cap above y = 8.5 on a 10 cm ball: pi h^2 (3r - h) / 3
  const double cap = kPi * 1.5 * 1.5 * (30 - 1.5) / 3;
  // measured 2026-11-04: lost 76.6 cm3 against the cap's 67.2 (the dent's sloped rim), kept 1.1
  EXPECT_GT(lost, 0.8 * cap);
  EXPECT_LT(std::fabs(kept), 0.05 * lost) << "lost " << lost << " at no bulge, " << kept << " at full";
}

TEST(Contact, noBulgeIsNoVolume)
{
  const Soup   ball = sphere(10, 48, 96);
  const Soup   prop = box({-6.f, 8.5f, -6.f}, {6.f, 20.f, 6.f});
  const auto   none = resolve(ball, {ColliderRef{shared(prop), Mat4{}, 1}}, 0.f);
  const double cap  = kPi * 1.5 * 1.5 * (30 - 1.5) / 3;
  const double lost = enclosed(ball.positions, ball.tris) - enclosed(none.positions, ball.tris);
  EXPECT_NEAR(lost, cap, 0.25 * cap);
  // and nothing rose: a dimple only
  for(size_t v = 0; v < ball.positions.size() / 3; v++)
  {
    const double r0 = std::hypot(ball.positions[v * 3], ball.positions[v * 3 + 1], ball.positions[v * 3 + 2]);
    const double r1 = std::hypot(none.positions[v * 3], none.positions[v * 3 + 1], none.positions[v * 3 + 2]);
    EXPECT_LE(r1, r0 + 1e-3);
  }
}

// A U of skin dented on its lower arm: the bulge spreads along the skin, which runs 30 cm round to the
// upper arm, so the upper arm 3 cm away through the air doesn't move.
TEST(Contact, aBulgeDoesNotCrossAGap)
{
  const Soup u = sheet(43, 3, 0.5, [](double s) {
    if(s <= 20)
      return std::array<double, 2>{s, 0};
    if(s <= 23)
      return std::array<double, 2>{20, s - 20};
    return std::array<double, 2>{20 - (s - 23), 3};
  });
  // the lower arm faces up, into the U
  ASSERT_GT(u.positions.size(), 0u);
  const Soup prop = box({1.f, -0.8f, -1.f}, {4.f, 1.5f, 1.f});
  const auto r    = resolve(u, {ColliderRef{shared(prop), Mat4{}, 1}}, 1.f, 5.f);
  ASSERT_EQ(r.patches, 1);
  bool moved = false;
  for(size_t v = 0; v < u.positions.size() / 3; v++)
  {
    const float* a = &u.positions[v * 3];
    const float* b = &r.positions[v * 3];
    const float  d = std::fabs(a[0] - b[0]) + std::fabs(a[1] - b[1]) + std::fabs(a[2] - b[2]);
    if(a[1] > 2.9f)
      EXPECT_LT(d, 1e-3f) << "upper arm at " << a[0] << ", " << a[2];
    else if(a[1] < 0.1f && d > 0.01f && (a[0] < 1.f || a[0] > 4.f))
      moved = true;
  }
  EXPECT_TRUE(moved) << "the lower arm bulged nowhere";
}

// An open plate: behind its face within its edges is inside; past an edge, or in front, isn't.
TEST(Contact, anOpenPropCountsOnlyBehindItsFaces)
{
  Soup plate;
  plate.positions = {-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1};
  plate.tris      = {0, 2, 1, 0, 3, 2};  // facing up
  const TriangleSet set(plate.positions, plate.tris);
  Stamps&           stamps = threadStamps();
  ASSERT_FALSE(set.closed(0));
  EXPECT_TRUE(insideOf(set, {0, -0.5, 0}, 3, 8, stamps).inside);
  EXPECT_FALSE(insideOf(set, {1.6, -0.3, 0}, 3, 8, stamps).inside);
  EXPECT_FALSE(insideOf(set, {0, 0.5, 0}, 3, 8, stamps).inside);
  EXPECT_FALSE(insideOf(set, {0, -3.5, 0}, 3, 8, stamps).inside) << "past the depth";
}

// A slab 1 cm thick: a point 0.7 cm in is inside, and leaves through the near face.
TEST(Contact, aThinClosedPropUsesParity)
{
  const Soup        slab = box({-5.f, 0.f, -5.f}, {5.f, 1.f, 5.f});
  const TriangleSet set(slab.positions, slab.tris);
  ASSERT_TRUE(set.closed(0));
  const Inside in = insideOf(set, {0, 0.3, 0}, 3, 8, threadStamps());
  ASSERT_TRUE(in.inside);
  EXPECT_NEAR(in.depth, 0.3, 1e-6);
  EXPECT_NEAR(in.exit[1], 0, 1e-6);
  EXPECT_NEAR(in.normal[1], -1, 1e-6);
}

// 4 cm into a closed cube is inside; with its top off, the far side's 3 cm cap leaves it alone.
TEST(Contact, aClosedPropIsInsideDeeperThanAnOpenOne)
{
  Soup cube = box({-5.f, -5.f, -5.f}, {5.f, 5.f, 5.f});
  {
    const TriangleSet set(cube.positions, cube.tris);
    EXPECT_TRUE(insideOf(set, {0, 1, 0}, 3, 8, threadStamps()).inside);
  }
  // the two triangles at y = 5
  Soup open;
  open.positions = cube.positions;
  for(size_t t = 0; t + 2 < cube.tris.size(); t += 3)
    if(!(cube.positions[cube.tris[t] * 3 + 1] > 4 && cube.positions[cube.tris[t + 1] * 3 + 1] > 4 && cube.positions[cube.tris[t + 2] * 3 + 1] > 4))
      open.tris.insert(open.tris.end(), {cube.tris[t], cube.tris[t + 1], cube.tris[t + 2]});
  ASSERT_EQ(open.tris.size(), cube.tris.size() - 6);
  const TriangleSet set(open.positions, open.tris);
  EXPECT_FALSE(set.closed(0));
  EXPECT_FALSE(insideOf(set, {0, 1, 0}, 3, 8, threadStamps()).inside);
}

// Two bodies 1 cm into each other each give way by their share, each meeting the other undented: the
// softer (0.75 against 0.25) takes three quarters and the other the rest, so they meet, with no gap
// past the clearance and no overlap. The rounds after smoothing hold the share; they don't take a
// share of what is left each time.
TEST(Contact, twoBodiesSplitTheOverlapBySoftness)
{
  ContactSettings soft, firm, none;
  soft.softness = 0.75f;
  firm.softness = 0.25f;
  none.softness = 0;
  EXPECT_DOUBLE_EQ(overlapShare(soft, &firm), 0.75);
  EXPECT_DOUBLE_EQ(overlapShare(firm, &soft), 0.25);
  EXPECT_DOUBLE_EQ(overlapShare(none, &none), 0.5);
  ContactSettings off = firm;
  off.on              = false;
  EXPECT_DOUBLE_EQ(overlapShare(soft, &off), 1) << "a body that doesn't dent gives no way";
  EXPECT_DOUBLE_EQ(overlapShare(soft, nullptr), 1);

  // 5 cm balls 9 cm apart: 1 cm in, between x = 4 and 5
  const Soup a  = sphere(5, 48, 96);
  const Soup b  = carried(a, turnedScaled(0, 1, {9, 0, 0}));
  const auto ra = resolve(a, {ColliderRef{shared(b), Mat4{}, 0.75}}, 0.f);
  const auto rb = resolve(b, {ColliderRef{shared(a), Mat4{}, 0.25}}, 0.f);
  const auto at = [](const Soup& s, float x) {
    for(size_t v = 0; v < s.positions.size() / 3; v++)
      if(std::fabs(s.positions[v * 3] - x) < 1e-4f && std::fabs(s.positions[v * 3 + 1]) < 1e-4f && std::fabs(s.positions[v * 3 + 2]) < 1e-4f)
        return v;
    return size_t(-1);
  };
  const size_t va = at(a, 5), vb = at(b, 4);
  ASSERT_NE(va, size_t(-1));
  ASSERT_NE(vb, size_t(-1));
  const float xa = ra.positions[va * 3], xb = rb.positions[vb * 3];
  // each by its share of the overlap and the clearance, 1.05 cm
  EXPECT_NEAR(xa, 5 - 0.75f * 1.05f, 0.02f);
  EXPECT_NEAR(xb, 4 + 0.25f * 1.05f, 0.02f);
  EXPECT_GE(xb - xa, 0.f) << "they still overlap";
  EXPECT_LT(xb - xa, 0.1f);
}

// A prop turned and scaled up is queried in its own space; the answer is the one a copy carried into
// the body's space gives. A prop stretched unevenly is such a copy.
TEST(Physics, aScaledPropsOwnSpaceQueryMatchesATransformedCopy)
{
  const Soup body = flat(5, 0.25);
  const Soup prop = box({-0.6f, -0.5f, -0.6f}, {0.6f, 2.f, 0.6f});
  for(const Mat4& m : {turnedScaled(30, 2, {0.3, 0, -0.2}), [] {
         Mat4 s;
         s.e[0] = 1.5, s.e[5] = 2, s.e[10] = 1;
         return s;
       }()})
  {
    const auto own  = resolve(body, {ColliderRef{shared(prop), m, 1}});
    const auto copy = resolve(body, {ColliderRef{shared(carried(prop, m)), Mat4{}, 1}});
    ASSERT_GT(own.patches, 0);
    EXPECT_EQ(own.patches, copy.patches);
    for(size_t i = 0; i < own.positions.size(); i++)
      ASSERT_NEAR(own.positions[i], copy.positions[i], 2e-3f) << i;
  }
}

// One projection of the volume constraint puts back exactly what it is set to, each particle in
// proportion to its falloff.
TEST(Physics, oneVolumeProjectionPutsBackBetaV)
{
  std::mt19937                           rng(7);
  std::uniform_real_distribution<double> unit(0.1, 1.0);
  DisplacementVolume                     row;
  std::vector<double>                    x;
  std::vector<double>                    falloff;
  for(uint32_t i = 0; i < 50; i++)
  {
    const double area = unit(rng), f = unit(rng);
    double       n[3] = {unit(rng) - 0.5, unit(rng) - 0.5, unit(rng) - 0.5};
    const double len  = std::hypot(n[0], n[1], n[2]);
    row.particles.push_back(i);
    row.weight.push_back(f / area);
    falloff.push_back(f);
    for(int a = 0; a < 3; a++)
    {
      row.an.push_back(area * n[a] / len);
      row.start.push_back(unit(rng));
      x.push_back(row.start.back());
    }
  }
  const double betaV = 3.7;
  row.target         = betaV;
  ASSERT_TRUE(projectVolume(row, x));
  EXPECT_NEAR(volumeError(row, x), 0, 1e-9);
  // the move of each along its normal over its falloff is the same for all
  double ratio = 0;
  for(size_t k = 0; k < row.particles.size(); k++)
  {
    const double area = std::hypot(row.an[k * 3], row.an[k * 3 + 1], row.an[k * 3 + 2]);
    double       h    = 0;
    for(int a = 0; a < 3; a++)
      h += (x[k * 3 + a] - row.start[k * 3 + a]) * row.an[k * 3 + a] / area;
    if(k == 0)
      ratio = h / falloff[k];
    EXPECT_NEAR(h / falloff[k], ratio, 1e-9);
  }
}

// The grid's closest triangle and ray crossings are a brute-force search's, over a random soup.
TEST(Physics, surfaceIndexMatchesBruteForce)
{
  std::mt19937                          rng(11);
  std::uniform_real_distribution<float> at(-10.f, 10.f), off(-1.f, 1.f);
  Soup                                  soup;
  for(uint32_t t = 0; t < 300; t++)
  {
    const float cx = at(rng), cy = at(rng), cz = at(rng);
    for(int k = 0; k < 3; k++)
      soup.positions.insert(soup.positions.end(), {cx + off(rng), cy + off(rng), cz + off(rng)});
    soup.tris.insert(soup.tris.end(), {t * 3, t * 3 + 1, t * 3 + 2});
  }
  const SurfaceIndex          index(soup.positions, soup.tris);
  const std::vector<uint32_t> tri{0, 1, 2};
  Stamps&                     stamps = threadStamps();
  for(int q = 0; q < 200; q++)
  {
    const double p[3] = {at(rng), at(rng), at(rng)};
    double       d[3] = {off(rng), off(rng), off(rng)};
    const double len  = std::hypot(d[0], d[1], d[2]);
    for(double& c : d)
      c /= len;
    double best = INFINITY, nearest = INFINITY;
    int    crossed = 0;
    for(uint32_t t = 0; t < 300; t++)
    {
      const SurfaceIndex one(std::span<const float>(soup.positions).subspan(t * 9, 9), tri);
      best = std::min(best, one.search(p[0], p[1], p[2], 0, {}, INFINITY, stamps.stamp, stamps.next(1)).d2);
      crossed += one.crossings(p, d, [](uint32_t) { return true; }, stamps.stamp, stamps.next(1));
      nearest = std::min(nearest, one.firstHit(p, d, INFINITY, stamps.stamp, stamps.next(1)).t);
    }
    EXPECT_NEAR(index.search(p[0], p[1], p[2], 0, {}, INFINITY, stamps.stamp, stamps.next(index.triangles())).d2, best, 1e-9);
    EXPECT_EQ(index.crossings(p, d, [](uint32_t) { return true; }, stamps.stamp, stamps.next(index.triangles())), crossed);
    const double first = index.firstHit(p, d, INFINITY, stamps.stamp, stamps.next(index.triangles())).t;
    if(std::isinf(nearest))
      EXPECT_TRUE(std::isinf(first));
    else
      EXPECT_NEAR(first, nearest, 1e-9);
  }
}

// An edit while a pass runs lets its result land, then runs it again on what the edit left.
TEST(Physics, aStillPassRunsAgainAfterAnEditDuringIt)
{
  StillPass<int>   pass(std::chrono::milliseconds(0));
  std::promise<int> done;
  int              starts = 0, finished = 0;
  const auto       start  = [&] {
    starts++;
    if(starts == 1)
      return done.get_future();
    std::promise<int> now;
    now.set_value(2);
    return now.get_future();
  };
  const auto finish = [&](const int* r) {
    finished += r ? *r : 0;
    return true;
  };
  pass.poke();
  pass.pump(start, finish);
  ASSERT_EQ(starts, 1);
  pass.poke();  // during it
  done.set_value(1);
  EXPECT_TRUE(pass.pump(start, finish));
  EXPECT_EQ(finished, 1);
  EXPECT_EQ(starts, 2);
  pass.pump(start, finish);
  EXPECT_EQ(finished, 3);
  EXPECT_FALSE(pass.busy());
}

// A cancelled pass's result is dropped when it lands, and nothing runs after it.
TEST(Physics, aCancelledPassDropsWhatLands)
{
  StillPass<int>    pass(std::chrono::milliseconds(0));
  std::promise<int> done;
  int               starts = 0, finished = 0;
  const auto        start  = [&] {
    starts++;
    return done.get_future();
  };
  const auto finish = [&](const int*) {
    finished++;
    return true;
  };
  pass.poke();
  pass.pump(start, finish);
  pass.cancel();
  done.set_value(1);
  EXPECT_FALSE(pass.pump(start, finish));
  EXPECT_EQ(finished, 0);
  EXPECT_EQ(starts, 1);
  EXPECT_FALSE(pass.busy());
}

TEST(Physics, aStillPassWaitsForItsUpstream)
{
  StillPass<int> pass(std::chrono::milliseconds(0));
  bool           upstream = true;
  int            starts   = 0;
  pass.waitWhile          = [&] { return upstream; };
  const auto start        = [&] {
    starts++;
    return std::future<int>();
  };
  const auto finish = [](const int*) { return false; };
  pass.poke();
  pass.pump(start, finish);
  EXPECT_EQ(starts, 0);
  EXPECT_TRUE(pass.busy());
  upstream = false;
  pass.pump(start, finish);
  EXPECT_EQ(starts, 1);
  EXPECT_FALSE(pass.busy());
}

namespace {

// A built mesh of a flat body, and a scene holding it unskinned under the root.
struct Rig
{
  SceneGraph      scene;
  int             root = -1;
  morph::MorphSet morphs{[this](int m) { return scene.meshes[size_t(m)].geo.get(); }};

  explicit Rig(const Soup& body)
  {
    root = scene.add(Node{});
    mesh::MeshInput in;
    in.positions = body.positions;
    in.polyStart.push_back(0);
    for(size_t t = 0; t + 2 < body.tris.size(); t += 3)
    {
      in.polyVerts.insert(in.polyVerts.end(), {int32_t(body.tris[t]), int32_t(body.tris[t + 1]), int32_t(body.tris[t + 2])});
      in.polyStart.push_back(int32_t(in.polyVerts.size()));
      in.polyMaterial.push_back(0);
    }
    Mesh m;
    m.geo            = std::make_shared<mesh::BuiltMesh>(mesh::build(in));
    m.materialGroups = {"Default"};
    m.materials      = {0};
    scene.meshes.push_back(std::move(m));
    scene.materials.push_back({});
    Node n;
    n.kind = NodeKind::eMesh;
    n.mesh = 0;
    scene.add(n, root);
  }
  float minY() const
  {
    const auto& geo = *scene.meshes[0].geo;
    float       y   = 1e9f;
    for(size_t v = 0; v < geo.position.size() / 3; v++)
      y = std::min(y, geo.position[v * 3 + 1]);
    return y;
  }
};

bool pumpUntilWritten(Contact& c)
{
  for(int i = 0; i < 6000; i++)
  {
    if(c.pump())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

}  // namespace

TEST(Contact, turningItOffClearsTheChannel)
{
  Rig     r(flat(3, 0.25));
  Contact c(r.scene, r.morphs, 0, r.root);
  c.colliders = [] { return std::vector<ColliderRef>{{shared(box({-1.1f, -1.f, -1.1f}, {1.1f, 4.f, 1.1f})), Mat4{}, 1}}; };
  ASSERT_TRUE(pumpUntilWritten(c));
  r.morphs.flush();
  EXPECT_EQ(r.morphs.get(contactChannelOf(0)), 1.0);
  EXPECT_EQ(c.patches(), 1);
  EXPECT_LT(r.minY(), -1.f);
  ContactSettings off = c.settings();
  off.on              = false;
  EXPECT_TRUE(c.set(off));
  r.morphs.flush();
  EXPECT_EQ(r.morphs.get(contactChannelOf(0)), 0.0);
  EXPECT_EQ(c.patches(), 0);
  EXPECT_NEAR(r.minY(), 0.f, 1e-5f);
  EXPECT_FALSE(c.busy());
}

// While a clip plays the dent is off; on pause a pass puts it back.
TEST(Contact, playbackHoldsTheChannelAtZero)
{
  Rig     r(flat(3, 0.25));
  Contact c(r.scene, r.morphs, 0, r.root);
  bool    playing = false;
  c.driven        = [&] { return playing; };
  c.colliders     = [] { return std::vector<ColliderRef>{{shared(box({-1.1f, -1.f, -1.1f}, {1.1f, 4.f, 1.1f})), Mat4{}, 1}}; };
  ASSERT_TRUE(pumpUntilWritten(c));
  playing = true;
  EXPECT_TRUE(c.pump());
  EXPECT_EQ(r.morphs.get(contactChannelOf(0)), 0.0);
  EXPECT_FALSE(c.busy());
  c.poke();
  EXPECT_FALSE(c.pump());
  playing = false;
  ASSERT_TRUE(pumpUntilWritten(c));
  EXPECT_EQ(r.morphs.get(contactChannelOf(0)), 1.0);
}

// A pass on what the last one left writes nothing, so the body's vertices aren't flushed again.
TEST(Contact, aPassThatChangesNothingWritesNothing)
{
  Rig     r(flat(3, 0.25));
  Contact c(r.scene, r.morphs, 0, r.root);
  c.colliders = [] { return std::vector<ColliderRef>{{shared(box({-1.1f, -1.f, -1.1f}, {1.1f, 4.f, 1.1f})), Mat4{}, 1}}; };
  ASSERT_TRUE(pumpUntilWritten(c));
  r.morphs.flush();
  r.morphs.takeChanges();
  c.poke();
  // until the pass has landed
  bool wrote = false;
  for(int i = 0; i < 6000 && c.busy(); i++)
  {
    wrote = c.pump() || wrote;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_FALSE(c.busy());
  EXPECT_FALSE(wrote);
  r.morphs.flush();
  EXPECT_TRUE(r.morphs.takeChanges().empty());
}

TEST(Physics, theChannelWriterSkipsAResultThatMatchesTheChannel)
{
  Rig           r(flat(2, 0.5));
  ChannelWriter w{"test:0", "Test", 0};
  const auto&   geo = *r.scene.meshes[0].geo;
  std::vector<float> delta(geo.cagePositions.size(), 0.f);
  delta[1] = -0.5f;
  EXPECT_TRUE(w.write(r.morphs, geo, delta));
  r.morphs.flush();
  r.morphs.takeChanges();
  EXPECT_FALSE(w.write(r.morphs, geo, delta));
  delta[1] -= 0.0005f;
  EXPECT_FALSE(w.write(r.morphs, geo, delta)) << "under kMinMove";
  delta[4] = 0.01f;
  EXPECT_TRUE(w.write(r.morphs, geo, delta));
  r.morphs.takeChanges();
  EXPECT_TRUE(w.off(r.morphs));
  EXPECT_FALSE(w.off(r.morphs));
  // back on with the same deltas: written, at weight 1
  EXPECT_TRUE(w.write(r.morphs, geo, delta));
  EXPECT_EQ(r.morphs.get("test:0"), 1.0);
}
