// Body maps (wardrobe plan step 3a): regions from a body's own skin, outer skin by surface, and the
// projection one body onto another within its regions, facing its way and within reach. Synthetic
// bodies only (the plan's rule 7); the real Genesis 8 Female onto Genesis 9 is fv_parity's bodymap step.

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <figure/bodymap/clone.hpp>
#include <figure/bodymap/regions.hpp>
#include <figure/wardrobe/fit/bases.hpp>

namespace {

using fv::bodymap::Body;
using fv::bodymap::RegionMask;
using fv::rig::Region;

// A sphere of `rings` x `segments`, its triangles facing out, every vertex in `region`.
Body sphere(double radius, RegionMask region = 0, int rings = 16, int segments = 24, double cx = 0, double cy = 0, double cz = 0)
{
  Body b;
  const auto add = [&](double x, double y, double z) {
    b.positions.insert(b.positions.end(), {float(cx + x), float(cy + y), float(cz + z)});
  };
  add(0, radius, 0);
  for(int r = 1; r < rings; r++)
  {
    const double theta = std::numbers::pi * r / rings;
    for(int s = 0; s < segments; s++)
    {
      const double phi = 2 * std::numbers::pi * s / segments;
      add(radius * std::sin(theta) * std::cos(phi), radius * std::cos(theta), radius * std::sin(theta) * std::sin(phi));
    }
  }
  add(0, -radius, 0);
  const auto     ring   = [&](int r, int s) { return uint32_t(1 + (r - 1) * segments + (s % segments)); };
  const uint32_t bottom = uint32_t(b.positions.size() / 3 - 1);
  for(int s = 0; s < segments; s++)
    b.tris.insert(b.tris.end(), {0u, ring(1, s + 1), ring(1, s)});
  for(int r = 1; r + 1 < rings; r++)
    for(int s = 0; s < segments; s++)
    {
      b.tris.insert(b.tris.end(), {ring(r, s), ring(r, s + 1), ring(r + 1, s)});
      b.tris.insert(b.tris.end(), {ring(r, s + 1), ring(r + 1, s + 1), ring(r + 1, s)});
    }
  for(int s = 0; s < segments; s++)
    b.tris.insert(b.tris.end(), {ring(rings - 1, s), ring(rings - 1, s + 1), bottom});
  b.outer.assign(b.tris.size() / 3, 1);
  b.regions.assign(b.positions.size() / 3, region);
  return b;
}

// A square plate of n x n cells in the xz plane at height y, facing up (or down), in `region`.
Body plate(double y, RegionMask region, bool up = true, int n = 8, double size = 20)
{
  Body b;
  for(int i = 0; i <= n; i++)
    for(int j = 0; j <= n; j++)
      b.positions.insert(b.positions.end(), {float(-size / 2 + size * i / n), float(y), float(-size / 2 + size * j / n)});
  const auto at = [&](int i, int j) { return uint32_t(i * (n + 1) + j); };
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
    {
      // facing +y: (i,j) -> (i,j+1) -> (i+1,j) turns from z toward x, whose cross is +y
      if(up)
      {
        b.tris.insert(b.tris.end(), {at(i, j), at(i, j + 1), at(i + 1, j)});
        b.tris.insert(b.tris.end(), {at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)});
      }
      else
      {
        b.tris.insert(b.tris.end(), {at(i, j), at(i + 1, j), at(i, j + 1)});
        b.tris.insert(b.tris.end(), {at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)});
      }
    }
  b.outer.assign(b.tris.size() / 3, 1);
  b.regions.assign(b.positions.size() / 3, region);
  return b;
}

// Two bodies as one: b's vertices and triangles after a's.
Body joined(const Body& a, const Body& b)
{
  Body out = a;
  const uint32_t base = uint32_t(a.positions.size() / 3);
  out.positions.insert(out.positions.end(), b.positions.begin(), b.positions.end());
  for(const uint32_t t : b.tris)
    out.tris.push_back(t + base);
  out.outer.insert(out.outer.end(), b.outer.begin(), b.outer.end());
  out.regions.insert(out.regions.end(), b.regions.begin(), b.regions.end());
  return out;
}

// the vertices of `onto` a projection put `from`'s vertex v between, with any weight
std::vector<uint32_t> cornersOf(const fv::bodymap::Projection& p, size_t v)
{
  std::vector<uint32_t> out;
  for(size_t j = 0; j < 3; j++)
    if(p.at.weights[v * 3 + j] > 1e-6f)
      out.push_back(p.at.verts[v * 3 + j]);
  return out;
}

const RegionMask kLeftThigh  = fv::bodymap::regionBit(Region::eThigh, 'l');
const RegionMask kRightThigh = fv::bodymap::regionBit(Region::eThigh, 'r');

}  // namespace

// A vertex's regions are its strongest bone's, and on a border its neighbour's too; a bone the spec
// doesn't know (a face bone) takes its parent's region, and left and right are apart.
TEST(BodyMap, AVertexsRegionsComeFromItsOwnSkin)
{
  const std::vector<fv::wardrobe::SkelBone> bones{{"hip", std::nullopt, {}},
                                                  {"pelvis", "hip", {}},
                                                  {"lThighBend", "pelvis", {}},
                                                  {"rThighBend", "pelvis", {}},
                                                  {"head", "hip", {}},
                                                  {"lEyelidUpper", "head", {}},
                                                  {"skirt01", std::nullopt, {}}};
  fv::mesh::SkinInput skin;
  skin.start = {0, 1, 3, 5, 6, 7};
  skin.bone   = {2, 2, 1, 2, 1, 5, 6};
  skin.weight = {1.f, 0.6f, 0.4f, 0.9f, 0.1f, 1.f, 1.f};
  const std::vector<RegionMask> regions = fv::bodymap::vertexRegions(skin, bones, 5);
  const RegionMask              pelvis  = fv::bodymap::regionBit(Region::ePelvis);
  EXPECT_EQ(regions[0], kLeftThigh);
  EXPECT_EQ(regions[1], kLeftThigh | pelvis);
  EXPECT_EQ(regions[2], kLeftThigh);
  EXPECT_EQ(regions[3], fv::bodymap::regionBit(Region::eHead));
  EXPECT_EQ(regions[4], 0u);
  EXPECT_NE(kLeftThigh, kRightThigh);
  EXPECT_EQ(fv::bodymap::regionBit(Region::eNone, 'l'), 0u);
}

// The eyes, lashes, mouth, teeth and tongue aren't outer skin; the lips and nails are.
TEST(BodyMap, OuterSkinIsKnownBySurface)
{
  for(const char* inner : {"Teeth", "Mouth", "Tongue", "EyeMoisture", "EyeReflection", "Lacrimals", "Cornea", "Irises", "Pupils", "Sclera", "Eyelashes", "Mouth Cavity", "Tear"})
    EXPECT_FALSE(fv::bodymap::outerSkin(inner)) << inner;
  for(const char* outer : {"Torso", "Face", "Lips", "Ears", "Arms", "Legs", "Fingernails", "Toenails", "Body", "Head", "EyeSocket"})
    EXPECT_TRUE(fv::bodymap::outerSkin(outer)) << outer;

  fv::wardrobe::BaseCage cage;
  cage.positions  = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
  cage.tris       = {0, 1, 2, 1, 3, 2};
  cage.surfaces   = {"Torso", "Teeth"};
  cage.triSurface = {0, 1};
  cage.skin.start.assign(5, 0);
  const Body b = fv::bodymap::bodyOf(cage);
  EXPECT_EQ(b.outer, (std::vector<uint8_t>{1, 0}));
  EXPECT_EQ(b.regions.size(), 4u);
}

// A body mapped onto itself is the identity: every vertex lands on itself, and the round trip,
// stretch and folds are nothing.
TEST(BodyMap, ABodyMappedOntoItselfIsTheIdentity)
{
  const Body                    b   = sphere(10, kLeftThigh);
  const fv::bodymap::BodyMap    map = fv::bodymap::mapBodies(b, b);
  const fv::bodymap::MapQuality q   = fv::bodymap::mapQuality(b, b, map);
  for(size_t v = 0; v < b.positions.size() / 3; v++)
  {
    ASSERT_TRUE(map.toReference.mapped[v]) << v;
    const std::vector<uint32_t> corners = cornersOf(map.toReference, v);
    ASSERT_EQ(corners.size(), 1u) << v;
    EXPECT_EQ(corners[0], v);
  }
  EXPECT_EQ(q.mapped, 1);
  EXPECT_EQ(q.inverseCovers, 1);
  EXPECT_LT(q.roundTrip.at("all").max, 1e-4);
  EXPECT_LT(q.distance.at("all").max, 1e-4);
  EXPECT_NEAR(q.stretchP95, 1, 1e-4);
  EXPECT_EQ(q.foldFree, 1);
}

// A sphere onto one 3% larger, of another tessellation: every vertex lands 0.3 cm out, the way back
// returns it within 2 mm, the map stretches by about the 3% and turns nothing over.
TEST(BodyMap, ABodyOntoALargerOneLandsJustOutsideAndComesBack)
{
  const Body                    source    = sphere(10, kLeftThigh, 16, 24);
  const Body                    reference = sphere(10.3, kLeftThigh, 20, 30);
  const fv::bodymap::BodyMap    map       = fv::bodymap::mapBodies(source, reference);
  const fv::bodymap::MapQuality q         = fv::bodymap::mapQuality(source, reference, map);
  EXPECT_EQ(q.mapped, 1);
  EXPECT_EQ(q.inverseCovers, 1);
  EXPECT_NEAR(q.distance.at("all").median, 0.3, 0.05);
  EXPECT_LT(q.roundTrip.at("all").p95, 0.2);
  EXPECT_GT(q.stretchP95, 1.0);
  EXPECT_LT(q.stretchP95, 1.1);
  EXPECT_EQ(q.foldFree, 1);
}

// A vertex matches only within its regions, only a triangle facing its way, only outer skin, and
// only within reach; with no region it matches anywhere.
TEST(BodyMap, AVertexMatchesWithinItsRegionsFacingItsWayOnSkinWithinReach)
{
  // the right thigh's plate is nearer, the left's further: the left thigh's vertex goes to its own
  const Body near  = plate(1, kRightThigh);
  const Body far   = plate(-1, kLeftThigh);
  const Body onto  = joined(near, far);
  const Body point = plate(0.2, kLeftThigh, true, 1, 2);
  const auto first = [](const fv::bodymap::Projection& p) { return p.at.verts[0]; };
  const fv::bodymap::Projection left = fv::bodymap::project(point, onto);
  ASSERT_TRUE(left.mapped[0]);
  EXPECT_GE(first(left), near.positions.size() / 3);
  // with no region, the nearest
  Body anyRegion = point;
  anyRegion.regions.assign(anyRegion.regions.size(), 0);
  const fv::bodymap::Projection any = fv::bodymap::project(anyRegion, onto);
  ASSERT_TRUE(any.mapped[0]);
  EXPECT_LT(first(any), near.positions.size() / 3);

  // the nearer plate faces away: skipped
  const Body facing = joined(plate(1, kLeftThigh, false), plate(-1, kLeftThigh, true));
  const fv::bodymap::Projection turned = fv::bodymap::project(point, facing);
  ASSERT_TRUE(turned.mapped[0]);
  EXPECT_GE(first(turned), plate(1, kLeftThigh).positions.size() / 3);

  // the nearer plate is teeth: skipped
  Body teeth = joined(plate(1, kLeftThigh), plate(-1, kLeftThigh));
  std::fill(teeth.outer.begin(), teeth.outer.begin() + std::ptrdiff_t(teeth.outer.size() / 2), uint8_t(0));
  const fv::bodymap::Projection skin = fv::bodymap::project(point, teeth);
  ASSERT_TRUE(skin.mapped[0]);
  EXPECT_GE(first(skin), plate(1, kLeftThigh).positions.size() / 3);

  // nothing within reach: unmapped
  EXPECT_FALSE(fv::bodymap::project(plate(20, kLeftThigh, true, 1, 2), far).mapped[0]);
  // a vertex on no outer skin isn't mapped at all
  Body hidden = point;
  hidden.outer.assign(hidden.outer.size(), 0);
  EXPECT_FALSE(fv::bodymap::project(hidden, onto).mapped[0]);
}

// The clone morph each Genesis type has on Genesis 9; none for Genesis 9 itself or Skyrim's.
TEST(BodyMap, EachGenesisTypeHasItsCloneOnGenesis9)
{
  using G = fv::wardrobe::Generation;
  EXPECT_EQ(fv::bodymap::cloneFile(G::eG8, "female"), "data/DAZ 3D/Genesis 9/Base/Morphs/Daz 3D/Clones/CloneGenesis8Female.dsf");
  EXPECT_EQ(fv::bodymap::cloneFile(G::eG81, "male"), "data/DAZ 3D/Genesis 9/Base/Morphs/Daz 3D/Clones/CloneGenesis8Male.dsf");
  EXPECT_EQ(fv::bodymap::cloneFile(G::eG3, ""), "data/DAZ 3D/Genesis 9/Base/Morphs/Daz 3D/Clones/CloneGenesis3Female.dsf");
  EXPECT_EQ(fv::bodymap::cloneFile(G::eG2, "male"), "data/DAZ 3D/Genesis 9/Base/Morphs/Daz 3D/Clones/CloneGenesis2Male.dsf");
  EXPECT_EQ(fv::bodymap::cloneFile(G::eG9, "female"), "");
  EXPECT_EQ(fv::bodymap::cloneFile(G::eSkyrim, "female"), "");
}

// The reference turned into the source's rest: a leg whose source is bent 10 degrees at the knee is
// turned 10 degrees at the knee, the shin and the foot below it, and lands on the source; the thigh
// stays. Each joint is found on each body as the band where two regions meet, from its own weights.
TEST(BodyMap, TheReferenceIsTurnedIntoTheSourcesRestAtItsLimbJoints)
{
  // hip, pelvis, thigh, shin, foot and big toe bones; a vertex on each band between them and one inside
  // the thigh, shin and foot
  fv::wardrobe::BaseCage reference;
  reference.skeleton = {{"hip", std::nullopt, {0, 100, 0}},     {"pelvis", "hip", {0, 100, 0}}, {"lThighBend", "pelvis", {10, 100, 0}},
                        {"lShin", "lThighBend", {10, 55, 0}}, {"lFoot", "lShin", {10, 10, 0}}, {"lBigToe", "lFoot", {20, 8, 0}}};
  reference.positions   = {10, 100, 0, 12, 80, 0, 10, 55, 0, 12, 40, 0, 12, 20, 0, 10, 10, 0, 14, 8, 0, 20, 8, 0};
  reference.skin.start  = {0, 2, 3, 5, 6, 7, 9, 10, 12};
  reference.skin.bone   = {1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5};
  reference.skin.weight = {0.5f, 0.5f, 1, 0.5f, 0.5f, 1, 1, 0.5f, 0.5f, 1, 0.5f, 0.5f};

  // the source: the same leg bent 10 degrees at the knee, everything below it turned toward +x (the foot
  // in the plane of the bend, so the shortest turn of its direction is the bend)
  const double kTilt = 10 * std::numbers::pi / 180;
  Body         source;
  source.positions = reference.positions;
  for(size_t v = 3; v < 8; v++)
  {
    const double dx = reference.positions[v * 3] - 10, dy = reference.positions[v * 3 + 1] - 55;
    source.positions[v * 3]     = float(10 + dx * std::cos(kTilt) - dy * std::sin(kTilt));
    source.positions[v * 3 + 1] = float(55 + dx * std::sin(kTilt) + dy * std::cos(kTilt));
  }
  source.regions = fv::bodymap::vertexRegions(reference.skin, reference.skeleton, 8);

  std::vector<fv::bodymap::LimbTurn> turns;
  const std::vector<float>           posed  = fv::bodymap::intoRest(reference, reference.positions, source, &turns);
  const auto                         turnOf = [&](const char* joint) {
    for(const auto& t : turns)
      if(t.joint == joint && t.side == 'l')
        return t.degrees;
    return -1.0;
  };
  EXPECT_NEAR(turnOf("thigh"), 0, 1e-4);
  EXPECT_NEAR(turnOf("shin"), 10, 1e-4);
  EXPECT_NEAR(turnOf("foot"), 10, 1e-4);
  for(size_t i = 0; i < posed.size(); i++)
    EXPECT_NEAR(posed[i], source.positions[i], 1e-4) << i;

  // the source as the reference is: nothing turns
  Body same;
  same.positions = reference.positions;
  same.regions   = source.regions;
  const std::vector<float> unmoved = fv::bodymap::intoRest(reference, reference.positions, same);
  for(size_t i = 0; i < unmoved.size(); i++)
    EXPECT_NEAR(unmoved[i], reference.positions[i], 1e-5) << i;
}

// A limb part with nothing below it to aim at is still turned into the source's rest: its own vertices
// fitted to the source's skin, by the turn about its joint that takes them there best. A tube leg bent
// 10 degrees at the knee, with no foot.
TEST(BodyMap, ALimbIsFittedToTheSourcesSkinAboutItsJoint)
{
  constexpr int kRing = 12;
  const double  kTilt = 10 * std::numbers::pi / 180;
  const auto    leg   = [&](bool bent, std::vector<RegionMask>* masks) {
    Body b;
    for(int r = 0; r <= 18; r++)
    {
      const double y = 100 - 5 * r;
      for(int k = 0; k < kRing; k++)
      {
        const double a = 2 * std::numbers::pi * k / kRing;
        double       x = 10 + 5 * std::cos(a), z = 5 * std::sin(a), py = y;
        if(bent && y < 55)
        {
          const double dx = x - 10, dy = y - 55;
          x  = 10 + dx * std::cos(kTilt) - dy * std::sin(kTilt);
          py = 55 + dx * std::sin(kTilt) + dy * std::cos(kTilt);
        }
        b.positions.insert(b.positions.end(), {float(x), float(py), float(z)});
      }
    }
    for(int r = 0; r < 18; r++)
      for(int k = 0; k < kRing; k++)
      {
        const uint32_t a = uint32_t(r * kRing + k), c = uint32_t(r * kRing + (k + 1) % kRing), d = a + kRing, e = c + kRing;
        b.tris.insert(b.tris.end(), {a, d, c, c, d, e});
      }
    b.outer.assign(b.tris.size() / 3, 1);
    if(masks)
      b.regions = *masks;
    return b;
  };
  // the skin: the top ring between pelvis and thigh, the knee's between thigh and shin
  fv::wardrobe::BaseCage reference;
  reference.skeleton = {{"hip", std::nullopt, {10, 100, 0}}, {"pelvis", "hip", {10, 100, 0}}, {"lThighBend", "pelvis", {10, 100, 0}}, {"lShin", "lThighBend", {10, 55, 0}}};
  const Body straight = leg(false, nullptr);
  reference.positions = straight.positions;
  reference.skin.start.push_back(0);
  for(int r = 0; r <= 18; r++)
    for(int k = 0; k < kRing; k++)
    {
      const double y = 100 - 5 * r;
      if(r == 0)
        reference.skin.bone.insert(reference.skin.bone.end(), {1, 2}), reference.skin.weight.insert(reference.skin.weight.end(), {0.5f, 0.5f});
      else if(y == 55)
        reference.skin.bone.insert(reference.skin.bone.end(), {2, 3}), reference.skin.weight.insert(reference.skin.weight.end(), {0.5f, 0.5f});
      else
        reference.skin.bone.push_back(y > 55 ? 2 : 3), reference.skin.weight.push_back(1);
      reference.skin.start.push_back(int32_t(reference.skin.bone.size()));
    }
  std::vector<RegionMask> masks = fv::bodymap::vertexRegions(reference.skin, reference.skeleton, straight.positions.size() / 3);
  const Body              source = leg(true, &masks);

  std::vector<fv::bodymap::LimbTurn> turns;
  const std::vector<float>           posed = fv::bodymap::intoRest(reference, reference.positions, source, &turns);
  double                             shin  = -1;
  for(const auto& t : turns)
    if(t.joint == "shin" && t.side == 'l')
      shin = t.degrees;
  EXPECT_NEAR(shin, 10, 0.5);
  // the bottom ring lands on the source's
  for(size_t v = 18 * kRing; v < 19 * kRing; v++)
    for(size_t c = 0; c < 3; c++)
      EXPECT_NEAR(posed[v * 3 + c], source.positions[v * 3 + c], 0.3) << v << " " << c;
}
