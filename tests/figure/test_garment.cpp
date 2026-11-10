// Garments as addresses (wardrobe plan step 4a): a cage converted to places on the reference body and
// heights off its skin, decoded onto a body of the reference's topology, and the cloth no address
// holds settled round what one does. Synthetic bodies only (the plan's rule 7); real garments on
// Genesis 9 are fv_parity's clearance step with "fit": "common".

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <numbers>
#include <utility>
#include <vector>

#include <figure/bodymap/clone.hpp>
#include <figure/garment/convert.hpp>
#include <figure/garment/decode.hpp>
#include <figure/garment/maps.hpp>
#include <figure/garment/weights.hpp>

namespace {

using fv::Quat;
using fv::Vec3;
using fv::bodymap::Body;
using fv::bodymap::TypeMap;

// A sphere of `rings` x `segments`, its triangles facing out.
Body sphere(double radius, int rings, int segments)
{
  Body       b;
  const auto add = [&](double x, double y, double z) { b.positions.insert(b.positions.end(), {float(x), float(y), float(z)}); };
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
  b.regions.assign(b.positions.size() / 3, 0);
  return b;
}

// A square plate of n x n cells in the xz plane at y = 0, facing up, `size` across.
Body plate(int n = 8, double size = 20)
{
  Body b;
  for(int i = 0; i <= n; i++)
    for(int j = 0; j <= n; j++)
      b.positions.insert(b.positions.end(), {float(-size / 2 + size * i / n), 0.f, float(-size / 2 + size * j / n)});
  const auto at = [&](int i, int j) { return uint32_t(i * (n + 1) + j); };
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
    {
      b.tris.insert(b.tris.end(), {at(i, j), at(i, j + 1), at(i + 1, j)});
      b.tris.insert(b.tris.end(), {at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)});
    }
  b.outer.assign(b.tris.size() / 3, 1);
  b.regions.assign(b.positions.size() / 3, 0);
  return b;
}

// Two plates facing each other, a crease `gap` cm deep: the lower in the xz plane facing up, the
// upper over it facing down. The lower's vertices come first.
Body crease(double gap, int n = 8, double size = 20)
{
  Body         b     = plate(n, size);
  const size_t lower = b.positions.size() / 3;
  const Body   top   = plate(n, size);
  for(size_t v = 0; v < lower; v++)
    b.positions.insert(b.positions.end(), {top.positions[v * 3], float(gap), top.positions[v * 3 + 2]});
  // turned over: the same triangles wound the other way
  for(size_t t = 0; t + 2 < top.tris.size(); t += 3)
    b.tris.insert(b.tris.end(), {uint32_t(lower) + top.tris[t], uint32_t(lower) + top.tris[t + 2], uint32_t(lower) + top.tris[t + 1]});
  b.outer.assign(b.tris.size() / 3, 1);
  b.regions.assign(b.positions.size() / 3, 0);
  return b;
}

// A skin of rows of (bone, weight), one row a vertex.
fv::mesh::SkinInput skinOf(const std::vector<std::vector<std::pair<int, float>>>& rows)
{
  fv::mesh::SkinInput s;
  s.start.push_back(0);
  for(const auto& row : rows)
  {
    for(const auto& [bone, weight] : row)
    {
      s.bone.push_back(bone);
      s.weight.push_back(weight);
    }
    s.start.push_back(int32_t(s.bone.size()));
  }
  return s;
}

// A vertex's influences, by bone.
std::map<int, float> rowOf(const fv::mesh::SkinInput& s, size_t v)
{
  std::map<int, float> out;
  for(int32_t i = s.start[v]; i < s.start[v + 1]; i++)
    out[s.bone[size_t(i)]] += s.weight[size_t(i)];
  return out;
}

// A type whose reference lies on it: the map of `source` onto `reference`.
TypeMap typeOf(const Body& source, const Body& reference)
{
  return TypeMap{source, reference, fv::bodymap::mapBodies(source, reference), {}};
}

double radiusOf(const std::vector<float>& p, size_t v)
{
  return std::sqrt(double(p[v * 3]) * p[v * 3] + double(p[v * 3 + 1]) * p[v * 3 + 1] + double(p[v * 3 + 2]) * p[v * 3 + 2]);
}

}  // namespace

TEST(Garment, ConvertedAndDecodedOntoItsOwnBodyIsUnchanged)
{
  const Body    body  = sphere(10, 16, 24);
  const TypeMap type  = typeOf(body, body);
  const Body    shell = sphere(10.6, 11, 17);  // another tessellation, so its vertices lie over faces and edges
  const auto    a     = fv::garment::convert(shell.positions, type);
  const auto    back  = fv::garment::decode(a, body).positions;
  ASSERT_EQ(back.size(), shell.positions.size());
  for(size_t v = 0; v < a.hold.size(); v++)
  {
    EXPECT_FLOAT_EQ(a.hold[v], 1.f) << v;
    for(size_t k = 0; k < 3; k++)
      EXPECT_NEAR(back[v * 3 + k], shell.positions[v * 3 + k], 1e-4) << v;
  }
}

TEST(Garment, AShellKeepsItsHeightOnAThickerBody)
{
  // the type's own body, the reference laid on it (another tessellation of the same sphere), and the
  // reference in the target's shape: 2 cm thicker
  const Body    source = sphere(10, 16, 24), laid = sphere(10, 20, 30), target = sphere(12, 20, 30);
  const TypeMap type   = typeOf(source, laid);
  const Body    shell  = sphere(10.8, 12, 18);
  const auto    a      = fv::garment::convert(shell.positions, type);
  const auto    worn   = fv::garment::decode(a, target).positions;
  for(size_t v = 0; v < a.hold.size(); v++)
  {
    ASSERT_GT(a.hold[v], 0.f) << v;
    // 0.8 cm off the skin still (the spheres are faceted: a millimetre's room), and over the same place
    EXPECT_NEAR(radiusOf(worn, v), 12.8, 0.1) << v;
    double along = 0;
    for(size_t k = 0; k < 3; k++)
      along += double(worn[v * 3 + k]) * shell.positions[v * 3 + k];
    EXPECT_GT(along / (radiusOf(worn, v) * 10.8), 0.999) << v;
  }
}

TEST(Garment, AnOffsetTurnsWithTheBodysFrame)
{
  // the reference's base lies turned a quarter about z from the type's body: up there is +x
  const Body source = plate();
  TypeMap    type   = typeOf(source, source);
  const Quat turn{0, 0, -std::sqrt(0.5), std::sqrt(0.5)};
  type.map.frameTurn.assign(source.positions.size() / 3, turn);
  Body target = source;
  for(size_t v = 0; v * 3 < target.positions.size(); v++)
  {
    const Vec3 p = fv::rotate({source.positions[v * 3], source.positions[v * 3 + 1], source.positions[v * 3 + 2]}, turn);
    target.positions[v * 3] = float(p.x), target.positions[v * 3 + 1] = float(p.y), target.positions[v * 3 + 2] = float(p.z);
  }
  // one vertex over the plate, one past its edge: its offset has a part across the skin
  const std::vector<float> item = {3, 1.5f, 2, 12, 1, 0};
  const auto               a    = fv::garment::convert(item, type);
  const auto               worn = fv::garment::decode(a, target).positions;
  EXPECT_NEAR(a.height[0], 1.5, 1e-5);
  EXPECT_NEAR(a.height[1], 1, 1e-5);
  for(size_t v = 0; v < 2; v++)
  {
    ASSERT_GT(a.hold[v], 0.f);
    const Vec3 want = fv::rotate({item[v * 3], item[v * 3 + 1], item[v * 3 + 2]}, turn);
    EXPECT_NEAR(worn[v * 3], want.x, 1e-4) << v;
    EXPECT_NEAR(worn[v * 3 + 1], want.y, 1e-4) << v;
    EXPECT_NEAR(worn[v * 3 + 2], want.z, 1e-4) << v;
  }
}

TEST(Garment, AnAddressHoldsLessWithDistance)
{
  const Body               source = plate();
  const TypeMap            type   = typeOf(source, source);
  const std::vector<float> item   = {0, 1, 0, 1, 3.5f, 1, 2, 6, 2, 3, -0.5f, 3};
  const auto               a      = fv::garment::convert(item, type);
  EXPECT_FLOAT_EQ(a.hold[0], 1.f);
  EXPECT_NEAR(a.hold[1], 0.5, 1e-4);  // half way between 2 and 5 cm
  EXPECT_FLOAT_EQ(a.hold[2], 0.f);
  // under the skin is held too, at a height below zero
  EXPECT_FLOAT_EQ(a.hold[3], 1.f);
  EXPECT_NEAR(a.height[3], -0.5, 1e-5);
  fv::garment::ConvertOptions tight;
  tight.inner = 0.5f, tight.outer = 0.9f;
  EXPECT_FLOAT_EQ(fv::garment::convert(item, type, nullptr, tight).hold[0], 0.f);
}

TEST(Garment, ClothNothingHoldsMovesWithTheClothThatIs)
{
  // a strip of 12 x 1 cells hanging from its first two columns, and a triangle joined to nothing held
  const int             n = 12;
  std::vector<float>    moved;
  std::vector<uint32_t> tris;
  for(int i = 0; i <= n; i++)
    for(int j = 0; j < 2; j++)
      moved.insert(moved.end(), {float(i), 0.f, float(j)});
  for(int i = 0; i < n; i++)
  {
    const uint32_t a = uint32_t(i * 2);
    tris.insert(tris.end(), {a, a + 1, a + 2, a + 2, a + 1, a + 3});
  }
  const uint32_t loose = uint32_t(moved.size() / 3);
  moved.insert(moved.end(), {40, 0, 0, 41, 0, 0, 40, 0, 1});
  tris.insert(tris.end(), {loose, loose + 1, loose + 2});
  std::vector<float> decoded(moved.size(), 0.f), hold(moved.size() / 3, 0.f);
  for(size_t v = 0; v < 4; v++)
  {
    hold[v]            = 1;
    decoded[v * 3]     = moved[v * 3] + 0.5f;
    decoded[v * 3 + 1] = moved[v * 3 + 1] + 2;
    decoded[v * 3 + 2] = moved[v * 3 + 2];
  }
  const auto out = fv::garment::settle(moved, decoded, hold, tris);
  for(size_t v = 0; v < loose; v++)
  {
    EXPECT_NEAR(out[v * 3], moved[v * 3] + 0.5, 1e-4) << v;
    EXPECT_NEAR(out[v * 3 + 1], 2, 1e-4) << v;
    EXPECT_NEAR(out[v * 3 + 2], moved[v * 3 + 2], 1e-4) << v;
  }
  for(size_t i = size_t(loose) * 3; i < moved.size(); i++)
    EXPECT_FLOAT_EQ(out[i], moved[i]) << i;
}

TEST(Garment, AVertexHeldInPartGoesPartOfTheWay)
{
  // a fan: a middle vertex held by half, its ring held whole and moving 1 cm up; its own address says 1.6
  std::vector<float>    moved = {0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1};
  std::vector<uint32_t> tris  = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1};
  std::vector<float>    decoded = moved, hold = {0.5f, 1, 1, 1, 1};
  decoded[1] = 1.6f;
  for(size_t v = 1; v < 5; v++)
    decoded[v * 3 + 1] = 1;
  const auto out = fv::garment::settle(moved, decoded, hold, tris);
  EXPECT_NEAR(out[1], 1.3, 1e-4);  // half its own 1.6, half its neighbours' 1
  for(size_t v = 1; v < 5; v++)
    EXPECT_NEAR(out[v * 3 + 1], 1, 1e-5);
}

TEST(Garment, ClothBridgingACreaseSharesThePull)
{
  // a strip of 8 x 1 cells, all of it held: its left half stays and its right half goes 2 cm along, as
  // if each lay on its own side of a crease. The edge between them would be pulled to three times its
  // length; let go, its ends share the pull with the cells beside them
  const int             n = 8;
  std::vector<float>    moved;
  std::vector<uint32_t> tris;
  for(int i = 0; i <= n; i++)
    for(int j = 0; j < 2; j++)
      moved.insert(moved.end(), {float(i), 0.f, float(j)});
  for(int i = 0; i < n; i++)
  {
    const uint32_t a = uint32_t(i * 2);
    tris.insert(tris.end(), {a, a + 1, a + 2, a + 2, a + 1, a + 3});
  }
  std::vector<float> decoded = moved, hold(moved.size() / 3, 1.f);
  for(int i = 5; i <= n; i++)
    for(int j = 0; j < 2; j++)
      decoded[size_t(i * 2 + j) * 3] += 2;
  const auto out = fv::garment::settle(moved, decoded, hold, tris);
  double     most = 0;
  for(int i = 0; i < n; i++)
    for(int j = 0; j < 2; j++)
      most = std::max(most, double(out[size_t((i + 1) * 2 + j) * 3] - out[size_t(i * 2 + j) * 3]));
  EXPECT_LT(most, 2.0);  // 3 let alone; 1.67 shared evenly over three cells
  // the cloth away from the crease is where its addresses say
  for(int i : {0, 1, 2, 3, 6, 7, 8})
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(out[size_t(i * 2 + j) * 3], decoded[size_t(i * 2 + j) * 3]) << i;
}

TEST(Garment, AnAddressMovesAVertexNoFurtherThanTheCap)
{
  const std::vector<float>    moved = {0, 0, 0}, decoded = {0, 100, 0}, hold = {1};
  fv::garment::SettleOptions options;
  options.cap    = 5;
  const auto out = fv::garment::settle(moved, decoded, hold, {}, options);
  EXPECT_NEAR(out[1], 5, 1e-5);
}

TEST(Garment, AShellGoesOntoABodyOfAnotherTopology)
{
  // made on one tessellation of a sphere; worn on a thicker body of a third tessellation, through the
  // reference laid on each
  const Body    source = sphere(10, 16, 24), laidOnSource = sphere(10, 20, 30);
  const Body    laidOnTarget = sphere(12, 20, 30), body = sphere(12, 14, 22);
  const TypeMap type  = typeOf(source, laidOnSource);
  const Body    shell = sphere(10.8, 12, 18);
  const auto    a     = fv::garment::convert(shell.positions, type);
  fv::garment::Target target;
  target.reference = &laidOnTarget;
  target.body      = &body;
  const auto worn  = fv::garment::decode(a, target);
  for(size_t v = 0; v < a.hold.size(); v++)
  {
    ASSERT_GT(worn.hold[v], 0.f) << v;
    EXPECT_NEAR(radiusOf(worn.positions, v), 12.8, 0.15) << v;
    // bound to the body's own vertices, at weights that sum to one
    float sum = 0;
    for(size_t j = 0; j < 3; j++)
    {
      EXPECT_LT(worn.at.verts[v * 3 + j], body.positions.size() / 3);
      sum += worn.at.weights[v * 3 + j];
    }
    EXPECT_NEAR(sum, 1, 1e-4) << v;
  }
}

TEST(Garment, AnOffsetTurnsBackIntoTheTargetsFrame)
{
  // the target's frames lie a quarter turn from the reference base's, as the source's do: what was
  // turned into the base's frame is turned back out of it, and the garment is unchanged
  const Body source = plate();
  TypeMap    type   = typeOf(source, source);
  const Quat turn{0, 0, -std::sqrt(0.5), std::sqrt(0.5)};
  type.map.frameTurn.assign(source.positions.size() / 3, turn);
  const std::vector<float> item = {12, 1, 0, -11, 0.5f, 3};  // past the plate's edges
  const auto               a    = fv::garment::convert(item, type);
  fv::garment::Target      target;
  target.reference = &source;
  target.body      = &source;
  target.frameTurn = &type.map.frameTurn;
  const auto worn  = fv::garment::decode(a, target);
  for(size_t i = 0; i < item.size(); i++)
    EXPECT_NEAR(worn.positions[i], item[i], 1e-4) << i;
}

TEST(Garment, AVertexInACreaseIsBoundOnItsAddresssSide)
{
  // cloth 2 mm over the lower side of a crease 1 cm deep, made for that side (its limb). On the target
  // the crease is 3 mm deep: the upper side is the nearer skin, and the address still names the lower
  const Body   source = crease(1.0), target = crease(0.3);
  const size_t lower  = source.positions.size() / 6;
  const TypeMap type  = typeOf(source, source);
  fv::wardrobe::BindFilter filter;
  filter.bodyLimb.assign(source.positions.size() / 3, 2);
  std::fill(filter.bodyLimb.begin(), filter.bodyLimb.begin() + ptrdiff_t(lower), int16_t(1));
  const std::vector<float> item = {0.3f, 0.2f, 0.4f, -4.1f, 0.2f, 2.2f};
  filter.itemMask.assign(2, 1 << 1);
  const auto a    = fv::garment::convert(item, type, &filter);
  const auto worn = fv::garment::decode(a, target);
  for(size_t v = 0; v < 2; v++)
  {
    ASSERT_FLOAT_EQ(worn.hold[v], 1.f);
    EXPECT_NEAR(worn.positions[v * 3 + 1], 0.2, 1e-5);
    for(size_t j = 0; j < 3; j++)
      if(worn.at.weights[v * 3 + j] != 0)
        EXPECT_LT(worn.at.verts[v * 3 + j], lower) << v;
  }
  // and that place replaces the closest point's in the mesh's binding, which is on the upper side
  const fv::wardrobe::SurfaceBinding bound = fv::wardrobe::bindToSurface(target.positions, target.tris, worn.positions);
  bool                               upper = false;
  for(size_t i = 0; i < bound.verts.size(); i++)
    upper = upper || (bound.weights[i] != 0 && bound.verts[i] >= lower);
  EXPECT_TRUE(upper);
  const fv::wardrobe::SurfaceBinding kept = fv::garment::withPlaces(bound, worn.at);
  for(size_t i = 0; i < kept.verts.size(); i++)
    if(kept.weights[i] != 0)
    {
      EXPECT_LT(kept.verts[i], lower) << i;
      EXPECT_TRUE(kept.used[kept.verts[i]]);
    }
  // a vertex decode didn't place keeps the closest point's
  fv::wardrobe::SurfaceBinding none = worn.at;
  std::fill(none.weights.begin(), none.weights.end(), 0.f);
  EXPECT_EQ(fv::garment::withPlaces(bound, none).verts, bound.verts);
}

TEST(Garment, TheReferencesOwnMapPutsEveryVertexAtItself)
{
  const Body    body  = sphere(10, 16, 24);
  const TypeMap type  = fv::garment::referenceMap(body);
  const Body    shell = sphere(10.5, 11, 17);
  const auto    a     = fv::garment::convert(shell.positions, type);
  const auto    back  = fv::garment::decode(a, body);
  for(size_t i = 0; i < shell.positions.size(); i++)
    EXPECT_NEAR(back.positions[i], shell.positions[i], 1e-4) << i;
  // on the reference the address is the binding
  EXPECT_EQ(back.at.verts, a.at.verts);
}

TEST(GarmentWeights, ClothOnTheSkinTakesTheSkinsWeightsThere)
{
  // a body of three vertices on bones 0, 1 and 2 (the last shared with 1), and a garment of two bones
  // of its own. Its first vertex lies on the skin at 0.5, 0.3, 0.2 of the three; its second hangs free
  const fv::mesh::SkinInput body = skinOf({{{0, 1.f}}, {{1, 1.f}}, {{1, 0.5f}, {2, 0.5f}}});
  const fv::mesh::SkinInput item = skinOf({{{0, 0.25f}, {1, 0.75f}}, {{1, 1.f}}});
  fv::wardrobe::SurfaceBinding at;
  at.verts   = {0, 1, 2, 0, 1, 2};
  at.weights = {0.5f, 0.3f, 0.2f, 1, 0, 0};
  const std::vector<float> hold = {1, 0};
  const auto               skin = fv::garment::withBodyWeights(item, 2, body, at, hold);
  ASSERT_EQ(skin.start.size(), 3u);
  // the body's bones come after the garment's two
  const auto first = rowOf(skin, 0);
  ASSERT_EQ(first.size(), 3u);
  EXPECT_NEAR(first.at(2), 0.5, 1e-6);
  EXPECT_NEAR(first.at(3), 0.3 + 0.1, 1e-6);
  EXPECT_NEAR(first.at(4), 0.1, 1e-6);
  // the free vertex keeps what it was made with
  const auto second = rowOf(skin, 1);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_NEAR(second.at(1), 1, 1e-6);
  // strongest first, as the drawn skin's four are taken
  EXPECT_EQ(skin.bone[size_t(skin.start[0])], 2);
}

TEST(GarmentWeights, ClothHeldInPartTakesAShareOfEach)
{
  const fv::mesh::SkinInput body = skinOf({{{0, 1.f}}});
  const fv::mesh::SkinInput item = skinOf({{{0, 1.f}}, {}});
  fv::wardrobe::SurfaceBinding at;
  at.verts   = {0, 0, 0, 0, 0, 0};
  at.weights = {1, 0, 0, 1, 0, 0};
  const auto skin = fv::garment::withBodyWeights(item, 1, body, at, std::vector<float>{0.25f, 0.5f});
  const auto a    = rowOf(skin, 0);
  EXPECT_NEAR(a.at(0), 0.75, 1e-6);
  EXPECT_NEAR(a.at(1), 0.25, 1e-6);
  // a vertex made with no weights rides the garment's root bone for the share that isn't the body's
  const auto b = rowOf(skin, 1);
  EXPECT_NEAR(b.at(item.rootBone), 0.5, 1e-6);
  EXPECT_NEAR(b.at(1), 0.5, 1e-6);
}

TEST(GarmentWeights, ACornerWithNoWeightsIsLeftOut)
{
  // bound between a weighted body vertex and one with none (an eyeball's, say): the weighted one's are all it takes
  const fv::mesh::SkinInput body = skinOf({{{3, 1.f}}, {}});
  const fv::mesh::SkinInput item = skinOf({{{0, 1.f}}});
  fv::wardrobe::SurfaceBinding at;
  at.verts   = {0, 1, 1};
  at.weights = {0.4f, 0.6f, 0};
  const auto row = rowOf(fv::garment::withBodyWeights(item, 1, body, at, std::vector<float>{1}), 0);
  ASSERT_EQ(row.size(), 1u);
  EXPECT_NEAR(row.at(1 + 3), 1, 1e-6);
  // and bound to nothing weighted, it keeps its own
  at.weights     = {0, 1, 0};
  const auto own = rowOf(fv::garment::withBodyWeights(item, 1, body, at, std::vector<float>{1}), 0);
  EXPECT_NEAR(own.at(0), 1, 1e-6);
}

TEST(GarmentWeights, TheDualQuaternionShareGoesOverWithTheWeights)
{
  fv::mesh::SkinInput body = skinOf({{{0, 1.f}}, {{0, 1.f}}});
  body.dualQuatMap         = {1.f, 0.5f};
  fv::mesh::SkinInput item = skinOf({{{0, 1.f}}, {{0, 1.f}}, {{0, 1.f}}});
  item.dualQuat            = 0;
  fv::wardrobe::SurfaceBinding at;
  at.verts   = {0, 1, 1, 0, 1, 1, 0, 1, 1};
  at.weights = {1, 0, 0, 0.5f, 0.5f, 0, 1, 0, 0};
  const auto skin = fv::garment::withBodyWeights(item, 1, body, at, std::vector<float>{1, 1, 0});
  ASSERT_EQ(skin.dualQuatMap.size(), 3u);
  EXPECT_NEAR(skin.dualQuatMap[0], 1, 1e-6);
  EXPECT_NEAR(skin.dualQuatMap[1], 0.75, 1e-6);
  EXPECT_NEAR(skin.dualQuatMap[2], 0, 1e-6);
  // all linear on both: no map
  body.dualQuatMap.clear();
  EXPECT_TRUE(fv::garment::withBodyWeights(item, 1, body, at, std::vector<float>{1, 1, 0}).dualQuatMap.empty());
}

TEST(GarmentWeights, HoldFallsWithTheDistanceOffTheBody)
{
  const std::vector<float> body = {0, 0, 0, 10, 0, 0, 0, 0, 10};
  const std::vector<float> item = {1, 1, 1, 1, 3.5f, 1, 1, 7, 1, 1, 1, 1};
  fv::wardrobe::SurfaceBinding at;
  at.verts   = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  at.weights = {0.8f, 0.1f, 0.1f, 0.8f, 0.1f, 0.1f, 0.8f, 0.1f, 0.1f, 0, 0, 0};
  const auto hold = fv::garment::holdByDistance(item, body, at);
  EXPECT_FLOAT_EQ(hold[0], 1.f);
  EXPECT_NEAR(hold[1], 0.5, 1e-4);
  EXPECT_FLOAT_EQ(hold[2], 0.f);
  EXPECT_FLOAT_EQ(hold[3], 0.f);  // bound to nothing
}
