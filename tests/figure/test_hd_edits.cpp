// HD morph detail (.dhdm): the file read, its edits placed on a refinement as OpenSubdiv numbers it,
// and the reference that applies them. With DAZ_LIBRARY set, Genesis 9's FACS Details are read, placed
// and checked against the basis they are measured in.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <tinygltf/json.hpp>

#include <figure/dson/dhdm.hpp>
#include <figure/figure.hpp>
#include <figure/io/gzip.hpp>
#include <figure/subdiv/catmark.hpp>
#include <figure/subdiv/hd_edits.hpp>

using namespace fv;
namespace fs = std::filesystem;

namespace {

// --- building a file

struct Bytes
{
  std::string s;
  template <class T>
  void put(T v)
  {
    s.append(reinterpret_cast<const char*>(&v), sizeof(T));
  }
};

struct Edit
{
  uint32_t face, sub, corner;
  float    a, b, c;
};

// One level's block: edits grouped by face in the order given.
std::string levelBlock(uint32_t faces, uint32_t level, const std::vector<Edit>& edits)
{
  Bytes body;
  for(size_t i = 0; i < edits.size();)
  {
    size_t j = i;
    while(j < edits.size() && edits[j].face == edits[i].face)
      j++;
    body.put<uint32_t>(edits[i].face);
    body.put<uint32_t>(uint32_t(j - i));
    for(; i < j; i++)
    {
      const Edit&    e     = edits[i];
      const uint32_t index = (e.sub << (16 - 2 * level)) | (e.corner << (14 - 2 * level));
      body.put<float>(e.a);
      if(level < 4)
      {
        EXPECT_EQ(index & 0xff, 0u) << "a short index has no low byte";
        body.put<uint8_t>(uint8_t(index >> 8));
        body.put<uint8_t>(uint8_t((level + 1) * 16));
      }
      else
      {
        body.put<uint8_t>(0);
        body.put<uint16_t>(uint16_t(index));
        body.put<uint8_t>(uint8_t((level + 1) * 16));
      }
      body.put<float>(e.b);
      body.put<float>(e.c);
    }
  }
  Bytes head;
  head.put<uint32_t>(faces);
  head.put<uint32_t>(level);
  head.put<uint32_t>(uint32_t(edits.size()));
  head.put<uint32_t>(uint32_t(body.s.size()));
  return head.s + body.s;
}

std::string file(uint32_t faces, const std::vector<std::vector<Edit>>& levels)
{
  Bytes b;
  b.put<uint32_t>(0xd0d0d0d0u);
  b.put<uint32_t>(uint32_t(levels.size()));
  b.put<float>(1.f);
  b.put<uint32_t>(uint32_t(levels.size()));
  for(size_t k = 0; k < levels.size(); k++)
    b.s += levelBlock(faces, uint32_t(k + 1), levels[k]);
  return b.s;
}

// --- a cage: a cube with one face split into a triangle and a pentagon's worth of corners

struct Cage
{
  std::vector<float>   positions;
  std::vector<int32_t> faceStart{0}, faceVerts;

  void face(std::initializer_list<int32_t> v)
  {
    faceVerts.insert(faceVerts.end(), v);
    faceStart.push_back(int32_t(faceVerts.size()));
  }
  subdiv::CageMesh mesh() const { return {int(positions.size() / 3), faceStart, faceVerts, {}, {}}; }
};

// A box with a triangle fan on its top, so the cage has quads and triangles.
Cage box()
{
  Cage c;
  c.positions = {0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 1, 1, 2.5f};
  c.face({0, 3, 2, 1});
  c.face({0, 1, 5, 4});
  c.face({1, 2, 6, 5});
  c.face({2, 3, 7, 6});
  c.face({3, 0, 4, 7});
  c.face({4, 5, 8});
  c.face({5, 6, 8});
  c.face({6, 7, 8});
  c.face({7, 4, 8});
  return c;
}

using Frame = std::function<std::array<float, 9>(uint32_t faceCorner)>;

// Refines level by level with the edits added in `frame` (the test's own loop, so other bases can be tried).
std::vector<double> refineFramed(const subdiv::Refinement& r, std::span<const float> cage, const subdiv::HdEdits& e, double weight, const Frame& frame)
{
  std::vector<double> cur(cage.begin(), cage.end());
  for(size_t k = 0; k < r.levels.size(); k++)
  {
    const subdiv::Stencils& s = r.levels[k];
    std::vector<double>     next(size_t(s.rows()) * 3, 0.0);
    for(int row = 0; row < s.rows(); row++)
      for(int j = s.rowStart[row]; j < s.rowStart[row + 1]; j++)
        for(int c = 0; c < 3; c++)
          next[size_t(row) * 3 + c] += double(s.weight[j]) * cur[size_t(s.col[j]) * 3 + c];
    if(k < e.levels.size())
    {
      const auto& l = e.levels[k];
      for(size_t i = 0; i < l.vertex.size(); i++)
      {
        const auto m = frame(l.faceCorner[i]);
        for(int c = 0; c < 3; c++)
          next[size_t(l.vertex[i]) * 3 + c] += weight * (m[c] * l.offset[i * 3] + m[3 + c] * l.offset[i * 3 + 1] + m[6 + c] * l.offset[i * 3 + 2]);
      }
    }
    cur = std::move(next);
  }
  return cur;
}

}  // namespace

TEST(HdEdits, AFileReadsAsWritten)
{
  const std::vector<std::vector<Edit>> levels = {
      {{3, 1, 2, 0.5f, -1.f, 2.f}, {3, 3, 0, 0.f, 0.f, 1.f}, {7, 0, 3, 1.f, 2.f, 3.f}},
      {{2, 13, 1, -0.25f, 0.125f, 4.f}},
      {},
      {{1, 200, 2, 9.f, 8.f, 7.f}},
  };
  const dson::Dhdm d = dson::parseDhdm(file(9, levels));
  EXPECT_EQ(d.faceCount, 9u);
  ASSERT_EQ(d.levels.size(), levels.size());
  for(size_t k = 0; k < levels.size(); k++)
  {
    ASSERT_EQ(d.levels[k].size(), levels[k].size()) << "level " << k + 1;
    for(size_t i = 0; i < levels[k].size(); i++)
    {
      const subdiv::FaceEdit& got = d.levels[k][i];
      const Edit&             want = levels[k][i];
      EXPECT_EQ(got.face, want.face);
      EXPECT_EQ(got.sub, want.sub);
      EXPECT_EQ(got.corner, want.corner);
      EXPECT_EQ(got.offset[0], want.a);
      EXPECT_EQ(got.offset[1], want.b);
      EXPECT_EQ(got.offset[2], want.c);
    }
  }
}

TEST(HdEdits, ABrokenFileIsRefused)
{
  const std::string good = file(9, {{{3, 1, 2, 0.5f, -1.f, 2.f}}});
  EXPECT_THROW(dson::parseDhdm(good.substr(0, good.size() - 3)), std::runtime_error);
  EXPECT_THROW(dson::parseDhdm("not a file at all, nope"), std::runtime_error);
  std::string wrongLevel = good;
  wrongLevel[16 + 4] = 2;  // the level header says level 2 first
  EXPECT_THROW(dson::parseDhdm(wrongLevel), std::runtime_error);
}

// An edit at level L names the vertex at the corner of that level's face: the same vertex the cage
// refined L times has there, whichever finer level it is placed on (a vertex keeps its index).
TEST(HdEdits, AnEditLandsOnTheCornerOfItsLevelsFace)
{
  const Cage c = box();
  for(int finest = 1; finest <= 3; finest++)
  {
    const subdiv::Refinement r = subdiv::refine(c.mesh(), finest);
    subdiv::LevelEdits       edits(static_cast<size_t>(finest));
    for(int level = 1; level <= finest; level++)
    {
      const uint32_t per = 1u << (2 * (level - 1));
      for(uint32_t f = 0; f + 1 < c.faceStart.size(); f++)
      {
        const uint32_t size = uint32_t(c.faceStart[f + 1] - c.faceStart[f]);
        for(uint32_t s = 0; s < size * per; s += 3)
          for(uint8_t corner = 0; corner < 4; corner++)
            edits[size_t(level - 1)].push_back({f, s, corner, {1, 0, 0}});
      }
    }
    const subdiv::HdEdits placed = subdiv::placeEdits(r, c.faceStart, edits);
    EXPECT_EQ(placed.dropped, 0u);
    ASSERT_EQ(placed.levels.size(), size_t(finest));
    for(int level = 1; level <= finest; level++)
    {
      const subdiv::Refinement atLevel = subdiv::refine(c.mesh(), level);
      const auto&              want    = edits[size_t(level - 1)];
      const auto&              got     = placed.levels[size_t(level - 1)];
      ASSERT_EQ(got.vertex.size(), want.size());
      const uint32_t per = 1u << (2 * (level - 1));
      for(size_t i = 0; i < want.size(); i++)
      {
        const size_t face = size_t(c.faceStart[want[i].face]) * per + want[i].sub;
        EXPECT_EQ(got.vertex[i], uint32_t(atLevel.quads[face * 4 + want[i].corner]))
            << "finest " << finest << ", level " << level << ", face " << want[i].face << " sub " << want[i].sub;
        EXPECT_EQ(got.faceCorner[i], uint32_t(c.faceStart[want[i].face]) + want[i].sub / per);
      }
    }
  }
}

TEST(HdEdits, EditsPastTheRefinementOrOffTheCageAreDropped)
{
  const Cage               c = box();
  const subdiv::Refinement r = subdiv::refine(c.mesh(), 1);
  subdiv::LevelEdits       edits(2);
  edits[0] = {{0, 3, 1, {1, 0, 0}}, {5, 3, 0, {1, 0, 0}}, {99, 0, 0, {1, 0, 0}}};  // a triangle has no fourth child; no face 99
  edits[1] = {{0, 0, 0, {1, 0, 0}}};
  const subdiv::HdEdits placed = subdiv::placeEdits(r, c.faceStart, edits);
  ASSERT_EQ(placed.levels.size(), 1u);
  EXPECT_EQ(placed.levels[0].vertex.size(), 1u);
  EXPECT_EQ(placed.dropped, 3u);
}

// The frame: x along the edge to the corner before, the face's normal, and -y, all unit and square.
TEST(HdEdits, ACornersFrameFollowsItsFace)
{
  const Cage c     = box();
  const auto owner = subdiv::cornerFaces(c.faceStart);
  // face 0 is the bottom, wound downward: corner 1 (vertex 3) has corner 0 (vertex 0) before it
  const auto m = subdiv::cornerFrame(c.positions, c.faceStart, c.faceVerts, owner, 1);
  EXPECT_NEAR(m[0], 0, 1e-6);
  EXPECT_NEAR(m[1], -1, 1e-6);
  EXPECT_NEAR(m[2], 0, 1e-6);
  EXPECT_NEAR(m[5], -1, 1e-6) << "the bottom's normal points down";
  for(uint32_t corner = 0; corner < uint32_t(c.faceVerts.size()); corner++)
  {
    const auto f = subdiv::cornerFrame(c.positions, c.faceStart, c.faceVerts, owner, corner);
    for(int a = 0; a < 3; a++)
      for(int b = 0; b < 3; b++)
      {
        const double dot = f[a * 3] * f[b * 3] + f[a * 3 + 1] * f[b * 3 + 1] + f[a * 3 + 2] * f[b * 3 + 2];
        EXPECT_NEAR(dot, a == b ? 1.0 : 0.0, 1e-5) << "corner " << corner;
      }
  }
}

// The reference: no edits is plain refinement; an edit at the finest level moves its vertex alone, by
// its frame; one at a coarser level spreads as that level's points spread.
TEST(HdEdits, TheReferenceAddsEditsLevelByLevel)
{
  const Cage               c = box();
  const subdiv::Refinement r = subdiv::refine(c.mesh(), 2);
  const std::vector<float> plain = subdiv::refineDense(r, c.positions, 3);
  const std::vector<float> unedited = subdiv::refineWithEdits(r, c.positions, c.faceStart, c.faceVerts, {});
  ASSERT_EQ(unedited.size(), plain.size());
  for(size_t i = 0; i < plain.size(); i++)
    EXPECT_NEAR(unedited[i], plain[i], 1e-5);  // the reference sums in double, refineDense rounds each term

  const auto owner = subdiv::cornerFaces(c.faceStart);
  subdiv::LevelEdits fine(2);
  fine[1] = {{2, 5, 2, {0.1f, 0.2f, -0.3f}}};
  const subdiv::HdEdits placed = subdiv::placeEdits(r, c.faceStart, fine);
  const std::pair<float, const subdiv::HdEdits*> one[] = {{0.5f, &placed}};
  const std::vector<float> moved = subdiv::refineWithEdits(r, c.positions, c.faceStart, c.faceVerts, one);
  const uint32_t           v     = placed.levels[1].vertex[0];
  const auto               m     = subdiv::cornerFrame(c.positions, c.faceStart, c.faceVerts, owner, placed.levels[1].faceCorner[0]);
  for(size_t i = 0; i < plain.size() / 3; i++)
    for(int k = 0; k < 3; k++)
    {
      const float want = i == v ? plain[i * 3 + k] + 0.5f * (m[k] * 0.1f + m[3 + k] * 0.2f + m[6 + k] * -0.3f) : plain[i * 3 + k];
      EXPECT_NEAR(moved[i * 3 + k], want, 1e-5) << "vertex " << i;
    }

  // a level-1 edit: the same as refining level 1's points, moved, once more
  subdiv::LevelEdits coarse(1);
  coarse[0] = {{6, 1, 0, {0.f, 0.4f, 0.f}}};
  const subdiv::HdEdits c1 = subdiv::placeEdits(r, c.faceStart, coarse);
  const std::pair<float, const subdiv::HdEdits*> two[] = {{1.f, &c1}};
  const std::vector<float> spread = subdiv::refineWithEdits(r, c.positions, c.faceStart, c.faceVerts, two);
  const subdiv::Refinement r1     = subdiv::refine(c.mesh(), 1);
  std::vector<float>       level1 = subdiv::refineDense(r1, c.positions, 3);
  const auto               m1     = subdiv::cornerFrame(c.positions, c.faceStart, c.faceVerts, owner, c1.levels[0].faceCorner[0]);
  for(int k = 0; k < 3; k++)
    level1[c1.levels[0].vertex[0] * 3 + k] += m1[3 + k] * 0.4f;
  subdiv::Refinement second;
  second.levels.push_back(r.levels[1]);
  const std::vector<float> want = subdiv::refineDense(second, level1, 3);
  ASSERT_EQ(spread.size(), want.size());
  for(size_t i = 0; i < want.size(); i++)
    EXPECT_NEAR(spread[i], want[i], 1e-5);
}

// The mixer's tables, applied as the GPU applies them, move the mesh as the reference does; weights
// change the offsets only, and a new set of edits makes the topology again.
TEST(HdEdits, MixedTablesMoveTheMeshAsTheReferenceDoes)
{
  const Cage c = box();
  const auto r = std::make_shared<subdiv::Refinement>(subdiv::refine(c.mesh(), 2));
  auto       a = std::make_shared<subdiv::LevelEdits>(2);
  (*a)[0]      = {{0, 1, 2, {0.05f, 0.1f, -0.02f}}, {5, 2, 0, {0.f, 0.2f, 0.f}}};
  (*a)[1]      = {{2, 5, 2, {0.1f, 0.2f, -0.3f}}, {7, 9, 1, {-0.1f, 0.f, 0.05f}}};
  auto b       = std::make_shared<subdiv::LevelEdits>(2);
  (*b)[1]      = {{2, 5, 2, {0.3f, 0.f, 0.f}}, {4, 0, 3, {0.f, -0.1f, 0.f}}};  // shares a's first fine edit

  subdiv::HdMixer mixer(r, c.faceStart, c.faceVerts);
  const subdiv::HdMixer::Weighted none[] = {{a, 0.f}, {b, 0.f}};
  EXPECT_EQ(mixer.mix(none), nullptr);

  const subdiv::HdMixer::Weighted justA[] = {{a, 0.7f}};
  const auto                      first   = mixer.mix(justA);
  ASSERT_TRUE(first);
  const subdiv::HdEdits placedA = subdiv::placeEdits(*r, c.faceStart, *a);
  const subdiv::HdEdits placedB = subdiv::placeEdits(*r, c.faceStart, *b);
  {
    const std::pair<float, const subdiv::HdEdits*> want[] = {{0.7f, &placedA}};
    const auto ref = subdiv::refineWithEdits(*r, c.positions, c.faceStart, c.faceVerts, want);
    const auto got = subdiv::refineWithTables(*r, c.positions, first.get());
    ASSERT_EQ(got.size(), ref.size());
    for(size_t i = 0; i < ref.size(); i++)
      EXPECT_NEAR(got[i], ref[i], 1e-5);
  }

  const subdiv::HdMixer::Weighted both[] = {{a, 0.25f}, {b, -0.5f}};
  const auto                      second = mixer.mix(both);
  ASSERT_TRUE(second);
  EXPECT_NE(second->topology, first->topology) << "b joined";
  {
    const std::pair<float, const subdiv::HdEdits*> want[] = {{0.25f, &placedA}, {-0.5f, &placedB}};
    const auto ref = subdiv::refineWithEdits(*r, c.positions, c.faceStart, c.faceVerts, want);
    const auto got = subdiv::refineWithTables(*r, c.positions, second.get());
    const auto plain = subdiv::refineWithTables(*r, c.positions, nullptr);
    double     moved = 0;
    for(size_t i = 0; i < ref.size(); i++)
    {
      EXPECT_NEAR(got[i], ref[i], 1e-5);
      moved = std::max(moved, std::abs(double(got[i]) - plain[i]));
    }
    EXPECT_GE(second->reach + 1e-6, moved) << "the reach bounds the move";
    EXPECT_EQ(second->topology->levels[1].frame.size(), 3u) << "the shared edit is one entry";
  }

  const subdiv::HdMixer::Weighted onlyB[] = {{a, 0.f}, {b, 1.f}};
  EXPECT_EQ(mixer.mix(onlyB)->topology, second->topology) << "a weight going to zero keeps the topology";
  EXPECT_EQ(mixer.dropped(), 0u);
}

// A Genesis 9 character from the library: FACS Detail Strength on and a brow unit set, the unit's
// FACS Details morph loads with its .dhdm and the body's flush mixes it into HD tables
TEST(HdEdits, AGenesis9CharactersBrowTakesItsWrinklesWhenTheLibraryIsAvailable)
{
  const char* lib = std::getenv("DAZ_LIBRARY");
  if(!lib || !*lib)
    GTEST_SKIP() << "DAZ_LIBRARY not set";
  const char* rel = std::getenv("HD_FIGURE") ? std::getenv("HD_FIGURE") : "People/Genesis 9/Characters/Leyla 9.duf";
  if(!fs::exists(fs::path(lib) / rel))
    GTEST_SKIP() << rel << " not installed";
  dson::LoadOptions options;
  options.subdivision = 2;
  auto f = Figure::load(std::make_shared<dson::Store>(std::make_shared<dson::Library>(std::vector<fs::path>{fs::path(lib)})), rel, options);
  ASSERT_TRUE(f && f->morphs);
  f->morphs->cageOnly = [](int) { return true; };
  f->settle();
  f->drain();
  for(const char* strength : {"facs_ctrl_FACSDetailStrength", "FACSDetails_facs_ctrl_FACStrength"})
    std::printf("dial %s: has %d\n", strength, int(f->morphs->has(strength)));
  for(const char* unit : {"facs_BrowDownLeft", "facs_bs_BrowDownLeft"})
    std::printf("dial %s: has %d\n", unit, int(f->morphs->has(unit)));
  f->morphs->set(f->morphs->has("facs_ctrl_FACSDetailStrength") ? "facs_ctrl_FACSDetailStrength" : "FACSDetails_facs_ctrl_FACStrength", 1);
  f->morphs->set(f->morphs->has("facs_BrowDownLeft") ? "facs_BrowDownLeft" : "facs_bs_BrowDownLeft", 1);
  f->settle();
  f->drain();
  const std::string detail = "FACSDetails_facs_bs_BrowDownLeft_HD2";
  std::printf("%s: has %d, weight %.3f\n", detail.c_str(), int(f->morphs->has(detail)), f->morphs->weightOf(detail));
  int withHd = 0, tables = 0;
  for(size_t mi = 0; mi < f->loaded.scene.meshes.size(); mi++)
  {
    for(const auto& [t, w] : f->morphs->activeTargets(int(mi)))
      if(t->hd)
      {
        withHd++;
        std::printf("mesh %zu: an active target with HD, weight %.3f x %.3f, %zu levels\n", mi, w, t->hdScale, t->hd->size());
      }
    if(const auto& geo = f->loaded.scene.meshes[mi].geo)
      std::printf("mesh %zu: %d cage vertices, %zu polygons, %zu levels, %zu finest quads, hidden sets %zu\n", mi, geo->dsonVertexCount,
                  geo->cagePolyStart.size() - 1, geo->refinement ? geo->refinement->levels.size() : size_t(0),
                  geo->refinement ? geo->refinement->quads.size() / 4 : size_t(0), geo->hiddenPolys.size());
    if(const auto hd = f->morphs->hdDetail(int(mi)))
    {
      tables++;
      std::printf("mesh %zu: HD tables, %zu frames, reach %.4f\n", mi, hd->topology->frames(), hd->reach);
    }
  }
  EXPECT_GT(withHd, 0) << "no active target carries HD detail";
  EXPECT_GT(tables, 0) << "no mesh mixed HD tables";
  for(size_t mi = 0; mi < f->loaded.scene.meshes.size(); mi++)
    if(const auto hd = f->morphs->hdDetail(int(mi)))
    {
      EXPECT_GT(hd->topology->frames(), 0u) << "mesh " << mi << ": every edit dropped";
      EXPECT_GT(hd->reach, 0.f) << "mesh " << mi;
    }
}

// Genesis 9's FACS Details, read and placed whole at its cage's two levels. Of the bases an offset
// could be in, the one the reference uses gives the smoothest field: a wrong basis turns neighbouring
// offsets different ways and the field turns to noise.
TEST(HdEdits, Genesis9FacsDetailsWhenTheLibraryIsAvailable)
{
  const char* lib = std::getenv("DAZ_LIBRARY");
  if(!lib || !*lib)
    GTEST_SKIP() << "DAZ_LIBRARY not set";
  const fs::path root(lib);
  const fs::path base   = root / "data/Daz 3D/Genesis 9/Base/Genesis9.dsf";
  const fs::path detail = root / "data/Daz 3D/Genesis 9/Base/Morphs/Daz 3D/FACS Details";
  if(!fs::exists(base) || !fs::exists(detail))
    GTEST_SKIP() << "no Genesis 9 FACS Details";

  const auto geo = nlohmann::json::parse(readMaybeCompressed(base))["geometry_library"][0];
  Cage       c;
  for(const auto& v : geo["vertices"]["values"])
    for(int k = 0; k < 3; k++)
      c.positions.push_back(v[k].get<float>());
  for(const auto& p : geo["polylist"]["values"])
  {
    for(size_t k = 2; k < p.size(); k++)
      c.faceVerts.push_back(p[k].get<int32_t>());
    c.faceStart.push_back(int32_t(c.faceVerts.size()));
  }
  const uint32_t           faces = uint32_t(c.faceStart.size() - 1);
  const subdiv::Refinement r     = subdiv::refine(c.mesh(), 2);
  const auto               owner = subdiv::cornerFaces(c.faceStart);

  // neighbours over the finest quads' edges
  std::vector<std::vector<uint32_t>> next(size_t(r.vertexCount));
  for(size_t q = 0; q < r.quads.size(); q += 4)
    for(int k = 0; k < 4; k++)
    {
      const uint32_t a = uint32_t(r.quads[q + k]), b = uint32_t(r.quads[q + (k + 1) % 4]);
      next[a].push_back(b);
      next[b].push_back(a);
    }
  // how far each moved vertex's offset strays from its neighbours' mean, over how far they move
  const auto roughness = [&](const std::vector<double>& moved, const std::vector<double>& plain) {
    double stray = 0, size = 0;
    for(size_t v = 0; v < next.size(); v++)
    {
      double d[3], mean[3] = {0, 0, 0};
      for(int k = 0; k < 3; k++)
        d[k] = moved[v * 3 + k] - plain[v * 3 + k];
      const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if(len < 1e-5 || next[v].empty())
        continue;
      for(uint32_t n : next[v])
        for(int k = 0; k < 3; k++)
          mean[k] += (moved[n * 3 + k] - plain[n * 3 + k]) / double(next[v].size());
      stray += std::sqrt((d[0] - mean[0]) * (d[0] - mean[0]) + (d[1] - mean[1]) * (d[1] - mean[1]) + (d[2] - mean[2]) * (d[2] - mean[2]));
      size += len;
    }
    return size > 0 ? stray / size : 0.0;
  };

  const subdiv::HdEdits none;
  const Frame           ours = [&](uint32_t fc) { return subdiv::cornerFrame(c.positions, c.faceStart, c.faceVerts, owner, fc); };
  const std::vector<double> plain = refineFramed(r, c.positions, none, 0, ours);
  // other bases: the same axes in another order or sign, and the object's own axes
  const std::vector<std::pair<std::string, Frame>> others = {
      {"x n y", [&](uint32_t fc) { auto m = ours(fc); for(int k = 6; k < 9; k++) m[k] = -m[k]; return m; }},
      {"x y n", [&](uint32_t fc) { auto m = ours(fc); std::array<float, 9> o{}; for(int k = 0; k < 3; k++) { o[k] = m[k]; o[3 + k] = -m[6 + k]; o[6 + k] = m[3 + k]; } return o; }},
      {"n x y", [&](uint32_t fc) { auto m = ours(fc); std::array<float, 9> o{}; for(int k = 0; k < 3; k++) { o[k] = m[3 + k]; o[3 + k] = m[k]; o[6 + k] = -m[6 + k]; } return o; }},
      {"object", [](uint32_t) { return std::array<float, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1}; }},
  };

  int files = 0;
  for(const auto& entry : fs::directory_iterator(detail))
  {
    if(entry.path().extension() != ".dhdm")
      continue;
    const std::string name = entry.path().filename().string();
    const dson::Dhdm  d    = dson::parseDhdm(readMaybeCompressed(entry.path()));
    EXPECT_EQ(d.faceCount, faces) << name;
    const subdiv::HdEdits placed = subdiv::placeEdits(r, c.faceStart, d.levels);
    EXPECT_EQ(placed.dropped, 0u) << name;
    if(files++ % 4 == 0)  // the basis on a quarter of them: each one refines the whole figure five times
    {
      const double mine = roughness(refineFramed(r, c.positions, placed, 1, ours), plain);
      std::string  line = name + ": ours " + std::to_string(mine);
      for(const auto& [label, frame] : others)
      {
        const double other = roughness(refineFramed(r, c.positions, placed, 1, frame), plain);
        line += ", " + label + " " + std::to_string(other);
        EXPECT_LT(mine, other) << name << " is smoother in " << label;
      }
      std::printf("%s\n", line.c_str());
    }
  }
  EXPECT_GT(files, 0);
}
