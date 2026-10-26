// Surface bindings kept on disk: a second bind of the same arrays reads the first back, every input
// is part of the key, and a file that does not read back whole is bound again.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include <figure/mesh/mesh_build.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/bind_cache.hpp>

namespace {

// A flat grid of n x n quads in the XY plane at height z.
fv::mesh::MeshInput grid(int n, float z)
{
  fv::mesh::MeshInput m;
  for(int y = 0; y <= n; y++)
    for(int x = 0; x <= n; x++)
      m.positions.insert(m.positions.end(), {float(x), float(y), z});
  m.polyStart.push_back(0);
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      const int a = y * (n + 1) + x;
      m.polyVerts.insert(m.polyVerts.end(), {a, a + 1, a + n + 2, a + n + 1});
      m.polyStart.push_back(int(m.polyVerts.size()));
      m.polyMaterial.push_back(0);
    }
  m.uvs.assign(size_t(n + 1) * (n + 1) * 2, 0.f);
  return m;
}

void expectSame(const fv::wardrobe::SurfaceBinding& a, const fv::wardrobe::SurfaceBinding& b)
{
  EXPECT_EQ(a.verts, b.verts);
  EXPECT_EQ(a.weights, b.weights);
  EXPECT_EQ(a.used, b.used);
}

// A cache folder of its own for each test, and the cache off again after it.
class BindCache : public ::testing::Test
{
protected:
  void SetUp() override
  {
    dir = std::filesystem::temp_directory_path()
          / ("fv_bind_cache_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::filesystem::remove_all(dir);
    fv::wardrobe::setBindCacheDir(dir);
  }
  void TearDown() override
  {
    fv::wardrobe::setBindCacheDir({});
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  size_t files() const
  {
    size_t n = 0;
    std::error_code ec;
    for(const auto& e : std::filesystem::directory_iterator(dir, ec))
      n += e.path().extension() == ".fvb";
    return n;
  }
  std::filesystem::path dir;
};

}  // namespace

TEST_F(BindCache, AMissBindsAndKeepsItAndTheNextBindReadsItBack)
{
  const auto body   = fv::mesh::build(grid(6, 0));
  const auto item   = fv::mesh::build(grid(3, 0.5f));
  const auto fresh  = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  const auto before = fv::wardrobe::bindCacheCounts();

  const auto first = fv::wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().misses, before.misses + 1);
  EXPECT_EQ(files(), 1u);
  expectSame(first, fresh);

  const auto second = fv::wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().hits, before.hits + 1);
  expectSame(second, fresh);
}

TEST_F(BindCache, StrandsAreKeptToo)
{
  const auto body = fv::mesh::build(grid(6, 0));
  // two strands of three points over the body, one lying on it, one hanging away from it
  const std::vector<float>   points{1, 1, 0.5f, 2, 1, 0.5f, 3, 1, 0.5f, 1, 3, 0.5f, 1, 3, 3, 1, 3, 6};
  const std::vector<int32_t> lineStart{0, 3, 6}, lineVerts{0, 1, 2, 3, 4, 5};
  const auto fresh = fv::wardrobe::bindStrands(body.cagePositions, body.dsonTris, points, lineStart, lineVerts, 1.5f);
  const auto first = fv::wardrobe::cachedBindStrands(body.cagePositions, body.dsonTris, points, lineStart, lineVerts, 1.5f);
  const auto hits  = fv::wardrobe::bindCacheCounts().hits;
  const auto again = fv::wardrobe::cachedBindStrands(body.cagePositions, body.dsonTris, points, lineStart, lineVerts, 1.5f);
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().hits, hits + 1);
  expectSame(first, fresh);
  expectSame(again, fresh);
  // another reach is another binding
  const auto misses = fv::wardrobe::bindCacheCounts().misses;
  fv::wardrobe::cachedBindStrands(body.cagePositions, body.dsonTris, points, lineStart, lineVerts, 2.5f);
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().misses, misses + 1);
}

TEST_F(BindCache, EveryInputIsPartOfTheKey)
{
  const auto body = fv::mesh::build(grid(6, 0));
  const auto item = fv::mesh::build(grid(3, 0.5f));
  fv::wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);

  // each changed by one element: a miss, and the binding of what was asked for
  auto missFor = [&](std::vector<float> P, std::vector<uint32_t> T, std::vector<float> D, std::vector<int32_t> pairs,
                     const fv::wardrobe::BindFilter* filter) {
    const auto misses = fv::wardrobe::bindCacheCounts().misses;
    const auto got    = fv::wardrobe::cachedBindToSurface(P, T, D, pairs, filter);
    EXPECT_EQ(fv::wardrobe::bindCacheCounts().misses, misses + 1);
    expectSame(got, fv::wardrobe::bindToSurface(P, T, D, pairs, filter));
  };
  auto P = body.cagePositions;
  P[4]   = std::nextafter(P[4], 10.f);  // one body coordinate by one ulp
  missFor(P, body.dsonTris, item.cagePositions, {}, nullptr);
  auto T = body.dsonTris;
  std::swap(T[0], T[1]);  // one triangle's winding
  missFor(body.cagePositions, T, item.cagePositions, {}, nullptr);
  auto D = item.cagePositions;
  D[7] += 0.25f;  // one item coordinate
  missFor(body.cagePositions, body.dsonTris, D, {}, nullptr);
  missFor(body.cagePositions, body.dsonTris, item.cagePositions, {0, 0}, nullptr);  // a graft pair
  fv::wardrobe::BindFilter filter;
  filter.bodyLimb.assign(body.cagePositions.size() / 3, 0);
  filter.itemMask.assign(item.cagePositions.size() / 3, 0);
  missFor(body.cagePositions, body.dsonTris, item.cagePositions, {}, &filter);  // a filter that allows all
  filter.itemMask[2] = 1;
  missFor(body.cagePositions, body.dsonTris, item.cagePositions, {}, &filter);  // one item vertex's mask
}

TEST_F(BindCache, AFileThatDoesNotReadBackWholeIsBoundAgain)
{
  const auto body  = fv::mesh::build(grid(6, 0));
  const auto item  = fv::mesh::build(grid(3, 0.5f));
  const auto fresh = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  fv::wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  ASSERT_EQ(files(), 1u);
  const auto file = std::filesystem::directory_iterator(dir)->path();
  const auto size = std::filesystem::file_size(file);
  std::filesystem::resize_file(file, size - 4);

  const auto misses = fv::wardrobe::bindCacheCounts().misses;
  expectSame(fv::wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions), fresh);
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().misses, misses + 1);
  EXPECT_EQ(std::filesystem::file_size(file), size);  // written again whole

  // the same size with other bytes in the header's key: a miss too
  {
    std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(8);
    const char junk[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    f.write(junk, sizeof junk);
  }
  expectSame(fv::wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions), fresh);
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().misses, misses + 2);
}

TEST_F(BindCache, WithNoFolderItOnlyBinds)
{
  fv::wardrobe::setBindCacheDir({});
  const auto body   = fv::mesh::build(grid(6, 0));
  const auto item   = fv::mesh::build(grid(3, 0.5f));
  const auto before = fv::wardrobe::bindCacheCounts();
  expectSame(fv::wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions),
             fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions));
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().hits, before.hits);
  EXPECT_EQ(fv::wardrobe::bindCacheCounts().misses, before.misses);
  EXPECT_FALSE(std::filesystem::exists(dir));
}
