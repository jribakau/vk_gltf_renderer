// Catmull-Clark against OpenSubdiv's: the same refined vertices in the same order, the same quads
// and the same UVs. data/catmark.json holds the test cages of the TypeScript suite
// (test/subdiv/meshes.ts) with OpenSubdiv's own refinement of each (test/subdiv/reference.json).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>

#include <tinygltf/json.hpp>

#include <figure/subdiv/catmark.hpp>

using nlohmann::json;
using namespace fv::subdiv;

namespace {

json loadMeshes()
{
  std::ifstream f(std::string(FV_TEST_DATA) + "/catmark.json");
  EXPECT_TRUE(f.good());
  return json::parse(f);
}

struct Cage
{
  int                  vertexCount = 0;
  std::vector<int32_t> faceStart{0}, faceVerts, faceValues;
  std::vector<float>   values, positions;

  CageMesh mesh() const { return {vertexCount, faceStart, faceVerts, faceValues, values}; }
};

Cage cageOf(const json& m)
{
  Cage c;
  c.vertexCount = int(m["positions"].size());
  for(const auto& p : m["positions"])
    for(double x : p)
      c.positions.push_back(float(x));
  for(const auto& f : m["faces"])
  {
    for(int v : f["v"])
      c.faceVerts.push_back(v);
    for(int uv : f["uv"])
      c.faceValues.push_back(uv);
    c.faceStart.push_back(int32_t(c.faceVerts.size()));
  }
  for(const auto& uv : m["uvs"])
    for(double x : uv)
      c.values.push_back(float(x));
  return c;
}

BoundaryInterpolation boundaryOf(const json& m)
{
  return m["boundary"] == "edges-and-corners" ? BoundaryInterpolation::eEdgesAndCorners : BoundaryInterpolation::eEdges;
}

double worstDiff(const std::vector<float>& a, const std::vector<double>& b)
{
  EXPECT_EQ(a.size(), b.size());
  double worst = 0;
  for(size_t i = 0; i < std::min(a.size(), b.size()); i++)
    worst = std::max(worst, std::abs(double(a[i]) - b[i]));
  return worst;
}

}  // namespace

TEST(FigureCatmark, MatchesOpenSubdiv)
{
  const json meshes = loadMeshes();
  ASSERT_EQ(meshes.size(), 8u);
  for(const auto& m : meshes)
  {
    SCOPED_TRACE(m["name"].get<std::string>());
    const Cage       c   = cageOf(m);
    const Refinement r   = refine(c.mesh(), m["level"], boundaryOf(m));
    const json&      ref = m["reference"];
    // topology, in OpenSubdiv's vertex order
    EXPECT_EQ(r.vertexCount * 3, int(ref["positions"].size()));
    std::vector<int32_t> faces;
    for(const auto& f : ref["faces"])
      for(int v : f)
        faces.push_back(v);
    EXPECT_EQ(r.quads, faces);
    // positions
    EXPECT_LT(worstDiff(refineDense(r, c.positions, 3), ref["positions"].get<std::vector<double>>()), 1e-5);
    // UVs per quad corner
    std::vector<float> uv;
    for(int32_t v : r.quadValues)
    {
      uv.push_back(r.values[2 * v]);
      uv.push_back(r.values[2 * v + 1]);
    }
    std::vector<double> refUv;
    for(const auto& p : ref["uvs"])
      for(double x : p)
        refUv.push_back(x);
    EXPECT_LT(worstDiff(uv, refUv), 1e-5);
  }
}

TEST(FigureCatmark, SparseEqualsDense)
{
  // a 30 x 30 grid: a few listed vertices reach few refined rows, most of them reach most rows
  const int            n = 30;
  std::vector<int32_t> faceStart{0}, faceVerts;
  for(int j = 0; j < n; j++)
    for(int i = 0; i < n; i++)
    {
      for(int v : {j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i})
        faceVerts.push_back(v);
      faceStart.push_back(int32_t(faceVerts.size()));
    }
  const int        nv   = (n + 1) * (n + 1);
  const Refinement grid = refine({nv, faceStart, faceVerts, {}, {}}, 2);
  std::vector<uint32_t> most;
  for(int v = 0; v < nv; v++)
    if(v % 7)
      most.push_back(uint32_t(v));
  for(const auto& idx : {std::vector<uint32_t>{40, 41, 300, 900}, most})
  {
    std::vector<float> deltas, dense(size_t(nv) * 3, 0.f);
    for(uint32_t v : idx)
    {
      const float d[3] = {float(std::sin(double(v))), float(std::cos(double(v))), float(int(v % 5) - 2)};
      deltas.insert(deltas.end(), d, d + 3);
      std::copy(d, d + 3, dense.begin() + v * 3);
    }
    const auto full   = refineDense(grid, dense, 3);
    const auto sparse = refineSparse(grid, idx, deltas, 3);
    // listed in order, each once
    for(size_t i = 1; i < sparse.indices.size(); i++)
      EXPECT_GT(sparse.indices[i], sparse.indices[i - 1]);
    std::vector<float> back(full.size(), 0.f);
    for(size_t i = 0; i < sparse.indices.size(); i++)
      std::copy_n(sparse.data.begin() + i * 3, 3, back.begin() + sparse.indices[i] * 3);
    double worst = 0;
    for(size_t i = 0; i < full.size(); i++)
      worst = std::max(worst, std::abs(double(back[i]) - full[i]));
    EXPECT_LT(worst, 1e-6);
  }
}

TEST(FigureCatmark, EachParentOncePerRowAndQuadsKnowTheirFace)
{
  for(const auto& m : loadMeshes())
  {
    SCOPED_TRACE(m["name"].get<std::string>());
    const Cage       c = cageOf(m);
    const Refinement r = refine(c.mesh(), m["level"], boundaryOf(m));
    for(const Stencils& s : r.levels)
      for(int row = 0; row < s.rows(); row++)
      {
        std::set<int32_t> cols(s.col.begin() + s.rowStart[row], s.col.begin() + s.rowStart[row + 1]);
        EXPECT_EQ(int(cols.size()), s.rowStart[row + 1] - s.rowStart[row]);
      }
    EXPECT_EQ(r.quadParent.size() * 4, r.quads.size());
    EXPECT_EQ(*std::max_element(r.quadParent.begin(), r.quadParent.end()), int(m["faces"].size()) - 1);
  }
}
