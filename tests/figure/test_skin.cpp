// Skin weights per drawn vertex, worked a chunk of rows at a time side by side (a hair has millions):
// the indices and weights are the one-pass loop's, bit for bit, on a subdivided mesh with a seam, rows
// past four and past eight influences, and vertices with none.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <figure/dson/rig.hpp>
#include <figure/dson/store.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/subdiv/catmark.hpp>

namespace {

struct Influences
{
  std::vector<int32_t> start, bone;
  std::vector<double>  weight;
};

void strongestFirst(Influences& inf, int from, int to)
{
  for(int i = from + 1; i < to; i++)
  {
    const int32_t b = inf.bone[i];
    const double  w = inf.weight[i];
    int           j = i - 1;
    for(; j >= from && inf.weight[j] < w; j--)
    {
      inf.bone[j + 1]   = inf.bone[j];
      inf.weight[j + 1] = inf.weight[j];
    }
    inf.bone[j + 1]   = b;
    inf.weight[j + 1] = w;
  }
}

// mesh::skin as it was, one pass over the rows
fv::mesh::SkinOutput serialSkin(const fv::mesh::BuiltMesh& mesh, const fv::mesh::SkinInput& s)
{
  const int  rows = int(s.start.size()) - 1;
  Influences inf;
  inf.start.assign(size_t(std::max(rows, 0)) + 1, 0);
  int bones = s.rootBone + 1;
  for(int v = 0; v < rows; v++)
  {
    for(int k = s.start[v]; k < s.start[v + 1]; k++)
      if(s.weight[k] > 0)
      {
        inf.bone.push_back(s.bone[k]);
        inf.weight.push_back(s.weight[k]);
        bones = std::max(bones, s.bone[k] + 1);
      }
    inf.start[v + 1] = int32_t(inf.bone.size());
  }
  if(mesh.refinement)
    for(const auto& lv : mesh.refinement->levels)
    {
      const int  n = lv.rows();
      Influences out;
      out.start.assign(size_t(n) + 1, 0);
      std::vector<double>  acc(bones, 0.0);
      std::vector<int32_t> seen(bones, -1), order(bones, 0);
      for(int row = 0; row < n; row++)
      {
        int met = 0;
        for(int k = lv.rowStart[row]; k < lv.rowStart[row + 1]; k++)
          for(int j = inf.start[lv.col[k]]; j < inf.start[lv.col[k] + 1]; j++)
          {
            const int b = inf.bone[j];
            if(seen[b] != row)
            {
              seen[b]      = row;
              acc[b]       = 0;
              order[met++] = b;
            }
            acc[b] += inf.weight[j] * double(lv.weight[k]);
          }
        const int from = int(out.bone.size());
        for(int i = 0; i < met; i++)
          if(acc[order[i]] > 1e-6)
          {
            out.bone.push_back(order[i]);
            out.weight.push_back(acc[order[i]]);
          }
        if(int(out.bone.size()) - from > 8)
        {
          strongestFirst(out, from, int(out.bone.size()));
          out.bone.resize(from + 8);
          out.weight.resize(from + 8);
        }
        out.start[row + 1] = int32_t(out.bone.size());
      }
      inf = std::move(out);
    }
  const size_t         count = mesh.renderIndex.size();
  fv::mesh::SkinOutput out;
  out.skinIndex.assign(count * 4, 0);
  out.skinWeight.assign(count * 4, 0.f);
  std::vector<uint8_t> sorted(inf.start.size(), 0);
  for(size_t i = 0; i < count; i++)
  {
    const uint32_t v    = mesh.renderIndex[i];
    const bool     has  = v + 1 < inf.start.size();
    const int      from = has ? inf.start[v] : 0;
    const int      n    = has ? inf.start[v + 1] - from : 0;
    if(n > 4 && !sorted[v])
    {
      strongestFirst(inf, from, from + n);
      sorted[v] = 1;
    }
    const int m   = std::min(4, n);
    double    sum = 0;
    for(int k = 0; k < m; k++)
      sum += inf.weight[from + k];
    if(sum <= 0)
    {
      out.skinIndex[i * 4]  = uint16_t(s.rootBone);
      out.skinWeight[i * 4] = 1;
      continue;
    }
    for(int k = 0; k < m; k++)
    {
      out.skinIndex[i * 4 + k]  = uint16_t(inf.bone[from + k]);
      out.skinWeight[i * 4 + k] = float(inf.weight[from + k] / sum);
    }
  }
  return out;
}

}  // namespace

TEST(Skin, MadeSideBySideItIsTheOnePassResultBitForBit)
{
  // a 120 x 120 grid at SubD 2 with a seam down column 40
  constexpr int       n = 120;
  fv::mesh::MeshInput in;
  for(int y = 0; y <= n; y++)
    for(int x = 0; x <= n; x++)
      in.positions.insert(in.positions.end(), {float(x), float(y), 0.f});
  in.polyStart.push_back(0);
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      const int a = y * (n + 1) + x;
      in.polyVerts.insert(in.polyVerts.end(), {a, a + 1, a + n + 2, a + n + 1});
      in.polyStart.push_back(int(in.polyVerts.size()));
      in.polyMaterial.push_back(0);
    }
  in.uvs.assign(size_t(n + 1) * (n + 1) * 2, 0.f);
  for(int y = 0; y < n; y++)
    for(int v : {y * (n + 1) + 40, (y + 1) * (n + 1) + 40})
    {
      in.overrides.insert(in.overrides.end(), {y * n + 40, v, int(in.uvs.size() / 2)});
      in.uvs.insert(in.uvs.end(), {0.5f, float(y) / n});
    }
  in.level        = 2;
  const auto mesh = fv::mesh::build(in);
  ASSERT_GT(mesh.renderIndex.size(), size_t(mesh.renderVertexCount));

  // 40 bones: each vertex weighted to a few that change across the grid, so subdivided rows meet
  // more than eight; a corner left with none, and some zero weights
  fv::mesh::SkinInput s;
  s.rootBone = 3;
  s.start.push_back(0);
  for(int v = 0; v < (n + 1) * (n + 1); v++)
  {
    const int x = v % (n + 1), y = v / (n + 1);
    if(x > 100 && y > 100)
    {
      s.start.push_back(int32_t(s.bone.size()));
      continue;
    }
    for(int k = 0; k < 3 + (v % 3); k++)
    {
      s.bone.push_back((x / 7 + y / 5 + k * 3) % 40);
      s.weight.push_back(k == 1 && v % 11 == 0 ? 0.f : float(1 + std::sin(double(v + k))) / float(k + 1));
    }
    s.start.push_back(int32_t(s.bone.size()));
  }

  const auto got = fv::mesh::skin(mesh, s), want = serialSkin(mesh, s);
  ASSERT_EQ(got.skinIndex.size(), want.skinIndex.size());
  EXPECT_EQ(got.skinIndex, want.skinIndex);
  ASSERT_EQ(got.skinWeight.size(), want.skinWeight.size());
  EXPECT_EQ(std::memcmp(got.skinWeight.data(), want.skinWeight.data(), got.skinWeight.size() * sizeof(float)), 0);
}

// DAZ's skinning (gpu-figures-plan.md F3): up to eight influences a drawn vertex, summing to 1, whose
// first four renormalised are glTF's four; a dual quaternion map is refined as positions are (a map
// linear across the grid stays so).
TEST(Skin, EightInfluencesAndADualQuaternionMapFollowTheRefinement)
{
  constexpr int       n = 6;
  fv::mesh::MeshInput in;
  for(int y = 0; y <= n; y++)
    for(int x = 0; x <= n; x++)
      in.positions.insert(in.positions.end(), {float(x), float(y), 0.f});
  in.polyStart.push_back(0);
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      const int a = y * (n + 1) + x;
      in.polyVerts.insert(in.polyVerts.end(), {a, a + 1, a + n + 2, a + n + 1});
      in.polyStart.push_back(int(in.polyVerts.size()));
      in.polyMaterial.push_back(0);
    }
  in.level        = 1;
  const auto mesh = fv::mesh::build(in);

  fv::mesh::SkinInput s;
  s.start.push_back(0);
  for(int v = 0; v < (n + 1) * (n + 1); v++)
  {
    for(int k = 0; k < (v % 2 ? 6 : 2); k++)
    {
      s.bone.push_back((v + k) % 10);
      s.weight.push_back(1.f + float(k));
    }
    s.start.push_back(int32_t(s.bone.size()));
    s.dualQuatMap.push_back(float(v % (n + 1)) / n);
  }
  s.dualQuat = 1.f;
  const auto out   = fv::mesh::skin(mesh, s);
  const size_t count = mesh.renderIndex.size();
  ASSERT_EQ(out.skinIndex8.size(), count * 8);
  ASSERT_EQ(out.skinWeight8.size(), count * 8);
  size_t more = 0;
  for(size_t i = 0; i < count; i++)
  {
    double sum = 0, first4 = 0;
    for(int k = 0; k < 8; k++)
      sum += out.skinWeight8[i * 8 + k];
    for(int k = 0; k < 4; k++)
      first4 += out.skinWeight8[i * 8 + k];
    EXPECT_NEAR(sum, 1.0, 1e-5) << i;
    more += out.skinWeight8[i * 8 + 4] > 0;
    for(int k = 0; k < 4; k++)
    {
      EXPECT_EQ(out.skinIndex[i * 4 + k], out.skinIndex8[i * 8 + k]) << i;
      EXPECT_NEAR(out.skinWeight[i * 4 + k], out.skinWeight8[i * 8 + k] / first4, 1e-5) << i;
    }
  }
  EXPECT_GT(more, 0u);
  ASSERT_EQ(out.dualQuatShare.size(), count);
  for(size_t i = 0; i < count; i++)
    EXPECT_NEAR(out.dualQuatShare[i], mesh.position[i * 3] / n, 1e-5) << i;

  // four at most: the eight are not kept
  fv::mesh::SkinInput few;
  few.start = {0};
  for(int v = 0; v < (n + 1) * (n + 1); v++)
  {
    few.bone.push_back(v % 3);
    few.weight.push_back(1.f);
    few.start.push_back(int32_t(few.bone.size()));
  }
  EXPECT_TRUE(fv::mesh::skin(mesh, few).skinIndex8.empty());
}

// A skin modifier's settings (Genesis 3, 8 and 9 files): DualQuat is all dual quaternion; a blend map
// blends it with linear, kept as one value when uniform (Genesis 9's is 0.5 everywhere).
TEST(Skin, TheSkinSettingsGiveTheDualQuaternionShare)
{
  fv::dson::Rig rig;
  rig.root  = 0;
  rig.bones = {{"hip", 0}, {"thigh", 1}};
  const auto binding = [&](const std::string& extra) {
    const auto doc = fv::dson::parseDoc(R"({"skin": {"joints": [
        {"id": "hip", "node": "#hip", "node_weights": {"count": 3, "values": [[0, 1], [1, 0.5], [2, 1]]}},
        {"id": "thigh", "node": "#thigh", "node_weights": {"count": 1, "values": [[1, 0.5]]}}]})"
                                        + extra + "}");
    return fv::dson::skinBinding(*doc, doc->json, rig, 3);
  };
  const auto linear = binding("");
  ASSERT_TRUE(linear);
  EXPECT_EQ(linear->input.dualQuat, 0.f);
  EXPECT_TRUE(linear->input.dualQuatMap.empty());
  const auto dq = binding(R"(, "extra": [{"type": "skin_settings", "general_map_mode": "DualQuat"}])");
  EXPECT_EQ(dq->input.dualQuat, 1.f);
  const auto half = binding(R"(, "extra": [{"type": "skin_settings", "general_map_mode": "DualQuat",
      "blend_mode": "BlendLinearDualQuat", "blend_weights": {"count": 3, "values": [[0, 0.5], [1, 0.5], [2, 0.5]]}}])");
  EXPECT_EQ(half->input.dualQuat, 0.5f);
  EXPECT_TRUE(half->input.dualQuatMap.empty());
  const auto mapped = binding(R"(, "extra": [{"type": "skin_settings", "general_map_mode": "DualQuat",
      "blend_mode": "BlendLinearDualQuat", "blend_weights": {"count": 2, "values": [[0, 0.25], [2, 1]]}}])");
  EXPECT_EQ(mapped->input.dualQuatMap, (std::vector<float>{0.25f, 0.f, 1.f}));
}

// Strand hair with no joints (FE Shawl, Pixie Cut) follows the figure in DAZ Studio through a dForce
// modifier. Left without a skin it stayed where the file puts it, whatever the figure's pose or shape.
TEST(Skin, StrandsWithoutJointsRideTheBoneTheirGroupNames)
{
  fv::dson::Rig rig;
  rig.root  = 0;
  rig.bones = {{"hip", 0}, {"neck2", 1}, {"head", 2}};
  const auto binding = [&](const std::string& joints, const std::string& map, const std::string& polys = "[]") {
    const auto doc = fv::dson::parseDoc(R"({"geo": {"polygon_groups": {"values": ["default", "head", "neck2", "other"]},
        "polylist": {"values": )" + polys + R"(},
        "polyline_list": {"values": [[1, 0, 0, 1], [2, 0, 2, 3], [3, 0, 4, 5]]}},
        "mod": {"skin": {"joints": )" + joints + R"(, "selection_map": [{"id": "m", "mappings": )" + map + R"(}]}}})");
    return fv::dson::strandBinding(*doc, doc->json.at("mod"), doc->json.at("geo"), rig, 7);
  };
  const auto bound = binding("[]", R"([["head", "head"], ["neck2", "neck2"], ["other", "no such bone"]])");
  ASSERT_TRUE(bound);
  // one full influence per vertex: the head's strand, the neck's, the head for a group it can't place,
  // and the head for a vertex on no strand
  ASSERT_EQ(bound->input.start, (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  const auto boneOf = [&](int v) { return bound->order[size_t(bound->input.bone[size_t(v)])]; };
  EXPECT_EQ(std::vector<int>({boneOf(0), boneOf(1)}), std::vector<int>({2, 2}));
  EXPECT_EQ(std::vector<int>({boneOf(2), boneOf(3)}), std::vector<int>({1, 1}));
  EXPECT_EQ(std::vector<int>({boneOf(4), boneOf(5), boneOf(6)}), std::vector<int>({2, 2, 2}));
  EXPECT_EQ(bound->input.weight, std::vector<float>(7, 1.f));

  // weights in the file are the file's business, and a mesh with polygons isn't strand hair
  EXPECT_FALSE(binding(R"([{"id": "head", "node": "#head"}])", "[]"));
  EXPECT_FALSE(binding("[]", "[]", "[[0, 0, 0, 1, 2]]"));
}
