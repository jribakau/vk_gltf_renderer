// Hair cards' strand votes: the weld and the edge counts, matched in parallel shards, give exactly
// what the one serial pass they replaced gave, on a mesh big enough to cross every chunk and shard.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include <figure/materials/hair_strands.hpp>

namespace {

// The serial weld and votes as they were (hair_strands.cpp before the shards), kept as the reference.
namespace ref {

using Vec3 = std::array<double, 3>;
constexpr double kEdgeRatio = 1.25;
constexpr double kWeld      = 1e-6;

Vec3   sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3   scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3   cross(const Vec3& a, const Vec3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
double len(const Vec3& a) { return std::hypot(a[0], a[1], a[2]); }

struct Frame
{
  Vec3 t, b;
};

std::optional<Frame> tangentFrame(const Vec3 p[3], std::span<const float> uv, const uint32_t i[3])
{
  const Vec3   e1 = sub(p[1], p[0]), e2 = sub(p[2], p[0]);
  const Vec3   nraw = cross(e1, e2);
  const double nlen = len(nraw);
  if(nlen < 1e-12)
    return std::nullopt;
  const Vec3   n   = scale(nraw, 1 / nlen);
  const double d1x = double(uv[i[1] * 2]) - uv[i[0] * 2], d1y = double(uv[i[1] * 2 + 1]) - uv[i[0] * 2 + 1];
  const double d2x = double(uv[i[2] * 2]) - uv[i[0] * 2], d2y = double(uv[i[2] * 2 + 1]) - uv[i[0] * 2 + 1];
  const double det = d1x * d2y - d2x * d1y;
  if(std::abs(det) < 1e-12)
    return std::nullopt;
  const double f         = 1 / det;
  const Vec3   tangent   = scale(sub(scale(e1, d2y), scale(e2, d1y)), f);
  const Vec3   bitangent = scale(sub(scale(e1, d2x), scale(e2, d1x)), f);
  const double w         = dot(cross(tangent, bitangent), n) > 0 ? 1 : -1;
  const Vec3   ortho     = sub(tangent, scale(n, dot(n, tangent)));
  const double ol        = len(ortho);
  if(ol < 1e-8)
    return std::nullopt;
  const Vec3 t = scale(ortho, 1 / ol);
  return Frame{t, scale(cross(n, t), w)};
}

std::optional<double> rotationOf(const Vec3& edge, const Frame& frame)
{
  const double tx = dot(edge, frame.t), ty = dot(edge, frame.b);
  if(std::hypot(tx, ty) < 1e-8)
    return std::nullopt;
  return std::atan2(-ty, tx);
}

std::vector<uint32_t> weldPositions(std::span<const float> position)
{
  const size_t n = position.size() / 3;
  float        lo = INFINITY, hi = -INFINITY;
  for(size_t i = 0; i < n * 3; i++)
  {
    lo = std::min(lo, position[i]);
    hi = std::max(hi, position[i]);
  }
  const double q = std::max(double(hi) - lo, 1e-12) * kWeld;
  struct Key
  {
    int64_t x, y, z;
    bool    operator==(const Key&) const = default;
  };
  struct Hash
  {
    size_t operator()(const Key& k) const { return size_t(k.x * 73856093LL) ^ size_t(k.y * 19349663LL) ^ size_t(k.z * 83492791LL); }
  };
  std::unordered_map<Key, uint32_t, Hash> seen;
  std::vector<uint32_t>                   ids(n);
  for(size_t i = 0; i < n; i++)
  {
    const Key key{int64_t(std::llround(position[i * 3] / q)), int64_t(std::llround(position[i * 3 + 1] / q)),
                  int64_t(std::llround(position[i * 3 + 2] / q))};
    ids[i] = seen.try_emplace(key, uint32_t(seen.size())).first->second;
  }
  return ids;
}

std::vector<fv::mat::StrandVote> strandVotes(const fv::mat::StrandMesh& part, std::span<const uint32_t> ids)
{
  const size_t vertexCount = part.position.size() / 3;
  const size_t count       = part.index.empty() ? vertexCount : part.index.size();
  auto         ranges      = part.ranges;
  if(ranges.empty())
    ranges.push_back({0, uint32_t(count)});
  const auto            at = [&](size_t i) -> uint32_t { return part.index.empty() ? uint32_t(i) : part.index[i]; };
  std::vector<uint32_t> tris;
  for(const auto& range : ranges)
  {
    const size_t end = std::min<size_t>(size_t(range.start) + range.count, count);
    for(size_t i = range.start; i + 2 < end; i += 3)
    {
      const uint32_t i0 = at(i), i1 = at(i + 1), i2 = at(i + 2);
      if(ids[i0] == ids[i1] || ids[i1] == ids[i2] || ids[i2] == ids[i0])
        continue;
      tris.insert(tris.end(), {i0, i1, i2});
    }
  }
  const auto edgeKey = [&](uint32_t a, uint32_t b) -> uint64_t {
    return ids[a] < ids[b] ? uint64_t(ids[a]) * vertexCount + ids[b] : uint64_t(ids[b]) * vertexCount + ids[a];
  };
  std::unordered_map<uint64_t, int> shared;
  for(size_t k = 0; k < tris.size(); k += 3)
    for(int e = 0; e < 3; e++)
      shared[edgeKey(tris[k + e], tris[k + (e + 1) % 3])]++;
  const auto pos = [&](uint32_t i) -> Vec3 { return {part.position[i * 3], part.position[i * 3 + 1], part.position[i * 3 + 2]}; };
  std::vector<fv::mat::StrandVote> votes;
  for(size_t k = 0; k < tris.size(); k += 3)
  {
    const uint32_t corner[3] = {tris[k], tris[k + 1], tris[k + 2]};
    const Vec3     p[3]      = {pos(corner[0]), pos(corner[1]), pos(corner[2])};
    const auto     frame     = tangentFrame(p, part.uv, corner);
    if(!frame)
      continue;
    int uses[3], least = 1 << 30;
    for(int e = 0; e < 3; e++)
    {
      uses[e] = shared.find(edgeKey(corner[e], corner[(e + 1) % 3]))->second;
      least   = std::min(least, uses[e]);
    }
    bool   haveBest = false;
    double bestLen = 0, second = 0;
    Vec3   bestEdge{};
    for(int e = 0; e < 3; e++)
    {
      if(uses[e] != least)
        continue;
      const uint32_t a = corner[e], b = corner[(e + 1) % 3];
      const Vec3     edge = sub(p[(e + 1) % 3], p[e]);
      const double   l    = len(edge);
      if(l <= 1e-8 || std::hypot(double(part.uv[b * 2]) - part.uv[a * 2], double(part.uv[b * 2 + 1]) - part.uv[a * 2 + 1]) <= 1e-8)
        continue;
      if(!haveBest || l > bestLen)
      {
        second   = haveBest ? bestLen : 0;
        haveBest = true;
        bestLen  = l;
        bestEdge = edge;
      }
      else if(l > second)
        second = l;
    }
    if(!haveBest || bestLen < second * kEdgeRatio)
      continue;
    if(const auto theta = rotationOf(bestEdge, *frame))
      votes.push_back({float(bestLen), float(*theta)});
  }
  return votes;
}

}  // namespace ref

// Hair cards as a hair's mesh has them: strips of quads along random directions, some cut per
// triangle (every corner its own vertex), some doubled in place (both sides), UV seams that repeat a
// position, and a few collapsed triangles.
struct Cards
{
  std::vector<float>    position, uv;
  std::vector<uint32_t> index;
};

Cards cards(int count, uint32_t seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  Cards                                 c;
  const auto vertex = [&](const std::array<float, 3>& p, float s, float t) {
    c.position.insert(c.position.end(), p.begin(), p.end());
    c.uv.insert(c.uv.end(), {s, t});
    return uint32_t(c.position.size() / 3 - 1);
  };
  for(int card = 0; card < count; card++)
  {
    const std::array<float, 3> o{u(rng) * 40 - 20, u(rng) * 40 + 140, u(rng) * 30 - 15};
    const float                yaw = u(rng) * 6.283f, pitch = u(rng) * 3.f - 1.5f;
    const std::array<float, 3> along{std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw)};
    const std::array<float, 3> across{-std::sin(yaw), 0.f, std::cos(yaw)};
    const float                length = 2 + u(rng) * 10, width = 0.2f + u(rng) * (card % 7 == 0 ? 3 * length : 1.f);
    const int                  segments = 1 + int(u(rng) * 4);
    const bool                 split = card % 3 == 0, doubled = card % 5 == 0, seam = card % 4 == 1;
    const float                s0 = u(rng), t0 = u(rng);
    const bool                 uvAlong = card % 2 == 0;
    const auto at = [&](int i, int side) {
      const float          a = length * i / segments, b = width * side;
      std::array<float, 3> p{};
      for(int k = 0; k < 3; k++)
        p[k] = o[k] + along[k] * a + across[k] * b;
      const float s = s0 + 0.1f * (uvAlong ? float(side) : float(i) / segments);
      const float t = t0 + 0.1f * (uvAlong ? float(i) / segments : float(side));
      return std::pair{p, std::array<float, 2>{s, t}};
    };
    for(int copy = 0; copy < (doubled ? 2 : 1); copy++)
    {
      std::vector<uint32_t> shared;
      if(!split)
        for(int i = 0; i <= segments; i++)
          for(int side = 0; side < 2; side++)
          {
            const auto [p, st] = at(i, side);
            shared.push_back(vertex(p, st[0], st[1]));
          }
      for(int i = 0; i < segments; i++)
      {
        const int quad[4][2] = {{i, 0}, {i, 1}, {i + 1, 1}, {i + 1, 0}};
        for(const auto& tri : {std::array{0, 1, 2}, std::array{0, 2, 3}})
          for(int k = 0; k < 3; k++)
          {
            const int corner = tri[copy ? 2 - k : k];
            const int qi = quad[corner][0], qs = quad[corner][1];
            if(split || (seam && qi == segments))
            {
              auto [p, st] = at(qi, qs);
              c.index.push_back(vertex(p, st[0] + (seam ? 0.5f : 0.f), st[1]));
            }
            else
              c.index.push_back(shared[size_t(qi) * 2 + qs]);
          }
      }
    }
    if(card % 97 == 0)  // a collapsed triangle
    {
      const auto [p, st] = at(0, 0);
      const uint32_t v   = vertex(p, st[0], st[1]);
      c.index.insert(c.index.end(), {v, v, vertex(p, st[0], st[1] + 0.1f)});
    }
  }
  return c;
}

void expectSameVotes(const std::vector<fv::mat::StrandVote>& got, const std::vector<fv::mat::StrandVote>& want)
{
  ASSERT_EQ(got.size(), want.size());
  for(size_t i = 0; i < got.size(); i++)
    ASSERT_TRUE(std::memcmp(&got[i], &want[i], sizeof got[i]) == 0)
        << "vote " << i << ": " << got[i].weight << ", " << got[i].theta << " vs " << want[i].weight << ", " << want[i].theta;
}

}  // namespace

TEST(HairStrands, TheWeldIsTheSerialOneWithIdsInFirstSeenOrder)
{
  const Cards c   = cards(40000, 7);
  const auto  ids = fv::mat::weldPositions(c.position);
  EXPECT_GT(c.position.size() / 3, size_t(4 * 65536));  // more than a few chunks
  EXPECT_EQ(ids, ref::weldPositions(c.position));
  // a lone vertex and none
  EXPECT_EQ(fv::mat::weldPositions(std::vector<float>{1, 2, 3}), std::vector<uint32_t>{0});
  EXPECT_TRUE(fv::mat::weldPositions({}).empty());
}

TEST(HairStrands, TheVotesAreTheSerialOnesBitForBit)
{
  const Cards  c = cards(40000, 11);
  const auto   ids = ref::weldPositions(c.position);
  const size_t half = c.index.size() / 6 * 3;
  // the whole mesh, two material ranges of it, and the unindexed corners
  for(const std::vector<fv::mat::StrandMesh::Range>& ranges :
      {std::vector<fv::mat::StrandMesh::Range>{}, {{0, uint32_t(half)}, {uint32_t(half + 300), uint32_t(c.index.size() - half - 300)}}})
  {
    const fv::mat::StrandMesh part{c.position, c.uv, c.index, ranges};
    const auto                want = ref::strandVotes(part, ids);
    EXPECT_GT(want.size(), 10000u);
    expectSameVotes(fv::mat::strandVotes(part, ids), want);
    expectSameVotes(fv::mat::strandVotes(part), want);
  }
  const fv::mat::StrandMesh unindexed{c.position, c.uv, {}, {}};
  expectSameVotes(fv::mat::strandVotes(unindexed), ref::strandVotes(unindexed, ids));
  // the cards agree: what the rotation is made from is the same too
  EXPECT_EQ(fv::mat::strandRotation(fv::mat::strandVotes({c.position, c.uv, c.index, {}})),
            fv::mat::strandRotation(ref::strandVotes({c.position, c.uv, c.index, {}}, ids)));
}
