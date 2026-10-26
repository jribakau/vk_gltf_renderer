#include <figure/materials/hair_strands.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include <figure/jobs/pool.hpp>

namespace fv::mat {

namespace {

using Vec3 = std::array<double, 3>;

constexpr double kAgree = 20.0 * std::numbers::pi / 180.0;
// the longest edge has to beat the next by this much, or the polygon is not a card
constexpr double kEdgeRatio = 1.25;
// vertices closer than this share of the mesh's size are one vertex when edges are matched
constexpr double kWeld = 1e-6;

Vec3   sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3   scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3   cross(const Vec3& a, const Vec3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
double len(const Vec3& a) { return std::hypot(a[0], a[1], a[2]); }

struct Frame
{
  Vec3 t, b;
};

// The tangent frame the renderer writes for one triangle, and the bitangent its hit shader
// rebuilds from it. None when the triangle or its UVs are degenerate.
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

// theta whose shader direction (cos t, -sin t) lies along `edge` in the tangent frame.
std::optional<double> rotationOf(const Vec3& edge, const Frame& frame)
{
  const double tx = dot(edge, frame.t), ty = dot(edge, frame.b);
  if(std::hypot(tx, ty) < 1e-8)
    return std::nullopt;
  return std::atan2(-ty, tx);
}

// splitmix64's finaliser: a key's bits spread over the table
uint64_t mix(uint64_t x)
{
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// For each of `count` keys, the first index whose key is equal to it (so first[i] <= i). The keys are
// spread over shards by hash, each shard matched on a thread of its own in index order, which gives
// what one pass in order gives. hash(i) is key i's hash, same(i, j) whether two keys are equal.
template <class Hash, class Same>
std::vector<uint32_t> firstOfEqual(size_t count, const Hash& hash, const Same& same)
{
  constexpr int    kShardBits = 6;
  constexpr size_t kShards    = size_t(1) << kShardBits;
  constexpr size_t kChunk     = 65536;
  std::vector<uint64_t> h(count);
  // per chunk, its indices per shard, and then where they go
  std::vector<std::array<uint32_t, kShards>> at((count + kChunk - 1) / kChunk);
  jobs::parallelFor(count, kChunk, [&](size_t begin, size_t end) {
    auto& c = at[begin / kChunk];
    c.fill(0);
    for(size_t i = begin; i < end; i++)
    {
      h[i] = hash(i);
      c[h[i] >> (64 - kShardBits)]++;
    }
  });
  std::array<size_t, kShards + 1> shardStart{};
  for(size_t s = 0, running = 0; s < kShards; s++)
  {
    shardStart[s] = running;
    for(auto& c : at)
    {
      const uint32_t n = c[s];
      c[s]             = uint32_t(running);
      running += n;
    }
    shardStart[s + 1] = running;
  }
  // each shard's indices in order: chunk after chunk, each chunk's in order
  std::vector<uint32_t> order(count);
  jobs::parallelFor(count, kChunk, [&](size_t begin, size_t end) {
    auto& c = at[begin / kChunk];
    for(size_t i = begin; i < end; i++)
      order[c[h[i] >> (64 - kShardBits)]++] = uint32_t(i);
  });
  std::vector<uint32_t> first(count);
  jobs::parallelFor(kShards, 1, [&](size_t begin, size_t end) {
    std::vector<uint32_t> table;
    for(size_t s = begin; s < end; s++)
    {
      const size_t n    = shardStart[s + 1] - shardStart[s];
      size_t       size = 16;
      while(size < n * 2)
        size *= 2;
      table.assign(size, UINT32_MAX);
      const size_t mask = size - 1;
      for(size_t k = shardStart[s]; k < shardStart[s + 1]; k++)
      {
        const uint32_t i = order[k];
        for(size_t p = h[i] & mask;; p = (p + 1) & mask)
        {
          const uint32_t j = table[p];
          if(j == UINT32_MAX)
          {
            table[p] = first[i] = i;
            break;
          }
          if(h[j] == h[i] && same(j, i))
          {
            first[i] = j;
            break;
          }
        }
      }
    }
  });
  return first;
}

// Smaller angle between two undirected axes, in 0..pi/2.
double axisDelta(double a, double b)
{
  double d = std::fmod(std::abs(a - b), std::numbers::pi);
  if(d > std::numbers::pi / 2)
    d = std::numbers::pi - d;
  return d;
}

}  // namespace

// Each vertex's id among the distinct positions. Cards split per triangle, unindexed meshes and UV
// seams repeat a position under several indices, and an edge between them is still shared.
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
  std::vector<Key> keys(n);
  const auto       first = firstOfEqual(
      n,
      [&](size_t i) {
        Key& k = keys[i];
        k      = {int64_t(std::llround(position[i * 3] / q)), int64_t(std::llround(position[i * 3 + 1] / q)),
                  int64_t(std::llround(position[i * 3 + 2] / q))};
        return mix(uint64_t(k.x) + mix(uint64_t(k.y) + mix(uint64_t(k.z))));
      },
      [&](size_t a, size_t b) { return keys[a] == keys[b]; });
  // ids in the order the positions are first met
  std::vector<uint32_t> ids(n);
  uint32_t              next = 0;
  for(size_t i = 0; i < n; i++)
    ids[i] = first[i] == i ? next++ : ids[first[i]];
  return ids;
}

std::vector<StrandVote> strandVotes(const StrandMesh& part)
{
  return strandVotes(part, weldPositions(part.position));
}

std::vector<StrandVote> strandVotes(const StrandMesh& part, std::span<const uint32_t> ids)
{
  const size_t vertexCount = part.position.size() / 3;
  const size_t count       = part.index.empty() ? vertexCount : part.index.size();
  std::vector<StrandMesh::Range> ranges = part.ranges;
  if(ranges.empty())
    ranges.push_back({0, uint32_t(count)});
  const auto                  at  = [&](size_t i) -> uint32_t { return part.index.empty() ? uint32_t(i) : part.index[i]; };
  std::vector<uint32_t>       tris;
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
  // edges by welded ends, edge e of triangle t at 3t + e: how many triangles share each
  const auto edgeKey = [&](uint32_t a, uint32_t b) -> uint64_t {
    return ids[a] < ids[b] ? uint64_t(ids[a]) * vertexCount + ids[b] : uint64_t(ids[b]) * vertexCount + ids[a];
  };
  std::vector<uint64_t> edges(tris.size());
  const auto            firstEdge = firstOfEqual(
      edges.size(),
      [&](size_t i) {
        const size_t k = i - i % 3;
        edges[i]       = edgeKey(tris[i], tris[k + (i - k + 1) % 3]);
        return mix(edges[i]);
      },
      [&](size_t a, size_t b) { return edges[a] == edges[b]; });
  std::vector<int> shared(edges.size(), 0);
  for(uint32_t f : firstEdge)
    shared[f]++;
  const auto pos = [&](uint32_t i) -> Vec3 { return {part.position[i * 3], part.position[i * 3 + 1], part.position[i * 3 + 2]}; };
  // the triangles' votes in triangle order, counted in parallel chunks that each write their own
  // list (the edge counts are only read now); the lists concatenated are what one pass would make
  constexpr size_t                     kChunk   = 16384;  // triangles
  const size_t                         triCount = tris.size() / 3;
  std::vector<std::vector<StrandVote>> chunks((triCount + kChunk - 1) / kChunk);
  jobs::parallelFor(triCount, kChunk, [&](size_t begin, size_t end) {
    std::vector<StrandVote>& votes = chunks[begin / kChunk];
    for(size_t t = begin; t < end; t++)
    {
      const size_t   k         = t * 3;
      const uint32_t corner[3] = {tris[k], tris[k + 1], tris[k + 2]};
      const Vec3     p[3]      = {pos(corner[0]), pos(corner[1]), pos(corner[2])};
      const auto     frame     = tangentFrame(p, part.uv, corner);
      if(!frame)
        continue;
      // the candidates are the triangle's least-shared edges: once per card, or twice where a card is
      // doubled at the same place (both sides), while the diagonal and the steps of a strip count more
      int uses[3], least = 1 << 30;
      for(int e = 0; e < 3; e++)
      {
        uses[e] = shared[firstEdge[k + e]];
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
  });
  std::vector<StrandVote> votes;
  for(auto& c : chunks)
    votes.insert(votes.end(), c.begin(), c.end());
  return votes;
}

std::optional<float> strandRotation(const std::vector<StrandVote>& votes)
{
  double c = 0, s = 0, weight = 0;
  for(const auto& v : votes)
  {
    c += v.weight * std::cos(2.0 * v.theta);
    s += v.weight * std::sin(2.0 * v.theta);
    weight += v.weight;
  }
  if(weight <= 0)
    return std::nullopt;
  const double strand = 0.5 * std::atan2(s, c);
  double       agreed = 0;
  for(const auto& v : votes)
    if(axisDelta(v.theta, strand) <= kAgree)
      agreed += v.weight;
  if(agreed / weight < kStrandConfidence)
    return std::nullopt;
  // across the strand: a quarter turn
  double r = strand + std::numbers::pi / 2;
  if(r > std::numbers::pi / 2)
    r -= std::numbers::pi;
  return float(r);
}

std::optional<float> strandStrength(const Surface& s, float roughness, float glossyShare)
{
  // Iray (MDL base::anisotropy_conversion) keeps the roughness along its tangent and divides it by
  // (1 - anisotropy) on the other axis. The renderer takes the glTF roughness r to alpha = r^2 and
  // raises the anisotropy axis to lerp(alpha, 1, strength^2). Iray stretches only its glossy layer,
  // while the dual lobes folded into the same glTF lobe stay round, so the stretch is weighted by
  // the glossy layer's share of that lobe.
  const float anisotropy = s.glossyAnisotropy.value_or(kDefaultGlossyAnisotropy);
  if(!(anisotropy > 0) || !(glossyShare > 0))
    return std::nullopt;
  const float alpha = roughness * roughness;
  if(alpha >= 1)
    return std::nullopt;
  const float stretch  = 1 + glossyShare * (1 / std::max(1 - anisotropy, 1e-4f) - 1);
  const float across   = std::min(1.f, alpha * stretch);
  const float strength = std::sqrt((across - alpha) / (1 - alpha));
  return strength > 1e-3f ? std::optional<float>(strength) : std::nullopt;
}

}  // namespace fv::mat
