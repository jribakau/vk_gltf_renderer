#include <figure/materials/hair_strands.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <unordered_map>

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

// Smaller angle between two undirected axes, in 0..pi/2.
double axisDelta(double a, double b)
{
  double d = std::fmod(std::abs(a - b), std::numbers::pi);
  if(d > std::numbers::pi / 2)
    d = std::numbers::pi - d;
  return d;
}

// Each vertex's id among the distinct positions. Cards split per triangle, unindexed meshes and UV
// seams repeat a position under several indices, and an edge between them is still shared.
std::vector<uint32_t> weld(std::span<const float> position)
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
    size_t operator()(const Key& k) const
    {
      return size_t(k.x * 73856093LL) ^ size_t(k.y * 19349663LL) ^ size_t(k.z * 83492791LL);
    }
  };
  std::unordered_map<Key, uint32_t, Hash> seen;
  std::vector<uint32_t>                   ids(n);
  for(size_t i = 0; i < n; i++)
  {
    const Key key{int64_t(std::llround(position[i * 3] / q)), int64_t(std::llround(position[i * 3 + 1] / q)),
                  int64_t(std::llround(position[i * 3 + 2] / q))};
    auto [it, fresh] = seen.try_emplace(key, uint32_t(seen.size()));
    ids[i]           = it->second;
  }
  return ids;
}

}  // namespace

std::vector<StrandVote> strandVotes(const StrandMesh& part)
{
  const size_t vertexCount = part.position.size() / 3;
  const size_t count       = part.index.empty() ? vertexCount : part.index.size();
  std::vector<StrandMesh::Range> ranges = part.ranges;
  if(ranges.empty())
    ranges.push_back({0, uint32_t(count)});
  const std::vector<uint32_t> ids = weld(part.position);
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
  // edges by welded ends
  const auto edgeKey = [&](uint32_t a, uint32_t b) -> uint64_t {
    return ids[a] < ids[b] ? uint64_t(ids[a]) * vertexCount + ids[b] : uint64_t(ids[b]) * vertexCount + ids[a];
  };
  std::unordered_map<uint64_t, int> shared;
  for(size_t k = 0; k < tris.size(); k += 3)
    for(int e = 0; e < 3; e++)
      shared[edgeKey(tris[k + e], tris[k + (e + 1) % 3])]++;
  const auto pos = [&](uint32_t i) -> Vec3 { return {part.position[i * 3], part.position[i * 3 + 1], part.position[i * 3 + 2]}; };
  std::vector<StrandVote> votes;
  for(size_t k = 0; k < tris.size(); k += 3)
  {
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
      uses[e] = shared[edgeKey(corner[e], corner[(e + 1) % 3])];
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
