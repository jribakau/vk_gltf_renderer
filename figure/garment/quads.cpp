#include <figure/garment/quads.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>

namespace fv::garment {

namespace {

using P3 = std::array<double, 3>;

P3 sub(const P3& a, const P3& b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
P3 cross(const P3& a, const P3& b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const P3& a, const P3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
// The cosine of the angle between two directions; -1 (opposed) when either has no length.
double cosine(const P3& a, const P3& b)
{
  const double l = std::sqrt(dot(a, a) * dot(b, b));
  return l > 0 ? std::clamp(dot(a, b) / l, -1.0, 1.0) : -1.0;
}

}  // namespace

Polys pairQuads(std::span<const float> positions, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts, const QuadOptions& options)
{
  const size_t  nv    = positions.size() / 3;
  const size_t  polys = polyStart.empty() ? 0 : polyStart.size() - 1;
  const auto    at    = [&](int32_t v) { return P3{positions[size_t(v) * 3], positions[size_t(v) * 3 + 1], positions[size_t(v) * 3 + 2]}; };
  const auto    triangle = [&](size_t p) {
    if(polyStart[p + 1] - polyStart[p] != 3)
      return false;
    for(int32_t k = polyStart[p]; k < polyStart[p + 1]; k++)
      if(polyVerts[size_t(k)] < 0 || size_t(polyVerts[size_t(k)]) >= nv)
        return false;
    return true;
  };
  // each directed edge of a triangle: the triangle, and which of its corners the edge starts at. An edge
  // walked the same way twice (three faces on it, or two wound against each other) pairs nothing
  struct Half
  {
    int32_t poly = -1;
    int     corner = 0;
    bool    twice = false;
  };
  std::unordered_map<uint64_t, Half> halves;
  const auto key = [](int32_t a, int32_t b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); };
  Polys      out;
  for(size_t p = 0; p < polys; p++)
  {
    if(!triangle(p))
      continue;
    out.triangles++;
    for(int c = 0; c < 3; c++)
    {
      const int32_t a = polyVerts[size_t(polyStart[p] + c)], b = polyVerts[size_t(polyStart[p] + (c + 1) % 3)];
      const auto [it, added] = halves.emplace(key(a, b), Half{int32_t(p), c, false});
      if(!added)
        it->second.twice = true;
    }
  }
  const double radians = 3.14159265358979323846 / 180;
  const double flat = std::cos(options.flat * radians), square = std::sin(options.square * radians);
  struct Pair
  {
    double  cost;
    int32_t p, q;
    std::array<int32_t, 4> quad;
  };
  std::vector<Pair> pairs;
  for(const auto& [k, h] : halves)
  {
    const int32_t a = int32_t(k >> 32), b = int32_t(uint32_t(k));
    if(a > b || h.twice)
      continue;
    const auto other = halves.find(key(b, a));
    if(other == halves.end() || other->second.twice || other->second.poly == h.poly)
      continue;
    // triangle (a, b, c) and, across a-b, (b, a, d): the quad c, a, d, b winds as they do
    const int32_t c = polyVerts[size_t(polyStart[size_t(h.poly)] + (h.corner + 2) % 3)];
    const int32_t d = polyVerts[size_t(polyStart[size_t(other->second.poly)] + (other->second.corner + 2) % 3)];
    if(c == d)
      continue;
    const P3     pa = at(a), pb = at(b), pc = at(c), pd = at(d);
    const P3     n1 = cross(sub(pb, pa), sub(pc, pa)), n2 = cross(sub(pa, pb), sub(pd, pb));
    const double fold = cosine(n1, n2);
    if(fold < flat)
      continue;
    // how far the worst corner is from a right angle, as the sine of that
    // (each corner written out: as a loop over the four by index, this compiler's optimised build found a
    // straight corner in every quad, and right ones once the loop printed them)
    const double atC = cosine(sub(pb, pc), sub(pa, pc)), atA = cosine(sub(pc, pa), sub(pd, pa));
    const double atD = cosine(sub(pa, pd), sub(pb, pd)), atB = cosine(sub(pd, pb), sub(pc, pb));
    const double off = std::max(std::max(std::abs(atC), std::abs(atA)), std::max(std::abs(atD), std::abs(atB)));
    if(off > square)
      continue;
    pairs.push_back({off + (1 - fold), h.poly, other->second.poly, {c, a, d, b}});
  }
  // the best first; alike, by where they are in the mesh, so the same mesh pairs the same way every time
  std::sort(pairs.begin(), pairs.end(), [](const Pair& x, const Pair& y) {
    if(x.cost != y.cost)
      return x.cost < y.cost;
    return std::pair(x.p, x.q) < std::pair(y.p, y.q);
  });
  std::vector<int32_t> quadOf(polys, -1);  // a triangle's pair, into `pairs`
  for(size_t i = 0; i < pairs.size(); i++)
    if(quadOf[size_t(pairs[i].p)] < 0 && quadOf[size_t(pairs[i].q)] < 0)
      quadOf[size_t(pairs[i].p)] = quadOf[size_t(pairs[i].q)] = int32_t(i);
  // in the polygons' own order: a quad where its first triangle was
  for(size_t p = 0; p < polys; p++)
  {
    if(quadOf[p] < 0)
    {
      out.verts.insert(out.verts.end(), polyVerts.begin() + polyStart[p], polyVerts.begin() + polyStart[p + 1]);
      out.start.push_back(int32_t(out.verts.size()));
      continue;
    }
    const Pair& pair = pairs[size_t(quadOf[p])];
    if(size_t(std::min(pair.p, pair.q)) != p)
      continue;
    out.verts.insert(out.verts.end(), pair.quad.begin(), pair.quad.end());
    out.start.push_back(int32_t(out.verts.size()));
    out.paired += 2;
  }
  return out;
}

}  // namespace fv::garment
