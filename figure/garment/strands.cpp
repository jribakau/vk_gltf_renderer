#include <figure/garment/strands.hpp>

#include <algorithm>
#include <cmath>

#include <figure/garment/convert.hpp>
#include <figure/garment/decode.hpp>
#include <figure/wardrobe/bind.hpp>

namespace fv::garment {

namespace {

struct P
{
  double x = 0, y = 0, z = 0;
};
P at(std::span<const float> p, size_t v)
{
  return {p[v * 3], p[v * 3 + 1], p[v * 3 + 2]};
}
P minus(P a, P b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
P cross(P a, P b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double dot(P a, P b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// The point a binding names on `body`, and its triangle's unit normal (its corners' winding).
bool placeOf(const wardrobe::SurfaceBinding& b, size_t v, std::span<const float> body, P& point, P& normal)
{
  if(b.verts.size() < (v + 1) * 3)
    return false;
  double sum = 0;
  point      = {};
  P c[3];
  for(size_t j = 0; j < 3; j++)
  {
    const uint32_t i = b.verts[v * 3 + j];
    if(size_t(i) * 3 + 2 >= body.size())
      return false;
    c[j]           = at(body, i);
    const double w = b.weights[v * 3 + j];
    point          = {point.x + w * c[j].x, point.y + w * c[j].y, point.z + w * c[j].z};
    sum += w;
  }
  if(sum <= 0)
    return false;
  point        = {point.x / sum, point.y / sum, point.z / sum};
  normal       = cross(minus(c[1], c[0]), minus(c[2], c[0]));
  const double n = std::sqrt(dot(normal, normal));
  if(n <= 1e-12)
    return false;
  normal = {normal.x / n, normal.y / n, normal.z / n};
  return true;
}

}  // namespace

CarriedRoots carryRoots(std::span<const float> made, std::span<const float> fitted, std::span<const int32_t> lineStart, std::span<const int32_t> lineVerts,
                        std::span<const float> sourcePositions, std::span<const uint32_t> sourceTris, const bodymap::TypeMap& sourceMap, const Target& onto,
                        std::span<const float> figure, float fade)
{
  CarriedRoots out;
  out.positions.assign(fitted.begin(), fitted.end());
  if(lineStart.size() < 2 || made.size() != fitted.size())
    return out;
  // each strand's root as made
  std::vector<size_t> rootOf;
  std::vector<float>  roots;
  for(size_t l = 0; l + 1 < lineStart.size(); l++)
  {
    const int32_t s = lineStart[l];
    if(s >= lineStart[l + 1] || size_t(s) >= lineVerts.size())
      continue;
    const size_t v = size_t(lineVerts[size_t(s)]);
    if(v * 3 + 2 >= made.size())
      continue;
    rootOf.push_back(l);
    roots.insert(roots.end(), {made[v * 3], made[v * 3 + 1], made[v * 3 + 2]});
  }
  if(roots.empty())
    return out;
  // its height over its own body, and its place on the figure
  const wardrobe::SurfaceBinding own       = wardrobe::bindToSurface(sourcePositions, sourceTris, roots);
  const Addresses                addresses = convert(roots, sourceMap);
  const Decoded                  placed    = decode(addresses, onto);
  for(size_t r = 0; r < rootOf.size(); r++)
  {
    if(r >= placed.hold.size() || placed.hold[r] < 1)
      continue;
    P f, n, g, m;
    if(!placeOf(own, r, sourcePositions, f, n) || !placeOf(placed.at, r, figure, g, m))
      continue;
    const double height = dot(minus(at(roots, r), f), n);
    const size_t l      = rootOf[r];
    const size_t root   = size_t(lineVerts[size_t(lineStart[l])]);
    // the figure's normal there turned to the side the fitted root lies on: the corners' winding
    // isn't the body's
    const P was = at(fitted, root);
    if(dot(minus(was, g), m) < 0)
      m = {-m.x, -m.y, -m.z};
    const P to{g.x + height * m.x, g.y + height * m.y, g.z + height * m.z};
    const P move = minus(to, was);
    // along the strand, all of the move at the root and none past `fade`
    double along = 0;
    P      prev  = was;
    for(int32_t i = lineStart[l]; i < lineStart[l + 1]; i++)
    {
      const size_t v = size_t(lineVerts[size_t(i)]);
      const P      p = at(fitted, v);
      along += std::sqrt(dot(minus(p, prev), minus(p, prev)));
      prev = p;
      if(along >= fade)
        break;
      const double t = along / fade, w = 1 - t * t * (3 - 2 * t);
      out.positions[v * 3] += float(w * move.x);
      out.positions[v * 3 + 1] += float(w * move.y);
      out.positions[v * 3 + 2] += float(w * move.z);
    }
    out.moved++;
  }
  return out;
}

}  // namespace fv::garment
