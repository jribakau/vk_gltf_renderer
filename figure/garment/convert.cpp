#include <figure/garment/convert.hpp>

#include <algorithm>
#include <cmath>

#include <figure/wardrobe/bind_cache.hpp>

namespace fv::garment {

namespace {

std::vector<uint32_t> outerTris(const bodymap::Body& body)
{
  std::vector<uint32_t> out;
  for(size_t t = 0; t * 3 + 2 < body.tris.size(); t++)
    if(t >= body.outer.size() || body.outer[t])
      out.insert(out.end(), body.tris.begin() + ptrdiff_t(t * 3), body.tris.begin() + ptrdiff_t(t * 3 + 3));
  return out;
}

Vec3 at(const std::vector<float>& p, size_t v)
{
  return {p[v * 3], p[v * 3 + 1], p[v * 3 + 2]};
}

}  // namespace

Addresses convert(std::span<const float> item, const bodymap::TypeMap& type, const wardrobe::BindFilter* filter, const ConvertOptions& options)
{
  const bodymap::Body& source    = type.source;
  const bodymap::Body& reference = type.reference;
  const size_t         n         = item.size() / 3;
  Addresses            out;
  out.at.verts.assign(n * 3, 0);
  out.at.weights.assign(n * 3, 0.f);
  out.at.used.assign(reference.positions.size() / 3, 0);
  out.height.assign(n, 0.f);
  out.across.assign(n * 3, 0.f);
  out.hold.assign(n, 0.f);
  const std::vector<uint32_t> sourceTris = outerTris(source), referenceTris = outerTris(reference);
  if(n == 0 || sourceTris.empty() || referenceTris.empty())
    return out;

  const wardrobe::SurfaceBinding onSource = wardrobe::cachedBindToSurface(source.positions, sourceTris, item, {}, filter);
  const std::vector<Vec3>        normals  = bodymap::outerNormals(source);
  const bodymap::Projection&     to       = type.map.toReference;
  const std::vector<Quat>&       turns    = type.map.frameTurn;

  // each held vertex's place on the source, carried onto the reference as it lies on the source
  std::vector<float>  carried;
  std::vector<size_t> held;
  for(size_t v = 0; v < n; v++)
  {
    Vec3   skin, normal, place;
    Quat   turn{0, 0, 0, 0};
    double bound = 0, placed = 0;
    for(size_t j = 0; j < 3; j++)
    {
      const size_t c = onSource.verts[v * 3 + j];
      const double w = onSource.weights[v * 3 + j];
      if(w == 0 || c * 3 + 2 >= source.positions.size())
        continue;
      bound += w;
      skin   = skin + at(source.positions, c) * w;
      if(c < normals.size())
        normal = normal + normals[c] * w;
      if(c >= to.mapped.size() || !to.mapped[c])
        continue;
      Vec3 corner;
      for(size_t k = 0; k < 3; k++)
        corner = corner + at(reference.positions, to.at.verts[c * 3 + k]) * double(to.at.weights[c * 3 + k]);
      place  = place + corner * w;
      placed += w;
      // the corners' turns blended on one side of the sphere
      const Quat   q    = c < turns.size() ? turns[c] : Quat{};
      const double sign = turn.x * q.x + turn.y * q.y + turn.z * q.z + turn.w * q.w < 0 ? -w : w;
      turn              = {turn.x + q.x * sign, turn.y + q.y * sign, turn.z + q.z * sign, turn.w + q.w * sign};
    }
    if(bound <= 0 || placed <= 0)
      continue;
    skin              = skin * (1 / bound);
    const Vec3   off  = Vec3{item[v * 3], item[v * 3 + 1], item[v * 3 + 2]} - skin;
    const double dist = length(off);
    double       hold = 1;
    if(dist >= options.outer)
      hold = 0;
    else if(dist > options.inner)
    {
      const double t = (dist - options.inner) / (options.outer - options.inner);
      hold           = 1 - t * t * (3 - 2 * t);
    }
    if(hold <= 0)
      continue;
    normal               = normalize(normal);
    const double height  = dot(off, normal);
    const Vec3   across  = rotate(off - normal * height, turn.normalized());
    out.height[v]        = float(height);
    out.across[v * 3]    = float(across.x);
    out.across[v * 3 + 1] = float(across.y);
    out.across[v * 3 + 2] = float(across.z);
    out.hold[v]           = float(hold);
    place                 = place * (1 / placed);
    carried.insert(carried.end(), {float(place.x), float(place.y), float(place.z)});
    held.push_back(v);
  }
  if(held.empty())
    return out;
  // the carried place lies on the reference or a hair off it (a triangle's corners land on a curved
  // skin): its closest point there is the address
  const wardrobe::SurfaceBinding onReference = wardrobe::bindToSurface(reference.positions, referenceTris, carried);
  for(size_t i = 0; i < held.size(); i++)
  {
    const size_t v   = held[i];
    double       sum = 0;
    for(size_t j = 0; j < 3; j++)
      sum += onReference.weights[i * 3 + j];
    if(sum <= 0)
    {
      out.hold[v] = 0;
      continue;
    }
    for(size_t j = 0; j < 3; j++)
    {
      const uint32_t c         = onReference.verts[i * 3 + j];
      out.at.verts[v * 3 + j]   = c;
      out.at.weights[v * 3 + j] = onReference.weights[i * 3 + j];
      if(c < out.at.used.size() && onReference.weights[i * 3 + j] != 0)
        out.at.used[c] = 1;
    }
  }
  return out;
}

}  // namespace fv::garment
