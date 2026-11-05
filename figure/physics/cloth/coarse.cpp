#include <figure/physics/cloth/coarse.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <type_traits>

#include <meshoptimizer.h>

#include <figure/jobs/pool.hpp>
#include <figure/physics/collide/surface_index.hpp>

namespace fv::physics::cloth {

namespace {

// A collider is coarsened only when it has this many times the faces its copy would: one about as
// fine as the cloth already costs what the body does
constexpr double kFiner = 2.0;
// and only when the copy keeps at most this share of them
constexpr double kMostKept = 0.5;
// nor when a vertex of the original lies further from it than this many times the error asked for
constexpr double kTooFar = 2.0;
// How many times a copy that lies too far off is made again, within half the error each time
constexpr int kAttempts = 4;

double distance(const float* p, uint32_t a, uint32_t b)
{
  const double x = p[a * 3] - p[b * 3], y = p[a * 3 + 1] - p[b * 3 + 1], z = p[a * 3 + 2] - p[b * 3 + 2];
  return std::sqrt(x * x + y * y + z * z);
}

double area(const float* p, uint32_t a, uint32_t b, uint32_t c)
{
  const double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
  const double vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
  const double x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
  return 0.5 * std::sqrt(x * x + y * y + z * z);
}

// The collider with only the vertices these faces use, in their order, and the faces into them.
Coarse copyOf(const Surface& collider, const std::vector<uint32_t>& faces)
{
  const size_t      n = collider.vertexCount();
  Coarse            c;
  std::vector<char> used(n, 0);
  for(uint32_t v : faces)
    used[v] = 1;
  std::vector<uint32_t> index(n, 0);
  for(size_t v = 0; v < n; v++)
    if(used[v])
    {
      index[v] = uint32_t(c.keep.size());
      c.keep.push_back(uint32_t(v));
    }
  Surface& s = c.surface;
  // what has a value a vertex keeps those of the kept ones; what doesn't (no skin) stays empty
  const auto take = [&](const auto& values, size_t per) {
    using Values = std::decay_t<decltype(values)>;
    return values.size() == n * per ? keepOf(values, c.keep, per) : Values{};
  };
  s.kind          = collider.kind;
  s.rest          = take(collider.rest, 3);
  s.posed         = take(collider.posed, 3);
  s.strength      = take(collider.strength, 1);
  s.stretch       = take(collider.stretch, 1);
  s.shear         = take(collider.shear, 1);
  s.bend          = take(collider.bend, 1);
  s.buckling      = take(collider.buckling, 1);
  s.bucklingRatio = take(collider.bucklingRatio, 1);
  s.density       = take(collider.density, 1);
  s.contraction   = take(collider.contraction, 1);
  s.damping       = take(collider.damping, 1);
  s.friction      = take(collider.friction, 1);
  s.offset        = take(collider.offset, 1);
  s.layer         = take(collider.layer, 1);
  s.selfCollide   = take(collider.selfCollide, 1);
  s.collide       = take(collider.collide, 1);
  s.visible       = take(collider.visible, 1);
  s.gravity       = collider.gravity;
  s.skinIndex     = take(collider.skinIndex, 4);
  s.skinWeight    = take(collider.skinWeight, 4);
  s.bones         = collider.bones;
  s.restLocal     = take(collider.restLocal, 3);
  s.toSpace       = collider.toSpace;
  s.tris.reserve(faces.size());
  for(uint32_t v : faces)
    s.tris.push_back(index[v]);
  return c;
}

// The farthest of these vertices (into `p`, xyz each) from the copy's faces; none when one lies
// further than sqrt(limit2).
std::optional<double> farthest(const Surface& copy, const float* p, const std::vector<uint32_t>& verts, double limit2)
{
  const SurfaceIndex index(copy.rest, copy.tris);
  double             far2   = 0;
  bool               tooFar = false;
  std::mutex         mutex;
  jobs::parallelFor(verts.size(), 2048, [&](size_t begin, size_t end) {
    double local = 0;
    bool   out   = false;
    for(size_t i = begin; i < end && !out; i++)
    {
      const size_t  v      = verts[i];
      Stamps&       stamps = threadStamps();
      const int32_t id     = stamps.next(index.triangles());
      const auto    hit    = index.search(p[v * 3], p[v * 3 + 1], p[v * 3 + 2], 0, {}, limit2, stamps.stamp, id);
      if(hit.tri < 0)
        out = true;
      else
        local = std::max(local, hit.d2);
    }
    std::lock_guard lock(mutex);
    far2   = std::max(far2, local);
    tooFar = tooFar || out;
  });
  if(tooFar)
    return std::nullopt;
  return std::sqrt(far2);
}

}  // namespace

float meanEdge(const Surface& s)
{
  const size_t n   = s.vertexCount();
  double       sum = 0;
  size_t       count = 0;
  for(size_t t = 0; t + 2 < s.tris.size(); t += 3)
  {
    const uint32_t c[3] = {s.tris[t], s.tris[t + 1], s.tris[t + 2]};
    if(c[0] >= n || c[1] >= n || c[2] >= n)
      continue;
    for(int k = 0; k < 3; k++)
      sum += distance(s.rest.data(), c[k], c[(k + 1) % 3]);
    count += 3;
  }
  return count ? float(sum / double(count)) : 0.f;
}

std::optional<Coarse> coarsen(const Surface& collider, float edge, float maxError)
{
  if(collider.kind != Surface::Kind::eCollider || !(edge > 0) || !(maxError > 0))
    return std::nullopt;
  const size_t n       = collider.vertexCount();
  const bool   perVert = collider.collide.size() == n;
  const float* p       = collider.rest.data();
  // the faces that collide (the model keeps only those), and the area they cover
  std::vector<uint32_t> faces;
  faces.reserve(collider.tris.size());
  double covered = 0;
  for(size_t t = 0; t + 2 < collider.tris.size(); t += 3)
  {
    const uint32_t a = collider.tris[t], b = collider.tris[t + 1], c = collider.tris[t + 2];
    if(a >= n || b >= n || c >= n || (perVert && !(collider.collide[a] && collider.collide[b] && collider.collide[c])))
      continue;
    faces.insert(faces.end(), {a, b, c});
    covered += area(p, a, b, c);
  }
  const size_t had = faces.size() / 3;
  // a face whose sides are `edge` covers sqrt(3)/4 edge^2
  const double each = std::sqrt(3.0) / 4.0 * double(edge) * double(edge);
  const size_t want = std::max<size_t>(1, size_t(std::ceil(covered / each)));
  if(double(had) < kFiner * double(want))
    return std::nullopt;
  // the vertices the copy is measured by: every one of a face that collides
  std::vector<uint32_t> verts;
  {
    std::vector<char> seen(n, 0);
    for(uint32_t v : faces)
      if(!seen[v])
      {
        seen[v] = 1;
        verts.push_back(v);
      }
  }
  const double limit2 = kTooFar * maxError * kTooFar * maxError;

  // Simplified to lie within the error (in the cage's units), then measured: the simplifier's own
  // estimate runs low (a sphere's copy sank 1.4 mm under it where it said 1). It also drops whole any
  // piece smaller than the error (a bead, a stitch), and when one lies far from the rest (a chain's
  // links, 1.4 mm across, under an error of 2.3 mm) the copy is too far off there: made again within
  // half the error, which keeps them, each in a few faces
  std::vector<uint32_t> out(faces.size());
  float                 within = maxError;
  for(int attempt = 0; attempt < kAttempts; attempt++, within *= 0.5f)
  {
    float        estimate = 0;
    const size_t count    = meshopt_simplify(out.data(), faces.data(), faces.size(), p, n, sizeof(float) * 3, want * 3, within,
                                             meshopt_SimplifyErrorAbsolute | meshopt_SimplifyPrune, &estimate);
    // within less it keeps more
    if(count == 0 || double(count / 3) > kMostKept * double(had))
      return std::nullopt;
    Coarse     c   = copyOf(collider, std::vector<uint32_t>(out.begin(), out.begin() + std::ptrdiff_t(count)));
    const auto far = farthest(c.surface, p, verts, limit2);
    if(!far)
      continue;
    c.error = float(*far);
    // the cloth keeps its offset from the original's surface wherever the copy lies under it
    for(float& o : c.surface.offset)
      o += c.error;
    return c;
  }
  return std::nullopt;
}

std::optional<Coarse> sameCopy(const Surface& collider, const Coarse& like)
{
  if(collider.kind != Surface::Kind::eCollider || like.keep.empty() || like.keep.back() >= collider.vertexCount())
    return std::nullopt;
  std::vector<uint32_t> faces;
  faces.reserve(like.surface.tris.size());
  for(uint32_t t : like.surface.tris)
    faces.push_back(like.keep[t]);
  Coarse c = copyOf(collider, faces);
  c.error  = like.error;
  for(float& o : c.surface.offset)
    o += c.error;
  return c;
}

}  // namespace fv::physics::cloth
