#include <figure/physics/cloth/coarse.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <type_traits>
#include <utility>

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

namespace {

// The light copy of a cloth is simplified to lie within this share of its faces' width of the cloth,
// so a ruffle or a pleat keeps more of its vertices than a flat panel
constexpr float kClothError = 0.5f;

struct D3
{
  double x = 0, y = 0, z = 0;
};
D3     operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
D3     operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
D3     operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
D3     cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
D3     pointOf(const float* p, uint32_t v) { return {p[v * 3], p[v * 3 + 1], p[v * 3 + 2]}; }
bool   normalize(D3& a)
{
  const double l = std::sqrt(dot(a, a));
  if(!(l > 1e-12))
    return false;
  a = a * (1 / l);
  return true;
}

// The closest point of q on the triangle abc, as its corners' weights (Ericson, Real-Time Collision
// Detection 5.1.5).
std::array<double, 3> closestWeights(D3 q, D3 a, D3 b, D3 c)
{
  const D3     ab = b - a, ac = c - a, ap = q - a;
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if(d1 <= 0 && d2 <= 0)
    return {1, 0, 0};
  const D3     bp = q - b;
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if(d3 >= 0 && d4 <= d3)
    return {0, 1, 0};
  const double vc = d1 * d4 - d3 * d2;
  if(vc <= 0 && d1 >= 0 && d3 <= 0)
  {
    const double v = d1 / (d1 - d3);
    return {1 - v, v, 0};
  }
  const D3     cp = q - c;
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if(d6 >= 0 && d5 <= d6)
    return {0, 0, 1};
  const double vb = d5 * d2 - d1 * d6;
  if(vb <= 0 && d2 >= 0 && d6 <= 0)
  {
    const double w = d2 / (d2 - d6);
    return {1 - w, 0, w};
  }
  const double va = d3 * d6 - d5 * d4;
  if(va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
  {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return {0, 1 - w, w};
  }
  const double denom = 1 / (va + vb + vc);
  const double v = vb * denom, w = vc * denom;
  return {1 - v - w, v, w};
}

// Each vertex's normal: its faces' normals weighted by their areas.
std::vector<D3> vertexNormals(const std::vector<uint32_t>& tris, const float* p, size_t n)
{
  std::vector<D3> out(n);
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
  {
    const D3 a = pointOf(p, tris[t]), b = pointOf(p, tris[t + 1]), c = pointOf(p, tris[t + 2]);
    const D3 face = cross(b - a, c - a);
    for(size_t k = 0; k < 3; k++)
      out[tris[t + k]] = out[tris[t + k]] + face;
  }
  for(D3& v : out)
    normalize(v);
  return out;
}

// The frame a vertex rides in at weights `w` on face `f` of a mesh at `p` with normals `nrm`: the
// corners' normals blended, the face's first side made square to it, and the third across both.
struct Frame
{
  D3 point, along, across, normal;
};
Frame frameAt(const std::vector<uint32_t>& tris, size_t f, const float* w, const float* p, const std::vector<D3>& nrm)
{
  const uint32_t a = tris[f * 3], b = tris[f * 3 + 1], c = tris[f * 3 + 2];
  const D3       pa = pointOf(p, a), pb = pointOf(p, b), pc = pointOf(p, c);
  Frame          fr;
  fr.point  = pa * w[0] + pb * w[1] + pc * w[2];
  fr.normal = nrm[a] * w[0] + nrm[b] * w[1] + nrm[c] * w[2];
  if(!normalize(fr.normal))
  {
    fr.normal = cross(pb - pa, pc - pa);
    if(!normalize(fr.normal))
      fr.normal = {0, 1, 0};
  }
  for(const D3 side : {pb - pa, pc - pa, pc - pb})
  {
    fr.along = side - fr.normal * dot(side, fr.normal);
    if(normalize(fr.along))
      break;
  }
  fr.across = cross(fr.normal, fr.along);
  return fr;
}

// The copy's polygons: its faces.
void facesArePolygons(Surface& s)
{
  s.polyStart.assign(1, 0);
  s.polyVerts.clear();
  for(size_t t = 0; t + 2 < s.tris.size(); t += 3)
  {
    for(size_t k = 0; k < 3; k++)
      s.polyVerts.push_back(int32_t(s.tris[t + k]));
    s.polyStart.push_back(int32_t(s.polyVerts.size()));
  }
}

// How each vertex of the cloth rides on its copy, made of `keep`: on the closest face around the copy's
// vertex nearest it along the cloth (so a pleat's vertices ride on their own fold, not the next one), at
// that point's weights, and the rest of the way in the frame there.
Riding rideOn(const Surface& cloth, const Surface& copy, const std::vector<uint32_t>& keep)
{
  const size_t n = cloth.vertexCount(), m = copy.vertexCount();
  const float* p = cloth.rest.data();
  // the cloth's edges, as lists of neighbours
  std::vector<uint32_t> start(n + 1, 0), next;
  {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for(size_t t = 0; t + 2 < cloth.tris.size(); t += 3)
      for(size_t k = 0; k < 3; k++)
      {
        const uint32_t a = cloth.tris[t + k], b = cloth.tris[t + (k + 1) % 3];
        if(a < n && b < n && a != b)
        {
          edges.emplace_back(a, b);
          edges.emplace_back(b, a);
        }
      }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    for(const auto& [a, b] : edges)
      start[a + 1]++;
    for(size_t v = 0; v < n; v++)
      start[v + 1] += start[v];
    next.reserve(edges.size());
    for(const auto& e : edges)
      next.push_back(e.second);
  }
  // the copy's vertex nearest each along the cloth's edges (Dijkstra from all of them at once)
  std::vector<double>   along(n, INFINITY);
  std::vector<uint32_t> nearest(n, UINT32_MAX);
  {
    using Item = std::pair<double, uint32_t>;
    std::vector<Item> heap;
    for(size_t i = 0; i < keep.size(); i++)
    {
      along[keep[i]]   = 0;
      nearest[keep[i]] = uint32_t(i);
      heap.emplace_back(0.0, keep[i]);
    }
    const auto later = [](const Item& a, const Item& b) { return a.first > b.first || (a.first == b.first && a.second > b.second); };
    std::make_heap(heap.begin(), heap.end(), later);
    while(!heap.empty())
    {
      std::pop_heap(heap.begin(), heap.end(), later);
      const auto [d, v] = heap.back();
      heap.pop_back();
      if(d > along[v])
        continue;
      for(uint32_t k = start[v]; k < start[v + 1]; k++)
      {
        const uint32_t u  = next[k];
        const double   du = d + distance(p, v, u);
        if(du < along[u] || (du == along[u] && nearest[v] < nearest[u]))
        {
          along[u]   = du;
          nearest[u] = nearest[v];
          heap.emplace_back(du, u);
          std::push_heap(heap.begin(), heap.end(), later);
        }
      }
    }
  }
  // the copy's faces around each of its vertices
  std::vector<uint32_t> faceStart(m + 1, 0), facesOf;
  {
    for(uint32_t v : copy.tris)
      faceStart[v + 1]++;
    for(size_t v = 0; v < m; v++)
      faceStart[v + 1] += faceStart[v];
    facesOf.resize(copy.tris.size());
    std::vector<uint32_t> fill(faceStart.begin(), faceStart.end() - 1);
    for(size_t t = 0; t < copy.tris.size(); t++)
      facesOf[fill[copy.tris[t]]++] = uint32_t(t / 3);
  }
  const float*          q   = copy.rest.data();
  const std::vector<D3> nrm = vertexNormals(copy.tris, q, m);
  // a piece of the cloth with none of the copy's vertices (none should be): the closest face of all
  const SurfaceIndex anywhere(copy.rest, copy.tris);

  Riding r;
  r.face.assign(n, 0);
  r.weights.assign(n * 3, 0.f);
  r.offset.assign(n * 3, 0.f);
  r.held.assign(n, 0);
  jobs::parallelFor(n, 1024, [&](size_t begin, size_t end) {
    std::vector<uint32_t> near;
    for(size_t v = begin; v < end; v++)
    {
      const D3 at = pointOf(p, uint32_t(v));
      // the faces around the nearest vertex of the copy and around its neighbours
      near.clear();
      if(nearest[v] != UINT32_MAX)
        for(uint32_t k = faceStart[nearest[v]]; k < faceStart[nearest[v] + 1]; k++)
          for(size_t c = 0; c < 3; c++)
          {
            const uint32_t u = copy.tris[size_t(facesOf[k]) * 3 + c];
            for(uint32_t j = faceStart[u]; j < faceStart[u + 1]; j++)
              near.push_back(facesOf[j]);
          }
      else
      {
        Stamps&       stamps = threadStamps();
        const int32_t id     = stamps.next(anywhere.triangles());
        const auto    hit    = anywhere.search(at.x, at.y, at.z, 0, {}, INFINITY, stamps.stamp, id);
        if(hit.tri >= 0)
          near.push_back(uint32_t(hit.tri));
      }
      std::sort(near.begin(), near.end());
      near.erase(std::unique(near.begin(), near.end()), near.end());
      double                best = INFINITY;
      uint32_t              face = 0;
      std::array<double, 3> w{1, 0, 0};
      for(uint32_t f : near)
      {
        const uint32_t a = copy.tris[size_t(f) * 3], b = copy.tris[size_t(f) * 3 + 1], c = copy.tris[size_t(f) * 3 + 2];
        const D3       pa = pointOf(q, a), pb = pointOf(q, b), pc = pointOf(q, c);
        const auto     fw = closestWeights(at, pa, pb, pc);
        const D3       on = pa * fw[0] + pb * fw[1] + pc * fw[2];
        const double   d2 = dot(at - on, at - on);
        if(d2 < best)
        {
          best = d2;
          face = f;
          w    = fw;
        }
      }
      r.face[v] = face;
      for(size_t k = 0; k < 3; k++)
        r.weights[v * 3 + k] = float(w[k]);
      const Frame fr  = frameAt(copy.tris, face, &r.weights[v * 3], q, nrm);
      const D3    off = at - fr.point;
      r.offset[v * 3]     = float(dot(off, fr.along));
      r.offset[v * 3 + 1] = float(dot(off, fr.across));
      r.offset[v * 3 + 2] = float(dot(off, fr.normal));
      // held to the skin as the model holds it: no strength, or not in the simulation
      const bool free = cloth.strength.size() == n && cloth.strength[v] > 1e-4f && (cloth.visible.size() != n || cloth.visible[v]);
      r.held[v]       = free ? 0 : 1;
    }
  });
  return r;
}

}  // namespace

std::optional<CoarseCloth> coarseCloth(const Surface& cloth, float edge)
{
  if(cloth.kind == Surface::Kind::eCollider || !(edge > 0))
    return std::nullopt;
  const size_t          n = cloth.vertexCount();
  const float*          p = cloth.rest.data();
  std::vector<uint32_t> faces;
  faces.reserve(cloth.tris.size());
  double covered = 0;
  for(size_t t = 0; t + 2 < cloth.tris.size(); t += 3)
  {
    const uint32_t a = cloth.tris[t], b = cloth.tris[t + 1], c = cloth.tris[t + 2];
    if(a >= n || b >= n || c >= n)
      continue;
    faces.insert(faces.end(), {a, b, c});
    covered += area(p, a, b, c);
  }
  const size_t had  = faces.size() / 3;
  const double each = std::sqrt(3.0) / 4.0 * double(edge) * double(edge);
  const size_t want = std::max<size_t>(1, size_t(std::ceil(covered / each)));
  if(double(had) < kFiner * double(want) || cloth.strength.size() != n)
    return std::nullopt;
  // simplified to faces of even sizes, with the cloth's strength in the error so where it goes from held
  // to free stays where it is: a step of all of it counts as far as a face is wide
  std::vector<uint32_t> out(faces.size());
  float                 estimate = 0;
  const float           weight   = edge;
  const size_t count = meshopt_simplifyWithAttributes(out.data(), faces.data(), faces.size(), p, n, sizeof(float) * 3, cloth.strength.data(),
                                                      sizeof(float), &weight, 1, nullptr, want * 3, kClothError * edge,
                                                      meshopt_SimplifyErrorAbsolute | meshopt_SimplifyRegularize, &estimate);
  if(count == 0 || double(count / 3) > kMostKept * double(had))
    return std::nullopt;
  out.resize(count);
  Coarse      copy = copyOf(cloth, out);
  CoarseCloth c;
  c.surface = std::move(copy.surface);
  c.keep    = std::move(copy.keep);
  facesArePolygons(c.surface);
  c.riding = rideOn(cloth, c.surface, c.keep);
  return c;
}

std::optional<CoarseCloth> sameCloth(const Surface& cloth, const CoarseCloth& like)
{
  if(cloth.kind == Surface::Kind::eCollider || like.keep.empty() || like.keep.back() >= cloth.vertexCount()
     || like.riding.face.size() != cloth.vertexCount())
    return std::nullopt;
  std::vector<uint32_t> faces;
  faces.reserve(like.surface.tris.size());
  for(uint32_t t : like.surface.tris)
    faces.push_back(like.keep[t]);
  Coarse      copy = copyOf(cloth, faces);
  CoarseCloth c;
  c.surface = std::move(copy.surface);
  c.keep    = std::move(copy.keep);
  facesArePolygons(c.surface);
  c.riding = like.riding;
  return c;
}

std::vector<float> ride(const CoarseCloth& c, const std::vector<float>& at)
{
  const size_t       n = c.riding.face.size();
  std::vector<float> out(n * 3, 0.f);
  if(at.size() != c.surface.vertexCount() * 3)
    return out;
  const std::vector<D3> nrm = vertexNormals(c.surface.tris, at.data(), c.surface.vertexCount());
  jobs::parallelFor(n, 4096, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
    {
      const Frame fr = frameAt(c.surface.tris, c.riding.face[v], &c.riding.weights[v * 3], at.data(), nrm);
      const D3    to = fr.point + fr.along * c.riding.offset[v * 3] + fr.across * c.riding.offset[v * 3 + 1] + fr.normal * c.riding.offset[v * 3 + 2];
      out[v * 3]     = float(to.x);
      out[v * 3 + 1] = float(to.y);
      out[v * 3 + 2] = float(to.z);
    }
  });
  return out;
}

}  // namespace fv::physics::cloth
