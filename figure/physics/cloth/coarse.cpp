#include <figure/physics/cloth/coarse.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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

namespace {

// A cloth mostly of quads (as DAZ's are, with a few triangles) is copied as quads along its own threads:
// the quads' diagonals give it its soft bias, and a copy of triangles has every side a thread (a
// triangle of springs can't hold the weft stiff with the bias soft). This share of its faces quads
constexpr double kQuadShare = 0.8;
// A ring of quads is collapsed while the faces either side of it come out no wider than this many times
// the faces asked for
constexpr double kGrow = 1.5;
// Passes, each of which can halve the faces' width
constexpr int kPasses = 6;
// A collapse that would turn a face further than this (the cosine) is left: it would fold the copy
constexpr double kNoFold = 0.3;

uint64_t sideKey(uint32_t a, uint32_t b)
{
  return (uint64_t(std::min(a, b)) << 32) | uint64_t(std::max(a, b));
}

// A face's normal, the fan of its corners, each vertex taken through `at`.
template <class At>
D3 faceNormal(const std::vector<uint32_t>& f, const float* p, At&& at)
{
  D3       out;
  const D3 o = pointOf(p, at(f[0]));
  for(size_t k = 1; k + 1 < f.size(); k++)
    out = out + cross(pointOf(p, at(f[k])) - o, pointOf(p, at(f[k + 1])) - o);
  return out;
}

// The faces as they are: each vertex through `find`, a corner repeated in turn once, and a face left with
// fewer than three corners, or one twice, gone.
template <class Find>
std::vector<std::vector<uint32_t>> resolved(const std::vector<std::vector<uint32_t>>& faces, Find&& find)
{
  std::vector<std::vector<uint32_t>> out;
  out.reserve(faces.size());
  for(const auto& f : faces)
  {
    std::vector<uint32_t> g;
    for(uint32_t v : f)
    {
      v = find(v);
      if(g.empty() || g.back() != v)
        g.push_back(v);
    }
    while(g.size() > 1 && g.front() == g.back())
      g.pop_back();
    if(g.size() < 3)
      continue;
    std::vector<uint32_t> sorted = g;
    std::sort(sorted.begin(), sorted.end());
    if(std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      continue;
    out.push_back(std::move(g));
  }
  return out;
}

// The cloth's polygons with rings of its quads collapsed until its faces are about `edge` wide. A ring
// is a strip of quads joined by their opposite sides; collapsing it merges the vertices along one side
// of the strip into those along the other, so the strip goes and the quads either side grow across it.
// A pass collapses rings both ways at once, but none beside one it collapsed (a quad would grow three
// times), nor one that would fold a face; a hem and where the cloth is held keep their vertices. None
// when the cloth isn't mostly quads.
std::optional<std::vector<std::vector<uint32_t>>> collapseRings(const Surface& cloth, float edge)
{
  const size_t                       n = cloth.vertexCount();
  const float*                       p = cloth.rest.data();
  std::vector<std::vector<uint32_t>> faces;
  size_t                             quads = 0;
  for(size_t f = 0; f + 1 < cloth.polyStart.size(); f++)
  {
    const int s = cloth.polyStart[f], e = cloth.polyStart[f + 1];
    if(e - s < 3 || size_t(e) > cloth.polyVerts.size())
      continue;
    std::vector<uint32_t> face;
    bool                  ok = true;
    for(int k = s; k < e; k++)
    {
      const int32_t v = cloth.polyVerts[size_t(k)];
      ok              = ok && v >= 0 && size_t(v) < n;
      if(ok)
        face.push_back(uint32_t(v));
    }
    if(!ok)
      continue;
    quads += face.size() == 4 ? 1 : 0;
    faces.push_back(std::move(face));
  }
  if(faces.empty() || double(quads) < kQuadShare * double(faces.size()))
    return std::nullopt;

  std::vector<uint32_t> rep(n);
  for(size_t v = 0; v < n; v++)
    rep[v] = uint32_t(v);
  const auto find = [&](uint32_t v) {
    while(rep[v] != v)
    {
      rep[v] = rep[rep[v]];
      v      = rep[v];
    }
    return v;
  };
  const auto held = [&](uint32_t v) { return !(cloth.strength.size() == n && cloth.strength[v] > 1e-4f); };
  constexpr uint32_t kNone = UINT32_MAX;

  for(int pass = 0; pass < kPasses; pass++)
  {
    faces           = resolved(faces, find);
    const size_t nf = faces.size();
    // each side's faces, and the vertices on a hem (a side with one face)
    std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, uint8_t>>> sides;
    sides.reserve(nf * 4);
    for(uint32_t f = 0; f < nf; f++)
      for(size_t k = 0; k < faces[f].size(); k++)
        sides[sideKey(faces[f][k], faces[f][(k + 1) % faces[f].size()])].emplace_back(f, uint8_t(k));
    std::vector<uint8_t> hem(n, 0);
    for(const auto& [key, on] : sides)
      if(on.size() == 1)
        hem[size_t(key >> 32)] = hem[size_t(key & 0xffffffffu)] = 1;
    // a quad's two pairs of opposite sides are nodes (face * 2 + side % 2), joined into rings where two
    // quads share a side
    std::vector<uint32_t> parent(nf * 2);
    for(size_t x = 0; x < parent.size(); x++)
      parent[x] = uint32_t(x);
    const auto root = [&](uint32_t x) {
      while(parent[x] != x)
      {
        parent[x] = parent[parent[x]];
        x         = parent[x];
      }
      return x;
    };
    for(const auto& [key, on] : sides)
      if(on.size() == 2 && faces[on[0].first].size() == 4 && faces[on[1].first].size() == 4)
      {
        const uint32_t a = root(on[0].first * 2 + on[0].second % 2), b = root(on[1].first * 2 + on[1].second % 2);
        if(a != b)
          parent[std::max(a, b)] = std::min(a, b);
      }
    struct Ring
    {
      std::vector<uint32_t>                       nodes;
      std::vector<std::pair<uint32_t, uint32_t>> merge;  // (from, to)
      double                                      mean = 0;
      bool                                        ok   = true;
    };
    std::vector<Ring>     rings;
    std::vector<uint32_t> ringOf(nf * 2, kNone);
    {
      std::unordered_map<uint32_t, uint32_t> idOf;
      for(uint32_t x = 0; x < nf * 2; x++)
      {
        if(faces[x / 2].size() != 4)
          continue;
        const uint32_t r  = root(x);
        auto [it, fresh]  = idOf.emplace(r, uint32_t(rings.size()));
        if(fresh)
          rings.emplace_back();
        ringOf[x] = it->second;
        rings[it->second].nodes.push_back(x);
      }
    }
    // a node's two sides, each from one side of its strip to the other: (v0 v1, v3 v2) or (v1 v2, v0 v3)
    const auto across = [&](uint32_t node, int which) {
      const auto& q = faces[node / 2];
      if(node % 2 == 0)
        return which == 0 ? std::pair{q[0], q[1]} : std::pair{q[3], q[2]};
      return which == 0 ? std::pair{q[1], q[2]} : std::pair{q[0], q[3]};
    };
    std::vector<int8_t> flip(nf * 2, -1);
    for(uint32_t id = 0; id < rings.size(); id++)
    {
      Ring& R = rings[id];
      // a ring through a quad both ways (a spiral) can't be collapsed
      for(uint32_t x : R.nodes)
        R.ok = R.ok && ringOf[x ^ 1u] != id;
      if(!R.ok)
        continue;
      // its sides oriented alike, from the start: a ring that comes back reversed (a Moebius strip) can't
      std::vector<uint32_t> todo{R.nodes.front()};
      flip[R.nodes.front()] = 0;
      while(!todo.empty() && R.ok)
      {
        const uint32_t x = todo.back();
        todo.pop_back();
        for(int which = 0; which < 2; which++)
        {
          auto [a, b] = across(x, which);
          if(flip[x])
            std::swap(a, b);
          for(const auto& [g, j] : sides[sideKey(a, b)])
          {
            if(faces[g].size() != 4 || g * 2 + j % 2 == x)
              continue;
            const uint32_t y      = g * 2 + j % 2;
            const auto [ya, yb]   = across(y, j < 2 ? 0 : 1);
            const int8_t   want   = ya == a ? 0 : 1;
            if(flip[y] < 0)
            {
              flip[y] = want;
              todo.push_back(y);
            }
            else if(flip[y] != want)
              R.ok = false;
            (void)yb;
          }
        }
      }
      if(!R.ok)
        continue;
      // its sides, each once; a vertex on two of them (a ring round a pole) can't be collapsed
      std::vector<std::pair<uint32_t, uint32_t>> sidesOf;
      for(uint32_t x : R.nodes)
        for(int which = 0; which < 2; which++)
        {
          auto [a, b] = across(x, which);
          if(flip[x])
            std::swap(a, b);
          sidesOf.emplace_back(a, b);
        }
      std::sort(sidesOf.begin(), sidesOf.end());
      sidesOf.erase(std::unique(sidesOf.begin(), sidesOf.end()), sidesOf.end());
      std::vector<uint32_t> verts;
      size_t                hemA = 0, hemB = 0, heldA = 0, heldB = 0;
      for(const auto& [a, b] : sidesOf)
      {
        verts.insert(verts.end(), {a, b});
        R.mean += distance(p, a, b);
        hemA += hem[a];
        hemB += hem[b];
        heldA += held(a) ? 1 : 0;
        heldB += held(b) ? 1 : 0;
      }
      std::sort(verts.begin(), verts.end());
      R.ok = std::adjacent_find(verts.begin(), verts.end()) == verts.end() && !sidesOf.empty();
      if(!R.ok)
        continue;
      R.mean /= double(sidesOf.size());
      // the side on a hem, or held, keeps its vertices: the other merges into it
      const bool intoB = hemB > hemA || (hemB == hemA && heldB > heldA);
      for(const auto& [a, b] : sidesOf)
        R.merge.emplace_back(intoB ? a : b, intoB ? b : a);
    }
    // which rings cross (share a quad), and each vertex's faces
    std::unordered_set<uint64_t> crossing;
    for(uint32_t f = 0; f < nf; f++)
      if(faces[f].size() == 4)
        crossing.insert(sideKey(ringOf[f * 2], ringOf[f * 2 + 1]));
    const auto crosses = [&](uint32_t a, uint32_t b) { return crossing.count(sideKey(a, b)) > 0; };
    std::vector<std::vector<uint32_t>> facesAt(n);
    for(uint32_t f = 0; f < nf; f++)
      for(uint32_t v : faces[f])
        facesAt[v].push_back(f);
    // the narrowest rings first
    std::vector<uint32_t> order;
    for(uint32_t id = 0; id < rings.size(); id++)
      if(rings[id].ok && 2 * rings[id].mean <= kGrow * double(edge))
        order.push_back(id);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return rings[a].mean < rings[b].mean; });
    std::vector<std::vector<uint32_t>> grownBy(nf), movedBy(n);
    std::unordered_map<uint32_t, uint32_t> moving;
    size_t                                 collapsed = 0;
    for(uint32_t id : order)
    {
      const Ring& R       = rings[id];
      const auto  inStrip = [&](uint32_t f) { return faces[f].size() == 4 && (ringOf[f * 2] == id || ringOf[f * 2 + 1] == id); };
      // beside a ring collapsed this pass, other than one it crosses
      bool blocked = false;
      for(uint32_t x : R.nodes)
        for(uint32_t by : grownBy[x / 2])
          blocked = blocked || !crosses(by, id);
      for(const auto& [from, to] : R.merge)
        for(uint32_t v : {from, to})
          for(uint32_t by : movedBy[v])
            blocked = blocked || !crosses(by, id);
      if(blocked)
        continue;
      // nor one that would fold a face
      moving.clear();
      for(const auto& [from, to] : R.merge)
        if(find(from) != find(to))
          moving[find(from)] = find(to);
      const auto now   = [&](uint32_t v) { return find(v); };
      const auto after = [&](uint32_t v) {
        v             = find(v);
        const auto it = moving.find(v);
        return it == moving.end() ? v : it->second;
      };
      for(const auto& [from, to] : R.merge)
        for(uint32_t f : facesAt[from])
        {
          if(inStrip(f))
            continue;
          const D3     was = faceNormal(faces[f], p, now), will = faceNormal(faces[f], p, after);
          const double lw = std::sqrt(dot(was, was)), ll = std::sqrt(dot(will, will));
          if(lw > 1e-12 && ll > 1e-12 && dot(was, will) < kNoFold * lw * ll)
            blocked = true;
        }
      if(blocked)
        continue;
      for(const auto& [v, to] : moving)
        rep[v] = to;
      for(const auto& [from, to] : R.merge)
        for(uint32_t v : {from, to})
        {
          movedBy[v].push_back(id);
          for(uint32_t f : facesAt[v])
            if(!inStrip(f))
              grownBy[f].push_back(id);
        }
      collapsed++;
    }
    if(collapsed == 0)
      break;
  }
  return resolved(faces, find);
}

}  // namespace

std::optional<CoarseCloth> coarseCloth(const Surface& cloth, float edge)
{
  if(cloth.kind == Surface::Kind::eCollider || !(edge > 0))
    return std::nullopt;
  const size_t n = cloth.vertexCount();
  const float* p = cloth.rest.data();
  if(cloth.strength.size() != n)
    return std::nullopt;
  // a cloth of quads: a copy of quads along its threads, its faces its polygons
  if(auto quads = collapseRings(cloth, edge))
  {
    std::vector<uint32_t> tris;
    for(const auto& f : *quads)
      for(size_t k = 1; k + 1 < f.size(); k++)
        tris.insert(tris.end(), {f[0], f[uint32_t(k)], f[uint32_t(k + 1)]});
    Coarse copy = copyOf(cloth, tris);
    if(copy.keep.empty() || double(copy.keep.size()) > kMostKept * double(n))
      return std::nullopt;
    CoarseCloth c;
    c.surface = std::move(copy.surface);
    c.keep    = std::move(copy.keep);
    std::vector<uint32_t> index(n, 0);
    for(size_t i = 0; i < c.keep.size(); i++)
      index[c.keep[i]] = uint32_t(i);
    c.surface.polyStart.assign(1, 0);
    c.surface.polyVerts.clear();
    for(const auto& f : *quads)
    {
      for(uint32_t v : f)
        c.surface.polyVerts.push_back(int32_t(index[v]));
      c.surface.polyStart.push_back(int32_t(c.surface.polyVerts.size()));
    }
    c.riding = rideOn(cloth, c.surface, c.keep);
    return c;
  }
  // otherwise of triangles, which hang stiffer
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
  if(double(had) < kFiner * double(want))
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
  // its polygons as the copy has them: quads, or its faces
  c.surface.polyStart = like.surface.polyStart;
  c.surface.polyVerts = like.surface.polyVerts;
  c.riding            = like.riding;
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

size_t keepOut(std::vector<float>& at, const std::vector<uint8_t>& held, const Surface& body, const std::vector<float>& bodyAt, float reach)
{
  const size_t nb = body.vertexCount(), n = at.size() / 3;
  if(bodyAt.size() != nb * 3 || body.tris.empty() || n == 0)
    return 0;
  // the body's faces that collide, as the model keeps them, and the most any keeps the cloth off
  const bool            perVert = body.collide.size() == nb;
  std::vector<uint32_t> tris;
  tris.reserve(body.tris.size());
  float most = 0;
  for(size_t t = 0; t + 2 < body.tris.size(); t += 3)
  {
    const uint32_t a = body.tris[t], b = body.tris[t + 1], c = body.tris[t + 2];
    if(a >= nb || b >= nb || c >= nb || (perVert && !(body.collide[a] && body.collide[b] && body.collide[c])))
      continue;
    tris.insert(tris.end(), {a, b, c});
  }
  const auto offsetOf = [&](uint32_t v) { return v < body.offset.size() && body.offset[v] > 0 ? body.offset[v] : 0.05f; };
  for(uint32_t v : tris)
    most = std::max(most, offsetOf(v));
  if(tris.empty())
    return 0;
  const SurfaceIndex  index(bodyAt, tris);
  const float*        q     = bodyAt.data();
  const double        near2 = double(reach + most) * double(reach + most);
  std::atomic<size_t> moved{0};
  jobs::parallelFor(n, 2048, [&](size_t begin, size_t end) {
    size_t count = 0;
    for(size_t v = begin; v < end; v++)
    {
      if(v < held.size() && held[v])
        continue;
      Stamps&       stamps = threadStamps();
      const int32_t id     = stamps.next(index.triangles());
      const D3      p{at[v * 3], at[v * 3 + 1], at[v * 3 + 2]};
      const auto    hit = index.search(p.x, p.y, p.z, 0, {}, near2, stamps.stamp, id);
      if(hit.tri < 0)
        continue;
      const uint32_t a = tris[size_t(hit.tri) * 3], b = tris[size_t(hit.tri) * 3 + 1], c = tris[size_t(hit.tri) * 3 + 2];
      const D3       pa = pointOf(q, a), pb = pointOf(q, b), pc = pointOf(q, c);
      D3             nrm = cross(pb - pa, pc - pa);
      if(!normalize(nrm))
        continue;
      const D3     on     = pa * hit.u + pb * hit.v + pc * hit.w;
      const double side   = dot(p - on, nrm);
      const double offset = (offsetOf(a) + offsetOf(b) + offsetOf(c)) / 3;
      // closer than the offset, from in front or from up to `reach` behind: out along the face to it
      if(side >= offset || side < -double(reach))
        continue;
      const D3 to    = p + nrm * (offset - side);
      at[v * 3]      = float(to.x);
      at[v * 3 + 1]  = float(to.y);
      at[v * 3 + 2]  = float(to.z);
      count++;
    }
    moved += count;
  });
  return moved;
}

}  // namespace fv::physics::cloth
