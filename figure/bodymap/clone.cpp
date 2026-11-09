#include <figure/bodymap/clone.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>

#include <figure/bodymap/cache.hpp>
#include <figure/bodymap/frames.hpp>
#include <figure/dson/store.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/math/xform.hpp>
#include <figure/physics/collide/surface_index.hpp>
#include <figure/wardrobe/fit/bases.hpp>

namespace fv::bodymap {

namespace {

using V = std::array<double, 3>;

V    vertexAt(const std::vector<float>& p, size_t v) { return {p[v * 3], p[v * 3 + 1], p[v * 3 + 2]}; }
V    minus(const V& a, const V& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V    cross(const V& a, const V& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
double dot(const V& a, const V& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double length(const V& a) { return std::sqrt(dot(a, a)); }
V    unit(const V& a)
{
  const double l = length(a);
  return l > 0 ? V{a[0] / l, a[1] / l, a[2] / l} : V{0, 0, 0};
}

std::vector<V> faceNormals(const Body& b)
{
  std::vector<V> n(b.tris.size() / 3);
  for(size_t t = 0; t < n.size(); t++)
  {
    const V a = vertexAt(b.positions, b.tris[t * 3]);
    n[t]      = unit(cross(minus(vertexAt(b.positions, b.tris[t * 3 + 1]), a), minus(vertexAt(b.positions, b.tris[t * 3 + 2]), a)));
  }
  return n;
}

// where a projection put `from`'s vertex v, on `onto`
V placeOf(const Projection& p, size_t v, const std::vector<float>& onto)
{
  V out{0, 0, 0};
  for(size_t j = 0; j < 3; j++)
  {
    const V      c = vertexAt(onto, p.at.verts[v * 3 + j]);
    const double w = p.at.weights[v * 3 + j];
    for(size_t i = 0; i < 3; i++)
      out[i] += w * c[i];
  }
  return out;
}

MapQuality::Spread spreadOf(std::vector<double> s)
{
  MapQuality::Spread out;
  out.n = s.size();
  if(s.empty())
    return out;
  std::sort(s.begin(), s.end());
  out.median = s[s.size() / 2];
  out.p95    = s[size_t(0.95 * double(s.size() - 1))];
  out.max    = s.back();
  return out;
}

}  // namespace

// Horn's quaternion: the eigenvector of his symmetric 4x4 matrix with the largest eigenvalue (Jacobi
// sweeps).
Quat bestTurn(const std::vector<Vec3>& a, const std::vector<Vec3>& b)
{
  double S[3][3] = {};
  for(size_t i = 0; i < a.size() && i < b.size(); i++)
  {
    const double p[3] = {a[i].x, a[i].y, a[i].z}, q[3] = {b[i].x, b[i].y, b[i].z};
    for(int r = 0; r < 3; r++)
      for(int c = 0; c < 3; c++)
        S[r][c] += p[r] * q[c];
  }
  const double xx = S[0][0], xy = S[0][1], xz = S[0][2], yx = S[1][0], yy = S[1][1], yz = S[1][2], zx = S[2][0], zy = S[2][1], zz = S[2][2];
  double       N[4][4] = {{xx + yy + zz, yz - zy, zx - xz, xy - yx},
                          {yz - zy, xx - yy - zz, xy + yx, zx + xz},
                          {zx - xz, xy + yx, -xx + yy - zz, yz + zy},
                          {xy - yx, zx + xz, yz + zy, -xx - yy + zz}};
  double       V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for(int sweep = 0; sweep < 64; sweep++)
  {
    double off = 0, scale = 0;
    for(int p = 0; p < 4; p++)
      for(int q = 0; q < 4; q++)
        (p == q ? scale : off) += std::abs(N[p][q]);
    if(off <= 1e-14 * std::max(scale, 1e-300))
      break;
    for(int p = 0; p < 3; p++)
      for(int q = p + 1; q < 4; q++)
      {
        if(std::abs(N[p][q]) < 1e-300)
          continue;
        const double theta = (N[q][q] - N[p][p]) / (2 * N[p][q]);
        const double t     = (theta >= 0 ? 1 : -1) / (std::abs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for(int k = 0; k < 4; k++)
        {
          const double kp = N[k][p], kq = N[k][q];
          N[k][p]         = c * kp - s * kq;
          N[k][q]         = s * kp + c * kq;
        }
        for(int k = 0; k < 4; k++)
        {
          const double pk = N[p][k], qk = N[q][k];
          N[p][k]         = c * pk - s * qk;
          N[q][k]         = s * pk + c * qk;
        }
        for(int k = 0; k < 4; k++)
        {
          const double kp = V[k][p], kq = V[k][q];
          V[k][p]         = c * kp - s * kq;
          V[k][q]         = s * kp + c * kq;
        }
      }
  }
  int best = 0;
  for(int k = 1; k < 4; k++)
    if(N[k][k] > N[best][best])
      best = k;
  return Quat{V[1][best], V[2][best], V[3][best], V[0][best]}.normalized();
}

Body bodyOf(const wardrobe::BaseCage& cage)
{
  Body b;
  b.positions = cage.positions;
  b.tris      = cage.tris;
  b.outer.assign(b.tris.size() / 3, 1);
  for(size_t t = 0; t < b.outer.size() && t < cage.triSurface.size(); t++)
    if(cage.triSurface[t] < cage.surfaces.size())
      b.outer[t] = outerSkin(cage.surfaces[cage.triSurface[t]]) ? 1 : 0;
  b.regions = vertexRegions(cage.skin, cage.skeleton, b.positions.size() / 3);
  return b;
}

std::vector<Vec3> outerNormals(const Body& body)
{
  std::vector<Vec3> n(body.positions.size() / 3);
  for(size_t t = 0; t * 3 + 2 < body.tris.size(); t++)
  {
    if(t < body.outer.size() && !body.outer[t])
      continue;
    const uint32_t a = body.tris[t * 3], c1 = body.tris[t * 3 + 1], c2 = body.tris[t * 3 + 2];
    const V f = cross(minus(vertexAt(body.positions, c1), vertexAt(body.positions, a)), minus(vertexAt(body.positions, c2), vertexAt(body.positions, a)));
    for(const uint32_t v : {a, c1, c2})
      n[v] = n[v] + Vec3{f[0], f[1], f[2]};
  }
  for(Vec3& v : n)
    v = normalize(v);
  return n;
}

std::vector<double> vertexAreas(const Body& body)
{
  std::vector<double> area(body.positions.size() / 3, 0.0);
  for(size_t t = 0; t * 3 + 2 < body.tris.size(); t++)
  {
    if(t < body.outer.size() && !body.outer[t])
      continue;
    const uint32_t a = body.tris[t * 3], b = body.tris[t * 3 + 1], c = body.tris[t * 3 + 2];
    const double   third = length(cross(minus(vertexAt(body.positions, b), vertexAt(body.positions, a)), minus(vertexAt(body.positions, c), vertexAt(body.positions, a)))) / 6;
    for(const uint32_t v : {a, b, c})
      area[v] += third;
  }
  return area;
}

CutEdges::CutEdges(const Body& body)
    : m_body(&body)
    , m_corner(body.positions.size() / 3, 0)
{
  std::map<std::pair<uint32_t, uint32_t>, int> count;
  for(size_t t = 0; t * 3 + 2 < body.tris.size(); t++)
    for(size_t j = 0; j < 3; j++)
    {
      const uint32_t a = body.tris[t * 3 + j], b = body.tris[t * 3 + (j + 1) % 3];
      count[{std::min(a, b), std::max(a, b)}]++;
    }
  for(const auto& [e, n] : count)
    if(n == 1)
    {
      m_edges.insert(e);
      m_corner[e.first] = m_corner[e.second] = 1;
    }
}

bool CutEdges::on(size_t tri, const double w[3]) const
{
  constexpr double kOn = 1e-6;
  const uint32_t   c[3] = {m_body->tris[tri * 3], m_body->tris[tri * 3 + 1], m_body->tris[tri * 3 + 2]};
  for(size_t j = 0; j < 3; j++)
  {
    if(w[j] > kOn)
      continue;
    // on the edge across from corner j...
    const uint32_t a = c[(j + 1) % 3], b = c[(j + 2) % 3];
    if(m_edges.count({std::min(a, b), std::max(a, b)}))
      return true;
    // ...or at the corner left when another weight is nought too
    for(size_t k = 0; k < 3; k++)
      if(k != j && w[k] <= kOn && m_corner[c[3 - j - k]])
        return true;
  }
  return false;
}

Projection project(const Body& from, const Body& onto, const ProjectOptions& options)
{
  const size_t n = from.positions.size() / 3;
  Projection   out;
  out.at.verts.assign(n * 3, 0);
  out.at.weights.assign(n * 3, 0.f);
  out.at.used.assign(onto.positions.size() / 3, 0);
  out.mapped.assign(n, 0);
  if(onto.tris.empty())
    return out;
  const physics::SurfaceIndex index(onto.positions, onto.tris);
  const std::vector<Vec3>     normals = outerNormals(from);
  const std::vector<V>        faces   = faceNormals(onto);
  const double                cosMax  = std::cos(options.maxAngle * 3.14159265358979323846 / 180);
  const double                reach2  = options.reach * options.reach;
  const std::optional<CutEdges> cut = options.refuseCut ? std::optional<CutEdges>(onto) : std::nullopt;
  // a vertex's regions as it matches them, each mask widened once
  std::unordered_map<RegionMask, RegionMask> widened;
  if(options.widen)
    for(const RegionMask m : from.regions)
      if(!widened.count(m))
        widened[m] = withNeighbours(m);
  jobs::parallelFor(n, 256, [&](size_t begin, size_t end) {
    physics::Stamps& stamps = physics::threadStamps();
    for(size_t v = begin; v < end; v++)
    {
      const V normal{normals[v].x, normals[v].y, normals[v].z};
      if(normal[0] == 0 && normal[1] == 0 && normal[2] == 0)
        continue;  // on no outer skin
      const RegionMask own    = v < from.regions.size() ? from.regions[v] : 0;
      const RegionMask mask   = options.widen ? widened.at(own) : own;
      const auto       accept = [&](size_t t) {
        if(t < onto.outer.size() && !onto.outer[t])
          return false;
        if(mask)
        {
          const RegionMask there = onto.regions[onto.tris[t * 3]] | onto.regions[onto.tris[t * 3 + 1]] | onto.regions[onto.tris[t * 3 + 2]];
          if(!(there & mask))
            return false;
        }
        return dot(faces[t], normal) >= cosMax;
      };
      const int32_t id  = stamps.next(index.triangles());
      const auto    hit = index.nearest(from.positions[v * 3], from.positions[v * 3 + 1], from.positions[v * 3 + 2], accept, reach2, stamps.stamp, id);
      if(hit.tri < 0)
        continue;
      if(cut)
      {
        const double w[3] = {hit.u, hit.v, hit.w};
        if(cut->on(size_t(hit.tri), w))
          continue;
      }
      for(size_t j = 0; j < 3; j++)
        out.at.verts[v * 3 + j] = onto.tris[size_t(hit.tri) * 3 + j];
      out.at.weights[v * 3]     = float(hit.u);
      out.at.weights[v * 3 + 1] = float(hit.v);
      out.at.weights[v * 3 + 2] = float(hit.w);
      out.mapped[v]             = 1;
    }
  });
  for(size_t v = 0; v < n; v++)
    if(out.mapped[v])
      for(size_t j = 0; j < 3; j++)
        out.at.used[out.at.verts[v * 3 + j]] = 1;
  return out;
}

BodyMap mapBodies(const Body& source, const Body& reference, const ProjectOptions& options)
{
  return {project(source, reference, options), project(reference, source, options)};
}

MapQuality mapQuality(const Body& source, const Body& reference, const BodyMap& map)
{
  MapQuality   q;
  const size_t ns = source.positions.size() / 3, nr = reference.positions.size() / 3;
  // the classes the gate reports by: borders where nearest points aren't one to one
  const auto has = [](RegionMask m, rig::Region r) { return (m & (regionBit(r, 'l') | regionBit(r, 'r') | regionBit(r))) != 0; };
  double     breastY = 0;
  size_t     breasts = 0;
  for(size_t v = 0; v < ns; v++)
    if(source.regions[v] && (source.regions[v] & ~(regionBit(rig::Region::eBreast, 'l') | regionBit(rig::Region::eBreast, 'r'))) == 0)
      breastY += source.positions[v * 3 + 1], breasts++;
  breastY = breasts ? breastY / double(breasts) : 0;
  const auto classOf = [&](size_t v) -> std::string {
    const RegionMask m = source.regions[v];
    for(const rig::Region r : {rig::Region::eThumb, rig::Region::eIndex, rig::Region::eMid, rig::Region::eRing, rig::Region::ePinky})
      if(has(m, r))
        return "fingers";
    if(has(m, rig::Region::eUpperArm) && (has(m, rig::Region::eChest) || has(m, rig::Region::eCollar) || has(m, rig::Region::eBreast)))
      return "armpit";
    if(has(m, rig::Region::eThigh) && has(m, rig::Region::ePelvis))
      return source.positions[v * 3 + 2] >= 0 ? "crotch" : "gluteal fold";
    if(has(m, rig::Region::eBreast) && (has(m, rig::Region::eAbdomen) || (has(m, rig::Region::eChest) && source.positions[v * 3 + 1] < breastY)))
      return "under-bust";
    if(has(m, rig::Region::eBreast))
      return "breast";
    return "rest";
  };
  std::map<std::string, std::vector<double>> trip, far;
  size_t                                     outer = 0, mapped = 0;
  std::vector<uint8_t>                       onSkin(ns, 0), refSkin(nr, 0);
  for(size_t t = 0; t * 3 + 2 < source.tris.size(); t++)
    if(t >= source.outer.size() || source.outer[t])
      for(size_t j = 0; j < 3; j++)
        onSkin[source.tris[t * 3 + j]] = 1;
  for(size_t t = 0; t * 3 + 2 < reference.tris.size(); t++)
    if(t >= reference.outer.size() || reference.outer[t])
      for(size_t j = 0; j < 3; j++)
        refSkin[reference.tris[t * 3 + j]] = 1;
  for(size_t v = 0; v < ns; v++)
  {
    if(!onSkin[v])
      continue;
    outer++;
    if(!map.toReference.mapped[v])
    {
      q.unmappedBy[regionNames(source.regions[v])]++;
      continue;
    }
    mapped++;
    const V           s     = vertexAt(source.positions, v);
    const std::string cls   = classOf(v);
    const V           there = placeOf(map.toReference, v, reference.positions);
    far[cls].push_back(length(minus(there, s)));
    far["all"].push_back(far[cls].back());
    if(far[cls].back() > 1)
      q.farBy[regionNames(source.regions[v])]++;
    // back: the reference corners' places on the source, blended as the vertex's place was
    V    back{0, 0, 0};
    bool whole = true;
    for(size_t j = 0; j < 3 && whole; j++)
    {
      const uint32_t r = map.toReference.at.verts[v * 3 + j];
      const double   w = map.toReference.at.weights[v * 3 + j];
      if(!map.fromReference.mapped[r])
      {
        whole = w <= 0;
        continue;
      }
      const V c = placeOf(map.fromReference, r, source.positions);
      for(size_t i = 0; i < 3; i++)
        back[i] += w * c[i];
    }
    if(!whole)
      continue;
    trip[cls].push_back(length(minus(back, s)));
    trip["all"].push_back(trip[cls].back());
  }
  for(auto& [k, s] : trip)
    q.roundTrip[k] = spreadOf(std::move(s));
  for(auto& [k, s] : far)
    q.distance[k] = spreadOf(std::move(s));
  q.mapped = outer ? double(mapped) / double(outer) : 0;
  size_t refOuter = 0, refMapped = 0;
  for(size_t r = 0; r < nr; r++)
    if(refSkin[r])
    {
      refOuter++;
      refMapped += map.fromReference.mapped[r] ? 1 : 0;
    }
  q.inverseCovers = refOuter ? double(refMapped) / double(refOuter) : 0;
  // stretch and folds: each mapped outer source triangle against the triangle its corners land on
  std::vector<double> stretch;
  size_t              triangles = 0, folded = 0;
  for(size_t t = 0; t * 3 + 2 < source.tris.size(); t++)
  {
    if(t < source.outer.size() && !source.outer[t])
      continue;
    const uint32_t a = source.tris[t * 3], b = source.tris[t * 3 + 1], c = source.tris[t * 3 + 2];
    if(!map.toReference.mapped[a] || !map.toReference.mapped[b] || !map.toReference.mapped[c])
      continue;
    const V e1 = minus(vertexAt(source.positions, b), vertexAt(source.positions, a));
    const V e2 = minus(vertexAt(source.positions, c), vertexAt(source.positions, a));
    const V nrm = cross(e1, e2);
    const double l1 = length(e1), area2 = length(nrm);
    if(l1 < 1e-9 || area2 < 1e-12)
      continue;
    // the source triangle in its own plane: e1 along x
    const V      u  = {e1[0] / l1, e1[1] / l1, e1[2] / l1};
    const V      w  = unit(cross(nrm, e1));
    const double s11 = l1, s21 = 0, s12 = dot(e2, u), s22 = dot(e2, w);
    const V      pa = placeOf(map.toReference, a, reference.positions);
    const V      f1 = minus(placeOf(map.toReference, b, reference.positions), pa);
    const V      f2 = minus(placeOf(map.toReference, c, reference.positions), pa);
    triangles++;
    if(dot(cross(f1, f2), nrm) <= 0)
    {
      folded++;
      q.foldedBy[regionNames(source.regions[a])]++;
      q.foldedTris.push_back(uint32_t(t));
      continue;
    }
    // J = F S^-1 (3x2), S = [[s11 s12] [s21 s22]]; its singular values from J^T J
    const double det = s11 * s22 - s12 * s21;
    V            j1, j2;
    for(size_t i = 0; i < 3; i++)
    {
      j1[i] = (f1[i] * s22 - f2[i] * s21) / det;
      j2[i] = (-f1[i] * s12 + f2[i] * s11) / det;
    }
    const double g11 = dot(j1, j1), g12 = dot(j1, j2), g22 = dot(j2, j2);
    const double mean = (g11 + g22) / 2, d = std::sqrt(std::max(0.0, mean * mean - (g11 * g22 - g12 * g12)));
    stretch.push_back(std::sqrt(mean + d));
    if(stretch.back() > 1.5)
      q.stretchedBy[regionNames(source.regions[a])]++;
  }
  const MapQuality::Spread st = spreadOf(stretch);
  q.stretchP95                = st.p95;
  q.stretchMax                = st.max;
  q.foldFree                  = triangles ? double(triangles - folded) / double(triangles) : 0;
  return q;
}

std::string cloneFile(wardrobe::Generation generation, const std::string& gender)
{
  const std::string sex = gender == "male" ? "Male" : "Female";
  std::string       name;
  switch(generation)
  {
    case wardrobe::Generation::eG2: name = "CloneGenesis2" + sex; break;
    case wardrobe::Generation::eG3: name = "CloneGenesis3" + sex; break;
    // Genesis 8.1 shares Genesis 8's mesh
    case wardrobe::Generation::eG8:
    case wardrobe::Generation::eG81: name = "CloneGenesis8" + sex; break;
    case wardrobe::Generation::eG9:
    case wardrobe::Generation::eSkyrim: return {};
  }
  return "data/DAZ 3D/Genesis 9/Base/Morphs/Daz 3D/Clones/" + name + ".dsf";
}

std::optional<Body> cloneReference(dson::Store& store, wardrobe::Generation generation, const std::string& gender)
{
  const std::string file = cloneFile(generation, gender);
  const auto        g9   = wardrobe::loadBaseCage(store, wardrobe::Generation::eG9, "");
  if(file.empty() || !g9)
    return std::nullopt;
  Body reference = bodyOf(*g9);
  try
  {
    const auto doc = store.load(file);
    if(!doc)
      return std::nullopt;
    const dson::Json* mods  = dson::get(&doc->json, "modifier_library");
    size_t            moved = 0;
    if(mods && mods->is_array())
      for(const dson::Json& m : *mods)
      {
        const auto rows = doc->rows(dson::get(dson::get(dson::get(&m, "morph"), "deltas"), "values"));
        for(int i = 0; rows && i < rows->count; i++)
        {
          const double v = rows->cell(i, 0);
          if(v < 0 || size_t(v) * 3 + 2 >= reference.positions.size())
            continue;
          for(int c = 0; c < 3; c++)
            reference.positions[size_t(v) * 3 + size_t(c)] += float(rows->cell(i, 1 + c));
          moved++;
        }
      }
    if(!moved)
      return std::nullopt;
  }
  catch(const std::exception&)
  {
    return std::nullopt;
  }
  return reference;
}

std::vector<float> intoRest(const wardrobe::BaseCage& reference, const std::vector<float>& shaped, const Body& source, std::vector<LimbTurn>* turns)
{
  const std::vector<wardrobe::SkelBone>& bones = reference.skeleton;
  const size_t                           nb = bones.size(), nv = shaped.size() / 3;
  std::vector<float>                     out = shaped;
  if(!nb || reference.skin.start.size() != nv + 1)
    return out;
  const std::vector<RegionMask> regions = vertexRegions(reference.skin, bones, nv);
  using R                               = rig::Region;

  // The units turned: each limb part, each finger and each toe as one, on each side. A unit turns about
  // its joint, found on each body as the middle of the band where its region meets the one above (the
  // knee where thigh meets shin); a limb part's first turn points it at the band where it meets the next.
  struct Unit
  {
    const char*              name;
    R                        own, above;
    std::vector<R>           next;
    int                      chain, level;
    char                     side = 0;
    Quat                     world;
    Vec3                     pivot;
    bool                     found = false;
  };
  std::vector<Unit> units;
  const std::vector<R> toes{R::eBigToe, R::eIndexToe, R::eMidToe, R::eRingToe, R::ePinkyToe};
  for(const char side : {'l', 'r'})
  {
    const int arm = side == 'l' ? 0 : 1, leg = arm + 2;
    units.push_back({"upperarm", R::eUpperArm, R::eCollar, {R::eForearm}, arm, 0, side});
    units.push_back({"forearm", R::eForearm, R::eUpperArm, {R::eHand}, arm, 1, side});
    units.push_back({"hand", R::eHand, R::eForearm, {R::eMid}, arm, 2, side});
    for(const auto& [name, r] : {std::pair{"thumb", R::eThumb}, std::pair{"index", R::eIndex}, std::pair{"mid", R::eMid}, std::pair{"ring", R::eRing}, std::pair{"pinky", R::ePinky}})
      units.push_back({name, r, R::eHand, {}, arm, 3, side});
    units.push_back({"thigh", R::eThigh, R::ePelvis, {R::eShin}, leg, 0, side});
    units.push_back({"shin", R::eShin, R::eThigh, {R::eFoot}, leg, 1, side});
    units.push_back({"foot", R::eFoot, R::eShin, toes, leg, 2, side});
    for(const auto& [name, r] : {std::pair{"bigtoe", R::eBigToe}, std::pair{"indextoe", R::eIndexToe}, std::pair{"midtoe", R::eMidToe}, std::pair{"ringtoe", R::eRingToe}, std::pair{"pinkytoe", R::ePinkyToe}})
      units.push_back({name, r, R::eFoot, {}, leg, 3, side});
  }
  // the pelvis has no side; a limb's region is its side's alone
  const auto sided = [](R r, char side) { return r == R::ePelvis ? regionBit(r) : regionBit(r, side); };
  const auto bandMiddle = [](const std::vector<float>& pos, const std::vector<RegionMask>& reg, RegionMask a, RegionMask b) -> std::optional<Vec3> {
    Vec3   sum;
    size_t n = 0;
    for(size_t v = 0; v < reg.size() && v * 3 + 2 < pos.size(); v++)
      if((reg[v] & a) && (reg[v] & b))
      {
        sum = sum + Vec3{pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]};
        n++;
      }
    if(!n)
      return std::nullopt;
    return sum * (1.0 / double(n));
  };
  // the first turns: a limb part pointed as the source's; a finger or toe as its hand or foot
  for(Unit& u : units)
  {
    const RegionMask own = sided(u.own, u.side), above = sided(u.above, u.side);
    const auto       from = bandMiddle(shaped, regions, own, above);
    if(!from)
      continue;
    u.found = true;
    u.pivot = *from;
    for(const Unit& p : units)
      if(p.chain == u.chain && p.level == u.level - 1 && p.found && u.level == 3)
        u.world = p.world;
    if(u.next.empty())
      continue;
    RegionMask next = 0;
    for(const R r : u.next)
      next |= sided(r, u.side);
    const auto to = bandMiddle(shaped, regions, own, next);
    const auto sFrom = bandMiddle(source.positions, source.regions, own, above), sTo = bandMiddle(source.positions, source.regions, own, next);
    if(!to || !sFrom || !sTo)
    {
      for(const Unit& p : units)
        if(p.chain == u.chain && p.level == u.level - 1 && p.found)
          u.world = p.world;
      continue;
    }
    const Vec3 dr = *to - *from, ds = *sTo - *sFrom;
    if(length(dr) > 1e-6 && length(ds) > 1e-6)
      u.world = quatFromUnitVectors(normalize(dr), normalize(ds));
  }

  // parents, and an order with every bone after its parent; each bone's unit
  std::unordered_map<std::string, size_t> byName;
  for(size_t b = 0; b < nb; b++)
    byName.emplace(bones[b].name, b);
  std::vector<int> parent(nb, -1);
  for(size_t b = 0; b < nb; b++)
    if(bones[b].parent)
      if(const auto it = byName.find(*bones[b].parent); it != byName.end() && it->second != b)
        parent[b] = int(it->second);
  std::vector<int> depth(nb, 0);
  for(size_t b = 0; b < nb; b++)
    for(int at = parent[b], guard = 0; at >= 0 && guard < int(nb); at = parent[size_t(at)], guard++)
      depth[b]++;
  std::vector<size_t> order(nb);
  for(size_t b = 0; b < nb; b++)
    order[b] = b;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return depth[a] < depth[b]; });
  std::vector<int> unitOf(nb, -1);
  for(size_t b = 0; b < nb; b++)
  {
    const rig::Name n = rig::parse(bones[b].name);
    if(!n.joint || n.joint->region == R::eNone)
      continue;
    for(size_t k = 0; k < units.size(); k++)
      if(units[k].found && units[k].own == n.joint->region && units[k].side == (n.joint->sided ? n.side : 0))
        unitOf[b] = int(k);
  }

  // The pose: a unit's bones (its twist bones too) turn about its joint; any other bone keeps its
  // parent's turn, so where its own centre is cancels out of the skin.
  std::vector<Quat> world(nb);
  std::vector<Vec3> centre(nb), at(nb);
  const auto        pose = [&] {
    for(const size_t b : order)
    {
      const int u = unitOf[b], p = parent[b];
      centre[b]   = u >= 0 ? units[size_t(u)].pivot : Vec3{bones[b].center[0], bones[b].center[1], bones[b].center[2]};
      world[b]    = u >= 0 ? units[size_t(u)].world : p >= 0 ? world[size_t(p)] : Quat{};
      at[b]       = p >= 0 ? at[size_t(p)] + rotate(centre[b] - centre[size_t(p)], world[size_t(p)]) : centre[b];
    }
  };
  const auto skinned = [&](size_t v) {
    const Vec3 p{shaped[v * 3], shaped[v * 3 + 1], shaped[v * 3 + 2]};
    Vec3       sum;
    double     total = 0;
    for(int32_t i = reference.skin.start[v]; i < reference.skin.start[v + 1]; i++)
    {
      const int    b = reference.skin.bone[size_t(i)];
      const double w = reference.skin.weight[size_t(i)];
      if(b < 0 || size_t(b) >= nb || w <= 0)
        continue;
      sum = sum + (at[size_t(b)] + rotate(p - centre[size_t(b)], world[size_t(b)])) * w;
      total += w;
    }
    return total > 0 ? sum * (1.0 / total) : p;
  };
  pose();

  // Each unit's turn fitted, parents first: its own vertices, as posed, to the closest points of the
  // source's skin in its region, by the turn about its joint that takes them there best; a few rounds,
  // as the closest points move with the turn. The units below it turn with it. A unit the source has
  // less than half the skin of (a carried body cut at the shoulder) isn't fitted: its few points would
  // turn the whole limb onto them.
  if(!source.tris.empty())
  {
    const physics::SurfaceIndex index(source.positions, source.tris);
    std::vector<RegionMask>     triRegions(source.tris.size() / 3, 0);
    for(size_t t = 0; t < triRegions.size(); t++)
      for(size_t j = 0; j < 3; j++)
        triRegions[t] |= source.tris[t * 3 + j] < source.regions.size() ? source.regions[source.tris[t * 3 + j]] : 0;
    Body shapedBody;
    shapedBody.positions                 = shaped;
    shapedBody.tris                      = reference.tris;
    shapedBody.regions                   = regions;
    const std::vector<double> refArea    = vertexAreas(shapedBody);
    const std::vector<double> sourceArea = vertexAreas(source);
    const auto                skinOf     = [](const Body& b, const std::vector<double>& area, RegionMask own) {
      double sum = 0;
      for(size_t v = 0; v < area.size() && v < b.regions.size(); v++)
        if(b.regions[v] & own)
          sum += area[v];
      return sum;
    };
    constexpr double kReach2 = 8.0 * 8.0;
    for(int level = 0; level < 4; level++)
      for(size_t k = 0; k < units.size(); k++)
      {
        Unit& u = units[k];
        if(!u.found || u.level != level)
          continue;
        const RegionMask own = sided(u.own, u.side);
        if(skinOf(source, sourceArea, own) < 0.5 * skinOf(shapedBody, refArea, own))
          continue;
        std::vector<size_t> verts;
        for(size_t v = 0; v < nv; v++)
          if(regions[v] == own)
            verts.push_back(v);
        if(verts.size() < 8)
          continue;
        for(int round = 0; round < 12; round++)
        {
          // the unit's joint as posed
          Vec3 joint = u.pivot;
          for(size_t b = 0; b < nb; b++)
            if(unitOf[b] == int(k))
            {
              joint = at[b];
              break;
            }
          std::vector<Vec3> from, to;
          physics::Stamps&  stamps = physics::threadStamps();
          for(const size_t v : verts)
          {
            const Vec3    p      = skinned(v);
            const auto    accept = [&](size_t t) { return (t >= source.outer.size() || source.outer[t]) && (triRegions[t] & own); };
            const int32_t id     = stamps.next(index.triangles());
            const auto    hit    = index.nearest(p.x, p.y, p.z, accept, kReach2, stamps.stamp, id);
            if(hit.tri < 0)
              continue;
            Vec3 q;
            for(size_t j = 0; j < 3; j++)
            {
              const uint32_t c = source.tris[size_t(hit.tri) * 3 + j];
              const double   w = j == 0 ? hit.u : j == 1 ? hit.v : hit.w;
              q                = q + Vec3{source.positions[c * 3], source.positions[c * 3 + 1], source.positions[c * 3 + 2]} * w;
            }
            from.push_back(p - joint);
            to.push_back(q - joint);
          }
          if(from.size() < 8)
            break;
          const Quat step = bestTurn(from, to);
          for(Unit& d : units)
            if(d.found && d.chain == u.chain && (&d == &u || d.level > u.level))
              d.world = (step * d.world).normalized();
          pose();
          if(2 * std::acos(std::min(1.0, std::abs(step.w))) < 0.01 * kDeg)
            break;
        }
      }
  }
  if(turns)
    for(const Unit& u : units)
      if(u.found)
        turns->push_back({u.name, u.side, 2 * std::acos(std::min(1.0, std::abs(u.world.w))) / kDeg});

  for(size_t v = 0; v < nv; v++)
  {
    const Vec3 p   = skinned(v);
    out[v * 3]     = float(p.x);
    out[v * 3 + 1] = float(p.y);
    out[v * 3 + 2] = float(p.z);
  }
  return out;
}

std::optional<CloneInputs> cloneInputs(dson::Store& store, wardrobe::Generation generation, const std::string& gender)
{
  CloneInputs in;
  in.base = wardrobe::loadBaseCage(store, generation, gender);
  in.g9   = wardrobe::loadBaseCage(store, wardrobe::Generation::eG9, "");
  if(!in.base || !in.g9)
    return std::nullopt;
  auto reference = cloneReference(store, generation, gender);
  if(!reference)
    return std::nullopt;
  in.reference = std::move(*reference);
  return in;
}

TypeMap makeCloneMap(const CloneInputs& in)
{
  TypeMap out{bodyOf(*in.base), in.reference, {}, {}};
  out.reference.positions = intoRest(*in.g9, in.reference.positions, out.source, &out.turns);
  out.map                 = mapBodies(out.source, out.reference);
  out.map.frameTurn       = frameTurns(vertexFrames(out.source), vertexFrames(bodyOf(*in.g9)), out.map.toReference);
  return out;
}

std::optional<TypeMap> cloneMap(dson::Store& store, wardrobe::Generation generation, const std::string& gender)
{
  const auto in = cloneInputs(store, generation, gender);
  if(!in)
    return std::nullopt;
  return cachedCloneMap(*in);
}

}  // namespace fv::bodymap
