#include <figure/dforce/solver.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

namespace fv::dforce {

namespace {

constexpr float DT    = 1.f / 30.f;
constexpr float DEPTH = 3;
constexpr float CAP   = 2;
constexpr float CELL  = 2;

enum class EdgeKind
{
  eStretch,
  eShear,
  eBend,
};

struct Edge
{
  int      a, b;
  float    rest, stiff;
  EdgeKind kind;
  float    ratio, buckle;
};

struct Tri
{
  int   mesh, a, b, c;
  int   layer;
  float offset, friction;
  bool  self;
};

using V3 = std::array<float, 3>;

struct Built
{
  int                  nv = 0;
  std::vector<int>     base;
  std::vector<float>   rest, posed, strength, damping, density, friction, offset, grav, inv;
  std::vector<int>     layer, meshOf;
  std::vector<uint8_t> visible, collides, dynamic;
  std::vector<Edge>    edges;
  std::vector<Tri>     tris;
};

float dist(const std::vector<float>& p, int a, int b)
{
  const float dx = p[b * 3] - p[a * 3], dy = p[b * 3 + 1] - p[a * 3 + 1], dz = p[b * 3 + 2] - p[a * 3 + 2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float compliance(float stiffness)
{
  const float s = std::min(1.f, std::max(stiffness, 0.01f));
  return ((1 - s) / s) * 0.02f;
}

Built build(const std::vector<SettleMesh>& meshes)
{
  Built b;
  int   nv = 0;
  for(const SettleMesh& m : meshes)
  {
    b.base.push_back(nv);
    nv += int(m.vertexCount());
  }
  b.nv = nv;
  b.rest.resize(size_t(nv) * 3);
  b.posed.resize(size_t(nv) * 3);
  b.grav.resize(size_t(nv) * 3);
  for(auto* v : {&b.strength, &b.damping, &b.density, &b.friction, &b.offset, &b.inv})
    v->assign(size_t(nv), 0.f);
  b.layer.assign(size_t(nv), 0);
  b.meshOf.assign(size_t(nv), 0);
  b.visible.assign(size_t(nv), 0);
  b.collides.assign(size_t(nv), 0);
  b.dynamic.assign(size_t(nv), 0);
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const SettleMesh& m   = meshes[mi];
    const int         o   = b.base[mi];
    const int         n   = int(m.vertexCount());
    const bool        sim = m.kind != SettleMesh::Kind::eCollider;
    std::copy(m.rest.begin(), m.rest.end(), b.rest.begin() + o * 3);
    std::copy(m.posed.begin(), m.posed.end(), b.posed.begin() + o * 3);
    for(int v = 0; v < n; v++)
    {
      const int i      = o + v;
      b.strength[i]    = sim && m.visible[v] ? m.strength[v] : 0;
      b.damping[i]     = m.damping[v];
      b.density[i]     = m.density[v] > 1 ? m.density[v] : 180;
      b.friction[i]    = m.friction[v];
      b.offset[i]      = m.offset[v] > 0 ? m.offset[v] : 0.05f;
      b.layer[i]       = m.layer[v];
      b.visible[i]     = m.visible[v];
      b.collides[i]    = m.collide[v];
      b.dynamic[i]     = sim && m.visible[v] ? 1 : 0;
      b.meshOf[i]      = int(mi);
      b.grav[i * 3]     = m.gravity[0];
      b.grav[i * 3 + 1] = m.gravity[1];
      b.grav[i * 3 + 2] = m.gravity[2];
    }
  }
  const auto g = [&](size_t mi, int v) { return b.base[mi] + v; };
  std::unordered_map<uint64_t, int> seen;
  const auto add = [&](int a, int bb, float stiff, EdgeKind kind, float ratio = 1, float buckle = 0) {
    if(a == bb)
      return;
    const int      lo = std::min(a, bb), hi = std::max(a, bb);
    const uint64_t key = ((uint64_t(lo) << 32) | uint64_t(hi)) * 3 + uint64_t(kind);
    if(seen.count(key))
      return;
    const float span = dist(b.rest, a, bb);
    if(span < 1e-6f)
      return;
    float scale = 1;
    if(kind == EdgeKind::eStretch)
    {
      const size_t ma = size_t(b.meshOf[a]), mb = size_t(b.meshOf[bb]);
      scale = (meshes[ma].contraction[size_t(a - b.base[ma])] + meshes[mb].contraction[size_t(bb - b.base[mb])]) * 0.5f;
    }
    seen[key] = int(b.edges.size());
    b.edges.push_back({a, bb, span * scale, stiff, kind, ratio, buckle});
  };
  // opposite vertex of an edge the first time a triangle uses it, for the bend hinge
  std::unordered_map<uint64_t, int> hinge;
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const SettleMesh& m = meshes[mi];
    const int         n = int(m.vertexCount());
    if(m.kind != SettleMesh::Kind::eCollider)
      for(size_t p = 0; p + 1 < m.polyStart.size(); p++)
      {
        const int s = m.polyStart[p], e = m.polyStart[p + 1];
        const int cnt = e - s;
        if(cnt < 2)
          continue;
        for(int k = 0; k < cnt; k++)
        {
          const int va = m.polyVerts[s + k], vb = m.polyVerts[s + (k + 1) % cnt];
          add(g(mi, va), g(mi, vb), (m.stretch[va] + m.stretch[vb]) * 0.5f, EdgeKind::eStretch);
        }
        if(cnt == 4)
        {
          const int   v0 = m.polyVerts[s], v1 = m.polyVerts[s + 1], v2 = m.polyVerts[s + 2], v3 = m.polyVerts[s + 3];
          const float sh = (m.shear[v0] + m.shear[v1] + m.shear[v2] + m.shear[v3]) * 0.25f;
          add(g(mi, v0), g(mi, v2), sh, EdgeKind::eShear);
          add(g(mi, v1), g(mi, v3), sh, EdgeKind::eShear);
        }
      }
    if(m.kind == SettleMesh::Kind::eAddon)
      continue;
    for(size_t t = 0; t + 2 < m.tris.size(); t += 3)
    {
      const uint32_t ia = m.tris[t], ib = m.tris[t + 1], ic = m.tris[t + 2];
      if(ia >= uint32_t(n) || ib >= uint32_t(n) || ic >= uint32_t(n))
        continue;
      const int   a = g(mi, int(ia)), bb = g(mi, int(ib)), c = g(mi, int(ic));
      const int   lay = int(std::lround((m.layer[ia] + m.layer[ib] + m.layer[ic]) / 3.0));
      const float off = (m.offset[ia] + m.offset[ib] + m.offset[ic]) / 3;
      const float frRaw = (m.friction[ia] + m.friction[ib] + m.friction[ic]) / 3;
      const float fr    = std::isfinite(frRaw) ? frRaw : 0.4f;
      const bool  self  = m.selfCollide[ia] || m.selfCollide[ib] || m.selfCollide[ic];
      const bool  shown = m.visible[ia] || m.visible[ib] || m.visible[ic] || m.kind == SettleMesh::Kind::eCollider;
      // a vertex on a colliding and a non-colliding surface collides, so the face needs all three
      if(shown && m.collide[ia] && m.collide[ib] && m.collide[ic])
        b.tris.push_back({int(mi), a, bb, c, lay, off > 0 ? off : 0.05f, fr, self});
      if(m.kind == SettleMesh::Kind::eCollider)
        continue;
      const int corners[3] = {a, bb, c};
      for(int k = 0; k < 3; k++)
      {
        const int      u = corners[k], v = corners[(k + 1) % 3], opp = corners[(k + 2) % 3];
        const uint64_t key = (uint64_t(std::min(u, v)) << 32) | uint64_t(std::max(u, v));
        const auto     it  = hinge.find(key);
        if(it == hinge.end())
          hinge[key] = opp;
        else if(it->second != opp)
        {
          const float stiff = (m.bend[ia] + m.bend[ib] + m.bend[ic]) / 3;
          const float buck  = (m.buckling[ia] + m.buckling[ib] + m.buckling[ic]) / 3;
          const float ratio = (m.bucklingRatio[ia] + m.bucklingRatio[ib] + m.bucklingRatio[ic]) / 3;
          add(it->second, opp, stiff, EdgeKind::eBend, ratio, buck);
        }
      }
    }
  }
  for(int i = 0; i < nv; i++)
    b.inv[i] = b.dynamic[i] && b.strength[i] > 1e-4f ? 180.f / b.density[i] : 0.f;
  return b;
}

int cellOf(float p)
{
  return int(std::floor(p / CELL));
}

uint64_t hashKey(int ix, int iy, int iz)
{
  return (uint64_t(uint32_t(ix) & 0x1FFFFF) << 42) | (uint64_t(uint32_t(iy) & 0x1FFFFF) << 21) | uint64_t(uint32_t(iz) & 0x1FFFFF);
}

using Buckets = std::unordered_map<uint64_t, std::vector<int>>;

// Spatial hash of the collision faces over their box in both `a` and `b` (the same array for one pose).
Buckets bucketTris(const std::vector<Tri>& tris, const std::vector<float>& a, const std::vector<float>& b)
{
  Buckets buckets;
  for(size_t t = 0; t < tris.size(); t++)
  {
    const Tri& tri = tris[t];
    float      lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(int k = 0; k < 6; k++)
    {
      const int   v = k % 3 == 0 ? tri.a : k % 3 == 1 ? tri.b : tri.c;
      const auto& p = k < 3 ? a : b;
      for(int c = 0; c < 3; c++)
      {
        lo[c] = std::min(lo[c], p[v * 3 + c]);
        hi[c] = std::max(hi[c], p[v * 3 + c]);
      }
    }
    for(int ix = cellOf(lo[0]); ix <= cellOf(hi[0]); ix++)
      for(int iy = cellOf(lo[1]); iy <= cellOf(hi[1]); iy++)
        for(int iz = cellOf(lo[2]); iz <= cellOf(hi[2]); iz++)
          buckets[hashKey(ix, iy, iz)].push_back(int(t));
  }
  return buckets;
}

std::optional<V3> triBasis(const std::vector<float>& p, int a, int b, int c)
{
  const float ax = p[b * 3] - p[a * 3], ay = p[b * 3 + 1] - p[a * 3 + 1], az = p[b * 3 + 2] - p[a * 3 + 2];
  const float bx = p[c * 3] - p[a * 3], by = p[c * 3 + 1] - p[a * 3 + 1], bz = p[c * 3 + 2] - p[a * 3 + 2];
  const float x = ay * bz - az * by, y = az * bx - ax * bz, z = ax * by - ay * bx;
  const float len = std::sqrt(x * x + y * y + z * z);
  if(len < 1e-8f)
    return std::nullopt;
  return V3{x / len, y / len, z / len};
}

struct Closest
{
  float signedDistance, dist;
};

// Closest point of `q` on the triangle in `tri`, and the signed distance along `n`.
Closest closest(float qx, float qy, float qz, const std::vector<float>& tri, int a, int b, int c, const V3& n)
{
  const float abx = tri[b * 3] - tri[a * 3], aby = tri[b * 3 + 1] - tri[a * 3 + 1], abz = tri[b * 3 + 2] - tri[a * 3 + 2];
  const float acx = tri[c * 3] - tri[a * 3], acy = tri[c * 3 + 1] - tri[a * 3 + 1], acz = tri[c * 3 + 2] - tri[a * 3 + 2];
  const float aqx = qx - tri[a * 3], aqy = qy - tri[a * 3 + 1], aqz = qz - tri[a * 3 + 2];
  const float d00 = abx * abx + aby * aby + abz * abz;
  const float d01 = abx * acx + aby * acy + abz * acz;
  const float d11 = acx * acx + acy * acy + acz * acz;
  const float d20 = abx * aqx + aby * aqy + abz * aqz;
  const float d21 = acx * aqx + acy * aqy + acz * aqz;
  const float den = d00 * d11 - d01 * d01;
  float       bv = 0, cv = 0;
  if(den > 1e-12f)
  {
    bv = std::min(1.f, std::max(0.f, (d11 * d20 - d01 * d21) / den));
    cv = std::min(1.f, std::max(0.f, (d00 * d21 - d01 * d20) / den));
    if(bv + cv > 1)
    {
      const float s = bv + cv;
      bv /= s;
      cv /= s;
    }
  }
  const float cx = tri[a * 3] + abx * bv + acx * cv;
  const float cy = tri[a * 3 + 1] + aby * bv + acy * cv;
  const float cz = tri[a * 3 + 2] + abz * bv + acz * cv;
  const float dx = qx - cx, dy = qy - cy, dz = qz - cz;
  return {dx * n[0] + dy * n[1] + dz * n[2], std::sqrt(dx * dx + dy * dy + dz * dz)};
}

// Moves free vertices that start inside, or within the gap of, a lower layer out to the gap along
// the nearest face of that layer. The nearest face decides inside, so a vertex just outside a thin
// limb is not sent through it by the face on the far side. Only the start moves: rest lengths keep
// the fitted shape.
void separateStart(std::vector<float>& x, const Built& built)
{
  const auto buckets = bucketTris(built.tris, x, x);
  const int  reach   = int(std::ceil(DEPTH / CELL));
  std::vector<uint32_t> seen(built.tris.size(), 0);
  uint32_t              stamp = 0;
  for(int i = 0; i < built.nv; i++)
  {
    if(!built.dynamic[i] || built.inv[i] <= 0 || !built.collides[i])
      continue;
    const float qx = x[i * 3], qy = x[i * 3 + 1], qz = x[i * 3 + 2];
    const int   cx = cellOf(qx), cy = cellOf(qy), cz = cellOf(qz);
    stamp++;
    int             best = -1;
    float           bestDist = DEPTH, bestSigned = 0;
    std::optional<V3> bestN;
    for(int ix = cx - reach; ix <= cx + reach; ix++)
      for(int iy = cy - reach; iy <= cy + reach; iy++)
        for(int iz = cz - reach; iz <= cz + reach; iz++)
        {
          const auto it = buckets.find(hashKey(ix, iy, iz));
          if(it == buckets.end())
            continue;
          for(int t : it->second)
          {
            if(seen[size_t(t)] == stamp)
              continue;
            seen[size_t(t)] = stamp;
            const Tri& tri  = built.tris[size_t(t)];
            if(tri.mesh == built.meshOf[i] || tri.layer >= built.layer[i])
              continue;
            const auto n = triBasis(x, tri.a, tri.b, tri.c);
            if(!n)
              continue;
            const Closest c = closest(qx, qy, qz, x, tri.a, tri.b, tri.c, *n);
            if(c.dist >= bestDist)
              continue;
            best       = t;
            bestDist   = c.dist;
            bestSigned = c.signedDistance;
            bestN      = n;
          }
        }
    if(best < 0 || !bestN)
      continue;
    const float gap = std::max(built.tris[size_t(best)].offset, built.offset[i]);
    if(bestSigned >= gap)
      continue;
    const float push = gap - bestSigned;
    for(int k = 0; k < 3; k++)
      x[i * 3 + k] += (*bestN)[size_t(k)] * push;
  }
}

void separate(int i, const Tri& tri, std::vector<float>& x, const std::vector<float>& prev, const std::vector<float>& cur,
              const std::vector<float>& before, int particleLayer, float particleOffset, float particleFriction)
{
  const auto n0 = triBasis(before, tri.a, tri.b, tri.c);
  const auto n1 = triBasis(cur, tri.a, tri.b, tri.c);
  if(!n0 || !n1)
    return;
  const Closest was = closest(prev[i * 3], prev[i * 3 + 1], prev[i * 3 + 2], before, tri.a, tri.b, tri.c, *n0);
  // higher layer hits the outside (positive normal); a lower layer hits the inside; the same layer stays on the side it came from
  int direction;
  if(particleLayer > tri.layer)
  {
    if(was.signedDistance < -0.05f)
      return;
    direction = 1;
  }
  else if(particleLayer < tri.layer)
  {
    if(was.signedDistance > 0.05f)
      return;
    direction = -1;
  }
  else
    direction = was.signedDistance >= 0 ? 1 : -1;
  const Closest now = closest(x[i * 3], x[i * 3 + 1], x[i * 3 + 2], cur, tri.a, tri.b, tri.c, *n1);
  const float   gap = std::max(tri.offset, particleOffset);
  const float   target = float(direction) * gap;
  if(direction > 0 ? now.signedDistance >= target : now.signedDistance <= target)
    return;
  if(now.dist > DEPTH)
    return;
  // past the edge, not under the face: the closest point is mostly sideways of the vertex
  if(std::fabs(now.signedDistance) < 0.6f * now.dist)
    return;
  const float push  = std::min(std::fabs(target - now.signedDistance), CAP);
  const float along = target - now.signedDistance > 0 ? 1.f : (target - now.signedDistance < 0 ? -1.f : 0.f);
  const V3&   n     = *n1;
  for(int k = 0; k < 3; k++)
    x[i * 3 + k] += n[size_t(k)] * push * along;
  // friction sheds the part of this step's motion that slid along the face. Velocity is rebuilt
  // from the position afterwards, so the correction has to be on the position.
  const float dx = x[i * 3] - prev[i * 3], dy = x[i * 3 + 1] - prev[i * 3 + 1], dz = x[i * 3 + 2] - prev[i * 3 + 2];
  const float dn = dx * n[0] + dy * n[1] + dz * n[2];
  const float fr = std::min(0.95f, std::max(0.f, (tri.friction + particleFriction) * 0.5f));
  x[i * 3] -= (dx - n[0] * dn) * fr;
  x[i * 3 + 1] -= (dy - n[1] * dn) * fr;
  x[i * 3 + 2] -= (dz - n[2] * dn) * fr;
}

// Pushes dynamic vertices out of faces they were outside of. `prev` is the position at the start of the step.
void collide(std::vector<float>& x, const std::vector<float>& prev, const Built& built, const std::vector<float>& triPos,
             const std::vector<float>& triPrev)
{
  const auto            buckets = bucketTris(built.tris, triPos, triPrev);
  std::vector<uint32_t> seen(built.tris.size(), 0);
  uint32_t              stamp = 1;
  for(int i = 0; i < built.nv; i++)
  {
    if(!built.dynamic[i] || built.inv[i] <= 0 || !built.collides[i])
      continue;
    int lo[3], hi[3];
    for(int c = 0; c < 3; c++)
    {
      lo[c] = cellOf(std::min(x[i * 3 + c], prev[i * 3 + c]));
      hi[c] = cellOf(std::max(x[i * 3 + c], prev[i * 3 + c]));
    }
    stamp++;
    if(stamp == 0xffffffffu)
    {
      std::fill(seen.begin(), seen.end(), 0u);
      stamp = 1;
    }
    for(int ix = lo[0]; ix <= hi[0]; ix++)
      for(int iy = lo[1]; iy <= hi[1]; iy++)
        for(int iz = lo[2]; iz <= hi[2]; iz++)
        {
          const auto it = buckets.find(hashKey(ix, iy, iz));
          if(it == buckets.end())
            continue;
          for(int t : it->second)
          {
            if(seen[size_t(t)] == stamp)
              continue;
            seen[size_t(t)] = stamp;
            const Tri& tri  = built.tris[size_t(t)];
            if(tri.a == i || tri.b == i || tri.c == i)
              continue;
            if(built.meshOf[i] == tri.mesh && !tri.self)
              continue;
            separate(i, tri, x, prev, triPos, triPrev, built.layer[i], built.offset[i], built.friction[i]);
          }
        }
  }
}

void solveEdges(std::vector<float>& x, const std::vector<float>& inv, const std::vector<Edge>& edges, std::vector<double>& lam)
{
  const float atScale = 1.f / (DT * DT);
  for(size_t e = 0; e < edges.size(); e++)
  {
    const Edge& edge = edges[e];
    const int   a = edge.a, b = edge.b;
    const float wa = inv[a], wb = inv[b];
    const float w = wa + wb;
    if(w <= 0)
      continue;
    const float dx = x[b * 3] - x[a * 3], dy = x[b * 3 + 1] - x[a * 3 + 1], dz = x[b * 3 + 2] - x[a * 3 + 2];
    const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if(len < 1e-8f)
      continue;
    float stiff = edge.stiff;
    if(edge.kind == EdgeKind::eBend && len < edge.rest * edge.ratio)
      stiff = edge.stiff * edge.buckle;
    const float C  = len - edge.rest;
    const float nx = dx / len, ny = dy / len, nz = dz / len;
    const float at = compliance(stiff) * atScale;
    const float dl = float((-C - at * lam[e]) / (w + at));
    lam[e] += dl;
    if(wa > 0)
    {
      x[a * 3] -= nx * wa * dl;
      x[a * 3 + 1] -= ny * wa * dl;
      x[a * 3 + 2] -= nz * wa * dl;
    }
    if(wb > 0)
    {
      x[b * 3] += nx * wb * dl;
      x[b * 3 + 1] += ny * wb * dl;
      x[b * 3 + 2] += nz * wb * dl;
    }
  }
}

int substepsFor(const Built& built, int forced)
{
  if(forced > 0)
    return std::min(240, forced);
  float maxMove = 0, minOffset = INFINITY;
  for(int i = 0; i < built.nv; i++)
  {
    if(!built.dynamic[i])
    {
      const float dx = built.posed[i * 3] - built.rest[i * 3], dy = built.posed[i * 3 + 1] - built.rest[i * 3 + 1],
                  dz = built.posed[i * 3 + 2] - built.rest[i * 3 + 2];
      maxMove = std::max(maxMove, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    else if(built.offset[i] < minOffset)
      minOffset = built.offset[i];
  }
  if(!std::isfinite(minOffset) || minOffset < 0.05f)
    minOffset = 0.05f;
  if(maxMove <= 0)
    return 8;
  return std::min(180, std::max(8, int(std::ceil(maxMove / (0.5f * minOffset)))));
}

}  // namespace

SettleMesh SettleMesh::collider(std::vector<float> rest, std::vector<float> posed, std::vector<uint32_t> tris, int layer, float offset,
                                float friction, const std::vector<uint8_t>& collide)
{
  SettleMesh m;
  const size_t n = rest.size() / 3;
  m.rest         = std::move(rest);
  m.posed        = std::move(posed);
  m.tris         = std::move(tris);
  m.strength.assign(n, 0.f);
  m.stretch.assign(n, 0.8f);
  m.shear.assign(n, 0.2f);
  m.bend.assign(n, 0.5f);
  m.buckling.assign(n, 0.05f);
  m.bucklingRatio.assign(n, 0.7f);
  m.density.assign(n, 180.f);
  m.contraction.assign(n, 1.f);
  m.damping.assign(n, 0.1f);
  m.friction.assign(n, friction);
  m.offset.assign(n, offset);
  m.layer.assign(n, int16_t(layer));
  m.selfCollide.assign(n, 0);
  m.collide = collide.size() == n ? collide : std::vector<uint8_t>(n, 1);
  m.visible.assign(n, 1);
  return m;
}

std::vector<std::vector<float>> settle(const std::vector<SettleMesh>& meshes, const SettleOptions& opts)
{
  const Built built = build(meshes);
  const int   nv    = built.nv;
  std::vector<float> x = built.rest;
  separateStart(x, built);
  std::vector<float>  prev(size_t(nv) * 3), vel(size_t(nv) * 3, 0.f), triPos(size_t(nv) * 3), triPrev(size_t(nv) * 3);
  std::vector<double> lam(built.edges.size(), 0.0);
  const int           steps = substepsFor(built, opts.substeps);
  const int           hold  = 16;
  const int           total = steps + hold;
  const auto&         rest = built.rest;
  const auto&         posed = built.posed;
  for(int s = 0; s < total; s++)
  {
    const float t0 = std::min(1.f, float(s) / float(steps));
    const float t1 = std::min(1.f, float(s + 1) / float(steps));
    prev = x;
    for(int i = 0; i < nv; i++)
    {
      if(!built.dynamic[i] || built.inv[i] <= 0)
      {
        for(int k = 0; k < 3; k++)
        {
          x[i * 3 + k]   = (1 - t1) * rest[i * 3 + k] + t1 * posed[i * 3 + k];
          vel[i * 3 + k] = 0;
        }
        continue;
      }
      for(int k = 0; k < 3; k++)
        vel[i * 3 + k] += built.grav[i * 3 + k] * DT;
      const float dens   = std::max(30.f, built.density[i]);
      const float damped = std::exp(-(built.damping[i] * 8 + opts.air * (180 / dens)) * DT);
      for(int k = 0; k < 3; k++)
        vel[i * 3 + k] *= damped;
      const float sp = std::sqrt(vel[i * 3] * vel[i * 3] + vel[i * 3 + 1] * vel[i * 3 + 1] + vel[i * 3 + 2] * vel[i * 3 + 2]);
      if(sp > 400)
        for(int k = 0; k < 3; k++)
          vel[i * 3 + k] *= 400 / sp;
      for(int k = 0; k < 3; k++)
        x[i * 3 + k] += vel[i * 3 + k] * DT;
    }
    std::fill(lam.begin(), lam.end(), 0.0);
    for(int it = 0; it < opts.iterations; it++)
      solveEdges(x, built.inv, built.edges, lam);
    for(int i = 0; i < nv; i++)
      for(int k = 0; k < 3; k++)
      {
        const int o = i * 3 + k;
        if(built.dynamic[i])
        {
          triPrev[o] = prev[o];
          triPos[o]  = x[o];
        }
        else
        {
          triPrev[o] = (1 - t0) * rest[o] + t0 * posed[o];
          triPos[o]  = (1 - t1) * rest[o] + t1 * posed[o];
        }
      }
    // the side test reads the particle from `prev`, which is where it was before this step's integration
    collide(x, prev, built, triPos, triPrev);
    for(int i = 0; i < nv; i++)
      if(built.dynamic[i])
        for(int k = 0; k < 3; k++)
          vel[i * 3 + k] = (x[i * 3 + k] - prev[i * 3 + k]) / DT;
  }
  // a partial strength pulls back toward the skinned pose once, so the step count does not change how pinned it reads
  for(int i = 0; i < nv; i++)
  {
    if(!built.dynamic[i])
      continue;
    const float s = built.strength[i];
    if(s >= 1)
      continue;
    for(int k = 0; k < 3; k++)
      x[i * 3 + k] = posed[i * 3 + k] + (x[i * 3 + k] - posed[i * 3 + k]) * s;
  }
  std::vector<std::vector<float>> out;
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const size_t o = size_t(built.base[mi]) * 3;
    out.emplace_back(x.begin() + long(o), x.begin() + long(o + meshes[mi].rest.size()));
  }
  return out;
}

}  // namespace fv::dforce
