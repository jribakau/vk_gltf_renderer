#include <figure/physics/cloth/capsules.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/geometry.hpp>
#include <figure/physics/skin.hpp>

namespace fv::physics::cloth {

namespace {

// Distance of p from the segment a-b.
float segmentDistance(const Vec3& a, const Vec3& b, const Vec3& p)
{
  const Vec3   ab  = b - a;
  const double ab2 = dot(ab, ab);
  double       t   = ab2 > 1e-12 ? dot(p - a, ab) / ab2 : 0.0;
  t                = std::min(1.0, std::max(0.0, t));
  return float(length(p - (a + ab * t)));
}

// The principal axis of points, by power iteration, as the ragdoll's capsule fit has it.
Vec3 principalAxis(const std::vector<Vec3>& pts, const Vec3& mu)
{
  double c[3][3]{};
  for(const Vec3& p : pts)
  {
    const double d[3] = {p.x - mu.x, p.y - mu.y, p.z - mu.z};
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++)
        c[i][j] += d[i] * d[j];
  }
  Vec3 v = normalize(Vec3{1, 0.7, 0.3});
  for(int it = 0; it < 60; it++)
  {
    const Vec3 n{c[0][0] * v.x + c[0][1] * v.y + c[0][2] * v.z, c[1][0] * v.x + c[1][1] * v.y + c[1][2] * v.z,
                 c[2][0] * v.x + c[2][1] * v.y + c[2][2] * v.z};
    if(length(n) < 1e-12)
      break;
    v = normalize(n);
  }
  return v;
}

// Whether p lies in the body: behind its nearest face, by the face's winding. A body with no faces
// (a test's cloud) is taken as closed about it.
bool inBody(const std::vector<float>& pos, const std::vector<uint32_t>& tris, const Vec3& p)
{
  if(tris.size() < 3)
    return true;
  float best   = INFINITY;
  bool  inside = false;
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
  {
    const int a = int(tris[t]), b = int(tris[t + 1]), c = int(tris[t + 2]);
    V3        n;
    if(!triNormal(pos, a, b, c, n))
      continue;
    const Closest k = closest(float(p.x), float(p.y), float(p.z), pos, a, b, c, n);
    if(k.dist < best)
    {
      best   = k.dist;
      inside = k.signedDistance < 0;
    }
  }
  return std::isfinite(best) && inside;
}

struct Fitted
{
  Vec3     a, b;
  float    radius;
  uint16_t bone;
};

}  // namespace

std::optional<Surface> capsuleFloor(const Surface& body)
{
  const size_t n = body.vertexCount();
  if(!body.skinned() || n == 0 || body.restLocal.size() != n * 3)
    return std::nullopt;
  const std::vector<float>& pos = body.restLocal;
  std::vector<Vec3>         all(n);
  for(size_t v = 0; v < n; v++)
    all[v] = {pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]};
  // each vertex's strongest bone
  std::vector<std::vector<size_t>> cluster(body.bones.size());
  for(size_t v = 0; v < n; v++)
  {
    int   best = -1;
    float w    = 0;
    for(size_t k = 0; k < 4; k++)
      if(body.skinWeight[v * 4 + k] > w)
      {
        w    = body.skinWeight[v * 4 + k];
        best = int(body.skinIndex[v * 4 + k]);
      }
    if(best >= 0 && size_t(best) < cluster.size())
      cluster[size_t(best)].push_back(v);
  }
  // each bone's pieces on the pool, gathered in bone order so the floor is the same every time
  std::vector<std::vector<Fitted>> perBone(cluster.size());
  jobs::parallelFor(cluster.size(), 1, [&](size_t begin, size_t end) {
    for(size_t k = begin; k < end; k++)
    {
      const std::vector<size_t>& verts = cluster[k];
      if(verts.size() < 40)
        continue;
      std::vector<Vec3> pts;
      pts.reserve(verts.size());
      Vec3 mu;
      for(size_t v : verts)
      {
        pts.push_back(all[v]);
        mu = mu + all[v];
      }
      mu              = mu * (1.0 / double(pts.size()));
      const Vec3 axis = principalAxis(pts, mu);
      std::vector<double> t, r;
      t.reserve(pts.size());
      r.reserve(pts.size());
      for(const Vec3& p : pts)
      {
        const double along = dot(p - mu, axis);
        t.push_back(along);
        r.push_back(length(p - mu - axis * along));
      }
      std::sort(t.begin(), t.end());
      std::nth_element(r.begin(), r.begin() + long(r.size() / 2), r.end());
      const double median = r[r.size() / 2];
      const double lo = t[size_t(double(t.size() - 1) * 0.02)], hi = t[size_t(double(t.size() - 1) * 0.98)];
      const double len = hi - lo;
      if(len < kFloorLeast || median < kFloorLeastRadius)
        continue;
      const int pieces = std::clamp(int(std::lround(len / kFloorPiece)), 1, 4);
      for(int piece = 0; piece < pieces; piece++)
      {
        const double pLo = lo + len * piece / pieces, pHi = lo + len * (piece + 1) / pieces;
        const double pLen = pHi - pLo;
        // the body's vertices that could lie within the thickest capsule this piece could be
        const Vec3          mid = mu + axis * ((pLo + pHi) / 2);
        const double        far = pLen / 2 + median + 1.0;
        std::vector<size_t> near;
        for(size_t v = 0; v < n; v++)
          if(length(all[v] - mid) <= far)
            near.push_back(v);
        // as thick as every vertex allows, its ends drawn in by up to 60% each when that makes it thicker (a
        // limb's closed end, a hand's or a foot's, binds a capsule that reaches it to nothing)
        double bestScore = 0, bestLo = pLo, bestHi = pHi, bestR = 0;
        for(int sl = 0; sl <= 6; sl++)
          for(int sh = 0; sh <= 6; sh++)
          {
            const double cLo = pLo + pLen * 0.1 * sl, cHi = pHi - pLen * 0.1 * sh;
            if(cHi < cLo)
              continue;
            const Vec3   a = mu + axis * cLo, b = mu + axis * cHi;
            double       least = median;
            for(size_t v : near)
              least = std::min(least, double(segmentDistance(a, b, all[v])));
            const double radius = least - kFloorMargin;
            if(radius < kFloorLeastRadius)
              continue;
            const double score = radius * radius * (cHi - cLo + 2 * radius);
            if(score > bestScore)
            {
              bestScore = score;
              bestLo    = cLo;
              bestHi    = cHi;
              bestR     = radius;
            }
          }
        if(bestScore <= 0)
          continue;
        // and in the body: a floor fitted to a cluster bent about its axis would hang beside it
        if(!inBody(pos, body.tris, mu + axis * ((bestLo + bestHi) / 2)))
          continue;
        perBone[k].push_back({mu + axis * bestLo, mu + axis * bestHi, float(bestR), uint16_t(k)});
      }
    }
  });
  std::vector<Fitted> fitted;
  for(const std::vector<Fitted>& f : perBone)
    fitted.insert(fitted.end(), f.begin(), f.end());
  if(fitted.empty())
    return std::nullopt;
  std::vector<float> ends;
  ends.reserve(fitted.size() * 6);
  for(const Fitted& f : fitted)
    ends.insert(ends.end(), {float(f.a.x), float(f.a.y), float(f.a.z), float(f.b.x), float(f.b.y), float(f.b.z)});
  const std::vector<float> rest  = transformPoints(body.toSpace, ends);
  Surface                  floor = Surface::collider(rest, rest, {}, kBodyLayer, 0.05f, 0.4f);
  const size_t             m     = fitted.size() * 2;
  floor.skinIndex.assign(m * 4, 0);
  floor.skinWeight.assign(m * 4, 0.f);
  for(size_t e = 0; e < m; e++)
  {
    floor.skinIndex[e * 4]  = fitted[e / 2].bone;
    floor.skinWeight[e * 4] = 1.f;
  }
  floor.bones     = body.bones;
  floor.restLocal = ends;
  floor.toSpace   = body.toSpace;
  floor.gravity   = body.gravity;
  skin(floor, floor.bones, floor.posed.data());
  for(size_t c = 0; c < fitted.size(); c++)
    floor.capsules.push_back({uint32_t(c * 2), uint32_t(c * 2 + 1), fitted[c].radius});
  return floor;
}

size_t insideFloor(const Surface& floor, const std::vector<float>& at, const std::vector<float>& points)
{
  size_t count = 0;
  for(size_t p = 0; p + 2 < points.size(); p += 3)
  {
    const Vec3 q{points[p], points[p + 1], points[p + 2]};
    for(const Surface::Capsule& c : floor.capsules)
    {
      if(size_t(c.b) * 3 + 2 >= at.size() || size_t(c.a) * 3 + 2 >= at.size())
        continue;
      const Vec3 a{at[c.a * 3], at[c.a * 3 + 1], at[c.a * 3 + 2]}, b{at[c.b * 3], at[c.b * 3 + 1], at[c.b * 3 + 2]};
      if(segmentDistance(a, b, q) < c.radius)
      {
        count++;
        break;
      }
    }
  }
  return count;
}

}  // namespace fv::physics::cloth
