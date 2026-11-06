#include <figure/physics/cloth/ride.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <figure/jobs/pool.hpp>
#include <figure/physics/collide/surface_index.hpp>

namespace fv::physics::cloth {

namespace {

// A face's point at `bary` and its frame there: e1 along its first side, n its normal (its winding's),
// e2 across. A face with no area has none (`ok` false), and an offset on it is taken as it is.
struct Frame
{
  double q[3], e1[3], e2[3], n[3];
  bool   ok = false;
};

Frame frameOf(const float* a, const float* b, const float* c, const float* bary)
{
  Frame f;
  double s[3], t[3];
  for(int k = 0; k < 3; k++)
  {
    f.q[k] = double(bary[0]) * a[k] + double(bary[1]) * b[k] + double(bary[2]) * c[k];
    s[k]   = double(b[k]) - a[k];
    t[k]   = double(c[k]) - a[k];
  }
  const double n[3] = {s[1] * t[2] - s[2] * t[1], s[2] * t[0] - s[0] * t[2], s[0] * t[1] - s[1] * t[0]};
  const double ls = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]), ln = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if(ls < 1e-12 || ln < 1e-12)
    return f;
  for(int k = 0; k < 3; k++)
  {
    f.e1[k] = s[k] / ls;
    f.n[k]  = n[k] / ln;
  }
  f.e2[0] = f.n[1] * f.e1[2] - f.n[2] * f.e1[1];
  f.e2[1] = f.n[2] * f.e1[0] - f.n[0] * f.e1[2];
  f.e2[2] = f.n[0] * f.e1[1] - f.n[1] * f.e1[0];
  f.ok    = true;
  return f;
}

// The closest point to p on the face abc (Ericson, Real-Time Collision Detection 5.1.5): its weight per
// corner into `bary`, and its squared distance from p.
double closestOn(const double p[3], const float* a, const float* b, const float* c, float bary[3])
{
  double ab[3], ac[3], ap[3], bp[3], cp[3];
  for(int k = 0; k < 3; k++)
  {
    ab[k] = double(b[k]) - a[k];
    ac[k] = double(c[k]) - a[k];
    ap[k] = p[k] - a[k];
    bp[k] = p[k] - b[k];
    cp[k] = p[k] - c[k];
  }
  const auto   dot = [](const double* x, const double* y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
  const double d1 = dot(ab, ap), d2 = dot(ac, ap), d3 = dot(ab, bp), d4 = dot(ac, bp), d5 = dot(ab, cp), d6 = dot(ac, cp);
  const double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
  double       v = 0, w = 0;
  if(d1 <= 0 && d2 <= 0)
    v = 0, w = 0;
  else if(d3 >= 0 && d4 <= d3)
    v = 1, w = 0;
  else if(vc <= 0 && d1 >= 0 && d3 <= 0)
    v = d1 / (d1 - d3), w = 0;
  else if(d6 >= 0 && d5 <= d6)
    v = 0, w = 1;
  else if(vb <= 0 && d2 >= 0 && d6 <= 0)
    v = 0, w = d2 / (d2 - d6);
  else if(va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6)), v = 1 - w;
  else if(const double sum = va + vb + vc; std::fabs(sum) > 1e-30)
    v = vb / sum, w = vc / sum;
  bary[0] = float(1 - v - w);
  bary[1] = float(v);
  bary[2] = float(w);
  double d = 0;
  for(int k = 0; k < 3; k++)
  {
    const double q = a[k] + ab[k] * v + ac[k] * w - p[k];
    d += q * q;
  }
  return d;
}

Frame frameOf(const Ride& ride, size_t i, const std::vector<float>& corners)
{
  const uint32_t* c = &ride.at[i * 3];
  return frameOf(&corners[size_t(c[0]) * 3], &corners[size_t(c[1]) * 3], &corners[size_t(c[2]) * 3], &ride.bary[i * 3]);
}

}  // namespace

Ride bindRide(const Surface& body, const std::vector<float>& bodyAt, const std::vector<float>& drape, const std::vector<uint8_t>& held, const Ride* hint,
              float nearCm, float farCm)
{
  Ride         ride;
  const size_t nb = body.vertexCount(), n = drape.size() / 3;
  if(bodyAt.size() != nb * 3 || body.tris.empty() || n == 0 || !(farCm > nearCm) || nearCm < 0)
    return ride;
  const auto  isHeld  = [&](size_t v) { return v < held.size() && held[v]; };
  const auto  weighOf = [&](double d2) { return std::clamp(float((double(farCm) - std::sqrt(d2)) / double(farCm - nearCm)), 0.f, 1.f); };
  const auto  cornersOf = [&](size_t f, uint32_t c[3]) {
    for(size_t k = 0; k < 3; k++)
      c[k] = body.tris[f * 3 + k];
    return c[0] < nb && c[1] < nb && c[2] < nb;
  };
  // per vertex: its face (of the body's), where on it, and how much it rides it
  std::vector<int64_t> face(n, -1);
  std::vector<float>   bary(n * 3), weight(n, 0.f);

  if(hint && hint->face.size() == hint->vertices.size())
  {
    // Between full binds each vertex the last ride had keeps its face, found again on it, and no other is
    // looked for: a step moves a dress little across the body (a fresh search was 12 ms of a 15 ms bind
    // on the Witch Dress, a vertex 3 to 8 cm off trying every face that near)
    jobs::parallelFor(hint->vertices.size(), 2048, [&](size_t begin, size_t end) {
      for(size_t i = begin; i < end; i++)
      {
        const size_t v = hint->vertices[i], f = hint->face[i];
        uint32_t     c[3];
        if(v >= n || isHeld(v) || f * 3 + 2 >= body.tris.size() || !cornersOf(f, c))
          continue;
        const double p[3] = {drape[v * 3], drape[v * 3 + 1], drape[v * 3 + 2]};
        float        bc[3];
        const float  w = weighOf(closestOn(p, &bodyAt[size_t(c[0]) * 3], &bodyAt[size_t(c[1]) * 3], &bodyAt[size_t(c[2]) * 3], bc));
        if(w <= 0)
          continue;
        face[v]   = int64_t(f);
        weight[v] = w;
        for(size_t k = 0; k < 3; k++)
          bary[v * 3 + k] = bc[k];
      }
    });
  }
  else
  {
    // the body's faces within reach of the drape's box: what lies further off carries nothing
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(size_t v = 0; v < n; v++)
      if(!isHeld(v))
        for(int k = 0; k < 3; k++)
        {
          lo[k] = std::min(lo[k], drape[v * 3 + size_t(k)]);
          hi[k] = std::max(hi[k], drape[v * 3 + size_t(k)]);
        }
    std::vector<uint32_t> tris, triOf;
    tris.reserve(body.tris.size());
    for(size_t f = 0; f * 3 + 2 < body.tris.size(); f++)
    {
      uint32_t c[3];
      if(!cornersOf(f, c))
        continue;
      bool apart = false;
      for(size_t k = 0; k < 3 && !apart; k++)
      {
        const float x = bodyAt[size_t(c[0]) * 3 + k], y = bodyAt[size_t(c[1]) * 3 + k], z = bodyAt[size_t(c[2]) * 3 + k];
        apart = std::max({x, y, z}) < lo[k] - farCm || std::min({x, y, z}) > hi[k] + farCm;
      }
      if(apart)
        continue;
      tris.insert(tris.end(), {c[0], c[1], c[2]});
      triOf.push_back(uint32_t(f));
    }
    if(tris.empty())
      return ride;
    // each vertex's nearest face within farCm
    const SurfaceIndex index(bodyAt, tris);
    const double       far2 = double(farCm) * farCm;
    jobs::parallelFor(n, 1024, [&](size_t begin, size_t end) {
      for(size_t v = begin; v < end; v++)
      {
        if(isHeld(v))
          continue;
        Stamps&       stamps = threadStamps();
        const int32_t id     = stamps.next(index.triangles());
        const auto    hit    = index.search(drape[v * 3], drape[v * 3 + 1], drape[v * 3 + 2], 0, {}, far2, stamps.stamp, id);
        if(hit.tri < 0)
          continue;
        const float w = weighOf(hit.d2);
        if(w <= 0)
          continue;
        face[v]         = int64_t(triOf[size_t(hit.tri)]);
        weight[v]       = w;
        bary[v * 3]     = float(hit.u);
        bary[v * 3 + 1] = float(hit.v);
        bary[v * 3 + 2] = float(hit.w);
      }
    });
  }

  // the corners those faces have, once each
  std::vector<uint32_t> slotOf(nb, std::numeric_limits<uint32_t>::max());
  for(size_t v = 0; v < n; v++)
    if(face[v] >= 0)
      for(size_t k = 0; k < 3; k++)
        slotOf[body.tris[size_t(face[v]) * 3 + k]] = 0;
  for(uint32_t b = 0; b < nb; b++)
    if(slotOf[b] == 0)
    {
      slotOf[b] = uint32_t(ride.corners.size());
      ride.corners.push_back(b);
    }
  for(size_t v = 0; v < n; v++)
  {
    if(face[v] < 0)
      continue;
    ride.vertices.push_back(uint32_t(v));
    ride.face.push_back(uint32_t(face[v]));
    for(size_t k = 0; k < 3; k++)
    {
      ride.at.push_back(slotOf[body.tris[size_t(face[v]) * 3 + k]]);
      ride.bary.push_back(bary[v * 3 + k]);
    }
    ride.weight.push_back(weight[v]);
  }
  std::vector<float> corners(ride.corners.size() * 3);
  for(size_t c = 0; c < ride.corners.size(); c++)
    for(size_t k = 0; k < 3; k++)
      corners[c * 3 + k] = bodyAt[size_t(ride.corners[c]) * 3 + k];
  std::vector<float> points(ride.vertices.size() * 3);
  for(size_t i = 0; i < ride.vertices.size(); i++)
    for(size_t k = 0; k < 3; k++)
      points[i * 3 + k] = drape[size_t(ride.vertices[i]) * 3 + k];
  ride.local = rideLocal(ride, corners, points);
  return ride;
}

std::vector<float> rideCorners(const Ride& ride, const SurfaceSkin& skin)
{
  std::vector<float> out(ride.corners.size() * 3);
  jobs::parallelFor(ride.corners.size(), 1024, [&](size_t begin, size_t end) {
    for(size_t c = begin; c < end; c++)
    {
      const Vec3 p   = skin.at(ride.corners[c]);
      out[c * 3]     = float(p.x);
      out[c * 3 + 1] = float(p.y);
      out[c * 3 + 2] = float(p.z);
    }
  });
  return out;
}

std::vector<float> rideAt(const Ride& ride, const std::vector<float>& corners, const std::vector<float>& local)
{
  const size_t       n = ride.vertices.size();
  std::vector<float> out(n * 3);
  if(local.size() != n * 3 || corners.size() != ride.corners.size() * 3)
    return out;
  jobs::parallelFor(n, 2048, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      const Frame  f = frameOf(ride, i, corners);
      const float* l = &local[i * 3];
      for(int k = 0; k < 3; k++)
        out[i * 3 + size_t(k)] = float(f.ok ? f.q[k] + l[0] * f.e1[k] + l[1] * f.e2[k] + l[2] * f.n[k] : f.q[k] + l[k]);
    }
  });
  return out;
}

std::vector<float> rideLocal(const Ride& ride, const std::vector<float>& corners, const std::vector<float>& points)
{
  const size_t       n = ride.vertices.size();
  std::vector<float> out(n * 3);
  if(points.size() != n * 3 || corners.size() != ride.corners.size() * 3)
    return out;
  jobs::parallelFor(n, 2048, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      const Frame  f    = frameOf(ride, i, corners);
      const double d[3] = {points[i * 3] - f.q[0], points[i * 3 + 1] - f.q[1], points[i * 3 + 2] - f.q[2]};
      if(!f.ok)
      {
        for(int k = 0; k < 3; k++)
          out[i * 3 + size_t(k)] = float(d[k]);
        continue;
      }
      out[i * 3]     = float(d[0] * f.e1[0] + d[1] * f.e1[1] + d[2] * f.e1[2]);
      out[i * 3 + 1] = float(d[0] * f.e2[0] + d[1] * f.e2[1] + d[2] * f.e2[2]);
      out[i * 3 + 2] = float(d[0] * f.n[0] + d[1] * f.n[1] + d[2] * f.n[2]);
    }
  });
  return out;
}

}  // namespace fv::physics::cloth
