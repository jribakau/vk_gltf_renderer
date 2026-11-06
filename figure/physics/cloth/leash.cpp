#include <figure/physics/cloth/leash.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <unordered_map>

#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/geometry.hpp>

namespace fv::physics::cloth {

Leashes leash(Ride ride, const Surface* cloth, const Surface& body)
{
  Leashes      l;
  const size_t n = ride.vertices.size(), nb = body.vertexCount();
  if(n == 0 || ride.local.size() != n * 3 || ride.face.size() != n)
    return l;
  // the gap a collision keeps, as the model has it: the larger of the face's (its corners' mean) and the vertex's
  const auto faceGap = [&](uint32_t f) {
    if(body.offset.size() != nb)
      return 0.f;
    float sum = 0;
    for(int k = 0; k < 3; k++)
      sum += body.offset[body.tris[size_t(f) * 3 + size_t(k)]];
    return sum / 3;
  };
  const auto vertexGap = [&](uint32_t v) { return cloth && v < cloth->offset.size() && cloth->offset[v] > 0 ? cloth->offset[v] : 0.05f; };
  l.reach.resize(n);
  l.gap.resize(n);
  for(size_t i = 0; i < n; i++)
  {
    float*      p   = &ride.local[i * 3];
    const float h   = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    const float gap = std::max(faceGap(ride.face[i]), vertexGap(ride.vertices[i]));
    if(p[2] >= 0)
    {
      // outside: held out by the gap, or by as little as it lies at, since where it lies now it may always be
      l.reach[i] = kLeashReach + kLeashSlope * std::max(0.f, h - kLeashClose);
      l.gap[i]   = std::min(p[2], gap);
    }
    else if(h <= kLeashInside && -p[2] >= 0.5f * h)
    {
      // under its face, in the body: held back out over it by the gap, from as deep as it is
      p[2]       = gap;
      l.reach[i] = kLeashReach + h + gap;
      l.gap[i]   = gap;
    }
    else
    {
      // beside a crease behind the plane of its nearest face, or deeper in than a limb can be told from
      // its far side: the collisions' alone
      l.reach[i] = 0;
      l.gap[i]   = -1.f;
    }
  }
  l.ride = std::move(ride);
  return l;
}

std::vector<int> facesAcross(const std::vector<uint32_t>& tris)
{
  // two faces share a side when they share its two corners, as cloth::Model::build pairs them
  const size_t                      nt = tris.size() / 3;
  std::vector<int>                  out(nt * 3, -1);
  std::unordered_map<uint64_t, int> open;
  open.reserve(nt * 2);
  for(size_t t = 0; t < nt; t++)
    for(size_t k = 0; k < 3; k++)
    {
      const uint32_t a = tris[t * 3 + k], b = tris[t * 3 + (k + 1) % 3];
      const uint64_t key = uint64_t(std::min(a, b)) << 32 | std::max(a, b);
      const auto [it, fresh] = open.try_emplace(key, int(t * 3 + k));
      if(fresh)
        continue;
      const int other = it->second;
      if(out[size_t(other)] < 0)
      {
        out[size_t(other)] = int(t);
        out[t * 3 + k]     = other / 3;
      }
    }
  return out;
}

bool holdOut(const std::vector<float>& bodyAt, const std::vector<uint32_t>& tris, const std::vector<int>& across, int& face, float gap, float* p)
{
  const int nt = int(tris.size() / 3);
  if(face < 0 || face >= nt || across.size() != tris.size())
    return false;
  // the face it is over: its own, or the nearest it is over two sides around it (as a resting vertex's
  // contact is followed, cloth::Island)
  const auto corner = [&](int t, int k) { return int(tris[size_t(t) * 3 + size_t(k)]); };
  const auto test   = [&](int t, Closest& c, V3& n) {
    if(!triNormal(bodyAt, corner(t, 0), corner(t, 1), corner(t, 2), n))
      return false;
    c = closest(p[0], p[1], p[2], bodyAt, corner(t, 0), corner(t, 1), corner(t, 2), n);
    return c.over;
  };
  Closest now{};
  V3      n{};
  int     at = -1;
  if(test(face, now, n))
    at = face;
  else
  {
    float best = INFINITY;
    for(int k = 0; k < 3; k++)
    {
      const int a = across[size_t(face) * 3 + size_t(k)];
      if(a < 0)
        continue;
      for(int l = -1; l < 3; l++)
      {
        const int t = l < 0 ? a : across[size_t(a) * 3 + size_t(l)];
        if(t < 0 || t == face)
          continue;
        Closest c{};
        V3      cn{};
        if(test(t, c, cn) && c.dist < best)
        {
          best = c.dist;
          now  = c;
          n    = cn;
          at   = t;
        }
      }
    }
  }
  if(at < 0)
    return false;
  face = at;
  if(now.signedDistance >= gap || now.signedDistance < -kLeashInside)
    return false;
  const float push = std::min(gap - now.signedDistance, kLeashPush);
  for(int k = 0; k < 3; k++)
    p[k] += n[size_t(k)] * push;
  return true;
}

size_t holdLeashes(const Leashes& l, const std::vector<float>& bodyAt, const std::vector<uint32_t>& tris, const std::vector<int>& across,
                   std::vector<float>& points, std::vector<uint8_t>* moved)
{
  const Ride&  ride = l.ride;
  const size_t n    = ride.vertices.size();
  if(points.size() != n * 3 || l.reach.size() != n || ride.face.size() != n)
    return 0;
  if(moved)
    moved->assign(n, 0);
  std::atomic<size_t> count{0};
  jobs::parallelFor(n, 2048, [&](size_t begin, size_t end) {
    size_t here = 0;
    for(size_t i = begin; i < end; i++)
    {
      if(l.gap[i] < 0)
        continue;
      // swung further from where it was tied than its reach: not this part of the body's to hold
      const uint32_t* c = &ride.at[i * 3];
      const RideFrame f = rideFrame(&bodyAt[size_t(ride.corners[c[0]]) * 3], &bodyAt[size_t(ride.corners[c[1]]) * 3], &bodyAt[size_t(ride.corners[c[2]]) * 3],
                                    &ride.bary[i * 3]);
      const float*    lo = &ride.local[i * 3];
      double          l2 = 0;
      for(int k = 0; k < 3; k++)
      {
        const double d = points[i * 3 + size_t(k)] - (f.q[k] + lo[0] * f.e1[k] + lo[1] * f.e2[k] + lo[2] * f.n[k]);
        l2 += d * d;
      }
      int face = int(ride.face[i]);
      if(!f.ok || l2 > double(l.reach[i]) * l.reach[i] || !holdOut(bodyAt, tris, across, face, l.gap[i], &points[i * 3]))
        continue;
      here++;
      if(moved)
        (*moved)[i] = 1;
    }
    count += here;
  });
  return count;
}

void carryOne(const RideFrame& from, const RideFrame& to, float* p)
{
  if(!from.ok || !to.ok)
    return;
  const double d[3] = {p[0] - from.q[0], p[1] - from.q[1], p[2] - from.q[2]};
  const double u = d[0] * from.e1[0] + d[1] * from.e1[1] + d[2] * from.e1[2];
  const double v = d[0] * from.e2[0] + d[1] * from.e2[1] + d[2] * from.e2[2];
  const double w = d[0] * from.n[0] + d[1] * from.n[1] + d[2] * from.n[2];
  for(int k = 0; k < 3; k++)
    p[k] = float(to.q[k] + u * to.e1[k] + v * to.e2[k] + w * to.n[k]);
}

std::vector<Tie> tiesOf(const Leashes& l, int clothBase, int bodyBase)
{
  const Ride&      ride = l.ride;
  std::vector<Tie> out;
  if(l.reach.size() != ride.vertices.size())
    return out;
  out.reserve(ride.vertices.size());
  for(size_t i = 0; i < ride.vertices.size(); i++)
  {
    Tie t{};
    t.vertex = clothBase + int(ride.vertices[i]);
    for(size_t k = 0; k < 3; k++)
    {
      t.corner[k] = bodyBase + int(ride.corners[ride.at[i * 3 + k]]);
      t.bary[k]   = ride.bary[i * 3 + k];
      t.local[k]  = ride.local[i * 3 + k];
    }
    t.reach = l.reach[i];
    t.gap   = l.gap[i];
    t.carry = ride.weight[i];
    out.push_back(t);
  }
  return out;
}

std::vector<Tie> tieTo(const std::vector<Surface>& surfaces, size_t bodySlot, const std::vector<size_t>& slots, const std::vector<std::vector<float>>& positions,
                       std::vector<Ride>& rides, bool fresh)
{
  std::vector<Tie> out;
  rides.resize(slots.size());
  if(bodySlot >= surfaces.size() || bodySlot >= positions.size() || positions.size() != surfaces.size())
    return out;
  // each surface's first vertex, as cloth::Model numbers them: the surfaces in order
  std::vector<int> base(surfaces.size() + 1, 0);
  for(size_t s = 0; s < surfaces.size(); s++)
    base[s + 1] = base[s] + int(surfaces[s].vertexCount());
  const Surface&            body   = surfaces[bodySlot];
  const std::vector<float>& bodyAt = positions[bodySlot];
  for(size_t k = 0; k < slots.size(); k++)
  {
    const size_t slot = slots[k];
    if(slot >= surfaces.size() || slot == bodySlot || positions[slot].size() != surfaces[slot].vertexCount() * 3)
    {
      rides[k] = {};
      continue;
    }
    const Surface& cloth = surfaces[slot];
    const bool     hint  = !fresh && !rides[k].empty();
    rides[k]             = bindRide(body, bodyAt, positions[slot], heldOf(cloth), hint ? &rides[k] : nullptr);
    const std::vector<Tie> more = tiesOf(leash(rides[k], &cloth, body), base[slot], base[bodySlot]);
    out.insert(out.end(), more.begin(), more.end());
  }
  return out;
}

std::vector<uint8_t> heldOf(const Surface& cloth)
{
  const size_t         n = cloth.vertexCount();
  std::vector<uint8_t> held(n, 0);
  for(size_t v = 0; v < n; v++)
    held[v] = !(v < cloth.strength.size() && cloth.strength[v] > 1e-4f) || (cloth.visible.size() == n && !cloth.visible[v]);
  return held;
}

}  // namespace fv::physics::cloth
