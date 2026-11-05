#include <figure/physics/flesh/touch.hpp>

#include <algorithm>
#include <cmath>

#include <figure/jobs/pool.hpp>
#include <figure/physics/collide/surface_index.hpp>
#include <figure/physics/contact/contact.hpp>

namespace fv::physics {

namespace {

// How near (cm) a particle or its skin comes to what it meets before it gets a plane: the flesh moves
// a few millimetres a frame at most.
constexpr double kNear = 0.5;
// How deep into a prop is still inside, as the still pass has it (contact.cpp).
constexpr double kDepth = 3, kClosedDepth = 8;
// particles a job looks at
constexpr size_t kChunk = 128;

// The point of segment a-b nearest p.
Vec3 nearestOn(const Vec3& a, const Vec3& b, const Vec3& p)
{
  const Vec3   ab = b - a;
  const double l2 = dot(ab, ab);
  const double t  = l2 > 0 ? std::clamp(dot(p - a, ab) / l2, 0.0, 1.0) : 0.0;
  return a + ab * t;
}

// The scale of a matrix that turns and scales alike along every axis; 0 when it doesn't.
double evenScale(const Mat4& m)
{
  const Vec3   c[3] = {{m.e[0], m.e[1], m.e[2]}, {m.e[4], m.e[5], m.e[6]}, {m.e[8], m.e[9], m.e[10]}};
  const double s    = length(c[0]);
  if(s < 1e-12)
    return 0;
  for(int a = 0; a < 3; a++)
  {
    if(std::fabs(length(c[a]) - s) > 1e-4 * s)
      return 0;
    if(std::fabs(dot(c[a], c[(a + 1) % 3])) > 1e-4 * s * s)
      return 0;
  }
  return s;
}

std::array<double, 8> keyOf(const FleshColliders::Limb& l)
{
  return {double(l.part), l.rest.a.x, l.rest.a.y, l.rest.a.z, l.rest.b.x, l.rest.b.y, l.rest.b.z, l.rest.radius};
}

}  // namespace

FleshTouch::FleshTouch(std::vector<uint8_t> part, std::vector<float> fromEdge, std::vector<uint32_t> joined, std::vector<float> rest)
    : m_part(std::move(part))
    , m_fromEdge(std::move(fromEdge))
    , m_rest(std::move(rest))
    , m_joined(std::move(joined))
{
}

void FleshTouch::allowFor(const std::vector<FleshColliders::Limb>& own)
{
  m_for.clear();
  for(const FleshColliders::Limb& l : own)
    m_for.push_back(keyOf(l));
  const size_t n = m_part.size(), count = own.size();
  m_allow.assign(n * count, -1.0);
  for(size_t i = 0; i < n; i++)
    for(size_t k = 0; k < count; k++)
    {
      const int     p    = own[k].part;
      const uint8_t mine = m_part[i];
      if(p == mine)
        continue;
      const bool joined = mine < m_joined.size() && p >= 0 && p < 32 && ((m_joined[mine] >> p) & 1u);
      if(joined && (i >= m_fromEdge.size() || m_fromEdge[i] < kPartsApart))
        continue;
      const Vec3 at{m_rest[i * 3], m_rest[i * 3 + 1], m_rest[i * 3 + 2]};
      const Capsule& c = own[k].rest;
      m_allow[i * count + k] = std::max(0.0, c.radius - length(at - nearestOn(c.a, c.b, at)));
    }
}

std::vector<FleshContact> FleshTouch::find(std::span<const float> targets, std::span<const double> offsets, const FleshColliders& colliders, const Mat4& toFlesh)
{
  const size_t              n = m_part.size();
  std::vector<FleshContact> out;
  if(targets.size() < n * 3 || offsets.size() < n * 3 || m_rest.size() < n * 3)
    return out;

  // its own limbs: what each particle meets of them and how deep, worked out again when they change
  // (a shape dial fits the ragdoll again)
  std::vector<FleshColliders::Limb> own;
  std::vector<size_t>               ownOf(colliders.limbs.size(), SIZE_MAX);
  std::vector<std::array<double, 8>> key;
  for(size_t k = 0; k < colliders.limbs.size(); k++)
    if(colliders.limbs[k].part >= 0)
    {
      ownOf[k] = own.size();
      own.push_back(colliders.limbs[k]);
      key.push_back(keyOf(colliders.limbs[k]));
    }
  if(key != m_for)
    allowFor(own);
  const size_t count = own.size();

  // the limbs in the targets' space, and the props, queried in their own
  struct Limb
  {
    Vec3   a, b;
    double radius = 0, share = 1;
    size_t own    = SIZE_MAX;
    Box    around;
  };
  std::vector<Limb> limbs;
  if(const double scale = evenScale(toFlesh); scale > 0)
    for(size_t k = 0; k < colliders.limbs.size(); k++)
    {
      const FleshColliders::Limb& l = colliders.limbs[k];
      Limb                        p{toFlesh.transformPoint(l.now.a), toFlesh.transformPoint(l.now.b), l.now.radius * scale, std::clamp(l.share, 0.0, 1.0), ownOf[k], {}};
      const double                r = p.radius + kNear;
      for(const Vec3& e : {p.a, p.b})
      {
        p.around.add(e.x - r, e.y - r, e.z - r);
        p.around.add(e.x + r, e.y + r, e.z + r);
      }
      limbs.push_back(p);
    }
  struct Set
  {
    const TriangleSet* set = nullptr;
    Mat4               to, from;
    double             scale = 1;
    Box                around;
  };
  std::vector<Set> sets;
  for(const ColliderRef& c : colliders.sets)
  {
    // a prop's triangles not read yet are the still pass's to read, on the pool: not here, mid-frame
    if(!c.set || c.share <= 0 || !c.set->made())
      continue;
    const Mat4   to = toFlesh * c.toBody;
    const double s  = evenScale(to);
    if(s <= 0)
      continue;
    sets.push_back({&c.set->get(), to, inverse(to), s, transformed(c.set->bounds(), to).grown(kClosedDepth)});
  }
  if(limbs.empty() && sets.empty())
    return out;

  std::vector<std::vector<FleshContact>> found((n + kChunk - 1) / kChunk);
  jobs::parallelFor(n, kChunk, [&](size_t begin, size_t end) {
    std::vector<FleshContact>& mine   = found[begin / kChunk];
    Stamps&                    stamps = threadStamps();
    for(size_t i = begin; i < end; i++)
    {
      const Vec3 s{targets[i * 3], targets[i * 3 + 1], targets[i * 3 + 2]};
      const Vec3 x = s + Vec3{offsets[i * 3], offsets[i * 3 + 1], offsets[i * 3 + 2]};
      const auto close = [&](const Box& b) { return b.contains(s.x, s.y, s.z) || b.contains(x.x, x.y, x.z); };
      for(const Limb& l : limbs)
      {
        const double allow = l.own == SIZE_MAX ? 0 : m_allow[i * count + l.own];
        if(allow < 0 || !close(l.around))
          continue;
        // by the deeper of where its skin is and where it is
        const Vec3   cs = nearestOn(l.a, l.b, s), cx = nearestOn(l.a, l.b, x);
        const double ds = l.radius - length(s - cs), dx = l.radius - length(x - cx);
        const bool   byX = dx > ds;
        const Vec3   p = byX ? x : s, c = byX ? cx : cs;
        const double dist = length(p - c);
        if(std::max(ds, dx) < allow - kNear || dist < 1e-6)
          continue;
        const Vec3   nn = (p - c) * (1 / dist);
        const Vec3   on = c + nn * l.radius;
        FleshContact k;
        k.particle = uint32_t(i);
        k.point    = {on.x, on.y, on.z};
        k.normal   = {nn.x, nn.y, nn.z};
        // another person gives way by their share of what its skin is in
        k.allow = l.own == SIZE_MAX ? (1 - l.share) * std::max(0.0, ds) : allow;
        mine.push_back(k);
      }
      for(const Set& t : sets)
      {
        if(!close(t.around))
          continue;
        // inside: out the way the deeper of the two leaves it; outside but near: the closest point's plane
        double       deepest = -1;
        FleshContact k;
        k.particle = uint32_t(i);
        for(const Vec3& p : {s, x})
        {
          const Vec3   q  = t.from.transformPoint(p);
          const Inside in = insideOf(*t.set, {q.x, q.y, q.z}, kDepth / t.scale, kClosedDepth / t.scale, stamps);
          if(!in.inside || in.depth * t.scale <= deepest)
            continue;
          deepest        = in.depth * t.scale;
          const Vec3 e   = t.to.transformPoint({in.exit[0], in.exit[1], in.exit[2]});
          const Vec3 out = normalize(t.to.transformPoint({in.exit[0] + in.normal[0], in.exit[1] + in.normal[1], in.exit[2] + in.normal[2]}) - e);
          k.point        = {e.x, e.y, e.z};
          k.normal       = {out.x, out.y, out.z};
        }
        if(deepest < 0)
        {
          const Vec3          q   = t.from.transformPoint(x);
          const SurfaceIndex& idx = t.set->index();
          const double        r   = kNear / t.scale;
          const auto          hit = idx.search(q.x, q.y, q.z, 0, {}, r * r, stamps.stamp, stamps.next(idx.triangles()));
          if(hit.tri < 0)
            continue;
          const auto   P = t.set->positions();
          const auto   T = t.set->tris();
          const size_t a = T[size_t(hit.tri) * 3] * 3ull, b = T[size_t(hit.tri) * 3 + 1] * 3ull, c = T[size_t(hit.tri) * 3 + 2] * 3ull;
          const Vec3   on = t.to.transformPoint({P[a] * hit.u + P[b] * hit.v + P[c] * hit.w, P[a + 1] * hit.u + P[b + 1] * hit.v + P[c + 1] * hit.w,
                                                 P[a + 2] * hit.u + P[b + 2] * hit.v + P[c + 2] * hit.w});
          const double dist = length(x - on);
          if(dist < 1e-6)
            continue;
          const Vec3 nn = (x - on) * (1 / dist);
          k.point       = {on.x, on.y, on.z};
          k.normal      = {nn.x, nn.y, nn.z};
        }
        mine.push_back(k);
      }
    }
  });
  for(std::vector<FleshContact>& f : found)
    out.insert(out.end(), f.begin(), f.end());
  return out;
}

}  // namespace fv::physics
