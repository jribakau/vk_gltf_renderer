#include <figure/dforce/strands.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

#include <figure/jobs/pool.hpp>

namespace fv::dforce {

namespace {

// The files' shape stiffnesses are per their own solver's step, in no stated unit: a pull of this share
// of them per step leaves hair hanging with its authored bends showing. The first span takes its
// stiffness at this many times over, so that hair leaves the scalp the way it grows before gravity has it.
constexpr float kShapeScale = 0.25f;
constexpr float kRootShape  = 12.f;
// Velocity kept per step: a settle wants rest, not the swing of real hair.
constexpr float kKeepVelocity = 0.95f;
// The share of a point's correction by the next one that its velocity takes (the paper's s_damp).
constexpr float kFollowDamping = 0.9f;
// A point moves no further per step, so it can't cross a collider between two looks.
constexpr float kMaxMove = 1.5f;
// How deep behind a collider a point is still brought back from, cm
constexpr float kDeepest = 3.f;
// The distance grid's spacing, cm: finer than the hair's spans, and coarse enough to bake in a moment.
constexpr float kCell = 1.5f;
// In a node of the grid: too far from any face to say
constexpr float kUnset = 1e9f;
// At Volume 1, on top of a strand's offset: how much further off a collider its root stands, cm, and
// as a multiple of the offset (a surface the file sets further out goes further out still)
constexpr float kVolumeLift = 4.f, kVolumeOffsets = 2.f;
// At Volume 1: the pull towards the way a strand grows at its root, per step, and how far along the
// strand that has faded to nothing, as a share of its length
constexpr float kVolumeHold = 0.15f, kVolumeHoldReach = 0.35f;
// Frames a strand must lie still before it stops being solved
constexpr int kStillFrames = 2;
// Under this share of strands still moving, the hair counts as at rest
constexpr float kAwakeShare = 0.005f;

struct P3
{
  float x = 0, y = 0, z = 0;
};
P3    operator+(P3 a, P3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
P3    operator-(P3 a, P3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
P3    operator*(P3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(P3 a, P3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
P3    cross(P3 a, P3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
float len(P3 a) { return std::sqrt(dot(a, a)); }
P3    unit(P3 a, P3 fallback = {0, -1, 0})
{
  const float l = len(a);
  return l > 1e-6f ? a * (1 / l) : fallback;
}
float mix(float a, float b, float t) { return a + (b - a) * t; }
// `v` turned by the shortest rotation that takes unit `from` to unit `to`
P3 turned(P3 v, P3 from, P3 to)
{
  const float c = dot(from, to);
  if(c > 0.9999f)
    return v;
  if(c < -0.9999f)
    return v * -1;
  const P3    axis = cross(from, to);
  const float k    = 1 / (1 + c);
  return v * c + cross(axis, v) + axis * (dot(axis, v) * k);
}
P3 at(const std::vector<float>& a, int i)
{
  return {a[size_t(i) * 3], a[size_t(i) * 3 + 1], a[size_t(i) * 3 + 2]};
}

struct Face
{
  P3 a, b, c, n;
};

P3 closestOn(const Face& f, P3 p)
{
  // Ericson, Real-Time Collision Detection 5.1.5
  const P3    ab = f.b - f.a, ac = f.c - f.a, ap = p - f.a;
  const float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if(d1 <= 0 && d2 <= 0)
    return f.a;
  const P3    bp = p - f.b;
  const float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if(d3 >= 0 && d4 <= d3)
    return f.b;
  const float vc = d1 * d4 - d3 * d2;
  if(vc <= 0 && d1 >= 0 && d3 <= 0)
    return f.a + ab * (d1 / (d1 - d3));
  const P3    cp = p - f.c;
  const float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if(d6 >= 0 && d5 <= d6)
    return f.c;
  const float vb = d5 * d2 - d1 * d6;
  if(vb <= 0 && d2 >= 0 && d6 <= 0)
    return f.a + ac * (d2 / (d2 - d6));
  const float va = d3 * d6 - d5 * d4;
  if(va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    return f.b + (f.c - f.b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  const float denom = 1 / (va + vb + vc);
  return f.a + ab * (vb * denom) + ac * (vc * denom);
}

// The signed distance to the colliders around the box the hair can reach, baked once on a grid: the
// colliders hold still through a settle, and a look then costs eight numbers however many faces are
// near, where hundreds of strand points a strand's worth of steps would look. Details under a cell go:
// hair rests on a shoulder, not an eyelash.
class DistanceField
{
public:
  DistanceField(const std::vector<SettleMesh>& colliders, float offset, P3 from, P3 to)
  {
    const float span = std::max({to.x - from.x, to.y - from.y, to.z - from.z, 0.f});
    // and coarser for an offset that large, which only meets broad shapes, as the band's cost goes
    // with its cube
    m_cell           = std::max({kCell, span / 192.f, offset / 4.f});
    // a point `offset` off a face or kDeepest behind it must find all eight of its corners known
    const float band  = std::max(offset, kDeepest) + m_cell * 1.74f;
    const float band2 = band * band;
    from              = from - P3{band, band, band};
    to                = to + P3{band, band, band};

    const auto lowest  = [](const Face& f) { return P3{std::min({f.a.x, f.b.x, f.c.x}), std::min({f.a.y, f.b.y, f.c.y}), std::min({f.a.z, f.b.z, f.c.z})}; };
    const auto highest = [](const Face& f) { return P3{std::max({f.a.x, f.b.x, f.c.x}), std::max({f.a.y, f.b.y, f.c.y}), std::max({f.a.z, f.b.z, f.c.z})}; };
    std::vector<Face> faces;
    P3                lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for(const SettleMesh& m : colliders)
      for(size_t t = 0; t + 2 < m.tris.size(); t += 3)
      {
        if(!m.collide.empty() && (!m.collide[m.tris[t]] || !m.collide[m.tris[t + 1]] || !m.collide[m.tris[t + 2]]))
          continue;
        const P3 a = at(m.posed, int(m.tris[t])), b = at(m.posed, int(m.tris[t + 1])), c = at(m.posed, int(m.tris[t + 2]));
        const P3 n = cross(b - a, c - a);
        if(len(n) < 1e-8f)
          continue;
        const Face f{a, b, c, unit(n)};
        const P3   l = lowest(f), h = highest(f);
        if(h.x + band < from.x || h.y + band < from.y || h.z + band < from.z || l.x - band > to.x || l.y - band > to.y || l.z - band > to.z)
          continue;
        faces.push_back(f);
        lo = {std::min(lo.x, l.x), std::min(lo.y, l.y), std::min(lo.z, l.z)};
        hi = {std::max(hi.x, h.x), std::max(hi.y, h.y), std::max(hi.z, h.z)};
      }
    if(faces.empty())
      return;
    m_lo = {std::max(from.x, lo.x - band), std::max(from.y, lo.y - band), std::max(from.z, lo.z - band)};
    hi   = {std::min(to.x, hi.x + band), std::min(to.y, hi.y + band), std::min(to.z, hi.z + band)};
    m_nx = int((hi.x - m_lo.x) / m_cell) + 2;
    m_ny = int((hi.y - m_lo.y) / m_cell) + 2;
    m_nz = int((hi.z - m_lo.z) / m_cell) + 2;
    m_d.assign(size_t(m_nx) * m_ny * m_nz, kUnset);

    // A layer of nodes at a time, each with the faces in reach of it, so threads share no node.
    const auto firstNode = [&](float v, float lowEnd, int n) { return std::clamp(int(std::ceil((v - band - lowEnd) / m_cell)), 0, n); };
    const auto lastNode  = [&](float v, float lowEnd, int n) { return std::clamp(int(std::floor((v + band - lowEnd) / m_cell)), -1, n - 1); };
    std::vector<std::vector<uint32_t>> layers(static_cast<size_t>(m_nz));
    for(uint32_t i = 0; i < faces.size(); i++)
      for(int z = firstNode(lowest(faces[i]).z, m_lo.z, m_nz); z <= lastNode(highest(faces[i]).z, m_lo.z, m_nz); z++)
        layers[size_t(z)].push_back(i);
    const size_t stride = size_t(m_nx) * m_ny;
    jobs::parallelFor(size_t(m_nz), 1, [&](size_t begin, size_t end) {
      // Which side of a shared edge or corner a node is on goes by the face turned most squarely to it:
      // the others can have it behind them from outside.
      std::vector<float> square;
      for(size_t z = begin; z < end; z++)
      {
        square.assign(stride, 0.f);
        float* layer = m_d.data() + stride * z;
        for(const uint32_t i : layers[z])
        {
          const Face& f = faces[i];
          const P3    l = lowest(f), h = highest(f);
          const int   y1 = lastNode(h.y, m_lo.y, m_ny), x0 = firstNode(l.x, m_lo.x, m_nx), x1 = lastNode(h.x, m_lo.x, m_nx);
          for(int y = firstNode(l.y, m_lo.y, m_ny); y <= y1; y++)
            for(int x = x0; x <= x1; x++)
            {
              const P3    node{m_lo.x + float(x) * m_cell, m_lo.y + float(y) * m_cell, m_lo.z + float(z) * m_cell};
              const P3    d  = node - closestOn(f, node);
              const float dd = dot(d, d);
              if(dd >= band2)
                continue;
              const size_t k    = size_t(x) + size_t(y) * size_t(m_nx);
              const float  seen = layer[k] * layer[k];
              const float  dist = std::sqrt(dd);
              const float  side = dot(d, f.n);
              const float  face = dist > 1e-5f ? std::fabs(side) / dist : 1.f;
              if(dd < seen * 0.999f || (dd <= seen * 1.001f && face > square[k]))
              {
                layer[k]  = side < 0 ? -dist : dist;
                square[k] = face;
              }
            }
        }
      }
    });
  }

  // Keeps `p` `gap` off the colliders. True on a contact, with `normal` the way it was pushed.
  bool pushOut(P3& p, float gap, P3& normal) const
  {
    if(m_d.empty())
      return false;
    const float fx = (p.x - m_lo.x) / m_cell, fy = (p.y - m_lo.y) / m_cell, fz = (p.z - m_lo.z) / m_cell;
    if(fx < 0 || fy < 0 || fz < 0 || fx >= float(m_nx - 1) || fy >= float(m_ny - 1) || fz >= float(m_nz - 1))
      return false;
    const int    ix = int(fx), iy = int(fy), iz = int(fz);
    const size_t sy = size_t(m_nx), sz = size_t(m_nx) * m_ny;
    const float* d  = m_d.data() + size_t(ix) + sy * iy + sz * iz;
    const float  c000 = d[0], c100 = d[1], c010 = d[sy], c110 = d[sy + 1];
    const float  c001 = d[sz], c101 = d[sz + 1], c011 = d[sz + sy], c111 = d[sz + sy + 1];
    if(std::max({c000, c100, c010, c110, c001, c101, c011, c111}) >= kUnset)
      return false;
    const float tx = fx - float(ix), ty = fy - float(iy), tz = fz - float(iz);
    const float x00 = mix(c000, c100, tx), x10 = mix(c010, c110, tx), x01 = mix(c001, c101, tx), x11 = mix(c011, c111, tx);
    const float y0 = mix(x00, x10, ty), y1 = mix(x01, x11, ty);
    const float dist = mix(y0, y1, tz);
    if(dist >= gap || dist <= -kDeepest)
      return false;
    const P3 slope{mix(mix(c100 - c000, c110 - c010, ty), mix(c101 - c001, c111 - c011, ty), tz), mix(x10 - x00, x11 - x01, tz), y1 - y0};
    if(len(slope) < 1e-5f)
      return false;
    normal = unit(slope);
    p      = p + normal * (gap - dist);
    return true;
  }

private:
  std::vector<float> m_d;
  P3                 m_lo;
  float              m_cell = kCell;
  int                m_nx = 0, m_ny = 0, m_nz = 0;
};

// One strand's simulated points: `at` indexes into the strand's own points
struct Chain
{
  std::vector<int> at;
  std::vector<P3>  x, prev, shape, shapeUnit;  // shape: each span's authored vector, x[i + 1] - x[i], as posed
  std::vector<float> length;
  // frames in a row it has barely moved
  int still = 0;
};

Chain chainOf(const StrandSet& s, size_t strand, float spacing)
{
  Chain      c;
  const int  begin = s.start[strand], n = s.start[strand + 1] - begin;
  const auto point = [&](int k) { return at(s.posed, s.verts[size_t(begin + k)]); };
  c.at.push_back(0);
  float run = 0;
  for(int k = 1; k < n; k++)
  {
    run += len(point(k) - point(k - 1));
    if(run >= spacing || k == n - 1)
    {
      c.at.push_back(k);
      run = 0;
    }
  }
  c.x.reserve(c.at.size());
  for(int k : c.at)
    c.x.push_back(point(k));
  c.prev = c.x;
  for(size_t i = 0; i + 1 < c.x.size(); i++)
  {
    c.shape.push_back(c.x[i + 1] - c.x[i]);
    c.shapeUnit.push_back(unit(c.shape.back()));
    // the chord, as the waves laid back on it hold the rest of the strand's length
    c.length.push_back(len(c.shape.back()));
  }
  return c;
}

}  // namespace

std::vector<float> settleStrands(const StrandSet& s, const std::vector<SettleMesh>& colliders, const StrandOptions& options)
{
  std::vector<float> out = s.posed;
  const size_t       strands = s.start.empty() ? 0 : s.start.size() - 1;
  if(!strands)
    return out;

  std::vector<Chain> chains(strands);
  jobs::parallelFor(strands, 64, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
      if(s.start[i + 1] - s.start[i] >= 2 && s.strength[i] > 0)
        chains[i] = chainOf(s, i, options.spacing);
  });

  // The field needs only what the hair can reach: its roots, out to its longest strand.
  float offset = 0, longest = 0;
  P3    lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
  for(size_t i = 0; i < strands; i++)
  {
    const Chain& c = chains[i];
    if(c.x.size() < 2 || !s.collide[i])
      continue;
    float length = 0;
    for(const float l : c.length)
      length += l;
    longest = std::max(longest, length);
    offset  = std::max(offset, s.offset[i] + s.volume * (kVolumeLift + kVolumeOffsets * s.offset[i]));
    lo      = {std::min(lo.x, c.x[0].x), std::min(lo.y, c.x[0].y), std::min(lo.z, c.x[0].z)};
    hi      = {std::max(hi.x, c.x[0].x), std::max(hi.y, c.x[0].y), std::max(hi.z, c.x[0].z)};
  }
  const DistanceField field(longest > 0 ? colliders : std::vector<SettleMesh>{}, offset, lo - P3{longest, longest, longest}, hi + P3{longest, longest, longest});

  const float h = 1.f / 30.f / float(std::max(1, options.substeps));
  const P3    g{s.gravity[0] * h * h, s.gravity[1] * h * h, s.gravity[2] * h * h};

  const auto step = [&](size_t strand) {
    Chain&       c = chains[strand];
    const size_t n = c.x.size();
    const float  shapeRoot = s.shapeRoot[strand] * kShapeScale, shapeTip = s.shapeTip[strand] * kShapeScale;
    const float  lift      = s.volume * (kVolumeLift + kVolumeOffsets * s.offset[strand]);
    const bool   collide   = s.collide[strand] != 0;
    const float  friction  = std::clamp(s.friction[strand], 0.f, 1.f);
    // where each point goes: its motion, damped, and gravity
    thread_local std::vector<P3> pred, correction;
    pred.resize(n);
    correction.resize(n);
    pred[0] = c.x[0];
    for(size_t i = 1; i < n; i++)
    {
      P3 move = (c.x[i] - c.prev[i]) * kKeepVelocity + g;
      if(const float l = len(move); l > kMaxMove)
        move = move * (kMaxMove / l);
      pred[i] = c.x[i] + move;
    }
    // Root to tip, each span back to its length, leaning towards its authored bend from the span above,
    // and away from the colliders, the correction kept for the velocity. Every point builds on the one
    // before it where that one came to rest, so one pass holds both.
    P3 parentNow = c.shapeUnit[0], parentWas = parentNow;
    for(size_t i = 1; i < n; i++)
    {
      const P3    was       = pred[i];
      const P3    shapeUnit = c.shapeUnit[i - 1];
      const P3    authored  = i == 1 ? shapeUnit : unit(turned(shapeUnit, parentWas, parentNow), shapeUnit);
      const float along     = float(i - 1) / float(std::max<size_t>(1, n - 2));
      const float hold      = s.volume * kVolumeHold * std::max(0.f, 1 - along / kVolumeHoldReach);
      const float pull      = std::min(1.f, std::max(hold, (i == 1 ? kRootShape : 1.f) * mix(shapeRoot, shapeTip, along)));
      const P3    now       = unit(pred[i] - pred[i - 1], authored);
      P3          dir       = unit(now * (1 - pull) + authored * pull, now);
      pred[i]               = pred[i - 1] + dir * c.length[i - 1];
      P3 normal;
      if(collide && field.pushOut(pred[i], s.offset[strand] + lift * (1 - along * along), normal))
      {
        // loses motion along what it rests on, and stays its length from the point above
        const P3 move    = pred[i] - c.x[i];
        const P3 tangent = move - normal * dot(move, normal);
        pred[i]          = pred[i] - tangent * friction;
        dir              = unit(pred[i] - pred[i - 1], dir);
        pred[i]          = pred[i - 1] + dir * c.length[i - 1];
      }
      correction[i] = pred[i] - was;
      parentWas     = shapeUnit;
      parentNow     = dir;
    }
    // the velocity next time: a point's own move, without most of what holding the next point to its
    // length will have asked of it, which would otherwise build up along the strand
    for(size_t i = 1; i < n; i++)
    {
      c.prev[i] = i + 1 < n ? c.x[i] + correction[i + 1] * kFollowDamping : c.x[i];
      c.x[i]    = pred[i];
    }
  };

  for(int frame = 0; frame < options.maxFrames; frame++)
  {
    if(options.cancelled && options.cancelled())
      break;
    std::atomic<size_t> awake{0};
    jobs::parallelFor(strands, 32, [&](size_t begin, size_t end) {
      size_t          moving = 0;
      std::vector<P3> was;
      for(size_t i = begin; i < end; i++)
      {
        Chain& c = chains[i];
        if(c.x.size() < 2 || c.still >= kStillFrames)
          continue;
        was = c.x;
        for(int k = 0; k < options.substeps; k++)
          step(i);
        float most = 0;
        for(size_t p = 0; p < was.size(); p++)
          most = std::max(most, len(c.x[p] - was[p]));
        // a frame or two in, once it has had the chance to start moving
        c.still = frame > 3 && most < options.restMove ? c.still + 1 : 0;
        if(c.still < kStillFrames)
          moving++;
      }
      awake += moving;
    });
    if(frame > 3 && float(awake.load()) <= float(strands) * kAwakeShare)
      break;
  }

  // Every point of each strand: along its span as moved, its offset from the span turned as the span
  // turned, then as far from the posed shape as the strand's strength says.
  jobs::parallelFor(strands, 64, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      const Chain& c = chains[i];
      if(c.x.size() < 2)
        continue;
      const int   first    = s.start[i];
      const float strength = std::min(1.f, s.strength[i]);
      const auto  posedAt  = [&](int k) { return at(s.posed, s.verts[size_t(first + k)]); };
      for(size_t span = 0; span + 1 < c.x.size(); span++)
      {
        const int a = c.at[span], b = c.at[span + 1];
        const P3  from = c.shapeUnit[span], to = unit(c.x[span + 1] - c.x[span], from);
        const P3  pa = posedAt(a);
        for(int k = a; k <= b; k++)
        {
          const P3    was    = posedAt(k);
          const float t      = dot(was - pa, from) / std::max(1e-5f, len(c.shape[span]));
          const P3    onSpan = pa + c.shape[span] * t;
          const P3    now    = c.x[span] + (c.x[span + 1] - c.x[span]) * t + turned(was - onSpan, from, to);
          const P3    final  = was + (now - was) * strength;
          const int   v      = s.verts[size_t(first + k)];
          out[size_t(v) * 3]     = final.x;
          out[size_t(v) * 3 + 1] = final.y;
          out[size_t(v) * 3 + 2] = final.z;
        }
      }
    }
  });
  return out;
}

}  // namespace fv::dforce
