#include <figure/physics/rain/rain_sim.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <figure/math/noise.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/physics/collide/surface_index.hpp>

namespace fv::physics {

namespace {

constexpr double kPi = 3.14159265358979323846;

// The step, s: a run moves at most 4 mm a step, less than it is wide, so it passes no bead by.
constexpr double kStep = 1.0 / 30;
// A bead meets the skin at this angle (the sweat tile's beads are 60 degrees; rain sits a little higher
// on skin that isn't oily).
constexpr double kContactAngle = 70 * kPi / 180;
// A bead runs once its contact radius squared times the sine of the slope passes this squared: on a
// wall, one wider than 3.2 mm. On a wet track the pinning is this share of a dry skin's. A run stops a
// little below where it starts, as a bead's advancing and receding angles differ.
constexpr double kStick = 1.6e-3, kWetStick = 0.6, kStopBelow = 0.85;
// how fast the heaviest run, m/s
constexpr double kRunSpeed = 0.12;
// the film a run leaves along its way, on dry skin and on a wet track, m
constexpr double kDryFilm = 10e-6, kWetFilm = 2e-6;
// on dry skin a run leaves a small bead every so far (m), of a contact radius between these (m)
constexpr double kResidueGap[2] = {4e-3, 12e-3}, kResidueRadius[2] = {0.25e-3, 0.5e-3};
// it wanders across its way by up to this share of it, over a noise this far across (m)
constexpr double kWander = 0.35, kWanderScale = 15e-3;
// Hanging straight down, a bead this heavy falls (m³: a drop about 3.9 mm across); less face down, a
// heavier one, and none that faces down less than this. What falls leaves a little of itself.
constexpr double kDripVolume = 30e-9, kDripFacing = 0.3, kDripLeft = 0.6e-3;
// After the rain: a bead's contact radius dries by this a second (a 1 mm one in about 45 s, as DripSim
// has it), the tracks in this long and the damp in this long (s)
constexpr double kEvaporate = 0.022e-3, kTrackDry = 25, kDampDry = 90;
// A track: its half width as a share of the run's contact radius, its height as it flowed as a share of
// that, the film it thins to (m) and how soon (s); a stretch no longer than this (m), so a run's wander
// shows; and no more stretches than this.
constexpr double kTrackWidth = 0.7, kTrackFresh = 0.3, kTrackFilm = 25e-6, kTrackThin = 1.5, kStretch = 6e-3;
constexpr size_t kMaxTracks = 20000;
// a bead smaller than this is gone (m); at most this many beads, and runs at once
constexpr double kGone       = 0.08e-3;
constexpr size_t kMaxBeads = 4000, kMaxRunners = 64;
// Rain slants by up to this, in gusts from any side, so the front and the back get some.
constexpr double kGust = 35 * kPi / 180;
// Drops a square metre a second at amount 1. Across, in mm: the smallest, the mean over that at amount
// 0 and 1 (an exponential, Marshall and Palmer's shape) and the largest; and the share of one that stays
// on the skin, the rest splashing off.
constexpr double kFlux = 400, kDropMin = 0.4, kDropMean[2] = {0.3, 1.2}, kDropMax = 4, kKept = 0.6;
// once the rain stops, at most this long for the runs to end (s)
constexpr double kSettle = 12;
// the damp all over at amount 0 and 1, reached in this long (s)
constexpr double kDamp[2] = {0.35, 1}, kDampTime = 8;
// the cells that find beads near a point (wider than two of the largest beads' radii), and those that
// say where a track went (about its width), m
constexpr double kBeadCell = 8e-3, kWetCell = 2.5e-3;

// V = kCap a³ for a cap of contact radius a at kContactAngle
const double kCap = [] {
  const double c = std::cos(kContactAngle), s = std::sin(kContactAngle);
  return kPi * (2 - 3 * c + c * c * c) / (3 * s * s * s);
}();

double radiusOf(double volume)
{
  return std::cbrt(std::max(volume, 0.0) / kCap);
}
double volumeOf(double radius)
{
  return kCap * radius * radius * radius;
}
double heightOf(double radius)
{
  return radius * (1 - std::cos(kContactAngle)) / std::sin(kContactAngle);
}

Vec3 cross(const Vec3& a, const Vec3& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double distance(const Vec3& a, const Vec3& b)
{
  return length(a - b);
}

// splitmix64: the same numbers on every compiler, which <random>'s distributions don't promise
struct Random
{
  uint64_t s;
  double   next()
  {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return double(z >> 11) * 0x1.0p-53;
  }
  double in(const double (&range)[2]) { return range[0] + (range[1] - range[0]) * next(); }
};

using CellKey = int64_t;
CellKey cellKey(const Vec3& p, double size)
{
  const auto c = [&](double v) { return CellKey(std::floor(v / size)) & 0x1FFFFF; };
  return c(p.x) | (c(p.y) << 21) | (c(p.z) << 42);
}

// Beads by where they are, in cells; a point's neighbours are in the 27 cells around it, in the order
// they came (so a seeded pass runs the same each time)
class BeadCells
{
public:
  void add(uint32_t id, const Vec3& p) { m_cells[cellKey(p, kBeadCell)].push_back(id); }
  void remove(uint32_t id, const Vec3& p)
  {
    if(auto it = m_cells.find(cellKey(p, kBeadCell)); it != m_cells.end())
      it->second.erase(std::remove(it->second.begin(), it->second.end(), id), it->second.end());
  }
  template <class Visit>
  void around(const Vec3& p, Visit&& visit) const
  {
    for(int dz = -1; dz <= 1; dz++)
      for(int dy = -1; dy <= 1; dy++)
        for(int dx = -1; dx <= 1; dx++)
          if(auto it = m_cells.find(cellKey(p + Vec3{dx * kBeadCell, dy * kBeadCell, dz * kBeadCell}, kBeadCell)); it != m_cells.end())
            for(uint32_t id : it->second)
              visit(id);
  }

private:
  std::unordered_map<CellKey, std::vector<uint32_t>> m_cells;
};

struct Drop
{
  uint32_t tri = 0;
  double   b1 = 0, b2 = 0;
  Vec3     at;  // where it rests, as the cells have it
  double   volume  = 0;
  bool     alive   = true;
  bool     running = false;
  int      track   = -1;  // the stretch of track it draws out
  double   toResidue = 0;  // dry skin to go before it leaves a small bead
  int      leftBy    = -1;  // a small bead's run, which doesn't take it back
};

struct Track
{
  uint32_t tri = 0;
  double   a1 = 0, a2 = 0, b1 = 0, b2 = 0;
  double   run  = 0;  // the contact radius of the run that made it, m
  double   made = 0;  // when it last flowed, s
  double   length = 0;
};

class Sim
{
public:
  Sim(const RainSurface& surface, const RainInput& in, const RainSettings& settings)
      : m_s(surface)
      , m_set(clamped(settings))
      , m_rng{0x5EEDull * (uint64_t(m_set.seed) + 1)}
  {
    const size_t nv = in.posed.size() / 3;
    for(uint32_t v : m_s.tris)
      if(v >= nv)
        return;
    if(m_s.tris.size() != m_s.triangles() * 3 || m_s.uv.size() != m_s.triangles() * 6 || m_s.across.size() != m_s.tris.size())
      return;
    const double toMetres = 1 / std::max(1e-12, in.unitsPerMetre);
    m_p.resize(nv);
    m_pf.resize(nv * 3);
    Vec3 lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
    for(size_t v = 0; v < nv; v++)
    {
      m_p[v] = Vec3{in.posed[v * 3], in.posed[v * 3 + 1], in.posed[v * 3 + 2]} * toMetres;
      for(int k = 0; k < 3; k++)
        m_pf[v * 3 + k] = float((&m_p[v].x)[k]);
      lo = {std::min(lo.x, m_p[v].x), std::min(lo.y, m_p[v].y), std::min(lo.z, m_p[v].z)};
      hi = {std::max(hi.x, m_p[v].x), std::max(hi.y, m_p[v].y), std::max(hi.z, m_p[v].z)};
    }
    if(nv == 0 || m_s.triangles() == 0)
      return;
    m_centre = (lo + hi) * 0.5;
    m_radius = std::max(1e-3, length(hi - lo) * 0.5);
    m_fn.resize(m_s.triangles());
    m_vn.assign(nv, Vec3{});
    for(size_t t = 0; t < m_s.triangles(); t++)
    {
      const uint32_t* c = &m_s.tris[t * 3];
      const Vec3      n = cross(m_p[c[1]] - m_p[c[0]], m_p[c[2]] - m_p[c[0]]);
      for(int k = 0; k < 3; k++)
        m_vn[c[k]] = m_vn[c[k]] + n;
      m_fn[t] = length(n) > 0 ? normalize(n) : Vec3{};
    }
    for(Vec3& n : m_vn)
      n = length(n) > 0 ? normalize(n) : Vec3{};
    m_down = length(in.down) > 0 ? normalize(in.down) : Vec3{0, -1, 0};
    m_index = SurfaceIndex(m_pf, m_s.tris);
    // the wander's noise differs with the seed
    m_wanderAt = Vec3{std::fmod(m_set.seed * 7.31, 97.0), std::fmod(m_set.seed * 3.17, 89.0), std::fmod(m_set.seed * 5.23, 83.0)};
    m_ok = true;
  }

  RainResult run()
  {
    RainResult out;
    if(!m_ok || !m_set.on)
      return out;
    const double rainTime = m_set.time;
    for(m_time = 0;; m_time += kStep)
    {
      if(m_time < rainTime)
        land();
      else if((m_runners.empty() && m_waiting.empty()) || m_time > rainTime + kSettle)
        break;
      stepRuns();
    }

    // after the rain, as long as it has dried
    const double dried = m_set.dried;
    for(const Drop& d : m_drops)
    {
      if(!d.alive)
        continue;
      const double a = radiusOf(d.volume) - kEvaporate * dried;
      if(a < kGone)
        continue;
      out.beads.push_back({d.tri, float(d.b1), float(d.b2), float(a), float(heightOf(a))});
    }
    const double trackLeft = std::max(0.0, 1 - dried / kTrackDry);
    if(trackLeft > 0)
      for(const Track& t : m_tracks)
      {
        if(t.length < 0.2e-3)
          continue;
        const double age = m_time - t.made + dried;
        const double h   = std::max(kTrackFilm, kTrackFresh * t.run * std::exp(-age / kTrackThin)) * trackLeft;
        out.tracks.push_back({t.tri, float(t.a1), float(t.a2), float(t.b1), float(t.b2), float(kTrackWidth * t.run), float(h)});
      }
    const double amount = m_set.amount;
    out.damp            = float((kDamp[0] + (kDamp[1] - kDamp[0]) * amount) * std::min(1.0, rainTime / kDampTime) * std::max(0.0, 1 - dried / kDampDry));
    out.landed          = m_landed;
    out.runs            = m_runs;
    out.dripped         = m_dripped;
    return out;
  }

private:
  Vec3 pointOf(uint32_t tri, double b1, double b2) const
  {
    const uint32_t* c = &m_s.tris[size_t(tri) * 3];
    return m_p[c[0]] * (1 - b1 - b2) + m_p[c[1]] * b1 + m_p[c[2]] * b2;
  }
  Vec3 normalAt(uint32_t tri, double b1, double b2) const
  {
    const uint32_t* c = &m_s.tris[size_t(tri) * 3];
    const Vec3      n = m_vn[c[0]] * (1 - b1 - b2) + m_vn[c[1]] * b1 + m_vn[c[2]] * b2;
    return length(n) > 1e-9 ? normalize(n) : m_fn[tri];
  }
  bool wetAt(const Vec3& p) const { return m_wet.count(cellKey(p, kWetCell)) != 0; }

  // How far past sticking a bead is where it is (above 1 it runs), and the way down the skin there.
  double push(uint32_t id, const Vec3& at, const Vec3& normal, Vec3& down) const
  {
    const Vec3   along = m_down - normal * dot(normal, m_down);
    const double slope = length(along);
    if(slope < 1e-6)
    {
      down = {};
      return 0;
    }
    down               = along * (1 / slope);
    const double a     = radiusOf(m_drops[id].volume);
    const double stick = kStick * (wetAt(at + down * (a + 0.5e-3)) ? kWetStick : 1);
    return a * a * slope / (stick * stick);
  }
  // Hanging under the skin and too heavy to stay
  bool falls(uint32_t id, const Vec3& normal) const
  {
    const double facing = dot(normal, m_down);
    return facing > kDripFacing && m_drops[id].volume * facing > kDripVolume;
  }

  // ---- the beads at rest

  // A bead where a drop landed or a run left one; it merges with what it touches.
  void add(uint32_t tri, double b1, double b2, double volume, int leftBy = -1)
  {
    if(m_resting >= kMaxBeads || volume <= 0)
      return;
    const uint32_t id = uint32_t(m_drops.size());
    Drop           d;
    d.tri    = tri;
    d.b1     = b1;
    d.b2     = b2;
    d.at     = pointOf(tri, b1, b2);
    d.volume = volume;
    d.leftBy = leftBy;
    m_drops.push_back(d);
    m_cells.add(id, d.at);
    m_resting++;
    settle(id);
  }
  void kill(uint32_t id)
  {
    Drop& d = m_drops[id];
    if(!d.alive)
      return;
    if(!d.running)
    {
      m_cells.remove(id, d.at);
      m_resting--;
    }
    d.alive   = false;
    d.running = false;
  }
  // A bead at rest that grew or came: the beads it touches go into the biggest, which keeps its place,
  // and then it may fall or run.
  void settle(uint32_t id)
  {
    for(int round = 0; round < 8; round++)
    {
      const Drop&  d     = m_drops[id];
      const double a     = radiusOf(d.volume);
      int64_t      other = -1;
      m_cells.around(d.at, [&](uint32_t o) {
        const Drop& e = m_drops[o];
        if(other < 0 && o != id && e.alive && !e.running && distance(d.at, e.at) < a + radiusOf(e.volume))
          other = o;
      });
      if(other < 0)
        break;
      uint32_t o = uint32_t(other);
      if(m_drops[o].volume > m_drops[id].volume)
        std::swap(id, o);
      m_drops[id].volume += m_drops[o].volume;
      m_drops[id].leftBy = -1;
      kill(o);
    }
    const Drop& d      = m_drops[id];
    const Vec3  normal = normalAt(d.tri, d.b1, d.b2);
    Vec3        down;
    if(falls(id, normal))
      drip(id);
    else if(push(id, d.at, normal, down) > 1)
    {
      if(m_runners.size() < kMaxRunners)
        start(id);
      else
        m_waiting.push_back(id);
    }
  }
  void start(uint32_t id)
  {
    Drop& d = m_drops[id];
    m_cells.remove(id, d.at);
    m_resting--;
    d.running   = true;
    d.track     = -1;
    d.toResidue = m_rng.in(kResidueGap);
    m_runners.push_back(id);
    m_runs++;
  }
  // it falls off, leaving a little of itself
  void drip(uint32_t id)
  {
    const Drop   d    = m_drops[id];
    const double left = std::min(d.volume * 0.2, volumeOf(kDripLeft));
    kill(id);
    m_dripped++;
    add(d.tri, d.b1, d.b2, left);
  }

  // ---- rain

  void land()
  {
    m_toLand += kFlux * m_set.amount * kPi * m_radius * m_radius * kStep;
    Stamps& stamps = threadStamps();
    for(; m_toLand >= 1; m_toLand -= 1)
    {
      // a gust tilts it towards a side
      const double tilt = kGust * m_rng.next(), turn = 2 * kPi * m_rng.next();
      Vec3         side1, side2;
      across(m_down, side1, side2);
      const Vec3 dir = normalize(m_down + (side1 * std::cos(turn) + side2 * std::sin(turn)) * std::tan(tilt));
      // from a point on the disc across it around the body, behind the body
      across(dir, side1, side2);
      const double r = m_radius * std::sqrt(m_rng.next()), at = 2 * kPi * m_rng.next();
      const Vec3   o = m_centre + (side1 * std::cos(at) + side2 * std::sin(at)) * r - dir * (2 * m_radius);
      // the drop's size, drawn whether it lands or not, so the sizes don't hang on what the rays meet
      const double mean = kDropMean[0] + (kDropMean[1] - kDropMean[0]) * m_set.amount;
      const double D    = std::min(kDropMax, kDropMin - mean * std::log(1 - m_rng.next())) * 1e-3;
      const double origin[3] = {o.x, o.y, o.z}, ray[3] = {dir.x, dir.y, dir.z};
      const SurfaceIndex::RayHit hit = m_index.firstHit(origin, ray, 4 * m_radius, stamps.stamp, stamps.next(m_index.triangles()));
      if(hit.tri < 0)
        continue;
      m_landed++;
      double b1, b2;
      baryOf(uint32_t(hit.tri), o + dir * hit.t, b1, b2);
      add(uint32_t(hit.tri), b1, b2, kPi / 6 * D * D * D * kKept);
    }
  }
  static void across(const Vec3& n, Vec3& a, Vec3& b)
  {
    a = normalize(std::fabs(n.x) < 0.9 ? cross(n, Vec3{1, 0, 0}) : cross(n, Vec3{0, 1, 0}));
    b = cross(n, a);
  }
  void baryOf(uint32_t tri, const Vec3& p, double& b1, double& b2) const
  {
    const uint32_t* c  = &m_s.tris[size_t(tri) * 3];
    const Vec3      e1 = m_p[c[1]] - m_p[c[0]], e2 = m_p[c[2]] - m_p[c[0]], q = p - m_p[c[0]];
    const double    d00 = dot(e1, e1), d01 = dot(e1, e2), d11 = dot(e2, e2), d20 = dot(q, e1), d21 = dot(q, e2);
    const double    det = d00 * d11 - d01 * d01;
    b1                  = det > 0 ? (d11 * d20 - d01 * d21) / det : 1.0 / 3;
    b2                  = det > 0 ? (d00 * d21 - d01 * d20) / det : 1.0 / 3;
    b1                  = std::clamp(b1, 0.0, 1.0);
    b2                  = std::clamp(b2, 0.0, 1.0 - b1);
  }

  // ---- runs

  void stepRuns()
  {
    std::vector<uint32_t> moving;
    moving.swap(m_runners);
    for(uint32_t id : moving)
      if(m_drops[id].alive && m_drops[id].running)
      {
        step(id);
        if(m_drops[id].alive && m_drops[id].running)
          m_runners.push_back(id);
      }
    // two runs that meet go on as one, the bigger
    for(size_t i = 0; i < m_runners.size(); i++)
      for(size_t j = i + 1; j < m_runners.size(); j++)
      {
        Drop &a = m_drops[m_runners[i]], &b = m_drops[m_runners[j]];
        if(!a.alive || !b.alive)
          continue;
        if(distance(pointOf(a.tri, a.b1, a.b2), pointOf(b.tri, b.b1, b.b2)) < radiusOf(a.volume) + radiusOf(b.volume))
        {
          Drop& big = a.volume >= b.volume ? a : b;
          Drop& small = &big == &a ? b : a;
          big.volume += small.volume;
          small.alive   = false;
          small.running = false;
        }
      }
    std::erase_if(m_runners, [&](uint32_t id) { return !m_drops[id].alive; });
    // beads that were too many to run start as others stop
    while(m_runners.size() < kMaxRunners && !m_waiting.empty())
    {
      const uint32_t id = m_waiting.front();
      m_waiting.erase(m_waiting.begin());
      const Drop& d = m_drops[id];
      Vec3        down;
      if(d.alive && !d.running && push(id, d.at, normalAt(d.tri, d.b1, d.b2), down) > 1)
        start(id);
    }
  }

  void step(uint32_t id)
  {
    const Drop& d      = m_drops[id];
    const Vec3  at     = pointOf(d.tri, d.b1, d.b2);
    const Vec3  normal = normalAt(d.tri, d.b1, d.b2);
    Vec3        down;
    const double pushed = push(id, at, normal, down);
    if(falls(id, normal))
    {
      drip(id);
      return;
    }
    if(pushed < kStopBelow)
    {
      stop(id);
      return;
    }
    const double a   = radiusOf(d.volume);
    const bool   wet = wetAt(at + down * (a + 0.5e-3));
    // it wanders across its way by a noise over the body
    const Vec3   q      = at * (1 / kWanderScale) + m_wanderAt;
    const double wander = (noise::noise3(q.x, q.y, q.z) * 2 - 1) * kWander;
    const Vec3   dir    = normalize(down + cross(normal, down) * wander);
    const double dist   = kRunSpeed * std::clamp(pushed - 0.6, 0.15, 1.0) * kStep;
    const uint32_t fromTri = d.tri;
    const double   from1 = d.b1, from2 = d.b2;
    uint32_t       tri = d.tri;
    double         b1 = d.b1, b2 = d.b2;
    const bool     onSkin = walk(tri, b1, b2, dir, dist, [&](uint32_t t, double a1, double a2, double c1, double c2) { draw(id, t, a1, a2, c1, c2, a); });
    {
      Drop& w = m_drops[id];
      w.tri   = tri;
      w.b1    = b1;
      w.b2    = b2;
      w.volume -= dist * 2 * a * (wet ? kWetFilm : kDryFilm);
    }
    // on dry skin it leaves a small bead now and then, where it was
    if(!wet)
    {
      m_drops[id].toResidue -= dist;
      const double r = m_rng.in(kResidueRadius);
      if(m_drops[id].toResidue <= 0 && m_drops[id].volume > 4 * volumeOf(r))
      {
        m_drops[id].volume -= volumeOf(r);
        m_drops[id].toResidue = m_rng.in(kResidueGap);
        add(fromTri, from1, from2, volumeOf(r), int(id));
      }
    }
    // it swallows the beads it meets
    const Vec3   now   = pointOf(tri, b1, b2);
    const double reach = radiusOf(m_drops[id].volume) + 0.5 * dist;
    std::vector<uint32_t> met;
    m_cells.around(now, [&](uint32_t o) {
      const Drop& e = m_drops[o];
      if(e.alive && !e.running && e.leftBy != int(id) && distance(now, e.at) < reach + radiusOf(e.volume))
        met.push_back(o);
    });
    for(uint32_t o : met)
    {
      m_drops[id].volume += m_drops[o].volume;
      kill(o);
    }
    if(m_drops[id].volume < volumeOf(kGone))
      kill(id);
    else if(!onSkin)
      stop(id);
  }
  void stop(uint32_t id)
  {
    Drop& d   = m_drops[id];
    d.running = false;
    d.track   = -1;
    d.at      = pointOf(d.tri, d.b1, d.b2);
    m_cells.add(id, d.at);
    m_resting++;
    // what it stops on goes into it; it doesn't start again on that alone
    for(int round = 0; round < 8; round++)
    {
      const Drop&  s     = m_drops[id];
      const double a     = radiusOf(s.volume);
      int64_t      other = -1;
      m_cells.around(s.at, [&](uint32_t o) {
        const Drop& e = m_drops[o];
        if(other < 0 && o != id && e.alive && !e.running && distance(s.at, e.at) < a + radiusOf(e.volume))
          other = o;
      });
      if(other < 0)
        break;
      m_drops[id].volume += m_drops[size_t(other)].volume;
      kill(uint32_t(other));
    }
  }

  // The stretch of track a run draws out in a triangle: the one it is on goes on while it stays in
  // the triangle and short; else a new one.
  void draw(uint32_t id, uint32_t tri, double a1, double a2, double b1, double b2, double runRadius)
  {
    Drop& d = m_drops[id];
    if(d.track >= 0)
    {
      Track& t = m_tracks[size_t(d.track)];
      if(t.tri == tri && t.length < kStretch && std::fabs(t.b1 - a1) + std::fabs(t.b2 - a2) < 1e-9)
      {
        t.b1     = b1;
        t.b2     = b2;
        t.run    = std::max(t.run, runRadius);
        t.made   = m_time;
        t.length = distance(pointOf(tri, t.a1, t.a2), pointOf(tri, b1, b2));
        markWet(tri, a1, a2, b1, b2);
        return;
      }
    }
    if(m_tracks.size() >= kMaxTracks)
    {
      d.track = -1;
      return;
    }
    d.track = int(m_tracks.size());
    m_tracks.push_back({tri, a1, a2, b1, b2, runRadius, m_time, distance(pointOf(tri, a1, a2), pointOf(tri, b1, b2))});
    markWet(tri, a1, a2, b1, b2);
  }
  void markWet(uint32_t tri, double a1, double a2, double b1, double b2)
  {
    const Vec3 a = pointOf(tri, a1, a2), b = pointOf(tri, b1, b2);
    const int  n = std::max(1, int(std::ceil(distance(a, b) / (0.5 * kWetCell))));
    for(int k = 0; k <= n; k++)
      m_wet.insert(cellKey(a + (b - a) * (double(k) / n), kWetCell));
  }

  // Moves a point on the skin `dist` metres along `dir` (each triangle takes it into its own plane),
  // across edges into the triangles beyond, telling `stretch` each piece inside one triangle. Where the
  // skin on both sides of an edge falls towards it (a crease: the spine, under the breasts), it goes on
  // along the edge. False when it reached an edge with nothing across, where it stays.
  template <class Stretch>
  bool walk(uint32_t& tri, double& b1, double& b2, Vec3 dir, double dist, Stretch&& stretch) const
  {
    int entered = -1;  // the edge it came in by
    for(int hop = 0; hop < 32 && dist > 1e-9; hop++)
    {
      const uint32_t* c  = &m_s.tris[size_t(tri) * 3];
      const Vec3      e1 = m_p[c[1]] - m_p[c[0]], e2 = m_p[c[2]] - m_p[c[0]];
      const Vec3&     n  = m_fn[tri];
      const double    a = dot(e1, e1), b = dot(e1, e2), cc = dot(e2, e2), det = a * cc - b * b;
      if(det <= 1e-30)
        return true;
      Vec3 d = dir - n * dot(n, dir);
      if(length(d) < 1e-12)
        return true;
      d = d * (dist / length(d));
      double x = 0, y = 0, f = 1;
      int    out = -1;
      for(int attempt = 0; attempt < 2; attempt++)
      {
        const double r1 = dot(e1, d), r2 = dot(e2, d);
        x                    = (cc * r1 - b * r2) / det;
        y                    = (a * r2 - b * r1) / det;
        const double bary[3] = {1 - b1 - b2, b1, b2}, delta[3] = {-(x + y), x, y};
        f                    = 1;
        out                  = -1;
        for(int k = 0; k < 3; k++)
          if(delta[k] < 0)
          {
            const double fk = std::max(0.0, bary[k]) / -delta[k];
            if(fk < f)
            {
              f   = fk;
              out = k;
            }
          }
        // straight back out by the edge it came in by: along the edge instead
        if(out < 0 || (out + 1) % 3 != entered || f > 1e-6 || attempt == 1)
          break;
        const Vec3 edge = m_p[c[(entered + 1) % 3]] - m_p[c[entered]];
        d               = edge * (dot(d, edge) / dot(edge, edge));
        dist            = length(d);
        if(dist < 1e-9)
          return true;
      }
      double n1 = b1 + x * f, n2 = b2 + y * f;
      if(out < 0)
      {
        stretch(tri, b1, b2, n1, n2);
        b1 = n1;
        b2 = n2;
        return true;
      }
      // on the edge across from corner `out`
      if(out == 1)
        n1 = 0;
      else if(out == 2)
        n2 = 0;
      else if(n1 + n2 > 0)
      {
        const double s = n1 + n2;
        n1 /= s;
        n2 /= s;
      }
      n1 = std::clamp(n1, 0.0, 1.0);
      n2 = std::clamp(n2, 0.0, 1.0 - n1);
      stretch(tri, b1, b2, n1, n2);
      dist *= 1 - f;
      dir = d;
      // edge k runs from corner k to k + 1: the one across from corner `out`
      const int     edge = (out + 1) % 3;
      const int32_t next = m_s.across[size_t(tri) * 3 + size_t(edge)];
      if(next < 0)
      {
        b1 = n1;
        b2 = n2;
        return false;
      }
      const double    w[3] = {1 - n1 - n2, n1, n2};
      const uint32_t  va = c[edge], vb = c[(edge + 1) % 3];
      const uint32_t* u  = &m_s.tris[size_t(next / 3) * 3];
      double          nb[3] = {0, 0, 0};
      for(int k = 0; k < 3; k++)
        nb[k] = u[k] == va ? w[edge] : u[k] == vb ? w[(edge + 1) % 3] : 0;
      tri     = uint32_t(next / 3);
      b1      = nb[1];
      b2      = nb[2];
      entered = next % 3;
    }
    return true;
  }

  const RainSurface& m_s;
  RainSettings       m_set;
  Random             m_rng;
  bool               m_ok = false;

  std::vector<Vec3>  m_p, m_vn, m_fn;  // posed (m), smooth normals per vertex, normals per triangle
  std::vector<float> m_pf;             // m_p for the index
  SurfaceIndex       m_index;
  Vec3               m_centre, m_down{0, -1, 0}, m_wanderAt;
  double             m_radius = 1;

  double                      m_time   = 0;
  double                      m_toLand = 0;
  std::vector<Drop>           m_drops;
  std::vector<Track>          m_tracks;
  BeadCells                   m_cells;    // the beads at rest
  size_t                      m_resting = 0;
  std::vector<uint32_t>       m_runners;  // the beads running, in the order they started
  std::vector<uint32_t>       m_waiting;  // heavy enough to run once there is room
  std::unordered_set<CellKey> m_wet;      // where tracks went
  int                         m_landed = 0, m_runs = 0, m_dripped = 0;
};

}  // namespace

RainSettings clamped(RainSettings s)
{
  const auto fix = [](float v, float lo, float hi, float otherwise) { return std::isfinite(v) ? std::clamp(v, lo, hi) : otherwise; };
  s.amount = fix(s.amount, 0.05f, 1.f, 0.6f);
  s.time   = fix(s.time, 1.f, 120.f, 30.f);
  s.dried  = fix(s.dried, 0.f, 300.f, 0.f);
  return s;
}

RainSurface rainSurface(std::vector<uint32_t> tris, std::vector<float> uv, std::vector<int32_t> group)
{
  RainSurface s{.tris = std::move(tris), .uv = std::move(uv), .group = std::move(group)};
  const size_t n = s.triangles();
  s.across.assign(n * 3, -1);
  if(s.tris.size() != n * 3)
    return s;
  // an edge's first triangle, -2 once it has two
  std::unordered_map<uint64_t, int32_t> first;
  first.reserve(n * 2);
  for(size_t t = 0; t < n; t++)
    for(int k = 0; k < 3; k++)
    {
      const uint32_t a = s.tris[t * 3 + size_t(k)], b = s.tris[t * 3 + size_t((k + 1) % 3)];
      if(a == b)
        continue;
      const uint64_t key   = (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
      const int32_t  here  = int32_t(t * 3 + size_t(k));
      auto [it, fresh]     = first.try_emplace(key, here);
      if(fresh)
        continue;
      if(it->second >= 0)
      {
        s.across[size_t(here)]       = it->second;
        s.across[size_t(it->second)] = here;
      }
      it->second = -2;
    }
  return s;
}

RainSurface rainSurface(const mesh::BuiltMesh& geo)
{
  std::vector<uint32_t> tris;
  std::vector<float>    uv;
  std::vector<int32_t>  group;
  const auto            corner = [&](uint32_t vertex, std::span<const float> uvs, int64_t uvIndex) {
    tris.push_back(vertex);
    const bool has = uvIndex >= 0 && size_t(uvIndex) * 2 + 1 < uvs.size();
    uv.push_back(has ? uvs[size_t(uvIndex) * 2] : 0.f);
    uv.push_back(has ? uvs[size_t(uvIndex) * 2 + 1] : 0.f);
  };
  if(geo.refined())
  {
    if(!geo.source)
      return {};
    const mesh::MeshInput&     m       = *geo.source;
    const std::vector<int32_t> corners = mesh::cornerUvs(m);
    for(size_t p = 0; p + 1 < m.polyStart.size(); p++)
    {
      const int32_t s = m.polyStart[p], e = m.polyStart[p + 1];
      const int32_t g = p < m.polyMaterial.size() ? m.polyMaterial[p] : 0;
      for(int32_t k = s + 1; k + 1 < e; k++)
      {
        for(int32_t c : {s, k, k + 1})
          corner(uint32_t(m.polyVerts[size_t(c)]), m.uvs, corners[size_t(c)]);
        group.push_back(g);
      }
    }
  }
  else
    for(const mesh::Group& g : geo.groups)
      for(uint32_t i = g.start; i + 2 < g.start + g.count && i + 2 < geo.fullIndex.size(); i += 3)
      {
        for(int k = 0; k < 3; k++)
        {
          const uint32_t drawn = geo.fullIndex[i + uint32_t(k)];
          corner(drawn < geo.renderIndex.size() ? geo.renderIndex[drawn] : drawn, geo.uv, drawn);
        }
        group.push_back(g.materialIndex);
      }
  return rainSurface(std::move(tris), std::move(uv), std::move(group));
}

RainResult simulateRain(const RainSurface& surface, const RainInput& in, const RainSettings& settings)
{
  return Sim(surface, in, settings).run();
}

bool uvToTangent(const std::array<Vec3, 3>& p, const std::array<std::array<float, 2>, 3>& uv, std::array<float, 4>& out)
{
  const Vec3   e1 = p[1] - p[0], e2 = p[2] - p[0];
  const double du1 = uv[1][0] - uv[0][0], dv1 = uv[1][1] - uv[0][1], du2 = uv[2][0] - uv[0][0], dv2 = uv[2][1] - uv[0][1];
  const double det = du1 * dv2 - du2 * dv1;
  const Vec3   n   = cross(e1, e2);
  if(std::fabs(det) < 1e-14 || length(n) < 1e-18)
    return false;
  const Vec3 pu = (e1 * dv2 - e2 * dv1) * (1 / det);  // dP/du
  const Vec3 pv = (e2 * du1 - e1 * du2) * (1 / det);  // dP/dv
  const Vec3 nn = normalize(n);
  const Vec3 t0 = pu - nn * dot(nn, pu);
  if(length(t0) < 1e-12)
    return false;
  const Vec3   t          = normalize(t0);
  const double handedness = dot(cross(pu, pv * -1.0), nn) > 0 ? 1.0 : -1.0;
  const Vec3   b          = cross(nn, t) * handedness;
  out                     = {float(dot(pu, t)), float(dot(pv, t)), float(dot(pu, b)), float(dot(pv, b))};
  return true;
}

std::map<int, mat::WaterLayer> waterLayers(const RainSurface& surface, const RainInput& in, const RainResult& result)
{
  std::map<int, mat::WaterLayer> out;
  const size_t                   nv = in.posed.size() / 3;
  for(uint32_t v : surface.tris)
    if(v >= nv)
      return out;
  // every group the skin has is damp, beads or none
  for(int32_t g : surface.group)
    if(g >= 0 && !out.count(g))
      out[g].damp = result.damp;
  const double toMetres = 1 / std::max(1e-12, in.unitsPerMetre);
  const auto   vertex   = [&](uint32_t v) { return Vec3{in.posed[v * 3], in.posed[v * 3 + 1], in.posed[v * 3 + 2]} * toMetres; };
  std::unordered_map<uint32_t, std::optional<std::array<float, 4>>> frames;
  const auto frameOf = [&](uint32_t tri) -> const std::optional<std::array<float, 4>>& {
    auto [it, fresh] = frames.try_emplace(tri);
    if(fresh)
    {
      const uint32_t*                        c = &surface.tris[size_t(tri) * 3];
      std::array<std::array<float, 2>, 3>    uv;
      for(int k = 0; k < 3; k++)
        uv[size_t(k)] = {surface.uv[size_t(tri) * 6 + size_t(k) * 2], surface.uv[size_t(tri) * 6 + size_t(k) * 2 + 1]};
      std::array<float, 4> m;
      if(uvToTangent({vertex(c[0]), vertex(c[1]), vertex(c[2])}, uv, m))
        it->second = m;
    }
    return it->second;
  };
  const auto uvAt = [&](uint32_t tri, float b1, float b2) {
    const float* t = &surface.uv[size_t(tri) * 6];
    const float  b0 = 1 - b1 - b2;
    return std::array<float, 2>{t[0] * b0 + t[2] * b1 + t[4] * b2, t[1] * b0 + t[3] * b1 + t[5] * b2};
  };
  const auto put = [&](uint32_t tri, std::array<float, 2> a, std::array<float, 2> b, float radius, float height) {
    if(tri >= surface.triangles() || surface.group[tri] < 0 || radius <= 0 || height <= 0)
      return;
    const auto& m = frameOf(tri);
    if(!m)
      return;
    out[surface.group[tri]].items.push_back({.a = a, .b = b, .toTangent = *m, .radius = radius, .height = height});
  };
  for(const RainBead& bead : result.beads)
  {
    const auto at = uvAt(bead.tri, bead.b1, bead.b2);
    put(bead.tri, at, at, bead.radius, bead.height);
  }
  for(const RainTrack& t : result.tracks)
    put(t.tri, uvAt(t.tri, t.a1, t.a2), uvAt(t.tri, t.b1, t.b2), t.radius, t.height);
  return out;
}

}  // namespace fv::physics
