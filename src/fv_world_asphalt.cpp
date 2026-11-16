#include "fv_world_asphalt.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <figure/jobs/pool.hpp>
#include <glm/glm.hpp>

namespace fv::world {

namespace {

// The wear's widths in its repeat: a whole number, so what tiles across it tiles along it
constexpr int kAlong = int(kAsphaltRepeat / kAsphaltWidth);
static_assert(double(kAlong) * kAsphaltWidth == kAsphaltRepeat, "a whole number of the wear's widths a repeat");

constexpr double kPi = 3.14159265358979323846;

// The wheel paths' middles across the road (two lanes, a car's wheels either side of each lane's
// middle), and between them where the oil drips; how far either side they reach
constexpr std::array<double, 4> kWheels{0.7, 2.3, 3.7, 5.3};
constexpr std::array<double, 2> kOil{1.5, 4.5};
constexpr double                kWheelReach = 0.3, kOilReach = 0.28;
// The middle line, where the road was laid in two halves
constexpr double kJoint = 3.0;

uint32_t mix(uint32_t v)
{
  v                = v * 747796405U + 2891336453U;
  const uint32_t w = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (w >> 22U) ^ w;
}

uint32_t hash(uint32_t a, uint32_t b, uint32_t c = 0U)
{
  return mix(mix(mix(a) + b) + c);
}

double unit(uint32_t h)
{
  return double(h >> 8U) * (1.0 / 16777216.0);
}

int wrap(int i, int n)
{
  const int r = i % n;
  return r < 0 ? r + n : r;
}

double clamp01(double v)
{
  return std::clamp(v, 0.0, 1.0);
}

double sq(double v)
{
  return v * v;
}

// 0 at `a`, 1 at `b`, smoothly (either way round)
double smooth(double a, double b, double v)
{
  const double t = clamp01((v - a) / (b - a));
  return t * t * (3.0 - 2.0 * t);
}

double fade(double t)
{
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// How far along the road `y` is from `from`, the shorter way round the repeat
double alongFrom(double y, double from)
{
  double d = std::fmod(y - from, kAsphaltRepeat);
  if(d < -0.5 * kAsphaltRepeat)
    d += kAsphaltRepeat;
  else if(d > 0.5 * kAsphaltRepeat)
    d -= kAsphaltRepeat;
  return d;
}

// And on from it, 0 up to the repeat
double onFrom(double y, double from)
{
  const double d = std::fmod(y - from, kAsphaltRepeat);
  return d < 0.0 ? d + kAsphaltRepeat : d;
}

// A run of numbers from a seed: what lays the wear's things out
struct Dice
{
  uint32_t seed;
  uint32_t n = 0;
  double   operator()() { return unit(hash(seed, n++, 0x9E3779B9U)); }
  double   in(double a, double b) { return a + (b - a) * (*this)(); }
  int      below(int k) { return std::min(int((*this)() * double(k)), k - 1); }
};

struct Directions
{
  std::array<glm::dvec2, 16> d;
  Directions()
  {
    for(size_t k = 0; k < d.size(); k++)
      d[k] = glm::dvec2(std::cos(2.0 * kPi * double(k) / 16.0), std::sin(2.0 * kPi * double(k) / 16.0));
  }
};
const Directions kDirections;

// Gradient noise that tiles every `nx` by `ny` cells, at (x, y) in cells: about -1 to 1
double gradient(double x, double y, int nx, int ny, uint32_t seed)
{
  const double   fx = std::floor(x), fy = std::floor(y);
  const int      ix = int(fx), iy = int(fy);
  const double   tx = x - fx, ty = y - fy;
  const uint32_t s  = mix(seed);
  const auto     at = [&](int i, int j, double dx, double dy) {
    const glm::dvec2& g = kDirections.d[mix(mix(s + uint32_t(wrap(ix + i, nx))) + uint32_t(wrap(iy + j, ny))) & 15U];
    return g.x * dx + g.y * dy;
  };
  const double a = at(0, 0, tx, ty), b = at(1, 0, tx - 1.0, ty), c = at(0, 1, tx, ty - 1.0), e = at(1, 1, tx - 1.0, ty - 1.0);
  const double u = fade(tx), v = fade(ty);
  const double lo = a + (b - a) * u, hi = c + (e - c) * u;
  return 1.4 * (lo + (hi - lo) * v);
}

// Value noise likewise: 0 to 1
double value(double x, double y, int nx, int ny, uint32_t seed)
{
  const double   fx = std::floor(x), fy = std::floor(y);
  const int      ix = int(fx), iy = int(fy);
  const double   tx = x - fx, ty = y - fy;
  const uint32_t s  = mix(seed);
  const auto     at = [&](int i, int j) { return unit(mix(mix(s + uint32_t(wrap(ix + i, nx))) + uint32_t(wrap(iy + j, ny)))); };
  const double u = tx * tx * (3.0 - 2.0 * tx), v = ty * ty * (3.0 - 2.0 * ty);
  const double lo = at(0, 0) + (at(1, 0) - at(0, 0)) * u, hi = at(0, 1) + (at(1, 1) - at(0, 1)) * u;
  return lo + (hi - lo) * v;
}

// Cells across the wear for features some `cell` metres across (a whole number of them)
int cellsAcross(double cell)
{
  return std::max(1, int(std::lround(kAsphaltWidth / cell)));
}

// A noise over the wear (x across, y along, metres), its features some `cell` metres across with
// `octaves` finer ones, each half the size and the strength: about -1 to 1, tiling as the wear does
double wearNoise(double x, double y, double cell, int octaves, uint32_t seed)
{
  int    n   = cellsAcross(cell);
  double sum = 0.0, amp = 1.0, norm = 0.0;
  for(int o = 0; o < octaves; o++, n *= 2, amp *= 0.5)
  {
    const double s = double(n) / kAsphaltWidth;
    sum += amp * gradient(x * s, y * s, n, n * kAlong, hash(seed, uint32_t(o)));
    norm += amp;
  }
  return sum / norm;
}

double wearValue(double x, double y, double cell, uint32_t seed)
{
  const int    n = cellsAcross(cell);
  const double s = double(n) / kAsphaltWidth;
  return value(x * s, y * s, n, n * kAlong, seed);
}

// The nearest two of a jittered grid's points (a cell's id is its point's hash) on a grid of `nx`
// by `ny` cells `size` metres across that tiles
struct Cells
{
  double   f1 = 1.0e9, f2 = 1.0e9;
  uint32_t id = 0;
};

Cells cells(double x, double y, double size, int nx, int ny, uint32_t seed)
{
  const double cx = x / size, cy = y / size;
  const int    ix = int(std::floor(cx)), iy = int(std::floor(cy));
  Cells        out;
  for(int j = -1; j <= 1; j++)
    for(int i = -1; i <= 1; i++)
    {
      const uint32_t h  = hash(seed, uint32_t(wrap(ix + i, nx)), uint32_t(wrap(iy + j, ny)));
      const double   px = double(ix + i) + 0.1 + 0.8 * unit(h), py = double(iy + j) + 0.1 + 0.8 * unit(mix(h));
      const double   d  = std::hypot(px - cx, py - cy) * size;
      if(d < out.f1)
      {
        out.f2 = out.f1;
        out.f1 = d;
        out.id = h;
      }
      else if(d < out.f2)
        out.f2 = d;
    }
  return out;
}

// Cells on the wear `cell` metres across (a whole number of them across it)
Cells wearCells(double x, double y, double cell, uint32_t seed)
{
  const int n = cellsAcross(cell);
  return cells(x, y, kAsphaltWidth / double(n), n, n * kAlong, seed);
}

// ---- what is on the road ---------------------------------------------------------------------------------

// A crack, or tar over one: its points (x across, y along, metres; y past the repeat goes round) and
// its half-width at each
struct Path
{
  std::vector<glm::dvec2> at;
  std::vector<double>     half;
};

// A repair: a square of new asphalt sawn into the old, from x0 to x1 across and `length` on from y0
struct Patch
{
  double x0 = 0.0, x1 = 0.0, y0 = 0.0, length = 0.0;
  double age  = 0.0;   // 0 laid lately, 1 as old as the road
  double step = 0.0;   // millimetres over the road (under it: sunk)
  bool   sealed = false;  // its edge sealed with tar, or open
};

struct Hole
{
  glm::dvec2 at{0.0};
  double     radius = 0.0, depth = 0.0;  // metres
  uint32_t   seed   = 0;
};

// Crazing: an oval of it in a wheel path
struct Crazed
{
  glm::dvec2 at{0.0};
  double     along = 0.0, across = 0.0;  // half its length and its width
};

struct Stain
{
  glm::dvec2 at{0.0};
  double     radius = 0.0, strength = 0.0;
};

struct Layout
{
  std::vector<Path>   cracks, tar;
  std::vector<Patch>  patches;
  std::vector<Hole>   holes;
  std::vector<Crazed> crazed;
  std::vector<Stain>  stains;
};

// The repair a point is in, if any, and how far in from its edge
const Patch* patchAt(const Layout& L, double x, double y, double& in)
{
  for(const Patch& p : L.patches)
  {
    const double a = onFrom(y, p.y0);
    if(x >= p.x0 && x <= p.x1 && a <= p.length)
    {
      in = std::min({x - p.x0, p.x1 - x, a, p.length - a});
      return &p;
    }
  }
  return nullptr;
}

// A crack wandering from `from` on `heading` (radians, 0 across the road) for `length` metres, a
// random walk of its heading; `half` its half-width at its start, thinning to its end
Path wander(glm::dvec2 from, double heading, double length, double half, uint32_t seed)
{
  Path         p;
  const int    steps = std::max(2, int(length / 0.02));
  const double turn  = 0.05 + 0.1 * unit(hash(seed, 7U));
  for(int k = 0; k <= steps; k++)
  {
    p.at.push_back(from);
    p.half.push_back(half * (1.0 - 0.6 * double(k) / double(steps)));
    heading += turn * (2.0 * unit(hash(seed, uint32_t(k))) - 1.0);
    from += 0.02 * glm::dvec2(std::cos(heading), std::sin(heading));
  }
  return p;
}

// Tar over a crack: its path smoothed, a little wider here and there
Path tarOver(const Path& c, double half, uint32_t seed)
{
  Path t;
  for(size_t i = 0; i < c.at.size(); i += 3)
  {
    glm::dvec2 m(0.0);
    int        n = 0;
    for(size_t j = i >= 8 ? i - 8 : 0; j < std::min(c.at.size(), i + 9); j++, n++)
      m += c.at[j];
    m /= double(n);
    t.at.push_back(m);
    t.half.push_back(half * (0.75 + 0.5 * clamp01(0.5 + 0.8 * wearNoise(m.x, m.y, 0.25, 2, seed))));
  }
  return t;
}

// A crack held to the stretch from y0 to y1 along: it stops where it would cross a joint
Path within(const Path& p, double y0, double y1)
{
  Path out;
  for(size_t i = 0; i < p.at.size() && p.at[i].y >= y0 && p.at[i].y <= y1; i++)
  {
    out.at.push_back(p.at[i]);
    out.half.push_back(p.half[i]);
  }
  return out;
}

Layout layout(uint32_t seed)
{
  Layout L;
  Dice   d{hash(seed, 1U)};
  // the joints: straight across from edge to edge, some sealed
  for(size_t j = 0; j + 1 < kAsphaltJoints.size(); j++)
  {
    const uint32_t s = hash(seed, 50U + uint32_t(j));
    const double   y = kAsphaltJoints[j];
    const bool     sealed = d() < 0.5;
    const double   half   = d.in(0.015, 0.022);
    Path           c, t;
    for(double x = -0.1; x <= kAsphaltWidth + 0.1 + 1e-9; x += 0.05)
    {
      c.at.emplace_back(x, y);
      c.half.push_back(0.003 + 0.0015 * clamp01(0.5 + 0.7 * wearNoise(x, y, 0.3, 1, s)));
      t.at.emplace_back(x, y);
      t.half.push_back(half * (0.8 + 0.4 * clamp01(0.5 + 0.8 * wearNoise(x, y, 0.25, 2, s + 1U))));
    }
    L.cracks.push_back(std::move(c));
    if(sealed)
      L.tar.push_back(std::move(t));
  }
  // between them, each stretch its own things, none of them across a joint
  for(size_t k = 0; k + 1 < kAsphaltJoints.size(); k++)
  {
    const double   y0 = kAsphaltJoints[k], y1 = kAsphaltJoints[k + 1];
    const uint32_t s  = hash(seed, 100U + uint32_t(k));
    // a crack across it, wandering, some from an edge part of the way; half of them sealed with
    // tar, and branches off them
    if(d() < 0.35)
    {
      const double ya = d.in(y0 + 0.8, y1 - 0.8);
      double       xa = 0.0, xb = kAsphaltWidth;
      if(const double r = d(); r < 0.25)
        xb = d.in(1.5, 4.5);
      else if(r < 0.5)
        xa = d.in(1.5, 4.5);
      const double slope = d.in(-0.08, 0.08);
      Path         c;
      double       walk = 0.0;
      for(double x = xa; x <= xb + 1e-9; x += 0.02)
      {
        walk = 0.85 * walk + 0.004 * (2.0 * unit(hash(s, uint32_t(c.at.size()))) - 1.0);
        c.at.emplace_back(x, ya + slope * (x - 0.5 * kAsphaltWidth) + 0.09 * wearNoise(x, ya, 1.2, 2, s) + walk);
        const double edge = std::min(x, kAsphaltWidth - x);
        c.half.push_back(0.0012 + 0.0016 * clamp01(0.5 + 0.7 * wearNoise(x, ya, 0.4, 1, s + 1U)) + 0.0015 * smooth(0.8, 0.0, edge));
      }
      if(d() < 0.5)
        L.tar.push_back(tarOver(c, d.in(0.014, 0.024), s + 2U));
      const int branches = d.below(4);
      for(int b = 0; b < branches; b++)
      {
        const glm::dvec2 from = c.at[size_t(d.below(int(c.at.size())))];
        const double     side = d() < 0.5 ? 1.0 : -1.0;
        const double     away = (d() < 0.5 ? 0.0 : kPi) + side * d.in(0.35, 1.2);
        L.cracks.push_back(within(wander(from, away, d.in(0.15, 0.7), d.in(0.0008, 0.0014), s + 10U + uint32_t(b)), y0 + 0.1, y1 - 0.1));
      }
      L.cracks.push_back(std::move(c));
    }
    // a repair: in a lane or most of the road's width, sawn square, weathered some
    if(d() < 0.35)
    {
      Patch        p;
      const bool   wide = d() < 0.25;
      const double w    = wide ? d.in(3.2, 5.6) : d.in(0.9, 2.4);
      p.x0              = wide ? d.in(0.1, kAsphaltWidth - w - 0.1) : double(d.below(2)) * 0.5 * kAsphaltWidth + d.in(0.1, 0.5 * kAsphaltWidth - 0.1 - w);
      p.x1              = p.x0 + w;
      p.length          = std::min(d.in(1.0, 3.2), y1 - y0 - 1.2);
      p.y0              = d.in(y0 + 0.5, y1 - 0.5 - p.length);
      p.age             = d.in(0.35, 1.0);
      p.step            = d.in(-2.0, 1.2);
      p.sealed          = d() < 0.6;
      L.patches.push_back(p);
    }
    // a pothole by an edge or in a wheel path, clear of the repair
    if(d() < 0.35)
      for(int tries = 0; tries < 10; tries++)
      {
        Hole h;
        if(d() < 0.55)
          h.at.x = d() < 0.5 ? d.in(0.1, 0.7) : kAsphaltWidth - d.in(0.1, 0.7);
        else
          h.at.x = kWheels[size_t(d.below(4))] + d.in(-0.2, 0.2);
        h.radius = d.in(0.08, 0.28);
        h.at.y   = d.in(y0 + 1.5 * h.radius + 0.2, y1 - 1.5 * h.radius - 0.2);
        h.depth  = d.in(0.015, 0.045);
        h.seed   = hash(s, 500U + uint32_t(tries));
        bool far = true;
        for(const Patch& p : L.patches)
          far = far && (h.at.x + h.radius < p.x0 - 0.2 || h.at.x - h.radius > p.x1 + 0.2 || h.at.y + h.radius < p.y0 - 0.2 || h.at.y - h.radius > p.y0 + p.length + 0.2);
        if(far)
        {
          L.holes.push_back(h);
          break;
        }
      }
    // crazing in a wheel path
    if(d() < 0.35)
    {
      const double along = std::min(d.in(0.6, 1.8), (y1 - y0 - 0.4) / 2.4);
      L.crazed.push_back({.at = glm::dvec2(kWheels[size_t(d.below(4))] + d.in(-0.1, 0.1), d.in(y0 + 1.2 * along + 0.2, y1 - 1.2 * along - 0.2)), .along = along, .across = d.in(0.3, 0.5)});
    }
    // a crack along a wheel path, and a few fine ones anywhere
    if(d() < 0.4)
    {
      const glm::dvec2 from(kWheels[size_t(d.below(4))] + d.in(-0.12, 0.12), d.in(y0 + 0.3, y1 - 2.0));
      L.cracks.push_back(within(wander(from, 0.5 * kPi, d.in(1.0, y1 - from.y - 0.3), d.in(0.0012, 0.002), s + 20U), y0 + 0.1, y1 - 0.1));
    }
    for(int f = 0; f < 2; f++)
    {
      const glm::dvec2 from(d.in(0.3, kAsphaltWidth - 0.3), d.in(y0 + 1.0, y1 - 1.0));
      L.cracks.push_back(within(wander(from, d.in(0.0, 2.0 * kPi), d.in(0.2, 0.9), d.in(0.0007, 0.0012), s + 30U + uint32_t(f)), y0 + 0.1, y1 - 0.1));
    }
    // oil between the wheels
    for(int o = 0, n = 1 + d.below(2); o < n; o++)
    {
      const double r = d.in(0.12, 0.45);
      L.stains.push_back({.at = glm::dvec2(kOil[size_t(d.below(2))] + d.in(-0.3, 0.3), d.in(y0 + 1.3 * r + 0.1, y1 - 1.3 * r - 0.1)), .radius = r, .strength = d.in(0.2, 0.5)});
    }
  }
  // down the middle, where the road was laid in two halves: open along most of it, sealed in places
  {
    const uint32_t s = hash(seed, 200U);
    Path           run;
    const auto     close = [&](Path& p, bool tar) {
      if(p.at.size() > 1)
      {
        if(tar)
          L.tar.push_back(tarOver(p, 0.018, s + 5U));
        L.cracks.push_back(std::move(p));
      }
      p = {};
    };
    bool wasSealed = false;
    for(double y = 0.0; y < kAsphaltRepeat; y += 0.02)
    {
      const bool open       = wearNoise(0.5, y, 4.0, 2, s + 2U) > -0.15;
      const bool sealedHere = wearNoise(1.5, y, 5.0, 1, s + 3U) > 0.25;
      if(!open || sealedHere != wasSealed)
        close(run, wasSealed);
      wasSealed = sealedHere;
      if(!open)
        continue;
      run.at.emplace_back(kJoint + 0.012 * wearNoise(kJoint, y, 2.0, 2, s) + 0.003 * wearNoise(kJoint, y, 0.15, 1, s + 1U), y);
      run.half.push_back(0.0012 + 0.0014 * clamp01(0.5 + 0.7 * wearNoise(kJoint, y, 0.5, 1, s + 4U)));
    }
    close(run, wasSealed);
  }
  return L;
}

// ---- the wear's texels -------------------------------------------------------------------------------------

struct Wear
{
  int                W = 0, H = 0;
  double             px = 0.0;               // metres a texel
  std::vector<float> crack, shoulder, tar;  // the cracks' ink, the dip either side of them, tar over them
};

// A path drawn into `ink` (and the dip either side of it into `dip`, reaching `dipReach` past its edge)
void draw(Wear& m, const Path& p, std::vector<float>& ink, std::vector<float>* dip, double dipReach)
{
  for(size_t s = 0; s + 1 < p.at.size(); s++)
  {
    const glm::dvec2 a = p.at[s], b = p.at[s + 1];
    const double     reach = std::max(p.half[s], p.half[s + 1]) + (dip ? dipReach : 0.0) + 2.0 * m.px;
    const int        i0 = std::max(0, int(std::floor((std::min(a.x, b.x) - reach) / m.px)));
    const int        i1 = std::min(m.W - 1, int(std::ceil((std::max(a.x, b.x) + reach) / m.px)));
    const int        j0 = int(std::floor((std::min(a.y, b.y) - reach) / m.px));
    const int        j1 = int(std::ceil((std::max(a.y, b.y) + reach) / m.px));
    const glm::dvec2 ab = b - a;
    const double     ab2 = std::max(glm::dot(ab, ab), 1e-12);
    for(int j = j0; j <= j1; j++)
    {
      const size_t row = size_t(wrap(j, m.H)) * size_t(m.W);
      for(int i = i0; i <= i1; i++)
      {
        const glm::dvec2 ap   = glm::dvec2((double(i) + 0.5) * m.px, (double(j) + 0.5) * m.px) - a;
        const double     t    = clamp01(glm::dot(ap, ab) / ab2);
        const double     dist = glm::length(ap - ab * t);
        const double     half = p.half[s] + t * (p.half[s + 1] - p.half[s]);
        // (a crack finer than a texel inks it less)
        const double c = clamp01((half + 0.5 * m.px - dist) / m.px) * std::min(1.0, 2.0 * half / m.px + 0.3);
        float&       o = ink[row + size_t(i)];
        o              = std::max(o, float(c));
        if(dip)
        {
          const double q = clamp01(1.0 - (dist - half) / dipReach);
          float&       e = (*dip)[row + size_t(i)];
          e              = std::max(e, float(q * q));
        }
      }
    }
  }
}

// What lies where: the wheels' paths, the oil, the edges
struct Setting
{
  double edge = 0.0;          // metres to the nearer edge
  double wheel = 0.0, oil = 0.0;
  double weather = 0.0;       // -1 to 1
  double dust = 0.0;          // soil washed onto the edge
  double broken = 0.0;        // the edge broken away
};

// How far the wheels' paths drift across at `y` (cars keep to their lanes, more or less): none at a
// joint, where the stretch on from it may be another
double driftAt(double y, uint32_t seed)
{
  size_t k = 0;
  while(k + 2 < kAsphaltJoints.size() && y >= kAsphaltJoints[k + 1])
    k++;
  const double t = (y - kAsphaltJoints[k]) / (kAsphaltJoints[k + 1] - kAsphaltJoints[k]);
  return 0.1 * wearNoise(0.0, y, 3.0, 2, seed + 4U) * std::sin(kPi * std::clamp(t, 0.0, 1.0));
}

Setting settingAt(double x, double y, double drift, uint32_t seed)
{
  Setting s;
  s.edge = std::min(x, kAsphaltWidth - x);
  for(const double c : kWheels)
    s.wheel += std::exp(-sq((x - c - drift) / kWheelReach));
  for(const double c : kOil)
    s.oil += std::exp(-sq((x - c - drift) / kOilReach));
  s.wheel   = std::min(s.wheel, 1.0);
  s.weather = wearNoise(x, y, 1.5, 4, seed + 1U);
  if(s.edge < 1.0)
  {
    s.dust               = smooth(0.75, 0.05, s.edge + 0.2 * wearNoise(x, y, 0.5, 2, seed + 8U));
    const double brokeTo = 0.04 + 0.16 * clamp01(0.5 + 0.6 * wearNoise(x, y, 0.6, 3, seed + 9U));
    s.broken             = smooth(brokeTo + 0.01, brokeTo - 0.01, s.edge);
  }
  return s;
}

// How much of a crazed oval a point is in, and the crazing's cell there
double crazedAt(const Layout& L, double x, double y, uint32_t seed)
{
  double most = 0.0;
  for(const Crazed& c : L.crazed)
  {
    const double q = sq((x - c.at.x) / c.across) + sq(alongFrom(y, c.at.y) / c.along);
    if(q < 2.0)
      most = std::max(most, smooth(1.0, 0.55, q + 0.35 * wearNoise(x, y, 0.35, 2, seed + 12U)));
  }
  return most;
}

// How far into a pothole a point is (1 in it, 0 out of it, between on its rim) and its crumbs round it
struct InHole
{
  double in = 0.0, rim = 0.0, depth = 0.0;
};

InHole holeAt(const Layout& L, double x, double y)
{
  InHole out;
  for(const Hole& h : L.holes)
  {
    const double dx = x - h.at.x, dy = alongFrom(y, h.at.y);
    const double r  = std::hypot(dx, dy);
    if(r > 1.5 * h.radius)
      continue;
    const double a  = std::atan2(dy, dx);
    const double rr = h.radius * (1.0 + 0.3 * wearNoise(h.at.x + std::cos(a) * h.radius, h.at.y + std::sin(a) * h.radius, 0.12, 2, h.seed));
    const double q  = r / std::max(rr, 1e-3);
    const double in = smooth(1.0, 0.85, q);
    if(in > out.in)
    {
      out.in    = in;
      out.depth = h.depth;
    }
    out.rim = std::max(out.rim, smooth(1.3, 1.0, q) * (1.0 - in));
  }
  return out;
}

struct Look
{
  glm::dvec3 colour{0.0};  // linear
  double     rough = 0.9, grain = 1.0, height = 0.0;  // height in millimetres
};

const glm::dvec3 kTar(0.028, 0.027, 0.026);
const glm::dvec3 kCrack(0.018, 0.017, 0.016);
const glm::dvec3 kMoss(0.045, 0.062, 0.03);
const glm::dvec3 kSoil(0.17, 0.14, 0.105);

Look texel(const Layout& L, const Wear& m, int i, int j, double drift, uint32_t seed)
{
  const double  x = (double(i) + 0.5) * m.px, y = (double(j) + 0.5) * m.px;
  const size_t  at = size_t(j) * size_t(m.W) + size_t(i);
  const Setting s  = settingAt(x, y, drift, seed);
  const double  mottle = wearNoise(x, y, 0.2, 3, seed + 2U), fine = wearNoise(x, y, 0.03, 2, seed + 3U);
  double        inPatch = 0.0;
  const Patch*  patch   = patchAt(L, x, y, inPatch);
  Look          out;

  // the binder greyed by the weather, and its stones showing as it wears: more where the wheels
  // went and at the edges, few on a repair
  const double v1 = wearValue(x, y, 0.008, seed + 5U), v2 = wearValue(x, y, 0.013, seed + 6U), hue = wearValue(x, y, 0.011, seed + 7U);
  double       exposed = clamp01(0.38 + 0.12 * s.weather + 0.22 * s.wheel + 0.18 * smooth(0.9, 0.2, s.edge));
  double       greyed  = clamp01(0.6 + 0.25 * s.weather + 0.1 * mottle);
  if(patch)
  {
    exposed = 0.1 + 0.25 * patch->age;
    greyed  = clamp01(0.75 * patch->age + 0.1 * mottle);
  }
  const double     stone = smooth(1.0 - exposed - 0.06, 1.0 - exposed + 0.06, 0.6 * v1 + 0.4 * v2);
  const glm::dvec3 stones = (0.13 + 0.16 * hue) * glm::dvec3(1.0 + 0.08 * (v2 - 0.5), 1.0, 1.0 - 0.1 * (v2 - 0.5));
  out.colour = glm::mix((0.045 + 0.07 * greyed) * glm::dvec3(1.02, 1.0, 0.96), stones, 0.85 * stone);
  out.rough  = 0.9 - 0.07 * stone + 0.03 * fine;
  out.height = 0.25 * fine + 0.35 * (stone - 0.5) + 0.6 * mottle;

  // the wheels' paths: paler, smoother, a little sunk; oil between them, and where it pooled
  const double wheel = patch ? 0.4 * s.wheel : s.wheel;
  out.colour *= (1.0 + 0.1 * wheel) * (1.0 - 0.22 * s.oil * clamp01(0.6 + 0.5 * mottle));
  out.rough -= 0.12 * wheel;
  out.height -= 3.0 * s.wheel;
  for(const Stain& st : L.stains)
  {
    const double r = std::hypot(x - st.at.x, alongFrom(y, st.at.y));
    if(r > 1.3 * st.radius)
      continue;
    const double q = r / (st.radius * (1.0 + 0.3 * wearNoise(x, y, 0.15, 2, seed + 13U)));
    if(q < 1.0)
    {
      const double blob = smooth(1.0, 0.45, q);
      out.colour *= 1.0 - st.strength * blob;
      out.rough -= 0.06 * blob;
    }
  }

  // ravelled: stones lost, pits where they were (at the edges, and where the weather was worst)
  if(!patch)
  {
    const double ravel = 0.5 * smooth(0.6, 0.1, s.edge) + 0.3 * std::max(0.0, s.weather - 0.5);
    if(ravel > 0.0)
    {
      const double pit = smooth(1.0 - 0.25 * ravel - 0.02, 1.0 - 0.25 * ravel + 0.02, wearValue(x, y, 0.01, seed + 14U));
      out.colour *= 1.0 - 0.5 * pit;
      out.height -= 1.5 * pit;
    }
  }

  // the edges: soil washed onto them, cracked into big pieces, broken away
  if(s.dust > 0.0)
  {
    const glm::dvec3 soil = kSoil * (0.85 + 0.3 * (v2 - 0.5) + 0.15 * fine);
    out.colour            = glm::mix(out.colour, soil, 0.5 * s.dust);
    out.rough += 0.05 * s.dust;
    out.grain = 1.0 - 0.3 * s.dust;
  }
  double crack = 0.0;
  if(s.edge < 0.8 && !patch)
  {
    const double near = smooth(0.7, 0.25, s.edge + 0.15 * wearNoise(x, y, 0.4, 1, seed + 10U));
    const Cells  c    = wearCells(x, y, 0.22, seed + 11U);
    crack             = std::max(crack, smooth(0.006, 0.0015, c.f2 - c.f1) * near);
  }
  if(s.broken > 0.0)
  {
    const glm::dvec3 gravel = glm::mix(kSoil * (0.7 + 0.4 * v2), glm::dvec3(0.1 + 0.18 * v1), smooth(0.4, 0.8, v1));
    out.colour              = glm::mix(out.colour, gravel, s.broken);
    out.rough               = glm::mix(out.rough, 0.97, s.broken);
    out.height -= 10.0 * s.broken;
    out.grain = glm::mix(out.grain, 1.0, s.broken);
  }

  // crazing: the wheel path broken into plates, some lost
  if(const double craze = patch ? 0.0 : crazedAt(L, x, y, seed); craze > 0.0)
  {
    const Cells  c     = wearCells(x, y, 0.15, seed + 15U);
    const double plate = unit(c.id) - 0.5;
    crack              = std::max(crack, smooth(0.005, 0.001, c.f2 - c.f1) * craze);
    out.colour *= 1.0 + 0.12 * plate * craze;
    out.height += 1.2 * plate * craze;
    if(unit(mix(c.id)) < 0.08 * craze)
    {
      const double lost = smooth(0.004, 0.012, 0.5 * (c.f2 - c.f1));
      out.colour        = glm::mix(out.colour, glm::dvec3(0.06 + 0.12 * v1), lost);
      out.height -= 12.0 * lost;
      out.rough = glm::mix(out.rough, 0.95, lost);
    }
  }

  // potholes: gravel and dark soil in them, crumbs round them
  if(!patch)
  {
    const InHole h = holeAt(L, x, y);
    if(h.in > 0.0 || h.rim > 0.0)
    {
      const glm::dvec3 bed = glm::mix(kSoil * 0.45, glm::dvec3(0.08 + 0.14 * v1), smooth(0.45, 0.75, v1));
      out.colour           = glm::mix(out.colour * (1.0 + 0.2 * h.rim), bed, h.in);
      out.rough            = glm::mix(out.rough, 0.95, h.in);
      out.height -= 1000.0 * h.depth * h.in * (0.75 + 0.25 * fine) + 1.0 * h.rim;
    }
  }

  // the cracks drawn into it (none under a repair), moss in those near the edges; tar over some
  if(!patch)
    crack = std::max(crack, double(m.crack[at]));
  const double tar = patch ? 0.0 : double(m.tar[at]), shoulder = patch ? 0.0 : double(m.shoulder[at]);
  crack *= 1.0 - tar;
  out.colour *= 1.0 - 0.12 * shoulder;
  out.colour = glm::mix(out.colour, kCrack, 0.9 * crack);
  if(crack > 0.0 && s.edge < 1.2)
    out.colour = glm::mix(out.colour, kMoss, crack * 0.6 * smooth(1.2, 0.2, s.edge) * clamp01(0.5 + wearNoise(x, y, 0.6, 1, seed + 16U)));
  out.rough  = glm::mix(out.rough, 0.97, crack);
  out.height -= 2.5 * crack + 0.6 * shoulder;
  out.grain = glm::mix(out.grain, 0.5, crack);
  out.colour = glm::mix(out.colour, kTar * (1.0 + 0.2 * fine), tar);
  out.rough  = glm::mix(out.rough, 0.42 + 0.08 * fine, tar);
  out.grain  = glm::mix(out.grain, 0.15, tar);
  out.height += 1.2 * tar;

  // a repair's edge: sealed with tar, or open; its new asphalt stands over the old or sunk
  if(patch)
  {
    out.height += patch->step;
    const double seam = smooth(0.03, 0.02, inPatch);
    if(patch->sealed)
    {
      out.colour = glm::mix(out.colour, kTar, seam);
      out.rough  = glm::mix(out.rough, 0.45, seam);
      out.grain  = glm::mix(out.grain, 0.15, seam);
      out.height += 0.8 * seam;
    }
    else
    {
      const double open = smooth(0.005, 0.0, inPatch);
      out.colour        = glm::mix(out.colour, kCrack, 0.9 * open);
      out.height -= 2.0 * open;
    }
  }

  out.colour *= 1.0 + 0.07 * mottle + 0.04 * fine;
  out.rough = std::clamp(out.rough, 0.3, 1.0);
  out.grain = clamp01(out.grain);
  return out;
}

// ---- the images ---------------------------------------------------------------------------------------------

uint8_t byteOf(double v)
{
  return uint8_t(std::lround(clamp01(v) * 255.0));
}

double srgbOf(double linear)
{
  linear = clamp01(linear);
  return linear <= 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

double linearOf(uint8_t srgb)
{
  const double v = double(srgb) / 255.0;
  return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}

// A height field's normals (heights in millimetres, `texel` millimetres apart), glTF's way round: X
// and Y as two bytes. Along it, it goes round; across, round only when `round`.
AsphaltImage normalsOf(const std::vector<float>& h, int W, int H, double texel, bool round)
{
  AsphaltImage out{.width = W, .height = H, .channels = 2, .pixels = std::vector<uint8_t>(size_t(W) * size_t(H) * 2U)};
  fv::jobs::parallelFor(size_t(H), 16, [&](size_t begin, size_t end) {
    for(size_t j = begin; j < end; j++)
    {
      const size_t up = size_t(wrap(int(j) - 1, H)) * size_t(W), down = size_t(wrap(int(j) + 1, H)) * size_t(W), row = j * size_t(W);
      for(int i = 0; i < W; i++)
      {
        const int    l  = round ? wrap(i - 1, W) : std::max(i - 1, 0);
        const int    r  = round ? wrap(i + 1, W) : std::min(i + 1, W - 1);
        const double dx = (double(h[row + size_t(r)]) - double(h[row + size_t(l)])) / (double(round ? 2 : r - l) * texel);
        const double dy = (double(h[down + size_t(i)]) - double(h[up + size_t(i)])) / (2.0 * texel);
        // (+Y toward v = 0: the height rising along v leans the normal toward v = 0)
        const glm::dvec3 n = glm::normalize(glm::dvec3(-dx, dy, 1.0));
        out.pixels[(row + size_t(i)) * 2U]      = byteOf(0.5 + 0.5 * n.x);
        out.pixels[(row + size_t(i)) * 2U + 1U] = byteOf(0.5 + 0.5 * n.y);
      }
    }
  });
  return out;
}

// The stones' grain: stones some 9 mm across set in a sand of fines, a few of them gone, tiled
AsphaltImage grainOf(int n, uint32_t seed)
{
  constexpr int stonesAcross = 168, finesAcross = 520;
  const double  stone = kAsphaltGrain / stonesAcross, fines = kAsphaltGrain / finesAcross, texel = kAsphaltGrain / double(n);
  std::vector<float> h(size_t(n) * size_t(n));
  fv::jobs::parallelFor(size_t(n), 16, [&](size_t begin, size_t end) {
    for(size_t j = begin; j < end; j++)
      for(int i = 0; i < n; i++)
      {
        const double   x = (double(i) + 0.5) * texel, y = (double(j) + 0.5) * texel;
        const Cells    c = cells(x, y, stone, stonesAcross, stonesAcross, seed);
        const double   e = 0.5 * (c.f2 - c.f1);  // (about how far in from the stone's edge)
        const double   gap = stone * (0.06 + 0.08 * unit(mix(c.id)));
        const double   top = std::sqrt(clamp01((e - gap) / (0.32 * stone)));
        double         hh  = unit(mix(c.id + 2U)) < 0.07 ? -0.5 * top : (0.5 + 0.8 * unit(mix(c.id + 1U))) * top;  // mm
        const Cells    f   = cells(x, y, fines, finesAcross, finesAcross, seed + 1U);
        hh += (hh <= 0.05 ? 0.25 : 0.06) * std::sqrt(clamp01(0.5 * (f.f2 - f.f1) / (0.35 * fines)));
        h[j * size_t(n) + size_t(i)] = float(hh);
      }
  });
  return normalsOf(h, n, n, texel * 1000.0, true);
}

}  // namespace

AsphaltMaps asphaltMaps(uint32_t seed, const AsphaltSize& size)
{
  const int W = std::max(size.across, 8), H = W * kAlong;
  const int P = std::clamp(size.paint, 1, W), k = W / P;
  Wear      m{.W = W, .H = H, .px = kAsphaltWidth / double(W)};
  m.crack.assign(size_t(W) * size_t(H), 0.0F);
  m.shoulder.assign(m.crack.size(), 0.0F);
  m.tar.assign(m.crack.size(), 0.0F);
  const Layout L = layout(seed);
  for(const Path& c : L.cracks)
    draw(m, c, m.crack, &m.shoulder, 0.02);
  for(const Path& t : L.tar)
    draw(m, t, m.tar, nullptr, 0.0);

  AsphaltMaps out;
  out.colour = {.width = W, .height = H, .channels = 4, .pixels = std::vector<uint8_t>(size_t(W) * size_t(H) * 4U)};
  out.rough  = {.width = W, .height = H, .channels = 2, .pixels = std::vector<uint8_t>(size_t(W) * size_t(H) * 2U)};
  std::vector<float> height(size_t(W) * size_t(H));
  fv::jobs::parallelFor(size_t(H), 16, [&](size_t begin, size_t end) {
    for(size_t j = begin; j < end; j++)
    {
      const double drift = driftAt((double(j) + 0.5) * m.px, seed);
      for(int i = 0; i < W; i++)
      {
        const Look   t  = texel(L, m, i, int(j), drift, seed);
        const size_t at = j * size_t(W) + size_t(i);
        for(int c = 0; c < 3; c++)
          out.colour.pixels[at * 4U + size_t(c)] = byteOf(srgbOf(t.colour[c]));
        out.colour.pixels[at * 4U + 3U] = 255;
        out.rough.pixels[at * 2U]       = byteOf(t.grain);
        out.rough.pixels[at * 2U + 1U]  = byteOf(t.rough);
        height[at]                      = float(t.height);
      }
    }
  });
  out.normal = normalsOf(height, W, H, m.px * 1000.0, false);
  out.grain  = grainOf(std::max(size.grain, 8), hash(seed, 900U));

  // the markings: whole paint, worn through where the wheels cross it, chipped, cracked where the road
  // is, and none over tar, a repair or a pothole (each came after it)
  out.paint = {.width = P, .height = H, .channels = 4, .pixels = std::vector<uint8_t>(size_t(P) * size_t(H) * 4U)};
  fv::jobs::parallelFor(size_t(H), 16, [&](size_t begin, size_t end) {
    for(size_t j = begin; j < end; j++)
    {
      const double drift = driftAt((double(j) + 0.5) * m.px, seed);
      for(int ip = 0; ip < P; ip++)
      {
        const double x = (double(ip) + 0.5) * kAsphaltWidth / double(P), y = (double(j) + 0.5) * m.px;
        glm::dvec3   under(0.0);
        double       crack = 0.0, tar = 0.0, shoulder = 0.0;
        for(int t = 0; t < k; t++)
        {
          const size_t at = j * size_t(W) + size_t(ip * k + t);
          for(int c = 0; c < 3; c++)
            under[c] += linearOf(out.colour.pixels[at * 4U + size_t(c)]) / double(k);
          crack    = std::max(crack, double(m.crack[at]));
          tar      = std::max(tar, double(m.tar[at]));
          shoulder = std::max(shoulder, double(m.shoulder[at]));
        }
        const Setting s       = settingAt(x, y, drift, seed);
        double        inPatch = 0.0;
        double        cover   = 1.0 - smooth(0.5, 0.8, 0.5 + 0.45 * wearNoise(x, y, 0.12, 3, seed + 20U) + 0.3 * s.wheel + 0.1 * s.weather);
        cover *= smooth(0.9, 0.85, wearValue(x, y, 0.015, seed + 21U));
        cover *= 1.0 - clamp01(1.6 * crack + 0.7 * shoulder);
        cover *= (1.0 - tar) * (1.0 - s.broken) * (1.0 - holeAt(L, x, y).in);
        if(const double craze = crazedAt(L, x, y, seed); craze > 0.0)
          cover *= 1.0 - craze * (unit(wearCells(x, y, 0.15, seed + 15U).id) < 0.45 ? 1.0 : 0.3);
        if(patchAt(L, x, y, inPatch))
          cover = 0.0;
        const glm::dvec3 paint = glm::dvec3(0.56, 0.54, 0.48) * (0.85 + 0.15 * wearNoise(x, y, 0.5, 2, seed + 22U)) * (1.0 - 0.25 * s.dust);
        const glm::dvec3 c     = glm::mix(under, paint, cover);
        const size_t     at    = j * size_t(P) + size_t(ip);
        for(int ch = 0; ch < 3; ch++)
          out.paint.pixels[at * 4U + size_t(ch)] = byteOf(srgbOf(c[ch]));
        out.paint.pixels[at * 4U + 3U] = 255;
      }
    }
  });
  return out;
}

}  // namespace fv::world
