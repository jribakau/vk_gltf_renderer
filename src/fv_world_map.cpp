#include "fv_world_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_map>


#include <figure/jobs/pool.hpp>

namespace fv::world {

namespace {

constexpr double kPi = 3.14159265358979323846;

// 0 at `a`, 1 at `b`, eased; `a` may be the larger
double smooth(double a, double b, double x)
{
  const double t = std::clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

double mix(double a, double b, double t)
{
  return a + (b - a) * t;
}

// A cell's neighbours, counter-clockwise from +i
constexpr int kDi[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kDj[8] = {0, 1, 1, 1, 0, -1, -1, -1};

// ---- the bake's numbers ------------------------------------------------------------------------------

// Rain. The air loses a share 1/e of its water over this much land (radians of the planet), takes it
// up over this much sea, and loses all but e^-kRainLift of it over a rise as high as the heights are.
// Lifted over a rise, it comes down again over kRainSettle: until then lower ground lifts it no further
// (rolling hills are not a mountain range, and behind a range it stays dry).
constexpr double kRainReach = 1.0, kRainFill = 0.15, kRainLift = 1.5, kRainSettle = 0.3;
// How wet the ground is by what the air over it holds, and how much a slope that lifts it adds
constexpr double kMoistFull = 0.75, kMoistPower = 0.7, kMoistLift = 25.0;
// What runs off a cell, by how wet it is: this much even where it never rains
constexpr double kRunoffDry = 0.15;

// Worn ground keeps at least this slope toward where its water runs, so a filled hollow drains
constexpr double kFillSlope = 0.002;
// A lake: a hollow at least this deep (metres, and this share of the radius by the relief), over at
// least so many cells, where the weather is at least this wet. Its water stands under the hollow's
// brim by kLakeUnder of its depth (between these, metres), and its cells are those at least
// kLakeShallow under that: at least kLakeWetCells of them.
constexpr double kLakeDeep = 5.0, kLakeDeepShare = 0.0006, kLakeWet = 0.3, kLakeCold = -3.0;
constexpr double kLakeUnder = 0.3, kLakeUnderLeast = 2.0, kLakeUnderMost = 10.0, kLakeShallow = 0.5;
constexpr uint32_t kLakeCells = 8, kLakeWetCells = 4;

// Wear. The heights the land is measured by are this share of the radius by the relief; a place that
// high rises by kWearSteady of the radius (by the relief) for each unit of wear, which is the height
// a stream holds against it, and never to over what it was: the ridges stay the stages', the valleys
// are the water's. kWear a round is how far a cut travels upstream: 3 in all reaches most of the way
// to the divides.
constexpr double   kHeights    = 0.03;
constexpr double   kWearSteady = 0.004;
constexpr double   kWear       = 0.125;
constexpr uint32_t kWearRounds = 24;
// No slope steeper than loose ground holds
constexpr double kTalus = 0.8;
// A valley's floor: where the wet ground upstream is at least kFloorFrom of a river's, its floor stands
// over its bed by up to kFloorMost of the heights the land is measured by, the whole of it by
// kFloorDoublings doublings of that water, and none where it runs steeper than twice kFloorSteep
constexpr double kFloorFrom = 0.1, kFloorMost = 0.08, kFloorDoublings = 4.0, kFloorSteep = 0.3;
// and no deeper than this share of its height over the water it ends in (the sea, a lake)
constexpr double kFloorOver = 0.5;
// Nowhere is worn to under a share of the height it had: most of it on low land (a plain by the sea
// stays a plain over it), little in the heights, where valleys cut deep. Between these shares of
// kHeights.
constexpr double kWearKeepLow = 0.8, kWearKeepHigh = 0.25, kWearKeepFrom = 0.1, kWearKeepTo = 0.6;

// A river: where the wet ground upstream is at least this (a share of the radius, squared; and so
// many cells), this wide there, wider by its water to the power kRiverGrow, and no wider than this
// share of a cell (its lookup's window is a few cells)
constexpr double kRiverFrom = 0.035, kRiverFromCells = 40.0, kRiverNarrow = 6.0, kRiverGrow = 0.4, kRiverWide = 0.8;
// Its banks ease into the land over this many half widths, between these (metres; shares of a cell)
constexpr double kBankHalves = 3.0, kBankLeast = 10.0, kBankMost = 0.9;
// Within this many metres of the sea's level a river's own comes down to it early: the low ground
// by a shore dips under the sea in places the map's cells do not see, and a river held a few metres
// over them would cross them on a bank of its own
constexpr double kRiverLow = 6.0;
// A river's point in a cell is up to this share of the cell off its middle, either way: cells in a
// row are not a straight line of river
constexpr double kRiverStray = 0.25;
// How many cells round a point its rivers are looked for in (kMapGutter)
constexpr int kWindow = kMapGutter;
// A lake's water is drawn where its share of the cells round a point is at least this (a half along
// a straight shore of cells: so a third of a cell past them)
constexpr double kLakeSheet = 0.25;
// Near a corner of the cube the map is level within this many cells and eases in over as many more;
// rivers within twice that of one are in its list
constexpr int kCornerLevel = 4;

// ---- lookups -----------------------------------------------------------------------------------------

struct Tap
{
  uint32_t at = 0;  // the lowest of four cells, in a field with gutters
  float    fx = 0.0F, fy = 0.0F;
};

Tap tapOf(const CubeGrid& g, const MapPoint& p)
{
  const double i = std::floor(p.x), j = std::floor(p.y);
  return {uint32_t(g.padded(p.face, int(i), int(j))), float(p.x - i), float(p.y - j)};
}

float read(const std::vector<float>& field, const Tap& t, uint32_t stride)
{
  const float a = field[t.at], b = field[t.at + 1], c = field[t.at + stride], d = field[t.at + stride + 1];
  const float low = a + (b - a) * t.fx, high = c + (d - c) * t.fx;
  return low + (high - low) * t.fy;
}

// The weights of four cells in a row for a point `f` of the way from the second to the third: a cubic
// B-spline
void splineWeights(double f, double w[4])
{
  const double f2 = f * f, f3 = f2 * f;
  w[0] = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  w[1] = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0;
  w[2] = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0;
  w[3] = f3 / 6.0;
}

// A smooth read (that spline over four cells each way): its slope has no creases, which a height's
// must not
double smoothRead(const CubeGrid& g, const std::vector<float>& field, const glm::dvec3& direction)
{
  const MapPoint p  = g.at(direction);
  const double   fi = std::floor(p.x), fj = std::floor(p.y);
  double         wx[4], wy[4];
  splineWeights(p.x - fi, wx);
  splineWeights(p.y - fj, wy);
  const size_t stride = g.stride();
  const size_t first  = g.padded(p.face, int(fi) - 1, int(fj) - 1);
  double       sum    = 0.0;
  for(int b = 0; b < 4; b++)
  {
    const float* row = field.data() + first + size_t(b) * stride;
    sum += wy[b] * (wx[0] * double(row[0]) + wx[1] * double(row[1]) + wx[2] * double(row[2]) + wx[3] * double(row[3]));
  }
  return sum;
}

// Which of the cube's corners a direction is nearest
uint32_t cornerOf(const glm::dvec3& d)
{
  return (d.x > 0.0 ? 1U : 0U) | (d.y > 0.0 ? 2U : 0U) | (d.z > 0.0 ? 4U : 0U);
}

// How many cells a cell is from its face's nearest corner (the larger of the two ways)
int fromCorner(int n, int i, int j)
{
  return std::max(std::min(i, n - 1 - i), std::min(j, n - 1 - j));
}

template <class T>
T bent(const T& a, const T& b, const T& c, double t)
{
  const double s = 1.0 - t;
  return a * (s * s) + b * (2.0 * s * t) + c * (t * t);
}

double bent(const float v[3], double t)
{
  return bent(double(v[0]), double(v[1]), double(v[2]), t);
}

struct Nearest
{
  double score = std::numeric_limits<double>::max();  // metres past the river's edge
  double away = 0.0, level = 0.0, half = 0.0;
};

// The piece's point nearest `d`, the curve taken as four straight stretches
void nearestOn(const RiverPiece& p, const glm::dvec3& d, double radius, Nearest& best)
{
  glm::dvec3 from = p.a;
  for(int k = 1; k <= 4; k++)
  {
    const glm::dvec3 to    = bent(p.a, p.b, p.c, 0.25 * k);
    const glm::dvec3 along = to - from;
    const double     len2  = glm::dot(along, along);
    const double     s     = len2 > 0.0 ? std::clamp(glm::dot(d - from, along) / len2, 0.0, 1.0) : 0.0;
    const double     away  = glm::length(d - (from + along * s)) * radius;
    const double     t     = 0.25 * (double(k - 1) + s);
    const double     half  = bent(p.half, t);
    if(away - half < best.score)
      best = {away - half, away, bent(p.level, t), half};
    from = to;
  }
}

// ---- the wind ----------------------------------------------------------------------------------------

// Which way the air goes over a place, a unit's length at most: toward the west by the equator and
// the poles and toward the east between, and a little toward the equator or away from it
glm::dvec3 wind(const glm::dvec3& d)
{
  const glm::dvec3 across = glm::cross(glm::dvec3(0.0, 1.0, 0.0), d);  // east, as long as the latitude's cosine
  const double     level  = glm::length(across);
  if(level < 1e-9)
    return glm::dvec3(0.0);
  const glm::dvec3 east     = across / level;
  const glm::dvec3 north    = glm::cross(d, east);
  const double     latitude = std::asin(std::clamp(d.y, -1.0, 1.0));
  return (east * -std::cos(3.0 * latitude) + north * (-0.35 * std::sin(6.0 * latitude))) * std::min(1.0, 3.0 * level);
}

// ---- sorting -----------------------------------------------------------------------------------------

// Sorts by the high 32 bits (a key), which carry the low 32 with them
void sortByKey(std::vector<uint64_t>& items, std::vector<uint64_t>& scratch)
{
  scratch.resize(items.size());
  for(const int shift : {32, 43, 54})
  {
    const int      bits = shift == 54 ? 10 : 11;
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    size_t         counts[2048] = {};
    for(const uint64_t v : items)
      counts[(v >> shift) & mask]++;
    size_t at = 0;
    for(size_t b = 0; b <= mask; b++)
    {
      const size_t c = counts[b];
      counts[b]      = at;
      at += c;
    }
    for(const uint64_t v : items)
      scratch[counts[(v >> shift) & mask]++] = v;
    items.swap(scratch);
  }
}

uint32_t bitsOf(float f)
{
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

}  // namespace

// ---- the grid ----------------------------------------------------------------------------------------

CubeGrid::CubeGrid(uint32_t cells)
    : m_n(std::max(cells, uint32_t(kMapGutter)))
    , m_edges(size_t(24) * m_n)
{
  // Each face's four sides: which face lies beyond, and how its cells run there. Asked of the sphere
  // itself: a point a hair off the edge, at a cell's middle along it.
  constexpr double kOff = 1.0 + 1e-7;
  for(uint8_t face = 0; face < 6; face++)
    for(int side = 0; side < 4; side++)
      for(uint32_t a = 0; a < m_n; a++)
      {
        const double    along = (double(a) + 0.5) * 2.0 / double(m_n) - 1.0;
        const double    u     = side == 0 ? -kOff : side == 1 ? kOff : along;
        const double    v     = side == 2 ? -kOff : side == 3 ? kOff : along;
        const FacePoint p     = locate(direction(face, u, v));
        Edge            e;
        e.face    = p.face;
        e.acrossU = std::abs(p.u) > std::abs(p.v);
        e.high    = (e.acrossU ? p.u : p.v) > 0.0;
        e.along   = uint32_t(std::clamp(std::floor(((e.acrossU ? p.v : p.u) + 1.0) * 0.5 * double(m_n)), 0.0, double(m_n - 1)));
        m_edges[(size_t(face) * 4 + size_t(side)) * m_n + a] = e;
      }
}

bool CubeGrid::cell(uint8_t face, int i, int j, uint32_t& out) const
{
  const int  n   = int(m_n);
  const bool iin = i >= 0 && i < n, jin = j >= 0 && j < n;
  if(iin && jin)
  {
    out = index(face, uint32_t(i), uint32_t(j));
    return true;
  }
  if(!iin && !jin)
    return false;
  int side = 0, depth = 0, along = 0;
  if(!iin)
  {
    side  = i < 0 ? 0 : 1;
    depth = i < 0 ? -1 - i : i - n;
    along = j;
  }
  else
  {
    side  = j < 0 ? 2 : 3;
    depth = j < 0 ? -1 - j : j - n;
    along = i;
  }
  const Edge&    e      = m_edges[(size_t(face) * 4 + size_t(side)) * m_n + size_t(along)];
  const uint32_t across = uint32_t(e.high ? n - 1 - depth : depth);
  out                   = e.acrossU ? index(e.face, across, e.along) : index(e.face, e.along, across);
  return true;
}

void CubeGrid::around(uint32_t c, uint32_t out[8]) const
{
  const uint32_t nn   = m_n * m_n;
  const uint32_t face = c / nn, rem = c - face * nn;
  const int      j = int(rem / m_n), i = int(rem - uint32_t(j) * m_n);
  if(i > 0 && j > 0 && i < int(m_n) - 1 && j < int(m_n) - 1)
  {
    for(int k = 0; k < 8; k++)
      out[k] = uint32_t(int(c) + kDi[k] + kDj[k] * int(m_n));
    return;
  }
  for(int k = 0; k < 8; k++)
    if(!cell(uint8_t(face), i + kDi[k], j + kDj[k], out[k]))
      out[k] = kNone;
}

glm::dvec3 CubeGrid::centre(uint32_t c) const
{
  const uint32_t nn   = m_n * m_n;
  const uint32_t face = c / nn, rem = c - face * nn;
  const uint32_t j = rem / m_n, i = rem - j * m_n;
  return direction(uint8_t(face), (double(i) + 0.5) * 2.0 / double(m_n) - 1.0, (double(j) + 0.5) * 2.0 / double(m_n) - 1.0);
}

MapPoint CubeGrid::at(const glm::dvec3& d) const
{
  const FacePoint p = locate(d);
  return {p.face, (p.u + 1.0) * 0.5 * double(m_n) - 0.5, (p.v + 1.0) * 0.5 * double(m_n) - 0.5};
}

uint32_t mapCells(double radius)
{
  const double wanted = 0.5 * kPi * radius / 60.0;
  uint32_t     n      = 64;
  while(n < 1024 && double(n) * 2.0 <= wanted)
    n *= 2;
  return n;
}

// ---- the bake ----------------------------------------------------------------------------------------

std::shared_ptr<const BaseMap> bake(const World& w, uint32_t asked, const std::atomic<bool>* stop, bool floors)
{
  using fv::jobs::parallelFor;
  // a power of two, so the rain's grid and the coarser levels divide it
  uint32_t n = 16;
  while(n < asked)
    n *= 2;
  const uint32_t rainN = std::min(n, std::clamp(n / 4U, 32U, 128U));
  auto           out   = std::make_shared<BaseMap>(n, rainN);
  BaseMap&       m     = *out;
  const CubeGrid& g    = m.grid;
  const uint32_t count = g.count(), nn = n * n;
  const double   R     = w.spec.radius;
  const double   cell  = 0.5 * kPi * R / double(n);
  const double   relief = std::max(w.spec.relief, 0.25);
  const double   heights = kHeights * R * relief;  // metres: what a height is measured by
  m.cell               = cell;
  const auto stopped   = [&] { return stop && stop->load(std::memory_order_relaxed); };
  constexpr uint32_t kNone = CubeGrid::kNone;

  // A cell's size by where on its face it is (the faces are alike): metres to its neighbour along i
  // and along j, and its area
  std::vector<float> stepI(nn), stepJ(nn), area(nn);
  for(uint32_t j = 0; j < n; j++)
    for(uint32_t i = 0; i < n; i++)
    {
      const double u = (double(i) + 0.5) * 2.0 / double(n) - 1.0, v = (double(j) + 0.5) * 2.0 / double(n) - 1.0, h = 1.0 / double(n);
      const glm::dvec3 di = direction(0, u + h, v) - direction(0, u - h, v), dj = direction(0, u, v + h) - direction(0, u, v - h);
      stepI[j * n + i]    = float(R * glm::length(di));
      stepJ[j * n + i]    = float(R * glm::length(dj));
      area[j * n + i]     = float(R * R * glm::length(glm::cross(di, dj)));
    }
  const auto stepTo = [&](uint32_t c, int k) {
    const uint32_t ij = c % nn;
    const float    a = float(kDi[k]) * stepI[ij], b = float(kDj[k]) * stepJ[ij];
    return std::sqrt(a * a + b * b);
  };

  // ---- the land as the stages have it ----
  std::vector<float> first(count);
  parallelFor(count, 2048, [&](size_t begin, size_t end) {
    for(size_t c = begin; c < end; c++)
      first[c] = float(sample(w, g.centre(uint32_t(c)), cell).elevation);
  });
  if(stopped())
    return nullptr;

  // ---- the rain ----
  const CubeGrid& rg     = m.rainGrid;
  const uint32_t  rcount = rg.count(), rstride = rg.stride(), block = n / rainN;
  std::vector<float> moist(rcount);
  {
    const auto selfOf = [&](uint32_t c) {
      const uint32_t face = c / (rainN * rainN), rem = c % (rainN * rainN);
      return uint32_t(rg.padded(uint8_t(face), int(rem % rainN), int(rem / rainN)));
    };
    // a cell's height (the sea's bed counts as its surface) and how much of it is sea
    std::vector<float> high(rcount), sea(rcount);
    parallelFor(rcount, 1024, [&](size_t begin, size_t end) {
      for(size_t c = begin; c < end; c++)
      {
        const uint32_t face = uint32_t(c) / (rainN * rainN), rem = uint32_t(c) % (rainN * rainN);
        const uint32_t ci = rem % rainN, cj = rem / rainN;
        float          sum = 0.0F, wet = 0.0F;
        for(uint32_t b = 0; b < block; b++)
          for(uint32_t a = 0; a < block; a++)
          {
            const float h = first[g.index(uint8_t(face), ci * block + a, cj * block + b)];
            sum += std::max(h, 0.0F);
            wet += h <= 0.0F ? 1.0F : 0.0F;
          }
        high[c] = sum / float(block * block);
        sea[c]  = wet / float(block * block);
      }
    });
    std::vector<float> padded, next(rcount);
    const auto         blur = [&](std::vector<float>& field) {
      rg.pad(field, padded);
      for(uint32_t c = 0; c < rcount; c++)
      {
        const uint32_t self = selfOf(c);
        float          sum  = 0.0F;
        for(int b = -1; b <= 1; b++)
          for(int a = -1; a <= 1; a++)
            sum += padded[size_t(int(self) + a + b * int(rstride))];
        next[c] = sum / 9.0F;
      }
      field.swap(next);
    };
    // only the broad relief lifts the wind
    for(int pass = 0; pass < 3; pass++)
      blur(high);
    rg.pad(high, padded);
    // where each cell's air comes from, and how warm it is
    const double       step = 0.5 * kPi / double(rainN);
    std::vector<Tap>   up(rcount);
    std::vector<float> lift(rcount, 0.0F), warm(rcount), gone(rcount), before(rcount);
    parallelFor(rcount, 1024, [&](size_t begin, size_t end) {
      for(size_t c = begin; c < end; c++)
      {
        const glm::dvec3 d    = rg.centre(uint32_t(c));
        const glm::dvec3 blow = wind(d);
        up[c]                 = tapOf(rg, rg.at(glm::normalize(d - blow * step)));
        before[c]             = read(padded, up[c], rstride);  // the ground it comes over
        warm[c]               = float(std::clamp(1.05 - 0.85 * std::pow(std::abs(d.y), 1.5), 0.15, 1.0));
        // how far the air goes in a step, of a cell: it rains out by the land it crosses, so little
        // in the calm belts between the winds (which the sea's damp air seeps into)
        gone[c] = float(std::max(glm::length(blow), 0.2));
      }
    });
    const float lose = float(step / kRainReach), fill = float(step / kRainFill), lifted = float(kRainLift / heights);
    const float settle = float(heights * step / kRainSettle);
    // `q`: the water the air holds, 0 to 1. `over`: the level it has been lifted to on its way.
    std::vector<float> q(rcount, 0.0F), qn(rcount), over(rcount, 0.0F), overn(rcount), paddedOver;
    const uint32_t     steps = rainN * 5 / 2;
    for(uint32_t s = 0; s < steps; s++)
    {
      if(stopped())
        return nullptr;
      rg.pad(q, padded);
      rg.pad(over, paddedOver);
      parallelFor(rcount, 4096, [&](size_t begin, size_t end) {
        for(size_t c = begin; c < end; c++)
        {
          const uint32_t self = selfOf(uint32_t(c));
          // carried by the wind, and spread a little
          float v = 0.9F * read(padded, up[c], rstride)
                    + 0.025F * (padded[self - 1] + padded[self + 1] + padded[self - rstride] + padded[self + rstride]);
          // lifted by ground higher than what it comes over and than what it was last lifted to
          const float was = std::max(read(paddedOver, up[c], rstride) - settle, before[c]);
          lift[c]         = std::max(high[c] - was, 0.0F);
          overn[c]        = std::max(high[c], was);
          if(sea[c] > 0.5F)
            v += fill * warm[c] * (1.0F - v);
          else
            v -= v * std::min(lose * gone[c] + lifted * lift[c], 0.6F);
          qn[c] = v;
        }
      });
      q.swap(qn);
      over.swap(overn);
    }
    for(uint32_t c = 0; c < rcount; c++)
    {
      const double slope = double(lift[c]) / (step * R);
      moist[c]           = float(std::clamp(kMoistFull * std::pow(double(q[c]) * (1.0 + (sea[c] > 0.5F ? 0.0 : kMoistLift * slope)), kMoistPower), 0.0, 1.0));
    }
    for(int pass = 0; pass < 2; pass++)
      blur(moist);
    rg.pad(moist, m.rain);
  }
  // what runs off each cell of the land: its area by how wet it is
  std::vector<float> runoff(count);
  parallelFor(count, 4096, [&](size_t begin, size_t end) {
    for(size_t c = begin; c < end; c++)
      runoff[c] = area[c % nn] * float(kRunoffDry + double(read(m.rain, tapOf(rg, rg.at(g.centre(uint32_t(c)))), rstride)));
  });
  if(stopped())
    return nullptr;

  // ---- the hollows, filled ----
  // `surface` is what the water runs over: the ground, a hollow's fill, a lake's top. A fill is level,
  // so its water is led out the way the flood came in (`via`), and the cells are kept in the order
  // the flood reached them (`reached`): each after the one it came by.
  std::vector<uint8_t>  kind(count, eMapLand);
  std::vector<float>    surface(first), level(count, 0.0F);
  std::vector<uint32_t> via(count, kNone), reached;
  std::vector<float>    viaStep(count, 1.0F);
  for(uint32_t c = 0; c < count; c++)
    if(first[c] <= 0.0F)
      kind[c] = eMapSea;
  {
    std::vector<uint8_t> state(count, 0);  // 1 reached, 2 reached and raised
    using Entry = std::pair<float, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    uint32_t nb[8];
    // from the coast: the land beside the sea
    uint32_t lowest = kNone;
    for(uint32_t c = 0; c < count; c++)
    {
      if(kind[c] == eMapSea)
        continue;
      if(lowest == kNone || first[c] < first[lowest])
        lowest = c;
      g.around(c, nb);
      bool coast = false;
      for(int k = 0; k < 8 && !coast; k++)
        coast = nb[k] != kNone && kind[nb[k]] == eMapSea;
      if(coast)
      {
        state[c] = 1;
        level[c] = first[c];
        open.push({first[c], c});
      }
    }
    if(open.empty() && lowest != kNone)
    {
      // no sea: the land's lowest place is where its water ends
      state[lowest] = 1;
      level[lowest] = first[lowest];
      open.push({first[lowest], lowest});
    }
    while(!open.empty())
    {
      const auto [e, c] = open.top();
      open.pop();
      reached.push_back(c);
      g.around(c, nb);
      for(int k = 0; k < 8; k++)
      {
        const uint32_t x = nb[k];
        if(x == kNone || kind[x] == eMapSea || state[x])
          continue;
        if(first[x] <= e)
        {
          // in a hollow: filled to the level that reached it
          surface[x] = e;
          state[x]   = 2;
        }
        else
          state[x] = 1;
        level[x]   = surface[x];
        via[x]     = c;
        viaStep[x] = stepTo(c, k);
        open.push({surface[x], x});
      }
    }
    if(stopped())
      return nullptr;

    // which hollows are lakes: deep enough, large enough, wet enough
    const double         deep = std::max(kLakeDeep, kLakeDeepShare * R * relief);
    std::vector<uint8_t> seen(count, 0);
    std::vector<uint32_t> members;
    for(uint32_t c = 0; c < count; c++)
    {
      if(seen[c] || state[c] != 2)
        continue;
      members.assign(1, c);
      seen[c]        = 1;
      double deepest = 0.0, wet = 0.0;
      for(size_t at = 0; at < members.size(); at++)
      {
        const uint32_t x = members[at];
        deepest          = std::max(deepest, double(level[x] - first[x]));
        wet += double(runoff[x]) / double(area[x % nn]);
        g.around(x, nb);
        for(int k = 0; k < 8; k++)
        {
          const uint32_t y = nb[k];
          if(y == kNone || seen[y] || state[y] != 2 || level[y] != level[c])
            continue;
          seen[y] = 1;
          members.push_back(y);
        }
      }
      if(deepest < deep || members.size() < kLakeCells || wet / double(members.size()) < kLakeWet)
        continue;
      // (no open water where it freezes: such a hollow is ground, under the snow)
      if(temperatureAt(w, g.centre(c), double(level[c])) < kLakeCold)
        continue;
      const float water = level[c] - float(std::clamp(kLakeUnder * deepest, kLakeUnderLeast, kLakeUnderMost));
      uint32_t    under = 0;
      for(const uint32_t x : members)
        under += first[x] < water - float(kLakeShallow);
      if(under < kLakeWetCells)
        continue;
      // the hollow is the lake's: its cells under the water, and the dry ground of it round them.
      // (`surface` stays the brim's level for all of it: the water's way out; `level` is the water's.)
      for(const uint32_t x : members)
      {
        kind[x]  = first[x] < water - float(kLakeShallow) ? eMapLake : eMapBasin;
        level[x] = water;
      }
      m.lakes++;
    }
  }
  // a lake's hollow keeps its rim: not worn under its brim; where it spills, down to it and no further
  std::vector<float> held(count, 0.0F);
  {
    uint32_t nb[8];
    for(uint32_t c = 0; c < count; c++)
    {
      if(kind[c] != eMapLake && kind[c] != eMapBasin)
        continue;
      g.around(c, nb);
      for(int k = 0; k < 8; k++)
      {
        const uint32_t x = nb[k];
        if(x == kNone || kind[x] != eMapLand)
          continue;
        held[x] = std::max(held[x], surface[x] <= surface[c] ? surface[c] : surface[c] + 0.3F);
      }
    }
  }

  // ---- the wear ----
  std::vector<uint32_t> down(count, kNone);
  std::vector<float>    downStep(count, 1.0F), rise(count, 0.0F);
  std::vector<double>   flow(count, 0.0);
  std::vector<uint64_t> order, scratch;
  order.reserve(count);
  const std::vector<float> was(surface);  // the land before the wear: it rises back to this and no higher
  for(uint32_t c = 0; c < count; c++)
    if(kind[c] == eMapLand)
    {
      rise[c] = float(kWearSteady * R * relief * kWear * std::clamp(double(first[c]) / heights, 0.0, 1.5));
      held[c] = std::max(held[c], first[c] * float(mix(kWearKeepLow, kWearKeepHigh, smooth(kWearKeepFrom * heights, kWearKeepTo * heights, double(first[c])))));
    }
  // where each cell's water runs: its steepest way down
  const auto route = [&] {
    parallelFor(count, 4096, [&](size_t begin, size_t end) {
      uint32_t nb[8];
      for(size_t c = begin; c < end; c++)
      {
        down[c] = kNone;
        if(kind[c] == eMapSea)
          continue;
        g.around(uint32_t(c), nb);
        float steepest = 0.0F;
        for(int k = 0; k < 8; k++)
        {
          const uint32_t x = nb[k];
          if(x == kNone)
            continue;
          const float step  = stepTo(uint32_t(c), k);
          const float slope = (surface[c] - (kind[x] == eMapSea ? 0.0F : surface[x])) / step;
          if(slope > steepest)
          {
            steepest    = slope;
            down[c]     = x;
            downStep[c] = step;
          }
        }
        // on a level fill, a lake's top: out the way the flood came in
        if(down[c] == kNone && via[c] != kNone && surface[via[c]] <= surface[c])
        {
          down[c]     = via[c];
          downStep[c] = viaStep[c];
        }
      }
    });
    // from the lowest up: a cell comes after the one it runs to. (Among cells as high as each other
    // the sort keeps the flood's order, which is that.)
    order.clear();
    for(const uint32_t c : reached)
      order.push_back((uint64_t(bitsOf(std::max(surface[c], 0.0F))) << 32) | c);
    sortByKey(order, scratch);
    // what runs through each: its own and all that runs to it
    for(const uint64_t item : order)
      flow[uint32_t(item)] = double(runoff[uint32_t(item)]);
    for(size_t at = order.size(); at-- > 0;)
    {
      const uint32_t c = uint32_t(order[at]);
      if(down[c] != kNone && kind[down[c]] != eMapSea)
        flow[down[c]] += flow[c];
    }
  };
  if(w.spec.relief > 0.0)
    for(uint32_t round = 0; round < kWearRounds; round++)
    {
      if(stopped())
        return nullptr;
      route();
      for(const uint64_t item : order)
      {
        const uint32_t c = uint32_t(item);
        if(kind[c] != eMapLand || down[c] == kNone)
          continue;
        const float to   = kind[down[c]] == eMapSea ? 0.0F : surface[down[c]];
        const float step = downStep[c];
        const float f    = float(kWear * std::sqrt(flow[c])) / step;
        float       h    = (surface[c] + rise[c] + f * to) / (1.0F + f);
        h                = std::min({h, to + float(kTalus) * step, was[c]});
        h                = std::max({h, to + float(kFillSlope) * step, held[c]});
        surface[c]       = h;
      }
    }
  route();

  // ---- the valleys' floors ----
  // What a river carries down it lays in its valley where it runs slower, and a valley's floor is
  // level across: the wear cuts a V, and a river of any size runs in a V of its own. Not simulated:
  // each river's floor stands over its bed by a depth that grows with its water (kFloor*), less where
  // it runs steep, never under the floor below it, and the ground that drains to it is filled to that
  // floor where it lies under it. So a valley is flat across as far as it is lower than its floor, and
  // its sides are as the wear left them. A river by the sea or a lake comes down to it: its floor is no
  // deeper than kFloorOver of its height over the water its own ends in (filled where it was largest,
  // the land by the sea stood up out of it in a wall). (Laying what the wear takes off, after Yuan et al. 2019, was
  // tried: in this bake, whose rise holds the land up to the stages' height, it lifted whole valleys,
  // still Vs, and filled the smaller ones to their ridges.)
  if(floors && w.spec.relief > 0.0)
  {
    const double floorFrom = kFloorFrom * std::max(kRiverFrom * R * kRiverFrom * R, kRiverFromCells * cell * cell);
    std::vector<float>    floorOf(count, 0.0F);  // a channel's floor; what drains to it, the floor it is filled to
    std::vector<uint32_t> channel(count, kNone);
    std::vector<float>    ends(count, 0.0F);  // the level of the water a cell's own ends in: the sea's, a lake's
    // from the sea up: a cell after the one its water runs to
    for(const uint64_t item : order)
    {
      const uint32_t c = uint32_t(item);
      if(kind[c] != eMapLand || down[c] == kNone)
        continue;
      const uint32_t r    = down[c];
      const bool     onTo = kind[r] == eMapLand && channel[r] != kNone;
      ends[c]             = kind[r] == eMapSea ? 0.0F : kind[r] == eMapLand ? ends[r] : level[r];
      if(flow[c] >= floorFrom)
      {
        const float  to    = kind[r] == eMapSea ? 0.0F : surface[r];
        const double steep = std::max(double(surface[c] - to), 0.0) / double(downStep[c]);
        const double depth = std::min(kFloorMost * heights * smooth(0.0, 1.0, std::log2(flow[c] / floorFrom) / kFloorDoublings)
                                          * (1.0 - smooth(kFloorSteep, 2.0 * kFloorSteep, steep)),
                                      kFloorOver * std::max(double(surface[c] - ends[c]), 0.0));
        double floor_ = double(surface[c]) + depth;
        // on down the valley: no lower than the floor it runs on to
        if(onTo && channel[r] == r)
          floor_ = std::max(floor_, double(floorOf[r]) + kFillSlope * double(downStep[c]));
        floorOf[c] = float(floor_);
        channel[c] = c;
      }
      else if(onTo)
      {
        // what drains to a river is filled to its floor, a little over it the further it is
        channel[c] = channel[r];
        floorOf[c] = floorOf[r] + float(kFillSlope) * downStep[c];
      }
    }
    // and nothing a lake's hollow spills to over its brim, nor the river from there on over that: the
    // lake would have no way out
    std::vector<float> under(count, std::numeric_limits<float>::max());
    for(uint32_t x = 0; x < count; x++)
      if((kind[x] == eMapLake || kind[x] == eMapBasin) && down[x] != kNone && kind[down[x]] == eMapLand)
        under[down[x]] = std::min(under[down[x]], surface[x] - float(kFillSlope) * downStep[x]);
    for(size_t at = order.size(); at-- > 0;)
    {
      const uint32_t c = uint32_t(order[at]);
      if(kind[c] == eMapLand && down[c] != kNone && kind[down[c]] == eMapLand && under[c] < std::numeric_limits<float>::max())
        under[down[c]] = std::min(under[down[c]], under[c] - float(kFillSlope) * downStep[c]);
    }
    for(uint32_t c = 0; c < count; c++)
      if(channel[c] != kNone)
        surface[c] = std::max(surface[c], std::min(floorOf[c], under[c]));
  }
  route();
  if(stopped())
    return nullptr;

  // ---- the rivers ----
  // In a lake's hollow the water's way on the map is over the brim-full fill, from wherever it comes
  // in to where the hollow spills. What is drawn is what a lake under its brim does: a river that
  // comes in runs down the hollow's own ground into the lake, and the river that leaves starts at
  // the lake and comes up the same kind of way to the brim, which it cuts down to the lake's level.
  // `toLake`: each dry cell of a hollow's way down to its lake, by a flood from the lake's cells up
  // over the hollow's bed.
  std::vector<uint32_t> toLake(count, kNone);
  {
    using Entry = std::pair<float, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    std::vector<uint8_t> seen(count, 0);
    uint32_t             nb[8];
    for(uint32_t c = 0; c < count; c++)
      if(kind[c] == eMapLake)
      {
        seen[c] = 1;
        open.push({level[c], c});
      }
    while(!open.empty())
    {
      const auto [e, c] = open.top();
      open.pop();
      g.around(c, nb);
      for(int k = 0; k < 8; k++)
      {
        const uint32_t x = nb[k];
        if(x == kNone || kind[x] != eMapBasin || seen[x])
          continue;
        seen[x]   = 1;
        toLake[x] = c;
        open.push({std::max(first[x], e), x});
      }
    }
  }
  const double riverFrom = std::max(kRiverFrom * R * kRiverFrom * R, kRiverFromCells * cell * cell);
  const auto   isRiver   = [&](uint32_t c) { return kind[c] == eMapLand && flow[c] >= riverFrom; };
  // a river's point in its cell: off the middle, by the cell
  const uint32_t strayBy = uint32_t(w.spec.seed) * 2246822519U + 374761393U;
  const auto     pointOf = [&](uint32_t c) {
    uint32_t h = (c + strayBy) * 2654435761U;
    h ^= h >> 15;
    h *= 2246822519U;
    h ^= h >> 13;
    const uint32_t face = c / nn, rem = c % nn;
    const double   si = (double(h & 0xFFFFU) / 65535.0 - 0.5) * 2.0 * kRiverStray, sj = (double(h >> 16U) / 65535.0 - 0.5) * 2.0 * kRiverStray;
    return direction(uint8_t(face), (double(rem % n) + 0.5 + si) * 2.0 / double(n) - 1.0, (double(rem / n) + 0.5 + sj) * 2.0 / double(n) - 1.0);
  };
  // the donor each river cell follows upstream: the one with the most water, a river's or a lake's
  std::vector<uint32_t> main(count, kNone);
  {
    std::vector<double> most(count, 0.0);
    for(const uint64_t item : order)
    {
      const uint32_t c = uint32_t(item), to = down[c];
      if(to == kNone || kind[to] == eMapSea || !isRiver(to) || !(isRiver(c) || kind[c] == eMapLake || kind[c] == eMapBasin))
        continue;
      if(flow[c] > most[to])
      {
        most[to] = flow[c];
        main[to] = c;
      }
    }
  }
  std::vector<int32_t> pieceOf(count, -1);
  std::vector<float>   water(count, 0.0F), half(count, 0.0F);
  const double         widest = kRiverWide * cell;
  // What a river cell's water runs into: the sea (level 0), a lake, or the next cell of the river
  struct Into
  {
    bool  sea = false, lake = false;
    float level = 0.0F;
  };
  const auto intoOf = [&](uint32_t c) {
    const uint32_t to = down[c];
    Into           into;
    into.sea   = to == kNone || kind[to] == eMapSea;
    into.lake  = !into.sea && (kind[to] == eMapLake || kind[to] == eMapBasin);  // (a lake, or its hollow)
    into.level = into.sea ? 0.0F : into.lake ? level[to] : water[to];
    return into;
  };
  for(const uint64_t item : order)  // from the sea up: a cell's level is no lower than the next one's
  {
    const uint32_t c = uint32_t(item);
    if(!isRiver(c))
      continue;
    half[c] = float(0.5 * std::min(kRiverNarrow * std::pow(flow[c] / riverFrom, kRiverGrow), widest));
    // its surface a little under the valley's floor, never rising downstream; in a lake's hollow
    // and out of it, the lake's (the river cuts the brim down to it)
    float own = surface[c] - std::min(1.0F, 0.5F * half[c]);
    if(const uint32_t d = main[c]; d != kNone && (kind[d] == eMapLake || kind[d] == eMapBasin))
      own = std::min(own, level[d]);
    // (and at the sea's level it is the sea's: a river a hand over it would draw as a step in the water)
    own      = own * float(smooth(0.0, kRiverLow, double(own)));
    water[c] = std::max(own < 0.3F ? 0.0F : own, intoOf(c).level);
  }
  for(const uint64_t item : order)  // from the sea up: a piece's downstream one is made before it
  {
    const uint32_t c = uint32_t(item);
    if(!isRiver(c))
      continue;
    const uint32_t to     = down[c];
    const Into     into   = intoOf(c);
    const bool     toSea  = into.sea, toLake = into.lake;
    const float    below  = into.level;

    RiverPiece p;
    p.b        = pointOf(c);
    p.level[1] = water[c];
    p.half[1]  = half[c];
    if(const uint32_t d = main[c]; d != kNone)
    {
      // (out of a lake: from in its water, or from the river that comes up its hollow's ground)
      const bool lake = kind[d] == eMapLake, hollow = lake || kind[d] == eMapBasin;
      p.a             = lake ? g.centre(d) : glm::normalize(p.b + pointOf(d));
      p.level[0]      = hollow ? std::max(level[d], water[c]) : 0.5F * (water[c] + std::max(water[d], water[c]));
      p.half[0]       = hollow ? half[c] : 0.5F * (half[c] + half[d]);
    }
    else
    {
      // a river's head: it starts thin
      p.a        = p.b;
      p.level[0] = water[c];
      p.half[0]  = 0.3F * half[c];
    }
    p.c        = to == kNone ? p.b : toSea || toLake ? g.centre(to) : pointOf(to);
    p.level[2] = below;
    p.half[2]  = half[c];
    if(!toSea && !toLake)
    {
      if(main[to] == c)
      {
        p.c        = glm::normalize(p.b + p.c);
        p.level[2] = 0.5F * (water[c] + below);
        p.half[2]  = 0.5F * (half[c] + half[to]);
      }
      else if(pieceOf[to] >= 0)
      {
        // a tributary: into the river's own curve, at its level there
        const RiverPiece& joined = m.pieces[size_t(pieceOf[to])];
        p.c                      = bent(joined.a, joined.b, joined.c, 0.5);
        p.level[2]               = std::min(float(bent(joined.level, 0.5)), water[c]);
      }
    }
    // (a level is written as a float after sums of floats: never up along the way)
    p.level[0] = std::max(p.level[0], p.level[1]);
    p.level[2] = std::min(p.level[2], p.level[1]);
    pieceOf[c] = int32_t(m.pieces.size());
    m.pieces.push_back(p);
  }
  // Between a river and a lake, across the dry ground of the lake's hollow: from the hollow's cell
  // `x` the river touches, down the hollow's ground to the lake, at the lake's level (it cuts its
  // way), or as far as a river already there. For a river that comes in (`x` is where it runs to),
  // and for the one that leaves (`x` is where its water comes from).
  const auto cross = [&](uint32_t river, uint32_t x) {
    uint32_t last = river;
    while(x != kNone && kind[x] == eMapBasin && pieceOf[x] < 0)
    {
      const uint32_t next = toLake[x];
      RiverPiece     p;
      p.b = pointOf(x);
      p.a = glm::normalize(p.b + pointOf(last));
      p.c = next == kNone ? p.b : kind[next] == eMapLake ? g.centre(next) : glm::normalize(p.b + pointOf(next));
      for(int k = 0; k < 3; k++)
      {
        p.level[k] = level[x];
        p.half[k]  = half[river];
      }
      pieceOf[x] = int32_t(m.pieces.size());
      m.pieces.push_back(p);
      last = x;
      x    = next;
    }
  };
  for(const uint64_t item : order)
  {
    const uint32_t c = uint32_t(item);
    if(!isRiver(c))
      continue;
    cross(c, down[c]);
    if(main[c] != kNone)
      cross(c, main[c]);
  }
  // near a corner of the cube a cell's window is not whole: the pieces there are listed
  for(uint32_t c = 0; c < count; c++)
    if(pieceOf[c] >= 0)
    {
      const uint32_t rem = c % nn;
      if(fromCorner(int(n), int(rem % n), int(rem / n)) < 2 * kCornerLevel + kWindow)
        m.corner[cornerOf(g.centre(c))].push_back(uint32_t(pieceOf[c]));
    }
  g.pad(pieceOf, m.river, false, int32_t(-1));
  m.reach.assign(count, 0);
  parallelFor(count, 4096, [&](size_t begin, size_t end) {
    const int stride = int(g.stride());
    for(size_t c = begin; c < end; c++)
    {
      const uint32_t face = uint32_t(c) / nn, rem = uint32_t(c) % nn;
      const int      i = int(rem % n), j = int(rem / n);
      if(fromCorner(int(n), i, j) < kWindow)
      {
        m.reach[c] = m.corner[cornerOf(g.centre(uint32_t(c)))].empty() ? 0 : 1;
        continue;
      }
      const int32_t* self = m.river.data() + g.padded(uint8_t(face), i, j);
      bool           any  = false;
      for(int b = -kWindow; b <= kWindow && !any; b++)
        for(int a = -kWindow; a <= kWindow && !any; a++)
          any = self[a + b * stride] >= 0;
      m.reach[c] = any ? 1 : 0;
    }
  });

  // ---- what is kept ----
  std::vector<float> lake(count, kDryCell), added(count, 0.0F);
  m.height.resize(count);
  m.flow.resize(count);
  for(uint32_t c = 0; c < count; c++)
  {
    if(kind[c] == eMapLake)
      lake[c] = level[c];
    m.height[c] = kind[c] == eMapLand ? surface[c] : first[c];
    m.flow[c]   = float(flow[c]);
    added[c]    = m.height[c] - first[c];
  }
  g.pad(lake, m.lake, false, kDryCell);
  // level near the cube's corners: three faces meet there, and their gutters cannot all agree
  {
    double sum[8] = {}, cells[8] = {};
    const auto each = [&](const auto& fn) {
      for(uint8_t face = 0; face < 6; face++)
        for(int j = 0; j < int(n); j++)
          for(int i = 0; i < int(n); i++)
            if(const int away = fromCorner(int(n), i, j); away < 2 * kCornerLevel)
            {
              const uint32_t c = g.index(face, uint32_t(i), uint32_t(j));
              // (which corner: the face's, by which quarter of it the cell is in)
              const uint32_t corner = cornerOf(direction(face, i < int(n) / 2 ? -1.0 : 1.0, j < int(n) / 2 ? -1.0 : 1.0));
              fn(c, corner, away);
            }
    };
    each([&](uint32_t c, uint32_t corner, int away) {
      if(away < kCornerLevel)
      {
        sum[corner] += double(added[c]);
        cells[corner] += 1.0;
      }
    });
    each([&](uint32_t c, uint32_t corner, int away) {
      const double mean = cells[corner] > 0.0 ? sum[corner] / cells[corner] : 0.0;
      added[c]          = float(mix(mean, double(added[c]), smooth(double(kCornerLevel) - 0.5, double(2 * kCornerLevel) - 0.5, double(away))));
    });
  }
  // and again at half the cells, and half again, for what draws coarsely
  {
    std::vector<float> at = std::move(added);
    for(uint32_t size = n;; size /= 2)
    {
      BaseMap::Level lvl{CubeGrid(size), {}};
      lvl.grid.pad(at, lvl.added);
      m.levels.push_back(std::move(lvl));
      if(size / 2 < 16)
        break;
      const uint32_t     halfN = size / 2;
      std::vector<float> coarse(size_t(6) * halfN * halfN);
      for(uint32_t face = 0; face < 6; face++)
        for(uint32_t j = 0; j < halfN; j++)
          for(uint32_t i = 0; i < halfN; i++)
          {
            const size_t a = (size_t(face) * size + 2 * j) * size + 2 * i;
            coarse[(size_t(face) * halfN + j) * halfN + i] = 0.25F * (at[a] + at[a + 1] + at[a + size] + at[a + size + 1]);
          }
      at.swap(coarse);
    }
  }
  // ---- the worn land's broad slope ----
  // At the corners of coarser grids, each from the plane that best fits the cells round it (the
  // nearer counting more), found in space and not on a face's own grid: a face's lines turn where
  // they cross an edge, and a slope read off them would turn with them.
  {
    std::vector<float> at   = m.height;
    uint32_t           size = n;
    const auto         halve = [&] {
      const uint32_t     halfN = size / 2;
      std::vector<float> coarse(size_t(6) * halfN * halfN);
      for(uint32_t face = 0; face < 6; face++)
        for(uint32_t j = 0; j < halfN; j++)
          for(uint32_t i = 0; i < halfN; i++)
          {
            const size_t a = (size_t(face) * size + 2 * j) * size + 2 * i;
            coarse[(size_t(face) * halfN + j) * halfN + i] = 0.25F * (at[a] + at[a + 1] + at[a + size] + at[a + size + 1]);
          }
      at.swap(coarse);
      size = halfN;
    };
    while(size > slopeCellsFor(n))
      halve();
    for(;; halve())
    {
      const uint32_t sn = size;
      const CubeGrid sg(sn);
      const uint32_t side = sn + 1, perFace = side * side;
      const double   reach = 2.5 * 0.5 * kPi * R / double(sn);  // metres: two and a half cells
      BaseMap::Slope level{sn, std::vector<glm::vec3>(size_t(6) * perFace, glm::vec3(0.0F))};
      parallelFor(level.at.size(), 2048, [&](size_t begin, size_t end) {
        for(size_t v = begin; v < end; v++)
        {
          const uint8_t    face = uint8_t(v / perFace);
          const int        i = int((v % perFace) % side), j = int((v % perFace) / side);
          const glm::dvec3 d = direction(face, 2.0 * double(i) / double(sn) - 1.0, 2.0 * double(j) / double(sn) - 1.0);
          const glm::dvec3 a = glm::normalize(glm::cross(std::abs(d.y) < 0.9 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), d));
          const glm::dvec3 b = glm::cross(d, a);
          double           sw = 0.0, sx = 0.0, sy = 0.0, sh = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0, sxh = 0.0, syh = 0.0;
          for(int q = j - 2; q <= j + 1; q++)
            for(int p = i - 2; p <= i + 1; p++)
            {
              uint32_t c = 0;
              if(!sg.cell(face, p, q, c))
                continue;
              const glm::dvec3 to  = sg.centre(c) - d;
              const double     x = R * glm::dot(to, a), y = R * glm::dot(to, b);
              const double     far = (x * x + y * y) / (reach * reach);
              if(far >= 1.0)
                continue;
              const double wgt = (1.0 - far) * (1.0 - far), h = double(at[c]);
              sw += wgt;
              sx += wgt * x;
              sy += wgt * y;
              sh += wgt * h;
              sxx += wgt * x * x;
              sxy += wgt * x * y;
              syy += wgt * y * y;
              sxh += wgt * x * h;
              syh += wgt * y * h;
            }
          if(sw <= 0.0)
            continue;
          const double mx = sx / sw, my = sy / sw, mh = sh / sw;
          const double cxx = sxx / sw - mx * mx, cxy = sxy / sw - mx * my, cyy = syy / sw - my * my;
          const double cxh = sxh / sw - mx * mh, cyh = syh / sw - my * mh;
          const double det = cxx * cyy - cxy * cxy;
          if(det <= 1e-9 * (cxx + cyy) * (cxx + cyy))
            continue;
          level.at[v] = glm::vec3(a * ((cxh * cyy - cyh * cxy) / det) + b * ((cyh * cxx - cxh * cxy) / det));
        }
      });
      // a corner on a face's edge is two faces' (three's, at the cube's own): one slope for all of them
      struct Shared
      {
        glm::vec3 sum{0.0F};
        float     count = 0.0F;
      };
      std::unordered_map<uint64_t, Shared> shared;
      const auto                           each = [&](const auto& fn) {
        for(uint8_t face = 0; face < 6; face++)
          for(uint32_t j = 0; j < side; j++)
            for(uint32_t i = 0; i < side; i++)
              if(i == 0 || j == 0 || i == sn || j == sn)
              {
                const glm::dvec3 d = direction(face, 2.0 * double(i) / double(sn) - 1.0, 2.0 * double(j) / double(sn) - 1.0);
                const auto       part = [](double x) { return uint64_t(std::llround((x + 1.0) * 500000.0)) & 0x1FFFFFU; };
                fn(level.at[size_t(face) * perFace + size_t(j) * side + i], shared[part(d.x) | (part(d.y) << 21U) | (part(d.z) << 42U)]);
              }
      };
      each([](const glm::vec3& slope, Shared& s) {
        s.sum += slope;
        s.count += 1.0F;
      });
      each([](glm::vec3& slope, Shared& s) { slope = s.sum / s.count; });
      m.slopes.push_back(std::move(level));
      if(size / 2 < 16)
        break;
    }
  }
  m.down = std::move(down);
  m.kind = std::move(kind);
  return out;
}

// ---- reading it --------------------------------------------------------------------------------------

double mapAdded(const BaseMap& m, const glm::dvec3& d, double footprint)
{
  if(m.levels.empty())
    return 0.0;
  const double lod  = std::log2(std::max(footprint / m.cell, 1.0));
  const size_t top  = m.levels.size() - 1;
  const size_t l    = std::min(size_t(lod), top);
  const double fine = smoothRead(m.levels[l].grid, m.levels[l].added, d);
  if(l == top || lod - double(l) <= 0.0)
    return fine;
  return mix(fine, smoothRead(m.levels[l + 1].grid, m.levels[l + 1].added, d), lod - double(l));
}

float mapRain(const BaseMap& m, const glm::dvec3& d)
{
  return read(m.rain, tapOf(m.rainGrid, m.rainGrid.at(d)), m.rainGrid.stride());
}

uint32_t slopeCellsFor(uint32_t cells)
{
  while(cells >= 512)
    cells /= 2;
  return cells;
}

int slopeLevelFor(const BaseMap& m, double radius, double metres)
{
  if(m.slopes.empty())
    return 0;
  const double finest = 0.5 * kPi * radius / double(m.slopes.front().cells);
  return std::clamp(int(std::lround(std::log2(std::max(metres / finest, 1e-6)))), 0, int(m.slopes.size()) - 1);
}

glm::dvec3 mapSlope(const BaseMap& m, const FacePoint& p, const glm::dvec3& d, int level)
{
  if(m.slopes.empty())
    return glm::dvec3(0.0);
  const BaseMap::Slope& s = m.slopes[size_t(std::clamp(level, 0, int(m.slopes.size()) - 1))];
  const double          n = double(s.cells);
  const double          x = (p.u + 1.0) * 0.5 * n, y = (p.v + 1.0) * 0.5 * n;
  const double          i = std::clamp(std::floor(x), 0.0, n - 1.0), j = std::clamp(std::floor(y), 0.0, n - 1.0);
  const float           fx = float(std::clamp(x - i, 0.0, 1.0)), fy = float(std::clamp(y - j, 0.0, 1.0));
  const size_t          side = size_t(s.cells) + 1;
  const glm::vec3*      at   = s.at.data() + size_t(p.face) * side * side + size_t(j) * side + size_t(i);
  const glm::vec3       low = at[0] + (at[1] - at[0]) * fx, high = at[side] + (at[side + 1] - at[side]) * fx;
  // along the ground here: what four corners say leans a little off it
  const glm::dvec3 slope(low + (high - low) * fy);
  return slope - d * glm::dot(slope, d);
}

Inland mapInland(const BaseMap& m, double radius, const glm::dvec3& d)
{
  Inland          in;
  const CubeGrid& g = m.grid;
  const int       n = int(g.cells());
  const MapPoint  p = g.at(d);
  in.cell           = m.cell;
  // a lake: of the cells round the point, the share that are one's, read smoothly (as smoothRead
  // does a height): a half along a straight shore of cells, and its lines are curves
  {
    const double fi = std::floor(p.x), fj = std::floor(p.y);
    double       wx[4], wy[4];
    splineWeights(p.x - fi, wx);
    splineWeights(p.y - fj, wy);
    const size_t stride = g.stride();
    const size_t first  = g.padded(p.face, int(fi) - 1, int(fj) - 1);
    // Two lakes may lie within the cells round a point, one at a cliff's foot and one over it: the
    // lake is the one with the larger share there, at its own level (the higher of the two took the
    // lower one's water up a tier's height along its near shore)
    float  levels[3] = {};
    double shares[3] = {};
    int    lakes     = 0;
    for(int b = 0; b < 4; b++)
      for(int a = 0; a < 4; a++)
        if(const float level = m.lake[first + size_t(b) * stride + size_t(a)]; level > 0.5F * kDryCell)
        {
          int k = 0;
          while(k < lakes && levels[k] != level)
            k++;
          if(k == lakes)
          {
            if(lakes == 3)
              continue;  // (a fourth among sixteen cells: left out)
            levels[lakes++] = level;
          }
          shares[k] += wx[a] * wy[b];
        }
    for(int k = 0; k < lakes; k++)
      if(shares[k] > in.lake)
      {
        in.lake      = shares[k];
        in.lakeLevel = double(levels[k]);
      }
  }
  const int i = std::clamp(int(std::floor(p.x + 0.5)), 0, n - 1), j = std::clamp(int(std::floor(p.y + 0.5)), 0, n - 1);
  if(!m.reach[g.index(p.face, uint32_t(i), uint32_t(j))])
    return in;
  Nearest best;
  if(fromCorner(n, i, j) < kWindow)
  {
    for(const uint32_t piece : m.corner[cornerOf(d)])
      nearestOn(m.pieces[piece], d, radius, best);
  }
  else
  {
    const int      stride = int(g.stride());
    const int32_t* self   = m.river.data() + g.padded(p.face, i, j);
    for(int b = -kWindow; b <= kWindow; b++)
      for(int a = -kWindow; a <= kWindow; a++)
        if(const int32_t piece = self[a + b * stride]; piece >= 0)
          nearestOn(m.pieces[size_t(piece)], d, radius, best);
  }
  if(best.score == std::numeric_limits<double>::max())
    return in;
  in.bank = std::clamp(kBankHalves * best.half, kBankLeast, kBankMost * m.cell);
  if(best.score >= in.bank)
    return in;  // past its banks
  in.river      = true;
  in.riverLevel = best.level;
  in.riverHalf  = best.half;
  in.riverAway  = best.away;
  return in;
}

// How much of a river something `footprint` across draws as a bed with water in it: all of it at four
// to its width, none of it at two
static double riverDrawn(const Inland& in, double footprint)
{
  const double width = 2.0 * in.riverHalf;
  return 1.0 - smooth(0.25 * width, 0.5 * width, footprint);
}

double Inland::shore() const
{
  double away = 1.0e9;
  if(river)
    away = riverAway - riverHalf;
  if(lake > 0.0)
    away = std::min(away, (kLakeSheet + 0.1 - lake) * cell / 0.6);  // (its share falls by 0.6 a cell across a shore)
  return away;
}

double inlandSheet(const Inland& in, double footprint)
{
  double sheet = kDry;
  // a lake's top, level over its cells and a little past them, then down under the ground
  if(in.lake > 0.0)
    sheet = in.lakeLevel - 50.0 * smooth(kLakeSheet, kLakeSheet - 0.11, in.lake) - 3000.0 * smooth(kLakeSheet - 0.11, 0.02, in.lake);
  if(in.river)
  {
    // level to just past the water's edge, then down under the bank; and under the ground altogether
    // where the river is too narrow for what draws it
    const double past = in.riverAway - in.riverHalf;
    const double dive = smooth(0.4, std::max(0.3 * in.bank, 1.0), past);
    sheet             = std::max(sheet, in.riverLevel - 40.0 * (1.0 - riverDrawn(in, footprint)) - 60.0 * dive - 3000.0 * smooth(0.5 * in.bank, in.bank, past));
  }
  return sheet;
}

double inlandGround(const Inland& in, double natural, double footprint)
{
  double e = natural;
  // A lake's shore is the ground's own, where it comes up through the lake's level. Where it does not
  // (the ground's fine relief dips under a lake the map's cells hold in), a low bank stands in for
  // it just outside the lake's cells: over the level where the water's sheet ends, and eased away to
  // nothing on both sides, under the lake and out on the land.
  if(in.lake > 0.0)
  {
    // (out on the land it falls away faster and faster, to under any ground where the lake's share
    // ends: nothing steps there, however far under the lake a valley beside it lies)
    const double out  = 1.0 - std::clamp(in.lake / (kLakeSheet - 0.1), 0.0, 1.0);
    const double bank = in.lakeLevel + 0.6 - 3.6 * (1.0 - smooth(0.5, kLakeSheet + 0.05, in.lake)) - 60.0 * smooth(0.45, 0.6, in.lake)
                        - 2000.0 * out * out * out * out;
    e                 = std::max(e, bank);
  }
  if(in.river)
  {
    // (in a lake the river is the lake)
    const double drawn = riverDrawn(in, footprint) * (1.0 - smooth(0.3, 0.5, in.lake));
    if(drawn > 0.0)
    {
      const double width = 2.0 * in.riverHalf, past = in.riverAway - in.riverHalf;
      const double depth = std::min(0.5 + 0.12 * width, 3.5);
      double       cut;
      if(past <= 0.0)
      {
        // the bed, which only ever deepens the ground (a river that runs out into the sea, or
        // through a pool, has their bed)
        const double x = in.riverAway / std::max(in.riverHalf, 1e-6);
        cut            = std::min(in.riverLevel - depth * (1.0 - x * x), e);
      }
      else
      {
        // the bank: up 1 in 4 for half a metre and gently after, easing into the land. Where the
        // ground is under the river's level it is built up to hold the river in, but not out of
        // the sea: a river's mouth has no banks.
        double bank = in.riverLevel + std::min(0.25 * past, 0.5 + 0.03 * past);
        if(bank > e)
          bank = mix(e, bank, smooth(0.0, 1.0, e));
        cut = mix(bank, e, smooth(0.3 * in.bank, in.bank, past));
      }
      e = mix(e, cut, drawn);
    }
  }
  return e;
}

float inlandPainted(const Inland& in, double footprint)
{
  if(!in.river)
    return 0.0F;
  const double width = 2.0 * in.riverHalf;
  const double cover = std::clamp((in.riverHalf - in.riverAway) / std::max(footprint, 1e-6) + 0.5, 0.0, 1.0) * std::min(1.0, width / std::max(footprint, 1e-6));
  return float(cover * (1.0 - riverDrawn(in, footprint)));
}

}  // namespace fv::world
