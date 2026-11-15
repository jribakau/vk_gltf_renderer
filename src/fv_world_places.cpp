#include "fv_world_places.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>

#include <figure/jobs/pool.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace fv::world {

namespace {

constexpr double kPi = 3.14159265358979323846;

// The most places with a road a planet has, and the most single things: a large planet's are further
// apart, not more
constexpr uint32_t kMostJoined = 500;
constexpr uint32_t kMostAlone  = 4000;
constexpr uint32_t kMostAlong  = 6000;
// A road is not led up ground steeper than this from one cell of the map to the next, and pays for
// what it climbs by the square of it: the land stands in terraces whose edges are this steep
// between two cells, and a road crosses one where it is lowest, as a ramp cut into it (kRoadGrade,
// kRoadCut)
constexpr double kRoadSteepest = 0.45;
constexpr double kRoadEasy     = 0.06;
// It is cut into the land or banked over it by this much at the most, metres: past that it is as
// steep as the land
constexpr double kRoadCut = 6.0;
// A place that reaches further than this stands in a clearing: no tree of the land's on its floor
// (a hut or a ruin alone has the wood up to its walls)
constexpr double kClearedFrom = 15.0;
// No road is led this near where the figures stand, metres
constexpr double kFiguresOwn = 40.0;
// A route is given up after this many cells are looked at
constexpr size_t kRouteMost = 25000;
// How strongly a route is drawn toward its end: 1 would be the straight way's own length, which a
// way along a road already there undercuts; less looks further afield for one
constexpr double kRouteDrawn = 0.8;
// A stream no wider than this is forded; a wider one is not crossed (there are no bridges)
constexpr double kFordHalf = 5.0;
// The ground a road's level is read from: what is finer is cut through or filled. And how far apart
// along it, metres.
constexpr double kRoadFootprint = 12.0;
constexpr double kRoadStep      = 32.0;
// The steepest a road runs, rise over run
constexpr double kRoadGrade = 0.14;
// Where a road is cut into the land or banked over it, its sides are no steeper than this (run over
// rise): its verges are as much wider, up to kBankMost metres more
constexpr double kBankRun  = 2.0;
constexpr double kBankMost = 12.0;

uint32_t mix(uint32_t v)
{
  v = v * 747796405U + 2891336453U;
  const uint32_t w = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (w >> 22U) ^ w;
}

uint32_t hash(uint32_t a, uint32_t b, uint32_t c = 0, uint32_t d = 0)
{
  return mix(mix(mix(mix(a) + b) + c) + d);
}

double unit(uint32_t h)
{
  return double(h >> 8U) * (1.0 / 16777216.0);
}

double smooth(double a, double b, double x)
{
  const double t = std::clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

double mixed(double a, double b, double t)
{
  return a + (b - a) * t;
}

// `count` directions spread evenly over the sphere
glm::dvec3 spread(uint32_t i, uint32_t count)
{
  const double y   = 1.0 - 2.0 * (double(i) + 0.5) / double(count);
  const double r   = std::sqrt(std::max(1.0 - y * y, 0.0));
  const double phi = double(i) * 2.399963229728653;
  return {r * std::cos(phi), y, r * std::sin(phi)};
}

// Two directions square to `d` and to each other: d x a = b
void tangents(const glm::dvec3& d, glm::dvec3& a, glm::dvec3& b)
{
  a = glm::normalize(glm::cross(std::abs(d.y) < 0.9 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), d));
  b = glm::cross(d, a);
}

// A site's own x and z on the planet (its y is its direction), turned by its heading as a yaw about
// +y turns a model
void axesOf(const Site& s, glm::dvec3& x, glm::dvec3& z)
{
  glm::dvec3 a, b;
  tangents(s.direction, a, b);
  x = a * std::cos(s.heading) + b * std::sin(s.heading);
  z = glm::cross(x, s.direction);
}

// The planet's point over a point (x, z) of a site's floor
glm::dvec3 fromSite(const World& w, const Site& s, double x, double z)
{
  glm::dvec3 ax, az;
  axesOf(s, ax, az);
  return glm::normalize(s.direction * (w.spec.radius + s.elevation) + ax * x + az * z);
}

// A site's floor under a direction: how high the plane that touches the planet at the site is there
double floorAt(const World& w, const Site& s, const glm::dvec3& d)
{
  return (w.spec.radius + s.elevation) / std::max(glm::dot(d, s.direction), 0.5) - w.spec.radius;
}

uint32_t cellOf(const CubeGrid& g, const glm::dvec3& d)
{
  const MapPoint p = g.at(d);
  const int      n = int(g.cells());
  return g.index(p.face, uint32_t(std::clamp(int(std::floor(p.x + 0.5)), 0, n - 1)), uint32_t(std::clamp(int(std::floor(p.y + 0.5)), 0, n - 1)));
}

// Metres across a cell at a face's middle, where they are largest
double cellAcross(const CubeGrid& g, double radius)
{
  return 0.5 * kPi * radius / double(g.cells());
}

template <class T>
T bent(const T& a, const T& b, const T& c, double t)
{
  const double s = 1.0 - t;
  return a * (s * s) + b * (2.0 * s * t) + c * (t * t);
}

// A road piece's point nearest a direction: metres from its middle line, and its level there
struct Nearest
{
  double away = std::numeric_limits<double>::max();
  double level = 0.0;
};
Nearest nearestOn(const RoadPiece& p, const glm::dvec3& d, double radius)
{
  Nearest          best;
  const int        stretches = p.length > 80.0F ? 8 : 4;
  glm::dvec3       from      = p.a;
  for(int k = 1; k <= stretches; k++)
  {
    const glm::dvec3 to    = bent(p.a, p.b, p.c, double(k) / stretches);
    const glm::dvec3 along = to - from;
    const double     len2  = glm::dot(along, along);
    const double     s     = len2 > 0.0 ? std::clamp(glm::dot(d - from, along) / len2, 0.0, 1.0) : 0.0;
    const double     away  = glm::length(d - (from + along * s)) * radius;
    if(away < best.away)
      best = {away, bent(double(p.level[0]), double(p.level[1]), double(p.level[2]), (double(k - 1) + s) / stretches)};
    from = to;
  }
  return best;
}

// The ground with the levelled sites near a point: `e` as the land has it, level on a site's floor
// and eased into it past the site's reach. `taken`, `site`: what the nearest pack says of the point.
double levelled(const Places& pl, const World& w, const glm::dvec3& d, double e, uint8_t* taken = nullptr, int32_t* at = nullptr)
{
  const uint32_t c = cellOf(pl.grid, d);
  for(uint32_t k = pl.first[c]; k < pl.first[c + 1]; k++)
  {
    const uint32_t item = pl.items[k];
    if(item & Places::kRoadItem)
      continue;
    const Site&  s    = pl.sites[item];
    const Pack&  pack = pl.catalogue->packs[s.pack];
    const double away = glm::length(d - s.direction) * w.spec.radius;
    const double edge = padEdge(pack.radius);
    if(away >= pack.radius + edge)
      continue;
    if(s.level)
    {
      const double flat = floorAt(w, s, d);
      e                 = flat + (e - flat) * smooth(pack.radius, pack.radius + edge, away);
      // (a place of any size stands in a clearing: its floor is open ground)
      if(taken && away < pack.radius && pack.radius > kClearedFrom)
        *taken |= kGroundOpen;
    }
    if(taken && away < pack.radius * 1.42)
    {
      glm::dvec3 ax, az;
      axesOf(s, ax, az);
      const glm::dvec3 rel  = d * ((w.spec.radius + s.elevation) / std::max(glm::dot(d, s.direction), 0.5)) - s.direction * (w.spec.radius + s.elevation);
      const uint8_t    bits = pack.groundAt(glm::dot(rel, ax), glm::dot(rel, az));
      if(bits)
      {
        *taken |= bits;
        if(at)
          *at = int32_t(item);
      }
    }
  }
  return e;
}

// ---- the index -------------------------------------------------------------------------------------------

using Bins = std::vector<std::pair<uint32_t, uint32_t>>;  // cell, item

// The cells `near` (those some points of a shape fell in) and the cells round them, of which `reaches`
// says which the shape reaches into: listed for `item`. A cell a shape only clips the corner of has
// none of the points, but is beside one that has.
template <class Reaches>
void binCells(Bins& bins, const CubeGrid& g, std::vector<uint32_t>& near, uint32_t item, const Reaches& reaches)
{
  std::sort(near.begin(), near.end());
  near.erase(std::unique(near.begin(), near.end()), near.end());
  const size_t own = near.size();
  for(size_t k = 0; k < own; k++)
  {
    uint32_t round[8];
    g.around(near[k], round);
    for(const uint32_t n : round)
      if(n != CubeGrid::kNone)
        near.push_back(n);
  }
  std::sort(near.begin(), near.end());
  near.erase(std::unique(near.begin(), near.end()), near.end());
  for(const uint32_t c : near)
    if(reaches(g.centre(c)))
      bins.emplace_back(c, item);
}

// Every cell a disc reaches into: those whose middle is within the disc's reach and a cell's own
// (no cell is more than three quarters of the widest from its middle to its corner)
void binDisc(Bins& bins, const CubeGrid& g, double radius, const glm::dvec3& d, double reach, uint32_t item)
{
  const double across = cellAcross(g, radius), step = 0.25 * across;
  glm::dvec3   a, b;
  tangents(d, a, b);
  std::vector<uint32_t> near;
  const int             n = int(std::ceil(reach / step)) + 1;
  for(int j = -n; j <= n; j++)
    for(int i = -n; i <= n; i++)
      if(double(i * i + j * j) * step * step <= (reach + 2.0 * step) * (reach + 2.0 * step))
        near.push_back(cellOf(g, glm::normalize(d + (a * (double(i) * step) + b * (double(j) * step)) / radius)));
  binCells(bins, g, near, item, [&](const glm::dvec3& centre) { return glm::length(centre - d) * radius <= reach + 0.75 * across; });
}

// Every cell a road piece's verges reach into, at their widest
void binPiece(Bins& bins, const CubeGrid& g, double radius, const RoadPiece& p, uint32_t item)
{
  const double across = cellAcross(g, radius), step = 0.25 * across;
  const double reach  = double(p.half) + double(p.verge) + kBankMost;
  const int    along  = std::max(2, int(std::ceil(double(p.length) / step)));
  const int    side   = int(std::ceil(reach / step)) + 1;
  std::vector<uint32_t> near;
  for(int k = 0; k <= along; k++)
  {
    const double     t  = double(k) / along;
    const glm::dvec3 at = glm::normalize(bent(p.a, p.b, p.c, t));
    glm::dvec3       run = (p.b - p.a) * (1.0 - t) + (p.c - p.b) * t;
    if(glm::dot(run, run) < 1e-24)
      run = p.c - p.a;
    const glm::dvec3 over = glm::normalize(glm::cross(at, run));
    for(int o = -side; o <= side; o++)
      near.push_back(cellOf(g, glm::normalize(at + over * (double(o) * step / radius))));
  }
  binCells(bins, g, near, item, [&](const glm::dvec3& centre) { return nearestOn(p, centre, radius).away <= reach + 0.75 * across; });
}

void index(Places& pl, double radius)
{
  Bins bins;
  for(uint32_t i = 0; i < pl.sites.size(); i++)
  {
    const Pack& pack = pl.catalogue->packs[pl.sites[i].pack];
    binDisc(bins, pl.grid, radius, pl.sites[i].direction, pl.sites[i].level ? pack.radius + padEdge(pack.radius) : pack.radius, i);
  }
  for(uint32_t i = 0; i < pl.roads.size(); i++)
    binPiece(bins, pl.grid, radius, pl.roads[i], i | Places::kRoadItem);
  std::sort(bins.begin(), bins.end());
  bins.erase(std::unique(bins.begin(), bins.end()), bins.end());
  pl.first.assign(size_t(pl.grid.count()) + 1, 0);
  pl.items.resize(bins.size());
  for(const auto& [cell, item] : bins)
    pl.first[cell + 1]++;
  for(size_t c = 0; c < pl.grid.count(); c++)
    pl.first[c + 1] += pl.first[c];
  for(size_t k = 0; k < bins.size(); k++)
    pl.items[k] = bins[k].second;  // (sorted by cell, then item)
}

// Metres from a point to the nearest road's middle line, of those within `reach` of it; huge with none
double roadAway(const Places& pl, const glm::dvec3& d, double radius, double reach)
{
  glm::dvec3 a, b;
  tangents(d, a, b);
  double best = std::numeric_limits<double>::max();
  // (a piece is listed in the cells its verges reach: the point's own cell and those `reach` round it)
  for(int probe = 0; probe < 5; probe++)
  {
    const glm::dvec3 q = probe == 0 ? d : glm::normalize(d + (probe < 3 ? a : b) * ((probe % 2 ? reach : -reach) / radius));
    const uint32_t   c = cellOf(pl.grid, q);
    for(uint32_t k = pl.first[c]; k < pl.first[c + 1]; k++)
      if(pl.items[k] & Places::kRoadItem)
        best = std::min(best, nearestOn(pl.roads[pl.items[k] & ~Places::kRoadItem], d, radius).away);
  }
  return best;
}

// How many runs of road pass within `reach` of a point
size_t waysNear(const Places& pl, const glm::dvec3& d, double radius, double reach)
{
  glm::dvec3 a, b;
  tangents(d, a, b);
  std::vector<uint32_t> ways;
  for(int probe = 0; probe < 5; probe++)
  {
    const glm::dvec3 q = probe == 0 ? d : glm::normalize(d + (probe < 3 ? a : b) * ((probe % 2 ? reach : -reach) / radius));
    const uint32_t   c = cellOf(pl.grid, q);
    for(uint32_t k = pl.first[c]; k < pl.first[c + 1]; k++)
      if(pl.items[k] & Places::kRoadItem)
        if(const RoadPiece& piece = pl.roads[pl.items[k] & ~Places::kRoadItem]; nearestOn(piece, d, radius).away < reach)
          ways.push_back(piece.way);
  }
  std::sort(ways.begin(), ways.end());
  return size_t(std::unique(ways.begin(), ways.end()) - ways.begin());
}

// ---- where a place can stand -----------------------------------------------------------------------------

struct Ground
{
  bool   ok    = false;
  double level = 0.0;  // the ground's mean under it
  double slope = 0.0;  // rise over run, the most from its middle
};

// The ground under something `radius` metres round `d`: dry and on one tier of the land out to
// where its levelled floor would ease into it
Ground groundFor(const World& w, const glm::dvec3& d, const Sample& s, double radius)
{
  const double R = w.spec.radius, footprint = radius > 20.0 ? 8.0 : 2.0;
  if(s.elevation < 3.0 || s.shore < 8.0F)
    return {};
  glm::dvec3 a, b;
  tangents(d, a, b);
  Ground g{true, s.elevation, 0.0};
  int    n = 1;
  for(const double away : {0.4 * radius, 0.75 * radius, radius, radius + padEdge(radius)})
    for(int turn = 0; turn < 8; turn++)
    {
      const double     angle = (double(turn) + (away > 0.5 * radius ? 0.5 : 0.0)) * (kPi / 4.0);
      const glm::dvec3 side  = a * std::cos(angle) + b * std::sin(angle);
      const Sample     there = sample(w, glm::normalize(d + side * (away / R)), footprint);
      if(!(there.elevation > 1.5 && there.shore > 5.0F && std::abs(there.tiers - s.tiers) < 0.02F))
        return {};
      if(away <= radius)
      {
        g.level += there.elevation;
        n++;
        g.slope = std::max(g.slope, std::abs(there.elevation - s.elevation) / away);
      }
    }
  g.level /= double(n);
  return g;
}

// How well each kind likes each of the planet's biomes
std::vector<std::array<float, kMostBiomes>> likings(const World& w, const Catalogue& cat)
{
  std::vector<std::array<float, kMostBiomes>> like(cat.kinds.size());
  for(size_t k = 0; k < cat.kinds.size(); k++)
  {
    like[k].fill(0.0F);
    for(size_t i = 0; i < w.biomes.size() && i < kMostBiomes; i++)
      for(const auto& [name, liking] : cat.kinds[k].biomes)
        if(name == w.biomes[i].name)
          like[k][i] = liking;
  }
  return like;
}

float likingAt(const std::array<float, kMostBiomes>& like, const Sample& s)
{
  float sum = 0.0F;
  for(size_t i = 0; i < kMostBiomes; i++)
    sum += s.share[i] * like[i];
  return sum;
}

// One of `weights` by `pick` (0..1); -1 when none weighs anything
int pickOf(std::span<const float> weights, double pick)
{
  double total = 0.0;
  for(const float v : weights)
    total += std::max(v, 0.0F);
  if(total <= 0.0)
    return -1;
  double at = pick * total;
  for(size_t i = 0; i < weights.size(); i++)
    if(weights[i] > 0.0F && (at -= double(weights[i])) < 0.0)
      return int(i);
  for(size_t i = weights.size(); i-- > 0;)
    if(weights[i] > 0.0F)
      return int(i);
  return -1;
}

uint32_t packOf(const Catalogue& cat, const PlaceKind& kind, double pick)
{
  std::vector<float> weights;
  for(const uint32_t p : kind.packs)
    weights.push_back(cat.packs[p].weight);
  return kind.packs[size_t(std::max(pickOf(weights, pick), 0))];
}

// ---- the roads' routes -----------------------------------------------------------------------------------

struct Router
{
  enum : uint8_t { eNo = 0, eLand, eFord };

  const CubeGrid&       g;
  double                radius;
  std::vector<float>    height;
  std::vector<uint8_t>  pass;     // by cell
  std::vector<uint8_t>  shore;    // water in a cell beside it
  std::vector<uint8_t>  blocked;  // a place's own ground, the figures'
  std::vector<uint8_t>  shy;      // round the figures: a road may pass, and would rather not
  std::vector<uint8_t>  road;     // a road runs through it already
  std::set<std::pair<uint32_t, uint32_t>> runs;  // from which cell to which (the lower first)
  std::vector<int32_t>  land;     // which land it is of, -1 none
  std::vector<float>    cost;
  std::vector<uint32_t> came;
  std::vector<uint32_t> touched;
  std::vector<glm::vec3> centres;  // every cell's, once: a route asks them over and over

  Router(const CubeGrid& grid, double r) : g(grid), radius(r) {}

  bool open(uint32_t c) const { return pass[c] != eNo && !blocked[c]; }

  // Rise over run from a cell to one beside it
  double climb(uint32_t from, uint32_t to) const
  {
    return std::abs(double(height[to]) - double(height[from])) / std::max(double(glm::length(centres[to] - centres[from])) * radius, 1.0);
  }

  // The lands: the cells a road can go between
  void lands()
  {
    land.assign(g.count(), -1);
    int32_t               next = 0;
    std::vector<uint32_t> stack;
    for(uint32_t c = 0; c < g.count(); c++)
    {
      if(pass[c] == eNo || land[c] >= 0)
        continue;
      land[c] = next;
      stack.push_back(c);
      while(!stack.empty())
      {
        const uint32_t at = stack.back();
        stack.pop_back();
        uint32_t round[8];
        g.around(at, round);
        for(const uint32_t n : round)
          if(n != CubeGrid::kNone && pass[n] != eNo && land[n] < 0 && climb(at, n) <= kRoadSteepest)
          {
            land[n] = next;
            stack.push_back(n);
          }
      }
      next++;
    }
  }

  // The cheapest way from cell `a` to cell `b`, both ends in it; false when there is none. With
  // kNone for `b`: to the nearest cell a road runs through already.
  bool route(uint32_t a, uint32_t b, std::vector<uint32_t>& path)
  {
    const bool toRoad = b == CubeGrid::kNone;
    if(toRoad && road[a])
      return false;  // (on one: nothing to lead)
    if(cost.empty())
    {
      cost.assign(g.count(), std::numeric_limits<float>::max());
      came.assign(g.count(), CubeGrid::kNone);
    }
    for(const uint32_t c : touched)
    {
      cost[c] = std::numeric_limits<float>::max();
      came[c] = CubeGrid::kNone;
    }
    touched.clear();
    const glm::vec3 goal  = toRoad ? glm::vec3(0.0F) : centres[b];
    const auto      ahead = [&](const glm::vec3& from) { return toRoad ? 0.0F : float(kRouteDrawn * double(glm::length(from - goal)) * radius); };
    using Entry = std::pair<float, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    cost[a] = 0.0F;
    touched.push_back(a);
    queue.emplace(0.0F, a);
    bool   found = false;
    size_t asked = 0;
    while(!queue.empty() && asked++ < kRouteMost)
    {
      const auto [f, at] = queue.top();
      queue.pop();
      if(toRoad ? road[at] != 0 : at == b)
      {
        found = true;
        b     = at;
        break;
      }
      const glm::vec3 here = centres[at];
      if(f > cost[at] + ahead(here) + 1.0F)
        continue;  // an older entry
      uint32_t round[8];
      g.around(at, round);
      for(int k = 0; k < 8; k++)
      {
        const uint32_t n = round[k];
        if(n == CubeGrid::kNone || !open(n))
          continue;
        // no road crosses another between four cells: it goes by one of the other's cells, and
        // they meet there (two that cross where neither has a point would each have its own level)
        if(k % 2)
          if(const uint32_t a = round[(k + 7) % 8], b = round[(k + 1) % 8]; a != CubeGrid::kNone && b != CubeGrid::kNone && runs.count({std::min(a, b), std::max(a, b)}))
            continue;
        const glm::vec3 there = centres[n];
        const double    len   = double(glm::length(there - here)) * radius;
        const double     slope = std::abs(double(height[n]) - double(height[at])) / std::max(len, 1.0);
        if(slope > kRoadSteepest)
          continue;
        double step = len * (1.0 + 0.5 * (slope / kRoadEasy) * (slope / kRoadEasy));
        if(shore[n])
          step *= 1.6;
        if(shy[n])
          step *= 2.5;
        if(road[n])
          step *= 0.4;
        if(pass[n] == eFord)
          step += 150.0;
        const float through = cost[at] + float(step);
        if(through < cost[n])
        {
          if(cost[n] == std::numeric_limits<float>::max())
            touched.push_back(n);
          cost[n] = through;
          came[n] = at;
          queue.emplace(through + ahead(there), n);
        }
      }
    }
    if(!found)
      return false;
    path.clear();
    for(uint32_t c = b; c != CubeGrid::kNone; c = came[c])
    {
      path.push_back(c);
      if(c == a)
        break;
    }
    std::reverse(path.begin(), path.end());
    return path.front() == a;
  }
};

// The roads as they are routed: which node runs to which, and by what kind of road
struct Graph
{
  std::map<uint32_t, std::vector<uint32_t>>       next;
  std::map<std::pair<uint32_t, uint32_t>, uint8_t> kind;

  void join(uint32_t u, uint32_t v, uint8_t k, const Catalogue& cat)
  {
    if(u == v)
      return;
    const auto key = std::make_pair(std::min(u, v), std::max(u, v));
    const auto at  = kind.find(key);
    if(at == kind.end())
    {
      kind[key] = k;
      next[u].push_back(v);
      next[v].push_back(u);
    }
    else if(cat.roads[k].width > cat.roads[at->second].width)
      at->second = k;
  }
  uint8_t of(uint32_t u, uint32_t v) const { return kind.at(std::make_pair(std::min(u, v), std::max(u, v))); }
};

struct Settling
{
  const World&             w;
  const Catalogue&         cat;
  Places&                  pl;
  double                   amount;
  const std::atomic<bool>* stop;
  double                   R;
  uint32_t                 seed;
  double                   keep;  // metres round the figures' place that stay theirs
  std::vector<std::array<float, kMostBiomes>> like;

  bool stopped() const { return stop && stop->load(std::memory_order_relaxed); }

  bool onLand(const glm::dvec3& d) const
  {
    if(!w.base)
      return true;
    const uint8_t kind = w.base->kind[cellOf(w.base->grid, d)];
    return kind == eMapLand || kind == eMapBasin;
  }
  double fromFigures(const glm::dvec3& d) const { return glm::length(d - w.frame.site) * R; }

  // ---- the places a road is led to
  struct Candidate
  {
    bool     ok = false;      // the ground will have it
    bool     picked = false;  // a kind likes the land there, and its pack is chosen
    uint32_t pack = 0;
    double   level = 0.0;
    float    spacing = 0.0F;
  };

  // What could stand at `d`: a kind by the land there, a pack of it, if the ground will have it
  Candidate candidateAt(const glm::dvec3& d, uint32_t salt, bool roads, double steeper = 1.0) const
  {
    if(!onLand(d))
      return {};
    const Sample       s = sample(w, d, 8.0);
    std::vector<float> weights(cat.kinds.size(), 0.0F);
    for(size_t k = 0; k < cat.kinds.size(); k++)
      if(!cat.kinds[k].packs.empty() && (roads ? cat.kinds[k].road >= 0 : cat.kinds[k].alone > 0.0F))
        weights[k] = (roads ? cat.kinds[k].weight : cat.kinds[k].alone) * likingAt(like[k], s);
    const int k = pickOf(weights, unit(hash(seed, salt, 1)));
    if(k < 0)
      return {};
    const PlaceKind& kind = cat.kinds[size_t(k)];
    // (fewer of a kind in a land it likes less)
    if(unit(hash(seed, salt, 2)) >= double(likingAt(like[size_t(k)], s)))
      return {};
    Candidate    c;
    c.picked       = true;
    c.pack         = packOf(cat, kind, unit(hash(seed, salt, 3)));
    const Ground g = groundFor(w, d, s, cat.packs[c.pack].radius);
    c.ok           = g.ok && g.slope <= double(kind.slope) * steeper;
    c.level        = g.level;
    c.spacing      = kind.spacing;
    return c;
  }

  double reachOf(uint32_t pack) const { return cat.packs[pack].radius + padEdge(cat.packs[pack].radius); }

  bool roomFor(const glm::dvec3& d, const Candidate& c, const std::vector<float>& spacings) const
  {
    if(fromFigures(d) < keep + reachOf(c.pack) + 10.0)
      return false;
    const double scale = 1.0 / std::sqrt(amount);
    for(size_t i = 0; i < pl.sites.size(); i++)
    {
      const double apart = glm::length(d - pl.sites[i].direction) * R;
      if(apart < 0.5 * double(c.spacing + spacings[i]) * scale || apart < reachOf(c.pack) + reachOf(pl.sites[i].pack) + 30.0)
        return false;
    }
    return true;
  }

  void stand(const glm::dvec3& d, const Candidate& c, double heading, bool level = true)
  {
    Site s;
    s.direction = d;
    s.elevation = c.level;
    s.heading   = heading;
    s.pack      = c.pack;
    s.level     = level;
    pl.sites.push_back(s);
  }

  void places()
  {
    std::vector<float> spacings;
    // one within a walk of the figures, outside what their own scene may level: the first that the
    // ground will have, going out and round
    {
      glm::dvec3 a, b;
      tangents(w.frame.site, a, b);
      const double from = unit(hash(seed, 7000)) * 2.0 * kPi;
      bool         done = false;
      for(int ring = 0; ring < 10 && !done; ring++)
        for(int turn = 0; turn < 16 && !done; turn++)
        {
          const double     angle = from + double(turn) * (kPi / 8.0) + double(ring) * 0.2;
          const glm::dvec3 side  = a * std::cos(angle) + b * std::sin(angle);
          const uint32_t   salt  = 7001U + uint32_t(ring * 16 + turn);
          // (which kind, by the land a little way out; then where its reach lets it stand)
          Candidate c = candidateAt(glm::normalize(w.frame.site + side * ((keep + 150.0) / R)), salt, true, 1e9);
          if(!c.picked)
            continue;
          const glm::dvec3 d = glm::normalize(w.frame.site + side * ((keep + reachOf(c.pack) + 15.0 + 50.0 * ring) / R));
          const Sample     s = sample(w, d, 8.0);
          const Ground     g = groundFor(w, d, s, cat.packs[c.pack].radius);
          if(!onLand(d) || !g.ok || g.slope > 1.5 * double(cat.kinds[cat.packs[c.pack].kind].slope))
            continue;
          c.level = g.level;
          stand(d, c, unit(hash(seed, salt, 4)) * 2.0 * kPi);
          spacings.push_back(c.spacing);
          done = true;
        }
    }
    if(stopped())
      return;
    // the rest: places spread over the planet, each asked in parallel, taken in a fixed order while
    // there is room
    const uint32_t         count = uint32_t(std::clamp(4.0 * kPi * R * R / (250.0 * 250.0), 4000.0, 40000.0));
    std::vector<Candidate> found(count);
    fv::jobs::parallelFor(count, 64, [&](size_t begin, size_t end) {
      for(size_t i = begin; i < end && !stopped(); i++)
        found[i] = candidateAt(spread(uint32_t(i), count), 8000U + uint32_t(i), true);
    });
    std::vector<uint32_t> order(count);
    for(uint32_t i = 0; i < count; i++)
      order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
      const uint32_t hx = hash(seed, 7900, x), hy = hash(seed, 7900, y);
      return hx != hy ? hx < hy : x < y;
    });
    for(const uint32_t i : order)
    {
      if(pl.sites.size() >= kMostJoined)
        break;
      const glm::dvec3 d = spread(i, count);
      if(!found[i].ok || !roomFor(d, found[i], spacings))
        continue;
      stand(d, found[i], unit(hash(seed, 8000U + i, 4)) * 2.0 * kPi);
      spacings.push_back(found[i].spacing);
    }
    pl.joined = uint32_t(pl.sites.size());
  }

  // ---- the roads between them
  const RoadKind* roadOf(const Site& s) const
  {
    const int r = cat.kinds[cat.packs[s.pack].kind].road;
    return r >= 0 ? &cat.roads[size_t(r)] : nullptr;
  }

  void router(Router& r) const
  {
    const uint32_t count = pl.grid.count();
    r.height.assign(count, 0.0F);
    r.pass.assign(count, Router::eLand);
    r.shore.assign(count, 0);
    r.blocked.assign(count, 0);
    r.shy.assign(count, 0);
    r.road.assign(count, 0);
    r.centres.resize(count);
    fv::jobs::parallelFor(count, 8192, [&](size_t begin, size_t end) {
      for(size_t c = begin; c < end; c++)
        r.centres[c] = glm::vec3(pl.grid.centre(uint32_t(c)));
    });
    if(w.base && w.base->grid.cells() == pl.grid.cells())
    {
      const BaseMap& m = *w.base;
      const int      n = int(pl.grid.cells());
      for(uint32_t c = 0; c < count; c++)
      {
        r.height[c] = m.height[c];
        if(m.kind[c] == eMapSea || m.kind[c] == eMapLake || m.height[c] < 1.0F)
          r.pass[c] = Router::eNo;
      }
      // (the map's land by a shore is the land with what the rain laid on it, which the ground itself
      // does not have under the sea: a bay may be where the map has a plain. Low cells are asked.)
      fv::jobs::parallelFor(count, 4096, [&](size_t begin, size_t end) {
        for(size_t c = begin; c < end && !stopped(); c++)
          if(r.pass[c] != Router::eNo && m.height[c] < 25.0F && sample(w, pl.grid.centre(uint32_t(c)), 8.0).elevation < 1.0)
            r.pass[c] = Router::eNo;
      });
      for(uint8_t face = 0; face < 6; face++)
        for(int j = 0; j < n; j++)
          for(int i = 0; i < n; i++)
            if(const int32_t piece = m.river[m.grid.padded(face, i, j)]; piece >= 0)
            {
              const uint32_t c = m.grid.index(face, uint32_t(i), uint32_t(j));
              if(r.pass[c] != Router::eNo)
                r.pass[c] = m.pieces[size_t(piece)].half[1] <= float(kFordHalf) ? Router::eFord : Router::eNo;
            }
    }
    else
    {
      // no map to read: the land itself, a cell at a time
      const double across = cellAcross(pl.grid, R);
      fv::jobs::parallelFor(count, 1024, [&](size_t begin, size_t end) {
        for(size_t c = begin; c < end && !stopped(); c++)
        {
          const Sample s = sample(w, pl.grid.centre(uint32_t(c)), across);
          r.height[c]    = float(s.elevation);
          if(s.elevation < 1.0 || s.water > s.elevation)
            r.pass[c] = Router::eNo;
        }
      });
    }
    // a road keeps a cell off the sea and the lakes (a cell is coarse: its shore is not the water's),
    // and pays for running by a stream
    std::vector<uint8_t> water(count, 0);
    for(uint32_t c = 0; c < count; c++)
      water[c] = r.pass[c] == Router::eNo ? 1 : 0;
    for(uint32_t c = 0; c < count; c++)
    {
      if(water[c])
        continue;
      uint32_t round[8];
      pl.grid.around(c, round);
      for(const uint32_t n : round)
      {
        if(n == CubeGrid::kNone)
          continue;
        if(water[n])
          r.pass[c] = Router::eNo;
        else if(r.pass[n] == Router::eFord)
          r.shore[c] = 1;
      }
    }
    r.lands();
    // nothing is led across a place's own ground, nor where the figures stand; round them, as far
    // as their scene may level the ground, a road may pass (a floor has it as paint) and would
    // rather go round
    const auto mark = [&](std::vector<uint8_t>& cells, const glm::dvec3& d, double reach) {
      Bins bins;
      binDisc(bins, pl.grid, R, d, reach, 0);
      for(const auto& [cell, item] : bins)
        if(glm::length(pl.grid.centre(cell) - d) * R < reach)
          cells[cell] = 1;
    };
    mark(r.blocked, w.frame.site, kFiguresOwn);
    mark(r.shy, w.frame.site, keep);
    // (out to a quarter of a cell past its ground: no other road passes between a place and the
    // cell its own leaves by)
    for(uint32_t i = 0; i < pl.joined; i++)
      mark(r.blocked, pl.sites[i].direction, reachOf(pl.sites[i].pack) + 0.25 * cellAcross(pl.grid, R));
  }

  // A cell a road can start in, at `d` or beside it; kNone when all round is water or someone's ground
  uint32_t openBy(const Router& r, const glm::dvec3& d) const
  {
    const uint32_t c = cellOf(pl.grid, d);
    if(r.open(c))
      return c;
    uint32_t round[8];
    pl.grid.around(c, round);
    uint32_t best = CubeGrid::kNone;
    double   near = std::numeric_limits<double>::max();
    for(const uint32_t n : round)
      if(n != CubeGrid::kNone && r.open(n))
        if(const double away = glm::length(pl.grid.centre(n) - d); away < near)
        {
          near = away;
          best = n;
        }
    return best;
  }

  void roads()
  {
    const uint32_t n = pl.joined;
    if(n < 1 || cat.roads.empty())
      return;
    Router r(pl.grid, R);
    router(r);
    if(stopped())
      return;
    // which land each place is on
    std::vector<int32_t> land(n, -1);
    for(uint32_t i = 0; i < n; i++)
    {
      const uint32_t c = cellOf(pl.grid, pl.sites[i].direction);
      land[i]          = r.land[c];
      uint32_t round[8];
      pl.grid.around(c, round);
      for(int k = 0; k < 8 && land[i] < 0; k++)
        if(round[k] != CubeGrid::kNone)
          land[i] = r.land[round[k]];
    }
    struct Link
    {
      double   apart;
      uint32_t a, b;
      bool     operator<(const Link& o) const { return apart != o.apart ? apart < o.apart : (a != o.a ? a < o.a : b < o.b); }
    };
    std::vector<Link> pairs;
    for(uint32_t a = 0; a < n; a++)
      for(uint32_t b = a + 1; b < n; b++)
        if(land[a] >= 0 && land[a] == land[b] && roadOf(pl.sites[a]) && roadOf(pl.sites[b]))
          pairs.push_back({glm::length(pl.sites[a].direction - pl.sites[b].direction) * R, a, b});
    std::sort(pairs.begin(), pairs.end());
    // a place faces the nearest of its land: its gate's way out is turned toward it
    std::vector<uint8_t> faced(n, 0);
    for(const Link& l : pairs)
      for(const auto& [from, to] : {std::make_pair(l.a, l.b), std::make_pair(l.b, l.a)})
      {
        Site&       s    = pl.sites[from];
        const Pack& pack = cat.packs[s.pack];
        if(faced[from] || pack.gates.empty())
          continue;
        faced[from] = 1;
        glm::dvec3 a, b;
        tangents(s.direction, a, b);
        const glm::dvec3 toward = pl.sites[to].direction - s.direction;
        const Gate&      gate   = pack.gates.front();
        s.heading = std::atan2(glm::dot(toward, b), glm::dot(toward, a)) - std::atan2(-gate.heading.y, gate.heading.x);
      }
    // the place a walk from the figures (the first, when the land had one): alone on its land, it
    // faces them
    const bool byFigures = fromFigures(pl.sites[0].direction) < keep + 800.0 && roadOf(pl.sites[0]) && !cat.packs[pl.sites[0].pack].gates.empty();
    if(byFigures && !faced[0])
    {
      Site&      s = pl.sites[0];
      glm::dvec3 a, b;
      tangents(s.direction, a, b);
      const glm::dvec3 toward = w.frame.site - s.direction;
      const Gate&      gate   = cat.packs[s.pack].gates.front();
      s.heading = std::atan2(glm::dot(toward, b), glm::dot(toward, a)) - std::atan2(-gate.heading.y, gate.heading.x);
      faced[0]  = 1;
    }
    // where each one's road leaves it: its gate, straight out to the rim of its floor, and from
    // there to a cell of the map just clear of its ground. Where that cell is water or another's
    // ground, the place is turned a little further until it is not.
    const uint32_t          count  = pl.grid.count();
    const double            across = cellAcross(pl.grid, R);
    std::vector<glm::dvec3> extra(size_t(2) * n);
    std::vector<uint32_t>   start(n, CubeGrid::kNone);
    for(uint32_t i = 0; i < n; i++)
    {
      Site&       s    = pl.sites[i];
      const Pack& pack = cat.packs[s.pack];
      if(pack.gates.empty() || !faced[i])
        continue;
      const Gate& gate = pack.gates.front();
      // (how far along the gate's way out a point is `from` the place's middle)
      const double along = glm::dot(gate.at, gate.heading), gone = glm::dot(gate.at, gate.at);
      const auto   out   = [&](double from) { return -along + std::sqrt(std::max(along * along - gone + from * from, 0.0)); };
      const double rim = out(pack.radius + 0.5 * padEdge(pack.radius)), clear = out(reachOf(s.pack) + across);
      const double first = s.heading;
      for(int turn = 0; turn < 12 && start[i] == CubeGrid::kNone; turn++)
      {
        s.heading                = first + double((turn + 1) / 2) * (turn % 2 ? 1.0 : -1.0) * (kPi / 6.0);
        extra[size_t(2) * i]     = fromSite(w, s, gate.at.x, gate.at.y);
        extra[size_t(2) * i + 1] = fromSite(w, s, gate.at.x + gate.heading.x * rim, gate.at.y + gate.heading.y * rim);
        start[i]                 = openBy(r, fromSite(w, s, gate.at.x + gate.heading.x * clear, gate.at.y + gate.heading.y * clear));
      }
      if(start[i] == CubeGrid::kNone)
        s.heading = first;
    }
    // which is joined to which: the shortest tree over each land's places that can be routed, and a
    // few ways round; each way over the land the cheaper where a road already runs
    std::vector<uint32_t> group(n);
    for(uint32_t i = 0; i < n; i++)
      group[i] = i;
    const auto root = [&](uint32_t i) {
      while(group[i] != i)
        i = group[i] = group[group[i]];
      return i;
    };
    Graph                 graph;
    std::vector<uint32_t> path;
    std::vector<uint32_t> nearer(n, 0), failed(n, 0);
    std::vector<uint8_t>  linked(n, 0);
    for(const Link& l : pairs)
    {
      if(stopped())
        return;
      const uint32_t ra = root(l.a), rb = root(l.b);
      const bool     tree = ra != rb;
      // (a way round: between two that are among each other's nearest three, one time in three)
      const bool round = !tree && nearer[l.a] < 3 && nearer[l.b] < 3 && unit(hash(seed, 7100, l.a, l.b)) < 0.35;
      nearer[l.a]++;
      nearer[l.b]++;
      if((!tree && !round) || start[l.a] == CubeGrid::kNone || start[l.b] == CubeGrid::kNone || failed[l.a] >= 4 || failed[l.b] >= 4)
        continue;
      if(!r.route(start[l.a], start[l.b], path))
      {
        failed[l.a]++;
        failed[l.b]++;
        continue;
      }
      group[ra] = rb;
      const RoadKind *ka = roadOf(pl.sites[l.a]), *kb = roadOf(pl.sites[l.b]);
      const uint8_t   kind = uint8_t((ka->width >= kb->width ? ka : kb) - cat.roads.data());
      std::vector<uint32_t> nodes{count + 2U * l.a, count + 2U * l.a + 1U};
      for(size_t k = 0; k < path.size(); k++)
      {
        nodes.push_back(path[k]);
        r.road[path[k]] = 1;
        if(k)
          r.runs.insert({std::min(path[k - 1], path[k]), std::max(path[k - 1], path[k])});
      }
      nodes.insert(nodes.end(), {count + 2U * l.b + 1U, count + 2U * l.b});
      for(size_t k = 0; k + 1 < nodes.size(); k++)
        graph.join(nodes[k], nodes[k + 1], kind, cat);
      linked[l.a] = linked[l.b] = 1;
    }
    // a place no way was found from to another is led to the nearest road there is: it joins that
    // road where it meets it
    for(uint32_t i = 0; i < n && !graph.next.empty(); i++)
    {
      if(stopped())
        return;
      if(linked[i] || start[i] == CubeGrid::kNone || !r.route(start[i], CubeGrid::kNone, path))
        continue;
      const uint8_t         kind = uint8_t(roadOf(pl.sites[i]) - cat.roads.data());
      std::vector<uint32_t> nodes{count + 2U * i, count + 2U * i + 1U};
      for(size_t k = 0; k < path.size(); k++)
      {
        nodes.push_back(path[k]);
        r.road[path[k]] = 1;
        if(k)
          r.runs.insert({std::min(path[k - 1], path[k]), std::max(path[k - 1], path[k])});
      }
      for(size_t k = 0; k + 1 < nodes.size(); k++)
        graph.join(nodes[k], nodes[k + 1], kind, cat);
    }
    // and from the place by the figures a road comes to where they stand, and ends there: whoever
    // stands there has a way to follow, to that place and by its roads to the others
    if(byFigures && start[0] != CubeGrid::kNone)
    {
      const glm::dvec3 toward = glm::normalize(pl.sites[0].direction - w.frame.site);
      const uint32_t   end    = openBy(r, glm::normalize(w.frame.site + toward * ((kFiguresOwn + across) / R)));
      if(end != CubeGrid::kNone && r.route(start[0], end, path))
      {
        const uint8_t         kind = uint8_t(roadOf(pl.sites[0]) - cat.roads.data());
        std::vector<uint32_t> nodes{count, count + 1U};
        nodes.insert(nodes.end(), path.begin(), path.end());
        for(size_t k = 0; k + 1 < nodes.size(); k++)
          graph.join(nodes[k], nodes[k + 1], kind, cat);
      }
    }
    if(graph.next.empty())
      return;
    for(auto& [node, next] : graph.next)
      std::sort(next.begin(), next.end());

    // every node's place and level: the ground's, a place's floor at its gate
    std::vector<uint32_t> nodes;
    for(const auto& [node, next] : graph.next)
      nodes.push_back(node);
    const auto at = [&](uint32_t node) { return node < count ? pl.grid.centre(node) : extra[node - count]; };
    std::vector<double> levels(nodes.size());
    fv::jobs::parallelFor(nodes.size(), 256, [&](size_t begin, size_t end) {
      for(size_t k = begin; k < end && !stopped(); k++)
      {
        const glm::dvec3 d = at(nodes[k]);
        levels[k]          = roadLevel(d, sample(w, d, kRoadFootprint).elevation);
      }
    });
    std::map<uint32_t, double> level;
    for(size_t k = 0; k < nodes.size(); k++)
      level[nodes[k]] = levels[k];

    // the runs of road between junctions and ends, each a line of nodes
    std::set<std::pair<uint32_t, uint32_t>> walked;
    const auto walk = [&](uint32_t from, uint32_t to) {
      std::vector<uint32_t> line{from, to};
      walked.insert({from, to});
      walked.insert({to, from});
      uint32_t last = from, here = to;
      while(graph.next[here].size() == 2 && here != from)
      {
        const auto&    next = graph.next[here];
        const uint32_t on   = next[0] == last ? next[1] : next[0];
        walked.insert({here, on});
        walked.insert({on, here});
        line.push_back(on);
        last = here;
        here = on;
      }
      way(graph, line, at, level);
    };
    for(const auto& [node, next] : graph.next)
      if(next.size() != 2)
        for(const uint32_t to : next)
          if(!walked.count({node, to}))
            walk(node, to);
    for(const auto& [node, next] : graph.next)  // (rings with no junction on them)
      for(const uint32_t to : next)
        if(!walked.count({node, to}))
          walk(node, to);
  }

  // The ground a road's level is read from at a point: the land's without its fine relief, a place's
  // floor where one is, and never under the sea
  double roadLevel(const glm::dvec3& d, double e) const { return std::max(levelled(pl, w, d, e), 1.0); }

  // A run of road as curves. Its line is from node to node through the middles between them, bent
  // at the nodes; along that, a point every kRoadStep metres takes the ground's level, and the road
  // is the curve through the middles between those, bent at them: so it rises and falls with the
  // land it crosses, and cuts or fills only what is finer than its steps.
  template <class At>
  void way(const Graph& graph, const std::vector<uint32_t>& line, const At& at, const std::map<uint32_t, double>& level)
  {
    const size_t            k     = line.size() - 1;
    const uint32_t          count = pl.grid.count();
    std::vector<glm::dvec3> P(line.size());
    for(size_t i = 0; i <= k; i++)
      P[i] = at(line[i]);
    // a way from cell to cell is a stair: its corners are eased (its ends stay, which other ways
    // share, and what a place's gate fixes)
    for(int pass = 0; pass < 2; pass++)
    {
      const std::vector<glm::dvec3> was = P;
      for(size_t i = 1; i < k; i++)
        if(line[i] < count)
          P[i] = glm::normalize(0.25 * was[i - 1] + 0.5 * was[i] + 0.25 * was[i + 1]);
    }
    const auto wider = [&](uint8_t x, uint8_t y) { return cat.roads[x].width >= cat.roads[y].width ? x : y; };
    // the points along it, each with the kind of road that leaves it
    std::vector<glm::dvec3> Q;
    std::vector<uint8_t>    K;
    const auto              stretch = [&](const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c, uint8_t kind) {
      const int n = std::max(1, int(std::lround((glm::length(b - a) + glm::length(c - b)) * R / kRoadStep)));
      for(int j = 0; j < n; j++)
      {
        Q.push_back(glm::normalize(bent(a, b, c, double(j) / n)));
        K.push_back(kind);
      }
    };
    const auto mid = [](const std::vector<glm::dvec3>& of, size_t i) { return glm::normalize(of[i] + of[i + 1]); };
    if(k == 1)
      stretch(P[0], mid(P, 0), P[1], graph.of(line[0], line[1]));
    else
    {
      stretch(P[0], glm::normalize(P[0] + mid(P, 0)), mid(P, 0), graph.of(line[0], line[1]));
      for(size_t i = 1; i < k; i++)
        stretch(mid(P, i - 1), P[i], mid(P, i), wider(graph.of(line[i - 1], line[i]), graph.of(line[i], line[i + 1])));
      stretch(mid(P, k - 1), glm::normalize(mid(P, k - 1) + P[k]), P[k], graph.of(line[k - 1], line[k]));
    }
    Q.push_back(P[k]);
    K.push_back(K.back());
    const size_t        n = Q.size() - 1;
    std::vector<double> L(Q.size());
    L[0] = level.at(line.front());
    L[n] = level.at(line.back());
    for(size_t j = 1; j < n; j++)
      L[j] = sample(w, Q[j], kRoadFootprint).elevation;
    const std::vector<double> ground = L;
    for(int pass = 0; pass < 2; pass++)
    {
      const std::vector<double> had = L;
      for(size_t j = 1; j < n; j++)
        L[j] = 0.25 * had[j - 1] + 0.5 * had[j] + 0.25 * had[j + 1];
    }
    for(size_t j = 1; j < n; j++)
      L[j] = roadLevel(Q[j], L[j]);
    // and no steeper than a road is: up a bluff it is cut into the top and banked out at the foot
    // (from both ends, which stay: where they are too far apart in height for its length, it is steep)
    std::vector<double> step(n);
    // (gentler by its ends: the runs that meet at a junction leave it nearly level with each other)
    for(size_t j = 0; j < n; j++)
      step[j] = kRoadGrade * (std::min(j, n - 1 - j) == 0 ? 0.5 : std::min(j, n - 1 - j) == 1 ? 0.75 : 1.0) * glm::length(Q[j + 1] - Q[j]) * R;
    for(int pass = 0; pass < 2; pass++)
    {
      for(size_t j = 1; j < n; j++)
        L[j] = std::clamp(L[j], L[j - 1] - step[j - 1], L[j - 1] + step[j - 1]);
      for(size_t j = n - 1; j >= 1; j--)
        L[j] = std::clamp(L[j], L[j + 1] - step[j], L[j + 1] + step[j]);
    }
    // (a long climb steeper than that is the land's: the road stays within a cutting's depth of it)
    for(size_t j = 1; j < n; j++)
      L[j] = roadLevel(Q[j], std::clamp(L[j], ground[j] - kRoadCut, ground[j] + kRoadCut));

    float      run   = 0.0F;
    const auto piece = [&](const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c, double la, double lb, double lc, uint8_t kind) {
      RoadPiece p;
      p.a = a, p.b = b, p.c = c;
      p.level[0] = float(la), p.level[1] = float(lb), p.level[2] = float(lc);
      p.half   = float(0.5 * cat.roads[kind].width);
      p.verge  = float(cat.roads[kind].verge);
      p.kind   = kind;
      p.way    = pl.ways;
      p.start  = run;
      p.length = float((glm::length(b - a) + glm::length(c - b)) * R);
      run += p.length;
      pl.roads.push_back(p);
    };
    if(n == 1)
      piece(Q[0], mid(Q, 0), Q[1], L[0], 0.5 * (L[0] + L[1]), L[1], K[0]);
    else
    {
      const auto half = [&](size_t i) { return 0.5 * (L[i] + L[i + 1]); };
      piece(Q[0], glm::normalize(Q[0] + mid(Q, 0)), mid(Q, 0), L[0], 0.5 * (L[0] + half(0)), half(0), K[0]);
      for(size_t i = 1; i < n; i++)
        piece(mid(Q, i - 1), Q[i], mid(Q, i), half(i - 1), L[i], half(i), wider(K[i - 1], K[i]));
      piece(mid(Q, n - 1), glm::normalize(mid(Q, n - 1) + Q[n]), Q[n], half(n - 1), 0.5 * (half(n - 1) + L[n]), L[n], K[n - 1]);
    }
    pl.ways++;
  }

  // ---- single things
  // Whether something `reach` metres round `d` stands clear of every site's ground
  bool clearOfSites(const glm::dvec3& d, double reach, bool levelledOnly = false) const
  {
    for(const Site& s : pl.sites)
      if(!levelledOnly || s.level)
        if(glm::length(d - s.direction) * R < reach + (s.level ? reachOf(s.pack) : cat.packs[s.pack].radius))
          return false;
    return true;
  }

  // Alone in the land: each kind's own count of places over the sphere, those the land will have
  void alone()
  {
    double per = 0.0;
    for(const PlaceKind& k : cat.kinds)
      per += k.packs.empty() ? 0.0 : double(k.alone);
    if(per <= 0.0)
      return;
    const double   area  = 4.0 * kPi * R * R * 1e-6;  // square kilometres, the sea's too
    const uint32_t count = uint32_t(std::min(area * per * amount, double(kMostAlone) * 3.0));
    if(!count)
      return;
    std::vector<Candidate>  found(count);
    std::vector<glm::dvec3> where(count);
    const double            apart = std::sqrt(4.0 * kPi / double(count));  // between two of them, as an angle
    fv::jobs::parallelFor(count, 64, [&](size_t begin, size_t end) {
      for(size_t i = begin; i < end && !stopped(); i++)
      {
        // (off its spot by up to a third of the way to the next: no rows)
        glm::dvec3 d = spread(uint32_t(i), count), a, b;
        tangents(d, a, b);
        const uint32_t salt = 9000U + uint32_t(i);
        d        = glm::normalize(d + (a * (unit(hash(seed, salt, 5)) - 0.5) + b * (unit(hash(seed, salt, 6)) - 0.5)) * (0.66 * apart));
        where[i] = d;
        found[i] = candidateAt(d, salt, false);
      }
    });
    uint32_t stood = 0;
    for(uint32_t i = 0; i < count && stood < kMostAlone; i++)
    {
      if(!found[i].ok)
        continue;
      const double reach = reachOf(found[i].pack);
      if(fromFigures(where[i]) < keep + reach + 5.0 || !clearOfSites(where[i], reach + 8.0))
        continue;
      if(roadAway(pl, where[i], R, reach) < reach + 12.0)
        continue;
      stand(where[i], found[i], unit(hash(seed, 9000U + i, 4)) * 2.0 * kPi);
      stood++;
    }
  }

  // Along the roads: each rule's spots down every run of road
  void along()
  {
    if(cat.along.empty() || pl.roads.empty())
      return;
    // a run's pieces, in their order
    std::vector<std::pair<uint32_t, uint32_t>> runs(pl.ways, {UINT32_MAX, 0});
    for(uint32_t i = 0; i < pl.roads.size(); i++)
    {
      auto& run  = runs[pl.roads[i].way];
      run.first  = std::min(run.first, i);
      run.second = std::max(run.second, i + 1);
    }
    uint32_t stood = 0;
    for(size_t rule = 0; rule < cat.along.size(); rule++)
    {
      const Along&     a    = cat.along[rule];
      const PlaceKind& kind = cat.kinds[a.kind];
      if(kind.packs.empty() || a.every <= 1.0)
        continue;
      for(uint32_t run = 0; run < pl.ways; run++)
      {
        const auto [from, to] = runs[run];
        if(from >= to)
          continue;
        const double length = double(pl.roads[to - 1].start) + double(pl.roads[to - 1].length);
        uint32_t     piece  = from;
        for(uint32_t spot = 0;; spot++)
        {
          const uint32_t salt = hash(uint32_t(rule) + 1U, run, spot);
          const double   s    = (double(spot) + unit(hash(seed, 9500, uint32_t(rule), run))) * a.every;
          if(s >= length || stood >= kMostAlong)
            break;
          while(piece + 1 < to && s >= double(pl.roads[piece].start) + double(pl.roads[piece].length))
            piece++;
          const RoadPiece& p = pl.roads[piece];
          if(p.kind != a.road || unit(hash(seed, salt, 1)) >= a.chance)
            continue;
          const double     t      = std::clamp((s - double(p.start)) / std::max(double(p.length), 1e-6), 0.0, 1.0);
          const glm::dvec3 on     = glm::normalize(bent(p.a, p.b, p.c, t));
          const glm::dvec3 run3   = (p.b - p.a) * (1.0 - t) + (p.c - p.b) * t;
          const glm::dvec3 across = glm::normalize(glm::cross(on, run3));
          const double     side   = (hash(seed, salt, 2) & 1U) ? a.side : -a.side;
          const glm::dvec3 d      = glm::normalize(on + across * (side / R));
          Candidate        c;
          c.pack = packOf(cat, kind, unit(hash(seed, salt, 3)));
          const double reach = cat.packs[c.pack].radius;
          // not on a place's own ground or the figures', nor in a stream or on another road
          if(fromFigures(d) < keep + reach || !clearOfSites(d, 0.5 * reach))
            continue;
          if(a.within > 0.0)
          {
            bool near = false;
            for(uint32_t i = 0; i < pl.joined && !near; i++)
              near = glm::length(d - pl.sites[i].direction) * R < a.within + cat.packs[pl.sites[i].pack].radius;
            if(!near)
              continue;
          }
          if(roadAway(pl, d, R, a.side + 2.0) < std::abs(a.side) - 0.5 || waysNear(pl, d, R, std::abs(a.side) + 8.0) > 1)
            continue;  // (nor by a junction, where which road it stands by is anyone's guess)
          const Sample ground = sample(w, d, 4.0);
          if(ground.elevation < 1.5 || ground.shore < 6.0F)
            continue;
          // its +x toward the road, and as much further as its rule turns it
          glm::dvec3 ta, tb;
          tangents(d, ta, tb);
          const glm::dvec3 toward = on - d;
          const double     facing = std::atan2(glm::dot(toward, tb), glm::dot(toward, ta));
          stand(d, c, facing + a.turn + a.spin * (2.0 * unit(hash(seed, salt, 4)) - 1.0), false);
          stood++;
        }
      }
    }
  }
};

}  // namespace

// ---- the catalogue -----------------------------------------------------------------------------------------

uint8_t Pack::groundAt(double x, double z) const
{
  const int i = int(std::floor(x / double(cell) + 0.5 * double(n))), j = int(std::floor(z / double(cell) + 0.5 * double(n)));
  return i >= 0 && j >= 0 && i < n && j < n ? ground[size_t(j) * size_t(n) + size_t(i)] : uint8_t(0);
}

std::shared_ptr<const Catalogue> readCatalogue(const nlohmann::json& j)
{
  const auto fail = [](const std::string& what) -> void { throw std::runtime_error("places.json: " + what); };
  auto       cat  = std::make_shared<Catalogue>();
  try
  {
    if(!j.is_object() || j.value("version", 0) != 1)
      fail("not a catalogue of version 1");
    const nlohmann::json& rules = j.at("rules");
    for(const auto& r : rules.at("roads"))
    {
      RoadKind road;
      road.name  = r.at("name").get<std::string>();
      road.width = std::clamp(r.value("width", 6.0), 1.0, 30.0);
      road.verge = std::clamp(r.value("verge", 3.0), 0.5, 30.0);
      if(const auto c = r.find("colour"); c != r.end() && c->is_array() && c->size() == 3)
        road.colour = glm::clamp(glm::vec3((*c)[0].get<float>(), (*c)[1].get<float>(), (*c)[2].get<float>()), 0.0F, 1.0F);
      road.earth = r.value("surface", std::string("asphalt")) == "earth";
      cat->roads.push_back(road);
    }
    if(cat->roads.size() > 250)
      fail("too many kinds of road");
    const auto roadNamed = [&](const std::string& name) {
      for(size_t i = 0; i < cat->roads.size(); i++)
        if(cat->roads[i].name == name)
          return int(i);
      fail("no road is called " + name);
      return -1;
    };
    const auto kindNamed = [&](const std::string& name, bool make) {
      for(size_t i = 0; i < cat->kinds.size(); i++)
        if(cat->kinds[i].name == name)
          return uint32_t(i);
      if(!make)
        fail("no pack is of the kind " + name);
      PlaceKind kind;
      kind.name = name;
      cat->kinds.push_back(kind);
      return uint32_t(cat->kinds.size() - 1);
    };
    for(const auto& k : rules.at("kinds"))
    {
      PlaceKind& kind = cat->kinds[kindNamed(k.at("kind").get<std::string>(), true)];
      if(k.contains("road"))
        kind.road = roadNamed(k.at("road").get<std::string>());
      kind.weight  = std::max(k.value("weight", 1.0F), 0.0F);
      kind.alone   = std::max(k.value("alone", 0.0F), 0.0F);
      kind.slope   = std::clamp(k.value("slope", 0.1F), 0.01F, 1.0F);
      kind.spacing = std::clamp(k.value("spacing", 1000.0F), 50.0F, 1.0e6F);
      if(const auto b = k.find("biomes"); b != k.end())
        for(const auto& [name, liking] : b->items())
          kind.biomes.emplace_back(name, std::clamp(liking.get<float>(), 0.0F, 1.0F));
      if((kind.road >= 0) == (kind.alone > 0.0F))
        fail("the kind " + kind.name + " is to stand by a road or alone, one of the two");
    }
    for(const auto& p : j.at("packs"))
    {
      Pack pack;
      pack.name   = p.at("name").get<std::string>();
      pack.kind   = kindNamed(p.at("kind").get<std::string>(), true);
      pack.radius = p.at("radius").get<double>();
      pack.weight = std::max(p.value("weight", 1.0F), 0.0F);
      if(!(pack.radius > 0.0 && pack.radius < 2000.0))
        fail("the pack " + pack.name + " reaches no way or too far");
      for(const auto& g : p.value("gates", nlohmann::json::array()))
      {
        Gate gate;
        gate.at      = {g.at("at").at(0).get<double>(), g.at("at").at(1).get<double>()};
        gate.heading = {g.at("heading").at(0).get<double>(), g.at("heading").at(1).get<double>()};
        if(glm::length(gate.heading) < 1e-6 || glm::length(gate.at) > pack.radius)
          fail("a gate of the pack " + pack.name + " leads nowhere or is outside it");
        gate.heading = glm::normalize(gate.heading);
        pack.gates.push_back(gate);
      }
      for(const auto& q : p.at("pieces"))
      {
        PackPiece piece;
        piece.file   = q.at("file").get<std::string>();
        piece.growth = q.value("growth", false);
        for(const auto& c : q.at("copies"))
        {
          if(!c.is_array() || c.size() != 4)
            fail("a copy of " + piece.file + " in the pack " + pack.name + " is not x, y, z and a turn");
          piece.copies.push_back({{c[0].get<float>(), c[1].get<float>(), c[2].get<float>()}, c[3].get<float>()});
        }
        if(piece.file.empty() || piece.copies.empty())
          fail("a piece of the pack " + pack.name + " has no file or stands nowhere");
        pack.pieces.push_back(std::move(piece));
      }
      if(pack.pieces.empty())
        fail("the pack " + pack.name + " is made of nothing");
      if(const auto c = p.find("clear"); c != p.end())
      {
        pack.cell = c->value("cell", 1.0F);
        pack.n    = c->value("n", 0);
        const nlohmann::json& rows = c->at("rows");
        if(!(pack.cell > 0.0F) || pack.n < 0 || pack.n > 4096 || rows.size() != size_t(pack.n))
          fail("the ground of the pack " + pack.name + " is not n rows");
        pack.ground.reserve(size_t(pack.n) * size_t(pack.n));
        for(const auto& row : rows)
        {
          const std::string cells = row.get<std::string>();
          if(cells.size() != size_t(pack.n))
            fail("a row of the ground of the pack " + pack.name + " is not n cells");
          for(const char ch : cells)
          {
            if(ch < '0' || ch > '3')
              fail("a cell of the ground of the pack " + pack.name + " is none of 0 to 3");
            pack.ground.push_back(uint8_t(ch - '0'));
          }
        }
      }
      cat->kinds[pack.kind].packs.push_back(uint32_t(cat->packs.size()));
      cat->packs.push_back(std::move(pack));
    }
    for(const PlaceKind& kind : cat->kinds)
    {
      if(kind.packs.empty())
        fail("no pack is of the kind " + kind.name);
      for(const uint32_t p : kind.packs)
        if(kind.road >= 0 && cat->packs[p].gates.empty())
          fail("the pack " + cat->packs[p].name + " has no gate for the road its kind is led");
    }
    for(const auto& a : rules.value("along", nlohmann::json::array()))
    {
      Along along;
      along.kind   = kindNamed(a.at("kind").get<std::string>(), false);
      along.road   = uint32_t(roadNamed(a.at("road").get<std::string>()));
      along.every  = std::max(a.value("every", 100.0), 2.0);
      along.chance = std::clamp(a.value("chance", 0.5), 0.0, 1.0);
      along.side   = a.value("side", 4.0);
      along.turn   = a.value("turn", 0.0);
      along.spin   = a.value("spin", 0.0);
      along.within = std::max(a.value("within", 0.0), 0.0);
      cat->along.push_back(along);
    }
  }
  catch(const nlohmann::json::exception& e)
  {
    fail(e.what());
  }
  return cat;
}

// ---- settling ----------------------------------------------------------------------------------------------

double padEdge(double radius)
{
  return std::clamp(0.5 * radius, 6.0, 40.0);
}

std::shared_ptr<const Places> settle(const World& world, std::shared_ptr<const Catalogue> catalogue, double amount, const std::atomic<bool>* stop)
{
  if(!catalogue || catalogue->packs.empty() || !(amount > 0.0))
    return nullptr;
  const double R     = world.spec.radius;
  auto         out   = std::make_shared<Places>(world.base ? world.base->grid.cells() : std::min(mapCells(R), 256U));
  out->catalogue     = catalogue;
  const double level = mostLevel(R);
  // (the planet as it is with nothing in the scene: a road may pass where a scene levels the ground,
  // and what is settled is the same whatever stands there)
  const World  plain = withClearing(world, kClearing);
  Settling     job{plain, *catalogue, *out, std::clamp(amount, 0.05, 4.0), stop, R, uint32_t(world.spec.seed) * 2654435761U + 77U,
                   level + std::max(kClearingEdge, 0.5 * level), likings(world, *catalogue)};
  job.places();
  if(job.stopped())
    return nullptr;
  index(*out, R);  // (the roads' levels are read against the places' floors)
  job.roads();
  if(job.stopped())
    return nullptr;
  index(*out, R);
  job.alone();
  job.along();
  if(job.stopped())
    return nullptr;
  index(*out, R);
  return out;
}

World withPlaces(const World& world, std::shared_ptr<const Places> places)
{
  World w = world;
  std::erase_if(w.stages, [](const Stage& s) { return s.name == "places"; });
  w.places = std::move(places);
  if(w.places)
  {
    // before the ground's colour, which a road has a say in; after all that shapes the land
    const auto colour = std::find_if(w.stages.begin(), w.stages.end(), [](const Stage& s) { return s.name == "colour"; });
    w.stages.insert(colour, Stage{"places", built});
  }
  return w;
}

void built(const World& w, const Point& p, Sample& s)
{
  const Places&  pl = *w.places;
  const uint32_t c  = cellOf(pl.grid, p.direction);
  if(pl.first[c] == pl.first[c + 1])
    return;
  const double R = w.spec.radius;
  // the roads near: the ground cut and filled to their level across them, eased into the land over
  // their verges. A run of road is one curve, and its nearest point says its level; where runs meet,
  // each has its say by how near it is: no step at a junction.
  struct Run
  {
    uint32_t         way;
    Nearest          near;
    const RoadPiece* piece;
  };
  std::array<Run, 6> runs;
  size_t             count = 0;
  bool               sites = false;
  for(uint32_t k = pl.first[c]; k < pl.first[c + 1]; k++)
  {
    const uint32_t item = pl.items[k];
    if(!(item & Places::kRoadItem))
    {
      sites = true;
      continue;
    }
    const RoadPiece& piece = pl.roads[item & ~Places::kRoadItem];
    const Nearest    near  = nearestOn(piece, p.direction, R);
    size_t           at    = 0;
    while(at < count && runs[at].way != piece.way)
      at++;
    if(at == count && count < runs.size())
      runs[count++] = {piece.way, near, &piece};
    else if(at < count && near.away < runs[at].near.away)
      runs[at] = {piece.way, near, &piece};
  }
  double           weight = 0.0, level = 0.0, most = 0.0, cover = 0.0;
  const RoadPiece* top = nullptr;
  for(size_t k = 0; k < count; k++)
  {
    const RoadPiece& piece = *runs[k].piece;
    const Nearest&   near  = runs[k].near;
    // (its verge: wider where it stands high over the land or deep in it)
    const double verge = std::clamp(kBankRun * std::abs(near.level - s.elevation), double(piece.verge), double(piece.verge) + kBankMost);
    const double share = 1.0 - smooth(double(piece.half), double(piece.half) + verge, near.away);
    if(share <= 0.0)
      continue;
    weight += share * share;
    level += share * share * near.level;
    if(share > most)
    {
      most = share;
      top  = &piece;
    }
    // how much of what draws the sample is its surface: a soft edge near, a share of a wide sample
    const double across = std::max(p.footprint, 1.5);
    cover = std::max(cover, std::clamp((double(piece.half) - near.away) / across + 0.5, 0.0, 1.0) * std::min(1.0, 2.0 * double(piece.half) / across));
  }
  if(top)
  {
    // by a stream the road comes down to its bank and its bed: a ford. (Over a cove of the sea it
    // is a causeway: no road is led across the sea, but a shore has inlets finer than its routes.)
    const double dry   = smooth(1.0, 7.0, double(s.shore));
    const double whole = 2.0 * (double(top->half) + double(top->verge));
    const double drawn = 1.0 - smooth(0.25 * whole, 0.5 * whole, p.footprint);
    s.elevation        = mixed(s.elevation, level / weight, most * drawn * dry);
    s.built.road       = float(cover * dry);
    const RoadKind& kind = pl.catalogue->roads[top->kind];
    s.built.colour       = kind.colour;
    s.built.earth        = kind.earth;
    if(s.built.road > 0.5F)
      s.built.taken |= kGroundCovered | kGroundKept;
    else if(most * dry > 0.5)
      s.built.taken |= kGroundKept;
  }
  if(sites)
    s.elevation = levelled(pl, w, p.direction, s.elevation, &s.built.taken, &s.built.site);
}

// ---- for what draws them -------------------------------------------------------------------------------------

glm::dmat4 siteFrame(const World& world, const Site& site)
{
  glm::dvec3 x, z;
  axesOf(site, x, z);
  const double     elevation = site.level ? site.elevation : sample(world, site.direction, 1.0).elevation;
  const glm::dvec3 at        = toScene(world, site.direction, elevation);
  glm::dmat4       m(1.0);
  m[0] = glm::dvec4(world.frame.turn * x, 0.0);
  m[1] = glm::dvec4(world.frame.turn * site.direction, 0.0);
  m[2] = glm::dvec4(world.frame.turn * z, 0.0);
  m[3] = glm::dvec4(at, 1.0);
  return m;
}

glm::mat4 copyPlace(const PieceCopy& copy)
{
  return glm::rotate(glm::translate(glm::mat4(1.0F), copy.at), copy.yaw, glm::vec3(0.0F, 1.0F, 0.0F));
}

double seenFrom(const Pack& pack)
{
  return std::clamp(30.0 * pack.radius, 400.0, 3000.0);
}

std::vector<uint32_t> sitesSeen(const World& world, const glm::dvec3& eye, std::span<const uint32_t> was)
{
  std::vector<uint32_t> seen;
  if(!world.places)
    return seen;
  const Places& pl = *world.places;
  for(uint32_t i = 0; i < pl.sites.size(); i++)
  {
    const Site&  s     = pl.sites[i];
    const double range = seenFrom(pl.catalogue->packs[s.pack]) * (std::binary_search(was.begin(), was.end(), i) ? kSplitHold : 1.0);
    if(glm::length(toScene(world, s.direction, s.elevation) - eye) < range)
      seen.push_back(i);
  }
  return seen;
}

}  // namespace fv::world
