#include "fv_world.hpp"
#include "fv_world_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fv::world {

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---- noise -------------------------------------------------------------------------------------------

// PCG's output mix: the same everywhere, so a seed's planet is
uint32_t mix(uint32_t v)
{
  v = v * 747796405U + 2891336453U;
  const uint32_t w = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (w >> 22U) ^ w;
}

// [0, 1)
double unit(uint32_t h)
{
  return double(h >> 8U) * (1.0 / 16777216.0);
}

double smooth(double a, double b, double x)
{
  const double t = std::clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// A lattice of its own for a seed's channel
uint32_t lattice(uint32_t seed, uint32_t channel)
{
  return mix(mix(seed) + channel);
}

// Smooth value noise on a unit lattice in space, [0, 1). In space and not on the sphere's surface:
// a sphere has no seam or pole to it.
double valueNoise3(const glm::dvec3& p, uint32_t base)
{
  const glm::dvec3 f = glm::floor(p);
  const int64_t    ix = int64_t(f.x), iy = int64_t(f.y), iz = int64_t(f.z);
  const glm::dvec3 t = p - f;
  const glm::dvec3 u = t * t * (3.0 - 2.0 * t);
  const auto       at = [&](int dx, int dy, int dz) {
    return unit(mix(mix(mix(base + uint32_t(ix + dx)) + uint32_t(iy + dy)) + uint32_t(iz + dz)));
  };
  const double x00 = at(0, 0, 0) + (at(1, 0, 0) - at(0, 0, 0)) * u.x;
  const double x10 = at(0, 1, 0) + (at(1, 1, 0) - at(0, 1, 0)) * u.x;
  const double x01 = at(0, 0, 1) + (at(1, 0, 1) - at(0, 0, 1)) * u.x;
  const double x11 = at(0, 1, 1) + (at(1, 1, 1) - at(0, 1, 1)) * u.x;
  const double y0  = x00 + (x10 - x00) * u.y;
  const double y1  = x01 + (x11 - x01) * u.y;
  return y0 + (y1 - y0) * u.z;
}

// What each octave is turned by as well as halved, so their lattices don't line up
const glm::dmat3& octaveTurn()
{
  static const glm::dmat3 turn = glm::mat3_cast(glm::angleAxis(0.65, glm::normalize(glm::dvec3(1.0, 2.0, 3.0))));
  return turn;
}

glm::dvec3 nextOctave(const glm::dvec3& p)
{
  return octaveTurn() * p * 2.0 + glm::dvec3(17.3, -9.1, 5.7);
}

// How much of an octave whose lattice cells are `cells` across is kept by something `finest` across
// (both in the first octave's cells): all of it at four of them a cell, none under two. 0: all.
double kept(double finest, double cells)
{
  return finest > 0.0 ? smooth(2.0 * finest, 4.0 * finest, cells) : 1.0;
}

// Octaves of value noise, [0, 1) and 0.5 on average; an octave too fine to draw is its mean
double fbm3(glm::dvec3 p, uint32_t seed, uint32_t channel, uint32_t octaves, double finest = 0.0)
{
  double sum = 0.0, amplitude = 0.5, total = 0.0, cells = 1.0;
  for(uint32_t o = 0; o < octaves; o++)
  {
    const double k = kept(finest, cells);
    sum += amplitude * (0.5 + (k > 0.0 ? k * (valueNoise3(p, lattice(seed, channel + o)) - 0.5) : 0.0));
    total += amplitude;
    amplitude *= 0.5;
    cells *= 0.5;
    p = nextOctave(p);
  }
  return sum / total;
}

// The same folded about its middle and squared: ridges where the noise crosses it, 0 to 1
double ridged3(glm::dvec3 p, uint32_t seed, uint32_t channel, uint32_t octaves, double finest = 0.0)
{
  constexpr double kMean = 1.0 / 3.0;  // of a squared fold of an even noise
  double           sum = 0.0, amplitude = 0.5, total = 0.0, cells = 1.0;
  for(uint32_t o = 0; o < octaves; o++)
  {
    const double k = kept(finest, cells);
    double       r = kMean;
    if(k > 0.0)
    {
      const double fold = 1.0 - std::abs(2.0 * valueNoise3(p, lattice(seed, channel + o)) - 1.0);
      r                 = kMean + k * (fold * fold - kMean);
    }
    sum += amplitude * r;
    total += amplitude;
    amplitude *= 0.5;
    cells *= 0.5;
    p = nextOctave(p);
  }
  return sum / total;
}

// The noise channels, ten apart: an fbm takes one an octave
constexpr uint32_t kChContinent = 10, kChRange = 20, kChRidge = 30, kChHills = 40, kChWarm = 50, kChWet = 60, kChDetail = 70,
                   kChShade = 90, kChPatch = 100, kChSite = 110, kChDither = 120, kChCliff = 130, kChGully = 150;

// How many of each noise's cells go across the planet's radius: the continents are the planet's
// size, the ranges a few to a continent, a hill a few hundred metres on a planet 20 km in radius
constexpr double kContinents = 1.7, kRanges = 2.3, kRidges = 7.0, kHills = 26.0;

// ---- the stages --------------------------------------------------------------------------------------

uint32_t seedOf(const World& w)
{
  return uint32_t(w.spec.seed);
}

// A stage's footprint in the cells of a noise that has `across` of them over the radius
double finestOf(const World& w, const Point& p, double across)
{
  return p.footprint * across / w.spec.radius;
}

double continentNoise(const Spec& spec, const glm::dvec3& direction, double finest)
{
  return fbm3(direction * kContinents, uint32_t(spec.seed), kChContinent, 5, finest);
}

// The land and the sea's floor: the land rising inland from the shore and the floor falling away from
// it, each at a slope from the first metre. (Level at the shore, a beach was kilometres wide: every
// plain within a few metres of the sea is sand.)
void continents(const World& w, const Point& p, Sample& s)
{
  s.continent    = (continentNoise(w.spec, p.direction, finestOf(w, p, kContinents)) - w.seaLevel) / w.seaSpread;
  // what is placed here by hand, for this stage and those after: land or sea where the author says
  if(w.spec.features)
  {
    s.placed = placedAt(*w.spec.features, p.direction, w.spec.radius);
    s.continent += double(s.placed.continent);
  }
  const double u = s.continent;
  const double R = w.spec.radius;
  if(u >= 0.0)
    s.elevation = (0.5 + 0.5 * w.spec.relief) * 0.006 * R * u * (u + 0.15) / (u + 0.5);
  else
    s.elevation = 0.012 * R * u * (0.15 - u) / (0.5 - u);
}

// Mountains in ranges, inland
void mountains(const World& w, const Point& p, Sample& s)
{
  double inland = smooth(0.15, 0.9, s.continent);
  // a placed range stands wherever it is put, by a coast or in the sea; a plain stands the land down
  const double placed = double(s.placed.range) * (1.0 - double(s.placed.level));
  if((inland <= 0.0 && placed <= 0.0) || w.spec.relief <= 0.0)
    return;
  const double ranges = fbm3(p.direction * kRanges + glm::dvec3(3.1, 7.7, 1.3), seedOf(w), kChRange, 3, finestOf(w, p, kRanges));
  double       range  = smooth(0.50, 0.66, ranges);
  if(w.spec.features)
  {
    // the seed's own ranges as far as they are let stand, and the placed one, its edge and its height
    // wandering with the seed's noise so it is no even bank
    range *= w.spec.features->wild * (1.0 - double(s.placed.level));
    const double put = placed * (0.6 + 0.8 * ranges);
    if(put > inland * range)
    {
      inland = 1.0;
      range  = put;
    }
  }
  if(range <= 0.0 || inland <= 0.0)
    return;
  const double ridge = ridged3(p.direction * kRidges, seedOf(w), kChRidge, 5, finestOf(w, p, kRidges));
  s.mountain         = float(inland * range * ridge);
  s.elevation += 0.045 * w.spec.radius * w.spec.relief * double(s.mountain);
}

// Hills, lower toward the shore and on the sea's floor. Their four octaves are a share of the radius:
// on a large planet the finest is some 500 m across (on one 20 km in radius, 100 m), and the land
// between it and the ground's bumps was smooth. There the series goes on, each octave half as wide
// and half as high as the last, while its cells are at least kHillsSmallest across; the four are as
// they were.
constexpr double   kHillsSmallest = 75.0;     // metres
constexpr uint32_t kHillsOctaves  = 4;
constexpr double   kHillsSum      = 0.9375;   // the four octaves' amplitudes, which fbm3 shares them by

void hills(const World& w, const Point& p, Sample& s)
{
  if(w.spec.relief <= 0.0)
    return;
  const double finest = finestOf(w, p, kHills);
  double       h      = fbm3(p.direction * kHills, seedOf(w), kChHills, kHillsOctaves, finest) - 0.5;
  {
    glm::dvec3 q = p.direction * kHills;
    for(uint32_t o = 0; o < kHillsOctaves; o++)
      q = nextOctave(q);
    double amplitude = std::ldexp(0.5, -int(kHillsOctaves)), cells = std::ldexp(1.0, -int(kHillsOctaves));
    for(uint32_t o = kHillsOctaves; o < 10 && cells * w.spec.radius / kHills >= kHillsSmallest; o++)
    {
      const double k = kept(finest, cells);
      if(k <= 0.0)
        break;  // (nor any finer)
      h += amplitude / kHillsSum * k * (valueNoise3(q, lattice(seedOf(w), kChHills + o)) - 0.5);
      amplitude *= 0.5;
      cells *= 0.5;
      q = nextOctave(q);
    }
  }
  const double land = 0.15 + 0.85 * smooth(0.0, 0.5, s.continent);
  s.elevation += 2.0 * h * 0.007 * w.spec.radius * w.spec.relief * land * (1.0 - 0.85 * double(s.placed.level));
  // the placed peaks and basins
  s.elevation += double(s.placed.elevation);
}

// Cliffs: the land stands in tiers, each a step over the one below along a winding line, so a tier's
// edge is a face of rock a few metres across however gently the land under it slopes. (Terraces cut
// at levels of the land's own height are only as steep as that land: on a plain they are ramps.) A
// noise over the ground, in metres: where it passes one of a few levels the land steps up, over a
// band of the noise that is a face's width where the noise changes at its usual rate, and no narrower
// than two cells of what draws it. The face is steepest under its crest, which is an edge, and eases
// into the ground at its foot; a finer noise moves its line a few metres either way, so it runs in
// buttresses and gullies, not as a bank's smooth curve.
// By a shore the tiers come up with the land's own height, a metre for a metre: twice the land's
// slope there, no more. (Stood straight out of the beach a tier was 30 m of wall a metre across; six
// metres for a metre made the few centimetres two levels of quads differ by into a row of teeth.)
constexpr double kCliffAcross = 1500.0;                 // metres across a tier's lobes
constexpr double kCliffHeight = 15.0;                   // metres a tier stands over the one below, at relief 1
constexpr double kCliffTiers[] = {0.56, 0.62, 0.68};    // of the noise: over a quarter of the land, a seventh, a fifteenth
constexpr double kCliffEdge  = 0.0015;                  // of the noise, either side of a level: a face some 6 m across
constexpr double kCliffSlope = 5.0e-4;                  // what the noise changes by a metre, about
constexpr double kCliffJag    = 0.02;                   // of the noise: how far the finer noise moves a face's line, end to end (some 8 m either way)
constexpr double kCliffJagAcross = 45.0;                // metres across a buttress
constexpr double kCliffShore = 1.0;                     // metres of tier a metre of the land's own height brings, at the shore

void cliffs(const World& w, const Point& p, Sample& s)
{
  if(w.spec.relief <= 0.0 || s.elevation <= 0.0)
    return;
  const double broad = fbm3(p.direction * (w.spec.radius / kCliffAcross) + glm::dvec3(5.3, -2.1, 8.8), seedOf(w), kChCliff, 3, p.footprint / kCliffAcross);
  const double jag   = fbm3(p.direction * (w.spec.radius / kCliffJagAcross), seedOf(w), kChCliff + 8, 2, p.footprint / kCliffJagAcross) - 0.5;
  const double tier  = broad + kCliffJag * jag;
  const double edge  = std::max(kCliffEdge, p.footprint * kCliffSlope);
  double       up = 0.0, face = 0.0;
  for(const double at : kCliffTiers)
  {
    // across the face, foot to crest
    const double x = std::clamp((tier - (at - edge)) / (2.0 * edge), 0.0, 1.0);
    up += x * x;
    face = std::max(face, 4.0 * x * (1.0 - x));
  }
  s.tiers            = float(up);
  const double tiers = kCliffHeight * std::min(w.spec.relief, 1.5) * up;
  if(tiers <= 0.0)
    return;
  // out of the shore, which stays where it is: all of the tiers' height by where the land's own
  // has brought a few metres
  const double short_ = std::exp(-kCliffShore * s.elevation / tiers);
  s.steepened         = float(1.0 + kCliffShore * short_);
  s.elevation += tiers * (1.0 - short_);
  // a face wider than a cliff is high is a slope: what draws it too coarsely to stand it up has none
  s.cliff = float(face * (1.0 - short_) * (1.0 - smooth(1.0, 4.0, edge / kCliffEdge)));
}

// What the base map says of the place: what the rain wore off the land or laid on it, and the water
// near. (What it laid on the land it did not lay in the sea: a shore stays where the stages put it.)
void base(const World& w, const Point& p, Sample& s)
{
  const double added = mapAdded(*w.base, p.direction, p.footprint);
  s.elevation += std::min(added, 0.0) + std::max(added, 0.0) * smooth(0.0, 4.0, s.elevation);
  s.inland = mapInland(*w.base, w.spec.radius, p.direction);
  s.shore  = float(std::min(s.inland.shore(), 1.0e9));
}

// Gullies: what running water cuts into a slope, between the valleys the base map wore (its cells are
// tens of metres to 150) and the ground's fine relief. No simulation: a point's own, from the way the
// land falls there, so it is as fine as what draws it and costs the bake nothing. The idea is Clay
// John's and Fewes's eroded noise, lately Rune Skovbo Johansen's erosion filter: ridges and furrows
// laid as stripes that run down the slope, then finer ones laid down the slope with the first ones'
// sides added to it, so they run into the larger from either side as tributaries do. Written here
// from that idea, and in space, as the planet's noises are.
//
// A size's stripes: each cell of a lattice has waves of its own, level across the slope at the
// point, with a phase of its own; the cells round a point are summed as phasors (Tricard et al.,
// "Procedural Phasor Noise", 2019), so the sum has one phase, to which the ridge's shape is given,
// and a length that falls where the cells disagree: there a gully ends or two join. What steers the
// next size is the sum's smooth slope, not the shaped one's: a direction that turned at once across
// a crest would break the finer stripes along it.
//
// Each size follows the land's fall as broadly as its own lattice (BaseMap::slopes), so the widest
// run down a mountain's flank and the finer down the spurs between them. None is steeper than the land
// is at the finest of those slopes: a valley's floor is steep only seen as broadly as the valley, and
// the widest gullies cut it tens of metres deep round the river on it.
//
// A gully's sides are a share of the slope it runs down, so it is as deep as the land is steep, and
// only on steep land: they fade in from a gentle slope to a steep one, as running water cuts deep
// only where it runs fast. (So the gentle land a road, a site or a hut keeps to is the land's own,
// and a road's level is found without them at all: fv::world::sampleWithout.)
constexpr double kGullyLargest  = 1.0 / 40.0;  // of the radius: from one of the widest gullies to the next
constexpr double kGullySmallest = 35.0;        // metres: and of the narrowest
constexpr double kGullyStripes  = 1.5;         // gullies across a cell of their lattice
constexpr double kGullySlope    = 0.5;         // the land's fall a size follows: over this share of its lattice's cells
constexpr double kGullySide     = 1.0;         // a gully's sides against the slope it runs down
constexpr double kGullyGain     = 0.78;        // each finer size's sides against the last's
constexpr double kGullySteep    = 0.8;         // the slope (rise over run) past which they cut no deeper
constexpr double kGullyFrom     = 0.03;        // the slopes over which they fade in
constexpr double kGullyFull     = 0.25;
constexpr double kGullySharp    = 0.6;         // crests and floors: 0 rounded, 1 creased
constexpr double kGullyBranch   = 0.6;         // how far a gully's sides turn the finer ones into it
constexpr double kGullyFloor    = 1.5;         // a size's slope no more than this of the finest's

// A cell's own phase: one of 256 turns of the circle
const std::array<glm::dvec2, 256>& gullyTurns()
{
  static const std::array<glm::dvec2, 256> turns = [] {
    std::array<glm::dvec2, 256> out{};
    for(size_t i = 0; i < out.size(); i++)
      out[i] = glm::dvec2(std::cos(2.0 * kPi * (double(i) + 0.5) / 256.0), std::sin(2.0 * kPi * (double(i) + 0.5) / 256.0));
    return out;
  }();
  return turns;
}

// (cos a, sin a) to within a ten-millionth: a quarter turn's polynomial, several times as fast as
// the library's two
glm::dvec2 turnOf(double a)
{
  const double q  = std::nearbyint(a * (2.0 / kPi));
  const double r  = a - q * (0.5 * kPi), r2 = r * r;
  const double sn = r * (1.0 - r2 * (1.0 / 6.0 - r2 * (1.0 / 120.0 - r2 * (1.0 / 5040.0))));
  const double c  = 1.0 - r2 * (0.5 - r2 * (1.0 / 24.0 - r2 * (1.0 / 720.0 - r2 * (1.0 / 40320.0))));
  switch(int64_t(q) & 3)
  {
    case 0: return {c, sn};
    case 1: return {-sn, c};
    case 2: return {-c, -sn};
    default: return {sn, -c};
  }
}

// How far round from (1, 0) the point (x, y) is, in half turns, 0 to 1 either way: |atan2| / pi to
// within a hundred-thousandth of a turn, and some times faster
double halfTurns(double y, double x)
{
  const double ax = std::abs(x), ay = std::abs(y), most = std::max(ax, ay);
  if(most <= 0.0)
    return 0.0;
  const double a = std::min(ax, ay) / most, a2 = a * a;
  double       r = ((-0.0464964749 * a2 + 0.15931422) * a2 - 0.327622764) * a2 * a + a;
  if(ay > ax)
    r = 0.5 * kPi - r;
  if(x < 0.0)
    r = kPi - r;
  return r / kPi;
}

}  // namespace

double gulliesAt(const World& w, const BaseMap& map, const glm::dvec3& direction, double footprint, const Sample& s)
{
  const double level = 1.0 - double(s.placed.level);  // a placed plain is a plain
  if(w.spec.relief <= 0.0 || s.elevation <= 1.0 || level <= 0.0)
    return 0.0;
  const Point     p{direction, footprint};
  const double    R       = w.spec.radius;
  const double    largest = kGullyLargest * R;
  const FacePoint on      = locate(p.direction);
  // the land's fall as broad as the widest gullies' cells, and finer for the finer
  int          broad = slopeLevelFor(map, R, kGullySlope * kGullyStripes * largest), read = -1;
  glm::dvec3   land(0.0), sides(0.0);  // the land's rise, and what the gullies so far add to it
  double       steep = 0.0;
  const double finest = kGullyFloor * glm::length(mapSlope(map, on, p.direction, 0));
  glm::dvec3 q     = p.direction * (R / (largest * kGullyStripes));
  glm::dmat3 turned(1.0);  // the lattice's axes against the planet's
  double     sum = 0.0, most = 0.0, gain = 1.0;
  uint32_t   octave = 0;
  for(double size = largest; size >= kGullySmallest; size *= 0.5, octave++, broad--, gain *= kGullyGain, q = nextOctave(q), turned = octaveTurn() * turned)
  {
    // (a stripe needs four points across to be drawn at all)
    const double k = smooth(4.0 * p.footprint, 8.0 * p.footprint, size);
    if(k <= 0.0)
      break;  // nor any finer
    if(const int from = std::max(broad, 0); from != read)
    {
      read  = from;
      land  = mapSlope(map, on, p.direction, from);
      steep = std::min(glm::length(land), finest);
    }
    // level across the rise, in the lattice's axes
    const glm::dvec3 across = glm::cross(p.direction, land + sides);
    const double     run    = glm::length(across);
    if(run < 1e-9 || steep < 1e-9)
      continue;
    const double side = kGullySide * std::min(steep, kGullySteep) * smooth(kGullyFrom, kGullyFull, steep) * level * gain;
    if(side <= 0.0)
      continue;
    const glm::dvec3 level_ = across / run;
    const glm::dvec3 t      = turned * level_ * (2.0 * kPi * kGullyStripes);
    // the eight cells whose middles stand round the point, each a wave e^(i(from.t + its own phase)):
    // the first's turn, and what a step of a cell along each axis takes off it (four turns, not eight)
    const glm::dvec3 f    = glm::floor(q - 0.5);
    const glm::dvec3 o    = q - f - 0.5;
    const auto       by   = [](const glm::dvec2& a, const glm::dvec2& b) { return glm::dvec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); };
    const glm::dvec2 stepX = turnOf(-t.x), stepY = turnOf(-t.y), stepZ = turnOf(-t.z);
    glm::dvec2       turns[8];
    turns[0] = turnOf(glm::dot(o, t));
    turns[1] = by(turns[0], stepX);
    turns[2] = by(turns[0], stepY);
    turns[3] = by(turns[1], stepY);
    for(int n = 0; n < 4; n++)
      turns[n + 4] = by(turns[n], stepZ);
    const uint32_t   ix = uint32_t(int64_t(f.x)), iy = uint32_t(int64_t(f.y)), iz = uint32_t(int64_t(f.z));
    const uint32_t   base = lattice(seedOf(w), kChGully + octave);
    glm::dvec2       wave(0.0);
    double           total = 0.0;
    for(int n = 0; n < 8; n++)
    {
      const int        dx = n & 1, dy = (n >> 1) & 1, dz = n >> 2;
      const glm::dvec3 from = o - glm::dvec3(dx, dy, dz);
      const double     far  = glm::dot(from, from);
      if(far >= 1.0)
        continue;  // (one at least is nearer: the point is in the cube of their middles)
      const double weight = (1.0 - far) * (1.0 - far);
      const uint32_t h = mix(base ^ ((ix + uint32_t(dx)) * 73856093U) ^ ((iy + uint32_t(dy)) * 19349663U) ^ ((iz + uint32_t(dz)) * 83492791U));
      wave += weight * by(turns[n], gullyTurns()[h >> 24U]);
      total += weight;
    }
    const double c = wave.x / total, sn = wave.y / total;
    // a crest where the phase is nothing, a floor half a turn on: a wave, and a roof's pitch
    const double length = std::sqrt(c * c + sn * sn);
    const double pitch  = length * (1.0 - 2.0 * halfTurns(sn, c));
    const double height = 0.25 * side * size;  // metres, crest over the middle
    sum += height * k * (c + (pitch - c) * kGullySharp);
    most += height * k;
    // its sides, added to the rise the finer ones run down
    sides -= level_ * (kGullyBranch * 0.5 * kPi * side * k * sn);
  }
  // none by the sea, whose shore stays where it is
  return sum * smooth(1.0, 1.0 + 2.0 * most, s.elevation);
}

namespace {

// (nor in a lake's cells, whose shore is theirs. A river's banks have them: they run into it, and its
// level is the land's with them, BaseMap's bake)
void gullies(const World& w, const Point& p, Sample& s)
{
  s.elevation += gulliesAt(w, *w.base, p.direction, p.footprint, s) * inlandRelief(s.inland, false);
}

// The rivers' beds and banks cut into the ground, and the water over it. After the ground's fine
// relief: a river's bed has none of it.
void water(const World&, const Point& p, Sample& s)
{
  if(!s.inland.any())
    return;
  // Where the ground is under the sea, a river is the sea: the map's cells are too coarse to say
  // where a shore's pools lie, and a river a little over the sea's level would stand out of them.
  if(s.inland.river)
    s.inland.riverLevel *= smooth(-0.5, 1.0, s.elevation);
  s.elevation = inlandGround(s.inland, s.elevation, p.footprint);
  s.water     = inlandSheet(s.inland, p.footprint);
  s.painted   = inlandPainted(s.inland, p.footprint);
}

// The climate: warm at the equator and cold at the poles and on the heights; wet where the wind
// brings the sea's rain (the base map's; with none, by the sea and where a noise has it so) and by
// a river or a lake
void climate(const World& w, const Point& p, Sample& s)
{
  const double warm = fbm3(p.direction * 5.0, seedOf(w), kChWarm, 2, finestOf(w, p, 5.0)) - 0.5;
  s.temperature     = float(temperatureAt(w, p.direction, s.elevation) + 6.0 * warm);
  const double wet   = fbm3(p.direction * 3.1 + glm::dvec3(-4.0, 2.5, 9.0), seedOf(w), kChWet, 3, finestOf(w, p, 3.1));
  if(w.base)
  {
    const double waterside = inlandNear(s.inland);
    s.moisture             = float(std::clamp(double(mapRain(*w.base, p.direction)) + 0.3 * (wet - 0.5) + 0.25 * waterside, 0.0, 1.0));
  }
  else
    s.moisture = float(std::clamp(0.9 * wet - 0.05 + 0.35 * std::exp(-1.5 * std::max(s.continent, 0.0)), 0.0, 1.0));
}

constexpr float kBiomeSoftTemperature = 2.5F;   // degrees either side of a boundary over which two biomes blend
constexpr float kBiomeSoftMoisture    = 0.06F;

float within(float x, float from, float to, float soft)
{
  return float(smooth(from - soft, from + soft, x) * (1.0 - smooth(to - soft, to + soft, x)));
}

// The biome by the climate: each row's share, and the ground's colour and roughness as their blend
void biome(const World& w, const Point& p, Sample& s)
{
  // (a wood's trees are all drawn on quads of up to some 4 m a vertex, and fewer of them as the quads
  // coarsen: fv_world_plants.hpp reachOf; with a canopy for those not drawn, its own colour stands
  // over the floor, groundMesh's lift)
  const float canopy = w.planted ? (w.canopied ? 0.0F : float(smooth(4.0, 12.0, p.footprint))) : 1.0F;
  glm::vec3 colour(0.0F);
  float     rough = 0.0F, total = 0.0F, most = 0.0F;
  for(size_t i = 0; i < w.biomes.size(); i++)
  {
    const Biome& b     = w.biomes[i];
    const float  share = within(s.temperature, b.temperatureFrom, b.temperatureTo, kBiomeSoftTemperature)
                         * within(s.moisture, b.moistureFrom, b.moistureTo, kBiomeSoftMoisture);
    colour += share * (b.floor.x >= 0.0F ? glm::mix(b.floor, b.colour, canopy) : b.colour);
    rough += share * b.rough;
    total += share;
    if(i < kMostBiomes)
      s.share[i] = share;
    if(share > most)
    {
      most    = share;
      s.biome = int(i);
    }
  }
  if(total > 0.0F)
  {
    s.colour = colour / total;
    s.rough  = rough / total;
    for(float& share : s.share)
      share /= total;
  }
  // a mountain's heights are rock whatever grows under them, and a cliff's face
  s.rock = std::max(float(smooth(0.3, 0.65, double(s.mountain))), s.cliff);
}

// The ground's fine relief: bumps from a few hundred metres across down to what draws it, each as
// high as a share of its width. In metres, not shares of the planet: the ground underfoot is the same
// on a small planet and a large one.
constexpr double kDetailLargest = 256.0;  // metres across the largest bump
constexpr double kDetailSmallest = 2.0;
constexpr double kDetailHeight   = 0.03;  // a bump's height over its width, at roughness 1
constexpr double kDetailByWater  = 0.25;  // of that, at a river's or a lake's edge

void detail(const World& w, const Point& p, Sample& s)
{
  if(w.spec.relief <= 0.0)
    return;
  // (the sea's floor is half as rough as the land, eased in across the shore: switched at the sea's
  // level, the ground stepped by half a bump's height along that line, metres of wall behind a
  // beach. Nothing that changes within a few metres may scale the bumps: they are up to 256 m across
  // and metres high, and a cliff's face made three times as rough stepped as the shore did.)
  const double amount = kDetailHeight * double(s.rough) * std::min(w.spec.relief, 1.5) * (0.5 + 0.5 * smooth(-3.0, 3.0, s.elevation))
                        * (kDetailByWater + (1.0 - kDetailByWater) * inlandRelief(s.inland));  // (a river's level is the land's without them)
  glm::dvec3   q      = p.direction * (w.spec.radius / kDetailLargest);
  uint32_t     octave = 0;
  for(double size = kDetailLargest; size >= kDetailSmallest; size *= 0.5, octave++)
  {
    const double k = smooth(2.0 * p.footprint, 4.0 * p.footprint, size);
    if(k <= 0.0)
      break;  // nor any finer
    s.elevation += 2.0 * (valueNoise3(q, lattice(seedOf(w), kChDetail + octave)) - 0.5) * amount * size * k;
    q = nextOctave(q);
  }
}

const glm::vec3 kSand{0.42F, 0.36F, 0.25F};
const glm::vec3 kGravel{0.16F, 0.145F, 0.115F};    // a river's bed, a lake's shallows
const glm::vec3 kRiverSeen{0.03F, 0.05F, 0.06F};   // a river from too far to draw its water
const glm::vec3 kSeaFloorDeep{0.07F, 0.08F, 0.07F};
const glm::vec3 kRock{0.19F, 0.175F, 0.16F};
constexpr double    kBeach = 2.5;  // metres over the sea a beach reaches

// Whether a lake's or a river's water stands over the ground here (water this little over the sea
// is the sea: its mesh goes under it there)
bool underInlandWater(const Sample& s)
{
  return s.water > s.elevation && s.water > kSeaMeets;
}

// The ground's colour: its biome's, in lighter and darker patches, sand at the shore and on the
// sea's floor
void colour(const World& w, const Point& p, Sample& s)
{
  // patches a hundred metres across, and others the size of a valley
  const double patch = fbm3(p.direction * (w.spec.radius / 90.0), seedOf(w), kChPatch, 2, p.footprint / 90.0);
  const double shade = fbm3(p.direction * 40.0, seedOf(w), kChShade, 2, finestOf(w, p, 40.0));
  s.colour *= float((0.85 + 0.3 * patch) * (0.9 + 0.2 * shade));
  // (a deep lake's bed may lie under the sea's level, and is the lake's all the same)
  if(s.elevation < 0.0 && !underInlandWater(s))
  {
    s.colour = glm::mix(kSand, kSeaFloorDeep, float(smooth(0.0, 40.0, -s.elevation)));
    s.rock   = 0.0F;
    s.sand   = 1.0F;
    return;
  }
  if(s.water > s.elevation)
  {
    // a river's or a lake's bed: gravel in the shallows, dark mud under a few metres
    const float deep = float(smooth(0.0, 6.0, s.water - s.elevation));
    s.colour         = glm::mix(kGravel, 0.6F * kSeaFloorDeep, deep);
    s.rock           = 0.0F;
    s.sand           = 1.0F;
    s.pebbles        = 1.0F - deep;
    return;
  }
  // a beach no narrower than what draws it
  const double top = kBeach + 0.015 * p.footprint;
  s.sand           = float(1.0 - smooth(0.6 * top, top, s.elevation));
  s.colour         = glm::mix(s.colour, kSand, s.sand);
  // damp gravel by a river or a lake, and a river too narrow to draw as water
  if(s.shore < 2.5F)
  {
    const float damp = 0.5F * float(1.0 - smooth(0.0, 2.5, double(s.shore)));
    s.colour         = glm::mix(s.colour, kGravel, damp);
    s.sand           = std::max(s.sand, damp);
    s.pebbles        = damp;
  }
  if(s.painted > 0.0F)
    s.colour = glm::mix(s.colour, kRiverSeen, s.painted);
  // a road's surface, in the patches it was laid and mended in; a track's is earth, drawn as sand is
  if(s.built.road > 0.0F)
  {
    s.colour = glm::mix(s.colour, s.built.colour * float(0.86 + 0.28 * shade), s.built.road);
    if(s.built.earth)
      s.sand = std::max(s.sand, s.built.road);
  }
}

// Level where the figures stand and as far as the scene's props reach, easing into the land round it.
// Level as the scene's floor is: on the plane that touches the planet at the site, which the sphere
// falls away from (a hand's breadth a hundred metres out on a planet 20 km in radius, and a wall
// that stands on the floor would stand that far over the ground).
void clearing(const World& w, const Point& p, Sample& s)
{
  const double level = std::max(w.spec.clearing, kClearing), edge = std::max(kClearingEdge, 0.5 * level);
  const double away  = glm::length(p.direction - w.frame.site) * w.spec.radius;
  if(away >= level + edge)
    return;
  const double flat = (w.spec.radius + w.frame.siteElevation) / std::max(glm::dot(p.direction, w.frame.site), 0.5) - w.spec.radius;
  s.elevation       = flat + (s.elevation - flat) * smooth(level, level + edge, away);
}

std::vector<Biome> biomeTable()
{
  // (above 5 degrees the rows are a ladder by how wet it is, and must leave no gap)
  // What grows: each row its own of the kit's plants (fv_world_plants.hpp). A wood is firs with
  // stands of pine and a few leafy trees; a grassland leafy trees with firs at its edges; the steppe
  // dry bushes, lone pines and thorn trees; the desert succulents, quiver trees and bare boulders; the
  // tundra low bushes, stones and stunted pines at its warm edge. A fir wood's floor and a meadow's
  // drifts are the warm, wet rows' alone. Along rivers and round lakes, willows and alders (riverside):
  // a belt broken into stands in the grassland and the wood, a wider one on the steppe, where they
  // are its only leafy trees, and a few at the desert's water.
  return {
      // (what each one's ground is drawn with: soil with grass on it in the grassland, a wood's floor
      // under the trees, stony earth in the dry lands, soil and stones in the tundra, snow)
      {.name = "snow", .temperatureTo = -5.0F, .colour = {0.78F, 0.80F, 0.84F}, .rough = 0.5F, .soil = Soil::eSnow},
      {.name = "tundra", .temperatureFrom = -5.0F, .temperatureTo = 5.0F, .colour = {0.17F, 0.16F, 0.11F}, .rough = 1.0F,
       .plants = {{.role = "rock", .perSquareMetre = 0.006, .scaleMin = 0.15F, .scaleMax = 0.8F, .sink = 0.1F, .cover = 0.5F, .patch = 40.0F},
                  {.role = "boulder", .perSquareMetre = 0.006, .scaleMin = 0.2F, .scaleMax = 0.9F, .sink = 0.12F, .cover = 0.4F, .patch = 50.0F},
                  {.role = "shrub", .perSquareMetre = 0.006, .scaleMin = 0.4F, .scaleMax = 0.8F, .sink = 0.03F, .cover = 0.25F, .patch = 30.0F},
                  {.role = "pine", .perSquareMetre = 0.0008, .scaleMin = 0.45F, .scaleMax = 0.9F, .sink = 0.1F, .cover = 0.15F, .patch = 120.0F}},
       .grass  = Grass::eCold,
       .soil   = Soil::eStony},
      {.name = "desert", .temperatureFrom = 5.0F, .moistureTo = 0.15F, .colour = {0.40F, 0.30F, 0.17F}, .rough = 0.6F,
       .plants = {{.role = "boulder", .perSquareMetre = 0.008, .scaleMin = 0.2F, .scaleMax = 1.2F, .sink = 0.12F, .cover = 0.35F, .patch = 70.0F},
                  {.role = "drybush", .perSquareMetre = 0.006, .scaleMin = 0.6F, .scaleMax = 1.3F, .sink = 0.03F, .cover = 0.3F, .patch = 45.0F},
                  {.role = "drytree", .perSquareMetre = 0.0012, .scaleMin = 0.8F, .scaleMax = 1.6F, .sink = 0.05F, .cover = 0.2F, .patch = 150.0F},
                  {.role = "riverside", .perSquareMetre = 0.006, .scaleMin = 0.7F, .scaleMax = 1.2F, .sink = 0.1F, .cover = 0.5F, .patch = 60.0F, .water = 15.0F}},
       .soil   = Soil::eDry},
      {.name = "steppe", .temperatureFrom = 5.0F, .moistureFrom = 0.15F, .moistureTo = 0.36F, .colour = {0.26F, 0.21F, 0.12F}, .rough = 0.9F,
       .plants = {{.role = "drybush", .perSquareMetre = 0.008, .scaleMin = 0.6F, .scaleMax = 1.2F, .sink = 0.03F, .cover = 0.3F, .patch = 35.0F},
                  {.role = "shrub", .perSquareMetre = 0.003, .scaleMin = 0.5F, .scaleMax = 1.1F, .sink = 0.03F, .cover = 0.3F, .patch = 35.0F},
                  {.role = "boulder", .perSquareMetre = 0.004, .scaleMin = 0.2F, .scaleMax = 0.9F, .sink = 0.1F, .cover = 0.3F, .patch = 60.0F},
                  {.role = "pine", .perSquareMetre = 0.003, .scaleMin = 0.8F, .scaleMax = 1.5F, .sink = 0.15F, .cover = 0.15F, .patch = 90.0F},
                  {.role = "drytree", .perSquareMetre = 0.002, .scaleMin = 0.8F, .scaleMax = 1.5F, .sink = 0.05F, .cover = 0.2F, .patch = 70.0F},
                  {.role = "riverside", .perSquareMetre = 0.015, .scaleMin = 0.8F, .scaleMax = 1.4F, .sink = 0.1F, .cover = 0.75F, .patch = 80.0F, .water = 35.0F}},
       .grass  = Grass::eDry,
       .soil   = Soil::eDry},
      {.name = "grassland", .temperatureFrom = 5.0F, .moistureFrom = 0.36F, .moistureTo = 0.68F, .colour = {0.085F, 0.14F, 0.045F}, .rough = 0.8F,
       .plants = {{.role = "broadleaf", .perSquareMetre = 0.012, .scaleMin = 1.2F, .scaleMax = 2.4F, .sink = 0.1F, .cover = 0.3F, .patch = 80.0F},
                  {.role = "tree", .perSquareMetre = 0.008, .scaleMin = 0.8F, .scaleMax = 1.9F, .sink = 0.15F, .cover = 0.25F, .patch = 110.0F},
                  {.role = "sapling", .perSquareMetre = 0.02, .scaleMin = 0.7F, .scaleMax = 1.6F, .cover = 0.35F, .patch = 30.0F},
                  {.role = "shrub", .perSquareMetre = 0.012, .scaleMin = 0.6F, .scaleMax = 1.3F, .sink = 0.03F, .cover = 0.3F, .patch = 35.0F},
                  {.role = "rock", .perSquareMetre = 0.004, .scaleMin = 0.2F, .scaleMax = 0.8F, .sink = 0.1F, .cover = 0.3F, .patch = 60.0F},
                  {.role = "riverside", .perSquareMetre = 0.02, .scaleMin = 0.9F, .scaleMax = 1.6F, .sink = 0.1F, .cover = 0.7F, .patch = 70.0F, .water = 30.0F}},
       .grass  = Grass::eMeadow,
       .drift  = "flower",
       .soil   = Soil::eGrass},
      {.name = "forest", .temperatureFrom = 5.0F, .moistureFrom = 0.68F, .colour = {0.045F, 0.085F, 0.03F}, .rough = 0.9F,
       .floor  = {0.075F, 0.06F, 0.035F},
       .plants = {{.role = "tree", .perSquareMetre = 0.018, .scaleMin = 1.0F, .scaleMax = 2.4F, .sink = 0.2F, .cover = 0.7F, .patch = 60.0F},
                  {.role = "pine", .perSquareMetre = 0.012, .scaleMin = 1.0F, .scaleMax = 1.8F, .sink = 0.2F, .cover = 0.55F, .patch = 90.0F},
                  {.role = "broadleaf", .perSquareMetre = 0.003, .scaleMin = 1.3F, .scaleMax = 2.4F, .sink = 0.1F, .cover = 0.3F, .patch = 50.0F},
                  {.role = "sapling", .perSquareMetre = 0.03, .scaleMin = 0.6F, .scaleMax = 1.7F, .cover = 0.55F, .patch = 18.0F},
                  {.role = "shrub", .perSquareMetre = 0.05, .scaleMin = 0.5F, .scaleMax = 1.2F, .sink = 0.03F, .cover = 0.45F, .patch = 22.0F},
                  {.role = "rock", .perSquareMetre = 0.02, .scaleMin = 0.15F, .scaleMax = 0.7F, .sink = 0.08F, .cover = 0.5F, .patch = 30.0F},
                  {.role = "stump", .perSquareMetre = 0.008, .scaleMin = 0.6F, .scaleMax = 1.2F, .sink = 0.03F},
                  {.role = "log", .perSquareMetre = 0.012, .scaleMin = 0.7F, .scaleMax = 1.5F},
                  {.role = "riverside", .perSquareMetre = 0.015, .scaleMin = 0.9F, .scaleMax = 1.5F, .sink = 0.1F, .cover = 0.7F, .patch = 60.0F, .water = 20.0F}},
       .grass  = Grass::eWood,
       .drift  = "fern",
       .soil   = Soil::eFloor},
  };
}

// ---- the sphere --------------------------------------------------------------------------------------

// `count` directions spread evenly over the sphere
glm::dvec3 spread(uint32_t i, uint32_t count)
{
  const double y   = 1.0 - 2.0 * (double(i) + 0.5) / double(count);
  const double r   = std::sqrt(std::max(1.0 - y * y, 0.0));
  const double phi = double(i) * 2.399963229728653;  // the golden angle
  return {r * std::cos(phi), y, r * std::sin(phi)};
}

// Two directions square to `d` and to each other
void tangents(const glm::dvec3& d, glm::dvec3& a, glm::dvec3& b)
{
  a = glm::normalize(glm::cross(std::abs(d.y) < 0.9 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), d));
  b = glm::cross(d, a);
}

struct Axes
{
  glm::dvec3 n, s, t;  // out of the face, and along its u and v: s x t = n
};
const Axes& axesOf(uint8_t face)
{
  static const std::array<Axes, 6> kAxes = {{
      {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
      {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
      {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
      {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
      {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
      {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},
  }};
  return kAxes[face < 6 ? face : 0];
}

// Where on a quad's edge (u or v, -1 to 1) its vertex `i` of `cells` is
double coordOf(const Quad& q, uint32_t tile, int i, uint32_t cells)
{
  return (double(tile) * double(cells) + double(i)) * (2.0 / (double(cells) * double(uint64_t(1) << q.level))) - 1.0;
}

// The stages but the clearing: the land as it is before the figures' place is levelled
Sample sampleUpTo(const World& world, const glm::dvec3& direction, double footprint, size_t stages)
{
  const Point p{direction, std::max(footprint, 1e-3)};
  Sample      s;
  for(size_t i = 0; i < stages && i < world.stages.size(); i++)
    world.stages[i].apply(world, p, s);
  return s;
}

// Where the figures stand: a level place on low, temperate land, away from the shore. Of a few
// thousand places spread over the planet, the most level of those that pass; none passing (a planet
// nearly all sea), the highest.
glm::dvec3 findSite(const World& w)
{
  constexpr uint32_t kPlaces = 3000;
  const double       R       = w.spec.radius;
  const uint32_t     first   = mix(lattice(seedOf(w), kChSite)) % kPlaces;
  glm::dvec3         best(0.0, 1.0, 0.0), highest(0.0, 1.0, 0.0);
  double             bestSlope = std::numeric_limits<double>::max(), top = std::numeric_limits<double>::lowest();
  const size_t       stages = w.stages.size();
  for(uint32_t k = 0; k < kPlaces; k++)
  {
    const glm::dvec3 d = spread((first + k) % kPlaces, kPlaces);
    const Sample     s = sampleUpTo(w, d, 8.0, stages);
    if(s.elevation > top)
    {
      top     = s.elevation;
      highest = d;
    }
    if(s.elevation < 3.0 || s.elevation > 0.004 * R + 12.0 || s.temperature < 8.0F || std::abs(d.y) > 0.7 || s.shore < 60.0F)
      continue;
    glm::dvec3 a, b;
    tangents(d, a, b);
    // level, and dry, all round: over the clearing's edge, over the ground a figure's shot shows
    // round it, and no shore within a few hundred metres (or a tenth of a very small planet's radius)
    double slope = 0.0;
    bool   dry   = true;
    for(const double away : {30.0, 60.0, 120.0, std::min(250.0, 0.1 * R)})
      for(int turn = 0; turn < 8 && dry; turn++)
      {
        const double     angle = double(turn) * (kPi / 4.0);
        const glm::dvec3 side  = a * std::cos(angle) + b * std::sin(angle);
        const Sample     there = sampleUpTo(w, glm::normalize(d + side * (away / R)), 8.0, stages);
        const double     e     = there.elevation;
        // (nor a cliff: all of it on the tier the place itself is on)
        dry                    = dry && e > 1.0 && there.shore > 20.0F && std::abs(there.tiers - s.tiers) < 0.02F;
        if(away <= 120.0)
          slope = std::max(slope, std::abs(e - s.elevation) / away);
      }
    if(dry && slope < bestSlope)
    {
      bestSlope = slope;
      best      = d;
    }
  }
  return bestSlope < std::numeric_limits<double>::max() ? best : highest;
}

// The rotation that takes `from` to +y by the shortest way
glm::dquat toUp(const glm::dvec3& from)
{
  const double c = from.y;  // its cosine with +y
  if(c < -1.0 + 1e-12)
    return {0.0, 1.0, 0.0, 0.0};  // straight down: half a turn about x
  const glm::dvec3 axis = glm::cross(from, glm::dvec3(0.0, 1.0, 0.0));
  return glm::normalize(glm::dquat(1.0 + c, axis.x, axis.y, axis.z));
}

// A quad's skirt: each edge vertex again, `depth` nearer the planet's centre, and the wall between
// `out`, when given: the surface's vertices that are no part of it; a wall between two is left out
void addSkirts(ChunkMesh& m, const World& w, uint32_t cells, double depth, const std::vector<uint8_t>* out = nullptr)
{
  const uint32_t side  = cells + 1;
  const auto     at    = [&](uint32_t i, uint32_t j) { return j * side + i; };
  const auto     lower = [&](uint32_t v) {
    const glm::dvec3 p      = m.centre + glm::dvec3(m.positions[v]);
    const glm::dvec3 inward = glm::normalize(w.frame.centre - p);
    m.positions.push_back(glm::vec3(glm::dvec3(m.positions[v]) + inward * depth));
    m.normals.push_back(m.normals[v]);
    m.colors.push_back(m.colors[v]);
    if(!m.layers.empty())
      m.layers.push_back(m.layers[v]);
    if(!m.layersMore.empty())
      m.layersMore.push_back(m.layersMore[v]);
    return uint32_t(m.positions.size() - 1);
  };
  // the edges in turn, each walked so the wall faces out of the quad
  const auto edge = [&](auto vertexAt) {
    uint32_t last = 0, lastLow = 0;
    for(uint32_t k = 0; k < side; k++)
    {
      const uint32_t v = vertexAt(k), low = lower(v);
      if(k > 0 && !(out && (*out)[v] && (*out)[last]))
        m.indices.insert(m.indices.end(), {last, lastLow, low, last, low, v});
      last    = v;
      lastLow = low;
    }
  };
  edge([&](uint32_t k) { return at(k, 0); });
  edge([&](uint32_t k) { return at(cells, k); });
  edge([&](uint32_t k) { return at(cells - k, cells); });
  edge([&](uint32_t k) { return at(0, cells - k); });
}

// `out`, when given: the vertices that are no part of the surface; a triangle of three is left out.
// A cell is cut along the shorter of its diagonals: where a crest or a foot crosses it, that is the
// one along the crease. Cut all one way, every cell a cliff's edge crossed against its diagonal was
// folded the wrong way, and the crest was a row of teeth a metre wide and as high as the cliff is
// steep.
void addGrid(ChunkMesh& m, uint32_t cells, const std::vector<uint8_t>* out = nullptr)
{
  const uint32_t side = cells + 1;
  m.indices.reserve(out ? 0 : size_t(cells) * cells * 6 + size_t(cells) * 24);
  const auto triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
    if(!(out && (*out)[a] && (*out)[b] && (*out)[c]))
      m.indices.insert(m.indices.end(), {a, b, c});
  };
  const auto apart = [&](uint32_t a, uint32_t b) {
    const glm::vec3 d = m.positions[a] - m.positions[b];
    return glm::dot(d, d);
  };
  for(uint32_t j = 0; j < cells; j++)
    for(uint32_t i = 0; i < cells; i++)
    {
      const uint32_t a = j * side + i, b = a + 1, c = a + side, d = c + 1;
      if(apart(a, d) <= apart(b, c))
      {
        triangle(a, b, d);
        triangle(a, d, c);
      }
      else
      {
        triangle(a, b, c);
        triangle(b, d, c);
      }
    }
}

}  // namespace

// ---- the planet --------------------------------------------------------------------------------------

World withClearing(const World& world, double clearing)
{
  World w         = world;
  w.spec.clearing = std::clamp(clearing, kClearing, mostLevel(w.spec.radius));
  return w;
}

World make(const Spec& asked, const Making& making)
{
  World w;
  w.spec        = asked;
  w.spec.radius = std::clamp(asked.radius, 2000.0, 100000.0);
  w.spec.sea    = std::clamp(asked.sea, 0.0, 0.9);
  w.spec.relief = std::clamp(asked.relief, 0.0, 2.0);
  w.spec.clearing = std::clamp(asked.clearing, kClearing, mostLevel(w.spec.radius));
  w.biomes      = biomeTable();

  // The sea's level: the continents' noise at the share of the sphere the sea is to cover
  constexpr uint32_t  kPlaces = 4096;
  std::vector<double> levels(kPlaces);
  double              mean = 0.0;
  for(uint32_t i = 0; i < kPlaces; i++)
  {
    levels[i] = continentNoise(w.spec, spread(i, kPlaces), 0.0);
    mean += levels[i] / double(kPlaces);
  }
  double variance = 0.0;
  for(const double l : levels)
    variance += (l - mean) * (l - mean) / double(kPlaces);
  w.seaSpread = std::max(std::sqrt(variance), 1e-6);
  std::sort(levels.begin(), levels.end());
  w.seaLevel = w.spec.sea > 0.0 ? levels[std::min(size_t(w.spec.sea * double(kPlaces)), size_t(kPlaces - 1))]
                                : levels.front() - w.seaSpread;  // no sea: the lowest land is well over it

  // The land's broad shape, then what the base map made of it (a flat planet has nothing to wear)
  w.stages = {{"continents", continents}, {"mountains", mountains}, {"hills", hills}, {"cliffs", cliffs}};
  if(making.base && w.spec.relief > 0.0)
    w.base = bake(w, making.map ? making.map : mapCells(w.spec.radius), making.stop, making.floors);
  if(w.base)
    w.stages.insert(w.stages.end(), {{"base", base}, {"gullies", gullies}});
  w.stages.insert(w.stages.end(), {{"climate", climate}, {"biome", biome}, {"detail", detail}});
  if(w.base)
    w.stages.push_back({"water", water});
  w.stages.push_back({"colour", colour});

  // where the figures stand, the origin
  // (the author's place for them, where that is dry land)
  w.frame.site = findSite(w);
  if(w.spec.features && w.spec.features->site && sampleUpTo(w, *w.spec.features->site, 8.0, w.stages.size()).elevation > 1.0)
    w.frame.site = *w.spec.features->site;
  w.frame.siteElevation = sampleUpTo(w, w.frame.site, 1.0, w.stages.size()).elevation;
  w.frame.turn          = toUp(w.frame.site);
  w.frame.centre        = glm::dvec3(0.0, -(w.spec.radius + w.frame.siteElevation), 0.0);
  w.stages.push_back({"clearing", clearing});

  // what it reflects seen whole: the land's colours and the sea's
  const glm::vec3     kSeaAlbedo{0.02F, 0.04F, 0.06F};
  glm::vec3           albedo(0.0F);
  constexpr uint32_t  kSeen = 512;
  for(uint32_t i = 0; i < kSeen; i++)
  {
    const Sample s = sample(w, spread(i, kSeen), w.spec.radius * 0.1);
    albedo += (s.elevation < 0.0 ? kSeaAlbedo : s.colour) / float(kSeen);
  }
  w.albedo = albedo;
  return w;
}

double noise(const World& world, const glm::dvec3& direction, double across, uint32_t channel)
{
  return fbm3(direction * (world.spec.radius / std::max(across, 1e-3)), uint32_t(world.spec.seed), 1000U + 10U * channel, 2);
}

double temperatureAt(const World& world, const glm::dvec3& direction, double elevation)
{
  // the highest land of an even relief is some 40 degrees colder than the shore under it
  const double lapse = 40.0 / (0.045 * world.spec.radius);
  return 30.0 - 48.0 * std::pow(std::abs(direction.y), 1.5) - lapse * std::max(elevation, 0.0);
}

Sample sample(const World& world, const glm::dvec3& direction, double footprint)
{
  return sampleUpTo(world, direction, footprint, world.stages.size());
}

Sample sampleWithout(const World& world, const glm::dvec3& direction, double footprint, std::string_view stage)
{
  const Point p{direction, std::max(footprint, 1e-3)};
  Sample      s;
  for(const Stage& each : world.stages)
    if(each.name != stage)
      each.apply(world, p, s);
  return s;
}

namespace {

// The share of bare rock where the ground slopes (rise over run): from where a slope is too steep for
// soil, to a cliff's bare face; not under water
float rockShare(const Sample& s, float slope)
{
  const float steep = s.elevation < 0.0 || s.water > s.elevation ? 0.0F : float(smooth(0.45, 0.9, double(slope)));
  return std::max(steep, s.rock);
}

}  // namespace

glm::vec3 surfaceColour(const World&, const Sample& s, float slope)
{
  return glm::mix(s.colour, kRock, rockShare(s, slope));
}

std::array<float, kGroundLayers> surfaceLayers(const World& world, const Sample& s, float slope)
{
  std::array<float, kGroundLayers> layers{};
  // the biomes' soils, by their shares; then sand over them, and rock over all
  for(size_t i = 0; i < world.biomes.size() && i < kMostBiomes; i++)
    if(const size_t soil = size_t(world.biomes[i].soil); soil < kGroundLayers)
      layers[soil] += s.share[i];
  const float rock = rockShare(s, slope);
  const float sand = std::clamp(s.sand, 0.0F, 1.0F);
  for(float& layer : layers)
    layer *= (1.0F - sand) * (1.0F - rock);
  // of the sand, a track's earth is the dry lands', and a bed's shallows and a damp shore pebbles
  const float earth   = s.built.earth ? std::min(std::clamp(s.built.road, 0.0F, 1.0F), sand) : 0.0F;
  const float pebbles = std::min(std::clamp(s.pebbles, 0.0F, 1.0F), sand - earth);
  layers[size_t(Soil::eSand)] += (sand - earth - pebbles) * (1.0F - rock);
  layers[size_t(Soil::eDry)] += earth * (1.0F - rock);
  layers[size_t(Soil::ePebbles)] += pebbles * (1.0F - rock);
  layers[size_t(Soil::eRock)] += rock;
  // asphalt is none of them: the ground's own material in the road's colour
  if(s.built.road > 0.0F && !s.built.earth)
    for(float& layer : layers)
      layer *= 1.0F - std::clamp(s.built.road, 0.0F, 1.0F);
  return layers;
}

// ---- the cube on the sphere --------------------------------------------------------------------------

glm::dvec3 direction(uint8_t face, double u, double v)
{
  const Axes& a = axesOf(face);
  return glm::normalize(a.n + a.s * std::tan(u * (kPi / 4.0)) + a.t * std::tan(v * (kPi / 4.0)));
}

FacePoint locate(const glm::dvec3& d)
{
  const glm::dvec3 m = glm::abs(d);
  FacePoint        out;
  if(m.x >= m.y && m.x >= m.z)
    out.face = d.x >= 0.0 ? 0 : 1;
  else if(m.y >= m.z)
    out.face = d.y >= 0.0 ? 2 : 3;
  else
    out.face = d.z >= 0.0 ? 4 : 5;
  const Axes&  a   = axesOf(out.face);
  const double out_of = glm::dot(d, a.n);
  out.u            = std::atan(glm::dot(d, a.s) / out_of) * (4.0 / kPi);
  out.v            = std::atan(glm::dot(d, a.t) / out_of) * (4.0 / kPi);
  return out;
}

FacePoint onFace(uint8_t face, const glm::dvec3& d)
{
  const Axes&  a      = axesOf(face);
  const double out_of = glm::dot(d, a.n);
  if(out_of <= 1e-3)
    return {uint8_t(255), 0.0, 0.0};
  return {face, std::atan(glm::dot(d, a.s) / out_of) * (4.0 / kPi), std::atan(glm::dot(d, a.t) / out_of) * (4.0 / kPi)};
}

bool contains(const Quad& q, const glm::dvec3& d)
{
  const FacePoint p = locate(d);
  if(p.face != q.face)
    return false;
  return p.u >= coordOf(q, q.x, 0, 1) && p.u < coordOf(q, q.x, 1, 1) && p.v >= coordOf(q, q.y, 0, 1) && p.v < coordOf(q, q.y, 1, 1);
}

std::array<Quad, 4> children(const Quad& q)
{
  const uint8_t  l = uint8_t(q.level + 1);
  const uint32_t x = q.x * 2, y = q.y * 2;
  return {{{q.face, l, x, y}, {q.face, l, x + 1, y}, {q.face, l, x, y + 1}, {q.face, l, x + 1, y + 1}}};
}

double width(const World& world, const Quad& q)
{
  return 0.5 * kPi * world.spec.radius / double(uint64_t(1) << q.level);
}

uint8_t finestLevel(const World& world)
{
  const double face = 0.5 * kPi * world.spec.radius;
  return uint8_t(std::clamp(std::ceil(std::log2(face / double(kGroundCells))), 0.0, 24.0));
}

// ---- the planet in the scene -------------------------------------------------------------------------

glm::dvec3 toScene(const World& world, const glm::dvec3& d, double elevation)
{
  return world.frame.turn * (d * (world.spec.radius + elevation)) + world.frame.centre;
}

glm::dvec3 toPlanet(const World& world, const glm::dvec3& point, double* fromCentre)
{
  const glm::dvec3 out = point - world.frame.centre;
  const double     r   = glm::length(out);
  if(fromCentre)
    *fromCentre = r;
  return r > 0.0 ? glm::conjugate(world.frame.turn) * (out / r) : world.frame.site;
}

double heightOver(const World& world, const glm::dvec3& point)
{
  double           r = 0.0;
  const glm::dvec3 d = toPlanet(world, point, &r);
  const double     over = r - world.spec.radius;
  // the ground as something that far away would see it, and then as near as it is
  const auto top = [&](double footprint) {
    const Sample s = sample(world, d, footprint);
    return std::max({s.elevation, s.water, 0.0});
  };
  const double first = over - top(std::max(0.05 * std::abs(over), 1.0));
  return over - top(std::max(0.05 * std::abs(first), 1.0));
}

double waterLevel(const World& world, const glm::dvec3& d, double footprint)
{
  if(!world.base)
    return 0.0;
  return std::max(0.0, sample(world, d, footprint).water);
}

double inlandWaterOver(const World& world, const glm::dvec3& point)
{
  if(!world.base)
    return kDry;
  double           r = 0.0;
  const glm::dvec3 d = toPlanet(world, point, &r);
  const Sample     s = sample(world, d, 1.0);
  if(!underInlandWater(s))
    return kDry;
  return r - world.spec.radius < s.water ? s.water : kDry;
}

glm::dvec3 viewUp(const World& world, const glm::dvec3& eye)
{
  const glm::dvec3 own(0.0, 1.0, 0.0);
  const glm::dvec3 fromCentre = eye - world.frame.centre;
  const double     r          = glm::length(fromCentre);
  if(r < 1e-6)
    return own;
  const double t    = std::clamp((glm::length(eye) / world.spec.radius - kUpOwnFrom) / (kUpOwnTo - kUpOwnFrom), 0.0, 1.0);
  const double ease = t * t * (3.0 - 2.0 * t);
  const glm::dvec3 up = glm::mix(own, fromCentre / r, ease);
  const double     n  = glm::length(up);
  return n > 1e-9 ? up / n : fromCentre / r;
}

double farPlane(const World& world, const glm::dvec3& eye)
{
  return 2.0 * (glm::length(eye - world.frame.centre) + world.spec.radius);
}

std::vector<Ground> nearestGrounds(const World& world, const glm::dvec3& from, double spacing, double fromNear)
{
  const double R = world.spec.radius;
  // the sweep's points, evenly over the sphere (a Fibonacci lattice), nearest first
  const size_t n = size_t(std::clamp(4.0 * kPi * R * R / (spacing * spacing), 64.0, 4.0e5));
  struct Point
  {
    glm::dvec3 d;
    double     away;
  };
  std::vector<Point> points;
  points.reserve(n);
  const double turn = kPi * (3.0 - std::sqrt(5.0));
  for(size_t i = 0; i < n; i++)
  {
    const double     y = 1.0 - 2.0 * (double(i) + 0.5) / double(n);
    const double     r = std::sqrt(std::max(0.0, 1.0 - y * y));
    const glm::dvec3 d(r * std::cos(turn * double(i)), y, r * std::sin(turn * double(i)));
    const double     away = std::acos(std::clamp(glm::dot(d, from), -1.0, 1.0)) * R;
    if(away >= fromNear)
      points.push_back({d, away});
  }
  std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.away < b.away; });

  const size_t                     kinds = std::min(world.biomes.size(), kMostBiomes) + 1;  // the last a beach
  std::vector<std::optional<Ground>> found(kinds);
  size_t                           left = kinds;
  constexpr float                  kAlone = 0.9F;  // the biome's share where it counts nearly alone
  for(const Point& p : points)
  {
    if(left == 0)
      break;
    const Sample s = sample(world, p.d, 0.25 * spacing);
    // dry ground to stand on, a little over the sea, not in a river or a lake nor by one
    if(s.elevation < 1.0 || s.water > s.elevation - 0.5 || s.inland.river || s.inland.lake > 0.05 || s.shore < 50.0F
       || s.rock > 0.3F || s.cliff > 0.1F)
      continue;
    const bool   beach = s.sand > 0.5F && s.elevation < 8.0;
    const size_t kind  = beach ? kinds - 1 : size_t(s.biome);
    if(kind >= kinds || found[kind] || (!beach && s.share[kind] < kAlone))
      continue;
    found[kind] = Ground{beach ? kBeachGround : world.biomes[kind].name, p.d, p.away};
    left--;
  }
  std::vector<Ground> out;
  for(const std::optional<Ground>& g : found)
    if(g)
      out.push_back(*g);
  std::sort(out.begin(), out.end(), [](const Ground& a, const Ground& b) { return a.distance < b.distance; });
  return out;
}

// ---- which quads draw --------------------------------------------------------------------------------

namespace {

// How far the eye (the planet's axes, from its centre) is from a quad's ground
double distanceTo(const World& w, const Quad& q, const glm::dvec3& eye, const glm::dvec3& eyeDirection)
{
  const Axes&  a      = axesOf(q.face);
  const double out_of = glm::dot(eyeDirection, a.n);
  const double size   = width(w, q);
  if(out_of > 0.2)
  {
    // the quad's point nearest under the eye, on the ground as the quad would draw it
    const double     u = std::clamp(std::atan(glm::dot(eyeDirection, a.s) / out_of) * (4.0 / kPi), coordOf(q, q.x, 0, 1), coordOf(q, q.x, 1, 1));
    const double     v = std::clamp(std::atan(glm::dot(eyeDirection, a.t) / out_of) * (4.0 / kPi), coordOf(q, q.y, 0, 1), coordOf(q, q.y, 1, 1));
    const glm::dvec3 d = direction(q.face, u, v);
    const double     e = std::max(sample(w, d, size / 8.0).elevation, 0.0);
    return glm::length(eye - d * (w.spec.radius + e));
  }
  // round the planet from the eye: its middle, less half its width
  const glm::dvec3 middle = direction(q.face, coordOf(q, q.x, 1, 2), coordOf(q, q.y, 1, 2)) * w.spec.radius;
  return std::max(glm::length(eye - middle) - 0.75 * size, 0.0);
}

// The eye (the planet's axes, from its centre), and where it will be kLead on: `led` that far from it
struct Eyes
{
  glm::dvec3 eye, eyeDirection, ahead, aheadDirection;
  double     led = 0.0;
};

void visit(const World& w, const Quad& q, const Eyes& e, uint8_t finest, const Selection* was, Selection& out)
{
  if(q.level < finest)
  {
    const bool   held  = was && std::binary_search(was->split.begin(), was->split.end(), q);
    const double reach = kSplit * width(w, q) * (held ? kSplitHold : 1.0);
    double       away  = distanceTo(w, q, e.eye, e.eyeDirection);
    // (from where it will be, when that can be within reach: no nearer than the way there less)
    if(away >= reach && e.led > 0.0 && away - e.led < reach)
      away = std::min(away, distanceTo(w, q, e.ahead, e.aheadDirection));
    if(away < reach)
    {
      out.split.push_back(q);
      for(const Quad& child : children(q))
        visit(w, child, e, finest, was, out);
      return;
    }
  }
  out.leaves.push_back(q);
}

}  // namespace

Selection select(const World& world, const glm::dvec3& eye, const Selection* was, const glm::dvec3& velocity)
{
  double           r = 0.0;
  const glm::dvec3 d = toPlanet(world, eye, &r);
  Eyes             eyes{.eye = d * r, .eyeDirection = d, .ahead = d * r, .aheadDirection = d};
  const double     speed = glm::length(velocity);
  if(speed > 0.0)
  {
    double           ra = 0.0;
    const glm::dvec3 da = toPlanet(world, eye + velocity * kLead, &ra);
    eyes.ahead          = da * ra;
    eyes.aheadDirection = da;
    eyes.led            = speed * kLead;
  }
  Selection out;
  // (no quad narrower than the eye passes in kPassing)
  uint8_t finest = finestLevel(world);
  while(finest > 0 && width(world, {0, finest, 0, 0}) < speed * kPassing)
    finest--;
  for(uint8_t face = 0; face < 6; face++)
    visit(world, {face, 0, 0, 0}, eyes, finest, was, out);
  std::sort(out.leaves.begin(), out.leaves.end());
  std::sort(out.split.begin(), out.split.end());
  return out;
}

// ---- a quad's triangles ------------------------------------------------------------------------------

ChunkMesh groundMesh(const World& world, const Quad& q, const LiftOf& lift)
{
  constexpr uint32_t N    = kGroundCells;
  constexpr int      G    = int(N) + 3;  // a ring of points round the quad's own, for its edges' normals
  const double       cell = width(world, q) / double(N);
  const auto         at   = [&](int i, int j) { return size_t(j + 1) * size_t(G) + size_t(i + 1); };

  std::vector<glm::dvec3> points(size_t(G) * size_t(G));
  std::vector<Sample>     samples(points.size());
  // with a lift: what stands over each point, and the points as drawn, over it
  std::vector<Lift>       lifts;
  std::vector<glm::dvec3> lifted;
  if(lift)
  {
    lifts.resize(points.size());
    lifted.resize(points.size());
  }
  for(int j = -1; j <= int(N) + 1; j++)
    for(int i = -1; i <= int(N) + 1; i++)
    {
      const glm::dvec3 d = direction(q.face, coordOf(q, q.x, i, N), coordOf(q, q.y, j, N));
      samples[at(i, j)]  = sample(world, d, cell);
      points[at(i, j)]   = toScene(world, d, samples[at(i, j)].elevation);
      if(lift)
      {
        const size_t k = at(i, j);
        lifts[k]       = lift(d, samples[k], cell);
        // (an assignment and an `if`, not a `?:` of the two: built Release by MSVC, that gave every
        // point of a row the ring's point at its start; the WorldCanopy tests catch it)
        lifted[k] = points[k];
        if(lifts[k].height > 0.0F)
          lifted[k] = toScene(world, d, samples[k].elevation + double(lifts[k].height));
      }
    }
  // what is drawn; the ground's own points say its slope, which its colour and layers go by
  const std::vector<glm::dvec3>& drawn = lift ? lifted : points;
  const auto normalOf = [&](const std::vector<glm::dvec3>& of, int i, int j) {
    return glm::normalize(glm::cross(of[at(i + 1, j)] - of[at(i - 1, j)], of[at(i, j + 1)] - of[at(i, j - 1)]));
  };

  ChunkMesh m;
  m.centre  = points[at(int(N) / 2, int(N) / 2)];
  m.lowest  = std::numeric_limits<double>::max();
  m.highest = std::numeric_limits<double>::lowest();
  const size_t count = size_t(N + 1) * size_t(N + 1);
  m.positions.reserve(count + 4 * size_t(N + 1));
  m.normals.reserve(count + 4 * size_t(N + 1));
  m.colors.reserve(count + 4 * size_t(N + 1));
  m.layers.reserve(count + 4 * size_t(N + 1));
  m.layersMore.reserve(count + 4 * size_t(N + 1));
  m.elevations.reserve(count);
  const uint32_t dither = lattice(uint32_t(world.spec.seed), kChDither);
  float          tallest = 0.0F;  // the most the lift raises a vertex
  for(int j = 0; j <= int(N); j++)
    for(int i = 0; i <= int(N); i++)
    {
      const glm::dvec3& p      = points[at(i, j)];
      const Sample&     s      = samples[at(i, j)];
      const glm::dvec3  up     = glm::normalize(p - world.frame.centre);
      const glm::dvec3  normal = normalOf(points, i, j);
      const double      level  = glm::dot(normal, up);
      const float       slope  = float(std::sqrt(std::max(1.0 - level * level, 0.0)) / std::max(level, 1e-3));
      m.positions.push_back(glm::vec3(drawn[at(i, j)] - m.centre));
      m.normals.push_back(glm::vec3(lift ? normalOf(drawn, i, j) : normal));
      // what stands over it, if anything: that share of it is the lift's colour, and none of its layers
      const Lift  over = lift ? lifts[at(i, j)] : Lift{};
      const float kept = 1.0F - std::clamp(over.share, 0.0F, 1.0F);
      tallest          = std::max(tallest, over.height);
      // a colour is kept in 8 bits a channel: dark ground would show its steps as contour lines, so
      // each vertex is nudged by up to half a step of its own
      glm::vec3 c = surfaceColour(world, s, slope);
      if(over.share > 0.0F)
        c = glm::mix(over.colour, c, kept);
      const uint32_t h = mix(mix(mix(dither + q.face * 64U + q.level) + uint32_t(q.x * N + uint32_t(i))) + uint32_t(q.y * N + uint32_t(j)));
      c += (glm::vec3(float(h & 255U), float((h >> 8U) & 255U), float((h >> 16U) & 255U)) / 255.0F - 0.5F) / 255.0F;
      m.colors.emplace_back(glm::clamp(c, 0.0F, 1.0F), 1.0F);
      // what it is drawn with: a byte each of the first four layers, the rest as floats
      uint32_t packed = 0;
      auto     layers = surfaceLayers(world, s, slope);
      if(over.share > 0.0F)
        for(float& l : layers)
          l *= kept;
      for(size_t l = 0; l < 4; l++)
        packed |= uint32_t(std::lround(255.0F * std::clamp(layers[l], 0.0F, 1.0F))) << (8U * uint32_t(l));
      m.layers.push_back(packed);
      static_assert(kGroundLayers == 8, "four layers in a word, four as floats");
      m.layersMore.emplace_back(layers[4], layers[5], layers[6], layers[7]);
      m.elevations.push_back(s.elevation);
      m.lowest  = std::min(m.lowest, s.elevation);
      m.highest = std::max(m.highest, s.elevation);
      m.inland  = m.inland || s.water > s.elevation - 0.3;
    }
  m.surface = uint32_t(m.positions.size());
  addGrid(m, N);
  // a coarser neighbour's edge misses what this one's finer ground adds, and the other way round:
  // up to about a cell's worth, and what stands over either edge
  addSkirts(m, world, N, cell + double(tallest));
  return m;
}

double groundAt(const World& world, const Quad& q, const ChunkMesh& ground, const glm::dvec3& d)
{
  constexpr uint32_t N    = kGroundCells;
  const uint32_t     side = N + 1;
  if(ground.elevations.size() < size_t(side) * side)
    return 0.0;
  // the cell the point is in, and where in it: the grid is even in the face's own measure
  const FacePoint p    = onFace(q.face, d);
  const double    step = coordOf(q, q.x, 1, N) - coordOf(q, q.x, 0, N);
  const double    fx = std::clamp((p.u - coordOf(q, q.x, 0, N)) / step, 0.0, double(N));
  const double    fy = std::clamp((p.v - coordOf(q, q.y, 0, N)) / step, 0.0, double(N));
  const uint32_t  i = std::min(uint32_t(fx), N - 1), j = std::min(uint32_t(fy), N - 1);
  const double    a = fx - double(i), b = fy - double(j);
  // its corners as the mesh has them, under any lift, and its triangles cut as addGrid cuts them
  const auto corner = [&](uint32_t ci, uint32_t cj) {
    return toScene(world, direction(q.face, coordOf(q, q.x, int(ci), N), coordOf(q, q.y, int(cj), N)), ground.elevations[size_t(cj) * side + ci]);
  };
  const glm::dvec3 c00 = corner(i, j), c10 = corner(i + 1, j), c01 = corner(i, j + 1), c11 = corner(i + 1, j + 1);
  const auto       apart = [&](const glm::dvec3& x, const glm::dvec3& y) {
    const glm::vec3 e = glm::vec3(x - ground.centre) - glm::vec3(y - ground.centre);
    return glm::dot(e, e);
  };
  std::array<glm::dvec3, 3> t;
  if(apart(c00, c11) <= apart(c10, c01))
    t = a >= b ? std::array<glm::dvec3, 3>{c00, c10, c11} : std::array<glm::dvec3, 3>{c00, c11, c01};
  else
    t = a + b <= 1.0 ? std::array<glm::dvec3, 3>{c00, c10, c01} : std::array<glm::dvec3, 3>{c10, c11, c01};
  // where the planet's radius through the point meets that triangle's plane
  const glm::dvec3 along  = world.frame.turn * d;
  const glm::dvec3 normal = glm::cross(t[1] - t[0], t[2] - t[0]);
  const double     facing = glm::dot(normal, along);
  if(std::abs(facing) < 1e-12 * glm::length(normal))
    return ground.elevations[size_t(j) * side + i];
  return glm::dot(normal, t[0] - world.frame.centre) / facing - world.spec.radius;
}

ChunkMesh waterMesh(const World& world, const Quad& q, uint32_t cells)
{
  const uint32_t N    = std::max(cells, 1U);
  const double   cell = width(world, q) / double(N);
  ChunkMesh      m;
  m.centre = toScene(world, direction(q.face, coordOf(q, q.x, 1, 2), coordOf(q, q.y, 1, 2)), 0.0);
  std::vector<uint8_t> sea;  // the vertices at the sea: it is no mesh's
  sea.reserve(size_t(N + 1) * (N + 1));
  for(int j = 0; j <= int(N); j++)
    for(int i = 0; i <= int(N); i++)
    {
      // a lake's level or a river's: level water, so its normal is straight up even where its
      // surface goes down under a bank. At the sea it goes down under the sea's own surface.
      const glm::dvec3 d      = direction(q.face, coordOf(q, q.x, i, N), coordOf(q, q.y, j, N));
      const double     level  = waterLevel(world, d, cell);
      const bool       atSea  = level <= kSeaMeets;
      const glm::dvec3 p      = toScene(world, d, atSea ? -kSeaUnder : level);
      m.positions.push_back(glm::vec3(p - m.centre));
      m.normals.push_back(glm::vec3(glm::normalize(p - world.frame.centre)));
      m.colors.emplace_back(1.0F);
      m.highest = std::max(m.highest, level);
      m.inland  = m.inland || !atSea;
      sea.push_back(atSea ? 1 : 0);
    }
  m.surface = uint32_t(m.positions.size());
  addGrid(m, N, &sea);
  // two levels of one sphere differ by how far their chords sag: a cell squared over eight radii
  addSkirts(m, world, N, 0.25 * cell + 4.0 * cell * cell / (8.0 * world.spec.radius), &sea);
  return m;
}

}  // namespace fv::world
