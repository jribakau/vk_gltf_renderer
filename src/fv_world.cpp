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
                   kChShade = 90, kChPatch = 100, kChSite = 110, kChDither = 120, kChCliff = 130;

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
  const double inland = smooth(0.15, 0.9, s.continent);
  if(inland <= 0.0 || w.spec.relief <= 0.0)
    return;
  const double range = smooth(0.50, 0.66, fbm3(p.direction * kRanges + glm::dvec3(3.1, 7.7, 1.3), seedOf(w), kChRange, 3, finestOf(w, p, kRanges)));
  if(range <= 0.0)
    return;
  const double ridge = ridged3(p.direction * kRidges, seedOf(w), kChRidge, 5, finestOf(w, p, kRidges));
  s.mountain         = float(inland * range * ridge);
  s.elevation += 0.045 * w.spec.radius * w.spec.relief * double(s.mountain);
}

// Hills, lower toward the shore and on the sea's floor
void hills(const World& w, const Point& p, Sample& s)
{
  if(w.spec.relief <= 0.0)
    return;
  const double h    = fbm3(p.direction * kHills, seedOf(w), kChHills, 4, finestOf(w, p, kHills)) - 0.5;
  const double land = 0.15 + 0.85 * smooth(0.0, 0.5, s.continent);
  s.elevation += 2.0 * h * 0.007 * w.spec.radius * w.spec.relief * land;
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
    const double waterside = 1.0 - smooth(0.0, std::min(1.2 * s.inland.cell, 120.0), double(s.shore));
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
  // (trees are drawn on quads up to some 8 m a vertex, fewer of them from 4: fv::world::keptOn)
  const float canopy = w.planted ? float(smooth(4.0, 12.0, p.footprint)) : 1.0F;
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

void detail(const World& w, const Point& p, Sample& s)
{
  if(w.spec.relief <= 0.0)
    return;
  // (the sea's floor is half as rough as the land, eased in across the shore: switched at the sea's
  // level, the ground stepped by half a bump's height along that line, metres of wall behind a
  // beach. Nothing that changes within a few metres may scale the bumps: they are up to 256 m across
  // and metres high, and a cliff's face made three times as rough stepped as the shore did.)
  const double amount = kDetailHeight * double(s.rough) * std::min(w.spec.relief, 1.5) * (0.5 + 0.5 * smooth(-3.0, 3.0, s.elevation));
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

// The ground's colour: its biome's, in lighter and darker patches, sand at the shore and on the
// sea's floor
void colour(const World& w, const Point& p, Sample& s)
{
  // patches a hundred metres across, and others the size of a valley
  const double patch = fbm3(p.direction * (w.spec.radius / 90.0), seedOf(w), kChPatch, 2, p.footprint / 90.0);
  const double shade = fbm3(p.direction * 40.0, seedOf(w), kChShade, 2, finestOf(w, p, 40.0));
  s.colour *= float((0.85 + 0.3 * patch) * (0.9 + 0.2 * shade));
  if(s.elevation < 0.0)
  {
    s.colour = glm::mix(kSand, kSeaFloorDeep, float(smooth(0.0, 40.0, -s.elevation)));
    s.rock   = 0.0F;
    return;
  }
  if(s.water > s.elevation)
  {
    // a river's or a lake's bed: gravel in the shallows, dark mud under a few metres
    s.colour = glm::mix(kGravel, 0.6F * kSeaFloorDeep, float(smooth(0.0, 6.0, s.water - s.elevation)));
    s.rock   = 0.0F;
    return;
  }
  // a beach no narrower than what draws it
  const double top = kBeach + 0.015 * p.footprint;
  s.colour         = glm::mix(s.colour, kSand, float(1.0 - smooth(0.6 * top, top, s.elevation)));
  // damp gravel by a river or a lake, and a river too narrow to draw as water
  if(s.shore < 2.5F)
    s.colour = glm::mix(s.colour, kGravel, 0.5F * float(1.0 - smooth(0.0, 2.5, double(s.shore))));
  if(s.painted > 0.0F)
    s.colour = glm::mix(s.colour, kRiverSeen, s.painted);
}

// Level where the figures stand, easing into the land round it
void clearing(const World& w, const Point& p, Sample& s)
{
  const double away = glm::length(p.direction - w.frame.site) * w.spec.radius;
  if(away >= kClearing + kClearingEdge)
    return;
  s.elevation = w.frame.siteElevation + (s.elevation - w.frame.siteElevation) * smooth(kClearing, kClearing + kClearingEdge, away);
}

std::vector<Biome> biomeTable()
{
  // (above 5 degrees the rows are a ladder by how wet it is, and must leave no gap)
  // What grows: the kit's plants are a northern meadow's and a fir wood's, so the warm, wet rows
  // have them thickest; the dry and the cold have rocks and little else.
  return {
      {.name = "snow", .temperatureTo = -5.0F, .colour = {0.78F, 0.80F, 0.84F}, .rough = 0.5F},
      {.name = "tundra", .temperatureFrom = -5.0F, .temperatureTo = 5.0F, .colour = {0.17F, 0.16F, 0.11F}, .rough = 1.0F,
       .plants = {{.role = "rock", .perSquareMetre = 0.012, .scaleMin = 0.15F, .scaleMax = 0.8F, .sink = 0.1F, .cover = 0.5F, .patch = 40.0F},
                  {.role = "shrub", .perSquareMetre = 0.006, .scaleMin = 0.4F, .scaleMax = 0.8F, .sink = 0.03F, .cover = 0.25F, .patch = 30.0F}},
       .grass  = Grass::eCold},
      {.name = "desert", .temperatureFrom = 5.0F, .moistureTo = 0.15F, .colour = {0.40F, 0.30F, 0.17F}, .rough = 0.6F,
       .plants = {{.role = "rock", .perSquareMetre = 0.008, .scaleMin = 0.2F, .scaleMax = 1.2F, .sink = 0.12F, .cover = 0.35F, .patch = 70.0F}}},
      {.name = "steppe", .temperatureFrom = 5.0F, .moistureFrom = 0.15F, .moistureTo = 0.36F, .colour = {0.26F, 0.21F, 0.12F}, .rough = 0.9F,
       .plants = {{.role = "shrub", .perSquareMetre = 0.01, .scaleMin = 0.5F, .scaleMax = 1.1F, .sink = 0.03F, .cover = 0.3F, .patch = 35.0F},
                  {.role = "rock", .perSquareMetre = 0.005, .scaleMin = 0.2F, .scaleMax = 0.9F, .sink = 0.1F, .cover = 0.3F, .patch = 60.0F},
                  {.role = "tree", .perSquareMetre = 0.004, .scaleMin = 0.8F, .scaleMax = 1.5F, .sink = 0.15F, .cover = 0.12F, .patch = 90.0F}},
       .grass  = Grass::eDry},
      {.name = "grassland", .temperatureFrom = 5.0F, .moistureFrom = 0.36F, .moistureTo = 0.68F, .colour = {0.085F, 0.14F, 0.045F}, .rough = 0.8F,
       .plants = {{.role = "tree", .perSquareMetre = 0.02, .scaleMin = 0.8F, .scaleMax = 1.9F, .sink = 0.15F, .cover = 0.3F, .patch = 80.0F},
                  {.role = "sapling", .perSquareMetre = 0.02, .scaleMin = 0.7F, .scaleMax = 1.6F, .cover = 0.35F, .patch = 30.0F},
                  {.role = "shrub", .perSquareMetre = 0.012, .scaleMin = 0.6F, .scaleMax = 1.3F, .sink = 0.03F, .cover = 0.3F, .patch = 35.0F},
                  {.role = "rock", .perSquareMetre = 0.004, .scaleMin = 0.2F, .scaleMax = 0.8F, .sink = 0.1F, .cover = 0.3F, .patch = 60.0F}},
       .grass  = Grass::eMeadow,
       .drift  = "flower"},
      {.name = "forest", .temperatureFrom = 5.0F, .moistureFrom = 0.68F, .colour = {0.045F, 0.085F, 0.03F}, .rough = 0.9F,
       .floor  = {0.075F, 0.06F, 0.035F},
       .plants = {{.role = "tree", .perSquareMetre = 0.026, .scaleMin = 1.0F, .scaleMax = 2.4F, .sink = 0.2F, .cover = 0.9F, .patch = 60.0F},
                  {.role = "sapling", .perSquareMetre = 0.03, .scaleMin = 0.6F, .scaleMax = 1.7F, .cover = 0.55F, .patch = 18.0F},
                  {.role = "shrub", .perSquareMetre = 0.05, .scaleMin = 0.5F, .scaleMax = 1.2F, .sink = 0.03F, .cover = 0.45F, .patch = 22.0F},
                  {.role = "rock", .perSquareMetre = 0.02, .scaleMin = 0.15F, .scaleMax = 0.7F, .sink = 0.08F, .cover = 0.5F, .patch = 30.0F},
                  {.role = "stump", .perSquareMetre = 0.008, .scaleMin = 0.6F, .scaleMax = 1.2F, .sink = 0.03F},
                  {.role = "log", .perSquareMetre = 0.012, .scaleMin = 0.7F, .scaleMax = 1.5F}},
       .grass  = Grass::eWood,
       .drift  = "fern"},
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

// `out`, when given: the vertices that are no part of the surface; a triangle of three is left out
void addGrid(ChunkMesh& m, uint32_t cells, const std::vector<uint8_t>* out = nullptr)
{
  const uint32_t side = cells + 1;
  m.indices.reserve(out ? 0 : size_t(cells) * cells * 6 + size_t(cells) * 24);
  const auto triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
    if(!(out && (*out)[a] && (*out)[b] && (*out)[c]))
      m.indices.insert(m.indices.end(), {a, b, c});
  };
  for(uint32_t j = 0; j < cells; j++)
    for(uint32_t i = 0; i < cells; i++)
    {
      const uint32_t a = j * side + i, b = a + 1, c = a + side, d = c + 1;
      triangle(a, b, d);
      triangle(a, d, c);
    }
}

}  // namespace

// ---- the planet --------------------------------------------------------------------------------------

World make(const Spec& asked, const Making& making)
{
  World w;
  w.spec        = asked;
  w.spec.radius = std::clamp(asked.radius, 2000.0, 100000.0);
  w.spec.sea    = std::clamp(asked.sea, 0.0, 0.9);
  w.spec.relief = std::clamp(asked.relief, 0.0, 2.0);
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
    w.base = bake(w, making.map ? making.map : mapCells(w.spec.radius), making.stop);
  if(w.base)
    w.stages.push_back({"base", base});
  w.stages.insert(w.stages.end(), {{"climate", climate}, {"biome", biome}, {"detail", detail}});
  if(w.base)
    w.stages.push_back({"water", water});
  w.stages.push_back({"colour", colour});

  // where the figures stand, the origin
  w.frame.site          = findSite(w);
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

glm::vec3 surfaceColour(const World&, const Sample& s, float slope)
{
  // rock from where a slope is too steep for soil, to a cliff's bare face; not under water
  const float steep = s.elevation < 0.0 || s.water > s.elevation ? 0.0F : float(smooth(0.45, 0.9, double(slope)));
  return glm::mix(s.colour, kRock, std::max(steep, s.rock));
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

void visit(const World& w, const Quad& q, const glm::dvec3& eye, const glm::dvec3& eyeDirection, uint8_t finest, const Selection* was, Selection& out)
{
  if(q.level < finest)
  {
    const bool   held  = was && std::binary_search(was->split.begin(), was->split.end(), q);
    const double reach = kSplit * width(w, q) * (held ? kSplitHold : 1.0);
    if(distanceTo(w, q, eye, eyeDirection) < reach)
    {
      out.split.push_back(q);
      for(const Quad& child : children(q))
        visit(w, child, eye, eyeDirection, finest, was, out);
      return;
    }
  }
  out.leaves.push_back(q);
}

}  // namespace

Selection select(const World& world, const glm::dvec3& eye, const Selection* was)
{
  double           r = 0.0;
  const glm::dvec3 d = toPlanet(world, eye, &r);
  Selection        out;
  const uint8_t    finest = finestLevel(world);
  for(uint8_t face = 0; face < 6; face++)
    visit(world, {face, 0, 0, 0}, d * r, d, finest, was, out);
  std::sort(out.leaves.begin(), out.leaves.end());
  std::sort(out.split.begin(), out.split.end());
  return out;
}

// ---- a quad's triangles ------------------------------------------------------------------------------

ChunkMesh groundMesh(const World& world, const Quad& q)
{
  constexpr uint32_t N    = kGroundCells;
  constexpr int      G    = int(N) + 3;  // a ring of points round the quad's own, for its edges' normals
  const double       cell = width(world, q) / double(N);
  const auto         at   = [&](int i, int j) { return size_t(j + 1) * size_t(G) + size_t(i + 1); };

  std::vector<glm::dvec3> points(size_t(G) * size_t(G));
  std::vector<Sample>     samples(points.size());
  for(int j = -1; j <= int(N) + 1; j++)
    for(int i = -1; i <= int(N) + 1; i++)
    {
      const glm::dvec3 d = direction(q.face, coordOf(q, q.x, i, N), coordOf(q, q.y, j, N));
      samples[at(i, j)]  = sample(world, d, cell);
      points[at(i, j)]   = toScene(world, d, samples[at(i, j)].elevation);
    }

  ChunkMesh m;
  m.centre  = points[at(int(N) / 2, int(N) / 2)];
  m.lowest  = std::numeric_limits<double>::max();
  m.highest = std::numeric_limits<double>::lowest();
  const size_t count = size_t(N + 1) * size_t(N + 1);
  m.positions.reserve(count + 4 * size_t(N + 1));
  m.normals.reserve(count + 4 * size_t(N + 1));
  m.colors.reserve(count + 4 * size_t(N + 1));
  const uint32_t dither = lattice(uint32_t(world.spec.seed), kChDither);
  for(int j = 0; j <= int(N); j++)
    for(int i = 0; i <= int(N); i++)
    {
      const glm::dvec3& p      = points[at(i, j)];
      const Sample&     s      = samples[at(i, j)];
      const glm::dvec3  up     = glm::normalize(p - world.frame.centre);
      glm::dvec3        normal = glm::normalize(glm::cross(points[at(i + 1, j)] - points[at(i - 1, j)], points[at(i, j + 1)] - points[at(i, j - 1)]));
      const double      level  = glm::dot(normal, up);
      const float       slope  = float(std::sqrt(std::max(1.0 - level * level, 0.0)) / std::max(level, 1e-3));
      m.positions.push_back(glm::vec3(p - m.centre));
      m.normals.push_back(glm::vec3(normal));
      // a colour is kept in 8 bits a channel: dark ground would show its steps as contour lines, so
      // each vertex is nudged by up to half a step of its own
      glm::vec3      c = surfaceColour(world, s, slope);
      const uint32_t h = mix(mix(mix(dither + q.face * 64U + q.level) + uint32_t(q.x * N + uint32_t(i))) + uint32_t(q.y * N + uint32_t(j)));
      c += (glm::vec3(float(h & 255U), float((h >> 8U) & 255U), float((h >> 16U) & 255U)) / 255.0F - 0.5F) / 255.0F;
      m.colors.emplace_back(glm::clamp(c, 0.0F, 1.0F), 1.0F);
      m.lowest  = std::min(m.lowest, s.elevation);
      m.highest = std::max(m.highest, s.elevation);
      m.inland  = m.inland || s.water > s.elevation - 0.3;
    }
  m.surface = uint32_t(m.positions.size());
  addGrid(m, N);
  // a coarser neighbour's edge misses what this one's finer ground adds, and the other way round:
  // up to about a cell's worth
  addSkirts(m, world, N, cell);
  return m;
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
