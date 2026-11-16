#include "fv_world_plants.hpp"

#include <algorithm>
#include <cmath>

#include "fv_world_map.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace fv::world {

namespace {

constexpr double kPi = 3.14159265358979323846;

uint32_t mix(uint32_t v)
{
  v = v * 747796405U + 2891336453U;
  const uint32_t w = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (w >> 22U) ^ w;
}

uint32_t hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
  return mix(mix(mix(mix(a) + b) + c) + d);
}

float unit(uint32_t h)
{
  return float(h >> 8U) * (1.0F / 16777216.0F);
}

double smooth(double a, double b, double x)
{
  const double t = std::clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Half a face's width, metres, at its middle
double halfFace(const World& w)
{
  return 0.25 * kPi * w.spec.radius;
}

// The level of `noise` its top `cover` share is over (the noise is near enough a bell about a half)
double coverLevel(double cover)
{
  const double p = std::clamp(1.0 - cover, 0.01, 0.99);
  return 0.5 + 0.13 * 4.91 * (std::pow(p, 0.14) - std::pow(1.0 - p, 0.14));
}

// Nothing grows here whatever the biome: the sea and its beach, a river and its banks, a lake.
// Rocks lie on a beach and by the water.
bool bare(const Sample& s, bool rock)
{
  if(s.elevation < (rock ? 0.3 : 2.5))
    return true;
  if(s.water > s.elevation - 0.2)
    return true;
  if(s.inland.river && s.inland.riverAway < s.inland.riverHalf + (rock ? 0.5 : 2.0))
    return true;
  return s.inland.lake > 0.2 && s.elevation < s.inland.lakeLevel + 0.5;
}

// Where the figures stand: nothing with a place of its own stands there (grass does, underfoot)
bool cleared(const World& w, const glm::dvec3& d)
{
  return glm::length(d - w.frame.site) * w.spec.radius < kClearing + 4.0;
}

// The turn that takes +y to `up` by the shortest way
glm::quat toUp(const glm::dvec3& up)
{
  const glm::dvec3 axis = glm::cross(glm::dvec3(0.0, 1.0, 0.0), up);
  const double     s    = glm::length(axis);
  if(s < 1e-9)
    return up.y > 0.0 ? glm::quat(1.0F, 0.0F, 0.0F, 0.0F) : glm::quat(0.0F, 1.0F, 0.0F, 0.0F);
  return glm::quat(glm::angleAxis(std::atan2(s, up.y), axis / s));
}

// The biome a place counts as, where several meet: one of them, by their shares and `pick` (0..1)
int biomeBy(const World& w, const Sample& s, float pick)
{
  float sum = 0.0F;
  for(size_t i = 0; i < w.biomes.size() && i < kMostBiomes; i++)
  {
    sum += s.share[i];
    if(pick < sum)
      return int(i);
  }
  return s.biome;
}

// The grass's tiles lie on a face's plane, in metres: its half width a whole number of the coarsest
// tiles, so the tiles end at the face's edge
double tilePlaneHalf(const World& w)
{
  return 8.0 * std::max(1.0, std::round(halfFace(w) / 8.0));
}

fv::backdrop::Spec flatSpec(const World& w, const char* kind, double density)
{
  fv::backdrop::Spec flat;
  flat.kind    = kind;
  flat.seed    = w.spec.seed;
  flat.density = density;
  return flat;
}

// A role's grid over a cube face, a plant a cell at the most, `thickest` of them a square metre where
// they stand thickest: calls each(u, v, cell, h) for every cell whose plant falls on the quad (a cell
// on the quad's edge is the quad's the plant falls in). `h`: the cell's own hash.
template <class Each>
void overCells(const World& w, const Quad& q, double thickest, uint32_t salt, Each&& each)
{
  const double  span = 2.0 / double(uint64_t(1) << q.level);
  const double  u0 = double(q.x) * span - 1.0, v0 = double(q.y) * span - 1.0;
  const double  n    = std::max(1.0, std::round(2.0 * halfFace(w) * std::sqrt(thickest)));
  const double  cell = 2.0 / n;  // in the face's own measure, -1 to 1 across it
  const int64_t x0 = int64_t(std::floor((u0 + 1.0) / cell)), x1 = int64_t(std::ceil((u0 + span + 1.0) / cell));
  const int64_t y0 = int64_t(std::floor((v0 + 1.0) / cell)), y1 = int64_t(std::ceil((v0 + span + 1.0) / cell));
  for(int64_t iy = y0; iy < y1; iy++)
    for(int64_t ix = x0; ix < x1; ix++)
    {
      const uint32_t h = hash(uint32_t(w.spec.seed) + salt, uint32_t(q.face), uint32_t(ix), uint32_t(iy));
      const double   u = (double(ix) + double(unit(h))) * cell - 1.0, v = (double(iy) + double(unit(mix(h + 1U)))) * cell - 1.0;
      if(u >= u0 && u < u0 + span && v >= v0 && v < v0 + span)
        each(u, v, cell, h);
    }
}

// A cell's share of a plant where they stand thickest: a face's cells are smaller toward its edges,
// and there are as many plants a square metre everywhere
double cellShare(const World& w, uint8_t face, double u, double v, double cell, double thickest)
{
  const double     e = 0.5 * cell;
  const glm::dvec3 across = direction(face, u + e, v) - direction(face, u - e, v), along = direction(face, u, v + e) - direction(face, u, v - e);
  return glm::length(glm::cross(across, along)) * w.spec.radius * w.spec.radius * thickest;
}

// Where a plant stands: on the planet's point `d` at `elevation`, about `centre`, upright there,
// turned by `turn` (0..1 of a whole turn) and scaled
glm::mat4 placeOn(const World& w, const glm::dvec3& d, double elevation, const glm::dvec3& centre, float turn, float scale)
{
  const glm::dvec3 at = toScene(w, d, elevation);
  glm::mat4        m  = glm::translate(glm::mat4(1.0F), glm::vec3(at - centre));
  m                   = m * glm::mat4_cast(toUp(glm::normalize(at - w.frame.centre)));
  m                   = glm::rotate(m, float(2.0 * kPi) * turn, glm::vec3(0.0F, 1.0F, 0.0F));
  return glm::scale(m, glm::vec3(scale));
}

// The ground's slope at `d`, where it is `elevation` high: its rise over a few metres two ways
double slopeAt(const World& w, const glm::dvec3& d, double elevation, double footprint)
{
  const double     reach = std::max(2.0, 2.0 * footprint);
  const glm::dvec3 a     = glm::normalize(glm::cross(d, std::abs(d.y) < 0.9 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0)));
  const glm::dvec3 b     = glm::cross(d, a);
  const double     da    = sample(w, glm::normalize(d + a * (reach / w.spec.radius)), footprint).elevation - elevation;
  const double     db    = sample(w, glm::normalize(d + b * (reach / w.spec.radius)), footprint).elevation - elevation;
  return std::sqrt(da * da + db * db) / reach;
}

// Steeper than this, a tree, a bush or a log has no footing (35 degrees)
constexpr double kStandSlope = 0.7;

// What a plant is decided by: the ground this fine, whichever quad holds it, so it is the same plant
// on all of them. (Its foot is on the ground as its quad draws it.)
constexpr double kPlantFootprint = 1.0;

// What grows by water only grows on about this share of its biome's land, as far as how many trees
// are seen from one place goes (Reach)
constexpr double kByWaterShare = 0.1;

// How thickly something that grows by water (Growth::water) grows `shore` metres from the water's
// edge: all of it to a third of the way, none past it
double byWater(float water, float shore)
{
  return water > 0.0F ? 1.0 - smooth(0.35 * double(water), double(water), double(shore)) : 1.0;
}

// Whether `d` is in a grove of what a biome grows in groves (Growth::cover under 1), 0 to 1 at its
// edge: of the role (of kStandingRoles) `role`, in the biome (of World::biomes) `biome`
double grove(const World& w, const glm::dvec3& d, const Growth& g, size_t role, size_t biome)
{
  const double level = coverLevel(double(g.cover));
  return smooth(level - 0.03, level + 0.03, noise(w, d, double(g.patch), uint32_t(role * kMostBiomes + biome)));
}

// How thickly a biome's trees stand together over a wide ground, a square metre (or a role's own, for
// what is no tree), its groves and its waters taken in: what decides how far a role is seen (Reach)
double woodOf(const Biome& b, std::string_view role, double density)
{
  double kind = 0.0;
  for(const Growth& g : b.plants)
    if(isTree(role) ? isTree(g.role) : g.role == role)
      kind += g.perSquareMetre * density * double(std::min(g.cover, 1.0F)) * (g.water > 0.0F ? kByWaterShare : 1.0);
  return kind;
}

// Whether nothing as tall as a tree stands at `d`, whatever grows there: the water and its shores, the
// figures' place, bare rock, what a place keeps, a place's floor
bool noTree(const World& w, const glm::dvec3& d, const Sample& s)
{
  return bare(s, false) || cleared(w, d) || s.rock > 0.5F || (s.built.taken & (kGroundKept | kGroundOpen));
}

// The stands of a role, a variant each: the furthest seen first
struct Placed
{
  glm::mat4 place;
  float     reach;
};
void standsOf(uint8_t role, std::vector<std::vector<Placed>>& placed, std::vector<Stand>& out)
{
  for(uint32_t v = 0; v < placed.size(); v++)
  {
    if(placed[v].empty())
      continue;
    std::stable_sort(placed[v].begin(), placed[v].end(), [](const Placed& a, const Placed& b) { return a.reach > b.reach; });
    Stand stand{.role = role, .variant = v};
    stand.places.reserve(placed[v].size());
    stand.reach.reserve(placed[v].size());
    for(const Placed& p : placed[v])
    {
      stand.places.push_back(p.place);
      stand.reach.push_back(p.reach);
    }
    out.push_back(std::move(stand));
  }
}

}  // namespace

// ---- what has a place ----------------------------------------------------------------------------------

Reach reachOf(std::string_view role, double perSquareMetre)
{
  // Trees are seen for kilometres (a 20 m one is a pixel or two of a traced frame at 12 km), the
  // smallest of them for a few hundred metres; the dry land's trees of 2 to 4 m for a couple of
  // kilometres; the undergrowth, rocks and dead wood on land for some hundreds of metres; what grows
  // on the sea floor, through water, a hundred metres or a few
  const bool   tree  = isTree(role), small = role == "drytree";
  const bool   land  = !tree && std::find(kStandingRoles.begin(), kStandingRoles.end(), role) != kStandingRoles.end();
  const double least = tree ? 350.0 : small ? 200.0 : 120.0, most = tree ? 12000.0 : small ? 2500.0 : land ? 800.0 : 400.0;
  const double seen  = tree ? kTreesSeen : land ? kUndergrowthSeen : kOthersSeen;
  // those seen from one place, with every one seen from `all`: all within it, and beyond it the share
  // (all / d)^2 of them at d, to `most`
  const auto count = [&](double all) { return perSquareMetre * kPi * all * all * (1.0 + 2.0 * std::log(most / all)); };
  if(!(perSquareMetre > 0.0) || count(most) <= seen)
    return {most, most};
  double lo = 1.0, hi = most;
  for(int i = 0; i < 40; i++)
  {
    const double mid = 0.5 * (lo + hi);
    (count(mid) < seen ? lo : hi) = mid;
  }
  return {std::max(lo, least), most};
}

double heldFrom(const World& w, const Quad& q)
{
  return q.level >= finestLevel(w) ? 0.0 : kHeldMargin * kSplit * width(w, q);
}

std::vector<Stand> standing(const World& w, const Quad& q, const glm::dvec3& centre, std::span<const uint32_t> variants, double density)
{
  std::vector<Stand> out;
  if(!(density > 0.0))
    return out;
  const double wide = width(w, q);
  // what draws the quad's ground: a plant stands on the ground as drawn
  const double footprint = wide / double(kGroundCells);
  const double held      = heldFrom(w, q);
  for(size_t r = 0; r < kStandingRoles.size() && r < variants.size(); r++)
  {
    const std::string_view role = kStandingRoles[r];
    if(variants[r] == 0)
      continue;
    // where each biome grows it, the thickest of them, and the thickest it stands over a wide ground
    // where it grows: a tree with the other trees there, a wood's firs and pines being one wood
    std::array<const Growth*, kMostBiomes> grows{};
    double                                 thickest = 0.0, wood = 0.0;
    // (what grows by water only: as far from it as the furthest of them, or 0 where any grows anywhere)
    float water = 0.0F;
    bool  anywhere = false;
    for(size_t b = 0; b < w.biomes.size() && b < kMostBiomes; b++)
    {
      for(const Growth& g : w.biomes[b].plants)
        if(g.role == role)
        {
          grows[b] = &g;
          thickest = std::max(thickest, g.perSquareMetre * density);
          water    = std::max(water, g.water);
          anywhere = anywhere || !(g.water > 0.0F);
        }
      // (what else stands on land is counted by its role, as before there were several)
      if(grows[b])
        wood = std::max(wood, woodOf(w.biomes[b], role, density));
    }
    if(!(thickest > 0.0))
      continue;
    // by water only, where there is none: no rivers nor lakes without the base map
    if(!anywhere && !w.base)
      continue;
    const Reach reach = reachOf(role, wood);
    if(reach.most < held)
      continue;  // a quad too far for any of them
    const bool                       rock = isStone(role);
    std::vector<std::vector<Placed>> placed(variants[r]);
    overCells(w, q, thickest, 7919U * uint32_t(r + 1), [&](double u, double v, double cell, uint32_t h) {
        // its size among its kind's, and by that how far it is seen from: the quad's, if that far
        const float  size = unit(mix(h + 5U));
        const double seen = reach.of(1.0 - double(size));
        if(seen < held)
          return;
        const glm::dvec3 d = direction(q.face, u, v);
        // what grows by water only, nowhere near any: one look at the base map, not the ground
        if(!anywhere && mapInland(*w.base, w.spec.radius, d).shore() >= double(water))
          return;
        const Sample s = sample(w, d, kPlantFootprint);
        if(bare(s, rock) || cleared(w, d) || (!rock && s.rock > 0.5F) || (s.built.taken & kGroundKept))
          return;
        // (a place stands in a clearing: what is as tall as a tree is none of its floor's)
        if((s.built.taken & kGroundOpen) && isTall(role))
          return;
        // how thickly it grows here: each biome's own by its share, in its groves
        double        odds = 0.0;
        const Growth* most = nullptr;
        float         top  = 0.0F;
        for(size_t b = 0; b < w.biomes.size() && b < kMostBiomes; b++)
        {
          const Growth* g = grows[b];
          if(!g || s.share[b] <= 0.0F)
            continue;
          double here = g->perSquareMetre * density / thickest;
          if(g->cover < 1.0F)
            here *= grove(w, d, *g, r, b);
          here *= byWater(g->water, s.shore);
          odds += double(s.share[b]) * here;
          if(s.share[b] > top)
          {
            top  = s.share[b];
            most = g;
          }
        }
        if(!most || double(unit(mix(h + 2U))) >= odds * cellShare(w, q.face, u, v, cell, thickest))
          return;
        // nothing but rocks on a cliff's shoulders or a bluff: where the tiers stand the land up, by
        // the ground's own slope (asked only there: two more looks at the ground)
        if(!rock && (s.cliff > 0.02F || s.steepened > 2.0F) && slopeAt(w, d, s.elevation, kPlantFootprint) > kStandSlope)
          return;
        const float  scale  = most->scaleMin + (most->scaleMax - most->scaleMin) * size;
        const double ground = footprint > 1.5 * kPlantFootprint ? sample(w, d, footprint).elevation : s.elevation;
        placed[mix(h + 3U) % variants[r]].push_back({placeOn(w, d, ground - double(most->sink * scale), centre, unit(mix(h + 4U)), scale), float(seen)});
    });
    standsOf(uint8_t(r), placed, out);
  }
  return out;
}

// ---- far woods as a canopy -----------------------------------------------------------------------------

namespace {

// What a wood's crowns are from afar, linear: a fir wood's (the forest's own colour from afar, Biome::
// colour, as it was drawn before there was a canopy), a leafy one's a little lighter
const glm::vec3 kNeedles{0.045F, 0.085F, 0.03F};
const glm::vec3 kLeaves{0.055F, 0.095F, 0.03F};

// A ground's points this far apart or nearer are each a crown's top or the floor between; this far or
// further, the wood as a whole over each, as high as its crowns are on the whole; between, between
constexpr double kCrownsTo = 12.0, kWoodFrom = 40.0;
// Seen at a slant over a wide ground, how much of it is crowns: the gaps between them are hidden by the
// crowns before them, as if they stood this many times as thickly
constexpr double kSlant = 2.0;
// A wood stands back from a river this far past its banks, and a cell of the ground's further
constexpr double kRiverRoom = 2.0;
constexpr uint32_t kCanopySalt = 0x6A09E667U;

}  // namespace

LiftOf canopyLift(const World& w, const Quad& q, const Canopy& canopy)
{
  // a quad holds the trees seen from where it draws (heldFrom): the finest, all of them
  if(!(heldFrom(w, q) > 0.0) || !(canopy.density > 0.0))
    return {};
  // and of those it holds, shows each only from as near as its reach: the canopy is every tree not
  // shown from where the quad's trees are on the whole while it draws
  const double away = kCanopyAway * width(w, q);
  struct Role
  {
    size_t                                 index = 0;  // of kStandingRoles
    double                                 hidden = 0.0;  // the share of its plants not shown from `away`: the smallest
    std::array<const Growth*, kMostBiomes> grows{};
    double                                 top = 0.0, crown = 0.0;  // its models' height and crown at a scale of 1
    bool                                   leafy = false;
  };
  std::vector<Role> roles;
  for(size_t r = 0; r < kStandingRoles.size(); r++)
  {
    const std::string_view role = kStandingRoles[r];
    if(!isTree(role) || !(canopy.height[r] > 0.0F))
      continue;
    Role   it{.index = r, .top = double(canopy.height[r]), .crown = double(canopy.crown[r]), .leafy = drawnAs(role) == "broadleaf"};
    double wood     = 0.0;
    bool   anywhere = false, any = false;
    for(size_t b = 0; b < w.biomes.size() && b < kMostBiomes; b++)
    {
      for(const Growth& g : w.biomes[b].plants)
        if(g.role == role)
        {
          it.grows[b] = &g;
          any         = true;
          anywhere    = anywhere || !(g.water > 0.0F);
        }
      if(it.grows[b])
        wood = std::max(wood, woodOf(w.biomes[b], role, canopy.density));
    }
    // (as `standing`: by water only, where there is none, none stand)
    if(!any || (!anywhere && !w.base))
      continue;
    // those seen from that far are the largest (Reach::of)
    const Reach  reach = reachOf(role, wood);
    const double seen  = away > reach.most ? 0.0 : std::min(1.0, (reach.all / away) * (reach.all / away));
    it.hidden          = 1.0 - seen;
    if(it.hidden > 0.0)
      roles.push_back(it);
  }
  if(roles.empty())
    return {};
  return [&w, roles = std::move(roles), density = canopy.density](const glm::dvec3& d, const Sample& s, double cell) -> Lift {
    // where no tree stands: as `standing`, and nothing over a cliff's face nor a road, nor by a river,
    // which would be buried under the slope up to the crowns
    if(noTree(w, d, s) || s.cliff > 0.02F || s.steepened > 2.0F || s.built.road > 0.0F)
      return {};
    if(s.inland.river && s.inland.riverAway < s.inland.riverHalf + kRiverRoom + cell)
      return {};
    // the trees a square metre not drawn here, each role's of each biome's, but for their groves: the
    // groves' noise is most of what a point costs, so it is asked last and only where it can matter
    struct Term
    {
      const Role*   role;
      const Growth* g;
      size_t        biome;
      double        here, least, most, mid, top;
    };
    std::array<Term, kStandingRoles.size() * kMostBiomes> terms;  // (none of them set but the first `n`)
    size_t                                                n = 0;
    double                                                crownsAtMost = 0.0;  // their crowns' share, every grove whole
    for(const Role& role : roles)
      for(size_t b = 0; b < w.biomes.size() && b < kMostBiomes; b++)
      {
        const Growth* g = role.grows[b];
        if(!g || s.share[b] <= 0.0F)
          continue;
        const double here = double(s.share[b]) * g->perSquareMetre * density * role.hidden * byWater(g->water, s.shore);
        if(!(here > 0.0))
          continue;
        // those not drawn are the smallest: the sizes up to `hidden` of the whole range
        const double least = double(g->scaleMin), most = least + double(g->scaleMax - g->scaleMin) * role.hidden;
        const double mid = 0.5 * (least + most);
        crownsAtMost += here * kPi * (role.crown * mid) * (role.crown * mid);
        terms[n++] = {.role = &role, .g = g, .biome = b, .here = here, .least = least, .most = most, .mid = mid, .top = role.top - double(g->sink)};
      }
    if(n == 0)
      return {};
    // near: each point a crown's top, by its share of the ground, or the floor between; (by where the
    // point is, not which quad's it is: two quads of a level draw their edge alike). A point that would
    // be the floor with every grove whole is the floor whatever its groves.
    const double wide = smooth(kCrownsTo, kWoodFrom, cell);
    uint32_t     h    = 0;
    if(wide < 1.0)
    {
      const double step = 0.25 * cell / w.spec.radius;
      const auto   key  = [&](double x) { return uint32_t(uint64_t(std::llround(x / step))); };
      h                 = hash(uint32_t(w.spec.seed) + kCanopySalt, key(d.x), key(d.y), key(d.z));
      if(wide <= 0.0 && double(unit(h)) >= 1.0 - std::exp(-crownsAtMost))
        return {};
    }
    // in their groves; on a ground coarser than they are, as much as they cover of it
    double    trees = 0.0, crowns = 0.0, mean = 0.0, low = 0.0, high = 0.0;
    glm::vec3 colour(0.0F);
    for(size_t k = 0; k < n; k++)
    {
      const Term&   t    = terms[k];
      const Growth* g    = t.g;
      double        here = t.here;
      if(g->cover < 1.0F)
      {
        const double groves = smooth(0.5 * double(g->patch), 2.0 * double(g->patch), cell);
        here *= groves >= 1.0 ? double(g->cover) : glm::mix(grove(w, d, *g, t.role->index, t.biome), double(g->cover), groves);
      }
      if(!(here > 0.0))
        continue;
      trees += here;
      crowns += here * kPi * (t.role->crown * t.mid) * (t.role->crown * t.mid);
      mean += here * t.top * t.mid;
      low += here * t.top * t.least;
      high += here * t.top * t.most;
      colour += float(here) * (t.role->leafy ? kLeaves : kNeedles);
    }
    if(!(trees > 0.0))
      return {};
    mean /= trees;
    low /= trees;
    high /= trees;
    const double covered   = 1.0 - std::exp(-crowns);
    double       topHeight = 0.0, topShare = 0.0;
    if(wide < 1.0)
    {
      if(double(unit(h)) < covered)
      {
        topHeight = low + (high - low) * double(unit(mix(h + 1U)));
        topShare  = 1.0;
      }
    }
    // far: the wood over each point, as high as its crowns stand on the whole, and as much of it seen
    const double woodShare = 1.0 - std::exp(-kSlant * crowns);
    Lift         out;
    out.height = float(glm::mix(topHeight, covered * mean, wide));
    out.share  = float(glm::mix(topShare, woodShare, wide));
    out.colour = colour / float(trees);
    return out;
  };
}

namespace {

// How a sea floor role grows: how thickly, in water how deep and how warm at its surface, how large
// (times its mesh), and in what share of the floor, in patches how far across
struct SeaGrowth
{
  std::string_view role;
  double           perSquareMetre;
  float            depthFrom, depthTo;
  float            warmFrom, warmTo;  // degrees C
  float            scaleMin, scaleMax;
  float            cover, patch;
};
constexpr SeaGrowth kSeaGrowths[] = {
    {"seagrass", 0.3, 0.8F, 9.0F, 2.0F, 100.0F, 0.7F, 1.4F, 0.45F, 35.0F},
    {"kelp", 0.03, 5.0F, 28.0F, -2.0F, 21.0F, 0.6F, 1.4F, 0.3F, 60.0F},
    {"coral", 0.06, 1.5F, 18.0F, 17.0F, 100.0F, 0.6F, 2.0F, 0.35F, 25.0F},
    {"sponge", 0.015, 4.0F, 40.0F, 0.0F, 100.0F, 0.5F, 1.6F, 0.3F, 30.0F},
    {"searock", 0.006, 0.5F, 60.0F, -100.0F, 100.0F, 0.4F, 2.5F, 0.35F, 45.0F},
};

}  // namespace

std::vector<Stand> seaStanding(const World& w, const Quad& q, const glm::dvec3& centre, std::span<const SeaMesh> meshes, double density)
{
  std::vector<Stand> out;
  const double       wide = width(w, q);
  if(!(density > 0.0))
    return out;
  const double footprint = wide / double(kGroundCells);
  const double held      = heldFrom(w, q);
  for(size_t r = 0; r < kSeaRoles.size(); r++)
  {
    const SeaGrowth* grow = nullptr;
    for(const SeaGrowth& g : kSeaGrowths)
      if(g.role == kSeaRoles[r])
        grow = &g;
    // the role's meshes, in the order they come
    std::vector<const SeaMesh*> of;
    for(const SeaMesh& m : meshes)
      if(m.role == r)
        of.push_back(&m);
    if(!grow || of.empty())
      continue;
    const double thickest = grow->perSquareMetre * density;
    const double level    = coverLevel(double(grow->cover));
    const Reach  reach    = reachOf(grow->role, thickest * double(grow->cover));
    if(reach.most < held)
      continue;
    std::vector<std::vector<Placed>> placed(of.size());
    overCells(w, q, thickest, 104729U * uint32_t(r + 1), [&](double u, double v, double cell, uint32_t h) {
      const float  size = unit(mix(h + 5U));
      const double seen = reach.of(1.0 - double(size));
      if(seen < held)
        return;
      const glm::dvec3 d = direction(q.face, u, v);
      const Sample     s = sample(w, d, kPlantFootprint);
      // the sea's own floor: not a lake's, not a river's bed where it runs out
      const double deep = -s.elevation;
      if(deep < double(grow->depthFrom) || deep > double(grow->depthTo) || s.inland.lake > 0.2 || (s.inland.river && s.inland.riverAway < s.inland.riverHalf + 1.0))
        return;
      const double warm = temperatureAt(w, d, 0.0);
      if(warm < double(grow->warmFrom) || warm > double(grow->warmTo))
        return;
      const double odds = smooth(level - 0.03, level + 0.03, noise(w, d, double(grow->patch), uint32_t(200 + r))) * cellShare(w, q.face, u, v, cell, thickest);
      if(double(unit(mix(h + 2U))) >= odds)
        return;
      // no taller than the water over it
      const uint32_t variant = mix(h + 3U) % uint32_t(of.size());
      const float    tallest = float(0.85 * deep) / std::max(of[variant]->height, 1e-3F);
      const float    scale   = std::min(grow->scaleMin + (grow->scaleMax - grow->scaleMin) * size, tallest);
      if(scale < 0.5F * grow->scaleMin)
        return;
      const double floor = footprint > 1.5 * kPlantFootprint ? sample(w, d, footprint).elevation : s.elevation;
      placed[variant].push_back({placeOn(w, d, floor, centre, unit(mix(h + 4U)), scale), float(seen)});
    });
    standsOf(uint8_t(r), placed, out);
  }
  return out;
}

// ---- grass ---------------------------------------------------------------------------------------------

std::vector<TileKind> tileKinds(const World& w, double density)
{
  std::vector<TileKind> out;
  const auto used = [&](Grass g) { return std::any_of(w.biomes.begin(), w.biomes.end(), [&](const Biome& b) { return b.grass == g; }); };
  const auto drifted = [&](std::string_view d) { return std::any_of(w.biomes.begin(), w.biomes.end(), [&](const Biome& b) { return b.drift == d; }); };
  // the meadow's three levels, and each other grass as that thinned, clumped and tinted
  struct Of
  {
    Grass     grass;
    double    thick;
    float     clumps;
    glm::vec3 tint;
  };
  const Of ofs[] = {
      {Grass::eMeadow, 1.0, 1.0F, glm::vec3(1.0F)},
      {Grass::eDry, 0.35, 0.5F, glm::vec3(1.1F, 0.9F, 0.5F)},
      {Grass::eWood, 1.0 / 7.0, 0.35F, glm::vec3(0.8F, 0.75F, 0.45F)},
      {Grass::eCold, 0.2, 0.4F, glm::vec3(0.75F, 0.68F, 0.5F)},
  };
  const auto meadow = fv::backdrop::grassCovers(flatSpec(w, "meadow", density));
  for(const Of& of : ofs)
  {
    if(!used(of.grass))
      continue;
    for(uint8_t level = 0; level < meadow.size(); level++)
    {
      fv::backdrop::Cover cover = meadow[level];
      cover.perSquareMetre *= of.thick;
      cover.clumps = of.clumps;
      cover.tint   = of.tint;
      cover.salt += 20U * uint32_t(of.grass);
      out.push_back({cover, of.grass, {}, level});
    }
  }
  for(const char* kind : {"meadow", "forest"})
  {
    const fv::backdrop::Drifts drifts = fv::backdrop::drifts(flatSpec(w, kind, density));
    if(!drifted(drifts.covers[0].role))
      continue;
    for(uint8_t level = 0; level < drifts.covers.size(); level++)
      out.push_back({drifts.covers[level], Grass::eNone, std::string_view(kind == std::string_view("meadow") ? "flower" : "fern"), level});
  }
  return out;
}

namespace {

// A tile's middle on its face: (u, v). The plane's x runs along the face's v and its z along its u,
// so a tile's own +y is out of the planet.
glm::dvec2 tileMiddle(const World& w, float size, int32_t x, int32_t z)
{
  const double half = tilePlaneHalf(w);
  return {(double(z) + 0.5) * double(size) / half, (double(x) + 0.5) * double(size) / half};
}

}  // namespace

std::vector<GrassTile> grassTiles(const World& w, std::span<const glm::dvec3> foci, std::span<const TileKind> kinds, GrassWas& was, double size)
{
  std::vector<GrassTile> out;
  if(kinds.empty())
    return out;
  const double half = tilePlaneHalf(w);
  const auto   kindOf = [&](Grass grass, std::string_view drift, uint8_t level) {
    for(size_t k = 0; k < kinds.size(); k++)
      if(kinds[k].grass == grass && kinds[k].drift == drift && kinds[k].level == level)
        return int(k);
    return -1;
  };
  // each focus, as the planet has it
  struct At
  {
    glm::dvec3 d;
    double     over;
  };
  std::vector<At> ats;
  for(const glm::dvec3& f : foci)
    ats.push_back({toPlanet(w, f), std::max(heightOver(w, f), 0.0)});
  fv::backdrop::Spec meadow = flatSpec(w, "meadow", 1.0), forest = flatSpec(w, "forest", 1.0);
  meadow.size = forest.size = size;
  was.pass++;
  size_t asked = 0;
  const double             reach  = double(fv::backdrop::grassReach(meadow).back()) + 16.0;
  for(uint8_t face = 0; face < 6; face++)
  {
    // the foci on this face's plane; those on a face beside it reach over its edge
    std::vector<fv::backdrop::Focus> on;
    for(const At& at : ats)
    {
      const FacePoint p = onFace(face, at.d);
      if(p.face != face)
        continue;  // behind it
      const double x = p.v * half, z = p.u * half;
      if(std::abs(x) < half + reach && std::abs(z) < half + reach)
        on.emplace_back(float(x), float(at.over), float(z));
    }
    if(on.empty())
    {
      was.faces[face].clear();
      continue;
    }
    // what a tile's ground is: the biome it counts as (one for all the tiles of an 8 m square, where
    // biomes meet), or none where nothing grows
    const auto look = [&](float size, int32_t x, int32_t z) -> const Biome* {
      const glm::dvec2 uv = tileMiddle(w, size, x, z);
      if(std::abs(uv.x) >= 1.0 || std::abs(uv.y) >= 1.0)
        return nullptr;  // on the face beside
      const glm::dvec3 d = direction(face, uv.x, uv.y);
      const Sample     s = sample(w, d, 2.0);
      if(bare(s, true) || s.elevation < 2.5 || s.rock > 0.5F || (s.built.taken & kGroundCovered))
        return nullptr;
      // (a road is narrower than a coarse tile, whose middle may be off it: none of a tile lies on one)
      if(w.places && size > 2.0F)
        for(int corner = 0; corner < 4; corner++)
        {
          const double q = 0.25 * double(size) / tilePlaneHalf(w);
          if(sample(w, direction(face, uv.x + (corner & 1 ? q : -q), uv.y + (corner & 2 ? q : -q)), 2.0).built.taken & kGroundCovered)
            return nullptr;
        }
      const int32_t cx = int32_t(std::floor((double(x) + 0.5) * double(size) / 8.0)), cz = int32_t(std::floor((double(z) + 0.5) * double(size) / 8.0));
      return &w.biomes[size_t(biomeBy(w, s, unit(hash(uint32_t(w.spec.seed) + 31U, face, uint32_t(cx), uint32_t(cz)))))];
    };
    // (as found the first time the tile was asked for: GrassWas::grounds)
    const auto biomeAt = [&](float size, int32_t x, int32_t z) -> const Biome* {
      asked++;
      const auto [found, fresh] = was.grounds.try_emplace(GrassWas::Key{face, size, x, z});
      if(fresh)
      {
        const Biome* b      = look(size, x, z);
        found->second.biome = b ? int8_t(b - w.biomes.data()) : int8_t(-1);
      }
      found->second.asked = was.pass;
      return found->second.biome < 0 ? nullptr : &w.biomes[size_t(found->second.biome)];
    };
    const auto                      covers = fv::backdrop::grassCovers(meadow);
    std::vector<fv::backdrop::Tile> grass  = fv::backdrop::grassTiles(meadow, on, was.faces[face]);
    for(const fv::backdrop::Tile& t : grass)
      if(const Biome* b = biomeAt(covers[t.level].tile, t.x, t.z); b && b->grass != Grass::eNone)
        if(const int k = kindOf(b->grass, {}, t.level); k >= 0)
          out.push_back({face, uint8_t(k), t.x, t.z});
    was.faces[face] = std::move(grass);
    for(const fv::backdrop::Spec* flat : {&meadow, &forest})
    {
      const fv::backdrop::Drifts drifts = fv::backdrop::drifts(*flat);
      const std::string_view     role   = flat == &meadow ? "flower" : "fern";
      for(const fv::backdrop::Tile& t : fv::backdrop::driftTiles(*flat, on))
        if(const Biome* b = biomeAt(drifts.covers[t.level].tile, t.x, t.z); b && b->drift == role)
          if(const int k = kindOf(Grass::eNone, role, t.level); k >= 0)
            out.push_back({face, uint8_t(k), t.x, t.z});
    }
  }
  // the tiles left behind are forgotten once they are most of what is kept
  if(was.grounds.size() > 3 * asked + 1024)
    std::erase_if(was.grounds, [&](const auto& ground) { return ground.second.asked != was.pass; });
  std::sort(out.begin(), out.end());
  return out;
}

GrassPlace grassPlace(const World& w, const GrassTile& tile, const TileKind& kind)
{
  const float      size = kind.cover.tile;
  const glm::dvec2 uv   = tileMiddle(w, size, tile.x, tile.z);
  const double     e    = 0.5 * double(size) / tilePlaneHalf(w);
  const auto       at   = [&](double u, double v) {
    const glm::dvec3 d = direction(tile.face, u, v);
    return toScene(w, d, sample(w, d, 1.0).elevation);
  };
  // the tile's own x along the face's v and its z along the face's u, on the ground: how far its
  // square reaches each way, and how it leans
  const glm::dvec3 x = at(uv.x, uv.y + e) - at(uv.x, uv.y - e), z = at(uv.x + e, uv.y) - at(uv.x - e, uv.y);
  const double     lx = glm::length(x), lz = glm::length(z);
  const glm::dvec3 ax = x / lx;
  const glm::dvec3 ay = glm::normalize(glm::cross(z, ax));
  const glm::dvec3 az = glm::cross(ax, ay);
  // a quarter turn or more about its own vertical by its cell, so its neighbours don't repeat it
  const uint32_t turns = hash(uint32_t(w.spec.seed), 7000U + uint32_t(tile.kind), uint32_t(tile.x), uint32_t(tile.z)) & 3U;
  GrassPlace     out;
  out.rotation = glm::quat(glm::quat_cast(glm::dmat3(ax, ay, az)) * glm::angleAxis(0.5 * kPi * double(turns), glm::dvec3(0.0, 1.0, 0.0)));
  const float sx = float(lx / double(size)), sz = float(lz / double(size));
  out.scale    = (turns & 1U) ? glm::vec3(sz, 1.0F, sx) : glm::vec3(sx, 1.0F, sz);
  out.position = at(uv.x, uv.y) - ay * double(kind.cover.sink) * std::max(w.spec.relief, 0.25);
  out.level    = float(glm::dot(ay, glm::normalize(out.position - w.frame.centre)));
  return out;
}

}  // namespace fv::world
