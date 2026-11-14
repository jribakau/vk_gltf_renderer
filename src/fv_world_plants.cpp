#include "fv_world_plants.hpp"

#include <algorithm>
#include <cmath>

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

}  // namespace

// ---- what has a place ----------------------------------------------------------------------------------

Kept keptOn(std::string_view role, double width)
{
  // A quad `width` across is between two and four of its widths from the eye (the finest, under
  // two). Trees are seen for a couple of kilometres: all of them on the finest quads, then fewer and
  // larger at each level out, as far woods are, so a wood keeps its cover with a tenth of its trees;
  // the undergrowth and what lies on the ground only on the finest, a hundred metres or two about.
  // Every plant is an instance the renderer's scene sync walks whenever anything in the scene
  // changes, and a planet's quads change as the camera moves: tens of thousands is what that bears.
  const bool tree = role == "tree" || role == "grown";
  if(width <= 70.0)
    return {1.0F, 1.0F};
  if(!tree)
    return {};
  if(width <= 140.0)
    return {0.6F, 1.3F};
  if(width <= 280.0)
    return {0.3F, 1.8F};
  if(width <= 560.0)
    return {0.1F, 2.6F};
  return {};
}

std::vector<Stand> standing(const World& w, const Quad& q, const glm::dvec3& centre, std::span<const uint32_t> variants, double density)
{
  std::vector<Stand> out;
  if(!(density > 0.0))
    return out;
  const double wide = width(w, q);
  // what draws the quad's ground: a plant stands on the ground as drawn
  const double footprint = wide / double(kGroundCells);
  for(size_t r = 0; r < kStandingRoles.size() && r < variants.size(); r++)
  {
    const std::string_view role = kStandingRoles[r];
    const Kept             kept = keptOn(role, wide);
    if(variants[r] == 0 || kept.share <= 0.0F)
      continue;
    // where each biome grows it, and the thickest of them
    std::array<const Growth*, kMostBiomes> grows{};
    double                                 thickest = 0.0;
    for(size_t b = 0; b < w.biomes.size() && b < kMostBiomes; b++)
      for(const Growth& g : w.biomes[b].plants)
        if(g.role == role)
        {
          grows[b] = &g;
          thickest = std::max(thickest, g.perSquareMetre * density);
        }
    if(!(thickest > 0.0))
      continue;
    const bool                          rock = role == "rock";
    std::vector<std::vector<glm::mat4>> placed(variants[r]);
    overCells(w, q, thickest, 7919U * uint32_t(r + 1), [&](double u, double v, double cell, uint32_t h) {
        if(unit(mix(h + 6U)) >= kept.share)
          return;
        const glm::dvec3 d = direction(q.face, u, v);
        const Sample     s = sample(w, d, footprint);
        if(bare(s, rock) || cleared(w, d) || (!rock && s.rock > 0.5F))
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
          {
            const double level = coverLevel(double(g->cover));
            here *= smooth(level - 0.03, level + 0.03, noise(w, d, double(g->patch), uint32_t(r * kMostBiomes + b)));
          }
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
        if(!rock && (s.cliff > 0.02F || s.steepened > 2.0F) && slopeAt(w, d, s.elevation, footprint) > kStandSlope)
          return;
        const float scale = (most->scaleMin + (most->scaleMax - most->scaleMin) * unit(mix(h + 5U))) * kept.grow;
        placed[mix(h + 3U) % variants[r]].push_back(placeOn(w, d, s.elevation - double(most->sink * scale), centre, unit(mix(h + 4U)), scale));
    });
    for(uint32_t v = 0; v < placed.size(); v++)
      if(!placed[v].empty())
        out.push_back({uint8_t(r), v, std::move(placed[v])});
  }
  return out;
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
  if(!(density > 0.0) || wide > 70.0)
    return out;
  const double footprint = wide / double(kGroundCells);
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
    const double                        thickest = grow->perSquareMetre * density;
    const double                        level    = coverLevel(double(grow->cover));
    std::vector<std::vector<glm::mat4>> placed(of.size());
    overCells(w, q, thickest, 104729U * uint32_t(r + 1), [&](double u, double v, double cell, uint32_t h) {
      const glm::dvec3 d = direction(q.face, u, v);
      const Sample     s = sample(w, d, footprint);
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
      const float    scale   = std::min(grow->scaleMin + (grow->scaleMax - grow->scaleMin) * unit(mix(h + 5U)), tallest);
      if(scale < 0.5F * grow->scaleMin)
        return;
      placed[variant].push_back(placeOn(w, d, s.elevation, centre, unit(mix(h + 4U)), scale));
    });
    for(uint32_t v = 0; v < placed.size(); v++)
      if(!placed[v].empty())
        out.push_back({uint8_t(r), v, std::move(placed[v])});
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
    const auto biomeAt = [&](float size, int32_t x, int32_t z) -> const Biome* {
      const glm::dvec2 uv = tileMiddle(w, size, x, z);
      if(std::abs(uv.x) >= 1.0 || std::abs(uv.y) >= 1.0)
        return nullptr;  // on the face beside
      const glm::dvec3 d = direction(face, uv.x, uv.y);
      const Sample     s = sample(w, d, 2.0);
      if(bare(s, true) || s.elevation < 2.5 || s.rock > 0.5F)
        return nullptr;
      const int32_t cx = int32_t(std::floor((double(x) + 0.5) * double(size) / 8.0)), cz = int32_t(std::floor((double(z) + 0.5) * double(size) / 8.0));
      return &w.biomes[size_t(biomeBy(w, s, unit(hash(uint32_t(w.spec.seed) + 31U, face, uint32_t(cx), uint32_t(cz)))))];
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
