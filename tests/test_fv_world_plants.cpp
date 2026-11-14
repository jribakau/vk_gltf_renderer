// What grows on a planet (src/fv_world_plants.hpp): a quad's plants are its children's, none stands in
// the water or where the figures do, each is on the ground and upright, a wood has more trees than a
// steppe; and the grass's tiles lie on the ground, none of them twice.

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <gtest/gtest.h>

#include "../src/fv_world_map.hpp"
#include "../src/fv_world_plants.hpp"

namespace {

namespace W = fv::world;

const W::World& planet()
{
  static const W::World w = W::make({}, {.map = 128});
  return w;
}

// As many of each role as a kit might have
const std::array<uint32_t, 7> kVariants = {3, 1, 2, 2, 3, 1, 1};

glm::dvec3 spread(uint32_t i, uint32_t count)
{
  const double y   = 1.0 - 2.0 * (double(i) + 0.5) / double(count);
  const double r   = std::sqrt(std::max(1.0 - y * y, 0.0));
  const double phi = double(i) * 2.399963229728653;
  return {r * std::cos(phi), y, r * std::sin(phi)};
}

// The quad of `level` a direction is on
W::Quad quadAt(const glm::dvec3& d, uint8_t level)
{
  const W::FacePoint p = W::locate(d);
  const double       n = double(uint64_t(1) << level);
  return {p.face, level, uint32_t(std::clamp((p.u + 1.0) * 0.5 * n, 0.0, n - 1.0)), uint32_t(std::clamp((p.v + 1.0) * 0.5 * n, 0.0, n - 1.0))};
}

// Every place of a quad's plants, in the scene, by role
std::map<uint8_t, std::vector<glm::dvec3>> placesOf(const W::World& w, const W::Quad& q)
{
  const glm::dvec3 centre = W::toScene(w, W::direction(q.face, 0.0, 0.0), 0.0);
  std::map<uint8_t, std::vector<glm::dvec3>> out;
  for(const W::Stand& s : W::standing(w, q, centre, kVariants))
    for(const glm::mat4& m : s.places)
      out[s.role].push_back(centre + glm::dvec3(m[3]));
  return out;
}

// A few quads of a level with plenty growing on them
std::vector<W::Quad> grown(const W::World& w, uint8_t level, size_t count, size_t least)
{
  std::vector<W::Quad> out;
  for(uint32_t i = 0; i < 4000 && out.size() < count; i++)
  {
    const glm::dvec3 d = spread(i, 4000);
    const W::Sample  s = W::sample(w, d, 30.0);
    if(s.elevation < 5.0 || w.biomes[size_t(s.biome)].plants.size() < 3)
      continue;
    const W::Quad q = quadAt(d, level);
    size_t        n = 0;
    for(const auto& [role, places] : placesOf(w, q))
      n += places.size();
    if(n >= least)
      out.push_back(q);
  }
  return out;
}

}  // namespace

TEST(WorldPlants, AQuadsPlantsAreAmongItsChildrens)
{
  const W::World& w = planet();
  // a level whose quads keep a share of their trees, over the finest, which keep every plant (keptOn)
  const uint8_t level = uint8_t(W::finestLevel(w) - 1);
  const W::Kept kept  = W::keptOn("tree", W::width(w, {0, level, 0, 0}));
  ASSERT_GT(kept.share, 0.3F);
  ASSERT_LT(kept.share, 1.0F);
  ASSERT_EQ(W::keptOn("tree", W::width(w, {0, W::finestLevel(w), 0, 0})).share, 1.0F);
  const std::vector<W::Quad> quads = grown(w, level, 4, 25);
  ASSERT_GE(quads.size(), 2U);
  for(const W::Quad& q : quads)
  {
    const auto whole = placesOf(w, q);
    std::map<uint8_t, std::vector<glm::dvec3>> parts;
    for(const W::Quad& child : W::children(q))
      for(const auto& [role, places] : placesOf(w, child))
        parts[role].insert(parts[role].end(), places.begin(), places.end());
    // its trees (and only trees: the undergrowth is the finest quads') stand where trees of its
    // children do, and are about that share of them
    size_t all = 0, found = 0;
    for(const auto& [role, places] : whole)
    {
      EXPECT_EQ(W::kStandingRoles[role], "tree");
      for(const glm::dvec3& p : places)
      {
        all++;
        // (a child's ground is drawn finer, so a plant's foot may be a little higher or lower on it)
        found += std::any_of(parts[role].begin(), parts[role].end(), [&](const glm::dvec3& o) { return glm::distance(o, p) < 2.5; });
      }
    }
    ASSERT_GT(all, 10U);
    EXPECT_GT(double(found), 0.95 * double(all));
    EXPECT_NEAR(double(all), double(kept.share) * double(parts[0].size()), 0.25 * double(all) + 4.0);
  }
}

TEST(WorldPlants, EachStandsOnTheGroundUprightAndOutOfTheWater)
{
  const W::World& w     = planet();
  const uint8_t   level = W::finestLevel(w);
  // quads with plenty growing, those round the figures, and those along the rivers
  std::vector<W::Quad> quads = grown(w, level, 6, 10);
  for(double a = 0.0; a < 6.28; a += 0.8)
    for(const double away : {0.0, 40.0, 90.0})
      quads.push_back(quadAt(W::toPlanet(w, glm::dvec3(std::cos(a), 0.0, std::sin(a)) * away), level));
  for(size_t k = 0; k < w.base->pieces.size(); k += 41)
    quads.push_back(quadAt(w.base->pieces[k].b, level));
  std::sort(quads.begin(), quads.end());
  quads.erase(std::unique(quads.begin(), quads.end()), quads.end());
  size_t plants = 0;
  for(const W::Quad& q : quads)
  {
    const glm::dvec3 centre = W::toScene(w, W::direction(q.face, 0.0, 0.0), 0.0);
    for(const W::Stand& stand : W::standing(w, q, centre, kVariants))
    {
      ASSERT_LT(stand.role, W::kStandingRoles.size());
      ASSERT_LT(stand.variant, kVariants[stand.role]);
      for(const glm::mat4& m : stand.places)
      {
        plants++;
        const glm::dvec3 at = centre + glm::dvec3(m[3]);
        double           r  = 0.0;
        const glm::dvec3 d  = W::toPlanet(w, at, &r);
        const W::Sample  s  = W::sample(w, d, W::width(w, q) / W::kGroundCells);
        // on the ground, sunk a little at the most
        EXPECT_LT(std::abs(r - w.spec.radius - s.elevation), 1.0);
        // upright: its own +y is the planet's vertical there, and it is as large each way
        const glm::dvec3 up    = glm::normalize(at - w.frame.centre);
        const double     scale = glm::length(glm::dvec3(m[1]));
        EXPECT_GT(glm::dot(glm::dvec3(m[1]) / scale, up), 0.9999);
        EXPECT_NEAR(glm::length(glm::dvec3(m[0])), scale, 1e-3 * scale);
        EXPECT_GT(scale, 0.1);
        EXPECT_LT(scale, 6.0);
        // out of the sea and off its beach's wet edge, out of rivers and lakes
        EXPECT_GT(s.elevation, 0.25);
        EXPECT_FALSE(s.water > s.elevation);
        EXPECT_FALSE(s.inland.river && s.inland.riverAway < s.inland.riverHalf);
        // and not where the figures stand
        EXPECT_GT(glm::length(at), W::kClearing + 3.0);
      }
    }
  }
  EXPECT_GT(plants, 200U);
}

TEST(WorldPlants, AWoodHasMoreTreesThanASteppe)
{
  const W::World& w     = planet();
  const uint8_t   level = W::finestLevel(w);
  std::map<std::string, std::pair<size_t, size_t>> trees;  // by biome: trees, quads
  for(uint32_t i = 0; i < 3000; i++)
  {
    const glm::dvec3 d = spread(i, 3000);
    const W::Sample  s = W::sample(w, d, 30.0);
    if(s.elevation < 10.0 || s.share[size_t(s.biome)] < 0.95F || s.rock > 0.1F)
      continue;
    auto& [count, quads] = trees[std::string(w.biomes[size_t(s.biome)].name)];
    if(quads >= 12)
      continue;
    quads++;
    const auto places = placesOf(w, quadAt(d, level));
    if(const auto found = places.find(0); found != places.end())
      count += found->second.size();
  }
  ASSERT_GE(trees["forest"].second, 5U);
  ASSERT_GE(trees["steppe"].second, 5U);
  const auto per = [&](const char* biome) { return double(trees[biome].first) / double(trees[biome].second); };
  EXPECT_GT(per("forest"), 30.0);  // a quad some 60 m across of fir wood
  EXPECT_GT(per("forest"), 2.0 * per("grassland"));
  EXPECT_GT(per("grassland"), 2.0 * per("steppe"));
  EXPECT_EQ(trees["snow"].first, 0U);
  EXPECT_EQ(trees["desert"].first, 0U);
}

TEST(WorldPlants, FarQuadsKeepFewerAndLarger)
{
  const W::World& w = planet();
  EXPECT_EQ(W::keptOn("shrub", 300.0).share, 0.0F);
  EXPECT_GT(W::keptOn("tree", 300.0).share, 0.0F);
  EXPECT_LT(W::keptOn("tree", 500.0).share, W::keptOn("tree", 200.0).share);
  EXPECT_GT(W::keptOn("tree", 500.0).grow, W::keptOn("tree", 200.0).grow);
  EXPECT_EQ(W::keptOn("tree", 1000.0).share, 0.0F);
  // a quad kilometres across has nothing with a place of its own, and is asked in no time
  EXPECT_TRUE(W::standing(w, {0, 2, 1, 1}, glm::dvec3(0.0), kVariants).empty());
  // no kit, or no density: nothing
  const std::array<uint32_t, 7> none{};
  const W::Quad                 near = quadAt(w.frame.site, W::finestLevel(w));
  EXPECT_TRUE(W::standing(w, near, glm::dvec3(0.0), none).empty());
  EXPECT_TRUE(W::standing(w, near, glm::dvec3(0.0), kVariants, 0.0).empty());
}

TEST(WorldPlants, TheGrassIsTilesOnTheGroundRoundWhereOneLooksFrom)
{
  const W::World&                w     = planet();
  const std::vector<W::TileKind> kinds = W::tileKinds(w);
  ASSERT_GE(kinds.size(), 3U);
  // a place with grass: the figures' own, if it has any, or the first grassland found
  glm::dvec3 from(0.0, 1.7, 0.0);
  for(uint32_t i = 0; i < 3000; i++)
  {
    const W::Sample s = W::sample(w, spread(i, 3000), 10.0);
    if(s.elevation > 10.0 && w.biomes[size_t(s.biome)].name == "grassland" && s.share[size_t(s.biome)] > 0.95F && s.shore > 200.0F)
    {
      from = W::toScene(w, spread(i, 3000), s.elevation + 1.7);
      break;
    }
  }
  W::GrassWas                       was;
  const std::vector<glm::dvec3>     foci{from};
  const std::vector<W::GrassTile>   tiles = W::grassTiles(w, foci, kinds, was);
  ASSERT_GT(tiles.size(), 100U);
  // no square of ground under two tiles of the grass: by a tile's reach in the finest level's cells
  std::set<std::tuple<int, int64_t, int64_t>> covered;
  size_t                                      thick = 0, level = 0;
  for(const W::GrassTile& t : tiles)
  {
    ASSERT_LT(t.kind, kinds.size());
    const W::TileKind& kind = kinds[t.kind];
    if(kind.grass != W::Grass::eNone)
    {
      const int cells = 1 << kind.level;
      thick += kind.level == 0;
      for(int j = 0; j < cells; j++)
        for(int i = 0; i < cells; i++)
          EXPECT_TRUE(covered.insert({int(t.face), int64_t(t.x) * cells + i, int64_t(t.z) * cells + j}).second);
    }
    // on the ground, a square of about its own size, its +y out of the planet
    const W::GrassPlace place = W::grassPlace(w, t, kind);
    if(place.level < W::kGrassLevel)
      continue;  // too steep for grass: left out
    level++;
    double              r     = 0.0;
    const glm::dvec3    d     = W::toPlanet(w, place.position, &r);
    const W::Sample     s     = W::sample(w, d, 1.0);
    EXPECT_LT(std::abs(r - w.spec.radius - s.elevation), 0.6);
    EXPECT_FALSE(s.water > s.elevation);
    EXPECT_NEAR(glm::length(place.rotation), 1.0F, 1e-4F);
    const glm::dvec3 up = glm::normalize(place.position - w.frame.centre);
    EXPECT_NEAR(glm::dot(glm::dvec3(place.rotation * glm::vec3(0.0F, 1.0F, 0.0F)), up), double(place.level), 1e-3);
    for(const int c : {0, 2})
    {
      EXPECT_GT(place.scale[c], 0.6F);
      EXPECT_LT(place.scale[c], 1.6F);
    }
    // near where one looks from
    EXPECT_LT(glm::distance(place.position, from), 260.0);
  }
  // thick grass underfoot, and most of the ground level enough for it
  EXPECT_GT(thick, 20U);
  EXPECT_GT(level, tiles.size() / 2);
  // asked again from the same place, the same tiles; from high up, none
  EXPECT_EQ(W::grassTiles(w, foci, kinds, was), tiles);
  W::GrassWas                   fresh;
  const std::vector<glm::dvec3> high{from + glm::normalize(from - w.frame.centre) * 800.0};
  EXPECT_TRUE(W::grassTiles(w, high, kinds, fresh).empty());
}

TEST(WorldSea, ItsPlantsAreMeshesOfASeed)
{
  const std::vector<W::SeaMesh> meshes = W::seaMeshes(1), again = W::seaMeshes(1), other = W::seaMeshes(2);
  ASSERT_EQ(meshes.size(), again.size());
  std::map<uint8_t, int> variants;
  for(size_t k = 0; k < meshes.size(); k++)
  {
    const W::SeaMesh& m = meshes[k];
    ASSERT_LT(m.role, W::kSeaRoles.size());
    variants[m.role]++;
    // the same for a seed, another for another
    EXPECT_EQ(m.positions, again[k].positions);
    EXPECT_NE(m.positions, other[k].positions);
    ASSERT_FALSE(m.positions.empty());
    ASSERT_EQ(m.positions.size(), m.normals.size());
    ASSERT_EQ(m.positions.size(), m.colors.size());
    EXPECT_EQ(m.indices.size() % 3, 0U);
    EXPECT_LT(m.indices.size() / 3, 20000U);
    for(const uint32_t i : m.indices)
      ASSERT_LT(i, m.positions.size());
    // it stands on y = 0, some decimetres to some metres high, and is no wider than a few metres
    EXPECT_GT(m.height, 0.25F);
    EXPECT_LT(m.height, 12.0F);
    glm::vec3 middle(0.0F);
    for(size_t v = 0; v < m.positions.size(); v++)
    {
      EXPECT_GT(m.positions[v].y, -0.3F);
      EXPECT_LT(std::abs(m.positions[v].x), 3.0F);
      EXPECT_NEAR(glm::length(m.normals[v]), 1.0F, 1e-3F);
      for(int c = 0; c < 3; c++)
      {
        EXPECT_GE(m.colors[v][c], 0.0F);
        EXPECT_LT(m.colors[v][c], 0.9F);
      }
      middle += m.positions[v] / float(m.positions.size());
    }
    // a rock's faces face out of it
    if(W::kSeaRoles[m.role] == "searock")
    {
      size_t out = 0;
      for(size_t v = 0; v < m.positions.size(); v++)
        out += glm::dot(m.normals[v], m.positions[v] - middle) > 0.0F;
      EXPECT_GT(out, m.positions.size() * 9 / 10);
    }
  }
  EXPECT_EQ(variants.size(), W::kSeaRoles.size());
}

TEST(WorldSea, ItsPlantsStandOnTheSeasFloorUnderItsSurface)
{
  const W::World&               w      = planet();
  const std::vector<W::SeaMesh> meshes = W::seaMeshes(w.spec.seed);
  const uint8_t                 level  = W::finestLevel(w);
  std::map<uint8_t, size_t>     byRole;
  size_t                        quads = 0;
  // (shallows from all over the planet, the warm seas and the cold: a stride through the sphere's points)
  for(uint32_t k = 0; k < 20000 && quads < 120; k++)
  {
    const glm::dvec3 d = spread((k * 7919U) % 20000U, 20000);
    const W::Sample  s = W::sample(w, d, 30.0);
    if(s.elevation > -1.5 || s.elevation < -25.0)
      continue;
    quads++;
    const W::Quad    q      = quadAt(d, level);
    const glm::dvec3 centre = W::toScene(w, W::direction(q.face, 0.0, 0.0), 0.0);
    for(const W::Stand& stand : W::seaStanding(w, q, centre, meshes))
    {
      ASSERT_LT(stand.role, W::kSeaRoles.size());
      // its mesh: the role's `variant`th
      const W::SeaMesh* mesh = nullptr;
      uint32_t          nth  = 0;
      for(const W::SeaMesh& m : meshes)
        if(m.role == stand.role && nth++ == stand.variant)
          mesh = &m;
      ASSERT_TRUE(mesh);
      byRole[stand.role] += stand.places.size();
      for(const glm::mat4& m : stand.places)
      {
        const glm::dvec3 at = centre + glm::dvec3(m[3]);
        double           r  = 0.0;
        const glm::dvec3 on = W::toPlanet(w, at, &r);
        const W::Sample  here = W::sample(w, on, W::width(w, q) / W::kGroundCells);
        // on the sea's floor, and its top under the water
        EXPECT_LT(here.elevation, -0.4);
        EXPECT_NEAR(r - w.spec.radius, here.elevation, 0.05);
        const double scale = glm::length(glm::dvec3(m[1]));
        EXPECT_LT(scale * double(mesh->height), -here.elevation);
        EXPECT_GT(glm::dot(glm::dvec3(m[1]) / scale, glm::normalize(at - w.frame.centre)), 0.9999);
        // kelp in cool water, coral in warm
        const double warm = W::temperatureAt(w, on, 0.0);
        if(W::kSeaRoles[stand.role] == "kelp")
          EXPECT_LT(warm, 21.5);
        if(W::kSeaRoles[stand.role] == "coral")
          EXPECT_GT(warm, 16.5);
      }
    }
  }
  ASSERT_GT(quads, 20U);
  // every role somewhere on the shallows
  for(uint8_t role = 0; role < W::kSeaRoles.size(); role++)
    EXPECT_GT(byRole[role], 5U) << W::kSeaRoles[role];
  EXPECT_GT(byRole[0], 200U);
  // none on a quad too coarse to be seen from the water, and none on dry land
  EXPECT_TRUE(W::seaStanding(w, {0, uint8_t(level - 2), 3, 3}, glm::dvec3(0.0), meshes).empty());
  EXPECT_TRUE(W::seaStanding(w, quadAt(w.frame.site, level), glm::dvec3(0.0), meshes).empty());
}
