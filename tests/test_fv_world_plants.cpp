// What grows on a planet (src/fv_world_plants.hpp): a plant is the same on whichever quad holds it,
// the larger are seen from further and those seen from one place stay under their count, what is
// seen from a place is on the quads that draw there, none stands in the water or where the figures
// do, each is on the ground and upright, a wood has more trees than a steppe; and the grass's tiles
// lie on the ground, none of them twice.

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

// Every plant of a quad: where it stands in the scene, how large it is and how far it is seen from
struct Plant
{
  uint8_t    role = 0;
  glm::dvec3 at{0.0};
  double     scale = 1.0, reach = 0.0;
};
std::vector<Plant> plantsOf(const W::World& w, const W::Quad& q)
{
  const glm::dvec3   centre = W::toScene(w, W::direction(q.face, 0.0, 0.0), 0.0);
  std::vector<Plant> out;
  for(const W::Stand& s : W::standing(w, q, centre, kVariants))
  {
    EXPECT_EQ(s.places.size(), s.reach.size());
    EXPECT_TRUE(std::is_sorted(s.reach.begin(), s.reach.end(), std::greater<float>())) << "the furthest seen first";
    for(size_t i = 0; i < s.places.size(); i++)
      out.push_back({s.role, centre + glm::dvec3(s.places[i][3]), double(glm::length(glm::vec3(s.places[i][1]))), double(s.reach[i])});
  }
  return out;
}

// A place in a wood, well inland
glm::dvec3 inAWood(const W::World& w, uint32_t skip = 0)
{
  for(uint32_t i = 0; i < 6000; i++)
  {
    const glm::dvec3 d = spread(i, 6000);
    const W::Sample  s = W::sample(w, d, 30.0);
    if(s.elevation > 10.0 && w.biomes[size_t(s.biome)].name == "forest" && s.share[size_t(s.biome)] > 0.95F && s.shore > 500.0F && skip-- == 0)
      return d;
  }
  ADD_FAILURE() << "no wood";
  return w.frame.site;
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

TEST(WorldPlants, APlantIsTheSameOnWhicheverQuadHoldsIt)
{
  const W::World& w = planet();
  // levels over the finest: a quad holds those of its children's plants that are seen from where
  // it can draw (heldFrom), each in its place, of its size and with its reach
  size_t fewer = 0;
  for(const int up : {2, 3})
  {
    const uint8_t              level = uint8_t(W::finestLevel(w) - up);
    const std::vector<W::Quad> quads = grown(w, level, 3, 25);
    ASSERT_GE(quads.size(), 2U) << up;
    for(const W::Quad& q : quads)
    {
      const double             held  = W::heldFrom(w, q);
      const std::vector<Plant> whole = plantsOf(w, q);
      std::vector<Plant>       parts;
      for(const W::Quad& child : W::children(q))
      {
        EXPECT_LT(W::heldFrom(w, child), held);
        const std::vector<Plant> of = plantsOf(w, child);
        parts.insert(parts.end(), of.begin(), of.end());
      }
      // (a child's ground is drawn finer, so a plant's foot may be higher or lower on it: the more,
      // the coarser the two are)
      const double off  = up <= 2 ? 2.5 : 10.0;
      const auto   same = [&](const Plant& a, const Plant& b) {
        return a.role == b.role && glm::distance(a.at, b.at) < off && std::abs(a.scale - b.scale) < 1e-4 && std::abs(a.reach - b.reach) < 1e-2;
      };
      ASSERT_GT(whole.size(), 10U);
      size_t heldToo = 0;
      for(const Plant& plant : whole)
      {
        EXPECT_GE(plant.reach, held);
        EXPECT_TRUE(std::any_of(parts.begin(), parts.end(), [&](const Plant& o) { return same(plant, o); })) << int(plant.role);
      }
      for(const Plant& plant : parts)
        if(plant.reach >= held)
        {
          heldToo++;
          EXPECT_TRUE(std::any_of(whole.begin(), whole.end(), [&](const Plant& o) { return same(plant, o); })) << int(plant.role);
        }
      EXPECT_EQ(heldToo, whole.size());
      EXPECT_GE(parts.size(), whole.size());
      fewer += parts.size() > whole.size();
    }
  }
  EXPECT_GT(fewer, 0U) << "the children hold the smaller ones too";
  // the finest hold every plant
  EXPECT_EQ(W::heldFrom(w, {0, W::finestLevel(w), 0, 0}), 0.0);
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

TEST(WorldPlants, TheLargerAreSeenFromFurther)
{
  const W::World& w = planet();
  // a role's reach: its largest from `most`, every one from `all`, those between by their size
  const W::Reach wood = W::reachOf("tree", 0.0234);
  EXPECT_DOUBLE_EQ(wood.of(0.0), wood.most);
  EXPECT_DOUBLE_EQ(wood.of(1.0), wood.all);
  EXPECT_GT(wood.most, 8000.0);
  EXPECT_GE(wood.all, 300.0);
  EXPECT_LT(wood.all, 900.0);
  for(double rank = 0.05; rank < 1.0; rank += 0.05)
    EXPECT_GE(wood.of(rank - 0.05), wood.of(rank));
  // thinner woods are seen whole from further, thicker ones from no nearer than a few hundred metres
  EXPECT_GT(W::reachOf("tree", 0.002).all, 2.0 * wood.all);
  EXPECT_DOUBLE_EQ(W::reachOf("tree", 1e-5).all, wood.most);
  EXPECT_GE(W::reachOf("tree", 1.0).all, 300.0);
  // the undergrowth, a hundred metres or a few
  const W::Reach bushes = W::reachOf("shrub", 0.0225);
  EXPECT_GE(bushes.all, 100.0);
  EXPECT_LE(bushes.most, 1000.0);
  // and the sea floor's no further than through water
  EXPECT_LE(W::reachOf("kelp", 0.01).most, 500.0);
  // a plant shows none of itself past its reach, and all of it well within
  EXPECT_EQ(W::grownAt(1001.0, 1000.0), 0.0F);
  EXPECT_GT(W::grownAt(990.0, 1000.0), 0.0F);
  EXPECT_LT(W::grownAt(990.0, 1000.0), 0.2F);
  EXPECT_EQ(W::grownAt(800.0, 1000.0), 1.0F);
  // on a quad: among a role's plants of one biome, the further seen are the larger
  const W::Quad      q = quadAt(inAWood(w), uint8_t(W::finestLevel(w) - 2));
  std::vector<Plant> trees;
  for(const Plant& plant : plantsOf(w, q))
    if(plant.role == 0)
      trees.push_back(plant);
  ASSERT_GT(trees.size(), 20U);
  std::sort(trees.begin(), trees.end(), [](const Plant& a, const Plant& b) { return a.reach > b.reach; });
  EXPECT_GT(trees.front().scale, trees.back().scale);
  // a quad kilometres across has nothing with a place of its own, and is asked in no time
  EXPECT_TRUE(W::standing(w, {0, 2, 1, 1}, glm::dvec3(0.0), kVariants).empty());
  // no kit, or no density: nothing
  const std::array<uint32_t, 7> none{};
  const W::Quad                 underfoot = quadAt(w.frame.site, W::finestLevel(w));
  EXPECT_TRUE(W::standing(w, underfoot, glm::dvec3(0.0), none).empty());
  EXPECT_TRUE(W::standing(w, underfoot, glm::dvec3(0.0), kVariants, 0.0).empty());
}

TEST(WorldPlants, WhatIsSeenFromAPlaceIsOnTheQuadsThatDrawThere)
{
  const W::World& w = planet();
  for(const uint32_t which : {0U, 40U})
  {
    const glm::dvec3   d   = inAWood(w, which);
    const glm::dvec3   eye = W::toScene(w, d, W::sample(w, d, 1.0).elevation + 30.0);
    const W::Selection cut = W::select(w, eye);
    // those seen: on the quads that draw, each within its reach of the eye
    std::map<W::Quad, std::vector<Plant>> drawn;
    size_t                                trees = 0, others = 0;
    for(const W::Quad& q : cut.leaves)
    {
      if(W::heldFrom(w, q) > 12000.0)
        continue;  // (too far for any)
      std::vector<Plant> of = plantsOf(w, q);
      for(const Plant& plant : of)
        if(glm::distance(plant.at, eye) < plant.reach)
          (plant.role <= 1 ? trees : others)++;
      drawn[q] = std::move(of);
    }
    // a wood all round: as many as a place may see, and no more
    EXPECT_GT(trees, 4000U);
    EXPECT_LT(double(trees), 1.15 * W::kTreesSeen);
    EXPECT_LT(double(others), 1.15 * W::kUndergrowthSeen * 5.0);
    // and every plant there is that the eye is within reach of is among them: of the finest quads
    // here and there about the eye, none is missing from the quad that draws its ground
    size_t looked = 0;
    for(uint32_t i = 0; i < 60; i++)
    {
      const double     away = 30.0 + 7000.0 * double(i) / 60.0, turn = 2.4 * double(i);
      const glm::dvec3 at   = eye + glm::dvec3(std::cos(turn), 0.0, std::sin(turn)) * away;
      const W::Quad    fine = quadAt(W::toPlanet(w, at), W::finestLevel(w));
      const auto       leaf = std::find_if(cut.leaves.begin(), cut.leaves.end(), [&](const W::Quad& q) {
        return q.face == fine.face && q.level <= fine.level && (fine.x >> (fine.level - q.level)) == q.x && (fine.y >> (fine.level - q.level)) == q.y;
      });
      ASSERT_NE(leaf, cut.leaves.end());
      const std::vector<Plant>& held = drawn[*leaf];
      for(const Plant& plant : plantsOf(w, fine))
      {
        if(glm::distance(plant.at, eye) >= 0.97 * plant.reach)
          continue;
        looked++;
        EXPECT_TRUE(std::any_of(held.begin(), held.end(), [&](const Plant& o) {
          return o.role == plant.role && glm::distance(o.at, plant.at) < 2.5 && std::abs(o.scale - plant.scale) < 1e-4;
        })) << "a " << W::kStandingRoles[plant.role] << " " << glm::distance(plant.at, eye) << " m off, seen from " << plant.reach << ", on a quad "
            << W::width(w, *leaf) << " m across";
      }
    }
    EXPECT_GT(looked, 50U);
  }
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
  // asked again, and from a few paces on: what grows on a tile's ground is remembered, and the tiles
  // are those a first asking finds there
  EXPECT_GT(was.grounds.size(), tiles.size() / 2);
  EXPECT_EQ(W::grassTiles(w, foci, kinds, was), tiles);
  {
    const std::vector<glm::dvec3> on{from + glm::dvec3(9.0, 0.0, 5.0)};
    W::GrassWas                   again = was, afresh = was;
    afresh.grounds.clear();
    EXPECT_EQ(W::grassTiles(w, on, kinds, again), W::grassTiles(w, on, kinds, afresh));
    // far away, and back: the tiles left behind are forgotten, and found again the same
    const std::vector<glm::dvec3> off{W::toScene(w, -W::toPlanet(w, from), 50.0)};
    for(int i = 0; i < 3; i++)
      W::grassTiles(w, off, kinds, again);
    EXPECT_LT(again.grounds.size(), 4 * tiles.size() + 1024);
    W::GrassWas back = was;
    EXPECT_EQ(W::grassTiles(w, foci, kinds, back), tiles);
  }
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
