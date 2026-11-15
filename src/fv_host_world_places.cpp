/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

//-------------------------------------------------------------------------------------------------------
// What people left on a planet, drawn (fv_world_places.hpp says what stands where and makes the ground
// theirs; this holds the models and stands them).
//
// - The catalogue is a folder the app names (settings.apply's placesDir): its places.json is read on a
//   worker when the folder is given, whatever the backdrop is. A planet made after that is settled on
//   the worker that makes it; one that stands when the catalogue comes, or when the backdrop's `places`
//   is another, is settled on a worker of its own and its quads made again for the ground the places
//   take (restreamPlanet: what draws stays until the new first cut is whole).
// - A site near the eye is shown (fv::world::sitesSeen): an entity a model of its pack, the model's
//   places in the site as copies, under the backdrop's root. The world's own, as its plants are: not
//   props, not selected, not saved, outside the floor and what the scene's standing reads.
// - A pack's models are read on a worker when the first site of it comes near, then brought into the
//   scene a few a frame, hidden under the root: what every site of that pack is copies of. So a
//   planet holds a pack's meshes once however many places are of it. A pack no site shows is kept
//   while it is among the last few used, then let go.
// - What grows over a place (PackPiece::growth: thousands of plants) is shown from nearer than the
//   place itself.
//-------------------------------------------------------------------------------------------------------

#include "fv_host_impl.hpp"

#include <fstream>

#include <nvutils/logger.hpp>

#include "fv_world_places.hpp"

namespace W = fv::world;

namespace {

// A pack's models are brought into the scene this many a frame
constexpr size_t kPackModelsAFrame = 2;
// Packs no site shows are held while they are among this many last used
constexpr size_t kPacksKept = 8;
// What grows over a place shows from this near, metres, and stays a little further
constexpr double kGrowthSeen = 260.0;
// The sites are looked over again once the eye has gone this far, metres
constexpr double kSeenAgain = 4.0;

}  // namespace

void FvHost::Impl::readPlaces()
{
  std::filesystem::path dir;
  {
    std::lock_guard lock(settingsMutex);
    dir = settings.placesDir;
  }
  const uint64_t request = ++placesRequest;
  if(dir.empty())
  {
    placesCatalogue.reset();
    placesDir.clear();
    settlePlanet();
    return;
  }
  loaders.spawn([this, dir, request] {
    std::shared_ptr<const W::Catalogue> catalogue;
    std::string                         error;
    try
    {
      std::ifstream in(dir / "places.json", std::ios::binary);
      if(!in)
        throw std::runtime_error("no places.json");
      catalogue = W::readCatalogue(nlohmann::json::parse(in));
    }
    catch(const std::exception& e)
    {
      error = e.what();
    }
    post([this, dir, catalogue, error, request] {
      if(request != placesRequest)
        return;  // the folder is another since
      if(catalogue)
        LOGI("planet: its places read: %zu packs of %zu kinds\n", catalogue->packs.size(), catalogue->kinds.size());
      // (a folder with no catalogue is a planet nobody lived on; one that cannot be read is a fault)
      else if(error.rfind("no places.json", 0) != 0)
        notice("The planet's places could not be read: " + error, "error");
      placesCatalogue = catalogue;
      placesDir       = dir;
      settlePlanet();
    });
  });
}

void FvHost::Impl::settlePlanet()
{
  if(!planetBuilt || !planetBuilt->world || !backdropBuilt || !renderer->feedsHostScene())
    return;  // (one being made is settled as it is made, or when it has come)
  PlanetBuilt& p      = *planetBuilt;
  const double amount = placesCatalogue ? backdropSpec.places : 0.0;
  if(p.settledWith == placesCatalogue && p.settledAmount == amount && p.settledPlan == backdropSpec.settlement)
    return;
  p.settledWith         = placesCatalogue;
  p.settledAmount       = amount;
  p.settledPlan         = backdropSpec.settlement;
  const uint64_t asked = ++p.settling;
  if(!(amount > 0.0))
  {
    placePlaces(nullptr);
    return;
  }
  loaders.spawn([this, world = p.world, catalogue = placesCatalogue, amount, plan = p.settledPlan, stop = planetStop, generation = p.made, asked] {
    const auto   started = std::chrono::steady_clock::now();
    const auto   places  = W::settle(W::withPlaces(*world, nullptr), catalogue, amount, stop.get(), plan);
    const double ms      = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    post([this, places, stop, generation, asked, ms] {
      if(stop->load() || !planetBuilt || planetBuilt->made != generation || planetBuilt->settling != asked)
        return;  // of a planet gone since, or settled otherwise
      if(places)
      {
        LOGI("planet: settled in %.0f ms: %u places, %zu single things, %zu pieces of road\n", ms, places->joined, places->sites.size() - places->joined, places->roads.size());
        sayUnplaced(*places);
      }
      placePlaces(places);
    });
  });
}

void FvHost::Impl::sayUnplaced(const W::Places& places)
{
  for(const std::string& what : places.unplaced)
    notice("The planet file's place " + what);
}

void FvHost::Impl::placePlaces(std::shared_ptr<const W::Places> places)
{
  PlanetBuilt& p = *planetBuilt;
  if(p.world->places == places)
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  dropSites(scene);
  p.world = std::make_shared<const W::World>(W::withPlaces(*p.world, std::move(places)));
  // its ground is another under them: its quads again, what draws staying until they have come
  restreamPlanet(scene);
  p.tiledFor = glm::dvec3(1.0e30);
  publishBackdrop();  // (how many it has, and where the nearest is)
}

void FvHost::Impl::dropSites(rs::RetainedScene& scene)
{
  PlanetBuilt& p = *planetBuilt;
  for(auto& [site, shown] : p.sites)
    for(const std::vector<rs::EntityId>* entities : {&shown.entities, &shown.growth})
      for(const rs::EntityId e : *entities)
        if(scene.entity(e))
          scene.destroyEntity(e);
  for(auto& [pack, held] : p.packs)
    for(const PlanetBuilt::HeldModel& model : held.models)
      for(const rs::EntityId e : model.roots)
        if(scene.entity(e))
          scene.destroyEntity(e);
  p.sites.clear();
  p.packs.clear();
  p.seen.clear();
  p.seenFor = glm::dvec3(1.0e30);
}

void FvHost::Impl::readPack(uint32_t pack)
{
  PlanetBuilt&           p    = *planetBuilt;
  PlanetBuilt::HeldPack& held = p.packs[pack];
  held.reading                = true;
  loaders.spawn([this, pack, catalogue = p.world->places->catalogue, dir = placesDir, generation = p.made, places = p.world->places] {
    const W::Pack&                                 of = catalogue->packs[pack];
    std::vector<std::shared_ptr<tinygltf::Model>>  read(of.pieces.size());
    std::string                                    missing;
    for(size_t k = 0; k < of.pieces.size(); k++)
    {
      auto        model = std::make_shared<tinygltf::Model>();
      std::string error;
      if(fv::composer::readModelFile(dir / pathOf(of.pieces[k].file), *model, error))
        read[k] = std::move(model);
      else if(missing.empty())
        missing = of.pieces[k].file + ": " + error;
    }
    post([this, pack, read = std::move(read), missing, generation, places]() mutable {
      if(!planetBuilt || planetBuilt->made != generation || !planetBuilt->world || planetBuilt->world->places != places)
        return;  // of a planet gone since, or one settled otherwise
      const auto found = planetBuilt->packs.find(pack);
      if(found == planetBuilt->packs.end())
        return;
      // (once a pack: a place with a piece missing stands without it)
      if(!missing.empty())
        notice("A place of the planet is missing a model: " + missing);
      found->second.read    = std::move(read);
      found->second.reading = false;
      found->second.models.resize(found->second.read.size());
    });
  });
}

bool FvHost::Impl::holdPack(rs::RetainedScene& scene, uint32_t pack, size_t& budget)
{
  PlanetBuilt&           p    = *planetBuilt;
  PlanetBuilt::HeldPack& held = p.packs[pack];
  if(held.held)
    return true;
  if(held.reading)
    return false;
  if(held.read.empty() && held.models.empty())
  {
    readPack(pack);
    return false;
  }
  const W::Pack&  of   = p.world->places->catalogue->packs[pack];
  const glm::mat4 root = glm::inverse(scene.world(backdropBuilt->root));
  for(; held.imported < held.read.size() && budget > 0; held.imported++)
  {
    const std::shared_ptr<tinygltf::Model>& model = held.read[held.imported];
    if(!model)
      continue;  // (missing: said when it was read)
    budget--;
    fv::timing::Scope time("planet: a place's model", 20);
    rs::ImportResult  r;
    try
    {
      r = rs::importGltf(scene, *model, {.prop = 0, .parent = backdropBuilt->root, .imageSearchPaths = {(placesDir / pathOf(of.pieces[held.imported].file)).parent_path()}});
    }
    catch(const std::exception& e)
    {
      notice("A place's " + of.pieces[held.imported].file + " could not be stood: " + problemOf(e));
      continue;
    }
    for(const rs::ClipId clip : r.clips)
      if(!clip.isNone())
        scene.release(clip);
    PlanetBuilt::HeldModel& made = held.models[held.imported];
    made.roots                   = r.roots;
    // every node that draws, where its file has it: what a site's copies are of
    for(const rs::EntityId e : r.entities)
    {
      const rs::Entity* entity = e.isNone() ? nullptr : scene.entity(e);
      if(!entity || !entity->mesh)
        continue;
      made.nodes.push_back({e, root * scene.world(e)});
      scene.setVisible(e, false);
    }
  }
  if(held.imported < held.read.size())
    return false;
  held.held = true;
  held.read.clear();  // (the scene has them now)
  return true;
}

void FvHost::Impl::standSite(rs::RetainedScene& scene, uint32_t site, bool growth)
{
  PlanetBuilt&                 p     = *planetBuilt;
  const W::Places&             pl    = *p.world->places;
  const W::Site&               s     = pl.sites[site];
  const W::Pack&               pack  = pl.catalogue->packs[s.pack];
  const PlanetBuilt::HeldPack& held  = p.packs.at(s.pack);
  PlanetBuilt::ShownSite&      shown = p.sites[site];
  // where it stands: its frame's turn under every copy, its middle the entities' place (a float is
  // millimetres fine on the far side of a planet; a place's own parts are exact to each other)
  const glm::dmat4 frame = W::siteFrame(*p.world, s);
  const glm::mat4  turn  = glm::mat4(glm::mat3(frame));
  shown.at               = glm::dvec3(frame[3]);
  shown.turn             = turn;
  for(size_t k = 0; k < pack.pieces.size() && k < held.models.size(); k++)
  {
    if(pack.pieces[k].growth != growth)
      continue;
    // (what of it stands at this site, and where: a site has its own share of what was left about)
    const std::vector<W::PieceCopy> stands = W::copiesAt(*p.world, site, uint32_t(k));
    if(stands.empty())
      continue;
    for(const PlanetBuilt::HeldModel::Node& node : held.models[k].nodes)
    {
      const rs::Entity* of = scene.entity(node.entity);
      if(!of || !of->mesh)
        continue;
      rs::MeshInstance             instance = *of->mesh;
      // (a model's own copies, a plant each, under each of its places in the site)
      const std::vector<glm::mat4> own      = instance.copies.empty() ? std::vector<glm::mat4>{glm::mat4(1.0F)} : instance.copies;
      std::vector<glm::mat4>       copies;
      copies.reserve(stands.size() * own.size());
      for(const W::PieceCopy& copy : stands)
      {
        const glm::mat4 place = turn * W::copyPlace(copy) * node.base;
        for(const glm::mat4& m : own)
          copies.push_back(place * m);
      }
      instance.copies      = std::move(copies);
      const rs::EntityId e = scene.createEntity(growth ? "Place growth" : "Place", {0, -1}, backdropBuilt->root);
      scene.setLocal(e, {.translation = glm::vec3(shown.at)});
      scene.setMeshInstance(e, std::move(instance));
      (growth ? shown.growth : shown.entities).push_back(e);
    }
  }
  (growth ? shown.grown : shown.stood) = true;
}

namespace {

// An entity's meshes as triangles in its own space: what a walker meets of it
std::shared_ptr<const fv::physics::SharedTriangles> trianglesOf(rs::RetainedScene& scene, rs::EntityId e)
{
  const rs::Entity* entity = scene.entity(e);
  if(!entity || !entity->mesh)
    return nullptr;
  std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>> parts;
  fv::physics::Box                                                  box;
  for(const rs::MeshPart& part : entity->mesh->parts)
  {
    const rs::MeshRecord* mesh = scene.mesh(part.mesh);
    if(!mesh || mesh->positions.empty() || mesh->indices.empty())
      continue;
    std::vector<float> xyz;
    xyz.reserve(mesh->positions.size() * 3);
    for(const glm::vec3& v : mesh->positions)
    {
      xyz.insert(xyz.end(), {v.x, v.y, v.z});
      box.add(v.x, v.y, v.z);
    }
    parts.emplace_back(std::move(xyz), std::vector<uint32_t>(mesh->indices.begin(), mesh->indices.begin() + mesh->indices.size() / 3 * 3));
  }
  if(parts.empty())
    return nullptr;
  return std::make_shared<const fv::physics::SharedTriangles>([parts = std::move(parts)] { return parts; }, box);
}

}  // namespace

void FvHost::Impl::placeSets(const glm::dvec3& near, double within, std::vector<PlacedSet>& out)
{
  if(!planetBuilt || !planetBuilt->world || !planetBuilt->world->places || !backdropBuilt || !renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  PlanetBuilt&       p     = *planetBuilt;
  const W::Places&   pl    = *p.world->places;
  const glm::dvec3   floor(0.0, double(groundHeight()), 0.0);
  for(const auto& [site, shown] : p.sites)
  {
    const W::Pack& pack = pl.catalogue->packs[pl.sites[site].pack];
    if(!shown.stood || glm::distance(near - floor, shown.at) > pack.radius + within)
      continue;
    const auto found = p.packs.find(pl.sites[site].pack);
    if(found == p.packs.end())
      continue;
    PlanetBuilt::HeldPack& held = found->second;
    held.walls.resize(held.models.size());
    // (as its entities stand: the root at the floor, the site's middle, its turn, each copy)
    const glm::mat4 toWorld = glm::translate(glm::mat4(1.0F), glm::vec3(floor + shown.at)) * shown.turn;
    for(size_t k = 0; k < pack.pieces.size() && k < held.models.size(); k++)
    {
      if(pack.pieces[k].growth)
        continue;
      const auto& nodes = held.models[k].nodes;
      auto&       walls = held.walls[k];
      if(walls.size() != nodes.size())
      {
        walls.clear();
        for(const PlanetBuilt::HeldModel::Node& node : nodes)
          walls.push_back(trianglesOf(scene, node.entity));
      }
      // (the copies that stand at this site, as standSite has them)
      const std::vector<W::PieceCopy> stands = W::copiesAt(*p.world, site, uint32_t(k));
      for(size_t n = 0; n < nodes.size(); n++)
        if(walls[n])
          for(const W::PieceCopy& copy : stands)
            out.push_back({walls[n], toWorld * W::copyPlace(copy) * nodes[n].base, false});
    }
  }
}

void FvHost::Impl::followPlaces(rs::RetainedScene& scene, const glm::dvec3& eye)
{
  PlanetBuilt& p = *planetBuilt;
  if(!p.world || !p.world->places || p.shown.empty())
    return;  // (none, or its ground has not come yet)
  const W::Places& pl = *p.world->places;
  const auto       gone = [&](std::vector<rs::EntityId>& entities) {
    for(const rs::EntityId e : entities)
      if(scene.entity(e))
        scene.destroyEntity(e);
    entities.clear();
  };
  if(glm::distance(eye, p.seenFor) > kSeenAgain)
  {
    p.seen    = W::sitesSeen(*p.world, eye, p.seen);
    p.seenFor = eye;
    // those no longer seen go, and their packs are no longer theirs
    for(auto it = p.sites.begin(); it != p.sites.end();)
    {
      if(std::binary_search(p.seen.begin(), p.seen.end(), it->first))
      {
        ++it;
        continue;
      }
      gone(it->second.entities);
      gone(it->second.growth);
      if(const auto held = p.packs.find(pl.sites[it->first].pack); held != p.packs.end())
      {
        held->second.users -= std::min(held->second.users, 1U);
        held->second.used = ++p.packClock;
      }
      it = p.sites.erase(it);
    }
    for(const uint32_t site : p.seen)
      if(!p.sites.count(site))
      {
        p.sites[site];
        p.packs[pl.sites[site].pack].users++;
      }
    // packs nobody shows, but for the last few used
    std::vector<std::pair<uint64_t, uint32_t>> idle;
    for(const auto& [pack, held] : p.packs)
      if(held.users == 0 && !held.reading)
        idle.emplace_back(held.used, pack);
    std::sort(idle.begin(), idle.end());
    for(size_t k = 0; k + kPacksKept < idle.size(); k++)
    {
      for(const PlanetBuilt::HeldModel& model : p.packs[idle[k].second].models)
        for(const rs::EntityId e : model.roots)
          if(scene.entity(e))
            scene.destroyEntity(e);
      p.packs.erase(idle[k].second);
    }
  }
  // every frame: the models of the packs that are wanted, a few; and the sites whose pack is held
  size_t budget = kPackModelsAFrame;
  for(auto& [site, shown] : p.sites)
  {
    if(!holdPack(scene, pl.sites[site].pack, budget))
      continue;
    if(!shown.stood)
      standSite(scene, site, false);
    // what grows over it, from near
    const double away = glm::distance(eye, shown.at);
    if(!shown.grown && away < kGrowthSeen)
      standSite(scene, site, true);
    else if(shown.grown && away > kGrowthSeen * W::kSplitHold)
    {
      gone(shown.growth);
      shown.grown = false;
    }
  }
}
