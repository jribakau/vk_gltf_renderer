//
// FvHost: a planet (fv_world.hpp; the backdrop's `planet` kind), drawn.
//
// What stands in the scene is the backdrop's root, as any backdrop's, and under it an entity a quad
// of the planet's ground (and one of its sea, where the ground dips under it), each with a mesh of its
// own about its place. No kit: the ground is its vertices' colours, the sea a body of clear water.
//
// The quads that draw are a cut of the planet's quadtrees: they cover it once. The camera says which
// cut is wanted (fv::world::select); what is shown follows it a step at a time, and at every step is
// a whole cut still, so there is never a hole to see the sky through:
//
// - a shown quad the selection has split gives way to its four children once all four are made;
// - shown quads under one the selection has whole give way to it once it is made.
//
// A quad's mesh is a few milliseconds of noise, so they are made on the pool, a batch at a time, and
// come back as hidden entities (their BLAS built by the time they show). The first cut is made whole
// before any of it shows: the state is `loading` till then, as while a kit is read.
//
// The planet itself is made on a worker first: its base map (the rain, the wear, rivers and lakes) is
// a second or two of work for 20 km of radius, more for a large one. Asked for another meanwhile (a
// slider dragged), the one being made is told to stop.
//
// Its plants are the backdrop's kit's (fv_world_plants.hpp), when the app has filled one: a planet
// stands without, bare, and is planted when the kit has been read (plantPlanet). What has a place (a
// tree, a bush, a rock) is made with its quad's ground and shown and let go with it, a variant an
// entity whose copies are its places; the kit's own entities are held hidden, as what the copies
// are of. The grass and its drifts are tiles round the camera and the figures (tilePlanet), the
// meadow's own way. Nothing sways in the wind yet.
//
// The sea floor's plants need no kit (fv_world_sea.hpp): their meshes are made with the planet and
// held hidden the same way, and a quad's arrive with it as the kit's do.
//
// The ground's grain is the kit's too (mapPlanetGround): a material a layer (grass, a wood's floor,
// rock, sand), made of the kit's ground maps with the colour map divided by its own mean, so the
// colour stays the ground's vertices'. The quads are still drawn with the one ground material; the
// renderer is told which materials the layers are, a quad's vertices carry their shares of each, and
// the path tracer draws a hit with one of them (shaders/world_ground.h.slang).
//
// Its lakes and rivers are meshes (fv::world::waterMesh), but the water is the sea's: the renderer is
// told which material they are drawn with and meets a hit of it as the water's surface
// (shaders/world_sea.h.slang), and, as the camera moves, what inland water the camera is in
// (fv::world::inlandWaterOver), for its paths to start there.
//

#include "fv_host_impl.hpp"
#include "fv_world_map.hpp"
#include "scene_import_gltf.hpp"

#include <chrono>

#include <figure/jobs/pool.hpp>

namespace {

namespace W = fv::world;

// How many quads are made at a time: each is a BLAS to build in the frame it arrives in. Fewer of a
// planted planet's: a quad's plants are hundreds of instances to bring into the scene in that frame.
constexpr size_t kPlanetBatch        = 32;
constexpr size_t kPlanetBatchPlanted = 8;

W::Quad parentOf(const W::Quad& q)
{
  return {q.face, uint8_t(q.level - 1), q.x / 2, q.y / 2};
}

bool isUnder(const W::Quad& q, const W::Quad& over)
{
  if(q.face != over.face || q.level <= over.level)
    return false;
  const int down = q.level - over.level;
  return (q.x >> down) == over.x && (q.y >> down) == over.y;
}

bool has(const std::vector<W::Quad>& sorted, const W::Quad& q)
{
  return std::binary_search(sorted.begin(), sorted.end(), q);
}

}  // namespace

// A quad's meshes, made: any thread
struct FvHost::Impl::PlanetMade
{
  W::Quad                     quad;
  W::ChunkMesh                ground;
  std::optional<W::ChunkMesh> sea;
  std::vector<W::Stand>       stands;     // what stands on it, about the ground's centre: the kit's plants,
  std::vector<W::Stand>       seaStands;  // and the sea floor's
};

glm::dvec3 FvHost::Impl::planetCentre() const
{
  if(!planetBuilt || !planetBuilt->world)
    return glm::dvec3(0.0);
  // the planet's own frame is the root's, which stands at the floor's height
  return planetBuilt->world->frame.centre + glm::dvec3(0.0, double(groundHeight()), 0.0);
}

void FvHost::Impl::standPlanet(rs::RetainedScene& scene)
{
  const fv::world::Spec asked = fv::backdrop::planetSpec(backdropSpec);
  const bool            alive = planetBuilt && backdropBuilt && scene.entity(backdropBuilt->root);
  // level another distance round its site and nothing else: the same planet, its quads made again
  if(alive && planetBuilt->world && planetBuilt->asked != asked)
    if(fv::world::Spec same = planetBuilt->asked; (same.clearing = asked.clearing, same == asked))
    {
      planetBuilt->world = std::make_shared<const fv::world::World>(fv::world::withClearing(*planetBuilt->world, asked.clearing));
      planetBuilt->asked = asked;
      restreamPlanet(scene);
    }
  if(alive && planetBuilt->asked == asked)
  {
    placeBackdrop();
    // (planted thicker or thinner: its plants again, on the planet as it is; a bare one's sea floor)
    if(planetBuilt->density != backdropSpec.density && planetBuilt->world)
    {
      planetBuilt->density = backdropSpec.density;
      if(planetBuilt->planted)
        plantPlanet(scene);
      else
        replantPlanet(scene);
    }
    planetBuilt->tiledFor = glm::dvec3(1.0e30);  // (its grass's reach may be another: laid again)
    return;  // as it stands, or as it is being made
  }
  if(backdropBuilt && scene.entity(backdropBuilt->root))
    scene.destroyEntity(backdropBuilt->root);
  backdropBuilt       = std::make_unique<BackdropBuilt>();
  backdropBuilt->root = scene.createEntity("Backdrop", {0, -1});
  backdropPlants      = 0;
  dropPlanet();
  planetBuilt    = std::make_unique<PlanetBuilt>();
  PlanetBuilt& p = *planetBuilt;
  p.asked        = asked;
  p.generation   = ++planetGeneration;

  // the ground: its vertices' colours on a rough surface
  rs::MaterialRecord ground;
  ground.owner             = {0, -1};
  ground.payload.name      = "Planet ground";
  ground.payload.metallic  = 0.0F;
  ground.payload.roughness = 1.0F;
  p.ground                 = scene.addMaterial(std::move(ground));
  // the lakes and rivers: clear water, which takes its colour from how far light goes in it (the
  // sea is the renderer's own: shaders/world_sea.h.slang)
  rs::MaterialRecord sea;
  sea.owner                = {0, -1};
  sea.payload.name         = "Planet sea";
  sea.payload.metallic     = 0.0F;
  sea.payload.roughness    = 0.02F;
  sea.payload.ior          = 1.33F;
  sea.payload.transmission = fv::mat::Transmission{.factor = 1.0F};
  sea.payload.volume       = fv::mat::Volume{.thickness           = 1.0F,
                                             .attenuationDistance = fv::world::kSeaColourDistance,
                                             .attenuationColor    = {fv::world::kSeaColour.x, fv::world::kSeaColour.y, fv::world::kSeaColour.z}};
  p.sea                    = scene.addMaterial(std::move(sea));
  // (the path tracer meets it as the water's surface; the ground's maps come with a kit: mapPlanetGround)
  renderer->setWorldGround({.ground = p.ground, .water = p.sea});
  p.density                = backdropSpec.density;

  // the sea floor's plants: made here (the kit has none), a mesh a variant, held hidden under the
  // root. Their colours are their vertices'; leaves are seen from both sides.
  rs::MaterialRecord plant;
  plant.owner             = {0, -1};
  plant.payload.name      = "Planet sea plant";
  plant.payload.metallic  = 0.0F;
  plant.payload.roughness = 0.85F;
  rs::MaterialRecord leaf = plant;
  leaf.payload.name        = "Planet sea leaf";
  leaf.payload.doubleSided = true;
  p.seaPlant               = scene.addMaterial(std::move(plant));
  p.seaLeaf                = scene.addMaterial(std::move(leaf));
  p.seaMeshes              = std::make_shared<const std::vector<W::SeaMesh>>(W::seaMeshes(asked.seed));
  for(const W::SeaMesh& m : *p.seaMeshes)
  {
    rs::MeshRecord record;
    record.name          = "Planet sea plant";
    record.owner         = {0, -1};
    record.positions     = m.positions;
    record.normals       = m.normals;
    record.colors        = m.colors;
    record.indices.own() = m.indices;
    record.submeshes.push_back({.firstIndex = 0, .indexCount = uint32_t(m.indices.size()), .material = 0, .order = 0});
    const rs::MeshId   mesh = scene.addMesh(std::move(record));
    const rs::EntityId e    = scene.createEntity("Planet sea plant", {0, -1}, backdropBuilt->root);
    rs::MeshInstance   instance;
    instance.parts.push_back({.mesh = mesh, .materials = {m.leaves ? p.seaLeaf : p.seaPlant}});
    scene.setMeshInstance(e, std::move(instance));
    scene.release(mesh);  // the entity's now
    scene.setVisible(e, false);
    p.seaPlants[m.role].push_back(e);
  }

  backdropState = "loading";
  placeBackdrop();
  planetStop = std::make_shared<std::atomic<bool>>(false);
  loaders.spawn([this, asked, stop = planetStop, generation = p.generation] {
    const auto started = std::chrono::steady_clock::now();
    auto       world   = std::make_shared<const fv::world::World>(fv::world::make(asked, {.stop = stop.get()}));
    const double ms    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    post([this, world, stop, generation, ms] {
      if(stop->load() || !planetBuilt || planetBuilt->generation != generation)
        return;  // of a planet that has gone since
      LOGI("planet: made in %.0f ms (%u map cells a face, %u lakes, %zu river pieces)\n", ms,
           world->base ? world->base->grid.cells() : 0U, world->base ? world->base->lakes : 0U,
           world->base ? world->base->pieces.size() : size_t(0));
      planetBuilt->world = world;
      if(backdropKit && renderer->feedsHostScene())
        plantPlanet(renderer->hostScene());
      followPlanet();  // asks for the first cut
    });
  });
}

void FvHost::Impl::dropPlanet()
{
  if(planetStop)
    planetStop->store(true);  // one still being made is nobody's now
  planetStop.reset();
  if(!planetBuilt)
    return;
  renderer->setWorldGround({});  // (its layers' materials went with the backdrop's root)
  renderer->setWorldWaterAtEye(HostRenderer::kNoWorldWater);
  if(renderer->feedsHostScene())
  {
    // (its entities went with the backdrop's root)
    rs::RetainedScene& scene = renderer->hostScene();
    for(const rs::MaterialId material : {planetBuilt->ground, planetBuilt->sea, planetBuilt->seaPlant, planetBuilt->seaLeaf})
      if(scene.material(material))
        scene.release(material);
  }
  planetBuilt.reset();
  // the view's up was the planet's (followPlanet): the scene's again
  if(const auto camera = renderer->getCameraManipulator(); camera && viewCamera.empty())
  {
    glm::dvec3 eye, centre, up;
    camera->getLookat(eye, centre, up);
    if(up != glm::dvec3(0.0, 1.0, 0.0))
      camera->setLookat(eye, centre, glm::dvec3(0.0, 1.0, 0.0), true);
  }
}

void FvHost::Impl::followPlanet()
{
  if(!planetBuilt || !backdropBuilt || !renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  if(!scene.entity(backdropBuilt->root))
    return;  // it went with a scene replaced: applyBackdrop makes it again
  PlanetBuilt& p = *planetBuilt;
  if(!p.world)
    return;  // being made
  const fv::world::World& world = *p.world;

  // the eye, in the root's frame; with no camera, a figure's
  glm::dvec3       eye(0.0, 1.7, 0.0);
  const glm::dvec3 floor(0.0, double(groundHeight()), 0.0);
  if(const auto camera = renderer->getCameraManipulator())
  {
    eye = glm::dvec3(camera->getEye()) - floor;
    // The free view's up is the planet's away from the figures (fv::world::viewUp), so the horizon
    // stays level on its far side; a scene camera keeps its own, and so does play mode's (a ship
    // banks). And its far plane clears the planet.
    if(viewCamera.empty() && !playing())
    {
      glm::dvec3 at, centre, up;
      camera->getLookat(at, centre, up);
      const glm::dvec3 wanted = fv::world::viewUp(world, eye);
      if(glm::distance(up, wanted) > 1e-7)
        camera->setLookat(at, centre, wanted, true);
    }
    const glm::dvec2 clip   = camera->getClipPlanes();
    const double     needed = fv::world::farPlane(world, eye);
    if(clip.y < needed)
      camera->setClipPlanes({clip.x, 2.0 * needed});
  }
  // the lake or the river the camera is in, if any: its paths start in that water (once it has
  // gone a hand's breadth: a look at the ground under it)
  if(glm::distance(eye, p.waterAtEyeFor) > 0.1)
  {
    p.waterAtEyeFor    = eye;
    const double level = fv::world::inlandWaterOver(world, eye);
    renderer->setWorldWaterAtEye(level > 0.0 ? float(level) : HostRenderer::kNoWorldWater);
  }
  // selected again once it has gone a twentieth of its height over the ground: a quad's reach is
  // twice its width, and the nearest is no nearer than that height
  if(!p.selected || glm::distance(eye, p.selectedFor) > std::max(0.5, 0.05 * p.eyeHeight))
  {
    p.selection   = fv::world::select(world, eye, p.selected ? &p.selection : nullptr);
    p.selectedFor = eye;
    p.eyeHeight   = std::abs(fv::world::heightOver(world, eye));
    p.selected    = true;
    if(!p.shown.empty())
      resolvePlanet(scene);  // what has arrived may no longer be wanted
  }
  // (the grass's tiles, once the camera has gone a few metres: a tile is two across at the least and
  // stays a little past where it would be laid, and every change is a pass of the renderer over the
  // scene's instances)
  if(p.planted && !p.shown.empty() && glm::distance(eye, p.tiledFor) >= 3.0)
    tilePlanet(scene, eye);
  if(p.busy)
    return;  // a batch is being made

  // what to make next: the whole first cut; then the children of shown quads the selection has split,
  // and the quads it has whole over shown ones
  std::vector<W::Quad> ask;
  if(p.shown.empty())
  {
    for(const W::Quad& q : p.selection.leaves)
      if(!p.arrived.count(q))
        ask.push_back(q);
  }
  else
  {
    for(const auto& [q, chunk] : p.shown)
    {
      if(has(p.selection.split, q))
      {
        for(const W::Quad& child : W::children(q))
          if(!p.arrived.count(child))
            ask.push_back(child);
      }
      else if(!has(p.selection.leaves, q))
      {
        W::Quad over = q;
        while(over.level > 0 && !has(p.selection.leaves, over))
          over = parentOf(over);
        if(has(p.selection.leaves, over) && !p.arrived.count(over))
          ask.push_back(over);
      }
    }
    std::sort(ask.begin(), ask.end());
    ask.erase(std::unique(ask.begin(), ask.end()), ask.end());
    const size_t batch = p.planted ? kPlanetBatchPlanted : kPlanetBatch;
    if(ask.size() > batch)
    {
      // the nearest first
      const auto away = [&](const W::Quad& q) {
        const double span = 2.0 / double(uint64_t(1) << q.level);
        return glm::distance(eye, W::toScene(world, W::direction(q.face, (double(q.x) + 0.5) * span - 1.0, (double(q.y) + 0.5) * span - 1.0), 0.0));
      };
      std::vector<std::pair<double, W::Quad>> byDistance;
      for(const W::Quad& q : ask)
        byDistance.emplace_back(away(q), q);
      std::sort(byDistance.begin(), byDistance.end());
      ask.clear();
      for(size_t i = 0; i < batch; i++)
        ask.push_back(byDistance[i].second);
    }
  }
  if(ask.empty())
    return;
  p.busy = true;
  loaders.spawn([this, held = p.world, ask = std::move(ask), generation = p.generation, planted = p.planted, variants = p.variants,
                 density = p.density, seaMeshes = p.seaMeshes] {
    auto made = std::make_shared<std::vector<PlanetMade>>(ask.size());
    fv::jobs::parallelFor(ask.size(), 1, [&](size_t begin, size_t end) {
      for(size_t i = begin; i < end; i++)
      {
        PlanetMade& m = (*made)[i];
        m.quad        = ask[i];
        m.ground      = W::groundMesh(*held, ask[i]);
        // a river's or a lake's water, cut at the ground's own points; the sea is no mesh
        // (shaders/world_sea.h.slang)
        if(m.ground.inland)
        {
          m.sea = W::waterMesh(*held, ask[i]);
          if(m.sea->indices.empty())
            m.sea.reset();
        }
        if(planted && m.ground.highest > 0.0)
          m.stands = W::standing(*held, ask[i], m.ground.centre, variants, density);
        if(seaMeshes && m.ground.lowest < 0.0)
          m.seaStands = W::seaStanding(*held, ask[i], m.ground.centre, *seaMeshes, density);
      }
    });
    post([this, made, generation] { planetArrived(*made, generation); });
  });
}

void FvHost::Impl::planetArrived(std::vector<PlanetMade>& made, uint64_t generation)
{
  if(!planetBuilt || planetBuilt->generation != generation)
    return;  // of a planet that has gone since
  PlanetBuilt& p = *planetBuilt;
  p.busy         = false;
  if(!backdropBuilt || !renderer->feedsHostScene() || !renderer->hostScene().entity(backdropBuilt->root))
    return;
  fv::timing::Scope time("planet: quads arrived", 4);
  rs::RetainedScene& scene  = renderer->hostScene();
  const auto         entity = [&](W::ChunkMesh& m, rs::MaterialId material, const char* name) {
    rs::MeshRecord record;
    record.name          = name;
    record.owner         = {0, -1};
    const uint32_t count = uint32_t(m.indices.size());
    record.positions     = std::move(m.positions);
    record.normals       = std::move(m.normals);
    record.colors        = std::move(m.colors);
    record.masks         = std::move(m.layers);  // the ground's: its vertices' shares of its layers
    record.indices.own() = std::move(m.indices);
    record.submeshes.push_back({.firstIndex = 0, .indexCount = count, .material = 0, .order = 0});
    const rs::MeshId   mesh = scene.addMesh(std::move(record));
    const rs::EntityId e    = scene.createEntity(name, {0, -1}, backdropBuilt->root);
    scene.setLocal(e, {.translation = glm::vec3(m.centre)});
    rs::MeshInstance instance;
    instance.parts.push_back({.mesh = mesh, .materials = {material}});
    scene.setMeshInstance(e, std::move(instance));
    scene.release(mesh);  // the entity's now
    scene.setVisible(e, false);
    return e;
  };
  for(PlanetMade& m : made)
  {
    if(p.arrived.count(m.quad) || p.shown.count(m.quad))
      continue;
    PlanetBuilt::Chunk chunk;
    const glm::vec3 centre(m.ground.centre);
    chunk.ground = entity(m.ground, p.ground, "Planet ground");
    if(m.sea)
      chunk.sea = entity(*m.sea, p.sea, "Planet sea");
    // what stands on it: each variant's places as the copies of an entity that draws the variant's
    // mesh (`of`: the entity that holds it, hidden; `base`: its own turn and scale under each place)
    const auto stand = [&](rs::EntityId of, const glm::mat4& base, std::vector<glm::mat4>& places) {
      const rs::Entity* held = scene.entity(of);
      if(!held || !held->mesh || places.empty())
        return;
      rs::MeshInstance instance = *held->mesh;
      for(glm::mat4& place : places)
        place = place * base;
      chunk.standing += places.size();
      instance.copies      = std::move(places);
      const rs::EntityId e = scene.createEntity("Planet plants", {0, -1}, backdropBuilt->root);
      scene.setLocal(e, {.translation = centre});
      scene.setMeshInstance(e, std::move(instance));
      scene.setVisible(e, false);
      chunk.plants.push_back(e);
    };
    // the kit's: none where something of the scene is over the ground (fv::backdrop::Standing, on
    // the plan of the floor: only what stands near the floor's height is under it)
    for(W::Stand& s : m.stands)
      if(const auto found = backdropBuilt->variants.find(std::string(W::kStandingRoles[s.role]));
         found != backdropBuilt->variants.end() && s.variant < found->second.size())
      {
        if(!backdropStanding.empty())
          std::erase_if(s.places, [&](const glm::mat4& place) {
            const glm::vec3 at = centre + glm::vec3(place[3]);
            return std::abs(at.y) < 2.0F * fv::backdrop::kOverhead
                   && backdropStanding.over(at.x, at.z, fv::backdrop::kCrownShare * found->second[s.variant].halfWidth * glm::length(glm::vec3(place[0])));
          });
        stand(found->second[s.variant].entity, found->second[s.variant].base, s.places);
      }
    // the sea floor's
    for(W::Stand& s : m.seaStands)
      if(s.role < p.seaPlants.size() && s.variant < p.seaPlants[s.role].size())
        stand(p.seaPlants[s.role][s.variant], glm::mat4(1.0F), s.places);
    p.arrived[m.quad] = chunk;
  }
  resolvePlanet(scene);
}

void FvHost::Impl::resolvePlanet(rs::RetainedScene& scene)
{
  PlanetBuilt& p    = *planetBuilt;
  const auto   show = [&](const W::Quad& q) {
    const PlanetBuilt::Chunk chunk = p.arrived.at(q);
    p.arrived.erase(q);
    scene.setVisible(chunk.ground, true);
    if(!chunk.sea.isNone())
      scene.setVisible(chunk.sea, true);
    for(const rs::EntityId e : chunk.plants)
      scene.setVisible(e, true);
    p.standing += chunk.standing;
    p.shown[q] = chunk;
  };
  const auto gone = [&](const PlanetBuilt::Chunk& chunk) {
    if(scene.entity(chunk.ground))
      scene.destroyEntity(chunk.ground);
    if(!chunk.sea.isNone() && scene.entity(chunk.sea))
      scene.destroyEntity(chunk.sea);
    for(const rs::EntityId e : chunk.plants)
      if(scene.entity(e))
        scene.destroyEntity(e);
  };
  // (a shown one's plants are no longer counted)
  const auto hide = [&](const W::Quad& q) {
    const PlanetBuilt::Chunk chunk = p.shown.at(q);
    p.standing -= std::min(p.standing, chunk.standing);
    gone(chunk);
    p.shown.erase(q);
  };

  if(p.shown.empty())
  {
    // the first cut, whole
    for(const W::Quad& q : p.selection.leaves)
      if(!p.arrived.count(q))
        return;  // (another selection since it was asked for: the rest is on its way)
    for(const W::Quad& q : p.selection.leaves)
      show(q);
    for(const auto& [q, chunk] : p.arrived)
      gone(chunk);
    p.arrived.clear();
    if(!p.stale.empty())
    {
      // made again for what stands in the scene (restreamPlanet): what drew until now goes, and the
      // grass is laid again, a tile under a floor left out
      for(const PlanetBuilt::Chunk& chunk : p.stale)
        gone(chunk);
      p.stale.clear();
      if(p.planted)
      {
        for(const auto& [tile, entity] : p.tiles)
          if(!entity.isNone())
          {
            scene.setVisible(entity, false);
            p.spareTiles[tile.kind].push_back(entity);
          }
        p.tiles.clear();
        p.tiled = 0;
        tilePlanet(scene, p.selectedFor);
      }
    }
    backdropState = "ready";
    scopeHaze();
    publishBackdrop();
    applyLighting(false);  // the ground's plane gives way to it, and the sky's planet is this one
    return;
  }

  // a shown quad the selection has split gives way to its four children, once all have arrived
  std::vector<W::Quad> shown;
  for(const auto& [q, chunk] : p.shown)
    shown.push_back(q);
  for(const W::Quad& q : shown)
  {
    if(!has(p.selection.split, q))
      continue;
    const auto kids = W::children(q);
    if(!std::all_of(kids.begin(), kids.end(), [&](const W::Quad& k) { return p.arrived.count(k) > 0; }))
      continue;
    for(const W::Quad& k : kids)
      show(k);
    hide(q);
  }
  // shown quads under one the selection has whole give way to it
  std::vector<W::Quad> arrived;
  for(const auto& [q, chunk] : p.arrived)
    arrived.push_back(q);
  for(const W::Quad& over : arrived)
  {
    if(!has(p.selection.leaves, over))
      continue;
    std::vector<W::Quad> under;
    for(const auto& [q, chunk] : p.shown)
      if(isUnder(q, over))
        under.push_back(q);
    if(under.empty())
      continue;
    for(const W::Quad& q : under)
      hide(q);
    show(over);
  }
  // what has arrived and is no longer waited for: a child of a quad no longer shown or no longer
  // split, a whole quad the selection no longer has
  for(auto it = p.arrived.begin(); it != p.arrived.end();)
  {
    const W::Quad& q      = it->first;
    const bool     child  = q.level > 0 && p.shown.count(parentOf(q)) && has(p.selection.split, parentOf(q));
    const bool     whole  = has(p.selection.leaves, q) && std::any_of(p.shown.begin(), p.shown.end(), [&](const auto& s) { return isUnder(s.first, q); });
    if(child || whole)
    {
      ++it;
      continue;
    }
    gone(it->second);
    it = p.arrived.erase(it);
  }
}

// ---- its plants ----------------------------------------------------------------------------------------

void FvHost::Impl::readPlanetKit(const std::filesystem::path& dir)
{
  const uint64_t request = ++backdropRequest;
  loaders.spawn([this, dir, request] {
    std::shared_ptr<const BackdropKit> kit;
    std::string                        error;
    try
    {
      kit = readBackdropKit(dir);
    }
    catch(const std::exception& e)
    {
      error = e.what();
    }
    post([this, kit, error, request] {
      LOGI("planet: its kit %s%s\n", kit ? "read" : "not read: ", kit ? (request != backdropRequest ? " (for a folder since changed)" : "") : error.c_str());
      if(request != backdropRequest)
        return;  // the folder is another since, or was filled: read again
      if(!kit)
      {
        // a folder not filled is a planet left bare; a kit that is there and cannot be read is a fault
        if(error.rfind("no kit.json", 0) != 0)
          notice("The planet's plants could not be read: " + error, "error");
        return;
      }
      backdropKit = kit;
      if(planetBuilt && planetBuilt->world && backdropBuilt && renderer->feedsHostScene() && renderer->hostScene().entity(backdropBuilt->root))
        plantPlanet(renderer->hostScene());
    });
  });
}

void FvHost::Impl::plantPlanet(rs::RetainedScene& scene)
{
  if(!planetBuilt || !planetBuilt->world || !backdropKit || !backdropBuilt || !scene.entity(backdropBuilt->root))
    return;
  fv::timing::Scope time("planet: planting", 20);
  PlanetBuilt&      p = *planetBuilt;
  // the tiles laid, and their meshes: made again below
  for(const auto& [tile, entity] : p.tiles)
    if(!entity.isNone() && scene.entity(entity))
      scene.destroyEntity(entity);
  for(const auto& [kind, spare] : p.spareTiles)
    for(const rs::EntityId entity : spare)
      if(scene.entity(entity))
        scene.destroyEntity(entity);
  for(const BackdropBuilt::Patch& patch : p.patches)
    if(scene.entity(patch.holder))
      scene.destroyEntity(patch.holder);
  p.tiles.clear();
  p.spareTiles.clear();
  p.patches.clear();
  p.grassWas = {};
  p.tiledFor = glm::dvec3(1.0e30);
  p.tiled    = 0;

  mapPlanetGround(scene);
  // the kit's plants, held under the root and hidden: what a quad's copies and a tile's mesh are of
  holdBackdropPlants(scene);
  for(const auto& [role, variants] : backdropBuilt->variants)
    for(const BackdropBuilt::Variant& v : variants)
      if(scene.entity(v.entity))
        scene.setVisible(v.entity, false);
  p.variants = {};
  for(size_t r = 0; r < W::kStandingRoles.size(); r++)
    if(const auto found = backdropBuilt->variants.find(std::string(W::kStandingRoles[r])); found != backdropBuilt->variants.end())
      p.variants[r] = uint32_t(found->second.size());
  p.density   = backdropSpec.density;
  // (the planet as one whose plants are drawn: the same, but for the ground under them)
  if(!p.world->planted)
  {
    fv::world::World with = *p.world;
    with.planted         = true;
    p.world              = std::make_shared<const fv::world::World>(std::move(with));
  }
  p.tileKinds = W::tileKinds(*p.world, p.density);
  for(const W::TileKind& kind : p.tileKinds)
    p.patches.push_back(makeBackdropPatch(scene, kind.cover));
  p.planted = true;
  replantPlanet(scene);
}

tinygltf::Model FvHost::Impl::planetGroundModel(const BackdropKit& kit, std::vector<size_t>& layerOf)
{
  tinygltf::Model  m;
  tinygltf::Buffer buffer;
  const auto       put = [&](const void* data, size_t bytes, int target, int componentType, size_t count, int type) {
    tinygltf::BufferView view;
    view.buffer     = 0;
    view.byteOffset = buffer.data.size();
    view.byteLength = bytes;
    view.target     = target;
    const auto* b   = static_cast<const unsigned char*>(data);
    buffer.data.insert(buffer.data.end(), b, b + bytes);
    m.bufferViews.push_back(view);
    tinygltf::Accessor a;
    a.bufferView    = int(m.bufferViews.size()) - 1;
    a.componentType = componentType;
    a.count         = count;
    a.type          = type;
    m.accessors.push_back(a);
    return int(m.accessors.size()) - 1;
  };
  // one small triangle, every layer's: nothing ever draws it
  const std::array<glm::vec3, 3> positions{glm::vec3(0.0F), glm::vec3(0.0F, 0.0F, 0.01F), glm::vec3(0.01F, 0.0F, 0.0F)};
  const std::array<glm::vec3, 3> normals{glm::vec3(0.0F, 1.0F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F)};
  const std::array<glm::vec2, 3> uvs{glm::vec2(0.0F), glm::vec2(0.0F, 1.0F), glm::vec2(1.0F, 0.0F)};
  const std::array<uint32_t, 3>  indices{0, 1, 2};
  const int position = put(positions.data(), sizeof(positions), TINYGLTF_TARGET_ARRAY_BUFFER, TINYGLTF_COMPONENT_TYPE_FLOAT, 3, TINYGLTF_TYPE_VEC3);
  const int normal   = put(normals.data(), sizeof(normals), TINYGLTF_TARGET_ARRAY_BUFFER, TINYGLTF_COMPONENT_TYPE_FLOAT, 3, TINYGLTF_TYPE_VEC3);
  const int uv       = put(uvs.data(), sizeof(uvs), TINYGLTF_TARGET_ARRAY_BUFFER, TINYGLTF_COMPONENT_TYPE_FLOAT, 3, TINYGLTF_TYPE_VEC2);
  const int index    = put(indices.data(), sizeof(indices), TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, 3, TINYGLTF_TYPE_SCALAR);
  m.accessors[size_t(position)].minValues = {0.0, 0.0, 0.0};
  m.accessors[size_t(position)].maxValues = {0.01, 0.0, 0.01};
  m.buffers.push_back(std::move(buffer));

  const auto texture = [&](const std::string& uri) {
    tinygltf::Image image;
    image.uri = uri;
    m.images.push_back(image);
    tinygltf::Texture tex;
    tex.source = int(m.images.size()) - 1;
    m.textures.push_back(tex);
    return int(m.textures.size()) - 1;
  };
  tinygltf::Mesh mesh;
  mesh.name = "Planet ground layers";
  layerOf.clear();
  for(size_t layer = 0; layer < W::kGroundLayers; layer++)
  {
    const auto found = kit.ground.find(std::string(W::kGroundKinds[layer]));
    if(found == kit.ground.end() || found->second.diffuse.empty() || std::min({found->second.mean.x, found->second.mean.y, found->second.mean.z}) <= 0.0F)
      continue;  // no maps for it, or a kit that does not say what its map's colour is
    const BackdropKit::Ground& g = found->second;
    tinygltf::Material         material;
    material.name = "Planet ground: " + std::string(W::kGroundKinds[layer]);
    // the ground's colour is its vertices': the map over its own mean gives it its grain
    material.pbrMetallicRoughness.baseColorFactor        = {1.0 / double(g.mean.x), 1.0 / double(g.mean.y), 1.0 / double(g.mean.z), 1.0};
    material.pbrMetallicRoughness.baseColorTexture.index = texture(g.diffuse);
    material.pbrMetallicRoughness.roughnessFactor        = 1.0;
    material.pbrMetallicRoughness.metallicFactor         = 0.0;
    if(!g.normal.empty())
      material.normalTexture.index = texture(g.normal);
    if(!g.arm.empty())
    {
      // occlusion, roughness and metal in one image, as glTF packs them
      material.pbrMetallicRoughness.metallicRoughnessTexture.index = texture(g.arm);
      material.pbrMetallicRoughness.metallicFactor                 = 1.0;
    }
    m.materials.push_back(material);
    tinygltf::Primitive prim;
    prim.mode                     = TINYGLTF_MODE_TRIANGLES;
    prim.attributes["POSITION"]   = position;
    prim.attributes["NORMAL"]     = normal;
    prim.attributes["TEXCOORD_0"] = uv;
    prim.indices                  = index;
    prim.material                 = int(m.materials.size()) - 1;
    mesh.primitives.push_back(prim);
    layerOf.push_back(layer);
  }
  m.meshes.push_back(mesh);
  tinygltf::Node node;
  node.name = "Planet ground layers";
  node.mesh = 0;
  m.nodes.push_back(node);
  tinygltf::Scene scene;
  scene.nodes = {0};
  m.scenes.push_back(scene);
  m.defaultScene = 0;
  return m;
}

void FvHost::Impl::mapPlanetGround(rs::RetainedScene& scene)
{
  PlanetBuilt& p = *planetBuilt;
  if(p.groundKit == backdropKit && scene.entity(p.groundLayers))
    return;  // made, of this kit
  if(scene.entity(p.groundLayers))
    scene.destroyEntity(p.groundLayers);
  p.groundLayers = {};
  p.groundKit    = backdropKit;
  HostRenderer::WorldGround ground{.ground = p.ground, .water = p.sea};
  std::vector<size_t>       layerOf;
  const tinygltf::Model     model = planetGroundModel(*backdropKit, layerOf);
  if(!layerOf.empty())
  {
    try
    {
      const rs::ImportResult r = rs::importGltf(scene, model, {.prop = 0, .parent = backdropBuilt->root, .imageSearchPaths = {backdropKit->dir}});
      if(!r.roots.empty())
      {
        p.groundLayers = r.roots.front();
        scene.setVisible(p.groundLayers, false);
        for(size_t i = 0; i < layerOf.size() && i < r.materials.size(); i++)
        {
          ground.layers[layerOf[i]] = r.materials[i];
          ground.tiles[layerOf[i]]  = backdropKit->ground.at(std::string(W::kGroundKinds[layerOf[i]])).tile;
        }
      }
    }
    catch(const std::exception& e)
    {
      notice("The planet's ground maps could not be read: " + problemOf(e));
    }
  }
  renderer->setWorldGround(ground);
}

void FvHost::Impl::replantPlanet(rs::RetainedScene& scene)
{
  PlanetBuilt& p = *planetBuilt;
  // the quads that stand have what was on them: made again, the first cut whole before any shows.
  // (A batch on its way was made as before: it is of a planet gone, as far as its arrival can tell.)
  const auto gone = [&](const PlanetBuilt::Chunk& chunk) {
    for(const rs::EntityId e : {chunk.ground, chunk.sea})
      if(!e.isNone() && scene.entity(e))
        scene.destroyEntity(e);
    for(const rs::EntityId e : chunk.plants)
      if(scene.entity(e))
        scene.destroyEntity(e);
  };
  for(const auto& [q, chunk] : p.shown)
    gone(chunk);
  for(const auto& [q, chunk] : p.arrived)
    gone(chunk);
  for(const PlanetBuilt::Chunk& chunk : p.stale)
    gone(chunk);
  p.shown.clear();
  p.arrived.clear();
  p.stale.clear();
  p.standing   = 0;
  p.selected   = false;
  p.busy       = false;
  p.generation = ++planetGeneration;
  if(backdropState == "ready")
  {
    backdropState = "loading";
    publishBackdrop();
  }
}

void FvHost::Impl::restreamPlanet(rs::RetainedScene& scene)
{
  PlanetBuilt& p = *planetBuilt;
  // what draws stays, as it is, until the first cut of the new is whole (resolvePlanet); what had
  // arrived and not shown is of the planet as it was. (A batch on its way too: another generation.)
  for(const auto& [q, chunk] : p.shown)
    p.stale.push_back(chunk);
  for(const auto& [q, chunk] : p.arrived)
  {
    for(const rs::EntityId e : {chunk.ground, chunk.sea})
      if(!e.isNone() && scene.entity(e))
        scene.destroyEntity(e);
    for(const rs::EntityId e : chunk.plants)
      if(scene.entity(e))
        scene.destroyEntity(e);
  }
  p.shown.clear();
  p.arrived.clear();
  p.standing   = 0;
  p.selected   = false;
  p.busy       = false;
  p.generation = ++planetGeneration;
}

void FvHost::Impl::clearPlanet(rs::RetainedScene& scene)
{
  PlanetBuilt& p = *planetBuilt;
  // level another distance: standPlanet makes its quads again (the planet itself, if it is still
  // being made); as level as it was: its plants' places again, where it has plants
  if(!p.world || p.asked != fv::backdrop::planetSpec(backdropSpec))
    standPlanet(scene);
  else if(p.planted)
    restreamPlanet(scene);
}

void FvHost::Impl::tilePlanet(rs::RetainedScene& scene, const glm::dvec3& eye)
{
  fv::timing::Scope       time("planet: tiles", 4);
  PlanetBuilt&            p     = *planetBuilt;
  const fv::world::World& world = *p.world;
  // thick round the figures and round the camera, as a meadow's (both in the root's frame)
  const std::array<glm::dvec3, 2>  foci{glm::dvec3(0.0), eye};
  const std::vector<W::GrassTile>  wanted = W::grassTiles(world, foci, p.tileKinds, p.grassWas, backdropSpec.size);
  p.tiledFor = eye;
  // a tile no longer asked for: hidden, and its entity the next tile's of that kind
  for(auto it = p.tiles.begin(); it != p.tiles.end();)
  {
    if(std::binary_search(wanted.begin(), wanted.end(), it->first))
    {
      ++it;
      continue;
    }
    if(!it->second.isNone())
    {
      scene.setVisible(it->second, false);
      p.spareTiles[it->first.kind].push_back(it->second);
      p.tiled -= std::min(p.tiled, p.patches[it->first.kind].plants);
    }
    it = p.tiles.erase(it);
  }
  for(const W::GrassTile& tile : wanted)
  {
    if(tile.kind >= p.patches.size() || p.tiles.count(tile))
      continue;
    const BackdropBuilt::Patch& patch = p.patches[tile.kind];
    if(!scene.entity(patch.holder))
      continue;  // the kit has none of its role
    const W::GrassPlace place = W::grassPlace(world, tile, p.tileKinds[tile.kind]);
    if(place.level < W::kGrassLevel)
    {
      p.tiles[tile] = rs::EntityId{};  // too steep for grass: remembered, so it is not asked again
      continue;
    }
    // nor where a floor of the scene's covers the ground (fv::backdrop::Standing): the middle of the
    // tile, as its square lies by the planet's face and not by the floor's axes
    if(const float half = 0.35F * p.tileKinds[tile.kind].cover.tile;
       !backdropStanding.empty() && std::abs(place.position.y) < 2.0 * double(fv::backdrop::kOverhead)
       && backdropStanding.covered(float(place.position.x) - half, float(place.position.z) - half, float(place.position.x) + half,
                                   float(place.position.z) + half)
              >= fv::backdrop::kTileTaken)
    {
      p.tiles[tile] = rs::EntityId{};  // (until what stands is another: every tile is laid again then)
      continue;
    }
    rs::EntityId               entity;
    std::vector<rs::EntityId>& spare = p.spareTiles[tile.kind];
    if(!spare.empty())
    {
      entity = spare.back();
      spare.pop_back();
    }
    else
    {
      entity = scene.createEntity("Planet tile", {0, -1}, backdropBuilt->root);
      scene.setMeshInstance(entity, rs::MeshInstance(patch.instance));
    }
    scene.setLocal(entity, {.translation = glm::vec3(place.position), .rotation = place.rotation, .scale = place.scale});
    scene.setVisible(entity, true);
    p.tiles[tile] = entity;
    p.tiled += patch.plants;
  }
  // the topic's count of plants, when it is off by more than a twentieth
  backdropPlants      = p.standing + p.tiled;
  const uint64_t said = backdropPlantsSaid;
  if(backdropPlants > said + said / 20 || backdropPlants + said / 20 < said)
    publishBackdrop();
}
