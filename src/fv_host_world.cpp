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

#include "fv_host_impl.hpp"

#include <figure/jobs/pool.hpp>

namespace {

namespace W = fv::world;

// How many quads are made at a time: each is a BLAS to build in the frame it arrives in
constexpr size_t kPlanetBatch = 32;

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
};

glm::dvec3 FvHost::Impl::planetCentre() const
{
  if(!planetBuilt)
    return glm::dvec3(0.0);
  // the planet's own frame is the root's, which stands at the floor's height
  return planetBuilt->world->frame.centre + glm::dvec3(0.0, double(groundHeight()), 0.0);
}

void FvHost::Impl::standPlanet(rs::RetainedScene& scene)
{
  const fv::world::Spec asked = fv::backdrop::planetSpec(backdropSpec);
  const bool            alive = planetBuilt && backdropBuilt && scene.entity(backdropBuilt->root);
  if(alive && planetBuilt->asked == asked)
  {
    placeBackdrop();
    return;  // as it stands, or as it is being made
  }
  fv::timing::Scope time("planet: made", 20);
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
  p.world        = std::make_shared<const fv::world::World>(fv::world::make(asked));

  // the ground: its vertices' colours on a rough surface
  rs::MaterialRecord ground;
  ground.owner             = {0, -1};
  ground.payload.name      = "Planet ground";
  ground.payload.metallic  = 0.0F;
  ground.payload.roughness = 1.0F;
  p.ground                 = scene.addMaterial(std::move(ground));
  // the sea: clear water, which takes its colour from how far light goes in it
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

  backdropState = "loading";
  placeBackdrop();
  followPlanet();  // asks for the first cut
}

void FvHost::Impl::dropPlanet()
{
  if(!planetBuilt)
    return;
  if(renderer->feedsHostScene())
  {
    // (its entities went with the backdrop's root)
    rs::RetainedScene& scene = renderer->hostScene();
    for(const rs::MaterialId material : {planetBuilt->ground, planetBuilt->sea})
      if(scene.material(material))
        scene.release(material);
  }
  planetBuilt.reset();
}

void FvHost::Impl::followPlanet()
{
  if(!planetBuilt || !backdropBuilt || !renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  if(!scene.entity(backdropBuilt->root))
    return;  // it went with a scene replaced: applyBackdrop makes it again
  PlanetBuilt&             p     = *planetBuilt;
  const fv::world::World& world = *p.world;

  // the eye, in the root's frame; with no camera, a figure's
  glm::dvec3 eye(0.0, 1.7, 0.0);
  if(const auto camera = renderer->getCameraManipulator())
    eye = glm::dvec3(camera->getEye()) - glm::dvec3(0.0, double(groundHeight()), 0.0);
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
    if(ask.size() > kPlanetBatch)
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
      for(size_t i = 0; i < kPlanetBatch; i++)
        ask.push_back(byDistance[i].second);
    }
  }
  if(ask.empty())
    return;
  p.busy = true;
  loaders.spawn([this, held = p.world, ask = std::move(ask), generation = p.generation] {
    auto made = std::make_shared<std::vector<PlanetMade>>(ask.size());
    fv::jobs::parallelFor(ask.size(), 1, [&](size_t begin, size_t end) {
      for(size_t i = begin; i < end; i++)
      {
        PlanetMade& m = (*made)[i];
        m.quad        = ask[i];
        m.ground      = W::groundMesh(*held, ask[i]);
        if(m.ground.lowest < 0.0)
          m.sea = W::seaMesh(*held, ask[i]);
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
  rs::RetainedScene& scene  = renderer->hostScene();
  const auto         entity = [&](W::ChunkMesh& m, rs::MaterialId material, const char* name) {
    rs::MeshRecord record;
    record.name          = name;
    record.owner         = {0, -1};
    const uint32_t count = uint32_t(m.indices.size());
    record.positions     = std::move(m.positions);
    record.normals       = std::move(m.normals);
    record.colors        = std::move(m.colors);
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
    chunk.ground = entity(m.ground, p.ground, "Planet ground");
    if(m.sea)
      chunk.sea = entity(*m.sea, p.sea, "Planet sea");
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
    p.shown[q] = chunk;
  };
  const auto gone = [&](const PlanetBuilt::Chunk& chunk) {
    if(scene.entity(chunk.ground))
      scene.destroyEntity(chunk.ground);
    if(!chunk.sea.isNone() && scene.entity(chunk.sea))
      scene.destroyEntity(chunk.sea);
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
    gone(p.shown.at(q));
    p.shown.erase(q);
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
    {
      gone(p.shown.at(q));
      p.shown.erase(q);
    }
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
