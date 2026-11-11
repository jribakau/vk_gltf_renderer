//
// FvHost: the backdrop (fv_backdrop.hpp), the lighting's `backdrop` section drawn.
//
// The kit is a folder the app fills (settings.apply's backdropKit) with plant models and ground
// textures and a kit.json that says what each is:
//
//   {"version": 1,
//    "plants": [{"role": "grass" | "flower" | "sapling" | "tree", "file": "<glTF, relative>",
//                "heaviest"?: <triangles: a variant of the file with more is left out>}, ...],
//    "ground": {"<kind>": {"diffuse"?, "normal"?, "arm"?: "<image, relative>", "tile"?: <metres>,
//                          "color"?: [r, g, b] linear, without a diffuse map,
//                          "tint"?, "tintDry"?: [r, g, b] linear, what the colour is multiplied by,
//                          the second in the terrain's drier drifts (tint's when not given)}}}
//
// Every mesh of a plant's file is a variant of it. The kit is read on a worker (a tree's file is
// hundreds of megabytes) and kept; what stands in the scene is one root entity of the host's own, as
// the sun is, with the ground under it; a tree's or a sapling's variant an entity, drawn at its
// places as instancing copies; and the grass and the flowers as tiles (fv_backdrop.hpp), an entity a
// tile drawn, all of a level sharing one mesh. Another seed, size, density or relief plants again in
// place: the copies, the tiles' meshes, and the ground when its shape changed. The camera's moves lay
// the tiles again (followBackdrop) and plant nothing.
//

#include "fv_host_impl.hpp"

#include <fstream>

#include "fv_compose.hpp"
#include "scene_import_gltf.hpp"

namespace {

uint64_t trianglesOf(const tinygltf::Model& model, const tinygltf::Mesh& mesh)
{
  uint64_t n = 0;
  for(const tinygltf::Primitive& p : mesh.primitives)
  {
    const int a = p.indices >= 0 ? p.indices : (p.attributes.count("POSITION") ? p.attributes.at("POSITION") : -1);
    if(a >= 0 && a < int(model.accessors.size()))
      n += model.accessors[size_t(a)].count / 3;
  }
  return n;
}

}  // namespace

tinygltf::Model FvHost::Impl::backdropTerrainModel(const fv::backdrop::Terrain& t, const BackdropKit::Ground& g)
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
  tinygltf::Primitive prim;
  prim.mode                      = TINYGLTF_MODE_TRIANGLES;
  prim.attributes["POSITION"]    = put(t.positions.data(), t.positions.size() * sizeof(glm::vec3), TINYGLTF_TARGET_ARRAY_BUFFER,
                                       TINYGLTF_COMPONENT_TYPE_FLOAT, t.positions.size(), TINYGLTF_TYPE_VEC3);
  prim.attributes["NORMAL"]      = put(t.normals.data(), t.normals.size() * sizeof(glm::vec3), TINYGLTF_TARGET_ARRAY_BUFFER,
                                       TINYGLTF_COMPONENT_TYPE_FLOAT, t.normals.size(), TINYGLTF_TYPE_VEC3);
  prim.attributes["TEXCOORD_0"]  = put(t.uvs.data(), t.uvs.size() * sizeof(glm::vec2), TINYGLTF_TARGET_ARRAY_BUFFER,
                                       TINYGLTF_COMPONENT_TYPE_FLOAT, t.uvs.size(), TINYGLTF_TYPE_VEC2);
  // the two tints by the terrain's shade, as vertex colours: they multiply the map
  std::vector<glm::vec3> colors(t.shades.size());
  for(size_t i = 0; i < colors.size(); i++)
    colors[i] = glm::mix(g.tint, g.tintDry, t.shades[i]);
  if(colors.size() == t.positions.size() && (g.tint != glm::vec3(1.0F) || g.tintDry != glm::vec3(1.0F)))
    prim.attributes["COLOR_0"] = put(colors.data(), colors.size() * sizeof(glm::vec3), TINYGLTF_TARGET_ARRAY_BUFFER,
                                     TINYGLTF_COMPONENT_TYPE_FLOAT, colors.size(), TINYGLTF_TYPE_VEC3);
  prim.indices                   = put(t.indices.data(), t.indices.size() * sizeof(uint32_t), TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER,
                                       TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, t.indices.size(), TINYGLTF_TYPE_SCALAR);
  glm::vec3 lo(std::numeric_limits<float>::max()), hi(std::numeric_limits<float>::lowest());
  for(const glm::vec3& p : t.positions)
  {
    lo = glm::min(lo, p);
    hi = glm::max(hi, p);
  }
  m.accessors[size_t(prim.attributes["POSITION"])].minValues = {lo.x, lo.y, lo.z};
  m.accessors[size_t(prim.attributes["POSITION"])].maxValues = {hi.x, hi.y, hi.z};
  m.buffers.push_back(std::move(buffer));

  tinygltf::Material material;
  material.name = "Backdrop ground";
  const auto texture = [&](const std::string& uri) {
    tinygltf::Image image;
    image.uri = uri;
    m.images.push_back(image);
    tinygltf::Texture tex;
    tex.source = int(m.images.size()) - 1;
    m.textures.push_back(tex);
    return int(m.textures.size()) - 1;
  };
  material.pbrMetallicRoughness.roughnessFactor = 1.0;
  material.pbrMetallicRoughness.metallicFactor  = 0.0;
  if(!g.diffuse.empty())
    material.pbrMetallicRoughness.baseColorTexture.index = texture(g.diffuse);
  else
    material.pbrMetallicRoughness.baseColorFactor = {g.color.x, g.color.y, g.color.z, 1.0};
  if(!g.normal.empty())
    material.normalTexture.index = texture(g.normal);
  if(!g.arm.empty())
  {
    // occlusion, roughness and metal in one image, as glTF packs them
    material.pbrMetallicRoughness.metallicRoughnessTexture.index = texture(g.arm);
    material.pbrMetallicRoughness.metallicFactor                 = 1.0;
  }
  m.materials.push_back(material);
  prim.material = 0;

  tinygltf::Mesh mesh;
  mesh.name = "Backdrop ground";
  mesh.primitives.push_back(prim);
  m.meshes.push_back(mesh);
  tinygltf::Node node;
  node.name = "Backdrop ground";
  node.mesh = 0;
  m.nodes.push_back(node);
  tinygltf::Scene scene;
  scene.nodes = {0};
  m.scenes.push_back(scene);
  m.defaultScene = 0;
  return m;
}

std::shared_ptr<const FvHost::Impl::BackdropKit> FvHost::Impl::readBackdropKit(const std::filesystem::path& dir)
{
  using Kit = BackdropKit;
  std::ifstream in(dir / "kit.json", std::ios::binary);
  if(!in)
    throw std::runtime_error("no kit.json in " + utf8Of(dir));
  const Json j = Json::parse(in, nullptr, false);
  if(!j.is_object() || !j.contains("plants") || !j["plants"].is_array())
    throw std::runtime_error("kit.json lists no plants");
  auto kit = std::make_shared<Kit>();
  kit->dir = dir;
  for(const Json& p : j["plants"])
  {
    if(!p.is_object() || !p.contains("role") || !p["role"].is_string() || !p.contains("file") || !p["file"].is_string())
      throw std::runtime_error("kit.json: a plant needs a role and a file");
    Kit::Plant plant{.role = p["role"].get<std::string>(), .file = dir / pathOf(p["file"].get<std::string>())};
    auto        model = std::make_shared<tinygltf::Model>();
    std::string error;
    if(!fv::composer::readModelFile(plant.file, *model, error))
      throw std::runtime_error(utf8Of(plant.file.filename()) + ": " + error);
    // a variant too heavy for a backdrop draws nothing: with no primitives the import makes no mesh of it
    if(const uint64_t heaviest = p.value("heaviest", uint64_t(0)); heaviest > 0)
      for(tinygltf::Mesh& mesh : model->meshes)
        if(trianglesOf(*model, mesh) > heaviest)
          mesh.primitives.clear();
    plant.model = std::move(model);
    kit->plants.push_back(std::move(plant));
  }
  if(j.contains("ground") && j["ground"].is_object())
    for(auto it = j["ground"].begin(); it != j["ground"].end(); ++it)
    {
      const Json& g = it.value();
      if(!g.is_object())
        continue;
      Kit::Ground ground;
      ground.diffuse = g.value("diffuse", std::string());
      ground.normal  = g.value("normal", std::string());
      ground.arm     = g.value("arm", std::string());
      ground.tile    = std::max(g.value("tile", ground.tile), 0.1F);
      const auto rgb = [&](const char* key, glm::vec3& out) {
        if(!g.contains(key) || !g[key].is_array() || g[key].size() != 3 || !g[key][0].is_number() || !g[key][1].is_number() || !g[key][2].is_number())
          return false;
        out = {g[key][0].get<float>(), g[key][1].get<float>(), g[key][2].get<float>()};
        return true;
      };
      rgb("color", ground.color);
      rgb("tint", ground.tint);
      if(!rgb("tintDry", ground.tintDry))
        ground.tintDry = ground.tint;
      kit->ground[it.key()] = ground;
    }
  return kit;
}

rs::EntityId FvHost::Impl::backdropRoot() const
{
  return backdropBuilt ? backdropBuilt->root : rs::EntityId{};
}

// The root at the floor's height: the ground is flat there, where the figures stand
void FvHost::Impl::placeBackdrop()
{
  if(!backdropBuilt || !renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  const rs::Entity*  root  = scene.entity(backdropBuilt->root);
  if(!root)
    return;
  const float floor = lighting.ground.autoHeight ? autoGround : float(lighting.ground.height);
  if(root->local.translation.y != floor)
    scene.setLocal(backdropBuilt->root, {.translation = {0.0F, floor, 0.0F}});
}

nvutils::Bbox FvHost::Impl::contentBounds()
{
  const rs::EntityId root = backdropRoot();
  if(root.isNone() || !renderer->feedsHostScene() || !renderer->hostScene().entity(root))
    return renderer->visibleSceneBounds();
  // the scene as fed, without the ground that reaches the horizon: what a view frames
  const std::array<glm::vec3, 2> b = renderer->hostScene().bounds(root);
  if(b[0].x <= b[1].x)
    return nvutils::Bbox(b[0], b[1]);
  // only the backdrop: where a figure would stand
  const float floor = lighting.ground.autoHeight ? autoGround : float(lighting.ground.height);
  return nvutils::Bbox(glm::vec3(-1.0F, floor, -1.0F), glm::vec3(1.0F, floor + 2.0F, 1.0F));
}

void FvHost::Impl::plantBackdrop(rs::RetainedScene& scene)
{
  if(!backdropBuilt || !backdropKit || !scene.entity(backdropBuilt->root))
    return;
  fv::timing::Scope             time("backdrop: planting", 20);
  BackdropBuilt&                built = *backdropBuilt;
  const fv::backdrop::Spec&     spec  = lighting.backdrop;
  // the ground, when its shape is another: made again, as its mesh may no longer hold its vertices to move
  if(built.groundSeed != spec.seed || built.groundRelief != spec.relief || !scene.entity(built.ground))
  {
    if(scene.entity(built.ground))
      scene.destroyEntity(built.ground);
    const auto found = backdropKit->ground.find(spec.kind);
    const BackdropKit::Ground g = found != backdropKit->ground.end() ? found->second : BackdropKit::Ground{};
    const tinygltf::Model     m = backdropTerrainModel(fv::backdrop::terrain(spec, g.tile), g);
    const rs::ImportResult    r = rs::importGltf(scene, m, {.prop = 0, .parent = built.root, .imageSearchPaths = {backdropKit->dir}});
    built.ground       = r.roots.empty() ? rs::EntityId{} : r.roots.front();
    built.groundSeed   = spec.seed;
    built.groundRelief = spec.relief;
  }
  // each layer's plants: a variant's places as its copies; a variant with none, and a role no layer
  // plants (the grass and the flowers, which are tiles), hidden: no copies would draw it once, where
  // its file has it
  std::map<std::string, std::vector<std::vector<glm::mat4>>> byRole;
  uint64_t                                                   plants = 0;
  const auto                                                 layers = fv::backdrop::layers(spec);
  for(uint32_t i = 0; i < layers.size(); i++)
  {
    const auto found = built.variants.find(layers[i].role);
    if(found == built.variants.end() || found->second.empty())
      continue;
    std::vector<std::vector<glm::mat4>>  placed = fv::backdrop::scatter(spec, layers[i], i, uint32_t(found->second.size()));
    std::vector<std::vector<glm::mat4>>& all    = byRole[layers[i].role];
    all.resize(placed.size());
    for(size_t v = 0; v < placed.size(); v++)
      all[v].insert(all[v].end(), placed[v].begin(), placed[v].end());
  }
  for(auto& [role, placed] : byRole)
  {
    const std::vector<BackdropBuilt::Variant>& variants = built.variants.at(role);
    for(size_t v = 0; v < variants.size(); v++)
    {
      const rs::Entity* e = scene.entity(variants[v].entity);
      if(!e || !e->mesh)
        continue;
      scene.setVisible(variants[v].entity, !placed[v].empty());
      if(placed[v].empty())
        continue;
      for(glm::mat4& m : placed[v])
        m = m * variants[v].base;
      plants += placed[v].size();
      rs::MeshInstance instance = *e->mesh;
      instance.copies           = std::move(placed[v]);
      scene.setMeshInstance(variants[v].entity, std::move(instance));
    }
  }
  for(const auto& [role, variants] : built.variants)
    if(!byRole.count(role))
      for(const BackdropBuilt::Variant& v : variants)
        if(scene.entity(v.entity))
          scene.setVisible(v.entity, false);
  built.placed = plants;

  // the grass and the flowers: their tiles' meshes made again for this state, and the tiles laid
  for(BackdropBuilt::Tiles* tiles : {&built.grass, &built.flowers})
  {
    for(const auto& [tile, entity] : tiles->drawn)
      if(scene.entity(entity))
        scene.destroyEntity(entity);
    for(const auto& [level, spare] : tiles->spare)
      for(const rs::EntityId entity : spare)
        if(scene.entity(entity))
          scene.destroyEntity(entity);
    for(const BackdropBuilt::Patch& patch : tiles->patches)
      if(scene.entity(patch.holder))
        scene.destroyEntity(patch.holder);
    *tiles = {};
  }
  built.grassWas.clear();
  for(const fv::backdrop::Cover& cover : fv::backdrop::grassCovers(spec))
    built.grass.patches.push_back(makeBackdropPatch(scene, cover));
  for(const fv::backdrop::Cover& cover : fv::backdrop::flowerCovers(spec))
    built.flowers.patches.push_back(makeBackdropPatch(scene, cover));
  tileBackdrop(scene);
}

FvHost::Impl::BackdropBuilt::Patch FvHost::Impl::makeBackdropPatch(rs::RetainedScene& scene, const fv::backdrop::Cover& cover)
{
  BackdropBuilt&       built = *backdropBuilt;
  BackdropBuilt::Patch out;
  const auto           found = built.variants.find(cover.role);
  if(found == built.variants.end() || found->second.empty())
    return out;
  const std::vector<BackdropBuilt::Variant>&       variants = found->second;
  const std::vector<std::vector<glm::mat4>> placed   = fv::backdrop::patch(lighting.backdrop, cover, uint32_t(variants.size()));
  // what every variant's mesh has, the merged one has: colours when any does (white for the rest),
  // tangents and UVs only when all do
  bool colours = false, tangents = true, uvs = true;
  for(const BackdropBuilt::Variant& v : variants)
    if(const rs::Entity* e = scene.entity(v.entity); e && e->mesh)
      for(const rs::MeshPart& part : e->mesh->parts)
        if(const rs::MeshRecord* m = scene.mesh(part.mesh))
        {
          colours  = colours || m->colors.size() == m->positions.size();
          tangents = tangents && m->tangents.size() == m->positions.size();
          uvs      = uvs && m->uv0.size() == m->positions.size();
        }
  rs::MeshRecord merged;
  merged.name  = "Backdrop " + cover.role;
  merged.owner = {0, -1};
  std::vector<rs::MaterialId>        materials;  // a run of the merged mesh each
  std::vector<std::vector<uint32_t>> runs;
  std::vector<glm::vec2>&            uv0 = merged.uv0.own();
  for(size_t v = 0; v < variants.size(); v++)
  {
    const rs::Entity* e = scene.entity(variants[v].entity);
    if(!e || !e->mesh)
      continue;
    for(const glm::mat4& place : placed[v])
    {
      const glm::mat4 to   = place * variants[v].base;
      const glm::mat3 turn = glm::mat3(glm::transpose(glm::inverse(glm::mat3(to))));
      for(const rs::MeshPart& part : e->mesh->parts)
      {
        const rs::MeshRecord* m = scene.mesh(part.mesh);
        if(!m || m->positions.empty())
          continue;
        const bool normals = m->normals.size() == m->positions.size();
        const uint32_t first = uint32_t(merged.positions.size());
        for(size_t i = 0; i < m->positions.size(); i++)
        {
          merged.positions.push_back(glm::vec3(to * glm::vec4(m->positions[i], 1.0F)));
          merged.normals.push_back(normals ? glm::normalize(turn * m->normals[i]) : glm::vec3(0.0F, 1.0F, 0.0F));
          if(tangents)
            merged.tangents.emplace_back(glm::normalize(glm::mat3(to) * glm::vec3(m->tangents[i])), m->tangents[i].w);
          if(uvs)
            uv0.push_back(m->uv0[i]);
          if(colours)
            merged.colors.push_back(m->colors.size() == m->positions.size() ? m->colors[i] : glm::vec4(1.0F));
        }
        for(uint32_t s = 0; s < m->submeshes.size(); s++)
        {
          const rs::Submesh&   run      = m->submeshes[s];
          const rs::MaterialId material = run.material < part.materials.size() ? part.materials[run.material] : rs::MaterialId{};
          size_t               slot     = size_t(std::find(materials.begin(), materials.end(), material) - materials.begin());
          if(slot == materials.size())
          {
            materials.push_back(material);
            runs.emplace_back();
          }
          for(uint32_t i = 0; i < run.indexCount; i++)
            runs[slot].push_back(first + m->indices[run.firstIndex + i]);
        }
      }
    }
    out.plants += placed[v].size();
  }
  if(merged.positions.empty())
    return {};
  std::vector<uint32_t>& indices = merged.indices.own();
  for(size_t slot = 0; slot < runs.size(); slot++)
  {
    merged.submeshes.push_back({.firstIndex = uint32_t(indices.size()), .indexCount = uint32_t(runs[slot].size()),
                                .material = uint32_t(slot), .order = int32_t(slot)});
    indices.insert(indices.end(), runs[slot].begin(), runs[slot].end());
  }
  const rs::MeshId mesh = scene.addMesh(std::move(merged));
  out.holder            = scene.createEntity("Backdrop " + cover.role + " tile", {0, -1}, built.root);
  scene.setVisible(out.holder, false);
  out.instance.parts.push_back({.mesh = mesh, .materials = materials});
  scene.setMeshInstance(out.holder, rs::MeshInstance(out.instance));
  scene.release(mesh);  // the holder's now
  return out;
}

void FvHost::Impl::tileBackdrop(rs::RetainedScene& scene)
{
  if(!backdropBuilt || !scene.entity(backdropBuilt->root))
    return;
  BackdropBuilt&                built = *backdropBuilt;
  const fv::backdrop::Spec&     spec  = lighting.backdrop;
  const std::vector<glm::vec2>  foci  = backdropFociNow();
  uint64_t                      plants = built.placed;
  const auto lay = [&](BackdropBuilt::Tiles& tiles, const std::vector<fv::backdrop::Tile>& wanted, std::span<const fv::backdrop::Cover> covers) {
    const std::set<fv::backdrop::Tile> want(wanted.begin(), wanted.end());
    // a tile no longer asked for: hidden, and its entity the next tile's of that patch
    for(auto it = tiles.drawn.begin(); it != tiles.drawn.end();)
    {
      if(want.count(it->first))
      {
        ++it;
        continue;
      }
      scene.setVisible(it->second, false);
      tiles.spare[it->first.level].push_back(it->second);
      it = tiles.drawn.erase(it);
    }
    for(const fv::backdrop::Tile& tile : wanted)
    {
      if(tile.level >= tiles.patches.size() || tile.level >= covers.size())
        continue;
      const BackdropBuilt::Patch& patch = tiles.patches[tile.level];
      if(!scene.entity(patch.holder))
        continue;
      plants += patch.plants;
      if(tiles.drawn.count(tile))
        continue;
      rs::EntityId              entity;
      std::vector<rs::EntityId>& spare = tiles.spare[tile.level];
      if(!spare.empty())
      {
        entity = spare.back();
        spare.pop_back();
      }
      else
      {
        entity = scene.createEntity("Backdrop tile", {0, -1}, built.root);
        scene.setMeshInstance(entity, rs::MeshInstance(patch.instance));
      }
      const fv::backdrop::Cover& cover = covers[tile.level];
      const fv::backdrop::Place  place = fv::backdrop::tilePlace(spec, cover.tile, tile.x, tile.z, cover.sink);
      scene.setLocal(entity, {.translation = place.position, .rotation = place.rotation});
      scene.setVisible(entity, true);
      tiles.drawn[tile] = entity;
    }
  };
  std::vector<fv::backdrop::Tile> grass = fv::backdrop::grassTiles(spec, foci, built.grassWas);
  const auto                      grassCovers  = fv::backdrop::grassCovers(spec);
  const auto                      flowerCovers = fv::backdrop::flowerCovers(spec);
  lay(built.grass, grass, grassCovers);
  lay(built.flowers, fv::backdrop::flowerTiles(spec, foci), flowerCovers);
  built.grassWas = std::move(grass);
  built.tiledFor = foci;
  backdropPlants = plants;
}

void FvHost::Impl::feedBackdrop(rs::RetainedScene& scene)
{
  if(!backdropKit)
    return;
  fv::timing::Scope time("backdrop: feeding", 20);
  if(backdropBuilt && scene.entity(backdropBuilt->root))
    scene.destroyEntity(backdropBuilt->root);
  backdropBuilt       = std::make_unique<BackdropBuilt>();
  BackdropBuilt& built = *backdropBuilt;
  built.root           = scene.createEntity("Backdrop", {0, -1});
  for(const BackdropKit::Plant& plant : backdropKit->plants)
  {
    rs::ImportResult r;
    try
    {
      r = rs::importGltf(scene, *plant.model, {.prop = 0, .parent = built.root, .imageSearchPaths = {plant.file.parent_path()}});
    }
    catch(const std::exception& e)
    {
      notice("The backdrop's " + utf8Of(plant.file.filename()) + " could not be planted: " + problemOf(e));
      continue;
    }
    for(const rs::ClipId clip : r.clips)
      if(!clip.isNone())
        scene.release(clip);
    // every node that draws is a variant: straight under the root, its own turn and scale kept for
    // its copies and its place in its file (a row for show) dropped
    for(const rs::EntityId e : r.entities)
    {
      const rs::Entity* entity = e.isNone() ? nullptr : scene.entity(e);
      if(!entity || !entity->mesh)
        continue;
      glm::mat4 base = scene.world(e);
      base[3]        = glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
      scene.setParent(e, built.root);
      scene.setLocal(e, {});
      built.variants[plant.role].push_back({.entity = e, .base = base});
    }
  }
  plantBackdrop(scene);
  placeBackdrop();
}

void FvHost::Impl::applyBackdrop(bool reread)
{
  // where the scene's content is, before anything is planted round it
  if(renderer->feedsHostScene() && fv::backdrop::planted(lighting.backdrop))
  {
    const nvutils::Bbox content = contentBounds();
    backdropContent = content.isEmpty() ? glm::vec2(0.0F) : glm::vec2(content.center().x, content.center().z);
  }
  standBackdrop(reread);
  scopeHaze();
}

std::vector<glm::vec2> FvHost::Impl::backdropFociNow()
{
  std::vector<glm::vec2> foci{backdropContent};
  if(const auto camera = renderer->getCameraManipulator())
  {
    const glm::vec3 eye = camera->getEye();
    foci.emplace_back(eye.x, eye.z);
  }
  return foci;
}

void FvHost::Impl::followBackdrop()
{
  if(!backdropBuilt || backdropState != "ready" || !renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  if(!scene.entity(backdropBuilt->root))
    return;
  // once a focus has gone a metre: a tile is two across at the least
  const std::vector<glm::vec2> foci  = backdropFociNow();
  const std::vector<glm::vec2>& last = backdropBuilt->tiledFor;
  bool                          moved = foci.size() != last.size();
  for(size_t i = 0; i < foci.size() && !moved; i++)
    moved = glm::distance(foci[i], last[i]) >= 1.0F;
  if(!moved)
    return;
  tileBackdrop(scene);
  // the topic's count of plants, when it is off by more than a twentieth: not a topic a metre
  const uint64_t said = backdropPlantsSaid;
  if(backdropPlants > said + said / 20 || backdropPlants + said / 20 < said)
    publishLighting();
}

void FvHost::Impl::scopeHaze()
{
  if(backdropRoot().isNone() || !renderer->feedsHostScene() || !renderer->hostScene().entity(backdropRoot()))
  {
    renderer->setHazeScope(nullptr);
    return;
  }
  const nvutils::Bbox content = contentBounds();
  renderer->setHazeScope(&content);
}

void FvHost::Impl::standBackdrop(bool reread)
{
  if(!renderer->feedsHostScene())
    return;
  rs::RetainedScene&        scene = renderer->hostScene();
  const fv::backdrop::Spec& spec  = lighting.backdrop;
  const auto                drop  = [&] {
    if(backdropBuilt && scene.entity(backdropBuilt->root))
      scene.destroyEntity(backdropBuilt->root);
    backdropBuilt.reset();
    backdropPlants = 0;
  };
  std::filesystem::path dir;
  {
    std::lock_guard lock(settingsMutex);
    dir = settings.backdropKit;
  }
  if(reread || (backdropKit && backdropKit->dir != dir))
  {
    drop();
    backdropKit.reset();
    backdropRequest++;  // a read under way is of the folder as it was
    if(backdropState == "loading")
      backdropState = "off";
  }
  if(!fv::backdrop::planted(spec))
  {
    drop();
    backdropState  = "off";
    backdropWanted = spec;
    return;
  }
  // what stands, planted again when the state is another
  if(backdropBuilt && scene.entity(backdropBuilt->root))
  {
    if(!(spec == backdropWanted))
    {
      backdropWanted = spec;
      plantBackdrop(scene);
    }
    placeBackdrop();
    backdropState = "ready";
    return;
  }
  backdropBuilt.reset();  // its entities went with a scene replaced
  backdropWanted = spec;
  if(backdropKit)
  {
    feedBackdrop(scene);
    backdropState = "ready";
    return;
  }
  if(dir.empty())
  {
    backdropState = "missing";
    return;
  }
  if(backdropState == "loading")
    return;  // its kit is being read
  backdropState          = "loading";
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
      if(request != backdropRequest)
        return;
      if(!kit)
      {
        // a folder not filled yet is the app's to say (the topic's state); a kit that is there and
        // cannot be read is a fault
        backdropState = "missing";
        if(error.rfind("no kit.json", 0) != 0)
          notice("The backdrop's plants could not be read: " + error, "error");
        publishLighting();
        return;
      }
      backdropKit   = kit;
      backdropState = "off";
      applyLighting(false);  // plants it, and the ground's plane gives way to it
    });
  });
}
