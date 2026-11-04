// Physics on the host's side (docs/physics-plan.md). For now body contact (figure/physics/contact/
// contact.hpp): the colliders a figure's pass meets, the look each frame at what moved near a figure,
// the figure.contact command and the contact/<prop> topic.

#include "fv_host_impl.hpp"

#include <figure/physics/collide/triangle_set.hpp>

namespace {

fv::Mat4 toMat4(const glm::mat4& m)
{
  fv::Mat4 out;
  for(int c = 0; c < 4; c++)
    for(int r = 0; r < 4; r++)
      out.e[size_t(c * 4 + r)] = m[c][r];
  return out;
}

// A glTF mesh's triangles in its node's space: each triangle primitive's positions and indices. A
// primitive drawn as points or lines meets nothing.
std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>> meshTriangles(const tinygltf::Model& model, int mesh)
{
  std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>> out;
  if(mesh < 0 || size_t(mesh) >= model.meshes.size())
    return out;
  for(const tinygltf::Primitive& p : model.meshes[size_t(mesh)].primitives)
  {
    if(p.mode != TINYGLTF_MODE_TRIANGLES && p.mode != -1)
      continue;
    const auto at = p.attributes.find("POSITION");
    if(at == p.attributes.end() || at->second < 0 || size_t(at->second) >= model.accessors.size())
      continue;
    std::vector<glm::vec3> positions;
    if(!tinygltf::utils::copyAccessorData(model, model.accessors[size_t(at->second)], positions))
      continue;
    std::vector<uint32_t> tris;
    if(p.indices >= 0 && size_t(p.indices) < model.accessors.size())
    {
      if(!tinygltf::utils::copyAccessorData(model, model.accessors[size_t(p.indices)], tris))
        continue;
    }
    else
      for(uint32_t i = 0; i < positions.size(); i++)
        tris.push_back(i);
    tris.resize(tris.size() / 3 * 3);
    std::vector<float> xyz;
    xyz.reserve(positions.size() * 3);
    for(const glm::vec3& v : positions)
      xyz.insert(xyz.end(), {v.x, v.y, v.z});
    out.emplace_back(std::move(xyz), std::move(tris));
  }
  return out;
}

// The box a glTF mesh's positions span, from their accessors' min and max (glTF requires them on
// positions), so a prop's bounds are known before its triangles are read.
fv::physics::Box meshBounds(const tinygltf::Model& model, int mesh)
{
  fv::physics::Box box;
  if(mesh < 0 || size_t(mesh) >= model.meshes.size())
    return box;
  for(const tinygltf::Primitive& p : model.meshes[size_t(mesh)].primitives)
  {
    const auto at = p.attributes.find("POSITION");
    if(at == p.attributes.end() || at->second < 0 || size_t(at->second) >= model.accessors.size())
      continue;
    const tinygltf::Accessor& a = model.accessors[size_t(at->second)];
    if(a.minValues.size() >= 3 && a.maxValues.size() >= 3)
    {
      box.add(a.minValues[0], a.minValues[1], a.minValues[2]);
      box.add(a.maxValues[0], a.maxValues[1], a.maxValues[2]);
    }
  }
  return box;
}

// How far around a body a prop's move can still change its dent: the reach, and how deep a vertex
// is still taken as inside a closed prop.
constexpr double kContactNear = 8;

}  // namespace

std::shared_ptr<const fv::physics::SharedTriangles> FvHost::Impl::contactMesh(const std::shared_ptr<const tinygltf::Model>& model, int mesh)
{
  const auto key = std::make_pair(model.get(), mesh);
  if(const auto it = contactMeshes.find(key); it != contactMeshes.end())
    return it->second;
  auto set = std::make_shared<const fv::physics::SharedTriangles>([model, mesh] { return meshTriangles(*model, mesh); }, meshBounds(*model, mesh));
  contactMeshes.emplace(key, set);
  return set;
}

// Every shown model prop's meshes, placed in the figure root's space (centimetres): where the
// renderer has each node now, so a prop riding a bone or carried by a parent is where it shows.
// Skinned meshes are left out: their triangles are not where the file has them.
std::vector<fv::physics::ColliderRef> FvHost::Impl::contactColliders(const Loaded& f)
{
  std::vector<fv::physics::ColliderRef> out;
  if(!renderer->feedsHostScene() || !f.feed)
    return out;
  rs::RetainedScene& scene = renderer->hostScene();
  const rs::EntityId root  = f.feed->entityOf(f.fig->loaded.root);
  if(!scene.entity(root))
    return out;
  const fv::Mat4 fromWorld = inverse(toMat4(scene.world(root)));
  for(const auto& p : props)
  {
    if(p->kind != PropKind::eModel || !p->model || !shown(*p))
      continue;
    const tinygltf::Model& model = *p->model;
    for(size_t node = 0; node < model.nodes.size() && node < p->imported.entities.size(); node++)
    {
      const tinygltf::Node& n = model.nodes[node];
      const rs::EntityId    e = p->imported.entities[node];
      if(n.mesh < 0 || n.skin >= 0 || !scene.entity(e))
        continue;
      out.push_back({contactMesh(p->model, n.mesh), fromWorld * toMat4(scene.world(e)), 1.0});
    }
  }
  return out;
}

// Once a frame, before the figures pump: a model prop that moved, showed, hid, came or went pokes
// the contact of each figure whose body it is near, before or after; a figure placed elsewhere meets
// its colliders anew. What a figure's own pose or shape does pokes it through its changes.
void FvHost::Impl::pokeContacts()
{
  std::vector<Loaded*> figures;
  for(const auto& p : props)
    if(p->figure && p->figure->contact && p->figure->contact->enabled() && p->figure->feed)
      figures.push_back(p->figure.get());
  if(figures.empty() || !renderer->feedsHostScene())
  {
    contactSeen.clear();
    return;
  }
  rs::RetainedScene& scene = renderer->hostScene();
  std::unordered_map<std::string, ContactSeen> now;
  for(const auto& p : props)
  {
    if(p->kind != PropKind::eModel || !p->model)
      continue;
    ContactSeen& seen = now[p->id];
    seen.shown        = shown(*p);
    for(size_t node = 0; node < p->model->nodes.size() && node < p->imported.entities.size(); node++)
    {
      const tinygltf::Node& n = p->model->nodes[node];
      const rs::EntityId    e = p->imported.entities[node];
      if(n.mesh < 0 || n.skin >= 0 || !scene.entity(e))
        continue;
      const glm::mat4 world = scene.world(e);
      seen.worlds.push_back(world);
      seen.bounds.add(fv::physics::transformed(contactMesh(p->model, n.mesh)->bounds(), toMat4(world)));
    }
  }
  std::vector<fv::physics::Box> changed;
  bool                          cameOrWent = now.size() != contactSeen.size();
  for(const auto& [id, s] : now)
  {
    const auto was = contactSeen.find(id);
    if(was != contactSeen.end() && was->second.worlds == s.worlds && was->second.shown == s.shown)
      continue;
    cameOrWent = cameOrWent || was == contactSeen.end();
    if(s.shown)
      changed.push_back(s.bounds);
    if(was != contactSeen.end() && was->second.shown)
      changed.push_back(was->second.bounds);
  }
  for(const auto& [id, s] : contactSeen)
    if(!now.count(id))
    {
      cameOrWent = true;
      if(s.shown)
        changed.push_back(s.bounds);
    }
  contactSeen = std::move(now);
  // a model no prop shows any more takes its triangles with it
  if(cameOrWent)
    for(auto it = contactMeshes.begin(); it != contactMeshes.end();)
    {
      const bool used = std::any_of(props.begin(), props.end(), [&](const auto& p) { return p->model.get() == it->first.first; });
      it              = used ? std::next(it) : contactMeshes.erase(it);
    }
  for(Loaded* f : figures)
  {
    const rs::EntityId root = f->feed->entityOf(f->fig->loaded.root);
    if(!scene.entity(root))
      continue;
    const glm::mat4 placed = scene.world(root);
    if(f->contactPlaced != placed)
    {
      f->contactPlaced = placed;
      f->contact->poke();
      continue;
    }
    const fv::physics::Box& body = f->contact->bodyBounds();
    if(changed.empty())
      continue;
    if(body.empty())
    {
      f->contact->poke();
      continue;
    }
    const double           scale = glm::length(glm::vec3(placed[0]));
    const fv::physics::Box around = fv::physics::transformed(body, toMat4(placed)).grown((f->contact->settings().reach + kContactNear) * scale);
    if(std::any_of(changed.begin(), changed.end(), [&](const fv::physics::Box& b) { return b.overlaps(around); }))
      f->contact->poke();
  }
}

// Topic "contact/<prop>": {enabled, busy, patches, softness, bulge, reach}: the body's contact dents
// (figure/physics/contact/contact.hpp), for a figure that has them.
void FvHost::Impl::publishContact(Loaded& f)
{
  if(!f.contact)
    return;
  f.contactVersion                 = f.contact->version();
  const fv::physics::ContactSettings& s = f.contact->settings();
  protocol->events().snapshot("contact/" + f.id, {{"enabled", s.on},
                                                  {"busy", f.contact->busy()},
                                                  {"patches", f.contact->patches()},
                                                  {"softness", s.softness},
                                                  {"bulge", s.bulge},
                                                  {"reach", s.reach}});
}

// A figure's contact settings as the scene file and the history keep them.
Json contactState(const Loaded& f)
{
  const fv::physics::ContactSettings& s = f.contact->settings();
  return {{"on", s.on}, {"softness", s.softness}, {"bulge", s.bulge}, {"reach", s.reach}};
}

// figure.contact {prop, on?, softness?, bulge?, reach?, gesture?}: the body dents where a prop presses
// into it (on by default), how much of the volume pushed in bulges around the dent (0 to 1), how far
// along the skin it reaches at least (1 to 10 cm), and the body's share of an overlap with another
// figure (0 to 1).
Json FvHost::Impl::figureContact(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->contact)
    throw std::runtime_error("no figure with a body is selected");
  fv::physics::ContactSettings s = figure->contact->settings();
  if(args.contains("on") && args["on"].is_boolean())
    s.on = args["on"].get<bool>();
  const auto number = [&](const char* key, float& out, float lo, float hi) {
    if(args.contains(key) && args[key].is_number())
      out = std::clamp(args[key].get<float>(), lo, hi);
  };
  number("softness", s.softness, 0.f, 1.f);
  number("bulge", s.bulge, 0.f, 1.f);
  number("reach", s.reach, 1.f, 10.f);
  figure->contact->set(s);
  publishContact(*figure);
  return {{"ok", true}};
}
