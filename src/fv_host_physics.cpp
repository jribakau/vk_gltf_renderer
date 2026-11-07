// Physics on the host's side (docs/physics-plan.md): body contact (figure/physics/contact/
// contact.hpp): the colliders a figure's pass meets, props and other people, the look each frame at
// what moved near a figure, the figure.contact command and the contact/<prop> topic; and Physics mode
// (figure/physics/rig/figure_physics.hpp), below.

#include "fv_host_impl.hpp"

#include <figure/physics/collide/triangle_set.hpp>
#include <figure/physics/flesh/follow.hpp>

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

// How far a person's flesh reaches past their bones, cm: their body's bounds without posing its cage.
constexpr double kFleshAroundBones = 25;

// Whether a figure's node shows: neither it nor a node above it hidden or taken off.
bool drawn(const fv::SceneGraph& scene, int node)
{
  for(int n = node; n >= 0; n = scene.nodes[size_t(n)].parent)
    if(!scene.nodes[size_t(n)].visible || scene.nodes[size_t(n)].removed)
      return false;
  return true;
}

// A person's bones where the renderer has them (the world, metres), grown by their flesh: what their
// body may touch, cheap enough to look at every frame. Empty without a pose.
fv::physics::Box boneBounds(const Loaded& f, rs::RetainedScene& scene)
{
  fv::physics::Box out;
  if(!f.feed || !f.fig->pose)
    return out;
  const rs::EntityId root = f.feed->entityOf(f.fig->loaded.root);
  if(!scene.entity(root))
    return out;
  const auto     worlds  = f.fig->loaded.scene.worlds();
  const fv::Mat4 placed  = toMat4(scene.world(root));
  const fv::Mat4 toWorld = placed * inverse(worlds[size_t(f.fig->loaded.root)]);
  for(const fv::pose::PoseBone& pb : f.fig->pose->bones())
    if(pb.bone >= 0 && size_t(pb.bone) < worlds.size())
    {
      const fv::Mat4& w = worlds[size_t(pb.bone)];
      const fv::Vec3  p = toWorld.transformPoint({w.e[12], w.e[13], w.e[14]});
      out.add(p.x, p.y, p.z);
    }
  const double scale = fv::length(fv::Vec3{placed.e[0], placed.e[1], placed.e[2]});
  return out.empty() ? out : out.grown(kFleshAroundBones * scale);
}

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

// A DAZ prop's shown meshes as triangle sets in its root's space, posed as it is now (its morphs and
// bones), or a person's body alone, without its own dent (which is theirs to make, not to be met):
// made again once its pose or shape moved, indexed on the pool the first time a pass needs one.
const std::vector<std::shared_ptr<const fv::physics::SharedTriangles>>& FvHost::Impl::contactSets(Loaded& prop)
{
  if(prop.contactSetsAt == prop.contactRev)
    return prop.contactSets;
  prop.contactSetsAt = prop.contactRev;
  prop.contactSets.clear();
  const fv::SceneGraph& scene = prop.fig->loaded.scene;
  const int             body  = prop.contact ? prop.contact->bodyMesh() : -1;
  // by node: copies of a prop drawing one mesh are each met where they stand
  std::vector<int>      nodes;
  for(size_t n = 0; n < scene.nodes.size(); n++)
    if(scene.nodes[n].mesh >= 0 && (body < 0 || scene.nodes[n].mesh == body) && drawn(scene, int(n)))
      nodes.push_back(int(n));
  std::vector<std::vector<float>> posed =
      fv::physics::posedNodeCages(scene, *prop.fig->morphs, nodes, prop.fig->loaded.root, body >= 0 ? fv::physics::contactChannelOf(body) : std::string());
  for(size_t k = 0; k < nodes.size(); k++)
  {
    const auto& geo = scene.meshes[size_t(scene.nodes[size_t(nodes[k])].mesh)].geo;
    if(posed[k].empty() || !geo || geo->dsonTris.empty())
      continue;
    const fv::physics::Box bounds = fv::physics::boundsOf(posed[k]);
    prop.contactSets.push_back(std::make_shared<const fv::physics::SharedTriangles>(
        [xyz = std::move(posed[k]), tris = geo->dsonTris]() mutable {
          std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>> out;
          out.emplace_back(std::move(xyz), std::move(tris));
          return out;
        },
        bounds));
  }
  return prop.contactSets;
}

// Every shown model prop's meshes and DAZ prop's, placed in the figure root's space (centimetres):
// where the renderer has each node now, so a prop riding a bone or carried by a parent is where it
// shows. A model's skinned meshes are left out: their triangles are not where the file has them.
// With `bodies`, another person's body too, of which this one takes its share of the overlap; for a
// `ragdoll`, the people not in Physics mode.
std::vector<fv::physics::ColliderRef> FvHost::Impl::contactColliders(const Loaded& f, bool bodies, bool ragdoll)
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
  for(const auto& p : props)
  {
    Loaded* prop = p->figure.get();
    if(!prop || prop == &f || (prop->contact && !bodies) || !prop->feed || !shown(*p))
      continue;
    // for a ragdoll, another in Physics mode is met by its capsules (stepTogether)
    if(ragdoll && prop->physics && prop->physics->settings().on)
      continue;
    const rs::EntityId at = prop->feed->entityOf(prop->fig->loaded.root);
    if(!scene.entity(at))
      continue;
    const fv::Mat4 toBody = fromWorld * toMat4(scene.world(at));
    const double   share  = prop->contact && f.contact ? fv::physics::overlapShare(f.contact->settings(), &prop->contact->settings()) : 1.0;
    for(const auto& set : contactSets(*prop))
      out.push_back({set, toBody, share});
  }
  return out;
}

// Another shown figure whose ragdoll is awake, near this one: a dent from it would show where it was
// a moment ago, so this one's dents are held at weight 0 until it rests.
bool FvHost::Impl::movesNear(const Loaded& f)
{
  if(!renderer->feedsHostScene())
    return false;
  rs::RetainedScene& scene = renderer->hostScene();
  fv::physics::Box   mine;
  for(const auto& p : props)
  {
    const Loaded* other = p->figure.get();
    if(!other || other == &f || !other->physics || !other->physics->settings().on || !other->physics->awake() || !shown(*p))
      continue;
    if(mine.empty())
      mine = boneBounds(f, scene);
    if(boneBounds(*other, scene).overlaps(mine))
      return true;
  }
  return false;
}

// What a person's soft flesh meets while it moves (figure/physics/flesh/touch.hpp), in their root's
// space: what their contact pass meets of props, the ragdoll capsules of the people near them where
// their bones are, and with self-contact their own. Nothing while body contact is off.
fv::physics::FleshColliders FvHost::Impl::fleshColliders(Loaded& f)
{
  fv::physics::FleshColliders out;
  if(!f.contact || !f.contact->enabled() || !renderer->feedsHostScene() || !f.feed)
    return out;
  const fv::physics::ContactSettings& mine = f.contact->settings();
  out.bulge                                = mine.bulge;
  out.sets                                 = contactColliders(f, false);
  if(mine.self && f.physics && f.physics->valid())
  {
    const auto& parts = f.physics->parts();
    if(f.limbParts.size() != parts.size())
    {
      f.limbParts.clear();
      for(const fv::physics::FigurePhysics::Part& p : parts)
        f.limbParts.push_back(fv::physics::partOfBone(f.fig->loaded.scene, p.node));
    }
    for(size_t k = 0; k < parts.size(); k++)
      out.limbs.push_back({f.physics->capsuleNow(k), 1.0, f.limbParts[k], f.physics->restCapsule(k)});
  }
  rs::RetainedScene& scene = renderer->hostScene();
  const rs::EntityId root  = f.feed->entityOf(f.fig->loaded.root);
  if(!scene.entity(root))
    return out;
  const fv::Mat4   fromWorld = inverse(toMat4(scene.world(root)));
  fv::physics::Box around;
  for(const auto& p : props)
  {
    const Loaded* other = p->figure.get();
    if(!other || other == &f || !other->physics || !other->physics->valid() || !other->feed || !shown(*p))
      continue;
    if(around.empty())
      around = boneBounds(f, scene);
    const rs::EntityId at = other->feed->entityOf(other->fig->loaded.root);
    if(!scene.entity(at) || !boneBounds(*other, scene).overlaps(around))
      continue;
    const fv::Mat4 toMine = fromWorld * toMat4(scene.world(at));
    const double   scale  = std::cbrt(std::fabs(toMine.e[0] * (toMine.e[5] * toMine.e[10] - toMine.e[6] * toMine.e[9]) - toMine.e[4] * (toMine.e[1] * toMine.e[10] - toMine.e[2] * toMine.e[9])
                                               + toMine.e[8] * (toMine.e[1] * toMine.e[6] - toMine.e[2] * toMine.e[5])));
    const double   share  = fv::physics::overlapShare(mine, other->contact ? &other->contact->settings() : nullptr);
    for(size_t k = 0; k < other->physics->parts().size(); k++)
    {
      const fv::physics::Capsule c = other->physics->capsuleNow(k);
      out.limbs.push_back({{toMine.transformPoint(c.a), toMine.transformPoint(c.b), c.radius * scale}, share, -1, {}});
    }
  }
  return out;
}

// Once a frame, before the figures pump: a model prop, a DAZ prop or another person that moved, showed,
// hid, came or went (or whose pose or shape changed) pokes the contact of each figure whose body it is
// near, before or after; a figure placed elsewhere meets its colliders anew. What a figure's own pose
// or shape does pokes it through its changes, and a dent is no change of pose.
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
  // a clip moves the props it poses every frame, and no pass runs while it plays: the first look after
  // it stops sees what moved
  if(renderer->editAnimationControl().play)
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  std::unordered_map<std::string, ContactSeen> now;
  for(const auto& p : props)
  {
    if(Loaded* prop = p->figure.get(); prop && prop->feed)
    {
      const rs::EntityId at = prop->feed->entityOf(prop->fig->loaded.root);
      if(!scene.entity(at))
        continue;
      ContactSeen& seen    = now[p->id];
      seen.shown           = shown(*p);
      seen.rev             = prop->contactRev;
      const glm::mat4 world = scene.world(at);
      seen.worlds.push_back(world);
      // a person by their bones: posing their cage every frame of a drag would cost milliseconds
      if(prop->contact)
        seen.bounds = boneBounds(*prop, scene);
      else
        for(const auto& set : contactSets(*prop))
          seen.bounds.add(fv::physics::transformed(set->bounds(), toMat4(world)));
      continue;
    }
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
  // what changed, by whose it is
  std::vector<std::pair<std::string, fv::physics::Box>> changed;
  bool                                                  cameOrWent = now.size() != contactSeen.size();
  for(const auto& [id, s] : now)
  {
    const auto was = contactSeen.find(id);
    if(was != contactSeen.end() && was->second.worlds == s.worlds && was->second.shown == s.shown && was->second.rev == s.rev)
      continue;
    cameOrWent = cameOrWent || was == contactSeen.end();
    if(s.shown)
      changed.emplace_back(id, s.bounds);
    if(was != contactSeen.end() && was->second.shown)
      changed.emplace_back(id, was->second.bounds);
  }
  for(const auto& [id, s] : contactSeen)
    if(!now.count(id))
    {
      cameOrWent = true;
      if(s.shown)
        changed.emplace_back(id, s.bounds);
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
    // its own changes poke it through its pump, not here
    const auto others = [&](const std::pair<std::string, fv::physics::Box>& c) { return c.first != f->id; };
    if(std::none_of(changed.begin(), changed.end(), others))
      continue;
    const fv::physics::Box& body = f->contact->bodyBounds();
    if(body.empty())
    {
      f->contact->poke();
      continue;
    }
    const double           scale = glm::length(glm::vec3(placed[0]));
    const fv::physics::Box around = fv::physics::transformed(body, toMat4(placed)).grown((f->contact->settings().reach + kContactNear) * scale);
    if(std::any_of(changed.begin(), changed.end(), [&](const auto& c) { return others(c) && c.second.overlaps(around); }))
      f->contact->poke();
  }
}

// Topic "contact/<prop>": {enabled, self, busy, held, patches, softness, bulge, reach}: the body's
// contact dents (figure/physics/contact/contact.hpp), for a figure that has them; `self` when its own
// parts press on one another too; `held` while they are at weight 0 because what they meet moves (a
// clip, another figure's ragdoll).
void FvHost::Impl::publishContact(Loaded& f)
{
  if(!f.contact)
    return;
  f.contactVersion                 = f.contact->version();
  const fv::physics::ContactSettings& s = f.contact->settings();
  protocol->events().snapshot("contact/" + f.id, {{"enabled", s.on},
                                                  {"self", s.self},
                                                  {"busy", f.contact->busy()},
                                                  {"held", f.contact->held()},
                                                  {"patches", f.contact->patches()},
                                                  {"softness", s.softness},
                                                  {"bulge", s.bulge},
                                                  {"reach", s.reach}});
}

// A figure's contact settings as the scene file and the history keep them.
Json contactState(const Loaded& f)
{
  const fv::physics::ContactSettings& s = f.contact->settings();
  return {{"on", s.on}, {"self", s.self}, {"softness", s.softness}, {"bulge", s.bulge}, {"reach", s.reach}};
}

// figure.contact {prop, on?, self?, softness?, bulge?, reach?, gesture?}: the body dents where a prop
// presses into it (on by default), and where its own parts press on one another (`self`, on by
// default), how much of the volume pushed in bulges around the dent (0 to 1), how far along the skin it
// reaches at least (1 to 10 cm), and the body's share of an overlap with another figure (0 to 1).
Json FvHost::Impl::figureContact(const Json& args)
{
  auto figure = bodyArg(args, "body contact");
  if(!figure || !figure->contact)
    throw std::runtime_error("no figure with a body is selected");
  fv::physics::ContactSettings s = figure->contact->settings();
  if(args.contains("on") && args["on"].is_boolean())
    s.on = args["on"].get<bool>();
  if(args.contains("self") && args["self"].is_boolean())
    s.self = args["self"].get<bool>();
  const auto number = [&](const char* key, float& out, float lo, float hi) {
    if(args.contains(key) && args[key].is_number())
      out = std::clamp(args[key].get<float>(), lo, hi);
  };
  number("softness", s.softness, 0.f, 1.f);
  number("bulge", s.bulge, 0.f, 1.f);
  number("reach", s.reach, 1.f, 10.f);
  const fv::physics::ContactSettings was = figure->contact->settings();
  figure->contact->set(s);
  // the people it presses into give way by its softness against theirs, or all of it while it's off
  if((s.on != was.on || s.softness != was.softness) && renderer->feedsHostScene())
  {
    rs::RetainedScene&     scene = renderer->hostScene();
    const fv::physics::Box mine  = boneBounds(*figure, scene);
    for(const auto& p : props)
      if(Loaded* other = p->figure.get(); other && other != figure.get() && other->contact && boneBounds(*other, scene).overlaps(mine))
        other->contact->poke();
  }
  publishContact(*figure);
  return {{"ok", true}};
}

//--------------------------------------------------------------------------------------------------
// Physics mode (figure/physics/rig/figure_physics.hpp, docs/physics-plan.md step 2): the frame's
// step, the Pull, physics.set and physics.restore, the physics/<prop> topic.

// The Hold of each body group that has its own ({"Left arm": 0.2, ...}).
static Json partsJson(const fv::physics::PhysicsSettings& s)
{
  Json parts = Json::object();
  for(const auto& [group, strength] : s.parts)
    parts[group] = strength;
  return parts;
}

// A figure's Physics mode as the scene file and the history keep it: the settings and, in Physics
// mode, where its parts are.
Json physicsState(const Loaded& f)
{
  const fv::physics::PhysicsSettings& s        = f.physics->settings();
  Json                                physical = Json::object();
  if(s.on)
    for(const auto& [key, value] : f.physics->physicalValues())
      physical[key] = value;
  Json pins = Json::array();
  for(const fv::physics::FigurePhysics::HeldPin& h : f.physics->pins())
    pins.push_back({{"bone", f.fig->pose->bones()[size_t(h.poseBone)].id},
                    {"turn", h.turn},
                    {"at", {h.at.x, h.at.y, h.at.z}},
                    {"rotation", {h.rotation.x, h.rotation.y, h.rotation.z, h.rotation.w}},
                    {"maxForce", h.maxForce}});
  return {{"on", s.on}, {"strength", s.strength}, {"gravity", s.gravity}, {"letFall", s.letFall}, {"parts", partsJson(s)}, {"physical", physical}, {"pins", pins}};
}

// What puts a figure's physics back as it is now: its parts' channels (where the motors aim), then
// the mode, the settings and where the parts are.
Json FvHost::Impl::physicsOps(const Loaded& f)
{
  Json values = Json::object();
  for(const fv::physics::FigurePhysics::Part& part : f.physics->parts())
    for(const int b : part.members)
    {
      const fv::pose::PoseBone& pb = f.fig->pose->bones()[size_t(b)];
      for(const fv::pose::Axis& a : pb.axes)
        if(a.kind != fv::pose::ChannelKind::eScale)
          values[pb.id + "/" + fv::pose::kindName(a.kind) + "/" + a.axis] = a.value;
    }
  Json state    = physicsState(f);
  state["prop"] = f.id;
  return Json::array({{{"cmd", "pose.set"}, {"args", {{"prop", f.id}, {"values", values}}}}, {{"cmd", "physics.restore"}, {"args", state}}});
}

// Topic "physics/<prop>": {available, on, awake, pulling, strength, parts, groups, gravity, letFall,
// pins}: the figure's Physics mode; `parts` the body groups with a Hold of their own, `groups` all it
// has, `pins` the parts held where they are ({bone, label, turn}).
void FvHost::Impl::publishPhysics(Loaded& f)
{
  if(!f.physics)
    return;
  f.physicsVersion                      = f.physics->version();
  const fv::physics::PhysicsSettings& s = f.physics->settings();
  Json pins = Json::array();
  for(const fv::physics::FigurePhysics::HeldPin& h : f.physics->pins())
  {
    const fv::pose::PoseBone& pb = f.fig->pose->bones()[size_t(h.poseBone)];
    pins.push_back({{"bone", pb.id}, {"label", pb.label}, {"turn", h.turn}});
  }
  // the body groups its parts are in, in order, each once: what may have a Hold of its own
  Json groups = Json::array();
  for(size_t i = 1; i < f.physics->parts().size(); i++)
    if(const std::string& g = f.physics->parts()[i].group; std::find(groups.begin(), groups.end(), g) == groups.end())
      groups.push_back(g);
  protocol->events().snapshot("physics/" + f.id, {{"available", f.physics->valid()},
                                                  {"on", s.on},
                                                  {"awake", f.physics->awake()},
                                                  {"pulling", f.physics->pulling()},
                                                  {"strength", s.strength},
                                                  {"parts", partsJson(s)},
                                                  {"groups", groups},
                                                  {"pins", pins},
                                                  {"gravity", s.gravity},
                                                  {"letFall", s.letFall}});
}

// Once a frame, after the queued edits and the clip, before the figures pump: each figure in Physics
// mode gets the world it is in (gravity and the ground in its root's space) and simulates the time
// since the last frame. What it reaches poses its bones, which the pump then takes.
void FvHost::Impl::stepPhysics()
{
  const auto   now     = std::chrono::steady_clock::now();
  const double elapsed = physicsFrame == std::chrono::steady_clock::time_point{} ? 0.0 : std::chrono::duration<double>(now - physicsFrame).count();
  physicsFrame         = now;
  physicsElapsed       = elapsed;
  std::vector<fv::physics::Together> together;
  for(const auto& p : props)
  {
    if(!p->figure || !p->figure->physics)
      continue;
    Loaded&    f    = *p->figure;
    const bool soft = f.secondary && f.secondary->settings().on;
    if((f.physics->settings().on || soft) && renderer->feedsHostScene() && f.feed)
    {
      rs::RetainedScene& scene = renderer->hostScene();
      const rs::EntityId root  = f.feed->entityOf(f.fig->loaded.root);
      if(scene.entity(root))
      {
        const glm::mat4 toRoot = glm::inverse(scene.world(root));
        const glm::vec3 down   = glm::normalize(glm::vec3(toRoot * glm::vec4(0, -1, 0, 0)));
        f.physics->setGravity({down.x * 980.0, down.y * 980.0, down.z * 980.0});
        if(f.secondary)
          f.secondary->setGravity({down.x * 980.0, down.y * 980.0, down.z * 980.0});
        if(lighting.ground.mode != "off")
        {
          const double    height = lighting.ground.autoHeight ? double(autoGround) : lighting.ground.height;
          const glm::vec3 at     = glm::vec3(toRoot * glm::vec4(0, float(height), 0, 1));
          const glm::vec3 up     = -down;
          f.physics->setGround(fv::physics::Plane{{up.x, up.y, up.z}, double(glm::dot(up, at))});
        }
        else
          f.physics->setGround(std::nullopt);
      }
    }
    // a clip playing on it keeps it awake and is swept across the frame
    f.physics->setClip(renderer->editAnimationControl().play && figureClipOf(currentClip) == &f);
    fv::Mat4 placed;
    if(renderer->feedsHostScene() && f.feed)
      if(const rs::EntityId root = f.feed->entityOf(f.fig->loaded.root); renderer->hostScene().entity(root))
        placed = toMat4(renderer->hostScene().world(root));
    together.push_back({f.physics.get(), placed, p->id});
  }
  // the ragdolls, each meeting the others as they were at the frame's start
  {
    fv::timing::Scope t("physics: frame", 2);
    fv::physics::stepTogether(together, elapsed);
  }
  for(const auto& p : props)
  {
    if(!p->figure || !p->figure->physics)
      continue;
    Loaded& f = *p->figure;
    if(f.physicsVersion != f.physics->version())
      publishPhysics(f);
    // the chest's soft bones answer the chest as the skeleton or the ragdoll has it now
    if(f.secondary)
    {
      fv::timing::Scope t("secondary: frame", 2);
      f.secondary->step(elapsed);
      if(f.secondaryVersion != f.secondary->version())
        publishSecondary(f);
    }
  }
}

// Once a frame for each person, after their pose and shape are on the graph: the soft flesh follows
// the skin as it is now, and where it has strayed from it goes to the renderer, with what is worn
// over it.
void FvHost::Impl::stepFlesh(Loaded& f)
{
  if(!f.flesh)
    return;
  bool moved;
  {
    fv::timing::Scope t("flesh: frame", 2);
    moved = f.flesh->step(physicsElapsed);
  }
  if(moved && renderer->feedsHostScene() && f.feed)
    showFlesh(f);
  if(f.fleshVersion != f.flesh->version())
    publishSoft(f);
}

// The body's offsets from its skin, and each worn item's that is bound near them (physics::followFlesh),
// in that item mesh's own posed space. The renderer adds a subdivided mesh's per cage vertex, after its
// cage's skin; a mesh drawn as it is takes them per drawn vertex, after its skin.
void FvHost::Impl::showFlesh(Loaded& f)
{
  fv::timing::Scope     t("flesh: show", 2);
  rs::RetainedScene&    scene = renderer->hostScene();
  const fv::SceneGraph& graph = f.fig->loaded.scene;
  const auto            put   = [&](int mesh, std::span<const float> cage) {
    if(f.feed->refines(mesh) || !graph.meshes[size_t(mesh)].geo)
      scene.setMeshDisplacement(f.feed->meshOf(mesh), cage);
    else
    {
      fv::physics::drawnOffsets(cage, graph.meshes[size_t(mesh)].geo->renderIndex, fleshDrawn);
      scene.setMeshDisplacement(f.feed->meshOf(mesh), fleshDrawn);
    }
  };
  const int                    bodyMesh = f.flesh->bodyMesh();
  const std::span<const float> body     = f.flesh->displacement();
  put(bodyMesh, body);
  const fv::physics::FleshSolver* solver = f.flesh->solver();
  if(!solver)
    return;
  std::vector<int>      nodeOf;
  std::vector<fv::Mat4> worlds;
  for(const fv::WornItem& w : f.fig->items)
    for(const fv::WornItem::Follow& follow : w.follows)
    {
      if(!follow.binding || follow.mesh < 0 || follow.mesh >= int(graph.meshes.size()) || !fv::physics::bindsTo(*follow.binding, solver->model().vertex))
        continue;
      // the item's posed space against the body's: their nodes' worlds, which DAZ's conforming items share
      if(nodeOf.empty())
      {
        nodeOf = fv::physics::nodeOfMesh(graph);
        worlds = graph.worlds();
      }
      const int        bodyNode = nodeOf[size_t(bodyMesh)], itemNode = nodeOf[size_t(follow.mesh)];
      std::optional<fv::Mat4> toItem;
      if(bodyNode >= 0 && itemNode >= 0 && bodyNode != itemNode)
      {
        const fv::Mat4 m    = fv::inverse(worlds[size_t(itemNode)]) * worlds[size_t(bodyNode)];
        bool           same = true;
        for(int c = 0; c < 3; c++)
          for(int r = 0; r < 3; r++)
            same = same && std::abs(m.e[size_t(c * 4 + r)] - (c == r ? 1.0 : 0.0)) < 1e-6;
        if(!same)
          toItem = m;
      }
      fv::physics::followFlesh(*follow.binding, follow.rigidity.get(), body, toItem ? &*toItem : nullptr, fleshItem);
      put(follow.mesh, fleshItem);
    }
}

namespace {

Json softnessJson(const fv::physics::FleshSettings& s)
{
  Json out = Json::object();
  for(int k = 0; k < fv::physics::kFleshParts; k++)
    out[fv::physics::fleshPartName(fv::physics::FleshPart(k))] = s.softness[size_t(k)];
  return out;
}

}  // namespace

// Topic "soft/<prop>": {available, on, awake, softness {chest, glutes, belly, thighs}, damping}: the
// soft flesh (figure/physics/flesh/flesh.hpp).
void FvHost::Impl::publishSoft(Loaded& f)
{
  if(!f.flesh)
    return;
  f.fleshVersion                      = f.flesh->version();
  const fv::physics::FleshSettings& s = f.flesh->settings();
  protocol->events().snapshot("soft/" + f.id, {{"available", f.flesh->valid()},
                                              {"on", s.on},
                                              {"awake", f.flesh->awake()},
                                              {"softness", softnessJson(s)},
                                              {"damping", s.damping}});
}

// The soft flesh's settings as the scene file and the history keep them.
Json softState(const Loaded& f)
{
  const fv::physics::FleshSettings& s = f.flesh->settings();
  return {{"on", s.on}, {"softness", softnessJson(s)}, {"damping", s.damping}};
}

// physics.soft {prop, on?, softness?: {chest?, glutes?, belly?, thighs?}, damping?, gesture?}: the
// soft flesh trails the body's motion (off by default), how soft each part is (0 held to the skin to
// 1 loosest) and how soon it settles (0 to 1). A body with nothing to move keeps the settings.
Json FvHost::Impl::physicsSoft(const Json& args)
{
  auto figure = bodyArg(args, "soft flesh");
  if(!figure || !figure->flesh)
    throw std::runtime_error("no figure with a body is selected");
  fv::physics::FleshSettings s = figure->flesh->settings();
  if(args.contains("on") && args["on"].is_boolean())
    s.on = args["on"].get<bool>();
  if(const auto it = args.find("softness"); it != args.end() && it->is_object())
    for(int k = 0; k < fv::physics::kFleshParts; k++)
      if(const auto v = it->find(fv::physics::fleshPartName(fv::physics::FleshPart(k))); v != it->end() && v->is_number())
        s.softness[size_t(k)] = std::clamp(v->get<double>(), 0.0, 1.0);
  if(args.contains("damping") && args["damping"].is_number())
    s.damping = std::clamp(args["damping"].get<double>(), 0.0, 1.0);
  figure->flesh->set(s);
  publishSoft(*figure);
  return {{"ok", true}};
}

// Topic "secondary/<prop>": {available, on, awake, bounce, damping}: the chest's soft bones
// (figure/physics/rig/secondary.hpp).
void FvHost::Impl::publishSecondary(Loaded& f)
{
  if(!f.secondary)
    return;
  f.secondaryVersion                      = f.secondary->version();
  const fv::physics::SecondarySettings& s = f.secondary->settings();
  protocol->events().snapshot("secondary/" + f.id,
                              {{"available", f.secondary->valid()}, {"on", s.on}, {"awake", f.secondary->awake()}, {"bounce", s.bounce}, {"damping", s.damping}});
}

// The soft bones' settings as the scene file and the history keep them.
Json secondaryState(const Loaded& f)
{
  const fv::physics::SecondarySettings& s = f.secondary->settings();
  return {{"on", s.on}, {"bounce", s.bounce}, {"damping", s.damping}};
}

// physics.secondary {prop, on?, bounce?, damping?, gesture?}: the chest's soft bones answer its
// motion (off by default), how loose (0 firm to 1 loose) and how soon they settle (0 to 1). A figure
// without such bones keeps the settings and nothing moves.
Json FvHost::Impl::physicsSecondary(const Json& args)
{
  auto figure = bodyArg(args, "the chest's soft bones");
  if(!figure || !figure->secondary)
    throw std::runtime_error("no figure with a body is selected");
  fv::physics::SecondarySettings s = figure->secondary->settings();
  if(args.contains("on") && args["on"].is_boolean())
    s.on = args["on"].get<bool>();
  if(args.contains("bounce") && args["bounce"].is_number())
    s.bounce = std::clamp(args["bounce"].get<double>(), 0.0, 1.0);
  if(args.contains("damping") && args["damping"].is_number())
    s.damping = std::clamp(args["damping"].get<double>(), 0.0, 1.0);
  figure->secondary->set(s);
  publishSecondary(*figure);
  return {{"ok", true}};
}

// physics.set {prop, on?, strength?, parts?, gravity?, letFall?, gesture?}: Physics mode on or off
// (leaving keeps the pose it reached in the channels), Hold (0 limp to 1 rigid), a body group's own
// Hold (parts: {"Left arm": 0.2}; null: back to the figure's), gravity, and whether the hip is let
// fall.
Json FvHost::Impl::physicsSet(const Json& args)
{
  auto figure = bodyArg(args, "Physics mode");
  if(!figure || !figure->physics)
    throw std::runtime_error("no figure with a body is selected");
  fv::physics::PhysicsSettings s = figure->physics->settings();
  if(args.contains("on") && args["on"].is_boolean())
    s.on = args["on"].get<bool>();
  if(args.contains("strength") && args["strength"].is_number())
    s.strength = std::clamp(args["strength"].get<double>(), 0.0, 1.0);
  if(const auto it = args.find("parts"); it != args.end() && it->is_object())
    for(auto g = it->begin(); g != it->end(); ++g)
      if(g.value().is_number())
        s.parts[g.key()] = std::clamp(g.value().get<double>(), 0.0, 1.0);
      else if(g.value().is_null())
        s.parts.erase(g.key());
  if(args.contains("gravity") && args["gravity"].is_boolean())
    s.gravity = args["gravity"].get<bool>();
  if(args.contains("letFall") && args["letFall"].is_boolean())
    s.letFall = args["letFall"].get<bool>();
  if(s.on && !figure->physics->valid())
    throw std::runtime_error("this figure has no body bones for physics");
  const bool leaving = figure->physics->settings().on && !s.on;
  figure->physics->set(s);
  if(leaving)
  {
    // the channels took the reached pose: every panel showing them
    std::vector<std::string> bones;
    for(const auto& part : figure->physics->parts())
      for(const int b : part.members)
        bones.push_back(figure->fig->pose->bones()[size_t(b)].id);
    patchPose(*figure, bones);
  }
  publishPhysics(*figure);
  return {{"ok", true}};
}

// physics.restore {prop, on, strength, gravity, letFall, physical}: the history's and a scene file's
// way back to a figure's Physics mode, its channels already set. Off drops the physical layer
// without keeping it in the channels; on takes the parts back where `physical` has them, at rest.
Json FvHost::Impl::physicsRestore(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->physics)
    return {{"ok", false}};
  fv::physics::PhysicsSettings s;
  s.on       = args.value("on", false) && figure->physics->valid();
  s.strength = std::clamp(args.value("strength", s.strength), 0.0, 1.0);
  s.gravity  = args.value("gravity", s.gravity);
  s.letFall  = args.value("letFall", s.letFall);
  if(const auto it = args.find("parts"); it != args.end() && it->is_object())
    for(auto g = it->begin(); g != it->end(); ++g)
      if(g.value().is_number())
        s.parts[g.key()] = std::clamp(g.value().get<double>(), 0.0, 1.0);
  figure->physics->set(s, false);
  if(s.on)
  {
    std::vector<std::pair<std::string, double>> values;
    if(const auto it = args.find("physical"); it != args.end() && it->is_object())
      for(auto v = it->begin(); v != it->end(); ++v)
        if(v.value().is_number())
          values.emplace_back(v.key(), v.value().get<double>());
    if(!values.empty())
      figure->physics->restorePhysical(values);
    // the pins as they were held
    std::vector<fv::physics::FigurePhysics::HeldPin> pins;
    for(const Json& j : args.value("pins", Json::array()))
    {
      const fv::pose::PoseBone* pb = j.is_object() ? figure->fig->pose->get(j.value("bone", "")) : nullptr;
      const Json                at = j.is_object() ? j.value("at", Json()) : Json(), turn = j.is_object() ? j.value("rotation", Json()) : Json();
      if(!pb || !at.is_array() || at.size() != 3 || !turn.is_array() || turn.size() != 4)
        continue;
      fv::physics::FigurePhysics::HeldPin h;
      h.poseBone = int(pb - figure->fig->pose->bones().data());
      h.turn     = j.value("turn", false);
      h.at       = {at[0].get<double>(), at[1].get<double>(), at[2].get<double>()};
      h.rotation = fv::Quat{turn[0].get<double>(), turn[1].get<double>(), turn[2].get<double>(), turn[3].get<double>()};
      h.maxForce = j.value("maxForce", 0.0);
      pins.push_back(h);
    }
    figure->physics->restorePins(pins);
  }
  publishPhysics(*figure);
  return {{"ok", true}};
}

// physics.pin {prop, bone, on?, rotation?}: in Physics mode, the part a bone is in (a finger: its
// hand) held where it is now, its turn too unless rotation is false; on false lets it go.
Json FvHost::Impl::physicsPin(const Json& args)
{
  auto figure = bodyArg(args, "a pin");
  if(!figure || !figure->physics || !figure->physics->settings().on)
    throw std::runtime_error("a pin holds a figure in Physics mode");
  const fv::pose::PoseBone* pb = figure->fig->pose->get(args.value("bone", ""));
  if(!pb)
    throw std::runtime_error("no bone " + args.value("bone", std::string()));
  const int  bone = int(pb - figure->fig->pose->bones().data());
  const bool on   = args.value("on", true);
  const bool done = on ? figure->physics->pin(bone, args.value("rotation", true)) : figure->physics->unpin(bone);
  if(on && !done)
    throw std::runtime_error(pb->label + " is no part of the body to pin");
  publishPhysics(*figure);
  return {{"ok", true}};
}

// The camera's ray through a render pixel, in the world (metres).
bool FvHost::Impl::cameraRay(const glm::vec2& px, glm::vec3& origin, glm::vec3& dir) const
{
  auto             manip = renderer->getCameraManipulator();
  const VkExtent2D size  = renderer->renderSize();
  if(!manip || !size.width || !size.height)
    return false;
  glm::dvec3 eye, center, up;
  manip->getLookat(eye, center, up);
  const glm::dvec3 f      = glm::normalize(center - eye);
  const glm::dvec3 r      = glm::normalize(glm::cross(f, up));
  const glm::dvec3 u      = glm::cross(r, f);
  const double     t      = std::tan(glm::radians(double(manip->getFov())) * 0.5);
  const double     aspect = double(size.width) / double(size.height);
  const double     sx     = (2.0 * px.x / size.width - 1) * t * aspect;
  const double     sy     = (1 - 2.0 * px.y / size.height) * t;
  origin                  = glm::vec3(eye);
  dir                     = glm::vec3(glm::normalize(f + r * sx + u * sy));
  return true;
}

// The Pull (view.set {pick: "pull"}): a press on a figure in Physics mode takes hold of the part
// under it, a move drags that point along the pointer at the depth it was taken at, and the release
// leaves the parts where they got to, as the channels' new pose: one step of the history.
bool FvHost::Impl::pullAt(const std::string& phase, const glm::vec2& px)
{
  glm::vec3 eye, dir;
  if(!cameraRay(px, eye, dir) || !renderer->feedsHostScene())
    return false;
  rs::RetainedScene& scene = renderer->hostScene();
  // the ray in a figure's root space (centimetres)
  const auto toRoot = [&](Loaded& f, fv::Vec3& o, fv::Vec3& d) {
    if(!f.feed)
      return false;
    const rs::EntityId root = f.feed->entityOf(f.fig->loaded.root);
    if(!scene.entity(root))
      return false;
    const glm::mat4 inv = glm::inverse(scene.world(root));
    const glm::vec3 a   = glm::vec3(inv * glm::vec4(eye, 1));
    const glm::vec3 b   = glm::normalize(glm::vec3(inv * glm::vec4(dir, 0)));
    o                   = {a.x, a.y, a.z};
    d                   = {b.x, b.y, b.z};
    return true;
  };
  if(phase == "down")
  {
    // the selected figure first
    std::vector<std::shared_ptr<Prop>> order;
    for(const auto& p : props)
      if(p->figure && p->figure->physics && p->figure->physics->settings().on)
      {
        if(p->id == view->selectedProp())
          order.insert(order.begin(), p);
        else
          order.push_back(p);
      }
    for(const auto& p : order)
    {
      fv::Vec3 o, d;
      double   depth = 0;
      if(!toRoot(*p->figure, o, d))
        continue;
      const Json before = physicsOps(*p->figure);
      if(!p->figure->physics->grab(o, d, depth))
        continue;
      pull = PullHold{p->id, depth, before};
      publishPhysics(*p->figure);
      return true;
    }
    return false;
  }
  const auto p = pull ? findProp(pull->prop) : nullptr;
  if(!p || !p->figure || !p->figure->physics)
  {
    pull.reset();
    return false;
  }
  Loaded&  f = *p->figure;
  fv::Vec3 o, d;
  if(toRoot(f, o, d))
    f.physics->drag(o + d * pull->depth);
  if(phase != "up")
    return true;
  // let go: the channels take the reached pose, so the motors hold it there
  const std::vector<fv::pose::ChannelValue> values = f.physics->release();
  f.fig->pose->setMany(values);
  std::vector<std::string> bones;
  for(const auto& v : values)
    if(const fv::pose::PoseBone* pb = f.fig->pose->get(v.bone); pb && std::find(bones.begin(), bones.end(), pb->id) == bones.end())
      bones.push_back(pb->id);
  patchPose(f, bones);
  const Json after = physicsOps(f);
  if(after != pull->before && !replaying)
  {
    journal.record({.redo = after, .undo = pull->before, .label = "Pull " + p->name, .key = std::string(), .at = std::chrono::steady_clock::now()});
    publishHistory();
  }
  pull.reset();
  publishPhysics(f);
  return true;
}
