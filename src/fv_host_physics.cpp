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

// Whether a figure's node shows: neither it nor a node above it hidden or taken off.
bool drawn(const fv::SceneGraph& scene, int node)
{
  for(int n = node; n >= 0; n = scene.nodes[size_t(n)].parent)
    if(!scene.nodes[size_t(n)].visible || scene.nodes[size_t(n)].removed)
      return false;
  return true;
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
// bones): made again once its pose or shape moved, indexed on the pool the first time a pass needs one.
const std::vector<std::shared_ptr<const fv::physics::SharedTriangles>>& FvHost::Impl::contactSets(Loaded& prop)
{
  if(prop.contactSetsAt == prop.contactRev)
    return prop.contactSets;
  prop.contactSetsAt = prop.contactRev;
  prop.contactSets.clear();
  const fv::SceneGraph& scene = prop.fig->loaded.scene;
  std::vector<int>      meshes;
  for(size_t n = 0; n < scene.nodes.size(); n++)
    if(scene.nodes[n].mesh >= 0 && drawn(scene, int(n)))
      meshes.push_back(scene.nodes[n].mesh);
  std::vector<std::vector<float>> posed = fv::physics::posedCages(scene, *prop.fig->morphs, meshes, prop.fig->loaded.root);
  for(size_t k = 0; k < meshes.size(); k++)
  {
    const auto& geo = scene.meshes[size_t(meshes[k])].geo;
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
// Another person is met from step 3 on, where each body takes its share of an overlap.
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
  for(const auto& p : props)
  {
    Loaded* prop = p->figure.get();
    if(!prop || prop == &f || prop->contact || !prop->feed || !shown(*p))
      continue;
    const rs::EntityId at = prop->feed->entityOf(prop->fig->loaded.root);
    if(!scene.entity(at))
      continue;
    const fv::Mat4 toBody = fromWorld * toMat4(scene.world(at));
    for(const auto& set : contactSets(*prop))
      out.push_back({set, toBody, 1.0});
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
  // a clip moves the props it poses every frame, and no pass runs while it plays: the first look after
  // it stops sees what moved
  if(renderer->editAnimationControl().play)
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  std::unordered_map<std::string, ContactSeen> now;
  for(const auto& p : props)
  {
    if(Loaded* prop = p->figure.get(); prop && !prop->contact && prop->feed)
    {
      const rs::EntityId at = prop->feed->entityOf(prop->fig->loaded.root);
      if(!scene.entity(at))
        continue;
      ContactSeen& seen    = now[p->id];
      seen.shown           = shown(*p);
      seen.rev             = prop->contactRev;
      const glm::mat4 world = scene.world(at);
      seen.worlds.push_back(world);
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
  std::vector<fv::physics::Box> changed;
  bool                          cameOrWent = now.size() != contactSeen.size();
  for(const auto& [id, s] : now)
  {
    const auto was = contactSeen.find(id);
    if(was != contactSeen.end() && was->second.worlds == s.worlds && was->second.shown == s.shown && was->second.rev == s.rev)
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

//--------------------------------------------------------------------------------------------------
// Physics mode (figure/physics/rig/figure_physics.hpp, docs/physics-plan.md step 2): the frame's
// step, the Pull, physics.set and physics.restore, the physics/<prop> topic.

// A figure's Physics mode as the scene file and the history keep it: the settings and, in Physics
// mode, where its parts are.
Json physicsState(const Loaded& f)
{
  const fv::physics::PhysicsSettings& s        = f.physics->settings();
  Json                                physical = Json::object();
  if(s.on)
    for(const auto& [key, value] : f.physics->physicalValues())
      physical[key] = value;
  return {{"on", s.on}, {"strength", s.strength}, {"gravity", s.gravity}, {"letFall", s.letFall}, {"physical", physical}};
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

void FvHost::Impl::publishPhysics(Loaded& f)
{
  if(!f.physics)
    return;
  f.physicsVersion                      = f.physics->version();
  const fv::physics::PhysicsSettings& s = f.physics->settings();
  protocol->events().snapshot("physics/" + f.id, {{"available", f.physics->valid()},
                                                  {"on", s.on},
                                                  {"awake", f.physics->awake()},
                                                  {"pulling", f.physics->pulling()},
                                                  {"strength", s.strength},
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
  for(const auto& p : props)
  {
    if(!p->figure || !p->figure->physics)
      continue;
    Loaded& f = *p->figure;
    if(f.physics->settings().on && renderer->feedsHostScene() && f.feed)
    {
      rs::RetainedScene& scene = renderer->hostScene();
      const rs::EntityId root  = f.feed->entityOf(f.fig->loaded.root);
      if(scene.entity(root))
      {
        const glm::mat4 toRoot = glm::inverse(scene.world(root));
        const glm::vec3 down   = glm::normalize(glm::vec3(toRoot * glm::vec4(0, -1, 0, 0)));
        f.physics->setGravity({down.x * 980.0, down.y * 980.0, down.z * 980.0});
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
    {
      fv::timing::Scope t("physics: frame", 2);
      f.physics->step(elapsed);
    }
    if(f.physicsVersion != f.physics->version())
      publishPhysics(f);
  }
}

// physics.set {prop, on?, strength?, gravity?, letFall?, gesture?}: Physics mode on or off (leaving
// keeps the pose it reached in the channels), Hold (0 limp to 1 rigid), gravity, and whether the hip
// is let fall.
Json FvHost::Impl::physicsSet(const Json& args)
{
  auto figure = figureArg(args);
  if(!figure || !figure->physics)
    throw std::runtime_error("no figure with a body is selected");
  fv::physics::PhysicsSettings s = figure->physics->settings();
  if(args.contains("on") && args["on"].is_boolean())
    s.on = args["on"].get<bool>();
  if(args.contains("strength") && args["strength"].is_number())
    s.strength = std::clamp(args["strength"].get<double>(), 0.0, 1.0);
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
  }
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
