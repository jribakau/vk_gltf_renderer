#include <figure/dforce/dforce.hpp>

#include <algorithm>
#include <cmath>

#include <figure/dforce/skin.hpp>
#include <figure/dforce/solver.hpp>
#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/coarse.hpp>

namespace fv::dforce {

namespace {

using namespace std::chrono_literals;

constexpr auto  kDebounce = 400ms;
constexpr float kAir      = 0.15f;
// Velocity decay per second a live step adds once the pose has stood still for a while, so the cloth
// comes to rest; until then it falls and swings freely. Seconds: free for this long, then this long
// to reach it, within the hold.
constexpr float kLiveRestDamping = 6.f;
constexpr float kLiveFree = 1.f, kLiveCalming = 1.f, kLiveHold = 3.f;
// How much of a frame a settle on the GPU may take, seconds: the app stays usable while it runs.
constexpr double kSettleSlice = 0.05;
// A live step is as long as the time since the last one, within these, seconds: a slow frame doesn't
// make slow motion, and frames faster than the shortest wait for each other. Longer than the longest,
// the GPU's simulation is carried with the pose through the rest (ClothBackend::carry), the CPU's slows.
// A step's collision passes grow with its length, and so does each one's grid: past about 45 ms a step
// on the GPU beside the path tracer took longer than it simulated, so the next was longer still (a walk
// held at the 1/20 s this was, half its steps late); below it they shorten, to a frame or two.
constexpr float kLiveMinStep = 1.f / 60.f, kLiveMaxStep = 1.f / 30.f;
// A landed step's drape is eased in over as long as the step took, but no longer than this: a step
// that came late (a rebuild, a slow frame) would leave the cloth that far behind
constexpr float kLiveEaseMost = 0.1f;
// How far gravity must turn to matter, cm/s^2 on any axis: about a twentieth of a degree.
constexpr float kGravityChange = 1.f;
// A collider much finer than the cloth meets it as a coarse copy (physics/cloth/coarse.hpp), with
// faces about as wide as the cloth's and at most this share of their width off its surface, but of no
// more than this width (cm): the cloth lies that much further off, and a light copy's faces are wide
constexpr float kCoarseError = 0.25f, kCoarseErrorWidth = 1.f;

// Whether the node of every mesh is shown: not taken off, not hidden, nor anything above it.
std::vector<bool> shownNodes(const SceneGraph& scene)
{
  std::vector<bool> shown(scene.nodes.size(), true);
  // parents are added before children
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    const Node& n = scene.nodes[i];
    shown[i]      = n.visible && !n.removed && (n.parent < 0 || shown[size_t(n.parent)]);
  }
  return shown;
}

bool sameGravity(const std::array<float, 3>& a, const std::array<float, 3>& b)
{
  for(int k = 0; k < 3; k++)
    if(std::fabs(a[size_t(k)] - b[size_t(k)]) > kGravityChange)
      return false;
  return true;
}

// What a coarse copy is made from: its rest cage, its faces and which collide, its strength and what is
// in the simulation, and the edge and error it is made for (FNV-1a over their bytes).
uint64_t shapeKey(const SettleMesh& s, float edge, float maxError)
{
  uint64_t   h   = 1469598103934665603ull;
  const auto mix = [&](const void* data, size_t bytes) {
    const auto* b = static_cast<const unsigned char*>(data);
    for(size_t i = 0; i < bytes; i++)
    {
      h ^= b[i];
      h *= 1099511628211ull;
    }
  };
  mix(s.rest.data(), s.rest.size() * sizeof(float));
  mix(s.tris.data(), s.tris.size() * sizeof(uint32_t));
  mix(s.collide.data(), s.collide.size());
  mix(s.strength.data(), s.strength.size() * sizeof(float));
  mix(s.visible.data(), s.visible.size());
  mix(&edge, sizeof edge);
  mix(&maxError, sizeof maxError);
  return h;
}

bool samePose(const std::vector<Mat4>& a, const std::vector<Mat4>& b)
{
  if(a.size() != b.size())
    return false;
  for(size_t k = 0; k < a.size(); k++)
    for(size_t c = 0; c < 16; c++)
      if(std::fabs(a[k].e[c] - b[k].e[c]) > 1e-6)
        return false;
  return true;
}

}  // namespace

std::string channelOf(int mesh)
{
  return "dforce:" + std::to_string(mesh);
}

// What a settle needs and where each result goes, taken on the application thread.
struct Dforce::Job
{
  std::vector<SettleMesh> bodies;
  // the mesh each body is, for reading its bones again; -1 when it has no skin
  std::vector<int> slotMesh;
  // a coarse copy's vertices of its mesh's cage, per slot (physics/cloth/coarse.hpp); empty: all of them
  std::vector<std::vector<uint32_t>> slotKeep;
  // the body's slot, what a light copy's cloth is kept out of; SIZE_MAX: none
  size_t bodySlot = SIZE_MAX;
  struct Drape
  {
    int                mesh = -1;
    size_t             slot = 0;  // into bodies and the settle's result
    std::vector<float> rest;
    CageSkin           skin;
    // the pose the positions are read back through: the job's, or the live step's
    SkinPose pose;
    Mat4     back;  // job space -> mesh-local
    // its light copy, simulated in its place in `bodies` (null: the cloth itself)
    std::shared_ptr<const physics::cloth::CoarseCloth> light;
  };
  std::vector<Drape> drapes;
  // strand hair that falls, which collides with `bodies` and is no part of their settle
  struct Hair
  {
    int                mesh = -1;
    StrandSet          strands;
    std::vector<float> rest;
    CageSkin           skin;
    SkinPose           pose;
    Mat4               back;  // job space -> mesh-local
  };
  std::vector<Hair> hairs;
};

Dforce::Dforce(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root)
    : m_scene(scene)
    , m_morphs(morphs)
    , m_bodyMesh(bodyMesh)
    , m_root(root)
{
  m_settleGravity = currentGravity();
}

Dforce::~Dforce()
{
  // the pool keeps the snapshot it was given; only wait so a late result cannot outlive the owner. Out
  // of date first, so it ends within a frame: a GPU settle on the pool mustn't run on past its device
  if(m_jobStale)
    m_jobStale->store(true);
  if(m_job.valid())
    m_job.wait();
  for(LiveRun& run : m_liveRuns)
    if(run.drapes.valid())
      run.drapes.wait();
  if(m_hairStale)
    m_hairStale->store(true);
  if(m_hairJob.valid())
    m_hairJob.wait();
}

bool Dforce::settlingOnGpu() const
{
  return m_settling || (m_jobOnGpu && m_job.valid());
}

std::vector<int> Dforce::dynamics() const
{
  std::vector<int> out;
  const auto       shown = shownNodes(m_scene);
  for(size_t n = 0; n < m_scene.nodes.size(); n++)
  {
    const Node& node = m_scene.nodes[n];
    if(node.kind != NodeKind::eMesh || node.mesh < 0 || !shown[n])
      continue;
    const Mesh& m = m_scene.meshes[size_t(node.mesh)];
    // once, though DAZ Studio instances may draw it from more nodes
    if(m.dforce && m.dforce->simulated() && node.mesh != m_bodyMesh && std::find(out.begin(), out.end(), node.mesh) == out.end())
      out.push_back(node.mesh);
  }
  return out;
}

bool Dforce::active() const
{
  return !dynamics().empty();
}

bool Dforce::busy() const
{
  return m_scheduled || m_job.valid() || m_hairJob.valid() || m_settling || m_liveSettling;
}

// What has stopped simulating (its switch off, frozen, hidden) goes back to the shape it conforms to:
// no settle will come for it, and with nothing else draping none comes at all.
void Dforce::dropStale()
{
  const std::vector<int>                      live = dynamics();
  std::vector<std::pair<std::string, double>> names;
  for(auto it = m_draped.begin(); it != m_draped.end();)
  {
    if(std::find(live.begin(), live.end(), *it) != live.end())
    {
      ++it;
      continue;
    }
    if(m_morphs.has(channelOf(*it)))
      names.emplace_back(channelOf(*it), 0.0);
    it = m_draped.erase(it);
    m_version++;
  }
  if(!names.empty())
    m_morphs.setMany(names);
}

void Dforce::poke()
{
  dropStale();
  if(!m_enabled || !active())
    return;
  if(m_mode == Mode::eSettle && playing && playing())
  {
    m_skippedPlaying = true;
    return;
  }
  m_gen++;
  if(m_jobStale)
    m_jobStale->store(true);
  if(m_hairStale)
    m_hairStale->store(true);
  m_scheduled = true;
  m_due       = std::chrono::steady_clock::now() + kDebounce;
  m_version++;
}

void Dforce::changed(const std::vector<int>& meshes, bool nodesMoved)
{
  if(m_mode != Mode::eSettle)
  {
    // Bones are read again next frame. A mesh changed with them is the pose's correctives (under a clip,
    // every frame), and a collider's change is its pose too (a dent, a dial on the body): the next step
    // skins from the new rest shape in place. A cloth's own shape changed with nothing moving is a dial:
    // the simulation is built again once it stops. The pose never puts off a build that is due (it
    // reads the shapes as they are then), so one starts while a clip plays
    bool rebuild = false;
    for(int mesh : meshes)
    {
      const bool slot  = m_live && m_liveJob && mesh >= 0
                        && std::find(m_liveJob->slotMesh.begin(), m_liveJob->slotMesh.end(), mesh) != m_liveJob->slotMesh.end();
      const bool drape = slot && std::any_of(m_liveJob->drapes.begin(), m_liveJob->drapes.end(), [&](const Job::Drape& d) { return d.mesh == mesh; });
      if(slot && (nodesMoved || !drape))
        m_liveReshaped.insert(mesh);
      else if(!nodesMoved)
        rebuild = true;
    }
    if(rebuild)
      poke();
    return;
  }
  if(nodesMoved || !meshes.empty())
    poke();
}

void Dforce::setPlacement(const Mat4& placed)
{
  m_placed = placed;
  if(m_skippedPlaying && !(playing && playing()))
  {
    // what changed while a clip played: the drape catches up now it has stopped
    m_skippedPlaying = false;
    if(m_mode == Mode::eSettle)
      poke();
  }
  // live reads it next frame. Against what the last settle was asked for, so a slow turn adds up; a
  // turn while a clip plays waits for it to stop, as the drape does
  if(m_mode != Mode::eSettle || (playing && playing()))
    return;
  const std::array<float, 3> now = currentGravity();
  if(!sameGravity(now, m_settleGravity))
  {
    m_settleGravity = now;
    poke();
  }
}

void Dforce::resimulate()
{
  m_enabled   = true;
  m_scheduled = true;
  m_due       = std::chrono::steady_clock::now();
  m_gen++;
  if(m_jobStale)
    m_jobStale->store(true);
  if(m_hairStale)
    m_hairStale->store(true);
  m_version++;
}

void Dforce::cancel()
{
  m_gen++;
  if(m_jobStale)
    m_jobStale->store(true);
  if(m_hairStale)
    m_hairStale->store(true);
  if(m_scheduled || m_job.valid() || m_hairJob.valid() || m_settling || m_live)
    m_version++;
  m_scheduled = false;
  m_settling.reset();
  // a settle on the pool keeps its own hold on it, and lets it go when it ends
  m_gpuSim.reset();
  // a live step in flight is dropped when it lands; the simulation goes
  m_live.reset();
  m_liveJob.reset();
  m_liveSettling = false;
  m_easeFor      = 0;
  m_shown.clear();
}

void Dforce::setEnabled(bool on)
{
  m_enabled = on;
  std::vector<std::pair<std::string, double>> names;
  for(auto it = m_draped.begin(); it != m_draped.end();)
  {
    if(!m_morphs.has(channelOf(*it)))
    {
      it = m_draped.erase(it);
      continue;
    }
    names.emplace_back(channelOf(*it), on ? 1.0 : 0.0);
    ++it;
  }
  if(!names.empty())
    m_morphs.setMany(names);
  if(on)
    poke();
  else
    cancel();
  m_version++;
}

void Dforce::setMode(Mode mode)
{
  if(mode == m_mode)
    return;
  m_mode = mode;
  cancel();
  m_version++;
  if(m_enabled)
    resimulate();
}

void Dforce::setLight(bool on)
{
  if(on == lightCloth)
    return;
  lightCloth = on;
  // what is draped was made the other way: made again, as a mode switch makes it
  cancel();
  m_version++;
  if(m_enabled)
    resimulate();
}

std::shared_ptr<Dforce::Job> Dforce::prepare(uint64_t) const
{
  const std::vector<int> dyn = dynamics();
  if(dyn.empty())
    return nullptr;
  auto       job    = std::make_shared<Job>();
  const auto worlds = m_scene.worlds();
  const auto nodeOf = nodeOfMesh(m_scene);

  struct Skinned
  {
    std::vector<float> rest, posed;
    CageSkin           skin;
    SkinPose           pose;
    bool               skinned = false;
  };
  // The skin as the solver takes it, so the pose transition turns the mesh's bones rather than
  // sliding its vertices along chords: the bones' skin matrices in the mesh's space, the rest cage
  // there, and the matrix into the job's space.
  const auto attachSkin = [](SettleMesh& out, const Skinned& s, const Mat4& to) {
    if(!s.skinned)
      return;
    out.skinIndex  = s.skin.index;
    out.skinWeight = s.skin.weight;
    out.bones      = skinMatrices(s.pose);
    out.restLocal  = s.rest;
    out.toSpace    = to;
  };
  // The cage's rest shape and its skinned pose, both in the mesh's local centimetres.
  const auto skinCage = [&](int mesh, const mesh::RenderTarget* except) -> std::optional<Skinned> {
    const Mesh& m = m_scene.meshes[size_t(mesh)];
    if(!m.geo || m.geo->cagePositions.empty())
      return std::nullopt;
    Skinned s;
    s.rest = m_morphs.morphedCage(mesh, except);
    if(!m.skinned() || m.cageSkin.start.size() != s.rest.size() / 3 + 1)
    {
      const size_t n = s.rest.size() / 3;
      s.skin.index.assign(n * 4, 0);
      s.skin.weight.assign(n * 4, 0.f);
      for(size_t v = 0; v < n; v++)
        s.skin.weight[v * 4] = 1;
      s.pose.bones = {Mat4{}};
      s.posed      = s.rest;
      return s;
    }
    s.skinned = true;
    s.skin    = cageSkin(m.cageSkin);
    s.pose    = currentSkin(m_scene, worlds, nodeOf, mesh);
    s.posed   = skinLocal(s.rest, s.skin, s.pose);
    return s;
  };
  // The job's space is the figure root's, in centimetres as the solver's constants are, whatever
  // scale and placement the root has in the world (the host draws in metres).
  const Mat4 rootWorld = m_root >= 0 && size_t(m_root) < worlds.size() ? worlds[size_t(m_root)] : Mat4{};
  const Mat4 fromWorld = inverse(rootWorld);
  const auto toSpace   = [&](int mesh) { return fromWorld * (nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : Mat4{}); };

  const std::array<float, 3> gravity = currentGravity();

  // the body is always a collider, in the same space as the cloth, and under every cloth layer
  size_t bodySlot = SIZE_MAX;
  if(m_bodyMesh >= 0 && size_t(m_bodyMesh) < m_scene.meshes.size())
    if(auto body = skinCage(m_bodyMesh, nullptr))
    {
      bodySlot      = job->bodies.size();
      const Mat4 to = toSpace(m_bodyMesh);
      SettleMesh collider = SettleMesh::collider(transformPoints(to, body->rest), transformPoints(to, body->posed),
                                                 m_scene.meshes[size_t(m_bodyMesh)].geo->dsonTris, kBodyLayer, 0.2f, 0.4f);
      attachSkin(collider, *body, to);
      job->bodies.push_back(std::move(collider));
      job->slotMesh.push_back(body->skinned ? m_bodyMesh : -1);
    }

  const auto shown = shownNodes(m_scene);
  const auto consider = [&](int mesh) {
    const Mesh& m = m_scene.meshes[size_t(mesh)];
    if(mesh == m_bodyMesh || !m.geo || nodeOf[size_t(mesh)] < 0 || !shown[size_t(nodeOf[size_t(mesh)])])
      return;
    if(m.dforce && m.dforce->simulatesStrands() && m.dforce->simulated())
    {
      const dson::DforceData& data = *m.dforce;
      const auto own     = m_morphs.cpuTarget(channelOf(mesh), mesh);
      auto       skinned = skinCage(mesh, own.get());
      if(!skinned)
        return;
      const Mat4 to = toSpace(mesh);
      Job::Hair  h;
      h.mesh               = mesh;
      h.strands.posed      = transformPoints(to, skinned->posed);
      h.strands.start      = data.strandStart;
      h.strands.verts      = data.strandVerts;
      h.strands.strength   = data.strandStrength;
      h.strands.offset     = data.strandOffset;
      h.strands.friction   = data.strandFriction;
      h.strands.shapeRoot  = data.strandShapeRoot;
      h.strands.shapeTip   = data.strandShapeTip;
      h.strands.collide    = data.strandCollide;
      h.strands.volume     = data.strandVolume;
      h.strands.styled     = data.strandStyled;
      h.strands.gravity    = gravity;
      h.rest               = std::move(skinned->rest);
      h.skin               = std::move(skinned->skin);
      h.pose               = std::move(skinned->pose);
      h.back               = inverse(to);
      job->hairs.push_back(std::move(h));
      return;
    }
    if(!m.dforce || !m.dforce->hasPolygons())
    {
      // no simulation settings of its own (a belt, a bra, shoes): a collider at the surface
      // default layer, as DAZ collides with every visible object, when the host says it is one
      if(collidesWith && !collidesWith(mesh))
        return;
      if(m.geo->cagePositions.empty() || m.geo->dsonTris.empty())
        return;
      auto skinned = skinCage(mesh, nullptr);
      if(!skinned)
        return;
      const Mat4 to       = toSpace(mesh);
      SettleMesh collider = SettleMesh::collider(transformPoints(to, skinned->rest), transformPoints(to, skinned->posed), m.geo->dsonTris,
                                                 kPlainLayer, 0.2f, 0.4f);
      attachSkin(collider, *skinned, to);
      job->bodies.push_back(std::move(collider));
      job->slotMesh.push_back(skinned->skinned ? mesh : -1);
      return;
    }
    const dson::DforceData& data = *m.dforce;
    if(data.strength.size() * 3 != m.geo->cagePositions.size())
      return;
    const auto own      = m_morphs.cpuTarget(channelOf(mesh), mesh);
    auto       skinned  = skinCage(mesh, data.simulated() ? own.get() : nullptr);
    if(!skinned)
      return;
    const Mat4 to   = toSpace(mesh);
    auto       rest = transformPoints(to, skinned->rest);
    auto       posed = transformPoints(to, skinned->posed);
    if(data.simulated())
    {
      SettleMesh s;
      s.rest   = std::move(rest);
      s.posed  = std::move(posed);
      s.tris   = m.geo->dsonTris;
      s.polyStart = data.polyStart;
      s.polyVerts = data.polyVerts;
      s.kind   = data.kind == dson::DforceKind::eAddon ? SettleMesh::Kind::eAddon : SettleMesh::Kind::eDynamic;
      s.strength = data.strength;
      s.stretch = data.stretch;
      s.shear = data.shear;
      s.bend = data.bend;
      s.buckling = data.buckling;
      s.bucklingRatio = data.bucklingRatio;
      s.density = data.density;
      s.contraction = data.contraction;
      s.damping = data.damping;
      s.friction = data.friction;
      s.offset = data.offset;
      s.layer = data.layer;
      s.selfCollide = data.selfCollide;
      s.collide = data.surfaceCollide;
      s.visible = data.surfaceVisible;
      s.gravity = gravity;
      attachSkin(s, *skinned, to);
      Job::Drape d;
      d.mesh = mesh;
      d.slot = job->bodies.size();
      d.rest = std::move(skinned->rest);
      d.skin = std::move(skinned->skin);
      d.pose = std::move(skinned->pose);
      d.back = inverse(to);
      job->bodies.push_back(std::move(s));
      job->slotMesh.push_back(skinned->skinned ? mesh : -1);
      job->drapes.push_back(std::move(d));
    }
    else if(data.visible)
    {
      SettleMesh collider = SettleMesh::collider(std::move(rest), std::move(posed),
                                                 data.kind == dson::DforceKind::eAddon ? std::vector<uint32_t>{} : m.geo->dsonTris,
                                                 data.layer.empty() ? 0 : data.layer[0], data.offset.empty() ? 0.2f : data.offset[0],
                                                 data.friction.empty() ? 0.4f : data.friction[0], data.surfaceCollide);
      attachSkin(collider, *skinned, to);
      job->bodies.push_back(std::move(collider));
      job->slotMesh.push_back(skinned->skinned ? mesh : -1);
    }
  };
  for(int mesh = 0; mesh < int(m_scene.meshes.size()); mesh++)
    consider(mesh);
  if(job->drapes.empty() && job->hairs.empty())
    return nullptr;

  job->slotKeep.resize(job->bodies.size());
  job->bodySlot = bodySlot;
  // Light cloth: a cloth much finer than the faces asked for simulates as a coarse copy of itself, and
  // its own vertices ride on it. Before the colliders, which are then copied as fine as the copy
  std::map<uint64_t, std::shared_ptr<const physics::cloth::CoarseCloth>> lights;
  if(lightCloth && lightClothEdge > 0)
  {
    timing::Scope t("dforce: light cloth", 1);
    for(Job::Drape& d : job->drapes)
    {
      SettleMesh&                                        s   = job->bodies[d.slot];
      const uint64_t                                     key = shapeKey(s, lightClothEdge, 0);
      std::shared_ptr<const physics::cloth::CoarseCloth> c;
      if(const auto made = m_light.find(key); made != m_light.end())
      {
        // made for this shape before: the same copy, of the cloth as it is posed now
        if(made->second)
          if(auto same = physics::cloth::sameCloth(s, *made->second))
            c = std::make_shared<const physics::cloth::CoarseCloth>(std::move(*same));
        lights[key] = made->second;
      }
      else
      {
        const size_t had = s.vertexCount();
        if(auto fresh = physics::cloth::coarseCloth(s, lightClothEdge))
          c = std::make_shared<const physics::cloth::CoarseCloth>(std::move(*fresh));
        lights[key] = c;
        if(c && timing::enabled())
          timing::line("dforce: " + m_scene.meshes[size_t(d.mesh)].name + " simulates as a light copy of " + std::to_string(c->keep.size())
                       + " of its " + std::to_string(had) + " vertices, faces " + std::to_string(lightClothEdge) + " cm wide");
      }
      if(!c)
        continue;
      s                     = c->surface;
      job->slotKeep[d.slot] = c->keep;
      d.light               = std::move(c);
    }
  }
  m_light = std::move(lights);

  // What's worn much finer than the cloth (a necklace's 150k triangles of chain under a neckline, lace)
  // is met as a coarse copy, as fine as the cloth; not the body, the cage already, where a gap shows most
  std::map<uint64_t, std::optional<physics::cloth::Coarse>> copies;
  if(coarseColliders && !job->drapes.empty())
  {
    timing::Scope t("dforce: coarse colliders", 1);
    double        sum = 0;
    size_t        faces = 0;
    for(const Job::Drape& d : job->drapes)
    {
      const SettleMesh& s = job->bodies[d.slot];
      sum += double(physics::cloth::meanEdge(s)) * double(s.tris.size() / 3);
      faces += s.tris.size() / 3;
    }
    const float edge = faces ? float(sum / double(faces)) : 0.f;
    for(size_t slot = 0; slot < job->bodies.size(); slot++)
    {
      if(slot == bodySlot)
        continue;
      const size_t   had = job->bodies[slot].tris.size() / 3;
      const float    err = kCoarseError * std::min(edge, kCoarseErrorWidth);
      const uint64_t key = shapeKey(job->bodies[slot], edge, err);
      std::optional<physics::cloth::Coarse> c;
      if(const auto made = m_coarse.find(key); made != m_coarse.end())
      {
        // made for this shape before: the same copy, of the collider as it is posed now
        if(made->second)
          c = physics::cloth::sameCopy(job->bodies[slot], *made->second);
        copies[key] = made->second;
        if(!c)
          continue;
      }
      else
      {
        c           = physics::cloth::coarsen(job->bodies[slot], edge, err);
        copies[key] = c;
        if(!c)
          continue;
        if(timing::enabled())
        {
          const int   mesh = job->slotMesh[slot];
          std::string name = mesh >= 0 ? m_scene.meshes[size_t(mesh)].name : std::to_string(slot);
          timing::line("dforce: " + name + " collides as " + std::to_string(c->surface.tris.size() / 3) + " of its "
                       + std::to_string(had) + " faces, " + std::to_string(c->error) + " cm off (the cloth's edges "
                       + std::to_string(edge) + " cm)");
        }
      }
      job->bodies[slot]   = std::move(c->surface);
      job->slotKeep[slot] = std::move(c->keep);
    }
  }
  // only what this job has is kept: a shape changed or taken off goes
  m_coarse = std::move(copies);
  return job;
}

std::optional<Dforce::Prepared> Dforce::prepared() const
{
  const auto job = prepare(m_gen);
  if(!job)
    return std::nullopt;
  Prepared p;
  p.bodies   = job->bodies;
  p.slotMesh = job->slotMesh;
  for(const Job::Drape& d : job->drapes)
  {
    p.drapes.emplace_back(d.mesh, int(d.slot));
    p.light.push_back(d.light);
  }
  for(const Job::Hair& h : job->hairs)
    p.hairs.emplace_back(h.mesh, h.strands);
  return p;
}

std::array<float, 3> Dforce::currentGravity() const
{
  const auto worlds    = m_scene.worlds();
  const Mat4 rootWorld = m_root >= 0 && size_t(m_root) < worlds.size() ? worlds[size_t(m_root)] : Mat4{};
  // world down in the job's space, through all the placement has: what hangs plumb as drawn under a
  // scale that isn't uniform, or a mirror, is not what a rotation alone gives
  const Mat4   fromWorld = inverse(m_placed * rootWorld);
  const double x = -fromWorld.e[4], y = -fromWorld.e[5], z = -fromWorld.e[6];
  const double scale = 980.0 / std::max(1e-12, std::sqrt(x * x + y * y + z * z));
  return {float(x * scale), float(y * scale), float(z * scale)};
}

std::vector<std::vector<Mat4>> Dforce::currentBones(const Job& job) const
{
  std::vector<std::vector<Mat4>> out(job.slotMesh.size());
  const auto                     worlds = m_scene.worlds();
  const auto                     nodeOf = nodeOfMesh(m_scene);
  for(size_t slot = 0; slot < job.slotMesh.size(); slot++)
  {
    const int mesh = job.slotMesh[slot];
    if(mesh < 0 || size_t(mesh) >= m_scene.meshes.size() || !m_scene.meshes[size_t(mesh)].skinned())
      continue;
    out[slot] = skinMatrices(currentSkin(m_scene, worlds, nodeOf, mesh));
  }
  return out;
}

bool Dforce::pump()
{
  if(m_mode != Mode::eSettle)
    return pumpLive();
  bool wrote = false;
  if(m_job.valid() && m_job.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    std::vector<std::vector<float>> positions;
    try
    {
      positions = m_job.get();
    }
    catch(const std::exception&)
    {
      positions.clear();
    }
    const uint64_t gen = m_jobGen;
    auto           job = std::move(m_snapshot);
    m_version++;
    // one that failed isn't kept: the next settle makes another
    if(m_gpuSim && m_gpuSim->failed())
      m_gpuSim.reset();
    if(gen == m_gen && m_enabled && job && !positions.empty())
      wrote = apply(*job, positions);
  }
  if(m_hairJob.valid() && m_hairJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    std::vector<std::vector<float>> positions;
    try
    {
      positions = m_hairJob.get();
    }
    catch(const std::exception&)
    {
      positions.clear();
    }
    const auto job = std::move(m_hairSnapshot);
    m_hairSnapshot.reset();
    m_version++;
    if(m_hairGen == m_gen && m_enabled && job && positions.size() == job->hairs.size())
    {
      timing::Scope t("dforce: hair lands");
      // an item taken off while this ran is not draped
      const std::vector<int> live = dynamics();
      for(size_t i = 0; i < positions.size(); i++)
      {
        const Job::Hair& h = job->hairs[i];
        if(std::find(live.begin(), live.end(), h.mesh) == live.end())
          continue;
        wrote = writeDrape(h.mesh, drapeDelta(h.rest, transformPoints(h.back, positions[i]), h.skin, h.pose)) || wrote;
      }
    }
  }
  // A settle on the pool: on `sim` when there is one (the GPU solver on a queue of its own), else on the
  // CPU, which also takes over when `sim` fails. An edit since it began ends it within a frame (`stale`):
  // the new settle needn't wait for what would be dropped
  const auto onThePool = [this](std::shared_ptr<Job> job, std::shared_ptr<std::atomic<bool>> stale, std::shared_ptr<ClothBackend> sim) {
    m_jobStale = stale;
    m_jobOnGpu = sim != nullptr;
    m_job      = jobs::pool().submit([job, stale, sim]() mutable {
      if(sim)
      {
        sim->settle();
        const bool                      landed = !stale->load() && !sim->failed();
        std::vector<std::vector<float>> out;
        if(landed)
          out = sim->positions();
        // the driver keeps it for the next settle; if it has let it go meanwhile (dForce turned off), its
        // device memory goes here, before the result is seen: whoever waits for it may then take the
        // device down
        sim.reset();
        if(landed || stale->load())
          return out;
      }
      SettleOptions opts;
      opts.air       = kAir;
      opts.cancelled = [stale] { return stale->load(); };
      return settle(job->bodies, opts);
    });
  };
  if(m_settling)
  {
    // an edit since it began: what it has is of the old pose, and the new settle needn't wait for it
    if(m_jobGen != m_gen || !m_enabled)
    {
      m_settling.reset();
      m_snapshot.reset();
      m_version++;
    }
    else if(m_settling->settleSome(kSettleSlice))
    {
      const auto sim = std::move(m_settling);
      m_settling.reset();
      m_version++;
      if(sim->failed())
        onThePool(m_snapshot, std::make_shared<std::atomic<bool>>(false), nullptr);
      else
      {
        wrote = apply(*m_snapshot, sim->positions()) || wrote;
        m_snapshot.reset();
      }
    }
  }
  if(m_scheduled && !m_job.valid() && !m_hairJob.valid() && !m_settling && std::chrono::steady_clock::now() >= m_due && !(waitWhile && waitWhile()))
  {
    m_scheduled = false;
    if(m_enabled && active())
    {
      std::shared_ptr<Job> job;
      {
        timing::Scope t("dforce: prepare");
        job = prepare(m_gen);
      }
      if(job)
      {
        m_settleGravity = currentGravity();
        if(job->drapes.empty())
          m_gpuSim.reset();
        if(!job->drapes.empty())
        {
          m_snapshot = job;
          m_jobGen   = m_gen;
          // the GPU solver when there is one: in slices here when it must run on this thread, else on the
          // pool as the CPU's settle runs, and this thread never waits on it
          auto          stale = std::make_shared<std::atomic<bool>>(false);
          SettleOptions opts;
          opts.air       = kAir;
          opts.cancelled = [stale] { return stale->load(); };
          if(makeGpuSim)
          {
            // the one the last settle ran on, for the same garments and shapes, keeps its model and buffers
            if(!m_gpuSim || !m_gpuSim->restart(job->bodies, opts))
              m_gpuSim = makeGpuSim(job->bodies, opts);
            m_settling = m_gpuSim;
          }
          if(!m_settling || !m_settling->ownerThreadOnly())
            onThePool(job, stale, std::exchange(m_settling, nullptr));
        }
        if(!job->hairs.empty())
        {
          m_hairSnapshot = job;
          m_hairGen      = m_gen;
          m_hairStale    = std::make_shared<std::atomic<bool>>(false);
          m_hairJob      = jobs::pool().submit([job, stale = m_hairStale] {
            timing::Scope t("dforce: hair settle");
            StrandOptions opts;
            opts.cancelled = [stale] { return stale->load(); };
            std::vector<std::vector<float>> out;
            for(const Job::Hair& h : job->hairs)
              out.push_back(settleStrands(h.strands, job->bodies, opts));
            return out;
          });
        }
        m_version++;
      }
    }
  }
  // a newer edit that arrives while one runs on the pool waits for it: that result is dropped above, then this starts
  return wrote;
}

// Live: the runs whose drapes are made are written, or eased in, in the order they began; then, once the
// simulation is free (the drapes of the last step may still be on their way), unless it is to be built
// again or nothing has moved, the figure's bones go to it and it steps once more.
bool Dforce::pumpLive()
{
  const auto frameAt = clock ? clock() : std::chrono::steady_clock::now();
  // the last step's drapes a frame further in, every frame, whether or not the next has landed
  bool       wrote   = easeLive(frameAt);
  // the simulation is free: nothing is stepping (the drapes of the last step may still be on their way)
  const bool free = m_liveRuns.empty() || m_liveRuns.back().stepped.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  // Rest shapes the pose or a dent moved since the last step, read as the figure's pump left them, while
  // the simulation is between steps: what it skins from next, and what a drape's deltas are taken
  // against (a run in flight took its own as it began, and is shifted by the change when it lands)
  if(free && m_live && m_liveJob && !m_liveReshaped.empty())
  {
    timing::Accum t("dforce: live, the rest shapes read");
    for(size_t slot = 0; slot < m_liveJob->slotMesh.size(); slot++)
    {
      const int mesh = m_liveJob->slotMesh[slot];
      if(mesh < 0 || !m_liveReshaped.count(mesh))
        continue;
      const auto         drape = std::find_if(m_liveJob->drapes.begin(), m_liveJob->drapes.end(), [&](const Job::Drape& d) { return d.slot == slot; });
      std::vector<float> whole = liveRest(mesh, drape != m_liveJob->drapes.end());
      // a coarse copy skins only what it kept; a drape's own deltas are of all of it
      std::vector<float> rest = whole;
      if(const auto& keep = m_liveJob->slotKeep[slot]; !keep.empty() && whole.size() > size_t(keep.back()) * 3 + 2)
        rest = physics::cloth::keepOf(whole, keep, 3);
      if(rest.size() != m_liveJob->bodies[slot].restLocal.size())
        continue;
      if(drape != m_liveJob->drapes.end())
        drape->rest = std::move(whole);
      m_liveJob->bodies[slot].restLocal = rest;
      m_liveRestMoved = m_live->setRest(slot, std::move(rest)) || m_liveRestMoved;
    }
    m_liveReshaped.clear();
  }
  while(!m_liveRuns.empty() && m_liveRuns.front().drapes.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    LiveRun                         run = std::move(m_liveRuns.front());
    std::vector<std::vector<float>> deltas;
    m_liveRuns.pop_front();
    try
    {
      deltas = run.drapes.get();
    }
    catch(const std::exception&)
    {
      deltas.clear();
    }
    // one of a simulation since made again, or let go: dropped
    if(!m_liveJob || run.job != m_liveJob)
      continue;
    if(run.settle && m_liveSettling)
    {
      m_liveSettling = false;
      m_version++;
    }
    if(m_liveGen != m_gen || !m_enabled || !m_live || deltas.size() != m_liveJob->drapes.size())
      continue;
    // its deltas were taken against the rest shapes as it began: the drapes are drawn from them as they
    // are now, so by as much as those moved since (the pose's correctives, under a clip, every frame)
    for(size_t k = 0; k < deltas.size() && k < run.rests->size(); k++)
    {
      const std::vector<float>& then = (*run.rests)[k];
      const std::vector<float>& now  = m_liveJob->drapes[k].rest;
      if(then.size() == now.size() && deltas[k].size() == now.size())
        for(size_t i = 0; i < now.size(); i++)
          deltas[k][i] += then[i] - now[i];
    }
    if(run.settle || m_shown.size() != deltas.size())
    {
      // the settle that began live, at once, as a settle's drape is
      timing::Accum t("dforce: live, the drape written");
      wrote     = writeDeltas(*m_liveJob, deltas) || wrote;
      m_shown   = std::move(deltas);
      m_easeFor = 0;
    }
    else
    {
      // eased in from what is shown, over as long as since the last landed (or since it began, if it came
      // after): the next one lands about as long after
      m_easeFrom  = m_shown;
      m_easeTo    = std::move(deltas);
      m_easeStart = frameAt;
      m_easeFor   = std::clamp(std::chrono::duration<float>(frameAt - std::max(run.began, m_liveLanded)).count(), kLiveMinStep, kLiveEaseMost);
    }
    m_liveLanded = frameAt;
  }
  // a frame the simulation is still stepping through: counted, so a slow step shows as frames without cloth
  if(!free)
  {
    timing::accumulate("dforce: live, a frame its step was still running", 0);
    return wrote;
  }
  if(!m_enabled)
    return wrote;
  // a rebuild that is due: a dial changed a rest shape, something was worn, or live was just turned on
  if(m_scheduled && std::chrono::steady_clock::now() >= m_due)
  {
    m_scheduled = false;
    m_live.reset();
    m_liveJob.reset();
    m_easeFor = 0;
    m_shown.clear();
    // the new job reads every shape as it is now
    m_liveReshaped.clear();
    m_liveRestMoved = false;
    if(!active())
      return wrote;
    m_liveJob = prepare(m_gen);
    // strand hair settles only: it has no live simulation
    if(m_liveJob && m_liveJob->drapes.empty())
      m_liveJob.reset();
    if(!m_liveJob)
      return wrote;
    SettleOptions opts;
    opts.air = kAir;
    if(m_mode == Mode::eLiveGpu && makeGpuSim)
      m_live = makeGpuSim(m_liveJob->bodies, opts);
    if(!m_live)
      m_live = std::make_shared<ClothSim>(m_liveJob->bodies, opts);
    m_liveBones    = currentBones(*m_liveJob);
    m_liveGravity  = currentGravity();
    m_liveGen      = m_gen;
    m_liveSettling = true;
    m_liveMoving   = true;
    // the settle holds the pose for as long as a still pose is ever held
    m_liveStill    = kLiveHold;
    m_liveLast     = clock ? clock() : std::chrono::steady_clock::now();
    m_version++;
    auto sim = m_live;
    if(sim->ownerThreadOnly())
    {
      // a GPU simulation runs here, on the thread that owns its queue: the settle blocks this frame
      sim->settle();
      m_liveSettling = false;
      wrote          = apply(*m_liveJob, sim->positions()) || wrote;
      return wrote;
    }
    launchLive(sim, [](ClothBackend& s) { s.settle(); }, true);
    return wrote;
  }
  if(!m_live || m_scheduled)
    return wrote;
  // one stepping and one making its drapes at most: the next waits for the older's drapes, and its time
  // carries over
  if(m_liveRuns.size() >= 2)
    return wrote;
  // the pose now, and which way is down; a pose that stands still needs no step once the cloth is at
  // rest, or has had as long as a settle gives it: a few vertices caught in a fold may never be
  std::optional<timing::Accum> posing(std::in_place, "dforce: live, the pose read");
  auto bones = currentBones(*m_liveJob);
  // a rest shape moved (a dent, the correctives) wakes it as a bone does
  bool moved = m_liveRestMoved;
  for(size_t slot = 0; slot < bones.size(); slot++)
    moved = moved || !samePose(bones[slot], m_liveBones[slot]);
  const std::array<float, 3> gravity = currentGravity();
  const bool                 turned  = !sameGravity(gravity, m_liveGravity);
  if(moved || turned)
    m_liveStill = 0;
  const auto now = clock ? clock() : std::chrono::steady_clock::now();
  if(!moved && !turned && !m_liveMoving && (m_live->lastMove() < 0.01f || m_liveStill >= kLiveHold))
  {
    m_liveLast = now;
    return wrote;
  }
  // shorter than the shortest step: the time carries over, and what woke it is still there to see
  const float elapsed = std::chrono::duration<float>(now - m_liveLast).count();
  if(elapsed < kLiveMinStep)
    return wrote;
  const float seconds = std::min(elapsed, kLiveMaxStep);
  // Longer than a step may simulate (a slow frame, or the settle that began it): the pose has gone on
  // meanwhile. The cloth is carried with it through the time it can't simulate, where the simulation
  // can (the GPU's), and sweeps only the last step's worth: a sweep through all of it was dearer, so the
  // next step came later still (a walk once live had settled: steps of 0.4 to 1.2 s)
  const float late = elapsed > kLiveMaxStep ? 1.f - kLiveMaxStep / elapsed : 0.f;
  m_liveLast          = now;
  for(size_t slot = 0; slot < bones.size(); slot++)
    if(!samePose(bones[slot], m_liveBones[slot]))
      m_live->setTarget(slot, bones[slot]);
  if(turned)
  {
    m_liveGravity = gravity;
    m_live->setGravity(gravity);
  }
  if(!moved)
    m_liveStill += seconds;
  // the drape of this step is read back through the pose it was stepped to
  const auto worlds = m_scene.worlds();
  const auto nodeOf = nodeOfMesh(m_scene);
  for(Job::Drape& d : m_liveJob->drapes)
    if(m_liveJob->slotMesh[d.slot] >= 0)
      d.pose = currentSkin(m_scene, worlds, nodeOf, d.mesh);
  posing.reset();
  m_liveBones  = std::move(bones);
  m_liveMoving = moved || turned;
  m_liveRestMoved = false;
  auto        sim     = m_live;
  const float damping = moved ? 0.f : kLiveRestDamping * std::clamp((m_liveStill - kLiveFree) / kLiveCalming, 0.f, 1.f);
  if(sim->ownerThreadOnly())
  {
    {
      timing::Accum t("dforce: live, the step");
      sim->step(damping, seconds, late);
    }
    timing::Accum t("dforce: live, the drape written");
    return apply(*m_liveJob, sim->positions()) || wrote;
  }
  launchLive(
      sim,
      [damping, seconds, late](ClothBackend& s) {
        timing::Accum t("dforce: live, the step");
        s.step(damping, seconds, late);
      },
      false);
  return wrote;
}

void Dforce::launchLive(std::shared_ptr<ClothBackend> sim, std::function<void(ClothBackend&)> advance, bool settle)
{
  LiveRun run;
  run.job    = m_liveJob;
  run.settle = settle;
  run.began  = clock ? clock() : std::chrono::steady_clock::now();
  // what the drapes are taken against and read back through as this step begins: the job's change with
  // the next step's, which may begin while this one's drapes are made
  auto rests = std::make_shared<std::vector<std::vector<float>>>();
  auto poses = std::make_shared<std::vector<SkinPose>>();
  rests->reserve(m_liveJob->drapes.size());
  poses->reserve(m_liveJob->drapes.size());
  for(const Job::Drape& d : m_liveJob->drapes)
  {
    rests->push_back(d.rest);
    poses->push_back(d.pose);
  }
  run.rests    = rests;
  auto stepped = std::make_shared<std::promise<void>>();
  run.stepped  = stepped->get_future();
  run.drapes   = jobs::pool().submit([sim = std::move(sim), advance = std::move(advance), stepped, job = m_liveJob, rests, poses, hook = beforeLiveDrapes] {
    std::vector<std::vector<float>> positions;
    try
    {
      advance(*sim);
      positions = sim->positions();
    }
    catch(...)
    {
      stepped->set_value();
      throw;
    }
    // the simulation is free for the next step from here: only what it gave is read below
    stepped->set_value();
    if(hook)
      hook();
    if(positions.empty())
      return std::vector<std::vector<float>>{};
    return deltasOf(*job, positions, rests.get(), poses.get());
  });
  m_liveRuns.push_back(std::move(run));
}

bool Dforce::easeLive(std::chrono::steady_clock::time_point now)
{
  if(m_easeFor <= 0)
    return false;
  if(!m_liveJob || m_liveGen != m_gen || !m_enabled)
  {
    m_easeFor = 0;
    return false;
  }
  const float a = std::clamp(std::chrono::duration<float>(now - m_easeStart).count() / m_easeFor, 0.f, 1.f);
  if(a <= 0)
    return false;
  m_shown.resize(m_easeTo.size());
  for(size_t k = 0; k < m_easeTo.size(); k++)
  {
    const std::vector<float>& to   = m_easeTo[k];
    const std::vector<float>& from = k < m_easeFrom.size() ? m_easeFrom[k] : to;
    std::vector<float>&       at   = m_shown[k];
    if(from.size() != to.size() || a >= 1)
    {
      at = to;
      continue;
    }
    at.resize(to.size());
    for(size_t i = 0; i < to.size(); i++)
      at[i] = from[i] + (to[i] - from[i]) * a;
  }
  if(a >= 1)
    m_easeFor = 0;
  timing::Accum t("dforce: live, the drape eased");
  return writeDeltas(*m_liveJob, m_shown);
}

std::vector<float> Dforce::liveRest(int mesh, bool drape) const
{
  // the morph set's kept cage, and a drape's own target taken back out of it (as prepare's skinCage
  // leaves it out), rather than every target summed again
  std::vector<float> rest = m_morphs.morphedCage(mesh);
  if(!drape)
    return rest;
  const auto   own = m_morphs.cpuTarget(channelOf(mesh), mesh);
  const double w   = own ? m_morphs.weightOf(channelOf(mesh)) : 0.0;
  if(own && w != 0)
    for(size_t i = 0; i < own->cageIndices.size(); i++)
    {
      const size_t v = own->cageIndices[i];
      if(v * 3 + 2 >= rest.size())
        continue;
      for(size_t c = 0; c < 3; c++)
        rest[v * 3 + c] -= float(own->cageDeltas[i * 3 + c] * w);
    }
  return rest;
}

bool Dforce::writeDrape(int mesh, const std::vector<float>& delta)
{
  const Mesh& m = m_scene.meshes[size_t(mesh)];
  if(!m.geo)
    return false;
  mesh::CageDeltas cage;
  for(size_t v = 0; v * 3 + 2 < delta.size(); v++)
  {
    const float x = delta[v * 3], y = delta[v * 3 + 1], z = delta[v * 3 + 2];
    if(std::fabs(x) + std::fabs(y) + std::fabs(z) <= 1e-4f)
      continue;
    cage.verts.push_back(int32_t(v));
    cage.deltas.insert(cage.deltas.end(), {x, y, z});
  }
  auto target = std::make_shared<const mesh::RenderTarget>(cage.verts.empty() ? mesh::RenderTarget{} : mesh::target(*m.geo, cage));
  morph::ChannelOptions opts;
  opts.hidden = true;
  opts.label  = "dForce";
  opts.value  = 1.0;
  opts.min    = 0.0;
  opts.max    = 1.0;
  // a target only goes on at a weight: one left at 0 would have this settle run for nothing
  if(m_morphs.has(channelOf(mesh)))
    m_morphs.setMany({{channelOf(mesh), 1.0}});
  m_morphs.replaceCpuTarget(channelOf(mesh), mesh, std::move(target), opts);
  if(m_draped.insert(mesh).second)
    m_version++;
  return true;
}

std::vector<std::vector<float>> Dforce::deltasOf(const Job& job, const std::vector<std::vector<float>>& positions,
                                                 const std::vector<std::vector<float>>* rests, const std::vector<SkinPose>* poses)
{
  timing::Accum                   t("dforce: the drapes' deltas");
  std::vector<std::vector<float>> out(job.drapes.size());
  for(size_t k = 0; k < job.drapes.size(); k++)
  {
    const Job::Drape&         d    = job.drapes[k];
    const std::vector<float>& rest = rests && k < rests->size() ? (*rests)[k] : d.rest;
    const SkinPose&           pose = poses && k < poses->size() ? (*poses)[k] : d.pose;
    if(d.slot >= positions.size())
      continue;
    if(!d.light)
    {
      out[k] = drapeDelta(rest, transformPoints(d.back, positions[d.slot]), d.skin, pose);
      continue;
    }
    // a light copy's cloth rides on it, out of the body where the copy's faces cross a curve of it, and
    // where it is held to its skin it stays, as the cloth would
    std::vector<float> at;
    {
      timing::Accum ridden("dforce: a light drape ridden on its copy");
      at = physics::cloth::ride(*d.light, positions[d.slot]);
    }
    if(job.bodySlot < positions.size() && job.bodySlot < job.bodies.size())
    {
      timing::Accum kept("dforce: a light drape kept out of the body");
      physics::cloth::keepOut(at, d.light->riding.held, job.bodies[job.bodySlot], positions[job.bodySlot], 2.f, d.light.get(), &positions[d.slot]);
    }
    {
      timing::Accum skinned("dforce: a light drape's inverse skin");
      out[k] = drapeDelta(rest, transformPoints(d.back, at), d.skin, pose);
    }
    const auto& held = d.light->riding.held;
    for(size_t v = 0; v < held.size() && v * 3 + 2 < out[k].size(); v++)
      if(held[v])
        out[k][v * 3] = out[k][v * 3 + 1] = out[k][v * 3 + 2] = 0.f;
  }
  return out;
}

bool Dforce::writeDeltas(const Job& job, const std::vector<std::vector<float>>& deltas)
{
  bool wrote = false;
  // an item taken off while this ran is not draped
  const std::vector<int> live = dynamics();
  for(size_t k = 0; k < job.drapes.size() && k < deltas.size(); k++)
  {
    const Job::Drape& d = job.drapes[k];
    if(deltas[k].empty() || d.mesh < 0 || size_t(d.mesh) >= m_scene.meshes.size()
       || std::find(live.begin(), live.end(), d.mesh) == live.end() || !m_scene.meshes[size_t(d.mesh)].geo)
      continue;
    timing::Accum t(m_mode == Mode::eSettle ? "dforce: a drape's morph" : "dforce: live, a drape's morph");
    wrote = writeDrape(d.mesh, deltas[k]) || wrote;
  }
  if(wrote && m_mode == Mode::eSettle)
    m_version++;
  return wrote;
}

bool Dforce::apply(const Job& job, const std::vector<std::vector<float>>& positions)
{
  return writeDeltas(job, deltasOf(job, positions));
}

}  // namespace fv::dforce
