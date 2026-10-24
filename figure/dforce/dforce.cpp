#include <figure/dforce/dforce.hpp>

#include <algorithm>
#include <cmath>

#include <figure/dforce/skin.hpp>
#include <figure/dforce/solver.hpp>
#include <figure/jobs/pool.hpp>

namespace fv::dforce {

namespace {

using namespace std::chrono_literals;

constexpr auto  kDebounce = 400ms;
constexpr float kAir      = 0.15f;
// Velocity decay per second a live step adds while the pose stands still, so the cloth comes to rest.
constexpr float kLiveRestDamping = 6.f;

std::vector<float> transformPoints(const Mat4& m, const std::vector<float>& src)
{
  std::vector<float> out(src.size());
  const auto&        e = m.e;
  for(size_t i = 0; i + 2 < src.size(); i += 3)
  {
    const double x = src[i], y = src[i + 1], z = src[i + 2];
    out[i]     = float(e[0] * x + e[4] * y + e[8] * z + e[12]);
    out[i + 1] = float(e[1] * x + e[5] * y + e[9] * z + e[13]);
    out[i + 2] = float(e[2] * x + e[6] * y + e[10] * z + e[14]);
  }
  return out;
}

// Morphed cage: the fitted base plus every active morph's cage deltas, except the drape itself.
std::vector<float> morphedCage(const morph::MorphSet& morphs, int mesh, const mesh::BuiltMesh& geo, const mesh::RenderTarget* except)
{
  std::vector<float> out = geo.cagePositions;
  for(const auto& [target, weight] : morphs.activeTargets(mesh, except))
    for(size_t i = 0; i < target->cageIndices.size(); i++)
    {
      const size_t v = target->cageIndices[i];
      if(v * 3 + 2 >= out.size())
        continue;
      out[v * 3] += float(target->cageDeltas[i * 3] * weight);
      out[v * 3 + 1] += float(target->cageDeltas[i * 3 + 1] * weight);
      out[v * 3 + 2] += float(target->cageDeltas[i * 3 + 2] * weight);
    }
  return out;
}

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

// The node each mesh is drawn by, or -1.
std::vector<int> nodeOfMesh(const SceneGraph& scene)
{
  std::vector<int> nodeOf(scene.meshes.size(), -1);
  for(size_t n = 0; n < scene.nodes.size(); n++)
    if(scene.nodes[n].kind == NodeKind::eMesh && scene.nodes[n].mesh >= 0)
      nodeOf[size_t(scene.nodes[n].mesh)] = int(n);
  return nodeOf;
}

// A skinned mesh's pose as the figure has it now: three's 'attached' skinning, matrixWorld⁻¹ ·
// bone · inverse · bindMatrix, with the mesh's world as it is now (the host scales the root to
// metres after the bind, and rebinds read that scale).
SkinPose currentSkin(const SceneGraph& scene, const std::vector<Mat4>& worlds, const std::vector<int>& nodeOf, int mesh)
{
  const Mesh& m = scene.meshes[size_t(mesh)];
  SkinPose    pose;
  pose.bind        = m.bindMatrix;
  pose.bindInverse = inverse(nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : m.bindMatrix);
  pose.bones.reserve(m.bones.size());
  for(size_t k = 0; k < m.bones.size(); k++)
    pose.bones.push_back(worlds[size_t(m.bones[k])] * (k < m.boneInverses.size() ? m.boneInverses[k] : Mat4{}));
  return pose;
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
  struct Drape
  {
    int                mesh = -1;
    size_t             slot = 0;  // into bodies and the settle's result
    std::vector<float> rest;
    Top4               skin;
    // the pose the positions are read back through: the job's, or the live step's
    SkinPose pose;
    Mat4     back;  // job space -> mesh-local
  };
  std::vector<Drape> drapes;
};

Dforce::Dforce(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root)
    : m_scene(scene)
    , m_morphs(morphs)
    , m_bodyMesh(bodyMesh)
    , m_root(root)
{
}

Dforce::~Dforce()
{
  // the pool keeps the snapshot it was given; only wait so a late result cannot outlive the owner
  if(m_job.valid())
    m_job.wait();
  if(m_liveStep.valid())
    m_liveStep.wait();
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
    if(m.dforce && m.dforce->simulated() && node.mesh != m_bodyMesh)
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
  return m_scheduled || m_job.valid() || m_liveSettling;
}

void Dforce::poke()
{
  if(!m_enabled || !active())
    return;
  if(m_mode == Mode::eSettle && playing && playing())
    return;
  m_gen++;
  m_scheduled = true;
  m_due       = std::chrono::steady_clock::now() + kDebounce;
  m_version++;
}

void Dforce::changed(const std::vector<int>& meshes, bool nodesMoved)
{
  if(m_mode != Mode::eSettle)
  {
    // bones are read again next frame; a changed mesh is a new rest shape, so the simulation is
    // built again once the dial stops
    if(!meshes.empty())
      poke();
    return;
  }
  if(nodesMoved || !meshes.empty())
    poke();
}

void Dforce::resimulate()
{
  m_enabled   = true;
  m_scheduled = true;
  m_due       = std::chrono::steady_clock::now();
  m_gen++;
  m_version++;
}

void Dforce::cancel()
{
  m_gen++;
  if(m_scheduled || m_job.valid() || m_live)
    m_version++;
  m_scheduled = false;
  // a live step in flight is dropped when it lands; the simulation goes
  m_live.reset();
  m_liveJob.reset();
  m_liveSettling = false;
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
    Top4               skin;
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
    s.rest = morphedCage(m_morphs, mesh, *m.geo, except);
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
    s.skin    = top4(m.cageSkin);
    s.pose    = currentSkin(m_scene, worlds, nodeOf, mesh);
    s.posed   = skinLocal(s.rest, s.skin, s.pose);
    return s;
  };
  // The job's space is the figure root's, in centimetres as the solver's constants are, whatever
  // scale and placement the root has in the world (the host draws in metres).
  const Mat4 rootWorld = m_root >= 0 && size_t(m_root) < worlds.size() ? worlds[size_t(m_root)] : Mat4{};
  const Mat4 fromWorld = inverse(rootWorld);
  const auto toSpace   = [&](int mesh) { return fromWorld * (nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : Mat4{}); };

  // world down, turned into the root's frame
  const Vec3 g = rotate(Vec3{0, -980, 0}, rotationOf(rootWorld).inverse());
  const std::array<float, 3> gravity{float(g.x), float(g.y), float(g.z)};

  // the body is always a collider, in the same space as the cloth, and under every cloth layer
  if(m_bodyMesh >= 0 && size_t(m_bodyMesh) < m_scene.meshes.size())
    if(auto body = skinCage(m_bodyMesh, nullptr))
    {
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
    if(mesh == m_bodyMesh || !m.dforce || !m.dforce->hasPolygons() || !m.geo || nodeOf[size_t(mesh)] < 0 || !shown[size_t(nodeOf[size_t(mesh)])])
      return;
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
  if(job->drapes.empty())
    return nullptr;
  return job;
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
    out[slot] = currentSkin(m_scene, worlds, nodeOf, mesh).bones;
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
    if(gen == m_gen && m_enabled && job && !positions.empty())
      wrote = apply(*job, positions);
  }
  if(m_scheduled && !m_job.valid() && std::chrono::steady_clock::now() >= m_due)
  {
    m_scheduled = false;
    if(m_enabled && active())
    {
      if(auto job = prepare(m_gen))
      {
        m_snapshot = job;
        m_jobGen   = m_gen;
        m_job      = jobs::pool().submit([job] {
          SettleOptions opts;
          opts.air = kAir;
          return settle(job->bodies, opts);
        });
        m_version++;
      }
    }
  }
  // a newer edit that arrives while one runs waits for it: that result is dropped above, then this starts
  return wrote;
}

// Live: the step in flight lands and is written; then, unless the simulation is to be built again
// or nothing has moved, the figure's bones now go to the simulation and it steps once more.
bool Dforce::pumpLive()
{
  bool wrote = false;
  if(m_liveStep.valid())
  {
    if(m_liveStep.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      return false;
    std::vector<std::vector<float>> positions;
    try
    {
      positions = m_liveStep.get();
    }
    catch(const std::exception&)
    {
      positions.clear();
    }
    if(m_liveSettling)
    {
      m_liveSettling = false;
      m_version++;
    }
    if(m_liveGen == m_gen && m_enabled && m_liveJob && m_live && !positions.empty())
      wrote = apply(*m_liveJob, positions);
  }
  if(!m_enabled)
    return wrote;
  // a rebuild that is due: a dial changed a rest shape, something was worn, or live was just turned on
  if(m_scheduled && std::chrono::steady_clock::now() >= m_due)
  {
    m_scheduled = false;
    m_live.reset();
    m_liveJob.reset();
    if(!active())
      return wrote;
    m_liveJob = prepare(m_gen);
    if(!m_liveJob)
      return wrote;
    SettleOptions opts;
    opts.air = kAir;
    if(m_mode == Mode::eLiveGpu && makeGpuSim)
      m_live = makeGpuSim(m_liveJob->bodies, opts);
    if(!m_live)
      m_live = std::make_shared<ClothSim>(m_liveJob->bodies, opts);
    m_liveBones    = currentBones(*m_liveJob);
    m_liveGen      = m_gen;
    m_liveSettling = true;
    m_liveMoving   = true;
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
    m_liveStep = jobs::pool().submit([sim] {
      sim->settle();
      return sim->positions();
    });
    return wrote;
  }
  if(!m_live || m_scheduled)
    return wrote;
  // the pose now; a pose that stands still with the cloth at rest needs no step
  auto bones = currentBones(*m_liveJob);
  bool moved = false;
  for(size_t slot = 0; slot < bones.size(); slot++)
    if(!samePose(bones[slot], m_liveBones[slot]))
    {
      moved = true;
      m_live->setTarget(slot, bones[slot]);
    }
  if(!moved && !m_liveMoving && m_live->lastMove() < 0.01f)
    return wrote;
  // the drape of this step is read back through the pose it was stepped to
  const auto worlds = m_scene.worlds();
  const auto nodeOf = nodeOfMesh(m_scene);
  for(Job::Drape& d : m_liveJob->drapes)
    if(m_liveJob->slotMesh[d.slot] >= 0)
      d.pose = currentSkin(m_scene, worlds, nodeOf, d.mesh);
  m_liveBones  = std::move(bones);
  m_liveMoving = moved;
  auto        sim     = m_live;
  const float damping = moved ? 0.f : kLiveRestDamping;
  if(sim->ownerThreadOnly())
  {
    sim->step(damping);
    return apply(*m_liveJob, sim->positions()) || wrote;
  }
  m_liveStep = jobs::pool().submit([sim, damping] {
    sim->step(damping);
    return sim->positions();
  });
  return wrote;
}

bool Dforce::apply(const Job& job, const std::vector<std::vector<float>>& positions)
{
  bool wrote = false;
  // an item taken off while this ran is not draped
  const std::vector<int> live = dynamics();
  for(const Job::Drape& d : job.drapes)
  {
    if(d.slot >= positions.size() || d.mesh < 0 || size_t(d.mesh) >= m_scene.meshes.size()
       || std::find(live.begin(), live.end(), d.mesh) == live.end())
      continue;
    const Mesh& m = m_scene.meshes[size_t(d.mesh)];
    if(!m.geo)
      continue;
    const auto local = transformPoints(d.back, positions[d.slot]);
    const auto delta = drapeDelta(d.rest, local, d.skin, d.pose);
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
    m_morphs.replaceCpuTarget(channelOf(d.mesh), d.mesh, std::move(target), opts);
    if(m_draped.insert(d.mesh).second)
      m_version++;
    wrote = true;
  }
  if(wrote && m_mode == Mode::eSettle)
    m_version++;
  return wrote;
}

}  // namespace fv::dforce
