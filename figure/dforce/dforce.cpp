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

}  // namespace

std::string channelOf(int mesh)
{
  return "dforce:" + std::to_string(mesh);
}

// What a settle needs and where each result goes, taken on the application thread.
struct Dforce::Job
{
  std::vector<SettleMesh> bodies;
  struct Drape
  {
    int                mesh = -1;
    size_t             slot = 0;  // into bodies and the settle's result
    std::vector<float> rest;
    Top4               skin;
    SkinPose           pose;
    Mat4               back;  // job space -> mesh-local
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

void Dforce::poke()
{
  if(!m_enabled || (playing && playing()) || !active())
    return;
  m_gen++;
  m_scheduled = true;
  m_due       = std::chrono::steady_clock::now() + kDebounce;
  m_version++;
}

void Dforce::changed(const std::vector<int>& meshes, bool nodesMoved)
{
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
  if(m_scheduled || m_job.valid())
    m_version++;
  m_scheduled = false;
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

std::shared_ptr<Dforce::Job> Dforce::prepare(uint64_t) const
{
  const std::vector<int> dyn = dynamics();
  if(dyn.empty())
    return nullptr;
  auto               job    = std::make_shared<Job>();
  const auto         worlds = m_scene.worlds();
  std::vector<int>   nodeOf(m_scene.meshes.size(), -1);
  for(size_t n = 0; n < m_scene.nodes.size(); n++)
    if(m_scene.nodes[n].kind == NodeKind::eMesh && m_scene.nodes[n].mesh >= 0)
      nodeOf[size_t(m_scene.nodes[n].mesh)] = int(n);

  struct Skinned
  {
    std::vector<float> rest, posed;
    Top4               skin;
    SkinPose           pose;
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
    s.skin = top4(m.cageSkin);
    // three's 'attached' skinning: matrixWorld⁻¹ · bone · inverse · bindMatrix, with the mesh's world as
    // it is now (the host scales the root to metres after the bind, and rebinds read that scale)
    s.pose.bind        = m.bindMatrix;
    s.pose.bindInverse = inverse(nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : m.bindMatrix);
    for(size_t k = 0; k < m.bones.size(); k++)
      s.pose.bones.push_back(worlds[size_t(m.bones[k])] * (k < m.boneInverses.size() ? m.boneInverses[k] : Mat4{}));
    s.posed = skinLocal(s.rest, s.skin, s.pose);
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
      job->bodies.push_back(SettleMesh::collider(transformPoints(to, body->rest), transformPoints(to, body->posed),
                                                 m_scene.meshes[size_t(m_bodyMesh)].geo->dsonTris, kBodyLayer, 0.2f, 0.4f));
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
      Job::Drape d;
      d.mesh = mesh;
      d.slot = job->bodies.size();
      d.rest = std::move(skinned->rest);
      d.skin = std::move(skinned->skin);
      d.pose = std::move(skinned->pose);
      d.back = inverse(to);
      job->bodies.push_back(std::move(s));
      job->drapes.push_back(std::move(d));
    }
    else if(data.visible)
    {
      job->bodies.push_back(SettleMesh::collider(std::move(rest), std::move(posed),
                                                 data.kind == dson::DforceKind::eAddon ? std::vector<uint32_t>{} : m.geo->dsonTris,
                                                 data.layer.empty() ? 0 : data.layer[0], data.offset.empty() ? 0.2f : data.offset[0],
                                                 data.friction.empty() ? 0.4f : data.friction[0], data.surfaceCollide));
    }
  };
  for(int mesh = 0; mesh < int(m_scene.meshes.size()); mesh++)
    consider(mesh);
  if(job->drapes.empty())
    return nullptr;
  return job;
}

bool Dforce::pump()
{
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
    m_draped.insert(d.mesh);
    wrote = true;
  }
  if(wrote)
    m_version++;
  return wrote;
}

}  // namespace fv::dforce
