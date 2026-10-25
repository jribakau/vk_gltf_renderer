#include <figure/wardrobe/collider.hpp>

#include <algorithm>
#include <cmath>

#include <figure/dforce/skin.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/collide.hpp>
#include <figure/wardrobe/fit/regions.hpp>

namespace fv::wardrobe {

namespace {

using namespace std::chrono_literals;

// A drag moves a bone every frame: the pass waits for a pause this long, so it runs on the pose the
// user stopped at rather than on every frame of the way there.
constexpr auto kDebounce = 120ms;
// Clearance kept between the garment and the skin, and how deep inside the body a vertex is still
// taken as pushed through the surface it is closest to (deeper is the other side of a limb).
constexpr float kMargin = 0.25f;
constexpr float kDepth  = 3.f;
// A correction below this (centimetres) is not worth a morph.
constexpr float kMinMove = 1e-3f;

// Whether the node of every mesh is shown: not taken off, not hidden, nor anything above it.
std::vector<bool> shownNodes(const SceneGraph& scene)
{
  std::vector<bool> shown(scene.nodes.size(), true);
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    const Node& n = scene.nodes[i];
    shown[i]      = n.visible && !n.removed && (n.parent < 0 || shown[size_t(n.parent)]);
  }
  return shown;
}

}  // namespace

std::string collideChannelOf(int mesh)
{
  return "collide:" + std::to_string(mesh);
}

// What a pass needs and where each result goes, taken on the owning thread.
struct Collider::Job
{
  // the body as posed now, in the root's space, with the limb of each vertex
  std::vector<float>    body;
  std::vector<uint32_t> bodyTris;
  std::vector<int16_t>  bodyLimbs;
  struct Item
  {
    int                   mesh = -1;
    dson::Smoothing       settings;
    std::vector<float>    posed;  // in the root's space
    std::vector<uint32_t> tris;
    std::vector<int16_t>  limbs;
    // for the write-back: the rest cage, its skin and pose in the mesh's space, and the way back there
    std::vector<float> rest;
    dforce::Top4       skin;
    dforce::SkinPose   pose;
    Mat4               back;
  };
  std::vector<Item> items;
};

Collider::Collider(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root)
    : m_scene(scene)
    , m_morphs(morphs)
    , m_bodyMesh(bodyMesh)
    , m_root(root)
{
}

Collider::~Collider()
{
  if(m_job.valid())
    m_job.wait();
}

bool Collider::busy() const
{
  return m_scheduled || m_job.valid();
}

void Collider::changed(const std::vector<int>& meshes, bool nodesMoved)
{
  if(nodesMoved || !meshes.empty())
    poke();
}

void Collider::poke()
{
  m_gen++;
  if(!m_enabled)
    return;
  if(!m_scheduled)
    m_version++;
  m_scheduled = true;
  m_due       = std::chrono::steady_clock::now() + kDebounce;
}

void Collider::setEnabled(bool on)
{
  if(on == m_enabled)
    return;
  m_enabled = on;
  m_version++;
  if(on)
  {
    poke();
    return;
  }
  m_scheduled = false;
  m_gen++;
  std::vector<std::pair<std::string, double>> off;
  for(int mesh : m_corrected)
    if(m_morphs.has(collideChannelOf(mesh)))
      off.emplace_back(collideChannelOf(mesh), 0.0);
  m_corrected.clear();
  if(!off.empty())
    m_morphs.setMany(off);
}

std::shared_ptr<Collider::Job> Collider::prepare() const
{
  if(m_bodyMesh < 0 || size_t(m_bodyMesh) >= m_scene.meshes.size() || !settings)
    return nullptr;
  const Mesh& body = m_scene.meshes[size_t(m_bodyMesh)];
  if(!body.geo || body.geo->cagePositions.empty())
    return nullptr;
  const auto worlds = m_scene.worlds();
  const auto nodeOf = dforce::nodeOfMesh(m_scene);
  const Mat4 rootWorld = m_root >= 0 && size_t(m_root) < worlds.size() ? worlds[size_t(m_root)] : Mat4{};
  const Mat4 fromWorld = inverse(rootWorld);
  const auto toSpace = [&](int mesh) { return fromWorld * (nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : Mat4{}); };
  // a mesh's cage in its own space as posed now: skinned when it has a skin, else as it is
  const auto posedCage = [&](int mesh, const std::vector<float>& rest, dforce::Top4* skin, dforce::SkinPose* pose) {
    const Mesh& m = m_scene.meshes[size_t(mesh)];
    if(!m.skinned() || m.cageSkin.start.size() != rest.size() / 3 + 1)
      return rest;
    dforce::Top4     s = dforce::top4(m.cageSkin);
    dforce::SkinPose p = dforce::currentSkin(m_scene, worlds, nodeOf, mesh);
    auto             posed = dforce::skinLocal(rest, s, p);
    if(skin)
      *skin = std::move(s);
    if(pose)
      *pose = std::move(p);
    return posed;
  };

  auto job       = std::make_shared<Job>();
  job->body      = dforce::transformPoints(toSpace(m_bodyMesh), posedCage(m_bodyMesh, m_morphs.morphedCage(m_bodyMesh), nullptr, nullptr));
  job->bodyTris  = body.geo->dsonTris;
  job->bodyLimbs = meshLimbs(m_scene, body);

  const auto shown = shownNodes(m_scene);
  for(const auto& [mesh, s] : settings())
  {
    if(!s.enabled || mesh == m_bodyMesh || mesh < 0 || size_t(mesh) >= m_scene.meshes.size())
      continue;
    const Mesh& m = m_scene.meshes[size_t(mesh)];
    if(!m.geo || m.geo->cagePositions.empty() || !m.skinned() || nodeOf[size_t(mesh)] < 0 || !shown[size_t(nodeOf[size_t(mesh)])])
      continue;
    if(m.cageSkin.start.size() != m.geo->cagePositions.size() / 3 + 1 || (owned && owned(mesh)))
      continue;
    Job::Item item;
    item.mesh     = mesh;
    item.settings = s;
    // the shape under the last correction, so a pass measures the garment, not its own push
    const auto own = m_morphs.cpuTarget(collideChannelOf(mesh), mesh);
    item.rest      = m_morphs.morphedCage(mesh, own.get());
    const Mat4 to  = toSpace(mesh);
    item.posed     = dforce::transformPoints(to, posedCage(mesh, item.rest, &item.skin, &item.pose));
    item.tris      = m.geo->dsonTris;
    item.limbs     = meshLimbs(m_scene, m);
    item.back      = inverse(to);
    job->items.push_back(std::move(item));
  }
  if(job->items.empty())
    return nullptr;
  return job;
}

bool Collider::pump()
{
  bool wrote = false;
  if(m_job.valid() && m_job.wait_for(0s) == std::future_status::ready)
  {
    const auto positions = m_job.get();
    if(m_snapshot)
      wrote = apply(*m_snapshot, positions);
    m_snapshot.reset();
    // edits that arrived while it ran: once more, on the shape they left
    if(m_jobGen != m_gen && m_enabled)
    {
      m_scheduled = true;
      m_due       = std::chrono::steady_clock::now();
    }
    if(!m_scheduled)
      m_version++;
  }
  if(!m_scheduled || m_job.valid() || std::chrono::steady_clock::now() < m_due)
    return wrote;
  m_scheduled = false;
  m_jobGen    = m_gen;
  auto job    = prepare();
  if(!job)
  {
    // nothing left to push: whatever was corrected shows as it conforms
    std::vector<std::pair<std::string, double>> off;
    for(int mesh : m_corrected)
      if(m_morphs.has(collideChannelOf(mesh)))
        off.emplace_back(collideChannelOf(mesh), 0.0);
    if(!off.empty())
    {
      m_morphs.setMany(off);
      wrote = true;
    }
    m_corrected.clear();
    m_version++;  // no longer busy
    return wrote;
  }
  m_snapshot = job;
  m_job      = jobs::pool().submit([job] {
    std::vector<std::vector<float>> out;
    const Surface body{job->body, job->bodyTris};
    for(const Job::Item& item : job->items)
    {
      std::vector<float> positions = item.posed;
      const auto         filter    = limbFilter(job->bodyLimbs, item.limbs);
      const int          passes    = std::clamp(item.settings.collision, 1, 8);
      const int          smooth    = std::clamp(item.settings.smoothing, 1, 8);
      for(int pass = 0; pass < passes; pass++)
      {
        const SurfaceBinding binding = bindToSurface(body.positions, body.tris, positions, {}, filter ? &*filter : nullptr);
        std::vector<float>   pushed  = pushOut(positions, body, binding, kMargin, kDepth);
        if(std::equal(pushed.begin(), pushed.end(), positions.begin()))
          break;
        positions = smoothCorrection(positions, pushed, item.tris, smooth);
      }
      out.push_back(std::move(positions));
    }
    return out;
  });
  return wrote;
}

bool Collider::apply(const Job& job, const std::vector<std::vector<float>>& positions)
{
  bool          wrote = false;
  std::set<int> still;
  for(size_t k = 0; k < job.items.size() && k < positions.size(); k++)
  {
    const Job::Item& item = job.items[k];
    if(item.mesh < 0 || size_t(item.mesh) >= m_scene.meshes.size())
      continue;
    const Mesh& m = m_scene.meshes[size_t(item.mesh)];
    if(!m.geo || m.geo->cagePositions.size() != item.rest.size())
      continue;
    const auto local = dforce::transformPoints(item.back, positions[k]);
    const auto delta = dforce::drapeDelta(item.rest, local, item.skin, item.pose);
    mesh::CageDeltas cage;
    for(size_t v = 0; v * 3 + 2 < delta.size(); v++)
    {
      const float x = delta[v * 3], y = delta[v * 3 + 1], z = delta[v * 3 + 2];
      if(std::fabs(x) + std::fabs(y) + std::fabs(z) <= kMinMove)
        continue;
      cage.verts.push_back(int32_t(v));
      cage.deltas.insert(cage.deltas.end(), {x, y, z});
    }
    const bool had = m_corrected.count(item.mesh) > 0;
    if(cage.verts.empty() && !had)
      continue;
    auto target = std::make_shared<const mesh::RenderTarget>(cage.verts.empty() ? mesh::RenderTarget{} : mesh::target(*m.geo, cage));
    morph::ChannelOptions opts;
    opts.hidden = true;
    opts.label  = "Collision";
    opts.value  = 1.0;
    opts.min    = 0.0;
    opts.max    = 1.0;
    m_morphs.replaceCpuTarget(collideChannelOf(item.mesh), item.mesh, std::move(target), opts);
    // a channel turned off earlier keeps its weight: back to 1
    if(m_morphs.get(collideChannelOf(item.mesh)) != 1.0)
    {
      m_morphs.set(collideChannelOf(item.mesh), 1.0);
      m_morphs.flush();
    }
    wrote = true;
    if(!cage.verts.empty())
      still.insert(item.mesh);
  }
  // meshes this pass no longer listed (taken off, now simulated) lose their correction
  for(int mesh : m_corrected)
    if(!still.count(mesh) && m_morphs.has(collideChannelOf(mesh))
       && std::none_of(job.items.begin(), job.items.end(), [&](const Job::Item& i) { return i.mesh == mesh; }))
    {
      m_morphs.set(collideChannelOf(mesh), 0.0);
      wrote = true;
    }
  if(still != m_corrected)
    m_version++;
  m_corrected = std::move(still);
  return wrote;
}

}  // namespace fv::wardrobe
