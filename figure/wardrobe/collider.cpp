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
// The clearance an item comes to at skin tight: half a millimetre shows as fabric on the skin, where
// the 2.5 above shows as a gap close up.
constexpr float kTightMargin = 0.05f;
// How far off the skin a second pull still reaches: what the first left near it after smoothing, and
// not what it left part of the way on purpose, fading out towards a free hem.
constexpr float kTightSecondReach = 1.5f;
constexpr float kTightReach       = 6.f;
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

// How far an item lies from the body in its rest shape: the median signed distance of its cage from
// the body's, along the skin. The layer order: a bra lies closer than the shirt over it.
double restClearance(std::span<const float> body, std::span<const uint32_t> bodyTris, std::span<const float> item)
{
  const SurfaceBinding binding = bindToSurface(body, bodyTris, item);
  std::vector<double>  dist;
  for(size_t v = 0; v * 3 + 2 < item.size(); v++)
  {
    const size_t k = v * 3;
    const double w = double(binding.weights[k]) + binding.weights[k + 1] + binding.weights[k + 2];
    if(w <= 0)
      continue;
    const size_t ia = binding.verts[k], ib = binding.verts[k + 1], ic = binding.verts[k + 2];
    const V3     n  = triNormal(body, ia, ib, ic);
    double       d  = 0;
    for(size_t c = 0; c < 3; c++)
    {
      const double s = body[ia * 3 + c] * binding.weights[k] + body[ib * 3 + c] * binding.weights[k + 1] + body[ic * 3 + c] * binding.weights[k + 2];
      d += (item[k + c] - s) * n[c];
    }
    dist.push_back(d);
  }
  if(dist.empty())
    return 0;
  std::nth_element(dist.begin(), dist.begin() + long(dist.size() / 2), dist.end());
  return dist[dist.size() / 2];
}

// What a pass needs and where each result goes, taken on the owning thread.
struct Collider::Job
{
  // the body as posed now, in the root's space, with the limb of each vertex
  std::vector<float>    body;
  std::vector<uint32_t> bodyTris;
  std::vector<int16_t>  bodyLimbs;
  // its rest shape, to order the items by how far from it each lies
  std::vector<float> bodyRest;
  // the items inside out: each is pushed out of the body and of the items before it
  struct Item
  {
    int                   mesh = -1;
    dson::Smoothing       settings;
    double                clearance = 0;  // from the body at rest, cm; the layer order
    int                   worn      = 0;  // its place in the order worn, for ties
    std::vector<float>    posed;          // in the root's space
    std::vector<float>    restInSpace;    // its rest shape there, for its clearance
    std::vector<uint32_t> tris;
    std::vector<int16_t>  limbs;
    // for the write-back: the rest cage, its skin and pose in the mesh's space, and the way back there
    std::vector<float> rest;
    dforce::CageSkin   skin;
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

void Collider::pokeNow()
{
  m_gen++;
  if(!m_enabled)
    return;
  // an earlier due time stands: a drag doesn't put it off
  if(!m_scheduled)
  {
    m_version++;
    m_due = std::chrono::steady_clock::now();
  }
  m_scheduled = true;
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
  const auto posedCage = [&](int mesh, const std::vector<float>& rest, dforce::CageSkin* skin, dforce::SkinPose* pose) {
    const Mesh& m = m_scene.meshes[size_t(mesh)];
    if(!m.skinned() || m.cageSkin.start.size() != rest.size() / 3 + 1)
      return rest;
    dforce::CageSkin s = dforce::cageSkin(m.cageSkin, dazSkinning);
    dforce::SkinPose p = dforce::currentSkin(m_scene, worlds, nodeOf, mesh);
    auto             posed = dforce::skinLocal(rest, s, p);
    if(skin)
      *skin = std::move(s);
    if(pose)
      *pose = std::move(p);
    return posed;
  };

  auto       job      = std::make_shared<Job>();
  const auto bodyRest = m_morphs.morphedCage(m_bodyMesh);
  job->body           = dforce::transformPoints(toSpace(m_bodyMesh), posedCage(m_bodyMesh, bodyRest, nullptr, nullptr));
  job->bodyTris       = body.geo->dsonTris;
  job->bodyLimbs      = meshLimbs(m_scene, body);
  job->bodyRest       = dforce::transformPoints(toSpace(m_bodyMesh), bodyRest);

  std::vector<int> order;
  if(wornOrder)
    order = wornOrder();
  const auto placeWorn = [&](int mesh) {
    const auto it = std::find(order.begin(), order.end(), mesh);
    return it == order.end() ? int(order.size()) + mesh : int(it - order.begin());
  };

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
    item.restInSpace = dforce::transformPoints(to, item.rest);
    item.worn        = placeWorn(mesh);
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
    // inside out: closer to the skin first; two within a millimetre in the order worn
    for(Job::Item& item : job->items)
      item.clearance = restClearance(job->bodyRest, job->bodyTris, item.restInSpace);
    std::stable_sort(job->items.begin(), job->items.end(), [](const Job::Item& a, const Job::Item& b) {
      if(std::fabs(a.clearance - b.clearance) > 0.1)
        return a.clearance < b.clearance;
      return a.worn < b.worn;
    });
    std::vector<std::vector<float>> out;
    // what lies under the item being pushed: the body, then every item pushed before it
    std::vector<float>    under      = job->body;
    std::vector<uint32_t> underTris  = job->bodyTris;
    std::vector<int16_t>  underLimbs = job->bodyLimbs;
    for(const Job::Item& item : job->items)
    {
      std::vector<float> positions = item.posed;
      const Surface      surface{under, underTris};
      const auto         filter = limbFilter(underLimbs, item.limbs);
      const int          passes = std::clamp(item.settings.collision, 1, 8);
      const int          smooth = std::clamp(item.settings.smoothing, 1, 8);
      const float        tight  = std::clamp(item.settings.tight, 0.f, 1.f);
      const float        margin = kMargin + (kTightMargin - kMargin) * tight;
      if(tight > 0)
      {
        // all the way on, then back as far as the amount says, so the slider is even along its length
        std::vector<float> pulled = positions;
        for(const float reach : {kTightReach, kTightSecondReach})
        {
          const SurfaceBinding binding = bindToSurface(surface.positions, surface.tris, pulled, {}, filter ? &*filter : nullptr);
          pulled = smoothCorrection(pulled, pullIn(pulled, surface, binding, kTightMargin, 1.f, reach), item.tris, smooth);
        }
        for(size_t i = 0; i < positions.size(); i++)
          positions[i] += (pulled[i] - positions[i]) * tight;
      }
      for(int pass = 0; pass < passes; pass++)
      {
        const SurfaceBinding binding = bindToSurface(surface.positions, surface.tris, positions, {}, filter ? &*filter : nullptr);
        std::vector<float>   pushed  = pushOut(positions, surface, binding, margin, kDepth);
        if(std::equal(pushed.begin(), pushed.end(), positions.begin()))
          break;
        positions = smoothCorrection(positions, pushed, item.tris, smooth);
      }
      const uint32_t offset = uint32_t(under.size() / 3);
      under.insert(under.end(), positions.begin(), positions.end());
      for(uint32_t t : item.tris)
        underTris.push_back(t + offset);
      underLimbs.insert(underLimbs.end(), item.limbs.begin(), item.limbs.end());
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
