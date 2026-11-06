#include <figure/physics/cloth/model.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <unordered_map>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/geometry.hpp>
#include <figure/physics/collide/tri_grid.hpp>

namespace fv::physics::cloth {

namespace {

// Compliance (cm per dyne) at stiffness 0.5; (1 - s) / s scales it, so stiffness 1 is rigid.
// Stretch: the compliance alone lets a hanging metre of cloth stretch under 1% at the default 0.8;
// the one sweep per substep adds a sag of its own, a few percent on a metre.
constexpr float kStretchCompliance = 3e-5f;
constexpr float kShearCompliance   = 3e-3f;
constexpr float kBendCompliance    = 2e-3f;
// A vertex of partial strength s is tethered to its skinned position with compliance
// kAttachSag * (s / (1 - s))^4 per unit of inverse mass: under gravity it hangs g * kAttachSag
// * (s / (1 - s))^4 below that position whatever its mass. A band painted at 0.2 stays on the skin,
// 0.5 gives a centimetre and a half, and a garment whose surfaces are set to 0.7-0.9 throughout
// (common: the Kai Dress is 0.7-0.88) hangs from what it lies on and only leans on its skin, 44 cm of
// give at 0.7. Judged by its drape, with no measurement of what DAZ Studio does with a strength.
constexpr float kAttachSag = 1.5e-3f;
// Reach links: seed vertices about this many centimetres apart over the surface, each level linked
// to the seeds within kReachSpan of its spacing. On the Kai Dress with no tether at all, two levels
// keep on the body what falls off without them, for about 10% more time (2026-10-31:
// docs/improvements.md).
constexpr float kReachSpacing[] = {4.f, 16.f};
constexpr float kReachSpan      = 2.2f;
constexpr int   kReachMostLinks = 12;
// Mass floor, grams, and the area credited to a vertex no polygon touches, cm^2.
constexpr float kMinMass     = 1e-4f;
constexpr float kDefaultArea = 1.f;

float dist(const std::vector<float>& p, int a, int b)
{
  const float dx = p[b * 3] - p[a * 3], dy = p[b * 3 + 1] - p[a * 3 + 1], dz = p[b * 3 + 2] - p[a * 3 + 2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float compliance(float stiffness, float scale)
{
  const float s = std::min(1.f, std::max(stiffness, 0.01f));
  return ((1 - s) / s) * scale;
}

}  // namespace

namespace {

// Links seeds of each spacing to the seeds around them, as long as the shortest way between them over
// the mesh's stretch edges: its threads, which a shear doesn't lengthen, so a link leaves shear to the
// fabric, and over a surface curved in its rest shape the way is longer than the chord, so it can
// still unfold. As compliant as the softest edge on the way (a sheet's give over a patch doesn't grow
// with the patch, as a chain's would, but a soft panel on the way is still soft). A vertex takes the
// nearest few: a band that strands hang from would otherwise be linked to every strand.
void addReachLinks(Model& model)
{
  const size_t nv = size_t(model.nv);
  struct Step
  {
    int   to;
    float length, alpha;
  };
  std::vector<std::vector<Step>> around(nv);
  for(const Distance& e : model.edges)
    if(e.kind == DistanceKind::eStretch && model.dynamic[size_t(e.a)] && model.dynamic[size_t(e.b)])
    {
      around[size_t(e.a)].push_back({e.b, e.rest, e.alpha});
      around[size_t(e.b)].push_back({e.a, e.rest, e.alpha});
    }
  std::vector<float> dist(nv, INFINITY), softest(nv, 0.f);
  std::vector<int>   touched, links(nv, 0);
  using Open = std::pair<float, int>;
  const auto walk = [&](int from, float reach, const auto& visit) {
    std::priority_queue<Open, std::vector<Open>, std::greater<Open>> open;
    dist[size_t(from)]    = 0;
    softest[size_t(from)] = 0;
    touched.push_back(from);
    open.push({0.f, from});
    while(!open.empty())
    {
      const auto [d, v] = open.top();
      open.pop();
      if(d > dist[size_t(v)])
        continue;
      visit(v, d, softest[size_t(v)]);
      for(const Step& s : around[size_t(v)])
      {
        const float next = d + s.length;
        if(next > reach || next >= dist[size_t(s.to)])
          continue;
        if(dist[size_t(s.to)] == INFINITY)
          touched.push_back(s.to);
        dist[size_t(s.to)]    = next;
        softest[size_t(s.to)] = std::max(softest[size_t(v)], s.alpha);
        open.push({next, s.to});
      }
    }
    for(int t : touched)
      dist[size_t(t)] = INFINITY;
    touched.clear();
  };
  for(const float spacing : kReachSpacing)
  {
    std::vector<char> covered(nv, 0), isSeed(nv, 0);
    std::vector<int>  seeds;
    for(int i = 0; i < model.nv; i++)
      if(model.dynamic[size_t(i)] && !covered[size_t(i)])
      {
        isSeed[size_t(i)] = 1;
        seeds.push_back(i);
        walk(i, spacing, [&](int v, float, float) { covered[size_t(v)] = 1; });
      }
    for(int s : seeds)
      walk(s, spacing * kReachSpan, [&](int v, float d, float alpha) {
        // pinned to pinned holds nothing
        if(v <= s || !isSeed[size_t(v)] || (model.inv[size_t(s)] <= 0 && model.inv[size_t(v)] <= 0))
          return;
        if(links[size_t(s)] >= kReachMostLinks || links[size_t(v)] >= kReachMostLinks)
          return;
        links[size_t(s)]++;
        links[size_t(v)]++;
        model.edges.push_back({s, v, d, alpha, DistanceKind::eReach, 1.f, 0.f});
      });
  }
}

// The edges in blocks of the vertex range (physics/particles): an edge with a kinematic end goes with
// its free end, since the other is never written; the edges that hold length lead each list.
void partitionEdges(Model& model)
{
  model.blocks = physics::particles::partitionRows(
      model.nv, model.edges.size(), [&](size_t e) { return std::pair{model.edges[e].a, model.edges[e].b}; },
      [&](int v) { return !(model.inv[size_t(v)] > 0); }, [&](size_t e) { return holdsLength(model.edges[e].kind); });
}

}  // namespace

Model Model::build(const std::vector<Surface>& meshes, float air, bool reachLinks)
{
  timing::Scope tBuild("dforce: model");
  Model model;
  // the model's members by their plain names below
  auto& nv = model.nv;
  auto& base = model.base;
  auto& meshOf = model.meshOf;
  auto& rest = model.rest;
  auto& posed = model.posed;
  auto& strength = model.strength;
  auto& dampRate = model.dampRate;
  auto& friction = model.friction;
  auto& offset = model.offset;
  auto& grav = model.grav;
  auto& inv = model.inv;
  auto& layer = model.layer;
  auto& visible = model.visible;
  auto& collides = model.collides;
  auto& dynamic = model.dynamic;
  auto& edges = model.edges;
  auto& tris = model.tris;
  auto& staticTris = model.staticTris;
  auto& dynamicTris = model.dynamicTris;
  auto& allTris = model.allTris;
  auto& ringStart = model.ringStart;
  auto& ring = model.ring;
  auto& ring2Start = model.ring2Start;
  auto& ring2 = model.ring2;
  auto& kinematicVerts = model.kinematicVerts;
  auto& freeVerts = model.freeVerts;
  auto& softVerts = model.softVerts;
  auto& attAlpha = model.attAlpha;
  auto& maxOffset = model.maxOffset;
  auto& minOffset = model.minOffset;
  nv = 0;
  for(const Surface& m : meshes)
  {
    base.push_back(nv);
    nv += int(m.vertexCount());
  }
  rest.resize(size_t(nv) * 3);
  posed.resize(size_t(nv) * 3);
  grav.resize(size_t(nv) * 3);
  for(auto* v : {&strength, &dampRate, &friction, &offset, &inv})
    v->assign(size_t(nv), 0.f);
  layer.assign(size_t(nv), 0);
  meshOf.assign(size_t(nv), 0);
  visible.assign(size_t(nv), 0);
  collides.assign(size_t(nv), 0);
  dynamic.assign(size_t(nv), 0);
  std::vector<float> mass(size_t(nv), 0.f), density(size_t(nv), 180.f);
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const Surface& m   = meshes[mi];
    const int         o   = base[mi];
    const int         n   = int(m.vertexCount());
    const bool        sim = m.kind != Surface::Kind::eCollider;
    std::copy(m.rest.begin(), m.rest.end(), rest.begin() + o * 3);
    std::copy(m.posed.begin(), m.posed.end(), posed.begin() + o * 3);
    for(int v = 0; v < n; v++)
    {
      const int i   = o + v;
      strength[i]   = sim && m.visible[v] ? m.strength[v] : 0;
      density[i]    = m.density[v] > 1 ? m.density[v] : 180;
      const float d = std::max(30.f, density[i]);
      dampRate[i]   = m.damping[v] * 8 + air * (180 / d);
      friction[i]   = std::isfinite(m.friction[v]) ? m.friction[v] : 0.4f;
      offset[i]     = m.offset[v] > 0 ? m.offset[v] : 0.05f;
      layer[i]      = m.layer[v];
      visible[i]    = m.visible[v];
      collides[i]   = m.collide[v];
      dynamic[i]    = sim && m.visible[v] ? 1 : 0;
      meshOf[i]     = int(mi);
      grav[i * 3]   = m.gravity[0];
      grav[i * 3 + 1] = m.gravity[1];
      grav[i * 3 + 2] = m.gravity[2];
    }
  }
  const auto g = [&](size_t mi, int v) { return base[mi] + v; };
  std::unordered_map<uint64_t, int> seen;
  const auto add = [&](int a, int bb, float stiff, DistanceKind kind, float ratio = 1, float buckle = 0) {
    if(a == bb)
      return;
    const int      lo = std::min(a, bb), hi = std::max(a, bb);
    const uint64_t key = ((uint64_t(lo) << 32) | uint64_t(hi)) * 3 + uint64_t(kind);
    if(seen.count(key))
      return;
    const float span = dist(rest, a, bb);
    if(span < 1e-6f)
      return;
    float scale = 1, alpha = 0;
    if(kind == DistanceKind::eStretch)
    {
      const size_t ma = size_t(meshOf[a]), mb = size_t(meshOf[bb]);
      scale = (meshes[ma].contraction[size_t(a - base[ma])] + meshes[mb].contraction[size_t(bb - base[mb])]) * 0.5f;
      alpha = compliance(stiff, kStretchCompliance);
    }
    else if(kind == DistanceKind::eShear)
      alpha = compliance(stiff, kShearCompliance);
    else
      alpha = compliance(stiff, kBendCompliance);
    seen[key] = int(edges.size());
    edges.push_back({a, bb, span * scale, alpha, kind, ratio, buckle});
  };
  // a polygon edge as one polygon walks it, with the corners before and after it
  struct Side
  {
    int before, after;
  };
  std::unordered_map<uint64_t, std::vector<std::pair<std::array<int, 2>, Side>>> sides;
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const Surface& m = meshes[mi];
    const int         n = int(m.vertexCount());
    if(m.kind != Surface::Kind::eCollider)
    {
      sides.clear();
      for(size_t p = 0; p + 1 < m.polyStart.size(); p++)
      {
        const int s = m.polyStart[p], e = m.polyStart[p + 1];
        const int cnt = e - s;
        if(cnt < 2)
          continue;
        // mass: the polygon's area shared among its corners
        double area = 0;
        if(cnt >= 3)
        {
          const int v0 = g(mi, m.polyVerts[s]);
          for(int k = 1; k + 1 < cnt; k++)
          {
            const int    v1 = g(mi, m.polyVerts[s + k]), v2 = g(mi, m.polyVerts[s + k + 1]);
            const double ax = rest[v1 * 3] - rest[v0 * 3], ay = rest[v1 * 3 + 1] - rest[v0 * 3 + 1], az = rest[v1 * 3 + 2] - rest[v0 * 3 + 2];
            const double bx = rest[v2 * 3] - rest[v0 * 3], by = rest[v2 * 3 + 1] - rest[v0 * 3 + 1], bz = rest[v2 * 3 + 2] - rest[v0 * 3 + 2];
            const double cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
            area += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
          }
        }
        for(int k = 0; k < cnt; k++)
        {
          const int va = m.polyVerts[s + k], vb = m.polyVerts[s + (k + 1) % cnt];
          const int a = g(mi, va), bb = g(mi, vb);
          mass[a] += float(area / cnt * 1e-4 * density[a]);
          add(a, bb, (m.stretch[va] + m.stretch[vb]) * 0.5f, DistanceKind::eStretch);
          if(cnt >= 3)
          {
            const int before = g(mi, m.polyVerts[s + (k + cnt - 1) % cnt]), after = g(mi, m.polyVerts[s + (k + 2) % cnt]);
            const uint64_t key = (uint64_t(std::min(a, bb)) << 32) | uint64_t(std::max(a, bb));
            sides[key].push_back({{a, bb}, {before, after}});
          }
        }
        if(cnt == 4)
        {
          const int   v0 = m.polyVerts[s], v1 = m.polyVerts[s + 1], v2 = m.polyVerts[s + 2], v3 = m.polyVerts[s + 3];
          const float sh = (m.shear[v0] + m.shear[v1] + m.shear[v2] + m.shear[v3]) * 0.25f;
          add(g(mi, v0), g(mi, v2), sh, DistanceKind::eShear);
          add(g(mi, v1), g(mi, v3), sh, DistanceKind::eShear);
        }
      }
      // bend: across an edge two polygons share, the corner next to each endpoint on one side is
      // held at its rest distance from the corner next to the same endpoint on the other side
      for(const auto& [key, occ] : sides)
      {
        if(occ.size() != 2)
          continue;
        const auto& [edgeP, sideP] = occ[0];
        const auto& [edgeQ, sideQ] = occ[1];
        // the corner next to vertex u of the edge, in a polygon that walks the edge as (e0 -> e1)
        const auto nextTo = [](int u, const std::array<int, 2>& e, const Side& sd) { return u == e[0] ? sd.before : sd.after; };
        for(int u : edgeP)
        {
          const int p = nextTo(u, edgeP, sideP), q = nextTo(u, edgeQ, sideQ);
          const int lp = p - base[mi], lq = q - base[mi], le0 = edgeP[0] - base[mi], le1 = edgeP[1] - base[mi];
          const float stiff = (m.bend[lp] + m.bend[lq] + m.bend[le0] + m.bend[le1]) * 0.25f;
          const float buck  = (m.buckling[lp] + m.buckling[lq] + m.buckling[le0] + m.buckling[le1]) * 0.25f;
          const float ratio = (m.bucklingRatio[lp] + m.bucklingRatio[lq] + m.bucklingRatio[le0] + m.bucklingRatio[le1]) * 0.25f;
          add(p, q, stiff, DistanceKind::eBend, ratio, buck);
        }
      }
    }
    if(m.kind == Surface::Kind::eAddon)
      continue;
    for(size_t t = 0; t + 2 < m.tris.size(); t += 3)
    {
      const uint32_t ia = m.tris[t], ib = m.tris[t + 1], ic = m.tris[t + 2];
      if(ia >= uint32_t(n) || ib >= uint32_t(n) || ic >= uint32_t(n))
        continue;
      const int   a = g(mi, int(ia)), bb = g(mi, int(ib)), c = g(mi, int(ic));
      const int   lay = int(std::lround((m.layer[ia] + m.layer[ib] + m.layer[ic]) / 3.0));
      const float off = (m.offset[ia] + m.offset[ib] + m.offset[ic]) / 3;
      const float frRaw = (m.friction[ia] + m.friction[ib] + m.friction[ic]) / 3;
      const float fr    = std::isfinite(frRaw) ? frRaw : 0.4f;
      const bool  self  = m.selfCollide[ia] || m.selfCollide[ib] || m.selfCollide[ic];
      const bool  shown = m.visible[ia] || m.visible[ib] || m.visible[ic] || m.kind == Surface::Kind::eCollider;
      // a vertex on a colliding and a non-colliding surface collides, so the face needs all three
      if(shown && m.collide[ia] && m.collide[ib] && m.collide[ic])
        tris.push_back({int(mi), a, bb, c, lay, off > 0 ? off : 0.05f, fr, self, m.kind != Surface::Kind::eCollider});
    }
  }
  for(size_t mi = 0; mi < meshes.size(); mi++)
    for(const Surface::Capsule& c : meshes[mi].capsules)
      if(c.a < meshes[mi].vertexCount() && c.b < meshes[mi].vertexCount() && c.radius > 0)
        model.capsules.push_back({g(mi, int(c.a)), g(mi, int(c.b)), c.radius});
  attAlpha.assign(size_t(nv), 0.f);
  for(int i = 0; i < nv; i++)
  {
    if(dynamic[i] && strength[i] > 1e-4f)
    {
      if(mass[i] <= 0)
        mass[i] = kDefaultArea * 1e-4f * density[i];
      inv[i] = 1.f / std::max(kMinMass, mass[i]);
      freeVerts.push_back(i);
      if(strength[i] < 1 - 1e-3f)
      {
        const float s = std::clamp(strength[i], 1e-4f, 0.999f);
        const float r = s / (1 - s);
        attAlpha[i]   = kAttachSag * inv[i] * r * r * r * r;
        softVerts.push_back(i);
      }
    }
    else
      kinematicVerts.push_back(i);
  }
  maxOffset = 0;
  minOffset = INFINITY;
  for(const Face& t : tris)
    maxOffset = std::max(maxOffset, t.offset);
  for(int i : freeVerts)
  {
    maxOffset = std::max(maxOffset, offset[i]);
    minOffset = std::min(minOffset, offset[i]);
  }
  if(!std::isfinite(minOffset) || minOffset < 0.05f)
    minOffset = 0.05f;
  for(size_t t = 0; t < tris.size(); t++)
  {
    allTris.push_back(int(t));
    (tris[t].dynamic ? dynamicTris : staticTris).push_back(int(t));
  }
  {
    // two faces share an edge when they share its two vertices (the global index includes the mesh)
    std::unordered_map<uint64_t, int> open;
    open.reserve(tris.size() * 2);
    model.triNext.assign(tris.size() * 3, -1);
    for(size_t t = 0; t < tris.size(); t++)
    {
      const int v[3] = {tris[t].a, tris[t].b, tris[t].c};
      for(int k = 0; k < 3; k++)
      {
        const uint32_t lo = uint32_t(std::min(v[k], v[(k + 1) % 3])), hi = uint32_t(std::max(v[k], v[(k + 1) % 3]));
        const uint64_t key = uint64_t(lo) << 32 | hi;
        const auto [it, fresh] = open.try_emplace(key, int(t * 3 + size_t(k)));
        if(fresh)
          continue;
        const int other = it->second;
        if(model.triNext[size_t(other)] < 0)
        {
          model.triNext[size_t(other)]      = int(t);
          model.triNext[t * 3 + size_t(k)] = other / 3;
        }
      }
    }
  }
  if(reachLinks)
  {
    timing::Scope t("dforce: reach links");
    addReachLinks(model);
  }
  timing::Scope tRings("dforce: rings");
  ringStart.assign(size_t(nv) + 1, 0);
  for(const Distance& e : edges)
    if(e.kind == DistanceKind::eStretch)
    {
      ringStart[size_t(e.a) + 1]++;
      ringStart[size_t(e.b) + 1]++;
    }
  for(int i = 0; i < nv; i++)
    ringStart[size_t(i) + 1] += ringStart[size_t(i)];
  ring.resize(size_t(ringStart[size_t(nv)]));
  std::vector<int> fill(ringStart.begin(), ringStart.end() - 1);
  for(const Distance& e : edges)
    if(e.kind == DistanceKind::eStretch)
    {
      ring[size_t(fill[size_t(e.a)]++)] = e.b;
      ring[size_t(fill[size_t(e.b)]++)] = e.a;
    }
  ring2Start.assign(size_t(nv) + 1, 0);
  std::vector<int> two;
  for(int i = 0; i < nv; i++)
  {
    two.clear();
    for(int k = ringStart[size_t(i)]; k < ringStart[size_t(i) + 1]; k++)
    {
      const int j = ring[size_t(k)];
      two.push_back(j);
      for(int l = ringStart[size_t(j)]; l < ringStart[size_t(j) + 1]; l++)
        two.push_back(ring[size_t(l)]);
    }
    std::sort(two.begin(), two.end());
    two.erase(std::unique(two.begin(), two.end()), two.end());
    ring2.insert(ring2.end(), two.begin(), two.end());
    ring2Start[size_t(i) + 1] = int(ring2.size());
  }
  partitionEdges(model);
  return model;
}

void separateStart(const Model& model, std::vector<float>& x)
{
  timing::Scope t("dforce: separate the start");
  const auto& tris     = model.tris;
  const auto& meshOf   = model.meshOf;
  const auto& layer    = model.layer;
  const auto& offset   = model.offset;
  const auto& collides = model.collides;
  // A vertex looks only at faces of other meshes, on layers under it, so what none can look at stays out of the
  // grid: on a hair, thousands of its own faces lie around every vertex, and passing over them all took 40 s.
  int top = std::numeric_limits<int>::min(), second = top, topMesh = -1;
  for(int i : model.freeVerts)
  {
    if(!collides[i])
      continue;
    if(layer[i] > top)
    {
      if(meshOf[i] != topMesh)
        second = top;
      top     = layer[i];
      topMesh = meshOf[i];
    }
    else if(meshOf[i] != topMesh && layer[i] > second)
      second = layer[i];
  }
  std::vector<int> underSomething;
  for(int t : model.allTris)
    if(tris[size_t(t)].layer < (tris[size_t(t)].mesh == topMesh ? second : top))
      underSomething.push_back(t);
  TriGrid grid;
  grid.build(tris, underSomething, x, x, 0.f);
  // The lowest layer first, so what lies on a moved cloth sees where it went. A layer's vertices look only at what
  // is under them, so they go side by side, and their moves land together when the layer is done.
  std::map<int, std::vector<int>> byLayer;
  for(int i : model.freeVerts)
    if(collides[i])
      byLayer[layer[i]].push_back(i);
  for(const auto& [level, verts] : byLayer)
  {
    std::vector<float> moves(verts.size() * 3, 0.f);
    jobs::parallelFor(verts.size(), 4096, [&](size_t begin, size_t end) {
      std::vector<uint32_t> seen(tris.size(), 0);
      uint32_t              stamp = 0;
      for(size_t vi = begin; vi < end; vi++)
      {
        const int   i  = verts[vi];
        const float qx = x[i * 3], qy = x[i * 3 + 1], qz = x[i * 3 + 2];
        const float lo[3] = {qx - kDepth, qy - kDepth, qz - kDepth}, hi[3] = {qx + kDepth, qy + kDepth, qz + kDepth};
        stamp++;
        int   best = -1;
        float bestDist = kDepth, bestSigned = 0;
        V3    bestN{};
        grid.query(lo, hi, [&](const TriGrid::Item& item) {
          const int t = item.tri;
          if(item.mesh == meshOf[i] || seen[size_t(t)] == stamp)
            return;
          seen[size_t(t)] = stamp;
          const Face& tri  = tris[size_t(t)];
          if(tri.layer >= level)
            return;
          for(int c = 0; c < 3; c++)
          {
            const float blo = std::min(x[tri.a * 3 + c], std::min(x[tri.b * 3 + c], x[tri.c * 3 + c]));
            const float bhi = std::max(x[tri.a * 3 + c], std::max(x[tri.b * 3 + c], x[tri.c * 3 + c]));
            const float q   = c == 0 ? qx : c == 1 ? qy : qz;
            if(q < blo - bestDist || q > bhi + bestDist)
              return;
          }
          V3 n;
          if(!triNormal(x, tri.a, tri.b, tri.c, n))
            return;
          const Closest c = closest(qx, qy, qz, x, tri.a, tri.b, tri.c, n);
          if(c.dist >= bestDist)
            return;
          best       = t;
          bestDist   = c.dist;
          bestSigned = c.signedDistance;
          bestN      = n;
        });
        if(best < 0)
          continue;
        const float gap = std::max(tris[size_t(best)].offset, offset[i]);
        if(bestSigned >= gap)
          continue;
        const float push = gap - bestSigned;
        for(int k = 0; k < 3; k++)
          moves[vi * 3 + size_t(k)] = bestN[size_t(k)] * push;
      }
    });
    for(size_t vi = 0; vi < verts.size(); vi++)
      for(int k = 0; k < 3; k++)
        x[verts[vi] * 3 + k] += moves[vi * 3 + size_t(k)];
  }
}

}  // namespace fv::physics::cloth
