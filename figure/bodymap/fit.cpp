#include <figure/bodymap/fit.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>

#include <figure/bodymap/frames.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/physics/collide/surface_index.hpp>
#include <figure/wardrobe/fit/bases.hpp>

namespace fv::bodymap {

namespace {

using R = rig::Region;

Vec3 at(const std::vector<float>& p, size_t v)
{
  return {p[v * 3], p[v * 3 + 1], p[v * 3 + 2]};
}

Vec3 crossOf(const Vec3& a, const Vec3& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// A joint band on a body: where a region meets the one above (the knee, where thigh meets shin), its
// vertices with their shares of its area, and its middle.
struct Band
{
  R                                        region   = R::eNone;
  rig::Landmark                            landmark = rig::Landmark::eNone;
  std::vector<std::pair<uint32_t, double>> members;
  Vec3                                     middle;
};
using BandKey = std::pair<RegionMask, RegionMask>;  // the region's bit and the one above's

std::map<BandKey, Band> bandsOf(const Body& body)
{
  std::map<BandKey, Band> bands;
  for(int r = int(R::ePelvis); r <= int(R::ePinkyToe); r++)
  {
    const R up = regionAbove(R(r));
    if(up == R::eNone)
      continue;
    for(const char side : {'l', 'r'})
    {
      if(!regionSided(R(r)) && side == 'r')
        continue;
      const char s = regionSided(R(r)) ? side : 0;
      Band& b    = bands[{regionBit(R(r), s), regionBit(up, regionSided(up) ? s : 0)}];
      b.region   = R(r);
      b.landmark = regionLandmark(R(r));
    }
  }
  const std::vector<double> area = vertexAreas(body);
  for(size_t v = 0; v < area.size() && v < body.regions.size(); v++)
    if(const RegionMask m = body.regions[v]; m && area[v] > 0)
      for(auto& [key, b] : bands)
        if((m & key.first) && (m & key.second))
          b.members.emplace_back(uint32_t(v), area[v]);
  for(auto it = bands.begin(); it != bands.end();)
  {
    double total = 0;
    for(const auto& [v, a] : it->second.members)
      total += a;
    if(total <= 0)
    {
      it = bands.erase(it);
      continue;
    }
    for(auto& [v, a] : it->second.members)
    {
      a /= total;
      it->second.middle = it->second.middle + at(body.positions, v) * a;
    }
    ++it;
  }
  return bands;
}

// In place: A = L L^T for a symmetric positive definite n x n A (row major), L in its lower half.
bool cholesky(std::vector<double>& a, size_t n)
{
  for(size_t j = 0; j < n; j++)
  {
    double d = a[j * n + j];
    for(size_t k = 0; k < j; k++)
      d -= a[j * n + k] * a[j * n + k];
    if(d <= 0)
      return false;
    d              = std::sqrt(d);
    a[j * n + j]   = d;
    const double inv = 1 / d;
    jobs::parallelFor(n - j - 1, 64, [&](size_t begin, size_t end) {
      for(size_t i = j + 1 + begin; i < j + 1 + end; i++)
      {
        double s = a[i * n + j];
        for(size_t k = 0; k < j; k++)
          s -= a[i * n + k] * a[j * n + k];
        a[i * n + j] = s * inv;
      }
    });
  }
  return true;
}

// Solves L L^T x = b for each of b's three columns, in place.
void choleskySolve(const std::vector<double>& l, size_t n, std::vector<Vec3>& b)
{
  for(size_t i = 0; i < n; i++)
  {
    Vec3 s = b[i];
    for(size_t k = 0; k < i; k++)
      s = s - b[k] * l[i * n + k];
    b[i] = s * (1 / l[i * n + i]);
  }
  for(size_t i = n; i-- > 0;)
  {
    Vec3 s = b[i];
    for(size_t k = i + 1; k < n; k++)
      s = s - b[k] * l[k * n + i];
    b[i] = s * (1 / l[i * n + i]);
  }
}

// The deformation graph over the reference's skin.
struct Graph
{
  std::vector<Vec3>                    node;   // where each node sits
  std::vector<RegionMask>              mask;   // its regions
  std::vector<std::array<uint32_t, 4>> near;   // per vertex: the nodes it follows...
  std::vector<std::array<double, 4>>   weight; // ...and how much
  std::vector<std::vector<uint32_t>>   linked; // per node: its neighbours
};

constexpr size_t kPerVertex = 4, kLinks = 8;

Graph graphOf(const Body& body, const std::vector<Vec3>& normals, size_t count)
{
  Graph        g;
  const size_t nv = body.positions.size() / 3;
  // the nodes: spread over the outer skin, each the vertex farthest from those before it
  std::vector<double> far(nv, -1);
  size_t              first = nv;
  for(size_t v = 0; v < nv; v++)
    if(dot(normals[v], normals[v]) > 0)
    {
      far[v] = INFINITY;
      first  = std::min(first, v);
    }
  for(size_t pick = first; pick < nv && g.node.size() < count;)
  {
    const Vec3 p = at(body.positions, pick);
    g.node.push_back(p);
    g.mask.push_back(pick < body.regions.size() ? body.regions[pick] : 0);
    far[pick]   = -1;
    size_t next = nv;
    double most = 0;
    for(size_t v = 0; v < nv; v++)
      if(far[v] >= 0)
      {
        const Vec3 d = at(body.positions, v) - p;
        far[v]       = std::min(far[v], dot(d, d));
        if(far[v] > most)
          most = far[v], next = v;
      }
    pick = next;
  }
  const size_t n = g.node.size();
  // a vertex follows nodes in its regions or next to them; one with none, any
  std::unordered_map<RegionMask, RegionMask> widened;
  for(const RegionMask m : body.regions)
    if(!widened.count(m))
      widened[m] = withNeighbours(m);
  const auto allowed = [&](RegionMask vertex, RegionMask node) { return !vertex || !node || (widened.at(vertex) & node); };
  g.near.assign(nv, {});
  g.weight.assign(nv, {});
  jobs::parallelFor(nv, 256, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
    {
      const RegionMask m = v < body.regions.size() ? body.regions[v] : 0;
      const Vec3       p = at(body.positions, v);
      // the kPerVertex + 1 nearest allowed nodes (any, when too few are allowed)
      for(int pass = 0; pass < 2; pass++)
      {
        std::array<std::pair<double, uint32_t>, kPerVertex + 1> best;
        best.fill({INFINITY, 0});
        for(uint32_t j = 0; j < n; j++)
        {
          if(pass == 0 && !allowed(m, g.mask[j]))
            continue;
          const Vec3   d  = g.node[j] - p;
          const double d2 = dot(d, d);
          if(d2 >= best.back().first)
            continue;
          best.back() = {d2, j};
          for(size_t k = kPerVertex; k > 0 && best[k].first < best[k - 1].first; k--)
            std::swap(best[k], best[k - 1]);
        }
        if(best[1].first == INFINITY && pass == 0)
          continue;
        const double reach = std::sqrt(best[kPerVertex].first == INFINITY ? best[kPerVertex - 1].first * 4 + 1e-6 : best[kPerVertex].first);
        double       sum   = 0;
        for(size_t k = 0; k < kPerVertex; k++)
        {
          const double w     = best[k].first == INFINITY ? 0 : std::pow(std::max(0.0, 1 - std::sqrt(best[k].first) / std::max(reach, 1e-9)), 2);
          g.near[v][k]       = best[k].second;
          g.weight[v][k]     = w;
          sum += w;
        }
        for(size_t k = 0; k < kPerVertex; k++)
          g.weight[v][k] = sum > 0 ? g.weight[v][k] / sum : (k == 0 ? 1 : 0);
        break;
      }
    }
  });
  // the links: each node's nearest allowed nodes, both ways
  std::vector<std::set<uint32_t>> links(n);
  for(uint32_t j = 0; j < n; j++)
  {
    std::vector<std::pair<double, uint32_t>> by;
    for(uint32_t k = 0; k < n; k++)
      if(k != j && allowed(g.mask[j], g.mask[k]))
      {
        const Vec3 d = g.node[k] - g.node[j];
        by.emplace_back(dot(d, d), k);
      }
    const size_t keep = std::min(kLinks, by.size());
    std::partial_sort(by.begin(), by.begin() + ptrdiff_t(keep), by.end());
    for(size_t i = 0; i < keep; i++)
    {
      links[j].insert(by[i].second);
      links[by[i].second].insert(j);
    }
  }
  g.linked.resize(n);
  for(uint32_t j = 0; j < n; j++)
    g.linked[j].assign(links[j].begin(), links[j].end());
  return g;
}

}  // namespace

FitResult fitReference(const wardrobe::BaseCage& cage, const Body& reference, const Body& source, const FitOptions& options)
{
  FitResult    out;
  const size_t nv = reference.positions.size() / 3;
  out.positions   = reference.positions;
  if(!nv || source.tris.empty())
    return out;

  // 1. one scale and shift, from the joint bands both bodies have that a pose moves little: the torso's,
  // the neck's and head's, the arms' roots and the legs' (a T-posed elbow lies far from a lowered one; a
  // wider stance moves a knee a little sideways)
  {
    const auto        from = bandsOf(reference), to = bandsOf(source);
    std::vector<Vec3> a, b;
    for(const auto& [key, band] : from)
      if(const auto it = to.find(key); it != to.end())
        switch(band.region)
        {
          case R::eAbdomen:
          case R::eChest:
          case R::eBreast:
          case R::eNeck:
          case R::eHead:
          case R::eCollar:
          case R::eUpperArm:
          case R::eThigh:
          case R::eShin:
          case R::eFoot:
            a.push_back(band.middle), b.push_back(it->second.middle);
            break;
          default:
            break;
        }
    Vec3 ma, mb;
    for(size_t i = 0; i < a.size(); i++)
      ma = ma + a[i] * (1.0 / double(a.size())), mb = mb + b[i] * (1.0 / double(a.size()));
    double num = 0, den = 0;
    for(size_t i = 0; i < a.size(); i++)
      num += dot(a[i] - ma, b[i] - mb), den += dot(a[i] - ma, a[i] - ma);
    out.scale = a.size() >= 2 && den > 1 ? std::clamp(num / den, 0.7, 1.4) : 1.0;
    out.shift = a.empty() ? Vec3{} : mb - ma * out.scale;
    for(size_t v = 0; v < nv; v++)
    {
      const Vec3 p                = at(reference.positions, v) * out.scale + out.shift;
      out.positions[v * 3]        = float(p.x);
      out.positions[v * 3 + 1]    = float(p.y);
      out.positions[v * 3 + 2]    = float(p.z);
    }
  }

  // 2. the limbs into the source's rest
  out.positions = intoRest(cage, out.positions, source, &out.turns, true);

  // 3. the embedded deformation
  Body posed      = reference;
  posed.positions = out.positions;
  const std::vector<Vec3>   restNormals = outerNormals(posed);
  const Graph               graph       = graphOf(posed, restNormals, size_t(std::max(options.nodes, 8)));
  const size_t              n           = graph.node.size();
  out.nodes                             = n;
  const std::vector<double> area        = vertexAreas(posed);
  double                    meanArea    = 0;
  size_t                    outer       = 0;
  for(size_t v = 0; v < nv; v++)
    if(area[v] > 0)
      meanArea += area[v], outer++;
  meanArea = outer ? meanArea / double(outer) : 1;

  const physics::SurfaceIndex index(source.positions, source.tris);
  std::vector<RegionMask>     triRegions(source.tris.size() / 3, 0);
  std::vector<Vec3>           faces(source.tris.size() / 3);
  for(size_t t = 0; t < triRegions.size(); t++)
  {
    for(size_t j = 0; j < 3; j++)
      triRegions[t] |= source.tris[t * 3 + j] < source.regions.size() ? source.regions[source.tris[t * 3 + j]] : 0;
    const Vec3 a = at(source.positions, source.tris[t * 3]);
    faces[t]     = normalize(crossOf(at(source.positions, source.tris[t * 3 + 1]) - a, at(source.positions, source.tris[t * 3 + 2]) - a));
  }
  const std::vector<Vec3> sourceNormals = outerNormals(source);
  // the source's cut: a closest point on it lies past what the body has
  const CutEdges cut(source);
  std::unordered_map<RegionMask, RegionMask> widened;
  for(const RegionMask m : reference.regions)
    if(!widened.count(m))
      widened[m] = withNeighbours(m);

  // the landmarks: each limb joint's band on the reference held to the middle of the source's, as the
  // nodes carry its vertices (a node's share of a band: its vertices' weights for it, by their shares)
  struct Held
  {
    std::vector<std::pair<uint32_t, double>> members;  // the band's reference vertices and their shares
    std::map<uint32_t, double>               nodes;    // the nodes' shares of the band
    Vec3                                     target;
  };
  std::vector<Held> held;
  {
    const auto to = bandsOf(source);
    for(auto& [key, band] : bandsOf(posed))
      if(const auto it = to.find(key); it != to.end() && band.landmark == rig::Landmark::eFull)
      {
        Held h{band.members, {}, it->second.middle};
        for(const auto& [v, share] : band.members)
          for(size_t k = 0; k < kPerVertex; k++)
            h.nodes[graph.near[v][k]] += share * graph.weight[v][k];
        held.push_back(std::move(h));
      }
  }
  out.landmarks = held.size();

  std::vector<Quat> turn(n);
  std::vector<Vec3> move(n);
  std::vector<float> current = out.positions;
  const auto        deform  = [&] {
    jobs::parallelFor(nv, 1024, [&](size_t begin, size_t end) {
      for(size_t v = begin; v < end; v++)
      {
        const Vec3 p = at(out.positions, v);
        Vec3       q;
        for(size_t k = 0; k < kPerVertex; k++)
        {
          const uint32_t j = graph.near[v][k];
          q                = q + (rotate(p - graph.node[j], turn[j]) + graph.node[j] + move[j]) * graph.weight[v][k];
        }
        current[v * 3]     = float(q.x);
        current[v * 3 + 1] = float(q.y);
        current[v * 3 + 2] = float(q.z);
      }
    });
  };
  const double             cosMax = std::cos(options.maxAngle * kDeg);
  const double             reach2 = options.reach * options.reach;
  std::vector<Vec3>        target(nv);
  std::vector<uint8_t>     pulled(nv, 0);
  std::vector<double>      gap(nv, 0);
  std::vector<double>      matrix(n * n);
  // each outer vertex's pull, at the reference as it now is: toward the plane of the source's closest
  // skin in its regions (or next to them), facing its way, within reach and not past the source's cut
  const auto pullAll = [&](double within2) {
    posed.positions                = current;
    const std::vector<Vec3> normal = outerNormals(posed);
    jobs::parallelFor(nv, 256, [&](size_t begin, size_t end) {
      physics::Stamps& stamps = physics::threadStamps();
      for(size_t v = begin; v < end; v++)
      {
        pulled[v] = 0;
        if(area[v] <= 0 || dot(normal[v], normal[v]) == 0)
          continue;
        const RegionMask m      = v < reference.regions.size() ? widened.at(reference.regions[v]) : 0;
        const Vec3       p      = at(current, v);
        const auto       accept = [&](size_t t) {
          if(t < source.outer.size() && !source.outer[t])
            return false;
          if(m && triRegions[t] && !(triRegions[t] & m))
            return false;
          return dot(faces[t], normal[v]) >= cosMax;
        };
        const int32_t id  = stamps.next(index.triangles());
        const auto    hit = index.nearest(p.x, p.y, p.z, accept, within2, stamps.stamp, id);
        if(hit.tri < 0)
          continue;
        const double w[3] = {hit.u, hit.v, hit.w};
        Vec3         c, nc;
        for(size_t j = 0; j < 3; j++)
        {
          const uint32_t corner = source.tris[size_t(hit.tri) * 3 + j];
          c                     = c + at(source.positions, corner) * w[j];
          nc                    = nc + sourceNormals[corner] * w[j];
        }
        nc = length(nc) > 1e-9 ? normalize(nc) : faces[size_t(hit.tri)];
        // on the source's cut: a vertex just over it still lies on the skin's going on; one further along
        // lies past what the source has
        if(cut.on(size_t(hit.tri), w))
        {
          const Vec3 off = c - p;
          if(length(off - nc * dot(nc, off)) > options.overCut)
            continue;
        }
        target[v] = p + nc * dot(nc, c - p);
        gap[v]    = length(c - p);
        pulled[v] = 1;
      }
    });
  };
  // and the other way: each source vertex pulls the reference's closest skin (in its regions or next to
  // them, facing its way, within reach) onto the plane of its own, so a part the reference's vertices
  // don't reach (a bigger bust's underside) still has the reference laid on it
  struct Back
  {
    uint32_t corner[3];
    double   share[3];
    Vec3     target;
    bool     hit = false;
  };
  const size_t              ns         = source.positions.size() / 3;
  const std::vector<double> sourceArea = vertexAreas(source);
  std::vector<Back>         back(ns);
  std::unordered_map<RegionMask, RegionMask> sourceWidened;
  for(const RegionMask m : source.regions)
    if(!sourceWidened.count(m))
      sourceWidened[m] = withNeighbours(m);
  std::vector<RegionMask> refTriRegions(reference.tris.size() / 3, 0);
  for(size_t t = 0; t < refTriRegions.size(); t++)
    for(size_t j = 0; j < 3; j++)
      refTriRegions[t] |= reference.tris[t * 3 + j] < reference.regions.size() ? reference.regions[reference.tris[t * 3 + j]] : 0;
  const auto pullBack = [&](double within2) {
    const physics::SurfaceIndex onto(current, reference.tris);
    std::vector<Vec3>           face(reference.tris.size() / 3);
    for(size_t t = 0; t < face.size(); t++)
    {
      const Vec3 a = at(current, reference.tris[t * 3]);
      face[t]      = normalize(crossOf(at(current, reference.tris[t * 3 + 1]) - a, at(current, reference.tris[t * 3 + 2]) - a));
    }
    jobs::parallelFor(ns, 256, [&](size_t begin, size_t end) {
      physics::Stamps& stamps = physics::threadStamps();
      for(size_t s = begin; s < end; s++)
      {
        back[s].hit = false;
        const Vec3& ns_ = sourceNormals[s];
        if(sourceArea[s] <= 0 || dot(ns_, ns_) == 0)
          continue;
        const RegionMask m      = s < source.regions.size() ? sourceWidened.at(source.regions[s]) : 0;
        const auto       accept = [&](size_t t) {
          if(t < reference.outer.size() && !reference.outer[t])
            return false;
          if(m && refTriRegions[t] && !(refTriRegions[t] & m))
            return false;
          return dot(face[t], ns_) >= cosMax;
        };
        const Vec3    p   = at(source.positions, s);
        const int32_t id  = stamps.next(onto.triangles());
        const auto    hit = onto.nearest(p.x, p.y, p.z, accept, within2, stamps.stamp, id);
        if(hit.tri < 0)
          continue;
        Back&        b    = back[s];
        const double w[3] = {hit.u, hit.v, hit.w};
        Vec3         q;
        for(size_t j = 0; j < 3; j++)
        {
          b.corner[j] = reference.tris[size_t(hit.tri) * 3 + j];
          b.share[j]  = w[j];
          q           = q + at(current, b.corner[j]) * w[j];
        }
        b.target = q + ns_ * dot(ns_, p - q);
        b.hit    = true;
      }
    });
  };
  // a source pull's nodes and their shares (its three corners' nodes, by the corners' shares)
  const auto backNodes = [&](const Back& b) {
    std::vector<std::pair<uint32_t, double>> nodes;
    for(size_t i = 0; i < 3; i++)
      for(size_t k = 0; k < kPerVertex; k++)
      {
        const uint32_t j = graph.near[b.corner[i]][k];
        const double   c = b.share[i] * graph.weight[b.corner[i]][k];
        auto           it = std::find_if(nodes.begin(), nodes.end(), [&](const auto& e) { return e.first == j; });
        if(it == nodes.end())
          nodes.emplace_back(j, c);
        else
          it->second += c;
      }
    return nodes;
  };

  const int rounds = std::max(options.rounds, 1);
  for(int round = 0; round < rounds; round++)
  {
    const double stiff = options.stiffFrom * std::pow(options.stiffTo / options.stiffFrom, rounds > 1 ? double(round) / double(rounds - 1) : 1.0);
    deform();
    pullAll(reach2);
    pullBack(reach2);
    // the moves: the pull at each vertex against the graph's stiffness, one solve for all three axes
    std::fill(matrix.begin(), matrix.end(), 0.0);
    double data = 0;
    for(size_t v = 0; v < nv; v++)
    {
      if(!pulled[v])
        continue;
      const double a = area[v] / meanArea;
      for(size_t i = 0; i < kPerVertex; i++)
        for(size_t k = 0; k < kPerVertex; k++)
          matrix[graph.near[v][i] * n + graph.near[v][k]] += a * graph.weight[v][i] * graph.weight[v][k];
      data += a;
    }
    for(size_t s = 0; s < ns; s++)
    {
      if(!back[s].hit)
        continue;
      const double a     = sourceArea[s] / meanArea;
      const auto   nodes = backNodes(back[s]);
      for(const auto& [j, cj] : nodes)
        for(const auto& [k, ck] : nodes)
          matrix[j * n + k] += a * cj * ck;
      data += a;
    }
    size_t links = 0;
    for(const auto& l : graph.linked)
      links += l.size();
    const double lambda = stiff * std::max(data, 1.0) / double(std::max<size_t>(links, 1));
    for(uint32_t j = 0; j < n; j++)
      for(const uint32_t k : graph.linked[j])
      {
        matrix[j * n + j] += lambda;
        matrix[k * n + k] += lambda;
        matrix[j * n + k] -= lambda;
        matrix[k * n + j] -= lambda;
      }
    // each landmark weighs as a share of all the pull
    const double hold = options.landmarkWeight * std::max(data, 1.0);
    for(const Held& h : held)
      for(const auto& [j, a] : h.nodes)
        for(const auto& [k, b] : h.nodes)
          matrix[j * n + k] += hold * a * b;
    for(size_t j = 0; j < n; j++)
      matrix[j * n + j] += 1e-6 * (lambda + std::max(data, 1.0) / double(n));
    if(!cholesky(matrix, n))
      break;
    // the right side at the nodes' turns: each pull less where the turns alone take its vertex, and each
    // link as its node's turn carries it
    const auto rightSide = [&] {
      std::vector<Vec3> b(n);
      for(size_t v = 0; v < nv; v++)
      {
        if(!pulled[v])
          continue;
        const double a = area[v] / meanArea;
        const Vec3   p = at(out.positions, v);
        Vec3         rest;
        for(size_t k = 0; k < kPerVertex; k++)
        {
          const uint32_t j = graph.near[v][k];
          rest             = rest + (rotate(p - graph.node[j], turn[j]) + graph.node[j]) * graph.weight[v][k];
        }
        const Vec3 want = target[v] - rest;
        for(size_t i = 0; i < kPerVertex; i++)
          b[graph.near[v][i]] = b[graph.near[v][i]] + want * (graph.weight[v][i] * a);
      }
      for(size_t s = 0; s < ns; s++)
      {
        if(!back[s].hit)
          continue;
        const double a = sourceArea[s] / meanArea;
        Vec3         rest;
        for(size_t i = 0; i < 3; i++)
        {
          const uint32_t v = back[s].corner[i];
          const Vec3     p = at(out.positions, v);
          for(size_t k = 0; k < kPerVertex; k++)
          {
            const uint32_t j = graph.near[v][k];
            rest             = rest + (rotate(p - graph.node[j], turn[j]) + graph.node[j]) * (back[s].share[i] * graph.weight[v][k]);
          }
        }
        const Vec3 want = back[s].target - rest;
        for(const auto& [j, c] : backNodes(back[s]))
          b[j] = b[j] + want * (a * c);
      }
      for(uint32_t j = 0; j < n; j++)
        for(const uint32_t k : graph.linked[j])
        {
          const Vec3 edge = graph.node[k] - graph.node[j];
          const Vec3 e    = (edge - rotate(edge, turn[j])) * lambda;
          b[j]            = b[j] + e;
          b[k]            = b[k] - e;
        }
      for(const Held& h : held)
      {
        Vec3 rest;
        for(const auto& [v, share] : h.members)
        {
          const Vec3 p = at(out.positions, v);
          for(size_t k = 0; k < kPerVertex; k++)
          {
            const uint32_t j = graph.near[v][k];
            rest             = rest + (rotate(p - graph.node[j], turn[j]) + graph.node[j]) * (share * graph.weight[v][k]);
          }
        }
        for(const auto& [j, a] : h.nodes)
          b[j] = b[j] + (h.target - rest) * (hold * a);
      }
      return b;
    };
    // alternate the moves and the turns, twice, on the one factorisation
    for(int pass = 0; pass < 2; pass++)
    {
      std::vector<Vec3> b = rightSide();
      choleskySolve(matrix, n, b);
      move = b;
      // each node's turn: the one that best takes its links as they rest onto them as they are now
      jobs::parallelFor(n, 16, [&](size_t begin, size_t end) {
        for(size_t j = begin; j < end; j++)
        {
          std::vector<Vec3> from, to;
          for(const uint32_t k : graph.linked[j])
          {
            from.push_back(graph.node[k] - graph.node[j]);
            to.push_back(graph.node[k] + move[k] - graph.node[j] - move[j]);
          }
          if(from.size() >= 2)
            turn[j] = bestTurn(from, to);
        }
      });
    }
  }
  deform();

  // 4. what the graph is too coarse for (a bust, a finger): each vertex moved along the source's normal
  // to its skin, the moves spread over the mesh so neighbours go together, a few times over
  if(options.refine > 0)
  {
    std::vector<std::vector<uint32_t>> ring(nv);
    for(size_t t = 0; t * 3 + 2 < reference.tris.size(); t++)
      for(size_t j = 0; j < 3; j++)
      {
        const uint32_t a = reference.tris[t * 3 + j], b = reference.tris[t * 3 + (j + 1) % 3];
        ring[a].push_back(b);
        ring[b].push_back(a);
      }
    for(auto& r : ring)
    {
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
    }
    const double        near2 = options.refineReach * options.refineReach;
    std::vector<Vec3>   want(nv), move2(nv), next(nv);
    std::vector<double> weight(nv);
    std::vector<uint8_t> held2(nv);
    for(int pass = 0; pass < options.refine; pass++)
    {
      pullAll(near2);
      pullBack(near2);
      // each vertex's move: its own pull and the source's on the skin round it, by area
      for(size_t v = 0; v < nv; v++)
      {
        weight[v] = pulled[v] ? area[v] : 0;
        want[v]   = pulled[v] ? (target[v] - at(current, v)) * area[v] : Vec3{};
      }
      for(size_t s = 0; s < ns; s++)
        if(back[s].hit)
        {
          Vec3 q;
          for(size_t i = 0; i < 3; i++)
            q = q + at(current, back[s].corner[i]) * back[s].share[i];
          const Vec3 d = back[s].target - q;
          for(size_t i = 0; i < 3; i++)
          {
            want[back[s].corner[i]] = want[back[s].corner[i]] + d * (back[s].share[i] * sourceArea[s]);
            weight[back[s].corner[i]] += back[s].share[i] * sourceArea[s];
          }
        }
      for(size_t v = 0; v < nv; v++)
      {
        held2[v] = weight[v] > 0;
        want[v]  = held2[v] ? want[v] * (1 / weight[v]) : Vec3{};
      }
      move2 = want;
      for(int sweep = 0; sweep < 16; sweep++)
      {
        jobs::parallelFor(nv, 1024, [&](size_t begin, size_t end) {
          for(size_t v = begin; v < end; v++)
          {
            Vec3 sum;
            for(const uint32_t u : ring[v])
              sum = sum + move2[u];
            const double k = double(ring[v].size()), anchor = held2[v] ? 2 * k : 0;
            next[v]        = k + anchor > 0 ? (want[v] * anchor + sum) * (1 / (k + anchor)) : Vec3{};
          }
        });
        std::swap(move2, next);
      }
      for(size_t v = 0; v < nv; v++)
        for(size_t i = 0; i < 3; i++)
          current[v * 3 + i] += float(i == 0 ? move2[v].x : i == 1 ? move2[v].y : move2[v].z);
    }
    pullAll(near2);
  }
  out.positions = current;
  std::vector<double> gaps;
  for(size_t v = 0; v < nv; v++)
    if(pulled[v])
      gaps.push_back(gap[v]);
  out.pulled = gaps.size();
  if(!gaps.empty())
  {
    std::sort(gaps.begin(), gaps.end());
    out.residualMedian = gaps[gaps.size() / 2];
    out.residualP95    = gaps[size_t(0.95 * double(gaps.size() - 1))];
  }
  return out;
}

TypeMap makeFittedMap(const FitInputs& in, const FitOptions& options, FitResult* fit)
{
  TypeMap out;
  out.source              = bodyOf(*in.source);
  FitResult made          = fitReference(*in.g9, in.reference, out.source, options);
  out.reference           = in.reference;
  out.reference.positions = made.positions;
  out.turns               = made.turns;
  ProjectOptions there, back;
  there.widen   = true;
  back.widen    = true;
  back.refuseCut = true;
  out.map.toReference   = project(out.source, out.reference, there);
  out.map.fromReference = project(out.reference, out.source, back);
  out.map.frameTurn     = frameTurns(vertexFrames(out.source), vertexFrames(bodyOf(*in.g9)), out.map.toReference);
  if(fit)
    *fit = std::move(made);
  return out;
}

std::vector<int32_t> matchCut(const wardrobe::BaseCage& cut, const wardrobe::BaseCage& whole)
{
  const size_t         nc = cut.positions.size() / 3, nw = whole.positions.size() / 3;
  std::vector<int32_t> out(nc, -1);
  if(cut.uvs.size() != nc * 2 || whole.uvs.size() != nw * 2)
    return out;
  // the whole body's vertices by UV, on a grid finer than any two a body's UVs hold apart
  constexpr double kCell = 1e-4, kSame = 2e-5;
  const auto       key = [](double u, double v) { return (int64_t(std::floor(u / kCell)) << 32) ^ int64_t(std::floor(v / kCell) + (1ll << 30)); };
  std::unordered_map<int64_t, std::vector<uint32_t>> byUv;
  for(uint32_t w = 0; w < nw; w++)
    byUv[key(whole.uvs[w * 2], whole.uvs[w * 2 + 1])].push_back(w);
  for(size_t c = 0; c < nc; c++)
  {
    const double u = cut.uvs[c * 2], v = cut.uvs[c * 2 + 1];
    const Vec3   p = at(cut.positions, c);
    double       best = INFINITY;
    for(int du = -1; du <= 1; du++)
      for(int dv = -1; dv <= 1; dv++)
        if(const auto it = byUv.find(key(u + du * kCell, v + dv * kCell)); it != byUv.end())
          for(const uint32_t w : it->second)
            if(std::abs(whole.uvs[w * 2] - u) <= kSame && std::abs(whole.uvs[w * 2 + 1] - v) <= kSame)
              if(const Vec3 d = at(whole.positions, w) - p; dot(d, d) < best)
                best = dot(d, d), out[c] = int32_t(w);
  }
  // a vertex on a UV seam may keep the other side's UV: it takes the nearest whole vertex among the
  // neighbours of its own neighbours' matches, a few times over as matches spread
  std::vector<std::vector<uint32_t>> cutRing(nc), wholeRing(nw);
  const auto ringsOf = [](const std::vector<uint32_t>& tris, std::vector<std::vector<uint32_t>>& ring) {
    for(size_t t = 0; t * 3 + 2 < tris.size(); t++)
      for(size_t j = 0; j < 3; j++)
      {
        ring[tris[t * 3 + j]].push_back(tris[t * 3 + (j + 1) % 3]);
        ring[tris[t * 3 + (j + 1) % 3]].push_back(tris[t * 3 + j]);
      }
  };
  ringsOf(cut.tris, cutRing);
  ringsOf(whole.tris, wholeRing);
  for(int sweep = 0; sweep < 4; sweep++)
  {
    bool grew = false;
    for(size_t c = 0; c < nc; c++)
    {
      if(out[c] >= 0)
        continue;
      const Vec3 p    = at(cut.positions, c);
      double     best = INFINITY;
      for(const uint32_t n : cutRing[c])
        if(out[n] >= 0)
          for(const uint32_t w : wholeRing[size_t(out[n])])
            if(const Vec3 d = at(whole.positions, w) - p; dot(d, d) < best)
              best = dot(d, d), out[c] = int32_t(w);
      grew = grew || out[c] >= 0;
    }
    if(!grew)
      break;
  }
  return out;
}

CutFit cutFit(const wardrobe::BaseCage& cut, const wardrobe::BaseCage& whole, const std::vector<int32_t>& match)
{
  CutFit              out;
  std::vector<double> apart;
  const size_t        nc = cut.positions.size() / 3, nw = whole.positions.size() / 3;
  for(size_t c = 0; c < nc && c < match.size(); c++)
    if(match[c] >= 0 && size_t(match[c]) < nw)
      apart.push_back(length(at(cut.positions, c) - at(whole.positions, size_t(match[c]))));
  out.matched = apart.size();
  if(apart.empty())
    return out;
  std::sort(apart.begin(), apart.end());
  out.median = apart[apart.size() / 2];
  out.p95    = apart[size_t(0.95 * double(apart.size() - 1))];
  return out;
}

std::vector<float> atPreset(const wardrobe::BaseCage& whole, const wardrobe::BaseCage& cut, const std::vector<int32_t>& match)
{
  const size_t nc = cut.positions.size() / 3, nw = whole.positions.size() / 3;
  // each matched whole vertex's move: to the middle of the cut vertices matched to it
  std::vector<Vec3> move(nw);
  std::vector<int>  count(nw, 0);
  for(size_t c = 0; c < nc && c < match.size(); c++)
    if(match[c] >= 0 && size_t(match[c]) < nw)
    {
      move[size_t(match[c])] = move[size_t(match[c])] + (at(cut.positions, c) - at(whole.positions, size_t(match[c])));
      count[size_t(match[c])]++;
    }
  std::vector<std::vector<uint32_t>> ring(nw);
  for(size_t t = 0; t * 3 + 2 < whole.tris.size(); t++)
    for(size_t j = 0; j < 3; j++)
    {
      const uint32_t a = whole.tris[t * 3 + j], b = whole.tris[t * 3 + (j + 1) % 3];
      if(a < nw && b < nw)
      {
        ring[a].push_back(b);
        ring[b].push_back(a);
      }
    }
  // the others: the nearest matched vertex's move, ring by ring out from them
  std::vector<uint8_t>  reached(nw, 0);
  std::vector<uint32_t> front;
  for(uint32_t w = 0; w < nw; w++)
    if(count[w] > 0)
    {
      move[w]    = move[w] * (1.0 / count[w]);
      reached[w] = 1;
      front.push_back(w);
    }
  while(!front.empty())
  {
    std::vector<uint32_t> next;
    for(const uint32_t w : front)
      for(const uint32_t n : ring[w])
        if(!reached[n])
        {
          reached[n] = 1;
          move[n]    = move[w];
          next.push_back(n);
        }
    front = std::move(next);
  }
  // evened out: each a share of the way to its neighbours' mean, the matched ones kept
  constexpr int kSweeps = 60;
  for(int sweep = 0; sweep < kSweeps; sweep++)
    for(size_t i = 0; i < nw; i++)
    {
      // forwards, then backwards, so a move spreads both ways alike
      const size_t w = sweep % 2 ? nw - 1 - i : i;
      if(count[w] > 0 || !reached[w] || ring[w].empty())
        continue;
      Vec3 mean;
      for(const uint32_t n : ring[w])
        mean = mean + move[n];
      move[w] = mean * (1.0 / double(ring[w].size()));
    }
  std::vector<float> out(whole.positions);
  for(size_t w = 0; w < nw; w++)
    if(reached[w])
    {
      out[w * 3]     += float(move[w].x);
      out[w * 3 + 1] += float(move[w].y);
      out[w * 3 + 2] += float(move[w].z);
    }
  return out;
}

TypeMap cutMap(const TypeMap& whole, const Body& cut, const std::vector<int32_t>& match)
{
  TypeMap      out;
  const size_t nc = cut.positions.size() / 3, nr = whole.reference.positions.size() / 3, nw = whole.source.positions.size() / 3;
  out.source      = cut;
  out.reference   = whole.reference;
  out.turns       = whole.turns;
  auto& there     = out.map.toReference;
  there.at.verts.assign(nc * 3, 0);
  there.at.weights.assign(nc * 3, 0.f);
  there.at.used.assign(nr, 0);
  there.mapped.assign(nc, 0);
  out.map.frameTurn.assign(nc, Quat{});
  std::vector<int32_t> inCut(nw, -1);
  for(size_t c = 0; c < nc && c < match.size(); c++)
  {
    const int32_t w = match[c];
    if(w < 0 || size_t(w) >= nw)
      continue;
    inCut[size_t(w)] = int32_t(c);
    if(!whole.map.toReference.mapped[size_t(w)])
      continue;
    for(size_t j = 0; j < 3; j++)
    {
      there.at.verts[c * 3 + j]   = whole.map.toReference.at.verts[size_t(w) * 3 + j];
      there.at.weights[c * 3 + j] = whole.map.toReference.at.weights[size_t(w) * 3 + j];
      there.at.used[there.at.verts[c * 3 + j]] = 1;
    }
    there.mapped[c] = 1;
    if(size_t(w) < whole.map.frameTurn.size())
      out.map.frameTurn[c] = whole.map.frameTurn[size_t(w)];
  }
  auto& back = out.map.fromReference;
  back.at.verts.assign(nr * 3, 0);
  back.at.weights.assign(nr * 3, 0.f);
  back.at.used.assign(nc, 0);
  back.mapped.assign(nr, 0);
  for(size_t r = 0; r < nr; r++)
  {
    if(!whole.map.fromReference.mapped[r])
      continue;
    bool all = true;
    for(size_t j = 0; j < 3 && all; j++)
    {
      const int32_t c = inCut[whole.map.fromReference.at.verts[r * 3 + j]];
      all             = c >= 0 || whole.map.fromReference.at.weights[r * 3 + j] <= 0;
      back.at.verts[r * 3 + j]   = uint32_t(std::max(c, 0));
      back.at.weights[r * 3 + j] = whole.map.fromReference.at.weights[r * 3 + j];
    }
    if(!all)
      continue;
    back.mapped[r] = 1;
    for(size_t j = 0; j < 3; j++)
      back.at.used[back.at.verts[r * 3 + j]] = 1;
  }
  return out;
}

}  // namespace fv::bodymap
