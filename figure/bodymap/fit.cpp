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
  out.positions = intoRest(cage, out.positions, source, &out.turns);

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
  const int rounds = std::max(options.rounds, 1);
  for(int round = 0; round < rounds; round++)
  {
    const double stiff = options.stiffFrom * std::pow(options.stiffTo / options.stiffFrom, rounds > 1 ? double(round) / double(rounds - 1) : 1.0);
    deform();
    pullAll(reach2);
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
    const double      near2 = options.refineReach * options.refineReach;
    std::vector<Vec3> want(nv), move2(nv), next(nv);
    for(int pass = 0; pass < options.refine; pass++)
    {
      pullAll(near2);
      for(size_t v = 0; v < nv; v++)
        want[v] = pulled[v] ? target[v] - at(current, v) : Vec3{};
      move2 = want;
      for(int sweep = 0; sweep < 16; sweep++)
      {
        jobs::parallelFor(nv, 1024, [&](size_t begin, size_t end) {
          for(size_t v = begin; v < end; v++)
          {
            Vec3 sum;
            for(const uint32_t u : ring[v])
              sum = sum + move2[u];
            const double k = double(ring[v].size()), anchor = pulled[v] ? 2 * k : 0;
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

}  // namespace fv::bodymap
