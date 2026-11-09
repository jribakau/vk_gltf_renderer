#include <figure/physics/contact/contact.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/physics/collide/cage_graph.hpp>
#include <figure/physics/volume.hpp>
#include <figure/rig/spec.hpp>
#include <figure/wardrobe/fit/regions.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace fv::physics {

namespace {

using namespace std::chrono_literals;

// A drag moves a bone every frame: the pass waits for a pause this long, as the collider does.
constexpr auto kDebounce = 120ms;
// How far off a collider's surface the skin ends, cm.
constexpr double kClearance = 0.05;
// How deep a vertex is still taken as pushed in through the surface it is closest to, cm: deeper is
// the far side of a thin prop. On a closed piece the crossings make inside certain, so deeper still.
constexpr double kDepth       = 3;
constexpr double kClosedDepth = 8;
// Laplacian steps on the correction a round, and rounds of smoothing and pushing out again.
constexpr int kSmooth = 3;
constexpr int kRounds = 2;
constexpr float kApart = kPartsApart;

// A collider placed for this pass: queried in its own space, or, when its matrix stretches it
// unevenly, as a copy carried into the body's.
struct Placed
{
  const TriangleSet*                 set = nullptr;
  std::shared_ptr<const TriangleSet> copy;
  Mat4                               toBody, fromBody;
  double                             scale = 1;  // the body's centimetres per unit of the set
  double                             share = 1;
  Box                                inBody;      // its bounds in the body's space, grown by the depth
};

// The scale of a matrix that is a rotation and a scale the same along every axis; 0 when it isn't.
double uniformScale(const Mat4& m)
{
  const Vec3   c[3] = {{m.e[0], m.e[1], m.e[2]}, {m.e[4], m.e[5], m.e[6]}, {m.e[8], m.e[9], m.e[10]}};
  const double s    = length(c[0]);
  if(s < 1e-12)
    return 0;
  for(int a = 0; a < 3; a++)
  {
    if(std::fabs(length(c[a]) - s) > 1e-4 * s)
      return 0;
    if(std::fabs(dot(c[a], c[(a + 1) % 3])) > 1e-4 * s * s)
      return 0;
  }
  return s;
}

std::vector<Placed> place(std::span<const ColliderRef> colliders, const Box& body)
{
  std::vector<Placed> out;
  for(const ColliderRef& c : colliders)
  {
    if(!c.set || c.share <= 0)
      continue;
    const Box around = transformed(c.set->bounds(), c.toBody).grown(kClosedDepth);
    if(!around.overlaps(body))
      continue;
    Placed p;
    p.share  = std::min(c.share, 1.0);
    p.inBody = around;
    p.scale  = uniformScale(c.toBody);
    if(p.scale > 0)
    {
      p.set      = &c.set->get();
      p.toBody   = c.toBody;
      p.fromBody = inverse(c.toBody);
    }
    else
    {
      // stretched unevenly: its distances don't scale alike, so its triangles go into the body's space
      const TriangleSet&  set = c.set->get();
      std::vector<float> positions(set.positions().begin(), set.positions().end());
      for(size_t i = 0; i + 2 < positions.size(); i += 3)
      {
        const Vec3 q     = c.toBody.transformPoint({positions[i], positions[i + 1], positions[i + 2]});
        positions[i]     = float(q.x);
        positions[i + 1] = float(q.y);
        positions[i + 2] = float(q.z);
      }
      p.copy  = std::make_shared<const TriangleSet>(std::move(positions), std::vector<uint32_t>(set.tris().begin(), set.tris().end()));
      p.set   = p.copy.get();
      p.scale = 1;
    }
    if(!p.set->index().empty())
      out.push_back(std::move(p));
  }
  return out;
}

// The body met by itself: its surface as the pass starts, and per vertex its part and how far it is
// along the skin from the edge of its part, up to kApart, unposed. Two vertices of different parts are
// at least the sum of those apart along the skin, since the way between leaves the one part and enters
// the other. Each part's triangles (those with a vertex on it) are indexed apart, so a vertex searches
// only the parts it is near and not the skin around it, which is its own.
struct Self
{
  struct Piece
  {
    std::vector<uint32_t> tris;
    SurfaceIndex          index;
    Box                   near;  // its bounds, grown by the depth
  };
  std::span<const float>   positions;
  std::span<const uint8_t> parts;
  std::vector<Piece>       pieces;
  std::vector<float>       fromEdge;
  // per vertex: how deep it may stay, half its depth when it first went in (the other side gives way
  // by the other half); below 0 before
  std::vector<double> keep;

  Self(const ContactInput& in)
      : positions(in.body)
      , parts(in.parts)
      , fromEdge(partEdges(in.rest.size() == in.body.size() ? in.rest : in.body, in.polyStart, in.polyVerts, in.tris, in.parts).fromEdge)
      , keep(in.parts.size(), -1.0)
  {
    pieces.resize(size_t(*std::max_element(parts.begin(), parts.end())) + 1);
    for(size_t t = 0; t + 2 < in.tris.size(); t += 3)
    {
      // once in each part it touches
      const uint8_t a = parts[in.tris[t]], b = parts[in.tris[t + 1]], c = parts[in.tris[t + 2]];
      const auto    add = [&](uint8_t p) { pieces[p].tris.insert(pieces[p].tris.end(), {in.tris[t], in.tris[t + 1], in.tris[t + 2]}); };
      add(a);
      if(b != a)
        add(b);
      if(c != a && c != b)
        add(c);
    }
    for(Piece& piece : pieces)
    {
      piece.index = SurfaceIndex(positions, piece.tris);
      for(uint32_t v : piece.tris)
        piece.near.add(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
      piece.near = piece.near.grown(kDepth);
    }
  }

  // Where vertex v, at p, is inside another part as the pass started: behind the closest face of a
  // part far enough along the skin, and not past its edge.
  Inside inside(size_t v, const double* p, Stamps& stamps) const
  {
    Inside                   out;
    const uint8_t            own  = parts[v];
    const float              from = fromEdge[v];
    SurfaceIndex::Hit        hit;
    const std::vector<uint32_t>* on = nullptr;
    hit.d2                          = kDepth * kDepth;
    for(size_t k = 0; k < pieces.size(); k++)
    {
      const Piece& piece = pieces[k];
      if(k == own || piece.index.empty() || !piece.near.contains(p[0], p[1], p[2]))
        continue;
      const auto found = piece.index.nearest(
          p[0], p[1], p[2],
          [&](uint32_t t) {
            for(int c = 0; c < 3; c++)
            {
              const uint32_t u = piece.tris[t * 3 + c];
              if(parts[u] == own || from + fromEdge[u] < kApart)
                return false;
            }
            return true;
          },
          hit.d2, stamps.stamp, stamps.next(piece.index.triangles()));
      if(found.tri >= 0 && found.d2 <= hit.d2)
      {
        hit = found;
        on  = &piece.tris;
      }
    }
    if(!on)
      return out;
    const std::vector<uint32_t>& tris = *on;
    const size_t a = tris[size_t(hit.tri) * 3] * 3ull, b = tris[size_t(hit.tri) * 3 + 1] * 3ull, c = tris[size_t(hit.tri) * 3 + 2] * 3ull;
    const double ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const double vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    std::array<double, 3> n   = {uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx};
    const double          len = std::hypot(n[0], n[1], n[2]);
    if(len < 1e-20)
      return out;
    std::array<double, 3> q;
    for(int k = 0; k < 3; k++)
    {
      n[size_t(k)] /= len;
      q[size_t(k)] = positions[a + k] * hit.u + positions[b + k] * hit.v + positions[c + k] * hit.w;
    }
    // within about 37 degrees of its face's normal: the skin a vertex may not meet leaves an edge in
    // what it may, which it would otherwise be drawn to sideways
    const double d[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    const double dist = std::hypot(d[0], d[1], d[2]);
    const double in   = -(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]);
    if(in <= 0 || in < 0.8 * dist)
      return out;
    out = {true, q, n, dist};
    return out;
  }
};

// Each body vertex inside a collider moves onto its surface by its share; inside several, the one
// that moves it furthest. It goes back into the body along the skin's normal (`normals`, unit, per
// vertex), as Softie casts its rays, so skin pressed into a seat near its front edge goes up and not
// out through the front; where that ray meets nothing within the depth, to the closest point.
// `target` gets where, `hit` whether.
// Against another body (a share under 1) the rest of the overlap is left for that body to give way:
// `keep`, per collider and vertex, is how deep the vertex may stay, set from where it starts (`first`)
// and held after, so the later rounds don't take a share of what is left each time. A vertex that
// wasn't in at the start keeps nothing: the other body, which met this one undented, makes no room.
// Inside another part of the body (`self`), it moves to its closest point by half, held as `keep` is
// from the round it first went in: that part gives way by the other half. `hit` is 2 for those.
void contacts(std::span<const double> x, std::span<const double> normals, const std::vector<Placed>& placed, std::vector<std::vector<double>>& keep,
              Self* self, bool first, std::vector<double>& target, std::vector<uint8_t>& hit)
{
  const size_t n = x.size() / 3;
  target.assign(n * 3, 0.0);
  hit.assign(n, 0);
  if(first)
  {
    keep.assign(placed.size(), {});
    for(size_t c = 0; c < placed.size(); c++)
      if(placed[c].share < 1)
        keep[c].assign(n, 0.0);
  }
  jobs::parallelFor(n, 256, [&](size_t begin, size_t end) {
    Stamps& stamps = threadStamps();
    for(size_t v = begin; v < end; v++)
    {
      const double* p    = &x[v * 3];
      double        best = -1;
      for(size_t k = 0; k < placed.size(); k++)
      {
        const Placed& c = placed[k];
        if(!c.inBody.contains(p[0], p[1], p[2]))
          continue;
        const Vec3   q  = c.fromBody.transformPoint({p[0], p[1], p[2]});
        Inside in = insideOf(*c.set, {q.x, q.y, q.z}, kDepth / c.scale, kClosedDepth / c.scale, stamps);
        if(!in.inside)
          continue;
        // the skin's inward normal in the set's space
        const Vec3   o    = c.fromBody.transformPoint({0, 0, 0});
        const Vec3   w    = c.fromBody.transformPoint({-normals[v * 3], -normals[v * 3 + 1], -normals[v * 3 + 2]});
        const Vec3   back = normalize(Vec3{w.x - o.x, w.y - o.y, w.z - o.z});
        const Inside out  = exitAlong(*c.set, {q.x, q.y, q.z}, {back.x, back.y, back.z}, (in.depth > kDepth / c.scale ? kClosedDepth : kDepth) / c.scale, stamps);
        if(out.inside)
          in = out;
        const double clear = kClearance / c.scale;
        const Vec3   to    = c.toBody.transformPoint({in.exit[0] + in.normal[0] * clear, in.exit[1] + in.normal[1] * clear, in.exit[2] + in.normal[2] * clear});
        const double depth = std::sqrt((to.x - p[0]) * (to.x - p[0]) + (to.y - p[1]) * (to.y - p[1]) + (to.z - p[2]) * (to.z - p[2]));
        if(depth <= 0)
          continue;
        if(first && !keep[k].empty())
          keep[k][v] = (1 - c.share) * depth;
        const double stay = keep[k].empty() ? 0 : keep[k][v];
        if(depth <= stay)
          continue;
        // toward the exit, stopping `stay` short of it
        const double f  = (depth - stay) / depth;
        const double mx = (to.x - p[0]) * f, my = (to.y - p[1]) * f, mz = (to.z - p[2]) * f;
        const double move = mx * mx + my * my + mz * mz;
        if(move > best)
        {
          best              = move;
          target[v * 3]     = p[0] + mx;
          target[v * 3 + 1] = p[1] + my;
          target[v * 3 + 2] = p[2] + mz;
          hit[v]            = 1;
        }
      }
      if(!self)
        continue;
      const Inside in = self->inside(v, p, stamps);
      if(!in.inside)
        continue;
      const double to[3] = {in.exit[0] + in.normal[0] * kClearance, in.exit[1] + in.normal[1] * kClearance, in.exit[2] + in.normal[2] * kClearance};
      const double depth = std::hypot(to[0] - p[0], to[1] - p[1], to[2] - p[2]);
      if(self->keep[v] < 0)
        self->keep[v] = depth / 2;
      if(depth <= self->keep[v])
        continue;
      const double f    = (depth - self->keep[v]) / depth;
      const double m[3] = {(to[0] - p[0]) * f, (to[1] - p[1]) * f, (to[2] - p[2]) * f};
      const double move = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
      if(move > best)
      {
        best = move;
        for(int a = 0; a < 3; a++)
          target[v * 3 + a] = p[a] + m[a];
        hit[v] = 2;
      }
    }
  });
}

// Moves the vertices that are inside onto their targets. True when any was.
bool pushOut(std::span<double> x, const std::vector<double>& target, const std::vector<uint8_t>& hit)
{
  bool any = false;
  for(size_t v = 0; v < hit.size(); v++)
    if(hit[v])
    {
      for(int a = 0; a < 3; a++)
        x[v * 3 + a] = target[v * 3 + a];
      any = true;
    }
  return any;
}

// The volume a row's particles have been pushed in by, from where they started.
double pushedIn(const DisplacementVolume& row, std::span<const double> x)
{
  double v = 0;
  for(size_t k = 0; k < row.particles.size(); k++)
  {
    const size_t i = size_t(row.particles[k]) * 3;
    double       d = 0;
    for(int a = 0; a < 3; a++)
      d += row.an[k * 3 + a] * (x[i + a] - row.start[k * 3 + a]);
    v += std::max(0.0, -d);
  }
  return v;
}

}  // namespace

std::string contactChannelOf(int mesh)
{
  return "contact:" + std::to_string(mesh);
}

ContactResult resolveContacts(const ContactInput& in)
{
  ContactResult out;
  out.positions.assign(in.body.begin(), in.body.end());
  const size_t n = in.body.size() / 3;
  if(!n)
    return out;
  const std::vector<Placed> placed  = place(in.colliders, boundsOf(in.body));
  const bool                pressed = in.parts.size() == n;
  if(placed.empty() && !pressed)
    return out;
  std::vector<double>       x(in.body.begin(), in.body.end());
  const std::vector<double> start = x;
  const CageGraph           graph(in.body, in.polyStart, in.polyVerts, in.tris);
  std::vector<double>       normals(n * 3);
  for(size_t v = 0; v < n; v++)
    for(int a = 0; a < 3; a++)
      normals[v * 3 + a] = graph.normal(v)[a];
  const std::unique_ptr<Self>      self = pressed ? std::make_unique<Self>(in) : nullptr;
  std::vector<double>              target;
  std::vector<uint8_t>             hit;
  std::vector<std::vector<double>> keep;
  contacts(x, normals, placed, keep, self.get(), true, target, hit);
  out.pressed = int(std::count(hit.begin(), hit.end(), uint8_t(2)));
  if(!pushOut(x, target, hit))
    return out;

  // the patches: what is inside, joined by the cage's edges
  std::vector<int32_t>  patchOf(n, -1);
  std::vector<std::vector<uint32_t>> patches;
  for(size_t v = 0; v < n; v++)
  {
    if(!hit[v] || patchOf[v] >= 0)
      continue;
    std::vector<uint32_t> patch{uint32_t(v)};
    patchOf[v] = int32_t(patches.size());
    for(size_t k = 0; k < patch.size(); k++)
      for(uint32_t u : graph.next(patch[k]))
        if(hit[u] && patchOf[u] < 0)
        {
          patchOf[u] = int32_t(patches.size());
          patch.push_back(u);
        }
    patches.push_back(std::move(patch));
  }
  out.patches = int(patches.size());

  // around each, the skin that takes the volume back: along the surface from the patch, to a reach
  // that grows with how deep it went. Its share fades from the rim, (1 - r/R)^2.
  const double                    beta = std::clamp(double(in.bulge), 0.0, 1.0);
  std::vector<DisplacementVolume> rows;
  if(beta > 0)
    for(const std::vector<uint32_t>& patch : patches)
    {
      double deepest = 0;
      for(uint32_t v : patch)
        deepest = std::max(deepest, std::hypot(x[v * 3] - start[v * 3], x[v * 3 + 1] - start[v * 3 + 1], x[v * 3 + 2] - start[v * 3 + 2]));
      const float reach = float(std::max(double(in.reach), 3 * deepest));
      const auto  ring  = graph.within(patch, reach, [&](uint32_t u) { return patchOf[u] < 0; });
      DisplacementVolume row;
      const auto         add = [&](uint32_t v, double weight) {
        row.particles.push_back(v);
        row.weight.push_back(weight);
        const double* nv = graph.normal(v);
        for(int a = 0; a < 3; a++)
        {
          row.an.push_back(graph.area(v) * nv[a]);
          row.start.push_back(start[v * 3 + a]);
        }
      };
      for(uint32_t v : patch)
        add(v, 0);
      for(const auto& [v, r] : ring)
        if(patchOf[v] < 0 && graph.area(v) > 0)
        {
          const double f = 1 - r / reach;
          add(v, f * f / graph.area(v));
        }
      rows.push_back(std::move(row));
    }
  // the dent's volume, as it stands, put back: at a bulge of 1 the swept volume comes to 0
  const auto bulge = [&] {
    for(DisplacementVolume& row : rows)
    {
      row.target = -(1 - beta) * pushedIn(row, x);
      projectVolume(row, x);
    }
  };
  bulge();
  for(int round = 0; round < kRounds; round++)
  {
    std::vector<float> correction(n * 3);
    for(size_t i = 0; i < correction.size(); i++)
      correction[i] = float(x[i] - start[i]);
    correction = wardrobe::laplacian(std::move(correction), in.tris, n, kSmooth);
    for(size_t i = 0; i < correction.size(); i++)
      x[i] = start[i] + correction[i];
    bulge();
    contacts(x, normals, placed, keep, self.get(), false, target, hit);
    pushOut(x, target, hit);
  }
  for(size_t i = 0; i < x.size(); i++)
    out.positions[i] = float(x[i]);
  return out;
}

namespace {

// A mesh's morphed cage less a hidden channel's target at its weight.
std::vector<float> cageWithout(const morph::MorphSet& morphs, int mesh, const std::string& channel)
{
  std::vector<float> scratch;
  const auto         view = morphs.morphedCageView(mesh, scratch);
  std::vector<float> out(view.begin(), view.end());
  if(channel.empty())
    return out;
  if(const dson::TargetPtr own = morphs.cpuTarget(channel, mesh))
    if(const double w = morphs.weightOf(channel); w != 0)
      for(size_t k = 0; k < own->cageIndices.size(); k++)
        if(size_t(own->cageIndices[k]) * 3 + 2 < out.size())
          for(int a = 0; a < 3; a++)
            out[own->cageIndices[k] * 3 + a] -= float(own->cageDeltas[k * 3 + a] * w);
  return out;
}

}  // namespace

double overlapShare(const ContactSettings& self, const ContactSettings* other)
{
  if(!other || !other->on)
    return 1;
  const double a = std::clamp(double(self.softness), 0.0, 1.0), b = std::clamp(double(other->softness), 0.0, 1.0);
  return a + b > 0 ? a / (a + b) : 0.5;
}

uint8_t partOfBone(const SceneGraph& scene, int node)
{
  for(int b = node; b >= 0 && size_t(b) < scene.nodes.size(); b = scene.nodes[size_t(b)].parent)
  {
    const rig::Name p    = rig::parse(scene.nodes[size_t(b)].name);
    const bool      left = p.side != 'r';
    if(p.slot == "collar")
      return part::kTorso;
    if(p.slot == "upperarm")
      return left ? part::kLUpperArm : part::kRUpperArm;
    if(p.slot == "forearm")
      return left ? part::kLForearm : part::kRForearm;
    switch(rig::limbOf(p))
    {
      case rig::Limb::eHead:
        return part::kHead;
      case rig::Limb::eHand:
        return left ? part::kLHand : part::kRHand;
      case rig::Limb::eThigh:
        return left ? part::kLThigh : part::kRThigh;
      case rig::Limb::eShin:
        return left ? part::kLShin : part::kRShin;
      case rig::Limb::eFoot:
        return left ? part::kLFoot : part::kRFoot;
      case rig::Limb::eArm:
      case rig::Limb::eNone:
        break;
    }
  }
  return part::kTorso;
}

std::vector<uint8_t> bodyParts(const SceneGraph& scene, const Mesh& mesh)
{
  std::vector<int16_t> boneParts;
  for(int b : mesh.bones)
    boneParts.push_back(partOfBone(scene, b));
  const size_t               count = mesh.geo ? mesh.geo->cagePositions.size() / 3 : 0;
  const std::vector<int16_t> each  = wardrobe::vertexLimbs(count, mesh.cageSkin.start.empty() ? nullptr : &mesh.cageSkin, boneParts);
  std::vector<uint8_t>       out(each.size());
  for(size_t v = 0; v < each.size(); v++)
    out[v] = uint8_t(each[v]);
  return out;
}

PartEdges partEdges(std::span<const float> positions, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts,
                    std::span<const uint32_t> tris, std::span<const uint8_t> parts, float reach)
{
  PartEdges out;
  out.fromEdge.assign(parts.size(), reach);
  if(parts.empty() || positions.size() != parts.size() * 3)
    return out;
  out.joined.assign(size_t(*std::max_element(parts.begin(), parts.end())) + 1, 0);
  const CageGraph       graph(positions, polyStart, polyVerts, tris);
  std::vector<uint32_t> edge;
  for(size_t v = 0; v < graph.size(); v++)
  {
    bool on = false;
    for(uint32_t u : graph.next(v))
      if(parts[u] != parts[v])
      {
        on = true;
        out.joined[parts[v]] |= 1u << parts[u];
      }
    if(on)
      edge.push_back(uint32_t(v));
  }
  // an edge between parts joins two vertices on edges, both at 0, so the distances stay in their part
  for(const auto& [v, d] : graph.within(edge, reach, [](uint32_t) { return true; }))
    out.fromEdge[v] = d;
  return out;
}

namespace {

// Each (mesh, node) asked: the mesh's cage morphed and posed, made once however many nodes draw it,
// placed where the node stands (-1: at the root).
std::vector<std::vector<float>> posedAt(const SceneGraph& scene, const morph::MorphSet& morphs, std::span<const std::pair<int, int>> asked,
                                        int root, const std::string& without)
{
  std::vector<std::vector<float>> out(asked.size());
  const auto worlds    = scene.worlds();
  const auto nodeOf    = physics::nodeOfMesh(scene);
  const Mat4 rootWorld = root >= 0 && size_t(root) < worlds.size() ? worlds[size_t(root)] : Mat4{};
  std::unordered_map<int, std::vector<float>> posedOf;
  for(size_t k = 0; k < asked.size(); k++)
  {
    const auto [mesh, node] = asked[k];
    if(mesh < 0 || size_t(mesh) >= scene.meshes.size())
      continue;
    const Mesh& m = scene.meshes[size_t(mesh)];
    if(!m.geo || m.geo->cagePositions.empty())
      continue;
    auto at = posedOf.find(mesh);
    if(at == posedOf.end())
    {
      std::vector<float> posed = cageWithout(morphs, mesh, without);
      if(m.skinned() && m.cageSkin.start.size() == posed.size() / 3 + 1)
        posed = physics::skinLocal(posed, physics::cageSkin(m.cageSkin), physics::currentSkin(scene, worlds, nodeOf, mesh));
      at = posedOf.emplace(mesh, std::move(posed)).first;
    }
    out[k] = physics::transformPoints(inverse(rootWorld) * (node >= 0 ? worlds[size_t(node)] : Mat4{}), at->second);
  }
  return out;
}

}  // namespace

std::vector<std::vector<float>> posedCages(const SceneGraph& scene, const morph::MorphSet& morphs, std::span<const int> meshes, int root,
                                           const std::string& without)
{
  // each where its first node stands
  const auto                       nodeOf = physics::nodeOfMesh(scene);
  std::vector<std::pair<int, int>> asked;
  for(int mesh : meshes)
    asked.emplace_back(mesh, mesh >= 0 && size_t(mesh) < nodeOf.size() ? nodeOf[size_t(mesh)] : -1);
  return posedAt(scene, morphs, asked, root, without);
}

std::vector<std::vector<float>> posedNodeCages(const SceneGraph& scene, const morph::MorphSet& morphs, std::span<const int> nodes, int root,
                                               const std::string& without)
{
  std::vector<std::pair<int, int>> asked;
  for(int node : nodes)
    asked.emplace_back(node >= 0 && size_t(node) < scene.nodes.size() ? scene.nodes[size_t(node)].mesh : -1, node);
  return posedAt(scene, morphs, asked, root, without);
}

// What a pass needs and where its result goes, taken on the owning thread.
struct Contact::Job
{
  // the body as posed now, without its own dent, in the root's space
  std::vector<float>       body;
  std::vector<uint32_t>    tris;
  std::vector<int32_t>     polyStart, polyVerts;
  std::vector<ColliderRef> colliders;
  ContactSettings          settings;
  // per vertex, when the body's parts press on one another, and the body unposed in the root's space
  std::vector<uint8_t> parts;
  std::vector<float>   unposed;
  // for the write-back: the rest cage under the dent, its skin and pose in the mesh's space, and the way there
  std::vector<float> rest;
  bool               skinned = false;
  physics::CageSkin   skin;
  physics::SkinPose   pose;
  Mat4               back;
};

Contact::Contact(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root)
    : m_scene(scene)
    , m_morphs(morphs)
    , m_bodyMesh(bodyMesh)
    , m_root(root)
    , m_pass(std::chrono::duration_cast<std::chrono::milliseconds>(kDebounce))
{
  m_writer.name  = contactChannelOf(bodyMesh);
  m_writer.label = "Contact";
  m_writer.mesh  = bodyMesh;
  poke();
}

Contact::~Contact() = default;

bool Contact::busy() const
{
  return m_settings.on && !m_held && m_pass.busy();
}

bool Contact::set(const ContactSettings& s)
{
  if(s == m_settings)
    return false;
  const bool was = m_settings.on;
  m_settings     = s;
  m_pass.bump();
  if(!s.on)
  {
    m_pass.cancel();
    m_patches = 0;
    m_weight = m_aim = 0;
    m_fadeIn         = false;
    return was && m_writer.off(m_morphs);
  }
  if(!was)
    m_pass.poke();
  else
    m_pass.pokeNow();
  return false;
}

void Contact::changed(const std::vector<int>& meshes, bool nodesMoved)
{
  if(nodesMoved || std::find(meshes.begin(), meshes.end(), m_bodyMesh) != meshes.end())
    poke();
}

void Contact::poke()
{
  if(m_settings.on)
    m_pass.poke();
}

std::shared_ptr<Contact::Job> Contact::prepare()
{
  timing::Scope t("contact: prepare", 2);
  if(m_bodyMesh < 0 || size_t(m_bodyMesh) >= m_scene.meshes.size())
    return nullptr;
  const Mesh& body = m_scene.meshes[size_t(m_bodyMesh)];
  if(!body.geo || body.geo->cagePositions.empty())
    return nullptr;
  auto job       = std::make_shared<Job>();
  job->colliders = colliders ? colliders() : std::vector<ColliderRef>{};
  if(m_settings.self && m_parts.size() != body.geo->cagePositions.size() / 3)
    m_parts = bodyParts(m_scene, body);
  // a body all of one part (no skin) has nothing to press on itself
  const bool self = m_settings.self && std::any_of(m_parts.begin(), m_parts.end(), [&](uint8_t p) { return p != m_parts[0]; });
  if(job->colliders.empty() && !self)
    return nullptr;
  job->settings = m_settings;
  if(self)
    job->parts = m_parts;

  // the body under its own dent: the shape kept, less this channel's target at its weight
  job->rest = cageWithout(m_morphs, m_bodyMesh, m_writer.name);

  const auto worlds    = m_scene.worlds();
  const auto nodeOf    = physics::nodeOfMesh(m_scene);
  const Mat4 rootWorld = m_root >= 0 && size_t(m_root) < worlds.size() ? worlds[size_t(m_root)] : Mat4{};
  const int  node      = nodeOf[size_t(m_bodyMesh)];
  const Mat4 toSpace   = inverse(rootWorld) * (node >= 0 ? worlds[size_t(node)] : Mat4{});
  job->skinned         = body.skinned() && body.cageSkin.start.size() == job->rest.size() / 3 + 1;
  std::vector<float> posed;
  if(job->skinned)
  {
    job->skin = physics::cageSkin(body.cageSkin);
    job->pose = physics::currentSkin(m_scene, worlds, nodeOf, m_bodyMesh);
    posed     = physics::skinLocal(job->rest, job->skin, job->pose);
  }
  else
    posed = job->rest;
  job->body      = physics::transformPoints(toSpace, posed);
  job->back      = inverse(toSpace);
  if(!job->parts.empty())
    job->unposed = physics::transformPoints(toSpace, job->rest);
  job->tris      = body.geo->dsonTris;
  job->polyStart = body.geo->cagePolyStart;
  job->polyVerts = body.geo->cagePolyVerts;
  return job;
}

bool Contact::apply(const Landed* landed)
{
  timing::Scope t("contact: write", 2);
  const Mesh&   body   = m_scene.meshes[size_t(m_bodyMesh)];
  const int     before = m_patches;
  bool          wrote  = false;
  if(!landed)
  {
    // nothing near the body: whatever was dented shows as it is
    m_patches = 0;
    m_weight  = 1;
    if(body.geo && m_morphs.cpuTarget(m_writer.name, m_bodyMesh))
      wrote = m_writer.write(m_morphs, *body.geo, std::vector<float>(body.geo->cagePositions.size(), 0.f));
  }
  else if(body.geo && body.geo->cagePositions.size() == landed->delta.size())
  {
    m_patches = landed->patches;
    // after a hold it fades in from where the fade out left it; otherwise (a prop moved, a slider) at once
    if(!m_fadeIn)
      m_weight = 1;
    wrote = m_writer.write(m_morphs, *body.geo, landed->delta, m_weight);
  }
  m_aim    = 1;
  m_fadeIn = false;
  if(m_patches != before)
    m_pass.bump();
  return wrote;
}

bool Contact::fade()
{
  // a hitch doesn't skip the fade
  const auto   now = std::chrono::steady_clock::now();
  const double dt  = std::min(std::chrono::duration<double>(now - m_at).count(), 0.1);
  m_at             = now;
  if(m_weight == m_aim)
    return false;
  m_weight = m_aim > m_weight ? std::min(m_aim, m_weight + dt / kFade) : std::max(m_aim, m_weight - dt / kFade);
  if(!m_writer.weigh(m_morphs, m_weight))
    return false;
  m_morphs.flush();
  return true;
}

bool Contact::pump()
{
  if(!m_settings.on)
    return false;
  if(driven && driven())
  {
    if(!m_held)
    {
      m_held = true;
      m_aim  = 0;
      m_pass.cancel();
      m_pass.bump();
    }
    return fade();
  }
  if(m_held)
  {
    // paused: a pass on the pose it stopped at, which fades in
    m_held   = false;
    m_fadeIn = true;
    m_pass.bump();
    m_pass.poke();
  }
  const bool faded = fade();
  return m_pass.pump(
      [this] {
        auto job = prepare();
        if(!job)
          return std::future<Landed>();
        m_bounds = boundsOf(job->body);
        return jobs::pool().submit([job] {
          ContactInput in;
          in.body      = job->body;
          in.tris      = job->tris;
          in.polyStart = job->polyStart;
          in.polyVerts = job->polyVerts;
          in.colliders = job->colliders;
          in.bulge     = job->settings.bulge;
          in.reach     = job->settings.reach;
          in.parts     = job->parts;
          in.rest      = job->unposed;
          const ContactResult r = resolveContacts(in);
          // back to the mesh's space, and to the cage deltas whose skinning gives it
          Landed     out;
          const auto local = physics::transformPoints(job->back, r.positions);
          if(job->skinned)
            out.delta = physics::drapeDelta(job->rest, local, job->skin, job->pose);
          else
          {
            out.delta.resize(local.size());
            for(size_t i = 0; i < local.size(); i++)
              out.delta[i] = local[i] - job->rest[i];
          }
          out.patches = r.patches;
          return out;
        });
      },
      [this](const Landed* landed) { return apply(landed); }) || faded;
}

}  // namespace fv::physics
