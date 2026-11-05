#include <figure/physics/cloth/island.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/geometry.hpp>
#include <figure/physics/collide/tri_grid.hpp>

namespace fv::physics::cloth {

namespace {

// The largest push one collision pass gives a vertex, cm.
constexpr float kCap = 2.f;
// Speed limit, cm/s: a vertex that escapes a fold does not fly off.
constexpr float kMaxSpeed = 400.f;
// Substeps a frame is cut into: at least this many for the constraints to converge, at most this
// many however fast the colliders move. Collision passes are fewer; see frame().
constexpr int kMinSubsteps = 24, kMaxSubsteps = 48;
// The pose transition: colliders move at most this fast, within these bounds, seconds.
constexpr float kTransitionSpeed = 60.f, kMinTransition = 0.25f, kMaxTransition = 0.75f;
// Velocity decay added per second while the pose is held, so the cloth comes to rest in a few frames.
constexpr float kHoldDamping = 6.f;
// Collision passes per frame, at least, and the most anything may move between two: half the depth
// a vertex can still be pushed back from.
constexpr int   kMinCollisionPasses = 4;
constexpr float kPassSpacing        = 0.5f * kDepth;
// How far around itself a vertex looks for faces, cm, beyond the gap; finding none, it need not look
// again until that much has moved.
constexpr float kSearch = 0.5f;
// How far behind the nearest face of a lower layer a lost vertex is still brought back from, cm:
// what the wardrobe collider trusts on a conforming garment. Deeper than that, the nearest face is
// as likely the far side of a limb.
constexpr float kRecoverDepth = 3.f;

using Item = TriGrid::Item;

// How a mesh's kinematic vertices (colliders, pinned cloth) move with the pose: a sweep of its
// bones from one pose to another.
struct Kinematic
{
  bool                  skinned = false;
  std::vector<BonePose> from, to;
  std::vector<Mat34>    at;     // toSpace * bone(t), per bone, for the current t
  std::vector<int>      verts;  // global indices of the vertices this moves
  std::vector<int>      soft;   // and of the free vertices tethered to where it would put them
};

}  // namespace

struct Island::State : Model
{
  std::vector<Surface>   meshes;
  Options                opts;
  std::vector<Kinematic> kin;

  // state: `cprev` is where everything was at the last collision pass and `tested` where each free
  // vertex was when it last looked for faces, which decide the side it is on; `prev` is the substep
  // before, which gives velocities. `budget` is how far a vertex may still move before it has to
  // look for faces again (0 while it touches one). `anchor` is where the pose would put each soft
  // vertex at the current substep (only those entries are kept up to date).
  std::vector<float>  x, prev, cprev, tested, budget, vel, snap, endPos, anchor;
  // the face of the pose (a limb, a belt) and the face of cloth a vertex last rested on, or -1. Two,
  // since a vertex between a limb and a fold over it rests on both: with one, it swapped between
  // them every pass and was held against neither
  std::vector<int>    contact, clothContact;
  int                 passCount = 0;
  std::vector<double> lam, lamAtt;
  std::vector<V3>     n0, n1, boxLo, boxHi;
  TriGrid             staticGrid, dynamicGrid;
  bool                forward = true;
  int                 substeps   = kMinSubsteps;
  int                 collisions = kMinCollisionPasses;
  float               transition = kMinTransition;
  float               lastMove   = 0;
  int                 recovered  = 0;
  // where the kinematic vertices are heading this frame, and how far the furthest of them goes
  float               kinematicMove = 0;
  // What a vertex's search reach takes beside its own offset: 0, so the vertex's own. This member hid
  // the model's maxOffset (the largest gap of any face or vertex, which the GPU solver's reach takes)
  // before step 12, and was never set; kept, so the drapes stay the same to the bit.
  float               maxOffset  = 0;

  State(std::vector<Surface> in, const Options& o)
      : Model(Model::build(in, o.air, o.reachLinks))
      , meshes(std::move(in))
      , opts(o)
  {
    bindKinematics();
    x = rest;
    separateStart();
    prev   = x;
    cprev  = x;
    tested = x;
    budget.assign(size_t(nv), 0.f);
    contact.assign(size_t(nv), -1);
    clothContact.assign(size_t(nv), -1);
    vel.assign(size_t(nv) * 3, 0.f);
    snap.resize(size_t(nv) * 3);
    endPos.resize(size_t(nv) * 3);
    anchor = x;
    lam.assign(edges.size(), 0.0);
    lamAtt.assign(size_t(nv), 0.0);
    n0.resize(tris.size());
    n1.resize(tris.size());
    boxLo.resize(tris.size());
    boxHi.resize(tris.size());
    plan();
  }

  void bindKinematics();
  void plan();
  // Writes the pose at fraction t of the kinematic vertices into `out` (only those entries), and
  // of the soft vertices into `anchors` when given.
  void kinematicAt(float t, std::vector<float>& out, std::vector<float>* anchors);
  void separateStart();
  void recoverLost();
  // The face a vertex resting on face `t` is over now: `t`, or the nearest face two edges around it
  // that the vertex slid onto; -1 when it is over none (it left the surface). Fills `now` and `n`.
  int  followContact(int i, int t, Closest& now, V3& n) const;
  void holdContacts(bool grip);
  // One sweep over the constraints (the stretch edges alone when `stretchOnly`) and the tethers.
  void solveEdges(float dt, bool stretchOnly = false);
  void collide(float kinematicStep);
  void frame(float t0, float t1, float extraDamping, float seconds);
};

// The pose's hold on each mesh: its bones, swept from the identity (the memorized pose) to the
// target the mesh came with.
void Island::State::bindKinematics()
{
  kin.resize(meshes.size());
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const Surface& m = meshes[mi];
    Kinematic&        k = kin[mi];
    k.skinned           = m.skinned();
    if(k.skinned)
    {
      k.to.reserve(m.bones.size());
      for(const Mat4& b : m.bones)
        k.to.push_back(poseOf(b));
      k.from.assign(m.bones.size(), poseOf(Mat4{}));
      k.at.resize(m.bones.size());
    }
  }
  for(int i : kinematicVerts)
    kin[size_t(meshOf[i])].verts.push_back(i);
  for(int i : softVerts)
    kin[size_t(meshOf[i])].soft.push_back(i);
}

void Island::State::plan()
{
  // how far the pose moves what it holds, along the chord: the transition's length and its substeps
  float maxMove = 0;
  for(int i : kinematicVerts)
  {
    const float dx = posed[i * 3] - rest[i * 3], dy = posed[i * 3 + 1] - rest[i * 3 + 1], dz = posed[i * 3 + 2] - rest[i * 3 + 2];
    maxMove = std::max(maxMove, std::sqrt(dx * dx + dy * dy + dz * dz));
  }
  transition = opts.transition > 0 ? opts.transition : std::min(kMaxTransition, std::max(kMinTransition, maxMove / kTransitionSpeed));
  const int   frames   = std::max(1, int(std::lround(transition / opts.frameDt)));
  const int   byMotion = int(std::ceil(maxMove / float(frames) / kPassSpacing));
  if(opts.substeps > 0)
    substeps = std::min(240, opts.substeps);
  else
    substeps = std::min(kMaxSubsteps, std::max(kMinSubsteps, byMotion));
  collisions = std::min(substeps, std::max(kMinCollisionPasses, byMotion));
}

void Island::State::kinematicAt(float t, std::vector<float>& out, std::vector<float>* anchors)
{
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    Kinematic& k = kin[mi];
    if(k.verts.empty() && (!anchors || k.soft.empty()))
      continue;
    const Surface& m = meshes[mi];
    if(!k.skinned)
    {
      const auto lerp = [&](const std::vector<int>& verts, std::vector<float>& dst) {
        for(int i : verts)
          for(int c = 0; c < 3; c++)
            dst[i * 3 + c] = (1 - t) * rest[i * 3 + c] + t * posed[i * 3 + c];
      };
      lerp(k.verts, out);
      if(anchors)
        lerp(k.soft, *anchors);
      continue;
    }
    for(size_t b = 0; b < k.to.size(); b++)
      k.at[b] = Mat34::of(m.toSpace * sweepAt(k.from[b], k.to[b], t));
    const int  o    = base[mi];
    const auto skin = [&](const std::vector<int>& verts, std::vector<float>& dst) {
      jobs::parallelFor(verts.size(), 2048, [&](size_t begin, size_t end) {
        for(size_t vi = begin; vi < end; vi++)
        {
          const int    i = verts[vi];
          const size_t v = size_t(i - o);
          float        blend[12] = {};
          float        wsum      = 0;
          for(size_t j = 0; j < 4; j++)
          {
            const float  w    = m.skinWeight[v * 4 + j];
            const size_t bone = m.skinIndex[v * 4 + j];
            if(w == 0 || bone >= k.at.size())
              continue;
            wsum += w;
            for(int c = 0; c < 12; c++)
              blend[c] += k.at[bone].m[c] * w;
          }
          if(wsum <= 0)
          {
            const Vec3 p = m.toSpace.transformPoint({m.restLocal[v * 3], m.restLocal[v * 3 + 1], m.restLocal[v * 3 + 2]});
            dst[i * 3]     = float(p.x);
            dst[i * 3 + 1] = float(p.y);
            dst[i * 3 + 2] = float(p.z);
            continue;
          }
          const float px = m.restLocal[v * 3], py = m.restLocal[v * 3 + 1], pz = m.restLocal[v * 3 + 2];
          dst[i * 3]     = blend[0] * px + blend[3] * py + blend[6] * pz + blend[9];
          dst[i * 3 + 1] = blend[1] * px + blend[4] * py + blend[7] * pz + blend[10];
          dst[i * 3 + 2] = blend[2] * px + blend[5] * py + blend[8] * pz + blend[11];
        }
      });
    };
    skin(k.verts, out);
    if(anchors)
      skin(k.soft, *anchors);
  }
}

// Moves free vertices that start inside, or within the gap of, a lower layer out to the gap along
// the nearest face of that layer. The nearest face decides inside, so a vertex just outside a thin
// limb is not sent through it by the face on the far side. Only the start moves: rest lengths keep
// the fitted shape.
void Island::State::separateStart()
{
  cloth::separateStart(*this, x);
}

// Puts back free vertices that lost the surface they were outside of: one touching no face that
// lies behind the nearest static face of a lower layer, within kRecoverDepth, is moved out to the
// gap along that face and remembered as outside there, so the next pass keeps it out. The nearest
// face decides, as at the start, so a vertex just outside a thin limb is not sent through it.
void Island::State::recoverLost()
{
  recovered = 0;
  if(staticTris.empty())
    return;
  std::atomic<int> count{0};
  jobs::parallelFor(freeVerts.size(), 256, [&](size_t begin, size_t end) {
    std::vector<uint32_t> seen(tris.size(), 0);
    uint32_t              stamp = 0;
    int                   mine  = 0;
    for(size_t vi = begin; vi < end; vi++)
    {
      const int i = freeVerts[vi];
      if(!collides[i] || contact[i] >= 0)
        continue;
      const float qx = x[i * 3], qy = x[i * 3 + 1], qz = x[i * 3 + 2];
      const float lo[3] = {qx - kRecoverDepth, qy - kRecoverDepth, qz - kRecoverDepth};
      const float hi[3] = {qx + kRecoverDepth, qy + kRecoverDepth, qz + kRecoverDepth};
      stamp++;
      int   best     = -1;
      float bestDist = kRecoverDepth, bestSigned = 0;
      V3    bestN{};
      staticGrid.query(lo, hi, [&](const Item& item) {
        const int t = item.tri;
        if(seen[size_t(t)] == stamp)
          return;
        seen[size_t(t)] = stamp;
        const Face& tri  = tris[size_t(t)];
        if(tri.layer >= layer[i])
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
      if(best < 0 || bestSigned >= 0)
        continue;
      const float gap  = std::max(tris[size_t(best)].offset, offset[i]);
      const float push = gap - bestSigned;
      for(int k = 0; k < 3; k++)
      {
        x[i * 3 + k] += bestN[size_t(k)] * push;
        tested[i * 3 + k] = x[i * 3 + k];
      }
      // resting on that face now: the passes keep it there while its neighbours catch up
      contact[i] = best;
      budget[i]  = 0;
      mine++;
    }
    count += mine;
  });
  recovered = count.load();
}

// Between collision passes, a vertex resting on a face of the pose (a limb, a belt) is held outside
// it every substep: its edges can pull it centimetres into the limb between two passes, past the
// depth the next pass pushes back from, and the frame after puts it back, a tug of war that never
// settles. The face as the pose has it this substep holds it, while it stays over the face.
int Island::State::followContact(int i, int t, Closest& now, V3& n) const
{
  const float qx = x[i * 3], qy = x[i * 3 + 1], qz = x[i * 3 + 2];
  const Face&  tri = tris[size_t(t)];
  if(triNormal(x, tri.a, tri.b, tri.c, n))
  {
    now = closest(qx, qy, qz, x, tri.a, tri.b, tri.c, n);
    if(now.over)
      return t;
  }
  int cand[12], nc = 0;
  for(int k = 0; k < 3; k++)
  {
    const int a = triNext[size_t(t) * 3 + size_t(k)];
    if(a < 0)
      continue;
    cand[nc++] = a;
    for(int l = 0; l < 3; l++)
    {
      const int b = triNext[size_t(a) * 3 + size_t(l)];
      if(b >= 0 && b != t)
        cand[nc++] = b;
    }
  }
  int   at   = -1;
  float best = INFINITY;
  for(int k = 0; k < nc; k++)
  {
    const Face& nt = tris[size_t(cand[k])];
    V3         nn;
    if(!triNormal(x, nt.a, nt.b, nt.c, nn))
      continue;
    const Closest c = closest(qx, qy, qz, x, nt.a, nt.b, nt.c, nn);
    if(c.over && c.dist < best)
    {
      best = c.dist;
      now  = c;
      n    = nn;
      at   = cand[k];
    }
  }
  return at;
}

void Island::State::holdContacts(bool grip)
{
  jobs::parallelFor(freeVerts.size(), 1024, [&](size_t begin, size_t end) {
    for(size_t fi = begin; fi < end; fi++)
    {
      const int i = freeVerts[fi];
      // a cloth face moves with the solve; its passes handle it
      const int t = contact[i];
      if(t < 0)
        continue;
      Closest   now{};
      V3        n{};
      const int at = followContact(i, t, now, n);
      if(at < 0)
        continue;
      contact[i]      = at;
      const Face& held = tris[size_t(at)];
      const float gap = std::max(held.offset, offset[i]);
      int direction;
      if(layer[i] != held.layer)
        direction = layer[i] > held.layer ? 1 : -1;
      else
      {
        const float side = (tested[i * 3] - x[held.a * 3]) * n[0] + (tested[i * 3 + 1] - x[held.a * 3 + 1]) * n[1] + (tested[i * 3 + 2] - x[held.a * 3 + 2]) * n[2];
        direction        = side >= 0 ? 1 : -1;
      }
      const float target = float(direction) * gap;
      if(direction > 0 ? now.signedDistance >= target : now.signedDistance <= target)
        continue;
      const float push = std::min(std::fabs(target - now.signedDistance), kCap) * (target - now.signedDistance > 0 ? 1.f : -1.f);
      for(int k = 0; k < 3; k++)
        x[i * 3 + k] += n[size_t(k)] * push;
      // Coulomb: this substep's slide over the face is taken back as far as the push's normal force
      // holds it. This is where a resting vertex meets its face: the collision pass after it finds
      // the vertex in place already. Not in the hold after a pass, with nothing after it to see that
      // the slide took the vertex under the next face
      if(!grip)
        continue;
      float rub[3];
      for(int k = 0; k < 3; k++)
      {
        const float face = ((x[held.a * 3 + k] - prev[held.a * 3 + k]) + (x[held.b * 3 + k] - prev[held.b * 3 + k]) + (x[held.c * 3 + k] - prev[held.c * 3 + k])) / 3.f;
        rub[k]           = (x[i * 3 + k] - prev[i * 3 + k]) - face;
      }
      const float dn = rub[0] * n[0] + rub[1] * n[1] + rub[2] * n[2];
      for(int k = 0; k < 3; k++)
        rub[k] -= n[size_t(k)] * dn;
      const float slide = std::sqrt(rub[0] * rub[0] + rub[1] * rub[1] + rub[2] * rub[2]);
      if(slide > 1e-9f)
      {
        const float grip  = (held.friction + friction[i]) * 0.5f * std::fabs(push);
        const float share = std::min(1.f, grip / slide);
        for(int k = 0; k < 3; k++)
          x[i * 3 + k] -= rub[k] * share;
      }
    }
  });
}

void Island::State::solveEdges(float dt, bool stretchOnly)
{
  const float atScale = 1.f / (dt * dt);
  const auto  project = [&](int e) {
    const Distance& edge = edges[size_t(e)];
    // not in a full sweep (one follows a collision pass): see the top of island.hpp
    if(edge.kind == DistanceKind::eReach && !stretchOnly)
      return;
    // a bend strut shorter than its buckling length folds at the buckling stiffness
    const auto alphaAt = [&](float len) {
      if(edge.kind == DistanceKind::eBend && len < edge.rest * edge.ratio)
        return edge.buckle > 0 ? edge.alpha / edge.buckle : 1e6f;
      return edge.alpha;
    };
    // a reach link only ever pulls
    physics::particles::projectDistance(x.data(), edge.a, edge.b, inv[edge.a], inv[edge.b], edge.rest, alphaAt, atScale, lam[size_t(e)], edge.kind == DistanceKind::eReach);
  };
  // sweeps alternate direction, so a correction reaches both ends of a chain in two of them
  physics::particles::sweepBlocks(blocks, stretchOnly, forward, project);
  forward = !forward;
  // the tethers of partial strength: a distance-0 constraint to the skinned position, as compliant
  // as the strength makes it; each writes its own vertex only
  jobs::parallelFor(softVerts.size(), 1024, [&](size_t begin, size_t end) {
    for(size_t k = begin; k < end; k++)
    {
      const int i = softVerts[k];
      physics::particles::projectAttach(x.data(), i, &anchor[size_t(i) * 3], inv[i], attAlpha[i], atScale, lamAtt[size_t(i)]);
    }
  });
}

// Pushes free vertices out of faces they were outside of when they last looked (`tested`). `snap`
// is where everything is now, so the faces of a cloth that collides with itself are read from one
// moment while its vertices move. A vertex far from every face looks again only once it or the
// colliders could have closed that distance; one resting on a face tests that face alone while it
// stays over it (or the face it slid onto), and looks around every few passes.
void Island::State::collide(float kinematicStep)
{
  if(tris.empty())
    return;
  passCount++;
  const bool lookAround = passCount % 4 == 0;
  if(!dynamicTris.empty() && lookAround)
    dynamicGrid.build(tris, dynamicTris, snap, snap, 0.f);
  jobs::parallelFor(tris.size(), 4096, [&](size_t begin, size_t end) {
    for(size_t t = begin; t < end; t++)
    {
      const Face& tri = tris[t];
      if(!triNormal(cprev, tri.a, tri.b, tri.c, n0[t]))
        n0[t] = {0, 0, 0};
      if(!triNormal(snap, tri.a, tri.b, tri.c, n1[t]))
        n1[t] = {0, 0, 0};
      for(int c = 0; c < 3; c++)
      {
        boxLo[t][size_t(c)] = std::min(snap[tri.a * 3 + c], std::min(snap[tri.b * 3 + c], snap[tri.c * 3 + c]));
        boxHi[t][size_t(c)] = std::max(snap[tri.a * 3 + c], std::max(snap[tri.b * 3 + c], snap[tri.c * 3 + c]));
      }
    }
  });
  jobs::parallelFor(freeVerts.size(), 512, [&](size_t begin, size_t end) {
    thread_local std::vector<uint32_t> seen;
    thread_local uint32_t              stamp = 0;
    if(seen.size() != tris.size())
    {
      seen.assign(tris.size(), 0);
      stamp = 0;
    }
    for(size_t fi = begin; fi < end; fi++)
    {
      const int i = freeVerts[fi];
      if(!collides[i])
        continue;
      float sincePass = 0;
      for(int c = 0; c < 3; c++)
        sincePass += (x[i * 3 + c] - cprev[i * 3 + c]) * (x[i * 3 + c] - cprev[i * 3 + c]);
      sincePass = std::sqrt(sincePass);
      // the budget pays for the vertex's own motion and the colliders'. A part of the cloth coming
      // at it is not charged: it is met at most a budget late, well within the depth a vertex can
      // still be pushed back from.
      budget[i] -= sincePass + kinematicStep;
      if(budget[i] > 0)
        continue;
      float moved = 0;
      for(int c = 0; c < 3; c++)
        moved += (x[i * 3 + c] - tested[i * 3 + c]) * (x[i * 3 + c] - tested[i * 3 + c]);
      moved = std::sqrt(moved);
      const float qx = x[i * 3], qy = x[i * 3 + 1], qz = x[i * 3 + 2];
      bool        pushed = false;
      // The narrow phase against one face: whether it pushed, and whether the vertex is over the
      // face and close enough to count as resting on it. `known` is the closest point on a face of the
      // pose that followContact found just before: a pass moves only the cloth, so it is the one
      // this would find in `snap`.
      const auto narrow = [&](int t, bool& resting, const Closest* known = nullptr) {
        resting        = false;
        const Face& tri = tris[size_t(t)];
        if(tri.a == i || tri.b == i || tri.c == i)
          return;
        // the vertex where it is now: an earlier face in this pass may have moved it
        const float px = x[i * 3], py = x[i * 3 + 1], pz = x[i * 3 + 2];
        const float gap  = std::max(tri.offset, offset[i]);
        const float near = std::min(kDepth, gap + moved + kinematicStep);
        const V3&   blo  = boxLo[size_t(t)];
        const V3&   bhi  = boxHi[size_t(t)];
        if(px < blo[0] - near || px > bhi[0] + near || py < blo[1] - near || py > bhi[1] + near || pz < blo[2] - near || pz > bhi[2] + near)
          return;
        const V3& na = n0[size_t(t)];
        const V3& nb = n1[size_t(t)];
        if((na[0] == 0 && na[1] == 0 && na[2] == 0) || (nb[0] == 0 && nb[1] == 0 && nb[2] == 0))
          return;
        const Closest was = closest(tested[i * 3], tested[i * 3 + 1], tested[i * 3 + 2], cprev, tri.a, tri.b, tri.c, na);
        // a higher layer meets the outside (positive normal), a lower layer the inside; the same layer stays on the side it came from
        int direction;
        if(layer[i] > tri.layer)
        {
          if(was.signedDistance < -0.05f)
            return;
          direction = 1;
        }
        else if(layer[i] < tri.layer)
        {
          if(was.signedDistance > 0.05f)
            return;
          direction = -1;
        }
        else
          direction = was.signedDistance >= 0 ? 1 : -1;
        const Closest now    = known ? *known : closest(px, py, pz, snap, tri.a, tri.b, tri.c, nb);
        const float   target = float(direction) * gap;
        resting              = now.over && now.dist <= gap + near;
        if(direction > 0 ? now.signedDistance >= target : now.signedDistance <= target)
          return;
        if(now.dist > kDepth)
          return;
        // past the edge, not under the face: the closest point is mostly sideways of the vertex
        if(std::fabs(now.signedDistance) < 0.6f * now.dist)
          return;
        const float push  = std::min(std::fabs(target - now.signedDistance), kCap);
        const float along = target - now.signedDistance > 0 ? 1.f : -1.f;
        pushed = true;
        for(int k = 0; k < 3; k++)
          x[i * 3 + k] += nb[size_t(k)] * push * along;
        // friction sheds the part of this step's motion that slid along the face. Velocity is rebuilt
        // from the position afterwards, so the correction has to be on the position.
        const float dx = x[i * 3] - prev[i * 3], dy = x[i * 3 + 1] - prev[i * 3 + 1], dz = x[i * 3 + 2] - prev[i * 3 + 2];
        const float dn = dx * nb[0] + dy * nb[1] + dz * nb[2];
        const float fr = std::min(0.95f, std::max(0.f, (tri.friction + friction[i]) * 0.5f));
        x[i * 3] -= (dx - nb[0] * dn) * fr;
        x[i * 3 + 1] -= (dy - nb[1] * dn) * fr;
        x[i * 3 + 2] -= (dz - nb[2] * dn) * fr;
        // the contact absorbs the motion along the normal: the push made up for several substeps
        // of sinking, and divided by one substep it would read as a bounce
        prev[i * 3]     = x[i * 3] - (dx - nb[0] * dn) * (1 - fr);
        prev[i * 3 + 1] = x[i * 3 + 1] - (dy - nb[1] * dn) * (1 - fr);
        prev[i * 3 + 2] = x[i * 3 + 2] - (dz - nb[2] * dn) * (1 - fr);
      };
      // still riding the faces it rested on (or the one of the pose it slid onto): those alone, most
      // passes, the cloth's first so the body has the last word. Resting on cloth alone, the pose's
      // faces are still looked for (below), so a fold never presses the vertex into a limb unseen
      if((contact[i] >= 0 || clothContact[i] >= 0) && !lookAround)
      {
        bool resting = false;
        if(clothContact[i] >= 0)
        {
          narrow(clothContact[i], resting);
          if(!resting)
            clothContact[i] = -1;
        }
        if(contact[i] >= 0)
        {
          Closest   now{};
          V3        n{};
          const int at = followContact(i, contact[i], now, n);
          if(at >= 0)
            contact[i] = at;
          narrow(contact[i], resting, &now);
          if(resting)
          {
            budget[i] = 0;
            for(int c = 0; c < 3; c++)
              tested[i * 3 + c] = x[i * 3 + c];
            continue;
          }
          contact[i] = -1;
        }
      }
      const float reach = std::max(offset[i], maxOffset) + kSearch;
      float       lo[3], hi[3];
      for(int c = 0; c < 3; c++)
      {
        lo[c] = std::min(x[i * 3 + c], tested[i * 3 + c]) - reach;
        hi[c] = std::max(x[i * 3 + c], tested[i * 3 + c]) + reach;
      }
      float nearest = reach;
      if(++stamp == 0)
      {
        std::fill(seen.begin(), seen.end(), 0u);
        stamp = 1;
      }
      const auto test = [&](const Item& item) {
        if(item.a == i || item.b == i || item.c == i)
          return;
        if(item.mesh == meshOf[i])
        {
          if(!item.self)
            return;
          const int* first = ring2.data() + ring2Start[size_t(i)];
          const int* last  = ring2.data() + ring2Start[size_t(i) + 1];
          if(std::binary_search(first, last, item.a) || std::binary_search(first, last, item.b) || std::binary_search(first, last, item.c))
            return;
        }
        const float ex = std::max(0.f, std::max(item.lo[0] - qx, qx - item.hi[0])), ey = std::max(0.f, std::max(item.lo[1] - qy, qy - item.hi[1])),
                    ez = std::max(0.f, std::max(item.lo[2] - qz, qz - item.hi[2]));
        const float toBox = std::sqrt(ex * ex + ey * ey + ez * ez);
        nearest           = std::min(nearest, toBox);
        if(toBox > reach)
          return;
        if(seen[size_t(item.tri)] == stamp)
          return;
        seen[size_t(item.tri)] = stamp;
        bool resting           = false;
        narrow(item.tri, resting);
        if(resting)
          (tris[size_t(item.tri)].dynamic ? clothContact : contact)[i] = item.tri;
      };
      // new contacts with cloth are looked for every few passes: a face a vertex rests on is kept
      // above, and what is near a vertex of a cloth is mostly its own neighbourhood. The pose's faces
      // come last, so the body has the last word over a fold pressing the vertex into it
      if(!dynamicGrid.empty() && lookAround)
        dynamicGrid.query(lo, hi, test);
      if(!staticGrid.empty())
        staticGrid.query(lo, hi, test);
      // clear of everything by `nearest`: no need to look again until that much has moved
      budget[i] = pushed || contact[i] >= 0 || clothContact[i] >= 0 ? 0.f : std::max(0.f, nearest - std::max(offset[i], maxOffset) - 0.05f);
      for(int c = 0; c < 3; c++)
        tested[i * 3 + c] = x[i * 3 + c];
    }
  });
  cprev = x;
}

void Island::State::frame(float t0, float t1, float extraDamping, float seconds)
{
  const int   n  = substeps;
  const float dt = (seconds > 0 ? seconds : opts.frameDt) / float(n);
  // where the pose takes what it holds by the end of the frame, and how far: the colliders' cells
  // for the whole frame cover that sweep
  endPos = x;
  kinematicAt(t1, endPos, nullptr);
  kinematicMove = 0;
  for(int i : kinematicVerts)
  {
    const float dx = endPos[i * 3] - x[i * 3], dy = endPos[i * 3 + 1] - x[i * 3 + 1], dz = endPos[i * 3 + 2] - x[i * 3 + 2];
    kinematicMove = std::max(kinematicMove, dx * dx + dy * dy + dz * dz);
  }
  kinematicMove = std::sqrt(kinematicMove);
  if(!staticTris.empty())
    staticGrid.build(tris, staticTris, x, endPos, 0.f);
  if(opts.recoverLost)
    recoverLost();
  std::vector<float> frameStart = x;
  // which substeps collide: enough passes that nothing moves more than kPassSpacing between two,
  // spread over the frame with the last one at its end
  const int passes = std::min(n, std::max(collisions, int(std::ceil(std::max(lastMove, kinematicMove) / kPassSpacing))));
  int       nextCollision = n - 1 - (passes - 1) * (n / passes);
  for(int s = 0; s < n; s++)
  {
    const float ts = t0 + (t1 - t0) * float(s + 1) / float(n);
    prev.swap(x);
    x = prev;
    kinematicAt(ts, x, &anchor);
    for(int i : kinematicVerts)
      for(int k = 0; k < 3; k++)
        vel[i * 3 + k] = 0;
    for(int i : freeVerts)
      physics::particles::integrate(&x[size_t(i) * 3], &vel[size_t(i) * 3], &grav[size_t(i) * 3], std::exp(-(dampRate[i] + extraDamping) * dt), dt, kMaxSpeed);
    std::fill(lam.begin(), lam.end(), 0.0);
    std::fill(lamAtt.begin(), lamAtt.end(), 0.0);
    for(int it = 0; it < std::max(1, opts.iterations); it++)
      solveEdges(dt);
    for(int it = 0; it < opts.stretchSweeps; it++)
      solveEdges(dt, true);
    holdContacts(true);
    if(s == nextCollision)
    {
      snap = x;
      collide(kinematicMove / float(passes));
      // the pushes stretched edges; one more sweep shares them out before velocities are read
      solveEdges(dt);
      holdContacts(false);
      nextCollision += std::max(1, n / passes);
    }
    for(int i : freeVerts)
      for(int k = 0; k < 3; k++)
        vel[i * 3 + k] = (x[i * 3 + k] - prev[i * 3 + k]) / dt;
  }
  float moved = 0;
  for(int i : freeVerts)
  {
    const float dx = x[i * 3] - frameStart[i * 3], dy = x[i * 3 + 1] - frameStart[i * 3 + 1], dz = x[i * 3 + 2] - frameStart[i * 3 + 2];
    moved = std::max(moved, dx * dx + dy * dy + dz * dz);
  }
  lastMove = std::sqrt(moved);
}

Island::Island(std::vector<Surface> surfaces, const Options& opts)
    : m_state(std::make_unique<State>(std::move(surfaces), opts))
{
}

Island::~Island() = default;

void Island::frame(float t0, float t1, float extraDamping, float seconds)
{
  m_state->frame(t0, t1, extraDamping, seconds);
}

void Island::setGravity(const std::array<float, 3>& gravity)
{
  std::vector<float>& grav = m_state->grav;
  for(size_t i = 0; i + 2 < grav.size(); i += 3)
    std::copy(gravity.begin(), gravity.end(), grav.begin() + long(i));
}

bool Island::setTarget(size_t surface, const std::vector<Mat4>& bones)
{
  State& s = *m_state;
  if(surface >= s.kin.size() || !s.kin[surface].skinned || bones.size() != s.kin[surface].to.size())
    return false;
  Kinematic& k = s.kin[surface];
  // the kinematic vertices are where the last frame left them, at its t1 of the old sweep: that
  // pose is where the new one sets out from. A sweep cut short keeps the pose it reached.
  k.from = k.to;
  for(size_t b = 0; b < bones.size(); b++)
    k.to[b] = poseOf(bones[b]);
  return true;
}

float Island::lastMove() const
{
  return m_state->lastMove;
}

int Island::recovered() const
{
  return m_state->recovered;
}

const Model& Island::model() const
{
  return *m_state;
}

int Island::substepsPerFrame() const
{
  return m_state->substeps;
}

float Island::transitionSeconds() const
{
  return m_state->transition;
}

std::vector<std::vector<float>> Island::positions() const
{
  const State&                    s = *m_state;
  std::vector<std::vector<float>> out;
  for(size_t mi = 0; mi < s.meshes.size(); mi++)
  {
    const size_t o = size_t(s.base[mi]) * 3;
    out.emplace_back(s.x.begin() + long(o), s.x.begin() + long(o + s.meshes[mi].rest.size()));
  }
  return out;
}

void Island::settle()
{
  const Options& opts   = m_state->opts;
  const int            frames = std::max(1, int(std::lround(transitionSeconds() / opts.frameDt)));
  const auto cancelled = [&] { return opts.cancelled && opts.cancelled(); };
  for(int f = 0; f < frames; f++)
  {
    if(cancelled())
      return;
    frame(float(f) / float(frames), float(f + 1) / float(frames));
  }
  const int hold = std::max(1, int(std::ceil(opts.maxHold / opts.frameDt)));
  for(int h = 0; h < hold; h++)
  {
    if(cancelled())
      return;
    frame(1, 1, kHoldDamping);
    if(lastMove() < opts.restMove)
      break;
  }
}

std::vector<std::vector<float>> settle(const std::vector<Surface>& meshes, const Options& opts)
{
  Island sim(meshes, opts);
  sim.settle();
  return sim.positions();
}

}  // namespace fv::physics::cloth
