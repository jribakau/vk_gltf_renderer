#include <figure/dforce/solver.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>

#include <figure/jobs/pool.hpp>

namespace fv::dforce {

namespace {

// Grid cell for the collision faces, cm. Cloth and body polygons are about a centimetre.
constexpr float kCell = 2.f;
// A vertex further behind a face than this is left where it is (it came from the other side).
constexpr float kDepth = 2.f;
// The largest push one collision pass gives a vertex, cm.
constexpr float kCap = 2.f;
// Speed limit, cm/s: a vertex that escapes a fold does not fly off.
constexpr float kMaxSpeed = 400.f;
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
// Substeps a frame is cut into: at least this many for the constraints to converge, at most this
// many however fast the colliders move. Collision passes are fewer; see frame().
constexpr int kMinSubsteps = 24, kMaxSubsteps = 48;
// The pose transition: colliders move at most this fast, within these bounds, seconds.
constexpr float kTransitionSpeed = 60.f, kMinTransition = 0.25f, kMaxTransition = 1.5f;
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

using V3 = std::array<float, 3>;

// A 3x4 float matrix (column-major, 3 rows), for skinning in bulk.
struct Mat34
{
  float m[12];
  static Mat34 of(const Mat4& a)
  {
    Mat34 o;
    for(int c = 0; c < 4; c++)
      for(int r = 0; r < 3; r++)
        o.m[c * 3 + r] = float(a.e[c * 4 + r]);
    return o;
  }
};

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

int cellOf(float p)
{
  return int(std::floor(p / kCell));
}

// A face as a collision grid holds it: its corners, mesh and box, so a scan touches nothing else
// until a face is close.
struct Item
{
  int     tri, a, b, c, mesh;
  uint8_t self;
  float   lo[3], hi[3];
};

// A spatial hash of faces over their boxes: a flat open-addressed table of cells, each with a run
// of indices into the face table, built in two counting passes and reused frame to frame.
class TriGrid
{
public:
  // Each of the triangles `ids` covers the cells of its box over both `a` and `b`, grown by `pad`.
  void build(const std::vector<Tri>& tris, const std::vector<int>& ids, const std::vector<float>& a, const std::vector<float>& b, float pad)
  {
    const size_t n = ids.size();
    m_empty        = n == 0;
    m_face.resize(n);
    m_lo.resize(n * 3);
    m_hi.resize(n * 3);
    size_t entries = 0;
    for(size_t t = 0; t < n; t++)
    {
      const Tri& tri = tris[size_t(ids[t])];
      Item&      it  = m_face[t];
      it.tri         = ids[t];
      it.a           = tri.a;
      it.b           = tri.b;
      it.c           = tri.c;
      it.mesh        = tri.mesh;
      it.self        = tri.self;
      for(int c = 0; c < 3; c++)
      {
        it.lo[c] = INFINITY;
        it.hi[c] = -INFINITY;
      }
      for(int k = 0; k < 6; k++)
      {
        const int   v = k % 3 == 0 ? tri.a : k % 3 == 1 ? tri.b : tri.c;
        const auto& p = k < 3 ? a : b;
        for(int c = 0; c < 3; c++)
        {
          it.lo[c] = std::min(it.lo[c], p[v * 3 + c]);
          it.hi[c] = std::max(it.hi[c], p[v * 3 + c]);
        }
      }
      size_t cells = 1;
      for(int c = 0; c < 3; c++)
      {
        it.lo[c] -= pad;
        it.hi[c] += pad;
        m_lo[t * 3 + c] = cellOf(it.lo[c]);
        m_hi[t * 3 + c] = cellOf(it.hi[c]);
        cells *= size_t(m_hi[t * 3 + c] - m_lo[t * 3 + c] + 1);
      }
      entries += cells;
    }
    size_t slots = 64;
    while(slots < entries * 2)
      slots *= 2;
    m_mask = slots - 1;
    m_keys.assign(slots, kEmpty);
    m_count.assign(slots, 0);
    for(size_t t = 0; t < n; t++)
      eachCell(t, [&](uint64_t key) { m_count[insert(key)]++; });
    m_start.resize(slots + 1);
    m_start[0] = 0;
    for(size_t s = 0; s < slots; s++)
      m_start[s + 1] = m_start[s] + m_count[s];
    m_items.resize(entries);
    std::fill(m_count.begin(), m_count.end(), 0);
    for(size_t t = 0; t < n; t++)
      eachCell(t, [&](uint64_t key) {
        const size_t s                                     = find(key);
        m_items[size_t(m_start[s]) + size_t(m_count[s]++)] = int(t);
      });
  }

  // Calls f(item) for every face in the cells of the box; a face in several of them is reported
  // several times.
  template <class F>
  void query(const float lo[3], const float hi[3], F&& f) const
  {
    const int x0 = cellOf(lo[0]), x1 = cellOf(hi[0]), y0 = cellOf(lo[1]), y1 = cellOf(hi[1]), z0 = cellOf(lo[2]), z1 = cellOf(hi[2]);
    for(int ix = x0; ix <= x1; ix++)
      for(int iy = y0; iy <= y1; iy++)
        for(int iz = z0; iz <= z1; iz++)
        {
          const size_t s = find(keyOf(ix, iy, iz));
          if(s == kNone)
            continue;
          for(int k = m_start[s]; k < m_start[s + 1]; k++)
            f(m_face[size_t(m_items[size_t(k)])]);
        }
  }
  bool empty() const { return m_empty; }

private:
  static constexpr uint64_t kEmpty = ~uint64_t(0);
  static constexpr size_t   kNone  = ~size_t(0);

  static uint64_t keyOf(int ix, int iy, int iz)
  {
    return (uint64_t(uint32_t(ix) & 0x1FFFFF) << 42) | (uint64_t(uint32_t(iy) & 0x1FFFFF) << 21) | uint64_t(uint32_t(iz) & 0x1FFFFF);
  }
  static size_t hash(uint64_t key)
  {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return size_t(key);
  }
  size_t insert(uint64_t key)
  {
    size_t s = hash(key) & m_mask;
    while(m_keys[s] != kEmpty && m_keys[s] != key)
      s = (s + 1) & m_mask;
    m_keys[s] = key;
    return s;
  }
  size_t find(uint64_t key) const
  {
    size_t s = hash(key) & m_mask;
    while(m_keys[s] != kEmpty)
    {
      if(m_keys[s] == key)
        return s;
      s = (s + 1) & m_mask;
    }
    return kNone;
  }
  template <class F>
  void eachCell(size_t t, F&& f) const
  {
    for(int ix = m_lo[t * 3]; ix <= m_hi[t * 3]; ix++)
      for(int iy = m_lo[t * 3 + 1]; iy <= m_hi[t * 3 + 1]; iy++)
        for(int iz = m_lo[t * 3 + 2]; iz <= m_hi[t * 3 + 2]; iz++)
          f(keyOf(ix, iy, iz));
  }

  bool                  m_empty = true;
  std::vector<Item>     m_face;
  std::vector<int>      m_items, m_lo, m_hi;
  std::vector<uint64_t> m_keys;
  std::vector<int>      m_count, m_start;
  size_t                m_mask = 0;
};

bool triNormal(const std::vector<float>& p, int a, int b, int c, V3& n)
{
  const float ax = p[b * 3] - p[a * 3], ay = p[b * 3 + 1] - p[a * 3 + 1], az = p[b * 3 + 2] - p[a * 3 + 2];
  const float bx = p[c * 3] - p[a * 3], by = p[c * 3 + 1] - p[a * 3 + 1], bz = p[c * 3 + 2] - p[a * 3 + 2];
  const float x = ay * bz - az * by, y = az * bx - ax * bz, z = ax * by - ay * bx;
  const float len = std::sqrt(x * x + y * y + z * z);
  if(len < 1e-8f)
    return false;
  n = {x / len, y / len, z / len};
  return true;
}

struct Closest
{
  float signedDistance, dist;
  // the vertex projects onto the face: it rests on the face, and a face holding it keeps it while
  // this holds (a vertex crossing an edge is held by one face or the next all the way)
  bool over;
};

// Closest point of `q` on the triangle in `tri`, and the signed distance along `n`.
Closest closest(float qx, float qy, float qz, const std::vector<float>& tri, int a, int b, int c, const V3& n)
{
  const float abx = tri[b * 3] - tri[a * 3], aby = tri[b * 3 + 1] - tri[a * 3 + 1], abz = tri[b * 3 + 2] - tri[a * 3 + 2];
  const float acx = tri[c * 3] - tri[a * 3], acy = tri[c * 3 + 1] - tri[a * 3 + 1], acz = tri[c * 3 + 2] - tri[a * 3 + 2];
  const float aqx = qx - tri[a * 3], aqy = qy - tri[a * 3 + 1], aqz = qz - tri[a * 3 + 2];
  const float d00 = abx * abx + aby * aby + abz * abz;
  const float d01 = abx * acx + aby * acy + abz * acz;
  const float d11 = acx * acx + acy * acy + acz * acz;
  const float d20 = abx * aqx + aby * aqy + abz * aqz;
  const float d21 = acx * aqx + acy * aqy + acz * aqz;
  const float den = d00 * d11 - d01 * d01;
  float       bv = 0, cv = 0;
  bool        over = false;
  if(den > 1e-12f)
  {
    bv       = (d11 * d20 - d01 * d21) / den;
    cv       = (d00 * d21 - d01 * d20) / den;
    over     = bv >= -1e-3f && cv >= -1e-3f && bv + cv <= 1 + 1e-3f;
    bv       = std::min(1.f, std::max(0.f, bv));
    cv       = std::min(1.f, std::max(0.f, cv));
    if(bv + cv > 1)
    {
      const float s = bv + cv;
      bv /= s;
      cv /= s;
    }
  }
  const float cx = tri[a * 3] + abx * bv + acx * cv;
  const float cy = tri[a * 3 + 1] + aby * bv + acy * cv;
  const float cz = tri[a * 3 + 2] + abz * bv + acz * cv;
  const float dx = qx - cx, dy = qy - cy, dz = qz - cz;
  return {dx * n[0] + dy * n[1] + dz * n[2], std::sqrt(dx * dx + dy * dy + dz * dz), over};
}

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

BonePose poseOf(const Mat4& m)
{
  BonePose b;
  b.t = {m.e[12], m.e[13], m.e[14]};
  b.r = rotationOf(m);
  const Vec3   c0{m.e[0], m.e[1], m.e[2]}, c1{m.e[4], m.e[5], m.e[6]}, c2{m.e[8], m.e[9], m.e[10]};
  const double det = c0.x * (c1.y * c2.z - c1.z * c2.y) - c0.y * (c1.x * c2.z - c1.z * c2.x) + c0.z * (c1.x * c2.y - c1.y * c2.x);
  b.s = {length(c0) * (det < 0 ? -1 : 1), length(c1), length(c2)};
  return b;
}

// The move from one pose to the other is a screw: a turn about an axis through a pivot and a slide
// along that axis. A skin matrix is a turn about its joint, so the pivot is the joint, and the sweep
// turns the limb about it; interpolating translation and rotation apart would turn it about the
// mesh's origin and carry the joint off along an arc of its own.
Mat4 sweepAt(const BonePose& from, const BonePose& to, float t)
{
  if(t >= 1)
    return compose(to.t, to.r, to.s);
  if(t <= 0)
    return compose(from.t, from.r, from.s);
  const Vec3 scale = from.s + (to.s - from.s) * t;
  // the move applied after `from`: rotation rd about an axis, translation td
  Quat rd = to.r * from.r.inverse();
  if(rd.w < 0)
    rd = {-rd.x, -rd.y, -rd.z, -rd.w};
  const Vec3   td      = to.t - rotate(from.t, rd);
  const double sinHalf = std::sqrt(std::max(0.0, 1 - rd.w * rd.w));
  const Quat   ru      = slerp(Quat{}, rd, t);
  if(sinHalf < 1e-6)
    return compose(from.t + td * t, (ru * from.r).normalized(), scale);
  const Vec3   axis  = Vec3{rd.x, rd.y, rd.z} * (1 / sinHalf);
  const double along = dot(td, axis);
  const Vec3   perp  = td - axis * along;
  // the pivot: the point in the plane across the axis the turn carries by `perp`
  const double tanHalf = sinHalf / rd.w;
  const Vec3   across{axis.y * perp.z - axis.z * perp.y, axis.z * perp.x - axis.x * perp.z, axis.x * perp.y - axis.y * perp.x};
  const Vec3   pivot = perp * 0.5 + across * (0.5 / tanHalf);
  const Vec3   tu    = pivot - rotate(pivot, ru) + axis * (along * t);
  return compose(rotate(from.t, ru) + tu, (ru * from.r).normalized(), scale);
}


void skinMesh(const SettleMesh& m, const std::vector<Mat4>& bones, float* out)
{
  const size_t n = m.vertexCount();
  if(!m.skinned() || bones.size() != m.bones.size())
  {
    std::copy(m.posed.begin(), m.posed.end(), out);
    return;
  }
  std::vector<Mat34> at(bones.size());
  for(size_t b = 0; b < bones.size(); b++)
    at[b] = Mat34::of(m.toSpace * bones[b]);
  for(size_t v = 0; v < n; v++)
  {
    float blend[12] = {};
    float wsum      = 0;
    for(size_t j = 0; j < 4; j++)
    {
      const float  w    = m.skinWeight[v * 4 + j];
      const size_t bone = m.skinIndex[v * 4 + j];
      if(w == 0 || bone >= at.size())
        continue;
      wsum += w;
      for(int c = 0; c < 12; c++)
        blend[c] += at[bone].m[c] * w;
    }
    const float px = m.restLocal[v * 3], py = m.restLocal[v * 3 + 1], pz = m.restLocal[v * 3 + 2];
    if(wsum <= 0)
    {
      const Vec3 p = m.toSpace.transformPoint({px, py, pz});
      out[v * 3]     = float(p.x);
      out[v * 3 + 1] = float(p.y);
      out[v * 3 + 2] = float(p.z);
      continue;
    }
    out[v * 3]     = blend[0] * px + blend[3] * py + blend[6] * pz + blend[9];
    out[v * 3 + 1] = blend[1] * px + blend[4] * py + blend[7] * pz + blend[10];
    out[v * 3 + 2] = blend[2] * px + blend[5] * py + blend[8] * pz + blend[11];
  }
}

struct ClothSim::Impl : ClothModel
{
  std::vector<SettleMesh> meshes;
  SettleOptions           opts;
  std::vector<Kinematic>  kin;

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
  float               maxOffset  = 0;
  float               minOffset  = 0.05f;

  Impl(std::vector<SettleMesh> in, const SettleOptions& o)
      : ClothModel(ClothModel::build(in, o.air, o.reachLinks))
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

namespace {

// blocks of the vertex range, two per pool thread for balance; DSON vertex order is spatially
// coherent, so few edges cross blocks. An edge belongs to the block of its free vertices; one
// with a kinematic end goes with its free end, since the other is never written.
// Links seeds of each spacing to the seeds around them, as long as the shortest way between them over
// the mesh's stretch edges: its threads, which a shear doesn't lengthen, so a link leaves shear to the
// fabric, and over a surface curved in its rest shape the way is longer than the chord, so it can
// still unfold. As compliant as the softest edge on the way (a sheet's give over a patch doesn't grow
// with the patch, as a chain's would, but a soft panel on the way is still soft). A vertex takes the
// nearest few: a band that strands hang from would otherwise be linked to every strand.
void addReachLinks(ClothModel& model)
{
  const size_t nv = size_t(model.nv);
  struct Step
  {
    int   to;
    float length, alpha;
  };
  std::vector<std::vector<Step>> around(nv);
  for(const Edge& e : model.edges)
    if(e.kind == EdgeKind::eStretch && model.dynamic[size_t(e.a)] && model.dynamic[size_t(e.b)])
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
        model.edges.push_back({s, v, d, alpha, EdgeKind::eReach, 1.f, 0.f});
      });
  }
}

void partitionEdges(ClothModel& model)
{
  const int blocks = int(std::max<size_t>(1, 2 * (jobs::pool().size() + 1)));
  model.blockEdges.assign(size_t(blocks), {});
  model.crossingEdges.clear();
  const auto blockOf = [&](int v) { return int(size_t(v) * size_t(blocks) / size_t(std::max(1, model.nv))); };
  for(size_t e = 0; e < model.edges.size(); e++)
  {
    const Edge& edge = model.edges[e];
    const bool  fa = model.inv[edge.a] > 0, fb = model.inv[edge.b] > 0;
    if(!fa && !fb)
      continue;
    const int ba = blockOf(edge.a), bb = blockOf(edge.b);
    if(fa && fb && ba != bb)
      model.crossingEdges.push_back(int(e));
    else
      model.blockEdges[size_t(fa ? ba : bb)].push_back(int(e));
  }
  // stretch edges first in every list, so a sweep over them alone is a prefix
  const auto stretchFirst = [&](std::vector<int>& list) {
    const auto mid = std::stable_partition(list.begin(), list.end(), [&](int e) { return holdsLength(model.edges[size_t(e)].kind); });
    return int(mid - list.begin());
  };
  model.blockStretch.clear();
  for(auto& list : model.blockEdges)
    model.blockStretch.push_back(stretchFirst(list));
  model.crossingStretch = stretchFirst(model.crossingEdges);
}

}  // namespace

ClothModel ClothModel::build(const std::vector<SettleMesh>& meshes, float air, bool reachLinks)
{
  ClothModel model;
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
  for(const SettleMesh& m : meshes)
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
    const SettleMesh& m   = meshes[mi];
    const int         o   = base[mi];
    const int         n   = int(m.vertexCount());
    const bool        sim = m.kind != SettleMesh::Kind::eCollider;
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
  const auto add = [&](int a, int bb, float stiff, EdgeKind kind, float ratio = 1, float buckle = 0) {
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
    if(kind == EdgeKind::eStretch)
    {
      const size_t ma = size_t(meshOf[a]), mb = size_t(meshOf[bb]);
      scale = (meshes[ma].contraction[size_t(a - base[ma])] + meshes[mb].contraction[size_t(bb - base[mb])]) * 0.5f;
      alpha = compliance(stiff, kStretchCompliance);
    }
    else if(kind == EdgeKind::eShear)
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
    const SettleMesh& m = meshes[mi];
    const int         n = int(m.vertexCount());
    if(m.kind != SettleMesh::Kind::eCollider)
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
          add(a, bb, (m.stretch[va] + m.stretch[vb]) * 0.5f, EdgeKind::eStretch);
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
          add(g(mi, v0), g(mi, v2), sh, EdgeKind::eShear);
          add(g(mi, v1), g(mi, v3), sh, EdgeKind::eShear);
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
          add(p, q, stiff, EdgeKind::eBend, ratio, buck);
        }
      }
    }
    if(m.kind == SettleMesh::Kind::eAddon)
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
      const bool  shown = m.visible[ia] || m.visible[ib] || m.visible[ic] || m.kind == SettleMesh::Kind::eCollider;
      // a vertex on a colliding and a non-colliding surface collides, so the face needs all three
      if(shown && m.collide[ia] && m.collide[ib] && m.collide[ic])
        tris.push_back({int(mi), a, bb, c, lay, off > 0 ? off : 0.05f, fr, self, m.kind != SettleMesh::Kind::eCollider});
    }
  }
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
  for(const Tri& t : tris)
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
    addReachLinks(model);
  ringStart.assign(size_t(nv) + 1, 0);
  for(const Edge& e : edges)
    if(e.kind == EdgeKind::eStretch)
    {
      ringStart[size_t(e.a) + 1]++;
      ringStart[size_t(e.b) + 1]++;
    }
  for(int i = 0; i < nv; i++)
    ringStart[size_t(i) + 1] += ringStart[size_t(i)];
  ring.resize(size_t(ringStart[size_t(nv)]));
  std::vector<int> fill(ringStart.begin(), ringStart.end() - 1);
  for(const Edge& e : edges)
    if(e.kind == EdgeKind::eStretch)
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

// The pose's hold on each mesh: its bones, swept from the identity (the memorized pose) to the
// target the mesh came with.
void ClothSim::Impl::bindKinematics()
{
  kin.resize(meshes.size());
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const SettleMesh& m = meshes[mi];
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

void ClothSim::Impl::plan()
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

void ClothSim::Impl::kinematicAt(float t, std::vector<float>& out, std::vector<float>* anchors)
{
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    Kinematic& k = kin[mi];
    if(k.verts.empty() && (!anchors || k.soft.empty()))
      continue;
    const SettleMesh& m = meshes[mi];
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
void ClothSim::Impl::separateStart()
{
  dforce::separateStart(*this, x);
}

void separateStart(const ClothModel& model, std::vector<float>& x)
{
  const auto& tris     = model.tris;
  const auto& meshOf   = model.meshOf;
  const auto& layer    = model.layer;
  const auto& offset   = model.offset;
  const auto& collides = model.collides;
  TriGrid     grid;
  grid.build(tris, model.allTris, x, x, 0.f);
  std::vector<uint32_t> seen(tris.size(), 0);
  uint32_t              stamp = 0;
  for(int i : model.freeVerts)
  {
    if(!collides[i])
      continue;
    const float qx = x[i * 3], qy = x[i * 3 + 1], qz = x[i * 3 + 2];
    const float lo[3] = {qx - kDepth, qy - kDepth, qz - kDepth}, hi[3] = {qx + kDepth, qy + kDepth, qz + kDepth};
    stamp++;
    int   best = -1;
    float bestDist = kDepth, bestSigned = 0;
    V3    bestN{};
    grid.query(lo, hi, [&](const Item& item) {
      const int t = item.tri;
      if(seen[size_t(t)] == stamp)
        return;
      seen[size_t(t)] = stamp;
      const Tri& tri  = tris[size_t(t)];
      if(tri.mesh == meshOf[i] || tri.layer >= layer[i])
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
      x[i * 3 + k] += bestN[size_t(k)] * push;
  }
}

// Puts back free vertices that lost the surface they were outside of: one touching no face that
// lies behind the nearest static face of a lower layer, within kRecoverDepth, is moved out to the
// gap along that face and remembered as outside there, so the next pass keeps it out. The nearest
// face decides, as at the start, so a vertex just outside a thin limb is not sent through it.
void ClothSim::Impl::recoverLost()
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
        const Tri& tri  = tris[size_t(t)];
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
int ClothSim::Impl::followContact(int i, int t, Closest& now, V3& n) const
{
  const float qx = x[i * 3], qy = x[i * 3 + 1], qz = x[i * 3 + 2];
  const Tri&  tri = tris[size_t(t)];
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
    const Tri& nt = tris[size_t(cand[k])];
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

void ClothSim::Impl::holdContacts(bool grip)
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
      const Tri& held = tris[size_t(at)];
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

void ClothSim::Impl::solveEdges(float dt, bool stretchOnly)
{
  const float atScale = 1.f / (dt * dt);
  const auto  project = [&](int e) {
    const Edge& edge = edges[size_t(e)];
    // not in a full sweep (one follows a collision pass): see the top of solver.hpp
    if(edge.kind == EdgeKind::eReach && !stretchOnly)
      return;
    const int   a = edge.a, b = edge.b;
    const float wa = inv[a], wb = inv[b];
    const float w = wa + wb;
    const float dx = x[b * 3] - x[a * 3], dy = x[b * 3 + 1] - x[a * 3 + 1], dz = x[b * 3 + 2] - x[a * 3 + 2];
    const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if(len < 1e-8f)
      return;
    float alpha = edge.alpha;
    // a bend strut shorter than its buckling length folds at the buckling stiffness
    if(edge.kind == EdgeKind::eBend && len < edge.rest * edge.ratio)
      alpha = edge.buckle > 0 ? alpha / edge.buckle : 1e6f;
    const float C  = len - edge.rest;
    const float nx = dx / len, ny = dy / len, nz = dz / len;
    const float at = alpha * atScale;
    float       dl = float((-C - at * lam[size_t(e)]) / (w + at));
    // a reach link only ever pulls: what it has given back it can't give again
    if(edge.kind == EdgeKind::eReach)
      dl = float(std::min(lam[size_t(e)] + double(dl), 0.0) - lam[size_t(e)]);
    lam[size_t(e)] += dl;
    if(wa > 0)
    {
      x[a * 3] -= nx * wa * dl;
      x[a * 3 + 1] -= ny * wa * dl;
      x[a * 3 + 2] -= nz * wa * dl;
    }
    if(wb > 0)
    {
      x[b * 3] += nx * wb * dl;
      x[b * 3 + 1] += ny * wb * dl;
      x[b * 3 + 2] += nz * wb * dl;
    }
  };
  // sweeps alternate direction, so a correction reaches both ends of a chain in two of them
  const auto sweep = [&](const std::vector<int>& list, int count) {
    const size_t n = stretchOnly ? size_t(count) : list.size();
    if(forward)
      for(size_t k = 0; k < n; k++)
        project(list[k]);
    else
      for(size_t k = n; k-- > 0;)
        project(list[k]);
  };
  jobs::parallelFor(blockEdges.size(), 1, [&](size_t begin, size_t end) {
    for(size_t b = begin; b < end; b++)
      sweep(blockEdges[b], blockStretch[b]);
  });
  sweep(crossingEdges, crossingStretch);
  forward = !forward;
  // the tethers of partial strength: a distance-0 constraint to the skinned position, as compliant
  // as the strength makes it; each writes its own vertex only
  jobs::parallelFor(softVerts.size(), 1024, [&](size_t begin, size_t end) {
    for(size_t k = begin; k < end; k++)
    {
      const int   i  = softVerts[k];
      const float dx = x[i * 3] - anchor[i * 3], dy = x[i * 3 + 1] - anchor[i * 3 + 1], dz = x[i * 3 + 2] - anchor[i * 3 + 2];
      const float C  = std::sqrt(dx * dx + dy * dy + dz * dz);
      if(C < 1e-8f)
        continue;
      const float w  = inv[i];
      const float at = attAlpha[i] * atScale;
      const float dl = float((-C - at * lamAtt[size_t(i)]) / (w + at));
      lamAtt[size_t(i)] += dl;
      const float s = w * dl / C;
      x[i * 3] += dx * s;
      x[i * 3 + 1] += dy * s;
      x[i * 3 + 2] += dz * s;
    }
  });
}

// Pushes free vertices out of faces they were outside of when they last looked (`tested`). `snap`
// is where everything is now, so the faces of a cloth that collides with itself are read from one
// moment while its vertices move. A vertex far from every face looks again only once it or the
// colliders could have closed that distance; one resting on a face tests that face alone while it
// stays over it (or the face it slid onto), and looks around every few passes.
void ClothSim::Impl::collide(float kinematicStep)
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
      const Tri& tri = tris[t];
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
        const Tri& tri = tris[size_t(t)];
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

void ClothSim::Impl::frame(float t0, float t1, float extraDamping, float seconds)
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
    {
      const float damped = std::exp(-(dampRate[i] + extraDamping) * dt);
      for(int k = 0; k < 3; k++)
        vel[i * 3 + k] = (vel[i * 3 + k] + grav[i * 3 + k] * dt) * damped;
      const float sp = std::sqrt(vel[i * 3] * vel[i * 3] + vel[i * 3 + 1] * vel[i * 3 + 1] + vel[i * 3 + 2] * vel[i * 3 + 2]);
      if(sp > kMaxSpeed)
        for(int k = 0; k < 3; k++)
          vel[i * 3 + k] *= kMaxSpeed / sp;
      for(int k = 0; k < 3; k++)
        x[i * 3 + k] += vel[i * 3 + k] * dt;
    }
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

ClothSim::ClothSim(std::vector<SettleMesh> meshes, const SettleOptions& opts)
    : m_impl(std::make_unique<Impl>(std::move(meshes), opts))
{
}

ClothSim::~ClothSim() = default;

void ClothSim::frame(float t0, float t1, float extraDamping, float seconds)
{
  m_impl->frame(t0, t1, extraDamping, seconds);
}

void ClothSim::setGravity(const std::array<float, 3>& gravity)
{
  std::vector<float>& grav = m_impl->grav;
  for(size_t i = 0; i + 2 < grav.size(); i += 3)
    std::copy(gravity.begin(), gravity.end(), grav.begin() + long(i));
}

void ClothSim::setTarget(size_t mesh, const std::vector<Mat4>& bones)
{
  Impl& s = *m_impl;
  if(mesh >= s.kin.size() || !s.kin[mesh].skinned || bones.size() != s.kin[mesh].to.size())
    return;
  Kinematic& k = s.kin[mesh];
  // the kinematic vertices are where the last frame left them, at its t1 of the old sweep: that
  // pose is where the new one sets out from. A sweep cut short keeps the pose it reached.
  k.from = k.to;
  for(size_t b = 0; b < bones.size(); b++)
    k.to[b] = poseOf(bones[b]);
  targetSet();
}

float ClothSim::lastMove() const
{
  return m_impl->lastMove;
}

int ClothSim::recovered() const
{
  return m_impl->recovered;
}

const ClothModel& ClothSim::model() const
{
  return *m_impl;
}

int ClothSim::substepsPerFrame() const
{
  return m_impl->substeps;
}

float ClothSim::transitionSeconds() const
{
  return m_impl->transition;
}

std::vector<std::vector<float>> ClothSim::positions() const
{
  const Impl&                     s = *m_impl;
  std::vector<std::vector<float>> out;
  for(size_t mi = 0; mi < s.meshes.size(); mi++)
  {
    const size_t o = size_t(s.base[mi]) * 3;
    out.emplace_back(s.x.begin() + long(o), s.x.begin() + long(o + s.meshes[mi].rest.size()));
  }
  return out;
}

SettleMesh SettleMesh::collider(std::vector<float> rest, std::vector<float> posed, std::vector<uint32_t> tris, int layer, float offset,
                                float friction, const std::vector<uint8_t>& collide)
{
  SettleMesh m;
  const size_t n = rest.size() / 3;
  m.rest         = std::move(rest);
  m.posed        = std::move(posed);
  m.tris         = std::move(tris);
  m.strength.assign(n, 0.f);
  m.stretch.assign(n, 0.8f);
  m.shear.assign(n, 0.2f);
  m.bend.assign(n, 0.5f);
  m.buckling.assign(n, 0.05f);
  m.bucklingRatio.assign(n, 0.7f);
  m.density.assign(n, 180.f);
  m.contraction.assign(n, 1.f);
  m.damping.assign(n, 0.1f);
  m.friction.assign(n, friction);
  m.offset.assign(n, offset);
  m.layer.assign(n, int16_t(layer));
  m.selfCollide.assign(n, 0);
  m.collide = collide.size() == n ? collide : std::vector<uint8_t>(n, 1);
  m.visible.assign(n, 1);
  return m;
}

void ClothSim::settle()
{
  const SettleOptions& opts   = m_impl->opts;
  const int            frames = std::max(1, int(std::lround(transitionSeconds() / opts.frameDt)));
  for(int f = 0; f < frames; f++)
    frame(float(f) / float(frames), float(f + 1) / float(frames));
  const int hold = std::max(1, int(std::ceil(opts.maxHold / opts.frameDt)));
  for(int h = 0; h < hold; h++)
  {
    frame(1, 1, kHoldDamping);
    if(lastMove() < opts.restMove)
      break;
  }
}

std::vector<std::vector<float>> settle(const std::vector<SettleMesh>& meshes, const SettleOptions& opts)
{
  ClothSim sim(meshes, opts);
  sim.settle();
  return sim.positions();
}

}  // namespace fv::dforce
