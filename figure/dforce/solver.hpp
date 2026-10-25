#pragma once

// A position-based settle for dForce surfaces. One job holds every dynamic mesh and every
// collider: collision layers are sided constraints, not separate solves.
//
// The cloth is simulated in frames of `frameDt`, each cut into substeps with one constraint sweep
// per substep (small steps converge where many sweeps over a large step do not). Masses come from
// each surface's density and the area around the vertex, so the stiffness numbers are compliances in
// physical units and a heavy fabric hangs differently from a light one. Colliders and pinned vertices
// sweep from the memorized pose to the target pose along their bones (a limb turns instead of
// shrinking along a chord), and the side a vertex is on is taken from where it was, so a limb
// sweeping through cloth pushes it out the side it started on. Pinning does not scale with the
// step count: strength 0 is infinite mass, and a partial strength is a tether to the skinned
// position solved with the edges every substep, as compliant as the strength makes it (a skirt's
// band fades from pinned to free over a few rows without tearing).
//
// Because the side is kept, a vertex that starts inside a lower layer (a fitted shape overlapping
// the body) would stay inside; those are moved out along the nearest face before the first step.
//
// `ClothSim` is the persistent state: `settle()` runs the pose transition on it and then holds the
// pose until the cloth is still, and a live simulation gives it the pose the figure has now every
// frame and steps it once, so the cloth follows a drag.

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <figure/math/xform.hpp>

namespace fv::dforce {

struct SettleMesh
{
  // Morphed cage at the memorized pose, in the space the whole job shares (xyz per vertex).
  std::vector<float> rest;
  // The same cage skinned at the target pose.
  std::vector<float> posed;
  std::vector<uint32_t> tris;
  // Polygon boundaries: polygon i is polyVerts[polyStart[i] .. polyStart[i + 1]).
  std::vector<int32_t> polyStart{0}, polyVerts;
  enum class Kind
  {
    eCollider,  // collides only
    eDynamic,
    eAddon,  // simulated, its faces are not colliders
  } kind = Kind::eCollider;
  // 0 pinned, 1 free; already Dynamics Strength x influence
  std::vector<float>   strength, stretch, shear, bend, buckling, bucklingRatio, density, contraction, damping, friction, offset;
  std::vector<int16_t> layer;
  std::vector<uint8_t> selfCollide;
  // 0: simulated, but neither pushed by faces nor a face that pushes
  std::vector<uint8_t> collide;
  std::vector<uint8_t> visible;
  std::array<float, 3> gravity{0, 0, 0};

  // How the mesh is skinned, so the transition can turn its bones: four influences per vertex into
  // `bones`, each bone's skin matrix at the target pose in the mesh's own space (the identity at the
  // memorized pose), the mesh's rest cage in that space, and the matrix from it into the job's.
  // Without them the vertices move along straight lines between `rest` and `posed`.
  std::vector<uint16_t> skinIndex;
  std::vector<float>    skinWeight;
  std::vector<Mat4>     bones;
  std::vector<float>    restLocal;
  Mat4                  toSpace;

  size_t vertexCount() const { return rest.size() / 3; }
  bool   skinned() const
  {
    const size_t n = vertexCount();
    return !bones.empty() && skinIndex.size() == n * 4 && skinWeight.size() == n * 4 && restLocal.size() == n * 3;
  }
  // A collider from positions with the same value on every vertex.
  static SettleMesh collider(std::vector<float> rest, std::vector<float> posed, std::vector<uint32_t> tris, int layer, float offset,
                             float friction, const std::vector<uint8_t>& collide = {});
};

// The figure's layer: below any a surface can set, so cloth always lies outside the body.
constexpr int kBodyLayer = -0x8000;
// The layer of a worn item with no simulation settings, DAZ's default "Collision Layer": the same
// as a surface that leaves it, so a belt over a skirt keeps the skirt under it and a bra under a
// dress keeps the dress outside, by the side each started on.
constexpr int kPlainLayer = 0;

// What a settle is made of, built once from the meshes' rest shapes: every vertex of every mesh
// in one index space, the distance constraints, the collision faces and who moves what. Shared by
// the CPU solver and a GPU one, so both simulate the same cloth.
enum class EdgeKind
{
  eStretch,
  eShear,
  eBend,
};

struct Edge
{
  int      a, b;
  float    rest, alpha;  // compliance, cm per dyne
  EdgeKind kind;
  float    ratio, buckle;
};

struct Tri
{
  int   mesh, a, b, c;
  int   layer;
  float offset, friction;
  bool  self, dynamic;
};

struct ClothModel
{
  int                  nv = 0;
  std::vector<int>     base, meshOf;
  std::vector<float>   rest, posed, strength, dampRate, friction, offset, grav, inv;
  std::vector<int>     layer;
  std::vector<uint8_t> visible, collides, dynamic;
  std::vector<Edge>    edges;
  // the sweep in parallel: edges whose free vertices all lie in one block of the vertex range run
  // block by block on the pool, the few that cross blocks after them on one thread. Each list
  // starts with its stretch edges, this many, so a sweep over those alone is a prefix
  std::vector<std::vector<int>> blockEdges;
  std::vector<int>              crossingEdges;
  std::vector<int>              blockStretch;
  int                           crossingStretch = 0;
  std::vector<Tri>              tris;
  // faces the pose moves and faces the cloth moves
  std::vector<int> staticTris, dynamicTris, allTris;
  // each vertex's neighbours over stretch edges, and the neighbours of those, sorted: a cloth does
  // not collide with the faces around a vertex, whose planes it is in or next to
  std::vector<int> ringStart, ring, ring2Start, ring2;
  // per face, the faces of its mesh across each of its three edges (-1 at a border): a vertex held
  // on a face hands the hold to the next face when it slides across an edge
  std::vector<int> triNext;
  std::vector<int> kinematicVerts;  // every vertex a pose moves
  std::vector<int> freeVerts;       // every vertex the cloth moves
  // the free vertices of partial strength, tethered to their skinned position: each one's tether
  // compliance (cm per dyne, 0 for the rest), from kAttachSag and its strength
  std::vector<int>   softVerts;
  std::vector<float> attAlpha;
  float              maxOffset = 0, minOffset = 0.05f;

  static ClothModel build(const std::vector<SettleMesh>& meshes, float air);
};

// A bone's skin matrix taken apart, so a sweep between two poses can turn it.
struct BonePose
{
  Vec3 t, s;
  Quat r;
};
BonePose poseOf(const Mat4& m);
Mat4     sweepAt(const BonePose& from, const BonePose& to, float t);

// A skinned mesh's vertices at these bones (its skin matrices in the mesh's space), in the job's
// space; an unskinned mesh gives its `posed`. `out` is xyz per vertex.
void skinMesh(const SettleMesh& m, const std::vector<Mat4>& bones, float* out);

// Moves the free vertices of `x` (the model's rest shape, usually) that start inside, or within the
// gap of, a lower layer out to the gap along the nearest face of that layer, so a fitted shape that
// overlaps the body does not start inside it. Both solvers do this once before their first step.
void separateStart(const ClothModel& model, std::vector<float>& x);

struct SettleOptions
{
  // Air resistance, DAZ's 0.15
  float air = 0.15f;
  // One frame of simulated time, seconds
  float frameDt = 1.f / 30.f;
  // Substeps per frame; 0 picks them from the motion (a collider moves at most half a gap per substep)
  int substeps = 0;
  // Constraint sweeps per substep
  int iterations = 1;
  // Further sweeps per substep over the stretch edges alone: one sweep leaves a hanging cloth's
  // rows stretched where they carry what hangs below (twice their length under a real skirt's
  // band), and the stretch edges are a third of the constraints, so these converge it at a third
  // of a sweep's cost each. Five bring that skirt's worst edge to 1.3x and its 99th centile to 1.07.
  int stretchSweeps = 5;
  // Seconds the pose transition takes; 0 picks them from how far the colliders move
  float transition = 0;
  // Seconds the cloth may take to come to rest once the pose is reached
  float maxHold = 3.f;
  // Every frame, a free vertex touching nothing that lies behind the nearest face of a lower layer
  // within reach is put back outside it: cloth pulled through a limb by its edges between two
  // collision passes, or carried through by the sweep, would otherwise stay inside for good
  bool recoverLost = true;
  // The largest move of a vertex over a frame, in cm, under which the cloth counts as still
  float restMove = 0.01f;
};

// A simulation, however it runs: the CPU solver below, or a GPU one the renderer provides.
class ClothBackend
{
public:
  virtual ~ClothBackend() = default;
  virtual void  frame(float t0, float t1, float extraDamping = 0) = 0;
  virtual void  setTarget(size_t mesh, const std::vector<Mat4>& bones) = 0;
  virtual void  settle() = 0;
  virtual float lastMove() const = 0;
  virtual std::vector<std::vector<float>> positions() const = 0;
  // True when it must be driven on the thread that made it (a GPU queue), never on the pool.
  virtual bool ownerThreadOnly() const { return false; }
  // Live: one frame. After a setTarget the pose sweeps from the one it reached to the new target
  // (0 to 1); otherwise it holds where the last sweep ended (1), as a settle's hold frames do. A
  // sweep from 0 on every frame would set the colliders and the pinned cloth out from the pose
  // they memorized (the rest pose, after a settle) each frame and drag them through the cloth to
  // the target again.
  void step(float extraDamping = 0)
  {
    frame(m_targetPending ? 0.f : 1.f, 1, extraDamping);
    m_targetPending = false;
  }

protected:
  // For setTarget: the next step sweeps.
  void targetSet() { m_targetPending = true; }

private:
  bool m_targetPending = false;
};

// The simulation state: constraints built once from the rest shapes, then advanced a frame at a time.
class ClothSim final : public ClothBackend
{
public:
  ClothSim(std::vector<SettleMesh> meshes, const SettleOptions& opts = {});
  ~ClothSim() override;
  ClothSim(const ClothSim&)            = delete;
  ClothSim& operator=(const ClothSim&) = delete;

  // Advances one frame: the colliders and pinned vertices sweep from pose fraction `t0` to `t1`
  // (0 the pose they set out from, 1 the target) while the cloth integrates. `extraDamping` is a
  // further velocity decay per second, for bringing a held pose to rest quickly.
  void frame(float t0, float t1, float extraDamping = 0) override;
  // Live: a new target for a skinned mesh's bones (its skin matrices in the mesh's space, as
  // `SettleMesh::bones`). The pose its kinematic vertices have reached becomes where they set out
  // from, so the next frame(0, 1) sweeps them to the new target. Meshes not given keep theirs.
  void setTarget(size_t mesh, const std::vector<Mat4>& bones) override;
  // The pose transition over its frames, then the hold until the cloth is still (or `maxHold`).
  void settle() override;
  // How far the fastest dynamic vertex moved over the last frame, cm.
  float lastMove() const override;
  // Vertices the last frame found lost inside a lower layer and put back (SettleOptions::recoverLost).
  int recovered() const;
  // Substeps a frame takes and the seconds the transition takes, as the options or the motion set them.
  int   substepsPerFrame() const;
  float transitionSeconds() const;
  // Positions for every mesh, in the same order and space as `rest` / `posed`.
  std::vector<std::vector<float>> positions() const override;
  const ClothModel&               model() const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

// Simulated positions for every mesh after the transition and the hold, in the same order and space
// as `rest` / `posed`.
std::vector<std::vector<float>> settle(const std::vector<SettleMesh>& meshes, const SettleOptions& opts = {});

}  // namespace fv::dforce
