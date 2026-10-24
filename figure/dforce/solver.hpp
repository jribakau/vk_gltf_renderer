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
// step count: strength 0 is infinite mass, and a partial strength is blended once after the solve.
//
// Because the side is kept, a vertex that starts inside a lower layer (a fitted shape overlapping
// the body) would stay inside; those are moved out along the nearest face before the first step.
//
// `ClothSim` is the persistent state a live simulation advances a frame at a time; `settle()` runs
// the pose transition on it and then holds the pose until the cloth is still.

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
  // Seconds the pose transition takes; 0 picks them from how far the colliders move
  float transition = 0;
  // Seconds the cloth may take to come to rest once the pose is reached
  float maxHold = 3.f;
  // The largest move of a vertex over a frame, in cm, under which the cloth counts as still
  float restMove = 0.01f;
};

// The simulation state: constraints built once from the rest shapes, then advanced a frame at a time.
class ClothSim
{
public:
  ClothSim(std::vector<SettleMesh> meshes, const SettleOptions& opts = {});
  ~ClothSim();
  ClothSim(const ClothSim&)            = delete;
  ClothSim& operator=(const ClothSim&) = delete;

  // Advances one frame: the colliders and pinned vertices sweep from pose fraction `t0` to `t1`
  // (0 the memorized pose, 1 the target) while the cloth integrates. `extraDamping` is a further
  // velocity decay per second, for bringing a held pose to rest quickly.
  void frame(float t0, float t1, float extraDamping = 0);
  // How far the fastest dynamic vertex moved over the last frame, cm.
  float lastMove() const;
  // Substeps a frame takes and the seconds the transition takes, as the options or the motion set them.
  int   substepsPerFrame() const;
  float transitionSeconds() const;
  // Positions for every mesh, in the same order and space as `rest` / `posed`, with partial
  // strengths blended toward the target pose.
  std::vector<std::vector<float>> positions() const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

// Simulated positions for every mesh after the transition and the hold, in the same order and space
// as `rest` / `posed`.
std::vector<std::vector<float>> settle(const std::vector<SettleMesh>& meshes, const SettleOptions& opts = {});

}  // namespace fv::dforce
