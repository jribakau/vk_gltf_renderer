#pragma once

// A position-based settle for dForce surfaces. One job holds every dynamic mesh and every
// collider: collision layers are sided constraints, not separate solves.
//
// Substeps shrink until the fastest collider vertex moves less than half the gap, and the side a
// vertex is on is taken from where it was, so a limb sweeping through cloth pushes it out the
// side it started on. Pinning does not scale with that count: strength 0 is infinite mass, and a
// partial strength is blended once after the solve.
//
// Because the side is kept, a vertex that starts inside a lower layer (a fitted shape overlapping
// the body) would stay inside; those are moved out along the nearest face before the first step.
// A port of src/lib/dforce/solver.ts.

#include <array>
#include <cstdint>
#include <vector>

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

  size_t vertexCount() const { return rest.size() / 3; }
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
  // Forces the pose-transition substep count; 0 uses the motion rule.
  int substeps   = 0;
  int iterations = 4;
};

// Simulated positions for every mesh, in the same order and space as `rest` / `posed`.
std::vector<std::vector<float>> settle(const std::vector<SettleMesh>& meshes, const SettleOptions& opts = {});

}  // namespace fv::dforce
