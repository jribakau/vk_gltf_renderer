#pragma once

// What a cloth island is made of (docs/physics-plan.md step 12): a surface per mesh of the job, with
// its cage at the memorized pose and at the target, its polygons, the simulation's settings per vertex,
// and how it is skinned. A collider is a surface the cloth meets and doesn't move. dForce's SettleMesh,
// moved here unchanged.

#include <array>
#include <cstdint>
#include <vector>

#include <figure/math/xform.hpp>

namespace fv::physics::cloth {

struct Surface
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
  static Surface collider(std::vector<float> rest, std::vector<float> posed, std::vector<uint32_t> tris, int layer, float offset,
                          float friction, const std::vector<uint8_t>& collide = {});
};

// The figure's layer: below any a surface can set, so cloth always lies outside the body.
constexpr int kBodyLayer = -0x8000;
// The layer of a worn item with no simulation settings, DAZ's default "Collision Layer": the same
// as a surface that leaves it, so a belt over a skirt keeps the skirt under it and a bra under a
// dress keeps the dress outside, by the side each started on.
constexpr int kPlainLayer = 0;

// A skinned surface's vertices at these bones (its skin matrices in the mesh's space), in the job's
// space; an unskinned one gives its `posed`. `out` is xyz per vertex. Four influences, linearly: the
// body cloth meets is dForce's own, for parity, until physics-plan step 13c.
void skin(const Surface& m, const std::vector<Mat4>& bones, float* out);

}  // namespace fv::physics::cloth
