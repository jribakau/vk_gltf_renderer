#pragma once

// What a cloth island is made of (docs/physics-plan.md step 12): a surface per mesh of the job, with
// its cage at the memorized pose and at the target, its polygons, the simulation's settings per vertex,
// and how it is skinned. A collider is a surface the cloth meets and doesn't move.

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/physics/cloth/geometry.hpp>
#include <figure/physics/skin.hpp>

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
  // The skin as the drawn mesh's skin pass has it (physics::CageSkin): eight influences per vertex
  // where `skinIndex8` isn't empty, and the dual quaternion share mixed in, so the body the cloth
  // meets is the body drawn (physics-plan step 13c). All empty: the four above, linearly (a job file
  // of version 1, a test's surface).
  std::vector<uint16_t> skinIndex8;
  std::vector<float>    skinWeight8;
  float                 dualQuat = 0;
  std::vector<float>    dualQuatShare;

  size_t vertexCount() const { return rest.size() / 3; }
  bool   skinned() const
  {
    const size_t n = vertexCount();
    return !bones.empty() && skinIndex.size() == n * 4 && skinWeight.size() == n * 4 && restLocal.size() == n * 3;
  }
  // Skinned as drawn: eight influences or a dual quaternion share, of the right sizes
  bool drawnSkin() const
  {
    const size_t n = vertexCount();
    if(!skinned() || !(skinIndex8.size() == n * 8 || dualQuat > 0 || dualQuatShare.size() == n))
      return false;
    return (skinIndex8.empty() || (skinIndex8.size() == n * 8 && skinWeight8.size() == n * 8))
           && (dualQuatShare.empty() || dualQuatShare.size() == n);
  }
  SkinView skinView() const { return {skinIndex, skinIndex8, skinWeight, skinWeight8, dualQuatShare, dualQuat}; }
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

// The same surface but for its pose: everything a cloth model is built from (the rest cage, the polygons,
// the settings per vertex, the skin and its space) is equal, and only `posed`, the bones' matrices (not
// how many) and `gravity` may differ. A simulation kept between settles keeps its model for such a
// surface (cloth::Backend::restart). A field added to Surface that the model reads goes in here too.
bool sameButPose(const Surface& a, const Surface& b);
// The first of those that differs ("rest", "strength", ...), or null when none does: for the log. With
// `colliderRest`, a collider's rest cage (`rest`, `restLocal`, of the same size) counts as its pose too:
// a model holds nothing of it but its copy (the start, and the push out of the body, read it), and a
// body's contact dents change it with every pose.
const char* poseFreeDifference(const Surface& a, const Surface& b, bool colliderRest = false);

// A skinned surface at one pose of its bones (their skin matrices in the mesh's space): each vertex's
// skin into the job's space. As drawn where the surface has the drawn skin (Surface::drawnSkin: eight
// influences, the dual quaternion share), else four influences, linearly, to the bit as before. The
// surface and the bones outlive it.
class SurfaceSkin
{
public:
  SurfaceSkin(const Surface& m, std::vector<Mat4> bones);
  // vertex v at the pose, in the job's space (its rest cage there when it has no weight)
  Vec3 at(size_t v) const;
  // vertex v's skin into the job's space (toSpace times its blended matrix); false when it has no weight
  bool matrix(size_t v, Mat4& out) const;

private:
  const Surface&            m_mesh;
  std::vector<Mat4>         m_bones;  // in the mesh's space
  std::vector<Mat34>        m_linear;  // toSpace * bone: the four, linearly
  std::vector<Mat4>         m_linear4;  // the same in doubles, for matrix()
  std::optional<SkinBlend>  m_blend;   // the drawn skin
};

// A skinned surface's vertices at these bones (its skin matrices in the mesh's space), in the job's
// space, as SurfaceSkin has them; an unskinned one gives its `posed`. `out` is xyz per vertex.
void skin(const Surface& m, const std::vector<Mat4>& bones, float* out);

}  // namespace fv::physics::cloth
