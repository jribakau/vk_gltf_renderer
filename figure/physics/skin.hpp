#pragma once

// Bind-space drape from a simulated mesh-local position.
//
// Linear blend skinning is (sum w_i M_i) p, and the 3x3 of that blend includes scale: it shrinks at
// a joint. The drape offset is that 3x3 inverted, using the same weights the GPU skin uses. With
// DAZ's skinning (gpu-figures-plan.md F3) a vertex's dual quaternion part is affine too (the blended
// rotation times the blended stretch, and a translation), so its share mixes into the same matrix.
// A near-singular blend (a collapsed joint) leaves the vertex undraped. The result is exact at
// cage vertices; subdivided vertices are the stencil of those deltas, then skinned with their own
// weights. A port of src/lib/dforce/skin.ts.

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics {

// How a mesh is skinned: each bone's skin matrix (its world matrix times its inverse bind), and the
// mesh's bind matrix and its inverse.
struct SkinPose
{
  std::vector<Mat4> bones;
  Mat4              bind, bindInverse;
};

// The node each mesh is drawn by, or -1.
std::vector<int> nodeOfMesh(const SceneGraph& scene);

// A skinned mesh's pose as the figure has it now: three's 'attached' skinning, matrixWorld⁻¹ ·
// bone · inverse · bindMatrix, with the mesh's world as it is now (the host scales the root to
// metres after the bind, and rebinds read that scale). `worlds` is scene.worlds().
SkinPose currentSkin(const SceneGraph& scene, const std::vector<Mat4>& worlds, const std::vector<int>& nodeOf, int mesh);

// Points through a matrix, xyz per point.
std::vector<float> transformPoints(const Mat4& m, const std::vector<float>& src);

// How a mesh's cage is skinned: the four strongest influences per cage vertex, renormalised (the
// solver's pose transition turns them), and the eight strongest and the file's dual quaternion share,
// as the drawn mesh's skin pass blends them.
struct CageSkin
{
  std::vector<uint16_t> index;  // four per vertex
  std::vector<float>    weight;
  std::vector<uint16_t> index8;  // eight per vertex; empty: the four
  std::vector<float>    weight8;
  float                 dualQuat = 0;
  std::vector<float>    dualQuatShare;  // per vertex; empty: dualQuat for every vertex
};
CageSkin cageSkin(const mesh::SkinInput& in);

// A skin's influences seen in place: a CageSkin's, or a cloth surface's (cloth::Surface::skinView).
// Four per vertex, and eight where `index8` isn't empty; the dual quaternion share per vertex, or
// `dualQuat` for every vertex when `dualQuatShare` is empty.
struct SkinView
{
  std::span<const uint16_t> index, index8;
  std::span<const float>    weight, weight8, dualQuatShare;
  float                     dualQuat = 0;

  static SkinView of(const CageSkin& s) { return {s.index, s.index8, s.weight, s.weight8, s.dualQuatShare, s.dualQuat}; }
  bool            dualQuats() const { return dualQuat > 0 || !dualQuatShare.empty(); }
};

namespace detail {
// A skin matrix as a unit dual quaternion (x y z w each) and the stretch left over (rows), as
// rs::jointDualQuat takes the GPU's joints apart, in doubles
struct JointDq
{
  std::array<double, 4> q{0, 0, 0, 1}, d{0, 0, 0, 0};
  std::array<double, 9> k{1, 0, 0, 0, 1, 0, 0, 0, 1};
};
}  // namespace detail

// One pose's skin as the drawn mesh's skin pass has it: the skin matrices (each bone's, in the mesh's
// space) taken apart for the dual quaternion share once, then each vertex's skin as one matrix, the
// weighted sum of its matrices mixed with its dual quaternion skin by its share. skinLocal and
// drapeDelta blend with it, and so does the cloth's kinematic sweep. The views and the matrices are
// borrowed: they outlive it.
class SkinBlend
{
public:
  SkinBlend(std::span<const Mat4> bones, const SkinView& skin);
  // vertex v's skin into `out` (column-major, as Mat4::e); the vertex's weight total, which is 0 (and
  // `out` meaningless) when it has none
  double vertex(size_t v, std::array<double, 16>& out) const;

private:
  std::span<const Mat4>        m_bones;
  SkinView                     m_skin;
  std::vector<detail::JointDq> m_dqs;
};

// Each bone's skin matrix in the mesh's space: bindInverse * bone * bind.
std::vector<Mat4> skinMatrices(const SkinPose& pose);

// Mesh-local skinned positions: bindInverse * (sum w_i bone_i) * bind * p, mixed with the dual
// quaternion skin by the vertex's share.
std::vector<float> skinLocal(const std::vector<float>& rest, const CageSkin& skin, const SkinPose& pose);
// The same for some of the vertices alone, xyz per vertex listed (soft flesh's lobes, every frame).
std::vector<float> skinLocal(std::span<const float> rest, const CageSkin& skin, const SkinPose& pose, std::span<const uint32_t> vertices);

// Cage deltas whose skinning approximates `simulated` (mesh-local, same space as skinLocal). A
// vertex whose blended 3x3 is singular keeps a zero delta.
std::vector<float> drapeDelta(const std::vector<float>& rest, const std::vector<float>& simulated, const CageSkin& skin, const SkinPose& pose);
// The same for some of the vertices alone: `simulated` and the deltas are xyz per vertex listed (Live's
// drape where it rides the body, every frame the figure moves).
std::vector<float> drapeDelta(std::span<const float> rest, std::span<const float> simulated, const CageSkin& skin, const SkinPose& pose,
                              std::span<const uint32_t> vertices);

}  // namespace fv::physics
