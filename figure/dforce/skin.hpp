#pragma once

// Bind-space drape from a simulated mesh-local position.
//
// Linear blend skinning is (sum w_i M_i) p, and the 3x3 of that blend includes scale: it shrinks at
// a joint. The drape offset is that 3x3 inverted, using the same four weights the GPU skin uses.
// A near-singular blend (a collapsed joint) leaves the vertex undraped. The result is exact at
// cage vertices; subdivided vertices are the stencil of those deltas, then skinned with their own
// weights. A port of src/lib/dforce/skin.ts.

#include <cstdint>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/mesh/mesh_build.hpp>

namespace fv::dforce {

// How a mesh is skinned: each bone's skin matrix (its world matrix times its inverse bind), and the
// mesh's bind matrix and its inverse.
struct SkinPose
{
  std::vector<Mat4> bones;
  Mat4              bind, bindInverse;
};

// The four strongest influences per cage vertex, renormalised.
struct Top4
{
  std::vector<uint16_t> index;
  std::vector<float>    weight;
};
Top4 top4(const mesh::SkinInput& in);

// Each bone's skin matrix in the mesh's space: bindInverse * bone * bind.
std::vector<Mat4> skinMatrices(const SkinPose& pose);

// Mesh-local skinned positions: bindInverse * (sum w_i bone_i) * bind * p. `index` and `weight` are
// four per vertex.
std::vector<float> skinLocal(const std::vector<float>& rest, const Top4& skin, const SkinPose& pose);

// Cage deltas whose skinning approximates `simulated` (mesh-local, same space as skinLocal). A
// vertex whose blended 3x3 is singular keeps a zero delta.
std::vector<float> drapeDelta(const std::vector<float>& rest, const std::vector<float>& simulated, const Top4& skin, const SkinPose& pose);

}  // namespace fv::dforce
