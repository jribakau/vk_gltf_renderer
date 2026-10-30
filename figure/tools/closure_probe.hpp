// What a material row becomes in the path tracer's closure, measured on the CPU: the shader's
// evaluateMaterial and BSDF (shaders/gltf_material_eval.h.slang, nvshaders) compiled as C++. The
// numbers a change to the lowering or the closure shows up as, before anyone looks (the app's
// docs/material-model-plan.md, phase M1). Textures are neutral: white, and flat normal maps, so a
// map's factor is measured and not its pixels, which only the cooker reads.
//
// GLM is seen differently in the one source file (the shader code needs its swizzles), so this
// header names no GLM or shader type: a row and its texture infos pass as bytes.

#pragma once

#include <cstddef>
#include <vector>

namespace fv::probe {

// As nvshaders' LOBE_* defines
constexpr int kLobeCount = 7;
const char*   lobeName(int lobe);

struct ClosureLobe
{
  float weight = 0;              // the share of the light this lobe gets at this view
  float albedo[3] = {0, 0, 0};   // what it carries of the light it gets
};

struct ClosureAt
{
  float       cosine = 1;
  ClosureLobe lobes[kLobeCount];
  float       total[3] = {0, 0, 0};  // every lobe's weight times its albedo: the surface's albedo here
};

struct ClosureOptions
{
  std::vector<float> cosines{1.0f, 0.7f, 0.4f, 0.15f};
  int                samples = 8192;
  // the interactive preview's closure of a medium: evaluateMaterial's scatterAsThin
  bool thinScatter = false;
};

std::vector<ClosureAt> probeClosureBytes(const void* row, size_t rowSize, const void* infos, size_t infoSize, size_t infoCount, const ClosureOptions& options);

// `Row` is shaderio::GltfShadeMaterial and `Info` shaderio::GltfTextureInfo, with the table as
// MaterialCache::convertOne fills it: a row names a texture by its place in `infos`, 0 for none
template <class Row, class Info>
std::vector<ClosureAt> probeClosure(const Row& row, const std::vector<Info>& infos, const ClosureOptions& options = {})
{
  return probeClosureBytes(&row, sizeof(Row), infos.data(), sizeof(Info), infos.size(), options);
}

}  // namespace fv::probe
