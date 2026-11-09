// A glTF material's closure, through the renderer's own row conversion: for the unit tests and the
// library sweep. Not with closure_probe.hpp's GLM rule: this side sees GLM as the renderer does.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <figure/materials/lower.hpp>
#include <figure/scene/graph.hpp>
#include <figure/gltf/gltf_build.hpp>

#include "closure_probe.hpp"
#include "gltf_material_cache.hpp"

namespace fv::probe {

// The texture table a row is converted with: images that are their texture index, mips cooked or not
inline nvvkgltf::TextureSlotTable probeSlots(const ClosureOptions& options)
{
  nvvkgltf::TextureSlotTable slots;
  if(options.cookedMips)
    slots.cookedMips.assign(256, 1);
  return slots;
}

inline std::vector<ClosureAt> closureOfGltf(const tinygltf::Material& material, const ClosureOptions& options = {})
{
  std::vector<shaderio::GltfTextureInfo> infos(1);  // index 0 is the row's "no texture"
  const shaderio::GltfShadeMaterial      row = nvvkgltf::MaterialCache::convertOne(material, probeSlots(options), 0, infos);
  return probeClosure(row, infos, options);
}

// The roughness each lobe gets, through the same conversion
inline LobeRoughness roughnessOfGltf(const tinygltf::Material& material, const ClosureOptions& options = {})
{
  std::vector<shaderio::GltfTextureInfo> infos(1);
  const shaderio::GltfShadeMaterial      row = nvvkgltf::MaterialCache::convertOne(material, probeSlots(options), 0, infos);
  return probeRoughness(row, infos, options);
}

// A scene material as the build lowers it, every texture it asks for a neutral one of its own
inline std::vector<ClosureAt> closureOf(const Material& m, mat::SkinMode skin, const ClosureOptions& options = {})
{
  int                      next = 0;
  std::vector<std::string> extensions;
  mat::LowerOptions        lowering;
  lowering.skin = skin;
  const auto resolve = [&](const mat::TextureRequest&) {
    mat::ResolvedTexture r;
    r.index = next++;
    return r;
  };
  return closureOfGltf(fv::build::materialToGltf(m, lowering, resolve, extensions), options);
}

}  // namespace fv::probe
