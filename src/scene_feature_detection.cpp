/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Detects which glTF material extensions are present in the current scene and
// folds them into a compact SceneFeatureSet used by path-tracer shader variant
// selection ("optimize shader for current scene" mode).
//

#include "scene_feature_detection.hpp"

#include <array>
#include <tinygltf/tiny_gltf.h>

#include <figure/materials/material_gltf.hpp>

#include "tinygltf_utils.hpp"

namespace nvvkgltf {

namespace {

struct FeatureName
{
  SceneFeatureSet::Feature feature;
  const char*              name;
};

constexpr std::array<FeatureName, SceneFeatureSet::kFeatureCount> kFeatureNames{{
    {SceneFeatureSet::eTransmission, "transmission"},
    {SceneFeatureSet::eVolume, "volume"},
    {SceneFeatureSet::eScatter, "scatter"},
    {SceneFeatureSet::eClearcoat, "clearcoat"},
    {SceneFeatureSet::eIridescence, "iridescence"},
    {SceneFeatureSet::eAnisotropy, "anisotropy"},
    {SceneFeatureSet::eSheen, "sheen"},
    {SceneFeatureSet::eDispersion, "dispersion"},
    {SceneFeatureSet::eDiffuseTransmission, "diffuseTransmission"},
    {SceneFeatureSet::eRetroreflection, "retroreflection"},
    {SceneFeatureSet::eUnlit, "unlit"},
    {SceneFeatureSet::eSpecular, "specular"},
    {SceneFeatureSet::eIor, "ior"},
    {SceneFeatureSet::eSpecularGlossiness, "specularGlossiness"},
    {SceneFeatureSet::eDetail, "detail"},
    {SceneFeatureSet::eHair, "hair"},
    {SceneFeatureSet::eTextureTransform, "textureTransform"},
    {SceneFeatureSet::eDlssGuide, "dlssGuide"},
}};

bool valueTreeHasTextureTransform(const tinygltf::Value& v)
{
  if(v.IsObject())
  {
    if(v.Has("index") && v.Get("index").IsNumber() && v.Get("index").GetNumberAsInt() >= 0 && v.Has("extensions")
       && v.Get("extensions").IsObject() && v.Get("extensions").Has(KHR_TEXTURE_TRANSFORM_EXTENSION_NAME))
      return true;

    for(const auto& entry : v.Get<tinygltf::Value::Object>())
    {
      if(valueTreeHasTextureTransform(entry.second))
        return true;
    }
  }
  else if(v.IsArray())
  {
    for(const auto& child : v.Get<tinygltf::Value::Array>())
    {
      if(valueTreeHasTextureTransform(child))
        return true;
    }
  }
  return false;
}

// Returns true if any texture slot inside a material uses a transform.
inline bool materialHasTextureTransform(const fv::mat::Material& m)
{
  bool found = false;
  fv::mat::forEachSlot(m, [&](const fv::mat::TextureSlot& slot) { found = found || (slot.has() && slot.transform); });
  if(found)
    return true;

  // an extension nothing here reads can still be sampled through one
  for(const auto& ext : m.foreignExtensions)
  {
    if(valueTreeHasTextureTransform(ext.second))
      return true;
  }

  return false;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// True when this feature set covers every feature in the other set
bool SceneFeatureSet::isSupersetOf(const SceneFeatureSet& o) const
{
  return (m_bits & o.m_bits) == o.m_bits;
}

//--------------------------------------------------------------------------------------------------
// Union: every feature either set needs
SceneFeatureSet& SceneFeatureSet::operator|=(const SceneFeatureSet& o)
{
  m_bits |= o.m_bits;
  return *this;
}

//--------------------------------------------------------------------------------------------------
// Exact equality on all scene-feature toggles
bool SceneFeatureSet::operator==(const SceneFeatureSet& o) const
{
  return m_bits == o.m_bits;
}

//--------------------------------------------------------------------------------------------------
// Comma-separated list of enabled features for logs/UI
std::string SceneFeatureSet::toString() const
{
  std::string out;
  for(const FeatureName& featureName : kFeatureNames)
  {
    if(!has(featureName.feature))
      continue;
    if(!out.empty())
      out += ",";
    out += featureName.name;
  }
  return out;
}

//--------------------------------------------------------------------------------------------------
// Count currently-disabled extension gates
int SceneFeatureSet::unusedExtensionCount() const
{
  // Count of MAT_EXT_* style flags that are currently false. The dlssGuide flag is
  // intentionally not counted here because it is not a MAT_EXT_* (it's its own gate).
  std::bitset<kFeatureCount> extensionBits = m_bits;
  extensionBits.reset(static_cast<std::size_t>(eDlssGuide));
  return static_cast<int>(kExtensionFeatureCount - extensionBits.count());
}

//--------------------------------------------------------------------------------------------------
// One material's feature toggles, dependencies promoted
SceneFeatureSet detectMaterialFeatures(const fv::mat::Material& m)
{
  SceneFeatureSet set{};

  if(m.transmission)
    set.enable(SceneFeatureSet::eTransmission);
  if(m.volume)
    set.enable(SceneFeatureSet::eVolume);
  if(m.scatter)
    set.enable(SceneFeatureSet::eScatter);
  if(m.clearcoat)
    set.enable(SceneFeatureSet::eClearcoat);
  if(m.iridescence)
    set.enable(SceneFeatureSet::eIridescence);
  if(m.anisotropy)
    set.enable(SceneFeatureSet::eAnisotropy);
  if(m.sheen)
    set.enable(SceneFeatureSet::eSheen);
  if(m.dispersion)
    set.enable(SceneFeatureSet::eDispersion);
  if(m.diffuseTransmission)
    set.enable(SceneFeatureSet::eDiffuseTransmission);
  if(m.retroreflection)
    set.enable(SceneFeatureSet::eRetroreflection);
  if(m.unlit)
    set.enable(SceneFeatureSet::eUnlit);
  if(m.specular)
    set.enable(SceneFeatureSet::eSpecular);
  if(m.ior)
    set.enable(SceneFeatureSet::eIor);
  if(m.specularGlossiness)
    set.enable(SceneFeatureSet::eSpecularGlossiness);
  if(m.detail)
    set.enable(SceneFeatureSet::eDetail);
  if(m.hair)
    set.enable(SceneFeatureSet::eHair);

  if(materialHasTextureTransform(m))
    set.enable(SceneFeatureSet::eTextureTransform);

  // Feature dependency promotion. The path tracer's bounce-loop code paths form a
  // strict containment chain, and consumers of this set should not have to re-derive
  // it. We do it once here so operator== / toString() / unusedExtensionCount()
  // all see the fully-promoted truth. Each promoted bit is an OR of bits, so promoting
  // per material and then taking the union equals promoting the union.
  //
  //   scatter  =>  volume   (volumetric scatter samples Henyey-Greenstein inside a medium -
  //                          the medium has to exist, makeVolumeMedium is gated on volume.
  //                          Thin-walled scatter does not strictly need it, but promoting
  //                          conservatively keeps the containment chain simple and is harmless:
  //                          the volume code is skipped when thickness == 0.)
  //   volume   =>  transmission   (the bounce loop only enters a volume via the
  //                                transmission isInside toggle; without transmission
  //                                pt.isInside never flips so the volume code is dead)
  if(set.has(SceneFeatureSet::eScatter))
    set.enable(SceneFeatureSet::eVolume);
  if(set.has(SceneFeatureSet::eVolume))
    set.enable(SceneFeatureSet::eTransmission);

  return set;
}

SceneFeatureSet detectMaterialFeatures(const tinygltf::Material& m)
{
  return detectMaterialFeatures(fv::mat::fromGltf(m));
}

//--------------------------------------------------------------------------------------------------
// Scan materials/model and collect the scene's active feature toggles
SceneFeatureSet detectSceneFeatures(const std::vector<tinygltf::Material>& materials, const tinygltf::Model* /*model*/)
{
  SceneFeatureSet set{};
  for(const auto& m : materials)
    set |= detectMaterialFeatures(m);
  return set;
}

//--------------------------------------------------------------------------------------------------
// Convenience overload for full-model feature detection
SceneFeatureSet detectSceneFeatures(const tinygltf::Model& model)
{
  return detectSceneFeatures(model.materials, &model);
}

}  // namespace nvvkgltf
