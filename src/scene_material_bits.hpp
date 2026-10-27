/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// What a material decides for everything that draws with it: its TLAS instance flags and mask, and
// which raster lists it goes in. One definition for the legacy scene (SceneRtx, Scene's shaded-node
// buckets) and the retained one (rs::Residency). Vulkan-free: the flag values are Vulkan's, checked
// against the enums where Vulkan is included (gltf_scene_rtx.cpp).
//

#include <cstdint>
#include <set>

#include <tinygltf/tiny_gltf.h>

#include "shaders/shaderio.h"
#include "tinygltf_utils.hpp"

namespace nvvkgltf {

constexpr uint32_t kInstanceCullDisable = 0x1;  // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
constexpr uint32_t kInstanceForceOpaque = 0x4;  // VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR

struct MaterialBits
{
  uint32_t instanceFlags = 0;
  uint8_t  instanceMask  = 0;
  // the raster buckets' key (Scene::computeMaterialBucketKey): alpha mode OPAQUE, double-sided, transmissive
  bool alphaOpaque  = false;
  bool doubleSided  = false;
  bool transmission = false;

  // the raster lists, as Scene::getShadedNodes sorts render nodes
  bool solid() const { return alphaOpaque && !doubleSided && !transmission; }
  bool solidDoubleSided() const { return alphaOpaque && doubleSided; }
  bool blend() const { return !alphaOpaque || transmission; }
  bool operator==(const MaterialBits&) const = default;
};

inline MaterialBits materialBits(const tinygltf::Material& mat)
{
  MaterialBits                             bits;
  const KHR_materials_transmission         transmission        = tinygltf::utils::getTransmission(mat);
  const KHR_materials_volume               volume              = tinygltf::utils::getVolume(mat);
  const KHR_materials_diffuse_transmission diffuseTransmission = tinygltf::utils::getDiffuseTransmission(mat);

  // Opaque: no any-hit needed (faster)
  if(transmission.factor == 0.0f && mat.alphaMode == "OPAQUE" && diffuseTransmission.diffuseTransmissionFactor == 0.0F)
    bits.instanceFlags |= kInstanceForceOpaque;
  // No culling for double-sided materials, and for any material a path can transmit through: the ray
  // then reaches the far side of the object from behind.
  if(mat.doubleSided || volume.thicknessFactor > 0.0F || transmission.factor > 0.0F
     || diffuseTransmission.diffuseTransmissionFactor > 0.0F)
    bits.instanceFlags |= kInstanceCullDisable;

  // A surface a scatter walk passes through (shaderio.h INSTANCE_MASK_WALK)
  const bool walkThrough = mat.extras.IsObject() && mat.extras.Has("fvWalkThrough")
                           && mat.extras.Get("fvWalkThrough").IsBool() && mat.extras.Get("fvWalkThrough").Get<bool>();
  bits.instanceMask      = walkThrough ? INSTANCE_MASK_OTHER : INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER;

  bits.alphaOpaque = mat.alphaMode == "OPAQUE";
  bits.doubleSided = mat.doubleSided;
  if(const auto* ext = tinygltf::utils::findExtension(mat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME))
  {
    float factor = 0.f;
    tinygltf::utils::getValue(*ext, "transmissionFactor", factor);
    bits.transmission = factor > 0.f;
  }
  return bits;
}

// The texture indices a material samples as colour (sRGB), as SceneVk::findSrgbImages decides it:
// base colour, emission, and the colour textures of specular, sheen, diffuse transmission, scatter
// and specular-glossiness. (A texture's own `gamma` extra, which findSrgbImages also reads, is a
// glTF texture's and not the material's.)
inline std::set<int> srgbTextures(const tinygltf::Material& mat)
{
  std::set<int> out;
  const auto    add = [&](int index) {
    if(index >= 0)
      out.insert(index);
  };
  const auto fromExtension = [&](const char* ext, const char* name) {
    const auto it = mat.extensions.find(ext);
    if(it != mat.extensions.end() && it->second.Has(name) && it->second.Get(name).Has("index"))
      add(it->second.Get(name).Get("index").GetNumberAsInt());
  };
  add(mat.pbrMetallicRoughness.baseColorTexture.index);
  add(mat.emissiveTexture.index);
  fromExtension("KHR_materials_specular", "specularColorTexture");
  fromExtension("KHR_materials_sheen", "sheenColorTexture");
  fromExtension("KHR_materials_diffuse_transmission", "diffuseTransmissionColorTexture");
  fromExtension(KHR_MATERIALS_SCATTER_EXTENSION_NAME, "multiscatterColorTexture");
  fromExtension(KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME, "multiscatterColorTexture");
  fromExtension("KHR_materials_pbrSpecularGlossiness", "diffuseTexture");
  fromExtension("KHR_materials_pbrSpecularGlossiness", "specularGlossinessTexture");
  return out;
}

}  // namespace nvvkgltf
