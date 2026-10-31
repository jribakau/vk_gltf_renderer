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

#include <figure/materials/material.hpp>
#include <figure/materials/material_gltf.hpp>

#include "shaders/shaderio.h"

namespace nvvkgltf {

constexpr uint32_t kInstanceCullDisable = 0x1;  // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
constexpr uint32_t kInstanceForceOpaque = 0x4;  // VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR

struct MaterialBits
{
  uint32_t instanceFlags = 0;
  uint8_t  instanceMask  = 0;
  // the raster buckets' key (Scene::computeMaterialBucketKey): no texel is part see-through (alpha mode
  // OPAQUE, or MASK: the prepass tests the cutout and its depth is the main pass's test), double-sided,
  // transmissive
  bool alphaSolid   = false;
  bool doubleSided  = false;
  bool transmission = false;

  // the raster lists, as Scene::getShadedNodes sorts render nodes
  bool solid() const { return alphaSolid && !doubleSided && !transmission; }
  bool solidDoubleSided() const { return alphaSolid && doubleSided; }
  bool blend() const { return !alphaSolid || transmission; }
  bool operator==(const MaterialBits&) const = default;
};

// No ray can keep the surface and it emits nothing (materialBits gives it no instance mask)
inline bool invisible(const fv::mat::Material& mat)
{
  if(mat.alphaMode == fv::mat::AlphaMode::eOpaque || mat.specularGlossiness)
    return false;
  for(float e : mat.emissive)
    if(e > 0.0f)
      return false;
  // as the shader compares them
  const float alpha = mat.baseColor[3];
  return mat.alphaMode == fv::mat::AlphaMode::eMask ? alpha < mat.alphaCutoff : alpha <= 0.0F;
}

inline MaterialBits materialBits(const fv::mat::Material& mat)
{
  MaterialBits bits;
  const float  transmission        = mat.transmission ? mat.transmission->factor : 0.0f;
  const float  diffuseTransmission = mat.diffuseTransmission ? mat.diffuseTransmission->factor : 0.0f;
  const float  thickness           = mat.volume ? mat.volume->thickness : 0.0f;
  const bool   opaque              = mat.alphaMode == fv::mat::AlphaMode::eOpaque;

  // Opaque: no any-hit needed (faster)
  if(transmission == 0.0f && opaque && diffuseTransmission == 0.0F)
    bits.instanceFlags |= kInstanceForceOpaque;
  // No culling for double-sided materials, and for any material a path can transmit through: the ray
  // then reaches the far side of the object from behind.
  if(mat.doubleSided || thickness > 0.0F || transmission > 0.0F || diffuseTransmission > 0.0F)
    bits.instanceFlags |= kInstanceCullDisable;

  // A surface a scatter walk passes through (shaderio.h INSTANCE_MASK_WALK)
  bits.instanceMask = mat.walkThrough ? INSTANCE_MASK_OTHER : INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER;
  // A surface no ray can keep (surface_opacity.h.slang: the base colour factor's alpha only ever
  // multiplies down) and that emits nothing is left out of every ray: a helper mesh at opacity 0, such
  // as a hair's face dummy, otherwise stops each ray that crosses it for an alpha test that discards
  // it. An emitter stays, however transparent (a ghost light). Spec-gloss takes its alpha elsewhere.
  if(invisible(mat))
    bits.instanceMask = 0;

  bits.alphaSolid   = opaque || mat.alphaMode == fv::mat::AlphaMode::eMask;
  bits.doubleSided  = mat.doubleSided;
  bits.transmission = transmission > 0.f;
  return bits;
}

inline MaterialBits materialBits(const tinygltf::Material& mat)
{
  return materialBits(fv::mat::fromGltf(mat));
}

// The texture indices a material samples as colour (sRGB), as SceneVk::findSrgbImages decides it:
// base colour, emission, and the colour textures of specular, sheen, diffuse transmission, scatter
// and specular-glossiness. (A texture's own `gamma` extra, which findSrgbImages also reads, is a
// glTF texture's and not the material's.)
inline std::set<int> srgbTextures(const fv::mat::Material& mat)
{
  std::set<int> out;
  const auto    add = [&](const fv::mat::TextureSlot& slot) {
    if(slot.has())
      out.insert(slot.index);
  };
  add(mat.baseColorMap);
  add(mat.emissiveMap);
  if(mat.specular)
    add(mat.specular->colorMap);
  if(mat.clearcoat)
    add(mat.clearcoat->colorMap);
  if(mat.sheen)
    add(mat.sheen->colorMap);
  if(mat.diffuseTransmission)
    add(mat.diffuseTransmission->colorMap);
  if(mat.scatter)
    add(mat.scatter->multiscatterColorMap);
  if(mat.specularGlossiness)
  {
    add(mat.specularGlossiness->diffuseMap);
    add(mat.specularGlossiness->specularGlossinessMap);
  }
  return out;
}

inline std::set<int> srgbTextures(const tinygltf::Material& mat)
{
  return srgbTextures(fv::mat::fromGltf(mat));
}

}  // namespace nvvkgltf
