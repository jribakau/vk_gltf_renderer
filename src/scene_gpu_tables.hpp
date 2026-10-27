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
// Scene-wide GPU tables derived from what draws, shared by the legacy scene (SceneVk) and the
// retained one (rs::Residency): the punctual lights as the shaders read them, and the emissive
// triangles next-event estimation samples through a Vose alias table. Vulkan-free.
//

#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>

#include "shaders/gltf_scene_io.h.slang"

namespace nvvkgltf {

// A light at `world` (its node's world matrix). No light: a pure EXT_lights_ies light, whose profile
// shapes it and whose node sets its colour and intensity (the caller folds those in).
shaderio::GltfLight packLight(const tinygltf::Light* light, const glm::mat4& world);
// An EXT_lights_ies light, which stands alone: the profile reshapes a point light of the node's
// colour and multiplier (shaders/gltf_light_ies.h.slang)
shaderio::GltfLight packIesLight(const glm::mat4& world, int profile, float multiplier, const glm::vec3& color);

// One drawn surface that may emit: its render node and prim, its radiance (emissive factor, strength
// and a textured emitter's mean, as the hit side reads it), and its triangles in world space through
// `world`
struct Emitter
{
  uint32_t                   renderNodeID = 0;
  uint32_t                   renderPrimID = 0;
  glm::vec3                  radiance{0.f};
  bool                       doubleSided = false;
  glm::mat4                  world{1.f};
  std::span<const glm::vec3> positions;
  std::span<const uint32_t>  indices;
};

struct EmissiveTable
{
  std::vector<shaderio::EmissiveTriangle> triangles;
  float                                   totalWeight = 0.f;  // GltfScene::emissiveTotalWeight
  float                                   meanLum     = 0.f;  // GltfScene::emissiveMeanLum
};

// Every non-degenerate triangle of the emitters that emit, weighted by area times defensive
// luminance (pulled toward the area-weighted mean, so a dim emitter is never starved), with a Vose
// alias table over those weights for O(1) selection.
EmissiveTable buildEmissiveTable(std::span<const Emitter> emitters);

// Rec. 709 luminance; must match the shader's luminance() (nvshaders/functions.h.slang) so the pdf's
// area term cancels
inline float emissiveLuminance(const glm::vec3& radiance)
{
  return 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
}

}  // namespace nvvkgltf
