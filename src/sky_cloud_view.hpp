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

// Where the physical sky's view image looks (SkyClouds::recordView): the clouds where the camera
// looks, at the screen's resolution, beside the sky image that holds them all round. Header-only
// arithmetic, so the tests read it without the renderer.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../shaders/sky_clouds_io.h.slang"  // CLOUD_VIEW_CAPACITY

// A perspective map about `forward`, in the environment's frame, out to the tangents of its
// half-width and half-height in right's and up's .w (cloudViewDir in sky_clouds.h.slang), filling
// `size` texels of the view image.
struct SkyCloudView
{
  glm::vec4  right{0.0F};
  glm::vec4  up{0.0F};
  glm::vec3  forward{0.0F};
  glm::uvec2 size{0, 0};

  bool operator==(const SkyCloudView& o) const { return right == o.right && up == o.up && forward == o.forward && size == o.size; }
};

// How much wider than the camera's view the view image reaches, as a share of its half-extent's
// tangent: room for the frame's jitter, a nudge of the camera, and the fade to the sky image
// (cloudViewShare) outside the frame.
constexpr float kCloudViewMargin = 0.1F;
// The widest half-extent the view image takes, as a tangent (about 56 degrees): past it, a texel a
// pixel wide at its centre is no finer than the sky image's toward its edges.
constexpr float kCloudViewMaxTangent = 1.5F;

// The view image for a camera: its view and projection matrices as the path tracer's camera rays take
// them (getRay in camera_ray.h.slang), the environment's orientation (environment frame to world), and
// the frame's size in pixels. A little wider than the camera's view (kCloudViewMargin), a texel a
// pixel up to the image's capacity. None for a camera the sky image serves as well: orthographic, or
// wider than about 110 degrees.
inline std::optional<SkyCloudView> cloudViewFor(const glm::mat4& view, const glm::mat4& projection, const glm::quat& envRotation, glm::uvec2 frameSize)
{
  if(frameSize.x == 0 || frameSize.y == 0 || projection[2][3] == 0.0F)
    return std::nullopt;  // no frame, or orthographic: its rays are parallel, one direction a frame

  // The frame's corners as getRay aims them, in the camera's space, where it looks down -z: the
  // widest tangent each way covers a frustum off its axis too.
  const glm::mat4 projectionInverse = glm::inverse(projection);
  float           tanRight          = 0.0F;
  float           tanUp             = 0.0F;
  for(const glm::vec2 corner : {glm::vec2(-1.0F, -1.0F), glm::vec2(1.0F, -1.0F), glm::vec2(-1.0F, 1.0F), glm::vec2(1.0F, 1.0F)})
  {
    glm::vec4 p = projectionInverse * glm::vec4(corner, -1.0F, 1.0F);
    p /= p.w;
    if(!(p.z < 0.0F))
      return std::nullopt;
    tanRight = std::max(tanRight, std::abs(p.x / p.z));
    tanUp    = std::max(tanUp, std::abs(p.y / p.z));
  }
  tanRight *= 1.0F + kCloudViewMargin;
  tanUp *= 1.0F + kCloudViewMargin;
  if(!(tanRight > 0.0F && tanUp > 0.0F) || tanRight > kCloudViewMaxTangent || tanUp > kCloudViewMaxTangent)
    return std::nullopt;

  // The camera's axes in the world, then in the environment's frame, where the clouds are baked.
  const glm::mat4 viewInverse = glm::inverse(view);
  const glm::quat toEnv       = glm::conjugate(glm::normalize(envRotation));
  const glm::vec3 right       = glm::normalize(toEnv * glm::vec3(viewInverse[0]));
  const glm::vec3 up          = glm::normalize(toEnv * glm::vec3(viewInverse[1]));
  const glm::vec3 forward     = glm::normalize(toEnv * -glm::vec3(viewInverse[2]));

  // A texel a pixel, the margin's too, as far as the image's capacity goes.
  const float      wide     = float(frameSize.x) * (1.0F + kCloudViewMargin);
  const float      high     = float(frameSize.y) * (1.0F + kCloudViewMargin);
  const float      capacity = float(CLOUD_VIEW_CAPACITY);
  const float      scale    = std::min({1.0F, capacity / wide, capacity / high});
  const glm::uvec2 size{std::clamp(uint32_t(std::ceil(wide * scale)), 1U, uint32_t(CLOUD_VIEW_CAPACITY)),
                        std::clamp(uint32_t(std::ceil(high * scale)), 1U, uint32_t(CLOUD_VIEW_CAPACITY))};
  return SkyCloudView{.right = glm::vec4(right, tanRight), .up = glm::vec4(up, tanUp), .forward = forward, .size = size};
}
