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

// Where the physical sky's clouds are seen from. The clouds are a place in the world, kilometres up
// and over a particular stretch of land, so what a camera sees of them depends on where it stands:
// they pass overhead as it crosses the land and come nearer as it climbs. Two images hold them for
// the camera, each baked from a place (an anchor):
//
//  - the sky image, all round, from the scene's origin or, once the camera has gone some way from
//    where it was last baked, from where the camera then was (SkyCloudFollow, cloudFollowStep);
//  - the view image, where the camera looks, at the screen's resolution, from where it stands
//    (SkyCloudView, cloudViewFor).
//
// Between bakes a reader shifts an image by the clouds' distance (cloudDirFromAnchor in
// sky_clouds.h.slang). Header-only arithmetic, so the tests read it without the renderer.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../shaders/sky_clouds_io.h.slang"  // CLOUD_VIEW_CAPACITY

// A place the clouds are seen from: where it is on the land from the scene's origin, km along the
// environment's x and z, and how high above the planet's surface, km.
struct SkyCloudAnchor
{
  glm::vec2 land{0.0F};
  float     altitude{0.0F};

  bool operator==(const SkyCloudAnchor& o) const { return land == o.land && altitude == o.altitude; }
};

// `eye` from `anchor`, km in the environment's frame (y up).
inline glm::vec3 cloudEyeFrom(const SkyCloudAnchor& anchor, const SkyCloudAnchor& eye)
{
  return {eye.land.x - anchor.land.x, eye.altitude - anchor.altitude, eye.land.y - anchor.land.y};
}

inline float cloudAnchorDistance(const SkyCloudAnchor& a, const SkyCloudAnchor& b)
{
  return glm::length(cloudEyeFrom(a, b));
}

// The lowest a camera stands above the planet's surface, km: one under the floor of its scene still
// looks at the clouds from above ground.
constexpr float kCloudViewMinAltitude = 0.001F;

// Where a camera stands: its view matrix, the environment's orientation (environment frame to world),
// a scene unit a metre, the scene's origin `originAltitudeKm` above the planet's surface (the sky
// bake's observer).
inline SkyCloudAnchor cloudEyeOf(const glm::mat4& view, const glm::quat& envRotation, float originAltitudeKm)
{
  const glm::vec3 eyeKm = glm::conjugate(glm::normalize(envRotation)) * glm::vec3(glm::inverse(view)[3]) * 0.001F;
  return {.land = glm::vec2(eyeKm.x, eyeKm.z), .altitude = std::max(originAltitudeKm + eyeKm.y, kCloudViewMinAltitude)};
}

// How far the camera goes from where an image was baked before the image is baked again from where
// the camera is, km. Within it the image is shifted by the clouds' distance, which is right to well
// under a degree: at 10 m under a cloud base 750 m up the shift is 0.76 degrees.
constexpr float kCloudFollowDistance = 0.010F;
// The frames a bake of the sky image for the camera is spread over, a band of rows each: all at once
// it is a hitch of some 25 ms, a few times a second in flight.
constexpr uint32_t kCloudFollowFrames = 32;
// ... unless the camera is this far from the image it has, km: put down somewhere else, not flown
// there. The shift is no picture of the sky from that far, so the image is baked whole, at once.
constexpr float kCloudFollowJump = 0.1F;
// A camera among the clouds (above their bases) has them tens of metres off, not hundreds: the image
// is baked again sooner and over fewer frames, each a longer band. Shifted as far as under the
// clouds, a cloud beside the camera smeared across the frame.
constexpr float    kCloudFollowDistanceAmong = 0.004F;
constexpr uint32_t kCloudFollowFramesAmong   = 12;

// The sky image that follows the camera. Image 0 is the sky bake's, from the scene's origin (`home`),
// which also lights the scene and is never baked over; 1 and 2 are the follow images, baked in turn
// from where the camera is, the one the readers are not taking. `front` is the one they take, baked
// from `anchor`.
struct SkyCloudFollow
{
  int            front{0};
  SkyCloudAnchor anchor{};
  SkyCloudAnchor home{};
  bool           baking{false};  // a bake of the other follow image is under way, a band a frame
  SkyCloudAnchor bakingFrom{};
  uint32_t       nextRow{0};

  // After a sky bake from `origin`: image 0, and nothing under way.
  void reset(const SkyCloudAnchor& origin) { *this = SkyCloudFollow{.anchor = origin, .home = origin}; }
};

// What a frame bakes of it: rows [firstRow, firstRow + rows) of image `target` as seen from `from`;
// `completes` when they are its last, and the readers take it from this frame on (`frontChanged`,
// which is also set when they go back to image 0).
struct SkyCloudFollowStep
{
  bool           bake{false};
  int            target{0};
  SkyCloudAnchor from{};
  uint32_t       firstRow{0};
  uint32_t       rows{0};
  bool           completes{false};
  bool           frontChanged{false};
};

// One frame of the follow, for a camera at `eye`; `imageRows` the sky image's height, `band` what its
// bands of rows are a multiple of (the bake's workgroup).
//
//  - Within kCloudFollowDistance of the origin the readers take image 0 again: the sky bake's, at its
//    full quality.
//  - A camera that has gone kCloudFollowDistance from the front image's anchor starts a bake from
//    where it is, and each frame after bakes the next band, until the image is whole and the readers
//    take it. The camera has moved on by then, by what it flies in kCloudFollowFrames frames, and the
//    next bake starts when that is too far in its turn. A camera that stops meanwhile gets the image
//    it was owed all the same: whether it is at rest is not asked, since a frame in which a flight's
//    camera did not move looks the same as one at rest.
//  - Past kCloudFollowJump from the front image's anchor the rest of the bake goes in this frame.
//
// `among`: the camera is among the clouds, and the follow keeps the shorter distance and count
// (kCloudFollowDistanceAmong, kCloudFollowFramesAmong).
inline SkyCloudFollowStep cloudFollowStep(SkyCloudFollow& f, const SkyCloudAnchor& eye, uint32_t imageRows, uint32_t band, bool among = false)
{
  const float        distance = among ? kCloudFollowDistanceAmong : kCloudFollowDistance;
  const uint32_t     frames   = among ? kCloudFollowFramesAmong : kCloudFollowFrames;
  SkyCloudFollowStep step;
  if(cloudAnchorDistance(f.home, eye) <= kCloudFollowDistance)
  {
    step.frontChanged = f.front != 0;
    f.front           = 0;
    f.anchor          = f.home;
    f.baking          = false;
    return step;
  }
  const float gone = cloudAnchorDistance(f.anchor, eye);
  if(!f.baking)
  {
    if(gone <= distance)
      return step;
    f.baking     = true;
    f.bakingFrom = eye;
    f.nextRow    = 0;
  }
  else if(gone > kCloudFollowJump && cloudAnchorDistance(f.bakingFrom, eye) > distance)
  {
    // put down elsewhere mid-bake: what was under way is of the wrong place
    f.bakingFrom = eye;
    f.nextRow    = 0;
  }
  const uint32_t each = std::max((imageRows + frames - 1) / frames, 1U);
  step.bake           = true;
  step.target         = f.front == 1 ? 2 : 1;
  step.from           = f.bakingFrom;
  step.firstRow       = f.nextRow;
  step.rows = gone > kCloudFollowJump ? imageRows - f.nextRow : std::min(((each + band - 1) / band) * band, imageRows - f.nextRow);
  f.nextRow += step.rows;
  if(f.nextRow >= imageRows)
  {
    step.completes    = true;
    step.frontChanged = true;
    f.front           = step.target;
    f.anchor          = f.bakingFrom;
    f.baking          = false;
  }
  return step;
}

// A perspective map about `forward`, in the environment's frame, out to the tangents of its
// half-width and half-height in right's and up's .w (cloudViewDir in sky_clouds.h.slang), filling
// `size` texels of the view image, as seen from `eye`, where the camera stands.
struct SkyCloudView
{
  glm::vec4      right{0.0F};
  glm::vec4      up{0.0F};
  glm::vec3      forward{0.0F};
  glm::uvec2     size{0, 0};
  SkyCloudAnchor eye{};

  bool operator==(const SkyCloudView& o) const { return right == o.right && up == o.up && forward == o.forward && size == o.size && eye == o.eye; }
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
// pixel up to the image's capacity, seen from where the camera stands (cloudEyeOf). None for a camera
// the sky image serves as well: orthographic, or wider than about 110 degrees.
inline std::optional<SkyCloudView> cloudViewFor(const glm::mat4& view,
                                                const glm::mat4& projection,
                                                const glm::quat& envRotation,
                                                glm::uvec2       frameSize,
                                                float            originAltitudeKm = 0.0F)
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
  return SkyCloudView{.right   = glm::vec4(right, tanRight),
                      .up      = glm::vec4(up, tanUp),
                      .forward = forward,
                      .size    = size,
                      .eye     = cloudEyeOf(view, envRotation, originAltitudeKm)};
}
