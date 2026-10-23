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
// The Figure Viewer's Light tab as the host keeps it: one state, patched by `lighting.set` and
// published as the `lighting` topic in the same shape, and the conversions from the app's units to
// the renderer's. Pure, so it is tested without a device (tests/test_fv_lighting.cpp).
//
//   env:     source studio|hdr|sky|plain|none, path, intensity, rotation (degrees about +Y),
//            background environment|blurred|color, blur (three's blurriness), color, plainColor, skyPreset
//   sun:     enabled, intensity, azimuth, elevation (degrees), color, softness, shadows
//   shadow:  size (texels), contactHardening, bias, normalBias (shadow-map texels)
//   ground:  mode floor|catcher|off, height ("auto" or world units), color, roughness, darkness, fade
//   tonemap: method aces|agx|neutral|filmic, exposure, contrast, saturation, vignette,
//            temperature (Kelvin), tint (Duv)
//
// Colours are "#rrggbb" in sRGB, as the panel shows them.
//

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "fv_events.hpp"

namespace fv::lighting {

struct Env
{
  std::string source     = "studio";
  std::string path;  // the HDR or EXR, when source is hdr
  double      intensity  = 1.0;
  double      rotation   = 0.0;
  std::string background = "blurred";
  double      blur       = 0.35;
  std::string color      = "#2a2d33";
  std::string plainColor = "#808080";
  int         skyPreset  = 0;  // the physical sky's atmosphere: Earth, Mars, Alien
};

struct Sun
{
  bool        enabled   = true;
  double      intensity = 2.2;
  double      azimuth   = 35.0;
  double      elevation = 50.0;
  std::string color     = "#fff4e6";
  double      softness  = 3.0;
  bool        shadows   = true;
};

struct Shadow
{
  int    size             = 4096;
  bool   contactHardening = true;
  double bias             = 1.0;
  double normalBias       = 1.5;
};

struct Ground
{
  std::string mode = "floor";
  bool        autoHeight = true;
  double      height     = 0.0;  // when not autoHeight
  std::string color      = "#5a5d62";
  double      roughness  = 0.9;
  double      darkness   = 0.6;
  double      fade       = 3.0;
};

struct Tonemap
{
  std::string method      = "aces";
  double      exposure    = 1.0;
  double      contrast    = 1.0;
  double      saturation  = 1.0;
  double      vignette    = 0.0;
  double      temperature = 6500.0;
  double      tint        = 0.0;
};

struct State
{
  Env     env;
  Sun     sun;
  Shadow  shadow;
  Ground  ground;
  Tonemap tonemap;
};

Json toJson(const State& s);

// `s` with an RFC 7386 merge patch applied. A null resets that key (or section) to its default.
// Unknown keys, wrong types and bad enum values throw std::invalid_argument and leave nothing
// applied; numbers are clamped to their ranges. The topic's read-only keys (env.name, env.size,
// env.loading) are accepted and ignored, so a saved topic object replays as a patch.
State applyPatch(const State& s, const Json& patch);

// The merge patch that turns `from` into `to` (objects recurse, anything else is replaced).
Json mergeDiff(const Json& from, const Json& to);

// ---- conversions (ports of src/lib/viewer/Lighting.ts and src/lib/rtx/export.ts)

// "#rrggbb" (sRGB) to linear RGB, as three's Color.set does. Throws std::invalid_argument.
glm::vec3 linearColor(const std::string& hex);

// Unit vector toward the sun: (cos el sin az, sin el, cos el cos az).
glm::vec3 sunDirection(double azimuthDeg, double elevationDeg);

// The sun node's rotation: glTF aims a directional light down its -Z, so +Z goes to the sun.
glm::quat sunRotation(double azimuthDeg, double elevationDeg);

// The light's `extras.radius`: its disc radius in km at the sun's distance, 0.15 degrees of angular
// diameter per unit of softness (sunRadiusKm).
double sunRadiusKm(double softness);

// The renderer's hdrBlur, a fraction of the mip chain, for three's backgroundBlurriness on an
// image of this size (hdrBlur).
double hdrBlur(double blurriness, uint32_t width, uint32_t height);

// The renderer's exposure for the panel's: three's ACES divides by 0.6 first (tmExposure).
double tmExposure(const std::string& method, double exposure);

// The renderer's tmMethod for a tone-map name.
int tmMethod(const std::string& method);

// The environment's rotation about +Y as a glTF-order quaternion (x, y, z, w).
glm::vec4 envRotation(double degrees);

// A DAZ dome rotation as the panel's: three's lat-long seam is half a turn from Iray's.
double presetRotation(double dazDegrees);

// Degrees wrapped to [-180, 180).
double wrapDegrees(double degrees);

// The built-in studio (createStudioEnvironment): RGBA32F rows from the top, the layout the renderer
// takes an environment in.
std::vector<float> studioImage(uint32_t width = 512, uint32_t height = 256);

// The registered renderer parameters `s` sets, by name, for nvmcp's setParameters. `envSize` is
// the environment image's size, which hdrBlur depends on. Ground height is the host's (auto needs
// the scene), so it isn't here.
Json rendererParams(const State& s, glm::uvec2 envSize);

}  // namespace fv::lighting
