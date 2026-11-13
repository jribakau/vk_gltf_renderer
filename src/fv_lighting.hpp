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
//            background environment|blurred|color, blur (three's blurriness), color, plainColor, skyPreset,
//            skyVisibility (km one sees through the physical sky's air, which veils the distance and
//            whitens the horizon alike: sky_visibility.hpp; 0 the atmosphere's own), protocol 65
//   sun:     enabled, intensity, azimuth, elevation (degrees), color, softness
//   moon:    enabled, azimuth, elevation (degrees, as the sun's), phase (0 new, 0.25 first quarter,
//            0.5 full, 0.75 last quarter: the app places it by that, sunPath.ts moonAt) and light
//            (the moon, not the sun, is the physical sky's light: its light beats the twilight's,
//            the app's call). The physical sky's only; protocol 71
//   clouds:  coverage (0 clear to 1 overcast: the share of the land they shade), thickness (km from
//            the base to the highest tops), seed (which arrangement), type cumulus|stratocumulus|
//            stratus -- the low tier's -- and mid (altocumulus) and high (cirrus: the share of the sky
//            it veils), the tiers over it, each 0 to 1; drift (km/h the wind carries them all over the
//            land, 0 to 300, 0 they hold still) and driftFrom (degrees, where it blows from, an
//            azimuth as the sun's), protocol 68. The physical sky's only
//   haze:    enabled, visibility (metres: where the haze leaves 2% of a dark object's contrast),
//            glow (how much it scatters forward, the halo round a light seen through it:
//            Henyey-Greenstein g, 0 to 0.9), height (metres it thins over above the floor, 0
//            even through the height), patches (0 even to 1). A thin haze in the air, lit by every
//            light of the scene, between the eye and what it sees (shaders/haze.h.slang)
//   ground:  mode floor|water|catcher|off, height ("auto" or world units), color, roughness,
//            darkness, fade, water (0 dry to 1 a sheet of it, on a floor), depth (metres, of the
//            water mode's body of water), waves (on it, 0 still to 1), ripples (on either, 0 still to 1)
//   tonemap: method aces|agx|neutral|filmic|shaped, exposure, contrast, saturation, vignette,
//            temperature (Kelvin), tint (Duv), followClouds (the exposure opens up for what the
//            physical sky's clouds take from the light, as the meter measures it at each bake: the
//            app's camera metering), skyShare (of a clear sky's light on the figure, the share the
//            sky gives rather than the sun, at this sun: the app's measured tables, sunPath.ts);
//            read-only, cloudLight (the light the clouds leave the figure, against a clear sky's);
//            look (null, or a grade over those sliders: Look's fields, the vectors [r, g, b]; from
//            protocol 57 with a cheap camera's bend, fringe, clip and noise) and
//            lookStrength (0 to 1, how much of it); bloom (0 to 0.25, the share of the image's light
//            moved into a glow round bright light, 0 none) and star (0 to 1, the share of that glow
//            drawn as spikes across the aperture's blades; none for a round aperture), protocol 40;
//            localShadows and localHighlights (0 to 1 each, 0 none: how far dark regions are lifted and
//            bright ones held toward mid grey, region by region; src/local_exposure.hpp), protocol 41;
//            toe and shoulder (0 to 1 each, the Shaped curve's handles: how deep its shadows go, and
//            how early and gently its highlights roll off; nvshaders tonemapShaped), protocol 43;
//            highlightsToWhite (0 to 1, 0.5 Khronos's own: how fast Neutral's and Shaped's compressed
//            colours go white; nvshaders highlightDesaturation), protocol 63
//
// The backdrop (the land round the scene) was a section here until protocol 70: it is a state of its
// own now (fv_backdrop_state.hpp, `backdrop.set`).
//
// Colours are "#rrggbb" in sRGB, as the panel shows them. A patch from before protocol 17 (a kept
// scene's) may also carry sun.shadows and a shadow section, the raster view's shadow map: they are
// taken and dropped.
//

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "fv_events.hpp"
#include "fv_patch.hpp"

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
  double      skyVisibility = 0.0;  // km through its air; 0 the atmosphere's own
};

struct Sun
{
  bool        enabled   = true;
  double      intensity = 2.2;
  double      azimuth   = 35.0;
  double      elevation = 50.0;
  std::string color     = "#fff4e6";
  double      softness  = 3.0;
};

// The physical sky's moon (src/moon.hpp). Off for an app from before protocol 71, which sends none.
struct Moon
{
  bool   enabled   = false;
  double azimuth   = 0.0;
  double elevation = -90.0;
  double phase     = 0.5;
  bool   light     = false;
};

struct Clouds
{
  double      coverage  = 0.0;
  double      thickness = 1.45;
  int         seed      = 0;
  std::string type      = "cumulus";
  double      mid       = 0.0;
  double      high      = 0.0;
  double      drift     = 0.0;   // km/h; 0 they hold still
  double      driftFrom = 90.0;  // degrees, an azimuth as the sun's
};

struct Haze
{
  bool   enabled    = false;
  double visibility = 100.0;  // beams in a room, the far wall of a hall veiled; at the floor
  double glow       = 0.7;
  double height     = 0.0;  // metres it thins over above the floor; 0: even through the height
  double patches    = 0.0;
};

struct Ground
{
  std::string mode = "floor";
  bool        autoHeight = true;
  double      height     = 0.0;  // when not autoHeight
  // 20% grey, linear: average ground, the reflectance daylighting assumes (CIE's 0.2); grass, dry
  // soil and worn concrete sit near it. The physical sky's own ground is the same (sky_bruneton.cpp)
  std::string color      = "#7c7c7c";
  double      roughness  = 0.9;
  double      darkness   = 0.6;
  double      fade       = 3.0;
  double      water      = 0.0;
  double      ripples    = 0.25;
  double      depth      = 0.5;  // knee-deep on a figure
  double      waves      = 0.4;
};

// A look (the app's View › Image › Look): a grade over the panel's own Image sliders, which it moves
// rather than replaces. Every value is the identity at its default; `id` and `name` are the app's.
struct Look
{
  std::string id;
  std::string name;
  double      warmth       = 0.0;  // mireds the white balance moves the picture warmer (below 0 cooler)
  double      tint         = 0.0;  // Duv added to the panel's tint
  double      contrast     = 1.0;  // multiplying the panel's
  double      saturation   = 1.0;  // multiplying the panel's
  double      vibrance     = 0.0;
  double      vignette     = 0.0;  // added to the panel's
  double      shadows      = 0.0;  // the tone biases (smhAdjust)
  double      midtones     = 0.0;
  double      highlights   = 0.0;
  glm::dvec3  lift{0.0};
  glm::dvec3  gamma{1.0};
  glm::dvec3  gain{1.0};
  glm::dvec3  shadowTint{1.0};  // split toning's multipliers (splitToning)
  glm::dvec3  highlightTint{1.0};
  double      splitBalance = 0.0;
  double      mono         = 0.0;  // 0 colour to 1 black and white, through monoMix
  glm::dvec3  monoMix{0.2126, 0.7152, 0.0722};
  double      posterize    = 0.0;  // levels a channel; under 2 none
  double      posterizeMix = 1.0;
  double      grain        = 0.0;
  // A cheap camera's (protocol 57), 0 to 1 each: its lens (src/lens.hpp) and its sensor (tonemap_io's
  // sensorClip, sensorNoise)
  double      bend         = 0.0;  // barrel distortion
  double      fringe       = 0.0;  // colour fringing toward the corners
  double      clip         = 0.0;  // highlights that clip a channel at a time, in place of the curve's
  double      noise        = 0.0;  // noise that is strongest in the shadows, and coloured
};

struct Tonemap
{
  std::string method      = "neutral";
  double      exposure    = 1.0;
  double      contrast    = 1.0;
  double      saturation  = 1.0;
  double      vignette    = 0.0;
  double      temperature = 6500.0;
  double      tint        = 0.0;
  bool        followClouds = false;  // off for an app from before protocol 35, which doesn't send it
  double      skyShare     = 1.0;
  std::optional<Look> look;          // none: the sliders alone (protocol 39)
  double              lookStrength = 1.0;
  double              bloom        = 0.0;  // of the image's light, into the glow (protocol 40; src/bloom.hpp)
  double              star         = 0.0;  // of the glow, into the star across the aperture's blades
  double              localShadows    = 0.0;  // dark regions lifted toward mid grey (protocol 41; src/local_exposure.hpp)
  double              localHighlights = 0.0;  // bright regions held toward it
  double              toe             = 0.5;  // the Shaped curve's handles (protocol 43; tonemapShaped)
  double              shoulder        = 0.5;
  double              highlightsToWhite = 0.5;  // Neutral's and Shaped's (protocol 63; highlightDesaturation)
};

// What the tone mapper grades with: the panel's sliders with the look over them at its strength, each
// of the look's values that far from its identity. The white balance moves in mireds, so a look's
// warmth is the same shift at every setting of the slider.
struct Grade
{
  double     temperature = 6500.0;
  double     tint        = 0.0;
  double     contrast    = 1.0;
  double     saturation  = 1.0;
  double     vignette    = 0.0;
  double     vibrance    = 0.0;
  double     shadows     = 0.0;
  double     midtones    = 0.0;
  double     highlights  = 0.0;
  glm::dvec3 lift{0.0};
  glm::dvec3 gamma{1.0};
  glm::dvec3 gain{1.0};
  glm::dvec3 shadowTint{1.0};
  glm::dvec3 highlightTint{1.0};
  double     splitBalance = 0.0;
  double     mono         = 0.0;
  glm::dvec3 monoMix{0.2126, 0.7152, 0.0722};
  double     posterize    = 0.0;
  double     posterizeMix = 0.0;
  double     grain        = 0.0;
  double     bend         = 0.0;
  double     fringe       = 0.0;
  double     clip         = 0.0;
  double     noise        = 0.0;
};
Grade grade(const Tonemap& t);

// `t` with a look shown in place of its own, for lighting.preview: `preview.look`, a look as
// tonemap.look takes it or null for none, at `preview.strength` (`t`'s by default). Throws
// std::invalid_argument as applyPatch does.
Tonemap previewed(const Tonemap& t, const Json& preview);

// The camera's flash (the view's lens.flash): a small spot light a little above the lens, looking
// where the camera looks, wide enough for a wide lens's frame. Above the lens, as a camera's is, so
// what it lights throws a thin hard shadow below it.
constexpr double kFlashAbove  = 0.06;  // metres over the lens, along the camera's up
constexpr double kFlashRadius = 0.01;  // metres: a hard light
constexpr double kFlashInner  = 0.75;  // radians from its axis: full strength inside
constexpr double kFlashOuter  = 1.05;  // and none outside
struct FlashPose
{
  glm::vec3 position{0.0F};
  glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};  // a spot light shines along its -Z, as a camera looks
};
FlashPose flashPose(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up);

// What the physical sky's clouds leave of its light, as the renderer's last full bake measured it
// (EnvBaker::cloudLight): `sky`, the sky's own light on a figure, with the clouds against without;
// `sun`, the share of the sun that gets past them. 1 and 1 under a clear sky.
struct CloudReading
{
  double sky = 1.0;
  double sun = 1.0;
  bool   operator==(const CloudReading&) const = default;
};

struct State
{
  Env     env;
  Sun     sun;
  Moon    moon;
  Clouds  clouds;
  Haze    haze;
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
using fv::patch::mergeDiff;

// ---- conversions (ports of src/lib/viewer/Lighting.ts and src/lib/rtx/export.ts)

// "#rrggbb" (sRGB) to linear RGB, as three's Color.set does. Throws std::invalid_argument.
glm::vec3 linearColor(const std::string& hex);

// Unit vector toward the sun: (cos el sin az, sin el, cos el cos az).
glm::vec3 sunDirection(double azimuthDeg, double elevationDeg);

// Whether the moon is the physical sky's light: it is there, the app says its light beats the
// twilight's, and the sky is the physical one.
bool moonIsLight(const State& s);

// Where the sky's light stands, azimuth and elevation in degrees: the moon's when it is the light
// (moonIsLight), else the sun's. The sun light's entity is aimed there (FvHost::Impl::applySun).
glm::dvec2 skyLightAngles(const State& s);

// The path tracer's pre-exposure for an exposure (the tone mapper's, tmExposure): 1 up to 256, then the
// exposure itself rounded to a power of four, so a dark scene's light reaches DLSS-RR, which takes no
// exposure, in a half float's range. A change restarts a still, so it moves only in steps.
double preExposure(double exposure);

// The physical sky's light against its sun's: the moon's share of it (moon::lightShare) when the
// moon is the light, else 1. Everything the sky gives is scaled by it (hdrIntensity).
double skyLightScale(const State& s);

// The sun node's rotation: glTF aims a directional light down its -Z, so +Z goes to the sun.
glm::quat sunRotation(double azimuthDeg, double elevationDeg);

// sunDirection's inverse: the panel's azimuth (-180..180) and elevation, in degrees, of a unit
// vector toward the sun.
glm::dvec2 sunAngles(const glm::vec3& toSun);

// The sun in a lat-long environment image (RGBA32F, rows from the top, the renderer's mapping:
// latlongToDir): the unit direction toward it in the environment's own frame, or none when no spot
// stands out as one (an overcast sky, a studio). Read before the renderer takes the pixels.
std::optional<glm::vec3> environmentSun(const float* rgba, uint32_t width, uint32_t height);

// That direction in the world for an environment turned by `rotationDeg` (envRotation), as the
// panel's angles.
glm::dvec2 environmentSunAngles(const glm::vec3& envSun, double rotationDeg);

// The light's `extras.radius`: its disc radius in km at the sun's distance, 0.15 degrees of angular
// diameter per unit of softness (sunRadiusKm).
double sunRadiusKm(double softness);

// The renderer's hdrBlur, a fraction of the mip chain, for three's backgroundBlurriness on an
// image of this size (hdrBlur).
double hdrBlur(double blurriness, uint32_t width, uint32_t height);

// The renderer's exposure for the panel's: three's ACES divides by 0.6 first (tmExposure).
double tmExposure(const std::string& method, double exposure);

// The light the physical sky's clouds leave a figure, against a clear sky's at the same sun: the
// sky's share of a clear day's (tonemap.skyShare) as the sky now gives it, and the sun's as much of it
// as gets through. 1 for every other environment.
double cloudLight(const State& s, const CloudReading& clouds);

// What the exposure is multiplied by for the clouds: the light they take, given back, when the
// camera follows them (tonemap.followClouds); else 1.
double cloudExposure(const State& s, const CloudReading& clouds);

// The renderer's tmMethod for a tone-map name.
int tmMethod(const std::string& method);

// The environment's rotation about +Y as a glTF-order quaternion (x, y, z, w).
glm::vec4 envRotation(double degrees);

// A DAZ dome rotation as the panel's: three's lat-long seam is half a turn from Iray's.
double presetRotation(double dazDegrees);

// Degrees wrapped to [-180, 180).
using fv::patch::wrapDegrees;

// The built-in studio (createStudioEnvironment): RGBA32F rows from the top, the layout the renderer
// takes an environment in.
std::vector<float> studioImage(uint32_t width = 512, uint32_t height = 256);

// The registered renderer parameters `s` sets, by name, for nvmcp's setParameters. `envSize` is
// the environment image's size, which hdrBlur depends on. Ground height is the host's (auto needs
// the scene), so it isn't here; a backdrop that stands sets some over these
// (fv::backdrop::rendererParams). `clouds`, the renderer's last reading of the physical sky's clouds,
// for the exposure (cloudExposure).
Json rendererParams(const State& s, glm::uvec2 envSize, const CloudReading& clouds = {});

// The still's firefly clamp to send, given the one the exposure asks for (rendererParams'
// ptFireflyClampStill) and the one the renderer has: the one it has while it is within 4x of the one
// asked for, since a new value restarts the still. Inside that band its 64 display units stay 16-256,
// white under every tone curve, so exposure and tone-curve edits keep a converged still.
double stillFireflyClamp(double wanted, double applied);

}  // namespace fv::lighting
