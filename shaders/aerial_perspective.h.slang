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

#ifndef AERIAL_PERSPECTIVE_H
#define AERIAL_PERSPECTIVE_H 1

// Aerial perspective: the air between the camera and a surface.
//
// The environment answers what a ray sees when it *escapes* the scene. This answers what it sees
// when it stops at something -- the same atmosphere, integrated over a segment instead of a whole
// ray. It is why distant hills go pale and blue while near ones do not, and it is the one thing a
// baked lat-long can never supply, because it depends on how far away the surface is.
//
// The caller declares the LUT bindings at its own set index and passes them in, as the Bruneton
// port does everywhere else.
//
// Physical sky only. The other environments do not model air: a gradient is a painting, a loaded
// HDR already has whatever haze the photographer stood in, and inventing an atmosphere for either
// would be this renderer overruling the author.

#include "shaderio.h"
#include "nvshaders/functions.h.slang"  // luminance, quatRotateInverse
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"
#include "sky_bruneton_radiance.h.slang"  // skyEyeRadius

// Cheap test the call sites use to skip the work entirely.
//
// `sunDisk` is non-null exactly for the physical sky -- the same idiom the background pass uses to
// decide whether to draw a solar disk -- so there is no second flag to keep in step with the first.
bool hasAerialPerspective(SceneFrameInfo* frameInfo)
{
  return frameInfo->sunDisk != nullptr && frameInfo->aerialPerspectiveScale > 0.0F;
}

// In-scattered light and transmittance along the segment, by marching it.
//
// **Why a march and not a table lookup.** GetSkyRadianceToPoint answers the same question in two
// texture fetches, by subtracting everything scattered beyond the surface from everything scattered
// beyond the eye. That is elegant and it is what upstream ships, and it fails here in a way that is
// impossible to unsee: the table is stored in two halves, rays that meet the planet and rays that
// escape, and which half to read is decided by where the *extended* ray goes. A view ray tipping
// past the horizon switches halves while the five kilometres of air being shaded does not change,
// and the halves do not agree. Measured on a 10 km mountain, the escaping half spikes in the few
// rows just above the horizon -- where its mu parameterisation is most compressed -- and draws a
// hard line across the rock. Forcing the other half removes the line and is wrong by 20/255 higher
// up, where the ray really does escape.
//
// Marching has no halves, no branch and no subtraction of two large nearly-equal numbers. It is what
// Unreal and Hillaire do for aerial perspective, for the same reason.
//
// Multiple scattering is the one approximation: Bruneton's tables integrate it along whole rays and
// cannot hand back the local source term a march needs, so it comes from the irradiance table as an
// isotropic term (aerialSkyLight): the sky above and the ground's light below. In clear air the
// sun's single scattering is most of the light; in a mist the sun is spent within the first
// kilometre and this is nearly all of it, and a haze's air is raised to the tables' own light along the
// ray where it falls short of it (hazeGlow, below).
//
// **The sun's share, and one point on it.** Every point of the march is treated as sunlit, with only
// the planet's shadow in it: the atmosphere knows nothing of the scene. So the sun's single
// scattering is also returned on its own, together with one point on the segment picked in
// proportion to it -- `sunEstimate` is the one-sample estimate of `sunInScatter` taken at that point
// (step chosen with probability luminance / total, `u.y` placing it within the step). A caller that
// traces scene visibility from that point to the sun (the path tracer) replaces `sunInScatter` by
// `visibility * sunEstimate`, which is unbiased and converges as samples accumulate. The
// multiple-scattering term is light from the whole sky, not from the sun, and is left as it is.
// The light at radius `r` coming to the air from every direction but the sun's own, per unit of its
// scattering, for an isotropic phase: (1 / 4 pi) times its integral over the sphere. The sky above
// is taken as even, of the radiance E / pi that gives the irradiance table's E on a level surface,
// and so is the ground below, of what it reflects of the sky's light and the sun's (the
// atmosphere's ground albedo, a backdrop's land's colour under one): each half the sphere,
// (E + albedo (E + E_sun)) / 2 pi. E / 4 pi had been taken, half the sky and none of the ground: a
// clear day's air is the sun's and didn't show it (the meadow's tree line within 0.001 of 255), and
// in 5 km of visibility the tree line 260 m off came out at 0.05 of the screen, now 0.08 (2026-11-12).
// Even, it has no glow toward the sun, which a hazy sky's light near it is: hazeGlow supplies that.
float3 aerialSkyLight(SkyAtmosphereParameters atmosphere, Sampler2D<float3> irradianceTexture, float r, float muSun, float3 sunTransmittance)
{
  const float3 sky         = GetIrradiance(atmosphere, irradianceTexture, r, muSun);
  const float3 sunOnGround = atmosphere.solarIrradiance * sunTransmittance * max(muSun, 0.0F);
  return (sky + atmosphere.groundAlbedo * (sky + sunOnGround)) * (1.0F / (2.0F * M_PI));
}

// **The haze's glow.** In a haze the sky's light is mostly light scattered many times, and it keeps a
// glow toward the sun that aerialSkyLight, even, has none of: a misted wood stayed dark under a sky
// ten times as bright beside it. The sky's own tables have it, along whole rays: what they scatter
// toward the eye along a direction, per unit of the air's opacity along it, is the light the air
// sends that way on average, and a segment of that air, of opacity 1 - T, sends that share of it.
// Where the air before a surface sends less than that, it is raised to it (hazeGlow): in a thick
// haze, a distant wood fades into the sky above it, as a misted one does.
//
// It holds for air the same along the ray, which a mist near the ground is and a clear day's air,
// its light mostly the sun's single scattering that the march has exactly, is not: it is weighed in
// by the visibility (hazeGlowWeight), none at kHazeGlowFromKm and over, all of it at kHazeGlowFullKm
// and under; a clear day is marched as before, and costs nothing more.
//
// Only the half of the tables for rays that escape is read: a ray under the horizon takes the light of
// one kHazeGlowHorizonMu over it, at its own angle to the sun. The other half, for rays that meet the
// planet, is of rays a few metres long from an eye by the ground, read between rows of its height
// that are of longer ones: their light over the short ray's opacity drew a band four times the sky's
// under the horizon (2 km of visibility), where GetSkyRadianceToPoint drew its line
// (sky_bruneton_functions.h.slang). A misted land fades into its horizon's light. Light coming toward
// an eye that looks down comes up from under it, the ground's, far dimmer than the horizon's glow
// and what aerialSkyLight already takes: the horizon's light is tapered out by kHazeGlowDownMu under
// the horizon (about 30 degrees), and the ground at the feet keeps the march's own. Taken whole, the
// few metres of air before the grass by the camera, counted over by a backdrop, put a horizon's glow
// on it: the ground 4.5 times as bright as on a clear day under a 60-degree sun (2 km).
//
// The tables are read from where the sky the camera sees is seen from (`skyRadius`; skyEyeRadius),
// which over a backdrop is not where the march's eye stands: that counts the land's air several
// times over, its height with it. Read from the march's, a far hill seen from 150 m up stood 40%
// darker than the sky over it (5 km of visibility), where the air before it is as thick as the
// sky's.
//
// (The numbers above were measured on the tables before their steps were integrated exactly, when a
// mist's sky came out several times too bright: sky_bruneton_functions.h.slang,
// IntegrateExponentialStep. What they ruled out still stands.)
static const float kHazeGlowFromKm     = 30.0F;
static const float kHazeGlowFullKm     = 8.0F;
static const float kHazeGlowHorizonMu  = 0.03F;
static const float kHazeGlowDownMu     = 0.5F;
static const float kKoschmiederContrast = 3.912F;  // -ln 0.02: what a visibility's 2% of contrast is in optical depth

// The share of hazeGlow to take for an atmosphere: by its visibility by the ground, as
// sky_visibility::aerosolScale sets it (Rayleigh and aerosol extinction in the green).
float hazeGlowWeight(SkyAtmosphereParameters atmosphere)
{
  const float extinction = atmosphere.rayleighScattering.y * GetProfileDensity(atmosphere.rayleighDensity, 0.0F)
                           + atmosphere.mieExtinction.y * GetProfileDensity(atmosphere.mieDensity, 0.0F);
  const float visibility = kKoschmiederContrast / max(extinction, 1e-6F);
  return saturate((kHazeGlowFromKm - visibility) / (kHazeGlowFromKm - kHazeGlowFullKm));
}

// What a haze's air of `opacity` (1 - its transmittance) sends toward `observer` along `dir`, by
// the tables (see above): the light it would raise the march's to.
float3 hazeGlow(SkyAtmosphereParameters atmosphere,
                Sampler2D<float3>       transmittanceTexture,
                Sampler3D<float4>       scatteringTexture,
                Sampler3D<float4>       singleMieScatteringTexture,
                float3                  observer,
                float3                  dir,
                float3                  sunDirection,
                float3                  opacity)
{
  const float r    = max(length(observer), atmosphere.bottomRadius + 1e-4F);
  const float mu   = dot(observer, dir) / r;
  const float mu_s = dot(observer, sunDirection) / r;
  const float nu   = dot(dir, sunDirection);

  const float sinHorizon = atmosphere.bottomRadius / r;
  const float horizon    = -sqrt(max(1.0F - sinHorizon * sinHorizon, 0.0F));
  const float above      = max(mu, horizon + kHazeGlowHorizonMu);
  const float taper      = saturate(1.0F + (mu - horizon) / kHazeGlowDownMu);
  if(taper <= 0.0F)
    return float3(0.0F);

  // what the tables scatter toward the eye along it, over the opacity along it to the top of the air
  float3       mie;
  const float3 rayleigh = GetCombinedScattering(atmosphere, scatteringTexture, singleMieScatteringTexture, r, above, mu_s, nu, false, mie);
  const float3 light    = rayleigh * RayleighPhaseFunction(nu) + mie * MiePhaseFunction(atmosphere.miePhaseFunctionG, nu);
  const float3 through  = GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r, above);
  return light / max(float3(1.0F) - through, float3(1e-4F)) * opacity * taper;
}

struct AerialMarch
{
  float3 inScatter;      // everything the air adds, sun included
  float3 transmittance;  // what is left of the surface behind it
  float3 sunInScatter;   // the sun's single-scattering part of inScatter
  float3 sunEstimate;    // one-sample estimate of sunInScatter at `sunPoint`; zero when the sun adds nothing
  float  sunPoint;       // where that sample stands, 0 at the eye to 1 at the surface
  float3 glow;           // what hazeGlow raised inScatter by, a part of it besides the sun's
};

AerialMarch marchAerialPerspective(SkyAtmosphereParameters atmosphere,
                                   Sampler2D<float3>       transmittanceTexture,
                                   Sampler3D<float4>       scatteringTexture,
                                   Sampler3D<float4>       singleMieScatteringTexture,
                                   Sampler2D<float3>       irradianceTexture,
                                   float3                  observer,
                                   float                   skyRadius,
                                   float3                  displacement,
                                   float3                  sunDirection,
                                   float2                  u)
{
  // One step per quarter kilometre, between one and sixteen: up to 4 km a step is at most 250 m,
  // beyond that the sixteen steps lengthen with the segment, as every step did before. The air
  // varies over kilometres (the aerosol's scale height is about 1.2 km), and the per-step
  // integration below is analytic rather than a Riemann sum, so a step only has to be short against
  // that scale. A room is metres deep and gets one step.
  const float kMaxStepKm = 0.25F;
  const int   kMaxSteps  = 16;

  const float  d      = length(displacement);
  const float3 dir    = displacement / d;
  const float  nu     = dot(dir, sunDirection);
  const int    kSteps = clamp(int(ceil(d / kMaxStepKm)), 1, kMaxSteps);

  // Angular terms are constant along a straight segment, so they leave the loop.
  const float rayleighPhase = RayleighPhaseFunction(nu);
  const float miePhase      = MiePhaseFunction(atmosphere.miePhaseFunctionG, nu);
  const float stepLength    = d / float(kSteps);

  float3 inScatter  = float3(0.0F);
  float3 throughput = float3(1.0F);

  // The sun's share, and the streaming pick over it: step i replaces the current choice with
  // probability luminance_i / (running total), one uniform rescaled after each decision.
  float3 sunInScatter = float3(0.0F);
  float  sunTotal     = 0.0F;
  float3 sunPicked    = float3(0.0F);
  float  pickedLum    = 0.0F;
  int    pickedStep   = 0;
  float  pick         = u.x;

  for(int i = 0; i < kSteps; ++i)
  {
    const float3 p        = observer + dir * (float(i) + 0.5F) * stepLength;
    const float  r        = max(length(p), atmosphere.bottomRadius);
    const float  altitude = r - atmosphere.bottomRadius;

    const float rayleighDensity   = GetProfileDensity(atmosphere.rayleighDensity, altitude);
    const float mieDensity        = GetProfileDensity(atmosphere.mieDensity, altitude);
    const float absorptionDensity = GetProfileDensity(atmosphere.absorptionDensity, altitude);

    const float3 rayleighScatter = atmosphere.rayleighScattering * rayleighDensity;
    const float3 mieScatter      = atmosphere.mieScattering * mieDensity;
    // Extinction, not scattering: the aerosol absorbs as well as redirects, and the ozone only
    // absorbs. A segment that used scattering here would never go dark, only bluer.
    const float3 extinction =
        max(rayleighScatter + atmosphere.mieExtinction * mieDensity + atmosphere.absorptionExtinction * absorptionDensity,
            float3(1e-9F));

    // How much sun reaches this point, with the planet's own shadow already in it.
    const float  muSun            = dot(p, sunDirection) / r;
    const float3 sunTransmittance = GetTransmittanceToSun(atmosphere, transmittanceTexture, r, muSun);

    const float3 sunSource =
        atmosphere.solarIrradiance * sunTransmittance * (rayleighScatter * rayleighPhase + mieScatter * miePhase);
    const float3 skySource = (rayleighScatter + mieScatter) * aerialSkyLight(atmosphere, irradianceTexture, r, muSun, sunTransmittance);

    // Integrated over the step in closed form, assuming the source is constant across it: the
    // energy-conserving form, which does not lose light as the step count drops.
    const float3 stepTransmittance = exp(-extinction * stepLength);
    const float3 stepFraction      = throughput * (float3(1.0F) - stepTransmittance) / extinction;
    const float3 sunStep           = sunSource * stepFraction;
    inScatter += sunStep + skySource * stepFraction;
    sunInScatter += sunStep;
    throughput *= stepTransmittance;

    const float weight = luminance(sunStep);
    if(weight > 0.0F)
    {
      sunTotal += weight;
      const float keep = weight / sunTotal;
      if(pick < keep)
      {
        sunPicked  = sunStep;
        pickedLum  = weight;
        pickedStep = i;
        pick       = pick / keep;
      }
      else
      {
        pick = (pick - keep) / (1.0F - keep);
      }
    }
  }

  // In a haze, no less than the tables' light along the ray (hazeGlow)
  float3      glow   = float3(0.0F);
  const float weight = hazeGlowWeight(atmosphere);
  if(weight > 0.0F)
  {
    const float3 table = hazeGlow(atmosphere, transmittanceTexture, scatteringTexture, singleMieScatteringTexture,
                                  normalize(observer) * skyRadius, dir, sunDirection, float3(1.0F) - throughput);
    glow               = weight * max(table - inScatter, float3(0.0F));
    inScatter += glow;
  }

  AerialMarch march;
  march.inScatter     = inScatter;
  march.glow          = glow;
  march.transmittance = throughput;
  march.sunInScatter  = sunInScatter;
  march.sunEstimate   = (pickedLum > 0.0F) ? sunPicked * (sunTotal / pickedLum) : float3(0.0F);
  march.sunPoint      = (float(pickedStep) + u.y) / float(kSteps);
  return march;
}

// What the camera sees of a surface, once the air in between has had its say.
//
//     seen = surfaceRadiance * transmittance + inScatter
//
// `worldPos` is the surface, `cameraPos` the eye, both in world units. The atmosphere works in
// kilometres from the planet's centre, and the two spaces are bridged by the same fixed-observer
// policy the bake uses: the eye sits at the observer altitude above the surface, wherever it is in
// the scene, and the surface sits where the scene puts it *relative* to the eye.
//
// That is the only mapping that can be right here. A scene is a few metres or a few hundred wide
// and stands nowhere in particular on a planet 6371 km in radius; what aerial perspective needs to
// know is not where the scene is but how much air is inside it, and that is the relative
// displacement. It also means the haze does not lurch when the camera moves, which a mapping
// anchored to the scene's origin would do.
//
// The conversion assumes **one scene unit is one metre**, which is what glTF specifies, and
// `aerialPerspectiveScale` multiplies it for the scenes that do not obey -- or for an artist who
// wants more air than the model gives. It is a distance scale rather than an opacity: doubling it
// says "this scene is twice as big", which stays inside the model instead of tinting its output.
//
// The segment is marched in the environment's frame, as the bake is: up is the rotated sky's up, so
// a tilted sky tilts its haze with it.
//
// Applied to the whole radiance the surface sends back, indirect bounces included. That is exact
// for the segment between the eye and the first surface, which is what this models, and it says
// nothing about the air *inside* the scene between later bounces -- correct at the scale a glTF
// asset actually occupies, and the place to look first if someone ever renders a valley.
AerialMarch evalAerialPerspective(float3                  worldPos,
                                  float3                  cameraPos,
                                  SceneFrameInfo*         frameInfo,
                                  SkyAtmosphereParameters atmosphere,
                                  Sampler2D<float3>       transmittanceTexture,
                                  Sampler3D<float4>       scatteringTexture,
                                  Sampler3D<float4>       singleMieScatteringTexture,
                                  Sampler2D<float3>       irradianceTexture,
                                  float2                  u)
{
  const float  kmPerSceneUnit = 0.001F * frameInfo->aerialPerspectiveScale;
  const float3 displacement   = quatRotateInverse(frameInfo->envRotation, (worldPos - cameraPos) * kmPerSceneUnit);

  // A surface the eye is standing on has no air in front of it, and normalize() of a zero vector
  // is a NaN that would spread across the whole pixel.
  if(dot(displacement, displacement) < 1e-16F)
  {
    AerialMarch none;
    none.inScatter     = float3(0.0F);
    none.transmittance = float3(1.0F);
    none.sunInScatter  = float3(0.0F);
    none.sunEstimate   = float3(0.0F);
    none.sunPoint      = 0.0F;
    none.glow          = float3(0.0F);
    return none;
  }

  // Up is +Y in the environment's frame, as it is in the bake: the model only reaches the world
  // through dot products against this vector.
  const float3 observer = float3(0.0F, atmosphere.bottomRadius + frameInfo->observerAltitudeKm, 0.0F);
  const float3 sunDir   = quatRotateInverse(frameInfo->envRotation, frameInfo->sunDirection);
  const float  skyRadius =
      skyEyeRadius(atmosphere, frameInfo->sunDisk->observerRadius, frameInfo->observerAltitudeKm, frameInfo->aerialPerspectiveScale);
  AerialMarch march = marchAerialPerspective(atmosphere, transmittanceTexture, scatteringTexture, singleMieScatteringTexture,
                                             irradianceTexture, observer, skyRadius, displacement, sunDir, u);

  // Under a cloud the air in the scene is in its shadow as well as the scene is. The sums take the
  // share at the scene's origin (SkySunDisk::cloudShadow, 1 under a clear sky); the one-sample
  // estimate is left clear, for the caller to take the share where its point stands (cloudSunShare
  // in gltf_pathtrace.slang), which puts the air under a gap in the light and the rest in shade.
  // A haze's glow is the sun's light scattered many times over kilometres of it, and under the clouds
  // takes the share of the distant haze the sun still reaches (sunlitGround), as the sky below the
  // horizon does: the origin's own shade left a misted wood black under a cloud against a lit sky.
  const float cloudShadow = frameInfo->sunDisk->cloudShadow;
  const float glowShare   = frameInfo->sunDisk->sunlitGround;
  march.inScatter -= march.sunInScatter * (1.0F - cloudShadow) + march.glow * (1.0F - glowShare);
  march.sunInScatter *= cloudShadow;
  march.glow *= glowShare;

  // The transmittance is dimensionless and needs no conversion; the in-scatter is spectral radiance
  // at three wavelengths and does, exactly as the bake converts what GetSkyRadiance hands it. Miss
  // this and the two halves disagree by orders of magnitude: distant surfaces dim correctly and
  // gain no haze to replace what they lost, so a mountain goes black instead of blue.
  const float3 toScene = SKY_RADIANCE_TO_LUMINANCE * frameInfo->envIntensity;
  march.inScatter *= toScene;
  march.sunInScatter *= toScene;
  march.sunEstimate *= toScene;
  march.glow *= toScene;
  return march;
}

#endif  // AERIAL_PERSPECTIVE_H
