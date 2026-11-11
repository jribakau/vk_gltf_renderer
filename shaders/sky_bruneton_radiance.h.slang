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

//
// The physical sky's radiance toward a direction, sun excluded -- one function for the two places
// that need it: the bake (env_bake.slang), which writes it into the lighting environment, and the
// path tracer, which evaluates it per ray for what the camera and mirrors see
// (evalPhysicalSkyBackground in gltf_pathtrace.slang). Sharing it is what keeps the lit scene and
// the visible sky the same sky; a second copy would drift silently.
//
// The observer is a point on the +Y axis, `observerRadius` km from the planet's centre. The bake's
// is fixed wherever the scene camera is, which is what lets the sky be baked at all -- an observer
// that moved with the camera would invalidate the image every frame -- and it publishes the radius
// it used in SkySunDisk::observerRadius. The path tracer's per-ray caller stands at the camera's own
// height (skyEyeRadius), which by the ground is the bake's and in the air is not: there the sky
// the camera sees has its horizon lower than the lighting field's, by a degree or two. Up is +Y because the rest of the renderer is
// Y-up; the model only ever uses the observer through dot products, so the axis is ours to pick.
//

#ifndef SKY_BRUNETON_RADIANCE_H
#define SKY_BRUNETON_RADIANCE_H

#include "nvshaders/constants.h.slang"  // M_PI
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"

// Cosine of the zenith angle at which the ground starts, for an observer `r` km from the planet's
// centre: below it a ray ends on the planet. The bake publishes it (SkySunDisk::horizonMu), and the
// cloud image starts its rows there.
float brunetonHorizonMu(SkyAtmosphereParameters atmosphere, float r)
{
  const float sinHorizon = atmosphere.bottomRadius / r;
  return -sqrt(max(1.0F - sinHorizon * sinHorizon, 0.0F));
}

// Where the camera's eye is for the sky it sees: at its own height over the scene's floor, which is
// where the bake's observer stands. `eyeAltitudeKm` is the eye aerial perspective stands
// (SceneFrameInfo::observerAltitudeKm), whose height over that floor is multiplied by `airScale`
// (aerialPerspectiveScale: a backdrop counts its land's air several times over); the sky is seen
// from the height itself, as the cloud images are baked, whose air is summed against this sky.
// Never under the bake's observer, and inside the atmosphere with room to its top.
float skyEyeRadius(SkyAtmosphereParameters atmosphere, float bakeObserverRadius, float eyeAltitudeKm, float airScale)
{
  const float top   = atmosphere.topRadius - 0.05F * (atmosphere.topRadius - atmosphere.bottomRadius);
  const float floor = bakeObserverRadius - atmosphere.bottomRadius;
  const float above = airScale > 0.0F ? max(eyeAltitudeKm - floor, 0.0F) / airScale : 0.0F;
  return clamp(bakeObserverRadius + above, min(bakeObserverRadius, top), top);
}

// The light the air scatters toward the observer along `dir`, to the ground or out of the atmosphere:
// the sky above the horizon, and below it the haze in front of the ground (evalBrunetonSkyRadiance
// adds the ground). In the units evalBrunetonSkyRadiance returns.
float3 evalBrunetonSkyInScatter(SkyAtmosphereParameters atmosphere,
                                Sampler2D<float3>       transmittanceTexture,
                                Sampler3D<float4>       scatteringTexture,
                                Sampler3D<float4>       singleMieScatteringTexture,
                                float                   observerRadius,
                                float3                  dir,
                                float3                  sunDirection)
{
  float3 transmittance;
  return GetSkyRadiance(atmosphere, transmittanceTexture, scatteringTexture, singleMieScatteringTexture,
                        float3(0.0F, observerRadius, 0.0F), dir, sunDirection, transmittance)
         * SKY_RADIANCE_TO_LUMINANCE;
}

// `dir` and `sunDirection` are unit vectors in the environment's frame. Returns luminance, in the
// same units as the baked image. `sunlitGround` is the share of the ground the sun reaches past the
// clouds (SkySunDisk::sunlitGround), 1 under a clear sky; the clouds themselves are composed over
// this by the caller (composeClouds in sky_clouds.h.slang).
float3 evalBrunetonSkyRadiance(SkyAtmosphereParameters atmosphere,
                               Sampler2D<float3>       transmittanceTexture,
                               Sampler3D<float4>       scatteringTexture,
                               Sampler3D<float4>       singleMieScatteringTexture,
                               Sampler2D<float3>       irradianceTexture,
                               float                   observerRadius,
                               float3                  dir,
                               float3                  sunDirection,
                               float                   sunlitGround)
{
  const float3 observer = float3(0.0F, observerRadius, 0.0F);

  // Spectral radiance at three wavelengths is not a colour until the CIE integral is applied.
  // Skipping it leaves the sky both far too dark and too blue, since the three factors differ.
  float3       transmittance;
  const float3 skyRadiance = GetSkyRadiance(atmosphere, transmittanceTexture, scatteringTexture,
                                            singleMieScatteringTexture, observer, dir, sunDirection, transmittance);
  float3       color       = skyRadiance * SKY_RADIANCE_TO_LUMINANCE;

  // Below the horizon, add the ground.
  //
  // GetSkyRadiance already returns the right thing for a downward ray -- the scattering LUT is
  // parameterised by whether the ray meets the ground, so what comes back is the light scattered
  // into the segment between the eye and the ground, and nothing beyond it. From an observer a
  // metre up that segment is short, which is why this half of the sphere was almost black: the
  // model was never asked what the ground itself reflects.
  //
  // So that is the only term missing. A Lambertian surface lit by sun and sky, attenuated by the
  // air between it and the eye -- and because the in-scatter above is already integrated over the
  // same segment, the ground fades into the horizon haze instead of ending at a seam. It is the
  // same aerial perspective the renderers apply per ray, for the one case this function can answer
  // on its own: a fixed observer looking at its own planet.
  const float mu = dir.y;  // the observer sits on +Y, so its zenith is +Y and mu is just dir.y
  if(RayIntersectsGround(atmosphere, observerRadius, mu))
  {
    const float  d      = DistanceToBottomAtmosphereBoundary(atmosphere, observerRadius, mu);
    const float3 point  = observer + dir * d;
    const float3 normal = normalize(point);

    float3       skyIrradiance;
    const float3 sunIrradiance = GetSunAndSkyIrradiance(atmosphere, transmittanceTexture, irradianceTexture, point,
                                                        normal, sunDirection, skyIrradiance);

    // Lambertian: albedo/pi times the irradiance it receives. Each irradiance is converted with
    // the factors for its own spectrum, as upstream's GetSunAndSkyIlluminance does: sunlight with
    // the sun's, skylight with the sky's. Under clouds only the open share of the land is sunlit.
    const float3 groundIlluminance =
        sunIrradiance * (SUN_RADIANCE_TO_LUMINANCE * sunlitGround) + skyIrradiance * SKY_RADIANCE_TO_LUMINANCE;
    const float3 groundColor = atmosphere.groundAlbedo * (1.0F / M_PI) * groundIlluminance;

    color += groundColor * GetTransmittance(atmosphere, transmittanceTexture, observerRadius, mu, d, true);
  }

  return color;
}

#endif  // SKY_BRUNETON_RADIANCE_H
