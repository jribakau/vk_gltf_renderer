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

#ifndef SKY_WORLD_H
#define SKY_WORLD_H 1

// The physical sky round a planet that stands in the scene (src/fv_world.hpp, src/sky_world.hpp;
// SceneFrameInfo::worldRadiusKm, worldCentre).
//
// Without one the sky has no place: its planet's centre is straight under whoever looks, and its
// light is one for the whole scene, measured at the bake's observer. That is right for a scene a few
// hundred metres across on a planet thousands of kilometres in radius. A planet twenty kilometres in
// radius that the camera flies round is neither: the eye may be over its far side or outside its air,
// and the sun that lights the origin has set on a quarter of it.
//
// So with a planet every question of the sky is asked from a place: a point of the scene, as
// kilometres from the planet's centre (skyWorldPoint), in the world's own axes. The planet doesn't
// turn with the environment's rotation, and neither does anything here: the sun's direction is the
// world's.
//
//   * what a ray sees of the sky (evalWorldSkyRadiance): from where it starts, inside the air or
//     outside it, to the top of the air or to the planet's sea level;
//   * the sun's light at a point (worldSunThrough): through the air over that point, none past the
//     planet's edge, so dusk runs round it;
//   * the sky's light at a point (worldSkyLight): the lighting field is baked at the scene's origin,
//     with that place's zenith and sun; elsewhere it is turned so its zenith is the point's and its
//     sun's bearing the sun's, and scaled by what the sky gives a level surface there against at the
//     origin. An approximation: the sun's height in it stays the origin's.
//
// Callers branch on skyHasWorld and keep what they did before in the other branch, so a scene with no
// planet is traced by the same code as before, bit for bit.

#include "shaderio.h"
#include "nvshaders/constants.h.slang"  // M_PI
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"

// Whether a planet stands in the scene under the physical sky.
bool skyHasWorld(SceneFrameInfo* frameInfo)
{
  return frameInfo->worldRadiusKm > 0.0F && frameInfo->sunDisk != nullptr;
}

// A point of the scene (world space, metres) as the sky's: kilometres from the planet's centre.
float3 skyWorldPoint(SceneFrameInfo* frameInfo, float3 worldPos)
{
  return (worldPos - frameInfo->worldCentre) * 0.001F;
}

// A ray from `p` (km from the centre) along `dir`, brought into the air: from outside it to where it
// enters, from under the sea's level (the sea's floor, a camera in the water) up to it.
struct WorldAirRay
{
  bool   inAir;    // false: the ray never meets the air
  float3 p;        // where it is in the air
  float  r;        // that point's distance from the centre
  float  mu;       // the cosine of the ray's angle from the point's zenith
  float  skipped;  // how far along the ray that point is, km
};

WorldAirRay worldAirRay(SkyAtmosphereParameters atmosphere, float3 p, float3 dir)
{
  WorldAirRay ray = {};
  float       r   = length(p);
  float       rmu = dot(p, dir);
  ray.inAir       = true;
  ray.skipped     = 0.0F;
  if(r > atmosphere.topRadius)
  {
    const float disc = rmu * rmu - r * r + atmosphere.topRadius * atmosphere.topRadius;
    if(rmu >= 0.0F || disc <= 0.0F)
    {
      ray.inAir = false;
      ray.p     = p;
      ray.r     = r;
      ray.mu    = rmu / r;
      return ray;
    }
    const float d = -rmu - sqrt(disc);
    p += dir * d;
    rmu += d;
    r           = atmosphere.topRadius;
    ray.skipped = d;
  }
  const float lowest = atmosphere.bottomRadius + 1e-4F;
  if(r < lowest)
  {
    const float lift = lowest / max(r, 1e-6F);
    p *= lift;
    rmu *= lift;
    r = lowest;
  }
  ray.p  = p;
  ray.r  = r;
  ray.mu = clamp(rmu / r, -1.0F, 1.0F);
  return ray;
}

// What the air leaves of the sun's light (or the moon's) coming to `p` from `lightDirection`: all of
// it above the air, none of it past the planet's edge, the disk cut by the horizon in between.
float3 worldSunThrough(SkyAtmosphereParameters atmosphere, Sampler2D<float3> transmittanceTexture, float3 p, float3 lightDirection)
{
  const WorldAirRay ray = worldAirRay(atmosphere, p, lightDirection);
  if(!ray.inAir)
    return float3(1.0F);
  return GetTransmittanceToSun(atmosphere, transmittanceTexture, ray.r, ray.mu);
}

// What a ray from `eye` (km from the centre) along `dir` sees of the sky lit from `sunDirection`:
// the light the air scatters toward it, and where the ray comes down on the planet's sea level, that
// ground lit by the sun and the sky, through the air in between. Luminance, in the units of
// evalBrunetonSkyRadiance, which this is from any place. `through` is what the air leaves of what
// lies behind it: the stars, the sun's and the moon's disks; none of it where the planet is.
float3 evalWorldSkyRadiance(SkyAtmosphereParameters atmosphere,
                            Sampler2D<float3>       transmittanceTexture,
                            Sampler3D<float4>       scatteringTexture,
                            Sampler3D<float4>       singleMieScatteringTexture,
                            Sampler2D<float3>       irradianceTexture,
                            float3                  eye,
                            float3                  dir,
                            float3                  sunDirection,
                            out float3              through)
{
  through               = float3(1.0F);
  const WorldAirRay ray = worldAirRay(atmosphere, eye, dir);
  if(!ray.inAir)
    return float3(0.0F);

  const float mu_s   = dot(ray.p, sunDirection) / ray.r;
  const float nu     = dot(dir, sunDirection);
  const bool  ground = RayIntersectsGround(atmosphere, ray.r, ray.mu);

  float3       mie;
  const float3 scattering =
      GetCombinedScattering(atmosphere, scatteringTexture, singleMieScatteringTexture, ray.r, ray.mu, mu_s, nu, ground, mie);
  float3 color = (scattering * RayleighPhaseFunction(nu) + mie * MiePhaseFunction(atmosphere.miePhaseFunctionG, nu)) * SKY_RADIANCE_TO_LUMINANCE;

  if(ground)
  {
    const float  d      = DistanceToBottomAtmosphereBoundary(atmosphere, ray.r, ray.mu);
    const float3 point  = ray.p + dir * d;
    const float3 normal = normalize(point);
    float3       skyIrradiance;
    const float3 sunIrradiance =
        GetSunAndSkyIrradiance(atmosphere, transmittanceTexture, irradianceTexture, point, normal, sunDirection, skyIrradiance);
    const float3 illuminance = sunIrradiance * SUN_RADIANCE_TO_LUMINANCE + skyIrradiance * SKY_RADIANCE_TO_LUMINANCE;
    color += atmosphere.groundAlbedo * (1.0F / M_PI) * illuminance * GetTransmittance(atmosphere, transmittanceTexture, ray.r, ray.mu, d, true);
    through = float3(0.0F);
  }
  else
  {
    through = GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, ray.r, ray.mu);
  }
  return color;
}

// The sky's light at a place of the planet, from the lighting field baked at the scene's origin.
struct WorldSkyLight
{
  // the place's frame and the origin's: toward the sun along the ground, straight up, and the third
  float3 sunward;
  float3 up;
  float3 side;
  float3 originSunward;
  float3 originSide;
  // what the sky gives a level surface there against at the origin
  float3 scale;
};

// The frame's first axis: toward the sun along the ground of a place whose zenith is `up`.
float3 worldSunward(float3 up, float3 sunDirection)
{
  float3 along = sunDirection - up * dot(sunDirection, up);
  if(dot(along, along) < 1e-8F)
  {
    // the sun straight overhead, or under: any bearing
    const float3 other = abs(up.x) < 0.9F ? float3(1.0F, 0.0F, 0.0F) : float3(0.0F, 0.0F, 1.0F);
    along              = other - up * dot(other, up);
  }
  return normalize(along);
}

// `p`: the place, km from the centre. `originRadius`: the bake's observer's (SkySunDisk::observerRadius),
// which stands over the scene's origin, its zenith the world's +Y. `secondDirection`, `secondScale`: the
// other body's sky (SceneFrameInfo::skySecondDirection), 0 for none.
WorldSkyLight worldSkyLight(SkyAtmosphereParameters atmosphere,
                            Sampler2D<float3>       irradianceTexture,
                            float3                  p,
                            float                   originRadius,
                            float3                  sunDirection,
                            float3                  secondDirection,
                            float                   secondScale)
{
  WorldSkyLight sky = {};
  const float   r   = clamp(length(p), atmosphere.bottomRadius, atmosphere.topRadius);
  sky.up            = p / max(length(p), 1e-6F);
  sky.sunward       = worldSunward(sky.up, sunDirection);
  sky.side          = cross(sky.up, sky.sunward);
  const float3 originUp = float3(0.0F, 1.0F, 0.0F);
  sky.originSunward     = worldSunward(originUp, sunDirection);
  sky.originSide        = cross(originUp, sky.originSunward);

  float3 here  = GetIrradiance(atmosphere, irradianceTexture, r, dot(sky.up, sunDirection));
  float3 there = GetIrradiance(atmosphere, irradianceTexture, originRadius, sunDirection.y);
  if(secondScale > 0.0F)
  {
    here += secondScale * GetIrradiance(atmosphere, irradianceTexture, r, dot(sky.up, secondDirection));
    there += secondScale * GetIrradiance(atmosphere, irradianceTexture, originRadius, secondDirection.y);
  }
  // the night's own light is the same all round the planet: a hundred-millionth of the sun's, which
  // also keeps a night at both places from dividing nothing by nothing
  const float3 night = atmosphere.solarIrradiance * 1e-8F;
  sky.scale          = min((here + night) / (there + night), float3(1e6F));
  return sky;
}

// A direction at the place, as the direction of the origin's lighting field that holds its light.
float3 worldSkyToField(WorldSkyLight sky, float3 dir)
{
  return sky.originSunward * dot(dir, sky.sunward) + float3(0.0F, dot(dir, sky.up), 0.0F) + sky.originSide * dot(dir, sky.side);
}

// ... and a direction of the field as the place's.
float3 worldSkyFromField(WorldSkyLight sky, float3 dir)
{
  return sky.sunward * dot(dir, sky.originSunward) + sky.up * dir.y + sky.side * dot(dir, sky.originSide);
}

#endif  // SKY_WORLD_H
