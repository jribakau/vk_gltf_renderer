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
// The air under the clouds. The clear sky (evalBrunetonSkyRadiance) holds the light the air scatters
// along a ray as though the clouds were not there: every part of it sunlit, and lit by the whole
// clear sky. Under the layers the clouds take a share of both. The cloud bake measures the two shares
// once a bake (airMain in sky_clouds_bake.slang): the sun's, as what reaches the land through the
// clouds toward it, which a low sun's slant through a broken deck all but closes; and the sky's, as
// what is left of the sky's light with the clouds over it.
//
// The air in front of a cloud is lit by those shares in the bake (marchClouds' airInFront). This is
// the rest of the air under the layers, seen in the clear sky's own light: through the gaps between
// clouds, up to where the ray leaves the top of the highest layer, and below the horizon, all of the
// air in front of the ground. Toward a low sun that air is the brightest part of the sky; without its
// shade it glowed at the horizon through a broken deck, and below it under a closed one as brightly
// as under a clear sky.
//
// The shade is a share of the clear sky, not an amount: the air's in-scatter is marched here, and the
// share of it in the layers' shadow times the clear sky is what goes. The sky's tables and a march
// integrate the same air differently, by as much as half of it toward a low sun below the horizon,
// so taking the march's amount off the tables' sky would leave or take the difference.
//
// A share, not shafts: the gaps' own pattern of light and shadow in the air is not marched.
//
// The cloud bake takes what the gaps' air loses off each texel, by the texel's transparency, so the
// image composes to the shaded sky (sky_clouds_bake.slang main); below the horizon, where the image
// does not reach, the path tracer's background and the environment bake take it off per ray.
//

#ifndef SKY_CLOUDS_AIR_H
#define SKY_CLOUDS_AIR_H 1

#include "aerial_perspective.h.slang"  // brings the Bruneton functions
#include "sky_clouds.h.slang"          // cloudLayerSegment

// What the air scatters toward the eye along [start, end] of a ray, `throughput` the transmittance
// from the eye to `start`, carried on: the sun's single scattering (marchAerialPerspective's source
// terms, its planet shadow in it) and everything, the sky's light included. Steps lengthen with
// distance on a square, the dense air near the start in short ones.
struct CloudAirStretch
{
  float3 sun;
  float3 all;
};

CloudAirStretch cloudAirStretch(SkyAtmosphereParameters atmosphere,
                                Sampler2D<float3>       transmittanceTexture,
                                Sampler2D<float3>       irradianceTexture,
                                float3                  observer,
                                float3                  dir,
                                float3                  sunDirection,
                                float                   start,
                                float                   end,
                                int                     steps,
                                inout float3            throughput)
{
  CloudAirStretch stretch;
  stretch.sun = float3(0.0F);
  stretch.all = float3(0.0F);
  if(end <= start)
    return stretch;

  const float nu            = dot(dir, sunDirection);
  const float rayleighPhase = RayleighPhaseFunction(nu);
  const float miePhase      = MiePhaseFunction(atmosphere.miePhaseFunctionG, nu);
  float       previous      = start;
  for(int i = 0; i < steps; ++i)
  {
    const float  s          = float(i + 1) / float(steps);
    const float  t          = start + (end - start) * s * s;
    const float  stepLength = t - previous;
    const float3 p          = observer + dir * (0.5F * (t + previous));
    previous                = t;
    const float r           = max(length(p), atmosphere.bottomRadius);
    const float altitude    = r - atmosphere.bottomRadius;

    const float3 rayleighScatter = atmosphere.rayleighScattering * GetProfileDensity(atmosphere.rayleighDensity, altitude);
    const float  mieDensity      = GetProfileDensity(atmosphere.mieDensity, altitude);
    const float3 mieScatter      = atmosphere.mieScattering * mieDensity;
    const float3 extinction      = max(rayleighScatter + atmosphere.mieExtinction * mieDensity
                                           + atmosphere.absorptionExtinction * GetProfileDensity(atmosphere.absorptionDensity, altitude),
                                       float3(1e-9F));

    const float  muSun            = dot(p, sunDirection) / r;
    const float3 sunTransmittance = GetTransmittanceToSun(atmosphere, transmittanceTexture, r, muSun);
    const float3 sunSource = atmosphere.solarIrradiance * sunTransmittance * (rayleighScatter * rayleighPhase + mieScatter * miePhase);
    const float3 skySource = (rayleighScatter + mieScatter) * aerialSkyLight(atmosphere, irradianceTexture, r, muSun, sunTransmittance);

    const float3 stepTransmittance = exp(-extinction * stepLength);
    const float3 stepFraction      = throughput * (float3(1.0F) - stepTransmittance) / extinction;
    stretch.sun += sunSource * stepFraction;
    stretch.all += (sunSource + skySource) * stepFraction;
    throughput *= stepTransmittance;
  }
  return stretch;
}

// What the clouds' shade takes from the clear sky's air along `dir`, from an observer `observerRadius`
// from the planet's centre on +Y: of the light the air the ray sees scatters, the share inside the
// shell from the ground to `topHeight` (km above the surface, the highest layer's top) that the sun
// no longer reaches (1 - sunShare) and the sky's light no longer reaches (1 - skyShare), times
// `clearAir`, that light as the sky's tables have it (evalBrunetonSkyInScatter: the air alone, not the
// ground behind it). A share of the march's own light, per channel, so never more than `clearAir`;
// zero for a clear sky.
float3 cloudShadeOfClearSky(SkyAtmosphereParameters atmosphere,
                            Sampler2D<float3>       transmittanceTexture,
                            Sampler2D<float3>       irradianceTexture,
                            float                   observerRadius,
                            float3                  dir,
                            float3                  sunDirection,
                            float                   topHeight,
                            float                   sunShare,
                            float                   skyShare,
                            float3                  clearAir)
{
  const float sunShaded = 1.0F - clamp(sunShare, 0.0F, 1.0F);
  const float skyShaded = 1.0F - clamp(skyShare, 0.0F, 1.0F);
  if((sunShaded <= 0.0F && skyShaded <= 0.0F) || topHeight <= 0.0F)
    return float3(0.0F);

  // The ray's stretch in the shell, from the observer or from where it comes down into the shell when
  // the observer is above the clouds, to where it leaves the top or meets the ground; and the whole
  // of the air it sees, to the ground or out of the atmosphere.
  const float  R       = atmosphere.bottomRadius;
  const float  mu      = dir.y;
  const float2 segment = cloudLayerSegment(observerRadius, mu, R, R + topHeight);
  if(segment.y <= segment.x)
    return float3(0.0F);
  const float whole = RayIntersectsGround(atmosphere, observerRadius, mu) ? DistanceToBottomAtmosphereBoundary(atmosphere, observerRadius, mu) :
                                                                            DistanceToTopAtmosphereBoundary(atmosphere, observerRadius, mu);

  const float3          observer   = float3(0.0F, observerRadius, 0.0F);
  float3                throughput = float3(1.0F);
  const CloudAirStretch before     = cloudAirStretch(atmosphere, transmittanceTexture, irradianceTexture, observer, dir, sunDirection,
                                                     0.0F, segment.x, 8, throughput);
  const CloudAirStretch shell = cloudAirStretch(atmosphere, transmittanceTexture, irradianceTexture, observer, dir, sunDirection,
                                                segment.x, segment.y, 24, throughput);
  const CloudAirStretch after = cloudAirStretch(atmosphere, transmittanceTexture, irradianceTexture, observer, dir, sunDirection,
                                                segment.y, max(whole, segment.y), 12, throughput);

  const float3 all   = max(before.all + shell.all + after.all, float3(1e-20F));
  const float3 taken = shell.sun * sunShaded + (shell.all - shell.sun) * skyShaded;
  return clearAir * (taken / all);
}

#endif  // SKY_CLOUDS_AIR_H
