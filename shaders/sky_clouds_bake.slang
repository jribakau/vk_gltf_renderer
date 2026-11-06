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
// Bakes the cloud image: for every direction the fixed observer sees above its horizon, a march
// through the cloud layers, then the air between the observer and the clouds. Recorded by
// SkyClouds::record() just before the environment bake, which composes it over the sky
// (env_bake.slang), as the path tracer does for what the camera sees (gltf_pathtrace.slang). What a
// texel holds, and why, is sky_clouds.h.slang.
//
// The march is takram's (three-geospatial, packages/clouds/src/shaders/clouds.frag: marchClouds,
// marchOpticalDepth, approximateRadianceFromGround; MIT, Copyright (c) 2024 Shota Matsuda, notice in
// sky_clouds_noise.h.slang), in kilometres, with three changes for a bake. Takram's beer shadow map
// -- the optical depth to the sun beyond what a pixel's own short march reaches, rendered from the
// sun each frame -- is a second, longer march toward the sun here (farSunOpticalDepth): a bake can
// afford it. Their temporal blue noise becomes a fixed per-texel dither, and what their frames
// average over time the noise volumes' mips average here (noiseLod). And the air in front of the
// clouds is this renderer's own march (marchAerialPerspective), lit as the land under the clouds is.
//

#include "nvshaders/constants.h.slang"
#include "aerial_perspective.h.slang"  // marchAerialPerspective; brings the Bruneton functions
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"
#include "sky_bruneton_radiance.h.slang"  // brunetonHorizonMu
#include "sky_clouds_io.h.slang"
#include "sky_clouds.h.slang"

// clang-format off
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeOut, 0)]]        [[vk::image_format("rgba16f")]] RWTexture2D<float4> g_out;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeShape, 0)]]      Sampler3D<float>                   g_shape;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeDetail, 0)]]     Sampler3D<float>                   g_detail;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeWeather, 0)]]    Sampler2D<float4>                  g_weather;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeTurbulence, 0)]] Sampler2D<float4>                  g_turbulence;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeParams, 0)]]     ConstantBuffer<SkyCloudsParameters> g_params;

// Set 1 is SkyBruneton's runtime set, as in the environment bake.
[[vk::binding(0, 1)]] Sampler2D<float3>                       g_transmittance;
[[vk::binding(1, 1)]] Sampler3D<float4>                       g_scattering;
[[vk::binding(2, 1)]] Sampler3D<float4>                       g_singleMie;
[[vk::binding(3, 1)]] Sampler2D<float3>                       g_irradiance;
[[vk::binding(4, 1)]] ConstantBuffer<SkyAtmosphereParameters> g_atmosphere;

[[vk::push_constant]] ConstantBuffer<SkyCloudsPushConstant> pc;
// clang-format on

static const float kInvFourPi = 0.0795774715F;

// takram's mip level from distance (clouds.frag: rayStartTexelsPerPixel + rayDistance * 1e-5, in
// metres): 0 near, 1 at 100 km. It drops the detail noise and the short march toward the sun with
// distance, where a texel covers more than either resolves.
float mipLevelAt(float distance)
{
  return log2(max(1.0F, 1.0F + distance * 0.01F));
}

// The level of a noise volume to read at a distance: where one texel of the cloud image covers
// several of the volume's, the level that averages them. Takram's frames average them over time
// instead, each frame at another jitter; a bake is one frame, so without it the far clouds alias
// to grain. `texelKm` is one of the volume's texels, in km.
float noiseLod(float distance, float texelKm)
{
  return log2(max(1.0F, distance * pc.texelAngle / texelKm));
}

static const float kShapeTexelKm  = CLOUD_SHAPE_PERIOD / float(CLOUD_SHAPE_NOISE_SIZE);
static const float kDetailTexelKm = CLOUD_DETAIL_PERIOD / float(CLOUD_DETAIL_NOISE_SIZE);

float2 weatherUv(float3 position)
{
  return (float2(position.x, position.z) + g_params.weatherOffset) / CLOUD_WEATHER_PERIOD;
}

CloudWeather weatherAt(float3 position, float height, float mipLevel)
{
  return cloudSampleWeather(g_params, g_weather.SampleLevel(weatherUv(position), mipLevel), height);
}

// The medium at `position`, with the weather already read there (sampleMedia). The noise is laid on
// the observer's tangent plane horizontally and on altitude vertically, so the layers follow the
// planet's curve out to the horizon. `far` leaves out the turbulence and the detail: the long march
// toward the sun reads only the shape, as takram's shadow map does.
CloudMedia mediaAt(float3 position, float height, CloudWeather weather, float mipLevel, float distance, float jitter, bool far)
{
  float3 q = float3(position.x, height, position.z);
  if(g_params.turbulence != 0 && !far)
  {
    const float2 uv = weatherUv(position) * (CLOUD_WEATHER_PERIOD / CLOUD_TURBULENCE_PERIOD);
    q += (g_turbulence.SampleLevel(uv, 0.0F).xyz * 2.0F - 1.0F) * g_params.turbulenceDisplacement * cloudTurbulenceWeight(weather);
  }
  const float shape      = g_shape.SampleLevel(q / CLOUD_SHAPE_PERIOD + g_params.shapeOffset, noiseLod(distance, kShapeTexelKm));
  const bool  withDetail = !far && g_params.shapeDetail != 0 && mipLevel * 0.5F + (jitter - 0.5F) * 0.5F < 0.5F;
  const float detail     = withDetail ? g_detail.SampleLevel(q / CLOUD_DETAIL_PERIOD, noiseLod(distance, kDetailTexelKm)) : 0.0F;
  return cloudSampleMedia(g_params, weather, shape, detail, withDetail);
}

// The optical depth from `origin` along `dir`, a few short steps that lengthen (marchOpticalDepth):
// the shadow's edge, which the long march is too coarse for. Fewer steps with distance; with none,
// takram's fudge of half an optical depth. `rayDistance` is where it stopped.
float marchOpticalDepth(float3 origin, float3 dir, int maxIterations, float mipLevel, float distance, float jitter, out float rayDistance)
{
  rayDistance              = 0.0F;
  const int iterationCount = int(max(0.0F, lerp(float(maxIterations + 1), 1.0F, mipLevel) - jitter));
  if(iterationCount == 0)
    return 0.5F;
  float stepSize     = g_params.minSecondaryStepSize / float(iterationCount);
  float nextDistance = stepSize * jitter;
  float opticalDepth = 0.0F;
  for(int i = 0; i < iterationCount; ++i)
  {
    rayDistance                = nextDistance;
    const float3       p       = origin + dir * rayDistance;
    const float        height  = length(p) - g_atmosphere.bottomRadius;
    const CloudWeather weather = weatherAt(p, height, mipLevel);
    if(cloudWeatherHasDensity(g_params, weather))
      opticalDepth += mediaAt(p, height, weather, mipLevel, distance, jitter, false).extinction * stepSize;
    nextDistance += stepSize;
    stepSize *= g_params.secondaryStepScale;
  }
  return opticalDepth;
}

// The rest of the way to the sun, in steps doubling from 200 m to a few kilometres out: the other
// clouds, and the far side of this one, that shade the point. What takram's beer shadow map gives
// a frame; the shape alone, since it is what blocks the light at that range.
float farSunOpticalDepth(float3 origin, float3 dir, float start, float distance)
{
  float t            = start;
  float stepSize     = 0.2F;
  float opticalDepth = 0.0F;
  for(int i = 0; i < g_params.farSunIterations; ++i)
  {
    const float3 p      = origin + dir * (t + 0.5F * stepSize);
    const float  height = length(p) - g_atmosphere.bottomRadius;
    if(height > g_params.maxHeight)
      break;  // above every layer: nothing more toward the sun
    const CloudWeather weather = weatherAt(p, height, 1.0F);
    if(cloudWeatherHasDensity(g_params, weather))
      opticalDepth += mediaAt(p, height, weather, 1.0F, distance, 0.5F, true).extinction * stepSize;
    t += stepSize;
    stepSize *= 2.0F;
  }
  return opticalDepth;
}

// Light from the ground under the clouds, bounced up into them (approximateRadianceFromGround): the
// land lit by the sky and, where it gets through, the sun, attenuated through the cloud below.
float3 radianceFromGround(float3 position, float3 normal, float mipLevel, float distance, float jitter)
{
  float        unused;
  const float  depth  = marchOpticalDepth(position, -normal, g_params.groundIterations, mipLevel, distance, jitter, unused);
  const float3 ground = normal * g_atmosphere.bottomRadius;
  float3       skyIrradiance;
  const float3 sunIrradiance =
      GetSunAndSkyIrradiance(g_atmosphere, g_transmittance, g_irradiance, ground, normal, pc.sunDirection, skyIrradiance);
  const float3 irradiance = skyIrradiance * SKY_RADIANCE_TO_LUMINANCE + sunIrradiance * (SUN_RADIANCE_TO_LUMINANCE * pc.sunlitAir);
  return g_atmosphere.groundAlbedo * (1.0F / M_PI) * irradiance * exp(-depth);
}

// Jimenez's interleaved gradient noise: where each texel's march starts within its step, so the
// steps' edges dither instead of banding. The image is baked once, so the dither is frozen; the
// bilinear lookups that read it smooth it.
float stepJitter(uint2 pixel)
{
  return frac(52.9829189F * frac(0.06711056F * float(pixel.x) + 0.00583715F * float(pixel.y)));
}

float4 marchClouds(float r0, float3 dir, uint2 pixel)
{
  const float  R       = g_atmosphere.bottomRadius;
  const float2 segment = cloudLayerSegment(r0, dir.y, R + g_params.minHeight, R + g_params.maxHeight);
  const float  rayNear = segment.x;
  const float  rayFar  = min(segment.y, g_params.maxRayDistance);
  if(rayFar <= rayNear)
    return float4(0.0F, 0.0F, 0.0F, 1.0F);

  const float3 observer = float3(0.0F, r0, 0.0F);
  const float3 origin   = observer + dir * rayNear;
  const float3 sunDir   = pc.sunDirection;
  const float  cosTheta = dot(dir, sunDir);
  const float  jitter   = stepJitter(pixel);
  const float  span     = rayFar - rayNear;

  float3 radianceIntegral      = float3(0.0F);
  float  transmittanceIntegral = 1.0F;
  float  weightedDistanceSum   = 0.0F;
  float  transmittanceSum      = 0.0F;

  float stepSize    = g_params.minStepSize + (g_params.perspectiveStepScale - 1.0F) * rayNear;
  float rayDistance = stepSize * jitter * 2.0F;  // takram doubles the jitter against spatial aliasing

  for(int i = 0; i < g_params.maxIterations; ++i)
  {
    if(rayDistance > span)
      break;

    const float3 position = origin + dir * rayDistance;
    const float  height   = length(position) - R;
    const float  distance = rayNear + rayDistance;
    const float  mipLevel = mipLevelAt(distance);

    // Between layers, and where the weather allows no cloud, step longer.
    if(cloudInsideIntervals(g_params, height))
    {
      stepSize *= g_params.perspectiveStepScale;
      rayDistance += lerp(stepSize, g_params.maxStepSize, min(1.0F, mipLevel));
      continue;
    }
    const CloudWeather weather = weatherAt(position, height, mipLevel);
    if(!cloudWeatherHasDensity(g_params, weather))
    {
      stepSize *= g_params.perspectiveStepScale;
      rayDistance += lerp(stepSize, g_params.maxStepSize, min(1.0F, mipLevel));
      continue;
    }

    const CloudMedia media = mediaAt(position, height, weather, mipLevel, distance, jitter, false);
    if(media.extinction > g_params.minExtinction)
    {
      const float  r      = length(position);
      const float  muS    = dot(position, sunDir) / r;
      const float3 normal = position / r;

      // Scalar irradiance at the point (takram's GetSunAndSkyScalarIrradiance): the sun after the
      // air above, no cosine, and the sky's from all round. Each in its own spectrum's units.
      const float3 sunIrradiance =
          g_atmosphere.solarIrradiance * GetTransmittanceToSun(g_atmosphere, g_transmittance, r, muS) * SUN_RADIANCE_TO_LUMINANCE;
      const float3 skyIrradiance = GetIrradiance(g_atmosphere, g_irradiance, r, muS) * (2.0F * M_PI) * SKY_RADIANCE_TO_LUMINANCE;

      float3 radiance = float3(0.0F);
      if(max(sunIrradiance.x, max(sunIrradiance.y, sunIrradiance.z)) > 0.0F)
      {
        float       sunRayDistance;
        const float nearDepth = marchOpticalDepth(position, sunDir, g_params.sunIterations, mipLevel, distance, jitter, sunRayDistance);
        const float farDepth = farSunOpticalDepth(position, sunDir, max(sunRayDistance, g_params.minSecondaryStepSize), distance);
        radiance              = sunIrradiance * cloudMultipleScattering(nearDepth + farDepth, cosTheta, g_params.octaves);
      }

      if(g_params.groundIterations > 0 && height < g_params.shadowTopHeight && mipLevel < 0.5F)
        radiance += radianceFromGround(position, normal, mipLevel, distance, jitter) * kInvFourPi * g_params.groundBounceScale;

      // takram's crude sky gradient: brighter toward each layer's top. Better than none in the shadows.
      const float skyGradient = dot(weather.heightFraction * 0.5F + 0.5F, media.weight);
      radiance += skyIrradiance * kInvFourPi * skyGradient * g_params.skyLightScale;

      radiance *= media.scattering;
      // Beer-Powder: thin edges darker, where light has not yet built up by scattering
      radiance *= 1.0F - g_params.powderScale * exp(-media.extinction * g_params.powderExponent);

      // Energy-conserving integration over the step (Hillaire 2016, 5.6.3)
      const float transmittance = exp(-media.extinction * stepSize);
      radianceIntegral += transmittanceIntegral * (radiance - radiance * transmittance) / max(media.extinction, 1e-7F);
      transmittanceIntegral *= transmittance;

      // Where the clouds are, on average, for the air in front of them (5.9.1)
      weightedDistanceSum += distance * transmittanceIntegral;
      transmittanceSum += transmittanceIntegral;
    }

    if(transmittanceIntegral <= g_params.minTransmittance)
      break;

    // Shorter steps once inside the clouds
    stepSize *= g_params.perspectiveStepScale;
    rayDistance += stepSize;
  }

  // takram reports what is left as opacity rescaled to reach 1 at minTransmittance; the image keeps
  // its complement.
  const float transmittance =
      clamp((transmittanceIntegral - g_params.minTransmittance) / (1.0F - g_params.minTransmittance), 0.0F, 1.0F);
  if(transmittance >= 1.0F)
    return float4(0.0F, 0.0F, 0.0F, 1.0F);

  // The air between the observer and the clouds, to where they are on average. Its sunlit share is
  // the share of the land under the clouds the sun reaches (cloudSunlitAir): under a deck the haze
  // is grey, not the blue of the clear sky's.
  const float frontDepth = transmittanceSum > 0.0F ? weightedDistanceSum / transmittanceSum : rayNear;
  const AerialMarch air =
      marchAerialPerspective(g_atmosphere, g_transmittance, g_irradiance, observer, dir * frontDepth, sunDir, float2(0.5F));
  const float3 airInFront = (air.inScatter - air.sunInScatter * (1.0F - pc.sunlitAir)) * SKY_RADIANCE_TO_LUMINANCE;
  return cloudTexel(radianceIntegral, transmittance, air.transmittance, airInFront);
}

[shader("compute")]
[numthreads(CLOUD_WORKGROUP_SIZE, CLOUD_WORKGROUP_SIZE, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
  if(any(id.xy >= pc.imageSize))
    return;

  // The observer and its horizon are the environment bake's (publishSunDisk in env_bake.slang).
  const float r0        = g_atmosphere.bottomRadius + pc.observerAltitude;
  const float horizonMu = brunetonHorizonMu(g_atmosphere, r0);

  const float2 uv  = (float2(id.xy) + 0.5F) / float2(pc.imageSize);
  const float3 dir = cloudImageDir(uv, asin(horizonMu));

  float4 texel = float4(0.0F, 0.0F, 0.0F, 1.0F);
  if(g_params.coverage > 0.0F && !RayIntersectsGround(g_atmosphere, r0, dir.y))
    texel = marchClouds(r0, dir, id.xy);
  g_out[id.xy] = texel;
}
