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
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeFiltered, 0)]]   [[vk::image_format("rgba16f")]] RWTexture2D<float4> g_filtered;

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

// The land's variety at a point (cloudVariety): the turbulence map read at the three coarse tilings.
CloudVariety varietyAt(float3 position)
{
  const float2 xz = float2(position.x, position.z) + g_params.weatherOffset;
  return cloudVariety(g_turbulence.SampleLevel(cloudSystemUv(xz), 0.0F), g_turbulence.SampleLevel(cloudRegionUv(xz), 0.0F),
                      g_turbulence.SampleLevel(cloudLocalUv(xz), 0.0F));
}

// The weather at a point, read where the land's variety bends the map to, and moved by it.
CloudWeather weatherAt(float3 position, float height, float mipLevel)
{
  const CloudVariety v  = varietyAt(position);
  const float2       uv = weatherUv(position) + v.weatherWarp / CLOUD_WEATHER_PERIOD;
  return cloudSampleVariedWeather(g_params, g_weather.SampleLevel(uv, mipLevel), height, v);
}

// The two volumes at `q` for a layer, at its scale and its stretch along x: (shape, detail), the
// detail 0 where it is left out. The level is the finer axis's, so a streak keeps its edges.
float2 noiseAt(float3 q, float scale, float stretch, float distance, bool withDetail)
{
  const float3 f      = float3(scale * stretch, scale, scale);
  const float  shape  = g_shape.SampleLevel(q * f / CLOUD_SHAPE_PERIOD + g_params.shapeOffset, noiseLod(distance, kShapeTexelKm / scale));
  const float  detail = withDetail ? g_detail.SampleLevel(q * f / CLOUD_DETAIL_PERIOD, noiseLod(distance, kDetailTexelKm / scale)) : 0.0F;
  return float2(shape, detail);
}

// The medium at `position`, with the weather already read there (sampleMedia). The noise is laid on
// the observer's tangent plane horizontally and on altitude vertically, so the layers follow the
// planet's curve out to the horizon. `far` leaves out the turbulence and the detail: the long march
// toward the sun reads only the shape, as takram's shadow map does.
CloudMedia mediaAt(float3 position, float height, CloudWeather weather, float mipLevel, float distance, float jitter, bool far)
{
  // the billows read where the land's variety bends them to (cloudVariety)
  float3 q = float3(position.x + weather.variety.z, height, position.z + weather.variety.w);
  if(g_params.turbulence != 0 && !far)
  {
    const float2 uv = weatherUv(position) * (CLOUD_WEATHER_PERIOD / CLOUD_TURBULENCE_PERIOD);
    q += (g_turbulence.SampleLevel(uv, 0.0F).xyz * 2.0F - 1.0F) * cloudTurbulenceDisplacement(g_params, weather);
  }
  // Each tier's layers at their own scale and stretch (shapeScales, shapeStretches), read only where
  // they have cloud: the tiers lie at different heights, so a point reads the volumes once almost
  // everywhere. The low tier's two layers share the first's. A texel of a volume covers less of the
  // sky the finer it is read.
  const bool withDetail = !far && g_params.shapeDetail != 0 && mipLevel * 0.5F + (jitter - 0.5F) * 0.5F < 0.5F;
  float4     shape      = float4(0.0F);
  float4     detail     = float4(0.0F);
  const float minDensity = g_params.minDensity;
  if(weather.density.x > minDensity || weather.density.y > minDensity)
  {
    const float2 v = noiseAt(q, g_params.shapeScales.x, g_params.shapeStretches.x, distance, withDetail);
    shape.xy       = float2(v.x);
    detail.xy      = float2(v.y);
  }
  if(weather.density.z > minDensity)
  {
    const float2 v = noiseAt(q, g_params.shapeScales.z, g_params.shapeStretches.z, distance, withDetail);
    shape.z        = v.x;
    detail.z       = v.y;
  }
  if(weather.density.w > minDensity)
  {
    const float2 v = noiseAt(q, g_params.shapeScales.w, g_params.shapeStretches.w, distance, withDetail);
    shape.w        = v.x;
    detail.w       = v.y;
  }
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

// The far march's longest step, and the optical depth past which no octave of the light is left
// worth carrying (the last of eight sees 1/128 of it, and its share is 1/128).
static const float kFarSunMaxStep  = 2.0F;
static const float kFarSunOpaque   = 200.0F;

// The rest of the way to the sun, in steps doubling from 200 m to kFarSunMaxStep and on at that
// until the ray leaves the layers: the other clouds, and the far side of this one, that shade the
// point. What takram's beer shadow map gives a frame; the shape alone, since it is what blocks the
// light at that range. A high sun leaves the layers within a few steps. A low one runs through them
// for tens of km, and a march that stopped a few km out let a sunset's light into every cloud, the
// clouds between them and the sun casting none: their bodies took the sun's orange where the sky's
// cooler light, all that reaches them, should be.
float farSunOpticalDepth(float3 origin, float3 dir, float start, float distance)
{
  float t            = start;
  float stepSize     = 0.2F;
  float opticalDepth = 0.0F;
  for(int i = 0; i < g_params.farSunIterations && opticalDepth < kFarSunOpaque; ++i)
  {
    const float3 p      = origin + dir * (t + 0.5F * stepSize);
    const float  height = length(p) - g_atmosphere.bottomRadius;
    if(height > g_params.maxHeight)
      break;  // above every layer: nothing more toward the sun
    if(!cloudInsideIntervals(g_params, height))
    {
      const CloudWeather weather = weatherAt(p, height, 1.0F);
      if(cloudWeatherHasDensity(g_params, weather))
        opticalDepth += mediaAt(p, height, weather, 1.0F, distance, 0.5F, true).extinction * stepSize;
    }
    t += stepSize;
    stepSize = min(stepSize * 2.0F, kFarSunMaxStep);
  }
  return opticalDepth;
}

// How much of the sky's light reaches a point in the clouds: through the cloud above it, as diffuse
// light gets through a slab -- two-stream, 1 / (1 + 3/4 (1 - g) tau), g 0.85 a droplet's -- over the
// optical depth up to the layers' top in steps doubling from 100 m. Takram's in its place is a
// gradient, a layer's base lit half as well as its top whatever lies above it: under a deck that
// lit the whole underside alike, as flat as a painted ceiling, where a real one's thick parts go dark
// and its thin ones let the light through. The shape alone, as the far march toward the sun reads.
static const float kSkyDiffusion = 0.75F * (1.0F - 0.85F);

float skyVisibility(float3 position, float3 up, float distance)
{
  float t            = 0.0F;
  float stepSize     = 0.1F;
  float opticalDepth = 0.0F;
  for(int i = 0; i < g_params.skyIterations; ++i)
  {
    const float3 p      = position + up * (t + 0.5F * stepSize);
    const float  height = length(p) - g_atmosphere.bottomRadius;
    if(height > g_params.maxHeight)
      break;
    if(!cloudInsideIntervals(g_params, height))
    {
      const CloudWeather weather = weatherAt(p, height, 1.0F);
      if(cloudWeatherHasDensity(g_params, weather))
        opticalDepth += mediaAt(p, height, weather, 1.0F, distance, 0.5F, true).extinction * stepSize;
    }
    t += stepSize;
    stepSize *= 2.0F;
  }
  return 1.0F / (1.0F + kSkyDiffusion * opticalDepth);
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
// steps' edges dither instead of banding. The same value picks how far the short marches go and
// whether the detail noise is read, which at the clouds' edges flips texel by texel. Takram's
// frames each take another and average over time; here `sampleIndex` walks it along the golden
// ratio, and main averages the samples in the texel.
// Where in the raw image a texel's march goes. The sky image's, and the preview's, at its own place; a
// view bake's (pc.viewForward.w), whose rows are another length, one after another through the raw
// image's, which holds as many texels as the view image's capacity.
uint2 rawTexel(uint2 texel)
{
  if(pc.viewForward.w == 0.0F)
    return texel;
  const uint i = texel.y * pc.imageSize.x + texel.x;
  return uint2(i % CLOUD_RAW_WIDTH, i / CLOUD_RAW_WIDTH);
}

float stepJitter(uint2 pixel, int sampleIndex)
{
  const float ign = frac(52.9829189F * frac(0.06711056F * float(pixel.x) + 0.00583715F * float(pixel.y)));
  return frac(ign + 0.618033989F * float(sampleIndex));
}

float4 marchClouds(float r0, float3 dir, float jitter)
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

      // The sky's light, through the cloud above the point (skyVisibility)
      radiance += skyIrradiance * kInvFourPi * skyVisibility(position, normal, distance) * g_params.skyLightScale;

      radiance *= media.scattering;
      // Beer-Powder: thin edges darker, where light has not yet built up by scattering
      radiance *= 1.0F - dot(media.weight, g_params.powderScales) * exp(-media.extinction * g_params.powderExponent);

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

  // The samples spread over the texel's footprint (Roberts' R2 sequence) as well as over the jitter,
  // so the texel is the clouds averaged over it: edges smooth, not stepped, and the far clouds,
  // smaller than a texel, are not one point of them. The texel composes linearly (composeClouds),
  // so averaging texels averages what they draw. A sample below the horizon sees no cloud.
  // A view bake maps the texel through the camera's view instead (cloudViewDir).
  const float horizonElevation = asin(horizonMu);
  const int   samples          = max(1, g_params.samples);
  const bool  anyTier          = any(g_params.densityScales > float4(0.0F));
  const bool  view             = pc.viewForward.w != 0.0F;
  float4      texel            = float4(0.0F);
  for(int s = 0; s < samples; ++s)
  {
    const float2 offset = samples == 1 ? float2(0.5F) : frac(float2(0.5F) + float(s) * float2(0.754877666F, 0.569840291F));
    const float2 uv     = (float2(id.xy) + offset) / float2(pc.imageSize);
    const float3 dir = view ? cloudViewDir(uv, pc.viewRight.xyz, pc.viewRight.w, pc.viewUp.xyz, pc.viewUp.w, pc.viewForward.xyz) : cloudImageDir(uv, horizonElevation);
    if(anyTier && !RayIntersectsGround(g_atmosphere, r0, dir.y))
      texel += marchClouds(r0, dir, stepJitter(id.xy, s));
    else
      texel += float4(0.0F, 0.0F, 0.0F, 1.0F);
  }
  g_out[rawTexel(id.xy)] = texel / float(samples);
}

// How sharply the filter keeps neighbours of another transparency out: one 0.15 apart counts a third.
static const float kFilterAlphaSharpness = 50.0F;

// The settled bake's march (main) into the raw image, then this into the cloud image: each texel
// averaged with its eight neighbours, a tent's weights, around the horizon and clamped at its ends.
// Eight marches a texel leave noise where a march's light varies most with its jitter -- the shadowed
// bodies of clouds under a low sun, read through many octaves of the light that got in -- and as
// the jitter is a texel's interleaved gradient noise, the noise wears that noise's weave. Any 3x3
// block of it holds the jitter's whole range, so the neighbours' marches are the ones this texel's
// lacked: the weave averages out, as takram's frames average it over time. A neighbour counts less
// the further its transparency is from the texel's own (kFilterAlphaSharpness), so the clouds'
// edges against the sky stay a texel sharp; inside a cloud, where the transparency is the same, the
// filter blurs a texel's light by a little under one. The texel composes linearly (composeClouds),
// so the average of texels is what they draw. A view bake's image ends at its sides as well.
[shader("compute")]
[numthreads(CLOUD_WORKGROUP_SIZE, CLOUD_WORKGROUP_SIZE, 1)]
void filterMain(uint3 id: SV_DispatchThreadID)
{
  if(any(id.xy >= pc.imageSize))
    return;
  const int2   size   = int2(pc.imageSize);
  const bool   view   = pc.viewForward.w != 0.0F;
  const float4 centre = g_out[rawTexel(id.xy)];
  float4       sum    = float4(0.0F);
  float        weight = 0.0F;
  for(int dy = -1; dy <= 1; ++dy)
  {
    for(int dx = -1; dx <= 1; ++dx)
    {
      const int    x  = int(id.x) + dx;
      const int2   q  = int2(view ? clamp(x, 0, size.x - 1) : (x + size.x) % size.x, clamp(int(id.y) + dy, 0, size.y - 1));
      const float4 v  = g_out[rawTexel(uint2(q))];
      const float  da = v.w - centre.w;
      const float  w  = float((2 - abs(dx)) * (2 - abs(dy))) * exp(-da * da * kFilterAlphaSharpness);
      sum += v * w;
      weight += w;
    }
  }
  g_filtered[id.xy] = sum / weight;
}
