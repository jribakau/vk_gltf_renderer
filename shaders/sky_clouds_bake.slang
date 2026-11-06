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
// through the cloud layer, lit by the sun through the atmosphere's own transmittance and by the sky's
// irradiance, then the air between the observer and the clouds. Recorded by SkyClouds::record() just
// before the environment bake, which composes it over the sky (env_bake.slang), as the path tracer
// does for what the camera sees (gltf_pathtrace.slang). What a texel holds, and why, is
// sky_clouds.h.slang.
//

#include "nvshaders/constants.h.slang"
#include "aerial_perspective.h.slang"  // marchAerialPerspective; brings the Bruneton functions
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"
#include "sky_bruneton_radiance.h.slang"  // brunetonHorizonMu
#include "sky_clouds.h.slang"
#include "sky_clouds_io.h.slang"

// clang-format off
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeOut, 0)]]     [[vk::image_format("rgba16f")]] RWTexture2D<float4> g_out;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeShape, 0)]]   Sampler3D<float>    g_shape;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeDetail, 0)]]  Sampler3D<float>    g_detail;
[[vk::binding(SkyCloudsBakeBindings::eCloudBakeWeather, 0)]] Sampler2D<float>    g_weather;

// Set 1 is SkyBruneton's runtime set, as in the environment bake.
[[vk::binding(0, 1)]] Sampler2D<float3>                       g_transmittance;
[[vk::binding(1, 1)]] Sampler3D<float4>                       g_scattering;
[[vk::binding(2, 1)]] Sampler3D<float4>                       g_singleMie;
[[vk::binding(3, 1)]] Sampler2D<float3>                       g_irradiance;
[[vk::binding(4, 1)]] ConstantBuffer<SkyAtmosphereParameters> g_atmosphere;

[[vk::push_constant]] ConstantBuffer<SkyCloudsPushConstant> pc;
// clang-format on

// Density at `p` (planet-centred, km) and how far up the layer it is. The noise is laid on the
// observer's tangent plane horizontally and on altitude vertically, so the layer follows the
// planet's curve out to the horizon. `withDetail` false skips the erosion, for the light march:
// its mean stands in for it, and the light it lets through is what matters there, not the edges.
float densityAt(float3 p, bool withDetail, out float h)
{
  const float r        = length(p);
  const float altitude = r - g_atmosphere.bottomRadius;
  h                    = (altitude - pc.bottom) / pc.thickness;
  if(h <= 0.0F || h >= 1.0F)
    return 0.0F;

  const float2 xz      = float2(p.x, p.z) + pc.offset;
  const float  weather = cloudWeather(g_weather.SampleLevel(xz / CLOUD_WEATHER_PERIOD, 0.0F),
                                      g_weather.SampleLevel(xz / CLOUD_WEATHER_PERIOD2 + float2(0.37F, 0.61F), 0.0F));
  if(cloudLocalCoverage(weather, pc.level) <= 0.0F)
    return 0.0F;  // the weather allows none here: no need to read the volumes

  const float3 q      = float3(xz.x, altitude, xz.y);
  const float  shape  = g_shape.SampleLevel(q / CLOUD_SHAPE_PERIOD, 0.0F);
  const float  detail = withDetail ? g_detail.SampleLevel(q / CLOUD_DETAIL_PERIOD, 0.0F) : 0.5F;
  return cloudDensity(shape, detail, weather, h, pc.level);
}

// The cloud's optical depth from `p` toward the sun, in steps that lengthen away from the point,
// where the shadow's edge needs them least: 60 m first, about 1.6 km in all at six.
float sunOpticalDepth(float3 p, float3 sunDir)
{
  float depth = 0.0F;
  float t     = 0.0F;
  float step  = 0.06F;
  for(int i = 0; i < pc.lightSteps; ++i)
  {
    float      h;
    const float d = densityAt(p + sunDir * (t + 0.5F * step), false, h);
    depth += d * CLOUD_EXTINCTION * step;
    t += step;
    step *= 1.6F;
  }
  return depth;
}

// Jimenez's interleaved gradient noise: where in its step each texel's march samples, so that the
// steps' edges dither instead of banding. The image is baked once, so the dither is frozen; the
// bilinear lookups that read it smooth it.
float stepJitter(uint2 pixel)
{
  return frac(52.9829189F * frac(0.06711056F * float(pixel.x) + 0.00583715F * float(pixel.y)));
}

float4 marchClouds(float r0, float3 dir, uint2 pixel)
{
  const float rBottom = g_atmosphere.bottomRadius + pc.bottom;
  float2      segment = cloudLayerSegment(r0, dir.y, rBottom, rBottom + pc.thickness);
  segment.y           = min(segment.y, CLOUD_MAX_DISTANCE);
  if(segment.y <= segment.x)
    return float4(0.0F, 0.0F, 0.0F, 1.0F);

  const float3 observer = float3(0.0F, r0, 0.0F);
  const float3 sunDir   = pc.sunDirection;
  const float  cosTheta = dot(dir, sunDir);
  const float  dt       = (segment.y - segment.x) / float(pc.viewSteps);
  const float  jitter   = stepJitter(pixel);

  float  transmittance = 1.0F;
  float3 light         = float3(0.0F);
  float  depthSum      = 0.0F;  // distance weighted by what each step hides, for the air in front
  float  hiddenSum     = 0.0F;
  for(int i = 0; i < pc.viewSteps; ++i)
  {
    const float  t = segment.x + (float(i) + jitter) * dt;
    const float3 p = observer + dir * t;
    float        h;
    const float  density = densityAt(p, true, h);
    if(density <= 0.0F)
      continue;

    const float r   = length(p);
    const float muS = dot(p, sunDir) / r;
    // Sunlight at the point after the air above it -- reddened and dimmed at a low sun, gone in the
    // planet's shadow -- and the sky's light on it from above. Each in its own spectrum's units, as
    // the ground's are (sky_bruneton_radiance.h.slang).
    const float3 sunIrradiance =
        g_atmosphere.solarIrradiance * GetTransmittanceToSun(g_atmosphere, g_transmittance, r, muS) * SUN_RADIANCE_TO_LUMINANCE;
    const float3 skyIrradiance = GetIrradiance(g_atmosphere, g_irradiance, r, muS) * SKY_RADIANCE_TO_LUMINANCE;
    const float  sunDepth      = max(sunIrradiance.x, max(sunIrradiance.y, sunIrradiance.z)) > 0.0F ? sunOpticalDepth(p, sunDir) : 0.0F;

    // Integrated over the step with the light held constant across it: what the step hides of what
    // is behind it, times what it sends itself. Exact for a thick step as for a thin one, so a
    // distant cloud crossed in one step is as bright as a near one crossed in twenty.
    const float stepTransmittance = exp(-density * CLOUD_EXTINCTION * dt);
    const float hidden            = transmittance * (1.0F - stepTransmittance);
    light += hidden * cloudLight(sunIrradiance, sunDepth, cosTheta, skyIrradiance, h);
    depthSum += hidden * t;
    hiddenSum += hidden;
    transmittance *= stepTransmittance;
    if(transmittance < 0.003F)
    {
      transmittance = 0.0F;
      break;
    }
  }
  if(hiddenSum <= 0.0F)
    return float4(0.0F, 0.0F, 0.0F, 1.0F);

  // The air between the observer and the clouds, to where they are on average. Its sunlit share is
  // the share of the land under the layer the sun reaches (cloudSunlitAir): under a deck the haze is
  // grey, not the blue of the clear sky's.
  const AerialMarch air = marchAerialPerspective(g_atmosphere, g_transmittance, g_irradiance, observer,
                                                 dir * (depthSum / hiddenSum), sunDir, float2(0.5F));
  const float3 airInFront = (air.inScatter - air.sunInScatter * (1.0F - pc.sunlitAir)) * SKY_RADIANCE_TO_LUMINANCE;
  return cloudTexel(light, transmittance, air.transmittance, airInFront);
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
  if(pc.level > 0.0F && !RayIntersectsGround(g_atmosphere, r0, dir.y))
    texel = marchClouds(r0, dir, id.xy);
  g_out[id.xy] = texel;
}
