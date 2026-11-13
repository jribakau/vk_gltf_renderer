/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The rain pass (src/rain.hpp): the drops in the air, each a streak its fall makes over the shutter,
// drawn over the image the tone mapper takes. A drop shows the world round it, mostly what lies
// behind it, as a wide lens would (Garg and Nayar 2006): its light is the environment's mean round the
// way the eye looks through it, and the sun's where it is backlit (rainSunPhase). How much of a pixel
// a streak covers is the time the drop spends over it: its diameter over the streak's length.

#include "rain_io.h.slang"
#include "rain.h.slang"
#include "nvshaders/functions.h.slang"  // getSphericalUv, quatRotateInverse

// clang-format off
[[vk::binding(BindingPoints::eTexturesHdr, 0)]] Sampler2D texturesHdr[];
[[vk::binding(RainBindings::eRainIn, 1)]]       Sampler2D<float4>   u_in;
[[vk::binding(RainBindings::eRainOut, 1)]]      RWTexture2D<float4> u_out;
[[vk::binding(RainBindings::eRainDepth, 1)]]    Sampler2D<float>    u_depth;

[[vk::push_constant]] ConstantBuffer<RainPush> pc;
// clang-format on

// ---- the light table -----------------------------------------------------------------------------

// The direction at the middle of the table's texel `uv`, as getSphericalUv maps it
float3 rainLightDirection(float2 uv)
{
  const float theta = (uv.x - 0.5F) * 6.28318531F;
  const float gamma = (uv.y - 0.5F) * 3.14159265F;
  return float3(cos(gamma) * cos(theta), -sin(gamma), cos(gamma) * sin(theta));
}

// The cosine-weighted mean of the environment round each of the table's directions: what a drop seen
// that way shows of it, before the share it lets through. A workgroup a texel, a direction a thread
// (128, a Fibonacci spiral), read from the environment's mip whose texels are a few degrees across.
groupshared float3 s_lightSum[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void rainLight(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
  const uint texel = group.x;
  if(texel >= RAIN_LIGHT_WIDTH * RAIN_LIGHT_HEIGHT)
    return;
  RainFrame*      f  = pc.frame;
  SceneFrameInfo* fi = f->frameInfo;
  const float2    uv = (float2(float(texel % RAIN_LIGHT_WIDTH), float(texel / RAIN_LIGHT_WIDTH)) + 0.5F) / float2(RAIN_LIGHT_WIDTH, RAIN_LIGHT_HEIGHT);
  const float3    n  = rainLightDirection(uv);
  const float3    t  = normalize(abs(n.y) < 0.99F ? cross(float3(0.0F, 1.0F, 0.0F), n) : cross(float3(1.0F, 0.0F, 0.0F), n));
  const float3    b  = cross(n, t);
  uint            width, height, levels;
  texturesHdr[HDR_IMAGE_INDEX].GetDimensions(0, width, height, levels);
  const float mip = clamp(log2(float(max(width, 1u)) / 128.0F), 0.0F, float(max(levels, 1u) - 1u));
  // a Fibonacci spiral over the disk, lifted onto the hemisphere: cosine-weighted
  const uint   k   = thread.x;
  const float  r   = sqrt((float(k) + 0.5F) / 128.0F);
  const float  phi = float(k) * 2.39996323F;
  const float3 dir = t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0F - r * r, 0.0F));
  s_lightSum[k]    = texturesHdr[HDR_IMAGE_INDEX].SampleLevel(getSphericalUv(quatRotateInverse(fi->envRotation, dir)), mip).rgb;
  GroupMemoryBarrierWithGroupSync();
  for(uint stride = 64u; stride > 0u; stride >>= 1u)
  {
    if(k < stride)
      s_lightSum[k] += s_lightSum[k + stride];
    GroupMemoryBarrierWithGroupSync();
  }
  if(k == 0u)
    f->light[texel] = float4(s_lightSum[0] * (fi->envIntensity / 128.0F), 0.0F);
}

// The table at direction `dir`, bilinear (round in longitude)
float3 rainLightAt(RainFrame* f, float3 dir)
{
  const float2 uv = getSphericalUv(dir) * float2(RAIN_LIGHT_WIDTH, RAIN_LIGHT_HEIGHT) - 0.5F;
  const int2   i  = int2(floor(uv));
  const float2 w  = uv - float2(i);
  const int    y0 = clamp(i.y, 0, RAIN_LIGHT_HEIGHT - 1);
  const int    y1 = clamp(i.y + 1, 0, RAIN_LIGHT_HEIGHT - 1);
  const int    x0 = (i.x + RAIN_LIGHT_WIDTH) % RAIN_LIGHT_WIDTH;
  const int    x1 = (i.x + 1 + RAIN_LIGHT_WIDTH) % RAIN_LIGHT_WIDTH;
  const float3 a  = lerp(f->light[y0 * RAIN_LIGHT_WIDTH + x0].rgb, f->light[y0 * RAIN_LIGHT_WIDTH + x1].rgb, w.x);
  const float3 c  = lerp(f->light[y1 * RAIN_LIGHT_WIDTH + x0].rgb, f->light[y1 * RAIN_LIGHT_WIDTH + x1].rgb, w.x);
  return lerp(a, c, w.y);
}

// ---- the image -----------------------------------------------------------------------------------

// The image as it comes, into the one the drops are drawn over
[shader("compute")]
[numthreads(RAIN_WORKGROUP_SIZE, RAIN_WORKGROUP_SIZE, 1)]
void rainCopy(uint3 id: SV_DispatchThreadID)
{
  const int2 p = int2(id.xy);
  if(p.x >= pc.size.x || p.y >= pc.size.y)
    return;
  u_out[p] = u_in.Load(int3(p, 0));
}

// ---- the drops -----------------------------------------------------------------------------------

// Each of a cell's slots once (a workgroup of RAIN_CULL_GROUP slots of a cell): the drop there if it
// has one, it falls here (no shelter over it), it is within reach and on screen; lit, and counted into
// the draw. A faint drop (a far one, under a pixel) is drawn at a chance of its coverage over
// minCoverage and that much stronger, which keeps the rain's look and draws a fraction of them.
[shader("compute")]
[numthreads(RAIN_CULL_GROUP, 1, 1)]
void rainCull(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
  RainFrame* f    = pc.frame;
  const uint slot = group.x * RAIN_CULL_GROUP + thread.x;
  if(slot >= f->slots)
    return;
  const int4     cell = f->cells[group.y];
  const RainDrop d    = rainDropAt(cell.xyz, slot, f->cellSize, f->shear, f->time, f->chance, f->seed, f->rate);
  if(d.present == 0)
    return;
  // under a roof, a tree or a figure: no rain falls there
  const int texel = rainShelterTexel(d.position, f->shear, f->shelterCentre);
  if(texel >= 0 && d.position.y < f->shelter[texel])
    return;
  const float distance = length(d.position - f->camera);
  if(distance < f->minDistance || distance > f->maxDistance)
    return;

  const float3 velocity = float3(f->wind.x, -d.speed, f->wind.y);
  const float3 p1       = d.position;
  const float3 p0       = p1 - velocity * f->shutter;
  const float4 c0       = mul(float4(p0, 1.0F), f->viewProj);
  const float4 c1       = mul(float4(p1, 1.0F), f->viewProj);
  if(c0.w < 0.5F * f->minDistance || c1.w < 0.5F * f->minDistance)
    return;
  // off screen: both ends past the same edge
  if((c0.x < -c0.w && c1.x < -c1.w) || (c0.x > c0.w && c1.x > c1.w) || (c0.y < -c0.w && c1.y < -c1.w) || (c0.y > c0.w && c1.y > c1.w)
     || (c0.z > c0.w && c1.z > c1.w))
    return;

  // its width at either end, pixels, and the time it spends over a point of its streak
  const float dropM  = d.diameter * 1e-3F;
  const float width0 = dropM * f->focal / c0.w;
  const float width1 = dropM * f->focal / c1.w;
  float       along  = dropM / (length(velocity) * f->shutter + dropM);
  // a faint one at a chance, stronger for it
  const float coverage = along * min(max(width0, width1), 1.0F);
  if(coverage < f->minCoverage)
  {
    const uint key = rainHash((uint(cell.x) * 73856093U) ^ (uint(cell.y) * 19349663U) ^ (uint(cell.z) * 83492791U) ^ 0x5bd1e995U) + slot * 0x9E3779B9U;
    const float chance = coverage / f->minCoverage;
    if(rainUnit(key) >= chance)
      return;
    along /= chance;
  }

  // its light: the environment round the way through it (most of it lets through, a little it
  // reflects from in front), and the sun's where it shines through
  SceneFrameInfo* fi   = f->frameInfo;
  const float3    view = normalize(p1 - f->camera);
  float3          lit  = 0.9F * rainLightAt(f, view) + 0.06F * rainLightAt(f, -view);
  if(fi->sunDisk != nullptr)
  {
    const SkySunDisk disk = *fi->sunDisk;
    const float3     sun  = disk.radiance * (6.28318531F * (1.0F - cos(disk.angularRadius))) * (fi->envIntensity * fi->skySunScale * disk.cloudShadow);
    lit += sun * rainSunPhase(dot(fi->sunDirection, view));
  }

  uint at = 0u;
  InterlockedAdd(f->drawArgs[1], 1u, at);
  if(at >= RAIN_MAX_STREAKS)
    return;
  const float2 half_ = 0.5F * f->viewport;
  RainStreak   s;
  s.ends          = float4(c0.xy / c0.w * half_, c1.xy / c1.w * half_);
  s.depth         = float4(c0.z, c0.w, c1.z, c1.w);
  s.light         = float4(lit * fi->outputScale, along);
  s.width         = float4(width0, width1, 0.0F, 0.0F);
  f->streaks[at] = s;
}

struct RainVertexOut
{
  float4                 position : SV_Position;
  noperspective float    across : TEXCOORD0;    // pixels from the streak's axis
  noperspective float    halfCore : TEXCOORD1;  // half the streak's drawn width, at least half a pixel
  nointerpolation float3 light : TEXCOORD2;     // the drop's light, pre-exposed
  noperspective float    coverage : TEXCOORD3;  // the share of a pixel on its axis it covers over the shutter
};

RainVertexOut rainCulled()
{
  RainVertexOut o;
  o.position = float4(2.0F, 2.0F, 2.0F, 1.0F);  // past the far plane: nothing is drawn
  o.across   = 0.0F;
  o.halfCore = 0.0F;
  o.light    = float3(0.0F);
  o.coverage = 0.0F;
  return o;
}

// Six corners a drop rainCull kept: its streak from where it was when the shutter opened to where it
// is, as a quad along it on screen, half a pixel wider than the drop each side for its edge
[shader("vertex")]
RainVertexOut rainVertex(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID)
{
  RainFrame* f = pc.frame;
  if(instanceId >= RAIN_MAX_STREAKS)
    return rainCulled();
  const RainStreak st    = f->streaks[instanceId];
  const float2     s0    = st.ends.xy;
  const float2     s1    = st.ends.zw;
  const float2     along = s1 - s0;
  const float      len   = length(along);
  const float2     axis  = len > 1e-3F ? along / len : float2(0.0F, 1.0F);
  const float2     side  = float2(-axis.y, axis.x);

  // corners: (end, side) per triangle pair
  const uint  corner     = vertexId % 6u;
  const bool  atEnd      = corner == 1u || corner == 4u || corner == 5u;
  const float toward     = (corner == 2u || corner == 3u || corner == 5u) ? 1.0F : -1.0F;
  const float z          = atEnd ? st.depth.z : st.depth.x;
  const float w          = atEnd ? st.depth.w : st.depth.y;
  const float dropPixels = atEnd ? st.width.y : st.width.x;
  const float halfCore   = 0.5F * max(dropPixels, 1.0F);
  const float halfWidth  = halfCore + 0.5F;
  const float2 s         = (atEnd ? s1 + axis * halfCore : s0 - axis * halfCore) + side * (toward * halfWidth);

  RainVertexOut o;
  o.position = float4(s / (0.5F * f->viewport) * w, z, w);
  o.across   = toward * halfWidth;
  o.halfCore = halfCore;
  o.light    = st.light.rgb;
  o.coverage = st.light.a * min(dropPixels, 1.0F);
  return o;
}

// A streak's pixel: covered across its width (a pixel's soft edge), and hidden where the scene is
// nearer than the drop
[shader("fragment")]
float4 rainFragment(RainVertexOut i) : SV_Target0
{
  RainFrame* f     = pc.frame;
  const int2 texel = int2(i.position.xy * (f->depthSize / f->viewport));
  if(u_depth.Load(int3(texel, 0)) < i.position.z)
    discard;
  const float a = i.coverage * saturate(i.halfCore + 0.5F - abs(i.across));
  return float4(i.light * a, a);
}
