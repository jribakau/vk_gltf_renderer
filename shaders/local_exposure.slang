// Local exposure (src/local_exposure.hpp): shadows lifted and highlights held, region by region, on
// the image the tone mapper takes (the grid and the curve in local_exposure.h.slang).
//   lexpSplat: one workgroup a cell: its 2x2 blocks' stops from mid grey summed into the cell's bins,
//              in groupshared integers (fixed point), written as (stops summed, blocks)
//   lexpBlur:  the grid blurred 1 2 1 across the cells and the bins, one thread a cell's bin
//   lexpApply: each pixel's region read back from the blurred grid at its place and its own stops
//              (trilinear, weighed by the pixels each corner holds), the region's gain on its colour

#include "local_exposure_io.h.slang"
#include "local_exposure.h.slang"

// clang-format off
[[vk::binding(LocalExposureBindings::eLexpImage)]]   Sampler2D<float4>          u_image;
[[vk::binding(LocalExposureBindings::eLexpGrid)]]    RWStructuredBuffer<float2> u_grid;
[[vk::binding(LocalExposureBindings::eLexpBlurred)]] RWStructuredBuffer<float2> u_blurred;
[[vk::binding(LocalExposureBindings::eLexpOut)]]     RWTexture2D<float4>        u_out;

[[vk::push_constant]] ConstantBuffer<LocalExposurePushConstant> pc;
// clang-format on

static const float kHalfMax = 65504.0;  // Largest half float: the output is RGBA16F

float luminanceOf(float3 c)
{
  return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Pixel p of the image, as it is (the sampler at its centre)
float4 pixel(int2 p)
{
  return u_image.SampleLevel((float2(p) + 0.5) / float2(pc.size), 0);
}

groupshared int  g_sum[LEXP_BINS];    // a bin's stops, in 1/LEXP_FIXED
groupshared uint g_count[LEXP_BINS];  // its blocks

[shader("compute")]
[numthreads(LEXP_WORKGROUP_SIZE, LEXP_WORKGROUP_SIZE, 1)]
void lexpSplat(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID, uint index: SV_GroupIndex)
{
  if(index < LEXP_BINS)
  {
    g_sum[index]   = 0;
    g_count[index] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // The cell's pixels two by two: each 2x2 block's mean in one bilinear tap at its middle (the
  // sampler clamps at the image's edge), a quarter of the reads and the sums of every pixel. The apply
  // reads each pixel's own luminance, so an edge stays sharp.
  const int  blocks = pc.cell / 2;  // a side's, the cell's side being even
  const int2 origin = int2(group.xy) * blocks;
  for(int y = int(local.y); y < blocks; y += LEXP_WORKGROUP_SIZE)
  {
    for(int x = int(local.x); x < blocks; x += LEXP_WORKGROUP_SIZE)
    {
      const int2 block = origin + int2(x, y);
      if(block.x * 2 >= pc.size.x || block.y * 2 >= pc.size.y)
        continue;
      const float l = luminanceOf(u_image.SampleLevel((float2(block) * 2.0 + 1.0) / float2(pc.size), 0).rgb);
      if(!isfinite(l))  // a sun disk past half floats, or a NaN: no region's level
        continue;
      const float s = lexpStops(l, pc.exposure);
      const int   b = lexpBin(s);
      InterlockedAdd(g_sum[b], int(round(s * LEXP_FIXED)));
      InterlockedAdd(g_count[b], 1u);
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if(index < LEXP_BINS)
    u_grid[lexpIndex(int(group.x), int(group.y), int(index), pc.cells.x)] =
        float2(float(g_sum[index]) / LEXP_FIXED, float(g_count[index]));
}

[shader("compute")]
[numthreads(LEXP_BLUR_GROUP, 1, 1)]
void lexpBlur(uint3 id: SV_DispatchThreadID)
{
  const int i = int(id.x);
  if(i >= pc.cells.x * pc.cells.y * LEXP_BINS)
    return;
  const int bin  = i % LEXP_BINS;
  const int cell = i / LEXP_BINS;
  const int cx   = cell % pc.cells.x;
  const int cy   = cell / pc.cells.x;
  float2    sum  = float2(0.0);
  for(int dy = -1; dy <= 1; dy++)
  {
    for(int dx = -1; dx <= 1; dx++)
    {
      for(int dz = -1; dz <= 1; dz++)
      {
        const int x = clamp(cx + dx, 0, pc.cells.x - 1);
        const int y = clamp(cy + dy, 0, pc.cells.y - 1);
        const int z = clamp(bin + dz, 0, LEXP_BINS - 1);
        sum += lexpBlurWeight(dx) * lexpBlurWeight(dy) * lexpBlurWeight(dz) * u_grid[lexpIndex(x, y, z, pc.cells.x)];
      }
    }
  }
  u_blurred[i] = sum;
}

[shader("compute")]
[numthreads(LEXP_WORKGROUP_SIZE, LEXP_WORKGROUP_SIZE, 1)]
void lexpApply(uint3 id: SV_DispatchThreadID)
{
  const int2 p = int2(id.xy);
  if(p.x >= pc.size.x || p.y >= pc.size.y)
    return;
  const float4 c = pixel(p);
  const float  l = luminanceOf(c.rgb);
  if(!isfinite(l))
  {
    u_out[p] = float4(clamp(c.rgb, float3(-kHalfMax), float3(kHalfMax)), c.a);
    return;
  }
  const float s = lexpStops(l, pc.exposure);

  // The grid at the pixel's place, cell k's centre at (k + 0.5) cells, and at its own stops
  const float2 g  = (float2(p) + 0.5) / float(pc.cell) - 0.5;
  const int2   g0 = int2(floor(g));
  const float2 t  = g - float2(g0);
  const float  z  = lexpBinCoord(s);
  const int    z0 = int(floor(z));
  const float  tz = z - float(z0);
  float2       at = float2(0.0);
  for(int dy = 0; dy <= 1; dy++)
  {
    for(int dx = 0; dx <= 1; dx++)
    {
      for(int dz = 0; dz <= 1; dz++)
      {
        const int   x = clamp(g0.x + dx, 0, pc.cells.x - 1);
        const int   y = clamp(g0.y + dy, 0, pc.cells.y - 1);
        const int   b = min(z0 + dz, LEXP_BINS - 1);
        const float w = (dx != 0 ? t.x : 1.0 - t.x) * (dy != 0 ? t.y : 1.0 - t.y) * (dz != 0 ? tz : 1.0 - tz);
        at += w * u_blurred[lexpIndex(x, y, b, pc.cells.x)];
      }
    }
  }
  // The region's level: the stops of the pixels round it at about its own; with none, its own
  const float base = at.y > 1e-3 ? at.x / at.y : s;
  const float gain = exp2(lexpGain(base, pc.shadows, pc.highlights));
  u_out[p]         = float4(clamp(c.rgb * gain, float3(-kHalfMax), float3(kHalfMax)), c.a);
}
