// Bloom (src/bloom.hpp): a glow and a star added to the image the tone mapper takes, keeping its light.
//   bloomDown:      a level from the finer one: Jimenez's 13 taps (bloom.h.slang); on the image itself,
//                   each tap in the tone mapper's white (pc.scale) and held under pc.limit
//   bloomUp:        a level's glow: the level, and the coarser level's glow through a tent, mixed by
//                   pc.tail: the levels summed with geometric weights from the coarsest up
//   bloomStar:      the star on the quarter-size level: taps along each line across the aperture's
//                   edges, both ways, normalised (bloomStarNorm)
//   bloomComposite: the image with pc.share of its light moved into the glow and, of that, pc.star
//                   into the star, back in the image's units (pc.scale)

#include "bloom_io.h.slang"
#include "bloom.h.slang"

// clang-format off
[[vk::binding(BloomBindings::eBloomA)]]   Sampler2D<float4>   u_a;
[[vk::binding(BloomBindings::eBloomB)]]   Sampler2D<float4>   u_b;
[[vk::binding(BloomBindings::eBloomC)]]   Sampler2D<float4>   u_c;
[[vk::binding(BloomBindings::eBloomOut)]] RWTexture2D<float4> u_out;

[[vk::push_constant]] ConstantBuffer<BloomPushConstant> pc;
// clang-format on

static const float kHalfMax = 65504.0;  // Largest half float: the levels and the output are RGBA16F

// A tap of A in the pyramid's units: scaled, its largest channel held under the limit, and nothing
// that is not a finite, positive colour
float3 tapA(float2 uv)
{
  float3 c = u_a.SampleLevel(uv, 0).rgb * pc.scale;
  if(!all(isfinite(c)))
    return float3(0.0);
  c = max(c, float3(0.0));
  const float m = max(c.r, max(c.g, c.b));
  if(pc.limit > 0.0 && m > pc.limit)
    c *= pc.limit / m;
  return c;
}

// The destination pixel's centre, in the uv every source shares
float2 centre(int2 p)
{
  return (float2(p) + 0.5) / float2(pc.outSize);
}

[shader("compute")]
[numthreads(BLOOM_WORKGROUP_SIZE, BLOOM_WORKGROUP_SIZE, 1)]
void bloomDown(uint3 id: SV_DispatchThreadID)
{
  const int2 p = int2(id.xy);
  if(p.x >= pc.outSize.x || p.y >= pc.outSize.y)
    return;
  const float2 uv  = centre(p);
  float3       sum = float3(0.0);
  for(int i = 0; i < 13; i++)
    sum += bloomDownWeight(i) * tapA(uv + bloomDownOffset(i) * pc.aTexel);
  u_out[p] = float4(min(sum, float3(kHalfMax)), 1.0);
}

[shader("compute")]
[numthreads(BLOOM_WORKGROUP_SIZE, BLOOM_WORKGROUP_SIZE, 1)]
void bloomUp(uint3 id: SV_DispatchThreadID)
{
  const int2 p = int2(id.xy);
  if(p.x >= pc.outSize.x || p.y >= pc.outSize.y)
    return;
  const float2 uv     = centre(p);
  float3       coarse = float3(0.0);
  for(int i = 0; i < 9; i++)
    coarse += bloomTentWeight(i) * u_b.SampleLevel(uv + bloomTentOffset(i) * pc.bTexel, 0).rgb;
  const float3 fine = u_a.SampleLevel(uv, 0).rgb;
  u_out[p]          = float4(min(fine * (1.0 - pc.tail) + coarse * pc.tail, float3(kHalfMax)), 1.0);
}

[shader("compute")]
[numthreads(BLOOM_WORKGROUP_SIZE, BLOOM_WORKGROUP_SIZE, 1)]
void bloomStar(uint3 id: SV_DispatchThreadID)
{
  const int2 p = int2(id.xy);
  if(p.x >= pc.outSize.x || p.y >= pc.outSize.y)
    return;
  const float2 uv    = centre(p);
  const int    lines = bloomStarLines(pc.blades);
  float3       sum   = float3(0.0);
  for(int l = 0; l < lines; l++)
  {
    const float a = bloomStarAngle(pc.blades, pc.rotation, l);
    // the image's rows go down: its top is -y
    const float2 step = float2(cos(a), -sin(a)) * pc.starStep * pc.aTexel;
    for(int k = 1; k <= BLOOM_STAR_TAPS; k++)
    {
      const float w = bloomStarWeight(k);
      sum += w * (u_a.SampleLevel(uv + step * float(k), 0).rgb + u_a.SampleLevel(uv - step * float(k), 0).rgb);
    }
  }
  sum      = lines > 0 ? sum / bloomStarNorm(lines) : float3(0.0);
  u_out[p] = float4(min(sum, float3(kHalfMax)), 1.0);
}

[shader("compute")]
[numthreads(BLOOM_WORKGROUP_SIZE, BLOOM_WORKGROUP_SIZE, 1)]
void bloomComposite(uint3 id: SV_DispatchThreadID)
{
  const int2 p = int2(id.xy);
  if(p.x >= pc.outSize.x || p.y >= pc.outSize.y)
    return;
  const float2 uv    = centre(p);
  const float4 image = u_c.SampleLevel(uv, 0);
  float3       glow  = float3(0.0);
  for(int i = 0; i < 9; i++)
    glow += bloomTentWeight(i) * u_a.SampleLevel(uv + bloomTentOffset(i) * pc.aTexel, 0).rgb;
  if(pc.star > 0.0)
    glow = glow * (1.0 - pc.star) + u_b.SampleLevel(uv, 0).rgb * pc.star;
  // Written as a weighted sum: an infinite pixel (a sun disk past half floats) stays that pixel's
  const float3 rgb = image.rgb * (1.0 - pc.share) + glow * (pc.scale * pc.share);
  u_out[p]         = float4(clamp(rgb, float3(-kHalfMax), float3(kHalfMax)), image.a);
}
