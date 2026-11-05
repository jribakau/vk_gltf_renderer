#ifndef FV_LIGHT_RIS_H
#define FV_LIGHT_RIS_H

// Resampled importance sampling of a vertex's light sample (Talbot, Cline and Egbert 2005: the
// candidate stage of ReSTIR, Bitterli et al. 2020, without its reuse). The vertex draws several light
// samples the usual way (sampleLights: the pick by estimate, the light tree, a triangle), weighs each
// by what it would give the pixel unshadowed, keeps one in proportion to that, and traces a shadow
// ray for it alone. Among dozens of emitters the pick's estimates know neither the BSDF nor which
// way the surface faces a lamp's own glow, so most single samples land on light the surface barely
// takes; the kept candidate is one it does.
//
// Unbiased, with any mix of point, sized, area and environment samples: a candidate's weight is the
// luminance of its own one-sample estimate (radiance over the pdf it was drawn with, times the BSDF,
// times its MIS weight), which is the target over the source pdf of generalized RIS (Lin et al. 2022)
// in the measure the sample was drawn in. The kept estimate is scaled by the candidates' mean weight
// over its own, so its expectation is the mean of the candidates' estimates, each unbiased. Visibility
// is not in the weights: a kept candidate that is shadowed gives nothing, as a single sample does.
//
// The MIS weights stay the single sample's (the pick's pdf against the BSDF's): they split the
// integrand between the two techniques whatever density each draws with, so the BSDF ray's weight on
// the far side is unchanged. Slang and C++ (tests/test_light_ris.cpp), so the tests run the shader's
// arithmetic.

// The path tracer compiles all of this in only where a build asks for it (gltf_pathtrace.slang)
#ifndef USE_LIGHT_RIS
#define USE_LIGHT_RIS 0
#endif

static const int LIGHT_RIS_MAX_CANDIDATES = 16;

struct LightReservoir
{
  float weightSum;  // the candidates' weights so far
  float kept;       // the weight of the candidate kept (0: none yet)
  int   count;      // candidates seen, those of weight 0 included: they count in the mean
  bool  keptLast;   // the candidate just added replaced the one kept
};

LightReservoir lightReservoirEmpty()
{
  LightReservoir r;
  r.weightSum = 0.0F;
  r.kept      = 0.0F;
  r.count     = 0;
  r.keptLast  = false;
  return r;
}

// `r` with a candidate of weight `weight` seen, `u` uniform in [0, 1). The candidate is kept with
// probability weight / weightSum, which leaves every candidate so far kept in proportion to its
// weight. A weight that is not finite, as a sample with NaN in it, is never kept and adds nothing.
LightReservoir lightReservoirAdd(LightReservoir r, float weight, float u)
{
  r.count += 1;
  r.keptLast = false;
  if(!(weight > 0.0F) || !(weight < 3.0e38F))
    return r;
  r.weightSum += weight;
  if(u * r.weightSum < weight)
  {
    r.kept     = weight;
    r.keptLast = true;
  }
  return r;
}

// What the kept candidate's estimate is multiplied by: the candidates' mean weight over its own.
// 0 when no candidate had any weight (the vertex takes no light sample).
float lightReservoirScale(LightReservoir r)
{
  if(!(r.kept > 0.0F) || r.count <= 0)
    return 0.0F;
  return r.weightSum / (float(r.count) * r.kept);
}

//-----------------------------------------------------------------------------------------------------
// Reuse from frame to frame (ReSTIR's temporal pass, Bitterli et al. 2020, Algorithm 4 with its
// M weights). A pixel's first surface hit keeps the sample it shaded, with the weight W that makes
// f(y) W an estimate of the light, and M, the candidates it stands for. The next frame takes that
// record in as one more candidate, re-weighed at its own shading point: target there (the sample's
// unshadowed contribution, in the measure it was stored in: on the emitter's surface, or in
// directions for the environment) times W times M. A record whose sample was shadowed keeps M but
// W 0, so what a pixel carries forward is what it found lit, and the hidden lamps fall away.
//
// Unbiased where the record's shading point is this one (a still: the same pixel, only its jitter
// moved): the record then estimates the light that reaches the point unshadowed, the current
// candidates all of it, and their M weights sum to one wherever light arrives. Elsewhere (a moving
// view's reprojected pixel, the jitter across an edge) a record is used only if its point and normal
// are close (lightReuseSameSurface), and what bias is left lies along edges and shadow boundaries.

static const uint LIGHT_REUSE_NONE        = 0u;  // nothing kept: a miss, a fibre, a light the record cannot name
static const uint LIGHT_REUSE_EMISSIVE    = 1u;  // index: the emissive triangle, uv: the point's barycentrics (b1, b2)
static const uint LIGHT_REUSE_ENVIRONMENT = 2u;  // uv: the world direction, octahedral

struct LightReuseRecord
{
  float  depth;   // the shading point's depth along the view, in the camera of the frame that wrote it
  float3 normal;  // its shading normal
  float  target;  // the sample's target there (its unshadowed contribution, in the record's measure)
  uint   kind;    // LIGHT_REUSE_*
  uint   index;
  float2 uv;
  float  W;  // f(y) W estimates the light (0: the sample was shadowed)
  float  M;  // the candidates it stands for (0: no record)
};

// Octahedral mapping of a unit vector to [-1, 1]^2 and back (Cigolle et al. 2014)
float2 lightReuseOctEncode(float3 n)
{
  const float s = abs(n.x) + abs(n.y) + abs(n.z);
  float2      p = float2(n.x, n.y) / max(s, 1e-20F);
  if(n.z < 0.0F)
    p = float2((1.0F - abs(p.y)) * (p.x >= 0.0F ? 1.0F : -1.0F), (1.0F - abs(p.x)) * (p.y >= 0.0F ? 1.0F : -1.0F));
  return p;
}

float3 lightReuseOctDecode(float2 p)
{
  float3      n = float3(p.x, p.y, 1.0F - abs(p.x) - abs(p.y));
  const float t = max(-n.z, 0.0F);
  n.x += (n.x >= 0.0F) ? -t : t;
  n.y += (n.y >= 0.0F) ? -t : t;
  return normalize(n);
}

uint lightReusePackSnorm2(float2 v)
{
  const uint x = uint(int(round(clamp(v.x, -1.0F, 1.0F) * 32767.0F)) & 0xFFFF);
  const uint y = uint(int(round(clamp(v.y, -1.0F, 1.0F) * 32767.0F)) & 0xFFFF);
  return x | (y << 16u);
}

float2 lightReuseUnpackSnorm2(uint p)
{
  const int x = int(p << 16u) >> 16;
  const int y = int(p) >> 16;
  return float2(max(float(x) / 32767.0F, -1.0F), max(float(y) / 32767.0F, -1.0F));
}

uint lightReusePackUnorm2(float2 v)
{
  const uint x = uint(round(clamp(v.x, 0.0F, 1.0F) * 65535.0F));
  const uint y = uint(round(clamp(v.y, 0.0F, 1.0F) * 65535.0F));
  return x | (y << 16u);
}

float2 lightReuseUnpackUnorm2(uint p)
{
  return float2(float(p & 0xFFFFu), float(p >> 16u)) / 65535.0F;
}

// A record as the two uint4 it is kept in. The kind takes the index's top two bits.
uint4 lightReusePackA(LightReuseRecord r)
{
  return uint4(asuint(r.depth), lightReusePackSnorm2(lightReuseOctEncode(r.normal)), asuint(r.target), 0u);
}

uint4 lightReusePackB(LightReuseRecord r)
{
  const uint uv = (r.kind == LIGHT_REUSE_ENVIRONMENT) ? lightReusePackSnorm2(r.uv) : lightReusePackUnorm2(r.uv);
  return uint4((r.kind << 30u) | (r.index & 0x3FFFFFFFu), uv, asuint(r.W), asuint(r.M));
}

LightReuseRecord lightReuseUnpack(uint4 a, uint4 b)
{
  LightReuseRecord r;
  r.depth  = asfloat(a.x);
  r.normal = lightReuseOctDecode(lightReuseUnpackSnorm2(a.y));
  r.target = asfloat(a.z);
  r.kind   = b.x >> 30u;
  r.index  = b.x & 0x3FFFFFFFu;
  r.uv     = (r.kind == LIGHT_REUSE_ENVIRONMENT) ? lightReuseUnpackSnorm2(b.y) : lightReuseUnpackUnorm2(b.y);
  r.W      = asfloat(b.z);
  r.M      = asfloat(b.w);
  if(!(r.M > 0.0F) || !(r.W >= 0.0F) || !(r.W < 3.0e38F))  // nothing, or a record no frame wrote
  {
    r.M = 0.0F;
    r.W = 0.0F;
  }
  return r;
}

// Whether a record's shading point is close enough to this one to stand in for it: at its depth, in the
// camera that wrote the record, within 2%, and its normal near this one's
bool lightReuseSameSurface(LightReuseRecord r, float depth, float3 normal)
{
  return depth > 0.0F && abs(r.depth - depth) <= 0.02F * depth && dot(r.normal, normal) >= 0.9F;
}

// The M weights (Bitterli et al.'s), by what a candidate is: the record and the current candidates
// share the light a record can hold (emitters, the environment), each candidate 1 / (currentM + recordM)
// and the record recordM / (currentM + recordM); a punctual light, which no record holds, the current
// candidates alone, 1 / currentM each. They sum to one for every light, and none depends on what was
// drawn: a pixel's M is the candidates it took in, kept whatever its sample was (a sample the record
// cannot hold, or one that weighed nothing, is kept as W 0). A history wiped on such an outcome left
// the current candidates their full weight on top of what the history covered: 6% too bright on
// 7TEST, three times in its darkest corners.
float lightReuseRecordM(LightReuseRecord r, float currentM, float cap)
{
  return min(r.M, cap * currentM);
}

// The record's M, as much less as its sample looks brighter here than where it was found (its target
// there over its target here, at most 1). W was made for that point: carried to one where the sample
// gives several times more (a normal map's bump, the jitter across a lit edge), it made a firefly Ray
// Reconstruction kept, a wall of them on 7TEST's moving view. With it the record weighs at most what it
// did there. Biased where that happens (the balance heuristic's ratio, taken from the record's own
// sample for every candidate), and not at all where the sample looks no brighter, as on a still's pixel.
float lightReuseEffectiveM(float recordM, LightReuseRecord r, float recordTargetHere)
{
  if(!(recordTargetHere > r.target) || !(r.target > 0.0F))
    return recordM;
  return recordM * (r.target / recordTargetHere);
}

// A current candidate's M weight, `storable` if a record can hold it (LIGHT_REUSE_EMISSIVE, _ENVIRONMENT)
float lightReuseCandidateShare(bool storable, float currentM, float recordM)
{
  return 1.0F / (storable ? currentM + recordM : currentM);
}

// The record's resampling weight: its M weight times its target here (its sample's unshadowed
// contribution at this point, in its own measure) times its W
float lightReuseRecordWeight(LightReuseRecord r, float recordTargetHere, float currentM, float recordM)
{
  const float w = recordM / (currentM + recordM) * recordTargetHere * r.W;
  return (w > 0.0F && w < 3.0e38F) ? w : 0.0F;
}

#endif
