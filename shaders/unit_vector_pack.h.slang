#ifndef FV_UNIT_VECTOR_PACK_H
#define FV_UNIT_VECTOR_PACK_H

// Vertex normals and tangents in 32 bits each, as the vertex streams hold them: a unit vector on the
// octahedron (Cigolle et al. 2014, "A Survey of Efficient Representations for Independent Unit
// Vectors"), folded over its lower half and flattened to a square, each side quantised. A normal is
// 16 bits a side, within about 0.005 degrees; a tangent 16 and 15, its handedness in the top bit,
// within about 0.01. Falcor packs a normal in 6 bytes and a tangent's pair in 12; ours is 4 and 4,
// where floats were 12 and 16.
//
// A vector of no length (a degenerate triangle's normal, a vertex no UV triangle reaches) keeps no
// direction: it has a code of its own, the low half all ones, which a direction never takes (its
// sides end at 65534), and comes back as zero, as it went in.
// Slang and C++ (tests/test_unit_vector_pack.cpp, Residency::uploadStreams and the tests' fake GPU),
// so what the CPU packs is what the shaders unpack.

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

static const uint kUnitVectorZero = 0xFFFFu;  // the low half that stands for no direction

inline float unitPackAbs(float v)
{
  return v < 0.0F ? -v : v;
}

inline float unitPackSign(float v)
{
  return v < 0.0F ? -1.0F : 1.0F;
}

// v in -1..1 to 0..2 * steps, the nearest step
inline uint unitPackQuantise(float v, float steps)
{
  const float c = v < -1.0F ? -1.0F : (v > 1.0F ? 1.0F : v);
  const float s = c * steps;
  return uint(int(s + (s < 0.0F ? -0.5F : 0.5F)) + int(steps));
}

// The square's point for a direction of any length, or false for none (no length, or not finite)
inline bool unitPackOct(float3 v, NVSHADERS_OUT_TYPE(float) x, NVSHADERS_OUT_TYPE(float) y)
{
  const float sum = unitPackAbs(v.x) + unitPackAbs(v.y) + unitPackAbs(v.z);
  x               = 0.0F;
  y               = 0.0F;
  if(!(sum > 1.0e-30F) || !(sum < 3.0e38F))  // NaN fails both
    return false;
  x = v.x / sum;
  y = v.y / sum;
  if(v.z < 0.0F)
  {
    const float ox = x;
    x              = (1.0F - unitPackAbs(y)) * unitPackSign(ox);
    y              = (1.0F - unitPackAbs(ox)) * unitPackSign(y);
  }
  return true;
}

// The unit direction at the square's point
inline float3 unitUnpackOct(float x, float y)
{
  float3 v = float3(x, y, 1.0F - unitPackAbs(x) - unitPackAbs(y));
  if(v.z < 0.0F)
  {
    const float ox = v.x;
    v.x            = (1.0F - unitPackAbs(v.y)) * unitPackSign(ox);
    v.y            = (1.0F - unitPackAbs(ox)) * unitPackSign(v.y);
  }
  return normalize(v);
}

// A normal's direction (any length) in 32 bits; no length: kUnitVectorZero's code
inline uint packUnitVector(float3 n)
{
  float x, y;
  if(!unitPackOct(n, x, y))
    return 0xFFFFFFFFu;
  return unitPackQuantise(x, 32767.0F) | (unitPackQuantise(y, 32767.0F) << 16);
}

inline float3 unpackUnitVector(uint packed)
{
  if((packed & 0xFFFFu) == kUnitVectorZero)
    return float3(0.0F);
  return unitUnpackOct(float(int(packed & 0xFFFFu) - 32767) / 32767.0F, float(int((packed >> 16) & 0xFFFFu) - 32767) / 32767.0F);
}

// A tangent's direction (xyz, any length) and its handedness (the sign of w) in 32 bits
inline uint packTangent(float4 t)
{
  const uint handed = t.w < 0.0F ? 0x80000000u : 0u;
  float      x, y;
  if(!unitPackOct(float3(t.x, t.y, t.z), x, y))
    return kUnitVectorZero | handed;
  return unitPackQuantise(x, 32767.0F) | (unitPackQuantise(y, 16383.0F) << 16) | handed;
}

// xyz unit (zero for none) and w +1 or -1
inline float4 unpackTangent(uint packed)
{
  const float w = (packed & 0x80000000u) != 0u ? -1.0F : 1.0F;
  if((packed & 0xFFFFu) == kUnitVectorZero)
    return float4(0.0F, 0.0F, 0.0F, w);
  const float3 t = unitUnpackOct(float(int(packed & 0xFFFFu) - 32767) / 32767.0F, float(int((packed >> 16) & 0x7FFFu) - 16383) / 16383.0F);
  return float4(t.x, t.y, t.z, w);
}

NAMESPACE_SHADERIO_END()

#endif
