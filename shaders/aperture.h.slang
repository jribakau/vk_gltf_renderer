#ifndef FV_APERTURE_H
#define FV_APERTURE_H

// The camera's aperture: a point on it, uniform over its area, for depth of field. Slang and C++
// (tests/test_aperture.cpp), so the tests run the sampler the shader runs.
//
// `blades` below 3 is a round aperture: the disc samplePixel always had, from the same two numbers.
// Otherwise the aperture is a polygon of `blades` straight blades with its corners on the unit
// circle, one corner at angle `rotation` (radians, from the camera's right toward its up). `u1` picks
// the wedge between two corners and, stretched back over [0, 1), where along its outer edge the
// point lies; `u2` its distance out, as sqrt so the triangle is covered evenly. Both consume the same
// two random numbers, so a round aperture renders as it did.
float2 sampleAperture(float u1, float u2, int blades, float rotation)
{
  const float twoPi = 6.28318530717958647692F;
  const float r     = sqrt(u2);
  if(blades < 3)
  {
    const float phi = u1 * twoPi;
    return float2(cos(phi), sin(phi)) * r;
  }
  const float  wedge = u1 * float(blades);
  const float  k     = min(floor(wedge), float(blades - 1));
  const float  t     = wedge - k;
  const float  a0    = rotation + k * twoPi / float(blades);
  const float  a1    = a0 + twoPi / float(blades);
  const float2 c0    = float2(cos(a0), sin(a0));
  const float2 c1    = float2(cos(a1), sin(a1));
  return (c0 + (c1 - c0) * t) * r;
}

#endif
