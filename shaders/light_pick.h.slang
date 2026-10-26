#ifndef FV_LIGHT_PICK_H
#define FV_LIGHT_PICK_H

// Which light next-event estimation samples at a shading point. Each punctual light and the
// environment is picked in proportion to an estimate of what it gives that point, so a sun below
// the surface's horizon or a dim fill no longer takes the share a uniform pick gave it (Unreal's
// InitLightPickingCdf / SelectLight, plus horizon culling). Slang and C++ (tests/test_light_pick.cpp),
// so the tests run the arithmetic the shader runs.
//
// An estimate is the largest channel of the irradiance the light gives at normal incidence, its
// occlusion unknown: the channel measure of the environment's alias table (hdr_ibl.cpp), whose
// integral over 4 is the environment's estimate (a uniform sky of that power on a surface facing it).

// Above this many punctual lights the pick is uniform and nothing is culled: weighing every light at
// every vertex would cost more than it saves.
static const int LIGHT_PICK_MAX_LIGHTS = 16;
// The share spread evenly over the candidates, for the light an estimate underrates: one the
// figure shades, or a glossy lobe's reflection of a light the irradiance calls dim.
static const float LIGHT_PICK_DEFENSIVE = 0.1F;

// A directional light travelling along `lightDir`, or a point or spot light at `lightPos` of radius
// `radius` aiming along `lightDir`. `power`: the largest channel of colour times intensity;
// `invRange`: 1 / the KHR_lights_punctual range, 0 for none; the spot's cone in radians.
//
// The irradiance singleLightContribution (light_contrib.h.slang) gives the point, before the
// sample's direction is drawn. With `cull`, 0 for a light that lies wholly below the plane of
// `normal`, where next-event estimation would discard its sample: a directional light whose centre
// is below it (singleLightContribution's own test), a sphere whose top is.
float punctualLightPickEstimate(bool   directional,
                                bool   spot,
                                float3 lightPos,
                                float3 lightDir,
                                float  radius,
                                float  power,
                                float  invRange,
                                float  innerAngle,
                                float  outerAngle,
                                float3 pos,
                                float3 normal,
                                bool   cull)
{
  if(!(power > 0.0F))
    return 0.0F;
  if(directional)
    return (cull && dot(normal, -lightDir) <= 0.0F) ? 0.0F : power;

  const float3 toLight = lightPos - pos;
  const float  d2      = dot(toLight, toLight);
  const float  d       = sqrt(d2);
  if(cull && dot(normal, toLight) <= -radius)
    return 0.0F;

  float irradiance;
  if(radius > 0.0F)
  {
    // The sphere's solid angle, as singleLightContribution computes it; the point inside it sees a hemisphere
    const float r                    = (d > 0.0F) ? min(radius / d, 1.0F) : 1.0F;
    const float r2p1                 = r * r + 1.0F;
    const float oneMinusCosHalfAngle = r * r / (r2p1 + sqrt(r2p1));
    irradiance                       = power / (radius * radius) * 2.0F * oneMinusCosHalfAngle;
  }
  else
    irradiance = power / max(d2, 1e-20F);

  if(invRange > 0.0F)
  {
    const float x2 = d2 * invRange * invRange;
    irradiance *= clamp(1.0F - x2 * x2, 0.0F, 1.0F);
  }
  if(spot)
  {
    if(!(d > 0.0F))
      return 0.0F;
    const float cosAngle = dot(lightDir, -toLight / d);
    const float cosOuter = cos(outerAngle);
    const float cosInner = cos(innerAngle);
    if(cosAngle <= cosOuter)
      return 0.0F;
    if(cosAngle < cosInner)
    {
      const float t = (cosAngle - cosOuter) / (cosInner - cosOuter);
      irradiance *= t * t;
    }
  }
  return irradiance;
}

// The probability of picking a candidate of `estimate` among `candidates` (lights and environment
// whose estimates are above 0) summing to `total`. 0 for a light that cannot light the point.
float lightPickProbability(float estimate, float total, int candidates)
{
  if(!(estimate > 0.0F) || !(total > 0.0F) || candidates <= 0)
    return 0.0F;
  return (1.0F - LIGHT_PICK_DEFENSIVE) * (estimate / total) + LIGHT_PICK_DEFENSIVE / float(candidates);
}

// The power heuristic (exponent 2): the weight of a sample of pdf `a` against the other technique's
// pdf `b` for the same direction. As a ratio, so a near-delta lobe's pdf is never squared into
// infinity (Unreal's MISWeightPower). 1 when the other technique cannot produce the sample.
float misPowerWeight(float a, float b)
{
  if(!(b > 0.0F))
    return 1.0F;
  if(a >= b)
  {
    const float r = b / a;
    return 1.0F / (1.0F + r * r);
  }
  const float r  = a / b;
  const float r2 = r * r;
  return r2 / (1.0F + r2);
}

#endif
