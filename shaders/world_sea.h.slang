/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// A planet's sea (src/fv_world.hpp): the ground's body of water (ground.h.slang) on a sphere.
//
// The sea is no mesh. It is the sphere at the planet's sea level, which a ray meets by itself, and
// everything under it is in the water: a path's ray refracts at the surface, is charged the water's
// extinction and takes the light it scatters, a shadow ray crosses it straight with its Fresnel share
// and the caustic the surface focuses, and a camera under it starts in it. All of it is
// ground.h.slang's, asked about a sphere instead of a level plane, but for three things:
//
//   * the waves are in the surface's normal and in its caustics, not in its shape: the surface a ray
//     meets is the sphere. The ground's water is a heightfield, walked for every ray that passes its
//     waves; a third of a sea frame went on that walk (11-14: 18.4 ms against 13.7 with a still sea,
//     over shallows), for a waterline that laps a few centimetres. A reflection the tilted normal
//     would send under the sphere is not taken (PbrMaterial::faceSided), so a path is always on the
//     side of the surface it believes it is.
//   * heights are distances from the sphere, taken from the point of it straight over its centre
//     (`top`) so that near that point, where the scene is, they keep a float's whole precision: the
//     planet's centre is tens of kilometres off, and a distance from it is only good to millimetres.
//   * the waves are the plane's, laid on the sphere from the three axes' planes and blended where
//     two meet (within 30 degrees of an axis there is one, and round the top it is the plane's own,
//     point for point); the caustic is worked out on the plane that touches the sphere over the lit
//     point (light comes down a few tens of metres at most before the water has spread what the
//     surface focused).
//
// The sea's swell is the ground water's, SEA_SWELL_SCALE times as long: a pond's wavelets read as a
// model's sea. Its water carries particles as well as its own molecules (seaScattered).
//
// Slang and plain C++ (tests/test_ground.cpp, with the ground's).
#ifndef WORLD_SEA_H_SLANG
#define WORLD_SEA_H_SLANG

#include "ground.h.slang"

// The driver compiles a copy of a function for every place that calls it and unrolls a short loop
// into as many, and a shadow ray's share of this file is compiled wherever a shadow ray is traced
// (docs/RENDERING_ARCHITECTURE.md, "The path tracer's build time"). So the waves are asked for in one
// place of each function, in a loop the driver is told to keep: written out call by call, with the
// surface a heightfield, the sea doubled the path tracer's cold build (11-14).
#ifdef __cplusplus
#define SEA_LOOP
#else
#define SEA_LOOP [loop]
#endif

// How many times as long as the ground water's swell the sea's is (its steepness the same)
static const float SEA_SWELL_SCALE = 4.0F;
// The particles in the water: what they scatter, per metre, the same in every band, and how far
// forward (Henyey-Greenstein's g). Coastal water: the sea's own molecules alone are the clearest
// ocean's, in which the sun draws no shafts and nothing fades with distance but its red.
static const float SEA_PARTICLES        = 0.04F;
static const float SEA_PARTICLE_FORWARD = 0.85F;
// What a ray loses to the water, per metre. The particles' forward peak leaves light going the way it
// went, so only the rest of their scattering counts as lost (the similarity relation); the light they
// send to a ray is still their whole phase function's (seaScattered), so round the sun's direction a
// glow is counted that the beam never lost.
static const float3 SEA_EXTINCTION = GROUND_WATER_EXTINCTION + SEA_PARTICLES * (1.0F - SEA_PARTICLE_FORWARD);

struct Sea
{
  float3 top;      // the point of its surface straight up (+y) from its centre, world space
  float  radius;   // metres
  float  waves;    // its swell, 0 still to 1
  float  ripples;  // the ripples on it, 0 to 1
};

// Whether a planet with a sea stands in the scene
bool worldHasSea(SceneFrameInfo* frameInfo)
{
  return frameInfo->worldSeaRadius > 0.0F;
}

Sea worldSea(SceneFrameInfo* frameInfo)
{
  Sea sea;
  sea.top     = frameInfo->worldSeaTop;
  sea.radius  = frameInfo->worldSeaRadius;
  sea.waves   = frameInfo->worldSeaWaves;
  sea.ripples = frameInfo->worldSeaRipples;
  return sea;
}

// `pos` from the sea's centre
float3 seaFromCentre(Sea sea, float3 pos)
{
  const float3 q = pos - sea.top;
  return float3(q.x, q.y + sea.radius, q.z);
}

// The planet's vertical at `pos`
float3 seaUp(Sea sea, float3 pos)
{
  return normalize(seaFromCentre(sea, pos));
}

// The square of `pos`'s distance from the centre less the radius's: 2 R h + h^2 at a height h over
// the sea, so negative under it. From `top`, so that near it nothing large is taken from something
// large.
float seaOver(Sea sea, float3 pos)
{
  const float3 q = pos - sea.top;
  return dot(q, q) + 2.0F * sea.radius * q.y;
}

// How high `pos` is over the sea
float seaHeight(Sea sea, float3 pos)
{
  const float over = seaOver(sea, pos);
  return over / (sqrt(max(over + sea.radius * sea.radius, 0.0F)) + sea.radius);
}

// Whether `pos` is under the sea's surface
bool seaUnder(Sea sea, float3 pos)
{
  return seaOver(sea, pos) < 0.0F;
}

// The waves at a point of the sea's surface, as a footprint sees them
struct SeaWaves
{
  float3 slope;     // the waves' gradient over the sphere, world axes: the swell's and the ripples'
  float  variance;  // per axis: what the footprints average out, for the roughness
};

// The waves at `pos` (on the sphere or near it): the swell for a footprint `footprint` metres wide,
// the ripples for one of `rippleFootprint`. Each axis's plane of waves counts by how squarely the
// sphere faces it, from none at 60 degrees off: y's first, which round the top is the only one.
SeaWaves seaWavesAt(Sea sea, float3 pos, float footprint, float waves, float rippleFootprint, float ripples)
{
  SeaWaves w;
  w.slope    = float3(0.0F, 0.0F, 0.0F);
  w.variance = 0.0F;
  if(waves <= 0.0F && ripples <= 0.0F)
    return w;
  const float3 r     = seaFromCentre(sea, pos);
  const float3 n     = normalize(r);
  const float  bx    = max(abs(n.x) - 0.5F, 0.0F);
  const float  by    = max(abs(n.y) - 0.5F, 0.0F);
  const float  bz    = max(abs(n.z) - 0.5F, 0.0F);
  const float  total = bx * bx + by * by + bz * bz;  // never 0: one of them is 0.077 at least
  SEA_LOOP
  for(int plane = 0; plane < 3; ++plane)
  {
    const float b = plane == 0 ? by : (plane == 1 ? bx : bz);
    if(b <= 0.0F)
      continue;
    // the plane's axes in the world, and the point on it
    const float3      x       = plane == 1 ? float3(0.0F, 1.0F, 0.0F) : float3(1.0F, 0.0F, 0.0F);
    const float3      z       = plane == 2 ? float3(0.0F, 1.0F, 0.0F) : float3(0.0F, 0.0F, 1.0F);
    const float2      q       = float2(dot(r, x), dot(r, z));
    const GroundWaves swell   = groundWavesAt(q / SEA_SWELL_SCALE, footprint / SEA_SWELL_SCALE, waves);
    const GroundSlope rippled = groundRipplesAt(q, rippleFootprint, ripples);
    const float       k       = b * b / total;
    w.slope += k * (x * (swell.slope.slope.x + rippled.slope.x) + z * (swell.slope.slope.y + rippled.slope.y));
    w.variance += k * (swell.slope.variance + rippled.variance);
  }
  // a plane's slope lies in the plane: what the sphere's surface has of it
  w.slope -= n * dot(n, w.slope);
  return w;
}

// The sea's surface as a ray's shading meets it at `pos`: its normal, up, tilted by the waves a
// footprint `footprint` metres wide sees, and what that footprint averages out of them
struct SeaSurface
{
  float3 up;        // the waves' normal
  float  variance;  // per axis
};

SeaSurface seaSurfaceAt(Sea sea, float3 pos, float footprint)
{
  const SeaWaves w = seaWavesAt(sea, pos, footprint, sea.waves, footprint, sea.ripples);
  SeaSurface     s;
  s.up       = normalize(seaUp(sea, pos) - w.slope);
  s.variance = w.variance;
  return s;
}

// Where a ray meets the sea's surface, and which side of it the ray starts on
struct SeaCrossing
{
  float t;          // along the ray; negative: it meets no water
  bool  fromAbove;  // it comes down onto the water from the air
  bool  under;      // it starts under the surface
};

// Where a ray first meets the sea's surface between 0 and tMax
SeaCrossing seaCrossing(Sea sea, float3 origin, float3 direction, float tMax)
{
  SeaCrossing  c;
  const float3 q = origin - sea.top;
  // seaOver along the ray is t^2 + 2 B t + C
  const float B = dot(q, direction) + sea.radius * direction.y;
  const float C = dot(q, q) + 2.0F * sea.radius * q.y;
  c.t           = -1.0F;
  c.under       = C < 0.0F;
  c.fromAbove   = !c.under;
  const float inside = B * B - C;
  if(inside < 0.0F)
    return c;  // past the planet
  // the farther root, and the nearer from the product of the two, so that a ray coming straight
  // down keeps its precision
  const float outward = -(B + (B < 0.0F ? -1.0F : 1.0F) * sqrt(inside));
  const float other   = outward != 0.0F ? C / outward : 0.0F;
  // from the air the first of the two, from the water the one ahead
  const float t = c.under ? max(outward, other) : min(outward, other);
  if(t > 0.0F && t < tMax)
    c.t = t;
  return c;
}

// Light from `toLight` (x and z along the surface, y up), coming in through the surface at `p`, a
// point of the plane that touches the sea at `over`: where it lands `depth` under that plane, along
// x and z from `over` (groundWaterLanding)
float2 seaLanding(Sea sea, float3 over, float3 x, float3 z, float3 p, float3 toLight, float depth, GroundCausticReach reach)
{
  const SeaWaves w    = seaWavesAt(sea, p, reach.swellOut, reach.waves, reach.rippleOut, reach.ripples);
  const float3   bent = groundIntoWater(-toLight, normalize(float3(-dot(w.slope, x), 1.0F, -dot(w.slope, z))));
  const float3   from = p - over;
  return float2(dot(from, x), dot(from, z)) + float2(bent.x, bent.z) * (depth / max(-bent.y, 1e-4F));
}

// How much more light from `toLight`, above the sea, reaches `origin` under it than a still sea would
// let through (groundWaterCaustic). The places on the surface are points of the scene, and kilometres
// from the origin a float keeps those to a fraction of a millimetre at best: the steps the landing's
// growth is measured across are taken as the floats took them, not as they were asked for, or the
// caustic far out at sea is noise.
float seaCaustic(Sea sea, float3 origin, float3 toLight)
{
  if(sea.waves <= 0.0F && sea.ripples <= 0.0F)
    return 1.0F;
  const float3 up    = seaUp(sea, origin);
  const float3 x     = normalize(cross(abs(up.y) < 0.9F ? float3(0.0F, 1.0F, 0.0F) : float3(1.0F, 0.0F, 0.0F), up));
  const float3 z     = cross(up, x);
  const float  depth = max(-seaHeight(sea, origin), 0.0F);
  const float3 light = float3(dot(toLight, x), dot(toLight, up), dot(toLight, z));
  const float3 still = groundIntoWater(-light, float3(0.0F, 1.0F, 0.0F));
  const float  path  = depth / max(-still.y, 1e-4F);
  GroundCausticReach reach;
  reach.waves     = sea.waves;
  reach.ripples   = sea.ripples;
  reach.swellOut  = GROUND_CAUSTIC_SWELL_REACH * sea.waves * path;
  reach.rippleOut = GROUND_CAUSTIC_RIPPLE_REACH * sea.ripples * path;
  // so deep that every octave has spread its light: the still sea's
  if((sea.waves <= 0.0F || reach.swellOut >= 0.5F * SEA_SWELL_SCALE * GROUND_SWELL_LENGTH[0])
     && (sea.ripples <= 0.0F || reach.rippleOut >= 0.5F * GROUND_RIPPLE_SCALE))
    return 1.0F;
  const float3 over   = origin + up * depth;
  float3       entry  = over - (x * still.x + z * still.z) * path;
  float3       alongX = entry;
  float3       alongZ = entry;
  float2       at     = float2(0.0F, 0.0F);
  float2       dx     = float2(0.0F, 0.0F);
  float2       dz     = float2(0.0F, 0.0F);
  // One place asks where light lands (SEA_LOOP): three steps towards the place it comes in to land at
  // `origin`, the landing there, and the landings a step off along x and along z
  SEA_LOOP
  for(int i = 0; i < 6; ++i)
  {
    const float3 p   = i < 4 ? entry : (i == 4 ? alongX : alongZ);
    const float2 off = seaLanding(sea, over, x, z, p, light, depth, reach);
    if(i < 3)
      entry -= x * off.x + z * off.y;
    else if(i == 3)
    {
      at = off;
      // a millimetre, or a few of the floats' own steps where those are larger
      const float h = max(1e-3F, 4e-7F * (abs(entry.x) + abs(entry.y) + abs(entry.z)));
      alongX        = entry + x * h;
      alongZ        = entry + z * h;
    }
    else if(i == 4)
      dx = off - at;
    else
      dz = off - at;
  }
  const float2 sx      = float2(dot(alongX - entry, x), dot(alongX - entry, z));
  const float2 sz      = float2(dot(alongZ - entry, x), dot(alongZ - entry, z));
  const float  stepped = abs(sx.x * sz.y - sx.y * sz.x);
  if(!(stepped > 0.0F))
    return 1.0F;
  const float det = abs(dx.x * dz.y - dx.y * dz.x) / stepped;
  return 1.0F / max(det, 1.0F / GROUND_CAUSTIC_MAX);
}

// What a shadow ray keeps of its light through the sea (groundWaterShadowTransmittance)
float3 seaShadowTransmittance(Sea sea, float3 origin, float3 direction, float tMax)
{
  const float3 clear = float3(1.0F, 1.0F, 1.0F);
  const float3 r     = seaFromCentre(sea, origin);
  // Over the sea and not going down, as most shadow rays are: in the air all the way
  if(seaOver(sea, origin) >= 0.0F && dot(direction, r) >= 0.0F)
    return clear;
  const SeaCrossing crossing = seaCrossing(sea, origin, direction, tMax);
  if(!crossing.under && crossing.t < 0.0F)
    return clear;  // the ray stays in the air
  // how far the ray runs in the water: to the surface, or from it to the light; all of it when the
  // light is in the water too
  const float wet     = crossing.under ? (crossing.t >= 0.0F ? crossing.t : tMax) : tMax - crossing.t;
  float       through = 1.0F;
  float       focus   = 1.0F;
  if(crossing.t >= 0.0F)
  {
    through = 1.0F - groundWaterFresnel(abs(dot(direction, seaUp(sea, origin + direction * crossing.t))));
    if(crossing.under)
      focus = seaCaustic(sea, origin, direction);
  }
  return through * focus * exp(-SEA_EXTINCTION * wet);
}

// The particles' phase function, per steradian (Henyey-Greenstein)
float seaParticlePhase(float cosTheta)
{
  const float g = SEA_PARTICLE_FORWARD;
  const float d = 1.0F + g * g - 2.0F * g * cosTheta;
  return (1.0F - g * g) / (4.0F * 3.14159265F * d * sqrt(d));
}

// What a metre of the sea sends on of light that meets it at `cosTheta` to the way it is sent, per
// steradian: its molecules' share and its particles'
float3 seaScattered(float cosTheta)
{
  return GROUND_WATER_SCATTERING * groundWaterPhase(cosTheta) + float3(1.0F, 1.0F, 1.0F) * (SEA_PARTICLES * seaParticlePhase(cosTheta));
}

//-----------------------------------------------------------------------
// The water there is: the planet's sea when one stands, the ground's otherwise
//-----------------------------------------------------------------------

bool waterUnder(float3 pos, SceneFrameInfo* frameInfo)
{
  if(worldHasSea(frameInfo))
    return seaUnder(worldSea(frameInfo), pos);
  return groundUnderWater(pos, frameInfo);
}

float3 waterShadowTransmittance(float3 origin, float3 direction, float tMax, SceneFrameInfo* frameInfo)
{
  if(worldHasSea(frameInfo))
    return seaShadowTransmittance(worldSea(frameInfo), origin, direction, tMax);
  return groundWaterShadowTransmittance(origin, direction, tMax, frameInfo);
}

float3 waterExtinction(SceneFrameInfo* frameInfo)
{
  if(worldHasSea(frameInfo))
    return SEA_EXTINCTION;
  return GROUND_WATER_EXTINCTION;
}

#endif  // WORLD_SEA_H_SLANG
