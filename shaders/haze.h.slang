/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Haze: a thin, even medium in the air, lit by every light of the scene, on the camera's segment
// only (between the eye and what its ray hits, or the haze's edge where it hits nothing). Light from
// later bounces reaches the eye through that same segment, so the path's radiance is dimmed by its
// transmittance and the light the haze scatters toward the eye is added. Light on its way to a
// surface, a mirror's or a window's view, is not hazed (docs/falcor-ideas.md, the air as its own
// medium, is that).
//
// The haze fills a sphere around the scene, its lights and the eye: a beam in front of the sky
// shows, and the sky is seen through as much haze as the scene holds, not through an endless fog.
//
// One scatter point a camera path, drawn half by the transmittance from where the ray enters the
// haze and half equiangularly toward a guide point (Kulla and Fajardo 2012): a point on a light
// drawn from somewhere on the segment. The light the scatter point is lit by is drawn there, as a
// scatter of the random walk draws it (volumeScatterNEE), so a spot's beam is found wherever it
// crosses the ray. The mixture's density is the pdf, which keeps it unbiased whichever light the
// guide was (one-sample MIS, the balance heuristic, given the guide). Equiangular sampling takes the
// 1/r^2 out of a point light's halo: a ray passing a light at 5 cm on a 10 m segment gets a
// noise 0.9 times its mean a sample, 5.7 times by the transmittance alone (tests/test_haze.cpp).
// A still's point picks its light from several candidates by what each gives the eye, the phase
// function's share included (HazeReservoir), and cuts its rare bright samples to the firefly clamp;
// a moving frame keeps the light pick's one sample for Ray Reconstruction.
//
// Compiled as C++ for its test (tests/test_haze.cpp).

#ifndef HAZE_H
#define HAZE_H

// Meteorological visibility: where a black object's contrast against the horizon falls to 2%
// (Koschmieder), so the extinction is -ln(0.02) / V. 0 or less: no haze.
float hazeExtinctionFromVisibility(float visibility)
{
  return visibility > 0.0F ? 3.912023F / visibility : 0.0F;
}

// 1 - e^-x for x >= 0, without the cancellation near 0 (a series below 0.1, within 2e-8 there)
float hazeOneMinusExp(float x)
{
  if(x < 0.1F)
    return x * (1.0F - x * 0.5F * (1.0F - x * (1.0F / 3.0F) * (1.0F - x * 0.25F * (1.0F - x * 0.2F))));
  return 1.0F - exp(-x);
}

// -ln(1 - z) for z in [0, 1), as carefully
float hazeMinusLogOneMinus(float z)
{
  if(z < 0.1F)
    return z * (1.0F + z * (0.5F + z * (1.0F / 3.0F + z * (0.25F + z * (0.2F + z * (1.0F / 6.0F + z * (1.0F / 7.0F)))))));
  return -log(1.0F - z);
}

// The stretch [x, y] of the ray o + t d (d unit), t in [0, tEnd], inside the sphere of centre c and
// radius r; y <= x when it has none.
float2 hazeSegment(float3 o, float3 d, float tEnd, float3 c, float r)
{
  const float3 oc   = o - c;
  const float  b    = dot(oc, d);
  const float  disc = b * b - (dot(oc, oc) - r * r);
  if(!(disc > 0.0F))
    return float2(0.0F, 0.0F);
  const float root = sqrt(disc);
  return float2(max(-b - root, 0.0F), min(-b + root, tEnd));
}

// How far a ray from x, inside that sphere, travels along d (unit) before it leaves the haze
float hazeExitDistance(float3 x, float3 d, float3 c, float r)
{
  const float3 xc = x - c;
  const float  b  = dot(xc, d);
  return max(-b + sqrt(max(b * b - (dot(xc, xc) - r * r), 0.0F)), 0.0F);
}

// The share of scatter points drawn toward the guide, when there is one
static const float HAZE_GUIDE_SHARE = 0.5F;

// The equiangular half: where on the ray the guide point is closest (delta, from o), how far from
// the ray it is there (h), and the angles of the segment's ends as seen from the guide
struct HazeGuide
{
  float delta;
  float h;
  float thetaA;
  float thetaB;
  bool  on;  // false: no guide (a light at infinity), every point by the transmittance
};

HazeGuide hazeGuide(float3 o, float3 d, float t0, float t1, float3 guidePoint, bool on)
{
  HazeGuide g;
  g.on    = on;
  g.delta = dot(guidePoint - o, d);
  // a light on the ray itself (a lamp the camera looks straight at) still has its angles
  g.h      = max(length(guidePoint - (o + g.delta * d)), 1e-4F);
  g.thetaA = atan2(t0 - g.delta, g.h);
  g.thetaB = atan2(t1 - g.delta, g.h);
  g.on     = on && g.thetaB > g.thetaA;
  return g;
}

HazeGuide hazeNoGuide()
{
  HazeGuide g;
  g.on     = false;
  g.delta  = 0.0F;
  g.h      = 1.0F;
  g.thetaA = 0.0F;
  g.thetaB = 0.0F;
  return g;
}

// A distance on [t0, t1] drawn by the transmittance from t0, sigma the extinction
float hazeSampleTransmittance(float u, float sigma, float t0, float t1)
{
  const float t = t0 + hazeMinusLogOneMinus(u * hazeOneMinusExp(sigma * (t1 - t0))) / sigma;
  return clamp(t, t0, t1);
}

// A distance on [t0, t1] drawn equiangularly toward the guide (g.on)
float hazeSampleEquiangular(float u, float t0, float t1, HazeGuide g)
{
  return clamp(g.delta + g.h * tan(g.thetaA + u * (g.thetaB - g.thetaA)), t0, t1);
}

// The density hazeSampleEquiangular draws t with
float hazeEquiangularPdf(float t, HazeGuide g)
{
  const float s = t - g.delta;
  return g.h / ((g.thetaB - g.thetaA) * (g.h * g.h + s * s));
}

// The scatter point's distance: `choice` picks the half, `u` places it
float hazeSampleDistance(float u, float choice, float sigma, float t0, float t1, HazeGuide g)
{
  if(g.on && choice < HAZE_GUIDE_SHARE)
    return hazeSampleEquiangular(u, t0, t1, g);
  return hazeSampleTransmittance(u, sigma, t0, t1);
}

// The density hazeSampleDistance draws t with
float hazeDistancePdf(float t, float sigma, float t0, float t1, HazeGuide g)
{
  const float byTransmittance = sigma * exp(-sigma * (t - t0)) / hazeOneMinusExp(sigma * (t1 - t0));
  if(!g.on)
    return byTransmittance;
  const float s           = t - g.delta;
  const float equiangular = g.h / ((g.thetaB - g.thetaA) * (g.h * g.h + s * s));
  return (1.0F - HAZE_GUIDE_SHARE) * byTransmittance + HAZE_GUIDE_SHARE * equiangular;
}

// What a scatter point at t carries of the light it is lit by in the direction of the eye, before
// the phase function: the haze's scattering there (its albedo is 1, so the extinction), the
// transmittance back to where the ray entered it, over the density t was drawn with. Drawn by the
// transmittance alone this is 1 - e^(-sigma (t1 - t0)), what the segment scatters in all.
float hazeScatterWeight(float t, float sigma, float t0, float t1, HazeGuide g)
{
  return sigma * exp(-sigma * (t - t0)) / hazeDistancePdf(t, sigma, t0, t1, g);
}

// The scatter point's light, chosen among several drawn there (resampled importance sampling, as
// Talbot's): each candidate is a light sample of the light pick, weighed by what it would give the
// eye -- its light over its pdf, times the phase function toward it and the haze on the way -- and one
// is kept with the chance its weight has of their sum. The pick weighs lights by their power and
// distance only; at a glow of 0.7 the phase function sends 180 times more light on than back, so a
// lamp ahead of the eye counts far more than one behind it, which only the weights see. The kept
// candidate's light then counts for all of them: its light over its pdf, times (sum of the weights)
// / (candidates x its weight). One shadow ray, for the kept one. A candidate of weight 0 counts as
// one of them. Unbiased whatever the weights, as long as a light that gives anything weighs above 0.
struct HazeReservoir
{
  float sum;    // of the weights so far
  float kept;   // the kept candidate's weight; 0: none yet
  bool  taken;  // the last candidate seen is now the kept one
};

HazeReservoir hazeReservoir()
{
  HazeReservoir r;
  r.sum   = 0.0F;
  r.kept  = 0.0F;
  r.taken = false;
  return r;
}

// The reservoir once it has seen a candidate of weight w, `taken` if that one is now the kept one;
// `u` in [0, 1), read only when there is one to take the place of (hazeReservoirNeedsDraw), so a
// single candidate draws no number
bool hazeReservoirNeedsDraw(HazeReservoir r, float w)
{
  return w > 0.0F && r.sum > 0.0F;
}

HazeReservoir hazeReservoirTake(HazeReservoir r, float w, float u)
{
  r.taken = false;
  if(!(w > 0.0F))
    return r;
  r.sum += w;
  if(r.kept > 0.0F && u * r.sum >= w)
    return r;
  r.kept  = w;
  r.taken = true;
  return r;
}

// What the kept candidate's light over its pdf is multiplied by, of `count` candidates; 0 for none
float hazeReservoirWeight(HazeReservoir r, int count)
{
  return r.kept > 0.0F ? r.sum / (float(count) * r.kept) : 0.0F;
}

//-----------------------------------------------------------------------
// Uneven haze: thicker near the floor, in patches
//
// The density relative to the extinction at the floor (hazeExtinction): exp(-(height above the
// floor) / H), 1 at and below the floor, times 1 + patches x a smooth noise in [-1, 1] (so 0 to 2
// times it, the same on average). The height's part is integrated exactly along a ray
// (hazeHeightDepth) and a distance drawn by it exactly (hazeHeightInvert), so a layer of fog at the
// floor seen from above draws its points inside it. The patches dim the camera's segment by four
// stratified lookups spread evenly over the haze it crosses (HazeColumn, an estimate of the optical
// depth that is unbiased, its exponential a hair high where the lookups differ), and the scatter
// point takes the density where it lands (hazeDensity); a light's way to the point is dimmed by the
// height alone.
//-----------------------------------------------------------------------

struct HazeShape
{
  float floorY;        // the floor's height
  float invHeight;     // 1 / H; 0: even through the height
  float patches;       // 0 to 1: how far the density swings with the noise
  float invPatchSize;  // the noise's frequency, per metre
};

bool hazeEven(HazeShape s)
{
  return !(s.invHeight > 0.0F) && !(s.patches > 0.0F);
}

// (1 - e^-x) / x for x >= 0, 1 at 0
float hazeOneMinusExpOver(float x)
{
  return x > 0.0F ? hazeOneMinusExp(x) / x : 1.0F;
}

// The height's factor at height y
float hazeHeightFactor(HazeShape s, float y)
{
  return s.invHeight > 0.0F ? exp(-max(y - s.floorY, 0.0F) * s.invHeight) : 1.0F;
}

// The integral of the height's factor along o + t d (d unit) from ta to tb: the stretch below the
// floor at 1, the rest e^(-z/H) over z rising or falling linearly, from its lower end so nothing
// overflows
float hazeHeightDepth(HazeShape s, float3 o, float3 d, float ta, float tb)
{
  const float len = max(tb - ta, 0.0F);
  if(!(s.invHeight > 0.0F) || !(len > 0.0F))
    return len;
  const float za = o.y + d.y * ta - s.floorY;
  const float zb = o.y + d.y * tb - s.floorY;
  if(za <= 0.0F && zb <= 0.0F)
    return len;
  float below = 0.0F;
  float zLo   = min(za, zb);
  float zHi   = max(za, zb);
  if(zLo < 0.0F)
  {
    // it crosses the floor (d.y is not 0: the ends' heights differ in sign)
    const float tf = (s.floorY - o.y) / d.y;
    below          = za < 0.0F ? tf - ta : tb - tf;
    below          = clamp(below, 0.0F, len);
    zLo            = 0.0F;
  }
  const float above = len - below;
  return below + exp(-zLo * s.invHeight) * above * hazeOneMinusExpOver((zHi - zLo) * s.invHeight);
}

// The length from a point at height z (above the floor) along a ray rising k per unit (k = d.y)
// over which the height's factor integrates to tau, at most `most`
float hazeExponentialInvert(float z, float k, float tau, float invHeight, float most)
{
  if(!(tau > 0.0F))
    return 0.0F;
  const float H = 1.0F / invHeight;
  const float A = exp(-z * invHeight);  // the factor where it starts
  float       len;
  if(abs(k) * most * invHeight < 1e-4F)
    len = A > 0.0F ? tau / A : most;  // all but level
  else if(k > 0.0F)
  {
    // rising: the factor falls, tau = (H / k) A (1 - e^(-k len / H))
    const float q = A > 0.0F ? tau * k * invHeight / A : 1.0F;
    len           = q < 1.0F ? H / k * hazeMinusLogOneMinus(q) : most;
  }
  else
  {
    // falling: tau = (H / |k|) A (e^(|k| len / H) - 1)
    const float m = -k;
    const float p = A > 0.0F ? tau * m * invHeight / A : 1e30F;
    len           = p < 1e4F ? H / m * hazeMinusLogOneMinus(p / (1.0F + p)) : (H * log(A + tau * m * invHeight) + z) / m;
  }
  return clamp(len, 0.0F, most);
}

// The distance t in [ta, tb] at which hazeHeightDepth from ta reaches tau (0 to its whole)
float hazeHeightInvert(HazeShape s, float3 o, float3 d, float ta, float tb, float tau)
{
  const float len = max(tb - ta, 0.0F);
  if(!(s.invHeight > 0.0F))
    return ta + clamp(tau, 0.0F, len);
  const float za = o.y + d.y * ta - s.floorY;
  const float zb = o.y + d.y * tb - s.floorY;
  if(za <= 0.0F && zb <= 0.0F)
    return ta + clamp(tau, 0.0F, len);
  if(za < 0.0F)
  {
    // below the floor first, then rising out of it
    const float tf = clamp((s.floorY - o.y) / d.y, ta, tb);
    if(tau <= tf - ta)
      return ta + max(tau, 0.0F);
    return tf + hazeExponentialInvert(0.0F, d.y, tau - (tf - ta), s.invHeight, tb - tf);
  }
  if(zb < 0.0F)
  {
    // falling to the floor, then below it
    const float tf    = clamp((s.floorY - o.y) / d.y, ta, tb);
    const float upper = hazeHeightDepth(s, o, d, ta, tf);
    if(tau <= upper)
      return ta + hazeExponentialInvert(za, d.y, tau, s.invHeight, tf - ta);
    return tf + clamp(tau - upper, 0.0F, tb - tf);
  }
  return ta + hazeExponentialInvert(za, d.y, tau, s.invHeight, len);
}

// A smooth noise in [-1, 1]: two octaves of value noise on an integer lattice, quintic between
float hazeHash(int x, int y, int z)
{
  uint h = uint(x) * 0x8da6b343u ^ uint(y) * 0xd8163841u ^ uint(z) * 0xcb1ab31fu;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return float(h >> 8) * (1.0F / 16777216.0F);
}

float hazeValueNoise(float3 p)
{
  const float3 c = floor(p);
  const float3 f = p - c;
  const float3 w = f * f * f * (f * (f * 6.0F - 15.0F) + 10.0F);
  const int    x = int(c.x);
  const int    y = int(c.y);
  const int    z = int(c.z);
  const float  a = hazeHash(x, y, z) + (hazeHash(x + 1, y, z) - hazeHash(x, y, z)) * w.x;
  const float  b = hazeHash(x, y + 1, z) + (hazeHash(x + 1, y + 1, z) - hazeHash(x, y + 1, z)) * w.x;
  const float  e = hazeHash(x, y, z + 1) + (hazeHash(x + 1, y, z + 1) - hazeHash(x, y, z + 1)) * w.x;
  const float  g = hazeHash(x, y + 1, z + 1) + (hazeHash(x + 1, y + 1, z + 1) - hazeHash(x, y + 1, z + 1)) * w.x;
  const float  near = a + (b - a) * w.y;
  const float  far  = e + (g - e) * w.y;
  return near + (far - near) * w.z;
}

float hazeNoise(float3 p)
{
  return 2.0F * (0.65F * hazeValueNoise(p) + 0.35F * hazeValueNoise(p * 2.03F + float3(17.1F, 9.3F, 41.7F))) - 1.0F;
}

// The patches' factor at x: 1 + patches x noise, 0 to 2
float hazePatchFactor(HazeShape s, float3 x)
{
  return s.patches > 0.0F ? max(1.0F + s.patches * hazeNoise(x * s.invPatchSize), 0.0F) : 1.0F;
}

// The density at x relative to the extinction at the floor
float hazeDensity(HazeShape s, float3 x)
{
  return hazeHeightFactor(s, x.y) * hazePatchFactor(s, x);
}

// The camera's segment as the uneven haze is integrated over it: the height's depth, split in four
// strata of equal depth, each with the patches' factor at a point of it (the same jitter in each)
static const int HAZE_STRATA = 4;

struct HazeColumn
{
  float depth;    // hazeHeightDepth over the segment
  float factor0;  // the patches' factor in each stratum (four fields: an array in this struct
  float factor1;  // costs the shader local memory)
  float factor2;
  float factor3;
  float total;    // the segment's optical depth over the extinction: depth x their mean
};

float hazeColumnFactor(HazeColumn c, int i)
{
  return i == 0 ? c.factor0 : (i == 1 ? c.factor1 : (i == 2 ? c.factor2 : c.factor3));
}

HazeColumn hazeColumn(HazeShape s, float3 o, float3 d, float t0, float t1, float jitter)
{
  HazeColumn c;
  c.depth   = hazeHeightDepth(s, o, d, t0, t1);
  c.factor0 = 1.0F;
  c.factor1 = 1.0F;
  c.factor2 = 1.0F;
  c.factor3 = 1.0F;
  if(s.patches > 0.0F)
  {
    const float q = c.depth / float(HAZE_STRATA);
    c.factor0     = hazePatchFactor(s, o + d * hazeHeightInvert(s, o, d, t0, t1, (0.0F + jitter) * q));
    c.factor1     = hazePatchFactor(s, o + d * hazeHeightInvert(s, o, d, t0, t1, (1.0F + jitter) * q));
    c.factor2     = hazePatchFactor(s, o + d * hazeHeightInvert(s, o, d, t0, t1, (2.0F + jitter) * q));
    c.factor3     = hazePatchFactor(s, o + d * hazeHeightInvert(s, o, d, t0, t1, (3.0F + jitter) * q));
  }
  c.total = c.depth * 0.25F * (c.factor0 + c.factor1 + c.factor2 + c.factor3);
  return c;
}

// The column's optical depth (over the extinction) up to where the height's depth is tauH
float hazeColumnDepthTo(HazeColumn c, float tauH)
{
  const float q   = c.depth / float(HAZE_STRATA);
  float       acc = 0.0F;
  for(int i = 0; i < HAZE_STRATA; i++)
    acc += hazeColumnFactor(c, i) * clamp(tauH - float(i) * q, 0.0F, q);
  return acc;
}

// The patches' factor of the stratum the height's depth tauH falls in
float hazeColumnFactorAt(HazeColumn c, float tauH)
{
  const float q = c.depth / float(HAZE_STRATA);
  return hazeColumnFactor(c, q > 0.0F ? clamp(int(tauH / q), 0, HAZE_STRATA - 1) : 0);
}

// The height's depth at which the column's optical depth reaches tauHat (0 to c.total)
float hazeColumnHeightDepthAt(HazeColumn c, float tauHat)
{
  const float q   = c.depth / float(HAZE_STRATA);
  float       acc = 0.0F;
  for(int i = 0; i < HAZE_STRATA; i++)
  {
    const float f    = hazeColumnFactor(c, i);
    const float next = acc + f * q;
    if(tauHat <= next && f > 0.0F)
      return float(i) * q + (tauHat - acc) / f;
    acc = next;
  }
  return c.depth;
}

// A distance on the camera's segment drawn by the column's transmittance (the uneven haze's
// hazeSampleTransmittance), `u` in [0, 1)
float hazeColumnSampleTransmittance(HazeShape s, HazeColumn c, float3 o, float3 d, float sigma, float t0, float t1, float u)
{
  const float tauHat = hazeSampleTransmittance(u, sigma, 0.0F, c.total);
  return hazeHeightInvert(s, o, d, t0, t1, hazeColumnHeightDepthAt(c, tauHat));
}

// The density the uneven haze's mixture draws t with (hazeDistancePdf's)
float hazeColumnDistancePdf(HazeShape s, HazeColumn c, float3 o, float3 d, float sigma, float t0, float t, HazeGuide g)
{
  const float tauH            = hazeHeightDepth(s, o, d, t0, t);
  const float slope           = sigma * hazeHeightFactor(s, o.y + d.y * t) * hazeColumnFactorAt(c, tauH);
  const float byTransmittance = slope * exp(-sigma * hazeColumnDepthTo(c, tauH)) / hazeOneMinusExp(sigma * c.total);
  if(!g.on)
    return byTransmittance;
  return (1.0F - HAZE_GUIDE_SHARE) * byTransmittance + HAZE_GUIDE_SHARE * hazeEquiangularPdf(t, g);
}

// The uneven haze's hazeScatterWeight: the density where the point lands (with its patches) and the
// column's transmittance to it, over the density it was drawn with
float hazeColumnScatterWeight(HazeShape s, HazeColumn c, float3 o, float3 d, float sigma, float t0, float t, HazeGuide g)
{
  const float pdf = hazeColumnDistancePdf(s, c, o, d, sigma, t0, t, g);
  if(!(pdf > 0.0F))
    return 0.0F;
  const float tauHat = hazeColumnDepthTo(c, hazeHeightDepth(s, o, d, t0, t));
  return sigma * hazeDensity(s, o + d * t) * exp(-sigma * tauHat) / pdf;
}

#endif  // HAZE_H
