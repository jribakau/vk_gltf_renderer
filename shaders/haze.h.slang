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

// The scatter point's distance: `choice` picks the half, `u` places it
float hazeSampleDistance(float u, float choice, float sigma, float t0, float t1, HazeGuide g)
{
  if(g.on && choice < HAZE_GUIDE_SHARE)
    return clamp(g.delta + g.h * tan(g.thetaA + u * (g.thetaB - g.thetaA)), t0, t1);
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

#endif  // HAZE_H
