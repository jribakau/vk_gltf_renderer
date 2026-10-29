#ifndef FV_WALK_GUIDE_H
#define FV_WALK_GUIDE_H

// Dwivedi guiding for a scatter walk entered from a surface (ePtWalkGuiding), as Cycles' random walk
// does it (Meng, Hanika and Dachsbacher 2016, "Improving the Dwivedi Sampling Scheme"). In a dense,
// bright medium most of what enters comes back out near where it went in, but a walk that samples its
// phase function wanders deep and takes hundreds of events doing it. A guided step instead draws its
// direction towards the entry surface (the entry's normal, facing out of the medium) and stretches its
// free flight along it, so walks come back out sooner and with less spread in their weights.
//
// Unbiased: each step after a scatter is drawn guided or classic (the phase function and the medium's
// own free flight) at random, and both its direction and its distance are weighed against the two
// techniques together (the balance heuristic) and against the three channels' distance pdfs.
// Slang and C++ (tests/test_walk_guide.cpp), so the tests run the arithmetic the shader runs.

static const float WALK_GUIDE_TWO_PI = 6.28318530718F;
// A diffusion length past this is a medium that barely absorbs: the guided directions are nearly
// uniform there, so guiding would only add noise.
static const float WALK_GUIDE_MAX_LENGTH = 1.0e4F;

struct WalkGuide
{
  float diffusionLength;  // v, in mean free paths: the guided directions' pdf goes as 1 / (v - cos)
  float phaseLog;         // log((v + 1) / (v - 1)), the pdf's normalisation
  float fraction;         // the share of steps drawn guided; 0 is no guiding
};

// The guide for a medium whose largest channel has single-scattering albedo `albedo` and whose
// phase function has anisotropy `g`. The largest albedo, as Cycles takes: too low a one guides too
// hard and gives fireflies, too high a one only some noise. The fraction is Cycles' heuristic, half
// the steps for an isotropic medium and fewer the more forward or backward it scatters.
WalkGuide makeWalkGuide(float albedo, float g)
{
  WalkGuide w;
  w.diffusionLength = 0.0F;
  w.phaseLog        = 0.0F;
  w.fraction        = 0.0F;
  if(!(albedo > 0.0F))
    return w;
  // Meng et al., eq. 67, a fit to the exact diffusion length of a half-space
  const float p = pow(min(albedo, 1.0F), 2.44294F - 0.0215813F * albedo + 0.578637F / albedo);
  const float v = 1.0F / sqrt(max(1.0F - p, 0.0F));
  if(!(v > 1.0F + 1.0e-4F) || !(v < WALK_GUIDE_MAX_LENGTH))
    return w;
  w.diffusionLength = v;
  w.phaseLog        = log((v + 1.0F) / (v - 1.0F));
  w.fraction        = 1.0F - max(0.5F, pow(abs(g), 0.125F));
  return w;
}

// The cosine to the entry normal of a guided direction, from a uniform `u` in [0, 1)
float walkGuideSampleCos(WalkGuide w, float u)
{
  const float v = w.diffusionLength;
  return clamp(v - (v + 1.0F) * exp(-u * w.phaseLog), -1.0F, 1.0F);
}

// The guided directions' pdf over solid angle, at `cosTheta` to the entry normal
float walkGuidePdf(WalkGuide w, float cosTheta)
{
  return 1.0F / (WALK_GUIDE_TWO_PI * (w.diffusionLength - cosTheta) * w.phaseLog);
}

// A guided step's extinction over the medium's, along a direction at `cosTheta` to the entry normal:
// below 1 towards the surface, so the step goes further that way, above 1 away from it.
float walkGuideStretch(WalkGuide w, float cosTheta)
{
  return 1.0F - cosTheta / w.diffusionLength;
}

// The pdf of one step's outcome, per channel: a scatter after distance `t` (`reached` false), or
// reaching the surface at `t` with no scatter on the way. Classic distance sampling has
// sigma_t exp(-sigma_t t), and exp(-sigma_t t) for the surface. With guiding (`fraction` above 0) the
// guided step's, at sigma_t times `stretch`, joins it, weighted by `factor`: the guided direction pdf
// over the phase function's at this step's direction (the phase function's own cancels against the
// phase function in the estimate, so neither appears).
float3 walkStepPdf(float3 sigmaT, float t, bool reached, float fraction, float stretch, float factor)
{
  const float3 transmittance = exp(-sigmaT * t);
  float3       pdf           = reached ? transmittance : sigmaT * transmittance;
  if(fraction > 0.0F)
  {
    const float3 guidedSigmaT = sigmaT * stretch;
    const float3 guidedT      = exp(-guidedSigmaT * t);
    const float3 guided       = reached ? guidedT : guidedSigmaT * guidedT;
    pdf                       = pdf * (1.0F - fraction) + guided * (factor * fraction);
  }
  return pdf;
}

#endif
