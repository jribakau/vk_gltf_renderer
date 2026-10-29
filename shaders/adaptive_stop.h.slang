// An accumulation's per-pixel stop (ePtAdaptiveStop): whether a pixel's mean has converged. Compiles
// as C++ for tests/test_adaptive_stop.cpp.
#ifndef ADAPTIVE_STOP_H
#define ADAPTIVE_STOP_H

// Cycles' rule: the mean's standard error at most `threshold` times the root of the pixel's luminance,
// so dark pixels stop sooner, as their noise shows less. `second`: the mean of the samples' luminance
// squared; `mean`: the mean luminance; at least `minSamples`, so a rare bright path has had its chance.
// `relax` (1 or more) widens the bound where the tone mapper flattens the pixel's brightness, so its
// noise shows less there too (adaptiveRelaxPosition). Decided on the samples themselves, so very
// slightly biased.
inline bool adaptiveConverged(float second, float mean, int samples, float threshold, int minSamples, float relax)
{
  if(samples < minSamples || samples < 2)
    return false;
  const float spread   = second - mean * mean;
  const float variance = (spread > 0.0f ? spread : 0.0f) * float(samples) / float(samples - 1);
  // the standard error squared, variance / samples, against (relax * threshold)^2 * mean
  return variance <= relax * relax * threshold * threshold * float(samples) * (mean > 1e-4f ? mean : 1e-4f);
}

// The relax table (adaptive_relax.hpp): ADAPTIVE_RELAX_STOPS entries, the first at mid grey's luminance
// in the scene (`base`), each a stop brighter than the one before. Where a pixel's mean falls in it:
// -1 at or below mid grey, or with no table (base 0), where nothing is relaxed; past the last entry, the
// last.
#define ADAPTIVE_RELAX_STOPS 12
inline float adaptiveRelaxPosition(float mean, float base)
{
  if(base <= 0.0f || mean <= base)
    return -1.0f;
  const float p = log2(mean / base);
  return p < float(ADAPTIVE_RELAX_STOPS - 1) ? p : float(ADAPTIVE_RELAX_STOPS - 1);
}

// The threshold in the scene's units for one in displayed brightness: the tone mapper multiplies by the
// exposure first, so a dark scene brought up by it converges to the same visible noise
inline float adaptiveSceneThreshold(float threshold, float exposure)
{
  return threshold / sqrt(exposure > 1e-4f ? exposure : 1e-4f);
}

#endif  // ADAPTIVE_STOP_H
