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

// What a pixel's convergence image keeps beside its moments (OutputImage::eConvergence's second channel)
// while the stop is on: the samples it has taken, positive once it has converged and negative while it
// samples on. The warp's vote before a pass reads it as converged above 0; the Samples view
// (shaders/sample_map.slang) reads the count. 0: nothing written by the stop (an exact still, the Time
// view's mean). A converged pixel in a warp that samples on takes those samples too, so the count is
// what the pixel took, not when it first converged.
inline float adaptiveStateValue(bool converged, int samples)
{
  return converged ? float(samples) : -float(samples);
}
inline bool adaptiveStateConverged(float state)
{
  return state > 0.0f;
}
inline int adaptiveStateSamples(float state)
{
  return int((state < 0.0f ? -state : state) + 0.5f);
}

// The Samples view's scale, 0 to 1, by the log of the samples a pixel took: 0 at the minimum the stop
// keeps (or fewer, while a still begins), 1 at the cap. A cap at or below the minimum has a scale a
// stop wide above it.
inline float adaptiveSamplesRamp(int samples, int minSamples, int cap)
{
  const float lo = log2(float(minSamples > 1 ? minSamples : 1));
  const float hi = log2(float(cap > minSamples ? cap : (minSamples > 1 ? minSamples : 1) * 2));
  const float t  = (log2(float(samples > 1 ? samples : 1)) - lo) / (hi - lo);
  return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

// The threshold in the scene's units for one in displayed brightness: the tone mapper multiplies by the
// exposure first, so a dark scene brought up by it converges to the same visible noise
inline float adaptiveSceneThreshold(float threshold, float exposure)
{
  return threshold / sqrt(exposure > 1e-4f ? exposure : 1e-4f);
}

#endif  // ADAPTIVE_STOP_H
