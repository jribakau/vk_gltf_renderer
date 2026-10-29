// An accumulation's per-pixel stop (ePtAdaptiveStop): whether a pixel's mean has converged. Compiles
// as C++ for tests/test_adaptive_stop.cpp.
#ifndef ADAPTIVE_STOP_H
#define ADAPTIVE_STOP_H

// Cycles' rule: the mean's standard error at most `threshold` times the root of the pixel's luminance,
// so dark pixels stop sooner, as their noise shows less. `second`: the mean of the samples' luminance
// squared; `mean`: the mean luminance; at least `minSamples`, so a rare bright path has had its chance.
// Decided on the samples themselves, so very slightly biased.
inline bool adaptiveConverged(float second, float mean, int samples, float threshold, int minSamples)
{
  if(samples < minSamples || samples < 2)
    return false;
  const float spread   = second - mean * mean;
  const float variance = (spread > 0.0f ? spread : 0.0f) * float(samples) / float(samples - 1);
  // the standard error squared, variance / samples, against threshold^2 * mean
  return variance <= threshold * threshold * float(samples) * (mean > 1e-4f ? mean : 1e-4f);
}

// The threshold in the scene's units for one in displayed brightness: the tone mapper multiplies by the
// exposure first, so a dark scene brought up by it converges to the same visible noise
inline float adaptiveSceneThreshold(float threshold, float exposure)
{
  return threshold / sqrt(exposure > 1e-4f ? exposure : 1e-4f);
}

#endif  // ADAPTIVE_STOP_H
