// The still's per-pixel Keep detail (still_keep.slang, StillKeep in src/still_keep.hpp): where a pixel of
// the accumulation has already converged, it is shown as it is, and the denoiser's image only where it
// is still noisy, so the denoiser cannot smooth or shift what the samples have settled. Against
// 2048-sample references it never did worse than RR alone and halved a figure's error in daylight
// (docs/denoising.md, the still's model). Compiles as C++ for tests/test_still_keep.cpp.
#ifndef STILL_KEEP_H
#define STILL_KEEP_H

// Below this many samples a pixel's spread is too uncertain to trust (a rare bright path not yet
// drawn): the denoised image alone
#define STILL_KEEP_MIN_SAMPLES 16
// Where the share falls from 1 to 0, as multiples of the threshold
#define STILL_KEEP_FROM 0.5f
#define STILL_KEEP_TO 1.5f

// A pixel's noise, squared: its mean's standard error over the root of its brightness, the adaptive
// stop's measure (adaptiveConverged), so a dark pixel's noise counts for less as it shows less. From
// `second`, the mean of the samples' luminance squared, `mean`, the mean luminance, and `samples`.
// Squared so neighbours can be averaged as variances are. Negative with too few samples to tell.
inline float stillKeepNoise(float second, float mean, int samples)
{
  if(samples < STILL_KEEP_MIN_SAMPLES)
    return -1.0f;
  const float spread   = second - mean * mean;
  const float variance = (spread > 0.0f ? spread : 0.0f) * float(samples) / float(samples - 1);
  return variance / float(samples) / (mean > 1e-4f ? mean : 1e-4f);
}

// The share of the raw accumulation a pixel keeps, from its noise squared (stillKeepNoise, averaged
// over the pixel and its neighbours) against `threshold` (the noise in the scene's units, the
// adaptive stop's adaptiveSceneThreshold of the setting): 1 at half the threshold or less, 0 at one and
// a half times it or more, and 0 with too few samples or the threshold 0 (off).
inline float stillKeepShare(float noiseSquared, float threshold)
{
  if(threshold <= 0.0f || noiseSquared < 0.0f)
    return 0.0f;
  const float noise = sqrt(noiseSquared);
  const float t     = (STILL_KEEP_TO * threshold - noise) / ((STILL_KEEP_TO - STILL_KEEP_FROM) * threshold);
  return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

#endif  // STILL_KEEP_H
