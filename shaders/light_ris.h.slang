#ifndef FV_LIGHT_RIS_H
#define FV_LIGHT_RIS_H

// Resampled importance sampling of a vertex's light sample (Talbot, Cline and Egbert 2005: the
// candidate stage of ReSTIR, Bitterli et al. 2020, without its reuse). The vertex draws several light
// samples the usual way (sampleLights: the pick by estimate, the light tree, a triangle), weighs each
// by what it would give the pixel unshadowed, keeps one in proportion to that, and traces a shadow
// ray for it alone. Among dozens of emitters the pick's estimates know neither the BSDF nor which
// way the surface faces a lamp's own glow, so most single samples land on light the surface barely
// takes; the kept candidate is one it does.
//
// Unbiased, with any mix of point, sized, area and environment samples: a candidate's weight is the
// luminance of its own one-sample estimate (radiance over the pdf it was drawn with, times the BSDF,
// times its MIS weight), which is the target over the source pdf of generalized RIS (Lin et al. 2022)
// in the measure the sample was drawn in. The kept estimate is scaled by the candidates' mean weight
// over its own, so its expectation is the mean of the candidates' estimates, each unbiased. Visibility
// is not in the weights: a kept candidate that is shadowed gives nothing, as a single sample does.
//
// The MIS weights stay the single sample's (the pick's pdf against the BSDF's): they split the
// integrand between the two techniques whatever density each draws with, so the BSDF ray's weight on
// the far side is unchanged. Slang and C++ (tests/test_light_ris.cpp), so the tests run the shader's
// arithmetic.

static const int LIGHT_RIS_MAX_CANDIDATES = 16;

struct LightReservoir
{
  float weightSum;  // the candidates' weights so far
  float kept;       // the weight of the candidate kept (0: none yet)
  int   count;      // candidates seen, those of weight 0 included: they count in the mean
  bool  keptLast;   // the candidate just added replaced the one kept
};

LightReservoir lightReservoirEmpty()
{
  LightReservoir r;
  r.weightSum = 0.0F;
  r.kept      = 0.0F;
  r.count     = 0;
  r.keptLast  = false;
  return r;
}

// `r` with a candidate of weight `weight` seen, `u` uniform in [0, 1). The candidate is kept with
// probability weight / weightSum, which leaves every candidate so far kept in proportion to its
// weight. A weight that is not finite, as a sample with NaN in it, is never kept and adds nothing.
LightReservoir lightReservoirAdd(LightReservoir r, float weight, float u)
{
  r.count += 1;
  r.keptLast = false;
  if(!(weight > 0.0F) || !(weight < 3.0e38F))
    return r;
  r.weightSum += weight;
  if(u * r.weightSum < weight)
  {
    r.kept     = weight;
    r.keptLast = true;
  }
  return r;
}

// What the kept candidate's estimate is multiplied by: the candidates' mean weight over its own.
// 0 when no candidate had any weight (the vertex takes no light sample).
float lightReservoirScale(LightReservoir r)
{
  if(!(r.kept > 0.0F) || r.count <= 0)
    return 0.0F;
  return r.weightSum / (float(r.count) * r.kept);
}

#endif
