#ifndef FV_CONTRIBUTION_ROULETTE_H
#define FV_CONTRIBUTION_ROULETTE_H

// Russian roulette on a ray by what it can add: a term the pixel already holds `rest` of light beside
// is traced with probability `term / (share * rest)`, at most 1, and weighed by one over it. The
// estimate stays unbiased, and the noise it adds is bounded by the share: its variance is at most
// `share * term * rest`, a relative deviation of at most `share` of the pixel when term reaches the
// share. A term that is all the pixel has (rest 0) is always traced. Slang and C++
// (tests/test_contribution_roulette.cpp).

// The chance to trace a ray that adds up to `term` (a luminance) to a pixel holding `rest` besides
inline float contributionTraceChance(float term, float rest, float share)
{
  if(!(term > 0.0F))
    return 0.0F;
  if(!(rest > 0.0F) || !(share > 0.0F))
    return 1.0F;
  const float chance = term / (share * rest);
  return chance < 1.0F ? chance : 1.0F;
}

#endif
