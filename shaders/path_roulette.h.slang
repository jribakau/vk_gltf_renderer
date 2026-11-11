#ifndef FV_PATH_ROULETTE_H
#define FV_PATH_ROULETTE_H

// Russian roulette on a path's next surface bounce: past its first few vertices a path goes on with
// the chance of the light it still carries (its throughput's largest channel, plus a floor), at most
// a cap, and the paths that go on are weighed up by one over that chance. Unbiased: what the paths
// that end would have brought, the survivors bring. Slang and C++ (tests/test_path_roulette.cpp).
//
// Where it starts, as surface events counted from the camera ray's hit (0): the third, on a still
// and on most of a DLSS frame's vertices. On a DLSS frame's cut-out surfaces (alpha-masked, not
// hair: a leaf, a twig, a blade of grass) at `PathtracePushConstant::cutoutRouletteFromDlss`, the
// first by default: a ray off a fir's needles, which reflect a sixth of the light, went on through
// the canopy, and the next one too, before anything could end it, and those bounces were two thirds
// of a moving frame in a forest (docs/RENDERING_ARCHITECTURE.md, the backdrop). Not everywhere:
// from the first bounce on every surface a night club's moving frame was a third cheaper too, but
// Ray Reconstruction holds back the few survivors that carry a lamp's bounced light, and its shadows
// came out 5 to 12% darker than the still's. Outdoors, where the sun lights directly, it was not.
static const int   PATH_ROULETTE_FROM_STILL       = 3;
static const int   PATH_ROULETTE_FROM_CUTOUT_DLSS = 1;
static const float PATH_ROULETTE_FLOOR      = 0.001F;

// The chance a path goes on, carrying `carried` (its throughput's largest channel), at most `cap`
inline float pathContinueChance(float carried, float cap)
{
  const float chance = carried + PATH_ROULETTE_FLOOR;
  return chance < cap ? chance : cap;
}

#endif
