#ifndef FV_BLOOM_H
#define FV_BLOOM_H

// Bloom's kernels (src/bloom.hpp, shaders/bloom.slang): their taps and weights, Slang and C++
// (tests/test_bloom.cpp), so the tests sum the kernels the shader runs.
//
// The glow is a sum of Gaussians that keeps the image's light: a pyramid of blurred copies, each half
// the size of the one before (bloomDownTap: Jimenez's 13 bilinear taps, weights summing to 1), summed
// back up from the coarsest with geometric weights (each level BLOOM_TAIL of the one before it, the
// tent of bloomTentTap between them), so the glow has a long tail; the composite moves a share of
// the image's light into it. The star: spikes across the aperture's straight edges, as a polygon's
// diffraction draws them, one line through each pair of parallel edges, so an even number of blades
// gives as many spikes and an odd number twice as many; none for a round aperture.

static const int   BLOOM_MAX_LEVELS   = 8;
static const int   BLOOM_MIN_SIZE     = 8;      // a level's smaller side at least this, in pixels
static const float BLOOM_TAIL         = 0.65F;  // each coarser level's weight over the finer one's
static const int   BLOOM_STAR_LEVEL   = 2;      // the star is drawn on the quarter-size level
static const int   BLOOM_STAR_TAPS    = 32;     // a spike's taps, each way
static const float BLOOM_STAR_LENGTH  = 0.08F;  // a spike's length, of the image's height
static const float BLOOM_STAR_FALLOFF = 4.0F;   // a spike's weight, exp(-FALLOFF x its share of the length)
// The light the glow takes in, in units of the tone mapper's white after its exposure: enough for a
// lamp to bloom, short of a sun disk's or a firefly's halo swamping the frame
static const float BLOOM_LIMIT = 4096.0F;

// The pyramid's levels below the image for a size: halving until the smaller side would go under
// BLOOM_MIN_SIZE, at most BLOOM_MAX_LEVELS
int bloomLevels(int width, int height)
{
  int levels = 0;
  int side   = min(width, height);
  while(levels < BLOOM_MAX_LEVELS && side / 2 >= BLOOM_MIN_SIZE)
  {
    side /= 2;
    levels++;
  }
  return levels;
}

// A level's side from the image's: halved `level` times, rounded up
int bloomLevelSide(int side, int level)
{
  return max(1, (side + (1 << level) - 1) >> level);
}

// Tap `i` of 13 of the downsample, in the source level's texels from the destination pixel's centre
// (each a bilinear tap: the mean of the four texels around it), and its weight
float2 bloomDownOffset(int i)
{
  if(i == 0)
    return float2(0.0F, 0.0F);
  if(i < 5)  // the inner square
    return float2((i & 1) != 0 ? -1.0F : 1.0F, (i & 2) != 0 ? -1.0F : 1.0F);
  if(i < 9)  // the outer cross
  {
    const int k = i - 5;
    return k < 2 ? float2(k == 0 ? -2.0F : 2.0F, 0.0F) : float2(0.0F, k == 2 ? -2.0F : 2.0F);
  }
  const int k = i - 9;  // the outer corners
  return float2((k & 1) != 0 ? -2.0F : 2.0F, (k & 2) != 0 ? -2.0F : 2.0F);
}
float bloomDownWeight(int i)
{
  if(i == 0)
    return 0.125F;
  if(i < 5)
    return 0.125F;
  if(i < 9)
    return 0.0625F;
  return 0.03125F;
}

// Tap `i` of 9 of the tent that reads a coarser level, in its texels, and its weight (1 2 1 / 16)
float2 bloomTentOffset(int i)
{
  return float2(float(i % 3 - 1), float(i / 3 - 1));
}
float bloomTentWeight(int i)
{
  const int x = i % 3, y = i / 3;
  return float((x == 1 ? 2 : 1) * (y == 1 ? 2 : 1)) / 16.0F;
}

// The star's lines for `blades` aperture blades: none below 3 (round), half of them when even
// (opposite edges are parallel), all of them when odd (each line two spikes)
int bloomStarLines(int blades)
{
  if(blades < 3)
    return 0;
  return (blades % 2 == 0) ? blades / 2 : blades;
}

// Line `line`'s angle, radians in the image from its right toward its top: across an edge, which
// lies between corners `rotation` + 2 pi k / blades (sampleAperture's polygon)
float bloomStarAngle(int blades, float rotation, int line)
{
  const float pi = 3.14159265358979323846F;
  return rotation + pi * float(2 * line + 1) / float(blades);
}

// A spike's tap `k` (1 to BLOOM_STAR_TAPS) weight, before the kernel is normalised
float bloomStarWeight(int k)
{
  return exp(-BLOOM_STAR_FALLOFF * float(k) / float(BLOOM_STAR_TAPS));
}

// The star kernel's sum over `lines` lines, both ways along each: dividing by it keeps the light
float bloomStarNorm(int lines)
{
  float sum = 0.0F;
  for(int k = 1; k <= BLOOM_STAR_TAPS; k++)
    sum += bloomStarWeight(k);
  return 2.0F * float(lines) * sum;
}

#endif  // FV_BLOOM_H
