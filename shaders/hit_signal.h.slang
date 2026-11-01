// What the first hit of a path wants of its material's textures: how large a one it could show. The Fit reads
// the counts (rs::ImageFit, docs/scene-memory-plan.md). Compiles as C++ for tests/test_hit_signal.cpp.
#ifndef HIT_SIGNAL_H
#define HIT_SIGNAL_H

// Per material: a long edge of 256 or less, 512, 1K, 2K, 4K, 8K or more
#define HIT_SIGNAL_BINS 6

// `uvFootprint`: what a pixel covers of a texture's 0..1, whatever ptTexGradScale makes of it for the sampling,
// so a long edge of 2 / uvFootprint gives the pixel two texels. Two: one softens what one pixel shows, and the
// pixel is the render size's, which DLSS shows larger. 0: it samples no texture.
inline int hitSignalBin(float uvFootprint)
{
  int   bin  = 0;
  float edge = 256.0f;
  while(bin < HIT_SIGNAL_BINS - 1 && uvFootprint > 0.0f && edge * uvFootprint < 2.0f)
  {
    edge *= 2.0f;
    bin++;
  }
  return bin;
}

// Which pixels count: one in sixteen, others each frame
inline bool hitSignalPixel(int x, int y, int frame)
{
  return ((x + frame) & 3) == 0 && ((y + (frame >> 2)) & 3) == 0;
}

#endif
