#ifndef CUTOUT_CELL_H
#define CUTOUT_CELL_H

// A far cut-out read a cell of its texels at a time while DLSS denoises the frame
// (surface_opacity.h.slang, cutoutCellUv): which of a map's texels a hit reads. Slang and C++ alike
// (tests/test_cutout_cell.cpp).

#include "hash.h.slang"

// A cell's hash: its place among the cells of its size, and the size (a power of two, its log2)
uint cutoutCellHash(int x, int y, int level)
{
  return lowbias32(lowbias32(lowbias32(uint(x)) ^ uint(y)) ^ uint(level));
}

// The texel a hit at `texel` (in the map's texels, fractional) reads, its centre, for a ray whose
// footprint there covers `texels` of them across. Under one, where it falls. Wider, the map is taken
// a square cell at a time, a power of two across: the one at or under the footprint, or the one past
// it, which of the two by a hash of the wider cell against how far between them the footprint is, so
// a card goes from one to the next a cell at a time as it nears or recedes. A cell reads one of its
// texels: one of its four quarters picked by its hash, then a quarter of that, down to a texel. So
// every texel of a map is as likely to stand for its cell (a pattern that repeats every so many
// texels is kept at its share, never at a power of two's alias of it), and a cell reads what one of
// its quarters does: the cells that give way to a wider one change a quarter at a time and never all.
float2 cutoutCellTexel(float2 texel, float texels)
{
  if(!(texels >= 1.0F))  // (a NaN too)
    return texel;
  const float x     = log2(texels);
  const float wider = floor(x) + 1.0F;
  const float span  = exp2(wider);
  const int   bx    = int(floor(texel.x / span));
  const int   by    = int(floor(texel.y / span));
  const float h     = float(cutoutCellHash(bx, by, int(wider)) >> 8u) * (1.0F / 16777216.0F);
  int         level = h < x - floor(x) ? int(wider) : int(wider) - 1;
  const float cell  = exp2(float(level));
  int         ax    = int(floor(texel.x / cell));
  int         ay    = int(floor(texel.y / cell));
  for(; level > 0; level--)
  {
    const uint q = cutoutCellHash(ax, ay, level);
    ax           = ax * 2 + int(q & 1u);
    ay           = ay * 2 + int((q >> 1u) & 1u);
  }
  return float2(float(ax) + 0.5F, float(ay) + 0.5F);
}

#endif
