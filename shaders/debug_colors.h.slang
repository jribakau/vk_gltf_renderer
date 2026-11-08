#ifndef FV_DEBUG_COLORS_H
#define FV_DEBUG_COLORS_H

// Colours for the Display views that show a value or an id per pixel (applyVisualization in
// common.h.slang): which material or which part of a prop a pixel is, and how large the mesh's
// triangles are there. Display values, written as they are (a debug view passes the tone mapper by).
// Slang and C++ (tests/test_debug_colors.cpp).

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// A colour of its own for an id (a material, a prop's part). Its hue steps round the circle by the
// golden ratio, so ids one apart are 0.38 of the circle apart and the first few dozen spread evenly;
// its brightness takes one of three levels in turn, which parts ids that land near in hue.
inline float3 idColor(uint id)
{
  const float t     = float(id % 65536u) * 0.6180339887F;
  const float hue   = (t - floor(t)) * 6.0F;  // round the six sides of the RGB hexagon
  const uint  third = id % 3u;
  const float level = third == 0u ? 0.95F : (third == 1u ? 0.72F : 0.5F);
  const float3 pure = float3(clamp(abs(hue - 3.0F) - 1.0F, 0.0F, 1.0F), clamp(2.0F - abs(hue - 2.0F), 0.0F, 1.0F),
                             clamp(2.0F - abs(hue - 4.0F), 0.0F, 1.0F));
  return (pure * 0.8F + float3(0.2F)) * level;  // a little white: no colour is pure, none black
}

// Viridis (matplotlib's, by Smith and van der Walt): dark violet through blue and green to yellow,
// brighter all the way, read the same by most colour-blind eyes. Its five-stop palette (#440154,
// #3b528b, #21918c, #5ec962, #fde725), between the stops linearly; `t` 0 to 1, clamped.
inline float3 viridis(float t)
{
  const float  x = clamp(t, 0.0F, 1.0F) * 4.0F;
  const float  i = min(floor(x), 3.0F);
  const float  f = x - i;
  const float3 a = i < 1.0F ? float3(0.267F, 0.004F, 0.329F) :
                   i < 2.0F ? float3(0.231F, 0.322F, 0.545F) :
                   i < 3.0F ? float3(0.129F, 0.569F, 0.549F) :
                              float3(0.369F, 0.788F, 0.384F);
  const float3 b = i < 1.0F ? float3(0.231F, 0.322F, 0.545F) :
                   i < 2.0F ? float3(0.129F, 0.569F, 0.549F) :
                   i < 3.0F ? float3(0.369F, 0.788F, 0.384F) :
                              float3(0.992F, 0.906F, 0.145F);
  return a + (b - a) * f;
}

// The Density view's scale: a triangle's size (the side of the square of twice its area, in metres:
// a quad's side, where a mesh of quads is cut in two) on a log scale, from half a millimetre (0) to
// half a metre (1), ten stops; one subdivision level halves it, a tenth of the ramp
static const float kTriangleSizeLog2Min = -11.0F;  // 0.49 mm
static const float kTriangleSizeLog2Max = -1.0F;   // 50 cm

inline float triangleSizeRamp(float size)
{
  if(!(size > 0.0F))
    return 0.0F;
  return clamp((log2(size) - kTriangleSizeLog2Min) / (kTriangleSizeLog2Max - kTriangleSizeLog2Min), 0.0F, 1.0F);
}

NAMESPACE_SHADERIO_END()

#endif
