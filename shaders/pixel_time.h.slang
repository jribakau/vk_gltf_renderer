#ifndef FV_PIXEL_TIME_H
#define FV_PIXEL_TIME_H

// The Time view (Display › Time, the path tracer's USE_PIXEL_TIME variant): how long each pixel's
// thread ran by the GPU's own clock (VK_KHR_shader_clock at device scope), shown against the frame's
// average pixel, and the frame's time summed by what each pixel's camera ray met first. Under SER a
// thread's time includes its waits in the reorders and for its warp, so it tells where a frame's time
// goes, not a pixel's exact cost. Slang and C++ (tests/test_debug_colors.cpp, src/pixel_time.cpp).

#include "debug_colors.h.slang"

NAMESPACE_SHADERIO_BEGIN()

// The frame's sums, four uints a slot: the time's low and high 32 bits (in the clock's ticks shifted
// down by kPixelTimeUnitShift, or a count: kPixelTimeCount*), the samples taken, and one unused. A slot is what the first camera ray
// of the pixel's first sample met:
static const uint kPixelTimeSky      = 0u;           // nothing: the environment, the sky or a backplate
static const uint kPixelTimeGround   = 1u;           // the ground
static const uint kPixelTimeWater    = 2u;           // the ground's water
static const uint kPixelTimeOther    = 3u;           // a mesh past the slots the host gave (SceneFrameInfo::pixelTimeSlots)
static const uint kPixelTimeInstance = 4u;           // the first mesh's: kPixelTimeInstance plus its instance (render node)
static const uint kPixelTimeUnset    = 0xFFFFFFFFu;  // no camera ray has met anything yet
// The sums count 16 of the clock's ticks as one, so a warp's 32 pixels fit a uint even in a frame of a
// second (NVIDIA's device clock is in nanoseconds); the image keeps every tick
static const uint kPixelTimeUnitShift = 4u;

// What the view sums in place of the time (SceneFrameInfo::pixelTimeCount, the debug `pixelTimeCount`):
// the rays a pixel traced -- closest hits, shadow rays and the selection ray alike -- a walk's steps
// through a scattering medium alone, or its shadow rays alone, each as many as there were (no shift).
// Counts tell how much a pixel traces from how dear each of its rays is, which the time can't.
static const int kPixelTimeCountClock      = 0;
static const int kPixelTimeCountRays       = 1;
static const int kPixelTimeCountWalkSteps  = 2;
static const int kPixelTimeCountShadowRays = 3;

// The view's scale: the time of a pixel's sample over the frame's average, from an eighth (0) to eight
// times it (1) on a log scale, the average in the middle
static const float kPixelTimeStops = 3.0F;

inline float pixelTimeRamp(float ratio)
{
  return clamp((log2(max(ratio, 1e-6F)) + kPixelTimeStops) / (2.0F * kPixelTimeStops), 0.0F, 1.0F);
}

inline float3 pixelTimeColor(float ratio)
{
  return viridis(pixelTimeRamp(ratio));
}

NAMESPACE_SHADERIO_END()

#if USE_PIXEL_TIME
// The pixel's count in a count mode, from processPixel's start; the ray tracer (raytracer_interface.h.slang)
// and the walk's step (pathTraceOneBounce) add to it
static uint g_timeCount = 0u;
#define PIXEL_TIME_COUNT(what)                                                                                         \
  if(pushConst.frameInfo->pixelTimeCount == (what))                                                                    \
  g_timeCount++
#else
#define PIXEL_TIME_COUNT(what)
#endif

#endif
