/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Host/device interface of the rain pass (src/rain.hpp; shaders/rain.slang, rain_shelter.slang):
// the drops falling in the air, drawn over the image the tone mapper takes.
//

#ifndef RAIN_IO_H
#define RAIN_IO_H 1

#include "nvshaders/slang_types.h"
#include "shaderio.h"

NAMESPACE_SHADERIO_BEGIN()

#define RAIN_WORKGROUP_SIZE 16

// Set 1, push descriptors (set 0 is the scene's: its environment image), in VK_IMAGE_LAYOUT_GENERAL
enum RainBindings
{
  eRainIn    = 0,  // Sampler2D: the image as it comes
  eRainOut   = 1,  // RWTexture2D: the image the drops are drawn over (RGBA16F)
  eRainDepth = 2,  // Sampler2D<float>: the scene's depth, as the path tracer wrote it
  eRainTlas  = 3,  // the scene, for the shelter map
};

// A drop drawn this frame, as rainCull leaves it for the draw
struct RainStreak
{
  float4 ends;   // where its streak starts and ends on screen, pixels from the middle (x, y, x, y)
  float4 depth;  // the two ends' clip z and w
  float4 light;  // its light, pre-exposed; and the time it spends over a point of its streak, of the shutter's
  float4 width;  // its width at the two ends, pixels
};
#define RAIN_MAX_STREAKS 262144  // drops drawn a frame at most
#define RAIN_CULL_GROUP 64       // a cell's slots a workgroup of rainCull

// One frame's rain, written by the host each frame (a buffer of its own a frame in flight)
struct RainFrame
{
  float4x4 viewProj;  // the scene camera's: the drops are drawn and compared with its depth by it
  float3   camera;    // the eye, metres
  float    time;      // seconds of falling
  float2   shear;     // rainShear(wind)
  float    cellSize;  // metres
  float    chance;    // of each of a cell's slots holding a drop (its mean over its slots)
  float2   wind;      // m/s, x and z
  float    shutter;   // seconds: a streak is a drop's fall over it
  uint     slots;     // drops a cell at most
  float2   viewport;  // the image's size, pixels
  float2   depthSize;  // the depth image's part the path tracer wrote, pixels
  float    rate;       // mm/h
  uint     seed;
  float    focal;      // pixels a metre at a metre: the projection's
  float    shelterTop;  // the height the shelter map's rays start from
  float2   shelterCentre;  // where the map's middle is on the sheared ground
  float    minDistance;    // nearer drops are not drawn (they'd fill the frame, out of focus)
  float    maxDistance;    // nor farther ones (their streaks are under a pixel: the haze)
  SceneFrameInfo* frameInfo;  // the sun, the environment's turn and strength, the pre-exposure
  int4*           cells;      // the cells drawn (x, y, z), `slots` drops each
  float4*         light;      // the light table: RAIN_LIGHT_WIDTH x RAIN_LIGHT_HEIGHT, written by rainLight
  float*          shelter;    // the shelter map: RAIN_SHELTER_SIZE squared, written by rainShelter
  RainStreak*     streaks;    // the drops drawn, RAIN_MAX_STREAKS at most, written by rainCull
  uint*           drawArgs;   // the draw's VkDrawIndirectCommand: rainCull counts the drops into its instances
  float           minCoverage;  // a faint drop is drawn at a chance of its coverage over this, that much stronger
  uint            _pad0;
};

struct RainPush
{
  RainFrame* frame;
  int2       size;  // the dispatch's: the image's, or the light table's or the shelter map's
};

NAMESPACE_SHADERIO_END()

#endif  // RAIN_IO_H
