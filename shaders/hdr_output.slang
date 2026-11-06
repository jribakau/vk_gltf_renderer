/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// HDR display output (src/hdr_output.hpp): the image an HDR display is shown, scRGB half floats.
//   tonemapHdr:  from the tonemapper's input, as the SDR image is made, raised by the HDR gain
//                (hdr_output.h.slang) and put at the display's SDR white
//   composeHdr:  what the overlays drew into the SDR image since (the selection outline, the grid,
//                the gizmo), at SDR white; or, with nothing tone mapped this frame, the whole SDR image
//   exportLinear: the tonemapper's input with its exposure and white balance, linear and unclamped, for
//                an EXR export: the light the SDR image is made from, without the curve and the grading

#include "nvshaders/tonemap_functions.h.slang"
#include "hdr_output_io.h.slang"
#include "hdr_output.h.slang"

// clang-format off
[[vk::binding(HdrOutputBindings::eHdrInput)]]   Sampler2D<float4>   u_input;
[[vk::binding(HdrOutputBindings::eHdrSdr)]]     RWTexture2D<float4> u_sdr;
[[vk::binding(HdrOutputBindings::eHdrSdrKept)]] RWTexture2D<float4> u_sdrKept;
[[vk::binding(HdrOutputBindings::eHdrOut)]]     RWTexture2D<float4> u_out;

[[vk::push_constant]] ConstantBuffer<HdrOutputPushConstant> pc;
// clang-format on

static const float kHalfMax = 65504.0;  // Largest half float: the output is RGBA16F

// The SDR curve's result for a colour the curve takes, linear and before any grading: applyTonemap's
// curves, without the grading it does after them
float3 sdrCurve(int method, float3 c)
{
  switch(method)
  {
    case TONEMAP_METHOD(eFilmic):
      c = tonemapFilmic(c);
      break;
    case TONEMAP_METHOD(eUncharted2):
      c = tonemapUncharted2(c);
      break;
    case TONEMAP_METHOD(eClip):
      c = toSrgb(c);
      break;
    case TONEMAP_METHOD(eACES):
      c = tonemapACES(c);
      break;
    case TONEMAP_METHOD(eAgX):
      c = tonemapAgX(c);
      break;
    case TONEMAP_METHOD(eKhronosPBR):
      return hdrNeutral(c, 1.0F);
  }
  return toLinear(saturate(c));
}

[shader("compute")]
[numthreads(TONEMAP_WORKGROUP_SIZE, TONEMAP_WORKGROUP_SIZE, 1)]
void tonemapHdr(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  int2 size;
  u_out.GetDimensions(size.x, size.y);
  const int2 pixel = int2(dispatchThreadID.xy);
  if(pixel.x >= size.x || pixel.y >= size.y)
    return;

  const float4 color = u_input.Load(int3(pixel, 0));
  float3       hdr;
  if(pc.tm.isActive == 1)
  {
    // The SDR image's pixel, graded as it is but not dithered (half floats have no 8-bit steps to
    // hide), raised by how much brighter the curve gets with the display's headroom
    TonemapperData tm = pc.tm;
    tm.dither         = 0;
    const float3 sdr  = toLinear(saturate(applyTonemap(tm, color.xyz, float2(pixel), float2(size))));
    const float3 seen = mul(tm.inputMatrix, color.xyz);  // what the curve takes, as in applyTonemap
    // the exposure view's colours stay at SDR white: they mark the SDR curve's clipping, not the display's
    hdr = tm.falseColor != 0 ? sdr : sdr * hdrGain(tm.method, seen, sdrCurve(tm.method, seen), pc.peak);
  }
  else
  {
    // A debug view: its values as the SDR image shows them
    hdr = toLinear(saturate(color.xyz));
  }
  u_out[pixel] = float4(min(hdr * pc.paperWhite, float3(kHalfMax)), 1.0);
}

[shader("compute")]
[numthreads(TONEMAP_WORKGROUP_SIZE, TONEMAP_WORKGROUP_SIZE, 1)]
void composeHdr(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  int2 size;
  u_out.GetDimensions(size.x, size.y);
  const int2 pixel = int2(dispatchThreadID.xy);
  if(pixel.x >= size.x || pixel.y >= size.y)
    return;

  // A pixel an overlay drew into differs from the SDR image as the tonemapper left it: it takes the
  // overlay's colour at SDR white, the rest keeps the HDR tone mapping
  const float4 sdr = u_sdr[pixel];
  if(pc.whole != 0 || any(sdr != u_sdrKept[pixel]))
    u_out[pixel] = float4(toLinear(saturate(sdr.xyz)) * pc.paperWhite, 1.0);
}

[shader("compute")]
[numthreads(TONEMAP_WORKGROUP_SIZE, TONEMAP_WORKGROUP_SIZE, 1)]
void exportLinear(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  int2 size;
  u_out.GetDimensions(size.x, size.y);
  const int2 pixel = int2(dispatchThreadID.xy);
  if(pixel.x >= size.x || pixel.y >= size.y)
    return;

  // As the tonemapper takes it, before its curve; a debug view's values as they are. Opaque: the
  // image as the viewport shows it, background included
  const float4 color = u_input.Load(int3(pixel, 0));
  const float3 c     = pc.tm.isActive == 1 ? mul(pc.tm.inputMatrix, color.xyz) : color.xyz;
  u_out[pixel]       = float4(c, 1.0);
}
