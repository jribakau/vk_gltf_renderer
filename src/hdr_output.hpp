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

#pragma once

#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>

#include "present_hdr.hpp"
#include "resources.hpp"
#include "shaders/hdr_output_io.h.slang"

//-------------------------------------------------------------------------------------------------------
// HdrOutput
//
// The frame an HDR display is shown. When the window's display is in HDR mode (the presenter says so,
// frame_present.hpp) and the presentHdr setting allows it, each frame gets an HDR image beside the SDR
// one: RGBA16F scRGB, the SDR image's white at the display's SDR white level and its highlights raised
// towards the display's peak by the tone curve's own headroom (shaders/hdr_output.h.slang). Everything
// else (the stream, screenshots and exports, the overlays) keeps the SDR image, which stays as it was.
//
//   hdr.setDisplay(display, allowed, peakNits);        // each frame, before it is rendered
//   hdr.cmdTonemap(cmd, resources, size, tm, input);   // right after the SDR tonemapper, with its input
//   hdr.cmdKeepSdr(cmd, resources);                    // once the SDR image is final, before the overlays
//   hdr.cmdFinish(cmd, resources);                     // the frame's end: the overlays go into the HDR image
//
// The overlays (selection outline, grid, gizmo, markers) draw into the SDR image only; a pixel that
// differs from the kept copy afterwards is one they drew, and takes its SDR colour at SDR white. A frame
// that tone mapped nothing (busy, an empty scene, a DLSS guide view, auto exposure) shows the SDR image
// whole in the HDR one.
//-------------------------------------------------------------------------------------------------------
class HdrOutput
{
public:
  void init(Resources& resources);
  void deinit(Resources& resources);
  void onResize(VkCommandBuffer cmd, VkExtent2D size);

  // The display the frames go to and whether HDR output is allowed (presentHdr); `peakNits` above 0
  // stands for the display's peak (presentHdrPeak)
  void setDisplay(const DisplayHdr& display, bool allowed, float peakNits);
  // Frames are HDR: the presenter shows image(), RGBA16F
  bool active() const { return m_active; }
  const HdrLevels& levels() const { return m_levels; }
  // The HDR image, once a frame has made it; null before
  VkImage image() const;

  // The HDR image tone mapped from the SDR tonemapper's input and settings (`tm` as it ran, `size` its
  // dispatch). Nothing when not active.
  void cmdTonemap(VkCommandBuffer cmd, Resources& resources, VkExtent2D size, const shaderio::TonemapperData& tm,
                  const VkDescriptorImageInfo& input);
  // The SDR image as it is now, before the overlays; after cmdTonemap only
  void cmdKeepSdr(VkCommandBuffer cmd, Resources& resources);
  // The frame's end: what the overlays drew since cmdKeepSdr, or the whole SDR image when this frame
  // tone mapped nothing, goes into the HDR image
  void cmdFinish(VkCommandBuffer cmd, Resources& resources);

  // GPU memory a pixel of the frame costs while active (the HDR image and the kept SDR copy)
  static constexpr uint32_t kBytesPerPixel = 8 + 4;

  // An EXR export's image (RGBA32F, the frame's size): the tonemapper's input with its exposure and
  // white balance, linear and unclamped, before the curve and the grading, whether or not frames are
  // HDR. Recorded with its read back (GltfRenderer::saveExr), then let go.
  void       cmdExportLinear(VkCommandBuffer cmd, Resources& resources, VkExtent2D size, const shaderio::TonemapperData& tm,
                             const VkDescriptorImageInfo& input);
  VkImage    exportImage() const;  // null before cmdExportLinear and after releaseExport
  VkExtent2D exportSize() const { return m_export.getSize(); }
  // Nothing may still use it: the device idle
  void releaseExport();

private:
  enum Image
  {
    eImageHdr,       // RGBA16F scRGB: what the display is shown
    eImageSdrKept,   // RGBA8: the SDR image before the overlays
    eImageCount,
  };

  bool ensureTarget(VkCommandBuffer cmd, Resources& resources);
  // Every binding takes an image, the ones an entry point doesn't read too
  void dispatch(VkCommandBuffer cmd, Resources& resources, VkPipeline pipeline, VkExtent2D size,
                const shaderio::HdrOutputPushConstant& push, const VkDescriptorImageInfo& input,
                const VkDescriptorImageInfo& kept, const VkDescriptorImageInfo& out);

  nvvk::RenderTarget          m_target;
  nvvk::RenderTarget          m_export;   // RGBA32F, while an EXR export is due
  VkPipeline                  m_exportPipeline{};
  Resources*                  m_resources{};
  nvvk::DescriptorBindings    m_bindings;
  VkDescriptorSetLayout       m_descriptorSetLayout{};
  VkPipelineLayout            m_pipelineLayout{};
  VkPipeline                  m_tonemapPipeline{};
  VkPipeline                  m_composePipeline{};
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker{};

  bool      m_active = false;
  HdrLevels m_levels;
  bool      m_tonemapped = false;  // This frame
  bool      m_kept       = false;  // This frame
  shaderio::TonemapperData m_tm{};  // As this frame's cmdTonemap took it
};
