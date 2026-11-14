/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>

#include <glm/glm.hpp>

#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>

#include "gpu_memory_tracker.hpp"
#include "resources.hpp"
#include "shaders/shaderio.h"

//-------------------------------------------------------------------------------------------------------
// LightGroups: a still that keeps its lights apart (ptLightGroups, shaders/light_mix.h.slang).
//
// A picture is the sum of what each of its lights gives it, so the Light tab's mix (a strength and a
// tint for each light group) can be changed on a finished still with nothing traced again, if the
// still kept each group's share. While this is on, the accumulation writes the whole picture into an
// image here (not the rendered image) and each group's share into one of its own, a sample noting one
// group's, the groups the scene has light in taking turns
// (PathTracer binds them at shaderio::eResultImage and eLightGroup0), and the mix (light_mix.slang)
// writes the rendered image from them: after every frame of the accumulation, and on an idle frame
// whenever the mix changed. Everything after it (the still's denoiser, the tone mapper, an export) reads
// the rendered image as before.
//
// The images (one RGBA32F and LIGHT_GROUP_COUNT RGBA16F, 64 bytes a pixel) exist only while it is on.
//
// Usage:
//   groups.init(resources);                    // once
//   groups.ensure(cmd, resources, on);         // every frame: the images follow the setting and the size
//   groups.cmdMix(cmd, rendered, mix);         // after the accumulation's frame, or when the mix changed
//   groups.deinit(resources);
//-------------------------------------------------------------------------------------------------------
class LightGroups
{
public:
  using Mix = std::array<glm::vec4, LIGHT_GROUP_COUNT>;  // each group's strength times its tint, in .xyz

  void init(Resources& resources);
  void deinit(Resources& resources);

  // The images follow `on` and the G-buffer's size; released when off. Waits for the queue when it
  // releases them (rare: the setting went off).
  void ensure(VkCommandBuffer cmd, Resources& resources, bool on);
  // The images exist, at the G-buffer's size
  bool ready() const { return m_allocated; }

  VkDescriptorImageInfo total() const { return m_target.getColorStorageImageInfo(0); }
  VkDescriptorImageInfo group(int index) const { return m_target.getColorStorageImageInfo(1 + index); }

  // A still begins: no group has a share yet (each group's image is written only on its turns)
  void cmdClearGroups(VkCommandBuffer cmd);
  // The images take over an accumulation they did not see begin: the whole picture so far is
  // `rendered` (RGBA32F, in VK_IMAGE_LAYOUT_GENERAL), and no group but the sky has a share of it
  void cmdAdopt(VkCommandBuffer cmd, VkImage rendered);

  // The rendered image from the images here, under `mix`. Storage images in VK_IMAGE_LAYOUT_GENERAL;
  // the caller orders the writes before it and the reads after it with barriers.
  // `used`: the groups the scene has light in, a bit a group (PathTracer::lightGroupsUsed).
  void cmdMix(VkCommandBuffer cmd, const VkDescriptorImageInfo& rendered, const Mix& mix, uint32_t used);

  // The mix the rendered image was last written with (cmdMix), and whether the images hold a still's
  // samples (the accumulation that wrote them is the one on screen)
  const Mix& mixed() const { return m_mixed; }
  bool       holdsStill() const { return m_allocated && m_holdsStill; }
  void       setHoldsStill(bool holds) { m_holdsStill = holds; }

private:
  void release(Resources& resources);

  VkDevice                    m_device           = VK_NULL_HANDLE;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvvk::RenderTarget          m_target;  // [0] the whole (RGBA32F), [1 + g] group g's share (RGBA16F)
  bool                        m_targetInit = false;
  bool                        m_allocated  = false;
  bool                        m_holdsStill = false;
  Mix                         m_mixed{};

  nvvk::DescriptorBindings m_bindings;
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipelineLayout         m_pipelineLayout{};
  VkPipeline               m_pipeline{};
};

// Whether a mix leaves every group as it is
inline bool lightMixNeutral(const LightGroups::Mix& mix)
{
  for(const glm::vec4& m : mix)
    if(m.x != 1.0F || m.y != 1.0F || m.z != 1.0F)
      return false;
  return true;
}
