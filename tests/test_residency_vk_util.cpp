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

// What the retained scene's Vulkan backend converts and plans without a device
// (src/scene_residency_vk_util.hpp): the BLAS scratch batches (I8), TLAS rows, glTF samplers and
// buffer usage.

#include <gtest/gtest.h>

#include <random>

#include <tinygltf/tiny_gltf.h>

#include "scene_residency_vk_util.hpp"

using namespace rs;

// Side by side at aligned offsets until the budget, then a new batch from 0; one build larger than
// the budget is a batch of its own
TEST(ResidencyVkUtil, ScratchBatchesFillTheBudgetThenStartAgain)
{
  const VkDeviceSize sizes[] = {100, 300, 50, 2000, 10};
  const ScratchPlan  plan    = planScratch(sizes, 256, 1024);
  ASSERT_EQ(plan.places.size(), 5u);
  EXPECT_EQ(plan.batches, 3u);
  const ScratchPlan::Place want[] = {{0, 0}, {0, 256}, {0, 768}, {1, 0}, {2, 0}};
  for(size_t i = 0; i < 5; i++)
  {
    EXPECT_EQ(plan.places[i].batch, want[i].batch) << i;
    EXPECT_EQ(plan.places[i].offset, want[i].offset) << i;
  }
  EXPECT_EQ(plan.size, 2048u);  // the lone large build's
  EXPECT_EQ(planScratch({}, 256, 1024).batches, 0u);
}

// I8 over random queues: every build's scratch lies inside the buffer at the alignment, builds of
// one batch never overlap, and a batch of more than one stays within the budget
TEST(ResidencyVkUtil, ScratchCoversEveryBuildOfABatch)
{
  std::mt19937 rng(7);
  for(int round = 0; round < 200; round++)
  {
    const VkDeviceSize        alignment = VkDeviceSize(1) << (rng() % 9);
    const VkDeviceSize        budget    = 1 + rng() % 5000;
    std::vector<VkDeviceSize> sizes(rng() % 40);
    for(VkDeviceSize& s : sizes)
      s = rng() % 3000;
    const ScratchPlan plan = planScratch(sizes, alignment, budget);
    ASSERT_EQ(plan.places.size(), sizes.size());
    std::vector<std::vector<std::pair<VkDeviceSize, VkDeviceSize>>> batches(plan.batches);
    for(size_t i = 0; i < sizes.size(); i++)
    {
      const ScratchPlan::Place p    = plan.places[i];
      const VkDeviceSize       need = std::max<VkDeviceSize>(sizes[i], 1);
      ASSERT_LT(p.batch, plan.batches);
      EXPECT_EQ(p.offset % alignment, 0u);
      EXPECT_LE(p.offset + need, plan.size);
      for(const auto& [begin, end] : batches[p.batch])
        EXPECT_TRUE(p.offset >= end || p.offset + need <= begin) << "round " << round << " build " << i;
      batches[p.batch].push_back({p.offset, p.offset + need});
      if(i > 0)
        EXPECT_GE(p.batch, plan.places[i - 1].batch);  // in queue order
    }
    for(const auto& batch : batches)
      if(batch.size() > 1)
        EXPECT_LE(batch.back().second, budget);
  }
}

TEST(ResidencyVkUtil, ATlasRowBecomesAnInstance)
{
  TlasRow row;
  row.transform[0][3]                           = 5.f;
  row.customIndex                               = 0x1234567;  // above 24 bits: the low 24 are kept
  row.mask                                      = 0x81;
  row.flags                                     = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
  row.blas                                      = AccelHandle{9};
  const VkAccelerationStructureInstanceKHR inst = vkTlasInstance(row, 0xABCD00);
  EXPECT_EQ(inst.transform.matrix[0][3], 5.f);
  EXPECT_EQ(inst.transform.matrix[1][1], 1.f);
  EXPECT_EQ(inst.instanceCustomIndex, 0x234567u);
  EXPECT_EQ(inst.mask, 0x81u);
  EXPECT_EQ(inst.flags, uint32_t(VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR));
  EXPECT_EQ(inst.accelerationStructureReference, 0xABCD00u);

  // a hole references nothing and is masked out, whatever its row says
  row.blas                                      = {};
  const VkAccelerationStructureInstanceKHR hole = vkTlasInstance(row, 0xABCD00);
  EXPECT_EQ(hole.accelerationStructureReference, 0u);
  EXPECT_EQ(hole.mask, 0u);
}

TEST(ResidencyVkUtil, AGltfSamplerAsVulkan)
{
  const VkSamplerCreateInfo plain = gltfSamplerCreateInfo({});
  EXPECT_EQ(plain.minFilter, VK_FILTER_LINEAR);
  EXPECT_EQ(plain.magFilter, VK_FILTER_LINEAR);
  EXPECT_EQ(plain.mipmapMode, VK_SAMPLER_MIPMAP_MODE_LINEAR);
  EXPECT_EQ(plain.addressModeU, VK_SAMPLER_ADDRESS_MODE_REPEAT);
  EXPECT_EQ(plain.maxLod, VK_LOD_CLAMP_NONE);  // a texture with no sampler still mipmaps

  const VkSamplerCreateInfo pixel = gltfSamplerCreateInfo({.magFilter = TINYGLTF_TEXTURE_FILTER_NEAREST,
                                                           .minFilter = TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST,
                                                           .wrapS     = TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE,
                                                           .wrapT     = TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT});
  EXPECT_EQ(pixel.minFilter, VK_FILTER_NEAREST);
  EXPECT_EQ(pixel.magFilter, VK_FILTER_NEAREST);
  EXPECT_EQ(pixel.mipmapMode, VK_SAMPLER_MIPMAP_MODE_NEAREST);
  EXPECT_EQ(pixel.addressModeU, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  EXPECT_EQ(pixel.addressModeV, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);

  // a non-mipmapped minFilter keeps trilinear; an unknown wrap falls back to repeat
  const VkSamplerCreateInfo odd = gltfSamplerCreateInfo({.minFilter = TINYGLTF_TEXTURE_FILTER_LINEAR, .wrapS = 12345});
  EXPECT_EQ(odd.mipmapMode, VK_SAMPLER_MIPMAP_MODE_LINEAR);
  EXPECT_EQ(odd.addressModeU, VK_SAMPLER_ADDRESS_MODE_REPEAT);
}

// The GPU build lines: each kind's spans summed, a span that read back out of order counted as 0,
// and stamps missing for a span leave it out
TEST(ResidencyVkUtil, BuildSpansSumPerKind)
{
  const BuildSpan  spans[] = {BuildSpan::eBlas, BuildSpan::eTlas, BuildSpan::eBlas, BuildSpan::eBlas, BuildSpan::eTlas};
  const uint64_t   stamps[] = {100, 300, 300, 310, 400, 1000, 2000, 1500};
  const BuildTimes times    = sumBuildSpans(spans, stamps, 0.5);
  EXPECT_DOUBLE_EQ(times.blasMs, 400.0);  // (200 + 600 + 0) * 0.5
  EXPECT_DOUBLE_EQ(times.tlasMs, 5.0);
  EXPECT_EQ(times.blasSpans, 3u);
  EXPECT_EQ(times.tlasSpans, 1u);
  EXPECT_EQ(sumBuildSpans({}, {}, 1.0).blasSpans, 0u);
}

TEST(ResidencyVkUtil, BufferUsageAndBuildFlags)
{
  const VkBufferUsageFlags2 indices = vkBufferUsage(eBufferIndex | eBufferAccelInput);
  EXPECT_TRUE(indices & VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT);
  EXPECT_TRUE(indices & VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  EXPECT_TRUE(indices & VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);
  EXPECT_TRUE(indices & VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
  EXPECT_FALSE(indices & VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT);
  EXPECT_EQ(vkAccelBuildFlags(eAccelPreferFastTrace | eAccelAllowUpdate),
            VkBuildAccelerationStructureFlagsKHR(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                 | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
}
