/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Rain: the drops falling in the air, drawn over the image the tone mapper takes (see rain.hpp).

#include "rain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>

#include "rain_cells.hpp"
#include "_autogen/rain.slang.h"
#include "_autogen/rain_shelter.slang.h"

// The physics, as the shaders have it
namespace rain_physics {
using namespace glm;
using float2 = glm::vec2;
using float3 = glm::vec3;
using int3   = glm::ivec3;
using uint   = uint32_t;
#include "shaders/rain.h.slang"
}  // namespace rain_physics

static_assert(offsetof(shaderio::RainFrame, frameInfo) % 8 == 0, "RainFrame's pointers are 8-aligned on both sides");
static_assert(offsetof(shaderio::RainFrame, frameInfo) == 160, "RainFrame's layout is the shader's (scalar)");
static_assert(sizeof(shaderio::RainPush) <= 128, "RainPush fits any device's push constants");

static constexpr const char* kMemoryCategory = "Rain";
static constexpr VkFormat    kFormat         = VK_FORMAT_R16G16B16A16_SFLOAT;
// RainFrame, then the cells, in a frame's buffer
static constexpr size_t kCellsOffset = (sizeof(shaderio::RainFrame) + 15) & ~size_t(15);
static constexpr size_t kFrameBytes  = kCellsOffset + Rain::kMaxCells * sizeof(glm::ivec4);

void Rain::init(Resources& resources)
{
  m_device           = resources.allocator.getDevice();
  m_alloc            = &resources.allocator;
  m_appMemoryTracker = &resources.appMemoryTracker;
  m_app              = resources.app;
  NVVK_CHECK(m_out.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Rain out"}));
  createPipelines(resources);
}

void Rain::createPipelines(Resources& resources)
{
  m_bindings.addBinding(shaderio::RainBindings::eRainIn, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_bindings.addBinding(shaderio::RainBindings::eRainOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_bindings.addBinding(shaderio::RainBindings::eRainDepth, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_bindings.addBinding(shaderio::RainBindings::eRainTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_setLayout));
  NVVK_DBG_NAME(m_setLayout);

  // set 0 the scene's (its environment image), set 1 the pass's own
  const std::array<VkDescriptorSetLayout, 2> sets{resources.descriptorSetLayout[0], m_setLayout};
  const VkPushConstantRange                  push{.stageFlags = VK_SHADER_STAGE_ALL, .size = sizeof(shaderio::RainPush)};
  const VkPipelineLayoutCreateInfo           layoutInfo{
                .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount         = uint32_t(sets.size()),
                .pSetLayouts            = sets.data(),
                .pushConstantRangeCount = 1,
                .pPushConstantRanges    = &push,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  const auto compute = [&](const uint32_t* code, size_t bytes, const char* entry, VkPipeline& pipeline) {
    const VkShaderModuleCreateInfo    shaderInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code};
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &shaderInfo, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = entry},
        .layout = m_pipelineLayout,
    };
    NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &pipeline));
    NVVK_DBG_NAME(pipeline);
  };
  compute(rain_slang, rain_slang_sizeInBytes, "rainLight", m_lightPipeline);
  compute(rain_slang, rain_slang_sizeInBytes, "rainCopy", m_copyPipeline);
  compute(rain_slang, rain_slang_sizeInBytes, "rainCull", m_cullPipeline);
  compute(rain_shelter_slang, rain_shelter_slang_sizeInBytes, "rainShelter", m_shelterPipeline);

  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
      .nextStage              = VK_SHADER_STAGE_FRAGMENT_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = rain_slang_sizeInBytes,
      .pCode                  = rain_slang,
      .pName                  = "rainVertex",
      .setLayoutCount         = uint32_t(sets.size()),
      .pSetLayouts            = sets.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &push,
  };
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_vertexShader));
  NVVK_DBG_NAME(m_vertexShader);
  shaderInfo.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
  shaderInfo.nextStage = 0;
  shaderInfo.pName     = "rainFragment";
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_fragmentShader));
  NVVK_DBG_NAME(m_fragmentShader);

  // the streaks over the image, premultiplied: the drop's light over what it covers
  m_drawState.inputAssemblyState.topology                = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  m_drawState.rasterizationState.cullMode                = VK_CULL_MODE_NONE;
  m_drawState.depthStencilState.depthTestEnable          = VK_FALSE;
  m_drawState.depthStencilState.depthWriteEnable         = VK_FALSE;
  m_drawState.colorBlendEnables[0]                       = VK_TRUE;
  m_drawState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  m_drawState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  m_drawState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
  m_drawState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  m_drawState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  m_drawState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
}

void Rain::deinit(Resources& /*resources*/)
{
  vkDestroyPipeline(m_device, m_lightPipeline, nullptr);
  vkDestroyPipeline(m_device, m_copyPipeline, nullptr);
  vkDestroyPipeline(m_device, m_shelterPipeline, nullptr);
  vkDestroyPipeline(m_device, m_cullPipeline, nullptr);
  m_cullPipeline = VK_NULL_HANDLE;
  vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
  vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_setLayout, nullptr);
  m_bindings.clear();
  m_lightPipeline = m_copyPipeline = m_shelterPipeline = VK_NULL_HANDLE;
  m_vertexShader = m_fragmentShader = VK_NULL_HANDLE;
  m_pipelineLayout                  = VK_NULL_HANDLE;
  m_setLayout                       = VK_NULL_HANDLE;

  for(nvvk::Buffer& b : m_frames)
    if(b.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(b);
  m_frames.clear();
  if(m_light.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_light);
  for(nvvk::Buffer* b : {&m_light, &m_shelter, &m_streaks, &m_drawArgs})
    if(b->buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(*b);
  m_light = m_shelter = m_streaks = m_drawArgs = {};

  if(m_appMemoryTracker && m_size.width != 0 && m_size.height != 0)
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
  m_out.deinit();
  m_size = {};
}

void Rain::registerParameters(SettingsRegistry* settings, std::function<void()> wetChanged)
{
  const auto changed = [wetChanged](const nvutils::ParameterBase* const) {
    if(wetChanged)
      wetChanged();
  };
  // Not persisted: the host sends them with the lighting (fv_lighting.cpp), after every load too
  settings->add({.name = "rainRate", .help = "Rain: how hard it rains, mm/h (0 = none, 2.5 light, 7.5 heavy, 50 a downpour)", .callbackSuccess = changed},
                &m_settings.rate, Persist::eNo, 0.0f, 300.0f);
  settings->add({.name = "rainWind", .help = "Rain: the wind at the ground, m/s, which slants the drops", .callbackSuccess = changed},
                &m_settings.wind, Persist::eNo, 0.0f, 40.0f);
  settings->add({.name = "rainWindFrom", .help = "Rain: where the wind blows from (degrees, an azimuth as the sun's)", .callbackSuccess = changed},
                &m_settings.windFrom, Persist::eNo, -360.0f, 360.0f);
  settings->add({"rainShutter", "Rain: the camera's shutter, seconds: a drop's streak is its fall while it is open"},
                &m_settings.shutter, Persist::eNo, 1.0f / 8000.0f, 30.0f);
  settings->add({"rainFalling", "Rain: the drops fall (off: they hold where they are)"}, &m_settings.falling, Persist::eNo);
  settings->add({"rainReach", "Rain: metres out to which drops are drawn"}, &m_settings.reach, Persist::eNo, 2.0f, 40.0f);
}

float Rain::wetness() const
{
  return active() ? rain_physics::rainWetness(m_settings.rate) : 0.0f;
}

float Rain::puddles() const
{
  return active() ? rain_physics::rainPuddleShare(m_settings.rate) : 0.0f;
}

void Rain::onResize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(m_size.width == 0 || m_size.height == 0)  // never allocated: the first frame with rain does
    return;
  releaseTarget();
  ensureTarget(cmd, size);
}

void Rain::releaseTarget()
{
  if(m_size.width == 0 || m_size.height == 0)
    return;
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
  m_out.deinit();
  NVVK_CHECK(m_out.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Rain out"}));
  m_size = {};
}

bool Rain::ensureTarget(VkCommandBuffer cmd, VkExtent2D size)
{
  if(size.width == m_size.width && size.height == m_size.height)
    return true;
  if(m_appMemoryTracker && m_size.width != 0 && m_size.height != 0)
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
  NVVK_CHECK(m_out.update(cmd, size));
  m_size = size;
  if(m_appMemoryTracker)
    m_appMemoryTracker->track(kMemoryCategory, m_out, 1);
  return true;
}

void Rain::cmdPrepare(VkCommandBuffer cmd, Resources& resources, VkExtent2D size, const View& view)
{
  m_prepared = false;
  if(!active() || size.width == 0 || size.height == 0)
    return;
  NVVK_DBG_SCOPE(cmd);

  // the clock: real time while the drops fall, a long stall counting as a short frame
  const auto now = std::chrono::steady_clock::now();
  if(m_settings.falling && m_lastFrame.time_since_epoch().count() != 0)
  {
    const float seconds = std::min(std::chrono::duration<float>(now - m_lastFrame).count(), kLongestFrame);
    m_time              = std::fmod(m_time + seconds, kClockWrap);
    ++m_version;
  }
  m_lastFrame = now;

  // the buffers: one a frame in flight for the host's, the table and the map the GPU's
  const size_t ring = std::max<size_t>(1, m_app ? m_app->getFrameCycleSize() : 3);
  if(m_frames.size() < ring)
    m_frames.resize(ring);
  nvvk::Buffer& frame = m_frames[m_app ? size_t(m_app->getFrameCycleIndex()) % m_frames.size() : 0];
  if(frame.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(frame, kFrameBytes, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
    NVVK_DBG_NAME(frame.buffer);
  }
  if(m_light.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_light, RAIN_LIGHT_WIDTH * RAIN_LIGHT_HEIGHT * sizeof(glm::vec4), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
    NVVK_DBG_NAME(m_light.buffer);
    NVVK_CHECK(m_alloc->createBuffer(m_shelter, RAIN_SHELTER_SIZE * RAIN_SHELTER_SIZE * sizeof(float),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    NVVK_DBG_NAME(m_shelter.buffer);
    NVVK_CHECK(m_alloc->createBuffer(m_streaks, RAIN_MAX_STREAKS * sizeof(shaderio::RainStreak), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
    NVVK_DBG_NAME(m_streaks.buffer);
    NVVK_CHECK(m_alloc->createBuffer(m_drawArgs, sizeof(VkDrawIndirectCommand),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    NVVK_DBG_NAME(m_drawArgs.buffer);
  }
  if(frame.mapping == nullptr)
    return;

  // the frame: the camera, the wind, the cells it sees
  const glm::mat4 viewProj = view.proj * view.view;
  const glm::vec3 camera   = glm::vec3(glm::inverse(view.view)[3]);
  const float     from     = glm::radians(m_settings.windFrom);
  const glm::vec2 wind     = -glm::vec2(std::sin(from), std::cos(from)) * m_settings.wind;
  const glm::vec2 shear    = rain_physics::rainShear(wind);
  const float     reach    = m_settings.reach;
  const std::vector<glm::ivec4> cells = rain_cells::visible(viewProj, camera, shear, kCellSize, reach, kMaxCells);
  const float perCell = rain_physics::rainDropsPerCubicMetre(m_settings.rate, rain_physics::RAIN_MIN_DIAMETER) * kCellSize * kCellSize * kCellSize;
  const uint32_t slots = uint32_t(std::clamp(std::ceil(perCell), 1.0f, float(kMaxSlots)));

  shaderio::RainFrame rf{};
  rf.viewProj      = viewProj;
  rf.camera        = camera;
  rf.time          = m_time;
  rf.shear         = shear;
  rf.cellSize      = kCellSize;
  rf.chance        = std::min(perCell / float(slots), 1.0f);
  rf.wind          = wind;
  rf.shutter       = m_settings.shutter;
  rf.slots         = slots;
  rf.viewport      = glm::vec2(float(size.width), float(size.height));
  rf.depthSize     = glm::vec2(float(view.depthExtent.width), float(view.depthExtent.height));
  rf.rate          = m_settings.rate;
  rf.seed          = 1u;
  rf.focal         = std::abs(view.proj[1][1]) * 0.5f * float(size.height);
  rf.shelterTop    = camera.y + 200.0f;
  rf.shelterCentre = glm::vec2(camera.x - shear.x * camera.y, camera.z - shear.y * camera.y);
  rf.minDistance   = kMinDistance;
  rf.maxDistance   = reach;
  rf.frameInfo     = reinterpret_cast<shaderio::SceneFrameInfo*>(resources.bFrameInfo.address);
  rf.cells         = reinterpret_cast<glm::ivec4*>(frame.address + kCellsOffset);
  rf.light         = reinterpret_cast<glm::vec4*>(m_light.address);
  rf.shelter       = reinterpret_cast<float*>(m_shelter.address);
  rf.streaks       = reinterpret_cast<shaderio::RainStreak*>(m_streaks.address);
  rf.drawArgs      = reinterpret_cast<uint32_t*>(m_drawArgs.address);
  rf.minCoverage   = kMinCoverage;
  // the shelter map's middle on whole texels, so it doesn't shimmer as the camera moves
  const float texel = rain_physics::RAIN_SHELTER_EXTENT / float(RAIN_SHELTER_SIZE);
  rf.shelterCentre  = glm::floor(rf.shelterCentre / texel) * texel;
  std::memcpy(frame.mapping, &rf, sizeof(rf));
  if(!cells.empty())
    std::memcpy(reinterpret_cast<char*>(frame.mapping) + kCellsOffset, cells.data(), cells.size() * sizeof(glm::ivec4));

  m_frameAddress = frame.address;
  m_slots        = slots;
  m_cellCount    = uint32_t(cells.size());
  m_planShear    = shear;
  m_planCentre   = rf.shelterCentre;

  const shaderio::RainPush push{.frame = reinterpret_cast<shaderio::RainFrame*>(m_frameAddress), .size = {RAIN_SHELTER_SIZE, RAIN_SHELTER_SIZE}};
  // the last frame's readers of the map, and the scene's build this frame
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  const VkAccelerationStructureKHR tlas = resources.gpuView().tlas();
  // the shelter map: traced, or open everywhere without a scene to trace
  if(tlas != VK_NULL_HANDLE)
  {
    nvvk::WriteSetContainer writes;
    writes.append(m_bindings.getWriteSet(shaderio::RainBindings::eRainTlas), tlas);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, uint32_t(writes.size()), writes.data());
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_shelterPipeline);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, RAIN_SHELTER_SIZE / 16, RAIN_SHELTER_SIZE / 16, 1);
  }
  else
    vkCmdFillBuffer(cmd, m_shelter.buffer, 0, VK_WHOLE_SIZE, 0xFF800000u);  // -inf: open
  // read by the path tracer (where it rains on what) and the drops
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
  m_prepared = true;
}

VkDescriptorImageInfo Rain::cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input, VkExtent2D size,
                                     const View& view, nvvk::ProfilerGpuTimer& timer)
{
  if(!m_prepared || size.width == 0 || size.height == 0 || view.depthExtent.width == 0 || !ensureTarget(cmd, size))
    return input;
  NVVK_DBG_SCOPE(cmd);
  const uint32_t           slots = m_slots;
  const bool               any   = m_cellCount > 0;
  const shaderio::RainPush pushFrame{.frame = reinterpret_cast<shaderio::RainFrame*>(m_frameAddress), .size = {int(size.width), int(size.height)}};

  // The previous frame's readers of the image and the buffers, and whatever wrote the image. In a
  // section of its own: the profiler's sections start at the top of the pipe, so the first after
  // this would count the wait
  {
    auto section = timer.cmdFrameSection(cmd, "Rain wait");
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
  }

  VkDescriptorImageInfo in = input;
  in.sampler               = resources.linearSampler;
  const VkDescriptorImageInfo depth{.sampler     = resources.linearSampler,
                                    .imageView   = resources.gBuffers.getDepthImageView(),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  nvvk::WriteSetContainer          writes;
  writes.append(m_bindings.getWriteSet(shaderio::RainBindings::eRainIn), in);
  writes.append(m_bindings.getWriteSet(shaderio::RainBindings::eRainOut), m_out.getColorStorageImageInfo(0));
  writes.append(m_bindings.getWriteSet(shaderio::RainBindings::eRainDepth), depth);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &resources.descriptorSet, 0, nullptr);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, uint32_t(writes.size()), writes.data());

  // the draw: six corners a drop, none counted yet
  const VkDrawIndirectCommand none{.vertexCount = 6, .instanceCount = 0, .firstVertex = 0, .firstInstance = 0};
  vkCmdUpdateBuffer(cmd, m_drawArgs.buffer, 0, sizeof(none), &none);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  // the light table, and the image copied for the drops to be drawn over
  {
  auto section = timer.cmdFrameSection(cmd, "Rain light and copy");
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_lightPipeline);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pushFrame), &pushFrame);
  vkCmdDispatch(cmd, RAIN_LIGHT_WIDTH * RAIN_LIGHT_HEIGHT, 1, 1);

  // the image, copied for the drops to be drawn over
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_copyPipeline);
  const VkExtent2D groups = nvvk::getGroupCounts(size, VkExtent2D{RAIN_WORKGROUP_SIZE, RAIN_WORKGROUP_SIZE});
  vkCmdDispatch(cmd, groups.width, groups.height, 1);
  }

  // the drops: the shelter map and the light table read
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
  if(any)
  {
    auto section = timer.cmdFrameSection(cmd, "Rain cull");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cullPipeline);
    vkCmdDispatch(cmd, (slots + RAIN_CULL_GROUP - 1) / RAIN_CULL_GROUP, m_cellCount, 1);
  }

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                             | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT
                             | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

  // the drops
  if(any)
  {
    auto section = timer.cmdFrameSection(cmd, "Rain drops");
    const VkRenderingAttachmentInfo color{.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                          .imageView   = m_out.getColorAttachmentView(0),
                                          .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                          .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
                                          .storeOp     = VK_ATTACHMENT_STORE_OP_STORE};
    const VkRenderingInfo           rendering{.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                              .renderArea           = {{0, 0}, size},
                                              .layerCount           = 1,
                                              .colorAttachmentCount = 1,
                                              .pColorAttachments    = &color};
    vkCmdBeginRendering(cmd, &rendering);
    const VkViewport viewport{0.0f, 0.0f, float(size.width), float(size.height), 0.0f, 1.0f};
    vkCmdSetViewportWithCount(cmd, 1, &viewport);
    const VkRect2D scissor{{0, 0}, size};
    vkCmdSetScissorWithCount(cmd, 1, &scissor);
    m_drawState.cmdApplyAllStates(cmd);
    m_drawState.cmdApplyDynamicState(cmd, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
    nvvk::GraphicsPipelineState::cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});
    vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &resources.descriptorSet, 0, nullptr);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 1, uint32_t(writes.size()), writes.data());
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pushFrame), &pushFrame);
    vkCmdDrawIndirect(cmd, m_drawArgs.buffer, 0, 1, sizeof(VkDrawIndirectCommand));
    vkCmdEndRendering(cmd);
  }

  // Read by the lens, bloom, local exposure, the tone mapper and the HDR output
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

  VkDescriptorImageInfo out = m_out.getColorStorageImageInfo(0);
  out.sampler               = resources.linearSampler;
  return out;
}
