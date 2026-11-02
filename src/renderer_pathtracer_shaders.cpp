/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// PathTracer's shaders and pipelines: compiling, the SPIR-V cache, the RQ and RTX pipelines and their
// warm-up, and the variants a frame switches between.

#include <fmt/format.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <span>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include <nvaftermath/aftermath.hpp>

#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/specialization.hpp>

#include "renderer_pathtracer.hpp"

#include "scene_shader_macros.hpp"
#include "pipeline_build_scope.hpp"

// Pre-compiled shaders
#include "_autogen/gltf_pathtrace.slang.h"
#if defined(USE_DLSS)
// The same path tracer built with USE_DLSS_SHADER=1 (CMakeLists.txt): what a DLSS frame asks for.
#include "_autogen/gltf_pathtrace.slang_dlss.h"
#endif

// Toggle deferred host operations for RTX pipeline creation.
// 1 = use VkDeferredOperationKHR to parallelize SPIR-V->ISA compilation across worker threads.
// 0 = create the pipeline synchronously on the calling thread (useful for comparing compile
//     times or as a workaround if the driver misbehaves with deferred ops).
// NOTE: the NVIDIA driver currently does not populate VkPipelineCreationFeedback when a
// deferred op is used, so the cache-hit flag is only meaningful with USE_DEFERRED_RTX_COMPILE=0.
#ifndef USE_DEFERRED_RTX_COMPILE
#define USE_DEFERRED_RTX_COMPILE 0
#endif

namespace {

// The RTX pipeline's shader stages and groups. createRtxPipeline() and the warm-up both build from
// it, so they ask the driver for the same pipeline (a different one would only miss the cache).
struct RtxStages
{
  enum Stage
  {
    eRaygen,
    eMiss,
    eShadowMiss,
    eClosestHit,
    eAnyHit,
    eShadowAnyHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  std::vector<VkRayTracingShaderGroupCreateInfoKHR>             groups;

  explicit RtxStages(VkShaderModule module)
  {
    stages[eRaygen]     = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_RAYGEN_BIT_KHR, module, "rgenMain"};
    stages[eMiss]         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR, module, "rmissMain"};
    stages[eClosestHit]   = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, module, "rchitMain"};
    stages[eAnyHit]       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, module, "rahitMain"};
    stages[eShadowMiss]   = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR, module, "rmissShadow"};
    stages[eShadowAnyHit] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, module, "rahitShadow"};

    VkRayTracingShaderGroupCreateInfoKHR group{
        .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .generalShader      = VK_SHADER_UNUSED_KHR,
        .closestHitShader   = VK_SHADER_UNUSED_KHR,
        .anyHitShader       = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR,
    };
    // Raygen, Miss-0, Shadow Miss-1
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    for(uint32_t general : {uint32_t(eRaygen), uint32_t(eMiss), uint32_t(eShadowMiss)})
    {
      group.generalShader = general;
      groups.push_back(group);
    }
    // Hit Group-0
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.anyHitShader     = eAnyHit;
    groups.push_back(group);
    // Hit Group-1 (shadow): any-hit only -- shadow rays skip closest-hit (TraceShadow)
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader     = eShadowAnyHit;
    groups.push_back(group);
  }
};

VkPipelineCreateFlags rtxPipelineFlags(bool opacityMicromaps, bool captureStatistics)
{
  // Allow the pipeline to use opacity micromaps (EXT_mesh_opacity_micromap); harmless when
  // geometry is not carrying a micromap. Only set when the device supports VK_KHR_opacity_micromap.
  VkPipelineCreateFlags flags = opacityMicromaps ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_KHR) : VkPipelineCreateFlags(0);
  if(captureStatistics)
    flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  return flags;
}

// A path-tracer pipeline from `module`, as createRqPipeline() or createRtxPipeline() builds it (no
// deferred operation): any thread, as it touches only the device and the pipeline cache.
// VK_NULL_HANDLE when the creation failed (`result` says why).
VkPipeline createPathTracePipeline(VkDevice              device,
                                   VkPipelineCache       cache,
                                   VkPipelineLayout      layout,
                                   VkShaderModule        module,
                                   bool                  rayQuery,
                                   int32_t               useSER,
                                   int32_t               transparency,
                                   VkPipelineCreateFlags rtxFlags,
                                   VkPipelineCreateFlags rqFlags,
                                   VkResult&             result)
{
  nvvk::Specialization specialization;
  specialization.add(0, useSER);  // USE_SER
#if defined(USE_DLSS)
  specialization.add(1, transparency);  // USE_DLSS_TRANSP
#else
  (void)transparency;
#endif
  VkPipeline pipeline = VK_NULL_HANDLE;
  if(rayQuery)
  {
    const VkComputePipelineCreateInfo info{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags  = rqFlags,
        .stage  = {.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module              = module,
                   .pName               = "computeMain",
                   .pSpecializationInfo = specialization.getSpecializationInfo()},
        .layout = layout,
    };
    result = vkCreateComputePipelines(device, cache, 1, &info, nullptr, &pipeline);
  }
  else
  {
    RtxStages table(module);
    table.stages[RtxStages::eRaygen].pSpecializationInfo = specialization.getSpecializationInfo();
    const VkRayTracingPipelineCreateInfoKHR info{
        .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .flags                        = rtxFlags,
        .stageCount                   = static_cast<uint32_t>(table.stages.size()),
        .pStages                      = table.stages.data(),
        .groupCount                   = static_cast<uint32_t>(table.groups.size()),
        .pGroups                      = table.groups.data(),
        .maxPipelineRayRecursionDepth = 2,
        .layout                       = layout,
    };
    result = vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, cache, 1, &info, nullptr, &pipeline);
  }
  if(result != VK_SUCCESS)
  {
    vkDestroyPipeline(device, pipeline, nullptr);
    return VK_NULL_HANDLE;
  }
  return pipeline;
}
}  // namespace

namespace {

constexpr uint64_t kSpirvCacheVersion   = 3;
constexpr size_t   kMaxCachedSpirvBytes = 64ULL * 1024 * 1024;
// Keys kept in shaders/ after pruning, least recently used dropped first. Each key is one
// variant (DLSS, wireframe, optimal-shader feature set, ...) of several MB.
constexpr size_t kMaxCachedSpirvKeys = 32;
// A source written this close to the compile's start may have been read half-saved; that compile
// is not cached. Also covers file systems with coarse timestamps.
constexpr auto kSourceSettleTime = std::chrono::seconds(2);
// Temporary files older than this belong to a process that was killed while writing.
constexpr auto kStaleTempAge = std::chrono::minutes(10);

uint64_t fnv1a(uint64_t hash, const void* data, size_t size)
{
  return nvvk::fnv1a64(data, size, hash);
}

uint64_t fnv1a(uint64_t hash, std::string_view text)
{
  hash = fnv1a(hash, text.data(), text.size());
  const char separator = 0;
  return fnv1a(hash, &separator, 1);
}

struct SourceStamp
{
  std::uintmax_t size    = 0;
  std::int64_t   mtimeNs = 0;
};

bool readStamp(const std::filesystem::path& path, SourceStamp& stamp)
{
  std::error_code ec;
  if(!std::filesystem::is_regular_file(path, ec))
    return false;
  stamp.size = std::filesystem::file_size(path, ec);
  if(ec)
    return false;
  const auto mtime = std::filesystem::last_write_time(path, ec);
  if(ec)
    return false;
  stamp.mtimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch()).count();
  return true;
}

std::filesystem::path canonicalDep(const std::filesystem::path& path)
{
  std::error_code       ec;
  std::filesystem::path canon = std::filesystem::weakly_canonical(path, ec);
  if(ec || canon.empty())
    return path;
  return canon;
}

void hashOption(uint64_t& hash, const slang::CompilerOptionEntry& option)
{
  const int name = static_cast<int>(option.name);
  const int kind = static_cast<int>(option.value.kind);
  hash           = fnv1a(hash, &name, sizeof(name));
  hash           = fnv1a(hash, &kind, sizeof(kind));
  hash           = fnv1a(hash, &option.value.intValue0, sizeof(option.value.intValue0));
  hash           = fnv1a(hash, &option.value.intValue1, sizeof(option.value.intValue1));
  hash           = fnv1a(hash, option.value.stringValue0 ? std::string_view(option.value.stringValue0) : std::string_view{});
  hash           = fnv1a(hash, option.value.stringValue1 ? std::string_view(option.value.stringValue1) : std::string_view{});
}

// Everything that changes the SPIR-V except the source text itself (that lives in .deps): macros,
// every compiler option (capabilities included), targets, the search paths, this executable, and
// the Slang build. Two trees or two builds therefore do not share a file.
uint64_t spirvCacheKey(nvslang::SlangCompiler& compiler)
{
  uint64_t hash = fnv1a(nvvk::kFnv1aOffset, std::string_view("pt-spirv"));
  hash          = fnv1a(hash, &kSpirvCacheVersion, sizeof(kSpirvCacheVersion));
  const char* tag = spGetBuildTagString();
  hash            = fnv1a(hash, tag ? std::string_view(tag) : std::string_view{});

  const std::filesystem::path executable = nvutils::getExecutablePath();
  hash                                   = fnv1a(hash, nvutils::utf8FromPath(executable));
  SourceStamp exeStamp{};
  if(readStamp(executable, exeStamp))
  {
    hash = fnv1a(hash, &exeStamp.size, sizeof(exeStamp.size));
    hash = fnv1a(hash, &exeStamp.mtimeNs, sizeof(exeStamp.mtimeNs));
  }

  for(const std::filesystem::path& searchPath : compiler.searchPaths())
    hash = fnv1a(hash, nvutils::utf8FromPath(canonicalDep(searchPath)));
  for(const slang::PreprocessorMacroDesc& macro : compiler.macros())
  {
    hash = fnv1a(hash, macro.name ? std::string_view(macro.name) : std::string_view{});
    hash = fnv1a(hash, macro.value ? std::string_view(macro.value) : std::string_view{});
  }
  for(const slang::CompilerOptionEntry& option : compiler.options())
    hashOption(hash, option);
  for(const slang::TargetDesc& target : compiler.targets())
  {
    const int format = static_cast<int>(target.format);
    const int floatMode = static_cast<int>(target.floatingPointMode);
    const int lineMode  = static_cast<int>(target.lineDirectiveMode);
    const int scalar    = target.forceGLSLScalarBufferLayout ? 1 : 0;
    hash                = fnv1a(hash, &format, sizeof(format));
    hash                = fnv1a(hash, &target.profile, sizeof(target.profile));
    hash                = fnv1a(hash, &target.flags, sizeof(target.flags));
    hash                = fnv1a(hash, &floatMode, sizeof(floatMode));
    hash                = fnv1a(hash, &lineMode, sizeof(lineMode));
    hash                = fnv1a(hash, &scalar, sizeof(scalar));
    for(uint32_t i = 0; i < target.compilerOptionEntryCount && target.compilerOptionEntries != nullptr; ++i)
      hashOption(hash, target.compilerOptionEntries[i]);
  }
  return hash;
}

constexpr std::string_view kSpvPrefix = "spv ";

// The blob a .deps names on its first line ("spv <hash>"), or an empty path.
std::filesystem::path blobOfDeps(std::istream& deps, const std::filesystem::path& dir, const std::string& stem)
{
  std::string line;
  if(!std::getline(deps, line))
    return {};
  if(!line.empty() && line.back() == '\r')
    line.pop_back();
  if(!line.starts_with(kSpvPrefix) || line.size() != kSpvPrefix.size() + 16)
    return {};
  return dir / (stem + "-" + line.substr(kSpvPrefix.size()) + ".spv");
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// What the live shader / pipeline lack for the current settings. Called every rendered frame, so it
// stays cheap: build the wanted key, then one lock and a handful of compares.
PathTracer::PendingCompileWork PathTracer::getPendingCompileWork(const Resources& resources)
{
  const VariantKey wanted = makeVariantKey(resources);
  const bool       useSER = m_useSER && m_supportSER;

  std::lock_guard<std::mutex> lock(m_compileMutex);
  const VkPipeline            live = (m_renderTechnique == RenderTechnique::RayQuery) ? m_rqPipeline : m_rtxPipeline;
  return {
      .recompile       = !wanted.sameShader(m_compiled),
      .stalePipeline   = useSER != m_pipelineUseSER || wanted.dlssTransparency != m_compiled.dlssTransparency,
      .missingPipeline = live == VK_NULL_HANDLE,
  };
}

std::optional<nvvkgltf::SceneFeatureSet> PathTracer::liveShaderFeatures()
{
  std::lock_guard<std::mutex> lock(m_compileMutex);
  if(!m_compiled.optimal || m_compiledIsFallback)
    return std::nullopt;
  return m_compiled.features;
}

bool PathTracer::wantOpacityMicromapShader(const Resources& resources)
{
  return resources.gpuView().hasOpacityMicromaps();
}

//--------------------------------------------------------------------------------------------------
// Ensure shader binaries and the selected technique pipeline are available. Runs on the compile
// worker (or inline when headless); getPendingCompileWork() says what is missing.
void PathTracer::ensureShadersAndPipelines(Resources& resources)
{
  const PipelineBuildScope building;  // /stats says so while this holds the application thread
  m_useSER = m_useSER && m_supportSER;  // a device without SER support can never turn it on

  // A shader macro changed (wireframe, visualization, opacity micromaps, optimal mode and the
  // scene's feature set, DLSS, denoiser guides). compileShader() also destroys the pipelines built from the old shader.
  PendingCompileWork work = getPendingCompileWork(resources);
  if(work.recompile)
  {
    VariantKey compiled;
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      compiled = m_compiled;
    }
    LOGI("[PathTracer] Shader recompile: %s\n", compiled.shaderDiff(makeVariantKey(resources)).c_str());
    if(m_busy)
      m_busy->setReason("Compiling Slang shaders...");
    {
      const ShaderWorkScope stage(ShaderWork::eCompile);
      compileShader(resources);
    }
    work = getPendingCompileWork(resources);
  }

  // SER (--ptUseSER) and the DLSS transparency mode (--dlssTransparency) are pipeline specialization constants, not shader macros, so they need no recompile --
  // only the pipelines built with the old values have to go. The SBT is generated from the RTX
  // pipeline, so it goes with it.
  if(work.stalePipeline)
  {
    // SYNC NOTE: the previous frame may still use these pipelines. prepareFrame() (or reloadShader())
    // drained the queue on the application thread before this ran, and no path-tracer frame is
    // recorded until it is done (the BusyState gates GltfRenderer::onRender). No wait here: this
    // runs on the compile worker while the application thread submits the BusyState's frames, and
    // host access to a queue must be externally synchronized.
    const VariantKey wantKey = makeVariantKey(resources);
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
      vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
      m_rqPipeline                = VK_NULL_HANDLE;
      m_rtxPipeline               = VK_NULL_HANDLE;
      m_pipelineUseSER            = m_useSER;
      m_compiled.dlssTransparency = wantKey.dlssTransparency;
      resources.allocator.destroyBuffer(m_sbtBuffer);
      m_sbtBuffer  = {};
      m_sbtRegions = {};
    }
    work = getPendingCompileWork(resources);
  }

  if(work.missingPipeline)
  {
    const ShaderWorkScope stage(ShaderWork::ePipeline);
    if(m_renderTechnique == RenderTechnique::RayQuery)
    {
      if(m_busy)
        m_busy->setReason("Creating Ray Query pipeline...");
      createRqPipeline(resources);
    }
    else
    {
      if(m_busy)
        m_busy->setReason("Creating RTX pipeline...");
      createRtxPipeline(resources);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Launches a background worker that (re)compiles the Slang shader and builds the ray
// tracing pipeline for the currently selected technique.
void PathTracer::startAsyncCompile(Resources& resources)
{
  if(m_compileThread.joinable())
    m_compileThread.join();

  // Without a BusyState nothing gates the next frame, which then joins this thread in
  // startAsyncCompile(): the compile still completes, synchronously.
  if(m_busy)
    m_busy->start("Preparing path tracer...");

  m_compileThread = std::thread([this, &resources]() {
    ensureShadersAndPipelines(resources);
    if(m_busy)
      m_busy->stop();
  });
}

//--------------------------------------------------------------------------------------------------
// Create the pipeline
void PathTracer::createPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);
  // Set 3 is the physical sky's scattering LUTs, read by aerial perspective. Always in the layout,
  // even for a scene that never selects that sky: a pipeline layout is built once and the
  // environment changes at runtime, so making the layout depend on it would mean rebuilding the
  // pipeline every time someone switched to an HDR.
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts{resources.descriptorSetLayout[0], resources.descriptorSetLayout[1],
                                                          resources.hdrIbl.getDescriptorSetLayout(),
                                                          resources.skyLutDescriptorSetLayout};

  // Creating the pipeline layout
  static_assert(sizeof(shaderio::PathtracePushConstant) <= 256, "the push constants Vulkan guarantees on NVIDIA");
  static_assert(offsetof(shaderio::PathtracePushConstant, adaptiveRelax) % 16 == 0, "a float4 array's alignment in the shader");
  VkPushConstantRange        pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PathtracePushConstant)};
  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = uint32_t(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  discardVariantJob();  // it builds against the layout
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);
}

//--------------------------------------------------------------------------------------------------
// After a renderer update the driver has none of the path tracer's pipelines, and creating one takes
// 10-25 s (the ray-query one the longest). The first path-traced frame asks for the embedded DLSS
// build and a still for the embedded build itself (isEmbeddedBaseVariant), and everything their
// pipelines are made from is known at attach: the embedded SPIR-V, the technique, the layout, SER and
// the DLSS transparency mode. So a thread of its own per build creates each into the pipeline cache
// now, while the app starts and a scene loads, and createRtxPipeline() or createRqPipeline() waits
// for them and finds it there. The threads touch only the device, the pipeline cache (pipeline
// creation may share one between threads) and the values they were handed.
void PathTracer::startPipelineWarm(Resources& resources)
{
#if defined(USE_DLSS)
  if(!m_supportSER || m_pipelineLayout == VK_NULL_HANDLE)
    return;
  std::lock_guard lock(m_warmMutex);
  if(!m_warmThreads.empty())
    return;
  const VkPipelineLayout      layout       = m_pipelineLayout;
  const int32_t               useSER       = m_useSER ? 1 : 0;
  const int32_t               transparency = static_cast<int32_t>(m_dlss->getTransparencyMode());
  const VkPipelineCreateFlags flags        = rtxPipelineFlags(resources.settings.opacityMicromapSupported, m_capturePipelineStats);
  const VkPipelineCreateFlags rqFlags = m_capturePipelineStats ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) : 0u;
  auto warm = [this, layout, useSER, transparency, flags, rqFlags](std::span<const uint32_t> code, const char* build, bool rayQuery) {
    const ShaderWarmScope    warming;  // /stats says so
    const auto               start = std::chrono::steady_clock::now();
    VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size_bytes(),
        .pCode    = code.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if(vkCreateShaderModule(m_device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
      return;
    VkResult         result   = VK_ERROR_UNKNOWN;
    const auto       built    = std::chrono::steady_clock::now();
    const VkPipeline pipeline = createPathTracePipeline(m_device, m_pipelineCache.getCache(), layout, module, rayQuery,
                                                        useSER, transparency, flags, rqFlags, result);
    const double     ms       = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - built).count();
    vkDestroyPipeline(m_device, pipeline, nullptr);
    vkDestroyShaderModule(m_device, module, nullptr);
    LOGI("%s pipeline warm-up (%s): %s (%.2f ms)\n", rayQuery ? "RQ" : "RTX", build, result == VK_SUCCESS ? "done" : "failed",
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    if(result == VK_SUCCESS)
      notePipelineBuild(nvvk::fnv1a64(code.data(), code.size_bytes()), rayQuery, useSER, transparency, rayQuery ? rqFlags : flags, ms);
    // the creation that follows is a cache hit and saves nothing, so the cache is saved here
    if(result == VK_SUCCESS)
      m_pipelineCache.saveAsync();
  };
  // each build with the technique its view renders with (onRender)
  const bool rqDlss  = m_supportRayQuery && m_techniqueDlss == RenderTechnique::RayQuery;
  const bool rqStill = m_supportRayQuery && m_techniqueStill == RenderTechnique::RayQuery;
  m_warmThreads.emplace_back(warm, std::span<const uint32_t>(gltf_pathtrace_slang_dlss, gltf_pathtrace_slang_dlss_sizeInBytes / sizeof(uint32_t)),
                             "DLSS", rqDlss);
  m_warmThreads.emplace_back(warm, std::span<const uint32_t>(gltf_pathtrace_slang, gltf_pathtrace_slang_sizeInBytes / sizeof(uint32_t)),
                             "still", rqStill);
#endif
}

int32_t PathTracer::transparencyMode() const
{
#if defined(USE_DLSS)
  return static_cast<int32_t>(m_dlss->getTransparencyMode());
#else
  return 0;
#endif
}

void PathTracer::rememberModuleCode(VkShaderModule module, const void* code, size_t bytes)
{
  std::lock_guard lock(m_moduleCodeMutex);
  m_moduleCode[module] = nvvk::fnv1a64(code, bytes);
}

uint64_t PathTracer::moduleCode(VkShaderModule module)
{
  std::lock_guard lock(m_moduleCodeMutex);
  const auto      it = m_moduleCode.find(module);
  return it != m_moduleCode.end() ? it->second : 0;
}

void PathTracer::notePipelineBuild(uint64_t code, bool rayQuery, int32_t useSER, int32_t transparency, VkPipelineCreateFlags flags, double ms)
{
  if(code == 0)
    return;  // a module made from code nobody hashed
  const uint64_t state[] = {code, rayQuery ? 1u : 0u, uint64_t(useSER), uint64_t(transparency), uint64_t(flags), m_driverVersion};
  if(m_buildRecord.note(nvvk::fnv1a64(state, sizeof(state)), ms))
    LOGW("[PathTracer] A pipeline built before compiled again (%.0f ms): the driver's shader cache is full\n", ms);
}

void PathTracer::joinPipelineWarm()
{
  std::vector<std::thread> warm;
  {
    std::lock_guard lock(m_warmMutex);
    warm.swap(m_warmThreads);
  }
  for(std::thread& t : warm)
    if(t.joinable())
      t.join();
}

//--------------------------------------------------------------------------------------------------
// Create the compute pipeline
void PathTracer::createRqPipeline(Resources& /*resources*/)
{
  SCOPED_TIMER(__FUNCTION__);
  // as createRtxPipeline(): a warm-up of this pipeline still running is waited for, not repeated
  joinPipelineWarm();

  nvvk::Specialization specialization;
  specialization.add(0, m_useSER ? 1 : 0);  // USE_SER
#if defined(USE_DLSS)
  specialization.add(1, static_cast<int32_t>(m_dlss->getTransparencyMode()));  // USE_DLSS_TRANSP
#endif

  VkPipelineShaderStageCreateInfo shaderStage{
      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
      .module              = m_shaderModule,
      .pName               = "computeMain",
      .pSpecializationInfo = specialization.getSpecializationInfo(),
  };

  // Query the driver about pipeline cache hit / compilation duration (core in Vulkan 1.3)
  VkPipelineCreationFeedback           feedback{};
  VkPipelineCreationFeedbackCreateInfo feedbackInfo{
      .sType                     = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
      .pPipelineCreationFeedback = &feedback,
  };

  VkComputePipelineCreateInfo cpCreateInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext  = &feedbackInfo,
      .flags  = m_capturePipelineStats ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) : 0u,
      .stage  = shaderStage,
      .layout = m_pipelineLayout,
  };

  // NOTE: if the creation is slow, disable the validation layers for faster creation (--vvl 0)
  VkPipeline rqPipeline = VK_NULL_HANDLE;
  const auto rqStart    = std::chrono::steady_clock::now();
  NVVK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache.getCache(), 1, &cpCreateInfo, nullptr, &rqPipeline));
  NVVK_DBG_NAME(rqPipeline);
  notePipelineBuild(moduleCode(m_shaderModule), true, m_useSER ? 1 : 0, transparencyMode(), cpCreateInfo.flags,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rqStart).count());

  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    m_rqPipeline = rqPipeline;
    ++m_pipelineGeneration;
  }

  const bool valid = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
  const bool hit   = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
  LOGI("RQ pipeline creation: %s, %s (%.2f ms)\n", valid ? "valid" : "invalid", hit ? "CACHE HIT" : "cache miss",
       feedback.duration / 1e6);
  logPipelineStatistics(rqPipeline, "RQ");
  if(!hit)
    m_pipelineCache.saveAsync();
}

//--------------------------------------------------------------------------------------------------
// One log line per pipeline executable with the statistics the driver reports for it (register
// count, spills, ...): what decides occupancy, and so the first thing to read after a shader edit
// that should change it. Only when VK_KHR_pipeline_executable_properties is enabled, which also
// makes the pipelines capture them (VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR). NVIDIA's driver
// reports the ray-query (compute) pipeline but no executables for the ray tracing pipeline.
void PathTracer::logPipelineStatistics(VkPipeline pipeline, const char* label) const
{
  if(!m_capturePipelineStats)
    return;
  const VkPipelineInfoKHR pipelineInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipeline};
  uint32_t                executableCount = 0;
  vkGetPipelineExecutablePropertiesKHR(m_device, &pipelineInfo, &executableCount, nullptr);
  std::vector<VkPipelineExecutablePropertiesKHR> executables(executableCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  vkGetPipelineExecutablePropertiesKHR(m_device, &pipelineInfo, &executableCount, executables.data());
  for(uint32_t i = 0; i < executableCount; ++i)
  {
    const VkPipelineExecutableInfoKHR executableInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipeline, .executableIndex = i};
    uint32_t statCount = 0;
    vkGetPipelineExecutableStatisticsKHR(m_device, &executableInfo, &statCount, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> stats(statCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    vkGetPipelineExecutableStatisticsKHR(m_device, &executableInfo, &statCount, stats.data());
    std::string line;
    for(size_t s = 0; s < stats.size(); s++)
    {
      const VkPipelineExecutableStatisticKHR& stat = stats[s];
      line += fmt::format("{}{}=", line.empty() ? "" : ", ", stat.name);
      switch(stat.format)
      {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
          line += stat.value.b32 ? "true" : "false";
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
          line += std::to_string(stat.value.i64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
          line += std::to_string(stat.value.u64);
          break;
        default:
          line += fmt::format("{:.3g}", stat.value.f64);
          break;
      }
    }
    LOGI("%s pipeline executable %u (%s): %s\n", label, i, executables[i].name, line.c_str());
  }
}

//--------------------------------------------------------------------------------------------------
// Create the RTX pipeline
void PathTracer::createRtxPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);
  // A warm-up of this pipeline still running: wait for it and find it in the cache, rather than
  // have the driver compile it a second time beside it
  joinPipelineWarm();
  // Creating all shaders
  RtxStages table(m_shaderModule);
  auto&     stages        = table.stages;
  auto&     shader_groups = table.groups;

  // Shader Execution Reorder (SER)
  nvvk::Specialization specialization;
  specialization.add(0, m_useSER ? 1 : 0);  // USE_SER
#if defined(USE_DLSS)
  specialization.add(1, static_cast<int32_t>(m_dlss->getTransparencyMode()));  // USE_DLSS_TRANSP
#endif
  stages[RtxStages::eRaygen].pSpecializationInfo = specialization.getSpecializationInfo();


  // Query the driver about pipeline cache hit / compilation duration (core in Vulkan 1.3)
  VkPipelineCreationFeedback           feedback{};
  VkPipelineCreationFeedbackCreateInfo feedbackInfo{
      .sType                     = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
      .pPipelineCreationFeedback = &feedback,
  };

  const VkPipelineCreateFlags flags = rtxPipelineFlags(resources.settings.opacityMicromapSupported, m_capturePipelineStats);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .pNext                        = &feedbackInfo,
      .flags                        = flags,
      .stageCount                   = static_cast<uint32_t>(stages.size()),  // Stages are shaders
      .pStages                      = stages.data(),
      .groupCount                   = static_cast<uint32_t>(shader_groups.size()),
      .pGroups                      = shader_groups.data(),
      .maxPipelineRayRecursionDepth = 2,  // Ray depth
      .layout                       = m_pipelineLayout,
  };
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
    m_rtxPipeline = VK_NULL_HANDLE;
  }


  // Time the create (+ join, when deferred) wall-clock ourselves. The NVIDIA driver does
  // not populate VkPipelineCreationFeedback when a VkDeferredOperationKHR is used, so
  // feedback.duration and the VALID_BIT cannot be relied on on the deferred path.
  // NOTE: if the creation is slow, disable the validation layers for faster creation (--vvl 0)
  const auto createStart = std::chrono::steady_clock::now();
  VkPipeline rtxPipeline = VK_NULL_HANDLE;

#if USE_DEFERRED_RTX_COMPILE
  // Deferred host operation: lets the driver split the SPIR-V->ISA compile across worker
  // threads. On a cache hit the call returns VK_SUCCESS immediately and the join loop is a
  // no-op, so this is safe to do unconditionally.
  VkDeferredOperationKHR deferredOp = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateDeferredOperationKHR(m_device, nullptr, &deferredOp));

  VkResult createResult = vkCreateRayTracingPipelinesKHR(m_device, deferredOp, m_pipelineCache.getCache(), 1,
                                                         &rtPipelineCreateInfo, nullptr, &rtxPipeline);

  if(createResult == VK_OPERATION_DEFERRED_KHR)
  {
    // Parallelize the compile across worker threads. We spawn (maxConcurrency - 1) workers
    // and have the calling thread also participate, so we use all available parallelism
    // reported by the driver without over-subscribing.
    const uint32_t maxConcurrency = vkGetDeferredOperationMaxConcurrencyKHR(m_device, deferredOp);
    const uint32_t hwConcurrency  = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t threadCount    = std::min(maxConcurrency, hwConcurrency);

    auto joinLoop = [this, deferredOp]() {
      VkResult r;
      do
      {
        r = vkDeferredOperationJoinKHR(m_device, deferredOp);
      } while(r == VK_THREAD_IDLE_KHR);
      // VK_THREAD_DONE_KHR: no more work available for this thread (others may still be running)
      // VK_SUCCESS: whole deferred op finished
    };

    std::vector<std::thread> workers;
    workers.reserve(threadCount > 0 ? threadCount - 1 : 0);
    for(uint32_t i = 1; i < threadCount; ++i)
      workers.push_back(std::thread(joinLoop));
    joinLoop();
    for(auto& t : workers)
      t.join();

    // All threads returned either VK_THREAD_DONE_KHR or VK_SUCCESS; poll the final result
    // (per spec, VK_NOT_READY means workers are done but some bookkeeping remains).
    VkResult finalResult;
    do
    {
      finalResult = vkGetDeferredOperationResultKHR(m_device, deferredOp);
    } while(finalResult == VK_NOT_READY);
    NVVK_CHECK(finalResult);
  }
  else
  {
    NVVK_CHECK(createResult);
  }

  vkDestroyDeferredOperationKHR(m_device, deferredOp, nullptr);
  const bool wasDeferred = (createResult == VK_OPERATION_DEFERRED_KHR);
#else
  // Synchronous path: no deferred op, driver is expected to populate VkPipelineCreationFeedback.
  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, m_pipelineCache.getCache(), 1,
                                            &rtPipelineCreateInfo, nullptr, &rtxPipeline));
  const bool wasDeferred = false;
#endif

  NVVK_DBG_NAME(rtxPipeline);

  const double elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - createStart).count();

  // Log pipeline creation status and whether deferred compilation was used. On the deferred
  // path the driver may skip filling in VkPipelineCreationFeedback, so we rely on our own
  // wall-clock timing and treat the cache-hit bit as a best-effort hint.
  const bool  valid = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
  const char* cacheStatus =
      valid ? ((feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) ? "CACHE HIT" : "cache miss") :
              "cache status unknown";
  LOGI("RTX pipeline creation: %s, deferred=%s (%.2f ms)\n", cacheStatus, wasDeferred ? "yes" : "no", elapsedMs);
  notePipelineBuild(moduleCode(m_shaderModule), false, m_useSER ? 1 : 0, transparencyMode(), flags, elapsedMs);
  logPipelineStatistics(rtxPipeline, "RTX");
  // Unknown feedback (the deferred path leaves it unset) is not a hit, so it still saves.
  if((feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) == 0)
    m_pipelineCache.saveAsync();

  // Create the Shading Binding Table
  {
    nvvk::Buffer                sbtBuffer;
    nvvk::SBTGenerator::Regions sbtRegions;
    createSbt(resources, rtxPipeline, m_shaderModule, sbtBuffer, sbtRegions);
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      resources.allocator.destroyBuffer(m_sbtBuffer);
      m_rtxPipeline = rtxPipeline;
      m_sbtBuffer   = sbtBuffer;
      m_sbtRegions  = sbtRegions;
      ++m_pipelineGeneration;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Run once from onAttach, before any compile.
// Keeps the shaders/ folder bounded: the kMaxCachedSpirvKeys most recently used keys (a hit
// touches its .deps), blobs no .deps names, and temporary files a killed writer left. Blobs and
// temporary files younger than kStaleTempAge are left alone: another renderer may be between
// writing a blob and publishing its .deps.
void PathTracer::pruneShaderCache() const
{
  if(m_shaderCacheDir.empty())
    return;
  // pipeline_cache.bin and its temporaries sit in the parent folder (the exe's without --cacheDir).
  nvvk::removeStaleTempFiles(m_shaderCacheDir.parent_path(), kStaleTempAge, "pipeline_cache.bin.");
  nvvk::removeStaleTempFiles(m_shaderCacheDir, kStaleTempAge, "pt-");

  struct Entry
  {
    std::filesystem::path                deps;
    std::filesystem::file_time_type      used;
  };
  std::vector<Entry>                    keys;
  std::vector<std::filesystem::path>    blobs;
  std::error_code                       ec;
  for(std::filesystem::directory_iterator it(m_shaderCacheDir, ec), end; !ec && it != end; it.increment(ec))
  {
    std::error_code entryEc;
    if(!it->is_regular_file(entryEc))
      continue;
    const auto extension = it->path().extension();
    if(extension == ".deps")
      keys.push_back({it->path(), it->last_write_time(entryEc)});
    else if(extension == ".spv")
      blobs.push_back(it->path());
  }

  std::sort(keys.begin(), keys.end(), [](const Entry& a, const Entry& b) { return a.used > b.used; });
  std::unordered_set<std::string> named;
  for(size_t i = 0; i < keys.size(); ++i)
  {
    std::error_code removeEc;
    if(i >= kMaxCachedSpirvKeys)
    {
      std::filesystem::remove(keys[i].deps, removeEc);
      continue;
    }
    std::ifstream               deps(keys[i].deps);
    const std::filesystem::path blob = blobOfDeps(deps, m_shaderCacheDir, nvutils::utf8FromPath(keys[i].deps.stem()));
    if(!blob.empty())
      named.insert(nvutils::utf8FromPath(blob.filename()));
  }

  const auto cutoff = std::filesystem::file_time_type::clock::now() - kStaleTempAge;
  for(const std::filesystem::path& blob : blobs)
  {
    std::error_code blobEc;
    if(named.contains(nvutils::utf8FromPath(blob.filename())))
      continue;
    const auto written = std::filesystem::last_write_time(blob, blobEc);
    if(!blobEc && written < cutoff)
      std::filesystem::remove(blob, blobEc);
  }
}

//--------------------------------------------------------------------------------------------------
// .deps starts with "spv <hash>" naming the blob written before it. The following lines are
// "size mtimeNs path" (canonical when written) and match only when each file is unchanged.
// A hit touches the .deps, which is what pruneShaderCache orders by.
bool PathTracer::loadCachedSpirv(uint64_t key, std::vector<uint32_t>& spirv) const
{
  spirv.clear();
  if(m_shaderCacheDir.empty())
    return false;

  const std::string           stem     = fmt::format("pt-{:016x}", key);
  const std::filesystem::path depsPath = m_shaderCacheDir / (stem + ".deps");

  std::ifstream deps(depsPath);
  if(!deps)
    return false;
  const std::filesystem::path spvPath = blobOfDeps(deps, m_shaderCacheDir, stem);
  if(spvPath.empty())
    return false;

  std::string line;
  bool        any = false;
  while(std::getline(deps, line))
  {
    if(!line.empty() && line.back() == '\r')
      line.pop_back();
    if(line.empty())
      continue;
    const auto sp1 = line.find(' ');
    const auto sp2 = sp1 == std::string::npos ? std::string::npos : line.find(' ', sp1 + 1);
    if(sp1 == std::string::npos || sp2 == std::string::npos)
      return false;
    SourceStamp expected{};
    try
    {
      expected.size    = static_cast<std::uintmax_t>(std::stoull(line.substr(0, sp1)));
      expected.mtimeNs = std::stoll(line.substr(sp1 + 1, sp2 - sp1 - 1));
    }
    catch(const std::exception&)
    {
      return false;
    }
    const std::filesystem::path path = nvutils::pathFromUtf8(line.substr(sp2 + 1));
    SourceStamp                 actual{};
    if(path.empty() || !readStamp(path, actual) || actual.size != expected.size || actual.mtimeNs != expected.mtimeNs)
      return false;
    any = true;
  }
  if(!any)
    return false;

  std::error_code ec;
  const auto      spvSize = std::filesystem::file_size(spvPath, ec);
  if(ec || spvSize == 0 || spvSize > kMaxCachedSpirvBytes || (spvSize % sizeof(uint32_t)) != 0)
    return false;

  std::ifstream spv(spvPath, std::ios::binary);
  if(!spv)
    return false;
  spirv.resize(static_cast<size_t>(spvSize / sizeof(uint32_t)));
  spv.read(reinterpret_cast<char*>(spirv.data()), static_cast<std::streamsize>(spvSize));
  if(!spv)
  {
    spirv.clear();
    return false;
  }
  deps.close();
  std::filesystem::last_write_time(depsPath, std::filesystem::file_time_type::clock::now(), ec);
  return true;
}

//--------------------------------------------------------------------------------------------------
// compileFile loads the entry as a source string, so the module's dependency list can omit
// gltf_pathtrace.slang itself. The resolved entry path is always recorded. A source written at
// or after `compileStart` (less kSourceSettleTime) may not be the text the compile read, so that
// compile is not cached; the next one is.
void PathTracer::storeCachedSpirv(uint64_t key, nvslang::SlangCompiler& compiler, std::filesystem::file_time_type compileStart)
{
  if(m_shaderCacheDir.empty())
    return;
  const uint32_t* spirv      = compiler.getSpirv();
  const size_t    spirvBytes = compiler.getSpirvSize();
  if(spirv == nullptr || spirvBytes == 0 || (spirvBytes % sizeof(uint32_t)) != 0 || spirvBytes > kMaxCachedSpirvBytes)
    return;

  std::vector<std::filesystem::path> deps;
  std::unordered_set<std::string>    seen;  // canonical paths already in deps
  auto                               addDep = [&](const std::filesystem::path& path) {
    if(path.empty())
      return;
    const std::filesystem::path canon = canonicalDep(path);
    const std::string           utf8  = nvutils::utf8FromPath(canon);
    if(utf8.empty() || !seen.insert(utf8).second)
      return;
    deps.push_back(canon);
  };

  addDep(nvutils::findFile("gltf_pathtrace.slang", compiler.searchPaths(), false));
  if(slang::IModule* module = compiler.getSlangModule())
  {
    const int32_t count = static_cast<int32_t>(module->getDependencyFileCount());
    for(int32_t i = 0; i < count; ++i)
    {
      const char* dep = module->getDependencyFilePath(i);
      if(dep != nullptr && dep[0] != '\0')
        addDep(nvutils::pathFromUtf8(dep));
    }
  }
  if(deps.empty())
  {
    LOGW("Slang SPIR-V cache: no source files to stamp; not writing the cache\n");
    return;
  }

  // loadModuleFromSourceString records the entry as a content hash, not a path. Those entries
  // are not files and are skipped.
  const std::int64_t settledNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>((compileStart - kSourceSettleTime).time_since_epoch()).count();
  std::ostringstream text;
  for(const std::filesystem::path& path : deps)
  {
    SourceStamp stamp{};
    if(!readStamp(path, stamp))
      continue;
    const std::string utf8 = nvutils::utf8FromPath(path);
    if(stamp.mtimeNs >= settledNs)
    {
      LOGW("Slang SPIR-V cache: '%s' changed around the compile; not writing the cache\n", utf8.c_str());
      return;
    }
    text << stamp.size << ' ' << stamp.mtimeNs << ' ' << utf8 << '\n';
  }
  const std::string sources = text.str();
  if(sources.empty())
  {
    LOGW("Slang SPIR-V cache: no source files to stamp; not writing the cache\n");
    return;
  }

  const uint64_t    spirvHash = fnv1a(nvvk::kFnv1aOffset, spirv, spirvBytes);
  const std::string spirvKey  = fmt::format("{:016x}", spirvHash);
  const std::string body      = std::string(kSpvPrefix) + spirvKey + "\n" + sources;

  std::error_code             ec;
  const std::string           stem     = fmt::format("pt-{:016x}", key);
  const std::filesystem::path spvPath  = m_shaderCacheDir / (stem + "-" + spirvKey + ".spv");
  const std::filesystem::path depsPath = m_shaderCacheDir / (stem + ".deps");
  std::filesystem::path       previousBlob;
  {
    std::ifstream previous(depsPath);
    if(previous)
      previousBlob = blobOfDeps(previous, m_shaderCacheDir, stem);
  }
  // The blob is named by its contents, so replacing .deps is the only publish step. A reader
  // keeps seeing the previous blob until that rename, and two writers do not share a .tmp name.
  std::string error;
  if(!std::filesystem::is_regular_file(spvPath, ec) && !nvvk::writeFileAtomic(spvPath, spirv, spirvBytes, &error))
  {
    LOGW("Slang SPIR-V cache: %s\n", error.c_str());
    return;
  }
  if(!nvvk::writeFileAtomic(depsPath, body.data(), body.size(), &error))
  {
    LOGW("Slang SPIR-V cache: %s\n", error.c_str());
    return;
  }
  // Only this key's .deps names its blobs. A reader that opened the old .deps just before the
  // replace finds the blob gone and compiles instead.
  if(!previousBlob.empty() && previousBlob != spvPath)
    std::filesystem::remove(previousBlob, ec);
}

//--------------------------------------------------------------------------------------------------
// Compile the shader
bool PathTracer::reloadShader(Resources& resources)
{
  if(m_compileThread.joinable())
    m_compileThread.join();

  // SYNC NOTE: User-initiated shader reload — cached pipelines may still be in flight from a
  // recent variant switch; live handles are destroyed in compileShader() below.
  NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
  discardVariantJob();  // built from the sources before the edit
  m_failedVariant.reset();
  destroyVariantCache(resources);
  m_skipVariantCache      = true;
  const bool compiledFile = compileShader(resources, true);
  m_skipVariantCache      = false;
  return compiledFile;
}

//--------------------------------------------------------------------------------------------------
// Compile the shader
bool PathTracer::compileShader(Resources& resources, bool fromFile)
{
  SCOPED_TIMER(__FUNCTION__);
  // Variant cache: try to find a previously-compiled shader + pipelines for the variant the
  // user wants this frame. On hit we skip both the Slang recompile and the RT pipeline link
  // (the slow part - the driver can take several seconds per RTX pipeline). On miss we
  // proceed with the normal compile path and the new entry will be stored on the next
  // variant switch.
  // The key is read once: it depends on the frame (see makeVariantKey()), and the macros below and
  // m_compiled must describe the same variant.
  const VariantKey key = makeVariantKey(resources);
  if(fromFile && !m_skipVariantCache)
  {
    // Every caller drained the queue first (prepareFrame, reloadShader) or has rendered nothing yet (onAttach)
    if(swapVariant(resources, key, /*gpuIdle=*/true))
    {
      LOGI("[PathTracer] Variant cache hit.\n");
      return true;
    }
    // Cache miss: handles were cleared; will rebuild below.
  }

  // The SPIR-V the module is created from: the embedded build unless the embedded DLSS build, the
  // disk cache or a from-file compile replaces it below
  std::span<const uint32_t> spirv(gltf_pathtrace_slang, gltf_pathtrace_slang_sizeInBytes / sizeof(uint32_t));

  // Compile from shader file if requested, used when reloading the shader
  bool                  compiledFromFile = false;
  std::vector<uint32_t> cachedSpirv;  // A disk-cache hit; spirv points here until the module exists
  if(fromFile)
  {
    SCOPED_TIMER("Slang compile from file");

    resources.slangCompiler.clearMacros();

    const std::vector<std::pair<std::string, std::string>> macros = variantMacros(key);
    if(key.optimal)
    {
      const auto& fs = key.features;
      LOGI(
          "[PathTracer] Optimal shader: scene uses [%s]; %d GLTF_USE_* gates off "
          "(each set explicitly to 0 or 1).\n",
          fs.toString().c_str(), fs.unusedExtensionCount());
    }

    for(const auto& [k, v] : macros)
      resources.slangCompiler.addMacro({k.c_str(), v.c_str()});

    // A user-forced reload (reloadShader) compiles from source even when the embedded DLSS build
    // or the disk cache would serve, and the result replaces that disk entry.
    const uint64_t spirvKey = spirvCacheKey(resources.slangCompiler);
    if(!m_skipVariantCache && isEmbeddedDlssVariant(resources))
    {
#if defined(USE_DLSS)
      spirv            = {gltf_pathtrace_slang_dlss, gltf_pathtrace_slang_dlss_sizeInBytes / sizeof(uint32_t)};
      compiledFromFile = true;
      LOGI("Slang: embedded DLSS path tracer, no compile\n");
#endif
    }
    else if(!m_skipVariantCache && isEmbeddedBaseVariant(resources))
    {
      // the variant a still hands off from (DLSS off, guides on) is the embedded build itself
      spirv            = {gltf_pathtrace_slang, gltf_pathtrace_slang_sizeInBytes / sizeof(uint32_t)};
      compiledFromFile = true;
      LOGI("Slang: embedded path tracer, no compile\n");
    }
    else if(!m_skipVariantCache && loadCachedSpirv(spirvKey, cachedSpirv))
    {
      spirv            = cachedSpirv;
      compiledFromFile = true;
      LOGI("Slang SPIR-V cache hit %016llx\n", static_cast<unsigned long long>(spirvKey));
#if defined(AFTERMATH_AVAILABLE)
      // The compile callback in GltfRenderer does not run when the SPIR-V is loaded from disk.
      AftermathCrashTracker::getInstance().addShaderBinary(std::span<const uint32_t>(cachedSpirv.data(), cachedSpirv.size()));
#endif
    }
    else if(const auto compileStart = std::filesystem::file_time_type::clock::now();
            resources.slangCompiler.compileFile("gltf_pathtrace.slang"))
    {
      LOGI("Slang SPIR-V cache %s %016llx\n", m_skipVariantCache ? "bypassed (reload)" : "miss",
           static_cast<unsigned long long>(spirvKey));
      spirv            = {resources.slangCompiler.getSpirv(), resources.slangCompiler.getSpirvSize() / sizeof(uint32_t)};
      compiledFromFile = true;
      storeCachedSpirv(spirvKey, resources.slangCompiler, compileStart);
    }
    else
    {
      LOGI("Slang SPIR-V cache miss %016llx\n", static_cast<unsigned long long>(spirvKey));
      // Falls back to the embedded SPIR-V, which is built with USE_OMM=0: in a scene with
      // micromaps its ray queries lack the OpacityMicromapIdKHR opt-in. The fallback stands in for
      // the wanted variant (m_compiledIsFallback) until a setting changes or the user reloads.
      LOGW("Error compiling gltf_pathtrace.slang; the embedded shader stands in until the next change or reload\n");
    }
  }

  // Create a shader module
  {
    SCOPED_TIMER("Create Shader Module");
    vkDestroyShaderModule(m_device, m_shaderModule, nullptr);

    VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size_bytes(),
        .pCode    = spirv.data(),
    };
    NVVK_CHECK(vkCreateShaderModule(m_device, &moduleInfo, nullptr, &m_shaderModule));
    NVVK_DBG_NAME(m_shaderModule);
    rememberModuleCode(m_shaderModule, spirv.data(), spirv.size_bytes());
  }

  // Record which variant this compile is, so getPendingCompileWork() can tell whether the settings
  // still match it. A from-file compile (or the embedded DLSS build standing in for one) follows
  // the live settings; so does a failed one, as the variant its fallback stands in for, or every
  // frame would retry the compile. The embedded SPIR-V at startup mirrors the CMake build (see
  // CMakeLists.txt): not optimal, all-on features, USE_GUIDE_SHADER=1 iff DLSS/OptiX/OIDN is
  // available at build time, and USE_DLSS_SHADER=0 (DLSS is never active at startup; when it
  // engages, the next compile takes the embedded DLSS build). Its pipelines are specialized with
  // the live transparency mode, which the key keeps.
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
#if defined(USE_DLSS) || defined(USE_OPTIX_DENOISER) || defined(USE_OIDN_DENOISER)
    constexpr bool kEmbeddedDlssGuide = true;
#else
    constexpr bool kEmbeddedDlssGuide = false;
#endif
    m_compiledIsFallback = fromFile && !compiledFromFile;
    if(compiledFromFile || m_compiledIsFallback)
      m_compiled = key;
    else
      m_compiled = VariantKey{.dlssGuide = kEmbeddedDlssGuide, .dlssTransparency = key.dlssTransparency};

    // Destroy pipeline since there is a new shader
    destroyPipelinesLocked();
  }

  return !fromFile || compiledFromFile;
}

//--------------------------------------------------------------------------------------------------
// The embedded DLSS build is CMake's path tracer flags plus USE_DLSS_SHADER=1: guide buffers on
// (USE_DLSS implies them), SER available, and wireframe, visualization, opacity micromaps and the
// optimal shader at their off defaults. Any other variant still compiles from source.
bool PathTracer::isEmbeddedDlssVariant(const Resources& resources) const
{
#if defined(USE_DLSS)
  if(!m_supportSER)
    return false;
  return makeVariantKey(resources).sameShader(VariantKey{.dlss = true, .dlssGuide = true});
#else
  return false;
#endif
}

//--------------------------------------------------------------------------------------------------
// The embedded build itself is the same flags without USE_DLSS_SHADER: DLSS off and guide buffers on
// when a denoiser is built in, which is what a still asks for before it hands off to OptiX. Taking
// it saves the Slang compile and, as its SPIR-V is the one the warm-up built a pipeline from, the
// pipeline build too.
bool PathTracer::isEmbeddedBaseVariant(const Resources& resources) const
{
  if(!m_supportSER)
    return false;
#if defined(USE_DLSS) || defined(USE_OPTIX_DENOISER) || defined(USE_OIDN_DENOISER)
  return makeVariantKey(resources).sameShader(VariantKey{.dlssGuide = true});
#else
  return makeVariantKey(resources).sameShader(VariantKey{});
#endif
}

//--------------------------------------------------------------------------------------------------
// The variant the current settings ask for -- the variant-cache key compileShader() looks up.
// DLSS's guide captures cost every sample, so they are compiled in only for the frames DLSS renders:
// with OptiX or OIDN also on, frame 0 and the hand-off delay's (m_dlssShaderFrame), while the
// accumulation's frames after them run the regular variant. Each camera start and stop is then a
// cached-variant swap in prepareFrame(); a moving camera stays on DLSS's.
// While DLSS is still loading, its variant is compiled ahead of its first frame.
PathTracer::VariantKey PathTracer::makeVariantKey(const Resources& resources) const
{
  return VariantKey{
      resources.settings.wireframe,
      resources.settings.visualization != shaderio::Visualization::eRendered,
      wantOpacityMicromapShader(resources),
      resources.settings.optimalShader,
      isDlssEnabled() && m_dlssShaderFrame,
      resources.currentFeatureSet.has(nvvkgltf::SceneFeatureSet::eDlssGuide),
      resources.settings.optimalShader ? resources.currentFeatureSet : nvvkgltf::SceneFeatureSet{},
#if defined(USE_DLSS)
      m_dlss ? static_cast<int>(m_dlss->getTransparencyMode()) : 0,
#else
      0,
#endif
  };
}

std::string PathTracer::VariantKey::shaderDiff(const VariantKey& to) const
{
  std::string diff;
  auto        add = [&](const char* name, bool a, bool b) {
    if(a != b)
      diff += std::string(diff.empty() ? "" : ", ") + name + (a ? " 1->0" : " 0->1");
  };
  add("wireframe", wireframe, to.wireframe);
  add("visualize", visualize, to.visualize);
  add("omm", omm, to.omm);
  add("optimal", optimal, to.optimal);
  add("dlss", dlss, to.dlss);
  add("dlssGuide", dlssGuide, to.dlssGuide);
  if(optimal && to.optimal && features != to.features)
    diff += std::string(diff.empty() ? "" : ", ") + "features [" + features.toString() + "] -> [" + to.features.toString() + "]";
  return diff.empty() ? "no shader field changed" : diff;
}

//--------------------------------------------------------------------------------------------------
// Swap in the wanted variant if the cache already holds it. A hit is only a handle swap, so
// onRender() does it inline instead of going through the async worker and the BusyState, whose
// completion resets the accumulation. On a miss nothing is touched: the live handles stay put for
// compileShader() to park.
bool PathTracer::tryRestoreCachedVariant(Resources& resources)
{
  if(m_skipVariantCache)
    return false;
  const VariantKey key = makeVariantKey(resources);
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    if(std::none_of(m_variantCache.begin(), m_variantCache.end(),
                    [&](const VariantCacheEntry& e) { return e.key == key; }))
      return false;
  }
  const bool hit = swapVariant(resources, key, /*gpuIdle=*/false);  // the previous frame may be in flight
  if(hit)  // debug level: with DLSS-RR and OptiX or OIDN, every camera start and stop swaps
    LOGD("[PathTracer] Variant cache hit (swapped in place).\n");
  return hit;
}

void PathTracer::destroyPipelinesLocked()
{
  vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
  m_rtxPipeline = VK_NULL_HANDLE;
  vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
  m_rqPipeline = VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Variant cache: park the active shader+pipelines+SBT under the variant they were built for, and
// try to restore handles for a target variant. Returns true on cache hit (caller can skip the
// expensive Slang recompile + RT pipeline link + SBT build).
//
// Reasoning for moving handles rather than copying: VkPipeline / VkShaderModule / nvvk::Buffer
// are not reference-counted across copies. We always have exactly one "live" instance per
// variant; the cache holds the others. On cache hit the current active handles are swapped with
// the cached ones (active handles for the OLD variant get parked, NEW handles become active).
//
// The SBT belongs to the RTX pipeline that built it (shader group handles are pipeline-specific)
// so we have to swap them together. Forgetting to swap the SBT causes a crash on the first
// dispatched ray because traversal jumps to handles from a different pipeline.
//
// Nothing is destroyed on a plain hit, which is why the per-frame swap needs no queue wait. What is
// destroyed (a duplicate parked entry, pipelines built with the other SER value, an LRU victim) may
// still be used by the frame in flight, so without gpuIdle it is released through the app's frame
// ring, which runs the callback once that frame's fence has been waited on.
bool PathTracer::swapVariant(Resources& resources, const VariantKey& newKey, bool gpuIdle)
{
  std::lock_guard<std::mutex> lock(m_compileMutex);

  auto release = [&](VkShaderModule module, VkPipeline rtx, VkPipeline rq, nvvk::Buffer sbt) {
    releaseVariant(resources, module, rtx, rq, sbt, gpuIdle);
  };

  // 1) Park the currently-active shader/pipelines/SBT so we don't leak them. The embedded fallback
  // standing in for a variant that failed to compile is not that variant: it is released instead,
  // so the next request for the variant compiles it again.
  const VariantKey currentKey = m_compiled;
  const bool       hasLive    = m_shaderModule != VK_NULL_HANDLE || m_rtxPipeline != VK_NULL_HANDLE
                       || m_rqPipeline != VK_NULL_HANDLE || m_sbtBuffer.buffer != VK_NULL_HANDLE;
  if(hasLive && m_compiledIsFallback)
  {
    release(m_shaderModule, m_rtxPipeline, m_rqPipeline, m_sbtBuffer);
  }
  else if(hasLive)
  {
    // If the current variant already exists in the cache (possible after a switch that
    // happened without our knowledge), update it in place rather than duplicating.
    bool foundExisting = false;
    for(auto& e : m_variantCache)
    {
      if(e.key == currentKey)
      {
        // Reclaim any stale handles already cached for this key before overwriting.
        release(e.shaderModule != m_shaderModule ? e.shaderModule : VK_NULL_HANDLE,
                e.rtxPipeline != m_rtxPipeline ? e.rtxPipeline : VK_NULL_HANDLE,
                e.rqPipeline != m_rqPipeline ? e.rqPipeline : VK_NULL_HANDLE,
                e.sbtBuffer.buffer != m_sbtBuffer.buffer ? e.sbtBuffer : nvvk::Buffer{});
        e.shaderModule = m_shaderModule;
        e.rtxPipeline  = m_rtxPipeline;
        e.rqPipeline   = m_rqPipeline;
        e.useSER       = m_pipelineUseSER;
        e.sbtBuffer    = m_sbtBuffer;
        e.sbtRegions   = m_sbtRegions;
        foundExisting  = true;
        break;
      }
    }
    if(!foundExisting)
    {
      // Front is most-recently used, so the LRU trim below evicts from the back.
      m_variantCache.insert(m_variantCache.begin(), VariantCacheEntry{currentKey, m_shaderModule, m_rtxPipeline, m_rqPipeline,
                                                                      m_pipelineUseSER, m_sbtBuffer, m_sbtRegions});
    }
  }

  // Detach handles from the live slots; if we find a hit we'll fill them from cache,
  // otherwise the caller will rebuild fresh.
  m_compiledIsFallback = false;
  m_shaderModule       = VK_NULL_HANDLE;
  m_rtxPipeline  = VK_NULL_HANDLE;
  m_rqPipeline   = VK_NULL_HANDLE;
  m_sbtBuffer    = {};
  m_sbtRegions   = {};

  // 2) Look up newKey. On hit, lift its handles out of the cache slot (so we don't
  // hold duplicate ownership) and remove the slot from the cache.
  for(auto it = m_variantCache.begin(); it != m_variantCache.end(); ++it)
  {
    if(it->key == newKey)
    {
      m_shaderModule = it->shaderModule;  // SER does not affect the SPIR-V, so this is always reusable
      if(it->useSER == m_pipelineUseSER)
      {
        m_rtxPipeline = it->rtxPipeline;
        m_rqPipeline  = it->rqPipeline;
        m_sbtBuffer   = it->sbtBuffer;
        m_sbtRegions  = it->sbtRegions;
        ++m_pipelineGeneration;
      }
      else
      {
        // Cached with the other SER setting: discard the pipelines (and the SBT built from the RTX
        // one) and let ensureShadersAndPipelines() rebuild them at the current setting.
        release(VK_NULL_HANDLE, it->rtxPipeline, it->rqPipeline, it->sbtBuffer);
        m_rtxPipeline = VK_NULL_HANDLE;
        m_rqPipeline  = VK_NULL_HANDLE;
        m_sbtBuffer   = {};
        m_sbtRegions  = {};
      }
      m_compiled = newKey;
      m_variantCache.erase(it);
      return true;
    }
  }

  // 3) Enforce LRU bound: drop oldest entries if over capacity.
  while(m_variantCache.size() > kVariantCacheMaxEntries)
  {
    const VariantCacheEntry& victim = m_variantCache.back();
    release(victim.shaderModule, victim.rtxPipeline, victim.rqPipeline, victim.sbtBuffer);
    m_variantCache.pop_back();
  }

  return false;  // miss: caller must (re)compile + (re)build pipelines + (re)build SBT
}

void PathTracer::releaseVariant(Resources& resources, VkShaderModule module, VkPipeline rtx, VkPipeline rq, nvvk::Buffer sbt, bool gpuIdle)
{
  if(module == VK_NULL_HANDLE && rtx == VK_NULL_HANDLE && rq == VK_NULL_HANDLE && sbt.buffer == VK_NULL_HANDLE)
    return;
  auto destroy = [device = m_device, alloc = &resources.allocator, module, rtx, rq, sbt]() mutable {
    vkDestroyShaderModule(device, module, nullptr);
    vkDestroyPipeline(device, rtx, nullptr);
    vkDestroyPipeline(device, rq, nullptr);
    if(sbt.buffer != VK_NULL_HANDLE)
      alloc->destroyBuffer(sbt);
  };
  if(gpuIdle || resources.app == nullptr)
    destroy();
  else
    resources.app->submitResourceFree(std::move(destroy));
}

//--------------------------------------------------------------------------------------------------
// The shader macros of a variant: what compileShader() compiles
// it with, and a variant job with it, so both find the same SPIR-V in the disk cache.
std::vector<std::pair<std::string, std::string>> PathTracer::variantMacros(const VariantKey& key) const
{
  std::vector<std::pair<std::string, std::string>> macros = {
      {"AVAILABLE_SER", std::to_string(m_supportSER)},
      {"WIREFRAME", std::to_string(int(key.wireframe))},
      // Debug-only visualization (incl. the OMM debug payload field). Compiled out of the normal
      // render path so it adds no payload/register cost; a mode switch triggers a recompile.
      {"USE_VISUALIZE", std::to_string(key.visualize ? 1 : 0)},
      // Ray queries against a BLAS with an opacity micromap need an explicit opt-in (see gltf_pathtrace.slang).
      {"USE_OMM", std::to_string(key.omm ? 1 : 0)},
  };
  nvvkgltf::SceneFeatureSet guide;
  guide.set(nvvkgltf::SceneFeatureSet::eDlssGuide, key.dlssGuide);
  nvvkgltf::appendPathTracerDlssShaderMacro(macros, guide, key.dlss);
  // Scene-aware optimal mode: every GLTF_USE_* set to 0 or 1 from the feature set. Never
  // MAT_EXT_X=0 (that would change GltfShadeMaterial layout while the host uploads the all-on struct).
  if(key.optimal)
    nvvkgltf::appendPathTracerOptimalMacros(macros, key.features);
  return macros;
}

//--------------------------------------------------------------------------------------------------
// The SBT of a ray tracing pipeline built from `module`
void PathTracer::createSbt(Resources& resources, VkPipeline pipeline, VkShaderModule module, nvvk::Buffer& sbtBuffer, nvvk::SBTGenerator::Regions& regions)
{
  RtxStages                               table(module);
  const VkRayTracingPipelineCreateInfoKHR info{
      .sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount = static_cast<uint32_t>(table.stages.size()),
      .pStages    = table.stages.data(),
      .groupCount = static_cast<uint32_t>(table.groups.size()),
      .pGroups    = table.groups.data(),
  };
  nvvk::SBTGenerator sbtGenerator;
  sbtGenerator.init(m_device, m_rtPipelineProperties);
  const size_t bufferSize = sbtGenerator.calculateSBTBufferSize(pipeline, info);
  NVVK_CHECK(resources.allocator.createBuffer(
      sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, sbtGenerator.getBufferAlignment()));
  NVVK_DBG_NAME(sbtBuffer.buffer);
  NVVK_CHECK(sbtGenerator.populateSBTBuffer(sbtBuffer.address, bufferSize, sbtBuffer.mapping));
  regions = sbtGenerator.getSBTRegions();
  sbtGenerator.deinit();
}

//--------------------------------------------------------------------------------------------------
// D15 (daz-viewer docs/retained-scene-plan.md, "Shader variants under incremental edits"): on the
// retained scene, a variant that only adds scene features to the optimal build is built off the
// frame. The live variant draws meanwhile, or a cached one with the same shader otherwise and
// features between the live one's and the wanted ones: the residency holds the edits that need
// more, so nothing drawn does. Anything else (a setting, the variant a scene replace wants, a
// first compile, a job that failed) is compiled before the frame, as before.
bool PathTracer::buildsOffTheFrame(Resources& resources, const PendingCompileWork& work)
{
  const VariantKey wanted = makeVariantKey(resources);
  if(!work.recompile || work.stalePipeline || !wanted.optimal || !resources.gpuView().variantMayBuildOffFrame()
     || (m_failedVariant && *m_failedVariant == wanted))
    return false;
  const bool rayQuery = m_renderTechnique == RenderTechnique::RayQuery;
  VariantKey live;
  bool       liveDraws = false;
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    live      = m_compiled;
    liveDraws = !m_compiledIsFallback && (rayQuery ? m_rqPipeline : m_rtxPipeline) != VK_NULL_HANDLE;
  }
  if(!live.optimal)
    return false;
  const auto standsIn = [&](const VariantKey& k) {
    VariantKey same = k;
    same.features   = wanted.features;
    return same == wanted && wanted.features.isSupersetOf(k.features) && k.features.isSupersetOf(live.features);
  };
  if(!liveDraws || !standsIn(live))
  {
    std::optional<VariantKey> cached;
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      for(const VariantCacheEntry& e : m_variantCache)
        if(standsIn(e.key) && e.useSER == m_pipelineUseSER && (rayQuery ? e.rqPipeline : e.rtxPipeline) != VK_NULL_HANDLE)
        {
          cached = e.key;
          break;
        }
    }
    if(!cached || !swapVariant(resources, *cached, /*gpuIdle=*/false))
      return false;
  }
  if(!m_variantJobs.job())
    startVariantJob(resources, wanted);
  return true;
}

// The job: a snapshot of what the variant is built from, taken on the application thread
void PathTracer::startVariantJob(Resources& resources, const VariantKey& key)
{
  VariantCompileJob job{
      .key         = key,
      .macros      = variantMacros(key),
      .searchPaths = resources.slangCompiler.searchPaths(),
      .targets     = resources.slangCompiler.targets(),
      .options     = resources.slangCompiler.options(),
      // the technique of the view the variant draws (a sibling is built for the other view)
      .rayQuery    = m_supportRayQuery && (key.dlss ? m_techniqueDlss : m_techniqueStill) == RenderTechnique::RayQuery,
      .useSER      = m_useSER && m_supportSER,
      .layout      = m_pipelineLayout,
      .rtxFlags    = rtxPipelineFlags(resources.settings.opacityMicromapSupported, m_capturePipelineStats),
      .rqFlags     = m_capturePipelineStats ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) : 0u,
  };
  VariantKey live;
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    live = m_compiled;
  }
  LOGI("[PathTracer] Variant building off the frame: %s\n", live.shaderDiff(key).c_str());
  m_variantJobs.start(std::move(job), [this](const VariantCompileJob& j) { return buildVariant(j); });
}

// On the worker: the job's Slang compile (or the disk cache's SPIR-V) on a compiler of its own, then
// its module and pipeline. Nothing of Resources, and nothing destroyed but what it made and failed.
PathTracer::VariantBuilt PathTracer::buildVariant(const VariantCompileJob& job)
{
  const ShaderWarmScope  working;  // /stats says so: behind the view, like the warm-up
  const auto             start = std::chrono::steady_clock::now();
  nvslang::SlangCompiler compiler;  // a global session of its own: the application thread's is not shared
  compiler.targets() = job.targets;
  compiler.options() = job.options;
  compiler.addSearchPaths(job.searchPaths);
  for(const auto& [name, value] : job.macros)
    compiler.addMacro({name.c_str(), value.c_str()});
  const uint64_t        spirvKey = spirvCacheKey(compiler);
  std::vector<uint32_t> spirv;
  if(!loadCachedSpirv(spirvKey, spirv))
  {
    const auto compileStart = std::filesystem::file_time_type::clock::now();
    if(!compiler.compileFile("gltf_pathtrace.slang"))
    {
      LOGW("[PathTracer] Variant job: Slang compile failed\n%s\n", compiler.getLastDiagnosticMessage().c_str());
      return {};
    }
    spirv.assign(compiler.getSpirv(), compiler.getSpirv() + compiler.getSpirvSize() / sizeof(uint32_t));
    storeCachedSpirv(spirvKey, compiler, compileStart);
  }
  VariantBuilt                   built;
  const VkShaderModuleCreateInfo moduleInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size() * sizeof(uint32_t),
      .pCode    = spirv.data(),
  };
  if(vkCreateShaderModule(m_device, &moduleInfo, nullptr, &built.module) != VK_SUCCESS)
    return {};
  rememberModuleCode(built.module, spirv.data(), spirv.size() * sizeof(uint32_t));
  VkResult   result     = VK_ERROR_UNKNOWN;
  const auto buildStart = std::chrono::steady_clock::now();
  built.pipeline = createPathTracePipeline(m_device, m_pipelineCache.getCache(), job.layout, built.module, job.rayQuery,
                                           job.useSER ? 1 : 0, job.key.dlssTransparency, job.rtxFlags, job.rqFlags, result);
  if(built.pipeline)
    notePipelineBuild(moduleCode(built.module), job.rayQuery, job.useSER ? 1 : 0, job.key.dlssTransparency,
                      job.rayQuery ? job.rqFlags : job.rtxFlags,
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count());
  if(!built.pipeline)
  {
    LOGW("[PathTracer] Variant job: pipeline creation failed (%d)\n", int(result));
    vkDestroyShaderModule(m_device, built.module, nullptr);
    return {};
  }
  m_pipelineCache.saveAsync();
  built.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return built;
}

// At frame top: a finished job's variant into the cache, with its SBT, where the frame's
// tryRestoreCachedVariant() swaps it in. With DLSS on, the same features under the other DLSS
// state is built next, for the next camera start or stop.
void PathTracer::installBuiltVariant(Resources& resources)
{
  std::optional<std::pair<VariantCompileJob, VariantBuilt>> done = m_variantJobs.take();
  if(!done)
    return;
  const VariantCompileJob& job   = done->first;
  VariantBuilt&            built = done->second;
  if(!built.module)
  {
    LOGW("[PathTracer] The variant built off the frame failed; it is compiled before the frame instead\n");
    m_failedVariant = job.key;
    return;
  }
  if(job.layout != m_pipelineLayout)  // never used: no frame holds it
  {
    vkDestroyPipeline(m_device, built.pipeline, nullptr);
    vkDestroyShaderModule(m_device, built.module, nullptr);
    return;
  }
  VariantCacheEntry entry{.key = job.key, .shaderModule = built.module, .useSER = job.useSER};
  if(job.rayQuery)
    entry.rqPipeline = built.pipeline;
  else
  {
    entry.rtxPipeline = built.pipeline;
    createSbt(resources, built.pipeline, built.module, entry.sbtBuffer, entry.sbtRegions);
  }
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    for(auto it = m_variantCache.begin(); it != m_variantCache.end(); ++it)
      if(it->key == job.key)
      {
        releaseVariant(resources, it->shaderModule, it->rtxPipeline, it->rqPipeline, it->sbtBuffer, /*gpuIdle=*/false);
        m_variantCache.erase(it);
        break;
      }
    m_variantCache.insert(m_variantCache.begin(), entry);
    while(m_variantCache.size() > kVariantCacheMaxEntries)
    {
      const VariantCacheEntry& victim = m_variantCache.back();
      releaseVariant(resources, victim.shaderModule, victim.rtxPipeline, victim.rqPipeline, victim.sbtBuffer, /*gpuIdle=*/false);
      m_variantCache.pop_back();
    }
  }
  LOGI("[PathTracer] Variant built off the frame in %.0f ms: [%s]\n", built.ms, job.key.features.toString().c_str());

  if(!isDlssEnabled())
    return;
  VariantKey sibling = job.key;
  sibling.dlss       = !job.key.dlss;
  bool known         = false;
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    known = m_compiled == sibling
            || std::any_of(m_variantCache.begin(), m_variantCache.end(),
                           [&](const VariantCacheEntry& e) { return e.key == sibling; });
  }
  if(!known)
    startVariantJob(resources, sibling);
}

// A job's work thrown away (teardown, a shader reload, a new pipeline layout): waited for, and what
// it built destroyed, which no frame has used
void PathTracer::discardVariantJob()
{
  m_variantJobs.join();
  if(std::optional<std::pair<VariantCompileJob, VariantBuilt>> done = m_variantJobs.take())
  {
    vkDestroyPipeline(m_device, done->second.pipeline, nullptr);
    vkDestroyShaderModule(m_device, done->second.module, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Destroy every cached variant (called from onDetach). Live handles are NOT touched here -
// onDetach's explicit vkDestroyPipeline / vkDestroyShaderModule cover the active slots; the active SBT
// is freed in onDetach via resources.allocator.destroyBuffer(m_sbtBuffer).
void PathTracer::destroyVariantCache(Resources& resources)
{
  for(auto& e : m_variantCache)
  {
    if(e.shaderModule != VK_NULL_HANDLE)
      vkDestroyShaderModule(m_device, e.shaderModule, nullptr);
    if(e.rtxPipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(m_device, e.rtxPipeline, nullptr);
    if(e.rqPipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(m_device, e.rqPipeline, nullptr);
    if(e.sbtBuffer.buffer != VK_NULL_HANDLE)
      resources.allocator.destroyBuffer(e.sbtBuffer);
  }
  m_variantCache.clear();
}
