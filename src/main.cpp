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

// #define USE_NSIGHT_AFTERMATH

// #define USE_DBG_PRINTF


#include <fmt/format.h>
#include <stb/stb_image.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <unordered_set>
#include <string>
#undef APIENTRY

#include <nvaftermath/aftermath.hpp>
#include <nvapp/application.hpp>
#include <nvapp/elem_dbgprintf.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/parameter_sequencer.hpp>
#include <nvvk/context.hpp>
#include <nvvk/validation_settings.hpp>

#include "renderer.hpp"
#include "mcp_timing.hpp"
#include "frame_stream.hpp"
#include "frame_present.hpp"
#include "fv_host.hpp"
#include <figure/dson/store.hpp>
#include <figure/wardrobe/bind_cache.hpp>
#include "fv_protocol.hpp"
#include "mcp_scene.hpp"

#include <algorithm>
#include "version.hpp"

nvutils::ProfilerManager g_profilerManager;  // #PROFILER

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvapp::ApplicationCreateInfo appInfo;
  nvvk::ContextInitInfo        vkSetup;
  nvutils::Logger&             logger   = nvutils::Logger::getInstance();
  nvutils::Logger::LogLevel    logLevel = nvutils::Logger::LogLevel::eINFO;
  nvutils::Logger::ShowFlags   logShow  = nvutils::Logger::ShowBits::eSHOW_TIME;
#if NDEBUG
  logger.breakOnError(false);
#endif

  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              cli(nvutils::getExecutablePath().stem().string(), {".txt"});

  // Global variables
  std::filesystem::path sceneFilename{};             // "shader_ball.gltf"};  // Default scene
  std::filesystem::path saveSceneFilename{};         // --savefile: writes on change, never persisted
  std::filesystem::path hdrFilename{"std_env.hdr"};  // Default HDR
  std::filesystem::path cacheDir{};                  // --cacheDir: pipeline cache and compiled SPIR-V. Empty: next to the exe
  std::filesystem::path loadPresetFilename;          // --loadSkyPreset, applied on write
  std::filesystem::path savePresetFilename;          // --saveSkyPreset, written on write
  std::filesystem::path iniFilename;                 // --iniFile: settings for an automated run (opt-in)

  // Application defaults overrides
  appInfo.preferredVsyncOffMode = VK_PRESENT_MODE_MAILBOX_KHR;

  // Command line parameters registration.
  //
  // scenefile/hdrfile load on change, so they work identically from the command line, a benchmark
  // sequence, and nvpro_set_parameters. `renderer` stays null until after the start-up parse, so
  // the callbacks no-op there (the renderer has no Vulkan device yet, and a load attempt would
  // only report the start-up file as broken); the explicit load further down handles that case.
  GltfRenderer* renderer = nullptr;  // assigned after the start-up parse, below
  parameterRegistry.add({.name = "scenefile",
                         .help = "Input scene filename (loads immediately when set at runtime)",
                         .callbackSuccess =
                             [&renderer, &sceneFilename](const nvutils::ParameterBase* const) {
                               // Warn if scene load fails so caller knows old scene is still active.
                               if(renderer && !renderer->loadSceneFile(sceneFilename))
                               {
                                 LOGW("scenefile '%s' did not load; previous scene (if any) is still active.\n",
                                      nvutils::utf8FromPath(sceneFilename).c_str());
                               }
                             }},
                        {".gltf"}, &sceneFilename);
  // Saving is a parameter for the same reason loading is: one registration covers the command
  // line, benchmark sequences and nvpro_set_parameters at once, and it runs on the application
  // thread where touching the scene is legal. It is deliberately not persisted -- remembering a
  // path that was already written would re-save on the next launch.
  parameterRegistry.add({.name = "savefile",
                         .help = "Write the current scene to this path (saves immediately when set at runtime)",
                         .callbackSuccess =
                             [&renderer, &saveSceneFilename](const nvutils::ParameterBase* const) {
                               if(renderer && !renderer->save(saveSceneFilename))
                               {
                                 LOGW("savefile '%s' was not written.\n", nvutils::utf8FromPath(saveSceneFilename).c_str());
                               }
                             }},
                        {".gltf", ".glb"}, &saveSceneFilename);
  parameterRegistry.add({.name = "hdrfile",
                         .help = "Input HDR filename (loads immediately when set at runtime)",
                         .callbackSuccess =
                             [&renderer, &hdrFilename](const nvutils::ParameterBase* const) {
                               // Warn if HDR load fails so caller knows old HDR is still active.
                               if(renderer && !renderer->loadHdrEnvironment(hdrFilename))
                               {
                                 LOGW("hdrfile '%s' did not load; previous HDR (if any) is still active.\n",
                                      nvutils::utf8FromPath(hdrFilename).c_str());
                               }
                             }},
                        {".hdr", ".exr"}, &hdrFilename);
  // A custom parameter, so MCP's nvpro_set_parameters does not offer it (it skips custom
  // parsers): the folder is read once, before the path tracer attaches. A sequencer step that
  // sets it later is rejected rather than silently ignored.
  bool cacheDirApplied = false;
  parameterRegistry.addCustom({"cacheDir",
                               "Directory for the pipeline cache and the compiled path-tracer SPIR-V (default: next to "
                               "the executable). Startup only"},
                              1,
                              [&cacheDir, &cacheDirApplied](const nvutils::ParameterBase* const, std::span<char const* const> args,
                                                            const std::filesystem::path&) {
                                if(cacheDirApplied)
                                {
                                  LOGE("--cacheDir only applies at startup\n");
                                  return false;
                                }
                                cacheDir = nvutils::pathFromUtf8(args[0]);
                                return true;
                              });
  parameterRegistry.add({"iniFile",
                         "Restore settings from this .ini in a headless, scripted or benchmark run (those ignore "
                         "the default .ini, so their result does not depend on the last interactive session)"},
                        {".ini"}, &iniFilename);

  // Sky presets. Actions, like hdrfile above: the value is a path and setting it does the thing,
  // which is what makes them reachable from a benchmark sequence and from MCP without a second
  // registration. Neither persists -- remembering "load this preset" would replay it next launch.
  parameterRegistry.add({.name = "loadSkyPreset",
                         .help = "Apply a .sky.json preset (loads immediately when set at runtime)",
                         .callbackSuccess =
                             [&renderer, &loadPresetFilename](const nvutils::ParameterBase* const) {
                               if(renderer)
                                 renderer->loadSkyPreset(loadPresetFilename);
                             }},
                        {".json"}, &loadPresetFilename);
  parameterRegistry.add({.name = "saveSkyPreset",
                         .help = "Write the current sky to a .sky.json",
                         .callbackSuccess =
                             [&renderer, &savePresetFilename](const nvutils::ParameterBase* const) {
                               if(renderer)
                                 renderer->saveSkyPreset(savePresetFilename);
                             }},
                        {".json"}, &savePresetFilename);
  parameterRegistry.addVector({"size", "Size of the window to be created", "s"}, &appInfo.windowSize);
  // Always headless: the app draws the frames itself. Accepted for what still passes it.
  bool headlessFlag = true;
  parameterRegistry.add({"headless"}, &headlessFlag, true);
  appInfo.headless = true;
  parameterRegistry.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
#ifdef USE_NVMCP
  bool     enableMcp = false;
  uint32_t mcpPort   = 7671;
  parameterRegistry.add({"mcp", "Serve the shader-timing tools over MCP (see src/mcp_timing.cpp)"}, &enableMcp, true);
  parameterRegistry.add({"mcpPort", "Port for the --mcp endpoint (1-65535)"}, &mcpPort);
#endif
  parameterRegistry.add({"vvl", "Activate Vulkan Validation Layer"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"logLevel", "Log level: [Stats:1, Info:3, Warning:4, Error:5]"}, reinterpret_cast<int*>(&logLevel));
  parameterRegistry.add({"logShow", "Show extra log info (bitset): [0:None, 1:Time, 2:Level]"}, reinterpret_cast<int*>(&logShow));
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);
  parameterRegistry.add({"vsyncOffMode", "Preferred VSync Off mode: [0:Immediate, 1:Mailbox, 2:FIFO, 3:FIFO Relax]"},
                        reinterpret_cast<int*>(&appInfo.preferredVsyncOffMode));
  bool useOpacityMicromap = true;
  parameterRegistry.add({"useOpacityMicromap", "Use EXT_mesh_opacity_micromap opacity micromaps when supported"}, &useOpacityMicromap);

  // Render server for a host application (Figure Viewer): a headless run that serves its frames
  // over HTTP (see src/frame_stream.hpp) while the host drives it over --mcp.
  bool     enableStream     = false;
  uint32_t streamPort       = 7672;
  int      streamQuality    = 92;
  bool     streamFullChroma = false;
  int      streamFps        = 60;
  // The stream element is created after the start-up parse; quality and chroma can be changed at
  // runtime (nvpro_set_parameters), which forwards them to it. Before it exists the callbacks no-op
  // and the values go into its create info.
  std::shared_ptr<FrameStream> frameStream;
  parameterRegistry.add({"stream", "Headless only: serve the rendered frames over HTTP (MJPEG) for a host application"}, &enableStream, true);
  parameterRegistry.add({"streamPort", "Port for the --stream endpoint (1-65535)"}, &streamPort);
  parameterRegistry.add({.name            = "streamQuality",
                         .help            = "JPEG quality of the --stream frames (1-100); changeable at runtime",
                         .callbackSuccess = [&frameStream, &streamQuality](const nvutils::ParameterBase* const) {
                           if(frameStream)
                             frameStream->setQuality(streamQuality);
                         }},
                        &streamQuality);
  parameterRegistry.add({.name = "streamFullChroma",
                         .help = "Encode every --stream frame with 4:4:4 chroma instead of 4:2:0 (slower; for a still view that refines)",
                         .callbackSuccess = [&frameStream, &streamFullChroma](const nvutils::ParameterBase* const) {
                           if(frameStream)
                             frameStream->setFullChroma(streamFullChroma);
                         }},
                        &streamFullChroma);
  parameterRegistry.add({"streamFps", "Frame-rate cap of a --stream run"}, &streamFps);
  // Figure Viewer: its UI talks to the --stream server directly (fv_protocol.hpp); the token keeps
  // other local processes and pages out.
  std::string fvToken;
  parameterRegistry.add({"fvToken", "Token every --stream request must carry (?t=); empty: none"}, &fvToken);
  // Live editing by a host (src/mcp_scene.cpp): keep every BLAS refittable instead of compacting it.
  bool editable = false;
  parameterRegistry.add({"editable", "Keep geometry refittable for live edits over --mcp (implied by --stream)"}, &editable, true);

  // Don't show the profiler by default
  auto profilerSettings  = std::make_shared<nvapp::ElementProfiler::ViewSettings>();
  profilerSettings->show = false;

  // Create renderer early so it can register its parameters
  auto elemGltfRenderer = std::make_shared<GltfRenderer>(&parameterRegistry, &cli);

  // Adding the parameter registry to the command line parser and parsing arguments
  cli.add(parameterRegistry);
  cli.parse(argc, argv);
  // Only now may the scenefile/hdrfile callbacks act: during the parse above the renderer has no
  // device yet, so a load would fail and report the start-up file as broken. The explicit load
  // further down is what handles the command-line case.
  renderer = elemGltfRenderer.get();


  elemGltfRenderer->setCacheDir(cacheDir);
  cacheDirApplied = true;
  // parsed DSON files (a strand hair's 76 MB geometry) beside the pipeline cache
  fv::dson::setParsedCacheDir((cacheDir.empty() ? nvutils::getExecutablePath().parent_path() : cacheDir) / "dson");
  fv::wardrobe::setBindCacheDir((cacheDir.empty() ? nvutils::getExecutablePath().parent_path() : cacheDir) / "bind");
  elemGltfRenderer->setEditable(editable || enableStream);

  elemGltfRenderer->alignMaxSamplesForHeadless(appInfo.headlessFrameCount);
  elemGltfRenderer->applyHeadlessDefaults();

  // ImGui::LoadIniSettingsFromDisk() silently keeps the defaults when it cannot read the file, so an
  // automated run would report results for a configuration it never restored. Reject it up front.
  if(!iniFilename.empty() && !std::ifstream(iniFilename))
  {
    LOGE("--iniFile '%s' cannot be read\n", nvutils::utf8FromPath(iniFilename).c_str());
    return -1;
  }

  // Using the command line parameters
  logger.setMinimumLogLevel(logLevel);
  logger.setShowFlags(logShow);

  auto elemProfiler   = std::make_shared<nvapp::ElementProfiler>(&g_profilerManager, profilerSettings);

#ifdef USE_DBG_PRINTF
  auto elemDbgPrintf = std::make_shared<nvapp::ElementDbgPrintf>();
#endif

  // Extension feature needed.
  // clang-format off
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR computeDerivativesFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  VkPhysicalDeviceNestedCommandBufferFeaturesEXT nestedCmdFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT reorderFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT};
  VkPhysicalDeviceOpacityMicromapFeaturesKHR ommFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR};
  VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR addressCmdFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR};
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelineExecFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  // clang-format on

  // Fragment shader barycentrics are for the rasterizer alone; a USE_SOFTWARE_VULKAN build (lavapipe,
  // for CI) starts without them and can only path trace.
#ifdef USE_SOFTWARE_VULKAN
  constexpr bool kBarycentricsRequired = false;
#else
  constexpr bool kBarycentricsRequired = true;
#endif

  // Requesting the extensions and features needed
  vkSetup.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  vkSetup.deviceExtensions   = {
      {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
      {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
      {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
      {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
      {VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature},
      {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &computeDerivativesFeature},
      {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
      {VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeatures, kBarycentricsRequired},
      {VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME, &nestedCmdFeature},
      {VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &reorderFeature, false},
      {VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME, &addressCmdFeature, false},  // Optional: required by VK_KHR_opacity_micromap
      {VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME, &ommFeature, false},                // Optional: Opacity Micromap (OMM)
      {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, &pipelineExecFeature, false},  // Optional: path-tracer register counts in the log
      {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, nullptr, false},  // Optional: the VRAM budget a frame size is checked against (GltfRenderer::refuseFrameSize)
  };

  // Only request the graphics queue
  vkSetup.queues = {{VK_QUEUE_GRAPHICS_BIT}};

#ifdef USE_OPTIX_DENOISER
  // Instance extensions
  vkSetup.instanceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  vkSetup.instanceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

  // Device extensions
  vkSetup.deviceExtensions.emplace_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef WIN32
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif  // WIN32
#endif  // USE_OPTIX_DENOISER
#if defined(_WIN32) && !defined(USE_OPTIX_DENOISER)
  // native presentation (frame_present_win32.cpp) shares images and a fence with D3D11; the
  // denoiser's block above asks for the same
  vkSetup.instanceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  vkSetup.instanceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#endif


#if defined(USE_NSIGHT_AFTERMATH)
  // Adding the Aftermath extension to the device and initialize the Aftermath
  auto& aftermath = AftermathCrashTracker::getInstance();
  aftermath.initialize();
  aftermath.addExtensions(vkSetup.deviceExtensions);
  // The callback function is called when a validation error is triggered. This will wait to give time to dump the GPU crash.
  nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });
#endif

  nvvk::ValidationSettings validation{};
#ifdef USE_DBG_PRINTF
  validation.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);
  validation.printf_to_stdout = VK_FALSE;  // Different from default to capture output

#else
  validation.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
#endif

  // Optimize VVL for fast pipeline creation while keeping critical validation
  if(vkSetup.enableValidationLayers)
  {
    // Disable expensive shader validation during pipeline creation
    validation.check_shaders         = VK_FALSE;
    validation.check_shaders_caching = VK_FALSE;
  }

  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();


#if defined(USE_DLSS)
  // Probe BOTH DLSS Ray Reconstruction (RR) and DLSS Super Resolution (SR) extension
  // requirements, union by extension name (RR and SR share most extensions) so we ask for each
  // extension only once when creating the Vulkan instance + device. Track per-feature query
  // success so we can split availability into dlssRrHardwareAvailable / dlssSrHardwareAvailable.
  static std::vector<VkExtensionProperties> extraInstanceExtRr;
  static std::vector<VkExtensionProperties> extraInstanceExtSr;
  NgxContext::getRequiredInstanceExtensions(NVSDK_NGX_Feature_RayReconstruction, {}, extraInstanceExtRr);
  NgxContext::getRequiredInstanceExtensions(NVSDK_NGX_Feature_SuperSampling, {}, extraInstanceExtSr);
  {
    std::unordered_set<std::string> seen;
    auto                            appendOnce = [&seen, &vkSetup](const VkExtensionProperties& ext) {
      if(seen.insert(ext.extensionName).second)
        vkSetup.instanceExtensions.push_back(ext.extensionName);
    };
    for(auto& ext : extraInstanceExtRr)
      appendOnce(ext);
    for(auto& ext : extraInstanceExtSr)
      appendOnce(ext);
  }

  // After selecting the device, we also request extensions DLSS needs using
  // nvvk::Context's callback. NGX may report that one or both features are not
  // available on this GPU; we keep the device alive either way and just track
  // per-feature query success.
  static bool                               dlssRrQueryOk = false;
  static bool                               dlssSrQueryOk = false;
  static std::vector<VkExtensionProperties> extraDeviceExtRr;
  static std::vector<VkExtensionProperties> extraDeviceExtSr;
  vkSetup.postSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& vkSetup) {
    dlssRrQueryOk = (NVSDK_NGX_Result_Success
                     == NgxContext::getRequiredDeviceExtensions(NVSDK_NGX_Feature_RayReconstruction, {}, instance,
                                                                physicalDevice, extraDeviceExtRr));
    dlssSrQueryOk = (NVSDK_NGX_Result_Success
                     == NgxContext::getRequiredDeviceExtensions(NVSDK_NGX_Feature_SuperSampling, {}, instance,
                                                                physicalDevice, extraDeviceExtSr));

    std::unordered_set<std::string> seen;
    auto                            appendOnce = [&seen, &vkSetup](const VkExtensionProperties& ext) {
      if(seen.insert(ext.extensionName).second)
        vkSetup.deviceExtensions.push_back({.extensionName = ext.extensionName, .required = false, .specVersion = ext.specVersion});
    };
    if(dlssRrQueryOk)
      for(auto& ext : extraDeviceExtRr)
        appendOnce(ext);
    if(dlssSrQueryOk)
      for(auto& ext : extraDeviceExtSr)
        appendOnce(ext);

    return true;  // Continue with this device (even if DLSS is not available)
  };
#endif


#if defined(USE_OIDN_DENOISER)
  // #OIDN - Request the device extensions vk_oidn can use (VK_KHR_cooperative_matrix for the tensor
  // cores when supported), after whatever the callback above already asks for.
  {
    const std::optional<std::function<bool(VkInstance, VkPhysicalDevice, nvvk::ContextInitInfo&)>> previousCallback =
        vkSetup.postSelectPhysicalDeviceCallback;
    vkSetup.postSelectPhysicalDeviceCallback = [previousCallback](VkInstance instance, VkPhysicalDevice physicalDevice,
                                                                  nvvk::ContextInitInfo& info) {
      if(previousCallback && !(*previousCallback)(instance, physicalDevice, info))
        return false;
      OidnDenoiser::requestDeviceExtensions(instance, physicalDevice, info);
      return true;
    };
  }
#endif

  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Failed to initialize Vulkan context!");
    return -1;
  }

  // Check per-feature extension enablement after vkContext.init: the NGX query may have
  // succeeded but the loader could still have dropped optional extensions on the floor.
  bool dlssRrHardwareAvailable = false;
  bool dlssSrHardwareAvailable = false;
#if defined(USE_DLSS)
  dlssRrHardwareAvailable = dlssRrQueryOk;
  for(auto& ext : extraDeviceExtRr)
    dlssRrHardwareAvailable &= vkContext.hasExtensionEnabled(ext.extensionName);

  dlssSrHardwareAvailable = dlssSrQueryOk;
  for(auto& ext : extraDeviceExtSr)
    dlssSrHardwareAvailable &= vkContext.hasExtensionEnabled(ext.extensionName);

  if(!dlssRrHardwareAvailable)
    LOGW("DLSS-RR: Required Vulkan extensions not available - Ray Reconstruction will be disabled\n");
  if(!dlssSrHardwareAvailable)
    LOGW("DLSS-SR: Required Vulkan extensions not available - DLSS-SR will be disabled\n");

  elemGltfRenderer->setDlssHardwareAvailability(dlssRrHardwareAvailable, dlssSrHardwareAvailable);
#endif

  // SER is optional. Some drivers fill VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT
  // even without exposing the extension, so the path tracer must not trust that alone.
  elemGltfRenderer->setSerAvailable(vkContext.hasExtensionEnabled(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
                                    && (reorderFeature.rayTracingInvocationReorder == VK_TRUE));

  // Opacity Micromap (OMM) is optional: it may be filtered out if unsupported, and the feature
  // flag confirms the driver actually enabled it. When unavailable, the glTF extension is ignored.
  const bool ommHardware =
      vkContext.hasExtensionEnabled(VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME) && (ommFeature.micromap == VK_TRUE)
      && vkContext.hasExtensionEnabled(VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME)
      && (addressCmdFeature.deviceAddressCommands == VK_TRUE);  // micromaps are created with vkCreateAccelerationStructure2KHR
  // A scene that carries micromaps on hardware without OMM says so when it loads (createVulkanScene).
  elemGltfRenderer->setOpacityMicromapAvailable(ommHardware, useOpacityMicromap);

  // Names of the extensions the device was created with (the vk_oidn denoiser picks its features from them)
  {
    std::vector<std::string> deviceExtensions;
    for(size_t i = 0; i < vkContext.contextInfo.deviceExtensions.size(); i++)
    {
      const char* name = vkContext.contextInfo.deviceExtensions[i].extensionName;
      if(vkContext.hasExtensionEnabled(name))
        deviceExtensions.push_back(std::string(name));
    }
    elemGltfRenderer->setDeviceExtensions(std::move(deviceExtensions));
  }


  // Application information
  appInfo.name = fmt::format("{} {} ({})", nvutils::getExecutablePath().stem().string(), APP_VERSION_STRING, "Slang");
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();
  appInfo.useMenu        = false;
  // Settings come from the defaults and the command line, so the same command line gives the same
  // image; --iniFile opts into a specific file (loaded below, once every settings handler registered).
  appInfo.persistSettings = false;

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  app.addElement(elemGltfRenderer);
  if(enableStream)
  {
    if(streamPort == 0 || streamPort > 65535)
    {
      LOGE("--streamPort %u is not a valid TCP port (1-65535); frames are not served.\n", unsigned(streamPort));
    }
    else
    {
      GltfRenderer* r = elemGltfRenderer.get();
      auto          protocol = std::make_shared<FvProtocol>();
      // native presentation into the app's window (frame_present.hpp): it tells the stream when the
      // app shows frames itself, so the stream it needs is set once that exists
      auto streamRef = std::make_shared<std::weak_ptr<FrameStream>>();
      auto present   = std::make_shared<FramePresent>(FramePresentCreateInfo{
          .image       = [r] { return r->tonemappedImage(); },
          .imageSize   = [r] { return r->renderSize(); },
          .drawn       = [r] { return r->drawnFrameCount(); },
          .refining    = [r] { return r->isRefiningStill(); },
          .takeInput   = [streamRef]() -> int64_t {
            auto fs = streamRef->lock();
            return fs ? fs->takeFrameInput() : 0;
          },
          .localViewer = [streamRef](bool local) {
            if(auto fs = streamRef->lock())
              fs->setLocalViewer(local);
          },
      });
      // after the renderer, so the copy sees the frame it just rendered
      frameStream = std::make_shared<FrameStream>(FrameStreamCreateInfo{
          .port      = uint16_t(streamPort),
          .quality   = std::clamp(streamQuality, 1, 100),
          .maxFps    = std::clamp(streamFps, 1, 240),
          .image     = [r] { return r->tonemappedImage(); },
          .imageSize = [r] { return r->renderSize(); },
          .refining  = [r] { return r->isRefiningStill(); },
          .drawn     = [r] { return r->drawnFrameCount(); },
          .still     = [r] { return r->isStill(); },
          .token     = fvToken,
          .routes    = [protocol, present](httplib::Server& http, const std::atomic<bool>& running) {
            protocol->registerRoutes(http, running);
            present->registerRoutes(http);
          },
          .stopping  = [protocol] { protocol->stop(); },
          .refuseSize = [r, present](VkExtent2D size) { return r->refuseFrameSize(size, present->bytesPerPixel()); },
          .profile    = [r] { return r->frameProfileJson(); },
          .frameGpuMs = [r] { return r->frameGpuMs(); },
      });
      frameStream->setFullChroma(streamFullChroma);
      app.addElement(frameStream);
      *streamRef = frameStream;
      // after the stream: its pacing and GPU wait come first, and it hands the image back in GENERAL
      app.addElement(present);
      // the Figure Viewer's scene, driven over the protocol the stream serves
      app.addElement(std::make_shared<FvHost>(r, protocol, &parameterRegistry));
      r->setStreamHost(true);
    }
  }
#ifdef USE_NVMCP
  if(enableMcp && (mcpPort == 0 || mcpPort > 65535))
  {
    // Narrowing a bad value would silently bind a different port than the client was told to use.
    LOGE("--mcpPort %u is not a valid TCP port (1-65535); the MCP endpoint is disabled.\n", unsigned(mcpPort));
    enableMcp = false;
  }
  if(enableMcp)
  {
    if(auto elemMcp = createTimingMcpServer({.port = uint16_t(mcpPort)}, elemGltfRenderer))
    {
      registerSceneMcpTools(*elemMcp, elemGltfRenderer);
      app.addElement(elemMcp);
    }
  }
#endif
  app.addElement(elemProfiler);
#ifdef USE_DBG_PRINTF
  app.addElement(elemDbgPrintf);
#endif

  // Loading the scene and the HDR
#ifdef USE_DEFAULT_SCENE
  // If USE_DEFAULT_SCENE is enabled and no scene file is specified, load the default scene
  if(sceneFilename.empty())
  {
    sceneFilename = "shader_ball.gltf";
  }
#endif

  // Load a scene if specified, otherwise the application starts empty
  if(!sceneFilename.empty())
  {
    elemGltfRenderer->createScene(sceneFilename);
  }
  // Load an HDR if specified
  if(!hdrFilename.empty())
  {
    elemGltfRenderer->createHDR(hdrFilename);
  }

  // Sky presets. Their parameter callbacks fire while the command line is being read, before this
  // renderer exists, so the start-up values are handed over here -- and applied at the first frame
  // top, not now: the ini restore and the scene's authored sky both land there and would otherwise
  // overwrite the preset, or be missing from the one saved. At runtime the callbacks reach the
  // renderer directly, so a benchmark sequence or an MCP write still applies immediately.
  if(!loadPresetFilename.empty() || !savePresetFilename.empty())
  {
    elemGltfRenderer->queueStartupSkyPresets(loadPresetFilename, savePresetFilename);
  }

  // --iniFile: an automated run restores this file on purpose (see persistSettings above).
  if(!iniFilename.empty())
  {
    LOGI("Restoring settings from '%s'\n", nvutils::utf8FromPath(iniFilename).c_str());
    ImGui::LoadIniSettingsFromDisk(nvutils::utf8FromPath(iniFilename).c_str());
  }

  app.run();

  // A start-up --savefile writes once the run is over, not during the parse: the callback above
  // no-ops while `renderer` is still null, and even afterwards the scene is not fully set up
  // until the first frame has applied deferred state (a glTF-authored sky among it). Saving here
  // captures what was actually rendered. Setting savefile at runtime, from MCP or a benchmark
  // sequence, goes through the callback and saves immediately.
  if(!saveSceneFilename.empty() && elemGltfRenderer)
  {
    if(!elemGltfRenderer->save(saveSceneFilename))
    {
      LOGW("savefile '%s' was not written.\n", nvutils::utf8FromPath(saveSceneFilename).c_str());
    }
  }

  app.deinit();

  // Clear callbacks before scope ends to avoid dangling references
  nvutils::Logger::getInstance().setLogCallback(nullptr);
#if defined(USE_NSIGHT_AFTERMATH)
  nvvk::CheckError::getInstance().setCallbackFunction(nullptr);
#endif

  // Deinit Vulkan context
  vkContext.deinit();

  return 0;
}
