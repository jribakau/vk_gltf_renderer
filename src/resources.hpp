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

// Core resource management header for the Vulkan glTF renderer.
//
// Defines the main resource structures and settings for the Vulkan path tracer. Manages Vulkan
// resources, glTF scene data, environment maps, and rendering settings.

#pragma once
#include <algorithm>
#include <bitset>
#include <cstdint>
#include <memory>
#include <source_location>
#include <string>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "image_prefetch.hpp"
#include "sample_schedule.hpp"
#include "shaders/shaderio.h"  // Shared between host and device

#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/frame_uploader.hpp>
#include "scene_gpu_view.hpp"
#include "scene_residency_vk.hpp"
#include "animation_control.hpp"
#include "gpu_memory_tracker.hpp"
#include "scene_feature_detection.hpp"
#include <nvapp/application.hpp>
#include <nvapp/imgui_texture.hpp>


// Which special view the user wants in the viewport. The DLSS guide-buffer entries that used
// to live here are gone -- DLSS guide selection is now encapsulated inside the Dlss class
// (see Dlss::activeGuideImage() in dlss.hpp). What remains are the app-level overrides of the
// rendered output: the regular renderer output, or the still denoised by DLSS-RR.
enum class DisplayBuffer
{
  eRendered,           // Final rendered image (default)
  eRrStillDenoised,    // DLSS-RR over the still's final image (RrStillDenoiser::getDescriptorImageInfo)
};

enum DirtyFlags
{
  eDirtyTangents,  // When tangents need to be pushed to GPU

  eNumDirtyFlags  // Keep last - Number of dirty flags
};

// Questions about an environment type, asked in several places and answered once. Free functions
// rather than members: they are properties of the enum, and both the renderer and the Environment
// panel need them.

// True when `env` has an analytic background field the renderer can evaluate per ray -- the
// sun-bearing counterpart of the baked lighting image. Mirrors hasAuthoredSkyBackground() in
// shaders/sky_background.h.slang, which is the shader-side authority.
inline bool hasAuthoredSkyBackground(shaderio::EnvSystem env)
{
  return env == shaderio::EnvSystem::ePlain || env == shaderio::EnvSystem::eGradient;
}

// True when `env` is an analytic sky EnvBaker produces a lat-long image for, rather than one that
// arrives from a file.
inline bool isBakedEnvironment(shaderio::EnvSystem env)
{
  return env == shaderio::EnvSystem::ePlain || env == shaderio::EnvSystem::eGradient || env == shaderio::EnvSystem::eSky;
}

// The SkyType the bake dispatches for an environment, or -1 when `env` is not a sky the bake
// produces at all (HDR, or none).
//
// Two of them because the questions differ: authoredSkyType is "which authored OMI type is this",
// which the per-ray background pass asks; bakeSkyType additionally maps the physical sky onto its
// Bruneton evaluator, which only the bake needs to know.
inline int authoredSkyType(shaderio::EnvSystem env)
{
  switch(env)
  {
    case shaderio::EnvSystem::ePlain:
      return int(shaderio::SkyType::ePlain);
    case shaderio::EnvSystem::eGradient:
      return int(shaderio::SkyType::eGradient);
    default:
      return -1;  // Not an authored sky: the HDR / physical / none background paths apply.
  }
}

inline int bakeSkyType(shaderio::EnvSystem env)
{
  if(env == shaderio::EnvSystem::eSky)
    return int(shaderio::SkyType::ePhysicalBruneton);
  return authoredSkyType(env);
}

// True when `env` is a sky that has a sun at all -- the two analytic types that draw one. Plain has
// none, and a panorama's sun is already in its image.
inline bool skyHasSun(shaderio::EnvSystem env)
{
  return env == shaderio::EnvSystem::eSky || env == shaderio::EnvSystem::eGradient;
}

struct Settings
{
  shaderio::Visualization visualization          = shaderio::Visualization::eRendered;  // Visualization mode
  bool                    wireframe              = false;                      // Wireframe overlay on rendered meshes
  bool                    pixelTime              = false;                      // The Time view (GltfRenderer::setPixelTime)
  int                     pixelTimeCount         = 0;  // `pixelTimeCount` (debug): what it sums (SceneFrameInfo::pixelTimeCount)
  shaderio::EnvSystem     envSystem              = shaderio::EnvSystem::eSky;  // Environment system: Sky or HDR
  bool                    showAxis               = true;                       // Show the axis (bottom left)
  bool                    showGrid               = false;                      // Show infinite grid
  bool                    showGizmo              = false;                      // Show transform gizmo on selected node
  bool                    snapEnabled            = false;  // Snap gizmo transforms to grid increments
  float                   gridUnit               = 1.0f;   // Grid base unit (world units)
  float                   snapRotation           = 45.0f;  // Rotation snap increment (degrees)
  float                   snapScale              = 0.1f;   // Scale snap increment
  float                   snapMove               = 0.1f;   // Move snap increment (world units), apart from the grid's unit
  bool                    gizmoLocal             = false;  // The gizmo along the object's own axes
  float hdrEnvIntensity         = 1.0f;   // Intensity of the environment (HDR)
  // The environment's orientation: a unit quaternion taking its own frame to world, stored x, y, z, w
  // -- OMI_environment_sky's `rotation`, verbatim, and the only orientation any environment has. It
  // turns the HDR image, every analytic sky, and the compass Time of Day places the sun with.
  // Stored as given (the command line may not normalise it); read through envRotationQuat().
  glm::vec4 envRotation          = {0.0f, 0.0f, 0.0f, 1.0f};
  float     hdrBlur              = 0.0f;                      // Blur of the environment (HDR)
  glm::vec3 silhouetteColor      = {0.933f, 0.580f, 0.180f};  // Color of the silhouette
  bool      useSolidBackground   = false;                     // Use solid background color
  glm::vec3 solidBackgroundColor = {0.0f, 0.0f, 0.0f};        // Solid background color
  int       maxSamples           = {2000};  // Path tracer: stop accumulating at this many samples per pixel
  // Sun position as the UI and the command line express it. Resources::sunDirection stays the
  // single source of truth; these are kept in sync with it both ways (see syncSunAngles).
  float sunAzimuth   = 180.0f;  // degrees, compass bearing: 0 = north, 90 = east (SkySun::setAngles)
  float sunElevation = 45.0f;   // degrees

  // Time of Day: where and when, from which the two angles above are computed (src/sun_position.*).
  //
  // Convenience state, and ini-only on purpose. OMI_environment_sky has no field for a place or a
  // date -- it describes a medium, not a moment -- so writing them would fork the schema for
  // something the sun's direction already captures exactly. What travels with the scene is the
  // marked sun light's rotation; these reconstruct the widget on the same machine.
  //
  // todDate and todUtcOffset are seeded from the system clock at start-up, so a fresh install
  // opens on today rather than on an arbitrary epoch.
  float       todHour      = 12.0f;   // local clock hours past midnight, at todUtcOffset
  float       todLatitude  = 47.37f;  // Zurich
  float       todLongitude = 8.54f;
  float       todUtcOffset = 1.0f;          // hours east of UTC, as a clock offset (not a time zone)
  std::string todDate      = "2026-01-01";  // yyyy-mm-dd
  // Write-only: picking a city overwrites the three above and nothing reads it back, which is why
  // it does not persist -- restoring it after them would undo whatever was edited since. The panel
  // derives the name it shows from the coordinates instead (sun_position::cityAt).
  std::string todCity;

  // Physical sky (Bruneton). Kept here, in the same place as every other user-settable value,
  // rather than inside SkyBruneton -- that class owns GPU resources, not user intent, and
  // SettingsRegistry needs a stable address.
  //
  // Altitude is in metres because that is what a person types; SkyBruneton works in kilometres.
  float atmoSunAngularRadius = 0.004675f;  // radians; Earth's sun is 0.00935/2
  float atmoObserverAltitude = 300.0f;     // metres above the planet surface
  // Aerial perspective: how much air the scene contains between the eye and a surface. A distance
  // multiplier on glTF's metre, so 1.0 means "this scene is in metres, as the format says" and 2.0
  // means "treat it as twice the size"; 0 switches the term off. A viewer setting rather than scene
  // data -- it describes how big the model is meant to be, which no extension has a field for.
  float     atmoAerialPerspectiveScale = 1.0f;
  glm::vec3 atmoGroundAlbedo           = {0.2f, 0.2f, 0.2f};  // linear; what the ground reflects, as the floor does
  // What the land round the scene reflects, when the scene says (a procedural backdrop's): it
  // takes the ground albedo's place in buildAtmosphereParameters, so the atmosphere's own value
  // stays what a preset or a file made it. Zero: none.
  glm::vec3 atmoLandAlbedo = {0.0f, 0.0f, 0.0f};
  // Where the land round the scene ends, scene units from the world's origin: between the two what
  // the camera hit gives way to the physical sky's own ground (SceneFrameInfo::landFadeFrom). 0: none.
  float landFadeFrom = 0.0f;
  float landFadeTo   = 0.0f;

  // The rest of the atmosphere: what the planet is made of, and how big it is.
  //
  // These describe a *world*, and all of them are scene data: they arrive from the glTF and go
  // back out to it -- the first six through OMI_environment_sky's `physical` type, the rest
  // through the NV_environment_sky_atmosphere block beside it.
  //
  // Persisted all the same, because the scene cannot lose: a load overwrites them outright, and it
  // happens after the .ini is restored. So persistence only decides what you get when *no* scene
  // carries an atmosphere, and there the answer should be the one you last chose.
  //
  // Units are km and km^-1 throughout, matching SkyBruneton. OMI states m^-1; that conversion
  // happens at the file boundary, not here.
  glm::vec3 atmoRayleighScattering = {0.00580234f, 0.0135578f, 0.0331f};
  // Haze. Upstream's Bruneton demo uses 0.003996, which is an aerosol optical depth of about 0.005 --
  // cleaner than the cleanest place on Earth, and it shows: it left the sun outrunning the sky by
  // 2-3x at every elevation. Measured direct-to-diffuse illuminance on a horizontal surface against
  // a real clear day:
  //
  //     elevation      60     45     30     15
  //     upstream     13.4   11.1    8.0    4.0
  //     here          6.0    4.9    3.5    1.7
  //     measured    ~6-7     ~5     ~3    ~1.3
  //
  // 0.06 /km is an optical depth of ~0.08, which is a clear continental day rather than a mountain
  // observatory. It barely changes how much light reaches the scene (total illuminance moves under
  // 1%); what it changes is the *balance*, which is what was wrong.
  glm::vec3 atmoMieScattering       = {0.06f, 0.06f, 0.06f};
  float     atmoMieAnisotropy       = 0.8f;                         // Cornette-Shanks g; forward-scattering at g > 0
  glm::vec3 atmoSolarIrradiance     = {1.474f, 1.8504f, 1.91198f};  // W/m^2 at the top of the atmosphere
  float     atmoRayleighScaleHeight = 8.0f;                         // km; the air thins by 1/e over this height
  float     atmoMieScaleHeight      = 1.2f;                         // km; aerosols hug the ground far more closely
  // Single-scattering albedo of the aerosol: what fraction of what it removes it scatters rather
  // than absorbs. Extinction is derived from it, because that is the ratio people reason about --
  // "how sooty is the haze" -- and because OMI carries no absorption field at all.
  float     atmoMieAlbedo       = 0.9f;
  // How far one sees through the air at the ground, km: the aerosol above scaled to it
  // (sky_visibility.hpp, applied in buildAtmosphereParameters, so the values above stay the
  // atmosphere's own). 0: as the atmosphere has it, which for Earth's is about 49 km.
  float atmoVisibility = 0.0f;
  glm::vec3 atmoOzoneExtinction = {0.000649717f, 0.0018809f, 8.50167e-05f};  // 1/km
  float     atmoOzoneCenter     = 25.0f;                                     // km; peak of the ozone tent
  float     atmoOzoneWidth      = 30.0f;                                     // km; full width of the tent, zero to zero
  float     atmoBottomRadius    = 6360.0f;                                   // km; planet radius
  float     atmoThickness       = 60.0f;                                     // km; atmosphere depth above the surface
  // Applying a preset overwrites every field above. It is an action rather than a mode: nothing
  // reads it back, and the UI works out which preset (if any) the current values match.
  int atmoPreset = 0;  // AtmospherePresetIndex

  // The physical sky's clouds (SkyClouds): three tiers over one another, and the wind they drift on.
  // Viewer settings for now -- no extension carries them yet -- and inert under any other environment.
  float cloudCoverage  = 0.0f;  // 0 clear .. 1 overcast
  float cloudThickness = 1.45f;  // km: the type's depth, base to the highest tops
  int   cloudSeed      = 0;     // which arrangement of the clouds
  int   cloudType      = 0;     // the low tier's kind: 0 cumulus, 1 stratocumulus, 2 stratus (CLOUD_TYPE_*)
  float cloudMidCoverage  = 0.0f;  // the mid tier, altocumulus: 0 none .. 1 a closed deck
  float cloudHighCoverage = 0.0f;  // the high tier, cirrus: the share of the sky it veils
  float cloudDrift        = 0.0f;   // km/h the wind carries them over the land; 0 they hold still
  float cloudDriftFrom    = 90.0f;  // degrees: where the wind blows from, an azimuth as the sun's

  // Authored sky parameters (glTF OMI_environment_sky). Defaults match the extension's own. When a
  // loaded scene carries the extension it overwrites these; they are the fallback for every scene
  // that does not, and what the user edits in the Environment panel.
  glm::vec3 plainColor           = {0.5f, 0.5f, 0.5f};
  glm::vec3 gradientBottomColor  = {0.2f, 0.169f, 0.133f};
  glm::vec3 gradientHorizonColor = {0.646f, 0.656f, 0.67f};
  glm::vec3 gradientTopColor     = {0.385f, 0.454f, 0.55f};
  glm::vec3 gradientSunColor     = {1.0f, 1.0f, 1.0f};
  float     gradientBottomCurve  = 0.02f;  // Horizon -> bottom falloff
  float     gradientTopCurve     = 0.15f;  // Horizon -> top falloff
  float     gradientSunAngleMax  = 1.74f;  // Angular extent of the sun glow (radians)
  float     gradientSunCurve     = 0.05f;  // Disk -> sky falloff across the glow
  // Write OMI_environment_sky when saving the scene. Off keeps the viewer non-authoring.
  bool      envSaveToGltf           = false;
  bool      useInfinitePlane        = false;                     // Use infinite plane
  bool      isShadowCatcher         = true;                      // Infinite place only catch shadow
  float     infinitePlaneDistance   = 0;                         // Distance/height of the infinite plane
  glm::vec3 infinitePlaneBaseColor  = glm::vec3(0.5, 0.5, 0.5);  // Default gray color
  float     infinitePlaneMetallic   = 0.0;                       // Default non-metallic
  float     infinitePlaneRoughness  = 0.5;                       // Default medium roughness
  float     shadowCatcherDarkness   = 0.0f;                      // Non-physical shadow darkening
  // The ground fades out over a disk of this many scene radii around the scene (0: no fade), as the
  // Figure Viewer's floor does.
  float groundFade = 0.0f;
  // Water on the ground, 0 dry to 1 a sheet of it, and its ripples (SceneFrameInfo::groundWater)
  float groundWater   = 0.0f;
  float groundRipples = 0.25f;
  // A body of water over the ground: its depth in metres (0: none) and its waves (SceneFrameInfo::waterDepth)
  float groundWaterDepth = 0.0f;
  float groundWaves      = 0.4f;
  // A multiplier on the light the sky's sun gives (SceneFrameInfo::skySunScale).
  float skySunScale = 1.0f;
  // Haze in the air, lit by the scene's lights (shaders/haze.h.slang): its visibility in metres, the
  // distance at which it leaves 2% of a dark object's contrast (0: no haze), and how strongly it
  // scatters forward (Henyey-Greenstein g), which is what puts a halo round a light seen through it.
  float hazeVisibility = 0.0f;
  float hazeAnisotropy = 0.7f;
  // A still's scatter point picks its light from this many candidates (SceneFrameInfo::hazeCandidates)
  int hazeCandidates = 4;
  // Uneven haze: the height in metres it thins over above the floor (0: even through the height), how
  // patchy it is (0 to 1) and the patches' size in metres (0: from the scene's size)
  float hazeHeight    = 0.0f;
  float hazePatches   = 0.0f;
  float hazePatchSize = 0.0f;
  bool      dlssRrHardwareAvailable = false;  // DLSS Ray Reconstruction hardware/extensions available (set at startup)
  bool dlssSrHardwareAvailable = false;  // DLSS Super Resolution / DLAA hardware/extensions available (set at startup)
  bool serHardwareAvailable    = false;  // VK_EXT_ray_tracing_invocation_reorder enabled on the device (set at startup)
  bool opacityMicromapSupported = false;  // VK_KHR_opacity_micromap available (set at startup); gates EXT_mesh_opacity_micromap
  bool opacityMicromapMissing = false;  // --useOpacityMicromap is on but the device lacks VK_KHR_opacity_micromap
  bool shaderClockAvailable   = false;  // VK_KHR_shader_clock's device clock enabled (set at startup): the Time view
  DisplayBuffer displayBuffer = DisplayBuffer::eRendered;  // Which buffer to display in viewport

  // `blasScratchBudgetMB`: the retained scene's BLAS builds of one batch share this much scratch
  uint32_t blasScratchBudgetMB = 512;
  // `residencyMinInstances`: the retained scene's least instance capacity (start-up only)
  uint32_t residencyMinInstances = 256;
  // `blasRefitsBeforeRebuild`: a retained BLAS refitted this many times is built again (0: never)
  uint32_t blasRefitsBeforeRebuild = 0;
  // `blasCompactStatic`: a retained static BLAS is compacted once its size is known (on since P7's
  // measure: Khronos props' BLASes 58% smaller, the image the same)
  bool blasCompactStatic = true;
  // `blasCompactDeforming`: a retained deformed BLAS (posed figures, hair) is built to be refitted and
  // compacted, and its compacted copy refitted (on: about a seventh smaller, falcor-ideas.md)
  bool blasCompactDeforming = true;
  // `blasTimeBuilds` (debug): the GPU time of a frame's BLAS builds and refits and its TLAS build, a
  // `GPU BLAS frame` timing line for one frame in a few (GltfRenderer::syncRetainedScene)
  bool blasTimeBuilds = false;
  // `ptEstimator` (debug): what counts the light both a light sample and a BSDF ray reach
  // (SceneFrameInfo::estimator): 0 both (MIS), 1 light samples alone, 2 BSDF rays alone
  int ptEstimator = 0;
  // the air's sun-shadow ray's roulette (SceneFrameInfo::aerialSunShare)
  float ptAerialSunShare = 0.05F;
  // an accumulation's shadow rays' roulette (SceneFrameInfo::shadowRayShare); 0: every one traced
  float ptShadowShare = 0.05F;
  // `ptSkinSlab` (debug): the thin preview's slab through thin skin (SceneFrameInfo::skinSlab)
  bool ptSkinSlab = true;
  // `ptSkinSpread` (debug): the thin preview's spread of skin's light (SceneFrameInfo::skinSpread)
  bool ptSkinSpread = true;
  // `ptSkinView` (debug): the thin preview's skin as the walk returns it by angle (SceneFrameInfo::skinView)
  bool ptSkinView = true;
  // `residencyRetireOnReplace` (debug): a stream host's scene replace syncs the new scene into the
  // backend it has, retiring the old one's slots, instead of starting the backend over
  bool residencyRetireOnReplace = false;
  // `textureGraceMB`: the retained scene's decoded images nothing samples, kept for reuse
  uint32_t textureGraceMB = 256;
  // `sceneBudgetMB`: the GPU memory the scene is fitted to, where less than the device's budget; 0: that budget
  uint32_t sceneBudgetMB = 0;
  // `memoryFillMB`: device memory held and touched every frame, standing in for a scene that large in a
  // measure of the device's limit (scripts/memory-limit-check.py); 0: none
  uint32_t memoryFillMB = 0;
  // `memoryFillWriteMB`: how much of it is written whole each frame, in turn
  uint32_t memoryFillWriteMB = 0;

  // Enable scene-based shader optimization. When true, only features used by the scene are
  // enabled, reducing shader size and register usage at the cost of a one-time recompile per scene change.
  bool optimalShader = false;

#ifndef NDEBUG
#endif

  // `envRotation` as a quaternion, normalised. The stored value may not be unit -- the command line
  // and MCP write it verbatim -- and a zero-length one is no rotation at all, so it reads as identity.
  [[nodiscard]] glm::quat envRotationQuat() const
  {
    const glm::quat q(envRotation.w, envRotation.x, envRotation.y, envRotation.z);
    const float     len = glm::length(q);
    return (len > 1e-6f) ? q / len : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
};


struct Resources
{
  enum ImageType
  {
    eImgTonemapped,
    eImgRendered,
    eImgSelection,
    eImgCount,  // Sentinel: number of frame-target color attachments. KEEP LAST.
  };

  nvapp::Application* app{nullptr};

  VkInstance               instance{};
  std::vector<std::string> deviceExtensions;  // Extensions enabled on the device (set at startup)
  nvvk::ResourceAllocator  allocator{};       // Vulkan Memory Allocator
  nvvk::FrameUploader      staging;           // Per-frame + load-path staging copies (records into caller cmd)

  nvvk::SamplerPool      samplerPool{};    // Texture Sampler Pool
  VkCommandPool          commandPool{};    // Command pool for secondary command buffer
  nvslang::SlangCompiler slangCompiler{};  // Slang compiler

  EmptySceneGpuView emptyGpuView;
  const SceneGpuView& gpuView() const
  {
    return retainedGpu ? static_cast<const SceneGpuView&>(*retainedGpu) : emptyGpuView;
  }

  // The retained scene (scene_residency_vk.hpp): the store the host feeds (GltfRenderer::hostScene)
  // or --scenefile imports into, kept for the session, and its residency.
  std::unique_ptr<rs::RetainedScene>   retainedScene;
  // the store is a stream host's own (GltfRenderer::hostScene)
  bool                                 hostFed = false;
  bool                                 sceneFromFile = false;  // --scenefile: no host, so the renderer keeps the firefly clamp
  std::unique_ptr<rs::RetainedSceneVk> retainedGpu;
  // Between a cleanup that keeps images for reuse and the next scene replace: the retained scene,
  // emptied, its decoded images in grace. The next replace takes it; a cleanup that keeps
  // nothing destroys it. Not a session's scene (gpuView() never reads it).
  std::unique_ptr<rs::RetainedSceneVk> retainedSpare;

  nvvkgltf::ImagePrefetch imagePrefetch;  // what a host decoded ahead for the retained scene's upload

  // Resources
  nvvk::HdrIbl hdrIbl;  // HDR environment map

  // The physical sky's scattering LUTs, as the shading passes see them.
  //
  // Both renderers build their pipeline layouts from Resources and neither should have to know
  // which class owns the tables, so SkyBruneton publishes its runtime set here once at start-up --
  // the same shape `hdrIbl` above already has, minus the ownership. Valid for the whole session:
  // the images are allocated at init and only their *contents* change when the atmosphere does.
  //
  // Read by aerial perspective (shaders/aerial_perspective.h.slang). The bake reaches the same set
  // directly, because EnvBaker is handed it at init.
  VkDescriptorSetLayout skyLutDescriptorSetLayout{};
  VkDescriptorSet       skyLutDescriptorSet{};
  // The physical sky's cloud image (SkyClouds), published the same way for the same reason: the path
  // tracer composes it over what the camera sees. Clear, it composes to the sky unchanged.
  VkDescriptorSetLayout skyCloudDescriptorSetLayout{};
  VkDescriptorSet       skyCloudDescriptorSet{};
  // Main frame target (tonemapped + rendered + selection + depth). Accessor cheat sheet:
  //   raster attachment  -> getColorAttachmentView() / getDepthImageView()
  //   compute write      -> getColorStorageImageInfo()
  //   sampled read       -> getColorSampleDescriptorImageInfo(..., linearSampler)
  //   ImGui              -> getUiImageView() + tonemappedUi
  nvvk::RenderTarget gBuffers;
  nvapp::ImTexture   tonemappedUi{};   // Viewport display (eImgTonemapped only)
  VkSampler          linearSampler{};  // Linear sampler (visual helpers, etc.)
  nvvk::Buffer       bFrameInfo;       // Scene/Frame information
  // The sky's sun. Every sky type with a sun reads it, and the light the renderer supplies for
  // that sun aims along it. Driven by sunAzimuth/sunElevation -- themselves driven by the Time of
  // Day widget -- unless the scene marks one of its directional lights as the sun, in which case
  // that light owns the direction and this follows it (see GltfRenderer::syncSunFromMarkedLight).
  // It used to live inside the MDL sky's parameter block, which is why it survived that sky's
  // deletion while nothing else in it did.
  //
  // `sunYIsUp` is not a constant: it tracks the camera's up axis, because a Z-up scene has to
  // interpret the same azimuth/elevation pair differently.
  glm::vec3                                   sunDirection{-1.23413404e-08F, 0.707106829F, 0.707106709F};
  bool                                        sunYIsUp{true};
  nvshaders::Tonemapper                       tonemapper{};                       // Tonemapper
  shaderio::TonemapperData                    tonemapperData{.autoExposure = 1};  // Tonemapper data
  std::shared_ptr<nvutils::CameraManipulator> cameraManip;         // Camera manipulator (owned by GltfRenderer)
  std::filesystem::path                       headlessOutputPath;  // --output: override for headless image save path
  int accumulatedSamples{0};     // Path tracer: samples per pixel in eImgRendered as of the last recorded frame
  std::vector<int> sampleStops;  // Path tracer: sorted counts a frame must end on (--outputSamples)

  // Pipeline
  std::array<nvvk::DescriptorBindings, 2> descriptorBinding{};    // Descriptor bindings: 0: textures, 1: tlas
  std::array<VkDescriptorSetLayout, 2>    descriptorSetLayout{};  // Descriptor set layout
  VkDescriptorSet                         descriptorSet{};        // Descriptor set for the textures
  VkDescriptorPool                        descriptorPool{};

  // Animation playback state
  AnimationControl animationControl{};

  int frameCount{0};
  // Restarts of the accumulation for anything but the camera (GltfRenderer::resetFrame): an edit, a light, a
  // setting. The moving view's bounce cache lets its light go faster after one (RadianceCache).
  uint64_t sceneChanges{0};

  // The adaptive stop found all but a few of this accumulation's pixels done (the path tracer sets it,
  // and clears it as an accumulation starts)
  bool adaptiveDone{false};

  // The accumulation is complete: rendering stops until the next reset (resetFrame). The path tracer
  // stops when its image holds settings.maxSamples samples per pixel, or once its adaptive stop is
  // done. The frame bound stops a path tracer that adds no samples (nothing to trace); a frame that
  // does render adds at least one sample, so it never cuts an accumulation short.
  bool accumulationComplete() const
  {
    return sample_schedule::accumulationComplete(frameCount, accumulatedSamples, settings.maxSamples, adaptiveDone);
  }


  // Frames that wrote SceneFrameInfo and called the active renderer's onRender(), so the previous-frame
  // matrices moved. Frames skipped once maxFrames is reached do not count.
  // Dlss compares it across evaluate() calls to find frames it missed.
  uint64_t drawnFrameCount{0};

  // #DLSS: True if any node transforms changed this frame (for DLSS per-instance motion).
  bool dlssInstanceMotionActive{false};

  // #DLSS: True if vertex positions may have been rewritten this frame (skin, morph, set_vertices)
  // and prevPositions holds the ones from before (for DLSS per-vertex motion).
  bool dlssDeformMotionActive{false};

  // Selection: set of render node indices (TLAS order). One primitive = set of size 1; node + branch = many.
  std::unordered_set<int> selectedRenderNodes;

  // Selection bitmask for silhouette: one bit per render node (GPU buffer + CPU mirror)
  std::vector<uint32_t> selectionBitMask;
  nvvk::Buffer          bSelectionBitMask;
  bool                  selectionDirty = true;

  // The hover (GltfRenderer::setHover): whether it is on, so the path tracer keeps the object IDs for
  // it (SelectionWrite::read), the render nodes of the item under the pointer, and their bitmask
  bool                    hoverOn = false;
  std::unordered_set<int> hoveredRenderNodes;
  std::vector<uint32_t>   hoverBitMask;
  nvvk::Buffer            bHoverBitMask;

  // A silhouette's bitmask from its render nodes, one bit each. Cleared and refilled each call.
  static void fillBitMask(std::vector<uint32_t>& mask, const std::unordered_set<int>& renderNodes, int numRenderNodes)
  {
    const size_t numWords = numRenderNodes > 0 ? (static_cast<size_t>(numRenderNodes) + 31u) / 32u : 1u;
    mask.resize(numWords, 0u);
    std::fill(mask.begin(), mask.end(), 0u);
    for(int rnIdx : renderNodes)
    {
      if(rnIdx >= 0 && static_cast<size_t>(rnIdx) < numRenderNodes)
      {
        const size_t word = static_cast<size_t>(rnIdx) / 32u;
        const size_t bit  = static_cast<size_t>(rnIdx) % 32u;
        mask[word] |= (1u << bit);
      }
    }
  }
  // Build CPU-side selection bitmask from selectedRenderNodes.
  void updateSelectionBitMask(int numRenderNodes) { fillBitMask(selectionBitMask, selectedRenderNodes, numRenderNodes); }

  nvvkgltf::GpuMemoryTracker appMemoryTracker;  // Application-level GPU memory tracking (frame targets, denoisers, etc.)

  Settings settings;

  std::bitset<32> dirtyFlags;

  // Scene feature set (KHR_materials_* and denoiser guide buffer);
  nvvkgltf::SceneFeatureSet currentFeatureSet{};

};
