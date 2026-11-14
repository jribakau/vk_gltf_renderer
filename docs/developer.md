# Developer Guide -- Vulkan Ray Tracing glTF Renderer

> **For contributors and agents.** High-level architecture, source map, and the material system — concepts plus pointers into `src/` and `shaders/`, not a restatement of the code. For the scene data-flow deep dive see [Rendering Architecture](RENDERING_ARCHITECTURE.md).

This project is developer-oriented: the **RTX path tracer** is the renderer and the reference for ray tracing glTF PBR materials; DLSS Ray Reconstruction keeps it interactive while the view moves. (A forward rasterizer it once had beside the path tracer was removed on 2026-11-02.)

## Quick Navigation

- [Architecture](#architecture)
- [Scene Graph](#scene-graph)
- [Source Code Structure](#source-code-structure)
- [Material System](#material-system)
- [Common Development Workflows](#common-development-workflows)
- [Testing](#testing)
- [Contributing](#contributing)

For the internal data-flow walkthrough (retained scene -> residency -> GPU/TLAS), including how ray tracing acceleration structures are built, see [Rendering Architecture](RENDERING_ARCHITECTURE.md).


---

## Architecture

### Application Framework

`nvapp::Application` provides the Vulkan application lifecycle. You attach `nvapp::IAppElement` instances, and each is called for different states:

| Callback | Purpose |
|---|---|
| `onAttach()` | Create Vulkan resources, compile shaders, set up descriptor sets |
| `onDetach()` | Clean up all GPU resources |
| `onRender(cmd)` | Record rendering commands (the path tracer, then tone map) |
| `onResize()` | Recreate G-Buffer and notify renderers |
| `onUIRender()` | Frame-top work that may wait on the GPU: pending loads, environment bakes, resizes (nothing draws a UI; the name is `nvapp`'s) |

In `main()` we attach:
- `GltfRenderer` — the main element (scene loading, rendering)
- with `--stream`: the frame stream, the presenter and `FvHost`, which answers the app
- with `--mcp`: the MCP endpoint
- `ElementProfiler` — GPU execution timing
- `ElementDbgPrintf` — shader `printf` debug output (only when built with `USE_DBG_PRINTF`)

### Path-traced depth

The path tracer writes NDC depth (`outDepth`) so the overlays drawn after it (grid, gizmo) can
depth-test against the path-traced image. A frame writes it at the size it traces, from the depth
image's corner: a DLSS frame at DLSS's input size, an accumulation's frame at the image's own. The
depth starts over where the picture does (`sample_schedule::frameStartsDepth`: a first frame, and the
frame a still takes over from DLSS), and a still's later frames only bring it nearer, so a strand one
sample missed still hides the grid. The overlays read it at the size of the frame that last started it
(`PathTracer::depthExtent`), not by whether DLSS is up: DLSS stays up through a still it has handed to
the accumulation. It writes the G-buffer depth image directly as a storage image when the depth format
supports it.
`nvvk::RenderTarget` only grants STORAGE usage in that case, and Vulkan does not require it for any
depth format; `D32_SFLOAT` lacks it on AMD. There `PathTracer` binds an `R32_SFLOAT` staging image
instead and copies it into the depth image through a buffer (`PathTracer::cmdCopyStagedDepth`),
since core Vulkan has no direct copy between color and depth images.

---

## Scene Graph

The scene is a retained store (`rs::RetainedScene`) kept resident by `rs::Residency`, shared by both renderers: instances, prims and materials in slots, in tables the shaders read.

> RN0 == node0 -> node1 -> prim0
> RN1 == node0 -> node1 -> prim1
> RN2 == node0 -> node2 -> prim0
> RN3 == node0 -> node2 -> prim1
> RN4 == node0 -> node3 -> prim1'

---

## Source Code Structure

Primary files grouped by subsystem (not an exhaustive listing — treat `src/` and
`shaders/` as the source of truth for the complete set):

```text
src/
├── main.cpp                    # Entry point, Vulkan context, CLI parsing
├── renderer.cpp/hpp            # Main renderer orchestrator (GltfRenderer): attach and detach, the frame
├── renderer_settings.cpp       # Its settings: every value the command line, the app and MCP can set
├── renderer_environment.cpp    # Its environment: sky, presets, bakes, the HDR panorama, the firefly clamp
├── renderer_scene.cpp          # Its scene: a file or the host's retained scene going in, the sync, the clip
├── host_renderer.hpp          # HostRenderer: what the stream host asks of the renderer; GltfRenderer
│                               #   implements it, tests/common/fake_host_renderer.hpp runs the host without a GPU
├── fv_journal.cpp/hpp          # The edit history's steps (undo, redo, merge); fv_host_journal.cpp captures
│                               #   each edit's undo and redo from the host's state as it applies
├── renderer_selection.cpp      # What the host calls for picks, selection, Fit and the gizmo; selection events into Resources
├── renderer_base.hpp           # BaseRenderer: abstract base of the PathTracer
├── renderer_pathtracer.cpp/hpp # Monte Carlo path tracer (Vulkan ray tracing + ray query)
├── renderer_pathtracer_shaders.cpp # Its shaders and pipelines: compiling, the SPIR-V cache, the variants
├── sample_schedule.hpp         # Samples per frame: ending on stops (cap, --outputSamples), the Auto SPP controller
├── renderer_silhouette.cpp/hpp # Selection and hover outlines (compute shader)
├── env_image_loader.cpp/hpp    # Decodes .hdr / .exr lat-longs to float RGBA for HdrIbl
├── env_texels.hpp              # An EXR's NaN, infinite and negative texels made fit to light by
├── env_baker.cpp/hpp           # Bakes an analytic sky into the lat-long image + importance buffer
│                               #   that HdrIbl consumes (preview vs. commit)
├── gltf_environment_sky.cpp/hpp# OMI_environment_sky parse/write/strip, raw-value passthrough
├── mcp_timing.cpp/hpp          # Optional MCP endpoint (--mcp): recompile shaders, time a
│                               #   GPU pass over a window (docs/mcp.md)
├── camera_controls.cpp/hpp     # What a viewport drag does to the camera (dragMotion), and the manipulator it drives
├── settings_registry.hpp       # One declaration per setting -> CLI + MCP + ImGui.ini
├── animation_control.cpp/hpp   # Animation playback state, scrubbing and clip choice
├── busy_state.hpp              # Busy flag: a background scene job is running
├── resources.hpp               # Shared Vulkan resources and settings (`nvvk::FrameUploader` staging)
│
├── deform_pipelines.cpp/hpp    # The skinning and morph compute pipelines
├── gltf_meshopt.cpp/hpp        # KHR/EXT_meshopt_compression decoded in place
├── image_prefetch.cpp/hpp      # Images a host decodes ahead for the upload to take
├── sampled_image.cpp/hpp       # A sampled VkImage from decoded mip levels
├── gltf_scene_merger.cpp/hpp   # Compose props into one glTF: append a model's resources, rebased
├── gltf_material_cache.cpp/hpp # Packs fv::mat::Material into shader rows; the rows the residency uploads
├── gltf_image_loader.cpp/hpp   # Image decoding (DDS, KTX, STB, WebP); a KTX inflates only the levels under its limit
├── image_levels.hpp            # Which stored mip levels to keep: Max texture size, and lowering images to fit GPU memory
├── ies_profile.cpp/hpp         # EXT_lights_ies: IESNA LM-63 photometric profile parser, and a model's profiles loaded
├── scene_retained.cpp/hpp      # RetainedScene: an owned scene of entities and refcounted mesh, material,
│                               #   texture, skin, light, camera and clip records, with what changed since
│                               #   asked, joint matrices from its world matrices, and the shader features
│                               #   its drawn materials need (counted, not scanned).
│                               #   In progress for Figure Viewer, not yet on the render path (unit-tested)
├── scene_retained_handles.hpp  # Its generational handles and slot map
├── scene_gpu_view.hpp          # SceneGpuView: what the renderers read of the scene on the GPU, and the
│                               #   per-slot side tables
├── scene_residency.cpp/hpp     # rs::Residency: a RetainedScene kept resident through rs::GpuBackend,
│                               #   synced once a frame (slots with holes, capacity tables, BLAS per
│                               #   prim, the TLAS); Vulkan-free, held to tests/common/residency_oracle.*
├── scene_material_bits.hpp     # A material's TLAS flags and mask, raster lists and sRGB textures
├── scene_gpu_tables.cpp/hpp    # Lights as the shaders read them; the emissive triangles' alias table
├── scene_water.cpp/hpp         # FV water's table: per material slot a grid over the UVs its domes reach
├── scene_gpu_backend.hpp       # rs::GpuBackend: all the retained residency layer will call on the GPU,
│                               #   Vulkan-free (tests/common/fake_gpu_backend.* is its GPU-free twin)
├── scene_residency_vk.cpp/hpp  # VulkanGpuBackend (rs::GpuBackend on the allocator, the frame uploader and
│                               #   the scene descriptor set; BLAS builds in scratch batches) and
│                               #   RetainedSceneVk, the retained scene's SceneGpuView, a stream session's
├── scene_residency_vk_util.cpp/hpp # What VulkanGpuBackend converts and plans apart from the device
│                               #   (scratch batches, TLAS rows, glTF samplers); unit-tested
├── scene_import_gltf.cpp/hpp   # A glTF model into a RetainedScene
├── scene_export_gltf.cpp/hpp   # A RetainedScene (or a subtree) out as a fresh glTF model
├── scene_clip_player.cpp/hpp   # ClipPlayer: a clip played into entity TRS and morph weights, AnimationSystem's
│                               #   interpolation ported
├── fv_figure_feed.cpp/hpp      # A DAZ figure's scene graph straight into the retained scene, diffed
│                               #   against what it fed, with narrow syncs for poses, dials and edits
├── fv_prop_feed.cpp/hpp        # The Figure Viewer's props fed into the retained scene as deltas
├── fv_scene_map.cpp/hpp        # Entities and materials back to the host's props, topic nodes and surfaces
├── fv_backdrop.cpp/hpp         # The procedural backdrop's shape, Vulkan-free: the ground's height and mesh, and
│                               #   where each plant of a meadow or a forest stands, from a seed;
│                               #   fv_host_backdrop.cpp plants it from a kit
├── fv_backdrop_state.cpp/hpp   # The backdrop's state over the protocol (backdrop.set, the backdrop topic), and
│                               #   the renderer parameters one that stands sets over the lighting's
├── fv_world.cpp/hpp            # A small planet, Vulkan-free: what a point of it is, by stages (land and sea,
│                               #   mountains, climate, biome, colour); a cube on the sphere as quadtrees, the
│                               #   quads that draw for an eye, and a quad's mesh; fv_host_world.cpp stands it
│                               #   under the backdrop's root, a batch of quads at a time (the `planet` kind)
├── fv_patch.hpp                # What a merge patch's fields are checked with (lighting.set, backdrop.set), and
│                               #   the merge patch between two states that a topic's change is published as
│
├── sky_sun.cpp/hpp             # SkySun: who the sun is (renderer state, or a scene light marked
│                               #   as the sky's) and every path that aims it
├── sky_bruneton.cpp/hpp        # SkyBruneton: the physical sky's scattering LUTs and the passes that build them
├── sky_clouds.cpp/hpp          # SkyClouds: the physical sky's cloud layers -- their noise, the cloud images and
│                               #   the pass that bakes it ahead of the environment bake (docs/sky.md)
├── sky_cloud_view.hpp          # Where the clouds are seen from: the sky image that follows the camera
│                               #   (cloudFollowStep) and the view image where it looks (unit-tested)
├── sky_visibility.hpp          # The air's visibility as a setting: the aerosol scaled to a distance one sees
│                               #   through it (unit-tested)
├── sun_position.cpp/hpp        # NOAA solar position: (place, UTC instant) -> bearing + altitude.
├── sky_ground.cpp/hpp          # The physical sky's ground albedo: the backdrop's land, the water's, or the setting.
│                               #   Pure, self-contained, nothing else links against it
├── sky_world.hpp               # The physical sky over a small planet: the atmosphere's heights and densities
│                               #   scaled to its radius. Pure
├── frame_stream.cpp/hpp        # --stream: the headless run's rendered frames as MJPEG over HTTP, for a
│                               #   host application. Paces the loop to --streamFps, except a still being
│                               #   refined with a client connected, which runs unpaced
├── frame_pacing.hpp            # The pacing decision on its own, free of Vulkan (unit-tested)
├── frame_present.cpp/hpp       # Native presentation: the frames into a host window's DirectComposition
│                               #   surface (POST /present/attach), through images shared with D3D11
├── frame_present_win32.cpp     # Its D3D11 side: the swapchain, the shared textures and fence (Windows)
├── present_hdr.hpp             # An HDR display's SDR white and peak, as the HDR output takes them (unit-tested)
├── hdr_output.cpp/hpp          # The frame an HDR display is shown (scRGB), and EXR exports' linear image
├── white_balance.cpp/hpp       # Auto white balance: the frame's light, and the white balance that makes it grey
├── lens.cpp/hpp                # A cheap wide lens on the tone mapper's input: barrel distortion and colour fringing
├── rain.cpp/hpp                # Rain in the air: drops culled, lit and drawn as streaks over the tone mapper's input
├── rain_cells.hpp              # The rain's cells a frame draws (plain C++, tested)
├── sample_map.cpp/hpp          # Display › Samples: each pixel of a still by the samples it took, in place of the tone mapper
├── invalid_samples.hpp         # What the path tracer threw away as not finite, tallied per accumulation and logged
├── present_pacing.hpp          # Where native presentation waits for the swapchain: before the input or the present (unit-tested)
├── present_policy.hpp          # Who watches a --stream run, and what FrameStream copies for them (unit-tested)
├── present_slots.hpp           # The shared images' hand-off between Vulkan and D3D (unit-tested)
│
├── dlss.cpp/hpp                # DLSS adapter (Ray Reconstruction + Super Resolution)
├── dlss_wrapper.cpp/hpp        # NGX / DLSS SDK wrapper (project-independent)
├── rr_still_denoiser.cpp/hpp   # DLSS-RR as the still's denoiser (its own NGX feature and preset, averaged guides)
├── rr_presets.hpp              # DLSS-RR's presets as dlssPreset and rrStillPreset number them; the still's default
├── still_keep.cpp/hpp          # The still's Keep detail per pixel: converged pixels of the accumulation over RR's image
├── denoiser_schedule.hpp       # The still denoiser's auto-denoise interval, DLSS hand-off weight
├── denoiser_handoff.cpp/hpp    # DLSS hand-off: blend from DLSS-RR's moving image to the denoised still
├── radiance_cache.cpp/hpp      # The moving view's bounce cache: its table on the device and the pass after each frame
│
├── gizmo_transform_vk.cpp/hpp  # 3D translate/rotate/scale gizmo
├── gizmo_grid_vk.cpp/hpp       # Infinite procedural grid
├── gizmo_visuals_vk.cpp/hpp    # Helper overlay manager
│
├── tinygltf_utils.cpp/hpp      # TinyGLTF material + node extension extraction (visibility/selectability/hoverability);
│                               #   tangents, also over plain arrays and remade near vertices that moved
├── tinygltf_converter.cpp/hpp  # OBJ to glTF converter
├── tiny_stb_implementation.cpp # STB library implementation (single compilation unit)
├── pipeline_cache_util.cpp/hpp # Vulkan pipeline cache persistence
├── pipeline_build_record.hpp   # The pipelines built before: tells a full driver cache from a new shader (unit-tested)
├── driver_shader_cache.hpp     # NVIDIA's shader cache of the renderer's own, set before Vulkan starts, emptied when full (unit-tested)
├── scene_feature_detection.cpp/hpp # Detect which extensions a scene uses
├── scene_shader_macros.cpp/hpp # Emit -DGLTF_USE_* macros from detected features
├── timeline_pipeline.cpp/hpp   # Timeline-semaphore submission helper
├── gpu_memory_tracker.hpp      # GPU memory category tracking
└── utils.hpp                   # Miscellaneous utilities

shaders/
├── gltf_pathtrace.slang        # Path tracer (compute + ray generation)
├── silhouette.comp.slang       # Edge detection for the selection's and the hover's outlines
├── skinning.comp.slang         # GPU skinning
├── morph.comp.slang            # GPU morph-target blending
├── animation_io.h.slang        # Skinning/morph shader I/O
├── shaderio.h                  # Host/device shared structures
├── pathtrace_functions.h.slang # Path-tracer hit/shading helpers (getShadowTransmission, etc.)
├── path_roulette.h.slang       # Where the surface roulette starts (still, a DLSS frame's cut-outs) and its chance (unit-tested)
├── sobol.h.slang               # Owen-scrambled Sobol points for the accumulation's sampling decisions (ptSampler)
├── common.h.slang              # Shared shader utilities
├── get_hit.h.slang             # Hit state interpolation
├── water.h.slang               # FV water at a hit: the domes in its material slot's grid cell
├── raytracer_interface.h.slang # RayQuery vs traditional RT abstraction
├── dlss_util.h                 # DLSS motion vector / guide buffer utilities
├── gizmo_grid.slang            # Infinite grid rendering
├── gizmo_grid_shaderio.h.slang # Grid shader I/O
├── gizmo_visuals.slang         # Transform gizmo overlays
├── gizmo_visuals_shaderio.h.slang # Gizmo shader I/O
├── denoiser_handoff.slang      # DLSS hand-off blend of two images
├── radiance_cache.h.slang      # The moving view's bounce cache: cells, keys, a path's sample, a cell's mean (unit-tested as C++)
├── radiance_cache_resolve.slang # Folds a frame's samples into the cells' means and frees stale cells
├── hdr_output.slang            # HDR display output (tone map, overlays) and the EXR export's linear image
├── hdr_output.h.slang          # Its tone curve: Neutral with the display's peak, the others' shoulder opened (unit-tested)
├── hdr_output_io.h.slang       # Its bindings and push constant
├── lens.slang                  # The lens pass: the rendered image bent and its colours split toward the corners
├── rain.slang                  # The rain pass: its light table, the drops culled and drawn
├── rain_shelter.slang          # The rain's shelter map: the first surface a drop meets, traced along the slant
├── rain.h.slang                # The rain's physics (sizes, speeds, cells), C++ too
├── rain_io.h.slang             # Its frame, streak record and bindings
├── lens.h.slang                # Its mapping, picture to rendered image (unit-tested as C++)
├── lens_io.h.slang             # Its bindings and push constant
├── sample_map.slang            # The Samples view: viridis by a still pixel's samples (adaptiveSamplesRamp)
├── sample_map_io.h.slang       # Its bindings and push constant
├── invalid_samples.h.slang     # A frame's block of non-finite samples and guides: counts and the first pixels
├── gltf_light_ies.h.slang      # EXT_lights_ies: photometric-profile falloff lookup
├── sky_background.h.slang      # Background (sun-bearing) field for authored skies
├── sky_world.h.slang           # The physical sky round a planet that stands in the scene: the sky a ray sees
│                               #   from any place, the sun's and the sky's light at a point of the planet
├── world_bodies.h.slang        # The other bodies in that planet's sky as spheres: where a ray meets one, what
│                               #   one leaves of the sun's disk, its markings and shading; C++ too (unit-tested)
├── water_optics.h.slang        # The ground's water as a medium (IOR, absorption, scattering) and its albedo; C++ too
├── sky_omi_shaderio.h.slang    # SkyType + the per-type OMI_environment_sky parameter blocks
├── sky_gradient.h.slang        # OMI `gradient` evaluator (bands + sun), shared by bake and background
├── env_bake.slang              # Bakes any sky into the lat-long image + importance buffer; one
│                               #   switch, with the physical case reading SkyBruneton's LUTs and
│                               #   composing SkyClouds' image over them
├── sky_clouds.h.slang          # The clouds (takram's model): the layers, what is cloud, how it is lit, the image's
│                               #   mapping and composition (unit-tested as C++)
├── sky_clouds_noise.h.slang    # The tiling noise they are cut from, takram's generators (unit-tested as C++)
├── sky_clouds_noise.slang      # Writes that noise into its textures, once
├── sky_clouds_bake.slang       # Marches the layers into the cloud image for the fixed observer (takram's march),
│                               #   and into the images the camera sees them in from where it is
├── sky_clouds_io.h.slang       # Their bindings, parameter block and push constant
│
│   # Local material fork (see "Material System" below)
├── gltf_material_config.h      # MAT_EXT_* compile-time feature flags
├── gltf_eval_config.h          # GLTF_USE_* runtime behavior gates
├── gltf_scene_io.h.slang       # GltfShadeMaterial / GltfLight / GltfScene shaderio structs
├── gltf_vertex_access.h.slang  # Vertex buffer accessors
├── thin_scatter.h.slang        # KHR_materials_scatter's thin-walled lobe split (unit-tested as C++)
└── gltf_material_eval.h.slang  # evaluateMaterial(): GltfShadeMaterial -> PbrMaterial (also compiled as C++: figure/tools/closure_probe)
```

---

## Material System

The retained scene keeps [`fv::mat::Material`](../figure/materials/material.hpp): a DAZ surface lowers into it
(`figure/materials/lower.cpp`), a glTF file's material converts into it, and glTF is what it is written to only when
writing a file (`figure/materials/material_gltf.cpp`). Which lobes are present (optional fields) decide the shader's
features. `MaterialCache` packs the shader's row from it.

The glTF material struct (`shaderio::GltfShadeMaterial`) and its `evaluateMaterial()` function
are **forked locally** into `shaders/gltf_*.h.slang` rather than using the upstream
`nvpro_core2/nvshaders/` versions. The fork exists for two reasons:

1. **Forkability**: the struct layout can be tailored to this project without affecting other
   samples that share `nvpro_core2`.
2. **Extension gating**: every `KHR_materials_*` extension (plus `KHR_texture_transform`) is
   controlled by a compile-time `MAT_EXT_*` flag in
   [`shaders/gltf_material_config.h`](../shaders/gltf_material_config.h).
   When a flag is `0`, that extension's fields drop out of the relevant struct
   (`GltfShadeMaterial` for `KHR_materials_*`, `GltfTextureInfo` for `KHR_texture_transform`)
   and its code paths drop out of `evaluateMaterial()` / `getTexture()` / `getShadowTransmission()`.
   Defaults are all `1`, reproducing the upstream behavior byte-for-byte.
3. **Behavior gating (runtime-safe)**: `GLTF_USE_*` in
   [`shaders/gltf_eval_config.h`](../shaders/gltf_eval_config.h) gates material eval, path-tracer
   state (`PathTracerState`, volume segment, shadow transmission) without changing buffer layout. Optimal mode passes each gate as `-DGLTF_USE_X=0` or `1`
   from `SceneFeatureSet` (never `-DMAT_EXT_X=0` at runtime). `MAT_EXT_*` remains layout authority
   on host and device only.

### Key design points

- **`gltf_scene_io.h.slang`** and **`gltf_vertex_access.h.slang`** intentionally reuse the
  upstream include guards (`DH_SCN_DESC_H`, `VERTEX_ACCESSORS_H`) so that transitive
  includes from `nvpro_core2` headers (e.g. `nvshaders/light_contrib.h.slang`) become
  no-ops. The local version must be included first in every translation unit.
- **`gltf_material_eval.h.slang`** uses its own guard (`GLTF_MATERIAL_EVAL_H`) because no
  upstream code transitively includes it.
- `PbrMaterial` and the BSDF interface (`nvshaders/bsdf_functions.h.slang`) stay in
  `nvpro_core2`. The local `evaluateMaterial()` initializes `PbrMaterial` with
  `defaultPbrMaterial()` first, then overrides per extension — gating an extension off
  safely leaves the corresponding BSDF fields at their neutral defaults.
- Struct members carry their defaults inline, so `GltfShadeMaterial m = {};` works; no
  parallel `default*()` helpers to keep in sync (see `shaders/gltf_scene_io.h.slang`).
- Host-side layout is guarded by `static_assert`s in
  [`src/gltf_material_cache.cpp`](../src/gltf_material_cache.cpp) (8-byte
  alignment/stride plus anchor offsets), so flipping any `MAT_EXT_*` flag that breaks the
  layout fails the build immediately.
- Retroreflection is a worked example of the pattern: `MAT_EXT_RETROREFLECTION` gates
  `KHR_materials_retroreflection` on both host and device, with the evaluation living in
  [`gltf_material_eval.h.slang`](../shaders/gltf_material_eval.h.slang). See the
  [KHR_materials_retroreflection](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_retroreflection/README.md) specification.

### Texture channels: KTX2 swizzle and two-channel sources

Container formats can store fewer channels than the shader expects, in two different ways, and the
renderer resolves both on the host so the shader stays simple.

- **Authored swizzle.** A KTX2 file may carry a `KTXswizzle` key (e.g. a BC5 metallic-roughness map
  that declares `1rg1`). `nv_ktx` parses it into `KTXImage::swizzle`;
  [`gltf_image_loader.cpp`](../src/gltf_image_loader.cpp) converts it to a `VkComponentMapping`, and
  `createSampledImage()` applies it to the `VkImageView` — so the swizzle costs nothing at sample
  time and both render paths see the remapped channels. The same field is used to expand
  single-channel `stb_image` results to grayscale.
- **Missing channels.** A two-channel format (BC5, `R8G8`, EAC_R11G11) has no third channel to
  swizzle from; Vulkan samples blue as a synthetic `0`. That is fine for data that only occupies two
  channels, but it breaks tangent-space normal maps, whose Z is implied rather than stored — decoding
  the synthetic `0` to `-1` flips the normal into the surface and the surface renders black.
  The residency infers the component count from the decoded `VkFormat` and
  marks the **exactly two-channel** ones in `GltfTextureInfo::flags`; `decodeTangentSpaceNormal()` in
  [`gltf_material_eval.h.slang`](../shaders/gltf_material_eval.h.slang) rebuilds Z from the
  unit-length assumption. Three-channel maps keep their authored Z, and single-channel formats are
  excluded — their green is synthetic too, so there is no second axis to reconstruct from.

Because the flag depends on the decoded image format, images are made **before** a material's row:
the material cache bakes the per-texture sampler slot and format flags (`TextureSlotTable`) into `GltfTextureInfo`.

### Worked example: `KHR_materials_scatter` (two modes, one material block)

`KHR_materials_scatter` is the most involved extension in the fork because it changes either the
surface BSDF or the volume transport, depending on the material. Both branches live in the scatter
block of [`gltf_material_eval.h.slang`](../shaders/gltf_material_eval.h.slang); the mode is
selected by `PbrMaterial::thickness` (i.e. by `KHR_materials_volume`), matching the extension spec.

- **Thin-walled** (no volume, or thickness 0). The spec turns the transmissive part of the
  dielectric base into a blend of the specular BTDF and a Lambertian BSDF that splits its energy
  between the two hemispheres by the scatter anisotropy. There is no dedicated scatter lobe in
  `nvpro_core2`'s BSDF, so the block re-expresses that blend with the lobes that already exist. It
  moves `transmission * scatterStrength` out of the specular BTDF into the diffuse bucket, then
  splits that bucket with `diffuseTransmissionFactor`. Note what the factor is: the forward share
  **of the whole diffuse bucket**, not the anisotropy's forward fraction — the bucket also holds
  the untouched diffuse base, so the fraction has to be normalised by the bucket's own weight.
  Conflating the two is what makes the mapping wrong for `transmissionFactor < 1`, and it is only
  by coincidence correct at `transmissionFactor = 1`. Done that way the lobe weights match the
  spec exactly for any transmission, strength and anisotropy. The spec also gives the residual
  specular BTDF and the diffuse reflection lobe *different* colors, and `PbrMaterial::baseColor`
  alone can only carry one — hence `PbrMaterial::transmissionTint` in `nvpro_core2`, a
  multiplier on the BTDF lobe that is `float3(1)` for every other material.
- **Volumetric** (thickness > 0). The surface BSDF is untouched; the block only derives
  `PbrMaterial::scatterCoefficient` (σs) by splitting the `KHR_materials_volume` extinction with
  the Kulla-Conty single-scatter albedo. The transport that consumes it is
  `handleVolumeScatter()` in [`pathtrace_functions.h.slang`](../shaders/pathtrace_functions.h.slang).

Some things about the transport are worth knowing before touching it:

- glTF media are **homogeneous**, so the free-flight distance is sampled analytically per spectral
  channel and combined with the balance heuristic — not with delta tracking against a majorant.
  A single majorant makes the low-extinction channels collide (and therefore random-walk) far too
  often; because those extra collisions are lossless, their effective per-collision albedo drifts
  towards 1 and the medium's perceived color shifts away from `multiscatterColor`.
- `VolumeMedium` stores **absorption**, not extinction, alongside the scattering coefficient. A
  medium authored near white has σa as a tiny difference of two large numbers — recovering it by
  subtraction under limited precision quantizes the albedo straight to 1.
- In-volume Russian roulette has its **own** probability cap (`RR_PCONT_CAP_VOLUME`). The surface
  loop's roulette follows the throughput up to 1, since the bounce limit ends a path that keeps its
  energy. A walk has no such limit (a dense, high-albedo medium keeps the throughput near 1 for
  hundreds of events), so its cap must stay below 1: without roulette such a path never terminates.
- **Bounce limits.** `ptMaxDepth` (0–64) counts the still's surface events. `ptMaxDepthDlss`
  caps DLSS frames below it (`sample_schedule::frameMaxDepth`). Fibres, clear glass and water have
  their own budgets. The app sends 32 and 8. At 8 a closed room of albedo 0.9 keeps 6.1 of its 10
  parts of light; at 32 it keeps 9.7. The roulette ends most paths long before either limit, so
  the limit costs time only where the throughput stays high. The surface roulette starts at the
  third surface event, and on a DLSS frame's cut-out surfaces (leaves, twigs, grass) at the first
  (`path_roulette.h.slang`, `ptCutoutRouletteFromDlss`): unbiased either way, and on a moving frame
  the paths through dark foliage end early instead of making two more bounces each. On every
  surface it darkened a lamp-lit interior's moving shadows by 5 to 12% (Ray Reconstruction). In a real office lit by daylight
  (2026-11-09), a still frame took the same time from 8 to 64 bounces. At 8 it kept all but 0.23%
  of the light.
- Surfaces with a diffuse transmission lobe are lit from **both** hemispheres, so `sampleLights()`
  takes a flag that disables the punctual-light hemisphere rejection in
  `singleLightContribution()`. Without it, a light behind a backlit leaf or sheet contributes
  nothing. For the same reason their TLAS instances disable back-face culling (`materialBits`):
  a diffusely transmitted ray reaches the far wall of a closed object from behind. Rough thin glass
  is lit from both sides too (`roughThinGlass` in `pathTraceOneBounce`): shadow rays stop at it, so
  the emitter inside a frosted bulb is sampled from the envelope, through its transmission.
- Straight-through shadow transmission (`getShadowTransmission()`) passes only a **clear crossing**:
  smooth (`CLEAR_CROSSING_ROUGHNESS`, perceptual, from the material's own roughness and its texture as
  `crossingRoughnessMetallic()` reads them for the shadow ray and the path's vertex alike) specular
  transmission with no thin-walled scatter, weighed by what the surface reflects (Fresnel times
  `specularFactor`, plus the clearcoat). A path's own ray that refracts through such a surface is a
  null event for MIS (`pathTraceOneBounce`): it keeps the previous vertex's sample pdf and light
  pick, and an emitter it reaches converts its pdf by the summed distance (`nullDistance`). The
  light behind glass is then counted once, as that vertex's shadow ray or its BSDF ray. A rough or
  scattering surface blocks shadow rays, and the BSDF ray that reaches a light behind it has weight 1.
- The environment's **compensated table** (`EnvCompensatedCell`) carries cumulative tables of its
  distribution beside the alias method's: each cell's row-conditional CDF in `cdf`, and the row
  marginal in `height` entries after the grid (`compensatedEnvTableSize`). Sobol frames invert them
  (`environmentSampleCompensatedStratified`), which keeps the points stratified; DLSS frames take the
  alias method's O(1) pick. Both give a cell the same pdf.
- The GGX reflection lobes use the **height-correlated** Smith term; their albedo (the layering's
  `ggxSchlickAlbedo`, the metal's multiple-scattering compensation `ggxMultiScatterFactor`, which an
  edge tint scales, `FV_materials_edge_tint` below) comes
  from a table baked from the lobe (`nvshaders/ggx_albedo_table.h.slang`). Change the lobe, rerun
  `utils/ggx_albedo_table.py`: `BsdfLobes.TheGgxAlbedoFitIsTheLobesIntegral` checks them against
  each other.
- A rough boundary's **refraction** lobe (`btdf_ggx_smith_sample`) is single scattering: light its
  refracted ray's masking blocks is lost, where on a real microsurface it meets the surface again and
  mostly goes in. At index 1.4 and roughness 0.5 the lobe let in 7% too little at 70°, 12% at 80°; at
  roughness 0.7, 20% and 35%. Entering a medium from its rarer side, the lobe is now scaled up to Heitz
  et al. 2016's microsurface random walk (every order of scattering): `ggxTransmissionGain`, from two
  tables by index, roughness and view in `nvshaders/ggx_transmission_table.h.slang` (the lobe's share of
  the walk's light, and its transmission with no Fresnel), baked by `utils/ggx_transmission_table.py`
  (about 20 minutes). A specular weight below 1, as DAZ's skin splits its reflection over two lobes,
  lets in all it doesn't reflect. `BsdfLobes.ARoughBoundaryLetsInWhatItsMicrosurfaceDoes` holds the lobe
  to the walk written again in the test, `ALowerSpecularLetsInAllItDoesNotReflect` the sum to the light
  that arrived. Not from the denser side, where total internal reflection is part of the lobe, nor on a
  thin wall or Gaussian slopes (`bsdfRegainsMaskedTransmission`). The reflection's own multiple
  scattering is not given back: 0.7% of the light at 70° there.
- The **sheen** takes from the layers below what it reflects, max(sheenColor) times its lobe's albedo
  (KHR_materials_sheen's albedo scaling), and its lobe is divided by the same, as the specular lobes
  are. The albedo is baked from the lobe too (`nvshaders/sheen_albedo_table.h.slang`, rerun
  `utils/sheen_albedo_table.py`; `BsdfLobes.TheSheenAlbedoTableIsTheLobesIntegral`).
- A **coat** darkens the layers under it by OpenPBR 1.1's (1 − K) / (1 − E K) (`bsdfCoatDarkening`).
  The ground truth is the coat and the base stacked exactly (`coatedReference` in
  `tests/test_bsdf_lobes.cpp`: a position-free walk through an exact rough dielectric, checked against
  the smooth coat's closed form). `BsdfLobes.ACoatDarkensADiffuseBaseAsTheLayersStackedExactly` and
  `ACoatOverASpecularBaseAgainstTheExactStack` hold the closure to it and print the table. E is the
  base's colour with its own dielectric specular over it (`bsdfCoatBaseSpecular`: its weight times its
  Fresnel's mean), and the factor darkens all the base sends up, its specular and a metal's included.
  Over a diffuse base the formula is the physics (within 1.5%). Over a base with its own reflection it
  isn't: under a coat the base sees the bent view, and the closure gives it the unbent one (the app's
  `docs/improvements.md`).
- A lobe is **picked** apart from what it weighs (`bsdfLobePick`): by its weight times the light its
  own value gives back, so a dark base under a specular or a coat stops taking nearly every sample;
  each pick is at least a tenth of its weight. A sample carries weight over pick (`bsdfPickShare`),
  and the mixture pdf both MIS sides read is the picks' (`bsdfEvaluateSumWeighted`).
- A bounce leaves on the **side of the face** its event says (`bsdfFaceAllows`): a reflection on the
  face normal's side, a transmission across it, whatever a normal map does to the shading normal.
  The sampler drops the rest, and the pdf leaves those directions out; a light sample there keeps
  its value at MIS weight 1, so gate light samples on `bsdfEvaluated`, not on the pdf. The ground
  plane and water opt out (`PbrMaterial::faceSided`): their waves are the surface.
- A **bumped diffuse lobe** takes the relief's own shadow (`bumpShadowing`): Smith's G1 of the light's
  angle to the smooth normal, for the slopes the normal and detail maps hold over their whole image
  (measured from the decoded image, `imageSlopeVariance`, carried in the material row). Over a map's
  texels, Lambert's cosine at the bent normal times it takes in what the smooth surface does, so a low
  light no longer brightens a bumped surface, and each texel keeps its own cosine, so the relief still
  shows. Where the slopes are unknown (a map that isn't two-channel) it is the shadow-terminator term.
- The **diffuse exit** (`ePtVolumeDiffuseExit`) is Lambertian times the boundary's Fresnel
  transmittance toward the light, over its cosine-weighted mean (`diffuseExitFresnelWeight`,
  `ePtExitFresnel`): the entry paid (1 − F) on the camera's side, so the exit pays it on the
  light's. It weighs the sample, not the pdf, so both MIS sides keep the cosine lobe.
- The **thin preview's slab** (`skin_slab.h.slang`, `ptSkinSlab`, debug, on): where the preview
  shades measured skin (`ePtScatterAsThin` on DLSS frames, `ePtSimplifyScatter` after a rough
  bounce), a probe goes in within 45° of the smooth normal (`probeSlabFarSide`, `INSTANCE_MASK_WALK`,
  up to eight diffusion lengths). If it reaches the same material's far side, Kubelka-Munk (`skinSlab`)
  splits the map's light by that depth along the normal: R comes back here, T goes through. One
  sample takes one: this point with R, or the far side's point as a Lambertian surface with T, the
  diffuse share of this side's lobes and the exit Fresnel. Each is weighed by one over its chance, so
  it is unbiased for the model. The far side is shaded after the first-hit guides, which keep this
  side. Against a walk through a slab (`SkinSlab.ItLetsThroughWhatTheWalkDoes`) Kubelka-Munk is
  within a few percent on light channels. Lateral light is the rest: on a back-lit ear (faceTest,
  hair off, the Sun low behind, 256 samples) the glow's red went from 0.51 of the walk's to 0.79,
  green 0.76 to 0.87. A straight-in probe gave 0.73 with faceted, hard-edged glow and a dark concha.
  The cost is +6-9% of the preview's trace where skin fills the frame.
- The **thin preview's spread** (`skin_spread.h.slang`, `ptSkinSpread`, debug, on): the preview takes
  skin's light where it lands, the still's walk lets it out millimetres away, so shadows cast on skin
  (a pendant's, the jaw's on the neck) were hard in the moving view. On the camera's hit on preview
  skin, the light sample's diffuse share (`BounceScratch::spreadContribution`) is shadowed per channel
  at a point of its own. One radius of the unit profile and one direction serve all three channels:
  each channel's point lies at that radius over its own rate, so each gets an unweighted sample of its
  own profile. Where no shadow edge is near, every point sees what the hit does and no noise is added.
  A shared radius with per-channel weights instead adds 50-60% colour noise to all lit skin, and as
  a control variate it goes negative near edges, red 5-18% of the time. A probe straight down from
  above each point finds the skin (`probeSpreadPoint`: `INSTANCE_MASK_WALK`, any scattering skin met
  from outside). The point then sends its own shadow ray to the same point of the light. A channel
  with no point, or whose point has the light below its horizon, keeps the hit's shadow, so the
  terminator stays as the preview shades it. The glossy share keeps the hit's shadow.
  - **Profile shape:** Burley's normalized diffusion.
  - **Scale:** fitted to our own walk, s = k / d with k = 2.1772 + 0.3585 / c + 0.6248 c and
    c = sqrt(1 − alpha). It is within 3.3% of the best fit to the walk over maps of 0.02 to 0.95
    (`SkinSpread.ItSpreadsAsTheWalkDoes` walks it in C++).
  - **Not Christensen-Burley's fit:** theirs spreads two to five times wider at our scatter
    distances.
  - **Cost:** three short probes and three shadow rays, on the camera's skin hit only.
- The **thin preview's skin by angle** (`skinViewAlbedo` in `skin_albedo.h.slang`, `ptSkinView`, debug,
  on): the preview shades measured skin as the still's walk returns it. On the camera's side, the
  walk enters by refraction along the view, so its albedo is 1 − H(mu0) sqrt(1 − alpha) at the
  refracted cosine mu0, rising at a slant, darker channels most. H is Hapke's approximation,
  `hFunctionApprox`: about 1% above the exact. With the face-on inversion it is taken as a change
  from head-on, within 0.3% (`SkinViewAlbedo.*`). On the light's side the walk leaves through a
  Lambertian exit that pays the boundary's Fresnel, so the preview's skin takes `exitFresnelEta` too:
  on the light sample's diffuse, and on a continuation the diffuse lobe drew. Over a sky that factor
  averages to 1; under one light it runs from 1.05 head-on to 0.44 at 85°. Both follow the
  preview's softened diffuse (`diffuseSoftening`): each channel's share s about the smooth normal
  takes the view and the light at the smooth normal's cosine, the rest at the map's
  (`softenedSkinViewAlbedo`, `softenedExitFresnelWeight`). Taken at the map's normal alone, every bump
  took its own colour and exit weight again, and the relief the softening evens out came back,
  stronger than the walk's and darker on average (faceTest's cheek close up, 2026-11-12: relief 0.037
  against the walk's 0.031, brightness 0.953; with the split 0.028 and 0.964).
  - **Measured** on a slab under one directional light (scratch `skin_angles.py`), the walk against
    the preview. Before: 4-6% above head-on and up to 17% at a 70° view, by Ft/c̄ in the light's
    angle. After, with skin's specular: within 0-2% to a 45° view.
  - **The remaining gap:** at 70° the preview is 2.5-4% brighter where the boundary is rough
    (roughness 0.5; 0.1 matches). That is the still's: its rough transmission is single-scattering
    (G1 of the refracted ray times 1 − F per microfacet, over 1 − E_R), and the light masking blocks
    is lost, where the preview's layering keeps it.
- A **sized light** (a point or spot light with a radius, a directional one with a disc, the sky's
  own sun aside) is reached by both techniques: `sampleLights()` draws its cone with a real pdf, and
  `sizedLightHits()` tests each path segment against the spheres (and, on escape, the discs),
  weighed against the light pick of the vertex the ray left. `LightPick` keeps that vertex and its
  cull normal, and `sizedLightNeePdf()` redoes its pick for the light that was hit. The radiance is
  the light sample's own (irradiance over the cone, a spot's falloff along the axis), so the two
  estimate the same light. A volume scatter's light sample stays `DIRAC`, and nothing is gathered in
  a medium or on a walk's way out. Camera rays see the lights only with `ePtLightsVisible`.
- **`ptEstimator`** (debug, `SceneFrameInfo::estimator`) says which technique counts a direction
  that both a light sample and a BSDF ray can reach. 0 is both, by the power heuristic, the
  renderer's. 1 is light samples alone, 2 BSDF rays alone (`misLightWeight`, `misBsdfWeight` in
  `shaders/light_pick.h.slang`, at every MIS site). A direction only one technique reaches stays that
  technique's in every mode: a delta lobe or light, the last vertex's light sample, a smooth vertex,
  and the environment cells the compensated table leaves to the BSDF. So each mode is a whole
  estimator, and the three images agree unless a pdf is wrong:
  - a light sampler's own: light alone splits from BSDF alone;
  - a BSDF sampler's own: the same split, the other way;
  - one side's account of the other's pdf (a pick, the tree's climb, a table): only MIS moves.
  `tests/render/estimator_agreement.py` renders a scene with every light kind a BSDF ray can reach,
  in each mode at 8 seeds, with the light tree and with the table by power. At 0 the image is
  bit-identical to before the switch. The test turns the firefly clamp off (`--ptFireflyClamp 0`):
  it cuts one technique's rare bright samples more than the other's. A command line's clamp holds
  when the dome loads; before 2026-11-08 `calibrateFireflyClamp` set it again from the dome.
  `tests/render/emitter_skinned.py` uses the modes on an emitter a skin has stretched, against the
  same surface unskinned: the light table knows the first by its mesh's own positions, so the test
  fails if an emissive triangle's pdf is of any area but the one drawn
  ([RENDERING_ARCHITECTURE.md](RENDERING_ARCHITECTURE.md), the emissive table).
- **A still's shadow rays are traced as often as they can matter** (`ptShadowShare`,
  `SceneFrameInfo::shadowRayShare`). A light sample that would add less than the share of the pixel's
  mean so far, were the light to reach it, has its shadow ray traced with the chance of its size
  against that, and is weighed by one over it (`shaders/contribution_roulette.h.slang`, the roulette
  the air's sun ray takes; where `pathTraceOneBounce` sets `bounce.contribution`). The mean is the
  accumulation's own image, read once the pixel has a few samples (`g_shadowRest` in `processPixel`),
  which the sample being drawn has no part in, so the estimate stays unbiased. A sample let through
  is never brighter than the share of the pixel. DLSS's frames, which have no mean to go by, the
  accumulation's first samples and the Time view trace every ray; at 0 every ray is traced and no
  number is drawn for it, as before the setting. Measured, it saves a few percent of a still's frame on a close-up of dark hair
  and about one elsewhere: the app's `docs/improvements.md` has the numbers.
- **The Time view** (Display › Time, view.set's display `time`, `--pixelTime 1` on a command line)
  times each pixel's thread by the device's clock (`VK_KHR_shader_clock`, `shaderDeviceClock`, an
  optional extension) in a variant of its own, `USE_PIXEL_TIME` (`VariantKey::pixelTime`, never
  embedded). The paths are the shaded image's; `processPixel` reads the clock at its top and before
  the result, and writes the time against the frame's average in viridis
  (`shaders/pixel_time.h.slang`), an eighth to eight times, a display value the tone mapper, bloom and
  local exposure pass by. An accumulation keeps the mean of its frames' times in the convergence
  image, with its adaptive stop off; a DLSS frame shows its own. Each pixel's time and samples are
  also summed by what its first camera ray met (the sky, the ground, the water, an instance), one
  set of atomics for each slot a warp holds, in `SceneFrameInfo::pixelTime`. The host copies the
  sums back per frame in flight (`PathTracer::preparePixelTime`); the average sample from them
  scales the next frames, and the host publishes them as the `time` topic by prop and node
  (`FvView::publishPixelTimes`), or logs them (`[time]`) on a run without one. It is wall time: under
  SER a thread's time includes its waits in the reorders and for its warp, and a cheap pixel that ran
  beside dear ones on the same SM reads dearer, so the sky shows blocks. Compare objects, not tiles.
  The debug `pixelTimeCount` (1 rays, 2 walk steps, 3 shadow rays; `SceneFrameInfo::pixelTimeCount`)
  sums a count in place of the time, through the same image, sums and topic. Rays are counted where
  the ray tracer traces them (`PIXEL_TIME_COUNT` in `raytracer_interface.h.slang`), a walk's steps
  where `pathTraceOneBounce` draws their free flight. Read beside the time, it tells how much a pixel
  traces from how dear each of its rays is. The topic's `perSample` is the frame's average (the
  clock's ticks, nanoseconds on NVIDIA, or the count) and `counting` says which. Key a breakdown by
  `prop` and `node`, never by `name`: two figures' bodies can share one, and summing their lines adds
  their costs (1TEST's skin read 2.8 times the average sample that way; it is 1.45 and 1.34).
- Path regularisation (later lobes at least as rough as the roughest material so far) runs only
  where `ePtRegularize` is set (`ptRegularize`: DLSS frames by default), and never on a surface that
  refracts without scattering.
- The thickness texture, glTF's stand-in for the path length through the medium, is not read: the
  ray measures that length, and `thicknessFactor` alone selects thin-walled vs. volume. It is also what makes a
  surface a medium boundary: the path and its shadow rays count themselves inside only across
  surfaces with a thickness.
- Light reaching the camera through a medium is attenuated wherever it comes from. The medium in
  front of a hit (`processVolumeSegment()`) is resolved before the surface's emission is added, and
  a shadow ray knows the medium it starts in (the path's, from a surface or a scatter point) and
  attenuates every segment it travels inside one, including the last, up to a light inside the
  medium. `tests/render/volume_transmittance.py` checks both against analytic transmittance.
- A path knows it is in a medium only by having crossed into one, so a camera path starts in the
  medium the camera is in: once per accumulation of a scene with a `KHR_materials_volume` material,
  `PathTracer::probeCameraMedium()` runs the path tracer's own pipeline for a single thread
  (`probeCameraMedium()` in [`gltf_pathtrace.slang`](../shaders/gltf_pathtrace.slang)), which
  follows the center camera ray to the nearest medium boundary and, if it meets it from behind,
  records that medium for `pathTrace()` to start in. Without it, a camera inside a fog sees no fog
  in front of it and fog behind the far wall.

A medium is bounded by its mesh, and a path leaves it through a back face. The path keeps the medium
it is in and the one around it (`PathTracerState::volumeMaterial` and `outerMaterial`, the materials
that bound them), and what a surface is to them is decided in one place, `mediumBoundary` in
[`medium_stack.h.slang`](../shaders/medium_stack.h.slang), which `tests/test_medium_stack.cpp`
compiles as C++. The segment up to the hit is charged first in every case.

- A front face met in a medium that scatters, a walk through skin: the mesh had an opening the walk
  escaped through (a figure's eye sockets or mouth) or the walk reached what sits in one (an eyeball,
  a tooth). The medium ends there, before the hit is shaded, instead of darkening everything the path
  meets afterwards, and the path is back in the medium around it, if there is one (a hand in a tub of
  water).
- A front face met in one that only absorbs (glass, a liquid): the surface is inside the medium, and
  the path stays in it, its shadow rays too. Its index of refraction is against the medium's, not
  air's. If it bounds a medium of its own (oil in a bottle that is one solid of glass, ice in a
  drink), a refraction enters that one, and leaving it puts the path back in the first.
- Where two solids overlap (a drink modelled a little into its glass), the path is in the one it
  entered last, and the back of the other, met inside it, is no boundary: the ray goes on from it
  with no event. There are no priorities, as Falcor's `InteriorList` has: DAZ materials have no such
  channel.
- The back of an absorbing solid met from outside every medium (the path started inside it, or its
  mesh is open) is that solid's way out: the path is outside after it. With one inside flag it
  counted as a way in, and the path absorbed all the way to the next surface. A scattering surface's
  back is still a way in: a walk that ended at an eyeball goes on from the inside of the lid.
- Two deep: a third medium met inside two drops the outermost, which is then left into air. The
  medium around is set where the inner surface is met, not where the path crosses it.
  Shadow rays keep one inside flag and one extinction (`getShadowTransmission`): through nested
  media their absorption is approximate.
- The medium around is kept only in the kernel of a scene that draws a clear solid, a medium that
  only absorbs (`SceneFeatureSet::eNestedMedia`, `VariantKey::nestedMedia`, `USE_NESTED_MEDIA`): a
  shader variant of its own, optimal shader or not, compiled before the frame the first time such a
  scene is drawn. In every kernel that one number of path state cost a frame 3% (1TEST,
  2026-11-10); without it the rules above are the one-medium rules, since nothing absorbs.

`tests/render/medium_nested.py` checks a liquid contained in glass, overlapping it and apart from
it against the layers' analytic transmission.

### `FV_materials_detail`: a tiled detail layer

A vendor extension written by [Figure Viewer](https://github.com/jribakau/daz-viewer) for DAZ
PBRSkin's detail layer: fine pores in a small normal map repeated many times across the surface,
which no single normal map at the base map's resolution can hold. Fields and their meaning are
documented on `FV_materials_detail` in [`src/tinygltf_utils.hpp`](../src/tinygltf_utils.hpp).
The tiling is the detail textures' own `KHR_texture_transform`, so `getTexture()` scales the
sampling gradient with it; the weight texture uses the surface's UVs.

Figure Viewer also writes it for a garment's built-in fabric (the Clothes tab's `fabric` edit): the
fabric table in [`figure/materials/fabric.cpp`](../figure/materials/fabric.cpp) gives each weave a
tileable height map and roughness map in the assets library, its physical tile size and its relief.
Lowering (`fabricDetail`) replaces the surface's detail layer with one at weight 1 whose tiling
comes from the material's measured `metresPerUv`, so the weave keeps its real size whatever the
garment's UV layout. The relief is fixed per fabric rather than derived from the tiling, so a
change of the fabric's scale only changes the texture transform and cooks nothing new. A GLB export
leaves the detail layer out, as it does PBRSkin's.

It also writes it for skin's micro-relief (`skinMicroDetail` in the same file): skin under a random
walk with no detail map or top coat of its own gets a tiled height map of fine creases and pores, the
app's like the fabrics'. The walk melts it out of the diffuse light, so what shows is the highlights
breaking up.

Where a hit's footprint picks a mip past the first (`MeshState::texGrad` above 0), the relief that
mip averaged away becomes roughness on every lobe, after the films (`evaluateMaterial`): the
variance of the slopes it averaged, from how short the mip's mean normal is (Toksvig,
`toksvigVariance` through `footprintVariance`), as alpha² + 2 × variance × (scale × weight)². The
app's path tracer samples every hit at mip 0 (`ptTexGradScale` 0, which the app doesn't send), and
there nothing was averaged: a pixel's own samples resolve the relief, and a texel's length is only
its encoding's error, so `footprintVariance` gives 0. BC7's error is no sliver at the base: on the
sweat beads' map it reads as 13 times the variance the 8-bit normals hold (0.012 against 0.0009),
twice at mip 1, and agrees from mip 2 on. A detail map cooks to keep its length (`normalLength`:
BC7 from unit texels, where BC5 would rebuild it) for the preview below, and the shader reads the
length only where the image's mips came with it (`eTexInfoCookedMips`, from the residency's
`storedLevels`): a raw stand-in's or an Original texture's texels are short themselves.

The moving view's thin preview leaves a measured skin's relief out altogether, where one sample a
pixel would only shimmer, and takes all of it as roughness: the whole map's variance, from its last
mip (`getTextureMean`). The detail's roughness multiplier applies there as in the still.

Every other texture a figure surface samples gets the surface's UV transform as
`KHR_texture_transform` from lowering (`surfaceUv` in
[`figure/materials/lower.cpp`](../figure/materials/lower.cpp)): Iray Uber's authored Horizontal /
Vertical Tiles and Offset, with the Clothes tab's tile, offset and rotation edit laid after it. The
detail layer's normal and roughness textures keep their own tiling. A build whose textures are not
row-flipped (`LowerOptions::flipV`) mirrors the transform in V (`mirroredV`), so the picture lands
where it does in a flipped build. The matrix the renderer builds (`KHR_texture_transform::
updateTransform`) is the spec's offset, rotation, scale order with the rotation sense of the spec's
own example; `tests/test_material_cache.cpp` pins it.

In `evaluateMaterial()` the detail block runs right after the base normal map: it bends the
already bent normal in the surface's tangent frame (`applyDetailNormal()`), so the detail layers
on top of the normal map instead of replacing it, and scales roughness under the same weight. The
clearcoat block layers the same detail on the clearcoat normal. It follows the usual gating
(`MAT_EXT_DETAIL` for layout, `GLTF_USE_DETAIL` from `SceneFeatureSet::eDetail`) and has no
Inspector section.

### `FV_materials_eye`: the eye's chamber behind a tear film

A vendor extension written by Figure Viewer on the surface a camera sees an eye through, the tear
film (`mat::eyeShell`, and `chamberIor` on `mat::Material`). The surface stays thin-walled to the
BSDF, shadow rays and the medium rules. Only a path's ray entering through its front face takes a
different direction, refracted into the chamber at `chamberIor` (`rayIntoEyeChamber()` in
[`shaders/pathtrace_functions.h.slang`](../shaders/pathtrace_functions.h.slang)), and the clear-glass
guides follow that ray. Rays leaving the eye go straight, so the eye needs no medium, there is no total
internal reflection to lose light to, and a lit iris is as bright as without it. A GLB export leaves it
out. It has no Inspector section.

**The cornea's caustic.** The surfaces in the chamber (the iris, the white: `inChamberIor` on
`mat::Material`, the same extension, a negative `chamberIor` in the shade material) take a lamp's or
the sun's light sample bent through the film too, so a light from the side gathers on the far side of
the iris and leaves the near rim in the film's shadow
([`shaders/eye_caustic.h.slang`](../shaders/eye_caustic.h.slang), `ptEyeCaustic`). The ray from the
shaded point through the film to the light is found by Newton's method on the film's own mesh, which a
ray query sees alone (`INSTANCE_MASK_TEAR_FILM`, `EyeFilmMesh`); the sample then takes the direction
the light arrives by, the light scaled by how much the film concentrates it (the beam's cross-section
over the patch it lands on, so a flat film changes nothing), and a shadow ray from where it came
through. Where no ray comes through, the sample gives nothing. Environment and emitter samples, and
BSDF rays leaving the eye, still go straight. The maths is shared with
`tests/test_eye_caustic.cpp`, which holds it to a beam traced through a sphere's cap from the light's
side.

### `FV_materials_pane`: thin-walled glass that is a pane

A vendor extension written by Figure Viewer on a refracting surface DAZ marks Thin Walled: a pane drawn
as one polygon. It has no fields. Such a surface reflects at both its faces, the light between them
going back and forth (incoherently): per polarisation 2r/(1 + r) of one face's r (`plate_fresnel()` in
`nvshaders/pbr_ggx_microfacet.h.slang`), 7.7% head-on for glass where one face gives 4%, 15% at 60
degrees. It travels as `GltfShadeMaterial::plate` to `PbrMaterial::plate`, and every place that weighs
a dielectric's reflection against its transmission reads it: the lobe weights' albedo
(`ggxPlateAlbedo()`), each microfacet's Fresnel (`BSDF_FRESNEL_PLATE`), the transmission lobe, the shadow
rays (`getShadowTransmission()`) and the clear-glass guide. Only where the surface is thin-walled
(`bsdfIsPlate()`): a medium's wall is one interface. The eye's shells keep one interface, and so does a
glTF file's thin-walled transmission, which the extension's absence leaves as KHR_materials_transmission
defines it. A GLB export leaves it out. A pane modelled as a closed box with Thin Walled on counts four
faces.

### `FV_materials_procedural`: sweat and the like over a surface

A vendor extension written by Figure Viewer for a layer an edit puts over a figure's surface. A layer
is one row of the table in [`figure/materials/procedural.cpp`](../figure/materials/procedural.cpp), which
says where it lands and what it does there, so a new one is a row, its tile and its edit.

Where it lands is a field over the body: soft patches around the skeleton's landmarks in the rest pose
(`bodyLandmarks`, from the bones' anatomical slots, so Genesis 2 to 9 share it), which a layer can break
up with 3D noise (sweat's film doesn't, as patches of film read as blotches; its beads do, so they
cluster). A layer has two fields, its film's and its beads', each in a channel.
It is written per drawn vertex into a channel of the mesh's own stream, `masks`, beside `colors`
(`proceduralMasks`, `FigureFeed::syncMeshes`). So a layer has no UV seams, rides with the skin through
any pose, and bakes nothing into a surface's maps. A mesh has the stream only once a layer has been on it
(`FedMesh::masks`): the layer's first coming makes the mesh again, and its amount after that only changes
the material row.

What it does is in `evaluateMaterial()`: the amount grows the layer from where the mask is highest until,
at 1, it covers everything the mask touches; a tile says what is finer than the mesh (how early each
droplet shows). The film is the surface itself made smoother and a little darker; the beads are a clear coat of
their own, domes from the tile's normal map at water's roughness (0.04). Where a footprint picks a
coarser mip, its shortened normals turn into coat roughness as any lobe's over relief does (alpha² +
2 × variance at the bead normal's scale); at mip 0, the app's, each bead keeps its glint and a still's
samples average far ones. The gating promotes clearcoat and specular with `eProcedural`
(`detectMaterialFeatures`), since a scene with neither would compile out what the layer writes to. As the
detail layer, the droplets leave a measured skin's thin preview, where one sample a pixel would only
shimmer. A GLB export leaves the extension out; the masks are not exported either.

A row holds one layer, the first the surface takes (`proceduralOn`): a second at once needs the row's
fields twice and the evaluation in a loop.

### FV water: rain on the skin

Not a glTF extension: water lying on a surface, kept beside its material record
(`rs::MaterialRecord::water`, [`figure/materials/water.hpp`](../figure/materials/water.hpp)) rather than in
it, since a pose moves it and no material edit should cost that. Rain on a person's skin makes it
([`figure/physics/rain/`](../figure/physics/rain/rain_sim.hpp)): once the pose stops, a seeded replay from a
dry skin on the posed cage (a `StillPass`), whose beads and stretches of track lie in the skin's UVs as
domes (spherical caps). Each dome carries the 2x2 that takes an offset in UV to metres along the tangent
frame the renderer builds (`uvToTangent`, after `vertex_tangents.comp.slang`), so it is round on the skin
whatever the UVs do there, and its normal needs no tangent of its own. `FigureFeed::setWater` gives each
surface that takes water (`takesWater`) its layer, v turned when its mesh is fed so.

The residency packs every slot's water into one table (`rs::packWater`, `src/scene_water.*`): per material
slot a grid over the UVs its domes reach, each cell listing the domes that reach into it, and how damp the
surface is. At a hit, `sampleWater` ([`shaders/water.h.slang`](../shaders/water.h.slang)) takes the highest
dome among its cell's; `evaluateMaterial()` lays the damp over the surface as sweat's film is, and a dome as
a clear coat with the dome's normal. It is drawn by the procedural layer's code, so a material with water
counts as `eProcedural` (`detectMaterialFeatures`); a shader compiled without it leaves the table unread. As
a coat, a dome reflects but doesn't bend what is under it. A GLB export leaves it out.

### `FV_materials_flakes`: Iray Uber's metallic flakes

A vendor extension written by Figure Viewer for the Uber's Metallic Flakes channels (`Surface::Flakes`,
`mat::Flakes`). The flakes are MDL `base.mdl`'s `flake_noise_texture` and `flake_noise_bump_texture` as
the Uber calls them, ported to [`shaders/flake_noise.h.slang`](../shaders/flake_noise.h.slang): cells of
the mesh's own UVs `size` across, made ragged by a smooth noise, each a flake with a weight and a tilt
from a lattice noise at its cell. The lattice is a hash rather than `base.mdl`'s permutation tables, so
the pattern is not Iray's flake for flake, only alike in size, weight and tilt.

In `evaluateMaterial()` a flake replaces the base outright where it shows: a metal of the flakes'
colour and roughness on the flake's normal, with the coat still over it. Iray mixes each flake in by its
weight; here a flake shows or not by a draw that stays with its cell (`flakeShows`), the weight being
the chance, so a sparkle keeps still from frame to frame under DLSS rather than averaging to a sheen.
The Uber's flake thin film, the roughness map and Backface flakes are not read. A GLB export leaves the
extension out.

### `FV_materials_edge_tint`: a named metal's edge

A vendor extension, `{edgeTintFactor: [r, g, b]}` (`mat::Material::edgeTint`, `PbrMaterial::edgeTint`):
OpenPBR's F82 tint, its `specular_color` on a metal. glTF's metal is Schlick's from the base colour to
white, which real metals overshoot toward grazing: measured chrome, iron and titanium reflect 0.15-0.2
less than Schlick's at 80-85 degrees, aluminium 0.1, and gold and copper keep more of their hue there.
With a tint the metal lobe takes `f82Fresnel` (nvshaders `bsdf_functions.h.slang`, `BSDF_FRESNEL_F82`):
Schlick's less a term in mu (1 - mu)^6 that makes it the tint times Schlick's at mu = 1/7, about 82
degrees, the same head-on and at grazing. Turquin's multiple scattering gives the lost light back by
the f0 scaled by what the tint leaves of the Fresnel's hemispherical average (`f82AverageShare`,
closed form).
A flake keeps Schlick's: it is a metal of its own colour.

Neither DAZ nor glTF authors one. A surface gets it by naming a metal (`Edits::metal`, material.set's
`metal`), from the table in [`figure/materials/metals.cpp`](../figure/materials/metals.cpp):
each metal's colour head-on and its tint, from measured optical constants integrated over the visible
spectrum ([`utils/metal_edge_tints.py`](../utils/metal_edge_tints.py) prints it and names the data). The
lowering and a model's edits put the tint on a surface with metal, and the metal's colour where the
colour is neither edited nor a map's. Without a named metal there is no extension, `edgeTintFactor`
stays white and the lobe is Schlick's, bit for bit (`BsdfLobes.AMetalWithoutAnEdgeTintIsSchlicksBitForBit`).
The DLSS specular guide takes the tint off its Schlick fit (`EnvBRDFApprox2`) by what the tint leaves
of the lobe's albedo (`dlssEdgeTintShare`, [`shaders/dlss_specular_guide.h.slang`](../shaders/dlss_specular_guide.h.slang)):
the Fresnel's own ratio at the view for a mirror, its average for a rough lobe, within 5% of the lobe
from a mirror to alpha 1 and head-on to 87 degrees (`DlssSpecularGuide.TheEdgeTintShareIsTheLobes`;
[`utils/f82_guide_share.py`](../utils/f82_guide_share.py) integrates it offline). Schlick's alone was a
third too bright on a polished chrome rim.

### Woven cutouts

A garment's or a prop's BLEND cutout is a weave (`mat::Material::woven`, set by the lowering and kept
in glTF as the `fvWoven` extra): `getOpacity` takes the ray's direction and
[`shaders/woven_opacity.h.slang`](../shaders/woven_opacity.h.slang) closes the gaps between two sets of
threads as the surface turns away from it, so sheer cloth darkens towards its outline. Camera rays,
shadow rays and light samples all see the same opacity; holes and solid cloth do not change.

Glass that says so by its name (`mat::saysGlass`: a set's window painted as a half-open cutout) is
the exception: no weave, as open at a slant as head-on. Woven or not, such a sheet is a film
(`mat::Material::film`, the `fvFilm` extra): the moving view draws its cover at the surface instead of
by the triangle's cells ([denoising.md](denoising.md#stochastic-alpha)).

### A wear that doesn't fit GPU memory

`GltfRenderer::geometryShortfall` weighs what a wear would add. Falling short is not the end while
`fv::wearCanShrink` says there is less to build: the wear's worker prepares it again one
`fv::oneStepDown` lower (a subdivision level less, half the strands, to a quarter) and commits again.
`mesh::Lines::keep` is the share of a hair's strands drawn, chosen by a golden-ratio sequence so a
smaller share's strands are some of a larger one's. They are dropped from the drawn ribbons only: the
cage, and so the skin, morphs, binding and simulation, keep them all, and `loadDson` widens what is left
by as much. The reduction lives on `WornItem::reduced` and in no file. What a wear replaces (the same
garment again, `Figure::itemsReplacedBy`) counts as leaving when it is weighed, and an item taken off is
fed out at once, so its memory is there for what comes next.

System memory follows the same way. An item taken off lets go of its built mesh and skin
(`Figure::removeItem`; the slot stays, since the indices of what came after it count on it, so
`Mesh::geo` can be null), and of the parsed files only its figure's store still held (`Store::trim`, not
while a `Store::Hold` stands: a loader's raw pointers into a file are good only while it lives). A wear is
given the skeleton and the body (`wearBase`), not every mesh's skin. A refined mesh's record reads the
stencils and triangles in its `BuiltMesh` (`RefineTables::owner`), and a build lets go of the faces its
stencils were worked out from. Large builds go side by side as far as they fit (`jobs::MemoryGate`, on
`mesh::buildBytes`), and one alone always goes.

Short of GPU memory, one plan says what gives way, the cheapest loss first (`rs::planToFit`,
`memory_plan.hpp`): images nothing samples, items nothing shows (`Residency::letGoHidden`: gone until something
shows them, and not what the scene is "drawn without"), then the largest textures' top levels, with the face a
level above and what is marked Keep full (`MaterialRecord::keepFull`) last of all. `rs::MemoryWatch` says when and
the renderer asks for a step of it at a time (`Residency::kFitStep`), reading the memory again between. A level
comes down by copying the levels kept into a new image (`GpuBackend::copyImageLevels`): nothing read, nothing
decoded, a frame's copy commands. An image with no stored levels (a PNG kept whole) comes down the same way, by
the levels the GPU made, and comes back all at once. Room back, what lies in system memory comes first
(`Residency::bringBack`: an image by a copy, a mesh made again in one sync, so no frame goes without it), then
the textures. These swaps, and a BLAS compacted, leave the picture as it was (`SyncResult::samePicture`), so a
still goes on through them and starts once more when textures stop moving.

Under the same pressure, what nobody is editing settles (`Residency::settle`), before any texture comes down, and
what would show waits for what is about to (`settlingSoon`), once and for as long as settling takes: a
mesh with drawers only, none written for `settleQuietFrames`, the selected figure's last of all (`setLive`, which
wakes it as it becomes selected). What only remakes it goes: tables, cage, skin inputs, morph deltas, its own streams, last
frame's positions. Its drawers' outputs stay, so the picture does, and its BVHs are built again compacted, one a
sync and only with room for it beside the old one (`m_settleBuilds`). Anything that would change it wakes it first
in the same sync (`wakeWhatChanged`: a pose, a morph weight, a vertex edit, a new draw, being selected); an unchanged
wake writes nothing. A drawer's matrices are written before each run, so a pack laid out again no longer runs every
drawer: a wear doesn't skin every other figure.

A strand hair's record is thin once the feed knows every drawer poses its cage (`MeshRecord::thinVertices`): `positions`, `normals` and the drawn skin are
empty, and code asks `vertexCount()`, `hasNormals()` and `skins()`. It stays fat when a group's material gives light, since `syncEmissive` reads the
triangles. `mesh::skin` isn't made for a ribbon: an export makes it with `mesh::skin(geo, cageSkin)`. `widenRibbons` turns the ribbon's streams into `offsets`
and lets them go, so it works once. `MorphSet::cageOnly` is `FigureFeed::refinesOnceFed`, which is true of a mesh not yet fed: a wear's dials come before its
first sync, and would otherwise run the whole CPU morph on it and keep what that made. To find what the heap holds, a temporary `operator new` that keeps a
`CaptureStackBackTrace` of large allocations, a linker map from `_LINK_=/MAP:file` and one relink name it per function; that is how this was found.

What can be worked out off the frame is. `mesh::build` makes the normal tables (`BuiltMesh::normalStart`, `normalTriangles`), which the feed only spans. A hair's
votes on its strands' rotation become a `mat::StrandTally` on the worker, and the feed adds tallies (`strandRotation`) instead of reading millions of votes for every feed.

Past that, a worn item comes down in detail while it is in the scene. A subdivided or strand mesh keeps what it was built from (`BuiltMesh::source`), and `mesh::remade` builds it again at another level or
share of strands over the cage it has now, from which everything on the cage stays true: morphs, a pose, a drape, bindings. `Lines::keep` draws nested shares, so a quarter's strands are among a half's. The
Fit only asks (`rs::ItemDetail`, `FitRungs::detail`): the host offers its items once a second (`FvHost::Impl::offerItemDetail`), makes one at a time on a worker (`stepItemDetail`, `Figure::buildRemade`),
and the feed swaps the record because the mesh changed. So nothing that outlives a build may hold a `BuiltMesh`: `dson::MeshRef` and a follower hold the cage's vertex count, and a test checks that the old mesh goes.
A target's drawn deltas are made per mesh size, and `MorphSet::meshRemade` lets go of what was kept of the old drawn vertices.

Near full, what first hits want decides what gives way first. A scene with room never counts. Otherwise eight frames of each rest count one pixel in sixteen, per material and by how large a long edge the hit could show at two texels a pixel (`shaders/hit_signal.h.slang`, compiled as C++ for its test; `PathTracer::countHits`). The footprint is `worldFoot * texelDensity` and not `texGrad`: `ptTexGradScale` is 0 in the app, which would make every hit want nothing. Counting starts on whichever frame finds memory near full, not only frame 1, since memory can run short in the middle of a rest, and a view whose samples end inside the window hands over what it counted (`onAccumulationComplete`). Moving frames never count.

`rs::ImageFit::weight` divides a level's bytes by the share of hits that would miss it, with `kLeastHitLoss` as the floor, so a level no hit wants goes first and the largest of those first. `detailSteps` orders worn items the same way, and an item no ray reaches goes before any texture (`FitRungs::detailFirst`). A new count is then followed through by `Residency::followHits`, a step at each reading of the memory, each only once the last shows in what is free (nothing queued, nothing retiring, no item being made), and without the wait a raising has after a lowering, because the view just asked. With room, what hits miss comes back, and only that (`raiseImages`'s `onlyWhatHitsWant`). Without room, `lowerForExchange` frees what the raise needs at its peak, from what hits want at least twice less. It needs hits that really are missed on the rising side: comparing weights alone had a small level nobody wanted trade against a large level nobody wanted, three times over, and everything went to 256 for one close-up. Items come back most reached first, and those no ray reaches after the textures (`nextDetailUp`).

Grey maps are half the size from their next cook on: the cooker writes one grey as BC4 with a `KTXswizzle`
(`texcache.rs GreyMap`), which the images it cooked before need not, and a lobe's roughness from the file the
base roughness uses asks for the same image.

A cooked map of one value everywhere (a flat roughness, a black mask, a flat normal map) loads as the one
block it is made of, a 4x4 image of one level (`constantBlock` in `src/gltf_image_loader.cpp`,
`LoadedImageData::constant`): 27 of the 1811 maps in the user's cache on 11-06, 374 MiB when resident. Level 0
must be that block repeated and level 2 the same block, so a pattern repeating every four texels, which has
averaged to another colour by level 2, stays whole. Only a level 0 its zstd shrinks below a thousandth is
inflated to look, so other maps cost nothing; the chain the planner reads (`readImageChain`) says 4x4 too,
leaving it nothing to lower or raise, and a light's mean reads the block (`imageMeanLuminance`). The cache
is read as it is: nothing recooks. Count a cache with `ImageLoader.DISABLED_CountConstantMapsInACache`.

Copies of a prop the scene sets alike draw one `fv::Mesh`, as DAZ Studio instances do, so the feed makes one
record and the residency one BLAS for all of them, each copy a TLAS row of its own (`loadDson`'s `sharedMesh`). A
copy is alike in its geometry, UV set, level and materials (the indices `materialKey` already shared). A figure's,
a wear's, a skinned or simulated mesh, a strand hair, a shell or its source, and a node a scene dial sits on keep
their own. A prop on the scene's figure is an item of it (the night club is a figure, its lights hang from its
walls, which are bones), and shares when its part has no graft, rigidity or smoothing. Such an item lists the
shared mesh, and taking it off frees the mesh only once no shown node draws it (`liveMeshes`). On the night
club, 103 of 158 mesh nodes are copies, 2.18M of the 3.52M triangles drawn; the 14 meshes still repeating a
geometry differ (eight artworks with eight pictures). Count a scene with `SceneLoad.DISABLED_CountSharedMeshesInAScene`.
Code that places a mesh by its node must take the node: `physics::nodeOfMesh` gives only the first, so a body
meets each copy through `posedNodeCages`.

A commit either finishes or takes off what it merged (`commitWear`'s `TakeOff`), so a failed allocation
anywhere in it leaves the figure whole; a test fails each allocation of a wear in turn. Workers report what
they let escape (`FvWorkers::onError`, `sideBySide`), and a figure's part of the frame reports it and goes on.

`rs::RamWatch` reads system memory once a second: short under a sixteenth of it or 2 GiB, of it or of what
can still be committed. Then what is only kept for later goes: props removed for an undo
(`kParkedBytes`, 4 GiB otherwise), textures decoded ahead (`ImagePrefetch`, 1 GiB otherwise, and what nobody
takes for two minutes goes anyway), the shared parsed files; a notice says so, once. The folders on disk
keep to budgets too: parsed files 5 GB (`.fvd`, `io::trimCacheFolder`; a damaged one is deleted and written
again), bindings 1 GB. The texture cooker's eviction (`texcache.rs`) leaves what this session handed out, so
a scene larger than its cap stays over it until the next start, and its parallel work runs on half the
threads at below-normal priority (`background_threads`).

### Textures following GPU memory

The budget moves while a scene is open: another program takes memory and gives it back. Once a second
`GltfRenderer::watchMemory` reads the headroom and `rs::MemoryWatch` says what to do, asked of the next
sync (`Residency::lowerImages`, `raiseImages`; a sync with no scene change runs for them). The policy's
times keep a wandering budget from turning into textures going up and down, and `kMargin` stays free
after either, so the one doesn't call for the other. A change of texture restarts a still, so a still
being refined and an export (`render.hold`) are left alone until memory passes the driver's budget.
A raising that has begun carries on through the still it restarted, at `kRaiseBatch` bytes a second,
because the frame waits for the decode. An image is decoded whole beside the one it replaces, so
`raiseToFit` counts its whole chain. `MaterialRecord::favoured` (a figure's face, `mat::onTheFace`) makes
a material's images count as a level smaller on the way down and on the way back, so they stay at least a
level above the rest. A large image has its own allocation (`createSampledImage`, 4 MB and up): freed inside
a block it could only be used by this process, and the driver's usage would hardly move. A render with no host
doesn't watch, so that two of them compare; `isAutomatedRun()` can't say so, as the app's renderer is headless too.
`scripts/texture-memory-check.py` runs all this on the GPU, which the unit tests never reach. The `memory` topic
carries what is lowered; `src/lib/native/quality.ts` joins it with the wardrobe's for the viewport and the Export dialog.

### `FV_materials_hair`: a fibre on a ribbon

OmniHair strand hair is drawn as flat ribbons. The path tracer shades those with a fibre BSDF
(Chiang's R, TT and TRT lobes, plus the remaining orders, averaged across the fibre) instead of a
dielectric sheet. The strand direction is the ribbon's own bitangent, so each hit follows its own
strand. It runs root to tip, as the cuticle's tilt shifts the highlights along it: the mesh's
bitangent (`HitState::frameSign` undoes the turn `getHitState` gives a back face's frame), against it
or along it by where the root is (`hairRootV`), the same on both faces of a ribbon (`makeHairHit`).
Root and tip absorption stay separate and are blended by the along-UV. The first stretch of each
ribbon narrows in the mesh, so the body stays opaque and does not pay any-hit; a cutout that is
already blended also fades its opacity there. The panel's roughness and specular edit the fibre
(β, and the reflecting orders), not only the unused sheet lobe. Fields are on `FV_materials_hair`
in [`src/tinygltf_utils.hpp`](../src/tinygltf_utils.hpp). Card hair takes the fibre too
(`hairFromColor`: each hit's absorption from the colour the card shows), its strand across the
measured anisotropy rotation; a card's root is not known, so its strands keep the bitangent's way,
which may tilt its highlights the wrong way along the strand, but one way on both faces. A glTF written for other tools keeps the old diffuse-transmission
sheet, because they have no fibre BSDF. Gated like the detail layer (`MAT_EXT_HAIR`, `GLTF_USE_HAIR` from `SceneFeatureSet::eHair`).

### Strand hair the file only guides

A strand file holds guides; DAZ grows the rest when it renders. `prHairsFor` (`figure/dson/dforce.cpp`)
reads what each surface grows, and `grownLines` (`figure/mesh/mesh_build.cpp`) grows it:

- **Root Radius:** PR Hairs Per Guide around each guide. The PS multiple is not grown.
- **Target Surfaces:** a density (hairs per cm²) at the PR stage, else the PS one, on the mesh the hair's
  node hangs on (`growthSurfaceFor`). No channel says which polygons: Surface Growth Groups is empty in
  the one product that uses this mode, so the area is the polygons whose corners all carry a guide of
  that surface. On AO Moonborn that adds up to the area the guides were authored on, and the
  surrounding tube and scalp grow nothing.

Either way a grown hair is points of its guide (one stencil column each, so skin, morphs and fits are
the guide's) plus a rest-shape offset, `Strands::shift`. A density's hairs lie between the three guides
around the root and toward their nearest as the single guide settings say, and take random length, frizz
and scraggle. DAZ documents none of these: the amounts' units come from the sliders' labels and ranges,
and the shapes are uncalibrated. Not grown: clumping, random angles, auto parting, Surface Lattice Smooth.
A hair is drawn with a point per centimetre (`kGrownSegmentCm`), not the file's 0.3 cm, and a mesh
grows at most `kMostHairsGrown`.

### Adding a new KHR_materials_* extension

1. Define a new `MAT_EXT_<NAME>` flag in [`gltf_material_config.h`](../shaders/gltf_material_config.h),
   default `1`.
2. In [`gltf_scene_io.h.slang`](../shaders/gltf_scene_io.h.slang), add the new struct fields
   (+ any texture slots) inside `#if MAT_EXT_<NAME>` with inline default values.
3. Add `GLTF_USE_<NAME>` in [`gltf_eval_config.h`](../shaders/gltf_eval_config.h) (default
   `MAT_EXT_<NAME>`) and the matching `#error` guard.
4. In [`gltf_material_eval.h.slang`](../shaders/gltf_material_eval.h.slang), add the evaluation
   block inside `#if GLTF_USE_<NAME>`, overriding the relevant `PbrMaterial` fields. Respect
   the existing block order (volume block runs before IOR, etc.).
5. In [`src/gltf_material_cache.cpp`](../src/gltf_material_cache.cpp), add the matching CPU-side
   population block under `#if MAT_EXT_<NAME>` — layout only; never toggle `MAT_EXT_*` at runtime.
6. In [`src/scene_feature_detection.cpp`](../src/scene_feature_detection.cpp) and
   [`src/scene_shader_macros.cpp`](../src/scene_shader_macros.cpp), detect scene usage and emit
   `-DGLTF_USE_<NAME>=0|1` in optimal mode.
7. If the path tracer reads the new field outside `evaluateMaterial()` (e.g.
   `getShadowTransmission()` in `pathtrace_functions.h.slang`), gate with `#if GLTF_USE_<NAME>`.
8. Update the extension support list in [`README.md`](../README.md).

### Adding a non-PBR material extension (guide-buffer pattern)

Some material extensions don't influence the BSDF at all — they feed denoiser / post-process
inputs instead. `EXT_DLSS_NR` is the reference implementation: it writes a 4-channel per-material
mask into a dedicated DLSS-NR guide-buffer slot rather than touching `evaluateMaterial()`. The
checklist differs from the PBR pattern above:

1. `MAT_EXT_*` flag + struct field in `gltf_material_config.h` / `gltf_scene_io.h.slang` (same
   as PBR, but **no** `gltf_eval_config.h` / `gltf_material_eval.h.slang` entry needed).
2. Add a `GBuffer` slot to `OutputImage` enum (`shaders/shaderio.h`) and the matching `VkFormat`
   to `kRrInnerFormats[]` (`src/dlss.cpp`).
3. Write the slot in `processPixel` (`shaders/gltf_pathtrace.slang`) inside the first-hit block,
   guarded by `USE_DLSS_SHADER && MAT_EXT_<NAME>`. Propagate through `GuideScratch` →
   `GuideOutput` → `packSampleResult` (`shaders/pathtrace_functions.h.slang`) if the value needs
   to survive path tracing.
4. Register the new slot in `kRrSlots[]` (`src/renderer_pathtracer.cpp`) so it is wired into
   the `outImages` bindless array.
5. Wire the new NGX input in `Dlss::evaluateNr()` or the appropriate evaluate function
   (`src/dlss.cpp`), reading directly from `m_innerGBuffer`.
6. tinygltf helpers (`src/tinygltf_utils.{hpp,cpp}`), material cache loading
   (`src/gltf_material_cache.cpp`), follow the same pattern as PBR extensions — see
   `EXT_DLSS_NR` for the complete example.
7. Update [`docs/denoising.md`](denoising.md) (for guide-buffer / denoiser integration).

## Lighting

`EXT_lights_ies` provides photometrically accurate angular light distribution using IESNA LM-63
profile files. Per spec it defines a **self-sufficient point light** and does not compose with
`KHR_lights_punctual`. If a node carries both extensions, `EXT_lights_ies` wins and the KHR
attachment on that specific node is dropped at load with a warning (`rs::importGltf`); the shared KHR
light entity in `model.lights[]` is preserved, so other nodes referencing it are unaffected.

![IES profiles — ring and scatter distributions on floor and wall](images/LightsIES_original.jpg)

[`src/ies_profile.hpp`](../src/ies_profile.hpp) parses the `.ies` file into a normalized,
azimuthally-averaged 1D candela curve (see its "Simplification" note on why genuinely asymmetric
fixtures lose their azimuthal variation), and `loadIesProfiles` reads a model's profiles for both
scenes (a `bufferView`, or a `uri` on the image search paths); `GltfScene.iesProfiles` carries the
flattened per-profile tables to the GPU, and `GltfLight.iesProfile` indexes into it (see
`shaders/gltf_scene_io.h.slang`). `nvvkgltf::packIesLight` (scene_gpu_tables.hpp) makes the light
row both scenes upload. The retained scene keeps the profiles in the store
(`RetainedScene::iesProfiles`) and an IES node's light as a `LightRecord` with `ies` set.
Both render paths evaluate the profile identically via `evalIesFactor()` in
[`shaders/gltf_light_ies.h.slang`](../shaders/gltf_light_ies.h.slang).

The Inspector exposes `multiplier` and `color` as editable property-editor rows on any node
carrying `EXT_lights_ies` (`UiInspector::renderNodeProperties` -> `renderIesEditor`). Edits are
undoable via `EditNodeIesCommand`. The profile index itself is read-only (it refers to the loaded
`.ies` file array). A node that also carries `KHR_lights_punctual` keeps only the IES light: the KHR
attachment is ignored.

## The moving view's bounce cache

In a room most of a moving frame goes on light that has bounced many times. With `ptRadianceCache` on,
the frames DLSS draws stop a path at the first surface it reaches after a rough bounce and take what that
surface sends back from a cache kept in world space, a hash grid of cells that one path in sixteen
feeds; the cache's own header, [`shaders/radiance_cache.h.slang`](../shaders/radiance_cache.h.slang),
describes the scheme and holds its constants. What the camera sees directly, in a mirror or through
glass is always traced, and a still never reads the cache: it is biased (a cell holds one colour, and
follows a change over some frames).

- The path tracer's part is compiled in only for the variant of DLSS's frames
  (`USE_RADIANCE_CACHE`, `VariantKey::radianceCache`), so a still runs none of its code. A variant with
  it and one without stand in for each other while the other builds off the frame.
- [`src/radiance_cache.hpp`](../src/radiance_cache.hpp) owns the table and runs the pass after each
  such frame. The frame's info carries the table's address (`SceneFrameInfo::radianceCache`); a null
  address is a frame without a cache.
- The cells keep their light across stills. A restart of the accumulation for anything but the camera
  (`Resources::sceneChanges`, counted by `GltfRenderer::resetFrame`) cuts every cell's weight, so an
  edited light or pose is taken up within some frames; replacing the scene empties the table.

## Environment Lighting

`Settings::envSystem` (`shaderio::EnvSystem`) selects the environment source. The shaders do not
branch on it: every mode but `eNone` reaches them as **one lat-long image plus its alias table**
(`Resources::hdrIbl`). That is the whole point of the design — sampling and MIS stay free of
per-sky-type code, and adding a sky type adds no shader branches.

An HDR file arrives there by being loaded. The analytic types — the physical sky and the
`OMI_environment_sky` types — arrive there by being **baked**: [`src/env_baker.*`](../src/env_baker.hpp)
dispatches [`shaders/env_bake.slang`](../shaders/env_bake.slang) — one switch over every sky type
— into a lat-long image and an importance buffer, then hands the pair to
`nvvk::HdrIbl::updateFromGpuImage`. The bake's
bindings and push constant are ours ([`shaders/env_bake_io.h.slang`](../shaders/env_bake_io.h.slang));
what nvpro_core2 fixes is the lat-long mapping and texel weighting (`latlongToDir`,
`texelSolidAngle` in `nvshaders/functions.h.slang`), next to the functions that read the image back.
What a sky *is* stays in this repository, next to the glTF loader that reads it.

The bake size is fixed (`kEnvBakeSize` in `src/env_baker.hpp`), and the alias table is **not**
built at it. The bake sums its importance per cell of a coarser sampling grid (`kEnvSamplingGrid`,
same header) — a shared-memory reduction inside each workgroup, so it costs next to nothing — and
`updateFromGpuImage` builds the table over the cells, so its cost is the grid's, not the image's. A sample picks a cell and then a
direction uniformly in solid angle within it, and the PDF written into the image's alpha is the
cell's, so sampling and MIS agree. That suits a baked sky, whose lighting field is smooth because its
sun is a light of its own; an HDR file keeps one entry per texel, since a photographed sun is a few
texels wide. Shaders therefore sample with the grid (`SceneFrameInfo::envSamplingGridWidth/Height`),
never the image size.
`EnvBaker` splits that into a cheap `preview()` (colour only, while something is moving) and a full
`commit()`. Which one runs is decided by `GltfRenderer::requestEnvironmentPreview()`: every edit
source reports through it, and the commit is paid on the first frame that asks for no preview. That
settle test, rather than an ImGui "is a control held" question, is what keeps a sun driven by the
gizmo or by an animation from committing once per frame. `GltfRenderer::updateHdrImages()` remains the single place that writes the
`texturesCube[]` / `texturesHdr[HDR_IMAGE_INDEX]` descriptors.

The image and its alias table are stored in the **environment's own frame**. One orientation,
`SceneFrameInfo::envRotation` — a unit quaternion, `OMI_environment_sky`'s `rotation` verbatim —
takes that frame to world. Every lookup brings a world direction into it with `quatRotateInverse`
(`nvshaders/functions.h.slang`). Rotating the environment therefore never re-bakes;
only the physical sky does, when the sun moves relative to it.

### Two fields: lighting and background

The environment is **two** fields, and which one a ray reads is the part that has to be right:

- **Lighting field** — the baked lat-long image, deliberately **sun-free**. The sun is a directional
  `KHR_lights_punctual` that next-event estimation samples directly; baking it into the image too
  would light every rough surface with roughly two suns.
- **Background field** — evaluated analytically per ray, sun included. Per-ray evaluation is also
  what keeps a half-degree sun disk sharp, which no practical bake resolution could. The authored
  sky types evaluate it in [`shaders/sky_background.h.slang`](../shaders/sky_background.h.slang).
  The physical sky's background is evaluated per ray in the path tracer too
  (`evalPhysicalSkyBackground` in `shaders/gltf_pathtrace.slang`), through the same function the
  bake uses (`shaders/sky_bruneton_radiance.h.slang`), plus the half-degree disk
  (`evalSunDiskBackground` in the same header, fed by a radiance the bake publishes), with the
  clouds composed over it from their own image ([sky.md § Clouds](sky.md#clouds)). What the
  camera sees is then independent of the bake resolution — the horizon, where sky meets ground in a
  hard edge, is what a lat-long would otherwise smear.

A camera ray, and an escaped ray whose last bounce was delta/specular, read the background field;
any other escaped ray reads the lighting field **plus the sun disk** (`evalSkySunDisk`),
MIS-weighted against the sun's NEE sample. The disk is needed there because a glTF roughness of 0
is clamped to a tiny but finite lobe, not a delta: NEE's cone samples almost never land in it, so
without the disk smooth glass or a mirror shows no sun. The header states the rule at its
definition because getting it wrong is silent and off by roughly 2x.

### Panorama: where the image URI lives

OMI's `panorama.equirectangular` is an index into the glTF's `textures[]`, not a path. An `.hdr`
or `.exr` is not a standard glTF image mime type, and putting one into `images[]`/`buffers[]` to
satisfy that would bloat every scene that has a sky — so this renderer stores a URI in a sibling
`NV_environment_sky_panorama` block on the same sky entry:

```json
"skies": [{
  "type": "panorama",
  "extensions": { "NV_environment_sky_panorama": { "uri": "../env/studio.exr" } }
}]
```

The interop gap is deliberate and worth stating: a reader that understands only OMI sees
`type: "panorama"` with no source and falls back to its own default. That is a visible gap rather
than a silent wrong answer, which is the better failure. An authored `equirectangular` index is
never invented but always preserved, since index 0 is a valid texture — inventing one would load
the wrong image rather than fall back.

The URI resolves against **the file that named it**, not against the resources directories — two scenes
naming the same file must not load each other's environment. On save it is written relative to the
file being written when the image lives under it, and absolute when it does not, so a scene and
its environment move together without a chain of `..` that breaks at the first move.

Decoding is `src/env_image_loader.{cpp,hpp}`, which is also the single place that decides which
formats can be an environment. It dispatches on content rather than file extension, and covers
both `.hdr` and `.exr` — see that header for why EXR cannot live inside `nvvk`.

### Where the sky lives in the scene

`OMI_environment_sky` is scene data, not a viewer preference.
[`src/gltf_environment_sky.*`](../src/gltf_environment_sky.hpp) parses the extension into a
`SkyDescriptor` and writes it back on save, keeping the raw `tinygltf::Value` so sky types and keys
the renderer does not model survive a round trip untouched. `GltfRenderer` applies a parsed sky at
the top of the frame *after* the settings-restore hooks have run, so a scene that carries a sky wins
over the persisted `.ini`.

## Common Development Workflows

### Add a command-line parameter

1. Register it once: `m_settings.add(...)` (`SettingsRegistry`) for a renderer setting, or
   `parameterRegistry.add(...)` in `main.cpp` for an app-level flag.
2. Wire it to renderer or app settings.
3. Nothing to list by hand: it appears in `--help`.

### Add a renderer setting

1. Add state in shared settings/resources.
2. Bind and consume it in `renderer_pathtracer.cpp`.
3. Persist with the settings handler when needed. The app reaches it by name through the host's
   parameter messages; `src/fv_parameter_filter.hpp` lists what the host keeps to itself.

### Add or update glTF extension support

1. Parse and serialize extension fields in `tinygltf_utils.*`.
2. If the extension affects the material model (any `KHR_materials_*`), follow the
   checklist in [Material System → Adding a new KHR_materials_* extension](#adding-a-new-khr_materials_-extension).
3. Otherwise, integrate runtime behavior in scene/shader code as needed.
4. **If the extension stores any index** (into accessors/bufferViews/textures or a root-level array
   it defines itself), add its remapping to `gltf_scene_merger.cpp` in the same change: otherwise it
   is silently dropped or cross-linked when props are composed.
5. Update extension documentation in [README.md](../README.md).

### Modify sync/data-flow behavior

Use [RENDERING_ARCHITECTURE.md](RENDERING_ARCHITECTURE.md) as the source of truth for:
- slots, side tables and the sync order
- GPU buffer synchronization
- TLAS/BLAS lifecycle

### Run under a validator

The app starts the renderer with its own arguments, so a diagnostic run is asked for in the
environment, read in `main.cpp`:

- `FV_VVL` turns on the Vulkan validation layer, or GPU-assisted validation, which checks what
  shaders read through descriptors.
- `FV_RTV` turns on NVIDIA's ray-tracing validation (`VK_NV_ray_tracing_validation`): the driver
  checks what acceleration structures are built and refitted from and how they are used, and says
  what is wrong (a NaN vertex in a refit, a mostly degenerate BLAS) before the device goes. Each
  message is written to the log as it comes and kept for a lost device's log
  (`gpu_trail::validationMessage`, `gpu_crash_trail.hpp`). The driver names no object of ours: the
  primitive count in its text tells the mesh.

Either marks the run as validated in `/stats`, so the app's watchdog waits through slower frames.

---

## The retained scene and `SceneGpuView`

The renderer draws a retained scene (daz-viewer docs/retained-scene-plan.md), which a stream host
feeds (below) or which `--scenefile` imports a file into. Both renderers read it through
`SceneGpuView` (src/scene_gpu_view.hpp, `Resources::gpuView()`); with no scene loaded yet that is an
`EmptySceneGpuView`.

### The retained scene in a stream session

The host (fv_host.hpp) feeds `GltfRenderer::hostScene()`, a `rs::RetainedScene` that lives for the
session, so its handles stay unique. A composition feeds the props' changes through
`fv::PropFeed` (fv_prop_feed.hpp): a prop that came gets a prop root entity (owner tag `{N, -1}` for
`p:N`) with a figure through its `fv::FigureFeed`, a model through `rs::importGltf` (what it leaves
out said once, D23), or a camera or a light as a record; one that went is destroyed, its instances
retiring at their frame; a figure whose wardrobe changed is fed again, keeping the handles of what it
still has; the rest are set in place (parent, transform, visibility, a light's or a camera's record)
and keep theirs, so the selection, the clip that plays and the camera stay. The sun (owner
`{0, -1}`) is a light record marked as the sky's sun. The backdrop (the same owner, one root
entity named "Backdrop") is the host's too: see "The procedural backdrop" in
[RENDERING_ARCHITECTURE.md](RENDERING_ARCHITECTURE.md#the-procedural-backdrop). Each frame's sync takes the changes, with no
device wait. Only the first composition is a scene replace: `replaceHostScene` waits for the
device, cleans up as a load does (the store and the selection stay) and makes the store resident,
waited for; with `[fv-time]` on, the `GPU BLAS build` and `GPU TLAS build` lines time its builds
(`VulkanGpuBackend::timeBuilds`); the `Scene (retained, host):` line counts
images decoded (and of those, taken from the host's prefetches) and reused. "Auto" ground is the
store's lowest drawn point after each composition (D6), the backdrop left out.

Between compositions the host writes the store directly: a pose step, a dial, a rebind and an item
shown or hidden through the feed's narrow syncs (`syncTransforms`, `syncVertices` with the tangents
near the moved vertices, `syncSkins`); a prop's transform and visibility, a light's payload and the
sun as entity and record edits. A figure's material edit is lowered by `FvMaterials` with its own
options and written by the feed (`syncMaterials`); a model's is its file's material with the edits
laid over, set on its import's record. Each frame `syncRetainedScene` moves the clip on and the
residency syncs what changed.

A subdivided figure mesh, or a widened strand mesh (daz-viewer `docs/gpu-figures-plan.md`), is fed with
its morphed cage and its `rs::RefineTables`,
and the residency makes its positions and normals on the GPU (`refineOnGpu`: the stencil levels in
`stencil.comp.slang`, the welded normals in `vertex_normals.comp.slang`, the UV seam splits as a
gather) before the refits and the skin pass. A strand mesh's one level copies each point to its
ribbon's two edges, and the tables' `offsets` (each edge's side times the half width `widenRibbons`
kept) open it. A mesh whose materials need tangents gets them from the drawn positions, normals
and UVs as well (`vertex_tangents.comp.slang`, over each vertex's corners in `computeTangents`'
order and arithmetic), so the feed makes and sends none for it, and a drag's shading is never
stale. A dial's step then moves the cage alone:
`MorphSet::cageOnly` keeps the morphed cage and reports a cage change, fed by `syncCages`
(`RetainedScene::setMeshCage`). A figure moves only cages from the start
(`dson::LoadOptions::cageMorphs`, by `mesh::drawnFromCage`, the rule the feed refines by); the host
points the predicate at the feed's `refines` once it composes, and a mesh it does not refine
catches up at the next `pump`. The CPU's drawn vertices of the rest stay as loaded until an export
reads them (`MorphSet::catchUp`), and a morph target refines its drawn deltas only when a flush or
that catch-up first asks (`RenderTarget::drawn`): a character's shape, a follower's projections and
dForce's drape steps keep cage deltas alone. A refined prim's bounds (the selection box, the scene
bounds, the auto ground) are its cage's around the cage vertices its vertices are made from
(`Prim::cageSupport`), widened by the longest offset.

A figure is skinned as its file asks:
`skinBinding` reads the skin modifier's `skin_settings` (Genesis 3 and 8 all dual quaternion,
Genesis 9 half dual quaternion, half linear), the records carry up to eight influences
(`joints1`/`weights1`) and the dual quaternion share (`dualQuat`, `dualQuatShares`), and
`skinning.comp.slang` mixes the linear blend with dual quaternion skinning by it, from the joints'
dual quaternions the residency packs (`rs::JointDualQuat` in `scene_skin.hpp`: rotation and
translation, and the stretch left over, blended linearly first). dForce and the collider skin the
cage the same way (`dforce::CageSkin`); glTF export stays at four influences, linear, and says so.

A refined mesh is posed in DAZ Studio's order: the records carry the cage's own influences
(`cageJoints`, `cageWeights` and the second four, the cage's shares), and the mesh's drawer
(`DrawerGpu::posed`) skins the cage, then runs the refine chain on the posed cage into its outputs
(`Residency::refineChains`), so a pose step refines again. Chains run one at a time over one scratch
buffer the size of the largest (`m_chainScratch`), let go of after 600 idle frames. A strand ribbon's
`offsets` turn with its points: the same skin pass turns unit axes (`m_unitAxes`) into a frame per
cage point, which the last stencil level reads (`frameX`, `frameY` in `stencil.comp.slang`). A catch-up
of the CPU's drawn vertices (`MeshChange::caughtUp`) is no reason for dForce or the collider to run again.

DAZ HD morphs (a morph's `.dhdm` detail, such as Genesis 9's FACS Details wrinkles) ride on the same
chain. `figure/dson/dhdm` reads the file into the morph's target; a cage-only mesh's flush mixes its
active targets' edits by weight (`subdiv::HdMixer`, `MorphSet::hdDetail`) into tables per level: each
row's edits, an offset in the frame of a cage face corner. The feed hands them over with the cage
(`RetainedScene::setMeshHd`), and each stencil level adds its rows' edits, the frames made from the
cage the chain refines, so the detail turns with the pose as DAZ Studio's does (`hdStart` .. `hdFrameVertices`
in `stencil.comp.slang`). Detail at levels finer than the figure's subdivision is left out. The format
and the frame are `figure/subdiv/hd_edits.hpp`'s; `subdiv::hdMove` is the shader's arithmetic, which the
fake backend and the oracle share.

Such a mesh holds nothing the route it doesn't take would read: no positions, normals or tangents of
its own, and no skin on its drawn vertices (`MeshGpu::hasBase`). They come, refined once then, when
something needs them (`Residency::ensureBase`): a static draw of the mesh, or a drawer that can't pose,
one with morph targets on the record, or tangents with no tables to make them from.

A vertex stream's normals and tangents are 4 bytes each (`shaders/unit_vector_pack.h.slang`), where
a `float3` and a `float4` took 12 and 16. A normal is octahedral, 16 bits a coordinate. A tangent is
octahedral at 16 and 15 bits with its handedness in the top bit. The reserved code `0xFFFF` in the low
half means no direction and reads back as zero. The worst round trip is under 0.005 degrees for a
normal and 0.01 for a tangent (`tests/test_unit_vector_pack.cpp`). The residency packs what it
uploads (`packedNormals`, `packedTangents`). The skin, morph and tangent passes read and write them
packed, and so does the gather of a refined mesh's drawn normals (`outUnit` in `stencil.comp.slang`).
A strand ribbon's frames (`frameX`, `frameY`) and the unit axes they come from are packed too.
Positions, the stencil levels' sums and the welded normals of a refine stay in floats. The fake GPU's
passes pack as the shaders do (`NormalSpan`, `TangentSpan` in `tests/common/fake_gpu_backend.cpp`).
The oracle packs each value once, at the pass that packs it, since a packed value need not repack to
the same bits.

What is new comes a few at a time, since every BVH a sync asks for goes in the frame's one submission: a sync takes
`ResidencyOptions::arrivalTriangles` of new meshes, and always one however large. The rest wait in `Residency::arriving()`, not drawn and not
parked, and the next sync takes them with no change in the scene.

A sync decodes the images it needs in one batch, side by side on the worker pool
(`VulkanGpuBackend::createImages`), a file the host prefetched taken from
`ImagePrefetch::take` instead. Each material slot owns a span of the texture-info table
(`MaterialCache::convertOne` at its start), so a material edit uploads its row and span only; a span
that finds no room below the uint16 limit repacks every span once, and one that still doesn't fit is
drawn without its textures, logged as `Retained scene: the texture-info table is full`. A material
or light edit resets DLSS (`SyncResult::dlssReset`).

A mesh an entity skins, or a mesh with morph targets, is deformed for that entity: its drawer owns
the output streams the morph pass and then the skin pass write (through
`nvvkgltf::DeformPipelines`), and its prims' BLASes are built over them and refit
after each pass. A pass runs only when what it reads changed: the weights, the joint matrices or the
mesh's vertices (`rs::Residency`, "Deformation"). DLSS motion comes from two tables the residency
keeps (`SceneGpuView::prevPositionTable`, `prevObjectToWorldTable`): a drawer's positions are copied
before a sync writes them; at the first sync that doesn't, its row points at the positions themselves,
and the copy goes 120 frames later, so a drag that pauses keeps it; and an instance keeps the world it was drawn with for one sync after it
moves. The clip that plays is a `rs::ClipPlayer` on the store (`setHostClip`), driven by
`AnimationControl`.

Four settings, none remembered:

- `blasScratchBudgetMB` (512): the scratch one batch of the retained scene's BLAS builds shares
  (`VulkanGpuBackend`); the next batch regrows it. Live.
- `textureGraceMB` (256): the retained scene's decoded images that nothing samples stay decoded, up
  to this much, oldest out first, for the next scene replace or edit to take. Live. Images are keyed
  by file (path, size, write time) or by a hash of held bytes, and the colour space
  (`rs::textureKeyOf`), and shared by every texture that is that image.
- `sceneBudgetMB` (0): the GPU memory the scene is fitted to, where less than the device's budget; 0 is that budget.
  A tenth of it stays free, and what DLSS would add to the frame. `Residency::fitQueuedToMemory`
  runs before each batch of decodes: idle decodes go first, then top levels, the largest first, down to 256, of the batch and
  of what is resident (decoded again, the slots rewritten, the frame reset). It reads free room as the driver's usage less the
  allocator's free space inside its blocks, as `gpuAllocatedMB` in the frame profile shows. A scene that grew is fitted with
  nothing queued too. `GltfRenderer::geometryShortfall` says when a load or wear won't fit even with every texture lowered, its
  meshes priced from their counts in the residency's stream formats and the driver's BLAS sizes (`scene_price.hpp`). What can
  come in smaller does; the rest comes all the same, with a notice. A mesh that finds no memory at all, on the device or in
  system memory, is parked (`Residency::setParked`): nothing of it resident, named in a notice, tried again when there is room.
  With a scene open it follows the budget both ways (see "Textures following GPU memory").
  What is resident is in `Residency::ledger`, by who reads it and per mesh: the `[memory]` log line and `render.ledger`.
  Past the budget nothing fails: the allocator's next choice of memory type is system memory, so an allocation takes
  100 ms and more and stays there, and rays read it over the bus (`MemoryLedger::spilled`, `scripts/memory-limit-check.py`).
- `residencyMinInstances` (256): the least instance capacity of the retained scene's tables and
  TLAS, which double from there. Start-up only (`--residencyMinInstances`), so not among the
  parameters a host can set (fv_parameter_filter.hpp).
- `residencyRetireOnReplace` (off), a debug setting read at each scene replace: the backend stays
  and syncs the newly fed scene over the old one instead of starting over, so every composition
  retires the old scene's slots and takes others, with holes between, capacity growth and deferred
  frees running on the GPU. Run one session with it on, under the validation layer in a debug
  build. The replace's device wait stays.

---

## Testing

Tests are disabled by default. Enable with `BUILD_TESTING=ON`:

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --target vk_gltf_renderer_tests
ctest --test-dir build -C Release --output-on-failure
```

See [tests/README.md](../tests/README.md) for details.

---

## Contributing

This project uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO). By contributing, you certify that you have the right to submit the work under the project's open source license.

Please open issues for bug reports and feature requests. Pull requests should target the `main` branch and include a clear description of the change.

### Line endings

All text files are stored in the repository with LF. This is enforced by `.gitattributes`
(`* text=auto`), with binary asset extensions pinned explicitly so they are never
EOL-rewritten.

`* text=auto` normalizes to LF on commit but sets no working-tree `eol` attribute, so what
you get on checkout is governed by your own Git configuration rather than by your platform:

| Setting | Working tree |
|---|---|
| `core.autocrlf=true` (Git for Windows default) | CRLF |
| `core.autocrlf=input` | LF |
| `core.autocrlf=false` | follows `core.eol`, default `native` — CRLF on Windows, LF elsewhere |
| `core.eol=lf` | LF, on any platform |

You do not need to change whichever you use: every combination round-trips cleanly, so
editing a file produces a diff of only the lines you actually changed.

The one-time renormalization that established this is listed in `.git-blame-ignore-revs`,
so `git blame` skips it (GitHub and GitLab apply that file automatically).

