# Denoising & Upscaling — DLSS, OptiX, OIDN, and Motion Vectors

> **For contributors and agents.** Conceptual map of the AI denoise/upscale backends and the
> guide-buffer + motion-vector pipeline they consume. Exact formats, flags, and struct fields live
> in the cited source — grep the symbols, don't trust values copied here.

## Backends

- **DLSS** — Ray Reconstruction (RR) and Super Resolution (SR): `src/dlss.{cpp,hpp}` (app adapter,
  jitter, guide buffers, UI) over `src/dlss_wrapper.{cpp,hpp}` (project-independent NGX/Vulkan wrapper).
- **OptiX AI Denoiser** — `src/optix_denoiser.{cpp,hpp}` (+ CUDA interop in `src/vk_cuda*`).
- **OIDN AI Denoiser** — `src/oidn_denoiser.{cpp,hpp}` over [vk_oidn](https://github.com/nvpro-samples/vk_oidn),
  which runs the Intel Open Image Denoise networks with Vulkan compute shaders. CMake downloads the
  prebuilt SDK package of the release pinned by `VK_OIDN_VERSION`; nothing of vk_oidn is built here.
- The **path tracer** (`renderer_pathtracer`) is the reference producer of the guide buffers; the
  **rasterizer** produces a subset for DLSS-SR.

## OptiX blend with the accumulation

The OptiX denoiser runs on the accumulated image, so the further accumulation has converged, the
less denoising it needs; at full strength it keeps smoothing away detail the samples already resolve
(skin pores, hair strands). **Blend at max frames** (`--optixBlendMax`, `OptiXDenoiser::Settings::blendMax`)
mixes the noisy accumulation back in through OptiX's `blendFactor`, rising linearly with the frame
count from 0 to that share at `ptMaxFrames`. Upscale 2X has no input at output size to blend with, so
it always shows the fully denoised result.

## DLSS startup

`Dlss::init` starts a nonblocking NGX prewarm as soon as the renderer is attached. This does not
require a loaded scene: the worker only initializes NGX and probes feature availability. The actual
DLSS feature handle and sized render targets are still created later from `Dlss::updateSize`, where
a command buffer and display extent are available. Disabled DLSS-RR therefore reports **Ready** once
NGX is warmed, but its heavier guide buffers stay unsized until the user enables it. A feature create
that fails leaves DLSS in **Loading** (no frame evaluates) and is asked again on the next resize or
DLSS setting change, not every frame.

## DLSS-RR input resolution

DLSS-RR path-traces at an input resolution NGX recommends for the viewport, then upscales to it.
`Dlss::updateSizeRr` asks NGX for the optimal input size of the RR **Mode** (`--dlssRrQuality`,
`Dlss::Quality` without Off: 1 DLAA, 2 Quality, 3 Balanced, 4 Performance, 5 Ultra Performance)
and creates the feature at that same mode. **Quality**, the default, traces about 67% of the
width and height, so the path tracer traces fewer pixels with DLSS-RR on than off; that usually
more than pays for the DLSS pass itself. **DLAA** traces at native resolution.

The DLSS-SR quality setting (`--dlssQuality`) is unrelated: it only sizes the rasterizer's Super
Resolution input. The DLSS-RR panel shows the resolution actually in use
("Input resolution"), and each resize is logged as `DLSS-RR: input WxH for output WxH`. Changing
the mode from any source (UI, command line, benchmark sequence, MCP) recreates the feature and
resets its temporal history; `Dlss::needsRecreate` compares the setting with the mode the feature
was built with.

## Guide buffers

The path tracer writes all guide buffers in `processPixel` (`shaders/gltf_pathtrace.slang`, first-hit
captures around the `USE_DLSS_SHADER` / `USE_GUIDE_SHADER` blocks). They are declared as the
`OutputImage` enum (`shaders/shaderio.h`), bound to NGX inputs in `Dlss::setResourcesRr`
(`src/dlss.cpp`), and consumed in `DlssFeature::cmdEvaluate` (`src/dlss_wrapper.cpp`). Roles (grep
the enum for the authoritative list and formats):

| Guide (`OutputImage`) | Role | NGX input |
|---|---|---|
| `eResultImage` | Noisy 1-spp path-traced radiance (the image being denoised/upscaled) | `pInColor` |
| `eDlssAlbedo` | Diffuse albedo (base color; see [clear glass](#clear-glass-primary-surface-replacement)) | `pInDiffuseAlbedo` |
| `eDlssSpecAlbedo` | Specular albedo (`EnvBRDFApprox2` in `shaders/dlss_util.h`) | `pInSpecularAlbedo` |
| `eDlssNormalRoughness` | World-space normal + roughness, packed | `pInNormals` (and `pInRoughness` in packed mode) |
| `eDlssMotion` | Pixel-space motion vectors (see below) | `pInMotionVectors` |
| `eDlssDepth` | NDC depth (`z/w`, `1.0` on miss) | `pInDepth` |
| `eDlssSpecularHitDist` | Specular hit distance, captured at the first reflection bounce | `pInSpecularHitDistance` (optional) |

- **DLSS-SR** uses only color + depth + motion — the material guides above are RR-only.
- **OptiX and OIDN** share one shader path for their guides: `eDenoiserAlbedo` and
  `eDenoiserNormal` (world-space normal), written when `ePtDenoiserGuides` is set. The shader does
  not know which denoiser reads them: each owns its own two images, and the path tracer binds those
  of the denoiser on that frame to the two slots, which **alias** `eDlssAlbedo` and
  `eDlssSpecAlbedo` (OptiX and OIDN are exclusive, and neither writes guides on a DLSS frame).
  Unlike DLSS's, these guides are **accumulated** like the color, averaging every sample of each
  frame, so their edges are anti-aliased (and blurred by depth of field) the same way, as Open Image
  Denoise requires and OptiX's noise-free guides allow. They keep a sample count of their own
  (`PathtracePushConstant::guideSamples`, from `GuideAccumulation` in `src/denoiser_schedule.hpp`):
  they restart when they were not written on the previous frame, for instance when the denoiser is
  turned on mid-accumulation, while the color goes on. Where nothing is hit, the albedo is the
  normalized background color, the way the vk_oidn sample treats emitters, and the normal is zero.
  OIDN reads them as they are; OptiX's copy pass (`optix_image_to_buffer.slang`) converts the normal
  to normalized camera space, (0,0,0) where nothing was hit, as the OptiX Programming Guide asks.
- Guides are produced at DLSS **render** resolution every frame while the denoiser is active; the
  main-GBuffer depth/selection are only written on frame 0 / after a reset.

## Accumulated-image denoisers (OptiX, OIDN)

OptiX and OIDN denoise the path tracer's accumulated image, not each frame. Both share the same
scheduling: **Auto** runs each time the accumulation reaches a multiple of N samples per pixel
(`AutoDenoiseSchedule`, `src/denoiser_schedule.hpp`), so the same interval gives the same image
quality whatever the samples per frame, and once more on the final image when the accumulation
reaches **Max Samples**, so the image left on screen is denoised from all its samples. **Denoise
Now** runs on demand, and the result is shown through the `DisplayBuffer` override of the
tonemapper input (`GltfRenderer::runTonemapPass`). With DLSS-RR also
engaged, DLSS owns frame 0 and the accumulated-image denoisers take over from frame 1
(`PathTracer::getEffectiveDlssEnabled` and its OptiX / OIDN counterparts), blending from DLSS's
image: see [DLSS hand-off](#dlss-hand-off).

Only one of them is on at a time (`PathTracer::keepOneAccumulationDenoiser`): both would denoise
the same image, and only one result can be shown. Turning one on turns the other off; when both come
on together (ImGui.ini, the command line), OIDN stays on.

They differ in where the work runs:

- **OptiX** runs on CUDA. `OptiXDenoiser::denoiseOneShot` waits for the queue, copies the images
  into exported buffers, denoises, and copies the result back, in temporary command buffers. It
  therefore always denoises the *previous* frame's accumulation. The frame that completes the
  accumulation is the last one the path tracer renders, so its image is denoised on the first idle
  frame after it (`PathTracer::onAccumulationComplete`); a headless run needs one frame past the cap
  for it. None of this is in the frame's command buffer, so a Vulkan GPU timer cannot see it. The
  Profiler window's *Optix denoiser* row therefore takes its GPU time from CUDA events around the
  invoke (the Vulkan copies excluded), comparable to OIDN's row; its CPU time is the host wall time,
  the wait for the frames in flight included. `OptiXDenoiser::getTimings` breaks the wall time down
  (queue drain, copies, invoke), shown under the OptiX settings.
- **OIDN** records `vkoidn::Denoiser::cmdDenoise` in the frame's own command buffer, right after the
  path tracer (`OidnDenoiser::updateDenoiser`), reading the storage images directly. A frame is
  denoised in that same frame, without host waits, so an interval of 1 denoises interactively, and
  the frame that reaches the cap is denoised in that frame too. Like OptiX's, its Profiler row
  (*OIDN denoiser*) is a frame section present on every frame OIDN is on, empty on the frames that
  do not denoise; the GPU time in its settings and in `DENOISER_TIMING` keeps only the frames that
  did (`OidnDenoiser::readGpuTime`). The network is created per size,
  quality and guide set (`OidnDenoiser::ensureTargets`); creating it records the upload of its
  weights. vk_oidn's device extensions are requested by
  `OidnDenoiser::requestDeviceExtensions` from the device-selection callback in `src/main.cpp`, and
  its context picks its optional features (tensor cores through `VK_KHR_cooperative_matrix`) from
  the enabled extension list (`Resources::deviceExtensions`).

## DLSS hand-off

With DLSS-RR and OptiX or OIDN both on, every frame of a moving camera is a frame 0, denoised by
DLSS. Once the camera stops, the accumulation denoiser takes the frames after it, and its first
images, made of few samples, are worse than DLSS's. `DenoiserHandoff` (`src/denoiser_handoff.{hpp,cpp}`)
blends from one to the other instead of switching:

- **DLSS's image is kept.** Frame 1, before the path tracer overwrites `eImgRendered`, copies it into
  an image of the same format. This happens once per camera stop, never while moving.
  `PathTracer::m_dlssFrame0` says DLSS actually rendered frame 0 (not while it is still loading).
- **It is no sample of the accumulation.** Frame 1 restarts the sample count
  (`PathTracer::setupPushConstant`), whether or not the hand-off is on, so the accumulated image and
  its spp hold path-traced samples only. This also keeps OptiX's 2x upscale from reading DLSS's
  display-size image as its half-size input (`OptiXDenoiser::updateDenoiser`).
- **The frames after it run the regular shader.** DLSS's guide captures (`USE_DLSS_SHADER`) cost
  every sample, so they are compiled in only for the frames DLSS renders
  (`PathTracer::makeVariantKey`). Each camera start and stop swaps the two cached variants in
  `PathTracer::prepareFrame`: handles move, nothing is destroyed, so there is no queue wait and no
  recompile (what a swap does evict is released through the app's frame ring once the frame in flight
  has passed). A moving camera stays on DLSS's. The first
  time a combination is used, the missing variant compiles behind the BusyWindow, which restarts the
  accumulation.
- **The image shown** is `lerp(dlss, denoised, handoffWeight(spp, end))` (`denoiser_schedule.hpp`),
  computed by `shaders/denoiser_handoff.slang`, with `end` the **DLSS Hand-off** setting
  (`dlssHandoffSpp`), capped at Max Samples so the final image is always the denoised one alone.
  The weight is a smoothstep over the samples per pixel: low while the denoised image is still
  blotchy, exactly 1 at `end`. It follows the accumulation every frame, not only when a denoise lands.
  Until the first denoise of the accumulation, DLSS's image is shown, past `end` too while Auto will
  make one (OptiX denoises one frame behind); from then on, the pass stops.
- **The denoises come sooner.** Until `end`, Auto denoises each time the samples per pixel double,
  and once at `end` (`AutoDenoiseSchedule::isDue`), then at its interval. Each doubling halves the
  noise variance, so each denoise is a visible step. The schedule stays in samples per pixel, not in
  time, so headless runs give the same images.
- **Shown in place of the denoiser's view.** The capture sets the `DisplayBuffer` to the denoiser's,
  and `runTonemapPass` takes the hand-off's image while `PathTracer::handoffImage` has one. Picking
  **Rendered**, turning the denoiser off, or a new accumulation ends the hand-off; without Auto there
  is none. `--outputSamples` saves nothing during the blend, which is no accumulation of a known
  count: the counts it skips are saved with the first image after it. Set `--dlssHandoffSpp 0` to
  measure the denoiser alone at low sample counts.

The blend image is RGBA16F, clamped to the half-float range (a sun disk exceeds it); the kept DLSS
image is RGBA32F, copied as is.

## Motion vectors — what is and isn't captured

Motion vectors are pixel-space (`InMVScale = 1`, low-res) and computed by `calculateMotionVector`
(`shaders/dlss_util.h`): `MV = (prevNDC - currNDC) * 0.5 * resolution`. Two sources of motion are
combined:

- **Camera motion** — reprojection through `SceneFrameInfo::prevMVP` vs `viewProjMatrix`
  (`shaders/shaderio.h`; both are unjittered and updated per frame in `GltfRenderer::onRender`,
  `src/renderer.cpp`). `prevMVP` holds the previous rendered frame's view-projection and is *not*
  cleared by `resetFrame()`, so a continuous camera move keeps correct camera MVs.
- **Instance (node) motion** — previous render-node transforms snapshotted by
  `snapshot_prev_transforms.comp.slang` (host: `TransformComputeVk::cmdSnapshotPrevObjectToWorld`)
  and applied via `prevRenderNodeObjectToWorld` in `shaders/gltf_pathtrace.slang`. Gated by
  `Resources::dlssInstanceMotionActive` (bound only on frames where transforms actually change) -
  this gate must see `Scene::DirtyFlags::nodes` as populated by *this* frame's transform writes,
  which is why the gate check in `GltfRenderer::onRender()` runs after `updateInteractivityGraphs()`
  (see [docs/interactivity.md](interactivity.md) for a real bug this ordering fixed - a graph-driven
  per-tick move ghosted for its entire duration when the check ran before the tick instead of after).

**Per-vertex skin/morph deformation (path tracer).** Skinning and morph overwrite the position
buffer in place (`AnimationVk::dispatchAnimation`; the CPU fallback is `SceneVk::uploadPrimitives`),
and `set_vertices` does the same. A bone pose is not instance motion: skinning writes the new
positions while the mesh node's transform stays put, so instance motion alone reprojects the
current shape under camera motion only, and Ray Reconstruction smears the surface while a pose, a
shape dial or a clip moves it.

`PrevPositionsVk` (`src/gltf_scene_prev_positions_vk.hpp`) keeps a copy of each deforming
primitive's positions, 12 bytes a vertex. Deforming here means skinned, morphed, edited live, or
named by an upload queued for this frame. The copy is made in
`GltfRenderer::snapshotPrevPositions` at the start of the frame, before the interactivity tick,
the host uploads and the animation reconcile write, and only when one of them ran the frame
before. When nothing wrote in between, the copy taken earlier still matches. On frames where a
writer ran, `dlssDeformMotionActive` binds `PathtracePushConstant::prevVertexPositions`, a table of
device addresses, one per render primitive.

`dlssPrevHitPos` interpolates the hit triangle's previous corners with the hit's barycentrics and
applies the previous node transform when that moved too, or the current one otherwise. It costs
three position reads on the primary guide and on the clear-glass replacement surface; no rays are
added. The live buffer is not swapped for a second one: the BLAS refit reuses build data holding
the fixed vertex address, and the shaders read positions through the scene description's vertex
pointers.

A dolly or orbit of a posed figure is camera motion and was already correct. The rasterizer lacks
this and also lacks the instance vectors the path tracer has: `vertexMain` in
`shaders/gltf_raster.slang` multiplies the current position by `prevMVP`.

## Clear glass: primary surface replacement

A camera ray that lands on clear glass sees *through* it, but by default every guide describes
the glass itself: a flat normal, near-white albedo, and the glass's own depth and motion. When the
camera moves, DLSS-RR reprojects the content behind the glass using the glass's motion vectors
and has no guide edges to hold it, so that content smears and ghosts. The more glass layers
there are (a box has two faces, nested boxes four), the worse it gets.

The RR **Transparency** setting (`Dlss::TransparencyMode`, `--dlssTransparency`, the
`USE_DLSS_TRANSP` specialization constant) selects **Improved** to fix this with primary surface
replacement (PSR). `dlssTraceClearGlass` (`shaders/pathtrace_functions.h.slang`) follows the specular
transmission direction through every clear-glass layer (`dlssIsClearGlass`). It goes straight
through thin-walled glass, as the BSDF does, and refracts through glass with a volume. It stops at the
first surface that is not clear glass, which then provides the albedo (mixed with the glass's Fresnel share),
normal, and roughness guides. The DLSS depth and motion vectors use a virtual point along the
camera ray at the full chain length, moved by that surface's instance motion, so they follow
the content rather than the glass. The main depth buffer (`outDepth`, used for picking and
overlays) still holds the glass.

The chain is deterministic, unlike the path. Stochastic alpha in the guides makes them flicker
frame to frame, which RR turns into speckle. Instead, `dlssTraceSkipThinGlass` resolves alpha with a
fixed threshold: cut-out texels are skipped, and an alpha-blend texel counts as a surface from half
opacity.

Cost: each traversal rejects thin clear glass inside the ray query itself, using only the material
factors (`dlssClearGlassFromFactors`), so a stack of glass panes costs one ray. Another traversal
starts only at refracting glass, or at glass whose clarity depends on a texture. The surface found
is shaded once. The extra work is paid only by pixels whose primary hit is clear glass. Measure it
with the path-trace GPU timer (see [benchmarking.md](benchmarking.md)); the cost grows with
glass screen coverage.

The guides must agree with the noisy color. The path spends one bounce per glass surface, so
when the chain crosses as many glass surfaces as `--ptMaxDepth` allows, the path never reached the
surface behind and the pixel is dark. That pixel keeps the glass's own guides; otherwise RR gets a
bright albedo against a black color and tints the result.

What PSR does not capture: the glass's own reflections and refraction edges are lighting with no
guide edge, so they stay softer than the reference. Rough, tinted, or partly opaque glass is left
alone, because there the glass surface is what the camera sees.

## Stochastic alpha

Hair and lashes are `BLEND`. The path tracer keeps a hit when a uniform draw falls within the
opacity. `stochasticOpacityU` (`shaders/pathtrace_functions.h.slang`) supplies that draw to
`rahitMain` and the ray-query trace. Shadow rays draw only at a surface that bounds a medium,
since the rest of the ray depends on whether it crossed. A surface that only filters the light
passes its expected transmission, `(1 - opacity) + opacity * T` (`shadowCrossSurface`): exact, so a
lash's shadow is neither noisy in a still nor a frozen stipple on DLSS frames.

While DLSS denoises the frame (`PathtracerFlags::ePtUseDlss`) the draw is a hash of the hit,
`alphaHashU`. Its inputs are the render node, the triangle, and the two independent barycentrics
quantized onto the fixed grid `ALPHA_HASH_BINS`. The frame is not an input, and the path seed is not
advanced. That seed is `xxhash32(pixel, frameCount)`, and with DLSS `frameCount` is the Halton index,
so drawing from it re-rolls the same point every frame and Ray Reconstruction crawls the edge.

- Raw barycentric bits avalanche under the half-pixel jitter, which is why they are quantized.
- The grid belongs to the triangle, so a dolly does not reshuffle the pattern, and motion vectors
  can follow it.
- The render node, not the shared render primitive, salts the hash, so instances of one mesh cut
  independently.

Tried and dropped: hashed alpha testing (Wyman and McGuire 2017) on object-space cells one pixel
wide, with `BLEND` alpha sampled at the pixel's mip. On a hair figure it looked worse. Filtering
turned the maps' crisp 0/1 strands into partial opacity, and pixel-sized cells turned that into
per-pixel dither that RR kept as grain. The hair looked frizzy when still and shimmered no less
in motion. Point-sampled hair texels are mostly 0 or 1, which no hash changes, so the hair's
shimmer in motion has another cause. That cause is not yet found.

Every other frame draws from the path seed, which advances it. `USE_DLSS_SHADER` stays on for the
whole time DLSS is enabled, including frames where OptiX accumulates, so the runtime flag is the
gate. A frozen hash averages to 0 or 1 across those frames and never to the soft opacity.

Every test keeps a hit when the draw is below the opacity, and the draw is in [0, 1): the hash
keeps its top 24 bits and the path draw is clamped below 1. An opaque or mask-on texel is always
kept and a cut-out texel never is, for camera and shadow rays alike (the expected transmission
gives the same at opacity 1 and 0).

`dlssTraceSkipThinGlass` calls `alphaHashU` directly, so the clear-glass guides commit the cells
the color keeps rather than cutting at half opacity.

**Color-before-transparency guide.** NGX also accepts `pInColorBeforeTransparency` (a noisy color
snapshot before transparencies are composited), meant for raster pipelines with ghosting
particles. We prototyped it here by retracing pixels whose primary hit was transparent, with those
surfaces skipped. On top of PSR it added rays and a full-resolution RGBA32F buffer and slightly
increased the error, so it is not wired. A path tracer has no "before transparency" image to
snapshot.

## Sky / background

Primary misses use a point-at-infinity model (`w = 0`) so camera translation cancels and only
rotation moves the environment. See the `calculateMotionVector` overload comment in
`shaders/dlss_util.h`.

## Jitter & reset

- Halton jitter is applied as a ray subpixel offset (path tracer) or clip-space shift (raster),
  never baked into `viewProjMatrix` / `prevMVP`; NGX de-jitters via `InJitterOffset` (negated in
  `src/dlss_wrapper.cpp`).
- Temporal history is **intentionally not reset on camera motion.** `InReset` is only for
  discontinuities (resize, preset/quality change, re-enable, and — since the fix documented in
  [docs/interactivity.md](interactivity.md) — a material/light property write via
  `BaseRenderer::notifyDlssContentReset()`, called from `GltfRenderer::updateSceneChanges()`
  whenever `Scene::DirtyFlags::materials`/`lights` is non-empty) via `Dlss::notifyReset` /
  `m_forceResetUntilFrame`. Do not reset on dolly/pan — it discards accumulation and adds noise
  without fixing MV correctness. Node-transform-only changes are deliberately excluded from this
  reset — those are already correctly handled by instance motion vectors (see "Instance (node)
  motion" above); only *appearance* discontinuities (texture/material content changing at a
  stationary surface, which motion vectors can't describe) need a full history discard.
- **A drawn frame that skips `evaluate()` drops the history.** `GltfRenderer` counts the frames it
  draws (`Resources::drawnFrameCount`), and `Dlss::evaluate` resets when the drawn frame before it
  did not evaluate: DLSS was off, the other renderer drew, a pipeline compile skipped the frame, or
  OptiX took it. The camera or the content may have changed on that frame without NGX seeing it,
  and the motion vectors cover one frame only. This holds whichever path wrote `dlssEnable` (the
  checkbox, the command line, a benchmark, MCP), so a host that turns DLSS off while a still
  accumulates and back on at the next change gets the same reset as the checkbox. The cost is one
  softer frame after each gap. Frames that are not drawn, once `maxFrames` is reached, do not count,
  so a still with DLSS left on keeps its history when the camera moves again. Only the checkbox
  frees the RR inner GBuffer; a `dlssEnable` write from elsewhere keeps it, so a host that turns
  DLSS off for every still does not recreate the feature at the next move.
- **Switching DLSS off restarts the path tracer's accumulation.** DLSS frames count toward
  `maxFrames` and the accumulated-sample total like any other frame, and the result image holds the
  last 1-spp DLSS input. Every DLSS setting therefore resets the frame (the `resetAccumulation` hook
  `PathTracer::registerParameters` passes to `Dlss::registerParameters`), as do the path-tracer
  settings that change a frame's content. Without it, accumulation after a DLSS run averages new
  samples into that one noisy frame at the weight of every DLSS frame, or renders nothing once the
  count has reached `maxFrames`.

## Debugging motion vectors

In the DLSS panel, expand **Developer Guide Buffers** and select the **Motion** thumbnail
(`Dlss::buildGuideEntries`, `src/dlss.cpp`) to visualize the MV buffer while moving the camera.

---

## DLSS-NR (Neural Rendering)

DLSS-NR (`NVSDK_NGX_Feature_DLSSNR`) is a **display-resolution image enhancer** that runs
**after** DLSS-RR or DLSS-SR and **after tonemapping**. It is orthogonal to RR/SR: any instance
(path-tracer RR or rasterizer SR) can optionally run NR on top. Gated by the `USE_DLSSNR` CMake
option and the `#if defined(USE_DLSSNR)` guards in `src/dlss.{cpp,hpp}`.

### Why post-tonemap?

NR expects **LDR input** (0–1 range). Running it on the pre-tonemap HDR image
(`eImgRendered`, linear float) causes visible corruption on bright light sources — the
values blow out the internal normalisation. The correct pipeline is:

```text
path tracer / rasterizer
  → DLSS-RR or DLSS-SR evaluate    (→ eImgRendered)
  → tonemap()                       (→ eImgTonemapped, R8G8B8A8_UNORM)
  → Dlss::evaluateNr()              (reads & writes eImgTonemapped)
```

`evaluateNr()` is called from `GltfRenderer::onRender` (`src/renderer.cpp`) immediately after
`tonemap()`. Call sites: the path tracer and rasterizer both call `setNrImage(eImgTonemapped,…)`
at resize time (`renderer_pathtracer.cpp`, `renderer_rasterizer.cpp`).

### NR GBuffer

NR needs a **SFLOAT scratch** image to avoid format conversion artefacts inside the NGX temporal
buffers. `m_nrGBuffer` (R32G32B32A32_SFLOAT, display resolution) is allocated in
`Dlss::updateSizeNr`. After NR evaluates, `evaluateNr()` blits the SFLOAT scratch back to
`eImgTonemapped` (R8G8B8A8_UNORM) using `vkCmdBlitImage2KHR`, which handles the format
conversion automatically.

### Temporal stability — depth and motion vectors

Without depth and motion, NR re-computes local structure detection independently each frame,
producing a characteristic **flicker** between two local-structure states. `evaluateNr()` provides
`pInDepth` and `pInMVec` so NGX can stabilise across frames.

| Kind | Depth source | Motion source |
|---|---|---|
| RR (path tracer) | `eDlssDepth` colour attachment, R16_SFLOAT linearised ViewZ | `eDlssMotion` colour attachment, R16G16_SFLOAT |
| SR (rasterizer) | hardware depth attachment (reversed-Z), `VK_IMAGE_ASPECT_DEPTH_BIT` | `getSrImage(SrSlot::eMotion)` |

MV are in render-pixel space; NR operates at display resolution. `InMVecScaleX/Y` is set to
`displayWidth / renderWidth` (and `…Y`). `InDepthInverted = 0` for RR (linearised), `1` for SR
(reversed-Z).

### NR control parameters

`NrSettings` (nested in `Dlss`, `src/dlss.hpp`) exposes the per-session global knobs:

| Field | NGX param | Range |
|---|---|---|
| `intensity` | `InIntensity` | [0, 1] |
| `localToneStrength` | `InLocalToneStrength` | [0, 1] |
| `localStructureStrength` | `InLocalStructureStrength` | [0, 1] |
| `globalToneStrength` | `InGlobalToneStrength` | [0, 1] |
| `skinStructureStrength` | `InSkinStructureStrength` | [0, 1] |
| `style` | `InStyle` | SDK enum |
| `useAutoMask` | `InUseAutoMask` | bool |

These are edited in the DLSS panel's NR settings pane and apply globally to the entire image.

### Per-material NR control mask — `EXT_DLSS_NR`

`pInControlMask` is a 4-channel image passed to NGX that **multiplies** the global strength
values per pixel:

| Channel | Meaning | Default |
|---|---|---|
| R | intensity mask | 1.0 |
| G | local tone strength mask | 1.0 |
| B | local structure strength mask | 1.0 |
| A | global tone strength mask | 1.0 |

`(1, 1, 1, 1)` is a no-op (global values pass through unchanged). Setting a channel to 0 disables
that effect for the covered pixels. This enables per-character / per-object NR tuning — e.g.
suppress local structure on a background material while keeping it on a foreground character.

This is exposed as the `EXT_DLSS_NR` glTF material extension. The full pipeline:

1. **Material data** — `EXT_DLSS_NR.nrMask` (vec4, default (1,1,1,1)) stored via
   `getExtDlssNr` / `setExtDlssNr` (`src/tinygltf_utils.{hpp,cpp}`). Loaded into
   `GltfShadeMaterial.nrMask` by `gltf_material_cache.cpp` under `#if MAT_EXT_DLSS_NR`.
   Gated by `MAT_EXT_DLSS_NR` in `shaders/gltf_material_config.h` (defaults to 1).

2. **GBuffer slot** — `OutputImage::eNrMask` (index 8, R16G16B16A16_SFLOAT) is the 9th colour
   attachment in the RR inner GBuffer. The path tracer writes it in `processPixel`
   (`shaders/gltf_pathtrace.slang`) on the first-hit block under `USE_DLSS_SHADER &&
   MAT_EXT_DLSS_NR`, reading directly from `materials[materialIndex].nrMask`. The slot is
   registered in `kRrSlots[]` in `renderer_pathtracer.cpp` so it is wired into the `outImages`
   bindless array. The format is declared in `kRrInnerFormats[]` (`src/dlss.cpp`).

3. **Temporal accumulation** — the mask is captured in `GuideScratch.nrMask` (first-hit only,
   no per-bounce accumulation) and copied to `GuideOutput.nrMask` by `packSampleResult`
   (`shaders/pathtrace_functions.h.slang`). Both structs are guarded by `USE_DLSS_SHADER`.

4. **NGX wiring** — `Dlss::evaluateNr()` reads `eNrMask` directly from `m_innerGBuffer` (Kind::RR
   only) and passes it as `pInControlMask` with `InControlMaskSubrectSize` at render resolution.
   When a mask is present, `InUseAutoMask = 0`; when absent (Kind::SR or path tracer not active),
   it falls back to the `useAutoMask` setting.

5. **UI** — `UiInspector::materialDlssNr()` (`src/ui_inspector.cpp`) renders four `SliderFloat`
   controls (Intensity / Local Tone / Local Structure / Global Tone) via
   `renderMaterialExtensionSection()`, which provides the Add / Remove scaffolding automatically.
   The extension appears in the **Material Extensions** section of the inspector.

6. **glTF support** — `"EXT_DLSS_NR"` is declared in `m_supportedExtensions`
   (`src/gltf_scene.cpp`) so the loader preserves the extension data on round-trip save/load.

> **Rasterizer note:** The rasterizer does not write `eNrMask` (no 4th framebuffer attachment in
> the SR pass). For Kind::SR, `pInControlMask = nullptr` and `InUseAutoMask` follows the global
> `useAutoMask` setting. Per-material masking from `EXT_DLSS_NR` is a path-tracer-only feature.

### Adding a new NR parameter

If a future NGX SDK exposes a new per-pixel input (e.g. skin mask, emissive mask):

1. Add a field to `GltfShadeMaterial` (gated by a new `MAT_EXT_*` flag in
   `gltf_material_config.h`).
2. Add a `GBuffer` slot to `OutputImage` (`shaders/shaderio.h`) and the matching format to
   `kRrInnerFormats[]` (`src/dlss.cpp`).
3. Write the slot in the path tracer first-hit block (`shaders/gltf_pathtrace.slang`), propagate
   through `GuideScratch` → `GuideOutput` → `packSampleResult` if needed.
4. Register the slot in `kRrSlots[]` (`src/renderer_pathtracer.cpp`).
5. Wire the NGX input in `Dlss::evaluateNr()` (`src/dlss.cpp`), mirroring the `pInControlMask`
   pattern.
6. Add tinygltf helpers, material cache loading, glTF extension registration, and inspector UI
   following the `EXT_DLSS_NR` pattern.
