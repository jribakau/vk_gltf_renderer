# Denoising — DLSS Ray Reconstruction, Moving and Still, and Motion Vectors

> **For contributors and agents.** Conceptual map of the AI denoise/upscale backends and the
> guide-buffer + motion-vector pipeline they consume. Exact formats, flags, and struct fields live
> in the cited source — grep the symbols, don't trust values copied here.

## Backends

- **DLSS** — Ray Reconstruction (RR) and Super Resolution (SR): `src/dlss.{cpp,hpp}` (app adapter,
  jitter, guide buffers, UI) over `src/dlss_wrapper.{cpp,hpp}` (project-independent NGX/Vulkan wrapper).
  RR denoises the moving view, one sample per pixel a frame.
- **DLSS-RR's still** — `src/rr_still_denoiser.{cpp,hpp}`: RR as the denoiser of the accumulated still
  ([below](#dlss-rr-as-the-stills-denoiser)), the only one since 2026-11-09. OptiX and OIDN, which
  denoised the still before it, were taken out then (see [history](#the-denoisers-before-it)).
- The **path tracer** (`renderer_pathtracer`) produces the guide buffers.

## Blend with the accumulation (Keep detail)

The still's denoiser runs on the accumulated image, so the further the accumulation has converged,
the less denoising it needs; at full strength it keeps smoothing away detail the samples already
resolve (skin pores, hair strands). **Keep detail** mixes the noisy accumulation back in, rising
linearly with the samples from 0 to that share at the cap (`sample_schedule::denoiseBlend`,
`--rrStillBlendMax`), by a pass over the denoised image (`ImageBlend` in `src/denoiser_handoff.hpp`,
the hand-off's weighted sum, written over the RGBA32F output without the hand-off's half-float clamp).

### The denoisers before it

Until 2026-11-09 the still could also be denoised by OptiX (on CUDA, through copies to shared
buffers) or OIDN (Intel Open Image Denoise, run by Vulkan compute through the prebuilt vk_oidn
package). OIDN was the default from 2026-11-07: on 1TEST, a night club lit by candles and bulbs behind
glass, it left 8-17% less colourful noise than OptiX and denoised in 10.5 ms of GPU against OptiX's
64 ms with its copies. Ray reconstruction replaced it as the default the same day, the user finding
that the skin and the wallpaper keep their texture where OIDN smooths them, and the two were then
removed: the app runs on NVIDIA RTX cards only, which all run RR. The noise on 1TEST is fireflies of
caustics (0.37% of the pixels, 55% of the raw error), which no network removed: docs/improvements.md
in the app, *Checked and rejected*.

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
the mode from any source (the app, command line, MCP) recreates the feature and
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
| `eDlssSpecAlbedo` | Specular albedo (`dlssSpecularAlbedo` in `shaders/pathtrace_functions.h.slang`: `EnvBRDFApprox2` from `shaders/dlss_util.h`, a named metal's edge tint taken off it by `dlssEdgeTintShare`) | `pInSpecularAlbedo` |
| `eDlssNormalRoughness` | World-space normal + roughness, packed | `pInNormals` (and `pInRoughness` in packed mode) |
| `eDlssMotion` | Pixel-space motion vectors (see below) | `pInMotionVectors` |
| `eDlssDepth` | NDC depth (`z/w`, `1.0` on miss) | `pInDepth` |
| `eDlssSpecularHitDist` | Specular hit distance, captured at the first reflection bounce; on clear glass and water, a mirror ray's on every DLSS frame (`dlssTraceMirror`); on an eye's tear film, half the film's radius behind it on every sample (`shaders/mirror_image.h.slang`), so the catchlight goes with the eye | `pInSpecularHitDistance` (optional) |
| `eDlssSpecularMotion` | The motion of reflections, every pixel: of a reflection's image, or the surface's own where there is none (see [below](#the-motion-of-reflections)) | `pInMotionVectorsReflections` (optional) |

- **DLSS-SR** uses only color + depth + motion — the material guides above are RR-only.
- **The still's guides** (DLSS-RR's still): `eDenoiserAlbedo` and `eDenoiserNormal` (world-space
  normal), written when `ePtDenoiserGuides` is set, plus those `ePtRrStillGuides` adds
  ([below](#dlss-rr-as-the-stills-denoiser)). The still denoiser owns the images, and the path tracer
  binds them to slots that **alias** `eDlssAlbedo` and `eDlssSpecAlbedo`, never on a DLSS frame.
  Unlike the moving view's, these guides are **accumulated** like the color, averaging every sample of
  each frame, so their edges are anti-aliased (and blurred by depth of field) the same way. They keep a
  sample count of their own (`PathtracePushConstant::guideSamples`, from `GuideAccumulation` in
  `src/denoiser_schedule.hpp`): they restart when they were not written on the previous frame, for
  instance when the denoiser is turned on mid-accumulation, while the color goes on. Where nothing is
  hit, the albedo is the normalized background color and the normal is zero.
- Guides are produced at DLSS **render** resolution every frame while the denoiser is active; the
  main-GBuffer depth/selection are only written on frame 0 / after a reset.

## DLSS-RR as the still's denoiser

`RrStillDenoiser` (`src/rr_still_denoiser.{hpp,cpp}`, `--rrStillEnable`; the app's Render tab ›
Denoise) makes DLSS-RR the still's denoiser. It denoises the path tracer's accumulated image, not each
frame: **Auto** evaluates it each time the accumulation reaches a multiple of N samples per pixel
(`AutoDenoiseSchedule`, `src/denoiser_schedule.hpp`; `--rrStillAutoDenoiseSpp`, the app sends 8), so
the same interval gives the same image quality whatever the samples per frame. The result is shown
through the `DisplayBuffer` override of the tonemapper input (`GltfRenderer::runTonemapPass`). With
DLSS-RR on for the moving view, DLSS owns frame 0 and the still's denoiser takes over from frame 1
(`PathTracer::getEffectiveRrStillEnabled`); DLSS's last moving image stays until the first
evaluation and blends into it (see [DLSS hand-off](#dlss-hand-off)). It also evaluates on the frame
that completes the still, or on the idle frame after it when the adaptive stop ended it early
(`PathTracer::onAccumulationComplete`).
With an interval of 0 only that final image is evaluated, and the refine shows the raw
accumulation. It is its own NGX feature, made in the moving view's NGX context
(`Dlss::ngxContext()`) at DLAA, the input the accumulation at the output's size; the moving view's
feature is left alone. Its images and feature exist only while it is on.

The guides are RR's own, averaged over the still's samples like the colour, so their edges are
anti-aliased as the colour's are: its frames run the DLSS shader variant without `ePtUseDlss`
(nothing of the lighting changes: that variant's other code only writes guides or is gated on the
flag) and `ePtRrStillGuides` makes the guide accumulation add RR's specular albedo, linear roughness
(the normal's w, packed) and the nearest guide depth to the diffuse albedo and normal the others
take; through clear glass they describe the surface behind it, as on the moving view. Motion is
zero and the history reset: one evaluation (`--rrStillPasses` evaluates the same image again,
building history on it). Keep detail mixes the raw accumulation back in
([above](#blend-with-the-accumulation-keep-detail)). The first still compiles the DLSS variant for the
still's settings once.

It turning on or off restarts the still (`PathTracer::pollStillDenoiser`), whether from the setting
or from NGX coming up, which an idle still's frames take with `Dlss::pollInit()` since they do not
tick DLSS. Without the restart the guides would cover only the samples after it, and the pass
would skip. A headless run given `--ptMaxSamples` holds its exit while RR's still is wanted and
NGX is still starting, or its final pass has not run (`PathTracer::stillDenoisePending`). A small
scene's still otherwise ends before NGX's async start does, and saves the raw image.

Judged on a dim room lit by candles (11-07), against OIDN: the skin and the wallpaper keep their
texture, where OIDN smooths them.

## DLSS hand-off

With DLSS-RR and its still both on, every frame of a moving camera is a frame 0, denoised by
DLSS. Once the camera stops, the still's denoiser takes the frames after it, and its first
images, made of few samples, are worse than DLSS's. `DenoiserHandoff` (`src/denoiser_handoff.{hpp,cpp}`)
blends from one to the other instead of switching:

- **DLSS's image is kept.** Frame 1, before the path tracer overwrites `eImgRendered`, copies it into
  an image of the same format. This happens once per camera stop, never while moving.
  `PathTracer::m_dlssFrame0` says DLSS actually rendered frame 0 (not while it is still loading).
- **It is no sample of the accumulation.** Frame 1 restarts the sample count
  (`PathTracer::setupPushConstant`), whether or not the hand-off is on, so the accumulated image and
  its spp hold path-traced samples only.
- **The frames after it run the shader they need.** DLSS's guide captures (`USE_DLSS_SHADER`) cost
  every sample, so they are compiled in only for the frames that use them: DLSS's, and the still's
  frames when its denoiser is RR, which gathers RR's guides; a still without a denoiser runs the
  regular variant (`PathTracer::makeVariantKey`). Each switch between them swaps the cached variants in
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
  make one; from then on, the pass stops.
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
- **Instance (node) motion** — an instance keeps the world it was drawn with for one sync after it
  moves (`SceneGpuView::prevObjectToWorldTable`), applied via `prevRenderNodeObjectToWorld` in
  `shaders/gltf_pathtrace.slang`. Gated by `Resources::dlssInstanceMotionActive`, set after the
  frame's sync: a per-tick move ghosted for its entire duration when the check ran first.

**Per-vertex skin/morph deformation (path tracer).** The skin and morph passes write a drawer's
output positions, and a vertex edit writes a mesh's. A bone pose is not instance motion: skinning writes the new
positions while the mesh node's transform stays put, so instance motion alone reprojects the
current shape under camera motion only, and Ray Reconstruction smears the surface while a pose, a
shape dial or a clip moves it.

The residency keeps a copy of those positions, 12 bytes a vertex, only while they move: made just
before a sync writes them (`SceneGpuView::prevPositionTable`). At the first sync that doesn't, the row
points at the positions themselves, which are the same numbers, so a surface at rest has zero motion
from the same reads and a still's guide doesn't change; the copy goes 120 frames later. On frames where a
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

### The motion of reflections

RR moves a pixel's reflection by the motion it is given in `pInMotionVectorsReflections`
(`eDlssSpecularMotion`); without it, by its own guess from the specular hit distance and the camera
matrices (NVIDIA's RR guide, 3.4.8). It takes the field instead of its guess, so the field covers every
pixel, in pixels like the motion vectors:

- **Where the pixel shows a reflection** (a hit distance was captured): the reflection's image, the hit
  distance beyond the first hit along the camera ray (`mirrorImagePoint`, `shaders/mirror_image.h.slang`),
  which is where a flat mirror shows the point and where RR itself puts it. That image moves by
  `GuideOutput::specularDelta`: the reflected point's own motion (`dlssPrevHitPos`) mirrored in the
  first hit's plane, plus twice the first hit's motion along its normal (`mirrorImageDelta`). In a still
  scene this is RR's own guess. The difference is a moving figure seen in a mirror, a window or paint,
  or a moving mirror. A turning mirror is taken as moving only.
- **The sky in a reflection** (65504): the image at infinity along the camera ray, moving as the sky
  does.
- **An eye's tear film:** its image, half the film's radius behind it, moves with the eye.
- **No reflection** (a matte pixel, or a sample whose first bounce was diffuse): the surface's own
  motion, which is RR's guess for a hit distance of 0. On glossy surfaces a pixel switches between the
  two from frame to frame, as its hit distance already came and went with the sample's lobe.

**Clear glass and water** get their reflection on every DLSS frame: a mirror ray off the first hit's
shading normal (`dlssTraceMirror`, `shaders/pathtrace_functions.h.slang`), past thin panes and through
alpha as the see-through chain goes (`dlssTraceSkipThinGlass`), with the floor where it is nearer. The
path itself reflects there on a few percent of its samples (the Fresnel), so the hit distance over a
window or water was mostly missing. The image is counted from the glass, where the reflection is,
whatever depth primary surface replacement gives RR. What a surface behind the glass reflects in turn
(a metal behind a window) moves with the glass's reflection in this field: one field per pixel.

Cost: one ray a pixel where the first hit is clear glass or water, on DLSS frames only; eyes are left
out, their tear film's image needs no ray. Checked by `MirrorImage.AFlatMirrorsImageLiesAlongTheView`
and `MirrorImage.AnImageMovesAsWhatItReflectsAndItsMirror`.

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
specular albedo, normal, and roughness guides. That surface is described as it would be seen directly
(`dlssSpecularAlbedo`, its diffuse albedo less its metal on DLSS frames), so a metal or a gloss behind a
window keeps its reflection's colour instead of reading as a texture. Its specular albedo is added to
the glass's own, less the glass's Fresnel share, not blended with it: a blend would leave a matte surface
behind glass a specular albedo of about F squared against the glass's reflection of F. Under the water
(Ground = Water) the floor is the ground's colour, metal and roughness. The DLSS depth and motion vectors use a virtual point along the
camera ray where the eye sees that surface, moved by its instance motion, so they follow
the content rather than the glass. Through thin glass that is the chain's length. Under the water
and through solid glass the surface is seen nearer: through flat surfaces, a point s down a ray
refracted from index n0 into n is seen s n0 / n along the camera ray, the image that holds as the
eye moves across, on the line through the point along the normal (`dlssSeenScale`,
`DlssGlassPsr::seenLength`; the floor 50 cm under the water looks 38 cm deep head-on, nearer at a
slant). Something moving in the water is seen moving along the vertical by the same squeeze
(`dlssSeenDepthRatio`). From under the water, what is above is seen farther. Waves form another
image, which this leaves out, as Falcor does; an eye's iris, under 1 mm behind its cornea, keeps its
straight depth. The main depth buffer (`outDepth`, used for picking and overlays) still holds the
glass.

The surface behind comes as it comes through (`DlssGlassPsr::through`): its albedo and specular albedo
are multiplied by every glass on the way, each crossing by its base colour (as the BSDF tints
transmission, a pane's deepened at the slant it is crossed at, `dlssCrossingTint`), by a
volume's absorption over the segment inside it, and by the water's (`GROUND_WATER_ABSORPTION`) down
to a mesh or the floor under it. A thin pane skipped inside the traversal tints by its base colour
factor at the slant of its face (`dlssPaneTint`), a texture on it not looked up, and counts only if it lies
in front of the nearest hit: the first `DLSS_PANE_SLOTS` panes are kept with their distance until the
traversal ends. So a tinted bottle at a bar, or a car's tinted window, tells RR the darker colour the
pixel holds.

Where the glass's own reflection outweighs what comes through, the guides stay the glass's
(`dlssReflectionShare`, `shaders/dlss_specular_guide.h.slang`): with F the reflection and T what
comes through, the share F / (F + (1 - F) luminance(T)) past `DLSS_REFLECTION_KEEPS_SURFACE` (0.5)
keeps the glass's normal, depth and motion, which the reflection moves with (NVIDIA's RR guide, 3.4.1:
prefer the reflection at the primary hit). Water switches at about 82 degrees from the vertical, a car
window of 0.8 a little past 80, a pane too dark to see through (0.02) head-on. The roughness guide
follows the share. The switch is an edge in the guides: if a line shows across the water from a
low camera, this is it.

The chain is deterministic, unlike the path. Stochastic alpha in the guides makes them flicker
frame to frame, which RR turns into speckle. Instead, `dlssTraceSkipThinGlass` resolves alpha with a
fixed threshold: cut-out texels are skipped, and an alpha-blend texel counts as a surface from half
opacity.

Cost: each traversal rejects thin clear glass inside the ray query itself, using only the material
factors (`dlssClearGlassFromFactors`), so a stack of glass panes costs one ray. Another traversal
starts only at refracting glass, or at glass whose clarity depends on a texture. The surface found
is shaded once. The extra work is paid only by pixels whose primary hit is clear glass. Measure it
with the path-trace GPU timer; the cost grows with
glass screen coverage.

The guides must agree with the noisy color. Clear glass spends a budget of its own, not surface
bounces (`ptGlassBounces`, `PathtracePushConstant::glassBudget`), so when the chain crosses more glass
surfaces than that allows, the path never reached the surface behind and the pixel is dark. That pixel
keeps the glass's own guides; otherwise RR gets a bright albedo against a black color and tints the result.

What PSR does not capture: the glass's own reflections and refraction edges are lighting with no
guide edge, so they stay softer than the reference. Rough or partly opaque glass is left alone, because
there the glass surface is what the camera sees; tinted glass is replaced, with its tint in the guides
(above).

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
whole time DLSS is enabled, including the frames of DLSS-RR's still, so the runtime flag is the
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
  discontinuities (resize, preset/quality change, re-enable, and a material/light property write via
  `BaseRenderer::notifyDlssContentReset()`, after a sync that says so (`SyncResult::dlssReset`)) via `Dlss::notifyReset` /
  `m_forceResetUntilFrame`. Do not reset on dolly/pan — it discards accumulation and adds noise
  without fixing MV correctness. Node-transform-only changes are deliberately excluded from this
  reset — those are already correctly handled by instance motion vectors (see "Instance (node)
  motion" above); only *appearance* discontinuities (texture/material content changing at a
  stationary surface, which motion vectors can't describe) need a full history discard.
- **A drawn frame that skips `evaluate()` drops the history.** `GltfRenderer` counts the frames it
  draws (`Resources::drawnFrameCount`), and `Dlss::evaluate` resets when the drawn frame before it
  did not evaluate: DLSS was off, a pipeline compile skipped the frame, or the still's denoiser
  took it. The camera or the content may have changed on that frame without NGX seeing it,
  and the motion vectors cover one frame only. This holds whichever path wrote `dlssEnable` (the
  checkbox, the command line, a benchmark, MCP), so a host that turns DLSS off while a still
  accumulates and back on at the next change gets the same reset as the checkbox. The cost is one
  softer frame after each gap. Frames that are not drawn, once `maxFrames` is reached, do not count,
  so a still with DLSS left on keeps its history when the camera moves again. Only the checkbox
  frees the RR inner GBuffer; a `dlssEnable` write from elsewhere keeps it, so a host that turns
  DLSS off for every still does not recreate the feature at the next move.
- **A DLSS frame's samples share its jitter.** `ptSamplesDlss` (1 by default) is how many samples
  a DLSS frame traces, all through the frame's one jittered position: the guides, written from the
  last sample, describe that point, so the extra samples take noise out of its light only. Under
  Ray Reconstruction a still's frame of 1 sample looked no better than a moving one; the app sends
  more while a still refines with DLSS (Render › Refine with DLSS › Samples a frame, 4 by default)
  and 1 when the view moves, with no reset, so RR keeps its history. They count toward Max Samples:
  128 at 4 a frame end after 32 frames. The renderer traces them only once the view has held still
  for 250 ms, and only as many as fit 33 ms of path trace at the last measured cost of a sample
  (`sample_schedule::dlssFrameSamples`): at first they were traced from the app's word alone, so
  the frame a drag landed on was one of 4 samples, and so were the one or two after it until the
  app's 1 arrived, and the drag's start lagged (asked 11-06).
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
   bindless array. The format is declared in `kRrInnerFormats[]` (`src/dlss.cpp`). The write
   happens only while DLSS-NR runs (`ePtWriteNrMask`, from `Dlss::isNrActive()`). A build without
   `USE_DLSSNR` doesn't allocate the slot (`VK_FORMAT_UNDEFINED`), and `Dlss::getRrAttachment` binds
   the specular albedo's image there, which the shader then never writes through it.

3. **Temporal accumulation** — the mask is captured in `GuideScratch.nrMask` (first-hit only,
   no per-bounce accumulation) and copied to `GuideOutput.nrMask` by `packSampleResult`
   (`shaders/pathtrace_functions.h.slang`). Both structs are guarded by `USE_DLSS_SHADER`.

4. **NGX wiring** — `Dlss::evaluateNr()` reads `eNrMask` directly from `m_innerGBuffer` (Kind::RR
   only) and passes it as `pInControlMask` with `InControlMaskSubrectSize` at render resolution.
   When a mask is present, `InUseAutoMask = 0`; when absent (Kind::SR or path tracer not active),
   it falls back to the `useAutoMask` setting.

5. **glTF support** — the extension's data goes through the material payload
   (`figure/materials/material_gltf.cpp`).

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
6. Add tinygltf helpers, material cache loading and glTF extension registration, following the `EXT_DLSS_NR` pattern.
