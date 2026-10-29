# Path tracer cleanup notes

Review of `src/renderer_pathtracer.cpp` and `src/renderer_pathtracer.hpp`, 2026-10-29, with
`src/renderer.cpp` and `shaders/gltf_pathtrace.slang` added the same day (their sections are at the
end). Nothing major: details to go through later.

## Summary

A full read found nothing worth optimizing on the per-frame path and six duplications in the
cold pipeline and variant code. The host work in `onRender` is a few compares, a handful of
uncontended mutex locks, one push-descriptor write and one push constant, against a GPU frame of
tens of milliseconds. The items below are maintenance, not speed.

## Duplicated code

All of these are in code that runs at startup, on a setting change, or on a variant swap. None
runs per ray or per pixel.

1. **Three copies of the pipeline-creation flow.** The anonymous helper `createPathTracePipeline`
   builds the specialization constants, the compute create info and the RTX create info.
   `createRqPipeline` and `createRtxPipeline` rebuild the same three things by hand, differing
   only in the chained `VkPipelineCreationFeedbackCreateInfo` and the deferred-op path, which
   `USE_DEFERRED_RTX_COMPILE` compiles out by default. Giving the helper an optional `pNext`
   would remove roughly 80 lines and guarantee the warm-up thread and the real creation ask the
   driver for the identical pipeline, which the comment on `RtxStages` says is the intent.

2. **The ray-query flags expression appears three times**: `startPipelineWarm`,
   `createRqPipeline` and `startVariantJob` each spell out
   `m_capturePipelineStats ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0`. RTX has an
   `rtxPipelineFlags()` helper; a matching `rqPipelineFlags()` closes that.

3. **`getEffectiveDlssEnabled` duplicates `dlssFrame()`.** It ignores its `resources`
   parameter and returns `m_dlssFrame`, which the header already exposes through `dlssFrame()`.
   Its four call sites (`onRender` twice, `pushDescriptorSet`, `setupPushConstant`) could read
   the member directly and the function could go.

4. **The DLSS-engaged test is written out twice** in `getEffectiveOptixEnabled` and
   `getEffectiveOidnEnabled` (`s == eLoading || s == eActive`) instead of calling
   `isDlssEnabled()`, which has the identical body in the header. The two functions are the same
   function with a different denoiser pointer and could share one private helper.

5. **Two copies of the LRU trim loop** on `m_variantCache`, in `swapVariant` and
   `installBuiltVariant`. `destroyVariantCache` and `onDetach` also hand-destroy the four handle
   types that `releaseVariant(..., gpuIdle = true)` already knows how to destroy.

6. **`makeVariantKey` is rebuilt repeatedly** within one frame and one compile. `prepareFrame`
   triggers it up to four times through `getPendingCompileWork`, `tryRestoreCachedVariant` and
   `buildsOffTheFrame`. `compileShader` computes `key` and then calls `isEmbeddedDlssVariant` and
   `isEmbeddedBaseVariant`, which each compute it again. Each build copies a `SceneFeatureSet`.
   Cheap, but the two `isEmbedded*` predicates would be cleaner taking the key as a parameter.

## Comment drift

The repository's guide asks for these to be fixed on sight.

- `m_rqPipeline` in the header is commented `// Ray tracing pipeline`. It is the ray-query
  compute pipeline.
- The comment above `getEffectiveDlssEnabled` in the .cpp describes a strict multi-condition
  check (NGX up, inner G-buffer populated, exclusivity rule) that no longer exists. The function
  returns a member decided at the frame's start.
- `updateStatistics` opens with two blank lines. The constructor sets `m_pushConst.frameCount`
  to zero, which value-initialization already does. Cosmetic only.

## Not worth doing

An earlier review proposed five changes. Checked against the code:

| Claim | Verdict |
| --- | --- |
| Replace `glm::inverse` of the projection matrix with `1 / proj[1][1]` for `pixelAngle` | Already done. There is no `glm::inverse` in the file; the code reads `[1][1]` directly with an epsilon guard. |
| Guard `std::log(sceneRadius)` against NaN | No bug. The radius is a non-negative length behind `hasContent()`. A zero radius gives `-inf`, and `std::max(-inf, 0.0f)` is zero. The ternary form is a style choice. |
| Fold the two float divisions in the throughput calculation | Once per frame, nanoseconds. The code also uses `m_frameSamples`, not `m_pushConst.numSamples` as quoted, so the suggestion was written against an older revision. |
| `glm::distance` instead of `glm::length(a - b)` | Identical after inlining; `glm::distance` is implemented as `length(p1 - p0)`. Cosmetic. |
| Drop the `frames > 0` guard on `sppPerFrame` as always true | Wrong. `GltfRenderer::resetFrame()` sets `frameCount` to -1, so `frames` can be zero when the UI draws between a reset and the next `onRender`. Without the guard the panel prints `inf spp/frame`. Keep it. |

Also not worth touching: the repeated `handoffEnd` and `handoffDenoiser` calls in `onRender`, the
multiple `getEffectiveOptixEnabled` and `getEffectiveOidnEnabled` calls per frame, the
`std::rotate` on `m_recentFrameSamples`, and the mutex locks in `getPendingCompileWork`. All are
nanoseconds against a GPU-bound frame.

## Suggested commit

One commit covering items 1 to 5 above plus the three comment fixes. Item 6 is optional. Verify
with a relink (`pnpm renderer:build`) and the gtest suite (`pnpm renderer:test`); no render is
needed since nothing changes what a frame draws.

## Architecture

The file is one class holding four unrelated lifetimes: per-frame rendering, pipeline and shader
builds on background threads, an on-disk cache, and denoiser policy. The direction is already
set: five helpers have been carved out of it (`sample_schedule`, `selection_write`,
`denoiser_handoff`, `variant_compile_job`, `pipeline_cache_util`). What follows is what is left,
in order of payoff against risk. The class is big because it coordinates a lot, not because the
coordination itself is wrong; moves 1 to 3 would take it from one 3200-line class to a
coordinator of about half that over three testable parts.

### 1. The SPIR-V disk cache is a pure module trapped in a Vulkan class

About 350 lines (the FNV hashing, `SourceStamp`, `spirvCacheKey`, `blobOfDeps`,
`pruneShaderCache`, `loadCachedSpirv`, `storeCachedSpirv`) depend on nothing in `PathTracer` but
`m_shaderCacheDir`. It is also the only part of the file with subtle failure modes: atomic
publish, torn reads, the timestamp settle window, LRU pruning across processes. None of that can
be unit tested today because it sits behind a class that needs a `VkDevice`. Pulling it into its
own `SpirvDiskCache` is a zero-behavior-change move that makes it testable under the existing
gtest suite. Highest value per unit of risk.

### 2. The variant cache moves five handles by hand

A live variant is five separate members (`m_shaderModule`, `m_rtxPipeline`, `m_rqPipeline`,
`m_sbtBuffer`, `m_sbtRegions`) plus `m_pipelineUseSER`, and `VariantCacheEntry` repeats the same
fields. Every ownership transfer (swap in, park, evict, release, teardown) is written as five
assignments. That is why `releaseVariant` takes four handles, why the LRU trim exists twice, and
why `onDetach` and `destroyVariantCache` destroy the same things by hand (items 5 and 6 under
"Duplicated code" are symptoms of this). One value type, say `PipelineSet`, with a single destroy
routine makes the live slot and a cache entry the same thing. `swapVariant` becomes a swap of two
values and the duplication disappears as a side effect rather than by cleanup.

### 3. OptiX and OIDN are the same thing with two names

Thirteen functions contain a paired OptiX block and OIDN block, and the .cpp has 69 denoiser
preprocessor branches. The two denoisers already present near-identical surfaces: enabled, guide
state, albedo and normal images, denoise count, auto-denoise, final denoise, UI. The one real
difference is that OptiX runs on CUDA outside the command buffer, which is an implementation
detail, not a policy one. An `AccumulationDenoiser` interface plus one "the active accumulation
denoiser" pointer would collapse most of those thirteen pairs to one line each, and the
exclusivity rule in `keepOneAccumulationDenoiser` becomes a pointer assignment. This is the
biggest cut but touches `optix_denoiser` and `oidn_denoiser` too, so schedule it rather than slip
it in.

### 4. The frame's decisions are spread across booleans and re-asked

`m_dlssFrame`, `m_prevDlssFrame`, `m_handoffFrame`, `m_dlssFrame0`, `m_dlssShaderFrame` and the
three `getEffective*` queries are consulted from `onRender`, `pushDescriptorSet`,
`setupPushConstant` and the hand-off. The DLSS decision was already moved to "decided once at
the frame's start"; the OptiX and OIDN decisions were not. A single per-frame plan value, built
once at the top of `onRender` and passed down (who renders, which technique, render size,
samples, which guides, whether the hand-off captures), makes the frame read as a straight line
and makes it impossible for two steps to disagree.

### 5. Three thread mechanisms build pipelines off the application thread

The compile thread with the busy window, the warm-up threads at startup
(`startPipelineWarm`), and the variant job (`VariantJobs`). The latter two are the same job:
build a pipeline from a module, layout and flags on another thread and hand back handles. They
already share `createPathTracePipeline`. One worker with a queue would serve both. Lower
priority: each works and they do not interact.

### Smaller observations

- The UI panel (`onUIRender`, about 260 lines) lives in the renderer file while the repo's
  convention is one panel per `ui_*` file. The rasterizer has the same shape, so this is a
  repo-wide choice rather than this file's fault.
- Encapsulation is better than the header suggests. Only `m_pushConst` and
  `m_performanceTarget` are touched from outside the class, so the public data section is mostly
  historical and could be closed.
- Not worth changing: the ray-query versus ray-tracing branches are small and local, and the
  adaptive sampler is already extracted.

## renderer.cpp

5900 lines for one class already split across four other files (`ui_renderer.cpp`,
`renderer_selection.cpp`, the two renderers). Nothing on the per-frame path is worth optimizing;
the duplication comes from copying sequences instead of naming them. Counts verified by grep.

### Duplicated sequences

1. **Scene load finalization exists three times.** `finalizeSceneSetup` is documented as the
   shared tail, but `createSceneFromDescriptor` repeats about 30 of its lines by hand (GPU build,
   animation strip, interactivity autoplay, feature set, firefly clamp, UI wiring, ground plane,
   cameras, textures with cleanup on failure, recent files). `rebuildVulkanSceneInternal` repeats
   the animation-strip and autoplay block a third time. The descriptor path could call
   `finalizeSceneSetup`.
2. **The GPU scene sync is written three times.** The split between the GPU transform path and the
   CPU path (sync materials and lights, flush the scene descriptor, upload staging, then either
   dispatch the transform compute or sync the TLAS) appears in `reconcileAnimationGpuState` and
   twice inside `updateSceneChanges`, once per branch. The GPU branch of `updateSceneChanges`
   inlines the body of the `updateSceneChanges_SyncGpuBuffers` helper that sits beside it. The
   barriers between the three copies differ subtly, which is where sync bugs come from.
3. **Load an HDR, select it, recalibrate** appears four times. `loadHdrEnvironment` already is that
   helper, and its own comment lists the other three sites.
4. **Scene construction boilerplate** (make a scene, register the WebP extension) appears four
   times. One `makeScene()` closes it.
5. **The threaded load pattern** (queue wait, clear the load pipeline, clean up, start the busy
   window, set the save filename, detach a thread, stop the busy window) appears three times across
   `addSceneFromFile` and `onFileDrop`, plus a synchronous fourth in `loadSceneFile`.
6. **Denoiser ladders again.** `dlssGuideRequired`, `displayedImageSamples`, `runTonemapPass` and
   `logDenoiserTimings` each hold a paired OptiX and OIDN block. Same root cause as architecture
   item 3 above.
7. **Dirty-flag preservation across a clear** is hand-written twice with different sets: five flags
   in `reconcileAnimationGpuState`, one in `updateSceneChanges_Finalize`. A `clearExcept(mask)` on
   the scene's dirty flags replaces both.

### To check, not confirmed

The three scene loaders run on detached threads capturing `this`, and nothing waits for them on
shutdown: `onDetach` waits for the device, not for the busy window, and the main loop has no busy
check. Closing the window during a load looks like it tears down `Resources` under the worker.
Not reproduced. If real, the fix is a joinable thread member or a wait on the busy window in
`onDetach`.

### Architecture

- **The constructor is a 560-line settings table** (tonemapper, sun, time of day, gradient and
  physical sky, atmosphere, retained scene, ground, shadows). The two renderers already register
  their own through `registerParameters`; the sky and atmosphere block belongs with `SkySun` or
  the environment panel, the tonemapper block with the tonemapper, the retained-scene block with
  the retained backend.
- **Eleven `m_pending*` members and eight `applyPending*` methods** implement one idea: work
  deferred to the frame top, each pair with its own ordering comment. A single ordered queue of
  frame-top actions makes the ordering explicit.
- **`onRender` is 390 lines** and builds the 60-field `SceneFrameInfo` inline. `buildFrameInfo()`
  is the obvious cut; the environment preview and commit block is a second.
- **`updateFrameCounter` keeps instance state in a function-local static** (`ref_camera`).
  Harmless with one renderer, wrong in principle; make it a member.
- **The one-line `updateSceneChanges_*` wrappers** (`_BlasRebuild`, `_RasterizerInvalidate`,
  `_TlasUpdate`) add indirection without hiding anything, while item 2 was left in place.

Not worth doing: the per-frame `glm::inverse` of the view and projection, the 256-step gradient
integral, the per-frame copy of the light list in `fillShadowAndGround`, the double
`benchmarkFrameInfo()` call.

## gltf_pathtrace.slang

Well-built where it matters: live state is trimmed deliberately, the shadow ray is deferred to
shrink registers across the trace, the hot paths are compile-time gated, and opaque instances skip
any-hit through the instance flags. The items below are per-vertex redundancies, dead code, and
one function that has outgrown itself.

### Per-vertex work that could be cheaper

Items 1, 3 and 4 are the ones to time with the MCP shader tool (`docs/mcp.md`); each is a one-line
change behind a flag.

1. **Light selection evaluates every punctual light twice per vertex.** `lightPickSet` loops over
   the lights computing each one's pick estimate, then `sampleLights` loops again recomputing the
   same estimates to walk to the chosen one. Bounded at `LIGHT_PICK_MAX_LIGHTS` (16), so up to 32
   estimate evaluations per vertex. Keep the first pass's estimates, or a running prefix.
2. **Sized-light hits recompute the pick set per hit light.** `sizedLightHits` runs once per
   segment on every bounce, loops over all lights with a hit test each, and for every light the
   ray hits calls `sizedLightNeePdf`, which calls `lightPickSet` again: a third pass over the
   lights per bounce when a sphere or disc light is in view.
3. **`getHitState` computes things most bounces never read.** The second shadow-terminator offset
   (`shadowPosOther`) exists only for a scatter walk's diffuse exit but is computed on every hit.
   Both texture-coordinate sets are fetched and interpolated whether or not the material uses the
   second. These are the per-bounce vertex loads.
4. **Any-hit copies the whole material.** `getOpacity` (`surface_opacity.h.slang`) copies the
   88-field `GltfShadeMaterial` per candidate, then reads seven fields. On a figure every lash card
   and hair ribbon is a non-opaque candidate. The compiler probably narrows the load; measure. The
   early return for an opaque material is dead in practice: opaque materials are forced opaque at
   the instance (`SceneRtx::updateInstanceFlagsCache`), so any-hit never runs for them.
5. **Small things.** The pixel color is divided by the sample count and multiplied back in the
   accumulation. `samplePixel` extracts the camera's forward, right and up by three matrix
   multiplies per sample; they belong in `SceneFrameInfo`. The barycentrics are rebuilt for
   `dlssPrevHitPos` after `getHitState` already had them.

### Dead code and drift

- `doDebug` is set from the mouse coordinate and read nowhere in any shader.
- The accumulation reads `outImages[0]` by literal while the write beside it uses `eResultImage`.
- Leftover commented calls: `//DirectLight directLight;`, `// bsdfEvaluateSimple`,
  `// bsdfSampleSimple`.
- `pathTraceOneBounce` opens an extra brace scope around its entire body, with the closing `return`
  outside it.
- The banner comments on `pathTraceOneBounce` and `pathTrace` are the same paragraph twice.
- `max(v.x, max(v.y, v.z))` appears 17 times across this file and `pathtrace_functions.h.slang`.
  One `maxComponent` helper.

### Structure

`pathTraceOneBounce` is about 700 lines in one function, with five compile-time axes (volume,
scatter, hair, transmission, DLSS guides) interleaved; hair alone is eight `#if` islands. The
banner comments already mark the seams: the environment miss, the first-hit guide capture, the
diffuse exit, next-event estimation, the continuation sample, the shadow-ray setup. Each could be
a function taking the state and the hit, which would also make the register-pressure reasoning in
the comments checkable per piece. The DLSS first-hit guide block, with its clear-glass
primary-surface trace, is a self-contained 60 lines with nothing to do with light transport.

### Not verified

The BSDF helpers (`bsdfEvaluate`, `bsdfEvaluateAll`, `bsdfSampleCore`) live in the nvpro_core2
fork, which was not checked out when this was written. Next-event estimation draws three random
numbers into the evaluate call's inputs on every vertex; if the all-lobes variant ignores them,
those draws are wasted and shift the random sequence. Check in the fork before acting.
