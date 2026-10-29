# Path tracer cleanup notes

Review of `src/renderer_pathtracer.cpp` and `src/renderer_pathtracer.hpp`, 2026-10-29.
Nothing major: details to go through later.

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
