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
