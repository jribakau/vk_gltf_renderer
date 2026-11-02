# Rendering Architecture -- Vulkan Ray Tracing Data Flow

> **For contributors and agents.** How the scene reaches the shaders: the retained store, its
> residency, the acceleration structures, and what the path tracer reads of them. Concepts plus
> pointers into `src/` and `shaders/`, not a restatement of the code.

## The retained scene

The scene is an `rs::RetainedScene` (`scene_retained.*`): entities, meshes, materials, lights, skins
and clips, with stable handles. A stream host feeds it (`GltfRenderer::hostScene`; developer.md, "The
retained scene in a stream session"), and `--scenefile` imports a file into it through
`rs::importGltf`, as a prop's model goes. glTF is a format at the edge: nothing keeps the model after
the import. `rs::RetainedSceneVk` (`scene_residency_vk.*`) keeps the store resident, and both
renderers read it through `SceneGpuView`.

```
rs::RetainedScene (entities, meshes, materials, lights)
        | takeChanges(), once a frame
        v
rs::Residency: instance, prim and material slots; capacity-sized tables
        |                                   |
        v                                   v
GltfScene tables the shaders read      BLAS per drawn submesh, TLAS over capacity
```

`rs::Residency` derives instances from entities incrementally. A mesh an entity skins, or one with
morph targets, is deformed for that entity: its own prims, over output streams the morph and skin
passes write and refit.

**Slots and side tables.** An entity's draw item (entity, mesh, submesh, instancing copy) is an
instance: a render-node row and a TLAS row, whose custom index is its prim slot. A submesh drawn
somewhere is a prim: its own index buffer and BLAS over its mesh's streams, or over its drawer's
outputs when the entity deforms it. Instance, prim and material slots come lowest-first from pools;
a freed slot is a hole (a zeroed row, a null TLAS row) until its retire frame, and only then goes
back to its pool, so a frame in flight never reads another object in a slot it drew. The tables
are sized by capacity (a power of two at or above the high-water mark); the shaders skip a hole
(`kInvalidRenderNodeMapping`). Beside them the residency keeps: per material slot a
`GltfShadeMaterial` row and a span of the texture-info table; texture and sampler slots (slot 0
the defaults), one image per texture and colour space; dense lights in document order; the
emissive table over shown instances whose materials emit; per prim slot the address of its
previous positions (at rest, its positions) and per instance slot its previous world (the DLSS
motion tables); and the
packed joint matrices, normal matrices and morph weights of every drawer.

**BLAS lifecycle.** A prim's BLAS is built in the frame its prim is made, recorded into that
frame's command buffer in batches whose scratch is bounded (`blasScratchBudgetMB`); refit when its
vertices are written (a dial, a skin or morph pass); built again when a stream it reads had to be
replaced, or after `blasRefitsBeforeRebuild` refits (0, off, by default: a 45° arm swing traced
no slower refit-only); and retired with its prim after the last instance lets go of it, its memory
freed a frame cycle later (`GpuBackend::retire`). A static prim (drawn from its mesh's own streams)
is built for compaction (`blasCompactStatic`, on): the build writes its compacted size to a query,
and once that frame has retired the next sync records the compacted copy, puts it in the TLAS and
retires the original. A compacted BLAS can't be refitted, so a vertex edit builds it again,
refittable from then on. The TLAS is built again over capacity whenever a row changes, keeping its
handle until capacity grows.

**Sync order** (`rs::Residency::sync`, once a frame before the passes): a frame-to-write barrier
(nothing below overwrites what a frame in flight reads before it); last sync's moved worlds and
written positions copied for motion; instances for the draw items that came, holes for those that
went, and their prims, meshes and material slots; the materials edited (an edit the live variant
can't draw held, D15); the images those need, decoded in one batch; vertex edits uploaded and their
BLASes refit; the deform passes; capacity grown; material rows and spans; the draw lists; the
render-node and prim rows and the TLAS; the motion tables; IES profiles, lights and the emissive
table; the scene descriptor; then a transfer-to-shader barrier. Everything a sync frees goes
through the retire queue; nothing waits for idle.

### Opacity micromaps (optional)

When a scene uses `EXT_mesh_opacity_micromap` and the device exposes `VK_KHR_opacity_micromap`
(and `VK_KHR_device_address_commands`, which it depends on), the residency (`scene_residency_vk.*`)
builds one opacity micromap per root `micromaps[]` entry and uploads the per-submesh micromap index
buffers alongside the geometry. Under the KHR extension a micromap is
an acceleration structure of type `VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR`: it is
created with `vkCreateAccelerationStructure2KHR` and built with `vkCmdBuildAccelerationStructuresKHR`.
A prim's BLAS build then attaches a
`VkAccelerationStructureTrianglesOpacityMicromapKHR` to each primitive's triangle geometry
(`triangles.pNext`), and the path-tracer RT pipeline is created with the OMM flag
(`VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_KHR`).

Ray queries opt in separately: every entry point that runs a ray query against a BLAS carrying a
micromap must declare the SPIR-V `OpacityMicromapIdKHR` execution mode. Its capability is only
legal on a device with the extension, so it cannot live in the embedded SPIR-V. Instead
`gltf_pathtrace.slang` emits it (inline SPIR-V on `computeMain` and `rgenMain`) behind the
`USE_OMM` macro, which `PathTracer` sets only when `SceneGpuView::hasOpacityMicromaps()` says so; a scene
that gains or loses micromaps triggers a recompile like any other variant switch.

This only accelerates alpha-tested traversal — microtriangles flagged "unknown" still run the
existing any-hit alpha logic — so the image is identical to the non-OMM path. When the extension
is unsupported (or `--useOpacityMicromap 0`), the whole subsystem is skipped and rendering falls
back to the regular alpha path.

The `eOpacityMicromap` entry of `enum Visualization` (see `shaders/shaderio.h`) adds a debug view
that colors the primary surface by how OMM traversal resolved it, using the fact that alpha
evaluation only happens for "unknown" micro-triangles: OMM-opaque micro-triangles are committed by
the hardware without any alpha work (green); "unknown" ones invoke the any-hit shader, which sets
`payload.ommUnknown` (yellow); transparent ones are culled so those rays miss the mesh and show the
environment behind. No baked micromap buffers are read by the shaders. Both techniques consult the
micromap: the RT pipeline through the pipeline flag above, the inline RayQuery technique through
`USE_OMM`.

This debug view (and its `payload.ommUnknown` field) is compiled out of the normal render path
behind the `USE_VISUALIZE` macro — the host sets it (see `PathTracer::compileShader`) only when the
selected `Visualization` mode is not `eRendered`, so shipping renders carry no extra payload field
or visualization branches. Because it is a compile gate, switching between normal rendering and any
visualization mode triggers a shader recompile (cached per `VariantKey`, so repeat switches are
fast); switching *among* visualization modes does not, since they share `USE_VISUALIZE == 1` and
select the mode at runtime via `frameInfo->visualization`.

---

## Emissive area lights

The path tracer treats emissive triangle **meshes** as area lights it samples directly (next-event
estimation + MIS in `sampleLights` / `gltf_pathtrace.slang`), instead of only finding them by chance
when a bounce ray happens to hit one. Punctual lights and the environment are sampled the same way.

**How it works, step by step**

Host — build the emitter list (`rs::Residency`, with `buildEmissiveTable` in `scene_gpu_tables.*`):

1. Walk every shown **instance** and look at its material.
2. Keep it if it emits: its **selection radiance**, `emissiveFactor × emissiveStrength ×
   emissiveLightScale`, is non-black. `emissiveLightScale` (`GltfShadeMaterial`) is 1 for a constant
   emitter and the emissive texture's mean luminance for a textured one (`imageMeanLuminance`, from a
   64 × 64 grid of the decoded texels; 1 where the format cannot be read, so the factor alone weighs
   it; 0 for an all-black texture, which is then no light).
3. For each triangle of that primitive, store a **reference** — render node, primitive, triangle
   index — plus the selection **radiance** (not baked vertices, so the shader reads live
   geometry/transforms). A textured emitter's light sample reads the texel at its point; the pdf, and
   a BSDF hit's MIS weight (`emitterSelectionRadiance`), use the selection radiance, which is what
   picked it.
4. Compute the triangle's **world-space area** (verts × node transform) and give it a **selection
   weight = area × luminance** (see the defensive tweak below). Build a **Vose alias table** from the
   weights (`aliasProb` / `aliasIndex` per triangle) for O(1) GPU selection, plus the totals
   (`numEmissiveTriangles`, `emissiveTotalWeight`, `emissiveMeanLum`).
5. Upload the flat list + those scalars to the scene descriptor.

Device — sample a light at each surface hit (`sampleLights`):

6. See what light sources exist: **environment**, **punctual lights**, **emissive triangles**.
7. **Choose a category**: first "a light" vs "the environment", then within lights "a punctual light"
   vs "an emissive triangle" (the probabilities balance the categories).
8. **For an emissive triangle**: pick one in **O(1) via the alias table** (bright/large ones win more)
   → read its live world vertices → **sample a random point** on it → form direction/distance and the
   sample's **pdf** → evaluate the surface BSDF toward it → **cast a shadow ray**; if unblocked, add
   the light, **MIS-weighted** against BSDF sampling.
9. Separately, when a normal bounce ray **hits an emitter directly** (`gltf_pathtrace.slang`, the
   emissive add), its glow is added but **MIS-weighted**, so the "sampled via NEE" and "hit by chance"
   paths combine without double-counting.
   The weights on the two sides agree only because the NEE pdf includes the probability of picking
   the category in step 7, which is what the hit side recomputes. Punctual lights are the exception:
   no bounce ray can hit one, so their samples take no MIS weight at all. All but the sky's sun,
   whose disk a bounce ray *can* see when it escapes: its cone sample is MIS-weighted against the
   BSDF like an emitter (`skySunNeePdf` / `computeSkySunHitMisWeight` in
   `shaders/pathtrace_functions.h.slang`). Without that, a lobe narrower than the disk — smooth
   glass, a polished mirror — would never show the sun.

**Details / why it's shaped this way**

The list is derived data, regenerated wholesale (never patched). It stores references, but each
emitter's **selection weight is its world-space area × luminance**, so it also depends on the
emitter's transform. The sync rebuilds it when materials, the emitter set, or an emitter's transform
change. The shader samples the emitter at its **live** GPU transform, and the pdf cancels the world
area (below), so the host table and the live geometry must agree on that area — a stale table biases
the estimate, it does not merely add noise.

Selection is **power-weighted** (an O(1) **Vose alias table**, not a CDF binary search), with a
**defensive floor** so a dim emitter next to a bright one is never starved: each emitter's luminance
is pulled toward the scene's area-weighted mean, `lum' = (1−f)·lum + f·meanLum`
(`EMISSIVE_DEFENSIVE_FRACTION`, kept in sync between host and shader). The alias table samples the
same distribution `P(i) = weightᵢ / emissiveTotalWeight`, so the pdf is unchanged.
The world area **cancels** in the solid-angle pdf (`pdf = lum' · dist² / (emissiveTotalWeight · cos)`),
so the BSDF-hit MIS weight is recoverable from local hit geometry (the emitter's radiance +
`emissiveTotalWeight` + `emissiveMeanLum`) with **no per-triangle lookup**; and because `meanLum = W/A`
the total weight `W` is unchanged. The weight uses world area, so the sync rebuilds the list whenever
an emitter's transform changes. Morph/skin-deformed emitters are the one remaining case that falls
back to last-rebuilt-pose sampling. The scene-descriptor scalars refresh whenever the count
**or** the weights change, so radiance-only edits (e.g. dragging emissive strength) take effect
immediately.

---

## Ray tracing hit correlation

```
When a ray hits TLAS instance[i] (Slang RT / ray query builtins):
  InstanceIndex()  = i             -- instance slot, index into renderNodes SSBO
  InstanceID()     = prim slot     -- index into renderPrimitives (instanceCustomIndex)
  Shader reads:
    renderNodes[InstanceIndex()]        → transform, materialID
    renderPrimitives[InstanceID()]      → vertex/index buffer info
```

## Shadow rays

A shadow ray (`TraceShadow` in `shaders/raytracer_interface.h.slang`, one implementation per
technique) answers one question: how much light gets through to the sample point. Every technique
treats a **committed hit as fully blocked**, which shapes the traversal:

- **Opaque instances** (`getInstanceFlag` sets `VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR` for
  materials with no transmission, alpha mode `OPAQUE`, no diffuse transmission) never run any-hit.
  Hitting one commits, and the rays use `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`, so traversal
  stops at the first such hit instead of searching for the closest.
- **Other instances** run the any-hit logic (`rahitShadow`, or the candidate loop in the ray-query
  path), both through `shadowCrossSurface`, and ignore the hit, so the ray keeps going:
  - A surface that only filters the light (alpha, thin glass) multiplies in its expected
    transmission, `(1 - opacity) + opacity * getShadowTransmission`. Glass passes only where it
    is smooth (a clear crossing, which the path's own ray crosses as a null event for MIS); rough
    or scattering glass blocks.
  - A surface that bounds a medium (`shadowSurfaceBoundsMedium`) keeps the stochastic opacity
    test, since what the rest of the ray travels through depends on it. It is also where the
    Beer-Lambert segment since the previous boundary is charged.
  - Below `SHADOW_ROULETTE_BELOW` the transmission is rouletted. A lost roulette accepts the hit
    and ends the search, which counts as blocked.
- **The ground.** The infinite plane is analytic, not in the TLAS. `TraceShadow` multiplies in
  `infinitePlaneShadowTransmittance` (`shaders/ground.h.slang`): what a path's own ray would miss
  where the shadow ray crosses the floor, and 1 for the shadow catcher.

The ray tracing pipeline's shadow hit group has no closest-hit shader. The non-SER path traces with
`RAY_FLAG_SKIP_CLOSEST_HIT_SHADER` and reads the result from `rmissShadow`, which sets
`ShadowPayload::missed`: if the miss shader never ran, the ray was blocked. The SER path checks
`HitObject::IsHit()` instead and never runs closest-hit or miss shaders.
