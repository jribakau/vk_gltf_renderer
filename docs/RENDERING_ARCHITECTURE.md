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
   emitter and the emissive texture's mean luminance for a textured one (`imageMeanLuminance`: an
   8-bit image's from a 64 × 64 grid of its texels; a cooked BC7 or BC4 one's from its 1 × 1 level,
   which the cooker averages in linear light, or from its one block where it held one value and loaded
   as that (`LoadedImageData::constant`), decoded by `bc7FirstTexel`; 1 where neither can be
   read, so the factor alone weighs it; 0 for an all-black texture, which is then no light).
3. For each triangle of that primitive, in order, store a **reference** — render node, primitive,
   triangle index — plus the radiance (not baked vertices, so the shader reads live
   geometry/transforms) and its **selection**: the weight over its area that both MIS sides read. A
   textured emitter's light sample reads the texel at its point; the pdf, and a BSDF hit's MIS weight
   (`emitterSelection`, which finds the hit triangle's record at its leaf's first plus its primitive
   index, never the material row), use the selection, which is what picked it. A degenerate triangle
   keeps its place at selection 0, never sampled. A texture that lands later with a different mean
   changes the row, and the residency rebuilds the table in that sync (`m_relitSlots`).
4. A **textured** emitter's triangles are each weighed by their own texels: the residency asks the
   backend once for its texture's mean colour over every triangle of its drawn run
   (`GpuBackend::averageTexels`, `shaders/emissive_texels.comp.slang`: 64 points a triangle at the
   level of detail that spreads them over its texels), keeps the answer by mesh, run, texture, UV set
   and transform (`Residency::TexelKey`), and builds the table again when it comes back a few frames
   later; until then, and on a backend without the pass, the emitter is weighed whole. A facade whose
   emission mask is mostly black then spends its light samples on its lit signs.
5. Compute the triangle's **world-space area** (verts × node transform) and give it a **selection
   weight = area × selection** (see the defensive tweak below). Build a **Vose alias table** from the
   weights (`aliasProb` / `aliasIndex` per triangle) for O(1) GPU selection, plus the totals
   (`numEmissiveTriangles`, `emissiveTotalWeight`).
6. Upload the flat list + those scalars to the scene descriptor.

Device — sample a light at each surface hit (`sampleLights`):

7. See what light sources exist: **environment**, **punctual lights**, **emissive triangles**.
8. **Choose a category**: first "a light" vs "the environment", then within lights "a punctual light"
   vs "an emissive triangle" (the probabilities balance the categories).
9. **For an emissive triangle**: pick one in **O(1) via the alias table** (bright/large ones win more)
   → read its live world vertices → **sample a random point** on it → form direction/distance and the
   sample's **pdf** → evaluate the surface BSDF toward it → **cast a shadow ray**; if unblocked, add
   the light, **MIS-weighted** against BSDF sampling.
10. Separately, when a normal bounce ray **hits an emitter directly** (`gltf_pathtrace.slang`, the
   emissive add), its glow is added but **MIS-weighted**, so the "sampled via NEE" and "hit by chance"
   paths combine without double-counting.
   The weights on the two sides agree only because the NEE pdf includes the probability of picking
   the category in step 8, which is what the hit side recomputes. Punctual lights are the exception:
   no bounce ray can hit one, so their samples take no MIS weight at all. All but the sky's sun,
   whose disk a bounce ray *can* see when it escapes: its cone sample is MIS-weighted against the
   BSDF like an emitter (`skySunNeePdf` / `computeSkySunHitMisWeight` in
   `shaders/pathtrace_functions.h.slang`). Without that, a lobe narrower than the disk — smooth
   glass, a polished mirror — would never show the sun.

**One pass or two.** The pick is scored first (`lightPickSet`: every punctual light's estimate and
every node of the light tree's cut, up to 64) and the light then found by walking those shares again
to where the draw's number falls (`lightDrawOf`). A still keeps that walk: it inverts the pick's sum,
so a Sobol number stays spread over the lights. A DLSS frame's draws, of plain random numbers, pick
as they score instead (`lightPickSetDrawn`, `PickReservoirs` in `light_pick.h.slang`: a reservoir by
estimate and an even one, then the defensive mixture), each light with exactly its probability. The
walk is a second pass that a warp pays to the end for its furthest-walking thread: on 7TEST (117
emitters, a 64-node cut) at 4K the moving frame's path trace went 42.4 → 33.6 ms (2026-11-07); with
a light or two there is nothing to save.

**Details / why it's shaped this way**

The list is derived data, regenerated wholesale (never patched). It stores references, but each
emitter's **selection weight is its world-space area × luminance**, so it also depends on the
emitter's transform. The sync rebuilds it when materials, the emitter set, or an emitter's transform
change. The shader samples the emitter at its **live** GPU transform, and the pdf cancels the world
area (below), so the host table and the live geometry must agree on that area — a stale table biases
the estimate, it does not merely add noise.

Selection is **power-weighted** (an O(1) **Vose alias table**, not a CDF binary search), with a
**defensive floor** so a dim emitter next to a bright one is never starved: each emitter's mean
luminance `L` is pulled toward the scene's area-weighted mean, `(1−f)·L + f·meanLum`
(`EMISSIVE_DEFENSIVE_FRACTION`, host only), and that weight is shared among its triangles by area
times their own luminance (`Emitter::triangleLuminance` where it has one, so a black part of a
textured emitter stays at 0). The alias table samples the same distribution
`P(i) = weightᵢ / emissiveTotalWeight`, so the pdf is unchanged. The world area **cancels** in the
solid-angle pdf (`pdf = selectionᵢ · dist² / (W · cos)`, with `W` the total or the leaf's weight), so
the BSDF-hit MIS weight is recoverable from local hit geometry and the hit triangle's record alone;
the shader holds none of the host's weighting policy. The weight uses world area, so the sync rebuilds the list whenever
an emitter's transform changes. Morph/skin-deformed emitters are the one remaining case that falls
back to last-rebuilt-pose sampling. The scene-descriptor scalars refresh whenever the count
**or** the weights change, so radiance-only edits (e.g. dragging emissive strength) take effect
immediately.

---

## Haze in the air

A thin, even haze lit by every light of the scene (`shaders/haze.h.slang`), set by `hazeVisibility`
(metres: where it leaves 2% of a dark object's contrast, so the extinction is 3.912 / V; 0 is none)
and `hazeAnisotropy` (its Henyey-Greenstein g, the halo round a light seen through it). Its albedo is
1. It is on the camera's segment only, applied at the end of `pathTrace` beside the aerial
perspective: every bounce's light reaches the eye through that segment, so the path's radiance is
dimmed by the segment's transmittance and what the haze scatters toward the eye is added. Light on its
way to a surface, a mirror's view and a window's are not hazed, and the haze is not defocused (the
pinhole's eye, as the aerial perspective).

- **Where it is:** a sphere round the scene's bounds, its punctual lights and the eye
  (`GltfRenderer::fillHaze`, into `SceneFrameInfo::hazeCenter`/`hazeRadius`). The eye is inside it and
  a light's beam reaches the scene inside it; a ray that hits nothing is hazed to the sphere's edge,
  so a beam in front of the sky shows and the sky is seen through as much haze as the scene holds.
  Not over the shadow catcher, nor for an orthographic camera.
- **One scatter point a camera path** (`hazeCameraSegment`, `shaders/pathtrace_functions.h.slang`):
  its distance drawn half by the transmittance from where the ray enters the haze and half
  equiangularly (Kulla and Fajardo 2012) toward a guide, a point on a light `sampleLights` drew from
  a point of the segment; none for a light at infinity. The point is then lit by a light sample drawn
  there, as a walk's scatter is (`volumeScatterNEE`), with one shadow ray dimmed by the haze to the
  sphere's edge. The mixture's density is the pdf, so the estimate is unbiased whichever light the
  guide was; drawing the point's own light there, not reusing the guide's, is what finds a spot's
  beam wherever it crosses the ray (a light picked from elsewhere on the segment gets nothing where
  the cone misses that point).
- **A still's light, from candidates** (`HazeReservoir`, `hazeCandidates`, 4): the light pick weighs
  lights by power and distance alone, and at a glow of 0.7 the phase function sends 180 times more
  light on than back. A still's scatter point draws several light samples, keeps one with the chance
  its weight (its light over its pdf, the phase function toward it, the haze on the way) has of their
  sum, and traces one shadow ray for it; the kept light counts for them all. The lights are scored
  once for them all and the c-th is drawn from the c-th of equal slices of that pick (stratified),
  so their draws rise and one walk over the lights finds them all (`LightDrawWalk`, the walk
  `sampleLights` also takes); on a night street of 30 lamps and 12 signs four cost 2.86 ms a sample
  against one's 2.35 (3.52 with a scoring and a walk each) and haze off's 1.05. A still also cuts the
  haze's samples to the firefly clamp's level, as it does bounced light's. A moving frame keeps the
  pick's one sample and its clamp-free light, bit for bit, for Ray Reconstruction.
- **What a moving frame pays for it:** two light draws a path, the guide's and the scatter point's,
  and one shadow ray. On 7TEST (one sun, 117 emitters, a 64-node cut) at 4K the draws were most of
  the haze's 29.5 ms, the shadow ray ~2 ms, and the largest part of the draws was `lightDrawOf`'s walk:
  a second pass over the 64 nodes that a warp pays to the end for its furthest-walking thread. A
  draw on its own (the guide, a moving frame's one candidate) is now drawn as the lights are scored
  (`lightPickSetDrawn`, `PickReservoirs` in `light_pick.h.slang`: a reservoir by estimate and an even
  one, then the defensive mixture), each light with exactly its `lightPickProbability`, so the noise
  is of the same kind; and the guide takes only its point (`pointOnly`: no material, emission or
  opacity read). Haze's cost there 29.5 → 19.9 ms a moving frame; on the night street one
  candidate's 2.35 → 1.70 ms a still sample (haze off 1.05). A still's several candidates keep their
  one walk (stratified).
- **Checked:** `tests/test_haze.cpp` holds the estimator to a point light's single scattering by
  quadrature (itself held to thin isotropic haze's closed form) and the drawn distances to their
  density; a ray passing a light at 5 cm on a 10 m segment has a noise 0.9 times its mean a sample,
  5.7 by the transmittance alone. Rendered (a point light 3 m ahead in 20 m visibility, g 0.7, 1024
  samples), rings of pixels from 1 to 26 degrees off it came within 0.1% of the integral.
- **Uneven** (`hazeHeight`, `hazePatches`, `hazePatchSize`; `HazeShape` in `haze.h.slang`): the
  density relative to the floor's, exp(-(height above the floor) / H) (1 below it) times 1 + patches
  x a two-octave value noise in [-1, 1]. The floor is the ground plane's (a body of water's surface)
  or the scene's lowest point. The height's part is integrated along a ray in closed form
  (`hazeHeightDepth`, from the segment's lower end so nothing overflows) and inverted to draw a
  distance (`hazeHeightInvert`), so a layer of fog seen from above draws its points inside it. The
  patches dim the camera's segment by four lookups stratified over its height-weighted depth
  (`HazeColumn`: the optical depth unbiased, its exponential a hair high, 0.005% in the test), the
  transmittance half of the mixture draws by that column and the equiangular half as before, and the
  scatter point weighs the density where it lands; the light's way to the point is dimmed by the
  height alone. Both at 0, the closed forms of even haze run, to rounding as before. On the night
  street at 30 m visibility a still sample went 4.59 → 4.94 ms with a height, 5.11 with patches, 5.68
  with both. Not ratio tracking, which would multiply everything seen through the haze by its noise.
  The patches do not move.
- **Not in it:** the haze's multiple scattering, light bounced off surfaces into it, and its shade on
  what it lights (docs/falcor-ideas.md: the air as its own medium). DLSS-RR has no guide for light in
  the air.

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
  where the shadow ray crosses the floor, and 1 for the shadow catcher. A path's hit takes
  `groundMaterial` (`shaders/pathtrace_functions.h.slang`): the plane's colour and roughness under
  the water the host sets on it (`groundWaterAt`), damp ground and puddles from a noise field in
  world space, evaluated at the hit and filtered by its ray footprint. No texture and no extra ray;
  the shadow test is the same wet or dry.
- **A body of water over the ground** (`groundHasWater`). Its surface is a heightfield of gentle
  waves (`groundWavesAt`) that a path's ray meets analytically after the floor
  (`checkWaterIntersection`, `groundWaterCrossing`), shaded as smooth water with the waves and
  ripples in its normal (`waterSurfaceMaterial`), and what the ray's footprint averages of them in
  its roughness as Gaussian slopes, a Beckmann lobe (`PbrMaterial::gaussianSlopes`; sky.md § What a
  reflection sees). The path keeps its own `inWater` flag beside a
  mesh's medium: each segment under the surface is charged pure water's absorption, and what is
  under it sees water, not air, outside its surfaces. Crossing the surface is a clear crossing for MIS
  and spends a budget of its own, the glass's size. A shadow ray crosses the surface straight
  (`groundWaterShadowTransmittance`), keeping its Fresnel share and what the water does not
  absorb. From under the water to a light above it also takes the caustic there
  (`groundWaterCaustic`): it finds where the surface lets in the light that lands at its origin and
  brightens it by how much the surface focuses that light on the way down, the inverse of the
  landing's Jacobian. Ripple and swell octaves that focus before that depth are left out of the
  bending, so the landing does not fold and the floor keeps the light that came in on average.
  Only light samples carry it; a path's own ray refracts through the surface as it is.

The ray tracing pipeline's shadow hit group has no closest-hit shader. The non-SER path traces with
`RAY_FLAG_SKIP_CLOSEST_HIT_SHADER` and reads the result from `rmissShadow`, which sets
`ShadowPayload::missed`: if the miss shader never ran, the ray was blocked. The SER path checks
`HitObject::IsHit()` instead and never runs closest-hit or miss shaders.
