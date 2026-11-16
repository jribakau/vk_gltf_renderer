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

**What a sync looks at.** A sync costs what changed, not what is drawn: a world of a few hundred
thousand instancing copies moves a ship, shows a tree or takes in a quad of ground for the price of
those. The store says which entities had their mesh instance set or dropped
(`SceneChanges::instances`), which moved or were shown (`transforms`, `visibility`, descendants
included) and, of an entity only some copies of which did, which copies (`SceneChanges::copies`:
`moveCopies`, `setCopiesShown`). The residency keeps each entity's instances by entity
(`Residency::EntityDraw`: its runs, each as many times as it has copies) and goes through those
entities alone: the structure pass diffs an entity's runs against the ones it had (a run that stays
keeps its copies' slots), the row pass places the instances of the entities that moved, or only the
copies named. Every entity is gone through only when a mesh's own state says what is drawn (parked,
arriving a few a sync, brought back, a fresh scene), which costs the entities, not their copies.
The TLAS rows are worked out again only for slots placed, shown or linked since the last build and
for the instances of a prim whose BLAS is another (found by `m_primRuns`); the emissive table finds
its emitters by their materials (`m_materialRuns`); the scene's bounds are kept a box an entity; the
raster lists are made when read. `Residency::lastSyncWork` counts what a sync looked at, and
the tests hold it to the change (`RetainedResidency.ASyncLooksOnlyAtWhatChanged`). A copy can be
hidden by itself (`MeshInstance::hidden`): it keeps its instance and its slot, its TLAS row without
its BLAS, as a hidden entity's.

**Slots and side tables.** An entity's draw item (entity, mesh, submesh, instancing copy) is an
instance: a render-node row and a TLAS row, whose custom index is its prim slot. The row's mask and
flags are its material's (`materialBits`), less the camera's bit for an instance the camera looks
through (`Entity::offCamera`, `instanceMaskOf`): the camera's own rays and the picks trace with that
bit alone, every other ray with all of them, so such an instance still shades, bounces light and
shows in mirrors. Light and shadow linking rides the same two rows (`rs::LightLink`, an entity's own
or its nearest parent's, else the scene's): the lamps that leave an instance dark are a bit each in
its render node (`GltfRenderNode::unlit` against `GltfLight::litBit`; the path tracer gives such a
lamp no strength at that surface, `getPunctualLight`), and a lamp whose shadows only some instances
cast traces its shadow rays with one `INSTANCE_MASK_SHADOW_*` bit (`GltfLight::shadowMask`), which
an instance that casts it none has lost from its mask. A hidden instance's row has no BLAS, so no ray meets it. A submesh drawn
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
retires the original. That BLAS isn't built to be refitted, so a vertex edit builds it again,
refittable from then on. A deformed prim (a posed figure, hair) is built both to be refitted and for
compaction (`blasCompactDeforming`, on): its compacted copy keeps the build data and is refitted in
the original's place, a seventh smaller (an updatable BVH compacts to 86-89%; a static one to about
41%). Each build of one, a reshape's included, makes a copy a frame or two later. Settling under
memory pressure still builds a quiet mesh's BVH again static and compacted, the smaller of the two.
The TLAS is built again over capacity whenever a row changes, keeping its handle until capacity
grows. Its rows are kept between builds (`Residency::buildTlas`): a row's transform is made again only for
a slot moved or rebound since the last build, and only the rows that differ are written to the device
(`GpuBackend::buildTlasRows`; every row into a new TLAS). Which BLASes its rows point at is counted
a BLAS (`TlasRefs`), so one let go of is kept exactly until a TLAS is built without it. The build
itself is the device's, over every row: 0.4 to 1.4 ms at 130 to 170 thousand rows on an RTX 4080
(`blasTimeBuilds` logs it for a frame that only moved instances too).

**Sync order** (`rs::Residency::sync`, once a frame before the passes): a frame-to-write barrier
(nothing below overwrites what a frame in flight reads before it); last sync's moved worlds and
written positions copied for motion; instances for the draw items that came, holes for those that
went, and their prims, meshes and material slots; the materials edited (an edit the live variant
can't draw held, D15: the variant it needs builds off the frame, and is taken in by the next frame
that traces, or with no frame traced when the still is complete, `PathTracer::takeBuiltVariant`; the
sync after that lets the edit through); the images those need, decoded in one batch; vertex edits uploaded and their
BLASes refit; the deform passes; capacity grown; material rows and spans; the draw lists; the
render-node and prim rows and the TLAS; the motion tables; IES profiles, the materials' water
(`scene_water.*`), lights and the emissive table; the scene descriptor; then a transfer-to-shader
barrier. Everything a sync frees goes
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
   geometry/transforms) and its **weight**, which both MIS sides read. A
   textured emitter's light sample reads the texel at its point; the pdf, and a BSDF hit's MIS weight
   (`emitterWeight`, which finds the hit triangle's record at its leaf's first plus its primitive
   index, never the material row), use the weight, which is what picked it. A degenerate triangle
   keeps its place at weight 0, never sampled. A texture that lands later with a different mean
   changes the row, and the residency rebuilds the table in that sync (`m_relitSlots`).
4. A **textured** emitter's triangles are each weighed by their own texels: the residency asks the
   backend once for its texture's mean colour over every triangle of its drawn run
   (`GpuBackend::averageTexels`, `shaders/emissive_texels.comp.slang`: 64 points a triangle at the
   level of detail that spreads them over its texels), keeps the answer by mesh, run, texture, UV set
   and transform (`Residency::TexelKey`), and builds the table again when it comes back a few frames
   later; until then, and on a backend without the pass, the emitter is weighed whole. A facade whose
   emission mask is mostly black then spends its light samples on its lit signs.
5. Compute the triangle's **world-space area** (the mesh record's vertices × node transform) and give
   it a **weight = area × selection luminance** (see the defensive tweak below). Build a **Vose alias table** from the
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
to where the draw's number falls (`lightDrawOf`). A DLSS frame's draws pick as they score instead
(`lightPickScore`, `PickReservoirs` in `light_pick.h.slang`: a reservoir by estimate and an even
one, then the defensive mixture), each light with exactly its probability, and so do a still's
where the light tree is on (`ptStillPickAsScored`). The walk is a second pass that a warp pays to
the end for its furthest-walking thread: on 7TEST (117 emitters, a 64-node cut) at 4K the moving
frame's path trace went 42.4 → 33.6 ms (2026-11-07). The walk inverts the pick's sum, so a still's
Sobol number stays spread over the lights; picking as scored gives that up. On 7TEST at 1280×720 a
still sample went 21.7 → 16.5 ms and its noise rose 2-11% (the relMSE at 64 and 256 samples), so
the error at equal time fell 13-16%. A still with no tree, or past 64 punctual lights, keeps the
walk; with a light or two there is nothing to save.

**Details / why it's shaped this way**

The list is derived data, regenerated wholesale (never patched). It stores references, but each
emitter's **weight is its world-space area × luminance**, so it also depends on the
emitter's transform. The sync rebuilds it when materials, the emitter set, or an emitter's transform
change. The shader samples the emitter's **live** geometry, and its pdfs are of the triangle's area
there (below), so a table that lags or never sees a deformation only sends the samples to the wrong
places: more noise, the same picture.

Selection is **power-weighted** (an O(1) **Vose alias table**, not a CDF binary search), with a
**defensive floor** so a dim emitter next to a bright one is never starved: each emitter's mean
luminance `L` is pulled toward the scene's area-weighted mean, `(1−f)·L + f·meanLum`
(`EMISSIVE_DEFENSIVE_FRACTION`, host only), and that weight is shared among its triangles by area
times their own luminance (`Emitter::triangleLuminance` where it has one, so a black part of a
textured emitter stays at 0). The alias table samples the same distribution
`P(i) = weightᵢ / emissiveTotalWeight`, so the pdf is unchanged. The point is then drawn evenly over
the triangle as the shader reads it, so the solid-angle pdf is
`weightᵢ · dist² / (W · areaᵢ · cos)` (`emissiveTrianglePdf`), with `W` the total or the leaf's weight
and `areaᵢ` the triangle's world area **there**, which both sides compute from its vertices
(`sampleEmissiveTriangle`, `triangleWorldArea` at a BSDF hit). Until 2026-11-10 the table stored the
weight over its own area and the pdf let the area cancel, which is right only while the table's area
is the live one: a skinned emitter's table is built from the mesh record's positions, before its skin
(and before a glTF morph's weights), so a triangle a pose had stretched was dimmed by the stretch, and
one a dial was still moving by the lag. The BSDF-hit MIS weight is recoverable from local hit geometry
and the hit triangle's record alone; the shader holds none of the host's weighting policy. The weight
uses world area, so the sync rebuilds the list whenever an emitter's transform changes.
Skin- and morph-deformed emitters are still **weighed and bounded** by the mesh record's positions
(the triangles' shares within the emitter, and its leaf's centre, radius and cone in the light tree):
that costs noise where a pose has carried the emitter far from its bind pose, not correctness. The scene-descriptor scalars refresh whenever the count
**or** the weights change, so radiance-only edits (e.g. dragging emissive strength) take effect
immediately.

---

## Light groups

A picture is the sum of what each of its lights gives it, so a group of lights made twice as strong,
or tinted, is that group's share of the picture times two, or times the tint. The Light tab's mix
(the lighting's `mix`, `fv_lighting.hpp`) does that for six groups (`shaderio.h` `LIGHT_GROUP_*`):
the sky (the environment and what is seen of it, the air's and the clouds' own light),
the sun (the lighting's sun under any environment, the sky's sun or moon), glowing surfaces, and
three of lamps (`GltfLight::group`, a lamp's `group` in `prop.set`).

Every contribution a path adds goes through `addRadiance` or `addLight`
(`shaders/pathtrace_functions.h.slang`) with the group of the light it came from: a light sample
carries its light's (`DirectLight::group`), a ray that meets an emitter, the sky, the sun's disc or
a sized lamp knows what it met. What dims the camera's segment (the air, the haze, clouds in front)
dims every group alike (`dimRadiance`). Two things follow from the frame's variant:

- **A frame that keeps no groups** (every moving frame, and a still with `ptLightGroups` off)
  multiplies each contribution by its group's factor as it is added (`SceneFrameInfo::lightMix`,
  with `eSceneLightMix`; no cost while the mix is neutral). A change to the mix starts such a still
  again. The light pick does not know the mix: a group turned off is still sampled.
- **A still that keeps its lights apart** (`ptLightGroups`, the variant's `USE_LIGHT_GROUPS`,
  `VariantKey::lightGroups`: an accumulation's frames alone, so a camera start or stop is a cached
  swap) notes one group's share a sample (`PathTracerState::share`, `g_shareGroup`): the groups the
  scene has light in (`SceneFrameInfo::lightGroupsUsed`, `PathTracer::lightGroupsUsed`) take turns, a
  pixel's samples in order from a start of its own (`lightGroupOfSample`,
  `shaders/light_mix.h.slang`). The frame writes that group's mean image beside the whole
  (`LightGroups`, `src/light_groups.hpp`: one RGBA32F and six RGBA16F, 64 bytes a pixel, bound at
  `eResultImage` and `eLightGroup0`; cleared as a still begins). The rendered image is then written
  by a pass over them (`shaders/light_mix.slang`): with each group's factor m and the least of them
  a, it is a times the whole plus (m - a) times each group's image. Every term is a light times a
  factor that is not negative, so the mix is never under zero; it is right on average whatever the
  images' noise; and with every factor 1 it is the whole picture bit for bit. Everything after it
  (the still's denoiser, Keep detail, the tone mapper, an export) reads the rendered image as
  before. On an idle frame of a finished still a changed mix runs the pass and the still's denoiser
  again (`RrStillDenoiser::redoFinal`), with nothing traced.

Why one share a sample: a path that carried every group's share made a still half again as slow (a
face at 1280x960: 33 ms a sample against 21 with five shares, 42% more with three; indexing them
by the group, looping over them or naming them made no difference), while one share costs about a
twentieth (23.8 against 22.5). The price is noise in what the mix changes: a group's image holds a
K-th of the samples with K groups in turn, so a group made brighter adds its image's noise, and
with a group turned off the mix is the other groups' images alone. The whole picture, and so the
still while the mix is neutral, has every sample as before.

What the camera sees where its ray met nothing (`tryPrimaryMissBackplate`: a solid colour, the
sky drawn analytically, a blurred HDRI) is the sky's (`setBackplate`), under either kind of frame.
A ray that meets two sized lamps of different groups at once counts both as the brighter one's. The
firefly clamp acts on each contribution before the mix either way.

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
  (`lightPickScore`, `PickReservoirs` in `light_pick.h.slang`: a reservoir by estimate and an even
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
- **Other instances** run the any-hit logic, through `shadowCrossSurface`, and ignore the hit, so
  the ray keeps going. Both techniques run it in the candidate loop of an inline ray query
  (`traceShadowInline`). In the RTX pipeline that loop runs in raygen, under the specialization
  constant `USE_INLINE_SHADOW`, which `--ptInlineShadow` sets when a pipeline is created. On Ada it
  took 15-18% off a moving frame's path trace and 12-16% off a still's (1TEST, 7TEST, 10TEST,
  2026-11-08). `--ptInlineShadow 0` brings back the any-hit shader, `rahitShadow`, for an A/B:
  - A surface that only filters the light (alpha, thin glass) multiplies in its expected
    transmission, `(1 - opacity) + opacity * getShadowTransmission`. Glass passes only where it
    is smooth (a clear crossing, which the path's own ray crosses as a null event for MIS); rough
    or scattering glass blocks. A thin pane's colour is a crossing's face-on and deepens at a
    slant, raised to 1 / cos of the refracted angle as the path through the glass lengthens
    (nvpro_core2 `bsdfThinTintAtSlant`, read off the triangle's face here and the shading normal
    in the BSDF, which takes it for eval and sample alike): 7TEST's car windows, 0.77 face-on,
    pass 0.73 at 60 degrees and 0.70 toward grazing. Iray keeps a crossing's colour at every angle;
    a colourless pane is unchanged.
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
  mesh's medium: each segment under the surface is charged pure seawater's extinction (its
  absorption and its scattering), takes the light the water scatters toward it
  (`groundWaterInScatter`: one point drawn along the segment, one light sample and shadow ray there,
  single scattering, so the open sea past the floor's fade glows blue and the clouds' shadow on the
  sun reaches it), and what is under it sees water, not air, outside its surfaces. Crossing the surface is a clear crossing for MIS
  and spends a budget of its own, the glass's size. A shadow ray crosses the surface straight
  (`groundWaterShadowTransmittance`), keeping its Fresnel share and what the water does not
  take. From under the water to a light above it also takes the caustic there
  (`groundWaterCaustic`): it finds where the surface lets in the light that lands at its origin and
  brightens it by how much the surface focuses that light on the way down, the inverse of the
  landing's Jacobian. Ripple and swell octaves that focus before that depth are left out of the
  bending, so the landing does not fold and the floor keeps the light that came in on average.
  Only light samples carry it; a path's own ray refracts through the surface as it is.

**None at a mirror.** A path's vertex whose every weighted lobe is a mirror (`smoothVertex`,
`shaders/smooth_vertex.h.slang`: GGX alpha at most `SMOOTH_VERTEX_ALPHA`, 5e-4, and no diffuse or
sheen, judged after the path's regularisation) evaluates no BSDF for its light sample and traces no
shadow ray: its value and MIS weight there are next to nothing. That is clear glass and bottles at
roughness 0, the eye's tear film at its 0.02 cap, polished metal. Its BSDF ray's hits weigh 1 instead
(`LightPick::bsdfOnly`, which the four hit-side weights read before any of them rebuilds the pick from
its point: the environment, the sky's sun, the emitters with their light tree, the sized lights), so
the image is the same in expectation. The pick is still drawn. A punctual (DIRAC) light keeps its
sample, and so does one where a lobe may scatter that the BSDF ray never leaves by (`bsdfFaceAllows`:
on or past the face for a reflection, above it for a refraction), weighed 1 only where the BSDF's pdf
is 0 (`smoothVertexLightWeight`). Not at a path's last vertex, whose BSDF ray is never traced. The
threshold is not the clear crossing's (alpha 0.01): a lobe that wide sees the sun or a lamp's bulb
better through the light sample. `ptSmoothBsdfOnly`, on; measured 2026-11-07 a still's sample 2.1%
cheaper at 1TEST's bar, 0.6% in its saved view, nothing measurable in a face or an eye close-up.

The ray tracing pipeline's shadow hit group has no closest-hit shader. The non-SER path traces with
`RAY_FLAG_SKIP_CLOSEST_HIT_SHADER` and reads the result from `rmissShadow`, which sets
`ShadowPayload::missed`: if the miss shader never ran, the ray was blocked. The SER path checks
`HitObject::IsHit()` instead and never runs closest-hit or miss shaders.

## A cheap camera's lens and sensor

What makes a picture read as footage is mostly the camera's defects, so a look can ask for them
(the app's Footage looks; a look's `bend`, `fringe`, `clip` and `noise` in the lighting's
`tonemap.look`, at the look's strength). All off by default, against the physical-accuracy target.

- **The lens** (`src/lens.hpp`, `shaders/lens.slang`, the mapping in `shaders/lens.h.slang`;
  `lensBend`, `lensFringe`): a pass on the image the tone mapper and the HDR output take, before
  bloom. Barrel distortion draws the rendered image in toward its corners, most at the centre, so
  the middle is larger, the edges are squeezed and straight lines bow; the corners stay, so nothing
  outside the rendered frame is needed. Fringing reads red a little nearer the centre and blue a
  little further out, by nothing at the centre and most at the corners. An .exr export keeps the
  image without it.
- **A bent picture and the image as rendered:** the path tracer, the pickers and the overlays work
  in the image as rendered. A frame that draws an overlay (the selection's outline, the grid, the
  gizmo; `GltfRenderer::overlaysShown`) is therefore not bent, so an overlay never sits beside what
  it marks, and a click or a pixel readout on a bent frame is taken to its place in the rendered
  image first (`Lens::rendered`). A screenshot's frame draws no overlay, so it is bent.
- **The sensor** (nvshaders' `applyTonemap`; `tmSensorClip`, `tmSensorNoise`): `clip` gives the
  tone curve's highlights way to a response that is straight up to white and clips there a channel
  at a time (`sensorClipCurve`), exposed so middle grey stays where the curve has it; on an HDR
  display those highlights stay at SDR white, as the Clip method's do. `noise` is strongest in the
  shadows and coloured (`sensorNoise`), unlike film's grain, and moves with each frame drawn as the
  grain does.
- **Checked:** `tests/test_lens.cpp` (the centre and the corners stay, nothing outside the frame is
  read, the middle's and the corners' magnification, the fringes' sides and sizes),
  `TonemapLooks.ASensorClips*` and `TonemapLooks.SensorNoise*` in `tests/test_hdr_output.cpp`,
  `FvLighting.ALooksCameraGoesToTheLensAndTheSensor`. Seen on a grid of glowing boxes rendered
  headless: the rows bow outward, the edges fringe red and blue, the black around them is noisy. Its
  cost is not measured.

## The hover's outline

With the hover on (`view.set`'s `hover`, `GltfRenderer::setHover`), what is under a resting pointer
is outlined, beside the selection's outline and by the same silhouette pass, in a pale colour and
under the selection's.

- **What is under the pointer:** the app says where the pointer rests (the input event `hover`);
  the renderer makes a ray pick there with the click's picker, when no click's pick is waiting or
  in flight, and again when the camera has changed under a resting pointer. The pick goes to the
  view host, which names the entity to outline (`fv::hoverItem`): a prop whole (a character with
  its hair and what it wears), a set by the node hit.
- **It is what a click selects.** An object pick selects the same entity, so on a set the selection
  is the part hit, not the set. The host puts the gizmo on such a part where `node.set` can move it
  (`FvHost::Impl::movableNodeAt`), a drag writing the part's own transform as one step of the
  history, and sits it on the part's origin, or on its bottom middle where the origin is off the
  part (`nodePivot`: a set's parts are often modelled in place, their origin the set's).
- **No frame is traced for it.** The outline is drawn from the selection image's object IDs, which
  a moving DLSS frame does not trace unless something is selected. The hover reads them only once
  the view rests (`SelectionWrite::read`): the first resting frame traces them once, a moving frame
  pays nothing, and while they are stale nothing is outlined. A change of what is outlined changes
  the image without a frame drawn, so the presenter counts it (`GltfRenderer::shownVersion`).
- **Not through a bent lens:** the object IDs are the rendered image's, so a frame the lens bends
  draws no hover outline and makes no hover pick, and the hover does not straighten the picture as
  the other overlays do.
- **Checked:** `SelectionWrite.TheHoverReadsItOnceTheViewRests`,
  `FvSceneMap.AHoverTakesACharacterWholeAndASetByItsNodes`,
  `FvHost.aClickOnASetSelectsThePartAndTheGizmoMovesIt`. Seen through the host on a one-figure
  scene: an outline round body, clothes and hair as one with the pointer on the figure, none with
  it on the backdrop or gone, and the still's samples climbing through all of it.

## The camera's flash

A lens can carry a flash (`FvView::Lens::flash`, the view in use's: the free view's or a camera
prop's; `camera.set`). It is a scene light like any other, so the path tracer has nothing of its
own for it: the host keeps a spot light on an entity of its own just above the lens, looking where
the camera looks (`fv::lighting::flashPose`), and moves it when the camera has moved
(`FvHost::Impl::applyFlash`, every frame; only what changed is written, so a still view's still
refines). The light is made the first time a flash is on and hidden while it is off, so a scene that
never used one carries no light for it. Its strength is in a spot light prop's units.

- **Checked:** `FvLighting.TheFlashSitsOverTheLensAndLooksWhereTheCameraLooks`. Seen through the
  host on a one-figure scene under a dim studio: the frame brighter with it on, still lit from the
  camera after a quarter turn round the figure, and back to the picture without it when off.

## Bloom

A lens's glow round bright light, and a star when the aperture has blades (`src/bloom.hpp`,
`shaders/bloom.slang`, the kernels in `shaders/bloom.h.slang`), on the image the tone mapper and the
HDR output take, after DLSS, the denoisers and the hand-off; an .exr export keeps the image without
it. Off by default (a camera look, against the physical-accuracy target): the View tab's Bloom and
Star (the lighting's `tonemap.bloom` and `tonemap.star`, protocol 40; `bloomShare`, `bloomStar`).

- **The glow keeps the light:** `bloom` of each pixel's light (0 to 0.25) moves into a sum of
  Gaussians: a pyramid of halved levels (Jimenez's 13 bilinear taps) down to an 8-pixel side, at
  most 8, summed back up through a tent with each coarser level 0.65 of the finer one's weight, so
  the glow has a long tail. A glow is as strong as the light it comes from: a lamp seen head-on many
  times white glows, a sign at white barely, and the rest of the frame keeps its look. The pyramid
  is in the tone mapper's white (its exposure), each tap's largest channel held under 4096 of it, so
  a sun disk's halo doesn't swamp the frame and half floats hold it.
- **The star:** spikes across the aperture's straight edges, as a polygon's diffraction draws them:
  one line through each pair of parallel edges, so an even number of blades gives as many spikes and
  an odd number twice as many; none for a round aperture. Drawn on the quarter-size level, 32 taps
  each way along each line, a spike 8% of the frame's height, normalised, so it too moves light
  rather than adds it. Its share of the glow is `star`; the lens's blades and rotation come with
  `setDepthOfField`.
- **Checked:** `tests/test_bloom.cpp` runs the pyramid on the CPU: a bright point keeps its light
  to 1e-4 with the glow and with a six-blade star, the star's spike outshines the glow between
  spikes, the kernels sum to one, the spikes cross the edges. Measured on 1TEST at 3840x2160 while
  orbiting: 0.70 ms a frame for the glow, 0.92 with a six-blade star, against a 150-175 ms path
  trace. Seen on 1TEST's Camera 2: the blue spotlight's halo and the downlights' six-pointed stars.

## Local exposure

Dark regions of the frame lifted and bright ones held back toward mid grey, each region as a whole,
as a photographer dodges and burns a print (`src/local_exposure.hpp`, `shaders/local_exposure.slang`,
the grid and the curve in `shaders/local_exposure.h.slang`), on the image the tone mapper and the HDR
output take, after bloom; an .exr export keeps the image without it. Off by default: the View tab's
Lift shadows and Hold highlights (the lighting's `tonemap.localShadows` and `tonemap.localHighlights`,
protocol 41; `localExposureShadows`, `localExposureHighlights`), 0 to 1 each.

- **The regions:** a bilateral grid (Chen, Paris and Durand 2007; Unreal's default local exposure).
  The luminance of each 2x2 block, in stops from mid grey after the tone mapper's exposure, is summed
  into a grid of cells (40 down the image's height, even-sided, so the same share of the frame at
  every size) by 32 bins of 0.75 stops each, from -16 to +8; the grid is blurred 1 2 1 across cells
  and bins, and each pixel reads it back at its place and its own luminance. That is the level of
  the region it is in, a region ending where the light jumps by more than a bin or two, so a dark
  face against a bright wall is one region and the wall another: no halo forms at the edge. The
  blocks are a quarter of the splat's work; their one cost is the block across an edge, a sample
  between the two sides, which moves the brighter side's gain by at most 0.07 stops within 8 pixels
  of an edge both of whose sides are past the knee.
- **The curve:** within a stop of mid grey nothing moves (the knee bends over a stop); beyond it a
  region is brought back by 0.6 times the slider of each further stop, at most 3 stops either way.
  A region's new level never falls as its old one rises (the slope is at least 0.4), and every pixel
  of a region gets the region's gain, so texture and detail keep their contrast. Lift alone never
  darkens, Hold alone never brightens.
- **Why not exposure fusion:** Mertens' fusion of three exposures, the review's first suggestion,
  was prototyped on 1TEST first. Its curve for a flat region flattens the mid-tones (five stops into
  one at full strength, a third of their contrast), several weightings let one tone overtake another,
  and even with weights that follow the curve above it darkened 4.6-7% of the pixels (deep shadow
  beside brighter ones) when asked only to lift. That overshoot is the method's own: the fine detail
  comes from the lifted exposure, the region's level from neighbours lifted less. The grid lifts
  none of them darker.
- **The firefly clamp:** indirect light is clamped at 16 times white, which the curves show as white
  anyway. Held highlights can bring a large region that bright back under white, so a clamped caustic
  pool could read a little flat at a strong Hold; a single clamped pixel in a dark region is lifted
  with its region, not held.
- **Checked:** `tests/test_local_exposure.cpp` runs the grid on the CPU: a flat image gets the curve's
  gain to 0.01 stops, a step from -5 to 0 (or +4) stops gets each side's own gain up to the edge
  pixel (but for the blocks' fringe above), a half-stop checker in a lifted region keeps more than 80% of its contrast (a curve on each
  pixel would keep 40%), and the curve keeps tones in order. On the GPU (1TEST, Camera 2, the tone
  mapper set to Clip so the screen is the exposed light) the image matched a numpy copy of the pass to
  8-bit rounding (mean 0.34 levels, at most 2, as with it off). Its cost while orbiting: at 3840x2160 1.2-1.5 ms a frame (the splat 0.53, the apply 0.66-0.97), about 1% of the ~135 ms path trace; at 2160x1395 0.35 ms in its cleanest round (splat 0.19, apply 0.15) and 2.6-2.9 ms in three others, while the tone mapper's own time there varied 0.5-1.2 ms: the noise is outside the pass and was not tracked down.
  The profiler has a section for each pass, and a "Local exposure wait" one before them: a section
  starts at the top of the pipe, so the first after a barrier would count the wait for the work
  before it. The first splat, every pixel on its own, took about 0.6-0.9 ms at 4K; summing a wave's
  pixels of one bin before the atomics (with WaveActiveMin: the Slang profile here has no ballot)
  was slower, 1.7-1.8 ms. The 2x2 blocks took it to 0.53 ms.

## The exposure adapting to the view

The exposure the host sets is for the light there is (the sun's height, the sky, a physical camera's
settings); it opens up for a dark view and closes down for a bright one as an eye takes to a barn on a
sunny day, metered from the frame (`src/exposure_adapt_vk.hpp`; the arithmetic and its constants in
`src/exposure_adapt.hpp`). Off by default: the Camera tab's Adapt to the view (the lighting's
`tonemap.adapt`, protocol 84; `tmAdapt`), in Simple and Auto, not Manual.

- **The meter:** nvpro's tone mapper histogram (`nvshaders::Tonemapper::cmdHistogram`, the Histogram
  pass alone) of the image the tone mapper takes, before the lens, bloom and local exposure, counted
  more at the frame's middle, in a range about mid grey at the host's exposure. It goes into a
  host-visible slot per frame in flight and is read at a later frame's start, as the adaptive stop's
  pixel count is. nvpro's own auto exposure is not used: it puts the view's median at mid grey, which
  would move every scene, and counts what is under its range as black.
- **The reading and the adaptation:** the mean of the middle half of what is lit, in stops from mid
  grey; within a knee nothing moves, so a view the exposure suits keeps it; past it a share of each
  stop is made up, never all (a dark room still reads dark), up to a limit each way; eased over
  seconds, faster into the light than into the dark, at any frame rate.
- **Where it acts:** a factor on the host's exposure, `Resources::exposureAdapt`, applied by
  `Resources::shownTonemapper`, so the tone mapper, the HDR image, the still's adaptive stop and its
  denoiser's threshold, local exposure, an .exr and the pixel readout all take the exposure as
  adapted. The path tracer's firefly clamps, set for the host's exposure, are divided by it.
- **Checked:** `tests/test_exposure_adapt.cpp` (the meter's range against the shader's binning, the
  reading, the curve, the easing at three frame rates); `tests/test_fv_lighting.cpp` for the host's
  part. Not yet run on the card.

## The Shaped tone curve

A film curve with a toe and a shoulder to set (nvshaders `tonemapShaped`, tone mapper method 6,
`tmShapedToe`, `tmShapedShoulder`; the lighting's `tonemap.method` "shaped", `tonemap.toe` and
`tonemap.shoulder`, protocol 43): the View tab's Tone curve › Shaped, with Toe and Shoulder under it.
Our own curve, not Unreal's (`FilmToneMap` is engine code under its EULA): the same idea, toe and
shoulder handles on a filmic curve, worked out here.

- **In stops:** stops over mid grey in, stops over mid grey out. Mid grey (0.18) comes out at 0.14
  with a slope of 1.3 stops a stop, as Khronos PBR Neutral (the default curve) gives it, so the
  exposure means the same under both and the host sends it unboosted (ACES alone gets 1/0.6).
- **The shoulder:** a rational quadratic Bezier from mid grey, along that slope, to white at 16,
  which it reaches flat. Its middle point's weight is the Shoulder: 4 at 0 (highlights bright and
  rolling off late: within 0.03 of Neutral from mid grey up), 1 halfway (between ACES and AgX), a
  quarter at 1 (rolling off from just over mid grey, gently). A Bezier inside its control polygon
  cannot fall, so the curve rises for every weight, and it is white by 16, which the firefly clamp's
  "white anyway" counts on (`fv_lighting.cpp` kFireflyClamp).
- **The toe:** under mid grey the same slope, steepened past 1.5 stops down (a soft knee a stop
  wide) by up to 1.5 stops a stop more: Toe 0 leaves the shadows open, 0.5 about as Neutral's, 1
  crushed. Each handle moves only its own side of mid grey.
- **Colour:** the brightest channel goes through the curve and the colour is scaled with it, keeping
  its hue, then eased to white by how much that channel was compressed (Neutral's way, at Neutral's
  rate by default; see below), so a bright red pales toward white instead of turning orange as a
  curve on each channel would.
- **HDR:** its result is opened toward the display's peak above half of SDR white, as for ACES and
  AgX (`hdrOpenShoulder`).
- **Checked:** `tests/test_hdr_output.cpp` (ShapedCurve.*): mid grey and white where they belong for
  every toe and shoulder, rising with no step from -14 to +6.5 stops, the same slope either side of mid
  grey, each handle moving its own side only, Toe 0.5 with Shoulder 0 within 0.03 of Neutral (sRGB)
  from -4 to +4 stops, the hue kept and a bright red eased toward white, and the tone mapper choosing
  it by its method with its handles. Prototyped in numpy first (scratchpad shaped2.py): a curve of
  Lottes' form (x^a over b x^ad + c) was tried and dropped, as its shoulder handle moved the shadows
  too (at -4 stops 0.019 to 0.078) and without it the shoulder barely moved anything.

## Highlights to white

How fast a colour Khronos PBR Neutral or Shaped compresses goes white (nvshaders
`highlightDesaturation`, `tmHighlightsToWhite`; the lighting's `tonemap.highlightsToWhite`, protocol
63): the View tab's Image › Highlights to white, under either curve. Both curves ease a compressed
colour toward white by how much its brightest channel was compressed, at a fixed rate (Khronos's
`desaturation`); the handle sets that rate, cubic in the handle so its middle is Khronos's own exactly
and a picture made before it is unchanged. At 0 a neon tube or a lit red curtain keeps its hue to the
top; toward 1 bright lights burn out white with a coloured edge, as a sensor's do. Under the knee
nothing moves. The HDR display's Neutral (`hdrNeutral`) takes the same rate, so the HDR image is the
SDR one opened toward the peak at every setting. ACES, AgX and Cineon have their own and ignore it.
Checked: `HighlightsToWhite.*` in `tests/test_hdr_output.cpp`, `FvLighting.HighlightsToWhiteGoesToTheRenderer`.

## Auto white balance

`render.whiteBalance` (protocol 63; the app's Image › Auto white) answers the white balance that makes
the light the next frame was lit by grey; the app sets the sliders to it. The frame's tonemapper input
is read back through the EXR export's pass with the tone mapper off, so before exposure and white
balance (`GltfRenderer::readTonemapInput`, `updateLightRead` in `onPostRender`, as a screenshot is
saved). `src/white_balance.hpp` holds the rest, Vulkan-free:

- **The light** (`estimateLight`): an iterative grey-pixel average over about 512² pixels spread over
  the frame, the brightest 2% (lamps, the sun, neon, clipped highlights) and the near-black left out.
  Each round weighs a pixel by how grey it looks with the light found so far taken out, so a surface
  only the light colours counts fully once the light is known and one coloured of itself (a red curtain,
  a blue sofa) stays discounted.
- **The setting** (`forLight`): Gauss-Newton on mireds and Duv against nvshaders'
  `getColorCorrectionMatrix` itself, the matrix the tone mapper applies, so the answer makes the light
  grey whatever that matrix's colour science; clamped to the renderer's range, and said so.
- **Checked:** `tests/test_white_balance.cpp` (a light made from a setting gives that setting back; the
  answer makes off-locus lights grey; a frame two-fifths red curtain with a bright blue sign gives its
  3000 K light within about 140 K, where the plain average is far off), `FvHost.WhiteBalanceAnswersTheFramesLight`.

## The procedural backdrop

Land around the scene, out to the horizon. It is a state of its own, one `fv::backdrop::Spec`
(`src/fv_backdrop_state.hpp`): `backdrop.set` patches it as a step of the history, and the `backdrop`
topic publishes it (until protocol 70 it was a section of the lighting). It meets the lighting in two
places only: it stands at the lighting's floor height, and its renderer parameters go over the
lighting's (`fv::backdrop::rendererParams`, `backdropParams`). Two kinds, the same
machinery with other numbers: a meadow (grass and flowers, bushes, a few rocks, firs in groves that
close into a wood round it, wooded hills beyond) and a forest (a glade in a fir wood that runs to
the horizon, on a floor of needles with ferns, tufts of grass, bushes, mossy rocks, stumps and dead
wood). What a kind is made of is `layers`, `grassCovers` and `drifts`. Two halves:

- **The shape** (`src/fv_backdrop.hpp`, Vulkan-free): `height` is the ground, flat where the scene
  stands, rolling beyond it, rising to hills toward the horizon of a camera on the ground
  (`kHorizon`) and, past those, to ridges some hundreds of metres high by the land's end
  (`kLandsEnd`), kilometres off: a camera in the air looks out over land to the haze and not at a
  disc's edge. `terrain` is its mesh, a disc of rings that widen with distance, so the triangles are
  small underfoot and large far off, and fine enough that the creases between its triangles stay
  under the grass's blades; what colours it (the drier drifts, the far woods) changes over no less
  than a few rings, or the mesh would draw it as blocks: its patches grow with distance as the rings
  do (`kPatchGrowth`; `sizedNoise`, the noise at doublings of its size, blended, since a size that
  simply grew with the place's distance would leave a function of its direction, stripes out from
  the origin), their noise keeps no octave finer than four rings (`meshDetail`), and the woods' edge
  is four rings across by the noise's own slope there. Sharper, a colour flipped from one vertex to
  the next and was drawn as a hard-edged triangle tens of metres across (checked by
  `BackdropGround.TheGroundsColoursChangeOverSeveralVertices`). The trees' own groves (`scatter`)
  keep every octave. All of it
  is a pure function of the spec: the same seed is the same land on every machine. What stands on
  the ground stands on the mesh (`groundAt`, the height on its triangles), not on `height`: the
  height's bumps, some ten metres across, bulge through triangles metres wide a few tens of metres
  out, and grass laid on `height` was buried to half its blades on their ridges, pale squares from
  the air, and floated over their hollows.
- **Trees, saplings, bushes, rocks and dead wood have places** (`layers`, `scatter`): a jittered
  grid thinned by patch noise, per variant of the role. They never move, but to sway in the wind (below). The trees stand out to
  the horizon, so a camera that flies off or climbs sees wooded hills and not bare ground: a second
  tree layer carries the first on from where it ends, thinning with the distance past that
  (`Layer::thinFrom`) while each tree grows by the root of it, so the far hills keep their cover
  with a few thousand trees, grown as old firs are. A meadow's far layer starts as closed as its
  near one ends (`closeNear`) and opens into woods and pasture: from the air the wood round the
  meadow has no outer rim. Where growth is spent the trees thin out to nothing (`kTreesEnd`), and
  the wood carries on as a colour of the ground (`woods`, the kit's `tintWoods`) in the same
  groves, which grow with distance as the mesh does (`Layer::patchGrows`): the last sparse trees
  stand on ground already dark and read as a wood's edge. **Mature woods** (`Spec::mature`) add one layer after a kind's own, so none of theirs moves: grown firs
  (the kit's `grown` role, one 14 m tree of 2.3M triangles the app fetches only for it) at 1 to 1.5
  times, a few hundred among the near woods' young ones. The far woods stay young firs grown with
  distance. All the trees, every one a grown fir, had cost about half again a frame (2026-11-11);
  near only, they add some 140 MB (the kit's textures compressed) and, flown in the forest at 1920 x
  1080 on an RTX 4080 with the moving view's settings (2026-11-12), 0.4 ms to a frame's 8.1 ms of
  path tracing walking through it, 0.3 ms from 250 m up, and 1.3 ms (16%) flying over the canopy.
- **Grass and its drifts are tiles** (`grassCovers`, `drifts`, `patch`, `grassTiles`, `driftTiles`, `tilePlace`):
  a square of ground with its few hundred plants is one mesh, and the ground is covered by squares
  of three kinds, thick grass in 2 m tiles where one looks from, thinner and larger in 4 m tiles
  beyond, sparse tufts in 8 m tiles out to where the trees have closed; a tile of one level is four
  of the next, so no ground is covered twice or left out. "Where one looks from" is two places, the
  scene's content and the camera, so there is grass under any view; each is a point with its height
  over the ground (`Focus`), and a level reaches that far from the eye, so a camera in the air has
  the thick grass under it only once it can see its blades, and none above the last level's reach.
  The ground under the grass has the grass's colour as seen from above (the kit's tints), so where
  the grass thins, from one level to the next, it doesn't turn paler. A move of the camera changes
  the few tiles at the levels' boundaries, tens of metres off, and nothing near it; a cell that
  split stays split a little further than where it splits, so a view hovering at a boundary
  changes nothing. What stands in drifts (a meadow's flowers, a forest's ferns) is tiles on the
  middle grid where a drift noise says. A forest's grass is a seventh as thick and in clumps
  (`Cover::clumps`): plants that sparse, spread evenly, show the grid they stand on.
  It is also darker and yellower than the kit's meadow green (`Cover::tint`): its tiles draw with
  tinted copies of the grass's materials, the light the blades pass on tinted with what they
  reflect, which a vertex colour would not do.
- **The planting** (`src/fv_host_backdrop.cpp`): the plants are files in a kit folder the app fills
  (`settings.apply {backdropKit}`), listed by its `kit.json` with a role each (`readBackdropKit`, read
  on a loader thread), and by a `kit.local.json` beside it when there is one: plants of the user's own
  that may not be passed on with the app (BlenderKit trees made by the app's
  `scripts/blender/blenderkit_trees.py`), which the app's download leaves alone and which stand beside
  the kit's of their roles (`fvhost::kitPlants` reads either list). Under one host-owned root entity each plant file is imported once and each of
  its mesh nodes is a variant. Only the files of the roles the kind plants are in the scene
  (`fv::backdrop::roles`, `holdBackdropPlants`): a model's maps are held uncompressed, some 17 MB
  a file, so a meadow doesn't hold the ferns and stumps nor a forest the flowers. A tree's variant
  is drawn at all its places through the mesh's
  `copies`, the instancing path. A tile's mesh is the grass's variants merged at their places in it
  (`makeBackdropPatch`), held by an entity that is never shown; each tile drawn is an entity with
  that mesh, so one bottom-level structure serves every tile of a level, and laying the tiles
  (`tileBackdrop`) shows, moves and hides entities, keeping the hidden ones for the next tiles.
  `followBackdrop` lays them again each time a focus has gone a metre.
  The ground is a model built in memory from `terrain` with the kit's ground maps. A tiled map
  is one colour from far off, and a photographed grass ground is mostly soil: the kit gives two tints,
  mixed by the terrain's `shades` (drifts some tens of metres across) into vertex colours, which bring
  the ground between the plants to the grass's colour as it renders.
  A change of the state plants again in place (`plantBackdrop`: the copies and the tiles' meshes made
  again); a change of kind, seed or relief rebuilds the ground too, and another kind lets go of the
  plants only the last one used.
- **Clear of what stands in the scene** (`fv::backdrop::Standing`; `readStanding`, `clearBackdrop` in
  `src/fv_host_backdrop.cpp`): the land is the one part of the backdrop that does not follow from
  the spec alone. The triangles of everything shown that is not the backdrop's own, a figure's skin
  or a strand of hair are laid on the plan of the ground as a grid of cells (an instanced mesh's
  copies by their boxes), and three things follow. The ground is level as far as the props reach
  (`Spec::clearing`, which the host sets by `levelFor` and no patch or file carries), rising into
  the same hills beyond, so a house put down forty metres out is not half in a hillside. Nothing
  with a place stands where something is over the ground within its own width (`clearOf`): no fir
  in a house, against a fence or under a pipe. And a tile of grass is left out where most of it is
  covered low down (`tileTaken`): a floor, a slab, a car, but not a roof with no floor under it nor
  a wall, which is over its line and covers nothing. A grid, not the props' bounds: a fence round a
  yard is one mesh whose box is the yard. A tile is one mesh, so the grass ends at tiles' edges: up
  to a tile's width of it reaches in under a floor's edge or is missing outside it. That width is a
  thick tile's at any distance: a cell that something covers a part of is split down to the thick
  tiles however far it is from where one looks (`grassTiles`' `standing`), since a road is narrower
  than a coarse tile, none of which would be left out, and from afar the grass grew over it; what
  stands in drifts (flowers, ferns) is left out at a smaller share (`kDriftTaken`). The grid is read
  again when the scene's entities that draw have moved and then stood still (their places and
  meshes hashed, `standingKey`, asked a couple of times a second while the scene's bounds change),
  so a prop dragged or a ship in flight plants nothing until it rests; then the places and the tiles
  are made again, and the ground and all of it when the reach moved to another ten metres. A planet
  does the same (`clearPlanet` in `src/fv_host_world.cpp`): its ground is level as far as the props
  reach and its site is good for (`fv::world::Spec::clearing`, `mostLevel`), on the plane of the
  scene's floor and not on the sphere, which has fallen a hand's breadth under that plane a hundred
  metres out; a quad's plants are left out where something is over them as it arrives, and a tile of
  grass where a floor lies. Its quads are asked for again (`restreamPlanet`), the same planet with
  another clearing (`withClearing`: nothing is baked again), and what drew stays until the first cut
  of the new is whole, so the land never goes.
- **Why tiles.** Each grass plant an instance of its own was built first, thinned by its distance
  from the camera. Following the camera then meant some 200,000 instances made again: 50 ms to
  place them and another 160 ms in the renderer's sync, a fifth of a second up to two and a half
  times a second in flight, with every plant near the camera changing size as its distance did.
  With tiles the same flight's worst gap between frames is 30 to 50 ms and none is longer.

**The wind** (`Spec::wind`, `windFrom`): the trees,
saplings and bushes sway, each turned whole about its base (`swayOf`, `swayedPlace` in
`fv_backdrop.hpp`), its top leaning downwind and swinging at its role's rate with a phase of its own,
harder where one of two gust waves rolling over the land at the wind's speed is passing. A plant is
one rigid copy of a shared mesh, so its trunk bends with its crown and no branch flutters; a fir's
lean is under a degree at 30 km/h, a bush's a few. Each frame in a wind the host turns the places
of the variants with any near the camera (`swayBackdrop`, beside `followBackdrop`; `swayFrame` and
`swayed` turn the drawn copy directly) and hands over only those, and those that swayed last time and
no longer stand near, back at rest (`RetainedScene::moveCopies`): a transform change, so the residency
moves their rows and keeps last frame's for DLSS motion, and no BLAS is touched. Past its role's `Sway::full` a plant's sway eases off, and past its `end` it stands,
its row unchanged: a tree's reach is some hundreds of metres, a bush's a hundred, where their tops'
centimetres come under a pixel. The grass, in tiles of one mesh, stands still. A scene that changes every frame is a
moving one: while the wind blows the still waits. It plants nothing (`samePlanting`).

The water moves in the same wind (`fv::backdrop::waterMotion`, `moveWater`,
`HostRenderer::setWaterMotion`): the ground's body of water and its puddles, and a planet's sea and
lakes, which have no plants that sway. The frame carries a moment and a heading
(`SceneFrameInfo::waterTime`, `waterHeading`), which the path tracer's entry points hand to the
water's functions (`groundWaterMotion` in `shaders/ground.h.slang`): a wave train's phase runs at
the rate its length gives it over deep water, and the noise that roughens the trains and the
ripples on them drift downwind (`groundDrift`, `GROUND_RIPPLE_DRIFT`). The waves go downwind calm or not; in a calm the moment is 0 and
the water stands, so a still refines. A new moment starts the image again only where there is
water to draw. The moment is the host's wind clock wrapped (`kWaterClockWrap`) for a float's sake.
DLSS is told the water's look moves with its waves ([denoising.md](denoising.md#motion-vectors--what-is-and-isnt-captured)),
for which the ripples drift as one field and the frame carries the step from the frame before
(`waterStep`).

Measured (2026-11-12, RTX 4080, 1080p, the moving view at rest on the ground, 40 km/h): the meadow
17.5 ms between frames, as calm; the forest 21.5 against 17.5, path tracing unchanged (11.1 ms). The
forest's is the sync's CPU, which a change anywhere in a scene of its size paid then (since
2026-11-15 a sync looks only at what changed: "What a sync looks at", above). Before the bushes and saplings had a reach of their own, 15,000 copies swayed there and the
frames came every 33 ms. The user's forest (size 200, density 2, Mature woods) draws 190,000
instances and sways 40,000 of them: the sync took 25 ms a frame and the host's sway 3.6, frames every
44 ms. The TLAS rows are kept from build to build and only the moved or changed ones written
(`buildTlasRows`: 6.1 to 1.8 ms), the moved slots are lists sorted once, not sets (the compare 6.4
to 1.3), a copy under an entity that only translates is offset rather than multiplied (the gather
4.4 to 3.3), and the sway turns only the near copies with what a frame shares worked out once (3.6 to
2.3): frames every 30 ms against 17.4 calm. What is left is about a millisecond or two each: the
gather, the motion and render-node rows, the TLAS rows, the sway.

**Where it ends.** The ground stops at `kLandsEnd`, and past it is only what the physical sky
draws under its horizon: the planet, given the land's own colour (the kit's `albedo`, sent as
`atmoLandAlbedo`). Between `kLandFadeFrom` and `kLandFadeTo` what the camera hit gives way to that
(`landFadeFrom`, `landFadeTo`), so the land has no rim; see [sky.md](sky.md#fixed-observer) for
why the two do not simply meet. Under another environment than the physical sky the land ends at
its edge.

**The air over it** (`Spec::air`): how many metres of air each metre
of the land counts for under the physical sky, sent as `atmoAerialPerspectiveScale` while a backdrop
stands and 1 without one (see [sky.md](sky.md#how-much-air)). It is what gives the land its
distance, and it plants nothing (`samePlanting`).

**How bright the plants are** is the kit's doing, in the app (`backdrop_kit.rs`, `Leaf`): Poly
Haven's leaf cards are matte and pass no light on. Against an 18% grey ball in the same sun (linear
stills), the meadow's grass as it came reflected 3.4%, 2.4 stops under the ball and as red as
green, where a real meadow is about a stop under. Its colour map is a real leaf's brightness (9%),
so brightening the colour alone could not make that up without blades lighter than any leaf (2.8
times it gave 6.3%); a real leaf passes on about as much as it reflects. With the leaf materials
given that (`KHR_materials_diffuse_transmission`) and a greener colour the grass is 7.4% seen
steeply and 8.6% along the ground, at the same cost a frame (8.0 against 8.1 ms looking down into
it). The firs were where real conifers are, under 5%, and only lose some red.

What gives way while a backdrop stands: the flat ground plane is off (`useInfinitePlane`; it comes back
while the kit is missing or loading), and "Auto" ground, framing the scene and the fly speed measure
the scene without it (`RetainedScene::bounds(except)`, `contentBounds`), since its ground reaches the
horizon. The haze is scoped the same way (`setHazeScope`, `scopeHaze`): its sphere is round the
scene's bounds, and round a 900 m ground the sky was seen through a kilometre of haze that only the
scene's lights light, which left a midday sky dark grey; so the distance is not hazed. The root sits
at the floor's height, and the ground is flat within the clearing, so the physics floor and Drop
still meet it. The `backdrop` topic says `state` (off, missing, loading, ready) and `plants` beside
the spec. Only a backdrop that stands sets its parameters: one asked for whose kit is missing changes
nothing of the picture.

Measured on an RTX 4080 at 1080p, the moving view, with the app's kit (Poly Haven's grass, two
flowers, a fern, a bush, fir saplings, three young firs, two sets of mossy rocks, two stumps, a
trunk and dry branches, 1K maps), by the origin, against 2.4 ms of path tracing on the bare floor
under the same sky:

| | Path tracing | From 250 m up | GPU memory | Stands after |
|---|---|---|---|---|
| Meadow | 14.3 to 15.2 ms | 13.9 to 15.9 ms | 635 MB | 0.4 s |
| Forest | 16.3 to 17.8 ms | 19.2 ms | 540 MB | 0.3 s |

A forest's cost is its trees: without them the same view is 6.5 ms, without the undergrowth 0.4 ms
less, without the grass and ferns 1.7 ms less. The firs are 0.4 to 0.7 million triangles each with
alpha-tested twigs, and a view in a wood is all firs; half as many of everything still costs
15.7 ms, so the count is not what it turns on. The trees out to the horizon cost a meadow's ground
view nothing measurable (15.2 ms with them, 15.1 before). The meadow's bushes and rocks are 85 MB
of its memory, their maps mostly.

**What the trees cost, and why** (2026-11-11, the moving view as the app sends it, 8 bounces, flown
at 15 m/s; a walk through the forest's glade, a flight over its canopy, and from 250 m up): 14.2,
15.0 and 15.0 ms, of which the trees were 9 to 10 (5.2, 7.4 and 5.1 without them). Not their
cut-outs (twigs made opaque: 14.3, 14.9, 14.8), nor their triangles (a quarter of the twig cards,
each grown to cover as much: 12.9, 15.1, 11.9), nor the bounces past the fourth (4 or 8 the same),
nor the camera rays (one bounce: 3.8, 5.7, 4.1 against 3.2, 3.4, 3.0 without trees). It was the
second and third bounces: a ray off a needle, which reflects a sixth of the light, went on through
the canopy, and the next one too, since the roulette started at the third bounce. A DLSS frame now
starts it at the first on a cut-out surface (`path_roulette.h.slang`, `ptCutoutRouletteFromDlss`):
the forest 8.8, 12.4 and 9.4 ms, the meadow 7.0, 10.1 and 8.6 (from 10.2, 14.4 and 12.6), the moving
image's brightness within 4% in every band from its shadows to its highlights, on a figure in the
glade too, and its flicker within the run-to-run spread. From the camera ray's own hit on a leaf
(0) it was cheaper again (5.7 ms in the glade) but darkened the glade's shade by 6-7%. Stills keep
the third. The near trees (to 160 m) are where the cost was from the glade, the far ones
from the air. On every surface it was cheaper still (7.9, 12.7 and 8.5 ms here, and a third off a
night club's and a night street's moving frames), but the night club's shadows came out 5 to 12%
darker than the still's: Ray Reconstruction holds back the few survivors that carry a lamp's
bounced light.

The land past its hills, the far trees and the sky from the camera's height cost a frame nothing
that could be measured (meadow 14.4 to 14.7 ms by the origin, forest 16.7; from 250 m up 15.4 and
21.0), and 46 MB (681 for the meadow). From 700 m up, looking out, a frame is 5 to 7 ms of path
tracing; from 2 km, 4.4.

A flight of 120 m at 15 m/s over the meadow (measured before the far trees, which a flight does
not touch): the gaps between frames 18.5 ms at the median (17.3 over the bare floor, the stream's
60 a second), 29 at the 99th percentile, 32 to 49 at worst. Before the tiles, plants as instances:
400,000 alpha-tested grass clumps 8.4 ms, about 0.8 KB of GPU memory an instance.
Checked: `tests/test_fv_backdrop.cpp` (the shape, and `BackdropState.*` for the state),
`FvLighting.TheBackdropIsNoPartOfTheLighting`, `FvHost.ABackdropIsPlantedFromItsKit`.

### A planet

The `planet` kind is not land round the scene but a small world under it (`src/fv_world.hpp`), and
shares the backdrop's state, root entity and renderer parameters with the two planted kinds, and
their kit when there is one. It needs no kit: without one it stands bare.

- **What it is** is a function of a point on the sphere: `fv::world::sample` runs an ordered list of
  stages over a sample (the continents and the sea's floor, mountains, hills, the tiers the land
  stands in with cliffs at their edges, the climate, the biome, the ground's fine relief, its
  colour). A biome is a row of a table. Every stage takes the size of what will draw the sample and
  adds nothing finer. A stage may not scale or switch what another adds by anything that changes
  within a few metres: the ground then steps along that line, which a mesh draws as a row of teeth
  (`WorldMapBake.DISABLED_Steps` in the tests finds a step and names its stage).
- **What a point alone cannot say** is baked once a planet over the whole of it, on a cube map
  (`BaseMap`, `src/fv_world_map.hpp`, `bake`): the rain the wind carries in from the sea, the hollows
  filled to where they spill (the deep ones hold lakes), what the running water wore off the land
  (the stream power law, solved from the sea up), the valleys' floors filled flat across by a depth
  that grows with a river's water (no simulation; none by the sea, none over a lake's brim), and the
  rivers, kept as curves with a level and a width. Three stages read it: `base` adds the wear to
  the land and finds the water near a point,
  `gullies` furrows the steep slopes finer than the map's cells, `water` cuts the rivers' beds and
  says where water lies. The gullies are no simulation: stripes laid down the land's fall as the map
  has it (`BaseMap::slopes`, the worn land's slope at a few breadths), each size steered by the sides
  of the one before so they branch, after the erosion filters of Clay John, Fewes and Rune Skovbo
  Johansen (written here from the idea, as phasor noise in space). A road's level is found without
  them (`sampleWithout`): it is cut and banked through them. The map's cells are tens of metres and
  the ground is drawn to a metre, so what is drawn is never the map's cells: a lake's shore is where
  the ground comes up through its level, a river's bed is cut from its curve. A river's level is the
  ground's as it is drawn (the stages' land finer than the cells, the wear's, the gullies' that run
  into it: `gulliesAt`), followed down and never up, so it cuts through what stands in its way and
  never runs on a bank of its own; its banks are a valley's sides eased into the land, and on a
  gentle floor its water meanders off its curve. `WorldMapRivers.DISABLED_Banks` counts how its
  rivers lie in a planet's land. The header says what that takes (a lake under its hollow's brim,
  rivers at the sea's level by a shore, the cube's corners). The planet is made on a worker
  (`standPlanet`), since the bake is about a second for 20 km of radius and a few for the largest; a
  flat planet (no relief) has no map.
- **What is put on it by hand** (`src/fv_world_features.hpp`): a seed grows its land from noise, and
  an author who wants a particular world says where some of it is: land or sea, a mountain range
  along an arc, a peak, a basin, a plain. Each is a point or an arc of a great circle with a width,
  and acts on the measure of the stage that takes it (`placedAt`, read once a sample by the first
  stage), so the seed's noise still fills it in. They shape the land before the base map is baked:
  the rain falls on a placed range and rivers run off it as off the seed's own. A planet file is a
  seed's numbers, these, and what it says of where people lived (`Settlement`, under the places
  below) (`parsePlanet`). The backdrop names one (`planet`, a file's name in the
  folder the app gives as `planetsDir`); the host reads it at every patch (`readPlanetFile`), its
  numbers become the backdrop's and what it places a part of which planet it is, so a file changed
  since is another planet from the next patch on.
- **Where the figures stand** is a level place on land the seed picks (`site`), or the author's where
  that is dry land. The planet is turned
  and set so that place is the origin with +y up; its centre is `Frame::centre`, a radius below.
- **What draws** is a cut of six quadtrees, one a cube face (`select`): a quad splits while the eye is
  within about three times its width (`kSplit`), so the ground underfoot is a metre a vertex and the far side a
  kilometre. `FvHost::Impl::followPlanet` (`src/fv_host_world.cpp`) moves the shown cut toward the
  selected one a step at a time, and at every step it is a whole cut: a shown quad gives way to its
  four children only once all four are made, and shown quads give way to the one over them once it
  is. An eye that moves is led: the cut is chosen for where it is and for where it will be a moment
  on (`kLead`, from its steps over the last frames), so a quad is asked for before the eye is at its
  distance. Meshes are made on the job pool a few small batches at a time, the nearest along the
  eye's way first, and come into the scene as hidden entities for a couple of milliseconds a frame
  (`landPlanet`), so their BLAS is built before they show and a frame takes in no more than it can.
  A quad's edges carry skirts, since neighbours of different levels don't
  meet exactly and a ray would find the crack.
- **The ground's colour** is its vertices' (the fields'), and **its grain** a map's where the
  backdrop's kit has ground maps (`shaders/world_ground.h.slang`, `mapPlanetGround`). The quads are
  all drawn with one material; the kit's maps make a material a layer (`fv::world::Soil`: grass, a
  wood's floor, rock, sand, snow, the dry lands' stony earth, the tundra's soil and stones, a bed's
  pebbles), each colour map divided by its own mean so that it changes the ground's grain and not
  its colour. A biome's soil is one of them, sand is laid over it (a beach, the sea's floor, a
  river's bed, of which the shallows and a damp shore are pebbles and a track's earth the dry lands'),
  and rock over all. A vertex carries its place's share of each layer (`surfaceLayers`): the first
  four in the mesh's masks, a byte each, the rest as floats in its two texture coordinates, which
  the ground has no other use for and which blend across a triangle as the masks do. The path
  tracer, at a hit of the ground's material, takes one layer by a noise over the ground and
  evaluates that material instead: a blend would be several material evaluations a hit.
  The maps have no coordinates of the mesh's; they are laid by where the hit is in the scene, on
  the axis plane the ground faces most squarely (a cliff's face from its side), the normal map's
  frame with them. The frame carries the materials' slots (`GltfRenderer::setWorldGround`,
  `SceneFrameInfo::worldGround`); with none, the ground is drawn as its mesh has it.
- **The sea** is no mesh: it is the ground's body of water (`shaders/ground.h.slang`) asked about a
  sphere (`shaders/world_sea.h.slang`; `worldSeaRadius`, `worldSeaTop`, `worldSeaWaves`,
  `worldSeaRipples`, the backdrop's `waves`). A path's ray meets its surface, the sphere at the
  sea's level, after whatever mesh it hit and refracts there by a normal the waves tilt; under it
  the path is in the water (a camera under it starts there), charged its extinction and given the
  light it scatters, sun shafts and all; a shadow ray crosses the surface straight, with the caustic
  the waves focus. The waves are in the normal and the caustics, not in the surface's shape: a
  heightfield, as the ground's water is, is walked by every ray that passes it, and compiled into
  every place a shadow ray is traced (the header has the numbers). Heights over the sea are taken
  from the point of it over the planet's centre, which is by the scene, so they keep a float's
  precision where the figures are. The header says what is approximate: the waves laid on the
  sphere from three planes, the caustic worked out on a tangent plane, the particles' forward
  scattering.
- **Lakes and rivers** are a second mesh a quad with inland water on it (`waterMesh`): their
  surfaces where they lie (`waterLevel`), cut at the ground's own points so the shore is where the
  two cross. By a bank the surface goes down under the ground, which hides it, and at a river's
  mouth under the sea. The water is the sea's all the same: the frame says which material the
  meshes are drawn with (`setWorldGround`'s `water`, `SceneFrameInfo::worldWater`), and a path that
  meets one is at the water's surface as it is at the sea's, with calmer waves; from there the
  water is asked about the sea's sphere made as much larger as the lake stands over the sea
  (`worldSeaAt`, `g_waterLevel`), so a shadow ray crosses that sphere and never the mesh. The mesh
  is no surface to a ray that meets it from the side its path is on already, nor are its skirt's
  walls or the end of a river under the sea (`worldInlandHit`), and the sea's sphere is none to a
  path that goes down through it in a lake whose bed lies under the sea's level. No sphere says
  where a lake is, so the host tells the renderer what inland water the camera is in as it moves
  (`fv::world::inlandWaterOver`, `setWorldWaterAtEye`). Left approximate: a path in a river is in
  water as high as where it went in, and a camera deep in a lake under the sea's level sees the
  sea's surface over it. `worldInlandWater` off draws the mesh as the glass it is made as, for
  comparison. A river too narrow for what draws a quad is not water there but a line of colour.
  The camera-medium probe (`ptCameraMedium`) must be off, as the app has it: its ray passes through
  the ground and can meet a lake on the far side from behind.
- **What grows** (`src/fv_world_plants.hpp`) is a biome's row too: what stands on it and how
  thickly, its grass, what drifts in its grass. With the backdrop's kit read (`readPlanetKit`,
  `plantPlanet`) the planet is planted with the plants a meadow and a forest are and with roles of
  its own (`kStandingRoles`: leafy trees, pines, the dry land's small trees and bushes, bare
  boulders, and riverside trees), each biome growing its own of them, all held the same way. What
  grows by water only (`Growth::water`: the riverside role's willows and alders) stands within so
  many metres of a river's or a lake's edge, thickest nearest it, in a belt broken into stands: the
  grassland's within 30 m, the wood's 20, the steppe's 35 (its only leafy trees), the desert's 15. A
  river is known as far as its banks reach (`Inland::bank`, 20 m by a stream), so a stream's belt
  is narrower than a wide river's; one look at the base map (`mapInland`) passes over a cell far
  from water before the ground is sampled, and the role adds some 4% to the planting. A kit with no
  riverside models of its own (the app's download has none) draws them with its leafy trees
  (`drawnAs`). A tree, a bush or a rock
  has a place that follows from the planet alone (a grid a role over a cube face) and a reach, how
  far it is seen from, by its size among its kind's (`reachOf`): a wood's largest trees for
  kilometres, the dry land's small trees for a couple, a wood's smallest and its undergrowth for a
  few hundred metres, the share seen at a distance falling as its square, so those seen from one
  place stay about as many however far the largest are seen (a wood of several kinds of tree is
  counted as one wood). A quad holds the plants whose reach passes where the quad
  can draw (`heldFrom`), each decided on the ground as the finest quads have it: a plant is the same
  plant, in its place and of its size, whichever quad holds it, and a quad taking another's place
  changes none. They are made with the quad's ground on the pool and arrive with it, a variant an
  entity. The host gives the entity as copies the plants that may be seen from where the eye is and
  shows each by its own distance (`showPlanetPlants`: hidden past its reach, by the copy's own flag,
  and growing to its size over the last of the way in, in a few steps, fewer the further it is seen
  from, so it comes into view small; a frame gives no more entities other copies, and takes in no
  more quads' plants, than `kPlantsAFrame` instances, which the scene's sync goes through), again whenever
  the eye has gone a few hundredths of its distance to the quad. The trees a quad does not show from
  where its trees are on the whole while it draws (`kCanopyAway` of its widths), the smaller of a
  wood's, stand on its ground as a **canopy** (`canopyLift`, `groundMesh`'s `Lift`,
  `World::canopied`): the quad's own mesh is raised as high as their crowns over the ground they would
  stand on, by the same rule `standing` grows them by and none where it stands none. Where the mesh's
  points are a few metres apart each one is a crown's top or the floor between it, by how much of the
  ground those crowns cover (a hash of where the point is, so two quads of a level draw their edge
  alike); tens of metres apart, the wood as a whole, its colour as much as a slanting view sees of its
  crowns. A raised point takes the canopy's colour and none of the ground's layers; the rest is the
  floor (`Biome::floor`, `World::planted`) at any distance, and the skirts hang as much further as the
  canopy stands. Nothing reads the mesh back (every query of the ground asks `sample`), so the walker,
  plants and roads stand on the ground under it. The quads round the eye show every tree and lift
  nothing; the `worldCanopy` setting turns it off for a comparison (the quads are made again). Without
  it a far wood is flat ground its canopy's colour. Grass and its flowers and ferns are the meadow's tiles, laid by the meadow's own
  rule on a cube face's plane round the camera and the figures (`tilePlanet`), each leant and
  stretched to its square of the sphere. They are laid again every few metres the camera goes, and
  what grows on a tile's ground is looked up on the planet once a tile, not once a laying
  (`GrassWas::grounds`): a view that moves asks for the same few thousand tiles each time.
- **The sea floor** has plants of its own, which need no kit (`src/fv_world_sea.hpp`): seagrass,
  kelp, corals, sponges and rocks as small meshes made from the planet's seed (the generators after
  abyssal-garden's, MIT), held hidden under the root as the kit's plants are and placed the same
  way (`seaStanding`): by the water's depth and warmth, in patches, none taller than the water over
  it, each with a reach as the land's undergrowth has.
- **The sky** takes the planet's radius, its air scaled to it, and its centre (`worldRadius`,
  `worldAirScale`, `worldCentre`), and is then asked every question from a place of the planet: what
  a ray sees from inside the air or from orbit, the sun's light at a hit, the sky's light there
  ([sky.md](sky.md#a-small-planet)). Clouds, rain and the haze are laid out on a level plane round
  the origin and are off while a planet stands.
- **Other bodies** in its sky (`bodies`, up to four: radius, distance, where the figures see them,
  colour, rock or gas) are the frame's, not the scene's: `FvHost::Impl::applyBodies` places them
  from the figures' floor and hands them to `GltfRenderer::setWorldBodies`, and the shaders test
  rays against them as spheres ([sky.md](sky.md#a-small-planet)). They make no planet again.
- **What people left on it** (`src/fv_world_places.hpp`, `src/fv_host_world_places.cpp`): places,
  the roads between them, single things by the roads and alone. The planet decides where and shapes
  its ground; what they are is data, a catalogue the app's tools write (a folder named by
  `settings.apply`'s `placesDir`), so another kind of place is an entry there and no code here.
  - A *pack* is something laid out once: its models and where their copies stand about its middle,
    how far it reaches, its gates, and a grid of the ground it covers. A *kind* is the packs that
    are alike and the rule for where one stands (the lands it likes, the steepest ground, its
    distance from the next, the road led to it or how many stand alone, and the most of it a
    planet has); an *along* rule stands a kind beside a road, or on it.
  - No two sites of a pack need be alike. A copy can be one that may or may not stand (what was
    left lying about), and `copiesAt` gives each site its own share of those, from where the site
    is: the host stands, and a walker meets, only what it returns. Where the ground is not
    levelled for a site (a group of things by a road), it also puts each copy on the ground under
    it.
  - `fv::world::settle` reads the planet and the catalogue: places on level dry ground (one always
    a walk from the figures, outside what their scene may level), each land's places joined by the
    shortest tree of routes that can be found over the base map's cells (the cheaper where a road
    already runs, so they merge; across a stream by a ford, a wider river by a bridge, which a
    route pays dearly for and crosses straight; up a terrace of the land where it is lowest, as a
    ramp cut into it), the routes as curves the way rivers are, then the single things. Where a
    finished run of road is over a river its kind does not ford (a track fords a stream, an asphalt
    road bridges all but the smallest), its level is held from bank to bank clear of the water
    (`Places::bridges`, found along the run as its pieces lie, and looked up by way: `bridgeAt`). A place no route reached is led to the nearest road, and from the place by the figures
    a road comes to where they stand and ends there, so there is a way to follow from the start.
    The same for a seed whatever the threads. It runs on the worker that makes the planet, or on
    one of its own when the catalogue or the backdrop's `places` amount comes later; the planet
    itself is not made again.
  - A planet file has its say (`fv::world::Settlement`, its `places` list): a *region* is a point
    or an arc with a width, as a feature is, with an amount of places against elsewhere's (0 none,
    2 twice as many and nearer each other) and, if it names them, the only kinds that stand in it;
    all of it within most of its width and easing to nothing at its edge (`regionReach`,
    `settledAmount`, `settledKind`). A region of more is given spots of its own to ask, and of all
    the spots where a region is only its amount's share is asked, so the places' density follows
    the amount wherever the spread's spots happen to fall. Routes pay more to cross a region of few
    places or none, so the roads go round one where a way round is not dear. A *place by name* is
    one of a kind with a road, stood first on the nearest ground round its point that will have
    it; `Places::names` keeps which site has which name, and what could not stand is said in
    `Places::unplaced`, which the host shows as notices. None of it shapes the land: a changed
    settlement settles the planet again (the backdrop's `settlement`, outside `planetSpec`).
  - `withPlaces` adds a stage before the ground's colour: a place's floor is level on the plane
    that touches the planet at its middle and eases into the land, a road is cut and banked to its
    own level across its width (finely drawn quads only; from afar it is a share of the colour),
    but not over a bridge's water, where the ground stays the river's, nor past an open end of a
    way (`Places::wayEnds`: a gate, or where a road gives out), where it stops square; and
    `Sample::built` says what a point has of it. The planet's plants and grass read that:
    nothing planted where a pack keeps clear or a road runs, no tree on a place's floor. A point
    finds what is near it through lists by base-map cell, which hold every cell a place or a road
    reaches into (a cell only clipped at a corner too).
  - The host shows the sites near the eye (`followPlaces`, `fv::world::sitesSeen`): an entity a
    model, its places in the site as copies, under the backdrop's root, so they are the world's
    as its plants are (no props, not saved, outside the floor and the scene's standing). A pack's
    models are read on a worker when its first site comes near and brought in a few a frame,
    held hidden: every site of a pack is copies of the same meshes, so memory goes with the packs
    held, not with the places. What grows over a place shows from nearer. In play mode a place's
    walls are met as a prop's are (`placeSets`).
  - Roads and bridges as meshes (`src/fv_world_roads.hpp`), made with a quad's ground on the pool
    and an entity of the quad's beside its ground: an asphalt road's surface, a ribbon draped on
    the ground as that quad draws it (its own triangles, so the ribbon is never under them) a few
    centimetres over it, with its edges down into the ground, and worn paint on it (a dashed middle
    line, broken edge lines), on the quads fine enough for it, out to some 700 m; further off the
    road is its ground's paint, and a track is always the ground's earth. A bridge, its deck under
    the road at the road's level, parapets, piers in the water, an abutment into each bank, on the
    quad that holds its middle at any level. A stretch of road is the quad's its middle falls in,
    so quads of any levels side by side draw it once. Their materials are the planet's
    (`PlanetBuilt::roadSurface`, `roadPaint`, `roadBridge`), plain colours at first. The asphalt's
    maps are the planet's own (`src/fv_world_asphalt.hpp`, made once on a worker: `paveRoads`): an
    old road's wear laid across its carriageway edge to edge and along it for a repeat (paler, smoother
    wheel paths, dusty and broken edges, cracks, some of them tarred, crazing, repairs, potholes), its
    stones' grain as the material's detail layer, and the markings' paint worn on the same
    coordinates, so a crack runs on through a line. The wear is cut into stretches by joints
    (`kAsphaltJoints`, straight cracks across it, some sealed) that nothing else crosses: each way
    lays the stretches in an order of its own, each either way along and either way across, the
    surface cut on a joint where one gives way to the next, so no run of road is another's; and the
    vertices' colours tint it by a weathering wider than a stretch. Where a way ends (`WayEnd`, found
    as the planet is settled): running on into the next way, their surfaces meet; joining a road
    from the side, its surface stops at that road's edge; at an open end it is broken off, dusty, its
    edge going down into the ground, its paint stopping short. A bridge takes the kit's concrete once
    the kit has it (`mapPlanetRoads`).
- **Its map** (`src/fv_world_chart.hpp`, the `planet.map` command, the app's World › Planet › Map):
  an author's view of the planet, north up. `chartImage` draws a window of latitude and longitude as
  an image, a sample a pixel at the pixel's footprint (so the stages leave out what is finer than it):
  the ground in `surfaceColour`'s colours shaded by its slope, the sea darker the deeper it is, lakes,
  and the base map's rivers and the settled roads drawn over it as lines a pixel wide at the least.
  `planet.map` makes it on the command's own thread from the planet as it stands, and sends it as a
  PNG with what goes over it as data: where the figures stand, the planet file's features and regions,
  the places a road leads to (by name where the file names them), and what could not stand. The app
  draws those, the grid and the outlines on the sphere, and asks again for each view it is moved to.
- **The camera** (`FvHost::Impl::followPlanet`, `applyFly`): the free view's "up" is the scene's by
  the figures and the planet's own vertical away from them (`fv::world::viewUp`), so the horizon
  stays level on the far side; a flight's speed goes with its height over the ground
  (`fv::world::flySpeed`); the far plane is kept past the planet's far side.

## The pixel readout

What is at a pixel and how bright it is (`view.inspect {x, y, w, h}`, the `inspect` topic, protocol 44;
the app's Display › Pixel readout, where a click reads out instead of selecting).

- **The reading:** one thread of the path tracer's own pipeline (`ePtInspectPixel`, `inspectPixel` in
  `gltf_pathtrace.slang`), recorded at the frame's end with the last traced frame's constants
  (`PathTracer::cmdInspectPixel`). The camera ray through the pixel's centre meets a mesh, the ground or
  the water's surface as the paths do (`checkInfinitePlaneIntersection`, `checkWaterIntersection`), with
  a fixed draw for stochastic opacity so a strand gives the same answer each time; the surface is
  evaluated as the paths evaluate their first hit (`evaluateMaterial` with the frame's thin-preview and
  skin flags, `groundMaterial`, `waterSurfaceMaterial`), before any regularisation. With it, what the
  frame's image holds at the pixel: a still's mean so far, or a moving frame's single sample.
  `shaderio::PixelInspect` holds it (its layout checked against the shaders' with the camera medium's, `StructLayout.SceneFrameInfo`), through
  `PathtracePushConstant::pixelInspect`, which fills the push constant's last 8 of 256 bytes.
- **No new branch in the kernel:** the camera medium's probe and the readout share the one-thread branch
  of `computeMain` and `rgenMain`, whose existence alone costs the path tracing kernel 1-2% (the probe's
  note). Its code still sits in the kernel: a still's sample at 1TEST's bar (2160 x 1395, the user's
  Render tab) took 64.19-64.37 ms before and 64.41-64.91 ms after (sessions before, after, after,
  before; the first session's warm-up dropped), +0.35% best to best, +0.5% at the median, against
  about 0.3% between rounds: at the edge of the noise. A pipeline of its own would take it out of the
  kernel, at another pipeline per variant.
- **When:** at once for a new pixel, then every 250 ms while frames are traced, and once after the last,
  so it settles on the finished still (`GltfRenderer::cmdInspect`); one reading in flight, collected
  when its frame's semaphore has signalled (`updateInspectState`), as the click pick is. The host adds
  the mesh and material hit (`instancePickAt`, the pick's) and the exposed luminance: the radiance after
  the last frame's exposure and white balance (`getColorCorrectionMatrix`), what the false-colour view
  bands in stops from middle grey. `FvView::publishInspect` names the prop, node and surface as the
  selection does.
- **Checked:** `FvHost.ThePixelReadoutPublishesWhatIsAtThePixel` (the command's pixels, its refusals,
  the topic for a mesh, the ground and nothing, stopping), the layout test, the app's
  `test/pixelReadout.test.ts`. The shader's reading itself is not unit-tested: no GPU in the suite.

## The path tracer's build time

The driver's cold build of the path tracer is what the app's "Building shaders" badge waits on after a
shader change (the start-up warm-ups, `PathTracer::startPipelineWarm`, a thread for each embedded build)
and what a variant's first use waits on. A rebuild that leaves the shaders alone builds nothing: the
driver's cache (<cacheDir>/driver) is keyed by the SPIR-V, not by the exe.

- **What costs:** the ray tracing pipeline's raygen shader is one function. NVIDIA's compiler inlines
  every call, `[noinline]` (SPIR-V's DontInline) included, which it ignores, so a function called from
  two places is compiled twice, with everything it calls. The build's time goes with how many such
  copies there are much more than with the features (2026-11-07, RTX 4080, the still's variant compiled
  from source, driver cache cold, runs of the same source 15% apart): hair, clearcoat, sheen,
  iridescence or transmission switched off each moved it within that, while the extra copies of the
  light sample (the walk's, the shadow catcher's, the haze's) were half of it, 57.6 s against 26.8 s
  without them.
- **One call each:**
  - `processVolumeSegment`, a walk's step with its light sample and shadow ray, called once for a hit
    and a miss alike. It used to be called at the miss and again at the hit.
  - `sampleLightDraw`, called once in `sampleLights` instead of in each of its two branches.
  - `lightPickScore`, the scoring of the lights with or without a draw. It used to be two functions
    (`lightPickSet` is now its wrapper).
  - The haze's guide and candidates share one loop, the guide as its first pass.
  - A lobe's mirrored evaluation under retroreflection is a `[loop]`'s second pass, which the driver
    does keep (`bsdfEvaluateLobeBlended` and `bsdfEvaluate`, nvshaders). The second copy in line used to
    double the lobes of every sum over them: next-event estimation's, and the sampler's mixture pdf.

  Each draws the same numbers in the same order as before.
- **One sample a pass (2026-11-10):** `samplePixel`, the whole kernel, was called for a frame's first
  sample and again in a loop for its later ones, so all of it was compiled twice. The kernel now traces
  one sample a pixel, and the host runs a pass of it for each sample of the frame
  (`PathTracer::cmdTraceFrameSamples`, `PathtracePushConstant::frameSample`): a later pass is no first
  frame, has the samples before it as its total and its guides' count, and adds its sample to what the
  passes before it left in the images. A frame of one sample, which is every frame the app draws but a
  still refined with DLSS, is one pass as before.
  - **The build:** the start-up warm-ups, side by side, took 27.2 s (still) and 33.0 s (DLSS), against
    61.8 s and 75.2 s the same morning.
  - **The frame:** the second copy also ran slower, since it carried the frame's sums (colour, second
    moment, guides) through every trace and reorder. On the club (1TEST, 2160x1395, 20 bounces, trace
    GPU time, two sessions of each build alternating): a sample of a still 84.5 ms at one a frame,
    106.8 at two and 111.5 at four before; 81.4 ms at any count after. The moving frame 20.87 to
    20.48 ms. A single loop in the shader around the one call (tried first) halved the build too but
    cost the one-sample frame 8-10%, the sums being live through the kernel; with the sums left out of
    it (a timing experiment) that cost went, which is what a pass a sample gives without losing them.
  - **The image:** a later pass draws its own numbers (its seed takes `frameSample`), where the loop
    went on with the first sample's sequence, so a frame of several samples is another sample set of
    the same estimator: stills of 64 samples at four a frame against one a frame have the same mean
    within 0.05% and the same noise between seeds, and the moving view at four a frame loses its
    near-black pixel values as the loop's did. At one a frame the golden gate's views differ by
    rounding, more than the reshapings below gave: at 32 samples about 1% of the pixels (relMSE 1e-8 to
    1e-11); at the first sample 0.05-0.07% of them, two thirds by under 1e-5 of their value and the
    most by 3%, the same path a little off, with the image's mean the same to 3e-7. A single loop in
    the shader, by contrast, left every bit as it was.
- **What the kernel carries, measured (2026-11-10):** a path's state is live across every trace and
  reorder, and what that costs depends on how often it is written. On the club (1TEST, 2160x1395, trace
  GPU time, timing-only builds against the current one): 32 values that never change, carried through
  the kernel, cost 1%; 16 values added to `PathTracerState` and updated at every bounce cost 10% of
  the moving frame and of a still's sample alike. The other way, the light-pick record
  (`PathTracerState::lightPick`) not carried at all would save 4%, the ray read back from the hit
  object after the reorder saves nothing, and the guides (written once, at the first hits) under 3%.
  None of those was built (the app's `docs/improvements.md`, *Checked and rejected*); the rule they
  leave is for new state: a value a path rewrites at every bounce costs about 0.6% of the frame, a
  value written once next to nothing.
- **Measured:**

  | Shaders | Cold build of the still's pipeline |
  |---|---|
  | Before | 64.9 s, 66.8 s |
  | The walk's step, `sampleLights`' draw and the scorings one call each | 44.2 s |
  | Also the haze's guide and candidates in one loop | 42.0 s, 51.0 s |
  | Also retroreflection's two passes as one loop | 15% less again: 54.7 s to 46.6 s, from one session to the next |

  At start-up the two warm-ups, built side by side with the shaders before the retroreflection loop,
  took 45.6 s (still) and 56.2 s (DLSS), against about 70 and 83 s before.
- **Tried, and no use:**
  - Deferred host operations (`USE_DEFERRED_RTX_COMPILE`): 57-62 s. The driver doesn't split this
    pipeline over threads, or the raygen shader is all of it.
  - A separate pipeline for the one-thread modes (the camera medium's probe, the pixel readout): no
    change.
  - Slang's debug info off: within the noise once the copies were gone.
  - Slang `-O2`: Slang itself took 300 s.
  
  `FV_SLANG_DEBUG` and `FV_SLANG_OPT` (0-3) set the run-time compile's levels, for such measurements.
- **The image:** the golden gate's four views differ only in float rounding: at most 7.6e-6, relMSE
  at most 1.3e-15, in 1-10% of the pixels, where a changed draw shows as 1e-3 or more. The driver orders
  and contracts the reshaped arithmetic a little differently.
- **How to measure:**
  1. Copy the exe into a scratch tree beside its own copy of `shaders/` and `nvshaders/` (it finds
     them at `../../shaders` from `_bin/Release`), with its own driver cache (fv.Session's `cache_dir`).
  2. Per variant, edit the sources and give an output a fresh constant, so the SPIR-V is new.
  3. Call `vk_gltf_reload_shaders` (it compiles from source, bypassing the SPIR-V cache) and read the
     next frame's `RTX pipeline creation: cache miss ... (N ms)`.
  
  Interleave the variants with baselines; an MCP session ends after about 10 minutes.
- **Copies left, by their measured share:**
  - The walk's light sample, about 8 s: its own call beside the surface's.
  - The surface light sample's two BSDF evaluations (`bsdfEvaluateAll` and `bsdfEvaluate`), about 5 s.
  - `sizedLightHits`, at the miss and at the hit.
  
  A new call of a heavy function (a light sample, a shadow ray with its any-hit, a material evaluation)
  costs build time in this measure: put it on an existing call when it can share one.

## Reflection sampling: bounded visible normals

Every GGX reflection lobe draws its bounce by Eto and Tokuyoshi 2023's bounded VNDF sampling
(`hvd_ggx_sample_vndf_bounded`, nvshaders' `pbr_ggx_microfacet.h.slang`, in `brdf_ggx_smith_sample`
and `brdf_ggx_smith_eval`). That covers the specular, metal, coat and second specular lobes, and so the
flakes and the water's Gaussian slopes. Transmission keeps Heitz's sampler, since a refraction can
leave through the part the bound cuts.

- **The bound:** Dupuy and Benyoub 2023 draw the visible normals from a spherical cap in the lobe's
  stretched space. The bound (`ggx_vndf_reflection_bound`) cuts the cap where every normal it would
  give reflects below the surface. An anisotropic lobe takes its smaller alpha, which cuts less and
  stays exact.
- **The pdf:** a direction's pdf is the visible normals' with G1 replaced by
  `ggx_vndf_reflection_g1`, 2 k.z / (bound * k.z + t). The sample's weight is G2 over that, on both
  MIS sides alike.
- **What it gives:** fewer samples lost below the surface, so less noise in what rough reflections
  carry, with the same light. For a white lobe, against the visible-normal sampler
  (`utils/bounded_vndf.py`):

  | GGX alpha | Samples lost below the surface | Variance of the lobe's albedo estimate |
  |---|---|---|
  | 0.3 | 3-9% → 1-4% | ×0.53-0.65 |
  | 0.6 | 6-23% → 0.4-6% | ×0.38-0.42 |
  | 1.0 | 9-41% → none | ×0.18-0.35 |

  The albedo is the same to the 4th decimal.
- **Whole frames:** `scripts/noise-vs-reference.py`'s views against their references, relMSE at 256
  samples (capped), same seed:

  | View | relMSE |
  |---|---|
  | helmet (rough metal) | -6% |
  | chair | -8% |
  | alpha cutouts | -1% |
  | Mirel | +2%, within the noise |

  At 64 samples: -4%, -8%, -0.4%, -2%.
- **Checked:**
  - `BsdfLobes.TheBoundedSamplerLosesFewerReflectionsBelowTheSurface`: the lost share, and the
    chi-square of rough and anisotropic metal at grazing views.
  - `BsdfLobes.TheBoundedSamplerKeepsTheMeanWithLessVariance`: the mean within 4 sigma of the
    visible-normal sampler's at 400,000 samples, the variance under 0.8 of it.
  - The existing pdf, albedo-fit and energy tests.
  
  `tests/figure/data/closure_baseline.json` was rewritten: the closure probe's sampled lobe albedos moved by
  -1.4% to +0.7%, mean -0.14%, as noise of another draw.

## How rough a bounce was

A path carries `pathRoughness`, the roughest event along it so far (perceptual). Three things read it:

- `prefixIndirect`, which says whether the firefly clamp acts on the light a path gathers next;
- the lobe mixture for direct paths (`ePtLobeMixtureDirect`, up to `DIRECT_PATH_ROUGHNESS` 0.1);
- the skin's scatter simplification (from `SCATTER_SIMPLIFY_ROUGHNESS` 0.15).

A glossy event counts as the roughness of the lobe that took it. `bsdfSampleCore` returns that lobe,
and `bsdfLobeRoughness` (nvshaders' `bsdf_functions.h.slang`) gives its roughness:

- the coat's and the second specular lobe's own roughness;
- the base's larger anisotropic alpha for the specular, metal and transmission lobes;
- 1 for sheen, which is broad at any roughness.

An impulse counts as 0, a diffuse event as 1, and a fibre's glossy event as its longitudinal
roughness. This is how Falcor's `isSampleRough` judges a sample.

Until 2026-11-07 a glossy event took the smoothest of the specular, coat and second lobe. A bounce off
the rough base under a sharp coat (car paint, a PBRSkin top coat, clearcoated leather) then read as
sharp: the clamp skipped the light it reached, and the lobe mixture and the unsimplified skin walk
carried on behind it.

Checked by `BsdfLobes.ALobeIsAsRoughAsItsOwnRoughness` (each lobe's roughness) and
`BsdfLobes.ASampleReturnsTheLobeThatTookIt`. The second test checks three things: lobes come back as
often as `bsdfLobePick` picks them, each with its own kind of event; under a sharp coat or second lobe,
some of the glossy samples now read as rough; on sheen cloth, all of them do.
