# CLAUDE.md — Agent guide for vk_gltf_renderer

Read this first. It is a **router**, not a manual: it points you at the right
doc and states the rules to follow. Keep it thin — deep knowledge lives in
`docs/`. If you change an architectural invariant, update the relevant `docs/`
file **and** this file in the same commit.

## What this is

The render process of Figure Viewer, a fork that no longer merges upstream: it runs headless only,
its host (`src/fv_*`) driving it over a local protocol, with no window or panels of its own. It began as
a Vulkan **ray tracing** renderer for glTF 2.0 scenes, built on
[nvpro_core2](https://github.com/nvpro-samples/nvpro_core2) with
[Slang](https://github.com/shader-slang/slang) shaders. It has one render path, the **RTX path
tracer** (`renderer_pathtracer`), kept interactive by DLSS Ray Reconstruction while the view moves.
The rasterizer it once had beside it was removed (2026-11-02): don't bring a second render path back
for previews.

C++20. Host and device share structs via `shaders/shaderio.h`.

## Where to read before you code (do this instead of grepping the whole tree)

| Your task | Read |
|---|---|
| Understand overall architecture, app lifecycle, source map | [docs/developer.md](docs/developer.md) |
| Scene data flow: retained scene → residency → GPU tables / BLAS / TLAS | [docs/RENDERING_ARCHITECTURE.md](docs/RENDERING_ARCHITECTURE.md) |
| Sky models: physical atmosphere, its clouds, gradient, units and calibration, the sun, glTF mapping | [docs/sky.md](docs/sky.md) |
| Environment *pipeline*: the analytic bake, lighting vs background fields, preview/commit | [docs/developer.md](docs/developer.md) (§ Environment Lighting) |
| MCP shader timing: enabling `--mcp`, the tool surface, measuring a change | [docs/mcp.md](docs/mcp.md) |
| DLSS-RR denoising (moving and still), motion vectors, jitter/reset (incl. why animated meshes ghost) | [docs/denoising.md](docs/denoising.md) |
| Test suite: running (CTest vs. direct), adding tests, benchmarks | [tests/README.md](tests/README.md) |
| Where scene/model assets come from | [docs/resources.md](docs/resources.md) |

`docs/developer.md` contains the full annotated `src/` and `shaders/` file map —
use it to locate a subsystem instead of fanning out readers across the tree.

## Source of truth — read the code, not the doc

For anything the compiler or loader **enumerates or enforces**, the code is authoritative and
docs drift. Grep the symbol below instead of trusting a prose list, and when a doc disagrees
with the code, the code wins (fix the doc — see "Keep the docs true").

| You need… | Grep / read this (authoritative) |
|---|---|
| What a glTF file brings in | `rs::importGltf` in `src/scene_import_gltf.cpp`; materials in `figure/materials/material_gltf.cpp` |
| Visualization / debug modes | `enum Visualization` in `shaders/shaderio.h` |
| Environment types | `enum EnvSystem` in `shaders/shaderio.h` |
| Command-line / MCP / persisted settings (names, defaults, whether they persist) | `m_settings.add(...)` in `src/renderer*.cpp`, plus `parameterRegistry.add(...)` in `src/main.cpp` |
| What a viewport drag does to the camera | `camera_controls::dragMotion` in `src/camera_controls.hpp`; the app's view input in `FvHost::Impl` (`src/fv_host_live.cpp`) |
| Material extension gates | `MAT_EXT_*` in `shaders/gltf_material_config.h`, `GLTF_USE_*` in `shaders/gltf_eval_config.h` |
| Host/device structs & layout | `shaders/shaderio.h` + the `*_io.h.slang` / `*_shaderio.h.slang` headers |
| Which files are tests/benchmarks | `tests/CMakeLists.txt` |
| MCP tool names, schemas, and annotations | `registerTool(...)` in `src/mcp_timing.cpp` |
| CMake options & defaults | root `CMakeLists.txt` |

## Source layout (one line each; full map in developer.md)

- `src/renderer*.{cpp,hpp}` — the render backend (`BaseRenderer` → `PathTracer`), plus a separate
  `Silhouette` compute pass, orchestrated by `renderer.cpp` (`GltfRenderer`).
- `src/scene_retained*`, `src/scene_residency*`, `src/scene_import_gltf.*`, `src/scene_export_gltf.*` —
  the scene: its store, kept resident (GPU tables, BLAS, TLAS), and glTF in and out.
- `src/env_baker.*`, `src/gltf_environment_sky.*` — the analytic sky bake (an authored or
  physical sky becomes the same lat-long image + alias table an HDR file would) and
  `OMI_environment_sky` load/save.
- `src/sky_clouds.*` — the physical sky's clouds, a port of takram's three-geospatial model: tiling
  noise, the cloud images the bake and the path tracer compose over the sky, and the pass that bakes
  them (`shaders/sky_clouds*.slang`).
- `src/sky_bruneton.*` — the Bruneton precomputed-atmospheric-scattering LUTs and the six compute
  passes that build them. The only physical sky; the older MDL one was deleted from this repo (it
  stays in `nvpro_core2` for other samples).
- `src/fv_world.*`, `src/fv_host_world.cpp`, `src/sky_world.hpp`, `shaders/sky_world.h.slang`,
  `shaders/world_bodies.h.slang` — a small planet as the backdrop's `planet` kind: its fields by
  stages, its quadtree meshes drawn a cut at a time, the sky scaled to it and asked from any place of
  it, on the ground or in orbit, and the other bodies in that sky
  ([docs/RENDERING_ARCHITECTURE.md](docs/RENDERING_ARCHITECTURE.md#a-planet)).
- `src/sun_position.*` — NOAA solar position.
  A pure function with no dependency on the rest of the renderer; nothing else links against it.
- `src/sky_preset.*` — `.sky.json`: one sky as a file of its own. The text layer only; the
  serializer is `gltf_environment_sky`'s, so a preset and a glTF cannot drift.
- `src/mcp_timing.{cpp,hpp}` — the optional MCP endpoint (`--mcp`): recompile
  shaders and time a GPU pass, so an agent can measure a shader change. Loading
  and every other setting go through the parameter registry instead. Built only
  when `NVPRO2_ENABLE_nvmcp` is on. See [docs/mcp.md](docs/mcp.md).
- `figure/` (target `fv_figure`, Vulkan-free) and `src/fv_*` — the Figure Viewer's native scene: DAZ figures, morphs,
  posing, wardrobe, materials, dForce and GLB export, hosted in the renderer and driven over a loopback protocol.
  The host is one `FvHost::Impl` (`src/fv_host_impl.hpp`) split by area into `src/fv_host_*.cpp`, its
  commands registered in `src/fv_host.cpp`. The protocol is listed in `src/fv_host.hpp` and
  `src/fv_view.hpp`; the app's `native/README.md` describes how it drives the renderer, and its
  `docs/improvements.md` holds the open issues.
- `src/dlss*`, `src/rr_still_denoiser*` — DLSS Ray Reconstruction, the one denoiser: the moving view's,
  and the still's over the accumulated image (OptiX and OIDN were removed 2026-11-09; the app is
  NVIDIA RTX only). `src/denoiser_handoff*` blends from the moving image to the denoised still
  ([docs/denoising.md](docs/denoising.md#dlss-hand-off)).
- `src/gizmo_*` — transform gizmo, grid, overlays.
- `shaders/*.slang` + `shaders/shaderio.h` — GPU code and host/device structs.

## Invariants — do not break these

- **The scene's GPU resources are the retained scene's.** Scene buffers, textures and descriptors
  are built once, fed by a stream host or imported by `--scenefile`; the path tracer, the silhouette
  and the pickers read them through `SceneGpuView` (`Resources::gpuView()`), never the backend's own
  objects.
- **Instances are derived from entities**, incrementally (`rs::Residency`): don't hand-mutate its
  tables; edit the store (`GltfRenderer::hostScene`), and there is no glTF model beside it.
- **Materials are evaluated in one place.** New glTF material extensions land in the path tracer's
  Slang material eval (`shaders/gltf_material_eval.h.slang`, gated by `MAT_EXT_*` flags in
  `shaders/gltf_material_config.h`).
- **Host/device structs live in `shaders/shaderio.h`** (and the `*_io.h.slang` /
  `*_shaderio.h.slang` headers). Change both sides together; never let CPU and
  GPU layouts drift.
- **Every user-settable value is declared once**, through `SettingsRegistry`
  (`src/settings_registry.hpp`). One call feeds the command line, benchmark
  sequences, MCP, and the persisted `.ini`; `Persist` says whether it survives a
  restart. Never register a setting in two places again — that is what let the
  two old lists drift.
- **Scene editing has one route.** A stream session's scene is the host's: it
  edits the retained store through its mutation API only (the feed, the
  importer, entity and record edits), and undo is the host's edit history
  (`src/fv_journal.hpp`, the `history.*` commands).
- **Scene, model, and Vulkan state is application-thread-owned.** An MCP tool
  that touches them sets `runOnApplicationThread` and stays short. A tool that
  waits on rendered frames must not, since the application thread is the one
  producing them — `GltfRenderer::measureTimer` is the deliberate exception and
  reads only the profiler's mutex-guarded snapshots, never the scene.

## Fix what you find — auto-fix on discovery

When you find a bug, drift, gap, or clear defect while working — **even if it's outside the scope
of the current task** — fix it in the same change instead of only reporting it. Don't leave
known-bad code or docs behind. ("Keep the docs true" below is the docs-specific instance of this.)

- **Fix it now:** typos, stale/incorrect docs, dead code, obvious bugs, missing null checks, wrong
  comments. Root-cause it (no symptom patches), prefer the elegant fix, and verify it with a build
  and unit tests before claiming done. In Figure Viewer, renders, render tests, benchmarks and A/B
  runs are by request: see "Unit tests yes, manual QA by request" in the app repository's
  `CLAUDE.md`.
- **Ask first:** scope-expanding or architectural changes, public API / on-disk-format / behavior
  changes, anything destructive (deleting files, migrations, history rewrites), or anything that
  would break an **Invariant** above.
- **Can't fix it now?** Leave a visible marker and call it out in your summary so the next reader
  (or a grep) finds it — e.g. `// TODO:` or `<!-- DRIFT? doc says X; code does Y -->`. Never
  silently leave known-bad text, and never guess a fix you can't verify.

## Keep the docs true — every agent maintains this

The **code is the source of truth.** The docs (`docs/`, this file, `README.md`)
exist to explain *how things work at a high level* and to point at *where* they
live. They must never restate what the code already says — no copied signatures,
no exhaustive enumerations the code defines, no line-by-line narration that
drifts the moment someone edits.

This is a public, NVIDIA-facing repository read by many developers. Hold the
docs to the same bar as the code: accurate, concise, professional. As you work,
you are expected to keep them that way:

- **Fix drift when you see it.** If a doc is wrong, stale, or contradicts the
  code — even if it's outside your task — correct it or, if you can't verify the
  intended behavior, flag it clearly rather than leaving known-bad text.
- **Update docs in the same change as the code.** When you alter behavior that a
  doc describes (an invariant, a data-flow step, a workflow, a build option),
  update that doc and this file together. A code change that invalidates a doc
  is not complete until the doc is fixed.
- **Complete what's missing.** When you add a feature or glTF extension, add its
  high-level explanation and a pointer to the right file — enough for the next
  reader to find and understand it, not a re-implementation in prose.
- **Verify before you trust.** A doc may lag the code. Before relying on a doc claim that
  names a symbol, flag, value, or path, grep it (see "Source of truth"). If it no longer
  exists or disagrees with the code, treat the code as correct and repair the doc.
- **Flag what you can't verify.** If you spot likely drift but can't confirm the intended
  behavior, leave a visible marker instead of silent bad text — e.g.
  `<!-- DRIFT? doc says X; src/foo.cpp seems to do Y -->` — so the next reader (or a grep)
  finds it.

**Never hard-code in docs** (these are exactly what drifts — link to the symbol instead):

- Enumerations the code defines (extensions, visualization modes, CLI flags, enum values).
- Exact numeric values the code owns (thresholds, sizes, offsets, frame counts).
- Verbatim struct layouts or function signatures — describe the role, point to the header.
- Source line numbers — reference a symbol or section anchor, never `around line N`.

**When you change code, update its doc (reverse of the routing table):**

| You changed… | Update |
|---|---|
| Scene sync / dirty-flag / BLAS-TLAS data flow | [docs/RENDERING_ARCHITECTURE.md](docs/RENDERING_ARCHITECTURE.md) |
| App lifecycle, source layout, material fork, build/test | [docs/developer.md](docs/developer.md) + this file |
| DLSS-RR (moving or still), motion-vector / jitter / reset behavior | [docs/denoising.md](docs/denoising.md) |
| A sky model, its units, the sun, or the `OMI_environment_sky` mapping | [docs/sky.md](docs/sky.md) |
| Environment sources, the sky bake, or the lighting/background split | [docs/developer.md](docs/developer.md) |
| MCP tools or the timing measurement | [docs/mcp.md](docs/mcp.md) |
| Added a glTF material extension | path tracer eval first, then README extension list + developer.md checklist |

## Conventions

- **Reuse before you write.** Prefer existing nvpro_core2 helpers (`nvvk`,
  `nvapp`, `nvutils`, `nvgui`) and the project's own utilities over new code.
  Check `docs/developer.md` and the sibling `nvpro_core2/` repo first. Keep it
  DRY — this codebase is meant to grow cleanly, not accrete duplicates.
- Match the naming and file-pairing already in `src/` (`foo.cpp` / `foo.hpp`,
  `*_vk` for Vulkan-resource owners, `scene_*` for scene subsystems).
- Slang shaders, not GLSL.
- **Initialise a struct you declare in Slang** (`Foo f = {};`): a bare `Foo f;` gets none of its member
  defaults, and a path that leaves a field unset reads garbage. The build makes Slang's uninitialised-variable
  warning (E41035) an error for the shaders it compiles; the runtime-compiled variants are not checked.
- **Text files are stored as LF**, enforced by `.gitattributes` (`* text=auto`).
  Your working tree follows your own `core.autocrlf` / `core.eol`, not your OS:
  `autocrlf=true` gives CRLF, `input` gives LF, and `false` defers to `core.eol`
  (default `native` — CRLF on Windows). All of them round-trip cleanly, so keep
  whichever you use; set `core.eol=lf` if you want LF locally. Never commit
  a file with CRLF and never use `* -text`; both reintroduce the whole-file diffs
  that renormalization removed.

## Build / run / test

```bash
# Sibling clone recommended: nvpro_core2 next to vk_gltf_renderer
# (CMake auto-downloads nvpro_core2 into build/_deps if not found locally)
# Windows
cmake -B build -S . -A x64
cmake --build build --config Release
./_bin/Release/vk_gltf_renderer.exe

# Linux
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./_bin/Release/vk_gltf_renderer
```

Key CMake options: `USE_DLSS`, `USE_DRACO`,
`NVPRO2_ENABLE_nvmcp` (all `ON`; the last one adds the `--mcp` endpoint and fetches
cpp-mcp at configure time — see [docs/mcp.md](docs/mcp.md)),
`BUILD_TESTING` (`OFF` — turn on to build unit tests + benchmarks; `ctest` runs the
unit tests, run the benchmark executable directly), `FV_PHYSICS_GPU` (`ON` — the GPU cloth solver, `src/fv_physics_gpu*` and
`shaders/physics_cloth*.slang`, the app's "Live (GPU)" mode; a build without it says so in the
protocol's `features` and the app hides the mode).
