# Figure Viewer's renderer

The render process of [Figure Viewer](../..): a Vulkan path tracer and rasterizer that runs headless,
driven by its host (`src/fv_*`) over a local protocol, and hands the app its frames. It began as a fork
of NVIDIA's [vk_gltf_renderer](https://github.com/nvpro-samples/vk_gltf_renderer) on
[nvpro_core2](https://github.com/nvpro-samples/nvpro_core2), and no longer merges it: the standalone
window, its panels and the features the app has no use for are gone.

- `figure/`: DAZ content: DSON, morphs, poses, wardrobe, materials, dForce.
- `src/`: the renderer and the host.
- `shaders/`: Slang shaders, shared structs in `shaders/shaderio.h`.
- `tests/`: `pnpm renderer:test` from the app. `tests/README.md` says more.

Start with [CLAUDE.md](CLAUDE.md), which says what to read for a given task. Build with
`pnpm renderer:build` from the app.
