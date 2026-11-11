// A material's pictures in seconds, with no renderer: fv_parity's {"slice": label} step. For each
// surface, the BRDF's slices (each lobe and summed), a ball under a white furnace and a ball under one
// light, from the path tracer's own material evaluation and BSDF compiled as C++ (closure_probe's
// probeViews). Textures are neutral (white, flat normal maps): a lobe's shape and strength, not a map's.
// Before-and-after pictures for a material change, and for an agent to look at. Not a test.

#pragma once

#include <filesystem>

#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>

namespace fv::probe {

// `step`:
//   "out"       a folder against `root`; each surface's pictures go in a folder of its own there
//   "file"      a library file to load (as the sweep does), or
//   "gltf"      one glTF material (a glTF document's materials[] entry), or neither: the materials of
//               the figure the case loaded (`loaded`, null when none)
//   "surfaces"  only these surface names (default all)
//   "skin"      "measured" (default), "volume", "thin" or "raster", as the lowering takes it
//   "size"      the pictures' size in pixels (256)
// Each surface's folder holds sheet.png (top: the lit ball, the furnace ball, the summed slice; below:
// each lobe's slice), the same pictures as .png, and their linear values as .pfm. Returns, by surface,
// its folder, its lobes and its albedo head-on and at grazing.
dson::Json materialViews(const std::filesystem::path& libraryDir, const std::filesystem::path& root, const dson::Json& step,
                         const std::vector<Material>* loaded);

}  // namespace fv::probe
