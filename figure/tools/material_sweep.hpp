// The real library's materials, measured: every surface of every file that loads as an item, through
// the parser, the lowering and the closure (closure_probe). fv_parity's {"sweep": label} step
// (cases/material_sweep.json, `pnpm library:materials`). What it answers: which shaders the library
// holds, which lowered features it reaches, which authored channels nothing reads, and, against the
// baseline it keeps, which surfaces a change moved.
//
// Material presets are not swept: a preset lands on an item the file doesn't name, so it has no
// surfaces of its own. The report counts them as skipped.

#pragma once

#include <filesystem>

#include <figure/dson/store.hpp>

namespace fv::probe {

// `step`: "baseline" (a path against `root`: written on the first run, compared after), "update"
// (true: rewrite it), "match" (only files whose path holds this text), "limit" (at most this many
// items), "threads" (0: half the machine's), "skin" ("measured", "volume", "thin", "raster").
dson::Json materialSweep(const std::filesystem::path& libraryDir, const std::filesystem::path& root, const dson::Json& step);

}  // namespace fv::probe
