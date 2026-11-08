#pragma once

// The water table the path tracer reads (GltfScene::waterGrids and the arrays after it): the water on
// each material slot (rs::MaterialRecord::water) as a grid over the UVs its domes reach, each cell
// listing the domes that reach into it, so a hit looks at a few domes, not all of them
// (shaders/water.h.slang). Made again whenever any of it changes: a pass of rain lands every few
// seconds at most, and the table is a few hundred kilobytes.

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <figure/materials/water.hpp>

#include "shaders/gltf_scene_io.h.slang"

namespace rs {

struct WaterTables
{
  std::vector<shaderio::WaterGrid> grids;  // one per slot, a dry one zero
  std::vector<glm::uvec2>          cells;
  std::vector<uint32_t>            refs;
  std::vector<shaderio::WaterItem> items;
};

// How far a dome reaches past its own outline: the skin around a bead or a track is wet too.
constexpr float kWaterHalo = 1.5f;
// A cell's size on the skin, m: about a bead across, so a hit looks at a handful.
constexpr float kWaterCell = 5e-3f;
// At most this many cells a side.
constexpr uint32_t kWaterCellsMax = 512;

// `slots` grids, the layers in theirs (a slot past `slots` is left out).
WaterTables packWater(uint32_t slots, std::span<const std::pair<uint32_t, const fv::mat::WaterLayer*>> layers);

}  // namespace rs
