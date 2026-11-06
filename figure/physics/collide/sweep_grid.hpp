#pragma once

// The collision grid the GPU cloth solver finds faces in (src/fv_physics_gpu.cpp, physics-plan step 13b):
// every face over its sweep through a frame, its corners from `start` to `end`, filed under each cell of
// `cellSize` its box meets. An open-addressed table of the cells met, linear probing from
// sweepCellHash in a power of two of slots, each cell with its run of `items`: the faces in it in
// ascending order. The GPU's nearest-face search (shaders/physics_cloth.comp.slang, nearestFace) takes
// the first of equally near faces as it meets them, so that order is part of the drape; where a cell
// lies in the table is not.
//
// Made on the host every frame the pose moves, so its cost is the host's share of a GPU frame.

#include <cstddef>
#include <array>
#include <cstdint>
#include <vector>

namespace fv::physics {

// A cell of the table: shaderio::ClothGpuCell's layout (32 bytes). `start` < 0: an empty slot.
struct SweepCell
{
  int32_t x, y, z, start;
  int32_t count, pad0, pad1, pad2;
};

struct SweepGrid
{
  std::vector<SweepCell> cells;  // the table: a power of two of slots
  std::vector<int32_t>   items;  // each cell's faces, a run from its start
  size_t                 entries = 0;  // a face once for every cell it is in
  size_t                 used    = 0;  // cells met
  uint32_t mask() const { return uint32_t(cells.size() - 1); }

  // the builder's, kept between frames: each face's cells, and each entry's slot
  std::vector<std::array<int, 6>> ranges;
  std::vector<uint32_t>           slotOf;
};

// As the kernels hash a cell (cellHash): the coordinates mixed, then finished as MurmurHash3's fmix32
// does, so neighbouring cells land apart in the table and a probe stays short in a fuller one.
inline uint32_t sweepCellHash(int ix, int iy, int iz)
{
  uint32_t h = (uint32_t(ix) * 73856093u) ^ (uint32_t(iy) * 19349663u) ^ (uint32_t(iz) * 83492791u);
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

// `corners`: three vertex indices a face; `start`, `end`: xyz a vertex. `out` keeps its buffers.
void buildSweepGrid(const std::vector<int32_t>& corners, const float* start, const float* end, float cellSize, SweepGrid& out);

// The same grid as it was first made (one thread, the table sized from the entries): the reference the
// faster one is held to.
void buildSweepGridReference(const std::vector<int32_t>& corners, const float* start, const float* end, float cellSize, SweepGrid& out);

// Whether two grids hold the same cells, each with the same faces in the same order (where a cell lies
// in its table aside): what the kernels would find in them is the same. For the tests and the bench.
bool sameSweepGrids(const SweepGrid& a, const SweepGrid& b);

}  // namespace fv::physics
