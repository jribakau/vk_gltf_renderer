#include <figure/physics/collide/sweep_grid.hpp>

#include <algorithm>
#include <array>
#include <cmath>

#include <figure/jobs/pool.hpp>

namespace fv::physics {

namespace {

int cellOf(float p, float cellSize)
{
  return int(std::floor(p / cellSize));
}

}  // namespace

void buildSweepGridReference(const std::vector<int32_t>& corners, const float* start, const float* end, float cellSize, SweepGrid& out)
{
  const size_t                    n = corners.size() / 3;
  std::vector<std::array<int, 6>> range(n);
  size_t                          entries = 0;
  for(size_t t = 0; t < n; t++)
  {
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(int k = 0; k < 6; k++)
    {
      const int    v = corners[t * 3 + size_t(k % 3)];
      const float* p = k < 3 ? start : end;
      for(int c = 0; c < 3; c++)
      {
        lo[c] = std::min(lo[c], p[size_t(v) * 3 + c]);
        hi[c] = std::max(hi[c], p[size_t(v) * 3 + c]);
      }
    }
    size_t count = 1;
    for(int c = 0; c < 3; c++)
    {
      range[t][c]     = cellOf(lo[c], cellSize);
      range[t][c + 3] = cellOf(hi[c], cellSize);
      count *= size_t(range[t][c + 3] - range[t][c] + 1);
    }
    entries += count;
  }
  size_t slots = 64;
  while(slots < entries * 2)
    slots *= 2;
  std::vector<SweepCell>& table = out.cells;
  table.assign(slots, SweepCell{0, 0, 0, -1, 0, 0, 0, 0});
  const auto find = [&](int ix, int iy, int iz) {
    size_t s = sweepCellHash(ix, iy, iz) & (slots - 1);
    while(table[s].start >= 0 && (table[s].x != ix || table[s].y != iy || table[s].z != iz))
      s = (s + 1) & (slots - 1);
    return s;
  };
  // count per cell, then place: `start` doubles as the "taken" flag, so the first pass stores the
  // count there and the second turns it into offsets
  size_t used = 0;
  for(size_t t = 0; t < n; t++)
    for(int ix = range[t][0]; ix <= range[t][3]; ix++)
      for(int iy = range[t][1]; iy <= range[t][4]; iy++)
        for(int iz = range[t][2]; iz <= range[t][5]; iz++)
        {
          const size_t s = find(ix, iy, iz);
          if(table[s].start < 0)
          {
            table[s] = {ix, iy, iz, 0, 0, 0, 0, 0};
            used++;
          }
          table[s].count++;
        }
  int at = 0;
  for(SweepCell& c : table)
    if(c.start >= 0)
    {
      c.start = at;
      at += c.count;
      c.count = 0;
    }
  out.items.assign(std::max<size_t>(1, entries), 0);
  for(size_t t = 0; t < n; t++)
    for(int ix = range[t][0]; ix <= range[t][3]; ix++)
      for(int iy = range[t][1]; iy <= range[t][4]; iy++)
        for(int iz = range[t][2]; iz <= range[t][5]; iz++)
        {
          SweepCell& c                          = table[find(ix, iy, iz)];
          out.items[size_t(c.start + c.count++)] = int32_t(t);
        }
  out.entries = entries;
  out.used    = used;
}

bool sameSweepGrids(const SweepGrid& a, const SweepGrid& b)
{
  if(a.entries != b.entries || a.used != b.used)
    return false;
  size_t seen = 0;
  for(const SweepCell& c : a.cells)
  {
    if(c.start < 0)
      continue;
    seen++;
    // the cell in b, as the kernels find it
    const uint32_t mask = b.mask();
    uint32_t       s    = sweepCellHash(c.x, c.y, c.z) & mask;
    const SweepCell* d  = nullptr;
    for(uint32_t probe = 0; probe <= mask; probe++)
    {
      const SweepCell& e = b.cells[(s + probe) & mask];
      if(e.start < 0)
        break;
      if(e.x == c.x && e.y == c.y && e.z == c.z)
      {
        d = &e;
        break;
      }
    }
    if(!d || d->count != c.count)
      return false;
    if(!std::equal(a.items.begin() + c.start, a.items.begin() + c.start + c.count, b.items.begin() + d->start))
      return false;
  }
  return seen == a.used;
}

void buildSweepGrid(const std::vector<int32_t>& corners, const float* start, const float* end, float cellSize, SweepGrid& out)
{
  // each face's cells: the box its corners sweep through, on the pool (each face its own)
  const size_t n = corners.size() / 3;
  out.ranges.resize(n);
  jobs::parallelFor(n, 4096, [&](size_t begin, size_t stop) {
    for(size_t t = begin; t < stop; t++)
    {
      float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
      for(int k = 0; k < 6; k++)
      {
        const int    v = corners[t * 3 + size_t(k % 3)];
        const float* p = k < 3 ? start : end;
        for(int c = 0; c < 3; c++)
        {
          lo[c] = std::min(lo[c], p[size_t(v) * 3 + c]);
          hi[c] = std::max(hi[c], p[size_t(v) * 3 + c]);
        }
      }
      auto& r = out.ranges[t];
      for(int c = 0; c < 3; c++)
      {
        r[c]     = cellOf(lo[c], cellSize);
        r[c + 3] = cellOf(hi[c], cellSize);
      }
    }
  });
  size_t entries = 0;
  for(const auto& r : out.ranges)
    entries += size_t(r[3] - r[0] + 1) * size_t(r[4] - r[1] + 1) * size_t(r[5] - r[2] + 1);
  // The table is sized by the cells met, not by the entries (a face is in two to six cells, and many
  // faces share one: 7,000 cells against 110,000 entries on a dress), at most an eighth full: a lookup
  // then reads 1.1 slots to find a cell and 1.2 to find one isn't there, with the hash's mix (2 to 4 MB
  // against the 8 to 32 the entries asked). A frame doesn't know its cells before it meets them: it
  // starts at the last frame's size and doubles when it fills, up to the reference's size, half full at
  // most however many cells there are (faces far apart have a cell each).
  constexpr size_t load = 8;
  size_t           cap  = 64;
  while(cap < entries * 2)
    cap *= 2;
  size_t slots = 64;
  while(slots < load * out.used && slots < cap)
    slots *= 2;
  std::vector<SweepCell>& table = out.cells;
  const SweepCell         empty{0, 0, 0, -1, 0, 0, 0, 0};
  table.assign(slots, empty);
  const auto find = [&](int ix, int iy, int iz) {
    size_t s = sweepCellHash(ix, iy, iz) & (slots - 1);
    while(table[s].start >= 0 && (table[s].x != ix || table[s].y != iy || table[s].z != iz))
      s = (s + 1) & (slots - 1);
    return s;
  };
  const auto grow = [&] {
    std::vector<SweepCell> was;
    was.swap(table);
    slots *= 2;
    table.assign(slots, empty);
    for(const SweepCell& c : was)
      if(c.start >= 0)
        table[find(c.x, c.y, c.z)] = c;
  };
  // count per cell, then place: `start` doubles as the "taken" flag, as in the reference. Each entry's
  // slot is kept for the placing, unless the table grew meanwhile and moved the cells
  size_t used = 0;
  bool   grew = false;
  out.slotOf.resize(entries);
  size_t k = 0;
  for(size_t t = 0; t < n; t++)
  {
    const auto& r = out.ranges[t];
    for(int ix = r[0]; ix <= r[3]; ix++)
      for(int iy = r[1]; iy <= r[4]; iy++)
        for(int iz = r[2]; iz <= r[5]; iz++)
        {
          size_t s = find(ix, iy, iz);
          if(table[s].start < 0)
          {
            if((used + 1) * load > slots && slots < cap)
            {
              grow();
              grew = true;
              s    = find(ix, iy, iz);
            }
            table[s] = {ix, iy, iz, 0, 0, 0, 0, 0};
            used++;
          }
          table[s].count++;
          out.slotOf[k++] = uint32_t(s);
        }
  }
  if(grew)
  {
    k = 0;
    for(size_t t = 0; t < n; t++)
    {
      const auto& r = out.ranges[t];
      for(int ix = r[0]; ix <= r[3]; ix++)
        for(int iy = r[1]; iy <= r[4]; iy++)
          for(int iz = r[2]; iz <= r[5]; iz++)
            out.slotOf[k++] = uint32_t(find(ix, iy, iz));
    }
  }
  int at = 0;
  for(SweepCell& c : table)
    if(c.start >= 0)
    {
      c.start = at;
      at += c.count;
      c.count = 0;
    }
  // the faces into their cells' runs, in ascending order
  out.items.resize(std::max<size_t>(1, entries));
  k = 0;
  for(size_t t = 0; t < n; t++)
  {
    const auto&  r     = out.ranges[t];
    const size_t cells = size_t(r[3] - r[0] + 1) * size_t(r[4] - r[1] + 1) * size_t(r[5] - r[2] + 1);
    for(size_t j = 0; j < cells; j++)
    {
      SweepCell& c                          = table[out.slotOf[k++]];
      out.items[size_t(c.start + c.count++)] = int32_t(t);
    }
  }
  out.entries = entries;
  out.used    = used;
}

}  // namespace fv::physics
