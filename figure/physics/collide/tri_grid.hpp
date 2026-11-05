#pragma once

// A spatial hash of triangles over their boxes, for particles meeting a surface that moves (cloth
// meeting the posed body and itself): a flat open-addressed table of cells, each with a run of indices
// into the face table, built in two counting passes and reused frame to frame.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fv::physics {

class TriGrid
{
public:
  // A face as the grid holds it: its corners, mesh and box, so a scan touches nothing else until a
  // face is close.
  struct Item
  {
    int     tri, a, b, c, mesh;
    uint8_t self;
    float   lo[3], hi[3];
  };

  // Each of the faces `ids` of `tris` (each with corners a, b, c, a mesh and a self flag) covers the
  // cells of its box over both `a` and `b`, grown by `pad`.
  template <class Faces>
  void build(const Faces& tris, const std::vector<int>& ids, const std::vector<float>& a, const std::vector<float>& b, float pad)
  {
    const size_t n = ids.size();
    m_empty        = n == 0;
    m_face.resize(n);
    m_lo.resize(n * 3);
    m_hi.resize(n * 3);
    size_t entries = 0;
    for(size_t t = 0; t < n; t++)
    {
      const auto& tri = tris[size_t(ids[t])];
      Item&      it  = m_face[t];
      it.tri         = ids[t];
      it.a           = tri.a;
      it.b           = tri.b;
      it.c           = tri.c;
      it.mesh        = tri.mesh;
      it.self        = tri.self;
      for(int c = 0; c < 3; c++)
      {
        it.lo[c] = INFINITY;
        it.hi[c] = -INFINITY;
      }
      for(int k = 0; k < 6; k++)
      {
        const int   v = k % 3 == 0 ? tri.a : k % 3 == 1 ? tri.b : tri.c;
        const auto& p = k < 3 ? a : b;
        for(int c = 0; c < 3; c++)
        {
          it.lo[c] = std::min(it.lo[c], p[v * 3 + c]);
          it.hi[c] = std::max(it.hi[c], p[v * 3 + c]);
        }
      }
      size_t cells = 1;
      for(int c = 0; c < 3; c++)
      {
        it.lo[c] -= pad;
        it.hi[c] += pad;
        m_lo[t * 3 + c] = cellOf(it.lo[c]);
        m_hi[t * 3 + c] = cellOf(it.hi[c]);
        cells *= size_t(m_hi[t * 3 + c] - m_lo[t * 3 + c] + 1);
      }
      entries += cells;
    }
    size_t slots = 64;
    while(slots < entries * 2)
      slots *= 2;
    m_mask = slots - 1;
    m_keys.assign(slots, kEmpty);
    m_count.assign(slots, 0);
    for(size_t t = 0; t < n; t++)
      eachCell(t, [&](uint64_t key) { m_count[insert(key)]++; });
    m_start.resize(slots + 1);
    m_start[0] = 0;
    for(size_t s = 0; s < slots; s++)
      m_start[s + 1] = m_start[s] + m_count[s];
    m_items.resize(entries);
    std::fill(m_count.begin(), m_count.end(), 0);
    for(size_t t = 0; t < n; t++)
      eachCell(t, [&](uint64_t key) {
        const size_t s                                     = find(key);
        m_items[size_t(m_start[s]) + size_t(m_count[s]++)] = int(t);
      });
  }

  // Calls f(item) for every face in the cells of the box; a face in several of them is reported
  // several times.
  template <class F>
  void query(const float lo[3], const float hi[3], F&& f) const
  {
    const int x0 = cellOf(lo[0]), x1 = cellOf(hi[0]), y0 = cellOf(lo[1]), y1 = cellOf(hi[1]), z0 = cellOf(lo[2]), z1 = cellOf(hi[2]);
    for(int ix = x0; ix <= x1; ix++)
      for(int iy = y0; iy <= y1; iy++)
        for(int iz = z0; iz <= z1; iz++)
        {
          const size_t s = find(keyOf(ix, iy, iz));
          if(s == kNone)
            continue;
          for(int k = m_start[s]; k < m_start[s + 1]; k++)
            f(m_face[size_t(m_items[size_t(k)])]);
        }
  }
  bool empty() const { return m_empty; }

private:
  // Grid cell for the collision faces, cm. Cloth and body polygons are about a centimetre.
  static constexpr float kCell = 2.f;
  static int cellOf(float p) { return int(std::floor(p / kCell)); }

  static constexpr uint64_t kEmpty = ~uint64_t(0);
  static constexpr size_t   kNone  = ~size_t(0);

  static uint64_t keyOf(int ix, int iy, int iz)
  {
    return (uint64_t(uint32_t(ix) & 0x1FFFFF) << 42) | (uint64_t(uint32_t(iy) & 0x1FFFFF) << 21) | uint64_t(uint32_t(iz) & 0x1FFFFF);
  }
  static size_t hash(uint64_t key)
  {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return size_t(key);
  }
  size_t insert(uint64_t key)
  {
    size_t s = hash(key) & m_mask;
    while(m_keys[s] != kEmpty && m_keys[s] != key)
      s = (s + 1) & m_mask;
    m_keys[s] = key;
    return s;
  }
  size_t find(uint64_t key) const
  {
    size_t s = hash(key) & m_mask;
    while(m_keys[s] != kEmpty)
    {
      if(m_keys[s] == key)
        return s;
      s = (s + 1) & m_mask;
    }
    return kNone;
  }
  template <class F>
  void eachCell(size_t t, F&& f) const
  {
    for(int ix = m_lo[t * 3]; ix <= m_hi[t * 3]; ix++)
      for(int iy = m_lo[t * 3 + 1]; iy <= m_hi[t * 3 + 1]; iy++)
        for(int iz = m_lo[t * 3 + 2]; iz <= m_hi[t * 3 + 2]; iz++)
          f(keyOf(ix, iy, iz));
  }

  bool                  m_empty = true;
  std::vector<Item>     m_face;
  std::vector<int>      m_items, m_lo, m_hi;
  std::vector<uint64_t> m_keys;
  std::vector<int>      m_count, m_start;
  size_t                m_mask = 0;
};

}  // namespace fv::physics
