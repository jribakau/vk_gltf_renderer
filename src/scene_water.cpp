#include "scene_water.hpp"

#include <algorithm>
#include <cmath>

namespace rs {

namespace {

// The least a UV offset of length 1 can be on the skin, m: toTangent's smaller singular value
double leastStretch(const std::array<float, 4>& m)
{
  const double a = m[0], b = m[1], c = m[2], d = m[3];
  const double f = a * a + b * b + c * c + d * d, det = std::fabs(a * d - b * c);
  return std::sqrt(std::max(0.0, (f - std::sqrt(std::max(0.0, f * f - 4 * det * det))) / 2));
}

float median(std::vector<float>& v)
{
  std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(v.size() / 2), v.end());
  return v[v.size() / 2];
}

}  // namespace

WaterTables packWater(uint32_t slots, std::span<const std::pair<uint32_t, const fv::mat::WaterLayer*>> layers)
{
  WaterTables out;
  out.grids.assign(slots, shaderio::WaterGrid{});
  for(const auto& [slot, layer] : layers)
  {
    if(slot >= slots || !layer)
      continue;
    shaderio::WaterGrid& grid = out.grids[slot];
    grid.damp                 = std::clamp(layer->damp, 0.f, 1.f);

    // each dome's box in UV, out to its halo
    struct Box
    {
      float lo[2], hi[2];
    };
    std::vector<Box>      boxes;
    std::vector<uint32_t> kept;
    std::vector<float>    perU, perV;  // metres a UV unit along u and v
    float                 lo[2] = {INFINITY, INFINITY}, hi[2] = {-INFINITY, -INFINITY};
    for(uint32_t i = 0; i < layer->items.size(); i++)
    {
      const fv::mat::WaterItem& item  = layer->items[i];
      const double              least = leastStretch(item.toTangent);
      if(!(least > 1e-6) || !(item.radius > 0) || !(item.height > 0))
        continue;
      const float reach = float(kWaterHalo * item.radius / least);
      Box         box{{std::min(item.a[0], item.b[0]) - reach, std::min(item.a[1], item.b[1]) - reach},
                      {std::max(item.a[0], item.b[0]) + reach, std::max(item.a[1], item.b[1]) + reach}};
      if(!std::isfinite(box.lo[0]) || !std::isfinite(box.lo[1]) || !std::isfinite(box.hi[0]) || !std::isfinite(box.hi[1]))
        continue;
      for(int k = 0; k < 2; k++)
      {
        lo[k] = std::min(lo[k], box.lo[k]);
        hi[k] = std::max(hi[k], box.hi[k]);
      }
      boxes.push_back(box);
      kept.push_back(i);
      const auto& m = item.toTangent;
      perU.push_back(std::sqrt(m[0] * m[0] + m[2] * m[2]));
      perV.push_back(std::sqrt(m[1] * m[1] + m[3] * m[3]));
    }
    if(kept.empty())
      continue;

    // cells about kWaterCell across on the skin, as the domes' UVs mostly stretch
    const float    extent[2] = {std::max(hi[0] - lo[0], 1e-6f), std::max(hi[1] - lo[1], 1e-6f)};
    const float    scale[2]  = {median(perU) / kWaterCell, median(perV) / kWaterCell};
    const uint32_t w = std::clamp(uint32_t(std::ceil(extent[0] * scale[0])), 1u, kWaterCellsMax);
    const uint32_t h = std::clamp(uint32_t(std::ceil(extent[1] * scale[1])), 1u, kWaterCellsMax);
    grid.uvMin       = {lo[0], lo[1]};
    grid.cellsPerUv  = {float(w) / extent[0], float(h) / extent[1]};
    grid.width       = w;
    grid.height      = h;
    grid.firstCell   = uint32_t(out.cells.size());

    const uint32_t firstItem = uint32_t(out.items.size());
    for(uint32_t k : kept)
    {
      const fv::mat::WaterItem& item = layer->items[k];
      shaderio::WaterItem       gpu{};
      gpu.toTangent = {item.toTangent[0], item.toTangent[1], item.toTangent[2], item.toTangent[3]};
      gpu.a         = {item.a[0], item.a[1]};
      gpu.b         = {item.b[0], item.b[1]};
      gpu.radius    = item.radius;
      gpu.height    = item.height;
      out.items.push_back(gpu);
    }
    // the cells each dome reaches into: counted, then filled in their runs
    const auto span = [&](const Box& b, uint32_t& x0, uint32_t& x1, uint32_t& y0, uint32_t& y1) {
      const auto at = [](float v, float lo, float per, uint32_t n) { return uint32_t(std::clamp(int64_t(std::floor((v - lo) * per)), int64_t(0), int64_t(n) - 1)); };
      x0 = at(b.lo[0], lo[0], grid.cellsPerUv.x, w);
      x1 = at(b.hi[0], lo[0], grid.cellsPerUv.x, w);
      y0 = at(b.lo[1], lo[1], grid.cellsPerUv.y, h);
      y1 = at(b.hi[1], lo[1], grid.cellsPerUv.y, h);
    };
    std::vector<uint32_t> count(size_t(w) * h, 0);
    for(const Box& b : boxes)
    {
      uint32_t x0, x1, y0, y1;
      span(b, x0, x1, y0, y1);
      for(uint32_t y = y0; y <= y1; y++)
        for(uint32_t x = x0; x <= x1; x++)
          count[size_t(y) * w + x]++;
    }
    uint32_t at = uint32_t(out.refs.size());
    out.cells.resize(size_t(grid.firstCell) + count.size());
    for(size_t c = 0; c < count.size(); c++)
    {
      out.cells[grid.firstCell + c] = {at, count[c]};
      at += count[c];
    }
    out.refs.resize(at);
    std::fill(count.begin(), count.end(), 0u);
    for(uint32_t j = 0; j < boxes.size(); j++)
    {
      uint32_t x0, x1, y0, y1;
      span(boxes[j], x0, x1, y0, y1);
      for(uint32_t y = y0; y <= y1; y++)
        for(uint32_t x = x0; x <= x1; x++)
        {
          const size_t c                                   = size_t(y) * w + x;
          out.refs[out.cells[grid.firstCell + c].x + count[c]++] = firstItem + j;
        }
    }
  }
  return out;
}

}  // namespace rs
