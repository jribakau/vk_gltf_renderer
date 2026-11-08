#include "pixel_time.hpp"

#include <algorithm>
#include <map>

#include <nvshaders/slang_types.h>

#include "../shaders/pixel_time.h.slang"

namespace pixel_time {

float Reading::scale() const
{
  return ticks > 0 ? float(double(samples) / double(ticks)) : 0.0F;
}

Reading read(std::span<const uint32_t> sums)
{
  Reading out;
  for(size_t i = 0; i + kSlotWords <= sums.size(); i += kSlotWords)
  {
    const uint64_t samples = sums[i + 2];
    if(samples == 0)
      continue;
    const uint64_t units = (uint64_t(sums[i + 1]) << 32) | sums[i];
    const Slot     slot{.slot = uint32_t(i / kSlotWords), .ticks = units << shaderio::kPixelTimeUnitShift, .samples = samples};
    out.ticks += slot.ticks;
    out.samples += slot.samples;
    out.slots.push_back(slot);
  }
  return out;
}

std::vector<Share> shares(const std::vector<Group>& groups, size_t top, const std::string& rest)
{
  std::map<std::string, Group> merged;
  uint64_t                     ticks = 0, samples = 0;
  for(const Group& g : groups)
  {
    Group& m = merged[g.key];
    m.key    = g.key;
    m.ticks += g.ticks;
    m.samples += g.samples;
    ticks += g.ticks;
    samples += g.samples;
  }
  std::vector<Group> order;
  for(auto& [key, g] : merged)
    order.push_back(g);
  // by time, then by key, so equal shares keep one order
  std::sort(order.begin(), order.end(), [](const Group& a, const Group& b) { return a.ticks != b.ticks ? a.ticks > b.ticks : a.key < b.key; });

  const double average = samples > 0 ? double(ticks) / double(samples) : 0.0;
  const auto   line    = [&](const std::string& key, uint64_t t, uint64_t n) {
    return Share{.key     = key,
                 .share   = ticks > 0 ? double(t) / double(ticks) : 0.0,
                 .cost    = n > 0 && average > 0.0 ? double(t) / double(n) / average : 0.0,
                 .samples = n};
  };
  std::vector<Share> out;
  uint64_t           restTicks = 0, restSamples = 0;
  for(size_t i = 0; i < order.size(); i++)
  {
    if(i < top)
      out.push_back(line(order[i].key, order[i].ticks, order[i].samples));
    else
    {
      restTicks += order[i].ticks;
      restSamples += order[i].samples;
    }
  }
  if(restSamples > 0)
    out.push_back(line(rest, restTicks, restSamples));
  return out;
}

}  // namespace pixel_time
