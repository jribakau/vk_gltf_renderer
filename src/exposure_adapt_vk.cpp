#include "exposure_adapt_vk.hpp"

#include <array>
#include <cstring>

#include <nvapp/application.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "exposure_adapt.hpp"

static_assert(exposure_adapt::kBins == EXPOSURE_HISTOGRAM_SIZE, "the meter reads the tone mapper's histogram as it is");

namespace {

// Where the reading starts counting: a pixel at the middle of the frame counts fully, one this far
// out (of the frame's width) not at all (nvshaders centerMeteringWeight)
constexpr float kMeterSpread = 0.75F;

}  // namespace

void ExposureAdapt::registerParameters(SettingsRegistry* settings)
{
  // Not persisted: the host sends it with the lighting (fv_lighting.cpp), after every load too
  settings->add({"tmAdapt", "Tonemapper: the exposure adapts to the view, metered from the frame [Off:0, On:1]"}, &m_on,
                Persist::eNo, 0, 1);
}

void ExposureAdapt::deinit(Resources& resources)
{
  resources.allocator.destroyBuffer(m_readback);
  m_slots.clear();
}

void ExposureAdapt::update(Resources& resources)
{
  const auto  now     = std::chrono::steady_clock::now();
  const float seconds = m_last == std::chrono::steady_clock::time_point{} ? 0.0F :
                                                                             std::min(std::chrono::duration<float>(now - m_last).count(), 0.5F);
  m_last = now;
  if(!on())
  {
    m_have   = false;
    m_stops  = 0.0F;
    m_target = 0.0F;
    resources.exposureAdapt = 1.0F;
    return;
  }
  // The frame that last used this frame's slot is done: the application waited for it
  if(resources.app && !m_slots.empty())
  {
    const uint32_t slot = resources.app->getFrameCycleIndex() % uint32_t(m_slots.size());
    Slot&          s    = m_slots[slot];
    if(s.metered)
    {
      s.metered = false;
      std::array<uint32_t, exposure_adapt::kBins> bins{};
      std::memcpy(bins.data(), m_readback.mapping + slot * sizeof(bins), sizeof(bins));
      if(const std::optional<float> reading = exposure_adapt::meteredStops(bins))
      {
        // (the bins were counted about mid grey at the exposure of then: from mid grey at the one of now)
        const float exposure = resources.shownTonemapper(false).exposure;
        m_target = exposure_adapt::adaptationFor(*reading + std::log2(std::max(exposure, 1e-30F) / std::max(s.exposure, 1e-30F)));
        if(!m_have)
          m_stops = m_target;
        m_have = true;
      }
    }
  }
  if(m_have)
    m_stops = exposure_adapt::eased(m_stops, m_target, seconds);
  resources.exposureAdapt = std::exp2(m_stops);
}

void ExposureAdapt::cmdMeter(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& image, const VkDescriptorImageInfo& out, VkExtent2D size)
{
  if(!on() || !resources.app)
    return;
  const uint32_t     slots = std::max(resources.app->getFrameCycleSize(), 1u);
  const VkDeviceSize block = sizeof(uint32_t) * exposure_adapt::kBins;
  if(m_slots.size() != slots)
  {
    resources.allocator.destroyBuffer(m_readback);
    NVVK_CHECK(resources.allocator.createBuffer(m_readback, slots * block, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    NVVK_DBG_NAME(m_readback.buffer);
    m_slots.assign(slots, {});
  }
  const uint32_t slot = resources.app->getFrameCycleIndex() % slots;

  // counted about mid grey at the host's exposure (the factor left out: the reading is of the light)
  shaderio::TonemapperData   tm    = resources.shownTonemapper(false);
  const exposure_adapt::Range range = exposure_adapt::meterRange(tm.exposure);
  tm.evMinValue                     = range.evMin;
  tm.evMaxValue                     = range.evMax;
  tm.enableCenterMetering           = 1;
  tm.centerMeteringSize             = kMeterSpread;
  resources.tonemapper.cmdHistogram(cmd, size, tm, image, out, m_readback.buffer, slot * block);
  m_slots[slot] = {.metered = true, .exposure = tm.exposure};
}
