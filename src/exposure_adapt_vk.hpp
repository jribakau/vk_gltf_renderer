#pragma once

#include <chrono>
#include <vector>

#include <nvvk/resources.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"

//-------------------------------------------------------------------------------------------------------
// ExposureAdapt
//
// Exposure that adapts to the view (tmAdapt, the host's tonemap.adapt; the arithmetic and its
// constants in src/exposure_adapt.hpp). Each frame the tone mapper's histogram of the image it takes
// goes into a slot of a host-visible buffer, a slot per frame in flight; at a later frame's start,
// once the application has waited for the frame that filled it, the slot is read, and the
// adaptation eases toward what it asks for. It is a factor on the host's exposure,
// Resources::exposureAdapt, which Resources::shownTonemapper applies: the tone mapper, the HDR image,
// the still's adaptive stop and its denoiser's threshold, local exposure, an .exr and the pixel
// readout take the exposure as adapted, and the path tracer's firefly clamps follow it.
//
//   exposureAdapt.update(resources);                          // a frame's start, before the trace
//   exposureAdapt.cmdMeter(cmd, resources, image, out, size);  // in runTonemapPass, before the lens
//
// Off, the factor is 1, nothing is metered and nothing is allocated; on again, its first reading is
// taken at once rather than eased into.
//-------------------------------------------------------------------------------------------------------
class ExposureAdapt
{
public:
  void registerParameters(SettingsRegistry* settings);
  void deinit(Resources& resources);

  bool on() const { return m_on != 0; }

  // A frame's start, before anything reads the exposure: Resources::exposureAdapt
  void update(Resources& resources);

  // The histogram of `image`, the tone mapper's input before the lens, bloom and local exposure, into
  // this frame's slot; `out` the tone mapper's output, which its passes' descriptors name
  void cmdMeter(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& image, const VkDescriptorImageInfo& out, VkExtent2D size);

  // The stops the exposure is opened by now (closed: negative)
  float stops() const { return m_stops; }

private:
  struct Slot
  {
    bool  metered  = false;
    float exposure = 1.0F;  // the host's, shown, when it was metered: the bins' range follows it
  };

  int                                   m_on = 0;
  nvvk::Buffer                          m_readback;
  std::vector<Slot>                     m_slots;
  float                                 m_stops  = 0.0F;  // the adaptation now
  float                                 m_target = 0.0F;  // and what the last reading asked for
  bool                                  m_have   = false;
  std::chrono::steady_clock::time_point m_last{};
};
