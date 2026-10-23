#pragma once

// DAZ .duf files that aren't models: pose, material, camera, light and render-settings presets.
// The loader recognises them up front so a host can apply what it supports and name the rest,
// instead of failing with "no geometry". A port of src/lib/loaders/dson/presets.ts.
// Applying an environment preset is lighting (P4); the image and its settings are reported here.

#include <optional>
#include <string>

#include <figure/dson/store.hpp>

namespace fv::dson {

struct EnvironmentPreset
{
  // Library path of the HDR, e.g. "/Runtime/Textures/.../DTHDR-OutdoorA.hdr".
  std::string           image;
  std::optional<double> intensity;
  // Dome rotation in degrees.
  std::optional<double> rotation;
  // Whether the HDR is drawn as the background ("Draw Dome").
  std::optional<bool>   drawDome;
};

struct PresetInfo
{
  // Human name of the preset kind, e.g. "pose preset".
  std::string                        kind;
  std::optional<EnvironmentPreset>   environment;
};

// A .duf meant to be put on a figure that is loaded (clothing, hair, anatomy, hand props): marked a
// wearable, or with geometry nodes parented to the selection.
bool isWearable(const Json& doc);

// None when the file builds geometry; otherwise what kind of preset it is and any environment it sets.
std::optional<PresetInfo> describePreset(const Json& doc);

}  // namespace fv::dson
