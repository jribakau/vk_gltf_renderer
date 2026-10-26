#pragma once

// Readings of a Surface the glTF lowering needs: how Iray's specular layers fold into one lobe plus
// a clearcoat, whether a surface is skin and how the path tracer shades it, and the medium under a
// volume skin. A port of src/lib/materials/lobes.ts and the volume half of rtx/gltf.ts.

#include <optional>

#include <figure/materials/surface.hpp>

namespace fv::mat {

struct SpecularLobes
{
  // glTF's specularFactor: 1 is a plain dielectric (4% at normal incidence)
  float intensity = 0;
  // the main lobe's roughness: the glossy layer's, or the broad dual lobe's without a glossy layer
  ScalarMap roughness;
  // the tighter dual lobe as a clearcoat, when it is tight enough to show
  struct Coat
  {
    float     weight = 0;
    ScalarMap roughness;
  };
  std::optional<Coat> clearcoat;
  // the share of `intensity` that is Iray's glossy layer, the only layer Glossy Anisotropy stretches
  float glossyShare = 1;
};

// Iray's specular at normal incidence. The glossy layer and the broader of the two dual lobes add
// up into the main lobe; the tighter dual lobe becomes a clearcoat, the one extra lobe glTF has, so
// skin keeps Iray's sharp wet highlight over its broad one.
SpecularLobes specularLobes(const Surface& s);

enum class SkinRole
{
  eNone,
  eFlesh,  // the volume walk (or the thin preview)
  eThin,   // nails and teeth: scatter at the surface only
};

// Eyes, corneas, hair, lashes and anything unnamed are never skin, whatever their translucency.
SkinRole skinRole(const Surface& s);

// Hair's translucency as a layer over its base colour: Iray Uber hair sends Translucency Weight of
// the diffuse light through the strand's Translucency Color. Only from a map of its own.
std::optional<MakeupLayer> hairTranslucency(const Surface& s);

// The colour that stands in for a skin's Translucency Color when the file gives none.
Rgb fallbackTranslucencyColor();

// What a volume skin is made of, for the renderer.
struct VolumeMedium
{
  float distance = 1;  // attenuationDistance
  Rgb   attenuation{1, 1, 1};
  float anisotropy = 0;  // Henyey-Greenstein g (SSS Direction)
  Rgb   albedo{1, 1, 1};
  Rgb   multiscatter{1, 1, 1};  // multiscatterColorFactor: what the renderer's albedo inversion turns back into `albedo`
};

// How far light spreads in human skin, per channel, in metres: the diffuse mean free path
// 1 / sqrt(3 sigma_a sigma_t') of Jensen et al.'s measured skin1 (2001), red 3.7 mm, green 1.4 mm,
// blue 0.7 mm. Pores, about 0.1 to 0.3 mm, are far smaller: their shading melts into the skin.
constexpr Rgb kSkinScatterDistance{0.00367f, 0.00137f, 0.00068f};

// The medium under a volume skin, from Iray's measurements when the surface has them, otherwise
// Jensen's skin1.
VolumeMedium volumeMedium(const std::optional<Subsurface>& sss);

// The renderer's multiToSingleScatterAlbedo, and its inverse by bisection.
float singleScatterAlbedo(float rho, float g);
float multiscatterForAlbedo(float albedo, float g);

// Linear RGB tint of a blackbody at `kelvin`, relative to Iray's neutral 6500 K and at unit
// luminance (emission.ts temperatureTint).
Rgb temperatureTint(float kelvin);

}  // namespace fv::mat
