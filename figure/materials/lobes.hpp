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

// Hair cards: a hair surface that is not OmniHair, cut out of its cards by an opacity map. A scalp
// cap or brows (hair by name too) are painted surfaces lying on the skin, not locks of fibres.
bool hairCards(const Surface& s);

// The azimuthal roughness (β_n) a card fibre gets: pbrt's default, which DAZ cards do not author.
constexpr float kCardAzimuthalRoughness = 0.3f;

// The longitudinal roughness (β_m) whose highlight is as wide along the strand as the GGX lobe of
// glTF roughness `roughness` it stands for: the fibre's longitudinal spread (Chiang 2016,
// 0.726 β + 0.812 β² + 3.7 β^20 radians) set to the Gaussian of the same full width at half maximum
// as GGX's reflected lobe at α = roughness² (2.57 α), i.e. 1.09 α. Roughness 0.4 is β_m 0.2; taking
// the roughness as β_m made the highlight about 2.4 times as wide.
float cardLongitudinalRoughness(float roughness);

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
// The same for the white of the eye, in metres. The sclera scatters strongly and barely absorbs, so
// light would travel millimetres in it, but it is a layer about 0.6 to 1 mm thick over the dark
// choroid, which takes up what goes deeper: light comes back out about that far from where it went
// in, far enough to dissolve the half-millimetre stripes of the lashes' shadows.
constexpr Rgb kScleraScatterDistance{0.0009f, 0.0008f, 0.0007f};

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
