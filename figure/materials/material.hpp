#pragma once

// The material the renderer keeps and draws from, whatever it came from: a DAZ surface lowers into
// it (lower.hpp), a glTF file's material converts into it (material_gltf.hpp), and the shader's row
// is packed from it (nvvkgltf::MaterialCache). glTF is what it is written to only when writing a file.
//
// A lobe that is left out is not the same as one at zero: which ones are present decide the shader's
// features (nvvkgltf::detectMaterialFeatures), so they're optional.

#include <array>
#include <cfloat>
#include <map>
#include <optional>
#include <string>

#include <tinygltf/tiny_gltf.h>

#include <figure/materials/surface.hpp>

namespace fv::mat {

using Rgba = std::array<float, 4>;

// KHR_texture_transform's offset, rotation (radians) and scale: uv' = offset + R(rotation) · scale · uv,
// with R(θ) = [[cos θ, sin θ], [−sin θ, cos θ]] (the spec's example and sample viewer, not the
// matrix its README prints).
struct UvTransform
{
  std::array<float, 2> scale{1, 1};
  std::array<float, 2> offset{0, 0};
  float                rotation = 0;
  bool                 identity() const;
  bool                 operator==(const UvTransform&) const = default;
};

// One map of a material.
struct TextureSlot
{
  int                        index    = -1;  // in the material's texture table; in a Lowered, in its requests. -1: no map
  int                        texCoord = 0;
  std::optional<float>       scale;           // a normal map's strength, an occlusion map's
  std::optional<UvTransform> transform;       // left out: the identity, which the shader then doesn't pay for
  bool                       flipGreen = false;  // the image's green points the other way (Y down): the shader negates it

  // What a file carried here that nothing reads, to write back: the transform's own texCoord (the
  // renderer picks the UV set by `texCoord`), other extensions, extras
  std::optional<int>     transformTexCoord;
  tinygltf::ExtensionMap foreignExtensions;
  tinygltf::Value        extras;

  bool has() const { return index >= 0; }
  bool operator==(const TextureSlot&) const = default;
};

enum class AlphaMode
{
  eOpaque,
  eMask,
  eBlend,
};

struct Specular
{
  float       factor = 1;
  Rgb         color{1, 1, 1};
  TextureSlot map, colorMap;
  bool        operator==(const Specular&) const = default;
};

struct Clearcoat
{
  float       factor = 0, roughness = 0;
  TextureSlot map, roughnessMap, normalMap;
  bool        operator==(const Clearcoat&) const = default;
};

// A second specular lobe, under the coat: DAZ's tight dual lobe, skin's wet highlight over its broad
// one. No layer, so it darkens nothing under it, and it lies on the base's normal.
struct Specular2
{
  float       factor = 0, roughness = 0;
  TextureSlot map, roughnessMap;  // the weight in red, the roughness in green
  bool        operator==(const Specular2&) const = default;
};

struct Sheen
{
  Rgb         color{0, 0, 0};
  float       roughness = 0;
  TextureSlot colorMap, roughnessMap;
  bool        operator==(const Sheen&) const = default;
};

struct Transmission
{
  float       factor = 0;
  TextureSlot map;
  bool        operator==(const Transmission&) const = default;
};

struct Volume
{
  float       thickness = 0;
  TextureSlot thicknessMap;
  float       attenuationDistance = FLT_MAX;
  Rgb         attenuationColor{1, 1, 1};
  bool        operator==(const Volume&) const = default;
};

struct Scatter
{
  float       strength = 0;
  TextureSlot strengthMap;
  Rgb         multiscatterColor{1, 1, 1};
  TextureSlot multiscatterColorMap;
  float       anisotropy = 0;
  bool        operator==(const Scatter&) const = default;
};

struct Anisotropy
{
  float       strength = 0, rotation = 0;
  TextureSlot map;
  bool        operator==(const Anisotropy&) const = default;
};

struct Iridescence
{
  float       factor = 0, ior = 1.3f, thicknessMinimum = 100, thicknessMaximum = 400;
  TextureSlot map, thicknessMap;
  bool        operator==(const Iridescence&) const = default;
};

struct DiffuseTransmission
{
  float       factor = 0;
  Rgb         color{1, 1, 1};
  TextureSlot map, colorMap;
  bool        operator==(const DiffuseTransmission&) const = default;
};

struct Retroreflection
{
  float       factor = 0;
  TextureSlot map;
  bool        operator==(const Retroreflection&) const = default;
};

struct SpecularGlossiness
{
  Rgba        diffuse{1, 1, 1, 1};
  Rgb         specular{1, 1, 1};
  float       glossiness = 1;
  TextureSlot diffuseMap, specularGlossinessMap;
  bool        operator==(const SpecularGlossiness&) const = default;
};

// DAZ PBRSkin's detail layer: a tiled normal map bending the normal on top of the normal map and the
// coat's (scaled by its slot's `scale` and by the weight), and a roughness multiplier under the weight.
struct Detail
{
  float       weight = 0, roughness = 1;
  TextureSlot normalMap, weightMap, roughnessMap;
  bool        operator==(const Detail&) const = default;
};

// A strand shaded as a fibre: OmniHair's β_m and β_n, Chiang's σ_a at root and tip, blended by the
// ribbon's along-UV, whose ramp at the scalp is rootFade and whose root is at rootV.
struct Hair
{
  float longitudinal = 0.2f, azimuthal = 0.2f;
  Rgb   rootAbsorption{0, 0, 0}, tipAbsorption{0, 0, 0};
  float rootFade = 0, rootV = 0;
  float specular = 1;  // scales the reflecting orders
  // hair cards: the absorption from the colour each hit shows (Chiang's fit inverted), not root and tip
  bool absorptionFromColor = false;
  // the strand's direction: across it is the UV tangent turned by this angle (radians); unset, the
  // strand is the UV bitangent
  std::optional<float> strandRotation;
  bool                 operator==(const Hair&) const = default;
};

struct Material
{
  std::string name;
  std::string id;  // who it is in the scene ("m:2"): what the host's tools call it

  Rgba        baseColor{1, 1, 1, 1};  // alpha is the opacity
  TextureSlot baseColorMap;
  float       metallic = 1, roughness = 1;
  TextureSlot metallicRoughnessMap;  // roughness in green, metalness in blue, each times its factor
  AlphaMode   alphaMode   = AlphaMode::eOpaque;
  float       alphaCutoff = 0.5f;
  bool        doubleSided = false;
  TextureSlot normalMap, occlusionMap;

  Rgb                  emissive{0, 0, 0};
  TextureSlot          emissiveMap;
  std::optional<float> emissiveStrength;  // left out: 1

  std::optional<Specular>            specular;
  std::optional<float>               ior;  // left out: 1.5
  std::optional<Clearcoat>           clearcoat;
  std::optional<Specular2>           specular2;
  std::optional<Sheen>               sheen;
  std::optional<Transmission>        transmission;
  std::optional<Volume>              volume;
  std::optional<Scatter>             scatter;
  std::optional<Rgb>                 scatterDistance;  // how far light spreads under the surface, per channel, in scene units
  std::optional<Anisotropy>          anisotropy;
  std::optional<Iridescence>         iridescence;
  std::optional<float>               dispersion;
  std::optional<DiffuseTransmission> diffuseTransmission;
  std::optional<Retroreflection>     retroreflection;
  std::optional<SpecularGlossiness>  specularGlossiness;
  std::optional<Detail>              detail;
  std::optional<Hair>                hair;
  std::optional<Rgba>                dlssNrMask;  // per-material DLSS NR strengths; left out: 1, 1, 1, 1
  bool                               unlit = false;

  bool walkThrough = false;  // a surface a scatter walk passes through: light under the skin ignores it

  // What a file carried that nothing reads, to write back
  tinygltf::ExtensionMap foreignExtensions;
  tinygltf::Value        foreignExtras;

  bool operator==(const Material&) const = default;
};

// Calls fn(slot) for each of the material's maps, in the order they're packed in its row
template <class M, class Fn>
void forEachSlot(M& m, Fn&& fn)
{
  fn(m.emissiveMap);
  fn(m.normalMap);
  fn(m.baseColorMap);
  fn(m.metallicRoughnessMap);
  fn(m.occlusionMap);
  if(m.transmission)
    fn(m.transmission->map);
  if(m.volume)
    fn(m.volume->thicknessMap);
  if(m.clearcoat)
  {
    fn(m.clearcoat->roughnessMap);
    fn(m.clearcoat->map);
    fn(m.clearcoat->normalMap);
  }
  if(m.specular2)
  {
    fn(m.specular2->map);
    fn(m.specular2->roughnessMap);
  }
  if(m.specular)
  {
    fn(m.specular->map);
    fn(m.specular->colorMap);
  }
  if(m.iridescence)
  {
    fn(m.iridescence->map);
    fn(m.iridescence->thicknessMap);
  }
  if(m.anisotropy)
    fn(m.anisotropy->map);
  if(m.sheen)
  {
    fn(m.sheen->colorMap);
    fn(m.sheen->roughnessMap);
  }
  if(m.specularGlossiness)
  {
    fn(m.specularGlossiness->diffuseMap);
    fn(m.specularGlossiness->specularGlossinessMap);
  }
  if(m.diffuseTransmission)
  {
    fn(m.diffuseTransmission->map);
    fn(m.diffuseTransmission->colorMap);
  }
  if(m.retroreflection)
    fn(m.retroreflection->map);
  if(m.scatter)
  {
    fn(m.scatter->strengthMap);
    fn(m.scatter->multiscatterColorMap);
  }
  if(m.detail)
  {
    fn(m.detail->normalMap);
    fn(m.detail->weightMap);
    fn(m.detail->roughnessMap);
  }
}

}  // namespace fv::mat
