#pragma once

// A surface as DAZ describes it, independent of any renderer: the values parsed from an Iray Uber,
// PBRSkin or legacy material (dson/materials.hpp parseSurface), which materials/lower.hpp turns
// into the renderer's glTF material. A port of src/lib/materials/surface.ts, layers.ts and the
// Surface half of daz.ts.
//
// Colours are linear RGB. Textures are library-relative image paths, empty when there is none.

#include <array>
#include <map>
#include <optional>
#include <string>

namespace fv::mat {

using Rgb = std::array<float, 3>;

enum class Kind
{
  eSkin,
  eCornea,
  eEye,
  eLashes,
  eNails,
  eTeeth,
  eMouth,
  eHair,
  eOther,
};

// What a surface's name says it is ("Face", "Eye Moisture Left", "Eyelashes", ...), the same rules
// as classifySurface in daz.ts.
Kind classify(const std::string& name);

// A colour with an optional map that multiplies it.
struct ColorMap
{
  Rgb         color{1, 1, 1};
  std::string map;
};

// A value with an optional map that multiplies it.
struct ScalarMap
{
  float       value = 0;
  std::string map;
};

// Iray's glossy layer: a dielectric specular lobe over the diffuse base.
struct Glossy
{
  float     weight       = 0.5f;  // Glossy Layered Weight (PBRSkin has none: 0)
  float     reflectivity = 0.5f;  // Principled-style: 0.5 is a plain dielectric (4% at normal incidence)
  ScalarMap roughness{0.6f, {}};  // also the roughness of a material with no glossy layer
};

// Iray's dual-lobe specular: two lobes sharing one reflectivity, split by `ratio`.
struct DualLobe
{
  float     weight       = 0;
  float     reflectivity = 0.5f;
  ScalarMap lobe1, lobe2;
  float     ratio = 1;  // lobe 1's share of the weight
};

// Iray's volume under the skin, for the light that enters: the colour left after travelling
// `transmittedDistance` (Transmitted Color) and the scattering measured over `scatterDistance`
// (SSS Color or Amount), with its direction. Distances in centimetres, DAZ's unit.
struct Subsurface
{
  Rgb         transmittedColor{1, 1, 1};
  float       transmittedDistance = 0;
  Rgb         scatterColor{1, 1, 1};
  std::string scatterMap;
  float       scatterDistance = 0;
  float       direction       = 0;  // Henyey-Greenstein anisotropy, -1 back to 1 forward
};

// Iray's translucency: the share of diffuse light that enters the skin instead of reflecting.
struct Translucency
{
  float                     weight = 0;
  std::optional<Rgb>        color;  // Translucency Color; none when the material gives none
  std::string               map;
  Rgb                       reflectanceTint{1, 1, 1};  // SSS Reflectance Tint, on the light that comes back out
  std::optional<Subsurface> subsurface;
};

// PBRSkin's detail layer: a small normal or height map tiled across the surface's UV tile on top of
// its normal and bump maps.
struct DetailLayer
{
  std::string map;
  bool        heightMap = false;  // else a normal map
  float       strength  = 1;
  float       weight    = 0;  // Detail Weight, 0..1
  std::string weightMap;
  std::array<float, 2> tiles{1, 1};
  std::array<float, 2> offset{0, 0};
  float       roughness = 1;  // Detail Specular Roughness Mult
  std::string roughnessMap;
  // A height map's relief as a fraction of one texture tile (TextureRequest::bumpHeight), fixed
  // instead of derived from the tiling: a built-in fabric's, so changing its scale cooks nothing.
  std::optional<float> bumpHeight;
};

// PBRSkin's makeup layer: a colour laid over the base colour by Makeup Weight.
struct MakeupLayer
{
  Rgb         color{1, 1, 1};
  std::string map;
  float       weight = 0;
  std::string weightMap;
};

// Iray emitters: brightness is resolved against the scene's exposure when the glTF is built.
struct Luminance
{
  float amount  = 0;  // cd/m2, or for a flux (lm, W) the total
  bool  perArea = false;
  bool  twoSided = false;
};

struct Emission
{
  ColorMap                color;
  float                   temperature = 6500;
  std::optional<Luminance> luminance;
};

// A map's own tiling over the surface's, from its DAZ texture instance (image_modification): it
// samples at tiles * (the surface's uv) + offset.
struct MapTiling
{
  std::array<float, 2> tiles{1, 1};
  std::array<float, 2> offset{0, 0};

  bool operator==(const MapTiling&) const = default;
};

// The shader a DSON material names: its extra's type ("studio/material/uber_iray", "daz_brick",
// "daz_shader") and what defines it, a brick's library file#id (".../PBRSkin.dsf#PBRSkin") or a
// shader's definition (".dse"). Iray Uber is defined inside DAZ Studio, so its url is empty. Empty
// when no layer names one (DAZ Default, a preset's overrides alone).
struct ShaderRef
{
  std::string type;
  std::string url;

  bool operator==(const ShaderRef&) const = default;
};

struct Surface
{
  std::string name;
  Kind        kind = Kind::eOther;
  // Where the values came from: an Iray Uber or PBRSkin material, a 3Delight-era shader read for
  // what its channels mean (dson/legacy_materials.hpp), or another's top-level channels
  enum class Shader
  {
    eIray,
    ePbrSkin,
    eDazDefault,
    eUberSurface,
    eAoa,
    eLegacy,
  } shader = Shader::eLegacy;
  // The shader the material's files name, kept as they say it. `shader` is still read from the
  // channels present: an Iray preset over a 3Delight material names no shader of its own.
  ShaderRef dazShader;

  ColorMap  base;
  ScalarMap opacity{1, {}};  // Cutout Opacity
  // The cutout map is read from its alpha channel (DAZ's texture instance grayscale_mode "alpha"),
  // not as grey
  bool opacityFromAlpha = false;
  // Iray Uber's Horizontal/Vertical Tiles and Offset: every map samples at uv * tiles + offset
  std::array<float, 2> tiles{1, 1};
  std::array<float, 2> offset{0, 0};
  // The maps a texture instance tiles over that (a fabric print 2.5 times across), by image. An
  // image two channels tile differently is left out, as is DAZ's default of no tiling of its own.
  std::map<std::string, MapTiling> mapTiling;
  // normal map with its strength, and the bump map with DAZ's Bump Strength
  std::string normalMap;
  float       normalStrength = 1;
  std::string bumpMap;
  float       bumpStrength = 1;
  // The bump map's full height in metres, as a legacy shader gives it (Negative/Positive Bump,
  // Bump Minimum/Maximum: DAZ's centimetres). None for Iray's, which bakes kBumpHeight.
  std::optional<float> bumpRange;

  Glossy                  glossy;
  std::optional<DualLobe> dual;  // none when the material has no dual-lobe channels
  ScalarMap               metallic;
  struct Refraction
  {
    float weight = 0;
    float ior    = 1.4f;
    // Refraction Weight's map: where the surface refracts (white) and where it shows itself (black)
    std::string map;
  };
  std::optional<Refraction> refraction;
  struct TopCoat
  {
    float weight    = 0;
    float roughness = 0.1f;
    // Top Coat Weight's map (where the coat is: a lip-gloss mask) and Top Coat Roughness's
    std::string weightMap;
    std::string roughnessMap;
    // PBRSkin's Top Coat Bump Weight: how much of the surface's relief the coat follows, 0 smooth
    float bumpWeight = 1;
    // linear; PBRSkin's Top Coat Color, which tints the coat's reflection only
    Rgb color{1, 1, 1};
  };
  std::optional<TopCoat>      topCoat;
  std::optional<Emission>     emission;
  std::optional<Translucency> translucency;  // none when the material has no translucency channels
  std::optional<DetailLayer>  detail;
  std::optional<MakeupLayer>  makeup;
  // Glossy Anisotropy, 0..1. None when the material has no such channel; 0 is an authored round highlight.
  std::optional<float> glossyAnisotropy;
  // Seen from both sides: strand hair, drawn as flat ribbons that face one way
  bool twoSided = false;
  // OmniHair fibre, when this surface is strand hair. Absorption is Chiang's σ_a at the root and
  // the tip; the ribbon's along-UV blends them. None for card hair and for every other shader.
  struct HairFibre
  {
    float longitudinal = 0.2f;  // OmniHair Roughness, 0..1, the longitudinal β
    float azimuthal    = 0.2f;  // Azimuthal Roughness, 0..1
    Rgb   rootSigma{0, 0, 0};
    Rgb   tipSigma{0, 0, 0};
  };
  std::optional<HairFibre> hairFibre;
};

// What the Material panel can change on a surface. Each field is set only once edited, so the glTF
// material keeps its own derivation of the rest. Colours are linear.
struct Edits
{
  std::optional<Rgb>         color;
  std::optional<float>       roughness, metalness, opacity;
  std::optional<std::string> alphaMode;  // opaque | cutout | hash | blend
  std::optional<bool>        doubleSided;
  std::optional<float>       normalStrength, bumpStrength, specular, clearcoat, clearcoatRoughness;
  std::optional<Rgb>         emissive;
  std::optional<float>       emissiveIntensity;
  // an eye's iris recoloured towards this, by irisStrength (0..1, 1 when unset), and its white
  // multiplied by scleraColor (eyeLayout says which surfaces have them)
  std::optional<Rgb>         irisColor, scleraColor;
  std::optional<float>       irisStrength;
  // the base colour's mean moved to this, its detail kept (a hair colour), by recolorStrength (0..1,
  // 1 when unset): baked into the base map, or the flat colour lerped when there is none
  std::optional<Rgb>         recolor;
  std::optional<float>       recolorStrength;
  // a skin tone: the tone reference's mean (LowerOptions::toneReference) moved to this by
  // toneStrength (0..1, 1 when unset), every skin surface by the same shift so their seams stay
  // matched; baked into the base map
  std::optional<Rgb>         tone;
  std::optional<float>       toneStrength;
  // PBRSkin makeup: its colour (a lipstick's; over a weight map it replaces the colour map) and its
  // weight, 0..1. Only on a surface with a makeup layer.
  std::optional<Rgb>         makeupColor;
  std::optional<float>       makeupWeight;
  // measured skin: how far light spreads under it, times human skin's (kSkinScatterDistance)
  std::optional<float>       scatterScale;
  // A map slot's image replaced by another library-relative image ("/Runtime/Textures/..."), or
  // taken away by "". The roughness map is the one the lowering uses (specularLobes).
  std::optional<std::string> baseMap, opacityMap, normalMap, bumpMap, roughnessMap;
  // The surface's maps tiled, moved and turned over its own tiling (Surface::tiles, offset): the
  // tiles times its own, an offset in UV units, a rotation in degrees (KHR_texture_transform's sense)
  std::optional<float>       uvScaleU, uvScaleV, uvOffsetU, uvOffsetV, uvRotation;
  // A built-in fabric's weave as a detail layer (fabric.hpp; "" none), its size times its natural
  // size and its normal's strength
  std::optional<std::string> fabric;
  std::optional<float>       fabricScale, fabricStrength;
  // KHR_materials_sheen: its colour (none or black: no sheen) and roughness (0.5 when unset)
  std::optional<Rgb>         sheenColor;
  std::optional<float>       sheenRoughness;

  bool empty() const;
  // Fields set in `o` replace those here.
  void merge(const Edits& o);
};

// How a surface's colour map holds an eye: only the iris (Genesis 8 "Irises"), the iris and the
// white in one image (Genesis 9 "Eye Left"/"Eye Right", both irises in one map), only the white
// (Genesis 8 "Sclera"); none for the rest (pupils, corneas, everything else).
enum class EyeLayout
{
  eNone,
  eIris,
  eShared,
  eSclera,
};
EyeLayout eyeLayout(const Surface& s);

// three's SRGBToLinear and its inverse, for the colours the panel exchanges as #rrggbb.
float srgbToLinear(double c);
float linearToSrgb(double c);

}  // namespace fv::mat
