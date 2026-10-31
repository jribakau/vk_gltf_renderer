#pragma once

// A Surface as the material the path tracer (and the rasterizer, which reads the same model)
// evaluates: Iray's layers folded into our Material. Built from the Surface and the panel's edits
// alone.
//
// The lowering is pure: textures come out as requests (what to cook from which image) that a host
// resolves into texture indices, so a live edit that changes none of them can leave the textures
// alone.

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <tinygltf/tiny_gltf.h>

#include <figure/materials/material.hpp>
#include <figure/materials/material_gltf.hpp>
#include <figure/materials/surface.hpp>
#include <figure/materials/texture_request.hpp>

namespace fv::mat {

// How skin is sent to the renderer. Volume is the path tracer's converged look; thin is its preview
// while the camera moves. The rasterizer would show transmission as see-through skin, so a
// real-time view takes none of it: plain diffuse and specular under the base colour. Measured is
// a volume too, made from measured human skin instead of the file's Iray medium (kSkinScatterDistance).
enum class SkinMode
{
  eRaster,
  eThin,
  eVolume,
  eMeasured,
};

struct LowerOptions
{
  SkinMode skin = SkinMode::eVolume;
  // Emissive strength: an Iray emitter's physical brightness, 1 for a plain glow.
  float emission = 1;
  // Hair: the anisotropy rotation measured from the cards, in the renderer's convention.
  std::optional<float> anisotropyRotation;
  // A file for other tools: cutouts are MASK (they sort blended surfaces), the renderer's own
  // extensions are left out and the anisotropy turns the way the spec's does.
  bool forFile = false;
  // The build turned V over (the textures are used as their files have them): a tiled texture's
  // V offset is mirrored to match.
  bool flipV = false;
  // An iris surface (EyeLayout::eIris): where each eye's iris is in its map, centre x, y and radius
  // as fractions of the image, y down (build::irisDiscs, from its UVs). None: its colour can't be changed.
  std::vector<std::array<float, 3>> irisDiscs;
  // The figure's skin map a skin tone shifts every skin surface against (dson::toneReference): the
  // same for all of them, so they move together. Empty: each against its own map.
  std::string toneReference;
  // How much surface one UV unit covers, in metres (Material::metresPerUv): a bump map's height
  // becomes slopes against it. 0: the bump maps keep the fixed slope per texel they had before.
  float metresPerUv = 0;
  // An eye surface, or a cornea, that the eye's tear film covers (a refracting moisture shell on the
  // figure, eyeShell): tissue under water reflects next to nothing, so it takes no specular.
  bool underTearFilm = false;
  // A surface strand-hair ribbons draw with (mesh::BuiltMesh::ribbon): they face one way, so it is
  // seen from both sides whatever the surface or an edit says.
  bool ribbon = false;
};

// The eye's wet shells. The tear film (a refracting moisture shell over the whole eye, or a
// refracting cornea where there is none) is the one interface of an eye in air that reflects, at
// Fresnel's full strength; the lid margin's meniscus is the same water. Surfaces of other kinds,
// and an eye shell that does not refract, are eNone.
enum class EyeShell
{
  eNone,
  eMoisture,
  eCornea,
  eMeniscus,
};
EyeShell eyeShell(const Surface& s);

// The tear film's index: the authored one where it is a watery eye's, else the cornea's
float tearFilmIor(const Surface& s);

// A bump map's full height range at Bump Strength 1, in metres. Set by eye: at 1 mm, CJ Evangeline's
// face (Bump Strength 2, as Iray presets often have it) came out leathery, each grey step of the
// JPEG a visible bump; a quarter of that reads as her skin.
constexpr float kBumpHeight = 0.00025f;

// The height a bump map of `metresPerUv` (scaled by `tiles` repeats) gets in UV units, the
// TextureRequest::bumpHeight the bake divides by; 0 when the size is unknown. Two significant figures.
// `height` is the map's full range in metres: a legacy shader's own (Surface::bumpRange), else kBumpHeight.
float bumpHeightInUv(float metresPerUv, float tiles = 1, float height = kBumpHeight);

// The values the Material panel shows for a surface: what the lowering used, edits included.
struct Effective
{
  Rgb                  color{1, 1, 1};
  float                roughness = 0, metalness = 0, opacity = 1;
  std::string          alphaMode = "opaque";  // opaque | cutout | blend
  bool                 doubleSided = false;
  std::optional<float> normalStrength, bumpStrength;  // set when the surface has such a map
  float                specular = 0, clearcoat = 0, clearcoatRoughness = 0.1f;
  Rgb                  emissive{0, 0, 0};
  float                emissiveIntensity = 1;
  bool                 colorBaked = false;  // the colour is in the base map's bake: a change to it cooks a texture
  // The colour can't be seen: a refracting surface looked straight through, with nothing to tint
  bool colorUnused = false;
  // Strand hair: the path tracer shades the fibre itself, whose colour isn't the base colour
  bool fibre = false;
  // What the clearcoat values move: "coat", the file's top coat; "tight", the sharper dual lobe, a
  // highlight of the surface and no layer over it; "" for neither, so a change adds a coat
  std::string coatRole;
  // The values that multiply a map, whose number is a factor and not what the surface shows
  std::vector<std::string> mapped;
};

// A material with its slots' indices naming `requests`, not textures
struct Lowered
{
  Material                    material;
  std::vector<TextureRequest> requests;
  Effective                   effective;
};

Lowered lower(const Surface& s, const Edits& edits, const LowerOptions& options);

// The surface with the edits' map slots laid over it (Edits::baseMap and the rest; "" takes a map
// away), what lower() starts from. The roughness map is not a Surface field of its own: it is
// specularLobes' (lobes.hpp), and lowerRoughnessMap gives it with the edit.
Surface withMapEdits(const Surface& s, const Edits& e);
std::string lowerRoughnessMap(const Surface& s, const Edits& e);

// One map of a surface as lower() uses it: the Clothes tab's texture list.
struct MapSlot
{
  std::string slot;   // base | opacity | normal | bump | roughness | translucency | emission | detail | makeup
  std::string label;  // "Base colour", "Cutout", ...
  std::string rel;    // the library-relative image (a layered map's bottom layer); "" when an edit took it away
  bool        edited = false;  // the image is an edit's (replaced or taken away)
};
// The surface's maps with the edits applied, in that order: a slot with no map and no edit is left out.
std::vector<MapSlot> mapSlots(const Surface& s, const Edits& e);

// Where a surface's own maps sample (every slot but a detail layer's maps): the edits' transform
// laid after the authored tiling, uv' = T_e + R_e · S_e · (S_a · uv + T_a).
UvTransform surfaceUv(const Surface& s, const Edits& e);
// The same transform for UVs whose V is turned over (v' = 1 − v) on both sides, with the images as
// their files have them (LowerOptions::flipV): the picture lands where it did.
UvTransform mirroredV(const UvTransform& t);
// `t` followed by v' = 1 − v: where an image whose rows run the other way than the UVs expect holds
// what `t` samples
UvTransform rowsFlipped(const UvTransform& t);

// What resolving one request gave.
struct ResolvedTexture
{
  int  index     = -1;  // texture index, -1: none, the slot is dropped
  bool flipGreen = false;  // the image holds a normal map whose green the renderer should negate
  // the image's rows run the other way than the UVs expect (a source image in a figure built for the
  // cooked, bottom-up ones): its slot samples it through rowsFlipped
  bool flipRows = false;
};

// The material with its slots' indices from `resolve`, each slot it can't make dropped.
Material resolved(const Lowered& lowered, const std::function<ResolvedTexture(const TextureRequest&)>& resolve);

// The same as a glTF material. The extensions it uses are appended to `extensionsUsed` once.
tinygltf::Material toGltf(const Lowered& lowered, const std::function<ResolvedTexture(const TextureRequest&)>& resolve,
                          std::vector<std::string>& extensionsUsed);

// The brightness an emitter has in the renderer: its luminance against the scene's exposure
// (Iray's tone mapper: exposure value and film ISO). 1 for a plain glow. `areaM2` is the emitting
// surface's area for a flux (lm, W).
float emissionStrength(const Surface& s, double areaM2, double exposureValue = 13, double iso = 100);

// Surfaces that emit a flux over their area need the area to be known.
bool needsArea(const Surface& s);

}  // namespace fv::mat
