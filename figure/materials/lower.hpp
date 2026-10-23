#pragma once

// A Surface as the glTF material the path tracer (and the rasterizer, which reads the same model)
// evaluates: Iray's layers folded into metallic-roughness plus the KHR_materials_* and FV_*
// extensions the renderer supports. Built from the Surface and the panel's edits alone. A port of
// toGltf in src/lib/rtx/gltf.ts.
//
// The lowering is pure: textures come out as requests (what to cook from which image) that a host
// resolves into glTF texture indices, so a live edit that changes none of them can leave the
// textures alone.

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <tinygltf/json.hpp>
#include <tinygltf/tiny_gltf.h>

#include <figure/materials/surface.hpp>
#include <figure/materials/texture_request.hpp>

namespace fv::mat {

// How skin is sent to the renderer. Volume is the path tracer's converged look; thin is its preview
// while the camera moves. The rasterizer would show transmission as see-through skin, so a
// real-time view takes none of it: plain diffuse and specular under the base colour.
enum class SkinMode
{
  eRaster,
  eThin,
  eVolume,
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
};

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
};

// A glTF material as JSON with its texture infos left as slots: {"slot": i, "texCoord": 0, ...},
// `i` indexing `requests`.
struct Lowered
{
  nlohmann::json              material;
  std::vector<TextureRequest> requests;
  Effective                   effective;
};

Lowered lower(const Surface& s, const Edits& edits, const LowerOptions& options);

// What resolving one request gave.
struct ResolvedTexture
{
  int  index     = -1;  // glTF texture index, -1: none, the slot is dropped
  bool flipGreen = false;  // the image holds a normal map whose green the renderer should negate
};

// The material with its slots replaced by texture infos from `resolve`. The extensions it uses are
// appended to `extensionsUsed` once.
tinygltf::Material toGltf(const Lowered& lowered, const std::function<ResolvedTexture(const TextureRequest&)>& resolve,
                          std::vector<std::string>& extensionsUsed);

// The brightness an emitter has in the renderer: its luminance against the scene's exposure
// (Iray's tone mapper: exposure value and film ISO). 1 for a plain glow. `areaM2` is the emitting
// surface's area for a flux (lm, W).
float emissionStrength(const Surface& s, double areaM2, double exposureValue = 13, double iso = 100);

// Surfaces that emit a flux over their area need the area to be known.
bool needsArea(const Surface& s);

// tinygltf::Value from JSON.
tinygltf::Value toValue(const nlohmann::json& j);

}  // namespace fv::mat
