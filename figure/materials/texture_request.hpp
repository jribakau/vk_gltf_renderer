#pragma once

// One texture a glTF material asks for: a library image, how it is read, and what the cooker (the
// Rust texture server) lays over or converts in it. The cooker's own queries are the vocabulary:
// `alpha=`, `mk=` (makeup), `bump=` and `ch=` (see src-tauri/src/protocol.rs).

#include <array>
#include <optional>
#include <string>

namespace fv::mat {

// PBRSkin makeup blended over a base colour in linear light: base x `factor` towards the makeup
// colour map x `tint` by `weightFactor` x the weight map's red. The base colour is the cooked image
// itself when `baseIsSource`, else the flat colour `factor` (a material with no base map).
struct MakeupRequest
{
  bool                 baseIsSource = true;
  std::array<float, 3> factor{1, 1, 1};
  std::string          colorRel;  // empty: the flat tint
  std::array<float, 3> tint{1, 1, 1};
  std::string          weightRel;  // empty: none
  float                weightFactor = 0;
};

struct TextureRequest
{
  std::string rel;
  bool        srgb   = true;
  bool        normal = false;  // a normal map: compressed more carefully
  std::string alphaRel;        // a cutout map folded into alpha; empty: none
  // "ng": a normal map with green inverted (glTF's convention for meshes without tangents);
  // "bn": a normal map derived from a height map; "mr": glTF metallic-roughness from a roughness map
  std::string channels;
  std::string bumpRel;  // a height map blended onto the normal map; empty: none
  float       bumpStrength = 1;
  std::optional<MakeupRequest> makeup;

  // Whether only the cooker can make it (the renderer reads plain png/jpg/webp itself).
  bool composed() const { return !alphaRel.empty() || !channels.empty() || !bumpRel.empty() || makeup.has_value(); }
};

// Two requests with the same key cook the same image.
std::string requestKey(const TextureRequest& req);

}  // namespace fv::mat
