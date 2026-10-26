#pragma once

// The built-in fabrics a garment's surface can wear as a detail layer (Edits::fabric): a tiled
// height map bending the normal and a roughness multiplier, the weave at its physical size on the
// surface. The images are the app's (it writes them into the assets library at startup, under
// kFabricDir); this table says what each one measures.

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <figure/materials/surface.hpp>

namespace fv::mat {

struct Fabric
{
  const char* key;  // Edits::fabric, and the images' names
  // The size of one texture tile on the surface, in metres. A tile holds several repeats of the
  // weave, so its mips stay clean.
  float tileMetres;
  // The height map's full range as a fraction of one tile (TextureRequest::bumpHeight): the relief
  // grows with the weave when it is scaled, so a scale change cooks nothing.
  float relief;
  // DetailLayer::roughness: the roughness map's mean is about 0.5, so this is about twice the mean
  // multiplier on the surface's roughness
  float roughness;
};

// Every built-in fabric.
std::span<const Fabric> fabrics();
// The fabric of that key, none when there is no such fabric.
const Fabric* findFabric(std::string_view key);

// Where the fabric images are, library-relative.
constexpr std::string_view kFabricDir = "/Runtime/Textures/FigureViewer/Fabrics/v1/";
std::string fabricHeightMap(const Fabric& f);
std::string fabricRoughnessMap(const Fabric& f);

// The detail layer the edits' fabric gives a surface that one UV unit spans `metresPerUv` of (0:
// unknown, a fixed tiling instead); none when the edits name no fabric this table has.
std::optional<DetailLayer> fabricDetail(const Edits& e, float metresPerUv);

}  // namespace fv::mat
