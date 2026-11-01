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

// Skin's micro-relief, for skin with no detail map of its own: the fine creases and pores that a
// character's normal map, at about a tenth of a millimetre a texel, barely holds (Leyla 9's has an RMS
// slope of about 0.065 across a face). Under a random walk they melt out of the diffuse light, as in
// real skin, so what they do is break up the highlights. Like the fabrics' images, the app's
// (src-tauri/src/fabrics.rs), written into the assets library at startup.
constexpr std::string_view kSkinMicroMap = "/Runtime/Textures/FigureViewer/Skin/v1/micro-height.png";
// One texture tile on the skin, in metres: 1024 texels of 25 microns, about 46 creased cells across.
constexpr float kSkinMicroTileMetres = 0.0256f;
// The height map's full range as a fraction of the tile, which makes the per-axis RMS slope about 0.07
// at strength 1 (a test in fabrics.rs holds the image to this number). Measured skin's whole slope is
// about 0.2 to 0.3 per axis (Weyrich 2006, Beckmann m of 0.3 to 0.45), most of it finer than this and
// left to the lobes' roughness.
constexpr float kSkinMicroRelief = 0.00146f;
// What the relief adds to a lobe's slopes (2 x 0.07^2 to its alpha squared) it takes back from its
// roughness at strength 1, so that a highlight too small to resolve the relief keeps about its width.
// One multiplier for all the lobes: the broad ones stay within about 5% (alpha 0.25 to 0.4), and the
// tightest (alpha 0.1), with less to give back, spread by about a third.
constexpr float kSkinMicroRoughness = 0.96f;

// The layer lower() gives flesh under a walk (SkinMode::eVolume, eMeasured) that one UV unit spans
// `metresPerUv` of, at the edits' strength. None for an unknown size, at strength 0, on a surface with a
// detail map of its own or a weight or roughness map that would be lost, and under a top coat: a lip
// gloss lies smooth over the relief, and the one normal would ripple its reflection.
std::optional<DetailLayer> skinMicroDetail(const Surface& s, const Edits& e, float metresPerUv);

}  // namespace fv::mat
