#pragma once

// One texture a glTF material asks for: a library image, how it is read, and what the cooker (the
// Rust texture server) lays over or converts in it. The cooker's own queries are the vocabulary:
// `alpha=`, `mk=` (makeup), `ir=` (iris), `rc=` (recolour), `bump=`, `metal=`, `mul=` and `ch=`
// (see src-tauri/src/protocol.rs).

#include <array>
#include <optional>
#include <string>
#include <vector>

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

// An eye's colour map with its iris recoloured, in Oklab, as the cooker's recolour_iris does it
// (texcache.rs; figure/materials/iris.hpp for a GLB). In each disc the pupil (fading out from
// lightness 0.35 to 0.2) is left out of the iris's mean; a pixel's lightness is scaled by the
// target's over that mean (clamped to 0.5..2), and its hue and chroma become the target's plus half
// its own departure from the mean, scaled by its lightness over the mean, so fibres and the limbal
// ring stay. Blended in by `strength`, in linear light; outside the irises the image is multiplied
// by `sclera`.
struct IrisRequest
{
  enum class Mode
  {
    eEye,   // iris and white in one image: each disc is a nominal iris the cooker fits the limbus around,
            // its edge feathered from 0.9 to 1.1 of the radius
    eIris,  // the image is iris: all of it changes, the discs only say where its mean is taken
  };
  std::optional<std::array<float, 3>> iris;  // linear; none: the iris as it is
  float                               strength = 1;
  Mode                                mode     = Mode::eEye;
  // centre x, y and radius, fractions of the image as its file has it (y down, r of the width)
  std::vector<std::array<float, 3>>   discs;
  std::optional<std::array<float, 3>> sclera;  // linear multiplier; none: the white as it is
};

// The cooker's ir= value, before it is percent-encoded as a whole:
// <iris r,g,b | ->|<strength>|<eye | iris>|<x,y,r>[;<x,y,r>...]|<sclera r,g,b | ->
std::string irisQuery(const IrisRequest& r);

// A whole base colour image moved to another colour, in Oklab, as the cooker's recolour does it
// (texcache.rs), after makeup (a hair's translucency colour) is laid over it. The mean is taken over
// the image weighted by its alpha (a cutout's gaps count for nothing); a pixel's lightness becomes the
// target's plus its departure from the mean, scaled by the target's over the mean (clamped to
// 0.5..1.5), and its hue and chroma the target's plus half its own departure, scaled by its
// lightness over the target's, so strands, roots and highlights stay. Blended in by `strength`, in
// linear light; alpha is kept.
//
// With a `reference` (a skin tone) the mean is that image's instead, and a pixel keeps all of its
// own hue and chroma departure: every image toned against the same reference moves by the same
// shift, so the seams between a figure's skin maps stay matched and lips stay redder than cheeks.
struct RecolorRequest
{
  std::array<float, 3> color{1, 1, 1};  // linear
  float                strength = 1;
  std::string          reference;  // library image; empty: the image's own mean
};

// The cooker's rc= value, before it is percent-encoded as a whole: <r,g,b>|<strength>, then
// |<reference source> when there is one (the host adds it: a source is the host's to encode).
std::string recolorQuery(const RecolorRequest& r);

struct TextureRequest
{
  std::string rel;
  bool        srgb   = true;
  bool        normal = false;  // a normal map: compressed more carefully
  std::string alphaRel;        // a cutout map folded into alpha; empty: none
  bool        alphaFromAlpha = false;  // the cutout is that map's alpha channel, not its grey (green)
  // "ng": a normal map with green inverted (glTF's convention for meshes without tangents);
  // "bn": a normal map derived from a height map; "mr": glTF metallic-roughness from a roughness map;
  // "mt": glTF metallic-roughness from a metal map alone (its grey in blue, the rest white);
  // "wa": a weight map's grey (the mean of R, G and B, DAZ's grayscale_mode "average") in all four
  // channels, so specularTexture's alpha and clearcoatTexture's red read the same weight
  std::string channels;
  std::string metalRel;  // "mr": a metal map whose grey goes in blue; empty: full metal there
  // Maps whose grey multiplies the colour channels, after the layers and before anything else: a
  // lobe's weight map times its reflectivity map. Plain images only (not layered).
  std::vector<std::string> mulRels;
  std::string bumpRel;  // a height map blended onto the normal map; empty: none
  float       bumpStrength = 1;
  // A height map's full range in UV units (bumpHeightInUv), which the bake turns into slopes against
  // the map's size: the same bump at any resolution. 0: a fixed slope per texel.
  float bumpHeight = 0;
  std::optional<MakeupRequest> makeup;
  std::optional<IrisRequest>   iris;
  std::optional<RecolorRequest> recolor;
  // A DAZ layered image's upper layers, laid over `rel` before anything else (Genesis 9's default
  // eyes: the iris over the sclera), each as layerText writes it
  std::vector<std::string> layers;

  // Whether only the cooker can make it (the renderer reads plain png/jpg/webp itself).
  bool composed() const
  {
    return !alphaRel.empty() || !channels.empty() || !metalRel.empty() || !bumpRel.empty() || makeup.has_value() || iris.has_value() || recolor.has_value()
           || !layers.empty() || !mulRels.empty();
  }
};

// A layered image's map: its layers bottom up, joined by this (no path holds it). A map that starts
// with it has no bottom of its own and goes over the map its channel has (dson::collectImages).
constexpr char kLayerSeparator = '|';
// The layers of a map, bottom up, each as layerText writes it: one for a plain image.
std::vector<std::string> mapLayers(const std::string& map);

// One upper layer of a layered image. Each pixel moves from what is under it toward the blend of the
// two, by the layer's alpha x `opacity` x the mask's grey. Blended in the image's own values, as an
// 8-bit editor does. texcache.rs cook and export_glb.cpp composeImage both lay layers this way.
struct MapLayer
{
  // empty: a flat `color`
  std::string file;
  // "" (over), "multiply", "overlay", "screen", "hard_light" or "color_burn"
  std::string blend;
  float       opacity = 1;
  // empty: none. Past its edge nothing of the layer shows
  std::string mask;
  std::array<float, 3> color{1.f, 1.f, 1.f};
  // The width of the image it was made for, when it may go over a map of another size: it is scaled
  // by that share. 0: at its own size from the top left (Genesis 9's iris is the top half of the
  // sclera's square).
  int canvas = 0;
};
// A layer as a map holds it: its file alone, for a layer with nothing else to say.
std::string layerText(const MapLayer& layer);
MapLayer    parseLayer(const std::string& text);
// The file a layer of a map names ("" for a flat colour).
std::string layerFile(const std::string& text);

// Two requests with the same key cook the same image.
std::string requestKey(const TextureRequest& req);

}  // namespace fv::mat
