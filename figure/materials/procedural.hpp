#pragma once

// Procedural layers over a figure's surfaces: sweat now, dirt and the like after it. A layer is one
// row of a table, which says two things:
//
//  - where it lands: a field over the body in its rest pose, from the skeleton's landmarks, written
//    per drawn vertex into one channel of the mesh's masks. It rides with the skin through any pose,
//    has no UV seams, and a surface's maps stay as they are.
//  - what it does there: a response the shader lays over the surface (Material::procedural), growing
//    from where the mask is highest as the amount (an edit, 0..1) rises, with a tile for what's finer
//    than the mesh.
//
// The images are the app's, like the fabrics' (src-tauri/src/fabrics.rs).

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <figure/materials/surface.hpp>
#include <figure/math/xform.hpp>

namespace fv::mat {

// Sweat's tile: 1024 texels of 50 microns, its largest beads 2.2 mm across and 0.64 mm high (a cap
// meeting the skin at 60 degrees). The height map's full range is that as a fraction of the tile, which
// the images are made to (a test in fabrics.rs holds them to their size).
constexpr float kSweatTileMetres = 0.0512f;
constexpr float kSweatRelief     = 0.0124f;
constexpr char  kSweatHeightMap[] = "/Runtime/Textures/FigureViewer/Sweat/v3/drops-height.png";
constexpr char  kSweatCoverMap[]  = "/Runtime/Textures/FigureViewer/Sweat/v3/drops-cover.png";

// A place on the body: a bone's rest position, by anatomical slot ("head", "upperarm", ...; the
// sided ones under "l:" and "r:"), and the figure's height, which offsets and sizes are fractions of.
struct BodyLandmarks
{
  std::map<std::string, Vec3> at;
  double                      height = 0;  // 0: no skeleton to go by
};

// The landmarks of bones (name, rest position in the mesh's world) as Genesis 2 to 9 name them, the
// first of a slot staying. Y is up. Without eyes and feet to measure the height from, none.
BodyLandmarks bodyLandmarks(std::span<const std::pair<std::string, Vec3>> bones);

// A soft patch of a layer's field, around a landmark, in fractions of the figure's height. On a sided
// slot it lies on both sides, x mirrored.
struct FieldPatch
{
  const char*           slot;
  std::array<double, 3> offset;
  std::array<double, 3> radius;
  float                 weight;
};

// How a field is made of the patches: together, over `floor` everywhere, broken up by noise of about
// `blotch` (fraction of the figure's height) by up to `blotchDepth` (0: none).
struct FieldShape
{
  float floor, blotch, blotchDepth;
};

struct ProceduralDef
{
  const char*             key;
  int                     channel, dropChannel;  // of the mesh's masks: the film's, the beads'
  std::optional<float> Edits::*amount;
  // where the film and the beads land, from the same patches; with no skeleton to go by, `unknownBody`
  // everywhere
  std::span<const FieldPatch> patches;
  FieldShape                  film, drops;
  float                       unknownBody;
  // what it does there (mat::Procedural)
  Rgb   tint;
  float filmRoughness, coat, dropRoughness;
  // its tile: one image's size on the surface in metres, the height map's full range as a fraction of
  // that, and the images
  float       tileMetres, relief;
  const char* heightMap;
  const char* coverMap;
};

// Every layer.
std::span<const ProceduralDef> proceduralDefs();

// Whether a surface can take the layer: a figure's skin, for sweat.
bool takesProcedural(const ProceduralDef& def, const Surface& s);

// The layer a surface wears with these edits: the first with an amount above 0 that it takes (a row
// of the shader's material holds one). None: no layer.
const ProceduralDef* proceduralOn(const Surface& s, const Edits& e);

// The layer's film (or beads) field at a point (the mesh's world), 0..1.
float proceduralField(const ProceduralDef& def, const FieldShape& shape, const BodyLandmarks& body, const Vec3& point);

// Per vertex of `position` (x, y, z, in the mesh's own space, which `toWorld` takes to the landmarks'),
// every layer's film and beads fields in their channels, 8 bits each, red the lowest: MeshRecord::masks.
std::vector<uint32_t> proceduralMasks(std::span<const float> position, const Mat4& toWorld, const BodyLandmarks& body);

}  // namespace fv::mat
