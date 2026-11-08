#pragma once

// A Skyrim Special Edition outfit piece (.nif) as a wearable load, for Figure::prepareWear: one item
// of every shape the file draws, skinned to the bones those use (kept with their Skyrim names, which
// wardrobe::parseBone reads), and each shape's material as Iray Uber channels over its textures. The
// body the outfit was built on (the CBBE or UNP shape Outfit Studio keeps in the file, or a sibling
// file's when this one has none, as boots don't) is the fit's base (LoadResult::sourceBase), so the
// shape correction measures the outfit against the body it was made for, at whatever preset that was.
//
// Positions go from Skyrim's units and axes (z up, facing +y, its left at -x) to DAZ's centimetres
// (y up, facing +z, its left at +x), scaled as a whole so the skeleton stands as tall as the figure's;
// the bone transfer then fits each limb. UVs flip to DAZ's (v up), and the normal maps are marked
// DirectX-style (green down), which they are: their curl is a fifth lower read that way.
//
// Textures sit beside the mesh under its Data folder (textures\...), outside the content library.
// They are linked into the library (hard links where the drive allows, else copies) under
// Runtime/Textures/FigureViewer/Skyrim/<the Data folder's hash>/, as images.rs does with a picked
// picture, so the cooker, the renderer's own lookup and saved scenes all find them by library path.

#include <filesystem>
#include <string>

#include <figure/dson/figure_loader.hpp>
#include <figure/nif/nif_file.hpp>

namespace fv::nif {

// A Skyrim unit in centimetres: 64 to the yard.
constexpr double kUnitCm = 2.54 * 36 / 64;

// Skyrim's axes to DAZ's, in centimetres.
std::array<double, 3> toDaz(const V3& p);

bool isNif(const std::string& path);

struct WearOptions
{
  // the content library folder textures are linked into; empty leaves them out
  std::filesystem::path textureHome;
};

// Throws std::runtime_error with the reason when nothing can be worn.
dson::LoadResult loadWearable(const std::filesystem::path& file, const dson::LoadOptions& options, const WearOptions& wear);

// The shapes that are the body an outfit was built on: skin-tinted, or carrying a body's morph file
// ("BODYTRI"), the largest first.
std::vector<size_t> bodyShapes(const File& nif);

// Where a texture a NIF names (`textures\x\y.dds`, or without its `textures\`) is on disk: under the
// first folder up from `meshDir` that holds it. Empty when none does.
std::filesystem::path findTexture(const std::filesystem::path& meshDir, const std::string& rel);

// The library path a texture is linked to under `home` (the content folder), linking it there first when
// it isn't, or is another file. Empty when it can't be (no home, or the drive refuses both a link and a copy).
std::string libraryTexture(const std::filesystem::path& home, const std::filesystem::path& dataRoot, const std::filesystem::path& source,
                           const std::string& rel, std::string* problem = nullptr);

}  // namespace fv::nif
