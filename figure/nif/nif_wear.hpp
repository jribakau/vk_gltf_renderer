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
// Textures sit beside the mesh under its Data folder (textures\...). A mod the app installed is in the
// content library already (Skyrim/<mod>/, src-tauri/src/skyrim.rs), and its textures are used where
// they are; a texture its own mod lacks is looked for in the other mods installed there, as the game's
// one Data folder would have it. A mesh worn from outside the library has its textures linked into it
// (hard links where the drive allows, else copies) under Runtime/Textures/FigureViewer/Skyrim/<the
// Data folder's hash>/, as images.rs does with a picked picture, so the cooker, the renderer's own
// lookup and saved scenes all find them by library path.

#include <filesystem>
#include <optional>
#include <string>

#include <figure/dson/figure_loader.hpp>
#include <figure/nif/nif_file.hpp>

namespace fv::nif {

// The content folder's folder of installed Skyrim mods, one folder each (src-tauri/src/skyrim.rs ROOT).
inline constexpr const char* kInstalledMods = "Skyrim";

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

// The body an outfit was built on (its own, or a sibling file's, as loadWearable finds it), at the
// game's size in DAZ's centimetres and axes, skinned to its Skyrim bones: what a body map reads (step 3b).
// Null when neither keeps one; throws std::runtime_error when the file doesn't read.
std::shared_ptr<const wardrobe::BaseCage> carriedBody(const std::filesystem::path& file);

// A body mod's body (femalebody_1.nif with the femalehands_1.nif and femalefeet_1.nif beside it): their
// body shapes as one cage, welded where they meet at the wrists and ankles, skinned to their bones by
// name, its surfaces "Body", "Hands" and "Feet"; at the game's size, as carriedBody. Null when the body
// file keeps no body; a hands or feet file missing is left out.
std::shared_ptr<const wardrobe::BaseCage> modBody(const std::filesystem::path& bodyFile);

// Whose body a piece is for, "female" or "male", as the game's own files tell it: a folder `f` or `m`
// (`female`, `male`) above it; "female" or "male" in its name; a name ending in f or m beside its twin
// of the other ending (torsof_1, torsom_1); a name ending in f beside the same name without it, or a
// name beside the same with an f, the man's being the one the game leaves unmarked (goldamuletf_1,
// goldamulet_1). Empty when none of that says (a helmet both wear, a mod's own naming).
std::string genderOf(const std::filesystem::path& file);

// The whole body a piece was made on, when one is on this machine (docs/wardrobe-plan.md, step 3b: "the
// source is a whole body where one is on the user's machine"): the game's own, or a body mod's.
struct WholeBody
{
  // the body file's body with its hands and feet (modBody), at the game's size; one object for every
  // piece made on that file, so what is made of it (its map) is made once
  std::shared_ptr<const wardrobe::BaseCage> body;
  // its vertices (xyz) at the piece's own preset, where the body the piece carries lies off the file's
  // (bodymap::atPreset); empty where it lies on it, or the piece carries none
  std::vector<float> preset;
  std::string           gender;  // "female" or "male": which body file it is
  std::filesystem::path file;
};
// Looked for under `meshes\actors\character\character assets` (femalebody_1.nif, malebody_1.nif, of the
// piece's own weight where there is one) in the piece's own Data folder, then in the other mods
// installed in `home` (the content folder) under Skyrim/. A piece that carries a body (`carried`, at the
// game's size) takes the body its own is cut from: the one its vertices match by UV and lie closest on,
// nine in ten matched and a median within 1.2 cm, which a preset of that body is and another body isn't.
// A piece that carries none takes its own Data folder's body of its gender, where genderOf says which:
// the game's own boots are made on the game's own feet, and a mod's boots on nobody knows what.
std::optional<WholeBody> wholeBody(const std::filesystem::path& file, const wardrobe::BaseCage* carried, const std::filesystem::path& home);

// Where a texture a NIF names (`textures\x\y.dds`, or without its `textures\`) is on disk: under the
// first folder up from `meshDir` that holds it. Empty when none does.
std::filesystem::path findTexture(const std::filesystem::path& meshDir, const std::string& rel);

// A texture a NIF names, in the mods installed in `home` (the content folder) under Skyrim/, the first
// by folder name that has it. Empty when none does.
std::filesystem::path findInstalledTexture(const std::filesystem::path& home, const std::string& rel);

// The library path a texture is linked to under `home` (the content folder), linking it there first when
// it isn't, or is another file. Empty when it can't be (no home, or the drive refuses both a link and a copy).
std::string libraryTexture(const std::filesystem::path& home, const std::filesystem::path& dataRoot, const std::filesystem::path& source,
                           const std::string& rel, std::string* problem = nullptr);

}  // namespace fv::nif
