#pragma once

// The scene as a GLB for other tools: the glTF the renderer shows, with the materials lowered for a
// file (plain PBR skin, MASK cutouts, no renderer extensions) and every texture composed here as a
// PNG embedded in the file, since the cooked textures are BC7 KTX2 that few tools read. The
// composing follows the texture cooker's rules (src-tauri/src/texcache.rs): cutouts folded into
// alpha, makeup baked into the base colour, bump maps blended into the normal map, normal maps
// with green inverted, roughness packed as metallic-roughness.

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <figure/gltf/gltf_build.hpp>

namespace fv::build {

struct ExportOptions
{
  // Finds a library image by its (decoded) path.
  std::function<std::optional<std::filesystem::path>(const std::string& rel)> find;
  // rows bottom-up, as the cooked images are; a figure built on the source images wants them as they are
  bool bottomUp = true;
};

// PNG bytes of the image a request asks for, rows bottom-up unless the options say otherwise (as
// the cooked ones are, for the DSON UVs); none when a source is missing or a format cannot be
// read. `why` says what.
std::optional<std::vector<uint8_t>> composeImage(const TextureRequest& request, const ExportOptions& options, std::string* why = nullptr);

// The export in two parts, so the slow one can run off the thread that owns the scene: the glTF is
// built from the scene (fast, with its images still to come), then its textures are composed and
// the file is written from that copy alone.
struct GlbExport
{
  tinygltf::Model             model;
  std::vector<TextureRequest> requests;
};
GlbExport prepareGlb(const SceneGraph& scene, int root);
// Returns what could not be made (a missing or unreadable image is replaced by a neutral one).
std::vector<std::string> finishGlb(GlbExport& job, const std::filesystem::path& out, const ExportOptions& options);

}  // namespace fv::build
