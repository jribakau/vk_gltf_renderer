#pragma once

// A SceneGraph as the renderer's glTF model: nodes with their transforms, one mesh per scene mesh
// (one primitive per material group, sharing the vertex attributes), skins with the inverse bind
// matrices three's SkinnedMesh.bind gave them, and materials. Built in memory and handed to the
// renderer (nvvkgltf::Scene::takeModel), not written to a file.
//
// UVs go out as DSON has them (v up). Images are referenced as their row order needs: the cooked
// textures are flipped by the cooker (flip=1, rows bottom-up as the RTX export always had them);
// a source image read as it is would show upside down, so without the cooker the UVs flip instead.

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <tinygltf/tiny_gltf.h>

#include <figure/scene/graph.hpp>

namespace fv::build {

// One texture a material uses: a library path, how it is read, and a cutout map to fold into alpha.
struct TextureRequest
{
  std::string rel;
  bool        srgb   = true;
  bool        normal = false;
  std::string alphaRel;  // empty: none
};

struct TextureImage
{
  std::string uri;       // absolute path
  std::string mimeType;  // "image/ktx2" for a cooked file (KHR_texture_basisu), else the source's
  // the image's rows run bottom-up (flipped for the DSON UVs); the build flips the UVs otherwise
  bool flipped = false;
};

struct BuildOptions
{
  // The image for a texture, or none to leave the slot empty (a flat colour shows).
  std::function<std::optional<TextureImage>(const TextureRequest&)> texture;
  // Whether textures arrive flipped (decides the UVs; every texture must agree).
  bool texturesFlipped = false;
};

struct BuildResult
{
  tinygltf::Model model;
  // scene node -> glTF node
  std::vector<int> nodeOf;
  // scene mesh -> glTF mesh
  std::vector<int> meshOf;
  // scene mesh -> glTF skin, -1 for an unskinned mesh
  std::vector<int> skinOf;
};

// A skinned mesh's inverse bind matrices as glTF stores them (16 floats per bone, column-major):
// GLTFExporter's, the bone inverse times the bind matrix.
std::vector<float> inverseBindMatrices(const Mesh& m);

// Every node reachable from `root`, in the order traversal meets them.
BuildResult buildGltf(const SceneGraph& scene, int root, const BuildOptions& options);

// The textures buildGltf asks options.texture for, each once: a host resolves them ahead, in
// parallel (cooking one can take seconds), and answers the build from what it found.
std::vector<TextureRequest> textureRequests(const SceneGraph& scene);
std::string                 requestKey(const TextureRequest& req);

}  // namespace fv::build
