#pragma once

// A SceneGraph as the renderer's glTF model: nodes with their transforms, one mesh per scene mesh
// (one primitive per material group, sharing the vertex attributes), skins with the inverse bind
// matrices three's SkinnedMesh.bind gave them, and materials. Built in memory and handed to the
// renderer (nvvkgltf::Scene::takeModel), not written to a file.
//
// UVs go out as DSON has them (v up). Images are referenced as their row order needs: the cooked
// textures are flipped by the cooker (flip=1, rows bottom-up as the RTX export always had them);
// a source image read as it is would show upside down, so without the cooker the UVs flip instead.

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tinygltf/tiny_gltf.h>

#include <figure/materials/lower.hpp>
#include <figure/scene/graph.hpp>

namespace fv::build {

using mat::TextureRequest;
using mat::requestKey;

struct TextureImage
{
  std::string uri;       // absolute path
  std::string mimeType;  // "image/ktx2" for a cooked file (KHR_texture_basisu), else the source's
  // the image's rows run bottom-up (flipped for the DSON UVs); the build flips the UVs otherwise
  bool flipped = false;
  // a normal map as the source holds it, green not inverted: the material says so (FV_texture_flip_green)
  bool flipGreen = false;
};

struct BuildOptions
{
  // The image for a texture, or none to leave the slot empty (a flat colour shows).
  std::function<std::optional<TextureImage>(const TextureRequest&)> texture;
  // Whether textures arrive flipped (decides the UVs; every texture must agree).
  bool texturesFlipped = false;
  // How skin is sent to the path tracer.
  mat::SkinMode skin = mat::SkinMode::eVolume;
  // A file for other tools (see mat::LowerOptions::forFile).
  bool forFile = false;
  // The scene's camera exposure, which sets how bright Iray emitters are.
  double exposureValue = 13, iso = 100;
  // Per scene material, the ones light under the skin passes through (walkThroughMaterials): their
  // glTF materials say `"fvWalkThrough": true` in their extras. Empty: none.
  std::vector<bool> walkThrough;
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
  // scene material -> the options its glTF material was lowered with (emission from the exposure
  // and the emitting area, the hair strand direction measured from the meshes, an iris's discs),
  // for later edits
  std::vector<mat::LowerOptions> lowering;
  // request key -> the glTF texture made for it, so a later edit reuses what the build already has
  std::map<std::string, mat::ResolvedTexture> textureOf;
};

// A mesh's drawn triangles: all of them, or without the polygons a worn item hides.
class DrawnView
{
public:
  explicit DrawnView(const mesh::BuiltMesh& g)
      : m_geo(g)
  {
    if(!g.hiddenPolys.empty())
      m_own = mesh::drawnTriangles(g);
  }
  const std::vector<uint32_t>&    index() const { return m_geo.hiddenPolys.empty() ? m_geo.fullIndex : m_own.index; }
  const std::vector<mesh::Group>& groups() const { return m_geo.hiddenPolys.empty() ? m_geo.groups : m_own.groups; }

private:
  const mesh::BuiltMesh& m_geo;
  mesh::Drawn            m_own;
};

// What a build reads from the meshes before it lowers a material, and the figure feed with it.
struct Drawing
{
  // every node reachable from the root, depth first as traversal meets them (taken-off subtrees left out)
  std::vector<int> order;
  // per scene mesh: its drawn triangles, null without geometry, and its position bounds (min x y z, max x y z)
  std::vector<std::unique_ptr<DrawnView>> drawn;
  std::vector<std::array<double, 6>>      bounds;
};
Drawing drawing(const SceneGraph& scene, int root);

// Per scene material, the options it is lowered with: requestOptions, the UV flip, ribbon
// double-sidedness, the emission its emitting area gives it and the hair strand direction its cards
// vote for. The one place the build and the feed take them from.
std::vector<mat::LowerOptions> lowering(const SceneGraph& scene, const Drawing& drawing, const BuildOptions& options);

// The figure's one sampler (repeat, trilinear).
tinygltf::Sampler figureSampler();

// Scene material `material` as the build emits it: materialToGltf, with its `m:<i>` id and the
// walk-through flag in its extras.
tinygltf::Material figureMaterial(const SceneGraph& scene, int material, const mat::LowerOptions& lowering,
                                  const std::function<mat::ResolvedTexture(const TextureRequest&)>& resolve,
                                  std::vector<std::string>& extensionsUsed, const BuildOptions& options);

// A skinned mesh's inverse bind matrices as glTF stores them (16 floats per bone, column-major):
// GLTFExporter's, the bone inverse times the bind matrix.
std::vector<float> inverseBindMatrices(const Mesh& m);

// Every node reachable from `root`, in the order traversal meets them.
BuildResult buildGltf(const SceneGraph& scene, int root, const BuildOptions& options);

// Measures the strand direction of the scene's hair cards ahead of a build, in parallel, and keeps
// it with each geometry: the build, which a host runs on its application thread after a wardrobe
// change, then finds it measured instead of welding a strand hair's cards (about a second) itself.
// `texturesFlipped` as the build will have it. A build measures what it doesn't find.
void measureStrands(const SceneGraph& scene, bool texturesFlipped);

// The textures buildGltf asks options.texture for, each once: a host resolves them ahead, in
// parallel (cooking one can take seconds), and answers the build from what it found.
std::vector<TextureRequest> textureRequests(const SceneGraph& scene, mat::SkinMode skin = mat::SkinMode::eVolume);

// A scene material as a glTF material, textures resolved by `resolve`. A host edits a material
// that is already in the model with it: `options` are the ones the build used (BuildResult::lowering).
tinygltf::Material materialToGltf(const Material& m, const mat::LowerOptions& options,
                                  const std::function<mat::ResolvedTexture(const TextureRequest&)>& resolve,
                                  std::vector<std::string>& extensionsUsed);

// The lowering options of scene material `material` that decide which textures it asks for, as
// both textureRequests and buildGltf set them: a texture resolved ahead is then the one the build
// looks up. `tearFilm`: the scene's eyes are under a tear film (computed once per scene).
mat::LowerOptions requestOptions(const SceneGraph& scene, int material, mat::SkinMode skin, bool tearFilm);

// Where a material's UV islands lie in its maps, one disc per eye: centre x, y and radius as
// fractions of the image, y down, in the tile the island is in; islands that overlap are one, the
// four largest are kept. For an iris surface of its own (Genesis 8 "Irises"), whose map it shares
// with nothing it can be told apart from without them (LowerOptions::irisDiscs).
std::vector<std::array<float, 3>> irisDiscs(const SceneGraph& scene, int material);

}  // namespace fv::build
