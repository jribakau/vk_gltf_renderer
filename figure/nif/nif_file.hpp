#pragma once

// Reading a Skyrim Special Edition mesh (.nif: Gamebryo 20.2.0.7, Bethesda stream 100), as far as
// wearing an outfit needs: the node tree, and each BSTriShape's vertices, triangles, skin weights
// and material. The layouts are niftools' nif.xml. A rotation is stored row by row and turns column
// vectors (v' = R v): read that way, a bone's place from the node tree is the one the skin's bind
// data gives it. Older Skyrim (stream 83, NiTriShape) and Fallout 4 files are refused with the reason.

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fv::nif {

using V3 = std::array<float, 3>;

// p' = s R p + t, R row-major.
struct Transform
{
  std::array<float, 9> r{1, 0, 0, 0, 1, 0, 0, 0, 1};
  V3                   t{0, 0, 0};
  float                s = 1;

  V3        apply(const V3& p) const;
  // this after b: (a * b)(p) = a(b(p))
  Transform operator*(const Transform& b) const;
  Transform inverse() const;
};

struct Node
{
  std::string name;
  int         parent = -1;  // into File::nodes
  Transform   local;
  uint32_t    flags = 0;    // NiAVObject flags; bit 0 hides it
};

// BSLightingShaderProperty, the parts a material reads.
struct Shader
{
  // BSLightingShaderType: 0 default, 1 environment map, 2 glow, 5 skin tint, 6 hair tint, ...
  uint32_t type   = 0;
  uint32_t flags1 = 0, flags2 = 0;
  std::array<float, 2> uvOffset{0, 0}, uvScale{1, 1};
  // BSShaderTextureSet's slots: diffuse, normal (specular in its alpha), glow or skin, height, environment
  // cube, environment mask, inner layer or tint, backlight or specular; empty where there is none
  std::vector<std::string> textures;
  V3    emissive{0, 0, 0};
  float emissiveMultiple = 1;
  float alpha            = 1;
  float glossiness       = 80;
  V3    specularColor{1, 1, 1};
  float specularStrength = 1;
  float envMapScale      = 0;  // type 1 only

  const std::string& texture(size_t slot) const;
  bool               specular() const { return flags1 & 0x1; }
  bool               modelSpaceNormals() const { return flags1 & (1u << 12); }
  bool               doubleSided() const { return flags2 & (1u << 4); }
};

// NiAlphaProperty: bit 0 blends, bit 9 tests against `threshold`.
struct Alpha
{
  uint16_t flags     = 0;
  uint8_t  threshold = 128;
  bool     blends() const { return flags & 0x1; }
  bool     tests() const { return flags & (1u << 9); }
};

struct Shape
{
  std::string name;
  int         parent = -1;  // into File::nodes
  Transform   local;
  uint32_t    flags = 0;
  std::vector<V3>                   positions, normals;
  std::vector<std::array<float, 2>> uvs;  // as stored: v runs down the image
  std::vector<uint32_t>             triangles;  // three per triangle
  // skinned: the bones (into File::nodes), each one's skin-to-bone transform, the transform into skin
  // space, and per vertex its influences: vertex v's are [weightStart[v], weightStart[v + 1])
  std::vector<int>       bones;
  std::vector<Transform> skinToBone;
  Transform              skinTransform;
  std::vector<uint32_t>  weightStart;
  std::vector<uint16_t>  weightBone;  // into `bones`
  std::vector<float>     weight;
  std::optional<Shader> shader;
  std::optional<Alpha>  alpha;
  // NiStringExtraData on it: name and value (an outfit's body carries "BODYTRI")
  std::vector<std::pair<std::string, std::string>> extra;

  bool skinned() const { return !bones.empty(); }
  bool hidden() const { return flags & 0x1; }
};

struct File
{
  uint32_t           userVersion = 0, bsVersion = 0;
  std::string        exporter;  // the export info's process script ("Exported using Outfit Studio.")
  std::vector<Node>  nodes;
  std::vector<Shape> shapes;

  // a node's transform into the file's root space
  Transform world(int node) const;
  int       find(const std::string& name) const;
};

// Throws std::runtime_error with what is wrong (not a NIF, the wrong game, cut short).
File read(std::span<const uint8_t> bytes);
File readFile(const std::filesystem::path& path);

// Half-precision float to float.
float halfToFloat(uint16_t h);

}  // namespace fv::nif
