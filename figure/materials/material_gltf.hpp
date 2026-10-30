#pragma once

// A material and glTF, both ways: a file's material as ours, and ours as a file's, KHR_* and FV_*.
// Reading a material and writing it back loses nothing a reader of glTF sees, but for a
// pbrMetallicRoughness's own extras and extensions and a material's MSFT_lod.

#include <string>
#include <vector>

#include <tinygltf/json.hpp>
#include <tinygltf/tiny_gltf.h>

#include <figure/materials/material.hpp>

namespace fv::mat {

Material fromGltf(const tinygltf::Material& material);

// The extensions it uses are appended to `extensionsUsed` once.
tinygltf::Material toGltf(const Material& material, std::vector<std::string>& extensionsUsed);
tinygltf::Material toGltf(const Material& material);

// As glTF's JSON, its texture infos naming their texture under `textureKey`: "slot" says plainly
// that a Lowered's indices are its requests, not a texture table.
nlohmann::json materialJson(const Material& material, const char* textureKey = "index");

// tinygltf::Value from JSON, and back.
tinygltf::Value toValue(const nlohmann::json& j);
nlohmann::json  fromValue(const tinygltf::Value& v);

}  // namespace fv::mat
