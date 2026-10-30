/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// glTF into the retained scene (daz-viewer docs/retained-scene-plan.md, "Imported models"): a
// tinygltf::Model's default scene as entities, its meshes as vertex sets with submeshes, its
// materials as payloads over one texture table, its lights, cameras and skins, its animations as
// clips. glTF stays a format at the edge: nothing keeps the Model after the import.
//
// What it draws matches what nvvkgltf::Scene draws of the same model, in the same order (its
// render nodes, depth first, primitive by primitive, copy by copy), tangents included bit for bit:
// the tests compare the two. Two differences on purpose: a primitive with no material takes the
// glTF default material, not the file's first (nvvkgltf::Scene's max(0, material)); and content
// the retained scene does not take is left out with a notice (decision D23).
//

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tinygltf/tiny_gltf.h>

#include "scene_retained.hpp"

namespace rs {

struct ImportOptions
{
  uint32_t prop = 0;   // the owner tags' prop
  EntityId parent;     // where the scene's roots go; none: roots of the retained scene
  int      scene = -1;  // the glTF scene; -1: the default one (or the first)
  // where an image's relative URI is looked for (the file's folder first, as
  // nvvkgltf::Scene::getImageSearchPaths). A URI naming a file, absolute or found here, is kept as
  // its decoded absolute path; one naming none, as the file wrote it.
  std::vector<std::filesystem::path> imageSearchPaths;
  // an earlier import of the same model whose meshes are still in the scene (a duplicate's
  // source): they are taken, a reference each, instead of made again, so the two share their
  // streams, prims and BLASes. Everything else (entities, materials, skins) is the import's own.
  const struct ImportResult* meshesOf = nullptr;
};

struct ImportResult
{
  std::vector<EntityId>            entities;   // per glTF node; none for a node not in the scene
  std::vector<EntityId>            roots;      // the scene's roots, in order
  std::vector<MaterialId>          materials;  // per glTF material; none when nothing drew it
  std::vector<std::vector<MeshId>> meshes;     // per glTF mesh: its vertex sets
  std::vector<SkinId>              skins;      // per glTF skin; none when nothing skins with it
  // per glTF animation: its node channels as a clip, none when it has none. Nothing in the scene
  // holds a clip, so the import hands the caller each one's reference, to release with the prop.
  std::vector<ClipId>              clips;
  std::vector<std::string>         notices;    // what was left out, once each
  // per glTF image, its record (none once nothing holds it: a material's table holds its textures),
  // and the texture table every material of the file shares, per glTF texture
  std::vector<TextureId>              images;
  std::shared_ptr<const TextureTable> textures;
};

ImportResult importGltf(RetainedScene& scene, const tinygltf::Model& model, const ImportOptions& options = {});

// Whether a material's surfaces need tangents: a normal map, or an anisotropy rotation, which is
// measured from the tangent (as nvvkgltf::Scene::createMissingTangentsForModel decides).
bool materialNeedsTangents(const fv::mat::Material& material);

// A node's local transform: its matrix when it has one, else its TRS
Transform transformOf(const tinygltf::Node& node);

// A glTF image as the import records it: its file found on `imageSearchPaths` (an absolute path),
// or the bytes or pixels it holds; and a glTF sampler's values (glTF's defaults for none)
TextureRecord textureRecordOf(const tinygltf::Model& model, size_t image, std::span<const std::filesystem::path> imageSearchPaths);
SamplerDesc samplerOf(const tinygltf::Model& model, int sampler);

}  // namespace rs
