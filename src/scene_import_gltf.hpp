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
  // nvvkgltf::Scene::getImageSearchPaths): one found is kept as its absolute path, one not found
  // (or with no paths given) as the file wrote it
  std::vector<std::filesystem::path> imageSearchPaths;
};

struct ImportResult
{
  std::vector<EntityId>            entities;   // per glTF node; none for a node not in the scene
  std::vector<EntityId>            roots;      // the scene's roots, in order
  std::vector<MaterialId>          materials;  // per glTF material; none when nothing drew it
  std::vector<std::vector<MeshId>> meshes;     // per glTF mesh: its vertex sets
  // per glTF animation: its node channels as a clip, none when it has none. Nothing in the scene
  // holds a clip, so the import hands the caller each one's reference, to release with the prop.
  std::vector<ClipId>              clips;
  std::vector<std::string>         notices;    // what was left out, once each
};

ImportResult importGltf(RetainedScene& scene, const tinygltf::Model& model, const ImportOptions& options = {});

// Whether a material's surfaces need tangents: a normal map, or an anisotropy rotation, which is
// measured from the tangent (as nvvkgltf::Scene::createMissingTangentsForModel decides).
bool materialNeedsTangents(const tinygltf::Material& material);

// A node's local transform: its matrix when it has one, else its TRS
Transform transformOf(const tinygltf::Node& node);

}  // namespace rs
