/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <tinygltf/tiny_gltf.h>

namespace nvvkgltf {

// Tracks index offsets when merging two models (current array sizes before append).
struct IndexRemapping
{
  int nodes       = 0;
  int meshes      = 0;
  int materials   = 0;
  int textures    = 0;
  int images      = 0;
  int samplers    = 0;
  int accessors   = 0;
  int bufferViews = 0;
  int buffers     = 0;
  int skins       = 0;
  int cameras     = 0;
  int animations  = 0;
  int lights      = 0;  // KHR_lights_punctual extension
  int micromaps   = 0;  // EXT_mesh_opacity_micromap root micromaps[] array
};

struct MergeResult
{
  int              firstNode = -1;
  int              lastNode  = -1;
  std::vector<int> roots;  // appended root node indices (subset of the range)

  [[nodiscard]] bool valid() const { return firstNode >= 0 && lastNode > firstNode; }
};

class SceneMerger
{
public:
  // Merge importedModel into baseModel. Imported scene roots become children of a new wrapper node.
  // The caller must set base model's "imported image" state so the loader uses the right basedir
  // per image: Scene stores baseImageCount and importedSceneBasePath for that.
  // If maxTextureCount is set and the combined texture count would exceed it, returns -1.
  // Returns the index of the new wrapper node in baseModel.nodes on success, or -1 on failure.
  static int merge(tinygltf::Model&        baseModel,
                   const tinygltf::Model&  importedModel,
                   const std::string&      importedSceneName,
                   std::optional<uint32_t> maxTextureCount = std::nullopt);

  // Append the resources of importedModel that nodes point at (buffers to meshes, materials,
  // cameras, lights, micromaps), with their indices rebased, and the animations' samplers. Nodes,
  // skins and animation channels are NOT appended: the caller makes node copies of its own, remaps
  // them with remapNodeResources(), and adds the skins and channels those copies need. Returns the
  // offsets the imported resources were appended at.
  // sharedImages[i] >= 0 says imported image i is already base image sharedImages[i]: its textures
  // point there and it is not appended, so the imported images after it are not at offsets.images + i.
  // sharedBuffers works the same way for buffers: the imported bufferViews read the base buffer.
  static IndexRemapping appendResources(tinygltf::Model&       baseModel,
                                        const tinygltf::Model& importedModel,
                                        std::span<const int>   sharedImages  = {},
                                        std::span<const int>   sharedBuffers = {});

  // Where a merge into baseModel appends each array: their sizes now, lights counted in
  // model.lights and micromaps in the root EXT_mesh_opacity_micromap array.
  static IndexRemapping offsetsOf(const tinygltf::Model& baseModel);

  // Every texture index a material holds, its extensions' included, moved up by `textureOffset`
  // (a material going into a model whose texture list it is appended to).
  static void rebaseMaterialTextures(tinygltf::Material& material, int textureOffset);

  // Rebase the resource indices a node of the imported model holds (mesh, camera, light,
  // EXT_mesh_gpu_instancing accessors) by `offsets`. Children and skin are left to the caller.
  static void remapNodeResources(tinygltf::Node& node, const IndexRemapping& offsets);

  // appendResources with the imported nodes too, then each imported default-scene root picks its own parent: parentOf(root)
  // returns an existing baseModel node to attach it under, or -1 to make it a root of
  // baseModel.scenes[sceneIndex]. For a host adding objects to a scene it mirrors (Figure Viewer
  // putting an item on a figure, under the node it hangs from). An empty import, a texture count
  // over maxTextureCount or a parent past the end of baseModel.nodes refuses the merge, before
  // anything is appended, and returns an invalid MergeResult.
  static MergeResult mergeUnder(tinygltf::Model&                                 baseModel,
                                const tinygltf::Model&                           importedModel,
                                const std::function<int(const tinygltf::Node&)>& parentOf,
                                int                                              sceneIndex,
                                std::optional<uint32_t>                          maxTextureCount = std::nullopt);


private:
  // Implementation in .cpp (helpers in anonymous namespace).
};

}  // namespace nvvkgltf
