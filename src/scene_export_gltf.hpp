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
// The retained scene out as glTF (daz-viewer docs/retained-scene-plan.md, "Export"): a fresh
// tinygltf::Model built from the scene's own records, never from a live model, so an export leaves
// the scene as it is (the API is const). Entities become nodes in the scene's order, a mesh instance
// a glTF mesh (one per distinct instance, shared by the entities that draw the same), a vertex set's
// streams one set of accessors that its submeshes' primitives share, each material its payload with
// its texture table's textures appended once, lights at their real intensity (a hidden one's node
// carries KHR_node_visibility false), cameras, skins, morph targets and instancing copies.
//

#include <vector>

#include <tinygltf/tiny_gltf.h>

#include "scene_retained.hpp"

namespace rs {

struct ExportOptions
{
  // what to write, each with everything under it; empty: the whole scene
  std::vector<EntityId> roots;
  // for a GLB: images held as bytes go into the binary buffer; otherwise as data URIs
  bool binary = true;
};

tinygltf::Model exportGltf(const RetainedScene& scene, const ExportOptions& options = {});

}  // namespace rs
