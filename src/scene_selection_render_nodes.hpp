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
// The render nodes a selected scene node draws: its own and its descendants'. What the silhouette
// and Fit read (Resources::selectedRenderNodes) when a node is selected, from a click redirected to
// an ancestor, the Scene Browser or a script. Vulkan-free.
//

#include <unordered_set>
#include <vector>

#include "gltf_scene.hpp"

inline std::unordered_set<int> renderNodesOfNode(const nvvkgltf::Scene& scene, int nodeIndex)
{
  std::vector<int> renderNodeIDs;
  scene.getRenderNodeRegistry().getAllRenderNodesForNodeRecursive(
      nodeIndex, [&scene](int nid) { return scene.getModel().nodes[static_cast<size_t>(nid)].children; }, renderNodeIDs);
  return {renderNodeIDs.begin(), renderNodeIDs.end()};
}
