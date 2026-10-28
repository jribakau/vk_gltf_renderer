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
// The loaded scene over --mcp, read-only (a windowed or --scenefile session's: a stream session's
// scene is its host's). The edit tools went with the Figure Viewer's move to the retained scene
// (decision D3 of daz-viewer docs/retained-scene-plan.md). Objects carry the id a host wrote into
// `extras.fvId`, where it wrote one. Tools:
//
//   vk_gltf_scene_info          ids, nodes, materials, meshes/primitives, animations, lights, state
//

#ifdef USE_NVMCP

#include <memory>

#include <nvmcp/element_mcp.hpp>

class GltfRenderer;

// Register the tools on `mcp` (before the element is added to the application).
void registerSceneMcpTools(nvmcp::Element& mcp, const std::shared_ptr<GltfRenderer>& renderer);

#endif  // USE_NVMCP
