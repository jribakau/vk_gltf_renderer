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
// Live scene editing over --mcp, for a host application that keeps its own copy of the scene
// (Figure Viewer) and mirrors its edits here instead of reloading the file.
//
// Objects are addressed by index or by the id the host wrote into `extras.fvId` of nodes,
// materials and mesh primitives when it exported the scene. Tools:
//
//   vk_gltf_scene_info          ids, nodes, materials, meshes/primitives, animations, lights, state
//   vk_gltf_edit {ops:[...]}    several edits applied in order, landing in the same frame
//   vk_gltf_set_material        {id|index, material: partial glTF material (factors, alpha, extensions)}
//   vk_gltf_set_node            {id|index, visible?, translation?, rotation? (xyzw), scale?}
//   vk_gltf_set_light           {id (node) | index (light), color?, intensity?, radius? (directional: disc radius in km at the sun's distance, soft shadows)}
//   vk_gltf_animation           {clip?, time?, play?, speed?, loop?, reset?, stop?}
//   vk_gltf_set_morph_weights   {id (primitive) | mesh, weights:[...]}
//   vk_gltf_set_vertices        {id (primitive) | mesh, primitive?, positions: base64 f32 xyz, normals?: base64,
//                                indices?: base64 u32 (only these vertices change, one per position)}
//
// Each op inside vk_gltf_edit is one of the single tools' inputs plus "op": "set_material" |
// "set_node" | "set_light" | "animation" | "set_morph_weights" | "set_vertices".
//

#ifdef USE_NVMCP

#include <memory>

#include <nvmcp/element_mcp.hpp>

class GltfRenderer;

// Register the tools on `mcp` (before the element is added to the application).
void registerSceneMcpTools(nvmcp::Element& mcp, const std::shared_ptr<GltfRenderer>& renderer);

#endif  // USE_NVMCP
