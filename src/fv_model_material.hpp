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
// A model prop's material as the Material panel's edits make it: the file's glTF material with the
// edited factors and extensions laid over it. One function for what the renderer draws
// (FvMaterials::writeModel) and what an export of the model writes, so the two cannot drift.
//

#include <array>

#include <tinygltf/tiny_gltf.h>

#include <figure/materials/material.hpp>
#include <figure/materials/surface.hpp>

namespace fv::model {

// The extensions an edit can add to a model's material; an export lists those the result has in
// extensionsUsed.
inline constexpr std::array<const char*, 4> kEditedExtensions{"KHR_materials_emissive_strength", "KHR_materials_specular",
                                                              "KHR_materials_clearcoat", "KHR_materials_sheen"};

// `original` with `edits` over it. Only what the panel edits changes; textures stay the file's.
fv::mat::Material editedMaterial(const fv::mat::Material& original, const fv::mat::Edits& edits);
// The same for a file's own material: with no edits it comes back as it is, every byte the file's.
tinygltf::Material editedMaterial(const tinygltf::Material& original, const fv::mat::Edits& edits);

}  // namespace fv::model
