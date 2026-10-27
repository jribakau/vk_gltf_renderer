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
// The Figure Viewer's scene as the one glTF the renderer draws (fv_host.hpp): every prop under a
// node of its own, its prop root, which carries where the prop stands and whether it shows. Under
// it goes the prop's own model (a figure the host built, a file it read), or a camera or a light.
//
// Prop root i is node i of the composed model, so the props' roots never move; everything a prop
// brings is appended after them in prop order, and `Placed::base` says where each of its arrays
// starts. Every extras.fvId a prop's model carries ("n:5", "m:2") is prefixed with the prop's id
// ("p:3/n:5"), so ids from two figures never collide. Vulkan-free: the tests compose without a
// renderer.
//

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>
#include <tinygltf/tiny_gltf.h>

#include "gltf_scene_merger.hpp"

namespace fv::composer {

struct Trs
{
  glm::vec3 translation{0.f};
  glm::quat rotation{1.f, 0.f, 0.f, 0.f};
  glm::vec3 scale{1.f};
};

struct Part
{
  std::string id;  // "p:3": the prop root's extras.fvId, and the prefix of every fvId under it
  std::string name;
  Trs         transform;
  bool        visible = true;
  int         parent  = -1;  // another part's index, or -1 for a root of the scene
  // what goes under the prop root: a model, a camera or a light (none: an empty prop)
  const tinygltf::Model*          model = nullptr;
  std::optional<tinygltf::Camera> camera;
  std::optional<tinygltf::Light>  light;
};

struct Placed
{
  int                      root = -1;  // the prop root node, which is the part's index
  nvvkgltf::IndexRemapping base;       // where the part's model's arrays start in the composed model
  int                      firstNode = -1, lastNode = -1;  // the model's nodes, [first, last)
  int                      camera = -1, light = -1;        // the camera or light, composed
};

// The scene of `parts`, and in `placed` (one per part) where each landed. A parent that is not an
// earlier or later part, or that would make a cycle, leaves the part at the root of the scene.
tinygltf::Model compose(const std::vector<Part>& parts, std::vector<Placed>& placed);

// The part whose prop root or model a composed node belongs to, or -1.
int partOfNode(const std::vector<Placed>& placed, int node);

// A glTF, GLB or OBJ file as a model that stands on its own wherever it is merged: its image URIs
// are absolute. False with `error` set when the file could not be read.
bool readModelFile(const std::filesystem::path& file, tinygltf::Model& model, std::string& error);

// A model ready to be written as one GLB: every buffer packed into the first, and every image
// file it points at read into it as a bufferView. False with `error` when an image can't be read.
bool packForGlb(tinygltf::Model& model, std::string& error);

// What a composed scene carries that no file should, stripped for an export (D23): the parts' own
// KHR_animation_pointer channels, KHR_interactivity graphs and KHR_materials_variants, whose indices
// SceneMerger does not rebase. One line per kind it took out, for the user.
std::vector<std::string> stripForFile(tinygltf::Model& model);

// The camera a camera prop puts under its root: perspective, its field of view from a focal length
// on 35 mm film (the vertical one, for a landscape frame of `aspect`).
tinygltf::Camera perspectiveCamera(double focalLengthMm, double aspect = 1.5);

}  // namespace fv::composer
