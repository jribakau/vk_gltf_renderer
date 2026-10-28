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
// The host's props fed into its retained scene as deltas (daz-viewer docs/retained-scene-plan.md,
// P5): a prop that came is fed under a prop root entity of its own, one that went is destroyed (its
// instances retire at their frame; what rode on it is lifted out first, as the host has put it at
// the scene's root), and the rest are brought up to date in place: parent (a prop's root, or a bone
// of a figure's), transform,
// visibility, a figure's graph through its feed, a camera's or a light's record. What did not change
// keeps its handles, so picks, the selection and the clip that plays stay where they are. Vulkan-free;
// on the retained scene's owner thread.
//

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fv_figure_feed.hpp"
#include "scene_import_gltf.hpp"
#include "scene_retained.hpp"

namespace fv {

// A prop as the host has it now
struct PropSpec
{
  std::string   id;          // "p:N"
  uint32_t      number = 0;  // N: the owner tags' prop
  std::string   name;
  std::string   parent;  // a prop id; "" for the scene's root
  // the node of the parent's figure it rides on (a bone: a hat on the head), -1 for the parent's
  // root; a parent with no such node fed is ridden on at its root
  int           parentNode = -1;
  rs::Transform transform;
  bool          visible = true;
  // what goes under its root, one of: a figure's scene graph and what its feed lowers with
  const SceneGraph*   figure     = nullptr;
  int                 figureRoot = -1;
  build::BuildOptions figureOptions;
  // the figure's graph changed in structure since the last sync (an item worn or taken off, a graft
  // shown or hidden) or its options did: fed whole again; else left to the host's narrow syncs
  bool figureChanged = false;
  // its subdivided meshes' morphed cages, for the GPU to refine (FigureFeedOptions::cageOf); taken
  // when the figure's feed is made
  std::function<std::vector<float>(int mesh)> figureCageOf;
  // DAZ's skinning for its meshes (FigureFeedOptions::dazSkinning), taken when the figure's feed is made
  bool figureDazSkinning = false;
  // DAZ's order for its refined meshes (FigureFeedOptions::dazOrder), taken when the figure's feed is made
  bool figureDazOrder = false;
  // a model's file, imported once (another file: imported again)
  std::shared_ptr<const tinygltf::Model> model;
  std::vector<std::filesystem::path>     imageSearchPaths;
  // a camera's or a light's record (a light at its real intensity: hidden, its entity shines with nothing)
  std::optional<tinygltf::Camera> camera;
  std::optional<tinygltf::Light>  light;
};

// What the feed holds of a prop
struct FedProp
{
  rs::EntityId                           root;
  std::unique_ptr<FigureFeed>            figure;
  std::shared_ptr<const tinygltf::Model> model;     // the file it imported
  rs::ImportResult                       imported;  // its clips' references held here
  rs::CameraId                           camera;
  rs::LightId                            light;
};

class PropFeed
{
public:
  explicit PropFeed(rs::RetainedScene& scene)
      : m_scene(scene)
  {
  }
  // everything it fed goes
  ~PropFeed() { clear(); }
  PropFeed(const PropFeed&)            = delete;
  PropFeed& operator=(const PropFeed&) = delete;

  struct Result
  {
    std::vector<std::string> added, removed;
    // per prop that came: what its import left out, once (D23)
    std::vector<std::pair<std::string, std::string>> notices;
  };
  // Brings the scene up to `props`, in their order
  Result sync(std::span<const PropSpec> props);
  // what it fed of a prop; null for one it doesn't hold
  FedProp*       fed(const std::string& id);
  const FedProp* fed(const std::string& id) const;
  // a prop that goes now, ahead of the next sync (its feed is let go of then)
  void drop(const std::string& id);
  // every prop goes
  void clear();

private:
  void release(FedProp& p);

  rs::RetainedScene&             m_scene;
  std::map<std::string, FedProp> m_props;
};

}  // namespace fv
