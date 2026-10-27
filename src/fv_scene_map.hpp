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
// How the Figure Viewer host addresses the composed scene (docs: daz-viewer
// docs/retained-scene-plan.md, oracle 8): where a figure's build landed, so its dials and poses
// reach the render primitives holding its mesh's bytes; and which prop, topic node and surface a
// glTF node or a picked render node belongs to. On the legacy scene's indices until the host moves to
// handles (P4).
//

#include <unordered_map>
#include <utility>
#include <vector>

namespace nvvkgltf {
class Scene;
struct IndexRemapping;
}  // namespace nvvkgltf

namespace fv {

// A figure's build in the composed scene: buildGltf's maps, rebased by the composer's offsets
struct FigureSceneMap
{
  std::vector<int>              nodeOf;   // scene node -> glTF node
  std::unordered_map<int, int>  sceneOf;  // glTF node -> scene node
  std::vector<int>              meshOf;   // scene mesh -> glTF mesh
  std::vector<int>              skinOf;   // scene mesh -> glTF skin
  std::vector<std::vector<int>> primsOf;  // scene mesh -> the render primitives owning its vertex sets (one per pose)
};

// From buildGltf's maps (indices in the figure's own glTF, -1 for none), the offsets its glTF was
// merged at, and the scene loaded from the merge.
FigureSceneMap mapFigure(std::vector<int>                nodeOf,
                         std::vector<int>                meshOf,
                         std::vector<int>                skinOf,
                         const nvvkgltf::IndexRemapping& base,
                         const nvvkgltf::Scene&          scene);

// The render primitives owning the vertex buffers a glTF mesh's render primitives read, one per set
// (Scene::vertexSets): an upload to one reaches every member, so a host uploads once per entry.
std::vector<int> vertexSetOwnersOfMesh(const nvvkgltf::Scene& scene, int gltfMesh);

// What a prop holds of the composed scene
struct PropAddress
{
  int              root = -1;  // glTF node of the prop root
  std::vector<int> gltfOf;     // topic node -> glTF node
  // its glTF materials, [materials, materials + materialCount): "m:<i>" of its materials topic
  int materials = 0, materialCount = 0;
};

// A pick or a selection, back in the host's terms
struct HostAddress
{
  int prop    = -1;  // index in the props SceneMap::set was given; -1 for none of them
  int node    = -1;  // topic node; -1 for the prop root
  int surface = -1;  // i of "m:<i>", for a pick on one of the prop's materials
};

class SceneMap
{
public:
  // The props now in the scene (anything derived from PropAddress), indexed as given
  template <class Prop>
  void set(const std::vector<Prop>& props)
  {
    clear();
    for(int p = 0; p < int(props.size()); p++)
      add(p, props[size_t(p)]);
  }
  void clear();

  // The prop and topic node of a glTF node (node -1 for the prop root), or prop -1
  HostAddress locate(int gltf) const;
  // The prop whose root a glTF node is, or -1
  int propOfRoot(int gltf) const;
  // The prop, topic node and surface a render node of `scene` draws
  HostAddress pick(const nvvkgltf::Scene& scene, int renderNode) const;

private:
  void add(int prop, const PropAddress& address);

  std::unordered_map<int, std::pair<int, int>> m_where;      // glTF node -> {prop, topic node}
  std::vector<std::pair<int, int>>             m_materials;  // prop -> {first, count}
};

}  // namespace fv
