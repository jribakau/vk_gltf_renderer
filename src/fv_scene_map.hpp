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
// How the Figure Viewer host addresses its retained scene (docs: daz-viewer
// docs/retained-scene-plan.md, oracle 8): which prop, topic node and surface an entity or a picked
// instance belongs to, by the handles the feed and the importer made.
//

#include <map>
#include <utility>
#include <vector>

#include "scene_retained.hpp"

namespace fv {

// What a prop holds of the scene
struct PropAddress
{
  rs::EntityId                root;       // its prop root
  std::vector<rs::EntityId>   entityOf;   // topic node -> its entity (none: not in the scene)
  std::vector<rs::MaterialId> materials;  // "m:<i>" of its materials topic -> its record (none: nothing draws it)
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

  // The prop and topic node of an entity (node -1 for the prop root), or prop -1
  HostAddress locate(rs::EntityId entity) const;
  // The prop whose root an entity is, or -1
  int propOfRoot(rs::EntityId entity) const;
  // The prop, topic node and surface an instance of `entity` drawing `material` is
  HostAddress pick(rs::EntityId entity, rs::MaterialId material) const;

private:
  void add(int prop, const PropAddress& address);

  std::map<rs::EntityId, std::pair<int, int>>   m_where;     // entity -> {prop, topic node}
  std::map<rs::MaterialId, std::pair<int, int>> m_surfaces;  // material -> {prop, i}
};

}  // namespace fv
