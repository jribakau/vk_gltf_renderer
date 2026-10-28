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

#include "fv_scene_map.hpp"

namespace fv {

void SceneMap::clear()
{
  m_where.clear();
  m_surfaces.clear();
}

void SceneMap::add(int prop, const PropAddress& address)
{
  if(!address.root.isNone())
    m_where[address.root] = {prop, -1};
  for(int i = 0; i < int(address.entityOf.size()); i++)
    if(!address.entityOf[size_t(i)].isNone())
      m_where[address.entityOf[size_t(i)]] = {prop, i};
  for(int i = 0; i < int(address.materials.size()); i++)
    if(!address.materials[size_t(i)].isNone())
      m_surfaces.try_emplace(address.materials[size_t(i)], prop, i);
}

HostAddress SceneMap::locate(rs::EntityId entity) const
{
  const auto it = m_where.find(entity);
  if(it == m_where.end())
    return {};
  return {.prop = it->second.first, .node = it->second.second};
}

int SceneMap::propOfRoot(rs::EntityId entity) const
{
  const auto it = m_where.find(entity);
  return it == m_where.end() || it->second.second != -1 ? -1 : it->second.first;
}

HostAddress SceneMap::pick(rs::EntityId entity, rs::MaterialId material) const
{
  HostAddress out = locate(entity);
  if(out.prop < 0)
    return out;
  // a surface of the prop hit, not another's
  const auto it = m_surfaces.find(material);
  if(it != m_surfaces.end() && it->second.first == out.prop)
    out.surface = it->second.second;
  return out;
}

}  // namespace fv
