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

#include "fv_prop_feed.hpp"

#include <set>

namespace fv {

namespace {
// tinygltf::Light's operator== compares the name, the type and the colour only
bool sameLight(const tinygltf::Light& a, const tinygltf::Light& b)
{
  return a.name == b.name && a.type == b.type && a.color == b.color && a.intensity == b.intensity && a.range == b.range
         && a.spot.innerConeAngle == b.spot.innerConeAngle && a.spot.outerConeAngle == b.spot.outerConeAngle
         && a.extras == b.extras && a.extensions == b.extensions;
}
}  // namespace

FedProp* PropFeed::fed(const std::string& id)
{
  const auto it = m_props.find(id);
  return it == m_props.end() ? nullptr : &it->second;
}

const FedProp* PropFeed::fed(const std::string& id) const
{
  const auto it = m_props.find(id);
  return it == m_props.end() ? nullptr : &it->second;
}

void PropFeed::release(FedProp& p)
{
  // the feed's entities first, then what is left under the root with it
  p.figure.reset();
  for(rs::ClipId c : p.imported.clips)
    m_scene.release(c);
  p.imported = {};
  if(m_scene.entity(p.root))
    m_scene.destroyEntity(p.root);
  p = {};
}

void PropFeed::drop(const std::string& id)
{
  if(FedProp* p = fed(id))
  {
    release(*p);
    m_props.erase(id);
  }
}

void PropFeed::clear()
{
  for(auto& [id, p] : m_props)
    release(p);
  m_props.clear();
}

PropFeed::Result PropFeed::sync(std::span<const PropSpec> props)
{
  Result                result;
  std::set<std::string> wanted;
  for(const PropSpec& s : props)
    wanted.insert(s.id);
  for(auto it = m_props.begin(); it != m_props.end();)
    if(!wanted.count(it->first))
    {
      result.removed.push_back(it->first);
      release(it->second);
      it = m_props.erase(it);
    }
    else
      ++it;

  // the roots first, so a prop can ride on one that comes after it
  for(const PropSpec& s : props)
  {
    FedProp& p = m_props[s.id];
    if(!m_scene.entity(p.root))
    {
      p.root = m_scene.createEntity(s.name, {s.number, -1});
      result.added.push_back(s.id);
    }
  }
  for(const PropSpec& s : props)
  {
    FedProp&           p      = m_props[s.id];
    const FedProp*     up     = s.parent.empty() ? nullptr : fed(s.parent);
    const rs::EntityId parent = up ? up->root : rs::EntityId{};
    if(m_scene.entity(p.root)->parent != parent)
      m_scene.setParent(p.root, parent);  // refused for a cycle: it stays where it was
    if(!(m_scene.entity(p.root)->local == s.transform))
      m_scene.setLocal(p.root, s.transform);
    if(m_scene.entity(p.root)->visible != s.visible)
      m_scene.setVisible(p.root, s.visible);
  }

  for(const PropSpec& s : props)
  {
    FedProp& p = m_props[s.id];
    if(s.figure)
    {
      if(!p.figure)
      {
        p.figure = std::make_unique<FigureFeed>(
            m_scene, FigureFeedOptions{.build = s.figureOptions, .prop = s.number, .parent = p.root, .cageOf = s.figureCageOf});
        p.figure->sync(*s.figure, s.figureRoot);
      }
      else if(s.figureChanged)
      {
        p.figure->setOptions(s.figureOptions);
        p.figure->sync(*s.figure, s.figureRoot);
      }
    }
    if(s.model && s.model != p.model)
    {
      // another file: the last one's import goes
      for(rs::ClipId c : p.imported.clips)
        m_scene.release(c);
      for(rs::EntityId e : p.imported.roots)
        if(m_scene.entity(e))
          m_scene.destroyEntity(e);
      const bool first = !p.model;
      p.model          = s.model;
      // a duplicate holds the model its source does: their meshes are shared
      const rs::ImportResult* source = nullptr;
      for(const auto& [id, other] : m_props)
        if(&other != &p && other.model == s.model && !other.imported.meshes.empty())
          source = &other.imported;
      p.imported =
          rs::importGltf(m_scene, *s.model,
                         {.prop = s.number, .parent = p.root, .imageSearchPaths = s.imageSearchPaths, .meshesOf = source});
      if(first)
        for(const std::string& n : p.imported.notices)
          result.notices.emplace_back(s.id, n);
    }
    if(s.camera)
    {
      const rs::CameraRecord* had = m_scene.camera(p.camera);
      if(!had || !(had->payload == *s.camera))
      {
        p.camera = m_scene.addCamera({.owner = {s.number, -1}, .payload = *s.camera});
        m_scene.setEntityCamera(p.root, p.camera);  // the last one is let go of with it
        m_scene.release(p.camera);
      }
    }
    if(s.light)
    {
      if(const rs::LightRecord* had = m_scene.light(p.light))
      {
        if(!sameLight(had->payload, *s.light))
          m_scene.setLight(p.light, *s.light);
      }
      else
      {
        p.light = m_scene.addLight({.owner = {s.number, -1}, .payload = *s.light});
        m_scene.setEntityLight(p.root, p.light);
        m_scene.release(p.light);
      }
    }
  }
  return result;
}

}  // namespace fv
