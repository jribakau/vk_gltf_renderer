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

#include "scene_retained.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>

#include <glm/gtc/matrix_transform.hpp>

namespace rs {

glm::mat4 Transform::local() const
{
  if(matrix)
    return *matrix;
  return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

std::array<glm::vec3, 2> MeshRecord::bounds() const
{
  std::array<glm::vec3, 2> b{glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest())};
  for(const glm::vec3& p : positions)
  {
    b[0] = glm::min(b[0], p);
    b[1] = glm::max(b[1], p);
  }
  return b;
}

std::array<float, 2> ClipRecord::range() const
{
  std::array<float, 2> r{std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
  for(const ClipSampler& s : samplers)
    for(float t : s.inputs)
    {
      r[0] = std::min(r[0], t);
      r[1] = std::max(r[1], t);
    }
  return r;
}

bool SceneChanges::empty() const
{
  return !structure && transforms.empty() && visibility.empty() && morphWeights.empty() && meshes.empty()
         && materials.empty() && lights.empty() && skins.empty() && !features && !iesProfiles;
}

RetainedScene::RetainedScene()
    : m_owner(std::this_thread::get_id())
{
}

void RetainedScene::assertOwner() const
{
  assert(std::this_thread::get_id() == m_owner && "the retained scene belongs to the application thread");
}

//--------------------------------------------------------------------------------------------------
// Records

MeshId RetainedScene::addMesh(MeshRecord mesh)
{
  assertOwner();
  m_boundsRevision++;
  return m_meshes.insert({std::move(mesh)});
}

MaterialId RetainedScene::addMaterial(MaterialRecord material)
{
  assertOwner();
  if(material.textures)
    for(const TextureRef& ref : *material.textures)
      retain(ref.texture);
  // nothing draws with it yet, so it counts for no feature until an instance names it
  const nvvkgltf::SceneFeatureSet features = nvvkgltf::detectMaterialFeatures(material.payload);
  return m_materials.insert({.value = std::move(material), .features = features});
}

TextureId RetainedScene::addTexture(TextureRecord texture)
{
  assertOwner();
  return m_textures.insert({std::move(texture)});
}

SkinId RetainedScene::addSkin(SkinRecord skin)
{
  assertOwner();
  return m_skins.insert({std::move(skin)});
}

LightId RetainedScene::addLight(LightRecord light)
{
  assertOwner();
  return m_lights.insert({std::move(light)});
}

CameraId RetainedScene::addCamera(CameraRecord camera)
{
  assertOwner();
  return m_cameras.insert({std::move(camera)});
}

ClipId RetainedScene::addClip(ClipRecord clip)
{
  assertOwner();
  return m_clips.insert({std::move(clip)});
}

void RetainedScene::retain(MeshId id)
{
  if(auto* r = m_meshes.get(id))
    r->refs++;
}
void RetainedScene::retain(MaterialId id)
{
  if(auto* r = m_materials.get(id))
    r->refs++;
}
void RetainedScene::retain(TextureId id)
{
  if(auto* r = m_textures.get(id))
    r->refs++;
}
void RetainedScene::retain(SkinId id)
{
  if(auto* r = m_skins.get(id))
    r->refs++;
}
void RetainedScene::retain(LightId id)
{
  if(auto* r = m_lights.get(id))
    r->refs++;
}
void RetainedScene::retain(CameraId id)
{
  if(auto* r = m_cameras.get(id))
    r->refs++;
}
void RetainedScene::retain(ClipId id)
{
  if(auto* r = m_clips.get(id))
    r->refs++;
}

void RetainedScene::release(MeshId id)
{
  assertOwner();
  if(auto* r = m_meshes.get(id); r && --r->refs == 0)
  {
    m_meshes.erase(id);
    m_boundsRevision++;
  }
}

void RetainedScene::release(MaterialId id)
{
  assertOwner();
  if(auto* r = m_materials.get(id); r && --r->refs == 0)
  {
    // instances hold a reference to what they draw with, so uses are left only after a release
    // too many; those instances draw nothing with it now, and the export writes nothing for it
    countFeatures(r->features, -int64_t(r->uses));
    const MaterialRecord gone = std::move(r->value);
    m_materials.erase(id);
    releaseMaterialTextures(gone);
  }
}

void RetainedScene::release(TextureId id)
{
  assertOwner();
  if(auto* r = m_textures.get(id); r && --r->refs == 0)
    m_textures.erase(id);
}

void RetainedScene::release(SkinId id)
{
  assertOwner();
  if(auto* r = m_skins.get(id); r && --r->refs == 0)
    m_skins.erase(id);
}

void RetainedScene::release(LightId id)
{
  assertOwner();
  if(auto* r = m_lights.get(id); r && --r->refs == 0)
    m_lights.erase(id);
}

void RetainedScene::release(CameraId id)
{
  assertOwner();
  if(auto* r = m_cameras.get(id); r && --r->refs == 0)
    m_cameras.erase(id);
}

void RetainedScene::release(ClipId id)
{
  assertOwner();
  if(auto* r = m_clips.get(id); r && --r->refs == 0)
    m_clips.erase(id);
}

std::vector<ClipId> RetainedScene::clips() const
{
  std::vector<ClipId> out;
  m_clips.forEach([&](ClipId id, const auto&) { out.push_back(id); });
  return out;
}

void RetainedScene::releaseMaterialTextures(const MaterialRecord& material)
{
  if(material.textures)
    for(const TextureRef& ref : *material.textures)
      release(ref.texture);
}

void RetainedScene::setMeshVertices(MeshId                 id,
                                    std::vector<glm::vec3> positions,
                                    std::vector<glm::vec3> normals,
                                    std::vector<glm::vec4> tangents,
                                    std::vector<glm::vec3> cage)
{
  assertOwner();
  auto* r = m_meshes.get(id);
  if(!r || positions.size() != r->value.positions.size() || (!normals.empty() && normals.size() != r->value.normals.size())
     || (!tangents.empty() && tangents.size() != r->value.positions.size())
     || (!cage.empty() && cage.size() != r->value.cage.size()))
    return;
  if(!cage.empty())
    r->value.cage = std::move(cage);
  r->value.positions = std::move(positions);
  if(!normals.empty())
    r->value.normals = std::move(normals);
  if(!tangents.empty())
  {
    r->value.tangents = std::move(tangents);
    r->value.tangentRev++;
  }
  r->value.contentRev++;
  m_changes.meshes.insert(id);
  m_boundsRevision++;
}

void RetainedScene::setSkin(SkinId id, SkinRecord skin)
{
  assertOwner();
  if(auto* r = m_skins.get(id))
  {
    r->value = std::move(skin);
    m_changes.skins.insert(id);
  }
}

void RetainedScene::setMaterial(MaterialId id, tinygltf::Material payload)
{
  assertOwner();
  if(auto* r = m_materials.get(id))
  {
    const nvvkgltf::SceneFeatureSet features = nvvkgltf::detectMaterialFeatures(payload);
    countFeatures(r->features, -int64_t(r->uses));
    countFeatures(features, r->uses);
    r->features      = features;
    r->value.payload = std::move(payload);
    r->value.rev++;
    m_changes.materials.insert(id);
  }
}

void RetainedScene::setMaterial(MaterialId id, tinygltf::Material payload, std::shared_ptr<const TextureTable> textures)
{
  assertOwner();
  auto* r = m_materials.get(id);
  if(!r)
    return;
  if(textures != r->value.textures)
  {
    if(textures)
      for(const TextureRef& ref : *textures)
        retain(ref.texture);
    MaterialRecord old{.textures = std::move(r->value.textures)};
    r->value.textures = std::move(textures);
    releaseMaterialTextures(old);
  }
  setMaterial(id, std::move(payload));
}

void RetainedScene::setLight(LightId id, const tinygltf::Light& payload)
{
  assertOwner();
  if(auto* r = m_lights.get(id))
  {
    r->value.payload = payload;
    m_changes.lights.insert(id);
  }
}

void RetainedScene::setLightIes(LightId id, std::optional<IesLight> ies)
{
  assertOwner();
  if(auto* r = m_lights.get(id); r && r->value.ies != ies)
  {
    r->value.ies = ies;
    m_changes.lights.insert(id);
  }
}

uint32_t RetainedScene::addIesProfile(IesProfileRecord profile)
{
  assertOwner();
  if(auto it = std::find(m_iesProfiles.begin(), m_iesProfiles.end(), profile); it != m_iesProfiles.end())
    return uint32_t(it - m_iesProfiles.begin());
  m_iesProfiles.push_back(std::move(profile));
  m_changes.iesProfiles = true;
  return uint32_t(m_iesProfiles.size() - 1);
}

void RetainedScene::setLightShown(LightId id, bool shown)
{
  assertOwner();
  if(auto* r = m_lights.get(id); r && r->value.shown != shown)
  {
    r->value.shown = shown;
    m_changes.lights.insert(id);
  }
}

void RetainedScene::setSkySun(LightId id, bool skySun)
{
  assertOwner();
  if(auto* r = m_lights.get(id); r && r->value.skySun != skySun)
  {
    r->value.skySun = skySun;
    m_changes.lights.insert(id);
  }
}

//--------------------------------------------------------------------------------------------------
// Entities

EntityId RetainedScene::createEntity(std::string name, OwnerTag owner, EntityId parent, int index)
{
  assertOwner();
  Entity e;
  e.name  = std::move(name);
  e.owner = owner;
  const EntityId id = m_entities.insert(std::move(e));
  if(m_derived.size() < m_entities.capacity())
    m_derived.resize(m_entities.capacity());
  m_derived[id.index] = {};
  link(id, m_entities.contains(parent) ? parent : EntityId{}, index);
  markTransformDirty(id);
  m_changes.structure = true;
  m_structureRevision++;
  return id;
}

void RetainedScene::destroyEntity(EntityId id)
{
  assertOwner();
  Entity* e = m_entities.get(id);
  if(!e)
    return;
  for(EntityId child : std::vector<EntityId>(e->children))
    destroyEntity(child);
  e = m_entities.get(id);
  if(e->mesh)
  {
    releaseInstance(*e->mesh);
    m_boundsRevision++;
  }
  if(!e->light.isNone())
    release(e->light);
  if(!e->camera.isNone())
    release(e->camera);
  unlink(id);
  m_entities.erase(id);
  m_changes.transforms.erase(id);
  m_changes.visibility.erase(id);
  m_changes.morphWeights.erase(id);
  m_changes.structure = true;
  m_structureRevision++;
}

bool RetainedScene::isUnder(EntityId id, EntityId ancestor) const
{
  for(const Entity* e = m_entities.get(id); e; e = m_entities.get(e->parent))
    if(e->parent == ancestor)
      return true;
  return false;
}

bool RetainedScene::setParent(EntityId id, EntityId parent, int index, bool keepWorld)
{
  assertOwner();
  if(!m_entities.contains(id) || (!parent.isNone() && (!m_entities.contains(parent) || parent == id || isUnder(parent, id))))
    return false;
  const glm::mat4 before = keepWorld ? world(id) : glm::mat4(1.f);
  unlink(id);
  link(id, parent, index);
  if(keepWorld)
  {
    const glm::mat4 parentWorld = parent.isNone() ? glm::mat4(1.f) : world(parent);
    m_entities[id].local        = Transform{.matrix = glm::inverse(parentWorld) * before};
  }
  markTransformDirty(id);
  markVisibilityDirty(id);
  m_changes.structure = true;
  m_structureRevision++;
  m_boundsRevision++;
  return true;
}

void RetainedScene::setLocal(EntityId id, const Transform& local)
{
  assertOwner();
  if(Entity* e = m_entities.get(id))
  {
    e->local = local;
    markTransformDirty(id);
    m_boundsRevision++;
  }
}

void RetainedScene::setVisible(EntityId id, bool visible)
{
  assertOwner();
  if(Entity* e = m_entities.get(id); e && e->visible != visible)
  {
    e->visible = visible;
    markVisibilityDirty(id);
    m_boundsRevision++;
  }
}

void RetainedScene::setMeshInstance(EntityId id, std::optional<MeshInstance> instance)
{
  assertOwner();
  Entity* e = m_entities.get(id);
  if(!e)
    return;
  if(instance)
    retainInstance(*instance);
  if(e->mesh)
    releaseInstance(*e->mesh);
  e->mesh = std::move(instance);
  m_changes.structure = true;
  m_structureRevision++;
  m_boundsRevision++;
}

void RetainedScene::setMorphWeights(EntityId id, std::vector<float> weights)
{
  assertOwner();
  if(Entity* e = m_entities.get(id); e && e->mesh)
  {
    e->mesh->morphWeights = std::move(weights);
    m_changes.morphWeights.insert(id);
  }
}

void RetainedScene::setEntityLight(EntityId id, LightId light)
{
  assertOwner();
  Entity* e = m_entities.get(id);
  if(!e)
    return;
  retain(light);
  if(!e->light.isNone())
    release(e->light);
  e->light            = m_lights.contains(light) ? light : LightId{};
  m_changes.structure = true;
  m_structureRevision++;
}

void RetainedScene::setEntityCamera(EntityId id, CameraId camera)
{
  assertOwner();
  Entity* e = m_entities.get(id);
  if(!e)
    return;
  retain(camera);
  if(!e->camera.isNone())
    release(e->camera);
  e->camera           = m_cameras.contains(camera) ? camera : CameraId{};
  m_changes.structure = true;
  m_structureRevision++;
}

void RetainedScene::retainInstance(const MeshInstance& instance)
{
  for(const MeshPart& part : instance.parts)
  {
    retain(part.mesh);
    for(MaterialId m : part.materials)
      retain(m);
  }
  retain(instance.skin);
  countUses(instance, 1);
}

void RetainedScene::releaseInstance(const MeshInstance& instance)
{
  // while its meshes are still there to say which slots their submeshes draw with
  countUses(instance, -1);
  for(const MeshPart& part : instance.parts)
  {
    release(part.mesh);
    for(MaterialId m : part.materials)
      release(m);
  }
  if(!instance.skin.isNone())
    release(instance.skin);
}

void RetainedScene::countUses(const MeshInstance& instance, int delta)
{
  for(const MeshPart& part : instance.parts)
    if(const MeshRecord* m = mesh(part.mesh))
      for(const Submesh& sub : m->submeshes)
        if(sub.material < part.materials.size())
          if(CountedMaterial* r = m_materials.get(part.materials[sub.material]))
          {
            r->uses += delta;
            countFeatures(r->features, delta);
          }
}

void RetainedScene::countFeatures(const nvvkgltf::SceneFeatureSet& features, int64_t delta)
{
  if(delta == 0)
    return;
  for(size_t f = 0; f < m_featureUses.size(); f++)
    if(features.has(nvvkgltf::SceneFeatureSet::Feature(f)))
    {
      assert(delta > 0 || m_featureUses[f] >= uint64_t(-delta));
      m_featureUses[f] = uint32_t(int64_t(m_featureUses[f]) + delta);
    }
}

nvvkgltf::SceneFeatureSet RetainedScene::features() const
{
  nvvkgltf::SceneFeatureSet out;
  for(size_t f = 0; f < m_featureUses.size(); f++)
    out.set(nvvkgltf::SceneFeatureSet::Feature(f), m_featureUses[f] > 0);
  return out;
}

void RetainedScene::link(EntityId id, EntityId parent, int index)
{
  m_entities[id].parent         = parent;
  std::vector<EntityId>& owners = parent.isNone() ? m_roots : m_entities[parent].children;
  if(index < 0 || index >= int(owners.size()))
    owners.push_back(id);
  else
    owners.insert(owners.begin() + index, id);
}

void RetainedScene::unlink(EntityId id)
{
  Entity&                e      = m_entities[id];
  std::vector<EntityId>& owners = m_entities.contains(e.parent) ? m_entities[e.parent].children : m_roots;
  owners.erase(std::remove(owners.begin(), owners.end(), id), owners.end());
  e.parent = {};
}

void RetainedScene::markTransformDirty(EntityId id)
{
  m_derived[id.index].dirty = true;
  m_anyDirty                = true;
  std::vector<EntityId> stack{id};
  while(!stack.empty())
  {
    const EntityId e = stack.back();
    stack.pop_back();
    m_changes.transforms.insert(e);
    for(EntityId c : m_entities[e].children)
      stack.push_back(c);
  }
}

void RetainedScene::markVisibilityDirty(EntityId id)
{
  m_derived[id.index].dirty = true;
  m_anyDirty                = true;
  std::vector<EntityId> stack{id};
  while(!stack.empty())
  {
    const EntityId e = stack.back();
    stack.pop_back();
    m_changes.visibility.insert(e);
    for(EntityId c : m_entities[e].children)
      stack.push_back(c);
  }
}

void RetainedScene::updateWorld()
{
  if(!m_anyDirty)
    return;
  const std::function<void(EntityId, const glm::mat4&, bool, bool)> walk = [&](EntityId id, const glm::mat4& parentWorld,
                                                                               bool parentShown, bool force) {
    Derived&      d    = m_derived[id.index];
    const Entity& e    = m_entities[id];
    const bool    redo = force || d.dirty;
    if(redo)
    {
      d.world = parentWorld * e.local.local();
      d.shown = parentShown && e.visible;
      d.dirty = false;
    }
    for(EntityId c : e.children)
      walk(c, d.world, d.shown, redo);
  };
  for(EntityId r : m_roots)
    walk(r, glm::mat4(1.f), true, false);
  m_anyDirty = false;
}

glm::mat4 RetainedScene::world(EntityId id)
{
  updateWorld();
  return m_entities.contains(id) ? m_derived[id.index].world : glm::mat4(1.f);
}

bool RetainedScene::shown(EntityId id)
{
  updateWorld();
  return m_entities.contains(id) && m_derived[id.index].shown;
}

std::vector<glm::mat4> RetainedScene::jointMatrices(EntityId id)
{
  const Entity* e = m_entities.get(id);
  if(!e || !e->mesh)
    return {};
  const SkinRecord* s = skin(e->mesh->skin);
  if(!s)
    return {};
  updateWorld();
  std::vector<glm::mat4> out(s->joints.size(), glm::mat4(1.f));
  const glm::mat4        invNode = glm::inverse(m_derived[id.index].world);
  for(size_t j = 0; j < s->joints.size(); j++)
    if(m_entities.contains(s->joints[j]))
      out[j] = invNode * m_derived[s->joints[j].index].world * (j < s->inverseBind.size() ? s->inverseBind[j] : glm::mat4(1.f));
  return out;
}

//--------------------------------------------------------------------------------------------------
// Views

std::vector<EntityId> RetainedScene::entitiesInOrder() const
{
  std::vector<EntityId> out;
  out.reserve(m_entities.size());
  std::vector<EntityId> stack(m_roots.rbegin(), m_roots.rend());
  while(!stack.empty())
  {
    const EntityId id = stack.back();
    stack.pop_back();
    out.push_back(id);
    const std::vector<EntityId>& children = m_entities[id].children;
    stack.insert(stack.end(), children.rbegin(), children.rend());
  }
  return out;
}

std::vector<DrawItem> RetainedScene::drawItems()
{
  updateWorld();
  std::vector<DrawItem> out;
  for(EntityId id : entitiesInOrder())
  {
    const Entity& e = m_entities[id];
    if(!e.mesh)
      continue;
    // the entity's runs in their source order (a glTF mesh's primitive order), then each copy
    struct Run
    {
      int32_t  order;
      size_t   part;
      uint32_t submesh;
    };
    std::vector<Run> runs;
    for(size_t p = 0; p < e.mesh->parts.size(); p++)
      if(const MeshRecord* m = mesh(e.mesh->parts[p].mesh))
        for(uint32_t s = 0; s < m->submeshes.size(); s++)
          if(m->submeshes[s].indexCount)
            runs.push_back({m->submeshes[s].order, p, s});
    std::stable_sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.order < b.order; });
    const Derived& d = m_derived[id.index];
    for(const Run& r : runs)
    {
      const MeshPart& part = e.mesh->parts[r.part];
      const uint32_t  slot = mesh(part.mesh)->submeshes[r.submesh].material;
      DrawItem        item{.entity = id, .mesh = part.mesh, .submesh = r.submesh,
                           .material = slot < part.materials.size() ? part.materials[slot] : MaterialId{}, .world = d.world, .shown = d.shown};
      if(e.mesh->copies.empty())
      {
        out.push_back(item);
        continue;
      }
      for(uint32_t c = 0; c < e.mesh->copies.size(); c++)
      {
        item.copy  = c;
        item.world = d.world * e.mesh->copies[c];
        out.push_back(item);
      }
    }
  }
  return out;
}

std::vector<LightItem> RetainedScene::lightItems()
{
  updateWorld();
  std::vector<LightItem> out;
  for(EntityId id : entitiesInOrder())
  {
    const Entity& e = m_entities[id];
    if(const LightRecord* l = light(e.light))
      out.push_back({id, e.light, m_derived[id.index].world, m_derived[id.index].shown && l->shown});
  }
  return out;
}

std::array<glm::vec3, 2> RetainedScene::bounds()
{
  std::array<glm::vec3, 2> b{glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest())};
  for(const DrawItem& item : drawItems())
  {
    if(!item.shown)
      continue;
    const std::array<glm::vec3, 2> mb = mesh(item.mesh)->bounds();
    if(mb[0].x > mb[1].x)
      continue;
    for(int c = 0; c < 8; c++)
    {
      const glm::vec3 corner(c & 1 ? mb[1].x : mb[0].x, c & 2 ? mb[1].y : mb[0].y, c & 4 ? mb[1].z : mb[0].z);
      const glm::vec3 w = glm::vec3(item.world * glm::vec4(corner, 1.f));
      b[0]              = glm::min(b[0], w);
      b[1]              = glm::max(b[1], w);
    }
  }
  return b;
}

SceneChanges RetainedScene::takeChanges()
{
  assertOwner();
  SceneChanges out = std::move(m_changes);
  m_changes        = {};
  std::erase_if(out.transforms, [&](EntityId id) { return !m_entities.contains(id); });
  std::erase_if(out.visibility, [&](EntityId id) { return !m_entities.contains(id); });
  std::erase_if(out.morphWeights, [&](EntityId id) { return !m_entities.contains(id); });
  std::erase_if(out.meshes, [&](MeshId id) { return !m_meshes.contains(id); });
  std::erase_if(out.materials, [&](MaterialId id) { return !m_materials.contains(id); });
  std::erase_if(out.lights, [&](LightId id) { return !m_lights.contains(id); });
  std::erase_if(out.skins, [&](SkinId id) { return !m_skins.contains(id); });
  out.features    = features() != m_featuresTaken;
  m_featuresTaken = features();
  return out;
}

}  // namespace rs
