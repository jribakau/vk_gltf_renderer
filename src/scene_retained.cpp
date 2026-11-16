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
#include <cstring>
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
  // A refined mesh's surface is made from its cage, which its morphs move while its positions stay as
  // loaded (or there are none): the cage's box holds the surface, widened by as far as an offset or an
  // HD edit moves a vertex from it (the residency's fitBounds)
  const bool fromCage = refine && !cage.empty();
  for(const glm::vec3& p : fromCage ? cage : positions)
  {
    b[0] = glm::min(b[0], p);
    b[1] = glm::max(b[1], p);
  }
  if(fromCage && b[0].x <= b[1].x)
  {
    float pad = hd ? hd->reach : 0.f;
    float far = 0.f;
    for(const glm::vec3& o : refine->offsets)
      far = std::max(far, glm::length(o));
    b[0] -= glm::vec3(pad + far);
    b[1] += glm::vec3(pad + far);
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
  return !structure && instances.empty() && transforms.empty() && visibility.empty() && copies.empty() && morphWeights.empty() && meshes.empty() && displacements.empty() && hd.empty()
         && materials.empty() && lights.empty() && skins.empty() && !features && !iesProfiles && !water;
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
  const nvvkgltf::SceneFeatureSet features = nvvkgltf::detectMaterialFeatures(material.payload, material.water != nullptr);
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

void RetainedScene::setMeshCage(MeshId id, std::vector<glm::vec3> cage)
{
  assertOwner();
  auto* r = m_meshes.get(id);
  if(!r || !r->value.refine || cage.empty() || cage.size() != r->value.cage.size())
    return;
  r->value.cage = std::move(cage);
  r->value.contentRev++;
  m_changes.meshes.insert(id);
  m_boundsRevision++;
}

bool RetainedScene::setMeshCage(MeshId id, std::span<const float> xyz)
{
  assertOwner();
  auto* r = m_meshes.get(id);
  if(!r || !r->value.refine || xyz.empty() || xyz.size() != r->value.cage.size() * 3)
    return false;
  static_assert(sizeof(glm::vec3) == 3 * sizeof(float));
  const size_t bytes = xyz.size() * sizeof(float);
  if(std::memcmp(r->value.cage.data(), xyz.data(), bytes) == 0)
    return false;
  std::memcpy(r->value.cage.data(), xyz.data(), bytes);
  r->value.contentRev++;
  m_changes.meshes.insert(id);
  m_boundsRevision++;
  return true;
}

bool RetainedScene::setMeshDisplacement(MeshId id, std::span<const float> xyz)
{
  assertOwner();
  auto* r = m_meshes.get(id);
  // a refined mesh's per cage vertex; one drawn as it is, per drawn vertex
  const size_t count = r ? (r->value.refine ? r->value.cage.size() : size_t(r->value.vertexCount())) : 0;
  if(!r || !count || (!xyz.empty() && xyz.size() != count * 3))
    return false;
  std::vector<glm::vec3>& d    = r->value.displacement;
  const bool              none = std::all_of(xyz.begin(), xyz.end(), [](float f) { return f == 0.f; });
  if(none)
  {
    if(d.empty())
      return false;
    d.clear();
  }
  else
  {
    static_assert(sizeof(glm::vec3) == 3 * sizeof(float));
    const size_t bytes = xyz.size() * sizeof(float);
    if(d.size() * 3 == xyz.size() && std::memcmp(d.data(), xyz.data(), bytes) == 0)
      return false;
    d.resize(xyz.size() / 3);
    std::memcpy(d.data(), xyz.data(), bytes);
  }
  m_changes.displacements.insert(id);
  return true;
}

bool RetainedScene::setMeshHd(MeshId id, std::shared_ptr<const fv::subdiv::HdTables> hd)
{
  assertOwner();
  auto* r = m_meshes.get(id);
  if(!r || !r->value.refine || r->value.hd == hd)
    return false;
  if(hd)
  {
    // the tables' rows are the refinement's, level for level
    const RefineTables& t = *r->value.refine;
    if(!hd->topology || hd->topology->levels.size() != t.levels.size() || hd->offsets.size() != t.levels.size())
      return false;
    for(size_t k = 0; k < t.levels.size(); k++)
      if(hd->topology->levels[k].start.size() != size_t(t.levels[k].rows()) + 1
         || hd->offsets[k].size() != hd->topology->levels[k].frame.size() * 3)
        return false;
  }
  r->value.hd = std::move(hd);
  m_changes.hd.insert(id);
  m_boundsRevision++;  // the edits reach past the cage's hull
  return true;
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

void RetainedScene::setMaterial(MaterialId id, fv::mat::Material payload)
{
  assertOwner();
  if(auto* r = m_materials.get(id))
  {
    const nvvkgltf::SceneFeatureSet features = nvvkgltf::detectMaterialFeatures(payload, r->value.water != nullptr);
    countFeatures(r->features, -int64_t(r->uses));
    countFeatures(features, r->uses);
    r->features      = features;
    r->value.payload = std::move(payload);
    r->value.rev++;
    m_changes.materials.insert(id);
  }
}

void RetainedScene::setMaterialWater(MaterialId id, std::shared_ptr<const fv::mat::WaterLayer> water)
{
  assertOwner();
  auto* r = m_materials.get(id);
  if(!r || r->value.water == water || (r->value.water && water && *r->value.water == *water))
    return;
  if((r->value.water != nullptr) != (water != nullptr))
  {
    const nvvkgltf::SceneFeatureSet features = nvvkgltf::detectMaterialFeatures(r->value.payload, water != nullptr);
    countFeatures(r->features, -int64_t(r->uses));
    countFeatures(features, r->uses);
    r->features = features;
  }
  r->value.water  = std::move(water);
  m_changes.water = true;
}

void RetainedScene::setMaterial(MaterialId id, fv::mat::Material payload, std::shared_ptr<const TextureTable> textures)
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

void RetainedScene::setLightLinkBits(LightId id, uint32_t litBit, uint8_t shadowBit)
{
  assertOwner();
  if(auto* r = m_lights.get(id); r && (r->value.litBit != litBit || r->value.shadowBit != shadowBit))
  {
    r->value.litBit    = litBit;
    r->value.shadowBit = shadowBit;
    m_changes.lights.insert(id);
  }
}

void RetainedScene::setLightGroup(LightId id, uint8_t group)
{
  assertOwner();
  if(auto* r = m_lights.get(id); r && r->value.group != group)
  {
    r->value.group = group;
    m_changes.lights.insert(id);
  }
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
  destroyEntities({&id, 1});
}

void RetainedScene::destroyEntities(std::span<const EntityId> ids)
{
  assertOwner();
  std::vector<EntityId> going;
  going.reserve(ids.size());
  for(const EntityId id : ids)
    if(m_entities.get(id))
      going.push_back(id);
  std::sort(going.begin(), going.end());
  going.erase(std::unique(going.begin(), going.end()), going.end());
  // off the lists they are on, each gone through once: a planet's quads giving way took a few of its
  // root's tens of thousands of children a frame, each with a pass over all of them
  std::vector<std::vector<EntityId>*> lists;
  for(const EntityId id : going)
  {
    const EntityId parent = m_entities[id].parent;
    lists.push_back(m_entities.contains(parent) ? &m_entities[parent].children : &m_roots);
  }
  std::sort(lists.begin(), lists.end());
  lists.erase(std::unique(lists.begin(), lists.end()), lists.end());
  for(std::vector<EntityId>* list : lists)
    std::erase_if(*list, [&](EntityId e) { return std::binary_search(going.begin(), going.end(), e); });
  for(const EntityId id : going)
    m_entities[id].parent = {};
  // (one under another of them is gone with it)
  for(const EntityId id : going)
    destroyUnlinked(id);
}

void RetainedScene::destroyUnlinked(EntityId id)
{
  Entity* e = m_entities.get(id);
  if(!e)
    return;
  // its children's list taken first: it goes whole, and erasing them one by one from a list of n was
  // n squared (a model of 400,000 nodes took 33 s to clear)
  const std::vector<EntityId> children = std::move(e->children);
  e->children.clear();
  for(EntityId child : children)
    destroyUnlinked(child);
  e = m_entities.get(id);
  if(e->mesh)
  {
    releaseInstance(*e->mesh);
    m_boundsRevision++;
    m_changes.instances.insert(id);  // (kept though it is gone: what it drew is let go of by it)
  }
  if(!e->light.isNone())
  {
    release(e->light);
    m_litStale = true;
  }
  if(!e->camera.isNone())
    release(e->camera);
  m_entities.erase(id);
  m_changes.transforms.erase(id);
  m_changes.visibility.erase(id);
  m_changes.copies.erase(id);
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
  m_litStale = true;  // (the lights' order may be another)
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

void RetainedScene::setMaterialKeepFull(MaterialId id, bool on)
{
  assertOwner();
  if(auto* r = m_materials.get(id))
    r->value.keepFull = on;
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

void RetainedScene::setOffCamera(EntityId id, bool offCamera)
{
  assertOwner();
  if(Entity* e = m_entities.get(id); e && e->offCamera != offCamera)
  {
    e->offCamera = offCamera;
    markVisibilityDirty(id);  // (the bounds are of what is there, seen or not)
  }
}

void RetainedScene::setShadowless(EntityId id, bool shadowless)
{
  assertOwner();
  if(Entity* e = m_entities.get(id); e && e->shadowless != shadowless)
  {
    e->shadowless = shadowless;
    markVisibilityDirty(id);
  }
}

void RetainedScene::setLightLink(EntityId id, std::optional<LightLink> link)
{
  assertOwner();
  if(Entity* e = m_entities.get(id); e && e->link != link)
  {
    e->link = link;
    markVisibilityDirty(id);
  }
}

void RetainedScene::setDefaultLightLink(LightLink link)
{
  assertOwner();
  if(m_defaultLink == link)
    return;
  m_defaultLink = link;
  for(EntityId r : m_roots)
    markVisibilityDirty(r);
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
  m_changes.instances.insert(id);
  m_changes.copies.erase(id);
  m_changes.structure = true;
  m_structureRevision++;
  m_boundsRevision++;
}

void RetainedScene::setCopies(EntityId id, std::vector<glm::mat4> copies)
{
  assertOwner();
  Entity* e = m_entities.get(id);
  if(!e || !e->mesh)
    return;
  if(e->mesh->copies.size() != copies.size() || copies.empty())
  {
    MeshInstance instance = *e->mesh;
    instance.copies       = std::move(copies);
    setMeshInstance(id, std::move(instance));
    return;
  }
  e->mesh->copies = std::move(copies);
  m_changes.transforms.insert(id);
  m_changes.copies[id].all = true;
}

void RetainedScene::moveCopies(EntityId id, std::span<const uint32_t> which, std::span<const glm::mat4> to)
{
  assertOwner();
  Entity* e = m_entities.get(id);
  if(!e || !e->mesh || which.empty())
    return;
  std::vector<glm::mat4>& copies = e->mesh->copies;
  for(size_t k = 0; k < which.size() && k < to.size(); k++)
    if(which[k] < copies.size())
      copies[which[k]] = to[k];
  m_changes.transforms.insert(id);
  if(CopyChanges& moved = m_changes.copies[id]; !moved.all)
    moved.which.insert(moved.which.end(), which.begin(), which.begin() + std::min(which.size(), to.size()));
}

void RetainedScene::setCopiesShown(EntityId id, std::span<const uint32_t> which, bool shown)
{
  assertOwner();
  Entity* e = m_entities.get(id);
  if(!e || !e->mesh || which.empty() || e->mesh->copies.empty())
    return;
  std::vector<uint8_t>& hidden = e->mesh->hidden;
  if(hidden.empty() && shown)
    return;  // every one shows already
  hidden.resize(e->mesh->copies.size(), 0);  // (one a copy: an instance set with more copies since has more)
  std::vector<uint32_t> changed;
  for(const uint32_t c : which)
    if(c < hidden.size() && hidden[c] != uint8_t(!shown))
    {
      hidden[c] = uint8_t(!shown);
      changed.push_back(c);
    }
  if(changed.empty())
    return;
  if(CopyChanges& moved = m_changes.copies[id]; !moved.all)
    moved.which.insert(moved.which.end(), changed.begin(), changed.end());
  m_changes.visibility.insert(id);
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
  m_litStale = true;
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

Submesh MeshRecord::drawnRun(uint32_t s) const
{
  Submesh  run   = submeshes[s];
  uint32_t first = run.firstIndex, end = run.firstIndex + run.indexCount;
  for(uint32_t j = 0; j < submeshes.size(); j++)
    if(j != s && submeshes[j].mergedInto == s && submeshes[j].indexCount > 0)
    {
      first = std::min(first, submeshes[j].firstIndex);
      end   = std::max(end, submeshes[j].firstIndex + submeshes[j].indexCount);
    }
  run.firstIndex = first;
  run.indexCount = end - first;
  return run;
}

uint32_t MeshRecord::submeshAt(uint32_t s, uint32_t primitive) const
{
  const uint32_t index = drawnRun(s).firstIndex + primitive * 3;
  for(uint32_t j = 0; j < submeshes.size(); j++)
    if((j == s || submeshes[j].mergedInto == s) && index >= submeshes[j].firstIndex
       && index < submeshes[j].firstIndex + submeshes[j].indexCount)
      return j;
  return s;
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

// The entity and its subtree into the change window's set, down to what an earlier call this window
// put there (a stamp, not the set: moveCopies puts an entity there without its children). A pose frame
// marks every bone: each subtree is walked once, not once a bone above it.
void RetainedScene::markDirty(EntityId id, std::set<EntityId>& into, uint64_t Derived::*marked)
{
  m_derived[id.index].dirty = true;
  m_dirtyTops.push_back(id);  // (twice is no matter: a second is passed over once it is walked)
  std::vector<EntityId> stack{id};
  while(!stack.empty())
  {
    const EntityId e = stack.back();
    stack.pop_back();
    if(std::exchange(m_derived[e.index].*marked, m_changeWindow) == m_changeWindow)
      continue;
    into.insert(e);
    for(EntityId c : m_entities[e].children)
      stack.push_back(c);
  }
}

void RetainedScene::markTransformDirty(EntityId id)
{
  markDirty(id, m_changes.transforms, &Derived::transformsMarked);
}

void RetainedScene::markVisibilityDirty(EntityId id)
{
  markDirty(id, m_changes.visibility, &Derived::visibilityMarked);
}

// Only the dirty entities' subtrees, each from its parent's Derived, current since nothing above it is
// dirty: one entity's edit costs its subtree, not the scene
void RetainedScene::updateWorld()
{
  if(m_dirtyTops.empty())
    return;
  std::vector<EntityId> stack;
  for(const EntityId top : std::exchange(m_dirtyTops, {}))
  {
    if(!m_entities.contains(top) || !m_derived[top.index].dirty)
      continue;  // gone, or brought up to date under another
    bool under = false;  // a dirty entity above it walks it
    for(EntityId x = m_entities[top].parent; !x.isNone() && !under; x = m_entities[x].parent)
      under = m_derived[x.index].dirty;
    if(under)
      continue;
    stack.push_back(top);
    while(!stack.empty())
    {
      const EntityId id = stack.back();
      stack.pop_back();
      Derived&       d      = m_derived[id.index];
      const Entity&  e      = m_entities[id];
      const Derived* parent = e.parent.isNone() ? nullptr : &m_derived[e.parent.index];
      const LightLink parentLink = parent ? parent->link : m_defaultLink;
      d.world     = (parent ? parent->world : glm::mat4(1.f)) * e.local.local();
      d.shown     = (!parent || parent->shown) && e.visible;
      d.offCamera = (parent && parent->offCamera) || e.offCamera;
      d.link      = e.link.value_or(parentLink);
      // shadowless cascades as offCamera does, whatever link a child takes as its own
      d.link.shadowless = parentLink.shadowless || e.shadowless;
      d.dirty           = false;
      for(EntityId c : e.children)
        stack.push_back(c);
    }
  }
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

bool RetainedScene::offCamera(EntityId id)
{
  updateWorld();
  return m_entities.contains(id) && m_derived[id.index].offCamera;
}

LightLink RetainedScene::lightLink(EntityId id)
{
  updateWorld();
  return m_entities.contains(id) ? m_derived[id.index].link : LightLink{};
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

std::vector<glm::mat4> RetainedScene::jointPose(EntityId id)
{
  const Entity* e = m_entities.get(id);
  if(!e || !e->mesh)
    return {};
  const SkinRecord* s = skin(e->mesh->skin);
  if(!s)
    return {};
  m_skinWalk++;
  const EntityId         anchor = skinAnchor(id, s->joints);
  std::vector<glm::mat4> out(s->joints.size(), glm::mat4(1.f));
  out.push_back(fromAnchor(id, anchor));
  for(size_t j = 0; j < s->joints.size(); j++)
    if(m_entities.contains(s->joints[j]))
      out[j] = fromAnchor(s->joints[j], anchor);
  return out;
}

EntityId RetainedScene::skinAnchor(EntityId id, const std::vector<EntityId>& joints)
{
  std::vector<EntityId> above;
  for(EntityId x = id; !x.isNone(); x = m_entities[x].parent)
    above.push_back(x);
  size_t highest = 0;
  for(EntityId joint : joints)
  {
    if(!m_entities.contains(joint))
      continue;
    EntityId x = joint;
    for(; !x.isNone() && m_derived[x.index].seenWalk != m_skinWalk; x = m_entities[x].parent)
    {
      m_derived[x.index].seenWalk = m_skinWalk;
      if(const auto at = std::find(above.begin(), above.end(), x); at != above.end())
      {
        highest = std::max(highest, size_t(at - above.begin()));
        break;
      }
    }
    // up past the top without meeting the mesh's entity: another tree
    if(x.isNone())
      return {};
  }
  return above[highest];
}

const glm::mat4& RetainedScene::fromAnchor(EntityId id, EntityId anchor)
{
  static const glm::mat4 identity(1.f);
  if(id == anchor || id.isNone())
    return identity;
  Derived& d = m_derived[id.index];
  if(d.anchorWalk != m_skinWalk)
  {
    d.anchored   = fromAnchor(m_entities[id].parent, anchor) * m_entities[id].local.local();
    d.anchorWalk = m_skinWalk;
  }
  return d.anchored;
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

std::optional<RetainedScene::Placement> RetainedScene::placementOf(EntityId id)
{
  updateWorld();
  const Entity* e = m_entities.get(id);
  if(!e || !e->mesh)
    return std::nullopt;
  const Derived& d = m_derived[id.index];
  Placement      at{.world = d.world, .shown = d.shown, .offCamera = d.offCamera, .link = d.link};
  if(!e->mesh->copies.empty())
  {
    at.copies    = &e->mesh->copies;
    at.hidden    = e->mesh->hidden.empty() ? nullptr : &e->mesh->hidden;
    at.translate = onlyTranslates(d.world);
  }
  return at;
}

RetainedScene::EntityDraws RetainedScene::drawsOf(EntityId id)
{
  EntityDraws                    out;
  const std::optional<Placement> at = placementOf(id);
  if(!at)
    return out;
  out.at          = *at;
  const Entity& e = m_entities[id];
  // the entity's runs in their source order (a glTF mesh's primitive order)
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
        if(m->drawsItself(s) && m->drawnRun(s).indexCount)  // a run merged into another is drawn by it
          runs.push_back({m->submeshes[s].order, p, s});
  std::stable_sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.order < b.order; });
  const Derived& d = m_derived[id.index];
  out.runs.reserve(runs.size());
  for(const Run& r : runs)
  {
    const MeshPart& part = e.mesh->parts[r.part];
    const uint32_t  slot = mesh(part.mesh)->submeshes[r.submesh].material;
    out.runs.push_back({.entity = id, .mesh = part.mesh, .submesh = r.submesh,
                        .material = slot < part.materials.size() ? part.materials[slot] : MaterialId{}, .world = d.world, .shown = d.shown, .offCamera = d.offCamera, .link = d.link});
  }
  return out;
}

std::vector<DrawItem> RetainedScene::drawItems()
{
  updateWorld();
  std::vector<DrawItem> out;
  out.reserve(m_drawItemsWere);  // as many as the last time, most likely: a scene of 100,000 grew the list 17 times
  for(EntityId id : entitiesInOrder())
  {
    if(!m_entities[id].mesh)
      continue;
    // the entity's runs, then each copy of each
    const EntityDraws draws = drawsOf(id);
    for(const DrawItem& run : draws.runs)
    {
      if(!draws.at.copies)
      {
        out.push_back(run);
        continue;
      }
      DrawItem item = run;
      for(uint32_t c = 0; c < draws.at.count(); c++)
      {
        item.copy  = c;
        item.world = draws.at.worldOf(c);
        item.shown = draws.at.shownOf(c);
        out.push_back(item);
      }
    }
  }
  m_drawItemsWere = out.size();
  return out;
}

std::vector<LightItem> RetainedScene::lightItems()
{
  updateWorld();
  // the entities with a light, in document order: looked for again only once one gains or loses its
  // light, goes, or something is moved in the tree (every entity was walked twice a frame, on a
  // planet tens of thousands, for its one or two lights)
  if(m_litStale)
  {
    m_lit.clear();
    for(EntityId id : entitiesInOrder())
      if(!m_entities[id].light.isNone())
        m_lit.push_back(id);
    m_litStale = false;
  }
  std::vector<LightItem> out;
  for(EntityId id : m_lit)
    if(const Entity* e = m_entities.get(id))
      if(const LightRecord* l = light(e->light))
        out.push_back({id, e->light, m_derived[id.index].world, m_derived[id.index].shown && l->shown});
  return out;
}

std::array<glm::vec3, 2> RetainedScene::shownBounds(EntityId root, EntityId except)
{
  updateWorld();
  std::array<glm::vec3, 2> b{glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest())};
  std::map<MeshId, std::array<glm::vec3, 2>> boxes;  // each mesh's, once
  std::vector<EntityId>                      stack;
  if(root.isNone())
    stack.assign(m_roots.begin(), m_roots.end());
  else if(m_entities.contains(root))
    stack.push_back(root);
  while(!stack.empty())
  {
    const EntityId id = stack.back();
    stack.pop_back();
    if(id == except)
      continue;
    const Entity& e = m_entities[id];
    stack.insert(stack.end(), e.children.begin(), e.children.end());
    if(!e.mesh)
      continue;
    const std::optional<Placement> at = placementOf(id);
    if(!at || !at->shown)
      continue;
    for(const MeshPart& part : e.mesh->parts)
    {
      const MeshRecord* m = mesh(part.mesh);
      if(!m)
        continue;
      auto [box, first] = boxes.try_emplace(part.mesh);
      if(first)
        box->second = m->bounds();
      const std::array<glm::vec3, 2>& mb = box->second;
      if(mb[0].x > mb[1].x)
        continue;
      for(uint32_t c = 0; c < at->count(); c++)
        if(at->shownOf(c))
          growByBox(b, mb[0], mb[1], at->worldOf(c));
    }
  }
  return b;
}

SceneChanges RetainedScene::takeChanges()
{
  assertOwner();
  SceneChanges out = std::move(m_changes);
  m_changes        = {};
  m_changeWindow++;
  std::erase_if(out.transforms, [&](EntityId id) { return !m_entities.contains(id); });
  std::erase_if(out.visibility, [&](EntityId id) { return !m_entities.contains(id); });
  std::erase_if(out.copies, [&](const auto& moved) { return !m_entities.contains(moved.first); });
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
