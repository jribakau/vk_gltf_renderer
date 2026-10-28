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
// The retained scene (daz-viewer docs/retained-scene-plan.md): what is drawn, as entities over an
// asset store, instead of one glTF model rebuilt on every structural edit. Vulkan-free: the
// residency layer turns its changes into GPU work; importers and the figure feed fill it; an
// exporter writes a glTF from it. Application-thread only (asserted in debug builds).
//
// Entities form the scene graph (a prop root, its nodes, bones). An entity may draw a mesh
// instance, carry a light or a camera. Records (meshes, materials, textures, skins, lights,
// cameras, clips) are shared by reference count: add*() hands the caller one reference, binding
// adds one, release() drops one, and a record goes with its last reference.
//
// It also keeps which shader features its materials need (features()), counted as instances and
// payloads change rather than found by a scan.
//

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <tinygltf/tiny_gltf.h>

#include "scene_feature_detection.hpp"
#include "scene_retained_handles.hpp"

namespace rs {

struct EntityTag;
struct MeshTag;
struct MaterialTag;
struct TextureTag;
struct SkinTag;
struct LightTag;
struct CameraTag;
struct ClipTag;
using EntityId   = Handle<EntityTag>;
using MeshId     = Handle<MeshTag>;
using MaterialId = Handle<MaterialTag>;
using TextureId  = Handle<TextureTag>;
using SkinId     = Handle<SkinTag>;
using LightId    = Handle<LightTag>;
using CameraId   = Handle<CameraTag>;
using ClipId     = Handle<ClipTag>;

// Who an object is to the host: its prop (the number of "p:N"; 0 for none) and its own index there
// (a figure's scene-graph index, a model file's index), so picks and edits map back without index
// arithmetic over a composed model.
struct OwnerTag
{
  uint32_t prop  = 0;
  int32_t  local = -1;
  bool     operator==(const OwnerTag&) const = default;
  auto     operator<=>(const OwnerTag&) const = default;
};

// A node's local transform as glTF has it: a matrix, or translation, rotation and scale.
struct Transform
{
  glm::vec3                translation{0.f};
  glm::quat                rotation{1.f, 0.f, 0.f, 0.f};
  glm::vec3                scale{1.f};
  std::optional<glm::mat4> matrix;

  // T * R * S, or the matrix (as tinygltf::utils::getNodeMatrix)
  glm::mat4 local() const;
  bool      operator==(const Transform&) const = default;
};

//--------------------------------------------------------------------------------------------------
// Records

// An opacity micromap (EXT_mesh_opacity_micromap): its pre-baked build inputs, in the layout
// VK_KHR_opacity_micromap builds from: the packed opacity bits, the per-triangle records
// (VkMicromapTriangleKHR, `triangleStride` bytes apart) and the usage histogram. Shared by the
// submeshes that index it.
struct OpacityMicromap
{
  struct Usage
  {
    uint32_t count = 0, subdivisionLevel = 0, format = 0;  // VkMicromapUsageKHR
    bool     operator==(const Usage&) const = default;
  };
  std::vector<unsigned char> data, triangles;
  uint32_t                   triangleStride = 8;
  std::vector<Usage>         usages;
  bool                       operator==(const OpacityMicromap&) const = default;
};

// A submesh's triangles over an opacity micromap: triangle i takes the micromap's triangle
// baseTriangle + indices[i], or baseTriangle + i when there are no indices. An index of -1 to -4
// (in its component type's bits) is a special one: the whole triangle transparent, opaque, unknown
// transparent or unknown opaque.
struct MicromapUse
{
  std::shared_ptr<const OpacityMicromap> micromap;
  uint32_t                               baseTriangle = 0;
  std::vector<unsigned char>             indices;                 // one per triangle, tightly packed
  int                                    indexComponentType = 0;  // TINYGLTF_COMPONENT_TYPE_*; 0: no indices
  bool                                   operator==(const MicromapUse&) const = default;
};

// One run of a mesh's indices drawn with one material: a glTF primitive over the mesh's vertices.
struct Submesh
{
  uint32_t firstIndex = 0;
  uint32_t indexCount = 0;
  // the instance's material slot this run draws with (MeshPart::materials)
  uint32_t material = 0;
  // where the run's primitive sits in its source (its glTF mesh's primitive index), the order an
  // entity draws its runs in
  int32_t order = 0;
  // EXT_mesh_opacity_micromap: the micromap its triangles take their opacity from; null for none
  std::shared_ptr<const MicromapUse> micromap;
  bool                               operator==(const Submesh&) const = default;
};

struct MorphTarget
{
  std::vector<glm::vec3> positions, normals, tangents;  // deltas; empty when the target has none
  bool                   operator==(const MorphTarget&) const = default;
};

// How a subdivided mesh's drawn vertices come from its cage (daz-viewer docs/gpu-figures-plan.md,
// F1): the subdivision stencils level by level, the refined vertex each drawn vertex is (UV seam
// splits share one), and the refined triangles with, per refined vertex, the ones whose area-weighted
// normals it sums, in the order fv::mesh::NormalWelder sums them. Shared by every record of the mesh.
struct RefineTables
{
  struct Level
  {
    std::vector<uint32_t> start, columns;  // row r reads columns[start[r] .. start[r + 1]) of the level before
    std::vector<float>    weights;
    uint32_t              rows() const { return start.empty() ? 0 : uint32_t(start.size() - 1); }
  };
  uint32_t              cageCount = 0;
  std::vector<Level>    levels;                        // the last one's rows are the refined vertices
  std::vector<uint32_t> drawnFrom;                     // per drawn vertex, its refined vertex
  std::vector<uint32_t> triangles;                     // refined vertices, three per triangle
  std::vector<uint32_t> normalStart, normalTriangles;  // per refined vertex
  // per refined vertex, added after the last level (strand ribbons' widths); empty: none
  std::vector<glm::vec3> offsets;
  // the drawn triangles whose materials need tangents (none: the mesh has no tangents) and, per drawn
  // vertex, its corners in them (tinygltf::utils::TangentAdjacency): the GPU makes the tangents too
  std::vector<uint32_t> tangentIndices, tangentStart, tangentCorners;
  uint32_t              refinedCount() const { return levels.empty() ? cageCount : levels.back().rows(); }
};

// Vertex streams one or more submeshes index: a glTF mesh's primitives over one set of attribute
// accessors (DAZ material groups), which the GPU keeps in one set of vertex buffers.
struct MeshRecord
{
  std::string              name;
  OwnerTag                 owner;
  std::vector<glm::vec3>   positions, normals;
  std::vector<glm::vec4>   tangents;  // generated when a submesh's material needs them
  std::vector<glm::vec2>   uv0, uv1;
  std::vector<glm::vec4>   colors;
  std::vector<glm::uvec4>  joints;
  std::vector<glm::vec4>   weights;
  // DAZ's skinning (daz-viewer docs/gpu-figures-plan.md F3): influences five to eight (empty: none),
  // and the share of the skinning that is dual quaternion rather than linear blend, per vertex when
  // `dualQuatShares` has one each
  std::vector<glm::uvec4> joints1;
  std::vector<glm::vec4>  weights1;
  float                   dualQuat = 0;
  std::vector<float>      dualQuatShares;
  std::vector<uint32_t>    indices;  // u32 always; a non-indexed primitive gets 0..n-1
  std::vector<Submesh>     submeshes;
  std::vector<MorphTarget> targets;
  std::vector<float>       targetWeights;  // the mesh's default morph weights
  // a subdivided mesh's cage as its morphs have moved it, and the tables that make the drawn
  // positions and normals from it: the residency then refines it on the GPU instead of uploading
  // them (the positions and normals here are what that gives)
  std::vector<glm::vec3>              cage;
  std::shared_ptr<const RefineTables> refine;
  // bumped by a vertex edit, and by one that sets tangents (a drag's steps keep them)
  uint32_t contentRev = 0;
  uint32_t tangentRev = 0;

  // the positions' axis-aligned bounds; empty (min > max) without vertices
  std::array<glm::vec3, 2> bounds() const;
};

struct SamplerDesc
{
  int  magFilter = -1, minFilter = -1;  // glTF enums; -1: unset
  int  wrapS = TINYGLTF_TEXTURE_WRAP_REPEAT, wrapT = TINYGLTF_TEXTURE_WRAP_REPEAT;
  bool operator==(const SamplerDesc&) const = default;
};

// An image, shared by everything that samples it.
struct TextureRecord
{
  std::string                name;
  // a path, absolute once the import found it (ImportOptions::imageSearchPaths); empty when the
  // bytes are held
  std::string                uri;
  std::vector<unsigned char> bytes;     // the encoded image when it came inside the file (a GLB, a data URI)
  std::string                mimeType;
  int                        width = -1, height = -1;  // when known (export reads them here)
  // decoded pixels, when the loader decoded the image itself (`component` channels of `bits`)
  std::vector<unsigned char> pixels;
  int                        component = 0, bits = 8;
  // the extension that names it as a source (KHR_texture_basisu, EXT_texture_webp), "" for core
  std::string sourceExtension;
};

// What a material's texture indices name: entry i of its table is the glTF texture i of the file it
// came from, so the payload keeps its source indices and is never rebased.
struct TextureRef
{
  TextureId   texture;
  SamplerDesc sampler;
};
using TextureTable = std::vector<TextureRef>;

struct MaterialRecord
{
  OwnerTag owner;
  // a glTF material whose texture indices index `textures`
  tinygltf::Material                  payload;
  std::shared_ptr<const TextureTable> textures;
  uint32_t                            rev = 0;  // bumped by setMaterial
};

struct SkinRecord
{
  std::string            name;
  std::vector<EntityId>  joints;
  std::vector<glm::mat4> inverseBind;  // one per joint
  EntityId               skeleton;
};

// An EXT_lights_ies photometric profile: its file as it came (an export writes it back) and its
// samples, kIesProfileSampleCount relative intensities from the nadir (all 1 when unreadable)
struct IesProfileRecord
{
  std::string          name;
  std::vector<uint8_t> bytes;
  std::vector<float>   samples;
  bool                 operator==(const IesProfileRecord&) const = default;
};

// EXT_lights_ies on a light's node. It stands alone: the light is a point light of this colour and
// multiplier that the profile reshapes, and a KHR light on the same node is not drawn.
struct IesLight
{
  uint32_t  profile                           = 0;  // into RetainedScene::iesProfiles()
  float     multiplier                        = 1.f;
  glm::vec3 color                             = glm::vec3(1.f);
  bool      operator==(const IesLight&) const = default;
};

struct LightRecord
{
  OwnerTag                owner;
  tinygltf::Light         payload;  // its real intensity even while hidden
  bool                    shown  = true;
  bool                    skySun = false;
  std::optional<IesLight> ies;  // in place of the payload's light
};

struct CameraRecord
{
  OwnerTag         owner;
  tinygltf::Camera payload;
};

// What a clip's channel drives on its entity: glTF's animation target paths, without
// KHR_animation_pointer's, which stays legacy-only (decision D23).
enum class ClipPath
{
  eTranslation,
  eRotation,
  eScale,
  eWeights,
};

enum class ClipInterpolation
{
  eLinear,
  eStep,
  eCubicSpline,
};

// Keys: their times and values. The values are flat floats, 3 per translation or scale, 4 per
// rotation (x, y, z, w), one per morph target for weights; a cubic spline key holds its
// in-tangent, its value and its out-tangent, in that order.
struct ClipSampler
{
  ClipInterpolation  interpolation = ClipInterpolation::eLinear;
  std::vector<float> inputs;
  std::vector<float> outputs;
  bool               operator==(const ClipSampler&) const = default;
};

struct ClipChannel
{
  EntityId target;  // one that is gone is skipped, never dereferenced
  ClipPath path    = ClipPath::eTranslation;
  uint32_t sampler = 0;
  bool     operator==(const ClipChannel&) const = default;
};

// An animation (a glTF one's node channels): what a ClipPlayer plays. Its keys do not change once
// it is added.
struct ClipRecord
{
  std::string              name;
  OwnerTag                 owner;  // the prop, and the animation's index in its file
  std::vector<ClipSampler> samplers;
  std::vector<ClipChannel> channels;

  // its samplers' first and last key times; empty (start > end) without keys
  std::array<float, 2> range() const;
};

//--------------------------------------------------------------------------------------------------
// Entities

// A mesh's vertex set drawn by an instance, and the material of each of its submeshes.
struct MeshPart
{
  MeshId                  mesh;
  std::vector<MaterialId> materials;  // per submesh (Submesh::material indexes it)
};

struct MeshInstance
{
  std::vector<MeshPart>  parts;
  SkinId                 skin;
  std::vector<float>     morphWeights;  // empty: the meshes' defaults
  std::vector<glm::mat4> copies;        // EXT_mesh_gpu_instancing local matrices; empty: drawn once
};

struct Entity
{
  std::string                 name;
  OwnerTag                    owner;
  EntityId                    parent;
  std::vector<EntityId>       children;
  Transform                   local;
  bool                        visible = true;  // its own; shown() cascades the parents'
  std::optional<MeshInstance> mesh;
  LightId                     light;
  CameraId                    camera;
};

// What the scene draws, one per (entity, mesh part, submesh, instancing copy), in the order the
// entity's runs and copies come: the residency layer's instances. A submesh with no indices draws
// nothing (a surface group a geograft hides whole) until it has some.
struct DrawItem
{
  EntityId   entity;
  MeshId     mesh;
  uint32_t   submesh = 0;
  MaterialId material;
  uint32_t   copy  = 0;
  glm::mat4  world = glm::mat4(1.f);
  bool       shown = true;
};

struct LightItem
{
  EntityId  entity;
  LightId   light;
  glm::mat4 world = glm::mat4(1.f);
  bool      shown = true;
};

// What changed since the last takeChanges(): what the residency layer has to bring to the GPU.
struct SceneChanges
{
  bool                 structure = false;  // entities, instances, lights or cameras came or went
  std::set<EntityId>   transforms;         // their world matrices moved (descendants included)
  std::set<EntityId>   visibility;         // their shown state may have changed (descendants included)
  std::set<EntityId>   morphWeights;       // their mesh instance's morph weights changed
  std::set<MeshId>     meshes;             // vertices edited
  std::set<MaterialId> materials;          // payload edited
  std::set<LightId>    lights;             // payload, shown or sun edited
  std::set<SkinId>     skins;              // joints or inverse binds edited
  bool                 features    = false;  // features() is not what the last takeChanges() saw
  bool                 iesProfiles = false;  // a profile was added
  bool                 empty() const;
};

class RetainedScene
{
public:
  RetainedScene();

  // ---- records (each add hands the caller one reference)
  MeshId     addMesh(MeshRecord mesh);
  MaterialId addMaterial(MaterialRecord material);
  TextureId  addTexture(TextureRecord texture);
  SkinId     addSkin(SkinRecord skin);
  LightId    addLight(LightRecord light);
  CameraId   addCamera(CameraRecord camera);
  ClipId     addClip(ClipRecord clip);

  void retain(MeshId id);
  void retain(MaterialId id);
  void retain(TextureId id);
  void retain(SkinId id);
  void retain(LightId id);
  void retain(CameraId id);
  void retain(ClipId id);
  void release(MeshId id);
  void release(MaterialId id);
  void release(TextureId id);
  void release(SkinId id);
  void release(LightId id);
  void release(CameraId id);
  void release(ClipId id);

  const MeshRecord*     mesh(MeshId id) const { return m_meshes.get(id) ? &m_meshes.get(id)->value : nullptr; }
  const MaterialRecord* material(MaterialId id) const { return m_materials.get(id) ? &m_materials.get(id)->value : nullptr; }
  const TextureRecord*  texture(TextureId id) const { return m_textures.get(id) ? &m_textures.get(id)->value : nullptr; }
  const SkinRecord*     skin(SkinId id) const { return m_skins.get(id) ? &m_skins.get(id)->value : nullptr; }
  const LightRecord*    light(LightId id) const { return m_lights.get(id) ? &m_lights.get(id)->value : nullptr; }
  const CameraRecord*   camera(CameraId id) const { return m_cameras.get(id) ? &m_cameras.get(id)->value : nullptr; }
  const ClipRecord*     clip(ClipId id) const { return m_clips.get(id) ? &m_clips.get(id)->value : nullptr; }
  uint32_t              refs(MeshId id) const { return m_meshes.get(id) ? m_meshes.get(id)->refs : 0; }
  uint32_t              refs(MaterialId id) const { return m_materials.get(id) ? m_materials.get(id)->refs : 0; }
  uint32_t              refs(TextureId id) const { return m_textures.get(id) ? m_textures.get(id)->refs : 0; }
  uint32_t              refs(SkinId id) const { return m_skins.get(id) ? m_skins.get(id)->refs : 0; }
  uint32_t              refs(LightId id) const { return m_lights.get(id) ? m_lights.get(id)->refs : 0; }
  uint32_t              refs(CameraId id) const { return m_cameras.get(id) ? m_cameras.get(id)->refs : 0; }
  uint32_t              refs(ClipId id) const { return m_clips.get(id) ? m_clips.get(id)->refs : 0; }
  size_t                meshCount() const { return m_meshes.size(); }
  size_t                materialCount() const { return m_materials.size(); }
  size_t                textureCount() const { return m_textures.size(); }
  size_t                skinCount() const { return m_skins.size(); }
  size_t                lightCount() const { return m_lights.size(); }
  size_t                cameraCount() const { return m_cameras.size(); }
  size_t                clipCount() const { return m_clips.size(); }
  std::vector<ClipId>   clips() const;  // in slot order

  // edits
  // (`normals`, `tangents` and `cage` empty: kept)
  void setMeshVertices(MeshId                 id,
                       std::vector<glm::vec3> positions,
                       std::vector<glm::vec3> normals,
                       std::vector<glm::vec4> tangents = {},
                       std::vector<glm::vec3> cage     = {});
  // A refined mesh's cage alone, moved (a dial's step): the residency refines it again; `positions`
  // and `normals` stay what they were until a setMeshVertices brings them up to it
  void setMeshCage(MeshId id, std::vector<glm::vec3> cage);
  // joints, inverse binds and skeleton replaced (a rebind)
  void setSkin(SkinId id, SkinRecord skin);
  void setMaterial(MaterialId id, tinygltf::Material payload);
  // the payload with the table its texture indices now index (its textures referenced, the old ones released)
  void setMaterial(MaterialId id, tinygltf::Material payload, std::shared_ptr<const TextureTable> textures);
  void setLight(LightId id, const tinygltf::Light& payload);
  void setLightIes(LightId id, std::optional<IesLight> ies);
  void setLightShown(LightId id, bool shown);
  void setSkySun(LightId id, bool skySun);

  // ---- IES profiles: append-only (a scene replace starts a new store); returns the profile's index,
  // the one already there when the same profile comes again (a lamp model removed and added back)
  uint32_t                          addIesProfile(IesProfileRecord profile);
  std::span<const IesProfileRecord> iesProfiles() const { return m_iesProfiles; }

  // ---- entities
  // A new entity under `parent` (none: a root), at `index` among its siblings (-1: last).
  EntityId createEntity(std::string name, OwnerTag owner, EntityId parent = {}, int index = -1);
  // The entity and everything under it go; what they referenced is released.
  void destroyEntity(EntityId id);
  // Moves an entity under another (none: a root) at `index` among its siblings (-1: last). Refused
  // (false) when `parent` is the entity or under it. `keepWorld`: its local transform changes so
  // it stays where it is.
  bool setParent(EntityId id, EntityId parent, int index = -1, bool keepWorld = false);
  void setLocal(EntityId id, const Transform& local);
  void setVisible(EntityId id, bool visible);
  // Draws `instance` (references its meshes, materials and skin); nullopt draws nothing.
  void setMeshInstance(EntityId id, std::optional<MeshInstance> instance);
  void setMorphWeights(EntityId id, std::vector<float> weights);
  void setEntityLight(EntityId id, LightId light);    // references it
  void setEntityCamera(EntityId id, CameraId camera);  // references it

  const Entity* entity(EntityId id) const { return m_entities.get(id); }
  size_t        entityCount() const { return m_entities.size(); }
  const std::vector<EntityId>& roots() const { return m_roots; }

  // world matrix and cascaded visibility, brought up to date first
  glm::mat4 world(EntityId id);
  bool      shown(EntityId id);
  // The joint matrices of the skin an entity's mesh instance draws with, from the world matrices:
  // inverse(world(id)) * world(joint) * the joint's inverse bind matrix, as
  // nvvkgltf::AnimationSystem::computeSkinning makes them (identity for a joint that is gone).
  // Empty when the entity draws no skinned mesh.
  std::vector<glm::mat4> jointMatrices(EntityId id);

  // ---- views, in document order: depth first from the roots, an entity before its children
  std::vector<EntityId>  entitiesInOrder() const;
  std::vector<DrawItem>  drawItems();
  std::vector<LightItem> lightItems();
  // the drawn positions' world-space bounds; empty (min > max) when nothing is drawn
  std::array<glm::vec3, 2> bounds();

  // ---- shader features
  // The KHR_materials_* and KHR_texture_transform features the scene's materials use: what
  // nvvkgltf::detectSceneFeatures finds in exportGltf(*this). The export writes the materials
  // entities draw with, so a material counts while a submesh of a mesh instance names it (by the
  // submesh's material slot), hidden or not, and empty or not (a group a geograft hides whole
  // can fill again); a material only a feed or an importer holds counts for nothing. Kept as a
  // use count per feature, moved as instances come and go (setMeshInstance, destroyEntity), as
  // payloads change (setMaterial) and at a material's last release, never by a rescan; a payload's
  // own bits (nvvkgltf::detectMaterialFeatures) are worked out once per revision. eDlssGuide is
  // the renderer's (the denoiser's) and never set here.
  nvvkgltf::SceneFeatureSet features() const;

  // ---- change tracking
  SceneChanges takeChanges();
  uint64_t     structureRevision() const { return m_structureRevision; }
  uint64_t     boundsRevision() const { return m_boundsRevision; }

private:
  template <class T>
  struct Counted
  {
    T        value;
    uint32_t refs = 1;
  };
  struct CountedMaterial
  {
    MaterialRecord            value;
    uint32_t                  refs = 1;
    uint32_t                  uses = 0;  // submeshes drawing with it (features())
    nvvkgltf::SceneFeatureSet features;  // its payload's, per revision
  };

  void assertOwner() const;
  void updateWorld();
  void markTransformDirty(EntityId id);
  void markVisibilityDirty(EntityId id);
  void unlink(EntityId id);
  void link(EntityId id, EntityId parent, int index);
  bool isUnder(EntityId id, EntityId ancestor) const;
  void retainInstance(const MeshInstance& instance);
  void releaseInstance(const MeshInstance& instance);
  void releaseMaterialTextures(const MaterialRecord& material);
  // `delta` uses of each material the instance's submeshes draw with
  void countUses(const MeshInstance& instance, int delta);
  void countFeatures(const nvvkgltf::SceneFeatureSet& features, int64_t delta);

  std::thread::id m_owner;

  SlotMap<Counted<MeshRecord>, MeshTag>         m_meshes;
  SlotMap<CountedMaterial, MaterialTag>         m_materials;
  SlotMap<Counted<TextureRecord>, TextureTag>   m_textures;
  SlotMap<Counted<SkinRecord>, SkinTag>         m_skins;
  SlotMap<Counted<LightRecord>, LightTag>       m_lights;
  SlotMap<Counted<CameraRecord>, CameraTag>     m_cameras;
  SlotMap<Counted<ClipRecord>, ClipTag>         m_clips;
  std::vector<IesProfileRecord>                 m_iesProfiles;

  SlotMap<Entity, EntityTag> m_entities;
  std::vector<EntityId>      m_roots;
  // per entity slot: world matrix, cascaded visibility, and whether they are stale
  struct Derived
  {
    glm::mat4 world = glm::mat4(1.f);
    bool      shown = true;
    bool      dirty = true;
  };
  std::vector<Derived> m_derived;
  bool                 m_anyDirty = false;

  // per SceneFeatureSet feature: submesh uses of materials needing it
  std::array<uint32_t, nvvkgltf::SceneFeatureSet::kFeatureCount> m_featureUses{};
  nvvkgltf::SceneFeatureSet m_featuresTaken;  // features() at the last takeChanges()

  SceneChanges m_changes;
  uint64_t     m_structureRevision = 0;
  uint64_t     m_boundsRevision    = 0;
};

}  // namespace rs
