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
// The retained scene on the GPU (docs: daz-viewer docs/retained-scene-plan.md, "GPU residency",
// "The sync and the frame contract"): what a RetainedScene draws, kept resident through
// rs::GpuBackend and brought up to date once per frame from the store's changes. Vulkan-free, so
// the fake backend and ResidencyOracle (tests/common) drive it in the test build; the Vulkan
// backend binds it to the renderer.
//
// - A mesh record's vertex streams are one buffer each; a submesh drawn somewhere is a prim: its own
//   uint32 index buffer and BLAS over the mesh's streams.
// - On a device that builds them, a submesh's opacity micromap (EXT_mesh_opacity_micromap) is built
//   once for the prims that link it and retired after the last of their BLASes; a prim that links
//   one holds its micromap indices. hasOpacityMicromaps() (the path tracer's USE_OMM) is whether
//   any prim does.
// - A draw item (entity, mesh, submesh, instancing copy) is an instance: a render-node row and a TLAS
//   row, the TLAS row's custom index its prim slot.
// - A mesh an entity draws with a skin, or a mesh with morph targets, is deformed for that entity (a
//   drawer): the drawer owns output positions, normals and tangents, which a morph pass and then a
//   skin pass write from the mesh's streams, and its own prims, whose BLASes are built over the
//   outputs and refit after each pass. A pass runs only when what it computes from changed: the
//   joint matrices, the morph weights, or the mesh's vertices (a dial). The joint matrices, their
//   normal matrices and the morph weights of every drawer are packed into three buffers.
// - Motion for DLSS: per prim slot, the address of what its positions held when the last frame was
//   drawn (0 for none: no deformation motion), copied at the top of a sync from positions written
//   the sync before, before anything writes them again; per instance slot, the world matrix the
//   last frame drew it with. A new slot starts with its current values, so it shows no motion,
//   never another object's.
// - Instance, prim and material slots come lowest-first from pools, so a fresh scene numbers them
//   in document order; a freed slot is a hole (a zeroed row, a null TLAS row) until its retire
//   frame, when it goes back to its pool. The tables are sized by capacity, a power of two at or
//   above the high-water mark, and the TLAS keeps its handle until capacity grows.
// - A material slot is a GltfShadeMaterial row; the textures its payload samples hold texture
//   slots, one per (texture, colour space, sampler), over one image per texture and colour space (a
//   map one material samples as colour and another as data is decoded both ways; slot 0 is the
//   default image);
//   samplers hold sampler slots (slot 0 the default sampler). Each material slot owns one span of
//   the texture-info table (entry 0 is "no texture"), placed below the uint16 limit; a sync converts
//   and uploads only the material slots that changed, their rows and their spans. An edit whose
//   payload needs a feature the live shader variant lacks (setShaderFeatures) is held: its slot keeps
//   the row it had until a sync whose variant has them (D15).
// - Lights are dense, in document order, packed again when one changes; the emissive table is
//   derived from the shown instances whose materials emit, built again in the frame its emitters
//   or their slots change. An emitter's vertex writes (a dial, a drape) are the exception: the
//   table's areas lag until the writes stop for emissiveSettleFrames, then it is built once.
// - Everything freed goes through GpuBackend::retire, a frame cycle later; nothing waits for idle.
//
// Application thread only.
//

#include <cstdint>
#include <array>
#include <map>
#include <optional>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "gltf_material_cache.hpp"
#include "scene_gpu_backend.hpp"
#include "scene_material_bits.hpp"
#include "scene_retained.hpp"

namespace nvutils {
class IDPool;
}

namespace rs {

struct ResidencyOptions
{
  uint32_t minInstances      = 256;           // residencyMinInstances: the least instance capacity
  uint32_t minPrims          = 256;           // and prim capacity
  uint64_t blasScratchBudget = 512ull << 20;  // blasScratchBudgetMB (the Vulkan backend's batches)
  // frames with no vertex write to an emitter before the emissive table takes its new areas
  uint64_t emissiveSettleFrames = 8;
  // textureGraceMB: the images nothing samples that stay decoded, for the next scene or edit to take
  uint64_t textureGraceBytes = 256ull << 20;
  // the texture-info table's entries, which GltfShadeMaterial names by uint16 (D17)
  uint32_t textureInfoLimit = 65536;
  // blasRefitsBeforeRebuild: a BLAS refitted this many times since its build is built again in place
  // of the next refit, as a refit keeps the build's tree however far the vertices move (0: never)
  uint32_t refitsBeforeRebuild = 0;
  // blasCompactStatic: the BLAS of a prim drawn from its mesh's own streams is built for compaction
  // and replaced by its compacted copy once the size is known; a vertex edit builds it again
  // refittable
  bool compactStatic = false;
};

// What makes two decodes the same image: the file (its path, size and write time) or the bytes or
// pixels a record holds (their hash and size), and the colour space. None for a file that isn't
// there or a record with nothing to decode.
struct TextureKey
{
  std::string path;
  int64_t     mtime                                = 0;
  uint64_t    size                                 = 0;
  uint64_t    hash                                 = 0;
  bool        srgb                                 = false;
  auto        operator<=>(const TextureKey&) const = default;
};
std::optional<TextureKey> textureKeyOf(const TextureRecord& record, bool srgb);

// What a sync did, for the renderer (see "The contract")
struct SyncResult
{
  bool changed           = false;  // any GPU op: the frame resets
  bool drawListChanged   = false;  // instances came or went
  bool buffersChanged    = false;  // a table or stream buffer was replaced
  bool visibilityChanged = false;  // an instance was shown or hidden
  bool tlasHandleChanged = false;  // the TLAS was recreated (capacity grew)
  bool verticesWritten   = false;  // a mesh's vertices were uploaded again, or a deform pass ran
  bool featuresChanged   = false;
  bool dlssReset         = false;  // a scene replace, or a material or light edited
};

class Residency
{
public:
  // A mesh record's streams on the GPU
  struct MeshGpu
  {
    BufferHandle         positions, normals, tangents, uv0, uv1, colors;
    uint32_t             vertexCount = 0;
    uint32_t             contentRev  = 0;
    uint32_t             tangentRev  = 0;
    std::vector<int32_t> prims;  // per submesh: its static prim slot, -1 while nothing draws it so
    // what its drawers deform with: joints (int4) and weights per vertex, and its targets' deltas
    // packed target-major; made when the first drawer needs them
    BufferHandle joints, weights, positionDeltas, normalDeltas, tangentDeltas;
    BufferHandle joints1, weights1, dualQuatShares;  // DAZ's skinning (MeshRecord::joints1)
    // DAZ's order (MeshRecord::cageJoints): the cage's influences and shares
    BufferHandle cageJoints, cageWeights, cageJoints1, cageWeights1, cageShares;
    uint32_t     targetCount = 0;
    // its positions as the last frame drew them, kept once a host edit writes them (static prims)
    BufferHandle prevPositions;
    bool         prevValid = false;
    // a refined mesh (MeshRecord::refine): its cage, and the tables and working streams that make
    // its positions and normals from it on the GPU (gpu-figures-plan.md F1); empty otherwise
    struct Refine
    {
      struct Level
      {
        BufferHandle start, columns, weights, out;  // out: this level's vertices
        uint32_t     rows = 0, sourceCount = 0;
      };
      BufferHandle       cage, refinedNormals, drawnFrom, triangles, normalStart, normalTriangles, offsets;
      BufferHandle       tangentTriangles, tangentStart, tangentCorners;  // none: the CPU's tangents
      std::vector<Level> levels;
      uint32_t           refinedCount = 0, triangleCount = 0, tangentTriangleCount = 0;
    };
    std::optional<Refine> refine;
  };
  // A mesh one entity draws deformed: what the passes write, its prims, and what they last ran with
  struct DrawerGpu
  {
    EntityId             entity;
    MeshId               mesh;
    BufferHandle         positions, normals, tangents;  // the outputs: its prims' vertex streams
    // with both passes, what the morph writes and the skin reads (the skin never reads its own output)
    BufferHandle         morphedPositions, morphedNormals, morphedTangents;
    BufferHandle         prevPositions;
    bool                 prevValid = false;
    std::vector<int32_t> prims;  // per submesh
    uint32_t             users       = 0;  // instances drawing it
    bool                 morph       = false, skin = false;
    uint32_t             targetCount = 0, jointCount = 0;
    uint64_t             weightOffset = 0, jointOffset = 0, normalOffset = 0;  // in the packs
    // a skin with dual quaternions (MeshRecord::dualQuat): its joints' in the dual quaternion pack
    bool                 dualQuat = false;
    float                dualQuatShare = 0;
    uint64_t             dualQuatOffset = 0;
    // DAZ's order (MeshRecord::cageJoints): the skin poses the mesh's cage, and the refine chain
    // makes the outputs from it, with working streams of the drawer's own
    struct Posed
    {
      BufferHandle              cage, refinedNormals;
      std::vector<BufferHandle> levels;
    };
    std::optional<Posed> posed;
    // the buffers `posed` holds
    std::vector<BufferHandle> posedBuffers() const
    {
      std::vector<BufferHandle> out;
      if(posed)
      {
        out = posed->levels;
        out.push_back(posed->cage);
        out.push_back(posed->refinedNormals);
      }
      return out;
    }
    // the inputs it last ran with; empty: it runs at the next sync
    std::vector<float>     lastWeights;
    std::vector<glm::mat4> lastJoints;
    bool                   ran = false;
  };
  using DrawerKey = std::pair<EntityId, MeshId>;
  // A submesh drawn: its index buffer, BLAS and bounds
  struct Prim
  {
    MeshId       mesh;
    EntityId     drawer;  // the entity it is deformed for; none: drawn from the mesh's own streams
    uint32_t     submesh = 0;
    BufferHandle indices;
    uint32_t     indexCount = 0;
    AccelHandle  blas;
    BlasInput    input;
    uint32_t     refits = 0;  // since its BLAS was built
    uint32_t     users  = 0;  // instances drawing it
    // its BLAS was built for compaction (not refittable), and the compacted copy is still to come
    bool compactable = false, compacting = false;
    // the opacity micromap its BLAS links (input.micromap) and its micromap index buffer; none for none
    const OpacityMicromap* micromap = nullptr;
    BufferHandle           micromapIndices;
    glm::vec3    boundsMin{0.f}, boundsMax{0.f};  // of the vertices its indices name, in mesh space
    // a refined mesh's: the cage vertices its vertices are made from, and how far a vertex lies off
    // them (strand ribbons' half widths), for the tables they were made from
    std::vector<uint32_t>             cageSupport;
    float                             cagePad = 0;
    std::weak_ptr<const RefineTables> supportOf;
  };
  // The raster lists (SceneGpuView::DrawList)
  enum class DrawList
  {
    eSolid,
    eSolidDoubleSided,
    eBlend,
    eAll,
  };
  // An opacity micromap built, and the prims that link it
  struct MicromapGpu
  {
    std::shared_ptr<const OpacityMicromap> record;  // held while built: its address is the key
    AccelHandle                            accel;
    uint32_t                               users = 0;
  };
  // A draw item
  struct Instance
  {
    EntityId   entity;
    MeshId     mesh;
    uint32_t   submesh = 0;
    uint32_t   copy    = 0;
    uint32_t   prim    = 0;
    MaterialId material;
    uint32_t   materialSlot = 0;
    glm::mat4  world        = glm::mat4(1.f);
    bool       shown        = true;
    bool       deformed     = false;  // its prim is its drawer's
  };

  explicit Residency(GpuBackend& gpu, ResidencyOptions options = {});
  ~Residency();
  Residency(const Residency&)            = delete;
  Residency& operator=(const Residency&) = delete;

  // A scene replace: what was resident is retired, and `scene` made resident whole (its pending
  // changes are taken).
  SyncResult replace(RetainedScene& scene);
  // This frame's sync: the store's changes since the last one, brought to the GPU.
  SyncResult sync(RetainedScene& scene);
  // Everything retired (the backend frees it a frame cycle later); the pools start over. Decoded
  // images go to the grace list, where the next scene takes what it shares with this one.
  void clear();
  // textureGraceMB: a smaller budget trims the grace list now
  void setTextureGrace(uint64_t bytes);
  // blasRefitsBeforeRebuild (ResidencyOptions::refitsBeforeRebuild), from the next refit on
  void setRefitsBeforeRebuild(uint32_t refits) { m_options.refitsBeforeRebuild = refits; }
  // blasCompactStatic (ResidencyOptions::compactStatic), for the BLASes built from now on
  void setCompactStatic(bool compact) { m_options.compactStatic = compact; }
  // the images the last replace or sync decoded and the ones it took already decoded
  struct ImageCounts
  {
    uint32_t decoded = 0;
    uint32_t reused  = 0;
  };
  ImageCounts imageCounts() const { return m_imageCounts; }
  // The scene features the live shader variant has compiled in (the path tracer's optimal build);
  // none: every feature. A replace writes every material whatever these are.
  void                                            setShaderFeatures(std::optional<nvvkgltf::SceneFeatureSet> features);
  const std::optional<nvvkgltf::SceneFeatureSet>& shaderFeatures() const { return m_shaderFeatures; }
  // whether a material slot is drawn with a feature the shader features lack (what a replace wrote):
  // the variant then has to be compiled before the frame is traced
  bool drawsBeyondShaderFeatures() const;
  // the materials whose last edit is held
  const std::set<MaterialId>& heldMaterials() const { return m_heldMaterials; }
  // what the last syncs could not do as asked (a material drawn without its textures), once each
  std::vector<std::string> takeNotices() { return std::exchange(m_notices, {}); }
  uint64_t    graceBytes() const { return m_graceBytes; }

  // ---- what the renderers read
  uint64_t    sceneDesc() const;  // GltfScene's address
  AccelHandle tlas() const { return m_tlas; }
  uint32_t    instanceCapacity() const { return m_instanceCapacity; }
  uint32_t    instanceHighWater() const { return uint32_t(m_instances.size()); }
  uint32_t    primCapacity() const { return m_primCapacity; }
  // live instances by how their material draws, in slot order; hidden ones included (a draw skips them)
  const std::vector<uint32_t>& drawList(DrawList list) const { return m_drawLists[size_t(list)]; }
  uint64_t                     drawListsRevision() const { return m_drawListsRevision; }
  bool                         hasTransmission() const { return m_hasTransmission; }
  bool                         hasOpacityMicromaps() const { return m_micromapPrims > 0; }

  // ---- what the oracle reads
  const Instance* instance(uint32_t slot) const;  // nullptr for a hole
  const Prim*     prim(uint32_t slot) const;
  uint32_t        primHighWater() const { return uint32_t(m_prims.size()); }
  const MeshGpu*  meshGpu(MeshId mesh) const;
  const DrawerGpu* drawer(EntityId entity, MeshId mesh) const;
  const std::map<DrawerKey, DrawerGpu>& drawers() const { return m_drawers; }
  // the deform passes the last sync ran
  struct DeformCounts
  {
    uint32_t morphs = 0, skins = 0;
  };
  DeformCounts deformCounts() const { return m_deformCounts; }
  BufferHandle jointPack() const { return m_jointPack; }
  BufferHandle normalPack() const { return m_normalPack; }
  BufferHandle weightPack() const { return m_weightPack; }
  // DLSS motion: a device address per prim slot (0: none), and a world matrix per instance slot
  BufferHandle prevPositionTable() const { return m_prevPositionTable; }
  BufferHandle prevWorldTable() const { return m_prevWorldTable; }
  // the world matrix an instance slot was last drawn with (what its prevWorldTable row holds)
  const glm::mat4* prevWorld(uint32_t slot) const;
  const std::map<const OpacityMicromap*, MicromapGpu>& micromaps() const { return m_micromaps; }
  BufferHandle    renderNodeTable() const { return m_renderNodes; }
  BufferHandle    renderPrimTable() const { return m_renderPrims; }
  BufferHandle    sceneDescBuffer() const { return m_sceneDesc; }
  // the material slot a material has, -1 while nothing draws it
  int32_t      materialSlot(MaterialId material) const;
  uint32_t     materialHighWater() const { return m_materialHighWater; }
  BufferHandle materialTable() const { return m_materialTable; }
  BufferHandle lightTable() const { return m_lightTable; }
  uint32_t     lightCount() const { return m_lightCount; }
  BufferHandle emissiveTable() const { return m_emissiveTable; }
  BufferHandle emissiveGroupTable() const { return m_emissiveGroupTable; }
  uint32_t     emissiveGroupCount() const { return m_emissiveGroupCount; }  // the light tree's cut
  uint32_t     emissiveNodeCount() const { return m_emissiveNodeCount; }    // the light tree's nodes
  BufferHandle emissiveLeafTable() const { return m_emissiveLeafTable; }
  uint32_t     emissiveLeafCount() const { return m_emissiveLeafCount; }
  uint32_t     emissiveCount() const { return m_emissiveCount; }
  float        emissiveTotalWeight() const { return m_emissiveTotalWeight; }
  float        emissiveMeanLum() const { return m_emissiveMeanLum; }
  // EXT_lights_ies: every profile's samples, end to end, in the store's order
  BufferHandle iesTable() const { return m_iesTable; }
  uint32_t     iesProfileCount() const { return m_iesProfileCount; }
  // an emitter's vertices were written and the table still has the areas from before
  bool emissiveSettling() const { return m_emissiveStale; }
  // a material slot's converted row, the emissive table's radiance source
  const shaderio::GltfShadeMaterial* shadeMaterial(uint32_t slot) const;
  BufferHandle                       textureInfoTable() const { return m_textureInfoTable; }
  uint32_t                           textureInfoLimit() const { return m_options.textureInfoLimit; }
  // the texture slot of a texture sampled so, as colour (sRGB) or as data, -1 for none; the image a
  // texture is when decoded so, none while unread. A texture one material samples as colour and
  // another as data (a map used as both) is two images.
  int32_t     textureSlot(TextureId texture, const GpuSamplerDesc& sampler, bool srgb) const;
  ImageHandle image(TextureId texture, bool srgb) const;
  int32_t     samplerSlot(const GpuSamplerDesc& sampler) const;
  // what a material slot's row was converted from: its payload, and the slot table its texture
  // indices resolve through
  struct MaterialState
  {
    MaterialId                 material;
    tinygltf::Material         payload;
    nvvkgltf::TextureSlotTable slots;
    nvvkgltf::MaterialBits     bits;
    nvvkgltf::SceneFeatureSet  features;             // its payload's (detectMaterialFeatures)
    uint32_t                   span            = 0;  // its texture infos' first entry
    uint32_t                   spanCount       = 0;
    bool                       texturesDropped = false;  // no room for its span: drawn without them
  };
  const MaterialState* materialState(uint32_t slot) const;
  // what live resources this layer holds (meshes' streams, prims' index buffers and BLASes, the
  // micromaps and their prims' indices, the tables, the TLAS), to tell a leak from what the backend
  // still has to free
  size_t liveBuffers() const;
  size_t liveAccels() const;
  size_t liveImages() const;  // the grace list's included

private:
  using InstanceKey = std::tuple<EntityId, MeshId, uint32_t, uint32_t>;  // entity, mesh, submesh, copy

  struct MaterialSlot
  {
    uint32_t slot  = 0;
    uint32_t users = 0;
    uint32_t rev   = 0;
    // per entry of its texture table: what it holds (none for an entry the payload doesn't sample):
    // the texture, its sampler, and whether it is sampled as colour
    std::vector<std::optional<std::tuple<TextureId, GpuSamplerDesc, bool>>> textures;
  };
  using SamplerKey = std::tuple<int, int, int, int>;
  using ImageKey   = std::pair<TextureId, bool>;  // a texture, and whether it is decoded as sRGB
  struct ImageGpu
  {
    ImageHandle               image;
    ImageInfo                 info;
    uint32_t                  users = 0;
    std::optional<TextureKey> key;  // shared through m_decoded; none: its own image
  };
  // one decode per key, shared by the textures that are that image; one nothing uses is in grace
  struct DecodedImage
  {
    ImageHandle image;
    ImageInfo   info;
    uint32_t    users    = 0;
    uint64_t    released = 0;  // when it went idle, by m_releaseClock (the oldest is trimmed first)
  };
  struct SlotUse
  {
    uint32_t slot  = 0;
    uint32_t users = 0;
  };

  void syncStructure(RetainedScene& scene, std::vector<DrawItem>& items, SyncResult& result);
  void syncRows(const std::vector<DrawItem>& items, SyncResult& result);
  std::set<MeshId> syncVertices(RetainedScene& scene, const std::set<MeshId>& meshes, SyncResult& result);
  void             snapshotPrevPositions();
  void             syncDeforms(RetainedScene& scene, std::set<MeshId>& written, SyncResult& result);
  bool             meshInUse(MeshId id) const;
  void             layOutPacks(SyncResult& result);
  void             writeMotionTables(SyncResult& result);
  bool             deforms(const RetainedScene& scene, EntityId entity, MeshId mesh) const;
  DrawerGpu&       acquireDrawer(const RetainedScene& scene, EntityId entity, MeshId mesh);
  void             releaseDrawer(const DrawerKey& key);
  void             makeDeformInputs(const MeshRecord& record, MeshGpu& mesh, bool skin);

  MeshGpu& acquireMesh(const RetainedScene& scene, MeshId id);
  // a submesh's prim: the mesh's own, or `drawer`'s when the entity draws it deformed
  uint32_t acquirePrim(const RetainedScene& scene, MeshId id, uint32_t submesh, EntityId drawer = {});
  void     releasePrim(uint32_t slot);
  void     linkMicromap(Prim& prim, const MicromapUse& use);
  void     unlinkMicromap(Prim& prim);
  uint32_t acquireMaterial(const RetainedScene& scene, MaterialId id);
  void     releaseMaterial(MaterialId id);
  void     acquireMaterialTextures(const RetainedScene& scene, MaterialId id, MaterialSlot& slot);
  void     releaseMaterialTextures(MaterialSlot& slot);
  void     acquireTexture(const RetainedScene& scene, TextureId id, const GpuSamplerDesc& sampler, bool srgb);
  void     releaseTexture(TextureId id, const GpuSamplerDesc& sampler, bool srgb);
  void     releaseImage(const ImageGpu& image);
  void     decodeQueued();
  void     trimGrace(uint64_t budget);
  void     syncMaterials(RetainedScene& scene, const std::set<MaterialId>& edited, SyncResult& result);
  void     writeMaterials(const RetainedScene& scene);
  void     convertMaterial(const RetainedScene& scene, uint32_t slot);
  bool     placeSpan(uint32_t count, uint32_t& start);
  void     repackSpans();
  void     freeSpan(uint32_t start, uint32_t count);
  void     buildDrawLists();
  void     syncLights(RetainedScene& scene);
  void     syncIesProfiles(const RetainedScene& scene);
  void     syncEmissive(const RetainedScene& scene);
  bool     emits(const Instance& inst) const;
  void     retireMesh(MeshId id);
  void     retireBuffer(BufferHandle buffer);
  void     retireAccel(AccelHandle accel);
  // the prim's BLAS built, for compaction when it is static and blasCompactStatic is on
  void build(Prim& prim);
  // the prim's BLAS refitted to its input, or built again once it has been refitted enough (or
  // when it was built for compaction)
  void refit(Prim& prim);
  // the compacted copies of the BLASes built for compaction whose sizes are known; whether any came
  bool takeCompacted();
  // the record's streams; `vertices`: only what setMeshVertices writes (positions, normals, and the
  // tangents when it set them)
  void uploadStreams(const MeshRecord& record, MeshGpu& gpu, bool vertices = false);
  // a refined record's tables and working streams (MeshGpu::Refine), when its cage and drawn
  // vertices are the sizes its tables name
  void makeRefine(const MeshRecord& record, MeshGpu& gpu);
  // retires a refined mesh's tables and working streams
  void retireRefine(MeshGpu& gpu);
  // the refined meshes' positions and normals made from their uploaded cages: each level's stencils,
  // the welded normals, then each refined vertex copied to its drawn splits, then the tangents of
  // those with tangent tables; ends compute-to-build
  void refineOnGpu(std::span<MeshGpu* const> meshes);
  // One refine chain: a cage (a mesh's, or a drawer's posed one) through its mesh's tables into
  // drawn positions, normals and tangents, over working streams of its own
  struct RefineRun
  {
    const MeshGpu*            mesh = nullptr;
    BufferHandle              cage, refinedNormals;
    std::vector<BufferHandle> levels;  // each level's output
    BufferHandle              positions, normals, tangents;
  };
  // the chains' passes side by side, a barrier between stages; what wrote the cages and what reads
  // the outputs are the caller's barriers
  void refineChains(std::span<const RefineRun> runs);

  void ensureCapacity(SyncResult& result);
  void writeRenderNodes(const std::set<uint32_t>& slots);
  void writeRenderPrims(const std::set<uint32_t>& slots);
  void buildTlas();
  void writeSceneDesc();

  GpuBackend&      m_gpu;
  ResidencyOptions m_options;

  // shared with the retire closures that give slots back, which may outlive a clear()
  std::shared_ptr<nvutils::IDPool> m_instancePool, m_primPool, m_materialPool;

  std::map<MeshId, MeshGpu>            m_meshes;
  std::map<DrawerKey, DrawerGpu>       m_drawers;
  bool                                 m_packsDirty = false;  // drawers came or went: the packs are laid out again
  BufferHandle                         m_jointPack, m_normalPack, m_weightPack, m_dualQuatPack;
  DeformCounts                         m_deformCounts;
  // positions written by the last sync, copied to their previous positions at the top of the next
  std::set<DrawerKey>                  m_drawersWritten;
  std::set<MeshId>                     m_meshesWritten;
  BufferHandle                         m_prevPositionTable, m_prevWorldTable;
  std::vector<glm::mat4>               m_prevWorlds;          // by instance slot: what the table row holds
  std::set<uint32_t>                   m_dirtyPrevPrims;      // prim slots whose table row changes
  std::set<uint32_t>                   m_dirtyPrevNodes;      // instance slots whose row changes
  std::set<uint32_t>                   m_movedLast;           // instance slots whose world the last sync moved
  std::vector<std::optional<Prim>>     m_prims;      // by prim slot, up to the high-water mark
  std::map<const OpacityMicromap*, MicromapGpu> m_micromaps;
  std::optional<nvvkgltf::SceneFeatureSet>      m_shaderFeatures;
  std::set<MaterialId>                          m_heldMaterials;
  bool m_retryHeld = false;  // the shader features changed since the held edits were last tried
  uint32_t                                      m_micromapPrims = 0;  // prims that link one
  std::vector<std::optional<Instance>> m_instances;  // by instance slot, up to the high-water mark
  std::map<InstanceKey, uint32_t>      m_slotOf;
  std::map<MaterialId, MaterialSlot>   m_materials;

  std::map<ImageKey, ImageGpu>                               m_images;
  std::map<std::tuple<TextureId, bool, SamplerKey>, SlotUse> m_textureSlots;
  std::map<SamplerKey, SlotUse>                        m_samplerSlots;
  std::shared_ptr<nvutils::IDPool>                     m_texturePool, m_samplerPool;
  std::map<TextureKey, DecodedImage>                   m_decoded;  // across clear()
  // the images a sync's acquisitions need, decoded in one batch before the materials are converted
  struct QueuedDecode
  {
    ImageKey                  image;
    ImageSource               source;  // over the store's record, which outlives the sync
    std::optional<TextureKey> key;
  };
  std::vector<QueuedDecode>                            m_decodeQueue;
  uint64_t                                             m_graceBytes   = 0;
  uint64_t                                             m_releaseClock = 0;
  ImageCounts                                          m_imageCounts;
  std::vector<std::optional<MaterialState>>            m_materialStates;  // by material slot
  uint32_t                                             m_materialHighWater = 0;
  BufferHandle                                         m_materialTable, m_textureInfoTable;
  std::set<uint32_t>                                   m_dirtyMaterials;         // material slots to convert
  std::shared_ptr<nvutils::IDPool>                     m_textureInfoPool;        // spans; entry 0 held
  std::vector<shaderio::GltfTextureInfo>               m_textureInfos;           // the table's contents
  bool                                                 m_spansRepacked = false;  // this write
  std::vector<std::string>                             m_notices;
  bool                                                 m_bitsChanged = false;  // a material's TLAS bits or raster lists
  bool                                                 m_tablesMoved = false;  // the material tables were made again
  std::vector<shaderio::GltfShadeMaterial>             m_shadeMaterials;       // the material table's rows

  BufferHandle                         m_lightTable, m_emissiveTable, m_emissiveGroupTable, m_emissiveLeafTable, m_iesTable;
  BufferHandle                         m_openPbrLuts;  // the same for every scene, here for its lifetime
  uint32_t                             m_emissiveGroupCount = 0;  // the light tree's cut
  uint32_t                             m_emissiveNodeCount  = 0;  // its nodes, the cut included
  uint32_t                             m_emissiveLeafCount  = 0;  // render nodes the leaf lookup covers
  uint32_t                             m_loggedLightCount   = 0;  // the light count the last [Lights] line said
  uint32_t                             m_lightCount          = 0;
  uint32_t                             m_iesProfileCount     = 0;
  uint32_t                             m_emissiveCount       = 0;
  float                                m_emissiveTotalWeight = 0.f;
  float                                m_emissiveMeanLum     = 0.f;
  bool                                 m_descDirty           = false;  // a table moved or a count changed
  bool                                 m_emissiveStale       = false;  // see emissiveSettling()
  uint64_t                             m_emitterWriteFrame   = 0;      // the last frame that wrote one
  std::array<std::vector<uint32_t>, 4> m_drawLists;
  uint64_t                             m_drawListsRevision = 0;
  bool                                 m_hasTransmission   = false;

  uint32_t     m_instanceCapacity = 0;
  uint32_t     m_primCapacity     = 0;
  BufferHandle m_renderNodes, m_renderPrims, m_sceneDesc;
  AccelHandle  m_tlas;
  uint32_t     m_tlasCapacity = 0;

  std::set<uint32_t> m_dirtyNodes;        // render-node rows to write: rebound or now holes
  bool               m_allNodes = false;  // every row, into a new table
  bool               m_allPrims = false;
  std::set<uint32_t> m_compacting;  // prim slots whose BLASes wait for their compacted copies
  std::vector<MeshId> m_refineAcquired;            // refined meshes made resident: refined from their cages next
  bool               m_micromapsLeftOut = false;  // the device builds none: said once
};

}  // namespace rs
