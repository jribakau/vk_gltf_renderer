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

#include "fv_dforce_gpu.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <cmath>
#include <cstring>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/clusters.hpp>
#include <figure/physics/collide/sweep_grid.hpp>
#include <glm/glm.hpp>
#include <nvapp/application.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>

#include "shaders/dforce_cloth_io.h.slang"
#include "_autogen/dforce_cloth.comp.slang.h"

using namespace fv;
using namespace fv::dforce;

namespace {

// As the CPU solver's (figure/dforce/solver.cpp); kept the same so both drape alike.
constexpr float kCell             = 2.f;
constexpr float kDepth            = 2.f;
constexpr float kCap              = 2.f;
constexpr float kMaxSpeed         = 400.f;
constexpr int   kMinSubsteps      = 24, kMaxSubsteps = 48;
constexpr float kTransitionSpeed  = 60.f, kMinTransition = 0.25f, kMaxTransition = 0.75f;
constexpr float kHoldDamping      = 6.f;
constexpr int   kMinCollisionPasses = 4;
constexpr float kPassSpacing      = 0.5f * kDepth;
constexpr float kSearch           = 0.5f;
constexpr float kRecoverDepth     = 3.f;

// the queue simulations run on, when there is one of their own (DforceGpu::useQueue)
nvvk::QueueInfo s_queue{};

constexpr VkBufferUsageFlags2 kUsage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                       | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT;

static_assert(sizeof(fv::physics::SweepCell) == sizeof(shaderio::DforceCell) && offsetof(fv::physics::SweepCell, start) == offsetof(shaderio::DforceCell, start)
                  && offsetof(fv::physics::SweepCell, count) == offsetof(shaderio::DforceCell, count),
              "the grid's cells are the kernels'");

// How a mesh's kinematic vertices move: a sweep of its bones from one pose to another.
struct Kinematic
{
  bool                  skinned = false;
  std::vector<BonePose> from, to;
};

// What a simulation needs of the device whatever its model: the pipeline layout, the kernels, and the
// command pool of its queue when it has one of its own. Made at its first start and kept through its
// restarts, so a model of other shapes (a dial turned) doesn't make them again.
struct Kernels
{
  VkDevice         device      = VK_NULL_HANDLE;
  VkCommandPool    commandPool = VK_NULL_HANDLE;
  VkPipelineLayout layout      = VK_NULL_HANDLE;
  VkPipeline       integrate{}, edges{}, attach{}, recover{}, hold{}, collide{}, velocity{}, cluster{}, commit{}, snap{}, boxes{}, collideWave{},
      recoverWave{};
  // the lanes of the device's waves, when the collision pass and the recovery go a wave a vertex
  // (collideWaveMain, recoverWaveMain: 32, with the ballots, shuffles and least they need); 0: a thread a vertex
  uint32_t waveLanes = 0;

  Kernels() = default;
  Kernels(const Kernels&)            = delete;
  Kernels& operator=(const Kernels&) = delete;
  ~Kernels()
  {
    if(!device)
      return;
    for(VkPipeline p : {integrate, edges, attach, recover, hold, collide, velocity, cluster, commit, snap, boxes, collideWave, recoverWave})
      if(p)
        vkDestroyPipeline(device, p, nullptr);
    if(layout)
      vkDestroyPipelineLayout(device, layout, nullptr);
    if(commandPool)
      vkDestroyCommandPool(device, commandPool, nullptr);
  }

  bool make(VkDevice dev, VkPhysicalDevice physical, const nvvk::QueueInfo& queue)
  {
    device = dev;
    {
      VkPhysicalDeviceSubgroupProperties subgroup{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
      VkPhysicalDeviceProperties2        props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
      vkGetPhysicalDeviceProperties2(physical, &props);
      const VkSubgroupFeatureFlags needs = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT
                                           | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
      if(subgroup.subgroupSize == 32 && (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (subgroup.supportedOperations & needs) == needs)
        waveLanes = 32;
    }
    if(queue.queue)
    {
      commandPool = nvvk::createTransientCommandPool(device, queue.familyIndex);
      if(!commandPool)
        return false;
    }
    VkPushConstantRange        pushRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::DforcePushConstant)};
    VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
    if(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS)
      return false;
    NVVK_DBG_NAME(layout);
    const auto pipe = [&](const char* entry, VkPipeline& out) {
      VkShaderModuleCreateInfo    shaderInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                             .codeSize = dforce_cloth_comp_slang_sizeInBytes,
                                             .pCode    = dforce_cloth_comp_slang};
      VkComputePipelineCreateInfo pipeInfo{.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                           .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      .pNext = &shaderInfo,
                                                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                      .pName = entry},
                                           .layout = layout};
      if(vkCreateComputePipelines(device, nullptr, 1, &pipeInfo, nullptr, &out) != VK_SUCCESS)
      {
        LOGE("dForce GPU: could not create the %s pipeline\n", entry);
        return false;
      }
      NVVK_DBG_NAME(out);
      return true;
    };
    return pipe("integrateMain", integrate) && pipe("edgesMain", edges) && pipe("attachMain", attach) && pipe("recoverMain", recover)
           && pipe("holdMain", hold) && pipe("collideMain", collide) && pipe("velocityMain", velocity) && pipe("clusterMain", cluster)
           && pipe("commitMain", commit) && pipe("snapMain", snap) && pipe("boxesMain", boxes)
           && (waveLanes == 0 || (pipe("collideWaveMain", collideWave) && pipe("recoverWaveMain", recoverWave)));
  }
};

double msSince(std::chrono::steady_clock::time_point t)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

// A frame in which the pose moves anything further than this, cm, goes in pieces, at most so many.
constexpr float kJumpPerPiece = 2.f;
constexpr int   kMostPieces   = 12;

struct DforceGpu::Impl
{
  nvvk::ResourceAllocator* alloc = nullptr;
  nvapp::Application*      app   = nullptr;
  std::vector<SettleMesh>  meshes;
  SettleOptions            opts;
  DforceGpu::Options       gpu;
  // a queue of its own and the pool its commands come from, or none: the renderer's, through `app`
  nvvk::QueueInfo          queue{};
  VkCommandPool            commandPool = VK_NULL_HANDLE;
  // the kernels and the command pool, kept through restarts; the handles below are theirs
  std::shared_ptr<Kernels> kernels;
  ClothModel               model;
  std::vector<Kinematic>   kin;
  // the cluster solve's partition, when it runs (gpu.clusterSize > 0 and the model fits it)
  fv::physics::cloth::Clusters        clusters;
  std::vector<shaderio::DforceEdge>   clusterRows;
  std::vector<glm::uvec2>             clusterSoft;
  DforceGpu::Stats                    stats;
  VkQueryPool                         queries   = VK_NULL_HANDLE;
  double                              msPerTick = 0;
  // the timestamps a frame wrote after its first two, in pairs around a phase, and which phase each is
  static constexpr uint32_t kTimedPairs = 256;
  std::vector<int>                    timedPhase;

  // edges sorted by colour, and where each colour starts; the stretch edges have the first colours
  // to themselves, so a sweep over them alone is those colours
  std::vector<int> colourStart;
  int              stretchColours = 0;
  int                lengthColours = 0;
  // as the buffers take them, from prepare() until create()
  std::vector<shaderio::DforceEdge> sorted;
  std::vector<shaderio::DforceTri>  faces, clothFaces;

  VkPipelineLayout layout{};
  VkPipeline       integratePipe{}, edgesPipe{}, attachPipe{}, recoverPipe{}, holdPipe{}, collidePipe{}, velocityPipe{}, clusterPipe{}, commitPipe{},
      snapPipe{}, boxesPipe{}, collideWavePipe{}, recoverWavePipe{};
  // the kernels' waveLanes, unless FV_DFORCE_GPU_DEBUG's 256 asks for a thread a vertex
  uint32_t waveLanes = 0;

  // the static faces' corners, three a face, and the grid made from them each frame (its buffers kept)
  std::vector<int32_t>     faceCorners;
  fv::physics::SweepGrid   grid;
  // the cloth's own faces, when one can meet a vertex (another garment's, or its own with Self Collide):
  // their corners, and their grid made each frame from where they were at its start
  bool                     clothOn = false;
  std::vector<int32_t>     clothCorners;
  fv::physics::SweepGrid   clothGrid;
  // the model's buffers; `cells` and `items` grow with the grid
  nvvk::Buffer x, prev, vel, cprev, kinStart, kinEnd, kinematic, freeVerts, inv, dampRate, offset, friction, layer, collides, grav, edges,
      lam, tris, cells, items, softVerts, attAlpha, lamAtt, contact, modelBuf, readback, clusterEdgeBuf, clusterLamBuf, clusterVertBuf, clusterVertStartBuf,
      clusterOwnedBuf, clusterColourBuf, clusterSoftBuf, clusterSoftStartBuf, xTwin, clothTris, clothCells, clothItems, meshOf, ring2Start, ring2,
      frameStart, snap, cprevSnap, faceBox, clothContact;
  size_t cellCapacity = 0, itemCapacity = 0, clothCellCapacity = 0, clothItemCapacity = 0;
  bool   gridFailed = false;

  // host copies: every position (the kinematic ones kept where the last frame ended them)
  std::vector<float> xHost;
  // the start (the rest shape, separated from the body), for a restart; and a restart whose state is
  // still to go back to it, on the owner's thread, and whose start is to be made again first (a
  // collider's rest cage changed: a body's contact dents)
  std::vector<float> xStart;
  bool               resetPending = false, startPending = false;
  int                substeps   = kMinSubsteps;
  int                collisions = kMinCollisionPasses;
  float              transition = kMinTransition;
  float              lastMove   = 0;
  // the pose a frame that held still was last given (negative: none): the next at the same place needs
  // neither the skin nor the collision grid again
  float                 heldAt = -1;
  // the collision passes so far: every fourth looks around for cloth faces (clothLook), as the CPU island does
  int                   passCount = 0;
  // live: simulated time a step was given that its whole substeps didn't take, carried to the next
  float                 liveCarry = 0;
  shaderio::DforceModel shaderModel{};
  // a settle under way, and what it has cost
  int    settled    = 0;
  double settleWork = 0, settleWait = 0;
  std::chrono::steady_clock::time_point settleStart;
  // a frame that came back with a position that is not a number: the frame is dropped and the
  // simulation stops, so one bad step cannot take the device down with it
  bool broken = false;
  // FV_DFORCE_GPU_DEBUG, bits: 1 leaves the recovery out, 2 the hold between passes; for finding
  // which kernel a device loss comes from. 256: the collision passes and the recovery a thread a vertex;
  // 512: a frame that takes over 250 ms stops the simulation
  int debug = 0;

  // ---- setup
  bool prepare();      // any thread: touches nothing but the model
  void posePrepare();  // prepare()'s parts a restart makes again: the bones' sweep, the transition
  bool create();       // the owner's thread
  void resetState();   // the owner's thread, for a restart: the state back to the start
  void destroy();
  template <class T>
  bool mapped(nvvk::Buffer& buf, const std::vector<T>& data, VmaMemoryUsage usage = VMA_MEMORY_USAGE_CPU_TO_GPU)
  {
    return mappedBytes(buf, data.data(), data.size() * sizeof(T), usage);
  }
  bool mappedBytes(nvvk::Buffer& buf, const void* data, size_t bytes, VmaMemoryUsage usage);

  // ---- a frame
  void kinematicAt(float t, std::vector<float>& out);
  bool buildGrid(const std::vector<float>& start, const std::vector<float>& end, shaderio::DforceModel& m);
  // a grid into its buffers, made larger when it outgrew them; false when they could not be made
  bool uploadGrid(const fv::physics::SweepGrid& g, nvvk::Buffer& cellBuf, size_t& cellCap, nvvk::Buffer& itemBuf, size_t& itemCap);
  // the cloth's faces where they are at the frame's start (xHost), into their grid and `m`
  bool buildClothGrid(shaderio::DforceModel& m);
  // `steps` substeps in `seconds`; 0, 0: a frame of opts.frameDt in `substeps`
  void frame(float t0, float t1, float extraDamping, float seconds = 0, int steps = 0);
  // the cloth carried with the pose to fraction t of the sweep (DforceGpu::carry)
  void carry(float t);
};

bool DforceGpu::Impl::mappedBytes(nvvk::Buffer& buf, const void* data, size_t bytes, VmaMemoryUsage usage)
{
  if(buf.buffer != VK_NULL_HANDLE)
    alloc->destroyBuffer(buf);
  buf = {};
  const size_t size = std::max<size_t>(16, bytes);
  const VkResult r = alloc->createBuffer(buf, size, kUsage, usage,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT
                                             | (usage == VMA_MEMORY_USAGE_GPU_TO_CPU ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT :
                                                                                       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
  if(r != VK_SUCCESS || !buf.mapping)
  {
    LOGE("dForce GPU: could not create a %zu byte buffer\n", size);
    return false;
  }
  NVVK_DBG_NAME(buf.buffer);
  if(data && bytes)
    std::memcpy(buf.mapping, data, bytes);
  return true;
}

bool DforceGpu::Impl::prepare()
{
  model = ClothModel::build(meshes, opts.air, opts.reachLinks);
  if(model.nv == 0 || model.freeVerts.empty())
    return false;
#pragma warning(suppress : 4996)  // getenv: read once, at start-up
  if(const char* env = std::getenv("FV_DFORCE_GPU_DEBUG"); env && *env)
    debug = std::atoi(env);

  // the cluster solve: the clusters' rows run in shared memory, the rows that cross them coloured as below
  if(gpu.clusterSize > 0)
  {
    clusters = fv::physics::cloth::buildClusters(model, gpu.clusterSize, std::min(gpu.clusterLocal, DFORCE_CLUSTER_LOCAL),
                                                 gpu.crossing == DforceGpu::Options::Crossing::eHalo);
    if(clusters.count() == 0 || clusters.mostLocal > DFORCE_CLUSTER_LOCAL || clusters.mostLocal > 0xFFFF)
    {
      LOGW("dForce GPU: the cluster solve can't take this model (%d vertices in a cluster at most); a dispatch a colour instead\n", clusters.mostLocal);
      clusters = {};
    }
  }
  if(clusters.count() > 0)
  {
    for(const auto& row : clusters.rows)
    {
      const Edge& ed = model.edges[size_t(row.edge)];
      clusterRows.push_back({int(row.a), int(row.b), ed.rest, ed.alpha, int(ed.kind), ed.ratio, ed.buckle, 0.f});
    }
    for(const auto& [l, s] : clusters.soft)
      clusterSoft.push_back({uint32_t(l), uint32_t(s)});
    // the rows that cross clusters, coloured (none with a halo: the clusters hold them)
    colourStart.assign(clusters.crossingStart.begin(), clusters.crossingStart.end());
    for(int e : clusters.crossing)
    {
      const Edge& ed = model.edges[size_t(e)];
      sorted.push_back({ed.a, ed.b, ed.rest, ed.alpha, int(ed.kind), ed.ratio, ed.buckle, 0.f});
    }
    stretchColours = clusters.crossingStretch;
    lengthColours  = clusters.crossingStretch + clusters.crossingReach;
    stats.clusters        = clusters.count();
    stats.clusterColours  = clusters.colours();
    stats.crossingColours = int(colourStart.size()) - 1;
    stats.crossingRows    = clusters.crossing.size();
    stats.haloRows        = clusters.haloRows;
    stats.mostLocal       = clusters.mostLocal;
    stats.rows            = clusters.rows.size() + clusters.crossing.size();
  }
  // greedy colouring: no two edges of a colour share a vertex, so a colour projects in parallel.
  // The stretch edges are coloured first, among themselves, so the sweeps over them alone (as the
  // CPU solver's stretchSweeps) are the first colours
  std::vector<int> colourOf(model.edges.size(), -1);
  int              colours = 0;
  const auto       colourKind = [&](const auto& wanted) {
    std::vector<uint64_t> used(size_t(model.nv), 0);
    for(size_t e = 0; e < model.edges.size(); e++)
    {
      const Edge& ed = model.edges[e];
      if(!wanted(ed.kind) || (model.inv[ed.a] <= 0 && model.inv[ed.b] <= 0))
        continue;
      // a pinned end is only read: edges that share one don't touch
      const bool     freeA = model.inv[ed.a] > 0, freeB = model.inv[ed.b] > 0;
      const uint64_t taken = (freeA ? used[size_t(ed.a)] : 0) | (freeB ? used[size_t(ed.b)] : 0);
      int            c     = 0;
      while(c < 63 && (taken >> c) & 1)
        c++;
      if(c >= 63)
      {
        LOGE("dForce GPU: a vertex has more than 63 edges of one kind; the CPU solver takes this one\n");
        return false;
      }
      colourOf[e] = colours + c;
      if(freeA)
        used[size_t(ed.a)] |= uint64_t(1) << c;
      if(freeB)
        used[size_t(ed.b)] |= uint64_t(1) << c;
    }
    int most = 0;
    for(size_t e = 0; e < model.edges.size(); e++)
      if(colourOf[e] >= colours)
        most = std::max(most, colourOf[e] - colours + 1);
    colours += most;
    return true;
  };
  // stretch, reach links, the rest: a stretch sweep takes the first two, a full sweep all but the links
  if(clusters.count() == 0)
  {
    if(!colourKind([](EdgeKind k) { return k == EdgeKind::eStretch; }))
      return false;
    stretchColours = colours;
    if(!colourKind([](EdgeKind k) { return k == EdgeKind::eReach; }))
      return false;
    lengthColours = colours;
    if(!colourKind([](EdgeKind k) { return !holdsLength(k); }))
      return false;
    colourStart.assign(size_t(colours) + 1, 0);
    for(int c = 0; c < colours; c++)
    {
      colourStart[size_t(c)] = int(sorted.size());
      for(size_t e = 0; e < model.edges.size(); e++)
        if(colourOf[e] == c)
        {
          const Edge& ed = model.edges[e];
          sorted.push_back({ed.a, ed.b, ed.rest, ed.alpha, int(ed.kind), ed.ratio, ed.buckle, 0.f});
        }
    }
    colourStart[size_t(colours)] = int(sorted.size());
    stats.rows = sorted.size();
  }

  for(int t : model.staticTris)
  {
    const Tri& tri = model.tris[size_t(t)];
    faces.push_back({tri.a, tri.b, tri.c, tri.layer, tri.offset, tri.friction, tri.mesh, tri.self ? 1 : 0});
    faceCorners.insert(faceCorners.end(), {tri.a, tri.b, tri.c});
  }
  // The cloth's faces, when any can meet a vertex of another garment (a petticoat under a skirt), or of
  // its own garment where that collides with itself (Self Collide): met as the CPU island meets them.
  // FV_DFORCE_GPU_DEBUG's 64 leaves them out, 128 a garment's own
  {
    const bool           own = (debug & 128) == 0;
    std::vector<uint8_t> movingMesh(meshes.size(), 0);
    for(int i : model.freeVerts)
      if(model.collides[size_t(i)])
        movingMesh[size_t(model.meshOf[size_t(i)])] = 1;
    bool canMeet = false;
    for(int t : model.dynamicTris)
    {
      const Tri& tri = model.tris[size_t(t)];
      for(size_t mi = 0; mi < meshes.size() && !canMeet; mi++)
        canMeet = movingMesh[mi] && (int(mi) != tri.mesh || (own && tri.self));
      if(canMeet)
        break;
    }
    clothOn = canMeet && gpu.clothCollide && !(debug & 64);
    if(clothOn)
      for(int t : model.dynamicTris)
      {
        const Tri& tri = model.tris[size_t(t)];
        clothFaces.push_back({tri.a, tri.b, tri.c, tri.layer, tri.offset, tri.friction, tri.mesh, own && tri.self ? 1 : 0});
        clothCorners.insert(clothCorners.end(), {tri.a, tri.b, tri.c});
      }
    stats.clothFaces = clothFaces.size();
  }

  // the start: the rest shape, with what overlaps the body moved out of it, as the CPU solver starts
  xHost = model.rest;
  separateStart(model, xHost);
  xStart = xHost;
  posePrepare();
  return true;
}

void DforceGpu::Impl::posePrepare()
{
  // the pose's hold on each mesh, swept from the memorized pose to the target it came with
  kin.assign(meshes.size(), Kinematic{});
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    kin[mi].skinned = meshes[mi].skinned();
    if(kin[mi].skinned)
    {
      for(const Mat4& b : meshes[mi].bones)
        kin[mi].to.push_back(poseOf(b));
      kin[mi].from.assign(meshes[mi].bones.size(), poseOf(Mat4{}));
    }
  }
  // the transition's length and the frame's substeps, from how far the pose moves what it holds
  float maxMove = 0;
  for(int i : model.kinematicVerts)
  {
    const float dx = model.posed[i * 3] - model.rest[i * 3], dy = model.posed[i * 3 + 1] - model.rest[i * 3 + 1],
                dz = model.posed[i * 3 + 2] - model.rest[i * 3 + 2];
    maxMove = std::max(maxMove, std::sqrt(dx * dx + dy * dy + dz * dz));
  }
  transition           = opts.transition > 0 ? opts.transition : std::min(kMaxTransition, std::max(kMinTransition, maxMove / kTransitionSpeed));
  const int frames     = std::max(1, int(std::lround(transition / opts.frameDt)));
  const int byMotion   = int(std::ceil(maxMove / float(frames) / kPassSpacing));
  substeps             = opts.substeps > 0 ? std::min(240, opts.substeps) : std::min(kMaxSubsteps, std::max(kMinSubsteps, byMotion));
  collisions           = std::min(substeps, std::max(kMinCollisionPasses, byMotion));
}

bool DforceGpu::Impl::create()
{
  fv::timing::Scope  t("dforce: buffers and kernels");
  const size_t       n = size_t(model.nv);
  std::vector<float> zeros(n * 3, 0.f);
  std::vector<float> lamZero(std::max<size_t>(1, sorted.size()), 0.f);
  std::vector<float> lamAttZero(std::max<size_t>(1, model.softVerts.size()), 0.f);
  std::vector<int>   collideFlags(n), layers(n), noContact(n, -1);
  for(size_t i = 0; i < n; i++)
  {
    collideFlags[i] = model.collides[i] ? 1 : 0;
    layers[i]       = model.layer[i];
  }
  bool ok = true;
  ok &= mapped(x, xHost);
  ok &= mapped(prev, xHost);
  ok &= mapped(vel, zeros);
  ok &= mapped(cprev, xHost);
  ok &= mapped(kinStart, xHost);
  ok &= mapped(kinEnd, xHost);
  ok &= mapped(kinematic, model.kinematicVerts);
  ok &= mapped(freeVerts, model.freeVerts);
  ok &= mapped(inv, model.inv);
  ok &= mapped(dampRate, model.dampRate);
  ok &= mapped(offset, model.offset);
  ok &= mapped(friction, model.friction);
  ok &= mapped(layer, layers);
  ok &= mapped(collides, collideFlags);
  ok &= mapped(grav, model.grav);
  ok &= mapped(edges, sorted);
  ok &= mapped(lam, lamZero);
  ok &= mapped(tris, faces);
  // written and read by the kernels alone, so on the device
  if(ok && !faces.empty())
  {
    ok = alloc->createBuffer(faceBox, faces.size() * 2 * sizeof(glm::vec4), kUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) == VK_SUCCESS;
    if(ok)
      NVVK_DBG_NAME(faceBox.buffer);
  }
  ok &= mapped(softVerts, model.softVerts.empty() ? std::vector<int>{0} : model.softVerts);
  ok &= mapped(attAlpha, model.attAlpha);
  ok &= mapped(lamAtt, lamAttZero);
  ok &= mapped(contact, noContact);
  if(clothOn)
    ok &= mapped(clothContact, noContact);
  if(clusters.count() > 0)
  {
    ok &= mapped(clusterEdgeBuf, clusterRows);
    ok &= mapped(clusterLamBuf, std::vector<float>(std::max<size_t>(1, clusterRows.size()), 0.f));
    ok &= mapped(clusterVertBuf, clusters.verts);
    ok &= mapped(clusterVertStartBuf, clusters.vertStart);
    ok &= mapped(clusterOwnedBuf, clusters.owned);
    ok &= mapped(clusterColourBuf, clusters.colourStart);
    ok &= mapped(clusterSoftBuf, clusterSoft.empty() ? std::vector<glm::uvec2>{glm::uvec2(0)} : clusterSoft);
    ok &= mapped(clusterSoftStartBuf, clusters.softStart);
    ok &= mapped(xTwin, xHost);
  }
  if(clothOn)
  {
    ok &= mapped(clothTris, clothFaces);
    ok &= mapped(meshOf, model.meshOf);
    ok &= mapped(ring2Start, model.ring2Start);
    ok &= mapped(ring2, model.ring2.empty() ? std::vector<int>{-1} : model.ring2);
    ok &= mapped(frameStart, xHost);
    ok &= mapped(snap, xHost);
    ok &= mapped(cprevSnap, xHost);
  }
  ok &= mappedBytes(modelBuf, nullptr, sizeof(shaderio::DforceModel), VMA_MEMORY_USAGE_CPU_TO_GPU);
  ok &= mappedBytes(readback, nullptr, n * 3 * sizeof(float), VMA_MEMORY_USAGE_GPU_TO_CPU);
  if(!ok)
    return false;

  if(!kernels)
  {
    fv::timing::Scope tk("dforce: kernels");
    auto              k = std::make_shared<Kernels>();
    if(!k->make(alloc->getDevice(), alloc->getPhysicalDevice(), queue))
      return false;
    kernels = std::move(k);
  }
  commandPool   = kernels->commandPool;
  layout        = kernels->layout;
  integratePipe = kernels->integrate;
  edgesPipe     = kernels->edges;
  attachPipe    = kernels->attach;
  recoverPipe   = kernels->recover;
  holdPipe      = kernels->hold;
  collidePipe   = kernels->collide;
  velocityPipe  = kernels->velocity;
  clusterPipe   = kernels->cluster;
  commitPipe    = kernels->commit;
  snapPipe      = kernels->snap;
  boxesPipe     = kernels->boxes;
  collideWavePipe = kernels->collideWave;
  recoverWavePipe = kernels->recoverWave;
  waveLanes       = (debug & 256) ? 0 : kernels->waveLanes;
  if(gpu.timestamps)
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(alloc->getPhysicalDevice(), &props);
    msPerTick = double(props.limits.timestampPeriod) * 1e-6;
    const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 + 2 * kTimedPairs};
    if(vkCreateQueryPool(alloc->getDevice(), &info, nullptr, &queries) != VK_SUCCESS)
      queries = VK_NULL_HANDLE;
  }
  LOGI("dForce GPU: %d vertices (%zu free, %zu soft), %zu edges in %d colours (%d stretch), %zu collision faces\n", model.nv,
       model.freeVerts.size(), model.softVerts.size(), sorted.size(), int(colourStart.size()) - 1, stretchColours, faces.size());
  LOGI("dForce GPU: %s, on %s\n", clusters.count() ? "the cluster solve" : "a dispatch a colour",
       queue.queue ? "a queue of its own (settles on the pool)" : "the renderer's queue (settles on its thread)");
  if(clusters.count())
    LOGI("dForce GPU: %d clusters of up to %d vertices, %d colours each, %zu rows held by two\n", clusters.count(), clusters.mostLocal,
         clusters.colours(), clusters.haloRows);
  if(clothOn)
    LOGI("dForce GPU: the cloth meets %zu cloth faces\n", clothFaces.size());
  sorted     = {};
  faces      = {};
  clothFaces = {};
  return true;
}

void DforceGpu::Impl::resetState()
{
  if(startPending)
  {
    // as prepare() makes it: the rest shape, with what overlaps the body moved out of it
    xStart = model.rest;
    separateStart(model, xStart);
    startPending = false;
  }
  resetPending = false;
  xHost        = xStart;
  // nothing is in flight between settles: the buffers are written as they lie
  const size_t bytes = xStart.size() * sizeof(float);
  for(nvvk::Buffer* b : {&x, &prev, &cprev, &kinStart, &kinEnd, &xTwin})
    if(b->mapping)
      std::memcpy(b->mapping, xStart.data(), bytes);
  for(nvvk::Buffer* b : {&vel, &lam, &lamAtt, &clusterLamBuf})
    if(b->mapping)
      std::memset(b->mapping, 0, size_t(b->bufferSize));
  // -1 everywhere: no face under any vertex
  if(contact.mapping)
    std::memset(contact.mapping, 0xFF, size_t(contact.bufferSize));
  if(clothContact.mapping)
    std::memset(clothContact.mapping, 0xFF, size_t(clothContact.bufferSize));
  passCount = 0;
  std::memcpy(grav.mapping, model.grav.data(), model.grav.size() * sizeof(float));
  heldAt    = -1;
  lastMove  = 0;
  settled   = 0;
  liveCarry = 0;
  // the run's counts start over; the cut's shape stays
  stats.frames     = 0;
  stats.dispatches = 0;
  stats.gpuMs = stats.waitMs = stats.collideMs = stats.recoverMs = stats.holdMs = stats.skinMs = stats.gridMs = stats.gridCopyMs = 0;
  stats.gridEntries = stats.gridSlots = 0;
}

void DforceGpu::Impl::destroy()
{
  if(!alloc)
    return;
  VkDevice device = alloc->getDevice();
  if(queries)
  {
    vkDestroyQueryPool(device, queries, nullptr);
    queries = VK_NULL_HANDLE;
  }
  // the kernels' handles are theirs: they go with the last simulation that holds them
  commandPool = VK_NULL_HANDLE;
  layout      = VK_NULL_HANDLE;
  for(VkPipeline* p : {&integratePipe, &edgesPipe, &attachPipe, &recoverPipe, &holdPipe, &collidePipe, &velocityPipe, &clusterPipe, &commitPipe, &snapPipe,
                       &boxesPipe, &collideWavePipe, &recoverWavePipe})
    *p = VK_NULL_HANDLE;
  kernels.reset();
  for(nvvk::Buffer* b : {&x, &prev, &vel, &cprev, &kinStart, &kinEnd, &kinematic, &freeVerts, &inv, &dampRate, &offset, &friction, &layer,
                         &collides, &grav, &edges, &lam, &tris, &cells, &items, &softVerts, &attAlpha, &lamAtt, &contact, &modelBuf, &readback,
                         &clusterEdgeBuf, &clusterLamBuf, &clusterVertBuf, &clusterVertStartBuf, &clusterOwnedBuf, &clusterColourBuf, &clusterSoftBuf,
                         &clusterSoftStartBuf, &xTwin, &clothTris, &clothCells, &clothItems, &meshOf, &ring2Start, &ring2, &frameStart, &snap,
                         &cprevSnap, &faceBox, &clothContact})
    if(b->buffer != VK_NULL_HANDLE)
      alloc->destroyBuffer(*b);
}

// Writes the pose at fraction t of the kinematic vertices, and of the soft ones (their tethers'
// anchors), into `out` (only those entries).
void DforceGpu::Impl::kinematicAt(float t, std::vector<float>& out)
{
  std::vector<float> tmp;
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const SettleMesh& m = meshes[mi];
    const int         o = model.base[mi];
    const size_t      n = m.vertexCount();
    if(kin[mi].skinned)
    {
      std::vector<Mat4> bones(kin[mi].to.size());
      for(size_t b = 0; b < bones.size(); b++)
        bones[b] = sweepAt(kin[mi].from[b], kin[mi].to[b], t);
      tmp.resize(n * 3);
      skinMesh(m, bones, tmp.data());
    }
    for(size_t v = 0; v < n; v++)
    {
      const int i = o + int(v);
      if(model.dynamic[i] && model.inv[i] > 0 && model.attAlpha[size_t(i)] <= 0)
        continue;
      for(int c = 0; c < 3; c++)
        out[size_t(i) * 3 + c] = kin[mi].skinned ? tmp[v * 3 + c] : (1 - t) * model.rest[size_t(i) * 3 + c] + t * model.posed[size_t(i) * 3 + c];
    }
  }
}

// The body's faces over their sweep through the frame, into the open-addressed cell table
// (figure/physics/collide/sweep_grid.hpp). False when the table's buffers could not be made: the frame
// then runs without collisions.
bool DforceGpu::Impl::buildGrid(const std::vector<float>& start, const std::vector<float>& end, shaderio::DforceModel& m)
{
  fv::physics::buildSweepGrid(faceCorners, start.data(), end.data(), kCell, grid);
  stats.gridEntries += grid.entries;
  stats.gridSlots += grid.cells.size();
  const auto copyStart = std::chrono::steady_clock::now();
  const bool ok        = uploadGrid(grid, cells, cellCapacity, items, itemCapacity);
  stats.gridCopyMs += msSince(copyStart);
  if(!ok)
    return false;
  m.cells    = reinterpret_cast<shaderio::DforceCell*>(cells.address);
  m.items    = reinterpret_cast<int*>(items.address);
  m.cellMask = grid.mask();
  return true;
}

bool DforceGpu::Impl::uploadGrid(const fv::physics::SweepGrid& g, nvvk::Buffer& cellBuf, size_t& cellCap, nvvk::Buffer& itemBuf, size_t& itemCap)
{
  bool ok = true;
  if(g.cells.size() > cellCap)
  {
    ok      = mapped(cellBuf, g.cells);
    cellCap = ok ? g.cells.size() : 0;
  }
  else
    std::memcpy(cellBuf.mapping, g.cells.data(), g.cells.size() * sizeof(fv::physics::SweepCell));
  if(ok && g.items.size() > itemCap)
  {
    ok      = mapped(itemBuf, g.items);
    itemCap = ok ? g.items.size() : 0;
  }
  else if(ok)
    std::memcpy(itemBuf.mapping, g.items.data(), g.items.size() * sizeof(int32_t));
  return ok;
}

bool DforceGpu::Impl::buildClothGrid(shaderio::DforceModel& m)
{
  const auto start = std::chrono::steady_clock::now();
  std::memcpy(frameStart.mapping, xHost.data(), xHost.size() * sizeof(float));
  fv::physics::buildSweepGrid(clothCorners, xHost.data(), xHost.data(), kCell, clothGrid);
  const bool ok = uploadGrid(clothGrid, clothCells, clothCellCapacity, clothItems, clothItemCapacity);
  m.clothCells    = reinterpret_cast<shaderio::DforceCell*>(clothCells.address);
  m.clothItems    = reinterpret_cast<int*>(clothItems.address);
  m.clothCellMask = clothGrid.mask();
  m.numClothTris  = ok ? uint32_t(clothCorners.size() / 3) : 0;
  stats.clothGridMs += msSince(start);
  fv::timing::accumulate("dforceGpu: a frame's cloth grid on the host", msSince(start));
  return ok;
}

void DforceGpu::Impl::frame(float t0, float t1, float extraDamping, float seconds, int stepCount)
{
  const size_t n = size_t(model.nv);
  shaderio::DforceModel& m             = shaderModel;
  float                  kinematicMove = 0;
  const bool             held          = t0 == t1 && t0 == heldAt;
  heldAt                               = t0 == t1 ? t0 : -1.f;
  if(!held)
  {
    // where the pose takes what it holds by the end of the frame
    std::vector<float> start = xHost, end = xHost;
    const auto         skinStart = std::chrono::steady_clock::now();
    kinematicAt(t0, start);
    kinematicAt(t1, end);
    stats.skinMs += msSince(skinStart);
    fv::timing::accumulate("dforceGpu: a frame's skin on the host", msSince(skinStart));
    for(int i : model.kinematicVerts)
    {
      const float dx = end[size_t(i) * 3] - start[size_t(i) * 3], dy = end[size_t(i) * 3 + 1] - start[size_t(i) * 3 + 1],
                  dz = end[size_t(i) * 3 + 2] - start[size_t(i) * 3 + 2];
      kinematicMove = std::max(kinematicMove, dx * dx + dy * dy + dz * dz);
    }
    kinematicMove = std::sqrt(kinematicMove);
    std::memcpy(kinStart.mapping, start.data(), n * 3 * sizeof(float));
    std::memcpy(kinEnd.mapping, end.data(), n * 3 * sizeof(float));

    m = {};
    m.x            = reinterpret_cast<glm::vec3*>(x.address);
    m.prev         = reinterpret_cast<glm::vec3*>(prev.address);
    m.vel          = reinterpret_cast<glm::vec3*>(vel.address);
    m.cprev        = reinterpret_cast<glm::vec3*>(cprev.address);
    m.kinStart     = reinterpret_cast<glm::vec3*>(kinStart.address);
    m.kinEnd       = reinterpret_cast<glm::vec3*>(kinEnd.address);
    m.kinematic    = reinterpret_cast<int*>(kinematic.address);
    m.freeVerts    = reinterpret_cast<int*>(freeVerts.address);
    m.inv          = reinterpret_cast<float*>(inv.address);
    m.dampRate     = reinterpret_cast<float*>(dampRate.address);
    m.offset       = reinterpret_cast<float*>(offset.address);
    m.friction     = reinterpret_cast<float*>(friction.address);
    m.layer        = reinterpret_cast<int*>(layer.address);
    m.collides     = reinterpret_cast<int*>(collides.address);
    m.grav         = reinterpret_cast<glm::vec3*>(grav.address);
    m.edges        = reinterpret_cast<shaderio::DforceEdge*>(edges.address);
    m.lam          = reinterpret_cast<float*>(lam.address);
    m.tris         = reinterpret_cast<shaderio::DforceTri*>(tris.address);
    m.softVerts    = reinterpret_cast<int*>(softVerts.address);
    m.attAlpha     = reinterpret_cast<float*>(attAlpha.address);
    m.lamAtt       = reinterpret_cast<float*>(lamAtt.address);
    m.contact      = reinterpret_cast<int*>(contact.address);
    m.faceBox      = reinterpret_cast<glm::vec4*>(faceBox.address);
    m.recoverDepth = kRecoverDepth;
    m.nv           = uint32_t(n);
    m.numKinematic = uint32_t(model.kinematicVerts.size());
    m.numFree      = uint32_t(model.freeVerts.size());
    m.numEdges     = uint32_t(colourStart.back());
    m.numTris      = uint32_t(model.staticTris.size());
    m.numSoft      = uint32_t(model.softVerts.size());
    m.cellSize     = kCell;
    m.depth        = kDepth;
    m.cap          = kCap;
    m.maxSpeed     = kMaxSpeed;
    m.searchReach  = model.maxOffset + kSearch;
    if(clusters.count() > 0)
    {
      m.clusterEdges       = reinterpret_cast<shaderio::DforceEdge*>(clusterEdgeBuf.address);
      m.clusterLam         = reinterpret_cast<float*>(clusterLamBuf.address);
      m.clusterVerts       = reinterpret_cast<int*>(clusterVertBuf.address);
      m.clusterVertStart   = reinterpret_cast<int*>(clusterVertStartBuf.address);
      m.clusterOwned       = reinterpret_cast<int*>(clusterOwnedBuf.address);
      m.clusterColourStart = reinterpret_cast<int*>(clusterColourBuf.address);
      m.clusterSoft        = reinterpret_cast<glm::uvec2*>(clusterSoftBuf.address);
      m.clusterSoftStart   = reinterpret_cast<int*>(clusterSoftStartBuf.address);
      m.numClusterEdges    = uint32_t(clusterRows.size());
      m.numClusters        = uint32_t(clusters.count());
      m.clusterColours     = uint32_t(clusters.colours());
      m.clusterStretch     = uint32_t(clusters.stretchColours);
      m.clusterReach       = uint32_t(clusters.reachColours);
    }
    if(clothOn)
    {
      m.clothTris  = reinterpret_cast<shaderio::DforceTri*>(clothTris.address);
      m.meshOf     = reinterpret_cast<int*>(meshOf.address);
      m.ring2Start = reinterpret_cast<int*>(ring2Start.address);
      m.ring2      = reinterpret_cast<int*>(ring2.address);
      m.frameStart = reinterpret_cast<glm::vec3*>(frameStart.address);
      m.snap       = reinterpret_cast<glm::vec3*>(snap.address);
      m.cprevSnap  = reinterpret_cast<glm::vec3*>(cprevSnap.address);
      m.clothContact = reinterpret_cast<int*>(clothContact.address);
    }
    const auto gridStart = std::chrono::steady_clock::now();
    const bool grid      = !model.staticTris.empty() && buildGrid(start, end, m);
    stats.gridMs += msSince(gridStart);
    fv::timing::accumulate("dforceGpu: a frame's collision grid on the host", msSince(gridStart));
    if(!grid)
    {
      if(!model.staticTris.empty() && !gridFailed)
        LOGE("dForce GPU: the collision grid's buffers could not be made; the cloth collides with nothing\n");
      gridFailed = !model.staticTris.empty();
      m.cells    = nullptr;
      m.items    = nullptr;
      m.cellMask = 0;
      m.numTris  = 0;
    }
    std::memcpy(modelBuf.mapping, &m, sizeof(m));
  }
  // the cloth's faces where the cloth is now, every frame: it moves while the pose holds
  if(clothOn)
  {
    if(!buildClothGrid(m) && !gridFailed)
    {
      LOGE("dForce GPU: the cloth's grid could not be made; it doesn't meet itself\n");
      gridFailed = true;
    }
    std::memcpy(modelBuf.mapping, &m, sizeof(m));
  }

  const int   steps  = stepCount > 0 ? stepCount : substeps;
  const float dt     = (seconds > 0 ? seconds : opts.frameDt) / float(steps);
  // the collision passes a frame of `substeps` takes at least, in proportion for a live step's share of one
  const int   least  = stepCount > 0 ? std::max(1, int(std::lround(float(collisions) * float(steps) / float(substeps)))) : collisions;
  const int   passes = std::min(steps, std::max(least, int(std::ceil(std::max(lastMove, kinematicMove) / kPassSpacing))));
  int         next   = steps - 1 - (passes - 1) * (steps / passes);
  fv::timing::accumulate("dforceGpu: a frame's collision passes", double(passes));
  float       tFaces = 0;

  const auto      frameStartTime = std::chrono::steady_clock::now();
  VkCommandBuffer cmd            = commandPool ? nvvk::createSingleTimeCommands(alloc->getDevice(), commandPool) : app->createTempCmdBuffer();
  if(queries)
  {
    vkCmdResetQueryPool(cmd, queries, 0, 2 + 2 * kTimedPairs);
    timedPhase.clear();
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries, 0);
  }
  const auto dispatch = [this](VkCommandBuffer c, uint32_t x, uint32_t y, uint32_t z) {
    stats.dispatches++;
    vkCmdDispatch(c, x, y, z);
  };
  // a phase's dispatch between two timestamps (0 collision, 1 hold, 2 the frame's recovery), while there are queries for it
  const auto timed = [&](int phase, const auto& record) {
    const bool time = queries && timedPhase.size() < kTimedPairs;
    const uint32_t q = 2 + 2 * uint32_t(timedPhase.size());
    if(time)
    {
      vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, q);
      timedPhase.push_back(phase);
    }
    record();
    if(time)
      vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, q + 1);
  };
  const auto      groups = [](uint32_t count) { return (count + DFORCE_WORKGROUP_SIZE - 1) / DFORCE_WORKGROUP_SIZE; };
  const auto      barrier = [&] {
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  };
  shaderio::DforcePushConstant pc{};
  pc.model        = reinterpret_cast<shaderio::DforceModel*>(modelBuf.address);
  pc.dt           = dt;
  pc.extraDamping = extraDamping;
  const auto colours = [&](size_t first, size_t last) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, edgesPipe);
    for(size_t c = first; c < last && c + 1 < colourStart.size(); c++)
    {
      pc.edgeStart = uint32_t(colourStart[c]);
      pc.edgeCount = uint32_t(colourStart[c + 1] - colourStart[c]);
      if(pc.edgeCount == 0)
        continue;
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      dispatch(cmd, groups(pc.edgeCount), 1, 1);
      barrier();
    }
  };
  // one sweep, as the CPU solver's: every edge but the links, or those that hold length; then the tethers.
  // The cluster solve runs the clusters' rows and tethers first, a workgroup a cluster, then the rows
  // that cross them
  // which of x and its twin holds the free vertices: the cluster sweeps go back and forth between them
  bool inTwin = false;
  // the free vertices back in x, when an odd number of cluster sweeps left them in its twin
  const auto home = [&] {
    if(!inTwin)
      return;
    pc.clusterSrc = reinterpret_cast<glm::vec3*>(xTwin.address);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, commitPipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    dispatch(cmd, groups(m.numFree), 1, 1);
    barrier();
    inTwin = false;
  };
  const auto sweep = [&](bool lengthOnly) {
    if(clusters.count() > 0)
    {
      pc.lengthOnly = lengthOnly ? 1 : 0;
      pc.clusterSrc = reinterpret_cast<glm::vec3*>(inTwin ? xTwin.address : x.address);
      pc.clusterDst = reinterpret_cast<glm::vec3*>(inTwin ? x.address : xTwin.address);
      inTwin        = !inTwin;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clusterPipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      dispatch(cmd, uint32_t(clusters.count()), 1, 1);
      barrier();
      // the crossing rows read and write x
      if(colourStart.size() > 1)
        home();
    }
    colours(0, size_t(stretchColours));
    if(lengthOnly)
      colours(size_t(stretchColours), size_t(lengthColours));
    else
      colours(size_t(lengthColours), colourStart.size() - 1);
    if(m.numSoft > 0 && clusters.count() == 0)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, attachPipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      dispatch(cmd, groups(m.numSoft), 1, 1);
      barrier();
    }
  };
  // resting vertices held on their faces where the pose has them at pc.t
  const auto hold = [&](bool grip) {
    home();
    if(m.numTris == 0 || (debug & 2))
      return;
    pc.grip = grip ? 1 : 0;
    timed(1, [&] {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, holdPipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      dispatch(cmd, groups(m.numFree), 1, 1);
      barrier();
    });
  };
  // each face's box where the pose has it at pc.t, for the collision pass or the recovery that follows
  const auto boxes = [&] {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, boxesPipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    dispatch(cmd, groups(m.numTris), 1, 1);
    barrier();
  };
  // the frame's start: cloth lost behind a face of a lower layer is put back, as on the CPU
  if(m.numTris > 0 && opts.recoverLost && !(debug & 1))
  {
    pc.t = 0;
    timed(2, [&] {
      boxes();
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waveLanes ? recoverWavePipe : recoverPipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      dispatch(cmd, groups(m.numFree * std::max(1u, waveLanes)), 1, 1);
      barrier();
    });
  }
  for(int s = 0; s < steps; s++)
  {
    pc.t      = float(s + 1) / float(steps);
    pc.tFaces = tFaces;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integratePipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    dispatch(cmd, groups(std::max({m.numKinematic, m.numFree, m.numEdges, m.numSoft, m.numClusterEdges})), 1, 1);
    barrier();
    for(int it = 0; it < std::max(1, opts.iterations); it++)
      sweep(false);
    for(int it = 0; it < opts.stretchSweeps; it++)
      sweep(true);
    hold(true);
    if(s == next && (m.numTris > 0 || m.numClothTris > 0))
    {
      timed(0, [&] {
        // the cloth's faces as the pass begins, so a vertex reads them as they are, not as far as
        // another has got
        if(m.numClothTris > 0)
        {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, snapPipe);
          vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
          dispatch(cmd, groups(m.nv), 1, 1);
          barrier();
        }
        if(m.numTris > 0)
          boxes();
        pc.clothLook    = m.numClothTris > 0 && (++passCount % 4) == 0 ? 1u : 0u;
        const bool wave = waveLanes && m.numTris > 0;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, wave ? collideWavePipe : collidePipe);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        dispatch(cmd, groups(m.numFree * (wave ? waveLanes : 1u)), 1, 1);
        barrier();
      });
      // the pushes stretched edges; one more sweep shares them out before velocities are read
      sweep(false);
      hold(false);
      tFaces = pc.t;
      next += std::max(1, steps / passes);
    }
    home();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, velocityPipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    dispatch(cmd, groups(m.numFree), 1, 1);
    barrier();
  }
  if(queries)
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, queries, 1);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  const VkBufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = n * 3 * sizeof(float)};
  vkCmdCopyBuffer(cmd, x.buffer, readback.buffer, 1, &copy);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_HOST_BIT);
  const auto waitStart = std::chrono::steady_clock::now();
  if(commandPool)
  {
    NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, alloc->getDevice(), commandPool, queue.queue));
  }
  else
    app->submitAndWaitTempCmdBuffer(cmd);
  const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStart).count();
  settleWait += waited;
  stats.waitMs += waited;
  fv::timing::accumulate("dforceGpu: a frame's wait on the GPU", waited);
  stats.frames++;
  stats.substeps = steps;
  if(queries)
  {
    uint64_t ticks[2] = {0, 0};
    if(vkGetQueryPoolResults(alloc->getDevice(), queries, 0, 2, sizeof ticks, ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
    {
      stats.gpuMs += double(ticks[1] - ticks[0]) * msPerTick;
      fv::timing::accumulate("dforceGpu: a frame on the GPU", double(ticks[1] - ticks[0]) * msPerTick);
    }
    std::vector<uint64_t> pairs(timedPhase.size() * 2);
    if(!pairs.empty()
       && vkGetQueryPoolResults(alloc->getDevice(), queries, 2, uint32_t(pairs.size()), pairs.size() * sizeof(uint64_t), pairs.data(), sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
              == VK_SUCCESS)
    {
      double phaseMs[3] = {0, 0, 0};
      for(size_t k = 0; k < timedPhase.size(); k++)
        phaseMs[std::clamp(timedPhase[k], 0, 2)] += double(pairs[2 * k + 1] - pairs[2 * k]) * msPerTick;
      stats.collideMs += phaseMs[0] + phaseMs[2];
      stats.recoverMs += phaseMs[2];
      stats.holdMs += phaseMs[1];
      fv::timing::accumulate("dforceGpu: a frame's recovery on the GPU", phaseMs[2]);
      fv::timing::accumulate("dforceGpu: a frame's collision passes on the GPU", phaseMs[0]);
      fv::timing::accumulate("dforceGpu: a frame's holds on the GPU", phaseMs[1]);
    }
  }

  std::vector<float> frameStart = std::move(xHost);
  xHost.resize(n * 3);
  std::memcpy(xHost.data(), readback.mapping, n * 3 * sizeof(float));
  if(std::any_of(xHost.begin(), xHost.end(), [](float v) { return !std::isfinite(v); }))
  {
    LOGE("dForce GPU: a position came back that is not a number; the step is dropped and the simulation stopped\n");
    xHost  = std::move(frameStart);
    broken = true;
    return;
  }
  float moved = 0;
  for(int i : model.freeVerts)
  {
    const float dx = xHost[size_t(i) * 3] - frameStart[size_t(i) * 3], dy = xHost[size_t(i) * 3 + 1] - frameStart[size_t(i) * 3 + 1],
                dz = xHost[size_t(i) * 3 + 2] - frameStart[size_t(i) * 3 + 2];
    moved = std::max(moved, dx * dx + dy * dy + dz * dz);
  }
  lastMove = std::sqrt(moved);
  // a frame that took far longer than a frame should is the one before a device loss: say what it saw
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStartTime).count();
  if(ms > 250)
  {
    float farthest = 0;
    for(float v : xHost)
      farthest = std::max(farthest, std::fabs(v));
    LOGW("dForce GPU: a frame took %.0f ms (%d substeps, %d collision passes, the fastest vertex moved %.1f cm, the farthest position %.0f cm)\n", ms,
         steps, passes, lastMove, farthest);
    // FV_DFORCE_GPU_DEBUG's 512: no frame after it, for trying a kernel that may run away
    if(debug & 512)
    {
      LOGW("dForce GPU: stopped after a long frame (FV_DFORCE_GPU_DEBUG 512)\n");
      broken = true;
    }
  }
}

DforceGpu::DforceGpu(nvvk::ResourceAllocator* alloc, nvapp::Application* app, std::vector<SettleMesh> meshes, const SettleOptions& opts,
                     const Options& gpu)
    : m_impl(std::make_shared<Impl>())
{
  m_impl->alloc  = alloc;
  m_impl->app    = app;
  m_impl->meshes = std::move(meshes);
  m_impl->opts   = opts;
  m_impl->gpu    = gpu;
  m_impl->queue  = s_queue;
  // FV_DFORCE_GPU_DEBUG's bits 4 and 8: a dispatch a colour, and the renderer's queue, as before step 13
  // (for an A/B against the cluster solve on a queue of its own); 32: each frame timed on the GPU (the
  // [fv-time] "a frame on the GPU" lines, against the wait for it)
#pragma warning(suppress : 4996)  // getenv: read once a simulation
  if(const char* env = std::getenv("FV_DFORCE_GPU_DEBUG"); env && *env)
  {
    const int bits = std::atoi(env);
    if(bits & 4)
      m_impl->gpu.clusterSize = 0;
    if(bits & 8)
      m_impl->queue = {};
    if(bits & 32)
      m_impl->gpu.timestamps = true;
    // 64: the cloth meets other garments' faces, 128 its own too (unfinished; read in prepare)
  }
  if(alloc && app)
    m_prepared = fv::jobs::pool().submit([impl = m_impl] { return impl->prepare(); });
}

DforceGpu::~DforceGpu()
{
  // still on the pool: it has made no buffers, and lets go of the rest when it ends; the kernels a
  // restart handed it go here, not on the pool after the device
  if(!m_prepared.valid())
    m_impl->destroy();
  else
    m_impl->kernels.reset();
}

bool DforceGpu::started(bool wait)
{
  if(!m_prepared.valid())
  {
    if(m_valid && m_impl->resetPending)
      m_impl->resetState();
    return m_valid;
  }
  if(!wait && m_prepared.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    return false;
  bool ok = false;
  try
  {
    ok = m_prepared.get();
  }
  catch(const std::exception& e)
  {
    LOGE("dForce GPU: %s\n", e.what());
  }
  m_valid = ok && m_impl->create();
  if(!m_valid)
  {
    m_impl->destroy();
    if(onStartFailed)
      onStartFailed();
  }
  return m_valid;
}

void DforceGpu::setGravity(const std::array<float, 3>& gravity)
{
  if(!started(true))
    return;
  // the frames before this one have finished: a step waits for its own
  std::vector<float>& grav = m_impl->model.grav;
  for(size_t i = 0; i + 2 < grav.size(); i += 3)
    std::copy(gravity.begin(), gravity.end(), grav.begin() + long(i));
  std::memcpy(m_impl->grav.mapping, grav.data(), grav.size() * sizeof(float));
}

void DforceGpu::frame(float t0, float t1, float extraDamping, float seconds)
{
  if(!started(true) || m_impl->broken)
    return;
  if(seconds > 0)
  {
    Impl& s = *m_impl;
    // Live: substeps as long as a settle's (a frame of frameDt in `substeps`), as many as the time it is
    // given takes, the rest of it carried to the next step, so the cloth is as stiff at any frame rate;
    // and one frame, never pieces. A frame's own passes (the recovery, the collisions, the hold) cost as
    // much as its substeps, and beside the path tracer a frame on the GPU takes 4-5 times as long as
    // alone: in pieces a step fell behind the clip, the pose then jumped further by the next, which went
    // in more pieces (a walk: a step 390 ms, then 154 with the substeps shared). One frame of 12 substeps
    // at 60 frames a second sweeps a walking foot 2-3 cm, the colliders' substeps between
    const float h     = s.opts.frameDt / float(std::max(1, s.substeps));
    const float total = seconds + s.liveCarry;
    const int   steps = std::max(1, int(std::floor(total / h)));
    s.liveCarry       = std::max(0.f, total - float(steps) * h);
    fv::timing::accumulate("dforceGpu: live substeps a step", double(steps));
    s.frame(t0, t1, extraDamping, float(steps) * h, steps);
    m_lastMove = m_impl->broken ? 0.f : m_impl->lastMove;
    return;
  }
  // a pose that jumps (a value typed, an undo) in one frame leaves behind what rests on it: a 15 degree
  // turn of the hip moves the shoulders 16 cm, and the top of a dress hanging from them fell 44. A drag
  // of the same turn over a second is fine. In pieces it costs a slow frame only when it happens
  Impl& s        = *m_impl;
  float farthest = 0;
  if(t0 != t1)
  {
    std::vector<float> from = s.xHost, to = s.xHost;
    const auto         skinStart = std::chrono::steady_clock::now();
    s.kinematicAt(t0, from);
    s.kinematicAt(t1, to);
    s.stats.skinMs += msSince(skinStart);
    for(int i : s.model.kinematicVerts)
    {
      const float dx = to[size_t(i) * 3] - from[size_t(i) * 3], dy = to[size_t(i) * 3 + 1] - from[size_t(i) * 3 + 1], dz = to[size_t(i) * 3 + 2] - from[size_t(i) * 3 + 2];
      farthest = std::max(farthest, dx * dx + dy * dy + dz * dz);
    }
  }
  const int   pieces = std::clamp(int(std::ceil(std::sqrt(farthest) / kJumpPerPiece)), 1, kMostPieces);
  // how many frames a step takes: a fast pose goes in pieces (the mean is the pieces a step)
  fv::timing::accumulate("dforceGpu: pieces a step", double(pieces));
  const float length = s.opts.frameDt / float(pieces);
  for(int p = 0; p < pieces && !s.broken; p++)
    s.frame(t0 + (t1 - t0) * float(p) / float(pieces), t0 + (t1 - t0) * float(p + 1) / float(pieces), extraDamping, length);
  m_lastMove = m_impl->broken ? 0.f : m_impl->lastMove;
}

void DforceGpu::setTarget(size_t mesh, const std::vector<Mat4>& bones)
{
  if(!started(true))
    return;
  Impl& s = *m_impl;
  if(mesh >= s.kin.size() || !s.kin[mesh].skinned || bones.size() != s.kin[mesh].to.size())
    return;
  s.kin[mesh].from = s.kin[mesh].to;
  for(size_t b = 0; b < bones.size(); b++)
    s.kin[mesh].to[b] = poseOf(bones[b]);
  s.heldAt = -1;
  targetSet();
}

// Each vertex the pose doesn't place moved as its own skin moves between the sweep's start and t: its
// blended bone matrix at t times the inverse of the one at the start, so a turned thigh turns the skirt on
// it rather than sliding it. Its velocity is kept: a step a few milliseconds late is carried that much,
// and a swinging hem goes on swinging.
void DforceGpu::Impl::carry(float t)
{
  const size_t       n  = size_t(model.nv);
  std::vector<float> to = xHost;
  for(size_t mi = 0; mi < meshes.size(); mi++)
  {
    const SettleMesh& m = meshes[mi];
    if(!kin[mi].skinned)
      continue;
    std::vector<Mat4> from(kin[mi].to.size()), at(kin[mi].to.size());
    for(size_t b = 0; b < from.size(); b++)
    {
      from[b] = m.toSpace * sweepAt(kin[mi].from[b], kin[mi].to[b], 0);
      at[b]   = m.toSpace * sweepAt(kin[mi].from[b], kin[mi].to[b], t);
    }
    const size_t o = size_t(model.base[mi]);
    fv::jobs::parallelFor(m.vertexCount(), 1024, [&](size_t begin, size_t end) {
      for(size_t v = begin; v < end; v++)
      {
        const size_t i = o + v;
        if(!model.dynamic[i] || model.inv[i] <= 0)
          continue;
        Mat4  a, b;
        float wsum = 0;
        a.e.fill(0);
        b.e.fill(0);
        for(size_t j = 0; j < 4; j++)
        {
          const float  w    = m.skinWeight[v * 4 + j];
          const size_t bone = m.skinIndex[v * 4 + j];
          if(w == 0 || bone >= from.size())
            continue;
          wsum += w;
          for(size_t c = 0; c < 16; c++)
          {
            a.e[c] += from[bone].e[c] * w;
            b.e[c] += at[bone].e[c] * w;
          }
        }
        if(wsum <= 0)
          continue;
        const Vec3 p = (b * inverse(a)).transformPoint({xHost[i * 3], xHost[i * 3 + 1], xHost[i * 3 + 2]});
        if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
          continue;
        to[i * 3]     = float(p.x);
        to[i * 3 + 1] = float(p.y);
        to[i * 3 + 2] = float(p.z);
      }
    });
  }
  // nothing is in flight between steps: the buffers are written as they lie
  xHost = std::move(to);
  for(nvvk::Buffer* buf : {&x, &prev, &cprev, &xTwin})
    if(buf->mapping)
      std::memcpy(buf->mapping, xHost.data(), n * 3 * sizeof(float));
  heldAt = -1;
}

bool DforceGpu::carry(float t)
{
  if(!started(true) || m_impl->broken)
    return false;
  const auto start = std::chrono::steady_clock::now();
  m_impl->carry(t);
  fv::timing::accumulate("dforceGpu: a late step's carry", msSince(start));
  return true;
}

bool DforceGpu::setRest(size_t mesh, std::vector<float> restLocal)
{
  if(!started(true))
    return false;
  Impl& s = *m_impl;
  if(mesh >= s.kin.size() || !s.kin[mesh].skinned || restLocal.size() != s.meshes[mesh].restLocal.size())
    return false;
  s.meshes[mesh].restLocal = std::move(restLocal);
  // a held frame skins nothing again: this one must
  s.heldAt = -1;
  return true;
}

void DforceGpu::settle()
{
  started(true);
  while(!settleSome(std::numeric_limits<double>::infinity()))
  {
  }
}

bool DforceGpu::settleSome(double seconds)
{
  if(!started(false))
    return !m_prepared.valid();
  Impl&      s      = *m_impl;
  const auto start  = std::chrono::steady_clock::now();
  const int  frames = std::max(1, int(std::lround(s.transition / s.opts.frameDt)));
  const int  hold   = std::max(1, int(std::ceil(s.opts.maxHold / s.opts.frameDt)));
  if(s.settled == 0)
  {
    s.settleStart = start;
    s.settleWait  = 0;
    s.settleWork  = 0;
  }
  bool done = s.broken;
  while(!done)
  {
    // an edit has made it out of date: what it has is dropped
    if(s.opts.cancelled && s.opts.cancelled())
    {
      done = true;
      break;
    }
    if(s.settled < frames)
      frame(float(s.settled) / float(frames), float(s.settled + 1) / float(frames));
    else
      frame(1, 1, kHoldDamping);
    s.settled++;
    done = s.broken || s.settled >= frames + hold || (s.settled > frames && m_lastMove < s.opts.restMove);
    if(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= seconds)
      break;
  }
  const auto end = std::chrono::steady_clock::now();
  s.settleWork += std::chrono::duration<double, std::milli>(end - start).count();
  if(done)
    LOGI("[fv-time] dforceGpuSettle %d frames %.1f ms of work, %.1f ms waiting on the GPU, %.1f ms in all (%.1f ms a frame, %d substeps; on the host %.1f ms skinning, %.1f the collision grid)\n",
         s.settled, s.settleWork, s.settleWait, std::chrono::duration<double, std::milli>(end - s.settleStart).count(), s.settleWork / std::max(1, s.settled),
         s.substeps, s.stats.skinMs, s.stats.gridMs);
  if(done && s.stats.frames > 0)
    LOGI("[fv-time] dforceGpuGrid %.0f entries and %.0f slots a frame, %.2f ms of its %.2f a frame copying\n",
         double(s.stats.gridEntries) / s.stats.frames, double(s.stats.gridSlots) / s.stats.frames, s.stats.gridCopyMs / s.stats.frames,
         s.stats.gridMs / s.stats.frames);
  return done;
}

bool DforceGpu::ownerThreadOnly() const
{
  return !m_impl->queue.queue;
}

bool DforceGpu::restart(std::vector<SettleMesh> meshes, const SettleOptions& opts)
{
  // FV_DFORCE_GPU_DEBUG's bit 16: a new simulation for every settle, as before this was kept (an A/B)
  if(failed() || (m_impl->debug & 16))
    return false;
  Impl& s    = *m_impl;
  bool  same = !m_prepared.valid() && m_valid && meshes.size() == s.meshes.size() && opts.air == s.opts.air && opts.reachLinks == s.opts.reachLinks;
  for(size_t mi = 0; same && mi < meshes.size(); mi++)
    if(const char* what = fv::physics::cloth::poseFreeDifference(meshes[mi], s.meshes[mi], true))
    {
      LOGI("dForce GPU: the model is made again: mesh %zu's %s changed\n", mi, what);
      same = false;
    }
  m_lastMove = 0;
  if(same)
  {
    // the model's parts of the pose: the posed cage, which way is down, and a collider's rest cage (the
    // start is made again from it); the rest is as it was
    for(size_t mi = 0; mi < meshes.size(); mi++)
    {
      const SettleMesh& m = meshes[mi];
      const size_t      o = size_t(s.model.base[mi]);
      std::copy(m.posed.begin(), m.posed.end(), s.model.posed.begin() + long(o * 3));
      for(size_t v = 0; v < m.vertexCount(); v++)
        std::copy(m.gravity.begin(), m.gravity.end(), s.model.grav.begin() + long((o + v) * 3));
      if(m.rest != s.meshes[mi].rest)
      {
        std::copy(m.rest.begin(), m.rest.end(), s.model.rest.begin() + long(o * 3));
        s.startPending = true;
      }
    }
    s.meshes = std::move(meshes);
    s.opts   = opts;
    s.posePrepare();
    // the buffers on the owner's thread, as the settle starts
    s.resetPending = true;
    return true;
  }
  // other shapes or settings: a model made again on the pool, with the kernels this one made. What it
  // had of the device goes now: nothing is in flight between settles
  auto fresh     = std::make_shared<Impl>();
  fresh->alloc   = s.alloc;
  fresh->app     = s.app;
  fresh->gpu     = s.gpu;
  fresh->queue   = s.queue;
  fresh->kernels = std::move(s.kernels);
  fresh->meshes  = std::move(meshes);
  fresh->opts    = opts;
  if(!m_prepared.valid())
    s.destroy();
  m_impl     = std::move(fresh);
  m_valid    = false;
  m_prepared = fv::jobs::pool().submit([impl = m_impl] { return impl->prepare(); });
  return true;
}

void DforceGpu::useQueue(const nvvk::QueueInfo& queue)
{
  s_queue = queue;
}

DforceGpu::Stats DforceGpu::stats() const
{
  return m_impl->stats;
}

bool DforceGpu::failed() const
{
  return !m_prepared.valid() && (!m_valid || m_impl->broken);
}

std::vector<std::vector<float>> DforceGpu::positions() const
{
  const Impl&                     s = *m_impl;
  std::vector<std::vector<float>> out;
  for(size_t mi = 0; mi < s.meshes.size(); mi++)
  {
    const size_t o = size_t(s.model.base[mi]) * 3;
    out.emplace_back(s.xHost.begin() + long(o), s.xHost.begin() + long(o + s.meshes[mi].rest.size()));
  }
  return out;
}
