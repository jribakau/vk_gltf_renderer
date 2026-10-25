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
#include <cmath>
#include <cstring>

#include <glm/glm.hpp>
#include <nvapp/application.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
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
constexpr float kTransitionSpeed  = 60.f, kMinTransition = 0.25f, kMaxTransition = 1.5f;
constexpr float kHoldDamping      = 6.f;
constexpr int   kMinCollisionPasses = 4;
constexpr float kPassSpacing      = 0.5f * kDepth;
constexpr float kSearch           = 0.5f;
constexpr float kRecoverDepth     = 3.f;

constexpr VkBufferUsageFlags2 kUsage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                       | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT;

uint32_t cellHash(int ix, int iy, int iz)
{
  return (uint32_t(ix) * 73856093u) ^ (uint32_t(iy) * 19349663u) ^ (uint32_t(iz) * 83492791u);
}

int cellOf(float p)
{
  return int(std::floor(p / kCell));
}

// How a mesh's kinematic vertices move: a sweep of its bones from one pose to another.
struct Kinematic
{
  bool                  skinned = false;
  std::vector<BonePose> from, to;
};

}  // namespace

struct DforceGpu::Impl
{
  nvvk::ResourceAllocator* alloc = nullptr;
  nvapp::Application*      app   = nullptr;
  std::vector<SettleMesh>  meshes;
  SettleOptions            opts;
  ClothModel               model;
  std::vector<Kinematic>   kin;

  // edges sorted by colour, and where each colour starts; the stretch edges have the first colours
  // to themselves, so a sweep over them alone is those colours
  std::vector<int> colourStart;
  int              stretchColours = 0;

  VkPipelineLayout layout{};
  VkPipeline       integratePipe{}, edgesPipe{}, attachPipe{}, recoverPipe{}, holdPipe{}, collidePipe{}, velocityPipe{};

  // the model's buffers; `cells` and `items` grow with the grid
  nvvk::Buffer x, prev, vel, cprev, kinStart, kinEnd, kinematic, freeVerts, inv, dampRate, offset, friction, layer, collides, grav, edges,
      lam, tris, cells, items, softVerts, attAlpha, lamAtt, contact, modelBuf, readback;
  size_t cellCapacity = 0, itemCapacity = 0;
  bool   gridFailed = false;

  // host copies: every position (the kinematic ones kept where the last frame ended them)
  std::vector<float> xHost;
  int                substeps   = kMinSubsteps;
  int                collisions = kMinCollisionPasses;
  float              transition = kMinTransition;
  float              lastMove   = 0;
  // a frame that came back with a position that is not a number: the frame is dropped and the
  // simulation stops, so one bad step cannot take the device down with it
  bool broken = false;
  // FV_DFORCE_GPU_DEBUG, bits: 1 leaves the recovery out, 2 the hold between passes; for finding
  // which kernel a device loss comes from
  int debug = 0;

  // ---- setup
  bool init();
  void destroy();
  template <class T>
  bool mapped(nvvk::Buffer& buf, const std::vector<T>& data, VmaMemoryUsage usage = VMA_MEMORY_USAGE_CPU_TO_GPU)
  {
    return mappedBytes(buf, data.data(), data.size() * sizeof(T), usage);
  }
  bool mappedBytes(nvvk::Buffer& buf, const void* data, size_t bytes, VmaMemoryUsage usage);
  bool makePipeline(const char* entry, VkPipeline& out);

  // ---- a frame
  void kinematicAt(float t, std::vector<float>& out);
  bool buildGrid(const std::vector<float>& start, const std::vector<float>& end, shaderio::DforceModel& m);
  void frame(float t0, float t1, float extraDamping);
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

bool DforceGpu::Impl::makePipeline(const char* entry, VkPipeline& out)
{
  VkShaderModuleCreateInfo shaderInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = dforce_cloth_comp_slang_sizeInBytes,
                                      .pCode    = dforce_cloth_comp_slang};
  VkComputePipelineCreateInfo pipeInfo{.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                       .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                  .pNext = &shaderInfo,
                                                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                  .pName = entry},
                                       .layout = layout};
  if(vkCreateComputePipelines(alloc->getDevice(), nullptr, 1, &pipeInfo, nullptr, &out) != VK_SUCCESS)
  {
    LOGE("dForce GPU: could not create the %s pipeline\n", entry);
    return false;
  }
  NVVK_DBG_NAME(out);
  return true;
}

bool DforceGpu::Impl::init()
{
  model = ClothModel::build(meshes, opts.air);
  if(model.nv == 0 || model.freeVerts.empty())
    return false;
#pragma warning(suppress : 4996)  // getenv: read once, at start-up
  if(const char* env = std::getenv("FV_DFORCE_GPU_DEBUG"); env && *env)
    debug = std::atoi(env);

  // the pose's hold on each mesh, swept from the memorized pose to the target it came with
  kin.resize(meshes.size());
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

  // greedy colouring: no two edges of a colour share a vertex, so a colour projects in parallel.
  // The stretch edges are coloured first, among themselves, so the sweeps over them alone (as the
  // CPU solver's stretchSweeps) are the first colours
  std::vector<int> colourOf(model.edges.size(), -1);
  int              colours = 0;
  const auto       colourKind = [&](bool stretch) {
    std::vector<uint64_t> used(size_t(model.nv), 0);
    for(size_t e = 0; e < model.edges.size(); e++)
    {
      const Edge& ed = model.edges[e];
      if((ed.kind == EdgeKind::eStretch) != stretch || (model.inv[ed.a] <= 0 && model.inv[ed.b] <= 0))
        continue;
      const uint64_t taken = used[size_t(ed.a)] | used[size_t(ed.b)];
      int            c     = 0;
      while(c < 63 && (taken >> c) & 1)
        c++;
      if(c >= 63)
      {
        LOGE("dForce GPU: a vertex has more than 63 edges of one kind; the CPU solver takes this one\n");
        return false;
      }
      colourOf[e] = colours + c;
      used[size_t(ed.a)] |= uint64_t(1) << c;
      used[size_t(ed.b)] |= uint64_t(1) << c;
    }
    int most = 0;
    for(size_t e = 0; e < model.edges.size(); e++)
      if(colourOf[e] >= colours)
        most = std::max(most, colourOf[e] - colours + 1);
    colours += most;
    return true;
  };
  if(!colourKind(true))
    return false;
  stretchColours = colours;
  if(!colourKind(false))
    return false;
  std::vector<shaderio::DforceEdge> sorted;
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

  std::vector<shaderio::DforceTri> faces;
  for(int t : model.staticTris)
  {
    const Tri& tri = model.tris[size_t(t)];
    faces.push_back({tri.a, tri.b, tri.c, tri.layer, tri.offset, tri.friction, 0.f, 0.f});
  }

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
  // the start: the rest shape, with what overlaps the body moved out of it, as the CPU solver starts
  xHost = model.rest;
  separateStart(model, xHost);
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
  ok &= mapped(softVerts, model.softVerts.empty() ? std::vector<int>{0} : model.softVerts);
  ok &= mapped(attAlpha, model.attAlpha);
  ok &= mapped(lamAtt, lamAttZero);
  ok &= mapped(contact, noContact);
  ok &= mappedBytes(modelBuf, nullptr, sizeof(shaderio::DforceModel), VMA_MEMORY_USAGE_CPU_TO_GPU);
  ok &= mappedBytes(readback, nullptr, n * 3 * sizeof(float), VMA_MEMORY_USAGE_GPU_TO_CPU);
  if(!ok)
    return false;

  VkPushConstantRange        pushRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::DforcePushConstant)};
  VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
  if(vkCreatePipelineLayout(alloc->getDevice(), &layoutInfo, nullptr, &layout) != VK_SUCCESS)
    return false;
  NVVK_DBG_NAME(layout);
  if(!makePipeline("integrateMain", integratePipe) || !makePipeline("edgesMain", edgesPipe) || !makePipeline("attachMain", attachPipe)
     || !makePipeline("recoverMain", recoverPipe) || !makePipeline("holdMain", holdPipe) || !makePipeline("collideMain", collidePipe)
     || !makePipeline("velocityMain", velocityPipe))
    return false;

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
  LOGI("dForce GPU: %d vertices (%zu free, %zu soft), %zu edges in %d colours (%d stretch), %zu collision faces\n", model.nv,
       model.freeVerts.size(), model.softVerts.size(), sorted.size(), colours, stretchColours, faces.size());
  return true;
}

void DforceGpu::Impl::destroy()
{
  if(!alloc)
    return;
  VkDevice device = alloc->getDevice();
  for(VkPipeline* p : {&integratePipe, &edgesPipe, &attachPipe, &recoverPipe, &holdPipe, &collidePipe, &velocityPipe})
    if(*p)
    {
      vkDestroyPipeline(device, *p, nullptr);
      *p = VK_NULL_HANDLE;
    }
  if(layout)
  {
    vkDestroyPipelineLayout(device, layout, nullptr);
    layout = VK_NULL_HANDLE;
  }
  for(nvvk::Buffer* b : {&x, &prev, &vel, &cprev, &kinStart, &kinEnd, &kinematic, &freeVerts, &inv, &dampRate, &offset, &friction, &layer,
                         &collides, &grav, &edges, &lam, &tris, &cells, &items, &softVerts, &attAlpha, &lamAtt, &contact, &modelBuf, &readback})
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

// The body's faces over their sweep through the frame, into the open-addressed cell table. False
// when the table's buffers could not be made: the frame then runs without collisions.
bool DforceGpu::Impl::buildGrid(const std::vector<float>& start, const std::vector<float>& end, shaderio::DforceModel& m)
{
  const size_t                 n = model.staticTris.size();
  std::vector<std::array<int, 6>> range(n);
  size_t                        entries = 0;
  for(size_t t = 0; t < n; t++)
  {
    const Tri& tri = model.tris[size_t(model.staticTris[t])];
    float      lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(int k = 0; k < 6; k++)
    {
      const int   v = k % 3 == 0 ? tri.a : k % 3 == 1 ? tri.b : tri.c;
      const auto& p = k < 3 ? start : end;
      for(int c = 0; c < 3; c++)
      {
        lo[c] = std::min(lo[c], p[size_t(v) * 3 + c]);
        hi[c] = std::max(hi[c], p[size_t(v) * 3 + c]);
      }
    }
    size_t count = 1;
    for(int c = 0; c < 3; c++)
    {
      range[t][c]     = cellOf(lo[c]);
      range[t][c + 3] = cellOf(hi[c]);
      count *= size_t(range[t][c + 3] - range[t][c] + 1);
    }
    entries += count;
  }
  size_t slots = 64;
  while(slots < entries * 2)
    slots *= 2;
  std::vector<shaderio::DforceCell> table(slots, shaderio::DforceCell{0, 0, 0, -1, 0, 0, 0, 0});
  const auto                        find = [&](int ix, int iy, int iz) {
    size_t s = cellHash(ix, iy, iz) & (slots - 1);
    while(table[s].start >= 0 && (table[s].x != ix || table[s].y != iy || table[s].z != iz))
      s = (s + 1) & (slots - 1);
    return s;
  };
  // count per cell, then place: `start` doubles as the "taken" flag, so the first pass stores the
  // count there and the second turns it into offsets
  for(size_t t = 0; t < n; t++)
    for(int ix = range[t][0]; ix <= range[t][3]; ix++)
      for(int iy = range[t][1]; iy <= range[t][4]; iy++)
        for(int iz = range[t][2]; iz <= range[t][5]; iz++)
        {
          const size_t s = find(ix, iy, iz);
          if(table[s].start < 0)
            table[s] = {ix, iy, iz, 0, 0, 0, 0, 0};
          table[s].count++;
        }
  int at = 0;
  for(shaderio::DforceCell& c : table)
    if(c.start >= 0)
    {
      c.start = at;
      at += c.count;
      c.count = 0;
    }
  std::vector<int> list(std::max<size_t>(1, entries), 0);
  for(size_t t = 0; t < n; t++)
    for(int ix = range[t][0]; ix <= range[t][3]; ix++)
      for(int iy = range[t][1]; iy <= range[t][4]; iy++)
        for(int iz = range[t][2]; iz <= range[t][5]; iz++)
        {
          shaderio::DforceCell& c            = table[find(ix, iy, iz)];
          list[size_t(c.start + c.count++)] = int(t);
        }
  if(slots > cellCapacity)
  {
    if(!mapped(cells, table))
    {
      cellCapacity = 0;
      return false;
    }
    cellCapacity = slots;
  }
  else
    std::memcpy(cells.mapping, table.data(), slots * sizeof(shaderio::DforceCell));
  if(list.size() > itemCapacity)
  {
    if(!mapped(items, list))
    {
      itemCapacity = 0;
      return false;
    }
    itemCapacity = list.size();
  }
  else
    std::memcpy(items.mapping, list.data(), list.size() * sizeof(int));
  m.cells    = reinterpret_cast<shaderio::DforceCell*>(cells.address);
  m.items    = reinterpret_cast<int*>(items.address);
  m.cellMask = uint32_t(slots - 1);
  return true;
}

void DforceGpu::Impl::frame(float t0, float t1, float extraDamping)
{
  const size_t n = size_t(model.nv);
  // where the pose takes what it holds by the end of the frame
  std::vector<float> start = xHost, end = xHost;
  kinematicAt(t0, start);
  kinematicAt(t1, end);
  float kinematicMove = 0;
  for(int i : model.kinematicVerts)
  {
    const float dx = end[size_t(i) * 3] - start[size_t(i) * 3], dy = end[size_t(i) * 3 + 1] - start[size_t(i) * 3 + 1],
                dz = end[size_t(i) * 3 + 2] - start[size_t(i) * 3 + 2];
    kinematicMove = std::max(kinematicMove, dx * dx + dy * dy + dz * dz);
  }
  kinematicMove = std::sqrt(kinematicMove);
  std::memcpy(kinStart.mapping, start.data(), n * 3 * sizeof(float));
  std::memcpy(kinEnd.mapping, end.data(), n * 3 * sizeof(float));

  shaderio::DforceModel m{};
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
  if(model.staticTris.empty() || !buildGrid(start, end, m))
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

  const int   steps  = substeps;
  const float dt     = opts.frameDt / float(steps);
  const int   passes = std::min(steps, std::max(collisions, int(std::ceil(std::max(lastMove, kinematicMove) / kPassSpacing))));
  int         next   = steps - 1 - (passes - 1) * (steps / passes);
  float       tFaces = 0;

  const auto      frameStartTime = std::chrono::steady_clock::now();
  VkCommandBuffer cmd            = app->createTempCmdBuffer();
  const auto      groups = [](uint32_t count) { return (count + DFORCE_WORKGROUP_SIZE - 1) / DFORCE_WORKGROUP_SIZE; };
  const auto      barrier = [&] {
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  };
  shaderio::DforcePushConstant pc{};
  pc.model        = reinterpret_cast<shaderio::DforceModel*>(modelBuf.address);
  pc.dt           = dt;
  pc.extraDamping = extraDamping;
  // one sweep: the colours in [first, last), then the tethers
  const auto sweep = [&](size_t first, size_t last) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, edgesPipe);
    for(size_t c = first; c < last && c + 1 < colourStart.size(); c++)
    {
      pc.edgeStart = uint32_t(colourStart[c]);
      pc.edgeCount = uint32_t(colourStart[c + 1] - colourStart[c]);
      if(pc.edgeCount == 0)
        continue;
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmd, groups(pc.edgeCount), 1, 1);
      barrier();
    }
    if(m.numSoft > 0)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, attachPipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmd, groups(m.numSoft), 1, 1);
      barrier();
    }
  };
  const size_t allColours = colourStart.size() - 1;
  // resting vertices held on their faces where the pose has them at pc.t
  const auto hold = [&] {
    if(m.numTris == 0 || (debug & 2))
      return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, holdPipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups(m.numFree), 1, 1);
    barrier();
  };
  // the frame's start: cloth lost behind a face of a lower layer is put back, as on the CPU
  if(m.numTris > 0 && opts.recoverLost && !(debug & 1))
  {
    pc.t = 0;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, recoverPipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups(m.numFree), 1, 1);
    barrier();
  }
  for(int s = 0; s < steps; s++)
  {
    pc.t      = float(s + 1) / float(steps);
    pc.tFaces = tFaces;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integratePipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups(std::max({m.numKinematic, m.numFree, m.numEdges, m.numSoft})), 1, 1);
    barrier();
    for(int it = 0; it < std::max(1, opts.iterations); it++)
      sweep(0, allColours);
    for(int it = 0; it < opts.stretchSweeps; it++)
      sweep(0, size_t(stretchColours));
    hold();
    if(s == next && m.numTris > 0)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, collidePipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmd, groups(m.numFree), 1, 1);
      barrier();
      // the pushes stretched edges; one more sweep shares them out before velocities are read
      sweep(0, allColours);
      hold();
      tFaces = pc.t;
      next += std::max(1, steps / passes);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, velocityPipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups(m.numFree), 1, 1);
    barrier();
  }
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  const VkBufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = n * 3 * sizeof(float)};
  vkCmdCopyBuffer(cmd, x.buffer, readback.buffer, 1, &copy);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_HOST_BIT);
  app->submitAndWaitTempCmdBuffer(cmd);

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
  }
}

DforceGpu::DforceGpu(nvvk::ResourceAllocator* alloc, nvapp::Application* app, std::vector<SettleMesh> meshes, const SettleOptions& opts)
    : m_impl(std::make_unique<Impl>())
{
  m_impl->alloc  = alloc;
  m_impl->app    = app;
  m_impl->meshes = std::move(meshes);
  m_impl->opts   = opts;
  m_valid        = alloc && app && m_impl->init();
  if(!m_valid)
    m_impl->destroy();
}

DforceGpu::~DforceGpu()
{
  m_impl->destroy();
}

void DforceGpu::frame(float t0, float t1, float extraDamping)
{
  if(!m_valid || m_impl->broken)
    return;
  m_impl->frame(t0, t1, extraDamping);
  m_lastMove = m_impl->broken ? 0.f : m_impl->lastMove;
}

void DforceGpu::setTarget(size_t mesh, const std::vector<Mat4>& bones)
{
  Impl& s = *m_impl;
  if(mesh >= s.kin.size() || !s.kin[mesh].skinned || bones.size() != s.kin[mesh].to.size())
    return;
  s.kin[mesh].from = s.kin[mesh].to;
  for(size_t b = 0; b < bones.size(); b++)
    s.kin[mesh].to[b] = poseOf(bones[b]);
  targetSet();
}

void DforceGpu::settle()
{
  if(!m_valid)
    return;
  const auto start  = std::chrono::steady_clock::now();
  const int  frames = std::max(1, int(std::lround(m_impl->transition / m_impl->opts.frameDt)));
  int        run    = 0;
  for(int f = 0; f < frames; f++, run++)
    frame(float(f) / float(frames), float(f + 1) / float(frames));
  const int hold = std::max(1, int(std::ceil(m_impl->opts.maxHold / m_impl->opts.frameDt)));
  for(int h = 0; h < hold; h++)
  {
    frame(1, 1, kHoldDamping);
    run++;
    if(m_lastMove < m_impl->opts.restMove)
      break;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  LOGI("[fv-time] dforceGpuSettle %d frames %.1f ms (%.1f ms a frame, %d substeps)\n", run, ms, ms / std::max(1, run), m_impl->substeps);
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
