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

// The dForce cloth solver on the GPU: experimental, built with FV_DFORCE_GPU. The same ClothModel
// the CPU solver builds (figure/dforce/solver.hpp) as buffers, stepped by the compute kernels of
// shaders/dforce_cloth.comp.slang. A frame is one command buffer: the kinematic vertices' start and
// end positions and the body's collision grid go up (the host skins the kinematic vertices and
// builds the grid, as the CPU solver does), the substeps are recorded, the positions come back.
// It runs on the thread that owns the queue (ownerThreadOnly), submitted and waited on at once, so
// a step costs the frame its GPU time; the proper pipelining into the renderer's frame is left
// for when it has been measured.
//
// Not done here: the cloth does not collide with itself, vertices keep no clearance budgets (every
// free vertex looks around on every pass; the GPU has the threads for it), and the kinematic
// vertices move along straight lines within a frame (the host sweeps them along their bones
// between frames). What the two share: the model, the tethers of partial strength, the sweeps over
// the stretch edges, the start separated from the body, the sweep after a collision pass, a vertex
// lost behind a face of a lower layer put back at a frame's start and a resting vertex held on its
// face every substep (recoverMain, holdMain), and a live step that holds the pose reached until a
// new target comes (ClothBackend::step).
//
// What the frames needn't wait for is made on the pool: the model, the colours and the start, which on a hair of
// 340,000 vertices take seconds. The buffers and pipelines come on the owner's thread once that is done, at the first
// call that needs them, and until then settleSome() says it isn't done and costs nothing.
//
// A frame that takes over 250 ms is logged with what it saw (such a frame is the one before a
// device loss: Windows resets a GPU held for about two seconds). FV_DFORCE_GPU_DEBUG in the
// environment leaves a kernel out, for bisecting one: 1 the recovery, 2 the hold.

#include <functional>
#include <future>
#include <memory>
#include <vector>

#include <nvvk/resource_allocator.hpp>

#include <figure/dforce/solver.hpp>

namespace nvapp {
class Application;
}

class DforceGpu final : public fv::dforce::ClothBackend
{
public:
  // How it runs, for the GPU cloth bench (src/tools/fv_cloth_bench.cpp, physics-plan step 13a); the app
  // takes the defaults.
  struct Options
  {
    // >0: the cluster solve, a workgroup a cluster of about this many free vertices, at most `clusterLocal`
    // with the pinned ones its rows read (figure/physics/cloth/clusters.hpp); 0: a dispatch a colour
    int  clusterSize  = 0;
    int  clusterLocal = 2048;
    // the rows that cross clusters: held by both clusters, each moving its own end (one dispatch a
    // sweep in all), or after the clusters, a dispatch a colour of them
    enum class Crossing
    {
      eHalo,
      eColoured,
    } crossing = Crossing::eHalo;
    bool timestamps   = false;  // time each frame on the GPU (stats().gpuMs)
  };
  // What it has done so far.
  struct Stats
  {
    int    frames = 0, dispatches = 0;
    double gpuMs = 0, waitMs = 0;
    // of gpuMs: the collision passes and the frame's recovery, and the holds on faces
    double collideMs = 0, holdMs = 0;
    // the cluster solve's shape (0 without it): clusters, colours a cluster runs, the rows that cross
    // and their colours
    int    clusters = 0, clusterColours = 0, crossingColours = 0, substeps = 0, mostLocal = 0;
    size_t crossingRows = 0, haloRows = 0, rows = 0;
  };

  DforceGpu(nvvk::ResourceAllocator* alloc, nvapp::Application* app, std::vector<fv::dforce::SettleMesh> meshes, const fv::dforce::SettleOptions& opts,
            const Options& gpu = {});
  ~DforceGpu() override;
  DforceGpu(const DforceGpu&)            = delete;
  DforceGpu& operator=(const DforceGpu&) = delete;

  void  frame(float t0, float t1, float extraDamping = 0, float seconds = 0) override;
  void  setTarget(size_t mesh, const std::vector<fv::Mat4>& bones) override;
  void  setGravity(const std::array<float, 3>& gravity) override;
  void  settle() override;
  bool  settleSome(double seconds) override;
  bool  failed() const override;
  float lastMove() const override { return m_lastMove; }
  std::vector<std::vector<float>> positions() const override;
  bool                            ownerThreadOnly() const override { return true; }

  // False: there is no device to run on, and every call is a no-op. Otherwise whether it can start is known
  // later: failed() and onStartFailed.
  bool valid() const { return m_valid || m_prepared.valid(); }

  // It could not start, for whoever would say so: called on the owner's thread
  std::function<void()> onStartFailed;

  Stats stats() const;

private:
  // What was made on the pool, into buffers. `wait`: for it, if it isn't done; otherwise false until it is.
  bool started(bool wait);

  struct Impl;
  std::shared_ptr<Impl> m_impl;  // the pool's job holds it too: what it is making outlives a settle an edit dropped
  std::future<bool>     m_prepared;
  bool                  m_valid    = false;
  float                 m_lastMove = 0;
};
