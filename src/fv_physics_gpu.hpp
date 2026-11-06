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

// The cloth solver on the GPU, built with FV_PHYSICS_GPU: the physics core's cloth on the GPU, a
// backend of the cloth driver's (physics/cloth/backend.hpp). The same cloth::Model the CPU island
// builds (figure/physics/cloth/model.hpp) as buffers, stepped by the compute kernels of
// shaders/physics_cloth.comp.slang. A frame is one command buffer: the kinematic vertices' start and
// end positions and the body's collision grid go up (the host skins the kinematic vertices and
// builds the grid, as the CPU solver does), the substeps are recorded, the positions come back.
// A frame is submitted and waited on at once. On a queue of its own (useQueue: a second queue the
// device gives, compute only on NVIDIA's) any thread may drive it, and the driver settles on the pool:
// the thread that draws never waits on it, and the cloth's compute runs beside the renderer's frames
// (physics-plan step 13b). Without one it runs on the thread that owns the renderer's queue
// (ownerThreadOnly), and a step costs that frame its GPU time.
//
// It solves in clusters (figure/physics/cloth/clusters.hpp, physics-plan step 13a): a workgroup holds a
// patch of the cloth in shared memory for a whole sweep, 10 dispatches a substep instead of about 145.
//
// A collision pass and a frame's recovery go a wave a vertex where the device's waves are 32 lanes (NVIDIA's):
// each face's box where the pose has it is made first, and a vertex's lanes look at 32 of its faces at once
// on those boxes, meeting the few it may touch one after another in the order a thread a vertex met them,
// so the drape is the same to the bit (physics-plan step 13c). A thread a vertex walked its hundreds of
// faces alone, and with a few thousand vertices most of the GPU stood idle.
//
// Cloth meets cloth as the CPU island has it: every fourth collision pass looks around for another
// garment's faces (and a garment's own, with Self Collide, past the two rings about a vertex), the others
// meet the one face a vertex rests on; each side of such a contact takes half the correction, since both
// move. Looked for every pass, a crumpled dress jammed (a vertex within the gap of 30 of its own faces,
// each push rewriting its velocity) and a full push from both sides threw vertices through the body.
//
// Not done here: vertices keep no clearance budgets (every free vertex looks around on every pass; the
// GPU has the threads for it), and the kinematic
// vertices move along straight lines within a frame (the host sweeps them along their bones
// between frames). What the two share: the model, the tethers of partial strength, the sweeps over
// the stretch edges, the start separated from the body, the sweep after a collision pass, a vertex
// lost behind a face of a lower layer put back at a frame's start and a resting vertex held on its
// face every substep (recoverMain, holdMain), and a live step that holds the pose reached until a
// new target comes (cloth::Backend::step).
//
// What the frames needn't wait for is made on the pool: the model, the colours and the start, which on a hair of
// 340,000 vertices take seconds. The buffers and pipelines come on the owner's thread once that is done, at the first
// call that needs them, and until then settleSome() says it isn't done and costs nothing.
//
// The driver keeps a figure's simulation between settles (restart, physics-plan step 13b): for the same garments,
// shapes and settings, only the pose and gravity new, it keeps its model, its buffers and its kernels and puts the
// state back to the start, so a pose edit's drape begins at once instead of half a second later. A body's new rest
// cage (its contact dents change with the pose) goes into the model as it is, and the start is made again from it.
// Other shapes (a dial turned) make the model again on the pool, with the kernels kept.
//
// A frame that takes over 250 ms is logged with what it saw (such a frame is the one before a
// device loss: Windows resets a GPU held for about two seconds). FV_PHYSICS_GPU_DEBUG in the
// environment leaves a kernel out, for bisecting one: 1 the recovery, 2 the hold; 4 solves a dispatch
// a colour and 8 on the renderer's queue, as before step 13; 16 makes a new simulation for every settle,
// as before the driver kept one; 256 runs the collision passes and the recovery a thread a vertex; 512 stops
// a simulation after a frame over 250 ms, for trying a kernel that may run away.

#include <functional>
#include <future>
#include <memory>
#include <vector>

#include <nvvk/resource_allocator.hpp>

#include <figure/physics/cloth/backend.hpp>

namespace nvapp {
class Application;
}

class ClothGpu final : public fv::physics::cloth::Backend
{
public:
  // How it runs, for the GPU cloth bench (src/tools/fv_cloth_bench.cpp, physics-plan step 13a); the app
  // takes the defaults.
  struct Options
  {
    // >0: the cluster solve, a workgroup a cluster of about this many free vertices, at most `clusterLocal`
    // with the pinned ones its rows read (figure/physics/cloth/clusters.hpp); 0: a dispatch a colour
    int  clusterSize  = 1024;
    int  clusterLocal = 2048;
    // the rows that cross clusters: held by both clusters, each moving its own end (one dispatch a
    // sweep in all), or after the clusters, a dispatch a colour of them
    enum class Crossing
    {
      eHalo,
      eColoured,
    } crossing = Crossing::eHalo;
    bool timestamps   = false;  // time each frame on the GPU (stats().gpuMs)
    // a garment meets another garment's faces, and its own where it collides with itself (Self Collide), as
    // on the CPU island (physics-plan step 13d). FV_PHYSICS_GPU_DEBUG's 64 leaves them out, 128 a garment's own
    bool clothCollide = true;
  };
  // What it has done so far.
  struct Stats
  {
    int    frames = 0, dispatches = 0;
    double gpuMs = 0, waitMs = 0;
    // of gpuMs: the collision passes and the frame's recovery (of which the recovery alone), and the holds on faces
    double collideMs = 0, recoverMs = 0, holdMs = 0;
    // on the host, each frame: the kinematic vertices skinned, and the collision grid built (of which
    // the copy into its buffers), with its entries (a face once per cell its sweep meets) and slots
    double skinMs = 0, gridMs = 0, gridCopyMs = 0;
    size_t gridEntries = 0, gridSlots = 0;
    // the cloth's own faces that collide (0: none can meet a vertex that may meet them), and their grid
    // on the host each frame
    size_t clothFaces = 0;
    double clothGridMs = 0;
    // the cluster solve's shape (0 without it): clusters, colours a cluster runs, the rows that cross
    // and their colours
    int    clusters = 0, clusterColours = 0, crossingColours = 0, substeps = 0, mostLocal = 0;
    size_t crossingRows = 0, haloRows = 0, rows = 0;
  };

  ClothGpu(nvvk::ResourceAllocator* alloc, nvapp::Application* app, std::vector<fv::physics::cloth::Surface> meshes, const fv::physics::cloth::Options& opts,
            const Options& gpu = {});
  ~ClothGpu() override;
  ClothGpu(const ClothGpu&)            = delete;
  ClothGpu& operator=(const ClothGpu&) = delete;

  void  frame(float t0, float t1, float extraDamping = 0, float seconds = 0) override;
  void  setTarget(size_t mesh, const std::vector<fv::Mat4>& bones) override;
  bool  setRest(size_t mesh, std::vector<float> restLocal) override;
  bool  carry(float t) override;
  void  setGravity(const std::array<float, 3>& gravity) override;
  void  settle() override;
  bool  settleSome(double seconds) override;
  bool  failed() const override;
  float lastMove() const override { return m_lastMove; }
  std::vector<std::vector<float>> positions() const override;
  bool                            ownerThreadOnly() const override;
  // cloth::Backend's: true unless it failed. The same shapes keep the model; others make it again on the pool.
  bool                            restart(std::vector<fv::physics::cloth::Surface> meshes, const fv::physics::cloth::Options& opts) override;

  // The queue every simulation made after this runs on, when the device has one beside the renderer's;
  // given once at start-up. Without it a simulation runs on the renderer's queue, on its thread.
  static void useQueue(const nvvk::QueueInfo& queue);

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
