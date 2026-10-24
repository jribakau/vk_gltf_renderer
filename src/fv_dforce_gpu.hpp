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
// Not done here: the cloth does not collide with itself, and vertices keep no clearance budgets or
// contact caches (every free vertex looks around on every pass; the GPU has the threads for it).
//
// This backend was written without a GPU at hand: it has not been compiled against the Vulkan SDK
// or run. Treat it as a draft to bring up, not a result.

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
  DforceGpu(nvvk::ResourceAllocator* alloc, nvapp::Application* app, std::vector<fv::dforce::SettleMesh> meshes, const fv::dforce::SettleOptions& opts);
  ~DforceGpu() override;
  DforceGpu(const DforceGpu&)            = delete;
  DforceGpu& operator=(const DforceGpu&) = delete;

  void  frame(float t0, float t1, float extraDamping = 0) override;
  void  setTarget(size_t mesh, const std::vector<fv::Mat4>& bones) override;
  void  settle() override;
  float lastMove() const override { return m_lastMove; }
  std::vector<std::vector<float>> positions() const override;
  bool                            ownerThreadOnly() const override { return true; }

  // Whether the pipelines and buffers were made; false leaves every call a no-op.
  bool valid() const { return m_valid; }

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
  bool                  m_valid    = false;
  float                 m_lastMove = 0;
};
