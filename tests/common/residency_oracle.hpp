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
// ResidencyOracle (docs: daz-viewer docs/retained-scene-plan.md, "The residency oracle"): after a
// sync, what the fake GPU holds, walked against the retained scene's CPU truth. Each check names
// its invariant. Run it after every sync of every residency test.
//
// Checked here so far: I1 (TLAS rows, their mask and flags), I2 (custom index), I3 (render-node and
// prim rows, addresses in live buffers), I4 (material rows, texture and sampler slots, sRGB), I13
// (capacity at or above the high-water mark), I17 (BLAS input alignment), the geometry bytes, every
// BLAS built or refit from the bytes the buffers hold now, and the raster lists. I5's leak half needs
// the retire queue drained: residencyLeaks().
//

#include <string>
#include <vector>

#include "fake_gpu_backend.hpp"
#include "scene_residency.hpp"
#include "scene_retained.hpp"

namespace rs {

// What is wrong, one line each; empty when nothing is.
std::vector<std::string> checkResidency(RetainedScene& scene, const Residency& residency, const FakeGpuBackend& gpu);

// Once nothing is waiting to retire (beginFrame() run framesInFlight() times since the last sync):
// a resource the fake still holds that the residency layer doesn't is a leak.
std::vector<std::string> residencyLeaks(const Residency& residency, const FakeGpuBackend& gpu);

}  // namespace rs
