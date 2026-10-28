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

#include <string_view>

namespace fv {

// Whether the Figure Viewer host lets params.get / params.set reach a registered parameter. Not
// those that replace or write the renderer's scene behind the host's back: `scenefile` loads a file
// over the composed scene (the host would go on addressing it by indices it no longer holds), and
// `savefile` writes it, the host's sun included. The protocol has load, prop.add and export.glb
// for those. Nor those read only at start-up, which a runtime write would leave meaning nothing
// (`residencyMinInstances`, `sceneBackend`).
inline bool hostExposesParameter(std::string_view name)
{
  return name != "scenefile" && name != "savefile" && name != "residencyMinInstances" && name != "sceneBackend";
}

}  // namespace fv
