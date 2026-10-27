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

#include <optional>
#include <string_view>

//-------------------------------------------------------------------------------------------------------
// TlasPolicy
//
// What a TLAS sync that keeps the instance count does with the existing structure (the `tlasPolicy`
// setting): `build` rebuilds it every time, as NVIDIA's ray tracing best practices advise for a TLAS;
// `update` refits it in place, and builds only when an UPDATE would be invalid. A change in the
// instance count recreates the TLAS under either policy (SceneRtx::rebuildTopLevelAS).
//-------------------------------------------------------------------------------------------------------
enum class TlasPolicy
{
  eBuild,
  eUpdate,
};

inline constexpr const char* tlasPolicyName(TlasPolicy policy)
{
  return policy == TlasPolicy::eBuild ? "build" : "update";
}

inline constexpr std::optional<TlasPolicy> parseTlasPolicy(std::string_view name)
{
  if(name == "build")
    return TlasPolicy::eBuild;
  if(name == "update")
    return TlasPolicy::eUpdate;
  return std::nullopt;
}

// Whether a sync builds rather than updates. `anyActiveStateChanged`: an instance's BLAS reference
// went from null to non-null or back. An UPDATE only refits instances already in the structure, so it
// cannot bring a newly active one in; `update` builds then too.
inline constexpr bool tlasSyncBuilds(TlasPolicy policy, bool anyActiveStateChanged)
{
  return policy == TlasPolicy::eBuild || anyActiveStateChanged;
}
