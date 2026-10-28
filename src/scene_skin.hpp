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

// DAZ's skinning (daz-viewer docs/gpu-figures-plan.md F3): a joint's matrix taken apart for dual
// quaternion skinning, as shaders/skinning.comp.slang reads it.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace rs {

// A joint's matrix as a rotation and translation (a unit dual quaternion, x y z w each) and what is
// left of it, the stretch (scale and shear: the matrix's 3x3 is the rotation times the stretch), as
// rows. The skin pass blends the stretches linearly and the dual quaternions, then stretches, turns
// and moves (Kavan's two-phase scheme); a joint without scale has the identity as its stretch.
struct JointDualQuat
{
  glm::vec4 real{0.f, 0.f, 0.f, 1.f}, dual{0.f};
  glm::vec4 stretch[3]{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
};
static_assert(sizeof(JointDualQuat) == 80, "five float4s in the shader");

// The rotation is the 3x3's columns made orthonormal in order (Gram-Schmidt), right-handed; a
// joint scaled to nothing keeps the identity rotation and all of the matrix as its stretch.
inline JointDualQuat jointDualQuat(const glm::mat4& m)
{
  const glm::mat3 a(m);
  glm::mat3       r(1.f);
  const float     l0 = glm::length(a[0]);
  const glm::vec3 c1 = l0 > 0.f ? a[1] - glm::dot(a[0] / l0, a[1]) * (a[0] / l0) : glm::vec3(0.f);
  const float     l1 = glm::length(c1);
  if(l0 > 1e-12f && l1 > 1e-12f)
  {
    r[0] = a[0] / l0;
    r[1] = c1 / l1;
    r[2] = glm::cross(r[0], r[1]);
  }
  const glm::quat q = glm::normalize(glm::quat_cast(r));
  const glm::mat3 k = glm::transpose(r) * a;
  const glm::vec3 t(m[3]);
  const glm::quat d = glm::quat(0.f, t.x, t.y, t.z) * q * 0.5f;
  JointDualQuat   out;
  out.real = glm::vec4(q.x, q.y, q.z, q.w);
  out.dual = glm::vec4(d.x, d.y, d.z, d.w);
  for(int row = 0; row < 3; row++)
    out.stretch[row] = glm::vec4(k[0][row], k[1][row], k[2][row], 0.f);
  return out;
}

}  // namespace rs
