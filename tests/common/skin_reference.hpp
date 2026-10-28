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

// One vertex skinned as shaders/skinning.comp.slang skins it, in floats: the fake backend runs the
// skin pass with it and the residency oracle checks the fake's output against it, so the two agree
// bit for bit (the GPU to float rounding).

#include <span>

#include <glm/glm.hpp>

#include "scene_skin.hpp"

namespace rs {

struct SkinJoints
{
  std::span<const glm::mat4>     matrices;
  std::span<const glm::mat3>     normalMatrices;
  std::span<const JointDualQuat> dualQuats;  // none: linear only
};

struct SkinnedVertex
{
  glm::vec3 position{0.f}, normal{0.f};
  glm::vec4 tangent{0.f};
};

// v turned by the unit quaternion q (x, y, z, w)
inline glm::vec3 quatRotate(const glm::vec4& q, const glm::vec3& v)
{
  const glm::vec3 u(q);
  return v + 2.f * glm::cross(u, glm::cross(u, v) + q.w * v);
}

// `joints` and `weights` hold a set of four per entry (two for eight influences); `share` is the
// vertex's dual quaternion share. A null normal or tangent is not skinned.
inline SkinnedVertex skinVertex(const glm::vec3&              p,
                                const glm::vec3*              n,
                                const glm::vec4*              t,
                                std::span<const glm::ivec4>   joints,
                                std::span<const glm::vec4>    weights,
                                const SkinJoints&             with,
                                float                         share)
{
  const uint32_t count = uint32_t(with.matrices.size());
  const auto     valid = [&](size_t s, int i) {
    return weights[s][i] > 0.f && joints[s][i] >= 0 && uint32_t(joints[s][i]) < count;
  };
  glm::vec3 pos(0.f), nrm(0.f), tan(0.f);
  for(size_t s = 0; s < joints.size(); s++)
    for(int i = 0; i < 4; i++)
      if(valid(s, i))
      {
        const float      w = weights[s][i];
        const glm::mat4& m = with.matrices[size_t(joints[s][i])];
        pos += w * glm::vec3(m * glm::vec4(p, 1.f));
        if(n)
          nrm += w * (with.normalMatrices[size_t(joints[s][i])] * *n);
        if(t)
          tan += w * (glm::mat3(m) * glm::vec3(*t));
      }
  SkinnedVertex out;
  if(with.dualQuats.size() != count || share <= 0.f)
  {
    out.position = pos;
    if(n)
      out.normal = glm::normalize(nrm);
    if(t)
      out.tangent = glm::vec4(glm::normalize(tan), t->w);
    return out;
  }

  glm::vec4 b0(0.f), b1(0.f), first(0.f);
  glm::vec3 k0(0.f), k1(0.f), k2(0.f);
  bool      met = false;
  for(size_t s = 0; s < joints.size(); s++)
    for(int i = 0; i < 4; i++)
      if(valid(s, i))
      {
        const float          w = weights[s][i];
        const JointDualQuat& q = with.dualQuats[size_t(joints[s][i])];
        if(!met)
          first = q.real;
        met            = true;
        const float sw = glm::dot(q.real, first) < 0.f ? -w : w;
        b0 += sw * q.real;
        b1 += sw * q.dual;
        k0 += w * glm::vec3(q.stretch[0]);
        k1 += w * glm::vec3(q.stretch[1]);
        k2 += w * glm::vec3(q.stretch[2]);
      }
  const float     len = glm::length(b0);
  const glm::vec4 q   = len > 0.f ? b0 / len : glm::vec4(0.f, 0.f, 0.f, 1.f);
  const glm::vec4 d   = len > 0.f ? b1 / len : glm::vec4(0.f);
  const glm::vec3 qv(q), dv(d);
  const glm::vec3 dqPos =
      quatRotate(q, glm::vec3(glm::dot(k0, p), glm::dot(k1, p), glm::dot(k2, p))) + 2.f * (q.w * dv - d.w * qv + glm::cross(qv, dv));
  out.position = pos * (1.f - share) + dqPos * share;
  if(n)
  {
    const glm::vec3 c0 = glm::cross(k1, k2), c1 = glm::cross(k2, k0), c2 = glm::cross(k0, k1);
    const float     side  = glm::dot(k0, c0) < 0.f ? -1.f : 1.f;
    const glm::vec3 dqNrm = quatRotate(q, side * glm::vec3(glm::dot(c0, *n), glm::dot(c1, *n), glm::dot(c2, *n)));
    out.normal            = glm::normalize(glm::normalize(nrm) * (1.f - share) + glm::normalize(dqNrm) * share);
  }
  if(t)
  {
    const glm::vec3 tt(*t);
    const glm::vec3 dqTan = quatRotate(q, glm::vec3(glm::dot(k0, tt), glm::dot(k1, tt), glm::dot(k2, tt)));
    out.tangent = glm::vec4(glm::normalize(glm::normalize(tan) * (1.f - share) + glm::normalize(dqTan) * share), t->w);
  }
  return out;
}

}  // namespace rs
