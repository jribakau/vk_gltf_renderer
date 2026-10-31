/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compute shader for skeletal skinning with up to 4 joint influences per vertex (8 with a second
// set). Transforms positions, normals, and tangents using joint matrices; with the joints' dual
// quaternions, mixes that linear blend with dual quaternion skinning by the vertex's share (DAZ's
// skin settings): the joints' stretches blended linearly, then the blended unit dual quaternion
// turns and moves (Kavan's two-phase scheme). tests/common/skin_reference.hpp runs the same floats.

#include "animation_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<SkinPushConstant> pc;

// v turned by the unit quaternion q (x, y, z, w)
// A joint scaled to zero leaves a normal of no length: normalize would make it NaN, which a sum spreads
// and DLSS keeps. It stays as it was made.
float3 normalizeOr(float3 v, float3 fallback)
{
  const float lengthSq = dot(v, v);
  return lengthSq > 1e-20 ? v / sqrt(lengthSq) : fallback;
}

float3 quatRotate(float4 q, float3 v)
{
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

[shader("compute")]
[numthreads(ANIMATION_WORKGROUP_SIZE, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint v = tid.x;
  if(v >= pc.vertexCount)
    return;

  float4 w = pc.weights[v];
  int4   j = pc.joints[v];

  bool hasNormals  = (pc.baseNormals != nullptr);
  bool hasTangents = (pc.baseTangents != nullptr);

  float3 skinnedPos = float3(0);
  float3 skinnedNrm = float3(0);
  float3 skinnedTan = float3(0);

  int  sets = pc.joints1 != nullptr && pc.weights1 != nullptr ? 2 : 1;
  for(int s = 0; s < sets; ++s)
  {
    if(s == 1)
    {
      w = pc.weights1[v];
      j = pc.joints1[v];
    }
    [unroll]
    for(int i = 0; i < 4; ++i)
    {
      float jw = w[i];
      if(jw > 0.0 && j[i] >= 0 && uint(j[i]) < pc.numJoints)
      {
        int ji = j[i];
        skinnedPos += jw * mul(float4(pc.basePositions[v], 1.0), pc.jointMatrices[ji]).xyz;

        if(hasNormals)
          skinnedNrm += jw * mul(pc.baseNormals[v], pc.normalMatrices[ji]);

        if(hasTangents)
          skinnedTan += jw * mul(pc.baseTangents[v].xyz, (float3x3)pc.jointMatrices[ji]);
      }
    }
  }

  float share = pc.jointDualQuats == nullptr ? 0.0 : (pc.dualQuatShares != nullptr ? pc.dualQuatShares[v] : pc.dualQuat);
  if(share <= 0.0)
  {
    pc.outPositions[v] = skinnedPos;

    if(hasNormals)
      pc.outNormals[v] = normalizeOr(skinnedNrm, pc.baseNormals[v]);

    if(hasTangents)
      pc.outTangents[v] = float4(normalizeOr(skinnedTan, pc.baseTangents[v].xyz), pc.baseTangents[v].w);
    return;
  }

  // the dual quaternions and stretches blended, each rotation on the first one's side
  float4 b0 = float4(0), b1 = float4(0), first = float4(0);
  float3 k0 = float3(0), k1 = float3(0), k2 = float3(0);
  bool   met = false;
  for(int s = 0; s < sets; ++s)
  {
    w = s == 0 ? pc.weights[v] : pc.weights1[v];
    j = s == 0 ? pc.joints[v] : pc.joints1[v];
    for(int i = 0; i < 4; ++i)
    {
      float jw = w[i];
      if(jw > 0.0 && j[i] >= 0 && uint(j[i]) < pc.numJoints)
      {
        uint   at   = uint(j[i]) * 5;
        float4 real = pc.jointDualQuats[at];
        if(!met)
          first = real;
        met       = true;
        float sw  = dot(real, first) < 0.0 ? -jw : jw;
        b0 += sw * real;
        b1 += sw * pc.jointDualQuats[at + 1];
        k0 += jw * pc.jointDualQuats[at + 2].xyz;
        k1 += jw * pc.jointDualQuats[at + 3].xyz;
        k2 += jw * pc.jointDualQuats[at + 4].xyz;
      }
    }
  }
  float  len = length(b0);
  float4 q   = len > 0.0 ? b0 / len : float4(0, 0, 0, 1);
  float4 d   = len > 0.0 ? b1 / len : float4(0);
  float3 p   = pc.basePositions[v];
  float3 dqPos = quatRotate(q, float3(dot(k0, p), dot(k1, p), dot(k2, p))) + 2.0 * (q.w * d.xyz - d.w * q.xyz + cross(q.xyz, d.xyz));
  pc.outPositions[v] = skinnedPos * (1.0 - share) + dqPos * share;

  if(hasNormals)
  {
    // a normal goes by the stretch's inverse transpose: its cofactors, on the determinant's side
    float3 n     = pc.baseNormals[v];
    float3 c0    = cross(k1, k2), c1 = cross(k2, k0), c2 = cross(k0, k1);
    float  side  = dot(k0, c0) < 0.0 ? -1.0 : 1.0;
    float3 dqNrm = quatRotate(q, side * float3(dot(c0, n), dot(c1, n), dot(c2, n)));
    pc.outNormals[v] = normalizeOr(normalizeOr(skinnedNrm, n) * (1.0 - share) + normalizeOr(dqNrm, n) * share, n);
  }

  if(hasTangents)
  {
    float3 t     = pc.baseTangents[v].xyz;
    float3 dqTan = quatRotate(q, float3(dot(k0, t), dot(k1, t), dot(k2, t)));
    pc.outTangents[v] = float4(normalizeOr(normalizeOr(skinnedTan, t) * (1.0 - share) + normalizeOr(dqTan, t) * share, t), pc.baseTangents[v].w);
  }
}
