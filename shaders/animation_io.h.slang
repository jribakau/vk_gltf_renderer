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

#ifndef ANIMATION_IO_H
#define ANIMATION_IO_H

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define ANIMATION_WORKGROUP_SIZE 256

// Normals and tangents, in and out, are the vertex streams' (unit_vector_pack.h.slang): packUnitVector
// and packTangent, 4 bytes each
struct SkinPushConstant
{
  float3*   basePositions;
  uint*     baseNormals;
  uint*     baseTangents;
  float4*   weights;
  int4*     joints;
  float4x4* jointMatrices;
  float3x3* normalMatrices;
  float3*   outPositions;
  uint*     outNormals;
  uint*     outTangents;
  // DAZ's skinning (figures; null otherwise): influences five to eight, the joints'
  // dual quaternions (rs::JointDualQuat, five float4s each; null: linear only) and the dual
  // quaternion share per vertex (null: dualQuat for every vertex)
  int4*     joints1;
  float4*   weights1;
  float4*   jointDualQuats;
  float*    dualQuatShares;
  uint      vertexCount;
  uint      numJoints;
  float     dualQuat;
};

struct MorphPushConstant
{
  float3* basePositions;
  uint*   baseNormals;   // packed, as the skin's
  uint*   baseTangents;
  float3* positionDeltas;
  float3* normalDeltas;
  float3* tangentDeltas;
  float*  morphWeights;
  float3* outPositions;
  uint*   outNormals;
  uint*   outTangents;
  uint    vertexCount;
  uint    numTargets;
};

// stencil.comp.slang: rowCount outputs, each the weighted sum of the base elements columns[rowStart[r]
// .. rowStart[r + 1]) names (weights null: each 1), or, with no rowStart, the one base element
// columns[r] names
struct StencilPushConstant
{
  float3* baseValues;
  float3* outValues;
  uint*   rowStart;
  uint*   columns;
  float*  weights;
  float3* offsets;  // per row, added to its sum; null: none
  // per source element, the axes its offset's X and Y turned to (the skin's tangent and normal of the rest's
  // axes, packed as the skin writes them); a row's offset turns as its first column does. Null: as it is.
  uint*   frameX;
  uint*   frameY;
  // set: each row's value goes here as a unit direction (packUnitVector), not into outValues: the drawn
  // normals gathered from the refined ones
  uint*   outUnit;
  // DAZ HD detail (figure/subdiv/hd_edits.hpp): row r's edits are hdStart[r] .. hdStart[r + 1], each an
  // offset in the frame of the cage face corner hdFrame names, made from the five cage vertices
  // hdFrameVertices lists for it (in hdCage, the cage the levels are refined from). Null: none.
  uint*   hdStart;
  uint*   hdFrame;
  float3* hdOffsets;
  float3* hdCage;
  int*    hdFrameVertices;
  uint    rowCount;
  uint    sourceCount;
};

// vertex_normals.comp.slang: per vertex, the unit sum of the cross products of the triangles
// triangleList[triangleStart[v] .. triangleStart[v + 1]) names
struct NormalsPushConstant
{
  float3* positions;
  uint3*  triangles;
  uint*   triangleStart;
  uint*   triangleList;
  float3* outNormals;
  uint    vertexCount;
  uint    triangleCount;
  uint    positionCount;
};

// vertex_tangents.comp.slang: per vertex, the UV tangents of the triangles (over the vertices) whose
// corners corners[cornerStart[v] .. cornerStart[v + 1]) names (3 * triangle + corner), across its
// normal; uvs null: each triangle's tangent is made from its first normal
struct TangentsPushConstant
{
  float3* positions;
  uint*   normals;  // the drawn normals, packed
  float2* uvs;
  uint3*  triangles;
  uint*   cornerStart;
  uint*   corners;
  uint*   outTangents;  // packTangent
  uint    vertexCount;
  uint    triangleCount;
};

NAMESPACE_SHADERIO_END()

#endif
