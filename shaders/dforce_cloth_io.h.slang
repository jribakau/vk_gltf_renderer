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

// The dForce cloth solver on the GPU (shaders/dforce_cloth.comp.slang, src/fv_dforce_gpu.cpp): the
// same model the CPU solver builds (figure/dforce/solver.hpp, ClothModel), as buffers. Everything
// static sits behind one DforceModel the push constant points at; the push constant carries what
// changes between dispatches.

#ifndef DFORCE_CLOTH_IO_H
#define DFORCE_CLOTH_IO_H

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define DFORCE_WORKGROUP_SIZE 256
// The cluster solve (physics-plan step 13a, figure/physics/cloth/clusters.hpp): a workgroup a cluster,
// holding its vertices (its own free ones and the pinned ghosts its rows read) in shared memory
#define DFORCE_CLUSTER_THREADS 256
#define DFORCE_CLUSTER_LOCAL 2048
// kind: 0 stretch, 1 shear, 2 bend
#define DFORCE_EDGE_BEND 2
#define DFORCE_EDGE_REACH 3

struct DforceEdge
{
  int   a, b;
  float rest, alpha;
  int   kind;
  float ratio, buckle, pad;
};

struct DforceTri
{
  int   a, b, c, layer;
  float offset, friction;
  int   mesh;  // a cloth face's mesh, and whether its own vertices meet it (Self Collide)
  int   self;
};

// A cell of the collision grid, open-addressed by the hash of its coordinates: `start` is -1 while
// the slot is empty, else where its faces begin in `items` and how many there are.
struct DforceCell
{
  int x, y, z, start;
  int count, pad0, pad1, pad2;
};

struct DforceModel
{
  float3* x;
  float3* prev;
  float3* vel;
  float3* cprev;  // where each free vertex was at its last collision pass: decides the side it is on
  float3* kinStart;  // kinematic and soft vertices at the frame's start and end, by vertex index
  float3* kinEnd;
  int*    kinematic;  // the vertices the pose moves
  int*    freeVerts;  // the vertices the cloth moves
  float*  inv;
  float*  dampRate;
  float*  offset;
  float*  friction;
  int*    layer;
  int*    collides;
  float3* grav;
  DforceEdge* edges;  // sorted by colour; no two edges of a colour share a vertex
  float*      lam;
  DforceTri*  tris;  // the faces the pose moves (the body)
  DforceCell* cells;
  int*        items;
  // The cloth's own faces: a garment meets another's (and its own with Self Collide, `self`, which the host
  // leaves off for now; not those around a vertex: ring2, its stretch neighbours and theirs, sorted, from
  // ring2Start). Their grid is made from
  // where they were at the frame's start (`frameStart`, every vertex); a collision pass reads them from
  // `snap` and `cprevSnap`, the positions and the last pass's as the pass began, so what one vertex
  // finds doesn't hang on how far the others have got
  DforceTri*  clothTris;
  DforceCell* clothCells;
  int*        clothItems;
  int*        meshOf;
  int*        ring2Start;
  int*        ring2;
  float3*     frameStart;
  float3*     snap;
  float3*     cprevSnap;
  int*        softVerts;  // the free vertices of partial strength, tethered to where the pose would put them
  float*      attAlpha;   // each one's tether compliance, by vertex index (0 for the rest)
  float*      lamAtt;     // and its multiplier, by soft index
  int*        contact;    // the face each free vertex rests on (set by the last collision pass, or a recovery), or -1
  // the cluster solve, when there is one: `edges` above are then only the rows that cross clusters.
  // Cluster c's vertices are clusterVerts[clusterVertStart[c] ..], its own first (clusterOwned[c] of
  // them); its colour k is clusterEdges[clusterColourStart[c * clusterColours + k] ..], each row's ends
  // local indices into those vertices; its tethered vertices clusterSoft[clusterSoftStart[c] ..] (x the
  // local index, y the soft index)
  DforceEdge* clusterEdges;
  float*      clusterLam;
  int*        clusterVerts;
  int*        clusterVertStart;
  int*        clusterOwned;
  int*        clusterColourStart;
  uint2*      clusterSoft;
  int*        clusterSoftStart;
  uint        nv, numKinematic, numFree, numEdges, numTris, cellMask, numSoft, numClusterEdges;
  uint        numClothTris, clothCellMask, pad0, pad1;
  uint        numClusters, clusterColours, clusterStretch, clusterReach;
  float       cellSize, depth, cap, maxSpeed, searchReach, recoverDepth;
};

struct DforcePushConstant
{
  DforceModel* model;
  uint         edgeStart, edgeCount;  // this dispatch's colour
  float        dt;                    // the substep
  float        t;                     // where in the frame the kinematic vertices are at the substep's end
  float        tFaces;                // where they were at the last collision pass
  float        extraDamping;
  uint         grip;                  // a hold that applies friction
  uint         lengthOnly;            // a cluster sweep over the rows that hold length (else a full sweep)
  // a cluster sweep reads the free vertices from one buffer and writes its own to the other (x and its
  // twin, in turn), so what it reads of another cluster's is where that was at the sweep's start
  float3*      clusterSrc;
  float3*      clusterDst;
};

NAMESPACE_SHADERIO_END()

#endif
