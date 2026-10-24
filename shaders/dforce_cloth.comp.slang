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

// The dForce cloth solver on the GPU: one substep is integrate, one dispatch per colour of edges,
// a collision pass on some substeps, then velocities. The host (src/fv_dforce_gpu.cpp) records the
// substeps of a frame and reads the positions back. The physics follows the CPU solver
// (figure/dforce/solver.cpp): XPBD with small steps, kinematic vertices swept through the frame,
// the side of a face read from the vertex's last pass, contacts absorbing normal motion.
// Differences: the cloth does not collide with itself here, and every free vertex looks for faces
// on every pass.

#include "dforce_cloth_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<DforcePushConstant> pc;

// Positions of a vertex the pose moves, at fraction t of the frame.
float3 kinematicAt(DforceModel m, int v, float t)
{
  return lerp(m.kinStart[v], m.kinEnd[v], t);
}

// A face's corners at fraction t: its vertices are kinematic.
void faceAt(DforceModel m, DforceTri tri, float t, out float3 a, out float3 b, out float3 c)
{
  a = kinematicAt(m, tri.a, t);
  b = kinematicAt(m, tri.b, t);
  c = kinematicAt(m, tri.c, t);
}

bool faceNormal(float3 a, float3 b, float3 c, out float3 n)
{
  float3 x   = cross(b - a, c - a);
  float  len = length(x);
  n          = len < 1e-8 ? float3(0) : x / len;
  return len >= 1e-8;
}

// Closest point of q on the triangle, the signed distance along n, the distance, and whether the
// closest point is well inside the face.
void closestOn(float3 q, float3 a, float3 b, float3 c, float3 n, out float signedDist, out float dist, out bool interior)
{
  float3 ab = b - a, ac = c - a, aq = q - a;
  float  d00 = dot(ab, ab), d01 = dot(ab, ac), d11 = dot(ac, ac), d20 = dot(ab, aq), d21 = dot(ac, aq);
  float  den = d00 * d11 - d01 * d01;
  float  bv = 0, cv = 0;
  interior = false;
  if(den > 1e-12)
  {
    bv       = (d11 * d20 - d01 * d21) / den;
    cv       = (d00 * d21 - d01 * d20) / den;
    interior = bv > 0.05 && cv > 0.05 && bv + cv < 0.95;
    bv       = clamp(bv, 0.0, 1.0);
    cv       = clamp(cv, 0.0, 1.0);
    if(bv + cv > 1)
    {
      float s = bv + cv;
      bv /= s;
      cv /= s;
    }
  }
  float3 p   = a + ab * bv + ac * cv;
  float3 d   = q - p;
  signedDist = dot(d, n);
  dist       = length(d);
}

uint cellHash(int ix, int iy, int iz)
{
  return (uint(ix) * 73856093u) ^ (uint(iy) * 19349663u) ^ (uint(iz) * 83492791u);
}

int cellOf(float p, float cellSize)
{
  return int(floor(p / cellSize));
}

// The kinematic vertices to their place at the substep's end, the free ones a step of gravity and
// damping, every edge's multiplier to zero.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void integrateMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  uint        k = tid.x;
  if(k < m.numKinematic)
  {
    int i      = m.kinematic[k];
    m.prev[i]  = m.x[i];
    m.x[i]     = kinematicAt(m, i, pc.t);
    m.vel[i]   = float3(0);
  }
  if(k < m.numFree)
  {
    int    i      = m.freeVerts[k];
    float3 v      = m.vel[i];
    float  damped = exp(-(m.dampRate[i] + pc.extraDamping) * pc.dt);
    v             = (v + m.grav[i] * pc.dt) * damped;
    float sp      = length(v);
    if(sp > m.maxSpeed)
      v *= m.maxSpeed / sp;
    m.prev[i] = m.x[i];
    m.x[i]    = m.x[i] + v * pc.dt;
    m.vel[i]  = v;
  }
  if(k < m.numEdges)
    m.lam[k] = 0;
}

// One colour of edges: an XPBD projection each, no two touching a vertex.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void edgesMain(uint3 tid: SV_DispatchThreadID)
{
  if(tid.x >= pc.edgeCount)
    return;
  DforceModel m  = *pc.model;
  uint        e  = pc.edgeStart + tid.x;
  DforceEdge  ed = m.edges[e];
  float       wa = m.inv[ed.a], wb = m.inv[ed.b];
  float       w  = wa + wb;
  if(w <= 0)
    return;
  float3 d   = m.x[ed.b] - m.x[ed.a];
  float  len = length(d);
  if(len < 1e-8)
    return;
  float alpha = ed.alpha;
  if(ed.kind == DFORCE_EDGE_BEND && len < ed.rest * ed.ratio)
    alpha = ed.buckle > 0 ? alpha / ed.buckle : 1e6;
  float  C  = len - ed.rest;
  float3 n  = d / len;
  float  at = alpha / (pc.dt * pc.dt);
  float  dl = (-C - at * m.lam[e]) / (w + at);
  m.lam[e] += dl;
  if(wa > 0)
    m.x[ed.a] = m.x[ed.a] - n * (wa * dl);
  if(wb > 0)
    m.x[ed.b] = m.x[ed.b] + n * (wb * dl);
}

// Pushes a free vertex out of the faces it was outside of at its last pass.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void collideMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.numFree)
    return;
  int i = m.freeVerts[tid.x];
  if(m.collides[i] == 0)
    return;
  float3 x0    = m.x[i];
  float3 was   = m.cprev[i];
  float  moved = length(x0 - was);
  float  reach = m.searchReach;
  float3 lo    = min(x0, was) - reach;
  float3 hi    = max(x0, was) + reach;
  int    layer = m.layer[i];
  float  off   = m.offset[i];
  float  fri   = m.friction[i];
  for(int ix = cellOf(lo.x, m.cellSize); ix <= cellOf(hi.x, m.cellSize); ix++)
    for(int iy = cellOf(lo.y, m.cellSize); iy <= cellOf(hi.y, m.cellSize); iy++)
      for(int iz = cellOf(lo.z, m.cellSize); iz <= cellOf(hi.z, m.cellSize); iz++)
      {
        uint s = cellHash(ix, iy, iz) & m.cellMask;
        // probe the open-addressed table for this cell
        for(uint probe = 0; probe <= m.cellMask; probe++)
        {
          DforceCell cell = m.cells[(s + probe) & m.cellMask];
          if(cell.start < 0)
            break;
          if(cell.x != ix || cell.y != iy || cell.z != iz)
            continue;
          for(int k = 0; k < cell.count; k++)
          {
            DforceTri tri = m.tris[m.items[cell.start + k]];
            float3    a0, b0, c0, a1, b1, c1;
            faceAt(m, tri, pc.tFaces, a0, b0, c0);
            faceAt(m, tri, pc.t, a1, b1, c1);
            float3 n0, n1;
            if(!faceNormal(a0, b0, c0, n0) || !faceNormal(a1, b1, c1, n1))
              continue;
            // the vertex where it is now: an earlier face may have moved it
            float3 p    = m.x[i];
            float  gap  = max(tri.offset, off);
            float  near = min(m.depth, gap + moved);
            float3 blo  = min(a1, min(b1, c1)) - near;
            float3 bhi  = max(a1, max(b1, c1)) + near;
            if(any(p < blo) || any(p > bhi))
              continue;
            float wasSigned, wasDist;
            bool  wasInterior;
            closestOn(was, a0, b0, c0, n0, wasSigned, wasDist, wasInterior);
            int direction;
            if(layer > tri.layer)
            {
              if(wasSigned < -0.05)
                continue;
              direction = 1;
            }
            else if(layer < tri.layer)
            {
              if(wasSigned > 0.05)
                continue;
              direction = -1;
            }
            else
              direction = wasSigned >= 0 ? 1 : -1;
            float nowSigned, nowDist;
            bool  nowInterior;
            closestOn(p, a1, b1, c1, n1, nowSigned, nowDist, nowInterior);
            float target = float(direction) * gap;
            if(direction > 0 ? nowSigned >= target : nowSigned <= target)
              continue;
            if(nowDist > m.depth)
              continue;
            // past the edge, not under the face
            if(abs(nowSigned) < 0.6 * nowDist)
              continue;
            float  push  = min(abs(target - nowSigned), m.cap);
            float  along = target - nowSigned > 0 ? 1.0 : -1.0;
            p            = p + n1 * (push * along);
            // friction sheds the slide of this substep; the contact absorbs the motion along the normal
            float3 d  = p - m.prev[i];
            float  dn = dot(d, n1);
            float  fr = clamp((tri.friction + fri) * 0.5, 0.0, 0.95);
            p         = p - (d - n1 * dn) * fr;
            m.prev[i] = p - (d - n1 * dn) * (1 - fr);
            m.x[i]    = p;
          }
          break;
        }
      }
  m.cprev[i] = m.x[i];
}

// Velocities from the substep's motion.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void velocityMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.numFree)
    return;
  int i    = m.freeVerts[tid.x];
  m.vel[i] = (m.x[i] - m.prev[i]) / pc.dt;
}
