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

// The dForce cloth solver on the GPU: one substep is integrate, one dispatch per colour of edges
// (then the stretch colours again, a few times), the tethers of partial strength, a collision pass
// on some substeps, then velocities. The host (src/fv_dforce_gpu.cpp) records the substeps of a
// frame and reads the positions back. The physics follows the CPU solver
// (figure/dforce/solver.cpp): XPBD with small steps, kinematic vertices swept through the frame,
// the side of a face read from the vertex's last pass, contacts absorbing normal motion.
// Differences: every free vertex looks for faces on every pass, a cloth's own faces are looked for
// from where they were at the frame's start (the CPU's grid of them is made every fourth pass), and
// the kinematic vertices move along straight lines within a frame (the host sweeps them along their
// bones between frames). As on the CPU, a vertex lost behind the nearest face of
// a lower layer is put back at the frame's start (recoverMain), and one resting on a face is held
// outside it every substep (holdMain).

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

// As the host's (figure/physics/collide/sweep_grid.hpp, sweepCellHash): the cell's coordinates mixed,
// then finished as MurmurHash3's fmix32 does, so neighbouring cells land apart in the table and a
// probe stays short in a fuller one
uint cellHash(int ix, int iy, int iz)
{
  uint h = (uint(ix) * 73856093u) ^ (uint(iy) * 19349663u) ^ (uint(iz) * 83492791u);
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
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
  if(k < m.numSoft)
    m.lamAtt[k] = 0;
  if(k < m.numClusterEdges)
    m.clusterLam[k] = 0;
}

// A vertex of partial strength is tethered to where the pose would put it: a distance-0 XPBD
// constraint as compliant as its strength makes it. Each writes its own vertex only.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void attachMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.numSoft)
    return;
  int    i      = m.softVerts[tid.x];
  float3 anchor = kinematicAt(m, i, pc.t);
  float3 d      = m.x[i] - anchor;
  float  C      = length(d);
  if(C < 1e-8)
    return;
  float w  = m.inv[i];
  float at = m.attAlpha[i] / (pc.dt * pc.dt);
  float dl = (-C - at * m.lamAtt[tid.x]) / (w + at);
  m.lamAtt[tid.x] += dl;
  m.x[i] = m.x[i] + d * (w * dl / C);
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
  // a reach link only ever pulls: what it has given back it can't give again
  if(ed.kind == DFORCE_EDGE_REACH)
    dl = min(m.lam[e] + dl, 0.0) - m.lam[e];
  m.lam[e] += dl;
  if(wa > 0)
    m.x[ed.a] = m.x[ed.a] - n * (wa * dl);
  if(wb > 0)
    m.x[ed.b] = m.x[ed.b] + n * (wb * dl);
}

// A cluster's vertices for a sweep: xyz and the inverse mass.
groupshared float4 clusterX[DFORCE_CLUSTER_LOCAL];

// One XPBD distance row over the cluster's shared vertices, as edgesMain projects one over the model's.
// Only the cluster's own vertices (local index under `owned`) move: a halo vertex, another cluster's
// end of a crossing row, counts with its inverse mass and stays where it was read.
void clusterRow(DforceModel m, uint r, float atScale, uint owned)
{
  DforceEdge ed = m.clusterEdges[r];
  float4     pa = clusterX[ed.a], pb = clusterX[ed.b];
  float      w  = pa.w + pb.w;
  if(w <= 0)
    return;
  float3 d   = pb.xyz - pa.xyz;
  float  len = length(d);
  if(len < 1e-8)
    return;
  float alpha = ed.alpha;
  if(ed.kind == DFORCE_EDGE_BEND && len < ed.rest * ed.ratio)
    alpha = ed.buckle > 0 ? alpha / ed.buckle : 1e6;
  float  C  = len - ed.rest;
  float3 n  = d / len;
  float  at = alpha * atScale;
  float  lam = m.clusterLam[r];
  float  dl  = (-C - at * lam) / (w + at);
  if(ed.kind == DFORCE_EDGE_REACH)
    dl = min(lam + dl, 0.0) - lam;
  m.clusterLam[r] = lam + dl;
  if(pa.w > 0 && uint(ed.a) < owned)
    clusterX[ed.a].xyz = pa.xyz - n * (pa.w * dl);
  if(pb.w > 0 && uint(ed.b) < owned)
    clusterX[ed.b].xyz = pb.xyz + n * (pb.w * dl);
}

// One sweep of a cluster, a workgroup a cluster (physics-plan step 13a): its vertices into shared
// memory, its colours one at a time between group barriers (the stretch rows', then the reach links'
// for a sweep that holds length or the rest's for a full one), its tethers, and its own vertices back.
// The rows that cross clusters run after it, a colour a dispatch, as edgesMain.
[shader("compute")]
[numthreads(DFORCE_CLUSTER_THREADS, 1, 1)]
void clusterMain(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID)
{
  DforceModel m      = *pc.model;
  uint        c      = group.x;
  int         first  = m.clusterVertStart[c];
  int         count  = m.clusterVertStart[c + 1] - first;
  int         owned  = m.clusterOwned[c];
  uint        t      = local.x;
  for(int l = int(t); l < count; l += DFORCE_CLUSTER_THREADS)
  {
    int v       = m.clusterVerts[first + l];
    // free vertices where the last sweep left them; pinned ones where the substep put them
    float w     = m.inv[v];
    clusterX[l] = float4(w > 0 ? pc.clusterSrc[v] : m.x[v], w);
  }
  GroupMemoryBarrierWithGroupSync();
  float atScale = 1.0 / (pc.dt * pc.dt);
  uint  colours = m.clusterColours;
  uint  base    = c * colours;
  uint  rest0   = m.clusterStretch + m.clusterReach;
  for(uint k = 0; k < colours; k++)
  {
    // a sweep that holds length skips the rest's colours, a full one the reach links'
    if(k >= m.clusterStretch && (pc.lengthOnly != 0 ? k >= rest0 : k < rest0))
      continue;
    int r0 = m.clusterColourStart[base + k], r1 = m.clusterColourStart[base + k + 1];
    for(int r = r0 + int(t); r < r1; r += DFORCE_CLUSTER_THREADS)
      clusterRow(m, uint(r), atScale, uint(owned));
    GroupMemoryBarrierWithGroupSync();
  }
  // the tethers of partial strength: each writes its own vertex only
  for(int s = m.clusterSoftStart[c] + int(t); s < m.clusterSoftStart[c + 1]; s += DFORCE_CLUSTER_THREADS)
  {
    uint2  ls     = m.clusterSoft[s];
    int    v      = m.clusterVerts[first + int(ls.x)];
    float4 p      = clusterX[ls.x];
    float3 d      = p.xyz - kinematicAt(m, v, pc.t);
    float  C      = length(d);
    if(C < 1e-8)
      continue;
    float at = m.attAlpha[v] * atScale;
    float dl = (-C - at * m.lamAtt[ls.y]) / (p.w + at);
    m.lamAtt[ls.y] += dl;
    clusterX[ls.x].xyz = p.xyz + d * (p.w * dl / C);
  }
  GroupMemoryBarrierWithGroupSync();
  for(int l = int(t); l < owned; l += DFORCE_CLUSTER_THREADS)
    pc.clusterDst[m.clusterVerts[first + l]] = clusterX[l].xyz;
}

// The free vertices back from x's twin, after an odd number of cluster sweeps.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void commitMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.numFree)
    return;
  int i  = m.freeVerts[tid.x];
  m.x[i] = pc.clusterSrc[i];
}

// The nearest face of a lower layer within `reach` of q, with the faces where the pose has them at
// fraction t, whose boxes boxesMain made for it: its index (-1 for none), the signed distance along
// its normal, and that normal.
int nearestFace(DforceModel m, float3 q, float reach, int layer, float t, out float signedDist, out float3 normal)
{
  int    best     = -1;
  float  bestDist = reach;
  signedDist      = 0;
  normal          = float3(0);
  // how much further than the nearest so far a face's box may lie and still have its distance made: 0.1 mm
  // and more far out, against closestOn's rounding of some 1e-5 cm at a figure's coordinates
  float  slack    = 0.01 + (abs(q.x) + abs(q.y) + abs(q.z)) * 1e-6;
  float3 lo = q - reach, hi = q + reach;
  for(int ix = cellOf(lo.x, m.cellSize); ix <= cellOf(hi.x, m.cellSize); ix++)
    for(int iy = cellOf(lo.y, m.cellSize); iy <= cellOf(hi.y, m.cellSize); iy++)
      for(int iz = cellOf(lo.z, m.cellSize); iz <= cellOf(hi.z, m.cellSize); iz++)
      {
        uint s = cellHash(ix, iy, iz) & m.cellMask;
        for(uint probe = 0; probe <= m.cellMask; probe++)
        {
          DforceCell cell = m.cells[(s + probe) & m.cellMask];
          if(cell.start < 0)
            break;
          if(cell.x != ix || cell.y != iy || cell.z != iz)
            continue;
          for(int k = 0; k < cell.count; k++)
          {
            int    f    = m.items[cell.start + k];
            float4 box0 = m.faceBox[2 * f], box1 = m.faceBox[2 * f + 1];
            if(asint(box1.w) >= layer)
              continue;
            // a face whose box lies further off than the nearest so far can't be nearer: skipped before its
            // corners are read. The slack is far over closestOn's rounding, so what is skipped would have
            // lost the comparison below anyway, and the nearest is the same face
            float3 out  = max(max(box0.xyz - q, q - box1.xyz), float3(0));
            float  room = bestDist + slack;
            if(dot(out, out) > room * room)
              continue;
            DforceTri tri = m.tris[f];
            float3    a, b, c, n;
            faceAt(m, tri, t, a, b, c);
            if(!faceNormal(a, b, c, n))
              continue;
            float sd, dist;
            bool  interior;
            closestOn(q, a, b, c, n, sd, dist, interior);
            if(dist < bestDist)
            {
              best       = f;
              bestDist   = dist;
              signedDist = sd;
              normal     = n;
            }
          }
          break;
        }
      }
  return best;
}

// Before a collision pass and the frame's recovery: each face's box where the pose has it at pc.t, with its
// offset and its layer, for the vertices to skip a face they are nowhere near before reading its corners.
// A little larger than the face's (a thousandth of a cm, and more far out), so the corners made again
// where a box passes, rounded as they may be, never lie outside it: what misses this box misses the
// face's own, and every pass and recovery comes out as it would without it, to the bit.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void boxesMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.numTris)
    return;
  DforceTri tri = m.tris[tid.x];
  float3    a, b, c;
  faceAt(m, tri, pc.t, a, b, c);
  float3 lo    = min(a, min(b, c)), hi = max(a, max(b, c));
  float3 slack = 0.001 + max(abs(lo), abs(hi)) * 1e-6;
  m.faceBox[2 * tid.x]     = float4(lo - slack, tri.offset);
  m.faceBox[2 * tid.x + 1] = float4(hi + slack, asfloat(tri.layer));
}

// At a frame's start: a free vertex touching nothing that lies behind the nearest face of a lower
// layer within recoverDepth is put back outside it, and rests on that face from now on.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void recoverMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.numFree)
    return;
  int i = m.freeVerts[tid.x];
  if(m.collides[i] == 0 || m.contact[i] >= 0)
    return;
  float  sd;
  float3 n;
  int    t = nearestFace(m, m.x[i], m.recoverDepth, m.layer[i], pc.t, sd, n);
  if(t < 0 || sd >= 0)
    return;
  DforceTri tri = m.tris[t];
  float     gap = max(tri.offset, m.offset[i]);
  m.x[i]       = m.x[i] + n * (gap - sd);
  m.cprev[i]   = m.x[i];
  m.contact[i] = t;
}

// The collision pass and the recovery a wave a vertex (collideWaveMain, recoverWaveMain), where the device's
// waves are 32 lanes (DforceGpu turns them on): a vertex met a cell's faces one after another on a thread
// of its own, and with a few thousand vertices most of the GPU stood idle while each walked its hundreds
// of faces (a light copy's pass cost as much as a dress five times its size). Its lanes now take 32 faces
// at a time, in the order the thread met them, and what they decide is what it decided, to the bit.

// nearestFace, the lanes of a wave sharing its faces. Each lane weighs a face of the next 32; the nearest of
// them, the first of equally near ones, becomes the nearest so far if it is nearer, as one after another
// they would. A lane skips a face its box puts further off than the nearest as the 32 began: that is no
// nearer than the nearest when the thread came to it, so what it skips the thread skipped or passed over.
int nearestFaceWave(DforceModel m, float3 q, float reach, int layer, float t, out float signedDist, out float3 normal)
{
  const uint lane     = WaveGetLaneIndex();
  const int  lanes    = int(WaveGetLaneCount());
  int        best     = -1;
  float      bestDist = reach;
  signedDist          = 0;
  normal              = float3(0);
  float  slack        = 0.01 + (abs(q.x) + abs(q.y) + abs(q.z)) * 1e-6;
  float3 lo = q - reach, hi = q + reach;
  for(int ix = cellOf(lo.x, m.cellSize); ix <= cellOf(hi.x, m.cellSize); ix++)
    for(int iy = cellOf(lo.y, m.cellSize); iy <= cellOf(hi.y, m.cellSize); iy++)
      for(int iz = cellOf(lo.z, m.cellSize); iz <= cellOf(hi.z, m.cellSize); iz++)
      {
        uint s = cellHash(ix, iy, iz) & m.cellMask;
        for(uint probe = 0; probe <= m.cellMask; probe++)
        {
          DforceCell cell = m.cells[(s + probe) & m.cellMask];
          if(cell.start < 0)
            break;
          if(cell.x != ix || cell.y != iy || cell.z != iz)
            continue;
          for(int first = 0; first < cell.count; first += lanes)
          {
            const int k    = first + int(lane);
            int       f    = -1;
            bool      near = false;
            float     sd = 0, dist = 0;
            float3    n = float3(0);
            if(k < cell.count)
            {
              f           = m.items[cell.start + k];
              float4 box0 = m.faceBox[2 * f], box1 = m.faceBox[2 * f + 1];
              float3 out  = max(max(box0.xyz - q, q - box1.xyz), float3(0));
              float  room = bestDist + slack;
              if(asint(box1.w) < layer && dot(out, out) <= room * room)
              {
                DforceTri tri = m.tris[f];
                float3    a, b, c;
                faceAt(m, tri, t, a, b, c);
                if(faceNormal(a, b, c, n))
                {
                  bool interior;
                  closestOn(q, a, b, c, n, sd, dist, interior);
                  near = dist < bestDist;
                }
              }
            }
            const float least   = WaveActiveMin(near ? dist : 3.0e38);
            const uint  winners = WaveActiveBallot(near && dist == least).x;
            if(winners != 0)
            {
              const uint j = firstbitlow(winners);
              best         = WaveReadLaneAt(f, j);
              bestDist     = least;
              signedDist   = WaveReadLaneAt(sd, j);
              normal       = WaveReadLaneAt(n, j);
            }
          }
          break;
        }
      }
  return best;
}

// recoverMain, a wave a vertex.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
[require(spvGroupNonUniformBallot, spvGroupNonUniformShuffle)]
void recoverWaveMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  const uint  v = tid.x / WaveGetLaneCount();
  if(v >= m.numFree)
    return;
  int i = m.freeVerts[v];
  if(m.collides[i] == 0 || m.contact[i] >= 0)
    return;
  float  sd;
  float3 n;
  int    t = nearestFaceWave(m, m.x[i], m.recoverDepth, m.layer[i], pc.t, sd, n);
  if(t < 0 || sd >= 0 || !WaveIsFirstLane())
    return;
  DforceTri tri = m.tris[t];
  float     gap = max(tri.offset, m.offset[i]);
  m.x[i]       = m.x[i] + n * (gap - sd);
  m.cprev[i]   = m.x[i];
  m.contact[i] = t;
}

// Every substep: a vertex resting on a face is held outside it where the pose has the face now,
// while it stays over the face, so its edges cannot pull it into the limb between two passes.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void holdMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.numFree)
    return;
  int i = m.freeVerts[tid.x];
  int t = m.contact[i];
  if(t < 0)
    return;
  DforceTri tri = m.tris[t];
  float3    a, b, c, n;
  faceAt(m, tri, pc.t, a, b, c);
  if(!faceNormal(a, b, c, n))
    return;
  float sd, dist;
  bool  interior;
  closestOn(m.x[i], a, b, c, n, sd, dist, interior);
  if(!interior)
    return;
  int layer = m.layer[i];
  int direction;
  if(layer != tri.layer)
    direction = layer > tri.layer ? 1 : -1;
  else
    direction = dot(m.cprev[i] - a, n) >= 0 ? 1 : -1;
  float gap    = max(tri.offset, m.offset[i]);
  float target = float(direction) * gap;
  if(direction > 0 ? sd >= target : sd <= target)
    return;
  float  push = min(abs(target - sd), m.cap) * (target - sd > 0 ? 1.0 : -1.0);
  float3 p    = m.x[i] + n * push;
  // Coulomb, as the CPU solver: this substep's slide over the face is taken back as far as the push's
  // normal force holds it. Not in the hold after a pass
  if(pc.grip == 0)
  {
    m.x[i] = p;
    return;
  }
  float3 slide = (p - m.prev[i]) - ((m.x[tri.a] - m.prev[tri.a]) + (m.x[tri.b] - m.prev[tri.b]) + (m.x[tri.c] - m.prev[tri.c])) / 3.0;
  slide        = slide - n * dot(slide, n);
  float along  = length(slide);
  if(along > 1e-9)
    p = p - slide * min(1.0, (tri.friction + m.friction[i]) * 0.5 * abs(push) / along);
  m.x[i] = p;
}

// Pushes a free vertex out of the faces it was outside of at its last pass, and notes the face it
// rests on, if any.
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
  // the cells between where the vertex was and where it is: a vertex that moved further since its
  // last pass than the depth it can be pushed back from is looked at where it is alone, or the
  // cells to walk grow with the cube of the move (a frame of them took two seconds and lost the
  // device); such a vertex is past what a pass recovers anyway, and recoverMain puts it back
  if(moved > m.depth)
  {
    was   = x0;
    moved = 0;
  }
  float3 lo    = min(x0, was) - reach;
  float3 hi    = max(x0, was) + reach;
  int    layer = m.layer[i];
  float  off   = m.offset[i];
  float  fri   = m.friction[i];
  int    resting = -1;
  // Another garment's faces (a garment's own with Self Collide, when the host gives them): filed where they were at the frame's
  // start, so they're looked for between where the vertex was then and where it is (the two layers of
  // a skirt move together); a vertex that has gone further than a pass recovers from, where it was then
  if(m.numClothTris > 0)
  {
    float3 v0  = m.frameStart[i];
    float3 clo = length(x0 - v0) > m.depth ? v0 - reach : min(x0, v0) - reach;
    float3 chi = length(x0 - v0) > m.depth ? v0 + reach : max(x0, v0) + reach;
    int    own = m.meshOf[i];
    for(int ix = cellOf(clo.x, m.cellSize); ix <= cellOf(chi.x, m.cellSize); ix++)
      for(int iy = cellOf(clo.y, m.cellSize); iy <= cellOf(chi.y, m.cellSize); iy++)
        for(int iz = cellOf(clo.z, m.cellSize); iz <= cellOf(chi.z, m.cellSize); iz++)
        {
          uint s = cellHash(ix, iy, iz) & m.clothCellMask;
          for(uint probe = 0; probe <= m.clothCellMask; probe++)
          {
            DforceCell cell = m.clothCells[(s + probe) & m.clothCellMask];
            if(cell.start < 0)
              break;
            if(cell.x != ix || cell.y != iy || cell.z != iz)
              continue;
            for(int k = 0; k < cell.count; k++)
            {
              int       f   = m.clothItems[cell.start + k];
              DforceTri tri = m.clothTris[f];
              if(tri.a == i || tri.b == i || tri.c == i)
                continue;
              if(tri.mesh == own && (tri.self == 0 || nearInCloth(m, i, tri.a) || nearInCloth(m, i, tri.b) || nearInCloth(m, i, tri.c)))
                continue;
              // a face filed in two cells the box meets is tested twice, as the body's are: the first
              // push leaves the vertex at the gap, so the second finds nothing to do
              float3 a1 = clothCornerNow(m, tri.a), b1 = clothCornerNow(m, tri.b), c1 = clothCornerNow(m, tri.c);
              float3 n1;
              if(!faceNormal(a1, b1, c1, n1))
                continue;
              float3 p    = m.x[i];
              float  gap  = max(tri.offset, off);
              float  near = min(m.depth, gap + moved);
              float3 blo  = min(a1, min(b1, c1)) - near;
              float3 bhi  = max(a1, max(b1, c1)) + near;
              if(any(p < blo) || any(p > bhi))
                continue;
              float3 a0 = clothCornerWas(m, tri.a), b0 = clothCornerWas(m, tri.b), c0 = clothCornerWas(m, tri.c);
              float3 n0;
              if(!faceNormal(a0, b0, c0, n0))
                continue;
              float wasSigned, wasDist;
              bool  wasInterior;
              closestOn(was, a0, b0, c0, n0, wasSigned, wasDist, wasInterior);
              // a side only where the vertex lay over or under the face at its last pass: one beside it,
              // past an edge, has none, and was pushed through (a walk, two layers sliding by each other)
              if(abs(wasSigned) < 0.6 * wasDist)
                continue;
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
              if(abs(nowSigned) < 0.6 * nowDist)
                continue;
              float  push  = min(abs(target - nowSigned), m.cap);
              float  along = target - nowSigned > 0 ? 1.0 : -1.0;
              p            = p + n1 * (push * along);
              float3 d     = p - m.prev[i];
              float  dn    = dot(d, n1);
              float  fr    = clamp((tri.friction + fri) * 0.5, 0.0, 0.95);
              p            = p - (d - n1 * dn) * fr;
              m.prev[i]    = p - (d - n1 * dn) * (1 - fr);
              m.x[i]       = p;
            }
            break;
          }
        }
  }
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
            int    f    = m.items[cell.start + k];
            float4 box0 = m.faceBox[2 * f], box1 = m.faceBox[2 * f + 1];
            // the vertex where it is now: an earlier face may have moved it. A face it is nowhere near is
            // left on its box from boxesMain, a little larger than the face's own, before its corners are read
            // (most of a cell's faces); what passes is tried on the face's own box as ever
            float3 p    = m.x[i];
            float  gap  = max(box0.w, off);
            float  near = min(m.depth, gap + moved);
            if(any(p < box0.xyz - near) || any(p > box1.xyz + near))
              continue;
            DforceTri tri = m.tris[f];
            float3    a0, b0, c0, a1, b1, c1;
            faceAt(m, tri, pc.t, a1, b1, c1);
            float3 blo  = min(a1, min(b1, c1)) - near;
            float3 bhi  = max(a1, max(b1, c1)) + near;
            if(any(p < blo) || any(p > bhi))
              continue;
            faceAt(m, tri, pc.tFaces, a0, b0, c0);
            float3 n0, n1;
            if(!faceNormal(a0, b0, c0, n0) || !faceNormal(a1, b1, c1, n1))
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
            if(nowInterior && nowDist <= gap + near)
              resting = f;
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
  m.cprev[i]   = m.x[i];
  m.contact[i] = resting;
}

// One of the pose's faces against a free vertex at p, as collideMain meets each in turn (it is that loop's
// body): p pushed out to the face's side and its slide shed, prev moved with it, and the face it rests on
// noted. True when it moved p.
bool meetFace(DforceModel m, int f, float3 was, float moved, int layer, float off, float fri, inout float3 p, inout float3 prev, inout int resting)
{
  DforceTri tri  = m.tris[f];
  float     gap  = max(tri.offset, off);
  float     near = min(m.depth, gap + moved);
  float3    a0, b0, c0, a1, b1, c1;
  faceAt(m, tri, pc.t, a1, b1, c1);
  float3 blo = min(a1, min(b1, c1)) - near;
  float3 bhi = max(a1, max(b1, c1)) + near;
  if(any(p < blo) || any(p > bhi))
    return false;
  faceAt(m, tri, pc.tFaces, a0, b0, c0);
  float3 n0, n1;
  if(!faceNormal(a0, b0, c0, n0) || !faceNormal(a1, b1, c1, n1))
    return false;
  float wasSigned, wasDist;
  bool  wasInterior;
  closestOn(was, a0, b0, c0, n0, wasSigned, wasDist, wasInterior);
  int direction;
  if(layer > tri.layer)
  {
    if(wasSigned < -0.05)
      return false;
    direction = 1;
  }
  else if(layer < tri.layer)
  {
    if(wasSigned > 0.05)
      return false;
    direction = -1;
  }
  else
    direction = wasSigned >= 0 ? 1 : -1;
  float nowSigned, nowDist;
  bool  nowInterior;
  closestOn(p, a1, b1, c1, n1, nowSigned, nowDist, nowInterior);
  if(nowInterior && nowDist <= gap + near)
    resting = f;
  float target = float(direction) * gap;
  if(direction > 0 ? nowSigned >= target : nowSigned <= target)
    return false;
  if(nowDist > m.depth)
    return false;
  // past the edge, not under the face
  if(abs(nowSigned) < 0.6 * nowDist)
    return false;
  float  push  = min(abs(target - nowSigned), m.cap);
  float  along = target - nowSigned > 0 ? 1.0 : -1.0;
  p            = p + n1 * (push * along);
  // friction sheds the slide of this substep; the contact absorbs the motion along the normal
  float3 d  = p - prev;
  float  dn = dot(d, n1);
  float  fr = clamp((tri.friction + fri) * 0.5, 0.0, 0.95);
  p         = p - (d - n1 * dn) * fr;
  prev      = p - (d - n1 * dn) * (1 - fr);
  return true;
}

// Whether p may meet face f, by its box from boxesMain: what misses it, meetFace leaves as it is.
bool mayMeet(DforceModel m, int f, float3 p, float off, float moved)
{
  float4 box0 = m.faceBox[2 * f], box1 = m.faceBox[2 * f + 1];
  float  near = min(m.depth, max(box0.w, off) + moved);
  return !(any(p < box0.xyz - near) || any(p > box1.xyz + near));
}

// collideMain over the pose's faces, a wave a vertex: its lanes look at the next 32 faces at once, each on
// its box, and those it may meet are met one after another in their order, on every lane alike, as the
// thread met them. A push moves the vertex: the faces after it among the 32 are looked at again from there.
// A garment meeting another's faces (numClothTris) runs collideMain.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
[require(spvGroupNonUniformBallot, spvGroupNonUniformShuffle)]
void collideWaveMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m     = *pc.model;
  const uint  lane  = WaveGetLaneIndex();
  const int   lanes = int(WaveGetLaneCount());
  const uint  v     = tid.x / uint(lanes);
  if(v >= m.numFree)
    return;
  int i = m.freeVerts[v];
  if(m.collides[i] == 0)
    return;
  float3 x0    = m.x[i];
  float3 was   = m.cprev[i];
  float  moved = length(x0 - was);
  float  reach = m.searchReach;
  if(moved > m.depth)
  {
    was   = x0;
    moved = 0;
  }
  float3 lo      = min(x0, was) - reach;
  float3 hi      = max(x0, was) + reach;
  int    layer   = m.layer[i];
  float  off     = m.offset[i];
  float  fri     = m.friction[i];
  int    resting = -1;
  float3 p       = x0;
  float3 prev    = m.prev[i];
  for(int ix = cellOf(lo.x, m.cellSize); ix <= cellOf(hi.x, m.cellSize); ix++)
    for(int iy = cellOf(lo.y, m.cellSize); iy <= cellOf(hi.y, m.cellSize); iy++)
      for(int iz = cellOf(lo.z, m.cellSize); iz <= cellOf(hi.z, m.cellSize); iz++)
      {
        uint s = cellHash(ix, iy, iz) & m.cellMask;
        for(uint probe = 0; probe <= m.cellMask; probe++)
        {
          DforceCell cell = m.cells[(s + probe) & m.cellMask];
          if(cell.start < 0)
            break;
          if(cell.x != ix || cell.y != iy || cell.z != iz)
            continue;
          for(int first = 0; first < cell.count; first += lanes)
          {
            const int k    = first + int(lane);
            const int f    = k < cell.count ? m.items[cell.start + k] : -1;
            uint      them = WaveActiveBallot(f >= 0 && mayMeet(m, f, p, off, moved)).x;
            while(them != 0)
            {
              const uint j = firstbitlow(them);
              // the faces up to j are done
              them &= ~((2u << j) - 1u);
              if(meetFace(m, WaveReadLaneAt(f, j), was, moved, layer, off, fri, p, prev, resting))
                them = WaveActiveBallot(f >= 0 && lane > j && mayMeet(m, f, p, off, moved)).x;
            }
          }
          break;
        }
      }
  if(!WaveIsFirstLane())
    return;
  m.x[i]       = p;
  m.prev[i]    = prev;
  m.cprev[i]   = p;
  m.contact[i] = resting;
}

// As a collision pass begins: every position, and every free vertex's at the last pass, as they are now.
[shader("compute")]
[numthreads(DFORCE_WORKGROUP_SIZE, 1, 1)]
void snapMain(uint3 tid: SV_DispatchThreadID)
{
  DforceModel m = *pc.model;
  if(tid.x >= m.nv)
    return;
  m.snap[tid.x]      = m.x[tid.x];
  m.cprevSnap[tid.x] = m.cprev[tid.x];
}

// Whether v is among i's stretch neighbours or theirs (sorted): a cloth doesn't collide with the faces
// around a vertex, whose planes it is in or next to.
bool nearInCloth(DforceModel m, int i, int v)
{
  int lo = m.ring2Start[i], hi = m.ring2Start[i + 1] - 1;
  while(lo <= hi)
  {
    int mid = (lo + hi) >> 1;
    int at  = m.ring2[mid];
    if(at == v)
      return true;
    if(at < v)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return false;
}

// A cloth face's corner as the pass began, and where it was at the last pass: a free corner's own, a
// pinned one's where the pose had it then.
float3 clothCornerNow(DforceModel m, int v)
{
  return m.snap[v];
}
float3 clothCornerWas(DforceModel m, int v)
{
  return m.inv[v] > 0 ? m.cprevSnap[v] : kinematicAt(m, v, pc.tFaces);
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
