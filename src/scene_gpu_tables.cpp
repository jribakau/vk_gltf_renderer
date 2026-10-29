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

#include "scene_gpu_tables.hpp"

#include <algorithm>
#include <limits>

#include <cmath>

namespace nvvkgltf {

shaderio::GltfLight packLight(const tinygltf::Light* light, const glm::mat4& world)
{
  shaderio::GltfLight info{};
  info.position  = world[3];
  info.direction = glm::normalize(glm::vec3(-world[2]));
  if(light)
  {
    // KHR_lights_punctual light (possibly also carrying EXT_lights_ies)
    info.innerAngle = static_cast<float>(light->spot.innerConeAngle);
    info.outerAngle = static_cast<float>(light->spot.outerConeAngle);
    info.color = light->color.size() == 3 ? glm::vec3(light->color[0], light->color[1], light->color[2]) : glm::vec3(1.0f);
    info.intensity = static_cast<float>(light->intensity);
    info.type      = light->type == "point" ? shaderio::eLightTypePoint :
                     light->type == "spot"  ? shaderio::eLightTypeSpot :
                                              shaderio::eLightTypeDirectional;
    info.radius    = light->extras.Has("radius") ? float(light->extras.Get("radius").GetNumberAsDouble()) : 0.0f;
    if(info.type == shaderio::eLightTypeDirectional)
    {
      const double sun_distance     = 149597870.0;  // km
      double       angular_size_rad = 2.0 * std::atan(info.radius / sun_distance);
      info.angularSizeOrInvRange    = static_cast<float>(angular_size_rad);
    }
    else
    {
      info.angularSizeOrInvRange = (light->range > 0.0) ? 1.0f / static_cast<float>(light->range) : 0.0f;
    }
  }
  else
  {
    // Pure EXT_lights_ies (no KHR_lights_punctual): IES profile governs angular distribution;
    // base intensity and color come from the node-level multiplier/color.
    info.color                 = glm::vec3(1.0f);
    info.intensity             = 1.0f;
    info.type                  = shaderio::eLightTypePoint;
    info.innerAngle            = 0.0f;
    info.outerAngle            = glm::radians(90.0f);
    info.angularSizeOrInvRange = 0.0f;  // infinite range
    info.radius                = 0.0f;
  }
  return info;
}

shaderio::GltfLight packIesLight(const glm::mat4& world, int profile, float multiplier, const glm::vec3& color)
{
  shaderio::GltfLight info = packLight(nullptr, world);
  info.iesProfile          = profile;
  info.color *= color;
  info.intensity *= multiplier;
  return info;
}

namespace {

// A Vose alias table over weights[first, first + count) of `tris`, its indices absolute
void buildAliasTable(std::vector<shaderio::EmissiveTriangle>& tris, const std::vector<double>& weights, size_t first, size_t count)
{
  double total = 0.0;
  for(size_t i = first; i < first + count; ++i)
    total += weights[i];
  if(count == 0 || !(total > 0.0))
    return;
  std::vector<float>    scaled(count);  // p_i * n; a bin with scaled < 1 borrows from a "large" bin
  std::vector<uint32_t> smallBins, largeBins;
  smallBins.reserve(count);
  largeBins.reserve(count);
  for(size_t i = 0; i < count; ++i)
  {
    scaled[i] = static_cast<float>(weights[first + i] / total * static_cast<double>(count));
    (scaled[i] < 1.0f ? smallBins : largeBins).push_back(static_cast<uint32_t>(i));
  }
  while(!smallBins.empty() && !largeBins.empty())
  {
    const uint32_t l = smallBins.back();
    smallBins.pop_back();
    const uint32_t g = largeBins.back();
    largeBins.pop_back();
    tris[first + l].aliasProb  = scaled[l];
    tris[first + l].aliasIndex = static_cast<uint32_t>(first + g);
    scaled[g]                  = (scaled[g] + scaled[l]) - 1.0f;  // g keeps the leftover probability
    (scaled[g] < 1.0f ? smallBins : largeBins).push_back(g);
  }
  for(uint32_t g : largeBins)  // numerical remainder: these bins are always taken as themselves
  {
    tris[first + g].aliasProb  = 1.0f;
    tris[first + g].aliasIndex = static_cast<uint32_t>(first + g);
  }
  for(uint32_t l : smallBins)
  {
    tris[first + l].aliasProb  = 1.0f;
    tris[first + l].aliasIndex = static_cast<uint32_t>(first + l);
  }
}

// An emitter, a leaf of the light tree: its node as the shader reads it, and its world bounds
struct TreeLeaf
{
  shaderio::EmissiveGroup node;
  glm::vec3               lo, hi;
};

// A node of the tree as it is built: its bounds and weight, its children (-1, -1 for a leaf) or its
// leaf
struct BuildNode
{
  glm::vec3 lo, hi;
  double    weight = 0.0;
  int       left = -1, right = -1;
  int       leaf = -1;
};

// The subtree over leaves[order[begin, end)): split at the median of their centres along the longest
// side of the centres' bounds. Returns its root.
int buildTree(std::vector<BuildNode>& nodes, std::span<const TreeLeaf> leaves, std::vector<int>& order, size_t begin, size_t end)
{
  const int index = int(nodes.size());
  nodes.push_back({});
  if(end - begin == 1)
  {
    const TreeLeaf& leaf = leaves[order[begin]];
    nodes[index]         = {.lo = leaf.lo, .hi = leaf.hi, .weight = leaf.node.weight, .leaf = order[begin]};
    return index;
  }
  glm::vec3 lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
  for(size_t i = begin; i < end; i++)
  {
    lo = glm::min(lo, leaves[order[i]].node.center);
    hi = glm::max(hi, leaves[order[i]].node.center);
  }
  const glm::vec3 extent = hi - lo;
  const int       axis   = extent.x >= extent.y && extent.x >= extent.z ? 0 : extent.y >= extent.z ? 1 : 2;
  const size_t    mid    = begin + (end - begin) / 2;
  std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                   [&](int a, int b) { return leaves[a].node.center[axis] < leaves[b].node.center[axis]; });
  const int left  = buildTree(nodes, leaves, order, begin, mid);
  const int right = buildTree(nodes, leaves, order, mid, end);
  nodes[index]    = {.lo     = glm::min(nodes[left].lo, nodes[right].lo),
                     .hi     = glm::max(nodes[left].hi, nodes[right].hi),
                     .weight = nodes[left].weight + nodes[right].weight,
                     .left   = left,
                     .right  = right};
  return index;
}

// The light tree over `leaves` as the shader reads it (light_tree.h.slang): its cut, at most `maxCut`
// nodes, first, then their subtrees, breadth first. The cut is the root split, the widest node first,
// until it holds `maxCut` nodes or leaves alone; with no more leaves than that, the leaves in order.
std::vector<shaderio::EmissiveGroup> buildLightTree(std::span<const TreeLeaf> leaves, size_t maxCut, uint32_t& cutCount)
{
  std::vector<shaderio::EmissiveGroup> out;
  if(leaves.size() <= maxCut)
  {
    for(const TreeLeaf& leaf : leaves)
      out.push_back(leaf.node);
    cutCount = uint32_t(out.size());
    return out;
  }
  std::vector<BuildNode> nodes;
  std::vector<int>       order(leaves.size());
  for(size_t i = 0; i < order.size(); i++)
    order[i] = int(i);
  const int root = buildTree(nodes, leaves, order, 0, leaves.size());

  std::vector<int> queue = {root};
  while(queue.size() < maxCut)
  {
    int   widest = -1;
    float size   = -1.f;
    for(size_t i = 0; i < queue.size(); i++)
      if(nodes[queue[i]].leaf < 0 && glm::length(nodes[queue[i]].hi - nodes[queue[i]].lo) > size)
      {
        widest = int(i);
        size   = glm::length(nodes[queue[i]].hi - nodes[queue[i]].lo);
      }
    if(widest < 0)
      break;
    const BuildNode& split = nodes[queue[widest]];
    queue[widest]          = split.left;
    queue.push_back(split.right);
  }
  cutCount = uint32_t(queue.size());

  // the cut, then breadth first below it: `queue` grows as it is read
  std::vector<int> outIndex(nodes.size(), -1);
  for(size_t i = 0; i < queue.size(); i++)
    outIndex[queue[i]] = int(i);
  for(size_t i = 0; i < queue.size(); i++)
    for(const int child : {nodes[queue[i]].left, nodes[queue[i]].right})
      if(child >= 0)
      {
        outIndex[child] = int(queue.size());
        queue.push_back(child);
      }
  out.resize(queue.size());
  for(size_t i = 0; i < queue.size(); i++)
  {
    const BuildNode& node = nodes[queue[i]];
    shaderio::EmissiveGroup& g = out[i];
    if(node.leaf >= 0)
      g = leaves[node.leaf].node;
    else
    {
      g        = {};
      g.center = 0.5f * (node.lo + node.hi);
      g.radius = 0.5f * glm::length(node.hi - node.lo);
      g.weight = float(node.weight);
      g.left   = outIndex[node.left];
      g.right  = outIndex[node.right];
    }
    g.parent = -1;
  }
  for(size_t i = 0; i < out.size(); i++)
    if(out[i].left >= 0)
    {
      out[out[i].left].parent  = int(i);
      out[out[i].right].parent = int(i);
    }
  return out;
}

}  // namespace

EmissiveTable buildEmissiveTable(std::span<const Emitter> emitters, int maxCut)
{
  // First pass: gather emissive triangles + per-triangle world area, and accumulate the totals used
  // to derive the area-weighted mean luminance for defensive sampling.
  EmissiveTable                            out;
  std::vector<shaderio::EmissiveTriangle>& tris = out.triangles;
  std::vector<float>                       triArea;  // parallel to tris (used by the alias pass below)
  double                                   sumArea = 0.0, sumAreaLum = 0.0;
  // each emitter's triangles, as a group, and their world bounds
  struct Span
  {
    size_t    first = 0;
    uint32_t  renderNodeID = 0;
    glm::vec3 lo{std::numeric_limits<float>::max()}, hi{-std::numeric_limits<float>::max()};
  };
  std::vector<Span> spans;
  for(const Emitter& e : emitters)
  {
    const float lum = emissiveLuminance(e.radiance);
    if(lum <= 0.0f || e.positions.empty() || e.indices.size() < 3)
      continue;
    spans.push_back({.first = tris.size(), .renderNodeID = e.renderNodeID});
    const uint32_t numTri = static_cast<uint32_t>(e.indices.size() / 3);
    for(uint32_t t = 0; t < numTri; ++t)
    {
      const uint32_t i0 = e.indices[3 * t + 0], i1 = e.indices[3 * t + 1], i2 = e.indices[3 * t + 2];
      if(i0 >= e.positions.size() || i1 >= e.positions.size() || i2 >= e.positions.size())
        continue;
      const glm::vec3 w0   = glm::vec3(e.world * glm::vec4(e.positions[i0], 1.0f));
      const glm::vec3 w1   = glm::vec3(e.world * glm::vec4(e.positions[i1], 1.0f));
      const glm::vec3 w2   = glm::vec3(e.world * glm::vec4(e.positions[i2], 1.0f));
      const float     area = 0.5f * glm::length(glm::cross(w1 - w0, w2 - w0));
      if(area <= 0.0f)
        continue;  // skip degenerate (zero-area) triangles
      sumArea += area;
      sumAreaLum += static_cast<double>(area) * lum;
      for(const glm::vec3& w : {w0, w1, w2})
      {
        spans.back().lo = glm::min(spans.back().lo, w);
        spans.back().hi = glm::max(spans.back().hi, w);
      }

      shaderio::EmissiveTriangle et{};
      et.radiance     = e.radiance;
      et.renderNodeID = e.renderNodeID;
      et.renderPrimID = e.renderPrimID;
      et.triangleID   = t;
      et.doubleSided  = e.doubleSided ? 1u : 0u;
      // aliasProb / aliasIndex are filled in the alias-table pass below.
      tris.push_back(et);
      triArea.push_back(area);
    }
  }

  // Second pass: defensive selection. Pull each emitter's luminance toward the area-weighted mean so
  // a dim emitter is never starved (lum' = (1-f)*lum + f*meanLum); the weight is area * lum'. The area
  // still cancels in the solid-angle pdf, so the shader recovers it from radiance + meanLum + totals.
  const float         meanLum = (sumArea > 0.0) ? static_cast<float>(sumAreaLum / sumArea) : 0.0f;
  const float         f       = shaderio::EMISSIVE_DEFENSIVE_FRACTION;
  std::vector<double> triWeight(tris.size());
  double              totalWeight = 0.0;
  for(size_t i = 0; i < tris.size(); ++i)
  {
    const float lum    = emissiveLuminance(tris[i].radiance);
    const float lumDef = (1.0f - f) * lum + f * meanLum;
    triWeight[i]       = static_cast<double>(triArea[i]) * lumDef;
    totalWeight += triWeight[i];
  }

  // Third pass: build a Vose alias table so the shader selects a triangle in O(1) (no CDF binary
  // search). It samples the exact same distribution P(i) = weight_i / totalWeight, so the pdf is
  // unchanged. See https://en.wikipedia.org/wiki/Alias_method. With the light tree, one per leaf: P(i)
  // = weight_i / the leaf's weight, the leaf picked by the shader.
  // (an emitter whose triangles were all degenerate leaves an empty span: no leaf)
  if(spans.empty() || maxCut <= 0)
    buildAliasTable(tris, triWeight, 0, tris.size());
  else
  {
    std::vector<TreeLeaf> leaves;
    for(size_t g = 0; g < spans.size(); ++g)
    {
      const size_t first = spans[g].first, end = g + 1 < spans.size() ? spans[g + 1].first : tris.size();
      if(end == first)
        continue;
      buildAliasTable(tris, triWeight, first, end - first);
      double weight = 0.0;
      for(size_t i = first; i < end; ++i)
        weight += triWeight[i];
      shaderio::EmissiveGroup leaf{.center       = 0.5f * (spans[g].lo + spans[g].hi),
                                   .radius       = 0.5f * glm::length(spans[g].hi - spans[g].lo),
                                   .weight       = static_cast<float>(weight),
                                   .first        = static_cast<uint32_t>(first),
                                   .count        = static_cast<uint32_t>(end - first),
                                   .renderNodeID = spans[g].renderNodeID,
                                   .left         = -1,
                                   .right        = -1,
                                   .parent       = -1};
      leaves.push_back({leaf, spans[g].lo, spans[g].hi});
    }
    out.groups = buildLightTree(leaves, size_t(maxCut), out.cutCount);
    for(size_t i = 0; i < out.groups.size(); i++)
      if(out.groups[i].left < 0)
      {
        const uint32_t node = out.groups[i].renderNodeID;
        if(node >= out.leafOfNode.size())
          out.leafOfNode.resize(size_t(node) + 1, -1);
        out.leafOfNode[node] = int32_t(i);
      }
  }
  out.totalWeight = static_cast<float>(totalWeight);
  out.meanLum     = meanLum;
  return out;
}

}  // namespace nvvkgltf
