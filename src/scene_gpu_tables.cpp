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

EmissiveTable buildEmissiveTable(std::span<const Emitter> emitters)
{
  // First pass: gather emissive triangles + per-triangle world area, and accumulate the totals used
  // to derive the area-weighted mean luminance for defensive sampling.
  EmissiveTable                            out;
  std::vector<shaderio::EmissiveTriangle>& tris = out.triangles;
  std::vector<float>                       triArea;  // parallel to tris (used by the alias pass below)
  double                                   sumArea = 0.0, sumAreaLum = 0.0;
  for(const Emitter& e : emitters)
  {
    const float lum = emissiveLuminance(e.radiance);
    if(lum <= 0.0f || e.positions.empty() || e.indices.size() < 3)
      continue;
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
  // unchanged. See https://en.wikipedia.org/wiki/Alias_method.
  if(!tris.empty() && totalWeight > 0.0)
  {
    const size_t          n = tris.size();
    std::vector<float>    scaled(n);  // p_i * n; a bin with scaled < 1 borrows from a "large" bin
    std::vector<uint32_t> smallBins, largeBins;
    smallBins.reserve(n);
    largeBins.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
      scaled[i] = static_cast<float>(triWeight[i] / totalWeight * static_cast<double>(n));
      (scaled[i] < 1.0f ? smallBins : largeBins).push_back(static_cast<uint32_t>(i));
    }
    while(!smallBins.empty() && !largeBins.empty())
    {
      const uint32_t l = smallBins.back();
      smallBins.pop_back();
      const uint32_t g = largeBins.back();
      largeBins.pop_back();
      tris[l].aliasProb  = scaled[l];
      tris[l].aliasIndex = g;
      scaled[g]          = (scaled[g] + scaled[l]) - 1.0f;  // g keeps the leftover probability
      (scaled[g] < 1.0f ? smallBins : largeBins).push_back(g);
    }
    for(uint32_t g : largeBins)  // numerical remainder: these bins are always taken as themselves
    {
      tris[g].aliasProb  = 1.0f;
      tris[g].aliasIndex = g;
    }
    for(uint32_t l : smallBins)
    {
      tris[l].aliasProb  = 1.0f;
      tris[l].aliasIndex = l;
    }
  }
  out.totalWeight = static_cast<float>(totalWeight);
  out.meanLum     = meanLum;
  return out;
}

}  // namespace nvvkgltf
