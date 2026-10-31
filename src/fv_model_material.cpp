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

#include "fv_model_material.hpp"

#include <figure/materials/material_gltf.hpp>

namespace fv::model {

fv::mat::Material editedMaterial(const fv::mat::Material& original, const fv::mat::Edits& e)
{
  fv::mat::Material m = original;
  if(e.color)
    for(size_t i = 0; i < 3; i++)
      m.baseColor[i] = (*e.color)[i];
  if(e.opacity)
  {
    m.baseColor[3] = *e.opacity;
    // or the slider would do nothing; an edit of the alpha mode, below, has the last word
    if(*e.opacity < 1 && m.alphaMode == fv::mat::AlphaMode::eOpaque)
      m.alphaMode = fv::mat::AlphaMode::eBlend;
  }
  if(e.roughness)
    m.roughness = *e.roughness;
  if(e.metalness)
    m.metallic = *e.metalness;
  if(e.alphaMode)
    m.alphaMode = *e.alphaMode == "opaque" ? fv::mat::AlphaMode::eOpaque : *e.alphaMode == "cutout" ? fv::mat::AlphaMode::eMask : fv::mat::AlphaMode::eBlend;
  if(e.doubleSided)
    m.doubleSided = *e.doubleSided;
  if(e.emissive)
    m.emissive = *e.emissive;
  if(e.emissiveIntensity)
    m.emissiveStrength = *e.emissiveIntensity;
  if(e.specular)
  {
    if(!m.specular)
      m.specular.emplace();
    m.specular->factor = *e.specular;
  }
  if(e.clearcoat || e.clearcoatRoughness)
  {
    if(!m.clearcoat)
      m.clearcoat.emplace();
    if(e.clearcoat)
      m.clearcoat->factor = *e.clearcoat;
    if(e.clearcoatRoughness)
      m.clearcoat->roughness = *e.clearcoatRoughness;
  }
  if(e.normalStrength && m.normalMap.has())
    m.normalMap.scale = *e.normalStrength;
  // sheen over the model's own: black is none, and a roughness alone changes only a sheen it has
  if(e.sheenColor)
  {
    if(!m.sheen)
      m.sheen.emplace();
    m.sheen->color = *e.sheenColor;
  }
  if(e.sheenRoughness && m.sheen)
    m.sheen->roughness = *e.sheenRoughness;
  return m;
}

tinygltf::Material editedMaterial(const tinygltf::Material& original, const fv::mat::Edits& e)
{
  return e.empty() ? original : fv::mat::toGltf(editedMaterial(fv::mat::fromGltf(original), e));
}

}  // namespace fv::model
