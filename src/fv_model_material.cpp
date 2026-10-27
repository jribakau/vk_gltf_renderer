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

namespace fv::model {

namespace {

void setExtValue(tinygltf::ExtensionMap& ext, const char* name, const char* key, tinygltf::Value value)
{
  tinygltf::Value::Object obj;
  if(auto it = ext.find(name); it != ext.end() && it->second.IsObject())
    obj = it->second.Get<tinygltf::Value::Object>();
  obj[key]  = std::move(value);
  ext[name] = tinygltf::Value(obj);
}

void setExtNumber(tinygltf::ExtensionMap& ext, const char* name, const char* key, double value)
{
  setExtValue(ext, name, key, tinygltf::Value(value));
}

}  // namespace

tinygltf::Material editedMaterial(const tinygltf::Material& original, const fv::mat::Edits& e)
{
  tinygltf::Material m = original;
  auto&              f = m.pbrMetallicRoughness.baseColorFactor;
  f.resize(4, 1.0);
  if(e.color)
    for(int i = 0; i < 3; i++)
      f[size_t(i)] = double((*e.color)[size_t(i)]);
  if(e.opacity)
    f[3] = double(*e.opacity);
  if(e.roughness)
    m.pbrMetallicRoughness.roughnessFactor = double(*e.roughness);
  if(e.metalness)
    m.pbrMetallicRoughness.metallicFactor = double(*e.metalness);
  if(e.alphaMode)
    m.alphaMode = *e.alphaMode == "opaque" ? "OPAQUE" : *e.alphaMode == "cutout" ? "MASK" : "BLEND";
  if(e.doubleSided)
    m.doubleSided = *e.doubleSided;
  if(e.emissive)
    m.emissiveFactor = {double((*e.emissive)[0]), double((*e.emissive)[1]), double((*e.emissive)[2])};
  if(e.emissiveIntensity)
    setExtNumber(m.extensions, "KHR_materials_emissive_strength", "emissiveStrength", double(*e.emissiveIntensity));
  if(e.specular)
    setExtNumber(m.extensions, "KHR_materials_specular", "specularFactor", double(*e.specular));
  if(e.clearcoat)
    setExtNumber(m.extensions, "KHR_materials_clearcoat", "clearcoatFactor", double(*e.clearcoat));
  if(e.clearcoatRoughness)
    setExtNumber(m.extensions, "KHR_materials_clearcoat", "clearcoatRoughnessFactor", double(*e.clearcoatRoughness));
  if(e.normalStrength && m.normalTexture.index >= 0)
    m.normalTexture.scale = double(*e.normalStrength);
  // sheen over the model's own: black is none, and a roughness alone changes only a sheen it has
  if(e.sheenColor)
    setExtValue(m.extensions, "KHR_materials_sheen", "sheenColorFactor",
                tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(double((*e.sheenColor)[0])), tinygltf::Value(double((*e.sheenColor)[1])),
                                                       tinygltf::Value(double((*e.sheenColor)[2]))}));
  if(e.sheenRoughness && m.extensions.count("KHR_materials_sheen"))
    setExtNumber(m.extensions, "KHR_materials_sheen", "sheenRoughnessFactor", double(*e.sheenRoughness));
  return m;
}

}  // namespace fv::model
