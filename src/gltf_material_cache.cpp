/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Packs a material (fv::mat::Material; a glTF one converts first) into GPU-friendly
// GltfShadeMaterial structures for use in shaders. Caches the converted
// material array so it can be rebuilt only when materials change.
//

#include "gltf_material_cache.hpp"

#include <algorithm>
#include <cfloat>
#include <cstddef>

#include <figure/materials/material_gltf.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "nvutils/logger.hpp"
#include "tinygltf_utils.hpp"

// Layout guardrails for the GPU material struct. These must hold regardless of which
// MAT_EXT_* flags are enabled so the material buffer array stride stays consistent and
// base (always-on) field offsets do not drift.
//
// The buffer is bound via Vulkan buffer_reference with scalar block layout
// (-force-glsl-scalar-layout in compile_slang). A trailing uint64_t _pad forces the
// struct's max member alignment to 8 so every materials[i] is 8-aligned on the GPU;
// the driver can then emit LDG.64 for adjacent 4-byte field pairs (roughness/metallic,
// alphaMode/alphaCutoff, ...) across every element of the array, not just element 0.
static_assert(alignof(shaderio::GltfShadeMaterial) >= 8,
              "GltfShadeMaterial must be at least 8-byte aligned (trailing uint64_t _pad enforces this)");
static_assert(sizeof(shaderio::GltfShadeMaterial) % 8 == 0,
              "GltfShadeMaterial size must be a multiple of 8 for LDG.64-friendly BDA array stride");
static_assert(offsetof(shaderio::GltfShadeMaterial, pbrBaseColorFactor) == 0, "pbrBaseColorFactor must stay at offset 0");
static_assert(offsetof(shaderio::GltfShadeMaterial, pbrRoughnessFactor) == 32,
              "pbrRoughnessFactor must stay at offset 32 (anchors base block)");
static_assert(offsetof(shaderio::GltfShadeMaterial, alphaMode) == 40, "alphaMode must stay at offset 40 (anchors base block)");
static_assert(offsetof(shaderio::GltfShadeMaterial, occlusionStrength) == 48,
              "occlusionStrength must stay at offset 48 (promoted to base block)");
static_assert(offsetof(shaderio::GltfShadeMaterial, doubleSided) == 52, "doubleSided must stay at offset 52 (promoted to base block)");

// GltfTextureInfo is indexed per texture reference, so its stride matters. texCoord (int8_t) and
// flags (uint8_t) are byte-sized deliberately: they fit in the padding after samplerIndex, keeping
// the struct at 8 bytes (32 with KHR_texture_transform) instead of growing the array. Widening
// either field silently costs 4 bytes per texture info on both host and device, so pin the size here.
static_assert(sizeof(shaderio::GltfTextureInfo) == (MAT_EXT_TEXTURE_TRANSFORM ? 32 : 8),
              "GltfTextureInfo must stay 8 bytes (32 with MAT_EXT_TEXTURE_TRANSFORM = 1); keep the "
              "sub-4-byte fields packed into the tail after samplerIndex");

namespace nvvkgltf {

namespace {

// textureSlots supplies the per-texture data resolved by SceneVk (sampler slot, source-format
// flags), indexed by the slot's index.
shaderio::GltfTextureInfo getTextureInfoImpl(const fv::mat::TextureSlot& slot, const TextureSlotTable& textureSlots)
{
  shaderio::GltfTextureInfo ti{};
  ti.index        = textureSlots.gpuTexture(slot.index);
  ti.samplerIndex = static_cast<int16_t>(textureSlots.samplerSlot(slot.index));
  ti.flags        = textureSlots.isTwoChannelSource(slot.index) ?
                        static_cast<uint8_t>(shaderio::TextureInfoFlagBits::eTexInfoTwoChannelSource) :
                        uint8_t{0};
  // Per reference, not per image: the same image can be sampled flipped in one slot and not another.
  if(slot.flipGreen)
    ti.flags |= static_cast<uint8_t>(shaderio::TextureInfoFlagBits::eTexInfoFlipGreen);

  // The renderer only uploads TEXCOORD_0 and TEXCOORD_1 (see VertexBuffers::texCoord0/texCoord1
  // in shaders/gltf_scene_io.h.slang). glTF allows TEXCOORD_N for higher N but we clamp
  // to 1 and warn once per occurrence so the user notices missing UV sets instead of
  // silently rendering with the wrong coordinates.
  if(slot.texCoord > 1)
  {
    LOGW(
        "GltfTextureInfo: texture references TEXCOORD_%d but the renderer only supports "
        "TEXCOORD_0 and TEXCOORD_1; clamping to TEXCOORD_1.\n",
        slot.texCoord);
  }
  ti.texCoord = static_cast<int8_t>(std::clamp(slot.texCoord, 0, 1));

#if MAT_EXT_TEXTURE_TRANSFORM
  KHR_texture_transform transform;  // the identity unless the slot has one
  if(slot.transform)
  {
    transform.offset   = {slot.transform->offset[0], slot.transform->offset[1]};
    transform.scale    = {slot.transform->scale[0], slot.transform->scale[1]};
    transform.rotation = slot.transform->rotation != 0 ? slot.transform->rotation : 0.0f;  // not -0
    transform.updateTransform();
  }
  ti.uvTransform = shaderio::float3x2(transform.uvTransform[0][0], transform.uvTransform[1][0], transform.uvTransform[0][1],
                                      transform.uvTransform[1][1], transform.uvTransform[0][2], transform.uvTransform[1][2]);
#endif
  return ti;
}

template <typename TextureHandler>
void populateShaderMaterial(shaderio::GltfShadeMaterial& dstMat, const fv::mat::Material& srcMat, TextureHandler handleTexture)
{
  using namespace fv::mat;
  const auto rgb = [](const Rgb& c) { return glm::vec3(c[0], c[1], c[2]); };

  dstMat.alphaMode          = srcMat.alphaMode == AlphaMode::eOpaque ? 0 : (srcMat.alphaMode == AlphaMode::eMask ? 1 : 2);
  dstMat.alphaCutoff        = srcMat.alphaCutoff;
  dstMat.doubleSided        = srcMat.doubleSided ? 1 : 0;
  dstMat.pbrBaseColorFactor = glm::vec4(srcMat.baseColor[0], srcMat.baseColor[1], srcMat.baseColor[2], srcMat.baseColor[3]);
  dstMat.pbrMetallicFactor  = srcMat.metallic;
  dstMat.pbrRoughnessFactor = srcMat.roughness;
  dstMat.normalTextureScale = srcMat.normalMap.scale.value_or(1.0f);
  dstMat.occlusionStrength  = srcMat.occlusionMap.scale.value_or(1.0f);
  dstMat.emissiveFactor     = rgb(srcMat.emissive);

  handleTexture(dstMat.emissiveTexture, srcMat.emissiveMap);
  handleTexture(dstMat.normalTexture, srcMat.normalMap);
  handleTexture(dstMat.pbrBaseColorTexture, srcMat.baseColorMap);
  handleTexture(dstMat.pbrMetallicRoughnessTexture, srcMat.metallicRoughnessMap);
  handleTexture(dstMat.occlusionTexture, srcMat.occlusionMap);

  // a lobe left out packs as one at its defaults
#if MAT_EXT_TRANSMISSION
  const Transmission transmission = srcMat.transmission.value_or(Transmission{});
  dstMat.transmissionFactor       = transmission.factor;
  handleTexture(dstMat.transmissionTexture, transmission.map);
#endif

#if MAT_EXT_IOR
  dstMat.ior = srcMat.ior.value_or(1.5f);
#endif

#if MAT_EXT_VOLUME
  const Volume volume     = srcMat.volume.value_or(Volume{});
  dstMat.attenuationColor = rgb(volume.attenuationColor);
  dstMat.thicknessFactor  = volume.thickness;
  // Spec range is (0, +inf); clamp out-of-range values to FLT_MAX (= no attenuation) on the CPU
  // so shaders can skip the branch and rely on division by FLT_MAX giving ~0 extinction.
  dstMat.attenuationDistance = (volume.attenuationDistance > 0.0f) ? volume.attenuationDistance : FLT_MAX;
  handleTexture(dstMat.thicknessTexture, volume.thicknessMap);
#endif

#if MAT_EXT_CLEARCOAT
  const Clearcoat clearcoat          = srcMat.clearcoat.value_or(Clearcoat{});
  dstMat.clearcoatFactor             = clearcoat.factor;
  dstMat.clearcoatRoughness          = clearcoat.roughness;
  dstMat.clearcoatNormalTextureScale = clearcoat.normalMap.scale.value_or(1.0f);
  handleTexture(dstMat.clearcoatRoughnessTexture, clearcoat.roughnessMap);
  handleTexture(dstMat.clearcoatTexture, clearcoat.map);
  handleTexture(dstMat.clearcoatNormalTexture, clearcoat.normalMap);
  dstMat.clearcoatColor = rgb(clearcoat.color);
  handleTexture(dstMat.clearcoatColorTexture, clearcoat.colorMap);
  const Specular2 specular2 = srcMat.specular2.value_or(Specular2{});
  dstMat.specular2Factor    = specular2.factor;
  dstMat.specular2Roughness = specular2.roughness;
  handleTexture(dstMat.specular2Texture, specular2.map);
  handleTexture(dstMat.specular2RoughnessTexture, specular2.roughnessMap);
#endif

#if MAT_EXT_SPECULAR
  const Specular specular    = srcMat.specular.value_or(Specular{});
  dstMat.specularFactor      = specular.factor;
  dstMat.specularColorFactor = rgb(specular.color);
  handleTexture(dstMat.specularTexture, specular.map);
  handleTexture(dstMat.specularColorTexture, specular.colorMap);
#endif

  // The strength is always applied: it only scales the emissive factor, no field of its own.
  dstMat.emissiveFactor *= srcMat.emissiveStrength.value_or(1.0f);

#if MAT_EXT_UNLIT
  dstMat.unlit = srcMat.unlit ? 1 : 0;
#endif

#if MAT_EXT_IRIDESCENCE
  const Iridescence iridescence      = srcMat.iridescence.value_or(Iridescence{});
  dstMat.iridescenceFactor           = iridescence.factor;
  dstMat.iridescenceIor              = iridescence.ior;
  dstMat.iridescenceThicknessMinimum = iridescence.thicknessMinimum;
  dstMat.iridescenceThicknessMaximum = iridescence.thicknessMaximum;
  handleTexture(dstMat.iridescenceTexture, iridescence.map);
  handleTexture(dstMat.iridescenceThicknessTexture, iridescence.thicknessMap);
#endif

#if MAT_EXT_ANISOTROPY
  const Anisotropy anisotropy = srcMat.anisotropy.value_or(Anisotropy{});
  dstMat.anisotropyRotation   = glm::vec2(glm::sin(anisotropy.rotation), glm::cos(anisotropy.rotation));
  dstMat.anisotropyStrength   = anisotropy.strength;
  handleTexture(dstMat.anisotropyTexture, anisotropy.map);
#endif

#if MAT_EXT_SHEEN
  const Sheen sheen           = srcMat.sheen.value_or(Sheen{});
  dstMat.sheenColorFactor     = rgb(sheen.color);
  dstMat.sheenRoughnessFactor = sheen.roughness;
  handleTexture(dstMat.sheenColorTexture, sheen.colorMap);
  handleTexture(dstMat.sheenRoughnessTexture, sheen.roughnessMap);
#endif

#if MAT_EXT_DISPERSION
  dstMat.dispersion = srcMat.dispersion.value_or(0.0f);
#endif

#if MAT_EXT_SPECULAR_GLOSSINESS
  // Default pbrModel is ePbrModelMetallicRoughness (struct member default); switch to
  // ePbrModelSpecularGlossiness only when the material actually has it.
  const SpecularGlossiness pbr = srcMat.specularGlossiness.value_or(SpecularGlossiness{});
  if(srcMat.specularGlossiness)
  {
    dstMat.pbrModel            = shaderio::PbrModel::ePbrModelSpecularGlossiness;
    dstMat.pbrDiffuseFactor    = glm::vec4(pbr.diffuse[0], pbr.diffuse[1], pbr.diffuse[2], pbr.diffuse[3]);
    dstMat.pbrSpecularFactor   = rgb(pbr.specular);
    dstMat.pbrGlossinessFactor = pbr.glossiness;
  }
  handleTexture(dstMat.pbrDiffuseTexture, pbr.diffuseMap);
  handleTexture(dstMat.pbrSpecularGlossinessTexture, pbr.specularGlossinessMap);
#endif

#if MAT_EXT_DIFFUSE_TRANSMISSION
  const DiffuseTransmission diffuseTransmission = srcMat.diffuseTransmission.value_or(DiffuseTransmission{});
  dstMat.diffuseTransmissionFactor              = diffuseTransmission.factor;
  dstMat.diffuseTransmissionColorFactor         = rgb(diffuseTransmission.color);
  handleTexture(dstMat.diffuseTransmissionTexture, diffuseTransmission.map);
  handleTexture(dstMat.diffuseTransmissionColorTexture, diffuseTransmission.colorMap);
#endif

#if MAT_EXT_RETROREFLECTION
  const Retroreflection retro  = srcMat.retroreflection.value_or(Retroreflection{});
  dstMat.retroreflectionFactor = retro.factor;
  handleTexture(dstMat.retroreflectionTexture, retro.map);
#endif

#if MAT_EXT_SCATTER
  const Scatter scatter          = srcMat.scatter.value_or(Scatter{});
  dstMat.scatterStrengthFactor   = std::clamp(scatter.strength, 0.0f, 1.0f);
  dstMat.multiscatterColorFactor = rgb(scatter.multiscatterColor);
  dstMat.scatterAnisotropy       = std::clamp(scatter.anisotropy, -0.999f, 0.999f);
  dstMat.scatterDistance         = glm::max(rgb(srcMat.scatterDistance.value_or(Rgb{0, 0, 0})), glm::vec3(0.0f));
  handleTexture(dstMat.scatterStrengthTexture, scatter.strengthMap);
  handleTexture(dstMat.multiscatterColorTexture, scatter.multiscatterColorMap);
#endif

#if MAT_EXT_DETAIL
  const Detail detail          = srcMat.detail.value_or(Detail{});
  dstMat.detailNormalScale     = detail.normalMap.scale.value_or(1.0f);
  dstMat.detailWeightFactor    = detail.weight;
  dstMat.detailRoughnessFactor = detail.roughness;
  handleTexture(dstMat.detailNormalTexture, detail.normalMap);
  handleTexture(dstMat.detailWeightTexture, detail.weightMap);
  handleTexture(dstMat.detailRoughnessTexture, detail.roughnessMap);
#endif

#if MAT_EXT_HAIR
  if(srcMat.hair)
  {
    const Hair& hair        = *srcMat.hair;
    dstMat.hairFactor       = 1.0f;
    dstMat.hairLongitudinal = hair.longitudinal;
    dstMat.hairAzimuthal    = hair.azimuthal;
    dstMat.hairRootFade     = hair.rootFade;
    dstMat.hairRootV        = hair.rootV;
    dstMat.hairSpecular     = hair.specular;
    dstMat.hairRootSigma    = rgb(hair.rootAbsorption);
    dstMat.hairTipSigma     = rgb(hair.tipAbsorption);
    dstMat.hairFromColor    = hair.absorptionFromColor ? 1.0f : 0.0f;
    if(hair.dye)
    {
      dstMat.hairDyed         = 1.0f;
      dstMat.hairRootColor    = rgb(hair.dye->rootColor);
      dstMat.hairTipColor     = rgb(hair.dye->tipColor);
      dstMat.hairMelaninSigma = rgb(hair.dye->melaninAbsorption);
      dstMat.hairMelaninMix   = hair.dye->melaninMix;
      dstMat.hairBlendBias    = hair.dye->bias;
      dstMat.hairBlendGain    = hair.dye->gain;
    }
    if(hair.strandRotation)
    {
      dstMat.hairStrandSin = glm::sin(*hair.strandRotation);
      dstMat.hairStrandCos = glm::cos(*hair.strandRotation);
    }
  }
#endif

#if MAT_EXT_DLSS_NR
  const Rgba nr = srcMat.dlssNrMask.value_or(Rgba{1, 1, 1, 1});
  dstMat.nrMask = glm::vec4(nr[0], nr[1], nr[2], nr[3]);
#endif
}

}  // namespace

void MaterialCache::buildFromMaterials(const std::vector<tinygltf::Material>& materials, const TextureSlotTable& textureSlots)
{
  m_textureInfos.clear();
  m_textureInfos.push_back({});
  m_shadeMaterials.clear();
  m_shadeMaterials.reserve(materials.size());
  for(const auto& srcMat : materials)
    append(fv::mat::fromGltf(srcMat), textureSlots);
}

void MaterialCache::append(const fv::mat::Material& srcMat, const TextureSlotTable& textureSlots)
{
  if(m_textureInfos.empty())
    m_textureInfos.push_back({});  // entry 0: no texture
  m_shadeMaterials.emplace_back(convertOne(srcMat, textureSlots, 0, m_textureInfos));
}

shaderio::GltfShadeMaterial MaterialCache::convertOne(const fv::mat::Material&                srcMat,
                                                      const TextureSlotTable&                 textureSlots,
                                                      uint32_t                                base,
                                                      std::vector<shaderio::GltfTextureInfo>& infos)
{
  shaderio::GltfShadeMaterial dstMat = {};
  populateShaderMaterial(dstMat, srcMat, [&](uint16_t& texIndex, const fv::mat::TextureSlot& slot) {
    const shaderio::GltfTextureInfo ti = getTextureInfoImpl(slot, textureSlots);
    texIndex                           = 0;
    if(ti.index != -1)
    {
      texIndex = static_cast<uint16_t>(base + infos.size());
      infos.push_back(ti);
    }
  });
  dstMat.emissiveLightScale = textureSlots.emissiveLightScale(srcMat.emissiveMap.index);
  return dstMat;
}

shaderio::GltfShadeMaterial MaterialCache::convertOne(const tinygltf::Material&               srcMat,
                                                      const TextureSlotTable&                 textureSlots,
                                                      uint32_t                                base,
                                                      std::vector<shaderio::GltfTextureInfo>& infos)
{
  return convertOne(fv::mat::fromGltf(srcMat), textureSlots, base, infos);
}

MaterialUpdateResult MaterialCache::updateMaterial(int index, const tinygltf::Material& srcGltf, const TextureSlotTable& textureSlots)
{
  MaterialUpdateResult result{};
  if(index < 0 || index >= static_cast<int>(m_shadeMaterials.size()))
    return result;

  const fv::mat::Material      srcMat = fv::mat::fromGltf(srcGltf);
  shaderio::GltfShadeMaterial& dstMat = m_shadeMaterials[index];
  shaderio::GltfTextureInfo*   infos  = m_textureInfos.data();

  populateShaderMaterial(dstMat, srcMat, [&](uint16_t& texIndex, const fv::mat::TextureSlot& slot) {
    const bool hasSrcTexture = slot.index != -1;

    if(texIndex == 0)
    {
      if(hasSrcTexture)
        result.topologyChanged = true;
      return;
    }
    if(!hasSrcTexture)
    {
      result.topologyChanged = true;
      return;
    }

    infos[texIndex]    = getTextureInfoImpl(slot, textureSlots);
    result.span.minIdx = std::min(result.span.minIdx, texIndex);
    result.span.maxIdx = std::max(result.span.maxIdx, texIndex);
    result.span.count++;
  });
  dstMat.emissiveLightScale = textureSlots.emissiveLightScale(srcMat.emissiveMap.index);
  return result;
}

void MaterialCache::clear()
{
  m_shadeMaterials.clear();
  m_textureInfos.clear();
}

}  // namespace nvvkgltf
