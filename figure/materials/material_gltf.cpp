#include <figure/materials/material_gltf.hpp>

#include <algorithm>

namespace fv::mat {

using Json = nlohmann::json;

namespace {

constexpr const char* kTransform    = "KHR_texture_transform";
constexpr const char* kFlipGreen    = "FV_texture_flip_green";
constexpr const char* kScatter      = "KHR_materials_scatter";
constexpr const char* kScatterDraft = "KHR_materials_volume_scatter";  // an earlier draft's name: read, never written
constexpr const char* kSubsurface   = "FV_materials_subsurface";
constexpr const char* kSpecular2    = "FV_materials_specular2";

// Reading a value, each leaving `out` alone where the file has none: glTF's defaults are the structs'.
void read(const tinygltf::Value& v, const char* key, float& out)
{
  if(v.IsObject() && v.Has(key) && v.Get(key).IsNumber())
    out = float(v.Get(key).GetNumberAsDouble());
}

template <size_t N>
void read(const tinygltf::Value& v, const char* key, std::array<float, N>& out)
{
  if(!v.IsObject() || !v.Has(key) || !v.Get(key).IsArray())
    return;
  const tinygltf::Value& a = v.Get(key);
  for(size_t i = 0; i < std::min(N, a.ArrayLen()); i++)
    if(a.Get(int(i)).IsNumber())
      out[i] = float(a.Get(int(i)).GetNumberAsDouble());
}

void read(const tinygltf::Value& v, const char* key, bool& out)
{
  if(v.IsObject() && v.Has(key) && v.Get(key).IsBool())
    out = v.Get(key).Get<bool>();
}

// A texture info's extensions: what the renderer reads of them, and the rest to write back
void readInfoExtensions(const tinygltf::ExtensionMap& extensions, TextureSlot& out)
{
  for(const auto& [name, value] : extensions)
  {
    if(name == kFlipGreen)
      out.flipGreen = true;
    else if(name == kTransform)
    {
      UvTransform t;
      read(value, "offset", t.offset);
      read(value, "scale", t.scale);
      read(value, "rotation", t.rotation);
      out.transform = t;
      if(value.IsObject() && value.Has("texCoord") && value.Get("texCoord").IsNumber())
        out.transformTexCoord = value.Get("texCoord").GetNumberAsInt();
    }
    else
      out.foreignExtensions[name] = value;
  }
}

template <class Info>
TextureSlot slotOf(const Info& info)
{
  TextureSlot out;
  out.index    = info.index;
  out.texCoord = info.texCoord;
  out.extras   = info.extras;
  readInfoExtensions(info.extensions, out);
  return out;
}

// A texture info inside an extension: an object of its own
void read(const tinygltf::Value& v, const char* key, TextureSlot& out)
{
  if(!v.IsObject() || !v.Has(key) || !v.Get(key).IsObject())
    return;
  const tinygltf::Value& t = v.Get(key);
  if(t.Has("index") && t.Get("index").IsNumber())
    out.index = t.Get("index").GetNumberAsInt();
  if(t.Has("texCoord") && t.Get("texCoord").IsNumber())
    out.texCoord = t.Get("texCoord").GetNumberAsInt();
  if(t.Has("scale") && t.Get("scale").IsNumber())
    out.scale = float(t.Get("scale").GetNumberAsDouble());
  if(t.Has("extras"))
    out.extras = t.Get("extras");
  if(t.Has("extensions") && t.Get("extensions").IsObject())
    readInfoExtensions(t.Get("extensions").Get<tinygltf::Value::Object>(), out);
}

Json rgbJson(const std::array<float, 3>& c)
{
  return Json::array({c[0], c[1], c[2]});
}

Json rgbaJson(const Rgba& c)
{
  return Json::array({c[0], c[1], c[2], c[3]});
}

// A slot as a texture info; `scaleKey`: where its scale goes, if it has one
Json infoJson(const TextureSlot& s, const char* textureKey, const char* scaleKey = "scale")
{
  Json out{{textureKey, s.index}, {"texCoord", s.texCoord}};
  if(s.scale)
    out[scaleKey] = *s.scale;
  Json extensions = Json::object();
  for(const auto& [name, value] : s.foreignExtensions)
    extensions[name] = fromValue(value);
  if(s.flipGreen)
    extensions[kFlipGreen] = Json::object();
  if(s.transform)
  {
    Json tt = {{"scale", Json::array({s.transform->scale[0], s.transform->scale[1]})},
               {"offset", Json::array({s.transform->offset[0], s.transform->offset[1]})}};
    if(s.transform->rotation != 0)
      tt["rotation"] = s.transform->rotation;
    if(s.transformTexCoord)
      tt["texCoord"] = *s.transformTexCoord;
    extensions[kTransform] = tt;
  }
  if(!extensions.empty())
    out["extensions"] = extensions;
  if(s.extras.Type() != tinygltf::NULL_TYPE)
    out["extras"] = fromValue(s.extras);
  return out;
}

void useExtension(std::vector<std::string>& used, const std::string& name)
{
  if(std::find(used.begin(), used.end(), name) == used.end())
    used.push_back(name);
}

// Every extension a texture info inside `v` names
void collectInfoExtensions(const Json& v, std::vector<std::string>& used)
{
  if(v.is_array())
    for(const Json& x : v)
      collectInfoExtensions(x, used);
  if(!v.is_object())
    return;
  if(v.contains("index") && v.contains("extensions") && v["extensions"].is_object())
    for(auto it = v["extensions"].begin(); it != v["extensions"].end(); ++it)
      useExtension(used, it.key());
  for(auto it = v.begin(); it != v.end(); ++it)
    collectInfoExtensions(it.value(), used);
}

void textureInfo(const Json& j, tinygltf::TextureInfo& out)
{
  out.index    = j.value("index", -1);
  out.texCoord = j.value("texCoord", 0);
  if(j.contains("extensions"))
    for(auto it = j["extensions"].begin(); it != j["extensions"].end(); ++it)
      out.extensions[it.key()] = toValue(it.value());
  if(j.contains("extras"))
    out.extras = toValue(j["extras"]);
}

}  // namespace

Material fromGltf(const tinygltf::Material& g)
{
  Material m;
  m.name = g.name;
  // the base
  const auto& factor = g.pbrMetallicRoughness.baseColorFactor;
  for(size_t i = 0; i < std::min<size_t>(4, factor.size()); i++)
    m.baseColor[i] = float(factor[i]);
  m.baseColorMap         = slotOf(g.pbrMetallicRoughness.baseColorTexture);
  m.metallic             = float(g.pbrMetallicRoughness.metallicFactor);
  m.roughness            = float(g.pbrMetallicRoughness.roughnessFactor);
  m.metallicRoughnessMap = slotOf(g.pbrMetallicRoughness.metallicRoughnessTexture);
  m.alphaMode   = g.alphaMode == "OPAQUE" ? AlphaMode::eOpaque : g.alphaMode == "MASK" ? AlphaMode::eMask : AlphaMode::eBlend;
  m.alphaCutoff = float(g.alphaCutoff);
  m.doubleSided = g.doubleSided;
  m.normalMap   = slotOf(g.normalTexture);
  if(g.normalTexture.index >= 0 || g.normalTexture.scale != 1)
    m.normalMap.scale = float(g.normalTexture.scale);
  m.occlusionMap = slotOf(g.occlusionTexture);
  if(g.occlusionTexture.index >= 0 || g.occlusionTexture.strength != 1)
    m.occlusionMap.scale = float(g.occlusionTexture.strength);
  for(size_t i = 0; i < std::min<size_t>(3, g.emissiveFactor.size()); i++)
    m.emissive[i] = float(g.emissiveFactor[i]);
  m.emissiveMap = slotOf(g.emissiveTexture);

  // the extensions, and the rest kept to write back
  for(const auto& [name, v] : g.extensions)
  {
    if(name == "KHR_materials_emissive_strength")
    {
      float strength = 1;
      read(v, "emissiveStrength", strength);
      m.emissiveStrength = strength;
    }
    else if(name == "KHR_materials_specular")
    {
      Specular s;
      read(v, "specularFactor", s.factor);
      read(v, "specularColorFactor", s.color);
      read(v, "specularTexture", s.map);
      read(v, "specularColorTexture", s.colorMap);
      m.specular = s;
    }
    else if(name == "KHR_materials_ior")
    {
      float ior = 1.5f;
      read(v, "ior", ior);
      m.ior = ior;
    }
    else if(name == "KHR_materials_clearcoat")
    {
      Clearcoat c;
      read(v, "clearcoatFactor", c.factor);
      read(v, "clearcoatRoughnessFactor", c.roughness);
      read(v, "clearcoatTexture", c.map);
      read(v, "clearcoatRoughnessTexture", c.roughnessMap);
      read(v, "clearcoatNormalTexture", c.normalMap);
      read(v, "fvCoatColor", c.color);
      read(v, "fvCoatColorTexture", c.colorMap);
      m.clearcoat = c;
    }
    else if(name == kSpecular2)
    {
      Specular2 s;
      read(v, "specular2Factor", s.factor);
      read(v, "specular2RoughnessFactor", s.roughness);
      read(v, "specular2Texture", s.map);
      read(v, "specular2RoughnessTexture", s.roughnessMap);
      m.specular2 = s;
    }
    else if(name == "KHR_materials_sheen")
    {
      Sheen s;
      read(v, "sheenColorFactor", s.color);
      read(v, "sheenRoughnessFactor", s.roughness);
      read(v, "sheenColorTexture", s.colorMap);
      read(v, "sheenRoughnessTexture", s.roughnessMap);
      m.sheen = s;
    }
    else if(name == "KHR_materials_transmission")
    {
      Transmission t;
      read(v, "transmissionFactor", t.factor);
      read(v, "transmissionTexture", t.map);
      m.transmission = t;
    }
    else if(name == "KHR_materials_volume")
    {
      Volume o;
      read(v, "thicknessFactor", o.thickness);
      read(v, "thicknessTexture", o.thicknessMap);
      read(v, "attenuationDistance", o.attenuationDistance);
      read(v, "attenuationColor", o.attenuationColor);
      m.volume = o;
    }
    else if(name == kSubsurface)
    {
      Rgb distance{0, 0, 0};
      read(v, "scatterDistance", distance);
      m.scatterDistance = distance;
    }
    else if(name == kScatter || name == kScatterDraft)
    {
      // The draft had no strength: scattering was always at full strength, and its colour defaulted to
      // black, which is what turned it off. The current name wins if a file has both.
      if(name == kScatterDraft && g.extensions.count(kScatter))
        continue;
      Scatter s;
      if(name == kScatterDraft)
      {
        s.strength          = 1;
        s.multiscatterColor = {0, 0, 0};
      }
      read(v, "scatterStrengthFactor", s.strength);
      read(v, "scatterStrengthTexture", s.strengthMap);
      read(v, "multiscatterColorFactor", s.multiscatterColor);
      read(v, "multiscatterColor", s.multiscatterColor);  // the draft's
      read(v, "multiscatterColorTexture", s.multiscatterColorMap);
      read(v, "scatterAnisotropy", s.anisotropy);
      m.scatter = s;
    }
    else if(name == "KHR_materials_anisotropy")
    {
      Anisotropy a;
      read(v, "anisotropyStrength", a.strength);
      read(v, "anisotropyRotation", a.rotation);
      read(v, "anisotropyTexture", a.map);
      m.anisotropy = a;
    }
    else if(name == "KHR_materials_iridescence")
    {
      Iridescence i;
      read(v, "iridescenceFactor", i.factor);
      read(v, "iridescenceIor", i.ior);
      read(v, "iridescenceThicknessMinimum", i.thicknessMinimum);
      read(v, "iridescenceThicknessMaximum", i.thicknessMaximum);
      read(v, "iridescenceTexture", i.map);
      read(v, "iridescenceThicknessTexture", i.thicknessMap);
      m.iridescence = i;
    }
    else if(name == "KHR_materials_dispersion")
    {
      float dispersion = 0;
      read(v, "dispersion", dispersion);
      m.dispersion = dispersion;
    }
    else if(name == "KHR_materials_diffuse_transmission")
    {
      DiffuseTransmission d;
      read(v, "diffuseTransmissionFactor", d.factor);
      read(v, "diffuseTransmissionColorFactor", d.color);
      read(v, "diffuseTransmissionTexture", d.map);
      read(v, "diffuseTransmissionColorTexture", d.colorMap);
      m.diffuseTransmission = d;
    }
    else if(name == "KHR_materials_retroreflection")
    {
      Retroreflection r;
      read(v, "retroreflectionFactor", r.factor);
      read(v, "retroreflectionTexture", r.map);
      m.retroreflection = r;
    }
    else if(name == "KHR_materials_pbrSpecularGlossiness")
    {
      SpecularGlossiness s;
      read(v, "diffuseFactor", s.diffuse);
      read(v, "specularFactor", s.specular);
      read(v, "glossinessFactor", s.glossiness);
      read(v, "diffuseTexture", s.diffuseMap);
      read(v, "specularGlossinessTexture", s.specularGlossinessMap);
      m.specularGlossiness = s;
    }
    else if(name == "FV_materials_detail")
    {
      Detail d;
      read(v, "detailWeightFactor", d.weight);
      read(v, "detailRoughnessFactor", d.roughness);
      read(v, "detailNormalTexture", d.normalMap);
      read(v, "detailWeightTexture", d.weightMap);
      read(v, "detailRoughnessTexture", d.roughnessMap);
      m.detail = d;
    }
    else if(name == "FV_materials_hair")
    {
      Hair h;
      read(v, "longitudinalRoughness", h.longitudinal);
      read(v, "azimuthalRoughness", h.azimuthal);
      read(v, "rootAbsorption", h.rootAbsorption);
      read(v, "tipAbsorption", h.tipAbsorption);
      read(v, "rootFade", h.rootFade);
      read(v, "rootV", h.rootV);
      read(v, "specularFactor", h.specular);
      read(v, "absorptionFromColor", h.absorptionFromColor);
      if(v.IsObject() && v.Has("strandRotation"))
      {
        float rotation = 0;
        read(v, "strandRotation", rotation);
        h.strandRotation = rotation;
      }
      m.hair = h;
    }
    else if(name == "EXT_DLSS_NR")
    {
      Rgba mask{1, 1, 1, 1};
      read(v, "nrMask", mask);
      m.dlssNrMask = mask;
    }
    else if(name == "KHR_materials_unlit")
      m.unlit = true;
    else
      m.foreignExtensions[name] = v;
  }

  // extras: the two the renderer reads, and the rest
  if(g.extras.IsObject())
  {
    tinygltf::Value::Object extras = g.extras.Get<tinygltf::Value::Object>();
    if(const auto it = extras.find("fvId"); it != extras.end() && it->second.IsString())
    {
      m.id = it->second.Get<std::string>();
      extras.erase(it);
    }
    if(const auto it = extras.find("fvWalkThrough"); it != extras.end() && it->second.IsBool())
    {
      m.walkThrough = it->second.Get<bool>();
      extras.erase(it);
    }
    if(!extras.empty())
      m.foreignExtras = tinygltf::Value(std::move(extras));
  }
  else
    m.foreignExtras = g.extras;
  return m;
}

Json materialJson(const Material& m, const char* key)
{
  const auto info = [&](const TextureSlot& s) { return infoJson(s, key); };

  Json out = {{"name", m.name}};
  Json pbr = {{"baseColorFactor", rgbaJson(m.baseColor)}, {"metallicFactor", m.metallic}, {"roughnessFactor", m.roughness}};
  if(m.baseColorMap.has())
    pbr["baseColorTexture"] = info(m.baseColorMap);
  if(m.metallicRoughnessMap.has())
    pbr["metallicRoughnessTexture"] = info(m.metallicRoughnessMap);
  out["pbrMetallicRoughness"] = pbr;
  out["alphaMode"]            = m.alphaMode == AlphaMode::eOpaque ? "OPAQUE" : m.alphaMode == AlphaMode::eMask ? "MASK" : "BLEND";
  if(m.alphaMode == AlphaMode::eMask || m.alphaCutoff != 0.5f)
    out["alphaCutoff"] = m.alphaCutoff;
  out["doubleSided"] = m.doubleSided;
  if(m.normalMap.has())
    out["normalTexture"] = info(m.normalMap);
  if(m.occlusionMap.has())
    out["occlusionTexture"] = infoJson(m.occlusionMap, key, "strength");
  if(m.emissive != Rgb{0, 0, 0})
    out["emissiveFactor"] = rgbJson(m.emissive);
  if(m.emissiveMap.has())
    out["emissiveTexture"] = info(m.emissiveMap);

  Json ext = Json::object();
  if(m.emissiveStrength)
    ext["KHR_materials_emissive_strength"] = {{"emissiveStrength", *m.emissiveStrength}};
  if(m.specular)
  {
    Json e = {{"specularFactor", m.specular->factor}, {"specularColorFactor", rgbJson(m.specular->color)}};
    if(m.specular->map.has())
      e["specularTexture"] = info(m.specular->map);
    if(m.specular->colorMap.has())
      e["specularColorTexture"] = info(m.specular->colorMap);
    ext["KHR_materials_specular"] = e;
  }
  if(m.ior)
    ext["KHR_materials_ior"] = {{"ior", *m.ior}};
  if(m.clearcoat)
  {
    const Clearcoat& c = *m.clearcoat;
    Json             e = {{"clearcoatFactor", c.factor}, {"clearcoatRoughnessFactor", c.roughness}};
    if(c.map.has())
      e["clearcoatTexture"] = info(c.map);
    if(c.roughnessMap.has())
      e["clearcoatRoughnessTexture"] = info(c.roughnessMap);
    if(c.normalMap.has())
      e["clearcoatNormalTexture"] = info(c.normalMap);
    if(c.color != Rgb{1, 1, 1})
      e["fvCoatColor"] = rgbJson(c.color);
    if(c.colorMap.has())
      e["fvCoatColorTexture"] = info(c.colorMap);
    ext["KHR_materials_clearcoat"] = e;
  }
  if(m.specular2)
  {
    Json e = {{"specular2Factor", m.specular2->factor}, {"specular2RoughnessFactor", m.specular2->roughness}};
    if(m.specular2->map.has())
      e["specular2Texture"] = info(m.specular2->map);
    if(m.specular2->roughnessMap.has())
      e["specular2RoughnessTexture"] = info(m.specular2->roughnessMap);
    ext[kSpecular2] = e;
  }
  if(m.sheen)
  {
    Json e = {{"sheenColorFactor", rgbJson(m.sheen->color)}, {"sheenRoughnessFactor", m.sheen->roughness}};
    if(m.sheen->colorMap.has())
      e["sheenColorTexture"] = info(m.sheen->colorMap);
    if(m.sheen->roughnessMap.has())
      e["sheenRoughnessTexture"] = info(m.sheen->roughnessMap);
    ext["KHR_materials_sheen"] = e;
  }
  if(m.transmission)
  {
    Json e = {{"transmissionFactor", m.transmission->factor}};
    if(m.transmission->map.has())
      e["transmissionTexture"] = info(m.transmission->map);
    ext["KHR_materials_transmission"] = e;
  }
  if(m.volume)
  {
    Json e = {{"thicknessFactor", m.volume->thickness}, {"attenuationColor", rgbJson(m.volume->attenuationColor)}};
    if(m.volume->attenuationDistance != FLT_MAX)
      e["attenuationDistance"] = m.volume->attenuationDistance;
    if(m.volume->thicknessMap.has())
      e["thicknessTexture"] = info(m.volume->thicknessMap);
    ext["KHR_materials_volume"] = e;
  }
  if(m.scatter)
  {
    Json e = {{"scatterStrengthFactor", m.scatter->strength},
              {"multiscatterColorFactor", rgbJson(m.scatter->multiscatterColor)},
              {"scatterAnisotropy", m.scatter->anisotropy}};
    if(m.scatter->strengthMap.has())
      e["scatterStrengthTexture"] = info(m.scatter->strengthMap);
    if(m.scatter->multiscatterColorMap.has())
      e["multiscatterColorTexture"] = info(m.scatter->multiscatterColorMap);
    ext[kScatter] = e;
  }
  if(m.scatterDistance)
    ext[kSubsurface] = {{"scatterDistance", rgbJson(*m.scatterDistance)}};
  if(m.anisotropy)
  {
    Json e = {{"anisotropyStrength", m.anisotropy->strength}, {"anisotropyRotation", m.anisotropy->rotation}};
    if(m.anisotropy->map.has())
      e["anisotropyTexture"] = info(m.anisotropy->map);
    ext["KHR_materials_anisotropy"] = e;
  }
  if(m.iridescence)
  {
    const Iridescence& i = *m.iridescence;
    Json               e = {{"iridescenceFactor", i.factor},
                            {"iridescenceIor", i.ior},
                            {"iridescenceThicknessMinimum", i.thicknessMinimum},
                            {"iridescenceThicknessMaximum", i.thicknessMaximum}};
    if(i.map.has())
      e["iridescenceTexture"] = info(i.map);
    if(i.thicknessMap.has())
      e["iridescenceThicknessTexture"] = info(i.thicknessMap);
    ext["KHR_materials_iridescence"] = e;
  }
  if(m.dispersion)
    ext["KHR_materials_dispersion"] = {{"dispersion", *m.dispersion}};
  if(m.diffuseTransmission)
  {
    const DiffuseTransmission& d = *m.diffuseTransmission;
    Json e = {{"diffuseTransmissionFactor", d.factor}, {"diffuseTransmissionColorFactor", rgbJson(d.color)}};
    if(d.map.has())
      e["diffuseTransmissionTexture"] = info(d.map);
    if(d.colorMap.has())
      e["diffuseTransmissionColorTexture"] = info(d.colorMap);
    ext["KHR_materials_diffuse_transmission"] = e;
  }
  if(m.retroreflection)
  {
    Json e = {{"retroreflectionFactor", m.retroreflection->factor}};
    if(m.retroreflection->map.has())
      e["retroreflectionTexture"] = info(m.retroreflection->map);
    ext["KHR_materials_retroreflection"] = e;
  }
  if(m.specularGlossiness)
  {
    const SpecularGlossiness& s = *m.specularGlossiness;
    Json e = {{"diffuseFactor", rgbaJson(s.diffuse)}, {"specularFactor", rgbJson(s.specular)}, {"glossinessFactor", s.glossiness}};
    if(s.diffuseMap.has())
      e["diffuseTexture"] = info(s.diffuseMap);
    if(s.specularGlossinessMap.has())
      e["specularGlossinessTexture"] = info(s.specularGlossinessMap);
    ext["KHR_materials_pbrSpecularGlossiness"] = e;
  }
  if(m.detail)
  {
    Json e = {{"detailWeightFactor", m.detail->weight}, {"detailRoughnessFactor", m.detail->roughness}};
    if(m.detail->normalMap.has())
      e["detailNormalTexture"] = info(m.detail->normalMap);
    if(m.detail->weightMap.has())
      e["detailWeightTexture"] = info(m.detail->weightMap);
    if(m.detail->roughnessMap.has())
      e["detailRoughnessTexture"] = info(m.detail->roughnessMap);
    ext["FV_materials_detail"] = e;
  }
  if(m.hair)
  {
    const Hair& h = *m.hair;
    Json e = {{"longitudinalRoughness", h.longitudinal}, {"azimuthalRoughness", h.azimuthal}, {"specularFactor", h.specular}};
    if(h.absorptionFromColor)
      e["absorptionFromColor"] = true;
    else
    {
      e["rootAbsorption"] = rgbJson(h.rootAbsorption);
      e["tipAbsorption"]  = rgbJson(h.tipAbsorption);
      e["rootFade"]       = h.rootFade;
      e["rootV"]          = h.rootV;
    }
    if(h.strandRotation)
      e["strandRotation"] = *h.strandRotation;
    ext["FV_materials_hair"] = e;
  }
  if(m.dlssNrMask)
    ext["EXT_DLSS_NR"] = {{"nrMask", rgbaJson(*m.dlssNrMask)}};
  if(m.unlit)
    ext["KHR_materials_unlit"] = Json::object();
  for(const auto& [name, value] : m.foreignExtensions)
    ext[name] = fromValue(value);
  out["extensions"] = ext;

  Json extras = m.foreignExtras.IsObject() ? fromValue(m.foreignExtras) : Json::object();
  if(!m.id.empty())
    extras["fvId"] = m.id;
  if(m.walkThrough)
    extras["fvWalkThrough"] = true;
  if(!extras.empty())
    out["extras"] = extras;
  else if(m.foreignExtras.Type() != tinygltf::NULL_TYPE && !m.foreignExtras.IsObject())
    out["extras"] = fromValue(m.foreignExtras);
  return out;
}

tinygltf::Material toGltf(const Material& material, std::vector<std::string>& extensionsUsed)
{
  const Json         j = materialJson(material);
  tinygltf::Material m;
  m.name          = j.value("name", std::string());
  const Json& pbr = j["pbrMetallicRoughness"];
  m.pbrMetallicRoughness.baseColorFactor = pbr["baseColorFactor"].get<std::vector<double>>();
  m.pbrMetallicRoughness.metallicFactor  = pbr["metallicFactor"].get<double>();
  m.pbrMetallicRoughness.roughnessFactor = pbr["roughnessFactor"].get<double>();
  if(pbr.contains("baseColorTexture"))
    textureInfo(pbr["baseColorTexture"], m.pbrMetallicRoughness.baseColorTexture);
  if(pbr.contains("metallicRoughnessTexture"))
    textureInfo(pbr["metallicRoughnessTexture"], m.pbrMetallicRoughness.metallicRoughnessTexture);
  m.alphaMode   = j.value("alphaMode", std::string("OPAQUE"));
  m.alphaCutoff = j.value("alphaCutoff", 0.5);
  m.doubleSided = j.value("doubleSided", false);
  if(j.contains("emissiveFactor"))
    m.emissiveFactor = j["emissiveFactor"].get<std::vector<double>>();
  if(j.contains("emissiveTexture"))
    textureInfo(j["emissiveTexture"], m.emissiveTexture);
  if(j.contains("normalTexture"))
  {
    const Json&           n = j["normalTexture"];
    tinygltf::TextureInfo info;
    textureInfo(n, info);
    m.normalTexture.index      = info.index;
    m.normalTexture.texCoord   = info.texCoord;
    m.normalTexture.extensions = std::move(info.extensions);
    m.normalTexture.extras     = std::move(info.extras);
    m.normalTexture.scale      = n.value("scale", 1.0);
  }
  if(j.contains("occlusionTexture"))
  {
    const Json&           o = j["occlusionTexture"];
    tinygltf::TextureInfo info;
    textureInfo(o, info);
    m.occlusionTexture.index      = info.index;
    m.occlusionTexture.texCoord   = info.texCoord;
    m.occlusionTexture.extensions = std::move(info.extensions);
    m.occlusionTexture.extras     = std::move(info.extras);
    m.occlusionTexture.strength   = o.value("strength", 1.0);
  }
  collectInfoExtensions(j, extensionsUsed);
  for(auto it = j["extensions"].begin(); it != j["extensions"].end(); ++it)
  {
    m.extensions[it.key()] = toValue(it.value());
    useExtension(extensionsUsed, it.key());
  }
  if(j.contains("extras"))
    m.extras = toValue(j["extras"]);
  return m;
}

tinygltf::Material toGltf(const Material& material)
{
  std::vector<std::string> used;
  return toGltf(material, used);
}

tinygltf::Value toValue(const Json& j)
{
  using V = tinygltf::Value;
  if(j.is_object())
  {
    V::Object o;
    for(auto it = j.begin(); it != j.end(); ++it)
      o[it.key()] = toValue(it.value());
    return V(std::move(o));
  }
  if(j.is_array())
  {
    V::Array a;
    for(const Json& x : j)
      a.push_back(toValue(x));
    return V(std::move(a));
  }
  if(j.is_boolean())
    return V(j.get<bool>());
  if(j.is_number_integer())
    return V(int(j.get<int64_t>()));
  if(j.is_number())
    return V(j.get<double>());
  if(j.is_string())
    return V(j.get<std::string>());
  return V();
}

Json fromValue(const tinygltf::Value& v)
{
  if(v.IsObject())
  {
    Json o = Json::object();
    for(const auto& [key, value] : v.Get<tinygltf::Value::Object>())
      o[key] = fromValue(value);
    return o;
  }
  if(v.IsArray())
  {
    Json a = Json::array();
    for(const tinygltf::Value& x : v.Get<tinygltf::Value::Array>())
      a.push_back(fromValue(x));
    return a;
  }
  if(v.IsBool())
    return v.Get<bool>();
  if(v.IsInt())
    return v.Get<int>();
  if(v.IsNumber())
    return v.GetNumberAsDouble();
  if(v.IsString())
    return v.Get<std::string>();
  return Json();
}

}  // namespace fv::mat
