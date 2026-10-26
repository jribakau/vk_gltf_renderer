#include <figure/materials/edits_json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <figure/materials/fabric.hpp>
#include <figure/materials/texture_request.hpp>

namespace fv::mat {

namespace {

float numberOf(const EditsJson& v, const char* key, float lo, float hi)
{
  if(!v.is_number())
    throw std::runtime_error(std::string(key) + " must be a number");
  return std::clamp(v.get<float>(), lo, hi);
}

// A map as the library knows it: "/Runtime/Textures/...", forward slashes. "" stays: no map.
std::string libraryRel(const std::string& key, std::string rel)
{
  std::replace(rel.begin(), rel.end(), '\\', '/');
  if(rel.empty())
    return rel;
  bool up = false;
  for(size_t from = 0; from <= rel.size();)
  {
    const size_t at = std::min(rel.find('/', from), rel.size());
    up              = up || rel.compare(from, at - from, "..") == 0;
    from            = at + 1;
  }
  // a drive or a scheme, a share, a way out of the library, or a layered map's separator
  if(rel.find(':') != std::string::npos || rel.rfind("//", 0) == 0 || up || rel.find(kLayerSeparator) != std::string::npos)
    throw std::runtime_error(key + " must be an image in the library (\"/Runtime/Textures/...\"), not " + rel);
  if(rel.front() != '/')
    rel.insert(rel.begin(), '/');
  return rel;
}

struct StringField
{
  const char*                       name;
  std::optional<std::string> Edits::*member;
};
constexpr StringField kMaps[] = {
    {"baseMap", &Edits::baseMap},     {"opacityMap", &Edits::opacityMap},     {"normalMap", &Edits::normalMap},
    {"bumpMap", &Edits::bumpMap},     {"roughnessMap", &Edits::roughnessMap},
};

struct ColorField
{
  const char*               name;
  std::optional<Rgb> Edits::*member;
};
constexpr ColorField kColors[] = {
    {"color", &Edits::color},     {"emissive", &Edits::emissive},       {"irisColor", &Edits::irisColor},
    {"scleraColor", &Edits::scleraColor}, {"recolor", &Edits::recolor}, {"tone", &Edits::tone},
    {"makeupColor", &Edits::makeupColor}, {"sheenColor", &Edits::sheenColor},
};

struct NumberField
{
  const char*                 name;
  std::optional<float> Edits::*member;
  float                       lo, hi;
};
constexpr NumberField kNumbers[] = {
    {"roughness", &Edits::roughness, 0, 1},
    {"metalness", &Edits::metalness, 0, 1},
    {"opacity", &Edits::opacity, 0, 1},
    {"normalStrength", &Edits::normalStrength, 0, 3},
    {"bumpStrength", &Edits::bumpStrength, 0, 2},
    {"specular", &Edits::specular, 0, 1},
    {"clearcoat", &Edits::clearcoat, 0, 1},
    {"clearcoatRoughness", &Edits::clearcoatRoughness, 0, 1},
    {"emissiveIntensity", &Edits::emissiveIntensity, 0, 1000},
    {"irisStrength", &Edits::irisStrength, 0, 1},
    {"recolorStrength", &Edits::recolorStrength, 0, 1},
    {"toneStrength", &Edits::toneStrength, 0, 1},
    {"makeupWeight", &Edits::makeupWeight, 0, 1},
    {"scatterScale", &Edits::scatterScale, 0.1f, 4},
    {"uvScaleU", &Edits::uvScaleU, 0.01f, 100},
    {"uvScaleV", &Edits::uvScaleV, 0.01f, 100},
    {"uvOffsetU", &Edits::uvOffsetU, -10, 10},
    {"uvOffsetV", &Edits::uvOffsetV, -10, 10},
    {"uvRotation", &Edits::uvRotation, -360, 360},
    {"fabricScale", &Edits::fabricScale, 0.1f, 10},
    {"fabricStrength", &Edits::fabricStrength, 0, 2},
    {"sheenRoughness", &Edits::sheenRoughness, 0, 1},
};

}  // namespace

std::string hexOf(const Rgb& linear)
{
  char buf[8];
  int  c[3];
  for(int i = 0; i < 3; i++)
    c[i] = int(std::lround(std::clamp(linearToSrgb(std::clamp(linear[i], 0.f, 1.f)), 0.f, 1.f) * 255.f));
  std::snprintf(buf, sizeof buf, "#%02x%02x%02x", c[0], c[1], c[2]);
  return buf;
}

std::optional<Rgb> parseHex(const EditsJson& v)
{
  if(!v.is_string())
    return std::nullopt;
  const std::string s = v.get<std::string>();
  if(s.size() != 7 || s[0] != '#')
    return std::nullopt;
  Rgb out;
  for(int i = 0; i < 3; i++)
  {
    char*      end = nullptr;
    const auto n   = std::strtol(s.substr(1 + i * 2, 2).c_str(), &end, 16);
    if(*end)
      return std::nullopt;
    out[i] = srgbToLinear(n / 255.0);
  }
  return out;
}

void applyValue(Edits& e, const std::string& key, const EditsJson& v)
{
  const bool drop = v.is_null();
  for(const ColorField& f : kColors)
    if(key == f.name)
    {
      if(drop)
        (e.*f.member).reset();
      else if(const auto c = parseHex(v))
        e.*f.member = *c;
      else
        throw std::runtime_error(key + " must be a #rrggbb colour");
      return;
    }
  for(const StringField& f : kMaps)
    if(key == f.name)
    {
      if(drop)
        (e.*f.member).reset();
      else if(v.is_string())
        e.*f.member = libraryRel(key, v.get<std::string>());
      else
        throw std::runtime_error(key + " must be an image's path, or \"\" for none");
      return;
    }
  for(const NumberField& f : kNumbers)
    if(key == f.name)
    {
      if(drop)
        (e.*f.member).reset();
      else
        e.*f.member = numberOf(v, f.name, f.lo, f.hi);
      return;
    }
  if(key == "alphaMode")
  {
    if(drop)
      e.alphaMode.reset();
    else if(v == "opaque" || v == "cutout" || v == "blend")
      e.alphaMode = v.get<std::string>();
    else
      throw std::runtime_error("alphaMode is opaque, cutout or blend");
  }
  else if(key == "doubleSided")
  {
    if(drop)
      e.doubleSided.reset();
    else if(v.is_boolean())
      e.doubleSided = v.get<bool>();
    else
      throw std::runtime_error("doubleSided must be a boolean");
  }
  else if(key == "fabric")
  {
    if(drop)
      e.fabric.reset();
    else if(v.is_string() && (v.get<std::string>().empty() || findFabric(v.get<std::string>())))
      e.fabric = v.get<std::string>();
    else
    {
      std::string known;
      for(const Fabric& f : fabrics())
        known += (known.empty() ? "" : ", ") + std::string(f.key);
      throw std::runtime_error("fabric is one of " + known + ", or \"\" for none");
    }
  }
  else
    throw std::runtime_error("unknown material value " + key);
}

EditsJson editedKeys(const Edits& e)
{
  EditsJson  out = EditsJson::array();
  const auto add = [&](bool set, const char* name) {
    if(set)
      out.push_back(name);
  };
  add(e.color.has_value(), "color");
  add(e.roughness.has_value(), "roughness");
  add(e.metalness.has_value(), "metalness");
  add(e.opacity.has_value(), "opacity");
  add(e.alphaMode.has_value(), "alphaMode");
  add(e.doubleSided.has_value(), "doubleSided");
  add(e.normalStrength.has_value(), "normalStrength");
  add(e.bumpStrength.has_value(), "bumpStrength");
  add(e.specular.has_value(), "specular");
  add(e.clearcoat.has_value(), "clearcoat");
  add(e.clearcoatRoughness.has_value(), "clearcoatRoughness");
  add(e.emissive.has_value(), "emissive");
  add(e.emissiveIntensity.has_value(), "emissiveIntensity");
  add(e.irisColor.has_value(), "irisColor");
  add(e.irisStrength.has_value(), "irisStrength");
  add(e.scleraColor.has_value(), "scleraColor");
  add(e.recolor.has_value(), "recolor");
  add(e.recolorStrength.has_value(), "recolorStrength");
  add(e.tone.has_value(), "tone");
  add(e.toneStrength.has_value(), "toneStrength");
  add(e.makeupColor.has_value(), "makeupColor");
  add(e.makeupWeight.has_value(), "makeupWeight");
  add(e.scatterScale.has_value(), "scatterScale");
  for(const StringField& f : kMaps)
    add((e.*f.member).has_value(), f.name);
  add(e.uvScaleU.has_value(), "uvScaleU");
  add(e.uvScaleV.has_value(), "uvScaleV");
  add(e.uvOffsetU.has_value(), "uvOffsetU");
  add(e.uvOffsetV.has_value(), "uvOffsetV");
  add(e.uvRotation.has_value(), "uvRotation");
  add(e.fabric.has_value(), "fabric");
  add(e.fabricScale.has_value(), "fabricScale");
  add(e.fabricStrength.has_value(), "fabricStrength");
  add(e.sheenColor.has_value(), "sheenColor");
  add(e.sheenRoughness.has_value(), "sheenRoughness");
  return out;
}

bool mapKey(std::string_view key)
{
  return std::any_of(std::begin(kMaps), std::end(kMaps), [&](const StringField& f) { return key == f.name; });
}

bool figureOnlyKey(std::string_view key)
{
  return mapKey(key) || key.starts_with("uv") || key.starts_with("fabric");
}

void clothesValues(const Edits& e, EditsJson& values)
{
  for(const StringField& f : kMaps)
    values[f.name] = e.*f.member ? EditsJson(*(e.*f.member)) : EditsJson(nullptr);
  values["uvScaleU"]       = e.uvScaleU.value_or(1.f);
  values["uvScaleV"]       = e.uvScaleV.value_or(1.f);
  values["uvOffsetU"]      = e.uvOffsetU.value_or(0.f);
  values["uvOffsetV"]      = e.uvOffsetV.value_or(0.f);
  values["uvRotation"]     = e.uvRotation.value_or(0.f);
  values["fabric"]         = e.fabric ? EditsJson(*e.fabric) : EditsJson(nullptr);
  values["fabricScale"]    = e.fabricScale.value_or(1.f);
  values["fabricStrength"] = e.fabricStrength.value_or(1.f);
  values["sheenColor"]     = e.sheenColor ? EditsJson(hexOf(*e.sheenColor)) : EditsJson(nullptr);
  values["sheenRoughness"] = e.sheenRoughness.value_or(0.5f);
}

}  // namespace fv::mat
