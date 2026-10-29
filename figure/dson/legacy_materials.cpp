#include <figure/dson/legacy_materials.hpp>

#include <algorithm>

namespace fv::dson {

namespace {

// The lowest roughness a legacy highlight becomes, unless the surface reflects
constexpr float kRoughnessFloor = 0.3f;
// Measured skin's specular, as DAZ's own Iray conversion of Genesis 2's skin (Iray Optimized
// Genesis 2 Female MAT: Glossy Layered Weight 1, Reflectivity 0.35, Roughness 0.4)
constexpr float kSkinReflectivity = 0.35f;
constexpr float kSkinRoughness    = 0.4f;
// A refracting surface whose file gives no usable index (DAZ Default's ior defaults to 0)
constexpr float kDefaultIor = 1.5f;

double num(const Channel* c, double d)
{
  return c && c->value.is_number() ? c->value.get<double>() : d;
}

// A switch: an "... Active" bool or a 0/1 number; `d` when the channel is missing
bool on(const Channel* c, bool d)
{
  if(!c)
    return d;
  if(c->value.is_boolean())
    return c->value.get<bool>();
  if(c->value.is_number())
    return c->value.get<double>() != 0;
  return d;
}

// A specular colour only switches its lobe off when black: its tint is a 3Delight colour trick, and
// its brightness was a second strength (DAZ's Iray Boxers keep the strength as the weight)
double lit(const Channel* c)
{
  if(!c || !c->value.is_array() || c->value.size() < 3)
    return 1;
  double m = 0;
  for(int i = 0; i < 3; i++)
    m = std::max(m, c->value[i].is_number() ? c->value[i].get<double>() : 0.0);
  return m > 0.01 ? 1 : 0;
}

float ior(const Channel* c)
{
  const double v = num(c, 0);
  return v >= 1.01 ? float(v) : kDefaultIor;
}

}  // namespace

LegacyShader classifyLegacy(const Channels& ch)
{
  auto has = [&](std::initializer_list<const char*> n) { return ch.find(n) != nullptr; };
  // Iray Uber, PBRSkin, OmniHair: parseSurface's own
  if(has({"Glossy Roughness", "Translucency Weight", "Metallic Weight", "Specular Lobe 2 Roughness Mult", "Detail Enable", "Makeup Enable", "Melanin"}))
    return LegacyShader::eNone;
  if(has({"sssmaterial", "cScatter_x", "value23"}))
    return LegacyShader::eAoa;
  if(has({"Subsurface Active", "Specular2 Roughness", "Map Tiling U"}))
    return LegacyShader::eUberSurface;
  // after the two above, which have a Diffuse and a Specular Strength too
  if(has({"diffuse_strength", "specular_strength", "Lighting Model"}))
    return LegacyShader::eDazDefault;
  return LegacyShader::eNone;
}

float legacyRoughness(LegacyShader shader, float glossiness, bool reflective)
{
  const float g = std::clamp(glossiness, 0.f, 1.f);
  // A surface that reflects the scene (3Delight's mirror term) is as sharp as its glossiness says
  if(reflective)
    return std::max(1 - g, 0.02f);
  // A highlight from the lights alone: DAZ Default's narrows over the top half of Glossiness only
  // (50% looks like UberSurface's 4%); UberSurface and AoA run on the whole range (UberSurface 50%
  // looks like Default's 85%, both about 0.5)
  const float r = shader == LegacyShader::eDazDefault ? 1 - (g - 0.47f) * (0.7f / 0.53f) : 1 - g;
  return std::clamp(r, kRoughnessFloor, 1.f);
}

void translateLegacy(LegacyShader shader, const Channels& ch, mat::Surface& s)
{
  using mat::Kind;
  using Shader = mat::Surface::Shader;
  if(shader == LegacyShader::eNone)
    return;
  auto       find       = [&](std::initializer_list<const char*> n) { return ch.find(n); };
  const bool dazDefault = shader == LegacyShader::eDazDefault;
  const bool uber       = shader == LegacyShader::eUberSurface;
  const bool aoa        = shader == LegacyShader::eAoa;
  const bool skin       = s.kind == Kind::eSkin;
  s.shader              = dazDefault ? Shader::eDazDefault : uber ? Shader::eUberSurface : Shader::eAoa;

  // DAZ Default's Lighting Model: Plastic, Metallic, Skin, Glossy (Plastic), Matte, Glossy (Metallic)
  const int  model = dazDefault ? int(num(find({"Lighting Model"}), 0)) : 0;
  const bool metal = model == 1 || model == 5;
  const bool matte = model == 4;
  // the share of light the shader sends through its subsurface instead of the plain diffuse
  const bool subsurface = uber ? on(find({"Subsurface Active"}), false) : aoa ? on(find({"value23"}), false) : model == 2;

  // Diffuse: off, or scaled by Diffuse Strength. Not on skin, nor with the subsurface on: there the
  // diffuse map is the albedo of the skin's volume, and the strength was 3Delight's split between
  // its diffuse and subsurface terms.
  if(uber && !on(find({"Diffuse Active"}), true))
  {
    s.base.color = {0, 0, 0};
    s.base.map.clear();
  }
  else if(!skin && !subsurface)
  {
    const float k = float(std::clamp(num(find({"Diffuse Strength", "diffuse_strength"}), 1), 0.0, 1.0));
    for(float& c : s.base.color)
      c *= k;
  }

  if(uber && !on(find({"Opacity Active"}), true))
    s.opacity = {1, {}};
  if(uber)
  {
    const double u = num(find({"Map Tiling U"}), 1), v = num(find({"Map Tiling V"}), 1);
    s.tiles        = {float(u != 0 ? u : 1), float(v != 0 ? v : 1)};
  }
  if(uber && !on(find({"Bump Active"}), true))
    s.bumpMap.clear();
  // The bump map's height range, in DAZ's centimetres: Negative and Positive Bump (DAZ Default's
  // bump_min and bump_max, AoA's), UberSurface's Bump Minimum and Maximum. Bump Strength scales it
  // as it does an Iray bump. An empty range (AoA's brick leaves both 0) keeps the fixed bake.
  const double low   = num(find({"Negative Bump", "Bump Minimum"}), 0);
  const double high  = num(find({"Positive Bump", "Bump Maximum"}), 0);
  if(high > low)
    s.bumpRange = float((high - low) * 0.01);
  // AoA's normal map strength is a channel of its own ("alpha"), not the map channel's value
  if(aoa && !s.normalMap.empty())
    s.normalStrength = float(num(find({"alpha"}), 1));

  // What reflects the scene: a reflection, a metal, and (below) refraction
  bool reflective = metal;
  if(dazDefault)
    reflective = reflective || num(find({"reflection_strength"}), 0) > 0;
  if(uber)
    reflective = reflective || (on(find({"Reflection Active"}), false) && num(find({"Reflection Strength"}), 0) > 0);
  if(aoa)
    reflective = reflective || num(find({"Reflection Strength"}), 0) > 0;

  // Refraction, never on hair and lashes (parseSurface's rule). AoA's "ior" is its subsurface's.
  s.refraction.reset();
  if(s.kind != Kind::eHair && s.kind != Kind::eLashes)
  {
    if(dazDefault && num(find({"refraction_strength"}), 0) > 0)
      s.refraction = mat::Surface::Refraction{float(std::clamp(num(find({"refraction_strength"}), 0), 0.0, 1.0)), ior(find({"ior"}))};
    // UberSurface refracts what its opacity lets through
    if(uber && on(find({"Refraction Active"}), false) && s.opacity.map.empty() && s.opacity.value < 1)
    {
      s.refraction = mat::Surface::Refraction{1 - s.opacity.value, ior(find({"Index of Refraction"}))};
      s.opacity    = {1, {}};
    }
  }
  reflective = reflective || s.refraction.has_value();

  s.dual.reset();
  s.topCoat.reset();
  // the glossy layer's values are replaced below, so maps read with Iray's names no longer belong to them
  s.glossy.weightMap.clear();
  s.glossy.reflectivityMap.clear();
  if(skin)
  {
    s.glossy.weight       = 1;
    s.glossy.reflectivity = kSkinReflectivity;
    s.glossy.roughness    = {kSkinRoughness, {}};
  }
  else
  {
    // Specular strength, and the second lobe's on top: one glossy layer
    double weight = 0;
    if(dazDefault)
      weight = num(find({"specular_strength"}), 0) * lit(find({"specular"}));
    else if(uber)
      weight = (on(find({"Specular Active"}), true) ? num(find({"Specular Strength"}), 0) * lit(find({"Specular Color"})) : 0)
               + (on(find({"Specular2 Active"}), false) ? num(find({"Specular2 Strength"}), 0) * lit(find({"Specular2 Color"})) : 0);
    else
      weight = num(find({"Specular Strength"}), 0) * lit(find({"Specular Color"})) + num(find({"value222"}), 0) * lit(find({"color2"}));
    s.glossy.weight       = float(std::clamp(weight, 0.0, 1.0));
    s.glossy.reflectivity = 0.5f;
    const float gloss     = float(num(find({"Glossiness"}), 0.5));
    s.glossy.roughness    = {matte ? 1.f : legacyRoughness(shader, gloss, reflective), {}};
  }
  if(metal)
    s.metallic = {1, {}};
}

}  // namespace fv::dson
