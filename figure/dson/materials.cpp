#include <figure/dson/materials.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>

namespace fv::dson {

void collectChannels(const Json* def, Channels& into, const std::unordered_map<std::string, std::string>& images, const std::string& file)
{
  if(!def || !def->is_object())
    return;
  auto put = [&](const Json& c) {
    const Json* inner = get(&c, "channel");
    const Json& ch    = inner && !inner->is_null() ? *inner : c;
    std::vector<std::string> own;
    for(const char* k : {"id", "label", "name"})
    {
      const std::string s = getString(&ch, k);
      if(!s.empty())
        own.push_back(norm(s));
    }
    if(own.empty())
      return;
    std::shared_ptr<Channel> prev;
    for(const auto& k : own)
      if((prev = into.shared(k)))
        break;
    // an override also replaces the names the earlier channel was stored under: the figure's Iray
    // "diffuse"/"Base Color" channel is overridden by a character's "diffuse"/"Diffuse Color"
    std::vector<std::string> keys = own;
    if(prev)
      for(const auto& k : prev->keys)
        if(into.shared(k) == prev)
          keys.push_back(k);
    {
      std::vector<std::string> unique;
      for(const auto& k : keys)
        if(std::find(unique.begin(), unique.end(), k) == unique.end())
          unique.push_back(k);
      keys = std::move(unique);
    }
    auto entry = std::make_shared<Channel>();
    if(const Json* v = get(&ch, "current_value"); v && !v->is_null())
      entry->value = *v;
    else if(const Json* v2 = get(&ch, "value"); v2 && !v2->is_null())
      entry->value = *v2;
    else if(prev)
      entry->value = prev->value;
    entry->image = prev ? prev->image : std::nullopt;
    if(ch.contains("image_file") || ch.contains("image"))
    {
      entry->image.reset();
      const std::string imageFile = getString(&ch, "image_file");
      const std::string image     = getString(&ch, "image");
      if(!imageFile.empty())
        entry->image = imageFile;
      else if(!image.empty())
      {
        auto ref = parseRef(image, file);
        if(ref)
          if(auto it = images.find(ref->id); it != images.end())
            entry->image = it->second;
      }
    }
    entry->keys = keys;
    for(const auto& k : keys)
      into.set(k, entry);
  };
  for(auto it = def->begin(); it != def->end(); ++it)
  {
    const Json& v = it.value();
    if(v.is_object() && v.contains("channel"))
      put(v);
    else if(it.key() == "extra" && v.is_array())
      for(const Json& ex : v)
        if(const Json* chans = get(&ex, "channels"); chans && chans->is_array())
          for(const Json& c : *chans)
            put(c);
  }
}

void collectImages(const Json& doc, std::unordered_map<std::string, std::string>& into)
{
  const Json* lib = get(&doc, "image_library");
  if(!lib || !lib->is_array())
    return;
  for(const Json& img : *lib)
  {
    const Json* layers = get(&img, "map");
    if(!layers || !layers->is_array())
      continue;
    for(const Json& l : *layers)
    {
      const Json* active = get(&l, "active");
      if(active && active->is_boolean() && !active->get<bool>())
        continue;
      const std::string url = getString(&l, "url");
      if(url.empty())
        continue;
      const std::string id = getString(&img, "id");
      if(!id.empty())
        into[id] = url;
      break;
    }
  }
}

std::string materialKey(const std::string& name, const Channels& ch)
{
  std::string key = name;
  for(const auto& k : ch.order())
  {
    const Channel* c = ch.get(k);
    key += "\n" + k + "=" + (c->value.is_null() ? std::string() : c->value.dump()) + "|" + c->image.value_or("");
  }
  return key;
}

namespace {

using mat::Rgb;

double num(const Channel* c, double d)
{
  return c && c->value.is_number() ? c->value.get<double>() : d;
}

bool truthy(const Channel* c)
{
  return c && ((c->value.is_boolean() && c->value.get<bool>()) || (c->value.is_number() && c->value.get<double>() != 0));
}

double clamp01(double v)
{
  return std::clamp(v, 0.0, 1.0);
}

// DSON colour channels hold sRGB values, as shown in the DAZ Studio colour pickers.
std::optional<Rgb> color(const Channel* c)
{
  if(!c || !c->value.is_array() || c->value.size() < 3 || !c->value[0].is_number())
    return std::nullopt;
  Rgb out;
  for(int i = 0; i < 3; i++)
    out[i] = mat::srgbToLinear(c->value[i].is_number() ? c->value[i].get<double>() : 0.0);
  return out;
}

Rgb colorOr(const Channel* c, Rgb d = {1, 1, 1})
{
  return color(c).value_or(d);
}

std::string imageOf(const Channel* c)
{
  return c && c->image ? *c->image : std::string();
}

mat::ScalarMap scalar(const Channel* c, double d)
{
  return {float(num(c, d)), imageOf(c)};
}

// OmniHair's Melanin Presets, White to Black; the eighth, Custom, uses the Melanin slider.
constexpr double kMelaninPresets[] = {0, 0.0025, 0.1, 0.175, 0.3, 0.65, 1};

// Chiang et al.'s fit between a hair colour and its absorption, by azimuthal roughness.
double chiangFit(double b)
{
  return 5.969 - 0.215 * b + 2.532 * std::pow(b, 2) - 10.73 * std::pow(b, 3) + 5.574 * std::pow(b, 4) + 0.245 * std::pow(b, 5);
}

// Iray's volume: Transmitted Color over Transmitted Measurement Distance (absorption), and the
// scattering over Scattering Measurement Distance, coloured by SSS Color in chromatic mode or grey
// by SSS Amount in mono mode (PBRSkin is always chromatic). None when the material has no
// measurement distance, or PBRSkin's Transmission and Sub Surface switches are off.
template <class Find, class Enabled>
std::optional<mat::Subsurface> readSubsurface(const Find& find, bool pbrSkin, const Enabled& enabled)
{
  if(!enabled("Transmission Enable") || !enabled("Sub Surface Enable"))
    return std::nullopt;
  const double transmittedDistance = num(find({"Transmitted Measurement Distance"}), 0);
  const double scatterDistance     = num(find({"Scattering Measurement Distance"}), 0);
  if(transmittedDistance <= 0 && scatterDistance <= 0)
    return std::nullopt;
  const bool     chromatic = pbrSkin || num(find({"SSS Mode"}), 0) == 1;
  const Channel* sssColor  = find({"SSS Color", "Subsurface Color"});
  const double   amount    = std::min(1.0, num(find({"SSS Amount"}), 0));
  mat::Subsurface s;
  s.scatterColor        = chromatic ? colorOr(sssColor) : Rgb{float(amount), float(amount), float(amount)};
  s.transmittedColor    = colorOr(find({"Transmitted Color"}));
  s.transmittedDistance = float(transmittedDistance);
  s.scatterMap          = chromatic ? imageOf(sssColor) : std::string();
  s.scatterDistance     = float(scatterDistance);
  s.direction           = float(std::clamp(num(find({"SSS Direction"}), 0), -0.99, 0.99));
  return s;
}

// PBRSkin's detail layer. None when there is nothing to draw.
template <class Find>
std::optional<mat::DetailLayer> readDetail(const Find& find)
{
  const Channel* weightCh = find({"Detail Weight"});
  const double   weight   = num(weightCh, 0);
  const Channel* mapCh    = find({"Detail Normal Map"});
  if(weight <= 0 || imageOf(mapCh).empty())
    return std::nullopt;
  mat::DetailLayer d;
  d.map = imageOf(mapCh);
  // Detail Normal Map Mode: 0 height map, 1 normal map (Diffeomorphic reads it so). Without the
  // channel, a file named like a normal map is taken for one.
  if(const Channel* modeCh = find({"Detail Normal Map Mode"}))
    d.heightMap = num(modeCh, 0) != 1;
  else
  {
    static const std::regex named("normal|[_-]nm?\\b|nrm", std::regex::ECMAScript | std::regex::icase);
    d.heightMap = !std::regex_search(d.map, named);
  }
  const Channel* roughCh = find({"Detail Specular Roughness Mult"});
  d.strength             = float(num(mapCh, 1));
  d.weight               = float(std::min(weight, 1.0));
  d.weightMap            = imageOf(weightCh);
  const double tx        = num(find({"Detail Horizontal Tiles"}), 1), ty = num(find({"Detail Vertical Tiles"}), 1);
  d.tiles                = {float(tx != 0 ? tx : 1), float(ty != 0 ? ty : 1)};
  d.offset               = {float(num(find({"Detail Horizontal Offset"}), 0)), float(num(find({"Detail Vertical Offset"}), 0))};
  d.roughness            = float(num(roughCh, 1));
  d.roughnessMap         = imageOf(roughCh);
  return d;
}

// PBRSkin's makeup layer: a colour laid over the base colour by Makeup Weight. None when off.
template <class Find>
std::optional<mat::MakeupLayer> readMakeup(const Find& find)
{
  const Channel* weightCh = find({"Makeup Weight"});
  const double   weight   = num(weightCh, 0);
  if(weight <= 0)
    return std::nullopt;
  const Channel* colorCh = find({"Makeup Base Color"});
  mat::MakeupLayer m;
  m.color     = colorOr(colorCh);
  m.map       = imageOf(colorCh);
  m.weight    = float(std::min(weight, 1.0));
  m.weightMap = imageOf(weightCh);
  return m;
}

}  // namespace

std::optional<mat::Rgb> omniHairColor(const Channels& ch)
{
  auto find = [&](std::initializer_list<const char*> n) { return ch.find(n); };
  const Channel* melaninCh = find({"Melanin"});
  if(!melaninCh)
    return std::nullopt;
  // Iray renders OmniHair as Chiang's hair fibre with the absorption OmniHairBase.mdl builds:
  // melanin (Melanin^2 x 2.4, split by Melanin Redness, with Bitterli's eumelanin and pheomelanin
  // spectra), and the dye colour (Hair Root and Tip Color through Chiang's colour-to-absorption
  // fit), the two mixed by sqrt(Melanin). Diffuse Strength ("Hair Color Weight") blends from
  // melanin alone to that mix; a white dye is no dye. A surface has no fibres for light to scatter
  // through, so the colour is read back through the same fit with its exponent scaled down to stand
  // for the light the fibres pass on to one another: at 0.45, melanin 0.5 at redness 0.5 gives
  // (0.26, 0.16, 0.06) linear, the albedo measured for medium brown hair, and melanin 1 a near-black
  // dark brown. Root and tip are averaged: a ribbon has no colour along the strand.
  constexpr int    kPresets = int(sizeof(kMelaninPresets) / sizeof(kMelaninPresets[0]));
  const double     preset   = num(find({"Melanin Presets"}), kPresets);
  const double     m        = preset >= 0 && preset < kPresets ? kMelaninPresets[int(preset)] : clamp01(num(melaninCh, 0.5));
  const double     red      = clamp01(num(find({"Melanin Redness"}), 0.5));
  const double     melanin  = m * m * 2.4;
  const double     eu = melanin * (1 - red), pheo = melanin * red;
  const double     sigmaMelanin[3] = {eu * 0.506 + pheo * 0.343, eu * 0.841 + pheo * 0.733, eu * 1.653 + pheo * 1.924};
  // the MDL's own pick: the azimuthal roughness when it is separate, else the longitudinal one
  const bool       anisotropic = find({"Anisotropic Roughness"}) ? truthy(find({"Anisotropic Roughness"})) : true;
  const double     rough       = clamp01(num(find({"Roughness"}), 0.2));
  const double     longitudinal = std::pow(0.726 * rough + 0.812 * rough * rough + 3.7 * std::pow(rough, 20), 2);
  const double     fit          = chiangFit(anisotropic ? clamp01(num(find({"Azimuthal Roughness"}), 0.2)) : longitudinal);
  const Rgb        root         = colorOr(find({"Hair Root Color"})), tip = colorOr(find({"Hair Tip Color"}));
  const Rgb        dye{(root[0] + tip[0]) / 2, (root[1] + tip[1]) / 2, (root[2] + tip[2]) / 2};
  const double     weight = clamp01(num(find({"Diffuse Strength"}), 0.8));
  const bool       undyed = dye[0] >= 1 && dye[1] >= 1 && dye[2] >= 1;
  Rgb              out;
  for(int i = 0; i < 3; i++)
  {
    double sigma = sigmaMelanin[i];
    if(!undyed)
    {
      const double sigmaDye = std::pow(std::log(std::clamp(double(dye[i]), 0.001, 1.0)) / fit, 2);
      const double mixed    = sigmaDye + (sigmaMelanin[i] - sigmaDye) * std::sqrt(m);
      sigma += (mixed - sigma) * weight;
    }
    out[i] = float(std::exp(-0.45 * std::sqrt(sigma) * fit));
  }
  return out;
}

mat::Surface parseSurface(const std::string& name, const Channels& ch)
{
  using namespace mat;
  auto find = [&](std::initializer_list<const char*> n) { return ch.find(n); };

  // OmniHair (strand hair): surfaces named Fine, Mid, Outer... are hair all the same
  const std::optional<Rgb> omniHair = omniHairColor(ch);
  Surface                  s;
  s.name = name;
  s.kind = omniHair ? Kind::eHair : classify(name);
  // PBRSkin is a different shader from Iray Uber: it has per-layer enable switches, a detail and a
  // makeup layer, and its own dual-lobe parameters. Only it has these channels.
  const bool pbrSkin = find({"Specular Lobe 2 Roughness Mult", "Detail Enable", "Makeup Enable"}) != nullptr;
  // A PBRSkin layer switch; Iray Uber has none, so its layers are always on.
  auto enabled = [&](const char* switchName) { return !pbrSkin || truthy(find({switchName})); };

  const Channel* base     = find({"Base Color", "diffuse", "Diffuse Color"});
  const Channel* cut      = find({"Cutout Opacity", "transparency", "opacity"});
  const Channel* normalCh = find({"Normal Map", "normal"});
  const Channel* bumpCh   = find({"Bump Strength", "bump"});

  // Roughness: Iray's Glossy Roughness, else the inverse of a legacy glossiness, else a guess.
  // Base Mixing 1 is Iray Uber's PBR Specular/Glossiness mode, whose layers read the Glossiness
  // channels and ignore the Roughness ones (hair often sets it, with every roughness left at 0).
  const bool specGloss = num(find({"Base Mixing"}), 0) == 1;
  auto       roughFromGloss = [&](const char* n) -> std::optional<double> {
    const Channel* g = find({n});
    if(specGloss && g)
      return clamp01(1 - num(g, 0.5));
    return std::nullopt;
  };
  const Channel* glossRough = find({"Glossy Roughness", "Specular Lobe 1 Roughness", "roughness"});
  const Channel* glossiness = find({"Glossiness"});
  const auto     glossyFromGloss = roughFromGloss("Glossiness");
  ScalarMap      roughness;
  if(glossyFromGloss)
    roughness = {float(*glossyFromGloss), {}};
  else if(glossRough)
    roughness = scalar(glossRough, 0.5);
  else
    roughness = {float(glossiness ? 1 - num(glossiness, 0.5) : 0.6), {}};

  // Specular layers. Iray's reflectivities are Principled-style: 0.5 is a plain dielectric. In Iray
  // Uber, Dual Lobe Specular Ratio is lobe 1's share of the dual-lobe weight and each lobe has its
  // own roughness. PBRSkin derives both lobes from Specular Lobe 1 Roughness and its map (times
  // Dual Lobe Specular Roughness Mult, lobe 2 times Specular Lobe 2 Roughness Mult on top), and its
  // Ratio is lobe 2's share. A material with neither a glossy weight nor dual-lobe channels gets
  // half a glossy layer, a believable dielectric.
  const Channel* glossW = find({"Glossy Layered Weight", "Glossy Weight"});
  const Channel* dualCh = find({"Dual Lobe Specular Weight"});
  s.glossy.weight       = glossW ? float(num(glossW, 0)) : dualCh ? 0.f : 0.5f;
  s.glossy.reflectivity = float(num(find({"Glossy Reflectivity"}), 0.5));
  s.glossy.roughness    = roughness;
  if(dualCh)
  {
    const double   ratio     = clamp01(num(find({"Dual Lobe Specular Ratio"}), 1));
    const Channel* lobe1Ch   = find({"Specular Lobe 1 Roughness"});
    const auto     lobe1Gloss = roughFromGloss("Specular Lobe 1 Glossiness");
    ScalarMap      lobe1 = lobe1Gloss ? ScalarMap{float(*lobe1Gloss), {}} : ScalarMap{float(num(lobe1Ch, roughness.value)), imageOf(lobe1Ch)};
    ScalarMap      lobe2;
    if(pbrSkin)
    {
      lobe1.value *= float(num(find({"Dual Lobe Specular Roughness Mult"}), 1));
      lobe2 = {lobe1.value * float(num(find({"Specular Lobe 2 Roughness Mult"}), 1)), lobe1.map};
    }
    else
    {
      const Channel* lobe2Ch    = find({"Specular Lobe 2 Roughness"});
      const auto     lobe2Gloss = roughFromGloss("Specular Lobe 2 Glossiness");
      lobe2 = lobe2Gloss ? ScalarMap{float(*lobe2Gloss), {}} : ScalarMap{float(num(lobe2Ch, lobe1.value)), imageOf(lobe2Ch)};
    }
    DualLobe d;
    d.weight       = enabled("Dual Lobe Specular Enable") ? float(num(dualCh, 0)) : 0.f;
    d.reflectivity = float(num(find({"Dual Lobe Specular Reflectivity"}), 0.5));
    d.lobe1        = lobe1;
    d.lobe2        = lobe2;
    d.ratio        = float(pbrSkin ? 1 - ratio : ratio);
    s.dual         = d;
  }

  const Channel* refr = find({"Refraction Weight"});
  const Channel* coat = find({"Top Coat Weight"});

  // Iray emitters
  const Channel*           emitCh = find({"Emission Color", "emission"});
  const std::optional<Rgb> emit   = color(emitCh);
  if(emit && (*emit)[0] + (*emit)[1] + (*emit)[2] > 0.01f)
  {
    Emission e;
    e.color       = {*emit, imageOf(emitCh)};
    e.temperature = float(num(find({"Emission Temperature"}), 6500));
    // emission without an Iray luminance keeps a plain glow
    const Channel* lum = find({"Luminance"});
    if(lum && lum->value.is_number() && lum->value.get<double>() > 0)
    {
      // Luminance Units: cd/m2, kcd/m2, cd/ft2, cd/cm2, lm, W
      constexpr double kUnitScale[] = {1, 1000, 10.7639, 10000, 1, 683};
      const Channel*   unitsCh      = find({"Luminance Units"});
      const int        unit         = unitsCh && unitsCh->value.is_number() ? int(unitsCh->value.get<double>()) : 0;
      Luminance        l;
      l.amount   = float(lum->value.get<double>() * (unit >= 0 && unit < 6 ? kUnitScale[unit] : 1.0));
      l.perArea  = unit >= 4;
      const Channel* two = find({"Two Sided Light"});
      l.twoSided = two && two->value.is_boolean() && two->value.get<bool>();
      e.luminance = l;
    }
    s.emission = e;
  }

  const Channel* anisoCh = find({"Glossy Anisotropy"});
  const Channel* transCh = find({"Translucency Weight"});
  if(transCh)
  {
    const Channel* transCol = find({"Translucency Color"});
    Translucency   t;
    t.weight          = float(std::clamp(enabled("Translucency Enable") ? num(transCh, 0) : 0, 0.0, 1.0));
    t.color           = color(transCol);
    t.map             = imageOf(transCol);
    t.reflectanceTint = colorOr(find({"SSS Reflectance Tint"}));
    t.subsurface      = readSubsurface(find, pbrSkin, enabled);
    s.translucency    = t;
  }

  s.shader = pbrSkin ? Surface::Shader::ePbrSkin
                     : (find({"Glossy Roughness", "Translucency Weight", "Metallic Weight"}) ? Surface::Shader::eIray : Surface::Shader::eLegacy);
  // OmniHair's diffuse is only its viewport colour
  s.base.color = omniHair ? *omniHair : colorOr(base);
  s.base.map   = omniHair ? std::string() : imageOf(base);
  s.opacity    = scalar(cut, 1);
  if(!imageOf(normalCh).empty())
  {
    s.normalMap      = imageOf(normalCh);
    s.normalStrength = float(num(normalCh, 1));
  }
  if(!imageOf(bumpCh).empty())
  {
    s.bumpMap      = imageOf(bumpCh);
    s.bumpStrength = float(num(bumpCh, 1));
  }
  s.metallic = scalar(find({"Metallic Weight", "Metallicity"}), 0);
  // hair and lashes use refraction to let light through the strands, not as glass
  if(refr && s.kind != Kind::eHair && s.kind != Kind::eLashes)
    s.refraction = Surface::Refraction{float(num(refr, 0)), float(num(find({"Refraction Index", "ior"}), 1.4))};
  if(num(coat, 0) > 0)
  {
    const auto gloss = roughFromGloss("Top Coat Glossiness");
    s.topCoat        = Surface::TopCoat{float(num(coat, 0)), float(gloss ? *gloss : num(find({"Top Coat Roughness"}), 0.1))};
  }
  if(pbrSkin && enabled("Detail Enable"))
    s.detail = readDetail(find);
  if(pbrSkin && enabled("Makeup Enable"))
    s.makeup = readMakeup(find);
  // 0 is authored: the highlight stays round. Absent is not the same thing.
  if(anisoCh)
    s.glossyAnisotropy = float(clamp01(num(anisoCh, 0)));
  s.twoSided = omniHair.has_value();
  return s;
}

Material buildMaterial(const std::string& name, const Channels& ch)
{
  Material m;
  m.name    = name;
  m.surface  = parseSurface(name, ch);
  m.channels = std::make_shared<const Channels>(ch);
  return m;
}

bool overrideChannels(Material& m, const std::vector<ChannelOverride>& overrides)
{
  if(overrides.empty() || !m.channels)
    return false;
  Json channels = Json::array();
  for(const ChannelOverride& o : overrides)
  {
    Json c = {{"id", o.channel}};
    if(o.value)
      c["current_value"] = *o.value;
    if(o.image)
      c["image_file"] = *o.image;
    channels.push_back({{"channel", c}});
  }
  const Json def = {{"extra", Json::array({{{"channels", channels}}})}};
  // a copy: the original's channels are shared with every material of the same definition
  Channels next = *m.channels;
  collectChannels(&def, next, {}, {});
  m.surface  = parseSurface(m.name, next);
  m.channels = std::make_shared<const Channels>(std::move(next));
  return true;
}

}  // namespace fv::dson
