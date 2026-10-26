#include <figure/materials/lower.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

#include <figure/materials/hair_strands.hpp>
#include <figure/materials/lobes.hpp>

namespace fv::mat {

using Json = nlohmann::json;

namespace {

constexpr Rgb kWhite{1, 1, 1};

Json rgbJson(const Rgb& c)
{
  return Json::array({c[0], c[1], c[2]});
}

Rgb mul(const Rgb& a, const Rgb& b)
{
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

Rgb lerp(const Rgb& a, const Rgb& b, float t)
{
  return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
}

struct Builder
{
  std::vector<TextureRequest> requests;
  bool                        flipV = false;  // LowerOptions::flipV

  // A texture info for a request. `tiles`/`offset` tile the texture through KHR_texture_transform.
  Json info(TextureRequest req, std::optional<float> scale = std::nullopt, std::optional<std::array<float, 2>> tiles = std::nullopt,
            std::optional<std::array<float, 2>> offset = std::nullopt)
  {
    // a layered image: the bottom layer is the source, the rest go over it
    if(req.rel.find(kLayerSeparator) != std::string::npos)
    {
      req.layers = mapLayers(req.rel);
      req.rel    = req.layers.front();
      req.layers.erase(req.layers.begin());
    }
    requests.push_back(std::move(req));
    Json out{{"slot", int(requests.size()) - 1}, {"texCoord", 0}};
    if(scale)
      out["scale"] = *scale;
    if(tiles && offset && ((*tiles)[0] != 1 || (*tiles)[1] != 1 || (*offset)[0] != 0 || (*offset)[1] != 0))
    {
      // DAZ samples at tiles·v + offset; with v' = 1 − v and the image unflipped that is tiles·v' + (1 − tiles − offset)
      const float offsetV = flipV ? 1 - (*tiles)[1] - (*offset)[1] : (*offset)[1];
      out["extensions"] = {{"KHR_texture_transform", {{"scale", Json::array({(*tiles)[0], (*tiles)[1]})}, {"offset", Json::array({(*offset)[0], offsetV})}}}};
    }
    return out;
  }

  Json plain(const std::string& rel, bool srgb, std::optional<float> scale = std::nullopt, std::optional<std::array<float, 2>> tiles = std::nullopt,
             std::optional<std::array<float, 2>> offset = std::nullopt)
  {
    TextureRequest r;
    r.rel  = rel;
    r.srgb = srgb;
    return info(std::move(r), scale, tiles, offset);
  }

  // The base colour image: the colour map with its iris recoloured, a cutout in its alpha and makeup
  // laid over it, then the whole recoloured; or, with a flat base colour, the colour baked into the image.
  Json base(const std::string& color, const Rgb& factor, const std::string& alpha, const std::optional<MakeupLayer>& makeup,
            const std::optional<IrisRequest>& iris = std::nullopt, const std::optional<RecolorRequest>& recolor = std::nullopt)
  {
    if(alpha.empty() && !makeup && !iris && !recolor)
      return color.empty() ? Json() : plain(color, true);
    TextureRequest r;
    r.srgb     = true;
    r.rel      = !color.empty() ? color : makeup && !makeup->map.empty() ? makeup->map : makeup && !makeup->weightMap.empty() ? makeup->weightMap : alpha;
    r.alphaRel = alpha;
    r.iris     = iris;
    r.recolor  = recolor;
    // a flat base colour (no base map) is a makeup bake with nothing laid over it
    if(makeup || color.empty())
    {
      MakeupRequest m;
      m.baseIsSource = !color.empty();
      m.factor       = factor;
      m.colorRel     = makeup ? makeup->map : std::string();
      m.tint         = makeup ? makeup->color : kWhite;
      m.weightRel    = makeup ? makeup->weightMap : std::string();
      m.weightFactor = makeup ? makeup->weight : 0.f;
      r.makeup       = m;
    }
    return info(std::move(r));
  }
};

float clamp(float v, float lo, float hi)
{
  return std::clamp(v, lo, hi);
}

// Chiang et al.'s fit between a fibre's colour and its absorption, by azimuthal roughness, as the
// hair shader (hair_bsdf.h.slang) and the DSON reader use it: colour = exp(-0.45 sqrt(sigma) fit).
float chiangFit(float b)
{
  b = clamp(b, 0, 1);
  return 5.969f - 0.215f * b + 2.532f * b * b - 10.73f * b * b * b + 5.574f * b * b * b * b + 0.245f * b * b * b * b * b;
}

// A fibre's root and tip absorption moved so their mean colour goes to `target` by `strength`, each
// scaled by the same ratio, so a dark root and a light tip stay apart. The path tracer shades a
// fibre from these, not from its map, so a recolour has to move them too.
void recolorFibre(Rgb& root, Rgb& tip, float azimuthal, const Rgb& target, float strength)
{
  const float fit    = chiangFit(azimuthal);
  const auto  colour = [&](float sigma) { return std::exp(-0.45f * std::sqrt(std::max(sigma, 0.f)) * fit); };
  const auto  sigma  = [&](float c) {
    const float s = std::log(clamp(c, 1e-3f, 0.999f)) / (-0.45f * fit);
    return s * s;
  };
  for(int i = 0; i < 3; i++)
  {
    const float r = colour(root[i]), t = colour(tip[i]);
    const float k = target[i] / std::max(0.5f * (r + t), 1e-3f);
    root[i]       = sigma(r + (r * k - r) * strength);
    tip[i]        = sigma(t + (t * k - t) * strength);
  }
}

// Genesis 9's eye UVs put both irises in the top half of one map. These are the nominal discs
// (centre x, y and radius, fractions of the image, y down) the cooker fits each limbus around:
// measured on Addy Florence's, Rachael's, CJ's and Ava's maps, radii 0.096 to 0.124 and pupils up to
// 0.011 off centre.
constexpr std::array<float, 3> kG9IrisLeft{0.249f, 0.249f, 0.108f};
constexpr std::array<float, 3> kG9IrisRight{0.749f, 0.249f, 0.108f};

// The recolour an eye's base map needs for its iris and sclera colours, none when it needs none.
// A Genesis 8 sclera is a surface of its own: its colour multiplies the factor, no bake.
std::optional<IrisRequest> irisRequest(const Surface& s, const Edits& e, const LowerOptions& o)
{
  const EyeLayout layout = eyeLayout(s);
  if(s.base.map.empty() || (layout != EyeLayout::eShared && layout != EyeLayout::eIris))
    return std::nullopt;
  IrisRequest r;
  r.iris = e.irisColor;
  // a strength alone changes nothing to cook
  r.strength = e.irisColor ? clamp(e.irisStrength.value_or(1.f), 0, 1) : 1.f;
  if(layout == EyeLayout::eShared)
  {
    if(!e.irisColor && !e.scleraColor)
      return std::nullopt;
    r.mode   = IrisRequest::Mode::eEye;
    r.discs  = {kG9IrisLeft, kG9IrisRight};
    r.sclera = e.scleraColor;
    return r;
  }
  // without the discs the cooker has nowhere to take the iris's mean (material.set refuses it)
  if(!e.irisColor || o.irisDiscs.empty())
    return std::nullopt;
  r.mode  = IrisRequest::Mode::eIris;
  r.discs = o.irisDiscs;
  return r;
}

}  // namespace

Lowered lower(const Surface& s, const Edits& e, const LowerOptions& opts)
{
  Builder        b;
  b.flipV = opts.flipV;
  Json           ext   = Json::object();
  Json           out   = {{"name", s.name}};
  const auto     lobes = specularLobes(s);
  const SkinRole role  = skinRole(s);
  const bool     refractive = s.refraction && s.refraction->weight > 0.05f;
  const Rgb      baseColor  = e.color.value_or(s.base.color);
  // Iray's refraction shows none of the base colour where it refracts, but glTF tints transmitted
  // light by it, so a map on a refracting surface would be looked through, on the way in and out
  // (Tri-X's eye presets put the whole eye's map on the eye moisture). Its colour goes as a flat one does.
  const std::string baseMap = refractive ? std::string() : s.base.map;

  // base colour: cutouts ride in its alpha, PBRSkin makeup is baked into it (glTF has one base
  // colour), blended in linear light
  std::optional<MakeupLayer> makeup = s.makeup ? s.makeup : hairTranslucency(s);
  // the panel's makeup: a colour over a weight map replaces the colour map (a lipstick's shape is the
  // weight map's); without one the colour map is the shape, and the colour tints it
  if(s.makeup && (e.makeupColor || e.makeupWeight))
  {
    if(e.makeupColor)
    {
      if(!makeup->weightMap.empty())
        makeup->map.clear();
      makeup->color = *e.makeupColor;
    }
    if(e.makeupWeight)
      makeup->weight = clamp(*e.makeupWeight, 0, 1);
  }
  const bool anyMap = !baseMap.empty() || !s.opacity.map.empty() || (makeup && (!makeup->map.empty() || !makeup->weightMap.empty()));
  Rgb        factor = baseColor;
  if(makeup)
    factor = anyMap ? kWhite : lerp(baseColor, makeup->color, makeup->weight);
  // an eye's colours that need no bake: an iris with no map takes the iris colour, a sclera surface
  // of its own is multiplied by the sclera colour
  const EyeLayout eye = eyeLayout(s);
  if(eye == EyeLayout::eIris && s.base.map.empty() && e.irisColor)
    factor = lerp(factor, *e.irisColor, clamp(e.irisStrength.value_or(1.f), 0, 1));
  if(eye == EyeLayout::eSclera && e.scleraColor)
    factor = mul(factor, *e.scleraColor);
  // a recolour: baked into the image when there is one, else the flat colour goes towards it. A
  // skin tone is one against the figure's tone reference, or the surface's own map without one.
  std::optional<RecolorRequest> recolor;
  if(e.tone)
    recolor = RecolorRequest{*e.tone, clamp(e.toneStrength.value_or(1.f), 0, 1), !opts.toneReference.empty() ? opts.toneReference : s.base.map};
  else if(e.recolor)
    recolor = RecolorRequest{*e.recolor, clamp(e.recolorStrength.value_or(1.f), 0, 1)};
  if(recolor && !anyMap)
    factor = lerp(factor, recolor->color, recolor->strength);
  // glTF tints transmitted light by the base colour; Iray's refraction doesn't (the eye moisture's
  // grey diffuse colour, passed through twice, left a quarter of the light to reach the iris)
  if(refractive && !anyMap)
    factor = lerp(factor, kWhite, s.refraction->weight);
  // Cutouts go out as BLEND: the renderer's any-hit shader skips a hit with probability 1 - alpha,
  // which is alpha hashing by another name, so they keep their soft edges. Refractive surfaces
  // (corneas) go out opaque with transmission doing the work.
  const bool hashed = !s.opacity.map.empty() || (s.opacity.value < 1 && !refractive);
  std::string alphaMode = hashed ? (opts.forFile ? "MASK" : "BLEND") : "OPAQUE";
  float       opacity   = e.opacity.value_or(s.opacity.value);
  if(refractive && s.opacity.map.empty() && !e.opacity)
  {
    alphaMode = "OPAQUE";
    opacity   = 1;
  }
  // an inspector edit of the alpha mode: hashed and blended alike are BLEND for this renderer
  if(e.alphaMode)
    alphaMode = *e.alphaMode == "opaque" ? "OPAQUE" : *e.alphaMode == "cutout"           ? "MASK" :
                (opts.forFile && *e.alphaMode == "hash") ? "MASK" :
                                                            "BLEND";
  // A live ribbon keeps the alpha mode it was given. The root narrows in the mesh, so the body does
  // not have to be blended just to soften the hairline; a cutout that is already blended still fades.
  const bool liveRibbon = s.hairFibre && s.twoSided && !opts.forFile;

  Effective eff;
  eff.color       = baseColor;
  eff.opacity     = opacity;
  eff.alphaMode   = alphaMode == "OPAQUE" ? "opaque" : alphaMode == "MASK" ? "cutout" : "blend";
  eff.doubleSided = e.doubleSided.value_or(hashed || s.twoSided);
  eff.metalness   = e.metalness.value_or(s.metallic.value);

  Json pbr = Json::object();
  // a flat colour baked into the image (no base map) must not tint it again; with no image at all
  // (makeup with no maps of its own) the factor is the colour
  const bool bakedFlat = anyMap && baseMap.empty() && (makeup || !s.opacity.map.empty());
  pbr["baseColorFactor"] = Json::array({(anyMap && makeup) || bakedFlat ? 1.f : factor[0], (anyMap && makeup) || bakedFlat ? 1.f : factor[1],
                                        (anyMap && makeup) || bakedFlat ? 1.f : factor[2], opacity});
  pbr["metallicFactor"]  = e.metalness.value_or(s.metallic.value);
  if(anyMap)
  {
    const Json info = b.base(baseMap, baseColor, s.opacity.map, makeup, irisRequest(s, e, opts), recolor);
    if(!info.is_null())
      pbr["baseColorTexture"] = info;
  }

  // roughness and metalness: one lobe from Iray's layers, with the limits the real-time view has
  float roughness = lobes.roughness.value;
  if(refractive)
    roughness = std::min(roughness, 0.1f);
  else if(s.kind == Kind::eEye)
    roughness = std::min(roughness, 0.35f);
  // glTF's range: PBRSkin's lobe multipliers and panel edits can pass it
  const float roughnessFactor = clamp(e.roughness.value_or(roughness), 0, 1);
  eff.roughness               = roughnessFactor;
  pbr["roughnessFactor"]      = roughnessFactor;
  // glTF metallic-roughness from a roughness map (roughness in green, full metalness in blue); a
  // separate metalness map would need both merged, which the cooker does not do
  if(!lobes.roughness.map.empty())
  {
    TextureRequest r;
    r.rel      = lobes.roughness.map;
    r.srgb     = false;
    r.channels = "mr";
    pbr["metallicRoughnessTexture"] = b.info(std::move(r));
  }
  out["pbrMetallicRoughness"] = pbr;
  out["alphaMode"]            = alphaMode;
  if(alphaMode == "MASK")
    out["alphaCutoff"] = 0.5;
  out["doubleSided"] = e.doubleSided.value_or(hashed || s.twoSided);

  // normal: the normal map and the bump map together when a surface has both (glTF has one slot),
  // else whichever one it has. DAZ skin keeps pores in the bump map. The bump strength is baked
  // into the blend; the normal map's strength stays the glTF scale.
  Json normal;
  if(!s.normalMap.empty() && !s.bumpMap.empty())
  {
    TextureRequest r;
    r.rel          = s.normalMap;
    r.srgb         = false;
    r.channels     = "ng";
    r.bumpRel      = s.bumpMap;
    r.bumpStrength = clamp(e.bumpStrength.value_or(s.bumpStrength), 0, 2);
    normal         = b.info(std::move(r), e.normalStrength.value_or(s.normalStrength));
    eff.normalStrength = e.normalStrength.value_or(s.normalStrength);
    eff.bumpStrength   = clamp(e.bumpStrength.value_or(s.bumpStrength), 0, 2);
  }
  else if(!s.normalMap.empty())
  {
    TextureRequest r;
    r.rel      = s.normalMap;
    r.srgb     = false;
    r.channels = "ng";
    normal     = b.info(std::move(r), e.normalStrength.value_or(s.normalStrength));
    eff.normalStrength = e.normalStrength.value_or(s.normalStrength);
  }
  else if(!s.bumpMap.empty())
  {
    TextureRequest r;
    r.rel      = s.bumpMap;
    r.srgb     = false;
    r.channels = "bn";
    normal     = b.info(std::move(r), clamp(e.bumpStrength.value_or(s.bumpStrength), 0, 2));
    eff.bumpStrength = clamp(e.bumpStrength.value_or(s.bumpStrength), 0, 2);
  }
  if(!normal.is_null())
    out["normalTexture"] = normal;

  // emission: an Iray emitter at its physical brightness
  if(s.emission)
  {
    // the temperature tint is normalised to luminance, so a warm light's red passes 1: glTF's
    // emissiveFactor stops there, and the excess goes into the strength
    Rgb         color = mul(s.emission->color.color, temperatureTint(s.emission->temperature));
    const float peak  = std::max({color[0], color[1], color[2]});
    const float scale = peak > 1 ? peak : 1.f;
    color             = {color[0] / scale, color[1] / scale, color[2] / scale};
    out["emissiveFactor"] = rgbJson(color);
    if(!s.emission->color.map.empty())
      out["emissiveTexture"] = b.plain(s.emission->color.map, true);
    ext["KHR_materials_emissive_strength"] = {{"emissiveStrength", opts.emission * scale}};
    eff.emissive          = color;
    eff.emissiveIntensity = opts.emission * scale;
  }
  // an emissive colour given in the panel to a surface that had none
  if(e.emissive && !s.emission && ((*e.emissive)[0] > 0 || (*e.emissive)[1] > 0 || (*e.emissive)[2] > 0))
  {
    out["emissiveFactor"]                  = rgbJson(*e.emissive);
    ext["KHR_materials_emissive_strength"] = {{"emissiveStrength", e.emissiveIntensity.value_or(1.f)}};
    eff.emissive                           = *e.emissive;
    eff.emissiveIntensity                  = e.emissiveIntensity.value_or(1.f);
  }
  else if(s.emission)
  {
    // edits of an emitter's colour and brightness
    if(e.emissive)
    {
      out["emissiveFactor"] = rgbJson(*e.emissive);
      eff.emissive          = *e.emissive;
    }
    if(e.emissiveIntensity)
    {
      ext["KHR_materials_emissive_strength"] = {{"emissiveStrength", *e.emissiveIntensity}};
      eff.emissiveIntensity                  = *e.emissiveIntensity;
    }
  }

  // A top coat occupies the only clearcoat slot, so a tight lobe that would have been the coat
  // stays in the base specular.
  // A file of strand hair (no fibre BSDF in other tools) is a dielectric sheet of index 1.55.
  // The live path shades the fibre itself and does not also put that sheet on it.
  const bool  ribbonHair = s.kind == Kind::eHair && s.twoSided;
  const float specular   = ribbonHair && !liveRibbon ? 1.f
                           : s.topCoat && lobes.clearcoat ? std::min(1.f, lobes.intensity + lobes.clearcoat->weight)
                                                          : lobes.intensity;
  eff.specular                  = e.specular.value_or(specular);
  ext["KHR_materials_specular"] = {{"specularFactor", eff.specular}, {"specularColorFactor", Json::array({1, 1, 1})}};
  if(ribbonHair && !liveRibbon)
    ext["KHR_materials_ior"] = {{"ior", 1.55}};

  // clearcoat: Iray's top coat, else the tighter dual lobe, on the same surface detail as the base
  struct Coat
  {
    float     weight;
    ScalarMap roughness;
  };
  std::optional<Coat> coat;
  if(s.topCoat)
    coat = Coat{s.topCoat->weight, {s.topCoat->roughness, {}}};
  else if(lobes.clearcoat)
    coat = Coat{lobes.clearcoat->weight, lobes.clearcoat->roughness};
  if(coat || e.clearcoat)
  {
    Json cc = {{"clearcoatFactor", e.clearcoat.value_or(coat ? coat->weight : 0.f)},
               {"clearcoatRoughnessFactor", clamp(e.clearcoatRoughness.value_or(coat ? coat->roughness.value : 0.1f), 0, 1)}};
    if(coat && !coat->roughness.map.empty())
      cc["clearcoatRoughnessTexture"] = b.plain(coat->roughness.map, false);
    if(!normal.is_null())
    {
      // the same slot again: the clearcoat shares the base's surface detail
      cc["clearcoatNormalTexture"] = normal;
    }
    ext["KHR_materials_clearcoat"] = cc;
    eff.clearcoat                  = cc["clearcoatFactor"].get<float>();
    eff.clearcoatRoughness         = cc["clearcoatRoughnessFactor"].get<float>();
  }

  // refraction
  float transmission = 0;
  if(refractive)
  {
    transmission               = s.refraction->weight;
    ext["KHR_materials_ior"]   = {{"ior", s.refraction->ior}};
  }

  // Refraction wins over skin: a cornea with a translucency weight stays glass.
  if(!refractive && role != SkinRole::eNone && opts.skin != SkinMode::eRaster)
  {
    // Translucency, as Iray Uber lays skin out: only (1 - Translucency Weight) of the diffuse light
    // reflects off the Base Color; the rest enters through the Translucency Color and scatters
    // back out. Genesis base colour maps are painted pale for that, and rendered as plain diffuse
    // they look grey.
    const auto& t = s.translucency;
    // Flesh without a translucency channel still scatters. Nails and teeth scatter only at their own weight.
    const float weight = t ? t->weight : role == SkinRole::eFlesh ? 0.5f : 0.f;
    const std::optional<Rgb> tint = t ? t->color : std::nullopt;
    // what comes back out: the translucency colour and map under the SSS reflectance tint; or,
    // without Iray data, the base map lightly tinted towards a flesh tone
    Rgb hue = fallbackTranslucencyColor();
    const float hueMax = std::max({hue[0], hue[1], hue[2], 1e-4f});
    for(float& v : hue)
      v /= hueMax;
    const Rgb scatterColor = t ? mul(tint.value_or(kWhite), t->reflectanceTint) : lerp(kWhite, hue, 0.25f);
    // the base map: a Translucency Color map is often a pale scattering map that washes the skin
    // out lit from the front. A skin tone recolours it as it does the base colour.
    const std::string scatterMap = !s.base.map.empty() ? s.base.map : t ? t->map : std::string();
    const auto        scatterTexture = [&](const Rgb& color) -> Json {
      if(scatterMap.empty())
        return Json();
      return makeup || recolor ? b.base(scatterMap, color, {}, makeup, std::nullopt, recolor) : b.plain(scatterMap, true);
    };
    // The same albedo for both modes. A map or makeup already carries the colour, so the factor
    // stays white and the shader does not apply it twice.
    const Rgb bakedFactor = makeup ? (!scatterMap.empty() || !makeup->map.empty() || !makeup->weightMap.empty() ? kWhite : lerp(scatterColor, makeup->color, makeup->weight))
                                   : scatterColor;
    const auto albedoTexture = [&]() -> Json {
      Json tex = scatterTexture(scatterColor);
      if(!tex.is_null())
        return tex;
      if(makeup && (!makeup->map.empty() || !makeup->weightMap.empty()))
        return b.base({}, scatterColor, {}, makeup);
      return Json();
    };
    if(role == SkinRole::eFlesh && opts.skin == SkinMode::eMeasured)
    {
      // Skin as measured, whatever the file's Iray medium says. Every bit of diffuse light enters
      // (skin has no diffuse layer above its tissue, only the oily film the specular lobes are),
      // and the renderer makes the medium from the skin map where the walk enters and how far
      // light spreads in skin per channel: the map is the colour lit from the front, the distance
      // how soft pores and shadow edges are and how far red glows through ears and fingers.
      transmission                   = 1;
      ext["KHR_materials_ior"]       = {{"ior", 1.4}};
      ext["KHR_materials_volume"]    = {{"thicknessFactor", 1}, {"attenuationDistance", 1}, {"attenuationColor", Json::array({1, 1, 1})}};
      ext["KHR_materials_scatter"]   = {{"scatterStrengthFactor", 1}, {"multiscatterColorFactor", Json::array({1, 1, 1})}, {"scatterAnisotropy", 0}};
      const float scale              = e.scatterScale.value_or(1.f);
      ext["FV_materials_subsurface"] = {{"scatterDistance", rgbJson({kSkinScatterDistance[0] * scale, kSkinScatterDistance[1] * scale, kSkinScatterDistance[2] * scale})}};
    }
    else if(role == SkinRole::eFlesh && opts.skin == SkinMode::eVolume)
    {
      // Volumetric subsurface scattering, Iray's way: the translucent share enters (transmission =
      // weight) and random-walks through a medium with Iray's own scattering and absorption per
      // channel, which is where the warmth comes from; the light leaves through the skin map as a
      // diffuse exit, which also serves the share that reflects as plain diffuse.
      const VolumeMedium v = volumeMedium(t ? t->subsurface : std::nullopt);
      transmission         = std::min(0.95f, weight);
      ext["KHR_materials_ior"]    = {{"ior", 1.4}};
      ext["KHR_materials_volume"] = {{"thicknessFactor", 1}, {"attenuationDistance", v.distance}, {"attenuationColor", rgbJson(v.attenuation)}};
      ext["KHR_materials_scatter"] = {{"scatterStrengthFactor", 1}, {"multiscatterColorFactor", rgbJson(v.multiscatter)}, {"scatterAnisotropy", v.anisotropy}};
      // The base colour stays the skin map's own. The medium already warms what scatters, and the
      // walk leaves through the base colour, so a translucency tint here would be applied twice.
    }
    else if(weight > 0)
    {
      // Thin-walled scattering: the weight goes out as transmission with no volume, and
      // KHR_materials_scatter at full strength turns that share into a Lambertian lobe coloured by
      // the base map, which the renderer evaluates at the surface: no random walk, so open eye
      // sockets and mouths stay clean. Anisotropy -0.8 sends a tenth of it on through, so backlit
      // ears and fingers glow. Makeup lies over the whole skin, so the light that scatters back
      // leaves through it too: it is baked into the scatter colour as into the base colour.
      transmission                = std::min(0.95f, weight);
      ext["KHR_materials_volume"] = {{"thicknessFactor", 0}, {"attenuationColor", Json::array({1, 1, 1})}};
      Json sc = {{"scatterStrengthFactor", 1}, {"multiscatterColorFactor", rgbJson(bakedFactor)}, {"scatterAnisotropy", -0.8}};
      if(Json tex = albedoTexture(); !tex.is_null())
        sc["multiscatterColorTexture"] = tex;
      ext["KHR_materials_scatter"] = sc;
    }
  }
  // The white of an eye under measured skin: dense scattering tissue, so light comes back out a few
  // tenths of a millimetre from where it went in and the lashes' shadows soften instead of printing
  // as stripes. The eyeball's surfaces are no closed medium for a walk (one darkened the eyes), so
  // the renderer spreads where its light samples look for shadow over the surface instead
  // (FV_materials_subsurface with no volume). A Genesis 9 eye's iris is in the same surface.
  if(const EyeLayout eye = eyeLayout(s);
     !refractive && opts.skin == SkinMode::eMeasured && (eye == EyeLayout::eShared || eye == EyeLayout::eSclera))
    ext["FV_materials_subsurface"] = {{"scatterDistance", rgbJson(kScleraScatterDistance)}};
  if(transmission > 0)
  {
    ext["KHR_materials_transmission"] = {{"transmissionFactor", transmission}};
    if(!ext.contains("KHR_materials_volume"))
      ext["KHR_materials_volume"] = {{"thicknessFactor", 0}, {"attenuationColor", Json::array({1, 1, 1})}};
  }

  // PBRSkin's detail layer: a tiled normal map in glTF's convention bending the normal on top of the
  // normal map, with a roughness multiplier, under a weight. No map: the roughness multiplier alone
  if(s.detail && !opts.forFile)
  {
    const DetailLayer& d  = *s.detail;
    Json               de = {{"detailWeightFactor", d.weight}, {"detailRoughnessFactor", d.roughness}};
    if(!d.map.empty())
    {
      TextureRequest r;
      r.rel                    = d.map;
      r.srgb                   = false;
      r.channels               = d.heightMap ? "bn" : "ng";
      de["detailNormalTexture"] = b.info(std::move(r), d.strength, d.tiles, d.offset);
    }
    if(!d.weightMap.empty())
      de["detailWeightTexture"] = b.plain(d.weightMap, false);
    if(!d.roughnessMap.empty())
      de["detailRoughnessTexture"] = b.plain(d.roughnessMap, false, std::nullopt, d.tiles, d.offset);
    ext["FV_materials_detail"] = de;
  }

  // Hair cards: the highlight stretches across the strands, a band. The rotation is measured from
  // the mesh in the renderer's convention. An authored Glossy Anisotropy of 0 keeps it round.
  // A live ribbon uses the per-hit strand instead of one angle for the whole material.
  if(s.kind == Kind::eHair && !liveRibbon && opts.anisotropyRotation)
    if(const auto strength = strandStrength(s, roughnessFactor, lobes.glossyShare))
      ext["KHR_materials_anisotropy"] = {{"anisotropyStrength", *strength}, {"anisotropyRotation", opts.forFile ? -*opts.anisotropyRotation : *opts.anisotropyRotation}};

  if(liveRibbon)
  {
    // Root and tip stay apart. The shader blends them by the ribbon's along-UV. rootV is 1 when
    // the build flipped V, so the fade and the colour still start at the scalp. A roughness or
    // specular edit from the panel moves the fibre, not only the unused sheet lobe.
    const Surface::HairFibre& f = *s.hairFibre;
    float betaM = f.longitudinal;
    float betaN = f.azimuthal;
    if(e.roughness)
    {
      const float ratio = f.longitudinal > 1e-4f ? f.azimuthal / f.longitudinal : 1.f;
      betaM = clamp(*e.roughness, 0.f, 1.f);
      betaN = clamp(betaM * ratio, 0.f, 1.f);
    }
    const float hairSpecular = clamp(e.specular.value_or(1.f), 0.f, 1.f);
    Rgb         rootSigma    = f.rootSigma;
    Rgb         tipSigma     = f.tipSigma;
    if(recolor)
      recolorFibre(rootSigma, tipSigma, betaN, recolor->color, recolor->strength);
    eff.roughness = betaM;
    eff.specular  = hairSpecular;
    out["pbrMetallicRoughness"]["roughnessFactor"] = betaM;
    ext["KHR_materials_specular"]["specularFactor"] = hairSpecular;
    ext["FV_materials_hair"] = {{"longitudinalRoughness", betaM},
                                {"azimuthalRoughness", betaN},
                                {"rootAbsorption", rgbJson(rootSigma)},
                                {"tipAbsorption", rgbJson(tipSigma)},
                                {"rootFade", 0.06f},
                                {"rootV", opts.flipV ? 1.f : 0.f},
                                {"specularFactor", hairSpecular}};
  }
  else if(hairCards(s) && !opts.forFile && opts.skin != SkinMode::eRaster)
  {
    // DAZ cards as fibres, in the path tracer: each hit's absorption from the colour the card shows
    // there (Chiang's fit inverted, as Unreal shades cards), so the painted strands keep their
    // shade, with a white R highlight, a tinted TRT band and light through the lock. The strand
    // runs across the measured anisotropy rotation, along UV v without one. The real-time view
    // keeps the GGX sheet above (it lowers with SkinMode::eRaster).
    const float betaM = clamp(roughnessFactor, 0.f, 1.f);
    Json        hair  = {{"longitudinalRoughness", betaM},
                         {"azimuthalRoughness", kCardAzimuthalRoughness},
                         {"specularFactor", clamp(e.specular.value_or(1.f), 0.f, 1.f)},
                         {"absorptionFromColor", true}};
    if(opts.anisotropyRotation)
      hair["strandRotation"] = *opts.anisotropyRotation;
    ext["FV_materials_hair"] = hair;
  }
  else if(ribbonHair)
  {
    // A file for other tools has no fibre BSDF. A flat transmission keeps a stack of ribbons from
    // shading itself black; the live path gets that from the transmitted lobe instead.
    const Json& c = out["pbrMetallicRoughness"]["baseColorFactor"];
    ext["KHR_materials_diffuse_transmission"] = {{"diffuseTransmissionFactor", 0.4},
                                                 {"diffuseTransmissionColorFactor", Json::array({c[0], c[1], c[2]})}};
  }

  out["extensions"] = ext;
  return {std::move(out), std::move(b.requests), std::move(eff)};
}

bool needsArea(const Surface& s)
{
  return s.emission && s.emission->luminance && s.emission->luminance->perArea;
}

float emissionStrength(const Surface& s, double areaM2, double exposureValue, double iso)
{
  if(!s.emission || !s.emission->luminance)
    return 1;
  const Luminance& l = *s.emission->luminance;
  double           luminance = l.amount;
  if(l.perArea)
    // a flux spreads over the surface: L = flux / (pi A), twice the area for two-sided emitters
    luminance = areaM2 > 0 ? l.amount / (std::numbers::pi * areaM2 * (l.twoSided ? 2 : 1)) : 0;
  // a reflected-light meter puts mid-grey at L = 12.5 * 2^EV / ISO, mapped to the 0.18 a mid-grey surface has
  return float(luminance * (0.18 * iso) / (12.5 * std::pow(2.0, exposureValue)));
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

namespace {

// Replaces every slot in `v` with a texture info from `resolve`, dropping the ones that cannot be made.
void resolveSlots(Json& v, const Lowered& lowered, const std::function<ResolvedTexture(const TextureRequest&)>& resolve, std::vector<std::string>& used)
{
  if(v.is_array())
  {
    for(Json& x : v)
      resolveSlots(x, lowered, resolve, used);
    return;
  }
  if(!v.is_object())
    return;
  if(v.contains("slot"))
  {
    const ResolvedTexture t = resolve(lowered.requests[v["slot"].get<size_t>()]);
    if(t.index < 0)
    {
      v = Json();
      return;
    }
    Json out = {{"index", t.index}, {"texCoord", v.value("texCoord", 0)}};
    if(v.contains("scale"))
      out["scale"] = v["scale"];
    Json extensions = v.contains("extensions") ? v["extensions"] : Json::object();
    if(t.flipGreen)
      extensions["FV_texture_flip_green"] = Json::object();
    if(!extensions.empty())
    {
      for(auto it = extensions.begin(); it != extensions.end(); ++it)
        if(std::find(used.begin(), used.end(), it.key()) == used.end())
          used.push_back(it.key());
      out["extensions"] = extensions;
    }
    v = out;
    return;
  }
  for(auto it = v.begin(); it != v.end(); ++it)
    resolveSlots(it.value(), lowered, resolve, used);
  for(auto it = v.begin(); it != v.end();)
    it = it.value().is_null() ? v.erase(it) : std::next(it);
}

void textureInfo(const Json& j, tinygltf::TextureInfo& out)
{
  out.index = j.value("index", -1);
  out.texCoord = j.value("texCoord", 0);
  if(j.contains("extensions"))
    for(auto it = j["extensions"].begin(); it != j["extensions"].end(); ++it)
      out.extensions[it.key()] = toValue(it.value());
}

}  // namespace

tinygltf::Material toGltf(const Lowered& lowered, const std::function<ResolvedTexture(const TextureRequest&)>& resolve,
                          std::vector<std::string>& extensionsUsed)
{
  Json j = lowered.material;
  resolveSlots(j, lowered, resolve, extensionsUsed);
  tinygltf::Material m;
  m.name = j.value("name", std::string());
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
    const Json& n = j["normalTexture"];
    m.normalTexture.index    = n.value("index", -1);
    m.normalTexture.texCoord = n.value("texCoord", 0);
    m.normalTexture.scale    = n.value("scale", 1.0);
    if(n.contains("extensions"))
      for(auto it = n["extensions"].begin(); it != n["extensions"].end(); ++it)
        m.normalTexture.extensions[it.key()] = toValue(it.value());
  }
  if(j.contains("extensions"))
    for(auto it = j["extensions"].begin(); it != j["extensions"].end(); ++it)
    {
      m.extensions[it.key()] = toValue(it.value());
      if(std::find(extensionsUsed.begin(), extensionsUsed.end(), it.key()) == extensionsUsed.end())
        extensionsUsed.push_back(it.key());
    }
  return m;
}

}  // namespace fv::mat
