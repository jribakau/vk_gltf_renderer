#include <figure/materials/lower.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numbers>

#include <figure/materials/fabric.hpp>
#include <figure/materials/hair_strands.hpp>
#include <figure/materials/lobes.hpp>

namespace fv::mat {

namespace {

constexpr Rgb kWhite{1, 1, 1};

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
  std::vector<TextureRequest>             requests;
  bool                                    flipV = false;  // LowerOptions::flipV
  UvTransform                             uv;             // where the surface's own maps sample (surfaceUv)
  const std::map<std::string, MapTiling>* mapTiling = nullptr;  // Surface::mapTiling

  // The tiling of its own a request's images share (a cooked image is sampled once, so its sources
  // must agree, or it keeps the surface's alone), and that a rotated surface transform can take: a
  // map tiled differently across and down would no longer turn with the rest.
  MapTiling tilingOf(const TextureRequest& req) const
  {
    std::optional<MapTiling> shared;
    std::vector<const std::string*> rels = {&req.rel, &req.alphaRel, &req.metalRel, &req.bumpRel, req.makeup ? &req.makeup->colorRel : nullptr,
                                            req.makeup ? &req.makeup->weightRel : nullptr};
    for(const std::string& m : req.mulRels)
      rels.push_back(&m);
    for(const std::string* rel : rels)
    {
      if(!rel || rel->empty())
        continue;
      MapTiling t;
      if(mapTiling)
        if(const auto it = mapTiling->find(*rel); it != mapTiling->end())
          t = it->second;
      if(shared && *shared != t)
        return {};
      shared = t;
    }
    if(!shared || (uv.rotation != 0 && shared->tiles[0] != shared->tiles[1]))
      return {};
    return *shared;
  }

  // A slot for a request, sampled through `own` (a detail layer's tiling) or else the surface's
  // transform with the map's own tiling over it.
  TextureSlot info(TextureRequest req, std::optional<float> scale = std::nullopt, std::optional<UvTransform> own = std::nullopt)
  {
    // the map samples at Ti·(T + R·S·uv) + Oi: a panel offset moves it with every other map
    UvTransform surface = uv;
    if(const MapTiling t = own ? MapTiling{} : tilingOf(req); t != MapTiling{})
    {
      surface.scale  = {t.tiles[0] * uv.scale[0], t.tiles[1] * uv.scale[1]};
      surface.offset = {t.tiles[0] * uv.offset[0] + t.offset[0], t.tiles[1] * uv.offset[1] + t.offset[1]};
    }
    // a layered image: the bottom layer is the source, the rest go over it
    if(req.rel.find(kLayerSeparator) != std::string::npos)
    {
      req.layers = mapLayers(req.rel);
      req.rel    = req.layers.front();
      req.layers.erase(req.layers.begin());
    }
    requests.push_back(std::move(req));
    TextureSlot out;
    out.index = int(requests.size()) - 1;
    out.scale = scale;
    // DAZ samples at offset + R·S·v; with v' = 1 − v and the image unflipped that is its mirror
    if(const UvTransform& daz = own ? *own : surface; !daz.identity())
      out.transform = flipV ? mirroredV(daz) : daz;
    return out;
  }

  TextureSlot plain(const std::string& rel, bool srgb, std::optional<float> scale = std::nullopt, std::optional<UvTransform> own = std::nullopt)
  {
    TextureRequest r;
    r.rel  = rel;
    r.srgb = srgb;
    return info(std::move(r), scale, own);
  }

  // The product of weight maps as one grey in every channel ("wa"), so that the specular's map (alpha)
  // and the coat's (red) can share the image. Only the first map may be a layered image: the
  // cooker multiplies plain images in, so a later one gives its bottom layer.
  TextureSlot weights(const std::vector<std::string>& maps)
  {
    TextureRequest r;
    r.rel      = maps.front();
    r.srgb     = false;
    r.channels = "wa";
    for(size_t i = 1; i < maps.size(); i++)
      r.mulRels.push_back(mapLayers(maps[i]).front());
    return info(std::move(r));
  }

  // The base colour image: the colour map with its iris recoloured, a cutout in its alpha (the
  // cutout map's own alpha when `alphaFromAlpha`) and makeup laid over it, then the whole
  // recoloured; or, with a flat base colour, the colour baked into the image.
  TextureSlot base(const std::string& color, const Rgb& factor, const std::string& alpha, const std::optional<MakeupLayer>& makeup,
            const std::optional<IrisRequest>& iris = std::nullopt, const std::optional<RecolorRequest>& recolor = std::nullopt,
            bool alphaFromAlpha = false)
  {
    if(alpha.empty() && !makeup && !iris && !recolor)
      return color.empty() ? TextureSlot{} : plain(color, true);
    TextureRequest r;
    r.srgb     = true;
    r.rel      = !color.empty() ? color : makeup && !makeup->map.empty() ? makeup->map : makeup && !makeup->weightMap.empty() ? makeup->weightMap : alpha;
    r.alphaRel       = alpha;
    r.alphaFromAlpha = !alpha.empty() && alphaFromAlpha;
    r.iris           = iris;
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
// DSON reader uses it: colour = exp(-0.45 sqrt(sigma) fit). The hair shader's own inverse for cards
// (hairColorToAbsorption) has no 0.45; which one is right is open (docs/lighting-audit.md, hair).
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

EyeShell eyeShell(const Surface& s)
{
  if(s.kind != Kind::eCornea || !s.refraction || s.refraction->weight <= 0.05f)
    return EyeShell::eNone;
  std::string name = s.name;
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if(name.find("moisture") != std::string::npos)
    return EyeShell::eMoisture;
  if(name.find("tear") != std::string::npos)
    return EyeShell::eMeniscus;
  return EyeShell::eCornea;
}

float tearFilmIor(const Surface& s)
{
  const float ior = s.refraction ? s.refraction->ior : 0.f;
  return ior >= 1.30f && ior <= 1.40f ? ior : 1.376f;
}

Lowered lower(const Surface& surface, const Edits& e, const LowerOptions& opts)
{
  // the panel's maps in place of the surface's, and its fabric in place of a detail layer
  Surface s = withMapEdits(surface, e);
  if(auto fabric = fabricDetail(e, opts.metresPerUv))
    s.detail = std::move(fabric);
  Builder b;
  b.flipV     = opts.flipV;
  b.uv        = surfaceUv(s, e);
  b.mapTiling = &s.mapTiling;
  Material       out;
  out.name             = s.name;
  const SkinRole role  = skinRole(s);
  const bool     refractive = s.refraction && s.refraction->weight > 0.05f;
  // refraction, and flesh in the volume and measured modes, write the index themselves
  auto lobes = specularLobes(s, !refractive && !(role == SkinRole::eFlesh && (opts.skin == SkinMode::eVolume || opts.skin == SkinMode::eMeasured)));
  if(e.roughnessMap)
    lobes.roughness.map = *e.roughnessMap;
  // The Weighted mode's diffuse keeps its share under the glossy lobe; the panel's colour is scaled
  // as the file's is. Refraction leaves the base colour behind anyway. Iray's Fresnel-free glossy
  // becomes a lobe within about 10% of flat up to 80 degrees; the shares are exact for a smooth
  // surface, and for a rough one as far as the BSDF's albedo fit. Left out: the translucency's share,
  // and Diffuse Weight's map.
  const float diffuseScale = refractive ? 1.f : weightedDiffuseScale(s);
  const Rgb   picked       = e.color.value_or(s.base.color);
  const Rgb   baseColor{picked[0] * diffuseScale, picked[1] * diffuseScale, picked[2] * diffuseScale};
  // The eye's wet shells (see the specular below): the tear film, and what lies under it
  const EyeShell shell    = eyeShell(s);
  const bool     tearFilm = shell == EyeShell::eMoisture || shell == EyeShell::eMeniscus || (shell == EyeShell::eCornea && !opts.underTearFilm);
  const bool     underFilm = opts.underTearFilm && (shell == EyeShell::eCornea || s.kind == Kind::eEye);
  // Iray's refraction shows none of the base colour where it refracts, but glTF tints transmitted
  // light by it, so a map on a refracting surface would be looked through, on the way in and out
  // (Tri-X's eye presets put the whole eye's map on the eye moisture). Its colour goes as a flat one does.
  // A refraction weight map keeps it: the surface shows its map where the mask is dark (Colm
  // Jackson's eyes paint the sclera and iris on the moisture over a black eyeball, clear only where
  // eye1_rf.png is white), which the map left out showed as black eyes.
  const std::string baseMap = refractive && s.refraction->map.empty() ? std::string() : s.base.map;

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
  // grey diffuse colour, passed through twice, left a quarter of the light to reach the iris). It
  // tints it by its own colour (Glossy Color or Refraction Color: a red wine), which the eye's water
  // never takes.
  if(refractive && !anyMap)
    factor = lerp(factor, tearFilm || underFilm ? kWhite : s.refraction->tint, s.refraction->weight);
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
  eff.color       = picked;
  eff.opacity     = opacity;
  eff.alphaMode   = alphaMode == "OPAQUE" ? "opaque" : alphaMode == "MASK" ? "cutout" : "blend";
  eff.doubleSided = opts.ribbon || e.doubleSided.value_or(hashed || s.twoSided || s.context == Context::eItem);
  const float metalness = e.metalness.value_or(s.metallic.value);
  eff.metalness         = metalness;

  // a flat colour baked into the image (no base map) must not tint it again; with no image at all
  // (makeup with no maps of its own) the factor is the colour
  const bool bakedFlat = anyMap && baseMap.empty() && (makeup || !s.opacity.map.empty());
  out.baseColor = {(anyMap && makeup) || bakedFlat ? 1.f : factor[0], (anyMap && makeup) || bakedFlat ? 1.f : factor[1],
                   (anyMap && makeup) || bakedFlat ? 1.f : factor[2], opacity};
  out.metallic  = metalness;
  if(anyMap)
    out.baseColorMap = b.base(baseMap, baseColor, s.opacity.map, makeup, irisRequest(s, e, opts), recolor, s.opacityFromAlpha);

  // roughness and metalness: one lobe from Iray's layers, with the limits the real-time view has
  // The eye's wet shells. The tear film reflects at Fresnel's full strength from a smooth surface:
  // Iray's glossy weight on G9's moisture (0.40, so 1% head-on and 40% of Fresnel at grazing) left
  // catchlights about 2.5 times too dim. Under it, tissue in water is close to index-matched and
  // reflects next to nothing, so the iris and sclera's own lobes (a second, broader sheen through
  // the film) and a stacked cornea's go.
  float roughness = lobes.roughness.value;
  if(tearFilm)
    roughness = std::min(roughness, 0.02f);
  else if(refractive)
    roughness = std::min(roughness, 0.1f);
  else if(s.kind == Kind::eEye)
    roughness = std::min(roughness, 0.35f);
  // PBRSkin's lobe multipliers and panel edits can pass its range
  const float roughnessFactor = clamp(e.roughness.value_or(roughness), 0, 1);
  eff.roughness               = roughnessFactor;
  out.roughness               = roughnessFactor;
  // Roughness in green, metalness in blue, each times its factor. Iray's
  // metal is Metallic Weight times its map, a mask of grey: the cooker puts that grey in the
  // roughness map's blue, or, with no roughness map, makes the image from the mask alone (green
  // white). With no metal the mask would only be multiplied by 0, so it is left out.
  const std::string metalMap = metalness > 0 ? s.metallic.map : std::string();
  if(!lobes.roughness.map.empty() || !metalMap.empty())
  {
    TextureRequest r;
    r.srgb = false;
    if(!lobes.roughness.map.empty())
    {
      r.rel      = lobes.roughness.map;
      r.channels = "mr";
      r.metalRel = metalMap.empty() ? std::string() : mapLayers(metalMap).front();
    }
    else
    {
      r.rel      = metalMap;
      r.channels = "mt";
    }
    out.metallicRoughnessMap = b.info(std::move(r));
  }
  out.alphaMode   = alphaMode == "OPAQUE" ? AlphaMode::eOpaque : alphaMode == "MASK" ? AlphaMode::eMask : AlphaMode::eBlend;
  out.doubleSided = eff.doubleSided;

  // normal: the normal map and the bump map together when a surface has both (there is one slot),
  // else whichever one it has. DAZ skin keeps pores in the bump map. The bump strength is baked
  // into the blend; the normal map's strength stays the slot's scale. A bump's height is per the
  // authored tile, the surface's times the map's own, so the panel's tiling changes no bake.
  const float authoredTiles = std::sqrt(std::abs(s.tiles[0] * s.tiles[1]));
  const auto  tilesOf       = [&](const TextureRequest& r) {
    const MapTiling t = b.tilingOf(r);
    return authoredTiles * std::sqrt(std::abs(t.tiles[0] * t.tiles[1]));
  };
  TextureSlot normal;
  if(!s.normalMap.empty() && !s.bumpMap.empty())
  {
    TextureRequest r;
    r.rel          = s.normalMap;
    r.srgb         = false;
    r.channels     = "ng";
    r.bumpRel      = s.bumpMap;
    r.bumpStrength = clamp(e.bumpStrength.value_or(s.bumpStrength), 0, 2);
    r.bumpHeight   = bumpHeightInUv(opts.metresPerUv, tilesOf(r), s.bumpRange.value_or(kBumpHeight));
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
    r.rel        = s.bumpMap;
    r.srgb       = false;
    r.channels   = "bn";
    r.bumpHeight = bumpHeightInUv(opts.metresPerUv, tilesOf(r), s.bumpRange.value_or(kBumpHeight));
    normal     = b.info(std::move(r), clamp(e.bumpStrength.value_or(s.bumpStrength), 0, 2));
    eff.bumpStrength = clamp(e.bumpStrength.value_or(s.bumpStrength), 0, 2);
  }
  // A tear film is smooth water: the normal map a preset hands it is the iris's own (Addy Florence's
  // eyes put the iris map on the moisture), which rippled the catchlight. The meniscus keeps its map,
  // the shape of the water along the lid.
  if(tearFilm && shell != EyeShell::eMeniscus)
  {
    normal = {};
    eff.normalStrength = 0;
    eff.bumpStrength   = 0;
  }
  out.normalMap = normal;

  // emission: an Iray emitter at its physical brightness
  if(s.emission)
  {
    // the temperature tint is normalised to luminance, so a warm light's red passes 1: the colour
    // stops there, and the excess goes into the strength
    Rgb         color = mul(s.emission->color.color, temperatureTint(s.emission->temperature));
    const float peak  = std::max({color[0], color[1], color[2]});
    const float scale = peak > 1 ? peak : 1.f;
    color             = {color[0] / scale, color[1] / scale, color[2] / scale};
    out.emissive = color;
    if(!s.emission->color.map.empty())
      out.emissiveMap = b.plain(s.emission->color.map, true);
    out.emissiveStrength  = opts.emission * scale;
    eff.emissive          = color;
    eff.emissiveIntensity = opts.emission * scale;
  }
  // an emissive colour given in the panel to a surface that had none
  if(e.emissive && !s.emission && ((*e.emissive)[0] > 0 || (*e.emissive)[1] > 0 || (*e.emissive)[2] > 0))
  {
    out.emissive          = *e.emissive;
    out.emissiveStrength  = e.emissiveIntensity.value_or(1.f);
    eff.emissive          = *e.emissive;
    eff.emissiveIntensity = e.emissiveIntensity.value_or(1.f);
  }
  else if(s.emission)
  {
    // edits of an emitter's colour and brightness
    if(e.emissive)
    {
      out.emissive = *e.emissive;
      eff.emissive = *e.emissive;
    }
    if(e.emissiveIntensity)
    {
      out.emissiveStrength  = *e.emissiveIntensity;
      eff.emissiveIntensity = *e.emissiveIntensity;
    }
  }

  // The tight dual lobe is a lobe of its own. For a file, which has only the coat, and in the real-time
  // view, whose BSDF has no second lobe, it is the coat unless a top coat is: then it stays in the
  // base specular.
  const bool secondLobe = lobes.clearcoat && opts.skin != SkinMode::eRaster && !opts.forFile;
  // A file of strand hair (no fibre BSDF in other tools) is a dielectric sheet of index 1.55.
  // The live path shades the fibre itself and does not also put that sheet on it.
  const bool  ribbonHair = s.kind == Kind::eHair && s.twoSided;
  const float specular   = ribbonHair && !liveRibbon ? 1.f
                           : s.topCoat && lobes.clearcoat && !secondLobe ? std::min(1.f, lobes.intensity + lobes.clearcoat->weight)
                                                          : lobes.intensity;
  eff.specular = e.specular.value_or(tearFilm ? 1.f : underFilm ? 0.f : specular);
  // Glossy Color tints the glossy layer's share of the lobe, normal to grazing; the dual lobes' share
  // stays white. This BSDF weighs the base under the lobe by its colour's brightest channel, capped
  // at 1, but tints the lobe by all of it, so the colour stays within 1. The eye's water is clear.
  const bool tinted        = !tearFilm && !underFilm && lobes.glossyShare > 0.01f;
  Rgb        specularColor = tinted ? lerp(kWhite, s.glossy.color, lobes.glossyShare) : kWhite;
  for(float& c : specularColor)
    c = clamp(c, 0, 1);
  Specular specularLobe;
  specularLobe.factor = eff.specular;
  specularLobe.color  = specularColor;
  if(tinted && !s.glossy.colorMap.empty())
    specularLobe.colorMap = b.plain(s.glossy.colorMap, true);
  // Where the lobe shines: the weight and reflectivity maps of the layers it is made of, in the
  // alpha the specular's map is read from (CJ Evangeline's Skin_SLW, the Weighted mode's Glossy
  // Weight map). A tight dual lobe folded in adds its share, so the sum keeps a map only when both
  // parts have the same. The eye's water and strand hair have their own.
  const std::vector<std::string> specularMaps = secondLobe || !s.topCoat || !lobes.clearcoat || lobes.clearcoat->weightMaps == lobes.intensityMaps
                                                    ? lobes.intensityMaps
                                                    : std::vector<std::string>{};
  if(!specularMaps.empty() && !tearFilm && !underFilm && !ribbonHair && specular > 0)
    specularLobe.map = b.weights(specularMaps);
  out.specular = specularLobe;
  // A layer brighter than 4% at normal incidence reflects through an index of its own. Glass, skin
  // and ribbon hair below set theirs over it.
  if(lobes.reflectance > 0.04f && lobes.intensity > 0 && !tearFilm && !underFilm)
    out.ior = iorForReflectance(lobes.reflectance);
  if(ribbonHair && !liveRibbon)
    out.ior = 1.55f;
  // sheen: the panel's, a cloth's soft rim of light at grazing angles
  if(e.sheenColor && ((*e.sheenColor)[0] > 0 || (*e.sheenColor)[1] > 0 || (*e.sheenColor)[2] > 0))
    out.sheen = Sheen{.color = *e.sheenColor, .roughness = clamp(e.sheenRoughness.value_or(0.5f), 0, 1)};

  // clearcoat: Iray's top coat, else the tighter dual lobe where it has no lobe of its own, on the
  // same surface detail as the base
  struct Coat
  {
    ScalarMap weight;
    ScalarMap roughness;
    float     bumpWeight = 1;
    // the tight dual lobe's weight maps, the same cooked image as the specular's
    std::vector<std::string> weightMaps;
  };
  std::optional<Coat> coat;
  const bool          colouredCoat = opts.skin != SkinMode::eRaster && !opts.forFile;
  if(tearFilm || underFilm)
    ;  // one interface, already the specular lobe; nothing reflects under it
  else if(s.topCoat)
  {
    // The coat's colour tints only its own reflection. The real-time view and a file have no such
    // thing, so there its luminance dims the coat's weight (with a little less light reaching the base
    // towards grazing than in Iray), and a map of it is left out.
    const Rgb&  c    = s.topCoat->color;
    const float tint = colouredCoat ? 1.f : 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
    coat = Coat{{s.topCoat->weight * tint, s.topCoat->weightMap}, {s.topCoat->roughness, s.topCoat->roughnessMap}, s.topCoat->bumpWeight};
  }
  else if(lobes.clearcoat && !secondLobe)
    coat = Coat{{lobes.clearcoat->weight, {}}, lobes.clearcoat->roughness, 1, lobes.clearcoat->weightMaps};
  // The panel's clearcoat sliders move the coat where there is one, else the second lobe, the surface's
  // tightest highlight.
  const bool slidersMoveSecondLobe = secondLobe && !s.topCoat;
  if(secondLobe && !tearFilm && !underFilm)
  {
    Specular2 lobe;
    lobe.factor    = clamp(slidersMoveSecondLobe ? e.clearcoat.value_or(lobes.clearcoat->weight) : lobes.clearcoat->weight, 0, 1);
    lobe.roughness = clamp(slidersMoveSecondLobe ? e.clearcoatRoughness.value_or(lobes.clearcoat->roughness.value) : lobes.clearcoat->roughness.value, 0, 1);
    if(!lobes.clearcoat->weightMaps.empty())
      lobe.map = b.weights(lobes.clearcoat->weightMaps);
    if(!lobes.clearcoat->roughness.map.empty())
      lobe.roughnessMap = b.plain(lobes.clearcoat->roughness.map, false);
    out.specular2 = lobe;
    if(slidersMoveSecondLobe)
    {
      eff.clearcoat          = lobe.factor;
      eff.clearcoatRoughness = lobe.roughness;
    }
  }
  if(coat || (e.clearcoat && !out.specular2))
  {
    Clearcoat cc;
    cc.factor    = clamp(e.clearcoat.value_or(coat ? coat->weight.value : 0.f), 0, 1);
    cc.roughness = clamp(e.clearcoatRoughness.value_or(coat ? coat->roughness.value : 0.1f), 0, 1);
    if(coat && !coat->weight.map.empty())
      cc.map = b.plain(coat->weight.map, false);
    else if(coat && !coat->weightMaps.empty())
      cc.map = b.weights(coat->weightMaps);
    if(coat && !coat->roughness.map.empty())
      cc.roughnessMap = b.plain(coat->roughness.map, false);
    if(s.topCoat && coat && colouredCoat)
    {
      cc.color = s.topCoat->color;
      if(!s.topCoat->colorMap.empty())
        cc.colorMap = b.plain(s.topCoat->colorMap, true);
    }
    if(normal.has())
    {
      // the same slot again: the clearcoat shares the base's surface detail, tilted by the coat's
      // bump weight (DAZ blends towards the smooth normal, which scales slopes by about the weight).
      // At weight 0 the slot stays with scale 0: left out, the shader would give the coat the
      // base's bent normal, the full relief.
      TextureSlot n    = normal;
      const float bump = coat ? coat->bumpWeight : 1.f;
      if(bump < 1)
        n.scale = n.scale.value_or(1.f) * bump;
      cc.normalMap = n;
    }
    out.clearcoat          = cc;
    eff.clearcoat          = cc.factor;
    eff.clearcoatRoughness = cc.roughness;
  }

  // refraction
  float       transmission = 0;
  std::string transmissionMap;  // Translucency Weight's map, where a skin lets light in
  if(refractive)
  {
    transmission = s.refraction->weight;
    out.ior      = tearFilm ? tearFilmIor(s) : s.refraction->ior;
  }

  // Refraction wins over skin: a cornea with a translucency weight stays glass.
  if(!refractive && role != SkinRole::eNone && opts.skin != SkinMode::eRaster)
  {
    // Translucency, as Iray Uber lays skin out: only (1 - Translucency Weight) of the diffuse light
    // reflects off the Base Color; the rest enters through the Translucency Color and scatters
    // back out. Genesis base colour maps are painted pale for that, and rendered as plain diffuse
    // they look grey.
    const auto& t = s.translucency;
    // A thin-walled PBRSkin (Translucency Enable off) has no inside: it stays a plain surface in
    // every mode, falling through to the thin branch at weight 0.
    const bool thinWalled = t && t->thinWalled;
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
    const auto        scatterTexture = [&](const Rgb& color) -> TextureSlot {
      if(scatterMap.empty())
        return {};
      return makeup || recolor ? b.base(scatterMap, color, {}, makeup, std::nullopt, recolor) : b.plain(scatterMap, true);
    };
    // The same albedo for both modes. A map or makeup already carries the colour, so the factor
    // stays white and the shader does not apply it twice.
    const Rgb bakedFactor = makeup ? (!scatterMap.empty() || !makeup->map.empty() || !makeup->weightMap.empty() ? kWhite : lerp(scatterColor, makeup->color, makeup->weight))
                                   : scatterColor;
    const auto albedoTexture = [&]() -> TextureSlot {
      TextureSlot tex = scatterTexture(scatterColor);
      if(tex.has())
        return tex;
      if(makeup && (!makeup->map.empty() || !makeup->weightMap.empty()))
        return b.base({}, scatterColor, {}, makeup);
      return {};
    };
    if(role == SkinRole::eFlesh && opts.skin == SkinMode::eMeasured && !thinWalled)
    {
      // Skin as measured, whatever the file's Iray medium says. Every bit of diffuse light enters
      // (skin has no diffuse layer above its tissue, only the oily film the specular lobes are),
      // and the renderer makes the medium from the skin map where the walk enters and how far
      // light spreads in skin per channel: the map is the colour lit from the front, the distance
      // how soft pores and shadow edges are and how far red glows through ears and fingers.
      transmission        = 1;
      out.ior             = 1.4f;
      out.volume          = Volume{.thickness = 1, .attenuationDistance = 1};
      out.scatter         = Scatter{.strength = 1};
      const float scale   = e.scatterScale.value_or(1.f);
      out.scatterDistance = Rgb{kSkinScatterDistance[0] * scale, kSkinScatterDistance[1] * scale, kSkinScatterDistance[2] * scale};
    }
    else if(role == SkinRole::eFlesh && opts.skin == SkinMode::eVolume && !thinWalled)
    {
      // Volumetric subsurface scattering, Iray's way: the translucent share enters (transmission =
      // weight) and random-walks through a medium with Iray's own scattering and absorption per
      // channel, which is where the warmth comes from; the light leaves through the skin map as a
      // diffuse exit, which also serves the share that reflects as plain diffuse.
      // All of it at weight 1, as Iray's translucency: the renderer enters the walk through a
      // diffuse lobe (ptDiffuseEntry), so the share no longer needs a diffuse layer left above it.
      const VolumeMedium v = volumeMedium(t ? t->subsurface : std::nullopt);
      transmission         = std::min(1.0f, weight);
      transmissionMap      = t ? t->weightMap : std::string();
      out.ior     = 1.4f;
      out.volume  = Volume{.thickness = 1, .attenuationDistance = v.distance, .attenuationColor = v.attenuation};
      out.scatter = Scatter{.strength = 1, .multiscatterColor = v.multiscatter, .anisotropy = v.anisotropy};
      // The base colour stays the skin map's own. The medium already warms what scatters, and the
      // walk leaves through the base colour, so a translucency tint here would be applied twice.
    }
    else if(weight > 0)
    {
      // Thin-walled scattering: the weight goes out as transmission with no volume, and
      // scatter at full strength turns that share into a Lambertian lobe coloured by
      // the base map, which the renderer evaluates at the surface: no random walk, so open eye
      // sockets and mouths stay clean. Anisotropy -0.8 sends a tenth of it on through, so backlit
      // ears and fingers glow. Makeup lies over the whole skin, so the light that scatters back
      // leaves through it too: it is baked into the scatter colour as into the base colour.
      // Nails and teeth take this branch in the measured mode too, so their weight map applies there.
      transmission    = std::min(1.0f, weight);
      transmissionMap = t ? t->weightMap : std::string();
      out.volume      = Volume{};
      out.scatter     = Scatter{.strength = 1, .multiscatterColor = bakedFactor, .multiscatterColorMap = albedoTexture(), .anisotropy = -0.8f};
    }
  }
  // The white of an eye under measured skin: dense scattering tissue, so light comes back out a few
  // tenths of a millimetre from where it went in and the lashes' shadows soften instead of printing
  // as stripes. The eyeball's surfaces are no closed medium for a walk (one darkened the eyes), so
  // the renderer spreads where its light samples look for shadow over the surface instead
  // (a scatter distance with no volume). A Genesis 9 eye's iris is in the same surface.
  if(const EyeLayout eye = eyeLayout(s);
     !refractive && opts.skin == SkinMode::eMeasured && (eye == EyeLayout::eShared || eye == EyeLayout::eSclera))
    out.scatterDistance = kScleraScatterDistance;
  if(transmission > 0)
  {
    out.transmission = Transmission{.factor = transmission};
    // where it refracts, or where a skin lets light in (pbr_skin.mdl's entering share is Translucency
    // Weight times its map): the weight's map, in red
    if(refractive && !s.refraction->map.empty())
      out.transmission->map = b.plain(s.refraction->map, false);
    else if(!transmissionMap.empty())
      out.transmission->map = b.plain(transmissionMap, false);
    if(!out.volume)
      out.volume = Volume{};
  }

  // PBRSkin's detail layer: a tiled normal map in glTF's convention bending the normal on top of the
  // normal map, with a roughness multiplier, under a weight. No map: the roughness multiplier alone
  if(s.detail && !opts.forFile)
  {
    const DetailLayer& d = *s.detail;
    Detail             de;
    de.weight    = d.weight;
    de.roughness = d.roughness;
    if(!d.map.empty())
    {
      TextureRequest r;
      r.rel                    = d.map;
      r.srgb                   = false;
      r.channels               = d.heightMap ? "bn" : "ng";
      if(d.heightMap)
        r.bumpHeight = d.bumpHeight ? *d.bumpHeight : bumpHeightInUv(opts.metresPerUv, std::sqrt(d.tiles[0] * d.tiles[1]));
      de.normalMap = b.info(std::move(r), d.strength, UvTransform{d.tiles, d.offset});
    }
    // the weight map lies on the surface's UVs; the detail's own maps keep their tiling (a weave its size)
    if(!d.weightMap.empty())
      de.weightMap = b.plain(d.weightMap, false);
    if(!d.roughnessMap.empty())
      de.roughnessMap = b.plain(d.roughnessMap, false, std::nullopt, UvTransform{d.tiles, d.offset});
    out.detail = de;
  }

  // Hair cards: the highlight stretches across the strands, a band. The rotation is measured from
  // the mesh in the renderer's convention. An authored Glossy Anisotropy of 0 keeps it round.
  // A live ribbon uses the per-hit strand instead of one angle for the whole material.
  if(s.kind == Kind::eHair && !liveRibbon && opts.anisotropyRotation)
    if(const auto strength = strandStrength(s, roughnessFactor, lobes.glossyShare))
      out.anisotropy = Anisotropy{.strength = *strength, .rotation = opts.forFile ? -*opts.anisotropyRotation : *opts.anisotropyRotation};

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
    out.roughness        = betaM;
    out.specular->factor = hairSpecular;
    out.hair             = Hair{.longitudinal   = betaM,
                                .azimuthal      = betaN,
                                .rootAbsorption = rootSigma,
                                .tipAbsorption  = tipSigma,
                                .rootFade       = 0.06f,
                                .rootV          = opts.flipV ? 1.f : 0.f,
                                .specular       = hairSpecular};
  }
  else if(hairCards(s) && !opts.forFile && opts.skin != SkinMode::eRaster)
  {
    // DAZ cards as fibres, in the path tracer: each hit's absorption from the colour the card shows
    // there (Chiang's fit inverted, as Unreal shades cards), so the painted strands keep their
    // shade, with a white R highlight, a tinted TRT band and light through the lock. The strand
    // runs across the measured anisotropy rotation, along UV v without one. The real-time view
    // keeps the GGX sheet above (it lowers with SkinMode::eRaster).
    const float betaM = cardLongitudinalRoughness(roughnessFactor);
    out.hair          = Hair{.longitudinal        = betaM,
                               .azimuthal           = kCardAzimuthalRoughness,
                               .specular            = clamp(e.specular.value_or(1.f), 0.f, 1.f),
                               .absorptionFromColor = true,
                               .strandRotation      = opts.anisotropyRotation};
  }
  else if(ribbonHair)
  {
    // A file for other tools has no fibre BSDF. A flat transmission keeps a stack of ribbons from
    // shading itself black; the live path gets that from the transmitted lobe instead.
    out.diffuseTransmission = DiffuseTransmission{.factor = 0.4f, .color = {out.baseColor[0], out.baseColor[1], out.baseColor[2]}};
  }

  return {std::move(out), std::move(b.requests), std::move(eff)};
}

bool needsArea(const Surface& s)
{
  return s.emission && s.emission->luminance && s.emission->luminance->perArea;
}

float bumpHeightInUv(float metresPerUv, float tiles, float height)
{
  if(!(metresPerUv > 0) || !(tiles > 0) || !(height > 0))
    return 0;
  const double h    = double(height) * tiles / metresPerUv;
  const double step = std::pow(10.0, std::floor(std::log10(h)) - 1);
  return float(std::round(h / step) * step);
}

Surface withMapEdits(const Surface& s, const Edits& e)
{
  Surface out = s;
  if(e.baseMap)
    out.base.map = *e.baseMap;
  if(e.opacityMap)
  {
    out.opacity.map      = *e.opacityMap;
    out.opacityFromAlpha = false;  // another image, read as grey
  }
  if(e.normalMap)
    out.normalMap = *e.normalMap;
  if(e.bumpMap)
    out.bumpMap = *e.bumpMap;
  return out;
}

std::string lowerRoughnessMap(const Surface& s, const Edits& e)
{
  return e.roughnessMap ? *e.roughnessMap : specularLobes(s).roughness.map;
}

std::vector<MapSlot> mapSlots(const Surface& surface, const Edits& e)
{
  const Surface        s = withMapEdits(surface, e);
  std::vector<MapSlot> out;
  const auto           add = [&](const char* slot, const char* label, const std::string& rel, bool edited) {
    if(!rel.empty() || edited)
      out.push_back({slot, label, mapLayers(rel).front(), edited});
  };
  add("base", "Base colour", s.base.map, e.baseMap.has_value());
  add("opacity", "Cutout", s.opacity.map, e.opacityMap.has_value());
  add("normal", "Normal", s.normalMap, e.normalMap.has_value());
  add("bump", "Bump", s.bumpMap, e.bumpMap.has_value());
  add("roughness", "Roughness", lowerRoughnessMap(s, e), e.roughnessMap.has_value());
  if(s.translucency)
    add("translucency", "Translucency", s.translucency->map, false);
  if(s.emission)
    add("emission", "Emission", s.emission->color.map, false);
  if(const Fabric* f = e.fabric ? findFabric(*e.fabric) : nullptr)
    add("detail", "Fabric", fabricHeightMap(*f), true);
  else if(s.detail)
    add("detail", "Detail", s.detail->map, false);
  if(s.makeup)
    add("makeup", "Makeup", s.makeup->map, false);
  return out;
}

bool UvTransform::identity() const
{
  return scale[0] == 1 && scale[1] == 1 && offset[0] == 0 && offset[1] == 0 && rotation == 0;
}

UvTransform surfaceUv(const Surface& s, const Edits& e)
{
  const std::array<float, 2> scale{e.uvScaleU.value_or(1.f), e.uvScaleV.value_or(1.f)};
  const float                theta = e.uvRotation.value_or(0.f) * float(std::numbers::pi / 180);
  const float                c = std::cos(theta), sn = std::sin(theta);
  // the authored offset, scaled and turned by the edit: T_e + R_e · S_e · T_a
  const float x = scale[0] * s.offset[0], y = scale[1] * s.offset[1];
  UvTransform t;
  t.scale    = {scale[0] * s.tiles[0], scale[1] * s.tiles[1]};
  t.rotation = theta;
  t.offset   = {e.uvOffsetU.value_or(0.f) + c * x + sn * y, e.uvOffsetV.value_or(0.f) - sn * x + c * y};
  return t;
}

UvTransform mirroredV(const UvTransform& t)
{
  // M(v) = (u, 1 − v) on both sides: M·(T + R(θ)·S·M(uv)) = (T_u + s_v·sin θ, 1 − T_v − s_v·cos θ) + R(−θ)·S·uv
  UvTransform out = t;
  out.rotation    = -t.rotation;
  out.offset      = {t.offset[0] + t.scale[1] * std::sin(t.rotation), 1 - t.offset[1] - t.scale[1] * std::cos(t.rotation)};
  return out;
}

UvTransform rowsFlipped(const UvTransform& t)
{
  // F(u, v) = (u, 1 − v) after T + R(θ)·S: F·(T + R(θ)·S·uv) = (T_u, 1 − T_v) + R(−θ)·diag(s_u, −s_v)·uv
  return {.scale = {t.scale[0], -t.scale[1]}, .offset = {t.offset[0], 1 - t.offset[1]}, .rotation = -t.rotation};
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

Material resolved(const Lowered& lowered, const std::function<ResolvedTexture(const TextureRequest&)>& resolve)
{
  Material out = lowered.material;
  forEachSlot(out, [&](TextureSlot& slot) {
    if(!slot.has())
      return;
    const ResolvedTexture t = resolve(lowered.requests[size_t(slot.index)]);
    if(t.index < 0)
    {
      slot = {};
      return;
    }
    slot.index     = t.index;
    slot.flipGreen = t.flipGreen;
    if(t.flipRows)
      slot.transform = rowsFlipped(slot.transform.value_or(UvTransform{}));
  });
  return out;
}

tinygltf::Material toGltf(const Lowered& lowered, const std::function<ResolvedTexture(const TextureRequest&)>& resolve,
                          std::vector<std::string>& extensionsUsed)
{
  return toGltf(resolved(lowered, resolve), extensionsUsed);
}

}  // namespace fv::mat
