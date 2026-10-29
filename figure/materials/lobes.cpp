#include <figure/materials/lobes.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace fv::mat {

SpecularLobes specularLobes(const Surface& s, bool ownIndex)
{
  // Lobe strengths in units of a plain dielectric's 4% at normal incidence. Iray's reflectivity is
  // 0.08 x its value there; the Weighted mode's glossy reflects its weight, with no Fresnel.
  const bool  weighted = s.mixing == Surface::Mixing::eWeighted;
  const float glossF   = weighted ? s.glossy.weight / 0.04f : s.glossy.weight * s.glossy.reflectivity * 2;
  // The index's F0, which glTF's specularFactor then scales down: Spec/Gloss's Glossy Specular,
  // which Glossy Layered Weight scales whole, rim and all; for the Weighted mode, the index whose
  // Fresnel is nearly flat. Where the index is not the lobe's own, 4%.
  const float f0 = !ownIndex ? 0.04f
                   : weighted ? kFlatF0
                   : s.mixing == Surface::Mixing::eSpecGloss ? std::clamp(0.08f * s.glossy.reflectivity, 0.04f, kFlatF0)
                                                             : 0.04f;
  ScalarMap roughness = s.glossy.roughness;
  if(!s.dual)
    return {std::clamp(0.04f * glossF / f0, 0.f, 1.f), roughness, std::nullopt, 1, f0};
  struct Lobe
  {
    ScalarMap r;
    float     share;
  };
  const Lobe l1{s.dual->lobe1, s.dual->ratio};
  const Lobe l2{s.dual->lobe2, 1 - s.dual->ratio};
  const bool firstTight = l1.r.value <= l2.r.value;
  const Lobe& tight     = firstTight ? l1 : l2;
  const Lobe& broad     = firstTight ? l2 : l1;
  const float dualF     = s.dual->weight * s.dual->reflectivity * 2;
  // no glossy layer: the main lobe is the broad dual lobe
  if(glossF <= 0.01f && dualF > 0)
    roughness = broad.r;
  const float tightF = dualF * tight.share;
  // Only a lobe tight enough to read as a separate coat leaves the base. Otherwise its share stays in the base lobe.
  const bool  promoted = tightF > 0.01f && tight.r.value < roughness.value - 0.05f;
  SpecularLobes out;
  if(promoted)
    out.clearcoat = SpecularLobes::Coat{std::min(1.f, tightF), tight.r};
  const float baseDual = promoted ? dualF * (1 - tight.share) : dualF;
  // a glossy layer too faint to be the main lobe has no share in it
  out.glossyShare = glossF > 0.01f && glossF + baseDual > 0 ? glossF / (glossF + baseDual) : 0.f;
  out.intensity   = std::clamp(0.04f * (glossF + baseDual) / f0, 0.f, 1.f);
  out.roughness   = roughness;
  out.reflectance = f0;
  return out;
}

float iorForReflectance(float f0)
{
  const float r = std::sqrt(std::clamp(f0, 0.f, kFlatF0));
  return (1 + r) / (1 - r);
}

float weightedDiffuseScale(const Surface& s)
{
  if(s.mixing != Surface::Mixing::eWeighted)
    return 1;
  // the lobe takes about Glossy Weight x its brightest tint from the light under it, at every angle
  const float tint  = std::min(1.f, std::max({s.glossy.color[0], s.glossy.color[1], s.glossy.color[2]}));
  const float under = 1 - s.glossy.weight * tint;
  return under > 1e-4f ? std::clamp(s.diffuseWeight / under, 0.f, 1.f) : 0.f;
}

SkinRole skinRole(const Surface& s)
{
  switch(s.kind)
  {
    case Kind::eCornea:
    case Kind::eEye:
    case Kind::eHair:
    case Kind::eLashes:
    case Kind::eOther:
      return SkinRole::eNone;
    case Kind::eNails:
    case Kind::eTeeth:
      return SkinRole::eThin;
    case Kind::eSkin:
    case Kind::eMouth:
      return SkinRole::eFlesh;
  }
  return SkinRole::eNone;
}

std::optional<MakeupLayer> hairTranslucency(const Surface& s)
{
  const auto& t = s.translucency;
  if(s.kind != Kind::eHair || !t || t->map.empty() || t->weight <= 0)
    return std::nullopt;
  MakeupLayer m;
  m.color  = t->color.value_or(Rgb{1, 1, 1});
  m.map    = t->map;
  m.weight = t->weight;
  return m;
}

bool hairCards(const Surface& s)
{
  if(s.kind != Kind::eHair || s.hairFibre || s.opacity.map.empty())
    return false;
  std::string name = s.name;
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  for(const char* painted : {"cap", "scalp", "brow"})
    if(name.find(painted) != std::string::npos)
      return false;
  return true;
}

float cardLongitudinalRoughness(float roughness)
{
  const double alpha  = std::clamp(double(roughness), 0.0, 1.0) * std::clamp(double(roughness), 0.0, 1.0);
  const double target = 2.57 * alpha / 2.355;
  const auto   spread = [](double b) { return 0.726 * b + 0.812 * b * b + 3.7 * std::pow(b, 20); };
  double       lo = 0, hi = 1;
  for(int i = 0; i < 40; i++)
    (spread(0.5 * (lo + hi)) < target ? lo : hi) = 0.5 * (lo + hi);
  return float(0.5 * (lo + hi));
}

Rgb fallbackTranslucencyColor()
{
  // defaultSkin.sssColor, '#ff5a3c'
  return {srgbToLinear(1.0), srgbToLinear(0x5a / 255.0), srgbToLinear(0x3c / 255.0)};
}

namespace {

// Jensen's skin1 (2001), per metre: reduced scattering and absorption. Red scatters furthest and is barely absorbed.
constexpr Rgb kSkinSigmaS{740, 880, 1010};
constexpr Rgb kSkinSigmaA{32, 170, 480};

struct Coefficients
{
  Rgb sigmaS, sigmaA;
};

// Iray's scattering and absorption coefficients per channel (per metre), or none when the surface measures neither.
// Absorption is always logarithmic in Transmitted Color; scattering is in SSS Color only where the
// shader says so (PBRSkin), else linear.
std::optional<Coefficients> measuredCoefficients(const std::optional<Subsurface>& sss)
{
  if(!sss)
    return std::nullopt;
  constexpr float cm = 0.01f;
  const float     dt = sss->transmittedDistance > 0 ? sss->transmittedDistance * cm : INFINITY;
  const float     ds = sss->scatterDistance > 0 ? sss->scatterDistance * cm : INFINITY;
  if(std::isinf(dt) && std::isinf(ds))
    return std::nullopt;
  Coefficients c{};
  float        peak = 0;
  for(int i = 0; i < 3; i++)
  {
    c.sigmaA[i] = !std::isinf(dt) ? -std::log(std::max(1e-4f, sss->transmittedColor[i])) / dt : 0.f;
    const float sc = sss->logScatter ? std::max(0.f, -std::log(std::max(1e-4f, sss->scatterColor[i]))) : sss->scatterColor[i];
    c.sigmaS[i]    = !std::isinf(ds) ? sc / ds : 0.f;
    peak        = std::max({peak, c.sigmaA[i], c.sigmaS[i]});
  }
  return peak > 0 ? std::optional<Coefficients>(c) : std::nullopt;
}

}  // namespace

float singleScatterAlbedo(float rho, float g)
{
  const float s = 4.09712f + 4.20863f * rho - std::sqrt(9.59217f + 41.6808f * rho + 17.7126f * rho * rho);
  return std::clamp((1 - s * s) / (1 - g * s * s), 0.f, 1.f);
}

float multiscatterForAlbedo(float albedo, float g)
{
  if(albedo >= singleScatterAlbedo(1, g))
    return 1;
  if(albedo <= singleScatterAlbedo(0, g))
    return 0;
  float lo = 0, hi = 1;
  for(int i = 0; i < 48; i++)
  {
    const float mid = (lo + hi) / 2;
    if(singleScatterAlbedo(mid, g) < albedo)
      lo = mid;
    else
      hi = mid;
  }
  return (lo + hi) / 2;
}

VolumeMedium volumeMedium(const std::optional<Subsurface>& sss)
{
  const auto  measured = measuredCoefficients(sss);
  const Rgb   sigmaS   = measured ? measured->sigmaS : kSkinSigmaS;
  const Rgb   sigmaA   = measured ? measured->sigmaA : kSkinSigmaA;
  VolumeMedium v;
  v.anisotropy = measured ? sss->direction : 0.f;
  Rgb   sigma{};
  float peak = 0;
  for(int i = 0; i < 3; i++)
  {
    sigma[i] = sigmaS[i] + sigmaA[i];
    peak     = std::max(peak, sigma[i]);
  }
  // puts the densest channel at exp(-1), above the shader's 0.001 log floor
  v.distance = peak > 0 ? 1 / peak : 1.f;
  for(int i = 0; i < 3; i++)
  {
    v.attenuation[i]  = std::exp(-v.distance * sigma[i]);
    v.albedo[i]       = sigma[i] > 0 ? sigmaS[i] / sigma[i] : 0.f;
    v.multiscatter[i] = multiscatterForAlbedo(v.albedo[i], v.anisotropy);
  }
  return v;
}

Rgb temperatureTint(float kelvin)
{
  if(!(kelvin >= 1))
    return {1, 1, 1};  // Iray: below 1 K the colour alone
  // math::blackbody as the MDL SDK evaluates it (libmdlrt.cpp mdl_blackbody, Copyright NVIDIA
  // Corporation, BSD-3-Clause; see third_party/mdl_sdk): X/Y and Z/Y of a Planck spectrum under the
  // CIE 1931 observer, fitted as rational functions of T, at Y = 1 in linear Rec.709. Nothing makes
  // 6500 K white: that blackbody sits just below D65 and comes out a touch magenta, as in Iray.
  const double t  = std::max(double(kelvin), 500.0);
  const double t2 = t * t, t3 = t2 * t, t4 = t2 * t2;
  const auto   fit = [&](double a, double b, double c, double d, double e, double f, double g, double h, double i, double j) {
    return (a + b * t + c * t2 + d * t3 + e * t4) / (f + g * t + h * t2 + i * t3 + j * t4);
  };
  const double X = fit(1.0, -1.44688464e+11, 3.65139181e+08, -5.24496810e+04, 7.03428125e+01, 1.0, -3.89910067e+10, 7.73122653e+07,
                       1.99017054e+04, 6.78027157e+01);
  const double Z = fit(1.0, -1.39076125e+11, 6.08958603e+08, -8.68632924e+05, 4.08263556e+02, -8.56817534e+03, -9.49412452e+11,
                       2.40214241e+09, 1.47983610e+05, 1.81111205e+02);
  return {float(std::max(0.0, 3.2406 * X - 1.5372 - 0.4986 * Z)), float(std::max(0.0, -0.9689 * X + 1.8758 + 0.0415 * Z)),
          float(std::max(0.0, 0.0557 * X - 0.2040 + 1.0570 * Z))};
}

}  // namespace fv::mat
