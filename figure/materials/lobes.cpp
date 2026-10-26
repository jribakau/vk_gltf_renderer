#include <figure/materials/lobes.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace fv::mat {

SpecularLobes specularLobes(const Surface& s)
{
  const float glossF = s.glossy.weight * s.glossy.reflectivity * 2;
  ScalarMap   roughness = s.glossy.roughness;
  if(!s.dual)
    return {std::clamp(glossF, 0.f, 1.f), roughness, std::nullopt, 1};
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
  out.intensity   = std::clamp(glossF + baseDual, 0.f, 1.f);
  out.roughness   = roughness;
  return out;
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
    c.sigmaS[i] = !std::isinf(ds) ? sss->scatterColor[i] / ds : 0.f;
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
  // Planckian locus from Kim et al. (2002), valid from 1667 K to 25000 K
  const auto rgb = [](double t) {
    t = std::clamp(t, 1667.0, 25000.0);
    const double x = t <= 4000 ? -0.2661239e9 / (t * t * t) - 0.2343589e6 / (t * t) + 0.8776956e3 / t + 0.17991
                               : -3.0258469e9 / (t * t * t) + 2.1070379e6 / (t * t) + 0.2226347e3 / t + 0.24039;
    const double y = t <= 2222   ? -1.1063814 * x * x * x - 1.3481102 * x * x + 2.18555832 * x - 0.20219683
                     : t <= 4000 ? -0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
                                 : 3.081758 * x * x * x - 5.8733867 * x * x + 3.75112997 * x - 0.37001483;
    const double X = x / y, Z = (1 - x - y) / y;
    return std::array<double, 3>{3.2404542 * X - 1.5371385 - 0.4985314 * Z, -0.969266 * X + 1.8760108 + 0.041556 * Z,
                                 0.0556434 * X - 0.2040259 + 1.0572252 * Z};
  };
  const auto c = rgb(kelvin);
  const auto w = rgb(6500);
  Rgb        out{float(std::max(0.0, c[0] / w[0])), float(std::max(0.0, c[1] / w[1])), float(std::max(0.0, c[2] / w[2]))};
  const float lum = 0.2126f * out[0] + 0.7152f * out[1] + 0.0722f * out[2];
  if(lum <= 0)
    return {1, 1, 1};
  for(float& v : out)
    v /= lum;
  return out;
}

}  // namespace fv::mat
