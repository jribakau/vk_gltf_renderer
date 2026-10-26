#include <figure/materials/surface.hpp>

#include <cmath>
#include <regex>
#include <utility>
#include <vector>

namespace fv::mat {

Kind classify(const std::string& name)
{
  using R = std::regex;
  static const std::vector<std::pair<Kind, R>> rules = [] {
    const auto f = R::ECMAScript | R::icase | R::optimize;
    return std::vector<std::pair<Kind, R>>{
        {Kind::eCornea, R("cornea|moisture|tear|eye.?refract|eyerefract", f)},
        {Kind::eLashes, R("lash", f)},
        {Kind::eHair, R("hair|scalp|brow|beard|mustache|fibermesh|strand|cap\\b|bangs|ponytail", f)},
        {Kind::eEye, R("sclera|iris|pupil|eye[s ._-]?(left|right|l|r)\\b|\\beyes?\\b|eyeball", f)},
        {Kind::eNails, R("nail", f)},
        {Kind::eTeeth, R("teeth|tooth", f)},
        {Kind::eMouth, R("mouth|gums|tongue|inner.?mouth|oral", f)},
        {Kind::eSkin,
         R("face|head|lip|ear|body|torso|arm|leg|neck|hip|skin|shoulder|hand|foot|feet|genital|anatomy|nipple|areola|chest|back\\b|forearm|"
           "thigh|shin|toe(?!nail)|finger(?!nail)",
           f)},
    };
  }();
  for(const auto& [kind, re] : rules)
    if(std::regex_search(name, re))
      return kind;
  return Kind::eOther;
}

float srgbToLinear(double c)
{
  return float(c < 0.04045 ? c * 0.0773993808 : std::pow(c * 0.9478672986 + 0.0521327014, 2.4));
}

float linearToSrgb(double c)
{
  return float(c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055);
}

bool Edits::empty() const
{
  return !color && !roughness && !metalness && !opacity && !alphaMode && !doubleSided && !normalStrength && !bumpStrength && !specular
         && !clearcoat && !clearcoatRoughness && !emissive && !emissiveIntensity && !irisColor && !scleraColor && !irisStrength
         && !recolor && !recolorStrength && !tone && !toneStrength && !makeupColor && !makeupWeight
         && !scatterScale;
}

void Edits::merge(const Edits& o)
{
  const auto take = [](auto& mine, const auto& theirs) {
    if(theirs)
      mine = theirs;
  };
  take(color, o.color);
  take(roughness, o.roughness);
  take(metalness, o.metalness);
  take(opacity, o.opacity);
  take(alphaMode, o.alphaMode);
  take(doubleSided, o.doubleSided);
  take(normalStrength, o.normalStrength);
  take(bumpStrength, o.bumpStrength);
  take(specular, o.specular);
  take(clearcoat, o.clearcoat);
  take(clearcoatRoughness, o.clearcoatRoughness);
  take(emissive, o.emissive);
  take(emissiveIntensity, o.emissiveIntensity);
  take(irisColor, o.irisColor);
  take(scleraColor, o.scleraColor);
  take(irisStrength, o.irisStrength);
  take(recolor, o.recolor);
  take(recolorStrength, o.recolorStrength);
  take(tone, o.tone);
  take(toneStrength, o.toneStrength);
  take(makeupColor, o.makeupColor);
  take(makeupWeight, o.makeupWeight);
  take(scatterScale, o.scatterScale);
}

EyeLayout eyeLayout(const Surface& s)
{
  if(s.kind != Kind::eEye)
    return EyeLayout::eNone;
  using R = std::regex;
  const auto f = R::ECMAScript | R::icase;
  // shared: Genesis 9's names only, since the iris discs the bake looks in are where its UVs put them
  static const R shared("^eye (left|right)$", f), iris("iris", f), sclera("sclera", f);
  if(std::regex_search(s.name, shared))
    return EyeLayout::eShared;
  if(std::regex_search(s.name, iris))
    return EyeLayout::eIris;
  if(std::regex_search(s.name, sclera))
    return EyeLayout::eSclera;
  return EyeLayout::eNone;
}

}  // namespace fv::mat
