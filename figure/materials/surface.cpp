#include <figure/materials/surface.hpp>

#include <cctype>
#include <cmath>
#include <regex>
#include <utility>
#include <vector>

namespace fv::mat {

namespace {

// "EyeMoisture_L" -> "eye moisture l"
std::string wordsOf(const std::string& name)
{
  std::string out;
  for(size_t i = 0; i < name.size(); i++)
  {
    const unsigned char c = static_cast<unsigned char>(name[i]);
    if(!std::isalpha(c))
    {
      if(!out.empty() && out.back() != ' ')
        out += ' ';
      continue;
    }
    if(std::isupper(c) && i > 0 && std::islower(static_cast<unsigned char>(name[i - 1])))
      out += ' ';
    out += char(std::tolower(c));
  }
  return out;
}

}  // namespace

Kind classify(const std::string& name, Context context)
{
  using R = std::regex;
  const auto f = R::ECMAScript | R::optimize;
  static const std::vector<std::pair<Kind, R>> rules{
      {Kind::eCornea, R("\\bcornea\\b|\\bmoisture\\b|\\btears?\\b|\\beye ?refract", f)},
      {Kind::eLashes, R("\\b(eye ?)?lash(es)?\\b", f)},
      {Kind::eHair, R("\\bhairs?\\b|\\bscalp\\b|\\b(eye ?)?brows?\\b|\\bbeard\\b|\\bmustache\\b|\\bfibermesh\\b|\\bstrands?\\b|\\bbangs\\b|\\bponytail\\b", f)},
      {Kind::eEye, R("\\bsclera\\b|\\biris(es)?\\b|\\bpupils?\\b|\\beyes?\\b(?! (socket|lid|reflection))|\\beyeball", f)},
      {Kind::eNails, R("\\b(finger ?|toe ?)?nails?\\b", f)},
      {Kind::eTeeth, R("\\bteeth\\b|\\btooth\\b", f)},
      {Kind::eMouth, R("\\bmouth\\b|\\bgums?\\b|\\btongue\\b|\\boral\\b|\\binner ?mouth\\b", f)},
      {Kind::eSkin,
       R("\\b(face|head|((up|low)er ?)?lips?|ears?|body|torso|(fore)?arms?|legs?|necks?|hips?|skin|shoulders?|hands?|foot|feet|genital(ia|s)?|anatomy|"
         "nipples?|areolae?|chest|thighs?|shins?|toes?|fingers?)\\b|\\bback$",
         f)},
  };
  // where hair is expected, what only hair is called there: a "Hairline", a "Skullcap", a "Cap"; on
  // anything else that is a bottle's cap. Never a "Chair".
  static const R hairHere("(^|[^c])hair|\\bscalp\\b|\\b(eye ?)?brows?\\b|\\bbeard\\b|\\bmustache\\b|\\bfibermesh\\b|\\bstrands?\\b|cap\\b|\\bbangs\\b|\\bponytail\\b", f);
  const std::string words = wordsOf(name);
  Kind              found = Kind::eOther;
  for(const auto& [kind, re] : rules)
    if(std::regex_search(words, kind == Kind::eHair && context != Context::eItem ? hairHere : re))
    {
      found = kind;
      break;
    }
  const bool flesh = found == Kind::eSkin || found == Kind::eMouth || found == Kind::eNails || found == Kind::eTeeth;
  if(!flesh || context == Context::eBody)
    return found;
  // an outfit made with the skin it shows says so ("Skin_Mat")
  static const R saysSkin("\\bskin\\b", f);
  return std::regex_search(words, saysSkin) ? found : Kind::eOther;
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
         && !scatterScale && !baseMap && !opacityMap && !normalMap && !bumpMap && !roughnessMap && !uvScaleU && !uvScaleV && !uvOffsetU
         && !uvOffsetV && !uvRotation && !fabric && !fabricScale && !fabricStrength && !sheenColor && !sheenRoughness;
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
  take(baseMap, o.baseMap);
  take(opacityMap, o.opacityMap);
  take(normalMap, o.normalMap);
  take(bumpMap, o.bumpMap);
  take(roughnessMap, o.roughnessMap);
  take(uvScaleU, o.uvScaleU);
  take(uvScaleV, o.uvScaleV);
  take(uvOffsetU, o.uvOffsetU);
  take(uvOffsetV, o.uvOffsetV);
  take(uvRotation, o.uvRotation);
  take(fabric, o.fabric);
  take(fabricScale, o.fabricScale);
  take(fabricStrength, o.fabricStrength);
  take(sheenColor, o.sheenColor);
  take(sheenRoughness, o.sheenRoughness);
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
