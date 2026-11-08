#include <figure/wardrobe/fit/generation.hpp>

#include <algorithm>
#include <regex>

namespace fv::wardrobe {

const char* label(Generation g)
{
  switch(g)
  {
    case Generation::eG2: return "Genesis 2";
    case Generation::eG3: return "Genesis 3";
    case Generation::eG8: return "Genesis 8";
    case Generation::eG81: return "Genesis 8.1";
    case Generation::eG9: return "Genesis 9";
    case Generation::eSkyrim: return "Skyrim";
  }
  return "Genesis";
}

bool sameTopology(Generation a, Generation b)
{
  const auto family = [](Generation g) { return g == Generation::eG3 || g == Generation::eG8 || g == Generation::eG81; };
  return family(a) && family(b);
}

bool sameRig(Generation a, Generation b)
{
  const auto g8 = [](Generation g) { return g == Generation::eG8 || g == Generation::eG81; };
  return a == b || (g8(a) && g8(b));
}

namespace {

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

}  // namespace

std::optional<Generation> generationFromPath(const std::string& path)
{
  if(path.empty())
    return std::nullopt;
  std::string s = lower(path);
  for(size_t at = s.find("%20"); at != std::string::npos; at = s.find("%20", at + 1))
    s.replace(at, 3, " ");
  // "Genesis 8.1", "Genesis8_1Female.dsf", a "Female 8_1" or "8_1" folder; not "Skirt08_1.nif"
  static const std::regex g81(R"(genesis\s*8\s*[._]\s*1|(^|/)((fe)?male )?8_1(/|$))");
  static const std::regex g9(R"(genesis\s*9\b|genesis9)");
  static const std::regex g8(R"(genesis\s*8\b|genesis8)");
  static const std::regex g3(R"(genesis\s*3\b|genesis3)");
  static const std::regex g2(R"(genesis\s*2\b|genesis2)");
  if(std::regex_search(s, g81))
    return Generation::eG81;
  if(std::regex_search(s, g9))
    return Generation::eG9;
  if(std::regex_search(s, g8))
    return Generation::eG8;
  if(std::regex_search(s, g3))
    return Generation::eG3;
  if(std::regex_search(s, g2))
    return Generation::eG2;
  return std::nullopt;
}

namespace {

// What the bones tell: a generation when only one has such bones, else whether they are Genesis 3
// or 8's (Bend/Twist limbs, a lower and upper chest), which share their names but for the heel.
struct BoneEvidence
{
  std::optional<Generation> sure;
  bool                      g3or8 = false;
};

BoneEvidence boneEvidence(const std::vector<std::string>& names)
{
  static const std::regex underscored(R"(^[lr]_(thigh|shin|foot|upperarm|forearm|hand))");
  static const std::regex heelRe(R"(^[lr]heel$)");
  static const std::regex bendRe(R"(^[lr](thigh|shldr|forearm)(bend|twist)$)");
  static const std::regex trunkRe(R"(^(abdomen|chest|neck)(lower|upper)$)");
  static const std::regex plainRe(R"(^[lr]thigh$)");
  bool                    under = false, heel = false, g3or8 = false, plain = false;
  for(const std::string& name : names)
  {
    const std::string n = lower(name);
    // Skyrim's bones ("NPC Head [Head]"), which no DAZ figure's are named like: a glove or a helmet has
    // only its hand's or head's
    if(n.rfind("npc ", 0) == 0)
      return {Generation::eSkyrim, false};
    under = under || std::regex_search(n, underscored);
    heel  = heel || std::regex_search(n, heelRe);
    g3or8 = g3or8 || std::regex_search(n, bendRe) || std::regex_search(n, trunkRe);
    plain = plain || std::regex_search(n, plainRe);
  }
  // underscored side prefixes (`l_thigh`) are Genesis 9, whose twist helpers would otherwise read as Genesis 8
  if(under)
    return {Generation::eG9, false};
  if(heel)
    return {Generation::eG3, false};
  if(g3or8)
    return {std::nullopt, true};
  if(plain)
    return {Generation::eG2, false};
  return {};
}

bool g3or8(Generation g)
{
  return g == Generation::eG3 || g == Generation::eG8 || g == Generation::eG81;
}

}  // namespace

std::optional<Generation> generationFromBones(const std::vector<std::string>& names)
{
  const BoneEvidence bones = boneEvidence(names);
  if(bones.g3or8)
    return Generation::eG8;
  return bones.sure;
}

std::optional<Generation> identifyGeneration(const std::string& path, const std::vector<std::string>& bones, const std::string& conformTarget,
                                             std::optional<Generation> stated)
{
  const BoneEvidence evidence = boneEvidence(bones);
  if(evidence.sure)
    return evidence.sure;
  for(const std::optional<Generation>& g : {stated, generationFromPath(conformTarget), generationFromPath(path)})
    if(g && (!evidence.g3or8 || g3or8(*g)))
      return g;
  // Genesis 3 or 8 bones in a folder that names neither: the heel would have told Genesis 3
  if(evidence.g3or8)
    return Generation::eG8;
  return std::nullopt;
}

std::string conformTargetOf(const dson::Json& doc)
{
  std::vector<std::string> blobs;
  if(const std::string id = dson::getString(dson::get(&doc, "asset_info"), "id"); !id.empty())
    blobs.push_back(id);
  const auto visit = [&](const dson::Json* list) {
    if(!list || !list->is_array())
      return;
    for(const dson::Json& n : *list)
    {
      if(const std::string url = dson::getString(&n, "url"); !url.empty())
        blobs.push_back(url);
      const dson::Json* extra = dson::get(&n, "extra");
      if(!extra || !extra->is_array())
        continue;
      for(const dson::Json& e : *extra)
      {
        if(const std::string ct = dson::getString(&e, "conform_target"); !ct.empty())
          blobs.push_back(ct);
        const dson::Json* channels = dson::get(&e, "channels");
        if(!channels || !channels->is_array())
          continue;
        for(const dson::Json& c : *channels)
        {
          std::string id = dson::getString(&c, "id");
          if(id.empty())
            id = dson::getString(&c, "name");
          const dson::Json* v = dson::get(&c, "current_value");
          static const std::regex conform("conform", std::regex::icase);
          if(std::regex_search(id, conform) && v && v->is_string())
            blobs.push_back(v->get<std::string>());
        }
      }
    }
  };
  visit(dson::get(dson::get(&doc, "scene"), "nodes"));
  visit(dson::get(&doc, "node_library"));
  for(const std::string& s : blobs)
    if(generationFromPath(s))
      return s;
  return {};
}

std::string genderHint(const std::string& path)
{
  const std::string s = lower(path);
  if(s.find("female") != std::string::npos)
    return "female";
  if(s.find("male") != std::string::npos)
    return "male";
  return {};
}

}  // namespace fv::wardrobe
