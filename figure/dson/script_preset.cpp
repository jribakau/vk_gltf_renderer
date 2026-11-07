#include <figure/dson/script_preset.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <figure/dson/channels.hpp>

namespace fv::dson {

namespace {

constexpr std::string_view kMarker = "filetype Materials Preset";

// The options of Iray Uber's enumerated properties, in order: what the .duf files list as each
// channel's enum_values (the AMG GT Coupe's, which every script preset in the library came with)
const std::map<std::string, std::vector<std::string_view>>& enumValues()
{
  static const std::map<std::string, std::vector<std::string_view>> values = [] {
    const std::vector<std::string_view> effect = {"Scatter Only", "Scatter & Transmit", "Scatter & Transmit Intensity"};
    std::map<std::string, std::vector<std::string_view>> v;
    v[norm("Base Mixing")]            = {"PBR Metallicity/Roughness", "PBR Specular/Glossiness", "Weighted"};
    v[norm("Luminance Units")]        = {"cd/m^2", "kcd/m^2", "cd/ft^2", "cd/cm^2", "lm", "W"};
    v[norm("Top Coat Layering Mode")] = {"Reflectivity", "Weighted", "Fresnel", "Custom Curve"};
    v[norm("Top Coat Bump Mode")]     = {"Height Map", "Normal Map"};
    v[norm("SSS Mode")]               = {"Mono", "Chromatic"};
    v[norm("BSDF Type")] = {"Standard Glossy", "Beckmann Smith", "GGX Smith", "Beckmann VCavities", "GGX VCavities", "Ward Geisler Moroder"};
    v[norm("Render Priority")] = {"Lowest", "Low", "Below Normal", "Normal", "Above Normal", "High", "Highest"};
    for(const char* name : {"Base Color Effect", "Diffuse Overlay Color Effect", "Glossy Color Effect", "Metallic Flakes Color Effect", "Top Coat Color Effect"})
      v[norm(name)] = effect;
    return v;
  }();
  return values;
}

// ---- reading the script's literals

void skipSpace(std::string_view s, size_t& i)
{
  while(i < s.size())
  {
    if(std::isspace(static_cast<unsigned char>(s[i])))
      i++;
    else if(s.compare(i, 2, "//") == 0)
      i = std::min(s.find('\n', i), s.size());
    else if(s.compare(i, 2, "/*") == 0)
    {
      const size_t end = s.find("*/", i + 2);
      i                = end == std::string_view::npos ? s.size() : end + 2;
    }
    else
      return;
  }
}

int hexDigit(char c)
{
  return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

void appendUtf8(std::string& out, unsigned cp)
{
  if(cp < 0x80)
    out += char(cp);
  else if(cp < 0x800)
  {
    out += char(0xC0 | (cp >> 6));
    out += char(0x80 | (cp & 0x3F));
  }
  else
  {
    out += char(0xE0 | (cp >> 12));
    out += char(0x80 | ((cp >> 6) & 0x3F));
    out += char(0x80 | (cp & 0x3F));
  }
}

// A string literal whose quote is at `i`; `i` after it. None when it doesn't close on its line.
std::optional<std::string> stringLiteral(std::string_view s, size_t& i)
{
  const char  quote = s[i];
  std::string out;
  for(size_t k = i + 1; k < s.size(); k++)
  {
    const char c = s[k];
    if(c == quote)
    {
      i = k + 1;
      return out;
    }
    if(c == '\n')
      return std::nullopt;
    if(c != '\\' || k + 1 >= s.size())
    {
      out += c;
      continue;
    }
    const char e = s[++k];
    switch(e)
    {
      case 'n':
        out += '\n';
        break;
      case 't':
        out += '\t';
        break;
      case 'r':
        out += '\r';
        break;
      case 'u':
      {
        unsigned cp = 0;
        for(int d = 0; d < 4; d++)
        {
          const int h = k + 1 < s.size() ? hexDigit(s[k + 1]) : -1;
          if(h < 0)
            return std::nullopt;
          cp = cp * 16 + unsigned(h);
          k++;
        }
        appendUtf8(out, cp);
        break;
      }
      default:
        out += e;  // \" \' \\ \/
    }
  }
  return std::nullopt;
}

// A literal at `i`: a string, a number, true or false, or an array of them; `i` after it. None for
// anything else (a name, an expression), `i` left where it was.
std::optional<Json> literal(std::string_view s, size_t& i)
{
  skipSpace(s, i);
  if(i >= s.size())
    return std::nullopt;
  const char c = s[i];
  if(c == '"' || c == '\'')
  {
    size_t k = i;
    auto   v = stringLiteral(s, k);
    if(!v)
      return std::nullopt;
    i = k;
    return Json(*v);
  }
  if(c == '[')
  {
    size_t k   = i + 1;
    Json   arr = Json::array();
    skipSpace(s, k);
    if(k < s.size() && s[k] == ']')
    {
      i = k + 1;
      return arr;
    }
    for(;;)
    {
      auto v = literal(s, k);
      if(!v)
        return std::nullopt;
      arr.push_back(std::move(*v));
      skipSpace(s, k);
      if(k < s.size() && s[k] == ',')
        k++;
      else if(k < s.size() && s[k] == ']')
      {
        i = k + 1;
        return arr;
      }
      else
        return std::nullopt;
    }
  }
  for(const auto& [word, value] : {std::pair{std::string_view("true"), true}, std::pair{std::string_view("false"), false}})
    if(s.compare(i, word.size(), word) == 0 && (i + word.size() >= s.size() || !(std::isalnum(static_cast<unsigned char>(s[i + word.size()])) || s[i + word.size()] == '_')))
    {
      i += word.size();
      return Json(value);
    }
  if(c == '-' || c == '+' || c == '.' || std::isdigit(static_cast<unsigned char>(c)))
  {
    size_t k = i;
    while(k < s.size() && (std::isdigit(static_cast<unsigned char>(s[k])) || std::strchr("+-.eE", s[k])))
      k++;
    const std::string token(s.substr(i, k - i));
    char*             end = nullptr;
    const double      v   = std::strtod(token.c_str(), &end);
    if(end != token.c_str() + token.size())
      return std::nullopt;
    i = k;
    return Json(v);
  }
  return std::nullopt;
}

// The arguments of a call whose '(' is at `i`, when every one is a literal; `i` after the ')'.
std::optional<std::vector<Json>> arguments(std::string_view s, size_t& i)
{
  size_t k = i + 1;
  std::vector<Json> out;
  skipSpace(s, k);
  if(k < s.size() && s[k] == ')')
  {
    i = k + 1;
    return out;
  }
  for(;;)
  {
    auto v = literal(s, k);
    if(!v)
      return std::nullopt;
    out.push_back(std::move(*v));
    skipSpace(s, k);
    if(k < s.size() && s[k] == ',')
      k++;
    else if(k < s.size() && s[k] == ')')
    {
      i = k + 1;
      return out;
    }
    else
      return std::nullopt;
  }
}

// ---- what the calls set

struct Property
{
  std::string         name;
  std::optional<Json> value;
  std::optional<Json> map;  // a path, "" for none
};

// a URL's part as the .duf files write it
std::string encode(std::string_view s, bool path)
{
  static const char* hex = "0123456789ABCDEF";
  std::string        out;
  for(unsigned char c : s)
    if(std::isalnum(c) || std::strchr("-._~()&',;=+!$*", c) || (path && c == '/'))
      out += char(c);
    else
    {
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 15];
    }
  return out;
}

// The value of a setNumericProperty-style call's value array: a number, a switch, or an option's
// index by its text. None for an option the scene names (a UV set).
std::optional<Json> numericValue(const std::string& property, const Json& values)
{
  if(!values.is_array() || values.empty())
    return std::nullopt;
  if(values.size() > 1)
    return std::all_of(values.begin(), values.end(), [](const Json& v) { return v.is_number(); }) ? std::optional<Json>(values) : std::nullopt;
  const Json& v = values[0];
  if(v.is_number() || v.is_boolean())
    return v;
  if(!v.is_string())
    return std::nullopt;
  const auto options = enumValues().find(norm(property));
  if(options == enumValues().end())
    return std::nullopt;
  const std::string text = v.get<std::string>();
  for(size_t k = 0; k < options->second.size(); k++)
    if(options->second[k] == text)
      return Json(int(k));
  return std::nullopt;
}

// The property a DzPresetHelper call sets, from its literal arguments; none for a call that sets
// none (the map setting, the target element).
std::optional<Property> propertyOf(std::string_view method, const std::vector<Json>& args)
{
  if(args.empty() || !args[0].is_string())
    return std::nullopt;
  Property p;
  p.name = args[0].get<std::string>();
  // the map follows the value; none given, the map goes (DAZ's Replace)
  const auto mapAt = [&](size_t k) { return Json(k < args.size() && args[k].is_string() ? args[k].get<std::string>() : std::string()); };
  if(method == "setColorProperty")
  {
    if(args.size() < 2 || !args[1].is_array() || args[1].size() < 3)
      return std::nullopt;
    Json rgb = Json::array();
    for(size_t c = 0; c < 3; c++)
    {
      if(!args[1][c].is_number())
        return std::nullopt;
      rgb.push_back(args[1][c].get<double>() / 255.0);
    }
    p.value = std::move(rgb);
    p.map   = mapAt(2);
  }
  else if(method == "setNumericProperty" || method == "setNumericPropertyWithAttributes")
  {
    // setNumericPropertyWithAttributes(name, clamped, min, max, [value], map)
    const size_t at = method == "setNumericProperty" ? 1 : 4;
    if(args.size() <= at)
      return std::nullopt;
    p.value = numericValue(p.name, args[at]);
    // an option's text says it is no map's
    if(args[at].is_array() && !args[at].empty() && args[at][0].is_number())
      p.map = mapAt(at + 1);
  }
  else if(method == "setPropertyWithString")
  {
    if(args.size() < 2 || !args[1].is_string())
      return std::nullopt;
    p.value = args[1];
  }
  else
    return std::nullopt;
  if(!p.value && !p.map)
    return std::nullopt;
  return p;
}

bool identStart(char c)
{
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}

bool identChar(char c)
{
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' || c == '.';
}

}  // namespace

bool isScriptMaterialsPreset(std::string_view text)
{
  // the saver's first line: "// DAZ Studio version 4.x filetype Materials Preset"
  const std::string_view first = text.substr(0, text.find('\n'));
  return first.find("DAZ Studio") != std::string_view::npos && first.find(kMarker) != std::string_view::npos;
}

std::optional<Json> scriptMaterialsPreset(std::string_view s)
{
  if(!isScriptMaterialsPreset(s))
    return std::nullopt;
  size_t i = s.find("prototype.setMaterialProperties");
  if(i == std::string_view::npos)
    return std::nullopt;
  i = s.find("switch", i);
  if(i == std::string_view::npos)
    return std::nullopt;
  i = s.find('{', i);
  if(i == std::string_view::npos)
    return std::nullopt;
  i++;
  // each surface's properties, in the order the script names them
  std::vector<std::pair<std::string, std::vector<Property>>> surfaces;
  const auto                                                 surface = [&](const std::string& name) -> std::vector<Property>& {
    for(auto& [n, props] : surfaces)
      if(n == name)
        return props;
    return surfaces.emplace_back(name, std::vector<Property>{}).second;
  };
  // the case labels the statements run under: as JavaScript falls through, every label since the
  // last break
  std::vector<std::string> labels;
  int                      depth = 1;
  while(i < s.size() && depth > 0)
  {
    const char c = s[i];
    if(c == '"' || c == '\'')
    {
      if(!stringLiteral(s, i))
        i++;
      continue;
    }
    if(s.compare(i, 2, "//") == 0 || s.compare(i, 2, "/*") == 0)
    {
      skipSpace(s, i);
      continue;
    }
    if(c == '{' || c == '}')
    {
      depth += c == '{' ? 1 : -1;
      i++;
      continue;
    }
    if(!identStart(c) || (i > 0 && identChar(s[i - 1])))
    {
      i++;
      continue;
    }
    size_t k = i;
    while(k < s.size() && identChar(s[k]))
      k++;
    const std::string_view word = s.substr(i, k - i);
    i                           = k;
    if(word == "case")
    {
      skipSpace(s, i);
      if(i < s.size() && (s[i] == '"' || s[i] == '\''))
        if(auto label = stringLiteral(s, i))
          labels.push_back(std::move(*label));
    }
    else if(word == "break")
      labels.clear();
    else if(word.rfind("g_oPresetHelper.", 0) == 0)
    {
      skipSpace(s, i);
      if(i >= s.size() || s[i] != '(')
        continue;
      const size_t open = i;
      const auto   args = arguments(s, i);
      if(!args)
      {
        i = open + 1;
        continue;
      }
      if(auto p = propertyOf(word.substr(16), *args))
        for(const std::string& label : labels)
        {
          std::vector<Property>& props = surface(label);
          // a property set twice keeps the later
          std::erase_if(props, [&](const Property& q) { return norm(q.name) == norm(p->name); });
          props.push_back(*p);
        }
    }
  }
  Json animations = Json::array();
  for(const auto& [name, props] : surfaces)
    for(const Property& p : props)
    {
      // the .duf files' id of Iray Uber's base colour
      const std::string id   = norm(p.name) == "diffusecolor" ? std::string("diffuse") : p.name;
      const std::string base = "name://@selection#materials/" + encode(name, false) + ":?" + encode(id, false) + "/";
      if(p.value)
        animations.push_back({{"url", base + "value"}, {"keys", Json::array({Json::array({0, *p.value})})}});
      if(p.map)
        animations.push_back({{"url", base + "image_file"}, {"keys", Json::array({Json::array({0, encode(p.map->get<std::string>(), true)})})}});
    }
  if(animations.empty())
    return std::nullopt;
  return Json{{"asset_info", {{"type", "preset_material"}}}, {"scene", {{"animations", std::move(animations)}}}};
}

}  // namespace fv::dson
