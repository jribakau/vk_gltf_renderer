#include "fv_world_features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <tinygltf/json.hpp>  // nlohmann::json

namespace fv::world {

namespace {

constexpr double kPi = 3.14159265358979323846;

double angleBetween(const glm::dvec3& a, const glm::dvec3& b)
{
  // (atan2 of the cross and the dot: an acos of the dot loses the small angles)
  return std::atan2(glm::length(glm::cross(a, b)), glm::dot(a, b));
}

double smoothstep(double x)
{
  const double t = std::clamp(x, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// A peak's shape by how far out a point is, 0 at its top to 1 at its foot: a cone that steepens toward
// the top and eases into the land, and inside its crater's rim a bowl a quarter of the peak deep at
// the most
double peakShape(double away, double crater)
{
  const auto cone = [](double r) {
    const double t = std::clamp(1.0 - r, 0.0, 1.0);
    return t * t * (1.6 - 0.6 * t);
  };
  if(crater <= 0.0 || away >= crater)
    return cone(away);
  const double inside = 1.0 - (away / crater) * (away / crater);
  return cone(crater) - std::min(0.25, 1.5 * crater) * inside;
}

}  // namespace

glm::dvec3 directionOf(double latitude, double longitude)
{
  const double lat = latitude * (kPi / 180.0), lon = longitude * (kPi / 180.0);
  return {std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
}

double distanceFrom(const Feature& f, const glm::dvec3& d, double radius)
{
  const glm::dvec3 across = glm::cross(f.from, f.to);
  const double     span   = glm::length(across);
  if(span < 1e-9)
    return angleBetween(d, f.from) * radius;  // a point (or two ends opposite, which name no arc)
  // on the arc where the point's foot on the great circle lies between the ends; past them, the nearer end
  const glm::dvec3 n = across / span;
  if(glm::dot(glm::cross(f.from, d), n) >= 0.0 && glm::dot(glm::cross(d, f.to), n) >= 0.0)
    return std::asin(std::clamp(std::abs(glm::dot(d, n)), 0.0, 1.0)) * radius;
  return std::min(angleBetween(d, f.from), angleBetween(d, f.to)) * radius;
}

double reach(const Feature& f, const glm::dvec3& direction, double radius)
{
  if(f.width <= 0.0)
    return 0.0;
  return smoothstep(1.0 - distanceFrom(f, direction, radius) / f.width);
}

PlacedLand placedAt(const Features& features, const glm::dvec3& direction, double radius)
{
  PlacedLand p;
  for(const Feature& f : features.list)
  {
    if(f.width <= 0.0)
      continue;
    const double away = distanceFrom(f, direction, radius) / f.width;
    if(away >= 1.0)
      continue;
    const double r = smoothstep(1.0 - away);
    switch(f.kind)
    {
      case FeatureKind::eLand:
        p.continent += float(f.amount * r);
        break;
      case FeatureKind::eSea:
        p.continent -= float(f.amount * r);
        break;
      case FeatureKind::eRange:
        // full over the middle two fifths of its width, so a range has a body and not one crest
        p.range = std::max(p.range, float(f.amount * smoothstep((1.0 - away) / 0.6)));
        break;
      case FeatureKind::ePeak:
        p.elevation += float(f.height * peakShape(away, std::clamp(f.crater, 0.0, 0.9)));
        break;
      case FeatureKind::eBasin:
        p.elevation -= float(f.height * r);
        break;
      case FeatureKind::ePlain:
        p.level = std::max(p.level, float(std::clamp(f.amount, 0.0, 1.0) * smoothstep((1.0 - away) / 0.5)));
        break;
    }
  }
  return p;
}

// ---- where people lived --------------------------------------------------------------------------------

double regionReach(const PlacesRegion& region, const glm::dvec3& direction, double radius)
{
  if(region.shape.width <= 0.0)
    return 0.0;
  return smoothstep((1.0 - distanceFrom(region.shape, direction, radius) / region.shape.width) / kRegionEdge);
}

double settledAmount(const Settlement& settlement, const glm::dvec3& direction, double radius)
{
  double amount = 1.0;
  for(const PlacesRegion& r : settlement.regions)
    if(const double in = regionReach(r, direction, radius); in > 0.0)
      amount *= 1.0 + (std::max(r.amount, 0.0) - 1.0) * in;
  return amount;
}

double settledKind(const Settlement& settlement, const std::string& kind, const glm::dvec3& direction, double radius)
{
  double share = 1.0;
  for(const PlacesRegion& r : settlement.regions)
  {
    if(r.kinds.empty())
      continue;
    const double in = regionReach(r, direction, radius);
    if(in <= 0.0)
      continue;
    double named = 0.0;
    for(const auto& [name, much] : r.kinds)
      if(name == kind)
        named = std::max(much, 0.0);
    share *= 1.0 + (named - 1.0) * in;
  }
  return share;
}

// ---- a planet as a file --------------------------------------------------------------------------------

namespace {

using Json = nlohmann::json;

[[noreturn]] void wrong(const std::string& where, const std::string& what)
{
  throw std::invalid_argument("planet file: " + where + ": " + what);
}

double number(const Json& j, const std::string& where, double from, double to)
{
  if(!j.is_number())
    wrong(where, "a number is wanted");
  const double v = j.get<double>();
  if(!std::isfinite(v) || v < from || v > to)
    wrong(where, "out of range (" + std::to_string(from) + " to " + std::to_string(to) + ")");
  return v;
}

glm::dvec3 place(const Json& j, const std::string& where)
{
  if(!j.is_array() || j.size() != 2)
    wrong(where, "a place is [latitude, longitude] in degrees");
  return directionOf(number(j[0], where + " latitude", -90.0, 90.0), number(j[1], where + " longitude", -360.0, 360.0));
}

FeatureKind kindOf(const Json& j, const std::string& where)
{
  static const std::pair<const char*, FeatureKind> kKinds[] = {{"land", FeatureKind::eLand},   {"sea", FeatureKind::eSea},
                                                               {"range", FeatureKind::eRange}, {"peak", FeatureKind::ePeak},
                                                               {"basin", FeatureKind::eBasin}, {"plain", FeatureKind::ePlain}};
  if(j.is_string())
    for(const auto& [name, kind] : kKinds)
      if(j.get<std::string>() == name)
        return kind;
  wrong(where, "a kind is one of land, sea, range, peak, basin, plain");
}

// Its line, as a feature's: `at`, or `from` and `to`; and its width
void shapeOf(const Json& j, const std::string& where, Feature& f)
{
  if(j.contains("at"))
  {
    if(j.contains("from") || j.contains("to"))
      wrong(where, "`at` names a point: no `from` or `to` beside it");
    f.from = f.to = place(j["at"], where + ".at");
  }
  else if(j.contains("from") && j.contains("to"))
  {
    f.from = place(j["from"], where + ".from");
    f.to   = place(j["to"], where + ".to");
    if(glm::dot(f.from, f.to) < -0.999)
      wrong(where, "its ends are opposite each other: no one arc joins them");
  }
  else
    wrong(where, "wants `at`, or `from` and `to`");
  if(!j.contains("width"))
    wrong(where, "no width");
  f.width = number(j["width"], where + ".width", 0.05, 1000.0) * 1000.0;
}

std::string nameOf(const Json& j, const std::string& where)
{
  if(!j.is_string() || j.get<std::string>().empty())
    wrong(where, "a name is wanted");
  return j.get<std::string>();
}

Feature feature(const Json& j, const std::string& where)
{
  if(!j.is_object())
    wrong(where, "a feature is an object");
  Feature f;
  if(!j.contains("kind"))
    wrong(where, "no kind");
  f.kind = kindOf(j["kind"], where + ".kind");
  // a point (`at`) or an arc (`from`, `to`), and its width
  shapeOf(j, where, f);
  for(const auto& [key, value] : j.items())
  {
    if(key == "kind" || key == "at" || key == "from" || key == "to" || key == "width")
      continue;
    if(key == "name")
      f.name = nameOf(value, where + ".name");
    else if(key == "amount")
      f.amount = number(value, where + ".amount", 0.0, 10.0);
    else if(key == "height" && f.kind == FeatureKind::ePeak)
      f.height = number(value, where + ".height", 0.0, 20000.0);
    else if(key == "depth" && f.kind == FeatureKind::eBasin)
      f.height = number(value, where + ".depth", 0.0, 20000.0);
    else if(key == "crater" && f.kind == FeatureKind::ePeak)
      f.crater = number(value, where + ".crater", 0.0, 0.9);
    else
      wrong(where, "`" + key + "` is nothing a " + j["kind"].get<std::string>() + " has");
  }
  if(f.kind == FeatureKind::ePeak && !j.contains("height"))
    wrong(where, "a peak wants a height");
  if(f.kind == FeatureKind::eBasin && !j.contains("depth"))
    wrong(where, "a basin wants a depth");
  return f;
}

// One of the file's `places`: a region, or a place by name
void settlementOf(const Json& j, const std::string& where, Settlement& s)
{
  if(!j.is_object())
    wrong(where, "an entry of places is an object");
  if(!j.contains("kind") || !j["kind"].is_string())
    wrong(where, "no kind: region or place");
  const std::string kind = j["kind"].get<std::string>();
  if(kind == "region")
  {
    PlacesRegion r;
    shapeOf(j, where, r.shape);
    for(const auto& [key, value] : j.items())
    {
      if(key == "kind" || key == "at" || key == "from" || key == "to" || key == "width")
        continue;
      if(key == "name")
        r.name = nameOf(value, where + ".name");
      else if(key == "amount")
        r.amount = number(value, where + ".amount", 0.0, 10.0);
      else if(key == "kinds")
      {
        if(!value.is_object() || value.empty())
          wrong(where + ".kinds", "the kinds that stand there and how much of each, as { \"farm\": 2 }");
        for(const auto& [name, much] : value.items())
          r.kinds.emplace_back(name, number(much, where + ".kinds." + name, 0.0, 100.0));
      }
      else
        wrong(where, "`" + key + "` is nothing a region has");
    }
    s.regions.push_back(std::move(r));
  }
  else if(kind == "place")
  {
    NamedPlace p;
    for(const char* wanted : {"name", "at", "place"})
      if(!j.contains(wanted))
        wrong(where, std::string("a place wants `") + wanted + "`");
    for(const auto& [key, value] : j.items())
    {
      if(key == "kind")
        continue;
      if(key == "name")
        p.name = nameOf(value, where + ".name");
      else if(key == "at")
        p.at = place(value, where + ".at");
      else if(key == "place")
        p.kind = nameOf(value, where + ".place");
      else
        wrong(where, "`" + key + "` is nothing a place has");
    }
    s.named.push_back(std::move(p));
  }
  else
    wrong(where + ".kind", "a kind of places is one of region, place");
}

}  // namespace

PlanetFile parsePlanet(const std::string& text)
{
  Json j;
  try
  {
    j = Json::parse(text);
  }
  catch(const Json::parse_error& e)
  {
    wrong("the file", e.what());
  }
  if(!j.is_object())
    wrong("the file", "an object is wanted");
  PlanetFile p;
  for(const auto& [key, value] : j.items())
  {
    if(key == "name")
    {
      if(!value.is_string())
        wrong("name", "a text is wanted");
      p.name = value.get<std::string>();
    }
    else if(key == "seed")
    {
      if(!value.is_number_integer())
        wrong("seed", "a whole number is wanted");
      p.seed = value.get<int>();
    }
    else if(key == "radius")
      p.radius = number(value, "radius", 2.0, 100.0);
    else if(key == "sea")
      p.sea = number(value, "sea", 0.0, 0.9);
    else if(key == "relief")
      p.relief = number(value, "relief", 0.0, 2.0);
    else if(key == "wild")
      p.features.wild = number(value, "wild", 0.0, 1.0);
    else if(key == "site")
      p.features.site = place(value, "site");
    else if(key == "features")
    {
      if(!value.is_array())
        wrong("features", "a list is wanted");
      for(size_t i = 0; i < value.size(); i++)
        p.features.list.push_back(feature(value[i], "features[" + std::to_string(i) + "]"));
    }
    else if(key == "places")
    {
      if(!value.is_array())
        wrong("places", "a list is wanted");
      for(size_t i = 0; i < value.size(); i++)
        settlementOf(value[i], "places[" + std::to_string(i) + "]", p.settlement);
    }
    else
      wrong(key, "nothing a planet has");
  }
  return p;
}

}  // namespace fv::world
