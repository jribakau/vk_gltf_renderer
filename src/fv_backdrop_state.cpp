#include "fv_backdrop_state.hpp"

#include <cmath>

#include "fv_lighting.hpp"
#include "fv_patch.hpp"
#include "sky_world.hpp"

namespace fv::backdrop {

namespace {

constexpr fv::patch::Checks kChecks{"backdrop"};

Json toJson(const Body& b)
{
  return {{"radius", b.radius}, {"distance", b.distance}, {"azimuth", b.azimuth}, {"elevation", b.elevation}, {"color", b.color}, {"kind", b.kind}};
}

// A body from its object: what it leaves out is the default's
Body body(const Json& v, const std::string& key)
{
  if(!v.is_object())
    kChecks.reject(key + " must be an object");
  Body b;
  for(auto it = v.begin(); it != v.end(); ++it)
  {
    const std::string& k    = it.key();
    const Json&        x    = it.value();
    const std::string  name = key + "." + k;
    if(x.is_null())
      continue;  // its default
    if(k == "radius")
      // a rock a few hundred metres across to a giant
      b.radius = kChecks.number(x, name, 0.05, 5000.0);
    else if(k == "distance")
      b.distance = kChecks.number(x, name, 1.0, 1.0e7);
    else if(k == "azimuth")
      b.azimuth = fv::patch::wrapDegrees(kChecks.number(x, name, -1e6, 1e6));
    else if(k == "elevation")
      b.elevation = kChecks.number(x, name, -90.0, 90.0);
    else if(k == "color")
    {
      if(!x.is_string())
        kChecks.reject(name + " must be a colour, \"#rrggbb\"");
      b.color = x.get<std::string>();
      try
      {
        fv::lighting::linearColor(b.color);
      }
      catch(const std::exception&)
      {
        kChecks.reject(name + " must be a colour, \"#rrggbb\"");
      }
    }
    else if(k == "kind")
      b.kind = kChecks.oneOf(x, name, {"rock", "gas"});
    else
      kChecks.reject("unknown key " + name);
  }
  // clear of the figures: no nearer than twice its own radius
  b.distance = std::max(b.distance, 2.0 * b.radius);
  return b;
}

}  // namespace

Json toJson(const Spec& s)
{
  Json bodies = Json::array();
  for(const Body& b : s.bodies)
    bodies.push_back(toJson(b));
  return {{"kind", s.kind},     {"seed", s.seed}, {"size", s.size},     {"density", s.density}, {"relief", s.relief},
          {"air", s.air},       {"mature", s.mature}, {"wind", s.wind}, {"windFrom", s.windFrom},
          {"radius", s.radius}, {"sea", s.sea},   {"bodies", bodies}};
}

Spec applyPatch(const Spec& s, const Json& patch)
{
  if(!patch.is_object())
    kChecks.reject("the patch must be an object");
  using fv::patch::field;
  const Spec d;
  Spec       out = s;
  for(auto it = patch.begin(); it != patch.end(); ++it)
  {
    const std::string& k = it.key();
    const Json&        v = it.value();
    if(k == "kind")
      field(out.kind, d.kind, v, [&](const Json& x) { return kChecks.oneOf(x, k, {"none", "meadow", "forest", "planet"}); });
    else if(k == "seed")
      field(out.seed, d.seed, v, [&](const Json& x) { return int(std::lround(kChecks.number(x, k, 0.0, double(1 << 20)))); });
    else if(k == "size")
      // a clearing's worth of grass to a field's: every metre across costs its plants
      field(out.size, d.size, v, [&](const Json& x) { return kChecks.number(x, k, 20.0, 200.0); });
    else if(k == "density")
      field(out.density, d.density, v, [&](const Json& x) { return kChecks.number(x, k, 0.25, 2.0); });
    else if(k == "relief")
      field(out.relief, d.relief, v, [&](const Json& x) { return kChecks.number(x, k, 0.0, 2.0); });
    else if(k == "air")
      field(out.air, d.air, v, [&](const Json& x) { return kChecks.number(x, k, 1.0, 20.0); });
    else if(k == "mature")
      field(out.mature, d.mature, v, [&](const Json& x) { return kChecks.boolean(x, k); });
    else if(k == "wind")
      // still air to a gale's at the treetops
      field(out.wind, d.wind, v, [&](const Json& x) { return kChecks.number(x, k, 0.0, 100.0); });
    else if(k == "windFrom")
      field(out.windFrom, d.windFrom, v, [&](const Json& x) { return fv::patch::wrapDegrees(kChecks.number(x, k, -1e6, 1e6)); });
    else if(k == "radius")
      // a planet to walk round in an afternoon, to one whose horizon is half a kilometre off
      field(out.radius, d.radius, v, [&](const Json& x) { return kChecks.number(x, k, 2.0, 100.0); });
    else if(k == "sea")
      field(out.sea, d.sea, v, [&](const Json& x) { return kChecks.number(x, k, 0.0, 0.9); });
    else if(k == "bodies")
      field(out.bodies, d.bodies, v, [&](const Json& x) {
        if(!x.is_array())
          kChecks.reject(k + " must be an array");
        if(x.size() > kMostBodies)
          kChecks.reject(k + " holds at most " + std::to_string(kMostBodies));
        std::vector<Body> bodies;
        for(size_t i = 0; i < x.size(); i++)
          bodies.push_back(body(x[i], k + "[" + std::to_string(i) + "]"));
        return bodies;
      });
    else if(k == "state" || k == "plants" || k == "latitude")
      continue;  // read-only, the host's
    else
      kChecks.reject("unknown key " + k);
  }
  return out;
}

Json rendererParams(const Spec& s, bool stands, const glm::vec3& landAlbedo, const glm::dvec3& planetCentre)
{
  const bool planet = stands && isPlanet(s);
  const bool land   = stands && !planet;
  Json       params = {
      // the air over a backdrop's land counts several times over (Spec::air); a scene's own, and a
      // planet's, once
      {"atmoAerialPerspectiveScale", land ? s.air : 1.0},
      // and its land gives way to the sky's own ground before it ends; a planet has no end
      {"landFadeFrom", land ? double(kLandFadeFrom) : 0.0},
      {"landFadeTo", land ? double(kLandFadeTo) : 0.0},
      // past it the planet under the physical sky is the land's colour, not average ground
      {"atmoLandAlbedo", stands ? Json::array({landAlbedo.x, landAlbedo.y, landAlbedo.z}) : Json::array({0.0F, 0.0F, 0.0F})},
      // a planet that stands is the sky's planet (sky_world.hpp)
      {"worldRadius", planet ? s.radius : 0.0},
      {"worldAirScale", planet ? double(sky_world::airScale(float(s.radius))) : 1.0},
      {"worldCentre", planet ? Json::array({planetCentre.x, planetCentre.y, planetCentre.z}) : Json::array({0.0, 0.0, 0.0})},
  };
  // a backdrop's ground is the ground: the plane under it isn't drawn
  if(stands)
    params["useInfinitePlane"] = false;
  // what is laid out on a level plane round the origin has no place on a planet yet
  if(planet)
  {
    params["cloudCoverage"]     = 0.0;
    params["cloudMidCoverage"]  = 0.0;
    params["cloudHighCoverage"] = 0.0;
    params["rainRate"]          = 0.0;
    params["hazeVisibility"]    = 0.0;
  }
  return params;
}

fv::world::Spec planetSpec(const Spec& s)
{
  return {.seed = s.seed, .radius = s.radius * 1000.0, .sea = s.sea, .relief = s.relief};
}

std::vector<PlacedBody> placedBodies(const Spec& s, const glm::dvec3& origin)
{
  std::vector<PlacedBody> placed;
  if(!isPlanet(s))
    return placed;
  for(const Body& b : s.bodies)
  {
    PlacedBody p;
    p.centre = origin + glm::dvec3(fv::lighting::sunDirection(b.azimuth, b.elevation)) * (b.distance * 1000.0);
    p.radius = b.radius * 1000.0;
    p.color  = fv::lighting::linearColor(b.color);
    p.kind   = b.kind == "gas" ? 1 : 0;
    placed.push_back(p);
  }
  return placed;
}

}  // namespace fv::backdrop
