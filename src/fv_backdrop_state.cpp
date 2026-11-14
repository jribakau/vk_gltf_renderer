#include "fv_backdrop_state.hpp"

#include <cmath>

#include "fv_patch.hpp"
#include "sky_world.hpp"

namespace fv::backdrop {

namespace {

constexpr fv::patch::Checks kChecks{"backdrop"};

}  // namespace

Json toJson(const Spec& s)
{
  return {{"kind", s.kind},     {"seed", s.seed}, {"size", s.size},     {"density", s.density}, {"relief", s.relief},
          {"air", s.air},       {"mature", s.mature}, {"wind", s.wind}, {"windFrom", s.windFrom},
          {"radius", s.radius}, {"sea", s.sea}};
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
    else if(k == "state" || k == "plants")
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

}  // namespace fv::backdrop
