/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fv_lighting.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>

#include <glm/gtx/quaternion.hpp>

namespace fv::lighting {

namespace {

constexpr double kPi = 3.14159265358979323846;
// The distance the renderer stands a directional light's disc at, in km (gltf_scene_vk.cpp).
constexpr double kSunDistanceKm = 149597870.0;

double radians(double deg)
{
  return deg * kPi / 180.0;
}

[[noreturn]] void reject(const std::string& what)
{
  throw std::invalid_argument("lighting: " + what);
}

double number(const Json& v, const std::string& key, double lo, double hi)
{
  if(!v.is_number())
    reject(key + " must be a number");
  const double x = v.get<double>();
  if(!std::isfinite(x))
    reject(key + " must be finite");
  return std::clamp(x, lo, hi);
}

bool boolean(const Json& v, const std::string& key)
{
  if(!v.is_boolean())
    reject(key + " must be true or false");
  return v.get<bool>();
}

std::string string(const Json& v, const std::string& key)
{
  if(!v.is_string())
    reject(key + " must be a string");
  return v.get<std::string>();
}

std::string oneOf(const Json& v, const std::string& key, std::initializer_list<const char*> values)
{
  const std::string s = string(v, key);
  for(const char* allowed : values)
    if(s == allowed)
      return s;
  std::string list;
  for(const char* allowed : values)
    list += (list.empty() ? "" : ", ") + std::string(allowed);
  reject(key + " is one of " + list + ", not \"" + s + "\"");
}

std::string color(const Json& v, const std::string& key)
{
  const std::string s = string(v, key);
  try
  {
    linearColor(s);
  }
  catch(const std::invalid_argument&)
  {
    reject(key + " must be a colour like \"#a0b1c2\", not \"" + s + "\"");
  }
  return s;
}

const Json& section(const Json& patch, const char* name)
{
  const Json& v = patch[name];
  if(!v.is_object())
    reject(std::string(name) + " must be an object or null");
  return v;
}

// Each field: null resets it, anything else is checked and clamped. `T` is the field's type.
template <class T, class Parse>
void field(T& dst, const T& def, const Json& v, Parse parse)
{
  dst = v.is_null() ? def : T(parse(v));
}

void patchEnv(Env& e, const Json& p)
{
  const Env d;
  for(auto it = p.begin(); it != p.end(); ++it)
  {
    const std::string& k   = it.key();
    const Json&        v   = it.value();
    const std::string  key = "env." + k;
    if(k == "source")
      field(e.source, d.source, v, [&](const Json& x) { return oneOf(x, key, {"studio", "hdr", "sky", "plain", "none"}); });
    else if(k == "path")
      field(e.path, d.path, v, [&](const Json& x) { return string(x, key); });
    else if(k == "intensity")
      field(e.intensity, d.intensity, v, [&](const Json& x) { return number(x, key, 0.0, 100.0); });
    else if(k == "rotation")
      field(e.rotation, d.rotation, v, [&](const Json& x) { return wrapDegrees(number(x, key, -1e6, 1e6)); });
    else if(k == "background")
      field(e.background, d.background, v, [&](const Json& x) { return oneOf(x, key, {"environment", "blurred", "color"}); });
    else if(k == "blur")
      field(e.blur, d.blur, v, [&](const Json& x) { return number(x, key, 0.0, 1.0); });
    else if(k == "color")
      field(e.color, d.color, v, [&](const Json& x) { return color(x, key); });
    else if(k == "plainColor")
      field(e.plainColor, d.plainColor, v, [&](const Json& x) { return color(x, key); });
    else if(k == "skyPreset")
      field(e.skyPreset, d.skyPreset, v, [&](const Json& x) { return int(std::lround(number(x, key, 0.0, 2.0))); });
    else if(k == "name" || k == "size" || k == "loading")
      continue;  // read-only, the host's
    else
      reject("unknown key " + key);
  }
}

void patchSun(Sun& s, const Json& p)
{
  const Sun d;
  for(auto it = p.begin(); it != p.end(); ++it)
  {
    const std::string& k   = it.key();
    const Json&        v   = it.value();
    const std::string  key = "sun." + k;
    if(k == "enabled")
      field(s.enabled, d.enabled, v, [&](const Json& x) { return boolean(x, key); });
    else if(k == "intensity")
      field(s.intensity, d.intensity, v, [&](const Json& x) { return number(x, key, 0.0, 100.0); });
    else if(k == "azimuth")
      field(s.azimuth, d.azimuth, v, [&](const Json& x) { return wrapDegrees(number(x, key, -1e6, 1e6)); });
    else if(k == "elevation")
      field(s.elevation, d.elevation, v, [&](const Json& x) { return number(x, key, -90.0, 90.0); });
    else if(k == "color")
      field(s.color, d.color, v, [&](const Json& x) { return color(x, key); });
    else if(k == "softness")
      field(s.softness, d.softness, v, [&](const Json& x) { return number(x, key, 0.0, 20.0); });
    else if(k == "shadows")
      field(s.shadows, d.shadows, v, [&](const Json& x) { return boolean(x, key); });
    else
      reject("unknown key " + key);
  }
}

void patchShadow(Shadow& s, const Json& p)
{
  const Shadow d;
  for(auto it = p.begin(); it != p.end(); ++it)
  {
    const std::string& k   = it.key();
    const Json&        v   = it.value();
    const std::string  key = "shadow." + k;
    if(k == "size")
      field(s.size, d.size, v, [&](const Json& x) {
        // a power of two, as the map is allocated
        return int(std::exp2(std::round(std::log2(number(x, key, 1024.0, 8192.0)))));
      });
    else if(k == "contactHardening")
      field(s.contactHardening, d.contactHardening, v, [&](const Json& x) { return boolean(x, key); });
    else if(k == "bias")
      field(s.bias, d.bias, v, [&](const Json& x) { return number(x, key, 0.0, 50.0); });
    else if(k == "normalBias")
      field(s.normalBias, d.normalBias, v, [&](const Json& x) { return number(x, key, 0.0, 50.0); });
    else
      reject("unknown key " + key);
  }
}

void patchGround(Ground& g, const Json& p)
{
  const Ground d;
  for(auto it = p.begin(); it != p.end(); ++it)
  {
    const std::string& k   = it.key();
    const Json&        v   = it.value();
    const std::string  key = "ground." + k;
    if(k == "mode")
      field(g.mode, d.mode, v, [&](const Json& x) { return oneOf(x, key, {"floor", "catcher", "off"}); });
    else if(k == "height")
    {
      if(v.is_null() || (v.is_string() && v.get<std::string>() == "auto"))
      {
        g.autoHeight = true;
        g.height     = d.height;
      }
      else if(v.is_number())
      {
        g.autoHeight = false;
        g.height     = number(v, key, -1e4, 1e4);
      }
      else
        reject(key + " must be \"auto\" or a number");
    }
    else if(k == "color")
      field(g.color, d.color, v, [&](const Json& x) { return color(x, key); });
    else if(k == "roughness")
      field(g.roughness, d.roughness, v, [&](const Json& x) { return number(x, key, 0.0, 1.0); });
    else if(k == "darkness")
      field(g.darkness, d.darkness, v, [&](const Json& x) { return number(x, key, 0.0, 1.0); });
    else if(k == "fade")
      field(g.fade, d.fade, v, [&](const Json& x) { return number(x, key, 0.0, 100.0); });
    else
      reject("unknown key " + key);
  }
}

void patchTonemap(Tonemap& t, const Json& p)
{
  const Tonemap d;
  for(auto it = p.begin(); it != p.end(); ++it)
  {
    const std::string& k   = it.key();
    const Json&        v   = it.value();
    const std::string  key = "tonemap." + k;
    if(k == "method")
      field(t.method, d.method, v, [&](const Json& x) { return oneOf(x, key, {"aces", "agx", "neutral", "filmic"}); });
    else if(k == "exposure")
      field(t.exposure, d.exposure, v, [&](const Json& x) { return number(x, key, 0.01, 100.0); });
    else if(k == "contrast")
      field(t.contrast, d.contrast, v, [&](const Json& x) { return number(x, key, 0.0, 2.0); });
    else if(k == "saturation")
      field(t.saturation, d.saturation, v, [&](const Json& x) { return number(x, key, 0.0, 2.0); });
    else if(k == "vignette")
      field(t.vignette, d.vignette, v, [&](const Json& x) { return number(x, key, -1.0, 1.0); });
    else if(k == "temperature")
      field(t.temperature, d.temperature, v, [&](const Json& x) { return number(x, key, 2000.0, 15000.0); });
    else if(k == "tint")
      field(t.tint, d.tint, v, [&](const Json& x) { return number(x, key, -0.03, 0.03); });
    else
      reject("unknown key " + key);
  }
}

}  // namespace

Json toJson(const State& s)
{
  const Env&     e = s.env;
  const Sun&     u = s.sun;
  const Shadow&  h = s.shadow;
  const Ground&  g = s.ground;
  const Tonemap& t = s.tonemap;
  return {
      {"env",
       {{"source", e.source},
        {"path", e.path},
        {"intensity", e.intensity},
        {"rotation", e.rotation},
        {"background", e.background},
        {"blur", e.blur},
        {"color", e.color},
        {"plainColor", e.plainColor},
        {"skyPreset", e.skyPreset}}},
      {"sun",
       {{"enabled", u.enabled},
        {"intensity", u.intensity},
        {"azimuth", u.azimuth},
        {"elevation", u.elevation},
        {"color", u.color},
        {"softness", u.softness},
        {"shadows", u.shadows}}},
      {"shadow", {{"size", h.size}, {"contactHardening", h.contactHardening}, {"bias", h.bias}, {"normalBias", h.normalBias}}},
      {"ground",
       {{"mode", g.mode},
        {"height", g.autoHeight ? Json("auto") : Json(g.height)},
        {"color", g.color},
        {"roughness", g.roughness},
        {"darkness", g.darkness},
        {"fade", g.fade}}},
      {"tonemap",
       {{"method", t.method},
        {"exposure", t.exposure},
        {"contrast", t.contrast},
        {"saturation", t.saturation},
        {"vignette", t.vignette},
        {"temperature", t.temperature},
        {"tint", t.tint}}},
  };
}

State applyPatch(const State& s, const Json& patch)
{
  if(!patch.is_object())
    reject("the patch must be an object");
  State out = s;
  for(auto it = patch.begin(); it != patch.end(); ++it)
  {
    const std::string& k = it.key();
    const bool         reset = it.value().is_null();
    if(k == "env")
      reset ? void(out.env = Env{}) : patchEnv(out.env, section(patch, "env"));
    else if(k == "sun")
      reset ? void(out.sun = Sun{}) : patchSun(out.sun, section(patch, "sun"));
    else if(k == "shadow")
      reset ? void(out.shadow = Shadow{}) : patchShadow(out.shadow, section(patch, "shadow"));
    else if(k == "ground")
      reset ? void(out.ground = Ground{}) : patchGround(out.ground, section(patch, "ground"));
    else if(k == "tonemap")
      reset ? void(out.tonemap = Tonemap{}) : patchTonemap(out.tonemap, section(patch, "tonemap"));
    else
      reject("unknown key " + k);
  }
  return out;
}

Json mergeDiff(const Json& from, const Json& to)
{
  if(!from.is_object() || !to.is_object())
    return to;
  Json out = Json::object();
  for(auto it = to.begin(); it != to.end(); ++it)
  {
    const auto old = from.find(it.key());
    if(old == from.end())
      out[it.key()] = it.value();
    else if(*old != it.value())
      out[it.key()] = (old->is_object() && it.value().is_object()) ? mergeDiff(*old, it.value()) : it.value();
  }
  for(auto it = from.begin(); it != from.end(); ++it)
    if(!to.contains(it.key()))
      out[it.key()] = nullptr;
  return out;
}

glm::vec3 linearColor(const std::string& hex)
{
  if(hex.size() != 7 || hex[0] != '#')
    throw std::invalid_argument("not a #rrggbb colour: " + hex);
  glm::vec3 c{};
  for(int i = 0; i < 3; i++)
  {
    int value = 0;
    for(int j = 0; j < 2; j++)
    {
      const char ch    = hex[1 + i * 2 + j];
      int        digit = 0;
      if(ch >= '0' && ch <= '9')
        digit = ch - '0';
      else if(ch >= 'a' && ch <= 'f')
        digit = ch - 'a' + 10;
      else if(ch >= 'A' && ch <= 'F')
        digit = ch - 'A' + 10;
      else
        throw std::invalid_argument("not a #rrggbb colour: " + hex);
      value = value * 16 + digit;
    }
    const double v = value / 255.0;
    c[i]           = float(v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4));
  }
  return c;
}

glm::vec3 sunDirection(double azimuthDeg, double elevationDeg)
{
  const double az = radians(azimuthDeg), el = radians(elevationDeg);
  return glm::vec3(float(std::cos(el) * std::sin(az)), float(std::sin(el)), float(std::cos(el) * std::cos(az)));
}

glm::quat sunRotation(double azimuthDeg, double elevationDeg)
{
  return glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), glm::normalize(sunDirection(azimuthDeg, elevationDeg)));
}

double sunRadiusKm(double softness)
{
  const double diameterDeg = 0.15 * std::max(0.0, softness);
  return kSunDistanceKm * std::tan(radians(diameterDeg / 2.0));
}

double hdrBlur(double blurriness, uint32_t width, uint32_t height)
{
  if(blurriness <= 0.0)
    return 0.0;
  const double w      = width ? double(width) : 2048.0;
  const double h      = height ? double(height) : w / 2.0;
  const double levels = std::floor(std::log2(std::max(w, h)));
  // one level sharper: matched by eye against three's blur on the built-in studio
  const double level = std::log2((blurriness * w) / (2.0 * kPi)) - 1.0;
  return std::clamp(level / levels, 0.0, 1.0);
}

double tmExposure(const std::string& method, double exposure)
{
  return method == "aces" ? exposure / 0.6 : exposure;
}

int tmMethod(const std::string& method)
{
  if(method == "aces")
    return 3;
  if(method == "agx")
    return 4;
  if(method == "neutral")
    return 5;
  return 0;  // filmic
}

glm::vec4 envRotation(double degrees)
{
  const glm::quat q = glm::angleAxis(float(radians(degrees)), glm::vec3(0.0F, 1.0F, 0.0F));
  return {q.x, q.y, q.z, q.w};
}

double presetRotation(double dazDegrees)
{
  // applyEnvironmentPreset (src/lib/actions.ts): ((r + 180) mod 360) - 180, a wrap with no offset
  return wrapDegrees(dazDegrees);
}

double wrapDegrees(double degrees)
{
  const double x = std::fmod(degrees + 180.0, 360.0);
  return (x < 0.0 ? x + 360.0 : x) - 180.0;
}

std::vector<float> studioImage(uint32_t width, uint32_t height)
{
  std::vector<float> data(size_t(width) * height * 4);
  const auto panel = [](double dx, double dy, double dz, double x, double y, double z, double size) {
    const double d = dx * x + dy * y + dz * z;
    const double t = std::max(0.0, (d - (1.0 - size)) / size);
    return t * t;
  };
  for(uint32_t j = 0; j < height; j++)
  {
    const double v     = (j + 0.5) / height;
    const double theta = v * kPi;  // 0 = up
    const double y     = std::cos(theta);
    const double r     = std::sin(theta);
    for(uint32_t i = 0; i < width; i++)
    {
      const double u   = (i + 0.5) / width;
      const double phi = u * kPi * 2.0;
      const double x   = r * std::cos(phi);
      const double z   = r * std::sin(phi);

      // base gradient: grey walls, darker floor, lighter ceiling
      const double t  = y * 0.5 + 0.5;
      const double tp = std::pow(t, 1.5);
      double       cr = 0.08 + 0.32 * tp;
      double       cg = 0.085 + 0.33 * tp;
      double       cb = 0.095 + 0.35 * tp;
      if(y < 0.0)
      {
        const double f = std::min(1.0, -y * 4.0);
        cr             = cr * (1.0 - f) + 0.05 * f;
        cg             = cg * (1.0 - f) + 0.05 * f;
        cb             = cb * (1.0 - f) + 0.055 * f;
      }
      // overhead softbox; warm key panel front-right, cool rim back-left
      const double top = panel(0, 1, 0, x, y, z, 0.12) * 6.0;
      const double key = panel(0.6, 0.35, 0.72, x, y, z, 0.08) * 9.0;
      const double rim = panel(-0.7, 0.25, -0.67, x, y, z, 0.06) * 5.0;
      float*       px  = data.data() + (size_t(j) * width + i) * 4;
      px[0]            = float(cr + top + key * 1.0 + rim * 0.75);
      px[1]            = float(cg + top + key * 0.93 + rim * 0.85);
      px[2]            = float(cb + top + key * 0.82 + rim * 1.0);
      px[3]            = 1.0F;
    }
  }
  return data;
}

Json rendererParams(const State& s, glm::uvec2 envSize)
{
  const Env&     e = s.env;
  const Tonemap& t = s.tonemap;
  const Ground&  g = s.ground;
  const auto     vec3 = [](const glm::vec3& c) { return Json::array({c.x, c.y, c.z}); };
  const glm::vec4 rot = envRotation(e.rotation);

  int envSystem = 1;  // studio and hdr are both a lat-long image
  if(e.source == "sky")
    envSystem = 0;
  else if(e.source == "none")
    envSystem = 2;
  else if(e.source == "plain")
    envSystem = 3;

  return {
      {"envSystem", envSystem},
      {"hdrIntensity", e.intensity},
      {"envRotation", Json::array({rot.x, rot.y, rot.z, rot.w})},
      {"hdrBlur", e.background == "blurred" ? hdrBlur(e.blur, envSize.x, envSize.y) : 0.0},
      {"useSolidBackground", e.background == "color"},
      {"solidBackgroundColor", vec3(linearColor(e.color))},
      {"plainColor", vec3(linearColor(e.plainColor))},
      {"atmoPreset", e.skyPreset},
      {"tmActive", 1},
      {"tmAutoExposure", 0},
      {"tmMethod", tmMethod(t.method)},
      {"tmExposure", tmExposure(t.method, t.exposure)},
      {"tmContrast", t.contrast},
      {"tmSaturation", t.saturation},
      {"tmVignette", t.vignette},
      {"tmTemperature", t.temperature},
      {"tmTint", t.tint},
      {"rasterShadows", s.sun.enabled && s.sun.shadows},
      {"shadowMapSize", s.shadow.size},
      {"shadowContactHardening", s.shadow.contactHardening},
      {"shadowBias", s.shadow.bias},
      {"shadowNormalBias", s.shadow.normalBias},
      {"useInfinitePlane", g.mode != "off"},
      {"isShadowCatcher", g.mode == "catcher"},
      {"infinitePlaneBaseColor", vec3(linearColor(g.color))},
      {"infinitePlaneRoughness", g.roughness},
      {"infinitePlaneMetallic", 0.0},
      {"shadowCatcherDarkness", g.darkness},
      {"groundFade", g.fade},
  };
}

}  // namespace fv::lighting
