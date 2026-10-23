/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The host's Light tab state and its conversions to the renderer's units. The expected numbers are
// the app's own functions evaluated in node (src/lib/viewer/Lighting.ts, src/lib/rtx/export.ts),
// so Real-time in the WebGPU viewer, the RTX bridge and the native renderer agree.

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <glm/gtx/quaternion.hpp>

#include "fv_lighting.hpp"

namespace L = fv::lighting;
using fv::Json;

namespace {

constexpr double kPi = 3.14159265358979323846;

void expectNear(const glm::vec3& a, const glm::vec3& b, float tol)
{
  EXPECT_NEAR(a.x, b.x, tol);
  EXPECT_NEAR(a.y, b.y, tol);
  EXPECT_NEAR(a.z, b.z, tol);
}

}  // namespace

TEST(FvLighting, SunDirectionIsLightingTs)
{
  // (cos el sin az, sin el, cos el cos az)
  expectNear(L::sunDirection(0, 0), {0, 0, 1}, 1e-6F);
  expectNear(L::sunDirection(90, 0), {1, 0, 0}, 1e-6F);
  expectNear(L::sunDirection(0, 90), {0, 1, 0}, 1e-6F);
  const double az = 35 * kPi / 180, el = 50 * kPi / 180;
  expectNear(L::sunDirection(35, 50), {float(std::cos(el) * std::sin(az)), float(std::sin(el)), float(std::cos(el) * std::cos(az))}, 1e-6F);
}

TEST(FvLighting, SunNodeAimsItsRaysAwayFromTheSun)
{
  // glTF's directional light shines down the node's -Z: its +Z must point at the sun
  const std::pair<double, double> aims[] = {{35.0, 50.0}, {-120.0, 5.0}, {180.0, 89.0}, {0.0, -10.0}};
  for(const auto& [az, el] : aims)
  {
    const glm::quat q = L::sunRotation(az, el);
    expectNear(q * glm::vec3(0, 0, 1), L::sunDirection(az, el), 1e-5F);
    EXPECT_NEAR(glm::length(q), 1.0F, 1e-5F);
  }
}

TEST(FvLighting, SunRadiusIsExportTs)
{
  EXPECT_NEAR(L::sunRadiusKm(1), 195823.26575625513, 1e-3);
  EXPECT_NEAR(L::sunRadiusKm(3), 587472.4815886379, 1e-3);
  EXPECT_NEAR(L::sunRadiusKm(12), 2350071.1357064317, 1e-2);
  EXPECT_EQ(L::sunRadiusKm(0), 0.0);
  EXPECT_EQ(L::sunRadiusKm(-2), 0.0);
}

TEST(FvLighting, HdrBlurIsExportTs)
{
  EXPECT_NEAR(L::hdrBlur(0.35, 2048, 1024), 0.5303573361543567, 1e-9);
  EXPECT_NEAR(L::hdrBlur(0.35, 512, 256), 0.42599229974421365, 1e-9);
  EXPECT_NEAR(L::hdrBlur(1, 8192, 4096), 0.7191156823482832, 1e-9);
  EXPECT_NEAR(L::hdrBlur(0.01, 2048, 1024), 0.0640588800684506, 1e-9);
  EXPECT_EQ(L::hdrBlur(0, 2048, 1024), 0.0);
  // no image yet: export.ts's 2048 fallback
  EXPECT_NEAR(L::hdrBlur(0.35, 0, 0), 0.5303573361543567, 1e-9);
}

TEST(FvLighting, AcesExposureIsDividedByPointSix)
{
  EXPECT_NEAR(L::tmExposure("aces", 1.2), 2.0, 1e-12);
  EXPECT_EQ(L::tmExposure("agx", 1.2), 1.2);
  EXPECT_EQ(L::tmExposure("neutral", 0.5), 0.5);
}

TEST(FvLighting, EnvRotationIsAboutPlusY)
{
  const glm::vec4 r = L::envRotation(90);
  glm::quat q;
  q.x = r.x;
  q.y = r.y;
  q.z = r.z;
  q.w = r.w;
  expectNear(q * glm::vec3(1, 0, 0), {0, 0, -1}, 1e-6F);
  expectNear(q * glm::vec3(0, 1, 0), {0, 1, 0}, 1e-6F);
  const glm::vec4 none = L::envRotation(0);
  EXPECT_NEAR(none.w, 1.0F, 1e-7F);
}

TEST(FvLighting, PresetRotationIsActionsTs)
{
  // ((r + 180) mod 360 + 360) mod 360 - 180, evaluated in node
  EXPECT_DOUBLE_EQ(L::presetRotation(90), 90);
  EXPECT_DOUBLE_EQ(L::presetRotation(270), -90);
  EXPECT_DOUBLE_EQ(L::presetRotation(-190), 170);
  EXPECT_DOUBLE_EQ(L::presetRotation(180), -180);
}

TEST(FvLighting, LinearColorIsSrgbDecoded)
{
  expectNear(L::linearColor("#ffffff"), {1, 1, 1}, 1e-6F);
  expectNear(L::linearColor("#000000"), {0, 0, 0}, 1e-6F);
  EXPECT_NEAR(L::linearColor("#808080").x, 0.2158605F, 1e-5F);
  EXPECT_THROW(L::linearColor("808080"), std::invalid_argument);
  EXPECT_THROW(L::linearColor("#80808g"), std::invalid_argument);
}

TEST(FvLighting, StudioKeyPanelLightsFromWhereThreeShowsIt)
{
  // The warm key panel is where red most exceeds blue (the rim is cool, the softbox neutral).
  const uint32_t     w = 512, h = 256;
  std::vector<float> img = L::studioImage(w, h);
  ASSERT_EQ(img.size(), size_t(w) * h * 4);
  uint32_t bi = 0, bj = 0;
  float    best = -1e9F;
  for(uint32_t j = 0; j < h; j++)
    for(uint32_t i = 0; i < w; i++)
    {
      const float* p = &img[(size_t(j) * w + i) * 4];
      if(p[0] - p[2] > best)
      {
        best = p[0] - p[2];
        bi   = i;
        bj   = j;
      }
    }
  const double u = (bi + 0.5) / w, v = (bj + 0.5) / h;

  // The renderer reads rows from the top: latlongToDir (nvshaders/functions.h.slang).
  const double    phi = 2 * kPi * u - kPi, theta = kPi * v;
  const glm::vec3 renderer(float(std::sin(theta) * std::cos(phi)), float(std::cos(theta)), float(std::sin(theta) * std::sin(phi)));

  // three uploads the same rows flipped (flipY), then equirectUV: u = atan2(z, x) / 2pi + 0.5,
  // v = asin(y) / pi + 0.5.
  const double    tu = u, tv = 1.0 - v;
  const double    az = (tu - 0.5) * 2 * kPi, el = (tv - 0.5) * kPi;
  const glm::vec3 three(float(std::cos(el) * std::cos(az)), float(std::sin(el)), float(std::cos(el) * std::sin(az)));

  expectNear(renderer, three, 1e-5F);
  // createStudioEnvironment's panel (0.6, 0.35, 0.72) is in the image's own frame, half a turn
  // from the lookup's azimuth
  expectNear(renderer, glm::normalize(glm::vec3(-0.6F, 0.35F, -0.72F)), 0.02F);
}

TEST(FvLighting, DefaultsRoundTripThroughJson)
{
  L::State s;
  s.sun.azimuth      = -40;
  s.ground.mode      = "catcher";
  s.ground.autoHeight = false;
  s.ground.height    = -0.25;
  s.env.source       = "hdr";
  s.env.path         = "C:/hdri/studio.hdr";
  const L::State back = L::applyPatch(L::State{}, L::toJson(s));
  EXPECT_EQ(L::toJson(back), L::toJson(s));
}

TEST(FvLighting, UnknownKeyRejectsTheWholePatch)
{
  const L::State s;
  EXPECT_THROW(L::applyPatch(s, {{"sun", {{"intensity", 5}}}, {"moon", {{"enabled", true}}}}), std::invalid_argument);
  EXPECT_THROW(L::applyPatch(s, {{"sun", {{"intensity", 5}, {"brightness", 1}}}}), std::invalid_argument);
  EXPECT_THROW(L::applyPatch(s, {{"ground", {{"mode", "lava"}}}}), std::invalid_argument);
  EXPECT_THROW(L::applyPatch(s, {{"tonemap", {{"exposure", "bright"}}}}), std::invalid_argument);
  EXPECT_THROW(L::applyPatch(s, {{"env", {{"color", "red"}}}}), std::invalid_argument);
  EXPECT_THROW(L::applyPatch(s, {{"env", 3}}), std::invalid_argument);
  EXPECT_THROW(L::applyPatch(s, Json::array()), std::invalid_argument);
}

TEST(FvLighting, NumbersAreClamped)
{
  const L::State s = L::applyPatch(L::State{},
                                   {{"sun", {{"elevation", 120}, {"intensity", -3}, {"azimuth", 190}}},
                                    {"shadow", {{"size", 3000}}},
                                    {"env", {{"blur", 4}}},
                                    {"ground", {{"darkness", 2}}}});
  EXPECT_EQ(s.sun.elevation, 90);
  EXPECT_EQ(s.sun.intensity, 0);
  EXPECT_EQ(s.sun.azimuth, -170);
  EXPECT_EQ(s.shadow.size, 4096);  // the nearest power of two
  EXPECT_EQ(s.env.blur, 1);
  EXPECT_EQ(s.ground.darkness, 1);
  EXPECT_EQ(L::applyPatch(L::State{}, {{"shadow", {{"size", 100}}}}).shadow.size, 1024);
  EXPECT_EQ(L::applyPatch(L::State{}, {{"shadow", {{"size", 1e6}}}}).shadow.size, 8192);
}

TEST(FvLighting, NullResetsAndReadOnlyKeysAreIgnored)
{
  L::State s     = L::applyPatch(L::State{}, {{"sun", {{"intensity", 7}, {"azimuth", 10}}}, {"ground", {{"height", 1.5}}}});
  s              = L::applyPatch(s, {{"sun", {{"intensity", nullptr}}}, {"ground", {{"height", "auto"}}}});
  const L::Sun d;
  EXPECT_EQ(s.sun.intensity, d.intensity);
  EXPECT_EQ(s.sun.azimuth, 10);
  EXPECT_TRUE(s.ground.autoHeight);
  s = L::applyPatch(s, {{"sun", nullptr}});
  EXPECT_EQ(s.sun.azimuth, d.azimuth);
  // a topic object, read-only fields and all, replays as a patch
  Json topic                = L::toJson(s);
  topic["env"]["name"]      = "studio.hdr";
  topic["env"]["size"]      = {2048, 1024};
  topic["env"]["loading"]   = true;
  EXPECT_NO_THROW(L::applyPatch(L::State{}, topic));
}

TEST(FvLighting, MergeDiffIsTheMinimalPatch)
{
  const Json a = {{"env", {{"intensity", 1}, {"name", "x"}}}, {"sun", {{"enabled", true}}}};
  const Json b = {{"env", {{"intensity", 2}}}, {"sun", {{"enabled", true}}}};
  EXPECT_EQ(L::mergeDiff(a, b), (Json{{"env", {{"intensity", 2}, {"name", nullptr}}}}));
  EXPECT_TRUE(L::mergeDiff(b, b).empty());
}

TEST(FvLighting, RendererParamsFollowTheState)
{
  L::State s;
  s.env.source       = "sky";
  s.env.background   = "color";
  s.ground.mode      = "catcher";
  s.sun.shadows      = false;
  s.tonemap.method   = "agx";
  s.tonemap.exposure = 1.5;
  const Json p       = L::rendererParams(s, {2048, 1024});
  EXPECT_EQ(p["envSystem"], 0);
  EXPECT_EQ(p["useSolidBackground"], true);
  EXPECT_EQ(p["hdrBlur"], 0.0);
  EXPECT_EQ(p["useInfinitePlane"], true);
  EXPECT_EQ(p["isShadowCatcher"], true);
  EXPECT_EQ(p["rasterShadows"], false);
  EXPECT_EQ(p["tmMethod"], 4);
  EXPECT_EQ(p["tmExposure"], 1.5);
  s.ground.mode = "off";
  s.env.source  = "plain";
  EXPECT_EQ(L::rendererParams(s, {})["useInfinitePlane"], false);
  EXPECT_EQ(L::rendererParams(s, {})["envSystem"], 3);
}
