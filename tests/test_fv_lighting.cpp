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

TEST(FvLighting, TheFloorIsAverageGround)
{
  // a neutral 20% grey, linear: what daylighting takes the ground to reflect
  const glm::vec3 c = L::linearColor(L::Ground{}.color);
  EXPECT_NEAR(0.2126F * c.x + 0.7152F * c.y + 0.0722F * c.z, 0.2F, 0.005F);
  EXPECT_FLOAT_EQ(c.x, c.y);
  EXPECT_FLOAT_EQ(c.y, c.z);
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
                                    {"env", {{"blur", 4}}},
                                    {"ground", {{"darkness", 2}}}});
  EXPECT_EQ(s.sun.elevation, 90);
  EXPECT_EQ(s.sun.intensity, 0);
  EXPECT_EQ(s.sun.azimuth, -170);
  EXPECT_EQ(s.env.blur, 1);
  EXPECT_EQ(s.ground.darkness, 1);
  // as wide as a softbox: 30 degrees, at 0.15 a unit
  EXPECT_EQ(L::applyPatch(L::State{}, {{"sun", {{"softness", 150}}}}).sun.softness, 150);
  EXPECT_EQ(L::applyPatch(L::State{}, {{"sun", {{"softness", 500}}}}).sun.softness, 200);
  EXPECT_NEAR(L::sunRadiusKm(200) / 149597870.0, std::tan(15.0 * 3.14159265358979 / 180.0), 1e-9);
  // twilight under the physical sky takes thousands (lighting.ts sentExposure)
  EXPECT_EQ(L::applyPatch(L::State{}, {{"tonemap", {{"exposure", 4000}}}}).tonemap.exposure, 4000);
  EXPECT_EQ(L::applyPatch(L::State{}, {{"tonemap", {{"exposure", 1e7}}}}).tonemap.exposure, 1e5);
  EXPECT_EQ(L::applyPatch(L::State{}, {{"tonemap", {{"exposure", 0}}}}).tonemap.exposure, 0.01);
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

// A scene kept before protocol 17 has the raster view's shadow map in its lighting: it still opens
TEST(FvLighting, TheRasterShadowMapOfAnOlderSceneIsDropped)
{
  const L::State s = L::applyPatch(L::State{}, {{"sun", {{"intensity", 3}, {"shadows", false}}},
                                                {"shadow", {{"size", 2048}, {"contactHardening", false}}}});
  EXPECT_EQ(s.sun.intensity, 3);
  const Json topic = L::toJson(s);
  EXPECT_FALSE(topic.contains("shadow"));
  EXPECT_FALSE(topic["sun"].contains("shadows"));
  EXPECT_FALSE(L::rendererParams(s, {}).contains("rasterShadows"));
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
  s.tonemap.method   = "agx";
  s.tonemap.exposure = 1.5;
  const Json p       = L::rendererParams(s, {2048, 1024});
  EXPECT_EQ(p["envSystem"], 0);
  EXPECT_EQ(p["useSolidBackground"], true);
  EXPECT_EQ(p["hdrBlur"], 0.0);
  EXPECT_EQ(p["useInfinitePlane"], true);
  EXPECT_EQ(p["isShadowCatcher"], true);
  EXPECT_EQ(p["tmMethod"], 4);
  EXPECT_EQ(p["tmExposure"], 1.5);
  // the firefly clamp goes with the lighting, 16 before the tone mapper's exposure
  EXPECT_DOUBLE_EQ(p["ptFireflyClamp"].get<double>(), 16.0 / 1.5);
  s.tonemap.method = "aces";
  EXPECT_DOUBLE_EQ(L::rendererParams(s, {})["ptFireflyClamp"].get<double>(), 16.0 / (1.5 / 0.6));
  // the still's clamp stays while the exposure keeps within 4x of the one it was sent for (a new one
  // restarts the still): a tone-curve change or a few slider ticks keep a converged still
  EXPECT_DOUBLE_EQ(L::stillFireflyClamp(64.0 / 1.5, 64.0), 64.0);
  EXPECT_DOUBLE_EQ(L::stillFireflyClamp(64.0 * 0.6, 64.0), 64.0);
  EXPECT_DOUBLE_EQ(L::stillFireflyClamp(64.0 / 4.0, 64.0), 64.0);
  EXPECT_DOUBLE_EQ(L::stillFireflyClamp(64.0 * 4.0, 64.0), 64.0);
  EXPECT_DOUBLE_EQ(L::stillFireflyClamp(64.0 / 5.0, 64.0), 64.0 / 5.0);
  EXPECT_DOUBLE_EQ(L::stillFireflyClamp(64.0 * 5.0, 64.0), 64.0 * 5.0);
  EXPECT_DOUBLE_EQ(L::stillFireflyClamp(40.0, 0.0), 40.0);
  s.tonemap.method = "agx";
  s.ground.mode = "off";
  s.env.source  = "plain";
  EXPECT_EQ(L::rendererParams(s, {})["useInfinitePlane"], false);
  EXPECT_EQ(L::rendererParams(s, {})["envSystem"], 3);
}

// Water on the ground: within 0..1, dry by default (as a scene from before it), and on a floor only
TEST(FvLighting, WaterWetsAFloorOnly)
{
  EXPECT_EQ(L::rendererParams(L::State{}, {})["groundWater"], 0.0);
  L::State s = L::applyPatch(L::State{}, {{"ground", {{"water", 0.6}, {"ripples", 3}}}});
  EXPECT_EQ(s.ground.water, 0.6);
  EXPECT_EQ(s.ground.ripples, 1);
  EXPECT_EQ(L::toJson(s)["ground"]["water"], 0.6);
  EXPECT_EQ(L::rendererParams(s, {})["groundWater"], 0.6);
  EXPECT_EQ(L::rendererParams(s, {})["groundRipples"], 1.0);
  s.ground.mode = "catcher";
  EXPECT_EQ(L::rendererParams(s, {})["groundWater"], 0.0);
  EXPECT_EQ(L::applyPatch(s, {{"ground", {{"water", nullptr}}}}).ground.water, 0.0);
}

// Clouds: none by default (as a scene from before them), clamped to the renderer's ranges, reset by
// null, and sent as its cloud settings
TEST(FvLighting, CloudsAreClearUntilSetAndReachTheRenderer)
{
  EXPECT_EQ(L::rendererParams(L::State{}, {})["cloudCoverage"], 0.0);
  L::State s = L::applyPatch(L::State{}, {{"clouds", {{"coverage", 0.4}, {"thickness", 9.0}, {"seed", 3.4}}}});
  EXPECT_EQ(s.clouds.coverage, 0.4);
  EXPECT_EQ(s.clouds.thickness, 4.0);
  EXPECT_EQ(s.clouds.seed, 3);
  EXPECT_EQ(L::toJson(s)["clouds"]["coverage"], 0.4);
  const fv::Json p = L::rendererParams(s, {});
  EXPECT_EQ(p["cloudCoverage"], 0.4);
  EXPECT_EQ(p["cloudThickness"], 4.0);
  EXPECT_EQ(p["cloudSeed"], 3);
  EXPECT_EQ(L::applyPatch(s, {{"clouds", nullptr}}).clouds.coverage, 0.0);
  EXPECT_THROW(L::applyPatch(s, {{"clouds", {{"wind", 1}}}}), std::invalid_argument);

  // the kind of cloud: cumulus until set, by name, reaching the renderer as its CLOUD_TYPE_* index
  EXPECT_EQ(s.clouds.type, "cumulus");
  EXPECT_EQ(p["cloudType"], 0);
  const L::State st = L::applyPatch(s, {{"clouds", {{"type", "stratus"}}}});
  EXPECT_EQ(L::toJson(st)["clouds"]["type"], "stratus");
  EXPECT_EQ(L::rendererParams(st, {})["cloudType"], 2);
  EXPECT_EQ(L::rendererParams(L::applyPatch(s, {{"clouds", {{"type", "stratocumulus"}}}}), {})["cloudType"], 1);
  // altocumulus and cirrus are the tiers over the low one, each its own cover
  EXPECT_THROW(L::applyPatch(s, {{"clouds", {{"type", "altocumulus"}}}}), std::invalid_argument);
  EXPECT_EQ(p["cloudMidCoverage"], 0.0);
  const L::State tiers = L::applyPatch(s, {{"clouds", {{"mid", 0.3}, {"high", 1.5}}}});
  EXPECT_EQ(L::rendererParams(tiers, {})["cloudMidCoverage"], 0.3);
  EXPECT_EQ(L::rendererParams(tiers, {})["cloudHighCoverage"], 1.0);
  EXPECT_EQ(L::toJson(tiers)["clouds"]["mid"], 0.3);
  EXPECT_EQ(tiers.clouds.coverage, 0.4);  // the low tier stays
  EXPECT_EQ(st.clouds.coverage, 0.4);  // the rest of the clouds stay
  EXPECT_THROW(L::applyPatch(s, {{"clouds", {{"type", "cirrus"}}}}), std::invalid_argument);
}

// Haze: off by default (as a scene from before it), kept with its visibility and glow while off, clamped
// to the renderer's ranges, and sent as its visibility, 0 while off (protocol 36)
TEST(FvLighting, HazeIsOffUntilSwitchedOnAndKeepsItsValues)
{
  const fv::Json none = L::rendererParams(L::State{}, {});
  EXPECT_EQ(none["hazeVisibility"], 0.0);
  EXPECT_EQ(none["hazeAnisotropy"], 0.7);
  EXPECT_EQ(L::toJson(L::State{})["haze"]["enabled"], false);

  const L::State dim = L::applyPatch(L::State{}, {{"haze", {{"visibility", 40.0}, {"glow", 2.0}}}});
  EXPECT_EQ(dim.haze.visibility, 40.0);
  EXPECT_EQ(dim.haze.glow, 0.9);
  EXPECT_EQ(L::rendererParams(dim, {})["hazeVisibility"], 0.0);  // set, but still off
  const L::State on = L::applyPatch(dim, {{"haze", {{"enabled", true}}}});
  EXPECT_EQ(L::rendererParams(on, {})["hazeVisibility"], 40.0);
  EXPECT_EQ(L::rendererParams(on, {})["hazeAnisotropy"], 0.9);
  EXPECT_EQ(L::toJson(on)["haze"]["visibility"], 40.0);
  EXPECT_EQ(L::applyPatch(on, {{"haze", {{"visibility", 0.0}}}}).haze.visibility, 1.0);
  EXPECT_FALSE(L::applyPatch(on, {{"haze", nullptr}}).haze.enabled);
  EXPECT_THROW(L::applyPatch(on, {{"haze", {{"colour", 1}}}}), std::invalid_argument);
  EXPECT_THROW(L::applyPatch(on, {{"haze", {{"enabled", 1}}}}), std::invalid_argument);
  // the topic replays as a patch
  EXPECT_EQ(L::applyPatch(L::State{}, L::toJson(on)).haze.visibility, 40.0);
}

// The exposure follows the clouds where the app's camera meters (protocol 35): what they take from the
// light given back, by the sky's share of a clear day's light and the share of the sun past them
TEST(FvLighting, TheExposureFollowsTheClouds)
{
  L::State s;
  s.env.source       = "sky";
  s.tonemap.exposure = 2.0;
  const L::CloudReading overcast{.sky = 0.5, .sun = 0.0};
  // an app from before it doesn't follow them: the exposure is its own
  EXPECT_EQ(L::rendererParams(s, {}, overcast)["tmExposure"], 2.0);
  s = L::applyPatch(s, {{"tonemap", {{"followClouds", true}, {"skyShare", 0.4}}}});
  // the sky gives 40% of a clear day's light, now at half of it, and the sun none: a fifth is left
  EXPECT_DOUBLE_EQ(L::cloudLight(s, overcast), 0.2);
  const Json p = L::rendererParams(s, {}, overcast);
  EXPECT_DOUBLE_EQ(p["tmExposure"].get<double>(), 2.0 / 0.2);
  // the firefly clamp goes with the exposure the camera takes
  EXPECT_DOUBLE_EQ(p["ptFireflyClamp"].get<double>(), 16.0 / (2.0 / 0.2));
  // a clear sky takes nothing; the sun out between clouds that brighten the sky stops it down a little
  EXPECT_DOUBLE_EQ(L::rendererParams(s, {}, {})["tmExposure"].get<double>(), 2.0);
  EXPECT_LT(L::cloudExposure(s, {.sky = 1.5, .sun = 1.0}), 1.0);
  EXPECT_GT(L::cloudExposure(s, {.sky = 1.5, .sun = 1.0}), 0.8);
  // the sun behind a thin veil: most of the light is the sun's, so most of the change
  EXPECT_DOUBLE_EQ(L::cloudLight(s, {.sky = 1.0, .sun = 0.5}), 0.4 + 0.6 * 0.5);
  // a deck that takes everything: ten stops at most
  EXPECT_DOUBLE_EQ(L::cloudExposure(s, {.sky = 0.0, .sun = 0.0}), 1000.0);
  // the sun off: the sky is all the light there is (the app sends a share of 1), so the shadow is nothing
  EXPECT_DOUBLE_EQ(L::cloudLight(L::applyPatch(s, {{"tonemap", {{"skyShare", 1.0}}}}), {.sky = 0.6, .sun = 0.0}), 0.6);
  // a manual camera doesn't follow: the light changes, the exposure doesn't
  EXPECT_EQ(L::cloudExposure(L::applyPatch(s, {{"tonemap", {{"followClouds", false}}}}), overcast), 1.0);
  // another environment has no clouds
  L::State hdr = s;
  hdr.env.source = "hdr";
  EXPECT_EQ(L::cloudLight(hdr, overcast), 1.0);
  EXPECT_EQ(L::cloudExposure(hdr, overcast), 1.0);
  // in the topic, clamped, and the topic with its read-only reading replays as a patch
  EXPECT_EQ(L::toJson(s)["tonemap"]["followClouds"], true);
  EXPECT_EQ(L::toJson(s)["tonemap"]["skyShare"], 0.4);
  EXPECT_EQ(L::applyPatch(s, {{"tonemap", {{"skyShare", 3}}}}).tonemap.skyShare, 1.0);
  Json topic                     = L::toJson(s);
  topic["tonemap"]["cloudLight"] = 0.2;
  const L::State back            = L::applyPatch(L::State{}, topic);
  EXPECT_TRUE(back.tonemap.followClouds);
  EXPECT_EQ(back.tonemap.skyShare, 0.4);
  EXPECT_THROW(L::applyPatch(s, {{"tonemap", {{"followClouds", 1}}}}), std::invalid_argument);
}

// A body of water: the ground's water mode, its depth and waves; a floor's wet layer goes with it
TEST(FvLighting, TheWaterModeIsABodyOfWater)
{
  EXPECT_EQ(L::rendererParams(L::State{}, {})["groundWaterDepth"], 0.0);
  L::State s = L::applyPatch(L::State{}, {{"ground", {{"mode", "water"}, {"depth", 1.2}, {"waves", 0.7}, {"water", 0.5}}}});
  Json     p = L::rendererParams(s, {});
  EXPECT_EQ(p["useInfinitePlane"], true);
  EXPECT_EQ(p["isShadowCatcher"], false);
  EXPECT_EQ(p["groundWaterDepth"], 1.2);
  EXPECT_EQ(p["groundWaves"], 0.7);
  EXPECT_EQ(p["groundWater"], 0.0);
  EXPECT_EQ(L::toJson(s)["ground"]["depth"], 1.2);
  // a puddle's depth at least, ten metres at most
  EXPECT_EQ(L::applyPatch(s, {{"ground", {{"depth", 0}}}}).ground.depth, 0.01);
  EXPECT_EQ(L::applyPatch(s, {{"ground", {{"depth", 50}}}}).ground.depth, 10.0);
  s.ground.mode = "floor";
  EXPECT_EQ(L::rendererParams(s, {})["groundWaterDepth"], 0.0);
}

namespace {

// A lat-long sky of luminance 1 with, optionally, a disc of `sun` around `toSun` (environment frame)
std::vector<float> latlong(uint32_t w, uint32_t h, glm::vec3 toSun, float sun, float radiusDeg)
{
  std::vector<float> px(size_t(w) * h * 4, 1.0F);
  const double       pi = 3.14159265358979323846;
  for(uint32_t y = 0; y < h; y++)
    for(uint32_t x = 0; x < w; x++)
    {
      const double    phi = 2.0 * pi * (x + 0.5) / w - pi, theta = pi * (y + 0.5) / h;
      const glm::vec3 d(float(std::cos(phi) * std::sin(theta)), float(std::cos(theta)), float(std::sin(phi) * std::sin(theta)));
      if(sun > 0 && glm::dot(d, toSun) >= std::cos(radiusDeg * pi / 180.0))
        for(int c = 0; c < 3; c++)
          px[(size_t(y) * w + x) * 4 + c] = sun;
    }
  return px;
}

}  // namespace

// The Light tab's "Aim at the HDR's sun": the bright disc of a lat-long image found as a direction,
// and that direction as the panel's angles once the environment is turned
TEST(FvLighting, FindsTheSunInAnEnvironmentImage)
{
  const glm::vec3 toSun = glm::normalize(glm::vec3(0.4F, 0.5F, -0.7F));
  const auto      img   = latlong(1024, 512, toSun, 5000.0F, 1.0F);
  const auto      found = fv::lighting::environmentSun(img.data(), 1024, 512);
  ASSERT_TRUE(found);
  EXPECT_GT(glm::dot(*found, toSun), std::cos(0.3 * 3.14159265358979 / 180.0));

  // the panel's angles aim the sun light there: unturned, and turned 90 degrees about +Y
  for(double rotation : {0.0, 90.0, -135.0})
  {
    const glm::dvec2 a     = fv::lighting::environmentSunAngles(*found, rotation);
    const glm::vec4  q     = fv::lighting::envRotation(rotation);
    const glm::vec3  world = glm::quat(q.w, q.x, q.y, q.z) * *found;
    EXPECT_GT(glm::dot(fv::lighting::sunDirection(a.x, a.y), world), 0.99999F) << rotation;
    EXPECT_GE(a.x, -180.0);
    EXPECT_LE(a.x, 180.0);
  }

  // an overcast sky, and a soft studio light a few times the sky, have no sun
  EXPECT_FALSE(fv::lighting::environmentSun(latlong(1024, 512, toSun, 0.0F, 0.0F).data(), 1024, 512));
  EXPECT_FALSE(fv::lighting::environmentSun(latlong(1024, 512, toSun, 6.0F, 20.0F).data(), 1024, 512));
}
