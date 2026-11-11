/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// GltfRenderer's settings: every value the command line, the app and MCP can set is declared here.

#include <algorithm>
#include <cmath>
#include <utility>

#include <glm/gtc/quaternion.hpp>

#include <nvutils/logger.hpp>

#include "renderer.hpp"

#include "gltf_environment_sky.hpp"
#include "sun_position.hpp"
#include "sky_preset.hpp"
#include "utils.hpp"

// The constructor registers the parameters that can be set from the command line
GltfRenderer::GltfRenderer(nvutils::ParameterRegistry* paramReg, const nvutils::ParameterParser* paramParser)
    : m_parameterParser(paramParser)
    , m_settings(paramReg)
{
  // Every user-settable value is declared once here (see settings_registry.hpp): the command line,
  // MCP, the app's parameter messages and ImGui.ini all follow from this one call. Persist says whether the
  // value is remembered between runs.
  m_settings.add({.name            = "envSystem",
                  .help            = "Environment: [Sky:0, HDR:1, None:2, Plain:3, Gradient:4]",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { onEnvironmentChanged(); }},
                 (int*)&m_resources.settings.envSystem, Persist::eYes, 0, 4);
  // what the path tracer reads of the environment: a change starts the still again, or it blends in
  const auto restartStill = [this](const nvutils::ParameterBase* const) { resetFrame(); };
  m_settings.add({.name = "hdrIntensity", .help = "HDR Environment Intensity", .callbackSuccess = restartStill},
                 &m_resources.settings.hdrEnvIntensity, Persist::eYes, 0.0F, 100.0F);
  // One orientation for every environment -- OMI_environment_sky's `rotation`, in its own form. It
  // is also where north is, so like the Time of Day settings, an edit re-places
  // the sun after the ini restore (see m_todDrivesSun).
  m_settings.addVector({.name = "envRotation",
                        .help = "Environment rotation: unit quaternion x y z w (glTF order); turns the HDR, "
                                "every sky, and the compass the sun is placed with",
                        .callbackSuccess =
                            [this](const nvutils::ParameterBase* const) {
                              if(skyHasSun(m_resources.settings.envSystem))
                                m_todDrivesSun = true;
                              onEnvironmentRotated();
                            }},
                       &m_resources.settings.envRotation, Persist::eYes);
  m_settings.add({.name = "hdrBlur", .help = "HDR Environment Blur", .callbackSuccess = restartStill},
                 &m_resources.settings.hdrBlur, Persist::eYes, 0.0F, 1.0F);
  m_settings.addVector({"silhouetteColor", "Color of the silhouette"}, &m_resources.settings.silhouetteColor, Persist::eYes);
  m_settings.add({"dbgVisualization", "Visualization Mode"}, (int*)&m_resources.settings.visualization, Persist::eYes);
  m_settings.add({"dbgWireframe", "Enable wireframe overlay"}, &m_resources.settings.wireframe, Persist::eYes);
  // Native presentation on an HDR display (hdr_output.hpp): the frames it is shown go to the display in
  // HDR, their highlights past SDR white. The stream, screenshots and exports stay SDR.
  m_settings.add({"presentHdr", "Presentation: on a display in HDR mode, show the frames in HDR"}, &m_presentHdr, Persist::eYes);
  m_settings.add({"presentHdrPeak", "Presentation: the HDR display's peak in nits (0 = as the display reports it)"},
                 &m_presentHdrPeak, Persist::eYes, 0.0F, 10000.0F);
  m_settings.add({"ptOptimalShader",
                  "Compile gltf_pathtrace.slang with GLTF_USE_* gates specialized per scene "
                  "(no runtime MAT_EXT_* changes; triggers shader recompile on scene/material change). Default off."},
                 &m_resources.settings.optimalShader, Persist::eYes);
  // A value rather than a trigger, so a host can turn it off again over MCP (triggers aren't exposed there).
  m_settings.add({.name = "useSolidBackground", .help = "Use solid color background", .callbackSuccess = restartStill},
                 &m_resources.settings.useSolidBackground, Persist::eYes);
  m_settings.addVector({.name = "solidBackgroundColor", .help = "Solid Background Color", .callbackSuccess = restartStill},
                       &m_resources.settings.solidBackgroundColor, Persist::eYes);
  m_settings.add({"ptMaxSamples", "Path tracer: stop accumulating at this many samples per pixel"},
                 &m_resources.settings.maxSamples, Persist::eYes, 1, 1000000);
  // Persisted. Headless and scripted runs neither read nor write the ini, so they can't change what
  // the next session plays.
  m_settings.add(
      {.name = "animPlay",
       .help = "Play animations [Off:0, On:1]. Default: on (last Play/Pause choice is remembered), off in headless runs. "
               "A scene with a behavior graph never autoplays",
       .callbackSuccess = [this](const nvutils::ParameterBase* const) { m_resources.animationControl.play = m_animPlay; }},
      &m_animPlay, Persist::eYes);
  // The ini restore writes m_animPlay directly, after a start-up scene may already have set
  // animationControl.play from the default. Only ever turn playback off here: a scene with a
  // behavior graph has already forced it off and must stay so.
  m_settings.addPostRestoreHook([this]() { m_resources.animationControl.play &= m_animPlay; });
  // Headless-only output path: consumed at start-up, so remembering it would be misleading.
  m_settings.add({"output", "Output image file path for headless mode"}, &m_resources.headlessOutputPath, Persist::eNo);
  m_settings.add({"outputSamples",
                  "Path tracer: also save the output image each time the displayed image reaches one of these "
                  "samples per pixel, e.g. \"1,2,4,8\". Written as <output>_<spp>spp<ext> (see --output); the "
                  "count is that of the image shown, denoised or not"},
                 &m_outputSamples, Persist::eNo);

  // Tonemapper. tmBrightness and tmVignette were once tmGamma and
  // tmWhitePoint -- names for fields TonemapperData does not have.
  m_settings.add({"tmMethod", "Tonemapper: Method [Filmic:0, Uncharted:1, Clip:2, ACES:3, AgX:4, KhronosPBR:5, Shaped:6]"},
                 &m_resources.tonemapperData.method, Persist::eYes, 0, 6);
  m_settings.add({"tmShapedToe", "Tonemapper: the Shaped curve's toe, 0 open shadows to 1 crushed"},
                 &m_resources.tonemapperData.shapedToe, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({"tmShapedShoulder", "Tonemapper: the Shaped curve's shoulder, 0 highlights rolling off late to 1 early and gently"},
                 &m_resources.tonemapperData.shapedShoulder, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({"tmHighlightsToWhite", "Tonemapper: how fast Neutral's and Shaped's compressed colours go white, 0 never to 1 fast (0.5 Khronos's)"},
                 &m_resources.tonemapperData.highlightsToWhite, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({"tmActive", "Tonemapper: Enable tone mapping [Off:0, On:1]"}, &m_resources.tonemapperData.isActive,
                 Persist::eYes, 0, 1);
  // Floor below 0.1, which is not enough headroom for a bright environment with
  // auto-exposure off -- what a reproducible capture wants. Out-of-range values are clamped rather
  // than refused, so too high a floor is silent: the run succeeds and the image is simply not the
  // one that was asked for.
  // The ceiling is the host's lighting.set exposure (1e5, for the physical sky at twilight) over
  // ACES's 0.6 (fv_lighting.cpp tmExposure), with room.
  m_settings.add({"tmExposure", "Tonemapper: Exposure multiplier"}, &m_resources.tonemapperData.exposure, Persist::eYes,
                 0.01F, 1e6F);
  m_settings.add({"tmContrast", "Tonemapper: Contrast"}, &m_resources.tonemapperData.contrast, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmBrightness", "Tonemapper: Brightness"}, &m_resources.tonemapperData.brightness, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmSaturation", "Tonemapper: Saturation"}, &m_resources.tonemapperData.saturation, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmVignette", "Tonemapper: Vignette (0 = none)"}, &m_resources.tonemapperData.vignette, Persist::eYes, -1.0F, 1.0F);
  m_lens.registerParameters(&m_settings);
  m_bloom.registerParameters(&m_settings);
  m_localExposure.registerParameters(&m_settings);
  m_settings.add({"tmDither", "Tonemapper: Dither [Off:0, On:1]"}, &m_resources.tonemapperData.dither, Persist::eYes, 0, 1);
  // White balance
  m_settings.add({"tmTemperature", "Tonemapper: White balance temperature (Kelvin)"},
                 &m_resources.tonemapperData.temperature, Persist::eYes, 2000.0F, 15000.0F);
  m_settings.add({"tmTint", "Tonemapper: White balance tint (ANSI C78.377 Duv)"}, &m_resources.tonemapperData.tint,
                 Persist::eYes, -0.03F, 0.03F);
  // Colour grading
  m_settings.add({"tmVibrance", "Tonemapper: Vibrance (boosts muted colors only)"},
                 &m_resources.tonemapperData.vibrance, Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmShadowBias", "Tonemapper: Shadow tone bias"}, &m_resources.tonemapperData.shadowBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmMidtoneBias", "Tonemapper: Midtone brightness bias"}, &m_resources.tonemapperData.midtoneBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmHighlightBias", "Tonemapper: Highlight tone bias"}, &m_resources.tonemapperData.highlightBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.addVector({"tmCoolColor", "Tonemapper: Split-toning tint for shadows"},
                       &m_resources.tonemapperData.coolColor, Persist::eYes);
  m_settings.addVector({"tmWarmColor", "Tonemapper: Split-toning tint for highlights"},
                       &m_resources.tonemapperData.warmColor, Persist::eYes);
  m_settings.add({"tmSplitBalance", "Tonemapper: Split-toning cool/warm balance"},
                 &m_resources.tonemapperData.splitBalance, Persist::eYes, -0.5F, 0.5F);
  // Looks (the app's Image › Look): black and white, lift/gamma/gain, posterising, grain
  m_settings.addVector({"tmMonoMix", "Tonemapper: Black-and-white weights of linear R, G, B (a colour filter's)"},
                       &m_resources.tonemapperData.monoMix, Persist::eYes);
  m_settings.add({"tmMono", "Tonemapper: Black and white, 0 colour to 1"}, &m_resources.tonemapperData.mono,
                 Persist::eYes, 0.0F, 1.0F);
  m_settings.addVector({"tmLift", "Tonemapper: Lift a channel (added to the shadows)"}, &m_resources.tonemapperData.lift,
                       Persist::eYes);
  m_settings.addVector({"tmGamma", "Tonemapper: Gamma a channel (above 1 lighter midtones)"},
                       &m_resources.tonemapperData.gamma, Persist::eYes);
  m_settings.addVector({"tmGain", "Tonemapper: Gain a channel"}, &m_resources.tonemapperData.gain, Persist::eYes);
  m_settings.add({"tmPosterize", "Tonemapper: Posterise to this many levels a channel (under 2 off)"},
                 &m_resources.tonemapperData.posterize, Persist::eYes, 0.0F, 64.0F);
  m_settings.add({"tmPosterizeMix", "Tonemapper: How much of the posterised image"},
                 &m_resources.tonemapperData.posterizeMix, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({"tmGrain", "Tonemapper: Film grain, 0 none to 1 heavy"}, &m_resources.tonemapperData.grain,
                 Persist::eYes, 0.0F, 1.0F);
  // A small sensor's picture (the app's Footage looks)
  m_settings.add({"tmSensorClip", "Tonemapper: A small sensor's highlights, clipped a channel at a time, in place of the curve's (0 none to 1)"},
                 &m_resources.tonemapperData.sensorClip, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({"tmSensorNoise", "Tonemapper: A small sensor's noise, strongest in the shadows and coloured (0 none to 1 heavy)"},
                 &m_resources.tonemapperData.sensorNoise, Persist::eYes, 0.0F, 1.0F);
  // Auto exposure (turn off for reproducible headless captures)
  m_settings.add({"tmAutoExposure", "Tonemapper: Auto-exposure [Off:0, On:1]"},
                 &m_resources.tonemapperData.autoExposure, Persist::eYes, 0, 1);
  m_settings.add({"tmAutoExposureSpeed", "Tonemapper: Auto-exposure adaptation speed"},
                 &m_resources.tonemapperData.autoExposureSpeed, Persist::eYes, 0.0F, 100.0F);
  m_settings.add({"tmEvMin", "Tonemapper: Auto-exposure minimum (EV100)"}, &m_resources.tonemapperData.evMinValue,
                 Persist::eYes, -24.0F, 24.0F);
  m_settings.add({"tmEvMax", "Tonemapper: Auto-exposure maximum (EV100)"}, &m_resources.tonemapperData.evMaxValue,
                 Persist::eYes, -24.0F, 24.0F);

  // Panel visibility: persisted before, but not settable. Exposed now so a scripted capture can
  // hide the chrome.

  // The sun. One direction is shared by every sky type that has a sun, which is why these are not
  // named sky*: Resources::sunDirection is the single source of truth and the angles drive it
  // through the callback. Everything else below is direct.
  //
  // SkySun is initialized here, not in onAttach: the command line is parsed between construction
  // and onAttach, and these callbacks fire during that parse.
  m_skySun.init(&m_resources,
                {
                    .resetFrame         = [this] { resetFrame(); },
                    .environmentPreview = [this] { requestEnvironmentPreview(); },
                });
  // SkySun::aim() reports the consequences: a re-bake for the physical sky (the only one whose image
  // depends on the sun) and an accumulation reset.
  const auto applySunAngles = [this](const nvutils::ParameterBase* const) {
    m_skySun.setAngles(m_resources.settings.sunAzimuth, m_resources.settings.sunElevation);
  };
  m_settings.add({.name = "sunAzimuth",
                  .help = "Sun azimuth as a compass bearing (degrees, 0 = north, 90 = east); shared by every sky type",
                  .callbackSuccess = applySunAngles},
                 &m_resources.settings.sunAzimuth, Persist::eYes, 0.0F, 360.0F);
  m_settings.add({.name = "sunElevation", .help = "Sun elevation (degrees); shared by every sky type", .callbackSuccess = applySunAngles},
                 &m_resources.settings.sunElevation, Persist::eYes, -90.0F, 90.0F);
  // ImGui.ini restore writes sunAzimuth/Elevation directly and never runs applySunAngles, so
  // Resources::sunDirection -- the value rendering actually uses -- would stay at
  // its default and the remembered sun position would silently not be restored. Re-run the
  // conversion once, after Application::run() has reloaded the ini (see onUIRender's one-shot).
  //
  // CLI precedence: the loadFilter installed in onAttach already skips ini writes for any key
  // ParameterParser::wasParsed() marked, so a CLI-set azimuth or elevation survives the ini
  // reload. This hook then recomputes sunDirection from the resulting per-key mix (CLI value
  // where the user overrode it, ini value otherwise), which is exactly the intended
  // CLI-wins-per-key behavior.
  m_settings.addPostRestoreHook([this, applySunAngles]() { applySunAngles(nullptr); });

  // Time of Day: the same sun, said differently. These do not add a second source of truth -- each
  // one recomputes sunAzimuth/sunElevation through m_skySun.applyTimeOfDay(), so the sun still has exactly
  // one owner and the two angles remain what is persisted, scripted and written to a marked light.
  //
  // Persisted so the widget reopens where it was left; the callback is what makes a command-line
  // --todHour actually move the sun. ini restore deliberately does not fire it: the sun's direction
  // is restored from its own persisted angles, and re-deriving it here would overwrite a sun that
  // had since been moved by hand.
  // Open on today and on this machine's clock, rather than on an arbitrary epoch somebody has to
  // correct before the widget says anything true. Read here, before the declarations below, so it
  // is the value SettingsRegistry captures as the default -- and therefore what "Reset All to
  // Default" comes back to.
  if(const sun_position::LocalClock now = sun_position::systemLocalClock(); now.valid)
  {
    m_resources.settings.todDate      = now.date;
    m_resources.settings.todHour      = now.hour;
    m_resources.settings.todUtcOffset = now.utcOffsetHours;
  }
  const auto applyTod = [this](const nvutils::ParameterBase* const) {
    m_todDrivesSun = true;
    m_skySun.applyTimeOfDay();
  };
  m_settings.add({.name = "todHour", .help = "Time of day, in local clock hours (0..24) at --todUtcOffset", .callbackSuccess = applyTod},
                 &m_resources.settings.todHour, Persist::eYes, 0.0F, 24.0F);
  m_settings.add({.name = "todDate", .help = "Date the sun is computed for, yyyy-mm-dd", .callbackSuccess = applyTod},
                 &m_resources.settings.todDate, Persist::eYes);
  m_settings.add({.name = "todLatitude", .help = "Observer latitude in degrees, positive north", .callbackSuccess = applyTod},
                 &m_resources.settings.todLatitude, Persist::eYes, -90.0F, 90.0F);
  m_settings.add({.name = "todLongitude", .help = "Observer longitude in degrees, positive east", .callbackSuccess = applyTod},
                 &m_resources.settings.todLongitude, Persist::eYes, -180.0F, 180.0F);
  m_settings.add({.name = "todUtcOffset", .help = "Hours the local clock runs ahead of UTC", .callbackSuccess = applyTod},
                 &m_resources.settings.todUtcOffset, Persist::eYes, -12.0F, 14.0F);

  // A city is an action, not a mode: it writes latitude, longitude and the offset, and nothing
  // reads it back -- the same shape as atmoPreset, and not persisted for the same reason.
  //
  // Named rather than indexed, unlike atmoPreset. Twenty-odd entries make an index unreadable in a
  // script, and worse, inserting a city would silently move every script and saved sequence that
  // used one. A name survives the list growing.
  m_settings.add({.name = "todCity",
                  .help = "Set latitude, longitude and UTC offset from a named city (see the Environment panel "
                          "for the list); standard time, no daylight saving",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        const int index = sun_position::findCity(m_resources.settings.todCity);
                        if(index < 0)
                        {
                          LOGW("--todCity: '%s' is not one of the listed cities; the location is unchanged.\n",
                               m_resources.settings.todCity.c_str());
                          return;
                        }
                        const sun_position::City& city    = sun_position::kCities[index];
                        m_resources.settings.todLatitude  = city.latitudeDeg;
                        m_resources.settings.todLongitude = city.longitudeDeg;
                        m_resources.settings.todUtcOffset = city.utcOffsetHours;
                        m_todDrivesSun                    = true;
                        m_skySun.applyTimeOfDay();
                      }},
                 &m_resources.settings.todCity, Persist::eNo);

  // Precedence, and the one place it has to be stated.
  //
  // The command line is parsed before the ini is read, so `--todHour 20` computes a sun and then
  // the sunAzimuth/sunElevation hook above puts the *remembered* sun back over it -- those two keys
  // were not on the command line, so the ini is allowed to supply them, and the hook that re-derives
  // the direction from them cannot tell it is undoing an explicit instruction. Silently: the run
  // simply renders yesterday's sun, which looks exactly like the widget not working.
  //
  // So if anything asked for a time of day, it asks again here, after the
  // restore. Registered after that hook, and hooks run in order, so it wins. With no such request
  // the flag is false and the remembered angles stand, which is what reopening the app should do.
  m_settings.addPostRestoreHook([this]() {
    if(m_todDrivesSun)
      m_skySun.applyTimeOfDay();
  });
  // Authored sky types (glTF OMI_environment_sky). Every one of these invalidates the baked
  // lighting environment, so they share one callback that marks it for a re-bake; the bake itself
  // runs on the application thread at the top of the next frame, never inside a parameter write.
  const auto invalidateBake = [this](const nvutils::ParameterBase* const) { onEnvironmentChanged(); };
  m_settings.add({.name = "envSaveToGltf", .help = "Write OMI_environment_sky into the glTF when saving the scene"},
                 &m_resources.settings.envSaveToGltf, Persist::eYes);
  m_settings.addVector({.name = "plainColor", .help = "Plain sky: solid color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.plainColor, Persist::eYes);
  m_settings.addVector({.name = "gradientBottomColor", .help = "Gradient sky: nadir color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientBottomColor, Persist::eYes);
  m_settings.addVector({.name = "gradientHorizonColor", .help = "Gradient sky: horizon color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientHorizonColor, Persist::eYes);
  m_settings.addVector({.name = "gradientTopColor", .help = "Gradient sky: zenith color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientTopColor, Persist::eYes);
  m_settings.addVector({.name = "gradientSunColor", .help = "Gradient sky: sun color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientSunColor, Persist::eYes);
  m_settings.add({.name = "gradientBottomCurve", .help = "Gradient sky: horizon-to-nadir falloff", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientBottomCurve, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "gradientTopCurve", .help = "Gradient sky: horizon-to-zenith falloff", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientTopCurve, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "gradientSunAngleMax", .help = "Gradient sky: angular extent of the sun glow (radians)", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientSunAngleMax, Persist::eYes, 0.0F, 3.1416F);
  m_settings.add({.name = "gradientSunCurve", .help = "Gradient sky: sun-to-sky falloff across the glow", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientSunCurve, Persist::eYes, 0.0F, 1.0F);

  // The physical sky's atmosphere.
  //
  // Every one of these goes through the same preview-then-settle path a sky colour does. Most of
  // them change nothing but the scattering tables, so the drag rebuilds those at preview quality
  // and the commit rebuilds them at full -- see SkyBruneton::Quality. Ground albedo, sun angular
  // radius and altitude are the exceptions that the bake also reads directly, so they move even
  // before the tables catch up.
  const auto invalidateAtmosphere = [this](const nvutils::ParameterBase* const) { requestEnvironmentPreview(); };
  m_settings.add({.name = "atmoSunAngularRadius", .help = "Physical sky: angular radius of the sun disk (radians)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoSunAngularRadius, Persist::eYes, 0.0F, 0.2F);
  m_settings.addVector({.name = "atmoGroundAlbedo", .help = "Physical sky: ground reflectance, linear RGB", .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoGroundAlbedo, Persist::eYes);
  m_settings.addVector({.name            = "atmoLandAlbedo",
                        .help            = "Physical sky: what the land round the scene reflects, linear RGB; it takes the "
                                           "ground reflectance's place. 0 0 0 for none",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoLandAlbedo, Persist::eNo);
  m_settings.add({.name            = "landFadeFrom",
                  .help            = "Physical sky: the distance from the origin past which what the camera hit gives way "
                                     "to the sky's own ground (scene units; 0: none)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.landFadeFrom, Persist::eNo, 0.0F, 1e9F);
  m_settings.add({.name            = "landFadeTo",
                  .help            = "Physical sky: the distance from the origin by which it has given way wholly",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.landFadeTo, Persist::eNo, 0.0F, 1e9F);
  // Altitude changes no LUT -- it is a bake input only -- so it is the cheap one of the three.
  m_settings.add({.name            = "atmoObserverAltitude",
                  .help            = "Physical sky: observer height above the ground (metres)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoObserverAltitude, Persist::eYes, 0.0F, 60000.0F);

  // Aerial perspective. Both shading paths read it straight out of SceneFrameInfo, so unlike every
  // other atmosphere setting it needs no re-bake and no LUT rebuild: it changes what a frame does
  // with the tables, not the tables. Hence resetFrame() alone.
  m_settings.add({.name            = "atmoAerialPerspectiveScale",
                  .help            = "Aerial perspective: metres of air per scene unit (0 disables)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.atmoAerialPerspectiveScale, Persist::eYes, 0.0F, 1000.0F);

  m_settings.addVector({.name            = "atmoSolarIrradiance",
                        .help            = "Physical sky: solar irradiance at the top of the atmosphere (W/m^2)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoSolarIrradiance, Persist::eYes);

  // Rayleigh -- the air itself.
  m_settings.addVector({.name            = "atmoRayleighScattering",
                        .help            = "Physical sky: Rayleigh scattering at the surface (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoRayleighScattering, Persist::eYes);
  m_settings.add({.name            = "atmoRayleighScaleHeight",
                  .help            = "Physical sky: height over which the air thins by 1/e (km)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoRayleighScaleHeight, Persist::eYes, 0.1F, 100.0F);

  // Mie -- the aerosol.
  m_settings.addVector({.name            = "atmoMieScattering",
                        .help            = "Physical sky: Mie scattering at the surface (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoMieScattering, Persist::eYes);
  m_settings.add({.name            = "atmoMieScaleHeight",
                  .help            = "Physical sky: height over which the aerosol thins by 1/e (km)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieScaleHeight, Persist::eYes, 0.1F, 100.0F);
  m_settings.add({.name            = "atmoMieAnisotropy",
                  .help            = "Physical sky: Cornette-Shanks asymmetry; > 0 scatters forward",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieAnisotropy, Persist::eYes, -0.99F, 0.99F);
  m_settings.add({.name            = "atmoMieAlbedo",
                  .help            = "Physical sky: aerosol single-scattering albedo; 1 scatters all it removes",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieAlbedo, Persist::eYes, 0.05F, 1.0F);
  m_settings.add({.name            = "atmoVisibility",
                  .help            = "Physical sky: how far one sees through the air at the ground (km); the aerosol is "
                                     "scaled to it. 0 leaves the atmosphere's own",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoVisibility, Persist::eYes, 0.0F, 1000.0F);

  // Ozone -- an absorbing layer well above the ground, and what keeps a clear zenith blue rather
  // than grey at low sun.
  m_settings.addVector({.name            = "atmoOzoneExtinction",
                        .help            = "Physical sky: absorption at the peak of the ozone layer (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoOzoneExtinction, Persist::eYes);
  m_settings.add({.name = "atmoOzoneCenter", .help = "Physical sky: altitude of the ozone peak (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoOzoneCenter, Persist::eYes, 0.0F, 200.0F);
  m_settings.add({.name = "atmoOzoneWidth", .help = "Physical sky: full thickness of the ozone layer (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoOzoneWidth, Persist::eYes, 0.1F, 400.0F);

  // The planet.
  m_settings.add({.name = "atmoBottomRadius", .help = "Physical sky: planet radius (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoBottomRadius, Persist::eYes, 100.0F, 20000.0F);
  m_settings.add({.name = "atmoThickness", .help = "Physical sky: atmosphere depth above the surface (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoThickness, Persist::eYes, 1.0F, 500.0F);

  // The clouds over the physical sky. Each re-bakes the cloud image with the sky, through the same
  // preview-then-settle path; none of them touches the atmosphere's tables.
  m_settings.add({.name = "cloudCoverage", .help = "Physical sky: the low clouds' cover, 0 clear to 1 overcast", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.cloudCoverage, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name            = "cloudThickness",
                  .help            = "Physical sky: the low clouds' depth from the base to the highest tops (km)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.cloudThickness, Persist::eYes, 0.2F, 4.0F);
  m_settings.add({.name = "cloudSeed", .help = "Physical sky: which arrangement of the clouds", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.cloudSeed, Persist::eYes, 0, 1 << 20);
  m_settings.add({.name            = "cloudType",
                  .help            = "Physical sky: the low clouds' kind [Cumulus:0, Stratocumulus:1, Stratus:2]",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.cloudType, Persist::eYes, 0, 2);
  m_settings.add({.name            = "cloudMidCoverage",
                  .help            = "Physical sky: altocumulus 3 km up, 0 none to 1 a closed deck",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.cloudMidCoverage, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name            = "cloudHighCoverage",
                  .help            = "Physical sky: cirrus 7.5 km up, the share of the sky it veils",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.cloudHighCoverage, Persist::eYes, 0.0F, 1.0F);

  // A preset is an action, not a mode: its callback overwrites every field above and nothing reads
  // it back. That is also why it does not persist -- restoring it after the individual values
  // would undo them.
  m_settings.add({.name = "atmoPreset",
                  .help = "Physical sky: apply an atmosphere preset [Earth:0, Mars:1, Alien:2]",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        applyAtmospherePreset(m_resources.settings, m_resources.settings.atmoPreset);
                        requestEnvironmentPreview();
                      }},
                 &m_resources.settings.atmoPreset, Persist::eNo, 0, eAtmospherePresetCount - 1);

  // The MDL sky's eleven knobs (skyMultiplier, skyHaze, skyRedBlueShift, skySaturation,
  // skyHorizonHeight, skyHorizonBlur, skyGroundColor, skyNightColor, skySunDiskScale,
  // skySunDiskIntensity, skySunGlowIntensity) were registered here until that sky was deleted.
  // They are deliberately not re-registered under new names: the Bruneton atmosphere is not a
  // renaming of those parameters, it is a different model. A persisted .ini still carrying them
  // is handled by SettingsRegistry, which ignores keys it does not know.

  // Live: the retained scene's next batch of BLAS builds regrows its scratch when it needs more.
  m_settings.add({.name = "blasScratchBudgetMB",
                  .help = "Retained scene: the scratch one batch of BLAS builds shares, in MB",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        m_resources.settings.blasScratchBudgetMB = std::max(m_resources.settings.blasScratchBudgetMB, 1u);
                        if(m_resources.retainedGpu)
                          m_resources.retainedGpu->setBlasScratchBudget(VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20);
                      }},
                 &m_resources.settings.blasScratchBudgetMB, Persist::eNo);
  // Start-up only, so not exposed to a host (fv_parameter_filter.hpp): read at each retained scene
  // replace, where a smaller capacity than the scene needs only grows by doubling.
  // Live: a smaller budget trims the grace list at once.
  m_settings.add({.name = "textureGraceMB",
                  .help = "Retained scene: decoded images nothing samples, kept for the next scene or edit, in MB",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        for(rs::RetainedSceneVk* retained : {m_resources.retainedGpu.get(), m_resources.retainedSpare.get()})
                          if(retained)
                            retained->setTextureGrace(VkDeviceSize(m_resources.settings.textureGraceMB) << 20);
                      }},
                 &m_resources.settings.textureGraceMB, Persist::eNo);
  m_settings.add({.name = "sceneBudgetMB",
                  .help = "Retained scene: the GPU memory the scene is fitted to, where less than the device's budget, in MB (0: that budget)"},
                 &m_resources.settings.sceneBudgetMB, Persist::eNo);
  m_settings.add({.name = "memoryFillMB",
                  .help = "Device memory held and touched every frame, for a measure of the device's limit, in MB (0: none)"},
                 &m_resources.settings.memoryFillMB, Persist::eNo);
  m_settings.add({.name = "memoryFillWriteMB", .help = "How much of memoryFillMB is written whole each frame, in turn, in MB"},
                 &m_resources.settings.memoryFillWriteMB, Persist::eNo);
  m_settings.add({.name = "residencyMinInstances", .help = "Retained scene: the least instance capacity (start-up only)"},
                 &m_resources.settings.residencyMinInstances, Persist::eNo);
  // Live: from the next refit on (P7, measured on Path Trace after a pose far from the load pose).
  m_settings.add({.name = "blasRefitsBeforeRebuild",
                  .help = "Retained scene: a BLAS refitted this many times is built again (0: never)",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        for(rs::RetainedSceneVk* retained : {m_resources.retainedGpu.get(), m_resources.retainedSpare.get()})
                          if(retained)
                            retained->setRefitsBeforeRebuild(m_resources.settings.blasRefitsBeforeRebuild);
                      }},
                 &m_resources.settings.blasRefitsBeforeRebuild, Persist::eNo);
  // Live: for the BLASes built from then on (P7, B56: measured on the scene's BLAS memory).
  m_settings.add({.name = "blasCompactStatic",
                  .help = "Retained scene: a static BLAS is compacted once its size is known",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        for(rs::RetainedSceneVk* retained : {m_resources.retainedGpu.get(), m_resources.retainedSpare.get()})
                          if(retained)
                            retained->setCompactStatic(m_resources.settings.blasCompactStatic);
                      }},
                 &m_resources.settings.blasCompactStatic, Persist::eNo);
  m_settings.add({.name = "blasCompactDeforming",
                  .help = "Retained scene: a deformed BLAS (posed figure, hair) is compacted and refitted as its copy",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        for(rs::RetainedSceneVk* retained : {m_resources.retainedGpu.get(), m_resources.retainedSpare.get()})
                          if(retained)
                            retained->setCompactDeforming(m_resources.settings.blasCompactDeforming);
                      }},
                 &m_resources.settings.blasCompactDeforming, Persist::eNo);
  m_settings.add({.name = "blasTimeBuilds",
                  .help = "Retained scene (debug): a timing line with the GPU time of a frame's BLAS builds and refits, one frame in a few"},
                 &m_resources.settings.blasTimeBuilds, Persist::eNo);
  // Debug, live (read at the next scene replace): every composition then retires the old scene's
  // slots and reallocates, so holes, slot reuse, capacity growth and deferred frees run on the GPU
  // before structural edits make them ordinary (daz-viewer docs/retained-scene-plan.md, P4).
  m_settings.add({.name = "residencyRetireOnReplace",
                  .help = "Retained scene (debug): a scene replace retires the old scene's slots and reallocates instead of starting over"},
                 &m_resources.settings.residencyRetireOnReplace, Persist::eNo);

  // Gizmo, grid and snap: previously persisted but unreachable from the command line or MCP.
  m_settings.add({"uiGridUnit", "Grid base unit (world units)"}, &m_resources.settings.gridUnit, Persist::eYes);
  m_settings.add({"uiSnapEnabled", "Snap gizmo transforms to grid increments"}, &m_resources.settings.snapEnabled, Persist::eYes);
  m_settings.add({"uiSnapRotation", "Rotation snap increment (degrees)"}, &m_resources.settings.snapRotation, Persist::eYes);
  m_settings.add({"uiSnapScale", "Scale snap increment"}, &m_resources.settings.snapScale, Persist::eYes);
  // the app's view.set space and snap (FvView), which it sends at every start: not kept here
  m_settings.add({"uiSnapMove", "Move snap increment (world units)"}, &m_resources.settings.snapMove, Persist::eNo);
  m_settings.add({"uiGizmoLocal", "The gizmo along the object's own axes, not its parent's"}, &m_resources.settings.gizmoLocal, Persist::eNo);

  // The ground: the path tracer's infinite plane.
  const auto groundChanged = [this](const nvutils::ParameterBase* const) { resetFrame(); };
  m_settings.add({.name = "useInfinitePlane", .help = "Ground: draw an infinite plane under the scene", .callbackSuccess = groundChanged},
                 &m_resources.settings.useInfinitePlane, Persist::eYes);
  m_settings.add({.name = "isShadowCatcher", .help = "Ground: the plane only darkens the background where shadows fall", .callbackSuccess = groundChanged},
                 &m_resources.settings.isShadowCatcher, Persist::eYes);
  m_settings.add({.name = "infinitePlaneDistance", .help = "Ground: height of the plane (world units)", .callbackSuccess = groundChanged},
                 &m_resources.settings.infinitePlaneDistance, Persist::eNo);
  m_settings.addVector({.name = "infinitePlaneBaseColor", .help = "Ground: plane color (linear)", .callbackSuccess = groundChanged},
                       &m_resources.settings.infinitePlaneBaseColor, Persist::eYes);
  m_settings.add({.name = "infinitePlaneMetallic", .help = "Ground: plane metalness", .callbackSuccess = groundChanged},
                 &m_resources.settings.infinitePlaneMetallic, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "infinitePlaneRoughness", .help = "Ground: plane roughness", .callbackSuccess = groundChanged},
                 &m_resources.settings.infinitePlaneRoughness, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "shadowCatcherDarkness", .help = "Ground: extra darkening of caught shadows (0..1)", .callbackSuccess = groundChanged},
                 &m_resources.settings.shadowCatcherDarkness, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "groundFade", .help = "Ground: fade the plane out over a disk of this many scene radii (0: no fade)", .callbackSuccess = groundChanged},
                 &m_resources.settings.groundFade, Persist::eYes, 0.0F, 100.0F);
  m_settings.add({.name = "groundWater", .help = "Ground: water on the plane, 0 dry, damp to 0.5, puddles from 0.3, 1 a sheet of it", .callbackSuccess = groundChanged},
                 &m_resources.settings.groundWater, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "groundRipples", .help = "Ground: ripples on its water, 0 still to 1", .callbackSuccess = groundChanged},
                 &m_resources.settings.groundRipples, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "groundWaterDepth", .help = "Ground: a body of water this many metres deep over the plane (0: none)", .callbackSuccess = groundChanged},
                 &m_resources.settings.groundWaterDepth, Persist::eYes, 0.0F, 10.0F);
  m_settings.add({.name = "groundWaves", .help = "Ground: the waves on the body of water, 0 still to 1 gentle", .callbackSuccess = groundChanged},
                 &m_resources.settings.groundWaves, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "skySunScale", .help = "Sky: multiplier on the light the sky's sun gives (0: no sun light)", .callbackSuccess = groundChanged},
                 &m_resources.settings.skySunScale, Persist::eNo, 0.0F, 100.0F);
  // Haze: SceneFrameInfo alone, as the aerial perspective (fillHaze)
  m_settings.add({.name            = "hazeVisibility",
                  .help            = "Haze: a thin haze in the air, lit by the scene's lights; its visibility in metres (0: none)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.hazeVisibility, Persist::eNo, 0.0F, 100000.0F);
  m_settings.add({.name            = "hazeAnisotropy",
                  .help            = "Haze: how much it scatters forward, the halo round a light (Henyey-Greenstein g, -0.9 to 0.9)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.hazeAnisotropy, Persist::eNo, -0.9F, 0.9F);
  // A debug check that light sampling and BSDF sampling each give the whole image (light_pick.h.slang)
  m_settings.add({.name            = "ptEstimator",
                  .help            = "PathTracer (debug): what counts light both a light sample and a BSDF ray reach "
                                     "[Both, weighed (MIS):0, Light samples alone:1, BSDF rays alone:2]; the images agree "
                                     "unless a pdf is wrong",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.ptEstimator, Persist::eNo, 0, 2);
  // The air's sun-shadow ray by its roulette (contribution_roulette.h.slang); 0 for the old ray on every path
  m_settings.add({.name            = "ptAerialSunShare",
                  .help            = "PathTracer: the air's sun-shadow ray is traced as often as keeps the noise it adds "
                                     "under this share of the pixel (physical sky; 0: on every path)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.ptAerialSunShare, Persist::eNo, 0.0F, 1.0F);
  // A still's shadow rays by the same roulette, against the pixel's mean so far; 0 for every ray traced
  m_settings.add({.name            = "ptShadowShare",
                  .help            = "PathTracer: a still's light sample traces its shadow ray as often as keeps the noise it "
                                     "adds under this share of the pixel's mean so far (0: every one)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.ptShadowShare, Persist::eNo, 0.0F, 1.0F);
  // The thin preview's slab (skin_slab.h.slang); off for the preview as it was, all of skin's light back where it went in
  m_settings.add({.name            = "ptSkinSlab",
                  .help            = "PathTracer (debug): the thin skin preview lets light through thin skin (ears, fingers) to its far side",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.ptSkinSlab, Persist::eNo);
  // The thin preview's spread (skin_spread.h.slang); off for the preview as it was, skin's light taken where it lands
  m_settings.add({.name            = "ptSkinSpread",
                  .help            = "PathTracer (debug): the thin skin preview softens shadows on skin as the still's walk does, "
                                     "taking each channel's light where the walk lets it out",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.ptSkinSpread, Persist::eNo);
  // The thin preview's skin by angle (skin_albedo.h.slang); off for the map's Lambertian, as before
  m_settings.add({.name            = "ptSkinView",
                  .help            = "PathTracer (debug): the thin skin preview returns light as the still's walk does, "
                                     "brighter at a slanting view and with the exit's Fresnel toward the light",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.ptSkinView, Persist::eNo);
  // Display › Time from a command line (view.set's display "time" in the app): each pixel coloured by how
  // long its thread ran against the frame's average (pixel_time.h.slang); where the frame's time went is
  // logged as it comes back
  m_settings.add({.name            = "pixelTime",
                  .help            = "Debug: each pixel coloured by its GPU time against the frame's average (the Time view; "
                                     "needs VK_KHR_shader_clock)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.pixelTime, Persist::eNo);
  // ...and what it sums in place of the time: how many rays a pixel traces, apart from how dear each is
  m_settings.add({.name            = "pixelTimeCount",
                  .help            = "Debug: what the Time view sums [Time:0, Rays:1, Walk steps:2, Shadow rays:3]",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.pixelTimeCount, Persist::eNo, 0, 3);
  m_settings.add({.name            = "hazeCandidates",
                  .help            = "Haze: lights a still's scatter point picks its light from, by what each gives the eye (1: the light pick's own)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.hazeCandidates, Persist::eNo, 1, 16);
  m_settings.add({.name            = "hazeHeight",
                  .help            = "Haze: thicker near the floor, thinning over this many metres above it (0: even through the height)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.hazeHeight, Persist::eNo, 0.0F, 10000.0F);
  m_settings.add({.name            = "hazePatches",
                  .help            = "Haze: how patchy it is, 0 even to 1 (the density swings from none to twice its own)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.hazePatches, Persist::eNo, 0.0F, 1.0F);
  m_settings.add({.name            = "hazePatchSize",
                  .help            = "Haze: the patches' size in metres (0: a quarter of the haze's radius, 0.5 to 8 m)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.hazePatchSize, Persist::eNo, 0.0F, 10000.0F);

  // Register PathTracer-specific command line parameters
  m_pathTracer.registerParameters(&m_settings, [this]() { resetFrame(); });

  // Saves the viewport without its overlays (scripts/render-scene.py). The next frame is drawn that
  // way and saved once submitted (onPostRender).
  paramReg->add({.name = "screenshot",
                 .help = "Save the viewport to this image, without its overlays",
                 .callbackSuccess =
                     [this](const nvutils::ParameterBase* const) {
                       if(m_app && !m_screenshotParam.empty())
                         m_screenshotPending = std::exchange(m_screenshotParam, {});
                     }},
                {".png", ".jpg", ".jpeg"}, &m_screenshotParam);

  // Initialize camera manipulator
  m_cameraManip = std::make_shared<DragCameraManipulator>();
  m_resources.cameraManip = m_cameraManip;  // Share with resources
}
