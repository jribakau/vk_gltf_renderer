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

// GltfRenderer's environment: the sky and its presets, the bakes behind it, the HDR panorama, and
// the firefly clamp that follows them.

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <utility>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>  // glm::rotation

#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/mipmaps.hpp>

#include "renderer.hpp"

#include <figure/io/timing.hpp>
#include "env_image_loader.hpp"
#include "gltf_environment_sky.hpp"
#include "sun_position.hpp"
#include "sky_preset.hpp"
#include "staging_config.hpp"
#include "tinygltf_utils.hpp"
#include "utils.hpp"

namespace {

// Angular radius of the disk the gradient sky draws. The extension has no field for it, so this is
// the sun's true radius as seen from Earth (~0.27 deg) -- enough for the disk to read as a sun.
// Authored decoration rather than a measurement, unlike the physical sky's, which comes from the
// atmosphere. One constant because three places need the same number: the drawn disk, the light's
// angular size, and what a save writes.
constexpr float kGradientSunAngularRadius = 0.00465f;

// How much brighter the gradient sky's sun is than the dome it sits under, as illuminance on a
// surface facing each.
//
// A number this file has to invent, because the gradient sky never measured anything: its colours
// are authored in arbitrary linear units where the dome sits near 1, and nothing in them says how
// bright a sun should be. So it is fitted to the real thing instead, against the clear-day
// direct:diffuse band a measured clear day gives -- which makes it checkable rather than merely
// plausible.
//
// Fitted across the whole band rather than at one sun angle, because this sky cannot follow its
// shape. A real sky's diffuse half dims as the sun drops, so direct:diffuse climbs faster than the
// sun's own cosine; the gradient's dome does not move with the sun at all (its bake excludes the
// sun entirely), so the modelled ratio is exactly proportional to sin(elevation). That leaves one
// free scale, and only 5.2-8.0 keeps every reference elevation from 15 to 80 degrees inside the
// band. 6.5 sits in the middle of that window: measured 1.8 / 3.4 / 4.8 / 5.8 / 6.6 against
// 1.0-2.2 / 2.2-4.0 / 3.5-6.5 / 4.5-8.0 / 5.0-9.0.
constexpr float kGradientSunToSkyRatio = 6.5f;

// Host-side twin of skyGradientBandWeight() in shaders/sky_gradient.h.slang. The two must agree:
// the sun's illuminance is scaled against the dome's, and the dome is this curve.
float gradientBandWeight(float c, float curve)
{
  const float safeCurve = std::max(curve, 1e-3f);
  return glm::clamp(1.0f - std::pow(1.0f - glm::clamp(c, 0.0f, 1.0f), 1.0f / safeCurve), 0.0f, 1.0f);
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Push an authored sky into the live settings. See the declaration for what `source` decides.
//
void GltfRenderer::applySkyDescriptor(const SkyDescriptor& sky, SkySource source, const std::filesystem::path& uriBase)
{
  Settings& settings = m_resources.settings;

  // An explicit `--envSystem` is a statement about this run, and it outranks the file: the same
  // rule the .ini already follows, extended to the place it should have reached first. Without
  // this a script that forces an environment silently gets whatever the scene authored instead,
  // and the measurement it was taking is of something else.
  //
  // Only the *selection* is suppressed. The sky's parameters are still applied, so forcing the
  // mode a scene authored still picks up its colours, and `m_environmentSky` keeps the descriptor
  // either way so a later save preserves it.
  //
  // A preset is the exception, and obviously so: loading one by hand at runtime *is* the statement,
  // and a flag from start-up has no business overruling it.
  // OMI's `rotation` is the renderer's own, quaternion for quaternion, tilt included.
  //
  // Only applied when the file actually states one. The rotation is viewer state -- the user, the
  // command line or MCP may have set it -- and loadScene() leaves viewer state alone on purpose; a
  // scene that says nothing about orientation must not silently reset it. A scene that does state
  // one is making a statement about its own sky, and that wins, like every other authored sky value
  // here. The sun is left where it is: a file's sun is its marked light and a preset's is its
  // sunRotation, both already in world space.
  if(sky.rotationAuthored)
    settings.envRotation = {sky.rotation.x, sky.rotation.y, sky.rotation.z, sky.rotation.w};

  const bool forcedOnCommandLine = source == SkySource::eScene && m_parameterParser && m_parameterParser->wasParsed("envSystem");
  const auto selectEnvironment = [&](shaderio::EnvSystem env) {
    if(forcedOnCommandLine)
    {
      if(settings.envSystem != env)
      {
        LOGI("OMI_environment_sky: scene authors a different environment; --envSystem wins for this run.\n");
      }
      return;
    }
    settings.envSystem = env;
  };

  switch(sky.type)
  {
    case SkyDescriptor::Type::ePlain:
      settings.plainColor = sky.plain.color;
      selectEnvironment(shaderio::EnvSystem::ePlain);
      break;
    case SkyDescriptor::Type::eGradient:
      settings.gradientBottomColor  = sky.gradient.bottomColor;
      settings.gradientHorizonColor = sky.gradient.horizonColor;
      settings.gradientTopColor     = sky.gradient.topColor;
      settings.gradientBottomCurve  = sky.gradient.bottomCurve;
      settings.gradientTopCurve     = sky.gradient.topCurve;
      settings.gradientSunAngleMax  = sky.gradient.sunAngleMax;
      settings.gradientSunCurve     = sky.gradient.sunCurve;
      selectEnvironment(shaderio::EnvSystem::eGradient);
      break;
    case SkyDescriptor::Type::ePanorama: {
      // The image is an external asset, so the URI resolves against the file that named it -- the
      // glTF or the preset, the same rule every other external asset follows
      // (docs/external_assets.md), and NOT against the resources directories, which would silently
      // load someone else's environment when two scenes name the same file.
      if(sky.panorama.uri.empty())
      {
        LOGW("OMI_environment_sky: panorama sky has no NV_environment_sky_panorama.uri%s; keeping the current environment.\n",
             sky.panorama.equirectangular >= 0 ? " (its `equirectangular` texture index is not supported)" : "");
        break;
      }
      const std::filesystem::path resolved =
          uriBase.empty() ? std::filesystem::path(sky.panorama.uri) : uriBase / std::filesystem::path(sky.panorama.uri);
      if(!std::filesystem::exists(resolved))
      {
        LOGW("OMI_environment_sky: panorama image not found: %s\n", nvutils::utf8FromPath(resolved).c_str());
        break;
      }
      createHDR(resolved);
      selectEnvironment(shaderio::EnvSystem::eHdr);
      break;
    }
    case SkyDescriptor::Type::ePhysical:
      // The inverse of the split on save: colour times coefficient, m^-1 back to km^-1.
      settings.atmoRayleighScattering = sky.physical.rayleighColor * (sky.physical.rayleighCoefficient * 1000.0F);
      settings.atmoMieScattering      = sky.physical.mieColor * (sky.physical.mieCoefficient * 1000.0F);
      settings.atmoMieAnisotropy      = sky.physical.mieAnisotropy;
      settings.atmoGroundAlbedo       = sky.physical.groundColor;

      // Then the NV overlay, for everything OMI cannot say. Only when the file carried one: a
      // scene with just the OMI block has not said what its planet is, so leaving the current
      // values alone is more honest than forcing Earth on it.
      if(sky.atmosphere.present)
      {
        const SkyDescriptor::Atmosphere& a = sky.atmosphere;
        settings.atmoSolarIrradiance       = a.solarIrradiance;
        settings.atmoSunAngularRadius      = a.sunAngularRadius;
        settings.atmoMieAlbedo             = a.mieAlbedo;
        // Metres in the file, kilometres in the model. Divided rather than multiplied by 0.001:
        // the reciprocal is not representable in binary, so scaling by it loses a few ulps on
        // every load, and a scene saved and reopened repeatedly would creep. Dividing by an exact
        // power-of-ten float keeps radii like 3389.5 km landing back on themselves.
        settings.atmoOzoneExtinction     = a.ozoneExtinction * 1000.0F;
        settings.atmoRayleighScaleHeight = a.rayleighScaleHeight / 1000.0F;
        settings.atmoMieScaleHeight      = a.mieScaleHeight / 1000.0F;
        settings.atmoOzoneCenter         = a.ozoneCenter / 1000.0F;
        settings.atmoOzoneWidth          = a.ozoneWidth / 1000.0F;
        settings.atmoBottomRadius        = a.planetRadius / 1000.0F;
        settings.atmoThickness           = a.atmosphereThickness / 1000.0F;
      }
      selectEnvironment(shaderio::EnvSystem::eSky);
      break;
    default:
      assert(false && "unhandled SkyDescriptor::Type");
      break;
  }

  // No calibrateFireflyClamp() here: hdrIbl still describes the previous environment until the
  // refresh at the next frame top, which calibrates when the environment kind changed.
  onEnvironmentChanged();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Capture the current sky as a `.sky.json`, and apply one.
//
// The sun angle travels with it, because a sky and the angle of its sun are one look. On the way
// out it is read from the direction the renderer is actually using, whichever of the four things
// set it; on the way in it is applied through m_skySun.aim(), so a scene whose sun is a marked light
// records the move on its undo stack exactly as a slider drag would.
//
// Note what is *not* undoable: everything else a preset changes is viewer state -- the atmosphere,
// the ambient terms, the environment type -- and no environment control in this renderer is on the
// undo stack. Putting presets there alone would mean a settings-snapshot command unlike anything
// else in undo_redo, for a consistency the panel does not otherwise have.
//
bool GltfRenderer::saveSkyPreset(const std::filesystem::path& path) const
{
  const EnvironmentState sky = skyDescriptorFromSettings(path);
  if(!sky.has_value())
  {
    LOGW("Sky preset: the current environment is not a sky that can be written (%s).\n", nvutils::utf8FromPath(path).c_str());
    return false;
  }

  sky_preset::Preset preset{.sky = *sky};
  if(skyHasSun(m_resources.settings.envSystem))
  {
    // glTF aims a light down -Z, so the rotation that takes +Z to the sun is what a node would
    // store -- the same convention SkySun::setMarkedDirection writes.
    preset.sunRotation = glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), glm::normalize(m_resources.sunDirection));
  }
  return sky_preset::save(path, preset);
}

bool GltfRenderer::loadSkyPreset(const std::filesystem::path& path)
{
  const std::optional<sky_preset::Preset> preset = sky_preset::load(path);
  if(!preset.has_value())
    return false;

  // A preset loaded after a scene supersedes that scene's sky, even one still waiting for the frame
  // top -- otherwise the scene's would land a frame later and silently undo this.
  m_environmentSkyPending = false;
  applySkyDescriptor(preset->sky, SkySource::ePreset, path.parent_path());

  // Absent means "leave the sun alone", which is why this is not an else.
  if(preset->sunRotation.has_value())
  {
    m_skySun.aim(*preset->sunRotation * glm::vec3(0.0F, 0.0F, 1.0F));
    m_skySun.syncAngles();
  }

  // The preset is now what a save would write, so a scene saved after loading one carries it.
  m_environmentSky = preset->sky;
  return true;
}

void GltfRenderer::queueStartupSkyPresets(const std::filesystem::path& load, const std::filesystem::path& save)
{
  m_pendingLoadSkyPreset = load;
  m_pendingSaveSkyPreset = save;
}

void GltfRenderer::applyPendingSkyPresets()
{
  if(!m_pendingLoadSkyPreset.empty())
    loadSkyPreset(std::exchange(m_pendingLoadSkyPreset, {}));
  if(!m_pendingSaveSkyPreset.empty())
    saveSkyPreset(std::exchange(m_pendingSaveSkyPreset, {}));
}

//--------------------------------------------------------------------------------------------------
// The URI to record for an environment image, relative to the glTF being written when that is
// possible and absolute when it is not.
//
// Relative is what makes a scene and its environment movable as a pair, which is the common case.
// It stops being the right answer once the image lives outside the glTF's tree -- on another
// drive, or in a shared asset library -- where a relative path would be a long chain of "..\"
// that breaks the moment either end moves. Absolute is honest there.
//
std::string GltfRenderer::relativeUriForSave(const std::filesystem::path& asset, const std::filesystem::path& destination) const
{
  const std::filesystem::path base = destination.parent_path();
  if(base.empty())
    return nvutils::utf8FromPath(asset);

  std::error_code             ec;
  const std::filesystem::path rel = std::filesystem::relative(asset, base, ec);
  if(ec || rel.empty() || *rel.begin() == "..")
    return nvutils::utf8FromPath(asset);

  // glTF URIs use forward slashes regardless of platform.
  std::string uri = nvutils::utf8FromPath(rel);
  std::replace(uri.begin(), uri.end(), '\\', '/');
  return uri;
}

//--------------------------------------------------------------------------------------------------
// Write the current sky into the model being serialized. Runs from Scene::save, on the copy that
// is actually written to disk.
//
// With the toggle off we strip rather than skip: a scene loaded with a sky and saved with the
// toggle off should not keep an extension that no longer matches what the user is looking at.
//
// Reconcile the scene's marked sun light with the sky being written.
//
// OMI_environment_sky describes a medium and says to add suns with KHR_lights_punctual, so a sky
// with a sun is only half-written without one: another renderer would build the atmosphere, find
// nothing to light it, and show an unlit sky. That failure is invisible here, because this
// renderer supplies its own sun when the scene has none -- which is exactly why it has to be
// fixed at the point of writing.
//
// The light is marked (gltf_environment_sky.hpp) so a reload knows which light is the sky's
// rather than guessing, and so this function knows which light is its own to manage. Any other
// light in the file is untouched.
//
// Intensity is the solar illuminance *above* the atmosphere, not the attenuated value this
// renderer shows. The atmosphere travels in the same file; a reader that applies it to an already
// attenuated sun would attenuate twice.
//
void GltfRenderer::writeSkySunLight(tinygltf::Model& outModel) const
{
  const int  existingNode = gltf_environment_sky::findSkySunNode(outModel);
  const bool wantSun      = m_resources.settings.envSaveToGltf && skyHasSun(m_resources.settings.envSystem);

  if(!wantSun)
  {
    // The sky being written has no sun, so nothing should still claim to be one.
    if(existingNode >= 0)
    {
      tinygltf::Node& node = outModel.nodes[existingNode];
      const bool      ours = gltf_environment_sky::isRendererOwnedSun(node);

      tinygltf::Value::Object extras = node.extras.Get<tinygltf::Value::Object>();
      extras.erase(gltf_environment_sky::kSkySunMarkerKey);
      node.extras = tinygltf::Value(std::move(extras));

      if(ours)
      {
        // A sun this renderer invented for an earlier save. Withdrawing it is what keeps the file
        // describing what is on screen.
        //
        // The node is emptied rather than erased: deleting it would renumber every node index in
        // the file. The light entry is left too, for the same reason -- compaction collects both
        // if the user asks for it.
        LOGI("OMI_environment_sky: withdrawing the sun light this renderer added; the sky being saved has none.\n");
        node.light = -1;
        node.extensions.erase("KHR_lights_punctual");
      }
      else
      {
        // Someone else's light that was chosen as the sun. It stops being the sun and stays
        // exactly as it was -- unpicking it is not the same as deleting it.
        LOGI("OMI_environment_sky: '%s' is no longer the sky's sun; the light itself is unchanged.\n", node.name.c_str());
      }
    }
    return;
  }

  // Colour and illuminance of the sun this sky implies, in the units the light carries.
  const shaderio::SkyAtmosphereParameters atmosphere = buildAtmosphereParameters();
  glm::vec3                               illuminance(0.0F);
  float                                   angularRadius = atmosphere.sunAngularRadius;
  if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky)
  {
    // Above the atmosphere on purpose -- see the note below on double attenuation. Shared with the
    // firefly clamp, which wants the same number for its own reasons.
    illuminance = skySunIlluminanceAboveAtmosphere();
  }
  else
  {
    // The gradient sky's sun is drawn rather than measured, so there is no disk radiance to
    // convert; the illuminance is scaled against the dome instead. Same number the renderer lights
    // with, so the file describes what is on screen.
    angularRadius = kGradientSunAngularRadius;
    illuminance   = gradientSunIlluminance() * m_resources.settings.hdrEnvIntensity;
  }

  const float     peak  = std::max({illuminance.x, illuminance.y, illuminance.z});
  const glm::vec3 color = (peak > 0.0F) ? illuminance / peak : glm::vec3(1.0F);

  // Rotation taking +Z to the sun: glTF aims a light down -Z, and the renderer's sunDirection
  // points at it. A world rotation; turned into the node's local one below.
  const glm::vec3 toSun = glm::normalize(m_resources.sunDirection);
  const glm::quat rot   = glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), toSun);

  int nodeIdx = existingNode;
  if(nodeIdx < 0)
  {
    tinygltf::Node node;
    node.name = "Sun";
    gltf_environment_sky::setSkySunMarker(node, gltf_environment_sky::kSkySunOwnerRenderer);
    outModel.nodes.push_back(node);
    nodeIdx = static_cast<int>(outModel.nodes.size()) - 1;

    const int sceneIndex = outModel.defaultScene >= 0 ? outModel.defaultScene : 0;
    if(sceneIndex < static_cast<int>(outModel.scenes.size()))
      outModel.scenes[sceneIndex].nodes.push_back(nodeIdx);
  }

  // A sun the user picked may sit under a rotated parent; the one created above is a root.
  const glm::quat local = gltf_environment_sky::localRotationForWorld(outModel, nodeIdx, rot);
  tinygltf::Node& node  = outModel.nodes[nodeIdx];
  if(!node.matrix.empty())
  {
    // A matrix would win over the TRS set below, so keep its translation and scale as TRS.
    glm::vec3 t, s;
    glm::quat r;
    tinygltf::utils::getNodeTRS(node, t, r, s);
    node.translation = {t.x, t.y, t.z};
    node.scale       = {s.x, s.y, s.z};
    node.matrix.clear();
  }
  node.rotation = {local.x, local.y, local.z, local.w};

  // Reuse the light this node already points at, so a re-save does not add one per save.
  //
  // `node.light` rather than the extensions map: that is the canonical reference in this codebase
  // -- tinygltf mirrors the extension into it on load and serialises from it on save, which is why
  // SceneEditor::addLightNode writes only this field.
  int        lightIdx    = node.light;
  const bool createdHere = (lightIdx < 0 || lightIdx >= static_cast<int>(outModel.lights.size()));
  if(createdHere)
  {
    outModel.lights.push_back({});
    lightIdx = static_cast<int>(outModel.lights.size()) - 1;
  }
  node.light = lightIdx;

  tinygltf::Light& light = outModel.lights[lightIdx];
  // Name only what we create. Adopting someone's "Directional Light" and calling it "Sun" is
  // their scene being edited by a viewer, which is the thing this whole design avoids.
  if(createdHere)
    light.name = "Sun";
  light.type      = "directional";
  light.color     = {color.x, color.y, color.z};
  light.intensity = peak;

  tinygltf::utils::syncExtensionsUsed(outModel);
}

void GltfRenderer::writeEnvironmentSky(tinygltf::Model& outModel, const std::filesystem::path& destination) const
{
  if(!m_resources.settings.envSaveToGltf)
  {
    gltf_environment_sky::strip(outModel);
    writeSkySunLight(outModel);  // drops the sun too: the environment is not part of this file
    return;
  }

  const EnvironmentState state = skyDescriptorFromSettings(destination);
  if(!state.has_value())
  {
    gltf_environment_sky::strip(outModel);
    writeSkySunLight(outModel);  // no sky, so no sun either: withdraw the marker it would leave behind
    return;
  }

  const int sceneIndex = outModel.defaultScene >= 0 ? outModel.defaultScene : 0;
  gltf_environment_sky::write(outModel, sceneIndex, *state);
  writeSkySunLight(outModel);
}

//--------------------------------------------------------------------------------------------------
// The live settings as an authored sky. See the declaration for what nullopt means.
//
EnvironmentState GltfRenderer::skyDescriptorFromSettings(const std::filesystem::path& destination) const
{
  const Settings& settings = m_resources.settings;

  // Start from the descriptor the scene was loaded with, so unmodelled keys and any sky type this
  // build does not render survive the round trip. A scene that authored nothing starts fresh.
  SkyDescriptor sky = m_environmentSky.value_or(SkyDescriptor{});

  // The environment's orientation is the sky's, the same quaternion, so it is written exactly.
  sky.rotation         = settings.envRotationQuat();
  sky.rotationAuthored = true;

  switch(settings.envSystem)
  {
    case shaderio::EnvSystem::ePlain:
      // A sky type this build does not recognize loads as this plain fallback (see
      // SkyDescriptor::unknownType). While the fallback is still on screen untouched, the file's
      // own sky is the truer thing to save; every branch that authors a sky of its own drops it.
      if(settings.plainColor != sky.plain.color)
        sky.unknownType.clear();
      sky.type        = SkyDescriptor::Type::ePlain;
      sky.plain.color = settings.plainColor;
      break;
    case shaderio::EnvSystem::eGradient:
      sky.unknownType.clear();
      sky.type                  = SkyDescriptor::Type::eGradient;
      sky.gradient.bottomColor  = settings.gradientBottomColor;
      sky.gradient.horizonColor = settings.gradientHorizonColor;
      sky.gradient.topColor     = settings.gradientTopColor;
      sky.gradient.bottomCurve  = settings.gradientBottomCurve;
      sky.gradient.topCurve     = settings.gradientTopCurve;
      sky.gradient.sunAngleMax  = settings.gradientSunAngleMax;
      sky.gradient.sunCurve     = settings.gradientSunCurve;
      break;
    case shaderio::EnvSystem::eSky: {
      // OMI describes the atmosphere as a magnitude plus a colour, and in m^-1; the model keeps a
      // per-channel coefficient in km^-1. Splitting on the largest channel is what makes the pair
      // reproduce the extension's own defaults in shape -- its rayleighColor is (0.3, 0.5, 1.0),
      // blue-dominant and normalised to 1 -- and it is exactly invertible, which is the property
      // the round trip depends on.
      const auto split = [](const glm::vec3& perChannelPerKm, float& outCoefficient, glm::vec3& outColor) {
        const float peak = std::max({perChannelPerKm.x, perChannelPerKm.y, perChannelPerKm.z});
        // Divided, not scaled by 0.001: the reciprocal is not representable in binary, so scaling
        // by it costs a few ulps and the value no longer lands back on itself when the file is
        // reopened and saved again.
        outCoefficient = peak / 1000.0F;  // km^-1 -> m^-1, as the extension specifies
        outColor       = (peak > 0.0F) ? perChannelPerKm / peak : glm::vec3(1.0F);
      };

      sky.unknownType.clear();
      sky.type = SkyDescriptor::Type::ePhysical;
      split(settings.atmoRayleighScattering, sky.physical.rayleighCoefficient, sky.physical.rayleighColor);
      split(settings.atmoMieScattering, sky.physical.mieCoefficient, sky.physical.mieColor);
      sky.physical.mieAnisotropy = settings.atmoMieAnisotropy;
      sky.physical.groundColor   = settings.atmoGroundAlbedo;

      // Everything OMI has no field for, in the sibling NV block. Kilometres in the model, metres
      // in the file.
      sky.atmosphere.present             = true;
      sky.atmosphere.solarIrradiance     = settings.atmoSolarIrradiance;
      sky.atmosphere.sunAngularRadius    = settings.atmoSunAngularRadius;
      sky.atmosphere.mieAlbedo           = settings.atmoMieAlbedo;
      sky.atmosphere.ozoneExtinction     = settings.atmoOzoneExtinction / 1000.0F;
      sky.atmosphere.rayleighScaleHeight = settings.atmoRayleighScaleHeight * 1000.0F;
      sky.atmosphere.mieScaleHeight      = settings.atmoMieScaleHeight * 1000.0F;
      sky.atmosphere.ozoneCenter         = settings.atmoOzoneCenter * 1000.0F;
      sky.atmosphere.ozoneWidth          = settings.atmoOzoneWidth * 1000.0F;
      sky.atmosphere.planetRadius        = settings.atmoBottomRadius * 1000.0F;
      sky.atmosphere.atmosphereThickness = settings.atmoThickness * 1000.0F;
      break;
    }
    case shaderio::EnvSystem::eHdr: {
      // eHdr *is* OMI's `panorama`; the enumerator kept its name and value because both the .ini
      // and the glTF store it as an integer.
      if(m_lastHdrFile.empty())
      {
        // Nothing loaded (the start-up dummy). Writing `panorama` with no source would describe an
        // environment that does not exist, so fall through to the preserve-or-strip rule below.
        if(!m_environmentSky.has_value())
          return std::nullopt;
        break;
      }
      sky.unknownType.clear();
      sky.type = SkyDescriptor::Type::ePanorama;
      // Relative to the glTF being written, so the pair moves together. An image outside that
      // tree keeps an absolute path rather than a chain of "..\" that only works from here.
      sky.panorama.uri = relativeUriForSave(m_lastHdrFile, destination);
      break;
    }
    default:
      // The active mode is not an authored OMI type (None). If the scene came with a sky,
      // re-emit it unchanged rather than inventing one or silently discarding the author's data;
      // if it did not, there is nothing to write.
      if(!m_environmentSky.has_value())
        return std::nullopt;
      break;
  }

  return sky;
}

//--------------------------------------------------------------------------------------------------
// Frame-top hook for a pending environment change, paired with onEnvironmentChanged().
//
// Runs from onUIRender rather than onRender:
// rebuilding the environment waits on the device and rewrites descriptors that in-flight frames
// are reading, which is not safe in the middle of recording a frame's command buffer. onUIRender
// is the frame-top point this renderer already uses for deferred GPU rebuilds, and it runs in the
// headless and benchmark paths too.
//
void GltfRenderer::applyPendingEnvironmentRefresh()
{
  if(!m_envBakeDirty)
    return;

  if(isBakedEnvironment(m_resources.settings.envSystem))
  {
    refreshBakedEnvironment();
    return;
  }

  m_envBakeDirty = false;

  if(m_envOwnedByBaker && m_resources.settings.envSystem == shaderio::EnvSystem::eHdr)
  {
    // Leaving a baked sky for the HDR path: hdrIbl still points at EnvBaker's image and holds an
    // alias table built over it, so the file has to be reloaded to take ownership back.
    if(m_lastHdrFile.empty() && m_lastHdrImage)
    {
      EnvImage copy = *m_lastHdrImage;
      uploadHdrEnvironment(std::move(copy), {});
    }
    else
      createHDR(m_lastHdrFile);
  }
  // eNone samples neither hdrIbl nor the prefiltered cubes, so a stale binding is harmless -- and
  // re-selecting a baked sky re-commits before anything reads it.

  // Same rule as refreshBakedEnvironment: calibrate on the first refresh after the kind changes.
  if(m_fireflyClampCalibratedFor != int(m_resources.settings.envSystem))
    calibrateFireflyClamp();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Assemble the authored sky parameters the shaders read.
//
// Built in one place because two consumers need the same numbers for different purposes: the bake
// evaluates them with the sun excluded (that image is the lighting field, and the sun is a
// directional light NEE samples), while the per-ray background evaluates them with the sun in.
// Splitting the fill would let the visible sky and the sky that lights the scene drift apart.
//
shaderio::SkyOmiParameters GltfRenderer::buildSkyOmiParameters() const
{
  const Settings&            settings = m_resources.settings;
  shaderio::SkyOmiParameters params{};

  params.plain.color = settings.plainColor;

  params.gradient.bottomColor  = settings.gradientBottomColor;
  params.gradient.horizonColor = settings.gradientHorizonColor;
  params.gradient.topColor     = settings.gradientTopColor;
  params.gradient.sunColor     = settings.gradientSunColor;
  params.gradient.bottomCurve  = settings.gradientBottomCurve;
  params.gradient.topCurve     = settings.gradientTopCurve;
  params.gradient.sunAngleMax  = settings.gradientSunAngleMax;
  params.gradient.sunCurve     = settings.gradientSunCurve;

  // Resources::sunDirection is the one sun the whole renderer shares -- the sky UI, the physical
  // sky and next-event estimation all read it -- so the visible disk lands exactly where the light
  // that casts the shadows is. The gradient is evaluated in the environment's frame, so the sun
  // is handed over in that frame too.
  params.gradient.sunDirection = sunInEnvironmentFrame();

  params.gradient.sunAngularRadius = kGradientSunAngularRadius;

  return params;
}

glm::vec3 GltfRenderer::sunInEnvironmentFrame() const
{
  // The same conversion the shaders apply before an environment lookup (quatRotateInverse).
  return glm::inverse(m_resources.settings.envRotationQuat()) * m_resources.sunDirection;
}

//--------------------------------------------------------------------------------------------------
// Settings::envRotation was edited. The rotation is where north is, so a sky's sun is placed again
// from Time of Day on the turned compass -- what North Offset used to do. That goes through aim(),
// which also re-bakes the physical sky and reaches a marked sun light. An HDR has no sun to place.
//
void GltfRenderer::onEnvironmentRotated()
{
  if(skyHasSun(m_resources.settings.envSystem))
    m_skySun.applyTimeOfDay();
  else
    m_skySun.syncAngles();  // the sun stays put, but its bearing is on the turned compass now
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// The ground's fade disk: `groundFade` scene radii around the scene's centre, 0 for a ground that
// does not fade (or none).
void GltfRenderer::fillGround(shaderio::SceneFrameInfo& finfo) const
{
  finfo.groundFadeRadius       = 0.0F;
  const Settings&     settings = m_resources.settings;
  finfo.groundWater            = settings.groundWater;
  finfo.groundRipples          = settings.groundRipples;
  finfo.waterDepth             = settings.groundWaterDepth;
  finfo.waterWaves             = settings.groundWaves;
  const SceneGpuView& view     = m_resources.gpuView();
  if(!settings.useInfinitePlane || settings.groundFade <= 0.0F || !view.hasContent() || view.renderNodeCapacity() == 0)
    return;
  const nvutils::Bbox bounds = view.bounds();
  const glm::vec3     center = bounds.center();
  finfo.groundFadeCenterX    = center.x;
  finfo.groundFadeCenterZ    = center.z;
  finfo.groundFadeRadius     = std::max(bounds.radius(), 1e-3F) * settings.groundFade;
}

//--------------------------------------------------------------------------------------------------
// The haze (shaders/haze.h.slang): its extinction from the visibility, -ln(0.02) / V
// (hazeExtinctionFromVisibility), and the sphere it fills. Round the scene's bounds, its lights and
// the eye, so the eye is in it and a light's beam reaches the scene inside it, and nothing beyond: the
// sky is seen through as much haze as the scene holds. A scene unit is a metre.
void GltfRenderer::fillHaze(shaderio::SceneFrameInfo& finfo) const
{
  finfo.hazeExtinction     = 0.0F;
  const Settings& settings = m_resources.settings;
  if(!(settings.hazeVisibility > 0.0F))
    return;
  const SceneGpuView& view   = m_resources.gpuView();
  const glm::vec3     eye    = glm::vec3(finfo.viewInv[3]);
  glm::vec3           center = eye;
  float               radius = 0.0F;
  float               lowest = eye.y;
  if(view.hasContent())
  {
    const nvutils::Bbox bounds = view.bounds();
    if(!bounds.isEmpty())
    {
      center = bounds.center();
      radius = bounds.radius();
      lowest = bounds.min().y;
    }
    for(const SceneGpuView::LightInfo& light : view.lights())
      if(!light.directional)
        radius = std::max(radius, glm::distance(glm::vec3(light.worldMatrix[3]), center));
  }
  radius                 = std::max(radius, glm::distance(eye, center));
  finfo.hazeExtinction   = 3.912023F / settings.hazeVisibility;
  finfo.hazeAnisotropy   = std::clamp(settings.hazeAnisotropy, -0.9F, 0.9F);
  finfo.hazeCenter       = center;
  finfo.hazeRadius       = radius * 1.01F + 1e-3F;  // the eye strictly inside
  finfo.hazeCandidates   = std::clamp(settings.hazeCandidates, 1, 16);
  // Uneven: the floor is the ground's (a body of water's surface over it, fillGround's waterDepth),
  // or the scene's lowest point where there is none
  finfo.hazeFloor     = settings.useInfinitePlane ? settings.infinitePlaneDistance + finfo.waterDepth : lowest;
  finfo.hazeInvHeight = settings.hazeHeight > 0.0F ? 1.0F / settings.hazeHeight : 0.0F;
  finfo.hazePatches   = std::clamp(settings.hazePatches, 0.0F, 1.0F);
  const float patch   = settings.hazePatchSize > 0.0F ? settings.hazePatchSize : std::clamp(0.25F * radius, 0.5F, 8.0F);
  finfo.hazeInvPatchSize = 1.0F / patch;
}

//--------------------------------------------------------------------------------------------------
// Illuminance of the gradient sky's sun, before `envIntensity`.
//
// The physical sky measures its sun: the bake reports a disk radiance, and radiance times the solid
// angle the disk subtends is an illuminance. The gradient sky measures nothing. `sunColor` is the
// colour the disk is *drawn* in, and running it through that same product is what left this sun
// visible but dark -- a half-degree disk subtends 6.8e-5 sr, so a `sunColor` of 1 became 7e-5 lux
// under a dome delivering about 1.5. Forty thousand times too dim, and silently so: the disk was
// still there to look at.
//
// So the number is scaled against the sky instead of derived from the disk. The dome's own
// illuminance is the one quantity the gradient sky does define -- it is those authored colours over
// a hemisphere -- and `kGradientSunToSkyRatio` puts the sun where a clear day puts it relative to
// that. `sunColor` keeps its job as the sun's colour *and* its relative brightness, so brightening
// it in the UI now brightens the light as well as the disk.
//
// The sky half only: the ground half of the gradient lights nothing that faces up, and including it
// would make the sun follow the ground colour.
//
glm::vec3 GltfRenderer::gradientSunIlluminance() const
{
  const Settings& settings = m_resources.settings;

  // Illuminance the dome alone puts on an up-facing surface:
  //   E = 2*PI * integral(0..PI/2) L(v) cos(v) sin(v) dv
  // with L the same horizon -> top band the shader evaluates, and v measured from the zenith.
  // Numeric because the band shape is a free exponent; 256 steps settle this to well under a
  // percent, and it is evaluated once per frame on the CPU.
  constexpr int kSteps = 256;
  glm::vec3     dome(0.0f);
  for(int i = 0; i < kSteps; ++i)
  {
    const float     v = (static_cast<float>(i) + 0.5f) / static_cast<float>(kSteps) * glm::half_pi<float>();
    const float     c = 1.0f - v / glm::half_pi<float>();
    const glm::vec3 L =
        glm::mix(settings.gradientHorizonColor, settings.gradientTopColor, gradientBandWeight(c, settings.gradientTopCurve));
    dome += L * std::cos(v) * std::sin(v);
  }
  dome *= glm::two_pi<float>() * glm::half_pi<float>() / static_cast<float>(kSteps);

  // One scalar for the dome: the ratio is a brightness, and letting it run per channel would tint
  // the sun by the inverse of the sky it stands against.
  const float domeIlluminance = (dome.x + dome.y + dome.z) / 3.0f;
  return settings.gradientSunColor * (kGradientSunToSkyRatio * domeIlluminance);
}

//--------------------------------------------------------------------------------------------------
// The atmosphere the physical sky is built from: Earth, with the panel's settings laid over it.
//
shaderio::SkyAtmosphereParameters GltfRenderer::buildAtmosphereParameters() const
{
  const Settings& s = m_resources.settings;

  // Starts from the Earth defaults for the two fields no setting covers: muSMin, which is LUT
  // coverage rather than physics, and the zero padding layer every density profile needs.
  shaderio::SkyAtmosphereParameters p = skyBrunetonEarthDefaults();

  p.solarIrradiance    = s.atmoSolarIrradiance;
  p.sunAngularRadius   = s.atmoSunAngularRadius;
  p.groundAlbedo       = s.atmoGroundAlbedo;
  p.rayleighScattering = s.atmoRayleighScattering;
  p.mieScattering      = s.atmoMieScattering;
  p.miePhaseFunctionG  = s.atmoMieAnisotropy;

  // Extinction is what the aerosol removes; scattering is the part it gives back. The albedo is
  // the ratio, and it is what the setting exposes -- OMI has no absorption field, so a scene that
  // sets mieScattering keeps the ratio rather than inheriting an unrelated extinction.
  p.mieExtinction = s.atmoMieScattering / std::max(s.atmoMieAlbedo, 1e-3F);

  p.absorptionExtinction = s.atmoOzoneExtinction;
  p.bottomRadius         = s.atmoBottomRadius;
  p.topRadius            = s.atmoBottomRadius + s.atmoThickness;

  // Both gases fall off exponentially, so a scale height is the whole profile. Layer 0 stays the
  // zero padding upstream inserts -- see skyBrunetonEarthDefaults().
  p.rayleighDensity.layers[1].expTerm  = 1.0F;
  p.rayleighDensity.layers[1].expScale = -1.0F / std::max(s.atmoRayleighScaleHeight, 1e-3F);
  p.mieDensity.layers[1].expTerm       = 1.0F;
  p.mieDensity.layers[1].expScale      = -1.0F / std::max(s.atmoMieScaleHeight, 1e-3F);

  // Ozone is a tent: zero at centre-width/2, one at the centre, zero again at centre+width/2.
  // Layer 0 covers the rise and layer 1 the fall, so the split is at the centre.
  const float halfWidth                      = std::max(s.atmoOzoneWidth, 1e-3F) * 0.5F;
  const float slope                          = 1.0F / halfWidth;
  p.absorptionDensity.layers[0].width        = s.atmoOzoneCenter;
  p.absorptionDensity.layers[0].expTerm      = 0.0F;
  p.absorptionDensity.layers[0].expScale     = 0.0F;
  p.absorptionDensity.layers[0].linearTerm   = slope;
  p.absorptionDensity.layers[0].constantTerm = 1.0F - s.atmoOzoneCenter * slope;
  p.absorptionDensity.layers[1].expTerm      = 0.0F;
  p.absorptionDensity.layers[1].expScale     = 0.0F;
  p.absorptionDensity.layers[1].linearTerm   = -slope;
  p.absorptionDensity.layers[1].constantTerm = 1.0F + s.atmoOzoneCenter * slope;

  return p;
}

//--------------------------------------------------------------------------------------------------
// Push the current settings into EnvBaker -- the parameters both the preview and the commit bake
// from.
//
// Separate from refreshBakedEnvironment because the preview path needs it too, and needs only
// this: these are member assignments, uploaded by the bake through the command buffer, so they
// are safe in the middle of recording a frame. setResolution is deliberately *not* here -- it
// reallocates the image, which is a frame-boundary operation and belongs to the commit alone.
//
// The atmosphere block is not here either: the bake reads it for ground albedo and the sun's
// angular size, but any change to it also fails SkyBruneton::lutsSatisfy, and the rebuild that
// follows uploads it.
//
void GltfRenderer::updateBakerParameters()
{
  // Metres in the UI, kilometres in the model.
  m_envBaker.setObserverAltitude(m_resources.settings.atmoObserverAltitude * 0.001F);

  m_envBaker.skyParams() = buildSkyOmiParameters();
  m_envBaker.setSkyType(shaderio::SkyType(bakeSkyType(m_resources.settings.envSystem)));
  m_envBaker.setSunDirection(sunInEnvironmentFrame());

  // The clouds bake with the sky, so they take their settings at the same moment.
  const Settings& s = m_resources.settings;
  m_skyClouds.setSettings({.coverage    = s.cloudCoverage,
                         .thicknessKm = s.cloudThickness,
                         .seed        = uint32_t(std::max(s.cloudSeed, 0)),
                         .type        = s.cloudType,
                         .midCoverage  = s.cloudMidCoverage,
                         .highCoverage = s.cloudHighCoverage});
}

//--------------------------------------------------------------------------------------------------
// The physical sky's clouds where the camera looks, at the screen's resolution. The sky image's
// texels are 7 pixels each through an 85 mm lens on a 1400-pixel-high view, so their edges smear.
// Baked once the camera has held still a frame, and again whenever a bake of the sky leaves the view
// image behind: 16 ms for a fair-weather cumulus in a 2160 x 1395 view, 49 ms for a sunset under
// three tiers, against the sky bake's 145 (RTX 4080, 2026-11-06). While the camera moves, the view
// image keeps serving the directions it covers, and the sky image the rest (cloudViewShare).
//
bool GltfRenderer::updateCloudView(VkCommandBuffer cmd)
{
  std::optional<SkyCloudView> wanted;
  if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky && !m_resources.settings.useSolidBackground && m_skyClouds.viewable())
    wanted = cloudViewFor(m_cameraManip->getViewMatrix(), m_cameraManip->getPerspectiveMatrix(),
                          m_resources.settings.envRotationQuat(),
                          glm::uvec2(m_resources.gBuffers.getSize().width, m_resources.gBuffers.getSize().height));
  const bool settled    = wanted.has_value() && wanted == m_lastWantedCloudView;
  m_lastWantedCloudView = wanted;
  if(!settled || m_skyClouds.viewHolds(*wanted))
    return false;
  m_skyClouds.recordView(cmd, *wanted);
  return true;
}

//--------------------------------------------------------------------------------------------------
// Rebuild the baked lighting environment from the current settings.
//
// This is the single point where an analytic sky becomes the environment the path tracer
// samples. After it runs, hdrIbl holds a lat-long image and an alias table built from EnvBaker's
// output instead of from an HDR file -- which is why the path tracer needs no per-sky-type branch
// to light a scene.
//
// Application thread only: EnvBaker::commit submits and waits.
//
void GltfRenderer::refreshBakedEnvironment()
{
  m_envBakeDirty = false;
  // A full commit supersedes any preview raised before it, and the commit that preview would owe.
  m_envPreviewPending = false;
  m_envCommitOwed     = false;
  if(!isBakedEnvironment(m_resources.settings.envSystem))
    return;

  // Both the bake and the descriptor rewrite below touch resources that in-flight frames are
  // still reading: the bake overwrites the lat-long image, and updateHdrImages() rewrites set 0
  // while a previously submitted frame may still be sampling it. createHDR() opens with
  // the same wait for the same reason -- without it this crashes, not merely trips validation.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  const Settings& settings = m_resources.settings;

  // Rebuild the LUTs unless they already hold this atmosphere at full quality. Compared by value
  // rather than tracked with a dirty flag so that an .ini restore, a benchmark script and an MCP
  // write are all covered without each having to remember to raise one -- and so that a settle
  // that changed nothing does not pay for a full precompute.
  //
  // Quality is part of the test, not just the parameters: a drag leaves the LUTs holding the right
  // atmosphere at preview quality, and this is the moment that owes the upgrade.
  if(settings.envSystem == shaderio::EnvSystem::eSky)
  {
    const shaderio::SkyAtmosphereParameters wanted = buildAtmosphereParameters();
    if(!m_skyBruneton.lutsSatisfy(wanted, SkyBruneton::kQualityFinal))
    {
      m_skyBruneton.setParameters(wanted);
      m_skyBruneton.precompute(m_resources);
    }
  }

  updateBakerParameters();
  m_envBaker.commit(m_resources);
  m_envOwnedByBaker = true;

  // Rebind the environment into the renderer's own descriptor set. HdrIbl::updateFromGpuImage
  // repoints HdrIbl's internal set, which is what the path tracer binds for the alias table --
  // but `texturesHdr[HDR_IMAGE_INDEX]` lives in set 0 here and still names whichever image was
  // bound last (at start-up, the dummy from createHDR("")). Without this the scene is lit and
  // backed by that stale image and the authored sky is silently ignored.
  {
    nvvk::WriteSetContainer    write{};
    const VkWriteDescriptorSet hdrTexture =
        m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr, m_resources.descriptorSet,
                                                     HDR_IMAGE_INDEX, 1U);
    write.append(hdrTexture, m_resources.hdrIbl.getHdrImage());
    vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
  }

  // A baked environment has its own luminance integral, exactly like a loaded HDR -- but this runs
  // on every commit, and a commit happens every time the sun moves or an atmosphere slider settles.
  //
  // So calibrate once, on the first commit after the environment changes kind, and then leave it
  // alone. Two reasons, and the second is the important one:
  //
  //   - The value would otherwise swing with the sun, because the sky's integral does.
  //   - It is a *starting point*, not a derived quantity. The clamp has to accommodate every light
  //     in the scene -- emissive materials, punctual lights, a bright HDR backplate -- and the
  //     environment integral knows about none of them. Once the user has adjusted it, recomputing
  //     from the environment alone throws that away, and does so on a gesture as ordinary as
  //     dragging the sun.
  //
  // This is where the calibration belongs rather than at the switch itself: hdrIbl's integral does
  // not describe the new environment until the bake above has run.
  if(m_fireflyClampCalibratedFor != int(settings.envSystem))
  {
    calibrateFireflyClamp();
  }
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Returns the scene-appropriate firefly clamp threshold:
//   - HDR or baked analytic environment → use its luminance integral (already calibrated to the
//     environment's range; a baked sky populates the same integral through updateFromGpuImage)
//   - Otherwise → a fixed baseline high enough to preserve legitimately bright highlights
void GltfRenderer::calibrateFireflyClamp()
{
  m_fireflyClampCalibratedFor = int(m_resources.settings.envSystem);
  // a host owns the clamp (ptFireflyClamp, from its exposure): an environment's kind changing must not take it.
  // Nor may a scene's environment take the command line's: a scripted run that asks for a clamp, or for none
  // (0), renders with it, as --envSystem wins over a scene's sky.
  const bool fromCommandLine = m_parameterParser && m_parameterParser->wasParsed("ptFireflyClamp");
  if((!m_resources.hostFed || m_resources.sceneFromFile) && !fromCommandLine)
    m_pathTracer.m_pushConst.fireflyClampThreshold = defaultFireflyClamp();
}

//--------------------------------------------------------------------------------------------------
// Where aerial perspective stands the eye. See the declaration for why this is not the sky's
// observer altitude.
//
float GltfRenderer::aerialPerspectiveEyeAltitudeKm() const
{
  const float liftKm = m_resources.settings.atmoObserverAltitude * 0.001F;

  if(!m_resources.gpuView().hasContent() || m_cameraManip == nullptr)
    return liftKm;

  const nvutils::Bbox bounds = m_resources.gpuView().bounds();
  if(!bounds.isEmpty())
  {
    // The scene's floor goes `liftKm` above the planet's surface, so nothing in it is ever
    // underground however the camera moves; the eye then stands wherever it stands above that
    // floor. A camera below the floor -- inside the terrain, or under a ground plane -- would
    // otherwise push itself under the surface instead, so the lift is the floor for that too.
    const float kmPerSceneUnit = 0.001F * m_resources.settings.atmoAerialPerspectiveScale;
    // The camera manipulator works in doubles; the difference is a scene-scale number, so it fits a
    // float long before it is turned into kilometres.
    const float aboveFloor = static_cast<float>(m_cameraManip->getEye().y - bounds.min().y) * kmPerSceneUnit;
    return liftKm + std::max(aboveFloor, 0.0F);
  }
  return liftKm;
}

glm::vec3 GltfRenderer::skySunIlluminanceAboveAtmosphere() const
{
  // The constant is written in shader types; float3 is an alias that lives in `shaderio`.
  using shaderio::float3;
  // The sun's factor set, not the sky's -- this is the sun. See SUN_SPECTRAL_TO_LUMINANCE.
  const glm::vec3 radianceToLuminance = SUN_RADIANCE_TO_LUMINANCE;
  return buildAtmosphereParameters().solarIrradiance * radianceToLuminance * m_resources.settings.hdrEnvIntensity;
}

float GltfRenderer::defaultFireflyClamp() const
{
  const shaderio::EnvSystem env = m_resources.settings.envSystem;
  if(env != shaderio::EnvSystem::eHdr && !isBakedEnvironment(env))
    return 30.0f;

  // The environment's own integral: the sum of max(R,G,B) over solid angle.
  float clamp = m_resources.hdrIbl.getIntegral();

  // The physical sky keeps its sun *out* of that image -- the sun is a directional light so that
  // its disk stays sharp and next-event estimation can sample it -- so the integral measures the
  // sky alone, and for a daylit scene the sky is the smaller half by a wide margin. An .hdr file
  // has its sun baked in and needs no correction, which is why this only surfaced once the
  // analytic skies grew a sun of their own.
  //
  // Added in the integral's own units: a disk of radiance L covering solid angle W contributes
  // max(L) * W to that sum, and radiance times solid angle is illuminance. So the term is the
  // largest channel of the sun's illuminance.
  //
  // Which sun, though -- the one above the atmosphere, not the one currently shining. The
  // attenuated sun is the physically honest answer to "how much light is there right now", and it
  // is the wrong answer here, because it goes to zero at dusk and takes the clamp with it: at 2
  // degrees below the horizon this threshold was 0.34, at 5 degrees 0.03, and at 10 degrees it
  // reached exactly 0 -- which the shader reads as "disabled", so the clamp flipped from crushing
  // every sample to not clamping at all. A clamp is a safety net against outliers, and a safety
  // net that scales itself to nothing at night is worse than a fixed one.
  //
  // The unattenuated value is a property of the star rather than of the hour, so it is stable
  // across a whole time-of-day scrub, and it still follows the atmosphere: Mars' sun is dimmer
  // than Earth's, and editing the solar irradiance moves it.
  //
  // The gradient sky keeps its sun out of the bake for the same reason and needs the same
  // correction; there is just no atmosphere to ask, so the term is the sun this sky implies.
  glm::vec3 sun(0.0f);
  if(env == shaderio::EnvSystem::eSky)
    sun = skySunIlluminanceAboveAtmosphere();
  else if(env == shaderio::EnvSystem::eGradient)
    sun = gradientSunIlluminance() * m_resources.settings.hdrEnvIntensity;
  clamp += std::max({sun.x, sun.y, sun.z});
  return clamp;
}

//--------------------------------------------------------------------------------------------------
// Load an environment map / scene by path.
//
// These back the --hdrfile / --scenefile parameters, which carry a callbackSuccess so that setting
// them does the load -- from the command line or nvpro_set_parameters alike
// (nvmcp invokes the callback on the application thread). Before that, writing either was accepted
// and did nothing, because both were only read once during start-up.
//
// Path resolution is delegated to createHDR / createScene: both call nvutils::findFile against
// nvsamples::getResourcesDirs() so bare resource-relative inputs (e.g. `--hdrfile studio.hdr`
// where studio.hdr ships under the sample's resources dir) work exactly like a cwd-relative or
// absolute path. Reporting success/failure is likewise the loaders' job: HdrIbl::isValid() flips
// off with a warning log when its input cannot be loaded -- so we no longer need a pre-flight
// std::filesystem::exists() gate here that would reject resource-relative inputs the loader would otherwise have found.
bool GltfRenderer::loadHdrEnvironment(const std::filesystem::path& filename)
{
  // Not attached yet: this is the start-up command-line parse, and main() performs that load
  // itself once the device exists.
  if(!m_app)
    return false;

  createHDR(filename);
  if(!m_resources.hdrIbl.isValid())
    return false;

  // Loading an environment the renderer is not told to sample would be another silent no-op.
  m_resources.settings.envSystem = shaderio::EnvSystem::eHdr;
  // A new HDR changes hdrIbl.getIntegral(), which defaultFireflyClamp() uses when envSystem is
  // eHdr. Refresh the path-tracer clamp in lock-step -- so a --hdrfile change from an MCP write
  // doesn't keep clamping against the previous HDR's integral.
  calibrateFireflyClamp();
  resetFrame();
  return true;
}

bool GltfRenderer::loadHdrEnvironment(EnvImage&& image, const std::filesystem::path& name)
{
  if(!m_app || !image.valid())
    return false;
  LOGI("HDR environment: %s (%ux%u)\n", name.empty() ? "(generated)" : nvutils::utf8FromPath(name).c_str(),
       image.size.width, image.size.height);
  uploadHdrEnvironment(std::move(image), name);
  if(!m_resources.hdrIbl.isValid())
    return false;
  m_resources.settings.envSystem = shaderio::EnvSystem::eHdr;
  calibrateFireflyClamp();
  resetFrame();
  return true;
}

//--------------------------------------------------------------------------------------------------
// Bind the HDR image (RGBA32F) the path tracer samples
//
void GltfRenderer::updateHdrImages()
{
  nvvk::WriteSetContainer    write{};
  const VkWriteDescriptorSet hdrTexture = m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr,
                                                                                      m_resources.descriptorSet, HDR_IMAGE_INDEX, 1U);
  write.append(hdrTexture, m_resources.hdrIbl.getHdrImage());
  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Create or load the HDR environment map
// If the filename is empty, a default environment map (empty) is created, which allow the descriptor set to be updated
void GltfRenderer::createHDR(const std::filesystem::path& hdrFilename)
{
  // Load an HDR and create the important sampling acceleration structure
  std::filesystem::path filename;
  if(!hdrFilename.empty())
  {
    if(std::filesystem::exists(hdrFilename))
      filename = hdrFilename;
    else
      filename = nvutils::findFile(hdrFilename, nvsamples::getResourcesDirs(), false);
  }
  // Decode here rather than through HdrIbl's path overload, so .hdr and .exr take one route --
  // see env_image_loader.hpp for why EXR cannot live inside nvvk. An empty or failed decode hands
  // over an empty span, which is what produces the dummy environment at start-up.
  EnvImage envImage = loadEnvImage(filename);

  // Say which file this is, or that there is none. A request that does not resolve, or an image
  // that does not decode, silently becomes the dummy environment above -- and a dummy is easy to
  // mistake for a real one, because plenty of studio HDRs genuinely are flat and achromatic.
  // Without a line here the only way to tell them apart is to know what the file looks like.
  if(hdrFilename.empty())
  {
    LOGI("HDR environment: none (dummy)\n");
  }
  else if(envImage.pixels.empty())
  {
    LOGW("HDR environment: %s could not be %s; using the dummy environment\n",
         nvutils::utf8FromPath(hdrFilename).c_str(), filename.empty() ? "found" : "decoded");
  }
  else
  {
    LOGI("HDR environment: %s (%ux%u)\n", nvutils::utf8FromPath(filename).c_str(), envImage.size.width, envImage.size.height);
  }

  const std::filesystem::path source = envImage.pixels.empty() ? std::filesystem::path{} : filename;
  uploadHdrEnvironment(std::move(envImage), source);
}

//--------------------------------------------------------------------------------------------------
// Install an already-decoded lat-long as the HDR environment. The decode is the slow half of an
// HDR load; split out so a caller can run it on a worker and only pay for the upload here.
void GltfRenderer::uploadHdrEnvironment(EnvImage&& envImage, const std::filesystem::path& sourceFile)
{
  // An HDRI reload can happen mid-frame. Wait until in-flight work finishes before tearing down
  // env images the path tracer may still be sampling.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
  // Each piece goes up whole, so the staging block holds the largest: the image, or its compensated
  // alias table, a cell a texel and one more a row. Sized from the image, not a fixed 1 GiB (an 8K
  // RGBA32F) that every 2K load made and destroyed as well.
  const VkDeviceSize   imageBytes = VkDeviceSize(envImage.pixels.size()) * sizeof(float);
  nvvk::FrameUploader uploader;
  NVVK_CHECK(uploader.init({
      .allocator = &m_resources.allocator,
      .blockSize = std::max<VkDeviceSize>(imageBytes + imageBytes / 4, 16ull * 1024 * 1024),
      .debugName = "hdrUpload",
  }));

  // An image with no file behind it (a generated studio) is kept, so leaving a baked sky for the
  // HDR path can put it back -- the reload there has nothing else to read. HdrIbl writes into the
  // pixels it is given, hence a copy taken before the upload.
  if(sourceFile.empty() && envImage.valid())
    m_lastHdrImage = std::make_shared<const EnvImage>(envImage);
  else
    m_lastHdrImage.reset();

  m_resources.hdrIbl.destroyEnvironment();
  m_resources.hdrIbl.loadEnvironment(cmd, uploader, std::span<float>(envImage.pixels), envImage.size, true);
  // Remembered so that switching to a baked sky and back to eHdr can rebind hdrIbl to this file:
  // EnvBaker::commit repoints hdrIbl at its own image, and only a reload undoes that. The resolved
  // path, not the request: a bare name found through the resources directories would otherwise be
  // saved as a panorama URI that resolves nowhere. Empty when nothing decoded -- the dummy is not a
  // panorama, and saving it as one would describe an environment that does not exist.
  m_lastHdrFile     = sourceFile;
  m_envOwnedByBaker = false;

  uploader.cmdUploadAppended(cmd);

  // Generate mipmaps for the HDR image
  VkExtent2D hdrSize = m_resources.hdrIbl.getHdrImageSize();
  if(hdrSize.width > 1 && hdrSize.height > 1)
  {
    nvvk::cmdGenerateMipmaps(cmd, m_resources.hdrIbl.getHdrImage().image, hdrSize, nvvk::mipLevels(hdrSize));
  }

  nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
  uploader.deinit();

  updateHdrImages();
}
