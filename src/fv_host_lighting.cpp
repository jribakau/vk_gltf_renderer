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

//
// FvHost: the Light tab (fv_lighting.hpp), the environment and the host's sun.
//

#include "fv_host_impl.hpp"

Json FvHost::Impl::lightingTopic() const
{
  Json t           = fv::lighting::toJson(lighting);
  t["env"]["name"]    = envName;
  t["env"]["size"]    = {envSize.x, envSize.y};
  t["env"]["loading"] = envLoading;
  // where the HDR's own sun is, as the panel's sun angles at the current rotation
  if(envSun && lighting.env.source == "hdr")
  {
    const glm::dvec2 a = fv::lighting::environmentSunAngles(*envSun, lighting.env.rotation);
    t["env"]["sun"]    = {{"azimuth", std::round(a.x * 10.0) / 10.0}, {"elevation", std::round(a.y * 10.0) / 10.0}};
  }
  else
    t["env"]["sun"] = nullptr;
  return t;
}

void FvHost::Impl::publishLighting()
{
  const Json now = lightingTopic();
  if(publishedLighting.is_null())
    protocol->events().snapshot("lighting", now);
  else if(Json diff = fv::lighting::mergeDiff(publishedLighting, now); !diff.empty())
    protocol->events().patch("lighting", diff);
  publishedLighting = now;
}

Json FvHost::Impl::lightingSet(const Json& patch)
{
  fv::lighting::State next = fv::lighting::applyPatch(lighting, patch);  // throws, leaving everything as it was
  if(next.env.source == "hdr" && next.env.path.empty())
    throw std::invalid_argument("lighting: env.path is required for an hdr environment");
  lighting = std::move(next);
  fv::timing::Scope t("lighting.set", 20);
  applyLighting(false);
  return lightingTopic();
}

// Writes what changed since the last call (everything, the first time) to the renderer: its
// parameters, the sun light, and the environment image, which loads as a job.
void FvHost::Impl::applyLighting(bool everything)
{
  if(everything)
  {
    appliedParams = Json::object();
    appliedSun    = nullptr;
  }
  {
    fv::timing::Scope t("requestEnvironment", 20);
    requestEnvironment();
  }

  Json params                     = fv::lighting::rendererParams(lighting, envSize);
  params["infinitePlaneDistance"] = lighting.ground.autoHeight ? double(autoGround) : lighting.ground.height;
  if(const auto still = appliedParams.find("ptFireflyClampStill"); still != appliedParams.end() && still->is_number())
    params["ptFireflyClampStill"] = fv::lighting::stillFireflyClamp(params["ptFireflyClampStill"].get<double>(), still->get<double>());
  Json changed                    = Json::object();
  for(auto it = params.begin(); it != params.end(); ++it)
    if(!appliedParams.contains(it.key()) || appliedParams[it.key()] != it.value())
      changed[it.key()] = it.value();
  if(!changed.empty())
  {
    std::string keys;
    for(auto it = changed.begin(); it != changed.end(); ++it)
      keys += (keys.empty() ? "" : ",") + it.key();
    fv::timing::Scope t("setParams " + keys, 20);
    setParams(changed);
    appliedParams.update(changed);
  }
  {
    fv::timing::Scope t("applySun", 20);
    applySun(everything);
  }
  publishLighting();
}

// The environment image the state asks for, if the renderer doesn't have it: the studio is built
// here, an HDR decodes on a worker. Each new choice bumps the request, so a load that finishes
// after a newer one is dropped.
void FvHost::Impl::requestEnvironment()
{
  const fv::lighting::Env& e    = lighting.env;
  const std::string        want = e.source == "studio" ? "studio" : e.source == "hdr" ? "hdr:" + e.path : "";
  if(want == wantedEnv)
    return;
  wantedEnv           = want;
  const uint64_t req  = ++envRequest;
  const bool     wasLoading = envLoading;
  envLoading          = false;
  if(wasLoading && envJob)
    job(envJob, {{"state", "done"}});
  if(want.empty() || want == loadedEnv)
  {
    // a sky, a plain colour, none, or the image already there: the envSystem parameter does it
    if(!want.empty())
      appliedParams.erase("envSystem");
    return;
  }
  if(want == "studio")
  {
    EnvImage img;
    img.pixels = fv::lighting::studioImage(kStudioWidth, kStudioHeight);
    img.size   = {kStudioWidth, kStudioHeight};
    if(renderer->loadHdrEnvironment(std::move(img), {}))
    {
      loadedEnv = want;
      envSun.reset();
      envName   = kStudioName;
      envSize   = {kStudioWidth, kStudioHeight};
      appliedParams.erase("ptFireflyClamp");  // an environment load recalibrates it; applyLighting sends ours
    }
    return;
  }
  const std::string path  = e.path;
  const std::string label = "Loading " + utf8Of(pathOf(path).filename());
  envLoading              = true;
  envJob                  = ++jobs;
  job(envJob, {{"label", label}, {"state", "running"}});
  loaders.spawn([this, path, want, req, jobId = envJob, label] {
    auto image = std::make_shared<EnvImage>(loadEnvImage(pathOf(path)));
    // before the renderer takes the pixels: it writes the sampling pdf into alpha
    const std::optional<glm::vec3> sun =
        image->valid() ? fv::lighting::environmentSun(image->pixels.data(), image->size.width, image->size.height) : std::nullopt;
    post([this, image, path, want, req, jobId, label, sun] {
      if(req != envRequest)
        return;  // another environment was chosen meanwhile, and closed this job
      envLoading = false;
      const glm::uvec2 size{image->size.width, image->size.height};
      if(!image->valid() || !renderer->loadHdrEnvironment(std::move(*image), pathOf(path)))
      {
        job(jobId, {{"label", label}, {"state", "failed"}, {"error", "could not read " + path}});
        notice("Could not read the environment " + path, "error");
        // the panel shows what is on screen: the HDR that was there before, else the studio
        if(loadedEnv.rfind("hdr:", 0) == 0)
        {
          wantedEnv         = loadedEnv;
          lighting.env.path = loadedEnv.substr(4);
          publishLighting();
        }
        else
        {
          lighting.env.source = "studio";
          lighting.env.path.clear();
          applyLighting(false);
        }
        return;
      }
      loadedEnv = want;
      envSun    = sun;
      envName   = utf8Of(pathOf(path).filename());
      envSize   = size;
      job(jobId, {{"label", label}, {"state", "done"}});
      appliedParams.erase("envSystem");
      appliedParams.erase("ptFireflyClamp");  // the load recalibrated it
      applyLighting(false);  // hdrBlur depends on the image's size
    });
  });
}

// The sun light's colour, strength and size, and its entity's rotation, when they changed.
void FvHost::Impl::applySun(bool force)
{
  if(!renderer->feedsHostScene())
    return;
  rs::RetainedScene& scene = renderer->hostScene();
  if(!scene.light(sunLight) || !scene.entity(sunEntity))
    return;
  const fv::lighting::Sun& s     = lighting.sun;
  const glm::vec3          color = fv::lighting::linearColor(s.color);
  const Json               light = {{"color", {color.x, color.y, color.z}},
                                    {"intensity", s.enabled ? s.intensity : 0.0},
                                    {"radius", fv::lighting::sunRadiusKm(s.softness)}};
  const Json               aim   = {s.azimuth, s.elevation};
  if(force || appliedSun.is_null() || appliedSun["light"] != light)
    scene.setLight(sunLight, sunPayload());
  if(force || appliedSun.is_null() || appliedSun["aim"] != aim)
    scene.setLocal(sunEntity, {.rotation = fv::lighting::sunRotation(s.azimuth, s.elevation)});
  appliedSun = {{"light", light}, {"aim", aim}};
}

// The "Sun" light as the current state has it: directional, its softness the sun's radius
tinygltf::Light FvHost::Impl::sunPayload() const
{
  const fv::lighting::Sun& s     = lighting.sun;
  const glm::vec3          color = fv::lighting::linearColor(s.color);
  tinygltf::Light          light;
  light.name      = "Sun";
  light.type      = "directional";
  light.color     = {color.x, color.y, color.z};
  light.intensity = s.enabled ? s.intensity : 0.0;
  tinygltf::Value::Object extras;
  extras["radius"] = tinygltf::Value(fv::lighting::sunRadiusKm(s.softness));
  light.extras     = tinygltf::Value(extras);
  return light;
}

// The sun on a root entity of the scene about to be made resident, written with the current state
// and marked as the sky's sun, so a physical sky follows it.
void FvHost::Impl::feedSun(rs::RetainedScene& scene)
{
  const fv::lighting::Sun& s = lighting.sun;
  sunEntity                  = scene.createEntity("Sun", {0, -1});
  scene.setLocal(sunEntity, {.rotation = fv::lighting::sunRotation(s.azimuth, s.elevation)});
  sunLight = scene.addLight({.owner = {0, -1}, .payload = sunPayload(), .skySun = true});
  scene.setEntityLight(sunEntity, sunLight);
  scene.release(sunLight);
  const glm::vec3 color = fv::lighting::linearColor(s.color);
  appliedSun            = {{"light",
                            {{"color", {color.x, color.y, color.z}},
                             {"intensity", s.enabled ? s.intensity : 0.0},
                             {"radius", fv::lighting::sunRadiusKm(s.softness)}}},
                           {"aim", {s.azimuth, s.elevation}}};
}

// A batch of renderer parameters, validated whole before any is applied; callbacks fire as over MCP.
void FvHost::Impl::setParams(const Json& values)
{
  if(!registry || values.empty())
    return;
  const auto params = exposedParameters(registry);
  toolJson(nvmcp::detail::setParameters(params, Json{{"values", values}}.dump()));
}
