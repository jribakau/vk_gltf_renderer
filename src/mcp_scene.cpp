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

#ifdef USE_NVMCP

#include "mcp_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <tinygltf/json.hpp>  // nlohmann::json
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

#include "gltf_scene_animation.hpp"
#include "gltf_scene_editor.hpp"
#include "renderer.hpp"
#include "tinygltf_utils.hpp"
#include "ui_animation.hpp"

using Json = nlohmann::json;

namespace {

//--------------------------------------------------------------------------------------------------
// Id maps: host ids (extras.fvId) -> indices, rebuilt per scene load / structural change.

struct PrimRef
{
  int mesh         = -1;
  int primitive    = -1;
  int renderPrimID = -1;
};

struct SceneIds
{
  const void* scene      = nullptr;
  uint64_t    generation = ~0ull;
  size_t      nodeCount = 0, materialCount = 0, primCount = 0;

  std::unordered_map<std::string, int>                  nodes;
  std::unordered_map<std::string, int>                  materials;
  std::unordered_map<std::string, std::vector<PrimRef>> prims;
  std::vector<std::vector<int>>                         renderPrimsOfMesh;  // mesh -> render prim ids
  std::unordered_set<int>                               joints;
};

std::string fvIdOf(const tinygltf::Value& extras)
{
  if(!extras.IsObject() || !extras.Has("fvId"))
    return {};
  const tinygltf::Value& v = extras.Get("fvId");
  return v.IsString() ? v.Get<std::string>() : std::string{};
}

void rebuildIds(SceneIds& ids, nvvkgltf::Scene& scene, uint64_t generation)
{
  const tinygltf::Model& model = scene.getModel();
  ids.scene                      = &scene;
  ids.generation                 = generation;
  ids.nodeCount                  = model.nodes.size();
  ids.materialCount              = model.materials.size();
  ids.primCount                  = scene.getRenderPrimitives().size();
  ids.nodes.clear();
  ids.materials.clear();
  ids.prims.clear();
  ids.joints.clear();
  ids.renderPrimsOfMesh.assign(model.meshes.size(), {});

  for(size_t i = 0; i < model.nodes.size(); i++)
    if(std::string id = fvIdOf(model.nodes[i].extras); !id.empty())
      ids.nodes.emplace(std::move(id), int(i));
  for(size_t i = 0; i < model.materials.size(); i++)
    if(std::string id = fvIdOf(model.materials[i].extras); !id.empty())
      ids.materials.emplace(std::move(id), int(i));

  const auto& renderPrims = scene.getRenderPrimitives();
  for(size_t rp = 0; rp < renderPrims.size(); rp++)
  {
    const auto& prim = renderPrims[rp];
    if(prim.meshID < 0 || prim.meshID >= int(model.meshes.size()) || !prim.pPrimitive)
      continue;
    ids.renderPrimsOfMesh[prim.meshID].push_back(int(rp));
    const auto& mesh = model.meshes[prim.meshID];
    int         p    = -1;
    for(size_t k = 0; k < mesh.primitives.size(); k++)
      if(&mesh.primitives[k] == prim.pPrimitive)
        p = int(k);
    if(std::string id = fvIdOf(prim.pPrimitive->extras); !id.empty())
      ids.prims[id].push_back({prim.meshID, p, int(rp)});
  }
  for(const auto& skin : model.skins)
    ids.joints.insert(skin.joints.begin(), skin.joints.end());
}

//--------------------------------------------------------------------------------------------------
// Small JSON helpers. nvmcp does not enforce input schemas, so every read is checked.

struct OpError
{
  std::string message;
};

[[noreturn]] void fail(std::string message)
{
  throw OpError{std::move(message)};
}

//--------------------------------------------------------------------------------------------------
// What a tool reads: the renderer, its scene and the ids. A tool returns a JSON result and throws
// OpError on bad input.

struct Ctx
{
  GltfRenderer&    renderer;
  nvvkgltf::Scene& scene;
  SceneIds&        ids;
};

Json animationState(Ctx& c)
{
  AnimationControl& ctrl = c.renderer.editAnimationControl();
  auto&             anim = c.scene.animation();
  Json              out  = {{"clips", anim.getNumAnimations()}, {"clip", ctrl.currentAnimation}, {"playing", ctrl.play},
                            {"speed", ctrl.speed},           {"loop", ctrl.loop}};
  if(ctrl.currentAnimation >= 0 && ctrl.currentAnimation < anim.getNumAnimations())
  {
    const auto& info = anim.getAnimationInfo(ctrl.currentAnimation);
    out["name"]      = info.name;
    out["time"]      = info.currentTime;
    out["start"]     = info.start;
    out["end"]       = info.end;
  }
  return out;
}

Json sceneInfo(Ctx& c)
{
  const tinygltf::Model& model = c.scene.getModel();
  const auto&            anim  = c.scene.animation();
  Json                   nodes = Json::array();
  for(size_t i = 0; i < model.nodes.size(); i++)
  {
    const auto& n  = model.nodes[i];
    std::string id = fvIdOf(n.extras);
    nodes.push_back({{"index", i},
                     {"id", id.empty() ? Json(nullptr) : Json(id)},
                     {"name", n.name},
                     {"mesh", n.mesh},
                     {"skin", n.skin},
                     {"light", n.light},
                     {"visible", tinygltf::utils::getNodeVisibility(n).visible}});
  }
  Json materials = Json::array();
  for(size_t i = 0; i < model.materials.size(); i++)
  {
    const auto& m  = model.materials[i];
    std::string id = fvIdOf(m.extras);
    materials.push_back({{"index", i}, {"id", id.empty() ? Json(nullptr) : Json(id)}, {"name", m.name}, {"alphaMode", m.alphaMode}});
  }
  std::unordered_set<int> skinned, morphed;
  for(const auto& t : anim.getSkinTasks())
  {
    skinned.insert(t.renderPrimID);
    skinned.insert(t.renderPrimIDs.begin(), t.renderPrimIDs.end());
  }
  for(uint32_t p : anim.getMorphPrimitives())
  {
    morphed.insert(int(p));
    const std::span<const int> members = c.scene.vertexSetMembers(int(p));
    morphed.insert(members.begin(), members.end());
  }
  Json        meshes      = Json::array();
  const auto& renderPrims = c.scene.getRenderPrimitives();
  for(size_t m = 0; m < model.meshes.size(); m++)
  {
    Json prims = Json::array();
    for(int rp : c.ids.renderPrimsOfMesh[m])
    {
      const auto& prim = renderPrims[rp];
      std::string id   = prim.pPrimitive ? fvIdOf(prim.pPrimitive->extras) : std::string{};
      prims.push_back({{"renderPrim", rp},
                       {"id", id.empty() ? Json(nullptr) : Json(id)},
                       {"material", prim.pPrimitive ? prim.pPrimitive->material : -1},
                       {"vertexCount", prim.vertexCount},
                       {"targets", prim.pPrimitive ? prim.pPrimitive->targets.size() : 0},
                       {"skinned", skinned.count(rp) != 0},
                       {"morphed", morphed.count(rp) != 0},
                       {"blasUpdatable", c.renderer.isBlasUpdatable(uint32_t(rp))}});
    }
    meshes.push_back({{"index", m}, {"name", model.meshes[m].name}, {"weights", model.meshes[m].weights.size()}, {"primitives", prims}});
  }
  Json animations = Json::array();
  for(int i = 0; i < anim.getNumAnimations(); i++)
  {
    const auto& info = anim.getAnimationInfo(i);
    animations.push_back({{"index", i}, {"name", info.name}, {"start", info.start}, {"end", info.end}});
  }
  Json lights = Json::array();
  for(size_t i = 0; i < model.lights.size(); i++)
    lights.push_back({{"index", i}, {"type", model.lights[i].type}, {"color", model.lights[i].color}, {"intensity", model.lights[i].intensity}});
  return {{"editable", c.renderer.isEditable()},
          {"nodes", nodes},
          {"materials", materials},
          {"meshes", meshes},
          {"animations", animations},
          {"lights", lights},
          {"animation", animationState(c)}};
}

}  // namespace

//--------------------------------------------------------------------------------------------------

void registerSceneMcpTools(nvmcp::Element& mcp, const std::shared_ptr<GltfRenderer>& renderer)
{
  const std::weak_ptr<GltfRenderer> weak = renderer;
  auto                              ids  = std::make_shared<SceneIds>();

  // Runs `body` with a resolved context, turning bad input into a tool error.
  auto withScene = [weak, ids](std::string_view payload, auto&& body) -> nvmcp::ToolResult {
    auto r = weak.lock();
    if(!r)
      return nvmcp::ToolResult::error(R"({"error":"renderer is shutting down"})");
    nvvkgltf::Scene* scene = r->editScene();
    if(!scene || !scene->valid())
      return nvmcp::ToolResult::error(R"({"error":"no scene is loaded"})");
    const Json input = payload.empty() ? Json::object() : Json::parse(payload, nullptr, /*allow_exceptions=*/false);
    if(input.is_discarded() || !input.is_object())
      return nvmcp::ToolResult::error(R"({"error":"arguments must be a JSON object"})");
    const tinygltf::Model& model = scene->getModel();
    if(ids->scene != scene || ids->generation != r->sceneGeneration() || ids->nodeCount != model.nodes.size()
       || ids->materialCount != model.materials.size() || ids->primCount != scene->getRenderPrimitives().size())
      rebuildIds(*ids, *scene, r->sceneGeneration());
    Ctx c{*r, *scene, *ids};
    try
    {
      return nvmcp::ToolResult::success(body(c, input).dump());
    }
    catch(const OpError& e)
    {
      return nvmcp::ToolResult::error(Json{{"error", e.message}}.dump());
    }
    catch(const std::exception& e)
    {
      return nvmcp::ToolResult::error(Json{{"error", e.what()}}.dump());
    }
  };

  mcp.registerTool({
      .name        = "vk_gltf_scene_info",
      .description = "Describe the loaded scene for live editing: nodes, materials, meshes with their primitives "
                     "(render primitive id, host id from extras.fvId, vertex count, skinned, morphed, whether the "
                     "BLAS can be refit), animations, lights and the animation state.",
      .handler     = [withScene](std::string_view p) { return withScene(p, [](Ctx& c, const Json&) { return sceneInfo(c); }); },
      .annotations = {.readOnlyHint = true, .destructiveHint = false, .idempotentHint = true, .openWorldHint = false},
      .runOnApplicationThread = true,
  });
}

#endif  // USE_NVMCP
