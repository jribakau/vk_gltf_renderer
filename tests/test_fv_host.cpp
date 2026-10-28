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

// FvHost without a GPU (daz-viewer docs/command-journal-plan.md, C0): its commands through the
// protocol's registry, the application thread being the test's own (onUpdate pumped while a
// command waits on it), over FakeHostRenderer's retained scene. The test library's box figure is
// loaded, dialled, posed, placed, given a material, dressed in its box shirt and removed, and the
// topics read back.

#include <chrono>
#include <filesystem>
#include <future>
#include <thread>

#include <gtest/gtest.h>
#include <nvutils/parameter_registry.hpp>

#include "common/fake_host_renderer.hpp"
#include "fv_host.hpp"
#include "fv_protocol.hpp"

namespace fs = std::filesystem;
using fv::Json;

namespace {

class HostRig
{
public:
  HostRig()
  {
    host = std::make_unique<FvHost>(&renderer, protocol, &registry);
    host->onAttach(nullptr);
    const Json applied = cmd("settings.apply", {{"libraryRoots", {(fs::path(FV_TEST_DATA) / "daz-library").generic_string()}},
                                                {"rootId", "r0"},
                                                {"subdivision", 1}});
    EXPECT_TRUE(applied.value("ok", false)) << applied.dump();
  }
  ~HostRig() { host->onDetach(); }

  // A command as the app sends it, the test's thread standing in for the application thread
  Json cmd(const std::string& name, const Json& args = Json::object())
  {
    auto answer = std::async(std::launch::async, [&] { return protocol->commands().dispatch({{"id", 1}, {"cmd", name}, {"args", args}}); });
    while(answer.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
      host->onUpdate();
    host->onUpdate();
    return answer.get();
  }

  // Frames until no job runs (a load's worker, a composition), or the deadline
  bool settle(std::chrono::seconds deadline = std::chrono::seconds(30))
  {
    const auto end = std::chrono::steady_clock::now() + deadline;
    while(std::chrono::steady_clock::now() < end)
    {
      host->onUpdate();
      const Json jobs    = topic("jobs");
      bool       running = false;
      for(const auto& [id, job] : jobs.items())
        running = running || (job.is_object() && job.value("state", "") == "running");
      if(!running)
      {
        // a frame more for what the last job posted back
        host->onUpdate();
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
  }

  Json topic(const std::string& name) const
  {
    uint64_t upTo = 0;
    for(const fv::Event& e : protocol->events().snapshots(&upTo))
      if(e.topic == name)
        return e.data;
    return Json::object();
  }

  FakeHostRenderer             renderer;
  nvutils::ParameterRegistry   registry;
  std::shared_ptr<FvProtocol>  protocol = std::make_shared<FvProtocol>();
  std::unique_ptr<FvHost>      host;
};

// the one figure prop's id
std::string figureId(const Json& scene)
{
  const Json props = scene.value("props", Json::object());
  for(const auto& [id, p] : props.items())
    if(p.value("kind", "") == "figure")
      return id;
  return "";
}

}  // namespace

TEST(FvHost, LoadsTheBoxFigureIntoItsRetainedScene)
{
  HostRig rig;
  const Json loaded = rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}});
  ASSERT_TRUE(loaded.value("ok", false)) << loaded.dump();
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty()) << rig.topic("scene").dump();
  EXPECT_GT(rig.renderer.hostScene().meshCount(), 0u);
  EXPECT_FALSE(rig.topic("morphs/" + id).empty());
  EXPECT_FALSE(rig.topic("pose/" + id).empty());
}

TEST(FvHost, ADialAPoseAndAPlaceShowInTheTopics)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());

  const Json dialled = rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", 0.5}}}});
  ASSERT_TRUE(dialled.value("ok", false)) << dialled.dump();
  ASSERT_TRUE(rig.settle());
  const Json morphs = rig.topic("morphs/" + id);
  EXPECT_NE(morphs.dump().find("Taper"), std::string::npos) << morphs.dump();

  const Json placed = rig.cmd("prop.set", {{"prop", id}, {"transform", {{"t", {1.0, 2.0, 3.0}}}}});
  ASSERT_TRUE(placed.value("ok", false)) << placed.dump();
  const Json scene = rig.topic("scene");
  ASSERT_TRUE(scene.contains("props") && scene["props"].contains(id)) << scene.dump();
  EXPECT_EQ(scene["props"][id].at("transform").at("t"), Json({1.0, 2.0, 3.0})) << scene.dump();

  const Json state = rig.cmd("scene.state", {{"prop", id}});
  ASSERT_TRUE(state.value("ok", false)) << state.dump();
  EXPECT_NE(state.dump().find("Taper"), std::string::npos) << state.dump();
}

TEST(FvHost, APoseSetAndResetShowInThePoseTopic)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());
  const auto headX = [&] { return rig.topic("pose/" + id).at("bones").at("head").at("axes").at("rotation/x").value("value", -1.0); };

  const Json posed = rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/x", 20.0}}}});
  ASSERT_GT(posed.at("result").value("changed", 0), 0) << posed.dump();
  EXPECT_DOUBLE_EQ(headX(), 20.0);

  ASSERT_TRUE(rig.cmd("pose.reset", {{"prop", id}}).value("ok", false));
  EXPECT_DOUBLE_EQ(headX(), 0.0);
}

TEST(FvHost, AMaterialSetAndResetShowInTheMaterialsTopic)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());
  const auto roughness = [&] { return rig.topic("materials/" + id).at("items").at("m:0").at("values").value("roughness", -1.0); };
  const double was = roughness();

  rig.cmd("material.set", {{"prop", id}, {"material", "m:0"}, {"values", {{"roughness", 0.8}}}});
  EXPECT_NEAR(roughness(), 0.8, 1e-6);

  rig.cmd("material.reset", {{"prop", id}, {"material", "m:0"}});
  EXPECT_NEAR(roughness(), was, 1e-6);
}

TEST(FvHost, AnItemIsWornAndTakenOff)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());
  const size_t bare = rig.renderer.hostScene().meshCount();

  const Json worn = rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(worn.at("result").contains("job")) << worn.dump();
  ASSERT_TRUE(rig.settle());
  const Json  items = rig.topic("wardrobe/" + id).value("items", Json::object());
  std::string item;
  for(const auto& [key, it] : items.items())
    if(it.value("rel", "").find("Box Shirt") != std::string::npos)
      item = key;
  ASSERT_FALSE(item.empty()) << items.dump();
  EXPECT_GT(rig.renderer.hostScene().meshCount(), bare);

  rig.cmd("wardrobe.remove", {{"prop", id}, {"item", item}});
  ASSERT_TRUE(rig.settle());
  EXPECT_FALSE(rig.topic("wardrobe/" + id).value("items", Json::object()).contains(item));
}

TEST(FvHost, ARemovedFigureLeavesTheSceneAndItsTopics)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());

  ASSERT_TRUE(rig.cmd("prop.remove", {{"prop", id}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_TRUE(figureId(rig.topic("scene")).empty()) << rig.topic("scene").dump();
  EXPECT_TRUE(rig.topic("morphs/" + id).empty());
  EXPECT_TRUE(rig.topic("pose/" + id).empty());
  EXPECT_EQ(rig.renderer.hostScene().meshCount(), 0u);
}
