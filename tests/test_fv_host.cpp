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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <set>
#include <thread>

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <httplib.h>

#include "common/fake_host_renderer.hpp"
#include "fv_host.hpp"
#include "fv_journal.hpp"
#include "fv_protocol.hpp"
#include <figure/io/timing.hpp>
#include "pipeline_build_scope.hpp"
#include "scene_export_gltf.hpp"

namespace fs = std::filesystem;
using fv::Json;

namespace {

class HostRig
{
public:
  HostRig()
  {
    // no parameter registry: the renderer's parameters are GltfRenderer's, which lighting writes
    host = std::make_unique<FvHost>(&renderer, protocol, nullptr);
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

  // An /input batch as the app sends it, and the frames that take it
  void input(const Json& events)
  {
    protocol->input(events);
    for(int i = 0; i < 2; i++)
      host->onUpdate();
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

TEST(FvHost, RenderStatusCountsTheStillsSamples)
{
  HostRig rig;
  rig.renderer.stillSamples   = 37;
  rig.renderer.stillSampleCap = 256;
  const Json status        = rig.cmd("render.status", Json::object()).value("result", Json::object());
  EXPECT_EQ(status.value("samples", -1), 37) << status.dump();
  EXPECT_EQ(status.value("maxSamples", -1), 256) << status.dump();
}

// The view draws a scene before anything is in it, its sky and ground, not the renderer's flat clear
TEST(FvHost, StartsWithAnEmptySceneResident)
{
  HostRig rig;
  EXPECT_EQ(rig.renderer.replaces, 1);
  EXPECT_EQ(rig.renderer.hostScene().meshCount(), 0u);
  // the first prop goes into it; it isn't made again
  const Json loaded = rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}});
  ASSERT_TRUE(loaded.value("ok", false)) << loaded.dump();
  ASSERT_TRUE(rig.settle());
  EXPECT_GT(rig.renderer.hostScene().meshCount(), 0u);
  EXPECT_EQ(rig.renderer.replaces, 1);
}

// A load or a wear GPU memory can't take is refused before it is in the scene, and says why
TEST(FvHost, WhatGpuMemoryCantTakeIsRefused)
{
  HostRig rig;
  rig.renderer.geometryRefusal = "no room";
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_GT(rig.renderer.askedGeometry, 0u);
  EXPECT_TRUE(figureId(rig.topic("scene")).empty());
  EXPECT_EQ(rig.renderer.hostScene().meshCount(), 0u);
  EXPECT_NE(rig.topic("notices").dump().find("wasn't loaded: no room"), std::string::npos) << rig.topic("notices").dump();

  // with room it loads; a wear without any leaves the figure as it was
  rig.renderer.geometryRefusal.clear();
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());
  const size_t bare = rig.renderer.hostScene().meshCount();
  rig.renderer.geometryRefusal = "no room";
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(rig.renderer.hostScene().meshCount(), bare);
  EXPECT_NE(rig.topic("notices").dump().find("wasn't added: no room"), std::string::npos) << rig.topic("notices").dump();
  const Json items = rig.topic("wardrobe/" + id).value("items", Json::object());
  EXPECT_EQ(items.dump().find("Box Shirt"), std::string::npos) << items.dump();
}

// A restore that couldn't put everything back says so in its job: the app won't take the scene for
// its file's, whose next save would lose the rest
TEST(FvHost, ARestoreReportsWhatItCouldNotPutBack)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());
  const auto restored = [&](const char* rel) {
    const Json answer = rig.cmd("scene.restore", {{"prop", id}, {"worn", Json::array({Json{{"rel", rel}, {"label", ""}, {"visible", true}}})}});
    EXPECT_TRUE(answer.value("ok", false)) << answer.dump();
    EXPECT_TRUE(rig.settle());
    const Json job = rig.topic("jobs").value(answer.at("result").at("job").dump(), Json::object());
    EXPECT_EQ(job.value("state", ""), "done") << job.dump();
    return job.value("problems", Json::array());
  };
  // what isn't in the library
  EXPECT_NE(restored("People/Test/Clothing/Not There.duf").dump().find("Not There"), std::string::npos);
  // what GPU memory can't take
  rig.renderer.geometryRefusal = "no room";
  EXPECT_NE(restored("People/Test/Clothing/Box Shirt.duf").dump().find("wasn't added: no room"), std::string::npos);
  // and nothing of the kind when it all goes on
  rig.renderer.geometryRefusal.clear();
  EXPECT_TRUE(restored("People/Test/Clothing/Box Shirt.duf").empty());
}

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

//--------------------------------------------------------------------------------------------------
// The edit history (C1)

namespace {

// What a round trip compares: every prop's place and fields, each figure's part of a scene file,
// the view in use, and the lighting
Json stateOf(HostRig& rig)
{
  Json scene   = rig.topic("scene");
  Json figures = Json::object();
  for(auto& [id, p] : scene["props"].items())
  {
    p.erase("stats");
    if(p.value("kind", "") == "figure")
      figures[id] = rig.cmd("scene.state", {{"prop", id}}).at("result");
  }
  Json lighting = rig.topic("lighting");
  if(lighting.contains("env"))
    for(const char* k : {"name", "size", "loading", "sun"})
      lighting["env"].erase(k);
  return {{"figures", figures}, {"scene", scene}, {"lighting", lighting}};
}

// An undo or a redo, and the jobs it starts (a restore that loads); its answer's result
Json historyStep(HostRig& rig, const char* cmd)
{
  const Json answer = rig.cmd(cmd);
  EXPECT_TRUE(answer.value("ok", false)) << answer.dump();
  EXPECT_TRUE(rig.settle());
  return answer.value("result", Json::object());
}

// The first place two states differ (numbers to within 1e-4), or empty. A key that is null on one
// side and missing on the other is the same: a topic's merge patch drops what it sets to null.
std::string differ(const Json& a, const Json& b, const std::string& at = "")
{
  if(a.is_number() && b.is_number())
    return std::abs(a.get<double>() - b.get<double>()) <= 1e-4 ? "" : at + ": " + a.dump() + " then " + b.dump();
  if(a.type() != b.type() || (a.is_array() && a.size() != b.size()))
    return at + ": " + a.dump() + " then " + b.dump();
  if(a.is_object())
  {
    for(auto it = a.begin(); it != a.end(); ++it)
      if(const std::string d = b.contains(it.key()) ? differ(it.value(), b[it.key()], at + "/" + it.key())
                               : it.value().is_null()  ? ""
                                                       : at + "/" + it.key() + " went";
         !d.empty())
        return d;
    for(auto it = b.begin(); it != b.end(); ++it)
      if(!a.contains(it.key()) && !it.value().is_null())
        return at + "/" + it.key() + " came";
    return "";
  }
  if(a.is_array())
  {
    for(size_t i = 0; i < a.size(); i++)
      if(const std::string d = differ(a[i], b[i], at + "/" + std::to_string(i)); !d.empty())
        return d;
    return "";
  }
  return a == b ? "" : at + ": " + a.dump() + " then " + b.dump();
}

double roundTo3(double v)
{
  return std::round(v * 1000.0) / 1000.0;
}

}  // namespace

TEST(FvHost, AnEditUndoesAndRedoes)
{
  HostRig rig;
  EXPECT_EQ(rig.topic("history"), Json({{"undo", Json::array()}, {"redo", Json::array()}, {"busy", false}}));
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());
  EXPECT_EQ(rig.topic("history").at("undo"), Json({"Add Box Figure"}));
  const auto headX = [&] { return rig.topic("pose/" + id).at("bones").at("head").at("axes").at("rotation/x").value("value", -1.0); };

  rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/x", 20.0}}}});
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Pose Head");
  const Json undone = rig.cmd("history.undo");
  EXPECT_EQ(undone.at("result").value("label", ""), "Pose Head") << undone.dump();
  EXPECT_DOUBLE_EQ(headX(), 0.0);
  EXPECT_EQ(rig.topic("history").at("redo"), Json({"Pose Head"}));
  rig.cmd("history.redo");
  EXPECT_DOUBLE_EQ(headX(), 20.0);

  // a dial lands in the frame after its command: its step is read there
  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", 0.5}}}});
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Morph Taper");
  rig.cmd("history.undo");
  EXPECT_EQ(rig.cmd("scene.state", {{"prop", id}}).at("result").at("morphs").value("Taper", 0.0), 0.0);

  // nothing changed: no step, and what could be redone still can
  const size_t steps = rig.topic("history").at("undo").size();
  rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/x", 20.0}}}});
  EXPECT_EQ(rig.topic("history").at("undo").size(), steps);
  EXPECT_EQ(rig.topic("history").at("redo"), Json({"Morph Taper"}));
}

// The round-trip oracle over seeded random sequences of edits, with undos and redos among them: the
// panel edits (C1) and the structural ones (C3: props added, loaded, duplicated, removed, the scene
// cleared). Undoing the steps since the setup puts the scene back as it was then; undoing every step
// empties it; redoing them all brings it to where it ended. More props are removed than are kept, so
// some come back by loading again.
TEST(FvHost, UndoingEveryEditPutsTheSceneBackAndRedoingThemBringsItForward)
{
  for(uint32_t seed = 1; seed <= 3; seed++)
  {
    HostRig rig;
    ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    ASSERT_TRUE(rig.settle());
    const std::string figure = figureId(rig.topic("scene"));
    ASSERT_FALSE(figure.empty());
    rig.cmd("wear", {{"prop", figure}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
    ASSERT_TRUE(rig.settle());
    rig.cmd("prop.add", {{"kind", "camera"}});
    rig.cmd("prop.add", {{"kind", "light"}, {"type", "spot"}});
    ASSERT_TRUE(rig.settle());
    const size_t setup = rig.topic("history").at("undo").size();
    const Json   start = stateOf(rig);

    std::mt19937 rng(seed);
    const auto   u    = [&](double a, double b) { return roundTo3(a + (b - a) * double(rng() >> 8) / 16777216.0); };
    const auto   coin = [&] { return rng() % 2 == 0; };
    std::string  log;
    for(int i = 0; i < 120; i++)
    {
      // what is in the scene now
      const Json               scene = rig.topic("scene");
      std::vector<std::string> props, figures, cameras, lights;
      for(const Json& id : scene.value("order", Json::array()))
      {
        const std::string kind = scene["props"][id.get<std::string>()].value("kind", "");
        props.push_back(id);
        (kind == "figure" ? figures : kind == "camera" ? cameras : lights).push_back(id);
      }
      const auto any = [&](const std::vector<std::string>& v) { return v.empty() ? std::string("p:0") : v[rng() % v.size()]; };
      const std::string id = any(figures);
      std::string       item;
      const Json        worn = rig.topic("wardrobe/" + id).value("items", Json::object());
      for(const auto& [key, it] : worn.items())
        item = key;

      std::string cmd;
      Json        args = Json::object();
      switch(rng() % 32)
      {
        case 0: cmd = "prop.set", args = {{"prop", id}, {"transform", {{"t", {u(-1, 1), 0.0, u(-1, 1)}}}}}; break;
        case 1: cmd = "prop.set", args = {{"prop", any(props)}, {"transform", {{"e", {u(-90, 90), u(-90, 90), 0.0}}}}}; break;
        case 2: cmd = "prop.set", args = {{"prop", any(lights)}, {"light", {{"intensity", u(0, 50)}, {"color", coin() ? "#ff8040" : "#4080ff"}}}}; break;
        case 3: cmd = "prop.set", args = {{"prop", any(cameras)}, {"camera", {{"focalLength", u(20, 120)}}}}; break;
        case 4: cmd = "prop.set", args = {{"prop", any(props)}, {"name", "Name " + std::to_string(i)}}; break;
        case 5: cmd = "prop.set", args = {{"prop", any(props)}, {"visible", coin()}}; break;
        case 6: cmd = "prop.set", args = {{"prop", any(lights)}, {"parent", coin() ? Json(any(props)) : Json()}}; break;
        case 7: cmd = "morph.set", args = {{"prop", id}, {"values", {{coin() ? "Taper" : "Stretch", u(0, 1)}}}}; break;
        case 8: cmd = "morph.reset", args = {{"prop", id}}; break;
        case 9: cmd = "pose.set", args = {{"prop", id}, {"values", {{std::string("head/rotation/") + "xyz"[rng() % 3], u(-50, 50)}}}}; break;
        case 10: cmd = "pose.reset", args = {{"prop", id}}; break;
        case 11:
          cmd  = "material.set";
          args = {{"prop", id},
                  {"material", "m:" + std::to_string(rng() % 2)},
                  {"values", coin() ? Json{{"roughness", u(0, 1)}} : Json{{"color", coin() ? "#336699" : "#aa2200"}}}};
          break;
        case 12: cmd = "material.reset", args = {{"prop", id}}; break;
        case 13: cmd = "wardrobe.visible", args = {{"prop", id}, {"item", item}, {"visible", coin()}}; break;
        case 14:
          cmd  = "lighting.set";
          args = coin() ? Json{{"sun", {{"azimuth", u(0, 360)}, {"enabled", coin()}}}} : Json{{"tonemap", {{"exposure", u(-2, 2)}}}};
          break;
        case 15:
        {
          std::vector<std::string> order = props;
          std::shuffle(order.begin(), order.end(), rng);
          cmd = "scene.order", args = {{"order", order}};
          break;
        }
        case 16: cmd = "view.use", args = {{"camera", coin() ? Json(any(cameras)) : Json()}}; break;
        case 17: cmd = "camera.set", args = {{"focalLength", u(20, 120)}}; break;
        case 18: cmd = "history.undo"; break;
        case 19: cmd = "history.redo"; break;
        case 20: cmd = "prop.add", args = coin() ? Json{{"kind", "camera"}} : Json{{"kind", "light"}, {"type", "point"}}; break;
        case 21:
        case 22: cmd = "prop.remove", args = {{"prop", any(props)}}; break;
        case 23: cmd = "prop.duplicate", args = {{"prop", any(props)}}; break;
        case 24: cmd = "load", args = {{"rel", "People/Test/Box Figure.duf"}}; break;
        case 25: cmd = "prop.set", args = {{"prop", coin() ? any(lights) : any(cameras)}, {"parent", id}, {"bone", coin() ? Json("head") : Json()}}; break;
        case 26: cmd = "wear", args = {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}}; break;
        case 27: cmd = "wardrobe.remove", args = {{"prop", id}, {"item", item}}; break;
        case 28: cmd = "load", args = {{"prop", id}, {"rel", coin() ? "People/Test/Poses/Box Look Left.duf" : "People/Test/Materials/Box Body Red.duf"}}; break;
        case 29: cmd = "material.skin", args = {{"prop", id}, {"rel", coin() ? Json("People/Test/Materials/Box Body Red.duf") : Json()}}; break;
        case 30: cmd = "wardrobe.color", args = {{"prop", id}, {"item", item}, {"rel", coin() ? Json("People/Test/Materials/Box Shirt Blue.duf") : Json()}}; break;
        case 31: cmd = "material.makeup", args = {{"prop", id}, {"rel", Json()}}; break;
        default:
          if(rng() % 4 == 0)
            cmd = "scene.clear";
          else
            cmd = "history.undo";
          break;
      }
      const Json answer = rig.cmd(cmd, args);
      log += cmd + " " + args.dump() + (answer.value("ok", false) ? "" : "  -> " + answer.dump()) + "\n";
      // a load, a figure's duplicate, an undo that loads: in before the next edit
      ASSERT_TRUE(rig.settle()) << log;
      if(answer.contains("result") && answer["result"].is_object())
        ASSERT_FALSE(answer["result"].contains("problems")) << answer.dump() << "\n" << log;
    }
    const Json end = stateOf(rig);

    while(rig.topic("history").at("undo").size() > setup)
      ASSERT_FALSE(historyStep(rig, "history.undo").contains("problems")) << log;
    EXPECT_EQ(differ(start, stateOf(rig)), "") << "seed " << seed << "\n" << log;
    while(!historyStep(rig, "history.undo").at("label").is_null())
      ;
    EXPECT_TRUE(rig.topic("scene").at("props").empty()) << rig.topic("scene").dump();
    for(int redone = 0; !historyStep(rig, "history.redo").at("label").is_null(); redone++)
      ASSERT_LT(redone, 300);
    EXPECT_EQ(differ(end, stateOf(rig)), "") << "seed " << seed << "\n" << log;
  }
}

// A prop rides on a figure's bone (a hat on the head): it stays where it is in the world when it
// gets on, the pose carries it, undo takes it off again, and when the figure goes it stays where the
// pose had it.
TEST(FvHost, APropRidesOnAFiguresBone)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string figure = figureId(rig.topic("scene"));
  const std::string lamp   = rig.cmd("prop.add", {{"kind", "light"}}).at("result").at("prop").get<std::string>();
  rig.cmd("prop.set", {{"prop", lamp}, {"transform", {{"t", {0.2, 1.8, 0.1}}}}});
  ASSERT_TRUE(rig.settle());
  rs::RetainedScene& store = rig.renderer.hostScene();
  const auto         world = [&] {
    for(rs::EntityId e : store.entitiesInOrder())
      if(store.entity(e)->owner == rs::OwnerTag{uint32_t(std::stoi(lamp.substr(2))), -1})
        return store.world(e);
    return glm::mat4(0.f);
  };
  // not `near`: <windows.h> defines it away
  const auto sameWorld = [](const glm::mat4& a, const glm::mat4& b) {
    for(int c = 0; c < 4; c++)
      if(glm::length(a[c] - b[c]) > 1e-4f)
        return false;
    return true;
  };
  const glm::mat4 standing = world();

  // on the head, where it was
  const Json on = rig.cmd("prop.set", {{"prop", lamp}, {"parent", figure}, {"bone", "head"}});
  ASSERT_TRUE(on.value("ok", false)) << on.dump();
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(rig.topic("scene")["props"][lamp].value("bone", Json()), "head");
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Change what Point light rides on");
  EXPECT_TRUE(sameWorld(world(), standing));

  // the head turns: the lamp with it
  rig.cmd("pose.set", {{"prop", figure}, {"values", {{"head/rotation/y", 40.0}}}});
  ASSERT_TRUE(rig.settle());
  const glm::mat4 turned = world();
  EXPECT_FALSE(sameWorld(turned, standing));
  rig.cmd("history.undo");
  ASSERT_TRUE(rig.settle());
  EXPECT_TRUE(sameWorld(world(), standing));
  rig.cmd("history.redo");
  ASSERT_TRUE(rig.settle());

  // a bone the figure hasn't, or one with nothing to ride on, is refused
  EXPECT_FALSE(rig.cmd("prop.set", {{"prop", lamp}, {"parent", figure}, {"bone", "no such bone"}}).value("ok", true));
  EXPECT_FALSE(rig.cmd("prop.set", {{"prop", lamp}, {"parent", nullptr}, {"bone", "head"}}).value("ok", true));

  // the figure goes: the lamp stays where the pose had it, on nothing
  rig.cmd("prop.remove", {{"prop", figure}});
  ASSERT_TRUE(rig.settle());
  const Json left = rig.topic("scene")["props"][lamp];
  EXPECT_TRUE(left.at("parent").is_null() && left.at("bone").is_null()) << left.dump();
  EXPECT_TRUE(sameWorld(world(), turned));
  // and back on the head with the figure's undo
  rig.cmd("history.undo");
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(rig.topic("scene")["props"][lamp].value("bone", Json()), "head");
  EXPECT_TRUE(sameWorld(world(), turned));
}

// A removed dressed figure comes back under its id with its look, pose and wardrobe: kept, and
// loaded again once more removals than are kept came after it.
TEST(FvHost, ARemovedFigureComesBackUnderItsIdAsItWas)
{
  for(bool reloads : {false, true})
  {
    HostRig rig;
    ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    ASSERT_TRUE(rig.settle());
    const std::string id = figureId(rig.topic("scene"));
    rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
    ASSERT_TRUE(rig.settle());
    rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", 0.4}}}});
    rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/y", 25.0}}}});
    rig.cmd("material.set", {{"prop", id}, {"material", "m:1"}, {"values", {{"roughness", 0.7}}}});
    rig.cmd("prop.set", {{"prop", id}, {"transform", {{"t", {0.5, 0.0, -0.25}}}}});
    const Json was = stateOf(rig);

    ASSERT_TRUE(rig.cmd("prop.remove", {{"prop", id}}).value("ok", false));
    ASSERT_TRUE(rig.settle());
    EXPECT_EQ(rig.topic("history").at("undo")[0], "Remove Box Figure");
    int removals = 1;
    if(reloads)
      for(; removals <= int(5); removals++)
      {
        const std::string camera = rig.cmd("prop.add", {{"kind", "camera"}}).at("result").at("prop").get<std::string>();
        rig.cmd("prop.remove", {{"prop", camera}});
      }
    for(int i = 0; i < removals * 2 - 1; i++)
      ASSERT_FALSE(historyStep(rig, "history.undo").contains("problems"));
    EXPECT_EQ(figureId(rig.topic("scene")), id);
    EXPECT_EQ(differ(was, stateOf(rig)), "") << (reloads ? "loaded again" : "kept");
    EXPECT_FALSE(rig.topic("morphs/" + id).empty());
    EXPECT_EQ(rig.topic("wardrobe/" + id).value("items", Json::object()).size(), 1u);
  }
}

// Putting an item on and taking it off are steps; an item comes back under its id, as it was.
TEST(FvHost, AnItemTakenOffComesBackUnderItsIdAsItWas)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Wear Box Shirt");
  const auto items = [&] { return rig.topic("wardrobe/" + id).value("items", Json::object()); };
  ASSERT_EQ(items().size(), 1u);
  const std::string item = items().begin().key();

  // the wear undone and redone: the same id
  historyStep(rig, "history.undo");
  EXPECT_TRUE(items().empty());
  historyStep(rig, "history.redo");
  ASSERT_TRUE(items().contains(item)) << items().dump();

  // hidden, its surface edited, its cloth set: taken off and put back, all of it
  // a wear's surfaces come after the figure's: the shirt's is the last
  int        shirt     = -1;
  const Json materials = rig.topic("materials/" + id).value("items", Json::object());
  for(const auto& [m, it] : materials.items())
    shirt = std::max(shirt, std::stoi(m.substr(2)));
  ASSERT_GE(shirt, 0);
  rig.cmd("wardrobe.visible", {{"prop", id}, {"item", item}, {"visible", false}});
  rig.cmd("material.set", {{"prop", id}, {"material", shirt}, {"values", {{"color", "#aa2200"}}}});
  rig.cmd("wardrobe.cloth", {{"prop", id}, {"item", item}, {"on", false}});
  const Json was = stateOf(rig);
  ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", item}}).value("ok", false));
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Take off Box Shirt");
  EXPECT_TRUE(items().empty());
  const Json undone = historyStep(rig, "history.undo");
  EXPECT_FALSE(undone.contains("problems")) << undone.dump();
  ASSERT_TRUE(items().contains(item)) << items().dump();
  EXPECT_FALSE(items()[item].value("visible", true));
  EXPECT_EQ(differ(was, stateOf(rig)), "");
}

namespace {
// What the figure of a saved `state` comes back as: in a session of its own, as a scene file opens
struct Reopened
{
  HostRig     rig;
  std::string id;
  Json        problems = Json::array();

  explicit Reopened(Json state, const std::string& rel = "People/Test/Box Figure.duf")
  {
    EXPECT_TRUE(rig.cmd("load", {{"rel", rel}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    id            = figureId(rig.topic("scene"));
    state["prop"] = id;
    const Json answer = rig.cmd("scene.restore", state);
    EXPECT_TRUE(answer.value("ok", false)) << answer.dump();
    EXPECT_TRUE(rig.settle());
    problems = rig.topic("jobs").value(answer.at("result").at("job").dump(), Json::object()).value("problems", Json::array());
  }
  Json items() { return rig.topic("wardrobe/" + id).value("items", Json::object()); }
  Json dial(const std::string& name) { return rig.topic("morphs/" + id).at("channels").value(name, Json::object()).value("value", Json()); }
};

Json wardrobeOf(HostRig& rig, const std::string& id)
{
  return rig.topic("wardrobe/" + id).value("items", Json::object());
}

// The worn item called `label`'s id ("i:3")
std::string itemNamed(const Json& items, const std::string& label)
{
  for(const auto& [id, item] : items.items())
    if(item.value("label", "") == label)
      return id;
  return "";
}

// The channel of `dial` on the item called `label`, as the channels topic lists it under the item's group
std::string dialOn(HostRig& rig, const std::string& id, const std::string& dial, const std::string& label)
{
  const Json channels = rig.topic("morphs/" + id).at("channels");
  for(const auto& [name, c] : channels.items())
    if(name.rfind(dial + "@", 0) == 0 && c.value("group", "") == label)
      return name;
  return "";
}
}  // namespace

// A preset over a skin chosen: a Head's tint leaves the Body's skin chosen, and both come back when the
// scene does, the skin named as it was. One over all of the skin is the skin. A skin chosen over the
// surfaces the tint set takes it off them, and the file agrees.
TEST(FvHost, APresetOverTheChosenSkinKeepsBothWhenReopened)
{
  const std::string tint = "People/Test/Materials/Box Head Tint.duf", red = "People/Test/Materials/Box Body Red.duf",
                    full = "People/Test/Materials/Box Skin Full.duf";
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id    = figureId(rig.topic("scene"));
  const auto        state = [&] { return rig.cmd("scene.state", {{"prop", id}}).at("result"); };
  const auto        names = [](const Json& s, const char* list) { return s.value(list, Json::array()); };
  const auto        surfaces = [](HostRig& r, const std::string& p) {
    Json       out       = Json::object();
    const Json materials = r.topic("materials/" + p).value("items", Json::object());
    for(const auto& [m, it] : materials.items())
      out[m] = it.value("values", Json::object());
    return out;
  };
  const auto apply = [&](const char* cmd, const std::string& rel) {
    ASSERT_TRUE(rig.cmd(cmd, {{"prop", id}, {"rel", rel}}).value("ok", false));
    ASSERT_TRUE(rig.settle());
  };
  const auto skinTopic = [&] { return rig.topic("materials/" + id).value("skin", Json::object()); };

  apply("material.skin", red);
  apply("load", tint);
  Json saved = state();
  EXPECT_EQ(saved.value("skin", ""), red) << saved.dump();
  EXPECT_EQ(names(saved, "materialPresets"), Json::array());
  EXPECT_EQ(names(saved, "materialPresetsOver"), Json::array({tint}));
  EXPECT_EQ(skinTopic().value("rel", ""), red) << skinTopic().dump();
  EXPECT_TRUE(skinTopic().value("preset", Json()).is_null()) << skinTopic().dump();
  {
    Reopened back(saved);
    EXPECT_TRUE(back.problems.empty()) << back.problems.dump();
    EXPECT_EQ(surfaces(back.rig, back.id), surfaces(rig, id));
    const Json again = back.rig.cmd("scene.state", {{"prop", back.id}}).at("result");
    EXPECT_EQ(again.value("skin", ""), red);
    EXPECT_EQ(names(again, "materialPresetsOver"), Json::array({tint}));
  }

  // undone, the look before: no preset over anything
  historyStep(rig, "history.undo");
  EXPECT_EQ(names(state(), "materialPresetsOver"), Json::array());
  historyStep(rig, "history.redo");
  EXPECT_EQ(names(state(), "materialPresetsOver"), Json::array({tint}));

  // a skin of all of it, chosen: the tint is gone from the Head, and a reopened scene has it gone too
  apply("material.skin", full);
  saved = state();
  EXPECT_EQ(names(saved, "materialPresetsOver"), Json::array());
  EXPECT_EQ(names(saved, "materialPresets"), Json::array({tint}));
  {
    Reopened back(saved);
    EXPECT_EQ(surfaces(back.rig, back.id), surfaces(rig, id));
  }

  // opened as a preset, over all of the skin, it is the skin: named so, and nothing is chosen
  apply("material.skin", red);
  apply("load", full);
  saved = state();
  EXPECT_FALSE(saved.contains("skin")) << saved.dump();
  EXPECT_EQ(skinTopic().value("preset", ""), full) << skinTopic().dump();
  EXPECT_EQ(names(saved, "materialPresetsOver"), Json::array());
}

// A surface's name means what its mesh is: the Body of a vest is no flesh, though the figure's is, and a
// preset over both, which reads their channels again, leaves that so.
TEST(FvHost, ASurfaceOnAGarmentIsNoFleshWhateverItsName)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Vest.duf"}});
  ASSERT_TRUE(rig.settle());
  const auto kinds = [&] {
    std::multiset<std::string> out;
    const Json                 items = rig.topic("materials/" + id).value("items", Json::object());
    for(const auto& [m, it] : items.items())
      if(it.value("name", "") == "Body")
        out.insert(it.value("kind", ""));
    return out;
  };
  const std::multiset<std::string> expected{"other", "skin"};
  EXPECT_EQ(kinds(), expected);
  ASSERT_TRUE(rig.cmd("load", {{"prop", id}, {"rel", "People/Test/Materials/Box Body Red.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(kinds(), expected);

  // and it is an open shell with an inside to see, where the body is closed; the surface's own choice wins
  const auto sides = [&](const char* kind) {
    const Json items = rig.topic("materials/" + id).value("items", Json::object());
    for(const auto& [m, it] : items.items())
      if(it.value("name", "") == "Body" && it.value("kind", "") == kind)
        return std::pair(m, it.at("values").value("doubleSided", false));
    return std::pair(std::string(), false);
  };
  EXPECT_TRUE(sides("other").second);
  EXPECT_FALSE(sides("skin").second);
  rig.cmd("material.set", {{"prop", id}, {"material", sides("other").first}, {"values", {{"doubleSided", false}}}});
  EXPECT_FALSE(sides("other").second);
}

// Skin tight is the item's: a step, kept when it is taken off and put back, and in the scene file.
TEST(FvHost, HowTightAnItemIsGoesWithIt)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  const std::string item  = itemNamed(wardrobeOf(rig, id), "Box Shirt");
  const auto        tight = [&] { return wardrobeOf(rig, id).at(item).value("tight", -1.0); };
  EXPECT_DOUBLE_EQ(tight(), 0.0);
  EXPECT_FALSE(rig.cmd("scene.state", {{"prop", id}}).at("result").at("worn")[0].contains("tight"));

  // a drag: answered at once, the latest applies in the frame, all of it one step of the history
  const Json gesture = {{"id", 7}};
  ASSERT_TRUE(rig.cmd("wardrobe.tight", {{"prop", id}, {"item", item}, {"amount", 0.25}, {"gesture", gesture}}).value("ok", false));
  ASSERT_TRUE(rig.cmd("wardrobe.tight", {{"prop", id}, {"item", item}, {"amount", 0.5}, {"gesture", gesture}}).value("ok", false));
  rig.host->onUpdate();
  EXPECT_DOUBLE_EQ(tight(), 0.5);
  ASSERT_TRUE(rig.cmd("wardrobe.tight", {{"prop", id}, {"item", item}, {"amount", 0.75}, {"gesture", gesture}}).value("ok", false));
  rig.host->onUpdate();
  EXPECT_DOUBLE_EQ(tight(), 0.75);
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Skin tight: Box Shirt");
  historyStep(rig, "history.undo");
  EXPECT_DOUBLE_EQ(tight(), 0.0);
  historyStep(rig, "history.redo");
  EXPECT_DOUBLE_EQ(tight(), 0.75);

  // off and back
  ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", item}}).value("ok", false));
  historyStep(rig, "history.undo");
  EXPECT_DOUBLE_EQ(tight(), 0.75);

  // reopened
  Reopened back(rig.cmd("scene.state", {{"prop", id}}).at("result"));
  EXPECT_TRUE(back.problems.empty()) << back.problems.dump();
  EXPECT_DOUBLE_EQ(back.items().at(itemNamed(back.items(), "Box Shirt")).value("tight", -1.0), 0.75);
}

// Skin all the way under opaque fabric isn't drawn while the fabric is tight, so it can't show through,
// and it comes back whenever that stops being so. Box Snug is a closed box just around the Box Figure.
TEST(ShaderWork, SaysWhatIsBeingBuiltAndForHowLong)
{
  EXPECT_EQ(shaderWorkNow().work, ShaderWork::eNone);
  EXPECT_STREQ(shaderWorkName(ShaderWork::eNone), "");

  // the warm-up goes on behind everything
  {
    const ShaderWarmScope one, two;
    EXPECT_EQ(shaderWorkNow().work, ShaderWork::eWarm);
    EXPECT_STREQ(shaderWorkName(shaderWorkNow().work), "warm");

    // a build that holds the view is the wait to explain, over the warm-up. Entering the build says
    // nothing yet: a frame with nothing to build passes through it
    const PipelineBuildScope building;
    EXPECT_EQ(shaderWorkNow().work, ShaderWork::eWarm);
    {
      const ShaderWorkScope compile(ShaderWork::eCompile);
      EXPECT_EQ(shaderWorkNow().work, ShaderWork::eCompile);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    {
      const ShaderWorkScope pipeline(ShaderWork::ePipeline);
      EXPECT_STREQ(shaderWorkName(shaderWorkNow().work), "pipeline");
      // counted from the build's start, across its stages
      EXPECT_GE(shaderWorkNow().ms, 15);
    }
    EXPECT_EQ(shaderWorkNow().work, ShaderWork::eWarm);
  }
  EXPECT_EQ(shaderWorkNow().work, ShaderWork::eNone);
}

TEST(FvHost, TightFabricHidesTheSkinUnderIt)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Snug.duf"}});
  ASSERT_TRUE(rig.settle());
  const std::string item   = itemNamed(wardrobeOf(rig, id), "Box Snug");
  const auto        hidden = [&] { return wardrobeOf(rig, id).at(item).value("skinHidden", -1); };
  // the collider's pass has to end first: that's when what's hidden is brought up to date
  const auto        fitted = [&] {
    for(int i = 0; i < 3000 && rig.topic("wardrobe/" + id).at("collision").value("busy", true); i++)
    {
      rig.host->onUpdate();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    rig.host->onUpdate();
    ASSERT_TRUE(rig.settle());
  };
  const auto        tight  = [&](double amount) {
    ASSERT_TRUE(rig.cmd("wardrobe.tight", {{"prop", id}, {"item", item}, {"amount", amount}}).value("ok", false));
    rig.host->onUpdate();
    fitted();
  };
  ASSERT_EQ(hidden(), 0);
  tight(1);
  const int covered = hidden();
  ASSERT_GT(covered, 0) << wardrobeOf(rig, id).dump();
  EXPECT_DOUBLE_EQ(wardrobeOf(rig, id).at(item).value("tight", 0.0), 1.0);
  tight(0);
  EXPECT_EQ(hidden(), 0);
  tight(0.5);
  EXPECT_EQ(hidden(), covered);

  // undone, and put back
  historyStep(rig, "history.undo");
  fitted();
  EXPECT_EQ(hidden(), 0);
  historyStep(rig, "history.redo");
  fitted();
  EXPECT_EQ(hidden(), covered);

  // hidden, its skin is back; shown, gone again
  ASSERT_TRUE(rig.cmd("wardrobe.visible", {{"prop", id}, {"item", item}, {"visible", false}}).value("ok", false));
  fitted();
  EXPECT_EQ(hidden(), 0);
  ASSERT_TRUE(rig.cmd("wardrobe.visible", {{"prop", id}, {"item", item}, {"visible", true}}).value("ok", false));
  fitted();
  EXPECT_EQ(hidden(), covered);

  // collision off: nothing holds it to the body
  ASSERT_TRUE(rig.cmd("wardrobe.collision", {{"prop", id}, {"on", false}}).value("ok", false));
  fitted();
  EXPECT_EQ(hidden(), 0);
  ASSERT_TRUE(rig.cmd("wardrobe.collision", {{"prop", id}, {"on", true}}).value("ok", false));
  fitted();
  EXPECT_EQ(hidden(), covered);

  // fabric that lets anything through gives the skin back
  const Json materials = rig.topic("materials/" + id).at("items");
  std::string shirt;
  for(const auto& [key, m] : materials.items())
    if(m.value("name", "") == "Shirt")
      shirt = key;
  ASSERT_FALSE(shirt.empty()) << materials.dump();
  rig.cmd("material.set", {{"prop", id}, {"material", shirt}, {"values", {{"opacity", 0.5}}}});
  fitted();
  EXPECT_EQ(hidden(), 0);
}

// A dial the character's file sets (the Box Figure's Stretch, at 0.5) stays where the user put it, at 0
// too: a file holds what differs from how the figure loads. What didn't change isn't in it, so a file
// from before, which never held a zero, still means "as loaded".
TEST(FvHost, ADialTurnedToNothingStaysThereWhenReopened)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id    = figureId(rig.topic("scene"));
  const auto        state = [&] { return rig.cmd("scene.state", {{"prop", id}}).at("result"); };
  EXPECT_FALSE(state().at("morphs").contains("Stretch")) << state().dump();

  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Stretch", 0.0}}}});
  rig.host->onUpdate();
  ASSERT_TRUE(state().at("morphs").contains("Stretch"));
  EXPECT_EQ(Reopened(state()).dial("Stretch"), 0.0);

  // and Reset morphs, the whole of what the file set
  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Stretch", 0.7}, {"Taper", 0.25}}}});
  rig.host->onUpdate();
  rig.cmd("morph.reset", {{"prop", id}});
  rig.host->onUpdate();
  Reopened reset(state());
  EXPECT_EQ(reset.dial("Stretch"), 0.0);
  EXPECT_EQ(reset.dial("Taper"), 0.0);

  // a file with nothing for it: as loaded
  Json old = state();
  old["morphs"].erase("Stretch");
  EXPECT_EQ(Reopened(old).dial("Stretch"), 0.5);
}

// A general body slider comes with a load, from the figure's bones alone, and is a dial like any other:
// the panel is told what it is, a scene file holds it, undo takes it back and Reset morphs zeroes it.
TEST(FvHost, ABodySliderIsADialFromLoadToReopening)
{
  const std::string rel = "People/Test/Shape Figure.duf";
  HostRig           rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", rel}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id    = figureId(rig.topic("scene"));
  const auto        state = [&] { return rig.cmd("scene.state", {{"prop", id}}).at("result"); };
  const Json        ch    = rig.topic("morphs/" + id).at("channels").value("fv_shape_waist", Json::object());
  EXPECT_EQ(ch.value("label", ""), "Waist") << ch.dump();
  EXPECT_EQ(ch.value("kind", ""), "Modifier/FvShape");
  EXPECT_EQ(ch.value("region", ""), "Waist");
  EXPECT_EQ(ch.value("min", 0.0), -1.0);
  EXPECT_EQ(ch.value("max", 0.0), 1.0);
  EXPECT_FALSE(ch.value("helper", false));
  EXPECT_FALSE(state().at("morphs").contains("fv_shape_waist")) << state().dump();

  rig.cmd("morph.set", {{"prop", id}, {"values", {{"fv_shape_waist", 0.5}}}});
  rig.host->onUpdate();
  EXPECT_EQ(state().at("morphs").value("fv_shape_waist", 0.0), 0.5);
  {
    Reopened reopened(state(), rel);
    EXPECT_EQ(reopened.dial("fv_shape_waist"), 0.5);
    EXPECT_TRUE(reopened.problems.empty()) << reopened.problems.dump();
  }

  rig.cmd("history.undo");
  rig.host->onUpdate();
  EXPECT_FALSE(state().at("morphs").contains("fv_shape_waist")) << state().dump();

  rig.cmd("morph.set", {{"prop", id}, {"values", {{"fv_shape_waist", -1.0}}}});
  rig.host->onUpdate();
  rig.cmd("morph.reset", {{"prop", id}});
  rig.host->onUpdate();
  EXPECT_EQ(rig.topic("morphs/" + id).at("channels").at("fv_shape_waist").value("value", 1.0), 0.0);
}

// A worn item's fit dial is named by the item's key. The key goes in the file, so the dial finds its
// garment after a session that took others off, and the same modifier on another garment stays out of it.
TEST(FvHost, AFitDialFindsItsGarmentWhenReopened)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  // on and off again: the next one's key has a gap before it
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Pants.duf"}});
  ASSERT_TRUE(rig.settle());
  rig.cmd("wardrobe.remove", {{"prop", id}, {"item", itemNamed(wardrobeOf(rig, id), "Box Pants")}});
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Pants.duf"}});
  ASSERT_TRUE(rig.settle());
  const std::string dial = dialOn(rig, id, "Loosen Neck", "Box Pants");
  ASSERT_FALSE(dial.empty()) << rig.topic("morphs/" + id).dump();
  rig.cmd("morph.set", {{"prop", id}, {"values", {{dial, 0.6}}}});
  rig.host->onUpdate();
  const Json saved = rig.cmd("scene.state", {{"prop", id}}).at("result");

  Reopened back(saved);
  EXPECT_TRUE(back.problems.empty()) << back.problems.dump();
  EXPECT_EQ(back.dial(dialOn(back.rig, back.id, "Loosen Neck", "Box Pants")), 0.6);
  EXPECT_EQ(back.dial(dialOn(back.rig, back.id, "Loosen Neck", "Box Shirt")), 0.0);
  // and the next save has it still
  EXPECT_EQ(back.rig.cmd("scene.state", {{"prop", back.id}}).at("result").at("morphs").value(dial, 0.0), 0.6);

  // A file from before: no keys, and a key from a session that's gone. The one garment with the dial is meant.
  Json old = saved;
  old["worn"] = Json::array({Json{{"rel", "People/Test/Clothing/Box Pants.duf"}, {"label", "Box Pants"}, {"visible", true}, {"id", 9}}});
  old["morphs"] = Json{{"Loosen Neck@w7", 0.3}};
  Reopened older(old);
  EXPECT_TRUE(older.problems.empty()) << older.problems.dump();
  EXPECT_EQ(older.dial(dialOn(older.rig, older.id, "Loosen Neck", "Box Pants")), 0.3);
  // nothing to go by: it is reported, and no garment takes it
  old["worn"].push_back(Json{{"rel", "People/Test/Clothing/Box Shirt.duf"}, {"label", "Box Shirt"}, {"visible", true}, {"id", 10}});
  Reopened unsure(old);
  EXPECT_NE(unsure.problems.dump().find("1 morph dial not found"), std::string::npos) << unsure.problems.dump();
  EXPECT_EQ(unsure.dial(dialOn(unsure.rig, unsure.id, "Loosen Neck", "Box Pants")), 0.0);
  EXPECT_EQ(unsure.dial(dialOn(unsure.rig, unsure.id, "Loosen Neck", "Box Shirt")), 0.0);
}

// A dial leaves with its item and comes back with it, by a take-off's undo, under its name, so the step
// that set it still finds it
TEST(FvHost, AnItemTakenOffKeepsItsDials)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  const std::string dial = dialOn(rig, id, "Loosen Neck", "Box Shirt");
  ASSERT_FALSE(dial.empty());
  rig.cmd("morph.set", {{"prop", id}, {"values", {{dial, 0.6}}}});
  rig.host->onUpdate();
  ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", itemNamed(wardrobeOf(rig, id), "Box Shirt")}}).value("ok", false));
  ASSERT_FALSE(rig.topic("morphs/" + id).at("channels").contains(dial));

  historyStep(rig, "history.undo");
  rig.host->onUpdate();
  const Json channels = rig.topic("morphs/" + id).at("channels");
  ASSERT_TRUE(channels.contains(dial)) << channels.dump();
  EXPECT_EQ(channels[dial].value("value", 0.0), 0.6);
  // the step before it: the dial set
  historyStep(rig, "history.undo");
  rig.host->onUpdate();
  EXPECT_EQ(rig.topic("morphs/" + id).at("channels")[dial].value("value", 1.0), 0.0);
}

// A file that puts on two pieces: each is taken off and comes back by itself, and the wear undone and
// redone is both of them. A piece taken off stays off when the scene is reopened.
TEST(FvHost, TheOtherPiecesOfAFileStayWhenOneComesBack)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Outfit.duf"}});
  ASSERT_TRUE(rig.settle());
  const auto items = [&] { return wardrobeOf(rig, id); };
  ASSERT_EQ(items().size(), 2u) << items().dump();
  const std::string shirt = itemNamed(items(), "Box Shirt"), pants = itemNamed(items(), "Box Pants");
  ASSERT_FALSE(shirt.empty());
  ASSERT_FALSE(pants.empty());
  rig.cmd("material.set", {{"prop", id}, {"material", "m:1"}, {"values", {{"roughness", 0.7}}}});
  // a piece that comes back is the last in the list: what it is goes by its key, not its place
  const auto state = [&] {
    Json s = stateOf(rig);
    for(auto& [fig, f] : s["figures"].items())
      std::sort(f["worn"].begin(), f["worn"].end(), [](const Json& a, const Json& b) { return a.value("id", 0) < b.value("id", 0); });
    return s;
  };
  const Json was = state();

  // either piece off and back
  for(const std::string& piece : {shirt, pants})
  {
    ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", piece}}).value("ok", false));
    ASSERT_EQ(items().size(), 1u);
    EXPECT_FALSE(items().contains(piece));
    const Json undone = historyStep(rig, "history.undo");
    EXPECT_FALSE(undone.contains("problems")) << undone.dump();
    ASSERT_EQ(items().size(), 2u) << items().dump();
    EXPECT_TRUE(items().contains(shirt));
    EXPECT_TRUE(items().contains(pants));
    EXPECT_EQ(differ(was, state()), "");
  }

  // the wear undone and redone: both pieces
  historyStep(rig, "history.undo");  // the material
  historyStep(rig, "history.undo");  // the wear
  EXPECT_TRUE(items().empty());
  historyStep(rig, "history.redo");
  ASSERT_EQ(items().size(), 2u) << items().dump();
  EXPECT_TRUE(items().contains(shirt));
  EXPECT_TRUE(items().contains(pants));

  // the second piece worn alone comes back alone, the first alone too
  for(const std::string& piece : {shirt, pants})
  {
    const std::string other = piece == shirt ? pants : shirt;
    ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", other}}).value("ok", false));
    Reopened back(rig.cmd("scene.state", {{"prop", id}}).at("result"));
    ASSERT_EQ(back.items().size(), 1u) << back.items().dump();
    EXPECT_FALSE(itemNamed(back.items(), wardrobeOf(rig, id).at(piece).value("label", "")).empty());
    historyStep(rig, "history.undo");
  }
}

// A scene file opening as the app sends it (a clear, the loads, their restores, places, the
// lighting), between history.begin and history.end: one step, whose undo is the scene before and
// whose redo is the scene opened.
TEST(FvHost, ASceneFileOpeningIsOneStep)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", 0.4}}}});
  rig.cmd("material.set", {{"prop", id}, {"material", "m:0"}, {"values", {{"roughness", 0.7}}}});
  rig.cmd("prop.add", {{"kind", "camera"}});
  Json part = rig.cmd("scene.state", {{"prop", id}}).at("result");
  const Json   before = stateOf(rig);
  const size_t steps  = rig.topic("history").at("undo").size();

  ASSERT_TRUE(rig.cmd("history.begin", {{"label", "Open Box Scene"}}).value("ok", false));
  EXPECT_TRUE(rig.topic("history").value("busy", false));
  rig.cmd("scene.clear");
  ASSERT_TRUE(rig.cmd("prop.add", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string opened = figureId(rig.topic("scene"));
  ASSERT_FALSE(opened.empty());
  part["prop"] = opened;
  part["morphs"]["Taper"] = -0.3;
  ASSERT_TRUE(rig.cmd("scene.restore", part).value("ok", false));
  ASSERT_TRUE(rig.settle());
  rig.cmd("prop.set", {{"prop", opened}, {"name", "Opened"}, {"visible", true}, {"parent", nullptr}});
  rig.cmd("prop.set", {{"prop", opened}, {"transform", {{"t", {0.5, 0.0, -0.25}}}}});
  rig.cmd("lighting.set", {{"tonemap", {{"exposure", 1.5}}}});
  // an undo waits for the end
  EXPECT_FALSE(rig.cmd("history.undo").value("ok", true));
  ASSERT_TRUE(rig.cmd("history.end").value("ok", false));

  const Json history = rig.topic("history");
  EXPECT_FALSE(history.value("busy", true));
  ASSERT_EQ(history.at("undo").size(), steps + 1) << history.dump();
  EXPECT_EQ(history.at("undo")[0], "Open Box Scene");
  const Json after = stateOf(rig);
  EXPECT_NE(differ(before, after), "");
  const Json undone = historyStep(rig, "history.undo");
  EXPECT_FALSE(undone.contains("problems")) << undone.dump();
  EXPECT_EQ(differ(before, stateOf(rig)), "") << "undone";
  const Json redone = historyStep(rig, "history.redo");
  EXPECT_FALSE(redone.contains("problems")) << redone.dump();
  EXPECT_EQ(differ(after, stateOf(rig)), "") << "redone";

  // a crash's copy opened: the history goes
  ASSERT_TRUE(rig.cmd("history.clear").value("ok", false));
  EXPECT_TRUE(rig.topic("history").at("undo").empty());
}

// The scene kept for a crash (C5): written a moment after the changes pause, it puts the scene back
// in another renderer as it was, under the same ids; a change after it is kept too; a torn copy is
// refused; and closing the scene as meant takes the copy away.
TEST(FvHost, AKeptSceneComesBackInAnotherRendererAsItWas)
{
  const fs::path path = fs::temp_directory_path() / "fv_recovery_test.json";
  fs::remove(path);
  const auto written = [&](fs::file_time_type after, HostRig& rig) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while(std::chrono::steady_clock::now() < end)
    {
      rig.settle();
      std::error_code ec;
      if(fs::exists(path) && fs::last_write_time(path, ec) > after)
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
  };
  const auto recovered = [&](Json& state) {
    HostRig    again;
    const Json answer = again.cmd("scene.recover", {{"path", path.generic_string()}});
    EXPECT_TRUE(answer.value("ok", false)) << answer.dump();
    EXPECT_TRUE(again.settle());
    state = stateOf(again);
    // a prop added after takes an id the kept ones don't have
    const Json added = again.cmd("prop.add", {{"kind", "light"}}).at("result");
    EXPECT_FALSE(state["scene"]["props"].contains(added.value("prop", ""))) << added.dump();
  };

  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", 0.4}}}});
  rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/y", 25.0}}}});
  rig.cmd("material.set", {{"prop", id}, {"material", "m:1"}, {"values", {{"roughness", 0.7}}}});
  rig.cmd("prop.set", {{"prop", id}, {"transform", {{"t", {0.5, 0.0, -0.25}}}}});
  const std::string camera = rig.cmd("prop.add", {{"kind", "camera"}}).at("result").at("prop").get<std::string>();
  rig.cmd("prop.set", {{"prop", camera}, {"parent", id}, {"bone", "head"}});
  rig.cmd("prop.add", {{"kind", "light"}});
  rig.cmd("scene.order", {{"order", {camera, id}}});
  rig.cmd("lighting.set", {{"tonemap", {{"exposure", 1.5}}}});
  rig.cmd("view.use", {{"camera", camera}});

  ASSERT_TRUE(rig.cmd("recovery.file", {{"path", path.generic_string()}}).value("ok", false));
  ASSERT_TRUE(written(fs::file_time_type::min(), rig));
  const auto peek = [&] { return rig.cmd("recovery.peek", {{"path", path.generic_string()}}).at("result"); };
  EXPECT_EQ(peek().value("props", 0), 3) << peek().dump();
  EXPECT_TRUE(peek().contains("at"));
  Json state;
  recovered(state);
  EXPECT_EQ(differ(stateOf(rig), state), "");

  // a look chosen after: kept too
  const auto at = fs::last_write_time(path);
  ASSERT_TRUE(rig.cmd("load", {{"prop", id}, {"rel", "People/Test/Materials/Box Body Red.duf"}}).value("ok", false));
  ASSERT_TRUE(written(at, rig));
  recovered(state);
  EXPECT_EQ(differ(stateOf(rig), state), "") << "after the look";

  // closed as meant: the copy goes
  ASSERT_TRUE(rig.cmd("recovery.file", {{"path", nullptr}}).value("ok", false));
  EXPECT_FALSE(fs::exists(path));
  EXPECT_EQ(peek().value("props", -1), 0);

  // a copy torn mid-write is refused, the scene left as it is
  std::ofstream(path) << R"({"version": 1, "ops": [{"cmd": "prop.res)";
  HostRig torn;
  EXPECT_EQ(peek().value("props", -1), 0);
  EXPECT_FALSE(torn.cmd("scene.recover", {{"path", path.generic_string()}}).value("ok", true));
  EXPECT_TRUE(torn.topic("scene").value("props", Json::object()).empty());
  fs::remove(path);
}

// A preset, a skin, an item's colour: each a step whose undo puts the figure's look back as it was,
// surfaces and all, and whose redo brings the change back.
TEST(FvHost, APresetOrALookChoiceUndoesToTheLookBefore)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  const Json        worn = rig.topic("wardrobe/" + id).value("items", Json::object());
  ASSERT_EQ(worn.size(), 1u);
  const std::string item = worn.begin().key();
  rig.cmd("material.set", {{"prop", id}, {"material", "m:0"}, {"values", {{"roughness", 0.5}}}});
  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", 0.2}}}});
  // the scene's state and each surface as the Material tab shows it
  const auto look = [&] {
    Json       state     = stateOf(rig);
    const Json materials = rig.topic("materials/" + id).value("items", Json::object());
    for(const auto& [m, it] : materials.items())
      state["surfaces"][m] = it.value("values", Json::object());
    return state;
  };

  const std::vector<std::pair<std::string, Json>> edits = {
      {"load", {{"prop", id}, {"rel", "People/Test/Poses/Box Look Left.duf"}}},
      {"load", {{"prop", id}, {"rel", "People/Test/Materials/Box Body Red.duf"}}},
      {"material.skin", {{"prop", id}, {"rel", "People/Test/Materials/Box Body Red.duf"}}},
      {"wardrobe.color", {{"prop", id}, {"item", item}, {"rel", "People/Test/Materials/Box Shirt Blue.duf"}}},
  };
  for(const auto& [cmd, args] : edits)
  {
    const Json   before = look();
    const size_t steps  = rig.topic("history").at("undo").size();
    ASSERT_TRUE(rig.cmd(cmd, args).value("ok", false)) << cmd;
    ASSERT_TRUE(rig.settle());
    ASSERT_EQ(rig.topic("history").at("undo").size(), steps + 1) << cmd << " " << args.dump();
    const Json after = look();
    EXPECT_NE(differ(before, after), "") << cmd << " changed nothing";
    EXPECT_FALSE(historyStep(rig, "history.undo").contains("problems"));
    EXPECT_EQ(differ(before, look()), "") << cmd << " undone";
    EXPECT_FALSE(historyStep(rig, "history.redo").contains("problems"));
    EXPECT_EQ(differ(after, look()), "") << cmd << " redone";
  }
}

//--------------------------------------------------------------------------------------------------
// Gestures (C2): a drag, from press to release, and a slider's gesture are one step each

namespace {

size_t undoSteps(HostRig& rig)
{
  return rig.topic("history").at("undo").size();
}

// longer than the history's merge window: what still merges past it is a gesture
void pauseBeyondMerge()
{
  std::this_thread::sleep_for(fv::Journal::kMerge + std::chrono::milliseconds(100));
}

}  // namespace

TEST(FvHost, AGizmoDragOfAPropIsOneStep)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_TRUE(rig.cmd("select", {{"prop", id}}).value("ok", false));
  const rs::EntityId root = rig.renderer.selectedEntity();
  ASSERT_TRUE(rig.renderer.gizmo.edit);
  const Json start = rig.topic("scene").at("props").at(id).at("transform");

  // as the renderer calls it: every move of the drag, then the release with the last transform
  const auto drag = [&](float to) {
    for(float x = 0.1f; x <= to; x += 0.1f)
    {
      rig.renderer.gizmo.edit(root, {x, 0.f, 0.f}, glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(1.f), false);
      if(x < 0.15f)
        pauseBeyondMerge();
    }
    rig.renderer.gizmo.edit(root, {to, 0.f, 0.f}, glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(1.f), true);
    rig.host->onUpdate();
  };
  const size_t loaded = undoSteps(rig);
  drag(0.5f);
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Move Box Figure");
  drag(0.8f);
  EXPECT_EQ(undoSteps(rig), loaded + 2);

  rig.cmd("history.undo");
  EXPECT_NEAR(rig.topic("scene").at("props").at(id).at("transform").at("t")[0].get<double>(), 0.5, 1e-5);
  rig.cmd("history.undo");
  EXPECT_EQ(differ(start, rig.topic("scene").at("props").at(id).at("transform")), "");
}

TEST(FvHost, AGizmoDragOfABoneIsOneStep)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id    = figureId(rig.topic("scene"));
  const Json        nodes = rig.topic("nodes/" + id).value("nodes", Json::object());
  int               head  = -1;
  for(const auto& [index, n] : nodes.items())
    if(n.value("name", "") == "head" || n.value("name", "") == "Head")
      head = std::stoi(index);
  ASSERT_GE(head, 0) << nodes.dump();
  ASSERT_TRUE(rig.cmd("select", {{"prop", id}, {"node", head}}).value("ok", false));
  const rs::EntityId bone  = rig.renderer.selectedEntity();
  const auto         headX = [&] { return rig.topic("pose/" + id).at("bones").at("head").at("axes").at("rotation/x").value("value", 0.0); };

  // the rig takes the drag once a frame
  for(float deg : {10.f, 20.f, 30.f})
  {
    rig.renderer.gizmo.edit(bone, glm::vec3(0.f), glm::angleAxis(glm::radians(deg), glm::vec3(1.f, 0.f, 0.f)), glm::vec3(1.f), false);
    rig.host->onUpdate();
    if(deg == 10.f)
      pauseBeyondMerge();
  }
  rig.renderer.gizmo.edit(bone, glm::vec3(0.f), glm::angleAxis(glm::radians(30.f), glm::vec3(1.f, 0.f, 0.f)), glm::vec3(1.f), true);
  rig.host->onUpdate();
  ASSERT_GT(std::abs(headX()), 1.0);
  EXPECT_EQ(rig.topic("history").at("undo"), Json({"Pose Head", "Add Box Figure"}));
  rig.cmd("history.undo");
  EXPECT_NEAR(headX(), 0.0, 1e-4);
}

TEST(FvHost, ADragOfTheViewThroughACameraIsOneStep)
{
  HostRig rig;
  const std::string camera = rig.cmd("prop.add", {{"kind", "camera"}}).at("result").at("prop").get<std::string>();
  rig.cmd("view.use", {{"camera", camera}});
  ASSERT_EQ(undoSteps(rig), 2u);  // added, and looking through it
  const Json start = rig.topic("scene").at("props").at(camera).at("transform");

  const auto orbit = [&](double from) {
    rig.input({{{"t", "down"}, {"b", 0}, {"x", from}, {"y", 300}}});
    rig.input({{{"t", "move"}, {"x", from + 40}, {"y", 310}}});
    pauseBeyondMerge();
    rig.input({{{"t", "move"}, {"x", from + 80}, {"y", 320}}, {{"t", "move"}, {"x", from + 120}, {"y", 330}}, {{"t", "up"}, {"b", 0}, {"x", from + 120}, {"y", 330}}});
  };
  orbit(400);
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Move Camera");
  EXPECT_EQ(undoSteps(rig), 3u);
  orbit(500);
  EXPECT_EQ(undoSteps(rig), 4u);
  EXPECT_NE(differ(start, rig.topic("scene").at("props").at(camera).at("transform")), "");

  rig.cmd("history.undo");
  rig.cmd("history.undo");
  rig.host->onUpdate();
  EXPECT_EQ(differ(start, rig.topic("scene").at("props").at(camera).at("transform")), "");
  EXPECT_EQ(undoSteps(rig), 2u);  // the view going back with the camera is no step of its own
}

namespace {

struct Look
{
  glm::dvec3 eye, center, up;
};

Look look(HostRig& rig)
{
  Look l;
  rig.renderer.getCameraManipulator()->getLookat(l.eye, l.center, l.up);
  return l;
}

// The view at a known place, in the frame size the fake renders
void placeView(HostRig& rig)
{
  auto manip = rig.renderer.getCameraManipulator();
  manip->setWindowSize({1280, 720});
  rig.host->onUpdate();  // the lens takes the frame's aspect
  manip->setLookat({0, 1, 3}, {0, 1, 0}, {0, 1, 0}, true);
}

// A world point's pixel in the current view
glm::dvec2 pixelOf(HostRig& rig, const glm::dvec3& p)
{
  auto             manip = rig.renderer.getCameraManipulator();
  const Look       l     = look(rig);
  const glm::dmat4 view  = glm::lookAt(l.eye, l.center, l.up);
  const glm::dmat4 proj  = glm::perspective(glm::radians(manip->getFov()), 1280.0 / 720.0, 0.01, 100.0);
  const glm::dvec3 win   = glm::project(p, view, proj, glm::dvec4(0, 0, 1280, 720));
  return {win.x, 720.0 - win.y};
}

}  // namespace

TEST(FvHost, AWheelNotchDolliesByTheSameShareAtAnyFrameSize)
{
  HostRig rig;
  placeView(rig);
  rig.input({{{"t", "wheel"}, {"dy", -100}, {"x", 640}, {"y", 360}}});
  const double notch = glm::length(look(rig).center - look(rig).eye);
  EXPECT_NEAR(notch, 3.0 * std::exp(-0.1), 1e-9);
  EXPECT_EQ(look(rig).center, glm::dvec3(0, 1, 0));

  // a touchpad's ten small steps as far as the notch, and a frame twice as wide the same
  placeView(rig);
  rig.renderer.getCameraManipulator()->setWindowSize({2560, 1440});
  Json steps = Json::array();
  for(int i = 0; i < 10; i++)
    steps.push_back({{"t", "wheel"}, {"dy", -10}, {"x", 640}, {"y", 360}});
  rig.input(steps);
  EXPECT_NEAR(glm::length(look(rig).center - look(rig).eye), notch, 1e-9);

  // however far it turns, the eye stops short of the point of interest, and comes back out
  Json many = Json::array();
  for(int i = 0; i < 200; i++)
    many.push_back({{"t", "wheel"}, {"dy", -100}});
  rig.input(many);
  const double closest = glm::length(look(rig).center - look(rig).eye);
  EXPECT_GT(closest, 0.005);
  EXPECT_LT(closest, 0.02);
  rig.input({{{"t", "wheel"}, {"dy", 100}}});
  EXPECT_GT(glm::length(look(rig).center - look(rig).eye), closest);
}

TEST(FvHost, AShiftWheelDolliesTowardThePointer)
{
  HostRig rig;
  placeView(rig);
  // the point under the pointer at the depth of the point of interest
  auto             manip  = rig.renderer.getCameraManipulator();
  const Look       before = look(rig);
  const glm::dmat4 view   = glm::lookAt(before.eye, before.center, before.up);
  const glm::dmat4 proj   = glm::perspective(glm::radians(manip->getFov()), 1280.0 / 720.0, 0.01, 100.0);
  const glm::dvec4 port(0, 0, 1280, 720);
  const double     depth = glm::project(before.center, view, proj, port).z;
  const glm::dvec3 under = glm::unProject(glm::dvec3(1000, 720 - 200, depth), view, proj, port);

  rig.input({{{"t", "wheel"}, {"dy", -300}, {"x", 1000}, {"y", 200}, {"shift", true}}});
  const Look after = look(rig);
  EXPECT_LT(glm::length(after.center - after.eye), 3.0 * 0.8);
  EXPECT_NEAR(glm::dot(glm::normalize(after.center - after.eye), glm::dvec3(0, 0, -1)), 1.0, 1e-9);  // still looks the same way
  const glm::dvec2 pixel = pixelOf(rig, under);
  EXPECT_NEAR(pixel.x, 1000, 1e-6);
  EXPECT_NEAR(pixel.y, 200, 1e-6);
}

TEST(FvHost, TheRightButtonLooksAroundAndFliesWithTheMovementKeys)
{
  HostRig rig;
  placeView(rig);
  // a right drag turns the view in place, half a turn across the frame's width: 60 pixels of 1280
  const double yaw = 60.0 / 1280.0 * glm::pi<double>();
  rig.input({{{"t", "down"}, {"b", 2}, {"x", 640}, {"y", 360}}, {{"t", "move"}, {"x", 700}, {"y", 360}}});
  // spread over the frames: some of it at once, nearly all of it a tenth of a second on
  const double soon = look(rig).center.x;
  for(int i = 0; i < 20; i++)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    rig.host->onUpdate();
  }
  Look l = look(rig);
  EXPECT_LT(soon, 0.9 * 3.0 * std::sin(yaw));
  EXPECT_NEAR(l.center.x, 3.0 * std::sin(yaw), 0.02);  // a drag to the right looks right
  EXPECT_NEAR(glm::length(l.eye - glm::dvec3(0, 1, 3)), 0.0, 1e-9);
  EXPECT_NEAR(glm::length(l.center - l.eye), 3.0, 1e-9);

  // the middle button let go during it leaves the right one's look
  rig.input({{{"t", "down"}, {"b", 1}, {"x", 700}, {"y", 360}}, {{"t", "up"}, {"b", 1}, {"x", 700}, {"y", 360}}});
  rig.input({{{"t", "move"}, {"x", 640}, {"y", 360}}});
  // letting the right button go turns the rest at once, the end of its step
  rig.input({{{"t", "up"}, {"b", 2}, {"x", 640}, {"y", 360}}});
  l = look(rig);
  EXPECT_NEAR(l.center.x, 0.0, 1e-9);

  // up and down turn at the rate across
  rig.input({{{"t", "down"}, {"b", 2}, {"x", 640}, {"y", 360}}, {{"t", "move"}, {"x", 640}, {"y", 420}}, {{"t", "up"}, {"b", 2}, {"x", 640}, {"y", 420}}});
  l = look(rig);
  EXPECT_NEAR(std::asin((l.eye.y - l.center.y) / 3.0), yaw, 1e-9);  // a drag down looks down
  placeView(rig);
  rig.input({{{"t", "down"}, {"b", 2}, {"x", 640}, {"y", 360}}});

  // W flies forward, keeping the point of interest ahead at its distance
  rig.input({{{"t", "fly"}, {"m", {0, 0, 1}}}});
  for(int i = 0; i < 10; i++)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    rig.input({{{"t", "fly"}, {"m", {0, 0, 1}}}});
  }
  Look flown = look(rig);
  EXPECT_LT(flown.eye.z, 3.0 - 0.02);
  EXPECT_NEAR(flown.eye.x, 0.0, 1e-9);
  EXPECT_NEAR(flown.eye.y, 1.0, 1e-9);
  EXPECT_NEAR(glm::length(flown.center - flown.eye), 3.0, 1e-9);
  // no faster than the empty scene's speed (half a metre a second) allows, with slack for a slow machine
  EXPECT_GT(flown.eye.z, 3.0 - 0.5 * 1.0);

  // Q sinks along the world's up
  const double y = flown.eye.y;
  rig.input({{{"t", "fly"}, {"m", {0, -1, 0}}}});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  rig.input({{{"t", "fly"}, {"m", {0, -1, 0}}}});
  EXPECT_LT(look(rig).eye.y, y);

  // the keys held but the button let go: no more flying
  rig.input({{{"t", "up"}, {"b", 2}, {"x", 640}, {"y", 360}}});
  flown = look(rig);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  rig.input({{{"t", "fly"}, {"m", {0, 0, 1}}}});
  EXPECT_EQ(look(rig).eye, flown.eye);
}

TEST(FvHost, ADoubleClickTurnsTheViewToTheSurfaceUnderIt)
{
  HostRig rig;
  // the batch a double-click sends: two clicks, then the browser's dblclick
  const auto click = [](double x) {
    return Json::array({{{"t", "down"}, {"b", 0}, {"x", x}, {"y", 180}}, {{"t", "up"}, {"b", 0}, {"x", x}, {"y", 180}}});
  };
  Json batch = click(320);
  for(const Json& e : click(320))
    batch.push_back(e);
  batch.push_back({{"t", "dblclick"}, {"x", 320}, {"y", 180}});
  rig.input(batch);
  // the first click selects, the second picks nothing (it would let go of a mesh the first selected),
  // and the double-click picks the point to turn to
  ASSERT_EQ(rig.renderer.picks.size(), 2u);
  EXPECT_FALSE(rig.renderer.picks[0].second);
  EXPECT_TRUE(rig.renderer.picks[1].second);
  EXPECT_FLOAT_EQ(rig.renderer.picks[1].first.x, 0.25f);
  EXPECT_FLOAT_EQ(rig.renderer.picks[1].first.y, 0.25f);

  // a third click picks again, and a click elsewhere soon after is a click of its own
  rig.input(click(320));
  rig.input(click(900));
  ASSERT_EQ(rig.renderer.picks.size(), 4u);
  EXPECT_FALSE(rig.renderer.picks[2].second);
  EXPECT_FALSE(rig.renderer.picks[3].second);
}

TEST(FvHost, FlyingStopsWhenTheKeysStopComing)
{
  HostRig rig;
  placeView(rig);
  rig.input({{{"t", "down"}, {"b", 2}, {"x", 640}, {"y", 360}}, {{"t", "fly"}, {"m", {1, 0, 0}}}});
  // the app went quiet with a key held (a release that got lost): the view stops moving
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  rig.host->onUpdate();
  const glm::dvec3 eye = look(rig).eye;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  rig.host->onUpdate();
  EXPECT_EQ(look(rig).eye, eye);
  EXPECT_LT(eye.x, 0.1);  // and never moved a frame further than the longest step
}

TEST(FvHost, ASlidersGestureIsOneStep)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  for(int i = 1; i <= 50; i++)
  {
    rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", i / 50.0}}}, {"gesture", {{"id", "slider 1"}, {"end", i == 50}}}});
    if(i == 25)
      pauseBeyondMerge();
  }
  EXPECT_EQ(rig.topic("history").at("undo"), Json({"Morph Taper", "Add Box Figure"}));
  // the same again after its end, and a pose slider's: steps of their own
  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Taper", 0.5}}}, {"gesture", {{"id", "slider 1"}}}});
  rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/y", 10.0}}}, {"gesture", {{"id", "slider 2"}}}});
  pauseBeyondMerge();
  rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/y", 20.0}}}, {"gesture", {{"id", "slider 2"}, {"end", true}}}});
  EXPECT_EQ(undoSteps(rig), 4u);

  rig.cmd("history.undo");
  rig.cmd("history.undo");
  rig.cmd("history.undo");
  EXPECT_EQ(rig.cmd("scene.state", {{"prop", id}}).at("result").at("morphs").value("Taper", 0.0), 0.0);
}

namespace {

// The app's texture server as the host sees it (GET /<token>/<root>/<rel>?...&path=1): a path to the
// cooked file, 202 for one not cooked yet when only the cache is asked (cached=1). Only the detail
// normal map is cooked to begin with; the cooks of the others wait until opened.
class FakeCooker
{
public:
  FakeCooker()
  {
    fs::create_directories(m_dir);
    m_server.Get(R"(/tok/r0/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
      const std::string rel     = req.matches[1];
      const bool        flipped = req.get_param_value("flip") == "1";
      if(req.get_param_value("cached") == "1" && rel.find("BoxDetail_NM") == std::string::npos)
      {
        res.status = 202;
        return;
      }
      // a cook for the cache (flipped) waits to be let through; a fallback's (unflipped) does not
      if(flipped && req.get_param_value("cached") != "1")
      {
        std::unique_lock lock(m_mutex);
        m_open.wait_for(lock, std::chrono::seconds(20), [this] { return m_opened; });
      }
      // a file per request, its rows' way in its name
      const fs::path file = m_dir / (std::to_string(std::hash<std::string>{}(req.target)) + (flipped ? "-up.ktx2" : "-down.ktx2"));
      std::ofstream(file, std::ios::binary) << "KTX";
      res.set_content(file.generic_string(), "text/plain");
    });
    m_port   = m_server.bind_to_any_port("127.0.0.1");
    m_thread = std::thread([this] { m_server.listen_after_bind(); });
  }
  ~FakeCooker()
  {
    open();
    m_server.stop();
    m_thread.join();
    std::error_code ec;
    fs::remove_all(m_dir, ec);
  }
  std::string url() const { return "http://127.0.0.1:" + std::to_string(m_port) + "/tok"; }
  void        open()
  {
    {
      std::lock_guard lock(m_mutex);
      m_opened = true;
    }
    m_open.notify_all();
  }

private:
  const fs::path          m_dir = fs::temp_directory_path() / "fv-host-cooker";
  httplib::Server         m_server;
  std::thread             m_thread;
  int                     m_port = 0;
  std::mutex              m_mutex;
  std::condition_variable m_open;
  bool                    m_opened = false;
};

// The figure's images as the renderer's scene has them, and how many of its texture slots sample
// theirs with V turned over (a source in a figure built for the cooked images)
struct Sampled
{
  std::vector<std::string> images;
  int                      turned = 0;
};
Sampled sampled(rs::RetainedScene& scene)
{
  Sampled               out;
  const tinygltf::Model model = rs::exportGltf(scene);
  for(const tinygltf::Image& image : model.images)
    out.images.push_back(image.uri);
  const std::function<void(const tinygltf::Value&)> walk = [&](const tinygltf::Value& v) {
    if(v.IsObject())
    {
      if(v.Has("KHR_texture_transform") && v.Get("KHR_texture_transform").Get("scale").Get(1).GetNumberAsDouble() < 0)
        out.turned++;
      for(const std::string& key : v.Keys())
        walk(v.Get(key));
    }
    else if(v.IsArray())
      for(size_t i = 0; i < v.ArrayLen(); i++)
        walk(v.Get(int(i)));
  };
  for(const tinygltf::Material& m : model.materials)
    for(const tinygltf::ExtensionMap* extensions : {&m.pbrMetallicRoughness.baseColorTexture.extensions, &m.normalTexture.extensions,
                                                    &m.emissiveTexture.extensions, &m.pbrMetallicRoughness.metallicRoughnessTexture.extensions,
                                                    &m.extensions})
      for(const auto& [name, value] : *extensions)
        if(name == "KHR_texture_transform")
          out.turned += value.Get("scale").Get(1).GetNumberAsDouble() < 0 ? 1 : 0;
        else
          walk(value);
  return out;
}

}  // namespace

// A figure loaded on a cold texture cache shows the cooked maps the cache has and, for the rest, the
// fallbacks, sampled with V turned over; each cook swaps in as it lands, the fallbacks go from the
// renderer's scene and every image ends up cooked, without a reload.
TEST(FvHost, AFigureOnAColdCacheTakesEachCookAsItLands)
{
  FakeCooker cooker;
  HostRig    rig;
  ASSERT_TRUE(rig.cmd("settings.apply", {{"texServer", cooker.url()}}).value("ok", false));
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure PBRSkin.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  rs::RetainedScene& scene  = rig.renderer.hostScene();
  const Sampled      before = sampled(scene);
  ASSERT_GE(before.images.size(), 2u);
  EXPECT_GT(before.turned, 0);
  const auto cooked = [](const std::vector<std::string>& images) {
    return size_t(std::count_if(images.begin(), images.end(), [](const std::string& uri) { return uri.ends_with("-up.ktx2"); }));
  };
  EXPECT_EQ(cooked(before.images), 1u);  // the cached detail map

  cooker.open();
  Sampled   after;
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  do
  {
    rig.host->onUpdate();
    after = sampled(scene);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } while(cooked(after.images) < after.images.size() && std::chrono::steady_clock::now() < end);
  EXPECT_EQ(cooked(after.images), after.images.size());
  EXPECT_EQ(after.turned, 0);
  EXPECT_EQ(after.images.size(), before.images.size());
  EXPECT_EQ(scene.textureCount(), after.images.size());  // the fallbacks went
}

// The frame as a linear image, once it is on disk: the path tracer's, a .hdr
TEST(FvHost, ACaptureIsAnHdrOfThePathTracersImage)
{
  HostRig                     rig;
  const fs::path file = fs::temp_directory_path() / "fv_capture_test.hdr";

  rig.renderer.setRenderSystem(RenderingMode::eRasterizer);
  EXPECT_FALSE(rig.cmd("render.capture", {{"path", file.generic_string()}}).value("ok", false));
  EXPECT_TRUE(rig.renderer.savedImage.empty());

  rig.renderer.setRenderSystem(RenderingMode::ePathtracer);
  EXPECT_FALSE(rig.cmd("render.capture", {{"path", "still.png"}}).value("ok", false));

  rig.renderer.stillSamples = 64;
  const Json answer         = rig.cmd("render.capture", {{"path", file.generic_string()}});
  EXPECT_TRUE(answer.value("ok", false)) << answer.dump();
  EXPECT_EQ(answer["result"].value("samples", 0), 64);
  EXPECT_EQ(rig.renderer.savedImage, file);
  fs::remove(file);
}

// A scene file makes a prop where it has it: no frame of it stands anywhere else, and there's nothing
// left to send for it after its load
TEST(FvHost, APropIsMadeWhereItsPlaceSays)
{
  HostRig    rig;
  const Json place = {{"name", "Stage"}, {"visible", false}, {"transform", {{"t", {1.5, 0.0, -2.0}}, {"r", {0.0, 0.0, 0.0, 1.0}}, {"s", {2.0, 2.0, 2.0}}}}};
  ASSERT_TRUE(rig.cmd("prop.add", {{"rel", "People/Test/Box Figure.duf"}, {"place", place}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const Json scene = rig.topic("scene");
  const Json f     = scene.at("props").at(figureId(scene));
  EXPECT_EQ(f.at("name"), "Stage");
  EXPECT_EQ(f.at("visible"), false);
  EXPECT_NEAR(f.at("transform").at("t")[0].get<double>(), 1.5, 1e-6);
  EXPECT_NEAR(f.at("transform").at("s")[1].get<double>(), 2.0, 1e-6);
  const rs::Entity* root = nullptr;
  for(rs::EntityId id : rig.renderer.hostScene().roots())
    if(const rs::Entity* e = rig.renderer.hostScene().entity(id); e && e->local.scale.x == 2.f)
      root = e;
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(root->local.translation, glm::vec3(1.5f, 0.f, -2.f));
  EXPECT_FALSE(root->visible);

  // a camera and a light take theirs at once
  const Json camera = rig.cmd("prop.add", {{"kind", "camera"}, {"place", {{"name", "Close"}, {"camera", {{"focalLength", 85}}}, {"transform", {{"t", {0.0, 1.0, 2.0}}}}}}}).at("result");
  const Json c      = rig.topic("scene").at("props").at(camera.at("prop").get<std::string>());
  EXPECT_EQ(c.at("name"), "Close");
  EXPECT_NEAR(c.at("camera").at("focalLength").get<double>(), 85, 1e-6);
  EXPECT_NEAR(c.at("transform").at("t")[2].get<double>(), 2.0, 1e-6);
}

// A file's saved view is set before its figures come, so the first of them isn't framed over it; a
// clear without one frames as before, and it holds for that one load only
TEST(FvHost, AClearThatKeepsTheViewLeavesTheFirstFigureUnframed)
{
  HostRig rig;
  placeView(rig);
  rig.cmd("scene.clear", {{"keepView", true}});
  rig.cmd("camera.goto", {{"position", {3.0, 2.0, 9.0}}, {"target", {1.0, 1.0, 0.0}}});
  ASSERT_TRUE(rig.cmd("prop.add", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  for(int i = 0; i < 4; i++)
    rig.host->onUpdate();
  const auto closeTo = [](const glm::dvec3& a, const glm::dvec3& b) { return glm::length(a - b) < 1e-3; };
  EXPECT_TRUE(closeTo(look(rig).eye, {3, 2, 9}));
  EXPECT_TRUE(closeTo(look(rig).center, {1, 1, 0}));

  rig.cmd("scene.clear");
  rig.cmd("camera.goto", {{"position", {3.0, 2.0, 9.0}}, {"target", {1.0, 1.0, 0.0}}});
  ASSERT_TRUE(rig.settle());
  ASSERT_TRUE(rig.cmd("prop.add", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  for(int i = 0; i < 4; i++)
    rig.host->onUpdate();
  EXPECT_FALSE(closeTo(look(rig).eye, {3, 2, 9}));
}

// A restore's worn items are fed once, by its composition, with the materials the file gives them: fed
// as they are committed, a heavy mesh is built and made resident twice in every open
TEST(FvHost, AWearIsFedByItsCompositionNotBeforeIt)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));

  std::mutex  mutex;
  int         fedWhole = 0;
  fv::timing::setSink([&](const std::string& line) {
    std::lock_guard lock(mutex);
    if(line.find("fed whole, outside a composition") != std::string::npos)
      fedWhole++;
  });
  const Json worn = Json::array({Json{{"rel", "People/Test/Clothing/Box Shirt.duf"}, {"label", ""}, {"visible", true}}});
  ASSERT_TRUE(rig.cmd("scene.restore", {{"prop", id}, {"worn", worn}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  // a wear from the library, as well
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  for(int i = 0; i < 4; i++)
    rig.host->onUpdate();
  fv::timing::setSink({});

  EXPECT_EQ(fedWhole, 0);
  // and what's worn is in the scene
  EXPECT_FALSE(rig.topic("wardrobe/" + id).value("items", Json::object()).empty());
}

// Figures take their worn items side by side, each its own: nothing of one reaches the other
TEST(FvHost, TwoFiguresRestoredAtOnceEachGetTheirOwn)
{
  HostRig rig;
  for(int i = 0; i < 2; i++)
    ASSERT_TRUE(rig.cmd("prop.add", {{"rel", "People/Test/Box Figure.duf"}, {"place", {{"name", "Box " + std::to_string(i)}}}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  std::vector<std::string> ids;
  const Json               scene = rig.topic("scene");
  for(const auto& [id, p] : scene.at("props").items())
    if(p.value("kind", "") == "figure")
      ids.push_back(id);
  ASSERT_EQ(ids.size(), 2u);

  const auto worn = [](std::initializer_list<const char*> rels) {
    Json out = Json::array();
    for(const char* rel : rels)
      out.push_back(Json{{"rel", rel}, {"label", ""}, {"visible", true}});
    return out;
  };
  for(int round = 0; round < 5; round++)
  {
    // the second figure's asked while the first's is still going
    const Json a = rig.cmd("scene.restore", {{"prop", ids[0]}, {"worn", worn({"People/Test/Clothing/Box Shirt.duf"})}});
    const Json b = rig.cmd("scene.restore", {{"prop", ids[1]}, {"worn", worn({"People/Test/Clothing/Box Shirt.duf", "People/Test/Clothing/Box Shirt.duf"})}});
    ASSERT_TRUE(a.value("ok", false)) << a.dump();
    ASSERT_TRUE(b.value("ok", false)) << b.dump();
    ASSERT_TRUE(rig.settle());
    const auto problems = [&](const Json& answer) { return rig.topic("jobs").value(answer.at("result").at("job").dump(), Json::object()).value("problems", Json::array()); };
    EXPECT_TRUE(problems(a).empty()) << problems(a).dump();
    EXPECT_TRUE(problems(b).empty()) << problems(b).dump();
  }
  for(const std::string& id : ids)
    EXPECT_FALSE(rig.topic("wardrobe/" + id).value("items", Json::object()).empty()) << id;
}
