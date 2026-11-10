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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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
#include "gizmo_pivot.hpp"
#include "gizmo_snap.hpp"
#include <figure/dson/dforce.hpp>
#include <figure/dson/listing.hpp>
#include <figure/dson/store.hpp>
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
    if(p.value("kind", "") == "character")
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

// An export holds the textures where they are, and what is lowered reaches the app as a topic
TEST(FvHost, TexturesFollowingMemoryReachTheApp)
{
  HostRig rig;
  EXPECT_TRUE(rig.cmd("render.hold", {{"on", true}}).value("ok", false));
  EXPECT_TRUE(rig.renderer.texturesHeld);
  EXPECT_TRUE(rig.cmd("render.hold", {{"on", false}}).value("ok", false));
  EXPECT_FALSE(rig.renderer.texturesHeld);

  ASSERT_TRUE(rig.renderer.memorySink);
  rig.renderer.memorySink({.lowered = 12, .largest = 1024, .parked = 2, .spilledBytes = 1 << 30});
  const Json memory = rig.topic("memory");
  EXPECT_EQ(memory.value("lowered", 0), 12) << memory.dump();
  EXPECT_EQ(memory.value("largest", 0), 1024) << memory.dump();
  EXPECT_EQ(memory.value("parked", 0), 2) << memory.dump();
  EXPECT_EQ(memory.value("spilledBytes", uint64_t(0)), uint64_t(1) << 30) << memory.dump();
}

// What is taken off leaves the renderer's scene at once, so its GPU memory is there for what comes next
TEST(FvHost, AnItemTakenOffLeavesTheScene)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  const size_t      bare = rig.renderer.hostScene().drawItems().size();
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  const size_t dressed = rig.renderer.hostScene().drawItems().size();
  ASSERT_GT(dressed, bare);

  const Json worn = rig.topic("wardrobe/" + id).value("items", Json::object());
  ASSERT_EQ(worn.size(), 1u);
  ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", worn.begin().key()}}).value("ok", false));
  EXPECT_EQ(rig.renderer.hostScene().drawItems().size(), bare);
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

// What GPU memory can't take comes all the same, said so: it runs from system memory, slower
TEST(FvHost, WhatGpuMemoryCantTakeComesAllTheSameAndSaysSo)
{
  HostRig rig;
  rig.renderer.geometryRefusal = "no room";
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_GT(rig.renderer.askedGeometry, 0u);
  EXPECT_EQ(rig.renderer.admittedGeometry, rig.renderer.askedGeometry);
  const std::string id = figureId(rig.topic("scene"));
  ASSERT_FALSE(id.empty());
  EXPECT_NE(rig.topic("notices").dump().find("came in past GPU memory: no room"), std::string::npos) << rig.topic("notices").dump();
  const size_t bare = rig.renderer.hostScene().meshCount();
  EXPECT_GT(bare, 0u);

  // and a wear
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  EXPECT_GT(rig.renderer.hostScene().meshCount(), bare);
  const Json items = rig.topic("wardrobe/" + id).value("items", Json::object());
  EXPECT_NE(items.dump().find("Box Shirt"), std::string::npos) << items.dump();
  EXPECT_NE(rig.topic("notices").dump().find("Box Shirt came in past GPU memory: no room"), std::string::npos) << rig.topic("notices").dump();
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
  // what GPU memory can't take is put back too, slower: nothing to report, so the scene stays its file's
  rig.renderer.geometryRefusal = "no room";
  EXPECT_TRUE(restored("People/Test/Clothing/Box Shirt.duf").empty());
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
// A DAZ animation played on a figure: the test library's Box Nod Animation turns the head from 0 to
// 40 degrees about y and the CTRL Nod dial from 0 to 1 over a second

namespace {

struct Nodding
{
  HostRig     rig;
  std::string id;

  // the box figure, and the animation opened onto it from the Library (`as` "") or imported ("animation")
  explicit Nodding(const std::string& as)
  {
    EXPECT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    id = figureId(rig.topic("scene"));
    EXPECT_FALSE(id.empty());
    open(as);
  }
  void open(const std::string& as)
  {
    Json args = {{"prop", id}, {"rel", "People/Test/Poses/Box Nod Animation.duf"}};
    if(!as.empty())
      args["as"] = as;
    EXPECT_TRUE(rig.cmd("load", args).value("ok", false));
    EXPECT_TRUE(rig.settle());
  }
  Json   anim() const { return rig.topic("animation"); }
  int    current() const { return anim().value("current", Json()).is_number() ? anim()["current"].get<int>() : -1; }
  double headY() const { return rig.topic("pose/" + id).at("bones").at("head").at("axes").at("rotation/y").value("value", -1.0); }
  double nod() const { return rig.topic("morphs/" + id).at("channels").value("CTRL Nod", Json::object()).value("value", -1.0); }
};

}  // namespace

TEST(FvHost, APoseFileWithKeysFromTheLibraryPosesAtItsEndAndWaitsAsAClip)
{
  Nodding n("");
  const Json anim = n.anim();
  ASSERT_GE(n.current(), 0) << anim.dump();
  const Json clip = anim.at("clips").at(std::to_string(n.current()));
  EXPECT_EQ(clip.value("kind", ""), "figure");
  EXPECT_EQ(clip.value("prop", ""), n.id);
  EXPECT_EQ(clip.value("name", ""), "Box Nod Animation");
  EXPECT_DOUBLE_EQ(clip.value("duration", 0.0), 1.0);
  // a pose preset as before, its last key: many such files are still poses saved for dForce
  EXPECT_FALSE(anim.value("playing", true));
  EXPECT_DOUBLE_EQ(anim.value("time", 0.0), 1.0);
  EXPECT_DOUBLE_EQ(n.headY(), 40.0);
  EXPECT_DOUBLE_EQ(n.nod(), 1.0);
  EXPECT_NE(n.rig.topic("notices").dump().find("the Anim tab plays it"), std::string::npos) << n.rig.topic("notices").dump();
}

TEST(FvHost, AnImportedAnimationPlaysAndASeekPosesTheFigureAtThatTime)
{
  Nodding n("animation");
  ASSERT_GE(n.current(), 0);
  EXPECT_TRUE(n.anim().value("playing", false));
  EXPECT_TRUE(n.rig.renderer.hostClip().isNone());

  n.rig.cmd("anim.seek", {{"time", 0.25}});
  EXPECT_FALSE(n.anim().value("playing", true));
  EXPECT_DOUBLE_EQ(n.anim().value("time", 0.0), 0.25);
  EXPECT_NEAR(n.headY(), 10.0, 1e-9);
  EXPECT_NEAR(n.nod(), 0.25, 1e-9);

  // a still pose file is no animation to import
  n.rig.cmd("load", {{"prop", n.id}, {"rel", "People/Test/Poses/Box Look Left.duf"}, {"as", "animation"}});
  ASSERT_TRUE(n.rig.settle());
  EXPECT_NE(n.rig.topic("notices").dump().find("is a still pose"), std::string::npos) << n.rig.topic("notices").dump();
  EXPECT_NEAR(n.headY(), 10.0, 1e-9);
}

TEST(FvHost, StopPutsTheFigureBackAndRemoveTakesTheAnimationOff)
{
  Nodding n("animation");
  const int clip = n.current();
  ASSERT_GE(clip, 0);
  n.rig.cmd("anim.seek", {{"time", 0.5}});
  EXPECT_NEAR(n.headY(), 20.0, 1e-9);

  // the pose it found
  n.rig.cmd("anim.stop");
  EXPECT_DOUBLE_EQ(n.headY(), 0.0);
  EXPECT_DOUBLE_EQ(n.nod(), 0.0);
  EXPECT_FALSE(n.anim().value("playing", true));

  n.rig.cmd("anim.select", {{"clip", clip}, {"play", false}});
  n.rig.cmd("anim.seek", {{"time", 1.0}});
  EXPECT_DOUBLE_EQ(n.headY(), 40.0);
  const Json removed = n.rig.cmd("anim.remove", {{"clip", clip}});
  ASSERT_TRUE(removed.value("ok", false)) << removed.dump();
  EXPECT_DOUBLE_EQ(n.headY(), 0.0);
  EXPECT_TRUE(n.anim().at("clips").empty()) << n.anim().dump();
  EXPECT_TRUE(n.anim().at("current").is_null());
}

// A clip plays through Physics mode: the ragdoll stays awake while it plays, even where the clip
// holds still, and once it pauses the ragdoll comes to rest where the clip left it.
TEST(FvHost, aClipPlaysThroughPhysicsMode)
{
  Nodding n("animation");
  ASSERT_TRUE(n.anim().value("playing", false));
  ASSERT_TRUE(n.rig.cmd("physics.set", {{"prop", n.id}, {"on", true}, {"strength", 1.0}}).value("ok", false));
  const auto awake = [&] { return n.rig.topic("physics/" + n.id).value("awake", false); };
  for(int f = 0; f < 30; f++)
  {
    n.rig.host->onUpdate();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if(f > 0 && n.anim().value("playing", false))
      ASSERT_TRUE(awake()) << "asleep while the clip plays, frame " << f;
  }
  n.rig.cmd("anim.seek", {{"time", 0.5}});
  EXPECT_FALSE(n.anim().value("playing", true));
  EXPECT_NEAR(n.headY(), 20.0, 1e-9);
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(6);
  while(awake() && std::chrono::steady_clock::now() < end)
  {
    n.rig.host->onUpdate();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_FALSE(awake()) << "still moving after the clip paused";
}

TEST(FvHost, AnotherAnimationOnTheFigureTakesThePlaceOfTheFirst)
{
  Nodding   n("");
  const int first = n.current();
  ASSERT_GE(first, 0);
  n.open("animation");
  const Json anim = n.anim();
  EXPECT_EQ(anim.at("clips").size(), 1u) << anim.dump();
  EXPECT_NE(n.current(), first);
  // what it goes back to is what the figure had before the first
  n.rig.cmd("anim.stop");
  EXPECT_DOUBLE_EQ(n.headY(), 0.0);
}

TEST(FvHost, AFigureAnimationStaysCurrentThroughACompositionAndLeavesWithItsFigure)
{
  Nodding   n("animation");
  const int clip = n.current();
  ASSERT_GE(clip, 0);
  n.rig.cmd("wear", {{"prop", n.id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(n.rig.settle());
  EXPECT_EQ(n.current(), clip) << n.anim().dump();
  n.rig.cmd("anim.seek", {{"time", 0.5}});
  EXPECT_NEAR(n.headY(), 20.0, 1e-9);

  ASSERT_TRUE(n.rig.cmd("prop.remove", {{"prop", n.id}}).value("ok", false));
  ASSERT_TRUE(n.rig.settle());
  EXPECT_TRUE(n.anim().at("clips").empty()) << n.anim().dump();
  EXPECT_TRUE(n.anim().at("current").is_null());
}

// A scene file keeps a figure's animation: the file it came from, where it was, whether it played,
// and the pose it found, which Stop puts back after it reopens in another renderer
TEST(FvHost, AnAnimationComesBackWithItsSceneWhereItWas)
{
  const auto saved = [](bool playing) {
    HostRig rig;
    EXPECT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    const std::string id = figureId(rig.topic("scene"));
    // the pose it finds, turned already
    rig.cmd("pose.set", {{"prop", id}, {"values", {{"head/rotation/y", 15.0}}}});
    EXPECT_TRUE(rig.cmd("load", {{"prop", id}, {"rel", "People/Test/Poses/Box Nod Animation.duf"}, {"as", "animation"}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    rig.cmd("anim.seek", {{"time", 0.5}});
    if(playing)
      rig.cmd("anim.play");
    return rig.cmd("scene.state", {{"prop", id}}).value("result", Json::object());
  };
  // in a renderer of its own, onto the same figure loaded again
  const auto reopened = [](Json state) {
    auto rig = std::make_unique<HostRig>();
    EXPECT_TRUE(rig->cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    EXPECT_TRUE(rig->settle());
    const std::string id = figureId(rig->topic("scene"));
    state["prop"]        = id;
    EXPECT_TRUE(rig->cmd("scene.restore", state).value("ok", false));
    EXPECT_TRUE(rig->settle());
    return std::pair{std::move(rig), id};
  };
  const auto headY = [](HostRig& rig, const std::string& id) {
    return rig.topic("pose/" + id).at("bones").at("head").at("axes").at("rotation/y").value("value", -1.0);
  };
  const auto nod = [](HostRig& rig, const std::string& id) {
    return rig.topic("morphs/" + id).at("channels").value("CTRL Nod", Json::object()).value("value", -1.0);
  };

  // paused halfway: the file by its library path, the time, and what the pose was before it
  const Json paused = saved(false);
  ASSERT_TRUE(paused.contains("animation")) << paused.dump();
  const Json& anim = paused["animation"];
  EXPECT_EQ(anim.value("rel", ""), "People/Test/Poses/Box Nod Animation.duf");
  EXPECT_DOUBLE_EQ(anim.value("time", 0.0), 0.5);
  EXPECT_FALSE(anim.value("playing", true));
  EXPECT_DOUBLE_EQ(anim.at("before").at("bones").value("head/rotation/y", 0.0), 15.0) << anim.dump();
  EXPECT_DOUBLE_EQ(anim.at("before").at("dials").value("CTRL Nod", -1.0), 0.0);
  {
    auto [rig, id] = reopened(paused);
    const Json topic = rig->topic("animation");
    ASSERT_EQ(topic.at("clips").size(), 1u) << topic.dump();
    EXPECT_FALSE(topic.value("playing", true));
    EXPECT_DOUBLE_EQ(topic.value("time", 0.0), 0.5);
    rig->host->onUpdate();
    EXPECT_NEAR(headY(*rig, id), 20.0, 1e-9);
    EXPECT_NEAR(nod(*rig, id), 0.5, 1e-9);
    // no notice that it plays: the scene put it back
    EXPECT_EQ(rig->topic("notices").dump().find("Playing"), std::string::npos) << rig->topic("notices").dump();
    // the pose it found before the scene was saved
    rig->cmd("anim.stop");
    EXPECT_DOUBLE_EQ(headY(*rig, id), 15.0);
    EXPECT_DOUBLE_EQ(nod(*rig, id), 0.0);
  }

  // playing: it plays again
  const Json playing = saved(true);
  EXPECT_TRUE(playing.at("animation").value("playing", false)) << playing.dump();
  {
    auto [rig, id] = reopened(playing);
    EXPECT_TRUE(rig->topic("animation").value("playing", false)) << rig->topic("animation").dump();
  }

  // a file no longer in the library: the figure comes back in the pose it was saved in, and says why
  Json gone                 = paused;
  gone["animation"]["rel"] = "People/Test/Poses/Gone Animation.duf";
  {
    auto [rig, id] = reopened(gone);
    EXPECT_TRUE(rig->topic("animation").at("clips").empty());
    EXPECT_NEAR(headY(*rig, id), 20.0, 1e-9);
    EXPECT_NE(rig->topic("notices").dump().find("Gone Animation.duf is not installed"), std::string::npos) << rig->topic("notices").dump();
  }
}

// A model's own clip and a figure's animation, one current at a time
TEST(FvHost, AModelsClipAndAFiguresAnimationTakeTurns)
{
  Nodding        n("animation");
  const fs::path dir = fs::temp_directory_path() / "fv_host_clip_test";
  fs::create_directories(dir);
  const fs::path gltf = dir / "slide.gltf";
  std::ofstream(gltf) << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"name":"slider"}],
    "buffers":[{"byteLength":32,"uri":"data:application/octet-stream;base64,AAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAA="}],
    "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":8},{"buffer":0,"byteOffset":8,"byteLength":24}],
    "accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR","min":[0],"max":[1]},
                 {"bufferView":1,"componentType":5126,"count":2,"type":"VEC3"}],
    "animations":[{"name":"Slide","samplers":[{"input":0,"output":1}],"channels":[{"sampler":0,"target":{"node":0,"path":"translation"}}]}]})";
  ASSERT_TRUE(n.rig.cmd("load", {{"path", gltf.generic_string()}}).value("ok", false));
  ASSERT_TRUE(n.rig.settle());
  const int  figureClip = n.current();
  int        modelClip  = -1;
  const Json clips      = n.anim().at("clips");
  for(const auto& [number, c] : clips.items())
    if(c.value("kind", "") == "clip")
      modelClip = std::stoi(number);
  ASSERT_GE(modelClip, 0) << n.anim().dump();
  ASSERT_NE(modelClip, figureClip);

  n.rig.cmd("anim.select", {{"clip", modelClip}, {"play", false}});
  EXPECT_EQ(n.current(), modelClip);
  EXPECT_FALSE(n.rig.renderer.hostClip().isNone());
  n.rig.cmd("anim.seek", {{"time", 0.5}});
  EXPECT_FLOAT_EQ(n.rig.renderer.hostClipTime(), 0.5f);

  n.rig.cmd("anim.select", {{"clip", figureClip}, {"play", false}});
  EXPECT_TRUE(n.rig.renderer.hostClip().isNone());
  n.rig.cmd("anim.seek", {{"time", 0.5}});
  EXPECT_NEAR(n.headY(), 20.0, 1e-9);
  // only a figure's animation can be taken off
  EXPECT_FALSE(n.rig.cmd("anim.remove", {{"clip", modelClip}}).value("ok", true));
  fs::remove_all(dir);
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
    if(p.value("format", "") == "DAZ")
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

// lighting.preview: a look shown while the app's pointer is on its chip is no step of the history and
// nothing in the topic; one chosen with lighting.set is a step, and undoes back to none
TEST(FvHost, ALookPreviewedLeavesNoStepAndAChosenOneUndoes)
{
  HostRig    rig;
  const Json before = rig.topic("lighting");
  const Json look   = {{"id", "noir"}, {"name", "Noir"}, {"mono", 1.0}, {"contrast", 1.3}};
  const Json shown  = rig.cmd("lighting.preview", {{"look", look}, {"strength", 1.0}});
  ASSERT_TRUE(shown.value("ok", false)) << shown.dump();
  EXPECT_TRUE(shown.at("result").value("previewing", false));
  EXPECT_EQ(differ(before, rig.topic("lighting")), "");
  EXPECT_EQ(rig.topic("history").at("undo"), Json::array());
  // a bad look is refused; no look ends it
  EXPECT_FALSE(rig.cmd("lighting.preview", {{"look", {{"sparkle", 1}}}}).value("ok", true));
  EXPECT_FALSE(rig.cmd("lighting.preview").at("result").value("previewing", true));

  rig.cmd("lighting.set", {{"tonemap", {{"look", look}, {"lookStrength", 0.8}}}});
  EXPECT_EQ(rig.topic("lighting").at("tonemap").at("look").value("id", ""), "noir");
  EXPECT_EQ(rig.topic("history").at("undo"), Json({"Lighting"}));
  historyStep(rig, "history.undo");
  EXPECT_EQ(differ(before, rig.topic("lighting")), "");
  historyStep(rig, "history.redo");
  EXPECT_EQ(rig.topic("lighting").at("tonemap").at("look").value("name", ""), "Noir");
  EXPECT_EQ(rig.topic("lighting").at("tonemap").value("lookStrength", 0.0), 0.8);
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
        (kind == "character" ? figures : kind == "camera" ? cameras : lights).push_back(id);
      }
      const auto any = [&](const std::vector<std::string>& v) { return v.empty() ? std::string("p:0") : v[rng() % v.size()]; };
      const std::string id = any(figures);
      std::string       item;
      const Json        worn = rig.topic("wardrobe/" + id).value("items", Json::object());
      for(const auto& [key, it] : worn.items())
        item = key;

      std::string cmd;
      Json        args = Json::object();
      switch(rng() % 33)
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
        case 32:
          cmd  = "material.shader";
          args = {{"prop", id}, {"material", "m:" + std::to_string(rng() % 2)}, {"rel", coin() ? Json() : coin() ? Json("Shader Presets/Test/Box Gold.duf") : Json("Shader Presets/Test/Box Tile 2.duf")}};
          break;
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

// The figure's dForce starts off, so hair standing straight out would stay that way until someone
// found the switch: wearing it turns dForce on and it falls. Hair styled as it is leaves it off.
TEST(FvHostWardrobe, HairLeftStandingOutFallsWhenItGoesOnAndHangingHairStaysAsItIs)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  EXPECT_FALSE(rig.topic("dforce/" + id).value("enabled", true));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Hanging Hair.duf"}});
  ASSERT_TRUE(rig.settle());
  EXPECT_FALSE(rig.topic("dforce/" + id).value("enabled", true));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Fan Hair.duf"}});
  ASSERT_TRUE(rig.settle());
  EXPECT_TRUE(rig.topic("dforce/" + id).value("enabled", false));
}

// What GPU memory can't take as it is comes in with less of its geometry, says so, and is refused only
// when it has nothing left to give. Put on with room again it has all of it.
TEST(FvHostWardrobe, AnItemThatDoesNotFitComesInSmaller)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id    = figureId(rig.topic("scene"));
  const auto        items = [&] { return rig.topic("wardrobe/" + id).value("items", Json::object()); };
  const auto        wear  = [&](const char* rel) {
    rig.cmd("wear", {{"prop", id}, {"rel", rel}});
    ASSERT_TRUE(rig.settle());
  };
  const auto takeOff = [&] {
    const Json worn = items();
    for(const auto& [item, value] : worn.items())
      ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", item}}).value("ok", false));
    ASSERT_TRUE(rig.settle());
  };
  const char* hair  = "People/Test/Clothing/Box Fan Hair.duf";
  const char* shirt = "People/Test/Clothing/Box Shirt.duf";

  wear(hair);
  ASSERT_EQ(items().size(), 1u);
  EXPECT_EQ(items().begin().value().value("reduced", "?"), "");
  EXPECT_TRUE(rig.renderer.lastKeptTextures);  // a first try keeps its textures: there is further to go
  const uint64_t hairBytes = rig.renderer.lastGeometry;
  takeOff();

  // a hair's first step keeps every strand, its segments in one piece
  rig.renderer.geometryLimit = hairBytes * 3 / 4;
  wear(hair);
  ASSERT_EQ(items().size(), 1u);
  EXPECT_EQ(items().begin().value().value("reduced", ""), "its strands in half the triangles");
  const uint64_t halfBytes = rig.renderer.lastGeometry;
  EXPECT_LE(halfBytes, rig.renderer.geometryLimit);  // about half: its cage doesn't thin
  EXPECT_GT(halfBytes, hairBytes / 3);
  EXPECT_NE(rig.topic("notices").dump().find("Box Fan Hair came in with its strands in half the triangles, to fit GPU memory"),
            std::string::npos)
      << rig.topic("notices").dump();
  takeOff();

  // then half its strands
  rig.renderer.geometryLimit = halfBytes * 3 / 4;
  wear(hair);
  ASSERT_EQ(items().size(), 1u);
  EXPECT_EQ(items().begin().value().value("reduced", ""), "half its strands");
  EXPECT_LE(rig.renderer.lastGeometry, rig.renderer.geometryLimit);
  EXPECT_NE(rig.topic("notices").dump().find("Box Fan Hair came in with half its strands, to fit GPU memory"), std::string::npos) << rig.topic("notices").dump();
  takeOff();
  rig.renderer.geometryLimit = hairBytes * 3 / 4;
  wear(hair);
  ASSERT_EQ(items().begin().value().value("reduced", ""), "its strands in half the triangles");

  // on again, what it replaces leaves as it comes: all of it needs room only for the difference
  wear(hair);
  ASSERT_EQ(items().size(), 1u);
  EXPECT_EQ(items().begin().value().value("reduced", "?"), "");
  EXPECT_EQ(rig.renderer.lastGeometry, hairBytes - halfBytes);
  takeOff();

  // a garment comes down a level, from the host's subdivision 1
  rig.renderer.geometryLimit = UINT64_MAX;
  wear(shirt);
  const uint64_t shirtBytes = rig.renderer.lastGeometry;
  takeOff();
  rig.renderer.geometryLimit = shirtBytes / 2;
  wear(shirt);
  ASSERT_EQ(items().size(), 1u);
  EXPECT_EQ(items().begin().value().value("reduced", ""), "Subdivision 0");
  EXPECT_LE(rig.renderer.lastGeometry, rig.renderer.geometryLimit);
  takeOff();

  // nothing small enough: the last try counts the textures all the way down, and it comes at its smallest
  rig.renderer.geometryLimit = 1;
  wear(hair);
  ASSERT_EQ(items().size(), 1u);
  EXPECT_EQ(items().begin().value().value("reduced", ""), "a quarter of its strands");
  EXPECT_FALSE(rig.renderer.lastKeptTextures);
  EXPECT_NE(rig.topic("notices").dump().find("Box Fan Hair came in past GPU memory: no room"), std::string::npos) << rig.topic("notices").dump();
  takeOff();

  rig.renderer.geometryLimit = UINT64_MAX;
  wear(hair);
  ASSERT_EQ(items().size(), 1u);
  EXPECT_EQ(items().begin().value().value("reduced", "?"), "");
}

// A hair's Volume: published for the slider, one step of undo, and kept by a scene file only once moved.
TEST(FvHostWardrobe, AHairsVolumeIsPublishedUndoneAndSaved)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Fan Hair.duf"}});
  ASSERT_TRUE(rig.settle());
  const auto items = [&] { return rig.topic("wardrobe/" + id).value("items", Json::object()); };
  ASSERT_EQ(items().size(), 1u);
  const std::string item   = items().begin().key();
  const auto        volume = [&] { return items()[item].at("cloth").value("volume", -1.f); };
  EXPECT_TRUE(items()[item].at("cloth").value("strands", false));
  EXPECT_FLOAT_EQ(volume(), fv::dson::kStrandVolume);
  const auto savedCloth = [&]() -> Json {
    for(const Json& w : rig.cmd("scene.state", {{"prop", id}}).at("result").value("worn", Json::array()))
      if(w.contains("cloth"))
        return w["cloth"];
    return nullptr;
  };
  EXPECT_TRUE(savedCloth().is_null());

  rig.cmd("wardrobe.cloth", {{"prop", id}, {"item", item}, {"volume", 0.8}});
  EXPECT_FLOAT_EQ(volume(), 0.8f);
  EXPECT_FLOAT_EQ(savedCloth().value("volume", -1.f), 0.8f);
  historyStep(rig, "history.undo");
  EXPECT_FLOAT_EQ(volume(), fv::dson::kStrandVolume);
  historyStep(rig, "history.redo");
  EXPECT_FLOAT_EQ(volume(), 0.8f);
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

// The Wardrobe tab's Light cloth (dforce.light): in the dforce topic, one step to undo and redo, kept
// with the scene and back with it.
TEST(FvHostWardrobe, LightClothIsPublishedUndoneAndSaved)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id    = figureId(rig.topic("scene"));
  const auto        light = [&] { return rig.topic("dforce/" + id).value("light", Json()); };
  const auto        saved = [&] { return rig.cmd("scene.state", {{"prop", id}}).at("result").value("dforceLight", Json()); };
  EXPECT_EQ(light(), Json(false));
  EXPECT_EQ(saved(), Json(false));
  ASSERT_TRUE(rig.cmd("dforce.light", {{"prop", id}, {"on", true}}).value("ok", false));
  EXPECT_EQ(light(), Json(true));
  EXPECT_EQ(saved(), Json(true));
  historyStep(rig, "history.undo");
  EXPECT_EQ(light(), Json(false));
  historyStep(rig, "history.redo");
  EXPECT_EQ(light(), Json(true));
  Reopened again(rig.cmd("scene.state", {{"prop", id}}).at("result"));
  EXPECT_EQ(again.rig.topic("dforce/" + again.id).value("light", Json()), Json(true));
}

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

// Keep full is the item's: a step of the history, kept when it is taken off and put back, in the scene file,
// and its surfaces say so to what fits memory.
TEST(FvHost, KeepFullGoesWithTheItem)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  const std::string item = itemNamed(wardrobeOf(rig, id), "Box Shirt");
  const auto        kept = [&] { return wardrobeOf(rig, id).at(item).value("keepFull", false); };
  // how many of the scene's materials are marked
  const auto marked = [&] {
    size_t            n     = 0;
    rs::RetainedScene& scene = rig.renderer.hostScene();
    for(const rs::DrawItem& d : scene.drawItems())
      if(const rs::MaterialRecord* m = scene.material(d.material); m && m->keepFull)
        n++;
    return n;
  };
  EXPECT_FALSE(kept());
  EXPECT_EQ(marked(), 0u);
  EXPECT_FALSE(rig.cmd("scene.state", {{"prop", id}}).at("result").at("worn")[0].contains("keepFull"));

  ASSERT_TRUE(rig.cmd("wardrobe.keepFull", {{"prop", id}, {"item", item}, {"on", true}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_TRUE(kept());
  EXPECT_GT(marked(), 0u);
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Keep full: Box Shirt");
  historyStep(rig, "history.undo");
  EXPECT_FALSE(kept());
  EXPECT_EQ(marked(), 0u);
  historyStep(rig, "history.redo");
  EXPECT_TRUE(kept());

  // off and back
  ASSERT_TRUE(rig.cmd("wardrobe.remove", {{"prop", id}, {"item", item}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  historyStep(rig, "history.undo");
  EXPECT_TRUE(kept());
  EXPECT_GT(marked(), 0u);

  // reopened
  Reopened back(rig.cmd("scene.state", {{"prop", id}}).at("result"));
  EXPECT_TRUE(back.problems.empty()) << back.problems.dump();
  EXPECT_TRUE(back.items().at(itemNamed(back.items(), "Box Shirt")).value("keepFull", false));
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
// Two pipelines build side by side, and a view that needs one waits for both: how many are done is what it can say
TEST(ShaderWork, CountsWhatIsBuildingSideBySide)
{
  {
    auto                  first = std::make_unique<ShaderWarmScope>();
    const ShaderWarmScope second;
    EXPECT_EQ(shaderWorkNow().done, 0);
    EXPECT_EQ(shaderWorkNow().total, 2);
    first.reset();
    EXPECT_EQ(shaderWorkNow().work, ShaderWork::eWarm);
    EXPECT_EQ(shaderWorkNow().done, 1);
    EXPECT_EQ(shaderWorkNow().total, 2);
    {
      const PipelineBuildScope building;
      const ShaderWorkScope    pipeline(ShaderWork::ePipeline);
      EXPECT_EQ(shaderWorkNow().work, ShaderWork::ePipeline);
      EXPECT_EQ(shaderWorkNow().done, 1);
      EXPECT_EQ(shaderWorkNow().total, 2);
    }
  }
  EXPECT_EQ(shaderWorkNow().total, 0);
  {
    const ShaderWarmScope next;
    EXPECT_EQ(shaderWorkNow().done, 0) << "a new round starts over";
    EXPECT_EQ(shaderWorkNow().total, 1);
    shader_work_detail::lastBuildMs() = 23000;
    EXPECT_EQ(shaderWorkNow().lastMs, 23000);
    shader_work_detail::lastBuildMs() = 0;
  }
  {
    const PipelineBuildScope building;
    const ShaderWorkScope    pipeline(ShaderWork::ePipeline);
    EXPECT_EQ(shaderWorkNow().total, 0) << "one with none behind it waits for itself";
  }
}

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

// A heel worn points the feet, but one coming back with a scene file doesn't: the file has the feet as
// they were saved, which the heel's pose would turn a second time. Feet flattened by hand after the
// heel went on stay flat, whether it turned the bone (Foot Heels) or set the figure's dial (Foot Boots).
TEST(FvHost, AHeelComingBackWithASceneLeavesTheFeetAsTheyWereSaved)
{
  const std::string figure = "People/Test/Foot Figure.duf", dial = "body_ctrl_lFootHeeledShoe";
  const auto footX = [](HostRig& rig, const std::string& id) {
    return rig.topic("pose/" + id).at("bones").at("l_foot").at("axes").at("rotation/x").value("value", -1.0);
  };
  for(const char* boots : {"People/Test/Clothing/Foot Heels.duf", "People/Test/Clothing/Foot Boots.duf"})
  {
    SCOPED_TRACE(boots);
    HostRig rig;
    ASSERT_TRUE(rig.cmd("load", {{"rel", figure}}).value("ok", false));
    ASSERT_TRUE(rig.settle());
    const std::string id = figureId(rig.topic("scene"));
    ASSERT_TRUE(rig.cmd("wear", {{"prop", id}, {"rel", boots}}).at("result").contains("job"));
    ASSERT_TRUE(rig.settle());
    const Json dialled = rig.topic("morphs/" + id).at("channels").value(dial, Json::object()).value("value", Json());
    ASSERT_TRUE(footX(rig, id) == 30 || dialled == 0.8) << footX(rig, id) << " " << dialled;

    rig.cmd("pose.set", {{"prop", id}, {"values", {{"l_foot/rotation/x", 0.0}}}});
    rig.cmd("morph.set", {{"prop", id}, {"values", {{dial, 0.0}}}});
    rig.host->onUpdate();
    Reopened back(rig.cmd("scene.state", {{"prop", id}}).at("result"), figure);
    EXPECT_FALSE(itemNamed(back.items(), "Foot Heel").empty()) << back.items().dump();
    EXPECT_EQ(footX(back.rig, back.id), 0);
    EXPECT_EQ(back.dial(dial), 0.0);
  }
}

// A layered image preset goes on as a makeup, and as a preset opened by itself (a tattoo): each was
// refused for being neither a material preset nor anything the viewport could put on.
TEST(FvHost, ALayeredImagePresetGoesOnAsAMakeupAndAsAPreset)
{
  const std::string rel = "People/Test/Materials/Box Tattoo.duf";
  HostRig           rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));

  const Json makeup = rig.cmd("material.makeup", {{"prop", id}, {"rel", rel}});
  ASSERT_TRUE(makeup.value("ok", false)) << makeup.dump();
  ASSERT_TRUE(rig.settle());
  const Json job = rig.topic("jobs").value(makeup.at("result").at("job").dump(), Json::object());
  EXPECT_EQ(job.value("state", ""), "done") << job.dump();
  EXPECT_EQ(job.value("makeup", ""), rel) << job.dump();
  EXPECT_GT(job.value("changed", 0), 0) << job.dump();

  const Json off = rig.cmd("material.makeup", {{"prop", id}, {"rel", Json()}});
  ASSERT_TRUE(off.value("ok", false)) << off.dump();
  ASSERT_TRUE(rig.settle());

  const Json opened = rig.cmd("load", {{"rel", rel}, {"prop", id}});
  ASSERT_TRUE(opened.value("ok", false)) << opened.dump();
  ASSERT_TRUE(rig.settle());
  const std::string notices = rig.topic("notices").dump();
  EXPECT_EQ(notices.find("isn't supported"), std::string::npos) << notices;
  EXPECT_EQ(notices.find("is not a material preset"), std::string::npos) << notices;
}

// The tattoos on a figure are a list: each can be hidden, faded and taken off, undo brings it back, a
// skin or makeup choice leaves them on, and a scene file holds them as they were.
TEST(FvHost, TattoosAreAListFromPuttingOnToReopening)
{
  const std::string one = "People/Test/Materials/Box Tattoo.duf", two = "People/Test/Materials/Box Tattoo Two.duf";
  const std::string full = "People/Test/Materials/Box Skin Full.duf";
  HostRig           rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id   = figureId(rig.topic("scene"));
  const auto        list = [&] {
    rig.host->onUpdate();
    return rig.topic("materials/" + id).at("skin").value("tattoos", Json::array());
  };
  const auto put = [&](const std::string& rel) {
    const Json answer = rig.cmd("material.tattoo", {{"prop", id}, {"rel", rel}});
    ASSERT_TRUE(answer.value("ok", false)) << answer.dump();
    ASSERT_TRUE(rig.settle());
    const Json job = rig.topic("jobs").value(answer.at("result").at("job").dump(), Json::object());
    EXPECT_EQ(job.value("state", ""), "done") << job.dump();
  };
  EXPECT_TRUE(list().empty());

  put(one);
  put(two);
  ASSERT_EQ(list().size(), 2u) << list().dump();
  EXPECT_EQ(list()[0].value("rel", ""), one);
  EXPECT_EQ(list()[1].value("rel", ""), two);
  EXPECT_TRUE(list()[0].value("on", false));
  EXPECT_TRUE(list()[0].value("layered", false));
  EXPECT_FLOAT_EQ(list()[0].value("opacity", 0.f), 1.f);

  // the same one again moves it to the top, no second copy
  put(one);
  ASSERT_EQ(list().size(), 2u);
  EXPECT_EQ(list()[1].value("rel", ""), one);

  // hidden and faded, each by itself
  ASSERT_TRUE(rig.cmd("material.tattoo.set", {{"prop", id}, {"rel", two}, {"on", false}}).value("ok", false));
  EXPECT_FALSE(list()[0].value("on", true));
  EXPECT_TRUE(list()[1].value("on", false));
  ASSERT_TRUE(rig.cmd("material.tattoo.set", {{"prop", id}, {"rel", one}, {"opacity", 0.25}}).value("ok", false));
  EXPECT_FLOAT_EQ(list()[1].value("opacity", 0.f), 0.25f);
  EXPECT_FLOAT_EQ(list()[0].value("opacity", 0.f), 1.f);
  EXPECT_FALSE(rig.cmd("material.tattoo.set", {{"prop", id}, {"rel", "People/Nowhere.duf"}, {"on", false}}).value("ok", true));

  // a scene file holds them as they are
  const Json saved = rig.cmd("scene.state", {{"prop", id}}).at("result");
  ASSERT_EQ(saved.value("tattoos", Json::array()).size(), 2u) << saved.dump();
  EXPECT_FALSE(saved.at("materialPresets").dump().find("Box Tattoo") != std::string::npos) << saved.dump();
  {
    Reopened   back(saved);
    const Json again = back.rig.cmd("scene.state", {{"prop", back.id}}).at("result").value("tattoos", Json::array());
    EXPECT_EQ(again, saved.at("tattoos")) << again.dump();
    EXPECT_TRUE(back.problems.empty()) << back.problems.dump();
  }

  // a file from before the list has a tattoo among its presets: it joins the list
  {
    Json older = saved;
    older.erase("tattoos");
    older["materialPresets"] = Json::array({one});
    Reopened   back(older);
    const Json joined = back.rig.cmd("scene.state", {{"prop", back.id}}).at("result");
    ASSERT_EQ(joined.value("tattoos", Json::array()).size(), 1u) << joined.dump();
    EXPECT_EQ(joined.at("tattoos")[0].value("rel", ""), one);
    EXPECT_EQ(joined.at("materialPresets").size(), 0u) << joined.dump();
  }

  // another skin keeps them on
  const Json skin = rig.cmd("material.skin", {{"prop", id}, {"rel", full}});
  ASSERT_TRUE(skin.value("ok", false)) << skin.dump();
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(list().size(), 2u);
  const Json own = rig.cmd("material.makeup", {{"prop", id}, {"rel", Json()}});
  ASSERT_TRUE(own.value("ok", false)) << own.dump();
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(list().size(), 2u);

  // off one at a time, undo brings it back, as it was
  ASSERT_TRUE(rig.cmd("material.tattoo.remove", {{"prop", id}, {"rel", one}}).value("ok", false));
  ASSERT_EQ(list().size(), 1u);
  EXPECT_EQ(list()[0].value("rel", ""), two);
  rig.cmd("history.undo");
  ASSERT_TRUE(rig.settle());
  ASSERT_EQ(list().size(), 2u) << list().dump();
  EXPECT_FLOAT_EQ(list()[1].value("opacity", 0.f), 0.25f);

  // and all together
  ASSERT_TRUE(rig.cmd("material.tattoo.remove", {{"prop", id}}).value("ok", false));
  EXPECT_TRUE(list().empty());
  EXPECT_FALSE(rig.cmd("scene.state", {{"prop", id}}).at("result").contains("tattoos"));
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

// A dial its file defaults on (the Shape Figure's Power, at 1) stands there from the load, and a file
// holds it only away from there, 0 included. A file from before, which has nothing for it, means where
// it now stands. Reset morphs goes back to it, and undoes back to the 0.
TEST(FvHost, ADialItsFileDefaultsOnHoldsThatFromLoadToReopening)
{
  const std::string rel = "People/Test/Shape Figure.duf";
  HostRig           rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", rel}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id    = figureId(rig.topic("scene"));
  const auto        state = [&] { return rig.cmd("scene.state", {{"prop", id}}).at("result"); };
  const auto        power = [&] { return rig.topic("morphs/" + id).at("channels").at("Power").value("value", -1.0); };
  // what a morph's file says is read on the pool: a load doesn't wait for it, so a dial takes it a frame later
  const auto arrives = [](HostRig& r, const std::function<bool()>& reached) {
    for(int i = 0; i < 5000 && !reached(); i++)
    {
      r.host->onUpdate();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return reached();
  };
  EXPECT_TRUE(arrives(rig, [&] { return power() == 1.0; }));
  EXPECT_FALSE(state().at("morphs").contains("Power")) << state().dump();
  {
    Reopened reopened(state(), rel);
    EXPECT_TRUE(arrives(reopened.rig, [&] { return reopened.dial("Power") == 1.0; }));
  }

  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Power", 0.0}}}});
  rig.host->onUpdate();
  ASSERT_TRUE(state().at("morphs").contains("Power")) << state().dump();
  EXPECT_EQ(state().at("morphs").value("Power", -1.0), 0.0);
  {
    // it stays through the default's arrival, however late that is
    Reopened reopened(state(), rel);
    EXPECT_TRUE(arrives(reopened.rig, [&] { return reopened.rig.topic("morphs/" + reopened.id).at("channels").at("Power").contains("path"); }));
    EXPECT_EQ(reopened.dial("Power"), 0.0);
  }

  rig.cmd("morph.reset", {{"prop", id}});
  rig.host->onUpdate();
  EXPECT_EQ(power(), 1.0);
  EXPECT_FALSE(state().at("morphs").contains("Power")) << state().dump();
  rig.cmd("history.undo");
  rig.host->onUpdate();
  EXPECT_EQ(power(), 0.0);
}

namespace {

// Frames until a channel of the morphs topic says what it can do: nothing waits for that, a settle included
Json reachOf(HostRig& rig, const std::string& id, const std::string& dial)
{
  const auto channel = [&] { return rig.topic("morphs/" + id).at("channels").value(dial, Json::object()); };
  for(int i = 0; i < 5000 && !channel().contains("reach"); i++)
  {
    rig.host->onUpdate();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return channel().value("reach", Json::object());
}

}  // namespace

// The panel is told what a dial can do: its deltas' size at the end of its range, the bones it turns and
// the morphs its formulas reach. A dial at a default of 0 doesn't say so.
TEST(FvHost, AMorphsTopicSaysWhatADialCanDo)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));

  const Json taper = reachOf(rig, id, "Taper");
  EXPECT_NEAR(taper.value("cm", 0.0), 8.485, 1e-9) << taper.dump();
  EXPECT_EQ(taper.value("deg", -1.0), 0.0);
  EXPECT_EQ(taper.value("morphs", 0), 1);
  EXPECT_EQ(taper.value("bones", -1), 0);
  EXPECT_EQ(taper.value("scales", -1), 0);
  const Json nod = reachOf(rig, id, "CTRL Nod");
  EXPECT_EQ(nod.value("deg", 0.0), 30.0) << nod.dump();
  EXPECT_EQ(nod.value("bones", 0), 1);
  EXPECT_EQ(nod.value("morphs", 0), 1);
  EXPECT_NEAR(nod.value("cm", 0.0), 8.485, 1e-9);

  const Json channels = rig.topic("morphs/" + id).at("channels");
  EXPECT_FALSE(channels.at("Taper").contains("rest"));
  EXPECT_FALSE(channels.at("Taper").contains("inert"));
  EXPECT_FALSE(channels.at("Taper").contains("items"));
  // the deltas of this one are all on the item, and its file has a block of none
  const Json lift = reachOf(rig, id, "Sleeve Lift");
  EXPECT_EQ(lift.value("cm", -1.0), 0.0) << lift.dump();
  EXPECT_EQ(lift.value("morphs", -1), 0);
  EXPECT_FALSE(channels.at("Sleeve Lift").contains("items"));

  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  const auto sleeve = [&] { return rig.topic("morphs/" + id).at("channels").at("Sleeve Lift"); };
  EXPECT_EQ(sleeve().value("items", Json::array()), Json::array({"Box Shirt"})) << sleeve().dump();
  EXPECT_FALSE(sleeve().contains("inert"));

  // a file read says what it holds, and the channel stays what it can do
  rig.cmd("morph.set", {{"prop", id}, {"values", {{"Sleeve Lift", 0.5}}}});
  ASSERT_TRUE(rig.settle());
  for(int i = 0; i < 5000 && !sleeve().contains("inert"); i++)
  {
    rig.host->onUpdate();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_TRUE(sleeve().value("inert", false)) << sleeve().dump();
  EXPECT_TRUE(sleeve().contains("reach"));
  EXPECT_EQ(sleeve().value("items", Json::array()), Json::array({"Box Shirt"}));
}

// A dial's default goes out when it isn't 0. A body slider has no file to say anything of it: unknown, and
// the panel counts that as there to be seen.
TEST(FvHost, AMorphsTopicSaysWhereADialRestsAndABodySliderHasNoReach)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Shape Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));

  const Json power = reachOf(rig, id, "Power");
  EXPECT_EQ(power.value("cm", -1.0), 0.0) << power.dump();
  const Json channels = rig.topic("morphs/" + id).at("channels");
  EXPECT_EQ(channels.at("Power").value("rest", 0.0), 1.0) << channels.at("Power").dump();
  const Json waist = channels.at("fv_shape_waist");
  EXPECT_FALSE(waist.contains("reach")) << waist.dump();
  EXPECT_FALSE(waist.contains("rest"));
  EXPECT_FALSE(waist.contains("inert"));
  EXPECT_FALSE(waist.contains("helper"));
  EXPECT_EQ(waist.value("kind", ""), "Modifier/FvShape");
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
  EXPECT_TRUE(peek().value("unreadable", false));
  EXPECT_FALSE(torn.cmd("scene.recover", {{"path", path.generic_string()}}).value("ok", true));
  EXPECT_TRUE(torn.topic("scene").value("props", Json::object()).empty());
  fs::remove(path);
}

TEST(FvHost, AGlbGoesWhereItWasAskedOutsideTheSystemsCodePage)
{
  const fs::path dir = fs::temp_directory_path() / L"fv glb \x30e9\x30a4\x30d6";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path out = dir / L"\x0160eima \x30e9.glb";
  HostRig rig;
  rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}});
  ASSERT_TRUE(rig.settle());
  const std::u8string where = out.generic_u8string();
  const Json answer = rig.cmd("export.glb", {{"prop", figureId(rig.topic("scene"))}, {"path", std::string(where.begin(), where.end())}});
  ASSERT_TRUE(answer.value("ok", false)) << answer.dump();
  ASSERT_TRUE(rig.settle());
  std::vector<fs::path> files;
  for(const auto& e : fs::directory_iterator(dir))
    files.push_back(e.path());
  ASSERT_EQ(files.size(), 1u);
  EXPECT_EQ(files[0].filename(), out.filename());
  EXPECT_GT(fs::file_size(out), 0u);
  fs::remove_all(dir);
}

TEST(FvHost, ACopyThatDidNotAllComeBackGoesAsideBeforeItIsWrittenOver)
{
  const fs::path path = fs::temp_directory_path() / "fv recovery aside.json";
  const fs::path keep = fs::temp_directory_path() / "fv recovery aside.prev.json";
  std::ofstream(path) << "the only copy";
  fs::remove(keep);
  HostRig rig;
  ASSERT_TRUE(rig.cmd("recovery.file", {{"path", path.generic_string()}, {"keepAs", keep.generic_string()}}).value("ok", false));
  std::string kept;
  std::getline(std::ifstream(keep), kept);
  EXPECT_EQ(kept, "the only copy");
  EXPECT_TRUE(rig.cmd("recovery.forget", {{"path", keep.generic_string()}}).value("ok", false));
  EXPECT_FALSE(fs::exists(keep));
  ASSERT_TRUE(rig.cmd("recovery.file", {{"path", nullptr}}).value("ok", false));
  EXPECT_FALSE(fs::exists(path));
}

TEST(FvHost, AnInstallIsSeenInTheLibraryThroughContentChanged)
{
  const fs::path dir = fs::temp_directory_path() / "fv content changed";
  fs::remove_all(dir);
  fs::create_directories(dir / "Morphs");
  std::ofstream(dir / "Morphs" / "one.dsf").put('x');
  const fv::dson::Library library({dir});
  fv::dson::clearListings();
  EXPECT_EQ(fv::dson::listFiles(library, "Morphs", ".dsf").size(), 1u);
  std::ofstream(dir / "Morphs" / "two.dsf").put('x');
  EXPECT_EQ(fv::dson::listFiles(library, "Morphs", ".dsf").size(), 1u);  // kept a while, as a load lists a lot
  HostRig rig;
  // a settings change that isn't one keeps what was read
  ASSERT_TRUE(rig.cmd("settings.apply", {{"maxTextureSize", 1024}}).value("ok", false));
  EXPECT_EQ(fv::dson::listFiles(library, "Morphs", ".dsf").size(), 1u);
  ASSERT_TRUE(rig.cmd("settings.apply", {{"contentChanged", true}}).value("ok", false));
  EXPECT_EQ(fv::dson::listFiles(library, "Morphs", ".dsf").size(), 2u);
  fv::dson::clearListings();
  fs::remove_all(dir);
}

namespace {

// The worn shirt's surface
std::string shirtSurface(HostRig& rig, const std::string& id)
{
  const Json items = rig.topic("materials/" + id).value("items", Json::object());
  for(const auto& [m, it] : items.items())
    if(it.value("name", "") == "Shirt")
      return m;
  return "";
}

}  // namespace

// A shader preset goes on the surface picked, whatever its name, and nowhere else. It replaces the
// surface and what the panel had on it; an option goes over it; each is a step; a scene file brings
// them back; a reset, or a colour for the item, takes them off.
TEST(FvHost, AShaderPresetGoesOnTheSurfacePicked)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("wear", {{"prop", id}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(rig.settle());
  const Json        worn = rig.topic("wardrobe/" + id).value("items", Json::object());
  ASSERT_EQ(worn.size(), 1u);
  const std::string item  = worn.begin().key();
  const std::string shirt = shirtSurface(rig, id);
  ASSERT_FALSE(shirt.empty());
  const auto surface = [&](const std::string& m) { return rig.topic("materials/" + id).at("items").at(m); };
  const auto choose  = [&](const Json& rel, const Json& on) {
    const Json answer = rig.cmd("material.shader", {{"prop", id}, {"material", on}, {"rel", rel}});
    EXPECT_TRUE(answer.value("ok", false)) << answer.dump();
    EXPECT_TRUE(rig.settle());
  };
  const char* gold = "Shader Presets/Test/Box Gold.duf";
  const char* tile = "Shader Presets/Test/Box Tile 2.duf";
  const Json  own  = surface(shirt);
  EXPECT_TRUE(own.value("shaderPreset", Json()).is_null());
  // tiling and a fabric stay on a new material, the rest of the panel's edits go
  rig.cmd("material.set", {{"prop", id}, {"material", shirt}, {"values", {{"roughness", 0.9}, {"uvScaleU", 2}}}});

  choose(gold, shirt);
  const Json on = surface(shirt);
  EXPECT_EQ(on.at("shaderPreset").at("rel"), gold);
  EXPECT_FLOAT_EQ(on.at("values").at("metalness").get<float>(), 1.f);
  EXPECT_FLOAT_EQ(on.at("values").at("roughness").get<float>(), 0.2f);
  EXPECT_EQ(on.at("edited"), Json::parse(R"(["uvScaleU","shaderPreset"])"));
  // the figure's own surfaces stay as they were
  for(const auto& [m, it] : rig.topic("materials/" + id).at("items").items())
    if(m != shirt)
      EXPECT_TRUE(it.value("shaderPreset", Json()).is_null()) << m;

  // an option goes over it, the edits it left alone stay
  rig.cmd("material.set", {{"prop", id}, {"material", shirt}, {"values", {{"roughness", 0.3}}}});
  choose(tile, shirt);
  EXPECT_EQ(surface(shirt).at("shaderPreset").at("rel"), gold);
  EXPECT_EQ(surface(shirt).at("shaderPreset").at("options"), Json::array({tile}));
  EXPECT_FLOAT_EQ(surface(shirt).at("values").at("roughness").get<float>(), 0.3f);

  // a scene file brings them back, under the edits
  const Json saved = rig.cmd("scene.state", {{"prop", id}}).at("result");
  ASSERT_TRUE(saved.contains("shaders")) << saved.dump();
  const Json shown = surface(shirt);
  choose(nullptr, shirt);
  EXPECT_TRUE(surface(shirt).value("shaderPreset", Json()).is_null());
  EXPECT_FALSE(rig.cmd("scene.state", {{"prop", id}}).at("result").contains("shaders"));
  Json restore    = {{"prop", id}, {"shaders", saved.at("shaders")}, {"materials", saved.at("materials")}};
  const Json back = rig.cmd("scene.restore", restore);
  ASSERT_TRUE(back.value("ok", false)) << back.dump();
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(differ(shown, surface(shirt)), "");

  // Reset garment takes it off, one step, and undo brings it back
  const size_t steps = rig.topic("history").at("undo").size();
  rig.cmd("material.reset", {{"prop", id}, {"material", Json::array({shirt})}});
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(rig.topic("history").at("undo").size(), steps + 1);
  EXPECT_EQ(differ(own, surface(shirt)), "");
  EXPECT_FALSE(historyStep(rig, "history.undo").contains("problems"));
  EXPECT_EQ(differ(shown, surface(shirt)), "");

  // a colour for the item replaces it
  ASSERT_TRUE(rig.cmd("wardrobe.color", {{"prop", id}, {"item", item}, {"rel", "People/Test/Materials/Box Shirt Blue.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_TRUE(surface(shirt).value("shaderPreset", Json()).is_null());

  // what it can't draw is said, and changes nothing
  const Json before = surface(shirt);
  for(const char* rel : {"Shader Presets/Test/Box Concrete.duf", "Shader Presets/Test/Box Base.duf"})
  {
    choose(rel, shirt);
    EXPECT_EQ(differ(before, surface(shirt)), "") << rel;
  }
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
      {"material.shader", {{"prop", id}, {"material", shirtSurface(rig, id)}, {"rel", "Shader Presets/Test/Box Gold.duf"}}},
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

// The gizmo drags a point of the prop's own: turned and scaled about it, that point stays where it
// is, and the prop's own transform comes back out of where the gizmo left it.
TEST(FvHost, aGizmoTurnAboutAPivotLeavesThePivotWhereItWas)
{
  const glm::vec3 pivot{-2.2f, 0.4f, 0.96f}, t{-1.f, 0.11f, -2.24f}, s{1.5f};
  const glm::quat r  = glm::angleAxis(glm::radians(90.f), glm::vec3(0, 1, 0));
  const glm::vec3 at = pivotPosition(t, r, s, pivot);
  const auto      world = [&](const glm::vec3& tt, const glm::quat& rr, const glm::vec3& ss) {
    return glm::translate(glm::mat4(1.f), tt) * glm::mat4_cast(rr) * glm::scale(glm::mat4(1.f), ss);
  };
  EXPECT_LT(glm::length(glm::vec3(world(t, r, s) * glm::vec4(pivot, 1.f)) - at), 1e-5f);
  // a turn and a scale on the gizmo, which stays put
  const glm::quat turned = glm::angleAxis(glm::radians(35.f), glm::vec3(0, 1, 0)) * r;
  const glm::vec3 scaled{2.f};
  const glm::vec3 t2 = translationFromPivot(at, turned, scaled, pivot);
  EXPECT_LT(glm::length(glm::vec3(world(t2, turned, scaled) * glm::vec4(pivot, 1.f)) - at), 1e-5f);
  // a move carries the prop by the same
  EXPECT_LT(glm::length(translationFromPivot(at + glm::vec3(0.3f, 0, 0), r, s, pivot) - (t + glm::vec3(0.3f, 0, 0))), 1e-5f);
}

// A figure turns about its own origin, its root bone, which a pose can carry far from the prop's
// (a chair's sit preset moves Genesis 9's root by metres): the gizmo's pivot is that bone in the
// prop's own space. A bone has none.
TEST(FvHost, theGizmoTurnsAFigureAboutItsRootBone)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  const std::string id = figureId(rig.topic("scene"));
  rig.cmd("prop.set", {{"prop", id}, {"transform", {{"t", {-1.0, 0.11, -2.24}}, {"e", {0.0, 90.0, 0.0}}}}});
  ASSERT_TRUE(rig.cmd("select", {{"prop", id}}).value("ok", false));
  const rs::EntityId root = rig.renderer.selectedEntity();
  ASSERT_TRUE(rig.renderer.gizmo.pivot);
  const std::optional<glm::vec3> home = rig.renderer.gizmo.pivot(root);
  ASSERT_TRUE(home);
  EXPECT_LT(glm::length(*home), 1e-5f) << "the Box Figure's root is at its origin";

  const Json nodes = rig.topic("nodes/" + id).value("nodes", Json::object());
  int        figureRoot = -1;
  for(const auto& [index, n] : nodes.items())
    if(n.value("name", "") == "BoxFigure")
      figureRoot = std::stoi(index);
  ASSERT_GE(figureRoot, 0) << nodes.dump();
  ASSERT_TRUE(rig.cmd("select", {{"prop", id}, {"node", figureRoot}}).value("ok", false));
  const rs::EntityId bone = rig.renderer.selectedEntity();
  EXPECT_FALSE(rig.renderer.gizmo.pivot(bone));

  // the root carried off, as a pose's translation would
  rs::RetainedScene& scene = rig.renderer.hostScene();
  rs::Transform      moved = {.translation = {120.f, 0.f, -45.f}};
  scene.setLocal(bone, moved);
  const std::optional<glm::vec3> pivot = rig.renderer.gizmo.pivot(root);
  ASSERT_TRUE(pivot);
  EXPECT_GT(glm::length(*pivot), 0.5f) << "the pivot stayed at the prop's origin";
  const glm::vec3 boneAt = glm::vec3(scene.world(bone)[3]);
  EXPECT_LT(glm::length(glm::vec3(scene.world(root) * glm::vec4(*pivot, 1.f)) - boneAt), 1e-4f);
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
    rig.input({{{"t", "down"}, {"b", 0}, {"x", from}, {"y", 300}, {"alt", true}}});
    ASSERT_TRUE(rig.renderer.pivotObserver);
    rig.renderer.pivotObserver(std::nullopt, false);
    rig.input({{{"t", "move"}, {"x", from + 40}, {"y", 310}, {"alt", true}}});
    pauseBeyondMerge();
    rig.input({{{"t", "move"}, {"x", from + 80}, {"y", 320}, {"alt", true}},
               {{"t", "move"}, {"x", from + 120}, {"y", 330}, {"alt", true}},
               {{"t", "up"}, {"b", 0}, {"x", from + 120}, {"y", 330}, {"alt", true}}});
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

// An editor's scheme: the left button alone moves no camera; Alt with it orbits the selection's
// middle, which stays where it is on screen, without picking anything
TEST(FvHost, AnAltDragOrbitsTheSelectionsMiddle)
{
  HostRig           rig;
  const std::string light = rig.cmd("prop.add", {{"kind", "light"}, {"transform", {{"t", {1.0, 1.0, 0.0}}}}}).at("result").at("prop").get<std::string>();
  ASSERT_TRUE(rig.settle());
  placeView(rig);
  ASSERT_TRUE(rig.cmd("select", {{"prop", light}}).value("ok", false));
  const glm::dvec3 middle(1, 1, 0);
  const glm::dvec2 at = pixelOf(rig, middle);

  rig.input({{{"t", "down"}, {"b", 0}, {"x", 640}, {"y", 360}}, {{"t", "move"}, {"x", 760}, {"y", 400}}, {{"t", "up"}, {"b", 0}, {"x", 760}, {"y", 400}}});
  EXPECT_EQ(look(rig).eye, glm::dvec3(0, 1, 3)) << "a plain drag moved the camera";

  rig.input({{{"t", "down"}, {"b", 0}, {"x", 640}, {"y", 360}, {"alt", true}},
             {{"t", "move"}, {"x", 760}, {"y", 400}, {"alt", true}},
             {{"t", "up"}, {"b", 0}, {"x", 760}, {"y", 400}, {"alt", true}}});
  EXPECT_TRUE(rig.renderer.pivots.empty()) << "a selection has a middle: nothing to pick";
  const Look l = look(rig);
  EXPECT_GT(glm::length(l.eye - glm::dvec3(0, 1, 3)), 0.1);
  EXPECT_NEAR(glm::length(l.eye - middle), glm::length(glm::dvec3(0, 1, 3) - middle), 1e-6);
  const glm::dvec2 now = pixelOf(rig, middle);
  EXPECT_NEAR(now.x, at.x, 1e-3);
  EXPECT_NEAR(now.y, at.y, 1e-3);
}

// With nothing selected an Alt drag turns about what is under the pointer as it starts: it waits for
// the pick; a double-click's point takes over until the selection changes
TEST(FvHost, WithNothingSelectedADragTurnsAboutWhatIsUnderThePointer)
{
  HostRig rig;
  placeView(rig);
  ASSERT_TRUE(rig.renderer.pivotObserver);
  const auto drag = [&](double toX) {
    rig.input({{{"t", "move"}, {"x", toX}, {"y", 380}, {"alt", true}}, {{"t", "up"}, {"b", 0}, {"x", toX}, {"y", 380}, {"alt", true}}});
  };
  rig.input({{{"t", "down"}, {"b", 0}, {"x", 640}, {"y", 360}, {"alt", true}}, {{"t", "move"}, {"x", 700}, {"y", 360}, {"alt", true}}});
  ASSERT_EQ(rig.renderer.pivots.size(), 1u);
  EXPECT_NEAR(rig.renderer.pivots[0].x, 0.5f, 1e-6f);
  EXPECT_EQ(look(rig).eye, glm::dvec3(0, 1, 3)) << "the drag turned before its pivot came";
  const glm::dvec3 hit(0, 1, -1);
  rig.renderer.pivotObserver(glm::vec3(hit), false);
  drag(720);
  Look l = look(rig);
  EXPECT_GT(glm::length(l.eye - glm::dvec3(0, 1, 3)), 0.1);
  EXPECT_NEAR(glm::length(l.eye - hit), 4.0, 1e-6);
  glm::dvec2 px = pixelOf(rig, hit);
  EXPECT_NEAR(px.x, 640, 1e-3);
  EXPECT_NEAR(px.y, 360, 1e-3);

  // a double-click's point, with nothing selected: no pick, it turns about that
  placeView(rig);
  const glm::dvec3 clicked(0.5, 1, 0);
  rig.renderer.pivotObserver(glm::vec3(clicked), true);
  const double before = glm::length(look(rig).eye - clicked);
  rig.input({{{"t", "down"}, {"b", 0}, {"x", 640}, {"y", 360}, {"alt", true}}});
  drag(760);
  EXPECT_EQ(rig.renderer.pivots.size(), 1u);
  EXPECT_NEAR(glm::length(look(rig).eye - clicked), before, 1e-6);

  // something selected afterwards: its middle again
  const std::string light = rig.cmd("prop.add", {{"kind", "light"}, {"transform", {{"t", {-1.0, 1.0, 0.0}}}}}).at("result").at("prop").get<std::string>();
  ASSERT_TRUE(rig.settle());
  ASSERT_TRUE(rig.cmd("select", {{"prop", light}}).value("ok", false));
  placeView(rig);
  rig.input({{{"t", "down"}, {"b", 0}, {"x", 640}, {"y", 360}, {"alt", true}}});
  drag(760);
  EXPECT_EQ(rig.renderer.pivots.size(), 1u);
  EXPECT_NEAR(glm::length(look(rig).eye - glm::dvec3(-1, 1, 0)), glm::length(glm::dvec3(0, 1, 3) - glm::dvec3(-1, 1, 0)), 1e-6);

  // the middle button pans, with no Alt, and Alt with the right dollies (toward the point of
  // interest, nothing selected)
  ASSERT_TRUE(rig.cmd("select", {{"prop", nullptr}}).value("ok", false));
  placeView(rig);
  rig.input({{{"t", "down"}, {"b", 1}, {"x", 640}, {"y", 360}}});
  rig.renderer.pivotObserver(std::nullopt, false);  // nothing under the pointer
  rig.input({{{"t", "move"}, {"x", 700}, {"y", 360}}, {{"t", "up"}, {"b", 1}, {"x", 700}, {"y", 360}}});
  l = look(rig);
  EXPECT_NEAR(l.eye.y, 1.0, 1e-9);
  EXPECT_GT(std::abs(l.eye.x), 1e-3) << "no pan";
  placeView(rig);
  rig.input({{{"t", "down"}, {"b", 2}, {"x", 640}, {"y", 360}, {"alt", true}}});
  rig.renderer.pivotObserver(std::nullopt, false);
  rig.input({{{"t", "move"}, {"x", 640}, {"y", 300}, {"alt", true}}, {{"t", "up"}, {"b", 2}, {"x", 640}, {"y", 300}, {"alt", true}}});
  l = look(rig);
  EXPECT_NEAR(l.eye.x, 0.0, 1e-9);
  EXPECT_GT(std::abs(l.eye.z - 3.0), 1e-3) << "no dolly";
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

TEST(FvHost, WithSelectOffAClickPicksNothingAndADoubleClickStillTurnsTheView)
{
  HostRig rig;
  const auto click = [](double x) {
    return Json::array({{{"t", "down"}, {"b", 0}, {"x", x}, {"y", 180}}, {{"t", "up"}, {"b", 0}, {"x", x}, {"y", 180}}});
  };
  ASSERT_TRUE(rig.cmd("view.set", {{"select", false}}).value("ok", false));
  rig.input(click(320));
  EXPECT_TRUE(rig.renderer.picks.empty());

  // a double-click turns the view to the surface under it, which selects nothing
  rig.input({{{"t", "dblclick"}, {"x", 320}, {"y", 180}}});
  ASSERT_EQ(rig.renderer.picks.size(), 1u);
  EXPECT_TRUE(rig.renderer.picks[0].second);

  ASSERT_TRUE(rig.cmd("view.set", {{"select", true}}).value("ok", false));
  rig.input(click(900));
  ASSERT_EQ(rig.renderer.picks.size(), 2u);
  EXPECT_FALSE(rig.renderer.picks[1].second);
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
  std::atomic<int> composedCooks{0}, unflippedCooks{0};

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
      // a cook for the cache (flipped) waits to be let through. One that can only be shown cooked is
      // cooked as the load goes (fallbackTexture), as it will stay: that one doesn't
      const bool composed = req.has_param("alpha") || req.has_param("mk") || req.has_param("ir") || req.has_param("rc") || req.has_param("ly") || req.has_param("mul");
      composedCooks += composed && flipped ? 1 : 0;
      unflippedCooks += flipped ? 0 : 1;
      if(flipped && !composed && req.get_param_value("cached") != "1")
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

// A figure loaded on a cold texture cache shows the cooked maps the cache has. What can only be shown
// cooked (a bake with something laid in it) is cooked once as the load goes, flipped as it will stay,
// and the load waits for that one only.
TEST(FvHost, AFigureOnAColdCacheCooksWhatItCanOnlyShowCookedOnce)
{
  FakeCooker cooker;
  HostRig    rig;
  ASSERT_TRUE(rig.cmd("settings.apply", {{"texServer", cooker.url()}}).value("ok", false));
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure PBRSkin.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  rs::RetainedScene& scene  = rig.renderer.hostScene();
  const Sampled      landed = sampled(scene);
  ASSERT_GE(landed.images.size(), 2u);
  EXPECT_EQ(landed.turned, 0);
  EXPECT_EQ(std::count_if(landed.images.begin(), landed.images.end(), [](const std::string& uri) { return uri.ends_with("-up.ktx2"); }), std::ptrdiff_t(landed.images.size()))
      << ::testing::PrintToString(landed.images);
  EXPECT_GT(cooker.composedCooks.load(), 0);
  EXPECT_EQ(cooker.unflippedCooks.load(), 0);
}

namespace {

// The PBRSkin box with nothing on its body but a plain colour map, which a cold cache shows from its
// source and cooks after the load, in a library root of its own beside the test's. Returns the rig's
// settings for both roots and the cooker.
Json plainMapSettings(const FakeCooker& cooker)
{
  const fs::path root = fs::temp_directory_path() / "fv-host-plain-map";
  fs::create_directories(root / "People");
  Json duf = Json::parse(std::ifstream(fs::path(FV_TEST_DATA) / "daz-library/People/Test/Box Figure PBRSkin.duf"));
  duf["scene"]["materials"][0]["extra"][0]["channels"] = Json::array(
      {{{"channel", {{"id", "Diffuse Color"}, {"type", "color"}, {"current_value", {1, 1, 1}}, {"image_file", "/Runtime/Textures/Test/BoxSkin_D.png"}}}}});
  std::ofstream(root / "People/Box Painted.duf") << duf.dump();
  return {{"texServer", cooker.url()},
          {"libraryRoots", {(fs::path(FV_TEST_DATA) / "daz-library").generic_string(), root.generic_string()}}};
}

}  // namespace

// What a cold cache cooks after the load lands one map at a time: a still waits for all of them, so
// it begins once instead of starting again as each lands, and render.status says how many are left
TEST(FvHost, AStillWaitsForTheTexturesStillCooking)
{
  FakeCooker cooker;
  HostRig    rig;
  const auto cooking = [&] { return rig.cmd("render.status").value("result", Json::object()).value("texturesCooking", -1); };
  ASSERT_TRUE(rig.cmd("settings.apply", plainMapSettings(cooker)).value("ok", false));
  EXPECT_EQ(cooking(), 0);
  EXPECT_FALSE(rig.renderer.texturesArriving);

  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Box Painted.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  // the plain maps' cooks wait at the cooker
  EXPECT_GT(cooking(), 0);
  EXPECT_TRUE(rig.renderer.texturesArriving);

  cooker.open();
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while(rig.renderer.texturesArriving && std::chrono::steady_clock::now() < end)
  {
    rig.host->onUpdate();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_FALSE(rig.renderer.texturesArriving);
  EXPECT_EQ(cooking(), 0);
}

// A derived map on a cold cache (a metal mask, into glTF's channels) stands in as one white texel
// until its cook lands, and the cook takes that texel's slot: nothing is composed for it at full size.
// Every map waiting so shares the one texel's file
TEST(FvHost, ADerivedMapOnAColdCacheWaitsForItsCookAsOneTexel)
{
  FakeCooker cooker;
  HostRig    rig;
  const fs::path root = fs::temp_directory_path() / "fv-host-metal-map";
  fs::create_directories(root / "People");
  Json duf = Json::parse(std::ifstream(fs::path(FV_TEST_DATA) / "daz-library/People/Test/Box Figure PBRSkin.duf"));
  duf["scene"]["materials"][0]["extra"][0]["channels"] = Json::array(
      {{{"channel", {{"id", "Diffuse Color"}, {"type", "color"}, {"current_value", {1, 1, 1}}, {"image_file", "/Runtime/Textures/Test/BoxSkin_D.png"}}}},
       {{"channel", {{"id", "Metallic Weight"}, {"type", "float"}, {"current_value", 1}, {"image_file", "/Runtime/Textures/Test/BoxMakeup_Mask.png"}}}}});
  std::ofstream(root / "People/Box Metal.duf") << duf.dump();
  ASSERT_TRUE(rig.cmd("settings.apply", {{"texServer", cooker.url()},
                                         {"libraryRoots", {(fs::path(FV_TEST_DATA) / "daz-library").generic_string(), root.generic_string()}}})
                  .value("ok", false));
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Box Metal.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  rs::RetainedScene& scene    = rig.renderer.hostScene();
  const auto         count    = [](const Sampled& s, auto&& is) { return std::count_if(s.images.begin(), s.images.end(), is); };
  const auto         flat     = [](const std::string& uri) { return uri.find("flat-white.png") != std::string::npos; };
  const auto         composed = [](const std::string& uri) { return uri.find("fv-composed") != std::string::npos && uri.find("flat-") == std::string::npos; };
  const Sampled      waiting  = sampled(scene);
  EXPECT_EQ(count(waiting, flat), 1) << ::testing::PrintToString(waiting.images);
  EXPECT_EQ(count(waiting, composed), 0);
  EXPECT_TRUE(rig.renderer.texturesArriving);
  // the colour map's source and the texel are stand-ins: decoded where they are made, the source at
  // half its size, not ahead
  EXPECT_EQ(rig.renderer.prefetches.load(), 0);

  cooker.open();
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while(rig.renderer.texturesArriving && std::chrono::steady_clock::now() < end)
  {
    rig.host->onUpdate();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ASSERT_FALSE(rig.renderer.texturesArriving);
  const Sampled landed = sampled(scene);
  EXPECT_EQ(count(landed, flat), 0);
  EXPECT_EQ(count(landed, [](const std::string& uri) { return uri.ends_with("-up.ktx2"); }), std::ptrdiff_t(landed.images.size()));
  EXPECT_EQ(landed.turned, 0);
}

// Cooks for a figure that has left the scene keep no still waiting
TEST(FvHost, CooksOfAFigureThatLeftTheSceneKeepNoStillWaiting)
{
  FakeCooker cooker;
  HostRig    rig;
  ASSERT_TRUE(rig.cmd("settings.apply", plainMapSettings(cooker)).value("ok", false));
  ASSERT_TRUE(rig.cmd("load", {{"rel", "People/Box Painted.duf"}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  ASSERT_TRUE(rig.renderer.texturesArriving);
  ASSERT_TRUE(rig.cmd("scene.clear").value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_FALSE(rig.renderer.texturesArriving);
  EXPECT_EQ(rig.cmd("render.status").value("result", Json::object()).value("texturesCooking", -1), 0);
  // the cooks still held let go, or the rig's detach waits out the cooker
  cooker.open();
}

// The renderer path traces only: render.mode sets the skin, and there is no raster view to ask for
TEST(FvHost, TheRendererPathTracesOnly)
{
  HostRig rig;
  EXPECT_EQ(rig.topic("render").value("mode", ""), "rtx");
  const Json answer = rig.cmd("render.mode", {{"skin", "thin"}});
  ASSERT_TRUE(answer.value("ok", false)) << answer.dump();
  EXPECT_EQ(answer["result"].value("mode", ""), "rtx");
  EXPECT_TRUE(rig.cmd("render.mode", {{"mode", "rtx"}, {"skin", "thin"}}).value("ok", false));
  EXPECT_FALSE(rig.cmd("render.mode", {{"mode", "raster"}}).value("ok", false));
  // solid glass (Iray Uber's Thin Walled off refracting as a solid) goes with the skin; unsent, it stays
  EXPECT_TRUE(rig.cmd("render.mode", {{"skin", "thin"}, {"solidGlass", true}}).value("ok", false));
  EXPECT_TRUE(rig.cmd("render.mode", {{"skin", "measured"}}).value("ok", false));
  EXPECT_FALSE(rig.cmd("render.mode", {{"skin", "measured"}, {"solidGlass", "yes"}}).value("ok", false));
}

// The Display menu's exposure view: the tone mapper's false colour over the shaded image, off in every
// other mode, and a name the host doesn't know changes nothing
TEST(FvHost, TheExposureViewIsTheToneMappersFalseColour)
{
  HostRig rig;
  EXPECT_FALSE(rig.renderer.falseColor);
  EXPECT_TRUE(rig.cmd("view.set", {{"display", "exposure"}}).value("ok", false));
  EXPECT_TRUE(rig.renderer.falseColor);
  EXPECT_FALSE(rig.cmd("view.set", {{"display", "heat"}}).value("ok", false));
  EXPECT_TRUE(rig.renderer.falseColor);
  EXPECT_TRUE(rig.cmd("view.set", {{"display", "clay"}}).value("ok", false));
  EXPECT_FALSE(rig.renderer.falseColor);
}

// The Display menu's Time view: the shaded image's paths, each pixel timed; the renderer's sums are
// published as `time`, the dearest first, until another view clears it. Without the device's clock
// the view is refused.
TEST(FvHost, TheTimeViewPublishesWhereTheFramesTimeWent)
{
  HostRig rig;
  EXPECT_TRUE(rig.cmd("view.set", {{"display", "time"}}).value("ok", false));
  EXPECT_TRUE(rig.renderer.pixelTime);
  EXPECT_FALSE(rig.renderer.falseColor);
  ASSERT_TRUE(rig.renderer.pixelTimeObserver);

  using Kind = HostRenderer::PixelTimes::Kind;
  HostRenderer::PixelTimes times;
  times.parts = {{.kind = Kind::eSky, .ticks = 100, .samples = 100},
                 {.kind = Kind::eGround, .ticks = 300, .samples = 100},
                 {.kind = Kind::eMesh, .ticks = 600, .samples = 50}};  // no pick to name it here
  times.ticks   = 1000;
  times.samples = 250;
  rig.renderer.pixelTimeObserver(times);
  Json j = rig.topic("time");
  ASSERT_EQ(j["parts"].size(), 3u) << j.dump();
  EXPECT_EQ(j["parts"][0].value("what", ""), "mesh");
  EXPECT_NEAR(j["parts"][0].value("share", 0.0), 0.6, 1e-9);
  EXPECT_NEAR(j["parts"][0].value("cost", 0.0), 3.0, 1e-9);  // 12 a sample against 4
  EXPECT_FALSE(j["parts"][0].contains("prop"));
  EXPECT_EQ(j["parts"][1].value("what", ""), "ground");
  EXPECT_EQ(j["parts"][2].value("what", ""), "sky");
  EXPECT_NEAR(j["parts"][2].value("cost", 0.0), 0.25, 1e-9);
  EXPECT_EQ(j.value("samples", 0), 250);
  EXPECT_NEAR(j.value("perSample", 0.0), 4.0, 1e-9);  // the frame's average, in the clock's ticks
  EXPECT_EQ(j.value("counting", ""), "ticks");

  // a count in place of the time (the debug pixelTimeCount): the same shares, the average in rays
  times.counting = 1;  // shaderio::kPixelTimeCountRays
  rig.renderer.pixelTimeObserver(times);
  j = rig.topic("time");
  EXPECT_EQ(j.value("counting", ""), "rays");
  EXPECT_NEAR(j["parts"][0].value("share", 0.0), 0.6, 1e-9);

  // another view: the renderer stops timing, the topic goes, and a late reading isn't published
  EXPECT_TRUE(rig.cmd("view.set", {{"display", "shaded"}}).value("ok", false));
  EXPECT_FALSE(rig.renderer.pixelTime);
  EXPECT_TRUE(rig.topic("time").is_null());
  rig.renderer.pixelTimeObserver(times);
  EXPECT_TRUE(rig.topic("time").is_null());

  // no clock on the device: refused, and the view stays as it was
  rig.renderer.clockAvailable = false;
  EXPECT_FALSE(rig.cmd("view.set", {{"display", "time"}}).value("ok", false));
  EXPECT_FALSE(rig.renderer.pixelTime);
}

// The wipe against a kept frame: keep the image as it is, move the divider, end it; a divider that
// is no number is refused
TEST(FvHost, AWipeKeepsAFrameAndMovesItsDivider)
{
  HostRig rig;
  EXPECT_TRUE(rig.cmd("view.wipe", {{"keep", true}, {"at", 0.5}}).value("ok", false));
  EXPECT_EQ(rig.renderer.wipeKept, 1);
  ASSERT_TRUE(rig.renderer.wipeAt);
  EXPECT_FLOAT_EQ(*rig.renderer.wipeAt, 0.5f);
  EXPECT_TRUE(rig.cmd("view.wipe", {{"at", 0.25}}).value("ok", false));
  EXPECT_EQ(rig.renderer.wipeKept, 1);  // moving it keeps nothing new
  EXPECT_FLOAT_EQ(*rig.renderer.wipeAt, 0.25f);
  EXPECT_TRUE(rig.cmd("view.wipe", {{"at", nullptr}}).value("ok", false));
  EXPECT_FALSE(rig.renderer.wipeAt);
  EXPECT_FALSE(rig.cmd("view.wipe", {{"at", "left"}}).value("ok", false));
}

// The pixel readout: view.inspect asks for a pixel in the view's own pixels (as input events give
// them), each reading the renderer makes is published as `inspect` with the material it found and the
// image's radiance there, and x null stops it and clears the topic. A pixel outside the view, or no
// number, is refused.
TEST(FvHost, ThePixelReadoutPublishesWhatIsAtThePixel)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("view.inspect", {{"x", 640}, {"y", 180}}).value("ok", false));
  ASSERT_EQ(rig.renderer.inspects.size(), 1u);
  ASSERT_TRUE(rig.renderer.inspects[0]);
  EXPECT_FLOAT_EQ(rig.renderer.inspects[0]->x, 0.5f);  // of the fake's 1280 x 720
  EXPECT_FLOAT_EQ(rig.renderer.inspects[0]->y, 0.25f);
  ASSERT_TRUE(rig.renderer.inspectObserver);

  // the ground, read at 64 samples
  HostRenderer::PixelReadout ground;
  ground.data.position  = glm::vec4(0.5f, 0.0f, -2.0f, 3.25f);
  ground.data.normal    = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
  ground.data.baseColor = glm::vec4(0.5f, 0.4f, 0.3f, 1.0f);
  ground.data.surface   = glm::vec4(0.0f, 0.6f, 1.5f, 1.0f);
  ground.data.radiance  = glm::vec4(0.02f, 0.015f, 0.01f, 64.0f);
  ground.data.ids       = glm::ivec4(-2, -1, -1, 0);
  ground.exposed        = 0.36f;
  rig.renderer.inspectObserver(ground);
  Json j = rig.topic("inspect");
  EXPECT_EQ(j.value("hit", ""), "ground") << j.dump();
  EXPECT_DOUBLE_EQ(j.value("x", 0.0), 640.0);
  EXPECT_NEAR(j.value("distance", 0.0), 3.25, 1e-6);
  EXPECT_NEAR(j["baseColor"][1].get<double>(), 0.4, 1e-6);
  EXPECT_NEAR(j.value("roughness", 0.0), 0.6, 1e-6);
  EXPECT_NEAR(j.value("ior", 0.0), 1.5, 1e-6);
  EXPECT_NEAR(j["radiance"][0].get<double>(), 0.02, 1e-6);
  EXPECT_EQ(j.value("samples", 0), 64);
  EXPECT_NEAR(j.value("exposed", 0.0), 0.36, 1e-6);
  EXPECT_FALSE(j.value("moving", true));
  EXPECT_TRUE(j.value("front", false));
  EXPECT_FALSE(j.contains("uv")) << "the ground has no texture coordinates";
  EXPECT_FALSE(j.contains("subsurface"));
  EXPECT_FALSE(j.contains("prop"));

  // a mesh on a moving frame, a scattering medium in it; no pick to name it here
  HostRenderer::PixelReadout skin = ground;
  skin.data.ids    = glm::ivec4(7, 120, 3, 1);
  skin.data.medium = glm::vec4(0.9f, 0.6f, 0.5f, 0.0f);
  skin.data.uv     = glm::vec4(0.25f, 0.75f, 640.0f, 180.0f);
  rig.renderer.inspectObserver(skin);
  j = rig.topic("inspect");
  EXPECT_EQ(j.value("hit", ""), "mesh");
  EXPECT_TRUE(j.value("moving", false));
  EXPECT_NEAR(j["uv"][1].get<double>(), 0.75, 1e-6);
  EXPECT_NEAR(j["subsurface"][0].get<double>(), 0.9, 1e-6);

  // the sky: nothing hit, the radiance still read
  HostRenderer::PixelReadout sky;
  sky.data.radiance = glm::vec4(1.5f, 2.0f, 3.0f, 64.0f);
  rig.renderer.inspectObserver(sky);
  j = rig.topic("inspect");
  EXPECT_TRUE(j.contains("hit") && j["hit"].is_null()) << j.dump();
  EXPECT_NEAR(j["radiance"][2].get<double>(), 3.0, 1e-6);

  // in the pixels of the frame the app asked for, as its input events are
  ASSERT_TRUE(rig.cmd("view.inspect", {{"x", 160}, {"y", 270}, {"w", 640}, {"h", 360}}).value("ok", false));
  ASSERT_EQ(rig.renderer.inspects.size(), 2u);
  EXPECT_FLOAT_EQ(rig.renderer.inspects[1]->x, 0.25f);
  EXPECT_FLOAT_EQ(rig.renderer.inspects[1]->y, 0.75f);

  // refused: outside the view, no number
  EXPECT_FALSE(rig.cmd("view.inspect", {{"x", 2000}, {"y", 10}}).value("ok", false));
  EXPECT_FALSE(rig.cmd("view.inspect", {{"x", "left"}, {"y", 10}}).value("ok", false));
  EXPECT_EQ(rig.renderer.inspects.size(), 2u);

  // stopped: the renderer stops reading, the topic goes, and a late reading is not published
  ASSERT_TRUE(rig.cmd("view.inspect", {{"x", nullptr}}).value("ok", false));
  ASSERT_EQ(rig.renderer.inspects.size(), 3u);
  EXPECT_FALSE(rig.renderer.inspects[2]);
  EXPECT_TRUE(rig.topic("inspect").is_null());
  rig.renderer.inspectObserver(ground);
  EXPECT_TRUE(rig.topic("inspect").is_null());
}

// The frame as a linear image, once it is on disk: the path tracer's, a .hdr
TEST(FvHost, ACaptureIsAnHdrOfThePathTracersImage)
{
  HostRig                     rig;
  const fs::path file = fs::temp_directory_path() / "fv_capture_test.hdr";

  EXPECT_FALSE(rig.cmd("render.capture", {{"path", "still.png"}}).value("ok", false));
  EXPECT_TRUE(rig.renderer.savedImage.empty());

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
    if(p.value("kind", "") == "character")
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

//--------------------------------------------------------------------------------------------------
// Body contact (figure/physics/contact/contact.hpp, docs/physics-plan.md step 1)

namespace {

std::string base64(const std::vector<uint8_t>& bytes)
{
  static const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string        out;
  for(size_t i = 0; i < bytes.size(); i += 3)
  {
    const uint32_t n = uint32_t(bytes[i]) << 16 | (i + 1 < bytes.size() ? uint32_t(bytes[i + 1]) << 8 : 0) | (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    out += abc[(n >> 18) & 63];
    out += abc[(n >> 12) & 63];
    out += i + 1 < bytes.size() ? abc[(n >> 6) & 63] : '=';
    out += i + 2 < bytes.size() ? abc[n & 63] : '=';
  }
  return out;
}

// A glTF of one closed box, in metres.
void writeBox(const fs::path& path, std::array<float, 3> lo, std::array<float, 3> hi)
{
  std::vector<float> p;
  for(int k = 0; k < 2; k++)
    for(int j = 0; j < 2; j++)
      for(int i = 0; i < 2; i++)
        p.insert(p.end(), {i ? hi[0] : lo[0], j ? hi[1] : lo[1], k ? hi[2] : lo[2]});
  const std::vector<uint32_t> t = {0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5};
  std::vector<uint8_t>        bytes(p.size() * 4 + t.size() * 4);
  std::memcpy(bytes.data(), p.data(), p.size() * 4);
  std::memcpy(bytes.data() + p.size() * 4, t.data(), t.size() * 4);
  const Json gltf = {
      {"asset", {{"version", "2.0"}}},
      {"scene", 0},
      {"scenes", {{{"nodes", {0}}}}},
      {"nodes", {{{"name", "box"}, {"mesh", 0}}}},
      {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}}}, {"indices", 1}}}}}}},
      {"buffers", {{{"byteLength", bytes.size()}, {"uri", "data:application/octet-stream;base64," + base64(bytes)}}}},
      {"bufferViews", {{{"buffer", 0}, {"byteOffset", 0}, {"byteLength", p.size() * 4}}, {{"buffer", 0}, {"byteOffset", p.size() * 4}, {"byteLength", t.size() * 4}}}},
      {"accessors",
       {{{"bufferView", 0}, {"componentType", 5126}, {"count", 8}, {"type", "VEC3"}, {"min", {lo[0], lo[1], lo[2]}}, {"max", {hi[0], hi[1], hi[2]}}},
        {{"bufferView", 1}, {"componentType", 5125}, {"count", t.size()}, {"type", "SCALAR"}}}}};
  std::ofstream(path) << gltf.dump();
}

// The Box Figure (20 cm wide, 40 tall, standing on the ground) and a box prop under its feet that
// reaches 2 cm into it, placed far off to start.
struct Seated
{
  HostRig     rig;
  fs::path    dir = fs::temp_directory_path() / "fv_contact_test";
  std::string figure, box;
  int         moves = 0;

  Seated()
  {
    fs::create_directories(dir);
    writeBox(dir / "box.gltf", {-0.15f, -0.1f, -0.15f}, {0.15f, 0.02f, 0.15f});
    EXPECT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    figure = figureId(rig.topic("scene"));
    EXPECT_TRUE(rig.cmd("load", {{"path", (dir / "box.gltf").generic_string()}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    const Json props = rig.topic("scene").value("props", Json::object());
    for(const auto& [id, p] : props.items())
      if(p.value("kind", "") == "model")
        box = id;
    place(5);
  }
  ~Seated() { fs::remove_all(dir); }

  void place(double x)
  {
    // each move a step of its own, as a drag is
    const Json placed = rig.cmd("prop.set", {{"prop", box}, {"transform", {{"t", {x, 0.0, 0.0}}}}, {"gesture", {{"id", ++moves}, {"end", true}}}});
    EXPECT_TRUE(placed.value("ok", false)) << placed.dump();
  }
  Json contact() const { return rig.topic("contact/" + figure); }
  // frames until `done` says so, or a few seconds
  bool until(const std::function<bool()>& done)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while(std::chrono::steady_clock::now() < end)
    {
      rig.host->onUpdate();
      if(done())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
  }
  bool rests()
  {
    return until([&] { return !contact().value("busy", true); });
  }
  int patches() const { return contact().value("patches", -1); }
};

}  // namespace

// A box moved under the figure's feet dents them; undone, the dent goes. A box moved about far from it
// starts no pass.
TEST(FvHost, aPropMovedIntoAFigurePokesItsContact)
{
  Seated s;
  ASSERT_TRUE(s.rests());
  EXPECT_EQ(s.patches(), 0) << s.contact().dump();
  EXPECT_TRUE(s.contact().value("enabled", false));
  s.place(0);
  ASSERT_TRUE(s.until([&] { return s.patches() > 0; })) << s.contact().dump();
  ASSERT_TRUE(s.rig.cmd("history.undo").value("ok", false));
  ASSERT_TRUE(s.until([&] { return s.patches() == 0; })) << s.contact().dump();
  ASSERT_TRUE(s.rests());
  s.place(10);
  for(int i = 0; i < 20; i++)
  {
    s.rig.host->onUpdate();
    EXPECT_FALSE(s.contact().value("busy", true)) << "a box moved from 5 m to 10 m away poked it";
  }
}

// A dent pokes the clothes' collider, and none of the passes pokes the others back: all come to rest.
TEST(FvHost, aDentSettlesWithoutPassesPokingEachOther)
{
  Seated s;
  s.rig.cmd("wear", {{"prop", s.figure}, {"rel", "People/Test/Clothing/Box Shirt.duf"}});
  ASSERT_TRUE(s.rig.settle());
  s.place(0);
  ASSERT_TRUE(s.until([&] { return s.patches() > 0; }));
  const auto collisionBusy = [&] { return s.rig.topic("wardrobe/" + s.figure).value("collision", Json::object()).value("busy", true); };
  ASSERT_TRUE(s.until([&] { return !s.contact().value("busy", true) && !collisionBusy(); }));
  for(int i = 0; i < 100; i++)
  {
    s.rig.host->onUpdate();
    ASSERT_FALSE(s.contact().value("busy", true)) << "frame " << i;
    ASSERT_FALSE(collisionBusy()) << "frame " << i;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

// The settings are a step of the history and part of the figure's scene state; a scene file from
// before body contact opens without it, and one from before self-contact without that.
TEST(FvHost, contactSettingsSurviveTheSceneFileAndUndo)
{
  Seated s;
  ASSERT_TRUE(s.rig.cmd("figure.contact", {{"prop", s.figure}, {"bulge", 0.3}}).value("ok", false));
  EXPECT_FLOAT_EQ(s.contact().value("bulge", 0.f), 0.3f);
  const Json state = s.rig.cmd("scene.state", {{"prop", s.figure}}).value("result", Json::object());
  ASSERT_TRUE(state.contains("contact")) << state.dump();
  EXPECT_FLOAT_EQ(state["contact"].value("bulge", 0.f), 0.3f);
  EXPECT_TRUE(state["contact"].value("on", false));
  ASSERT_TRUE(s.rig.cmd("history.undo").value("ok", false));
  EXPECT_FLOAT_EQ(s.contact().value("bulge", 0.f), 0.7f);
  ASSERT_TRUE(s.rig.cmd("history.redo").value("ok", false));
  EXPECT_FLOAT_EQ(s.contact().value("bulge", 0.f), 0.3f);

  // self-contact: on for a new figure, a step of its own, kept with the rest
  EXPECT_TRUE(s.contact().value("self", false)) << s.contact().dump();
  EXPECT_TRUE(state["contact"].value("self", false));
  ASSERT_TRUE(s.rig.cmd("figure.contact", {{"prop", s.figure}, {"self", false}}).value("ok", false));
  EXPECT_FALSE(s.contact().value("self", true));
  EXPECT_FALSE(s.rig.cmd("scene.state", {{"prop", s.figure}}).value("result", Json::object())["contact"].value("self", true));
  ASSERT_TRUE(s.rig.cmd("history.undo").value("ok", false));
  EXPECT_TRUE(s.contact().value("self", false));
  EXPECT_FLOAT_EQ(s.contact().value("bulge", 0.f), 0.3f) << "the undo took the bulge's step too";

  // restored with the block it was saved with, which from before self-contact has it off; then from a
  // file that has none
  Json restore       = state;
  restore["prop"]    = s.figure;
  restore["contact"] = {{"on", true}, {"softness", 0.5}, {"bulge", 0.9}, {"reach", 4}};
  s.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(s.rig.settle());
  EXPECT_FLOAT_EQ(s.contact().value("bulge", 0.f), 0.9f);
  EXPECT_FLOAT_EQ(s.contact().value("reach", 0.f), 4.f);
  EXPECT_FALSE(s.contact().value("self", true));
  restore["contact"]["self"] = true;
  s.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(s.rig.settle());
  EXPECT_TRUE(s.contact().value("self", false));
  restore.erase("contact");
  s.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(s.rig.settle());
  EXPECT_FALSE(s.contact().value("enabled", true)) << s.contact().dump();
}

// Rain on the skin (figure/physics/rain/rain.hpp): off for a new person; turned on, a pass lands its
// drops; the settings are held to their ranges, are steps of the history and part of the scene state;
// off dries the skin, and a file from before rain opens dry.
TEST(FvHost, rainLandsAndIsKeptWithTheSceneAndUndo)
{
  Seated     s;
  const auto rain = [&] { return s.rig.topic("rain/" + s.figure); };
  ASSERT_TRUE(rain().is_object() && !rain().empty()) << "a person has rain";
  EXPECT_FALSE(rain().value("enabled", true));
  ASSERT_TRUE(s.rig.cmd("figure.rain", {{"prop", s.figure}, {"on", true}, {"amount", 1.0}, {"time", 20}}).value("ok", false));
  EXPECT_TRUE(rain().value("enabled", false));
  ASSERT_TRUE(s.until([&] { return !rain().value("busy", true) && rain().value("landed", 0) > 0; })) << rain().dump();
  EXPECT_GT(rain().value("beads", 0), 0);

  const Json state = s.rig.cmd("scene.state", {{"prop", s.figure}}).value("result", Json::object());
  ASSERT_TRUE(state.contains("rain")) << state.dump();
  EXPECT_TRUE(state["rain"].value("on", false));
  EXPECT_FLOAT_EQ(state["rain"].value("amount", 0.f), 1.f);
  EXPECT_FLOAT_EQ(state["rain"].value("time", 0.f), 20.f);
  ASSERT_TRUE(s.rig.cmd("figure.rain", {{"prop", s.figure}, {"dried", 400}}).value("ok", false));
  EXPECT_FLOAT_EQ(rain().value("dried", 0.f), 300.f) << "held to its range";
  ASSERT_TRUE(s.rig.cmd("history.undo").value("ok", false));
  EXPECT_FLOAT_EQ(rain().value("dried", -1.f), 0.f);
  ASSERT_TRUE(s.rig.cmd("history.undo").value("ok", false));
  EXPECT_FALSE(rain().value("enabled", true));
  EXPECT_TRUE(s.until([&] { return rain().value("beads", 1) == 0; })) << "off dries the skin";

  Json restore    = state;
  restore["prop"] = s.figure;
  s.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(s.rig.settle());
  EXPECT_TRUE(rain().value("enabled", false));
  restore.erase("rain");
  s.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(s.rig.settle());
  EXPECT_FALSE(rain().value("enabled", true)) << "a file from before rain";
}

// A DAZ prop (a chair from the library: here the Box Figure's mesh opened from Props, so it has no
// dents of its own) is met as a model is: placed under the figure's feet it dents them, and posed away
// from under them it lets them go.
TEST(FvHost, aDazPropDentsAsItIsPosed)
{
  Seated s;
  ASSERT_TRUE(s.rests());
  ASSERT_TRUE(s.rig.cmd("prop.add", {{"rel", "Props/Test/Box Stool.duf"}}).value("ok", false));
  ASSERT_TRUE(s.rig.settle());
  std::string stool;
  const Json  props = s.rig.topic("scene").value("props", Json::object());
  for(const auto& [id, p] : props.items())
    if(p.value("kind", "") == "prop" && id != s.figure)
      stool = id;
  ASSERT_FALSE(stool.empty());
  EXPECT_TRUE(s.rig.topic("contact/" + stool).empty()) << "a prop got dents of its own";

  // twice the Box Figure's size, 40 cm wide and 80 tall, its top 1 cm above the figure's soles
  const Json placed = s.rig.cmd("prop.set", {{"prop", stool}, {"transform", {{"t", {0.0, -0.79, 0.0}}, {"s", {2.0, 2.0, 2.0}}}}, {"gesture", {{"id", 100}, {"end", true}}}});
  ASSERT_TRUE(placed.value("ok", false)) << placed.dump();
  ASSERT_TRUE(s.until([&] { return s.patches() > 0; })) << s.contact().dump();
  ASSERT_TRUE(s.rests());

  // its upper half (the head bone, from 20 cm up) turned 60 degrees takes its highest corner from
  // 40 cm to 30 + 10 sin 60 = 38.7, so 1.7 cm under the soles: its placement is the same, its shape not
  const Json posed = s.rig.cmd("pose.set", {{"prop", stool}, {"values", {{"head/rotation/x", 60.0}}}});
  ASSERT_TRUE(posed.value("ok", false)) << posed.dump();
  ASSERT_TRUE(s.until([&] { return s.patches() == 0; })) << s.contact().dump();
}

//--------------------------------------------------------------------------------------------------
// What a DAZ object is (protocol 45): a character, a creature, a vehicle, a set or a prop, by its file
// and its library folder; the app's reading of the library sharpens a prop; the user may treat it as
// another, and a body's machinery comes and goes with it.

namespace {

// a DAZ object added, and its id once it is in the scene
std::string addDaz(HostRig& rig, const Json& args)
{
  const Json before = rig.topic("scene").value("props", Json::object());
  const Json added  = rig.cmd("prop.add", args);
  EXPECT_TRUE(added.value("ok", false)) << added.dump();
  EXPECT_TRUE(rig.settle());
  const Json after = rig.topic("scene").value("props", Json::object());
  for(const auto& [id, p] : after.items())
    if(!before.contains(id))
      return id;
  return "";
}

Json propOf(HostRig& rig, const std::string& id)
{
  return rig.topic("scene").value("props", Json::object()).value(id, Json::object());
}

}  // namespace

TEST(FvHost, aDazObjectsKindComesFromItsFileAndFolder)
{
  HostRig           rig;
  const std::string person = addDaz(rig, {{"rel", "People/Test/Box Figure.duf"}});
  const std::string stool  = addDaz(rig, {{"rel", "Props/Test/Box Stool.duf"}});
  const std::string cart   = addDaz(rig, {{"rel", "Vehicles/Test/Box Cart.duf"}});
  EXPECT_EQ(propOf(rig, person).value("kind", ""), "character");
  EXPECT_TRUE(propOf(rig, person).value("body", false));
  EXPECT_TRUE(propOf(rig, person).contains("morphs"));
  EXPECT_EQ(propOf(rig, stool).value("kind", ""), "prop");
  EXPECT_FALSE(propOf(rig, stool).value("body", true));
  EXPECT_TRUE(propOf(rig, stool).value("posable", false)) << "a rigged prop still poses";
  EXPECT_EQ(propOf(rig, cart).value("kind", ""), "vehicle");
  // a body collides with its clothes; a prop builds none of that
  const Json personState = rig.cmd("scene.state", {{"prop", person}}).value("result", Json::object());
  EXPECT_TRUE(personState.contains("collision")) << personState.dump();
  EXPECT_EQ(personState.value("kind", ""), "character");
  const Json stoolState = rig.cmd("scene.state", {{"prop", stool}}).value("result", Json::object());
  EXPECT_FALSE(stoolState.contains("collision")) << stoolState.dump();
  EXPECT_FALSE(stoolState.contains("dforce"));
  EXPECT_EQ(stoolState.value("kind", ""), "prop");
  EXPECT_TRUE(rig.topic("dforce/" + stool).empty());
}

TEST(FvHost, theLibrarysKindOnlySharpensAProp)
{
  HostRig           rig;
  const std::string stool  = addDaz(rig, {{"rel", "Props/Test/Box Stool.duf"}, {"kind", "set"}});
  const std::string person = addDaz(rig, {{"rel", "People/Test/Box Figure.duf"}, {"kind", "prop"}});
  EXPECT_EQ(propOf(rig, stool).value("kind", ""), "set");
  EXPECT_EQ(propOf(rig, person).value("kind", ""), "character");
}

TEST(FvHost, aPropTreatedAsACreatureGainsABodyAndKeepsIt)
{
  HostRig           rig;
  const std::string stool = addDaz(rig, {{"rel", "Props/Test/Box Stool.duf"}});
  const Json        set   = rig.cmd("prop.set", {{"prop", stool}, {"kind", "creature"}});
  ASSERT_TRUE(set.value("ok", false)) << set.dump();
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(propOf(rig, stool).value("kind", ""), "creature");
  EXPECT_TRUE(propOf(rig, stool).value("body", false));
  Json state = rig.cmd("scene.state", {{"prop", stool}}).value("result", Json::object());
  EXPECT_EQ(state.value("kind", ""), "creature");
  EXPECT_TRUE(state.contains("collision")) << "a creature collides with its clothes: " << state.dump();
  EXPECT_FALSE(state.contains("contact")) << "only a character dents";
  const Json undo = rig.topic("history").value("undo", Json::array());
  ASSERT_FALSE(undo.empty());
  EXPECT_EQ(undo[0].get<std::string>(), "Treat Box Stool as a creature");

  // a copy is what the original is now
  const Json dup = rig.cmd("prop.duplicate", {{"prop", stool}});
  ASSERT_TRUE(dup.value("ok", false)) << dup.dump();
  ASSERT_TRUE(rig.settle());
  int        creatures = 0;
  const Json props     = rig.topic("scene").value("props", Json::object());
  for(const auto& [id, p] : props.items())
    creatures += p.value("kind", "") == "creature";
  EXPECT_EQ(creatures, 2);

  // undone, the copy goes, then the stool is a prop again without a body
  rig.cmd("history.undo");
  ASSERT_TRUE(rig.settle());
  rig.cmd("history.undo");
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(propOf(rig, stool).value("kind", ""), "prop");
  state = rig.cmd("scene.state", {{"prop", stool}}).value("result", Json::object());
  EXPECT_FALSE(state.contains("collision")) << state.dump();

  // a model, a camera or a group is no DAZ object to treat as one
  const Json camera = rig.cmd("prop.add", {{"kind", "camera"}});
  const Json refused = rig.cmd("prop.set", {{"prop", camera.value("result", Json::object()).value("prop", "")}, {"kind", "prop"}});
  EXPECT_FALSE(refused.value("ok", true)) << refused.dump();
}

TEST(FvHost, clothAndPhysicsNeedABody)
{
  HostRig           rig;
  const std::string stool = addDaz(rig, {{"rel", "Props/Test/Box Stool.duf"}});
  const Json        named = rig.cmd("dforce.enable", {{"prop", stool}, {"on", true}});
  EXPECT_FALSE(named.value("ok", true));
  EXPECT_NE(named.value("error", "").find("is a prop"), std::string::npos) << named.dump();
  // with no prop named, the only DAZ object there is no body to drape
  const Json unnamed = rig.cmd("dforce.enable", {{"on", true}});
  EXPECT_FALSE(unnamed.value("ok", true)) << unnamed.dump();
  const Json physics = rig.cmd("physics.set", {{"prop", stool}, {"on", true}});
  EXPECT_NE(physics.value("error", "").find("is a prop"), std::string::npos) << physics.dump();
}

// Clothing opened with a chair as the panels' figure goes on the one body there, not on the chair
TEST(FvHost, clothingNamedForAPropGoesOnTheBody)
{
  HostRig           rig;
  const std::string person = addDaz(rig, {{"rel", "People/Test/Box Figure.duf"}});
  const std::string stool  = addDaz(rig, {{"rel", "Props/Test/Box Stool.duf"}});
  const Json        props  = rig.topic("scene").value("props", Json::object());
  const Json        loaded = rig.cmd("load", {{"rel", "People/Test/Clothing/Box Shirt.duf"}, {"prop", stool}});
  ASSERT_TRUE(loaded.value("ok", false)) << loaded.dump();
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(rig.topic("scene").value("props", Json::object()).size(), props.size()) << "the shirt became a prop";
  EXPECT_FALSE(rig.topic("wardrobe/" + person).value("items", Json::object()).empty());
}

TEST(FvHost, aGroupIsAnEmptyCarrier)
{
  HostRig    rig;
  const Json added = rig.cmd("prop.add", {{"kind", "group"}, {"name", "Lights"}});
  ASSERT_TRUE(added.value("ok", false)) << added.dump();
  ASSERT_TRUE(rig.settle());
  const std::string group = added.value("result", Json::object()).value("prop", "");
  EXPECT_EQ(propOf(rig, group).value("kind", ""), "group");
  EXPECT_EQ(propOf(rig, group).value("name", ""), "Lights");
  const Json light = rig.cmd("prop.add", {{"kind", "light"}});
  ASSERT_TRUE(rig.cmd("prop.set", {{"prop", light.value("result", Json::object()).value("prop", "")}, {"parent", group}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  EXPECT_EQ(propOf(rig, light.value("result", Json::object()).value("prop", "")).value("parent", ""), group);
}

//--------------------------------------------------------------------------------------------------
// Placing (fv_host_place.cpp): Drop to the ground or onto what is below, Aim, Isolate.

namespace {

float heightOf(HostRig& rig, const std::string& id)
{
  return rig.topic("scene")["props"][id]["transform"]["t"][1].get<float>();
}

}  // namespace

// Dropped with nothing under it, the figure stands on the ground; over a box, on the box's top (2 cm
// up), whichever way it was off; undone, it is back where it was.
TEST(FvHost, aDroppedFigureRestsOnWhatIsUnderIt)
{
  Seated s;
  ASSERT_TRUE(s.rig.cmd("lighting.set", {{"ground", {{"height", 0.0}}}}).value("ok", false));
  // the box far off: the figure comes down onto the ground
  ASSERT_TRUE(s.rig.cmd("prop.set", {{"prop", s.figure}, {"transform", {{"t", {0.0, 0.5, 0.0}}}}}).value("ok", false));
  Json dropped = s.rig.cmd("prop.drop", {{"prop", s.figure}});
  ASSERT_TRUE(dropped.value("ok", false)) << dropped.dump();
  ASSERT_TRUE(s.rig.settle());
  const float onGround = heightOf(s.rig, s.figure);
  EXPECT_LT(onGround, 0.5f);
  // the box under it, and the figure sunk 10 cm into it: it comes up onto the box's top
  s.place(0);
  ASSERT_TRUE(s.rig.cmd("prop.set", {{"prop", s.figure}, {"transform", {{"t", {0.0, onGround - 0.1, 0.0}}}}}).value("ok", false));
  dropped = s.rig.cmd("prop.drop", {{"prop", s.figure}});
  ASSERT_TRUE(dropped.value("ok", false)) << dropped.dump();
  ASSERT_TRUE(s.rig.settle());
  EXPECT_NEAR(heightOf(s.rig, s.figure), onGround + 0.02f, 0.01f);
  const Json undo = s.rig.topic("history").value("undo", Json::array());
  ASSERT_FALSE(undo.empty());
  EXPECT_EQ(undo[0].get<std::string>(), "Drop Box Figure");
  s.rig.cmd("history.undo");
  EXPECT_NEAR(heightOf(s.rig, s.figure), onGround - 0.1f, 1e-4f);
}

// An aimed light's -Z points at the box's middle, from wherever it stands
TEST(FvHost, anAimedLightLooksAtTheProp)
{
  Seated     s;
  s.place(0);
  const Json added = s.rig.cmd("prop.add", {{"kind", "light"}, {"type", "spot"}, {"transform", {{"t", {1.0, 2.0, 1.0}}}}});
  ASSERT_TRUE(s.rig.settle());
  const std::string light = added.value("result", Json::object()).value("prop", "");
  const Json        aimed = s.rig.cmd("prop.aim", {{"prop", light}, {"at", s.box}});
  ASSERT_TRUE(aimed.value("ok", false)) << aimed.dump();
  const Json r = s.rig.topic("scene")["props"][light]["transform"]["r"];
  const glm::quat q(r[3].get<float>(), r[0].get<float>(), r[1].get<float>(), r[2].get<float>());
  const glm::vec3 looks = q * glm::vec3(0, 0, -1);
  const glm::vec3 want  = glm::normalize(glm::vec3(0, -0.04f, 0) - glm::vec3(1, 2, 1));
  EXPECT_NEAR(glm::dot(looks, want), 1.f, 1e-4f);
  const Json undo = s.rig.topic("history").value("undo", Json::array());
  ASSERT_FALSE(undo.empty());
  EXPECT_EQ(undo[0].get<std::string>(), "Aim Spot light at box");
}

// Isolate shows the box alone: the figure's meshes go from the scene, its own flag and the history
// untouched; a prop added meanwhile stays out; null shows all again
TEST(FvHost, isolateShowsOnlyWhatItNames)
{
  Seated     s;
  const auto shownItems = [&] {
    int n = 0;
    for(const rs::DrawItem& item : s.rig.renderer.hostScene().drawItems())
      n += item.shown;
    return n;
  };
  const int all     = shownItems();
  const auto before = s.rig.topic("history").value("undo", Json::array()).size();
  ASSERT_TRUE(s.rig.cmd("view.set", {{"solo", s.box}}).value("ok", false));
  ASSERT_TRUE(s.rig.settle());
  const int alone = shownItems();
  EXPECT_LT(alone, all);
  EXPECT_GT(alone, 0);
  EXPECT_TRUE(s.rig.topic("scene")["props"][s.figure].value("visible", false));
  EXPECT_EQ(s.rig.topic("scene").value("solo", Json::array()), Json::array({s.box}));
  EXPECT_EQ(s.rig.topic("history").value("undo", Json::array()).size(), before);
  ASSERT_TRUE(s.rig.cmd("prop.add", {{"rel", "Props/Test/Box Stool.duf"}}).value("ok", false));
  ASSERT_TRUE(s.rig.settle());
  EXPECT_EQ(shownItems(), alone) << "a prop added while isolated showed";
  ASSERT_TRUE(s.rig.cmd("view.set", {{"solo", nullptr}}).value("ok", false));
  ASSERT_TRUE(s.rig.settle());
  EXPECT_GT(shownItems(), all);
}

//--------------------------------------------------------------------------------------------------
// A node inside a prop (node.set): hidden or moved, kept in scene.state, one step of the history.

namespace {

int shownItems(HostRig& rig)
{
  int n = 0;
  for(const rs::DrawItem& item : rig.renderer.hostScene().drawItems())
    n += item.shown;
  return n;
}

// the first node of the prop's nodes topic of that kind
int nodeOfKind(HostRig& rig, const std::string& prop, const std::string& kind)
{
  const Json nodes = rig.topic("nodes/" + prop).value("nodes", Json::object());
  int        best = -1, order = 1 << 30;
  for(const auto& [id, n] : nodes.items())
    if(n.value("kind", "") == kind && n.value("order", 0) < order)
    {
      best  = std::atoi(id.c_str());
      order = n.value("order", 0);
    }
  return best;
}

}  // namespace

TEST(FvHost, aModelsNodeHidesAndMovesAndComesBack)
{
  Seated     s;
  const int  all  = shownItems(s.rig);
  const int  node = nodeOfKind(s.rig, s.box, "mesh");
  ASSERT_GE(node, 0) << s.rig.topic("nodes/" + s.box).dump();
  Json set = s.rig.cmd("node.set", {{"prop", s.box}, {"node", node}, {"visible", false}});
  ASSERT_TRUE(set.value("ok", false)) << set.dump();
  EXPECT_EQ(shownItems(s.rig), all - 1);
  EXPECT_TRUE(s.rig.topic("nodes/" + s.box)["nodes"][std::to_string(node)].value("hidden", false));
  Json state = s.rig.cmd("scene.state", {{"prop", s.box}}).value("result", Json::object());
  EXPECT_EQ(state["nodes"][std::to_string(node)].value("visible", true), false) << state.dump();

  // moved: the selection shows where it is, in metres
  set = s.rig.cmd("node.set", {{"prop", s.box}, {"node", node}, {"transform", {{"t", {0.0, 1.0, 0.0}}}}});
  ASSERT_TRUE(set.value("ok", false)) << set.dump();
  ASSERT_TRUE(s.rig.cmd("select", {{"prop", s.box}, {"node", node}}).value("ok", false));
  const Json sel = s.rig.topic("selection");
  EXPECT_NEAR(sel["state"]["t"][1].get<double>(), 1.0, 1e-5) << sel.dump();
  EXPECT_TRUE(sel["state"].value("moves", false));

  const Json undo = s.rig.topic("history").value("undo", Json::array());
  ASSERT_GE(undo.size(), 2u);
  EXPECT_EQ(undo[0].get<std::string>().rfind("Move ", 0), 0u) << undo.dump();
  EXPECT_EQ(undo[1].get<std::string>().rfind("Hide ", 0), 0u) << undo.dump();
  s.rig.cmd("history.undo");
  s.rig.cmd("history.undo");
  EXPECT_EQ(shownItems(s.rig), all);
  state = s.rig.cmd("scene.state", {{"prop", s.box}}).value("result", Json::object());
  EXPECT_FALSE(state.contains("nodes")) << state.dump();
}

// A DAZ prop's node stays hidden as the prop is posed (the feed writes the rest of its nodes every
// step); its root and what is posed are refused
TEST(FvHost, aDazPropsHiddenNodeStaysHiddenAsItPoses)
{
  HostRig           rig;
  const std::string stool = addDaz(rig, {{"rel", "Props/Test/Box Stool.duf"}});
  const int         all   = shownItems(rig);
  int               mesh  = nodeOfKind(rig, stool, "skinned");
  if(mesh < 0)
    mesh = nodeOfKind(rig, stool, "mesh");
  ASSERT_GE(mesh, 0) << rig.topic("nodes/" + stool).dump();
  ASSERT_TRUE(rig.cmd("node.set", {{"prop", stool}, {"node", mesh}, {"visible", false}}).value("ok", false));
  EXPECT_LT(shownItems(rig), all);
  ASSERT_TRUE(rig.cmd("pose.set", {{"prop", stool}, {"values", {{"head/rotation/x", 30.0}}}}).value("ok", false));
  for(int i = 0; i < 4; i++)
    rig.host->onUpdate();
  EXPECT_LT(shownItems(rig), all) << "a pose step showed it again";
  const Json state = rig.cmd("scene.state", {{"prop", stool}}).value("result", Json::object());
  EXPECT_EQ(state["nodes"][std::to_string(mesh)].value("visible", true), false) << state.dump();
  // a skinned mesh or a posed bone isn't moved here
  const Json moved = rig.cmd("node.set", {{"prop", stool}, {"node", mesh}, {"transform", {{"t", {0.0, 1.0, 0.0}}}}});
  EXPECT_FALSE(moved.value("ok", true)) << moved.dump();
  // nor is the figure's root, which is the prop's own Transform
  int        root  = -1;
  const Json nodes = rig.topic("nodes/" + stool).value("nodes", Json::object());
  for(const auto& [id, n] : nodes.items())
    if(n.value("depth", 1) == 0)
      root = std::atoi(id.c_str());
  ASSERT_GE(root, 0);
  EXPECT_FALSE(rig.cmd("node.set", {{"prop", stool}, {"node", root}, {"visible", false}}).value("ok", true));
  // shown again with null
  ASSERT_TRUE(rig.cmd("node.set", {{"prop", stool}, {"node", mesh}, {"visible", nullptr}}).value("ok", false));
  EXPECT_EQ(shownItems(rig), all);
}

//--------------------------------------------------------------------------------------------------
// Several selected: select's props, add and toggle, a Ctrl+Shift+click; the gizmo moves them together as one
// step; a group holds them, ungrouped they keep their place.

namespace {

Json selectedOf(HostRig& rig)
{
  return rig.topic("selection").value("props", Json::array());
}

// two lights and a camera, at known places
struct Three
{
  HostRig     rig;
  std::string a, b, c;
  Three()
  {
    const auto add = [&](const char* kind, std::array<double, 3> t) {
      const Json added = rig.cmd("prop.add", {{"kind", kind}, {"transform", {{"t", {t[0], t[1], t[2]}}}}});
      EXPECT_TRUE(added.value("ok", false)) << added.dump();
      return added.value("result", Json::object()).value("prop", std::string());
    };
    a = add("light", {0, 1, 0});
    b = add("light", {1, 1, 0});
    c = add("camera", {0, 1, 3});
    EXPECT_TRUE(rig.settle());
  }
  glm::vec3 at(const std::string& id)
  {
    const Json t = rig.topic("scene")["props"][id]["transform"]["t"];
    return {t[0].get<float>(), t[1].get<float>(), t[2].get<float>()};
  }
};

}  // namespace

TEST(FvHost, severalAreSelectedTheFirstThePrimary)
{
  Three t;
  ASSERT_TRUE(t.rig.cmd("select", {{"props", {t.a, t.b}}}).value("ok", false));
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.a, t.b}));
  EXPECT_EQ(t.rig.topic("selection").value("prop", ""), t.a);
  // one more, then one out: the primary moves to what is left
  ASSERT_TRUE(t.rig.cmd("select", {{"prop", t.c}, {"toggle", true}}).value("ok", false));
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.c, t.a, t.b}));
  ASSERT_TRUE(t.rig.cmd("select", {{"prop", t.c}, {"toggle", true}}).value("ok", false));
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.a, t.b}));
  // a prop removed leaves the selection
  ASSERT_TRUE(t.rig.cmd("prop.remove", {{"prop", t.b}}).value("ok", false));
  ASSERT_TRUE(t.rig.settle());
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.a}));
  // a plain select is one again
  ASSERT_TRUE(t.rig.cmd("select", {{"prop", t.c}}).value("ok", false));
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.c}));
}

// A Ctrl+click selects with Click to select off, where a plain click picks nothing; not while the
// app keeps the clicks for itself (its pixel readout)
TEST(FvHost, ACtrlClickSelectsWhateverSelectSays)
{
  HostRig rig;
  const auto click = [](double x, bool ctrl) {
    return Json::array({{{"t", "down"}, {"b", 0}, {"x", x}, {"y", 180}, {"ctrl", ctrl}}, {{"t", "up"}, {"b", 0}, {"x", x}, {"y", 180}, {"ctrl", ctrl}}});
  };
  ASSERT_TRUE(rig.cmd("view.set", {{"select", false}}).value("ok", false));
  rig.input(click(320, false));
  EXPECT_TRUE(rig.renderer.picks.empty());
  rig.input(click(320, true));
  ASSERT_EQ(rig.renderer.picks.size(), 1u);
  EXPECT_FALSE(rig.renderer.picks[0].second) << "a click's pick, not a double-click's turn of the view";
  // a Ctrl+drag pans and picks nothing
  rig.input(Json::array({{{"t", "down"}, {"b", 0}, {"x", 600}, {"y", 180}, {"ctrl", true}},
                         {{"t", "move"}, {"x", 640}, {"y", 180}, {"ctrl", true}},
                         {{"t", "up"}, {"b", 0}, {"x", 640}, {"y", 180}, {"ctrl", true}}}));
  EXPECT_EQ(rig.renderer.picks.size(), 1u);
  ASSERT_TRUE(rig.cmd("view.set", {{"selectCtrl", false}}).value("ok", false));
  rig.input(click(900, true));
  EXPECT_EQ(rig.renderer.picks.size(), 1u);
}

// A plain click on a prop selects it alone; a Ctrl+click puts it in the selection or takes it out,
// and on nothing changes nothing; an Alt+click (an orbit that didn't move) picks nothing
TEST(FvHost, ACtrlClickTogglesAPropInTheSelection)
{
  Three t;
  ASSERT_TRUE(t.rig.cmd("view.set", {{"helpers", false}}).value("ok", false));
  ASSERT_TRUE(t.rig.cmd("select", {{"prop", t.a}}).value("ok", false));
  rs::EntityId lightB;
  for(const rs::EntityId root : t.rig.renderer.hostScene().roots())
    if(const rs::Entity* e = t.rig.renderer.hostScene().entity(root); e && e->name == "Point light 2")
      lightB = root;
  ASSERT_FALSE(lightB.isNone());
  ASSERT_TRUE(t.rig.renderer.pickObserver);
  const auto click = [&](double x, Json mods) {
    Json up = {{"t", "up"}, {"x", x}, {"y", 100}, {"b", 0}};
    up.update(mods);
    t.rig.input(Json::array({{{"t", "down"}, {"x", x}, {"y", 100}, {"b", 0}}, up}));
  };
  click(100, {{"ctrl", true}});
  t.rig.renderer.pickObserver(HostRenderer::InstancePick{lightB, {}}, glm::vec3(1, 1, 0));
  t.rig.host->onUpdate();
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.b, t.a})) << t.rig.topic("selection").dump();
  click(300, {{"ctrl", true}});
  t.rig.renderer.pickObserver(HostRenderer::InstancePick{lightB, {}}, glm::vec3(1, 1, 0));
  t.rig.host->onUpdate();
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.a})) << "a second Ctrl+click didn't take it out";
  const size_t picks = t.rig.renderer.picks.size();
  click(500, {{"alt", true}});
  EXPECT_EQ(t.rig.renderer.picks.size(), picks) << "an Alt+click picked";
  click(700, Json::object());
  ASSERT_EQ(t.rig.renderer.picks.size(), picks + 1);
  t.rig.renderer.pickObserver(HostRenderer::InstancePick{lightB, {}}, glm::vec3(1, 1, 0));
  t.rig.host->onUpdate();
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.b}));
  click(900, {{"ctrl", true}});
  t.rig.renderer.pickObserver(std::nullopt, glm::vec3(0));
  t.rig.host->onUpdate();
  EXPECT_EQ(selectedOf(t.rig), Json::array({t.b}));
}

// A click on a prop selects it whole; on a set, the part hit (what the hover outlines)
TEST(FvHost, aClickOnASetSelectsThePartHit)
{
  HostRig           rig;
  const std::string stool = addDaz(rig, {{"rel", "Props/Test/Box Stool.duf"}});
  int               node  = nodeOfKind(rig, stool, "skinned");
  if(node < 0)
    node = nodeOfKind(rig, stool, "mesh");
  ASSERT_GE(node, 0) << rig.topic("nodes/" + stool).dump();
  ASSERT_TRUE(rig.cmd("select", {{"prop", stool}, {"node", node}}).value("ok", false));
  const rs::EntityId part = rig.renderer.selectedEntity();
  ASSERT_FALSE(part.isNone());
  ASSERT_TRUE(rig.cmd("select", {{"prop", nullptr}}).value("ok", false));
  ASSERT_TRUE(rig.renderer.pickObserver);
  const auto click = [&](double x) {
    rig.input(Json::array({{{"t", "down"}, {"x", x}, {"y", 100}, {"b", 0}}, {{"t", "up"}, {"x", x}, {"y", 100}, {"b", 0}}}));
    rig.renderer.pickObserver(HostRenderer::InstancePick{part, {}}, glm::vec3(0));
    rig.host->onUpdate();
    return rig.topic("selection");
  };
  Json sel = click(100);
  EXPECT_EQ(sel.value("prop", ""), stool);
  EXPECT_TRUE(sel["node"].is_null()) << sel.dump();
  EXPECT_NE(rig.renderer.selectedEntity(), part);

  const Json kind = rig.cmd("prop.set", {{"prop", stool}, {"kind", "set"}});
  ASSERT_TRUE(kind.value("ok", false)) << kind.dump();
  ASSERT_TRUE(rig.settle());
  sel = click(400);
  EXPECT_EQ(sel.value("prop", ""), stool);
  EXPECT_EQ(sel.value("node", -1), node) << sel.dump();
  EXPECT_EQ(rig.renderer.selectedEntity(), part);
  EXPECT_EQ(selectedOf(rig), Json::array({stool}));
}

// The gizmo on a selected part of a prop (object picking) moves it as node.set does, the drag one
// step; a part that goes where its bones are gets no gizmo
TEST(FvHost, theGizmoMovesASelectedPart)
{
  Seated    s;
  const int node = nodeOfKind(s.rig, s.box, "mesh");
  ASSERT_GE(node, 0) << s.rig.topic("nodes/" + s.box).dump();
  ASSERT_TRUE(s.rig.cmd("select", {{"prop", s.box}, {"node", node}}).value("ok", false));
  const rs::EntityId part = s.rig.renderer.selectedEntity();
  ASSERT_FALSE(part.isNone());
  ASSERT_TRUE(s.rig.renderer.gizmo.attach);
  EXPECT_NE(s.rig.renderer.gizmo.attach(part), 0u) << "no gizmo on a part node.set can move";

  const size_t steps = undoSteps(s.rig);
  ASSERT_TRUE(s.rig.renderer.gizmo.edit);
  s.rig.renderer.gizmo.edit(part, {0.f, 1.5f, 0.f}, glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(1.f), false);
  s.rig.renderer.gizmo.edit(part, {0.f, 2.f, 0.f}, glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(1.f), true);
  s.rig.host->onUpdate();
  const Json sel = s.rig.topic("selection");
  EXPECT_NEAR(sel["state"]["t"][1].get<double>(), 2.0, 1e-5) << sel.dump();
  EXPECT_EQ(undoSteps(s.rig), steps + 1);
  s.rig.cmd("history.undo");
  s.rig.host->onUpdate();
  EXPECT_LT(s.rig.topic("selection")["state"]["t"][1].get<double>(), 1.0) << s.rig.topic("selection").dump();

  // a figure's bone, selected in object picking: the Pose tab's to move
  const int bone = nodeOfKind(s.rig, s.figure, "bone");
  if(bone >= 0)
  {
    ASSERT_TRUE(s.rig.cmd("select", {{"prop", s.figure}, {"node", bone}}).value("ok", false));
    const rs::EntityId boneEntity = s.rig.renderer.selectedEntity();
    if(s.rig.topic("selection").contains("bone"))
      EXPECT_EQ(s.rig.renderer.gizmo.attach(boneEntity), 0u);
  }
}

// The gizmo on the primary moves the others by as much, and the drag is one step
TEST(FvHost, theGizmoMovesSeveralAsOneStep)
{
  Three t;
  ASSERT_TRUE(t.rig.cmd("select", {{"props", {t.a, t.b}}}).value("ok", false));
  const rs::EntityId root  = t.rig.renderer.selectedEntity();
  const size_t       steps = undoSteps(t.rig);
  ASSERT_TRUE(t.rig.renderer.gizmo.edit);
  t.rig.renderer.gizmo.edit(root, {0.f, 1.5f, 0.f}, glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(1.f), false);
  t.rig.renderer.gizmo.edit(root, {0.f, 2.f, 0.f}, glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(1.f), true);
  t.rig.host->onUpdate();
  EXPECT_NEAR(t.at(t.a).y, 2.f, 1e-5f);
  EXPECT_NEAR(t.at(t.b).y, 2.f, 1e-5f);
  EXPECT_NEAR(t.at(t.b).x, 1.f, 1e-5f);
  EXPECT_NEAR(t.at(t.c).y, 1.f, 1e-5f) << "what isn't selected moved";
  EXPECT_EQ(undoSteps(t.rig), steps + 1);
  EXPECT_EQ(t.rig.topic("history").at("undo")[0], "Move 2 props");
  t.rig.cmd("history.undo");
  EXPECT_NEAR(t.at(t.b).y, 1.f, 1e-5f);
}

// Grouped, the lights ride on a group at their bottom middle, in place; one undo takes it back.
// Ungrouped, they ride on nothing again, still in place, and the group is gone.
TEST(FvHost, aGroupHoldsWhatItWasGivenInPlace)
{
  Three      t;
  const Json grouped = t.rig.cmd("prop.add", {{"kind", "group"}, {"name", "Lamps"}, {"props", {t.a, t.b}}});
  ASSERT_TRUE(grouped.value("ok", false)) << grouped.dump();
  ASSERT_TRUE(t.rig.settle());
  const std::string g = grouped.value("result", Json::object()).value("prop", "");
  const Json        props = t.rig.topic("scene")["props"];
  EXPECT_EQ(props[t.a].value("parent", ""), g);
  EXPECT_EQ(props[t.b].value("parent", ""), g);
  EXPECT_NEAR(props[g]["transform"]["t"][0].get<double>(), 0.5, 1e-5);
  // in place: relative to the group now
  EXPECT_NEAR(t.at(t.b).x, 0.5f, 1e-5f);
  EXPECT_EQ(t.rig.topic("history").at("undo")[0], "Group 2 props");

  const Json ungrouped = t.rig.cmd("prop.ungroup", {{"prop", g}});
  ASSERT_TRUE(ungrouped.value("ok", false)) << ungrouped.dump();
  ASSERT_TRUE(t.rig.settle());
  EXPECT_FALSE(t.rig.topic("scene")["props"].contains(g));
  EXPECT_TRUE(t.rig.topic("scene")["props"][t.b]["parent"].is_null());
  EXPECT_NEAR(t.at(t.b).x, 1.f, 1e-5f);
  // both undone: the lights ride on nothing, the group never was
  t.rig.cmd("history.undo");
  t.rig.cmd("history.undo");
  ASSERT_TRUE(t.rig.settle());
  EXPECT_FALSE(t.rig.topic("scene")["props"].contains(g));
  EXPECT_TRUE(t.rig.topic("scene")["props"][t.a]["parent"].is_null());
  EXPECT_NEAR(t.at(t.b).x, 1.f, 1e-5f);
}

// A group's copy holds copies of what it held
TEST(FvHost, aGroupsDuplicateHoldsCopies)
{
  Three      t;
  const Json grouped = t.rig.cmd("prop.add", {{"kind", "group"}, {"props", {t.a, t.b}}});
  ASSERT_TRUE(t.rig.settle());
  const std::string g   = grouped.value("result", Json::object()).value("prop", "");
  const Json        dup = t.rig.cmd("prop.duplicate", {{"prop", g}});
  ASSERT_TRUE(dup.value("ok", false)) << dup.dump();
  ASSERT_TRUE(t.rig.settle());
  const std::string copy = dup.value("result", Json::object()).value("prop", "");
  int               held  = 0;
  const Json        props = t.rig.topic("scene")["props"];
  for(const auto& [id, p] : props.items())
    held += p.value("parent", Json()).is_string() && p["parent"].get<std::string>() == copy;
  EXPECT_EQ(held, 2);
}

// A snapped drag moves by whole steps from where it started, not onto the grid
TEST(GizmoSnap, StepsFromWhereTheDragStarted)
{
  EXPECT_NEAR(gizmo::snapFrom(0.37f, 0.55f, 0.25f), 0.62f, 1e-6f);
  EXPECT_NEAR(gizmo::snapFrom(0.37f, 0.45f, 0.25f), 0.37f, 1e-6f);
  EXPECT_NEAR(gizmo::snapFrom(1.f, 1.37f, 0.f), 1.37f, 1e-6f) << "no step moves freely";
  EXPECT_NEAR(gizmo::snapped(-0.13f, 0.1f), -0.1f, 1e-6f);
}

// view.set's space and snap reach the gizmo; snap null turns it off and keeps the steps
TEST(FvHost, theGizmosSpaceAndSnapReachTheRenderer)
{
  HostRig rig;
  ASSERT_TRUE(rig.cmd("view.set", {{"space", "local"}, {"snap", {{"move", 0.5}, {"rotate", 30}}}}).value("ok", false));
  EXPECT_TRUE(rig.renderer.gizmoOptions.local);
  EXPECT_TRUE(rig.renderer.gizmoOptions.snap);
  EXPECT_FLOAT_EQ(rig.renderer.gizmoOptions.move, 0.5f);
  EXPECT_FLOAT_EQ(rig.renderer.gizmoOptions.rotate, 30.f);
  ASSERT_TRUE(rig.cmd("view.set", {{"snap", nullptr}}).value("ok", false));
  EXPECT_FALSE(rig.renderer.gizmoOptions.snap);
  EXPECT_FLOAT_EQ(rig.renderer.gizmoOptions.move, 0.5f);
  EXPECT_FALSE(rig.cmd("view.set", {{"space", "sideways"}}).value("ok", true));
  EXPECT_FALSE(rig.cmd("view.set", {{"snap", {{"move", -1}}}}).value("ok", true));
}

// The pixel a world point is at, as cameraRay casts them: for a test that places by a ray
glm::vec2 placePixel(HostRig& rig, const glm::vec3& point)
{
  auto       manip = rig.renderer.getCameraManipulator();
  glm::dvec3 eye, center, up;
  manip->getLookat(eye, center, up);
  const glm::vec3  f = glm::normalize(glm::vec3(center - eye));
  const glm::vec3  r = glm::normalize(glm::cross(f, glm::vec3(up)));
  const glm::vec3  u = glm::cross(r, f);
  const glm::vec3  d = point - glm::vec3(eye);
  const float      t = std::tan(glm::radians(float(manip->getFov())) * 0.5f);
  const VkExtent2D size = rig.renderer.renderSize();
  const float      aspect = float(size.width) / float(size.height);
  const float      sx = glm::dot(d, r) / glm::dot(d, f), sy = glm::dot(d, u) / glm::dot(d, f);
  return {(sx / (t * aspect) + 1) * 0.5f * size.width, (1 - sy / t) * 0.5f * size.height};
}

// Placed by a ray at the box's top, a light lands there; a drag of two moves is one step; past the
// box, it lands on the ground ahead
TEST(FvHost, aPlacedPropLandsWhereThePointerMeetsTheScene)
{
  HostRig        rig;
  const fs::path dir = fs::temp_directory_path() / "fv_place_test";
  fs::create_directories(dir);
  writeBox(dir / "box.gltf", {-0.15f, -0.1f, -0.15f}, {0.15f, 0.02f, 0.15f});
  ASSERT_TRUE(rig.cmd("load", {{"path", (dir / "box.gltf").generic_string()}}).value("ok", false));
  ASSERT_TRUE(rig.settle());
  ASSERT_TRUE(rig.cmd("lighting.set", {{"ground", {{"height", -0.1}}}}).value("ok", false));
  const Json added = rig.cmd("prop.add", {{"kind", "light"}, {"transform", {{"t", {1.0, 2.0, 0.0}}}}});
  ASSERT_TRUE(rig.settle());
  const std::string light = added.value("result", Json::object()).value("prop", "");
  const size_t      steps = undoSteps(rig);
  const auto        place = [&](const glm::vec3& at, bool end) {
    const glm::vec2 px = placePixel(rig, at);
    for(int i = 0; i < 20; i++)  // until the box is indexed on the pool
    {
      const Json placed = rig.cmd("prop.place", {{"prop", light}, {"x", px.x}, {"y", px.y}, {"gesture", {{"id", "place:1"}, {"end", end}}}});
      EXPECT_TRUE(placed.value("ok", false)) << placed.dump();
      if(placed.value("result", Json::object()).value("hit", false))
        return;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  };
  place({0.05f, 0.02f, 0.f}, false);
  place({0.f, 0.02f, 0.f}, true);
  const Json t = rig.topic("scene")["props"][light]["transform"]["t"];
  EXPECT_NEAR(t[0].get<double>(), 0.0, 1e-3) << t.dump();
  EXPECT_NEAR(t[1].get<double>(), 0.02, 1e-3) << t.dump();
  EXPECT_NEAR(t[2].get<double>(), 0.0, 1e-3) << t.dump();
  EXPECT_EQ(undoSteps(rig), steps + 1);
  EXPECT_EQ(rig.topic("history").at("undo")[0], "Place Point light");
  // past the box: the ground ahead
  const glm::vec2 px     = placePixel(rig, {0.f, -0.1f, -1.f});
  const Json      placed = rig.cmd("prop.place", {{"prop", light}, {"x", px.x}, {"y", px.y}});
  EXPECT_FALSE(placed.value("result", Json::object()).value("hit", true));
  const Json g = rig.topic("scene")["props"][light]["transform"]["t"];
  EXPECT_NEAR(g[1].get<double>(), -0.1, 1e-3) << g.dump();
  EXPECT_NEAR(g[2].get<double>(), -1.0, 1e-2) << g.dump();
  fs::remove_all(dir);
}

// A figure exported with a dent carries it: the feet stand 2 cm into the box, so the dented body's
// lowest point is higher than the same export's without the dent.
TEST(FvHost, exportCarriesTheContactChannel)
{
  Seated s;
  s.place(0);
  ASSERT_TRUE(s.until([&] { return s.patches() > 0; }));
  ASSERT_TRUE(s.rests());
  const auto lowest = [&](const char* name) {
    const fs::path out = s.dir / name;
    EXPECT_TRUE(s.rig.cmd("export.glb", {{"prop", s.figure}, {"path", out.generic_string()}}).value("ok", false));
    EXPECT_TRUE(s.rig.settle());
    tinygltf::Model    model;
    tinygltf::TinyGLTF reader;
    std::string        error, warning;
    EXPECT_TRUE(reader.LoadBinaryFromFile(&model, &error, &warning, out.string())) << error;
    double low = INFINITY;
    for(const tinygltf::Mesh& m : model.meshes)
      for(const tinygltf::Primitive& p : m.primitives)
        if(const auto at = p.attributes.find("POSITION"); at != p.attributes.end() && model.accessors[size_t(at->second)].minValues.size() >= 2)
          low = std::min(low, model.accessors[size_t(at->second)].minValues[1]);
    return low;
  };
  const double dented = lowest("dented.glb");
  ASSERT_TRUE(s.rig.cmd("figure.contact", {{"prop", s.figure}, {"on", false}}).value("ok", false));
  s.rig.host->onUpdate();
  const double plain = lowest("plain.glb");
  EXPECT_GT(dented, plain + 0.001);
}

//--------------------------------------------------------------------------------------------------
// Contact between people (docs/physics-plan.md step 3): each body meets the other undented and gives
// way by its share.

namespace {

// Two Box Figures (20 cm boxes, corners only), both turned 45 degrees so a corner of each points at
// the other, their centres 27.28 cm apart: each corner 1 cm into the other box (side by side, every
// corner would sit on the other's faces).
struct Pressed
{
  HostRig     rig;
  std::string a, b;
  int         moves = 0;

  Pressed()
  {
    EXPECT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    a = figureId(rig.topic("scene"));
    EXPECT_TRUE(rig.cmd("prop.add", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    const Json props = rig.topic("scene").value("props", Json::object());
    for(const auto& [id, p] : props.items())
      if(p.value("kind", "") == "character" && id != a)
        b = id;
    EXPECT_FALSE(b.empty());
    rig.cmd("prop.set", {{"prop", a}, {"transform", {{"t", {0.0, 0.0, 0.0}}, {"e", {0.0, 45.0, 0.0}}}}});
    place(0.2728);
  }
  void place(double x)
  {
    const Json placed =
        rig.cmd("prop.set", {{"prop", b}, {"transform", {{"t", {x, 0.0, 0.0}}, {"e", {0.0, 45.0, 0.0}}}}, {"gesture", {{"id", ++moves}, {"end", true}}}});
    EXPECT_TRUE(placed.value("ok", false)) << placed.dump();
  }
  Json contact(const std::string& id) const { return rig.topic("contact/" + id); }
  int  patches(const std::string& id) const { return contact(id).value("patches", -1); }
  bool busy(const std::string& id) const { return contact(id).value("busy", true); }
  std::string dump() const { return contact(a).dump() + " " + contact(b).dump(); }
  bool until(const std::function<bool()>& done, double seconds = 5)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while(std::chrono::steady_clock::now() < end)
    {
      rig.host->onUpdate();
      if(done())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
  }
  bool rests()
  {
    return until([&] { return !busy(a) && !busy(b); });
  }
};

}  // namespace

// Pressed together, both dent, and the dents come to rest: neither's write pokes the other. All the
// give to one side, the firm one has nothing to dent and the soft one dents alone.
TEST(FvHost, twoPeoplePressedTogetherDentEachOtherAndRest)
{
  Pressed s;
  ASSERT_TRUE(s.until([&] { return s.patches(s.a) > 0 && s.patches(s.b) > 0; })) << s.dump();
  ASSERT_TRUE(s.rests()) << s.dump();
  for(int i = 0; i < 100; i++)
  {
    s.rig.host->onUpdate();
    ASSERT_FALSE(s.busy(s.a)) << "frame " << i << ": " << s.dump();
    ASSERT_FALSE(s.busy(s.b)) << "frame " << i << ": " << s.dump();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ASSERT_TRUE(s.rig.cmd("figure.contact", {{"prop", s.a}, {"softness", 1.0}}).value("ok", false));
  ASSERT_TRUE(s.rig.cmd("figure.contact", {{"prop", s.b}, {"softness", 0.0}}).value("ok", false));
  ASSERT_TRUE(s.until([&] { return s.patches(s.b) == 0 && !s.busy(s.a) && !s.busy(s.b); })) << s.dump();
  EXPECT_GT(s.patches(s.a), 0) << s.dump();
}

// Another person's pose change (the head turned) is a change to what this body meets: it looks again,
// and comes to rest. Moved off, both dents go.
TEST(FvHost, anotherFiguresPoseChangePokesContact)
{
  Pressed s;
  ASSERT_TRUE(s.until([&] { return s.patches(s.a) > 0 && s.patches(s.b) > 0; })) << s.dump();
  ASSERT_TRUE(s.rests()) << s.dump();
  ASSERT_TRUE(s.rig.cmd("pose.set", {{"prop", s.b}, {"values", {{"head/rotation/x", 30.0}}}}).value("ok", false));
  ASSERT_TRUE(s.until([&] { return s.busy(s.a); }, 1)) << "the other's pose didn't poke it: " << s.dump();
  ASSERT_TRUE(s.rests()) << s.dump();
  s.place(1.0);
  ASSERT_TRUE(s.until([&] { return s.patches(s.a) == 0 && s.patches(s.b) == 0; })) << s.dump();
}

// While the other's ragdoll moves, this body's dent is held at weight 0 (it would show where the other
// was a moment ago); once it rests, the dent comes back.
TEST(FvHost, aDentFromAnotherFiguresRagdollIsHeldWhileItMoves)
{
  Pressed s;
  ASSERT_TRUE(s.until([&] { return s.patches(s.a) > 0; })) << s.dump();
  ASSERT_TRUE(s.rests()) << s.dump();
  EXPECT_FALSE(s.contact(s.a).value("held", true));
  ASSERT_TRUE(s.rig.cmd("physics.set", {{"prop", s.b}, {"on", true}, {"strength", 0.5}}).value("ok", false));
  ASSERT_TRUE(s.rig.cmd("pose.set", {{"prop", s.b}, {"values", {{"head/rotation/x", 30.0}}}}).value("ok", false));
  ASSERT_TRUE(s.until([&] { return s.contact(s.a).value("held", false); }, 2)) << s.dump();
  const auto awake = [&] { return s.rig.topic("physics/" + s.b).value("awake", true); };
  ASSERT_TRUE(s.until([&] { return !awake() && !s.contact(s.a).value("held", true) && !s.busy(s.a); }, 10)) << s.dump();
  EXPECT_GT(s.patches(s.a), 0) << s.dump();
}

//--------------------------------------------------------------------------------------------------
// Physics mode (figure/physics/rig/figure_physics.hpp, docs/physics-plan.md step 2)

namespace {

// The Box Figure (a hip and a head, 40 cm tall) in front of a camera looking at its head from 2 m.
struct Ragdolled
{
  HostRig     rig;
  std::string figure;

  Ragdolled()
  {
    EXPECT_TRUE(rig.cmd("load", {{"rel", "People/Test/Box Figure.duf"}}).value("ok", false));
    EXPECT_TRUE(rig.settle());
    figure = figureId(rig.topic("scene"));
    rig.cmd("camera.goto", {{"position", {0.0, 0.3, 2.0}}, {"target", {0.0, 0.3, 0.0}}});
    rig.host->onUpdate();
  }
  Json physics() const { return rig.topic("physics/" + figure); }
  bool on() const { return physics().value("on", false); }
  // the head's rotation channels as the scene state keeps them
  double head(const char* axis)
  {
    const Json state = rig.cmd("scene.state", {{"prop", figure}}).value("result", Json::object());
    for(const Json& c : state.value("pose", Json::array()))
      if(c.value("bone", "") == "head" && c.value("kind", "") == "rotation" && c.value("axis", "") == axis)
        return c.value("value", 0.0);
    return 0;
  }
  // frames of real time until `done`, or a few seconds
  bool until(const std::function<bool()>& done, double seconds = 5)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while(std::chrono::steady_clock::now() < end)
    {
      rig.host->onUpdate();
      if(done())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
  }
};

}  // namespace

// Physics mode is a step of the history and part of the figure's scene state; a scene file from
// before it opens in Skeletal mode.
TEST(FvHost, physicsModeIsAStepAndTheSceneFileKeepsIt)
{
  Ragdolled r;
  ASSERT_TRUE(r.physics().value("available", false)) << r.physics().dump();
  EXPECT_FALSE(r.on());
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"on", true}}).value("ok", false));
  EXPECT_TRUE(r.on());
  EXPECT_EQ(r.rig.topic("history").at("undo")[0], "Physics on");
  const Json state = r.rig.cmd("scene.state", {{"prop", r.figure}}).value("result", Json::object());
  ASSERT_TRUE(state.contains("physics")) << state.dump();
  EXPECT_TRUE(state["physics"].value("on", false));
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_FALSE(r.on());
  ASSERT_TRUE(r.rig.cmd("history.redo").value("ok", false));
  EXPECT_TRUE(r.on());
  // and again, both ways: each replay rebuilds the other
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_FALSE(r.on());
  ASSERT_TRUE(r.rig.cmd("history.redo").value("ok", false));
  EXPECT_TRUE(r.on());

  Json restore    = state;
  restore["prop"] = r.figure;
  restore.erase("physics");
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_FALSE(r.on()) << "a file without the block opened in Physics mode";
  restore["physics"] = state["physics"];
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_TRUE(r.on());
}

// A pin is a step of the history and part of the figure's scene state; leaving Physics lets it go,
// and undoing that holds it again. Out of Physics mode there is nothing to pin.
TEST(FvHost, aPinIsAStepAndTheSceneFileKeepsIt)
{
  Ragdolled  r;
  const auto pins = [&] { return r.rig.topic("physics/" + r.figure).value("pins", Json::array()); };
  EXPECT_FALSE(r.rig.cmd("physics.pin", {{"prop", r.figure}, {"bone", "head"}}).value("ok", false)) << "pinned in Skeletal mode";
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"on", true}}).value("ok", false));
  const Json pinned = r.rig.cmd("physics.pin", {{"prop", r.figure}, {"bone", "head"}});
  ASSERT_TRUE(pinned.value("ok", false)) << pinned.dump();
  ASSERT_EQ(pins().size(), 1u) << r.physics().dump();
  EXPECT_EQ(pins()[0].value("bone", ""), "head");
  EXPECT_EQ(r.rig.topic("history").at("undo")[0].get<std::string>().rfind("Pin ", 0), 0u) << r.rig.topic("history").dump();
  const Json state = r.rig.cmd("scene.state", {{"prop", r.figure}}).value("result", Json::object());
  ASSERT_EQ(state.at("physics").value("pins", Json::array()).size(), 1u) << state.dump();
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_TRUE(pins().empty());
  ASSERT_TRUE(r.rig.cmd("history.redo").value("ok", false));
  EXPECT_EQ(pins().size(), 1u);

  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"on", false}}).value("ok", false));
  EXPECT_TRUE(pins().empty());
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_EQ(pins().size(), 1u) << "undoing leaving Physics didn't hold the pin again";

  Json restore    = state;
  restore["prop"] = r.figure;
  restore["physics"]["pins"] = Json::array();
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_TRUE(pins().empty());
  restore["physics"] = state["physics"];
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_EQ(pins().size(), 1u);
}

// The soft bones' settings are a step of the history and part of the figure's scene state; a file
// without the block has them off. The Box Figure has no such bones: it keeps the settings, and says so.
TEST(FvHost, softBodySettingsAreAStepAndTheSceneFileKeepsThem)
{
  Ragdolled  r;
  const auto soft = [&] { return r.rig.topic("secondary/" + r.figure); };
  EXPECT_FALSE(soft().value("available", true)) << soft().dump();
  EXPECT_FALSE(soft().value("on", true));
  ASSERT_TRUE(r.rig.cmd("physics.secondary", {{"prop", r.figure}, {"on", true}, {"bounce", 0.7}}).value("ok", false));
  EXPECT_TRUE(soft().value("on", false));
  EXPECT_DOUBLE_EQ(soft().value("bounce", 0.0), 0.7);
  EXPECT_EQ(r.rig.topic("history").at("undo")[0], "Soft body on");
  const Json state = r.rig.cmd("scene.state", {{"prop", r.figure}}).value("result", Json::object());
  ASSERT_TRUE(state.contains("secondary")) << state.dump();
  EXPECT_TRUE(state["secondary"].value("on", false));
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_FALSE(soft().value("on", true));
  ASSERT_TRUE(r.rig.cmd("history.redo").value("ok", false));
  EXPECT_TRUE(soft().value("on", false));

  Json restore    = state;
  restore["prop"] = r.figure;
  restore.erase("secondary");
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_FALSE(soft().value("on", true)) << "a file without the block opened with them on";
  restore["secondary"] = state["secondary"];
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_TRUE(soft().value("on", false));
  EXPECT_DOUBLE_EQ(soft().value("bounce", 0.0), 0.7);
}

// Soft flesh's settings are steps of the history, in the topic and the figure's scene state; a file
// without the block opens with it off. (The test figure has no lobes: the settings are kept all the same.)
TEST(FvHost, softFleshSettingsAreAStepAndTheSceneFileKeepsThem)
{
  Ragdolled  r;
  const auto soft = [&] { return r.rig.topic("soft/" + r.figure); };
  EXPECT_FALSE(soft().value("on", true)) << soft().dump();
  EXPECT_FALSE(soft().value("awake", true));
  ASSERT_TRUE(r.rig.cmd("physics.soft", {{"prop", r.figure}, {"on", true}}).value("ok", false));
  EXPECT_TRUE(soft().value("on", false));
  EXPECT_EQ(r.rig.topic("history").at("undo")[0], "Soft flesh on");
  ASSERT_TRUE(r.rig.cmd("physics.soft", {{"prop", r.figure}, {"softness", {{"glutes", 0.9}}}, {"damping", 0.6}}).value("ok", false));
  EXPECT_DOUBLE_EQ(soft().at("softness").value("glutes", 0.0), 0.9) << soft().dump();
  EXPECT_DOUBLE_EQ(soft().at("softness").value("chest", 0.0), 0.7) << "the others as they were";
  EXPECT_DOUBLE_EQ(soft().value("damping", 0.0), 0.6);
  const Json state = r.rig.cmd("scene.state", {{"prop", r.figure}}).value("result", Json::object());
  ASSERT_TRUE(state.contains("soft")) << state.dump();
  EXPECT_DOUBLE_EQ(state["soft"].at("softness").value("glutes", 0.0), 0.9);
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_DOUBLE_EQ(soft().at("softness").value("glutes", 0.0), 0.6);
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_FALSE(soft().value("on", true));
  ASSERT_TRUE(r.rig.cmd("history.redo").value("ok", false));
  EXPECT_TRUE(soft().value("on", false));

  Json restore    = state;
  restore["prop"] = r.figure;
  restore.erase("soft");
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_FALSE(soft().value("on", true)) << "a file without the block opened with it on";
  restore["soft"] = state["soft"];
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_TRUE(soft().value("on", false));
  EXPECT_DOUBLE_EQ(soft().at("softness").value("glutes", 0.0), 0.9);
}

// A body group's own Hold is a step of the history, in the topic and the figure's scene state; null
// gives the group back to the figure's Hold.
TEST(FvHost, aGroupsHoldIsAStepAndTheSceneFileKeepsIt)
{
  Ragdolled r;
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"on", true}}).value("ok", false));
  const auto held = [&] { return r.physics().value("parts", Json::object()); };
  EXPECT_TRUE(held().empty());
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"parts", {{"Head & neck", 0.2}}}}).value("ok", false));
  EXPECT_DOUBLE_EQ(held().value("Head & neck", 0.0), 0.2) << r.physics().dump();
  EXPECT_EQ(r.rig.topic("history").at("undo")[0], "Hold Head & neck");
  const Json state = r.rig.cmd("scene.state", {{"prop", r.figure}}).value("result", Json::object());
  EXPECT_DOUBLE_EQ(state.at("physics").value("parts", Json::object()).value("Head & neck", 0.0), 0.2) << state.dump();
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_TRUE(held().empty()) << r.physics().dump();
  ASSERT_TRUE(r.rig.cmd("history.redo").value("ok", false));
  EXPECT_DOUBLE_EQ(held().value("Head & neck", 0.0), 0.2);
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"parts", {{"Head & neck", nullptr}}}}).value("ok", false));
  EXPECT_TRUE(held().empty());

  Json restore    = state;
  restore["prop"] = r.figure;
  r.rig.cmd("scene.restore", restore);
  ASSERT_TRUE(r.rig.settle());
  EXPECT_DOUBLE_EQ(held().value("Head & neck", 0.0), 0.2) << r.physics().dump();
}

// A drag with the Pull tool on the figure's head turns it, the channels keep where it got to, and
// the pull is one step: undone, the head is back.
TEST(FvHost, aPullTurnsTheHeadAndIsOneStep)
{
  Ragdolled r;
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"on", true}, {"strength", 0.5}}).value("ok", false));
  ASSERT_TRUE(r.rig.cmd("view.set", {{"pick", "pull"}}).value("ok", false));
  ASSERT_TRUE(r.until([&] { return !r.physics().value("awake", true); }));
  const size_t steps = r.rig.topic("history").at("undo").size();

  r.rig.input({{{"t", "down"}, {"b", 0}, {"x", 640}, {"y", 360}}});
  ASSERT_TRUE(r.until([&] { return r.physics().value("pulling", false); }, 1)) << r.physics().dump();
  // 10 cm to the side, at 2 m: the head tips over
  r.rig.input({{{"t", "move"}, {"x", 640 + 60}, {"y", 360}}});
  r.until([] { return false; }, 1);
  r.rig.input({{{"t", "up"}, {"b", 0}, {"x", 640 + 60}, {"y", 360}}});
  ASSERT_TRUE(r.until([&] { return !r.physics().value("pulling", true); }, 1));
  const double z = r.head("z");
  EXPECT_GT(std::abs(z) + std::abs(r.head("x")), 3) << "the pull didn't turn the head";
  const Json undo = r.rig.topic("history").at("undo");
  ASSERT_EQ(undo.size(), steps + 1);
  EXPECT_EQ(undo[0], "Pull Box Figure");

  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_NEAR(r.head("z"), 0, 1e-6);
  EXPECT_NEAR(r.head("x"), 0, 1e-6);
  // a drag that starts off the figure orbits the camera, as it would without the tool
  r.rig.input({{{"t", "down"}, {"b", 0}, {"x", 40}, {"y", 40}}});
  r.rig.host->onUpdate();
  EXPECT_FALSE(r.physics().value("pulling", true));
  r.rig.input({{{"t", "up"}, {"b", 0}, {"x", 40}, {"y", 40}}});
}

// Leaving Physics mode keeps the pose the ragdoll reached in the channels; undone, it is in Physics
// mode again, the head where it was.
TEST(FvHost, leavingPhysicsKeepsTheReachedPose)
{
  Ragdolled r;
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"on", true}, {"strength", 0.5}}).value("ok", false));
  ASSERT_TRUE(r.rig.cmd("view.set", {{"pick", "pull"}}).value("ok", false));
  r.rig.input({{{"t", "down"}, {"b", 0}, {"x", 640}, {"y", 360}}});
  ASSERT_TRUE(r.until([&] { return r.physics().value("pulling", false); }, 1));
  r.rig.input({{{"t", "move"}, {"x", 640 + 60}, {"y", 360}}});
  r.until([] { return false; }, 1);
  r.rig.input({{{"t", "up"}, {"b", 0}, {"x", 640 + 60}, {"y", 360}}});
  ASSERT_TRUE(r.until([&] { return !r.physics().value("awake", true); }));
  const double z = r.head("z"), x = r.head("x");
  ASSERT_GT(std::abs(z) + std::abs(x), 3);
  ASSERT_TRUE(r.rig.cmd("physics.set", {{"prop", r.figure}, {"on", false}}).value("ok", false));
  EXPECT_FALSE(r.on());
  // what it shows, which settled a hair from what it was let go at
  EXPECT_NEAR(r.head("z"), z, 0.05);
  EXPECT_NEAR(r.head("x"), x, 0.05);
  ASSERT_TRUE(r.rig.cmd("history.undo").value("ok", false));
  EXPECT_TRUE(r.on());
  EXPECT_NEAR(r.head("z"), z, 1e-6);
}

// A VaM scene's Timeline clip on the box figure (docs/physics-plan.md step 16): opened as "vam", it
// plays in Physics mode as a figure's clip of the animation topic, with the Person's other clips; its
// hip and head controllers hold the ragdoll, and the head turns as the clip turns it (Unity's turn
// about y, mirrored onto ours). Stop lets it go, the head comes back, and the scene file leaves the
// clip out.
TEST(FvHost, aVamClipHoldsTheRagdollInPhysicsMode)
{
  Ragdolled r;
  // a key as Timeline writes it: 'D', the time, the value and the curve type, as hex float32s
  const auto key = [](float t, float v) {
    char          out[32];
    uint32_t      bits[2];
    std::memcpy(&bits[0], &t, 4);
    std::memcpy(&bits[1], &v, 4);
    std::string s = "D";
    for(uint32_t b : bits)
      for(int i = 0; i < 4; i++)
      {
        std::snprintf(out, sizeof out, "%02X", unsigned((b >> (8 * i)) & 255));
        s += out;
      }
    return s + "03";
  };
  // the head turned 40 degrees about Unity's y, held there over the clip
  const double half  = 20 * 3.14159265358979323846 / 180;
  const auto   track = [&](const char* controller, float y, float qy, float qw, float length) {
    Json t = {{"Controller", controller}, {"TargetsPosition", "1"}, {"TargetsRotation", "1"}};
    const float v[7] = {0, y, 0, 0, qy, 0, qw};
    const char* names[7] = {"X", "Y", "Z", "RotX", "RotY", "RotZ", "RotW"};
    for(int i = 0; i < 7; i++)
      t[names[i]] = Json::array({key(0, v[i]), key(length, v[i])});
    return t;
  };
  const auto clip = [&](const char* name, float length) {
    Json pose = Json::array({{{"id", "control"}, {"position", {{"x", "0"}, {"y", "0"}, {"z", "0"}}}, {"rotation", {{"x", "0"}, {"y", "0"}, {"z", "0"}}}},
                             {{"id", "hipControl"}, {"positionState", "On"}, {"rotationState", "On"}},
                             {{"id", "headControl"}, {"positionState", "On"}, {"rotationState", "On"}, {"holdRotationSpring", "450"}}});
    return Json{{"AnimationName", name},
                {"AnimationLength", std::to_string(length)},
                {"Loop", "0"},
                {"Controllers", Json::array({track("hipControl", 0.1f, 0, 1, length), track("headControl", 0.2f, float(std::sin(half)), float(std::cos(half)), length)})},
                {"Pose", {{"storables", pose}}}};
  };
  const Json scene = {{"atoms", Json::array({{{"id", "Person"},
                                              {"type", "Person"},
                                              {"storables", Json::array({{{"id", "plugin#0_VamTimeline.AtomPlugin"},
                                                                          {"Animation", {{"SerializeVersion", "283"}, {"Clips", Json::array({clip("A POS", 1), clip("Turn", 12)})}}}}})}}})}};
  const fs::path dir = fs::temp_directory_path() / "fv_host_vam_test";
  fs::create_directories(dir);
  const fs::path file = dir / "Turn Scene.json";
  std::ofstream(file) << scene.dump();

  ASSERT_TRUE(r.rig.cmd("load", {{"prop", r.figure}, {"path", file.string()}, {"as", "vam"}}).value("ok", false));
  ASSERT_TRUE(r.rig.settle());
  EXPECT_TRUE(r.on()) << "a VaM clip didn't turn Physics mode on";
  const Json anim = r.rig.topic("animation");
  ASSERT_TRUE(anim.value("current", Json()).is_number()) << anim.dump();
  const Json c = anim.at("clips").at(std::to_string(anim["current"].get<int>()));
  EXPECT_EQ(c.value("name", ""), "Turn") << "not the longest clip";
  EXPECT_EQ(c.value("kind", ""), "figure");
  EXPECT_DOUBLE_EQ(c.value("duration", 0.0), 12);
  ASSERT_TRUE(c.contains("vam")) << c.dump();
  EXPECT_EQ(c["vam"].value("person", ""), "Person");
  EXPECT_EQ(c["vam"].value("clips", Json::array()).size(), 2u);
  EXPECT_TRUE(anim.value("playing", false));

  // Unity's turn about y is the other way about ours
  const auto headY = [&] {
    const Json state = r.rig.cmd("scene.state", {{"prop", r.figure}}).value("result", Json::object());
    return state.value("physics", Json::object()).value("physical", Json::object()).value("head/rotation/y", 0.0);
  };
  EXPECT_TRUE(r.until([&] { return std::abs(headY() + 40) < 4; }, 8)) << "the head is at " << headY();
  const Json state = r.rig.cmd("scene.state", {{"prop", r.figure}}).value("result", Json::object());
  EXPECT_FALSE(state.contains("animation")) << "a VaM clip went into the scene file";

  r.rig.cmd("anim.stop");
  EXPECT_TRUE(r.until([&] { return std::abs(headY()) < 2; }, 8)) << "let go, the head stayed at " << headY();
  // a scene without a Person with clips, and a figure without the clip named
  r.rig.cmd("load", {{"prop", r.figure}, {"path", file.string()}, {"as", "vam"}, {"clip", "Nope"}});
  ASSERT_TRUE(r.rig.settle());
  EXPECT_NE(r.rig.topic("notices").dump().find("has no clip Nope"), std::string::npos) << r.rig.topic("notices").dump();
  fs::remove_all(dir);
}
