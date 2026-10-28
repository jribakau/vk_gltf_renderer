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
// FvHost: the protocol's commands and topics, and the application-thread hooks. The rest of its
// implementation is in the fv_host_<area>.cpp files (fv_host_impl.hpp).
//

#include "fv_host_impl.hpp"

//--------------------------------------------------------------------------------------------------

FvHost::FvHost(HostRenderer* renderer, std::shared_ptr<FvProtocol> protocol, nvutils::ParameterRegistry* registry)
    : m_impl(std::make_unique<Impl>())
{
  Impl& s     = *m_impl;
  s.renderer  = renderer;
  s.protocol  = std::move(protocol);
  s.registry  = registry;
  auto& cmds  = s.protocol->commands();
  Impl* impl  = m_impl.get();
  // the figure library's phase timers (loads, wears, glTF builds) go to the log
  fv::timing::setSink([](const std::string& line) { LOGI("%s\n", line.c_str()); });
  // what this build has beyond the protocol, so the app offers only that
#ifdef FV_DFORCE_GPU
  s.protocol->setFeatures({{"dforceGpu", true}});
#else
  s.protocol->setFeatures({{"dforceGpu", false}});
#endif

  cmds.add("settings.apply", [impl](const Json& args) {
    std::lock_guard lock(impl->settingsMutex);
    HostSettings&       st = impl->settings;
    if(args.contains("libraryRoots"))
    {
      st.roots.clear();
      for(const auto& r : args["libraryRoots"])
        st.roots.push_back(pathOf(r.get<std::string>()));
    }
    st.rootId         = args.value("rootId", st.rootId);
    st.texServer      = args.value("texServer", st.texServer);
    st.maxTextureSize = args.value("maxTextureSize", st.maxTextureSize);
    st.subdivision    = args.value("subdivision", st.subdivision);
    st.originalTextures = args.value("originalTextures", st.originalTextures);
    st.figureGpuRefine  = args.value("figureGpuRefine", st.figureGpuRefine);
    st.figureDazSkinning = args.value("figureDazSkinning", st.figureDazSkinning);
    st.figureDazOrder    = args.value("figureDazOrder", st.figureDazOrder);
    return Json{{"ok", true}};
  });
  // load works out on a worker what the file is, and comes to the application thread for the rest
  cmds.add("load", [impl](const Json& args) { return impl->load(args); });
  cmds.add("prop.add", [impl](const Json& args) {
    if(args.contains("path") || args.contains("rel"))
    {
      Json asProp  = args;
      asProp["as"] = "prop";
      return impl->load(asProp);
    }
    return impl->onApp([impl, args] { return impl->propAdd(args); });
  });
  cmds.add("prop.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("prop.set", args); }); });
  cmds.add("prop.remove", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      auto p = impl->findProp(args.at("prop").get<std::string>());
      if(!p)
        throw std::runtime_error("no such prop");
      impl->removeProp(p);
      impl->publishScene();
      impl->requestCompose("Removing " + p->name);
      return Json{{"ok", true}};
    });
  });
  cmds.add("prop.duplicate", [impl](const Json& args) {
    // a figure loads again, which waits on workers: answered with its job
    return impl->onApp([impl, args] { return impl->propDuplicate(args); });
  });
  cmds.add("scene.clear", [impl](const Json&) {
    return impl->onApp([impl] {
      // removing a prop is no step yet (C3): the steps of the props that went would find nothing
      impl->journal.clear();
      impl->publishHistory();
      return impl->sceneClear();
    });
  });
  cmds.add("scene.order", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("scene.order", args); }); });
  cmds.add("view.use", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      fv::timing::Scope t("view.use", 20);
      return impl->edit("view.use", args);
    });
  });
  // the edit history (fv_journal.hpp): the step to undo or redo, applied -> {label}, null when there is none
  cmds.add("history.undo", [impl](const Json&) { return impl->onApp([impl] { return impl->historyStep(true); }); });
  cmds.add("history.redo", [impl](const Json&) { return impl->onApp([impl] { return impl->historyStep(false); }); });
  for(const char* name : {"select", "play", "pause", "stop", "seek", "speed", "loop"})
    cmds.add(std::string("anim.") + name, [impl, cmd = std::string(name)](const Json& args) {
      return impl->onApp([impl, cmd, args] { return impl->animationCommand(cmd, args); });
    });
  cmds.add("export.glb", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->exportGlb(args); }); });
  cmds.add("scene.state", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->sceneState(args); }); });
  cmds.add("scene.restore", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->sceneRestore(args); }); });
  cmds.add("wear", [impl](const Json& args) { return impl->wear(args); });
  cmds.add("wardrobe.remove", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->wardrobeRemove(args); }); });
  cmds.add("wardrobe.visible", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("wardrobe.visible", args); }); });
  cmds.add("wardrobe.collision", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("wardrobe.collision", args); }); });
  cmds.add("wardrobe.cloth", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("wardrobe.cloth", args); }); });
  cmds.add("dforce.enable", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("dforce.enable", args); }); });
  cmds.add("dforce.mode", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("dforce.mode", args); }); });
  cmds.add("dforce.resimulate", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->dforceCommand("resimulate", args); }); });
  cmds.add("material.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("material.set", args); }); });
  cmds.add("material.reset", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("material.reset", args); }); });
  cmds.add("material.eyes", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materialEyes(args); }); });
  cmds.add("wardrobe.color", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->wardrobeColor(args); }); });
  cmds.add("material.skin", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materialSkin(args); }); });
  cmds.add("material.makeup", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->materialMakeup(args); }); });
  cmds.add("pose.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("pose.set", args); }); });
  cmds.add("pose.eyes", [impl](const Json& args) {
    return impl->onApp([impl, args] { return impl->journaled("pose.eyes", args, [impl, &args] {
      const auto p = impl->figureProp(args);
      if(!p || !p->figure->fig->pose)
        throw std::runtime_error("no such figure");
      auto manip = impl->renderer->getCameraManipulator();
      if(!manip)
        throw std::runtime_error("no camera");
      // the eye in the figure's own space, under its prop root
      const glm::vec3 eye = glm::vec3(glm::inverse(impl->worldOf(*p)) * glm::vec4(glm::vec3(manip->getEye()), 1.f));
      const int       n   = p->figure->fig->pose->aimEyes({eye.x, eye.y, eye.z});
      std::vector<std::string> bones;
      for(const auto& pb : p->figure->fig->pose->bones())
        if(fv::pose::isEyeBone(pb.name))
          bones.push_back(pb.id);
      impl->patchPose(*p->figure, bones);
      return Json{{"aimed", n}};
    }); });
  });
  cmds.add("pose.reset", [impl](const Json& args) {
    return impl->onApp([impl, args] { return impl->journaled("pose.reset", args, [impl, &args] {
      auto figure = impl->figureArg(args);
      if(!figure || !figure->fig->pose)
        throw std::runtime_error("no such figure");
      figure->fig->pose->reset();
      impl->protocol->events().snapshot("pose/" + figure->id, poseSnapshot(*figure));
      return Json{{"changed", 0}};
    }); });
  });
  // answered at once: the values merge with whatever else arrived this frame and apply in onUpdate
  cmds.add("morph.set", [impl](const Json& args) {
    std::lock_guard   lock(impl->morphMutex);
    const std::string fig = args.contains("prop") ? args.value("prop", "") : args.value("figure", "");
    if(!impl->morphFigures.count(fig))
      throw std::runtime_error("no such figure");
    const Json* values = args.contains("values") ? &args["values"] : nullptr;
    if(!values || !values->is_object())
      throw std::runtime_error("morph.set wants values: {name: value}");
    for(auto it = values->begin(); it != values->end(); ++it)
      if(it.value().is_number())
        impl->pendingMorphs[fig][it.key()] = it.value().get<double>();
    if(args.contains("gesture"))
      impl->pendingGestures[fig] = args["gesture"];
    return Json{{"ok", true}};
  });
  cmds.add("morph.reset", [impl](const Json& args) {
    return impl->onApp([impl, args] { return impl->journaled("morph.reset", args, [impl, &args] {
      auto figure = impl->figureArg(args);
      if(!figure || !figure->fig->morphs)
        throw std::runtime_error("no such figure");
      {
        std::lock_guard lock(impl->morphMutex);
        impl->pendingMorphs.erase(figure->id);
      }
      figure->fig->morphs->resetAll();
      return Json{{"ok", true}};
    }); });
  });
  cmds.add("render.mode", [impl](const Json& args) {
    const std::string mode = args.at("mode").get<std::string>();
    if(mode != "raster" && mode != "rtx")
      throw std::runtime_error("mode is raster or rtx");
    // RTX skin: a volume of measured human skin (the default), the file's Iray volume, or
    // thin-walled scattering at the surface
    const std::string skin = args.value("skin", std::string("measured"));
    if(skin != "volume" && skin != "thin" && skin != "measured")
      throw std::runtime_error("skin is volume, thin or measured");
    const fv::mat::SkinMode rtxSkin = skin == "thin"       ? fv::mat::SkinMode::eThin :
                                      skin == "measured" ? fv::mat::SkinMode::eMeasured :
                                                           fv::mat::SkinMode::eVolume;
    return impl->onApp([impl, mode, rtxSkin] {
      impl->renderer->setRenderSystem(mode == "rtx" ? RenderingMode::ePathtracer : RenderingMode::eRasterizer);
      impl->skinMode = mode == "rtx" ? rtxSkin : fv::mat::SkinMode::eRaster;
      for(const auto& p : impl->props)
        if(p->figure && p->materials)
        {
          // the feed lowers every material with the new skin, then the edits' own lowering goes over it
          if(p->figure->feed)
          {
            p->figure->feed->setOptions(buildOptionsOf(*p->figure, impl->skinMode));
            p->figure->feed->sync(p->figure->fig->loaded.scene, p->figure->fig->loaded.root);
          }
          p->materials->setSkinMode(impl->skinMode);
        }
      impl->protocol->events().snapshot("render", {{"mode", mode}});
      return Json{{"mode", mode}};
    });
  });
  // whether the renderer is still refining a still, for an image export that waits for it
  cmds.add("render.status", [impl](const Json&) {
    return impl->onApp([impl] {
      const VkExtent2D size = impl->renderer->renderSize();
      return Json{{"refining", impl->renderer->isRefiningStill()},
                  {"still", impl->renderer->isStill()},
                  {"converged", impl->renderer->isStillConverged()},
                  {"complete", impl->renderer->accumulationComplete()},
                  {"rtx", impl->renderer->renderSystem() == RenderingMode::ePathtracer},
                  // the frame's size: a /size lands before the next frame, and until then the rest is the old frame's
                  {"width", size.width},
                  {"height", size.height}};
    });
  });
  // GPU memory, for the Export dialog: the budget (VK_EXT_memory_budget; `known` false without it)
  // and what a pixel of the frame costs in the mode drawn, so a size is judged before it is asked for
  cmds.add("render.budget", [impl](const Json&) {
    return impl->onApp([impl] {
      const auto       b    = impl->renderer->deviceMemoryBudget();
      const VkExtent2D size = impl->renderer->renderSize();
      return Json{{"known", b.known},   {"budgetBytes", b.budget},
                  {"usageBytes", b.usage}, {"bytesPerPixel", impl->renderer->frameBytesPerPixel()},
                  {"width", size.width},   {"height", size.height}};
    });
  });
  cmds.add("lighting.set", [impl](const Json& args) { return impl->onApp([impl, args] { return impl->edit("lighting.set", args); }); });
  // any registered renderer parameter, read or written as nvmcp's parameter tools do
  cmds.add("params.get", [impl](const Json& args) {
    return impl->onApp([impl, args] { return toolJson(nvmcp::detail::getParameters(exposedParameters(impl->registry), args.dump())); });
  });
  cmds.add("params.set", [impl](const Json& args) {
    return impl->onApp([impl, args] {
      return toolJson(nvmcp::detail::setParameters(exposedParameters(impl->registry), args.dump()));
    });
  });
  // the app's own timings (an open's phases, measured where the user waits), into the timing log
  cmds.add("app.timing", [](const Json& args) {
    if(args.contains("lines") && args["lines"].is_array())
      for(const Json& l : args["lines"])
        if(l.is_string())
          fv::timing::line("app: " + l.get<std::string>());
    return Json::object();
  });
  // a clean exit (DLSS/NGX shut down), which the host asks for before it ends the process
  cmds.add("session.shutdown", [impl](const Json&) {
    impl->post([impl] {
      if(impl->app)
        impl->app->close();
    });
    return Json{{"ok", true}};
  });
  s.view = std::make_unique<FvView>(renderer, s.protocol->events(), s.protocol->commands(), [impl](std::function<Json()> f) {
    return impl->onApp(std::move(f));
  });
  s.view->onHelperPick([impl](const glm::vec2& px) { return impl->helperAt(px); });
  // camera.set edits the lens of the view in use: the camera looked through keeps it, else the free view
  s.view->onLensChange([impl](const FvView::Lens& lens) {
    if(const auto p = impl->findProp(impl->viewCamera))
      impl->journaled("camera.set", {{"prop", p->id}}, [impl, &p, &lens] {
        p->camera.lens = lens;
        impl->publishScene();
        return Json();
      });
    else
      impl->freeView.lens = lens;
  });
  s.protocol->setInputSink([impl](const Json& events) {
    std::lock_guard lock(impl->inputMutex);
    for(const auto& e : events)
      impl->input.push_back(e);
  });
  // the stream stops before this element detaches: its server joins the HTTP threads, which must not wait on this queue
  s.protocol->setStopHook([impl] { impl->stopAccepting(); });
  // The gizmo moves, turns and scales a selected prop (object picking), and poses a figure's bones
  // through its rig (bone picking), so limits and formulas apply.
  auto hasKind = [](const fv::pose::PoseBone& pb, fv::pose::ChannelKind kind) {
    return std::any_of(pb.axes.begin(), pb.axes.end(), [kind](const fv::pose::Axis& a) { return a.kind == kind; });
  };
  renderer->setGizmoHooks({
      .attach = [impl, hasKind](rs::EntityId entity) -> uint32_t {
        if(!impl->view)
          return 0;
        const std::string& pick = impl->view->pickMode();
        if(pick == "object")
        {
          if(!impl->view->propOfRoot(entity))
            return 0;
          const std::string& t = impl->view->transformMode();
          return t == "rotate" ? HostRenderer::eGizmoRotation :
                 t == "scale"  ? HostRenderer::eGizmoScale :
                                 HostRenderer::eGizmoTranslation;
        }
        if(pick != "bone")
          return 0;
        const auto [figure, pb] = impl->boneAt(entity);
        uint32_t flags          = 0;
        if(pb && hasKind(*pb, fv::pose::ChannelKind::eRotation))
          flags |= HostRenderer::eGizmoRotation;
        if(pb && hasKind(*pb, fv::pose::ChannelKind::eTranslation))
          flags |= HostRenderer::eGizmoTranslation;
        return flags;
      },
      .edit =
          [impl, hasKind](rs::EntityId entity, const glm::vec3& t, const glm::quat& r, const glm::vec3& scale, bool done) {
            if(const FvView::PropNodes* nodes = impl->view ? impl->view->propOfRoot(entity) : nullptr)
            {
              auto p = impl->findProp(nodes->id);
              if(!p)
                return;
              // one step of the history from press to release
              const Json args = {{"prop", p->id}, {"transform", Json::object()}, {"gesture", {{"id", "gizmo:" + std::to_string(impl->gizmoGesture)}, {"end", done}}}};
              impl->journaled("prop.set", args, [&] {
                p->transform = {t, glm::normalize(r), p->kind == PropKind::eCamera ? glm::vec3(1.f) : scale};
                p->euler.reset();
                impl->applyTransform(*p);
                impl->movedProps.insert(p->id);
                if(p->id == impl->viewCamera)
                  impl->viewFromCamera(*p);
                return Json();
              });
              if(done)
                impl->gizmoGesture++;
              return;
            }
            // the release repeats the last move's transform; a click on a handle with no drag has only the release
            if(done)
            {
              impl->gizmoGesture++;
              return;
            }
            const auto [figure, pb] = impl->boneAt(entity);
            if(!pb)
              return;
            impl->gizmoEdit = Impl::GizmoEdit{figure, pb->id, fv::Quat{r.x, r.y, r.z, r.w},
                                              hasKind(*pb, fv::pose::ChannelKind::eTranslation) ?
                                                  std::optional<fv::Vec3>(fv::Vec3{t.x, t.y, t.z}) :
                                                  std::nullopt,
                                              impl->gizmoGesture};
          },
  });
}

FvHost::~FvHost() = default;

void FvHost::onAttach(nvapp::Application* app)
{
  m_impl->app = app;
  m_impl->protocol->events().snapshot(
      "render", {{"mode", m_impl->renderer->renderSystem() == RenderingMode::ePathtracer ? "rtx" : "raster"}});
  m_impl->publishScene();
  m_impl->publishHistory();
}

void FvHost::onDetach()
{
  Impl& s = *m_impl;
  s.stopAccepting();
  s.protocol->setStopHook({});
  s.loadGeneration++;  // loads still running drop their result
  textureCalls().stopAll();  // and don't wait out a cook
  s.loaders.joinAll();
  s.queue.clear();
  s.renderer->setGizmoHooks({});
  s.renderer->setPickObserver({});
  if(s.view)
    s.view->setProps({});
  for(const auto& p : s.props)
    if(p->materials)
      p->materials->detach();
  // the feeds let go of what they fed while the renderer's scene is still there
  s.releaseFed();
  s.props.clear();
  s.pendingCompose.reset();
  {
    std::lock_guard morphLock(s.morphMutex);
    s.morphFigures.clear();
    s.pendingMorphs.clear();
  }
}

void FvHost::onUpdate()
{
  Impl& s = *m_impl;
  if(s.lastUpdateEnd && fv::timing::enabled())
    fv::timing::report("frame: renderer between updates", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *s.lastUpdateEnd).count(), 100);
  // the default look until the app sends its own: the renderer's remembered settings don't apply here
  if(!s.lightingStarted)
  {
    s.lightingStarted = true;
    try
    {
      s.applyLighting(true);
    }
    catch(const std::exception& e)
    {
      LOGW("[fv] lighting: %s\n", e.what());
    }
  }
  std::deque<std::function<void()>> work;
  {
    std::lock_guard lock(s.queueMutex);
    work.swap(s.queue);
  }
  // a phase of the frame that holds the application thread up shows in the timing log
  std::optional<fv::timing::Scope> phase;
  const auto timePhase = [&phase](const char* name) {
    phase.reset();
    phase.emplace(name, 20);
  };
  timePhase("frame: queued work");
  for(auto& f : work)
  {
    try
    {
      fv::timing::Scope task("frame: one queued task", 100);
      f();
    }
    catch(const std::exception& e)
    {
      s.notice(e.what(), "error");
    }
  }
  timePhase("frame: compose");
  // what this frame's work changed in the scene's structure, composed once
  if(s.pendingCompose)
  {
    try
    {
      s.compose();
    }
    catch(const std::exception& e)
    {
      s.notice(std::string("The scene could not be built: ") + e.what(), "error");
    }
  }
  timePhase("frame: input and morphs");
  // input first: a gizmo drag poses or moves this frame, and what the gizmo didn't take moves the camera
  std::vector<Json> events;
  {
    std::lock_guard lock(s.inputMutex);
    events.swap(s.input);
  }
  if(!events.empty())
  {
    s.normalizeInput(events);
    s.applyInput(s.view ? s.view->onInput(events) : events);
  }
  s.applyGizmoEdit();
  // not while a composition settles: the renderer moves the view to a scene camera for a frame or two
  if(!s.viewCamera.empty() && s.frameIn == 0)
    s.cameraFromView();
  s.applyPendingMorphs();
  for(const auto& p : s.props)
  {
    if(!p->figure)
      continue;
    Loaded& f = *p->figure;
    timePhase("frame: figure pump");
    fv::Figure::Changes changes = f.fig->pump();
    // this frame's edits first, then a settle or a collision pass that finished writes its result
    // and is drained on its own: its vertices never hide an edit to the same mesh, and are no
    // reason to run again
    // (a catch-up of the drawn vertices to a cage already reported moved nothing new)
    std::vector<int> moved;
    for(const fv::morph::MeshChange& m : changes.meshes)
      if(!m.caughtUp)
        moved.push_back(m.mesh);
    const auto drainInto = [&] {
      fv::Figure::Changes more = f.fig->drain();
      changes.nodes.insert(more.nodes.begin(), more.nodes.end());
      changes.meshes.insert(changes.meshes.end(), std::make_move_iterator(more.meshes.begin()), std::make_move_iterator(more.meshes.end()));
      changes.rebound.insert(changes.rebound.end(), more.rebound.begin(), more.rebound.end());
    };
    timePhase("frame: dForce and collision");
    if(f.dforce)
    {
      if(!changes.nodes.empty() || !moved.empty())
        f.dforce->changed(moved, !changes.nodes.empty());
      if(f.dforce->pump())
        drainInto();
      s.publishDforce(f, false);
    }
    if(f.collider)
    {
      if(!changes.nodes.empty() || !moved.empty())
        f.collider->changed(moved, !changes.nodes.empty());
      if(f.collider->pump())
        drainInto();
      if(f.collisionVersion != f.collider->version())
        s.publishWardrobe(f);
    }
    timePhase("frame: figure changes applied");
    if(!changes.nodes.empty())
    {
      const auto t0 = std::chrono::steady_clock::now();
      s.applyNodes(f, changes.nodes);
      s.dialTiming.add(DialTiming::eApplyNodes, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    if(!changes.meshes.empty())
    {
      const auto t0 = std::chrono::steady_clock::now();
      for(const fv::morph::MeshChange& mesh : changes.meshes)
        s.applyMesh(f, mesh);
      s.dialTiming.add(DialTiming::eApplyMesh, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    if(!changes.rebound.empty())
    {
      const auto t0 = std::chrono::steady_clock::now();
      s.applyRebind(f, changes.rebound);
      s.dialTiming.add(DialTiming::eApplyRebind, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    // the tangents of a mesh still for a moment: the whole body's take 15-20 ms, too much for
    // every step of a dial drag
    if(f.feed)
      f.feed->settleTangents(std::chrono::milliseconds(100));
    for(const std::string& w : f.fig->takeWarnings())
      s.notice(p->name + ": " + w);
    s.publishBusy(f);
    s.publishMorphs(f);
  }
  timePhase("frame: the rest");
  if(fv::timing::enabled())
    if(const std::string line = s.dialTiming.report(std::chrono::steady_clock::now()); !line.empty())
    {
      LOGI("%s\n", line.c_str());
      for(const std::string& phase : fv::timing::takeSummary())
        LOGI("%s\n", phase.c_str());
    }
  // props the gizmo moved: one patch a frame
  if(!s.movedProps.empty())
  {
    Json moved = Json::object();
    for(const std::string& id : s.movedProps)
      if(const auto p = s.findProp(id))
      {
        const Json now = s.propJson(*p);
        moved[id]      = {{"transform", now["transform"]}};
        if(now.contains("camera"))
          moved[id]["camera"] = now["camera"];
      }
    s.movedProps.clear();
    s.protocol->events().patch("scene", {{"props", moved}});
  }
  s.publishAnimation(false);
  // the first thing in an empty scene, framed once it is drawn
  if(s.frameIn > 0 && --s.frameIn == 0)
    s.frameCamera();
  s.drawHelpers();
  if(s.view)
    s.view->onFrame();
  phase.reset();
  s.lastUpdateEnd = std::chrono::steady_clock::now();
}
