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
// FvHost: loading a figure or a library prop, and installing it in the scene.
//

#include "fv_host_impl.hpp"

Json FvHost::Impl::load(const Json& args)
{
  HostSettings s;
  {
    std::lock_guard lock(settingsMutex);
    s = settings;
  }
  if(args.contains("path"))
  {
    std::string ext = pathOf(args["path"].get<std::string>()).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if(ext == ".gltf" || ext == ".glb" || ext == ".obj")
      return importModel(args["path"].get<std::string>());
  }
  // an absolute path, or a path in the content folders (a library item)
  std::optional<std::pair<std::vector<std::filesystem::path>, std::string>> library;
  std::string                                                              name;
  if(args.contains("rel"))
  {
    const std::string rel = args["rel"].get<std::string>();
    library               = std::make_pair(s.roots, rel);
    name                  = rel;
  }
  else
  {
    const std::string path = args.at("path").get<std::string>();
    library                = libraryFor(s, pathOf(path));
    if(!library)
      throw std::runtime_error(path + " is not in a DAZ content library (no data folder above it)");
    name = path;
  }
  // how it was opened, for a scene file
  const Json source = args.contains("rel") ? Json{{"rel", args["rel"]}} : Json{{"path", args["path"]}};
  // a wearable goes on the figure named or selected, unless it is to be a prop of its own
  const bool asProp = args.value("as", "") == "prop" || args.value("replace", false);
  std::string target = args.contains("prop") && args["prop"].is_string() ? args["prop"].get<std::string>() : "";
  std::vector<std::string> prefetch, iray;
  for(auto [key, into] : {std::pair{"prefetch", &prefetch}, std::pair{"iray", &iray}})
    if(args.contains(key) && args[key].is_array())
      for(const Json& r : args[key])
        if(r.is_string())
          into->push_back(r.get<std::string>());
  return loadFigure(s, library->first, library->second, fileStem(name), source, target, asProp, {}, std::move(prefetch), std::move(iray));
}

Json FvHost::Impl::loadFigure(const HostSettings& s, std::vector<std::filesystem::path> roots, std::string rel, const std::string& name, Json source,
                              std::string target, bool asProp, std::function<void(std::shared_ptr<Prop>)> then, std::vector<std::string> prefetch,
                              std::vector<std::string> iray, std::string id)
{
  const uint64_t    jobId      = ++jobs;
  const uint64_t    generation = loadGeneration;
  const std::string label      = "Loading " + utf8Of(pathOf(rel).filename());
  job(jobId, {{"label", label}, {"state", "running"}});

  loaders.spawn([this, s, roots = std::move(roots), rel = std::move(rel), jobId, label, asProp, source = std::move(source), target = std::move(target),
                 generation, then = std::move(then), prefetch = std::move(prefetch), iray = std::move(iray), id = std::move(id)] {
    try
    {
      auto lib   = std::make_shared<fv::dson::Library>(roots);
      auto store = std::make_shared<fv::dson::Store>(lib);
      // and their meshes built, at the level the wears will ask for
      for(const std::string& wear : prefetch)
        fv::dson::prefetchScene(store, wear, s.subdivision);
      // a preset is applied to a figure in the scene
      auto doc = store->load(rel);
      if(doc)
        if(auto preset = fv::dson::describePreset(doc->json))
        {
          // an environment preset's HDR, found through the preset's own library
          std::string envPath;
          if(preset->environment)
            if(auto found = lib->find(normRel(preset->environment->image)))
              envPath = utf8Of(*found);
          // a shader preset that sets values goes on surfaces as a material preset does
          const std::string kind = preset->setsMaterials ? "material preset" : preset->kind;
          post([this, body = doc->json, kind, jobId, rel, env = preset->environment, envPath, target] {
            applyPreset(body, kind, jobId, rel, env, envPath, target);
          });
          return;
        }
      // clothing, hair and anatomy go on a figure in the scene, as in DAZ Studio
      if(!asProp && doc && fv::dson::isWearable(doc->json))
      {
        const std::string wearer = onApp([this, target] {
          const auto p = figureProp(target.empty() ? Json::object() : Json{{"prop", target}});
          return p && p->figure->fig->pose ? p->id : std::string();
        });
        if(!wearer.empty())
        {
          // done once it is on: the app waits on this job to lay a colour preset over the piece,
          // which finds none of its surfaces while the wear is still loading
          const auto worn = onApp([this, wearer] { return figureArg(Json{{"prop", wearer}}); });
          WearRequest request{rel, "", {}, iray};
          request.step = true;
          startWear(worn, {std::move(request)}, {}, [this, jobId, label] { job(jobId, {{"label", label}, {"state", "done"}}); });
          return;
        }
        if(onApp([this] { return std::count_if(props.begin(), props.end(), [](const auto& p) { return p->figure && p->figure->fig->pose; }) > 1; }))
          throw std::runtime_error("Select the figure to put " + fileStem(rel) + " on.");
      }
      const auto t0 = std::chrono::steady_clock::now();
      auto       f = fv::Figure::load(store, rel, {.subdivision = s.subdivision, .cageMorphs = s.figureGpuRefine}, [this, jobId, label](const std::string& step) {
        job(jobId, {{"label", label + ": " + step}, {"state", "running"}});
      });
      if(generation != loadGeneration)
      {
        job(jobId, {{"label", label}, {"state", "done"}});
        return;  // the scene was cleared meanwhile
      }
      for(const auto& w : f->takeWarnings())
        notice(w);
      // centimetres to metres, after the skins were bound (as loaders/index.ts scales). A later
      // rebind reads bone and mesh worlds that both carry this scale, so it cancels back out.
      auto& root = f->loaded.scene.nodes[f->loaded.root];
      root.scale = root.scale * kCentimetre;

      // textures: every one the build asks for, resolved in parallel. With the cooker the figure is
      // built for the cooked images: the ones the cache has, and until the rest are cooked their
      // fallbacks (cookInPlace swaps each in as it lands). Original textures, or no cooker, builds it
      // for the sources, cutouts composed here.
      // the path tracer gets the skin's volume; the rasterizer would show it see-through
      const fv::mat::SkinMode skin = skinMode.load();
      job(jobId, {{"label", label + ": Textures"}, {"state", "running"}});
      const bool cooked = !s.texServer.empty() && !s.rootId.empty() && !s.originalTextures;
      std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images;
      std::vector<fv::build::TextureRequest>                                  uncooked;
      {
        fv::timing::Scope timeTextures("textures");
        const auto        requests = fv::build::textureRequests(f->loaded.scene, skin);
        const auto        found    = resolveTextures(s, lib.get(), requests, cooked, uncooked);
        prefetchImages(requests, found);
        for(size_t i = 0; i < requests.size(); i++)
          images[fv::build::requestKey(requests[i])] = found[i];
      }
      // a strand hair's direction is measured here, as a wear measures its own, so the feed on the
      // application thread finds it (the feed reads the scene graph: no glTF is built)
      {
        fv::timing::Scope t("strand directions");
        fv::build::measureStrands(f->loaded.scene, cooked);
      }
      LOGI("[fv] %s loaded in %.2f s: %zu nodes, %zu meshes, %zu materials, %zu textures\n", rel.c_str(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), f->loaded.scene.nodes.size(),
           f->loaded.scene.meshes.size(), f->loaded.scene.materials.size(), images.size());
      auto       shared  = std::make_shared<std::unique_ptr<fv::Figure>>(std::move(f));
      const bool flipped = cooked;
      post([this, shared, generation, jobId, flipped, skin, lib, images = std::move(images), uncooked = std::move(uncooked), source, then,
            id]() mutable {
        install(std::move(*shared), generation, jobId, flipped, skin, lib, std::move(images), std::move(uncooked), std::move(source), then, id);
      });
    }
    catch(const std::exception& e)
    {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", e.what()}});
      notice(e.what(), "error");
    }
  });
  return {{"job", jobId}};
}

void FvHost::Impl::cookInPlace(const std::shared_ptr<Loaded>& f, std::vector<fv::build::TextureRequest> requests)
{
  HostSettings s;
  {
    std::lock_guard lock(settingsMutex);
    s = settings;
  }
  // a worker the detach joins: a cook it waits on is stopped then, and what lands after is dropped
  loaders.spawn([this, s, requests = std::move(requests), weak = std::weak_ptr<Loaded>(f)]() mutable {
    cookAll(s, std::move(requests), [this, weak](const fv::build::TextureRequest& req, fv::build::TextureImage image) {
      const auto f = weak.lock();
      if(!f)
        return;
      {
        std::lock_guard lock(f->landedMutex);
        f->landed.emplace_back(req, std::move(image));
      }
      // several that land within a frame are taken by the first of these
      post([this, weak] {
        if(const auto f = weak.lock(); f && inScene(f))
          takeLanded(*f);
      });
    });
  });
}

void FvHost::Impl::takeLanded(Loaded& f)
{
  std::vector<std::pair<fv::build::TextureRequest, fv::build::TextureImage>> landed;
  {
    std::lock_guard lock(f.landedMutex);
    landed.swap(f.landed);
  }
  if(landed.empty())
    return;
  fv::timing::Scope                      t("cooks landed");
  std::vector<fv::build::TextureRequest> requests;
  for(auto& [req, image] : landed)
  {
    f.images.insert_or_assign(fv::build::requestKey(req), std::move(image));
    requests.push_back(req);
  }
  // without a feed yet, the composition that makes it lowers with these; a request it never fed
  // (its fallback had nothing) comes with the lowering below
  if(!f.feed)
    return;
  f.feed->retexture(requests);
  const std::shared_ptr<Prop> p = findProp(f.id);
  if(p && p->materials)
    p->materials->rewrite();
  else
  {
    std::vector<int> all(f.fig->loaded.scene.materials.size());
    std::iota(all.begin(), all.end(), 0);
    if(!f.feed->syncMaterials(f.fig->loaded.scene, all))
      f.feed->sync(f.fig->loaded.scene, f.fig->loaded.root);
  }
}

void FvHost::Impl::install(std::unique_ptr<fv::Figure>                                             fig,
                           uint64_t                                                                generation,
                           uint64_t                                                                jobId,
                           bool                                                                    texturesFlipped,
                           fv::mat::SkinMode                                                       skin,
                           std::shared_ptr<fv::dson::Library>                                      lib,
                           std::unordered_map<std::string, std::optional<fv::build::TextureImage>> images,
                           std::vector<fv::build::TextureRequest>                                  uncooked,
                           Json                                                                    source,
                           std::function<void(std::shared_ptr<Prop>)>                              then,
                           std::string                                                             id)
{
  // the scene was cleared meanwhile: this one is dropped, and its job ends
  if(generation != loadGeneration)
  {
    job(jobId, {{"state", "done"}});
    return;
  }
  if(const std::string refused = renderer->refuseGeometry(geometryBytes(fig->loaded.scene)); !refused.empty())
  {
    job(jobId, {{"state", "failed"}, {"error", refused}});
    notice(fileStem(fig->loaded.name) + " wasn't loaded: " + refused, "error");
    return;
  }
  auto f             = std::make_shared<Loaded>();
  f->fig             = std::move(fig);
  f->images          = std::move(images);
  f->source          = std::move(source);
  f->texturesFlipped = texturesFlipped;
  f->lib             = std::move(lib);
  if(!uncooked.empty())
    cookInPlace(f, std::move(uncooked));
  // its add-ons go on once it is in the scene; until then the eyes it will have aren't there to choose
  f->addonsPending = !f->fig->addons.empty() && f->fig->pose;
  // "Mirel.duf" reads as "Mirel"
  auto p    = addProp(PropKind::eFigure, fileStem(f->fig->loaded.name), id);
  f->id     = p->id;
  p->figure = f;
  if(f->fig->morphs && !f->fig->loaded.morphs.mainMeshes.empty())
  {
    const int body = f->fig->loaded.morphs.meshes[f->fig->loaded.morphs.mainMeshes[0]].mesh;
    f->dforce      = std::make_unique<fv::dforce::Dforce>(f->fig->loaded.scene, *f->fig->morphs, body, f->fig->loaded.root);
    f->dforce->playing = [this] { return renderer->editAnimationControl().play; };
#ifdef FV_DFORCE_GPU
    // the experimental GPU solver, for mode "gpu": made and stepped on the application thread
    f->dforce->makeGpuSim = [this](std::vector<fv::dforce::SettleMesh> meshes, const fv::dforce::SettleOptions& opts)
        -> std::shared_ptr<fv::dforce::ClothBackend> {
      if(!app)
        return nullptr;
      auto sim = std::make_shared<DforceGpu>(renderer->deviceAllocator(), app, std::move(meshes), opts);
      if(!sim->valid())
      {
        notice("dForce: the GPU solver could not start; simulating on the CPU");
        return nullptr;
      }
      return sim;
    };
#endif
    // the figure's shown clothing collides in the simulation (a belt, a bra, shoes); its anatomy,
    // hair and props do not
    f->dforce->collidesWith = [fig = f->fig.get()](int mesh) {
      for(const fv::WornItem& w : fig->items)
        if(w.visible && w.kind == "Clothing" && std::find(w.meshes.begin(), w.meshes.end(), mesh) != w.meshes.end())
          return true;
      return false;
    };
    // off until the Wardrobe toggle (or a scene file) turns it on: a settle costs seconds on a dressed figure
    f->dforce->setEnabled(false);
    {
      std::lock_guard lock(settingsMutex);
      f->dforce->dazSkinning = settings.figureDazSkinning;
    }
    // on from the start: a pass is a fraction of a second on the pool, and it is what keeps a garment
    // fitted to another shape or generation out of the body
    f->collider = std::make_unique<fv::wardrobe::Collider>(f->fig->loaded.scene, *f->fig->morphs, body, f->fig->loaded.root);
    {
      std::lock_guard lock(settingsMutex);
      f->collider->dazSkinning = settings.figureDazSkinning;
    }
    f->collider->settings = [fig = f->fig.get()] {
      std::map<int, fv::dson::Smoothing> all;
      for(const fv::WornItem& w : fig->items)
        if(w.visible)
          for(auto [mesh, smoothing] : w.collision)
          {
            smoothing.tight = w.tight;
            all.emplace(mesh, smoothing);
          }
      return all;
    };
    f->collider->wornOrder = [fig = f->fig.get()] {
      std::vector<int> order;
      for(const fv::WornItem& w : fig->items)
        order.insert(order.end(), w.meshes.begin(), w.meshes.end());
      return order;
    };
    f->collider->owned = [raw = f.get()](int mesh) {
      const fv::Mesh& m = raw->fig->loaded.scene.meshes[size_t(mesh)];
      return raw->dforce && raw->dforce->enabled() && m.dforce && m.dforce->simulated();
    };
  }
  {
    std::lock_guard lock(morphMutex);
    morphFigures.insert(p->id);
  }
  requestCompose("Adding " + p->name, [this, p, f, jobId, then, restored = !id.empty()] {
    if(!inScene(f))
    {
      job(jobId, {{"state", "done"}});
      return;
    }
    job(jobId, {{"state", "done"}, {"prop", p->id}, {"figure", p->id}});
    selectProp(p->id);
    // Genesis 9 leaves its eyes, mouth and lashes to add-ons the scene file lists
    if(!f->fig->addons.empty() && f->fig->pose)
    {
      std::vector<WearRequest> requests;
      for(const fv::dson::PostLoadAddon& a : f->fig->addons)
        requests.push_back({a.file, a.name, a.presets});
      f->addonsPending = true;
      startWear(f, std::move(requests), {}, [this, f] {
        f->addonsPending = false;
        if(const auto p = findProp(f->id); p && p->figure == f)
        {
          publishEyes(*p);
          publishSkin(*p);
        }
        for(const auto& after : std::exchange(f->afterAddons, {}))
          after();
      });
    }
    if(then)
      then(p);
    // a load is a step of the history; a duplicate's or a restore's `then` says what it is
    else if(!restored)
      recordAdded(p, "Add " + p->name);
  });
}

// The library path the figure was opened from, a rel when it came from the library.
std::string figureSource(const Loaded& figure)
{
  if(!figure.source.is_object())
    return {};
  if(figure.source.contains("rel"))
    return figure.source.value("rel", "");
  if(figure.source.contains("path"))
    return figure.source.value("path", "");
  return {};
}

// "Genesis 9", plus " Female" or " Male" when the path says. Bones cover a file whose path never does.
std::string figureGeneration(const Loaded& figure)
{
  const std::string source = figureSource(figure);
  std::vector<std::string> bones;
  if(figure.fig && figure.fig->pose)
    for(const auto& pb : figure.fig->pose->bones())
      if(!pb.name.empty())
        bones.push_back(pb.name);
  const auto gen = fv::wardrobe::identifyGeneration(source, bones, {});
  if(!gen)
    return {};
  std::string label = fv::wardrobe::label(*gen);
  const std::string gender = fv::wardrobe::genderHint(source);
  if(!gender.empty())
  {
    label.push_back(' ');
    label.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(gender[0]))));
    label.append(gender, 1, std::string::npos);
  }
  return label;
}

// What the panels learn when a figure is composed into the scene: its pose and what it wears.
void FvHost::Impl::publishFigure(Prop& p)
{
  if(p.figure->fig->pose)
    protocol->events().snapshot("pose/" + p.id, poseSnapshot(*p.figure));
  publishWardrobe(*p.figure);
  publishDforce(*p.figure, true);
}
