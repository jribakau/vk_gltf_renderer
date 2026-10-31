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

// The figure feed against the build (daz-viewer docs/retained-scene-plan.md, oracle 4): a figure fed
// straight into the retained scene holds what importing its fv::build::buildGltf model holds, and
// the importer is held to nvvkgltf::Scene by test_retained_import. Checked after a load, a dial, a
// pose, hidden polygons, a hidden node and a taken-off one, a dial on a normal-mapped body, and for
// DAZ instances; fv::checkFeed holds the store to the scene's own bytes after each, and is shown to
// catch a fed record corrupted through the store.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>

#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_rig.hpp>

#include <figure/materials/material_gltf.hpp>

#include "fv_figure_feed.hpp"
#include "common/fake_host_renderer.hpp"
#include "fv_figure_feed_check.hpp"
#include "fv_events.hpp"
#include "fv_materials.hpp"
#include "scene_import_gltf.hpp"

namespace {

namespace fs = std::filesystem;

const fs::path kLibrary = fs::path(FV_TEST_DATA) / "daz-library";

// Every texture resolves to its source file, uncooked (the UVs flip).
fv::build::BuildOptions textured()
{
  fv::build::BuildOptions o;
  o.texture = [](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
    return fv::build::TextureImage{.uri = (kLibrary / req.rel).generic_string(), .mimeType = "image/png"};
  };
  return o;
}

std::unique_ptr<fv::Figure> loadBox(const std::string& rel = "People/Test/Box Figure.duf")
{
  fv::dson::LoadOptions options;
  options.subdivision = 1;
  auto f = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<fs::path>{kLibrary})),
                            rel, options);
  f->settle();
  f->drain();
  return f;
}

// A material's texture indices as the files they name, so two scenes whose texture lists run in
// different orders compare.
void canonical(tinygltf::Value& v, const std::function<int(int)>& map)
{
  if(v.IsObject())
  {
    auto& o = v.Get<tinygltf::Value::Object>();
    for(auto& [key, child] : o)
      if(key == "index" && child.IsInt())
        child = tinygltf::Value(map(child.Get<int>()));
      else
        canonical(child, map);
  }
  else if(v.IsArray())
    for(auto& child : v.Get<tinygltf::Value::Array>())
      canonical(child, map);
}

fv::mat::Material canonical(const rs::RetainedScene& scene, const rs::MaterialRecord& m, std::map<std::string, int>& names)
{
  const auto map = [&](int index) {
    if(index < 0 || !m.textures || index >= int(m.textures->size()))
      return -1;
    const rs::TextureRecord* t = scene.texture((*m.textures)[size_t(index)].texture);
    return names.try_emplace(t ? t->uri : "", int(names.size())).first->second;
  };
  fv::mat::Material out = m.payload;
  fv::mat::forEachSlot(out, [&](fv::mat::TextureSlot& slot) {
    if(slot.has())
      slot.index = map(slot.index);
  });
  for(auto& [name, ext] : out.foreignExtensions)
    canonical(ext, map);
  return out;
}

std::vector<std::string> jointNames(const rs::RetainedScene& scene, rs::SkinId id)
{
  std::vector<std::string> out;
  if(const rs::SkinRecord* s = scene.skin(id))
    for(rs::EntityId j : s->joints)
      out.push_back(scene.entity(j) ? scene.entity(j)->name : "<none>");
  return out;
}

// What the retained scenes draw, entity for entity and item for item. `sameTextureOrder`: the
// texture lists match index for index too (a first feed, made in the order the build makes them).
void expectSame(rs::RetainedScene& fed, rs::RetainedScene& imported, bool sameTextureOrder)
{
  const auto fedEntities = fed.entitiesInOrder();
  const auto impEntities = imported.entitiesInOrder();
  ASSERT_EQ(fedEntities.size(), impEntities.size());
  for(size_t i = 0; i < fedEntities.size(); i++)
  {
    const rs::Entity& a = *fed.entity(fedEntities[i]);
    const rs::Entity& b = *imported.entity(impEntities[i]);
    EXPECT_EQ(a.name, b.name) << i;
    EXPECT_TRUE(a.local == b.local) << a.name;
    EXPECT_EQ(a.visible, b.visible) << a.name;
    EXPECT_EQ(a.mesh.has_value(), b.mesh.has_value()) << a.name;
    if(a.mesh && b.mesh)
      EXPECT_EQ(jointNames(fed, a.mesh->skin), jointNames(imported, b.mesh->skin)) << a.name;
    if(a.mesh && b.mesh && fed.skin(a.mesh->skin) && imported.skin(b.mesh->skin))
      EXPECT_EQ(fed.skin(a.mesh->skin)->inverseBind, imported.skin(b.mesh->skin)->inverseBind) << a.name;
  }

  const auto fedItems = fed.drawItems();
  const auto impItems = imported.drawItems();
  ASSERT_EQ(fedItems.size(), impItems.size());
  std::map<std::string, int> names;
  for(size_t i = 0; i < fedItems.size(); i++)
  {
    const rs::DrawItem& a = fedItems[i];
    const rs::DrawItem& b = impItems[i];
    EXPECT_EQ(fed.entity(a.entity)->name, imported.entity(b.entity)->name) << i;
    EXPECT_EQ(a.world, b.world) << i;
    EXPECT_EQ(a.shown, b.shown) << i;
    const rs::MeshRecord& ma = *fed.mesh(a.mesh);
    const rs::MeshRecord& mb = *imported.mesh(b.mesh);
    EXPECT_EQ(ma.name, mb.name) << i;
    EXPECT_EQ(ma.positions, mb.positions) << i;
    EXPECT_EQ(ma.normals, mb.normals) << i;
    EXPECT_EQ(ma.uv0, mb.uv0) << i;
    EXPECT_EQ(ma.tangents, mb.tangents) << i;
    EXPECT_EQ(ma.joints, mb.joints) << i;
    EXPECT_EQ(ma.weights, mb.weights) << i;
    const rs::Submesh& sa = ma.submeshes[a.submesh];
    const rs::Submesh& sb = mb.submeshes[b.submesh];
    EXPECT_EQ(sa.order, sb.order) << i;
    EXPECT_EQ(std::vector<uint32_t>(ma.indices.begin() + sa.firstIndex, ma.indices.begin() + sa.firstIndex + sa.indexCount),
              std::vector<uint32_t>(mb.indices.begin() + sb.firstIndex, mb.indices.begin() + sb.firstIndex + sb.indexCount))
        << i;
    const rs::MaterialRecord& ra = *fed.material(a.material);
    const rs::MaterialRecord& rb = *imported.material(b.material);
    EXPECT_TRUE(canonical(fed, ra, names) == canonical(imported, rb, names)) << ra.payload.name;
    if(sameTextureOrder)
    {
      EXPECT_TRUE(ra.payload == rb.payload) << ra.payload.name;
      ASSERT_EQ(ra.textures->size(), rb.textures->size());
      for(size_t t = 0; t < ra.textures->size(); t++)
      {
        const rs::TextureRecord& ta = *fed.texture((*ra.textures)[t].texture);
        const rs::TextureRecord& tb = *imported.texture((*rb.textures)[t].texture);
        EXPECT_EQ(ta.uri, tb.uri);
        EXPECT_EQ(ta.name, tb.name);
        EXPECT_EQ(ta.mimeType, tb.mimeType);
        EXPECT_EQ(ta.sourceExtension, tb.sourceExtension);
        EXPECT_TRUE((*ra.textures)[t].sampler == (*rb.textures)[t].sampler);
      }
    }
  }
}

// The store holds the scene's own bytes.
void expectFed(const fv::SceneGraph& scene, int root, const fv::FigureFeed& feed, const rs::RetainedScene& fed,
               const fv::build::BuildOptions& options)
{
  EXPECT_EQ(fv::checkFeed(scene, root, feed, fed, options), std::vector<std::string>{});
}

// The figure as the build has it, imported.
void importBuild(rs::RetainedScene& into, const fv::SceneGraph& scene, int root, const fv::build::BuildOptions& options)
{
  rs::importGltf(into, fv::build::buildGltf(scene, root, options).model);
}

}  // namespace

TEST(FigureFeed, AFedFigureIsTheImportOfItsBuild)
{
  // (PBRSkin's detail layer is an extension of its own, which asks for no tangents)
  for(const std::string rel : {"People/Test/Box Figure.duf", "People/Test/Box Figure PBRSkin.duf"})
  {
    SCOPED_TRACE(rel);
    const auto        f       = loadBox(rel);
    const auto        options = textured();
    rs::RetainedScene fed, imported;
    fv::FigureFeed    feed(fed, {.build = options});
    feed.sync(f->loaded.scene, f->loaded.root);
    expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
    importBuild(imported, f->loaded.scene, f->loaded.root, options);
    ASSERT_GT(fed.drawItems().size(), 0u);
    ASSERT_GT(fed.textureCount(), 0u);
    expectSame(fed, imported, true);
    // and a sync with nothing changed changes nothing
    (void)fed.takeChanges();
    feed.sync(f->loaded.scene, f->loaded.root);
    EXPECT_TRUE(fed.takeChanges().empty());
  }
}

TEST(FigureFeed, ADialAPoseAndHiddenPolygonsAreFedAsTheBuildHasThem)
{
  const auto        f       = loadBox();
  const auto        options = textured();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);
  const rs::MeshId body = feed.meshOf(0);

  // a dial moves the vertices in place (and the head's joint centre, so the skin binds again)
  f->morphs->set("Stretch", 1);
  f->morphs->settled();
  f->pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", 30}});
  const fv::Figure::Changes changes = f->drain();
  ASSERT_FALSE(changes.meshes.empty());
  (void)fed.takeChanges();
  feed.sync(f->loaded.scene, f->loaded.root);
  expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
  const rs::SceneChanges after = fed.takeChanges();
  EXPECT_EQ(feed.meshOf(0), body);  // the same record, its vertices set again
  EXPECT_TRUE(after.meshes.count(body));
  EXPECT_FALSE(after.transforms.empty());
  {
    rs::RetainedScene imported;
    importBuild(imported, f->loaded.scene, f->loaded.root, options);
    expectSame(fed, imported, true);
  }

  // a graft hides the polygons of the body's first group: a new record with that group empty
  fv::mesh::BuiltMesh&  geo   = *f->loaded.scene.meshes[0].geo;
  const fv::mesh::Group first = geo.groups[0];
  std::set<int32_t>     polys;
  for(uint32_t t = first.start / 3; t < (first.start + first.count) / 3; t++)
    polys.insert(int32_t(geo.triPoly[t]));
  geo.hiddenPolys["w1"].assign(polys.begin(), polys.end());
  feed.sync(f->loaded.scene, f->loaded.root);
  expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
  EXPECT_NE(feed.meshOf(0), body);
  EXPECT_EQ(fed.mesh(body), nullptr);  // the old record went with its last instance
  EXPECT_EQ(fed.mesh(feed.meshOf(0))->submeshes[0].indexCount, 0u);
  {
    rs::RetainedScene imported;
    importBuild(imported, f->loaded.scene, f->loaded.root, options);
    expectSame(fed, imported, true);
  }
}

// A normal-mapped body under a dial: its tangents are made again only near the vertices that moved
// (tinygltf::utils::updateTangents), and they are the ones the build's import makes over all of it.
TEST(FigureFeed, ADialOnANormalMappedBodyRemakesItsTangentsAsTheBuildHasThem)
{
  const auto f = loadBox();
  for(fv::Material& m : f->loaded.scene.materials)
    m.surface.normalMap = "/Runtime/Textures/Test/BoxDetail_NM.png";
  const auto        options = textured();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);
  const rs::MeshId body = feed.meshOf(0);
  ASSERT_FALSE(fed.mesh(body)->tangents.empty());
  expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);

  for(const double value : {1.0, 0.5})
  {
    f->morphs->set("Stretch", value);
    f->morphs->settled();
    ASSERT_FALSE(f->drain().meshes.empty());
    feed.sync(f->loaded.scene, f->loaded.root);
    EXPECT_EQ(feed.meshOf(0), body);
    expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
    rs::RetainedScene imported;
    importBuild(imported, f->loaded.scene, f->loaded.root, options);
    expectSame(fed, imported, true);
  }
}

TEST(FigureFeed, AHiddenNodeAndATakenOffOneAreFedAsTheBuildHasThem)
{
  const auto        f       = loadBox();
  const auto        options = textured();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);

  // hide the first mesh node, take off the last node that has no children (a leaf of the tree)
  fv::SceneGraph& scene = f->loaded.scene;
  int             mesh = -1, leaf = -1;
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    if(mesh < 0 && scene.nodes[i].kind == fv::NodeKind::eMesh)
      mesh = int(i);
    if(int(i) != f->loaded.root && scene.nodes[i].children.empty() && scene.nodes[i].kind != fv::NodeKind::eMesh)
      leaf = int(i);
  }
  ASSERT_GE(mesh, 0);
  ASSERT_GE(leaf, 0);
  scene.nodes[size_t(mesh)].visible = false;
  scene.nodes[size_t(leaf)].removed = true;
  const rs::EntityId gone = feed.entityOf(leaf);
  feed.sync(scene, f->loaded.root);
  expectFed(scene, f->loaded.root, feed, fed, options);
  EXPECT_EQ(fed.entity(gone), nullptr);
  EXPECT_TRUE(feed.entityOf(leaf).isNone());
  rs::RetainedScene imported;
  importBuild(imported, scene, f->loaded.root, options);
  expectSame(fed, imported, true);
}

// DAZ Studio instances: one mesh drawn from several nodes is one record, as it is one glTF mesh. Its
// normal map asks for tangents, made once for the record as nvvkgltf::Scene makes them.
TEST(FigureFeed, InstancesShareOneMeshWithItsTangents)
{
  fv::mesh::MeshInput input;
  input.positions = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  input.polyStart = {0, 4};
  input.polyVerts = {0, 1, 2, 3};
  input.polyMaterial = {0};
  input.uvs       = {0, 0, 1, 0, 1, 1, 0, 1};
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{.name = "Room"});
  scene.materials.resize(1);
  scene.materials[0].name              = "Floor";
  scene.materials[0].surface.normalMap = "/Runtime/Textures/Test/BoxDetail_NM.png";
  fv::Mesh m;
  m.name      = "floor";
  m.geo       = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(input));
  m.materials = {0};
  scene.meshes.push_back(m);
  for(double x : {0.0, 100.0, 200.0})
  {
    fv::Node n{.name = "Floor " + std::to_string(int(x)), .kind = fv::NodeKind::eMesh, .mesh = 0};
    n.position.x = x;
    scene.add(n, root);
  }

  const auto        options = textured();
  rs::RetainedScene fed, imported;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(scene, root);
  expectFed(scene, root, feed, fed, options);
  importBuild(imported, scene, root, options);
  expectSame(fed, imported, true);
  EXPECT_EQ(fed.meshCount(), 1u);
  EXPECT_EQ(fed.mesh(feed.meshOf(0))->tangents.size(), 4u);
  EXPECT_EQ(fed.refs(feed.meshOf(0)), 4u);  // the feed's and three instances'
  EXPECT_EQ(fed.drawItems().size(), 3u);

  // a node added later (a wear) joins without the others fed again
  const rs::MeshId floor = feed.meshOf(0);
  fv::Node         more{.name = "Floor 300", .kind = fv::NodeKind::eMesh, .mesh = 0};
  more.position.x = 300;
  scene.add(more, root, 1);
  (void)fed.takeChanges();
  feed.sync(scene, root);
  expectFed(scene, root, feed, fed, options);
  EXPECT_TRUE(fed.takeChanges().meshes.empty());
  EXPECT_EQ(feed.meshOf(0), floor);
  rs::RetainedScene again;
  importBuild(again, scene, root, options);
  expectSame(fed, again, true);
}

TEST(FigureFeed, AMaterialEditSetsThatMaterialOnly)
{
  const auto        f       = loadBox();
  const auto        options = textured();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);
  (void)fed.takeChanges();
  f->loaded.scene.materials[0].edits.roughness = 0.25f;
  feed.sync(f->loaded.scene, f->loaded.root);
  expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
  const rs::SceneChanges changes = fed.takeChanges();
  EXPECT_EQ(changes.materials, std::set<rs::MaterialId>{feed.materialOf(0)});
  rs::RetainedScene imported;
  importBuild(imported, f->loaded.scene, f->loaded.root, options);
  expectSame(fed, imported, false);
}

// A mesh's groups whose materials lower the same, their names aside, are drawn and traced as one
// prim (one BLAS: strand hair's groups overlap through the whole lock), each group still its own
// submesh; an edit that sets one apart asks for a sync, which parts it again.
TEST(FigureFeed, GroupsWithTheSameMaterialAreDrawnAsOne)
{
  const auto      f       = loadBox();
  const auto      options = textured();
  fv::SceneGraph& scene   = f->loaded.scene;
  const int       root    = f->loaded.root;
  int             mesh    = -1;
  for(size_t i = 0; i < scene.meshes.size(); i++)
    if(scene.meshes[i].geo && scene.meshes[i].geo->groups.size() >= 2
       && (mesh < 0 || scene.meshes[i].geo->groups.size() > scene.meshes[size_t(mesh)].geo->groups.size()))
      mesh = int(i);
  ASSERT_GE(mesh, 0) << "the box has no mesh with two groups";
  const fv::Mesh&  m = scene.meshes[size_t(mesh)];
  std::vector<int> groupMaterial;
  for(const fv::mesh::Group& g : m.geo->groups)
    groupMaterial.push_back(m.materials[size_t(g.materialIndex)]);
  // every group's material the first's, but for its name
  for(int mi : groupMaterial)
  {
    const std::string name = scene.materials[size_t(mi)].name;
    scene.materials[size_t(mi)] = scene.materials[size_t(groupMaterial[0])];
    scene.materials[size_t(mi)].name = name;
  }

  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(scene, root);
  expectFed(scene, root, feed, fed, options);
  const rs::MeshRecord* r = fed.mesh(feed.meshOf(mesh));
  ASSERT_TRUE(r);
  ASSERT_EQ(r->submeshes.size(), m.geo->groups.size());
  // the groups drawing something, in index order: the first draws the others
  std::vector<uint32_t> drawing;
  for(uint32_t k = 0; k < r->submeshes.size(); k++)
    if(r->submeshes[k].indexCount > 0)
      drawing.push_back(k);
  std::sort(drawing.begin(), drawing.end(), [&](uint32_t a, uint32_t b) { return r->submeshes[a].firstIndex < r->submeshes[b].firstIndex; });
  ASSERT_GE(drawing.size(), 2u);
  const uint32_t head = drawing.front(), last = drawing.back();
  EXPECT_TRUE(r->drawsItself(head));
  for(size_t i = 1; i < drawing.size(); i++)
    EXPECT_EQ(r->submeshes[drawing[i]].mergedInto, head) << drawing[i];
  const rs::Submesh run = r->drawnRun(head);
  EXPECT_EQ(run.firstIndex, r->submeshes[head].firstIndex);
  EXPECT_EQ(run.firstIndex + run.indexCount, r->submeshes[last].firstIndex + r->submeshes[last].indexCount);
  // a triangle of the last group, as a pick on the merged prim names it
  EXPECT_EQ(r->submeshAt(head, (r->submeshes[last].firstIndex - run.firstIndex) / 3), last);
  size_t items = 0;
  for(const rs::DrawItem& item : fed.drawItems())
    items += item.mesh == feed.meshOf(mesh) ? 1 : 0;
  EXPECT_EQ(items, size_t(1));

  // an edit sets the last group's material apart: the narrow sync leaves it to a sync, which parts it
  const int edited = groupMaterial[last];
  ASSERT_NE(edited, groupMaterial[head]);
  scene.materials[size_t(edited)].edits.roughness = 0.25f;
  EXPECT_FALSE(feed.syncMaterials(scene, std::vector<int>{edited}));
  feed.sync(scene, root);
  expectFed(scene, root, feed, fed, options);
  r = fed.mesh(feed.meshOf(mesh));
  ASSERT_TRUE(r);
  EXPECT_TRUE(r->drawsItself(last));
  EXPECT_EQ(r->drawnRun(last).indexCount, r->submeshes[last].indexCount);
}

// The narrow syncs a host runs every frame leave the store as a sync would: a dial and a pose (the
// vertices, the node transforms and the rebind they bring), a hidden item, a material edit.
TEST(FigureFeed, NarrowSyncsLeaveTheStoreAsASyncWould)
{
  const auto f = loadBox();
  for(fv::Material& m : f->loaded.scene.materials)
    m.surface.normalMap = "/Runtime/Textures/Test/BoxDetail_NM.png";
  const auto        options = textured();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);
  const int  root = f->loaded.root;
  const auto same = [&] {
    expectFed(f->loaded.scene, root, feed, fed, options);
    rs::RetainedScene imported;
    importBuild(imported, f->loaded.scene, root, options);
    expectSame(fed, imported, false);
  };

  f->morphs->set("Stretch", 1);
  f->morphs->settled();
  f->pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", 30}});
  const fv::Figure::Changes changes = f->drain();
  ASSERT_FALSE(changes.meshes.empty());
  ASSERT_FALSE(changes.nodes.empty());
  std::vector<int> moved;
  for(const auto& c : changes.meshes)
    moved.push_back(c.mesh);
  const std::vector<int> nodes(changes.nodes.begin(), changes.nodes.end());
  (void)fed.takeChanges();
  EXPECT_TRUE(feed.syncVertices(f->loaded.scene, moved));
  EXPECT_TRUE(feed.syncTransforms(f->loaded.scene, root, nodes));
  EXPECT_TRUE(feed.syncSkins(f->loaded.scene, changes.rebound));
  const rs::SceneChanges narrow = fed.takeChanges();
  EXPECT_TRUE(narrow.materials.empty());  // nothing lowered
  EXPECT_FALSE(narrow.structure);
  same();

  // the root's own transform keeps the host's scale
  {
    rs::RetainedScene scaled;
    fv::FigureFeed    feedScaled(scaled, {.build = options, .rootScale = 0.01});
    feedScaled.sync(f->loaded.scene, root);
    scaled.setLocal(feedScaled.entityOf(root), {});
    EXPECT_TRUE(feedScaled.syncTransforms(f->loaded.scene, root, std::vector<int>{root}));
    EXPECT_FLOAT_EQ(scaled.entity(feedScaled.entityOf(root))->local.scale.x,
                    float(f->loaded.scene.nodes[size_t(root)].scale.x * 0.01));
  }

  // a node hidden
  const int hidden                              = f->loaded.scene.nodes[size_t(root)].children.front();
  f->loaded.scene.nodes[size_t(hidden)].visible = false;
  EXPECT_TRUE(feed.syncTransforms(f->loaded.scene, root, std::vector<int>{hidden}));
  same();

  // a factor edit: that material only
  f->loaded.scene.materials[0].edits.roughness = 0.25f;
  (void)fed.takeChanges();
  EXPECT_TRUE(feed.syncMaterials(f->loaded.scene, std::vector<int>{0}));
  EXPECT_EQ(fed.takeChanges().materials, std::set<rs::MaterialId>{feed.materialOf(0)});
  same();

  // the normal map taken away: the meshes using it carry no tangents now, which only a sync remakes
  f->loaded.scene.materials[0].surface.normalMap.clear();
  EXPECT_FALSE(feed.syncMaterials(f->loaded.scene, std::vector<int>{0}));
  feed.sync(f->loaded.scene, root);
  same();

  // options that changed are lowered by a sync first
  feed.setOptions(options);
  EXPECT_FALSE(feed.syncMaterials(f->loaded.scene, std::vector<int>{0}));
  feed.sync(f->loaded.scene, root);
  EXPECT_TRUE(feed.syncMaterials(f->loaded.scene, std::vector<int>{0}));

  // geometry the feed was not fed with (hidden polygons) is a sync's
  fv::mesh::BuiltMesh& geo = *f->loaded.scene.meshes[0].geo;
  geo.hiddenPolys["w1"]    = {int32_t(geo.triPoly[0])};
  EXPECT_FALSE(feed.syncVertices(f->loaded.scene, std::vector<int>{0}));
  feed.sync(f->loaded.scene, root);
  same();
}

// A drag's steps keep the tangents the body had (making them on every step halved a dial drag's
// frame rate). Once it has been still for the time asked, or a sync comes, they are the ones the
// build's import makes, and they are made once.
TEST(FigureFeed, ADragsTangentsAreMadeOnceItStops)
{
  const auto f = loadBox();
  for(fv::Material& m : f->loaded.scene.materials)
    m.surface.normalMap = "/Runtime/Textures/Test/BoxDetail_NM.png";
  const auto        options = textured();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);
  (void)f->drain();
  const rs::MeshId             body   = feed.meshOf(0);
  const std::vector<glm::vec4> before = fed.mesh(body)->tangents;
  const auto                   drag   = [&](double value) {
    f->morphs->set("Taper", value);  // sides that lean: their tangents turn
    f->morphs->settled();
    std::vector<int> moved;
    for(const auto& c : f->drain().meshes)
      moved.push_back(c.mesh);
    ASSERT_FALSE(moved.empty());
    EXPECT_TRUE(feed.syncVertices(f->loaded.scene, moved, true));
  };
  const auto expectImported = [&] {
    expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
    rs::RetainedScene imported;
    importBuild(imported, f->loaded.scene, f->loaded.root, options);
    expectSame(fed, imported, true);
  };

  drag(1.0);
  drag(0.5);
  EXPECT_EQ(fed.mesh(body)->tangents, before);
  feed.settleTangents(std::chrono::hours(1));
  EXPECT_EQ(fed.mesh(body)->tangents, before);
  feed.settleTangents();
  EXPECT_NE(fed.mesh(body)->tangents, before);
  expectImported();
  (void)fed.takeChanges();
  feed.settleTangents();
  EXPECT_TRUE(fed.takeChanges().meshes.empty());

  // a sync makes them too, though the vertices are as the record has them
  drag(0.0);
  feed.sync(f->loaded.scene, f->loaded.root);
  EXPECT_EQ(feed.meshOf(0), body);
  expectImported();
  (void)fed.takeChanges();
  feed.settleTangents();
  EXPECT_TRUE(fed.takeChanges().meshes.empty());
}

// Vertices written in place with no change to report (a wear's looks set a scene's restored dials)
// reach the meshes a sync keeps: it compares them with the scene's (the host once took a
// composition for a rebuild, and a restored expression never showed). The change that comes after
// moves nothing, and nor does a sync with nothing moved.
TEST(FigureFeed, ASyncFeedsTheVerticesOfTheMeshesItKeeps)
{
  const auto        f       = loadBox();
  const auto        options = textured();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);
  (void)f->drain();

  f->morphs->set("Stretch", 1);
  f->morphs->settled();
  const fv::Figure::Changes changes = f->drain();
  ASSERT_FALSE(changes.meshes.empty());
  std::vector<int> moved;
  for(const auto& c : changes.meshes)
    moved.push_back(c.mesh);

  (void)fed.takeChanges();
  const rs::MeshId body = feed.meshOf(0);
  feed.sync(f->loaded.scene, f->loaded.root);
  EXPECT_EQ(feed.meshOf(0), body);
  EXPECT_TRUE(fed.takeChanges().meshes.count(body));
  expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
  rs::RetainedScene imported;
  importBuild(imported, f->loaded.scene, f->loaded.root, options);
  expectSame(fed, imported, false);

  EXPECT_TRUE(feed.syncVertices(f->loaded.scene, moved));
  EXPECT_TRUE(fed.takeChanges().meshes.empty());
  feed.sync(f->loaded.scene, f->loaded.root);
  EXPECT_TRUE(fed.takeChanges().meshes.empty());
}

TEST(FigureFeed, TheFeedLetsGoOfEverythingItFed)
{
  const auto        f = loadBox();
  rs::RetainedScene fed;
  {
    fv::FigureFeed feed(fed, {.build = textured(), .rootScale = 0.01});
    feed.sync(f->loaded.scene, f->loaded.root);
    expectFed(f->loaded.scene, f->loaded.root, feed, fed, textured());
    EXPECT_GT(fed.entityCount(), 0u);
    const rs::Entity* root = fed.entity(feed.entityOf(f->loaded.root));
    ASSERT_NE(root, nullptr);
    EXPECT_FLOAT_EQ(root->local.scale.x, float(f->loaded.scene.nodes[size_t(f->loaded.root)].scale.x * 0.01));
  }
  EXPECT_EQ(fed.entityCount(), 0u);
  EXPECT_EQ(fed.meshCount(), 0u);
  EXPECT_EQ(fed.materialCount(), 0u);
  EXPECT_EQ(fed.textureCount(), 0u);
}

// A texture that comes later (a cook that finished) grows the figure's texture list; the materials
// that ask for it are set again with the longer list.
TEST(FigureFeed, ATextureThatComesLaterIsAddedToTheList)
{
  const auto        f = loadBox();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = {}});
  feed.sync(f->loaded.scene, f->loaded.root);
  expectFed(f->loaded.scene, f->loaded.root, feed, fed, {});
  EXPECT_EQ(fed.textureCount(), 0u);
  const auto options = textured();
  feed.setOptions(options);
  (void)fed.takeChanges();
  feed.sync(f->loaded.scene, f->loaded.root);
  expectFed(f->loaded.scene, f->loaded.root, feed, fed, options);
  EXPECT_FALSE(fed.takeChanges().materials.empty());
  EXPECT_GT(feed.textures()->size(), 0u);
  rs::RetainedScene imported;
  importBuild(imported, f->loaded.scene, f->loaded.root, options);
  expectSame(fed, imported, false);
}

// A texture whose file is in a folder with a space in its name: the record holds the file's path,
// which opens, not the URI it was given ("a%20folder").
namespace fvhost {
std::string uriOfPath(const std::string& path);
}

TEST(FigureFeed, ATexturesRecordHoldsItsFilesPathNotItsUri)
{
  // a '+' too, which a URI's decoder would take for a space
  const fs::path dir = fs::temp_directory_path() / "fv feed+textures";
  fs::create_directories(dir);
  fv::build::BuildOptions options;
  options.texture = [&](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
    const fs::path to = dir / fs::path(req.rel).filename();
    std::error_code ec;
    fs::copy_file(kLibrary / req.rel.substr(req.rel.find_first_not_of('/')), to, fs::copy_options::overwrite_existing, ec);
    return fv::build::TextureImage{.uri = fvhost::uriOfPath(to.generic_string()), .mimeType = "image/png"};
  };
  const auto        f = loadBox();
  rs::RetainedScene fed;
  fv::FigureFeed    feed(fed, {.build = options});
  feed.sync(f->loaded.scene, f->loaded.root);
  ASSERT_GT(feed.textures()->size(), 0u);
  for(const rs::TextureRef& ref : *feed.textures())
  {
    const rs::TextureRecord* record = fed.texture(ref.texture);
    ASSERT_TRUE(record);
    EXPECT_EQ(record->uri.find('%'), std::string::npos) << record->uri;
    EXPECT_TRUE(fs::is_regular_file(fs::path(std::u8string(record->uri.begin(), record->uri.end())))) << record->uri;
  }
  fs::remove_all(dir);
}

// A figure on a cold texture cache is built for the cooked images and shows each missing one's source
// until its cook lands: the source's slots sample it with V turned over, and the cook takes its
// place in every table that held it (the materials keep their indices), the turn goes, and the
// source's record leaves the store.
TEST(FigureFeed, ACookThatLandsTakesItsSourcesPlace)
{
  const auto                                     f = loadBox("People/Test/Box Figure PBRSkin.duf");
  const fv::SceneGraph&                          scene = f->loaded.scene;
  std::map<std::string, fv::build::TextureImage> cooked;  // by request key, once cooked
  fv::build::BuildOptions                        options;
  options.texturesFlipped = true;
  options.texture         = [&](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
    if(const auto it = cooked.find(fv::build::requestKey(req)); it != cooked.end())
      return it->second;
    return fv::build::TextureImage{.uri = (kLibrary / req.rel).generic_string(), .mimeType = "image/png"};
  };
  rs::RetainedScene store;
  fv::FigureFeed    feed(store, {.build = options});
  feed.sync(scene, f->loaded.root);
  const auto requests = fv::build::textureRequests(scene, options.skin);
  ASSERT_GE(requests.size(), 2u);
  const size_t slots = feed.textures()->size();
  ASSERT_GT(slots, 0u);
  std::vector<rs::TextureId> sources;
  for(const rs::TextureRef& ref : *feed.textures())
    sources.push_back(ref.texture);

  // per material slot that samples a texture: whether its image is a cook, and whether V turns over
  struct Sampled
  {
    bool cook = false, turned = false;
  };
  const auto sampled = [&] {
    std::vector<Sampled> out;
    for(size_t mi = 0; mi < scene.materials.size(); mi++)
    {
      const rs::MaterialRecord* r = store.material(feed.materialOf(int(mi)));
      if(!r)
        continue;
      const fv::mat::Material& m = r->payload;
      for(const fv::mat::TextureSlot* slot : {&m.baseColorMap, &m.metallicRoughnessMap, &m.emissiveMap, &m.normalMap})
        if(slot->has())
        {
          const rs::TextureRecord* t = store.texture((*r->textures)[size_t(slot->index)].texture);
          out.push_back({t && t->mimeType == "image/ktx2", slot->transform && slot->transform->scale[1] < 0});
        }
    }
    return out;
  };
  const auto expectTurnedWhereSource = [&](const std::string& when) {
    const std::vector<Sampled> all = sampled();
    ASSERT_FALSE(all.empty()) << when;
    for(const Sampled& x : all)
      EXPECT_EQ(x.turned, !x.cook) << when;
  };
  expectTurnedWhereSource("all sources");

  std::vector<int> all(scene.materials.size());
  std::iota(all.begin(), all.end(), 0);
  const auto land = [&](size_t from, size_t to) {
    for(size_t i = from; i < to; i++)
      cooked[fv::build::requestKey(requests[i])] = {.uri = "/cache/" + std::to_string(i) + ".ktx2", .mimeType = "image/ktx2", .flipped = true};
    EXPECT_TRUE(feed.retexture(std::span(requests).subspan(from, to - from)));
    EXPECT_TRUE(feed.syncMaterials(scene, all));
  };
  // one cook, then the rest
  land(0, 1);
  expectTurnedWhereSource("one cooked");
  land(1, requests.size());
  expectTurnedWhereSource("all cooked");
  for(const Sampled& x : sampled())
    EXPECT_TRUE(x.cook);

  // the materials kept their indices: the table is as long, and every source is gone from the store
  EXPECT_EQ(feed.textures()->size(), slots);
  for(rs::TextureId t : sources)
    EXPECT_EQ(store.texture(t), nullptr);
  for(const std::string& problem : fv::checkFeed(scene, f->loaded.root, feed, store, options))
    ADD_FAILURE() << problem;
  // a request it never fed is the next lowering's
  EXPECT_FALSE(feed.retexture(std::vector<fv::build::TextureRequest>{{.rel = "/Runtime/Textures/none.png"}}));
}

// checkFeed bites: a fed record corrupted through the store's edits is caught, each as its problem.
TEST(FigureFeed, CheckFeedCatchesARecordCorruptedInTheStore)
{
  const auto            f       = loadBox();
  const auto            options = textured();
  const fv::SceneGraph& scene   = f->loaded.scene;
  const int             root    = f->loaded.root;
  ASSERT_TRUE(scene.meshes[0].skinned());
  ASSERT_GT(scene.meshes[0].bones.size(), 1u);

  // the body's record replaced by an edited copy on every node drawing it (the store has no edit of
  // indices, UVs or skin weights in place)
  const auto replaced = [&](rs::RetainedScene& store, const fv::FigureFeed& feed, const std::function<void(rs::MeshRecord&)>& edit) {
    rs::MeshRecord r = *store.mesh(feed.meshOf(0));
    edit(r);
    const rs::MeshId id = store.addMesh(std::move(r));
    for(size_t i = 0; i < scene.nodes.size(); i++)
      if(const rs::Entity* e = store.entity(feed.entityOf(int(i))); e && e->mesh && e->mesh->parts[0].mesh == feed.meshOf(0))
      {
        rs::MeshInstance instance = *e->mesh;
        instance.parts[0].mesh    = id;
        store.setMeshInstance(feed.entityOf(int(i)), std::move(instance));
      }
    store.release(id);
  };
  // the first node drawing the body: it holds the skin
  int body = -1;
  for(size_t i = 0; i < scene.nodes.size() && body < 0; i++)
    if(scene.nodes[i].kind == fv::NodeKind::eMesh && scene.nodes[i].mesh == 0)
      body = int(i);
  ASSERT_GE(body, 0);

  const auto caught = [&](const std::vector<std::string>& want, const std::function<void(rs::RetainedScene&, const fv::FigureFeed&)>& corrupt) {
    rs::RetainedScene store;
    fv::FigureFeed    feed(store, {.build = options});
    feed.sync(scene, root);
    ASSERT_EQ(fv::checkFeed(scene, root, feed, store, options), std::vector<std::string>{});
    corrupt(store, feed);
    const std::vector<std::string> problems = fv::checkFeed(scene, root, feed, store, options);
    for(const std::string& w : want)
      EXPECT_NE(std::find(problems.begin(), problems.end(), w), problems.end()) << w;
  };

  caught({"mesh 0 positions", "mesh 0 bounds"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    std::vector<glm::vec3> positions = store.mesh(feed.meshOf(0))->positions;
    positions[0].x += 1000;
    store.setMeshVertices(feed.meshOf(0), std::move(positions), {});
  });
  caught({"mesh 0 indices 0"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    replaced(store, feed, [](rs::MeshRecord& r) { std::swap(r.indices[r.submeshes[0].firstIndex], r.indices[r.submeshes[0].firstIndex + 1]); });
  });
  caught({"mesh 0 UVs"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    replaced(store, feed, [](rs::MeshRecord& r) { r.uv0[0].y += 0.5f; });
  });
  caught({"mesh 0 joints"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    replaced(store, feed, [](rs::MeshRecord& r) { r.joints[0].x += 1; });
  });
  caught({"mesh 0 weights"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    replaced(store, feed, [](rs::MeshRecord& r) { r.weights[0].x += 0.25f; });
  });
  // a group hidden whole that the scene still draws
  caught({"mesh 0 indices 0"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    replaced(store, feed, [](rs::MeshRecord& r) { r.submeshes[0].indexCount = 0; });
  });
  caught({"mesh 0 inverse binds"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    rs::SkinRecord skin = *store.skin(feed.skinOf(0));
    skin.inverseBind[0][3][0] += 1;
    store.setSkin(feed.skinOf(0), std::move(skin));
  });
  caught({"mesh 0 joint entities"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    rs::SkinRecord skin = *store.skin(feed.skinOf(0));
    std::swap(skin.joints[0], skin.joints[1]);
    store.setSkin(feed.skinOf(0), std::move(skin));
  });
  // the body's first submesh drawn with a material that is no scene material's
  caught({"mesh 0 material 0"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    rs::MeshInstance       instance = *store.entity(feed.entityOf(body))->mesh;
    const rs::MaterialId   other    = store.addMaterial({.owner = {0, 1000}});
    instance.parts[0].materials[store.mesh(feed.meshOf(0))->submeshes[0].material] = other;
    store.setMeshInstance(feed.entityOf(body), std::move(instance));
    store.release(other);
  });
  // an entity under the figure that no node reached has
  caught({"entity stray is no reached node's"}, [&](rs::RetainedScene& store, const fv::FigureFeed& feed) {
    store.createEntity("stray", {0, int32_t(scene.nodes.size())}, feed.entityOf(root));
  });

  // and the UVs are held to the flip the options ask for
  rs::RetainedScene store;
  fv::FigureFeed    feed(store, {.build = options});
  feed.sync(scene, root);
  fv::build::BuildOptions flipped = options;
  flipped.texturesFlipped         = true;
  const std::vector<std::string> problems = fv::checkFeed(scene, root, feed, store, flipped);
  EXPECT_NE(std::find(problems.begin(), problems.end(), "mesh 0 UVs"), problems.end());
}

//--------------------------------------------------------------------------------------------------
// A skin tone through a full sync: the feed and the host's materials together, as the host joins them,
// over the box figure with a skin map on each of its two surfaces. A bake is a texture hook held behind
// `gate`; what's known without one is every source image.
namespace {

struct TonedBox
{
  TonedBox()
  {
    for(size_t mi = 0; mi < scene.materials.size(); mi++)
    {
      fv::Material& m = scene.materials[mi];
      if(m.name == "Body" || m.name == "Head")
      {
        (m.name == "Body" ? body : head) = int(mi);
        m.surface.kind                   = fv::mat::Kind::eSkin;
        m.surface.base.map               = "/Runtime/Textures/Test/" + m.name + "_d.png";
      }
    }
    EXPECT_GE(body, 0);
    EXPECT_GE(head, 0);
    options.texture = [this](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
      if(gone.count(req.rel))
        return std::nullopt;
      if(!req.recolor)
        return fv::build::TextureImage{.uri = "/source" + req.rel, .mimeType = "image/png"};
      const auto it = images.find(fv::build::requestKey(req));
      return it == images.end() ? std::nullopt : std::optional(it->second);
    };
    feed = std::make_unique<fv::FigureFeed>(store, fv::FigureFeedOptions{.build = options});
    materials = std::make_unique<FvMaterials>(FvMaterials::Hooks{
        .renderer = &renderer,
        .events = &events,
        .post =
            [this](std::function<void()> run) {
              std::lock_guard lock(mutex);
              posted.push_back(std::move(run));
            },
        .notice  = [this](const std::string&, const char*) { notices++; },
        .nextJob = [this] { return uint64_t(++asked); },
        .job     = [](uint64_t, const fv::Json&) {},
        .texture = [this](const fv::build::TextureRequest& req, bool) -> std::optional<fv::build::TextureImage> {
          gate.wait();
          cooks++;
          if(failing)
            return std::nullopt;
          return fv::build::TextureImage{.uri = "/baked" + req.rel + "?" + req.recolor->reference, .mimeType = "image/png"};
        },
        .remember = [this](const fv::build::TextureRequest& req, const fv::build::TextureImage& image) { images[fv::build::requestKey(req)] = image; },
        .resolves = [this](const fv::build::TextureRequest& req) { return options.texture(req).has_value(); },
        .writeFigure =
            [this](std::vector<std::pair<int, fv::mat::LowerOptions>> written) {
              if(!feed->syncMaterials(scene, written))
                feed->sync(scene, f->loaded.root);
            },
    });
    open();
  }

  void attach()
  {
    feed->sync(scene, f->loaded.root);
    materials->attach("n:Box", &scene, feed->lowering(), false, options.skin, feed->takeKept());
  }

  // the figure fed whole, as a composition after a wear or a take-off feeds it
  void compose()
  {
    feed->sync(scene, f->loaded.root);
    materials->extend(feed->lowering(), feed->takeKept());
  }

  void tone(const char* color = "#aa7755")
  {
    materials->set({{"material", {"m:" + std::to_string(body), "m:" + std::to_string(head)}}, {"values", {{"tone", color}}}});
  }

  // what's been posted back, once a bake has
  void landed()
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for(;;)
    {
      std::function<void()> run;
      {
        std::lock_guard lock(mutex);
        if(!posted.empty())
        {
          run = std::move(posted.front());
          posted.pop_front();
        }
      }
      if(run)
      {
        run();
        return;
      }
      ASSERT_LT(std::chrono::steady_clock::now(), end) << "no bake landed";
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // the image a surface's base map samples; "" without one
  std::string base(int material) const
  {
    const rs::MaterialRecord* r = store.material(feed->materialOf(material));
    if(!r || !r->payload.baseColorMap.has())
      return "";
    const rs::TextureRecord* t = store.texture((*r->textures)[size_t(r->payload.baseColorMap.index)].texture);
    return t ? t->uri : "";
  }

  void open() { gate = opened.get_future().share(), opened.set_value(); }
  void close() { opened = std::promise<void>(), gate = opened.get_future().share(); }

  std::unique_ptr<fv::Figure> f     = loadBox();
  fv::SceneGraph&             scene = f->loaded.scene;
  int                         body = -1, head = -1;

  fv::build::BuildOptions                        options;
  std::map<std::string, fv::build::TextureImage> images;  // by request key: what a bake made
  std::set<std::string>                          gone;    // rels whose files aren't there
  std::atomic<int>                               asked{0};   // the bakes started
  std::atomic<int>                               cooks{0};   // the textures they made
  std::atomic<int>                               notices{0};
  std::atomic<bool>                              failing{false};
  std::promise<void>                             opened;
  std::shared_future<void>                       gate;
  std::mutex                                     mutex;
  std::deque<std::function<void()>>              posted;

  FakeHostRenderer                renderer;
  rs::RetainedScene               store;
  fv::EventHub                    events;
  std::unique_ptr<fv::FigureFeed> feed;
  std::unique_ptr<FvMaterials>    materials;
};

const std::string kBodyMap = "/Runtime/Textures/Test/Body_d.png", kHeadMap = "/Runtime/Textures/Test/Head_d.png";

}  // namespace

// Toned, then fed whole (a wear, a take-off, a geograft): every skin surface keeps its toned map, with
// no bake asked again, and the key is the one a scene's restore cooks ahead
TEST(FigureFeedTone, AToneKeepsEverySurfacesMapThroughAFullSync)
{
  TonedBox b;
  b.attach();
  b.tone();
  EXPECT_EQ(b.asked, 1);
  b.landed();
  const std::string body = b.base(b.body), head = b.base(b.head);
  EXPECT_EQ(body, "/baked" + kBodyMap + "?" + kBodyMap);
  EXPECT_EQ(head, "/baked" + kHeadMap + "?" + kBodyMap);

  b.compose();
  EXPECT_EQ(b.base(b.body), body);
  EXPECT_EQ(b.base(b.head), head);
  EXPECT_EQ(b.feed->lowering()[size_t(b.head)].toneReference, kBodyMap);
  EXPECT_EQ(b.asked, 1);
  EXPECT_TRUE(b.feed->takeKept().empty());

  // again, and through a geometry fallback that isn't followed by a composition
  b.feed->sync(b.scene, b.f->loaded.root);
  EXPECT_EQ(b.base(b.head), head);
  EXPECT_TRUE(b.feed->takeKept().empty());

  for(const fv::mat::TextureRequest& r : fv::build::textureRequests(b.scene, b.options.skin))
    if(r.recolor)
      EXPECT_TRUE(b.images.count(fv::build::requestKey(r))) << r.rel;
}

// A file with a tone in it: its restore cooks what the feed then asks for, and every surface tones
// against the figure's reference, not each against its own mean
TEST(FigureFeedTone, ARestoredToneTonesAgainstTheFiguresReference)
{
  TonedBox b;
  for(int mi : {b.body, b.head})
    b.scene.materials[size_t(mi)].edits.tone = fv::mat::Rgb{0.67f, 0.47f, 0.33f};
  int toned = 0;
  for(const fv::mat::TextureRequest& r : fv::build::textureRequests(b.scene, b.options.skin))
    if(r.recolor)
    {
      toned++;
      EXPECT_EQ(r.recolor->reference, kBodyMap) << r.rel;
      b.images[fv::build::requestKey(r)] = {.uri = "/cooked" + r.rel, .mimeType = "image/png"};
    }
  EXPECT_GE(toned, 2);
  b.attach();
  EXPECT_EQ(b.base(b.body), "/cooked" + kBodyMap);
  EXPECT_EQ(b.base(b.head), "/cooked" + kHeadMap);
  EXPECT_EQ(b.asked, 0);
}

// Fed whole while its bake is in flight: the surface goes on showing the map it has, and the bake's
// map takes over when it lands
TEST(FigureFeedTone, ABakeInFlightDoesNotTakeTheMapAway)
{
  TonedBox b;
  b.attach();
  const std::string before = b.base(b.head);
  EXPECT_EQ(before, "/source" + kHeadMap);
  b.close();
  b.tone();
  EXPECT_EQ(b.asked, 1);

  b.compose();
  EXPECT_EQ(b.base(b.body), "/source" + kBodyMap);
  EXPECT_EQ(b.base(b.head), before);
  EXPECT_EQ(b.asked, 1);  // the one in flight

  b.opened.set_value();
  b.landed();
  EXPECT_EQ(b.base(b.body), "/baked" + kBodyMap + "?" + kBodyMap);
  EXPECT_EQ(b.base(b.head), "/baked" + kHeadMap + "?" + kBodyMap);
}

// A key nothing has asked for yet (the reference itself changed under the tone): the surfaces keep
// what they show, are asked for, and get it. Nothing is left recorded that isn't on screen.
TEST(FigureFeedTone, AKeyNothingAskedForIsAskedFor)
{
  TonedBox b;
  b.attach();
  b.tone();
  b.landed();
  const std::string head = b.base(b.head);

  const std::string tanned = "/Runtime/Textures/Test/Tan_d.png";
  b.scene.materials[size_t(b.body)].surface.base.map = tanned;
  b.compose();
  EXPECT_EQ(b.base(b.head), head);
  EXPECT_EQ(b.asked, 2);
  b.landed();
  EXPECT_EQ(b.base(b.body), "/baked" + tanned + "?" + tanned);
  EXPECT_EQ(b.base(b.head), "/baked" + kHeadMap + "?" + tanned);
  EXPECT_TRUE(b.feed->takeKept().empty());
}

// A map that was never there holds nothing back: the rest of the material still changes, and nothing
// asks for it over and over
TEST(FigureFeedTone, AMissingFileDoesNotFreezeAMaterial)
{
  TonedBox b;
  b.gone.insert(kHeadMap);
  b.attach();
  EXPECT_EQ(b.base(b.head), "");
  const fv::mat::Material before = b.store.material(b.feed->materialOf(b.head))->payload;

  b.scene.materials[size_t(b.head)].surface.opacity.value = 0.5f;
  b.compose();
  EXPECT_FALSE(b.store.material(b.feed->materialOf(b.head))->payload == before);
  EXPECT_EQ(b.asked, 0);
}

// A bake that fails is written as it is, flat, once: the next full sync finds no map to keep, and asks
// for nothing
TEST(FigureFeedTone, ABakeThatFailsIsNotAskedAgain)
{
  TonedBox b;
  b.attach();
  b.failing = true;
  b.tone();
  b.landed();
  EXPECT_EQ(b.base(b.head), "");
  b.compose();
  b.compose();
  EXPECT_EQ(b.base(b.head), "");
  EXPECT_EQ(b.asked, 1);
}

// Textures switched off leave nothing on its way: the maps go, and the materials go on taking edits
TEST(FigureFeedTone, NoTexturesAtAllKeepsNothing)
{
  TonedBox b;
  b.attach();
  EXPECT_EQ(b.base(b.head), "/source" + kHeadMap);
  b.feed->setOptions({});
  b.feed->sync(b.scene, b.f->loaded.root);
  EXPECT_EQ(b.base(b.head), "");
  EXPECT_TRUE(b.feed->takeKept().empty());
  b.scene.materials[size_t(b.head)].edits.roughness = 0.35f;
  b.feed->sync(b.scene, b.f->loaded.root);
  EXPECT_FLOAT_EQ(b.store.material(b.feed->materialOf(b.head))->payload.roughness, 0.35f);
}

// A drag over a baked value: the bake it started with runs out, since a cook can't be stopped, and the
// one it ended on follows. Nothing between is made, and the surfaces end on the last.
TEST(FvMaterialsBake, ADragMakesTheFirstAndTheLastNotEveryStep)
{
  TonedBox b;
  b.attach();
  b.close();
  b.tone("#100000");
  EXPECT_EQ(b.asked, 1);
  for(const char* step : {"#200000", "#300000", "#400000", "#500000"})
    b.tone(step);
  EXPECT_EQ(b.asked, 1);

  b.opened.set_value();
  b.landed();
  EXPECT_EQ(b.asked, 2);
  b.landed();
  EXPECT_EQ(b.asked, 2);

  // the first's first texture, already being made, and the last's two. The first's second never was.
  EXPECT_EQ(b.cooks, 3);
  int toned = 0;
  for(const fv::mat::TextureRequest& r : fv::build::textureRequests(b.scene, b.options.skin))
    if(r.recolor)
    {
      toned++;
      EXPECT_TRUE(b.images.count(fv::build::requestKey(r))) << r.rel;
    }
  EXPECT_EQ(toned, 2);
  EXPECT_EQ(b.images.size(), 3u);
  EXPECT_EQ(b.base(b.head), "/baked" + kHeadMap + "?" + kBodyMap);
  EXPECT_EQ(b.notices, 0);
}

// Back to the colour being made: the bake in flight is wanted again, not made twice
TEST(FvMaterialsBake, AColourDraggedBackTakesTheBakeInFlight)
{
  TonedBox b;
  b.attach();
  b.close();
  b.tone("#100000");
  b.tone("#200000");
  b.tone("#100000");
  b.opened.set_value();
  b.landed();
  EXPECT_EQ(b.asked, 1);
  EXPECT_EQ(b.cooks, 2);
  EXPECT_EQ(b.base(b.head), "/baked" + kHeadMap + "?" + kBodyMap);
}

// What nobody wants by the time it's made isn't decoded for a scene that won't take it
TEST(FvMaterialsBake, ABakeNobodyWantsIsNotDecoded)
{
  TonedBox b;
  b.attach();
  b.close();
  b.tone("#100000");
  b.tone("#200000");
  b.opened.set_value();
  b.landed();
  EXPECT_EQ(b.renderer.prefetches, 0);
  b.landed();
  EXPECT_EQ(b.renderer.prefetches, 2);
}

// An edit taken back while its bake runs leaves nothing behind: no map from it, nothing called failed
TEST(FvMaterialsBake, AnEditTakenBackLeavesNoBakeBehind)
{
  TonedBox b;
  b.attach();
  b.close();
  b.tone();
  b.materials->reset({{"material", {"m:" + std::to_string(b.body), "m:" + std::to_string(b.head)}}});
  b.opened.set_value();
  b.landed();
  EXPECT_EQ(b.base(b.head), "/source" + kHeadMap);
  EXPECT_EQ(b.notices, 0);
  EXPECT_EQ(b.asked, 1);
  EXPECT_EQ(b.renderer.prefetches, 0);

  // and the same tone asked again is made: dropped isn't failed
  b.tone();
  EXPECT_EQ(b.asked, 2);
  b.landed();
  EXPECT_EQ(b.base(b.head), "/baked" + kHeadMap + "?" + kBodyMap);
}
