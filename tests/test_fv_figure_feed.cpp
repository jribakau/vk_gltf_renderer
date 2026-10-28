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
#include <filesystem>
#include <functional>
#include <map>
#include <set>

#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_rig.hpp>

#include "fv_figure_feed.hpp"
#include "fv_figure_feed_check.hpp"
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

tinygltf::Material canonical(const rs::RetainedScene& scene, const rs::MaterialRecord& m, std::map<std::string, int>& names)
{
  const auto map = [&](int index) {
    if(index < 0 || !m.textures || index >= int(m.textures->size()))
      return -1;
    const rs::TextureRecord* t = scene.texture((*m.textures)[size_t(index)].texture);
    return names.try_emplace(t ? t->uri : "", int(names.size())).first->second;
  };
  tinygltf::Material out = m.payload;
  for(tinygltf::TextureInfo* t : {&out.pbrMetallicRoughness.baseColorTexture, &out.pbrMetallicRoughness.metallicRoughnessTexture, &out.emissiveTexture})
    if(t->index >= 0)
      t->index = map(t->index);
  if(out.normalTexture.index >= 0)
    out.normalTexture.index = map(out.normalTexture.index);
  if(out.occlusionTexture.index >= 0)
    out.occlusionTexture.index = map(out.occlusionTexture.index);
  for(auto& [name, ext] : out.extensions)
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
