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
// pose, hidden polygons, a hidden node and a taken-off one, and for DAZ instances.

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <set>

#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_rig.hpp>

#include "fv_figure_feed.hpp"
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
  std::vector<int> moved;
  for(const auto& c : changes.meshes)
    moved.push_back(c.mesh);
  (void)fed.takeChanges();
  feed.sync(f->loaded.scene, f->loaded.root, moved);
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
  EXPECT_NE(feed.meshOf(0), body);
  EXPECT_EQ(fed.mesh(body), nullptr);  // the old record went with its last instance
  EXPECT_EQ(fed.mesh(feed.meshOf(0))->submeshes[0].indexCount, 0u);
  {
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
  const rs::SceneChanges changes = fed.takeChanges();
  EXPECT_EQ(changes.materials, std::set<rs::MaterialId>{feed.materialOf(0)});
  rs::RetainedScene imported;
  importBuild(imported, f->loaded.scene, f->loaded.root, options);
  expectSame(fed, imported, false);
}

TEST(FigureFeed, TheFeedLetsGoOfEverythingItFed)
{
  const auto        f = loadBox();
  rs::RetainedScene fed;
  {
    fv::FigureFeed feed(fed, {.build = textured(), .rootScale = 0.01});
    feed.sync(f->loaded.scene, f->loaded.root);
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
  EXPECT_EQ(fed.textureCount(), 0u);
  const auto options = textured();
  feed.setOptions(options);
  (void)fed.takeChanges();
  feed.sync(f->loaded.scene, f->loaded.root);
  EXPECT_FALSE(fed.takeChanges().materials.empty());
  EXPECT_GT(feed.textures()->size(), 0u);
  rs::RetainedScene imported;
  importBuild(imported, f->loaded.scene, f->loaded.root, options);
  expectSame(fed, imported, false);
}
