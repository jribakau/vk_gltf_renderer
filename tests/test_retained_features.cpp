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

// The retained scene's shader features (oracle 5 of daz-viewer docs/retained-scene-plan.md):
// RetainedScene::features(), counted as instances and materials come, go and are edited, equals
// nvvkgltf::detectSceneFeatures of the scene's export after every step of seeded random edits, on
// synthetic materials covering every extension the detector knows and on imported fixtures.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>

#include <figure/materials/material_gltf.hpp>

#include "scene_export_gltf.hpp"
#include "scene_feature_detection.hpp"
#include "scene_import_gltf.hpp"
#include "tinygltf_utils.hpp"

namespace {

using nvvkgltf::SceneFeatureSet;

// What the scene derives: everything but eDlssGuide, which is the denoiser's, not the scene's.
SceneFeatureSet materialBits(SceneFeatureSet set)
{
  set.set(SceneFeatureSet::eDlssGuide, false);
  return set;
}

void expectFeaturesOfExport(const rs::RetainedScene& scene, const std::string& what)
{
  const SceneFeatureSet kept     = materialBits(scene.features());
  const SceneFeatureSet exported = materialBits(nvvkgltf::detectSceneFeatures(rs::exportGltf(scene)));
  EXPECT_TRUE(kept == exported) << what << ": kept {" << kept.toString() << "}, export {" << exported.toString() << "}";
}

tinygltf::Value object(tinygltf::Value::Object o = {})
{
  return tinygltf::Value(std::move(o));
}

// {"index": i, "extensions": {"KHR_texture_transform": {}}}
tinygltf::Value transformedInfo(int index)
{
  return object({{"index", tinygltf::Value(index)}, {"extensions", object({{KHR_TEXTURE_TRANSFORM_EXTENSION_NAME, object()}})}});
}

// Every material extension the detector knows, and the feature it sets
struct KnownExtension
{
  const char*              name;
  SceneFeatureSet::Feature feature;
};
const KnownExtension kExtensions[] = {
    {KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME, SceneFeatureSet::eTransmission},
    {KHR_MATERIALS_VOLUME_EXTENSION_NAME, SceneFeatureSet::eVolume},
    {KHR_MATERIALS_SCATTER_EXTENSION_NAME, SceneFeatureSet::eScatter},
    {KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME, SceneFeatureSet::eScatter},
    {KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME, SceneFeatureSet::eClearcoat},
    {KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME, SceneFeatureSet::eIridescence},
    {KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME, SceneFeatureSet::eAnisotropy},
    {KHR_MATERIALS_SHEEN_EXTENSION_NAME, SceneFeatureSet::eSheen},
    {KHR_MATERIALS_DISPERSION_EXTENSION_NAME, SceneFeatureSet::eDispersion},
    {KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME, SceneFeatureSet::eDiffuseTransmission},
    {KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME, SceneFeatureSet::eRetroreflection},
    {KHR_MATERIALS_UNLIT_EXTENSION_NAME, SceneFeatureSet::eUnlit},
    {KHR_MATERIALS_SPECULAR_EXTENSION_NAME, SceneFeatureSet::eSpecular},
    {KHR_MATERIALS_IOR_EXTENSION_NAME, SceneFeatureSet::eIor},
    {KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME, SceneFeatureSet::eSpecularGlossiness},
    {FV_MATERIALS_DETAIL_EXTENSION_NAME, SceneFeatureSet::eDetail},
    {FV_MATERIALS_HAIR_EXTENSION_NAME, SceneFeatureSet::eHair},
};

// The places KHR_texture_transform can sit: the five core texture infos, the coat's map, and in an
// extension nothing reads, a texture info and one inside an array there
enum class TransformAt
{
  eBaseColor,
  eMetallicRoughness,
  eNormal,
  eOcclusion,
  eEmissive,
  eClearcoat,
  eExtension,
  eExtensionArray,
  eCount
};

// `index` -1 puts the transform on a texture info that names no texture, which counts for nothing
void addTransform(tinygltf::Material& m, TransformAt at, int index)
{
  const tinygltf::Value::Object transform{{KHR_TEXTURE_TRANSFORM_EXTENSION_NAME, object()}};
  switch(at)
  {
    case TransformAt::eBaseColor:
      m.pbrMetallicRoughness.baseColorTexture.index      = index;
      m.pbrMetallicRoughness.baseColorTexture.extensions = transform;
      break;
    case TransformAt::eMetallicRoughness:
      m.pbrMetallicRoughness.metallicRoughnessTexture.index      = index;
      m.pbrMetallicRoughness.metallicRoughnessTexture.extensions = transform;
      break;
    case TransformAt::eNormal:
      m.normalTexture.index      = index;
      m.normalTexture.extensions = transform;
      break;
    case TransformAt::eOcclusion:
      m.occlusionTexture.index      = index;
      m.occlusionTexture.extensions = transform;
      break;
    case TransformAt::eEmissive:
      m.emissiveTexture.index      = index;
      m.emissiveTexture.extensions = transform;
      break;
    case TransformAt::eClearcoat:
      m.extensions[KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME] = object({{"clearcoatTexture", transformedInfo(index)}});
      break;
    case TransformAt::eExtension:
      m.extensions["FV_nothing_reads"] = object({{"someTexture", transformedInfo(index)}});
      break;
    case TransformAt::eExtensionArray:
      m.extensions["FV_nothing_reads"] = object({{"layers", tinygltf::Value(tinygltf::Value::Array{object(), transformedInfo(index)})}});
      break;
    case TransformAt::eCount:
      break;
  }
}

fv::mat::Material randomPayload(std::mt19937& rng)
{
  tinygltf::Material m;
  m.name = "random";
  for(const KnownExtension& e : kExtensions)
    if(rng() % 8 == 0)
      m.extensions[e.name] = object();
  if(rng() % 3 == 0)
    addTransform(m, TransformAt(rng() % uint32_t(TransformAt::eCount)), rng() % 4 == 0 ? -1 : int(rng() % 3));
  return fv::mat::fromGltf(m);
}

// Four vertices, three submeshes over them (the middle one empty, as a group a geograft hides
// whole), each drawing with its own material slot
rs::MeshRecord quad()
{
  rs::MeshRecord m;
  m.name      = "quad";
  m.positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  m.normals.assign(4, glm::vec3(0, 0, 1));
  m.indices   = {0, 1, 2, 0, 2, 3};
  m.submeshes = {{.firstIndex = 0, .indexCount = 3, .material = 0, .order = 0},
                 {.firstIndex = 3, .indexCount = 0, .material = 1, .order = 1},
                 {.firstIndex = 3, .indexCount = 3, .material = 2, .order = 2}};
  return m;
}

rs::MeshRecord triangle()
{
  rs::MeshRecord m;
  m.name      = "triangle";
  m.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  m.indices   = {0, 1, 2};
  m.submeshes = {{.firstIndex = 0, .indexCount = 3, .material = 0}};
  return m;
}

rs::MeshInstance drawWith(rs::MeshId mesh, std::vector<rs::MaterialId> materials)
{
  return {.parts = {{.mesh = mesh, .materials = std::move(materials)}}};
}

// Seeded random steps over a scene, the oracle checked after each: material edits (both
// setMaterial overloads), material adds and releases, instance swaps, entity creates, destroys
// and hides. It holds a reference to the meshes and materials it may draw with, as a feed would.
class Fuzz
{
public:
  Fuzz(rs::RetainedScene& scene, uint32_t seed, std::vector<tinygltf::Material> payloads = {})
      : m_scene(scene)
      , m_rng(seed)
      , m_payloads(std::move(payloads))
  {
  }

  void hold(rs::MeshId id)
  {
    if(m_scene.mesh(id) && std::find(m_meshes.begin(), m_meshes.end(), id) == m_meshes.end())
    {
      m_scene.retain(id);
      m_meshes.push_back(id);
    }
  }
  void hold(rs::MaterialId id)
  {
    if(m_scene.material(id) && std::find(m_materials.begin(), m_materials.end(), id) == m_materials.end())
    {
      m_scene.retain(id);
      m_materials.push_back(id);
    }
  }

  void run(int steps, const std::string& what)
  {
    m_scene.takeChanges();
    SceneFeatureSet before = m_scene.features();
    for(int i = 0; i < steps; i++)
    {
      const std::string op = step();
      SCOPED_TRACE(what + ", step " + std::to_string(i) + " (" + op + ")");
      expectFeaturesOfExport(m_scene, what);
      const SceneFeatureSet after = m_scene.features();
      EXPECT_EQ(m_scene.takeChanges().features, after != before);
      before = after;
      if(::testing::Test::HasFailure())
        return;
    }
  }

  // Everything goes: the scene ends with no material and no feature
  void clear()
  {
    for(rs::MaterialId id : m_materials)
      m_scene.release(id);
    for(rs::MeshId id : m_meshes)
      m_scene.release(id);
    m_materials.clear();
    m_meshes.clear();
    for(rs::EntityId id : std::vector<rs::EntityId>(m_scene.roots()))
      m_scene.destroyEntity(id);
    EXPECT_EQ(m_scene.materialCount(), 0u);
    EXPECT_TRUE(m_scene.features() == SceneFeatureSet{}) << m_scene.features().toString();
    expectFeaturesOfExport(m_scene, "cleared");
  }

private:
  template <class T>
  const T& pick(const std::vector<T>& v)
  {
    return v[m_rng() % v.size()];
  }

  fv::mat::Material payload()
  {
    return !m_payloads.empty() && m_rng() % 2 ? fv::mat::fromGltf(pick(m_payloads)) : randomPayload(m_rng);
  }

  // what a step may draw with: the materials held, and those instances draw with
  std::vector<rs::MaterialId> drawable() const
  {
    std::vector<rs::MaterialId> out = m_materials;
    for(rs::EntityId id : m_scene.entitiesInOrder())
      if(const rs::Entity* e = m_scene.entity(id); e->mesh)
        for(const rs::MeshPart& part : e->mesh->parts)
          for(rs::MaterialId m : part.materials)
            if(m_scene.material(m))
              out.push_back(m);
    return out;
  }

  std::optional<rs::MeshInstance> instance(const rs::Entity* e)
  {
    const std::vector<rs::MaterialId> materials = drawable();
    if(m_meshes.empty() || materials.empty() || m_rng() % 6 == 0)
      return std::nullopt;
    // one material of what it draws swapped
    if(e && e->mesh && !e->mesh->parts.empty() && m_rng() % 2)
    {
      rs::MeshInstance i = *e->mesh;
      rs::MeshPart&    p = i.parts[m_rng() % i.parts.size()];
      if(!p.materials.empty())
        p.materials[m_rng() % p.materials.size()] = pick(materials);
      return i;
    }
    rs::MeshInstance i;
    for(uint32_t n = 1 + m_rng() % 2; n > 0; n--)
    {
      rs::MeshPart p{.mesh = pick(m_meshes)};
      uint32_t     slots = 0;
      for(const rs::Submesh& s : m_scene.mesh(p.mesh)->submeshes)
        slots = std::max(slots, s.material + 1);
      // now and then one slot short, so the last draws with no material
      if(slots && m_rng() % 8 == 0)
        slots--;
      for(uint32_t s = 0; s < slots; s++)
        p.materials.push_back(pick(materials));
      i.parts.push_back(std::move(p));
    }
    return i;
  }

  std::string step()
  {
    const std::vector<rs::EntityId> entities = m_scene.entitiesInOrder();
    switch(m_rng() % 8)
    {
      case 0:
      case 1: {
        const std::vector<rs::MaterialId> materials = drawable();
        if(materials.empty())
          return "edit: none";
        const rs::MaterialId id = pick(materials);
        switch(m_rng() % 4)
        {
          case 0:
            m_scene.setMaterial(id, payload(), m_scene.material(id)->textures);
            return "edit, same table";
          case 1:
            m_scene.setMaterial(id, payload(), m_rng() % 2 ? std::make_shared<const rs::TextureTable>() : nullptr);
            return "edit, other table";
          default:
            m_scene.setMaterial(id, payload());
            return "edit";
        }
      }
      case 2: {
        std::shared_ptr<const rs::TextureTable> table;
        if(!m_materials.empty())
          table = m_scene.material(pick(m_materials))->textures;
        const rs::MaterialId id = m_scene.addMaterial({.payload = payload(), .textures = table});
        m_materials.push_back(id);  // the add's reference is the one held
        return "add";
      }
      case 3: {
        if(m_materials.empty())
          return "release: none";
        const size_t i = m_rng() % m_materials.size();
        m_scene.release(m_materials[i]);
        m_materials.erase(m_materials.begin() + ptrdiff_t(i));
        return "release";
      }
      case 4:
      case 5: {
        if(entities.empty())
          return "swap: none";
        const rs::EntityId id = pick(entities);
        m_scene.setMeshInstance(id, instance(m_scene.entity(id)));
        return "swap";
      }
      case 6: {
        if(entities.size() > 2 && m_rng() % 2)
        {
          m_scene.destroyEntity(pick(entities));
          return "destroy";
        }
        const rs::EntityId parent = entities.empty() || m_rng() % 3 == 0 ? rs::EntityId{} : pick(entities);
        const rs::EntityId id     = m_scene.createEntity("fuzz", {}, parent);
        m_scene.setMeshInstance(id, instance(nullptr));
        return "create";
      }
      default: {
        if(entities.empty())
          return "hide: none";
        const rs::EntityId id = pick(entities);
        m_scene.setVisible(id, !m_scene.entity(id)->visible);
        return "hide";
      }
    }
  }

  rs::RetainedScene&              m_scene;
  std::mt19937                    m_rng;
  std::vector<tinygltf::Material> m_payloads;
  std::vector<rs::MeshId>         m_meshes;
  std::vector<rs::MaterialId>     m_materials;
};

tinygltf::Model load(const std::filesystem::path& file, bool& ok)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model    model;
  std::string        err, warn;
  ok = file.extension() == ".glb" ? loader.LoadBinaryFromFile(&model, &err, &warn, file.string()) :
                                    loader.LoadASCIIFromFile(&model, &err, &warn, file.string());
  return model;
}

}  // namespace

TEST(RetainedFeatures, EveryExtensionTheDetectorKnowsFromAMaterialThatDrawsIt)
{
  struct Case
  {
    std::string              what;
    tinygltf::Material       payload;
    SceneFeatureSet::Feature feature;
  };
  std::vector<Case> cases;
  for(const KnownExtension& e : kExtensions)
  {
    tinygltf::Material m;
    m.extensions[e.name] = object();
    cases.push_back({e.name, m, e.feature});
  }
  for(int at = 0; at < int(TransformAt::eCount); at++)
  {
    tinygltf::Material m;
    addTransform(m, TransformAt(at), 0);
    cases.push_back({"transform " + std::to_string(at), m, SceneFeatureSet::eTextureTransform});
  }

  SceneFeatureSet covered;
  for(const Case& c : cases)
  {
    SCOPED_TRACE(c.what);
    rs::RetainedScene    scene;
    const rs::MeshId     mesh = scene.addMesh(triangle());
    const rs::MaterialId mat  = scene.addMaterial({.payload = fv::mat::fromGltf(c.payload)});
    const rs::EntityId   e    = scene.createEntity("e", {});
    EXPECT_FALSE(scene.features().has(c.feature));  // not drawn yet
    scene.setMeshInstance(e, drawWith(mesh, {mat}));
    scene.release(mesh);
    scene.release(mat);
    EXPECT_TRUE(scene.features().has(c.feature)) << scene.features().toString();
    expectFeaturesOfExport(scene, c.what);
    covered |= scene.features();
    scene.destroyEntity(e);
    EXPECT_TRUE(scene.features() == SceneFeatureSet{});
    EXPECT_EQ(scene.materialCount(), 0u);
  }
  // every material bit is reachable from some case
  for(size_t f = 0; f < SceneFeatureSet::kExtensionFeatureCount; f++)
    EXPECT_TRUE(covered.has(SceneFeatureSet::Feature(f))) << "feature " << f;

  // a transform on a texture info that names no texture is not one
  rs::RetainedScene  scene;
  tinygltf::Material none;
  addTransform(none, TransformAt::eClearcoat, -1);
  const rs::MaterialId mat  = scene.addMaterial({.payload = fv::mat::fromGltf(none)});
  const rs::MeshId     mesh = scene.addMesh(triangle());
  scene.setMeshInstance(scene.createEntity("e", {}), drawWith(mesh, {mat}));
  EXPECT_FALSE(scene.features().has(SceneFeatureSet::eTextureTransform));
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eClearcoat));  // its host still is
  expectFeaturesOfExport(scene, "transform with no texture");
}

TEST(RetainedFeatures, OnlyWhatIsDrawnCounts)
{
  rs::RetainedScene scene;
  const rs::MeshId  mesh = scene.addMesh(quad());
  const auto        with = [&](const char* ext) {
    tinygltf::Material m;
    m.extensions[ext] = object();
    return scene.addMaterial({.payload = fv::mat::fromGltf(m)});
  };
  const rs::MaterialId plain = scene.addMaterial({});
  const rs::MaterialId held  = with(KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);
  const rs::MaterialId empty = with(KHR_MATERIALS_SHEEN_EXTENSION_NAME);
  const rs::MaterialId spare = with(KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME);

  // a material only held (by a feed, say) is not drawn and not exported
  EXPECT_TRUE(scene.features() == SceneFeatureSet{});
  expectFeaturesOfExport(scene, "held");

  // a slot past the last a submesh names draws nothing: the export writes no primitive with it
  const rs::EntityId e = scene.createEntity("e", {});
  scene.setMeshInstance(e, rs::MeshInstance{.parts = {{.mesh = mesh, .materials = {plain, empty, plain, spare}}}});
  EXPECT_FALSE(scene.features().has(SceneFeatureSet::eIridescence));
  expectFeaturesOfExport(scene, "a slot no submesh names");

  // the empty submesh's material counts, as the export writes its primitive
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eSheen));
  EXPECT_FALSE(scene.features().has(SceneFeatureSet::eTransmission));

  // hidden, it still counts: hiding is not a reason to rebuild a shader
  scene.setVisible(e, false);
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eSheen));
  expectFeaturesOfExport(scene, "hidden");

  // an edit to a material nothing draws moves nothing
  scene.takeChanges();
  tinygltf::Material clearcoat;
  clearcoat.extensions[KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME] = object();
  scene.setMaterial(held, fv::mat::fromGltf(clearcoat));
  EXPECT_FALSE(scene.features().has(SceneFeatureSet::eClearcoat));
  EXPECT_FALSE(scene.takeChanges().features);
  expectFeaturesOfExport(scene, "an undrawn material edited");

  for(rs::MaterialId m : {plain, held, empty, spare})
    scene.release(m);
  scene.release(mesh);
}

TEST(RetainedFeatures, ACountCrossesZeroAndBack)
{
  rs::RetainedScene    scene;
  const rs::MeshId     mesh = scene.addMesh(triangle());
  tinygltf::Material   coat;
  coat.extensions[KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME] = object();
  const rs::MaterialId coated = scene.addMaterial({.payload = fv::mat::fromGltf(coat)});
  const rs::MaterialId plain  = scene.addMaterial({});
  const rs::EntityId   a      = scene.createEntity("a", {});
  const rs::EntityId   b      = scene.createEntity("b", {});
  scene.takeChanges();

  // two entities draw the coat: one going leaves it on
  scene.setMeshInstance(a, drawWith(mesh, {coated}));
  scene.setMeshInstance(b, drawWith(mesh, {coated}));
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eClearcoat));
  EXPECT_TRUE(scene.takeChanges().features);
  scene.destroyEntity(a);
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eClearcoat));
  EXPECT_FALSE(scene.takeChanges().features);
  expectFeaturesOfExport(scene, "one of two gone");

  // the other swapped to a plain material: the count reaches zero
  scene.setMeshInstance(b, drawWith(mesh, {plain}));
  EXPECT_FALSE(scene.features().has(SceneFeatureSet::eClearcoat));
  EXPECT_TRUE(scene.takeChanges().features);
  expectFeaturesOfExport(scene, "swapped off");

  // and the plain material edited to a coat brings it back
  scene.setMaterial(plain, fv::mat::fromGltf(coat));
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eClearcoat));
  EXPECT_TRUE(scene.takeChanges().features);
  expectFeaturesOfExport(scene, "edited on");

  // off and on again between two takes is no change
  scene.setMaterial(plain, {});
  scene.setMaterial(plain, fv::mat::fromGltf(coat));
  EXPECT_FALSE(scene.takeChanges().features);

  // promoted dependencies: scatter brings volume and transmission, and they go with it
  tinygltf::Material scatter;
  scatter.extensions[KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME] = object();
  scene.setMaterial(plain, fv::mat::fromGltf(scatter));
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eTransmission));
  EXPECT_TRUE(scene.features().has(SceneFeatureSet::eVolume));
  expectFeaturesOfExport(scene, "scatter");
  scene.setMeshInstance(b, std::nullopt);
  EXPECT_TRUE(scene.features() == SceneFeatureSet{});
  expectFeaturesOfExport(scene, "nothing drawn");

  // a release too many takes a drawn material: what drew with it counts for nothing, as the
  // export writes nothing for it
  scene.setMeshInstance(b, drawWith(mesh, {coated}));
  scene.release(coated);
  scene.release(coated);
  EXPECT_EQ(scene.material(coated), nullptr);
  EXPECT_TRUE(scene.features() == SceneFeatureSet{});
  expectFeaturesOfExport(scene, "released under an instance");
  scene.destroyEntity(b);
  EXPECT_TRUE(scene.features() == SceneFeatureSet{});

  scene.release(plain);
  scene.release(mesh);
}

TEST(RetainedFeatures, RandomEditsOnSyntheticMaterials)
{
  for(uint32_t seed = 1; seed <= 24; seed++)
  {
    rs::RetainedScene scene;
    Fuzz              fuzz(scene, seed);
    std::mt19937      rng(seed * 7919u);
    const rs::MeshId  meshes[] = {scene.addMesh(quad()), scene.addMesh(triangle())};
    for(rs::MeshId m : meshes)
    {
      fuzz.hold(m);
      scene.release(m);
    }
    // one material per known extension and per transform place, and a few random ones
    std::vector<rs::MaterialId> materials;
    for(const KnownExtension& e : kExtensions)
    {
      tinygltf::Material m;
      m.extensions[e.name] = object();
      materials.push_back(scene.addMaterial({.payload = fv::mat::fromGltf(m)}));
    }
    for(int at = 0; at < int(TransformAt::eCount); at++)
    {
      tinygltf::Material m;
      addTransform(m, TransformAt(at), 0);
      materials.push_back(scene.addMaterial({.payload = fv::mat::fromGltf(m)}));
    }
    for(int i = 0; i < 4; i++)
      materials.push_back(scene.addMaterial({.payload = randomPayload(rng)}));
    for(rs::MaterialId m : materials)
    {
      fuzz.hold(m);
      scene.release(m);
    }
    // a few entities drawing them, some under others
    std::vector<rs::EntityId> entities;
    for(int i = 0; i < 6; i++)
    {
      const rs::EntityId parent = i >= 2 ? entities[rng() % entities.size()] : rs::EntityId{};
      entities.push_back(scene.createEntity("e" + std::to_string(i), {}, parent));
      const rs::MeshId mesh = meshes[i % 2];
      scene.setMeshInstance(entities.back(), drawWith(mesh, {materials[rng() % materials.size()], materials[rng() % materials.size()],
                                                             materials[rng() % materials.size()]}));
    }
    expectFeaturesOfExport(scene, "seed " + std::to_string(seed) + ", start");
    fuzz.run(300, "seed " + std::to_string(seed));
    fuzz.clear();
    if(HasFailure())
      return;
  }
}

TEST(RetainedFeatures, RandomEditsOnImportedFixtures)
{
  namespace fs = std::filesystem;
  int      tested = 0;
  uint32_t seed   = 1;
  for(const fs::path dir : {fs::path(GLTF_2_1_FIXTURES_DIR), fs::path(GLTF_SAMPLE_ASSETS_PATH)})
  {
    std::error_code ec;
    if(dir.empty() || !fs::exists(dir, ec))
      continue;
    for(const auto& entry : fs::recursive_directory_iterator(dir, ec))
    {
      const fs::path file = entry.path();
      if(file.extension() != ".gltf" && file.extension() != ".glb")
        continue;
      bool                  ok    = false;
      const tinygltf::Model model = load(file, ok);
      if(!ok || model.nodes.empty() || model.scenes.empty())
        continue;
      rs::RetainedScene      scene;
      const rs::ImportResult imported = rs::importGltf(scene, model, {.prop = 1});
      const std::string      what     = file.filename().string();
      expectFeaturesOfExport(scene, what + ", imported");
      // as imported, the drawn materials' features are the file's (less those of materials nothing draws)
      std::vector<tinygltf::Material> drawn;
      for(size_t i = 0; i < imported.materials.size(); i++)
        if(!imported.materials[i].isNone())
          drawn.push_back(model.materials[i]);
      EXPECT_TRUE(materialBits(scene.features()) == materialBits(nvvkgltf::detectSceneFeatures(drawn))) << what;

      // edits draw from the file's own payloads (its texture indices) as well as random ones
      Fuzz fuzz(scene, seed++, model.materials);
      for(const std::vector<rs::MeshId>& meshes : imported.meshes)
        for(rs::MeshId m : meshes)
          fuzz.hold(m);
      for(rs::MaterialId m : imported.materials)
        fuzz.hold(m);
      fuzz.run(60, what);
      fuzz.clear();
      tested++;
      if(HasFailure())
        return;
    }
  }
  if(tested == 0)
    GTEST_SKIP() << "no fixtures generated and no sample assets";
  std::printf("fuzzed the features of %d files\n", tested);
}
