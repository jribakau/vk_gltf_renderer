// A scene file's props load with what the scene gives each of them: its scene materials and the UV
// set they pick, found by the geometry instance, not by a library id another instance happens to have.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <figure/dson/figure_loader.hpp>
#include <figure/dson/store.hpp>

namespace {

namespace fs = std::filesystem;

void writeFile(const fs::path& p, const std::string& text)
{
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << text;
}

// A one-quad prop whose geometry is called "geometry", as every prop of a product exported together
// is, with its own UV set (also with the id every other prop's has) spanning `u0` to `u1` across.
void writeProp(const fs::path& root, const std::string& name, double u0, double u1)
{
  const std::string dir = "/data/T/" + name + "/";
  writeFile(root / ("data/T/" + name + "/" + name + ".dsf"),
            R"({"file_version":"0.6.0.0","asset_info":{"id":")" + dir + name + R"(.dsf"},
  "geometry_library":[{"id":"geometry","label":")" + name + R"(","type":"polygon_mesh",
    "vertices":{"count":4,"values":[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]},
    "polylist":{"count":1,"values":[[0,0,0,1,2,3]]},
    "polygon_groups":{"count":1,"values":["g"]},
    "polygon_material_groups":{"count":1,"values":["Surface )" + name + R"("]},
    "default_uv_set":")" + dir + R"(uv.dsf#default"}],
  "node_library":[{"id":")" + name + R"(","name":")" + name + R"(","type":"node"}]})");
  const std::string a = std::to_string(u0), b = std::to_string(u1);
  writeFile(root / ("data/T/" + name + "/uv.dsf"),
            R"({"file_version":"0.6.0.0","asset_info":{"id":")" + dir + R"(uv.dsf"},
  "uv_set_library":[{"id":"default","vertex_count":4,
    "uvs":{"count":4,"values":[[)" + a + ",0],[" + b + ",0],[" + b + ",1],[" + a + R"(,1]]}}]})");
}

const fv::Mesh* meshNamed(const fv::dson::LoadResult& r, const std::string& name)
{
  for(const auto& m : r.scene.meshes)
    if(m.name == name)
      return &m;
  return nullptr;
}

}  // namespace

TEST(SceneLoad, APropsSceneMaterialsAreItsInstancesNotThoseOfTheInstanceNamedLikeItsGeometry)
{
  const fs::path root = fs::temp_directory_path() / "fv-scene-load-test";
  fs::remove_all(root);
  writeProp(root, "Floor", 0.0, 1.0);
  writeProp(root, "Shelf", 0.25, 0.5);
  // the floor's instance keeps the bare id "geometry", the shelf's is "geometry-1"; each scene
  // material picks its own prop's UV set, and the floor's come first
  writeFile(root / "Scenes/Room.duf", R"({"file_version":"0.6.0.0","asset_info":{"id":"/Scenes/Room.duf"},
  "scene":{
    "nodes":[
      {"id":"Floor","url":"/data/T/Floor/Floor.dsf#Floor","geometries":[{"id":"geometry","url":"/data/T/Floor/Floor.dsf#geometry"}]},
      {"id":"Shelf","url":"/data/T/Shelf/Shelf.dsf#Shelf","geometries":[{"id":"geometry-1","url":"/data/T/Shelf/Shelf.dsf#geometry"}]}],
    "materials":[
      {"id":"Surface Floor-1","geometry":"#geometry","groups":["Surface Floor"],"uv_set":"/data/T/Floor/uv.dsf#default"},
      {"id":"Surface Shelf-1","geometry":"#geometry-1","groups":["Surface Shelf"],"uv_set":"/data/T/Shelf/uv.dsf#default"}]}})");

  fv::dson::Store store(std::make_shared<fv::dson::Library>(std::vector<fs::path>{root}));
  const auto      r = fv::dson::loadDson(store, "Scenes/Room.duf");

  auto uRange = [&](const std::string& name) {
    const fv::Mesh* m = meshNamed(r, name);
    EXPECT_NE(m, nullptr) << name;
    float lo = 1e9f, hi = -1e9f;
    if(m)
      for(size_t i = 0; i < m->geo->uv.size(); i += 2)
      {
        lo = std::min(lo, m->geo->uv[i]);
        hi = std::max(hi, m->geo->uv[i]);
      }
    return std::pair{lo, hi};
  };
  const auto [floorLo, floorHi] = uRange("Floor");
  EXPECT_FLOAT_EQ(floorLo, 0.f);
  EXPECT_FLOAT_EQ(floorHi, 1.f);
  const auto [shelfLo, shelfHi] = uRange("Shelf");
  EXPECT_FLOAT_EQ(shelfLo, 0.25f);
  EXPECT_FLOAT_EQ(shelfHi, 0.5f);

  fs::remove_all(root);
}
