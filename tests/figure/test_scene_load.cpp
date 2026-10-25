// A scene file's props load with what the scene gives each of them: its scene materials and the UV
// set they pick, found by the geometry instance, not by a library id another instance happens to have.
// DAZ Studio node instances draw their target again where they stand, sharing its mesh.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <figure/dson/figure_loader.hpp>
#include <figure/dson/store.hpp>
#include <figure/gltf/gltf_build.hpp>

namespace {

namespace fs = std::filesystem;
using fv::dson::Json;

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

Json translation(double x, double y, double z)
{
  return Json::array({{{"id", "x"}, {"current_value", x}}, {{"id", "y"}, {"current_value", y}}, {{"id", "z"}, {"current_value", z}}});
}

// A scene node for a prop writeProp wrote, with its geometry instance `gid`.
Json propNode(const std::string& name, const std::string& gid, double x, double y, double z, const std::string& parent = {})
{
  const std::string file = "/data/T/" + name + "/" + name + ".dsf";
  Json              n{{"id", name}, {"url", file + "#" + name}, {"label", name}, {"translation", translation(x, y, z)},
                      {"geometries", Json::array({{{"id", gid}, {"url", file + "#geometry"}}})}};
  if(!parent.empty())
    n["parent"] = "#" + parent;
  return n;
}

// A DAZ Studio node instance as the scene file has it: its definition (in the scene file's node
// library) holds the Instance Mode, its scene node the target; `mode` 0 also draws the target's children.
struct Instance
{
  std::string id, target;
  int         mode = 1;
  double      x = 0, y = 0, z = 0;
  std::string parent;
};

void writeScene(const fs::path& root, const std::vector<Json>& props, const std::vector<Instance>& instances)
{
  Json nodes = Json::array(), library = Json::array();
  for(const Json& p : props)
    nodes.push_back(p);
  for(const Instance& in : instances)
  {
    library.push_back({{"id", in.id}, {"name", in.id}, {"type", "node"},
                       {"extra", Json::array({{{"type", "studio_node_channels"},
                                               {"channels", Json::array({{{"channel", {{"id", "Instance Mode"}, {"type", "enum"}, {"value", 1}, {"current_value", in.mode}}}}})}}})}});
    Json n{{"id", in.id}, {"url", "#" + in.id}, {"label", in.id}, {"translation", translation(in.x, in.y, in.z)},
           {"extra", Json::array({{{"type", "studio/node/instance"}},
                                  {{"type", "studio_node_channels"},
                                   {"channels", Json::array({{{"channel", {{"id", "Instance Target"}, {"type", "node"}, {"node", "#" + in.target}}}}})}}})}};
    if(!in.parent.empty())
      n["parent"] = "#" + in.parent;
    nodes.push_back(n);
  }
  writeFile(root / "Scenes/Room.duf",
            Json{{"file_version", "0.6.0.0"}, {"asset_info", {{"id", "/Scenes/Room.duf"}}}, {"node_library", library}, {"scene", {{"nodes", nodes}}}}.dump());
}

fv::dson::LoadResult loadRoom(const fs::path& root)
{
  fv::dson::Store store(std::make_shared<fv::dson::Library>(std::vector<fs::path>{root}));
  return fv::dson::loadDson(store, "Scenes/Room.duf");
}

int meshIndex(const fv::dson::LoadResult& r, const std::string& name)
{
  for(size_t i = 0; i < r.scene.meshes.size(); i++)
    if(r.scene.meshes[i].name == name)
      return int(i);
  return -1;
}

// Where each node drawing the mesh stands (its world translation), in node order.
std::vector<std::array<double, 3>> drawnAt(const fv::dson::LoadResult& r, const std::string& name)
{
  const int                          mesh   = meshIndex(r, name);
  const auto                         worlds = r.scene.worlds();
  std::vector<std::array<double, 3>> out;
  for(size_t n = 0; n < r.scene.nodes.size(); n++)
    if(r.scene.nodes[n].kind == fv::NodeKind::eMesh && r.scene.nodes[n].mesh == mesh && mesh >= 0)
      out.push_back({worlds[n].e[12], worlds[n].e[13], worlds[n].e[14]});
  return out;
}

bool warned(const fv::dson::LoadResult& r, const std::string& text)
{
  return std::any_of(r.warnings.begin(), r.warnings.end(), [&](const std::string& w) { return w.find(text) != std::string::npos; });
}

class SceneInstances : public ::testing::Test
{
protected:
  void SetUp() override
  {
    root = fs::temp_directory_path() / "fv-scene-instances-test";
    fs::remove_all(root);
    writeProp(root, "Floor", 0.0, 1.0);
    writeProp(root, "Shelf", 0.25, 0.5);
  }
  void     TearDown() override { fs::remove_all(root); }
  fs::path root;
};

using At = std::vector<std::array<double, 3>>;

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

TEST_F(SceneInstances, AnInstanceDrawsItsTargetWhereItStandsSharingItsMesh)
{
  // where the instance stands takes the place of where the target does (DAZ Studio's placement;
  // the Stylish Apartment's stools only line up along the kitchen island this way)
  writeScene(root, {propNode("Floor", "geometry", 10, 0, 0)}, {{.id = "Floor Copy", .target = "Floor", .x = 100, .z = 50}});
  const auto r = loadRoom(root);
  EXPECT_EQ(r.scene.meshes.size(), 1u);
  EXPECT_EQ(drawnAt(r, "Floor"), (At{{10, 0, 0}, {100, 0, 50}}));
  EXPECT_FALSE(warned(r, "instance"));
  // the copy is no part of the morph context: the mesh is its target's
  EXPECT_EQ(r.morphs.meshes.size(), 1u);
}

TEST_F(SceneInstances, AnInstanceOfAnInstanceDrawsAtItsOwnPlaceWhateverTheFileOrder)
{
  writeScene(root, {propNode("Floor", "geometry", 10, 0, 0)},
             {{.id = "Second", .target = "First", .z = 200}, {.id = "First", .target = "Floor", .x = 100, .z = 50}});
  const auto r = loadRoom(root);
  At         at = drawnAt(r, "Floor");
  std::sort(at.begin(), at.end());
  EXPECT_EQ(at, (At{{0, 0, 200}, {10, 0, 0}, {100, 0, 50}}));
}

TEST_F(SceneInstances, NodeAndChildrenAlsoDrawsTheTargetsChildrenSingleNodeDoesNot)
{
  writeScene(root, {propNode("Floor", "geometry", 10, 0, 0), propNode("Shelf", "geometry-1", 0, 30, 0, "Floor")},
             {{.id = "With Children", .target = "Floor", .mode = 0, .x = 100}, {.id = "Alone", .target = "Floor", .mode = 1, .x = -100}});
  const auto r = loadRoom(root);
  EXPECT_EQ(drawnAt(r, "Floor"), (At{{10, 0, 0}, {100, 0, 0}, {-100, 0, 0}}));
  EXPECT_EQ(drawnAt(r, "Shelf"), (At{{10, 30, 0}, {100, 30, 0}}));
}

TEST_F(SceneInstances, InstancesHangingFromTheirOwnTargetDrawOnceEach)
{
  // A True Dream Home's Chair1: its "Node and Children" instances are its children, each an
  // instance of the one before; the children an instance draws leave the instances among them out
  writeScene(root, {propNode("Floor", "geometry", 10, 0, 0)},
             {{.id = "A", .target = "Floor", .mode = 0, .x = 50, .parent = "Floor"}, {.id = "B", .target = "A", .mode = 0, .x = 100, .parent = "Floor"}});
  const auto r = loadRoom(root);
  EXPECT_EQ(drawnAt(r, "Floor"), (At{{10, 0, 0}, {60, 0, 0}, {110, 0, 0}}));
}

TEST_F(SceneInstances, AnInstanceThatCannotBeDrawnIsNamed)
{
  writeScene(root, {propNode("Floor", "geometry", 10, 0, 0)},
             {{.id = "Lost", .target = "Nowhere"}, {.id = "After Lost", .target = "Lost"}, {.id = "Loop A", .target = "Loop B"}, {.id = "Loop B", .target = "Loop A"}});
  const auto r = loadRoom(root);
  EXPECT_EQ(drawnAt(r, "Floor").size(), 1u);
  EXPECT_TRUE(warned(r, "because their target isn't in the scene: Lost"));
  EXPECT_TRUE(warned(r, "because they draw themselves: Loop A, Loop B"));
  EXPECT_TRUE(warned(r, "because the instances they copy aren't drawn: After Lost"));
}

TEST_F(SceneInstances, TheGltfDrawsOneMeshFromEveryNodeThatShowsIt)
{
  writeScene(root, {propNode("Floor", "geometry", 10, 0, 0)}, {{.id = "One", .target = "Floor", .x = 100}, {.id = "Two", .target = "Floor", .x = 200}});
  const auto            r = loadRoom(root);
  fv::build::BuildOptions options;
  options.texture       = [](const fv::build::TextureRequest&) { return std::optional<fv::build::TextureImage>(); };
  const auto built      = fv::build::buildGltf(r.scene, r.root, options);
  ASSERT_EQ(built.model.meshes.size(), 1u);
  EXPECT_EQ(built.meshOf[0], 0);
  int drawing = 0;
  for(const auto& n : built.model.nodes)
    drawing += n.mesh == 0;
  EXPECT_EQ(drawing, 3);
  EXPECT_EQ(built.model.buffers.size(), 1u);
}
