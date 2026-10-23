// FigureDriver against a MorphSet and a PoseRig: a controller dial drives a morph through a spline
// and nods a bone, a bone drives a corrective, a shape moves a joint centre and the skin is re-bound
// there, and a baseline counts offsets from centres the scene already stored. A port of
// test/pose/driver.test.ts.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <figure/dson/rig.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/figure_driver.hpp>
#include <figure/scene/graph.hpp>

namespace {

const char* kShape = R"({
  "modifier_library": [{
    "id": "Shape",
    "channel": {"value": 0, "min": 0, "max": 1},
    "morph": {"deltas": {"values": [[3, 0, 10, 0]]}},
    "formulas": [{
      "output": "head:/data/Fig/Fig.dsf#head?center_point/y",
      "operations": [{"op": "push", "url": "Fig:#Shape?value"}, {"op": "push", "val": 5}, {"op": "mult"}]
    }]
  }]
})";

const char* kCtrl = R"({
  "modifier_library": [{
    "id": "CTRL",
    "channel": {"value": 0, "min": 0, "max": 1, "label": "Control"},
    "formulas": [
      {"output": "head:/data/Fig/Fig.dsf#head?rotation/x",
       "operations": [{"op": "push", "url": "Fig:#CTRL?value"}, {"op": "push", "val": 30}, {"op": "mult"}]},
      {"output": "Fig:/data/Fig/Morphs/Tail.dsf#Tail?value",
       "operations": [
         {"op": "push", "url": "Fig:#CTRL?value"},
         {"op": "push", "val": [0, 0, 0, 0, 0]},
         {"op": "push", "val": [0.5, 0.1, 0, 0, 0]},
         {"op": "push", "val": [1, 1, 0, 0, 0]},
         {"op": "push", "val": 3},
         {"op": "spline_linear"}
       ]}
    ]
  }]
})";

const char* kTail = R"({
  "modifier_library": [{
    "id": "Tail",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[2, 0, 1, 0]]}}
  }]
})";

const char* kJcm = R"({
  "modifier_library": [
    {"id": "JCMs On", "channel": {"type": "bool", "value": true}},
    {"id": "JCM", "channel": {"value": 0, "min": 0, "max": 1, "clamped": true},
     "morph": {"deltas": {"values": [[1, 0, 1, 0]]}},
     "formulas": [
       {"output": "Fig:#JCM?value",
        "operations": [{"op": "push", "url": "head:/data/Fig/Fig.dsf#head?rotation/x"}, {"op": "push", "val": 0.03333333333333333}, {"op": "mult"}]},
       {"output": "Fig:#JCM?value", "stage": "mult",
        "operations": [{"op": "push", "url": "Fig:#JCMs On?value"}]}
     ]}
  ]
})";

const char* kNodes = R"([
  {"id": "hip", "type": "figure", "center_point": [{"id": "y", "value": 0}]},
  {"id": "head", "type": "bone", "parent": "#hip", "center_point": [{"id": "y", "value": 15}],
   "rotation": [{"id": "x", "min": -180, "max": 180}, {"id": "y", "min": -180, "max": 180}, {"id": "z", "min": -180, "max": 180}]},
  {"id": "eye", "type": "bone", "parent": "#head", "center_point": [{"id": "y", "value": 25}]}
])";

class MemSource : public fv::pose::FigureSource
{
public:
  MemSource()
  {
    add("data/Fig/Morphs/Shape.dsf", kShape);
    add("data/Fig/Morphs/CTRL.dsf", kCtrl);
    add("data/Fig/Morphs/Tail.dsf", kTail);
    add("data/Fig/Morphs/Correctives/JCM.dsf", kJcm);
  }

  std::shared_ptr<const fv::dson::Doc> doc(const std::string& file) override
  {
    std::lock_guard lock(mutex);
    loaded.push_back(file);
    auto            it = docs.find(file);
    return it == docs.end() ? nullptr : it->second;
  }

  std::vector<fv::dson::MorphDef> targets(const fv::dson::Doc& doc, const std::string&) override
  {
    std::vector<fv::dson::MorphDef> out;
    const fv::dson::Json*           mods = fv::dson::get(&doc.json, "modifier_library");
    if(!mods || !mods->is_array())
      return out;
    for(const fv::dson::Json& m : *mods)
    {
      const fv::dson::Json* morph  = fv::dson::get(&m, "morph");
      const fv::dson::Json* deltas = morph ? fv::dson::get(morph, "deltas") : nullptr;
      const fv::dson::Json* values = deltas ? fv::dson::get(deltas, "values") : nullptr;
      if(!values || !values->is_array())
        continue;
      auto target = std::make_shared<fv::mesh::RenderTarget>();
      for(const fv::dson::Json& row : *values)
      {
        target->indices.push_back(uint32_t(row.at(0).get<int>()));
        target->deltas.push_back(float(row.at(1).get<double>()));
        target->deltas.push_back(float(row.at(2).get<double>()));
        target->deltas.push_back(float(row.at(3).get<double>()));
      }
      fv::dson::MorphPart part;
      part.mesh   = 0;
      part.target = target;
      fv::dson::MorphDef def;
      def.id = fv::dson::getString(&m, "id");
      def.parts.push_back(std::move(part));
      out.push_back(std::move(def));
    }
    return out;
  }

  std::optional<std::string> find(const std::string& id) override
  {
    const std::string suffix = "/" + id + ".dsf";
    for(const auto& [file, doc] : docs)
      if(file.size() >= suffix.size() && file.compare(file.size() - suffix.size(), suffix.size(), suffix) == 0)
        return file;
    return std::nullopt;
  }

  std::vector<std::string> filesLoaded() const
  {
    std::lock_guard lock(mutex);
    return loaded;
  }

  std::map<std::string, std::shared_ptr<fv::dson::Doc>> docs;

private:
  void add(const char* file, const char* text)
  {
    auto doc  = std::make_shared<fv::dson::Doc>();
    doc->json = fv::dson::Json::parse(text);
    docs.emplace(file, std::move(doc));
  }

  mutable std::mutex       mutex;
  std::vector<std::string> loaded;
};

struct Fig
{
  fv::SceneGraph                           scene;
  std::shared_ptr<fv::mesh::BuiltMesh>     geo;
  std::shared_ptr<MemSource>               source;
  std::unique_ptr<fv::morph::MorphSet>     morphs;
  std::unique_ptr<fv::pose::FigureDriver>  driver;
  std::unique_ptr<fv::pose::PoseRig>       pose;
  fv::dson::Rig                            rig;
  int                                      head = -1, eye = -1, meshNode = -1;
  int                                      asked = 0;

  float y(int vertex) const { return geo->position[size_t(vertex) * 3 + 1]; }
};

std::unique_ptr<Fig> setup()
{
  auto f      = std::make_unique<Fig>();
  f->geo      = std::make_shared<fv::mesh::BuiltMesh>();
  f->geo->position = {0, 0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0};
  f->geo->normal.assign(12, 0.f);
  f->geo->renderIndex     = {0, 1, 2, 3};
  f->geo->firstSplit      = {0, 1, 2, 3};
  f->geo->nextSplit       = {-1, -1, -1, -1};
  f->geo->renderVertexCount = 4;
  f->geo->renderTris      = {0, 1, 2};
  f->geo->dsonVertexCount = 4;

  auto rig = fv::dson::buildRig(f->scene, fv::dson::Json::parse(kNodes), nullptr);
  EXPECT_TRUE(rig);
  f->rig  = *rig;
  f->head = f->rig.byId("head");
  f->eye  = f->rig.byId("eye");

  fv::Mesh body;
  body.name = "body";
  body.geo  = f->geo;
  body.bones = {f->rig.byId("hip"), f->head};
  f->scene.meshes.push_back(std::move(body));
  fv::Node meshNode;
  meshNode.name = "body";
  meshNode.kind = fv::NodeKind::eMesh;
  meshNode.mesh = 0;
  f->meshNode   = f->scene.add(std::move(meshNode));

  f->source = std::make_shared<MemSource>();
  f->morphs = std::make_unique<fv::morph::MorphSet>([geo = f->geo](int) { return geo.get(); });
  f->driver = std::make_unique<fv::pose::FigureDriver>(*f->morphs, f->source);
  f->morphs->driver = f->driver.get();
  f->morphs->addLazy("Shape", f->driver->loader("data/Fig/Morphs/Shape.dsf", "Shape"));
  f->morphs->addLazy("CTRL", f->driver->loader("data/Fig/Morphs/CTRL.dsf", "CTRL"));

  f->pose = std::make_unique<fv::pose::PoseRig>(f->scene, f->rig, std::vector<const fv::dson::Rig*>{});
  f->driver->rig = f->pose.get();
  Fig* self      = f.get();
  auto loadCorrectives = [self] {
    if(self->asked)
      return;
    self->asked++;
    const std::string file = "data/Fig/Morphs/Correctives/JCM.dsf";
    auto              doc  = self->source->doc(file);
    self->driver->graph.addDoc(doc->json, file);
    for(fv::dson::MorphDef& def : self->source->targets(*doc, file))
      for(const fv::dson::MorphPart& p : def.parts)
      {
        fv::morph::ChannelOptions opts;
        opts.hidden = true;
        self->morphs->addTarget(def.id, p.mesh, p.target, opts);
      }
    self->driver->update();
  };
  f->driver->onDriven = loadCorrectives;
  f->pose->afterPose  = [self, loadCorrectives] {
    self->driver->update();
    loadCorrectives();
  };
  return f;
}

// One bone, weight 1: boneWorld · boneInverse · point, with the mesh and its bind at the origin.
fv::Vec3 skinHead(const Fig& f, int vertex)
{
  const std::vector<fv::Mat4> worlds = f.scene.worlds();
  const fv::Mesh&             mesh   = f.scene.meshes[0];
  const fv::Vec3              p{mesh.geo->position[size_t(vertex) * 3], mesh.geo->position[size_t(vertex) * 3 + 1],
                   mesh.geo->position[size_t(vertex) * 3 + 2]};
  const fv::Vec3              local = inverse(mesh.bindMatrix).transformPoint(p);
  const int                   joint = 1;  // head, the second skin bone
  return (worlds[size_t(mesh.bones[joint])] * mesh.boneInverses[size_t(joint)]).transformPoint(local);
}

}  // namespace

TEST(FigureDriver, ControllerDrivesAMorphABoneAndTheCorrective)
{
  auto        f = setup();
  f->morphs->set("CTRL", 0.75);
  f->morphs->settled();

  const auto loaded = f->source->filesLoaded();
  EXPECT_NE(std::find(loaded.begin(), loaded.end(), "data/Fig/Morphs/Tail.dsf"), loaded.end());
  const fv::morph::Channel* tail = f->morphs->channel("Tail");
  ASSERT_NE(tail, nullptr);
  EXPECT_TRUE(tail->hidden);
  EXPECT_NEAR(f->morphs->weightOf("Tail"), 0.55, 1e-4);
  EXPECT_NEAR(f->y(2), 20.55, 1e-4);

  const fv::morph::Channel* ctrl = f->morphs->channel("CTRL");
  ASSERT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->label, "Control");
  EXPECT_DOUBLE_EQ(f->morphs->get("CTRL"), 0.75);

  const auto posed = f->pose->posed("head", "rotation", "x");
  ASSERT_TRUE(posed);
  EXPECT_DOUBLE_EQ(*posed, 0);
  const fv::Quat& q    = f->scene.nodes[f->head].quaternion;
  const double    half = 22.5 * fv::kDeg / 2;
  EXPECT_NEAR(q.x, std::sin(half), 1e-5);
  EXPECT_NEAR(q.y, 0, 1e-5);
  EXPECT_NEAR(q.z, 0, 1e-5);
  EXPECT_NEAR(q.w, std::cos(half), 1e-5);

  EXPECT_EQ(f->asked, 1);
  EXPECT_NEAR(f->morphs->weightOf("JCM"), 0.75, 1e-4);

  f->pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", 7.5}});
  EXPECT_NEAR(f->morphs->weightOf("JCM"), 1, 1e-4);
  f->pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", -30}});
  EXPECT_NEAR(f->morphs->weightOf("JCM"), 0, 1e-4);

  f->pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", 0}});
  f->morphs->set("CTRL", 0);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Tail"), 0, 1e-6);
  EXPECT_NEAR(f->y(2), 20, 1e-4);
  const fv::Quat& back = f->scene.nodes[f->head].quaternion;
  EXPECT_NEAR(back.x, 0, 1e-5);
  EXPECT_NEAR(back.y, 0, 1e-5);
  EXPECT_NEAR(back.z, 0, 1e-5);
  EXPECT_NEAR(back.w, 1, 1e-5);
  EXPECT_NEAR(f->morphs->weightOf("JCM"), 0, 1e-4);
}

TEST(FigureDriver, AShapeMovesTheJointAndTheSkinRebindsThere)
{
  auto f = setup();
  f->morphs->set("Shape", 1);
  f->morphs->settled();
  if(f->pose->takeRestMoved())
    f->pose->rebind();

  EXPECT_NEAR(f->y(3), 40, 1e-4);
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 20, 1e-4);
  EXPECT_NEAR(f->scene.nodes[f->head].joint->center[1], 20, 1e-4);
  // the eye's own centre did not move, so it stays where it was in the figure
  EXPECT_NEAR(f->scene.worlds()[f->eye].e[13], 25, 1e-4);

  const fv::Vec3 atRest = skinHead(*f, 3);
  EXPECT_NEAR(atRest.y, 40, 1e-3);
  EXPECT_NEAR(atRest.z, 0, 1e-3);

  f->pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", 90}});
  const fv::Vec3 bent = skinHead(*f, 3);
  EXPECT_NEAR(bent.y, 20, 1e-3);
  EXPECT_NEAR(std::abs(bent.z), 20, 1e-3);

  f->pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", 0}});
  f->morphs->set("Shape", 0);
  f->morphs->settled();
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 15, 1e-4);
  EXPECT_NEAR(f->y(3), 30, 1e-4);
}

TEST(FigureDriver, BaselineCountsOffsetsFromCentresTheSceneStored)
{
  auto f = setup();
  f->morphs->set("Shape", 1);
  f->morphs->settled();
  // as if the scene file had stored the moved centre: nothing moves now, and the shape off moves it back
  f->pose->setCenterOffsets({});
  f->driver->captureBaseline();
  f->driver->update();
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 15, 1e-4);
  f->morphs->set("Shape", 0);
  f->morphs->settled();
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 10, 1e-4);
}
