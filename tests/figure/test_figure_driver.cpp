// FigureDriver against a MorphSet and a PoseRig: a controller dial drives a morph through a spline
// and nods a bone, a bone drives a corrective, a shape moves a joint centre and the skin is re-bound
// there, a baseline counts offsets from centres the scene already stored, and a dial that moves brings
// the files that read it. A port of test/pose/driver.test.ts.

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

// A smile moves the lips, whose own file multiplies them by a switch that is on unless turned off, as
// Mouth Sticky Power does to the morphs it reads.
const char* kSmile = R"({
  "modifier_library": [{
    "id": "Smile",
    "channel": {"value": 0, "min": 0, "max": 1},
    "formulas": [{"output": "Fig:/data/Fig/Morphs/Lips.dsf#Lips?value", "operations": [{"op": "push", "url": "Fig:#Smile?value"}]}]
  }]
})";

const char* kLips = R"({
  "modifier_library": [{
    "id": "Lips",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [{"output": "Fig:#Lips?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Power?value"}]}]
  }]
})";

const char* kPower = R"({
  "modifier_library": [{"id": "Power", "channel": {"value": 1, "min": 0, "max": 1}}]
})";

// A dial of its own, multiplied by a switch whose id isn't its file's name.
const char* kBrow = R"({
  "modifier_library": [{
    "id": "Brow",
    "channel": {"value": 0, "min": 0, "max": 1},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [{"output": "Fig:#Brow?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Switch?value"}]}]
  }]
})";

const char* kSwitch = R"({
  "modifier_library": [{"id": "Switch", "channel": {"value": 1, "min": 0, "max": 1}}]
})";

// A dial with nothing in its file, and morphs that say in their own files that it drives them.
const char* kNeck = R"({
  "modifier_library": [{"id": "Neck", "channel": {"value": 0, "min": 0, "max": 1}}]
})";

const char* kSwallow = R"({
  "modifier_library": [{
    "id": "Swallow",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [{"output": "Fig:#Swallow?value", "operations": [{"op": "push", "url": "Fig:#Neck?value"}]}]
  }]
})";

const char* kGulp = R"({
  "modifier_library": [{
    "id": "facs_cbs_Gulp",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [{"output": "Fig:#facs_cbs_Gulp?value", "operations": [{"op": "push", "url": "Fig:#Neck?value"}]}]
  }]
})";

// The same with ids that aren't their files' names, as most of Genesis 9's are not.
const char* kLift = R"({
  "modifier_library": [{"id": "Lift", "channel": {"value": 0, "min": 0, "max": 1}}]
})";

const char* kRaise = R"({
  "modifier_library": [{
    "id": "Raise",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [{"output": "Fig:#Raise?value", "operations": [{"op": "push", "url": "Fig:#Lift?value"}]}]
  }]
})";

// Reads a morph that Smile moves, whose dial stays at 0.
const char* kPout = R"({
  "modifier_library": [{
    "id": "Pout",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[2, 0, 3, 0]]}},
    "formulas": [{"output": "Fig:#Pout?value", "operations": [{"op": "push", "url": "Fig:#Lips?value"}]}]
  }]
})";

// A character's corrective for what a dial does, multiplied by the character's body, as most files
// that read a dial are. The body's id isn't its file's name.
const char* kGate = R"({
  "modifier_library": [{
    "id": "Gate-0x1",
    "channel": {"value": 0, "min": 0, "max": 1},
    "morph": {"deltas": {"values": [[2, 0, 1, 0]]}}
  }]
})";

const char* kGated = R"({
  "modifier_library": [{
    "id": "Gated",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [
      {"output": "Fig:#Gated?value", "operations": [{"op": "push", "url": "Fig:#Neck?value"}]},
      {"output": "Fig:#Gated?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Gate-0x1?value"}]}
    ]
  }]
})";

// The same with a multiplier a listing leaves out, which has no channel at all until something reads it.
const char* kGateHd = R"({
  "modifier_library": [{"id": "Gate_HD", "channel": {"value": 0, "min": 0, "max": 1}}]
})";

const char* kGatedHd = R"({
  "modifier_library": [{
    "id": "GatedHd",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [
      {"output": "Fig:#GatedHd?value", "operations": [{"op": "push", "url": "Fig:#Neck?value"}]},
      {"output": "Fig:#GatedHd?value", "stage": "mult", "operations": [{"op": "push", "url": "Fig:#Gate_HD?value"}]}
    ]
  }]
})";

// A character's dial, and its body in another file, which moves a joint.
const char* kChar = R"({
  "modifier_library": [{"id": "Char", "channel": {"value": 0, "min": 0, "max": 1}}]
})";

const char* kCharBody = R"({
  "modifier_library": [{
    "id": "CharBody",
    "channel": {"value": 0},
    "morph": {"deltas": {"values": [[1, 0, 2, 0]]}},
    "formulas": [
      {"output": "Fig:#CharBody?value", "operations": [{"op": "push", "url": "Fig:#Char?value"}]},
      {"output": "head:/data/Fig/Fig.dsf#head?center_point/y",
       "operations": [{"op": "push", "url": "Fig:#CharBody?value"}, {"op": "push", "val": 5}, {"op": "mult"}]}
    ]
  }]
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
    add("data/Fig/Morphs/Smile.dsf", kSmile);
    add("data/Fig/Morphs/Lips.dsf", kLips);
    add("data/Fig/Morphs/Power.dsf", kPower);
    add("data/Fig/Morphs/Brow.dsf", kBrow);
    add("data/Fig/Morphs/ctrl_Switch.dsf", kSwitch);
    add("data/Fig/Morphs/Neck.dsf", kNeck);
    add("data/Fig/Morphs/Swallow.dsf", kSwallow);
    add("data/Fig/Morphs/facs_cbs_Gulp.dsf", kGulp);
    add("data/Fig/Morphs/ctrl_Lift.dsf", kLift);
    add("data/Fig/Morphs/bs_Raise.dsf", kRaise);
    add("data/Fig/Morphs/Pout.dsf", kPout);
    add("data/Fig/Morphs/Char.dsf", kChar);
    add("data/Fig/Morphs/CharBody.dsf", kCharBody);
    add("data/Fig/Morphs/Gate Body.dsf", kGate);
    add("data/Fig/Morphs/Gated.dsf", kGated);
    add("data/Fig/Morphs/Gate_HD.dsf", kGateHd);
    add("data/Fig/Morphs/GatedHd.dsf", kGatedHd);
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
      auto                          target = std::make_shared<fv::mesh::RenderTarget>();
      fv::mesh::RenderTarget::Drawn drawn;
      for(const fv::dson::Json& row : *values)
      {
        drawn.indices.push_back(uint32_t(row.at(0).get<int>()));
        drawn.deltas.push_back(float(row.at(1).get<double>()));
        drawn.deltas.push_back(float(row.at(2).get<double>()));
        drawn.deltas.push_back(float(row.at(3).get<double>()));
      }
      target->setDrawn(std::move(drawn));
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

// What the figure's skim makes of these files, as Figure's morph details job does.
fv::pose::MorphFiles skim(const Fig& f, const std::vector<std::string>& stems)
{
  fv::pose::MorphFiles out;
  for(const std::string& stem : stems)
  {
    const std::string file = "data/Fig/Morphs/" + stem + ".dsf";
    const auto        info = fv::dson::morphInfoFromText(f.source->docs.at(file)->json.dump());
    EXPECT_TRUE(info);
    out.add(file, stem, *info);
  }
  return out;
}

void list(Fig& f, const std::string& stem)
{
  f.morphs->addLazy(stem, f.driver->loader("data/Fig/Morphs/" + stem + ".dsf", stem));
}

bool wasRead(const Fig& f, const std::string& stem)
{
  const auto loaded = f.source->filesLoaded();
  return std::find(loaded.begin(), loaded.end(), "data/Fig/Morphs/" + stem + ".dsf") != loaded.end();
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

// A switch that defaults on passes what it multiplies through, its file unread. Turned off it stops it
// and is read for the first time; a reset puts it back on.
TEST(FigureDriver, AMultSwitchAtItsRestPassesItsReaderThrough)
{
  const auto read = [](const Fig& f, const char* file) {
    const auto loaded = f.source->filesLoaded();
    return std::find(loaded.begin(), loaded.end(), file) != loaded.end();
  };
  auto f = setup();
  f->morphs->addLazy("Smile", f->driver->loader("data/Fig/Morphs/Smile.dsf", "Smile"));
  f->morphs->addLazy("Power", f->driver->loader("data/Fig/Morphs/Power.dsf", "Power"));
  fv::dson::MorphInfo info;
  info.rest = 1;
  EXPECT_TRUE(f->morphs->describe("Power", info));
  f->morphs->settled();
  EXPECT_FALSE(read(*f, "data/Fig/Morphs/Power.dsf"));

  f->morphs->set("Smile", 0.5);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Lips"), 0.5, 1e-6);
  EXPECT_NEAR(f->y(1), 11, 1e-4);
  EXPECT_FALSE(read(*f, "data/Fig/Morphs/Power.dsf"));
  EXPECT_DOUBLE_EQ(f->morphs->get("Power"), 1);

  f->morphs->set("Power", 0);
  f->morphs->settled();
  EXPECT_TRUE(read(*f, "data/Fig/Morphs/Power.dsf"));
  EXPECT_NEAR(f->morphs->weightOf("Lips"), 0, 1e-6);
  EXPECT_NEAR(f->y(1), 10, 1e-4);

  f->morphs->set("Power", 0.5);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Lips"), 0.25, 1e-6);

  f->morphs->resetAll();
  f->morphs->settled();
  EXPECT_DOUBLE_EQ(f->morphs->get("Power"), 1);
  EXPECT_DOUBLE_EQ(f->morphs->get("Smile"), 0);
  EXPECT_NEAR(f->morphs->weightOf("Lips"), 0, 1e-6);
  f->morphs->set("Smile", 1);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Lips"), 1, 1e-6);
}

// A switch whose id isn't its file's name (body_basejointcorrectives in
// body_ctrl_basejointcorrectives.dsf) can't be read through its dial until a skim of the file or the
// file itself says the two are one. Turned straight down to 0 it has to be read all the same, or its
// readers never hear of it.
TEST(FigureDriver, ASwitchNamedOtherThanItsFileTurnsOffAtNothing)
{
  auto f = setup();
  f->morphs->addLazy("Brow", f->driver->loader("data/Fig/Morphs/Brow.dsf", "Brow"));
  f->morphs->addLazy("ctrl_Switch", f->driver->loader("data/Fig/Morphs/ctrl_Switch.dsf", "ctrl_Switch"));
  fv::dson::MorphInfo info;
  info.id   = "Switch";
  info.rest = 1;
  f->morphs->describe("ctrl_Switch", info);
  f->morphs->set("Brow", 0.5);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Brow"), 0.5, 1e-6);

  f->morphs->set("ctrl_Switch", 0);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Brow"), 0, 1e-6);
  EXPECT_NEAR(f->y(1), 10, 1e-4);

  f->morphs->set("ctrl_Switch", 1);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Brow"), 0.5, 1e-6);
}

// A dial with no formulas drives a morph whose own file reads it, and nothing else would read that file.
TEST(FigureDriver, ADialBringsAMorphWhoseOwnFileReadsIt)
{
  auto f = setup();
  list(*f, "Neck");
  list(*f, "Swallow");
  f->driver->noteFiles(skim(*f, {"Neck", "Swallow"}));
  EXPECT_FALSE(f->driver->loadReaders());
  EXPECT_FALSE(wasRead(*f, "Swallow"));

  f->morphs->set("Neck", 0.5);
  f->morphs->settled();
  EXPECT_FALSE(wasRead(*f, "Swallow"));
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Swallow"), 0.5, 1e-6);
  EXPECT_NEAR(f->y(1), 11, 1e-4);
  // still the listing's dial, at its own value
  EXPECT_FALSE(f->morphs->channel("Swallow")->hidden);
  EXPECT_DOUBLE_EQ(f->morphs->get("Swallow"), 0);

  // read once, and it follows from here on
  EXPECT_FALSE(f->driver->loadReaders());
  f->morphs->set("Neck", 1);
  f->morphs->settled();
  EXPECT_NEAR(f->y(1), 12, 1e-4);
}

// A listing leaves out a hidden name (facs_cbs_*), so there's no channel to load until one is made.
TEST(FigureDriver, ADialBringsAReaderOfAHiddenName)
{
  auto f = setup();
  list(*f, "Neck");
  f->driver->noteFiles(skim(*f, {"facs_cbs_Gulp"}));
  f->morphs->set("Neck", 0.5);
  f->morphs->settled();
  EXPECT_FALSE(f->morphs->has("facs_cbs_Gulp"));

  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  const fv::morph::Channel* gulp = f->morphs->channel("facs_cbs_Gulp");
  ASSERT_NE(gulp, nullptr);
  EXPECT_TRUE(gulp->hidden);
  EXPECT_EQ(gulp->group, "Driven");
  EXPECT_NEAR(f->morphs->weightOf("facs_cbs_Gulp"), 0.5, 1e-6);
  EXPECT_NEAR(f->y(1), 11, 1e-4);
}

// Formulas go by ids and the listing by file names. The dial is found before its own file has been
// read, and the reader loads as the listing's dial.
TEST(FigureDriver, ADialBringsItsReaderWhenNeitherIdIsItsFilesName)
{
  auto f = setup();
  list(*f, "ctrl_Lift");
  list(*f, "bs_Raise");
  f->driver->noteFiles(skim(*f, {"ctrl_Lift", "bs_Raise"}));
  f->morphs->set("ctrl_Lift", 0.5);
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_FALSE(f->morphs->has("Raise"));
  EXPECT_NEAR(f->morphs->weightOf("bs_Raise"), 0.5, 1e-6);
  EXPECT_NEAR(f->y(1), 11, 1e-4);
}

// A scene's dial goes by the id. Its file loads as that one, or the listing's would take the id from it.
TEST(FigureDriver, AReaderASceneNamedLoadsUnderItsId)
{
  auto f = setup();
  list(*f, "ctrl_Lift");
  list(*f, "bs_Raise");
  f->morphs->addLazy("Raise", f->driver->loader("data/Fig/Morphs/bs_Raise.dsf", "Raise"));
  f->driver->noteFiles(skim(*f, {"ctrl_Lift", "bs_Raise"}));
  f->morphs->set("ctrl_Lift", 0.5);
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_EQ(f->driver->nameOf("Raise"), "Raise");
  EXPECT_NEAR(f->morphs->weightOf("Raise"), 0.5, 1e-6);
  EXPECT_TRUE(f->morphs->channel("bs_Raise")->lazy);
}

// A multiplier only scales what something else moved: moving it has nothing to bring.
TEST(FigureDriver, AMultOnlyReaderIsNotLoaded)
{
  auto f = setup();
  list(*f, "Power");
  f->driver->noteFiles(skim(*f, {"Lips"}));
  f->morphs->set("Power", 0.5);
  f->morphs->settled();
  EXPECT_FALSE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_FALSE(wasRead(*f, "Lips"));
}

// Standing at its default is no move, whatever the default: 130 corrective files read a switch that
// rests at 1. Off it, the readers come.
TEST(FigureDriver, ADialAtItsRestBringsNoReader)
{
  auto f = setup();
  list(*f, "Neck");
  list(*f, "Swallow");
  f->driver->noteFiles(skim(*f, {"Swallow"}));
  fv::dson::MorphInfo info;
  info.rest = 1;
  f->morphs->describe("Neck", info);
  f->morphs->settled();
  EXPECT_DOUBLE_EQ(f->morphs->get("Neck"), 1);
  EXPECT_FALSE(f->driver->loadReaders());

  f->morphs->set("Neck", 0);
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_TRUE(wasRead(*f, "Swallow"));
}

// A morph a controller moves has moved, though its own dial still says 0.
TEST(FigureDriver, ADrivenMorphBringsItsReaders)
{
  auto f = setup();
  list(*f, "Smile");
  f->driver->noteFiles(skim(*f, {"Pout"}));
  f->morphs->set("Smile", 0.5);
  f->morphs->settled();
  EXPECT_DOUBLE_EQ(f->morphs->get("Lips"), 0);
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Pout"), 0.5, 1e-6);
}

// Most files that read a dial are one character's, multiplied by its body. Moving the dial reads them on
// any figure, and they stand at 0 until the body is dialled in, whose own file hasn't been read.
TEST(FigureDriver, AReaderMultipliedByAnUnreadDialStaysAtZero)
{
  auto f = setup();
  list(*f, "Neck");
  list(*f, "Gate Body");
  list(*f, "Gated");
  f->driver->noteFiles(skim(*f, {"Neck", "Gate Body", "Gated"}));
  f->morphs->set("Neck", 1);
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_TRUE(wasRead(*f, "Gated"));
  EXPECT_FALSE(wasRead(*f, "Gate Body"));
  EXPECT_NEAR(f->morphs->weightOf("Gated"), 0, 1e-6);
  EXPECT_NEAR(f->y(1), 10, 1e-4);

  f->morphs->set("Gate Body", 0.5);
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("Gated"), 0.5, 1e-6);
  EXPECT_NEAR(f->y(1), 11, 1e-4);
}

TEST(FigureDriver, AReaderMultipliedByADialWithNoChannelStaysAtItsRest)
{
  auto f = setup();
  list(*f, "Neck");
  f->driver->noteFiles(skim(*f, {"Neck", "Gate_HD", "GatedHd"}));
  EXPECT_FALSE(f->morphs->has("Gate_HD"));
  f->morphs->set("Neck", 1);
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("GatedHd"), 0, 1e-6);

  // the file's default is what it reads at: one that rests on passes the morph through
  auto files = skim(*f, {"Neck", "Gate_HD", "GatedHd"});
  files.dials.at("Gate_HD").rest = 1;
  f->driver->noteFiles(std::move(files));
  EXPECT_NEAR(f->morphs->weightOf("GatedHd"), 1, 1e-6);
}

// A switch that rests on, read through the listing's dial before anything has read its file: what the
// skim says of it replaces the guess, and it does as it's told without its file being read.
TEST(FigureDriver, AnUnreadSwitchIsReadThroughTheListingsDial)
{
  auto f = setup();
  list(*f, "Brow");
  list(*f, "ctrl_Switch");
  auto info = fv::dson::morphInfoFromText(f->source->docs.at("data/Fig/Morphs/ctrl_Switch.dsf")->json.dump());
  ASSERT_TRUE(info);
  f->morphs->describe("ctrl_Switch", *info);
  f->driver->noteFiles(skim(*f, {"ctrl_Switch"}));
  f->morphs->set("Brow", 0.5);
  f->morphs->settled();
  EXPECT_FALSE(wasRead(*f, "ctrl_Switch"));
  EXPECT_NEAR(f->morphs->weightOf("Brow"), 0.5, 1e-6);
}

// A scene stores the joints where all its formulas put them, those of files read after it opened too.
// What such a file gives at the scene's own dials has moved them already; a dial moved since hasn't.
TEST(FigureDriver, AReaderArrivingAfterTheBaselineCountsFromTheStoredCentres)
{
  auto f = setup();
  list(*f, "Char");
  list(*f, "CharBody");
  f->driver->noteFiles(skim(*f, {"CharBody"}));
  f->morphs->set("Char", 1);
  f->morphs->settled();
  f->morphs->markLoaded();
  f->driver->captureBaseline();
  f->driver->update();

  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_NEAR(f->morphs->weightOf("CharBody"), 1, 1e-6);
  EXPECT_NEAR(f->y(1), 12, 1e-4);
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 15, 1e-4);

  f->morphs->set("Char", 0);
  f->morphs->settled();
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 10, 1e-4);
}

TEST(FigureDriver, AReaderOfADialMovedSinceTheSceneOpenedMovesTheJoint)
{
  auto f = setup();
  list(*f, "Char");
  list(*f, "CharBody");
  f->driver->noteFiles(skim(*f, {"CharBody"}));
  f->morphs->markLoaded();
  f->driver->captureBaseline();

  f->morphs->set("Char", 1);
  EXPECT_TRUE(f->driver->loadReaders());
  f->morphs->settled();
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 20, 1e-4);
}

TEST(FigureDriver, BaselineCountsOffsetsFromCentresTheSceneStored)
{
  auto f = setup();
  f->morphs->set("Shape", 1);
  f->morphs->settled();
  // as if the scene file had stored the moved centre: nothing moves now, and the shape off moves it back
  f->pose->setCenterOffsets({});
  f->morphs->markLoaded();
  f->driver->captureBaseline();
  f->driver->update();
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 15, 1e-4);
  f->morphs->set("Shape", 0);
  f->morphs->settled();
  EXPECT_NEAR(f->scene.nodes[f->head].position.y, 10, 1e-4);
}
