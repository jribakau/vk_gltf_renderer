// Preset files are recognised before they are loaded as models, and a pose preset's last keys
// drive bones and morph dials. A port of describePreset (src/lib/loaders/dson/presets.ts) and
// PoseRig.applyPreset.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>

#include <figure/dson/material_preset.hpp>
#include <figure/dson/materials.hpp>
#include <figure/dson/presets.hpp>
#include <figure/dson/script_preset.hpp>
#include <figure/dson/rig.hpp>
#include <figure/figure.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_rig.hpp>

namespace {

const char* kNodes = R"([
  {"id": "hip", "type": "figure", "name": "hip", "center_point": [{"id": "y", "value": 0}],
   "rotation": [{"id": "x", "min": -180, "max": 180}, {"id": "y", "min": -180, "max": 180}, {"id": "z", "min": -180, "max": 180}]},
  {"id": "head", "type": "bone", "name": "head", "parent": "#hip", "center_point": [{"id": "y", "value": 15}],
   "rotation": [{"id": "x", "min": -180, "max": 180}, {"id": "y", "min": -180, "max": 180}, {"id": "z", "min": -180, "max": 180}]}
])";

struct Rig
{
  fv::SceneGraph                     scene;
  std::unique_ptr<fv::pose::PoseRig> pose;
  std::unique_ptr<fv::morph::MorphSet> morphs;
};

Rig rig()
{
  Rig out;
  auto built = fv::dson::buildRig(out.scene, fv::dson::Json::parse(kNodes), nullptr);
  EXPECT_TRUE(built);
  out.pose   = std::make_unique<fv::pose::PoseRig>(out.scene, *built, std::vector<const fv::dson::Rig*>{});
  out.morphs = std::make_unique<fv::morph::MorphSet>([](int) { return static_cast<fv::mesh::BuiltMesh*>(nullptr); });
  out.morphs->addLazy("Smile", []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; });
  return out;
}

}  // namespace

TEST(Presets, namesAPoseAndLeavesAModelAlone)
{
  auto pose = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"preset_pose"},"scene":{"nodes":[]}})"));
  ASSERT_TRUE(pose);
  EXPECT_EQ(pose->kind, "pose preset");

  auto model = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"figure"},"geometry_library":[{}],"scene":{"nodes":[]}})"));
  EXPECT_FALSE(model.has_value());
}

// FeSoul's hair colours are shader presets that set channels; DAZ's "!Iray Uber Base" only names
// the shader. The first goes on surfaces as a material preset does.
TEST(Presets, AShaderPresetThatSetsValuesSetsMaterials)
{
  auto colour = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"preset_shader"},"scene":{
    "materials":[{"url":"/data/x.dsf#OmniHair"}],
    "animations":[{"url":"name://@selection#materials:?extra/studio_material_channels/channels/Melanin/value","keys":[[0,0.9]]}]}})"));
  ASSERT_TRUE(colour);
  EXPECT_EQ(colour->kind, "shader preset");
  EXPECT_TRUE(colour->setsMaterials);
  auto bare = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"preset_shader"},"scene":{"materials":[{"url":"#Default"}]}})"));
  ASSERT_TRUE(bare);
  EXPECT_FALSE(bare->setsMaterials);
  // a tattoo or a makeup: a stack of images on a surface's channels
  auto layered = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"preset_layered_image"},"scene":{"animations":[
    {"url":"name://@selection#materials/Head:?diffuse/image","keys":[[0,"#Look"]]}]}})"));
  ASSERT_TRUE(layered);
  EXPECT_EQ(layered->kind, "layered image preset");
  EXPECT_TRUE(layered->setsMaterials);
  auto noSurface = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"preset_layered_image"},"scene":{}})"));
  ASSERT_TRUE(noSurface);
  EXPECT_FALSE(noSurface->setsMaterials);

  auto material = fv::dson::describePreset(fv::dson::Json::parse(R"({"asset_info":{"type":"preset_material"},"scene":{}})"));
  ASSERT_TRUE(material);
  EXPECT_TRUE(material->setsMaterials);
}

TEST(Presets, readsAnEnvironmentAndGuessesCamerasAndLights)
{
  auto env = fv::dson::describePreset(fv::dson::Json::parse(R"({
    "asset_info": {"type": "preset_render_settings"},
    "scene": {"extra": [{"type": "studio_render_settings", "render_options": {"render_elements": [{
      "channels": [
        {"channel": {"id": "Environment Map", "image_file": "/Runtime/a.hdr"}},
        {"channel": {"id": "Environment Intensity", "current_value": 1.5}},
        {"channel": {"id": "Dome Rotation", "value": 30}},
        {"channel": {"id": "Draw Dome", "current_value": true}}
      ]
    }]}}]}
  })"));
  ASSERT_TRUE(env);
  EXPECT_EQ(env->kind, "render settings preset");
  ASSERT_TRUE(env->environment);
  EXPECT_EQ(env->environment->image, "/Runtime/a.hdr");
  EXPECT_EQ(env->environment->intensity, 1.5);
  EXPECT_EQ(env->environment->rotation, 30);
  EXPECT_EQ(env->environment->drawDome, true);

  auto cameras = fv::dson::describePreset(fv::dson::Json::parse(R"({"scene":{"nodes":[{"id":"Camera 1","url":"/data/camera"}]}})"));
  ASSERT_TRUE(cameras);
  EXPECT_EQ(cameras->kind, "camera preset");

  auto lights = fv::dson::describePreset(fv::dson::Json::parse(R"({"scene":{"nodes":[{"extra":[{"type":"studio/light"}]}]}})"));
  ASSERT_TRUE(lights);
  EXPECT_EQ(lights->kind, "light preset");

  auto unknown = fv::dson::describePreset(fv::dson::Json::parse(R"({"scene":{"nodes":[]}})"));
  ASSERT_TRUE(unknown);
  EXPECT_EQ(unknown->kind, "preset with no geometry");
}

TEST(Presets, appliesTheThreeUrlFormsTheLastKeyAndSkipsTheRest)
{
  Rig  g = rig();
  auto doc = fv::dson::Json::parse(R"({
    "scene": {"animations": [
      {"url": "name://@selection/head:?rotation/x/value", "keys": [[0, 0], [1, 30]]},
      {"url": "name://@selection#Smile:?value/value", "keys": [[0, 0.4]]},
      {"url": "hip:/data/Fig.dsf#head?rotation/y/value", "keys": [[0, 1], [1, 12]]},
      {"url": "name://@selection/missing:?rotation/z/value", "keys": [[1, 5]]},
      {"url": "not a url", "keys": [[1, 1]]}
    ]}
  })");
  const fv::PosePresetResult r = fv::applyPosePreset(g.pose.get(), g.morphs.get(), doc);
  EXPECT_EQ(r.applied, 3);
  ASSERT_EQ(r.skipped.size(), 2u);
  EXPECT_EQ(r.skipped[0], "missing rotation/z");
  EXPECT_EQ(r.skipped[1], "not a url");
  EXPECT_EQ(g.pose->posed("head", "rotation", "x"), 30);
  EXPECT_EQ(g.pose->posed("head", "rotation", "y"), 12);
  EXPECT_EQ(g.morphs->get("Smile"), 0.4);

  const fv::PosePresetResult none = g.pose ? fv::applyPosePreset(nullptr, g.morphs.get(), doc) : fv::PosePresetResult{};
  EXPECT_EQ(none.applied, 0);
}

TEST(Presets, aShapeOrPropertyPresetReachesNamedNodesAndReportsVisibility)
{
  Rig g = rig();
  g.morphs->addLazy("Anus Down@w1", []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; });
  auto doc = fv::dson::Json::parse(R"({
    "scene": {"animations": [
      {"url": "name://Anatomy_1006#Anus%20Down:?value/value", "keys": [[0, 0.7]]},
      {"url": "name://Anatomy_1006#Smile:?value/value", "keys": [[0, 0.2]]},
      {"url": "name://Anatomy_1006/head:?rotation/x/value", "keys": [[0, 40]]},
      {"url": "name://@selection/Anatomy_1006:?extra/studio_node_channels/channels/Visible/value", "keys": [[0, false]]},
      {"url": "name://@selection:?extra/studio_node_channels/channels/Visible/value", "keys": [[0, true]]}
    ]}
  })");
  std::vector<std::pair<std::string, std::string>> asked;
  const fv::PosePresetResult r = fv::applyPosePreset(g.pose.get(), g.morphs.get(), doc, [&](const std::string& node, const std::string& id) {
    asked.emplace_back(node, id);
    return node == "Anatomy_1006" && id == "Anus Down" ? std::string("Anus Down@w1") : std::string();
  });
  // the item's dial; not the figure's own Smile, which the url doesn't name
  EXPECT_EQ(g.morphs->get("Anus Down@w1"), 0.7);
  EXPECT_EQ(g.morphs->get("Smile"), 0.0);
  ASSERT_EQ(r.skipped.size(), 1u);
  EXPECT_EQ(r.skipped[0], "Anatomy_1006 Smile");
  // another node's bone is left to its skeleton, which follows the figure's
  EXPECT_EQ(g.pose->posed("head", "rotation", "x").value_or(0), 0);
  EXPECT_EQ(r.applied, 1);
  // a boolean key: the named node hidden; the selection itself has no node to report
  ASSERT_EQ(r.visibility.size(), 1u);
  EXPECT_EQ(r.visibility[0], std::make_pair(std::string("Anatomy_1006"), false));
  EXPECT_EQ(asked.size(), 2u);
}

// ---- DAZ Studio's script material presets (.dsa), read as data

namespace {

// As DAZ's preset saver writes them (the AMG GT Coupe's LIGHTS GAUGES ON, cut down): a call outside
// the switch, a call with a name for an argument, an option by its text, two cases sharing a block
const char* kScriptPreset = R"(// DAZ Studio version 4.12.0.86 filetype Materials Preset

var g_oPresetHelper = new DzPresetHelper;
DsActions.prototype.begin = function()
{
	g_oPresetHelper.setMapSetting( DzPresetHelper.Replace );
	g_oPresetHelper.setColorProperty( "Diffuse Color", [ 1, 2, 3 ] );
}
DsActions.prototype.setMaterialProperties = function( oMaterial, oShape ){
	this.m_oElement = oMaterial;
	var sID = this.m_oElement.name;
	if( oShape != undefined ){
		sID += "_" + oShape.getLabel();
	}
	switch( sID ){
			case "gauges":
				this.m_sMaterialType = "DzUberIrayMaterial";
				this.m_sMaterialName = "Iray Uber";
				this.prepareMaterial();
				g_oPresetHelper.setTargetElement( this.m_oElement );
				g_oPresetHelper.setPropertyWithString( "Tags", "" );
				g_oPresetHelper.setNumericProperty( "UV Set",  [ "Default UVs" ] );
				g_oPresetHelper.setColorProperty( "Diffuse Color",  [ 255, 255, 255 ],
					"/Runtime/Textures/SamGrey/Car 2026/Car_2.png" );
				g_oPresetHelper.setNumericPropertyWithAttributes( "Glossy Roughness", true, 0, 1, [ 0.25 ] );
				g_oPresetHelper.setColorProperty( "Emission Color",  [ 205, 247, 255 ],
					"/Runtime/Textures/SamGrey/Car 2026/Car_2.png" );
				g_oPresetHelper.setNumericProperty( "Two Sided Light",  [ 1 ] );
				g_oPresetHelper.setNumericProperty( "Luminance",  [ 2500 ] );
				g_oPresetHelper.setNumericProperty( "Luminance Units",  [ "lm" ] );
				g_oPresetHelper.setNumericProperty( "Base Mixing",  [ "Weighted" ] );
				g_oPresetHelper.setNumericPropertyWithAttributes( "Normal Map", true, 0, 1, [ 1 ] );
				break;
			case "lamp left":
			case "lamp right":
				g_oPresetHelper.setColorProperty( "Emission Color",  [ 0, 0, 0 ] );
				break;
		default:
			break;
	}
}

g_bCONTROL_PRESSED ? g_oGui.doDialog() : g_oGui.doNoDialog();
)";

// the value or map an animation of `doc` sets at `url`, null when none does
fv::dson::Json keyAt(const fv::dson::Json& doc, const std::string& url)
{
  for(const auto& a : doc["scene"]["animations"])
    if(a["url"] == url)
      return a["keys"][0][1];
  return nullptr;
}

const std::string kGauges = "name://@selection#materials/gauges:?";

}  // namespace

TEST(ScriptPreset, ReadsTheCasesAsAMaterialPreset)
{
  ASSERT_TRUE(fv::dson::isScriptMaterialsPreset(kScriptPreset));
  const auto doc = fv::dson::scriptMaterialsPreset(kScriptPreset);
  ASSERT_TRUE(doc);
  const auto info = fv::dson::describePreset(*doc);
  ASSERT_TRUE(info);
  EXPECT_EQ(info->kind, "material preset");
  EXPECT_TRUE(info->setsMaterials);
  // colours from 0-255, the base colour by the .duf files' id, its map as they write paths
  EXPECT_EQ(keyAt(*doc, kGauges + "diffuse/value"), fv::dson::Json::parse("[1.0, 1.0, 1.0]"));
  EXPECT_EQ(keyAt(*doc, kGauges + "diffuse/image_file"), "/Runtime/Textures/SamGrey/Car%202026/Car_2.png");
  const auto emission = keyAt(*doc, kGauges + "Emission%20Color/value");
  ASSERT_TRUE(emission.is_array());
  EXPECT_NEAR(emission[0].get<double>(), 205.0 / 255.0, 1e-9);
  EXPECT_NEAR(emission[1].get<double>(), 247.0 / 255.0, 1e-9);
  EXPECT_EQ(keyAt(*doc, kGauges + "Glossy%20Roughness/value"), 0.25);
  // options by their index among the shader's values; a UV set the scene names is left out
  EXPECT_EQ(keyAt(*doc, kGauges + "Luminance%20Units/value"), 4);
  EXPECT_EQ(keyAt(*doc, kGauges + "Base%20Mixing/value"), 2);
  EXPECT_TRUE(keyAt(*doc, kGauges + "UV%20Set/value").is_null());
  // a mappable property set without a map loses its own (DAZ's default, Replace)
  EXPECT_EQ(keyAt(*doc, kGauges + "Normal%20Map/image_file"), "");
  // both labels of a shared block; nothing from outside the switch
  for(const char* lamp : {"lamp%20left", "lamp%20right"})
    EXPECT_EQ(keyAt(*doc, std::string("name://@selection#materials/") + lamp + ":?Emission%20Color/value"), fv::dson::Json::parse("[0.0, 0.0, 0.0]"));
  size_t gauges = 0;
  for(const auto& a : (*doc)["scene"]["animations"])
    gauges += a["url"].get<std::string>().rfind(kGauges, 0) == 0 ? 1 : 0;
  // the tags; a value and a map (none but the two named) for the base colour, roughness, emission,
  // two-sided, luminance and the normal map; the units and mixing, options, have no map
  EXPECT_EQ(gauges, 15u);
}

TEST(ScriptPreset, OnlyTheSaversMaterialPresetsAreRead)
{
  // another script, even one that calls the helper, is no preset
  const std::string other = "// DAZ Studio version 4.12 filetype DAZ Script\nswitch(x){ case \"a\": g_oPresetHelper.setNumericProperty(\"Luminance\", [1]); break; }";
  EXPECT_FALSE(fv::dson::isScriptMaterialsPreset(other));
  EXPECT_FALSE(fv::dson::scriptMaterialsPreset(other));
  // a preset whose calls aren't literals sets nothing
  const std::string none = "// DAZ Studio version 4.12 filetype Materials Preset\nDsActions.prototype.setMaterialProperties = function(){ switch( sID ){ case \"a\": "
                           "g_oPresetHelper.setTargetElement( this.m_oElement ); g_oPresetHelper.setNumericProperty( \"Luminance\", [ x ] ); break; } }";
  EXPECT_FALSE(fv::dson::scriptMaterialsPreset(none));
}

// The gauges as the car's .duf has them, dark (Emission Color black at 2500 lm), and a lamp lit:
// the preset lights the one and puts the other out, and leaves the body alone.
TEST(ScriptPreset, LightsTheSurfacesItNames)
{
  const auto material = [](const std::string& name, const char* emission) {
    const fv::dson::Json def = fv::dson::Json::parse(std::string(R"({"diffuse":{"channel":{"id":"diffuse","label":"Base Color","name":"Diffuse Color","current_value":[1,1,1],
        "image_file":"/Runtime/Textures/SamGrey/Car%202026/Car_2.png"}},
      "extra":[{"type":"studio/material/uber_iray"},{"type":"studio_material_channels","channels":[
        {"channel":{"id":"Glossy Roughness","label":"Glossy Roughness","current_value":0}},
        {"channel":{"id":"Normal Map","label":"Normal Map","current_value":1,"image_file":"/n.png"}},
        {"channel":{"id":"Emission Color","label":"Emission Color","current_value":)")
                                                     + emission + R"(}},
        {"channel":{"id":"Two Sided Light","label":"Two Sided Light","current_value":true}},
        {"channel":{"id":"Luminance","label":"Luminance","current_value":2500}},
        {"channel":{"id":"Luminance Units","label":"Luminance Units","current_value":4}}]}]})");
    fv::dson::Channels ch;
    fv::dson::collectChannels(&def, ch, {}, "car.duf");
    return fv::dson::buildMaterial(name, ch);
  };
  fv::SceneGraph scene;
  scene.materials.push_back(material("gauges", "[0,0,0]"));
  scene.materials.push_back(material("lamp left", "[1,1,1]"));
  scene.materials.push_back(material("body", "[1,1,1]"));
  ASSERT_FALSE(scene.materials[0].surface.emission);
  ASSERT_TRUE(scene.materials[1].surface.emission);
  const auto r = fv::dson::applyMaterialPreset(*fv::dson::scriptMaterialsPreset(kScriptPreset), scene);
  EXPECT_EQ(r.changed, std::set<int>({0, 1}));
  EXPECT_EQ(r.missing, std::set<std::string>({"lamp right"}));
  const auto& gauges = scene.materials[0].surface;
  ASSERT_TRUE(gauges.emission);
  // the colour as a .duf giving it in 0-1 makes it
  const fv::mat::Surface asDuf = material("ref", "[0.803921568627451, 0.968627450980392, 1.0]").surface;
  ASSERT_TRUE(asDuf.emission);
  for(int c = 0; c < 3; c++)
    EXPECT_NEAR(gauges.emission->color.color[c], asDuf.emission->color.color[c], 1e-6f);
  EXPECT_EQ(gauges.emission->color.map, "/Runtime/Textures/SamGrey/Car%202026/Car_2.png");
  ASSERT_TRUE(gauges.emission->luminance);
  EXPECT_FLOAT_EQ(gauges.emission->luminance->amount, 2500.0f);
  EXPECT_TRUE(gauges.emission->luminance->perArea);  // lumens: a flux over the surface
  EXPECT_TRUE(gauges.emission->luminance->twoSided);
  EXPECT_NEAR(gauges.glossy.roughness.value, 0.25f, 1e-6f);
  EXPECT_TRUE(gauges.normalMap.empty());
  EXPECT_FALSE(scene.materials[1].surface.emission);
  EXPECT_TRUE(scene.materials[2].surface.emission);
}

// The store reads a .dsa as the .duf it stands for, and names any other script
TEST(ScriptPreset, TheStoreOpensItAndNoOtherScript)
{
  namespace fs        = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "fv-script-preset-test";
  fs::remove_all(root);
  fs::create_directories(root / "Props/Car/Lights");
  std::ofstream(root / "Props/Car/Lights/Car_GAUGES ON.dsa", std::ios::binary) << kScriptPreset;
  std::ofstream(root / "Props/Car/Lights/Tool.dsa", std::ios::binary) << "// DAZ Studio version 4.12 filetype DAZ Script\nprint(1);\n";
  {
    fv::dson::Store store(std::make_shared<fv::dson::Library>(std::vector<fs::path>{root}));
    const auto      doc = store.load("Props/Car/Lights/Car_GAUGES ON.dsa");
    ASSERT_TRUE(doc);
    EXPECT_EQ(keyAt(doc->json, kGauges + "Luminance%20Units/value"), 4);
    try
    {
      store.load("Props/Car/Lights/Tool.dsa");
      ADD_FAILURE() << "a script that isn't a material preset loaded";
    }
    catch(const std::exception& e)
    {
      EXPECT_NE(std::string(e.what()).find("Tool.dsa is a DAZ script"), std::string::npos) << e.what();
    }
  }
  fs::remove_all(root);
}
