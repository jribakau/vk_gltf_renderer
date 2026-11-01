// What the tests of the material pipeline share (test_materials*.cpp, test_material_closure.cpp).

#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>
#include <figure/dson/eye_materials.hpp>
#include <figure/dson/legacy_materials.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/dson/materials.hpp>
#include <figure/dson/skin_materials.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/materials/edits_json.hpp>
#include <figure/materials/fabric.hpp>
#include <figure/materials/lobes.hpp>
#include <figure/materials/lower.hpp>
#include <figure/tools/closure_row.hpp>
#include "tinygltf_utils.hpp"

namespace materials_test {

using fv::dson::Json;

// A lowering with its material as glTF's JSON, its slots under "slot": what these tests read it as,
// so a change to how the material is kept doesn't touch what they hold it to.
struct LoweredJson
{
  nlohmann::json                       material;
  std::vector<fv::mat::TextureRequest> requests;
  fv::mat::Effective                   effective;
};

inline LoweredJson loweredJson(const fv::mat::Surface& s, const fv::mat::Edits& e, const fv::mat::LowerOptions& o)
{
  fv::mat::Lowered l = fv::mat::lower(s, e, o);
  return {fv::mat::materialJson(l.material, "slot"), std::move(l.requests), std::move(l.effective)};
}

inline fv::dson::Channels channelsOf(const char* text)
{
  fv::dson::Channels ch;
  const Json         def = Json::parse(text);
  fv::dson::collectChannels(&def, ch, {}, "x.dsf");
  return ch;
}

inline const char* kSkin = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,0.8,0.7],"image_file":"/Runtime/Textures/face_d.jpg"}},
  {"channel":{"id":"Translucency Weight","type":"float","current_value":0.6}},
  {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/face_n.jpg"}},
  {"channel":{"id":"Bump Strength","type":"float","current_value":0.5,"image_file":"/Runtime/Textures/face_b.jpg"}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.4}},
  {"channel":{"id":"Dual Lobe Specular Weight","type":"float","current_value":0.5}},
  {"channel":{"id":"Specular Lobe 1 Roughness","type":"float","current_value":0.2}},
  {"channel":{"id":"Specular Lobe 2 Roughness","type":"float","current_value":0.5}},
  {"channel":{"id":"Dual Lobe Specular Ratio","type":"float","current_value":0.5}}
]}]})";

inline const char* kEyeMoisture = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[0.5,0.5,0.5]}},
  {"channel":{"id":"Refraction Weight","type":"float","current_value":1}},
  {"channel":{"id":"Refraction Index","type":"float","current_value":1.376}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.3}}
]}]})";

// Genesis 9's eye as Addy Florence's eyes give it: a thin-walled moisture shell over the eye with
// Iray's glossy weight 0.40 and the iris's normal map, and the eye under it with a mapped dual lobe.
inline const char* kG9Moisture = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1]}},
  {"channel":{"id":"Refraction Weight","type":"float","current_value":1}},
  {"channel":{"id":"Refraction Index","type":"float","current_value":1.38}},
  {"channel":{"id":"Thin Walled","type":"bool","current_value":true}},
  {"channel":{"id":"Glossy Layered Weight","type":"float","current_value":0.4016393}},
  {"channel":{"id":"Glossy Reflectivity","type":"float","current_value":0.5}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0}},
  {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/eyes_nm.jpg"}}
]}]})";

inline const char* kG9Eye = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/eyes_d.jpg"}},
  {"channel":{"id":"Translucency Weight","type":"float","current_value":0.8}},
  {"channel":{"id":"Dual Lobe Specular Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Dual Lobe Specular Weight","type":"float","current_value":1,"image_file":"/Runtime/Textures/eyes_s.jpg"}},
  {"channel":{"id":"Dual Lobe Specular Reflectivity","type":"float","current_value":0.5}},
  {"channel":{"id":"Dual Lobe Specular Ratio","type":"float","current_value":0.15}},
  {"channel":{"id":"Specular Lobe 1 Roughness","type":"float","current_value":0.6}},
  {"channel":{"id":"Specular Lobe 2 Roughness Mult","type":"float","current_value":0.4}},
  {"channel":{"id":"Bump Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/eyes_nm.jpg"}}
]}]})";

// Two worn items under a body group, each a figure with its own surfaces: "Eyelashes Upper" and
// "Eyelashes Lower" on the lashes, "Base" on the hair, and a "Base" of the body's own.
inline fv::SceneGraph wornScene()
{
  fv::SceneGraph scene;
  const auto     addFigure = [&](const std::string& dson, int parent, const std::vector<std::string>& surfaces) {
    fv::Mesh mesh;
    for(const std::string& s : surfaces)
    {
      mesh.materials.push_back(int(scene.materials.size()));
      scene.materials.push_back(fv::dson::buildMaterial(s, channelsOf(kSkin)));
    }
    scene.meshes.push_back(mesh);
    fv::Node group;
    group.dsonNode = dson;
    group.parent   = parent;
    scene.nodes.push_back(group);
    const int groupId = int(scene.nodes.size()) - 1;
    fv::Node shape;
    shape.parent = groupId;
    shape.mesh   = int(scene.meshes.size()) - 1;
    scene.nodes.push_back(shape);
    return groupId;
  };
  const int body = addFigure("Genesis9", -1, {"Body", "Base"});
  addFigure("Genesis9Eyelashes", body, {"Eyelashes Upper", "Eyelashes Lower"});
  addFigure("FEMessyHair", body, {"Base", "Transition"});
  return scene;
}

// OmniHair at FE Messy High Ponytail's defaults: Custom melanin 0.5 at redness 0.5, undyed.
inline std::string omniHair(const std::string& more = "")
{
  return R"({"extra":[{"type":"studio/material/omnihair","channels":[
    {"channel":{"id":"Melanin Presets","type":"enum","current_value":7}},
    {"channel":{"id":"Melanin","type":"float","current_value":0.5}},
    {"channel":{"id":"Melanin Redness","type":"float","current_value":0.5}},
    {"channel":{"id":"Azimuthal Roughness","type":"float","current_value":0.2}})"
         + more + "]}]}";
}

inline const char* kMadeUp = R"({"extra":[{"type":"studio/material/daz_brick","channels":[
  {"channel":{"id":"Diffuse Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/face_d.jpg"}},
  {"channel":{"id":"Makeup Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Makeup Weight","type":"float","current_value":0.8,"image_file":"/mask.png"}},
  {"channel":{"id":"Makeup Base Color","type":"color","current_value":[1,1,1],"image_file":"/lips.png"}}
]}]})";

using ChannelValues = std::vector<std::pair<std::string, Json>>;

// A surface of `shader`'s channels, each "id": value (a number, bool or [r,g,b]); an image where
// `images` gives one
inline fv::mat::Surface surfaceOf(const char* name, const char* shader, const ChannelValues& values, const std::map<std::string, std::string>& images = {})
{
  Json channels = Json::array();
  for(const auto& [id, value] : values)
  {
    Json c = {{"id", id}, {"type", value.is_array() ? "color" : value.is_boolean() ? "bool" : "float"}, {"current_value", value}};
    if(const auto it = images.find(id); it != images.end())
      c["image_file"] = it->second;
    channels.push_back({{"channel", c}});
  }
  const Json doc = {{"extra", Json::array({Json{{"type", shader}, {"channels", channels}}})}};
  return fv::dson::parseSurface(name, channelsOf(doc.dump().c_str()));
}

inline const char* kPbrSkinType = "studio/material/daz_brick";

inline const char* kUberType    = "studio/material/uber_iray";

inline const nlohmann::json& clearcoatOf(const LoweredJson& l)
{
  return l.material["extensions"]["KHR_materials_clearcoat"];
}

inline const nlohmann::json& secondLobeOf(const LoweredJson& l)
{
  return l.material["extensions"]["FV_materials_specular2"];
}

inline const fv::mat::TextureRequest& requestOf(const LoweredJson& l, const nlohmann::json& info)
{
  return l.requests[info["slot"].get<size_t>()];
}

inline const char* kAoaBrick = R"({"extra":[{"type":"studio/material/daz_brick","channels":[
  {"channel":{"id":"Ambient Color","label":"Subsurface Color","type":"color","current_value":[1,0.8,0.7],"image_file":"/Runtime/Textures/torso_d.jpg"}},
  {"channel":{"id":"Ambient Strength","label":"Subsurface Strength","type":"float","current_value":0.75}},
  {"channel":{"id":"color4","label":"Ambient Color","type":"color","current_value":[0,0,0]}},
  {"channel":{"id":"Ambient Strength2","label":"Ambient Strength","type":"float","current_value":0}},
  {"channel":{"id":"roughness","label":"Velvet Roughness","type":"float","current_value":0.2}},
  {"channel":{"id":"Glossiness","label":"Specular Glossiness","type":"float","current_value":0.65}},
  {"channel":{"id":"sssmaterial","label":"SSS Material Type","type":"enum","current_value":9}}
]}]})";

// Each layer collected over the ones before, as figure_loader merges a material's layers
inline fv::dson::Channels layered(std::initializer_list<const char*> layers)
{
  fv::dson::Channels ch;
  for(const char* text : layers)
  {
    const Json def = Json::parse(text);
    fv::dson::collectChannels(&def, ch, {}, "x.dsf");
  }
  return ch;
}

inline const char* kBoxers = R"({
  "diffuse":{"channel":{"id":"diffuse","label":"Diffuse Color","current_value":[0.7529412,0.7529412,0.7529412],"image_file":"/Runtime/Textures/boxers_d.jpg"}},
  "diffuse_strength":{"channel":{"id":"diffuse_strength","label":"Diffuse Strength","current_value":1}},
  "specular":{"channel":{"id":"specular","label":"Specular Color","current_value":[0.6,0.6,0.6]}},
  "specular_strength":{"channel":{"id":"specular_strength","label":"Specular Strength","current_value":0.3}},
  "glossiness":{"channel":{"id":"glossiness","label":"Glossiness","current_value":1}},
  "ambient":{"channel":{"id":"ambient","label":"Ambient Color","current_value":[0,0,0]}},
  "ambient_strength":{"channel":{"id":"ambient_strength","label":"Ambient Strength","current_value":1}},
  "reflection_strength":{"channel":{"id":"reflection_strength","label":"Reflection Strength","current_value":0}},
  "refraction_strength":{"channel":{"id":"refraction_strength","label":"Refraction Strength","current_value":0}},
  "ior":{"channel":{"id":"ior","label":"Index of Refraction","current_value":0}},
  "bump_min":{"channel":{"id":"bump_min","label":"Negative Bump","current_value":-0.01}},
  "bump_max":{"channel":{"id":"bump_max","label":"Positive Bump","current_value":0.01}},
  "transparency":{"channel":{"id":"transparency","label":"Opacity Strength","current_value":1}},
  "u_scale":{"channel":{"id":"u_scale","label":"Horizontal Tiles","current_value":1}},
  "extra":[{"type":"studio_material_channels","channels":[
    {"channel":{"id":"Lighting Model","label":"Lighting Model","current_value":0}}]}]})";

inline const char* kG2fTorso = R"({
  "diffuse":{"channel":{"id":"diffuse","label":"Diffuse Color","current_value":[1,1,1],"image_file":"/Runtime/Textures/g2f_torso.jpg"}},
  "transparency":{"channel":{"id":"transparency","label":"Opacity Strength","current_value":1}},
  "extra":[{"type":"studio/material/daz_shader","definition":"support/DAZ/Uber/shaderDefinitions/surface/omUberSurfaceDef.dse","channels":[
    {"channel":{"id":"Bump Active","current_value":true}},
    {"channel":{"id":"Bump Strength","current_value":0.25,"image_file":"/Runtime/Textures/g2f_torso_b.jpg"}},
    {"channel":{"id":"Bump Minimum","current_value":-0.1}},
    {"channel":{"id":"Bump Maximum","current_value":0.1}},
    {"channel":{"id":"Diffuse Active","current_value":true}},
    {"channel":{"id":"Diffuse Strength","current_value":0.9}},
    {"channel":{"id":"Opacity Active","current_value":true}},
    {"channel":{"id":"Map Tiling U","current_value":1}},
    {"channel":{"id":"Map Tiling V","current_value":1}},
    {"channel":{"id":"Specular Active","current_value":true}},
    {"channel":{"id":"Specular Color","current_value":[1,0.7882353,0.6627451]}},
    {"channel":{"id":"Specular Strength","current_value":0.55}},
    {"channel":{"id":"Glossiness","current_value":0.2}},
    {"channel":{"id":"Specular2 Active","current_value":true}},
    {"channel":{"id":"Specular2 Color","current_value":[0.745098,0.8627451,1]}},
    {"channel":{"id":"Specular2 Strength","current_value":0.3}},
    {"channel":{"id":"Specular2 Roughness","current_value":0.1}},
    {"channel":{"id":"Ambient Active","current_value":false}},
    {"channel":{"id":"Refraction Active","current_value":false}},
    {"channel":{"id":"Index of Refraction","current_value":1.3}},
    {"channel":{"id":"Reflection Active","current_value":false}},
    {"channel":{"id":"Reflection Strength","current_value":1}},
    {"channel":{"id":"Velvet Active","current_value":true}},
    {"channel":{"id":"Subsurface Active","current_value":true}},
    {"channel":{"id":"Subsurface Strength","current_value":0.5}}]}]})";

// The same surface switched to plain cloth: no subsurface, no bump, tiled twice across
inline const char* kUberCloth = R"({"extra":[{"channels":[
    {"channel":{"id":"Subsurface Active","current_value":false}},
    {"channel":{"id":"Bump Active","current_value":false}},
    {"channel":{"id":"Map Tiling U","current_value":2}},
    {"channel":{"id":"Map Tiling V","current_value":2}}]}]})";

inline const char* kTopValues = R"({"extra":[{"channels":[
    {"channel":{"id":"specular_strength","current_value":1}},{"channel":{"id":"glossiness","current_value":0.75}}]}]})";

inline const char* kMetallic  = R"({"extra":[{"channels":[{"channel":{"id":"Lighting Model","current_value":1}}]}]})";

inline const char* kRefracts  = R"({"extra":[{"channels":[{"channel":{"id":"refraction_strength","current_value":0.8}}]}]})";

inline const char* kUberLens  = R"({"transparency":{"channel":{"id":"transparency","current_value":0.1}},
    "extra":[{"channels":[{"channel":{"id":"Refraction Active","current_value":true}}]}]})";

inline const char* kPbrSkinFile = "data/Daz 3D/Built-in Content/Daz Iray PBRSkin/PBRSkin.dsf";

inline fv::dson::Channels collect(fv::dson::Channels ch, const char* text, const std::string& file)
{
  const Json def = Json::parse(text);
  fv::dson::collectChannels(&def, ch, {}, file);
  return ch;
}

// The G8M Basic Wear Boxers' two surfaces as the item's own scene layer sets them: DAZ Default,
// the Black textures
inline const char* kBoxersBlackMaps = R"({
  "diffuse":{"channel":{"id":"diffuse","current_value":[1,1,1],"image_file":"/Runtime/Textures/ABen/Basic%20Wear/Black/Boxers_BaseColor.jpg"}},
  "normal":{"channel":{"id":"normal","current_value":1,"image_file":"/Runtime/Textures/ABen/Basic%20Wear/Black/Boxers_Normal.jpg"}}})";

inline fv::SceneGraph boxersScene()
{
  fv::SceneGraph scene;
  for(const char* name : {"Boxers", "BoxersWaistband"})
    scene.materials.push_back(fv::dson::buildMaterial(name, layered({kBoxers, kBoxersBlackMaps})));
  return scene;
}

// A material preset on the Boxers' surfaces: `channels` values each (the first an Iray one unless
// `threeDelight`), the colour's maps unless `folder` is empty, and a dozen maps removed
inline Json boxersPreset(const std::string& folder, int channels, bool threeDelight = false)
{
  Json animations = Json::array();
  auto add        = [&](const std::string& surface, const std::string& channel, const std::string& field, const Json& value) {
    animations.push_back({{"url", "name://@selection#materials/" + surface + ":?extra/studio_material_channels/channels/" + channel + "/" + field},
                                 {"keys", Json::array({Json::array({0, value})})}});
  };
  for(const char* surface : {"Boxers", "BoxersWaistband"})
  {
    for(int i = 0; i < channels; i++)
      add(surface, i == 0 ? (threeDelight ? "glossiness" : "Glossy%20Roughness") : "Channel" + std::to_string(i), "value", 0.5);
    if(!folder.empty())
    {
      add(surface, "diffuse", "image_file", "/Runtime/Textures/ABen/Basic%20Wear/" + folder + "/Boxers_BaseColor.jpg");
      add(surface, "Normal%20Map", "image_file", "/Runtime/Textures/ABen/Basic%20Wear/" + folder + "/Boxers_Normal.jpg");
    }
    for(int i = 0; i < 12; i++)
      add(surface, "Unused" + std::to_string(i), "image", nullptr);
  }
  return {{"asset_info", {{"type", "preset_material"}}}, {"scene", {{"animations", animations}}}};
}

// PBRSkin.dsf's library entry, a few of its channels at the file's defaults: the brick graph and the
// labels, Horizontal Tiles2 labelled "Horizontal Tiles" among them
inline const char* kPbrSkinDefinition = R"({"id":"PBRSkin",
  "diffuse":{"channel":{"id":"diffuse","name":"Diffuse Color","label":"Base Color","type":"float_color","value":[0.7529414,0.7529414,0.7529414]}},
  "extra":[{"type":"studio/material/daz_brick","brick_settings":{}},{"type":"studio_material_channels","channels":[
  {"channel":{"id":"Metallic Weight","label":"Metallicity","type":"float","value":0}},
  {"channel":{"id":"Translucency Enable","type":"bool","value":false}},
  {"channel":{"id":"Translucency Weight","type":"float","value":0.8}},
  {"channel":{"id":"Dual Lobe Specular Enable","type":"bool","value":false}},
  {"channel":{"id":"Dual Lobe Specular Weight","type":"float","value":1}},
  {"channel":{"id":"Dual Lobe Specular Reflectivity","type":"float","value":0.5}},
  {"channel":{"id":"Dual Lobe Specular Roughness Mult","type":"float","value":1}},
  {"channel":{"id":"Specular Lobe 1 Roughness","type":"float","value":0.6}},
  {"channel":{"id":"Specular Lobe 2 Roughness Mult","type":"float","value":0.4}},
  {"channel":{"id":"Dual Lobe Specular Ratio","type":"float","value":0.15}},
  {"channel":{"id":"Normal Map","type":"float","value":1}},
  {"channel":{"id":"Bump Enable","type":"bool","value":false}},
  {"channel":{"id":"Makeup Enable","type":"bool","value":false}},
  {"channel":{"id":"Detail Enable","type":"bool","value":false}},
  {"channel":{"id":"Top Coat Weight","type":"float","value":0.5}},
  {"channel":{"id":"Horizontal Tiles2","label":"Horizontal Tiles","type":"float","value":1}},
  {"channel":{"id":"Vertical Tiles2","label":"Vertical Tiles","type":"float","value":1}}]}]})";

inline const fv::dson::Channel* channelOf(const fv::Material& m, const char* name)
{
  return m.channels->get(fv::dson::norm(name));
}

}  // namespace materials_test
