// The Clothes tab: map slots, the UV transform, fabrics and sheen; edits and what a surface says it moves.

#include "materials_test_util.hpp"

using namespace materials_test;

// ---- The Clothes tab: map slots, the UV transform, fabrics and sheen ----

namespace {

// A garment surface with a map in most slots and a top coat (so a clearcoat normal copies the normal).
fv::mat::Surface garment()
{
  fv::mat::Surface s;
  s.name                 = "Shirt";
  s.shader               = fv::mat::Surface::Shader::eIray;
  s.base.map             = "/Runtime/Textures/shirt_d.jpg";
  s.opacity.map          = "/Runtime/Textures/shirt_o.jpg";
  s.normalMap            = "/Runtime/Textures/shirt_n.jpg";
  s.glossy.roughness.map = "/Runtime/Textures/shirt_r.jpg";
  s.emission             = fv::mat::Emission{{{1, 1, 1}, "/Runtime/Textures/shirt_e.jpg"}};
  s.topCoat              = fv::mat::Surface::TopCoat{0.5f, 0.1f};
  return s;
}

// Every texture info in a lowered material.
void slotsOf(const nlohmann::json& j, std::vector<const nlohmann::json*>& out)
{
  if(j.is_object() && j.contains("slot"))
    out.push_back(&j);
  else if(j.is_structured())
    for(const auto& x : j)
      slotsOf(x, out);
}

fv::mat::UvTransform transformOf(const nlohmann::json& info)
{
  fv::mat::UvTransform t;
  if(!info.contains("extensions"))
    return t;
  const auto& tt = info["extensions"]["KHR_texture_transform"];
  t.scale        = {tt["scale"][0].get<float>(), tt["scale"][1].get<float>()};
  t.offset       = {tt["offset"][0].get<float>(), tt["offset"][1].get<float>()};
  t.rotation     = tt.value("rotation", 0.f);
  return t;
}

void expectSame(const fv::mat::UvTransform& a, const fv::mat::UvTransform& b, const std::string& what)
{
  EXPECT_NEAR(a.scale[0], b.scale[0], 1e-5f) << what;
  EXPECT_NEAR(a.scale[1], b.scale[1], 1e-5f) << what;
  EXPECT_NEAR(a.offset[0], b.offset[0], 1e-5f) << what;
  EXPECT_NEAR(a.offset[1], b.offset[1], 1e-5f) << what;
  EXPECT_NEAR(a.rotation, b.rotation, 1e-5f) << what;
}

}  // namespace

TEST(Materials, TheClothesEditsAreEditsLikeTheRest)
{
  using E = fv::mat::Edits;
  const std::vector<std::function<void(E&)>> setters = {
      [](E& e) { e.baseMap = "/a.png"; },
      [](E& e) { e.opacityMap = ""; },
      [](E& e) { e.normalMap = "/n.png"; },
      [](E& e) { e.bumpMap = "/b.png"; },
      [](E& e) { e.roughnessMap = "/r.png"; },
      [](E& e) { e.uvScaleU = 2; },
      [](E& e) { e.uvScaleV = 2; },
      [](E& e) { e.uvOffsetU = 0.5f; },
      [](E& e) { e.uvOffsetV = 0.5f; },
      [](E& e) { e.uvRotation = 45; },
      [](E& e) { e.fabric = "denim"; },
      [](E& e) { e.fabricScale = 2; },
      [](E& e) { e.fabricStrength = 0.5f; },
      [](E& e) { e.sheenColor = fv::mat::Rgb{1, 0, 0}; },
      [](E& e) { e.sheenRoughness = 0.2f; },
  };
  for(size_t i = 0; i < setters.size(); i++)
  {
    E one;
    setters[i](one);
    EXPECT_FALSE(one.empty()) << i;
    E merged;
    merged.merge(one);
    EXPECT_EQ(fv::mat::editedKeys(merged), fv::mat::editedKeys(one)) << i;
    EXPECT_EQ(fv::mat::editedKeys(one).size(), 1u) << i;
  }
  EXPECT_TRUE(E{}.empty());
}

// Iray Uber's own Horizontal/Vertical Tiles and Offset are the surface's tiling; PBRSkin's Detail
// ones stay the detail layer's.
TEST(Materials, ReadsAnIrayUbersSurfaceTiling)
{
  const auto s = fv::dson::parseSurface("Jeans", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/denim.jpg"}},
    {"channel":{"id":"Horizontal Tiles","type":"float","current_value":4}},
    {"channel":{"id":"Vertical Tiles","type":"float","current_value":0}},
    {"channel":{"id":"Horizontal Offset","type":"float","current_value":0.25}},
    {"channel":{"id":"Vertical Offset","type":"float","current_value":0.5}}]}]})"));
  EXPECT_FLOAT_EQ(s.tiles[0], 4);
  EXPECT_FLOAT_EQ(s.tiles[1], 1);  // 0 tiles is none: untiled
  EXPECT_FLOAT_EQ(s.offset[0], 0.25f);
  EXPECT_FLOAT_EQ(s.offset[1], 0.5f);
  const auto lowered = loweredJson(s, {}, {});
  expectSame(transformOf(lowered.material["pbrMetallicRoughness"]["baseColorTexture"]), {{4, 1}, {0.25f, 0.5f}, 0}, "base");

  const auto plain = fv::dson::parseSurface("Shirt", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/shirt.jpg"}}]}]})"));
  EXPECT_EQ(plain.tiles, (std::array<float, 2>{1, 1}));
  EXPECT_FALSE(loweredJson(plain, {}, {}).material["pbrMetallicRoughness"]["baseColorTexture"].contains("extensions"));
}

// A slot's image replaced asks for the new image; one taken away is gone, and without its cutout a
// surface is opaque and single-sided again. The texture list says which are edits.
TEST(Materials, AMapEditReplacesOrRemovesASlot)
{
  const fv::mat::Surface s = garment();
  fv::mat::Edits         e;
  e.baseMap          = "/Runtime/Textures/FigureViewer/User/0123456789abcdef/denim.png";
  e.roughnessMap     = "/Runtime/Textures/FigureViewer/User/0123456789abcdef/rough.png";
  e.bumpMap          = "/Runtime/Textures/FigureViewer/User/0123456789abcdef/bump.png";
  const auto lowered = loweredJson(s, e, {});
  const auto has     = [&](const LoweredJson& l, const std::string& rel) {
    return std::any_of(l.requests.begin(), l.requests.end(), [&](const fv::mat::TextureRequest& r) { return r.rel == rel; });
  };
  const auto& pbr  = lowered.material["pbrMetallicRoughness"];
  const auto  base = lowered.requests[pbr["baseColorTexture"]["slot"].get<size_t>()];
  EXPECT_EQ(base.rel, *e.baseMap);
  EXPECT_EQ(base.alphaRel, s.opacity.map);  // the cutout still rides in its alpha
  EXPECT_EQ(lowered.requests[pbr["metallicRoughnessTexture"]["slot"].get<size_t>()].rel, *e.roughnessMap);
  const auto normal = lowered.requests[lowered.material["normalTexture"]["slot"].get<size_t>()];
  EXPECT_EQ(normal.rel, s.normalMap);
  EXPECT_EQ(normal.bumpRel, *e.bumpMap);  // a bump map given to a surface with a normal map is blended in
  EXPECT_FALSE(has(lowered, s.base.map));
  EXPECT_EQ(lowered.material["alphaMode"], "BLEND");

  fv::mat::Edits removed;
  removed.opacityMap = "";
  removed.normalMap  = "";
  const auto bare    = loweredJson(s, removed, {});
  EXPECT_EQ(bare.material["alphaMode"], "OPAQUE");
  EXPECT_FALSE(bare.material["doubleSided"].get<bool>());
  EXPECT_TRUE(bare.requests[bare.material["pbrMetallicRoughness"]["baseColorTexture"]["slot"].get<size_t>()].alphaRel.empty());
  EXPECT_FALSE(bare.material.contains("normalTexture"));
  EXPECT_FALSE(bare.material["extensions"]["KHR_materials_clearcoat"].contains("clearcoatNormalTexture"));
  EXPECT_FALSE(has(bare, s.opacity.map));
  removed.alphaMode = "blend";  // an alpha edit still decides
  EXPECT_EQ(loweredJson(s, removed, {}).material["alphaMode"], "BLEND");

  const auto slots = fv::mat::mapSlots(s, removed);
  const auto slot  = [&](const char* name) {
    const auto it = std::find_if(slots.begin(), slots.end(), [&](const fv::mat::MapSlot& m) { return m.slot == name; });
    return it == slots.end() ? fv::mat::MapSlot{} : *it;
  };
  EXPECT_EQ(slot("base").rel, s.base.map);
  EXPECT_FALSE(slot("base").edited);
  EXPECT_EQ(slot("opacity").rel, "");
  EXPECT_TRUE(slot("opacity").edited);
  EXPECT_EQ(slot("opacity").label, "Cutout");
  EXPECT_EQ(slot("normal").rel, "");
  EXPECT_TRUE(slot("normal").edited);
  EXPECT_EQ(slot("roughness").rel, s.glossy.roughness.map);
  EXPECT_EQ(slot("emission").rel, s.emission->color.map);
  EXPECT_TRUE(slot("bump").slot.empty());  // no bump map, no edit: not listed
  EXPECT_EQ(fv::mat::mapSlots(s, e).at(0).rel, *e.baseMap);
}

// The panel's tiling lies over the authored one on every map that samples the surface's UVs, the
// clearcoat's copy of the normal included; a detail layer's own maps keep their tiling, its weight
// map follows the surface. Under flipV each is mirrored.
TEST(Materials, TheUvTransformGoesOnEverySurfaceSlotButTheDetail)
{
  fv::mat::Surface s = garment();
  s.tiles            = {2, 3};
  s.offset           = {0.1f, 0.2f};
  fv::mat::DetailLayer d;
  d.map          = "/Runtime/Textures/pores.jpg";
  d.weight       = 1;
  d.weightMap    = "/Runtime/Textures/pores_w.jpg";
  d.tiles        = {8, 8};
  d.roughnessMap = "/Runtime/Textures/pores_r.jpg";
  s.detail       = d;
  fv::mat::Edits e;
  e.uvScaleU   = 1.5f;
  e.uvOffsetV  = 0.25f;
  e.uvRotation = 30;

  const fv::mat::UvTransform surface = fv::mat::surfaceUv(s, e);
  EXPECT_NEAR(surface.scale[0], 3, 1e-6f);
  EXPECT_NEAR(surface.scale[1], 3, 1e-6f);
  EXPECT_NEAR(surface.rotation, 30 * 3.14159265f / 180, 1e-6f);
  const fv::mat::UvTransform detail{{8, 8}, {0, 0}, 0};

  for(const bool flipV : {false, true})
  {
    fv::mat::LowerOptions o;
    o.flipV            = flipV;
    const auto lowered = loweredJson(s, e, o);
    std::vector<const nlohmann::json*> infos;
    slotsOf(lowered.material, infos);
    EXPECT_GE(infos.size(), 8u);  // base, mr, normal, emissive, the clearcoat's normal, the detail's three
    for(const nlohmann::json* info : infos)
    {
      const std::string          rel  = lowered.requests[(*info)["slot"].get<size_t>()].rel;
      const fv::mat::UvTransform want = rel == d.map || rel == d.roughnessMap ? detail : surface;
      expectSame(transformOf(*info), flipV ? fv::mat::mirroredV(want) : want, rel + (flipV ? " flipped" : ""));
    }
  }
  // a transform changes no image: nothing to cook again
  const auto plain = loweredJson(s, {}, {}), moved = loweredJson(s, e, {});
  ASSERT_EQ(plain.requests.size(), moved.requests.size());
  for(size_t i = 0; i < plain.requests.size(); i++)
    EXPECT_EQ(fv::mat::requestKey(plain.requests[i]), fv::mat::requestKey(moved.requests[i]));
}

// The transforms against the renderer's own matrix (KHR_texture_transform::updateTransform, as the
// shader applies it): the edit lies after the authored tiling, and the flipV mirror samples the
// same texel from V turned over on both sides.
TEST(Materials, TheUvTransformComposesAndMirrorsInTheRenderersMatrix)
{
  const auto apply = [](const fv::mat::UvTransform& t, glm::vec2 uv) {
    KHR_texture_transform k;
    k.offset   = {t.offset[0], t.offset[1]};
    k.rotation = t.rotation;
    k.scale    = {t.scale[0], t.scale[1]};
    k.updateTransform();
    const glm::vec3 p{uv, 1};
    return glm::vec2(glm::dot(k.uvTransform[0], p), glm::dot(k.uvTransform[1], p));
  };
  fv::mat::Surface s;
  s.tiles  = {4, 2};
  s.offset = {0.3f, -0.1f};
  fv::mat::Edits e;
  e.uvScaleU   = 0.5f;
  e.uvScaleV   = 3;
  e.uvOffsetU  = 0.2f;
  e.uvOffsetV  = 0.7f;
  e.uvRotation = 40;
  const fv::mat::UvTransform edit{{0.5f, 3}, {0.2f, 0.7f}, 40 * 3.14159265f / 180};
  const fv::mat::UvTransform t = fv::mat::surfaceUv(s, e);
  for(const glm::vec2 uv : {glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0.3f, 0.8f), glm::vec2(-2, 5)})
  {
    const glm::vec2 authored = uv * glm::vec2(4, 2) + glm::vec2(0.3f, -0.1f);
    const glm::vec2 want = apply(edit, authored), got = apply(t, uv);
    EXPECT_NEAR(got.x, want.x, 1e-4f);
    EXPECT_NEAR(got.y, want.y, 1e-4f);
    // turned over: (u, 1 - v) in, (u', 1 - v') out
    const glm::vec2 mirrored = apply(fv::mat::mirroredV(t), {uv.x, 1 - uv.y});
    EXPECT_NEAR(mirrored.x, got.x, 1e-4f);
    EXPECT_NEAR(mirrored.y, 1 - got.y, 1e-4f);
  }
  // no rotation: the mirror the detail layers always had, offset 1 - tiles - offset
  const fv::mat::UvTransform m = fv::mat::mirroredV({{8, 4}, {0.5f, 0.25f}, 0});
  EXPECT_NEAR(m.offset[0], 0.5f, 1e-6f);
  EXPECT_NEAR(m.offset[1], 1 - 4 - 0.25f, 1e-6f);
  EXPECT_TRUE(fv::mat::surfaceUv({}, {}).identity());
}

// A source image in a figure built for the cooked, bottom-up ones (a cold texture cache): its slot
// samples it with V turned over after the surface's own transform, in the renderer's matrix, and
// only that slot; a slot with no transform of its own gets the bare flip.
TEST(Materials, AnImageWhoseRowsRunTheOtherWayIsSampledWithItsVTurnedOver)
{
  const auto apply = [](const fv::mat::UvTransform& t, glm::vec2 uv) {
    KHR_texture_transform k;
    k.offset   = {t.offset[0], t.offset[1]};
    k.rotation = t.rotation;
    k.scale    = {t.scale[0], t.scale[1]};
    k.updateTransform();
    const glm::vec3 p{uv, 1};
    return glm::vec2(glm::dot(k.uvTransform[0], p), glm::dot(k.uvTransform[1], p));
  };
  const fv::mat::UvTransform t{{3, 2}, {0.25f, -0.4f}, 0.7f};
  for(const glm::vec2 uv : {glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0.3f, 0.8f), glm::vec2(-2, 5)})
  {
    const glm::vec2 want = apply(t, uv), got = apply(fv::mat::rowsFlipped(t), uv);
    EXPECT_NEAR(got.x, want.x, 1e-4f);
    EXPECT_NEAR(got.y, 1 - want.y, 1e-4f);
  }
  expectSame(fv::mat::rowsFlipped({}), {{1, -1}, {0, 1}, 0}, "bare");

  // through toGltf: the base map's source runs the other way, the others are cooked
  const auto infoTransform = [](const tinygltf::TextureInfo& info) {
    fv::mat::UvTransform out;
    const auto           it = info.extensions.find("KHR_texture_transform");
    if(it == info.extensions.end())
      return out;
    const tinygltf::Value& tt = it->second;
    out.scale    = {float(tt.Get("scale").Get(0).GetNumberAsDouble()), float(tt.Get("scale").Get(1).GetNumberAsDouble())};
    out.offset   = {float(tt.Get("offset").Get(0).GetNumberAsDouble()), float(tt.Get("offset").Get(1).GetNumberAsDouble())};
    out.rotation = tt.Has("rotation") ? float(tt.Get("rotation").GetNumberAsDouble()) : 0.f;
    return out;
  };
  fv::mat::Surface tiled = garment();
  tiled.tiles            = {2, 3};
  for(const fv::mat::Surface& s : {garment(), tiled})
  {
    const fv::mat::UvTransform own     = fv::mat::surfaceUv(s, {});
    const auto                 lowered = fv::mat::lower(s, {}, {});
    std::vector<std::string>   used;
    const auto                 m = fv::mat::toGltf(
        lowered,
        [&](const fv::mat::TextureRequest& r) {
          const bool base = r.rel == s.base.map;
          return fv::mat::ResolvedTexture{base ? 1 : 2, false, base};
        },
        used);
    expectSame(infoTransform(m.pbrMetallicRoughness.baseColorTexture), fv::mat::rowsFlipped(own), "base");
    expectSame(infoTransform(m.emissiveTexture), own, "emissive");
    EXPECT_NE(std::find(used.begin(), used.end(), "KHR_texture_transform"), used.end());
  }
}

// A fabric is a detail layer: its height map at its physical size from metresPerUv, its relief
// fixed (a scale changes the tiling, never the image), its roughness multiplier, weight 1. A file
// for other tools leaves it out, as it does every detail.
TEST(Materials, AFabricIsADetailLayerOfItsOwnSize)
{
  const fv::mat::Fabric* denim = fv::mat::findFabric("denim");
  ASSERT_NE(denim, nullptr);
  for(const char* key : {"denim", "canvas", "linen", "knit", "satin", "corduroy", "leather"})
    EXPECT_NE(fv::mat::findFabric(key), nullptr) << key;
  EXPECT_EQ(fv::mat::fabrics().size(), 7u);
  EXPECT_EQ(fv::mat::fabricHeightMap(*denim), "/Runtime/Textures/FigureViewer/Fabrics/v1/denim-height.png");
  EXPECT_EQ(fv::mat::fabricRoughnessMap(*denim), "/Runtime/Textures/FigureViewer/Fabrics/v1/denim-rough.png");

  fv::mat::Surface s = garment();
  s.tiles            = {5, 5};  // the garment's tiling is not the weave's
  fv::mat::Edits e;
  e.fabric         = "denim";
  e.fabricStrength = 0.7f;
  fv::mat::LowerOptions o;
  o.metresPerUv      = 0.8f;
  const auto  lowered = loweredJson(s, e, o);
  const auto& de      = lowered.material["extensions"]["FV_materials_detail"];
  EXPECT_FLOAT_EQ(de["detailWeightFactor"].get<float>(), 1);
  EXPECT_FLOAT_EQ(de["detailRoughnessFactor"].get<float>(), denim->roughness);
  EXPECT_FALSE(de.contains("detailWeightTexture"));
  const auto& normal = de["detailNormalTexture"];
  EXPECT_FLOAT_EQ(normal["scale"].get<float>(), 0.7f);
  const float tiles = 0.8f / denim->tileMetres;
  expectSame(transformOf(normal), {{tiles, tiles}, {0, 0}, 0}, "normal");
  expectSame(transformOf(de["detailRoughnessTexture"]), {{tiles, tiles}, {0, 0}, 0}, "roughness");
  const fv::mat::TextureRequest& r = lowered.requests[normal["slot"].get<size_t>()];
  EXPECT_EQ(r.rel, fv::mat::fabricHeightMap(*denim));
  EXPECT_EQ(r.channels, "bn");
  EXPECT_FLOAT_EQ(r.bumpHeight, denim->relief);
  EXPECT_EQ(lowered.requests[de["detailRoughnessTexture"]["slot"].get<size_t>()].rel, fv::mat::fabricRoughnessMap(*denim));

  // twice as coarse: half the tiles, the same image
  e.fabricScale       = 2;
  const auto  coarse  = loweredJson(s, e, o);
  const auto& cnormal = coarse.material["extensions"]["FV_materials_detail"]["detailNormalTexture"];
  expectSame(transformOf(cnormal), {{tiles / 2, tiles / 2}, {0, 0}, 0}, "coarse");
  EXPECT_EQ(fv::mat::requestKey(coarse.requests[cnormal["slot"].get<size_t>()]), fv::mat::requestKey(r));

  // no measured size: a fixed tiling
  const auto unknown = loweredJson(s, e, {});
  EXPECT_GT(transformOf(unknown.material["extensions"]["FV_materials_detail"]["detailNormalTexture"]).scale[0], 1);

  o.forFile = true;
  EXPECT_FALSE(loweredJson(s, e, o).material["extensions"].contains("FV_materials_detail"));
  e.fabric = "";
  EXPECT_FALSE(loweredJson(s, e, {}).material["extensions"].contains("FV_materials_detail"));

  e.fabric         = "satin";
  const auto slots = fv::mat::mapSlots(s, e);
  ASSERT_FALSE(slots.empty());
  EXPECT_EQ(slots.back().slot, "detail");
  EXPECT_EQ(slots.back().rel, fv::mat::fabricHeightMap(*fv::mat::findFabric("satin")));
  EXPECT_TRUE(slots.back().edited);
}

TEST(Materials, ASheenEditIsKhrMaterialsSheen)
{
  const fv::mat::Surface s = garment();
  fv::mat::Edits         e;
  EXPECT_FALSE(loweredJson(s, e, {}).material["extensions"].contains("KHR_materials_sheen"));
  e.sheenColor        = fv::mat::Rgb{0.5f, 0.25f, 0.125f};
  const auto  lowered = loweredJson(s, e, {});
  const auto& sheen   = lowered.material["extensions"]["KHR_materials_sheen"];
  EXPECT_FLOAT_EQ(sheen["sheenColorFactor"][1].get<float>(), 0.25f);
  EXPECT_FLOAT_EQ(sheen["sheenRoughnessFactor"].get<float>(), 0.5f);
  e.sheenRoughness = 0.2f;
  EXPECT_FLOAT_EQ(loweredJson(s, e, {}).material["extensions"]["KHR_materials_sheen"]["sheenRoughnessFactor"].get<float>(), 0.2f);
  e.sheenColor = fv::mat::Rgb{0, 0, 0};
  EXPECT_FALSE(loweredJson(s, e, {}).material["extensions"].contains("KHR_materials_sheen"));
}

// material.set's values onto edits and back: what a surface publishes for a key is what material.set
// takes; a map is a library path, never a disk path; an unknown fabric or key is refused.
TEST(Materials, TheEditValuesRoundTrip)
{
  using fv::mat::EditsJson;
  const EditsJson in = {{"color", "#804020"},
                        {"roughness", 0.3},
                        {"baseMap", "/Runtime/Textures/x.png"},
                        {"opacityMap", ""},
                        {"normalMap", "Runtime\\Textures\\n.png"},
                        {"uvScaleU", 2.5},
                        {"uvScaleV", 1000},
                        {"uvOffsetU", -0.5},
                        {"uvRotation", 90},
                        {"fabric", "corduroy"},
                        {"fabricScale", 1.5},
                        {"fabricStrength", 0.4},
                        {"sheenColor", "#336699"},
                        {"sheenRoughness", 0.25}};
  fv::mat::Edits e;
  for(auto it = in.begin(); it != in.end(); ++it)
    fv::mat::applyValue(e, it.key(), it.value());
  EXPECT_EQ(*e.normalMap, "/Runtime/Textures/n.png");
  EXPECT_FLOAT_EQ(*e.uvScaleV, 100);  // clamped

  EditsJson values = EditsJson::object();
  fv::mat::clothesValues(e, values);
  values["color"]        = fv::mat::hexOf(*e.color);
  values["roughness"]    = *e.roughness;
  const EditsJson edited = fv::mat::editedKeys(e);
  EXPECT_EQ(edited.size(), in.size());
  // the values not edited: null or their defaults
  EXPECT_TRUE(values["bumpMap"].is_null());
  EXPECT_FLOAT_EQ(values["uvOffsetV"].get<float>(), 0);
  // the published values set again give the same edits
  fv::mat::Edits again;
  for(const auto& k : edited)
    fv::mat::applyValue(again, k.get<std::string>(), values[k.get<std::string>()]);
  EditsJson republished = EditsJson::object();
  fv::mat::clothesValues(again, republished);
  for(auto it = republished.begin(); it != republished.end(); ++it)
    EXPECT_EQ(it.value(), values[it.key()]) << it.key();
  EXPECT_EQ(fv::mat::editedKeys(again), edited);
  EXPECT_EQ(values["sheenColor"], "#336699");
  EXPECT_EQ(values["opacityMap"], "");

  // null drops an edit
  fv::mat::applyValue(again, "baseMap", nullptr);
  fv::mat::applyValue(again, "fabric", nullptr);
  EXPECT_FALSE(again.baseMap);
  EXPECT_FALSE(again.fabric);

  for(const char* bad : {"C:/Users/me/denim.png", "\\\\server\\share\\denim.png", "/Runtime/../../denim.png", "..", "/a|b.png", "file:///x.png"})
    EXPECT_THROW(fv::mat::applyValue(e, "baseMap", bad), std::runtime_error) << bad;
  EXPECT_THROW(fv::mat::applyValue(e, "baseMap", 3), std::runtime_error);
  EXPECT_THROW(fv::mat::applyValue(e, "fabric", "tweed"), std::runtime_error);
  EXPECT_THROW(fv::mat::applyValue(e, "sheenColor", "red"), std::runtime_error);
  EXPECT_THROW(fv::mat::applyValue(e, "uvScale", 2), std::runtime_error);
  fv::mat::applyValue(e, "roughnessMap", "/Runtime/Textures/a..b.png");  // dots in a name are no way up
  EXPECT_EQ(*e.roughnessMap, "/Runtime/Textures/a..b.png");

  for(const char* k : {"baseMap", "roughnessMap", "uvScaleU", "uvRotation", "fabric", "fabricStrength"})
    EXPECT_TRUE(fv::mat::figureOnlyKey(k)) << k;
  for(const char* k : {"sheenColor", "sheenRoughness", "color", "roughness"})
    EXPECT_FALSE(fv::mat::figureOnlyKey(k)) << k;
}

// A scene file keeps the edit, not what this render mode shows of it: reading saves back from the
// published values lost a scatter distance saved in Real-time, and made an emitter's power 1
TEST(Materials, EditsAreKeptAsTheyWereGivenWhateverASurfaceShows)
{
  using EditsJson = fv::mat::EditsJson;
  const EditsJson in = {{"color", "#336699"}, {"roughness", 0.25}, {"doubleSided", false}, {"alphaMode", "cutout"}, {"baseMap", "/Runtime/Textures/b.png"},
                        {"fabric", ""},       {"scatterScale", 2}, {"emissiveIntensity", 40}, {"normalStrength", 1.5}};
  fv::mat::Edits  e;
  for(auto it = in.begin(); it != in.end(); ++it)
    fv::mat::applyValue(e, it.key(), it.value());
  const EditsJson kept = fv::mat::editedValues(e);
  EXPECT_EQ(kept.size(), in.size());
  EXPECT_EQ(fv::mat::editedKeys(e).size(), in.size());
  // set again, they are the same edits and say the same
  fv::mat::Edits again;
  for(auto it = kept.begin(); it != kept.end(); ++it)
    fv::mat::applyValue(again, it.key(), it.value());
  EXPECT_EQ(fv::mat::editedValues(again), kept);
  EXPECT_FLOAT_EQ(*again.scatterScale, 2);
  EXPECT_FLOAT_EQ(*again.emissiveIntensity, 40);
  EXPECT_EQ(kept["color"], "#336699");

  // null drops a number and a colour, as it does a map: the panel's way back to the surface's own
  fv::mat::applyValue(again, "roughness", nullptr);
  fv::mat::applyValue(again, "color", nullptr);
  EXPECT_FALSE(again.roughness);
  EXPECT_FALSE(again.color);
  EXPECT_FALSE(fv::mat::editedValues(again).contains("roughness"));
  EXPECT_TRUE(fv::mat::editedValues(fv::mat::Edits{}).empty());
}

// The panel leaves out what can't matter: it reads that from here
TEST(Materials, ATearFilmHasNoReliefAndASurfaceSeenThroughNoColour)
{
  const auto     film = fv::dson::parseSurface("EyeMoisture Left", channelsOf(kG9Moisture));
  fv::mat::Edits relief;
  relief.normalStrength = 1;
  relief.bumpStrength   = 1;
  const auto shown = fv::mat::lower(film, relief, {.underTearFilm = true}).effective;
  EXPECT_FALSE(shown.normalStrength);
  EXPECT_FALSE(shown.bumpStrength);

  fv::mat::Surface glass;
  glass.refraction = fv::mat::Surface::Refraction{.weight = 1};
  EXPECT_TRUE(fv::mat::lower(glass, {}, {}).effective.colorUnused);
  glass.refraction->weight = 0.5f;
  EXPECT_FALSE(fv::mat::lower(glass, {}, {}).effective.colorUnused);
  EXPECT_FALSE(fv::mat::lower(fv::mat::Surface{}, {}, {}).effective.colorUnused);
}

TEST(Materials, AClearcoatOfNothingAddsNoCoat)
{
  fv::mat::Edits none;
  none.clearcoat = 0;
  EXPECT_FALSE(fv::mat::lower(fv::mat::Surface{}, none, {}).material.clearcoat);
  fv::mat::Edits some;
  some.clearcoat = 0.5f;
  const auto coated = fv::mat::lower(fv::mat::Surface{}, some, {});
  ASSERT_TRUE(coated.material.clearcoat);
  EXPECT_FLOAT_EQ(coated.effective.clearcoat, 0.5f);
}

// The Clearcoat sliders say what they move: a file's coat, or the sharper dual lobe, in both renderers
TEST(Materials, ASurfaceSaysWhatItsClearcoatValuesMove)
{
  fv::mat::Surface plain;
  plain.glossy.weight = 0;
  EXPECT_EQ(fv::mat::lower(plain, {}, {}).effective.coatRole, "");

  fv::mat::Surface dual = plain;
  dual.dual             = fv::mat::DualLobe{.weight = 1, .lobe1 = {.value = 0.2f}, .lobe2 = {.value = 0.6f}, .ratio = 0.5f};
  for(const auto mode : {fv::mat::SkinMode::eRaster, fv::mat::SkinMode::eVolume})
    EXPECT_EQ(fv::mat::lower(dual, {}, {.skin = mode}).effective.coatRole, "tight");

  fv::mat::Surface coated = dual;
  coated.topCoat          = fv::mat::Surface::TopCoat{.weight = 0.5f};
  for(const auto mode : {fv::mat::SkinMode::eRaster, fv::mat::SkinMode::eVolume})
    EXPECT_EQ(fv::mat::lower(coated, {}, {.skin = mode}).effective.coatRole, "coat");
}

// A number over a map is a factor; the panel marks it
TEST(Materials, ASurfaceSaysWhichOfItsValuesMultiplyAMap)
{
  const auto mappedOf = [](const fv::mat::Surface& s, const fv::mat::Edits& e, const fv::mat::LowerOptions& o) {
    return fv::mat::lower(s, e, o).effective.mapped;
  };
  const auto has = [](const std::vector<std::string>& list, const char* key) { return std::find(list.begin(), list.end(), key) != list.end(); };

  EXPECT_TRUE(mappedOf(fv::mat::Surface{}, {}, {}).empty());

  fv::mat::Surface s;
  s.base.map     = "/b.png";
  s.opacity.map  = "/o.png";
  s.metallic.map = "/m.png";
  fv::mat::Edits rough;
  rough.roughnessMap = "/r.png";
  const auto all = mappedOf(s, rough, {});
  for(const char* key : {"color", "opacity", "metalness", "roughness"})
    EXPECT_TRUE(has(all, key)) << key;
  EXPECT_FALSE(has(all, "emissive"));

  // the dual lobe's maps go with the sliders that move it
  fv::mat::Surface dual;
  dual.glossy.weight = 0;
  dual.dual          = fv::mat::DualLobe{.weight = 1, .lobe1 = {.value = 0.2f}, .lobe2 = {.value = 0.6f}, .ratio = 0.5f, .weightMap = "/w.png"};
  for(const auto mode : {fv::mat::SkinMode::eRaster, fv::mat::SkinMode::eVolume})
    EXPECT_TRUE(has(mappedOf(dual, {}, {.skin = mode}), "clearcoat"));
}

// In Thin mode, nails and teeth, most of a surface's light comes back through its scatter colour:
// Base color moved only the rest
TEST(Materials, BaseColorTintsThinScatter)
{
  const ChannelValues v = {{"diffuse", Json::array({1, 1, 1})}, {"Translucency Weight", 0.8}, {"SSS Reflectance Tint", Json::array({0.5, 0.5, 0.5})}};
  const std::map<std::string, std::string> images = {{"diffuse", "/Runtime/Textures/face_d.jpg"}};
  const auto face  = surfaceOf("Face", kUberType, v, images);
  const auto as    = loweredJson(face, {}, {.skin = fv::mat::SkinMode::eThin});
  fv::mat::Edits e;
  e.color          = fv::mat::Rgb{0.5f, 0.25f, 1};
  const auto tinted = loweredJson(face, e, {.skin = fv::mat::SkinMode::eThin});
  const auto factor = [](const auto& lowered, int i) { return lowered.material["extensions"]["KHR_materials_scatter"]["multiscatterColorFactor"][i].template get<float>(); };
  for(int i = 0; i < 3; i++)
    EXPECT_NEAR(factor(tinted, i), factor(as, i) * (*e.color)[size_t(i)], 1e-5f) << i;
}

// A strand or a card in the path tracer: the panel shows the fibre's numbers, and the flags say so
TEST(Materials, HairShowsTheFibresNumbers)
{
  fv::mat::Surface cards;
  cards.kind        = fv::mat::Kind::eHair;
  cards.name        = "Hair";
  cards.opacity.map = "/o.png";
  fv::mat::Edits map;
  map.roughnessMap = "/r.png";
  const auto traced = fv::mat::lower(cards, map, {.skin = fv::mat::SkinMode::eVolume}).effective;
  EXPECT_FLOAT_EQ(traced.specular, 1);
  EXPECT_FALSE(traced.fibre);  // a card is a sheet with a fibre's lobes: its base colour still colours it
  EXPECT_EQ(std::find(traced.mapped.begin(), traced.mapped.end(), "roughness"), traced.mapped.end());
  map.specular = 0.3f;
  EXPECT_FLOAT_EQ(fv::mat::lower(cards, map, {.skin = fv::mat::SkinMode::eVolume}).effective.specular, 0.3f);

  // the real-time view keeps the sheet, whose map counts
  const auto sheet = fv::mat::lower(cards, map, {.skin = fv::mat::SkinMode::eRaster}).effective;
  EXPECT_NE(std::find(sheet.mapped.begin(), sheet.mapped.end(), "roughness"), sheet.mapped.end());
}

// Legacy (3Delight) surfaces: the channels of Age of Armour's Subsurface shader, as in
// data/Age of Armour/Subsurface Shaders/AoA_Subsurface/AoA_Subsurface.dsf, where labels name other
// channels' ids, and the id-only layer a figure's scene lays over them (Genesis 3 Female.duf, Torso).
