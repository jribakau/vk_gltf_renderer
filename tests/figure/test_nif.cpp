// A Skyrim Special Edition mesh read and worn: its node tree, skinned shapes, materials and the body
// it was built on (figure/nif), and Skyrim's bone names read as Genesis slots (wardrobe/fit/bones).
// The meshes are written here in the layout nif.xml gives; FV_NIF=<an outfit .nif> also checks a real
// one (Outfit Studio's export).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <figure/dson/figure_loader.hpp>
#include <figure/nif/nif_file.hpp>
#include <figure/nif/nif_wear.hpp>
#include <figure/wardrobe/fit/bases.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/wardrobe/fit/regions.hpp>

namespace {

namespace fs = std::filesystem;
using fv::nif::V3;

// ------------------------------------------------------------------ a NIF writer, for the tests

struct Buf
{
  std::vector<uint8_t> b;
  template <class T>
  void put(T v)
  {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
  }
  void u8(uint8_t v) { put(v); }
  void u16(uint16_t v) { put(v); }
  void u32(uint32_t v) { put(v); }
  void i32(int32_t v) { put(v); }
  void u64(uint64_t v) { put(v); }
  void f32(float v) { put(v); }
  void v3(const V3& v)
  {
    for(float f : v)
      f32(f);
  }
  void sized(const std::string& s)
  {
    u32(uint32_t(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  }
  void shortString(const std::string& s)
  {
    u8(uint8_t(s.size() + 1));
    b.insert(b.end(), s.begin(), s.end());
    b.push_back(0);
  }
};

uint16_t toHalf(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  const int      exp  = int((x >> 23) & 0xFF) - 127 + 15;
  if(exp <= 0)
    return uint16_t(sign);
  return uint16_t(sign | (uint32_t(exp) << 10) | ((x >> 13) & 0x3FF));
}

struct Writer
{
  std::vector<std::string>          types, strings;
  std::vector<uint16_t>             blockType;
  std::vector<std::vector<uint8_t>> blocks;

  int32_t str(const std::string& s)
  {
    for(size_t i = 0; i < strings.size(); i++)
      if(strings[i] == s)
        return int32_t(i);
    strings.push_back(s);
    return int32_t(strings.size() - 1);
  }
  // a block's index before it is written, so blocks can name each other in any order
  int32_t reserve(const std::string& type)
  {
    size_t t = std::find(types.begin(), types.end(), type) - types.begin();
    if(t == types.size())
      types.push_back(type);
    blockType.push_back(uint16_t(t));
    blocks.emplace_back();
    return int32_t(blocks.size() - 1);
  }
  void set(int32_t block, Buf b) { blocks[size_t(block)] = std::move(b.b); }

  std::vector<uint8_t> bytes()
  {
    Buf h;
    const std::string line = "Gamebryo File Format, Version 20.2.0.7\n";
    h.b.insert(h.b.end(), line.begin(), line.end());
    h.u32(0x14020007);
    h.u8(1);
    h.u32(12);
    h.u32(uint32_t(blocks.size()));
    h.u32(100);
    h.shortString("");
    h.shortString("Exported by the tests.");
    h.shortString("");
    h.u16(uint16_t(types.size()));
    for(const auto& t : types)
      h.sized(t);
    for(uint16_t t : blockType)
      h.u16(t);
    for(const auto& b : blocks)
      h.u32(uint32_t(b.size()));
    h.u32(uint32_t(strings.size()));
    size_t longest = 0;
    for(const auto& s : strings)
      longest = std::max(longest, s.size());
    h.u32(uint32_t(longest));
    for(const auto& s : strings)
      h.sized(s);
    h.u32(0);  // groups
    for(const auto& b : blocks)
      h.b.insert(h.b.end(), b.begin(), b.end());
    h.u32(1);  // roots
    h.i32(0);
    return h.b;
  }
};

void avObject(Writer& w, Buf& b, const std::string& name, const V3& t, std::vector<int32_t> extra = {}, uint32_t flags = 0x8000E)
{
  b.i32(w.str(name));
  b.u32(uint32_t(extra.size()));
  for(int32_t e : extra)
    b.i32(e);
  b.i32(-1);  // controller
  b.u32(flags);
  b.v3(t);
  for(float f : {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f})
    b.f32(f);
  b.f32(1);
  b.i32(-1);  // collision
}

void node(Writer& w, int32_t block, const std::string& name, const V3& t, const std::vector<int32_t>& kids)
{
  Buf b;
  avObject(w, b, name, t);
  b.u32(uint32_t(kids.size()));
  for(int32_t k : kids)
    b.i32(k);
  b.u32(0);  // effects
  w.set(block, std::move(b));
}

struct Vert
{
  V3                   p;
  std::array<float, 2> uv;
  // two influences: bone index into the shape's bones, weight
  std::array<std::pair<uint8_t, float>, 2> skin;
};

// A skinned shape, its vertices in the skin partition (as stream 100 keeps a skinned one's), its skin
// data with every bone's bind (the inverse of where the bone stands: unturned here) and weights.
void skinnedShape(Writer& w, int32_t block, const std::string& name, const std::vector<Vert>& verts, const std::vector<uint16_t>& tris,
                  const std::vector<int32_t>& bones, const std::vector<V3>& boneAt, int32_t shader, int32_t alpha, std::vector<int32_t> extra = {})
{
  const int32_t skin = w.reserve("BSDismemberSkinInstance");
  const int32_t data = w.reserve("NiSkinData");
  const int32_t part = w.reserve("NiSkinPartition");
  // position + W, half UV, normal + bitangent Y, skin: 36 bytes
  const uint64_t desc = 9 | (4ull << 8) | (5ull << 16) | (6ull << 28) | (0x4Bull << 44);
  {
    Buf b;
    avObject(w, b, name, {0, 0, 0}, extra);
    for(int i = 0; i < 4; i++)
      b.f32(0);  // bounding sphere
    b.i32(skin);
    b.i32(shader);
    b.i32(alpha);
    b.u64(desc);
    b.u16(uint16_t(tris.size() / 3));
    b.u16(uint16_t(verts.size()));
    b.u32(0);  // its vertices are the partition's
    b.u32(0);  // particle data
    w.set(block, std::move(b));
  }
  {
    Buf b;
    b.i32(data);
    b.i32(part);
    b.i32(0);
    b.u32(uint32_t(bones.size()));
    for(int32_t n : bones)
      b.i32(n);
    b.u32(1);
    b.u16(257);
    b.u16(32);
    w.set(skin, std::move(b));
  }
  {
    Buf b;
    for(float f : {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f})
      b.f32(f);
    b.v3({0, 0, 0});
    b.f32(1);
    b.u32(uint32_t(bones.size()));
    b.u8(1);
    for(size_t k = 0; k < bones.size(); k++)
    {
      for(float f : {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f})
        b.f32(f);
      b.v3({-boneAt[k][0], -boneAt[k][1], -boneAt[k][2]});
      b.f32(1);
      for(int i = 0; i < 4; i++)
        b.f32(0);
      std::vector<std::pair<uint16_t, float>> list;
      for(size_t v = 0; v < verts.size(); v++)
        for(const auto& [bone, weight] : verts[v].skin)
          if(bone == k && weight > 0)
            list.push_back({uint16_t(v), weight});
      b.u16(uint16_t(list.size()));
      for(const auto& [v, weight] : list)
      {
        b.u16(v);
        b.f32(weight);
      }
    }
    w.set(data, std::move(b));
  }
  {
    Buf b;
    b.u32(1);
    b.u32(uint32_t(verts.size() * 36));
    b.u32(36);
    b.u64(desc);
    for(const Vert& v : verts)
    {
      b.v3(v.p);
      b.u32(0);
      b.u16(toHalf(v.uv[0]));
      b.u16(toHalf(v.uv[1]));
      b.u8(128);
      b.u8(128);
      b.u8(255);
      b.u8(128);
      for(const auto& [bone, weight] : v.skin)
        b.u16(toHalf(weight));
      for(int i = 0; i < 2; i++)
        b.u16(0);
      for(const auto& [bone, weight] : v.skin)
        b.u8(bone);
      b.u8(0);
      b.u8(0);
    }
    // one partition over every vertex and bone
    b.u16(uint16_t(verts.size()));
    b.u16(uint16_t(tris.size() / 3));
    b.u16(uint16_t(bones.size()));
    b.u16(0);
    b.u16(4);
    for(size_t k = 0; k < bones.size(); k++)
      b.u16(uint16_t(k));
    b.u8(1);
    for(size_t v = 0; v < verts.size(); v++)
      b.u16(uint16_t(v));
    b.u8(1);
    for(const Vert& v : verts)
      for(int j = 0; j < 4; j++)
        b.f32(j < 2 ? v.skin[size_t(j)].second : 0);
    b.u8(1);
    for(uint16_t i : tris)
      b.u16(i);
    b.u8(1);
    for(const Vert& v : verts)
      for(int j = 0; j < 4; j++)
        b.u8(j < 2 ? v.skin[size_t(j)].first : 0);
    b.u8(0);
    b.u8(1);
    b.u64(desc);
    for(uint16_t i : tris)
      b.u16(i);
    w.set(part, std::move(b));
  }
}

int32_t lightingShader(Writer& w, uint32_t type, const std::vector<std::string>& textures, float glossiness = 80)
{
  const int32_t shader = w.reserve("BSLightingShaderProperty");
  const int32_t set    = w.reserve("BSShaderTextureSet");
  Buf           b;
  b.u32(type);
  b.i32(-1);
  b.u32(0);
  b.i32(-1);
  b.u32(0x82400301);  // specular, skinned, ...
  b.u32(0x8021);
  for(float f : {0.f, 0.f, 1.f, 1.f})
    b.f32(f);
  b.i32(set);
  b.v3({0, 0, 0});
  b.f32(1);
  b.u32(3);
  b.f32(1);  // alpha
  b.f32(0);
  b.f32(glossiness);
  b.v3({1, 1, 1});
  b.f32(1);
  b.f32(0.3f);
  b.f32(2);
  if(type == 1)
    b.f32(0.8f);
  if(type == 5)
    b.v3({1, 1, 1});
  w.set(shader, std::move(b));
  Buf t;
  t.u32(uint32_t(textures.size()));
  for(const auto& s : textures)
    t.sized(s);
  w.set(set, std::move(t));
  return shader;
}

// Scene Root -> NPC COM -> NPC Pelvis -> (L Thigh, R Thigh, Spine); a skirt skinned to the pelvis, the
// left thigh and the spine with a UV seam, the CBBE body it was built on, and a physics proxy with no material.
std::vector<uint8_t> outfit()
{
  Writer        w;
  const int32_t root = w.reserve("NiNode");
  const int32_t com = w.reserve("NiNode"), pelvis = w.reserve("NiNode"), lThigh = w.reserve("NiNode"), rThigh = w.reserve("NiNode"),
                spine = w.reserve("NiNode");
  const int32_t skirt = w.reserve("BSTriShape"), body = w.reserve("BSTriShape"), proxy = w.reserve("BSTriShape");
  node(w, root, "Scene Root", {0, 0, 0}, {com, skirt, body, proxy});
  node(w, com, "NPC COM [COM ]", {0, 0, 60}, {pelvis});
  node(w, pelvis, "NPC Pelvis [Pelv]", {0, 0, 0}, {lThigh, rThigh, spine});
  node(w, lThigh, "NPC L Thigh [LThg]", {-6, 0, 0}, {});
  node(w, rThigh, "NPC R Thigh [RThg]", {6, 0, 0}, {});
  node(w, spine, "NPC Spine [Spn0]", {0, 0, 8}, {});
  const int32_t dress = lightingShader(w, 1,
                                       {"textures\\test\\dress_d.dds", "textures\\test\\dress_n.dds", "", "", "textures\\cube\\shiny_e.dds",
                                        "textures\\test\\dress_m.dds", "", "", ""});
  const int32_t alpha = w.reserve("NiAlphaProperty");
  {
    Buf b;
    b.i32(-1);
    b.u32(0);
    b.i32(-1);
    b.u16(0x200 | 0xEC);  // tests alpha
    b.u8(128);
    w.set(alpha, std::move(b));
  }
  // two triangles; vertex 3 is vertex 2 again with its UV, vertex 4 vertex 1 with another (a seam)
  const std::vector<Vert> verts{
      {{-6, 0, 60}, {0, 0}, {{{0, 0.5f}, {1, 0.5f}}}},
      {{-6, 0, 40}, {0, 1}, {{{1, 1.f}, {0, 0}}}},
      {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}},
      {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}},
      {{-6, 0, 40}, {0.5f, 1}, {{{1, 1.f}, {0, 0}}}},
      {{0, 0, 60}, {1, 0}, {{{2, 1.f}, {1, 0}}}},
  };
  skinnedShape(w, skirt, "_Test00_Skirt", verts, {0, 1, 2, 3, 4, 5}, {pelvis, lThigh, spine}, {{0, 0, 60}, {-6, 0, 60}, {0, 0, 68}}, dress, alpha);
  const int32_t tri = w.reserve("NiStringExtraData");
  {
    Buf b;
    b.i32(w.str("BODYTRI"));
    b.i32(w.str("actors\\character\\character assets\\femalebody.tri"));
    w.set(tri, std::move(b));
  }
  const int32_t skinShader = lightingShader(w, 5, {"textures\\actors\\character\\female\\femalebody_1.dds"});
  skinnedShape(w, body, "CBBE", {{{-5, 0, 60}, {0, 0}, {{{0, 1.f}, {1, 0}}}}, {{-5, 0, 40}, {0, 1}, {{{1, 1.f}, {0, 0}}}}, {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}}},
               {0, 1, 2}, {pelvis, lThigh}, {{0, 0, 60}, {-6, 0, 60}}, skinShader, -1, {tri});
  skinnedShape(w, proxy, "COLPANTS", {{{-5, 0, 60}, {0, 0}, {{{0, 1.f}, {1, 0}}}}, {{-5, 0, 40}, {0, 1}, {{{0, 1.f}, {1, 0}}}}, {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}}},
               {0, 1, 2}, {pelvis, lThigh}, {{0, 0, 60}, {-6, 0, 60}}, -1, -1);
  return w.bytes();
}

// A Data folder with the outfit in meshes\test and its textures (one byte each: only their place counts).
struct Data
{
  fs::path dir, home, nif;
  explicit Data(const std::string& name)
  {
    dir  = fs::temp_directory_path() / ("fv_nif_" + name);
    home = dir / "library";
    fs::remove_all(dir);
    fs::create_directories(dir / "data" / "meshes" / "test");
    fs::create_directories(dir / "data" / "textures" / "test");
    fs::create_directories(home);
    for(const char* t : {"dress_d.dds", "dress_n.dds", "dress_m.dds"})
      std::ofstream(dir / "data" / "textures" / "test" / t, std::ios::binary) << "x";
    nif                        = dir / "data" / "meshes" / "test" / "skirt_1.nif";
    const std::vector<uint8_t> bytes = outfit();
    std::ofstream(nif, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
  }
  ~Data()
  {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

const fv::dson::Channel* channelOf(const fv::Material& m, const char* name)
{
  return m.channels ? m.channels->get(fv::dson::norm(name)) : nullptr;
}

int nodeNamed(const fv::SceneGraph& s, const std::string& name)
{
  for(size_t i = 0; i < s.nodes.size(); i++)
    if(s.nodes[i].name == name)
      return int(i);
  return -1;
}

}  // namespace

TEST(Nif, ReadsTheNodeTreeShapesSkinAndMaterials)
{
  const std::vector<uint8_t> bytes = outfit();
  const fv::nif::File        f     = fv::nif::read(bytes);
  EXPECT_EQ(f.bsVersion, 100u);
  EXPECT_EQ(f.exporter, "Exported by the tests.");
  const int thigh = f.find("NPC L Thigh [LThg]");
  ASSERT_GE(thigh, 0);
  const V3 at = f.world(thigh).t;
  EXPECT_FLOAT_EQ(at[0], -6);
  EXPECT_FLOAT_EQ(at[2], 60);
  EXPECT_EQ(f.nodes[size_t(f.nodes[size_t(thigh)].parent)].name, "NPC Pelvis [Pelv]");

  ASSERT_EQ(f.shapes.size(), 3u);
  const fv::nif::Shape& skirt = f.shapes[0];
  EXPECT_EQ(skirt.name, "_Test00_Skirt");
  ASSERT_EQ(skirt.positions.size(), 6u);
  EXPECT_EQ(skirt.triangles.size(), 6u);
  ASSERT_EQ(skirt.uvs.size(), 6u);
  EXPECT_FLOAT_EQ(skirt.uvs[4][0], 0.5f);
  ASSERT_EQ(skirt.bones.size(), 3u);
  EXPECT_EQ(f.nodes[size_t(skirt.bones[1])].name, "NPC L Thigh [LThg]");
  // the skin data's bind puts each bone where the node tree does
  EXPECT_FLOAT_EQ(skirt.skinToBone[1].inverse().t[0], -6);
  for(size_t v = 0; v < 6; v++)
  {
    float sum = 0;
    for(uint32_t i = skirt.weightStart[v]; i < skirt.weightStart[v + 1]; i++)
      sum += skirt.weight[i];
    EXPECT_NEAR(sum, 1, 1e-6) << v;
  }
  ASSERT_TRUE(skirt.shader);
  EXPECT_EQ(skirt.shader->type, 1u);
  EXPECT_EQ(skirt.shader->texture(0), "textures\\test\\dress_d.dds");
  EXPECT_EQ(skirt.shader->texture(5), "textures\\test\\dress_m.dds");
  EXPECT_FLOAT_EQ(skirt.shader->envMapScale, 0.8f);
  ASSERT_TRUE(skirt.alpha);
  EXPECT_TRUE(skirt.alpha->tests());
  EXPECT_FALSE(skirt.alpha->blends());

  EXPECT_EQ(f.shapes[1].extra.at(0).first, "BODYTRI");
  EXPECT_EQ(fv::nif::bodyShapes(f), std::vector<size_t>{1});
  EXPECT_FALSE(f.shapes[2].shader);
}

TEST(Nif, RefusesOtherGamesAndDamage)
{
  std::vector<uint8_t> bytes = outfit();
  // stream 83: original Skyrim
  const size_t line = std::string(bytes.begin(), bytes.begin() + 64).find('\n') + 1;
  std::vector<uint8_t> le = bytes;
  le[line + 13] = 83;
  EXPECT_THROW(fv::nif::read(le), std::runtime_error);
  try
  {
    fv::nif::read(le);
  }
  catch(const std::runtime_error& e)
  {
    EXPECT_NE(std::string(e.what()).find("Special Edition"), std::string::npos) << e.what();
  }
  // cut short anywhere: an error, never a read past the end
  for(size_t n : {size_t(10), line + 20, bytes.size() / 2, bytes.size() - 20})
    EXPECT_THROW(fv::nif::read(std::span<const uint8_t>(bytes.data(), n)), std::runtime_error) << n;
  const std::string text = "plain text, not a mesh\n";
  EXPECT_THROW(fv::nif::read(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(text.data()), text.size())), std::runtime_error);
}

TEST(Nif, HalfFloats)
{
  EXPECT_EQ(fv::nif::halfToFloat(0x3C00), 1.f);
  EXPECT_EQ(fv::nif::halfToFloat(0x3800), 0.5f);
  EXPECT_EQ(fv::nif::halfToFloat(0xC000), -2.f);
  EXPECT_EQ(fv::nif::halfToFloat(0x0000), 0.f);
  EXPECT_FLOAT_EQ(fv::nif::halfToFloat(0x0001), std::ldexp(1.f, -24));
}

TEST(Nif, SkyrimBonesReadAsGenesisSlots)
{
  using fv::wardrobe::matchBone;
  using fv::wardrobe::parseBone;
  const fv::wardrobe::ParsedBone thigh = parseBone("NPC L Thigh [LThg]");
  EXPECT_EQ(thigh.side, 'l');
  EXPECT_EQ(thigh.slot, "thigh");
  EXPECT_TRUE(parseBone("NPC R UpperarmTwist1 [RUt1]").twist);
  EXPECT_EQ(parseBone("NPC Spine1 [Spn1]").slot, "spine2");

  const std::vector<std::string> g8{"Genesis8Female", "hip", "pelvis", "lThighBend", "lThighTwist", "lShin", "lFoot", "lToe", "abdomenLower",
                                    "abdomenUpper", "chestLower", "chestUpper", "neckLower", "head", "lCollar", "lShldrBend", "lShldrTwist",
                                    "lForearmBend", "lHand", "lThumb2", "lIndex1", "lPectoral", "rThighBend"};
  EXPECT_EQ(matchBone("NPC L Thigh [LThg]", g8), "lThighBend");
  EXPECT_EQ(matchBone("NPC R Thigh [RThg]", g8), "rThighBend");
  EXPECT_EQ(matchBone("NPC L Calf [LClf]", g8), "lShin");
  EXPECT_EQ(matchBone("NPC L Toe0 [LToe]", g8), "lToe");
  EXPECT_EQ(matchBone("NPC COM [COM ]", g8), "hip");
  EXPECT_EQ(matchBone("NPC Pelvis [Pelv]", g8), "pelvis");
  EXPECT_EQ(matchBone("NPC Spine [Spn0]", g8), "abdomenLower");
  EXPECT_EQ(matchBone("NPC Spine1 [Spn1]", g8), "abdomenUpper");
  EXPECT_EQ(matchBone("NPC Spine2 [Spn2]", g8), "chestLower");
  EXPECT_EQ(matchBone("NPC Neck [Neck]", g8), "neckLower");
  EXPECT_EQ(matchBone("NPC L Clavicle [LClv]", g8), "lCollar");
  EXPECT_EQ(matchBone("NPC L UpperArm [LUar]", g8), "lShldrBend");
  EXPECT_EQ(matchBone("NPC L UpperarmTwist1 [LUt1]", g8), "lShldrBend");
  EXPECT_EQ(matchBone("NPC L Forearm [LLar]", g8), "lForearmBend");
  EXPECT_EQ(matchBone("NPC L Finger01 [LF01]", g8), "lThumb2");
  EXPECT_EQ(matchBone("NPC L Breast", g8), "lPectoral");
  // what the figure has no joint for at all leads to nothing, and leaves G8's own names alone
  EXPECT_FALSE(matchBone("NPC Root [Root]", g8));
  EXPECT_FALSE(matchBone("Helga_Skirt_L1 00", g8));
  EXPECT_EQ(matchBone("lShldrBend", g8), "lShldrBend");

  const std::vector<std::string> g9{"Genesis9", "hip", "pelvis", "l_thigh", "l_shin", "l_foot", "l_toes", "spine1", "spine2", "spine3", "spine4",
                                    "neck1", "head", "l_shoulder", "l_upperarm", "l_forearm", "l_hand", "l_thumb2"};
  EXPECT_EQ(matchBone("NPC L Thigh [LThg]", g9), "l_thigh");
  EXPECT_EQ(matchBone("NPC Spine1 [Spn1]", g9), "spine2");
  EXPECT_EQ(matchBone("NPC L Clavicle [LClv]", g9), "l_shoulder");
  EXPECT_EQ(matchBone("NPC L Finger01 [LF01]", g9), "l_thumb2");

  EXPECT_EQ(fv::wardrobe::limbOfBone("NPC R Calf [RClf]"), fv::wardrobe::limb::kRShin);
  EXPECT_EQ(fv::wardrobe::limbOfBone("NPC L Finger21 [LF21]"), fv::wardrobe::limb::kLHand);
  EXPECT_EQ(fv::wardrobe::generationFromBones({"NPC Pelvis [Pelv]", "NPC L Thigh [LThg]"}), fv::wardrobe::Generation::eSkyrim);
  EXPECT_EQ(fv::wardrobe::generationFromBones({"lThighBend", "lShldrBend"}), fv::wardrobe::Generation::eG8);
}

TEST(Nif, WornAsOneItemOnItsBodyWithItsTexturesInTheLibrary)
{
  const Data             data("wear");
  fv::dson::LoadResult   w = fv::nif::loadWearable(data.nif, {}, {data.home});
  ASSERT_EQ(w.items.size(), 1u);
  // the body it was built on and the proxy aren't drawn
  ASSERT_EQ(w.items[0].meshes.size(), 1u);
  EXPECT_EQ(w.items[0].label, "skirt 1");
  ASSERT_EQ(w.parts.size(), 1u);
  EXPECT_TRUE(w.parts[0].conforms);
  const fv::Mesh& m = w.scene.meshes[size_t(w.items[0].meshes[0])];
  // the seam's twin vertices are welded: four cage vertices, the UV seam kept for the corner that differs
  EXPECT_EQ(m.geo->dsonVertexCount, 4);
  EXPECT_EQ(m.geo->polyCount, 2);
  // Skyrim (-6, 0, 60) units is DAZ (+6, 60, 0) units in centimetres: its left at +x, z up made y
  const double u = fv::nif::kUnitCm;
  EXPECT_NEAR(m.geo->cagePositions[0], 6 * u, 1e-4);
  EXPECT_NEAR(m.geo->cagePositions[1], 60 * u, 1e-4);
  EXPECT_NEAR(m.geo->cagePositions[2], 0, 1e-4);
  // skinned to the bones it names, the weights of its first vertex at each place
  ASSERT_EQ(m.bones.size(), 3u);
  EXPECT_EQ(w.scene.nodes[size_t(m.bones[1])].name, "NPC L Thigh [LThg]");
  EXPECT_EQ(m.cageSkin.start.size(), 5u);
  for(size_t v = 0; v < 4; v++)
  {
    float sum = 0;
    for(int32_t i = m.cageSkin.start[v]; i < m.cageSkin.start[v + 1]; i++)
      sum += m.cageSkin.weight[size_t(i)];
    EXPECT_NEAR(sum, 1, 1e-5) << v;
  }
  // the bones at rest unturned, each at its place from its parent
  const int thigh = nodeNamed(w.scene, "NPC L Thigh [LThg]");
  ASSERT_GE(thigh, 0);
  EXPECT_NEAR(w.scene.nodes[size_t(thigh)].joint->centerAuthored[0], 6 * u, 1e-4);
  EXPECT_EQ(w.scene.nodes[size_t(w.scene.nodes[size_t(thigh)].parent)].name, "NPC Pelvis [Pelv]");
  EXPECT_EQ(fv::wardrobe::generationFromBones({w.scene.nodes[size_t(thigh)].name}), fv::wardrobe::Generation::eSkyrim);

  // the CBBE body is the fit's base, skinned to the same skeleton's names
  ASSERT_TRUE(w.sourceBase);
  EXPECT_EQ(w.sourceBase->positions.size(), 9u);
  EXPECT_EQ(w.sourceBase->tris.size(), 3u);
  EXPECT_TRUE(std::any_of(w.sourceBase->skeleton.begin(), w.sourceBase->skeleton.end(), [](const auto& b) { return b.name == "NPC Pelvis [Pelv]"; }));

  // its material: Iray Uber channels over textures linked into the library
  ASSERT_EQ(m.materials.size(), 1u);
  const fv::Material& mat = w.scene.materials[size_t(m.materials[0])];
  EXPECT_EQ(mat.name, "Skirt");
  const fv::dson::Channel* diffuse = channelOf(mat, "Diffuse Color");
  ASSERT_TRUE(diffuse && diffuse->image);
  EXPECT_EQ(diffuse->image->rfind("/Runtime/Textures/FigureViewer/Skyrim/", 0), 0u) << *diffuse->image;
  EXPECT_TRUE(fs::is_regular_file(data.home / diffuse->image->substr(1)));
  EXPECT_EQ(mat.surface.base.map, *diffuse->image);
  // the alpha test cuts by the diffuse's alpha; the normal map is DirectX's; the environment mask is the metal
  EXPECT_EQ(mat.surface.opacity.map, *diffuse->image);
  EXPECT_TRUE(mat.surface.opacityFromAlpha);
  EXPECT_NE(mat.surface.normalMap.find("dress_n.dds"), std::string::npos);
  EXPECT_TRUE(mat.surface.normalDirectX);
  EXPECT_NE(mat.surface.metallic.map.find("dress_m.dds"), std::string::npos);
  EXPECT_FLOAT_EQ(mat.surface.metallic.value, 0.8f);
  EXPECT_EQ(mat.surface.shader, fv::mat::Surface::Shader::eIray);
  // Blinn-Phong 80 is a rough-ish sheen
  EXPECT_NEAR(mat.surface.glossy.roughness.value, std::sqrt(std::sqrt(2 / 82.0)), 1e-4);
  EXPECT_TRUE(w.warnings.empty()) << w.warnings.front();

  // worn again: the links are there already, and the same
  const fv::dson::LoadResult again = fv::nif::loadWearable(data.nif, {}, {data.home});
  EXPECT_EQ(*channelOf(again.scene.materials[0], "Diffuse Color")->image, *diffuse->image);
}

// A mod the app installed (src-tauri/src/skyrim.rs) is in the content folder already: its textures are
// used where they are, and one its own mod lacks comes from another installed beside it
TEST(Nif, AnInstalledModsTexturesAreUsedWhereTheyAre)
{
  const Data     data("installed");
  const fs::path mod   = data.home / fv::nif::kInstalledMods / "Helga CBBE";
  const fs::path other = data.home / fv::nif::kInstalledMods / "A Retexture";
  fs::create_directories(mod / "meshes" / "test");
  fs::create_directories(mod / "textures" / "test");
  fs::create_directories(other / "textures" / "test");
  fs::copy_file(data.nif, mod / "meshes" / "test" / "skirt_1.nif");
  for(const char* t : {"dress_d.dds", "dress_n.dds"})
    std::ofstream(mod / "textures" / "test" / t, std::ios::binary) << "x";
  std::ofstream(other / "textures" / "test" / "dress_m.dds", std::ios::binary) << "x";

  const fv::dson::LoadResult w = fv::nif::loadWearable(mod / "meshes" / "test" / "skirt_1.nif", {}, {data.home});
  ASSERT_EQ(w.items.size(), 1u);
  const fv::Material& mat = w.scene.materials[size_t(w.scene.meshes[size_t(w.items[0].meshes[0])].materials[0])];
  EXPECT_EQ(mat.surface.base.map, "/Skyrim/Helga%20CBBE/textures/test/dress_d.dds");
  EXPECT_EQ(mat.surface.normalMap, "/Skyrim/Helga%20CBBE/textures/test/dress_n.dds");
  EXPECT_EQ(mat.surface.metallic.map, "/Skyrim/A%20Retexture/textures/test/dress_m.dds");
  EXPECT_FALSE(fs::exists(data.home / "Runtime"));
  EXPECT_TRUE(w.warnings.empty()) << w.warnings.front();
}

TEST(Nif, BonesHangAsTheFiguresDoAtItsJoints)
{
  // a figure: hip -> pelvis -> thighs, hip -> abdomenLower, as Genesis 8 hangs them
  fv::SceneGraph figure;
  fv::dson::Rig  rig;
  const auto     bone = [&](const std::string& name, int parent, std::array<double, 3> c) {
    fv::Node n;
    n.name                = name;
    n.kind                = fv::NodeKind::eBone;
    n.joint               = std::make_shared<fv::JointData>();
    n.joint->id           = name;
    n.joint->center       = c;
    n.joint->centerAuthored = c;
    const int id          = figure.add(std::move(n), parent);
    rig.bones.emplace_back(name, id);
    return id;
  };
  rig.root        = bone("Genesis8Female", -1, {0, 0, 0});
  const int hip    = bone("hip", rig.root, {0, 100, 0});
  const int pelvis = bone("pelvis", hip, {0, 100, 0});
  bone("lThighBend", pelvis, {8, 95, 0});
  bone("rThighBend", pelvis, {-8, 95, 0});
  bone("abdomenLower", hip, {0, 105, 0});

  const Data             data("hang");
  fv::dson::LoadOptions  options;
  options.attach = fv::dson::AttachTarget{&figure, &rig, "w1"};
  const fv::dson::LoadResult w = fv::nif::loadWearable(data.nif, options, {data.home});
  const fv::SceneGraph&      s = w.scene;
  const int spine = nodeNamed(s, "NPC Spine [Spn0]"), thigh = nodeNamed(s, "NPC L Thigh [LThg]");
  ASSERT_GE(spine, 0);
  ASSERT_GE(thigh, 0);
  // Skyrim hangs the spine from the pelvis; it follows abdomenLower, which hangs from the hip
  EXPECT_EQ(s.nodes[size_t(s.nodes[size_t(spine)].parent)].name, "NPC COM [COM ]");
  EXPECT_EQ(s.nodes[size_t(s.nodes[size_t(thigh)].parent)].name, "NPC Pelvis [Pelv]");
  // at the figure's joints, the file's places kept for the fit
  const auto& c = s.nodes[size_t(thigh)].joint->center;
  EXPECT_DOUBLE_EQ(c[0], 8);
  EXPECT_DOUBLE_EQ(c[1], 95);
  const auto worlds = s.worlds();
  EXPECT_NEAR(worlds[size_t(thigh)].e[12], 8, 1e-9);
  EXPECT_NEAR(worlds[size_t(thigh)].e[13], 95, 1e-9);
  EXPECT_EQ(w.items[0].key, "w1");
}

// FV_NIF=<an Outfit Studio export with its body, a dress> checks a real file reads whole and consistent.
TEST(Nif, ARealOutfitReadsWhole)
{
  const char* path = std::getenv("FV_NIF");
  if(!path || !*path)
    GTEST_SKIP() << "FV_NIF names no .nif";
  const fv::nif::File f = fv::nif::readFile(path);
  ASSERT_FALSE(f.shapes.empty());
  size_t drawn = 0;
  for(const fv::nif::Shape& s : f.shapes)
  {
    for(uint32_t i : s.triangles)
      ASSERT_LT(i, s.positions.size()) << s.name;
    if(!s.skinned())
      continue;
    drawn += s.shader ? 1 : 0;
    // the skin's bind places every bone where the node tree has it
    for(size_t k = 0; k < s.bones.size(); k++)
    {
      const V3 bind = s.skinToBone[k].inverse().t;
      const V3 tree = f.world(s.bones[k]).t;
      for(int a = 0; a < 3; a++)
        EXPECT_NEAR(bind[size_t(a)], tree[size_t(a)], 0.05) << s.name << " " << f.nodes[size_t(s.bones[k])].name;
    }
    for(size_t v = 0; v + 1 < s.weightStart.size(); v++)
    {
      float sum = 0;
      for(uint32_t i = s.weightStart[v]; i < s.weightStart[v + 1]; i++)
        sum += s.weight[i];
      EXPECT_NEAR(sum, 1, 0.02) << s.name << " vertex " << v;
    }
  }
  EXPECT_GT(drawn, 0u);
  const fs::path home = fs::temp_directory_path() / "fv_nif_real";
  const fv::dson::LoadResult w = fv::nif::loadWearable(path, {}, {home});
  ASSERT_EQ(w.items.size(), 1u);
  EXPECT_FALSE(w.items[0].meshes.empty());
  EXPECT_TRUE(w.sourceBase) << "no body found";
  std::error_code ec;
  fs::remove_all(home, ec);
}
