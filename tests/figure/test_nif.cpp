// A Skyrim Special Edition mesh read and worn: its node tree, skinned shapes, materials and the body
// it was built on (figure/nif), and Skyrim's bone names read as Genesis slots (wardrobe/fit/bones).
// The meshes are written here in the layout nif.xml gives; FV_NIF=<an outfit .nif> also checks a real
// one (Outfit Studio's export).

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
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

// How a shape's skin is written, where a file may differ from the plain one.
struct SkinAs
{
  V3   space{0, 0, 0};      // the skin's own space: where the file's origin lies in it (NiSkinData's overall transform)
  bool dataWeights = true;  // NiSkinData lists each bone's vertices and weights
  bool vertexSkin  = true;  // the vertex data carries weights and bone indices
  bool partWeights = true;  // the partitions carry them
  int  partitions  = 1;     // each over every vertex; the triangles are the first's
};

// A skinned shape, its vertices in the skin partition (as stream 100 keeps a skinned one's), its skin
// data with every bone's bind (the inverse of where the bone stands: unturned here) and weights. `verts`
// and `boneAt` are in the file's space, whatever `as.space`.
void skinnedShape(Writer& w, int32_t block, const std::string& name, const std::vector<Vert>& verts, const std::vector<uint16_t>& tris,
                  const std::vector<int32_t>& bones, const std::vector<V3>& boneAt, int32_t shader, int32_t alpha, std::vector<int32_t> extra = {},
                  uint16_t slot = 32, const SkinAs& as = {})
{
  const int32_t skin = w.reserve("BSDismemberSkinInstance");
  const int32_t data = w.reserve("NiSkinData");
  const int32_t part = w.reserve("NiSkinPartition");
  // position + W, half UV, normal + bitangent Y, skin: 36 bytes; 24 without the skin
  const uint64_t desc   = as.vertexSkin ? 9 | (4ull << 8) | (5ull << 16) | (6ull << 28) | (0x4Bull << 44) : 6 | (4ull << 8) | (5ull << 16) | (0x0Bull << 44);
  const uint32_t stride = as.vertexSkin ? 36 : 24;
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
    b.u16(slot);
    w.set(skin, std::move(b));
  }
  {
    Buf b;
    for(float f : {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f})
      b.f32(f);
    b.v3(as.space);
    b.f32(1);
    b.u32(uint32_t(bones.size()));
    b.u8(as.dataWeights ? 1 : 0);
    for(size_t k = 0; k < bones.size(); k++)
    {
      for(float f : {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f})
        b.f32(f);
      b.v3({-boneAt[k][0] - as.space[0], -boneAt[k][1] - as.space[1], -boneAt[k][2] - as.space[2]});
      b.f32(1);
      for(int i = 0; i < 4; i++)
        b.f32(0);
      std::vector<std::pair<uint16_t, float>> list;
      for(size_t v = 0; v < verts.size() && as.dataWeights; v++)
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
    b.u32(uint32_t(as.partitions));
    b.u32(uint32_t(verts.size() * stride));
    b.u32(stride);
    b.u64(desc);
    for(const Vert& v : verts)
    {
      b.v3({v.p[0] + as.space[0], v.p[1] + as.space[1], v.p[2] + as.space[2]});
      b.u32(0);
      b.u16(toHalf(v.uv[0]));
      b.u16(toHalf(v.uv[1]));
      b.u8(128);
      b.u8(128);
      b.u8(255);
      b.u8(128);
      if(!as.vertexSkin)
        continue;
      for(const auto& [bone, weight] : v.skin)
        b.u16(toHalf(weight));
      for(int i = 0; i < 2; i++)
        b.u16(0);
      for(const auto& [bone, weight] : v.skin)
        b.u8(bone);
      b.u8(0);
      b.u8(0);
    }
    // each partition over every vertex and bone, the triangles in the first
    for(int k = 0; k < as.partitions; k++)
    {
      const std::vector<uint16_t> own = k == 0 ? tris : std::vector<uint16_t>{};
      b.u16(uint16_t(verts.size()));
      b.u16(uint16_t(own.size() / 3));
      b.u16(uint16_t(bones.size()));
      b.u16(0);
      b.u16(4);
      for(size_t i = 0; i < bones.size(); i++)
        b.u16(uint16_t(i));
      b.u8(1);
      for(size_t v = 0; v < verts.size(); v++)
        b.u16(uint16_t(v));
      b.u8(as.partWeights ? 1 : 0);
      for(const Vert& v : verts)
        for(int j = 0; j < 4 && as.partWeights; j++)
          b.f32(j < 2 ? v.skin[size_t(j)].second : 0);
      b.u8(1);
      for(uint16_t i : own)
        b.u16(i);
      b.u8(as.partWeights ? 1 : 0);
      for(const Vert& v : verts)
        for(int j = 0; j < 4 && as.partWeights; j++)
          b.u8(j < 2 ? v.skin[size_t(j)].first : 0);
      b.u8(0);
      b.u8(1);
      b.u64(desc);
      for(uint16_t i : own)
        b.u16(i);
    }
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
// `standing`: feet under the thighs (at z 5) and a neck on the spine (z 108) that nothing is skinned to.
// `slot`: the skirt's biped slot (32, the body's). `as`: how the skirt's skin is written. `thighBound`: how
// far from where it stands the skirt was bound with the left thigh, its vertices carried with it by their
// weights (the body keeps the plain bind, which is the node tree's). `odd`: three more shapes the reader
// can't draw: one whose skin is a block of another kind, a BSDynamicTriShape, and one with an effect shader.
// `bodyAt`: how far the body it carries lies from where the others' does (another preset, or another body).
// `bare`: it carries no body, what would be one being a lining of the dress's cloth.
std::vector<uint8_t> outfit(bool standing = false, uint16_t slot = 32, const SkinAs& as = {}, const V3& thighBound = {0, 0, 0}, bool odd = false,
                            const V3& bodyAt = {0, 0, 0}, bool bare = false)
{
  Writer        w;
  const int32_t root = w.reserve("NiNode");
  const int32_t com = w.reserve("NiNode"), pelvis = w.reserve("NiNode"), lThigh = w.reserve("NiNode"), rThigh = w.reserve("NiNode"),
                spine = w.reserve("NiNode");
  const int32_t lFoot = standing ? w.reserve("NiNode") : -1, rFoot = standing ? w.reserve("NiNode") : -1, neck = standing ? w.reserve("NiNode") : -1;
  const int32_t skirt = w.reserve("BSTriShape"), body = w.reserve("BSTriShape"), proxy = w.reserve("BSTriShape");
  const int32_t broken = odd ? w.reserve("BSTriShape") : -1, face = odd ? w.reserve("BSDynamicTriShape") : -1, glow = odd ? w.reserve("BSTriShape") : -1;
  const auto    only  = [&](int32_t n) { return n >= 0 ? std::vector<int32_t>{n} : std::vector<int32_t>{}; };
  node(w, root, "Scene Root", {0, 0, 0}, odd ? std::vector<int32_t>{com, skirt, body, proxy, broken, face, glow} : std::vector<int32_t>{com, skirt, body, proxy});
  if(odd)
  {
    // a shape whose skin is a node: no skin this reader knows
    Buf b;
    avObject(w, b, "Broken", {0, 0, 0});
    for(int i = 0; i < 4; i++)
      b.f32(0);
    b.i32(com);
    b.i32(-1);
    b.i32(-1);
    b.u64(9 | (4ull << 8) | (5ull << 16) | (6ull << 28) | (0x4Bull << 44));
    b.u16(0);
    b.u16(0);
    b.u32(0);
    b.u32(0);
    w.set(broken, std::move(b));
    Buf f;
    avObject(w, f, "Face", {0, 0, 0});
    w.set(face, std::move(f));
    const int32_t effect = w.reserve("BSEffectShaderProperty");
    w.set(effect, Buf{});
    skinnedShape(w, glow, "Glow", {{{-5, 0, 60}, {0, 0}, {{{0, 1.f}, {1, 0}}}}, {{-5, 0, 40}, {0, 1}, {{{0, 1.f}, {1, 0}}}}, {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}}},
                 {0, 1, 2}, {pelvis, lThigh}, {{0, 0, 60}, {-6, 0, 60}}, effect, -1);
  }
  node(w, com, "NPC COM [COM ]", {0, 0, 60}, {pelvis});
  node(w, pelvis, "NPC Pelvis [Pelv]", {0, 0, 0}, {lThigh, rThigh, spine});
  node(w, lThigh, "NPC L Thigh [LThg]", {-6, 0, 0}, only(lFoot));
  node(w, rThigh, "NPC R Thigh [RThg]", {6, 0, 0}, only(rFoot));
  node(w, spine, "NPC Spine [Spn0]", {0, 0, 8}, only(neck));
  if(standing)
  {
    node(w, lFoot, "NPC L Foot [Lft ]", {0, 0, -55}, {});
    node(w, rFoot, "NPC R Foot [Rft ]", {0, 0, -55}, {});
    node(w, neck, "NPC Neck [Neck]", {0, 0, 40}, {});
  }
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
  std::vector<Vert> verts{
      {{-6, 0, 60}, {0, 0}, {{{0, 0.5f}, {1, 0.5f}}}},
      {{-6, 0, 40}, {0, 1}, {{{1, 1.f}, {0, 0}}}},
      {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}},
      {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}},
      {{-6, 0, 40}, {0.5f, 1}, {{{1, 1.f}, {0, 0}}}},
      {{0, 0, 60}, {1, 0}, {{{2, 1.f}, {1, 0}}}},
  };
  for(Vert& v : verts)
    for(const auto& [bone, weight] : v.skin)
      if(bone == 1)
        for(size_t k = 0; k < 3; k++)
          v.p[k] += weight * thighBound[k];
  skinnedShape(w, skirt, "_Test00_Skirt", verts, {0, 1, 2, 3, 4, 5}, {pelvis, lThigh, spine},
               {{0, 0, 60}, {-6 + thighBound[0], thighBound[1], 60 + thighBound[2]}, {0, 0, 68}}, dress, alpha, {}, slot, as);
  const int32_t tri = w.reserve("NiStringExtraData");
  {
    Buf b;
    b.i32(w.str("BODYTRI"));
    b.i32(w.str("actors\\character\\character assets\\femalebody.tri"));
    w.set(tri, std::move(b));
  }
  const int32_t skinShader = lightingShader(w, 5, {"textures\\actors\\character\\female\\femalebody_1.dds"});
  std::vector<Vert> skin{{{-5, 0, 60}, {0, 0}, {{{0, 1.f}, {1, 0}}}}, {{-5, 0, 40}, {0, 1}, {{{1, 1.f}, {0, 0}}}}, {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}}};
  for(Vert& v : skin)
    for(size_t k = 0; k < 3; k++)
      v.p[k] += bodyAt[k];
  if(bare)
    skinnedShape(w, body, "Lining", skin, {0, 1, 2}, {pelvis, lThigh}, {{0, 0, 60}, {-6, 0, 60}}, dress, -1);
  else
    skinnedShape(w, body, "CBBE", skin, {0, 1, 2}, {pelvis, lThigh}, {{0, 0, 60}, {-6, 0, 60}}, skinShader, -1, {tri});
  skinnedShape(w, proxy, "COLPANTS", {{{-5, 0, 60}, {0, 0}, {{{0, 1.f}, {1, 0}}}}, {{-5, 0, 40}, {0, 1}, {{{0, 1.f}, {1, 0}}}}, {{0, 0, 40}, {1, 1}, {{{0, 1.f}, {1, 0}}}}},
               {0, 1, 2}, {pelvis, lThigh}, {{0, 0, 60}, {-6, 0, 60}}, -1, -1);
  return w.bytes();
}

// A Data folder with the outfit in meshes\test and its textures (one byte each: only their place counts).
struct Data
{
  fs::path dir, home, nif;
  explicit Data(const std::string& name, bool standing = false, uint16_t slot = 32)
      : Data(name, outfit(standing, slot))
  {
  }
  Data(const std::string& name, const std::vector<uint8_t>& bytes)
  {
    dir  = fs::temp_directory_path() / ("fv_nif_" + name);
    home = dir / "library";
    fs::remove_all(dir);
    fs::create_directories(dir / "data" / "meshes" / "test");
    fs::create_directories(dir / "data" / "textures" / "test");
    fs::create_directories(home);
    for(const char* t : {"dress_d.dds", "dress_n.dds", "dress_m.dds"})
      std::ofstream(dir / "data" / "textures" / "test" / t, std::ios::binary) << "x";
    nif = dir / "data" / "meshes" / "test" / "skirt_1.nif";
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
  // made for Skyrim, whatever bones it has
  EXPECT_EQ(w.items[0].madeFor, std::optional<fv::wardrobe::Generation>(fv::wardrobe::Generation::eSkyrim));
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

// An outfit stands as tall as the figure by the highest joint both have above the feet: here the neck,
// which neither it nor its body is skinned to (Skyrim's head is a mesh of its own). With no such joint
// it keeps Skyrim's size and says so.
TEST(Nif, ItStandsAsTallAsTheFigureByTheHighestJointBothHave)
{
  const auto figureWith = [](bool feet) {
    auto            out    = std::make_unique<std::pair<fv::SceneGraph, fv::dson::Rig>>();
    fv::SceneGraph& figure = out->first;
    fv::dson::Rig&  rig    = out->second;
    const auto      bone   = [&](const std::string& name, int parent, std::array<double, 3> c) {
      fv::Node n;
      n.name                  = name;
      n.kind                  = fv::NodeKind::eBone;
      n.joint                 = std::make_shared<fv::JointData>();
      n.joint->id             = name;
      n.joint->center         = c;
      n.joint->centerAuthored = c;
      const int id            = figure.add(std::move(n), parent);
      rig.bones.emplace_back(name, id);
      return id;
    };
    rig.root         = bone("Genesis8Female", -1, {0, 0, 0});
    const int hip    = bone("hip", rig.root, {0, 100, 0});
    const int pelvis = bone("pelvis", hip, {0, 100, 0});
    const int lThigh = bone("lThighBend", pelvis, {8, 95, 0});
    const int rThigh = bone("rThighBend", pelvis, {-8, 95, 0});
    if(feet)
    {
      bone("lFoot", lThigh, {8, 8, 0});
      bone("rFoot", rThigh, {-8, 8, 0});
    }
    const int abdomen = bone("abdomenLower", hip, {0, 105, 0});
    bone("neckLower", abdomen, {0, 150, 0});
    return out;
  };
  const Data  data("tall", true);
  const auto  standing = figureWith(true);
  fv::dson::LoadOptions options;
  options.attach = fv::dson::AttachTarget{&standing->first, &standing->second, "w1"};
  const fv::dson::LoadResult w = fv::nif::loadWearable(data.nif, options, {data.home});
  // neck to feet: 142 cm on the figure, 103 Skyrim units on the outfit
  const double u     = fv::nif::kUnitCm;
  const double scale = (150.0 - 8.0) / (103 * u);
  const int    thigh = nodeNamed(w.scene, "NPC L Thigh [LThg]");
  ASSERT_GE(thigh, 0);
  EXPECT_NEAR(w.scene.nodes[size_t(thigh)].joint->centerAuthored[1], 60 * u * scale, 1e-6);
  EXPECT_GE(nodeNamed(w.scene, "NPC L Foot [Lft ]"), 0) << "a bone nothing is skinned to is kept";
  for(const std::string& warning : w.warnings)
    EXPECT_EQ(warning.find("Skyrim's size"), std::string::npos) << warning;

  const auto footless = figureWith(false);
  options.attach      = fv::dson::AttachTarget{&footless->first, &footless->second, "w1"};
  const fv::dson::LoadResult same = fv::nif::loadWearable(data.nif, options, {data.home});
  const int                  t    = nodeNamed(same.scene, "NPC L Thigh [LThg]");
  ASSERT_GE(t, 0);
  EXPECT_NEAR(same.scene.nodes[size_t(t)].joint->centerAuthored[1], 60 * u, 1e-6);
  EXPECT_TRUE(std::any_of(same.warnings.begin(), same.warnings.end(), [](const std::string& s) { return s.find("Skyrim's size") != std::string::npos; }));
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

// The game wears a piece on the feet's biped slot (37), not the body's (32), as footwear: so is it
// fitted, though its bones (as many calf as foot ones on a boot) and a name like "Helga_boots_1"
// couldn't tell
TEST(Nif, APieceOnTheFeetsSlotIsFootwear)
{
  const Data           shoe("shoe", false, 37);
  const fv::nif::File  f = fv::nif::readFile(shoe.nif);
  const auto skirtShape = std::find_if(f.shapes.begin(), f.shapes.end(), [](const fv::nif::Shape& s) { return s.name == "_Test00_Skirt"; });
  ASSERT_NE(skirtShape, f.shapes.end());
  EXPECT_EQ(skirtShape->bodyParts, (std::vector<uint16_t>{37}));
  const fv::dson::LoadResult w = fv::nif::loadWearable(shoe.nif, {}, {shoe.home});
  ASSERT_EQ(w.items.size(), 1u);
  EXPECT_EQ(w.items[0].region, std::optional<fv::wardrobe::FitRegion>(fv::wardrobe::FitRegion::eShoe));
  const Data                 skirt("skirt");
  const fv::dson::LoadResult body = fv::nif::loadWearable(skirt.nif, {}, {skirt.home});
  EXPECT_FALSE(body.items[0].region);
  EXPECT_EQ(fv::wardrobe::regionOf("Skyrim/Helga/meshes/Fuse00/Armor/Helga/Helga_boots_1.nif/Helga boots 1", {}), fv::wardrobe::FitRegion::eBoot);
}

namespace {

// A shape vertex's influences by bone.
std::map<uint16_t, float> influencesOf(const fv::nif::Shape& s, size_t v)
{
  std::map<uint16_t, float> out;
  for(uint32_t i = s.weightStart[v]; i < s.weightStart[v + 1]; i++)
    out[s.weightBone[i]] += s.weight[i];
  return out;
}

// What a wear of `data` made: the skirt's cage, and each bone's place in the file by its name.
struct Worn
{
  std::vector<float>                         cage;
  std::map<std::string, std::array<double, 3>> joints;
};
Worn wornOf(const Data& data)
{
  const fv::dson::LoadResult w = fv::nif::loadWearable(data.nif, {}, {data.home});
  Worn                       out;
  out.cage = w.scene.meshes.at(size_t(w.items.at(0).meshes.at(0))).geo->cagePositions;
  for(const fv::Node& n : w.scene.nodes)
    if(n.joint)
      out.joints[n.name] = n.joint->centerAuthored;
  if(w.sourceBase)
    out.cage.insert(out.cage.end(), w.sourceBase->positions.begin(), w.sourceBase->positions.end());
  return out;
}

void expectSameWear(const Worn& a, const Worn& b)
{
  ASSERT_EQ(a.cage.size(), b.cage.size());
  for(size_t i = 0; i < a.cage.size(); i++)
    EXPECT_NEAR(a.cage[i], b.cage[i], 1e-3) << i;
  ASSERT_EQ(a.joints.size(), b.joints.size());
  for(const auto& [name, at] : a.joints)
    for(size_t k = 0; k < 3; k++)
      EXPECT_NEAR(at[k], b.joints.at(name)[k], 1e-3) << name;
}

}  // namespace

TEST(Nif, WeightsComeFromTheVertexDataWhenTheSkinDataKeepsNone)
{
  SkinAs as;
  as.dataWeights = false;
  as.partWeights = false;
  const fv::nif::File   f     = fv::nif::read(outfit(false, 32, as));
  const fv::nif::Shape& skirt = f.shapes[0];
  ASSERT_EQ(skirt.weightStart.size(), 7u);
  EXPECT_EQ(influencesOf(skirt, 0), (std::map<uint16_t, float>{{0, 0.5f}, {1, 0.5f}}));
  EXPECT_EQ(influencesOf(skirt, 1), (std::map<uint16_t, float>{{1, 1.f}}));
  EXPECT_EQ(influencesOf(skirt, 5), (std::map<uint16_t, float>{{2, 1.f}}));
}

TEST(Nif, AVertexTwoPartitionsShareIsWeightedOnce)
{
  // neither the skin data nor the vertices carry weights: the partitions', both of which hold every vertex
  SkinAs as;
  as.dataWeights = false;
  as.vertexSkin  = false;
  as.partitions  = 2;
  const fv::nif::File   f     = fv::nif::read(outfit(false, 32, as));
  const fv::nif::Shape& skirt = f.shapes[0];
  ASSERT_EQ(skirt.weightStart.size(), 7u);
  EXPECT_EQ(skirt.triangles.size(), 6u);
  EXPECT_EQ(influencesOf(skirt, 0), (std::map<uint16_t, float>{{0, 0.5f}, {1, 0.5f}}));
  EXPECT_EQ(skirt.weightStart[1] - skirt.weightStart[0], 2u);
  EXPECT_EQ(influencesOf(skirt, 5), (std::map<uint16_t, float>{{2, 1.f}}));
}

TEST(Nif, AShapeInItsOwnSkinSpaceStandsWithTheOthers)
{
  // the skirt's vertices and binds kept in a space of its own, as NiSkinData's overall transform says:
  // worn, it and its bones are where the plain file's are, on the body beside it
  const Data plain("space_plain");
  SkinAs     as;
  as.space = {3, -4, -10};
  const Data moved("space_moved", outfit(false, 32, as));
  expectSameWear(wornOf(moved), wornOf(plain));
}

TEST(Nif, AShapeBoundInAnotherPoseIsCarriedToTheFilesBind)
{
  // the skirt bound with the left thigh 2 units out and 3 up from where the body's bind and the node
  // tree have it: the game poses both by one skeleton, so the skirt's cloth on the thigh comes with it
  const Data plain("bind_plain");
  const Data moved("bind_moved", outfit(false, 32, {}, {-2, 0, 3}));
  expectSameWear(wornOf(moved), wornOf(plain));
  // read alone, the vertices are as the file has them
  const fv::nif::File f = fv::nif::read(outfit(false, 32, {}, {-2, 0, 3}));
  EXPECT_FLOAT_EQ(f.shapes[0].positions[1][0], -8);
  EXPECT_FLOAT_EQ(f.shapes[0].positions[0][0], -7);
}

TEST(Nif, AShapeThatCantBeReadIsLeftOutNotTheFile)
{
  const std::vector<uint8_t> bytes = outfit(false, 32, {}, {0, 0, 0}, true);
  const fv::nif::File        f     = fv::nif::read(bytes);
  // the skirt, the body, the proxy and the glow are read; the two that can't be are named with why
  ASSERT_EQ(f.shapes.size(), 4u);
  EXPECT_EQ(f.shapes[3].name, "Glow");
  EXPECT_TRUE(f.shapes[3].effect);
  EXPECT_FALSE(f.shapes[3].shader);
  ASSERT_EQ(f.skipped.size(), 2u);
  EXPECT_NE(f.skipped[0].find("Broken"), std::string::npos) << f.skipped[0];
  EXPECT_NE(f.skipped[0].find("NiNode"), std::string::npos) << f.skipped[0];
  EXPECT_NE(f.skipped[1].find("Face"), std::string::npos) << f.skipped[1];
  EXPECT_NE(f.skipped[1].find("BSDynamicTriShape"), std::string::npos) << f.skipped[1];

  // worn, the skirt comes in, and the wear says what didn't
  const Data                 data("odd", bytes);
  const fv::dson::LoadResult w = fv::nif::loadWearable(data.nif, {}, {data.home});
  ASSERT_EQ(w.items.size(), 1u);
  EXPECT_EQ(w.items[0].meshes.size(), 1u);
  size_t left = 0, glow = 0;
  for(const std::string& warning : w.warnings)
  {
    left += warning.find("was left out") != std::string::npos;
    glow += warning.find("Glow isn't drawn") != std::string::npos;
  }
  EXPECT_EQ(left, 2u);
  EXPECT_EQ(glow, 1u);
}

// A Data folder's file at `rel`, its folders made.
void put(const fs::path& data, const std::string& rel, const std::vector<uint8_t>& bytes = {})
{
  const fs::path p = data / fs::path(rel);
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

// Whose body a piece is for is read as the game's own files tell it: by a folder, by the name, or by the
// twin beside it.
TEST(Nif, APiecesGenderIsReadFromItsFolderAndItsName)
{
  const Data     data("gender");
  const fs::path d = data.dir / "data";
  for(const char* rel : {"meshes/armor/iron/f/cuirass_1.nif", "meshes/armor/iron/male/cuirass_1.nif", "meshes/clothes/farm/torsof_1.nif",
                         "meshes/clothes/farm/torsom_1.nif", "meshes/armor/rings/goldf_0.nif", "meshes/armor/rings/gold_0.nif",
                         "meshes/armor/db/hood_f_1.nif", "meshes/armor/db/hood_1.nif", "meshes/armor/guard/helm_1.nif",
                         "meshes/armor/guard/scarf_1.nif", "meshes/armor/guard/femalecuffs.nif", "meshes/armor/guard/malecuffs.nif"})
    put(d, rel);
  const auto of = [&](const char* rel) { return fv::nif::genderOf(d / fs::path(rel)); };
  EXPECT_EQ(of("meshes/armor/iron/f/cuirass_1.nif"), "female");
  EXPECT_EQ(of("meshes/armor/iron/male/cuirass_1.nif"), "male");
  EXPECT_EQ(of("meshes/clothes/farm/torsof_1.nif"), "female");
  EXPECT_EQ(of("meshes/clothes/farm/torsom_1.nif"), "male");
  // the man's is the one left unmarked
  EXPECT_EQ(of("meshes/armor/rings/goldf_0.nif"), "female");
  EXPECT_EQ(of("meshes/armor/rings/gold_0.nif"), "male");
  EXPECT_EQ(of("meshes/armor/db/hood_f_1.nif"), "female");
  EXPECT_EQ(of("meshes/armor/db/hood_1.nif"), "male");
  EXPECT_EQ(of("meshes/armor/guard/femalecuffs.nif"), "female");
  EXPECT_EQ(of("meshes/armor/guard/malecuffs.nif"), "male");
  // a last letter alone says nothing: a helm, a scarf
  EXPECT_EQ(of("meshes/armor/guard/helm_1.nif"), "");
  EXPECT_EQ(of("meshes/armor/guard/scarf_1.nif"), "");
}

// A piece that carries a body takes the whole body its own is cut from, where one is on the machine: in
// its own Data folder or an installed mod's, at its own preset where it lies off the file's, and not a
// body it doesn't lie on.
TEST(Nif, APieceIsMatchedToTheWholeBodyItWasCutFrom)
{
  const char*  kBody = "meshes/actors/character/character assets/femalebody_1.nif";
  const double u     = fv::nif::kUnitCm;
  {
    // the game's body installed as a mod of its own, the outfit carrying a cut of it as it is
    const Data data("whole");
    put(data.home / "Skyrim" / "The Game", kBody, outfit());
    const fv::dson::LoadResult w = fv::nif::loadWearable(data.nif, {}, {data.home});
    ASSERT_TRUE(w.sourceWhole);
    EXPECT_EQ(w.sourceWhole->positions.size(), 9u);
    EXPECT_TRUE(w.sourceWholePreset.empty());
    EXPECT_EQ(w.sourceGender, "female");
    EXPECT_EQ(w.sourceScale, 1);
    // the one object for the next piece made on it
    EXPECT_EQ(fv::nif::loadWearable(data.nif, {}, {data.home}).sourceWhole, w.sourceWhole);
    // with no library to look in there is none
    EXPECT_FALSE(fv::nif::loadWearable(data.nif, {}, {}).sourceWhole);
  }
  {
    // the outfit's body at a preset 0.3 units out: the whole body is read there
    const Data data("whole_preset", outfit(false, 32, {}, {0, 0, 0}, false, {0, 0.3f, 0}));
    put(data.dir / "data", kBody, outfit());
    const fv::dson::LoadResult w = fv::nif::loadWearable(data.nif, {}, {data.home});
    ASSERT_TRUE(w.sourceWhole);
    ASSERT_EQ(w.sourceWholePreset.size(), 9u);
    for(size_t v = 0; v < 3; v++)
    {
      EXPECT_NEAR(w.sourceWholePreset[v * 3], w.sourceWhole->positions[v * 3], 1e-4);
      EXPECT_NEAR(w.sourceWholePreset[v * 3 + 1], w.sourceWhole->positions[v * 3 + 1], 1e-4);
      // Skyrim's +y (its front) is DAZ's +z
      EXPECT_NEAR(w.sourceWholePreset[v * 3 + 2], w.sourceWhole->positions[v * 3 + 2] + 0.3 * u, 1e-4);
    }
  }
  {
    // a body 3 units (4 cm) off isn't the body it was cut from
    const Data data("whole_other", outfit(false, 32, {}, {0, 0, 0}, false, {0, 3, 0}));
    put(data.dir / "data", kBody, outfit());
    EXPECT_FALSE(fv::nif::loadWearable(data.nif, {}, {data.home}).sourceWhole);
  }
}

// A piece that carries no body takes its own Data folder's body of its gender, where its place or name
// says which; not another mod's body, which nothing says it was made on.
TEST(Nif, APieceThatCarriesNoBodyTakesItsOwnFoldersBodyOfItsGender)
{
  const char*    kBody = "meshes/actors/character/character assets/femalebody_1.nif";
  const auto     bare  = outfit(false, 32, {}, {0, 0, 0}, false, {0, 0, 0}, true);
  const Data     data("whole_bare", bare);
  const fs::path d = data.dir / "data";
  put(d, "meshes/armor/iron/f/cuirass_0.nif", bare);
  put(d, "meshes/armor/iron/m/cuirass_0.nif", bare);
  // no body anywhere yet
  EXPECT_FALSE(fv::nif::loadWearable(d / "meshes/armor/iron/f/cuirass_0.nif", {}, {data.home}).sourceWhole);
  // another mod's body is not taken
  put(data.home / "Skyrim" / "A Body", kBody, outfit());
  EXPECT_FALSE(fv::nif::loadWearable(d / "meshes/armor/iron/f/cuirass_0.nif", {}, {data.home}).sourceWhole);
  // its own folder's is: the woman's for the woman's piece (of the other weight, there being none of its own)
  put(d, kBody, outfit());
  const fv::dson::LoadResult w = fv::nif::loadWearable(d / "meshes/armor/iron/f/cuirass_0.nif", {}, {data.home});
  ASSERT_TRUE(w.sourceWhole);
  EXPECT_FALSE(w.sourceBase);
  EXPECT_EQ(w.sourceGender, "female");
  EXPECT_TRUE(w.sourceWholePreset.empty());
  for(const std::string& warning : w.warnings)
    EXPECT_EQ(warning.find("no body to fit"), std::string::npos) << warning;
  // the man's piece has no man's body there, and a piece that doesn't say whose it is takes none
  EXPECT_FALSE(fv::nif::loadWearable(d / "meshes/armor/iron/m/cuirass_0.nif", {}, {data.home}).sourceWhole);
  EXPECT_FALSE(fv::nif::loadWearable(data.nif, {}, {data.home}).sourceWhole);
}

// A folder of real meshes, named by FV_NIF_DIR (the game's own, unpacked from its archives): how many
// read, what the others say, and what the skins are like (their own spaces, shapes bound apart), then
// each piece a figure could wear loaded as one. A survey, printed: it fails only on a crash.
TEST(Nif, ARealFolderReads)
{
  const char* dir = std::getenv("FV_NIF_DIR");
  if(!dir || !*dir)
    GTEST_SKIP() << "FV_NIF_DIR names no folder";
  const auto lowered = [](std::string t) {
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return t;
  };
  // a message without its file's name and its numbers, so alike ones count together
  const auto kind = [](std::string why, const std::string& file) {
    for(size_t at = why.find(file); at != std::string::npos; at = why.find(file))
      why.replace(at, file.size(), "<file>");
    for(char& c : why)
      if(std::isdigit(static_cast<unsigned char>(c)))
        c = '#';
    return why.size() > 150 ? why.substr(0, 150) : why;
  };
  std::map<std::string, size_t>      unread, left, unworn, notices;
  std::map<std::string, std::string> example;
  size_t files = 0, read = 0, shapes = 0, skinned = 0, ownSpace = 0, apartFiles = 0, offTreeFiles = 0, noWeights = 0, worn = 0, wearable = 0, based = 0;
  double farthest = 0;
  const fs::path home = fs::temp_directory_path() / "fv_nif_folder";
  for(const auto& e : fs::recursive_directory_iterator(dir))
  {
    if(!e.is_regular_file() || lowered(e.path().extension().string()) != ".nif")
      continue;
    files++;
    const std::string name = e.path().filename().string();
    fv::nif::File     f;
    try
    {
      f = fv::nif::readFile(e.path());
    }
    catch(const std::exception& x)
    {
      const std::string k = kind(x.what(), name);
      unread[k]++;
      example.emplace(k, e.path().string());
      continue;
    }
    read++;
    for(const std::string& why : f.skipped)
    {
      const std::string k = kind(why.substr(why.find(": ") == std::string::npos ? 0 : why.find(": ") + 2), name);
      left[k]++;
      example.emplace(k, e.path().string());
    }
    // each bone's bind by the shapes that skin to it, in the file's space
    std::map<int, std::vector<fv::nif::Transform>> binds;
    bool                                           offTree = false;
    for(const fv::nif::Shape& sh : f.shapes)
    {
      shapes++;
      if(!sh.skinned())
        continue;
      skinned++;
      const fv::nif::Transform& g = sh.skinTransform;
      bool                      own = std::abs(g.s - 1) > 1e-3f;
      for(size_t i = 0; i < 3; i++)
        own = own || std::abs(g.t[i]) > 0.05f;
      for(size_t i = 0; i < 9; i++)
        own = own || std::abs(g.r[i] - (i % 4 == 0 ? 1.f : 0.f)) > 2e-3f;
      ownSpace += own ? 1 : 0;
      noWeights += sh.weight.empty() ? 1 : 0;
      const fv::nif::Transform toFile = g.inverse();
      for(size_t b = 0; b < sh.bones.size() && b < sh.skinToBone.size(); b++)
      {
        const fv::nif::Transform at = toFile * sh.skinToBone[b].inverse();
        binds[sh.bones[b]].push_back(at);
        const V3 tree = f.world(sh.bones[b]).t;
        for(size_t k = 0; k < 3; k++)
          offTree = offTree || std::abs(at.t[k] - tree[k]) > 0.05f;
      }
    }
    bool apart = false;
    for(const auto& [node, list] : binds)
      for(size_t i = 1; i < list.size(); i++)
        for(size_t k = 0; k < 3; k++)
        {
          const double d = std::abs(double(list[i].t[k]) - list[0].t[k]);
          apart          = apart || d > 0.05;
          farthest       = std::max(farthest, d);
        }
    apartFiles += apart ? 1 : 0;
    offTreeFiles += offTree ? 1 : 0;

    // what a figure could wear: not the actors' own, not the first person's or the ground's copy, one weight
    const std::string rel  = lowered(e.path().generic_string());
    const std::string stem = lowered(e.path().stem().string());
    const auto        ends = [&](const char* tail) { return stem.size() >= std::strlen(tail) && stem.compare(stem.size() - std::strlen(tail), std::strlen(tail), tail) == 0; };
    if(rel.find("/actors/") != std::string::npos || stem.rfind("1stperson", 0) == 0 || ends("gnd") || ends("go") || ends("_0"))
      continue;
    wearable++;
    try
    {
      const fv::dson::LoadResult w = fv::nif::loadWearable(e.path(), {}, {home});
      worn++;
      based += w.sourceBase ? 1 : 0;
      for(const std::string& n : w.warnings)
      {
        const std::string k = kind(n, name);
        notices[k]++;
        example.emplace(k, e.path().string());
      }
    }
    catch(const std::exception& x)
    {
      const std::string k = kind(x.what(), name);
      unworn[k]++;
      example.emplace(k, e.path().string());
    }
  }
  std::error_code ec;
  fs::remove_all(home, ec);
  std::cerr << "[nif folder] " << files << " files, " << read << " read; " << shapes << " shapes, " << skinned << " skinned, " << ownSpace
            << " in a skin space of their own, " << noWeights << " skinned with no weights; files with shapes bound apart: " << apartFiles << " (farthest "
            << farthest << " units), with a bind off the node tree: " << offTreeFiles << "\n";
  const auto list = [&](const char* title, const std::map<std::string, size_t>& m) {
    std::vector<std::pair<size_t, std::string>> by;
    for(const auto& [k, n] : m)
      by.emplace_back(n, k);
    std::sort(by.rbegin(), by.rend());
    std::cerr << "[nif folder] " << title << ": " << by.size() << " kinds\n";
    for(size_t i = 0; i < by.size() && i < 12; i++)
      std::cerr << "  " << by[i].first << "  " << by[i].second << "   (" << example[by[i].second] << ")\n";
  };
  list("not read", unread);
  list("shapes left out", left);
  std::cerr << "[nif folder] " << wearable << " to wear, " << worn << " loaded, " << based << " with a body to fit against\n";
  list("not worn", unworn);
  list("notices", notices);
  EXPECT_GT(files, 0u);
}
