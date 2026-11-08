#include <figure/nif/nif_file.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace fv::nif {

namespace {

constexpr uint32_t kVersion = 0x14020007;  // 20.2.0.7
constexpr uint32_t kSse     = 100;

// A bounds-checked little-endian cursor over the file.
class Reader
{
public:
  Reader(std::span<const uint8_t> b, size_t at = 0, size_t end = SIZE_MAX)
      : m_b(b)
      , m_at(at)
      , m_end(std::min(end, b.size()))
  {
  }
  size_t at() const { return m_at; }
  void   seek(size_t at) { m_at = at; }
  void   skip(size_t n)
  {
    need(n);
    m_at += n;
  }
  template <class T>
  T get()
  {
    need(sizeof(T));
    T v;
    std::memcpy(&v, m_b.data() + m_at, sizeof(T));
    m_at += sizeof(T);
    return v;
  }
  uint8_t  u8() { return get<uint8_t>(); }
  uint16_t u16() { return get<uint16_t>(); }
  uint32_t u32() { return get<uint32_t>(); }
  int32_t  i32() { return get<int32_t>(); }
  uint64_t u64() { return get<uint64_t>(); }
  float    f32() { return get<float>(); }
  V3       v3() { return {f32(), f32(), f32()}; }
  std::string chars(size_t n)
  {
    need(n);
    std::string s(reinterpret_cast<const char*>(m_b.data() + m_at), n);
    m_at += n;
    return s;
  }
  // a uint32 length and that many bytes
  std::string sized() { return chars(count(u32(), 1)); }
  // a byte length (counting a terminating zero) and that many bytes
  std::string shortString()
  {
    std::string s = chars(u8());
    while(!s.empty() && s.back() == '\0')
      s.pop_back();
    return s;
  }
  // `n` items of `size` bytes still fit: a count read from a damaged file can't allocate the world
  size_t count(uint64_t n, size_t size) const
  {
    if(size && n > (m_end - std::min(m_at, m_end)) / size)
      throw std::runtime_error("the file is cut short or damaged (a count of " + std::to_string(n) + " runs past its end)");
    return size_t(n);
  }

private:
  void need(size_t n) const
  {
    if(m_at > m_end || m_end - m_at < n)
      throw std::runtime_error("the file is cut short or damaged");
  }
  std::span<const uint8_t> m_b;
  size_t                   m_at, m_end;
};

struct Block
{
  std::string type;
  size_t      at = 0, size = 0;
};

struct Parser
{
  std::span<const uint8_t> bytes;
  std::vector<Block>       blocks;
  std::vector<std::string> strings;
  uint32_t                 bsVersion = 0;

  const Block* block(int32_t ref) const { return ref >= 0 && size_t(ref) < blocks.size() ? &blocks[size_t(ref)] : nullptr; }
  Reader       at(const Block& b) const { return Reader(bytes, b.at, b.at + b.size); }
  std::string  str(int32_t index) const { return index >= 0 && size_t(index) < strings.size() ? strings[size_t(index)] : std::string(); }
};

bool isNode(const std::string& type)
{
  // NiNode and the subclasses that add nothing to its layout
  return type == "NiNode" || type == "BSFadeNode" || type == "BSLeafAnimNode" || type == "NiBillboardNode" || type == "BSDamageStage"
         || type == "BSBlastNode" || type == "BSDebrisNode";
}

struct AvObject
{
  std::string          name;
  std::vector<int32_t> extra;
  uint32_t             flags = 0;
  Transform            local;
};

// NiObjectNET then NiAVObject, as stream 100 has them.
AvObject readAvObject(Reader& r, const Parser& p)
{
  AvObject o;
  o.name = p.str(r.i32());
  const size_t nExtra = r.count(r.u32(), 4);
  for(size_t i = 0; i < nExtra; i++)
    o.extra.push_back(r.i32());
  r.i32();  // controller
  o.flags   = r.u32();
  o.local.t = r.v3();
  for(float& v : o.local.r)
    v = r.f32();
  o.local.s = r.f32();
  r.i32();  // collision object
  return o;
}

// NiTransform: rotation, then translation and scale.
Transform readNiTransform(Reader& r)
{
  Transform t;
  for(float& v : t.r)
    v = r.f32();
  t.t = r.v3();
  t.s = r.f32();
  return t;
}

// BSVertexDesc's attributes and offsets (in bytes).
struct VertexDesc
{
  uint64_t raw = 0;
  size_t   size() const { return size_t(raw & 0xF) * 4; }
  size_t   uvAt() const { return size_t((raw >> 8) & 0xF) * 4; }
  size_t   normalAt() const { return size_t((raw >> 16) & 0xF) * 4; }
  size_t   skinAt() const { return size_t((raw >> 28) & 0xF) * 4; }
  uint32_t attributes() const { return uint32_t(raw >> 44) & 0xFFF; }
  bool     has(uint32_t bit) const { return attributes() & bit; }
};
constexpr uint32_t kVfVertex = 0x1, kVfUv = 0x2, kVfNormal = 0x8, kVfSkinned = 0x40;

float unitByte(uint8_t b)
{
  return float(b) / 255.f * 2.f - 1.f;
}

struct Vertices
{
  std::vector<V3>                   positions, normals;
  std::vector<std::array<float, 2>> uvs;
  // the weights in the vertex data, four per vertex, with indices that may be the partition's
  std::vector<std::array<float, 4>>   weights;
  std::vector<std::array<uint8_t, 4>> boneIndices;
};

// `count` vertices in BSVertexDataSSE (stream 100: positions always at full precision).
Vertices readVertices(Reader& r, const VertexDesc& desc, size_t count)
{
  Vertices     out;
  const size_t stride = desc.size();
  if(stride == 0 || !desc.has(kVfVertex))
    throw std::runtime_error("its vertices have no positions");
  r.count(count, stride);
  out.positions.resize(count);
  if(desc.has(kVfUv))
    out.uvs.resize(count);
  if(desc.has(kVfNormal))
    out.normals.resize(count);
  if(desc.has(kVfSkinned))
  {
    out.weights.resize(count);
    out.boneIndices.resize(count);
  }
  for(size_t v = 0; v < count; v++)
  {
    const size_t base = r.at();
    out.positions[v]  = r.v3();
    if(desc.has(kVfUv))
    {
      r.seek(base + desc.uvAt());
      const float u = halfToFloat(r.u16());
      out.uvs[v]    = {u, halfToFloat(r.u16())};
    }
    if(desc.has(kVfNormal))
    {
      r.seek(base + desc.normalAt());
      const float x  = unitByte(r.u8());
      const float y  = unitByte(r.u8());
      out.normals[v] = {x, y, unitByte(r.u8())};
    }
    if(desc.has(kVfSkinned))
    {
      r.seek(base + desc.skinAt());
      for(float& w : out.weights[v])
        w = halfToFloat(r.u16());
      for(uint8_t& b : out.boneIndices[v])
        b = r.u8();
    }
    r.seek(base + stride);
  }
  return out;
}

struct Partition
{
  std::vector<uint16_t>            bones, vertexMap;
  std::vector<float>               weights;      // numVertices x perVertex
  std::vector<uint8_t>             boneIndices;  // the same, into `bones`
  int                              perVertex = 0;
  std::vector<std::array<uint16_t, 3>> triangles;  // the partition's vertices
  std::vector<std::array<uint16_t, 3>> shapeTriangles;  // stream 100's copy in the shape's vertices
};

struct SkinPartition
{
  Vertices               vertices;
  std::vector<Partition> partitions;
};

std::vector<std::array<uint16_t, 3>> stripTriangles(const std::vector<std::vector<uint16_t>>& strips)
{
  std::vector<std::array<uint16_t, 3>> out;
  for(const auto& s : strips)
    for(size_t i = 2; i < s.size(); i++)
    {
      std::array<uint16_t, 3> t = (i % 2) ? std::array<uint16_t, 3>{s[i - 1], s[i - 2], s[i]} : std::array<uint16_t, 3>{s[i - 2], s[i - 1], s[i]};
      if(t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        out.push_back(t);
    }
  return out;
}

SkinPartition readSkinPartition(Reader r)
{
  SkinPartition out;
  const uint32_t count = r.u32();
  const uint32_t dataSize = r.u32();
  const uint32_t vertexSize = r.u32();
  VertexDesc     desc{r.u64()};
  if(dataSize > 0)
  {
    if(vertexSize == 0)
      throw std::runtime_error("its skin partition has vertices of no size");
    out.vertices = readVertices(r, desc, dataSize / vertexSize);
  }
  r.count(count, 10);
  for(uint32_t k = 0; k < count; k++)
  {
    Partition      p;
    const uint16_t nVerts = r.u16(), nTris = r.u16(), nBones = r.u16(), nStrips = r.u16();
    p.perVertex           = r.u16();
    for(size_t i = 0, n = r.count(nBones, 2); i < n; i++)
      p.bones.push_back(r.u16());
    if(r.u8())
      for(size_t i = 0, n = r.count(nVerts, 2); i < n; i++)
        p.vertexMap.push_back(r.u16());
    if(r.u8())
    {
      const size_t n = r.count(size_t(nVerts) * size_t(p.perVertex), 4);
      p.weights.resize(n);
      for(float& w : p.weights)
        w = r.f32();
    }
    std::vector<uint16_t> stripLengths;
    for(size_t i = 0, n = r.count(nStrips, 2); i < n; i++)
      stripLengths.push_back(r.u16());
    if(r.u8())  // has faces
    {
      if(nStrips)
      {
        std::vector<std::vector<uint16_t>> strips;
        for(uint16_t len : stripLengths)
        {
          std::vector<uint16_t> s(r.count(len, 2));
          for(uint16_t& i : s)
            i = r.u16();
          strips.push_back(std::move(s));
        }
        p.triangles = stripTriangles(strips);
      }
      else
        for(size_t i = 0, n = r.count(nTris, 6); i < n; i++)
          p.triangles.push_back({r.u16(), r.u16(), r.u16()});
    }
    if(r.u8())
    {
      p.boneIndices.resize(r.count(size_t(nVerts) * size_t(p.perVertex), 1));
      for(uint8_t& b : p.boneIndices)
        b = r.u8();
    }
    r.u8();  // LOD level
    r.u8();  // global vertex buffer
    r.u64();  // vertex desc
    for(size_t i = 0, n = r.count(nTris, 6); i < n; i++)
      p.shapeTriangles.push_back({r.u16(), r.u16(), r.u16()});
    out.partitions.push_back(std::move(p));
  }
  return out;
}

Shader readShader(Reader r, const Parser& p)
{
  Shader s;
  s.type = r.u32();
  r.i32();  // name
  const size_t nExtra = r.count(r.u32(), 4);
  r.skip(nExtra * 4);
  r.i32();  // controller
  s.flags1      = r.u32();
  s.flags2      = r.u32();
  s.uvOffset    = {r.f32(), r.f32()};
  s.uvScale     = {r.f32(), r.f32()};
  const int32_t textureSet = r.i32();
  s.emissive               = r.v3();
  s.emissiveMultiple       = r.f32();
  r.u32();  // texture clamp mode
  s.alpha = r.f32();
  r.f32();  // refraction strength
  s.glossiness       = r.f32();
  s.specularColor    = r.v3();
  s.specularStrength = r.f32();
  r.f32();  // lighting effect 1
  r.f32();  // lighting effect 2
  if(s.type == 1)
    s.envMapScale = r.f32();
  if(const Block* b = p.block(textureSet); b && b->type == "BSShaderTextureSet")
  {
    Reader t = p.at(*b);
    for(size_t i = 0, n = t.count(t.u32(), 4); i < n; i++)
      s.textures.push_back(t.sized());
  }
  return s;
}

Alpha readAlpha(Reader r)
{
  r.i32();  // name
  const size_t nExtra = r.count(r.u32(), 4);
  r.skip(nExtra * 4);
  r.i32();  // controller
  Alpha a;
  a.flags     = r.u16();
  a.threshold = r.u8();
  return a;
}

// The shape's skin: bones, their bind data and the weights per vertex (NiSkinData's lists where it
// keeps them, else the partitions'), and from the partitions the vertices and triangles when the
// shape keeps none of its own.
void readSkin(Shape& shape, int32_t skinRef, const Parser& p, const std::unordered_map<int32_t, int>& nodeOfBlock, Vertices& vertices,
              std::vector<uint32_t>& triangles)
{
  const Block* sb = p.block(skinRef);
  if(!sb)
    return;
  if(sb->type != "NiSkinInstance" && sb->type != "BSDismemberSkinInstance")
    throw std::runtime_error("its skin is a " + sb->type + ", which this reader doesn't know");
  Reader        r         = p.at(*sb);
  const int32_t dataRef   = r.i32();
  const int32_t partRef   = r.i32();
  r.i32();  // skeleton root
  const size_t nBones = r.count(r.u32(), 4);
  std::vector<int32_t> boneBlocks(nBones);
  for(int32_t& b : boneBlocks)
    b = r.i32();
  for(int32_t b : boneBlocks)
  {
    const auto it = nodeOfBlock.find(b);
    if(it == nodeOfBlock.end())
      throw std::runtime_error("a bone of its skin isn't a node in the file");
    shape.bones.push_back(it->second);
  }
  // the game's biped slots, a flag and a slot a partition
  if(sb->type == "BSDismemberSkinInstance")
  {
    const size_t nParts = r.count(r.u32(), 4);
    for(size_t i = 0; i < nParts; i++)
    {
      r.u16();  // flags
      shape.bodyParts.push_back(r.u16());
    }
  }

  std::optional<SkinPartition> part;
  if(const Block* pb = p.block(partRef); pb && pb->type == "NiSkinPartition")
    part = readSkinPartition(p.at(*pb));
  if(part && vertices.positions.empty())
    vertices = std::move(part->vertices);
  if(part && triangles.empty())
  {
    const size_t n = vertices.positions.size();
    for(const Partition& pt : part->partitions)
    {
      // stream 100 copies each triangle in the shape's own vertices; without the copy the partition's
      // go through its vertex map
      const bool copied = pt.shapeTriangles.size() == pt.triangles.size() && !pt.shapeTriangles.empty();
      for(size_t k = 0; k < (copied ? pt.shapeTriangles.size() : pt.triangles.size()); k++)
      {
        std::array<uint16_t, 3> t = copied ? pt.shapeTriangles[k] : pt.triangles[k];
        if(!copied && !pt.vertexMap.empty())
          for(uint16_t& i : t)
            i = i < pt.vertexMap.size() ? pt.vertexMap[i] : uint16_t(0xFFFF);
        if(t[0] < n && t[1] < n && t[2] < n)
          triangles.insert(triangles.end(), {t[0], t[1], t[2]});
      }
    }
  }

  const size_t                                 nv = vertices.positions.size();
  std::vector<std::vector<std::pair<uint16_t, float>>> influences(nv);
  bool                                         fromData = false;
  if(const Block* db = p.block(dataRef); db && db->type == "NiSkinData")
  {
    Reader d           = p.at(*db);
    shape.skinTransform = readNiTransform(d);
    const uint32_t nb  = d.u32();
    const bool     hasWeights = d.u8() != 0;
    if(nb != nBones)
      throw std::runtime_error("its skin data lists " + std::to_string(nb) + " bones for " + std::to_string(nBones));
    for(uint32_t b = 0; b < nb; b++)
    {
      shape.skinToBone.push_back(readNiTransform(d));
      d.skip(16);  // bounding sphere
      const uint16_t n = d.u16();
      if(!hasWeights)
        continue;
      d.count(n, 6);
      for(uint16_t i = 0; i < n; i++)
      {
        const uint16_t v = d.u16();
        const float    w = d.f32();
        if(v < nv && w > 0)
          influences[v].push_back({uint16_t(b), w});
      }
    }
    fromData = hasWeights;
  }
  else
    throw std::runtime_error("its skin has no bind data (NiSkinData)");
  // the partitions' weights, when the skin data keeps none: each partition's bone indices are its own
  if(!fromData && part)
    for(const Partition& pt : part->partitions)
      for(size_t k = 0; k < pt.vertexMap.size(); k++)
      {
        const size_t v = pt.vertexMap[k];
        if(v >= nv)
          continue;
        for(int j = 0; j < pt.perVertex; j++)
        {
          const size_t at = k * size_t(pt.perVertex) + size_t(j);
          if(at >= pt.weights.size() || at >= pt.boneIndices.size() || pt.weights[at] <= 0 || pt.boneIndices[at] >= pt.bones.size())
            continue;
          influences[v].push_back({pt.bones[pt.boneIndices[at]], pt.weights[at]});
        }
      }
  shape.weightStart.assign(1, 0);
  for(const auto& list : influences)
  {
    for(const auto& [b, w] : list)
    {
      shape.weightBone.push_back(b);
      shape.weight.push_back(w);
    }
    shape.weightStart.push_back(uint32_t(shape.weight.size()));
  }
}

Shape readShape(const Block& b, const Parser& p, const std::unordered_map<int32_t, int>& nodeOfBlock)
{
  Reader         r = p.at(b);
  const AvObject o = readAvObject(r, p);
  Shape          s;
  s.name  = o.name;
  s.flags = o.flags;
  s.local = o.local;
  r.skip(16);  // bounding sphere
  const int32_t skinRef   = r.i32();
  const int32_t shaderRef = r.i32();
  const int32_t alphaRef  = r.i32();
  VertexDesc    desc{r.u64()};
  const uint16_t nTris    = r.u16();
  const uint16_t nVerts   = r.u16();
  const uint32_t dataSize = r.u32();
  Vertices              vertices;
  std::vector<uint32_t> triangles;
  if(dataSize > 0)
  {
    vertices = readVertices(r, desc, nVerts);
    r.count(nTris, 6);
    for(uint32_t i = 0; i < uint32_t(nTris) * 3; i++)
      triangles.push_back(r.u16());
  }
  for(int32_t e : o.extra)
    if(const Block* eb = p.block(e); eb && eb->type == "NiStringExtraData")
    {
      Reader      er   = p.at(*eb);
      std::string name = p.str(er.i32());
      s.extra.emplace_back(std::move(name), p.str(er.i32()));
    }
  readSkin(s, skinRef, p, nodeOfBlock, vertices, triangles);
  for(uint32_t i : triangles)
    if(i >= vertices.positions.size())
      throw std::runtime_error("a triangle of " + s.name + " names a vertex it doesn't have");
  s.positions = std::move(vertices.positions);
  s.normals   = std::move(vertices.normals);
  s.uvs       = std::move(vertices.uvs);
  s.triangles = std::move(triangles);
  if(!s.skinned())
  {
    s.weightStart.clear();
    s.weightBone.clear();
    s.weight.clear();
  }
  if(const Block* sb = p.block(shaderRef); sb && sb->type == "BSLightingShaderProperty")
    s.shader = readShader(p.at(*sb), p);
  if(const Block* ab = p.block(alphaRef); ab && ab->type == "NiAlphaProperty")
    s.alpha = readAlpha(p.at(*ab));
  return s;
}

}  // namespace

V3 Transform::apply(const V3& p) const
{
  return {s * (r[0] * p[0] + r[1] * p[1] + r[2] * p[2]) + t[0], s * (r[3] * p[0] + r[4] * p[1] + r[5] * p[2]) + t[1],
          s * (r[6] * p[0] + r[7] * p[1] + r[8] * p[2]) + t[2]};
}

Transform Transform::operator*(const Transform& b) const
{
  Transform out;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      out.r[size_t(i * 3 + j)] = r[size_t(i * 3)] * b.r[size_t(j)] + r[size_t(i * 3 + 1)] * b.r[size_t(3 + j)] + r[size_t(i * 3 + 2)] * b.r[size_t(6 + j)];
  out.s = s * b.s;
  out.t = apply(b.t);
  return out;
}

Transform Transform::inverse() const
{
  Transform out;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      out.r[size_t(i * 3 + j)] = r[size_t(j * 3 + i)];
  out.s = s != 0 ? 1 / s : 1;
  // p = R^T (p' - t) / s
  const V3 rt = Transform{out.r, {0, 0, 0}, 1}.apply(t);
  out.t       = {-rt[0] * out.s, -rt[1] * out.s, -rt[2] * out.s};
  return out;
}

const std::string& Shader::texture(size_t slot) const
{
  static const std::string none;
  return slot < textures.size() ? textures[slot] : none;
}

Transform File::world(int node) const
{
  Transform out;
  for(int guard = 0; node >= 0 && size_t(node) < nodes.size() && guard < 1024; guard++)
  {
    out  = nodes[size_t(node)].local * out;
    node = nodes[size_t(node)].parent;
  }
  return out;
}

int File::find(const std::string& name) const
{
  for(size_t i = 0; i < nodes.size(); i++)
    if(nodes[i].name == name)
      return int(i);
  return -1;
}

float halfToFloat(uint16_t h)
{
  const uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t       exp  = (h >> 10) & 0x1F;
  uint32_t       mant = h & 0x3FF;
  uint32_t       bits;
  if(exp == 0)
  {
    if(mant == 0)
      bits = sign;
    else
    {
      // subnormal: normalise it
      exp = 127 - 15 + 1;
      while(!(mant & 0x400))
      {
        mant <<= 1;
        exp--;
      }
      bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
    }
  }
  else if(exp == 0x1F)
    bits = sign | 0x7F800000 | (mant << 13);
  else
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

File read(std::span<const uint8_t> bytes)
{
  Reader r(bytes);
  // "Gamebryo File Format, Version 20.2.0.7\n"
  std::string line;
  for(size_t i = 0; i < std::min<size_t>(bytes.size(), 128); i++)
  {
    if(bytes[i] == '\n')
    {
      line = std::string(reinterpret_cast<const char*>(bytes.data()), i);
      r.seek(i + 1);
      break;
    }
  }
  if(line.rfind("Gamebryo File Format", 0) != 0 && line.rfind("NetImmerse File Format", 0) != 0)
    throw std::runtime_error("it isn't a NIF file");
  const uint32_t version = r.u32();
  if(version != kVersion)
    throw std::runtime_error("it is a NIF of another game (version " + line.substr(line.rfind(' ') + 1) + "); only Skyrim Special Edition's are read");
  if(r.u8() != 1)
    throw std::runtime_error("it is a big-endian NIF");
  File out;
  out.userVersion       = r.u32();
  const uint32_t nBlocks = r.u32();
  out.bsVersion          = out.userVersion >= 3 ? r.u32() : 0;
  if(out.bsVersion == 83)
    throw std::runtime_error("it is an original Skyrim (Legendary Edition) mesh; convert it to Special Edition first (Cathedral Assets Optimizer, or Outfit Studio's export)");
  if(out.bsVersion != kSse)
    throw std::runtime_error("it is a Bethesda NIF of stream " + std::to_string(out.bsVersion) + "; only Skyrim Special Edition's (100) are read");
  r.shortString();  // author
  out.exporter = r.shortString();
  r.shortString();  // export script

  Parser p;
  p.bytes     = bytes;
  p.bsVersion = out.bsVersion;
  std::vector<std::string> types(r.count(r.u16(), 4));
  for(std::string& t : types)
    t = r.sized();
  p.blocks.resize(r.count(nBlocks, 6));
  for(Block& b : p.blocks)
  {
    const uint16_t t = r.u16() & 0x7FFF;
    if(t >= types.size())
      throw std::runtime_error("a block names a type the file doesn't list");
    b.type = types[t];
  }
  for(Block& b : p.blocks)
    b.size = r.u32();
  p.strings.resize(r.count(r.u32(), 4));
  r.u32();  // longest string
  for(std::string& s : p.strings)
    s = r.sized();
  r.skip(r.count(r.u32(), 4) * 4);  // groups
  size_t at = r.at();
  for(Block& b : p.blocks)
  {
    b.at = at;
    if(b.size > bytes.size() - std::min(at, bytes.size()))
      throw std::runtime_error("the file is cut short (its blocks run past its end)");
    at += b.size;
  }

  // the node tree: every node, then each one's children
  std::unordered_map<int32_t, int> nodeOfBlock;
  std::vector<std::pair<int, std::vector<int32_t>>> childrenOf;
  for(size_t k = 0; k < p.blocks.size(); k++)
  {
    if(!isNode(p.blocks[k].type))
      continue;
    Reader         nr = p.at(p.blocks[k]);
    const AvObject o  = readAvObject(nr, p);
    Node           n;
    n.name  = o.name;
    n.flags = o.flags;
    n.local = o.local;
    std::vector<int32_t> kids(nr.count(nr.u32(), 4));
    for(int32_t& c : kids)
      c = nr.i32();
    nodeOfBlock[int32_t(k)] = int(out.nodes.size());
    childrenOf.emplace_back(int(out.nodes.size()), std::move(kids));
    out.nodes.push_back(std::move(n));
  }
  std::unordered_map<int32_t, int> parentOfBlock;
  for(const auto& [node, kids] : childrenOf)
    for(int32_t c : kids)
    {
      if(const auto it = nodeOfBlock.find(c); it != nodeOfBlock.end())
        out.nodes[size_t(it->second)].parent = node;
      parentOfBlock[c] = node;
    }
  for(size_t k = 0; k < p.blocks.size(); k++)
  {
    const std::string& type = p.blocks[k].type;
    if(type == "NiTriShape" || type == "NiTriStrips")
      throw std::runtime_error("it keeps its meshes as " + type + ", as original Skyrim does; convert it to Special Edition first");
    // (a BSDynamicTriShape, a face, keeps its positions elsewhere: not read)
    if(type != "BSTriShape")
      continue;
    try
    {
      Shape s = readShape(p.blocks[k], p, nodeOfBlock);
      if(const auto it = parentOfBlock.find(int32_t(k)); it != parentOfBlock.end())
        s.parent = it->second;
      out.shapes.push_back(std::move(s));
    }
    catch(const std::exception& e)
    {
      throw std::runtime_error("shape " + std::to_string(k) + ": " + e.what());
    }
  }
  return out;
}

File readFile(const std::filesystem::path& path)
{
  std::ifstream in(path, std::ios::binary);
  if(!in)
    throw std::runtime_error("it can't be opened");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return read(bytes);
}

}  // namespace fv::nif
