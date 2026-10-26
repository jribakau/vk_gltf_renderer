#pragma once

// A built glTF read back against the scene it was built from: every mesh's attributes, indices and
// inverse bind matrices are the scene's own bytes, the views are aligned and apart, and the bytes
// between them are zero. For the tests and fv_parity: a figure's positions differ in their last bits
// from one process to the next (a morph's targets add up in the order of their addresses), so two
// builds are compared by what each holds, not by a hash.

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <figure/gltf/gltf_build.hpp>
#include <figure/mesh/mesh_build.hpp>

namespace fv::build {

// What does not match, empty when the build is the scene's.
inline std::vector<std::string> checkBuild(const SceneGraph& scene, const BuildResult& built, const BuildOptions& options)
{
  std::vector<std::string> problems;
  const tinygltf::Model&   model = built.model;
  if(model.buffers.size() != 1)
  {
    problems.push_back("not one buffer");
    return problems;
  }
  const std::vector<unsigned char>& data = model.buffers[0].data;
  // an accessor's bytes, or none when its view does not lie in the buffer
  const auto bytesOf = [&](int accessor) -> std::pair<const unsigned char*, size_t> {
    if(accessor < 0 || accessor >= int(model.accessors.size()))
      return {nullptr, 0};
    const tinygltf::Accessor& a = model.accessors[size_t(accessor)];
    if(a.bufferView < 0 || a.bufferView >= int(model.bufferViews.size()))
      return {nullptr, 0};
    const tinygltf::BufferView& v = model.bufferViews[size_t(a.bufferView)];
    if(v.byteOffset + v.byteLength > data.size() || a.byteOffset > v.byteLength)
      return {nullptr, 0};
    return {data.data() + v.byteOffset + a.byteOffset, v.byteLength - a.byteOffset};
  };
  const auto same = [&](int accessor, const void* want, size_t size, const std::string& what) {
    const auto [got, length] = bytesOf(accessor);
    if(!got || length != size || (size && std::memcmp(got, want, size) != 0))
      problems.push_back(what);
  };

  std::set<int> meshes;
  for(size_t mi = 0; mi < scene.meshes.size() && mi < built.meshOf.size(); mi++)
  {
    const int g = built.meshOf[mi];
    if(g < 0 || !meshes.insert(g).second)
      continue;
    const std::string      name = "mesh " + std::to_string(mi) + " ";
    const Mesh&            m    = scene.meshes[mi];
    const mesh::BuiltMesh& geo  = *m.geo;
    const size_t           count = geo.position.size() / 3;
    const mesh::Drawn      drawn = mesh::drawnTriangles(geo);
    const tinygltf::Mesh&  gm    = model.meshes[size_t(g)];
    if(gm.primitives.size() != drawn.groups.size() || gm.primitives.empty())
    {
      problems.push_back(name + "primitives");
      continue;
    }
    const tinygltf::Primitive& first = gm.primitives.front();
    const auto attribute = [&](const char* key) { return first.attributes.count(key) ? first.attributes.at(key) : -1; };
    same(attribute("POSITION"), geo.position.data(), geo.position.size() * 4, name + "positions");
    same(attribute("NORMAL"), geo.normal.data(), geo.normal.size() * 4, name + "normals");
    std::vector<float> uv = geo.uv;
    if(!options.texturesFlipped)
      for(size_t v = 0; v < count; v++)
        uv[v * 2 + 1] = 1 - uv[v * 2 + 1];
    same(attribute("TEXCOORD_0"), uv.data(), uv.size() * 4, name + "UVs");
    // the bounds of the positions
    if(const int p = attribute("POSITION"); p >= 0)
    {
      std::vector<double> lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
      for(size_t v = 0; v < count; v++)
        for(int c = 0; c < 3; c++)
        {
          lo[c] = std::min(lo[c], double(geo.position[v * 3 + c]));
          hi[c] = std::max(hi[c], double(geo.position[v * 3 + c]));
        }
      if(model.accessors[size_t(p)].minValues != lo || model.accessors[size_t(p)].maxValues != hi)
        problems.push_back(name + "bounds");
    }
    const bool skinned = m.skinned() && m.skin.skinIndex.size() == count * 4;
    if(skinned)
    {
      same(attribute("JOINTS_0"), m.skin.skinIndex.data(), m.skin.skinIndex.size() * 2, name + "joints");
      same(attribute("WEIGHTS_0"), m.skin.skinWeight.data(), m.skin.skinWeight.size() * 4, name + "weights");
    }
    else if(attribute("JOINTS_0") >= 0)
      problems.push_back(name + "joints on a mesh with no skin");
    for(size_t k = 0; k < drawn.groups.size(); k++)
    {
      const mesh::Group&         group = drawn.groups[k];
      const tinygltf::Primitive& p     = gm.primitives[k];
      const uint32_t*            idx   = drawn.index.data() + group.start;
      if(count <= 65535)
      {
        const std::vector<uint16_t> want(idx, idx + group.count);
        same(p.indices, want.data(), want.size() * 2, name + "indices " + std::to_string(k));
      }
      else
        same(p.indices, idx, size_t(group.count) * 4, name + "indices " + std::to_string(k));
      const int material = group.materialIndex < int(m.materials.size()) ? m.materials[group.materialIndex] : -1;
      if(p.material != material || p.attributes != first.attributes)
        problems.push_back(name + "primitive " + std::to_string(k));
    }
    if(mi < built.skinOf.size() && built.skinOf[mi] >= 0)
    {
      const std::vector<float> ibm = inverseBindMatrices(m);
      same(model.skins[size_t(built.skinOf[mi])].inverseBindMatrices, ibm.data(), ibm.size() * 4, name + "inverse binds");
    }
    else if(skinned)
      problems.push_back(name + "no skin");
  }
  if(meshes.size() != model.meshes.size())
    problems.push_back("meshes that are no scene mesh's");

  // views aligned to 4 and apart, and nothing but zeros between them
  std::vector<std::pair<size_t, size_t>> spans;
  for(const tinygltf::BufferView& v : model.bufferViews)
  {
    if(v.byteOffset % 4)
      problems.push_back("a view not aligned to 4");
    spans.emplace_back(v.byteOffset, v.byteOffset + v.byteLength);
  }
  std::sort(spans.begin(), spans.end());
  size_t at = 0;
  for(const auto& [begin, end] : spans)
  {
    if(begin < at)
      problems.push_back("views that overlap");
    for(size_t i = at; i < begin && i < data.size(); i++)
      if(data[i])
      {
        problems.push_back("bytes between views that are not zero");
        break;
      }
    at = std::max(at, end);
  }
  if(at > data.size())
    problems.push_back("a view past the buffer's end");
  return problems;
}

}  // namespace fv::build
