#include <figure/gltf/gltf_build.hpp>

#include <figure/materials/hair_strands.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

namespace fv::build {

namespace {

class Writer
{
public:
  explicit Writer(tinygltf::Model& model)
      : m_model(model)
  {
    if(m_model.buffers.empty())
      m_model.buffers.emplace_back();
  }

  // A buffer view over `bytes`, aligned to 4; returns its index.
  int view(const void* bytes, size_t size, int target, int stride = 0)
  {
    auto& data = m_model.buffers[0].data;
    while(data.size() % 4)
      data.push_back(0);
    const size_t offset = data.size();
    data.resize(offset + size);
    if(size)
      std::memcpy(data.data() + offset, bytes, size);
    tinygltf::BufferView v;
    v.buffer     = 0;
    v.byteOffset = offset;
    v.byteLength = size;
    v.byteStride = size_t(stride);
    v.target     = target;
    m_model.bufferViews.push_back(v);
    return int(m_model.bufferViews.size()) - 1;
  }

  int accessor(int view, int componentType, size_t count, int type, std::vector<double> min = {}, std::vector<double> max = {})
  {
    tinygltf::Accessor a;
    a.bufferView    = view;
    a.componentType = componentType;
    a.count         = count;
    a.type          = type;
    a.minValues     = std::move(min);
    a.maxValues     = std::move(max);
    m_model.accessors.push_back(a);
    return int(m_model.accessors.size()) - 1;
  }

private:
  tinygltf::Model& m_model;
};

tinygltf::Value extras(const std::string& fvId)
{
  tinygltf::Value::Object o;
  o["fvId"] = tinygltf::Value(fvId);
  return tinygltf::Value(o);
}

}  // namespace

std::vector<TextureRequest> textureRequests(const Material& m, mat::SkinMode skin)
{
  mat::LowerOptions o;
  o.skin = skin;
  return mat::lower(m.surface, m.edits, o).requests;
}

std::vector<TextureRequest> textureRequests(const SceneGraph& scene, mat::SkinMode skin)
{
  std::vector<TextureRequest> out;
  std::set<std::string>       seen;
  for(const Material& m : scene.materials)
    for(TextureRequest& req : textureRequests(m, skin))
      if(seen.insert(requestKey(req)).second)
        out.push_back(std::move(req));
  return out;
}

tinygltf::Material materialToGltf(const Material& m, const mat::LowerOptions& options,
                                  const std::function<mat::ResolvedTexture(const TextureRequest&)>& resolve, std::vector<std::string>& extensionsUsed)
{
  tinygltf::Material g = mat::toGltf(mat::lower(m.surface, m.edits, options), resolve, extensionsUsed);
  g.name               = m.name;
  return g;
}

std::vector<float> inverseBindMatrices(const Mesh& m)
{
  std::vector<float> ibm;
  for(size_t j = 0; j < m.bones.size(); j++)
  {
    const Mat4 b = m.boneInverses[j] * m.bindMatrix;
    for(double e : b.e)
      ibm.push_back(float(e));
  }
  return ibm;
}

BuildResult buildGltf(const SceneGraph& scene, int root, const BuildOptions& options)
{
  BuildResult      out;
  tinygltf::Model& model = out.model;
  model.asset.version   = "2.0";
  model.asset.generator = "Figure Viewer";
  Writer w(model);
  out.nodeOf.assign(scene.nodes.size(), -1);
  out.meshOf.assign(scene.meshes.size(), -1);
  out.skinOf.assign(scene.meshes.size(), -1);

  // nodes, depth first as traversal meets them
  std::vector<int> order;
  {
    std::vector<int> stack{root};
    while(!stack.empty())
    {
      const int i = stack.back();
      stack.pop_back();
      out.nodeOf[i] = int(order.size());
      order.push_back(i);
      const auto& children = scene.nodes[i].children;
      for(auto it = children.rbegin(); it != children.rend(); ++it)
        stack.push_back(*it);
    }
  }
  for(int i : order)
  {
    const Node&    n = scene.nodes[i];
    tinygltf::Node g;
    g.name        = n.name;
    g.translation = {n.position.x, n.position.y, n.position.z};
    g.rotation    = {n.quaternion.x, n.quaternion.y, n.quaternion.z, n.quaternion.w};
    g.scale       = {n.scale.x, n.scale.y, n.scale.z};
    g.extras      = extras("n:" + std::to_string(i));
    for(int c : n.children)
      g.children.push_back(out.nodeOf[c]);
    if(!n.visible)
    {
      // KHR_node_visibility, which the renderer reads
      tinygltf::Value::Object vis;
      vis["visible"] = tinygltf::Value(false);
      g.extensions["KHR_node_visibility"] = tinygltf::Value(vis);
    }
    model.nodes.push_back(std::move(g));
  }
  model.scenes.emplace_back();
  model.scenes[0].nodes = {out.nodeOf[root]};
  model.defaultScene    = 0;

  // what the materials need from the meshes: the area an emitter spreads its flux over, and the
  // direction of hair strands, which is measured from the cards' UVs as the glTF carries them
  const std::vector<Mat4> worlds = scene.worlds();
  std::vector<double>     emitArea(scene.materials.size(), 0.0);
  struct HairPart
  {
    std::vector<float>              uv;
    std::vector<mat::StrandMesh::Range> ranges;
    const mesh::BuiltMesh*          geo;
  };
  std::vector<std::vector<HairPart>> hairParts(scene.materials.size());
  for(int i : order)
  {
    const Node& n = scene.nodes[i];
    if(n.kind != NodeKind::eMesh || n.mesh < 0 || !scene.meshes[n.mesh].geo)
      continue;
    const Mesh&            m   = scene.meshes[n.mesh];
    const mesh::BuiltMesh& geo = *m.geo;
    const Mat4&            w4  = worlds[i];
    // the largest axis scale of the node's world matrix (three's getMaxScaleOnAxis)
    double s2 = 0;
    for(int c = 0; c < 3; c++)
      s2 = std::max(s2, w4.e[c * 4] * w4.e[c * 4] + w4.e[c * 4 + 1] * w4.e[c * 4 + 1] + w4.e[c * 4 + 2] * w4.e[c * 4 + 2]);
    std::map<int, std::vector<mat::StrandMesh::Range>> byMaterial;
    for(const mesh::Group& group : geo.groups)
    {
      const int mi = group.materialIndex < int(m.materials.size()) ? m.materials[group.materialIndex] : -1;
      if(mi < 0 || mi >= int(scene.materials.size()))
        continue;
      const mat::Surface& sf = scene.materials[mi].surface;
      if(mat::needsArea(sf))
      {
        double area = 0;
        const uint32_t* idx = geo.fullIndex.data() + group.start;
        for(uint32_t t = 0; t + 2 < group.count; t += 3)
        {
          const float* a = &geo.position[idx[t] * 3];
          const float* b = &geo.position[idx[t + 1] * 3];
          const float* c = &geo.position[idx[t + 2] * 3];
          const double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
          area += 0.5 * std::sqrt((uy * vz - uz * vy) * (uy * vz - uz * vy) + (uz * vx - ux * vz) * (uz * vx - ux * vz) + (ux * vy - uy * vx) * (ux * vy - uy * vx));
        }
        emitArea[mi] += area * s2;
      }
      if(sf.kind == mat::Kind::eHair)
        byMaterial[mi].push_back({group.start, group.count});
    }
    for(auto& [mi, ranges] : byMaterial)
    {
      HairPart part{geo.uv, std::move(ranges), &geo};
      if(!options.texturesFlipped)
        for(size_t v = 1; v < part.uv.size(); v += 2)
          part.uv[v] = 1 - part.uv[v];
      hairParts[mi].push_back(std::move(part));
    }
  }
  out.lowering.assign(scene.materials.size(), {});
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
  {
    mat::LowerOptions& lo = out.lowering[mi];
    lo.skin               = options.skin;
    lo.emission           = mat::emissionStrength(scene.materials[mi].surface, emitArea[mi], options.exposureValue, options.iso);
    // one material on several meshes is one glTF material, so those meshes vote together
    std::vector<mat::StrandVote> votes;
    for(const HairPart& p : hairParts[mi])
    {
      mat::StrandMesh part{p.geo->position, p.uv, p.geo->fullIndex, p.ranges};
      const auto      v = mat::strandVotes(part);
      votes.insert(votes.end(), v.begin(), v.end());
    }
    lo.anisotropyRotation = mat::strandRotation(votes);
  }

  // materials, and the textures they ask for
  std::map<std::string, mat::ResolvedTexture> textureOf;  // request key -> texture
  bool                                        usesBasisu = false;
  auto                                        texture    = [&](const TextureRequest& req) -> mat::ResolvedTexture {
    if(req.rel.empty() || !options.texture)
      return {};
    const std::string key = requestKey(req);
    if(auto it = textureOf.find(key); it != textureOf.end())
      return it->second;
    const auto image = options.texture(req);
    if(!image)
      return textureOf[key] = {};
    tinygltf::Image img;
    img.uri      = image->uri;
    img.mimeType = image->mimeType;
    img.name     = req.rel.substr(req.rel.find_last_of('/') + 1);
    model.images.push_back(std::move(img));
    if(model.samplers.empty())
    {
      tinygltf::Sampler s;
      // Genesis UVs use one 0..1 tile per surface group (UDIM-style); repeat maps each tile home
      s.wrapS     = TINYGLTF_TEXTURE_WRAP_REPEAT;
      s.wrapT     = TINYGLTF_TEXTURE_WRAP_REPEAT;
      s.magFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
      s.minFilter = TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR;
      model.samplers.push_back(s);
    }
    tinygltf::Texture t;
    t.sampler = 0;
    if(image->mimeType == "image/ktx2")
    {
      tinygltf::Value::Object basisu;
      basisu["source"] = tinygltf::Value(int(model.images.size()) - 1);
      t.extensions["KHR_texture_basisu"] = tinygltf::Value(basisu);
      usesBasisu = true;
    }
    else
      t.source = int(model.images.size()) - 1;
    model.textures.push_back(std::move(t));
    return textureOf[key] = {int(model.textures.size()) - 1, image->flipGreen};
  };
  // strand hair ribbons face one way, so their surfaces are seen from both sides
  std::vector<bool> onRibbon(scene.materials.size(), false);
  for(const Mesh& m : scene.meshes)
    if(m.geo && m.geo->ribbon)
      for(int mi : m.materials)
        if(mi >= 0 && mi < int(onRibbon.size()))
          onRibbon[mi] = true;
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
  {
    tinygltf::Material g = materialToGltf(scene.materials[mi], out.lowering[mi], texture, model.extensionsUsed);
    if(onRibbon[mi])
      g.doubleSided = true;
    g.extras = extras("m:" + std::to_string(mi));
    model.materials.push_back(std::move(g));
  }
  if(usesBasisu)
  {
    model.extensionsUsed.push_back("KHR_texture_basisu");
    model.extensionsRequired.push_back("KHR_texture_basisu");
  }

  // meshes and skins
  for(int i : order)
  {
    const Node& n = scene.nodes[i];
    if(n.kind != NodeKind::eMesh || n.mesh < 0)
      continue;
    const Mesh&            m   = scene.meshes[n.mesh];
    const mesh::BuiltMesh& geo = *m.geo;
    const size_t           count = geo.position.size() / 3;
    if(!count || geo.groups.empty())
      continue;

    std::vector<double> min{1e300, 1e300, 1e300}, max{-1e300, -1e300, -1e300};
    for(size_t v = 0; v < count; v++)
      for(int c = 0; c < 3; c++)
      {
        min[c] = std::min(min[c], double(geo.position[v * 3 + c]));
        max[c] = std::max(max[c], double(geo.position[v * 3 + c]));
      }
    const int position = w.accessor(w.view(geo.position.data(), geo.position.size() * 4, TINYGLTF_TARGET_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_FLOAT,
                                     count, TINYGLTF_TYPE_VEC3, min, max);
    const int normal = w.accessor(w.view(geo.normal.data(), geo.normal.size() * 4, TINYGLTF_TARGET_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_FLOAT,
                                  count, TINYGLTF_TYPE_VEC3);
    std::vector<float> uv = geo.uv;
    if(!options.texturesFlipped)
      for(size_t v = 0; v < count; v++)
        uv[v * 2 + 1] = 1 - uv[v * 2 + 1];
    const int texcoord = w.accessor(w.view(uv.data(), uv.size() * 4, TINYGLTF_TARGET_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_FLOAT, count, TINYGLTF_TYPE_VEC2);
    int joints = -1, weights = -1;
    if(m.skinned() && m.skin.skinIndex.size() == count * 4)
    {
      joints  = w.accessor(w.view(m.skin.skinIndex.data(), m.skin.skinIndex.size() * 2, TINYGLTF_TARGET_ARRAY_BUFFER),
                           TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, count, TINYGLTF_TYPE_VEC4);
      weights = w.accessor(w.view(m.skin.skinWeight.data(), m.skin.skinWeight.size() * 4, TINYGLTF_TARGET_ARRAY_BUFFER),
                           TINYGLTF_COMPONENT_TYPE_FLOAT, count, TINYGLTF_TYPE_VEC4);
    }

    tinygltf::Mesh gm;
    gm.name   = m.name;
    gm.extras = extras("g:" + std::to_string(n.mesh));
    for(const mesh::Group& group : geo.groups)
    {
      const uint32_t* first = geo.fullIndex.data() + group.start;
      tinygltf::Primitive p;
      if(count <= 65535)
      {
        std::vector<uint16_t> idx(first, first + group.count);
        p.indices = w.accessor(w.view(idx.data(), idx.size() * 2, TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT,
                               idx.size(), TINYGLTF_TYPE_SCALAR);
      }
      else
        p.indices = w.accessor(w.view(first, size_t(group.count) * 4, TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,
                               group.count, TINYGLTF_TYPE_SCALAR);
      p.attributes["POSITION"]   = position;
      p.attributes["NORMAL"]     = normal;
      p.attributes["TEXCOORD_0"] = texcoord;
      if(joints >= 0)
      {
        p.attributes["JOINTS_0"]  = joints;
        p.attributes["WEIGHTS_0"] = weights;
      }
      p.material = group.materialIndex < int(m.materials.size()) ? m.materials[group.materialIndex] : -1;
      p.mode     = TINYGLTF_MODE_TRIANGLES;
      gm.primitives.push_back(std::move(p));
    }
    model.meshes.push_back(std::move(gm));
    out.meshOf[n.mesh]                = int(model.meshes.size()) - 1;
    model.nodes[out.nodeOf[i]].mesh = out.meshOf[n.mesh];

    if(joints >= 0)
    {
      // GLTFExporter's inverse bind matrices: bone inverse times the bind matrix
      tinygltf::Skin     skin;
      for(int bone : m.bones)
        skin.joints.push_back(out.nodeOf[bone]);
      const std::vector<float> ibm = inverseBindMatrices(m);
      skin.inverseBindMatrices = w.accessor(w.view(ibm.data(), ibm.size() * 4, 0), TINYGLTF_COMPONENT_TYPE_FLOAT, m.bones.size(), TINYGLTF_TYPE_MAT4);
      model.skins.push_back(std::move(skin));
      model.nodes[out.nodeOf[i]].skin = int(model.skins.size()) - 1;
      out.skinOf[n.mesh]              = int(model.skins.size()) - 1;
    }
  }
  model.buffers[0].name = "figure";
  out.textureOf         = std::move(textureOf);
  return out;
}

}  // namespace fv::build
