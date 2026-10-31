#include <figure/gltf/gltf_build.hpp>

#include <figure/dson/material_preset.hpp>
#include <figure/dson/skin_materials.hpp>
#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/materials/hair_strands.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>

namespace fv::build {

namespace {

// The figure's buffer, laid out as views are asked for and filled at the end: a strand hair at SubD 2
// makes it ~400 MB, and appending view by view zero-filled each and copied everything before it
// again whenever the buffer grew, on one thread (0.18 s).
class Writer
{
public:
  explicit Writer(tinygltf::Model& model)
      : m_model(model)
  {
    if(m_model.buffers.empty())
      m_model.buffers.emplace_back();
    m_end = m_model.buffers[0].data.size();
  }

  // A buffer view over `bytes`, aligned to 4; returns its index. The bytes are copied by flush(), so
  // they stay where they are until then (keep() holds a temporary's).
  int view(const void* bytes, size_t size, int target, int stride = 0)
  {
    const size_t offset = (m_end + 3) & ~size_t(3);
    m_end               = offset + size;
    if(size)
      m_copies.push_back({offset, bytes, size});
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

  // Holds a temporary until flush(); its data.
  template <class T>
  const T* keep(std::vector<T> values)
  {
    auto held = std::make_shared<std::vector<T>>(std::move(values));
    m_kept.push_back(held);
    return held->data();
  }

  // The buffer at its size once (the padding zeros), and every view's bytes copied side by side.
  void flush()
  {
    auto& data = m_model.buffers[0].data;
    data.resize(m_end);
    constexpr size_t kPiece = size_t(4) << 20;
    struct Piece
    {
      unsigned char*       to;
      const unsigned char* from;
      size_t               size;
    };
    std::vector<Piece> pieces;
    for(const Copy& c : m_copies)
      for(size_t at = 0; at < c.size; at += kPiece)
        pieces.push_back({data.data() + c.offset + at, static_cast<const unsigned char*>(c.from) + at, std::min(kPiece, c.size - at)});
    jobs::parallelFor(pieces.size(), 1, [&](size_t begin, size_t end) {
      for(size_t i = begin; i < end; i++)
        std::memcpy(pieces[i].to, pieces[i].from, pieces[i].size);
    });
    m_copies.clear();
    m_kept.clear();
  }

private:
  struct Copy
  {
    size_t      offset;
    const void* from;
    size_t      size;
  };
  tinygltf::Model&                   m_model;
  size_t                             m_end = 0;
  std::vector<Copy>                  m_copies;
  std::vector<std::shared_ptr<void>> m_kept;
};

// The strand votes of one hair part (a mesh's drawn ranges of one material), measured once per
// geometry: a strand hair's cards take about a second to weld and measure, and the host builds the
// glTF again after every wardrobe change. A part is kept while its geometry lives; a dial that moves
// the hair leaves its cards' layout, and so the strand direction, as it was measured.
struct StrandPart
{
  std::weak_ptr<const mesh::BuiltMesh>                geo;
  std::vector<mat::StrandMesh::Range>                 ranges;
  size_t                                              indexCount = 0;
  bool                                                flipV      = false;
  std::shared_ptr<const std::vector<mat::StrandVote>> votes;
};
std::mutex                                                g_strandMutex;
std::map<const mesh::BuiltMesh*, std::vector<StrandPart>> g_strands;
// a geometry's weld, shared by its materials' votes (each used to weld the whole mesh again)
struct StrandWeld
{
  std::weak_ptr<const mesh::BuiltMesh>         geo;
  std::shared_ptr<const std::vector<uint32_t>> ids;
};
std::map<const mesh::BuiltMesh*, StrandWeld> g_welds;  // under g_strandMutex

std::shared_ptr<const std::vector<uint32_t>> weldOf(const std::shared_ptr<mesh::BuiltMesh>& geo)
{
  {
    std::lock_guard lock(g_strandMutex);
    if(auto it = g_welds.find(geo.get()); it != g_welds.end() && it->second.geo.lock() == geo)
      return it->second.ids;
  }
  auto ids = std::make_shared<const std::vector<uint32_t>>(mat::weldPositions(geo->position));
  std::lock_guard lock(g_strandMutex);
  std::erase_if(g_welds, [](const auto& kv) { return kv.second.geo.expired(); });
  auto& slot = g_welds[geo.get()];
  if(slot.geo.lock() != geo)  // first here (two parts of one mesh may have welded side by side: the first stays)
    slot = {geo, ids};
  return slot.ids;
}

bool sameRanges(const std::vector<mat::StrandMesh::Range>& a, const std::vector<mat::StrandMesh::Range>& b)
{
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](const mat::StrandMesh::Range& x, const mat::StrandMesh::Range& y) { return x.start == y.start && x.count == y.count; });
}

// The votes of `ranges` of `index`, with the UVs flipped as a build without cooked textures sends them.
std::shared_ptr<const std::vector<mat::StrandVote>> strandVotesOf(const std::shared_ptr<mesh::BuiltMesh>& geo,
                                                                  const std::vector<uint32_t>&           index,
                                                                  const std::vector<mat::StrandMesh::Range>& ranges,
                                                                  bool                                   flipV)
{
  {
    std::lock_guard lock(g_strandMutex);
    if(auto it = g_strands.find(geo.get()); it != g_strands.end())
      for(const StrandPart& p : it->second)
        if(p.geo.lock() == geo && p.flipV == flipV && p.indexCount == index.size() && sameRanges(p.ranges, ranges))
          return p.votes;
  }
  std::vector<float>     flipped;
  std::span<const float> uv = geo->uv;
  if(flipV)
  {
    flipped = geo->uv;
    for(size_t v = 1; v < flipped.size(); v += 2)
      flipped[v] = 1 - flipped[v];
    uv = flipped;
  }
  const auto weld  = weldOf(geo);
  auto       votes = [&] {
    timing::Scope t("strand votes", 20);
    return std::make_shared<const std::vector<mat::StrandVote>>(mat::strandVotes({geo->position, uv, index, ranges}, *weld));
  }();
  std::lock_guard lock(g_strandMutex);
  // geometry that is gone takes its parts with it (and an address used again starts afresh)
  std::erase_if(g_strands, [](const auto& kv) {
    return std::all_of(kv.second.begin(), kv.second.end(), [](const StrandPart& p) { return p.geo.expired(); });
  });
  auto& parts = g_strands[geo.get()];
  std::erase_if(parts, [&](const StrandPart& p) { return p.geo.lock() != geo; });
  parts.push_back({geo, ranges, index.size(), flipV, votes});
  return votes;
}

// The drawn ranges of each hair material of a mesh, by scene material.
std::map<int, std::vector<mat::StrandMesh::Range>> hairRanges(const SceneGraph& scene, const Mesh& m, const DrawnView& drawn)
{
  std::map<int, std::vector<mat::StrandMesh::Range>> out;
  for(const mesh::Group& group : drawn.groups())
  {
    const int mi = group.materialIndex < int(m.materials.size()) ? m.materials[group.materialIndex] : -1;
    if(mi >= 0 && mi < int(scene.materials.size()) && scene.materials[mi].surface.kind == mat::Kind::eHair)
      out[mi].push_back({group.start, group.count});
  }
  return out;
}

tinygltf::Value extras(const std::string& fvId)
{
  tinygltf::Value::Object o;
  o["fvId"] = tinygltf::Value(fvId);
  return tinygltf::Value(o);
}

}  // namespace

// Whether the figure's eyes have a tear film over them, a refracting moisture shell (mat::eyeShell):
// the eye surfaces and corneas under it are lowered without their own specular (LowerOptions::underTearFilm).
// A taken-off item's moisture is no film.
static bool hasTearFilm(const SceneGraph& scene)
{
  const std::vector<bool> live = dson::liveMaterials(scene);
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
    if(live[mi] && mat::eyeShell(scene.materials[mi].surface) == mat::EyeShell::eMoisture)
      return true;
  return false;
}

mat::LowerOptions requestOptions(const SceneGraph& scene, int material, mat::SkinMode skin, bool tearFilm, const std::string& toneReference)
{
  const Material&   m = scene.materials[size_t(material)];
  mat::LowerOptions o;
  o.skin          = skin;
  o.metresPerUv   = m.metresPerUv;
  o.underTearFilm = tearFilm;
  o.toneReference = toneReference;
  // an iris's colour asks for a texture only where its discs are known
  if(mat::eyeLayout(m.surface) == mat::EyeLayout::eIris)
    o.irisDiscs = irisDiscs(scene, material);
  return o;
}

std::vector<TextureRequest> textureRequests(const SceneGraph& scene, mat::SkinMode skin)
{
  std::vector<TextureRequest> out;
  std::set<std::string>       seen;
  const bool                  tearFilm = hasTearFilm(scene);
  const std::string           tone     = dson::toneReference(scene);
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
  {
    const Material& m = scene.materials[mi];
    for(TextureRequest& req : mat::lower(m.surface, m.edits, requestOptions(scene, int(mi), skin, tearFilm, tone)).requests)
      if(seen.insert(requestKey(req)).second)
        out.push_back(std::move(req));
  }
  return out;
}

std::vector<std::array<float, 3>> irisDiscs(const SceneGraph& scene, int material)
{
  struct Disc
  {
    double x = 0, y = 0, r = 0;
  };
  std::vector<Disc>       discs;
  const std::vector<bool> live = liveMeshes(scene);
  for(size_t mesh = 0; mesh < scene.meshes.size(); mesh++)
  {
    const Mesh& m = scene.meshes[mesh];
    if(!m.geo || !live[mesh])
      continue;
    const mesh::BuiltMesh& geo = *m.geo;
    // its triangles join their vertices into islands (union-find on drawn vertices, which are split
    // at UV seams)
    std::map<uint32_t, uint32_t> parent;
    const auto find = [&](uint32_t v) {
      while(parent[v] != v)
        v = parent[v] = parent[parent[v]];
      return v;
    };
    for(const mesh::Group& g : geo.groups)
    {
      if(g.materialIndex < 0 || g.materialIndex >= int(m.materials.size()) || m.materials[size_t(g.materialIndex)] != material)
        continue;
      for(uint32_t t = g.start; t + 2 < g.start + g.count && t + 2 < geo.fullIndex.size(); t += 3)
      {
        const uint32_t a = geo.fullIndex[t];
        for(uint32_t k = 0; k < 3; k++)
          parent.try_emplace(geo.fullIndex[t + k], geo.fullIndex[t + k]);
        for(uint32_t k = 1; k < 3; k++)
          parent[find(geo.fullIndex[t + k])] = find(a);
      }
    }
    std::map<uint32_t, std::vector<uint32_t>> islands;
    for(const auto& [v, p] : parent)
      if(size_t(v) * 2 + 1 < geo.uv.size())
        islands[find(v)].push_back(v);
    for(const auto& [root, verts] : islands)
    {
      // image space: x = u, y = 1 - v
      Disc d;
      for(uint32_t v : verts)
      {
        d.x += geo.uv[v * 2];
        d.y += 1.0 - geo.uv[v * 2 + 1];
      }
      d.x /= double(verts.size());
      d.y /= double(verts.size());
      for(uint32_t v : verts)
        d.r = std::max(d.r, std::hypot(geo.uv[v * 2] - d.x, 1.0 - geo.uv[v * 2 + 1] - d.y));
      // the tile it is in maps onto the image like the first
      d.x -= std::floor(d.x);
      d.y -= std::floor(d.y);
      discs.push_back(d);
    }
  }
  // islands that overlap are one eye (an iris cut into rings, or both eyes on one place in the map)
  for(bool merged = true; merged;)
  {
    merged = false;
    for(size_t i = 0; i < discs.size() && !merged; i++)
      for(size_t j = i + 1; j < discs.size() && !merged; j++)
      {
        const Disc&  a = discs[i];
        const Disc&  b = discs[j];
        const double d = std::hypot(a.x - b.x, a.y - b.y);
        if(d >= a.r + b.r)
          continue;
        // the smallest disc around both
        Disc c;
        if(d + b.r <= a.r)
          c = a;
        else if(d + a.r <= b.r)
          c = b;
        else
        {
          c.r          = (d + a.r + b.r) / 2;
          const double t = (c.r - a.r) / d;
          c.x          = a.x + (b.x - a.x) * t;
          c.y          = a.y + (b.y - a.y) * t;
        }
        discs[i] = c;
        discs.erase(discs.begin() + std::ptrdiff_t(j));
        merged = true;
      }
  }
  std::sort(discs.begin(), discs.end(), [](const Disc& a, const Disc& b) { return a.r > b.r; });
  std::vector<std::array<float, 3>> out;
  for(const Disc& d : discs)
    if(d.r > 1e-4 && out.size() < 4)
      out.push_back({float(d.x), float(d.y), float(std::min(d.r, 0.5))});
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

void measureStrands(const SceneGraph& scene, bool texturesFlipped)
{
  struct Part
  {
    const Mesh*                         mesh;
    std::shared_ptr<DrawnView>          drawn;
    std::vector<mat::StrandMesh::Range> ranges;
  };
  std::vector<Part> parts;
  for(const Mesh& m : scene.meshes)
  {
    if(!m.geo)
      continue;
    auto drawn = std::make_shared<DrawnView>(*m.geo);
    for(auto& [mi, ranges] : hairRanges(scene, m, *drawn))
      parts.push_back({&m, drawn, std::move(ranges)});
  }
  // each hair mesh welded once, side by side, then every part's votes
  std::vector<std::shared_ptr<mesh::BuiltMesh>> meshes;
  for(const Part& p : parts)
    if(std::find(meshes.begin(), meshes.end(), p.mesh->geo) == meshes.end())
      meshes.push_back(p.mesh->geo);
  jobs::parallelFor(meshes.size(), 1, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      timing::Scope t("strand weld", 20);
      weldOf(meshes[i]);
    }
  });
  jobs::parallelFor(parts.size(), 1, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
      strandVotesOf(parts[i].mesh->geo, parts[i].drawn->index(), parts[i].ranges, !texturesFlipped);
  });
}

// The figure's one sampler. Genesis UVs use one 0..1 tile per surface group (UDIM-style); repeat
// maps each tile home.
tinygltf::Sampler figureSampler()
{
  tinygltf::Sampler s;
  s.wrapS     = TINYGLTF_TEXTURE_WRAP_REPEAT;
  s.wrapT     = TINYGLTF_TEXTURE_WRAP_REPEAT;
  s.magFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
  s.minFilter = TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR;
  return s;
}

Drawing drawing(const SceneGraph& scene, int root)
{
  Drawing out;
  // nodes, depth first as traversal meets them
  {
    std::vector<int> stack{root};
    while(!stack.empty())
    {
      const int i = stack.back();
      stack.pop_back();
      if(scene.nodes[i].removed)
        continue;
      out.order.push_back(i);
      const auto& children = scene.nodes[i].children;
      for(auto it = children.rbegin(); it != children.rend(); ++it)
        stack.push_back(*it);
    }
  }
  timing::Scope phase("drawn triangles and bounds");
  // each mesh's drawn triangles (without what worn items hide) and position bounds, side by side
  out.drawn.resize(scene.meshes.size());
  out.bounds.resize(scene.meshes.size());
  jobs::parallelFor(scene.meshes.size(), 1, [&](size_t begin, size_t end) {
    for(size_t mi = begin; mi < end; mi++)
    {
      if(!scene.meshes[mi].geo)
        continue;
      const mesh::BuiltMesh& geo = *scene.meshes[mi].geo;
      out.drawn[mi]              = std::make_unique<DrawnView>(geo);
      std::array<double, 6>& b   = out.bounds[mi];
      b                          = {1e300, 1e300, 1e300, -1e300, -1e300, -1e300};
      for(size_t v = 0, count = geo.position.size() / 3; v < count; v++)
        for(int c = 0; c < 3; c++)
        {
          b[c]     = std::min(b[c], double(geo.position[v * 3 + c]));
          b[c + 3] = std::max(b[c + 3], double(geo.position[v * 3 + c]));
        }
    }
  });
  return out;
}

std::vector<mat::LowerOptions> lowering(const SceneGraph& scene, const Drawing& drawing, const BuildOptions& options)
{
  // what the materials need from the meshes: the area an emitter spreads its flux over, and the
  // direction of hair strands, which is measured from the cards' UVs as the glTF carries them
  timing::Scope           phase("emitters and strand votes");
  const std::vector<Mat4> worlds = scene.worlds();
  std::vector<double>     emitArea(scene.materials.size(), 0.0);
  std::vector<std::vector<std::shared_ptr<const std::vector<mat::StrandVote>>>> hairVotes(scene.materials.size());
  // a mesh drawn from several nodes (DAZ Studio instances) counts once: each copy emits as the one it copies
  std::vector<bool> counted(scene.meshes.size(), false);
  for(int i : drawing.order)
  {
    const Node& n = scene.nodes[i];
    if(n.kind != NodeKind::eMesh || n.mesh < 0 || !scene.meshes[n.mesh].geo || counted[n.mesh])
      continue;
    counted[n.mesh] = true;
    const Mesh&            m   = scene.meshes[n.mesh];
    const mesh::BuiltMesh& geo = *m.geo;
    const Mat4&            w4  = worlds[i];
    // the largest axis scale of the node's world matrix (three's getMaxScaleOnAxis)
    double s2 = 0;
    for(int c = 0; c < 3; c++)
      s2 = std::max(s2, w4.e[c * 4] * w4.e[c * 4] + w4.e[c * 4 + 1] * w4.e[c * 4 + 1] + w4.e[c * 4 + 2] * w4.e[c * 4 + 2]);
    const DrawnView& drawn = *drawing.drawn[n.mesh];
    for(const mesh::Group& group : drawn.groups())
    {
      const int mi = group.materialIndex < int(m.materials.size()) ? m.materials[group.materialIndex] : -1;
      if(mi < 0 || mi >= int(scene.materials.size()))
        continue;
      const mat::Surface& sf = scene.materials[mi].surface;
      if(mat::needsArea(sf))
      {
        double area = 0;
        const uint32_t* idx = drawn.index().data() + group.start;
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
    }
    for(const auto& [mi, ranges] : hairRanges(scene, m, drawn))
      hairVotes[mi].push_back(strandVotesOf(m.geo, drawn.index(), ranges, !options.texturesFlipped));
  }
  // strand hair ribbons face one way, so their surfaces are seen from both sides
  std::vector<bool>       onRibbon(scene.materials.size(), false);
  const std::vector<bool> liveMesh = liveMeshes(scene);
  for(size_t mesh = 0; mesh < scene.meshes.size(); mesh++)
    if(const Mesh& m = scene.meshes[mesh]; liveMesh[mesh] && m.geo && m.geo->ribbon)
      for(int mi : m.materials)
        if(mi >= 0 && mi < int(onRibbon.size()))
          onRibbon[mi] = true;
  std::vector<mat::LowerOptions> out(scene.materials.size());
  const bool        tearFilm = hasTearFilm(scene);
  const std::string tone     = dson::toneReference(scene);
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
  {
    mat::LowerOptions& lo = out[mi];
    lo                    = requestOptions(scene, int(mi), options.skin, tearFilm, tone);
    lo.forFile            = options.forFile;
    lo.flipV              = !options.texturesFlipped;
    lo.ribbon             = onRibbon[mi];
    lo.emission           = mat::emissionStrength(scene.materials[mi].surface, emitArea[mi], options.exposureValue, options.iso);
    // one material on several meshes is one glTF material, so those meshes vote together
    std::vector<mat::StrandVote> votes;
    for(const auto& v : hairVotes[mi])
      votes.insert(votes.end(), v->begin(), v->end());
    lo.anisotropyRotation = mat::strandRotation(votes);
  }
  return out;
}

mat::Material figureMaterial(const SceneGraph& scene, int material, const mat::LowerOptions& lowering,
                             const std::function<mat::ResolvedTexture(const TextureRequest&)>& resolve, const BuildOptions& options)
{
  const Material& m   = scene.materials[size_t(material)];
  mat::Material   out = mat::resolved(mat::lower(m.surface, m.edits, lowering), resolve);
  out.name            = m.name;
  out.id              = "m:" + std::to_string(material);
  out.walkThrough     = size_t(material) < options.walkThrough.size() && options.walkThrough[size_t(material)];
  return out;
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

  const Drawing drawn = drawing(scene, root);
  for(size_t k = 0; k < drawn.order.size(); k++)
    out.nodeOf[drawn.order[k]] = int(k);
  for(int i : drawn.order)
  {
    const Node&    n = scene.nodes[i];
    tinygltf::Node g;
    g.name        = n.name;
    g.translation = {n.position.x, n.position.y, n.position.z};
    g.rotation    = {n.quaternion.x, n.quaternion.y, n.quaternion.z, n.quaternion.w};
    g.scale       = {n.scale.x, n.scale.y, n.scale.z};
    g.extras      = extras("n:" + std::to_string(i));
    for(int c : n.children)
      if(out.nodeOf[c] >= 0)
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
  if(std::any_of(model.nodes.begin(), model.nodes.end(), [](const tinygltf::Node& g) { return g.extensions.count("KHR_node_visibility") > 0; })
     && std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), "KHR_node_visibility") == model.extensionsUsed.end())
    model.extensionsUsed.push_back("KHR_node_visibility");
  model.scenes.emplace_back();
  model.scenes[0].nodes = {out.nodeOf[root]};
  model.defaultScene    = 0;

  out.lowering = lowering(scene, drawn, options);

  // materials, and the textures they ask for
  std::optional<timing::Scope> phase;
  phase.emplace("materials");
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
      model.samplers.push_back(figureSampler());
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
    return textureOf[key] = {int(model.textures.size()) - 1, image->flipGreen, image->flipped != options.texturesFlipped};
  };
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
  {
    model.materials.push_back(mat::toGltf(figureMaterial(scene, int(mi), out.lowering[mi], texture, options), model.extensionsUsed));
  }
  // the live model keeps it even with no texture yet: a texture baked in later (FvMaterials::baked)
  // takes the figure's sampler, the first at its place in the composed model, not another prop's
  if(model.samplers.empty() && !options.forFile)
    model.samplers.push_back(figureSampler());
  if(usesBasisu)
  {
    model.extensionsUsed.push_back("KHR_texture_basisu");
    model.extensionsRequired.push_back("KHR_texture_basisu");
  }

  // meshes and skins
  phase.reset();
  phase.emplace("meshes and skins");
  for(int i : drawn.order)
  {
    const Node& n = scene.nodes[i];
    if(n.kind != NodeKind::eMesh || n.mesh < 0)
      continue;
    // drawn from another node already (a DAZ Studio instance, never skinned): the same glTF mesh
    if(out.meshOf[n.mesh] >= 0)
    {
      model.nodes[out.nodeOf[i]].mesh = out.meshOf[n.mesh];
      continue;
    }
    const Mesh&            m   = scene.meshes[n.mesh];
    const mesh::BuiltMesh& geo = *m.geo;
    const size_t           count = geo.position.size() / 3;
    const DrawnView&       view  = *drawn.drawn[n.mesh];
    if(!count || view.groups().empty())
      continue;

    const std::array<double, 6>& b = drawn.bounds[n.mesh];
    const std::vector<double>    min{b[0], b[1], b[2]}, max{b[3], b[4], b[5]};
    const int position = w.accessor(w.view(geo.position.data(), geo.position.size() * 4, TINYGLTF_TARGET_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_FLOAT,
                                     count, TINYGLTF_TYPE_VEC3, min, max);
    const int normal = w.accessor(w.view(geo.normal.data(), geo.normal.size() * 4, TINYGLTF_TARGET_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_FLOAT,
                                  count, TINYGLTF_TYPE_VEC3);
    const float* uv = geo.uv.data();
    if(!options.texturesFlipped)
    {
      std::vector<float> flipped = geo.uv;
      for(size_t v = 0; v < count; v++)
        flipped[v * 2 + 1] = 1 - flipped[v * 2 + 1];
      uv = w.keep(std::move(flipped));
    }
    const int texcoord = w.accessor(w.view(uv, geo.uv.size() * 4, TINYGLTF_TARGET_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_FLOAT, count, TINYGLTF_TYPE_VEC2);
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
    for(const mesh::Group& group : view.groups())
    {
      const uint32_t* first = view.index().data() + group.start;
      tinygltf::Primitive p;
      if(count <= 65535)
      {
        const uint16_t* idx = w.keep(std::vector<uint16_t>(first, first + group.count));
        p.indices = w.accessor(w.view(idx, size_t(group.count) * 2, TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER), TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT,
                               group.count, TINYGLTF_TYPE_SCALAR);
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
      std::vector<float> ibm  = inverseBindMatrices(m);
      const size_t       size = ibm.size() * 4;
      skin.inverseBindMatrices = w.accessor(w.view(w.keep(std::move(ibm)), size, 0), TINYGLTF_COMPONENT_TYPE_FLOAT, m.bones.size(), TINYGLTF_TYPE_MAT4);
      model.skins.push_back(std::move(skin));
      model.nodes[out.nodeOf[i]].skin = int(model.skins.size()) - 1;
      out.skinOf[n.mesh]              = int(model.skins.size()) - 1;
    }
  }
  phase.reset();
  phase.emplace("buffer");
  w.flush();
  model.buffers[0].name = "figure";
  out.textureOf         = std::move(textureOf);
  return out;
}

}  // namespace fv::build
