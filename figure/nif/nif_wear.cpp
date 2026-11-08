#include <figure/nif/nif_wear.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

#include <figure/dson/channels.hpp>
#include <figure/dson/materials.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/wardrobe/fit/bases.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace fv::nif {

namespace {

namespace fs = std::filesystem;
using D3     = std::array<double, 3>;

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

std::string forward(std::string s)
{
  std::replace(s.begin(), s.end(), '\\', '/');
  return s;
}

std::string utf8(const fs::path& p)
{
  const std::u8string s = p.generic_u8string();
  return std::string(s.begin(), s.end());
}

fs::path fromUtf8(const std::string& s)
{
  return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

uint64_t fnv1a(std::string_view s)
{
  uint64_t h = 1469598103934665603ULL;
  for(unsigned char c : s)
  {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

// A library path as DSON writes one: a leading slash, and what a URL can't carry plainly percent-encoded.
std::string encodeRel(const std::string& rel)
{
  static const char* hex = "0123456789ABCDEF";
  std::string        out = "/";
  for(unsigned char c : rel)
    if(c <= 0x20 || c == '%' || c == '#' || c == '?' || c == '|' || c >= 0x7F)
      out += {'%', hex[c >> 4], hex[c & 15]};
    else
      out += char(c);
  return out;
}

D3 scaled(const D3& p, double s)
{
  return {p[0] * s, p[1] * s, p[2] * s};
}

// ------------------------------------------------------------------ skeleton

struct Bone
{
  std::string name;
  int         parent = -1;  // into Skeleton::bones; -1 hangs from the item's root
  D3          at{};         // DAZ axes and centimetres, before the fit's scale
  int         node = -1;    // into File::nodes
};

struct Skeleton
{
  std::vector<Bone>                    bones;  // parents before children
  std::unordered_map<std::string, int> byName;
};

// The bones the shapes skin to, with every node above them up to the file's root (left out): their
// places from the skins' bind data where a skin has them, else from the node tree.
Skeleton skeletonOf(const File& f, const std::vector<size_t>& shapes)
{
  std::map<int, D3> bound;  // node -> its bind place, Skyrim's space
  std::set<int>     wanted;
  for(size_t s : shapes)
  {
    const Shape& sh = f.shapes[s];
    for(size_t i = 0; i < sh.bones.size(); i++)
    {
      const int node = sh.bones[i];
      if(i < sh.skinToBone.size() && !bound.count(node))
      {
        const V3 c  = sh.skinToBone[i].inverse().t;
        bound[node] = {c[0], c[1], c[2]};
      }
      for(int n = node, guard = 0; n >= 0 && guard < 512; n = f.nodes[size_t(n)].parent, guard++)
        if(f.nodes[size_t(n)].parent >= 0)  // the root itself is the item's
          wanted.insert(n);
    }
    // an unskinned shape rides on the node it hangs from
    if(!sh.skinned())
      for(int n = sh.parent, guard = 0; n >= 0 && guard < 512; n = f.nodes[size_t(n)].parent, guard++)
        if(f.nodes[size_t(n)].parent >= 0)
          wanted.insert(n);
  }
  Skeleton out;
  // depth first from the root's children, so a parent comes before its children
  std::map<int, std::vector<int>> children;
  for(int n : wanted)
    children[f.nodes[size_t(n)].parent].push_back(n);
  std::function<void(int, int)> add = [&](int node, int parentBone) {
    const auto it = children.find(node);
    if(it == children.end())
      return;
    for(int c : it->second)
    {
      Bone b;
      b.name   = f.nodes[size_t(c)].name;
      b.parent = parentBone;
      b.node   = c;
      V3 p;
      if(const auto bp = bound.find(c); bp != bound.end())
        p = {float(bp->second[0]), float(bp->second[1]), float(bp->second[2])};
      else
        p = f.world(c).t;
      b.at = toDaz(p);
      // a name met twice (two nodes alike) keeps one bone; the second is told apart
      std::string name = b.name;
      for(int k = 2; out.byName.count(name); k++)
        name = b.name + " #" + std::to_string(k);
      b.name = name;
      out.byName[name] = int(out.bones.size());
      out.bones.push_back(b);
      add(c, int(out.bones.size()) - 1);
    }
  };
  for(const auto& [parent, kids] : children)
    if(parent >= 0 && !wanted.count(parent))
      add(parent, -1);
  return out;
}

// The bone a node became, by its node.
int boneOfNode(const Skeleton& s, int node)
{
  for(size_t i = 0; i < s.bones.size(); i++)
    if(s.bones[i].node == node)
      return int(i);
  return -1;
}

std::vector<wardrobe::SkelBone> skelBones(const Skeleton& s, double scale)
{
  std::vector<wardrobe::SkelBone> out;
  for(const Bone& b : s.bones)
  {
    wardrobe::SkelBone k;
    k.name = b.name;
    if(b.parent >= 0)
      k.parent = s.bones[size_t(b.parent)].name;
    k.center = scaled(b.at, scale);
    out.push_back(std::move(k));
  }
  return out;
}

// Head to feet, cm: what the whole is scaled by.
std::optional<double> standingHeight(const std::vector<wardrobe::SkelBone>& bones)
{
  const wardrobe::SkelBone* head = wardrobe::findSlot(bones, 0, "head");
  const wardrobe::SkelBone* l    = wardrobe::findSlot(bones, 'l', "foot");
  const wardrobe::SkelBone* r    = wardrobe::findSlot(bones, 'r', "foot");
  if(!head || !l || !r)
    return std::nullopt;
  const double h = head->center[1] - (l->center[1] + r->center[1]) / 2;
  return h > 1 ? std::optional<double>(h) : std::nullopt;
}

// ------------------------------------------------------------------ meshes

// Vertices at one place made one: Skyrim splits them along UV seams and hard edges, a DAZ cage
// doesn't (its normals, the fit's smoothing and a drape all run over the welded mesh).
struct Weld
{
  std::vector<int> cageOf;  // shape vertex -> cage vertex
  std::vector<int> first;   // cage vertex -> the first shape vertex at it
};

Weld weld(const Shape& s)
{
  Weld                                  w;
  std::unordered_map<std::string, int> at;
  w.cageOf.resize(s.positions.size());
  for(size_t v = 0; v < s.positions.size(); v++)
  {
    std::string key(reinterpret_cast<const char*>(s.positions[v].data()), sizeof(V3));
    const auto [it, added] = at.emplace(std::move(key), int(w.first.size()));
    if(added)
      w.first.push_back(int(v));
    w.cageOf[v] = it->second;
  }
  return w;
}

struct Built
{
  mesh::MeshInput input;
  mesh::SkinInput skin;
  // per skin bone, the skeleton's bone
  std::vector<int> bones;
};

// The shape as a cage: welded positions in DAZ's space (times `scale`), its triangles, its UVs with
// the seams as overrides, and the skin of each cage vertex (its first shape vertex's).
Built cageOf(const File& f, const Shape& s, const Skeleton& sk, double scale)
{
  Built       out;
  const Weld  w   = weld(s);
  const auto  raw = [&](size_t v) {
    if(s.skinned())
      return s.positions[v];
    return (f.world(s.parent) * s.local).apply(s.positions[v]);
  };
  mesh::MeshInput& in = out.input;
  in.positions.reserve(w.first.size() * 3);
  for(int v : w.first)
  {
    const D3 p = scaled(toDaz(raw(size_t(v))), scale);
    in.positions.insert(in.positions.end(), {float(p[0]), float(p[1]), float(p[2])});
  }
  const bool hasUv = s.uvs.size() == s.positions.size();
  // a shape vertex whose UV isn't its cage vertex's gets a UV of its own after the cage's
  std::vector<int> uvOf(s.positions.size(), -1);
  if(hasUv)
  {
    for(int v : w.first)
      in.uvs.insert(in.uvs.end(), {s.uvs[size_t(v)][0], 1 - s.uvs[size_t(v)][1]});
    for(size_t v = 0; v < s.positions.size(); v++)
    {
      const int c = w.cageOf[v];
      if(s.uvs[v] == s.uvs[size_t(w.first[size_t(c)])])
        uvOf[v] = c;
      else
      {
        uvOf[v] = int(in.uvs.size() / 2);
        in.uvs.insert(in.uvs.end(), {s.uvs[v][0], 1 - s.uvs[v][1]});
      }
    }
  }
  in.polyStart.push_back(0);
  for(size_t t = 0; t + 2 < s.triangles.size(); t += 3)
  {
    const uint32_t a = s.triangles[t], b = s.triangles[t + 1], c = s.triangles[t + 2];
    const int      ca = w.cageOf[a], cb = w.cageOf[b], cc = w.cageOf[c];
    if(ca == cb || cb == cc || ca == cc)
      continue;
    const int poly = int(in.polyStart.size()) - 1;
    for(const auto& [shapeV, cageV] : {std::pair{a, ca}, std::pair{b, cb}, std::pair{c, cc}})
    {
      in.polyVerts.push_back(cageV);
      if(hasUv && uvOf[shapeV] != cageV)
        in.overrides.insert(in.overrides.end(), {poly, cageV, uvOf[shapeV]});
    }
    in.polyStart.push_back(int32_t(in.polyVerts.size()));
    in.polyMaterial.push_back(0);
  }
  in.materialCount = 1;

  // the skin: the shape's bones as the skeleton's, each cage vertex weighted as its first shape vertex
  mesh::SkinInput& skin = out.skin;
  skin.start.push_back(0);
  if(s.skinned())
  {
    for(int node : s.bones)
      out.bones.push_back(boneOfNode(sk, node));
    for(int v : w.first)
    {
      double sum = 0;
      for(uint32_t i = s.weightStart[size_t(v)]; i < s.weightStart[size_t(v) + 1]; i++)
        sum += s.weight[i];
      for(uint32_t i = s.weightStart[size_t(v)]; i < s.weightStart[size_t(v) + 1]; i++)
        if(sum > 0 && s.weight[i] / sum > 1e-4 && out.bones[s.weightBone[i]] >= 0)
        {
          skin.bone.push_back(s.weightBone[i]);
          skin.weight.push_back(float(s.weight[i] / sum));
        }
      skin.start.push_back(int32_t(skin.bone.size()));
    }
  }
  else
  {
    // wholly on the bone it hangs from
    out.bones.push_back(boneOfNode(sk, s.parent));
    for(size_t c = 0; c < w.first.size(); c++)
    {
      skin.bone.push_back(0);
      skin.weight.push_back(1);
      skin.start.push_back(int32_t(skin.bone.size()));
    }
  }
  return out;
}

// ------------------------------------------------------------------ materials

std::shared_ptr<dson::Channel> channel(const char* label, dson::Json value, std::optional<std::string> image = std::nullopt, dson::Json mod = nullptr)
{
  auto c               = std::make_shared<dson::Channel>();
  c->value             = std::move(value);
  c->image             = std::move(image);
  c->id                = dson::norm(label);
  c->keys              = {c->id};
  c->imageModification = std::move(mod);
  return c;
}

// `file`'s path under `folder`, forward slashes, when it is inside it.
std::optional<std::string> insideFolder(const fs::path& folder, const fs::path& file)
{
  if(folder.empty())
    return std::nullopt;
  std::error_code ec;
  const fs::path  base = fs::weakly_canonical(folder, ec);
  const fs::path  full = fs::weakly_canonical(file, ec);
  if(ec)
    return std::nullopt;
  const fs::path rel = full.lexically_relative(base);
  if(rel.empty() || *rel.begin() == ".." || rel.is_absolute())
    return std::nullopt;
  return utf8(rel);
}

struct TextureFinder
{
  fs::path                          meshDir, home;
  std::map<std::string, std::string> done;  // a texture's name in the file -> its library path ("" not found)
  std::vector<std::string>          missing, problems;

  std::string operator()(const std::string& rel)
  {
    if(rel.empty())
      return {};
    const std::string key = lower(forward(rel));
    if(const auto it = done.find(key); it != done.end())
      return it->second;
    std::string& out = done[key];
    fs::path     src = findTexture(meshDir, rel);
    // not in its own mod: in another installed beside it, as the game's one Data folder would have it
    if(src.empty())
      src = findInstalledTexture(home, rel);
    if(src.empty())
    {
      missing.push_back(forward(rel));
      return out;
    }
    // an installed mod's, in the content folder already: its own library path, nothing linked
    if(const std::optional<std::string> inside = insideFolder(home, src))
      return out = encodeRel(*inside);
    // the Data folder: where the texture's own path starts
    fs::path root = src.parent_path();
    for(size_t depth = std::count(key.begin(), key.end(), '/'); depth > 0 && root.has_parent_path(); depth--)
      root = root.parent_path();
    if(key.rfind("textures/", 0) != 0)
      root = root.parent_path();
    std::string problem;
    out = libraryTexture(home, root, src, key.rfind("textures/", 0) == 0 ? key : "textures/" + key, &problem);
    if(out.empty() && !problem.empty())
      problems.push_back(problem);
    return out;
  }
};

// Skyrim's lighting shader as Iray Uber's channels: the diffuse (its alpha the cutout where the shape
// tests or blends alpha), the normal map, a roughness from the Blinn-Phong glossiness, the glossy
// layer's weight from the specular strength, and the environment mask as the metal map (what a cube map
// reflects in the game is what is metal: buckles, studs, a retexture's "metallic" map).
mat::Surface surfaceOf(const std::string& name, const Shape& s, TextureFinder& find, dson::Channels& ch)
{
  const Shader& sh = *s.shader;
  ch.setShader({"studio/material/uber_iray", ""});
  const std::string diffuse = find(sh.texture(0));
  ch.set(dson::norm("Diffuse Color"), channel("Diffuse Color", dson::Json::array({1, 1, 1}), diffuse.empty() ? std::nullopt : std::optional(diffuse)));
  ch.set(dson::norm("Base Mixing"), channel("Base Mixing", 0));
  const bool cutout = s.alpha && (s.alpha->tests() || s.alpha->blends()) && !diffuse.empty();
  ch.set(dson::norm("Cutout Opacity"), cutout ? channel("Cutout Opacity", std::clamp(double(sh.alpha), 0.0, 1.0), diffuse, dson::Json{{"grayscale_mode", "alpha"}})
                                              : channel("Cutout Opacity", std::clamp(double(sh.alpha), 0.0, 1.0)));
  if(const std::string normal = sh.modelSpaceNormals() ? std::string() : find(sh.texture(1)); !normal.empty())
    ch.set(dson::norm("Normal Map"), channel("Normal Map", 1, normal, dson::Json{{"fv_directx_normal", true}}));
  // Blinn-Phong's exponent n is a Beckmann lobe of alpha sqrt(2 / (n + 2)); Iray squares its roughness
  const double alpha = std::sqrt(2 / (std::max(0.f, sh.glossiness) + 2.0));
  ch.set(dson::norm("Glossy Roughness"), channel("Glossy Roughness", std::clamp(std::sqrt(alpha), 0.1, 0.95)));
  ch.set(dson::norm("Glossy Layered Weight"), channel("Glossy Layered Weight", sh.specular() ? std::clamp(double(sh.specularStrength), 0.2, 1.0) : 0.2));
  ch.set(dson::norm("Glossy Reflectivity"), channel("Glossy Reflectivity", 0.5));
  if(sh.type == 1)
    if(const std::string mask = find(sh.texture(5)); !mask.empty())
      ch.set(dson::norm("Metallic Weight"), channel("Metallic Weight", std::clamp(double(sh.envMapScale), 0.0, 1.0), mask));
  if(!ch.get(dson::norm("Metallic Weight")))
    ch.set(dson::norm("Metallic Weight"), channel("Metallic Weight", 0));
  // Skyrim tiles in its own UVs, v down: in DAZ's, v up, the offset is what lands the same texel
  if(sh.uvScale != std::array<float, 2>{1, 1} || sh.uvOffset != std::array<float, 2>{0, 0})
  {
    ch.set(dson::norm("Horizontal Tiles"), channel("Horizontal Tiles", sh.uvScale[0]));
    ch.set(dson::norm("Vertical Tiles"), channel("Vertical Tiles", sh.uvScale[1]));
    ch.set(dson::norm("Horizontal Offset"), channel("Horizontal Offset", sh.uvOffset[0]));
    ch.set(dson::norm("Vertical Offset"), channel("Vertical Offset", 1 - sh.uvScale[1] - sh.uvOffset[1]));
  }
  return dson::parseSurface(name, ch, mat::Context::eItem);
}

// "_Fuse00_Helga_Dress" -> "Helga Dress"
std::string prettyName(std::string name)
{
  std::replace(name.begin(), name.end(), '_', ' ');
  const size_t first = name.find_first_not_of(' ');
  name               = first == std::string::npos ? std::string() : name.substr(first);
  // a modder's tag in front ("Fuse00 Helga Dress"): its first word when it ends in digits
  if(const size_t space = name.find(' '); space != std::string::npos && space > 0 && std::isdigit(static_cast<unsigned char>(name[space - 1])))
    name = name.substr(space + 1);
  return name.empty() ? std::string("Surface") : name;
}

// ------------------------------------------------------------------ the body it was built on

struct Base
{
  File                 file;
  size_t               shape = 0;
  Skeleton             skeleton;
};

// The body in this file, or in a sibling file of the same outfit (boots and a choker leave theirs to the
// dress): one with the same weight suffix (_0, _1) first.
std::optional<Base> findBase(const File& nif, const fs::path& file, std::vector<std::string>& notes)
{
  if(const auto bodies = bodyShapes(nif); !bodies.empty())
    return Base{nif, bodies[0], skeletonOf(nif, {bodies[0]})};
  const std::string stem   = lower(utf8(file.stem()));
  const std::string suffix = stem.size() > 2 && stem[stem.size() - 2] == '_' ? stem.substr(stem.size() - 2) : std::string();
  std::vector<fs::path> siblings;
  std::error_code       ec;
  for(fs::directory_iterator it(file.parent_path(), ec), end; !ec && it != end; it.increment(ec))
    if(it->is_regular_file(ec) && lower(utf8(it->path().extension())) == ".nif" && it->path() != file)
      siblings.push_back(it->path());
  std::stable_sort(siblings.begin(), siblings.end(), [&](const fs::path& a, const fs::path& b) {
    const auto rank = [&](const fs::path& p) {
      const std::string s = lower(utf8(p.stem()));
      return suffix.empty() || (s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0) ? 0 : 1;
    };
    return rank(a) < rank(b);
  });
  for(size_t i = 0; i < siblings.size() && i < 16; i++)
    try
    {
      File sib = readFile(siblings[i]);
      if(const auto bodies = bodyShapes(sib); !bodies.empty())
      {
        notes.push_back("measured against the body in " + utf8(siblings[i].filename()));
        Skeleton sk = skeletonOf(sib, {bodies[0]});
        return Base{std::move(sib), bodies[0], std::move(sk)};
      }
    }
    catch(const std::exception&)
    {
      // a sibling that doesn't read is no base
    }
  return std::nullopt;
}

std::shared_ptr<const wardrobe::BaseCage> baseCageOf(const Base& b, double scale)
{
  const Shape& s     = b.file.shapes[b.shape];
  Built        built = cageOf(b.file, s, b.skeleton, scale);
  auto         cage  = std::make_shared<wardrobe::BaseCage>();
  cage->positions    = std::move(built.input.positions);
  for(size_t p = 0; p + 1 < built.input.polyStart.size(); p++)
    for(int32_t k = built.input.polyStart[p]; k < built.input.polyStart[p + 1]; k++)
      cage->tris.push_back(uint32_t(built.input.polyVerts[size_t(k)]));
  // the skin's bones as the skeleton's
  cage->skin = built.skin;
  for(int32_t& bone : cage->skin.bone)
    bone = built.bones[size_t(bone)];
  cage->skeleton = skelBones(b.skeleton, scale);
  return cage;
}

}  // namespace

std::array<double, 3> toDaz(const V3& p)
{
  return {-double(p[0]) * kUnitCm, double(p[2]) * kUnitCm, double(p[1]) * kUnitCm};
}

bool isNif(const std::string& path)
{
  return path.size() > 4 && lower(path.substr(path.size() - 4)) == ".nif";
}

std::vector<size_t> bodyShapes(const File& nif)
{
  std::vector<size_t> out;
  for(size_t i = 0; i < nif.shapes.size(); i++)
  {
    const Shape& s    = nif.shapes[i];
    const bool   tint = s.shader && s.shader->type == 5;
    const bool   tri  = std::any_of(s.extra.begin(), s.extra.end(), [](const auto& e) { return e.first == "BODYTRI"; });
    if((tint || tri) && s.skinned() && !s.positions.empty())
      out.push_back(i);
  }
  std::stable_sort(out.begin(), out.end(), [&](size_t a, size_t b) { return nif.shapes[a].positions.size() > nif.shapes[b].positions.size(); });
  return out;
}

fs::path findTexture(const fs::path& meshDir, const std::string& rel)
{
  const std::string r = forward(rel);
  std::vector<fs::path> tails{fromUtf8(r)};
  if(lower(r).rfind("textures/", 0) != 0)
    tails.push_back(fs::path("textures") / fromUtf8(r));
  std::error_code ec;
  for(fs::path dir = meshDir; !dir.empty(); dir = dir.parent_path())
  {
    for(const fs::path& tail : tails)
      if(const fs::path p = dir / tail; fs::is_regular_file(p, ec))
        return p;
    if(dir == dir.parent_path())
      break;
  }
  return {};
}

fs::path findInstalledTexture(const fs::path& home, const std::string& rel)
{
  if(home.empty())
    return {};
  std::error_code ec;
  const fs::path  mods = home / kInstalledMods;
  if(!fs::is_directory(mods, ec))
    return {};
  // in name order, so the same texture comes from the same mod every time
  std::vector<fs::path> dirs;
  for(const auto& e : fs::directory_iterator(mods, ec))
    if(e.is_directory(ec))
      dirs.push_back(e.path());
  std::sort(dirs.begin(), dirs.end());
  const std::string r    = forward(rel);
  const fs::path    tail = lower(r).rfind("textures/", 0) == 0 ? fromUtf8(r) : fs::path("textures") / fromUtf8(r);
  for(const fs::path& d : dirs)
    if(const fs::path p = d / tail; fs::is_regular_file(p, ec))
      return p;
  return {};
}

std::string libraryTexture(const fs::path& home, const fs::path& dataRoot, const fs::path& source, const std::string& rel, std::string* problem)
{
  if(home.empty())
    return {};
  std::error_code ec;
  fs::path        root = fs::weakly_canonical(dataRoot, ec);
  if(ec)
    root = dataRoot;
  char hash[17];
  std::snprintf(hash, sizeof(hash), "%016llx", static_cast<unsigned long long>(fnv1a(lower(utf8(root)))));
  const std::string relPath = "Runtime/Textures/FigureViewer/Skyrim/" + std::string(hash) + "/" + lower(forward(rel));
  const fs::path    dest    = home / fromUtf8(relPath);
  const auto        size    = fs::file_size(source, ec);
  if(ec)
  {
    if(problem)
      *problem = utf8(source) + ": " + ec.message();
    return {};
  }
  // there already: the same file (a link), or a copy of it that is no older
  if(fs::is_regular_file(dest, ec) && fs::file_size(dest, ec) == size && !ec)
  {
    if(fs::equivalent(dest, source, ec) || fs::last_write_time(dest, ec) >= fs::last_write_time(source, ec))
      return encodeRel(relPath);
  }
  fs::create_directories(dest.parent_path(), ec);
  fs::remove(dest, ec);
  ec.clear();
  fs::create_hard_link(source, dest, ec);
  if(ec)
  {
    ec.clear();
    fs::copy_file(source, dest, fs::copy_options::overwrite_existing, ec);
  }
  if(ec)
  {
    if(problem)
      *problem = utf8(dest) + ": " + ec.message();
    return {};
  }
  return encodeRel(relPath);
}

dson::LoadResult loadWearable(const fs::path& file, const dson::LoadOptions& options, const WearOptions& wear)
{
  const std::string fileName = utf8(file.filename());
  File              nif;
  try
  {
    nif = readFile(file);
  }
  catch(const std::exception& e)
  {
    throw std::runtime_error(fileName + " can't be worn: " + e.what());
  }
  const std::vector<size_t> bodies = bodyShapes(nif);
  std::vector<size_t>       drawn;
  for(size_t i = 0; i < nif.shapes.size(); i++)
  {
    const Shape& s = nif.shapes[i];
    // no material: a physics proxy (HDT-SMP's collision shapes), never drawn in the game either
    if(!s.shader || s.hidden() || s.positions.empty() || s.triangles.empty())
      continue;
    if(std::find(bodies.begin(), bodies.end(), i) != bodies.end())
      continue;
    drawn.push_back(i);
  }
  if(drawn.empty())
    throw std::runtime_error(fileName + " has nothing to wear in it" + (bodies.empty() ? std::string() : " but the body it was made for"));

  dson::LoadResult out;
  std::vector<std::string> notes;
  const Skeleton           sk   = skeletonOf(nif, drawn);
  const std::optional<Base> base = findBase(nif, file, notes);

  // the figure's skeleton, as the fit reads it: the centres its files give
  std::vector<wardrobe::SkelBone> figure;
  std::vector<std::string>        figureNames;
  std::map<std::string, D3>       figureCenter;  // where its joints are now, which the item's bones take
  if(options.attach && options.attach->scene && options.attach->rig)
  {
    const SceneGraph& fs_ = *options.attach->scene;
    for(const auto& [id, node] : options.attach->rig->bones)
    {
      const fv::Node& n = fs_.nodes[size_t(node)];
      if(!n.joint)
        continue;
      wardrobe::SkelBone b;
      b.name = n.name;
      if(n.parent >= 0 && fs_.nodes[size_t(n.parent)].kind == NodeKind::eBone)
        b.parent = fs_.nodes[size_t(n.parent)].name;
      const auto& c = n.joint->centerAuthored != std::array<double, 3>{} ? n.joint->centerAuthored : n.joint->center;
      b.center      = {c[0], c[1], c[2]};
      figure.push_back(b);
      figureNames.push_back(n.name);
      figureCenter[n.name] = {n.joint->center[0], n.joint->center[1], n.joint->center[2]};
    }
  }
  // one scale for the whole, so the skeleton stands as tall as the figure's: girth with it
  double scale = 1;
  {
    std::optional<double> from = standingHeight(skelBones(sk, 1));
    if(!from && base)
      from = standingHeight(skelBones(base->skeleton, 1));
    const std::optional<double> to = standingHeight(figure);
    if(from && to)
      scale = std::clamp(*to / *from, 0.5, 2.0);
  }
  const std::vector<wardrobe::SkelBone> itemBones = skelBones(sk, scale);
  const std::map<std::string, wardrobe::Rigid> plan = figure.empty() ? std::map<std::string, wardrobe::Rigid>{} : wardrobe::rigidPlan(itemBones, figure);

  SceneGraph&       scene = out.scene;
  const std::string stem  = utf8(file.stem());
  std::string       label = stem;
  std::replace(label.begin(), label.end(), '_', ' ');
  out.name = stem;
  {
    fv::Node root;
    root.name = stem;
    out.root  = scene.add(std::move(root));
  }
  fv::Node groupNode;
  groupNode.name     = label;
  groupNode.dsonNode = stem;
  const int group    = scene.add(std::move(groupNode), out.root);

  // the skeleton: a root for the file, then its bones at rest unturned, each at its place on the figure
  // (a bone the figure has one for at that joint, the rest carried there with the bone above them), the
  // place in the file kept as the authored centre the fit reads
  dson::Rig rig;
  {
    fv::Node rootBone;
    rootBone.name     = stem;
    rootBone.kind     = NodeKind::eBone;
    rootBone.joint    = std::make_shared<JointData>();
    rootBone.joint->id    = stem;
    rootBone.joint->label = label;
    rig.root              = scene.add(std::move(rootBone), group);
    rig.bones.emplace_back(stem, rig.root);
  }
  // A follower copies its leader's turn relative to its parent, so each bone hangs from the bone that
  // follows its leader's parent: Skyrim's spine hangs from the pelvis, Genesis' from the hip, and as
  // in the file a pose's pelvis would turn the bodice twice. A bone leading to the joint its parent
  // leads to (a twist helper) stays under it, and a bone with no leader where the file has it.
  const std::vector<std::optional<std::string>> leaders = [&] {
    std::vector<std::optional<std::string>> l(sk.bones.size());
    if(!figure.empty())
      for(size_t i = 0; i < sk.bones.size(); i++)
        l[i] = wardrobe::matchBone(sk.bones[i].name, figureNames);
    return l;
  }();
  std::vector<int> parentOf(sk.bones.size());
  for(size_t i = 0; i < sk.bones.size(); i++)
    parentOf[i] = sk.bones[i].parent;
  if(!figure.empty())
  {
    std::map<std::string, std::optional<std::string>> figureParent;
    for(const wardrobe::SkelBone& b : figure)
      figureParent[b.name] = b.parent;
    std::map<std::string, int> follows;  // figure bone -> the first item bone following it
    for(size_t i = 0; i < sk.bones.size(); i++)
      if(leaders[i])
        follows.emplace(*leaders[i], int(i));
    const auto descends = [&](int bone, int from) {
      for(int p = bone, guard = 0; p >= 0 && guard < 512; p = parentOf[size_t(p)], guard++)
        if(p == from)
          return true;
      return false;
    };
    for(size_t i = 0; i < sk.bones.size(); i++)
    {
      if(!leaders[i])
        continue;
      const int was = sk.bones[i].parent;
      if(was >= 0 && leaders[size_t(was)] == leaders[i])
        continue;
      // the figure's nearest bone above the leader that an item bone follows; none: the item's root
      int want = -1;
      for(auto p = figureParent[*leaders[i]]; p; p = figureParent[*p])
        if(const auto it = follows.find(*p); it != follows.end() && it->second != int(i))
        {
          want = it->second;
          break;
        }
      // the item bone it hangs from now that follows anything; moved only when that follows another
      int now = was;
      while(now >= 0 && !leaders[size_t(now)])
        now = sk.bones[size_t(now)].parent;
      const auto leaderOf = [&](int b) { return b >= 0 ? leaders[size_t(b)] : std::optional<std::string>(); };
      if(leaderOf(now) != leaderOf(want) && (want < 0 || !descends(want, int(i))))
        parentOf[i] = want;
    }
  }
  // parents before children
  std::vector<size_t> order;
  {
    std::vector<std::vector<size_t>> kids(sk.bones.size() + 1);
    for(size_t i = 0; i < sk.bones.size(); i++)
      kids[parentOf[i] >= 0 ? size_t(parentOf[i]) : sk.bones.size()].push_back(i);
    std::vector<size_t> stack(kids.back().rbegin(), kids.back().rend());
    while(!stack.empty())
    {
      const size_t i = stack.back();
      stack.pop_back();
      order.push_back(i);
      stack.insert(stack.end(), kids[i].rbegin(), kids[i].rend());
    }
  }
  std::vector<int> boneNode(sk.bones.size(), -1);
  std::vector<D3>  boneAt(sk.bones.size());
  for(size_t i : order)
  {
    const Bone& b        = sk.bones[i];
    const D3    authored = scaled(b.at, scale);
    D3          at       = authored;
    if(leaders[i] && figureCenter.count(*leaders[i]))
      at = figureCenter[*leaders[i]];
    else if(const auto it = plan.find(b.name); it != plan.end())
    {
      const wardrobe::Rigid& g = it->second;
      const D3 d{authored[0] - g.from[0], authored[1] - g.from[1], authored[2] - g.from[2]};
      at = {g.r[0] * d[0] + g.r[1] * d[1] + g.r[2] * d[2] + g.to[0], g.r[3] * d[0] + g.r[4] * d[1] + g.r[5] * d[2] + g.to[1],
            g.r[6] * d[0] + g.r[7] * d[1] + g.r[8] * d[2] + g.to[2]};
    }
    boneAt[i]                    = at;
    auto joint                   = std::make_shared<JointData>();
    joint->id                    = b.name;
    joint->label                 = b.name;
    joint->center                = at;
    joint->centerAuthored        = authored;
    joint->endPoint              = at;
    const int parent             = parentOf[i];
    const D3  parentAt           = parent >= 0 ? boneAt[size_t(parent)] : D3{0, 0, 0};
    fv::Node bone;
    bone.name     = b.name;
    bone.kind     = NodeKind::eBone;
    bone.joint    = joint;
    bone.position = {at[0] - parentAt[0], at[1] - parentAt[1], at[2] - parentAt[2]};
    boneNode[i]   = scene.add(std::move(bone), parent >= 0 ? boneNode[size_t(parent)] : rig.root);
    rig.bones.emplace_back(b.name, boneNode[i]);
  }
  out.rigs.push_back(std::move(rig));

  dson::Item item;
  item.key   = options.attach ? options.attach->key : stem;
  item.label = label;
  item.objects.emplace_back(group, "");
  item.nodeNames.insert(stem);
  item.sources.insert(lower(utf8(file)));
  item.rigs.push_back(0);
  item.madeFor = wardrobe::Generation::eSkyrim;

  TextureFinder find{file.parent_path(), wear.textureHome, {}, {}, {}};
  std::vector<int> meshNodes;
  for(size_t si : drawn)
  {
    const Shape& s     = nif.shapes[si];
    Built        built = cageOf(nif, s, sk, scale);
    if(built.input.polyMaterial.empty())
      continue;
    Mesh m;
    m.name = s.name;
    m.geo  = std::make_shared<mesh::BuiltMesh>(mesh::build(built.input));
    for(int b : built.bones)
      m.bones.push_back(b >= 0 ? boneNode[size_t(b)] : out.rigs[0].root);
    m.cageSkin = built.skin;
    m.skin     = mesh::skin(*m.geo, m.cageSkin);

    const std::string surfaceName = prettyName(s.name);
    m.materialGroups              = {surfaceName};
    Material mat;
    mat.name = surfaceName;
    dson::Channels ch;
    mat.surface  = surfaceOf(surfaceName, s, find, ch);
    mat.context  = mat::Context::eItem;
    mat.channels = std::make_shared<const dson::Channels>(std::move(ch));
    mat.baseline = mat.channels;
    m.materials  = {int(scene.materials.size())};
    scene.materials.push_back(std::move(mat));

    const int meshIndex = int(scene.meshes.size());
    dson::MeshRef ref;
    ref.mesh         = meshIndex;
    ref.cageVertices = m.geo->dsonVertexCount;
    ref.geometryKey  = "nif:" + lower(utf8(file)) + "#" + lower(s.name);
    out.morphs.meshes.push_back(std::move(ref));
    dson::ItemPart part;
    part.item     = 0;
    part.mesh     = meshIndex;
    part.conforms = true;
    out.parts.push_back(part);
    scene.meshes.push_back(std::move(m));

    fv::Node meshNode;
    meshNode.name = s.name;
    meshNode.kind = NodeKind::eMesh;
    meshNode.mesh = meshIndex;
    meshNodes.push_back(scene.add(std::move(meshNode), group));
    item.meshes.push_back(meshIndex);
  }
  if(item.meshes.empty())
    throw std::runtime_error(fileName + " has nothing to wear in it");
  out.items.push_back(std::move(item));

  // bound at rest, as the DSON loader binds
  const auto worlds = scene.worlds();
  for(int n : meshNodes)
  {
    Mesh& m      = scene.meshes[size_t(scene.nodes[size_t(n)].mesh)];
    m.bindMatrix = worlds[size_t(n)];
    m.boneInverses.clear();
    for(int b : m.bones)
      m.boneInverses.push_back(inverse(worlds[size_t(b)]));
  }
  measureUvScales(scene);

  if(base)
    out.sourceBase = baseCageOf(*base, scale);
  else
    out.warnings.push_back(fileName + ": no body to fit it against (neither this file nor the others beside it keep the CBBE or UNP body), so only its bones were fitted.");
  if(wear.textureHome.empty())
    out.warnings.push_back(fileName + ": its textures weren't linked into the content library (none is set), so it shows untextured.");
  if(!find.missing.empty())
    out.warnings.push_back(fileName + ": " + std::to_string(find.missing.size()) + " of its textures aren't beside it (" + find.missing.front()
                           + (find.missing.size() > 1 ? " and more" : "") + "); install the outfit's own mod as well as a retexture's.");
  for(const std::string& p : find.problems)
    out.warnings.push_back("A texture couldn't be linked into the library: " + p);
  return out;
}

}  // namespace fv::nif
