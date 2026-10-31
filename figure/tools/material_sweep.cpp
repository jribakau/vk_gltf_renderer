// See material_sweep.hpp.

#include "material_sweep.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <figure/dson/channels.hpp>
#include <figure/dson/materials.hpp>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/io/gzip.hpp>
#include <figure/materials/lobes.hpp>
#include <figure/materials/surface.hpp>

#include "closure_row.hpp"
#include "scene_feature_detection.hpp"
#include "scene_import_gltf.hpp"
#include "scene_material_bits.hpp"

namespace fv::probe {

using dson::Json;

namespace {

const char* shaderName(mat::Surface::Shader s)
{
  switch(s)
  {
    case mat::Surface::Shader::eIray:
      return "iray";
    case mat::Surface::Shader::ePbrSkin:
      return "pbrskin";
    case mat::Surface::Shader::eDazDefault:
      return "dazdefault";
    case mat::Surface::Shader::eUberSurface:
      return "ubersurface";
    case mat::Surface::Shader::eAoa:
      return "aoa";
    case mat::Surface::Shader::eLegacy:
      break;
  }
  return "legacy";
}

const char* kindName(mat::Kind k)
{
  switch(k)
  {
    case mat::Kind::eSkin:
      return "skin";
    case mat::Kind::eCornea:
      return "cornea";
    case mat::Kind::eEye:
      return "eye";
    case mat::Kind::eLashes:
      return "lashes";
    case mat::Kind::eNails:
      return "nails";
    case mat::Kind::eTeeth:
      return "teeth";
    case mat::Kind::eMouth:
      return "mouth";
    case mat::Kind::eHair:
      return "hair";
    default:
      return "other";
  }
}

mat::SkinMode skinOf(const std::string& name)
{
  return name == "volume" ? mat::SkinMode::eVolume :
         name == "thin"   ? mat::SkinMode::eThin :
         name == "raster" ? mat::SkinMode::eRaster :
                            mat::SkinMode::eMeasured;
}

// A path as the library's relative paths are: UTF-8, forward slashes (.string() throws on some names)
std::string utf8(const std::filesystem::path& p)
{
  const std::u8string u = p.generic_u8string();
  return std::string(u.begin(), u.end());
}

double round3(float v)
{
  return std::round(double(v) * 1000.0) / 1000.0;
}

// The file's asset type, from the head of the file: the whole library holds tens of thousands of presets
std::string assetType(const std::filesystem::path& file)
{
  static const std::regex type("\"type\"\\s*:\\s*\"([^\"]+)\"");
  const std::string       text = fv::readMaybeCompressed(file);
  std::smatch             m;
  const std::string       head = text.substr(0, 8192);
  return std::regex_search(head, m, type) ? m[1].str() : std::string();
}

bool isItemType(const std::string& type)
{
  return type == "wearable" || type == "character" || type == "figure" || type == "prop" || type == "scene" || type == "scene_subset";
}

// A channel's value says something: not unset, zero, black or off
bool isSet(const dson::Channel& c)
{
  if(c.image)
    return true;
  const auto nonZero = [](const Json& v) { return (v.is_number() && v.get<double>() != 0.0) || (v.is_boolean() && v.get<bool>()); };
  if(c.value.is_array())
    return std::any_of(c.value.begin(), c.value.end(), nonZero);
  return nonZero(c.value);
}

// One thread's tally, merged at the end
struct Tally
{
  int                                loaded = 0, failed = 0, surfaces = 0;
  std::map<std::string, int>         shaders, kinds, extensions;
  std::map<std::string, std::string> failures;  // file -> why it did not load
  std::map<std::string, int>         layers;    // shader, layer -> surfaces: what competes for the coat's slot
  // shader -> channel -> {surfaces where it has a value or map and nothing read it, those with a map}
  std::map<std::string, std::map<std::string, std::pair<int, int>>> dropped;
  Json                               rows = Json::object();

  void merge(const Tally& o)
  {
    loaded += o.loaded;
    failed += o.failed;
    surfaces += o.surfaces;
    for(const auto& [k, v] : o.shaders)
      shaders[k] += v;
    for(const auto& [k, v] : o.kinds)
      kinds[k] += v;
    for(const auto& [k, v] : o.extensions)
      extensions[k] += v;
    for(const auto& [k, v] : o.layers)
      layers[k] += v;
    failures.insert(o.failures.begin(), o.failures.end());
    for(const auto& [shader, channels] : o.dropped)
      for(const auto& [channel, n] : channels)
      {
        auto& into = dropped[shader][channel];
        into.first += n.first;
        into.second += n.second;
      }
    rows.update(o.rows);
  }
};

// Everything the renderer takes from a material, as one hash: its row and texture infos byte for byte,
// its TLAS bits, shader features, sRGB slots and tangents. Each texture's number and flips come from
// its request alone, so a request that changes shows and the order they're asked in doesn't. A
// refactor of the material's path should move no surface's.
std::string rowFingerprint(const Material& m, const mat::LowerOptions& lowering)
{
  const auto mix = [](uint64_t h, const void* data, size_t size) {
    for(size_t i = 0; i < size; i++)
      h = (h ^ static_cast<const unsigned char*>(data)[i]) * 0x100000001b3ull;
    return h;
  };
  const auto resolve = [&](const mat::TextureRequest& req) {
    const std::string key = mat::requestKey(req);
    const uint64_t    h   = mix(0xcbf29ce484222325ull, key.data(), key.size());
    mat::ResolvedTexture r;
    r.index     = int(h & 0x3fff);
    r.flipGreen = (h >> 20 & 1) != 0;
    r.flipRows  = (h >> 21 & 1) != 0;
    return r;
  };
  const mat::Material material = mat::resolved(mat::lower(m.surface, m.edits, lowering), resolve);

  std::vector<shaderio::GltfTextureInfo> infos(1);
  const shaderio::GltfShadeMaterial      row  = nvvkgltf::MaterialCache::convertOne(material, {}, 0, infos);
  const nvvkgltf::MaterialBits           bits = nvvkgltf::materialBits(material);
  const std::string                      features = nvvkgltf::detectMaterialFeatures(material).toString();
  const std::set<int>                    srgb     = nvvkgltf::srgbTextures(material);
  const bool                             tangents = rs::materialNeedsTangents(material);

  uint64_t h = 0xcbf29ce484222325ull;
  h          = mix(h, &row, sizeof(row));
  h          = mix(h, infos.data(), infos.size() * sizeof(infos[0]));
  h          = mix(h, &bits.instanceFlags, sizeof(bits.instanceFlags));
  h          = mix(h, &bits.instanceMask, sizeof(bits.instanceMask));
  h          = mix(h, &bits.alphaSolid, sizeof(bits.alphaSolid));
  h          = mix(h, &bits.doubleSided, sizeof(bits.doubleSided));
  h          = mix(h, &bits.transmission, sizeof(bits.transmission));
  h          = mix(h, features.data(), features.size());
  for(int index : srgb)
    h = mix(h, &index, sizeof(index));
  h = mix(h, &tangents, sizeof(tangents));
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(h));
  // a file of it reads back as the same material, or the row says so
  return mat::fromGltf(mat::toGltf(material)) == material ? std::string(text) : std::string(text) + " not as a file";
}

void measure(const std::string& key, const Material& m, mat::SkinMode skin, Tally& t)
{
  const std::string shader = shaderName(m.surface.shader);
  const std::string kind   = kindName(m.surface.kind);
  t.surfaces++;
  t.shaders[shader]++;
  t.kinds[kind]++;

  // the layers a surface asks for, and what it asks of a coat's colour
  const mat::SpecularLobes lobes = mat::specularLobes(m.surface);
  const bool               coat  = m.surface.topCoat.has_value();
  if(coat)
    t.layers[shader + " top coat"]++;
  if(lobes.clearcoat)
    t.layers[shader + " tight dual lobe"]++;
  if(coat && lobes.clearcoat)
    t.layers[shader + " both"]++;
  if(coat && m.channels)
    if(const dson::Channel* c = m.channels->find({"Top Coat Color"}))
    {
      const bool white = c->value.is_array() && c->value.size() >= 3 && c->value[0] == 1.0 && c->value[1] == 1.0 && c->value[2] == 1.0;
      if(c->image)
        t.layers[shader + " top coat, coloured by a map"]++;
      else if(!white)
        t.layers[shader + " top coat, a flat colour"]++;
    }

  // what the parser reads of this surface's channels: the rest is authored and dropped
  if(m.channels)
  {
    dson::ChannelReads reads;
    (void)dson::parseSurface(m.name, *m.channels);
    std::set<const dson::Channel*> seen;
    for(const std::string& key : m.channels->order())
    {
      const auto c = m.channels->shared(key);
      if(!c || !seen.insert(c.get()).second || reads.read(c.get()) || !isSet(*c))
        continue;
      auto& n = t.dropped[shader][c->id.empty() ? key : c->id];
      n.first++;
      n.second += c->image ? 1 : 0;
    }
  }

  int                      next = 0;
  std::vector<std::string> used;
  mat::LowerOptions        lowering;
  lowering.skin            = skin;
  const auto resolve       = [&](const mat::TextureRequest&) {
    mat::ResolvedTexture r;
    r.index = next++;
    return r;
  };
  const tinygltf::Material gltf = build::materialToGltf(m, lowering, resolve, used);
  Json                     extensions = Json::array();
  for(const auto& [name, _] : gltf.extensions)
  {
    t.extensions[name]++;
    extensions.push_back(name);
  }

  ClosureOptions options;
  options.cosines = {1.0f, 0.4f};
  options.samples = 1024;
  Json closure    = Json::object();
  for(const ClosureAt& at : closureOfGltf(gltf, options))
  {
    Json lobes = Json::object();
    for(int l = 0; l < kLobeCount; l++)
      if(at.lobes[l].weight > 0.0005f)
        lobes[lobeName(l)] = round3(at.lobes[l].weight);
    closure[at.cosine == 1.0f ? "head" : "oblique"] = {{"albedo", {round3(at.total[0]), round3(at.total[1]), round3(at.total[2])}}, {"lobes", lobes}};
  }
  t.rows[key] = {{"shader", shader}, {"kind", kind}, {"extensions", extensions}, {"closure", closure}, {"row", rowFingerprint(m, lowering)}};
}

// What moved between two runs' rows: a line per surface, at most `limit` listed
Json compareRows(const Json& was, const Json& now, int limit)
{
  const auto far = [](const Json& a, const Json& b) {
    std::vector<std::pair<const Json*, const Json*>> stack{{&a, &b}};
    while(!stack.empty())
    {
      const auto [x, y] = stack.back();
      stack.pop_back();
      if(x->is_number() && y->is_number())
      {
        if(std::abs(x->get<double>() - y->get<double>()) > 0.005)
          return true;
      }
      else if(x->is_array() && y->is_array() && x->size() == y->size())
      {
        for(size_t i = 0; i < x->size(); i++)
          stack.push_back({&(*x)[i], &(*y)[i]});
      }
      else if(x->is_object() && y->is_object())
      {
        if(x->size() != y->size())
          return true;
        for(const auto& [k, v] : x->items())
        {
          if(!y->contains(k))
            return true;
          stack.push_back({&v, &(*y)[k]});
        }
      }
      else if(*x != *y)
        return true;
    }
    return false;
  };
  int  moved = 0, added = 0, removed = 0;
  Json list  = Json::array();
  std::map<std::string, int> byShader;
  for(const auto& [key, row] : now.items())
  {
    if(!was.contains(key))
    {
      added++;
      continue;
    }
    if(!far(was[key], row))
      continue;
    moved++;
    byShader[row.value("shader", "")]++;
    if(int(list.size()) < limit)
      list.push_back({{"surface", key}, {"was", was[key]}, {"now", row}});
  }
  for(const auto& [key, row] : was.items())
    if(!now.contains(key))
      removed++;
  return {{"moved", moved}, {"added", added}, {"removed", removed}, {"movedByShader", byShader}, {"first", list}};
}

}  // namespace

Json materialSweep(const std::filesystem::path& libraryDir, const std::filesystem::path& root, const Json& step)
{
  const auto start = std::chrono::steady_clock::now();
  const auto spent = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };

  std::vector<std::filesystem::path> files;
  const std::string                  match = step.value("match", std::string());
  for(const auto& e : std::filesystem::recursive_directory_iterator(libraryDir, std::filesystem::directory_options::skip_permission_denied))
  {
    if(!e.is_regular_file())
      continue;
    std::string ext = utf8(e.path().extension());
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if(ext == ".duf" && (match.empty() || utf8(e.path()).find(match) != std::string::npos))
      files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());

  const unsigned hardware = std::max(2u, std::thread::hardware_concurrency());
  const int      threads  = step.value("threads", 0) > 0 ? step.value("threads", 0) : int(hardware / 2);
  const int      limit    = step.value("limit", 0);
  const auto     skin     = skinOf(step.value("skin", std::string("measured")));
  const auto     library  = std::make_shared<dson::Library>(std::vector<std::filesystem::path>{libraryDir});

  std::atomic<size_t> next{0};
  std::atomic<int>    items{0}, done{0};
  std::mutex          mutex;
  Tally               total;
  std::map<std::string, int> skipped;
  const auto          worker = [&] {
    Tally                      tally;
    std::map<std::string, int> skips;
    for(size_t i = next++; i < files.size(); i = next++)
    {
      const std::string rel = utf8(std::filesystem::relative(files[i], libraryDir));
      try
      {
        const std::string type = assetType(files[i]);
        if(!isItemType(type) || (limit > 0 && items.fetch_add(1) >= limit))
        {
          skips[type.empty() ? "unknown" : type]++;
          continue;
        }
        // a store per item: a whole library's documents would not fit in memory at once
        auto store  = std::make_shared<dson::Store>(library);
        auto figure = Figure::load(store, rel, {.subdivision = 0});
        // a file can hold two surfaces of one name (two figures, say): the later ones by their place
        std::set<std::string> names;
        int                   index = 0;
        for(const Material& m : figure->loaded.scene.materials)
        {
          const std::string name = names.insert(m.name).second ? m.name : m.name + "@" + std::to_string(index);
          measure(rel + "#" + name, m, skin, tally);
          index++;
        }
        tally.loaded++;
      }
      catch(const std::exception& e)
      {
        tally.failed++;
        tally.failures[rel] = e.what();
      }
      const int n = ++done;
      if(n % 200 == 0)
        std::cerr << "[sweep] " << n << " files, " << int(spent()) << " s\n";
    }
    std::lock_guard lock(mutex);
    total.merge(tally);
    for(const auto& [k, v] : skips)
      skipped[k] += v;
  };
  std::vector<std::thread> pool;
  for(int t = 0; t < threads; t++)
    pool.emplace_back(worker);
  for(std::thread& t : pool)
    t.join();

  Json droppedJson = Json::object();
  for(const auto& [shader, channels] : total.dropped)
  {
    std::vector<std::pair<std::string, std::pair<int, int>>> sorted(channels.begin(), channels.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    Json list = Json::array();
    for(const auto& [channel, n] : sorted)
      list.push_back({{"channel", channel}, {"surfaces", n.first}, {"withMap", n.second}});
    droppedJson[shader] = list;
  }

  Json report = {{"files", {{"loaded", total.loaded}, {"failed", total.failed}, {"skipped", skipped}, {"failures", total.failures}}},
                 {"surfaces", total.surfaces},
                 {"shaders", total.shaders},
                 {"kinds", total.kinds},
                 {"extensions", total.extensions},
                 {"layers", total.layers},
                 {"dropped", droppedJson},
                 {"seconds", int(spent())}};

  if(step.contains("baseline"))
  {
    const std::filesystem::path file = root / step["baseline"].get<std::string>();
    if(!step.value("update", false) && std::filesystem::exists(file))
    {
      std::ifstream in(file, std::ios::binary);
      report["sinceBaseline"] = compareRows(Json::parse(in), total.rows, step.value("listMoved", 40));
    }
    else
    {
      std::filesystem::create_directories(file.parent_path());
      std::ofstream(file, std::ios::binary) << total.rows.dump();
      report["baselineWritten"] = utf8(file);
    }
  }
  return report;
}

}  // namespace fv::probe
