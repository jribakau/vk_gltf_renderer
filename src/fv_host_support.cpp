/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// FvHost's helpers that more than one of its files uses: paths, the texture server and the
// fallbacks, presets' and wear's addressing (declared in fv_host_impl.hpp).
//

#include "fv_host_impl.hpp"

namespace fvhost {

const char* kindName(PropKind k)
{
  switch(k)
  {
    case PropKind::eFigure:
      return "figure";
    case PropKind::eModel:
      return "model";
    case PropKind::eCamera:
      return "camera";
    case PropKind::eLight:
      return "light";
  }
  return "model";
}

std::filesystem::path pathOf(const std::string& utf8)
{
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

std::string utf8Of(const std::filesystem::path& p)
{
  const std::u8string s = p.generic_u8string();
  return std::string(s.begin(), s.end());
}

std::string encodeComponent(std::string_view s)
{
  static const char* hex = "0123456789ABCDEF";
  std::string        out;
  for(unsigned char c : s)
  {
    if(std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '!' || c == '~' || c == '*' || c == '\'' || c == '(' || c == ')')
      out += char(c);
    else
    {
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 15];
    }
  }
  return out;
}

std::string fmtNumber(double v)
{
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.5f", v);
  std::string out = buf;
  while(out.size() > 1 && out.back() == '0')
    out.pop_back();
  if(!out.empty() && out.back() == '.')
    out.pop_back();
  return out;
}

std::string encodeRel(std::string_view rel)
{
  std::string out, segment;
  auto        flush = [&] {
    if(!segment.empty())
      out += (out.empty() ? "" : "/") + encodeComponent(segment);
    segment.clear();
  };
  for(char c : rel)
  {
    if(c == '/' || c == '\\')
      flush();
    else
      segment += c;
  }
  flush();
  return out;
}

std::string normRel(const std::string& path)
{
  std::string rel = fv::dson::decodeUri(path);
  std::replace(rel.begin(), rel.end(), '\\', '/');
  const size_t at = rel.find_first_not_of('/');
  return at == std::string::npos ? std::string{} : rel.substr(at);
}

int pieceOfKey(const std::string& key)
{
  const size_t dot = key.find('.');
  return dot == std::string::npos ? 0 : std::max(std::atoi(key.c_str() + dot + 1), 0);
}

std::string relKey(const std::string& path)
{
  std::string key = normRel(path);
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return key;
}

std::string uriOfPath(const std::string& path)
{
  std::string out;
  for(unsigned char c : path)
  {
    // '+': the reading side, a URI's decoder, takes it for a space
    if(c == '%' || c == ' ' || c == '#' || c == '?' || c == '+' || c >= 0x80)
    {
      static const char* hex = "0123456789ABCDEF";
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 15];
    }
    else
      out += char(c);
  }
  return out;
}

std::string mimeOf(const std::string& path)
{
  std::string ext = path.substr(path.find_last_of('.') + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if(ext == "png")
    return "image/png";
  if(ext == "ktx2")
    return "image/ktx2";
  if(ext == "webp")
    return "image/webp";
  return "image/jpeg";
}

TextureCalls& textureCalls()
{
  static TextureCalls calls;
  return calls;
}

namespace {
uint64_t priced(const rs::MeshCounts& c, const HostRenderer& renderer)
{
  return rs::meshBufferBytes(c) + renderer.blasBytes(c.triangles, c.vertices, c.deforms());
}
}  // namespace

std::string problemOf(const std::exception& e)
{
  return dynamic_cast<const std::bad_alloc*>(&e) ? "System memory ran out." : e.what();
}

uint64_t propRamBytes(const Prop& p)
{
  uint64_t bytes = 0;
  if(p.figure)
  {
    for(const fv::Mesh& m : p.figure->fig->loaded.scene.meshes)
    {
      if(m.geo)
        bytes += fv::mesh::ramBytes(*m.geo);
      bytes += m.skin.skinIndex.size() * sizeof(uint16_t) + m.skin.skinWeight.size() * sizeof(float) + m.skin.skinIndex8.size() * sizeof(uint16_t)
               + m.skin.skinWeight8.size() * sizeof(float) + m.skin.dualQuatShare.size() * sizeof(float);
    }
  }
  else if(p.model)
    for(const tinygltf::Buffer& b : p.model->buffers)
      bytes += b.data.size();
  return bytes;
}

void sideBySide(size_t count, int threads, const std::function<void(size_t)>& fn)
{
  std::atomic<size_t>      next{0};
  std::mutex               mutex;
  std::exception_ptr       first;
  std::vector<std::thread> workers;
  for(int w = 0; w < std::min<int>(threads, int(count)); w++)
    workers.emplace_back([&] {
      for(size_t i; (i = next++) < count;)
        try
        {
          fn(i);
        }
        catch(...)
        {
          std::lock_guard lock(mutex);
          if(!first)
            first = std::current_exception();
        }
    });
  for(std::thread& w : workers)
    w.join();
  if(first)
    std::rethrow_exception(first);
}

uint64_t geometryBytes(const fv::SceneGraph& scene, const HostRenderer& renderer)
{
  uint64_t bytes = 0;
  for(const fv::Mesh& m : scene.meshes)
    bytes += priced(fv::meshCounts(m), renderer);
  return bytes;
}

uint64_t geometryBytes(const fv::SceneGraph& scene, const std::vector<int>& meshes, const HostRenderer& renderer)
{
  uint64_t bytes = 0;
  for(int m : meshes)
    if(m >= 0 && m < int(scene.meshes.size()))
      bytes += priced(fv::meshCounts(scene.meshes[size_t(m)]), renderer);
  return bytes;
}

uint64_t geometryBytes(const tinygltf::Model& model, const HostRenderer& renderer)
{
  uint64_t bytes = 0;
  for(const rs::MeshCounts& c : rs::modelMeshCounts(model))
    bytes += priced(c, renderer);
  return bytes;
}

CookedTexture cookedTexture(const HostSettings& s, const fv::build::TextureRequest& req, bool flip, bool cachedOnly, bool quiet)
{
  // http://127.0.0.1:<port>/<token>
  const std::string& base = s.texServer;
  const size_t       hostAt = base.find("://");
  const size_t       pathAt = base.find('/', hostAt == std::string::npos ? 0 : hostAt + 3);
  if(hostAt == std::string::npos || pathAt == std::string::npos)
    return {};
  httplib::Client cli(base.substr(0, pathAt));
  cli.set_read_timeout(cachedOnly ? 30s : 600s);
  std::string url = base.substr(pathAt) + "/" + s.rootId + "/" + encodeRel(normRel(req.rel)) + "?srgb=" + (req.srgb ? "1" : "0")
                    + "&flip=" + (flip ? "1" : "0");
  if(req.normal)
    url += "&nm=1";
  if(req.normalLength)
    url += "&nl=1";
  if(!req.channels.empty())
    url += "&ch=" + req.channels;
  // a source in a query value: <rootId>/<path>, percent-encoded
  const auto source = [&](const std::string& rel) { return encodeComponent(s.rootId + "/" + normRel(rel)); };
  if(!req.alphaRel.empty())
    url += "&alpha=" + source(req.alphaRel) + (req.alphaFromAlpha ? "&aa=1" : "");
  if(!req.metalRel.empty())
    url += "&metal=" + source(req.metalRel);
  if(!req.bumpRel.empty())
    url += "&bump=" + source(req.bumpRel) + "&bs=" + fmtNumber(req.bumpStrength);
  if(req.bumpHeight > 0)
    url += "&bh=" + fmtNumber(req.bumpHeight);
  if(req.makeup)
  {
    // mk=<base 1|0>|<colour source>|<weight source>|<weight>|<r,g,b tint>|<r,g,b factor>, percent-encoded as a whole
    const fv::mat::MakeupRequest& m = *req.makeup;
    const auto                    rgb = [](const std::array<float, 3>& c) { return fmtNumber(c[0]) + "," + fmtNumber(c[1]) + "," + fmtNumber(c[2]); };
    const std::string             mk = std::string(m.baseIsSource ? "1" : "0") + "|" + (m.colorRel.empty() ? "-" : source(m.colorRel)) + "|"
                           + (m.weightRel.empty() ? "-" : source(m.weightRel)) + "|" + fmtNumber(m.weightFactor) + "|" + rgb(m.tint) + "|" + rgb(m.factor);
    url += "&mk=" + encodeComponent(mk);
  }
  // ir=<iris r,g,b | ->|<strength>|<eye | iris>|<x,y,r;...>|<sclera r,g,b | ->, percent-encoded as a whole
  if(req.iris)
    url += "&ir=" + encodeComponent(fv::mat::irisQuery(*req.iris));
  // rc=<r,g,b>|<strength>[|<reference source>], percent-encoded as a whole
  if(req.recolor)
    url += "&rc=" + encodeComponent(fv::mat::recolorQuery(*req.recolor) + (req.recolor->reference.empty() ? "" : "|" + source(req.recolor->reference)));
  // ly=<layer>;<layer>..., percent-encoded as a whole. A layer is its source, or
  // <source>^<blend>^<opacity>^<mask source>^<r,g,b>^<canvas> (texcache.rs Layer::parse)
  if(!req.layers.empty())
  {
    std::string ly;
    for(const std::string& text : req.layers)
    {
      const fv::mat::MapLayer layer = fv::mat::parseLayer(text);
      const std::string       file  = layer.file.empty() ? std::string() : source(layer.file);
      ly += ly.empty() ? "" : ";";
      if(text == layer.file)
        ly += file;
      else
        ly += file + "^" + layer.blend + "^" + std::to_string(layer.opacity) + "^" + (layer.mask.empty() ? std::string() : source(layer.mask)) + "^"
              + std::to_string(layer.color[0]) + "," + std::to_string(layer.color[1]) + "," + std::to_string(layer.color[2]) + "^" + std::to_string(layer.canvas);
    }
    url += "&ly=" + encodeComponent(ly);
  }
  // mul=<source>;<source>..., percent-encoded as a whole
  if(!req.mulRels.empty())
  {
    std::string mul;
    for(const std::string& m : req.mulRels)
      mul += (mul.empty() ? "" : ";") + source(m);
    url += "&mul=" + encodeComponent(mul);
  }
  url += cachedOnly ? "&path=1&cached=1" : "&path=1";
  if(!textureCalls().add(&cli))
    return {};
  auto res = cli.Get(url);
  textureCalls().remove(&cli);
  if(res && res->status == 202 && cachedOnly)
    return {.uncooked = true};
  if(!res || res->status != 200)
  {
    if(!quiet && !textureCalls().stopped())
      LOGW("texture %s: %s\n", req.rel.c_str(), res ? (std::to_string(res->status) + " " + res->body).c_str() : "no answer");
    return {};
  }
  return {.image = fv::build::TextureImage{uriOfPath(res->body), "image/ktx2", flip}};
}

namespace {

// The fallbacks composed here (Original textures, or no cooker) and the cooks' one-texel stand-ins, in
// the temporary folder. The first time it is asked for in a run, the composed ones are trimmed to
// kComposedCapBytes, the least recently used first (one reused is touched): each cold load used to
// leave its derived maps here at full size, never read again once cooked.
constexpr uint64_t kComposedCapBytes = 2ull << 30;

const std::filesystem::path& composedFolder()
{
  static const std::filesystem::path dir = [] {
    std::error_code       ec;
    std::filesystem::path d = std::filesystem::temp_directory_path(ec) / "fv-composed";
    std::filesystem::create_directories(d, ec);
    if(const uint64_t removed = fv::textures::trimFolder(d, kComposedCapBytes, L"flat-"))
      LOGI("[fv] composed textures: %.2f GB of the least used removed\n", double(removed) / 1e9);
    return d;
  }();
  return dir;
}

// One texel as a PNG: white, which leaves the material's factors as they are, and a flat normal
constexpr uint8_t kFlatWhitePng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
    0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8, 0xff, 0xff, 0x3f, 0x00, 0x05, 0xfe, 0x02, 0xfe, 0x33,
    0x12, 0x95, 0x14, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
constexpr uint8_t kFlatNormalPng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
    0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x68, 0x68, 0xf8, 0x0f, 0x00, 0x03, 0x83, 0x02, 0x00, 0x3e,
    0x58, 0x28, 0x25, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};

// A derived map's stand-in until its cook lands (Fallback::eFlat). One texel, but a texture: the
// material has the slot the cook goes into, as FigureFeed::retexture fills only a slot it has.
std::optional<fv::build::TextureImage> flatTexture(bool normal, bool flip)
{
  const std::filesystem::path file = composedFolder() / (normal ? "flat-normal.png" : "flat-white.png");
  std::error_code             ec;
  if(!std::filesystem::exists(file, ec))
  {
    static std::mutex writing;
    std::lock_guard   lock(writing);
    const auto        tmp = file.string() + "." + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    {
      std::ofstream out(std::filesystem::path(tmp), std::ios::binary);
      out.write(reinterpret_cast<const char*>(normal ? kFlatNormalPng : kFlatWhitePng), std::streamsize(sizeof kFlatWhitePng));
    }
    // another run may have placed it meanwhile: either one will do
    std::filesystem::rename(tmp, file, ec);
    std::filesystem::remove(tmp, ec);
  }
  if(!std::filesystem::exists(file, ec))
    return std::nullopt;
  return fv::build::TextureImage{uriOfPath(utf8Of(file)), "image/png", flip};
}

}  // namespace

void cookAll(const HostSettings&                                                                  s,
             std::vector<fv::build::TextureRequest>                                               requests,
             const std::function<void(const fv::build::TextureRequest&, fv::build::TextureImage)>& landed)
{
  std::stable_partition(requests.begin(), requests.end(), [](const fv::build::TextureRequest& r) { return r.srgb; });
  sideBySide(requests.size(), 4, [&](size_t i) {
    if(CookedTexture c = cookedTexture(s, requests[i], true, false, true); c.image && landed)
      landed(requests[i], std::move(*c.image));
  });
}

std::vector<std::optional<fv::build::TextureImage>> resolveTextures(const HostSettings&                           s,
                                                                    const fv::dson::Library*                      lib,
                                                                    const std::vector<fv::build::TextureRequest>& requests,
                                                                    bool                                          flipped,
                                                                    std::vector<fv::build::TextureRequest>&       uncooked)
{
  const bool                                          cooker = !s.texServer.empty() && !s.rootId.empty();
  composedFolder();  // trimmed on the first load of a run, a warm one too
  std::vector<std::optional<fv::build::TextureImage>> found(requests.size());
  std::vector<uint8_t>                                missed(requests.size(), 0);
  sideBySide(requests.size(), 8, [&](size_t i) {
    const fv::build::TextureRequest& req = requests[i];
    if(!flipped)
      found[i] = lib ? fallbackTexture(s, *lib, req, cooker) : std::nullopt;
    else if(cooker)
    {
      CookedTexture c = cookedTexture(s, req, true, true);
      if(!c.uncooked)
        found[i] = std::move(c.image);
      else
      {
        // what can only be shown cooked (a cutout, a makeup, a .tif) is cooked now, as it will stay:
        // cooked unflipped and then again flipped, the load waits for one nothing reads again
        bool final = false;
        if(lib)
          found[i] = fallbackTexture(s, *lib, req, cooker, true, &final);
        missed[i] = final ? 0 : 1;
        // shown until its cook lands, so decoded smaller
        if(found[i] && missed[i])
          found[i]->standIn = true;
      }
    }
  });
  for(size_t i = 0; i < requests.size(); i++)
    if(missed[i])
      uncooked.push_back(requests[i]);
  return found;
}

std::optional<fv::build::TextureImage> sourceTexture(const fv::dson::Library& lib, const fv::build::TextureRequest& req)
{
  auto path = lib.find(normRel(req.rel));
  if(!path)
    return std::nullopt;
  const std::string p = utf8Of(*path);
  return fv::build::TextureImage{uriOfPath(p), mimeOf(p), false};
}

std::optional<fv::build::TextureImage> fallbackTexture(const HostSettings& s, const fv::dson::Library& lib, const fv::build::TextureRequest& req, bool cooker, bool flip, bool* final)
{
  using fv::textures::Fallback;
  const bool     derived = fv::textures::derivedChannels(req.channels);
  const bool     laid    = !req.alphaRel.empty() || req.makeup || req.iris || req.recolor || !req.layers.empty() || !req.mulRels.empty();
  // an iris or a recolour only the cooker makes (composeImage has neither): cooked under Original textures too
  const Fallback route =
      cooker && (req.iris || req.recolor) ? Fallback::eCook : fv::textures::fallbackOf(req.rel, derived, laid, cooker, s.originalTextures);
  if(route == Fallback::eCook)
  {
    if(final)
      *final = flip;
    return cookedTexture(s, req, flip, false).image;
  }
  // a source that isn't there has no stand-in, as it would have nothing composed
  if(route == Fallback::eFlat)
    return lib.find(normRel(req.rel)) ? flatTexture(req.channels == "bn", flip) : std::nullopt;
  // composed into a PNG beside the others made this way
  if(route == Fallback::eCompose)
  {
    std::error_code ec;
    const auto&     dir = composedFolder();
    // named by the request and by where its sources are, their size and their time (FNV-1a, the
    // same from build to build): a texture edited in place, or the same relative path in another
    // library, composes again instead of showing the old picture
    uint64_t   h   = 1469598103934665603ULL;
    const auto mix = [&h](const std::string& s) {
      for(unsigned char c : s)
      {
        h ^= c;
        h *= 1099511628211ULL;
      }
      h ^= '|';
      h *= 1099511628211ULL;
    };
    mix(fv::build::requestKey(req));
    {
      std::vector<std::string> sources = {req.rel, req.alphaRel, req.metalRel, req.bumpRel};
      for(const std::string& text : req.layers)
      {
        const fv::mat::MapLayer layer = fv::mat::parseLayer(text);
        sources.push_back(layer.file);
        sources.push_back(layer.mask);
      }
      sources.insert(sources.end(), req.mulRels.begin(), req.mulRels.end());
      if(req.makeup)
        sources.insert(sources.end(), {req.makeup->colorRel, req.makeup->weightRel});
      for(const std::string& rel : sources)
        if(!rel.empty())
          if(auto src = lib.find(normRel(rel)))
          {
            std::error_code e2;
            mix(utf8Of(*src));
            mix(std::to_string(std::filesystem::file_size(*src, e2)));
            mix(std::to_string(std::filesystem::last_write_time(*src, e2).time_since_epoch().count()));
          }
    }
    char name[32];
    std::snprintf(name, sizeof name, "%016llx", static_cast<unsigned long long>(h));
    const auto file = dir / (std::string(name) + ".png");
    if(!std::filesystem::exists(file, ec))
    {
      fv::build::ExportOptions o;
      o.find     = [&](const std::string& rel) { return lib.find(normRel(rel)); };
      o.bottomUp = false;
      std::string why;
      const auto  png = fv::build::composeImage(req, o, &why);
      if(!png)
      {
        LOGW("texture %s: %s\n", req.rel.c_str(), why.c_str());
        return derived ? std::nullopt : sourceTexture(lib, req);
      }
      std::filesystem::create_directories(dir, ec);
      const auto    tmp = file.string() + "." + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
      std::ofstream out(std::filesystem::path(tmp), std::ios::binary);
      out.write(reinterpret_cast<const char*>(png->data()), std::streamsize(png->size()));
      out.close();
      if(!out)
      {
        LOGW("texture %s: could not write %s\n", req.rel.c_str(), tmp.c_str());
        std::filesystem::remove(tmp, ec);
        return derived ? std::nullopt : sourceTexture(lib, req);
      }
      std::filesystem::rename(tmp, file, ec);
      if(ec)
      {
        LOGW("texture %s: could not place %s: %s\n", req.rel.c_str(), utf8Of(file).c_str(), ec.message().c_str());
        std::filesystem::remove(tmp, ec);
        return derived ? std::nullopt : sourceTexture(lib, req);
      }
    }
    else
      std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(), ec);  // used: kept by the trim
    return fv::build::TextureImage{uriOfPath(utf8Of(file)), "image/png", false};
  }
  if(route == Fallback::eNone)
    return std::nullopt;
  auto image = sourceTexture(lib, req);
  if(image && req.channels == "ng")
    image->flipGreen = true;
  return image;
}

std::string defaultBrowMaterial(const fv::SceneGraph& scene)
{
  static const std::string dir = "data/DAZ 3D/Genesis 9/Base/Tools/Script Loads/Eyebrows/";
  for(const fv::Node& n : scene.nodes)
  {
    if(n.dsonNode.rfind("G9EyebrowCards", 0) == 0)
      return dir + "Card Eyebrows MAT.duf";
    if(n.dsonNode.rfind("G9EyebrowFibers", 0) == 0)
      return dir + "Fiber Eyebrows MAT.duf";
  }
  return {};
}

std::optional<std::pair<std::vector<std::filesystem::path>, std::string>> libraryFor(const HostSettings& s, const std::filesystem::path& file)
{
  std::error_code ec;
  const auto      abs = std::filesystem::weakly_canonical(file, ec);
  for(const auto& root : s.roots)
  {
    const auto r   = std::filesystem::weakly_canonical(root, ec);
    const auto rel = abs.lexically_relative(r);
    if(!rel.empty() && *rel.begin() != "..")
    {
      // the file's own library first, then the other content folders for whatever it references
      std::vector<std::filesystem::path> roots{root};
      for(const auto& other : s.roots)
        if(other != root)
          roots.push_back(other);
      return std::make_pair(roots, utf8Of(rel));
    }
  }
  auto dir = abs.parent_path();
  for(int i = 0; i < 8 && !dir.empty(); i++, dir = dir.parent_path())
  {
    if(std::filesystem::is_directory(dir / "data", ec))
    {
      std::vector<std::filesystem::path> roots{dir};
      roots.insert(roots.end(), s.roots.begin(), s.roots.end());
      return std::make_pair(roots, utf8Of(abs.lexically_relative(dir)));
    }
    if(dir == dir.root_path())
      break;
  }
  return std::nullopt;
}

std::string axisKey(const fv::pose::Axis& a)
{
  return std::string(fv::pose::kindName(a.kind)) + "/" + a.axis;
}

Json poseSnapshot(const Loaded& f)
{
  Json bones = Json::object();
  int  order = 0;
  for(const auto& pb : f.fig->pose->bones())
  {
    if(pb.axes.empty())
      continue;
    Json axes = Json::object();
    int  i    = 0;
    for(const auto& a : pb.axes)
      axes[axisKey(a)] = {{"label", a.label}, {"kind", fv::pose::kindName(a.kind)}, {"axis", a.axis}, {"min", a.min}, {"max", a.max},
                          {"value", a.value}, {"initial", a.initial}, {"order", i++}};
    bones[pb.id] = {{"name", pb.name}, {"label", pb.label}, {"group", pb.group}, {"order", order++}, {"node", pb.bone}, {"axes", axes}};
  }
  return {{"figure", f.id}, {"changed", f.fig->pose->changed()}, {"eyes", f.fig->pose->hasEyes()}, {"bones", bones}};
}

std::string fileStem(const std::string& rel)
{
  const size_t slash = rel.find_last_of("/\\");
  std::string  name  = slash == std::string::npos ? rel : rel.substr(slash + 1);
  const size_t dot   = name.find_last_of('.');
  if(dot != std::string::npos)
    name.resize(dot);
  return name;
}

nvmcp::detail::ParameterList exposedParameters(const nvutils::ParameterRegistry* registry)
{
  nvmcp::detail::ParameterList out;
  if(!registry)
    return out;
  for(const nvutils::ParameterBase* p : registry->getParameters())
    if(p->destination.raw && p->type != nvutils::ParameterBase::Type::BOOL8_TRIGGER
       && p->type != nvutils::ParameterBase::Type::CUSTOM && p->type != nvutils::ParameterBase::Type::INVALID
       && fv::hostExposesParameter(p->info.name))
      out.push_back(p);
  return out;
}

Json toolJson(const nvmcp::ToolResult& r)
{
  if(r.isError)
    throw std::runtime_error(r.json);
  return Json::parse(r.json);
}

fv::build::BuildOptions buildOptionsOf(const Loaded& f, fv::mat::SkinMode skin)
{
  fv::build::BuildOptions options;
  options.texturesFlipped = f.texturesFlipped;
  options.skin            = skin;
  options.texture         = [&f](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
    auto it = f.images.find(fv::build::requestKey(req));
    return it == f.images.end() ? std::nullopt : it->second;
  };
  options.walkThrough = fv::walkThroughMaterials(f.fig->loaded.scene, f.fig->items);
  return options;
}

int itemIdOf(const Json& v)
{
  if(v.is_number_integer())
    return v.get<int>();
  const std::string s = v.get<std::string>();
  return std::atoi(s.c_str() + (s.rfind("i:", 0) == 0 ? 2 : 0));
}

int itemId(const Json& args)
{
  return itemIdOf(args.at("item"));
}

std::vector<int> itemIds(const Json& args)
{
  const Json&      v = args.at("item");
  std::vector<int> out;
  if(v.is_array())
    for(const Json& x : v)
      out.push_back(itemIdOf(x));
  else
    out.push_back(itemIdOf(v));
  return out;
}

const fv::WornItem* wornItem(const fv::Figure& fig, int id)
{
  const auto it = std::find_if(fig.items.begin(), fig.items.end(), [&](const fv::WornItem& i) { return i.id == id; });
  return it == fig.items.end() ? nullptr : &*it;
}

fv::dforce::Mode dforceMode(const std::string& name)
{
  if(name == "live")
    return fv::dforce::Mode::eLive;
  if(name == "gpu")
  {
#ifdef FV_DFORCE_GPU
    return fv::dforce::Mode::eLiveGpu;
#else
    throw std::runtime_error("this renderer was built without the GPU cloth solver (FV_DFORCE_GPU)");
#endif
  }
  return fv::dforce::Mode::eSettle;
}

}  // namespace fvhost
