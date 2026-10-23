#include <figure/gltf/export_glb.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace fv::build {

namespace {

struct Image
{
  int                  w = 0, h = 0;
  std::vector<uint8_t> px;  // RGBA
  uint8_t*             at(int x, int y) { return &px[(size_t(y) * size_t(w) + size_t(x)) * 4]; }
  const uint8_t*       at(int x, int y) const { return &px[(size_t(y) * size_t(w) + size_t(x)) * 4]; }
};

std::optional<Image> decode(const ExportOptions& o, const std::string& rel, std::string* why)
{
  const auto path = o.find ? o.find(rel) : std::nullopt;
  if(!path)
  {
    if(why)
      *why = rel + " is not in the library";
    return std::nullopt;
  }
  int         w = 0, h = 0, n = 0;
  const std::string utf8 = path->generic_string();
  unsigned char*    data = stbi_load(utf8.c_str(), &w, &h, &n, 4);
  if(!data)
  {
    if(why)
      *why = rel + " could not be read (" + (stbi_failure_reason() ? stbi_failure_reason() : "unknown format") + ")";
    return std::nullopt;
  }
  Image img;
  img.w = w;
  img.h = h;
  img.px.assign(data, data + size_t(w) * size_t(h) * 4);
  stbi_image_free(data);
  return img;
}

// Bilinear: the other maps of one material share the base's size.
Image resized(const Image& src, int w, int h)
{
  if(src.w == w && src.h == h)
    return src;
  Image out;
  out.w = w;
  out.h = h;
  out.px.resize(size_t(w) * size_t(h) * 4);
  for(int y = 0; y < h; y++)
  {
    const float fy = (float(y) + 0.5f) * float(src.h) / float(h) - 0.5f;
    const int   y0 = std::clamp(int(std::floor(fy)), 0, src.h - 1), y1 = std::min(y0 + 1, src.h - 1);
    const float ty = std::clamp(fy - float(y0), 0.f, 1.f);
    for(int x = 0; x < w; x++)
    {
      const float fx = (float(x) + 0.5f) * float(src.w) / float(w) - 0.5f;
      const int   x0 = std::clamp(int(std::floor(fx)), 0, src.w - 1), x1 = std::min(x0 + 1, src.w - 1);
      const float tx = std::clamp(fx - float(x0), 0.f, 1.f);
      for(int c = 0; c < 4; c++)
      {
        const float a = src.at(x0, y0)[c] * (1 - tx) + src.at(x1, y0)[c] * tx;
        const float b = src.at(x0, y1)[c] * (1 - tx) + src.at(x1, y1)[c] * tx;
        out.at(x, y)[c] = uint8_t(std::lround(a * (1 - ty) + b * ty));
      }
    }
  }
  return out;
}

float decodeSrgb(uint8_t v)
{
  const float c = float(v) / 255.f;
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

uint8_t encodeValue(float v, bool srgb)
{
  v = std::clamp(v, 0.f, 1.f);
  if(srgb)
    v = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
  return uint8_t(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
}

// PBRSkin makeup over a base colour, blended in linear light: base x factor towards the makeup
// colour x tint by the weight.
Image bakeMakeup(const Image& base, const mat::MakeupRequest& mk, const Image* color, const Image* weight, bool srgb)
{
  Image out;
  out.w = base.w;
  out.h = base.h;
  out.px.resize(base.px.size());
  const auto lin = [&](uint8_t v) { return srgb ? decodeSrgb(v) : float(v) / 255.f; };
  for(int y = 0; y < base.h; y++)
    for(int x = 0; x < base.w; x++)
    {
      const float wt = mk.weightFactor * (weight ? float(weight->at(x, y)[0]) / 255.f : 1.f);
      for(int c = 0; c < 3; c++)
      {
        const float from = (mk.baseIsSource ? lin(base.at(x, y)[c]) : 1.f) * mk.factor[size_t(c)];
        const float to   = (color ? lin(color->at(x, y)[c]) : 1.f) * mk.tint[size_t(c)];
        out.at(x, y)[c]  = encodeValue(from + (to - from) * wt, srgb);
      }
      out.at(x, y)[3] = mk.baseIsSource ? base.at(x, y)[3] : 255;
    }
  return out;
}

constexpr float kBumpSlope = 10.f;

// A normal map from a height map (the pixel's luma), in glTF's green convention for a mesh without
// tangents; edges wrap, as DAZ maps repeat. `flip`: v runs up the image.
Image bumpToNormal(const Image& src, bool flip)
{
  const float        sy = flip ? -1.f : 1.f;
  std::vector<float> height(size_t(src.w) * size_t(src.h));
  for(size_t i = 0; i < height.size(); i++)
    height[i] = (src.px[i * 4] * 0.299f + src.px[i * 4 + 1] * 0.587f + src.px[i * 4 + 2] * 0.114f) / 255.f;
  Image out;
  out.w = src.w;
  out.h = src.h;
  out.px.resize(src.px.size());
  for(int y = 0; y < src.h; y++)
  {
    const int y0 = (y + src.h - 1) % src.h, y1 = (y + 1) % src.h;
    for(int x = 0; x < src.w; x++)
    {
      const int   x0 = (x + src.w - 1) % src.w, x1 = (x + 1) % src.w;
      const float dx = (height[size_t(y) * size_t(src.w) + size_t(x1)] - height[size_t(y) * size_t(src.w) + size_t(x0)]) * 0.5f;
      const float dy = (height[size_t(y1) * size_t(src.w) + size_t(x)] - height[size_t(y0) * size_t(src.w) + size_t(x)]) * 0.5f;
      const float nx = -dx * kBumpSlope, ny = sy * dy * kBumpSlope, nz = 1.f;
      const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
      const auto  enc = [&](float v) { return uint8_t(std::lround(std::clamp((v / len * 0.5f + 0.5f) * 255.f, 0.f, 255.f))); };
      uint8_t*    p   = out.at(x, y);
      p[0] = enc(nx);
      p[1] = enc(ny);
      p[2] = enc(nz);
      p[3] = 255;
    }
  }
  return out;
}

// Reoriented normal mapping: the detail normal bends the base without flattening it (Stephen Hill,
// "Blending in Detail"). Both images are in one tangent space. `strength` scales the detail's x and y.
Image reorientedNormals(const Image& base, const Image& detail, float strength)
{
  Image out;
  out.w = base.w;
  out.h = base.h;
  out.px.resize(base.px.size());
  const auto dec = [](uint8_t v) { return float(v) / 255.f * 2.f - 1.f; };
  const auto enc = [](float v) { return uint8_t(std::lround(std::clamp((v * 0.5f + 0.5f) * 255.f, 0.f, 255.f))); };
  for(size_t i = 0; i < base.px.size() / 4; i++)
  {
    const uint8_t* b = &base.px[i * 4];
    const uint8_t* d = &detail.px[i * 4];
    const float    nx = dec(b[0]), ny = dec(b[1]), nz = dec(b[2]);
    float          dx = dec(d[0]) * strength, dy = dec(d[1]) * strength, dz = std::max(std::abs(dec(d[2])), 1e-4f);
    const float    len = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1e-8f);
    dx /= len;
    dy /= len;
    dz /= len;
    // t = n + (0, 0, 1), u = (-d.x, -d.y, d.z), r = t * dot(t, u) - u * t.z
    const float tx = nx, ty = ny, tz = nz + 1.f;
    const float ux = -dx, uy = -dy, uz = dz;
    const float dot = tx * ux + ty * uy + tz * uz;
    float       rx = tx * dot - ux * tz, ry = ty * dot - uy * tz, rz = tz * dot - uz * tz;
    const float rl = std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-8f);
    uint8_t*    p  = &out.px[i * 4];
    p[0] = enc(rx / rl);
    p[1] = enc(ry / rl);
    p[2] = enc(rz / rl);
    p[3] = 255;
  }
  return out;
}

void flipRows(Image& img)
{
  for(int y = 0; y < img.h / 2; y++)
    std::swap_ranges(img.at(0, y), img.at(0, y) + size_t(img.w) * 4, img.at(0, img.h - 1 - y));
}

}  // namespace

std::optional<std::vector<uint8_t>> composeImage(const TextureRequest& req, const ExportOptions& o, std::string* why)
{
  auto base = decode(o, req.rel, why);
  if(!base)
    return std::nullopt;
  Image img = std::move(*base);
  // the other maps, at the base's size
  const auto other = [&](const std::string& rel) -> std::optional<Image> {
    auto i = decode(o, rel, why);
    if(!i)
      return std::nullopt;
    return resized(*i, img.w, img.h);
  };
  if(req.makeup)
  {
    std::optional<Image> color, weight;
    if(!req.makeup->colorRel.empty() && !(color = other(req.makeup->colorRel)))
      return std::nullopt;
    if(!req.makeup->weightRel.empty() && !(weight = other(req.makeup->weightRel)))
      return std::nullopt;
    img = bakeMakeup(img, *req.makeup, color ? &*color : nullptr, weight ? &*weight : nullptr, req.srgb);
  }
  if(!req.alphaRel.empty())
  {
    const auto a = other(req.alphaRel);
    if(!a)
      return std::nullopt;
    // a cutout map's green channel is its alpha
    for(size_t i = 0; i < img.px.size() / 4; i++)
      img.px[i * 4 + 3] = a->px[i * 4 + 1];
  }
  // rows are flipped last: the derivations assume rows run top-down
  bool greenDone = false;
  if(!req.bumpRel.empty())
  {
    const auto height = other(req.bumpRel);
    if(!height)
      return std::nullopt;
    const Image detail = bumpToNormal(*height, true);
    for(size_t i = 0; i < img.px.size() / 4; i++)
      img.px[i * 4 + 1] = uint8_t(255 - img.px[i * 4 + 1]);
    img = reorientedNormals(img, detail, req.bumpStrength);
    greenDone = true;
  }
  if(req.channels == "bn")
    img = bumpToNormal(img, true);
  flipRows(img);
  if(req.channels == "ng" && !greenDone)
    for(size_t i = 0; i < img.px.size() / 4; i++)
      img.px[i * 4 + 1] = uint8_t(255 - img.px[i * 4 + 1]);
  if(req.channels == "mr")
    for(size_t i = 0; i < img.px.size() / 4; i++)
    {
      img.px[i * 4 + 2] = 255;
      img.px[i * 4 + 3] = 255;
    }
  bool alpha = false;
  for(size_t i = 3; i < img.px.size() && !alpha; i += 4)
    alpha = img.px[i] != 255;
  std::vector<uint8_t> png;
  const auto           sink = [](void* ctx, void* data, int size) {
    auto* out = static_cast<std::vector<uint8_t>*>(ctx);
    out->insert(out->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
  };
  if(alpha)
    stbi_write_png_to_func(sink, &png, img.w, img.h, 4, img.px.data(), img.w * 4);
  else
  {
    std::vector<uint8_t> rgb(size_t(img.w) * size_t(img.h) * 3);
    for(size_t i = 0; i < size_t(img.w) * size_t(img.h); i++)
      std::memcpy(&rgb[i * 3], &img.px[i * 4], 3);
    stbi_write_png_to_func(sink, &png, img.w, img.h, 3, rgb.data(), img.w * 3);
  }
  if(png.empty())
  {
    if(why)
      *why = "the PNG for " + req.rel + " could not be written";
    return std::nullopt;
  }
  return png;
}

namespace {

// A 1 x 1 stand-in for an image that could not be made: white, or a flat normal.
std::vector<uint8_t> neutralImage(const TextureRequest& req)
{
  const bool normal = req.channels == "ng" || req.channels == "bn" || req.normal;
  const unsigned char px[3] = {normal ? uint8_t(128) : uint8_t(255), normal ? uint8_t(128) : uint8_t(255), 255};
  std::vector<uint8_t> png;
  stbi_write_png_to_func([](void* ctx, void* data, int size) { static_cast<std::vector<uint8_t>*>(ctx)->insert(static_cast<std::vector<uint8_t>*>(ctx)->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size); }, &png, 1, 1, 3, px, 3);
  return png;
}

}  // namespace

GlbExport prepareGlb(const SceneGraph& scene, int root)
{
  GlbExport out;
  out.requests = textureRequests(scene, mat::SkinMode::eRaster);
  BuildOptions options;
  options.texturesFlipped = true;
  options.skin            = mat::SkinMode::eRaster;
  options.forFile         = true;
  options.texture         = [](const TextureRequest& req) -> std::optional<TextureImage> { return TextureImage{"fv-image:" + requestKey(req), "image/png", true}; };
  out.model               = buildGltf(scene, root, options).model;
  return out;
}

std::vector<std::string> finishGlb(GlbExport& job, const std::filesystem::path& out, const ExportOptions& o)
{
  std::vector<std::string> notes;
  tinygltf::Model&         model = job.model;
  std::map<std::string, const TextureRequest*> byKey;
  for(const TextureRequest& r : job.requests)
    byKey[requestKey(r)] = &r;
  if(model.buffers.empty())
    model.buffers.emplace_back();
  auto& data = model.buffers[0].data;
  for(tinygltf::Image& img : model.images)
  {
    if(img.uri.rfind("fv-image:", 0) != 0)
      continue;
    const auto req = byKey.find(img.uri.substr(9));
    if(req == byKey.end())
      continue;
    std::string why;
    auto        png = composeImage(*req->second, o, &why);
    if(!png)
    {
      notes.push_back(why);
      png = neutralImage(*req->second);
    }
    while(data.size() % 4)
      data.push_back(0);
    tinygltf::BufferView view;
    view.buffer     = 0;
    view.byteOffset = data.size();
    view.byteLength = png->size();
    data.insert(data.end(), png->begin(), png->end());
    model.bufferViews.push_back(view);
    img.uri.clear();
    img.bufferView = int(model.bufferViews.size()) - 1;
    img.mimeType   = "image/png";
  }
  // a file for other tools has no use for KTX2 and the like
  model.extensionsRequired.clear();
  model.extensionsUsed.erase(std::remove_if(model.extensionsUsed.begin(), model.extensionsUsed.end(),
                                            [](const std::string& e) { return e == "KHR_texture_basisu"; }),
                             model.extensionsUsed.end());
  // the images no texture reaches by KTX2 any more
  tinygltf::TinyGLTF writer;
  if(!writer.WriteGltfSceneToFile(&model, out.string(), false, true, false, true))
    throw std::runtime_error("the GLB could not be written to " + out.string());
  return notes;
}

}  // namespace fv::build
