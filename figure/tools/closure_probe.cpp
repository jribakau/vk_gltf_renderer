// See closure_probe.hpp. The shader's headers as C++, as tests/test_bsdf_lobes.cpp compiles the BSDF.

#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace glm;
#include <nvshaders/slang_types.h>
namespace shaderio {
inline vec3     lerp(vec3 a, vec3 b, float t) { return glm::mix(a, b, t); }
inline uint32_t asuint(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float asfloat(uint32_t u)
{
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline vec3  max(vec3 a, vec3 b) { return glm::max(a, b); }
inline float max(float a, double b) { return glm::max(a, float(b)); }
// Slang's literals are doubles that meet a vector
inline vec3  operator+(double a, vec3 b) { return float(a) + b; }
inline vec3  operator-(double a, vec3 b) { return float(a) - b; }
inline vec3  operator*(double a, vec3 b) { return float(a) * b; }
inline vec3  operator/(double a, vec3 b) { return float(a) / b; }
inline bvec3 operator>(vec3 a, double b) { return greaterThan(a, vec3(float(b))); }
// A vector times a matrix: Slang's rows are what the host's GLM matrices hold as columns
inline vec2  mul(vec3 v, mat3x2 m) { return v.x * m[0] + v.y * m[1] + v.z * m[2]; }
inline vec3  mul(vec3 v, mat3 m) { return v.x * m[0] + v.y * m[1] + v.z * m[2]; }
}  // namespace shaderio
using namespace shaderio;
#include <nvshaders/bsdf_functions.h.slang>

// What the evaluator asks of Slang: its texture arrays, each image one colour
struct SamplerState
{
};
struct Texture2D
{
  vec4 colour{1.0f};
  vec4 mean{1.0f};  // the last mip's
  vec4 SampleLevel(const SamplerState&, vec2, float lod) const { return lod >= 1.0f ? mean : colour; }
  vec4 SampleGrad(const SamplerState&, vec2, vec2, vec2) const { return colour; }
};
inline int NonUniformResourceIndex(int i)
{
  return i;
}
#include "../../shaders/gltf_material_eval.h.slang"
#include "../../shaders/debug_colors.h.slang"

#include "closure_probe.hpp"

namespace fv::probe {

const char* lobeName(int lobe)
{
  switch(lobe)
  {
    case LOBE_DIFFUSE_REFLECTION:
      return "diffuse";
    case LOBE_SPECULAR_TRANSMISSION:
      return "transmission";
    case LOBE_SPECULAR_REFLECTION:
      return "specular";
    case LOBE_METAL_REFLECTION:
      return "metal";
    case LOBE_SHEEN_REFLECTION:
      return "sheen";
    case LOBE_CLEARCOAT_REFLECTION:
      return "clearcoat";
    case LOBE_DIFFUSE_TRANSMISSION:
      return "diffuseTransmission";
    case LOBE_SPECULAR2_REFLECTION:
      return "specular2";
    default:
      return "?";
  }
}

namespace {

PbrMaterial evaluateBytes(const void* rowBytes, size_t rowSize, const void* infoBytes, size_t infoSize, size_t infoCount, const ClosureOptions& options)
{
  if(rowSize != sizeof(GltfShadeMaterial) || infoSize != sizeof(GltfTextureInfo))
    throw std::logic_error("probeClosure: the row isn't the shader's");

  GltfShadeMaterial row;
  std::memcpy(static_cast<void*>(&row), rowBytes, sizeof(row));
  std::vector<GltfTextureInfo> infos(std::max<size_t>(infoCount, 1));
  if(infoCount > 0)
    std::memcpy(static_cast<void*>(infos.data()), infoBytes, infoCount * sizeof(GltfTextureInfo));

  // Every image the infos name: white, and the normal maps flat
  int images = 1, samplers = 1;
  for(const GltfTextureInfo& info : infos)
  {
    images   = std::max(images, int(info.index) + 1);
    samplers = std::max(samplers, int(info.samplerIndex) + 1);
  }
  std::vector<Texture2D>    imageArray(static_cast<size_t>(images));
  std::vector<SamplerState> samplerArray(static_cast<size_t>(samplers));
  for(const uint16_t normal : {uint16_t(row.normalTexture), uint16_t(row.clearcoatNormalTexture), uint16_t(row.detailNormalTexture),
                               uint16_t(row.proceduralNormalTexture)})
    if(normal > 0 && normal < infos.size() && infos[normal].index >= 0)
      imageArray[static_cast<size_t>(infos[normal].index)].colour = imageArray[static_cast<size_t>(infos[normal].index)].mean = vec4(0.5f, 0.5f, 1.0f, 1.0f);
  for(const ProbeImage& image : options.images)
  {
    if(image.texture < 0 || image.texture >= images)
      throw std::logic_error("probeClosure: an image no texture info names");
    Texture2D& t = imageArray[static_cast<size_t>(image.texture)];
    t.colour     = vec4(image.colour[0], image.colour[1], image.colour[2], image.colour[3]);
    t.mean       = vec4(image.mean[0], image.mean[1], image.mean[2], image.mean[3]);
  }

  MeshState state;
  state.N     = vec3(0.0f, 0.0f, 1.0f);
  state.T     = vec3(1.0f, 0.0f, 0.0f);
  state.B     = vec3(0.0f, 1.0f, 0.0f);
  state.Ng    = state.N;
  state.tc[0]   = vec2(0.5f);
  state.tc[1]   = vec2(0.5f);
  state.texGrad = options.texGrad;
  state.masks   = vec4(options.masks[0], options.masks[1], options.masks[2], options.masks[3]);
  return evaluateMaterial(row, state, imageArray.data(), samplerArray.data(), infos.data(), options.thinScatter);
}

// What one lobe carries of the light it gets, seen from k1: the mean of its samples' bsdf_over_pdf
dvec3 lobeAlbedo(const PbrMaterial& mat, vec3 k1, int lobe, vec3 tint, int samples, std::mt19937& rng)
{
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  dvec3                                 carried(0.0);
  for(int i = 0; i < samples; i++)
  {
    BsdfSampleData s;
    s.k1 = k1;
    s.xi = vec3(u(rng), u(rng), u(rng));
    bsdfSampleLobe(s, mat, lobe, tint);
    if(s.event_type != BSDF_EVENT_ABSORB)
      carried += dvec3(s.bsdf_over_pdf);
  }
  return carried / double(samples);
}

// The surface's albedo seen at `cosine` from the normal: every lobe's by its weight
vec3 surfaceAlbedo(const PbrMaterial& mat, float cosine, int samples)
{
  const vec3   k1      = vec3(std::sqrt(std::max(0.0f, 1.0f - cosine * cosine)), 0.0f, cosine);
  vec3         tint    = mat.baseColor;
  const auto   weights = computeLobeWeights(mat, dot(k1, mat.N), tint);
  std::mt19937 rng(41);
  dvec3        total(0.0);
  for(int lobe = 0; lobe < LOBE_COUNT; lobe++)
    if(weights[lobe] > 0.0f)
      total += double(weights[lobe]) * lobeAlbedo(mat, k1, lobe, tint, samples, rng);
  return vec3(total);
}

// The BSDF's value toward k2 (the cosine there included, as it returns it) of every lobe, or of `only`
// (>= 0): the sum next-event estimation takes, its occlusion and face checks included, with the other
// lobes' weights 0
vec3 evaluateWeighted(const PbrMaterial& mat, vec3 k1, vec3 k2, int only)
{
  vec3 tint = mat.baseColor;
  auto s    = computeLobeWeights(mat, dot(k1, mat.N), tint);
  auto P    = bsdfLobePick(mat, s, tint);
  if(only >= 0)
    for(int lobe = 0; lobe < LOBE_COUNT; lobe++)
      if(lobe != only)
        s[lobe] = 0.0f;
  BsdfEvaluateData d;
  d.k1 = k1;
  d.k2 = k2;
  d.xi = vec3(0.5f);
  bsdfEvaluateSumWeighted(d, mat, s, P, tint, -1);
  return d.bsdf_diffuse + d.bsdf_glossy;
}

ViewImage blank(int w, int h, float value)
{
  ViewImage image;
  image.width  = w;
  image.height = h;
  image.rgb.assign(size_t(w) * h * 3, value);
  return image;
}

void put(ViewImage& image, int x, int y, vec3 v)
{
  float* p = image.at(x, y);
  p[0]     = v.x;
  p[1]     = v.y;
  p[2]     = v.z;
}

unsigned char srgbByte(float linear)
{
  const float c = std::clamp(linear, 0.0f, 1.0f);
  const float e = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
  return static_cast<unsigned char>(std::lround(e * 255.0f));
}

}  // namespace

LobeRoughness probeRoughnessBytes(const void* rowBytes, size_t rowSize, const void* infoBytes, size_t infoSize, size_t infoCount, const ClosureOptions& options)
{
  const PbrMaterial mat = evaluateBytes(rowBytes, rowSize, infoBytes, infoSize, infoCount, options);
  LobeRoughness     out;
  out.alpha[0]  = mat.roughness.x;
  out.alpha[1]  = mat.roughness.y;
  out.clearcoat = mat.clearcoatRoughness;
  out.specular2 = mat.specular2Roughness;
  return out;
}

std::vector<ClosureAt> probeClosureBytes(const void* rowBytes, size_t rowSize, const void* infoBytes, size_t infoSize, size_t infoCount, const ClosureOptions& options)
{
  static_assert(kLobeCount == LOBE_COUNT);
  const PbrMaterial mat = evaluateBytes(rowBytes, rowSize, infoBytes, infoSize, infoCount, options);

  std::vector<ClosureAt> out;
  for(const float cosine : options.cosines)
  {
    ClosureAt at;
    at.cosine     = cosine;
    const vec3 k1 = vec3(std::sqrt(std::max(0.0f, 1.0f - cosine * cosine)), 0.0f, cosine);
    vec3       tint = mat.baseColor;
    const auto weights = computeLobeWeights(mat, dot(k1, mat.N), tint);
    std::mt19937 rng(41);
    for(int lobe = 0; lobe < LOBE_COUNT; lobe++)
    {
      at.lobes[lobe].weight = weights[lobe];
      if(weights[lobe] <= 0.0f)
        continue;
      const dvec3 carried = lobeAlbedo(mat, k1, lobe, tint, options.samples, rng);
      for(int c = 0; c < 3; c++)
      {
        at.lobes[lobe].albedo[c] = float(carried[c]);
        at.total[c] += weights[lobe] * at.lobes[lobe].albedo[c];
      }
    }
    out.push_back(at);
  }
  return out;
}

MaterialViews probeViewsBytes(const void* rowBytes, size_t rowSize, const void* infoBytes, size_t infoSize, size_t infoCount,
                              const ClosureOptions& options, const ViewOptions& views)
{
  const PbrMaterial mat = evaluateBytes(rowBytes, rowSize, infoBytes, infoSize, infoCount, options);
  MaterialViews     out;

  // The slices: the half vector tilted by theta_h in the x-z plane, the view and the light theta_d
  // either side of it across that plane (phi_d 90 degrees)
  const int n = views.sliceSize;
  for(int lobe = 0; lobe < LOBE_COUNT; lobe++)
    out.slice[lobe] = blank(n, n, 0.0f);
  out.sliceTotal = blank(n, n, 0.0f);
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      float thetaH = 0, thetaD = 0;
      sliceAngles(x, y, n, thetaH, thetaD);
      const float sh = std::sin(thetaH), ch = std::cos(thetaH), sd = std::sin(thetaD), cd = std::cos(thetaD);
      const vec3  view(cd * sh, sd, cd * ch);
      const vec3  light(cd * sh, -sd, cd * ch);
      const float cosine = std::max(light.z, 1e-6f);  // the BSDF's value carries it; the slice is f
      put(out.sliceTotal, x, y, evaluateWeighted(mat, view, light, -1) / cosine);
      for(int lobe = 0; lobe < LOBE_COUNT; lobe++)
      {
        const vec3 f = evaluateWeighted(mat, view, light, lobe) / cosine;
        put(out.slice[lobe], x, y, f);
        if(f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)
          out.lobeUsed[lobe] = true;
      }
    }

  // The furnace ball: each point's radiance is the albedo at its view angle (an isotropic material's
  // depends on nothing else), from a table over the angle, linear between its entries
  const int         cosines = std::max(views.furnaceCosines, 2);
  std::vector<vec3> albedo(static_cast<size_t>(cosines));
  for(int i = 0; i < cosines; i++)
    albedo[static_cast<size_t>(i)] = surfaceAlbedo(mat, std::max(float(i) / float(cosines - 1), 0.01f), views.furnaceSamples);
  const int       b      = views.ballSize;
  const float     r      = 0.5f * float(b);
  const vec3      toward = normalize(vec3(views.light[0], views.light[1], views.light[2]));
  constexpr float kPi    = 3.14159265358979f;
  out.furnace            = blank(b, b, 1.0f);
  out.lit                = blank(b, b, kLitBackdrop);
  for(int y = 0; y < b; y++)
    for(int x = 0; x < b; x++)
    {
      const float nx = (float(x) + 0.5f - r) / r, ny = (r - float(y) - 0.5f) / r;
      const float rr = nx * nx + ny * ny;
      if(rr >= 1.0f)
        continue;
      const vec3  normal(nx, ny, std::sqrt(1.0f - rr));
      const float t = normal.z * float(cosines - 1);
      const int   i = std::min(int(t), cosines - 2);
      put(out.furnace, x, y, glm::mix(albedo[static_cast<size_t>(i)], albedo[static_cast<size_t>(i + 1)], t - float(i)));

      // the view (+z) and the light in the point's frame, its normal the frame's z
      const vec3 helper    = std::abs(normal.y) < 0.99f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
      const vec3 tangent   = normalize(cross(helper, normal));
      const vec3 bitangent = cross(normal, tangent);
      const auto local     = [&](vec3 v) { return vec3(dot(v, tangent), dot(v, bitangent), dot(v, normal)); };
      const vec3 k2        = local(toward);
      if(k2.z <= 0.0f)
      {
        put(out.lit, x, y, vec3(0.0f));  // the side the light doesn't reach
        continue;
      }
      put(out.lit, x, y, evaluateWeighted(mat, local(vec3(0.0f, 0.0f, 1.0f)), k2, -1) * kPi);
    }
  return out;
}

std::vector<unsigned char> ballPixels(const ViewImage& image)
{
  std::vector<unsigned char> px(image.rgb.size());
  for(size_t i = 0; i < px.size(); i++)
    px[i] = srgbByte(image.rgb[i]);
  return px;
}

std::vector<unsigned char> slicePixels(const ViewImage& image)
{
  std::vector<unsigned char> px(image.rgb.size());
  for(size_t i = 0; i + 2 < image.rgb.size(); i += 3)
  {
    // the luminance's place on the fixed log scale; the colour's own channels are in the float file
    const float lum = 0.2126f * image.rgb[i] + 0.7152f * image.rgb[i + 1] + 0.0722f * image.rgb[i + 2];
    vec3        c(0.0f);
    if(lum > 0.0f)
      c = viridis((std::log10(lum) - kSliceLogMin) / (kSliceLogMax - kSliceLogMin));
    for(int k = 0; k < 3; k++)
      px[i + k] = static_cast<unsigned char>(std::lround(std::clamp(c[k], 0.0f, 1.0f) * 255.0f));
  }
  return px;
}

}  // namespace fv::probe
