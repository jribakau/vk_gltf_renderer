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
  vec4 SampleLevel(const SamplerState&, vec2, float) const { return colour; }
  vec4 SampleGrad(const SamplerState&, vec2, vec2, vec2) const { return colour; }
};
inline int NonUniformResourceIndex(int i)
{
  return i;
}
#include "../../shaders/gltf_material_eval.h.slang"

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

std::vector<ClosureAt> probeClosureBytes(const void* rowBytes, size_t rowSize, const void* infoBytes, size_t infoSize, size_t infoCount, const ClosureOptions& options)
{
  if(rowSize != sizeof(GltfShadeMaterial) || infoSize != sizeof(GltfTextureInfo))
    throw std::logic_error("probeClosure: the row isn't the shader's");
  static_assert(kLobeCount == LOBE_COUNT);

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
  for(const uint16_t normal : {uint16_t(row.normalTexture), uint16_t(row.clearcoatNormalTexture), uint16_t(row.detailNormalTexture)})
    if(normal > 0 && normal < infos.size() && infos[normal].index >= 0)
      imageArray[static_cast<size_t>(infos[normal].index)].colour = vec4(0.5f, 0.5f, 1.0f, 1.0f);

  MeshState state;
  state.N     = vec3(0.0f, 0.0f, 1.0f);
  state.T     = vec3(1.0f, 0.0f, 0.0f);
  state.B     = vec3(0.0f, 1.0f, 0.0f);
  state.Ng    = state.N;
  state.tc[0] = vec2(0.5f);
  state.tc[1] = vec2(0.5f);
  const PbrMaterial mat = evaluateMaterial(row, state, imageArray.data(), samplerArray.data(), infos.data(), options.thinScatter);

  std::vector<ClosureAt> out;
  for(const float cosine : options.cosines)
  {
    ClosureAt at;
    at.cosine     = cosine;
    const vec3 k1 = vec3(std::sqrt(std::max(0.0f, 1.0f - cosine * cosine)), 0.0f, cosine);
    vec3       tint = mat.baseColor;
    const auto weights = computeLobeWeights(mat, dot(k1, mat.N), tint);
    std::mt19937                          rng(41);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for(int lobe = 0; lobe < LOBE_COUNT; lobe++)
    {
      at.lobes[lobe].weight = weights[lobe];
      if(weights[lobe] <= 0.0f)
        continue;
      dvec3 carried(0.0);
      for(int i = 0; i < options.samples; i++)
      {
        BsdfSampleData s;
        s.k1 = k1;
        s.xi = vec3(u(rng), u(rng), u(rng));
        bsdfSampleLobe(s, mat, lobe, tint);
        if(s.event_type != BSDF_EVENT_ABSORB)
          carried += dvec3(s.bsdf_over_pdf);
      }
      for(int c = 0; c < 3; c++)
      {
        at.lobes[lobe].albedo[c] = float(carried[c] / options.samples);
        at.total[c] += weights[lobe] * at.lobes[lobe].albedo[c];
      }
    }
    out.push_back(at);
  }
  return out;
}

}  // namespace fv::probe
