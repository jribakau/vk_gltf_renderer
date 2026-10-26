// Tangents made side by side (a mesh with many faces): every vertex's sum and handedness are the ones
// the serial loop made, bit for bit, on a mesh with shared vertices, a corner used twice in a face,
// degenerate UVs and vertices no face touches.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "tinygltf_utils.hpp"
#include "nvshaders/functions.h.slang"

namespace {

struct Soup
{
  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec2> uvs;
  std::vector<uint32_t>  indices;
};

Soup soup(uint32_t vertices, uint32_t faces, uint32_t seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::uniform_int_distribution<uint32_t> pick(0, vertices - 2);  // the last vertex is in no face
  Soup s;
  for(uint32_t v = 0; v < vertices; v++)
  {
    s.positions.push_back({u(rng), u(rng), u(rng)});
    s.normals.push_back(glm::normalize(glm::vec3(u(rng), u(rng), 1.5f)));
    s.uvs.push_back({u(rng), u(rng)});
  }
  for(uint32_t f = 0; f < faces; f++)
  {
    const uint32_t a = pick(rng), b = pick(rng), c = f % 97 == 0 ? a : pick(rng);  // some use a corner twice
    s.indices.insert(s.indices.end(), {a, b, c});
  }
  for(uint32_t v = 0; v < 50; v++)
    s.uvs[v] = s.uvs[v + 50];  // and some have UVs that coincide
  return s;
}

// The serial loop as it was (tinygltf_utils.cpp, simpleCreateTangents with UVs and normals).
std::vector<glm::vec4> serialTangents(const Soup& s)
{
  std::vector<glm::vec4> tangents(s.positions.size(), glm::vec4(0));
  for(size_t i = 0; i < s.indices.size() / 3; i++)
  {
    const uint32_t i0 = s.indices[i * 3 + 0], i1 = s.indices[i * 3 + 1], i2 = s.indices[i * 3 + 2];
    const glm::vec3& p0 = s.positions[i0];
    const glm::vec3& p1 = s.positions[i1];
    const glm::vec3& p2 = s.positions[i2];
    glm::vec4&       t0 = tangents[i0];
    glm::vec4&       t1 = tangents[i1];
    glm::vec4&       t2 = tangents[i2];
    const glm::vec3  n0 = s.normals[i0];
    const glm::vec2& uv0 = s.uvs[i0];
    const glm::vec2& uv1 = s.uvs[i1];
    const glm::vec2& uv2 = s.uvs[i2];
    glm::vec3        edge1    = p1 - p0;
    glm::vec3        edge2    = p2 - p0;
    glm::vec2        deltaUV1 = uv1 - uv0;
    glm::vec2        deltaUV2 = uv2 - uv0;
    float            f        = 1.0F;
    float            a        = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
    if(fabs(a) > 0.0F)
      f = 1.0f / a;
    glm::vec3 tangent    = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
    glm::vec3 bitangent  = f * (deltaUV2.x * edge1 - deltaUV1.x * edge2);
    float     handedness = (glm::dot(glm::cross(tangent, bitangent), n0) > 0.0F) ? 1.0F : -1.0F;
    t0                   = glm::vec4(tangent + glm::vec3(t0), handedness);
    t1                   = glm::vec4(tangent + glm::vec3(t1), handedness);
    t2                   = glm::vec4(tangent + glm::vec3(t2), handedness);
  }
  for(size_t v = 0; v < tangents.size(); v++)
  {
    glm::vec4& t0  = tangents[v];
    glm::vec3  n0  = s.normals[v];
    glm::vec3  ot0 = glm::normalize(glm::vec3(t0) - (glm::dot(n0, glm::vec3(t0)) * n0));
    if(glm::length2(ot0) < 0.1F || glm::any(glm::isnan(ot0)))
      ot0 = glm::vec3(shaderio::makeFastTangent(n0));
    t0 = glm::vec4(ot0, t0.w);
  }
  return tangents;
}

// The soup as a glTF primitive, its tangents made by tinygltf::utils.
std::vector<glm::vec4> utilsTangents(const Soup& s)
{
  tinygltf::Model model;
  model.buffers.emplace_back();
  tinygltf::Primitive p;
  p.attributes["POSITION"]   = tinygltf::utils::appendAccessor(model, 0, s.positions.data(), s.positions.size() * 12, TINYGLTF_COMPONENT_TYPE_FLOAT,
                                                                 TINYGLTF_TYPE_VEC3, s.positions.size());
  p.attributes["NORMAL"]     = tinygltf::utils::appendAccessor(model, 0, s.normals.data(), s.normals.size() * 12, TINYGLTF_COMPONENT_TYPE_FLOAT,
                                                               TINYGLTF_TYPE_VEC3, s.normals.size());
  p.attributes["TEXCOORD_0"] = tinygltf::utils::appendAccessor(model, 0, s.uvs.data(), s.uvs.size() * 8, TINYGLTF_COMPONENT_TYPE_FLOAT,
                                                                   TINYGLTF_TYPE_VEC2, s.uvs.size());
  p.indices = tinygltf::utils::appendAccessor(model, 0, s.indices.data(), s.indices.size() * 4, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,
                                              TINYGLTF_TYPE_SCALAR, s.indices.size());
  tinygltf::utils::createTangentAttribute(model, p);
  tinygltf::utils::simpleCreateTangents(model, p);
  const tinygltf::Accessor&   a = model.accessors[size_t(p.attributes["TANGENT"])];
  const tinygltf::BufferView& v = model.bufferViews[size_t(a.bufferView)];
  std::vector<glm::vec4>      out(a.count);
  std::memcpy(out.data(), model.buffers[0].data.data() + v.byteOffset, a.count * sizeof(glm::vec4));
  return out;
}

}  // namespace

TEST(Tangents, MadeSideBySideTheyAreTheSerialOnesBitForBit)
{
  // above the side-by-side threshold, and below it (the loop itself)
  for(const auto& [vertices, faces] : {std::pair<uint32_t, uint32_t>{30000, 60000}, {500, 900}})
  {
    const Soup s    = soup(vertices, faces, vertices);
    const auto want = serialTangents(s);
    const auto got  = utilsTangents(s);
    ASSERT_EQ(got.size(), want.size());
    EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size() * sizeof(glm::vec4)), 0) << faces << " faces";
  }
}
