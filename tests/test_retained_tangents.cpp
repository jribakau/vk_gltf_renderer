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

// Tangents that follow moved vertices (daz-viewer docs/retained-scene-plan.md, oracle 7): remade
// over the one-ring of the vertices that moved, they are the ones a full computeTangents makes, bit
// for bit, edit after edit. On seeded meshes with shared vertices, UV seams, degenerate UVs,
// zero-area faces, a corner used twice, NaNs, vertices no face touches, and submeshes of which only
// some need tangents; with and without UVs and normals; and on the Box Figure's body under a dial.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <glm/glm.hpp>

#include <figure/figure.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>

#include "tinygltf_utils.hpp"

namespace {

using tinygltf::utils::computeTangents;
using tinygltf::utils::tangentAdjacency;
using tinygltf::utils::updateTangents;

struct Mesh
{
  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec2> uvs;
  std::vector<uint32_t>  indices;  // the triangles of every submesh that needs tangents
  std::vector<uint32_t>  ignored;  // and of those that don't: their vertices get no faces
  uint32_t               isolated    = 0;  // a vertex no face touches
  uint32_t               untangented = 0;  // one only the submesh without tangents touches
};

// A wavy grid, its triangles shuffled and turned, cut by a UV seam (the vertices along it doubled,
// the faces past it on the doubles), with faces that use a corner twice, faces on three points of a
// line, faces whose UVs coincide, and vertices no face touches. Some faces go to a submesh that
// needs no tangents.
Mesh grid(uint32_t width, uint32_t height, uint32_t seed, bool nan)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  Mesh                                  m;
  const auto at = [&](uint32_t x, uint32_t y) { return y * width + x; };
  for(uint32_t y = 0; y < height; y++)
    for(uint32_t x = 0; x < width; x++)
    {
      m.positions.push_back({float(x) + 0.3f * u(rng), float(y) + 0.3f * u(rng), std::sin(float(x + y) * 0.4f) + 0.2f * u(rng)});
      m.normals.push_back(glm::normalize(glm::vec3(u(rng), u(rng), 1.5f)));
      m.uvs.push_back({float(x) / float(width) + 0.01f * u(rng), float(y) / float(height) + 0.01f * u(rng)});
    }
  // the seam: a column of doubles, the same point and normal under another UV
  const uint32_t        seamX = width / 2;
  std::vector<uint32_t> twin(height);
  for(uint32_t y = 0; y < height; y++)
  {
    twin[y] = uint32_t(m.positions.size());
    m.positions.push_back(m.positions[at(seamX, y)]);
    m.normals.push_back(m.normals[at(seamX, y)]);
    m.uvs.push_back(m.uvs[at(seamX, y)] + glm::vec2(1.f, 0.f));
  }
  std::vector<std::array<uint32_t, 3>> faces;
  for(uint32_t y = 0; y + 1 < height; y++)
    for(uint32_t x = 0; x + 1 < width; x++)
    {
      // a cell right of the seam reaches its left corners through the doubles
      const auto v = [&](uint32_t cx, uint32_t cy) { return x >= seamX && cx == seamX ? twin[cy] : at(cx, cy); };
      faces.push_back({v(x, y), v(x + 1, y), v(x + 1, y + 1)});
      faces.push_back({v(x, y), v(x + 1, y + 1), v(x, y + 1)});
    }
  std::uniform_int_distribution<uint32_t> pick(0, width * height - 1);
  for(uint32_t k = 0; k < width; k++)
  {
    const uint32_t a = pick(rng), b = pick(rng);
    faces.push_back({a, a, b});  // a corner used twice: no area, no UV area
    // a point on the line between two others: no area
    const uint32_t mid = uint32_t(m.positions.size());
    m.positions.push_back(0.5f * (m.positions[a] + m.positions[b]));
    m.normals.push_back(m.normals[a]);
    m.uvs.push_back({u(rng), u(rng)});
    faces.push_back({a, mid, b});
  }
  for(uint32_t k = 0; k < width / 2; k++)
  {
    const uint32_t a = pick(rng), b = pick(rng), c = pick(rng);
    m.uvs[b] = m.uvs[c] = m.uvs[a];  // three corners on one UV
    faces.push_back({a, b, c});
  }
  m.isolated = uint32_t(m.positions.size()) + 1;
  for(uint32_t k = 0; k < 5; k++)  // vertices no face touches, but for the first: the submesh without tangents
  {
    m.positions.push_back({u(rng), u(rng), u(rng)});
    m.normals.push_back(glm::normalize(glm::vec3(u(rng), u(rng), 1.5f)));
    m.uvs.push_back({u(rng), u(rng)});
  }
  m.untangented = m.isolated - 1;
  if(nan)
  {
    m.positions[pick(rng)].x = std::numeric_limits<float>::quiet_NaN();
    m.uvs[pick(rng)].y       = std::numeric_limits<float>::quiet_NaN();
  }
  std::shuffle(faces.begin(), faces.end(), rng);
  // three submeshes; the middle one needs no tangents
  for(size_t f = 0; f < faces.size(); f++)
  {
    std::array<uint32_t, 3> t = faces[f];
    std::rotate(t.begin(), t.begin() + rng() % 3, t.end());
    std::vector<uint32_t>& into = f >= faces.size() / 3 && f < faces.size() / 2 ? m.ignored : m.indices;
    into.insert(into.end(), t.begin(), t.end());
  }
  m.ignored.insert(m.ignored.end(), {m.untangented, faces[0][0], faces[0][1]});
  return m;
}

std::vector<glm::vec4> full(const Mesh& m, bool uvs, bool normals)
{
  std::vector<glm::vec4> t(m.positions.size(), glm::vec4(0.f));
  computeTangents(m.positions, normals ? std::span<const glm::vec3>(m.normals) : std::span<const glm::vec3>(),
                  uvs ? std::span<const glm::vec2>(m.uvs) : std::span<const glm::vec2>(), m.indices, t);
  return t;
}

// The vertices whose tangents a move of `changed` reaches, worked out from the triangles directly.
std::set<uint32_t> oneRing(const Mesh& m, const std::vector<uint32_t>& changed)
{
  std::set<uint32_t> out;
  for(uint32_t v : changed)
    if(v < m.positions.size())
      out.insert(v);
  const std::set<uint32_t> moved(changed.begin(), changed.end());
  for(size_t i = 0; i + 2 < m.indices.size(); i += 3)
    if(moved.count(m.indices[i]) || moved.count(m.indices[i + 1]) || moved.count(m.indices[i + 2]))
      out.insert({m.indices[i], m.indices[i + 1], m.indices[i + 2]});
  return out;
}

bool same(const glm::vec4& a, const glm::vec4& b)
{
  return std::memcmp(&a, &b, sizeof(glm::vec4)) == 0;
}

}  // namespace

TEST(RetainedTangents, TheAdjacencyListsEachVertexsCornersInOrder)
{
  const std::vector<uint32_t> indices = {0, 1, 2, 2, 1, 3, 3, 3, 0, 4};  // a corner used twice, a trailing index
  const auto                  a       = tangentAdjacency(indices, 6);
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(a->start, (std::vector<uint32_t>{0, 2, 4, 6, 9, 9, 9}));
  EXPECT_EQ(a->corners, (std::vector<uint32_t>{0, 8, 1, 4, 2, 3, 5, 6, 7}));
  // an index past the vertices: computeTangents can't make those either
  EXPECT_FALSE(tangentAdjacency(indices, 3).has_value());
  // the trailing index is no face's, so it may be past them
  EXPECT_TRUE(tangentAdjacency(std::vector<uint32_t>{0, 1, 2, 9}, 3).has_value());

  // an adjacency of other vertices or other triangles is refused, nothing written
  const std::vector<glm::vec3> positions(5, glm::vec3(1.f));
  std::vector<glm::vec4>       tangents(5, glm::vec4(7.f));
  const std::vector<uint32_t>  changed = {0};
  EXPECT_FALSE(updateTangents(positions, {}, {}, indices, *tangentAdjacency(indices, 6), changed, tangents));
  EXPECT_FALSE(updateTangents(positions, {}, {}, std::span(indices).first(6), *tangentAdjacency(indices, 5), changed, tangents));
  EXPECT_EQ(tangents, std::vector<glm::vec4>(5, glm::vec4(7.f)));
}

// Edits applied one after another to tangents remade incrementally, each compared with a full
// recompute; and each remake writes the one-ring and nothing else.
TEST(RetainedTangents, TheOneRingOfMovedVerticesIsAFullRecomputeBitForBit)
{
  struct Size
  {
    uint32_t width, height;
  };
  // under computeTangents' side-by-side threshold (its loop) and over it (about 18000 faces need
  // tangents, with UVs and normals)
  for(const Size size : {Size{14, 11}, Size{110, 100}})
    for(uint32_t seed = 1; seed <= 3; seed++)
      for(const bool uvs : {true, false})
        for(const bool normals : {true, false})
        {
          // NaNs in a position and a UV: with normals they end in the fallback tangent. Without,
          // the face normal takes them to the output, where the default NaN of a face with no
          // area may meet them, and which one's bits come out is the order of an addition, the
          // compiler's to choose.
          const bool nan = normals && seed == 2;
          SCOPED_TRACE(testing::Message() << size.width << "x" << size.height << " seed " << seed << (uvs ? " uv" : "")
                                          << (normals ? " normal" : "") << (nan ? " nan" : ""));
          Mesh       m     = grid(size.width, size.height, seed, nan);
          const auto ring  = tangentAdjacency(m.indices, m.positions.size());
          ASSERT_TRUE(ring.has_value());
          std::vector<glm::vec4> tangents = full(m, uvs, normals);

          std::mt19937                            rng(seed * 7919);
          std::uniform_real_distribution<float>   u(-1.f, 1.f);
          std::uniform_int_distribution<uint32_t> pick(0, uint32_t(m.positions.size()) - 1);
          const uint32_t                          width = size.width;
          for(int step = 0; step < 8; step++)
          {
            // which vertices move: one, a patch around one, every vertex, none, one no face touches,
            // one only a submesh without tangents uses, then scattered ones and the patch again
            std::vector<uint32_t> changed;
            switch(step)
            {
              case 0:
                changed = {pick(rng)};
                break;
              case 1:
              case 7: {
                const uint32_t c = pick(rng) % (width * size.height);
                for(int dy = -2; dy <= 2; dy++)
                  for(int dx = -2; dx <= 2; dx++)
                    if(const int64_t v = int64_t(c) + dy * int64_t(width) + dx; v >= 0 && v < int64_t(width * size.height))
                      changed.push_back(uint32_t(v));
                break;
              }
              case 2:
                for(uint32_t v = 0; v < m.positions.size(); v++)
                  changed.push_back(v);
                break;
              case 3:
                break;
              case 4:
                changed = {m.isolated};
                break;
              case 5:
                changed = {m.untangented};
                break;
              case 6:
                for(int k = 0; k < 40; k++)
                  changed.push_back(pick(rng));
                changed.push_back(uint32_t(m.positions.size()) + 3);  // past the vertices: left alone
                break;
            }
            for(uint32_t v : changed)
            {
              if(v >= m.positions.size())
                continue;
              // moved, sometimes onto another vertex (no area) or onto its UV (no UV area)
              const uint32_t other = pick(rng);
              switch(rng() % 4)
              {
                case 0:
                  m.positions[v] = m.positions[other];
                  break;
                case 1:
                  m.uvs[v] = m.uvs[other];
                  break;
                default:
                  m.positions[v] += 0.2f * glm::vec3(u(rng), u(rng), u(rng));
                  m.uvs[v] += 0.01f * glm::vec2(u(rng), u(rng));
                  break;
              }
              m.normals[v] = glm::normalize(glm::vec3(u(rng), u(rng), 1.5f));
            }

            const std::vector<glm::vec4> want = full(m, uvs, normals);
            const std::span<const glm::vec3> n = normals ? std::span<const glm::vec3>(m.normals) : std::span<const glm::vec3>();
            const std::span<const glm::vec2> t = uvs ? std::span<const glm::vec2>(m.uvs) : std::span<const glm::vec2>();

            // the one-ring is written and nothing else, and what was there before is never read
            const glm::vec4        poison(7.f);
            std::vector<glm::vec4> poisoned(m.positions.size(), poison);
            ASSERT_TRUE(updateTangents(m.positions, n, t, m.indices, *ring, changed, poisoned));
            const std::set<uint32_t> reached = oneRing(m, changed);
            size_t                   written = 0;
            for(uint32_t v = 0; v < m.positions.size(); v++)
              if(reached.count(v))
                written += same(poisoned[v], want[v]) ? 1 : 0;
              else
                EXPECT_TRUE(same(poisoned[v], poison)) << "step " << step << ": vertex " << v << " is past the one-ring";
            EXPECT_EQ(written, reached.size()) << "step " << step;

            // and edit after edit, the tangents are the full recompute's
            ASSERT_TRUE(updateTangents(m.positions, n, t, m.indices, *ring, changed, tangents));
            ASSERT_EQ(std::memcmp(tangents.data(), want.data(), want.size() * sizeof(glm::vec4)), 0) << "step " << step;
          }
        }
}

// The Box Figure's body (tests/figure/data) under a dial, over every group and over some: the body
// moves where the dial's deltas reach, and its tangents remade there are the full recompute's.
TEST(RetainedTangents, TheBoxFigureBodyUnderADial)
{
  namespace fs               = std::filesystem;
  const fs::path        root = fs::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions options;
  options.subdivision = 1;
  auto f = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<fs::path>{root})),
                            "People/Test/Box Figure.duf", options);
  f->settle();
  f->drain();
  const auto   body  = [&]() -> const fv::mesh::BuiltMesh& { return *f->loaded.scene.meshes[0].geo; };
  const size_t count = body().position.size() / 3;
  ASSERT_GT(count, 0u);
  ASSERT_EQ(body().normal.size(), count * 3);
  ASSERT_EQ(body().uv.size(), count * 2);

  struct Arrays
  {
    std::vector<glm::vec3> positions, normals;
    std::vector<glm::vec2> uvs;
  };
  const auto arrays = [&] {
    const fv::mesh::BuiltMesh& geo = body();
    EXPECT_EQ(geo.position.size(), count * 3);
    Arrays a;
    a.positions.resize(count);
    a.normals.resize(count);
    a.uvs.resize(count);
    std::memcpy(a.positions.data(), geo.position.data(), count * sizeof(glm::vec3));
    std::memcpy(a.normals.data(), geo.normal.data(), count * sizeof(glm::vec3));
    std::memcpy(a.uvs.data(), geo.uv.data(), count * sizeof(glm::vec2));
    return a;
  };

  // every group's triangles, and every other group's (the rest need no tangents)
  const std::vector<uint32_t> every = body().fullIndex;
  std::vector<uint32_t>       some;
  for(size_t g = 0; g < body().groups.size(); g += 2)
  {
    const fv::mesh::Group& group = body().groups[g];
    some.insert(some.end(), every.begin() + group.start, every.begin() + group.start + group.count);
  }
  for(const bool all : {true, false})
  {
    SCOPED_TRACE(all ? "every group" : "some groups");
    const std::vector<uint32_t>* indices = all ? &every : &some;
    ASSERT_FALSE(indices->empty());
    f->morphs->set("Stretch", 0);
    f->morphs->settled();
    f->drain();
    const auto ring = tangentAdjacency(*indices, count);
    ASSERT_TRUE(ring.has_value());
    Arrays                 was = arrays();
    std::vector<glm::vec4> tangents(count, glm::vec4(0.f));
    computeTangents(was.positions, was.normals, was.uvs, *indices, tangents);
    for(const double value : {1.0, 0.5, 0.0})
    {
      f->morphs->set("Stretch", value);
      f->morphs->settled();
      f->drain();
      const Arrays          now = arrays();
      std::vector<uint32_t> changed;
      for(uint32_t v = 0; v < count; v++)
        if(std::memcmp(&was.positions[v], &now.positions[v], sizeof(glm::vec3)) != 0
           || std::memcmp(&was.normals[v], &now.normals[v], sizeof(glm::vec3)) != 0)
          changed.push_back(v);
      ASSERT_FALSE(changed.empty()) << value;
      ASSERT_TRUE(updateTangents(now.positions, now.normals, now.uvs, *indices, *ring, changed, tangents));
      std::vector<glm::vec4> want(count, glm::vec4(0.f));
      computeTangents(now.positions, now.normals, now.uvs, *indices, want);
      EXPECT_EQ(std::memcmp(tangents.data(), want.data(), count * sizeof(glm::vec4)), 0) << value;
      was = now;
    }
  }
}
