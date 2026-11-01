/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// What a test's scene came to, boiled down to a number and compared with tests/golden/retained.txt.
// The numbers were recorded while the importer and the clip player were still compared directly with
// the CPU glTF scene they replaced, and matched it; that scene is gone, so a changed number means the
// output moved, not which side is right. FV_UPDATE_GOLDEN=1 rewrites the entries a run reaches.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <string>

#include <glm/glm.hpp>
#include <gtest/gtest.h>

class GoldenDigest
{
public:
  void addBytes(const void* data, size_t size)
  {
    const auto* p = static_cast<const unsigned char*>(data);
    for(size_t i = 0; i < size; i++)
      m_hash = (m_hash ^ p[i]) * 1099511628211ull;
  }
  void add(int64_t v) { addBytes(&v, sizeof(v)); }
  // Rounded, so a compiler's last bit doesn't count
  void add(float v) { add(int64_t(std::llround(double(v) * 4096.0))); }
  void add(const glm::vec3& v)
  {
    for(int i = 0; i < 3; i++)
      add(v[i]);
  }
  void add(const glm::vec4& v)
  {
    for(int i = 0; i < 4; i++)
      add(v[i]);
  }
  void add(const glm::mat4& m)
  {
    for(int i = 0; i < 4; i++)
      add(m[i]);
  }
  template <class T>
  void addAll(std::span<const T> values)
  {
    add(int64_t(values.size()));
    for(const T& v : values)
      add(v);
  }
  uint64_t value() const { return m_hash; }

private:
  uint64_t m_hash = 14695981039346656037ull;
};

inline void expectGolden(const std::string& name, const GoldenDigest& digest)
{
  const std::filesystem::path file = std::filesystem::path(FV_TEST_GOLDEN_DIR) / "retained.txt";
  std::map<std::string, std::string> entries;
  {
    std::ifstream in(file);
    std::string   line;
    while(std::getline(in, line))
      if(const size_t space = line.rfind(' '); space != std::string::npos)
        entries[line.substr(0, space)] = line.substr(space + 1);
  }
  std::ostringstream hex;
  hex << std::hex << digest.value();
  if(std::getenv("FV_UPDATE_GOLDEN"))
  {
    entries[name] = hex.str();
    std::ofstream out(file, std::ios::binary);
    for(const auto& [key, value] : entries)
      out << key << ' ' << value << '\n';
    return;
  }
  const auto it = entries.find(name);
  if(it == entries.end())
    ADD_FAILURE() << name << ": not in " << file.string();
  else
    EXPECT_EQ(hex.str(), it->second) << name;
}
