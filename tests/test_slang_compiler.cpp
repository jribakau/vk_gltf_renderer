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

// nvslang::SlangCompiler: its global session (Slang's core module, ~160 MB) made by the first compile,
// the profile and capabilities kept by name until then

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <nvslang/slang.hpp>

namespace {

const char* kSource = R"(
RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  output[id.x] = id.x * 2u;
}
)";

// As the renderer sets its compiler up
void setUp(nvslang::SlangCompiler& compiler)
{
  compiler.defaultTarget();
  compiler.defaultOptions();
  compiler.addCapability("spvRayQueryKHR");
  compiler.addCapability("spvGroupNonUniformBallot");
}

std::vector<uint32_t> spirvOf(const nvslang::SlangCompiler& compiler)
{
  return {compiler.getSpirv(), compiler.getSpirv() + compiler.getSpirvSize() / sizeof(uint32_t)};
}

}  // namespace

TEST(SlangCompiler, ItMakesNoGlobalSessionUntilItCompiles)
{
  nvslang::SlangCompiler compiler;
  setUp(compiler);
  EXPECT_FALSE(compiler.hasGlobalSession());
  EXPECT_EQ(compiler.profile(), "spirv_1_6");
  EXPECT_EQ(compiler.capabilities(), (std::vector<std::string>{"spvRayQueryKHR", "spvGroupNonUniformBallot"}));

  ASSERT_TRUE(compiler.loadFromSourceString("doubled", kSource)) << compiler.getLastDiagnosticMessage();
  EXPECT_TRUE(compiler.hasGlobalSession());
  const std::vector<uint32_t> spirv = spirvOf(compiler);
  ASSERT_GE(spirv.size(), 5u);
  EXPECT_EQ(spirv[0], 0x07230203u);  // SPIR-V's magic number
  EXPECT_EQ(spirv[1], 0x00010600u);  // version 1.6: the profile by name was looked up and used
}

TEST(SlangCompiler, ItsTargetsAndOptionsReadTheSameBeforeAndAfterACompile)
{
  // what a SPIR-V cache key is made of: a key taken before the first compile matches one taken after
  nvslang::SlangCompiler compiler;
  setUp(compiler);
  const std::vector<slang::TargetDesc> targets = compiler.targets();
  const size_t                         options = compiler.options().size();

  ASSERT_TRUE(compiler.loadFromSourceString("doubled", kSource)) << compiler.getLastDiagnosticMessage();
  ASSERT_EQ(compiler.targets().size(), targets.size());
  EXPECT_EQ(compiler.targets()[0].profile, targets[0].profile);
  EXPECT_EQ(compiler.options().size(), options);
}

TEST(SlangCompiler, ACopyByNamesCompilesAsTheOriginal)
{
  // what a variant job does: another compiler from the first one's targets, options and names
  nvslang::SlangCompiler original;
  setUp(original);
  nvslang::SlangCompiler copy;
  copy.targets() = original.targets();
  copy.options() = original.options();
  copy.setProfile(original.profile().c_str());
  for(const std::string& capability : original.capabilities())
    copy.addCapability(capability.c_str());

  ASSERT_TRUE(original.loadFromSourceString("doubled", kSource)) << original.getLastDiagnosticMessage();
  ASSERT_TRUE(copy.loadFromSourceString("doubled", kSource)) << copy.getLastDiagnosticMessage();
  EXPECT_EQ(spirvOf(copy), spirvOf(original));
}

TEST(SlangCompiler, AnUnknownCapabilityIsLeftOutAndTheCompileGoesOn)
{
  nvslang::SlangCompiler compiler;
  setUp(compiler);
  compiler.addCapability("noSuchCapabilityAnywhere");
  EXPECT_TRUE(compiler.loadFromSourceString("doubled", kSource)) << compiler.getLastDiagnosticMessage();
}
