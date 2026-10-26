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

// CPU-only unit test for nvvk::buildEnvAliasmap (Vose's method for environment importance
// sampling). Exercises the exact function backing both the file-load path
// (createEnvironmentAccel) and the GPU-image path (HdrIbl::updateFromGpuImage), so any
// regression here surfaces in both producer pipelines.
//
// The test does not stand in for the Monte-Carlo end-to-end validation the
// `--benchmark-env-pipeline` mode runs on the GPU; it pins the *algorithm*, not its use.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

// Pull in the standalone alias-map header rather than <nvvk/hdr_ibl.hpp>. The latter drags the
// whole Vulkan/VMA descriptor stack into the test binary (via nvvk::ResourceAllocator), which
// then fails to link because the test target does not compile a VMA implementation.
#include <nvvk/env_aliasmap.hpp>

namespace {

// Sum of a `size` array of ones and one large spike -- the classic sunset case where a single
// texel dominates the integral. Vose's method must correctly assign many low-energy texels as
// aliases of the spike.
std::vector<float> makeSunset(size_t size, float spike)
{
  std::vector<float> data(size, 1.0F);
  data[size / 2] = spike;
  return data;
}

// Uniform importance across all texels. Every accel.q must round to 1.0 and each texel
// remains its own alias.
std::vector<float> makeUniform(size_t size)
{
  return std::vector<float>(size, 3.14F);
}

// Random non-negative importance in [0..1). Sanity fixture for the general case.
std::vector<float> makeRandom(size_t size, uint32_t seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> dist(0.0F, 1.0F);
  std::vector<float>                    data(size);
  for(auto& v : data)
    v = dist(rng);
  return data;
}

// Direct evaluation of the Vose sampling equation used by hdr_env_sampling.h.slang:
//   P(pick j) = (1/N) * ( q[j] + sum_{i : alias[i] == j and q[i] < 1} (1 - q[i]) )
// This is what actually matters for the PT: samples produced by (i, u) draws must land on j
// with probability proportional to data[j]. Any bug in Vose partitioning will show up as a
// distribution that no longer matches data / sum(data).
std::vector<double> computeSamplingDistribution(const std::vector<shaderio::EnvAccel>& accel)
{
  const size_t        n = accel.size();
  std::vector<double> p(n, 0.0);
  for(size_t i = 0; i < n; ++i)
  {
    // Direct-hit contribution: uniform pick of index i, then u < q[i].
    p[i] += std::min<double>(accel[i].q, 1.0);
    // Alias contribution: uniform pick of some other index k with q[k] < 1 that aliases to i.
    if(accel[i].q < 1.0F)
      p[accel[i].alias] += 1.0 - static_cast<double>(accel[i].q);
  }
  for(auto& v : p)
    v /= static_cast<double>(n);
  return p;
}

}  // namespace


TEST(AliasTable, UniformImportanceKeepsIdentityAliases)
{
  constexpr size_t kSize = 128;
  const auto       data  = makeUniform(kSize);

  std::vector<shaderio::EnvAccel> accel(kSize);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);

  // std::accumulate over 128 floats does not produce the same rounding as a single multiply,
  // so we compare against the accumulated ground truth rather than 3.14F * kSize directly.
  const double expectedIntegral = std::accumulate(data.begin(), data.end(), 0.0);
  EXPECT_NEAR(integral, static_cast<float>(expectedIntegral), 1e-3F) << "Integral must equal sum(importance)";

  for(size_t i = 0; i < kSize; ++i)
  {
    // All q values should be exactly 1 (or extremely close, within float rounding). The alias
    // partition never triggers when q>=1 for every texel, so every texel keeps its own index.
    EXPECT_NEAR(accel[i].q, 1.0F, 1e-5F) << "Uniform input: accel[" << i << "].q must be ~1";
    EXPECT_EQ(accel[i].alias, static_cast<uint32_t>(i)) << "Uniform input: accel[" << i << "].alias must be self";
  }
}


TEST(AliasTable, SunsetSpikeAliasesLowEnergyTexelsToTheSpike)
{
  constexpr size_t kSize  = 256;
  constexpr float  kSpike = 200.0F;
  const auto       data   = makeSunset(kSize, kSpike);

  std::vector<shaderio::EnvAccel> accel(kSize);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);

  // Integral = (kSize - 1) * 1.0 + kSpike.
  EXPECT_FLOAT_EQ(integral, static_cast<float>(kSize - 1) + kSpike);

  // Every dim texel (q < 1) must alias to the spike, which is the only high-energy entry.
  const uint32_t spikeIdx       = kSize / 2;
  size_t         aliasedToSpike = 0;
  for(size_t i = 0; i < kSize; ++i)
  {
    if(i == spikeIdx)
      continue;
    // The low-energy texels have q < 1 and must alias to the spike (no other high-energy
    // texel exists in this fixture).
    EXPECT_LT(accel[i].q, 1.0F) << "Dim texel " << i << " should have q < 1";
    if(accel[i].alias == spikeIdx)
      ++aliasedToSpike;
  }
  EXPECT_EQ(aliasedToSpike, kSize - 1) << "All dim texels must alias to the single spike";
}


TEST(AliasTable, SamplingDistributionMatchesInputForRandomImportance)
{
  // The load-bearing invariant: after alias construction, the effective sampling probability
  // for each index j -- computed the same way the sampling shader does it -- must match the
  // input importance normalized to a PDF. This is what the PT relies on for unbiased NEE.
  constexpr size_t         kSize  = 512;
  const std::vector<float> data   = makeRandom(kSize, 0x1234u);
  const double             refSum = std::accumulate(data.begin(), data.end(), 0.0);

  std::vector<shaderio::EnvAccel> accel(kSize);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);
  EXPECT_NEAR(integral, static_cast<float>(refSum), 1e-3F);

  const std::vector<double> p      = computeSamplingDistribution(accel);
  double                    totalP = 0.0;
  for(size_t j = 0; j < kSize; ++j)
  {
    const double expected = static_cast<double>(data[j]) / refSum;
    // The tolerance is generous because Vose is done in single-precision floats.
    EXPECT_NEAR(p[j], expected, 1e-4) << "P(sample " << j << ") diverges from input importance";
    totalP += p[j];
  }
  EXPECT_NEAR(totalP, 1.0, 1e-4) << "Sampling distribution must integrate to 1";
}


TEST(AliasTable, AliasIndicesAreInBounds)
{
  constexpr size_t kSize = 1024;
  const auto       data  = makeRandom(kSize, 0xC0FFEEu);

  std::vector<shaderio::EnvAccel> accel(kSize);
  nvvk::buildEnvAliasmap(data, accel);

  for(size_t i = 0; i < kSize; ++i)
  {
    EXPECT_LT(accel[i].alias, static_cast<uint32_t>(kSize)) << "alias[" << i << "] out of range";
    EXPECT_GE(accel[i].q, 0.0F) << "q[" << i << "] must be non-negative";
  }
}


TEST(AliasTable, ZeroImportanceProducesFallbackIntegral)
{
  // All zeros -> integral would be 0; buildEnvAliasmap must return 1 (the sentinel) to avoid
  // downstream division by zero in HdrIbl.
  const std::vector<float>        data(64, 0.0F);
  std::vector<shaderio::EnvAccel> accel(64);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);

  EXPECT_FLOAT_EQ(integral, 1.0F);
  for(size_t i = 0; i < accel.size(); ++i)
  {
    EXPECT_FLOAT_EQ(accel[i].q, 0.0F);
    EXPECT_EQ(accel[i].alias, static_cast<uint32_t>(i));
  }
}

//--------------------------------------------------------------------------------------------------
// nvvk::buildCompensatedEnvAliasmap: the environment's table for light samples a BSDF sample is
// weighed against (MIS compensation)
//--------------------------------------------------------------------------------------------------
namespace {

constexpr double kPi = 3.14159265358979323846;

// The sampler's cell solid angles for a `width` x `height` lat-long grid
std::vector<double> cellSolidAngles(uint32_t width, uint32_t height)
{
  std::vector<double> area(size_t(width) * height);
  for(uint32_t y = 0; y < height; ++y)
    for(uint32_t x = 0; x < width; ++x)
      area[y * width + x] = (std::cos(y * kPi / height) - std::cos((y + 1) * kPi / height)) * 2.0 * kPi / width;
  return area;
}

// Importance (radiance times solid angle, as HdrIbl feeds it) of a dim sky with a few bright cells
std::vector<float> skyWithSuns(uint32_t width, uint32_t height, const std::vector<double>& area, uint32_t seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> dist(0.5F, 1.5F);
  std::vector<float>                    importance(area.size());
  for(size_t i = 0; i < area.size(); ++i)
    importance[i] = float(dist(rng) * area[i]);
  for(int k = 0; k < 5; ++k)
    importance[rng() % area.size()] *= 40.0F;
  return importance;
}

double powerWeight(double a, double b)
{
  return b > 0.0 ? a * a / (a * a + b * b) : 1.0;
}

}  // namespace

// The pdf integrates to one over the sphere, is 0 where the radiance is below the sphere's mean
// and in proportion to the excess above it elsewhere, and the alias method picks cells by it.
TEST(CompensatedAliasTable, ThePdfIsTheExcessOverTheMean)
{
  const uint32_t                    W = 16, H = 8;
  const std::vector<double>         area       = cellSolidAngles(W, H);
  const std::vector<float>          importance = skyWithSuns(W, H, area, 3);
  std::vector<shaderio::EnvCompensatedCell> cells(nvvk::compensatedEnvTableSize(W, H));
  nvvk::buildCompensatedEnvAliasmap(importance, W, H, cells);

  double total = 0.0;
  for(float v : importance)
    total += v;
  const double mean = total / (4.0 * kPi);

  double integral = 0.0, ratio = -1.0;
  int    zeros    = 0;
  for(size_t i = 0; i < area.size(); ++i)
  {
    integral += cells[i].pdf * area[i];
    const double excess = importance[i] / area[i] - mean;
    if(excess <= 0.0)
    {
      EXPECT_EQ(cells[i].pdf, 0.0F) << i;
      zeros++;
    }
    else
    {
      if(ratio < 0.0)
        ratio = cells[i].pdf / excess;
      EXPECT_NEAR(cells[i].pdf / excess, ratio, ratio * 1e-3) << i;
    }
  }
  EXPECT_NEAR(integral, 1.0, 1e-4);
  EXPECT_GT(zeros, int(area.size()) / 4);

  std::vector<shaderio::EnvAccel> accel(area.size());
  for(size_t i = 0; i < area.size(); ++i)
    accel[i] = {.alias = cells[i].alias, .q = cells[i].q};
  const std::vector<double> picked = computeSamplingDistribution(accel);
  for(size_t i = 0; i < area.size(); ++i)
    EXPECT_NEAR(picked[i], cells[i].pdf * area[i], 1e-4) << i;
}

// A uniform environment has no excess: it keeps the full distribution, 1 / 4pi everywhere.
TEST(CompensatedAliasTable, AUniformEnvironmentKeepsTheFullOne)
{
  const uint32_t                            W = 16, H = 8;
  const std::vector<double>                 area = cellSolidAngles(W, H);
  std::vector<float>                        importance(area.size());
  for(size_t i = 0; i < area.size(); ++i)
    importance[i] = float(2.0 * area[i]);
  std::vector<shaderio::EnvCompensatedCell> cells(nvvk::compensatedEnvTableSize(W, H));
  nvvk::buildCompensatedEnvAliasmap(importance, W, H, cells);
  for(size_t i = 0; i < area.size(); ++i)
    EXPECT_NEAR(cells[i].pdf, 1.0 / (4.0 * kPi), 1e-5);
}

// A furnace of the MIS it is for: light samples from the compensated table and uniform "BSDF"
// samples over the sphere, power-weighted against each other, estimate the environment's integral
// without bias -- where the table gives a cell nothing, the BSDF sample carries it alone.
TEST(CompensatedAliasTable, MisWithTheBsdfSampleIsUnbiased)
{
  const uint32_t                            W = 16, H = 8;
  const std::vector<double>                 area       = cellSolidAngles(W, H);
  const std::vector<float>                  importance = skyWithSuns(W, H, area, 9);
  std::vector<shaderio::EnvCompensatedCell> cells(nvvk::compensatedEnvTableSize(W, H));
  nvvk::buildCompensatedEnvAliasmap(importance, W, H, cells);

  double exact = 0.0;
  for(float v : importance)
    exact += v;

  std::mt19937                           rng(17);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::discrete_distribution<size_t>     byArea(area.begin(), area.end());
  const double                           bsdfPdf = 1.0 / (4.0 * kPi);
  const int                              N       = 400000;
  double                                 sum     = 0.0;
  for(int s = 0; s < N; ++s)
  {
    // The light sample: the alias method as environmentSampleCompensated runs it
    const size_t                       idx   = std::min(size_t(u(rng) * area.size()), area.size() - 1);
    const shaderio::EnvCompensatedCell entry = cells[idx];
    const size_t                       cell  = (u(rng) < entry.q) ? idx : entry.alias;
    const double                       L     = importance[cell] / area[cell];
    if(cells[cell].pdf > 0.0F)
      sum += L / cells[cell].pdf * powerWeight(cells[cell].pdf, bsdfPdf);
    // The BSDF sample: uniform over the sphere
    const size_t b = byArea(rng);
    sum += importance[b] / area[b] / bsdfPdf * powerWeight(bsdfPdf, cells[b].pdf);
  }
  EXPECT_NEAR(sum / N / exact, 1.0, 0.01);
}

// The cumulative tables after the grid (EnvCompensatedCell::cdf) pick each cell with the probability
// the alias method gives it, pdf times solid angle, as environmentSampleCompensatedStratified inverts
// them; and the inversion is monotonic, so stratified points stay in order over the rows and along
// each row.
TEST(CompensatedAliasTable, TheCumulativeTablesPickWhatTheAliasMethodDoes)
{
  const uint32_t                            W = 16, H = 8;
  const std::vector<double>                 area       = cellSolidAngles(W, H);
  const std::vector<float>                  importance = skyWithSuns(W, H, area, 5);
  std::vector<shaderio::EnvCompensatedCell> cells(nvvk::compensatedEnvTableSize(W, H));
  nvvk::buildCompensatedEnvAliasmap(importance, W, H, cells);
  const size_t rows = size_t(W) * H;

  // The shader's two searches: the first entry whose cdf passes u
  const auto search = [&](size_t first, uint32_t count, float u) {
    uint32_t lo = 0, hi = count - 1;
    while(lo < hi)
    {
      const uint32_t mid = (lo + hi) / 2;
      if(cells[first + mid].cdf > u)
        hi = mid;
      else
        lo = mid + 1;
    }
    return lo;
  };
  const int           n = 2048;
  std::vector<double> picked(rows, 0.0);
  uint32_t            lastRow = 0;
  for(int a = 0; a < n; ++a)
  {
    const float    ux = (a + 0.5f) / n;
    const uint32_t y  = search(rows, H, ux);
    EXPECT_GE(y, lastRow);
    lastRow           = y;
    uint32_t lastCell = 0;
    for(int b = 0; b < n; ++b)
    {
      const uint32_t x = search(size_t(y) * W, W, (b + 0.5f) / n);
      EXPECT_GE(x, lastCell);
      lastCell = x;
      picked[size_t(y) * W + x] += 1.0 / (double(n) * n);
    }
  }
  for(size_t i = 0; i < rows; ++i)
    EXPECT_NEAR(picked[i], cells[i].pdf * area[i], 2e-3) << i;
  EXPECT_EQ(cells[rows + H - 1].cdf, 1.0F);
  for(uint32_t y = 0; y < H; ++y)
    EXPECT_EQ(cells[size_t(y) * W + W - 1].cdf, 1.0F) << y;
}
