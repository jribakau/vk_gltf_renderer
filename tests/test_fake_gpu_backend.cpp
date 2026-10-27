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

// FakeGpuBackend's self-tests: the residency tests (P3a1 on) read the GPU's state from it, so it must
// hold what a GPU would and refuse what a GPU would get wrong.

#include <gtest/gtest.h>

#include <cstring>
#include <set>

#include "common/fake_gpu_backend.hpp"

using namespace rs;

namespace {

template <class T>
std::span<const std::byte> asBytes(const std::vector<T>& v)
{
  return std::as_bytes(std::span(v));
}

// a triangle's positions and indices in two buffers, and the BLAS input over them
struct Triangle
{
  BufferHandle positions;
  BufferHandle indices;
  BlasInput    input;
};

Triangle uploadTriangle(FakeGpuBackend& gpu, float x = 1.0f)
{
  const std::vector<float>    p = {0, 0, 0, x, 0, 0, 0, 1, 0};
  const std::vector<uint32_t> i = {0, 1, 2};
  Triangle                    t;
  t.positions = gpu.createBuffer({.size = sizeof(float) * p.size(), .usage = eBufferVertex | eBufferAccelInput});
  t.indices   = gpu.createBuffer({.size = sizeof(uint32_t) * i.size(), .usage = eBufferIndex | eBufferAccelInput});
  gpu.upload(t.positions, 0, asBytes(p));
  gpu.upload(t.indices, 0, asBytes(i));
  t.input = {.vertexAddress = gpu.bufferAddress(t.positions), .vertexCount = 3, .indexAddress = gpu.bufferAddress(t.indices), .indexCount = 3};
  gpu.barrier(GpuBarrier::eTransferToBuild);
  return t;
}

}  // namespace

TEST(FakeGpuBackend, BuffersHoldTheirBytesAtDistinctAlignedAddresses)
{
  FakeGpuBackend     gpu;
  std::set<uint64_t> addresses;
  for(uint64_t size : {1, 16, 17, 4096})
  {
    const BufferHandle b = gpu.createBuffer({.size = size});
    ASSERT_TRUE(b);
    EXPECT_EQ(gpu.bufferSize(b), size);
    EXPECT_EQ(gpu.bufferAddress(b) % 16, 0u);
    EXPECT_TRUE(addresses.insert(gpu.bufferAddress(b)).second);
  }

  const BufferHandle      a     = gpu.createBuffer({.size = 8});
  const BufferHandle      b     = gpu.createBuffer({.size = 8});
  const std::vector<char> hello = {'h', 'e', 'l', 'l', 'o', '!', '!', '!'};
  gpu.upload(a, 0, asBytes(hello));
  gpu.copyBuffer(a, 1, b, 2, 4);
  EXPECT_EQ(std::memcmp(gpu.buffer(b)->bytes.data() + 2, "ello", 4), 0);
  EXPECT_EQ(std::memcmp(gpu.bytesAt(gpu.bufferAddress(a) + 1, 4).data(), "ello", 4), 0);
  EXPECT_TRUE(gpu.bytesAt(gpu.bufferAddress(a) + 4, 5).empty());  // runs past the buffer's end
  EXPECT_TRUE(gpu.errors().empty());
}

TEST(FakeGpuBackend, OutOfRangeWritesAndDeadHandlesAreErrors)
{
  FakeGpuBackend          gpu;
  const BufferHandle      b    = gpu.createBuffer({.size = 4});
  const std::vector<char> five = {'1', '2', '3', '4', '5'};
  gpu.upload(b, 0, asBytes(five));
  EXPECT_EQ(gpu.errors().size(), 1u);
  gpu.copyBuffer(b, 2, b, 0, 4);
  EXPECT_EQ(gpu.errors().size(), 2u);

  gpu.destroyBuffer(b);
  EXPECT_EQ(gpu.buffer(b), nullptr);
  EXPECT_EQ(gpu.bufferAddress(b), 0u);
  gpu.upload(b, 0, {});
  gpu.destroyBuffer(b);
  EXPECT_EQ(gpu.errors().size(), 4u);
  EXPECT_EQ(gpu.liveBuffers(), 0u);
}

TEST(FakeGpuBackend, HandlesAreNeverReusedButAddressesAre)
{
  FakeGpuBackend     gpu;
  const BufferHandle first   = gpu.createBuffer({.size = 64});
  const uint64_t     address = gpu.bufferAddress(first);
  gpu.destroyBuffer(first);
  const BufferHandle second = gpu.createBuffer({.size = 64});
  EXPECT_NE(second, first);
  EXPECT_EQ(gpu.bufferAddress(second), address);  // the range is free again; a BLAS built over it must not refit
}

// A closure queued in frame f runs at the start of frame f + N, as nvapp::Application's free queue
// does with N frames in its ring: the GPU has finished every frame that could still use it.
TEST(FakeGpuBackend, RetireRunsAFrameCycleLater)
{
  FakeGpuBackend gpu(3);
  gpu.beginFrame();  // frame 1
  int ran = 0;
  gpu.retire([&] { ran++; });
  gpu.beginFrame();  // 2
  gpu.beginFrame();  // 3
  EXPECT_EQ(ran, 0);
  EXPECT_EQ(gpu.pendingRetires(), 1u);
  gpu.beginFrame();  // 4 = 1 + 3
  EXPECT_EQ(ran, 1);
  EXPECT_EQ(gpu.pendingRetires(), 0u);
  gpu.beginFrame();
  EXPECT_EQ(ran, 1);
}

TEST(FakeGpuBackend, ARetireQueuedByARetireWaitsItsOwnCycle)
{
  FakeGpuBackend gpu(2);
  int            second = 0;
  gpu.retire([&] { gpu.retire([&] { second++; }); });
  gpu.beginFrame();
  gpu.beginFrame();  // the first runs and queues the second, at frame 2
  EXPECT_EQ(second, 0);
  gpu.beginFrame();
  EXPECT_EQ(second, 0);
  gpu.beginFrame();  // 4 = 2 + 2
  EXPECT_EQ(second, 1);
}

TEST(FakeGpuBackend, WaitIdleIsCountedAndRunsNoRetire)
{
  FakeGpuBackend gpu;
  bool           ran = false;
  gpu.retire([&] { ran = true; });
  gpu.waitIdle();
  gpu.waitIdle();
  EXPECT_EQ(gpu.waitIdleCount(), 2u);
  EXPECT_FALSE(ran);
}

TEST(FakeGpuBackend, ABlasHashesTheBytesItWasBuiltFrom)
{
  FakeGpuBackend    gpu;
  Triangle          t    = uploadTriangle(gpu);
  const AccelHandle blas = gpu.buildBlas(t.input, eAccelPreferFastTrace | eAccelAllowUpdate);
  ASSERT_NE(gpu.blas(blas), nullptr);
  const uint64_t built = gpu.blas(blas)->contentHash;
  EXPECT_NE(built, 0u);
  EXPECT_EQ(gpu.contentHash(t.input), built);

  // the positions change: the GPU's BLAS is stale until a refit reads them again
  const std::vector<float> moved = {0, 0, 0, 2, 0, 0, 0, 1, 0};
  gpu.upload(t.positions, 0, asBytes(moved));
  EXPECT_NE(gpu.contentHash(t.input), built);
  EXPECT_EQ(gpu.blas(blas)->contentHash, built);
  gpu.barrier(GpuBarrier::eTransferToBuild);
  gpu.refitBlas(blas, t.input);
  EXPECT_EQ(gpu.blas(blas)->contentHash, gpu.contentHash(t.input));
  EXPECT_EQ(gpu.blas(blas)->refits, 1u);
  EXPECT_TRUE(gpu.errors().empty());

  // a BLAS over bytes no live buffer holds
  gpu.buildBlas({.vertexAddress = 16, .vertexCount = 3, .indexAddress = 16, .indexCount = 3}, 0);
  EXPECT_EQ(gpu.errors().size(), 1u);
}

TEST(FakeGpuBackend, RefitNeedsAllowUpdateTheSameAddressesAndNoCompaction)
{
  FakeGpuBackend gpu;
  Triangle       t = uploadTriangle(gpu);

  const AccelHandle fixed = gpu.buildBlas(t.input, eAccelAllowCompaction);
  gpu.refitBlas(fixed, t.input);
  EXPECT_EQ(gpu.errors().size(), 1u);

  const AccelHandle compacted = gpu.compactBlas(fixed);
  ASSERT_NE(gpu.blas(compacted), nullptr);
  EXPECT_TRUE(gpu.blas(compacted)->compacted);
  EXPECT_EQ(gpu.blas(compacted)->contentHash, gpu.blas(fixed)->contentHash);

  const AccelHandle refittable = gpu.buildBlas(t.input, eAccelAllowUpdate);
  Triangle          other      = uploadTriangle(gpu);
  gpu.refitBlas(refittable, other.input);  // moved to other buffers: a build, not a refit
  EXPECT_EQ(gpu.errors().size(), 2u);

  gpu.compactBlas(refittable);  // not built for compaction
  EXPECT_EQ(gpu.errors().size(), 3u);
}

TEST(FakeGpuBackend, ATlasKeepsItsRowsAndUpdatesOnlyOverTheSameBlases)
{
  FakeGpuBackend    gpu;
  Triangle          t    = uploadTriangle(gpu);
  const AccelHandle a    = gpu.buildBlas(t.input, 0);
  const AccelHandle b    = gpu.buildBlas(t.input, 0);
  const AccelHandle tlas = gpu.createTlas(4);
  gpu.barrier(GpuBarrier::eBuildToBuild);

  std::vector<TlasRow> rows(3);
  rows[0].blas = a;
  rows[0].mask = 0xFF;
  rows[2].blas = b;  // row 1 is a hole
  rows[2].mask = 0xFF;
  gpu.buildTlas(tlas, rows, TlasMode::eBuild);
  rows[0].transform[0][3] = 5.0f;  // moved: an update is valid
  gpu.buildTlas(tlas, rows, TlasMode::eUpdate);
  ASSERT_NE(gpu.tlas(tlas), nullptr);
  EXPECT_EQ(gpu.tlas(tlas)->rows.size(), 3u);
  EXPECT_EQ(gpu.tlas(tlas)->rows[0].transform[0][3], 5.0f);
  EXPECT_FALSE(gpu.tlas(tlas)->rows[1].blas);
  EXPECT_EQ(gpu.tlas(tlas)->builds, 1u);
  EXPECT_EQ(gpu.tlas(tlas)->updates, 1u);
  EXPECT_EQ(gpu.tlas(tlas)->lastMode, TlasMode::eUpdate);
  EXPECT_TRUE(gpu.errors().empty());

  rows[1].blas = b;  // a hole filled: the rows' references changed, so an update is invalid
  gpu.buildTlas(tlas, rows, TlasMode::eUpdate);
  EXPECT_EQ(gpu.errors().size(), 1u);
  rows.resize(5);  // past the capacity
  gpu.buildTlas(tlas, rows, TlasMode::eBuild);
  EXPECT_EQ(gpu.errors().size(), 2u);

  rows.resize(1);
  gpu.destroyAccel(a);
  gpu.buildTlas(tlas, rows, TlasMode::eBuild);  // row 0 names a destroyed BLAS
  EXPECT_EQ(gpu.errors().size(), 3u);
  EXPECT_EQ(gpu.liveAccels(), 2u);
}

// A build reading bytes written since the last transfer-to-build barrier, and a TLAS build over
// BLAS work since the last build-to-build, are errors; what the frame's shading and tracing would
// read unfenced is reported until its barrier
TEST(FakeGpuBackend, BarriersFenceWhatBuildsAndShadingRead)
{
  FakeGpuBackend gpu;
  Triangle       t = uploadTriangle(gpu);
  EXPECT_EQ(gpu.unfencedForTheFrame().size(), 1u);  // the uploads, for shading
  gpu.barrier(GpuBarrier::eTransferToShader);
  const AccelHandle blas = gpu.buildBlas(t.input, eAccelAllowUpdate);
  EXPECT_TRUE(gpu.errors().empty());

  const std::vector<float> moved = {0, 0, 0, 2, 0, 0, 0, 1, 0};
  gpu.upload(t.positions, 0, asBytes(moved));
  gpu.refitBlas(blas, t.input);  // no barrier since the upload
  EXPECT_EQ(gpu.errors().size(), 1u);

  const AccelHandle tlas = gpu.createTlas(1);
  TlasRow           row;
  row.blas = blas;
  gpu.buildTlas(tlas, std::span(&row, 1), TlasMode::eBuild);  // no build-to-build since the refit
  EXPECT_EQ(gpu.errors().size(), 2u);
  EXPECT_EQ(gpu.unfencedForTheFrame().size(), 2u);  // the upload and the builds
  gpu.barrier(GpuBarrier::eTransferToShader);
  gpu.barrier(GpuBarrier::eBuildToTrace);
  EXPECT_TRUE(gpu.unfencedForTheFrame().empty());
}

TEST(FakeGpuBackend, AnUpdateBeforeAnyBuildIsAnError)
{
  FakeGpuBackend    gpu;
  const AccelHandle tlas = gpu.createTlas(1);
  gpu.buildTlas(tlas, {}, TlasMode::eUpdate);
  EXPECT_EQ(gpu.errors().size(), 1u);
}

TEST(FakeGpuBackend, DescriptorsNameLiveImagesOrTheDefault)
{
  FakeGpuBackend                   gpu;
  const std::vector<unsigned char> pixel(4);
  ImageInfo                        info;
  const ImageHandle image = gpu.createImage({.pixels = pixel, .width = 1, .height = 1, .component = 4, .srgb = true}, info);
  EXPECT_EQ(info.width, 1u);
  EXPECT_FALSE(info.twoChannel);
  EXPECT_FALSE(gpu.createImage({.uri = "missing.png"}, info));
  gpu.writeTextureDescriptor(3, image);
  gpu.writeSamplerDescriptor(1, {.magFilter = 9728});
  EXPECT_EQ(gpu.textureDescriptors().at(3), image);
  EXPECT_EQ(gpu.samplerDescriptors().at(1).magFilter, 9728);

  // a freed slot goes back to the default image, then the image is destroyed
  gpu.writeTextureDescriptor(3, {});
  gpu.destroyImage(image);
  EXPECT_FALSE(gpu.textureDescriptors().at(3));
  EXPECT_TRUE(gpu.errors().empty());

  gpu.writeTextureDescriptor(4, image);
  EXPECT_EQ(gpu.errors().size(), 1u);
}

TEST(FakeGpuBackend, PassesAreRecordedInOrder)
{
  FakeGpuBackend     gpu;
  const BufferHandle in  = gpu.createBuffer({.size = 16});
  const BufferHandle out = gpu.createBuffer({.size = 16});
  gpu.dispatchDeform({.kind = DeformKind::eSkin, .inputs = {in}, .output = out, .vertexCount = 1});
  gpu.barrier(GpuBarrier::eComputeToBuild);
  gpu.barrier(GpuBarrier::eBuildToTrace);
  ASSERT_EQ(gpu.deforms().size(), 1u);
  EXPECT_EQ(gpu.deforms()[0].kind, DeformKind::eSkin);
  EXPECT_EQ(gpu.barriers(), (std::vector<GpuBarrier>{GpuBarrier::eComputeToBuild, GpuBarrier::eBuildToTrace}));
  EXPECT_TRUE(gpu.errors().empty());

  gpu.destroyBuffer(in);
  gpu.dispatchDeform({.kind = DeformKind::eMorph, .inputs = {in}, .output = out});
  EXPECT_EQ(gpu.errors().size(), 1u);
}

// The residency layer's pattern: a slot id goes back to its pool only in the retire closure that
// destroys its buffer, so nothing in flight reads a reused slot.
TEST(FakeGpuBackend, ARetiredBufferIsFreedAndItsSlotReusedAfterTheCycle)
{
  FakeGpuBackend     gpu(2);
  std::vector<int>   freeSlots;
  const BufferHandle b    = gpu.createBuffer({.size = 32});
  const int          slot = 7;
  gpu.retire([&] {
    gpu.destroyBuffer(b);
    freeSlots.push_back(slot);
  });
  gpu.beginFrame();
  EXPECT_EQ(gpu.liveBuffers(), 1u);
  EXPECT_TRUE(freeSlots.empty());
  gpu.beginFrame();
  EXPECT_EQ(gpu.liveBuffers(), 0u);
  EXPECT_EQ(freeSlots, std::vector<int>{7});
  EXPECT_TRUE(gpu.errors().empty());
}
