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

#pragma once

//
// The only thing the retained scene's residency layer calls on the GPU (docs: daz-viewer
// docs/retained-scene-plan.md, "Testing without a GPU"). Vulkan-free, so the residency code and its
// tests build without a device: the Vulkan backend wraps the allocator, the uploader and the AS
// builder; FakeGpuBackend (tests/common) keeps shadow bytes and a frame clock instead.
//
// Handles are opaque and never reused: a destroyed one names nothing. Commands are recorded into the
// current frame in call order; addresses are device addresses, valid from create to destroy.
//

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace rs {

template <class Tag>
struct GpuHandle
{
  uint64_t value = 0;  // 0: none

  explicit operator bool() const { return value != 0; }
  bool     operator==(const GpuHandle&) const = default;
};

struct GpuBufferTag;
struct GpuImageTag;
struct GpuAccelTag;
using BufferHandle = GpuHandle<GpuBufferTag>;
using ImageHandle  = GpuHandle<GpuImageTag>;
using AccelHandle  = GpuHandle<GpuAccelTag>;  // a BLAS or a TLAS

enum BufferUsageBits : uint32_t
{
  eBufferVertex     = 1u << 0,
  eBufferIndex      = 1u << 1,
  eBufferStorage    = 1u << 2,
  eBufferAccelInput = 1u << 3,  // read by a BLAS build (positions, indices)
};
using BufferUsageFlags = uint32_t;

struct BufferDesc
{
  uint64_t         size  = 0;
  BufferUsageFlags usage = 0;
  const char*      name  = nullptr;  // debug name
};

enum class ImageFormat : uint32_t
{
  eRgba8Unorm,
  eRgba8Srgb,
  eRgba16Float,
  eRgba32Float,
};

struct ImageDesc
{
  uint32_t    width     = 0;
  uint32_t    height    = 0;
  uint32_t    mipLevels = 1;  // the backend generates levels past 0
  ImageFormat format    = ImageFormat::eRgba8Unorm;
  const char* name      = nullptr;
};

// glTF sampler values (TINYGLTF_TEXTURE_FILTER_*, TINYGLTF_TEXTURE_WRAP_*); a filter of -1 is unset
struct SamplerDesc
{
  int  magFilter                            = -1;
  int  minFilter                            = -1;
  int  wrapS                                = 10497;  // REPEAT
  int  wrapT                                = 10497;
  bool operator==(const SamplerDesc&) const = default;
};

// One triangle geometry: float3 positions at vertexAddress with vertexStride, uint32 indices
struct BlasInput
{
  uint64_t vertexAddress                      = 0;
  uint32_t vertexStride                       = 12;
  uint32_t vertexCount                        = 0;
  uint64_t indexAddress                       = 0;
  uint32_t indexCount                         = 0;
  bool     opaque                             = true;
  bool     operator==(const BlasInput&) const = default;
};

enum AccelBuildBits : uint32_t
{
  eAccelPreferFastTrace = 1u << 0,
  eAccelAllowUpdate     = 1u << 1,
  eAccelAllowCompaction = 1u << 2,
};
using AccelBuildFlags = uint32_t;

// A TLAS instance row; a hole is a row with no BLAS and mask 0
struct TlasRow
{
  float       transform[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};  // object to world, row major
  uint32_t    customIndex     = 0;                                           // 24 bits
  uint8_t     mask            = 0;
  uint32_t    flags           = 0;  // VkGeometryInstanceFlagsKHR bits
  AccelHandle blas;
};

enum class TlasMode
{
  eBuild,
  eUpdate,  // only with the rows' count and BLASes of the previous build
};

// The execution dependencies the residency layer asks for between its passes
enum class GpuBarrier
{
  eTransferToShader,  // uploads and copies, then shading and the compute passes
  eTransferToBuild,   // uploads and copies, then AS builds
  eComputeToBuild,    // deform and transform passes, then AS builds
  eBuildToBuild,      // BLAS builds, then the TLAS build
  eBuildToTrace,      // AS builds, then ray tracing and raster
};

enum class DeformKind
{
  eMorph,
  eSkin,
};

// One deform pass over a mesh's vertices; the buffers bind in order
struct DeformDispatch
{
  DeformKind                kind = DeformKind::eMorph;
  std::vector<BufferHandle> inputs;
  BufferHandle              output;
  uint32_t                  vertexCount = 0;
};

class GpuBackend
{
public:
  virtual ~GpuBackend() = default;

  // ---------- Frame ----------
  // Starts the next frame: runs the retire closures of the frame that last used this frame's slot in
  // the ring, whose work has now completed.
  virtual void     beginFrame()  = 0;
  virtual uint64_t frame() const = 0;
  // Runs once no frame in flight can still use what the closure frees (a frame cycle later).
  virtual void retire(std::function<void()> closure) = 0;
  // A full idle wait; the retained path calls it only on scene replace and teardown.
  virtual void waitIdle() = 0;

  // ---------- Buffers ----------
  virtual BufferHandle createBuffer(const BufferDesc& desc) = 0;
  // Now: from a retire closure for a buffer a frame in flight may still read.
  virtual void     destroyBuffer(BufferHandle buffer)                                          = 0;
  virtual uint64_t bufferAddress(BufferHandle buffer) const                                    = 0;
  virtual uint64_t bufferSize(BufferHandle buffer) const                                       = 0;
  virtual void     upload(BufferHandle dst, uint64_t offset, std::span<const std::byte> bytes) = 0;
  virtual void copyBuffer(BufferHandle src, uint64_t srcOffset, BufferHandle dst, uint64_t dstOffset, uint64_t size) = 0;

  // ---------- Images and descriptors ----------
  // Level 0 from the pixels, the others generated.
  virtual ImageHandle createImage(const ImageDesc& desc, std::span<const std::byte> pixels) = 0;
  virtual void        destroyImage(ImageHandle image)                                       = 0;
  // A slot of the scene's texture array; no image writes the default one.
  virtual void writeTextureDescriptor(uint32_t slot, ImageHandle image)       = 0;
  virtual void writeSamplerDescriptor(uint32_t slot, const SamplerDesc& desc) = 0;

  // ---------- Acceleration structures ----------
  // A new BLAS, its build recorded in this frame.
  virtual AccelHandle buildBlas(const BlasInput& input, AccelBuildFlags flags) = 0;
  // In place; the BLAS was built with eAccelAllowUpdate, not compacted, and the input keeps its
  // addresses and index count.
  virtual void refitBlas(AccelHandle blas, const BlasInput& input) = 0;
  // A compacted copy of a BLAS built with eAccelAllowCompaction; the caller retires the original.
  virtual AccelHandle compactBlas(AccelHandle blas) = 0;
  // A TLAS of up to `capacity` rows; its handle stays until it is destroyed.
  virtual AccelHandle createTlas(uint32_t capacity)                                             = 0;
  virtual void        buildTlas(AccelHandle tlas, std::span<const TlasRow> rows, TlasMode mode) = 0;
  virtual void        destroyAccel(AccelHandle accel)                                           = 0;

  // ---------- Passes ----------
  virtual void dispatchDeform(const DeformDispatch& dispatch) = 0;
  virtual void barrier(GpuBarrier barrier)                    = 0;
};

}  // namespace rs
