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
#include <string>
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
using AccelHandle  = GpuHandle<GpuAccelTag>;  // a BLAS, a TLAS or an opacity micromap

enum BufferUsageBits : uint32_t
{
  eBufferVertex     = 1u << 0,
  eBufferIndex      = 1u << 1,
  eBufferStorage    = 1u << 2,
  eBufferAccelInput = 1u << 3,  // read by a BLAS build (positions, indices, micromap indices)
};
using BufferUsageFlags = uint32_t;

struct BufferDesc
{
  uint64_t         size  = 0;
  BufferUsageFlags usage = 0;
  const char*      name  = nullptr;  // debug name
};

// An image as it came, for the backend to decode: a file, encoded bytes (a GLB's, a data URI's), or
// pixels a loader already decoded
struct ImageSource
{
  std::string                    uri;  // an absolute path or a data URI
  std::span<const unsigned char> bytes;
  std::string                    mimeType;
  std::span<const unsigned char> pixels;  // width x height x component channels of `bits` bits
  int                            width = -1, height = -1, component = 0, bits = 8;
  bool                           srgb = false;  // sampled as colour
  std::string                    name;
};

// What decoding found, for the materials that sample the image
struct ImageInfo
{
  uint32_t width = 0, height = 0;
  bool     twoChannel    = false;  // exactly two colour channels (BC5, R8G8, ...): a normal map's X and Y
  float    meanLuminance = -1.f;   // linear, Rec. 709, of its texels; negative where unknown
  uint64_t bytes         = 0;      // the GPU memory it holds (the residency's grace budget)
};

// glTF sampler values (TINYGLTF_TEXTURE_FILTER_*, TINYGLTF_TEXTURE_WRAP_*); a filter of -1 is unset
struct GpuSamplerDesc
{
  int  magFilter                               = -1;
  int  minFilter                               = -1;
  int  wrapS                                   = 10497;  // REPEAT
  int  wrapT                                   = 10497;
  bool operator==(const GpuSamplerDesc&) const = default;
};

// One triangle geometry: float3 positions at vertexAddress with vertexStride, uint32 indices; and
// the opacity micromap its triangles take their opacity from, if any: triangle i takes the
// micromap's triangle micromapBaseTriangle + the i-th micromap index (micromapIndexSize bytes each
// at micromapIndexAddress), or micromapBaseTriangle + i with no index buffer
struct BlasInput
{
  uint64_t    vertexAddress = 0;
  uint32_t    vertexStride  = 12;
  uint32_t    vertexCount   = 0;
  uint64_t    indexAddress  = 0;
  uint32_t    indexCount    = 0;
  bool        opaque        = true;
  AccelHandle micromap;
  uint64_t    micromapIndexAddress               = 0;
  uint32_t    micromapIndexSize                  = 0;  // 1, 2 or 4
  uint32_t    micromapBaseTriangle               = 0;
  bool        operator==(const BlasInput&) const = default;
};

// An opacity micromap's build inputs, in VK_KHR_opacity_micromap's layout: the packed opacity bits,
// the per-triangle records (VkMicromapTriangleKHR, triangleStride bytes apart) and the usage
// histogram
struct MicromapUsage
{
  uint32_t count = 0, subdivisionLevel = 0, format = 0;  // VkMicromapUsageKHR
  bool     operator==(const MicromapUsage&) const = default;
};
struct MicromapInput
{
  std::span<const unsigned char> data, triangles;
  uint32_t                       triangleStride = 8;
  std::span<const MicromapUsage> usages;
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
  eComputeToBuild,    // deform and transform passes, then AS builds and the frame's shading
  eComputeToCompute,  // a deform pass, then the deform pass that reads what it wrote (morph, then skin)
  eBuildToBuild,      // BLAS builds, then the TLAS build
  eBuildToTrace,      // AS builds, then ray tracing and raster
  // what the frames still in flight read, then this frame's uploads, copies and deform passes that
  // overwrite it (a vertex stream, a table row): recorded before the first of them
  eFrameToWrite,
};

enum class DeformKind
{
  eMorph,
  eSkin,
};

// A range of a buffer a pass reads or writes from `offset` on; no buffer: none (the pass skips it)
struct BufferRange
{
  BufferHandle buffer;
  uint64_t     offset = 0;
  bool         operator==(const BufferRange&) const = default;
};

// One deform pass over a mesh's vertices, as shaders/morph.comp.slang and skinning.comp.slang take
// it (animation_io.h.slang). A morph adds each target's weighted deltas (targetCount blocks of
// vertexCount deltas, target-major) to the base; a skin blends four joints per vertex. Normals and
// tangents are written only where both the base and the output name a buffer.
struct DeformDispatch
{
  DeformKind  kind        = DeformKind::eMorph;
  uint32_t    vertexCount = 0;
  BufferRange basePositions, baseNormals, baseTangents;  // float3, float3, float4
  BufferRange outPositions, outNormals, outTangents;
  // morph: float3 deltas, and the targets' float weights
  BufferRange positionDeltas, normalDeltas, tangentDeltas, morphWeights;
  uint32_t    targetCount = 0;
  // skin: int4 joints and float4 weights per vertex, then the joints' float4x4 matrices and their
  // normal matrices as the shader reads float3x3 (36 bytes each, packed)
  BufferRange joints, weights, jointMatrices, normalMatrices;
  uint32_t    jointCount = 0;
};
// The packed normal matrices are glm::mat3 on the host and float3x3 in the shader: 36 bytes apart
inline constexpr uint64_t kNormalMatrixStride = 36;

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
  // The image decoded, its mips made; none when it can't be read (its slot keeps the default image).
  virtual ImageHandle createImage(const ImageSource& source, ImageInfo& info) = 0;
  // Several at once, in order (a backend may decode them side by side); `images` and `infos` as
  // long as `sources`
  virtual void createImages(std::span<const ImageSource> sources, std::span<ImageHandle> images, std::span<ImageInfo> infos)
  {
    for(size_t i = 0; i < sources.size(); i++)
      images[i] = createImage(sources[i], infos[i]);
  }
  virtual void destroyImage(ImageHandle image) = 0;
  // A slot of the scene's texture array; no image writes the default one.
  virtual void writeTextureDescriptor(uint32_t slot, ImageHandle image)          = 0;
  virtual void writeSamplerDescriptor(uint32_t slot, const GpuSamplerDesc& desc) = 0;

  // ---------- Acceleration structures ----------
  // Whether the device builds opacity micromaps (VK_KHR_opacity_micromap); without, no BLAS links one.
  virtual bool opacityMicromaps() const = 0;
  // A new opacity micromap, its inputs uploaded and its build recorded in this frame ahead of any
  // BLAS build that links it; outlives every BLAS that does.
  virtual AccelHandle buildMicromap(const MicromapInput& input) = 0;
  // A new BLAS, its build recorded in this frame.
  virtual AccelHandle buildBlas(const BlasInput& input, AccelBuildFlags flags) = 0;
  // In place; the BLAS was built with eAccelAllowUpdate, not compacted, and the input keeps its
  // addresses and index count.
  virtual void refitBlas(AccelHandle blas, const BlasInput& input) = 0;
  // A compacted copy of a BLAS built with eAccelAllowCompaction, recorded in this frame; the caller
  // retires the original. None while its compacted size isn't known (the frame that built it is in
  // flight): ask again in a later frame. The BLAS itself when it stays as built (no smaller, or no
  // size to read).
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
