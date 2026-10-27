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
// rs::GpuBackend without a GPU, for the residency tests (docs: daz-viewer
// docs/retained-scene-plan.md, "The seam"). It keeps what the GPU would hold, so a test (and from
// P3a1 the ResidencyOracle) can read it back:
//  - shadow bytes per buffer, at fake device addresses from an OffsetAllocator (16-byte units);
//  - each BLAS's input and a hash of the bytes it was built or refitted from;
//  - each TLAS's rows and how it was last built;
//  - the texture and sampler descriptor tables, barriers and deform dispatches in call order;
//  - a frame clock with N frames in flight that runs a retire closure N frames after it was queued,
//    as nvapp::Application's free queue does; and the waitIdle calls.
// Misuse (a dead handle, a range past a buffer's end, an invalid refit or TLAS update) is recorded
// in errors() rather than thrown, so a test can assert there was none.
//

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "scene_gpu_backend.hpp"

namespace OffsetAllocator {
class Allocator;
}

namespace rs {

class FakeGpuBackend : public GpuBackend
{
public:
  struct Buffer
  {
    BufferDesc             desc;
    uint64_t               address = 0;
    std::vector<std::byte> bytes;
    uint32_t               allocOffset   = 0;  // the OffsetAllocator allocation behind the address
    uint32_t               allocMetadata = 0;
  };
  struct Image
  {
    std::string name, uri;
    bool        srgb = false;
    ImageInfo   info;
  };
  struct Blas
  {
    BlasInput       input;
    AccelBuildFlags flags       = 0;
    uint64_t        contentHash = 0;  // positions and indices at the last build or refit
    bool            compacted   = false;
    uint32_t        builds      = 0;
    uint32_t        refits      = 0;
  };
  struct Tlas
  {
    uint32_t             capacity = 0;
    std::vector<TlasRow> rows;
    TlasMode             lastMode = TlasMode::eBuild;
    uint32_t             builds   = 0;
    uint32_t             updates  = 0;
  };

  explicit FakeGpuBackend(uint32_t framesInFlight = 3);
  ~FakeGpuBackend() override;

  // GpuBackend
  void         beginFrame() override;
  uint64_t     frame() const override { return m_frame; }
  void         retire(std::function<void()> closure) override;
  void         waitIdle() override;
  BufferHandle createBuffer(const BufferDesc& desc) override;
  void         destroyBuffer(BufferHandle buffer) override;
  uint64_t     bufferAddress(BufferHandle buffer) const override;
  uint64_t     bufferSize(BufferHandle buffer) const override;
  void         upload(BufferHandle dst, uint64_t offset, std::span<const std::byte> bytes) override;
  void copyBuffer(BufferHandle src, uint64_t srcOffset, BufferHandle dst, uint64_t dstOffset, uint64_t size) override;
  // "decodes" what it is given: width and height as the source says, two channels for two
  // components; a source with nothing in it, or a uri starting "missing", can't be read
  ImageHandle createImage(const ImageSource& source, ImageInfo& info) override;
  void        destroyImage(ImageHandle image) override;
  void        writeTextureDescriptor(uint32_t slot, ImageHandle image) override;
  void        writeSamplerDescriptor(uint32_t slot, const GpuSamplerDesc& desc) override;
  AccelHandle buildBlas(const BlasInput& input, AccelBuildFlags flags) override;
  void        refitBlas(AccelHandle blas, const BlasInput& input) override;
  AccelHandle compactBlas(AccelHandle blas) override;
  AccelHandle createTlas(uint32_t capacity) override;
  void        buildTlas(AccelHandle tlas, std::span<const TlasRow> rows, TlasMode mode) override;
  void        destroyAccel(AccelHandle accel) override;
  void        dispatchDeform(const DeformDispatch& dispatch) override;
  void        barrier(GpuBarrier barrier) override;

  // What the GPU holds; nullptr for a dead or unknown handle
  const Buffer* buffer(BufferHandle buffer) const;
  const Image*  image(ImageHandle image) const;
  const Blas*   blas(AccelHandle blas) const;
  const Tlas*   tlas(AccelHandle tlas) const;
  // The live buffer bytes [address, address + size) fall in, or an empty span when no single live
  // buffer holds the whole range
  std::span<const std::byte> bytesAt(uint64_t address, uint64_t size) const;
  // The hash a BLAS built from this input now would get; 0 when its ranges don't resolve
  uint64_t contentHash(const BlasInput& input) const;

  const std::map<uint32_t, ImageHandle>&    textureDescriptors() const { return m_textureDescriptors; }
  const std::map<uint32_t, GpuSamplerDesc>& samplerDescriptors() const { return m_samplerDescriptors; }
  const std::vector<GpuBarrier>&            barriers() const { return m_barriers; }
  const std::vector<DeformDispatch>&        deforms() const { return m_deforms; }
  uint32_t                                  framesInFlight() const { return m_framesInFlight; }
  uint32_t                                  waitIdleCount() const { return m_waitIdleCount; }
  size_t                                    pendingRetires() const { return m_retires.size(); }
  size_t                                    liveBuffers() const { return m_buffers.size(); }
  size_t                                    liveImages() const { return m_images.size(); }
  size_t                                    liveAccels() const { return m_blases.size() + m_tlases.size(); }
  const std::vector<std::string>&           errors() const { return m_errors; }
  // What the work recorded so far leaves unfenced for the frame's shading and tracing: uploads with
  // no transfer-to-shader barrier after them, builds with no build-to-trace barrier after them. (A
  // build that reads unfenced bytes, or a TLAS build over unfenced BLAS builds, is an error at once.)
  std::vector<std::string> unfencedForTheFrame() const;

  // The fake's address range: 16-byte units in a 32-bit allocator, from a base that no real buffer
  // address in a test could be confused with
  static constexpr uint64_t kAddressBase = 0x1000'0000'0000ull;
  static constexpr uint64_t kAddressUnit = 16;

private:
  struct Retire
  {
    uint64_t              frame;
    std::function<void()> closure;
  };

  void    error(std::string message);
  Buffer* liveBuffer(BufferHandle handle, const char* op);
  void    written(BufferHandle handle);
  // a build reading [address, address + size): an error when its buffer was written unfenced
  void readByBuild(uint64_t address, const char* op);

  uint32_t m_framesInFlight;
  uint64_t m_frame         = 0;
  uint64_t m_nextHandle    = 1;
  uint32_t m_waitIdleCount = 0;

  std::unique_ptr<OffsetAllocator::Allocator> m_addresses;
  std::unordered_map<uint64_t, Buffer>        m_buffers;
  std::unordered_map<uint64_t, Image>         m_images;
  std::unordered_map<uint64_t, Blas>          m_blases;
  std::unordered_map<uint64_t, Tlas>          m_tlases;
  std::map<uint32_t, ImageHandle>             m_textureDescriptors;
  std::map<uint32_t, GpuSamplerDesc>          m_samplerDescriptors;
  std::vector<GpuBarrier>                     m_barriers;
  // buffers written since the last barrier that fences them for builds, and for shading; BLAS
  // builds or refits since the last build-to-build, and AS builds since the last build-to-trace
  std::set<uint64_t>                          m_writtenBeforeBuild, m_writtenBeforeShading;
  bool                                        m_blasBeforeTlas = false, m_buildsBeforeTrace = false;
  std::vector<DeformDispatch>                 m_deforms;
  std::vector<Retire>                         m_retires;
  std::vector<std::string>                    m_errors;
};

}  // namespace rs
