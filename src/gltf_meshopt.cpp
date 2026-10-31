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

#include "gltf_meshopt.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#include <meshoptimizer/src/meshoptimizer.h>
#include <nvutils/logger.hpp>

#include "tinygltf_utils.hpp"

bool nvvkgltf::decompressMeshopt(tinygltf::Model& model)
{
  bool hasKHR = std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), KHR_MESHOPT_COMPRESSION_EXTENSION_NAME)
                != model.extensionsUsed.end();
  bool hasEXT = std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), EXT_MESHOPT_COMPRESSION_EXTENSION_NAME)
                != model.extensionsUsed.end();
  if(!hasKHR && !hasEXT)
  {
    return true;
  }

  for(tinygltf::Buffer& buffer : model.buffers)
  {
    if(buffer.data.empty())
    {
      buffer.data.resize(buffer.byteLength);
      buffer.extensions.erase(KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
      buffer.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
    }
  }

  // First used to tag buffers that can be removed after decompression
  std::vector<int> isFullyCompressedBuffer(model.buffers.size(), 1);

  for(auto& bufferView : model.bufferViews)
  {
    if(bufferView.buffer < 0)
      continue;

    bool warned = false;

    KHR_meshopt_compression mcomp;
    if(tinygltf::utils::getMeshoptCompression(bufferView, mcomp))
    {
      // Decoding logic derived from `decompressMeshopt` in meshoptimizer/gltf/parsegltf.cpp

      const tinygltf::Buffer& sourceBuffer = model.buffers[mcomp.buffer];
      const unsigned char*    source       = &sourceBuffer.data[mcomp.byteOffset];
      assert(mcomp.byteOffset + mcomp.byteLength <= sourceBuffer.data.size());

      tinygltf::Buffer& resultBuffer = model.buffers[bufferView.buffer];
      unsigned char*    result       = &resultBuffer.data[bufferView.byteOffset];
      assert(bufferView.byteOffset + bufferView.byteLength <= resultBuffer.data.size());

      int  rc   = -1;
      bool warn = false;

      switch(mcomp.compressionMode)
      {
        case KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_ATTRIBUTES:
          warn = meshopt_decodeVertexVersion(source, mcomp.byteLength) < 0;
          rc   = meshopt_decodeVertexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_TRIANGLES:
          warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) < 0;
          rc   = meshopt_decodeIndexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_INDICES:
          warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) < 0;
          rc   = meshopt_decodeIndexSequence(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
          break;

        default:
          break;
      }

      if(rc != 0)
      {
        LOGW("meshopt_compression decompression failed\n");
        return false;
      }

      if(warn && !warned)
      {
        LOGW("Warning: meshopt_compression data uses an unsupported or invalid encoding version\n");
        warned = true;
      }

      switch(mcomp.compressionFilter)
      {
        case KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_OCTAHEDRAL:
          meshopt_decodeFilterOct(result, mcomp.count, mcomp.byteStride);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_QUATERNION:
          meshopt_decodeFilterQuat(result, mcomp.count, mcomp.byteStride);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_EXPONENTIAL:
          meshopt_decodeFilterExp(result, mcomp.count, mcomp.byteStride);
          break;

        default:
          break;
      }

      bufferView.extensions.erase(KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
      bufferView.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
    }

    isFullyCompressedBuffer[bufferView.buffer] = 0;
  }

  // Remove fully compressed buffers; isFullyCompressedBuffer is repurposed as buffer index remap table
  size_t writeIndex = 0;
  for(size_t readIndex = 0; readIndex < model.buffers.size(); readIndex++)
  {
    if(isFullyCompressedBuffer[readIndex])
    {
      isFullyCompressedBuffer[readIndex] = -1;
    }
    else
    {
      isFullyCompressedBuffer[readIndex] = static_cast<int>(writeIndex);
      if(readIndex != writeIndex)
      {
        model.buffers[writeIndex] = std::move(model.buffers[readIndex]);
      }
      writeIndex++;
    }
  }
  model.buffers.resize(writeIndex);

  for(auto& bufferView : model.bufferViews)
  {
    if(bufferView.buffer >= 0)
    {
      bufferView.buffer = isFullyCompressedBuffer[bufferView.buffer];
    }
  }

  std::erase(model.extensionsRequired, KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
  std::erase(model.extensionsUsed, KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
  std::erase(model.extensionsRequired, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
  std::erase(model.extensionsUsed, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
  return true;
}
