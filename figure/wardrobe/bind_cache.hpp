#pragma once

// Surface bindings kept on disk. A binding is a pure function of the arrays it is made from (the
// body's unmorphed cage, the item's fitted cage, a graft's pairs, the limb filter, a strand's lines
// and reach), so it is stored under a hash of exactly those and of bind.cpp's source, and binding
// the same item to the same body again reads it back instead of searching: about a second for a
// strand hair, on every scene open after the first. Any change to an input, or to the binding code,
// is another key. Files that do not read back whole are misses.

#include <cstdint>
#include <filesystem>
#include <span>

#include <figure/wardrobe/bind.hpp>

namespace fv::wardrobe {

// Where the files go (created on first write); empty, the default, turns the cache off.
void                  setBindCacheDir(const std::filesystem::path& dir);
std::filesystem::path bindCacheDir();

// bindToSurface and bindStrands through the cache.
SurfaceBinding cachedBindToSurface(std::span<const float>    bodyPositions,
                                   std::span<const uint32_t> bodyTris,
                                   std::span<const float>    itemPositions,
                                   std::span<const int32_t>  pairs  = {},
                                   const BindFilter*         filter = nullptr);
SurfaceBinding cachedBindStrands(std::span<const float>    bodyPositions,
                                 std::span<const uint32_t> bodyTris,
                                 std::span<const float>    itemPositions,
                                 std::span<const int32_t>  lineStart,
                                 std::span<const int32_t>  lineVerts,
                                 float                     reach,
                                 const BindFilter*         filter = nullptr);

// Reads and writes since start-up, for tests and the timing log.
struct BindCacheCounts
{
  uint64_t hits = 0, misses = 0;
};
BindCacheCounts bindCacheCounts();

}  // namespace fv::wardrobe
