#pragma once

// Binding a worn item to the figure's body for auto-follow: every item vertex to the closest point
// on the body's unmorphed triangles (a barycentric position on one). A port of
// src/lib/loaders/dson/bind.ts.

#include <cstdint>
#include <span>
#include <vector>

namespace fv::wardrobe {

struct SurfaceBinding
{
  // per item cage vertex: three body cage vertices...
  std::vector<uint32_t> verts;
  // ...and their barycentric weights
  std::vector<float> weights;
  // body vertices some binding reads: a morph that moves none of them leaves the item alone
  std::vector<uint8_t> used;
};

// Restricts which body triangles an item vertex may bind to. `bodyLimb` is a limb id per body
// vertex (0 = anywhere); `itemMask` a bitmask of acceptable limb ids per item vertex (0 = any). A
// triangle is acceptable when one of its vertices has an allowed limb.
struct BindFilter
{
  std::vector<int16_t> bodyLimb;
  std::vector<int32_t> itemMask;
};

// `bodyPositions` (xyz) and `bodyTris` are the body's unmorphed cage, `itemPositions` the item's.
// `pairs` are (item vertex, body vertex) couples bound to each other outright (a geograft's seam).
SurfaceBinding bindToSurface(std::span<const float>    bodyPositions,
                             std::span<const uint32_t> bodyTris,
                             std::span<const float>    itemPositions,
                             std::span<const int32_t>  pairs  = {},
                             const BindFilter*         filter = nullptr);

}  // namespace fv::wardrobe
