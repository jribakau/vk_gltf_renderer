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

// The same for strand hair, polyline by polyline (`lineStart`/`lineVerts`, root first): each root is
// bound as above, and each later point to the closest body point if one lies within `reach` cm (the
// same answer the search above gives there), else as the point above it on its strand. Hair lying on
// the body follows the body under it; a strand hanging away from it moves with where it last touched,
// and the search for those points stops at `reach` instead of widening until it meets the body.
SurfaceBinding bindStrands(std::span<const float>    bodyPositions,
                           std::span<const uint32_t> bodyTris,
                           std::span<const float>    itemPositions,
                           std::span<const int32_t>  lineStart,
                           std::span<const int32_t>  lineVerts,
                           float                     reach,
                           const BindFilter*         filter = nullptr);

}  // namespace fv::wardrobe
