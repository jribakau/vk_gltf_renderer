#pragma once

// A garment's cage to addresses on the reference (docs/wardrobe-plan.md, step 4a). Each vertex is bound
// to the closest outer skin of the body it was made on, within its limb (as the old fitter binds a
// sleeve to its arm), and its offset from that skin is measured there, on the real body: a cloth 2 mm
// off the source's skin is 2 mm off the target's. The place itself then goes through the type's map
// onto the reference, where a few millimetres' slide moves the cloth along the skin and not into it.

#include <span>

#include <figure/bodymap/clone.hpp>
#include <figure/garment/format.hpp>

namespace fv::garment {

struct ConvertOptions
{
  // a vertex within `inner` cm of the skin is all its address's, beyond `outer` none of it
  float inner = 2, outer = 5;
};

// `item` (xyz) is the garment's cage in the rest of the body it was made on, `type` that body's map.
// `filter` keeps a vertex to its limb's skin (its `bodyLimb` per vertex of the type's body). A vertex
// the source has no skin for, or whose skin has no place on the reference, holds nothing.
Addresses convert(std::span<const float> item, const bodymap::TypeMap& type, const wardrobe::BindFilter* filter = nullptr, const ConvertOptions& options = {});

}  // namespace fv::garment
