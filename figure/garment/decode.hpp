#pragma once

// A garment's addresses onto a body (docs/wardrobe-plan.md, step 4a; "Decode"), and the cloth the
// addresses don't hold settled round what they do.
//
// Not here yet: a target of another topology than the reference's (a Genesis 9 dress on Genesis 8),
// which goes through that type's map.

#include <span>
#include <vector>

#include <figure/bodymap/clone.hpp>
#include <figure/garment/format.hpp>

namespace fv::garment {

// Each held vertex on `target`, a body of the reference's topology in its own shape (Genesis 9's base as
// the figure has it): the address's place on its skin, the height along its smooth normal there, and
// the offset across. xyz per cage vertex; a vertex holding nothing is left at the origin.
std::vector<float> decode(const Addresses& garment, const bodymap::Body& target);

struct SettleOptions
{
  int   smooth = 24;  // rounds the cloth's change is evened over its neighbours
  float cap    = 12;  // cm: the most a vertex's address may move it from where the bones put it
  // an edge between two held vertices pulled to this many times its length (or as much shorter) starts
  // to loosen their hold, and at `tearTo` they hold nothing: its ends lie either side of a crease
  float tearFrom = 1.3f, tearTo = 2.0f;
};

// The fitted cage. `moved` is the garment as the bone transfer put it on the target's skeleton,
// `decoded` and `hold` its addresses' answer. A vertex its address holds goes where that says; one it
// holds in part, part of the way; cloth it doesn't hold (a hem, a cape) takes the change of the held
// cloth nearest it along the mesh, evened out, so a skirt moves with its waist and keeps its own shape.
// Cloth joined to nothing held stays where the bones put it. Held cloth that bridges a crease, each
// side's address pulling its own way, is let go as far as the pull is hard (tearFrom, tearTo).
std::vector<float> settle(std::span<const float> moved, std::span<const float> decoded, std::span<const float> hold, std::span<const uint32_t> tris,
                          const SettleOptions& options = {});

}  // namespace fv::garment
