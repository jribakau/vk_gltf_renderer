#pragma once

// A garment's addresses onto a body (docs/wardrobe-plan.md, step 4a; "Decode"), and the cloth the
// addresses don't hold settled round what they do.

#include <span>
#include <vector>

#include <figure/bodymap/clone.hpp>
#include <figure/garment/format.hpp>

namespace fv::garment {

// The body a garment is decoded onto.
struct Target
{
  // The reference's topology as it lies on the target: Genesis 9's base as the figure has it, or, for
  // another type, the reference laid onto that type (its map's TypeMap::reference).
  const bodymap::Body* reference = nullptr;
  // The target's own skin, when it isn't the reference's topology (a Genesis 8 figure); null for the
  // reference itself. An address's place on the laid reference lies a hair off this skin, and is put
  // on it: the height is kept over the real skin.
  const bodymap::Body* body = nullptr;
  // Per vertex of `body`, the turn from its frame to the reference base's (its map's frameTurn); null,
  // or empty, for none.
  const std::vector<Quat>* frameTurn = nullptr;
};

struct Decoded
{
  // xyz per cage vertex; a vertex its address doesn't hold is left at the origin
  std::vector<float> positions;
  // per cage vertex, its place on the target's own skin: three of the target's vertices and their
  // weights (all zero where nothing is held), the binding its address names
  wardrobe::SurfaceBinding at;
  // the addresses' hold, less what found no skin on the target
  std::vector<float> hold;
};

// Each held vertex on the target: the address's place on its skin, the height along its smooth normal
// there, and the offset across, turned into the target's frame.
Decoded decode(const Addresses& garment, const Target& target);
// The same onto a body of the reference's topology.
Decoded decode(const Addresses& garment, const bodymap::Body& reference);

// `bound` (a mesh's binding to the body by closest points) with the places decode named instead,
// wherever `placed` has one (its weights not all zero): cloth in a crease stays on the side its
// address is on.
wardrobe::SurfaceBinding withPlaces(const wardrobe::SurfaceBinding& bound, const wardrobe::SurfaceBinding& placed);

struct SettleOptions
{
  int   smooth = 24;  // rounds the cloth's change is evened over its neighbours
  float cap    = 12;  // cm: the most a vertex's address may move it from where the bones put it
  // an edge between two held vertices pulled to this many times its length (or as much shorter) starts
  // to loosen their hold, and at `tearTo` they hold nothing: its ends lie either side of a crease
  float tearFrom = 1.3f, tearTo = 2.0f;
  int   mendPasses = 6;  // times mend looks for edges its last evening left torn
};

// The bone transfer's cage with its tears mended. `made` is the garment as it was made, `moved` as the
// bones put it on the target's skeleton. Where an edge comes out `tearTo` times its made length or
// more (or as much shorter), its ends were carried by bones that went different ways: a dress's centre
// seam between the legs, each side on its own thigh, which cross where the target stands narrower. Such
// vertices take the middle of how far their neighbours moved instead, and those that leaves torn after
// them, so the cloth keeps its shape across the seam.
std::vector<float> mend(std::span<const float> made, std::span<const float> moved, std::span<const uint32_t> tris, const SettleOptions& options = {});

// The fitted cage. `moved` is the garment as the bone transfer put it on the target's skeleton,
// `decoded` and `hold` its addresses' answer. A vertex its address holds goes where that says; one it
// holds in part, part of the way; cloth it doesn't hold (a hem, a cape) takes the change of the held
// cloth nearest it along the mesh, evened out, so a skirt moves with its waist and keeps its own shape.
// Cloth joined to nothing held stays where the bones put it. Held cloth that bridges a crease, each
// side's address pulling its own way, is let go as far as the pull is hard (tearFrom, tearTo).
std::vector<float> settle(std::span<const float> moved, std::span<const float> decoded, std::span<const float> hold, std::span<const uint32_t> tris,
                          const SettleOptions& options = {});

}  // namespace fv::garment
