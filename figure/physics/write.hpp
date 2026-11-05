#pragma once

// How a still pass's result goes into the figure: a hidden morph channel on one mesh, weight 1 (less
// while it fades in or out), holding the cage deltas whose skinning gives the result
// (dforce::drapeDelta's). The renderer,
// subdivision and export see it as any other morph. A delta under kMinMove is left out, and a result
// that matches what the channel holds is not written at all: a write flushes the mesh and a refined
// mesh reports its whole cage, so a pass that changes nothing costs nothing.

#include <span>
#include <string>

#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>

namespace fv::physics {

// A correction below this (centimetres, summed over x, y and z) is not worth a morph.
constexpr float kMinMove = 1e-3f;

struct ChannelWriter
{
  std::string name, label;
  int         mesh = -1;

  // Writes `delta` (xyz per cage vertex, mesh-local) at `weight`. True when it wrote: the set has
  // vertices to flush.
  bool write(morph::MorphSet& morphs, const mesh::BuiltMesh& geo, std::span<const float> delta, double weight = 1) const;
  // The channel at `weight`, its target kept. True when that changed it (the caller flushes).
  bool weigh(morph::MorphSet& morphs, double weight) const;
  // Weight 0, the target kept. True when it was on.
  bool off(morph::MorphSet& morphs) const { return weigh(morphs, 0); }
};

}  // namespace fv::physics
