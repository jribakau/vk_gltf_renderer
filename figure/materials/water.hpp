#pragma once

// Water lying on a surface: beads, and stretches of the track a run left, each a dome over the skin in
// the surface's own UVs, and how damp the surface is all over. Rain on a figure's skin makes it
// (figure/physics/rain/rain_sim.hpp); the renderer keeps it beside the surface's material, looks it up
// at a hit (shaders/water.h.slang) and lays it over the surface as the sweat layer's beads and film are
// (gltf_material_eval.h.slang). It is not part of the material: a pose moves it, which no material edit
// should cost.

#include <array>
#include <vector>

#include <figure/materials/surface.hpp>

namespace fv::mat {

// A dome over the skin: a bead (a == b), or a stretch of track from a to b, a ridge whose cross section
// is the dome's. Its outline is round on the skin whatever the UVs do there: `toTangent` (row-major)
// takes an offset in UV to metres along the surface's tangent and bitangent as the renderer makes them,
// the tangent along +u and the bitangent along -v (shaders/vertex_tangents.comp.slang).
struct WaterItem
{
  std::array<float, 2> a{}, b{};
  std::array<float, 4> toTangent{};
  float                radius = 0;  // m: a bead's contact radius, a track's half width
  float                height = 0;  // m: its top over the skin

  bool operator==(const WaterItem&) const = default;
};

struct WaterLayer
{
  std::vector<WaterItem> items;
  float                  damp = 0;  // how wet the surface is all over, 0..1

  bool operator==(const WaterLayer&) const = default;
};

// The same water for UVs whose v runs the other way (1 - v), as a mesh goes to the renderer when its
// textures' rows aren't flipped (FigureFeed): v and the bitangent both turn, so `toTangent`'s diagonal
// stays and the two entries off it change sign.
WaterLayer flippedV(const WaterLayer& layer);

// Whether water shows on a surface: a figure's skin and its nails.
bool takesWater(const Surface& s);

}  // namespace fv::mat
