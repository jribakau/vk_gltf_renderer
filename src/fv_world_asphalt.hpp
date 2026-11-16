#pragma once

//-------------------------------------------------------------------------------------------------------
// An asphalt road's maps, made from a seed (no kit, no files): an old road's wear as it lies across
// its carriageway and along it, the grain of its stones, and its markings' look on it.
//
// - The wear, kAsphaltWidth metres across (a road's carriageway from edge to edge is laid on that,
//   whatever its width) and kAsphaltRepeat along before it repeats: binder greyed by the weather
//   with its stones showing; paler, smoother and a little sunk where the wheels went, oil between
//   them; dust, crumbs and broken asphalt at the edges; cracks across it, one down its middle where
//   it was laid in two halves and some along the wheel paths, some of them sealed with tar; crazing
//   in a wheel path; repairs laid over it; potholes. Its colour, its normals and its roughness (and
//   in the roughness map's red, how much of the grain shows: little on tar). It is cut into stretches
//   by joints (kAsphaltJoints), straight cracks from one edge to the other, some of them sealed, as an
//   old road over a jointed base cracks; nothing crosses one, so a road may lay the stretches in any
//   order, each either way along and either way across, cut at the joints (fv_world_roads.cpp).
// - The grain: the stones' relief, a normal map tiled every kAsphaltGrain metres both ways, to lay
//   over the wear's (a material's detail layer).
// - The markings': the paint's colour where it is whole, the asphalt's where it is worn through,
//   cracked where the road is, and gone over the repairs and the tar (laid after it). Over the whole
//   of the wear, so a crack runs on through a line.
//
// The maps' coordinates are a road's (fv_world_roads.hpp, roadUv): u across from one edge to the
// other, v along, a repeat a unit. Row 0 is v = 0, and the normals are glTF's (+X toward u, +Y toward
// v = 0).
//
// Vulkan-free.
//-------------------------------------------------------------------------------------------------------

#include <array>
#include <cstdint>
#include <vector>

namespace fv::world {

// Metres across that the wear is laid out for; along it before it repeats; a repeat of the grain
inline constexpr double kAsphaltWidth  = 6.0;
inline constexpr double kAsphaltRepeat = 24.0;
inline constexpr double kAsphaltGrain  = 1.5;
// Its joints, metres along it: its first at its start, its last its repeat's end (the first again)
inline constexpr std::array<double, 5> kAsphaltJoints{0.0, 5.0, 11.0, 16.5, 24.0};

struct AsphaltImage
{
  int                  width = 0, height = 0, channels = 0;
  std::vector<uint8_t> pixels;  // rows of `width` texels of `channels` bytes, row 0 first
};

struct AsphaltMaps
{
  AsphaltImage colour;  // sRGB, RGBA (opaque)
  AsphaltImage normal;  // X and Y
  AsphaltImage rough;   // the grain's share in red, the roughness in green (glTF's)
  AsphaltImage grain;   // X and Y
  AsphaltImage paint;   // sRGB, RGBA (opaque)
};

struct AsphaltSize
{
  int across = 1024;  // the wear's texels across, kAsphaltRepeat / kAsphaltWidth times as many along
  int grain  = 1024;  // the grain's, a side
  int paint  = 256;   // the markings' across (along, as many as the wear's); `across` a whole number of them
};

AsphaltMaps asphaltMaps(uint32_t seed = 1, const AsphaltSize& size = {});

}  // namespace fv::world
