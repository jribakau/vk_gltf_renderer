// FV water (figure/materials/water.hpp, src/scene_water.hpp): what lies on a surface at a point of its
// UVs, looked up in its material slot's grid. The domes in the point's cell are spherical caps, round on
// the skin through each one's toTangent; the highest at the point is the water's surface there.
// gltf_material_eval.h.slang lays it over the surface: the damp as a film, the domes as a clear coat.

#ifndef WATER_H
#define WATER_H 1

#include "gltf_scene_io.h.slang"

// domes looked at in a cell, at most (a cell is about a bead across)
static const uint WATER_CELL_MAX = 24;
// how far past its outline a dome wets the skin (rs::kWaterHalo)
static const float WATER_HALO = 1.5F;

// The coat's tangent-space normal where a dome covers the point (xy; z is what that leaves), how much
// of the point the water covers (z), and how wet the skin is there (w): the damp all over, and all of it
// under a dome or its halo. Zero for a dry slot.
float4 sampleWater(GltfScene* scene, uint materialIndex, float2 uv)
{
  if(scene->waterGrids == nullptr || int(materialIndex) >= scene->numWaterGrids)
    return float4(0.0F);
  const WaterGrid grid = scene->waterGrids[materialIndex];
  float4          out  = float4(0.0F, 0.0F, 0.0F, grid.damp);
  if(grid.width == 0)
    return out;
  const int2 cell = int2(floor((uv - grid.uvMin) * grid.cellsPerUv));
  if(any(cell < 0) || cell.x >= int(grid.width) || cell.y >= int(grid.height))
    return out;
  const uint2 run = scene->waterCells[grid.firstCell + uint(cell.y) * grid.width + uint(cell.x)];
  float       top = 0.0F;
  for(uint i = 0; i < min(run.y, WATER_CELL_MAX); i++)
  {
    const WaterItem item = scene->waterItems[scene->waterRefs[run.x + i]];
    // the point and the dome's way from a to b, in metres along the tangent and the bitangent
    const float2x2 toTangent = float2x2(item.toTangent.xy, item.toTangent.zw);
    const float2   p         = mul(toTangent, uv - item.a);
    const float2   s         = mul(toTangent, item.b - item.a);
    const float2   d         = p - s * saturate(dot(p, s) / max(dot(s, s), 1e-20F));
    const float    r2        = dot(d, d);
    const float    a2        = item.radius * item.radius;
    if(r2 < a2 * WATER_HALO * WATER_HALO)
      out.w = 1.0F;
    if(r2 >= a2)
      continue;
    // a cap of contact radius a and height h is a piece of a sphere of radius (a² + h²) / 2h, whose
    // normal at the point is its offset over that radius
    const float radius = (a2 + item.height * item.height) / (2.0F * item.height);
    const float h      = sqrt(max(radius * radius - r2, 0.0F)) - (radius - item.height);
    if(h > top)
    {
      top    = h;
      out.xy = d / radius;
      out.z  = 1.0F;
    }
  }
  return out;
}

#endif
