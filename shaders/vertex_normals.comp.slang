// Compute shader for welded vertex normals: each vertex sums the area-weighted normals (the edge
// cross products) of the triangles its list names, in list order, and takes the unit result; a zero
// sum stays zero. fv::mesh::NormalWelder makes the same normals on the CPU, in doubles.

#include "animation_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<NormalsPushConstant> pc;

[shader("compute")]
[numthreads(ANIMATION_WORKGROUP_SIZE, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint v = tid.x;
  if(v >= pc.vertexCount)
    return;

  float3 sum = float3(0);
  for(uint k = pc.triangleStart[v]; k < pc.triangleStart[v + 1]; ++k)
  {
    uint t = pc.triangleList[k];
    if(t >= pc.triangleCount)
      continue;
    uint3 tri = pc.triangles[t];
    if(tri.x >= pc.positionCount || tri.y >= pc.positionCount || tri.z >= pc.positionCount)
      continue;
    float3 a = pc.positions[tri.x];
    sum += cross(pc.positions[tri.y] - a, pc.positions[tri.z] - a);
  }
  float l           = length(sum);
  pc.outNormals[v] = l > 0.0 ? sum / l : float3(0);
}
