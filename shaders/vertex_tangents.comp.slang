// Compute shader for vertex tangents: each vertex adds the UV tangents of the triangles its corner
// list names, in list order (the last one's handedness), then takes the part of the sum across its
// normal; tinygltf::utils::computeTangents makes the same tangents on the CPU, in floats.

#include "animation_io.h.slang"
#include "nvshaders/functions.h.slang"

[[vk::push_constant]]
ConstantBuffer<TangentsPushConstant> pc;

[shader("compute")]
[numthreads(ANIMATION_WORKGROUP_SIZE, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint v = tid.x;
  if(v >= pc.vertexCount)
    return;

  float4 t = float4(0);
  for(uint k = pc.cornerStart[v]; k < pc.cornerStart[v + 1]; ++k)
  {
    uint f = pc.corners[k] / 3;
    if(f >= pc.triangleCount)
      continue;
    uint3 tri = pc.triangles[f];
    if(tri.x >= pc.vertexCount || tri.y >= pc.vertexCount || tri.z >= pc.vertexCount)
      continue;
    float3 n0 = pc.normals[tri.x];
    if(pc.uvs == nullptr)
    {
      t = makeFastTangent(n0);
      continue;
    }
    float3 p0 = pc.positions[tri.x];
    float3 edge1 = pc.positions[tri.y] - p0;
    float3 edge2 = pc.positions[tri.z] - p0;
    float2 uv0      = pc.uvs[tri.x];
    float2 deltaUV1 = pc.uvs[tri.y] - uv0;
    float2 deltaUV2 = pc.uvs[tri.z] - uv0;
    float  r        = 1.0;
    float  a        = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
    if(abs(a) > 0.0)  // a degenerate UV triangle
      r = 1.0 / a;
    float3 tangent    = r * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
    float3 bitangent  = r * (deltaUV2.x * edge1 - deltaUV1.x * edge2);
    float  handedness = dot(cross(tangent, bitangent), n0) > 0.0 ? 1.0 : -1.0;
    t                 = float4(tangent + t.xyz, handedness);
  }

  // the part across the normal; a vertex no triangle touches gets +1 handedness
  float3 n  = pc.normals[v];
  float3 ot = normalize(t.xyz - dot(n, t.xyz) * n);
  if(dot(ot, ot) < 0.1 || any(isnan(ot)))
    ot = makeFastTangent(n).xyz;
  pc.outTangents[v] = float4(ot, t.w < 0.0 ? -1.0 : 1.0);
}
