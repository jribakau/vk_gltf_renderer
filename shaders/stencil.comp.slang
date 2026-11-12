// Compute shader for a stencil pass: each output row is the weighted sum of base elements its
// stencil names (one subdivision level of a figure's cage), or, without row starts, the one base
// element it names (a refined vertex copied to its drawn splits). In floats: fv::subdiv::refineDense
// rounds each step of the same sum from a double, so the last bits may differ. A row's offset, when
// given, is added to its sum (strand ribbons' widths and grown strands' shifts), turned as its first
// column has turned when that is given: an offset is in the rest shape, and a posed cage is not.
// With neither row starts nor columns, each row is its base element plus its offset (a posed cage's
// displacement). A level's HD edits, when given, are added last: DAZ HD morph detail, each offset in
// the frame of a cage face corner made from the cage the level was refined from.

#include "animation_io.h.slang"
#include "unit_vector_pack.h.slang"

[[vk::push_constant]]
ConstantBuffer<StencilPushConstant> pc;

// the direction of v, or none for none
float3 safeUnit(float3 v)
{
  float l = length(v);
  return l > 0.0 ? v / l : float3(0);
}

// a row's value: as it is, or as a packed unit direction (the drawn normals, outUnit)
void write(uint row, float3 value)
{
  if(pc.outUnit != nullptr)
    pc.outUnit[row] = packUnitVector(value);
  else
    pc.outValues[row] = value;
}

[shader("compute")]
[numthreads(ANIMATION_WORKGROUP_SIZE, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint row = tid.x;
  if(row >= pc.rowCount)
    return;

  if(pc.rowStart == nullptr && pc.columns == nullptr)
  {
    write(row, pc.baseValues[row] + (pc.offsets != nullptr ? pc.offsets[row] : float3(0)));
    return;
  }
  if(pc.rowStart == nullptr)
  {
    uint c = pc.columns[row];
    write(row, c < pc.sourceCount ? pc.baseValues[c] : float3(0));
    return;
  }

  float3 sum = float3(0);
  for(uint k = pc.rowStart[row]; k < pc.rowStart[row + 1]; ++k)
  {
    uint c = pc.columns[k];
    if(c < pc.sourceCount)
      sum += (pc.weights != nullptr ? pc.weights[k] : 1.0) * pc.baseValues[c];
  }
  if(pc.offsets != nullptr)
  {
    float3 offset = pc.offsets[row];
    if(pc.frameX != nullptr && pc.rowStart[row] < pc.rowStart[row + 1])
    {
      uint   c = pc.columns[pc.rowStart[row]];
      float3 x = unpackTangent(pc.frameX[c]).xyz;
      float3 z = cross(x, unpackUnitVector(pc.frameY[c]));
      float  l = length(z);
      if(l > 0.0)
      {
        z /= l;
        offset = x * offset.x + cross(z, x) * offset.y + z * offset.z;
      }
    }
    sum += offset;
  }
  // HD detail: each edit's offset in its cage face corner's frame (x toward the corner before, the
  // face's normal, and their cross), as hdMove in figure/subdiv/hd_edits.hpp makes it
  if(pc.hdStart != nullptr)
    for(uint k = pc.hdStart[row]; k < pc.hdStart[row + 1]; ++k)
    {
      uint   f  = pc.hdFrame[k] * 5;
      float3 p0 = pc.hdCage[pc.hdFrameVertices[f]];
      float3 n  = safeUnit(cross(pc.hdCage[pc.hdFrameVertices[f + 1]] - p0, pc.hdCage[pc.hdFrameVertices[f + 2]] - p0));
      float3 x  = safeUnit(pc.hdCage[pc.hdFrameVertices[f + 4]] - pc.hdCage[pc.hdFrameVertices[f + 3]]);
      float3 y  = safeUnit(cross(n, x));
      float3 o  = pc.hdOffsets[k];
      sum += x * o.x + n * o.y - y * o.z;
    }
  write(row, sum);
}
