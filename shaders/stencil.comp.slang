// Compute shader for a stencil pass: each output row is the weighted sum of base elements its
// stencil names (one subdivision level of a figure's cage), or, without row starts, the one base
// element it names (a refined vertex copied to its drawn splits). In floats: fv::subdiv::refineDense
// rounds each step of the same sum from a double, so the last bits may differ. A row's offset, when
// given, is added to its sum (strand ribbons' widths).

#include "animation_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<StencilPushConstant> pc;

[shader("compute")]
[numthreads(ANIMATION_WORKGROUP_SIZE, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint row = tid.x;
  if(row >= pc.rowCount)
    return;

  if(pc.rowStart == nullptr)
  {
    uint c              = pc.columns[row];
    pc.outValues[row] = c < pc.sourceCount ? pc.baseValues[c] : float3(0);
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
    sum += pc.offsets[row];
  pc.outValues[row] = sum;
}
