#include <figure/materials/water.hpp>

namespace fv::mat {

WaterLayer flippedV(const WaterLayer& layer)
{
  WaterLayer out = layer;
  for(WaterItem& item : out.items)
  {
    item.a[1]         = 1 - item.a[1];
    item.b[1]         = 1 - item.b[1];
    item.toTangent[1] = -item.toTangent[1];
    item.toTangent[2] = -item.toTangent[2];
  }
  return out;
}

bool takesWater(const Surface& s)
{
  return s.kind == Kind::eSkin || s.kind == Kind::eNails;
}

}  // namespace fv::mat
