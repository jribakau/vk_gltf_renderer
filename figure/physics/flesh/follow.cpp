#include <figure/physics/flesh/follow.hpp>

#include <algorithm>

namespace fv::physics {

bool bindsTo(const wardrobe::SurfaceBinding& binding, std::span<const uint32_t> bodyVertices)
{
  return std::any_of(bodyVertices.begin(), bodyVertices.end(), [&](uint32_t v) { return v < binding.used.size() && binding.used[v]; });
}

bool followFlesh(const wardrobe::SurfaceBinding& binding, const wardrobe::Rigidity* rigidity, std::span<const float> body, const Mat4* toItem,
                 std::vector<float>& out)
{
  const size_t n = binding.verts.size() / 3;
  out.assign(n * 3, 0.f);
  const size_t bodyCount = body.size() / 3;
  bool         any       = false;
  for(size_t v = 0; v < n; v++)
  {
    float d[3] = {0, 0, 0};
    for(size_t k = v * 3; k < v * 3 + 3; k++)
    {
      const size_t b = binding.verts[k];
      const float  w = binding.weights[k];
      if(b >= bodyCount || w == 0.f)
        continue;
      for(int i = 0; i < 3; i++)
        d[i] += body[b * 3 + i] * w;
    }
    if(d[0] == 0.f && d[1] == 0.f && d[2] == 0.f)
      continue;
    any = true;
    std::copy(d, d + 3, out.begin() + ptrdiff_t(v * 3));
  }
  if(!any)
    return false;
  // a rigid group moves as one, by its reference vertices' mean, as the item's morphs move it
  if(rigidity)
    for(const wardrobe::Rigidity::Group& g : rigidity->groups)
    {
      float mean[3] = {0, 0, 0};
      int   refs    = 0;
      for(uint32_t r : g.refs)
        if(r < n)
        {
          for(int i = 0; i < 3; i++)
            mean[i] += out[r * 3ull + i];
          refs++;
        }
      if(!refs)
        continue;
      for(float& m : mean)
        m /= float(refs);
      for(uint32_t m : g.mask)
        if(m < n && m < rigidity->weight.size())
          for(int i = 0; i < 3; i++)
            out[m * 3ull + i] += (mean[i] - out[m * 3ull + i]) * rigidity->weight[m];
    }
  // into the item mesh's space, where it isn't the body's
  if(toItem)
  {
    const auto& e = toItem->e;
    for(size_t v = 0; v < n; v++)
    {
      const double x = out[v * 3], y = out[v * 3 + 1], z = out[v * 3 + 2];
      out[v * 3]     = float(e[0] * x + e[4] * y + e[8] * z);
      out[v * 3 + 1] = float(e[1] * x + e[5] * y + e[9] * z);
      out[v * 3 + 2] = float(e[2] * x + e[6] * y + e[10] * z);
    }
  }
  return true;
}

void drawnOffsets(std::span<const float> cage, std::span<const uint32_t> drawnFrom, std::vector<float>& out)
{
  out.assign(drawnFrom.size() * 3, 0.f);
  const size_t cageCount = cage.size() / 3;
  for(size_t v = 0; v < drawnFrom.size(); v++)
    if(const size_t c = drawnFrom[v]; c < cageCount)
      for(int i = 0; i < 3; i++)
        out[v * 3 + i] = cage[c * 3 + i];
}

}  // namespace fv::physics
