#include <figure/physics/cloth/report.hpp>

#include <algorithm>
#include <cmath>

#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/collide.hpp>

namespace fv::physics::cloth {

std::vector<double> stretchRatios(const Model& model, size_t slot, size_t n, const std::vector<float>& out)
{
  const int           base = model.base[slot];
  std::vector<double> ratio;
  for(const Distance& e : model.edges)
  {
    if(e.kind != DistanceKind::eStretch || e.rest <= 0 || e.a < base || e.b < base || e.a >= base + int(n) || e.b >= base + int(n))
      continue;
    if(model.inv[size_t(e.a)] <= 0 && model.inv[size_t(e.b)] <= 0)
      continue;
    const size_t a = size_t(e.a - base) * 3, b = size_t(e.b - base) * 3;
    const double dx = out[b] - out[a], dy = out[b + 1] - out[a + 1], dz = out[b + 2] - out[a + 2];
    ratio.push_back(std::sqrt(dx * dx + dy * dy + dz * dz) / e.rest);
  }
  std::sort(ratio.begin(), ratio.end());
  return ratio;
}

size_t insideBody(const Surface& body, const std::vector<float>& bodyNow, const std::vector<float>& out)
{
  const wardrobe::Surface        surface{bodyNow, body.tris};
  const wardrobe::SurfaceBinding binding = wardrobe::bindToSurface(bodyNow, body.tris, out);
  const std::vector<float>       pushed  = wardrobe::pushOut(out, surface, binding, 0.f, 3.f);
  size_t                         inside  = 0;
  for(size_t k = 0; k + 2 < out.size(); k += 3)
    if(pushed[k] != out[k] || pushed[k + 1] != out[k + 1] || pushed[k + 2] != out[k + 2])
      inside++;
  return inside;
}

DrapeNumbers drapeNumbers(const Model& model, size_t slot, const Surface& mesh, const std::vector<float>& out, const Surface& body,
                          const std::vector<float>& bodyNow)
{
  DrapeNumbers d;
  const auto   ratio = stretchRatios(model, slot, mesh.vertexCount(), out);
  if(!ratio.empty())
  {
    d.stretchMax = ratio.back();
    d.stretchP99 = ratio[size_t(0.99 * double(ratio.size() - 1))];
  }
  d.stretchOver12 = size_t(std::count_if(ratio.begin(), ratio.end(), [](double x) { return x > 1.2; }));
  d.inside        = insideBody(body, bodyNow, out);
  float minOut = 1e9f, minPosed = 1e9f;
  for(size_t v = 0; v < mesh.vertexCount(); v++)
  {
    minOut   = std::min(minOut, out[v * 3 + 1]);
    minPosed = std::min(minPosed, mesh.posed[v * 3 + 1]);
  }
  d.hemDrop = double(minPosed - minOut);
  return d;
}

}  // namespace fv::physics::cloth
