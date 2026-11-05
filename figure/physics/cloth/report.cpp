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

namespace {

// Each of `points`' side of its closest face of `surface` (+1 out along the face's winding, -1 in), or 0
// where that face is further than `reach` or has no area.
std::vector<int> sides(const std::vector<float>& surface, const std::vector<uint32_t>& tris, const std::vector<float>& points, float reach)
{
  const wardrobe::SurfaceBinding bind = wardrobe::bindToSurface(surface, tris, points);
  std::vector<int>               out(points.size() / 3, 0);
  for(size_t v = 0; v < out.size(); v++)
  {
    const uint32_t* c = &bind.verts[v * 3];
    const float*    w = &bind.weights[v * 3];
    double          p[3], e1[3], e2[3];
    for(int k = 0; k < 3; k++)
    {
      p[k]  = w[0] * surface[c[0] * 3 + k] + w[1] * surface[c[1] * 3 + k] + w[2] * surface[c[2] * 3 + k];
      e1[k] = surface[c[1] * 3 + k] - surface[c[0] * 3 + k];
      e2[k] = surface[c[2] * 3 + k] - surface[c[0] * 3 + k];
    }
    const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const double d[3] = {points[v * 3] - p[0], points[v * 3 + 1] - p[1], points[v * 3 + 2] - p[2]};
    const double len  = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if(len < 1e-12 || dist > reach)
      continue;
    const double side = (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / len;
    out[v]            = side > 0 ? 1 : side < 0 ? -1 : 0;
  }
  return out;
}

}  // namespace

size_t wrongSide(const Surface& a, const std::vector<float>& aOut, const Surface& b, const std::vector<float>& bOut, float reach)
{
  if(b.tris.empty() || a.rest.empty())
    return 0;
  const std::vector<int> made = sides(b.rest, b.tris, a.rest, reach);
  const std::vector<int> now  = sides(bOut, b.tris, aOut, reach);
  size_t                 n    = 0;
  for(size_t v = 0; v < made.size() && v < now.size(); v++)
    if(made[v] != 0 && now[v] != 0 && made[v] != now[v])
      n++;
  return n;
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
