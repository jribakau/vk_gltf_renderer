#include <figure/garment/weights.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace fv::garment {

std::vector<float> holdByDistance(std::span<const float> item, std::span<const float> body, const wardrobe::SurfaceBinding& at, float inner, float outer)
{
  const size_t       n = item.size() / 3;
  std::vector<float> hold(n, 0.f);
  if(at.verts.size() < n * 3 || at.weights.size() < n * 3)
    return hold;
  for(size_t v = 0; v < n; v++)
  {
    double p[3] = {0, 0, 0}, sum = 0;
    for(size_t j = 0; j < 3; j++)
    {
      const size_t c = at.verts[v * 3 + j];
      const double w = at.weights[v * 3 + j];
      if(w == 0 || c * 3 + 2 >= body.size())
        continue;
      for(size_t k = 0; k < 3; k++)
        p[k] += w * body[c * 3 + k];
      sum += w;
    }
    if(sum <= 0)
      continue;
    double d2 = 0;
    for(size_t k = 0; k < 3; k++)
      d2 += (item[v * 3 + k] - p[k] / sum) * (item[v * 3 + k] - p[k] / sum);
    const double d = std::sqrt(d2);
    if(d <= inner)
      hold[v] = 1;
    else if(d < outer)
    {
      const double t = (d - inner) / (outer - inner);
      hold[v]        = float(1 - t * t * (3 - 2 * t));
    }
  }
  return hold;
}

mesh::SkinInput withBodyWeights(const mesh::SkinInput& item, size_t itemBones, const mesh::SkinInput& body, const wardrobe::SurfaceBinding& at,
                                std::span<const float> hold)
{
  const size_t n = item.start.empty() ? 0 : item.start.size() - 1;
  if(hold.size() != n || at.verts.size() < n * 3 || at.weights.size() < n * 3 || body.start.empty())
    return item;
  const size_t    bodyVerts = body.start.size() - 1;
  mesh::SkinInput out;
  out.rootBone = item.rootBone;
  out.dualQuat = item.dualQuat;
  out.start.reserve(n + 1);
  out.start.push_back(0);
  const auto shareOf = [](const mesh::SkinInput& s, size_t v) { return v < s.dualQuatMap.size() ? s.dualQuatMap[v] : s.dualQuat; };
  // a share map only when a vertex's differs from the garment's own
  std::vector<float>                  shares(n, item.dualQuat);
  bool                                mapped = !item.dualQuatMap.empty();
  std::vector<std::pair<int32_t, float>> row;
  for(size_t v = 0; v < n; v++)
  {
    row.clear();
    // its corners on the body that have weights, and how much of the vertex they are
    double h = std::clamp(double(hold[v]), 0.0, 1.0), bound = 0;
    for(size_t j = 0; j < 3 && h > 0; j++)
    {
      const size_t c = at.verts[v * 3 + j];
      if(c < bodyVerts && body.start[c + 1] > body.start[c])
        bound += at.weights[v * 3 + j];
    }
    if(bound <= 0)
      h = 0;
    double share = (1 - h) * shareOf(item, v);
    for(int32_t i = item.start[v]; i < item.start[v + 1] && h < 1; i++)
      row.emplace_back(item.bone[size_t(i)], float((1 - h) * item.weight[size_t(i)]));
    // a vertex made with no weights rides its root bone
    if(h < 1 && item.start[v + 1] == item.start[v])
      row.emplace_back(item.rootBone, float(1 - h));
    for(size_t j = 0; j < 3 && h > 0; j++)
    {
      const size_t c = at.verts[v * 3 + j];
      const double w = at.weights[v * 3 + j];
      if(w == 0 || c >= bodyVerts || body.start[c + 1] == body.start[c])
        continue;
      const double part = h * w / bound;
      // the corner's own weights, made to sum to one
      double total = 0;
      for(int32_t i = body.start[c]; i < body.start[c + 1]; i++)
        total += body.weight[size_t(i)];
      if(total <= 0)
        continue;
      for(int32_t i = body.start[c]; i < body.start[c + 1]; i++)
        row.emplace_back(int32_t(itemBones) + body.bone[size_t(i)], float(part * body.weight[size_t(i)] / total));
      share += part * shareOf(body, c);
    }
    // one influence a bone, the small ones dropped, strongest first
    std::sort(row.begin(), row.end());
    size_t kept = 0;
    for(size_t i = 0; i < row.size(); i++)
    {
      if(kept && row[kept - 1].first == row[i].first)
        row[kept - 1].second += row[i].second;
      else
        row[kept++] = row[i];
    }
    row.resize(kept);
    row.erase(std::remove_if(row.begin(), row.end(), [](const auto& r) { return r.second < 1e-3f; }), row.end());
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    double sum = 0;
    for(const auto& r : row)
      sum += r.second;
    for(const auto& r : row)
    {
      out.bone.push_back(r.first);
      out.weight.push_back(sum > 0 ? float(r.second / sum) : r.second);
    }
    out.start.push_back(int32_t(out.bone.size()));
    shares[v] = float(share);
    mapped    = mapped || std::abs(shares[v] - item.dualQuat) > 1e-6f;
  }
  if(mapped)
    out.dualQuatMap = std::move(shares);
  return out;
}

}  // namespace fv::garment
