#include <figure/garment/weights.hpp>

#include <algorithm>
#include <cmath>
#include <set>
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

mesh::SkinInput transferChains(const mesh::SkinInput& item, const std::vector<Joint>& itemBones, const std::map<std::string, std::string>& follow,
                               const std::vector<Joint>& figure, const std::vector<int>& bodyBone, std::span<const float> positions, double band,
                               size_t* moved)
{
  if(moved)
    *moved = 0;
  const size_t n = item.start.empty() ? 0 : item.start.size() - 1;
  if(!n || positions.size() < n * 3 || bodyBone.size() != figure.size())
    return item;
  std::map<std::string, int> figureIndex;
  for(size_t j = 0; j < figure.size(); j++)
    figureIndex[figure[j].name] = int(j);
  // each item bone's leader on the figure
  std::vector<int> leader(itemBones.size(), -1);
  std::set<int>    leaders;
  for(size_t i = 0; i < itemBones.size(); i++)
    if(const auto f = follow.find(itemBones[i].name); f != follow.end())
      if(const auto j = figureIndex.find(f->second); j != figureIndex.end())
      {
        leader[i] = j->second;
        leaders.insert(j->second);
      }
  // per item bone, the chains to its children's leaders through joints no item bone follows
  struct Chain
  {
    std::vector<int>      skipped;  // root first
    std::array<double, 3> from{}, to{};
  };
  std::vector<std::vector<Chain>> chains(itemBones.size());
  bool                            any = false;
  for(size_t c = 0; c < itemBones.size(); c++)
  {
    const int p = itemBones[c].parent;
    if(p < 0 || size_t(p) >= itemBones.size() || leader[size_t(p)] < 0 || leader[c] < 0 || leader[size_t(p)] == leader[c])
      continue;
    const int        fp = leader[size_t(p)], fc = leader[c];
    std::vector<int> between;
    int              j = figure[size_t(fc)].parent;
    for(; j >= 0 && j != fp; j = figure[size_t(j)].parent)
      between.push_back(j);
    if(j != fp || between.empty() || std::any_of(between.begin(), between.end(), [&](int b) { return leaders.count(b) > 0; }))
      continue;
    std::reverse(between.begin(), between.end());
    auto& own = chains[size_t(p)];
    if(std::none_of(own.begin(), own.end(), [&](const Chain& k) { return k.skipped == between; }))
    {
      own.push_back({between, figure[size_t(fp)].center, figure[size_t(fc)].center});
      any = true;
    }
  }
  if(!any)
    return item;
  const auto sub = [](const std::array<double, 3>& a, const std::array<double, 3>& b) { return std::array<double, 3>{a[0] - b[0], a[1] - b[1], a[2] - b[2]}; };
  const auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  mesh::SkinInput out;
  out.rootBone    = item.rootBone;
  out.dualQuat    = item.dualQuat;
  out.dualQuatMap = item.dualQuatMap;
  out.start.reserve(n + 1);
  out.start.push_back(0);
  std::vector<std::pair<int32_t, float>> row;
  for(size_t v = 0; v < n; v++)
  {
    row.clear();
    const std::array<double, 3> x{positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]};
    bool                        took = false;
    for(int32_t i = item.start[v]; i < item.start[v + 1]; i++)
    {
      const int32_t b = item.bone[size_t(i)];
      const float   w = item.weight[size_t(i)];
      if(b < 0 || size_t(b) >= chains.size() || chains[size_t(b)].empty())
      {
        row.emplace_back(b, w);
        continue;
      }
      // the chain whose axis lies nearest
      const Chain* chain = nullptr;
      double       best  = 1e300;
      for(const Chain& k : chains[size_t(b)])
      {
        const auto   d   = sub(k.to, k.from);
        const double len = dot(d, d);
        const double t   = len > 0 ? std::clamp(dot(sub(x, k.from), d) / len, 0.0, 1.0) : 0.0;
        const std::array<double, 3> foot{k.from[0] + t * d[0], k.from[1] + t * d[1], k.from[2] + t * d[2]};
        const double                dist = dot(sub(x, foot), sub(x, foot));
        if(dist < best)
          best = dist, chain = &k;
      }
      auto         d   = sub(chain->to, chain->from);
      const double len = std::sqrt(dot(d, d));
      if(len <= 0)
      {
        row.emplace_back(b, w);
        continue;
      }
      d = {d[0] / len, d[1] / len, d[2] / len};
      // how far past each skipped joint's centre it lies, never more past a later one
      std::vector<double> past(chain->skipped.size());
      for(size_t k = 0; k < past.size(); k++)
      {
        const double s = dot(sub(x, figure[size_t(chain->skipped[k])].center), d);
        const double t = std::clamp((s + band) / (2 * band), 0.0, 1.0);
        past[k]        = t * t * (3 - 2 * t);
        if(k > 0)
          past[k] = std::min(past[k], past[k - 1]);
      }
      double kept = w * (1 - past[0]);
      for(size_t k = 0; k < past.size(); k++)
      {
        const double share = w * (past[k] - (k + 1 < past.size() ? past[k + 1] : 0.0));
        const int    body  = bodyBone[size_t(chain->skipped[k])];
        if(share <= 0)
          continue;
        if(body < 0)
          kept += share;  // the body's skin hasn't it: the bone keeps it
        else
        {
          row.emplace_back(int32_t(itemBones.size()) + body, float(share));
          took = took || share > 1e-3;
        }
      }
      if(kept > 0)
        row.emplace_back(b, float(kept));
    }
    // one influence a bone, strongest first
    std::sort(row.begin(), row.end());
    size_t keptRows = 0;
    for(size_t i = 0; i < row.size(); i++)
    {
      if(keptRows && row[keptRows - 1].first == row[i].first)
        row[keptRows - 1].second += row[i].second;
      else
        row[keptRows++] = row[i];
    }
    row.resize(keptRows);
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for(const auto& r : row)
    {
      out.bone.push_back(r.first);
      out.weight.push_back(r.second);
    }
    out.start.push_back(int32_t(out.bone.size()));
    if(moved && took)
      (*moved)++;
  }
  return out;
}

}  // namespace fv::garment
