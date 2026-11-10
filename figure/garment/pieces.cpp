#include <figure/garment/pieces.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

#include <figure/bodymap/clone.hpp>

namespace fv::garment {

namespace {

// A vertex's influences by bone, summed.
std::map<int32_t, float> rowOf(const mesh::SkinInput& skin, size_t v)
{
  std::map<int32_t, float> row;
  if(v + 1 < skin.start.size())
    for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
      row[skin.bone[size_t(i)]] += skin.weight[size_t(i)];
  return row;
}

bool sameRows(const std::map<int32_t, float>& a, const std::map<int32_t, float>& b, float within)
{
  for(const auto& [bone, w] : a)
  {
    const auto it = b.find(bone);
    if(std::abs(w - (it == b.end() ? 0.f : it->second)) > within)
      return false;
  }
  for(const auto& [bone, w] : b)
    if(!a.count(bone) && w > within)
      return false;
  return true;
}

}  // namespace

Pieces hardPieces(std::span<const float> positions, std::span<const uint32_t> tris, const mesh::SkinInput& skin, const PieceOptions& options)
{
  const size_t n = positions.size() / 3;
  Pieces       out;
  out.of.assign(n, -1);
  // the cage's connected parts
  std::vector<uint32_t> parent(n);
  std::iota(parent.begin(), parent.end(), 0u);
  const auto find = [&](uint32_t v) {
    while(parent[v] != v)
      v = parent[v] = parent[parent[v]];
    return v;
  };
  std::vector<uint8_t> faced(n, 0);
  for(size_t i = 0; i + 2 < tris.size(); i += 3)
  {
    if(tris[i] >= n || tris[i + 1] >= n || tris[i + 2] >= n)
      continue;
    faced[tris[i]] = faced[tris[i + 1]] = faced[tris[i + 2]] = 1;
    parent[find(tris[i + 1])] = find(tris[i]);
    parent[find(tris[i + 2])] = find(tris[i]);
  }
  struct Part
  {
    std::vector<uint32_t> verts;
    float                 lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
  };
  std::map<uint32_t, Part> parts;
  for(size_t v = 0; v < n; v++)
  {
    if(!faced[v])
      continue;
    Part& p = parts[find(uint32_t(v))];
    p.verts.push_back(uint32_t(v));
    for(size_t k = 0; k < 3; k++)
    {
      p.lo[k] = std::min(p.lo[k], positions[v * 3 + k]);
      p.hi[k] = std::max(p.hi[k], positions[v * 3 + k]);
    }
  }
  size_t largest = 0;
  for(const auto& [root, p] : parts)
    largest = std::max(largest, p.verts.size());
  const bool skinned = skin.start.size() == n + 1;
  for(const auto& [root, p] : parts)
  {
    if(p.verts.size() < size_t(std::max(options.least, 1)))
      continue;
    double diagonal = 0;
    for(size_t k = 0; k < 3; k++)
      diagonal += double(p.hi[k] - p.lo[k]) * (p.hi[k] - p.lo[k]);
    // small, though not the body of a small garment; or every vertex on the same weights
    const bool little = std::sqrt(diagonal) <= options.small && (parts.size() > 1 && p.verts.size() < largest);
    bool       same   = false;
    if(skinned && std::sqrt(diagonal) <= options.plate)
    {
      const auto first = rowOf(skin, p.verts[0]);
      same             = !first.empty();
      for(size_t i = 1; i < p.verts.size() && same; i++)
        same = sameRows(first, rowOf(skin, p.verts[i]), options.sameWithin);
    }
    if(!little && !same)
      continue;
    for(uint32_t v : p.verts)
      out.of[v] = out.count;
    out.hard.push_back(same || std::sqrt(diagonal) <= options.stud);
    out.count++;
  }
  return out;
}

std::vector<float> keepPieces(std::span<const float> made, std::span<const float> fitted, const Pieces& pieces)
{
  std::vector<float> out(fitted.begin(), fitted.end());
  const size_t       n = fitted.size() / 3;
  if(made.size() != fitted.size() || pieces.of.size() != n || pieces.count <= 0)
    return out;
  std::vector<std::vector<uint32_t>> verts(size_t(pieces.count));
  for(size_t v = 0; v < n; v++)
    if(pieces.of[v] >= 0 && pieces.of[v] < pieces.count)
      verts[size_t(pieces.of[v])].push_back(uint32_t(v));
  const auto at = [](std::span<const float> p, size_t v) { return Vec3{p[v * 3], p[v * 3 + 1], p[v * 3 + 2]}; };
  for(const std::vector<uint32_t>& piece : verts)
  {
    if(piece.size() < 3)
      continue;
    Vec3 from, to;
    for(uint32_t v : piece)
    {
      from = from + at(made, v);
      to   = to + at(fitted, v);
    }
    from = from * (1.0 / double(piece.size()));
    to   = to * (1.0 / double(piece.size()));
    std::vector<Vec3> a, b;
    double            spreadA = 0, spreadB = 0;
    for(uint32_t v : piece)
    {
      a.push_back(at(made, v) - from);
      b.push_back(at(fitted, v) - to);
      spreadA += dot(a.back(), a.back());
      spreadB += dot(b.back(), b.back());
    }
    if(spreadA <= 1e-12)
      continue;
    const Quat   turn = bodymap::bestTurn(a, b);
    const double size = std::clamp(std::sqrt(spreadB / spreadA), 0.8, 1.25);
    for(size_t i = 0; i < piece.size(); i++)
    {
      const Vec3 p          = to + rotate(a[i], turn) * size;
      out[piece[i] * 3]     = float(p.x);
      out[piece[i] * 3 + 1] = float(p.y);
      out[piece[i] * 3 + 2] = float(p.z);
    }
  }
  return out;
}

mesh::SkinInput rigidSkin(const mesh::SkinInput& skin, const Pieces& pieces)
{
  const size_t n = skin.start.empty() ? 0 : skin.start.size() - 1;
  if(pieces.of.size() != n || pieces.count <= 0)
    return skin;
  // each piece's mean row and share
  std::vector<std::map<int32_t, double>> rows(size_t(pieces.count));
  std::vector<double>                    shares(size_t(pieces.count), 0.0), counts(size_t(pieces.count), 0.0);
  const auto shareOf = [&](size_t v) { return v < skin.dualQuatMap.size() ? skin.dualQuatMap[v] : skin.dualQuat; };
  for(size_t v = 0; v < n; v++)
  {
    const int32_t p = pieces.of[v];
    if(p < 0 || p >= pieces.count || (size_t(p) < pieces.hard.size() && !pieces.hard[size_t(p)]))
      continue;
    for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
      rows[size_t(p)][skin.bone[size_t(i)]] += skin.weight[size_t(i)];
    shares[size_t(p)] += shareOf(v);
    counts[size_t(p)] += 1;
  }
  mesh::SkinInput out;
  out.rootBone = skin.rootBone;
  out.dualQuat = skin.dualQuat;
  out.start.push_back(0);
  bool               mapped = !skin.dualQuatMap.empty();
  std::vector<float> map(n, skin.dualQuat);
  for(size_t v = 0; v < n; v++)
  {
    const int32_t p = pieces.of[v];
    if(p < 0 || p >= pieces.count || rows[size_t(p)].empty())
    {
      out.bone.insert(out.bone.end(), skin.bone.begin() + skin.start[v], skin.bone.begin() + skin.start[v + 1]);
      out.weight.insert(out.weight.end(), skin.weight.begin() + skin.start[v], skin.weight.begin() + skin.start[v + 1]);
      map[v] = shareOf(v);
    }
    else
    {
      // strongest first, as a vertex's row is read
      std::vector<std::pair<double, int32_t>> row;
      double                                  sum = 0;
      for(const auto& [bone, w] : rows[size_t(p)])
      {
        row.emplace_back(w, bone);
        sum += w;
      }
      std::sort(row.begin(), row.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
      for(const auto& [w, bone] : row)
      {
        out.bone.push_back(bone);
        out.weight.push_back(float(sum > 0 ? w / sum : 0));
      }
      map[v] = float(shares[size_t(p)] / counts[size_t(p)]);
    }
    out.start.push_back(int32_t(out.bone.size()));
    mapped = mapped || std::abs(map[v] - skin.dualQuat) > 1e-6f;
  }
  if(mapped)
    out.dualQuatMap = std::move(map);
  return out;
}

}  // namespace fv::garment
