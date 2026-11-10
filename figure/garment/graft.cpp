#include <figure/garment/graft.hpp>

#include <algorithm>
#include <utility>

#include <figure/bodymap/clone.hpp>
#include <figure/wardrobe/fit/bases.hpp>

namespace fv::garment {

namespace {

// a vertex with no value: off the map, or on no polygon
constexpr double kUnknown = -1e30;

// Where a projection puts each vertex of its `from` body: the `values` of the three it lands between,
// by their weights; kUnknown for one with no place, or landing only on vertices with none.
std::vector<double> carry(const bodymap::Projection& p, const std::vector<double>& values)
{
  const size_t        n = p.mapped.size();
  std::vector<double> out(n, kUnknown);
  if(p.at.verts.size() < n * 3 || p.at.weights.size() < n * 3)
    return out;
  for(size_t v = 0; v < n; v++)
  {
    if(!p.mapped[v])
      continue;
    double sum = 0, wsum = 0;
    for(size_t j = 0; j < 3; j++)
    {
      const uint32_t c = p.at.verts[v * 3 + j];
      const double   w = p.at.weights[v * 3 + j];
      if(w <= 0 || c >= values.size() || values[c] == kUnknown)
        continue;
      sum += w * values[c];
      wsum += w;
    }
    if(wsum > 0)
      out[v] = sum / wsum;
  }
  return out;
}

}  // namespace

std::vector<int32_t> hiddenThrough(const wardrobe::BaseCage& source, std::span<const int32_t> hidden, const bodymap::TypeMap& sourceMap,
                                   const bodymap::TypeMap& targetMap, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts)
{
  const size_t sourceVerts = source.positions.size() / 3;
  if(hidden.empty() || source.triPoly.size() * 3 != source.tris.size() || sourceMap.map.fromReference.mapped.empty()
     || sourceMap.source.positions.size() != source.positions.size() || polyStart.size() < 2)
    return {};
  // each source vertex: the share of the polygons round it the graft hides
  std::vector<uint8_t> isHidden(size_t(std::max(source.polyCount, 0)), 0);
  for(const int32_t p : hidden)
    if(p >= 0 && size_t(p) < isHidden.size())
      isHidden[size_t(p)] = 1;
  std::vector<std::pair<uint32_t, uint32_t>> round;  // (vertex, polygon), once each
  round.reserve(source.tris.size());
  for(size_t t = 0; t < source.triPoly.size(); t++)
    for(size_t k = 0; k < 3; k++)
      round.emplace_back(source.tris[t * 3 + k], source.triPoly[t]);
  std::sort(round.begin(), round.end());
  round.erase(std::unique(round.begin(), round.end()), round.end());
  // each vertex inside the region (all its polygons hidden), outside (none), or on its edge
  std::vector<int> hiddenRound(sourceVerts, 0), allRound(sourceVerts, 0);
  for(const auto& [v, p] : round)
  {
    if(v >= sourceVerts)
      continue;
    allRound[v]++;
    hiddenRound[v] += p < isHidden.size() && isHidden[p] ? 1 : 0;
  }
  std::vector<size_t> edge;
  for(size_t v = 0; v < sourceVerts; v++)
    if(hiddenRound[v] > 0 && hiddenRound[v] < allRound[v])
      edge.push_back(v);
  if(edge.empty())
    return {};
  // its signed distance from the edge, cm, inside positive: a field that blends near linearly, so the
  // corners' mean is the polygon's middle's. Read only within a few cm of the edge
  constexpr double kNear = 6;
  const auto       at    = [&](size_t v, size_t k) { return double(source.positions[v * 3 + k]); };
  std::vector<double> distance(sourceVerts, kUnknown);
  for(size_t v = 0; v < sourceVerts; v++)
  {
    if(!allRound[v])
      continue;
    double best = kNear * kNear;
    for(const size_t e : edge)
    {
      double d2 = 0;
      for(size_t k = 0; k < 3; k++)
        d2 += (at(v, k) - at(e, k)) * (at(v, k) - at(e, k));
      best = std::min(best, d2);
    }
    const double d = std::sqrt(best);
    distance[v]    = hiddenRound[v] == allRound[v] ? d : hiddenRound[v] == 0 ? -d : 0;
  }
  // onto the reference where it lands on the source, then the figure where it lands on the reference
  const std::vector<double> onReference = carry(sourceMap.map.fromReference, distance);
  const std::vector<double> onFigure    = carry(targetMap.map.toReference, onReference);
  std::vector<int32_t>      out;
  for(size_t p = 0; p + 1 < polyStart.size(); p++)
  {
    double sum = 0;
    int    known = 0, corners = 0;
    for(int32_t c = polyStart[p]; c < polyStart[p + 1]; c++, corners++)
    {
      const int32_t v = size_t(c) < polyVerts.size() ? polyVerts[size_t(c)] : -1;
      if(v < 0 || size_t(v) >= onFigure.size() || onFigure[size_t(v)] == kUnknown)
        continue;
      sum += onFigure[size_t(v)];
      known++;
    }
    // a polygon mostly off the map (an eye, the mouth's inside) is never under a graft
    if(known * 2 >= corners && known > 0 && sum > 0)
      out.push_back(int32_t(p));
  }
  return out;
}

}  // namespace fv::garment
