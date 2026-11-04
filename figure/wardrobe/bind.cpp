#include <figure/wardrobe/bind.hpp>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/physics/collide/surface_index.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace fv::wardrobe {

namespace {

using physics::SurfaceIndex;

void write(SurfaceBinding& out, size_t v, std::span<const uint32_t> T, const SurfaceIndex::Hit& hit)
{
  out.verts[v * 3]       = T[size_t(hit.tri) * 3];
  out.verts[v * 3 + 1]   = T[size_t(hit.tri) * 3 + 1];
  out.verts[v * 3 + 2]   = T[size_t(hit.tri) * 3 + 2];
  out.weights[v * 3]     = float(hit.u);
  out.weights[v * 3 + 1] = float(hit.v);
  out.weights[v * 3 + 2] = float(hit.w);
}

// A geograft's seam pairs bound outright, then the body vertices any binding reads.
void finish(SurfaceBinding& out, std::span<const int32_t> pairs, size_t nvDst, size_t nvSrc)
{
  for(size_t k = 0; k + 1 < pairs.size(); k += 2)
  {
    const int32_t g = pairs[k], b = pairs[k + 1];
    if(g < 0 || size_t(g) >= nvDst || b < 0 || size_t(b) >= nvSrc)
      continue;
    out.verts[size_t(g) * 3] = out.verts[size_t(g) * 3 + 1] = out.verts[size_t(g) * 3 + 2] = uint32_t(b);
    out.weights[size_t(g) * 3]                                                          = 1;
    out.weights[size_t(g) * 3 + 1] = out.weights[size_t(g) * 3 + 2] = 0;
  }
  for(size_t i = 0; i < out.verts.size(); i++)
    if(out.weights[i] != 0)
      out.used[out.verts[i]] = 1;
}

SurfaceBinding blank(size_t nvDst, size_t nvSrc)
{
  SurfaceBinding out;
  out.verts.assign(nvDst * 3, 0);
  out.weights.assign(nvDst * 3, 0.f);
  out.used.assign(nvSrc, 0);
  return out;
}


// Chunks of about a quarter of an even share per thread: an add-on of 2,000 vertices is searched on
// many threads, not as one chunk.
size_t chunkFor(size_t count, size_t least)
{
  const size_t share = (count + 4 * (jobs::pool().size() + 1) - 1) / (4 * (jobs::pool().size() + 1));
  return std::max(least, share);
}

}  // namespace

SurfaceBinding bindToSurface(std::span<const float>    P,
                             std::span<const uint32_t> T,
                             std::span<const float>    D,
                             std::span<const int32_t>  pairs,
                             const BindFilter*         filter)
{
  const size_t   nvDst = D.size() / 3;
  SurfaceBinding out   = blank(nvDst, P.size() / 3);
  const auto     tGrid = std::chrono::steady_clock::now();
  const SurfaceIndex index(P, T);
  if(index.empty())
    return out;
  const std::span<const int16_t> limbs = filter ? std::span<const int16_t>(filter->bodyLimb) : std::span<const int16_t>();
  timing::report("grid", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tGrid).count(), 5);
  // each vertex is found on its own, so the vertices are split among threads
  const auto tSearch = std::chrono::steady_clock::now();
  jobs::parallelFor(nvDst, chunkFor(nvDst, 64), [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
    {
      const int32_t mask = filter && v < filter->itemMask.size() ? filter->itemMask[v] : 0;
      physics::Stamps& stamps = physics::threadStamps();
      const int32_t    id     = stamps.next(index.triangles());
      const auto    hit  = index.search(D[v * 3], D[v * 3 + 1], D[v * 3 + 2], mask, limbs, INFINITY, stamps.stamp, id);
      if(hit.tri >= 0)
        write(out, v, T, hit);
    }
  });
  timing::report("search", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tSearch).count(), 5);
  finish(out, pairs, nvDst, P.size() / 3);
  return out;
}

SurfaceBinding bindStrands(std::span<const float>    P,
                           std::span<const uint32_t> T,
                           std::span<const float>    D,
                           std::span<const int32_t>  lineStart,
                           std::span<const int32_t>  lineVerts,
                           float                     reach,
                           const BindFilter*         filter)
{
  const size_t   nvDst = D.size() / 3;
  SurfaceBinding out   = blank(nvDst, P.size() / 3);
  const SurfaceIndex index(P, T);
  if(index.empty())
    return out;
  const std::span<const int16_t> limbs = filter ? std::span<const int16_t>(filter->bodyLimb) : std::span<const int16_t>();
  const auto   tSearch = std::chrono::steady_clock::now();
  const double reach2  = double(reach) * reach;
  const size_t lines   = lineStart.empty() ? 0 : lineStart.size() - 1;
  // line point c's vertex, or -1 when it has none
  auto at = [&](int32_t c) {
    const int32_t v = lineVerts[size_t(c)];
    return v >= 0 && size_t(v) < nvDst ? v : -1;
  };
  jobs::parallelFor(lines, std::clamp<size_t>(chunkFor(lines, 16), 16, 256), [&](size_t begin, size_t end) {
    auto find = [&](int32_t v, double maxDist2) {
      const int32_t mask = filter && size_t(v) < filter->itemMask.size() ? filter->itemMask[size_t(v)] : 0;
      physics::Stamps& stamps = physics::threadStamps();
      const int32_t    id     = stamps.next(index.triangles());
      return index.search(D[size_t(v) * 3], D[size_t(v) * 3 + 1], D[size_t(v) * 3 + 2], mask, limbs, maxDist2, stamps.stamp, id);
    };
    for(size_t l = begin; l < end; l++)
    {
      int32_t first = lineStart[l], last = lineStart[l + 1] - 1;
      while(first <= last && at(first) < 0)
        first++;
      while(last > first && at(last) < 0)
        last--;
      if(first > last)
        continue;
      // the root binds first, wherever the body is. Files store strands root first, but nothing
      // promises it, and a strand bound from its tip rides with whatever the tip hangs over. A root
      // within reach of the body is taken as stored; one further out gives way to the other end if
      // that one is nearer. A strand whose two ends both lie on the body keeps its order: a tip
      // resting on the back can be nearer the body than the root is to the scalp.
      SurfaceIndex::Hit above = find(at(first), INFINITY);
      int32_t           from = first, to = last, step = 1;
      if(last != first && above.d2 > reach2)
      {
        const auto other = find(at(last), above.d2);
        if(other.tri >= 0 && other.d2 < above.d2)
        {
          above = other;
          from = last, to = first, step = -1;
        }
      }
      if(above.tri >= 0)
        write(out, size_t(at(from)), T, above);
      // the rest bind where the body is within reach, and further out ride with the point above them
      for(int32_t c = from + step; c != to + step; c += step)
      {
        const int32_t v = at(c);
        if(v < 0)
          continue;
        const auto hit = find(v, above.tri < 0 ? INFINITY : reach2);
        if(hit.tri >= 0)
          above = hit;
        if(above.tri >= 0)
          write(out, size_t(v), T, above);
      }
    }
  });
  // points on no line bind on their own (strand hair has none, or next to none)
  std::vector<char> onLine(nvDst, 0);
  for(int32_t v : lineVerts)
    if(v >= 0 && size_t(v) < nvDst)
      onLine[size_t(v)] = 1;
  std::vector<uint32_t> loose;
  for(size_t v = 0; v < nvDst; v++)
    if(!onLine[v])
      loose.push_back(uint32_t(v));
  jobs::parallelFor(loose.size(), chunkFor(loose.size(), 64), [&](size_t begin, size_t end) {
    for(size_t k = begin; k < end; k++)
    {
      const size_t  v    = loose[k];
      const int32_t mask = filter && v < filter->itemMask.size() ? filter->itemMask[v] : 0;
      physics::Stamps& stamps = physics::threadStamps();
      const int32_t    id     = stamps.next(index.triangles());
      const auto    hit  = index.search(D[v * 3], D[v * 3 + 1], D[v * 3 + 2], mask, limbs, INFINITY, stamps.stamp, id);
      if(hit.tri >= 0)
        write(out, v, T, hit);
    }
  });
  timing::report("strand search", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tSearch).count(), 5);
  finish(out, {}, nvDst, P.size() / 3);
  return out;
}

}  // namespace fv::wardrobe
