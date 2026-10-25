#include <figure/mesh/normals.hpp>

#include <cmath>

#include <figure/jobs/pool.hpp>

namespace fv::mesh {

namespace {

// a full weld below this many triangles runs on the calling thread
constexpr size_t kParallelTris  = 4096;
constexpr size_t kParallelVerts = 8192;

// Writes (x, y, z) at unit length into `out` at vertex s; a zero sum stays zero.
void unit(std::span<float> out, size_t s, double x, double y, double z)
{
  double l = std::sqrt(x * x + y * y + z * z);
  if(l == 0)
    l = 1;
  out[s * 3]     = float(x / l);
  out[s * 3 + 1] = float(y / l);
  out[s * 3 + 2] = float(z / l);
}

}  // namespace

void weldNormals(std::span<const float> position, std::span<const uint32_t> renderIndex, std::span<const int32_t> firstSplit,
                 std::span<const uint32_t> renderTris, int renderCount, std::span<float> out)
{
  // current positions per render vertex (all splits share one position), as floats
  std::vector<float>  p(size_t(renderCount) * 3, 0.f);
  std::vector<double> sum(size_t(renderCount) * 3, 0.0);
  for(int d = 0; d < renderCount; d++)
  {
    const int32_t s = firstSplit[d];
    if(s < 0)
      continue;
    p[d * 3]     = position[s * 3];
    p[d * 3 + 1] = position[s * 3 + 1];
    p[d * 3 + 2] = position[s * 3 + 2];
  }
  for(size_t t = 0; t + 2 < renderTris.size(); t += 3)
  {
    const size_t a = size_t(renderTris[t]) * 3, b = size_t(renderTris[t + 1]) * 3, c = size_t(renderTris[t + 2]) * 3;
    const double e1x = double(p[b]) - p[a], e1y = double(p[b + 1]) - p[a + 1], e1z = double(p[b + 2]) - p[a + 2];
    const double e2x = double(p[c]) - p[a], e2y = double(p[c + 1]) - p[a + 1], e2z = double(p[c + 2]) - p[a + 2];
    const double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for(size_t v : {a, b, c})
    {
      sum[v] += nx;
      sum[v + 1] += ny;
      sum[v + 2] += nz;
    }
  }
  for(size_t i = 0; i < renderIndex.size(); i++)
  {
    const size_t d = size_t(renderIndex[i]) * 3;
    unit(out, i, sum[d], sum[d + 1], sum[d + 2]);
  }
}

NormalWelder::NormalWelder(std::span<const uint32_t> renderIndex, std::span<const int32_t> firstSplit, std::span<const int32_t> nextSplit,
                           std::span<const uint32_t> renderTris, int renderCount)
    : m_renderIndex(renderIndex)
    , m_firstSplit(firstSplit)
    , m_nextSplit(nextSplit)
    , m_renderTris(renderTris)
    , m_renderCount(renderCount)
{
}

void NormalWelder::full(std::span<const float> position, std::span<float> out)
{
  const size_t triCount = m_renderTris.size() / 3;
  if(triCount < kParallelTris)
  {
    weldNormals(position, m_renderIndex, m_firstSplit, m_renderTris, m_renderCount, out);
    return;
  }
  adjacency();
  m_triNormal.resize(triCount * 3);
  const auto& tris = m_renderTris;
  jobs::parallelFor(triCount, kParallelTris, [&](size_t begin, size_t end) {
    for(size_t ti = begin; ti < end; ti++)
    {
      const size_t t = ti * 3;
      const size_t a = size_t(m_firstSplit[tris[t]]) * 3, b = size_t(m_firstSplit[tris[t + 1]]) * 3, c = size_t(m_firstSplit[tris[t + 2]]) * 3;
      const double e1x = double(position[b]) - position[a], e1y = double(position[b + 1]) - position[a + 1], e1z = double(position[b + 2]) - position[a + 2];
      const double e2x = double(position[c]) - position[a], e2y = double(position[c + 1]) - position[a + 1], e2z = double(position[c + 2]) - position[a + 2];
      m_triNormal[t]     = e1y * e2z - e1z * e2y;
      m_triNormal[t + 1] = e1z * e2x - e1x * e2z;
      m_triNormal[t + 2] = e1x * e2y - e1y * e2x;
    }
  });
  // each render vertex sums its triangles in triangle order, as the serial weld adds them
  jobs::parallelFor(size_t(m_renderCount), kParallelVerts, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
    {
      double x = 0, y = 0, z = 0;
      for(int k = m_adjStart[v]; k < m_adjStart[v + 1]; k++)
      {
        const size_t t = size_t(m_adjTris[k]) * 3;
        x += m_triNormal[t];
        y += m_triNormal[t + 1];
        z += m_triNormal[t + 2];
      }
      for(int32_t s = m_firstSplit[v]; s >= 0; s = m_nextSplit[s])
        unit(out, size_t(s), x, y, z);
    }
  });
}

void NormalWelder::adjacency()
{
  if(!m_adjStart.empty())
    return;
  const int n = m_renderCount;
  m_adjStart.assign(size_t(n) + 1, 0);
  for(uint32_t v : m_renderTris)
    m_adjStart[v + 1]++;
  for(int v = 0; v < n; v++)
    m_adjStart[v + 1] += m_adjStart[v];
  std::vector<int32_t> fill(m_adjStart.begin(), m_adjStart.end() - 1);
  m_adjTris.assign(m_renderTris.size(), 0);
  for(size_t i = 0; i < m_renderTris.size(); i++)
    m_adjTris[fill[m_renderTris[i]]++] = int32_t(i / 3);
}

std::optional<std::pair<uint32_t, uint32_t>> NormalWelder::partial(std::span<const float> position, std::span<float> out, std::span<const uint32_t> moved,
                                                                          std::vector<uint32_t>* written)
{
  if(written)
    written->clear();
  if(moved.size() > m_renderIndex.size() / 4)
  {
    full(position, out);
    if(m_renderIndex.empty())
      return std::nullopt;
    return std::pair<uint32_t, uint32_t>{0, uint32_t(m_renderIndex.size() - 1)};
  }
  adjacency();
  if(m_vertexPass.empty())
  {
    m_vertexPass.assign(m_renderCount, 0);
    m_triPass.assign(m_renderTris.size() / 3, 0);
    m_affected.assign(m_renderCount, 0);
    m_sum.assign(size_t(m_renderCount) * 3, 0.0);
  }
  const int32_t pass = ++m_pass;
  const auto&   tris = m_renderTris;
  // the render vertices that moved, then every vertex sharing a triangle with one
  int count = 0;
  for(uint32_t m : moved)
  {
    const uint32_t v = m_renderIndex[m];
    for(int k = m_adjStart[v]; k < m_adjStart[v + 1]; k++)
    {
      const size_t t = size_t(m_adjTris[k]) * 3;
      for(int j = 0; j < 3; j++)
      {
        const uint32_t u = tris[t + j];
        if(m_vertexPass[u] != pass)
        {
          m_vertexPass[u]   = pass;
          m_affected[count++] = int32_t(u);
          m_sum[u * 3] = m_sum[u * 3 + 1] = m_sum[u * 3 + 2] = 0;
        }
      }
    }
  }
  // each triangle around them once: its normal goes to the corners being recomputed
  for(int i = 0; i < count; i++)
  {
    const int u = m_affected[i];
    for(int k = m_adjStart[u]; k < m_adjStart[u + 1]; k++)
    {
      const int ti = m_adjTris[k];
      if(m_triPass[ti] == pass)
        continue;
      m_triPass[ti]     = pass;
      const size_t   t  = size_t(ti) * 3;
      const uint32_t va = tris[t], vb = tris[t + 1], vc = tris[t + 2];
      const size_t   a = size_t(m_firstSplit[va]) * 3, b = size_t(m_firstSplit[vb]) * 3, c = size_t(m_firstSplit[vc]) * 3;
      const double   e1x = double(position[b]) - position[a], e1y = double(position[b + 1]) - position[a + 1], e1z = double(position[b + 2]) - position[a + 2];
      const double   e2x = double(position[c]) - position[a], e2y = double(position[c + 1]) - position[a + 1], e2z = double(position[c + 2]) - position[a + 2];
      const double   nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      for(uint32_t v : {va, vb, vc})
        if(m_vertexPass[v] == pass)
        {
          m_sum[v * 3] += nx;
          m_sum[v * 3 + 1] += ny;
          m_sum[v * 3 + 2] += nz;
        }
    }
  }
  uint32_t lo = UINT32_MAX, hi = 0;
  bool     any = false;
  for(int i = 0; i < count; i++)
  {
    const int    u = m_affected[i];
    const double x = m_sum[u * 3], y = m_sum[u * 3 + 1], z = m_sum[u * 3 + 2];
    for(int32_t s = m_firstSplit[u]; s >= 0; s = m_nextSplit[s])
    {
      unit(out, size_t(s), x, y, z);
      if(written)
        written->push_back(uint32_t(s));
      lo  = std::min(lo, uint32_t(s));
      hi  = std::max(hi, uint32_t(s));
      any = true;
    }
  }
  if(!any)
    return std::nullopt;
  return std::pair<uint32_t, uint32_t>{lo, hi};
}

}  // namespace fv::mesh
