#include <figure/subdiv/hd_edits.hpp>

#include <cmath>

namespace fv::subdiv {

namespace {

using Vec = std::array<double, 3>;

Vec at(std::span<const float> p, int32_t v)
{
  return {p[size_t(v) * 3], p[size_t(v) * 3 + 1], p[size_t(v) * 3 + 2]};
}
Vec sub(const Vec& a, const Vec& b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec cross(const Vec& a, const Vec& b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vec unit(const Vec& a)
{
  const double l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  return l > 0 ? Vec{a[0] / l, a[1] / l, a[2] / l} : Vec{0, 0, 0};
}

}  // namespace

bool HdEdits::empty() const
{
  for(const Level& l : levels)
    if(!l.vertex.empty())
      return false;
  return true;
}

HdEdits placeEdits(const Refinement& r, std::span<const int32_t> faceStart, const LevelEdits& edits)
{
  HdEdits out;
  const int    finest  = int(r.levels.size());
  const size_t faces   = faceStart.empty() ? 0 : faceStart.size() - 1;
  out.levels.resize(std::min<size_t>(edits.size(), size_t(finest)));
  for(size_t k = out.levels.size(); k < edits.size(); k++)
    out.dropped += edits[k].size();
  for(size_t k = 0; k < out.levels.size(); k++)
  {
    const int      level = int(k) + 1;
    const uint32_t per   = 1u << (2 * (level - 1));  // a level-1 face's descendants at this level
    HdEdits::Level& l    = out.levels[k];
    for(const FaceEdit& e : edits[k])
    {
      if(e.face >= faces || e.corner > 3)
      {
        out.dropped++;
        continue;
      }
      const uint32_t size = uint32_t(faceStart[e.face + 1] - faceStart[e.face]);
      if(e.sub >= size * per)
      {
        out.dropped++;
        continue;
      }
      // the level's face, numbered as OpenSubdiv numbers it: a face's children are consecutive, child
      // j at corner j, so face g's children are 4g .. 4g + 3 from level 1 on
      uint64_t g = uint64_t(faceStart[e.face]) * per + e.sub;
      // the vertex at its corner: every face from level 1 is a quad whose child j keeps corner j at
      // position j, and a vertex keeps its index from level to level, so follow corner j down to the
      // finest quads
      for(int d = level; d < finest; d++)
        g = g * 4 + e.corner;
      const size_t q = size_t(g) * 4 + e.corner;
      if(q >= r.quads.size())
      {
        out.dropped++;
        continue;
      }
      l.vertex.push_back(uint32_t(r.quads[q]));
      l.faceCorner.push_back(uint32_t(faceStart[e.face]) + e.sub / per);
      l.offset.insert(l.offset.end(), e.offset, e.offset + 3);
    }
  }
  return out;
}

std::vector<int32_t> cornerFaces(std::span<const int32_t> faceStart)
{
  std::vector<int32_t> out(faceStart.empty() ? 0 : size_t(faceStart.back()));
  for(size_t f = 0; f + 1 < faceStart.size(); f++)
    for(int32_t c = faceStart[f]; c < faceStart[f + 1]; c++)
      out[size_t(c)] = int32_t(f);
  return out;
}

std::array<float, 9> cornerFrame(std::span<const float> cagePositions, std::span<const int32_t> faceStart,
                                 std::span<const int32_t> faceVerts, std::span<const int32_t> cornerFace, uint32_t faceCorner)
{
  const int32_t f = cornerFace[faceCorner];
  const int32_t s = faceStart[size_t(f)], n = faceStart[size_t(f) + 1] - s;
  const int32_t i = int32_t(faceCorner) - s, prev = i > 0 ? i - 1 : n - 1;
  const Vec     v0 = at(cagePositions, faceVerts[size_t(s)]);
  const Vec     nrm = unit(cross(sub(at(cagePositions, faceVerts[size_t(s + 1)]), v0), sub(at(cagePositions, faceVerts[size_t(s + n - 1)]), v0)));
  const Vec     vi = at(cagePositions, faceVerts[size_t(s + i)]);
  const Vec     x  = unit(sub(at(cagePositions, faceVerts[size_t(s + prev)]), vi));
  const Vec     y  = unit(cross(nrm, x));
  return {float(x[0]), float(x[1]), float(x[2]), float(nrm[0]), float(nrm[1]), float(nrm[2]), float(-y[0]), float(-y[1]), float(-y[2])};
}

std::vector<float> refineWithEdits(const Refinement& r, std::span<const float> cagePositions, std::span<const int32_t> faceStart,
                                   std::span<const int32_t> faceVerts, std::span<const std::pair<float, const HdEdits*>> edits)
{
  const std::vector<int32_t> owner = cornerFaces(faceStart);
  std::vector<double>        cur(cagePositions.begin(), cagePositions.end());
  for(size_t k = 0; k < r.levels.size(); k++)
  {
    const Stencils&     s = r.levels[k];
    std::vector<double> next(size_t(s.rows()) * 3, 0.0);
    for(int row = 0; row < s.rows(); row++)
      for(int j = s.rowStart[row]; j < s.rowStart[row + 1]; j++)
        for(int c = 0; c < 3; c++)
          next[size_t(row) * 3 + c] += double(s.weight[j]) * cur[size_t(s.col[j]) * 3 + c];
    for(const auto& [weight, e] : edits)
    {
      if(!e || k >= e->levels.size())
        continue;
      const HdEdits::Level& l = e->levels[k];
      for(size_t i = 0; i < l.vertex.size(); i++)
      {
        const std::array<float, 9> m = cornerFrame(cagePositions, faceStart, faceVerts, owner, l.faceCorner[i]);
        const float* o = &l.offset[i * 3];
        for(int c = 0; c < 3; c++)
          next[size_t(l.vertex[i]) * 3 + c] += double(weight) * (double(m[c]) * o[0] + double(m[3 + c]) * o[1] + double(m[6 + c]) * o[2]);
      }
    }
    cur = std::move(next);
  }
  std::vector<float> out(cur.size());
  for(size_t i = 0; i < cur.size(); i++)
    out[i] = float(cur[i]);
  return out;
}

}  // namespace fv::subdiv
