#include <figure/subdiv/hd_edits.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

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

FrameVertices frameVertices(std::span<const int32_t> faceStart, std::span<const int32_t> faceVerts, std::span<const int32_t> cornerFace,
                            uint32_t faceCorner)
{
  const int32_t f = cornerFace[faceCorner];
  const int32_t s = faceStart[size_t(f)], n = faceStart[size_t(f) + 1] - s;
  const int32_t i = int32_t(faceCorner) - s, prev = i > 0 ? i - 1 : n - 1;
  return {faceVerts[size_t(s)], faceVerts[size_t(s + 1)], faceVerts[size_t(s + n - 1)], faceVerts[size_t(s + i)], faceVerts[size_t(s + prev)]};
}

std::array<float, 9> frameOf(std::span<const float> positions, const FrameVertices& v)
{
  const Vec v0  = at(positions, v[0]);
  const Vec nrm = unit(cross(sub(at(positions, v[1]), v0), sub(at(positions, v[2]), v0)));
  const Vec x   = unit(sub(at(positions, v[4]), at(positions, v[3])));
  const Vec y   = unit(cross(nrm, x));
  return {float(x[0]), float(x[1]), float(x[2]), float(nrm[0]), float(nrm[1]), float(nrm[2]), float(-y[0]), float(-y[1]), float(-y[2])};
}

std::array<float, 9> cornerFrame(std::span<const float> cagePositions, std::span<const int32_t> faceStart,
                                 std::span<const int32_t> faceVerts, std::span<const int32_t> cornerFace, uint32_t faceCorner)
{
  return frameOf(cagePositions, frameVertices(faceStart, faceVerts, cornerFace, faceCorner));
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

HdMixer::HdMixer(std::shared_ptr<const Refinement> r, std::vector<int32_t> faceStart, std::vector<int32_t> faceVerts)
    : m_refinement(std::move(r))
    , m_faceStart(std::move(faceStart))
    , m_faceVerts(std::move(faceVerts))
    , m_cornerFace(cornerFaces(m_faceStart))
{
}

void HdMixer::remake()
{
  const Refinement& r     = *m_refinement;
  auto              topo  = std::make_shared<HdTables::Topology>();
  const size_t      count = r.levels.size();
  topo->levels.resize(count);
  // the frames by the face corner they belong to, in the order met
  std::unordered_map<uint32_t, uint32_t> frameOfCorner;
  const auto frameIndex = [&](uint32_t corner) {
    auto [it, added] = frameOfCorner.try_emplace(corner, uint32_t(frameOfCorner.size()));
    if(added)
    {
      const FrameVertices v = frameVertices(m_faceStart, m_faceVerts, m_cornerFace, corner);
      topo->frameVertices.insert(topo->frameVertices.end(), v.begin(), v.end());
    }
    return it->second;
  };
  for(size_t k = 0; k < count; k++)
  {
    // every known edit's (row, frame), once each, by row
    std::vector<std::pair<uint32_t, uint32_t>> keys;
    for(const Known& known : m_known)
      if(k < known.placed.levels.size())
      {
        const HdEdits::Level& l = known.placed.levels[k];
        for(size_t i = 0; i < l.vertex.size(); i++)
          keys.emplace_back(l.vertex[i], frameIndex(l.faceCorner[i]));
      }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    HdTables::Topology::Level& level = topo->levels[k];
    const uint32_t             rows  = uint32_t(r.levels[k].rows());
    level.start.assign(size_t(rows) + 1, 0);
    level.frame.resize(keys.size());
    for(size_t e = 0; e < keys.size(); e++)
    {
      level.start[keys[e].first + 1]++;
      level.frame[e] = keys[e].second;
    }
    for(uint32_t row = 0; row < rows; row++)
      level.start[row + 1] += level.start[row];
    // each known edit's place among them
    for(Known& known : m_known)
    {
      known.entry.resize(count);
      known.entry[k].clear();
      if(k >= known.placed.levels.size())
        continue;
      const HdEdits::Level& l = known.placed.levels[k];
      known.entry[k].resize(l.vertex.size());
      for(size_t i = 0; i < l.vertex.size(); i++)
      {
        const std::pair<uint32_t, uint32_t> key{l.vertex[i], frameIndex(l.faceCorner[i])};
        known.entry[k][i] = uint32_t(std::lower_bound(keys.begin(), keys.end(), key) - keys.begin());
      }
    }
  }
  m_topology = std::move(topo);
}

std::shared_ptr<const HdTables> HdMixer::mix(std::span<const Weighted> edits)
{
  if(!m_refinement || m_refinement->levels.empty())
    return nullptr;
  bool any = false, joined = false;
  for(const auto& [e, w] : edits)
  {
    if(!e || w == 0.f)
      continue;
    any = true;
    if(std::none_of(m_known.begin(), m_known.end(), [&](const Known& k) { return k.edits == e; }))
    {
      Known k;
      k.edits  = e;
      k.placed = placeEdits(*m_refinement, m_faceStart, *e);
      m_dropped += k.placed.dropped;
      m_known.push_back(std::move(k));
      joined = true;
    }
  }
  if(!any)
    return nullptr;
  if(joined || !m_topology)
    remake();

  auto out      = std::make_shared<HdTables>();
  out->topology = m_topology;
  out->offsets.resize(m_topology->levels.size());
  for(size_t k = 0; k < out->offsets.size(); k++)
    out->offsets[k].assign(m_topology->levels[k].frame.size() * 3, 0.f);
  for(const auto& [e, w] : edits)
  {
    if(!e || w == 0.f)
      continue;
    const Known& known = *std::find_if(m_known.begin(), m_known.end(), [&](const Known& k) { return k.edits == e; });
    for(size_t k = 0; k < known.placed.levels.size(); k++)
    {
      const HdEdits::Level& l = known.placed.levels[k];
      for(size_t i = 0; i < l.vertex.size(); i++)
        for(int c = 0; c < 3; c++)
          out->offsets[k][size_t(known.entry[k][i]) * 3 + c] += w * l.offset[i * 3 + c];
    }
  }
  for(const std::vector<float>& level : out->offsets)
  {
    float longest = 0.f;
    for(size_t i = 0; i + 2 < level.size(); i += 3)
      longest = std::max(longest, std::sqrt(level[i] * level[i] + level[i + 1] * level[i + 1] + level[i + 2] * level[i + 2]));
    out->reach += longest;
  }
  return out;
}

std::vector<float> refineWithTables(const Refinement& r, std::span<const float> cagePositions, const HdTables* tables)
{
  std::vector<std::array<float, 9>> frames;
  if(tables && tables->topology)
  {
    const std::vector<int32_t>& fv = tables->topology->frameVertices;
    for(size_t f = 0; f + 4 < fv.size(); f += 5)
      frames.push_back(frameOf(cagePositions, {fv[f], fv[f + 1], fv[f + 2], fv[f + 3], fv[f + 4]}));
  }
  std::vector<float> cur(cagePositions.begin(), cagePositions.end());
  for(size_t k = 0; k < r.levels.size(); k++)
  {
    const Stencils&    s = r.levels[k];
    std::vector<float> next(size_t(s.rows()) * 3, 0.f);
    const bool         edited = tables && tables->topology && k < tables->topology->levels.size();
    for(int row = 0; row < s.rows(); row++)
    {
      float sum[3] = {0.f, 0.f, 0.f};
      for(int j = s.rowStart[row]; j < s.rowStart[row + 1]; j++)
        for(int c = 0; c < 3; c++)
          sum[c] += s.weight[j] * cur[size_t(s.col[j]) * 3 + c];
      if(edited)
      {
        const HdTables::Topology::Level& l = tables->topology->levels[k];
        for(uint32_t e = l.start[size_t(row)]; e < l.start[size_t(row) + 1]; e++)
        {
          const std::array<float, 9>& m = frames[l.frame[e]];
          const float*                o = &tables->offsets[k][size_t(e) * 3];
          for(int c = 0; c < 3; c++)
            sum[c] += m[c] * o[0] + m[3 + c] * o[1] + m[6 + c] * o[2];
        }
      }
      std::copy_n(sum, 3, &next[size_t(row) * 3]);
    }
    cur = std::move(next);
  }
  return cur;
}

}  // namespace fv::subdiv
