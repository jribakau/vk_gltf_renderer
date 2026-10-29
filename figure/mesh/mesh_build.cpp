#include <figure/mesh/mesh_build.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/mesh/normals.hpp>

namespace fv::mesh {

namespace {

// Triangles fanned from each polygon's first corner, as vertex triplets.
std::vector<uint32_t> fan(std::span<const int32_t> start, std::span<const int32_t> verts, int count)
{
  std::vector<uint32_t> out;
  size_t                n = 0;
  for(int f = 0; f < count; f++)
    n += size_t(std::max(0, start[f + 1] - start[f] - 2));
  out.reserve(n * 3);
  for(int f = 0; f < count; f++)
    for(int k = start[f] + 2; k < start[f + 1]; k++)
    {
      out.push_back(uint32_t(verts[start[f]]));
      out.push_back(uint32_t(verts[k - 1]));
      out.push_back(uint32_t(verts[k]));
    }
  return out;
}

// The UV index of every polygon corner: the vertex's own, unless a seam override gives the corner another.
std::vector<int32_t> cornerUvs(const MeshInput& m)
{
  std::vector<int32_t> out(m.polyVerts);
  const auto&          o = m.overrides;
  for(size_t i = 0; i + 2 < o.size(); i += 3)
  {
    const int pi = o[i];
    if(pi < 0 || pi + 1 >= int(m.polyStart.size()))
      continue;
    for(int k = m.polyStart[pi]; k < m.polyStart[pi + 1]; k++)
      if(m.polyVerts[k] == o[i + 1])
        out[k] = o[i + 2];
  }
  return out;
}

// Box min and max, and the sphere around the box's centre (as three computes them).
Bounds boundsOf(std::span<const float> position)
{
  Bounds b;
  b.min = {INFINITY, INFINITY, INFINITY};
  b.max = {-INFINITY, -INFINITY, -INFINITY};
  for(size_t i = 0; i + 2 < position.size(); i += 3)
    for(int c = 0; c < 3; c++)
    {
      b.min[c] = std::min(b.min[c], position[i + c]);
      b.max[c] = std::max(b.max[c], position[i + c]);
    }
  const double cx = (double(b.min[0]) + b.max[0]) / 2, cy = (double(b.min[1]) + b.max[1]) / 2, cz = (double(b.min[2]) + b.max[2]) / 2;
  double       r2 = 0;
  for(size_t i = 0; i + 2 < position.size(); i += 3)
  {
    const double dx = position[i] - cx, dy = position[i + 1] - cy, dz = position[i + 2] - cz;
    r2              = std::max(r2, dx * dx + dy * dy + dz * dz);
  }
  b.radius = float(std::sqrt(r2));
  return b;
}

struct Strands
{
  // point c of the lines (lines.verts[c]) becomes drawn vertices 2c and 2c + 1; past the lines'
  // points come the segments' midpoints, in the order of the segments
  subdiv::Refinement refinement;
  // per point: the unit direction across the ribbon, root-to-tip position, and its line
  std::vector<float>   side, along;
  std::vector<int32_t> line;
  float                spacing = 0;
};

// Median distance from each root to its nearest neighbour, on a grid of half-centimetre cells.
float rootSpacing(std::span<const float> p, const Lines& lines, const std::function<bool(int)>& ok)
{
  constexpr double cell = 0.5;
  auto             cellOf = [&](int v, int dx, int dy, int dz) {
    const int64_t ix = int64_t(std::floor(p[v * 3] / cell)) + dx, iy = int64_t(std::floor(p[v * 3 + 1] / cell)) + dy,
                  iz = int64_t(std::floor(p[v * 3 + 2] / cell)) + dz;
    // 21 bits a coordinate is ±5 km of half-centimetre cells
    return ((ix & 0x1fffff) << 42) | ((iy & 0x1fffff) << 21) | (iz & 0x1fffff);
  };
  std::unordered_map<int64_t, std::vector<int>> grid;
  std::vector<int>                              roots;
  for(size_t l = 0; l + 1 < lines.start.size(); l++)
  {
    const int c = lines.start[l];
    if(c >= lines.start[l + 1] || !ok(c))
      continue;
    const int v = lines.verts[c];
    roots.push_back(v);
    grid[cellOf(v, 0, 0, 0)].push_back(v);
  }
  std::vector<double> found;
  for(int v : roots)
  {
    double best = INFINITY;
    for(int dx = -1; dx <= 1; dx++)
      for(int dy = -1; dy <= 1; dy++)
        for(int dz = -1; dz <= 1; dz++)
        {
          auto it = grid.find(cellOf(v, dx, dy, dz));
          if(it == grid.end())
            continue;
          for(int u : it->second)
          {
            if(u == v)
              continue;
            const double d = std::hypot(double(p[u * 3]) - p[v * 3], double(p[u * 3 + 1]) - p[v * 3 + 1], double(p[u * 3 + 2]) - p[v * 3 + 2]);
            // roots that share a point are one root drawn twice
            if(d > 1e-5 && d < best)
              best = d;
          }
        }
    if(best < INFINITY)
      found.push_back(best);
  }
  if(found.empty())
    return 0;
  std::sort(found.begin(), found.end());
  return float(found[found.size() >> 1]);
}

// Flat ribbons along polylines, as a one-level "refinement" that copies each line point into two
// drawn vertices. Everything refined from the cage (skin weights, morphs, fits) reaches both copies
// through it unchanged. A ribbon lies across the strand and the direction out from the roots'
// centre, so it lies flat on the hair's volume and faces outward like a card would. Its UVs are its
// own (across, along), so a texture on the file's UV set does not reach it.
//
// Each segment is drawn in pieces (kPieces), split at points the refinement places between its ends,
// so they follow them through the pose: a ray tracer bounds each triangle with a box, and a long thin
// segment lying diagonally fills its box with little but air. Halving the length halves how far the
// boxes of a lock's thousands of strands reach into one another.
Strands strandRibbons(const MeshInput& m)
{
  const Lines& L      = *m.lines;
  const auto&  p      = m.positions;
  const int    nv     = int(p.size() / 3);
  const int    nl     = int(L.start.size()) - 1;
  const int    points = L.start[nl];
  auto         ok     = [&](int c) { return L.verts[c] >= 0 && L.verts[c] < nv; };

  // the segments drawn: their first point and their line
  std::vector<std::array<int, 2>> segments;
  for(int l = 0; l < nl; l++)
    for(int c = L.start[l]; c + 1 < L.start[l + 1]; c++)
      if(ok(c) && ok(c + 1))
        segments.push_back({c, l});
  // each segment drawn as this many: 2 took a still frame 57.7 -> 49.0 ms and the moving trace
  // 18.6 -> 15.3; 4 took the still 3% further for 360 MB more and the moving view nowhere (2026-10-29)
  constexpr int kPieces = 2;
  constexpr int kInner  = kPieces - 1;
  const int     total   = points + int(segments.size()) * kInner;  // the lines' points, then the inner points
  // segment s's inner point j (0..kInner-1), at t = (j + 1) / kPieces of the way from its first point
  auto inner = [&](size_t s, int j) { return points + int(s) * kInner + j; };

  subdiv::Stencils stencil;
  stencil.rowStart.assign(size_t(total) * 2 + 1, 0);
  stencil.parentCount = nv;
  for(int c = 0; c < points; c++)
    for(int k = 0; k < 2; k++)
    {
      stencil.col.push_back(ok(c) ? L.verts[c] : 0);
      stencil.weight.push_back(1.f);
      stencil.rowStart[c * 2 + k + 1] = int(stencil.col.size());
    }
  for(size_t s = 0; s < segments.size(); s++)
    for(int j = 0; j < kInner; j++)
    {
      const int   c = segments[s][0], mid = inner(s, j);
      const float t = float(j + 1) / kPieces;
      for(int k = 0; k < 2; k++)
      {
        stencil.col.insert(stencil.col.end(), {L.verts[c], L.verts[c + 1]});
        stencil.weight.insert(stencil.weight.end(), {1.f - t, t});
        stencil.rowStart[mid * 2 + k + 1] = int(stencil.col.size());
      }
    }

  Strands out;
  auto&   r = out.refinement;
  for(size_t s = 0; s < segments.size(); s++)
  {
    const int c = segments[s][0], d = c + 1;
    for(int j = 0; j < kPieces; j++)
    {
      const int a = j == 0 ? c : inner(s, j - 1), b = j == kInner ? d : inner(s, j);
      for(int v : {a * 2, a * 2 + 1, b * 2 + 1, b * 2})
      {
        r.quads.push_back(v);
        // one UV per drawn vertex (below), so the tangent frame runs along the strand
        r.quadValues.push_back(v);
      }
      r.quadParent.push_back(segments[s][1]);
    }
  }

  // the roots' centre, which a ribbon faces away from
  double cx = 0, cy = 0, cz = 0;
  int    roots = 0;
  for(int l = 0; l < nl; l++)
  {
    const int c = L.start[l];
    if(c >= L.start[l + 1] || !ok(c))
      continue;
    cx += p[L.verts[c] * 3];
    cy += p[L.verts[c] * 3 + 1];
    cz += p[L.verts[c] * 3 + 2];
    roots++;
  }
  if(roots)
  {
    cx /= roots;
    cy /= roots;
    cz /= roots;
  }

  out.side.assign(size_t(points) * 3, 0.f);
  out.along.assign(points, 0.f);
  out.line.assign(points, 0);
  for(int l = 0; l < nl; l++)
  {
    const int s = L.start[l], e = L.start[l + 1];
    double    px = 0, py = 0, pz = 0;
    bool      have = false;
    for(int c = s; c < e; c++)
    {
      out.line[c]  = l;
      out.along[c] = e - s > 1 ? float(double(c - s) / (e - s - 1)) : 0.f;
      if(!ok(c))
        continue;
      const int    a  = c > s && ok(c - 1) ? L.verts[c - 1] : L.verts[c];
      const int    b  = c + 1 < e && ok(c + 1) ? L.verts[c + 1] : L.verts[c];
      const double tx = double(p[b * 3]) - p[a * 3], ty = double(p[b * 3 + 1]) - p[a * 3 + 1], tz = double(p[b * 3 + 2]) - p[a * 3 + 2];
      const int    v  = L.verts[c];
      const double rx = p[v * 3] - cx, ry = p[v * 3 + 1] - cy, rz = p[v * 3 + 2] - cz;
      double       sx = ty * rz - tz * ry, sy = tz * rx - tx * rz, sz = tx * ry - ty * rx;
      double       len = std::hypot(sx, sy, sz);
      const double tl  = std::hypot(tx, ty, tz);
      if(len < 1e-6 * tl * std::hypot(rx, ry, rz) || len == 0)
      {
        // the strand runs straight out from the centre here: keep the last direction, or take any across it
        if(have)
        {
          sx = px;
          sy = py;
          sz = pz;
        }
        else if(std::abs(ty) < 0.9 * tl)
        {
          sx = tz;
          sy = 0;
          sz = -tx;
        }
        else
        {
          sx = 0;
          sy = -tz;
          sz = ty;
        }
        len = std::hypot(sx, sy, sz);
        if(len == 0)
          len = 1;
      }
      sx /= len;
      sy /= len;
      sz /= len;
      out.side[c * 3]     = float(px = sx);
      out.side[c * 3 + 1] = float(py = sy);
      out.side[c * 3 + 2] = float(pz = sz);
      have                = true;
    }
  }

  // a midpoint lies across the strand between its ends' directions, halfway along
  out.side.resize(size_t(total) * 3);
  out.along.resize(total);
  out.line.resize(total);
  for(size_t s = 0; s < segments.size(); s++)
    for(int j = 0; j < kInner; j++)
    {
      const int   c = segments[s][0], mid = inner(s, j), d = c + 1;
      const float t = float(j + 1) / kPieces;
      float       dir[3];
      for(int k = 0; k < 3; k++)
        dir[k] = (1.f - t) * out.side[c * 3 + k] + t * out.side[d * 3 + k];
      const float len = std::hypot(dir[0], dir[1], dir[2]);
      for(int k = 0; k < 3; k++)
        out.side[mid * 3 + k] = len > 1e-6f ? dir[k] / len : out.side[c * 3 + k];
      out.along[mid] = (1.f - t) * out.along[c] + t * out.along[d];
      out.line[mid]  = out.line[c];
    }

  // UVs across (0 on the -side edge, 1 on the +side) and along (0 root, 1 tip) the ribbon, not the
  // file's: those are each strand's root on the scalp, the same all along it, which leaves no
  // strand direction for an anisotropic highlight to follow
  r.values.assign(size_t(total) * 4, 0.f);
  for(int c = 0; c < total; c++)
  {
    r.values[c * 4 + 1] = out.along[c];
    r.values[c * 4 + 2] = 1;
    r.values[c * 4 + 3] = out.along[c];
  }
  r.levels.push_back(std::move(stencil));
  r.vertexCount = total * 2;
  out.spacing   = rootSpacing(p, L, ok);
  return out;
}

// Bone influences per vertex as compressed rows: vertex v's are [start[v], start[v + 1]).
struct Influences
{
  std::vector<int32_t> start, bone;
  std::vector<double>  weight;
};

// Sorts entries [from, to) of a row by weight, strongest first, keeping ties in order.
void strongestFirst(Influences& inf, int from, int to)
{
  for(int i = from + 1; i < to; i++)
  {
    const int32_t b = inf.bone[i];
    const double  w = inf.weight[i];
    int           j = i - 1;
    for(; j >= from && inf.weight[j] < w; j--)
    {
      inf.bone[j + 1]   = inf.bone[j];
      inf.weight[j + 1] = inf.weight[j];
    }
    inf.bone[j + 1]   = b;
    inf.weight[j + 1] = w;
  }
}

// Rows worked a chunk at a time side by side, each chunk into lists of its own, then put one after
// another: `row(r, out)` appends row r's entries to the chunk's lists. What comes out is what one
// pass over the rows makes.
template <class Row>
Influences rowsSideBySide(int rows, Row&& row)
{
  constexpr int           kChunk = 8192;
  const int               chunks = (rows + kChunk - 1) / kChunk;
  std::vector<Influences> parts(size_t(std::max(chunks, 0)));
  jobs::parallelFor(parts.size(), 1, [&](size_t begin, size_t end) {
    for(size_t c = begin; c < end; c++)
    {
      Influences& part  = parts[c];
      const int   first = int(c) * kChunk, last = std::min(rows, first + kChunk);
      part.start.assign(size_t(last - first) + 1, 0);
      for(int r = first; r < last; r++)
      {
        row(r, part);
        part.start[size_t(r - first) + 1] = int32_t(part.bone.size());
      }
    }
  });
  std::vector<size_t> base(parts.size() + 1, 0);
  for(size_t c = 0; c < parts.size(); c++)
    base[c + 1] = base[c] + parts[c].bone.size();
  Influences out;
  out.start.assign(size_t(std::max(rows, 0)) + 1, 0);
  out.bone.resize(base.back());
  out.weight.resize(base.back());
  jobs::parallelFor(parts.size(), 1, [&](size_t begin, size_t end) {
    for(size_t c = begin; c < end; c++)
    {
      const Influences& part = parts[c];
      std::copy(part.bone.begin(), part.bone.end(), out.bone.begin() + base[c]);
      std::copy(part.weight.begin(), part.weight.end(), out.weight.begin() + base[c]);
      for(size_t r = 1; r < part.start.size(); r++)
        out.start[c * kChunk + r] = int32_t(base[c] + size_t(part.start[r]));
    }
  });
  return out;
}

// One subdivision level's influences: each child vertex sums its parents' by the stencil weights,
// bones in the order first met. Under 1e-6 is dropped, and past eight only the strongest stay.
Influences refineInfluences(const Influences& src, const subdiv::Stencils& lv, int bones)
{
  // per bone, on each thread: its sum in this row, and the row it was last met in
  struct Scratch
  {
    std::vector<double>  acc;
    std::vector<int32_t> seen, order;
  };
  thread_local Scratch t;
  return rowsSideBySide(lv.rows(), [&](int row, Influences& out) {
    if(t.acc.size() < size_t(bones))
    {
      t.acc.resize(size_t(bones), 0.0);
      t.seen.resize(size_t(bones), -1);
      t.order.resize(size_t(bones), 0);
    }
    // the bones a row met are unmarked when it is done, so the next row (or call) starts clean
    int met = 0;
    for(int k = lv.rowStart[row]; k < lv.rowStart[row + 1]; k++)
    {
      const double w = lv.weight[k];
      const int    p = lv.col[k];
      for(int j = src.start[p]; j < src.start[p + 1]; j++)
      {
        const int b = src.bone[j];
        if(t.seen[b] != row)
        {
          t.seen[b]      = row;
          t.acc[b]       = 0;
          t.order[met++] = b;
        }
        t.acc[b] += src.weight[j] * w;
      }
    }
    for(int i = 0; i < met; i++)
      t.seen[t.order[i]] = -1;
    const int from = int(out.bone.size());
    for(int i = 0; i < met; i++)
    {
      const int b = t.order[i];
      if(t.acc[b] > 1e-6)
      {
        out.bone.push_back(b);
        out.weight.push_back(t.acc[b]);
      }
    }
    if(int(out.bone.size()) - from > 8)
    {
      strongestFirst(out, from, int(out.bone.size()));
      out.bone.resize(from + 8);
      out.weight.resize(from + 8);
    }
  });
}

}  // namespace

// Opens the zero-width ribbons to `widths[material]` (root, tip in cm), then welds normals across them.
void widenRibbons(mesh::BuiltMesh& m, const std::vector<std::array<double, 2>>& widths)
{
  if(!m.ribbon)
    return;
  auto& r = *m.ribbon;
  r.half.assign(r.along.size(), 0.f);
  // each vertex moves on its own, and a hair has millions of them
  jobs::parallelFor(r.along.size(), 16384, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      const int  mat = r.material[i];
      const auto w   = mat >= 0 && mat < int(widths.size()) ? widths[mat] : (widths.empty() ? std::array<double, 2>{0.1, 0.05} : widths[0]);
      // The first 6% of the strand grows from nothing, so the root is not a full-width sheet and the
      // body can stay opaque (no any-hit on every card). along is 0 at the root.
      constexpr double kRoot = 0.06;
      double           gate  = 1;
      if(r.along[i] < kRoot)
      {
        const double t = r.along[i] / kRoot;
        gate           = t * t * (3.0 - 2.0 * t);
      }
      const double half = (w[0] + (w[1] - w[0]) * r.along[i]) / 2 * gate;
      r.half[i]         = float(half);
      for(int k = 0; k < 3; k++)
        m.position[i * 3 + k] = float(m.position[i * 3 + k] + r.side[i * 3 + k] * half);
    }
  });
  NormalWelder(m.renderIndex, m.firstSplit, m.nextSplit, m.renderTris, m.renderVertexCount).full(m.position, m.normal);
}

BuiltMesh build(const MeshInput& m)
{
  BuiltMesh out;
  const int nv = int(m.positions.size() / 3);
  const int np = int(m.polyStart.size()) - 1;
  out.dsonTris        = fan(m.polyStart, m.polyVerts, np);
  out.dsonVertexCount = nv;
  out.polyCount       = std::max(np, 0);
  out.cagePositions   = m.positions;
  if(np > 0)
  {
    out.cagePolyStart = m.polyStart;
    out.cagePolyVerts = m.polyVerts;
  }
  else if(m.lines && m.lines->start.size() > 1)
  {
    out.cageLineStart = m.lines->start;
    out.cageLineVerts = m.lines->verts;
  }
  const std::vector<int32_t> uvOfCorner = cornerUvs(m);

  // the faces drawn: the cage's polygons, or the refined quads with the cage polygon each came from
  int                      faceCount = 0;
  std::span<const int32_t> faceStart;  // empty: quads, four corners each
  std::span<const int32_t> faceVerts, faceUvs, facePoly;  // facePoly empty: face f is polygon f
  std::vector<float>       renderPositions;
  int                      renderCount = 0;
  std::span<const float>   uvValues;
  std::optional<Strands>   strands;
  if(np <= 0 && m.lines && m.lines->start.size() > 1)
    strands = strandRibbons(m);
  // a strand's material is its line's
  std::span<const int32_t>               faceMaterial = strands ? std::span<const int32_t>(m.lines->material) : std::span<const int32_t>(m.polyMaterial);
  std::shared_ptr<subdiv::Refinement>    refinement;
  if(strands)
  {
    refinement      = std::make_shared<subdiv::Refinement>(std::move(strands->refinement));
    renderPositions = subdiv::refineDense(*refinement, m.positions, 3);
    renderCount     = refinement->vertexCount;
  }
  else if(m.level > 0 && np > 0)
  {
    int maxUv = int(m.uvs.size() / 2) - 1;
    for(int32_t uv : uvOfCorner)
      maxUv = std::max(maxUv, uv);
    std::vector<float> values(size_t(maxUv + 1) * 2, 0.f);
    std::copy_n(m.uvs.begin(), std::min(m.uvs.size(), values.size()), values.begin());
    refinement      = std::make_shared<subdiv::Refinement>(subdiv::refine({nv, m.polyStart, m.polyVerts, uvOfCorner, values}, m.level, m.boundary));
    renderPositions = subdiv::refineDense(*refinement, m.positions, 3);
    renderCount     = refinement->vertexCount;
  }
  else
  {
    renderPositions = m.positions;
    renderCount     = nv;
    faceCount       = std::max(np, 0);
    faceStart       = m.polyStart;
    faceVerts       = m.polyVerts;
    faceUvs         = uvOfCorner;
    uvValues        = m.uvs;
  }
  if(refinement)
  {
    faceCount = int(refinement->quadParent.size());
    faceVerts = refinement->quads;
    faceUvs   = refinement->quadValues;
    facePoly  = refinement->quadParent;
    uvValues  = refinement->values;
  }
  out.renderVertexCount = renderCount;

  // Split the render vertices by UV: each corner takes the split of its vertex with its UV, made when
  // first seen. A vertex's splits form a list in the order they were made (firstSplit, nextSplit).
  const int            corners = faceStart.empty() ? faceCount * 4 : faceStart[faceCount];
  std::vector<int32_t> firstSplit(renderCount, -1);
  std::vector<int32_t> nextSplitAll(corners), splitUv(corners);
  std::vector<uint32_t> splitVertex(corners), cornerSplit(corners);
  int                  count = 0;
  for(int c = 0; c < corners; c++)
  {
    const int v = faceVerts[c], uv = faceUvs.empty() ? v : faceUvs[c];
    int       sp = firstSplit[v], last = -1;
    while(sp >= 0 && splitUv[sp] != uv)
    {
      last = sp;
      sp   = nextSplitAll[sp];
    }
    if(sp < 0)
    {
      sp               = count++;
      splitVertex[sp]  = uint32_t(v);
      splitUv[sp]      = uv;
      nextSplitAll[sp] = -1;
      if(last < 0)
        firstSplit[v] = sp;
      else
        nextSplitAll[last] = sp;
    }
    cornerSplit[c] = uint32_t(sp);
  }
  out.renderIndex.assign(splitVertex.begin(), splitVertex.begin() + count);
  out.nextSplit.assign(nextSplitAll.begin(), nextSplitAll.begin() + count);
  out.firstSplit = std::move(firstSplit);
  out.position.assign(size_t(count) * 3, 0.f);
  out.uv.assign(size_t(count) * 2, 0.f);
  for(int i = 0; i < count; i++)
  {
    const size_t v       = size_t(out.renderIndex[i]) * 3;
    out.position[i * 3]     = renderPositions[v];
    out.position[i * 3 + 1] = renderPositions[v + 1];
    out.position[i * 3 + 2] = renderPositions[v + 2];
    const int u          = splitUv[i];
    if(u >= 0 && size_t(u) * 2 + 1 < uvValues.size())
    {
      out.uv[i * 2]     = uvValues[u * 2];
      out.uv[i * 2 + 1] = uvValues[u * 2 + 1];
    }
  }

  // Triangles: fanned per face, in face order (renderTris, on render vertices), and grouped by
  // material for drawing (fullIndex, on the splits, with the polygon of each in triPoly). A face whose
  // material index is negative is drawn by no group.
  auto startOf  = [&](int f) { return faceStart.empty() ? f * 4 : faceStart[f]; };
  auto sizeOf   = [&](int f) { return faceStart.empty() ? 4 : faceStart[f + 1] - faceStart[f]; };
  auto polyOf   = [&](int f) { return facePoly.empty() ? f : facePoly[f]; };
  int  materials = m.materialCount;
  for(int32_t mat : faceMaterial)
    materials = std::max(materials, mat + 1);
  std::vector<int32_t> perMaterial(materials, 0);
  size_t               triCount = 0;
  for(int f = 0; f < faceCount; f++)
  {
    const int n   = std::max(0, sizeOf(f) - 2);
    const int mat = faceMaterial[polyOf(f)];
    triCount += n;
    if(mat >= 0)
      perMaterial[mat] += n;
  }
  std::vector<int32_t> next(materials, 0);
  uint32_t             drawn = 0;
  for(int mi = 0; mi < materials; mi++)
  {
    next[mi] = int32_t(drawn);
    if(perMaterial[mi])
      out.groups.push_back({drawn * 3, uint32_t(perMaterial[mi]) * 3, mi});
    drawn += perMaterial[mi];
  }
  out.renderTris.assign(triCount * 3, 0);
  out.fullIndex.assign(size_t(drawn) * 3, 0);
  out.triPoly.assign(drawn, 0);
  size_t rt = 0;
  for(int f = 0; f < faceCount; f++)
  {
    const int pi = polyOf(f);
    const int mat = faceMaterial[pi];
    const int s = startOf(f), n = sizeOf(f);
    for(int k = 1; k + 1 < n; k++)
    {
      out.renderTris[rt++] = uint32_t(faceVerts[s]);
      out.renderTris[rt++] = uint32_t(faceVerts[s + k]);
      out.renderTris[rt++] = uint32_t(faceVerts[s + k + 1]);
      if(mat < 0)
        continue;
      const int t               = next[mat]++;
      out.fullIndex[t * 3]     = cornerSplit[s];
      out.fullIndex[t * 3 + 1] = cornerSplit[s + k];
      out.fullIndex[t * 3 + 2] = cornerSplit[s + k + 1];
      out.triPoly[t]           = uint32_t(pi);
    }
  }

  out.normal.assign(size_t(count) * 3, 0.f);
  // Strand ribbons have no width yet, so every triangle is degenerate and a weld would write zeros:
  // widenRibbons welds them once it has opened them.
  if(!strands)
  {
    timing::Scope tw("weld", 10);
    NormalWelder(out.renderIndex, out.firstSplit, out.nextSplit, out.renderTris, renderCount).full(out.position, out.normal);
  }

  // drawn vertex 2c is point c's edge on the -side, 2c + 1 on the +side
  if(strands)
  {
    Ribbon ribbon;
    ribbon.side.assign(size_t(count) * 3, 0.f);
    ribbon.along.assign(count, 0.f);
    ribbon.material.assign(count, 0);
    for(int i = 0; i < count; i++)
    {
      const uint32_t rv   = out.renderIndex[i];
      const uint32_t c    = rv >> 1;
      const float    sign = rv & 1 ? 1.f : -1.f;
      for(int k = 0; k < 3; k++)
        ribbon.side[i * 3 + k] = strands->side[c * 3 + k] * sign;
      ribbon.along[i]    = strands->along[c];
      ribbon.material[i] = m.lines->material[strands->line[c]];
    }
    ribbon.spacing = strands->spacing;
    out.ribbon     = std::move(ribbon);
  }
  out.refinement = std::move(refinement);
  out.bounds     = boundsOf(out.position);
  return out;
}

SkinOutput skin(const BuiltMesh& mesh, const SkinInput& s)
{
  // the cage's influences, without the zero ones
  const int rows  = int(s.start.size()) - 1;
  int       bones = s.rootBone + 1;
  for(size_t k = 0; k < s.bone.size(); k++)
    if(s.weight[k] > 0)
      bones = std::max(bones, s.bone[k] + 1);
  Influences inf = rowsSideBySide(std::max(rows, 0), [&](int v, Influences& out) {
    for(int k = s.start[v]; k < s.start[v + 1]; k++)
      if(s.weight[k] > 0)
      {
        out.bone.push_back(s.bone[k]);
        out.weight.push_back(s.weight[k]);
      }
  });
  if(mesh.refinement)
    for(const auto& lv : mesh.refinement->levels)
      inf = refineInfluences(inf, lv, bones);

  // the rows that keep four of more, strongest first; then each drawn vertex (a vertex's splits
  // share its row) takes its row's first four, side by side
  const int ninf = int(inf.start.size()) - 1;
  jobs::parallelFor(size_t(std::max(ninf, 0)), 16384, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
      if(inf.start[v + 1] - inf.start[v] > 4)
        strongestFirst(inf, inf.start[v], inf.start[v + 1]);
  });
  const size_t count = mesh.renderIndex.size();
  SkinOutput   out;
  out.skinIndex.assign(count * 4, 0);
  out.skinWeight.assign(count * 4, 0.f);
  // eight kept only where some vertex has more than four
  bool more = false;
  for(int v = 0; v < ninf && !more; v++)
    more = inf.start[v + 1] - inf.start[v] > 4;
  if(more)
  {
    out.skinIndex8.assign(count * 8, 0);
    out.skinWeight8.assign(count * 8, 0.f);
  }
  jobs::parallelFor(count, 16384, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      const uint32_t v    = mesh.renderIndex[i];
      const bool     has  = v + 1 < inf.start.size();
      const int      from = has ? inf.start[v] : 0;
      const int      n    = has ? inf.start[v + 1] - from : 0;
      const int      m    = std::min(4, n);
      double         sum  = 0;
      for(int k = 0; k < m; k++)
        sum += inf.weight[from + k];
      if(sum <= 0)
      {
        out.skinIndex[i * 4]  = uint16_t(s.rootBone);
        out.skinWeight[i * 4] = 1;
        continue;
      }
      for(int k = 0; k < m; k++)
      {
        out.skinIndex[i * 4 + k]  = uint16_t(inf.bone[from + k]);
        out.skinWeight[i * 4 + k] = float(inf.weight[from + k] / sum);
      }
      if(!more)
        continue;
      const int m8    = std::min(8, n);
      double    sum8  = 0;
      for(int k = 0; k < m8; k++)
        sum8 += inf.weight[from + k];
      for(int k = 0; k < m8; k++)
      {
        out.skinIndex8[i * 8 + k]  = uint16_t(inf.bone[from + k]);
        out.skinWeight8[i * 8 + k] = float(inf.weight[from + k] / sum8);
      }
    }
  });
  // where there were no weights the four above hold the root bone alone: so do the eight
  if(more)
    for(size_t i = 0; i < count; i++)
      if(out.skinWeight8[i * 8] == 0.f)
      {
        out.skinIndex8[i * 8]  = out.skinIndex[i * 4];
        out.skinWeight8[i * 8] = out.skinWeight[i * 4];
      }

  // the dual quaternion share: a map is refined like positions, then each drawn vertex takes its row's
  out.dualQuat = s.dualQuat;
  if(!s.dualQuatMap.empty() && int(s.dualQuatMap.size()) == rows)
  {
    std::vector<double> share(s.dualQuatMap.begin(), s.dualQuatMap.end());
    if(mesh.refinement)
      for(const auto& lv : mesh.refinement->levels)
      {
        std::vector<double> next(size_t(std::max(lv.rows(), 0)), 0.0);
        for(int row = 0; row < lv.rows(); row++)
          for(int k = lv.rowStart[row]; k < lv.rowStart[row + 1]; k++)
            if(size_t(lv.col[k]) < share.size())
              next[size_t(row)] += double(lv.weight[k]) * share[size_t(lv.col[k])];
        share = std::move(next);
      }
    out.dualQuatShare.assign(count, s.dualQuat);
    for(size_t i = 0; i < count; i++)
      if(mesh.renderIndex[i] < share.size())
        out.dualQuatShare[i] = float(share[mesh.renderIndex[i]]);
  }
  return out;
}

RenderTarget target(const BuiltMesh& mesh, const CageDeltas& d)
{
  std::vector<uint32_t> cageIndices;
  std::vector<float>    cageDeltas;
  for(size_t i = 0; i < d.verts.size(); i++)
  {
    const int v = d.verts[i];
    if(v < 0 || v >= mesh.dsonVertexCount)
      continue;
    cageIndices.push_back(uint32_t(v));
    cageDeltas.insert(cageDeltas.end(), d.deltas.begin() + i * 3, d.deltas.begin() + i * 3 + 3);
  }
  return target(mesh, std::move(cageIndices), std::move(cageDeltas));
}

RenderTarget target(const BuiltMesh&, std::vector<uint32_t> cageIndices, std::vector<float> cageDeltas)
{
  RenderTarget out;
  out.cageIndices = std::move(cageIndices);
  out.cageDeltas  = std::move(cageDeltas);
  return out;
}

const RenderTarget::Drawn& RenderTarget::drawn(const BuiltMesh& mesh) const
{
  std::call_once(m_lazy->once, [&] {
    Drawn&         out = m_lazy->drawn;
    subdiv::Sparse render;
    if(mesh.refinement)
      render = subdiv::refineSparse(*mesh.refinement, cageIndices, cageDeltas, 3);
    else
      render = {cageIndices, cageDeltas};
    // every split of a vertex that moves takes its delta: counted, then written, a chunk at a time
    // side by side, in the order one pass would write them
    constexpr size_t    kChunk = 16384;
    const size_t        n = render.indices.size(), chunks = (n + kChunk - 1) / kChunk;
    std::vector<size_t> at(chunks + 1, 0);
    const auto moves = [&](size_t i) { return render.data[i * 3] != 0 || render.data[i * 3 + 1] != 0 || render.data[i * 3 + 2] != 0; };
    const auto split = [&](uint32_t v) { return v < mesh.firstSplit.size() ? mesh.firstSplit[v] : -1; };
    jobs::parallelFor(chunks, 1, [&](size_t begin, size_t end) {
      for(size_t c = begin; c < end; c++)
        for(size_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); i++)
          if(moves(i))
            for(int32_t s = split(render.indices[i]); s >= 0; s = mesh.nextSplit[s])
              at[c + 1]++;
    });
    std::partial_sum(at.begin(), at.end(), at.begin());
    out.indices.resize(at[chunks]);
    out.deltas.resize(at[chunks] * 3);
    jobs::parallelFor(chunks, 1, [&](size_t begin, size_t end) {
      for(size_t c = begin; c < end; c++)
      {
        size_t o = at[c];
        for(size_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); i++)
          if(moves(i))
            for(int32_t s = split(render.indices[i]); s >= 0; s = mesh.nextSplit[s], o++)
            {
              out.indices[o] = uint32_t(s);
              std::copy_n(&render.data[i * 3], 3, &out.deltas[o * 3]);
            }
      }
    });
    m_lazy->made = true;
  });
  return m_lazy->drawn;
}

const RenderTarget::Drawn* RenderTarget::drawnIfMade() const
{
  return m_lazy->made ? &m_lazy->drawn : nullptr;
}

void RenderTarget::setDrawn(Drawn d)
{
  std::call_once(m_lazy->once, [&] {
    m_lazy->drawn = std::move(d);
    m_lazy->made  = true;
  });
}

bool RenderTarget::moves() const
{
  const auto any = [](const std::vector<float>& v) { return std::any_of(v.begin(), v.end(), [](float x) { return x != 0.f; }); };
  if(const Drawn* d = drawnIfMade(); d && cageDeltas.empty())
    return any(d->deltas);
  return any(cageDeltas);
}

bool drawnFromCage(const BuiltMesh& mesh)
{
  return mesh.refinement && (!mesh.ribbon || !mesh.ribbon->half.empty());
}

std::vector<float> shiftCage(BuiltMesh& mesh, std::span<const float> positions)
{
  if(positions.size() != mesh.cagePositions.size())
    throw std::runtime_error("the cage has " + std::to_string(mesh.cagePositions.size() / 3) + " vertices");
  std::vector<float> cageDelta(positions.size());
  for(size_t i = 0; i < positions.size(); i++)
    cageDelta[i] = float(double(positions[i]) - mesh.cagePositions[i]);
  std::copy(positions.begin(), positions.end(), mesh.cagePositions.begin());
  const std::vector<float> render = mesh.refinement ? subdiv::refineDense(*mesh.refinement, cageDelta, 3) : cageDelta;
  std::vector<float>       delta(mesh.renderIndex.size() * 3);
  for(size_t i = 0; i < mesh.renderIndex.size(); i++)
  {
    const size_t v   = size_t(mesh.renderIndex[i]) * 3;
    delta[i * 3]     = render[v];
    delta[i * 3 + 1] = render[v + 1];
    delta[i * 3 + 2] = render[v + 2];
  }
  return delta;
}

}  // namespace fv::mesh
