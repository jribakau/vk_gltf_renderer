#include <figure/subdiv/catmark.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>

#include <figure/jobs/pool.hpp>

namespace fv::subdiv {

namespace {

// ------------------------------------------------------------------------------------------------
// topology of one level

struct Level
{
  int                  nv = 0, nf = 0, ne = 0;
  std::vector<int32_t> faceStart, faceVerts, faceEdges, edgeVerts;
  std::vector<int32_t> edgeFaceStart, edgeFaces;
  std::vector<int32_t> vertFaceStart, vertFaces, vertFaceCorner;
  std::vector<int32_t> vertEdgeStart, vertEdges;
  // Infinitely sharp: boundary or non-manifold edges.
  std::vector<uint8_t> edgeSharp, edgeNonManifold;
  // Infinitely sharp vertices: sharpened corners and non-manifold vertices.
  std::vector<uint8_t> vertSharp, vertNonManifold;
  // face-varying values (UVs); empty without
  std::vector<int32_t> faceValues;
  std::vector<float>   values;
  bool                 hasValues = false;
};

// Prefix sums of per-key counts (in start[k + 1]), in place; returns a copy of the starts to fill from.
std::vector<int32_t> starts(std::vector<int32_t>& start)
{
  for(size_t i = 1; i < start.size(); i++)
    start[i] += start[i - 1];
  return std::vector<int32_t>(start.begin(), start.end() - 1);
}

// Fills the derived relations (edge-faces, vertex-faces, vertex-edges) from faces, face-edges and
// edge-vertices. Each list is in the order its items are met: faces in order, corners in order.
void relate(Level& l)
{
  const size_t corners = l.faceVerts.size();
  l.edgeFaceStart.assign(l.ne + 1, 0);
  l.vertFaceStart.assign(l.nv + 1, 0);
  l.vertEdgeStart.assign(l.nv + 1, 0);
  for(size_t c = 0; c < corners; c++)
  {
    l.edgeFaceStart[l.faceEdges[c] + 1]++;
    l.vertFaceStart[l.faceVerts[c] + 1]++;
  }
  for(int i = 0; i < l.ne * 2; i++)
    l.vertEdgeStart[l.edgeVerts[i] + 1]++;
  auto ef = starts(l.edgeFaceStart), vf = starts(l.vertFaceStart), ve = starts(l.vertEdgeStart);
  l.edgeFaces.assign(corners, 0);
  l.vertFaces.assign(corners, 0);
  l.vertFaceCorner.assign(corners, 0);
  for(int f = 0; f < l.nf; f++)
    for(int c = l.faceStart[f]; c < l.faceStart[f + 1]; c++)
    {
      l.edgeFaces[ef[l.faceEdges[c]]++] = f;
      const int at        = vf[l.faceVerts[c]]++;
      l.vertFaces[at]      = f;
      l.vertFaceCorner[at] = c - l.faceStart[f];
    }
  l.vertEdges.assign(size_t(l.ne) * 2, 0);
  for(int e = 0; e < l.ne; e++)
  {
    l.vertEdges[ve[l.edgeVerts[2 * e]]++]     = e;
    l.vertEdges[ve[l.edgeVerts[2 * e + 1]]++] = e;
  }
}

struct Ring
{
  std::vector<int> faces, corners;
  bool             boundary = false;
};

// The faces around a manifold vertex in order, each with its corner; none when they don't form one fan.
std::optional<Ring> ringOf(const Level& l, int v)
{
  const int s = l.vertFaceStart[v], n = l.vertFaceStart[v + 1] - s;
  if(!n)
    return std::nullopt;
  auto nextEdge = [&](int k) {
    const int f = l.vertFaces[s + k], fs = l.faceStart[f];
    return l.faceEdges[fs + l.vertFaceCorner[s + k]];
  };
  auto prevEdge = [&](int k) {
    const int f = l.vertFaces[s + k], fs = l.faceStart[f], size = l.faceStart[f + 1] - fs;
    const int c = l.vertFaceCorner[s + k];
    return l.faceEdges[fs + (c ? c - 1 : size - 1)];
  };
  auto faceCount = [&](int e) { return l.edgeFaceStart[e + 1] - l.edgeFaceStart[e]; };
  // a boundary vertex starts at the face whose leading edge (to its next corner) is a boundary
  int  start    = 0;
  bool boundary = false;
  for(int k = 0; k < n; k++)
  {
    if(faceCount(nextEdge(k)) == 1)
    {
      if(boundary)
        return std::nullopt;
      boundary = true;
      start    = k;
    }
  }
  Ring                 ring;
  std::vector<uint8_t> seen(n, 0);
  int                  k = start;
  for(;;)
  {
    if(seen[k])
      return std::nullopt;
    seen[k] = 1;
    ring.faces.push_back(l.vertFaces[s + k]);
    ring.corners.push_back(l.vertFaceCorner[s + k]);
    const int e = prevEdge(k);
    if(faceCount(e) == 1)
      break;
    if(faceCount(e) != 2)
      return std::nullopt;
    // the other face across e, entered through its leading edge
    int next = -1;
    for(int j = 0; j < n; j++)
      if(j != k && nextEdge(j) == e)
        next = j;
    if(next < 0)
      return std::nullopt;
    if(next == start)
      break;
    k = next;
  }
  ring.boundary = boundary;
  if(int(ring.faces.size()) != n)
    return std::nullopt;
  return ring;
}

// The cage as OpenSubdiv's base level: edges in order of first use, non-manifold features sharpened.
Level baseLevel(const CageMesh& m, BoundaryInterpolation boundaryMode)
{
  Level l;
  l.nf = int(m.faceStart.size()) - 1;
  l.nv = m.vertexCount;
  l.faceStart.assign(m.faceStart.begin(), m.faceStart.end());
  l.faceVerts.assign(m.faceVerts.begin(), m.faceVerts.end());
  l.faceEdges.assign(m.faceVerts.size(), 0);
  std::vector<uint8_t> edgeNm;
  // per edge, how many faces use it and the last one: all the build needs to know of its faces
  std::vector<int>                     edgeFaceCount, edgeLastFace;
  std::unordered_map<int64_t, int32_t> firstEdge;
  for(int f = 0; f < l.nf; f++)
  {
    const int fs = l.faceStart[f], size = l.faceStart[f + 1] - fs;
    for(int i = 0; i < size; i++)
    {
      const int     v0 = l.faceVerts[fs + i], v1 = l.faceVerts[fs + ((i + 1) % size)];
      const int64_t key = int64_t(std::min(v0, v1)) * l.nv + std::max(v0, v1);
      int           e   = -1;
      if(v0 != v1)
      {
        auto it = firstEdge.find(key);
        if(it != firstEdge.end())
          e = it->second;
      }
      bool newIsNm = v0 == v1;
      if(e >= 0)
      {
        if(edgeLastFace[e] == f)
        {
          edgeNm[e] = 1;
          newIsNm   = true;
          e         = -1;
        }
        else if(edgeFaceCount[e] > 1 || v0 == l.edgeVerts[2 * e])
          edgeNm[e] = 1;
      }
      if(e < 0)
      {
        e = int(edgeFaceCount.size());
        l.edgeVerts.push_back(v0);
        l.edgeVerts.push_back(v1);
        edgeNm.push_back(newIsNm ? 1 : 0);
        edgeFaceCount.push_back(0);
        edgeLastFace.push_back(-1);
        if(v0 != v1 && !firstEdge.count(key))
          firstEdge[key] = e;
      }
      edgeFaceCount[e]++;
      edgeLastFace[e]    = f;
      l.faceEdges[fs + i] = e;
    }
  }
  l.ne = int(edgeFaceCount.size());
  l.edgeSharp.assign(l.ne, 0);
  l.edgeNonManifold = std::move(edgeNm);
  l.vertSharp.assign(l.nv, 0);
  l.vertNonManifold.assign(l.nv, 0);
  if(!m.faceValues.empty() && !m.values.empty())
  {
    l.faceValues.assign(m.faceValues.begin(), m.faceValues.end());
    l.values.assign(m.values.begin(), m.values.end());
    l.hasValues = true;
  }
  relate(l);
  for(int e = 0; e < l.ne; e++)
  {
    if(l.edgeNonManifold[e])
      l.vertNonManifold[l.edgeVerts[2 * e]] = l.vertNonManifold[l.edgeVerts[2 * e + 1]] = 1;
    if(l.edgeFaceStart[e + 1] - l.edgeFaceStart[e] < 2 || l.edgeNonManifold[e])
      l.edgeSharp[e] = 1;
  }
  for(int v = 0; v < l.nv; v++)
  {
    const int faces = l.vertFaceStart[v + 1] - l.vertFaceStart[v];
    const int edges = l.vertEdgeStart[v + 1] - l.vertEdgeStart[v];
    if(!l.vertNonManifold[v] && faces && !ringOf(l, v))
      l.vertNonManifold[v] = 1;
    const bool corner = faces == 1 && edges == 2;
    if((corner && boundaryMode == BoundaryInterpolation::eEdgesAndCorners) || l.vertNonManifold[v])
      l.vertSharp[v] = 1;
  }
  return l;
}

// ------------------------------------------------------------------------------------------------
// rules

// Accumulates one stencil row, merging repeated columns: a parent's slot is looked up, not searched
// for. Columns keep the order they were first added in, and weights sum in doubles.
class Row
{
public:
  explicit Row(int parents)
      : m_slot(parents, 0)
      , m_stamp(parents, 0)
  {
  }

  void add(int col, double w)
  {
    if(m_stamp[col] == m_id)
    {
      weights[m_slot[col]] += w;
      return;
    }
    m_stamp[col] = m_id;
    m_slot[col]  = int(cols.size());
    cols.push_back(col);
    weights.push_back(w);
  }

  void clear()
  {
    cols.clear();
    weights.clear();
    m_id++;
  }

  std::vector<int32_t> cols;
  std::vector<double>  weights;

private:
  std::vector<int32_t> m_slot;
  std::vector<int32_t> m_stamp;
  int32_t              m_id = 1;
};

void addFaceCenter(const Level& l, int f, double w, Row& row)
{
  const int fs = l.faceStart[f], size = l.faceStart[f + 1] - fs;
  for(int c = fs; c < fs + size; c++)
    row.add(l.faceVerts[c], w / size);
}

int otherEnd(const Level& l, int e, int v)
{
  return l.edgeVerts[2 * e] == v ? l.edgeVerts[2 * e + 1] : l.edgeVerts[2 * e];
}

void vertexRow(const Level& l, int v, Row& row)
{
  if(l.vertSharp[v])
    return row.add(v, 1);
  int sharp = 0, a = -1, b = -1;
  for(int k = l.vertEdgeStart[v]; k < l.vertEdgeStart[v + 1]; k++)
  {
    const int e = l.vertEdges[k];
    if(!l.edgeSharp[e])
      continue;
    const int other = otherEnd(l, e, v);
    if(sharp == 0)
      a = other;
    else
      b = other;
    sharp++;
  }
  if(sharp > 2)
    return row.add(v, 1);
  if(sharp == 2)
  {
    row.add(v, 0.75);
    row.add(a, 0.125);
    row.add(b, 0.125);
    return;
  }
  const int    n = l.vertFaceStart[v + 1] - l.vertFaceStart[v];
  const double w = 1.0 / (double(n) * n);
  for(int k = l.vertEdgeStart[v]; k < l.vertEdgeStart[v + 1]; k++)
    row.add(otherEnd(l, l.vertEdges[k], v), w);
  for(int k = l.vertFaceStart[v]; k < l.vertFaceStart[v + 1]; k++)
    addFaceCenter(l, l.vertFaces[k], w, row);
  row.add(v, double(n - 2) / n);
}

void edgeRow(const Level& l, int e, Row& row)
{
  const int a = l.edgeVerts[2 * e], b = l.edgeVerts[2 * e + 1];
  if(l.edgeSharp[e])
  {
    row.add(a, 0.5);
    row.add(b, 0.5);
    return;
  }
  row.add(a, 0.25);
  row.add(b, 0.25);
  for(int k = l.edgeFaceStart[e]; k < l.edgeFaceStart[e + 1]; k++)
    addFaceCenter(l, l.edgeFaces[k], 0.25, row);
}

// ------------------------------------------------------------------------------------------------
// face-varying values (UVs)

using Uv = std::array<double, 2>;

// Ordered (value -> uv) pairs: the order values are met decides the child values' numbering.
struct ValueList
{
  std::vector<std::pair<int, Uv>> items;
  bool has(int value) const
  {
    for(const auto& [v, uv] : items)
      if(v == value)
        return true;
    return false;
  }
  void set(int value, Uv uv)
  {
    for(auto& [v, x] : items)
      if(v == value)
      {
        x = uv;
        return;
      }
    items.emplace_back(value, uv);
  }
};

// The child values of a vertex on a UV seam, per parent value. Each value's faces form a span around
// the vertex: one face is a UV corner and stays put, a span split again by a seam or a sharp edge is
// sharp too, and any other span refines as a crease between the values at its two ends.
ValueList seamVertexValues(const Level& p, int v, const std::vector<uint8_t>& seam)
{
  const auto& fv = p.faceValues;
  const auto& pv = p.values;
  ValueList   out;
  const int   s = p.vertFaceStart[v], n = p.vertFaceStart[v + 1] - s;
  auto        valueIn = [&](int f, int corner) { return fv[p.faceStart[f] + corner]; };
  auto        keep    = [&](int value) { out.set(value, {double(pv[2 * value]), double(pv[2 * value + 1])}); };
  std::optional<Ring> ring;
  if(!p.vertNonManifold[v])
    ring = ringOf(p, v);
  if(!ring || p.vertSharp[v])
  {
    for(int k = s; k < s + n; k++)
      if(!out.has(valueIn(p.vertFaces[k], p.vertFaceCorner[k])))
        keep(valueIn(p.vertFaces[k], p.vertFaceCorner[k]));
    return out;
  }
  const auto& faces    = ring->faces;
  const auto& corners  = ring->corners;
  const bool  boundary = ring->boundary;
  const int   m        = int(faces.size());
  auto        value    = [&](int i) { return valueIn(faces[i], corners[i]); };
  auto        size     = [&](int f) { return p.faceStart[f + 1] - p.faceStart[f]; };
  auto        leading  = [&](int i) { return p.faceEdges[p.faceStart[faces[i]] + corners[i]]; };
  auto        trailing = [&](int i) { return p.faceEdges[p.faceStart[faces[i]] + (corners[i] ? corners[i] - 1 : size(faces[i]) - 1)]; };
  auto        nextValue = [&](int i) { return valueIn(faces[i], (corners[i] + 1) % size(faces[i])); };
  auto        prevValue = [&](int i) { return valueIn(faces[i], corners[i] ? corners[i] - 1 : size(faces[i]) - 1); };
  auto        crease    = [&](int val, int first, int last) {
    const int a = nextValue(first), b = prevValue(last);
    out.set(val, {0.75 * pv[2 * val] + 0.125 * (double(pv[2 * a]) + pv[2 * b]),
                  0.75 * pv[2 * val + 1] + 0.125 * (double(pv[2 * a + 1]) + pv[2 * b + 1])});
  };

  // every face has the same value: a seam ends here (or a one-face corner); the span runs from a seam edge round to it
  bool single = true;
  for(int i = 1; i < m; i++)
    if(value(i) != value(0))
      single = false;
  if(single && !boundary)
  {
    int  start = -1, sharp = 0;
    bool discts = false;
    for(int i = 0; i < m; i++)
    {
      const int e = leading(i);
      if(seam[e])
      {
        if(start >= 0)
          discts = true;
        else
          start = i;
      }
      else if(p.edgeSharp[e])
        sharp++;
    }
    if(discts || sharp || start < 0 || m == 1)
      keep(value(0));
    else
      crease(value(0), start, (start + m - 1) % m);
    return out;
  }

  // runs of equal values around the ring (cyclic for an interior vertex), split where the value changes
  int rot = 0;
  if(!boundary)
    while(rot < m && value(rot) == value((rot + m - 1) % m))
      rot++;
  struct Run
  {
    int value, first, last, size, count, inner;
  };
  std::vector<Run> runs;  // in the order met, as a Map iterates
  int              i = 0;
  while(i < m)
  {
    const int first = (rot + i) % m;
    const int val   = value(first);
    int       last = first, len = 1, inner = 0;
    while(i + len < m && value((rot + i + len) % m) == val)
    {
      // the edge between two faces of the span: a seam or a sharp edge inside it makes it sharp
      const int e = trailing(last);
      if(seam[e] || p.edgeSharp[e])
        inner++;
      last = (rot + i + len) % m;
      len++;
    }
    auto it = std::find_if(runs.begin(), runs.end(), [&](const Run& r) { return r.value == val; });
    if(it != runs.end())
    {
      it->count++;
      it->size += len;
    }
    else
      runs.push_back({val, first, last, len, 1, inner});
    i += len;
  }
  for(const Run& r : runs)
  {
    if(r.count > 1 || r.inner || r.size == 1)
      keep(r.value);
    else
      crease(r.value, r.first, r.last);
  }
  return out;
}

// Refines the parent's UVs onto the child, as OpenSubdiv does with FVAR_LINEAR_CORNERS_ONLY. Where
// the UVs are continuous they follow the vertex rules. Along a UV seam each side refines as a
// crease of its own, and a UV corner (a value on one face only) stays put.
void refineValues(const Level& p, Level& c)
{
  const auto& fv = p.faceValues;
  const auto& pv = p.values;
  const int   nv = p.nv, nf = p.nf, ne = p.ne;

  // the value at each end of an edge, as one of its faces has them; an edge whose faces disagree is a seam
  int  ends[2] = {0, 0};
  auto edgeEndValues = [&](int e, int k) {
    const int f = p.edgeFaces[k], fs = p.faceStart[f], size = p.faceStart[f + 1] - fs;
    int       j = 0;
    while(j < size && p.faceEdges[fs + j] != e)
      j++;
    const int  a       = fv[fs + j], b = fv[fs + ((j + 1) % size)];
    const bool forward = p.faceVerts[fs + j] == p.edgeVerts[2 * e];
    ends[0]            = forward ? a : b;
    ends[1]            = forward ? b : a;
  };
  std::vector<uint8_t> seam(ne, 0), mismatch(nv, 0);
  for(int e = 0; e < ne; e++)
  {
    const int s = p.edgeFaceStart[e], n = p.edgeFaceStart[e + 1] - s;
    if(n < 2)
      continue;
    edgeEndValues(e, s);
    const int a0 = ends[0], b0 = ends[1];
    for(int k = s + 1; k < s + n; k++)
    {
      edgeEndValues(e, k);
      if(ends[0] != a0 || ends[1] != b0)
        seam[e] = 1;
    }
    if(seam[e])
      mismatch[p.edgeVerts[2 * e]] = mismatch[p.edgeVerts[2 * e + 1]] = 1;
  }
  // with smooth geometric corners, OpenSubdiv sharpens a UV corner of a one-face boundary vertex
  for(int v = 0; v < nv; v++)
  {
    const int faces = p.vertFaceStart[v + 1] - p.vertFaceStart[v];
    const int edges = p.vertEdgeStart[v + 1] - p.vertEdgeStart[v];
    if(!mismatch[v] && faces == 1 && edges > faces && !p.vertSharp[v])
      mismatch[v] = 1;
  }

  // the child values, two each
  std::vector<double> out;
  out.reserve(size_t(nv + nf + ne) * 2);
  auto push = [&](double u, double v) {
    out.push_back(u);
    out.push_back(v);
    return int(out.size() / 2) - 1;
  };
  // every face's centre in UV space (kept as float, as the TypeScript's Float32Array does)
  std::vector<float> centers(size_t(nf) * 2);
  for(int f = 0; f < nf; f++)
  {
    const int fs = p.faceStart[f], size = p.faceStart[f + 1] - fs;
    double    u = 0, w = 0;
    for(int c2 = fs; c2 < fs + size; c2++)
    {
      u += pv[2 * fv[c2]];
      w += pv[2 * fv[c2] + 1];
    }
    centers[2 * f]     = float(u / size);
    centers[2 * f + 1] = float(w / size);
  }

  // children of vertices: one value per parent value at the vertex; most vertices have one
  std::vector<int32_t>                  vertSingle(nv, -1);
  std::unordered_map<uint64_t, int32_t> vertValue;  // (seam vertex, parent value) -> child value
  auto keyOf = [](int a, int b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); };
  // the value at the far end of an edge from v, along the edge
  auto farValue = [&](int v, int e) {
    edgeEndValues(e, p.edgeFaceStart[e]);
    return p.edgeVerts[2 * e] == v ? ends[1] : ends[0];
  };
  for(int v = 0; v < nv; v++)
  {
    const int s = p.vertFaceStart[v], n = p.vertFaceStart[v + 1] - s;
    if(!n)
      continue;
    if(!mismatch[v])
    {
      const int value = fv[p.faceStart[p.vertFaces[s]] + p.vertFaceCorner[s]];
      // the vertex's own rule on the UVs: a corner stays, a crease follows its two sharp edges
      int sharpEdges = 0;
      for(int k = p.vertEdgeStart[v]; k < p.vertEdgeStart[v + 1]; k++)
        if(p.edgeSharp[p.vertEdges[k]])
          sharpEdges++;
      double u = 0, w = 0;
      if(p.vertSharp[v] || sharpEdges > 2)
      {
        u += pv[2 * value];
        w += pv[2 * value + 1];
      }
      else if(sharpEdges == 2)
      {
        for(int k = p.vertEdgeStart[v]; k < p.vertEdgeStart[v + 1]; k++)
        {
          if(!p.edgeSharp[p.vertEdges[k]])
            continue;
          const int x = farValue(v, p.vertEdges[k]);
          u += pv[2 * x] * 0.125;
          w += pv[2 * x + 1] * 0.125;
        }
        u += pv[2 * value] * 0.75;
        w += pv[2 * value + 1] * 0.75;
      }
      else
      {
        const double k2 = 1.0 / (double(n) * n);
        for(int k = s; k < s + n; k++)
        {
          u += centers[2 * p.vertFaces[k]] * k2;
          w += centers[2 * p.vertFaces[k] + 1] * k2;
        }
        for(int k = p.vertEdgeStart[v]; k < p.vertEdgeStart[v + 1]; k++)
        {
          const int x = farValue(v, p.vertEdges[k]);
          u += pv[2 * x] * k2;
          w += pv[2 * x + 1] * k2;
        }
        u += pv[2 * value] * (double(n - 2) / n);
        w += pv[2 * value + 1] * (double(n - 2) / n);
      }
      vertSingle[v] = push(u, w);
      continue;
    }
    for(const auto& [value, uv] : seamVertexValues(p, v, seam).items)
      vertValue[keyOf(v, value)] = push(uv[0], uv[1]);
  }
  // children of faces
  std::vector<int32_t> faceValue(nf);
  for(int f = 0; f < nf; f++)
    faceValue[f] = push(centers[2 * f], centers[2 * f + 1]);
  // children of edges: one value, or one per side of a seam
  std::unordered_map<uint64_t, int32_t> edgeValue;  // (edge, face) -> child value, for seams
  std::vector<int32_t>                  edgeSingle(ne, -1);
  for(int e = 0; e < ne; e++)
  {
    const int s = p.edgeFaceStart[e], n = p.edgeFaceStart[e + 1] - s;
    if(!n)
      continue;
    if(seam[e])
    {
      for(int k = s; k < s + n; k++)
      {
        edgeEndValues(e, k);
        const int a = ends[0], b = ends[1];
        edgeValue[keyOf(e, p.edgeFaces[k])] = push((double(pv[2 * a]) + pv[2 * b]) / 2, (double(pv[2 * a + 1]) + pv[2 * b + 1]) / 2);
      }
      continue;
    }
    edgeEndValues(e, s);
    const int a = ends[0], b = ends[1];
    double    u = 0, w = 0;
    if(p.edgeSharp[e])
    {
      u = (double(pv[2 * a]) + pv[2 * b]) / 2;
      w = (double(pv[2 * a + 1]) + pv[2 * b + 1]) / 2;
    }
    else
    {
      u = (double(pv[2 * a]) + pv[2 * b]) / 4;
      w = (double(pv[2 * a + 1]) + pv[2 * b + 1]) / 4;
      for(int k = s; k < s + n; k++)
      {
        const int fv2 = faceValue[p.edgeFaces[k]];
        u += out[2 * fv2] / 4;
        w += out[2 * fv2 + 1] / 4;
      }
    }
    edgeSingle[e] = push(u, w);
  }

  // the child faces' corners
  c.faceValues.assign(c.faceVerts.size(), 0);
  auto valueOfEdge = [&](int e, int f) { return seam[e] ? edgeValue.at(keyOf(e, f)) : edgeSingle[e]; };
  for(int f = 0; f < nf; f++)
  {
    const int fs = p.faceStart[f], size = p.faceStart[f + 1] - fs;
    for(int j = 0; j < size; j++)
    {
      const int cf     = fs + j;
      const int jPrev  = j ? j - 1 : size - 1;
      const int corner = p.faceVerts[fs + j];
      const int rot    = size == 4 ? j : 0;
      c.faceValues[cf * 4 + (rot & 3)] = vertSingle[corner] >= 0 ? vertSingle[corner] : vertValue.at(keyOf(corner, fv[fs + j]));
      c.faceValues[cf * 4 + ((rot + 1) & 3)] = valueOfEdge(p.faceEdges[fs + j], f);
      c.faceValues[cf * 4 + ((rot + 2) & 3)] = faceValue[f];
      c.faceValues[cf * 4 + ((rot + 3) & 3)] = valueOfEdge(p.faceEdges[fs + jPrev], f);
    }
  }
  c.values.resize(out.size());
  for(size_t k = 0; k < out.size(); k++)
    c.values[k] = float(out[k]);
  c.hasValues = true;
}

// ------------------------------------------------------------------------------------------------
// one refinement step

Level refineLevel(const Level& p, Stencils& stencils)
{
  const int nv = p.nv, nf = p.nf, ne = p.ne;
  const int corners = int(p.faceVerts.size());
  auto      vChild  = [](int v) { return v; };
  auto      fChild  = [nv](int f) { return nv + f; };
  auto      eChild  = [nv, nf](int e) { return nv + nf + e; };
  const int cnv     = nv + nf + ne;

  // stencils, rows in child vertex order
  stencils.rowStart.assign(cnv + 1, 0);
  stencils.col.clear();
  stencils.weight.clear();
  stencils.col.reserve(size_t(cnv) * 8);
  stencils.weight.reserve(size_t(cnv) * 8);
  stencils.parentCount = nv;
  Row  row(nv);
  auto emit = [&](int r) {
    for(size_t i = 0; i < row.cols.size(); i++)
    {
      stencils.col.push_back(row.cols[i]);
      stencils.weight.push_back(float(row.weights[i]));
    }
    stencils.rowStart[r + 1] = int32_t(stencils.col.size());
    row.clear();
  };
  for(int v = 0; v < nv; v++)
  {
    vertexRow(p, v, row);
    emit(vChild(v));
  }
  for(int f = 0; f < nf; f++)
  {
    addFaceCenter(p, f, 1, row);
    emit(fChild(f));
  }
  for(int e = 0; e < ne; e++)
  {
    edgeRow(p, e, row);
    emit(eChild(e));
  }

  Level c;
  c.nv = cnv;
  c.nf = corners;
  c.ne = corners + 2 * ne;
  // child edges: one per parent face corner (face centre to edge midpoint), then two per parent edge
  c.edgeVerts.assign(size_t(c.ne) * 2, 0);
  c.edgeSharp.assign(c.ne, 0);
  c.edgeNonManifold.assign(c.ne, 0);
  auto faceChildEdge = [&](int f, int j) { return p.faceStart[f] + j; };
  auto edgeChildEdge = [corners](int e, int j) { return corners + 2 * e + j; };
  for(int f = 0; f < nf; f++)
  {
    for(int cc = p.faceStart[f], j = 0; cc < p.faceStart[f + 1]; cc++, j++)
    {
      const int ce            = faceChildEdge(f, j);
      c.edgeVerts[2 * ce]     = fChild(f);
      c.edgeVerts[2 * ce + 1] = eChild(p.faceEdges[cc]);
    }
  }
  for(int e = 0; e < ne; e++)
  {
    for(int j = 0; j < 2; j++)
    {
      const int ce             = edgeChildEdge(e, j);
      c.edgeVerts[2 * ce]      = eChild(e);
      c.edgeVerts[2 * ce + 1]  = vChild(p.edgeVerts[2 * e + j]);
      c.edgeSharp[ce]          = p.edgeSharp[e];
      c.edgeNonManifold[ce]    = p.edgeNonManifold[e];
    }
  }

  // child faces: one quad per parent face corner, oriented as OpenSubdiv orients them
  c.faceStart.resize(size_t(c.nf) + 1);
  for(int i = 0; i <= c.nf; i++)
    c.faceStart[i] = i * 4;
  c.faceVerts.assign(size_t(c.nf) * 4, 0);
  c.faceEdges.assign(size_t(c.nf) * 4, 0);
  for(int f = 0; f < nf; f++)
  {
    const int fs = p.faceStart[f], size = p.faceStart[f + 1] - fs;
    for(int j = 0; j < size; j++)
    {
      const int cf           = fs + j;
      const int jPrev        = j ? j - 1 : size - 1;
      const int corner       = p.faceVerts[fs + j];
      const int ePrev        = p.faceEdges[fs + jPrev], eNext = p.faceEdges[fs + j];
      const int p0           = p.edgeVerts[2 * ePrev], p1 = p.edgeVerts[2 * ePrev + 1];
      const int n0           = p.edgeVerts[2 * eNext], n1 = p.edgeVerts[2 * eNext + 1];
      const int cornerInPrev = p0 != p1 ? (p0 != corner ? 1 : 0) : 1;
      const int cornerInNext = n0 != n1 ? (n0 != corner ? 1 : 0) : 0;
      // a quad's children are rotated so child j starts at corner j; other faces' all start at their corner
      const int rot = size == 4 ? j : 0;
      const int o0 = cf * 4 + (rot & 3), o1 = cf * 4 + ((rot + 1) & 3), o2 = cf * 4 + ((rot + 2) & 3), o3 = cf * 4 + ((rot + 3) & 3);
      c.faceVerts[o0] = vChild(corner);
      c.faceVerts[o1] = eChild(eNext);
      c.faceVerts[o2] = fChild(f);
      c.faceVerts[o3] = eChild(ePrev);
      c.faceEdges[o0] = edgeChildEdge(eNext, cornerInNext);
      c.faceEdges[o1] = faceChildEdge(f, j);
      c.faceEdges[o2] = faceChildEdge(f, jPrev);
      c.faceEdges[o3] = edgeChildEdge(ePrev, cornerInPrev);
    }
  }

  c.vertSharp.assign(cnv, 0);
  c.vertNonManifold.assign(cnv, 0);
  std::copy(p.vertSharp.begin(), p.vertSharp.end(), c.vertSharp.begin());
  std::copy(p.vertNonManifold.begin(), p.vertNonManifold.end(), c.vertNonManifold.begin());
  for(int e = 0; e < ne; e++)
    c.vertNonManifold[eChild(e)] = p.edgeNonManifold[e];

  relate(c);
  if(p.hasValues)
    refineValues(p, c);
  return c;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// public

// morph loads run on pool threads and share one refinement: the first to need the table builds it,
// and the others wait for it rather than build a second one over it
static std::mutex g_transposeMutex;

const Stencils::Transpose& Stencils::transpose() const
{
  std::lock_guard lock(g_transposeMutex);
  if(m_transpose)
    return *m_transpose;
  auto t = std::make_shared<Transpose>();
  t->start.assign(size_t(parentCount) + 1, 0);
  for(int32_t c : col)
    t->start[c + 1]++;
  for(int i = 0; i < parentCount; i++)
    t->start[i + 1] += t->start[i];
  std::vector<int32_t> fill(t->start.begin(), t->start.end() - 1);
  t->row.assign(col.size(), 0);
  const int n = rows();
  for(int r = 0; r < n; r++)
    for(int k = rowStart[r]; k < rowStart[r + 1]; k++)
      t->row[fill[col[k]]++] = r;
  m_transpose = std::move(t);
  return *m_transpose;
}

Refinement refine(const CageMesh& cage, int levels, BoundaryInterpolation boundary)
{
  Level                l = baseLevel(cage, boundary);
  Refinement           out;
  std::vector<int32_t> parent(l.nf);
  for(int f = 0; f < l.nf; f++)
    parent[f] = f;
  for(int k = 0; k < levels; k++)
  {
    Stencils s;
    Level    child = refineLevel(l, s);
    out.levels.push_back(std::move(s));
    // child face (parent face, corner j) is numbered by the parent's face-corner position
    std::vector<int32_t> next(child.nf);
    for(int f = 0; f < l.nf; f++)
      for(int c = l.faceStart[f]; c < l.faceStart[f + 1]; c++)
        next[c] = parent[f];
    parent = std::move(next);
    l      = std::move(child);
  }
  out.vertexCount = l.nv;
  out.quadParent  = std::move(parent);
  if(levels)
  {
    out.quads = std::move(l.faceVerts);
    if(l.hasValues)
    {
      out.quadValues = std::move(l.faceValues);
      out.values     = std::move(l.values);
    }
  }
  return out;
}

std::vector<float> refineDense(const Refinement& r, std::span<const float> data, int width)
{
  std::vector<float> src(data.begin(), data.end());
  for(const Stencils& s : r.levels)
  {
    const int          n = s.rows();
    std::vector<float> dst(size_t(n) * width, 0.f);
    // rows are independent and each sums in its own order, so chunks give the serial result
    jobs::parallelFor(size_t(n), 8192, [&](size_t begin, size_t end) {
      for(size_t row = begin; row < end; row++)
      {
        for(int k = s.rowStart[row]; k < s.rowStart[row + 1]; k++)
        {
          const double w = s.weight[k];
          const size_t c = size_t(s.col[k]) * width;
          for(int x = 0; x < width; x++)
          {
            float& d = dst[row * width + x];
            d        = float(double(d) + double(src[c + x]) * w);
          }
        }
      }
    });
    src = std::move(dst);
  }
  return src;
}

namespace {

// The positions of the set flags, in order: counted, then listed, a chunk at a time side by side.
std::vector<uint32_t> setPositions(const std::vector<uint8_t>& flags)
{
  constexpr size_t    kChunk = 65536;
  const size_t        chunks = (flags.size() + kChunk - 1) / kChunk;
  std::vector<size_t> at(chunks + 1, 0);
  jobs::parallelFor(chunks, 1, [&](size_t begin, size_t end) {
    for(size_t c = begin; c < end; c++)
      at[c + 1] = size_t(std::count(flags.begin() + c * kChunk, flags.begin() + std::min(flags.size(), (c + 1) * kChunk), uint8_t(1)));
  });
  std::partial_sum(at.begin(), at.end(), at.begin());
  std::vector<uint32_t> out(at[chunks]);
  jobs::parallelFor(chunks, 1, [&](size_t begin, size_t end) {
    for(size_t c = begin; c < end; c++)
    {
      size_t o = at[c];
      for(size_t i = c * kChunk; i < std::min(flags.size(), (c + 1) * kChunk); i++)
        if(flags[i])
          out[o++] = uint32_t(i);
    }
  });
  return out;
}

}  // namespace

Sparse refineSparse(const Refinement& r, std::span<const uint32_t> indices, std::span<const float> data, int width)
{
  // the first level reads the caller's lists where they are, each further one the level before's
  std::vector<uint32_t>     idxStore;
  std::vector<float>        valsStore;
  std::span<const uint32_t> idx  = indices;
  std::span<const float>    vals = data;
  for(const Stencils& s : r.levels)
  {
    const auto& t    = s.transpose();
    const int   rows = s.rows();
    // a parent's value is only read where `has` lists it, so the rest is never written (or touched)
    std::unique_ptr<float[]> parent(new float[size_t(s.parentCount) * width]);
    std::vector<uint8_t>     has(s.parentCount, 0), touched(rows, 0);
    for(size_t i = 0; i < idx.size(); i++)
    {
      has[idx[i]] = 1;
      for(int x = 0; x < width; x++)
        parent[size_t(idx[i]) * width + x] = vals[i * width + x];
    }
    // the child rows that read a listed vertex, in order: a long list (a correction on a whole
    // garment) marks them side by side and reads them off in order, a short one sorts what it found
    std::vector<uint32_t> found;
    if(idx.size() >= 4096 && idx.size() * 32 >= size_t(rows))
    {
      jobs::parallelFor(idx.size(), 4096, [&](size_t begin, size_t end) {
        for(size_t i = begin; i < end; i++)
          for(int k = t.start[idx[i]]; k < t.start[idx[i] + 1]; k++)
            std::atomic_ref<uint8_t>(touched[size_t(t.row[k])]).store(1, std::memory_order_relaxed);
      });
      found = setPositions(touched);
    }
    else
    {
      for(uint32_t p : idx)
        for(int k = t.start[p]; k < t.start[p + 1]; k++)
        {
          const int row = t.row[k];
          if(!touched[row])
          {
            touched[row] = 1;
            found.push_back(uint32_t(row));
          }
        }
      std::sort(found.begin(), found.end());
    }
    // each row sums in its own order, so chunks side by side give the serial result
    std::vector<float> out(found.size() * width, 0.f);
    jobs::parallelFor(found.size(), 4096, [&](size_t begin, size_t end) {
      for(size_t i = begin; i < end; i++)
      {
        const uint32_t row = found[i];
        for(int k = s.rowStart[row]; k < s.rowStart[row + 1]; k++)
        {
          const int c = s.col[k];
          if(!has[c])
            continue;
          const double w = s.weight[k];
          for(int x = 0; x < width; x++)
          {
            float& d = out[i * width + x];
            d        = float(double(d) + double(parent[size_t(c) * width + x]) * w);
          }
        }
      }
    });
    idxStore  = std::move(found);
    valsStore = std::move(out);
    idx       = idxStore;
    vals      = valsStore;
  }
  if(r.levels.empty())
    return {std::vector<uint32_t>(indices.begin(), indices.end()), std::vector<float>(data.begin(), data.end())};
  return {std::move(idxStore), std::move(valsStore)};
}

}  // namespace fv::subdiv
