#include <figure/wardrobe/fit/transfer.hpp>

#include <algorithm>
#include <cmath>
#include <set>

#include <figure/wardrobe/fit/bones.hpp>

namespace fv::wardrobe {

namespace {

V3     sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3     cross(const V3& a, const V3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
double len3(const V3& a) { return std::hypot(a[0], a[1], a[2]); }
V3     unit(const V3& a)
{
  const double l = len3(a);
  return l < 1e-9 ? V3{0, 0, 0} : V3{a[0] / l, a[1] / l, a[2] / l};
}

// The joint down the limb that sets a bone's direction, by slot, nearest first.
const std::map<std::string, std::vector<std::string>>& nextSlots()
{
  static const std::map<std::string, std::vector<std::string>> table{
      {"thigh", {"shin"}},       {"shin", {"foot"}},          {"foot", {"toes"}},           {"collar", {"upperarm"}},
      {"upperarm", {"forearm"}}, {"forearm", {"hand"}},       {"hand", {"mid1", "midmc"}},  {"spine1", {"spine2", "spine3"}},
      {"spine2", {"spine3"}},    {"spine3", {"spine4", "neck1"}}, {"spine4", {"neck1"}},    {"neck1", {"neck2", "head"}},
      {"neck2", {"head"}},
  };
  return table;
}

// Orthonormal frame with z along `dir`, rolled so x is perpendicular to `ref`. Columns x, y, z.
std::optional<std::array<V3, 3>> frame(const V3& dir, const V3& ref)
{
  const V3 z = unit(dir);
  const V3 x = unit(cross(ref, z));
  if(!x[0] && !x[1] && !x[2])
    return std::nullopt;
  return std::array<V3, 3>{x, cross(z, x), z};
}

// Frame `a` onto frame `b`, stretched by `stretch` along the bone (z): B diag(1, 1, s) A^T, row-major.
std::array<double, 9> between(const std::array<V3, 3>& a, const std::array<V3, 3>& b, double stretch)
{
  std::array<double, 9> r{};
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      r[size_t(i * 3 + j)] = b[0][i] * a[0][j] + b[1][i] * a[1][j] + stretch * b[2][i] * a[2][j];
  return r;
}

// Limb length ratios outside this are a wrong match, not a longer leg.
constexpr double kStretchLo = 0.8, kStretchHi = 1.25;

}  // namespace

const SkelBone* findSlot(const std::vector<SkelBone>& bones, char side, const std::string& slot)
{
  const SkelBone* twist = nullptr;
  for(const SkelBone& b : bones)
  {
    const ParsedBone p = parseBone(b.name);
    if(p.side != side || p.slot != slot)
      continue;
    if(!p.twist)
      return &b;
    if(!twist)
      twist = &b;
  }
  return twist;
}

std::map<std::string, Rigid> rigidPlan(const std::vector<SkelBone>& source, const std::vector<SkelBone>& target)
{
  std::vector<std::string> targetNames;
  for(const SkelBone& b : target)
    targetNames.push_back(b.name);
  std::map<std::string, const SkelBone*> targetBy, sourceBy;
  for(const SkelBone& b : target)
    targetBy[b.name] = &b;
  for(const SkelBone& b : source)
    sourceBy[b.name] = &b;
  std::map<std::string, Rigid>                          out;
  std::map<std::string, std::optional<std::string>>     leaderOf;
  const auto lead = [&](const std::string& name) -> std::optional<std::string> {
    auto it = leaderOf.find(name);
    if(it == leaderOf.end())
      it = leaderOf.emplace(name, matchBone(name, targetNames)).first;
    return it->second;
  };
  const Rigid still{};

  std::function<Rigid(const SkelBone&, int)> visit = [&](const SkelBone& b, int depth) -> Rigid {
    if(auto it = out.find(b.name); it != out.end())
      return it->second;
    const SkelBone* parent = nullptr;
    if(b.parent)
      if(auto p = sourceBy.find(*b.parent); p != sourceBy.end())
        parent = p->second;
    const Rigid                up     = parent && depth < 256 ? visit(*parent, depth + 1) : still;
    const auto                 leader = lead(b.name);
    std::optional<std::string> ancestorLeader;
    {
      const SkelBone* p = parent;
      for(int guard = 0; p && guard < 256; guard++)
      {
        ancestorLeader = lead(p->name);
        if(ancestorLeader)
          break;
        const SkelBone* next = nullptr;
        if(p->parent)
          if(auto it = sourceBy.find(*p->parent); it != sourceBy.end())
            next = it->second;
        p = next;
      }
    }
    const SkelBone* t = nullptr;
    if(leader)
      if(auto it = targetBy.find(*leader); it != targetBy.end())
        t = it->second;
    // A pectoral moves with the chest. Its joint isn't a landmark two generations share: Genesis 8's is
    // deep in the chest and Genesis 9's near its middle, 8 cm apart, and carrying a bust from one to the
    // other left domes in front of the body that the shape correction, capped, could not bring back.
    // The correction fits the bust; the pose still turns it about the figure's own joint.
    if(!t || leader == ancestorLeader || parseBone(b.name).slot == "pectoral")
    {
      out[b.name] = up;
      return up;
    }
    auto             r      = up.r;
    const ParsedBone parsed = parseBone(b.name);
    if(auto next = nextSlots().find(parsed.slot); next != nextSlots().end())
      for(const std::string& slot : next->second)
      {
        const SkelBone* sc = findSlot(source, parsed.side, slot);
        const SkelBone* tc = findSlot(target, parseBone(t->name).side, slot);
        if(!sc || !tc)
          continue;
        const V3 ds = sub(sc->center, b.center);
        const V3 dt = sub(tc->center, t->center);
        // the reference is picked once, from the source, so both frames roll the same way
        const bool vertical = std::abs(unit(ds)[1]) > 0.7;
        const V3   ref      = vertical ? V3{0, 0, 1} : V3{0, 1, 0};
        const auto fs       = frame(ds, ref);
        const auto ft       = frame(dt, ref);
        const double ratio  = len3(dt) / std::max(len3(ds), 1e-6);
        if(fs && ft)
          r = between(*fs, *ft, std::min(kStretchHi, std::max(kStretchLo, ratio)));
        break;
      }
    Rigid rigid;
    rigid.r    = r;
    rigid.from = b.center;
    rigid.to   = t->center;
    out[b.name] = rigid;
    return rigid;
  };
  for(const SkelBone& b : source)
    visit(b, 0);
  return out;
}

std::vector<float> rigidPositions(std::span<const float> positions, const mesh::SkinInput& skin, const std::vector<const Rigid*>& bones)
{
  const size_t       n = positions.size() / 3;
  std::vector<float> out(positions.size());
  for(size_t v = 0; v < n; v++)
  {
    const double px = positions[v * 3], py = positions[v * 3 + 1], pz = positions[v * 3 + 2];
    double       x = 0, y = 0, z = 0, wsum = 0;
    if(v + 1 < skin.start.size())
      for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
      {
        const double w = skin.weight[size_t(i)];
        const size_t b = size_t(skin.bone[size_t(i)]);
        const Rigid* g = b < bones.size() ? bones[b] : nullptr;
        if(w <= 0 || !g)
          continue;
        const double dx = px - g->from[0], dy = py - g->from[1], dz = pz - g->from[2];
        const auto&  r = g->r;
        x += w * (r[0] * dx + r[1] * dy + r[2] * dz + g->to[0]);
        y += w * (r[3] * dx + r[4] * dy + r[5] * dz + g->to[1]);
        z += w * (r[6] * dx + r[7] * dy + r[8] * dz + g->to[2]);
        wsum += w;
      }
    if(wsum <= 0)
    {
      out[v * 3] = float(px);
      out[v * 3 + 1] = float(py);
      out[v * 3 + 2] = float(pz);
    }
    else
    {
      out[v * 3] = float(x / wsum);
      out[v * 3 + 1] = float(y / wsum);
      out[v * 3 + 2] = float(z / wsum);
    }
  }
  return out;
}

namespace {

// Vertex `v`'s bound point on a per-vertex vec3 field of the surface, or none when unbound.
std::optional<V3> boundPoint(std::span<const float> field, const SurfaceBinding& b, size_t v)
{
  const size_t k  = v * 3;
  const double wa = b.weights[k], wb = b.weights[k + 1], wc = b.weights[k + 2];
  if(wa + wb + wc <= 0)
    return std::nullopt;
  const size_t a = b.verts[k] * 3ull, bb = b.verts[k + 1] * 3ull, c = b.verts[k + 2] * 3ull;
  return V3{field[a] * wa + field[bb] * wb + field[c] * wc, field[a + 1] * wa + field[bb + 1] * wb + field[c + 1] * wc,
            field[a + 2] * wa + field[bb + 2] * wb + field[c + 2] * wc};
}

}  // namespace

std::vector<float> laplacian(std::vector<float> field, std::span<const uint32_t> tris, size_t n, int passes)
{
  if(passes <= 0)
    return field;
  std::vector<int32_t> deg(n + 1, 0);
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
    for(int i = 0; i < 3; i++)
    {
      const uint32_t a = tris[t + size_t(i)];
      if(a < n && tris[t + size_t((i + 1) % 3)] < n)
        deg[a + 1] += 2;
    }
  for(size_t v = 0; v < n; v++)
    deg[v + 1] += deg[v];
  std::vector<int32_t> nbr(static_cast<size_t>(deg[n]));
  std::vector<int32_t> fill(deg.begin(), deg.begin() + std::ptrdiff_t(n));
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
    for(int i = 0; i < 3; i++)
    {
      const uint32_t a = tris[t + size_t(i)], b = tris[t + size_t((i + 1) % 3)];
      if(a >= n || b >= n)
        continue;
      nbr[size_t(fill[a]++)] = int32_t(b);
      nbr[size_t(fill[b]++)] = int32_t(a);
    }
  std::vector<float> cur = std::move(field);
  for(int pass = 0; pass < passes; pass++)
  {
    std::vector<float> next(cur.size());
    for(size_t v = 0; v < n; v++)
    {
      const int32_t s = deg[v], e = deg[v + 1];
      const size_t  k = v * 3;
      if(s == e)
      {
        next[k] = cur[k];
        next[k + 1] = cur[k + 1];
        next[k + 2] = cur[k + 2];
        continue;
      }
      double x = 0, y = 0, z = 0;
      for(int32_t i = s; i < e; i++)
      {
        const size_t u = size_t(nbr[size_t(i)]) * 3;
        x += cur[u];
        y += cur[u + 1];
        z += cur[u + 2];
      }
      const double inv = 0.5 / (e - s);
      next[k]     = float(cur[k] * 0.5 + x * inv);
      next[k + 1] = float(cur[k + 1] * 0.5 + y * inv);
      next[k + 2] = float(cur[k + 2] * 0.5 + z * inv);
    }
    cur = std::move(next);
  }
  return cur;
}

std::vector<float> shapeCorrection(std::span<const float> item, std::span<const uint32_t> itemTris, const Surface& base, const Surface& target,
                                   const ShapeInputs& in)
{
  const ShapeOptions& o  = in.shape;
  const size_t        nb = base.positions.size() / 3;
  // the target point of every base vertex, as a displacement
  std::vector<float> baseDelta(nb * 3, 0.f);
  if(in.sameTopology && target.positions.size() == base.positions.size())
  {
    for(size_t i = 0; i < baseDelta.size(); i++)
      baseDelta[i] = target.positions[i] - base.positions[i];
  }
  else
  {
    SurfaceBinding local;
    const SurfaceBinding* toTarget = in.baseToTarget;
    if(!toTarget)
    {
      local    = bindToSurface(target.positions, target.tris, base.positions, {}, in.baseFilter);
      toTarget = &local;
    }
    for(size_t v = 0; v < nb; v++)
    {
      const auto p = boundPoint(target.positions, *toTarget, v);
      if(!p)
        continue;
      baseDelta[v * 3]     = float((*p)[0] - base.positions[v * 3]);
      baseDelta[v * 3 + 1] = float((*p)[1] - base.positions[v * 3 + 1]);
      baseDelta[v * 3 + 2] = float((*p)[2] - base.positions[v * 3 + 2]);
    }
  }

  const size_t       n = item.size() / 3;
  std::vector<float> delta(item.size(), 0.f);
  const auto         onBase = bindToSurface(base.positions, base.tris, item, {}, in.itemFilter);
  for(size_t v = 0; v < n; v++)
  {
    const size_t k = v * 3;
    const auto   s = boundPoint(base.positions, onBase, v);
    if(!s)
      continue;
    const double dist = std::hypot(item[k] - (*s)[0], item[k + 1] - (*s)[1], item[k + 2] - (*s)[2]);
    const double w    = dist <= o.near ? 1 : dist >= o.far ? 0 : (o.far - dist) / (o.far - o.near);
    if(w <= 0)
      continue;
    const V3 d = *boundPoint(baseDelta, onBase, v);
    delta[k]     = float(d[0] * w);
    delta[k + 1] = float(d[1] * w);
    delta[k + 2] = float(d[2] * w);
  }
  const std::vector<float> smoothed = laplacian(std::move(delta), itemTris, n, o.smooth);
  std::vector<float>       out(item.size());
  for(size_t v = 0; v < n; v++)
  {
    const size_t k  = v * 3;
    double       dx = smoothed[k], dy = smoothed[k + 1], dz = smoothed[k + 2];
    const double len = std::hypot(dx, dy, dz);
    if(len > o.cap)
    {
      const double f = o.cap / len;
      dx *= f;
      dy *= f;
      dz *= f;
    }
    out[k]     = float(item[k] + dx);
    out[k + 1] = float(item[k + 1] + dy);
    out[k + 2] = float(item[k + 2] + dz);
  }
  return out;
}

namespace {

// The 2nd and 98th percentile of `values`.
std::pair<double, double> spread(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const auto at = [&](double q) { return values[size_t(std::min<double>(double(values.size() - 1), std::max(0.0, std::round(q * double(values.size() - 1)))))]; };
  return {at(0.02), at(0.98)};
}

}  // namespace

std::optional<LimbBox> limbBox(const V3& frameDir, const V3& origin, std::span<const float> source, const std::function<bool(size_t)>& sourceOn,
                               std::span<const float> target, const std::function<bool(size_t)>& targetOn)
{
  const auto axes = frame(frameDir, {0, 1, 0});
  if(!axes)
    return std::nullopt;
  const auto project = [&](std::span<const float> pos, const std::function<bool(size_t)>& on) {
    std::array<std::vector<double>, 3> cols;
    for(size_t v = 0; v < pos.size() / 3; v++)
    {
      if(!on(v))
        continue;
      const V3 d{pos[v * 3] - origin[0], pos[v * 3 + 1] - origin[1], pos[v * 3 + 2] - origin[2]};
      for(size_t a = 0; a < 3; a++)
        cols[a].push_back(d[0] * (*axes)[a][0] + d[1] * (*axes)[a][1] + d[2] * (*axes)[a][2]);
    }
    return cols;
  };
  const auto s = project(source, sourceOn);
  const auto t = project(target, targetOn);
  if(s[0].size() < 20 || t[0].size() < 20)
    return std::nullopt;
  LimbBox box;
  box.origin = origin;
  box.axes   = *axes;
  for(size_t a = 0; a < 3; a++)
  {
    const auto [slo, shi] = spread(s[a]);
    const auto [tlo, thi] = spread(t[a]);
    box.range[a]          = {slo, shi, tlo, thi};
  }
  return box;
}

std::vector<float> applyLimbBox(const LimbBox& box, std::span<const float> positions, const std::function<double(size_t)>& weight)
{
  std::vector<float> out(positions.begin(), positions.end());
  for(size_t v = 0; v < positions.size() / 3; v++)
  {
    const double w = weight(v);
    if(w <= 0)
      continue;
    const V3 d{positions[v * 3] - box.origin[0], positions[v * 3 + 1] - box.origin[1], positions[v * 3 + 2] - box.origin[2]};
    for(size_t a = 0; a < 3; a++)
    {
      const auto& [slo, shi, tlo, thi] = box.range[a];
      const double u     = d[0] * box.axes[a][0] + d[1] * box.axes[a][1] + d[2] * box.axes[a][2];
      const double scale = std::min(1.4, std::max(0.7, (thi - tlo) / std::max(shi - slo, 1e-6)));
      const double mid   = (slo + shi) / 2;
      const double moved = (tlo + thi) / 2 + (u - mid) * scale;
      for(size_t c = 0; c < 3; c++)
        out[v * 3 + c] = float(out[v * 3 + c] + w * (moved - u) * box.axes[a][c]);
    }
  }
  return out;
}

}  // namespace fv::wardrobe
