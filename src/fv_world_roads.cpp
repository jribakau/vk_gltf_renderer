#include "fv_world_roads.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

#include "fv_world_places.hpp"

namespace fv::world {

namespace {

uint32_t mix(uint32_t v)
{
  v = v * 747796405U + 2891336453U;
  const uint32_t w = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (w >> 22U) ^ w;
}

uint32_t hash(uint32_t a, uint32_t b, uint32_t c)
{
  return mix(mix(mix(a) + b) + c);
}

double unit(uint32_t h)
{
  return double(h >> 8U) * (1.0 / 16777216.0);
}

template <class T>
T bent(const T& a, const T& b, const T& c, double t)
{
  const double s = 1.0 - t;
  return a * (s * s) + b * (2.0 * s * t) + c * (t * t);
}

// ---- the shapes ---------------------------------------------------------------------------------------------

// Paint's thickness over the surface; the columns across a surface; how far past its edge it goes
// down into the ground, and how deep
constexpr double kPaintOver  = 0.015;
constexpr int    kColumns    = 5;
constexpr double kSkirtOut   = 0.35;
constexpr double kSkirtUnder = 0.08;
// The markings: a dash down the middle every so many metres, so long, and the share of them worn
// away; the edge lines in lengths of so many metres, the share of those gone; their widths, and how
// far in from the edge the edge lines run
constexpr double kDashEvery = 9.0, kDashLong = 3.0, kDashGone = 0.2, kDashWide = 0.12;
constexpr double kEdgeBlock = 5.0, kEdgeGone = 0.15, kEdgeWide = 0.15, kEdgeIn = 0.35;
// A bridge: its parapets' thickness and height over the road, how far its deck reaches past the
// road's edge, how deep it is (more for a long one), the length between piers and their thickness,
// an abutment's
constexpr double kParapetThick = 0.25, kParapetHigh = 0.9, kDeckOut = 0.55, kDeckDeep = 0.8;
constexpr double kPierEvery = 15.0, kPierThick = 1.2, kAbutmentThick = 1.4;
// Metres between a bridge's sections along it
constexpr double kDeckStep = 2.0;

// The quad's ground as its mesh draws it (groundMesh): its points sampled as the mesh samples them,
// each once, and between them the mesh's own triangles
class Drawn
{
public:
  Drawn(const World& w, const Quad& q)
      : w_(w), q_(q), cell_(width(w, q) / double(kGroundCells)), heights_(size_t(kGroundCells + 1) * size_t(kGroundCells + 1), kUnset)
  {
  }
  double cell() const { return cell_; }

  // The ground's elevation under `d` as drawn (on the quad, or by it: its edge's then)
  double at(const glm::dvec3& d)
  {
    constexpr int   N = int(kGroundCells);
    const FacePoint f = onFace(q_.face, d);
    const double    x = std::clamp(gridOf(f.u, q_.x), 0.0, double(N)), y = std::clamp(gridOf(f.v, q_.y), 0.0, double(N));
    const int       i = std::min(int(x), N - 1), j = std::min(int(y), N - 1);
    const double    fx = x - double(i), fy = y - double(j);
    const double    a = height(i, j), b = height(i + 1, j), c = height(i, j + 1), e = height(i + 1, j + 1);
    // the diagonal the mesh takes: the shorter (addGrid)
    const glm::dvec3 pa = point(i, j, a), pb = point(i + 1, j, b), pc = point(i, j + 1, c), pe = point(i + 1, j + 1, e);
    if(glm::dot(pa - pe, pa - pe) <= glm::dot(pb - pc, pb - pc))
      return fx >= fy ? a + fx * (b - a) + fy * (e - b) : a + fy * (c - a) + fx * (e - c);
    return fx + fy <= 1.0 ? a + fx * (b - a) + fy * (c - a) : e + (1.0 - fx) * (c - e) + (1.0 - fy) * (b - e);
  }

private:
  static constexpr double kUnset = 1.0e300;
  double                  scale() const { return double(kGroundCells) * double(uint64_t(1) << q_.level); }
  double                  gridOf(double u, uint32_t tile) const { return (u + 1.0) * 0.5 * scale() - double(tile) * double(kGroundCells); }
  double coord(uint32_t tile, int i) const { return (double(tile) * double(kGroundCells) + double(i)) * (2.0 / scale()) - 1.0; }
  glm::dvec3 dir(int i, int j) const { return direction(q_.face, coord(q_.x, i), coord(q_.y, j)); }
  double     height(int i, int j)
  {
    double& h = heights_[size_t(j) * size_t(kGroundCells + 1) + size_t(i)];
    if(h == kUnset)
      h = sample(w_, dir(i, j), cell_).elevation;
    return h;
  }
  glm::dvec3 point(int i, int j, double h) const { return toScene(w_, dir(i, j), h); }

  const World&        w_;
  const Quad&         q_;
  double              cell_;
  std::vector<double> heights_;
};

// A quad's face with its normal facing `outward` whichever order its corners come in, and its
// coordinates (two triangles; normals flat)
void face(RoadMesh& m, const std::array<glm::vec3, 4>& p, const std::array<glm::vec2, 4>& uv, const glm::dvec3& outward)
{
  glm::vec3 n = glm::cross(p[1] - p[0], p[2] - p[0]) + glm::cross(p[2] - p[0], p[3] - p[0]);
  const bool flip = glm::dot(glm::dvec3(n), outward) < 0.0;
  if(glm::dot(n, n) < 1e-20F)
    n = glm::vec3(outward);
  n                    = glm::normalize(flip ? -n : n);
  const uint32_t first = uint32_t(m.positions.size());
  for(int k = 0; k < 4; k++)
  {
    m.positions.push_back(p[size_t(k)]);
    m.normals.push_back(n);
    m.uvs.push_back(uv[size_t(k)]);
  }
  if(flip)
    m.indices.insert(m.indices.end(), {first, first + 2, first + 1, first, first + 3, first + 2});
  else
    m.indices.insert(m.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
}

// ---- a road's surface ---------------------------------------------------------------------------------------

// A section's points across a road: a skirt, the carriageway's columns, a skirt
constexpr int kAcross = kColumns + 2;

struct Section
{
  glm::dvec3                      centre{0.0}, side{0.0};  // the middle line's point; the unit way across it
  double                          along = 0.0;             // metres along its way
  std::array<glm::dvec3, kAcross> dir{};                   // each point's
  std::array<double, kAcross>     x{}, h{}, at{};          // metres across, its elevation, metres along the way
  bool                            dusty = false;           // at an open end: soil washed onto it
};

// A stretch of the asphalt's wear (between two of kAsphaltJoints) laid on a way: which, from where to
// where along the way, and whether it runs backward along it or is turned across it
struct Laid
{
  size_t k    = 0;
  double from = 0.0, to = 0.0;
  bool   back = false, flip = false;

  glm::vec2 uv(double x, double half, double along) const
  {
    const double    d  = along - from;
    const glm::vec2 on = roadUv(x, half, back ? kAsphaltJoints[k + 1] - d : kAsphaltJoints[k] + d);
    return glm::vec2(flip ? 1.0F - on.x : on.x, on.y);
  }
};

struct Surfaces
{
  const World&      w;
  const Places&     pl;
  const glm::dvec3& centre;
  Drawn&            drawn;
  RoadMeshes&       out;
  double            R, lift;
  uint32_t          seed;

  glm::vec3 at(const glm::dvec3& d, double elevation) const { return glm::vec3(toScene(w, d, elevation) - centre); }

  // The asphalt's tint at `d`: weathered wider than its maps' repeat, warmer or cooler here and there
  glm::vec4 tint(const glm::dvec3& d) const
  {
    const double grey = 1.0 + 0.3 * (noise(w, d, 35.0, 70U) - 0.5) + 0.16 * (noise(w, d, 9.0, 71U) - 0.5);
    const double warm = 0.06 * (noise(w, d, 60.0, 72U) - 0.5);
    return glm::vec4(float(grey * (1.0 + warm)), float(grey), float(grey * (1.0 - warm)), 1.0F);
  }

  // The stretches of the wear laid along a way that reach from `from` to `to` metres along it, in
  // order: from the way's start (part of the way into the first), each another than the one before,
  // each either way along and either way across
  std::vector<Laid> laidAlong(uint32_t way, double from, double to) const
  {
    constexpr uint32_t N      = uint32_t(kAsphaltJoints.size()) - 1;
    const auto         length = [](uint32_t k) { return kAsphaltJoints[k + 1] - kAsphaltJoints[k]; };
    std::vector<Laid>  laid;
    uint32_t           k   = hash(seed, way, 0xB10CU) % N;
    double             end = -length(k) * unit(hash(seed, way, 0x5EA7U));
    for(uint32_t m = 0;; m++)
    {
      if(m > 0)
        k = (k + 1U + hash(seed + 1U, way, m) % (N - 1U)) % N;
      const uint32_t turn = hash(seed + 2U, way, m);
      const Laid     l{k, end, end + length(k), (turn & 1U) != 0, (turn & 2U) != 0};
      if(l.to > from)
        laid.push_back(l);
      end = l.to;
      if(end >= to)
        return laid;
    }
  }

  // The way along a piece at `t`, along the sphere at `on`
  static glm::dvec3 runOf(const RoadPiece& p, double t, const glm::dvec3& on)
  {
    glm::dvec3 run = (p.b - p.a) * (1.0 - t) + (p.c - p.b) * t;
    run -= on * glm::dot(run, on);
    if(glm::dot(run, run) < 1e-30)
      run = p.c - p.a;
    return glm::normalize(run);
  }

  Section section(const RoadPiece& p, double t) const
  {
    Section s;
    s.centre           = glm::normalize(bent(p.a, p.b, p.c, t));
    s.side             = glm::normalize(glm::cross(runOf(p, t, s.centre), s.centre));
    s.along            = double(p.start) + t * double(p.length);
    const double level = bent(double(p.level[0]), double(p.level[1]), double(p.level[2]), t);
    // on a bridge and on to its ends the road is its own level (over the water the ground is the
    // river's bed); elsewhere it lies on the ground as drawn, which is the road's where it is fine
    const bool   deck = bridgeAt(pl, p.way, s.along, kBridgeEnds) != nullptr;
    const double half = double(p.half);
    for(int c = 0; c < kAcross; c++)
    {
      const size_t i = size_t(c);
      s.x[i]         = c == 0 ? -(half + kSkirtOut) : c == kAcross - 1 ? half + kSkirtOut : -half + 2.0 * half * double(c - 1) / double(kColumns - 1);
      s.dir[i]       = glm::normalize(s.centre + s.side * (s.x[i] / R));
      s.at[i]        = s.along;
    }
    for(int c = 1; c <= kColumns; c++)
    {
      const double g   = drawn.at(s.dir[size_t(c)]);
      s.h[size_t(c)] = (deck ? std::max(level, g) : g) + lift;
    }
    for(const int c : {0, kAcross - 1})
    {
      const double edge = s.h[c == 0 ? 1U : size_t(kColumns)];
      // (down into the ground, a lip on a bridge, and never a wall: a culvert's stream is under it)
      s.h[size_t(c)] = deck ? edge - 0.25 : std::clamp(drawn.at(s.dir[size_t(c)]) - kSkirtUnder, edge - 1.5, edge - 0.05);
    }
    return s;
  }

  // A section at an open end of its way, `toward` it +1 (its end) or -1 (its start): the asphalt
  // broken off there, each point a little short of the end, with soil washed onto it
  Section broken(const RoadPiece& p, double t, double toward) const
  {
    Section s = section(p, t);
    std::array<double, kAcross> back{};
    for(int c = 1; c <= kColumns; c++)
      back[size_t(c)] = 0.6 * unit(hash(seed, p.way, uint32_t(c) + (toward > 0.0 ? 16U : 0U)));
    back[0]                   = back[1];
    back[size_t(kAcross - 1)] = back[size_t(kColumns)];
    for(int c = 0; c < kAcross; c++)
    {
      const Section o = section(p, std::clamp(t - toward * back[size_t(c)] / double(p.length), 0.0, 1.0));
      s.dir[size_t(c)] = o.dir[size_t(c)];
      s.h[size_t(c)]   = o.h[size_t(c)];
      s.at[size_t(c)]  = o.at[size_t(c)];
    }
    s.dusty = true;
    return s;
  }

  // Past an open end's section, its edge going down into the ground as its sides do
  Section endSkirt(const Section& s, const RoadPiece& p, double t, double toward) const
  {
    Section          e   = s;
    const glm::dvec3 out = runOf(p, t, s.centre) * toward;
    for(size_t c = 0; c < size_t(kAcross); c++)
    {
      e.dir[c] = glm::normalize(s.dir[c] + out * (kSkirtOut / R));
      e.at[c]  = s.at[c] + toward * kSkirtOut;
      e.h[c]   = std::clamp(drawn.at(e.dir[c]) - kSkirtUnder, s.h[c] - 1.5, s.h[c] - 0.05);
    }
    e.dusty = false;
    return e;
  }

  // The surface's height `x` metres across a section
  static double across(const Section& s, double half, double x)
  {
    const double c = std::clamp((x + half) / (2.0 * half) * double(kColumns - 1), 0.0, double(kColumns - 1));
    const int    i = std::min(int(c), kColumns - 2);
    return s.h[size_t(i) + 1] + (c - double(i)) * (s.h[size_t(i) + 2] - s.h[size_t(i) + 1]);
  }

  // A section's vertices on a stretch of the wear (`down`: all of them a skirt, darker); the first's index
  uint32_t vertices(const Section& s, const Laid& l, double half, bool down)
  {
    RoadMesh&      m     = out.surface;
    const uint32_t first = uint32_t(m.positions.size());
    for(size_t c = 0; c < size_t(kAcross); c++)
    {
      m.positions.push_back(at(s.dir[c], s.h[c]));
      m.normals.emplace_back(0.0F);
      m.uvs.push_back(l.uv(s.x[c], half, s.at[c]));
      // (its skirts go down into the soil: darker; at an open end, soil on it)
      const bool      skirt = down || c == 0 || c == size_t(kAcross - 1);
      const glm::vec4 shade = skirt ? glm::vec4(0.8F, 0.8F, 0.8F, 1.0F) : s.dusty ? glm::vec4(1.25F, 1.15F, 1.0F, 1.0F) : glm::vec4(1.0F);
      m.colours.push_back(tint(s.dir[c]) * shade);
    }
    return first;
  }

  // The surface between two sections' vertices, as strips across: its normals are summed from its faces
  void between(uint32_t a, uint32_t b)
  {
    RoadMesh& m = out.surface;
    for(uint32_t k = 0; k + 1 < uint32_t(kAcross); k++)
    {
      const uint32_t A = a + k, A1 = A + 1, B = b + k, B1 = B + 1;
      for(const std::array<uint32_t, 3>& t : {std::array<uint32_t, 3>{A, A1, B1}, std::array<uint32_t, 3>{A, B1, B}})
      {
        const glm::vec3 n = glm::cross(m.positions[t[1]] - m.positions[t[0]], m.positions[t[2]] - m.positions[t[0]]);
        for(const uint32_t v : t)
        {
          m.normals[v] += n;
          m.indices.push_back(v);
        }
      }
    }
  }

  // A stripe of paint over the surface between two sections: from `f0` to `f1` of the way from one
  // to the other, from `x0` to `x1` across
  void paint(const Section& a, const Section& b, const Laid& l, double half, double f0, double f1, double x0, double x1)
  {
    const auto dir   = [&](double f, double x) { return glm::normalize(glm::mix(a.centre, b.centre, f) + glm::mix(a.side, b.side, f) * (x / R)); };
    const auto point = [&](double f, double x) { return at(dir(f, x), across(a, half, x) + f * (across(b, half, x) - across(a, half, x)) + kPaintOver); };
    const auto on    = [&](double f, double x) { return l.uv(x, half, a.along + f * (b.along - a.along)); };
    face(out.marking, {point(f0, x0), point(f0, x1), point(f1, x1), point(f1, x0)}, {on(f0, x0), on(f0, x1), on(f1, x1), on(f1, x0)},
         w.frame.turn * glm::mix(a.centre, b.centre, f0));
    for(const auto& [f, x] : {std::pair{f0, x0}, std::pair{f0, x1}, std::pair{f1, x1}, std::pair{f1, x0}})
      out.marking.colours.push_back(tint(dir(f, x)));
  }

  // The markings between two sections, from `lo` to `hi` metres along the way at most
  void markings(const Section& a, const Section& b, const RoadPiece& p, const Laid& l, double lo, double hi)
  {
    const double half = double(p.half), A = a.along, B = b.along, s0 = std::max(A, lo), s1 = std::min(B, hi);
    if(!(s1 > s0) || !(B > A))
      return;
    const auto f = [&](double s) { return (s - A) / (B - A); };
    // a dash down the middle
    for(double k = std::floor(s0 / kDashEvery); k * kDashEvery < s1; k += 1.0)
    {
      const double from = std::max(s0, k * kDashEvery), to = std::min(s1, k * kDashEvery + kDashLong);
      if(to > from && unit(hash(seed, p.way, uint32_t(int64_t(k)))) >= kDashGone)
        paint(a, b, l, half, f(from), f(to), -0.5 * kDashWide, 0.5 * kDashWide);
    }
    // and a line along each edge, broken
    for(int side = 0; side < 2; side++)
    {
      const double x = (side ? 1.0 : -1.0) * (half - kEdgeIn);
      for(double k = std::floor(s0 / kEdgeBlock); k * kEdgeBlock < s1; k += 1.0)
      {
        const double from = std::max(s0, k * kEdgeBlock), to = std::min(s1, (k + 1.0) * kEdgeBlock);
        if(to > from && unit(hash(seed + 1U + uint32_t(side), p.way, uint32_t(int64_t(k)))) >= kEdgeGone)
          paint(a, b, l, half, f(from), f(to), x - 0.5 * kEdgeWide, x + 0.5 * kEdgeWide);
      }
    }
  }

  // The stretches of a piece (pieces[index]) whose middles fall in the quad: from where its way's
  // start stops to where its end does (WayEnd), cut where one stretch of the wear laid on it gives way
  // to the next; broken off at an open end, its edge going down into the ground there
  void piece(const RoadPiece& p, size_t index, const Quad& q)
  {
    const double len = double(p.length);
    if(!(len > 0.0))
      return;
    const bool    ended = p.way < pl.wayEnds.size();
    const WayEnd  none{};
    const WayEnd& head  = ended && index == pl.wayPieces[p.way] ? pl.wayEnds[p.way][0] : none;
    const WayEnd& tail  = ended && index + 1 == pl.wayPieces[p.way + 1] ? pl.wayEnds[p.way][1] : none;
    const double  tFrom = std::clamp(double(head.cut) / len, 0.0, 1.0), tTo = 1.0 - std::clamp(double(tail.cut) / len, 0.0, 1.0);
    if(tTo - tFrom < 1e-6)
      return;
    const double half = double(p.half);
    const double step = std::clamp(drawn.cell(), 1.0, 4.0);
    const int    n    = std::max(1, int(std::ceil(len / step)));
    // (no paint on the broken end of the asphalt)
    const double            lo   = head.open ? double(p.start) + tFrom * len + 0.8 : -1e30;
    const double            hi   = tail.open ? double(p.start) + tTo * len - 0.8 : 1e30;
    const std::vector<Laid> laid = laidAlong(p.way, double(p.start) + tFrom * len, double(p.start) + tTo * len);
    // the sections made, by where along the piece; their vertices, by that and the stretch of the wear on them
    std::map<double, Section>                     sections;
    std::map<std::pair<double, size_t>, uint32_t> copies;
    const auto                                    sectionAt = [&](double t) -> const Section& {
      auto it = sections.find(t);
      if(it == sections.end())
        it = sections.emplace(t, head.open && t == tFrom ? broken(p, t, -1.0) : tail.open && t == tTo ? broken(p, t, 1.0) : section(p, t)).first;
      return it->second;
    };
    const auto copyOf = [&](double t, size_t i) {
      auto it = copies.find({t, i});
      if(it == copies.end())
        it = copies.emplace(std::make_pair(t, i), vertices(sectionAt(t), laid[i], half, false)).first;
      return it->second;
    };
    const auto laidAt = [&](double along) {
      size_t i = 0;
      while(i + 1 < laid.size() && laid[i].to <= along)
        i++;
      return i;
    };
    for(int k = 0; k < n; k++)
    {
      if(!contains(q, glm::normalize(bent(p.a, p.b, p.c, (double(k) + 0.5) / n))))
        continue;
      const double t0 = std::max(double(k) / n, tFrom), t1 = std::min(double(k + 1) / n, tTo);
      if(!(t1 > t0))
        continue;
      std::vector<double> cuts{t0};
      for(const Laid& l : laid)
        if(const double t = (l.from - double(p.start)) / len; t > t0 + 1e-9 && t < t1 - 1e-9)
          cuts.push_back(t);
      cuts.push_back(t1);
      for(size_t j = 0; j + 1 < cuts.size(); j++)
      {
        const size_t i = laidAt(double(p.start) + 0.5 * (cuts[j] + cuts[j + 1]) * len);
        between(copyOf(cuts[j], i), copyOf(cuts[j + 1], i));
        markings(sectionAt(cuts[j]), sectionAt(cuts[j + 1]), p, laid[i], lo, hi);
      }
      if(t0 == tFrom && head.open)
      {
        const size_t i = laidAt(double(p.start) + tFrom * len);
        between(vertices(endSkirt(sectionAt(tFrom), p, tFrom, -1.0), laid[i], half, true), copyOf(tFrom, i));
      }
      if(t1 == tTo && tail.open)
      {
        const size_t i = laidAt(double(p.start) + tTo * len);
        between(copyOf(tTo, i), vertices(endSkirt(sectionAt(tTo), p, tTo, 1.0), laid[i], half, true));
      }
    }
    // a point's normal from every face round it, whichever stretch of the wear they were laid on
    std::vector<glm::vec3>& normals = out.surface.normals;
    for(auto it = copies.begin(); it != copies.end();)
    {
      auto                            next = it;
      std::array<glm::vec3, kAcross> sum{};
      for(; next != copies.end() && next->first.first == it->first.first; ++next)
        for(size_t c = 0; c < size_t(kAcross); c++)
          sum[c] += normals[next->second + c];
      for(; it != next; ++it)
        for(size_t c = 0; c < size_t(kAcross); c++)
          normals[it->second + c] = sum[c];
    }
  }
};

// ---- a bridge ------------------------------------------------------------------------------------------------

struct Bridges
{
  const World&      w;
  const Places&     pl;
  const glm::dvec3& centre;
  const RoadTiles&  tiles;
  RoadMesh&         m;
  double            R;

  // A frame on the road at `along`: a point `a` metres on, `x` across and `y` over the road there
  struct Frame
  {
    OnWay      on;
    glm::dvec3 side{0.0};
  };
  Frame frameAt(uint32_t way, double along) const
  {
    Frame f;
    f.on   = onWay(pl, way, along);
    f.side = glm::normalize(glm::cross(f.on.forward, f.on.direction));
    return f;
  }
  glm::vec3 at(const Frame& f, double a, double x, double y) const
  {
    return glm::vec3(toScene(w, glm::normalize(f.on.direction + f.on.forward * (a / R) + f.side * (x / R)), f.on.level + y) - centre);
  }
  glm::vec2 uv(double u, double v) const { return glm::vec2(float(u / tiles.bridge), float(v / tiles.bridge)); }
  // a way of the planet's, in the scene
  glm::dvec3 vec(const glm::dvec3& v) const { return w.frame.turn * v; }

  // A box in a frame (its top left out: under the deck), from `y0` up to `y1`
  void box(const Frame& f, double a0, double a1, double x0, double x1, double y0, double y1)
  {
    const glm::dvec3 up = vec(f.on.direction), fw = vec(f.on.forward), sd = vec(f.side);
    face(m, {at(f, a0, x0, y0), at(f, a0, x1, y0), at(f, a0, x1, y1), at(f, a0, x0, y1)}, {uv(x0, y0), uv(x1, y0), uv(x1, y1), uv(x0, y1)}, -fw);
    face(m, {at(f, a1, x0, y0), at(f, a1, x1, y0), at(f, a1, x1, y1), at(f, a1, x0, y1)}, {uv(x0, y0), uv(x1, y0), uv(x1, y1), uv(x0, y1)}, fw);
    face(m, {at(f, a0, x0, y0), at(f, a1, x0, y0), at(f, a1, x0, y1), at(f, a0, x0, y1)}, {uv(a0, y0), uv(a1, y0), uv(a1, y1), uv(a0, y1)}, -sd);
    face(m, {at(f, a0, x1, y0), at(f, a1, x1, y0), at(f, a1, x1, y1), at(f, a0, x1, y1)}, {uv(a0, y0), uv(a1, y0), uv(a1, y1), uv(a0, y1)}, sd);
    face(m, {at(f, a0, x0, y0), at(f, a1, x0, y0), at(f, a1, x1, y0), at(f, a0, x1, y0)}, {uv(a0, x0), uv(a1, x0), uv(a1, x1), uv(a0, x1)}, -up);
  }

  // Where the ground is under a frame's points across it, the lowest, relative to the road
  double groundUnder(const Frame& f, double half) const
  {
    double lowest = 1.0e30;
    for(const double x : {-half, 0.0, half})
      lowest = std::min(lowest, sample(w, glm::normalize(f.on.direction + f.side * (x / R)), 1.0).elevation);
    return lowest - f.on.level;
  }

  void bridge(const Bridge& b)
  {
    const RoadPiece* piece = onWay(pl, b.way, 0.5 * (double(b.from) + double(b.to))).piece;
    if(!piece)
      return;
    const double half = double(piece->half), W = half + kDeckOut, inner = W - kParapetThick;
    const double span = double(b.to) - double(b.from);
    const double deep = kDeckDeep + std::min(0.01 * span, 0.6);
    const double s0 = std::max(double(b.from) - kBridgeEnds, 0.0), s1 = std::min(double(b.to) + kBridgeEnds, wayLength(pl, b.way));
    const int    n  = std::max(1, int(std::ceil((s1 - s0) / kDeckStep)));
    // the deck and its parapets, a stretch at a time between frames on the road
    Frame a = frameAt(b.way, s0);
    for(int k = 1; k <= n; k++)
    {
      const double sa = s0 + (s1 - s0) * double(k - 1) / n, sb = s0 + (s1 - s0) * double(k) / n;
      const Frame  c  = frameAt(b.way, sb);
      const auto   along = [&](double x0, double y0, double x1, double y1, const glm::dvec3& out, bool flat) {
        const std::array<glm::vec3, 4> p{at(a, 0.0, x0, y0), at(c, 0.0, x0, y0), at(c, 0.0, x1, y1), at(a, 0.0, x1, y1)};
        const double                   va = flat ? x0 : y0, vb = flat ? x1 : y1;
        face(m, p, {uv(sa, va), uv(sb, va), uv(sb, vb), uv(sa, vb)}, out);
      };
      const glm::dvec3 up = vec(a.on.direction), sd = vec(a.side);
      along(-inner, -0.02, inner, -0.02, up, true);                       // its top, under the road
      along(-W, -deep, W, -deep, -up, true);                              // its underside
      along(-W, -deep, -W, kParapetHigh, -sd, false);                     // its sides, up the parapets
      along(W, -deep, W, kParapetHigh, sd, false);
      along(-W, kParapetHigh, -inner, kParapetHigh, up, true);            // the parapets' tops
      along(inner, kParapetHigh, W, kParapetHigh, up, true);
      along(-inner, -0.02, -inner, kParapetHigh, sd, false);              // and their faces to the road
      along(inner, -0.02, inner, kParapetHigh, -sd, false);
      a = c;
    }
    // its ends, in the banks
    for(const double s : {s0, s1})
    {
      const Frame      f   = frameAt(b.way, s);
      const glm::dvec3 out = vec(s == s0 ? -f.on.forward : f.on.forward);
      for(const auto& [x0, x1, y0, y1] : {std::array<double, 4>{-W, W, -deep, -0.02}, std::array<double, 4>{-W, -inner, -0.02, kParapetHigh},
                                          std::array<double, 4>{inner, W, -0.02, kParapetHigh}})
        face(m, {at(f, 0.0, x0, y0), at(f, 0.0, x1, y0), at(f, 0.0, x1, y1), at(f, 0.0, x0, y1)}, {uv(x0, y0), uv(x1, y0), uv(x1, y1), uv(x0, y1)}, out);
    }
    // an abutment at each edge of the water, down into the bank; piers in the water between
    {
      const Frame from = frameAt(b.way, double(b.from)), to = frameAt(b.way, double(b.to));
      box(from, -kAbutmentThick, 0.0, -W, W, groundUnder(from, W) - 1.0, -deep);
      box(to, 0.0, kAbutmentThick, -W, W, groundUnder(to, W) - 1.0, -deep);
    }
    const int piers = span > 1.2 * kPierEvery ? int(span / kPierEvery) : 0;
    for(int k = 1; k <= piers; k++)
    {
      const Frame f = frameAt(b.way, double(b.from) + span * double(k) / double(piers + 1));
      box(f, -0.5 * kPierThick, 0.5 * kPierThick, -(half - 0.3), half - 0.3, groundUnder(f, half - 0.3) - 1.0, -deep);
    }
  }
};

}  // namespace

RoadMeshes roadMeshes(const World& w, const Quad& q, const glm::dvec3& centre, const RoadTiles& tiles)
{
  RoadMeshes out;
  if(!w.places || !w.places->catalogue)
    return out;
  const Places& pl   = *w.places;
  const double  R    = w.spec.radius;
  const double  wide = width(w, q);
  // the bridges whose middles are on it, at any level
  {
    Bridges bridges{w, pl, centre, tiles, out.bridge, R};
    for(const Bridge& b : pl.bridges)
      if(contains(q, onWay(pl, b.way, 0.5 * (double(b.from) + double(b.to))).direction))
        bridges.bridge(b);
  }
  // the roads' surfaces, where the ground is fine enough
  Drawn drawn(w, q);
  if(drawn.cell() > kSurfaceFootprint)
    return out;
  const double     span   = 2.0 / double(uint64_t(1) << q.level);
  const glm::dvec3 middle = direction(q.face, (double(q.x) + 0.5) * span - 1.0, (double(q.y) + 0.5) * span - 1.0);
  Surfaces         surfaces{w, pl, centre, drawn, out, R, surfaceLift(drawn.cell()), uint32_t(w.spec.seed) * 2246822519U + 13U};
  for(size_t i = 0; i < pl.roads.size(); i++)
  {
    const RoadPiece& p = pl.roads[i];
    if(pl.catalogue->roads[p.kind].earth)
      continue;  // (a track is the ground's earth)
    if(glm::length(p.b - middle) * R > 0.75 * wide + 0.6 * double(p.length) + double(p.half) + 5.0)
      continue;
    surfaces.piece(p, i, q);
  }
  RoadMesh& m = out.surface;
  for(glm::vec3& n : m.normals)
    n = glm::dot(n, n) > 1e-24F ? glm::normalize(n) : glm::vec3(0.0F, 1.0F, 0.0F);
  return out;
}

RoadMesh bridgeMesh(const World& w, const Bridge& b, const glm::dvec3& centre, const RoadTiles& tiles)
{
  RoadMesh out;
  if(!w.places)
    return out;
  Bridges bridges{w, *w.places, centre, tiles, out, w.spec.radius};
  bridges.bridge(b);
  return out;
}

glm::dvec3 bridgeMiddle(const World& w, const Bridge& b)
{
  if(!w.places)
    return glm::dvec3(0.0);
  const OnWay on = onWay(*w.places, b.way, 0.5 * (double(b.from) + double(b.to)));
  return toScene(w, on.direction, on.level);
}

}  // namespace fv::world
