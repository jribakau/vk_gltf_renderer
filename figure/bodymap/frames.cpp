#include <figure/bodymap/frames.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <optional>

namespace fv::bodymap {

namespace {

using R = rig::Region;

Vec3 cross(const Vec3& a, const Vec3& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3 vertexAt(const std::vector<float>& p, size_t v)
{
  return {p[v * 3], p[v * 3 + 1], p[v * 3 + 2]};
}

// a middle weighted by area, so a finer mesh somewhere (a face) doesn't pull it there
struct Mean
{
  Vec3   sum;
  double weight = 0;
  void   add(const Vec3& p, double w) { sum = sum + p * w, weight += w; }
  std::optional<Vec3> get() const { return weight > 0 ? std::optional<Vec3>(sum * (1.0 / weight)) : std::nullopt; }
};

// Each region's limb axis on the body (unit), by its bit's index. A limb part's runs joint to joint:
// from the middle of its band with the region above to that of its band with the region below (the
// upper arm's from the shoulder's band to the elbow's), as those are where each rig's weights hand
// over, which a region's own shape is not (one rig's collar reaches up the neck, another's out over
// the shoulder). A part at a limb's end (a finger, a toe, the head) runs from its band above to the
// middle of its own skin, or takes the region above's when it has hardly any skin of its own (a small
// toe all border).
std::array<Vec3, 64> regionAxes(const Body& body)
{
  // every region on each of its sides, parents first (the spec's order), with the bits of the regions
  // above and below
  struct Slot
  {
    R          region;
    RegionMask bit, above, below;
    Mean       own, holding, band, next;
  };
  std::vector<Slot> slots;
  for(int r = int(R::ePelvis); r <= int(R::ePinkyToe); r++)
  {
    const R    region = R(r), up = regionAbove(region);
    const bool sided  = regionSided(region);
    for(const char side : {'l', 'r'})
    {
      if(!sided && side == 'r')
        continue;
      const char s = sided ? side : 0;
      RegionMask below = 0;
      for(const R d : regionsBelow(region))
        below |= regionBit(d, regionSided(d) ? s : 0);
      slots.push_back({region, regionBit(region, s), up == R::eNone ? 0 : regionBit(up, regionSided(up) ? s : 0), below, {}, {}, {}, {}});
    }
  }
  const std::vector<double> area = vertexAreas(body);
  const size_t              nv   = std::min(body.regions.size(), area.size());
  for(size_t v = 0; v < nv; v++)
  {
    const RegionMask m = body.regions[v];
    if(!m || area[v] <= 0)
      continue;
    const Vec3 p = vertexAt(body.positions, v);
    for(Slot& s : slots)
    {
      if(!(m & s.bit))
        continue;
      s.holding.add(p, area[v]);
      if(m == s.bit)
        s.own.add(p, area[v]);
      if(m & s.above)
        s.band.add(p, area[v]);
      if(m & s.below)
        s.next.add(p, area[v]);
    }
  }
  std::array<Vec3, 64> axes{};
  for(const Slot& s : slots)
  {
    const auto own  = s.own.weight > 0.1 * s.holding.weight ? s.own.get() : std::nullopt;
    const auto band = s.band.get(), next = s.next.get();
    Vec3       axis;
    if(s.region == R::eBreast)
      axis = axes[size_t(std::countr_zero(regionBit(R::eChest)))];
    else if(band && next)
      axis = *next - *band;
    else if(band && own)
      axis = *own - *band;
    else if(!s.above && own)
      // the root: up to its band with the region below it that has no side (the pelvis's with the abdomen)
      for(const Slot& c : slots)
        if(c.above == s.bit && !regionSided(c.region))
          if(const auto up = c.band.get())
          {
            axis = *up - *own;
            break;
          }
    if(length(axis) < 1e-6)
      axis = s.above ? axes[size_t(std::countr_zero(s.above))] : Vec3{0, 1, 0};
    axes[size_t(std::countr_zero(s.bit))] = normalize(axis);
  }
  return axes;
}

}  // namespace

Frames vertexFrames(const Body& body)
{
  const size_t nv = body.positions.size() / 3;
  Frames       out;
  out.normal = outerNormals(body);
  out.tangent.assign(nv, Vec3{});
  const std::array<Vec3, 64> axes = regionAxes(body);
  for(size_t v = 0; v < nv && v < body.regions.size(); v++)
  {
    const Vec3& n = out.normal[v];
    if(dot(n, n) == 0)
      continue;
    Vec3 axis;
    int  count = 0;
    for(RegionMask m = body.regions[v]; m; m &= m - 1)
    {
      axis = axis + axes[size_t(std::countr_zero(m))];
      count++;
    }
    if(!count)
      continue;
    axis           = axis * (1.0 / count);
    out.tangent[v] = axis - n * dot(axis, n);
  }
  return out;
}

std::vector<Quat> frameTurns(const Frames& source, const Frames& onto, const Projection& p)
{
  const size_t      n = source.normal.size();
  std::vector<Quat> out(n);
  for(size_t v = 0; v < n && v < p.mapped.size(); v++)
  {
    const Vec3& ns = source.normal[v];
    if(!p.mapped[v] || dot(ns, ns) == 0)
      continue;
    Vec3 ng, tg;
    for(size_t j = 0; j < 3; j++)
    {
      const uint32_t c = p.at.verts[v * 3 + j];
      const double   w = p.at.weights[v * 3 + j];
      ng               = ng + onto.normal[c] * w;
      tg               = tg + onto.tangent[c] * w;
    }
    if(length(ng) < 1e-9)
      continue;
    ng               = normalize(ng);
    tg               = tg - ng * dot(tg, ng);
    const Quat bend  = quatFromUnitVectors(ns, ng);
    const Vec3& ts   = source.tangent[v];
    // how much of the twist about the normal to take: none where either tangent is barely set
    double share = std::clamp((std::min(length(ts), length(tg)) - 0.1) / 0.3, 0.0, 1.0);
    share        = share * share * (3 - 2 * share);
    if(share <= 0)
    {
      out[v] = bend;
      continue;
    }
    const Vec3   from  = rotate(normalize(ts), bend), to = normalize(tg);
    const double angle = share * std::atan2(dot(cross(from, to), ng), dot(from, to));
    const double s     = std::sin(angle / 2);
    out[v]             = (Quat{ng.x * s, ng.y * s, ng.z * s, std::cos(angle / 2)} * bend).normalized();
  }
  return out;
}

}  // namespace fv::bodymap
