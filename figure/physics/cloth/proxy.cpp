#include <figure/physics/cloth/proxy.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <figure/physics/cloth/ride.hpp>

namespace fv::physics::cloth {

namespace {

// How far round the body a drape is looked for over each of its vertices, to tell what lies under it.
constexpr float kUnderReach = 3.f;

// Each of `points` (xyz) bound to the body at rest within `reach` cm: its face and its signed distance (the
// face's winding outward), or none.
struct OnBody
{
  std::vector<int64_t> face;
  std::vector<float>   distance;
};
OnBody onBody(const Surface& body, const std::vector<float>& points, float reach)
{
  const size_t n = points.size() / 3;
  OnBody       out{std::vector<int64_t>(n, -1), std::vector<float>(n, INFINITY)};
  const Ride   r = bindRide(body, body.rest, points, std::vector<uint8_t>(n, 0), nullptr, 0.f, reach);
  for(size_t i = 0; i < r.vertices.size(); i++)
  {
    const float* l = &r.local[i * 3];
    const float  d = std::sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
    out.face[r.vertices[i]]     = int64_t(r.face[i]);
    out.distance[r.vertices[i]] = l[2] >= 0 ? d : -d;
  }
  return out;
}

}  // namespace

Proxied proxyLive(std::vector<Surface>& surfaces, size_t bodySlot, const std::vector<size_t>& drapes)
{
  Proxied out;
  if(bodySlot >= surfaces.size() || surfaces[bodySlot].tris.empty())
    return out;
  const bool floored = std::any_of(surfaces.begin(), surfaces.end(), [](const Surface& s) { return !s.capsules.empty(); });
  if(!floored)
    return out;
  Surface&     body = surfaces[bodySlot];
  const size_t nb   = body.vertexCount();
  const auto   isDrape = [&](size_t s) { return std::find(drapes.begin(), drapes.end(), s) != drapes.end(); };

  // Each drape's vertices on the skin are held to it, fading to free; and how near the drapes come to each
  // body vertex, so a worn item can tell whether it lies under them
  std::vector<float> drapeAt(nb, INFINITY);
  for(size_t slot : drapes)
  {
    if(slot >= surfaces.size() || slot == bodySlot)
      continue;
    Surface&     cloth = surfaces[slot];
    const size_t n     = cloth.vertexCount();
    const OnBody on    = onBody(body, cloth.rest, kUnderReach);
    for(size_t v = 0; v < n; v++)
    {
      if(on.face[v] < 0)
        continue;
      const float d = on.distance[v];
      for(size_t k = 0; k < 3; k++)
      {
        float& at = drapeAt[body.tris[size_t(on.face[v]) * 3 + k]];
        at        = std::min(at, d);
      }
      if(v >= cloth.strength.size() || cloth.strength[v] <= 1e-4f)
        continue;
      const float free = std::clamp((d - kOnSkin) / kOnSkinFade, 0.f, 1.f);
      if(free >= 1)
        continue;
      cloth.strength[v] *= free;
      (free <= 0 ? out.held : out.eased)++;
    }
  }

  // What is worn on the skin under the cloth is part of it: out of the simulation, and the body thicker there
  for(size_t slot = 0; slot < surfaces.size(); slot++)
  {
    Surface& worn = surfaces[slot];
    if(slot == bodySlot || isDrape(slot) || worn.kind != Surface::Kind::eCollider || worn.tris.empty() || !worn.capsules.empty())
      continue;
    const size_t n  = worn.vertexCount();
    const OnBody on = onBody(body, worn.rest, kOnSkin);
    if(worn.collide.size() != n)
      worn.collide.assign(n, 1);
    for(size_t v = 0; v < n; v++)
    {
      if(on.face[v] < 0 || on.distance[v] > kOnSkin)
        continue;
      const uint32_t* c = &body.tris[size_t(on.face[v]) * 3];
      // over the cloth there (a belt on a dress): the cloth's to meet as it is
      if(on.distance[v] > std::min({drapeAt[c[0]], drapeAt[c[1]], drapeAt[c[2]]}))
        continue;
      worn.collide[v] = 0;
      out.folded++;
      const float gap = v < worn.offset.size() ? worn.offset[v] : 0.2f;
      for(size_t k = 0; k < 3; k++)
        body.offset[c[k]] = std::max(body.offset[c[k]], std::max(0.f, on.distance[v]) + gap);
    }
  }

  // and the body's faces leave the simulation: its floor meets the cloth, and what is drawn is held out of them
  body.collide.assign(nb, 0);
  out.on = true;
  return out;
}

}  // namespace fv::physics::cloth
