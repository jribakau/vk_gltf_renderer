#include <figure/garment/decode.hpp>

#include <algorithm>
#include <cmath>

namespace fv::garment {

namespace {

Vec3 at(const std::vector<float>& p, size_t v)
{
  return {p[v * 3], p[v * 3 + 1], p[v * 3 + 2]};
}

}  // namespace

Decoded decode(const Addresses& garment, const bodymap::Body& reference)
{
  Target target;
  target.reference = &reference;
  return decode(garment, target);
}

Decoded decode(const Addresses& garment, const Target& target)
{
  const size_t n = garment.hold.size();
  Decoded      out;
  out.positions.assign(n * 3, 0.f);
  out.hold = garment.hold;
  if(!target.reference)
  {
    out.hold.assign(n, 0.f);
    return out;
  }
  const bodymap::Body& laid = *target.reference;
  const bodymap::Body& skin = target.body ? *target.body : laid;
  const size_t         nl = laid.positions.size() / 3, ns = skin.positions.size() / 3;
  out.at.verts.assign(n * 3, 0);
  out.at.weights.assign(n * 3, 0.f);
  out.at.used.assign(ns, 0);
  // each held address's place on the reference as it lies on the target
  std::vector<size_t> held;
  std::vector<float>  places;
  for(size_t v = 0; v < n; v++)
  {
    if(out.hold[v] <= 0)
      continue;
    Vec3   place;
    double sum = 0;
    for(size_t j = 0; j < 3; j++)
    {
      const size_t c = garment.at.verts[v * 3 + j];
      const double w = garment.at.weights[v * 3 + j];
      if(c >= nl || w == 0)
        continue;
      place = place + at(laid.positions, c) * w;
      sum += w;
    }
    if(sum <= 0)
    {
      out.hold[v] = 0;
      continue;
    }
    place = place * (1 / sum);
    places.insert(places.end(), {float(place.x), float(place.y), float(place.z)});
    held.push_back(v);
  }
  if(held.empty())
    return out;
  // on the target's own skin: the address itself on the reference's topology, else the skin's closest
  // point to that place (the laid reference lies within millimetres of it)
  wardrobe::SurfaceBinding onSkin;
  if(target.body)
  {
    std::vector<uint32_t> tris;
    for(size_t t = 0; t * 3 + 2 < skin.tris.size(); t++)
      if(t >= skin.outer.size() || skin.outer[t])
        tris.insert(tris.end(), skin.tris.begin() + ptrdiff_t(t * 3), skin.tris.begin() + ptrdiff_t(t * 3 + 3));
    if(tris.empty())
    {
      out.hold.assign(n, 0.f);
      return out;
    }
    onSkin = wardrobe::bindToSurface(skin.positions, tris, places);
  }
  const std::vector<Vec3>  normals = bodymap::outerNormals(skin);
  const std::vector<Quat>* turns   = target.frameTurn && !target.frameTurn->empty() ? target.frameTurn : nullptr;
  for(size_t i = 0; i < held.size(); i++)
  {
    const size_t v = held[i];
    Vec3         point, normal;
    Quat         turn{0, 0, 0, 0};
    double       sum = 0;
    for(size_t j = 0; j < 3; j++)
    {
      const size_t c = target.body ? onSkin.verts[i * 3 + j] : garment.at.verts[v * 3 + j];
      const double w = target.body ? onSkin.weights[i * 3 + j] : garment.at.weights[v * 3 + j];
      if(c >= ns)
        continue;
      out.at.verts[v * 3 + j] = uint32_t(c);
      if(w == 0)
        continue;
      out.at.weights[v * 3 + j] = float(w);
      out.at.used[c]            = 1;
      point                     = point + at(skin.positions, c) * w;
      sum += w;
      if(c < normals.size())
        normal = normal + normals[c] * w;
      if(turns && c < turns->size())
      {
        const Quat&  q    = (*turns)[c];
        const double sign = turn.x * q.x + turn.y * q.y + turn.z * q.z + turn.w * q.w < 0 ? -w : w;
        turn              = {turn.x + q.x * sign, turn.y + q.y * sign, turn.z + q.z * sign, turn.w + q.w * sign};
      }
    }
    if(sum <= 0)
    {
      out.hold[v] = 0;
      continue;
    }
    Vec3 across{garment.across[v * 3], garment.across[v * 3 + 1], garment.across[v * 3 + 2]};
    // kept in the reference base's frame: back out of it into the target's
    if(turns)
      across = rotate(across, turn.normalized().inverse());
    const Vec3 p             = point * (1 / sum) + normalize(normal) * double(garment.height[v]) + across;
    out.positions[v * 3]     = float(p.x);
    out.positions[v * 3 + 1] = float(p.y);
    out.positions[v * 3 + 2] = float(p.z);
  }
  return out;
}

wardrobe::SurfaceBinding withPlaces(const wardrobe::SurfaceBinding& bound, const wardrobe::SurfaceBinding& placed)
{
  wardrobe::SurfaceBinding out = bound;
  if(placed.verts.size() != out.verts.size() || placed.weights.size() != out.weights.size())
    return out;
  bool any = false;
  for(size_t i = 0; i + 2 < out.weights.size(); i += 3)
  {
    if(placed.weights[i] == 0 && placed.weights[i + 1] == 0 && placed.weights[i + 2] == 0)
      continue;
    for(size_t j = 0; j < 3; j++)
    {
      out.verts[i + j]   = placed.verts[i + j];
      out.weights[i + j] = placed.weights[i + j];
    }
    any = true;
  }
  if(!any)
    return out;
  std::fill(out.used.begin(), out.used.end(), uint8_t(0));
  for(size_t i = 0; i < out.verts.size(); i++)
    if(out.weights[i] != 0 && out.verts[i] < out.used.size())
      out.used[out.verts[i]] = 1;
  return out;
}

std::vector<float> settle(std::span<const float> moved, std::span<const float> decoded, std::span<const float> hold, std::span<const uint32_t> tris,
                          const SettleOptions& options)
{
  const size_t       n = moved.size() / 3;
  std::vector<float> out(moved.begin(), moved.end());
  if(decoded.size() != moved.size() || hold.size() != n)
    return out;
  // each vertex's neighbours along the cloth (an inner edge counts twice, which weighs them alike)
  std::vector<uint32_t> start(n + 1, 0), next;
  for(size_t i = 0; i + 2 < tris.size(); i += 3)
    for(size_t j = 0; j < 3; j++)
      if(tris[i + j] < n)
        start[tris[i + j] + 1] += 2;
  for(size_t v = 0; v < n; v++)
    start[v + 1] += start[v];
  next.assign(start[n], 0);
  {
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    for(size_t i = 0; i + 2 < tris.size(); i += 3)
      for(size_t j = 0; j < 3; j++)
      {
        const uint32_t a = tris[i + j], b = tris[i + (j + 1) % 3], c = tris[i + (j + 2) % 3];
        if(a >= n)
          continue;
        next[fill[a]++] = b < n ? b : a;
        next[fill[a]++] = c < n ? c : a;
      }
  }
  // what each address asks of its vertex
  std::vector<float>    want(n * 3, 0.f), change(n * 3, 0.f);
  std::vector<uint8_t>  reached(n, 0);
  std::vector<uint32_t> queue;
  for(size_t v = 0; v < n; v++)
  {
    if(hold[v] <= 0)
      continue;
    double d[3], len2 = 0;
    for(size_t k = 0; k < 3; k++)
    {
      d[k] = double(decoded[v * 3 + k]) - moved[v * 3 + k];
      len2 += d[k] * d[k];
    }
    const double scale = len2 > double(options.cap) * options.cap ? options.cap / std::sqrt(len2) : 1;
    for(size_t k = 0; k < 3; k++)
      want[v * 3 + k] = change[v * 3 + k] = float(d[k] * scale);
    reached[v] = 1;
    queue.push_back(uint32_t(v));
  }
  // Two held neighbours whose addresses would pull their edge to `tearFrom` times its length or more
  // (or as much shorter) lie on either side of a crease: cloth bridging the thighs, or the arm and
  // the chest, each end told to follow its own side. Such ends hold less, down to nothing at `tearTo`,
  // and take their neighbours' middle instead, which shares the pull over the cloth round them.
  std::vector<float> grip(hold.begin(), hold.end());
  for(size_t v = 0; v < n; v++)
  {
    if(hold[v] <= 0)
      continue;
    double worst = 1;
    for(uint32_t i = start[v]; i < start[v + 1]; i++)
    {
      const uint32_t u = next[i];
      if(u == v || hold[u] <= 0)
        continue;
      double was = 0, is = 0;
      for(size_t k = 0; k < 3; k++)
      {
        const double e = double(moved[v * 3 + k]) - moved[size_t(u) * 3 + k];
        const double f = e + want[v * 3 + k] - want[size_t(u) * 3 + k];
        was += e * e, is += f * f;
      }
      if(was <= 1e-10)
        continue;
      const double ratio = std::sqrt(is / was);
      worst              = std::max(worst, ratio < 1e-6 ? 1e6 : std::max(ratio, 1 / ratio));
    }
    if(worst > options.tearFrom)
    {
      const double t = std::min(1.0, (worst - options.tearFrom) / std::max(1e-6, double(options.tearTo) - options.tearFrom));
      grip[v]        = float(hold[v] * (1 - t * t * (3 - 2 * t)));
    }
  }
  // cloth nothing holds starts from the change of the held cloth nearest it, ring by ring
  for(size_t head = 0; head < queue.size(); head++)
  {
    const uint32_t v = queue[head];
    for(uint32_t i = start[v]; i < start[v + 1]; i++)
    {
      const uint32_t u = next[i];
      if(reached[u])
        continue;
      reached[u] = 1;
      for(size_t k = 0; k < 3; k++)
        change[u * 3 + k] = change[v * 3 + k];
      queue.push_back(u);
    }
  }
  // evened over the cloth: a vertex keeps its address's share and takes its neighbours' middle for the rest
  std::vector<float> even(n * 3);
  for(int round = 0; round < options.smooth; round++)
  {
    for(size_t v = 0; v < n; v++)
    {
      const uint32_t count = start[v + 1] - start[v];
      const double   h     = reached[v] ? std::min(1.0, double(grip[v])) : 1.0;
      if(count == 0 || h >= 1)
      {
        for(size_t k = 0; k < 3; k++)
          even[v * 3 + k] = h >= 1 && grip[v] > 0 ? want[v * 3 + k] : change[v * 3 + k];
        continue;
      }
      double sum[3] = {0, 0, 0};
      for(uint32_t i = start[v]; i < start[v + 1]; i++)
        for(size_t k = 0; k < 3; k++)
          sum[k] += change[size_t(next[i]) * 3 + k];
      for(size_t k = 0; k < 3; k++)
        even[v * 3 + k] = float(h * want[v * 3 + k] + (1 - h) * sum[k] / count);
    }
    change.swap(even);
  }
  for(size_t i = 0; i < out.size(); i++)
    out[i] += change[i];
  return out;
}

}  // namespace fv::garment
