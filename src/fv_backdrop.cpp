#include "fv_backdrop.hpp"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace fv::backdrop {

namespace {

constexpr float kPi = 3.14159265358979F;

// A hash of four numbers (PCG's output mix over a running sum): the same everywhere, so a seed's
// backdrop is
uint32_t mix(uint32_t v)
{
  v = v * 747796405U + 2891336453U;
  const uint32_t w = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (w >> 22U) ^ w;
}

uint32_t hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
  return mix(mix(mix(mix(a) + b) + c) + d);
}

// [0, 1)
float unit(uint32_t h)
{
  return float(h >> 8U) * (1.0F / 16777216.0F);
}

float smoothstep(float a, float b, float x)
{
  const float t = std::clamp((x - a) / (b - a), 0.0F, 1.0F);
  return t * t * (3.0F - 2.0F * t);
}

// Smooth value noise on a unit lattice, [0, 1)
float valueNoise(float x, float z, uint32_t seed, uint32_t channel)
{
  const float fx = std::floor(x), fz = std::floor(z);
  const int   ix = int(fx), iz = int(fz);
  const float tx = x - fx, tz = z - fz;
  const float ux = tx * tx * (3.0F - 2.0F * tx), uz = tz * tz * (3.0F - 2.0F * tz);
  const auto  at = [&](int dx, int dz) { return unit(hash(seed, channel, uint32_t(ix + dx), uint32_t(iz + dz))); };
  const float a = at(0, 0), b = at(1, 0), c = at(0, 1), d = at(1, 1);
  return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uz;
}

// Four octaves of it, [0, 1) and near 0.5 on average
float fbm(float x, float z, uint32_t seed, uint32_t channel)
{
  float sum = 0.0F, amplitude = 0.5F, total = 0.0F;
  for(uint32_t o = 0; o < 4; o++)
  {
    sum += amplitude * valueNoise(x, z, seed, channel + o);
    total += amplitude;
    amplitude *= 0.5F;
    // turned as well as halved, so the octaves' lattices don't line up
    const float nx = 1.6F * x + 1.2F * z, nz = -1.2F * x + 1.6F * z;
    x = nx + 17.3F;
    z = nz - 9.1F;
  }
  return sum / total;
}

// The noise channels: the ground's two scales, then each layer's cover
constexpr uint32_t kChannelHills = 1, kChannelFar = 11, kChannelShade = 21, kChannelCover = 100;

// How much of a layer's own density stands at `d` metres, and how much its plants have grown there
float thinning(const Layer& l, float d, float& growth)
{
  growth = 1.0F;
  float edge = l.fade > 0.0F ? 1.0F - smoothstep(l.to - l.fade, l.to, d) : 1.0F;
  if(l.fadeIn > 0.0F)
    edge *= smoothstep(l.from, l.from + l.fadeIn, d);
  if(l.thin <= 0.0F)
    return edge;
  const float f = 1.0F + (l.thinsGently ? d / l.thin : (d / l.thin) * (d / l.thin));
  growth        = std::min(std::sqrt(f), l.grow);
  return edge / f;
}

// The share of the ground a layer stands on at `d`: its cover, filled in toward its far edge
float coverAt(const Layer& l, float d)
{
  if(l.closeFar <= 0.0F || l.to <= l.from)
    return l.cover;
  const float t = (d - l.from) / (l.to - l.from);
  return l.cover + (1.0F - l.cover) * l.closeFar * smoothstep(0.35F, 1.0F, t);
}

// The noise level above which a share `cover` of the ground lies. fbm() is near a normal distribution
// about 0.5 (kNoiseSpread, measured by BackdropScatter.ALayersCoverIsItsShareOfTheGround); the normal's
// quantile by the logistic's, which is within a few percent of it
constexpr float kNoiseSpread = 0.135F;
float           coverLevel(float cover)
{
  const float p = std::clamp(cover, 0.001F, 0.999F);
  return 0.5F - kNoiseSpread * std::log(p / (1.0F - p)) / 1.702F;
}

}  // namespace

float height(const Spec& spec, float x, float z)
{
  if(!planted(spec) || spec.relief <= 0.0)
    return 0.0F;
  const float d    = std::sqrt(x * x + z * z);
  const float rise = smoothstep(kClearing, kClearing + kClearingEdge, d);
  if(rise <= 0.0F)
    return 0.0F;
  const uint32_t seed = uint32_t(spec.seed);
  // rolling ground, a metre and a half either way with hills some 45 m across; and further out
  // higher ground some 400 m across that only rises, so the hills are the horizon's
  const float rolling = 1.5F * (2.0F * fbm(x / 45.0F, z / 45.0F, seed, kChannelHills) - 1.0F);
  const float far     = std::min(d / kHorizon, 1.0F);
  const float hills   = 70.0F * far * std::sqrt(far) * fbm(x / 400.0F, z / 400.0F, seed, kChannelFar);
  return float(spec.relief) * rise * (rolling + hills);
}

Terrain terrain(const Spec& spec, float tile)
{
  Terrain t;
  // rings half a metre apart underfoot, then 6% further each, so the ground's detail follows what a
  // camera near the origin sees of it
  std::vector<float> radii;
  for(float r = 0.5F; r < kHorizon; r += std::max(0.5F, 0.06F * r))
    radii.push_back(r);
  radii.push_back(kHorizon);
  constexpr uint32_t kSegments = 192;
  const float        scale     = tile > 0.0F ? 1.0F / tile : 1.0F;
  const auto         vertex    = [&](float x, float z) {
    const float e = 0.25F;
    const float h = height(spec, x, z);
    // the height's gradient by central differences: the normal of the surface y = h(x, z)
    const float dx = (height(spec, x + e, z) - height(spec, x - e, z)) / (2.0F * e);
    const float dz = (height(spec, x, z + e) - height(spec, x, z - e)) / (2.0F * e);
    t.positions.emplace_back(x, h, z);
    t.normals.push_back(glm::normalize(glm::vec3(-dx, 1.0F, -dz)));
    t.uvs.emplace_back(x * scale, z * scale);
    t.shades.push_back(smoothstep(0.38F, 0.62F, fbm(x / 40.0F, z / 40.0F, uint32_t(spec.seed), kChannelShade)));
  };
  vertex(0.0F, 0.0F);
  for(const float r : radii)
    for(uint32_t s = 0; s < kSegments; s++)
    {
      const float a = 2.0F * kPi * float(s) / float(kSegments);
      vertex(r * std::cos(a), r * std::sin(a));
    }
  // counter-clockwise seen from above: the fan round the centre, then a quad a segment between rings
  const auto at = [&](size_t ring, uint32_t s) { return uint32_t(1 + ring * kSegments + s % kSegments); };
  for(uint32_t s = 0; s < kSegments; s++)
    t.indices.insert(t.indices.end(), {0U, at(0, s + 1), at(0, s)});
  for(size_t ring = 0; ring + 1 < radii.size(); ring++)
    for(uint32_t s = 0; s < kSegments; s++)
    {
      const uint32_t a = at(ring, s), b = at(ring, s + 1), c = at(ring + 1, s), d = at(ring + 1, s + 1);
      t.indices.insert(t.indices.end(), {a, b, c, b, d, c});
    }
  return t;
}

std::vector<Layer> layers(const Spec& spec)
{
  if(spec.kind != "meadow")
    return {};
  const float edge = float(spec.size) * 0.5F;
  const float fade = 0.35F * edge;
  return {
      // grass underfoot, thinning and growing with distance, round the scene and round the camera
      {.role = "grass", .perSquareMetre = 150.0, .from = 0.0F, .to = edge, .thin = 8.0F, .grow = 2.0F, .fade = fade, .follows = true},
      // and beyond it tufts, fewer and larger, fewer still with distance: seen from standing height they
      // hide the ground between them, so the grass has no end before the trees
      {.role = "grass", .perSquareMetre = 5.0, .from = edge - fade, .to = std::max(kTufts, edge + 40.0F), .thin = 20.0F, .grow = 1.3F,
       .thinsGently = true, .fade = 40.0F, .fadeIn = fade, .follows = true, .scaleMin = 1.8F, .scaleMax = 2.6F},
      // flowers in drifts
      {.role = "flower", .perSquareMetre = 8.0, .from = 0.0F, .to = edge, .thin = 10.0F, .grow = 1.5F, .fade = fade, .follows = true, .cover = 0.4F, .patch = 9.0F},
      // saplings here and there, clear of where the figures stand
      {.role = "sapling", .perSquareMetre = 0.03, .from = 7.0F, .to = edge + 10.0F, .cover = 0.5F, .patch = 25.0F, .scaleMin = 0.7F, .scaleMax = 1.6F},
      // trees in groves from 22 m, closing into woods toward 260 m; young ones to twice their size,
      // since a kit's trees are the ones light enough to stand in thousands
      {.role = "tree", .perSquareMetre = 0.02, .from = 22.0F, .to = 260.0F, .cover = 0.35F, .patch = 70.0F, .closeFar = 0.8F, .scaleMin = 0.8F, .scaleMax = 1.9F, .sink = 0.15F},
  };
}

float focusStep(const Spec& spec)
{
  return std::max(4.0F, float(spec.size) / 6.0F);
}

glm::vec2 focusNear(const Spec& spec, glm::vec2 at)
{
  const float step = focusStep(spec);
  return glm::round(at / step) * step;
}

std::vector<std::vector<glm::mat4>> scatter(const Spec& spec, const Layer& layer, uint32_t layerIndex, uint32_t variants, std::span<const glm::vec2> foci)
{
  std::vector<std::vector<glm::mat4>> out(variants);
  const double thickest = layer.perSquareMetre * spec.density;
  if(variants == 0 || !(thickest > 0.0) || layer.to <= layer.from)
    return out;
  // what the layer stands round: the origin, or each focus once
  std::vector<glm::vec2> centres;
  if(layer.follows)
    for(const glm::vec2& f : foci)
      if(std::find(centres.begin(), centres.end(), f) == centres.end())
        centres.push_back(f);
  if(centres.empty())
    centres.emplace_back(0.0F);
  // a jittered grid at the layer's thickest: a cell holds one plant at most, kept by the odds of the
  // layer's density there, so plants neither pile up nor line up. The grid is the world's: a cell's
  // plant is the same whichever centre asks for it
  const float    cell = float(1.0 / std::sqrt(thickest));
  const int      n    = int(std::ceil(layer.to / cell));
  const uint32_t seed = uint32_t(spec.seed);
  const uint32_t salt = 1000U + layerIndex;
  for(size_t c = 0; c < centres.size(); c++)
  {
    const glm::vec2 centre = centres[c];
    const int       cx = int(std::floor(centre.x / cell)), cz = int(std::floor(centre.y / cell));
    for(int iz = cz - n; iz <= cz + n; iz++)
      for(int ix = cx - n; ix <= cx + n; ix++)
      {
      const uint32_t h = hash(seed, salt, uint32_t(ix), uint32_t(iz));
      const float    x = (float(ix) + unit(h)) * cell;
      const float    z = (float(iz) + unit(mix(h + 1U))) * cell;
      const float    d = glm::distance(glm::vec2(x, z), centre);
      if(d < layer.from || d > layer.to)
        continue;
      // a place between two centres is the nearer one's (the first of two as near), and stands by
      // its distance to that one
      bool others = false;
      for(size_t o = 0; o < centres.size() && !others; o++)
      {
        const float od = o == c ? d : glm::distance(glm::vec2(x, z), centres[o]);
        others         = o != c && (od < d || (od == d && o < c));
      }
      if(others)
        continue;
      float growth = 1.0F;
      float odds   = thinning(layer, d, growth);
      if(const float cover = coverAt(layer, d); cover < 1.0F)
      {
        // where the noise is in its top `cover`, with a soft edge
        const float noise = fbm(x / layer.patch, z / layer.patch, seed, kChannelCover + 10U * layerIndex);
        const float level = coverLevel(cover);
        odds *= smoothstep(level - 0.02F, level + 0.02F, noise);
      }
      if(unit(mix(h + 2U)) >= odds)
        continue;
      const uint32_t variant = mix(h + 3U) % variants;
      const float    yaw     = 2.0F * kPi * unit(mix(h + 4U));
      const float    scale   = (layer.scaleMin + (layer.scaleMax - layer.scaleMin) * unit(mix(h + 5U))) * growth;
      glm::mat4      m       = glm::translate(glm::mat4(1.0F), glm::vec3(x, height(spec, x, z) - layer.sink * scale, z));
      m                      = glm::rotate(m, yaw, glm::vec3(0.0F, 1.0F, 0.0F));
      m                      = glm::scale(m, glm::vec3(scale));
      out[variant].push_back(m);
      }
  }
  return out;
}

uint64_t expectedCount(const Spec& spec, const Layer& layer)
{
  // rings a quarter of a metre wide: the density there, by its thinning and its cover
  double total = 0.0;
  for(float d = layer.from; d < layer.to; d += 0.25F)
  {
    float growth = 1.0F;
    total += 2.0 * kPi * (d + 0.125) * 0.25 * layer.perSquareMetre * spec.density * thinning(layer, d, growth) * coverAt(layer, d);
  }
  return uint64_t(total);
}

}  // namespace fv::backdrop
