#include "fv_backdrop.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

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
constexpr uint32_t kChannelHills = 1, kChannelFar = 11, kChannelShade = 21, kChannelDrift = 31, kChannelClump = 41, kChannelRidge = 51, kChannelCover = 100;

// How much of a layer's own density stands at `d` metres, and how much its plants have grown there
float thinning(const Layer& l, float d, float& growth)
{
  growth = 1.0F;
  const float edge = l.fade > 0.0F ? 1.0F - smoothstep(l.to - l.fade, l.to, d) : 1.0F;
  if(l.thin <= 0.0F)
    return edge;
  const float past = std::max(d - l.thinFrom, 0.0F) / l.thin;
  const float f    = 1.0F + past * past;
  growth        = std::min(std::sqrt(f), l.grow);
  return edge / f;
}

// The share of the ground a layer stands on at `d`: its cover, filled in toward its far edge or its
// near one
float coverAt(const Layer& l, float d)
{
  if((l.closeFar <= 0.0F && l.closeNear <= 0.0F) || l.to <= l.from)
    return l.cover;
  const float t      = (d - l.from) / (l.to - l.from);
  const float span   = l.closeSpan > 0.0F ? l.closeSpan / (l.to - l.from) : 0.3F;
  const float closed = std::max(l.closeFar * smoothstep(0.35F, 1.0F, t), l.closeNear * (1.0F - smoothstep(0.0F, span, t)));
  return l.cover + (1.0F - l.cover) * closed;
}

// How far across a layer's groves are at `d`
float patchAt(const Layer& l, float d)
{
  return std::max(l.patch, l.patchGrows * d);
}

// fbm() with features `across` metres wide, `base` at the least: the noise at the doublings of
// `base` either side of `across`, blended. (Dividing a place by a size that grows with its distance
// leaves a function of its direction alone, which is stripes out from the origin.)
float sizedNoise(float x, float z, float base, float across, uint32_t seed, uint32_t channel)
{
  const float level = std::max(std::log2(across / base), 0.0F);
  const float whole = std::floor(level);
  const float size  = base * std::exp2(whole);
  const float near  = fbm(x / size, z / size, seed, channel);
  const float share = smoothstep(0.0F, 1.0F, level - whole);
  if(share <= 0.0F)
    return near;
  return near + (fbm(x / (2.0F * size), z / (2.0F * size), seed, channel) - near) * share;
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
  // and past those, ridges a couple of kilometres across that climb to some hundreds of metres by
  // the land's end: the skyline from a rise, and the land a camera in the air looks out over
  const float ridges = d > kHorizon ? 320.0F * smoothstep(kHorizon, 5500.0F, d) * fbm(x / 2200.0F, z / 2200.0F, seed, kChannelRidge) : 0.0F;
  return float(spec.relief) * rise * (rolling + hills + ridges);
}

Terrain terrain(const Spec& spec, float tile)
{
  Terrain t;
  // rings half a metre apart underfoot, then 6% further each, so the ground's detail follows what a
  // camera near the origin sees of it
  std::vector<float> radii;
  for(float r = 0.5F; r < kLandsEnd; r += std::max(0.5F, kMeshGrowth * r))
    radii.push_back(r);
  radii.push_back(kLandsEnd);
  constexpr uint32_t kSegments = 192;
  const float        scale     = tile > 0.0F ? 1.0F / tile : 1.0F;
  // `e`: half the distance the slope is taken over, a ring's width where the rings are wide
  const auto         vertex    = [&](float x, float z, float e) {
    const float h = height(spec, x, z);
    // the height's gradient by central differences: the normal of the surface y = h(x, z)
    const float dx = (height(spec, x + e, z) - height(spec, x - e, z)) / (2.0F * e);
    const float dz = (height(spec, x, z + e) - height(spec, x, z - e)) / (2.0F * e);
    t.positions.emplace_back(x, h, z);
    t.normals.push_back(glm::normalize(glm::vec3(-dx, 1.0F, -dz)));
    t.uvs.emplace_back(x * scale, z * scale);
    // drifts some 40 m across, larger where the mesh is too coarse to draw those
    const float drift = std::max(40.0F, 3.0F * kMeshGrowth * std::sqrt(x * x + z * z));
    t.shades.push_back(smoothstep(0.38F, 0.62F, sizedNoise(x, z, 40.0F, drift, uint32_t(spec.seed), kChannelShade)));
    t.woods.push_back(woods(spec, x, z));
  };
  vertex(0.0F, 0.0F, 0.25F);
  for(const float r : radii)
    for(uint32_t s = 0; s < kSegments; s++)
    {
      const float a = 2.0F * kPi * float(s) / float(kSegments);
      vertex(r * std::cos(a), r * std::sin(a), std::max(0.25F, 0.5F * kMeshGrowth * r));
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
  if(spec.kind == "meadow")
  {
    const float edge = float(spec.size) * 0.5F;
    return {
        // (grass and flowers are tiles: grassCovers, drifts)
        // saplings here and there, clear of where the figures stand
        {.role = "sapling", .perSquareMetre = 0.03, .from = 7.0F, .to = edge + 10.0F, .cover = 0.5F, .patch = 25.0F, .scaleMin = 0.7F, .scaleMax = 1.6F},
        // trees in groves from 22 m, closing into woods toward kWoods; young ones to twice their size,
        // since a kit's trees are the ones light enough to stand in thousands
        {.role = "tree", .perSquareMetre = 0.02, .from = 22.0F, .to = kWoods, .cover = 0.35F, .patch = 70.0F, .closeFar = 0.8F, .scaleMin = 0.8F, .scaleMax = 1.9F, .sink = 0.15F},
        // (new layers go after these two: a layer's places follow from its index)
        // the wood carried on over the hills to the horizon, opening into woods and pasture: fewer
        // and larger with distance, as grown firs are, so the far hills cost a few thousand trees
        {.role = "tree", .perSquareMetre = 0.02, .from = kWoods, .to = kTreesEnd, .thin = 250.0F, .grow = 2.5F, .thinFrom = kWoods,
         .fade = 700.0F, .cover = 0.5F, .patch = 160.0F, .closeNear = 0.75F, .closeSpan = 190.0F, .patchGrows = 3.0F * kMeshGrowth, .scaleMin = 0.8F, .scaleMax = 1.9F, .sink = 0.15F},
        // bushes in thickets; a few rocks, half in the ground
        {.role = "shrub", .perSquareMetre = 0.012, .from = 9.0F, .to = kWoods, .fade = 40.0F, .cover = 0.3F, .patch = 35.0F, .scaleMin = 0.6F, .scaleMax = 1.3F, .sink = 0.03F},
        {.role = "rock", .perSquareMetre = 0.004, .from = 8.0F, .to = 400.0F, .fade = 80.0F, .cover = 0.3F, .patch = 60.0F, .scaleMin = 0.2F, .scaleMax = 0.8F, .sink = 0.1F},
    };
  }
  if(spec.kind == "forest")
  {
    const float near = 160.0F;
    return {
        // firs from the glade's edge, close enough to shut the view within some tens of metres, grown
        // to between one and two and a half times the kit's young ones
        {.role = "tree", .perSquareMetre = 0.026, .from = 7.5F, .to = near, .cover = 0.9F, .patch = 40.0F, .scaleMin = 1.0F, .scaleMax = 2.4F, .sink = 0.2F},
        // and on to the horizon, fewer and larger with distance
        {.role = "tree", .perSquareMetre = 0.026, .from = near, .to = kTreesEnd, .thin = 200.0F, .grow = 2.5F, .thinFrom = near,
         .fade = 700.0F, .cover = 0.9F, .patch = 120.0F, .patchGrows = 3.0F * kMeshGrowth, .scaleMin = 1.0F, .scaleMax = 2.4F, .sink = 0.2F},
        // the undergrowth, out to where a camera on the ground no longer tells it from the floor
        {.role = "sapling", .perSquareMetre = 0.03, .from = 5.5F, .to = 300.0F, .fade = 60.0F, .cover = 0.55F, .patch = 18.0F, .scaleMin = 0.6F, .scaleMax = 1.7F},
        {.role = "shrub", .perSquareMetre = 0.05, .from = 5.5F, .to = 300.0F, .fade = 60.0F, .cover = 0.45F, .patch = 22.0F, .scaleMin = 0.5F, .scaleMax = 1.2F, .sink = 0.03F},
        {.role = "rock", .perSquareMetre = 0.025, .from = 5.5F, .to = 400.0F, .fade = 80.0F, .cover = 0.5F, .patch = 30.0F, .scaleMin = 0.15F, .scaleMax = 0.7F, .sink = 0.08F},
        {.role = "stump", .perSquareMetre = 0.008, .from = 6.0F, .to = 300.0F, .fade = 60.0F, .scaleMin = 0.6F, .scaleMax = 1.2F, .sink = 0.03F},
        {.role = "log", .perSquareMetre = 0.015, .from = 5.5F, .to = 300.0F, .fade = 60.0F, .scaleMin = 0.7F, .scaleMax = 1.5F},
    };
  }
  return {};
}

float woods(const Spec& spec, float x, float z)
{
  // the layer that carries the trees on with distance: the last that thins
  const std::vector<Layer> all = layers(spec);
  int                      far = -1;
  for(size_t i = 0; i < all.size(); i++)
    if(all[i].role == "tree" && all[i].thin > 0.0F && all[i].grow > 1.0F)
      far = int(i);
  if(far < 0)
    return 0.0F;
  const Layer& l = all[size_t(far)];
  const float  d = std::sqrt(x * x + z * z);
  // its trees keep their cover until their growth is spent
  const float spent = l.thinFrom + l.thin * std::sqrt(l.grow * l.grow - 1.0F);
  const float paint = smoothstep(0.8F * spent, 1.3F * spent, d);
  if(paint <= 0.0F)
    return 0.0F;
  const float cover = coverAt(l, std::min(d, l.to));
  if(cover >= 1.0F)
    return paint;
  const float noise = sizedNoise(x, z, l.patch, patchAt(l, d), uint32_t(spec.seed), kChannelCover + 10U * uint32_t(far));
  const float level = coverLevel(cover);
  return paint * smoothstep(level - 0.02F, level + 0.02F, noise);
}

std::vector<std::vector<glm::mat4>> scatter(const Spec& spec, const Layer& layer, uint32_t layerIndex, uint32_t variants)
{
  std::vector<std::vector<glm::mat4>> out(variants);
  const double thickest = layer.perSquareMetre * spec.density;
  if(variants == 0 || !(thickest > 0.0) || layer.to <= layer.from)
    return out;
  // a jittered grid at the layer's thickest: a cell holds one plant at most, kept by the odds of the
  // layer's density there, so plants neither pile up nor line up
  const float    cell = float(1.0 / std::sqrt(thickest));
  const int      n    = int(std::ceil(layer.to / cell));
  const uint32_t seed = uint32_t(spec.seed);
  const uint32_t salt = 1000U + layerIndex;
  for(int iz = -n; iz < n; iz++)
    for(int ix = -n; ix < n; ix++)
    {
      const uint32_t h = hash(seed, salt, uint32_t(ix), uint32_t(iz));
      const float    x = (float(ix) + unit(h)) * cell;
      const float    z = (float(iz) + unit(mix(h + 1U))) * cell;
      const float    d = std::sqrt(x * x + z * z);
      if(d < layer.from || d > layer.to)
        continue;
      float growth = 1.0F;
      float odds   = thinning(layer, d, growth);
      if(const float cover = coverAt(layer, d); cover < 1.0F)
      {
        // where the noise is in its top `cover`, with a soft edge
        const float noise = sizedNoise(x, z, layer.patch, patchAt(layer, d), seed, kChannelCover + 10U * layerIndex);
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
  return out;
}

// ---- tiles

std::array<Cover, kGrassLevels> grassCovers(const Spec& spec)
{
  // thinner by about four a level and larger by the root of it and a little less: seen from standing
  // height a plant hides a strip of the ground behind it that lengthens with distance. Under firs
  // the grass is tufts on the needles, a seventh as thick
  // the grass is tufts on the needles, a seventh as thick and in clumps: spread evenly, plants that
  // sparse show the grid they stand on
  const bool   forest = spec.kind == "forest";
  const double thick  = (forest ? 1.0 / 7.0 : 1.0) * spec.density;
  const float  clumps = forest ? 0.35F : 1.0F;
  return {{
      {.role = "grass", .tile = 2.0F, .perSquareMetre = 150.0 * thick, .scaleMin = 0.85F, .scaleMax = 1.35F, .sink = 0.02F, .clumps = clumps, .salt = 0},
      {.role = "grass", .tile = 4.0F, .perSquareMetre = 36.0 * thick, .scaleMin = 1.3F, .scaleMax = 1.9F, .sink = 0.06F, .clumps = clumps, .salt = 1},
      {.role = "grass", .tile = 8.0F, .perSquareMetre = 5.0 * thick, .scaleMin = 1.9F, .scaleMax = 2.6F, .sink = 0.15F, .clumps = clumps, .salt = 2},
  }};
}

std::array<float, kGrassLevels> grassReach(const Spec& spec)
{
  const float size = float(spec.size);
  const float thinner = 0.8F * size;
  return {0.24F * size, thinner, std::max(130.0F, thinner + 60.0F)};
}

Drifts drifts(const Spec& spec)
{
  if(spec.kind == "forest")
    // ferns, a pace apart in a drift's middle: over half the floor, and seen further than a flower is
    return {.covers = {{
                {.role = "fern", .tile = 4.0F, .perSquareMetre = 1.6 * spec.density, .scaleMin = 0.8F, .scaleMax = 1.5F, .sink = 0.03F, .salt = 10},
                {.role = "fern", .tile = 4.0F, .perSquareMetre = 0.5 * spec.density, .scaleMin = 0.7F, .scaleMax = 1.3F, .sink = 0.03F, .salt = 11},
            }},
            .reach  = 0.9F * float(spec.size),
            .across = 14.0F,
            .middle = 0.3F,
            .ground = 0.55F};
  // flowers in drifts some 9 m across over two fifths of the ground
  return {.covers = {{
              {.role = "flower", .tile = 4.0F, .perSquareMetre = 9.0 * spec.density, .scaleMin = 0.8F, .scaleMax = 1.3F, .sink = 0.03F, .salt = 10},
              {.role = "flower", .tile = 4.0F, .perSquareMetre = 3.0 * spec.density, .scaleMin = 0.8F, .scaleMax = 1.3F, .sink = 0.03F, .salt = 11},
          }},
          .reach  = 0.5F * float(spec.size),
          .across = 9.0F,
          .middle = 0.22F,
          .ground = 0.42F};
}

std::vector<std::vector<glm::mat4>> patch(const Spec& spec, const Cover& cover, uint32_t variants)
{
  std::vector<std::vector<glm::mat4>> out(variants);
  if(variants == 0 || !(cover.perSquareMetre > 0.0) || !(cover.tile > 0.0F))
    return out;
  // a jittered grid: a plant a cell, as many cells a side as gives the tile its density; in clumps,
  // that many more cells, of which the share where a noise is highest keep their plant (the share
  // of this tile's own cells, so a tile has its count whatever its noise)
  const float    share = std::clamp(cover.clumps, 0.05F, 1.0F);
  const int      n     = std::max(1, int(std::lround(double(cover.tile) * std::sqrt(cover.perSquareMetre / double(share)))));
  const float    cell  = cover.tile / float(n);
  const uint32_t seed  = uint32_t(spec.seed);
  const uint32_t salt  = 5000U + cover.salt;
  const float    clump = 0.45F * cover.tile;
  const auto     noise = [&](int ix, int iz) {
    return fbm((float(ix) + 0.5F) * cell / clump, (float(iz) + 0.5F) * cell / clump, seed, kChannelClump + cover.salt);
  };
  float level = -1.0F;
  if(share < 1.0F)
  {
    std::vector<float> all;
    all.reserve(size_t(n) * size_t(n));
    for(int iz = 0; iz < n; iz++)
      for(int ix = 0; ix < n; ix++)
        all.push_back(noise(ix, iz));
    const size_t drop = std::min(all.size() - 1, size_t(std::lround(double(all.size()) * double(1.0F - share))));
    std::nth_element(all.begin(), all.begin() + ptrdiff_t(drop), all.end());
    level = all[drop];
  }
  for(int iz = 0; iz < n; iz++)
    for(int ix = 0; ix < n; ix++)
    {
      if(share < 1.0F && noise(ix, iz) < level)
        continue;
      const uint32_t h       = hash(seed, salt, uint32_t(ix), uint32_t(iz));
      const float    x       = (float(ix) + unit(h)) * cell - 0.5F * cover.tile;
      const float    z       = (float(iz) + unit(mix(h + 1U))) * cell - 0.5F * cover.tile;
      const uint32_t variant = mix(h + 3U) % variants;
      const float    yaw     = 2.0F * kPi * unit(mix(h + 4U));
      const float    scale   = cover.scaleMin + (cover.scaleMax - cover.scaleMin) * unit(mix(h + 5U));
      glm::mat4      m       = glm::translate(glm::mat4(1.0F), glm::vec3(x, 0.0F, z));
      m                      = glm::rotate(m, yaw, glm::vec3(0.0F, 1.0F, 0.0F));
      m                      = glm::scale(m, glm::vec3(scale));
      out[variant].push_back(m);
    }
  return out;
}

namespace {

// A cell's centre on the grid of tiles `size` across
glm::vec2 cellCentre(float size, int32_t x, int32_t z)
{
  return {(float(x) + 0.5F) * size, (float(z) + 0.5F) * size};
}

float nearest(std::span<const glm::vec2> foci, glm::vec2 at)
{
  float d = std::numeric_limits<float>::max();
  for(const glm::vec2& f : foci)
    d = std::min(d, glm::distance(f, at));
  return d;
}

// A cell that was split stays split out to this much further than where one splits
constexpr float kStaysSplit = 1.12F;

}  // namespace

std::vector<Tile> grassTiles(const Spec& spec, std::span<const glm::vec2> foci, std::span<const Tile> was)
{
  std::vector<Tile> out;
  if(!planted(spec))
    return out;
  const glm::vec2            origin(0.0F);
  std::span<const glm::vec2> round = foci.empty() ? std::span<const glm::vec2>(&origin, 1) : foci;
  const auto                 covers = grassCovers(spec);
  const auto                 reach  = grassReach(spec);
  // the cells that were split, by their own level: a thick tile's parent and grandparent, a middle
  // tile's parent
  std::set<Tile> split;
  for(const Tile& t : was)
  {
    if(t.level == 0)
    {
      split.insert({t.x >> 1, t.z >> 1, 1});
      split.insert({t.x >> 2, t.z >> 2, 2});
    }
    else if(t.level == 1)
      split.insert({t.x >> 1, t.z >> 1, 2});
  }
  const auto splits = [&](const Tile& cell) {
    const float d = nearest(round, cellCentre(covers[cell.level].tile, cell.x, cell.z));
    const float r = reach[cell.level - 1U];
    return d < r || (d < r * kStaysSplit && split.count(cell));
  };
  // every coarse cell within reach of a focus, once
  const float     coarse = covers[2].tile;
  std::set<Tile> cells;
  for(const glm::vec2& f : round)
  {
    const int n  = int(std::ceil(reach[2] / coarse)) + 1;
    const int cx = int(std::floor(f.x / coarse)), cz = int(std::floor(f.y / coarse));
    for(int z = cz - n; z <= cz + n; z++)
      for(int x = cx - n; x <= cx + n; x++)
        if(glm::distance(cellCentre(coarse, x, z), f) < reach[2])
          cells.insert({x, z, 2});
  }
  for(const Tile& c2 : cells)
  {
    if(!splits(c2))
    {
      out.push_back(c2);
      continue;
    }
    for(int j = 0; j < 2; j++)
      for(int i = 0; i < 2; i++)
      {
        const Tile c1{c2.x * 2 + i, c2.z * 2 + j, 1};
        if(!splits(c1))
        {
          out.push_back(c1);
          continue;
        }
        for(int q = 0; q < 2; q++)
          for(int p = 0; p < 2; p++)
            out.push_back({c1.x * 2 + p, c1.z * 2 + q, 0});
      }
  }
  return out;
}

std::vector<Tile> driftTiles(const Spec& spec, std::span<const glm::vec2> foci)
{
  std::vector<Tile> out;
  if(!planted(spec))
    return out;
  const glm::vec2            origin(0.0F);
  std::span<const glm::vec2> round = foci.empty() ? std::span<const glm::vec2>(&origin, 1) : foci;
  const Drifts               in    = drifts(spec);
  const float                size  = in.covers[0].tile;
  const float                reach = in.reach;
  // their middles thick, their edges thin
  const float middle = coverLevel(in.middle), edge = coverLevel(in.ground);
  std::set<Tile> cells;
  for(const glm::vec2& f : round)
  {
    const int n  = int(std::ceil(reach / size)) + 1;
    const int cx = int(std::floor(f.x / size)), cz = int(std::floor(f.y / size));
    for(int z = cz - n; z <= cz + n; z++)
      for(int x = cx - n; x <= cx + n; x++)
      {
        const glm::vec2 c = cellCentre(size, x, z);
        if(glm::distance(c, f) >= reach)
          continue;
        const float noise = fbm(c.x / in.across, c.y / in.across, uint32_t(spec.seed), kChannelDrift);
        if(noise > edge)
          cells.insert({x, z, uint8_t(noise > middle ? 0 : 1)});
      }
  }
  out.assign(cells.begin(), cells.end());
  return out;
}

Place tilePlace(const Spec& spec, float size, int32_t x, int32_t z, float sink)
{
  const glm::vec2 c = cellCentre(size, x, z);
  // the ground's lean across the tile, by its height either side
  const float     e  = 0.5F * size;
  const float     dx = (height(spec, c.x + e, c.y) - height(spec, c.x - e, c.y)) / (2.0F * e);
  const float     dz = (height(spec, c.x, c.y + e) - height(spec, c.x, c.y - e)) / (2.0F * e);
  const glm::vec3 up = glm::normalize(glm::vec3(-dx, 1.0F, -dz));
  // a turn about the vertical by the cell, then the lean: the shortest turn that takes +y to the ground's normal
  const uint32_t  h    = hash(uint32_t(spec.seed), 7000U + uint32_t(size * 16.0F), uint32_t(x), uint32_t(z));
  const glm::quat turn = glm::angleAxis(0.5F * kPi * float(h & 3U), glm::vec3(0.0F, 1.0F, 0.0F));
  const glm::vec3 axis = glm::cross(glm::vec3(0.0F, 1.0F, 0.0F), up);
  const float     s    = glm::length(axis);
  const glm::quat lean = s > 1e-6F ? glm::angleAxis(std::asin(std::min(s, 1.0F)), axis / s) : glm::quat(1.0F, 0.0F, 0.0F, 0.0F);
  return {.position = {c.x, height(spec, c.x, c.y) - sink * float(std::max(spec.relief, 0.25)), c.y}, .rotation = lean * turn};
}

std::vector<std::string> roles(const Spec& spec)
{
  std::set<std::string> out;
  if(planted(spec))
  {
    for(const Layer& layer : layers(spec))
      out.insert(layer.role);
    for(const Cover& cover : grassCovers(spec))
      out.insert(cover.role);
    for(const Cover& cover : drifts(spec).covers)
      out.insert(cover.role);
  }
  return {out.begin(), out.end()};
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
