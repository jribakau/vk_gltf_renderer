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

// Four octaves of it, [0, 1) and near 0.5 on average. `finest`, in lattice cells of the first octave:
// an octave whose cells are under twice this fades to its mean, and under this is gone (each octave's
// are half the last's). A colour drawn at a mesh's vertices has no use for detail finer than they are
// apart: sampled there it is drawn as triangles, a colour that flips across one edge.
float fbm(float x, float z, uint32_t seed, uint32_t channel, float finest = 0.0F)
{
  float sum = 0.0F, amplitude = 0.5F, total = 0.0F, cells = 1.0F;
  for(uint32_t o = 0; o < 4; o++)
  {
    const float kept = finest > 0.0F ? smoothstep(finest, 2.0F * finest, cells) : 1.0F;
    sum += amplitude * (0.5F + kept * (valueNoise(x, z, seed, channel + o) - 0.5F));
    total += amplitude;
    amplitude *= 0.5F;
    cells *= 0.5F;
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
// `finest`: metres, as fbm()'s (0 every octave).
float sizedNoise(float x, float z, float base, float across, uint32_t seed, uint32_t channel, float finest = 0.0F)
{
  const float level = std::max(std::log2(across / base), 0.0F);
  const float whole = std::floor(level);
  const float size  = base * std::exp2(whole);
  const float near  = fbm(x / size, z / size, seed, channel, finest / size);
  const float share = smoothstep(0.0F, 1.0F, level - whole);
  if(share <= 0.0F)
    return near;
  return near + (fbm(x / (2.0F * size), z / (2.0F * size), seed, channel, finest / (2.0F * size)) - near) * share;
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

namespace {

// The ground mesh's rings: half a metre apart underfoot, then kMeshGrowth further each, so the ground's detail
// follows what a camera near the origin sees of it; each of kSegments vertices round
constexpr uint32_t kSegments = 384;

// How far apart the rings are `d` metres out
float meshSpacing(float d)
{
  return std::max(0.5F, kMeshGrowth * d);
}

const std::vector<float>& meshRadii()
{
  static const std::vector<float> radii = [] {
    std::vector<float> out;
    for(float r = 0.5F; r < kLandsEnd; r += meshSpacing(r))
      out.push_back(r);
    out.push_back(kLandsEnd);
    return out;
  }();
  return radii;
}

// The finest detail a colour of the mesh's vertices keeps `d` metres out: four rings (fbm's `finest`)
float meshDetail(float d)
{
  return 2.0F * meshSpacing(d);
}

// A vertex of a ring's, on the ground's plan
glm::vec2 meshVertex(float r, uint32_t segment)
{
  const float a = 2.0F * kPi * float(segment) / float(kSegments);
  return {r * std::cos(a), r * std::sin(a)};
}

}  // namespace

float groundAt(const Spec& spec, float x, float z)
{
  const std::vector<float>& radii = meshRadii();
  const float               d     = std::sqrt(x * x + z * z);
  // inside the first ring, and in the clearing round it, the ground is flat; past the last, nothing
  if(d < radii.front() || d >= radii.back())
    return height(spec, x, z);
  const size_t   ring = size_t(std::upper_bound(radii.begin(), radii.end(), d) - radii.begin()) - 1U;
  float          a    = std::atan2(z, x);
  if(a < 0.0F)
    a += 2.0F * kPi;
  const uint32_t s = std::min(uint32_t(a / (2.0F * kPi) * float(kSegments)), kSegments - 1U);
  // the quad between the rings, split as terrain() splits it: (a, b, c) and (b, d, c)
  const glm::vec2 pa = meshVertex(radii[ring], s), pb = meshVertex(radii[ring], s + 1U);
  const glm::vec2 pc = meshVertex(radii[ring + 1], s), pd = meshVertex(radii[ring + 1], s + 1U);
  const glm::vec2 p(x, z);
  // barycentric weights of p in a triangle on the plan; the triangle p is in, or least outside of
  const auto weights = [&](glm::vec2 u, glm::vec2 v, glm::vec2 w) {
    const float area = (v.x - u.x) * (w.y - u.y) - (w.x - u.x) * (v.y - u.y);
    const float wu   = ((v.x - p.x) * (w.y - p.y) - (w.x - p.x) * (v.y - p.y)) / area;
    const float wv   = ((w.x - p.x) * (u.y - p.y) - (u.x - p.x) * (w.y - p.y)) / area;
    return glm::vec3(wu, wv, 1.0F - wu - wv);
  };
  const glm::vec3 first  = weights(pa, pb, pc);
  const glm::vec3 second = weights(pb, pd, pc);
  const auto      h      = [&](glm::vec2 q) { return height(spec, q.x, q.y); };
  if(std::min({first.x, first.y, first.z}) >= std::min({second.x, second.y, second.z}))
    return first.x * h(pa) + first.y * h(pb) + first.z * h(pc);
  return second.x * h(pb) + second.y * h(pd) + second.z * h(pc);
}

namespace {

// The layer that carries a kind's trees on with distance, whose far groves the ground is painted as
// (`woods`): found once for a spec, as the ground's mesh asks at each of its vertices
struct FarWoods
{
  bool     any = false;
  Layer    layer;
  uint32_t index = 0;  // in layers(): its cover's noise channel is its scatter's
  float    spent = 0.0F;  // how far out its trees keep their cover: until their growth is spent
};
FarWoods farWoods(const Spec& spec);
float    woodsAt(const FarWoods& far, const Spec& spec, float x, float z);

}  // namespace

Terrain terrain(const Spec& spec, float tile)
{
  Terrain t;
  const std::vector<float>& radii = meshRadii();
  const float               scale = tile > 0.0F ? 1.0F / tile : 1.0F;
  const FarWoods            far   = farWoods(spec);
  // `e`: half the distance the slope is taken over, a ring's width where the rings are wide
  const auto         vertex    = [&](float x, float z, float e) {
    const float h = height(spec, x, z);
    // the height's gradient by central differences: the normal of the surface y = h(x, z)
    const float dx = (height(spec, x + e, z) - height(spec, x - e, z)) / (2.0F * e);
    const float dz = (height(spec, x, z + e) - height(spec, x, z - e)) / (2.0F * e);
    t.positions.emplace_back(x, h, z);
    t.normals.push_back(glm::normalize(glm::vec3(-dx, 1.0F, -dz)));
    t.uvs.emplace_back(x * scale, z * scale);
    // drifts some 40 m across, larger where the mesh is too coarse to draw those, and no finer in
    // their edges than it draws
    const float d     = std::sqrt(x * x + z * z);
    const float drift = std::max(40.0F, kPatchGrowth * d);
    t.shades.push_back(smoothstep(0.38F, 0.62F, sizedNoise(x, z, 40.0F, drift, uint32_t(spec.seed), kChannelShade, meshDetail(d))));
    t.woods.push_back(woodsAt(far, spec, x, z));
  };
  vertex(0.0F, 0.0F, 0.25F);
  for(const float r : radii)
    for(uint32_t s = 0; s < kSegments; s++)
    {
      const glm::vec2 v = meshVertex(r, s);
      vertex(v.x, v.y, std::max(0.25F, 0.5F * kMeshGrowth * r));
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

namespace {

// Where a forest's near firs give way to its far ones, fewer and larger with distance
constexpr float kForestNear = 160.0F;

// A kind's own layers
std::vector<Layer> kindLayers(const Spec& spec);

// Mature woods (Spec::mature): grown firs among the near woods' young ones, after a kind's own layers
// so none of theirs moves. The kit's grown firs are 14 m (Poly Haven's fir_tree_01 as the app keeps
// it); at 1 to 1.5 times that they are the old trees of a wood, over the young ones' 6 to 20 m.
// A forest's stand among its near firs as one in five or so of the canopy's; a meadow's in its groves
// and at its wood's edge, and a few alone in the grass, as old trees are left standing in pasture.
Layer grownForest(float near)
{
  return {.role = "grown", .perSquareMetre = 0.005, .from = 12.0F, .to = near, .fade = 30.0F, .cover = 0.7F, .patch = 50.0F,
          .scaleMin = 1.0F, .scaleMax = 1.5F, .sink = 0.3F};
}
Layer grownMeadow()
{
  return {.role = "grown", .perSquareMetre = 0.003, .from = 30.0F, .to = kWoods, .fade = 40.0F, .cover = 0.4F, .patch = 70.0F,
          .closeFar = 0.6F, .scaleMin = 1.0F, .scaleMax = 1.5F, .sink = 0.3F};
}

}  // namespace

std::vector<Layer> layers(const Spec& spec)
{
  std::vector<Layer> all = kindLayers(spec);
  if(spec.mature && !all.empty())
    all.push_back(spec.kind == "forest" ? grownForest(kForestNear) : grownMeadow());
  return all;
}

namespace {

std::vector<Layer> kindLayers(const Spec& spec)
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
         .fade = 700.0F, .cover = 0.5F, .patch = 160.0F, .closeNear = 0.75F, .closeSpan = 190.0F, .patchGrows = kPatchGrowth, .scaleMin = 0.8F, .scaleMax = 1.9F, .sink = 0.15F},
        // bushes in thickets; a few rocks, half in the ground
        {.role = "shrub", .perSquareMetre = 0.012, .from = 9.0F, .to = kWoods, .fade = 40.0F, .cover = 0.3F, .patch = 35.0F, .scaleMin = 0.6F, .scaleMax = 1.3F, .sink = 0.03F},
        {.role = "rock", .perSquareMetre = 0.004, .from = 8.0F, .to = 400.0F, .fade = 80.0F, .cover = 0.3F, .patch = 60.0F, .scaleMin = 0.2F, .scaleMax = 0.8F, .sink = 0.1F},
    };
  }
  if(spec.kind == "forest")
  {
    const float near = kForestNear;
    return {
        // firs from the glade's edge, close enough to shut the view within some tens of metres, grown
        // to between one and two and a half times the kit's young ones
        {.role = "tree", .perSquareMetre = 0.026, .from = 7.5F, .to = near, .cover = 0.9F, .patch = 40.0F, .scaleMin = 1.0F, .scaleMax = 2.4F, .sink = 0.2F},
        // and on to the horizon, fewer and larger with distance
        {.role = "tree", .perSquareMetre = 0.026, .from = near, .to = kTreesEnd, .thin = 200.0F, .grow = 2.5F, .thinFrom = near,
         .fade = 700.0F, .cover = 0.9F, .patch = 120.0F, .patchGrows = kPatchGrowth, .scaleMin = 1.0F, .scaleMax = 2.4F, .sink = 0.2F},
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

}  // namespace

namespace {

FarWoods farWoods(const Spec& spec)
{
  // the layer that carries the trees on with distance: the last that thins
  FarWoods                 far;
  const std::vector<Layer> all = layers(spec);
  for(size_t i = 0; i < all.size(); i++)
    if(all[i].role == "tree" && all[i].thin > 0.0F && all[i].grow > 1.0F)
    {
      far.any   = true;
      far.layer = all[i];
      far.index = uint32_t(i);
    }
  if(far.any)
    far.spent = far.layer.thinFrom + far.layer.thin * std::sqrt(far.layer.grow * far.layer.grow - 1.0F);
  return far;
}

float woodsAt(const FarWoods& far, const Spec& spec, float x, float z)
{
  if(!far.any)
    return 0.0F;
  const Layer& l     = far.layer;
  const float  d     = std::sqrt(x * x + z * z);
  const float  paint = smoothstep(0.8F * far.spent, 1.3F * far.spent, d);
  if(paint <= 0.0F)
    return 0.0F;
  const float cover = coverAt(l, std::min(d, l.to));
  if(cover >= 1.0F)
    return paint;
  // the groves' noise as the trees' (scatter), but with no detail finer than the mesh draws; and its
  // edge four rings across at the least, by how fast the noise changes here, where a step at one
  // level was drawn a triangle at a time (at three, the smoothstep's steepest ring still took half the
  // way)
  const float    across  = patchAt(l, d);
  const float    detail  = meshDetail(d);
  const uint32_t channel = kChannelCover + 10U * far.index;
  const auto     noiseAt = [&](float u, float v) { return sizedNoise(u, v, l.patch, across, uint32_t(spec.seed), channel, detail); };
  const float    ring    = meshSpacing(d);
  const float    noise   = noiseAt(x, z);
  const float    change  = glm::length(glm::vec2(noiseAt(x + ring, z) - noise, noiseAt(x, z + ring) - noise));
  const float    edge    = std::max(0.02F, 2.0F * change);
  const float    level   = coverLevel(cover);
  return paint * smoothstep(level - edge, level + edge, noise);
}

}  // namespace

float woods(const Spec& spec, float x, float z)
{
  return woodsAt(farWoods(spec), spec, x, z);
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
      glm::mat4      m       = glm::translate(glm::mat4(1.0F), glm::vec3(x, groundAt(spec, x, z) - layer.sink * scale, z));
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
  // the grass is tufts on the needles, a seventh as thick and in clumps: spread evenly, plants that
  // sparse show the grid they stand on
  const bool   forest = spec.kind == "forest";
  const double thick  = (forest ? 1.0 / 7.0 : 1.0) * spec.density;
  const float  clumps = forest ? 0.35F : 1.0F;
  // shade-grown on brown needles, a wood's grass is darker and yellower than the meadow green the kit
  // gives it, which stood out of the olive floor as if lit from elsewhere
  const glm::vec3 tint = forest ? glm::vec3(0.8F, 0.75F, 0.45F) : glm::vec3(1.0F);
  return {{
      {.role = "grass", .tile = 2.0F, .perSquareMetre = 150.0 * thick, .scaleMin = 0.85F, .scaleMax = 1.35F, .sink = 0.02F, .clumps = clumps, .salt = 0, .tint = tint},
      {.role = "grass", .tile = 4.0F, .perSquareMetre = 36.0 * thick, .scaleMin = 1.3F, .scaleMax = 1.9F, .sink = 0.06F, .clumps = clumps, .salt = 1, .tint = tint},
      {.role = "grass", .tile = 8.0F, .perSquareMetre = 5.0 * thick, .scaleMin = 1.9F, .scaleMax = 2.6F, .sink = 0.15F, .clumps = clumps, .salt = 2, .tint = tint},
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

// How far the nearest focus is from a place on the ground
float nearest(std::span<const Focus> foci, glm::vec2 at)
{
  float d = std::numeric_limits<float>::max();
  for(const Focus& f : foci)
    d = std::min(d, glm::length(glm::vec3(f.x - at.x, f.y, f.z - at.y)));
  return d;
}

// How far along the ground from under a focus a reach goes: none when the focus is that high
float acrossGround(const Focus& f, float reach)
{
  return f.y < reach ? std::sqrt(reach * reach - f.y * f.y) : -1.0F;
}

// A cell that was split stays split out to this much further than where one splits
constexpr float kStaysSplit = 1.12F;

}  // namespace

std::vector<Tile> grassTiles(const Spec& spec, std::span<const Focus> foci, std::span<const Tile> was)
{
  std::vector<Tile> out;
  if(!planted(spec))
    return out;
  const Focus            origin(0.0F);
  std::span<const Focus> round  = foci.empty() ? std::span<const Focus>(&origin, 1) : foci;
  const auto             covers = grassCovers(spec);
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
  for(const Focus& f : round)
  {
    const float across = acrossGround(f, reach[2]);
    if(across < 0.0F)
      continue;
    const glm::vec2 under(f.x, f.z);
    const int       n  = int(std::ceil(across / coarse)) + 1;
    const int       cx = int(std::floor(under.x / coarse)), cz = int(std::floor(under.y / coarse));
    for(int z = cz - n; z <= cz + n; z++)
      for(int x = cx - n; x <= cx + n; x++)
        if(glm::distance(cellCentre(coarse, x, z), under) < across)
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

std::vector<Tile> driftTiles(const Spec& spec, std::span<const Focus> foci)
{
  std::vector<Tile> out;
  if(!planted(spec))
    return out;
  const Focus            origin(0.0F);
  std::span<const Focus> round = foci.empty() ? std::span<const Focus>(&origin, 1) : foci;
  const Drifts               in    = drifts(spec);
  const float                size  = in.covers[0].tile;
  const float                reach = in.reach;
  // their middles thick, their edges thin
  const float middle = coverLevel(in.middle), edge = coverLevel(in.ground);
  std::set<Tile> cells;
  for(const Focus& f : round)
  {
    const float across = acrossGround(f, reach);
    if(across < 0.0F)
      continue;
    const glm::vec2 under(f.x, f.z);
    const int       n  = int(std::ceil(across / size)) + 1;
    const int       cx = int(std::floor(under.x / size)), cz = int(std::floor(under.y / size));
    for(int z = cz - n; z <= cz + n; z++)
      for(int x = cx - n; x <= cx + n; x++)
      {
        const glm::vec2 c = cellCentre(size, x, z);
        if(glm::distance(c, under) >= across)
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
  // the ground's lean across the tile, by its height either side: the ground as drawn (groundAt)
  const float     e  = 0.5F * size;
  const float     dx = (groundAt(spec, c.x + e, c.y) - groundAt(spec, c.x - e, c.y)) / (2.0F * e);
  const float     dz = (groundAt(spec, c.x, c.y + e) - groundAt(spec, c.x, c.y - e)) / (2.0F * e);
  const glm::vec3 up = glm::normalize(glm::vec3(-dx, 1.0F, -dz));
  // a turn about the vertical by the cell, then the lean: the shortest turn that takes +y to the ground's normal
  const uint32_t  h    = hash(uint32_t(spec.seed), 7000U + uint32_t(size * 16.0F), uint32_t(x), uint32_t(z));
  const glm::quat turn = glm::angleAxis(0.5F * kPi * float(h & 3U), glm::vec3(0.0F, 1.0F, 0.0F));
  const glm::vec3 axis = glm::cross(glm::vec3(0.0F, 1.0F, 0.0F), up);
  const float     s    = glm::length(axis);
  const glm::quat lean = s > 1e-6F ? glm::angleAxis(std::asin(std::min(s, 1.0F)), axis / s) : glm::quat(1.0F, 0.0F, 0.0F, 0.0F);
  return {.position = {c.x, groundAt(spec, c.x, c.y) - sink * float(std::max(spec.relief, 0.25)), c.y}, .rotation = lean * turn};
}

Sway swayOf(std::string_view role)
{
  // A fir's trunk is stiff and swings slowly, a grown one's slower still (a 14 m tree's first mode is
  // some 0.2 to 0.3 a second); a sapling and a bush are supple and quick. A fir's top moves some 20 cm
  // at 30 km/h, a pixel through a 28 mm lens at 1080p some 300 m off; a bush's 1.5 m top some 6 cm,
  // a pixel at 100 m
  if(role == "tree")
    return {.lean = 0.8F, .hz = 0.30F, .full = 200.0F, .end = 300.0F};
  if(role == "grown")
    return {.lean = 0.6F, .hz = 0.22F, .full = 200.0F, .end = 300.0F};
  if(role == "sapling")
    return {.lean = 2.0F, .hz = 0.6F, .full = 60.0F, .end = 100.0F};
  if(role == "shrub")
    return {.lean = 2.5F, .hz = 0.9F, .full = 60.0F, .end = 100.0F};
  return {};
}

namespace {

// A number in [0, 1) for a place on the land, the same on every machine
double placeHash(double x, double z, double salt)
{
  const double h = std::sin(x * 12.9898 + z * 78.233 + salt * 37.719) * 43758.5453;
  return h - std::floor(h);
}

}  // namespace

SwayFrame swayFrame(const Spec& spec, double seconds)
{
  SwayFrame f;
  f.on = spec.wind > 0.0;
  if(!f.on)
    return f;
  // A wind's push goes as the square of its speed: twice the wind, four times the lean, and no more
  // than four times kSwayWindKmh's however hard it blows
  f.strength        = std::min(std::pow(spec.wind / double(kSwayWindKmh), 2.0), 4.0);
  const double from = glm::radians(spec.windFrom);
  f.toward          = -glm::dvec2(std::sin(from), std::cos(from));
  f.speed           = spec.wind / 3.6;  // metres a second
  f.seconds         = seconds;
  // A turn about the level axis across the wind (up x toward) tips the top downwind; one about the
  // wind's own axis tips it across
  f.across = glm::vec3(float(f.toward.x), 0.0F, float(f.toward.y));
  f.down   = glm::vec3(f.across.z, 0.0F, -f.across.x);
  return f;
}

glm::mat4 swayed(const SwayFrame& frame, Sway sway, const glm::mat4& place, const glm::mat4& standing, glm::vec2 eye)
{
  if(!sway.any() || !frame.on)
    return standing;
  const glm::vec3 base(place[3]);
  const glm::vec2 at(base.x, base.z);
  const float     distance = glm::distance(at, eye);
  if(distance >= sway.end)
    return standing;
  const double     reach = distance <= sway.full ? 1.0 : double(sway.end - distance) / double(sway.end - sway.full);
  constexpr double kTau  = 6.283185307179586;
  const double     t     = frame.seconds;

  // The gusts: two waves along the wind, at its speed and a little under, the second shorter; a
  // plant leans between a fifth and all of its lean as they pass
  const double along = frame.toward.x * at.x + frame.toward.y * at.y;
  const double gust  = 0.6 + 0.25 * std::sin(kTau * (along - frame.speed * t) / 70.0)
                      + 0.15 * std::sin(kTau * (along - 0.8 * frame.speed * t) / 23.0 + 1.3);
  // Its own swing: a phase of its own, and a tenth either way of its role's rate
  const double phase  = kTau * placeHash(at.x, at.y, 1.0);
  const double rate   = sway.hz * (0.9 + 0.2 * placeHash(at.x, at.y, 2.0));
  const double swing  = std::sin(kTau * rate * t + phase);
  const double across = std::sin(kTau * 1.3 * rate * t + 2.0 * phase);
  const double lean   = double(sway.lean) * frame.strength * reach;
  const double down   = lean * gust * (0.6 + 0.4 * swing);  // degrees, its top downwind
  const double side   = lean * 0.15 * across;               // and a little across the wind

  // the turn about the base: the copy's axes turned, and its origin turned about the base
  const glm::mat3 turn = glm::mat3_cast(glm::angleAxis(float(glm::radians(down)), frame.down) * glm::angleAxis(float(glm::radians(side)), frame.across));
  glm::mat4       out  = standing;
  for(int c = 0; c < 3; c++)
    out[c] = glm::vec4(turn * glm::vec3(standing[c]), standing[c].w);
  out[3] = glm::vec4(base + turn * (glm::vec3(standing[3]) - base), standing[3].w);
  return out;
}

glm::mat4 swayedPlace(const Spec& spec, Sway sway, const glm::mat4& place, glm::vec2 eye, double seconds)
{
  return swayed(swayFrame(spec, seconds), sway, place, place, eye);
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
