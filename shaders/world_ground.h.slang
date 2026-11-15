/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// A planet's ground (src/fv_world.hpp): which of its materials a hit is drawn with, and that
// material's maps laid on it.
//
// A planet's quads are all drawn with one material: their vertices' colours on a rough surface. Where
// the backdrop's kit has ground maps the ground has eight more, a layer each (fv::world::Soil: grass,
// a wood's floor, rock, sand, snow, dry earth, stony soil, pebbles): standard materials, their colour
// maps divided by their own mean so that the ground's colour stays its vertices' and the map gives it
// its grain. A vertex says how much of each layer its place is (the first four in HitState::masks, a
// channel a layer; the other four in its texture coordinates, which the ground has no other use for,
// its maps being laid by where a hit is; what they leave is the ground's own material), and a hit is
// drawn with one of them. The shares are measured out along a noise over the ground: where rock
// gives way to grass the rock ends in tongues and patches a pace or two across, as it does, and no
// hit evaluates more than one material. (A blend would be several evaluations at every hit of the
// ground, in a kernel where a loop round that one call cost a tenth of a frame.)
//
// The maps have no coordinates of the mesh's. They are laid by where the hit is in the scene, on the
// plane of the three axes' that the ground faces most squarely: level ground from above, a cliff's
// face from its side, with the map's top up. Where two planes are as good as each other the noise
// picks again, so the seam between them is patches of each and no line. The frame the normal map is
// read in goes with the plane.
//
// A map two metres across laid end to end shows as a grid from the air. So a map is laid two ways,
// the second turned a quarter and a little larger, in patches of the noise: no two repeats side by
// side are alike for long. And from far enough that a repeat is a few pixels the ground is its own
// material again, patch by patch (by the hit's footprint): a map sampled that sparsely is noise in
// the colour the denoiser is guided by, and its mean is the colour the vertices have.
//
// Slang and plain C++ (tests/test_ground.cpp).
#ifndef WORLD_GROUND_H_SLANG
#define WORLD_GROUND_H_SLANG

// The layers, in the order of a vertex's shares (fv::world::Soil)
static const int WORLD_GROUND_LAYERS = 8;
// The order they are measured out in along the noise: rock, stony soil, pebbles, sand, dry earth,
// snow, a wood's floor, grass (those that meet in one place are neighbours here: rock and the
// stones under it, pebbles and sand at a shore)
static const int WORLD_GROUND_ORDER[8] = {2, 6, 7, 3, 5, 4, 1, 0};
// Metres across the noise's patches, and across the grain in them
static const float WORLD_GROUND_PATCH = 1.7F;
static const float WORLD_GROUND_GRAIN = 0.37F;
// How sharply one plane takes over from another as the ground turns: the planes' weights are the
// normal's components squared this many times over
static const int WORLD_GROUND_SQUARINGS = 3;
// The noise is at its ends over some of the ground (worldGroundPick), where the least share of a
// layer, or of a plane, would have the place: so a layer needs this much of a place to show at all,
// and a plane this much of the three's weight (two planes share the ground within some 8 degrees of
// where it faces both alike, and a map is never laid at more than 55 degrees to it)
static const float WORLD_GROUND_LEAST_LAYER = 0.02F;
static const float WORLD_GROUND_LEAST_PLANE = 0.1F;
// A map's second way of lying: turned a quarter, and this many repeats to the first way's one
static const float WORLD_GROUND_SECOND_SCALE = 0.77F;
// The maps give way to the ground's own material as a hit's footprint grows from this share of a
// repeat to that: from a fortieth (a repeat forty pixels across) to an eighth
static const float WORLD_GROUND_FADE_FROM = 0.025F;
static const float WORLD_GROUND_FADE_TO   = 0.125F;

// Whether a mesh's material slot is that of a planet's ground with maps to draw it with
bool worldGroundIs(SceneFrameInfo* frameInfo, int material)
{
  return frameInfo->worldGround >= 0 && material == frameInfo->worldGround;
}

uint worldGroundHash(int3 cell)
{
  uint h = (uint(cell.x) * 0x8da6b343u) ^ (uint(cell.y) * 0xd8163841u) ^ (uint(cell.z) * 0xcb1ab31fu);
  h ^= h >> 15u;
  h *= 0x2c1b3c6du;
  h ^= h >> 12u;
  h *= 0x297a2d39u;
  h ^= h >> 15u;
  return h;
}

// Four smooth noises over space, each 0 to 1, their features a unit across
float4 worldGroundNoise(float3 p)
{
  const float3 corner = floor(p);
  const int3   cell   = int3(corner);
  const float3 f      = p - corner;
  const float3 u      = f * f * (3.0F - 2.0F * f);
  float4       sum    = float4(0.0F, 0.0F, 0.0F, 0.0F);
  for(int i = 0; i < 8; i++)
  {
    const int3   o = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    const uint   h = worldGroundHash(cell + o);
    uint         k = (h ^ (h >> 16u)) * 0x7feb352du;
    k ^= k >> 15u;
    const float3 w = lerp(1.0F - u, u, float3(o));
    sum += float4(float(h & 0xffffu), float(h >> 16u), float(k & 0xffffu), float(k >> 16u)) * (w.x * w.y * w.z);
  }
  return sum / 65535.0F;
}

// Four numbers at a place, each 0 to 1: what its layer, its plane, its map's way of lying and its
// fading out are picked by. Patches with a grain in them, spread about a half until they are about
// as often anywhere from 0 to 1, so that a tenth of rock has its spots.
float4 worldGroundPick(float3 pos)
{
  const float4 n = 0.65F * worldGroundNoise(pos / WORLD_GROUND_PATCH) + 0.35F * worldGroundNoise(pos / WORLD_GROUND_GRAIN + 17.3F);
  return clamp((n - 0.5F) * 2.2F + 0.5F, float4(0.0F, 0.0F, 0.0F, 0.0F), float4(1.0F, 1.0F, 1.0F, 1.0F));
}

// The slot of a layer's material (-1: none), and how many repeats of its maps a metre has
int worldGroundSlot(SceneFrameInfo* frameInfo, int layer)
{
  return layer < 4 ? frameInfo->worldGroundLayer[layer] : frameInfo->worldGroundLayerMore[layer - 4];
}
float worldGroundRepeats(SceneFrameInfo* frameInfo, int layer)
{
  return layer < 4 ? frameInfo->worldGroundTile[layer] : frameInfo->worldGroundTileMore[layer - 4];
}

// The layer a place of `shares` (the first four layers') and `more` (the other four's) is drawn with
// where the noise is `pick`; -1: the ground's own material
int worldGroundLayer(float4 shares, float4 more, float pick)
{
  const float total = shares.x + shares.y + shares.z + shares.w + more.x + more.y + more.z + more.w;
  // (a vertex's first shares are bytes: all but a hundredth of a place is the whole of it)
  const float at    = lerp(WORLD_GROUND_LEAST_LAYER, 1.0F - WORLD_GROUND_LEAST_LAYER, pick) * (total > 0.97F ? total : 1.0F);
  int         layer = -1;
  float       upTo  = 0.0F;
  for(int i = 0; i < WORLD_GROUND_LAYERS; i++)
  {
    const int l = WORLD_GROUND_ORDER[i];
    upTo += l < 4 ? shares[l] : more[l - 4];
    if(layer < 0 && at < upTo)
      layer = l;
  }
  return layer;
}

// The axis whose plane a place facing `normal` is drawn on where the noise is `pick`: 0 x, 1 y, 2 z
int worldGroundPlane(float3 normal, float pick)
{
  float3 w = abs(normal);
  for(int i = 0; i < WORLD_GROUND_SQUARINGS; i++)
    w = w * w;
  const float at = lerp(WORLD_GROUND_LEAST_PLANE, 1.0F - WORLD_GROUND_LEAST_PLANE, pick) * (w.x + w.y + w.z);
  return at < w.y ? 1 : (at < w.y + w.x ? 0 : 2);
}

// Whether a hit `footprint` metres across is too far for a map of `repeats` repeats a metre, where
// the noise is `pick`: nowhere under WORLD_GROUND_FADE_FROM of a repeat, everywhere past _FADE_TO
bool worldGroundFaded(float footprint, float repeats, float pick)
{
  const float share = footprint * repeats;
  return share > lerp(WORLD_GROUND_FADE_FROM, WORLD_GROUND_FADE_TO, pick);
}

struct WorldGroundHit
{
  bool   mapped;        // false: the ground's own material, as its mesh has it, and nothing below is set
  int    material;      // the slot of the layer's material
  float2 uv;            // where in its maps
  float3 tangent;       // the frame its normal map is read in: along u,
  float3 bitangent;     // and up the map
  float  texelDensity;  // repeats of its maps a metre
};

// What a hit of a planet's ground at `pos`, world space, is drawn with: `normal` its shading normal,
// `shares` and `more` its vertices' (HitState::masks, and its two texture coordinates), `footprint`
// the metres across what the hit stands for (0: as from near)
WorldGroundHit worldGroundAt(SceneFrameInfo* frameInfo, float3 pos, float3 normal, float4 shares, float4 more, float footprint)
{
  WorldGroundHit g;
  g.mapped       = false;
  g.material     = frameInfo->worldGround;
  g.uv           = float2(0.0F, 0.0F);
  g.tangent      = float3(1.0F, 0.0F, 0.0F);
  g.bitangent    = float3(0.0F, 0.0F, 1.0F);
  g.texelDensity = 0.0F;

  const float4 pick  = worldGroundPick(pos);
  const int    layer = worldGroundLayer(shares, more, pick.x);
  if(layer < 0 || worldGroundSlot(frameInfo, layer) < 0 || worldGroundFaded(footprint, worldGroundRepeats(frameInfo, layer), pick.w))
    return g;

  // the plane's own directions: along its maps' u, and along their v (down the map, so that a
  // cliff's strata lie as the map has them)
  const int plane = worldGroundPlane(normal, pick.y);
  float3    alongU = float3(1.0F, 0.0F, 0.0F);
  float3    alongV = float3(0.0F, -1.0F, 0.0F);
  if(plane == 1)
    alongV = float3(0.0F, 0.0F, 1.0F);
  else if(plane == 0)
    alongU = float3(0.0F, 0.0F, 1.0F);
  // its second way of lying; turned on level ground only, as a face's map keeps its top up
  g.texelDensity = worldGroundRepeats(frameInfo, layer);
  if(pick.z > 0.5F)
  {
    g.texelDensity *= WORLD_GROUND_SECOND_SCALE;
    if(plane == 1)
    {
      const float3 was = alongU;
      alongU           = alongV;
      alongV           = -was;
    }
  }

  g.mapped   = true;
  g.material = worldGroundSlot(frameInfo, layer);
  g.uv       = float2(dot(pos, alongU), dot(pos, alongV)) * g.texelDensity;
  // (the plane is one the normal all but faces, WORLD_GROUND_LEAST_PLANE: its u is never along the normal)
  g.tangent   = normalize(alongU - normal * dot(normal, alongU));
  g.bitangent = cross(normal, g.tangent);
  if(dot(g.bitangent, alongV) > 0.0F)
    g.bitangent = -g.bitangent;
  return g;
}

#endif
