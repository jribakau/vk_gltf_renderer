#ifndef FV_RADIANCE_CACHE_H
#define FV_RADIANCE_CACHE_H

// The moving view's bounce cache: the light a surface sends back, kept in world space from frame to
// frame, so a moving frame's path can stop at its first rough bounce's hit and read there what the rest
// of the path would have gathered. In a room that is a quarter to a third of the frame (2026-11-10, an
// RTX 4080 at 2160x1395 and twenty bounces, the camera swinging: an apartment close-up 19.1 -> 14.5 ms, a
// night club 22.9 -> 16.0 ms). Biased (a cell holds one colour for its few centimetres, whatever the
// direction), so only the frames DLSS draws read it and a still traces its paths whole; the club's
// image converged with it (ptRadianceCacheStill) came out 0.4% brighter, its colour unchanged.
//
// A hash grid of cells over the scene (Binder et al. 2018's spatial hashing; NVIDIA's SHaRC is built
// the same way): a cell is a cube whose size grows with its distance from the eye, by the side of
// the cube its surface faces. It holds the radiance leaving the surface there toward wherever paths
// came from, without the surface's own emission (the path that reads it has that already, with its MIS
// weight).
//
//  - One pixel in sixteen is an *update* path (radianceCacheUpdatePixel): at its first vertex that could
//    read the cache it reads nothing, goes on, and at its end adds to the vertex's cell what it gathered
//    from there on and the throughput it had there. Further on it reads the cache like any path, which
//    is how a cell comes to hold every later bounce and not only the path's own.
//  - A cell's radiance is its light gathered over its throughput received, channel by channel
//    (radianceCacheRadiance): the mean of what the vertex sent back along each path, each weighed by
//    what that path carries to its pixel. So the paths that read a cell get between them the light the
//    paths that fed it found, whatever their colours, and a path that arrived with next to no
//    throughput counts for next to nothing instead of being divided by it.
//  - Every other path stops at its first such vertex whose cell has received enough (RC_MIN_WEIGHT), and
//    goes on as before where it has not: an empty cache costs and shows what no cache does.
//  - After the frame a pass (radiance_cache_resolve.slang) folds each cell's sums of the frame into
//    what it keeps (radianceCacheMerge) and frees the cells no path has fed for a while.
//
// A vertex can read the cache once the path has scattered roughly (RC_PATH_ROUGHNESS), outside any
// medium, on a surface that is not a fibre and lets no light through, and far enough from the vertex
// before it for a cell's width not to show (radianceCacheFarEnough): what the camera sees, and what it
// sees in a mirror or through glass, is always traced.
//
// Slang and C++ (tests/test_radiance_cache.cpp), so the tests run the shader's arithmetic.

// The path tracer compiles the cache in only where a build asks for it (gltf_pathtrace.slang)
#ifndef USE_RADIANCE_CACHE
#define USE_RADIANCE_CACHE 0
#endif

#ifdef __cplusplus
#include <cstring>
#endif

NAMESPACE_SHADERIO_BEGIN()

// A slot's floats are kept as their bits
#ifdef __cplusplus
inline float radianceCacheFloat(uint bits)
{
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
inline uint radianceCacheBits(float f)
{
  uint bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}
#else
#define radianceCacheFloat asfloat
#define radianceCacheBits asuint
#endif

// A slot of the table, RC_WORDS uints
static const int RC_WORDS      = 14;
static const int RC_KEY        = 0;   // the cell's fingerprint (0: free)
static const int RC_AGE        = 1;   // resolves since a sample last landed in it
static const int RC_LIGHT      = 2;   // three floats: the light its paths gathered, as kept
static const int RC_WEIGHT     = 5;   // three floats: the throughput they arrived with, as kept
static const int RC_SUM_LIGHT  = 8;   // three floats: this frame's light, summed as the paths end
static const int RC_SUM_WEIGHT = 11;  // ...and this frame's throughput

// The resolve pass's workgroup (radiance_cache_resolve.slang), a thread a slot
static const int RC_RESOLVE_GROUP = 64;

// A cell lives in one of the RC_BUCKET slots its hash names: a lookup reads at most that many keys
static const int RC_BUCKET = 4;
// What a cell must have received before a path may stop at it, in throughput (its largest channel's):
// a few paths off light walls, more off dark ones, whose samples say less
static const float RC_MIN_WEIGHT = 1.0F;
// What a cell keeps at most: older light fades as new samples come (48: a second or two of a cell's own
// samples), and when something other than the camera changed, far less, so the cache follows a light
// being dragged within a few frames at the price of noisier cells meanwhile
static const float RC_MAX_WEIGHT     = 48.0F;
static const float RC_CHANGED_WEIGHT = 3.0F;
// Resolves without a sample after which a cell is freed
static const uint RC_MAX_AGE = 240u;
// The path's roughness (PathTracerState::pathRoughness) from which a vertex may read the cache
static const float RC_PATH_ROUGHNESS = 0.6F;
// Cells are `cell` wide up to this many of them from the eye, and double with each doubling of the
// distance from there, up to RC_MAX_LEVEL doublings
static const float RC_NEAR_CELLS = 64.0F;
static const int   RC_MAX_LEVEL  = 12;

// Which pixels trace an update path: one in sixteen, others each frame
inline bool radianceCacheUpdatePixel(int x, int y, int frame)
{
  return ((x + frame) & 3) == 0 && ((y + (frame >> 2)) & 3) == 0;
}

// How many times a cell `dist` from the eye has doubled, the nearest being `cell` wide
inline int radianceCacheLevel(float dist, float cell)
{
  const float ratio = dist / (cell * RC_NEAR_CELLS);
  if(!(ratio > 1.0F))
    return 0;
  const int level = int(floor(log2(ratio)));
  return level < RC_MAX_LEVEL ? level : RC_MAX_LEVEL;
}

inline float radianceCacheCellSize(int level, float cell)
{
  return cell * float(1 << level);
}

// The side of a cell's cube a surface of normal `n` faces, 0 to 5: +x, -x, +y, -y, +z, -z. The two
// sides of a wall are different cells, so what lights one never shows on the other.
inline int radianceCacheFace(float3 n)
{
  const float ax = abs(n.x);
  const float ay = abs(n.y);
  const float az = abs(n.z);
  if(ax >= ay && ax >= az)
    return n.x >= 0.0F ? 0 : 1;
  if(ay >= az)
    return n.y >= 0.0F ? 2 : 3;
  return n.z >= 0.0F ? 4 : 5;
}

// A point moved by up to half a cell along the two axes of its face, `u` uniform in [0, 1): a path
// that reads the cell there takes one of the cells around its point in proportion to how near each
// is, so over the pixels of a wall the cells' edges do not show
inline float3 radianceCacheJitter(float3 p, int face, float size, float2 u)
{
  const float a = (u.x - 0.5F) * size;
  const float b = (u.y - 0.5F) * size;
  if(face < 2)
    return p + float3(0.0F, a, b);
  if(face < 4)
    return p + float3(a, 0.0F, b);
  return p + float3(a, b, 0.0F);
}

// Whether a vertex `segment` from the one before it is far enough for its cell (`size` wide) to stand
// for it, `u` uniform in [0, 1): nearer, the cell's one colour would flatten what a corner or a contact
// shadows; the path takes one more bounce there and reads the cache at its next vertex. Between one
// and two cells the answer is drawn, so no line shows where it changes.
inline bool radianceCacheFarEnough(float segment, float size, float u)
{
  return segment >= size * (1.0F + u);
}

inline uint radianceCacheMix(uint h, uint k)
{
  k *= 0xcc9e2d51u;
  k = (k << 15) | (k >> 17);
  k *= 0x1b873593u;
  h ^= k;
  h = (h << 13) | (h >> 19);
  return h * 5u + 0xe6546b64u;
}

inline uint radianceCacheFinish(uint h)
{
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

// Where a cell is kept: the first slot of its bucket, and the fingerprint its slot holds (never 0)
struct RadianceCacheKey
{
  uint bucket;
  uint fingerprint;
};

// The key of the cell at `p` on a surface of normal `n`, seen from `eye`: `cell` the nearest cells'
// width, `slots` the table's size, a power of two. Two hashes of the cell's place, level and face: one
// names the bucket, the other tells its cells apart (two cells of a bucket sharing it, one in four
// thousand million, read each other's light).
inline RadianceCacheKey radianceCacheKey(float3 p, float3 n, float3 eye, float cell, uint slots)
{
  const float3 toEye = p - eye;
  const int    level = radianceCacheLevel(sqrt(toEye.x * toEye.x + toEye.y * toEye.y + toEye.z * toEye.z), cell);
  const float  size  = radianceCacheCellSize(level, cell);
  const uint   x     = uint(int(floor(p.x / size)));
  const uint   y     = uint(int(floor(p.y / size)));
  const uint   z     = uint(int(floor(p.z / size)));
  const uint   where = uint(level * 8 + radianceCacheFace(n));

  RadianceCacheKey key;
  key.bucket      = radianceCacheFinish(radianceCacheMix(radianceCacheMix(radianceCacheMix(radianceCacheMix(0x9747b28cu, x), y), z), where));
  key.bucket      = (key.bucket & (slots - 1u)) & ~uint(RC_BUCKET - 1);
  key.fingerprint = radianceCacheFinish(radianceCacheMix(radianceCacheMix(radianceCacheMix(radianceCacheMix(0x3c6ef372u, where), z), y), x));
  if(key.fingerprint == 0u)
    key.fingerprint = 1u;
  return key;
}

// What an update path adds to its cell: the light it gathered after the vertex, as its pixel got it
// (`gathered`: the path's radiance at its end less its radiance at the vertex), and its throughput at
// the vertex. `valid` false for a sample that is no number, or that arrived with no throughput at all.
struct RadianceCacheSample
{
  float3 light;
  float3 weight;
  bool   valid;
};

inline RadianceCacheSample radianceCacheSample(float3 gathered, float3 throughput)
{
  RadianceCacheSample s;
  s.light  = float3(max(gathered.x, 0.0F), max(gathered.y, 0.0F), max(gathered.z, 0.0F));  // rounding may leave a hair under nothing
  s.weight = float3(max(throughput.x, 0.0F), max(throughput.y, 0.0F), max(throughput.z, 0.0F));
  const float light  = s.light.x + s.light.y + s.light.z;
  const float weight = s.weight.x + s.weight.y + s.weight.z;
  s.valid = light >= 0.0F && light < 3.0e38F && weight > 0.0F && weight < 3.0e38F;
  return s;
}

// What a cell keeps: the light its paths gathered and the throughput they arrived with
struct RadianceCacheCell
{
  float3 light;
  float3 weight;
};

// How much a cell has received: its throughput's largest channel
inline float radianceCacheReceived(float3 weight)
{
  return max(weight.x, max(weight.y, weight.z));
}

inline bool radianceCacheValid(RadianceCacheCell c)
{
  return radianceCacheReceived(c.weight) >= RC_MIN_WEIGHT;
}

// The radiance a cell stands for: light over throughput, each channel its own. A channel no path
// arrived with (all of them through a pure red wall) holds nothing, and no path that reads it there
// carries any.
inline float3 radianceCacheRadiance(RadianceCacheCell c)
{
  return float3(c.weight.x > 1e-6F ? c.light.x / c.weight.x : 0.0F, c.weight.y > 1e-6F ? c.light.y / c.weight.y : 0.0F,
                c.weight.z > 1e-6F ? c.light.z / c.weight.z : 0.0F);
}

// A cell after a frame: the frame's sums added to what it kept, which is scaled down first so that the
// whole is at most `maxWeight` (RC_MAX_WEIGHT, or RC_CHANGED_WEIGHT on a frame after the scene changed).
// Scaling light and throughput alike leaves the radiance what it was; a frame without samples only cuts
// what is kept to `maxWeight`.
inline RadianceCacheCell radianceCacheMerge(RadianceCacheCell held, float3 sumLight, float3 sumWeight, float maxWeight)
{
  const float had  = radianceCacheReceived(held.weight);
  const float come = radianceCacheReceived(sumWeight);
  const float room = max(maxWeight - come, 0.0F);
  const float keep = had > room ? room / had : 1.0F;

  RadianceCacheCell c;
  c.light  = held.light * keep + sumLight;
  c.weight = held.weight * keep + sumWeight;
  return c;
}

// The slot holding `key`'s cell, or -1: a read, for the paths that stop at it
inline int radianceCacheFind(uint* table, RadianceCacheKey key)
{
  for(int i = 0; i < RC_BUCKET; i++)
  {
    const uint slot = key.bucket + uint(i);
    if(table[slot * uint(RC_WORDS) + uint(RC_KEY)] == key.fingerprint)
      return int(slot);
  }
  return -1;
}

// What a slot keeps
inline RadianceCacheCell radianceCacheRead(uint* table, int slot)
{
  const uint        base = uint(slot) * uint(RC_WORDS);
  RadianceCacheCell c;
  c.light  = float3(radianceCacheFloat(table[base + uint(RC_LIGHT)]), radianceCacheFloat(table[base + uint(RC_LIGHT) + 1u]),
                    radianceCacheFloat(table[base + uint(RC_LIGHT) + 2u]));
  c.weight = float3(radianceCacheFloat(table[base + uint(RC_WEIGHT)]), radianceCacheFloat(table[base + uint(RC_WEIGHT) + 1u]),
                    radianceCacheFloat(table[base + uint(RC_WEIGHT) + 2u]));
  return c;
}

NAMESPACE_SHADERIO_END()

#endif
