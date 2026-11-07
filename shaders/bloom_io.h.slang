//
// Host/device interface of bloom (src/bloom.hpp, shaders/bloom.slang): a glow and a star on the image
// the tone mapper takes.
//

#ifndef BLOOM_IO_H
#define BLOOM_IO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define BLOOM_WORKGROUP_SIZE 16

// Push descriptors, in VK_IMAGE_LAYOUT_GENERAL, the samplers clamping to the edge
enum BloomBindings
{
  eBloomA   = 0,  // Sampler2D: the pass's source (the image, a finer level, the glow)
  eBloomB   = 1,  // Sampler2D: a second source (the coarser level, the star)
  eBloomC   = 2,  // Sampler2D: the composite's image
  eBloomOut = 3,  // RWTexture2D: what the pass writes (RGBA16F)
};

struct BloomPushConstant
{
  int2   outSize;  // the destination's size: the dispatch
  float2 aTexel;   // one texel of A, in its uv
  float2 bTexel;   // one texel of B, in its uv
  float  scale;    // bloomDown on the image: its exposure (the pyramid is in the tone mapper's white); bloomComposite: 1 over it
  float  limit;    // bloomDown on the image: a tap's largest channel at most this (BLOOM_LIMIT); 0: none
  float  tail;     // bloomUp: the coarser level's share (BLOOM_TAIL)
  float  share;    // bloomComposite: the share of the image's light the glow takes
  float  star;     // bloomComposite: the share of the glow the star takes (0 without blades)
  int    blades;   // bloomStar: the aperture's blades
  float  rotation; // bloomStar: its first corner, radians, from the image's right toward its top
  float  starStep; // bloomStar: a tap's spacing, in A's texels
};

NAMESPACE_SHADERIO_END()

#endif  // BLOOM_IO_H
