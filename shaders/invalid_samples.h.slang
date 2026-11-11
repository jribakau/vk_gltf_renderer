#ifndef FV_INVALID_SAMPLES_H
#define FV_INVALID_SAMPLES_H

// What a frame's path tracer threw away as not finite (SceneFrameInfo::invalidSamples): a sample's colour
// (NaN or infinite, dropped to black so it can't stick in the accumulation), or a guide the denoisers read
// (replaced by 0 so DLSS-RR's history and the still's guides stay finite). A block of uints per frame in
// flight, zeroed before the frame traces and read back once it is done (PathTracer::prepareInvalidSamples):
// a count per kind, how many records were offered, and the first INVALID_SAMPLES_KEPT records, a pixel
// and its kind each. Slang and C++ (tests/test_invalid_samples.cpp).

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define INVALID_SAMPLE_COLOR 0u        // a sample's colour, dropped
#define INVALID_SAMPLE_DLSS_GUIDE 1u   // a DLSS frame's guide (depth, motion, normal, albedo, hit distance), replaced
#define INVALID_SAMPLE_STILL_GUIDE 2u  // the still's denoiser's averaged guide (albedo, normal, specular), replaced
#define INVALID_SAMPLE_KINDS 3u

#define INVALID_SAMPLES_KEPT 64u                                       // records kept a frame
#define INVALID_SAMPLES_OFFERED INVALID_SAMPLE_KINDS                   // the word counting records offered
#define INVALID_SAMPLES_FIRST (INVALID_SAMPLES_OFFERED + 1u)           // the first record
#define INVALID_SAMPLES_WORDS (INVALID_SAMPLES_FIRST + INVALID_SAMPLES_KEPT)  // a frame's block

// A record: the pixel's x and y in 14 bits each (images up to 16384 wide), its kind above them
inline uint invalidSampleRecord(uint x, uint y, uint kind)
{
  return (x & 0x3FFFu) | ((y & 0x3FFFu) << 14u) | (kind << 28u);
}
inline uint invalidSampleX(uint record)
{
  return record & 0x3FFFu;
}
inline uint invalidSampleY(uint record)
{
  return (record >> 14u) & 0x3FFFu;
}
inline uint invalidSampleKind(uint record)
{
  return record >> 28u;
}

NAMESPACE_SHADERIO_END()

#endif
