// tinygltf's out-of-line half for fv_parity, which the figure library leaves to whoever links it
// (the renderer and the tests compile src/tiny_stb_implementation.cpp). The retained scene's sources
// the feed step links compare glTF materials, which needs it. No images: the build fv_parity times
// has no textures, and export_glb.cpp keeps stb_image to itself.

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#pragma warning(push)
#pragma warning(disable : 4018)  // signed/unsigned mismatch
#pragma warning(disable : 4267)  // conversion from 'size_t' to 'uint32_t', possible loss of data
#pragma warning(disable : 4996)  // deprecated CRT functions
#include <tinygltf/tiny_gltf.h>
#pragma warning(pop)
