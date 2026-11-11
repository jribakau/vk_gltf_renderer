// What a material row becomes in the path tracer's closure, measured on the CPU: the shader's
// evaluateMaterial and BSDF (shaders/gltf_material_eval.h.slang, nvshaders) compiled as C++. The
// numbers a change to the lowering or the closure shows up as, before anyone looks (the app's
// docs/material-model-plan.md, phase M1). Textures are neutral: white, and flat normal maps, so a
// map's factor is measured and not its pixels, which only the cooker reads.
//
// GLM is seen differently in the one source file (the shader code needs its swizzles), so this
// header names no GLM or shader type: a row and its texture infos pass as bytes.

#pragma once

#include <cstddef>
#include <vector>

namespace fv::probe {

// As nvshaders' LOBE_* defines
constexpr int kLobeCount = 8;
const char*   lobeName(int lobe);

struct ClosureLobe
{
  float weight = 0;              // the share of the light this lobe gets at this view
  float albedo[3] = {0, 0, 0};   // what it carries of the light it gets
};

struct ClosureAt
{
  float       cosine = 1;
  ClosureLobe lobes[kLobeCount];
  float       total[3] = {0, 0, 0};  // every lobe's weight times its albedo: the surface's albedo here
};

// An image other than the neutral ones, by the texture's place in the row's table: what a sample at a
// footprint reads, and what its last mip holds (a normal map's mean, short where it varies)
struct ProbeImage
{
  int   texture   = 0;
  float colour[4] = {1, 1, 1, 1};
  float mean[4]   = {1, 1, 1, 1};
};

struct ClosureOptions
{
  std::vector<float> cosines{1.0f, 0.7f, 0.4f, 0.15f};
  int                samples = 8192;
  // the interactive preview's closure of a medium: evaluateMaterial's scatterAsThin
  bool thinScatter = false;
  std::vector<ProbeImage> images;
  // the images' mips came with them, as the cooker's do (eTexInfoCookedMips); off, as a raw stand-in's
  bool cookedMips = true;
  // the hit's texture footprint (MeshState::texGrad): 0 samples mip 0, as the app's path tracer does
  // everywhere; more, the footprint's mip (the images' `colour`)
  float texGrad = 0;
  // where procedural layers land, a channel each (MeshState::masks)
  float masks[4] = {0, 0, 0, 0};
};

// The roughness evaluateMaterial gives each lobe: GGX alpha for the base along T and B, and the
// coat's and the second specular lobe's as roughness
struct LobeRoughness
{
  float alpha[2]  = {0, 0};
  float clearcoat = 0;
  float specular2 = 0;
};

// A material drawn on the CPU (fv_parity's {"slice": label} step), linear RGB, row 0 at the top
struct ViewImage
{
  int                width = 0, height = 0;
  std::vector<float> rgb;  // width * height * 3

  float* at(int x, int y) { return &rgb[(size_t(y) * width + x) * 3]; }
  const float* at(int x, int y) const { return &rgb[(size_t(y) * width + x) * 3]; }
};

// What a material's views show (probeViews):
// - a slice per lobe and summed, Burley's layout: across, the half vector's angle from the normal
//   (theta_h, 0 at the left to 90 degrees); down, the angle between the half vector and the light
//   (theta_d, 0 at the top to 90 degrees), with the light and the view either side of the plane of
//   the normal and the half vector (phi_d 90 degrees), so both stay above the surface. The value is the
//   BRDF, f (cosine left out), each lobe by its weight at that view: a Lambertian lobe is flat at
//   albedo / pi, a specular one is brightest at theta_h 0 and narrows with roughness. Transmission
//   lobes have nothing above the surface and are left out.
// - a ball under a white furnace (radiance 1 from every direction, the background too): each point
//   shows the surface's albedo at the angle it is seen at, so a material that keeps its energy vanishes
//   into the background and one that loses it at grazing angles shows a dark rim.
// - a ball under one distant white light (irradiance pi head-on, so a white Lambertian facing it reads
//   1), from the upper left and in front, on dark grey (kLitBackdrop), so the unlit side's edge shows.
struct MaterialViews
{
  ViewImage slice[kLobeCount];
  bool        lobeUsed[kLobeCount] = {};  // a lobe with weight somewhere in the slice
  ViewImage sliceTotal;
  ViewImage furnace;
  ViewImage lit;
};

struct ViewOptions
{
  int   sliceSize       = 256;
  int   ballSize        = 256;
  int   furnaceCosines  = 64;    // the furnace ball's albedo is sampled at this many view angles, between them linearly
  int   furnaceSamples  = 4096;  // per lobe and angle
  float light[3]        = {-0.45F, 0.55F, 0.70F};  // toward the light, the view along +z and up +y
};

std::vector<ClosureAt> probeClosureBytes(const void* row, size_t rowSize, const void* infos, size_t infoSize, size_t infoCount, const ClosureOptions& options);
LobeRoughness probeRoughnessBytes(const void* row, size_t rowSize, const void* infos, size_t infoSize, size_t infoCount, const ClosureOptions& options);
MaterialViews probeViewsBytes(const void* row, size_t rowSize, const void* infos, size_t infoSize, size_t infoCount,
                              const ClosureOptions& options, const ViewOptions& views);

template <class Row, class Info>
MaterialViews probeViews(const Row& row, const std::vector<Info>& infos, const ClosureOptions& options = {}, const ViewOptions& views = {})
{
  return probeViewsBytes(&row, sizeof(Row), infos.data(), sizeof(Info), infos.size(), options, views);
}

// The slice's place for a pixel: theta_h across, theta_d down, at the pixel's centre (radians)
inline void sliceAngles(int x, int y, int size, float& thetaH, float& thetaD)
{
  constexpr float kQuarter = 1.57079632679F;
  thetaH                   = (float(x) + 0.5F) / float(size) * kQuarter;
  thetaD                   = (float(y) + 0.5F) / float(size) * kQuarter;
}

// An image for the eye, sRGB bytes: a ball's radiance clamped to 1; a slice's f on a log scale from
// 1e-3 (violet) to 1e2 (yellow) in viridis, fixed so two materials' slices compare, black where 0
std::vector<unsigned char> ballPixels(const ViewImage& image);
std::vector<unsigned char> slicePixels(const ViewImage& image);
constexpr float            kSliceLogMin = -3.0F, kSliceLogMax = 2.0F;
constexpr float            kLitBackdrop = 0.02F;  // linear: the lit ball's background

template <class Row, class Info>
LobeRoughness probeRoughness(const Row& row, const std::vector<Info>& infos, const ClosureOptions& options = {})
{
  return probeRoughnessBytes(&row, sizeof(Row), infos.data(), sizeof(Info), infos.size(), options);
}

// `Row` is shaderio::GltfShadeMaterial and `Info` shaderio::GltfTextureInfo, with the table as
// MaterialCache::convertOne fills it: a row names a texture by its place in `infos`, 0 for none
template <class Row, class Info>
std::vector<ClosureAt> probeClosure(const Row& row, const std::vector<Info>& infos, const ClosureOptions& options = {})
{
  return probeClosureBytes(&row, sizeof(Row), infos.data(), sizeof(Info), infos.size(), options);
}

}  // namespace fv::probe
