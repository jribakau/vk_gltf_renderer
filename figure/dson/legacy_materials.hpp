#pragma once

// 3Delight-era shaders read for what their channels mean: DAZ Studio's Default shader, omnifreaker's
// UberSurface and Age of Armour's Subsurface shader. parseSurface (materials.hpp) reads what every
// shader shares, the diffuse colour and map, cutout, normal and bump maps; translateLegacy then lays
// over it what the shader's own channels say about its specular, diffuse, opacity, tiling, bump
// depth and refraction, as a physically based surface. DAZ Studio's Iray does the same for a legacy
// surface it renders (dzirayrenderer translates these shaders' parameters); its exact mapping isn't
// public.
//
// Left out on purpose, as 3Delight-era tricks rather than physics: ambient (it multiplies the
// scene's ambient light, no emitter), reflection as a coat of its own (Fresnel already reflects;
// reflection only lowers the roughness floor), velvet, a tinted specular colour (a dielectric's
// reflectance is grey), and AoA's subsurface medium (the renderer's measured skin is AoA's Skin A).
//
// Skin (a surface classify() calls skin) gets measured skin's specular whatever its file says:
// these shaders' strengths and tints were tuned against 3Delight's lights, not measured.

#include <figure/dson/channels.hpp>
#include <figure/materials/surface.hpp>

namespace fv::dson {

enum class LegacyShader
{
  eNone,         // not a legacy shader this knows (Iray Uber, PBRSkin, OmniHair, anything else)
  eDazDefault,   // DAZ Studio's Default: diffuse_strength, specular_strength, glossiness, Lighting Model
  eUberSurface,  // omUberSurface: the "... Active" switches, Specular2, Map Tiling U/V
  eAoa,          // Age of Armour's Subsurface shader: sssmaterial, cScatter_*, value23 (Subsurface Off - On)
};

// Which legacy shader the channels are, by the channels only it has. Iray's are checked first:
// Iray Uber has a Glossiness and a Diffuse Strength too.
LegacyShader classifyLegacy(const Channels& ch);

// A legacy Glossiness (how narrow the highlight is, 0..1) as a perceptual roughness. Uncalibrated:
// fitted to DAZ forum comparisons (DS Default 50% looks like UberSurface 4%, 85% like 50%) and to
// DAZ's own Iray conversion of the G8M Basic Wear Boxers (Default, glossiness 1: a rough cotton).
// No lower than a floor: a sharp 3Delight highlight came from the lights alone and never mirrored
// the scene. A surface that does reflect (a reflection, refraction or metal) is as sharp as
// 1 - glossiness says.
float legacyRoughness(LegacyShader shader, float glossiness, bool reflective);

// Lays the shader's own channels over a surface parseSurface read from the same channels.
void translateLegacy(LegacyShader shader, const Channels& ch, mat::Surface& s);

}  // namespace fv::dson
