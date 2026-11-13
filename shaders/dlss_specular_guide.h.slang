#ifndef FV_DLSS_SPECULAR_GUIDE_H
#define FV_DLSS_SPECULAR_GUIDE_H

// What a metal's edge tint (OpenPBR's F82, f82Fresnel) leaves of its GGX lobe's directional albedo,
// against the Schlick lobe's, for DLSS-RR's specular albedo guide (dlssSpecularAlbedo), whose
// EnvBRDFApprox2 fit is Schlick's. Slang and C++ (tests/test_bsdf_lobes.cpp), so the tests run the
// rule the shader runs. After nvshaders' bsdf_functions.h.slang (f82Fresnel, f82AverageShare).
//
// A smooth lobe's half vectors are the view's normal, so its share is the Fresnel's own at the view;
// a rough lobe's spread over the hemisphere, so its share is the average one (f82AverageShare, which
// the metal's multiple scattering takes too). Between them a blend linear in alpha, complete at
// DLSS_EDGE_TINT_ROUGH_ALPHA. The view's cosine is floored at DLSS_EDGE_TINT_VIEW_FLOOR sqrt(alpha):
// toward grazing, a lobe's visible normals lean toward the view, so its half vectors stay off grazing
// where the Fresnel's own ratio would climb back to 1. Against the lobe's albedo integrated over its
// visible normals (utils/f82_guide_share.py) for chrome, iron, titanium, aluminium, copper and brass,
// from mirror to alpha 1 and head-on to 88 degrees: within 5% at worst. Schlick's alone (a share of
// 1) is off by up to 35% on a smooth chrome rim, the average alone by up to 29%.
static const float DLSS_EDGE_TINT_ROUGH_ALPHA = 0.5F;
static const float DLSS_EDGE_TINT_VIEW_FLOOR  = 0.4F;

// The share for a metal of head-on colour `f0` and edge tint `edgeTint`, at GGX alpha `alpha`, seen
// at `NoV` (either side). 1 for a white tint.
float3 dlssEdgeTintShare(float3 f0, float3 edgeTint, float alpha, float NoV)
{
  const float  a       = max(alpha, 0.0F);
  const float  mu      = min(max(abs(NoV), DLSS_EDGE_TINT_VIEW_FLOOR * sqrt(a)), 1.0F);
  const float3 schlick = schlickFresnel(f0, float3(1.0F), mu);
  const float3 view    = f82Fresnel(f0, edgeTint, mu) / max(schlick, float3(1e-6F));
  const float  k       = min(a / DLSS_EDGE_TINT_ROUGH_ALPHA, 1.0F);
  return lerp(view, f82AverageShare(f0, edgeTint), float3(k));
}

// The see-through guides (primary surface replacement, gltf_pathtrace.slang): where a camera ray meets
// clear glass or the water's surface, DLSS-RR is told about the surface behind it, as the pixel holds
// it. `fresnel` is what the glass or water reflects, `through` what it and every pane, solid glass and
// water after it let through to that surface (dlssTraceClearGlass: each crossing's colour, a volume's
// and the water's absorption).
//
// The reflection's share of what reaches the eye, for a surface behind of albedo 1:
// F / (F + (1 - F) luminance(through)). Past DLSS_REFLECTION_KEEPS_SURFACE the guides keep the
// reflecting surface's own normal, depth and motion, which the reflection moves with (NVIDIA's RR guide,
// 3.4.1: prefer the reflection at the primary hit): water from a low camera, a window edge-on, a pane
// too dark to see through.
static const float DLSS_REFLECTION_KEEPS_SURFACE = 0.5F;

float dlssReflectionShare(float fresnel, float3 through)
{
  const float t = (1.0F - fresnel) * (through.x * 0.2126F + through.y * 0.7152F + through.z * 0.0722F);
  return fresnel / max(fresnel + t, 1e-6F);
}

// What a dielectric reflects head-on between indices `n1` and `n2`, ((n2 - n1) / (n2 + n1))^2, the
// same from either side. Squared by a product: from inside (glass, under water) the ratio is negative,
// and pow of a negative base is undefined on the GPU (pbr_ggx_microfacet.h.slang's schlickFresnel).
float dlssDielectricF0(float n1, float n2)
{
  const float r = (n2 - n1) / (n2 + n1);
  return r * r;
}

// The diffuse albedo guide: the glass's or water's own colour where it reflects, the surface behind
// as it comes through elsewhere
float3 dlssSeeThroughAlbedo(float3 behind, float3 surfaceColour, float fresnel, float3 through)
{
  return lerp(through * behind, surfaceColour, float3(fresnel));
}

// The water's absorption over `distance` under its surface (ground.h.slang's GROUND_WATER_ABSORPTION,
// per channel, a metre at 0.75 / 0.95 / 0.99)
float3 dlssWaterThrough(float3 absorption, float distance)
{
  return exp(-absorption * max(distance, 0.0F));
}

// Under water and through solid glass the eye sees a surface nearer than it is. Through flat surfaces,
// a point s down a ray refracted from index n0 into n is seen s n0 / n along the camera ray: the image
// that holds as the eye moves across, on the line through the point along the surfaces' normal (its
// sideways offset is the point's; Falcor's getRefractionXform squeezes depth by the same ratio).
// DLSS-RR's depth and motion for the surface behind (gltf_pathtrace.slang's psrPos) are placed there,
// so the floor and a figure's legs under the water move with the parallax the eye sees. `scale` is
// n0 / n so far: each refraction from `iorFrom` into `iorTo` takes it on. Thin glass doesn't refract,
// so it doesn't move it.
float dlssSeenScale(float scale, float iorFrom, float iorTo)
{
  return scale * iorFrom / max(iorTo, 1e-6F);
}

// Under one flat surface that image is the point with its depth below the surface times this, the
// camera ray's cosine to the normal above it `cosI`, below it `cosT`, at index ratio `eta`: how far
// the eye sees something in the water move when it moves along the normal. 1 / eta head-on.
float dlssSeenDepthRatio(float cosI, float cosT, float eta)
{
  return cosI / max(eta * cosT, 1e-6F);
}

#endif  // FV_DLSS_SPECULAR_GUIDE_H
