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

#endif  // FV_DLSS_SPECULAR_GUIDE_H
