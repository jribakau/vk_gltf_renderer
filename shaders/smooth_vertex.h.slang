#ifndef FV_SMOOTH_VERTEX_H
#define FV_SMOOTH_VERTEX_H

// Where a path's vertex leaves the lights to its BSDF ray (pathTraceOneBounce): a vertex whose every
// weighted lobe is a mirror, as clear glass, an eye's tear film and polished metal are. Its light
// sample's BSDF value and MIS weight are next to nothing there, while its shadow ray, often through
// lashes and hair, costs as much as anywhere. So it takes none, and the BSDF ray that reaches a light
// weighs 1 (LightPick::bsdfOnly): the image is the same in expectation (Falcor's handleHit samples no
// light where a hit has no non-delta lobe). Slang and C++ (tests/test_bsdf_lobes.cpp), so the tests run
// the rule the shader runs. After PbrMaterial and DIRAC (nvshaders).

// GGX alpha. Just above the tear film's cap (perceptual 0.02, the figure lowering's, alpha 4e-4) so a
// film at the cap passes whatever its square rounds to; roughness 0 is MICROFACET_MIN_ROUGHNESS's 2e-6.
// No higher: a lobe this narrow is still ten times narrower than the sun's disc, and the BSDF ray finds
// a light that wide better than its light sample does. At CLEAR_CROSSING_ROUGHNESS (alpha 0.01) the
// lobe is wider than the sun or a lamp's bulb a few metres off, which the light sample finds better:
// a polished highlight would get noisier there.
static const float SMOOTH_VERTEX_ALPHA = 5e-4F;

// Whether every lobe of `mat` that carries weight (computeLobeWeights) is a GGX lobe of alpha at most
// SMOOTH_VERTEX_ALPHA, and none is diffuse. The material after the path's regularisation, which
// roughens smooth metal on DLSS frames. The specular, metal and transmission lobes share
// `roughness`; the diffuse lobes have weight only below what metal or transmission leave (none at
// either's 1); sheen (Charlie's lobe) and retroreflection are never mirrors here.
bool smoothVertex(PbrMaterial mat)
{
  if(max(mat.roughness.x, mat.roughness.y) > SMOOTH_VERTEX_ALPHA)
    return false;
  if(mat.metallic < 1.0F && mat.transmission < 1.0F)
    return false;
  if(mat.clearcoat > 0.0F && mat.clearcoatRoughness * mat.clearcoatRoughness > SMOOTH_VERTEX_ALPHA)
    return false;
  if(mat.specular2 > 0.0F && mat.specular2Roughness * mat.specular2Roughness > SMOOTH_VERTEX_ALPHA)
    return false;
  if(mat.sheenColor.x > 0.0F || mat.sheenColor.y > 0.0F || mat.sheenColor.z > 0.0F)
    return false;
  return !(mat.retroreflection > 0.0F);
}

// Whether a smooth vertex (smoothVertex) keeps a light sample toward `toLight` of pdf `lightPdf`, to
// evaluate it and weigh it by smoothVertexLightWeight. A punctual light's (DIRAC): no BSDF ray reaches
// it, so a point's catchlight comes only from its sample. And one where a lobe may scatter that the
// BSDF ray never leaves by (bsdfFaceAllows): a reflection on or past the face (a lobe whose view is
// behind a bump bends its normal back toward it, so it can reflect below the shading normal too), a
// refraction above it. Every other sample is the BSDF ray's: dropped, nothing evaluated or traced.
// Within the light samples' own hemisphere (pathTraceOneBounce's bounce.nextEventValid) the kept ones
// lie between the interpolated normal's plane and the face's.
bool smoothVertexSamplesLight(PbrMaterial mat, float3 toLight, float lightPdf)
{
  if(lightPdf == DIRAC)
    return true;
  if(!mat.faceSided)
    return false;
  if(dot(toLight, mat.Ng) > 0.0F)
    return dot(toLight, mat.N) < 0.0F && mat.transmission > 0.0F;  // the transmission lobe's side (bsdfLobeTransmits)
  return true;
}

// The MIS weight of a kept light sample (smoothVertexSamplesLight) toward which the BSDF, evaluated,
// gives pdf `bsdfPdf`: 1 for a punctual light and where the BSDF ray cannot go (`bsdfPdf` 0, the lobe's
// value there next-event estimation's alone), 0 where it can, since its hit there weighs 1
// (bsdfHitMisWeight). So each direction is counted once, by one technique.
float smoothVertexLightWeight(float lightPdf, float bsdfPdf)
{
  return (lightPdf == DIRAC || !(bsdfPdf > 0.0F)) ? 1.0F : 0.0F;
}

#endif  // FV_SMOOTH_VERTEX_H
