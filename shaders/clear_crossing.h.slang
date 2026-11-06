#ifndef FV_CLEAR_CROSSING_H
#define FV_CLEAR_CROSSING_H

// Which refracting surfaces shadow rays pass straight through, and which a path's vertex samples the
// lights behind instead. Slang and C++ (tests/figure/test_materials.cpp holds the figure lowering's
// cap to it), so the tests run the rule the shader runs.

// A clear surface this smooth (perceptual roughness: the cap the figure's lowering puts on refracting
// surfaces, fv::mat::kRefractingRoughnessCap) passes shadow rays straight through
// (getShadowTransmission), and a path's ray that refracts through it is a null event for MIS
// (pathTraceOneBounce). A rougher one blocks shadow rays, and a path's vertex on it, if thin, samples
// the lights behind it through its transmission. Both sides read the material's own roughness as
// crossingRoughnessMetallic does, never the regularised one. Perceptual, not GGX alpha: at the cap,
// 0.1f squared rounds to just above 0.01f, so every frosted glass at the cap blocked shadow rays and
// the emitter inside a bulb was left to the BSDF rays that happened on it.
static const float CLEAR_CROSSING_ROUGHNESS = 0.1F;

// Whether a refracting surface of perceptual `roughness` is a clear crossing
bool isClearCrossing(float roughness)
{
  return roughness <= CLEAR_CROSSING_ROUGHNESS;
}

// Whether a path's vertex on a refracting surface samples the lights behind it through its specular
// transmission (both sides): where shadow rays stop at it, as they do at rough glass, a thin surface
// with nothing scattering in it. A medium's wall refracts into a medium the light sample does not
// cross, and thin scatter's diffuse lobe already sees both sides. `transmissive`: its transmission is
// above MIN_TRANSMISSION.
bool samplesLightsThroughGlass(bool transmissive, float thickness, float roughness, float scatterStrength)
{
  return transmissive && thickness <= 0.0F && !isClearCrossing(roughness) && scatterStrength <= 0.0F;
}

#endif  // FV_CLEAR_CROSSING_H
