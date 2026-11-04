#ifndef SURFACE_OPACITY_H
#define SURFACE_OPACITY_H

// A hit's opacity for an alpha test, from its material and texture at mip 0: the path tracer's
// any-hit. Reads the including shader's pushConst.gltfScene, allImages and allSamplers.

#include "woven_opacity.h.slang"

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return 1.0 if it is opaque
// `objectDirection` is the ray's, in the instance's object space: a woven cutout's opacity depends
// on it (wovenOpacity). Zero takes the opacity head-on.
//----------------------------------------------------------
float getOpacity(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int triangleID, float3 barycentrics, float3 objectDirection)
{
  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  if(mat.alphaMode == AlphaMode::eAlphaModeOpaque)
    return 1.0;

  uint3 triangleIndex = getTriangleIndices(renderPrim, triangleID);

  float baseColorAlpha = 1;
#if GLTF_USE_SPECULAR_GLOSSINESS
  if(mat.pbrModel == PbrModel::ePbrModelSpecularGlossiness)
  {
    baseColorAlpha = mat.pbrDiffuseFactor.a;
    if(isTexturePresent(mat.pbrDiffuseTexture))
    {
      GltfTextureInfo texInfo = texInfos[mat.pbrDiffuseTexture];
      float2          uv = getInterpolatedVertexTexCoord(renderPrim, texInfo.texCoord, triangleIndex, barycentrics);
      uv                 = applyTextureTransform(texInfo, uv);
      baseColorAlpha *= allImages[NonUniformResourceIndex(texInfo.index)]
                            .SampleLevel(allSamplers[NonUniformResourceIndex(texInfo.samplerIndex)], uv, 0.0f)
                            .a;
    }
  }
  else
#endif
  {
    baseColorAlpha = mat.pbrBaseColorFactor.a;
    if(isTexturePresent(mat.pbrBaseColorTexture))
    {
      GltfTextureInfo texInfo = texInfos[mat.pbrBaseColorTexture];
      float2          uv = getInterpolatedVertexTexCoord(renderPrim, texInfo.texCoord, triangleIndex, barycentrics);
      uv                 = applyTextureTransform(texInfo, uv);

      baseColorAlpha *= allImages[NonUniformResourceIndex(texInfo.index)]
                            .SampleLevel(allSamplers[NonUniformResourceIndex(texInfo.samplerIndex)], uv, 0.0f)
                            .a;
    }
  }

  baseColorAlpha *= getInterpolatedVertexColor(renderPrim, triangleIndex, barycentrics).a;

#if GLTF_USE_HAIR
  // Ribbon roots fade in along the strand. Opacity 1 (the rest of the ribbon) is always kept.
  if(mat.hairFactor > 0.0 && mat.hairRootFade > 0.0 && mat.alphaMode != AlphaMode::eAlphaModeOpaque)
  {
    float2 uv    = getInterpolatedVertexTexCoord(renderPrim, 0, triangleIndex, barycentrics);
    float  along = abs(uv.y - mat.hairRootV);
    baseColorAlpha *= smoothstep(0.0, mat.hairRootFade, along);
  }
#endif

  if(mat.alphaMode == AlphaMode::eAlphaModeMask)
  {
    return baseColorAlpha >= mat.alphaCutoff ? 1.0 : 0.0;
  }

  // A weave: the triangle's own normal, as the threads lie in it, and the warp along its UV tangent
  if(mat.opacityWoven != 0 && baseColorAlpha > 0.0 && baseColorAlpha < 1.0 && dot(objectDirection, objectDirection) > 0.0)
  {
    const float3 p0 = getVertexPosition(renderPrim, int(triangleIndex.x));
    const float3 n  = cross(getVertexPosition(renderPrim, int(triangleIndex.y)) - p0, getVertexPosition(renderPrim, int(triangleIndex.z)) - p0);
    if(dot(n, n) > 0.0)
      baseColorAlpha = wovenOpacity(baseColorAlpha, objectDirection, n, getInterpolatedVertexTangent(renderPrim, triangleIndex, barycentrics).xyz);
  }

  return baseColorAlpha;
}

#endif
