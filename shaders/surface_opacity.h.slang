#ifndef SURFACE_OPACITY_H
#define SURFACE_OPACITY_H

// A hit's opacity for an alpha test, from its material and texture at mip 0. Shared by the path
// tracer's any-hit and the raster's ray-traced shadows (gltf_raster.slang). Reads the including
// shader's pushConst.gltfScene, allImages and allSamplers.

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return 1.0 if it is opaque
//----------------------------------------------------------
float getOpacity(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int triangleID, float3 barycentrics)
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

  return baseColorAlpha;
}

#endif
