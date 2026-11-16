#ifndef SURFACE_OPACITY_H
#define SURFACE_OPACITY_H

// A hit's opacity for an alpha test, from its material and texture at mip 0: the path tracer's
// any-hit. Reads the including shader's pushConst.gltfScene, allImages and allSamplers.
//
// A far cut-out on a frame DLSS denoises is read a cell of its texels at a time (cutoutCellUv), so a
// leaf smaller than the pixel stays in or out from one frame to the next.

#include "woven_opacity.h.slang"
#include "cutout_cell.h.slang"

// Where on a cut-out's map (a MASK material's alpha) a ray reads it, for a ray whose footprint is
// `footprint` metres across where it meets the triangle (pathtrace_functions.h.slang's
// cutoutFootprint: the pixel's cone times SceneFrameInfo::cutoutCell; 0 on every other frame): the
// texel cutoutCellTexel picks for the cell of texels about as wide as the footprint, under a texel
// of the map where it falls (`uv`).
//
// Why: a ray a pixel reads one texel of the many a far leaf card puts in the pixel, and DLSS's
// sub-pixel jitter picks another each frame, so a leaf or a blade thinner than the pixel is in one
// frame and out the next, and the denoiser can't settle it (the foliage shimmers). A cell's one
// decision stays put while the jitter moves inside it, and the card keeps the share of it that is
// solid (one texel at random of a cell, where a mip's average under a cut-off would thin a leafy
// card away). Its edges are a cell's, which DLSS antialiases as it does a mesh's.
float2 cutoutCellUv(GltfTextureInfo texInfo, uint2 size, GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, uint3 triangleIndex,
                    float2 uv, float footprint, float3 objectDirection)
{
  float2* texcoords = vertexTexCoords(renderPrim, texInfo.texCoord);
  if(texcoords == nullptr || size.x == 0 || size.y == 0)
    return uv;
  const float2 t0 = applyTextureTransform(texInfo, texcoords[triangleIndex.x]);
  const float2 t1 = applyTextureTransform(texInfo, texcoords[triangleIndex.y]);
  const float2 t2 = applyTextureTransform(texInfo, texcoords[triangleIndex.z]);
  const float3 p0 = getVertexPosition(renderPrim, int(triangleIndex.x));
  const float3 e1 = getVertexPosition(renderPrim, int(triangleIndex.y)) - p0;
  const float3 e2 = getVertexPosition(renderPrim, int(triangleIndex.z)) - p0;
  // the footprint as the triangle has it: a slanting ray's is as wide across and 1/cos as long, the
  // side of a square of its area
  float        slant = 1.0F;
  const float3 n     = cross(e1, e2);
  if(dot(objectDirection, objectDirection) > 0.0F && dot(n, n) > 0.0F)
    slant = rsqrt(max(abs(dot(normalize(objectDirection), normalize(n))), 0.05F));
  // the map's repeats a metre on the triangle, as the instance places it
  const float density = computeTexelDensity(mul(float4(e1, 0.0F), renderNode.objectToWorld).xyz,
                                            mul(float4(e2, 0.0F), renderNode.objectToWorld).xyz, t0, t1, t2);
  const float texels  = footprint * slant * density * sqrt(float(size.x) * float(size.y));
  return cutoutCellTexel(uv * float2(size), texels) / float2(size);
}

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return 1.0 if it is opaque
// `objectDirection` is the ray's, in the instance's object space: a woven cutout's opacity depends
// on it (wovenOpacity). Zero takes the opacity head-on.
//----------------------------------------------------------
// `footprint`: the cell a far cut-out is read by (cutoutCellUv); 0, the texel the hit falls on.
float getOpacity(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int triangleID, float3 barycentrics, float3 objectDirection,
                 float footprint = 0.0F)
{
  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  if(mat.alphaMode == AlphaMode::eAlphaModeOpaque)
    return 1.0;

  uint3 triangleIndex = getTriangleIndices(renderPrim, triangleID);
  // a cut-out read a cell at a time: a MASK's, never hair's (its strands are its own texels)
  bool cells = footprint > 0.0F && mat.alphaMode == AlphaMode::eAlphaModeMask;
#if GLTF_USE_HAIR
  cells = cells && mat.hairFactor <= 0.0F;
#endif

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
      if(cells)
      {
        uint2 size;
        allImages[NonUniformResourceIndex(texInfo.index)].GetDimensions(size.x, size.y);
        uv = cutoutCellUv(texInfo, size, renderNode, renderPrim, triangleIndex, uv, footprint, objectDirection);
      }
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
      if(cells)
      {
        uint2 size;
        allImages[NonUniformResourceIndex(texInfo.index)].GetDimensions(size.x, size.y);
        uv = cutoutCellUv(texInfo, size, renderNode, renderPrim, triangleIndex, uv, footprint, objectDirection);
      }

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
