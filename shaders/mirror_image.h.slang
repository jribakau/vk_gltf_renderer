#ifndef FV_MIRROR_IMAGE_H
#define FV_MIRROR_IMAGE_H

// Where a curved mirror shows what it reflects: DLSS-RR's specular hit distance on an eye. RR moves
// a pixel's reflection as if it lay that far beyond the surface its depth names, so told that a
// cornea's reflection reached a wall 3 m away (or the sky), it slid the catchlight across the eye
// while the camera moved, and on the samples that went through to the iris it was told nothing. A
// convex mirror of radius R shows a distant light at its focal point, R/2 behind its surface: a
// point fixed in the world (the centre, then R/2 towards the light), so while the camera moves a
// catchlight goes with the eye and barely across it. That is the distance RR is given, on every
// sample alike and with no ray.
// Slang and C++ (tests/test_mirror_image.cpp), so the tests run the arithmetic the shader runs.

// The radius of curvature of a smooth mesh across one triangle, from how its vertex normals turn
// along its edges: on a sphere of radius R a vertex's normal is (p - c) / R, so an edge is R times
// the difference of its ends' normals. `edge1`, `edge2`: the triangle's edges from vertex 0 in world
// space; `turn1`, `turn2`: the differences of the unit world normals along them (n1 - n0, n2 - n0).
// 0 where the surface is flat or bends away from its normals (concave, seen from its front), which
// is no mirror image in front of the surface's far side.
float meshMirrorRadius(float3 edge1, float3 edge2, float3 turn1, float3 turn2)
{
  const float lengths = dot(edge1, edge1) + dot(edge2, edge2);
  const float turn    = dot(turn1, turn1) + dot(turn2, turn2);
  // Normals that turn by under a ten-thousandth of a radian across it: flat, as far as a reflection
  // can tell, and the traced distance stands
  if(turn <= 1.0e-8F || dot(edge1, turn1) + dot(edge2, turn2) <= 0.0F)
    return 0.0F;
  return sqrt(lengths / turn);
}

// The specular hit distance RR is given for a convex mirror of radius `radius`: its focal point,
// R/2 behind the surface, counted from the depth RR has at that pixel. Where primary surface
// replacement put the surface seen through the mirror (the iris, about 3 mm into an eye whose
// cornea is about 7.8 mm round), that depth is `replacedDepth` along the camera ray; on the mirror
// itself, 0. The image is never put in front of that depth, and never at it: a pixel with no
// reflection is given 0, so a twentieth of the image distance is the least.
float mirrorImageDistance(float radius, float replacedDepth)
{
  const float image = 0.5F * radius;
  return max(image - replacedDepth, 0.05F * image);
}

#endif  // FV_MIRROR_IMAGE_H
