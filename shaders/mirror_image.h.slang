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

// DLSS-RR's motion of reflections (pInMotionVectorsReflections, eDlssSpecularMotion). A flat mirror
// shows a point H at H's image across its plane, which lies on the camera ray, as far beyond the
// mirror as H is from it: the point RR itself puts at the specular hit distance. Given a motion field
// for it, RR follows that instead (NVIDIA's RR guide, 3.4.8), so a reflection also moves when what it
// shows moves, or when the mirror does.

// The image of a reflection seen at `origin` along the camera ray `viewDir` (unit), `hitDist` beyond it
float3 mirrorImagePoint(float3 origin, float3 viewDir, float hitDist)
{
  return origin + viewDir * hitDist;
}

// How that image moves since the previous frame (previous position less current): the reflected
// point's motion `hitDelta` mirrored in the plane of unit `normal`, and the plane moved with the
// mirror's own motion `mirrorDelta` (an image moves twice as far as its mirror along the normal).
// A mirror that turns is taken as moving only.
float3 mirrorImageDelta(float3 hitDelta, float3 mirrorDelta, float3 normal)
{
  return hitDelta - 2.0F * normal * dot(normal, hitDelta - mirrorDelta);
}

#endif  // FV_MIRROR_IMAGE_H
