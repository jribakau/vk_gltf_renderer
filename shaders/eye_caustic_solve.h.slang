#ifndef FV_EYE_CAUSTIC_SOLVE_H
#define FV_EYE_CAUSTIC_SOLVE_H

// The walk of eye_caustic.h.slang, over the film its includer names: EYE_FILM is a type with
//
//   EyeExit leave(float3 origin, float3 w, float4 light)
//
// which follows a ray from `origin` (from the shaded point) along unit `w` to the film and out
// (eyeFilmExit), or says it does not leave (eyeNoExit). The path tracer's is the film's own mesh, met
// by a ray query (EyeFilmMesh, gltf_pathtrace.slang); the tests' is a sphere's cap. Two calls of
// `leave`, so the shader holds two copies of the query whatever the walk's length.

// The light sample at the shaded point, bent through the film: from the direction `start` (unit: the
// straight one) toward `light` (from the shaded point: a direction, w 0, or a place, w 1). `up`: the
// shaded surface's normal, unit. The walk's first ray goes along it and says whether there is a film
// over the point at all, and at least `minDepth` above it (EyeCaustic::film); it is also what is tried
// where the start finds no film (a light under the surface).
EyeCaustic eyeCausticSolve(EYE_FILM film, float3 start, float3 up, float4 light, float minDepth)
{
  EyeCaustic c      = eyeNoCaustic();
  float3     w      = up;
  bool       lifted = false;
  for(int step = 0; step < EYE_CAUSTIC_STEPS + 1; step++)
  {
    const EyeExit centre = film.leave(float3(0.0F), w, light);
    if(step == 0)
    {
      if(!centre.met || centre.distance < minDepth)
        return c;
      c.film = true;
      w      = start;
      continue;
    }
    if(!centre.met)
    {
      if(lifted)
        return c;
      lifted = true;
      w      = up;
      continue;
    }
    // Held in by total reflection, or out on the side away from the light (where the walk would
    // settle on the light's opposite, which the two axes square to the light cannot tell from it)
    if(!centre.valid || dot(centre.direction, centre.toLight) <= 0.0F)
    {
      w = centre.retry;
      continue;
    }
    const EyeFrame around  = eyeFrameToward(w, centre.normal);
    const EyeFrame toward  = eyeFrame(centre.toLight);
    const float2   off     = eyeResidual(centre, toward);
    const bool     arrived = dot(off, off) < EYE_CAUSTIC_TOLERANCE * EYE_CAUSTIC_TOLERANCE;
    // the rays beside it: two turned, and once it has arrived two moved, for the focus
    const float shift = EYE_CAUSTIC_SHIFT * centre.distance;
    const int   count = arrived ? 4 : 2;
    EyeExit     beside[4];
    for(int k = 0; k < 4; k++)
      beside[k] = eyeNoExit();
    for(int k = 0; k < count; k++)
    {
      const float3 axis   = (k & 1) == 0 ? around.a : around.b;
      const float3 origin = k < 2 ? float3(0.0F) : axis * shift;
      const float3 dir    = k < 2 ? normalize(w + axis * EYE_CAUSTIC_ANGLE) : w;
      beside[k]           = film.leave(origin, dir, light);
      if(!beside[k].valid)
        return c;
    }
    if(arrived)
    {
      c.arrival = w;
      c.exit    = centre.point;
      c.toLight = centre.toLight;
      c.normal  = centre.normal;
      c.focus   = eyeFocus(centre, beside[0], beside[1], beside[2], beside[3], toward, EYE_CAUSTIC_ANGLE, shift);
      c.valid   = true;
      return c;
    }
    const EyeTurn turn = eyeNewtonTurn(centre, beside[0], beside[1], toward, EYE_CAUSTIC_ANGLE);
    if(!turn.valid)
      return c;
    w = eyeTurned(w, around, turn.turn);
  }
  return c;
}

#endif  // FV_EYE_CAUSTIC_SOLVE_H
