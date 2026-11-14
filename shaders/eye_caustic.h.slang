#ifndef FV_EYE_CAUSTIC_H
#define FV_EYE_CAUSTIC_H

// The light an eye's tear film focuses on what lies behind it (FV_materials_eye): the cornea's caustic,
// the bright crescent a key from the side throws on the far side of the iris.
//
// A light sample at a point in the chamber used to go straight through the film, as through thin glass.
// It is bent here instead, as the camera's own ray already is on its way in (rayIntoEyeChamber): the
// ray that leaves the shaded point, crosses the film and goes on to the light is found on the film's
// own mesh (Hanika, Droske and Fascione 2015, "Manifold Next Event Estimation", for one refracting
// surface; Cycles' kernel/integrator/mnee.h does the same over several), and the light it carries is
// scaled by how much the film concentrates it there.
//
// The walk is Newton's method on the ray's direction at the shaded point: the direction it leaves the
// film with is compared with the light's, and the two rays a small turn away say how a turn moves it.
//
// The focus is the beam's own bookkeeping, not radiance's: a parallel beam's cross-section, square to
// the light, over the patch of it that lands round the shaded point, square to its arrival. So a flat
// film gives exactly what the straight sample gave (the cosine of the light on the film's plane), a
// lit iris stays as bright as its map says head-on, and only the film's curve moves light about. As
// the film stays thin-walled for the camera (no n^2 on the way in or out), it does here too.
//
// Slang and C++ (tests/test_eye_caustic.cpp), so the tests run the arithmetic the shader runs. The walk
// itself is eye_caustic_solve.h.slang, over a film its includer defines.

// The most the film brightens a light sample. On the caustic's edge the beam folds and the ratio has
// no bound; the sun's own width blurs that in the eye, and a cap keeps a pixel on the fold finite.
static const float EYE_CAUSTIC_MAX = 8.0F;
// The turn, radians, between the rays the walk compares: wide enough to span a triangle or two of the
// film (its normals are interpolated, so their slope is constant inside one)
static const float EYE_CAUSTIC_ANGLE = 0.05F;
// How far the shaded point is moved for the focus, as a share of its distance to the film
static const float EYE_CAUSTIC_SHIFT = 0.05F;
// The walk has arrived when the ray leaves the film within this of the light's direction (the sine of
// the angle between them: a tenth of a degree, a fifth of the sun's width)
static const float EYE_CAUSTIC_TOLERANCE = 2.0e-3F;
static const int   EYE_CAUSTIC_STEPS     = 8;
// The largest turn one step takes, radians: a step past it comes from a slope read across a fold
static const float EYE_CAUSTIC_MAX_TURN = 0.5F;

// Two unit vectors square to unit `n` and to each other
struct EyeFrame
{
  float3 a;
  float3 b;
};

EyeFrame eyeFrame(float3 n)
{
  EyeFrame     f;
  const float3 up = abs(n.y) < 0.9F ? float3(0.0F, 1.0F, 0.0F) : float3(1.0F, 0.0F, 0.0F);
  f.a             = normalize(cross(up, n));
  f.b             = cross(n, f.a);
  return f;
}

// Two unit vectors square to unit `w` and to each other that both lean toward unit `n`: a ray along
// `w` turned along either meets a surface of normal `n` less aslant than `w` does. So the rays the
// walk compares its own with stay clear of total reflection where that one is.
EyeFrame eyeFrameToward(float3 w, float3 n)
{
  const float3 lean  = n - w * dot(n, w);
  const float  size2 = dot(lean, lean);
  if(size2 < 1.0e-6F)
    return eyeFrame(w);
  EyeFrame     f;
  const float3 g = lean / sqrt(size2);
  const float3 h = cross(w, g);
  f.a            = (g + h) * 0.70710678F;
  f.b            = (g - h) * 0.70710678F;
  return f;
}

// A ray of the chamber where it leaves through the film
struct EyeExit
{
  float3 point;      // where, from the shaded point
  float3 direction;  // its direction in the air, unit
  float3 toLight;    // the light's direction from there, unit
  float3 normal;     // the film's there, unit, facing out of the chamber
  float3 retry;      // another way to try from the shaded point, where this one is no use (eyeFilmExit)
  float  distance;   // from the ray's origin to the film
  bool   met;        // it met the film from inside
  bool   valid;      // and left through it (no total reflection)
};

EyeExit eyeNoExit()
{
  EyeExit e;
  e.point     = float3(0.0F);
  e.direction = float3(0.0F, 0.0F, 1.0F);
  e.toLight   = float3(0.0F, 0.0F, 1.0F);
  e.normal    = float3(0.0F, 0.0F, 1.0F);
  e.retry     = float3(0.0F, 0.0F, 1.0F);
  e.distance  = 0.0F;
  e.met       = false;
  e.valid     = false;
  return e;
}

// A ray from `origin` (from the shaded point) along unit `w` that met the film `distance` on, at a
// point of unit `normal` (facing either way), out of a chamber of index `ior` into air. `light`: the
// light from the shaded point, its direction (w 0, unit) or its place (w 1).
EyeExit eyeFilmExit(float3 origin, float3 w, float distance, float3 normal, float ior, float4 light)
{
  EyeExit      e = eyeNoExit();
  const float3 n = dot(normal, w) > 0.0F ? -normal : normal;  // against the ray
  const float3 d = refract(w, n, ior);
  e.point            = origin + w * distance;
  const float3 place = float3(light.x, light.y, light.z);
  e.toLight          = light.w > 0.5F ? normalize(place - e.point) : place;
  e.normal           = -n;
  e.distance         = distance;
  e.met              = true;
  e.valid            = dot(d, d) > 0.5F;  // else past the critical angle: it stays in the eye
  if(e.valid)
    e.direction = normalize(d);
  // Where the ray is no use (held in, or out the far way from the light), the light's own ray through
  // that point of the film, bent into the chamber, is a better way from the shaded point; where the
  // film there faces away from the light, a lean toward the light, whose side of the film faces it.
  if(dot(e.toLight, e.normal) > 0.05F)
    e.retry = -normalize(refract(-e.toLight, e.normal, 1.0F / ior));
  else
    e.retry = normalize(w + e.toLight * 0.3F);
  return e;
}

// How far the ray's direction in the air is from the light's, and where it crossed the beam, both in
// the two axes square to the light
float2 eyeResidual(EyeExit e, EyeFrame light)
{
  const float3 r = e.direction - e.toLight;
  return float2(dot(r, light.a), dot(r, light.b));
}

float2 eyeBeamPoint(EyeExit e, EyeFrame light)
{
  return float2(dot(e.point, light.a), dot(e.point, light.b));
}

// The determinant of the 2x2 matrix whose columns are `c0` and `c1`, and the x that it takes to `r`
float eyeDet(float2 c0, float2 c1)
{
  return c0.x * c1.y - c1.x * c0.y;
}

float2 eyeSolve(float2 c0, float2 c1, float2 r)
{
  const float det = eyeDet(c0, c1);
  return float2(eyeDet(r, c1), eyeDet(c0, r)) / det;
}

// Unit `w` turned by `turn` (radians, small) along the two axes square to it
float3 eyeTurned(float3 w, EyeFrame around, float2 turn)
{
  return normalize(w + around.a * turn.x + around.b * turn.y);
}

// The turn that takes the centre ray's exit to the light's direction, by Newton's method: from the
// centre ray and the two turned by `angle` along each axis square to it. Not valid where a turn does
// not move the exit (the ray is on a fold).
struct EyeTurn
{
  float2 turn;
  bool   valid;
};

EyeTurn eyeNewtonTurn(EyeExit centre, EyeExit turnedA, EyeExit turnedB, EyeFrame light, float angle)
{
  EyeTurn      t;
  const float2 r0 = eyeResidual(centre, light);
  const float2 d0 = (eyeResidual(turnedA, light) - r0) / angle;
  const float2 d1 = (eyeResidual(turnedB, light) - r0) / angle;
  t.turn          = float2(0.0F);
  t.valid         = abs(eyeDet(d0, d1)) > 1.0e-6F;
  if(!t.valid)
    return t;
  t.turn            = eyeSolve(d0, d1, -r0);
  const float size2 = dot(t.turn, t.turn);
  if(size2 > EYE_CAUSTIC_MAX_TURN * EYE_CAUSTIC_MAX_TURN)
    t.turn *= EYE_CAUSTIC_MAX_TURN / sqrt(size2);
  return t;
}

// How much the film concentrates the light at the shaded point: the cross-section of the beam, square
// to the light, over the patch it lands on round the point, square to its arrival. From the ray that
// arrived, the two turned by `angle` and the two whose origin is moved by `shift` along the axes
// square to it.
//
// Moving the point by q moves where its ray crosses the beam by A q and its exit direction by C q;
// turning it by t moves them by B t and D t. A ray of the same beam keeps the light's direction, so
// its turn is -D^-1 C q, and it crosses the beam at (A - B D^-1 C) q: that matrix's determinant.
float eyeFocus(EyeExit centre, EyeExit turnedA, EyeExit turnedB, EyeExit movedA, EyeExit movedB, EyeFrame light, float angle, float shift)
{
  const float2 u0 = eyeBeamPoint(centre, light);
  const float2 r0 = eyeResidual(centre, light);
  const float2 a0 = (eyeBeamPoint(movedA, light) - u0) / shift;
  const float2 a1 = (eyeBeamPoint(movedB, light) - u0) / shift;
  const float2 c0 = (eyeResidual(movedA, light) - r0) / shift;
  const float2 c1 = (eyeResidual(movedB, light) - r0) / shift;
  const float2 b0 = (eyeBeamPoint(turnedA, light) - u0) / angle;
  const float2 b1 = (eyeBeamPoint(turnedB, light) - u0) / angle;
  const float2 d0 = (eyeResidual(turnedA, light) - r0) / angle;
  const float2 d1 = (eyeResidual(turnedB, light) - r0) / angle;
  if(!(abs(eyeDet(d0, d1)) > 1.0e-6F))
    return EYE_CAUSTIC_MAX;  // every ray from here leaves the same way: the point is at the focus
  const float2 t0 = eyeSolve(d0, d1, c0);
  const float2 t1 = eyeSolve(d0, d1, c1);
  const float2 m0 = a0 - (b0 * t0.x + b1 * t0.y);
  const float2 m1 = a1 - (b0 * t1.x + b1 * t1.y);
  const float  focus = abs(eyeDet(m0, m1));
  return focus < EYE_CAUSTIC_MAX ? focus : EYE_CAUSTIC_MAX;  // (a NaN is the cap)
}

// What a light sample at a point behind the film becomes
struct EyeCaustic
{
  float3 arrival;  // the light's direction at the shaded point, unit: what its BSDF is asked
  float3 exit;     // where the light came through the film, from the shaded point
  float3 toLight;  // the light's direction there, unit: the shadow ray's
  float3 normal;   // the film's there, unit, facing out of the chamber
  float  focus;    // how much the film concentrates it, against the light square to its arrival
  // The point lies under a film, deep enough for the film to move light (not the white of the eye,
  // which the film lies on): else the light sample stands as it was, straight
  bool film;
  // with `film`: the ray was found; else the light does not reach the point through the film, or the
  // point is deeper than a chamber is (the eye's inside behind the pupil), which takes none
  bool valid;
};

EyeCaustic eyeNoCaustic()
{
  EyeCaustic c;
  c.arrival = float3(0.0F, 0.0F, 1.0F);
  c.exit    = float3(0.0F);
  c.toLight = float3(0.0F, 0.0F, 1.0F);
  c.normal  = float3(0.0F, 0.0F, 1.0F);
  c.focus   = 1.0F;
  c.film    = false;
  c.valid   = false;
  return c;
}

#endif  // FV_EYE_CAUSTIC_H
