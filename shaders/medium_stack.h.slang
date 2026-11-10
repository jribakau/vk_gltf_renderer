#ifndef FV_MEDIUM_STACK_H
#define FV_MEDIUM_STACK_H

// The media a path is inside, two deep: the one it travels in and the one it was in when it entered
// that, which it is back in on leaving. A drink in a solid glass bottle, ice in a drink, a hand in a
// tub of water: the inner surface is met from inside the outer medium, so its index of refraction is
// against that medium's, not air's, and behind it the outer medium goes on. Slang and C++
// (tests/test_medium_stack.cpp), so the tests run the rules the shader runs.
//
// No priorities (Falcor's InteriorList has a 4-bit one a material authors; DAZ has no such channel):
// the path is in the medium it entered last. Where two solids overlap, a drink modelled a little into
// its glass, the face of the first met inside the second is no boundary (MediumBoundary::passed):
// the one boundary between them is the face the path crossed going in.
//
// Media are named by the material that bounds them. A third medium met inside two drops the
// outermost, whether the path goes into it or bounces off it: on the way out the path then leaves the
// second into air.
//
// The medium around is set when the inner surface is met (MediumBoundary::here), not when the path
// crosses it: the crossing then changes only what a path's state had before nesting.
//
// The path keeps the medium around only in the kernel of a scene that has a clear solid
// (USE_NESTED_MEDIA, SceneFeatureSet::eNestedMedia). Anywhere else there is none to keep, the
// state says -1, and these rules are the one-medium rules the path tracer had: kept in every kernel,
// that one number cost a frame 3% in scenes with no medium inside another (1TEST, 2026-11-10).

struct MediumState
{
  bool inside;   // the path is in a medium
  int  current;  // the material whose medium that is; -1: none, or a medium no material names (the camera's)
  // The medium around it; -1, or the current one itself: none. At the front of a surface with a medium
  // of its own it is set to the current one (mediumBoundary's `here`), which is what is around that
  // surface's medium once the path is in it, and nothing to go back to while it is not.
  int outer;
};

MediumState mediumOutside()
{
  MediumState s;
  s.inside  = false;
  s.current = -1;
  s.outer   = -1;
  return s;
}

// Out of the current medium, into the one around it
MediumState mediumLeft(MediumState s)
{
  MediumState left;
  left.inside  = s.inside && s.outer >= 0 && s.outer != s.current;
  left.current = left.inside ? s.outer : -1;
  left.outer   = -1;
  return left;
}

// Into `material`'s medium, from `s` as its surface was met (mediumBoundary's `here`, whose outer is
// the medium it was met in): only the current medium changes
MediumState mediumEntered(MediumState s, int material)
{
  MediumState entered;
  entered.inside  = true;
  entered.current = material;
  entered.outer   = s.inside ? s.outer : -1;
  return entered;
}

struct MediumBoundary
{
  // A face of the medium around the current one, met inside the current one (the two overlap): the path
  // goes on as if it were not there, in `here`
  bool passed;
  // The media on the ray's side of the surface, which it is shaded in and reflects back into. Not
  // always the state the path arrived with: a walk ends at any front face, and so does a medium whose
  // own front face the path meets (it was never in it: an open mesh took it for a solid); and at the
  // front of a surface with a medium of its own, the medium around is the current one from here on.
  MediumState here;
  // The media beyond, where a transmission through the surface takes the path
  MediumState beyond;
  // The ray is on the side of the surface's own medium (a back face of a surface that bounds one): its
  // index is this side's. Otherwise its own medium, if it bounds one, is beyond.
  bool ownSide;
  // The material whose medium the surface's own index of refraction is against: the one beyond it
  // from its own side, else the one it is met in, whether or not it bounds a medium itself (a spoon
  // in a drink). -1: air (or the camera's medium, which has no index).
  int against;
};

// What a surface is to the media a path is in. `material`: the surface's; `bounds`: it bounds a medium
// the path can enter (surfaceBoundsMedium, and not a scatter material shaded thin); `frontFace`: met
// from outside that medium; `walk`: the current medium scatters (a walk through skin, which an opening
// or what sits in one ends: frontFaceLeavesMedium's rule before the stack).
MediumBoundary mediumBoundary(MediumState s, int material, bool bounds, bool frontFace, bool walk)
{
  MediumBoundary b;
  b.passed  = false;
  b.here    = s;
  b.ownSide = false;
  if(frontFace)
  {
    if(s.inside && walk)
      b.here = mediumLeft(s);
    if(bounds && b.here.inside && b.here.current == material)
      b.here = mediumOutside();
  }
  if(frontFace && bounds && b.here.inside)
    b.here.outer = b.here.current;
  b.beyond  = b.here;
  b.against = b.here.inside ? b.here.current : -1;
  if(!bounds)
    return b;
  if(frontFace)
  {
    b.beyond = mediumEntered(b.here, material);
    return b;
  }
  if(s.inside && s.outer == material && s.current != material)
  {
    b.passed     = true;
    b.here.outer = -1;
    b.beyond     = b.here;
    return b;
  }
  // The back of the current medium's own surface, or of one the path did not know it was in (it
  // started inside it, or the mesh is open): either way it leaves that medium here
  b.ownSide = true;
  b.beyond  = mediumLeft(s);
  b.against = b.beyond.inside ? b.beyond.current : -1;
  return b;
}

#endif  // FV_MEDIUM_STACK_H
