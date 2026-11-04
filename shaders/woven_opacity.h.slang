#ifndef WOVEN_OPACITY_H
#define WOVEN_OPACITY_H

// A woven fabric's opacity seen along a direction: a material's cutout when its opacityWoven is set
// (clothing, mat::Material::woven). The cloth is two sets of round, opaque threads crossing at right
// angles in its plane, the warp along the UV tangent and the weft across it, each covering c of the
// surface head-on, so the authored opacity is what the two leave closed, 1 - (1 - c)^2. Seen
// obliquely across a set, the gaps between its threads foreshorten while the threads' round
// sections do not: its coverage becomes c |v x t| / |v . n|, up to 1. A sheer stocking closes up
// towards the leg's outline, as nylon does; a hole (0) and solid cloth (1) stay as they are. It is
// the same from either side and for either direction of travel, so camera rays and shadow rays
// agree. (Community Shaders' ExtendedTranslucency has the same two-set model, with a looser
// overlap term.)
//
// `v` is the ray's direction and `n`, `t` the surface's normal and warp, in one frame and of any
// length. `t` need not be square to `n`; where it lies along `n`, any axis in the plane serves.
float wovenOpacity(float opacity, float3 v, float3 n, float3 t)
{
  if(opacity <= 0.0F || opacity >= 1.0F)
    return opacity;
  n             = normalize(n);
  v             = normalize(v);
  float3 warp   = t - n * dot(n, t);
  if(dot(warp, warp) < 1e-12F)
    warp = abs(n.x) < 0.9F ? cross(n, float3(1.0F, 0.0F, 0.0F)) : cross(n, float3(0.0F, 1.0F, 0.0F));
  warp               = normalize(warp);
  const float3 weft  = cross(n, warp);
  const float facing = abs(dot(v, n));
  // a ray in the cloth's plane meets the threads side on
  if(facing < 1e-6F)
    return 1.0F;
  const float headOn    = 1.0F - sqrt(1.0F - opacity);  // one set's coverage seen along the normal
  const float warpCover = min(1.0F, headOn * length(cross(v, warp)) / facing);
  const float weftCover = min(1.0F, headOn * length(cross(v, weft)) / facing);
  return 1.0F - (1.0F - warpCover) * (1.0F - weftCover);
}

#endif
