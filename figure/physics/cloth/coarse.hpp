#pragma once

// A collider finer than cloth can feel: a necklace's links, lace, a boot's stitching. Cloth meets a
// collider through the faces near each of its vertices, so 150k triangles of chain under a neckline
// cost a dress five times what the body under it does, and change nothing a vertex a centimetre from
// the next can tell. The cloth meets a coarse copy instead, its faces about as wide as the cloth's
// own. The copy may lie a little off the original's surface: that goes on its collision offset, so
// the cloth still keeps out of the original.

#include <cstdint>
#include <optional>
#include <vector>

#include <figure/physics/cloth/surface.hpp>

namespace fv::physics::cloth {

// The mean length of a surface's faces' sides, at its rest cage; 0 without faces.
float meanEdge(const Surface& s);

struct Coarse
{
  Surface               surface;
  // the original's vertices it keeps, ascending: vertex i of `surface` is keep[i] of the original
  std::vector<uint32_t> keep;
  // how far it lies from the original, in the cage's units (cm): the farthest of the original's colliding
  // vertices from its faces, measured. Already on its offset
  float error = 0;
};

// A coarse copy of a collider, with faces whose sides are about `edge` long, simplified to lie within
// `maxError` of the original's surface. Only the faces that collide are kept (a face collides when its
// three vertices do). None when it isn't a collider, when it isn't much finer than that already, when
// the copy would keep most of its faces, or when it lies further off than twice `maxError`.
std::optional<Coarse> coarsen(const Surface& collider, float edge, float maxError);

// The copy `like` is, of the collider as it is now (its pose, its skin): the same vertices and faces,
// and its error on the offset. For a collider whose rest shape is the one `like` was made from; none
// when it hasn't the vertices.
std::optional<Coarse> sameCopy(const Surface& collider, const Coarse& like);

// What `keep` takes of an array with `per` values a vertex (xyz: 3).
template <typename T>
std::vector<T> keepOf(const std::vector<T>& values, const std::vector<uint32_t>& keep, size_t per)
{
  std::vector<T> out;
  out.reserve(keep.size() * per);
  for(uint32_t v : keep)
    for(size_t k = 0; k < per; k++)
      out.push_back(values[size_t(v) * per + k]);
  return out;
}

}  // namespace fv::physics::cloth
