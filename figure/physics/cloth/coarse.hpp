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

// A light copy of a cloth (asked for 2026-11-06): the cloth simulates as a coarse copy of itself, its
// faces about as wide as asked, and its own vertices ride on it, each on a face of the copy at fixed
// weights and an offset in the copy's frame there, so the folds, seams and pleats it was made with move
// with it. Folds narrower than the copy's faces can't form. The copy is made of the cloth's own
// vertices, so each keeps what it carries (its strength, its settings, its skin) as it is.
struct Riding
{
  // per vertex of the cloth: the face of the copy it rides on (into the copy's tris, three to a face)
  std::vector<uint32_t> face;
  // 3 per vertex: the face's corners' weights at the point it rides at
  std::vector<float> weights;
  // 3 per vertex: from that point along the face's first side, across it, and along the copy's normal
  std::vector<float> offset;
  // per vertex: 1 where the cloth is held to its skin (no strength), which it keeps
  std::vector<uint8_t> held;
};

struct CoarseCloth
{
  // simulated in the cloth's place; its faces are its polygons
  Surface surface;
  // the cloth's vertices it is made of, ascending
  std::vector<uint32_t> keep;
  Riding                riding;
};

// A light copy of a simulated cloth, its faces' sides about `edge` long, and how the cloth rides on it.
// None when it isn't simulated, or isn't much finer than that already.
std::optional<CoarseCloth> coarseCloth(const Surface& cloth, float edge);

// The copy `like` is, of the cloth as it is now (its pose, its skin), as sameCopy.
std::optional<CoarseCloth> sameCloth(const Surface& cloth, const CoarseCloth& like);

// The cloth's vertices where its copy's are `at` (xyz per vertex of the copy): xyz per vertex of the
// cloth, in the same space.
std::vector<float> ride(const CoarseCloth& c, const std::vector<float>& at);

// The copy's vertices keep out of the body, but its faces span a body's curve (a hip, the bust) as
// chords, and what rides on them can sink in. Each of `at` (xyz per vertex) closer to a face of `body`
// at `bodyAt` than its offset, or behind it by up to `reach`, is moved out along the face to the offset;
// the `held` ones (1 a vertex) stay. How many moved. With the `copy` they ride on, its vertices at
// `copyAt`, the same, looking only near the body for those whose face may put them near it.
size_t keepOut(std::vector<float>& at, const std::vector<uint8_t>& held, const Surface& body, const std::vector<float>& bodyAt, float reach = 2.f,
               const CoarseCloth* copy = nullptr, const std::vector<float>* copyAt = nullptr);

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
