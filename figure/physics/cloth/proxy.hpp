#pragma once

// Live cloth meets the body as games make cloth meet it (physics-plan step 13c). Live steps late and fast,
// and the body's own faces, met with sided collisions, keep a vertex that a step left inside a limb on
// the inside for good: every remedy tried against that (ties, a floor under the faces, recovery) caught
// one case and left the next. Games keep to three rules instead, and so does a Live job:
//
// - The body is met as its capsule floor alone (physics/cloth/capsules.hpp). A capsule has one inside,
//   so no step, however late or fast, can leave cloth on the wrong side of it. The body's faces leave the
//   simulation, and what is drawn is held out of them (leash.hpp's leashAll, each frame): a step the
//   simulation never reads back, so it can't trap anything either.
// - What a garment has on the skin at rest (a bodice, a waistband, a fitted sleeve) follows the skin as
//   a conforming garment does: held to its skin within kOnSkin cm of the body, free from kOnSkinFade cm
//   further, tethered between. Only what hangs free simulates. The drawn hold keeps the held part out of
//   the body too, as the wardrobe's collider would (it leaves a draped mesh to the simulation).
// - What is worn on the skin under the cloth (stockings, underwear) is part of the skin: its faces leave
//   the simulation, and it thickens the body the drawn cloth is held out of, by how far it lies off the
//   skin and its own gap. What stands off the skin (a boot's sole) or lies over the cloth (a belt) stays
//   a collider.
//
// A settle keeps the faces: it has the time to get them right, and a still is judged by them.
//
// Off unless asked for (Cloth::liveProxy, FV_DFORCE_PROXY_LIVE=1), as the faces leave fewer inside. On
// 10TEST's walk (fv_parity's "walk" with "live", the Witch Dress at Keep shape 0.767 on the CPU island,
// 20 steps every 90 ms, each drape bound afresh and held as drawn), it drew 1,240 of the dress's vertices
// inside the body over the walk, 1,203 as light cloth; the body's faces 369 and 426. Of the proxy's, 842
// were held to the skin, at the collars, under the bust and at the shoulders, carried into the body's
// creases by the garment's own skin (drawn where the body's faces carry it instead, as many); what hung
// free, 398, no fewer than the faces left in all. Neither showed the thighs or the cuffs a 4K viewport
// shows on the GPU.

#include <cstddef>
#include <vector>

#include <figure/physics/cloth/surface.hpp>

namespace fv::physics::cloth {

// What lies this near the body at rest is on it, cm; free from kOnSkinFade further off.
constexpr float kOnSkin     = 1.f;
constexpr float kOnSkinFade = 1.f;

struct Proxied
{
  bool   on     = false;  // the job meets the body as its floor
  size_t held   = 0;  // drape vertices on the skin, now held to it
  size_t eased  = 0;  // and those tethered between held and free
  size_t folded = 0;  // collider vertices on the skin under the cloth, now part of it
};

// `surfaces` made a Live job's: `body` the body's slot (its faces leave the simulation, and its offsets take
// what is worn on it), `drapes` the slots the cloth simulates in. The body's capsule floor is among the
// surfaces already; nothing changes without one. Everything is measured at rest, as the garments were made.
Proxied proxyLive(std::vector<Surface>& surfaces, size_t body, const std::vector<size_t>& drapes);

}  // namespace fv::physics::cloth
