#pragma once

//-------------------------------------------------------------------------------------------------------
// The backdrop's state as the host keeps it (the app's World tab; protocol 70): one fv::backdrop::Spec,
// patched by `backdrop.set` and published as the `backdrop` topic in the same shape, and the renderer
// parameters a backdrop that stands sets. Pure, so it is tested without a device
// (tests/test_fv_backdrop.cpp).
//
//   kind none|meadow|forest (ground that rolls away from where the figures stand, planted with the
//   kit's plants; while one stands the lighting's ground plane isn't drawn) or planet (a small world
//   under the scene, fv_world.hpp: bare, and it needs no kit; protocol 75), seed (which one), size
//   (metres across what is planted underfoot), density (of every layer's own), relief (how hilly, 0
//   flat), air (1 to 20: how many metres of air a metre of its land counts for, the distance's haze
//   under the physical sky), mature (grown firs among the near woods), wind (km/h, 0 to 100, 0 still:
//   the trees, saplings and bushes sway in it) and windFrom (degrees, an azimuth as the sun's); a
//   planet's radius (km, 2 to 100) and sea (the share of its surface under water, 0 to 0.9).
//
// Read-only in the topic, the host's: state (off; missing: no kit where settings.apply's backdropKit
// says; loading; ready) and plants (how many stand).
//
// Until protocol 70 this was the lighting's `backdrop` section, set with lighting.set.
//-------------------------------------------------------------------------------------------------------

#include "fv_backdrop.hpp"
#include "fv_events.hpp"
#include "fv_world.hpp"

namespace fv::backdrop {

Json toJson(const Spec& s);

// `s` with an RFC 7386 merge patch applied. A null resets that key to its default. Unknown keys, wrong
// types and bad enum values throw std::invalid_argument and leave nothing applied; numbers are clamped
// to their ranges. The topic's read-only keys (state, plants) are accepted and ignored, so a saved
// topic object replays as a patch.
Spec applyPatch(const Spec& s, const Json& patch);

// The registered renderer parameters the backdrop sets, by name, over the lighting's
// (fv::lighting::rendererParams). One that `stands` (planted, not just asked for: its kit may be
// missing, or being read) takes the ground plane's place, counts the air over its land `air` times,
// has its land give way to the sky's own ground before it ends, and gives the planet past it
// `landAlbedo`, what its kit says the land reflects from afar. With none standing the air counts once,
// nothing fades, and the plane is the lighting's to say.
//
// A planet that stands is the physical sky's planet: its radius and its air (sky_world.hpp), and its
// centre, `planetCentre` in the scene, which the eye's height is taken from; the air counts once and
// nothing fades. What is laid out on a level plane has no place on it yet and is off whatever the
// lighting says: the clouds, the rain and the mist.
Json rendererParams(const Spec& s, bool stands, const glm::vec3& landAlbedo, const glm::dvec3& planetCentre = glm::dvec3(0.0));

// The planet a spec asks for
fv::world::Spec planetSpec(const Spec& s);

}  // namespace fv::backdrop
