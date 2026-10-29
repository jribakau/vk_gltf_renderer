#pragma once

// A figure's eyes as material presets change them: which surfaces are the eyes, whether a preset
// is a pair of eyes, and choosing eyes so that one pair replaces the last instead of stacking on it
// (each choice starts from the character's own, the materials' baseline). Aiming the eyes is
// pose.eyes, elsewhere.

#include <set>

#include <figure/dson/material_preset.hpp>
#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dson {

// A surface of the figure's eyes: kind eEye (G9 "Eye Left", G8 "Irises", "Sclera", "Pupils"), or
// eCornea other than the tear film (Genesis9Tear's "Tear" keeps what it has).
bool isEyeSurface(const Material& m);
// The scene's eye surfaces, leaving out those that only taken-off items use.
std::set<int> eyeSurfaces(const SceneGraph& scene);
// A preset that lands on eye surfaces alone (and on some).
bool isEyePreset(const Json& doc, const SceneGraph& scene);

using EyesResult = PresetChoice;

// choosePreset on the eye surfaces: null `doc` is the character's own.
EyesResult chooseEyes(SceneGraph& scene, const Json* doc, const std::set<int>& previous = {}, const MaterialDefinitions& definitions = {});

}  // namespace fv::dson
