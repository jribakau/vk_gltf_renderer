#pragma once

// A figure's skin as material presets change it, the way eye_materials.hpp does its eyes: a skin
// preset replaces the last one instead of stacking on it (each choice starts from the character's
// own, the materials' baseline), and makeup is a layer over whichever skin is on: choosing other
// makeup takes off only the channels the last makeup set, so the skin under it stays. Also the map
// a skin tone shifts every skin surface against.

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <figure/dson/channels.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dson {

// The character's surfaces a skin or makeup preset may change: those in use, not its eyes (they are
// chosen apart) and not `leaveOut` (the host's worn clothing, props and hair).
std::set<int> skinSurfaces(const SceneGraph& scene, const std::set<int>& leaveOut);

// What a layered preset set on each material, by channel key: the channel it replaced (null: there
// was none) and the one it put there (null: it took the channel away, as a preset of another shader
// takes the old shader's).
struct LayerChannel
{
  std::shared_ptr<Channel> before, after;
};
struct LayerMaterial
{
  std::map<std::string, LayerChannel> channels;
  // the shader the material's files name, before and after, when the preset switched it
  std::optional<std::pair<mat::ShaderRef, mat::ShaderRef>> shader;
};
using PresetLayer = std::map<int, LayerMaterial>;

struct LayerChoice
{
  std::set<int>        changed;  // surfaces parsed again
  PresetLayer          layer;    // what this choice set, for the next one
  MaterialPresetResult preset;
};

// A preset laid over `surfaces` in place of the last one laid there: the channels `previous` set go
// back to what they were before it (one something else has set since keeps that), then the part of
// `doc` on `surfaces` goes over them; null `doc` only takes the last one off. Edits stay. A layer
// that switched a surface's shader comes off with the shader it replaced.
LayerChoice chooseLayer(SceneGraph& scene, const std::set<int>& surfaces, const Json* doc, const PresetLayer& previous,
                        const MaterialDefinitions& definitions = {});

// Several presets laid together, bottom up, each over what the one before left (tattoos), in place of
// the stack `previous`: `layer` takes them all off at once, so any of them can go or change by laying
// the rest again. `opacity` under 1 fades a layered preset (fadedLayers); a preset with no layers
// takes it as it comes. Nothing to lay only takes the stack off.
struct LaidPreset
{
  const Json* doc     = nullptr;
  float       opacity = 1;
};
LayerChoice chooseLayers(SceneGraph&                                            scene,
                         const std::set<int>&                                   surfaces,
                         const std::vector<LaidPreset>&                         presets,
                         const PresetLayer&                                     previous,
                         const std::function<MaterialDefinitions(const Json&)>& definitions = {});

// The layered preset with every layer's opacity times `opacity`. A whole image at the bottom then
// no longer covers what is under it, so it goes over the map there (collectImages).
Json fadedLayers(const Json& doc, float opacity);

// The base map of the figure's main skin surface (Genesis 9 "Body", Genesis 8 "Torso"; else the first
// skin surface with a map), the first layer of a layered one; empty when there is none. Not `leaveOut`.
std::string toneReference(const SceneGraph& scene, const std::set<int>& leaveOut = {});

}  // namespace fv::dson
