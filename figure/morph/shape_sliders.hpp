#pragma once

// General body sliders: a few dials for what a body is, -1 to +1 with 0 the character as loaded, on
// any Genesis figure. A figure's own morphs can't give them: Genesis 9 has none for the waist or the
// thighs. A girth is made here from the unmorphed cage, its skin and its bones' axes; a length or a
// size that the figure does have a dial for drives that dial through a formula.
//
// The channels are ordinary dials, so scene files, undo, looks and what follows the body take them
// as they do any other. A scene file holds only the slider's value: once released, a recipe's cap
// and bones are part of the file format, and a retune takes a new channel name.

#include <array>
#include <span>
#include <string>
#include <vector>

#include <figure/dson/morphs.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/figure_driver.hpp>
#include <figure/scene/graph.hpp>

namespace fv::morph {

// What the channels' `kind` says: the morph panel gives them a tab of their own.
inline constexpr const char* kShapeKind = "Modifier/FvShape";

// One bone of a mesh's skin, in skin-index order: its node's name and its axis as authored.
struct ShapeBone
{
  std::string           name;
  std::array<double, 3> center{}, end{};
};

// A slider that pushes the skin of some bones away from their axes.
struct GirthRecipe
{
  std::string              channel, label;
  // DAZ's region, which the panel sorts by
  std::string              region;
  // anatomical slots, as wardrobe::parseBone names them
  std::vector<std::string> slots;
  // how much of a vertex's distance from the axis it moves at +1
  float cap = 0.2f;
  // the share kept of the sideways move toward the body's midline: thighs meet there
  float inward = 1.f;
};

// A slider that moves a dial the figure already has.
struct DialRecipe
{
  std::string channel, label, region;
  std::string dial;
  // the dial's value at +1
  double scale = 1;
};

const std::vector<GirthRecipe>& girthRecipes();
const std::vector<DialRecipe>&  dialRecipes();

// A girth's cage deltas at +1: each vertex moves by the sum over the recipe's bones of its weight x
// cap x its offset from that bone's axis. A bone with no length has no axis, and moves nothing.
mesh::CageDeltas girthDeltas(std::span<const float> cage, const mesh::SkinInput& skin, const std::vector<ShapeBone>& bones, const GirthRecipe& recipe);

// Adds the sliders these meshes can have: a girth with none of its bones skinned, or a dial the
// figure doesn't have, is left out.
void addShapeSliders(MorphSet& morphs, pose::FigureDriver& driver, const SceneGraph& scene, const std::vector<const dson::MeshRef*>& bodies);

}  // namespace fv::morph
