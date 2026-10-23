#pragma once

// Cross-generation fitting of one worn item: alias the skeleton, carry the rest cage onto the target
// figure bone by bone, correct it toward the target body's shape, and push it out of the body. An
// item made for the figure's own generation is left exactly as authored. A port of
// src/lib/wardrobe/fit/fit.ts.

#include <optional>
#include <string>

#include <figure/dson/figure_loader.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/generation.hpp>

namespace fv::wardrobe {

struct FitDetail
{
  Generation source{}, target{};
  // whether the base mesh of the source generation was found, so the shape was corrected too
  bool shaped = false;
  // the pose rig's plan: item bone -> figure bone, and the item bones that inherit a followed ancestor
  LeaderPlan plan;
  // "Genesis 8 outfit on Genesis 9, refitted to its skeleton and shape."
  std::string note;
};

struct FitFigure
{
  const SceneGraph& scene;
  const dson::Rig&  rig;
  // the figure's body mesh (an index into scene.meshes)
  int mesh;
  // library path of the figure's folder (its Morphs folder's parent): names the generation
  std::string folder;
};

// The generations when `item` was made for another one than the figure's.
std::optional<std::pair<Generation, Generation>> generationsOf(const dson::LoadResult& wearable, const dson::Item& item, const std::string& rel,
                                                               const FitFigure& figure);

// Fits one item of `wearable` onto `figure` when it was made for another generation: its meshes' cages
// are rewritten in place (so call it before anything binds them). None leaves it untouched.
std::optional<FitDetail> fitItem(dson::Store& store, dson::LoadResult& wearable, const dson::Item& item, const std::string& rel, const FitFigure& figure);

}  // namespace fv::wardrobe
