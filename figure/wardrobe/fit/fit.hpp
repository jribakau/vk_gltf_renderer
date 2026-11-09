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
#include <figure/wardrobe/fit/transfer.hpp>

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
  // per object of the item (dson::Item::objects): for one hung from a bone (glasses on the head), that
  // bone's joint centre on the source generation's base, which it was placed around; none for one on
  // the figure, or when the base isn't installed
  std::vector<std::optional<V3>> mountFrom;
};

// The joint centres a prop hung from a bone was placed around, per object of `item`, from the base
// skeleton of the generation it was made for (see FitDetail::mountFrom).
std::vector<std::optional<V3>> mountsFrom(const dson::Item& item, const std::vector<SkelBone>* base);

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

// Whether `item` is known to be made for the figure's own rig. The bones it has beyond the figure's are
// then its own (a hair's strands, a skirt's panels), riding on the ones they share.
bool madeForFigure(const dson::LoadResult& wearable, const dson::Item& item, const std::string& rel, const FitFigure& figure);

// Fits one item of `wearable` onto `figure` when it was made for another generation: its meshes' cages
// are rewritten in place (so call it before anything binds them). None leaves it untouched.
std::optional<FitDetail> fitItem(dson::Store& store, dson::LoadResult& wearable, const dson::Item& item, const std::string& rel, const FitFigure& figure);

}  // namespace fv::wardrobe
