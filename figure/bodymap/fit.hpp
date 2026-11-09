#pragma once

// Fitted body maps (docs/wardrobe-plan.md, step 3b): a body type Genesis 9 has no clone of (a Skyrim
// body: CBBE, 3BA, UNP) is mapped by laying the reference onto it first, then projecting both ways as a
// clone map does. The reference is Genesis 9's base, shaped by the nearest clone where one is near.
//
// The fit, in four moves, each from both bodies' own regions:
// - one scale and shift, from the middles of the joint bands both bodies have (the knee's band, where
//   thigh meets shin);
// - the limbs turned into the type's rest (intoRest), each part the source has most of its skin of;
// - an embedded deformation: a graph of nodes spread over the reference's skin, each turning and moving
//   the skin round it, drawn onto the source's closest skin within its regions (or next to them),
//   facing its way, stiff at first and less so over the rounds, its limb joints' bands held softly to
//   the source's (the spec's full landmarks). A node's skin and its neighbours are only within its
//   regions and the ones next to them, so one thigh never pulls the other. Each round alternates the
//   nodes' moves (one linear solve, the graph's size) and their turns (Horn's fit to their neighbours);
// - each vertex moved the last way along the source's normal onto its skin, the moves spread over the
//   mesh, for what the graph is too coarse for (a bust, a finger).
//
// Where a limb lands along itself follows the joint bands, and rigs place those differently: on Genesis
// 8 Female and 9 the knee's lie 1.3 cm apart on the same leg, so a fit of the one onto the other slides
// the shin by that (fv_parity's bodymap step, "fit", measures it against the clone map).
//
// A region the source lacks (a carried body cut to what shows) has nothing drawing it: it follows its
// neighbours, and the projection back leaves it unmapped (ProjectOptions::refuseCut).

#include <vector>

#include <figure/bodymap/clone.hpp>

namespace fv::bodymap {

struct FitOptions
{
  int    nodes     = 500;   // the deformation graph's nodes
  int    rounds    = 30;    // rounds of closest points
  double stiffFrom = 1;     // the graph's stiffness against the pull, at the first round...
  double stiffTo   = 0.01;  // ...and at the last
  double reach     = 10;    // cm: a reference vertex further than this from the source isn't pulled
  double maxAngle  = 60;    // degrees the source's skin may face away from a pulled vertex
  // How hard each limb joint's band (the spec's full landmarks: shoulder, elbow, wrist, hip, knee) is held
  // to the source's, as a share of the whole pull: nearest points alone let a limb slide along itself.
  double landmarkWeight = 0.005;
  // Passes of the last, finest move: each vertex along the source's normal onto its skin, the moves spread
  // over the mesh, for what the graph is too coarse for (a bust, a finger); and how far it reaches (cm).
  int    refine      = 5;
  double refineReach = 3;
  // cm along the skin a reference vertex may lie past the source's cut and still be pulled onto the skin
  // going on from it: else the strip by the cut has nothing pulled onto it
  double overCut = 1;
};

struct FitResult
{
  std::vector<float>    positions;  // the reference laid onto the source
  double                scale = 1;  // the first move: p * scale + shift
  Vec3                  shift;
  std::vector<LimbTurn> turns;      // the second
  size_t                nodes = 0, landmarks = 0;
  // the last round: the reference vertices pulled, and how far they lay from the source's skin (cm)
  size_t pulled = 0;
  double residualMedian = 0, residualP95 = 0;
};
// `reference` is the reference's body (its positions shaped by a clone or not); `cage` its base cage,
// whose skin and skeleton pose it (Genesis 9's).
FitResult fitReference(const wardrobe::BaseCage& cage, const Body& reference, const Body& source, const FitOptions& options = {});

// What a fitted map is made from: the source body (a carried or a body mod's base cage), Genesis 9's
// base, and the reference: Genesis 9's base, shaped or not.
struct FitInputs
{
  std::shared_ptr<const wardrobe::BaseCage> source, g9;
  Body                                      reference;
};
// The reference fitted onto the source and projected both ways, the projection onto the source
// refusing its cut; with the frame turns. `fit`, when given, gets the fit's own numbers.
TypeMap makeFittedMap(const FitInputs& inputs, const FitOptions& options = {}, FitResult* fit = nullptr);

// A body cut from another of its topology (an outfit's carried body, from the body mod it was built on,
// at the outfit's own preset), matched vertex to vertex (the plan's rule 3): each cut vertex to the whole
// body's of the same UV, the nearest of those by position where several share it; a vertex whose UV is
// another seam side's, to the nearest whole vertex next to its neighbours' matches. -1 where none.
std::vector<int32_t> matchCut(const wardrobe::BaseCage& cut, const wardrobe::BaseCage& whole);

// The cut body's map through the whole body's: a matched vertex lands where its match does, turned as it
// is; a reference vertex keeps its place when the triangle it lands on is the cut body's too, else it is
// unmapped. `cut` is the cut body as a map reads it.
TypeMap cutMap(const TypeMap& whole, const Body& cut, const std::vector<int32_t>& match);

}  // namespace fv::bodymap
