#pragma once

// The numbers a drape is judged by without a viewport: how far its stretch rows are drawn out, how
// many of its vertices lie inside the body, how far its hem dropped. fv_parity's drape report and the
// GPU cloth bench (docs/physics-plan.md step 13a) take them from here, so both judge alike.

#include <cstddef>
#include <vector>

#include <figure/physics/cloth/model.hpp>

namespace fv::physics::cloth {

// The stretch rows of one simulated surface (its slot in the model, its `n` vertices at `out`) as a
// ratio of their rest length, sorted. A row between two pinned vertices is the skin's doing, not the
// solver's, and is left out.
std::vector<double> stretchRatios(const Model& model, size_t slot, size_t n, const std::vector<float>& out);

// How many of `out`'s vertices lie behind the body's closest face (the wardrobe's pushOut rule, within
// 3 cm), the body at `bodyNow`.
size_t insideBody(const Surface& body, const std::vector<float>& bodyNow, const std::vector<float>& out);

// How many of `a`'s vertices have gone to the other side of `b`, another simulated surface: each one's
// side of b's closest face where both lie as made, against its side where both lie now (`aOut`,
// `bOut`), for the vertices within `reach` cm of b both times. Garments of one layer keep the side they
// were made on (a skirt over its petticoat), as both solvers keep it, so any here went through.
size_t wrongSide(const Surface& a, const std::vector<float>& aOut, const Surface& b, const std::vector<float>& bOut, float reach = 2.f);

struct DrapeNumbers
{
  double stretchMax = 0, stretchP99 = 0;
  size_t stretchOver12 = 0;  // rows past 1.2 of their rest length
  size_t inside        = 0;
  double hemDrop       = 0;  // how far its lowest vertex lies under the posed shape's, cm
};
DrapeNumbers drapeNumbers(const Model& model, size_t slot, const Surface& mesh, const std::vector<float>& out, const Surface& body,
                          const std::vector<float>& bodyNow);

}  // namespace fv::physics::cloth
