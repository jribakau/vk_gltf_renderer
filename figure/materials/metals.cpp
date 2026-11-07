#include <figure/materials/metals.hpp>

#include <algorithm>
#include <iterator>

namespace fv::mat {

namespace {

// python utils/metal_edge_tints.py; each line's comment is its colour and how far the F82 curve
// through it is from the measured metal at worst, against Schlick's from the same f0
constexpr Metal kMetals[] = {
    {"gold", {1.0000f, 0.7327f, 0.3659f}, {0.9847f, 1.0000f, 0.9804f}},       // #ffdea3; off the measured metal by 0.020 (Schlick: 0.017)
    {"silver", {0.9891f, 0.9841f, 0.9776f}, {0.9933f, 0.9945f, 0.9990f}},     // #fefdfc; off the measured metal by 0.002 (Schlick: 0.007)
    {"copper", {0.9216f, 0.6251f, 0.5208f}, {0.9812f, 0.9470f, 0.9455f}},     // #f6cfbf; off the measured metal by 0.011 (Schlick: 0.043)
    {"brass", {0.9012f, 0.7828f, 0.4190f}, {0.9529f, 0.9783f, 1.0000f}},      // #f4e5ad; off the measured metal by 0.027 (Schlick: 0.047)
    {"aluminium", {0.9067f, 0.9161f, 0.9226f}, {0.8891f, 0.9179f, 0.9477f}},  // #f4f5f6; off the measured metal by 0.009 (Schlick: 0.108)
    {"chrome", {0.5496f, 0.5561f, 0.5537f}, {0.7418f, 0.7480f, 0.8150f}},     // #c4c5c4; off the measured metal by 0.025 (Schlick: 0.199)
    {"iron", {0.5289f, 0.5136f, 0.4938f}, {0.7655f, 0.7669f, 0.8024f}},       // #c0beba; off the measured metal by 0.018 (Schlick: 0.178)
    {"titanium", {0.6170f, 0.5799f, 0.5424f}, {0.7696f, 0.7901f, 0.8211f}},   // #cec8c2; off the measured metal by 0.015 (Schlick: 0.185)
};

}  // namespace

std::span<const Metal> metals()
{
  return kMetals;
}

const Metal* findMetal(std::string_view key)
{
  const auto it = std::find_if(std::begin(kMetals), std::end(kMetals), [&](const Metal& m) { return key == m.key; });
  return it == std::end(kMetals) ? nullptr : &*it;
}

const Metal* metalOf(const Edits& e)
{
  return e.metal ? findMetal(*e.metal) : nullptr;
}

Rgb metalColor(const Edits& e, const Rgb& own, bool mapped)
{
  if(e.color)
    return *e.color;
  const Metal* metal = metalOf(e);
  return metal && !mapped ? metal->f0 : own;
}

}  // namespace fv::mat
