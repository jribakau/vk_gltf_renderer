#pragma once

// What the soft flesh meets while it moves (docs/physics-plan.md step 11): props' triangles, other
// people's limbs and the figure's own, found once a frame as a plane for each particle near them
// (FleshContact), which the frame's substeps hold it in front of. Once the body stops, the still
// contact pass (contact.hpp) dents it, and the flesh hands its dents over (flesh.hpp).
//
// A limb is a ragdoll part's capsule (FigurePhysics), in either mode. A particle meets another
// person's limbs, and of its own those of another part of the body: of a part joined to its own only
// where it is 5 cm or more from its part's edge along the skin, as self-contact has it, so a bending
// hip doesn't dent the belly's crease. However deep one of its own capsules is in the flesh at rest,
// what the capsule's fit gets wrong, is allowed it.

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/physics/flesh/flesh.hpp>

namespace fv::physics {

class FleshTouch
{
public:
  // Per particle: its part of the body, how far it is from its part's edge along the skin (cm) and
  // where it is at rest (xyz, the cage's space); per part, the parts joined to it (bit q of joined[p]).
  FleshTouch(std::vector<uint8_t> part, std::vector<float> fromEdge, std::vector<uint32_t> joined, std::vector<float> rest);

  // This frame's contacts for particles at `targets` (xyz per particle first, as FleshSolver takes
  // them) and `offsets` from them, against `colliders` carried into the targets' space by `toFlesh`.
  std::vector<FleshContact> find(std::span<const float> targets, std::span<const double> offsets, const FleshColliders& colliders, const Mat4& toFlesh);

private:
  // which of these own limbs each particle meets, and how deep it may be in each
  void allowFor(const std::vector<FleshColliders::Limb>& own);

  std::vector<uint8_t>  m_part;
  std::vector<float>    m_fromEdge, m_rest;
  std::vector<uint32_t> m_joined;
  // the own limbs the table is for (each one's part, then its rest capsule's ends and radius), and per
  // particle and limb how deep it may be; below 0 it doesn't meet that limb
  std::vector<std::array<double, 8>> m_for;
  std::vector<double>                m_allow;
};

}  // namespace fv::physics
