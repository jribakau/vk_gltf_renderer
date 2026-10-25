#pragma once

// The strand direction of hair cards, for KHR_materials_anisotropy. The renderer builds a tangent
// per triangle from the UVs and turns the anisotropy direction off it by (cos t, -sin t); a hair
// card's strand runs along its longest unshared edge, and UV layouts do not share an axis, so the
// rotation is measured from the mesh. A port of src/lib/rtx/hairAnisotropy.ts.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <figure/materials/surface.hpp>

namespace fv::mat {

// Kept when this share of the strand length lies within 20 degrees of one direction.
constexpr float kStrandConfidence = 0.7f;
// Glossy Anisotropy for a hair material that has no such channel, once its cards agree.
constexpr float kDefaultGlossyAnisotropy = 0.75f;

// Positions and UVs (the ones the glTF carries) of one mesh, and the index ranges of one material.
struct StrandMesh
{
  std::span<const float>    position;  // xyz per vertex
  std::span<const float>    uv;        // uv per vertex
  std::span<const uint32_t> index;
  struct Range
  {
    uint32_t start, count;
  };
  std::vector<Range> ranges;  // empty: the whole index buffer
};

// One vote per triangle: its longest unshared edge, when that edge is clearly the long one.
struct StrandVote
{
  float weight, theta;
};
std::vector<StrandVote> strandVotes(const StrandMesh& part);
// Each vertex's id among the distinct positions (cards split per triangle and UV seams repeat a
// position): what strandVotes welds first. A mesh's materials share one weld through the overload.
std::vector<uint32_t>   weldPositions(std::span<const float> position);
std::vector<StrandVote> strandVotes(const StrandMesh& part, std::span<const uint32_t> weldIds);

// The anisotropy rotation (radians, across the strand, in -pi/2..pi/2) of the votes, or none when
// the long edges do not agree.
std::optional<float> strandRotation(const std::vector<StrandVote>& votes);

// The anisotropyStrength that roughens the main lobe across the strand as much as Iray does, or
// none when the material asks for a round highlight.
std::optional<float> strandStrength(const Surface& s, float roughness, float glossyShare);

}  // namespace fv::mat
