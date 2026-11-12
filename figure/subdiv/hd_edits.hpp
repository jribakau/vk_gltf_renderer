#pragma once

// DAZ HD morph detail on a subdivided mesh: a .dhdm file's edits (figure/dson/dhdm.hpp), placed on
// a Refinement's vertices, and the reference that applies them.
//
// Each subdivision level of the cage may carry edits. An edit names a cage face, a face of that
// level among the face's descendants, and a corner of it, which is one vertex of that level; its
// offset is measured in the frame of the cage face corner the descendant lies in, and added once the
// level is refined, before the next level is made from it. Edits are linear, so several morphs'
// offsets add, weighted.
//
// The frame of cage face corner i (corners v0 .. vn-1): x along the edge to the corner before,
// normalize(v[i-1] - v[i]); n the face's normal, normalize(cross(v1 - v0, v[n-1] - v0)); y =
// normalize(cross(n, x)). An offset (a, b, c) moves the vertex by a x + b n - c y. The format is
// undocumented; the layout and this basis follow edolstra's reading of it
// (https://github.com/edolstra/dhdm, src/subdivide.cc), and the basis was checked on Genesis 9's FACS
// Details: it gives the smoothest field of the bases tried (tests/figure/test_hd_edits.cpp).

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <figure/subdiv/catmark.hpp>

namespace fv::subdiv {

// One edit as the file has it.
struct FaceEdit
{
  uint32_t face   = 0;  // cage face
  uint32_t sub    = 0;  // the face's descendant at the edit's level, in OpenSubdiv's order
  uint8_t  corner = 0;  // the descendant's corner: the vertex edited
  float    offset[3]{};
};

// A file's edits, level by level (levels[0] edits level 1).
using LevelEdits = std::vector<std::vector<FaceEdit>>;

// Edits placed on a refinement, for the levels it has.
struct HdEdits
{
  struct Level
  {
    std::vector<uint32_t> vertex;      // the level's vertex (vertices keep their index at the next level)
    std::vector<uint32_t> faceCorner;  // the cage face corner whose frame the offset is in (faceStart[f] + i)
    std::vector<float>    offset;      // three per edit, in that frame
  };
  std::vector<Level> levels;  // levels[k] edits level k + 1
  // edits at levels the refinement doesn't make, or naming a face or corner it doesn't have
  size_t dropped = 0;

  bool empty() const;
};

// Places `edits` on `r`, made from a cage whose faces start at `faceStart`.
HdEdits placeEdits(const Refinement& r, std::span<const int32_t> faceStart, const LevelEdits& edits);

// The frame of cage face corner `faceCorner`: columns x, n, -y as above, so frame * offset is the move.
std::array<float, 9> cornerFrame(std::span<const float> cagePositions, std::span<const int32_t> faceStart,
                                 std::span<const int32_t> faceVerts, std::span<const int32_t> cornerFace, uint32_t faceCorner);

// The face each corner belongs to (faceStart's inverse), for cornerFrame.
std::vector<int32_t> cornerFaces(std::span<const int32_t> faceStart);

// The reference: `cagePositions` refined to the finest level with the weighted edits added level by
// level, frames from those cage positions.
std::vector<float> refineWithEdits(const Refinement& r, std::span<const float> cagePositions, std::span<const int32_t> faceStart,
                                   std::span<const int32_t> faceVerts, std::span<const std::pair<float, const HdEdits*>> edits);

// The five cage vertices a corner's frame is made from: its face's first, second and last, the
// corner's own and the one before it.
using FrameVertices = std::array<int32_t, 5>;
FrameVertices frameVertices(std::span<const int32_t> faceStart, std::span<const int32_t> faceVerts, std::span<const int32_t> cornerFace,
                            uint32_t faceCorner);
// The frame of those five points of `positions` (three floats a vertex): cornerFrame's.
std::array<float, 9> frameOf(std::span<const float> positions, const FrameVertices& v);

// One edit's move as the GPU makes it (shaders/stencil.comp.slang), in floats in the shader's order:
// `offset` in the frame of the five cage vertices `v` names in `cage` (three floats a vertex). The
// test backend and the residency oracle both run this.
std::array<float, 3> hdMove(std::span<const float> cage, const int32_t* v, const float* offset);

// A mesh's HD edits as the GPU adds them (the stencil pass, shaders/stencil.comp.slang): per level,
// each row's edits (start per row of the level, plus one), each naming a frame and an offset in it;
// the frames by their five cage vertices.
struct HdTables
{
  struct Topology
  {
    struct Level
    {
      std::vector<uint32_t> start;  // the level's rows, plus one
      std::vector<uint32_t> frame;  // per edit
    };
    std::vector<Level>  levels;
    std::vector<int32_t> frameVertices;  // five per frame
    size_t frames() const { return frameVertices.size() / 5; }
  };
  std::shared_ptr<const Topology> topology;
  std::vector<std::vector<float>> offsets;  // per level, three per edit: the weighted sum
  // the most the edits can move a vertex: each level's longest offset, summed (a frame is a rotation)
  float reach = 0;
};

// One mesh's HD morphs mixed by weight. Every edit set the mesh has been given keeps its place, so a
// weight that comes and goes changes offsets only, and the topology is made again only when a new
// set joins.
class HdMixer
{
public:
  HdMixer(std::shared_ptr<const Refinement> r, std::vector<int32_t> faceStart, std::vector<int32_t> faceVerts);

  using Weighted = std::pair<std::shared_ptr<const LevelEdits>, float>;
  // The weighted sum of `edits`; null when none has a weight.
  std::shared_ptr<const HdTables> mix(std::span<const Weighted> edits);

  // How many edits fell off the refinement (levels it doesn't make, faces it doesn't have).
  size_t dropped() const { return m_dropped; }

private:
  struct Known
  {
    std::shared_ptr<const LevelEdits> edits;
    std::vector<std::vector<uint32_t>> entry;  // per level, per placed edit: its place in the topology
    HdEdits                            placed;
  };
  void remake();

  std::shared_ptr<const Refinement>        m_refinement;
  std::vector<int32_t>                     m_faceStart, m_faceVerts, m_cornerFace;
  std::vector<Known>                       m_known;
  std::shared_ptr<const HdTables::Topology> m_topology;
  size_t                                   m_dropped = 0;
};

// The tables applied as the GPU applies them: each level refined from the one before (in floats,
// summed in the stencils' order), then its rows' edits added in frames of `cagePositions`.
std::vector<float> refineWithTables(const Refinement& r, std::span<const float> cagePositions, const HdTables* tables);

}  // namespace fv::subdiv
