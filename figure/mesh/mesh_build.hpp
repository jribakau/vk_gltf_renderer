#pragma once

// The heavy part of turning a DSON mesh into drawable buffers: splitting vertices along UV seams,
// subdividing SubD meshes, welding normals, refining skin weights and morph deltas from the cage
// onto the drawn vertices. A port of src/lib/workers/meshJobs.ts.
//
// Two vertex spaces exist (see src/lib/loaders/dson/geometry.ts):
//   - cage vertices: the DSON file's, `dsonVertexCount` of them;
//   - render vertices: the cage's without subdivision, the refined ones with it. Drawn vertices are
//     render vertices split along UV seams, and normals are welded over render vertices.

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <optional>
#include <span>
#include <vector>

#include <figure/subdiv/catmark.hpp>

namespace fv::mesh {

struct Lines
{
  // Each polyline's points in `verts`, plus the end; a material per line.
  std::vector<int32_t> start, verts, material;
};

struct MeshInput
{
  // Cage (DSON) vertex positions, xyz.
  std::vector<float> positions;
  // Each polygon's corners in polyVerts, plus the end.
  std::vector<int32_t> polyStart, polyVerts, polyMaterial;
  int                  materialCount = 1;
  // UVs, two floats per UV index; empty without a UV set.
  std::vector<float> uvs;
  // UV seam overrides as (polygon, vertex, uv index) triplets.
  std::vector<int32_t> overrides;
  int                  level    = 0;
  subdiv::BoundaryInterpolation boundary = subdiv::BoundaryInterpolation::eEdges;
  // Polylines (strand hair), drawn as ribbons when the mesh has no polygons.
  std::optional<Lines> lines;
};

// Strand ribbons, per drawn vertex: the unit direction its edge of the ribbon lies in from the
// strand, where along the strand it is (0 root, 1 tip) and its line's material. Built at zero
// width; widened once the materials give the widths.
struct Ribbon
{
  std::vector<float>   side;
  std::vector<float>   along;
  std::vector<int32_t> material;
  // Median distance between neighbouring roots, cm.
  float spacing = 0;
};

struct Group
{
  uint32_t start = 0;  // in indices
  uint32_t count = 0;
  int      materialIndex = 0;
};

struct Bounds
{
  std::array<float, 3> min{}, max{};
  float                radius = 0;
};

struct BuiltMesh
{
  // drawn vertices
  std::vector<float> position, uv, normal;
  // triangles grouped by material, on drawn vertices (with every polygon shown), and each one's cage polygon
  std::vector<uint32_t> fullIndex;
  std::vector<Group>    groups;
  std::vector<uint32_t> triPoly;
  // drawn vertex -> render vertex; render vertex -> first drawn vertex, then chained through nextSplit
  std::vector<uint32_t> renderIndex;
  std::vector<int32_t>  firstSplit, nextSplit;
  int                   renderVertexCount = 0;
  // triangles fanned per face on render vertices, for welded normals
  std::vector<uint32_t> renderTris;
  // triangles of the cage on DSON vertices, for fitting worn items to this mesh
  std::vector<uint32_t> dsonTris;
  int                   dsonVertexCount = 0;
  int                   polyCount       = 0;
  // the cage's polygon loops as the file has them (polygon i is cagePolyVerts[cagePolyStart[i] ..
  // cagePolyStart[i + 1])), for cloth constraints on a mesh whose file carries no simulation settings
  std::vector<int32_t> cagePolyStart{0}, cagePolyVerts;
  // strand hair's polylines over the cage, root first (line i is cageLineVerts[cageLineStart[i] ..
  // cageLineStart[i + 1])), for binding a strand to the body along its length; empty otherwise
  std::vector<int32_t> cageLineStart, cageLineVerts;
  // the cage positions (updated by shiftCage) and its subdivision, null when drawn unrefined
  std::vector<float>                         cagePositions;
  std::shared_ptr<const subdiv::Refinement> refinement;
  std::optional<Ribbon>                      ribbon;
  Bounds                                     bounds;
  // cage polygons a worn item hides (a geograft replaces them), by the item's key
  std::map<std::string, std::vector<int32_t>> hiddenPolys;

  bool refined() const { return refinement != nullptr; }
};

BuiltMesh build(const MeshInput& m);

// The triangles drawn: fullIndex without the polygons any worn item hides, and its groups.
struct Drawn
{
  std::vector<uint32_t> index;
  std::vector<Group>    groups;
};
Drawn drawnTriangles(const BuiltMesh& mesh);

// Bone influences per cage vertex as compressed rows: vertex v's are [start[v], start[v + 1]).
struct SkinInput
{
  std::vector<int32_t> start, bone;
  std::vector<float>   weight;
  // Bone that takes vertices with no weights.
  int rootBone = 0;
};

struct SkinOutput
{
  std::vector<uint16_t> skinIndex;   // four per drawn vertex
  std::vector<float>    skinWeight;  // four per drawn vertex, summing to 1
};

// The four strongest influences per drawn vertex, renormalised. On a subdivided mesh the influences
// are refined like positions first (eight kept per level).
SkinOutput skin(const BuiltMesh& mesh, const SkinInput& s);

struct CageDeltas
{
  std::vector<int32_t> verts;
  std::vector<float>   deltas;  // xyz per entry of verts
};

struct RenderTarget
{
  // drawn vertices and their xyz deltas
  std::vector<uint32_t> indices;
  std::vector<float>    deltas;
  // the cage deltas it was made from, out-of-range vertices dropped
  std::vector<uint32_t> cageIndices;
  std::vector<float>    cageDeltas;
};

// A morph target from cage deltas: refined onto the render vertices, then spread to every UV split.
RenderTarget target(const BuiltMesh& mesh, const CageDeltas& d);

// Replaces the cage positions and returns the delta of the drawn vertices (xyz each), so a fit can
// move the rest shape without discarding morphs already applied.
std::vector<float> shiftCage(BuiltMesh& mesh, std::span<const float> positions);

}  // namespace fv::mesh
