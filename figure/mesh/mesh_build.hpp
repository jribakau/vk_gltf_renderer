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
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <optional>
#include <span>
#include <vector>

#include <figure/subdiv/catmark.hpp>

namespace fv::mesh {

// A grown hair wandering off its course, `amount` cm at `frequency` cycles per cm; `depth` layers of
// it, each `ratio` as strong and `mult` as frequent as the last.
struct Scraggle
{
  float amount = 0, frequency = 0;
  int   depth = 1;
  float ratio = 0, mult = 1;
};

// DAZ's pre-render (PR) hairs on one material's strands: `perGuide` more strands around each line,
// their roots up to `radius` (cm) from its root and their spread scaled by 1 + `tipSeparation` at the
// tip, shaped by `bias` and `gain` along the strand.
//
// Or `density` hairs per cm² over a surface ("Target Surfaces" mode, Lines::surface): on the polygons
// whose corners all carry a guide of this material, each hair between those guides ("Surface Lattice"),
// and following the nearest one as far as the single guide settings say. `points` is the least a hair
// is drawn with, `segment` the length (cm) of the pieces DAZ would cut it into, which frizz is relative to.
// DAZ documents none of the look's settings: the slider's unit is read from its label and range, and
// what each does to a strand is uncalibrated against DAZ Studio.
struct Growth
{
  int      perGuide      = 0;
  float    radius        = 0;
  float    tipSeparation = 0;
  float    bias          = 0.5f;
  float    gain          = 0.5f;
  uint32_t seed          = 0;

  float density     = 0;
  int   points      = 8;
  float segment     = 1;
  float singleGuide = 0, singleBase = 0, singleTip = 0, singleBias = 0.5f;
  // how much shorter a hair is: up to `randomLength` at random, and `reduceLength` always (cm)
  float    randomLength = 0, reduceLength = 0;
  uint32_t lengthSeed   = 0;
  // a share of `segment` a point moves at random, from the root's to the tip's
  float    frizzBase = 0, frizzTip = 0;
  uint32_t frizzSeed = 0;
  std::array<Scraggle, 2> scraggle{};
};

// The mesh a "Target Surfaces" strand hair grows from, in the hair's own space.
struct GrowthSurface
{
  std::vector<float> positions;
  // polygon i is polyVerts[polyStart[i] .. polyStart[i + 1])
  std::vector<int32_t> polyStart{0}, polyVerts;
  // which geometry this is, for a key on what a build reads
  std::string key;
};

struct Lines
{
  // Each polyline's points in `verts`, plus the end; a material per line.
  std::vector<int32_t> start, verts, material;
  // the PR hairs grown per material; empty grows none
  std::vector<Growth> grow;
  // what a material with a density grows from; without one it grows nothing
  std::shared_ptr<const GrowthSurface> surface;
  // the share of the lines drawn, the file's and the grown alike. The cage keeps them all: the skin,
  // morphs and a simulation run on them as ever
  float keep = 1;
  // the pieces each segment is drawn in (strandRibbons), 1 or 2: one piece is half the triangles and
  // vertices for the same shape, as the second piece's point lies on the segment's chord, but looser
  // boxes for the ray tracer, so it is the memory fit's first step down for a hair (fv::WearQuality)
  int pieces = 2;
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
  // the file marks it SubD: how finely it is drawn is a choice (level)
  bool                 subdivisionSurface = false;
  subdiv::BoundaryInterpolation boundary = subdiv::BoundaryInterpolation::eEdges;
  // Polylines (strand hair), drawn as ribbons when the mesh has no polygons.
  std::optional<Lines> lines;
};

// Strand ribbons, per drawn vertex: the unit direction its edge of the ribbon lies in from the
// strand, where along the strand it is (0 root, 1 tip) and its line's material. Built at zero
// width; widened once the materials give the widths.
struct Ribbon
{
  // Per drawn vertex, until widenRibbons has made `offsets` of them and let them go
  std::vector<float>   side;
  std::vector<float>   along;
  std::vector<int32_t> material;
  // where along its strand the width has grown in from nothing at the root
  std::vector<float> rootFade;
  // a grown strand's offset from its guide, xyz per drawn vertex, in the rest shape (empty when the
  // lines grow none)
  std::vector<float> shift;
  // Per render vertex once widened (empty before), xyz: how far it is from its strand point, off its
  // guide and across by its width, cm. All that is read of a ribbon from then on.
  std::vector<float> offsets;
  // Median distance between neighbouring roots, cm.
  float spacing = 0;
  // Root and tip per material, cm, as it was widened: what a share of its strands has to cover (remade)
  std::vector<std::array<double, 2>> widths;
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
  // refined: per render vertex, the triangles it sums for its normal, in triangle order (normalStart
  // has one more: vertex v's are normalTriangles[normalStart[v] .. normalStart[v + 1]))
  std::vector<uint32_t> normalStart, normalTriangles;
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
  bool                                       subdivisionSurface = false;  // MeshInput::subdivisionSurface
  Bounds                                     bounds;
  // cage polygons a worn item hides (a geograft replaces them), by the item's key
  std::map<std::string, std::vector<int32_t>> hiddenPolys;
  // What it was built from, when it can be drawn finer or coarser (subdivided, or strand hair): remade. Its
  // positions are left out, cagePositions having them as they are now.
  std::shared_ptr<const MeshInput> source;

  bool refined() const { return refinement != nullptr; }

  // A strand hair the GPU draws from its cage reads none of its drawn positions, normals or splits once
  // fed: letGoOfDrawn lets go of them (290 MB of a 9.5M-vertex hair) and ensureDrawn makes them again
  // for whatever reads them later (an export, the CPU morph, a record that holds them). Meanwhile how
  // many drawn vertices there are, and whether they have normals, stay as they were.
  bool     drawnLetGo   = false;
  bool     letGoNormals = false;
  uint32_t drawnRev     = 0;  // moves on with each letting go: state kept over the drawn vertices starts again
  size_t   drawnCount() const { return drawnLetGo ? renderIndex.size() : position.size() / 3; }
  bool     hasNormals() const { return drawnLetGo ? letGoNormals : normal.size() == position.size(); }
};

BuiltMesh build(const MeshInput& m);
// The UV index of every polygon corner (as polyVerts): the vertex's own, unless a seam override gives the
// corner another.
std::vector<int32_t> cornerUvs(const MeshInput& m);
// The mesh again at another level, or with another share of its strands or pieces a segment (Lines::pieces),
// over `cage`: everything that lives on the cage stays true of it (morphs, a pose, a drape, bindings). Null
// when it has no source.
std::shared_ptr<BuiltMesh> remade(const BuiltMesh& mesh, std::span<const float> cage, int level, float keep, int pieces = 2);
// Roughly what building it takes of system memory at its peak: from the counts, before it is built, to
// tell whether several can go side by side.
uint64_t buildBytes(const MeshInput& m);
// What a built mesh holds of system memory.
uint64_t ramBytes(const BuiltMesh& m);

// Opens a strand mesh's zero-width ribbons to `widths[material]` (root, tip in cm), then welds
// normals across them.
void widenRibbons(BuiltMesh& m, const std::vector<std::array<double, 2>>& widths);

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
  // DAZ's skin settings: the share of a vertex's skinning that is dual quaternion rather than linear
  // blend (0 linear, 1 dual quaternion), `dualQuat` for every cage vertex unless `dualQuatMap` has
  // one per cage vertex
  float              dualQuat = 0;
  std::vector<float> dualQuatMap;
};

struct SkinOutput
{
  std::vector<uint16_t> skinIndex;   // four per drawn vertex
  std::vector<float>    skinWeight;  // four per drawn vertex, summing to 1
  // DAZ's skinning: eight per drawn vertex, strongest first where there are more than four, summing
  // to 1 (a slot with no influence weighs 0); empty when no vertex has more than four, the four
  // above then being the same influences
  std::vector<uint16_t> skinIndex8;
  std::vector<float>    skinWeight8;
  // the dual quaternion share (SkinInput::dualQuat), per drawn vertex when the map is not uniform
  float              dualQuat = 0;
  std::vector<float> dualQuatShare;
};

// The four strongest influences per drawn vertex, renormalised, and the eight strongest. On a
// subdivided mesh the influences are refined like positions first (eight kept per level), and a
// dual quaternion map is refined with them.
SkinOutput skin(const BuiltMesh& mesh, const SkinInput& s);

struct CageDeltas
{
  std::vector<int32_t> verts;
  std::vector<float>   deltas;  // xyz per entry of verts
};

// A morph target: its cage deltas, and its drawn vertices' deltas made from them the first time
// they are asked for. A mesh the GPU refines from its cage is moved by the cage
// deltas alone, so its targets never make theirs (daz-viewer docs/gpu-figures-plan.md F5).
struct RenderTarget
{
  // drawn vertices and their xyz deltas
  struct Drawn
  {
    std::vector<uint32_t> indices;
    std::vector<float>    deltas;
  };

  // the cage deltas, out-of-range vertices dropped
  std::vector<uint32_t> cageIndices;
  std::vector<float>    cageDeltas;

  // Refined onto `mesh`'s render vertices and spread to every UV split, once for a mesh of its size, by
  // whichever thread asks first (the others wait for it). The size is how a mesh made again shows.
  const Drawn& drawn(const BuiltMesh& mesh) const;
  // The drawn deltas, when made or given; else null.
  const Drawn* drawnIfMade() const;
  // Gives the drawn deltas outright, before anything asked for them (a target made by hand).
  void setDrawn(Drawn d);
  // Whether a cage delta, or a given drawn delta, is not zero.
  bool moves() const;

private:
  struct Lazy
  {
    std::mutex          mutex;
    std::atomic<size_t> madeFor{0};  // 0: nothing made; else the drawn positions' size, plus one
    Drawn               drawn;
  };
  static constexpr size_t kGiven = SIZE_MAX;  // madeFor of what setDrawn gave: good for any
  std::unique_ptr<Lazy> m_lazy = std::make_unique<Lazy>();
};

// A morph target from cage deltas, those of vertices the cage doesn't have left out.
RenderTarget target(int cageVertices, const CageDeltas& d);
RenderTarget target(const BuiltMesh& mesh, const CageDeltas& d);
// The same from cage vertices already in range, taken over rather than copied (a projection onto a
// hair's strands lists a million).
RenderTarget target(const BuiltMesh& mesh, std::vector<uint32_t> cageIndices, std::vector<float> cageDeltas);

// Whether a mesh's drawn vertices can be made from its cage alone, as the GPU makes them: subdivided,
// or strand ribbons once widened.
bool drawnFromCage(const BuiltMesh& mesh);

// Replaces the cage positions and returns the delta of the drawn vertices (xyz each), so a fit can
// move the rest shape without discarding morphs already applied.
std::vector<float> shiftCage(BuiltMesh& mesh, std::span<const float> positions);

// Lets go of a strand ribbon's drawn positions, normals and splits: only one the GPU draws from its cage
// (refined, its widths made into offsets). False when it isn't one, or has let go already.
bool letGoOfDrawn(BuiltMesh& mesh);
// Makes them again when they were let go, from the cage as it is (`cagePositions`): refined and moved
// by the ribbons' offsets as a drawer makes them, the splits from renderIndex in the order build made
// them, the normals welded, as widenRibbons left them to float rounding. Any thread; it writes into
// the mesh, whose const-ness only says it is read: what it makes is what was there.
void ensureDrawn(const BuiltMesh& mesh);

}  // namespace fv::mesh
