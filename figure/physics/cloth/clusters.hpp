#pragma once

// The cloth model cut into clusters for the GPU (docs/physics-plan.md, "The GPU path", step 13a): each
// cluster is a patch of free vertices, grown over the surface, that a workgroup holds in shared memory
// for a whole sweep, with the pinned vertices its rows read as ghosts it never writes. A cluster's rows
// are coloured among themselves (no two of a colour share a vertex it writes), so the workgroup runs a
// colour at a time between group barriers.
//
// The rows between two clusters' free vertices cross. With `halo`, each of the two clusters holds such a
// row, the other's end as a halo vertex it reads where it was at the sweep's start and never writes, and
// moves its own end by its share: one dispatch a sweep, and both shares from one state add up to the
// row's correction. Without, they are coloured over the whole model and run after the clusters, as the
// CPU runs the rows crossing its blocks.
//
// Every cluster has the same number of colours in each group, some of them empty, so one loop of the
// kernel serves all: the stretch colours, then the reach links', then the rest's. A sweep that holds
// length takes the first two groups; a full sweep the first and the third (as the CPU's and today's
// GPU sweeps: reach links only in the stretch sweeps).
//
// Built deterministically from the model alone: the same clusters on any machine.

#include <cstdint>
#include <vector>

#include <figure/physics/cloth/model.hpp>

namespace fv::physics::cloth {

struct Clusters
{
  // cluster c's vertices, global indices: verts[vertStart[c] .. vertStart[c + 1]), the first owned[c]
  // of them its own free vertices (in order), the rest ghosts and halo vertices, which it only reads
  std::vector<int> vertStart{0}, verts, owned;
  // a row as a cluster runs it: its ends as local indices into the cluster's vertices, and the model's
  // row (model.edges) it is, for the rest of what it holds
  struct Row
  {
    uint16_t a, b;
    int      edge;
  };
  std::vector<Row> rows;
  // cluster c's colour k: rows[colourStart[c * colours() + k] .. colourStart[c * colours() + k + 1])
  std::vector<int> colourStart;
  int              stretchColours = 0, reachColours = 0, restColours = 0;
  // the soft vertices (tethered to the skin) each cluster owns: local index, and the soft index
  // (model.softVerts) for their multipliers; cluster c's are soft[softStart[c] .. softStart[c + 1])
  std::vector<std::pair<uint16_t, int>> soft;
  std::vector<int>                      softStart{0};
  // the rows that cross clusters, coloured over the model, in the same three groups (none with `halo`)
  std::vector<int> crossing, crossingStart{0};
  int              crossingStretch = 0, crossingReach = 0, crossingRest = 0;
  // with `halo`: how many rows cross, each held by both its clusters
  size_t haloRows = 0;
  // the most vertices (owned and ghosts) one cluster holds
  int mostLocal = 0;

  int  count() const { return int(owned.size()); }
  int  colours() const { return stretchColours + reachColours + restColours; }
};

// Clusters of about `size` free vertices, none holding more than `mostLocal` vertices with its ghosts.
// Empty (count() 0) when a vertex has more rows of one kind than a colour mask can tell apart.
// With `halo`, a cluster's halo vertices count toward `mostLocal` only once all are cut; mostLocal() then
// says whether it fits.
Clusters buildClusters(const Model& model, int size, int mostLocal, bool halo = false);

}  // namespace fv::physics::cloth
