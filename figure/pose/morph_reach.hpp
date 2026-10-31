#pragma once

// What each dial of a figure's Morphs folder can do, worked out before any of them is moved: the panel
// tells the ones that move skin from the ones that barely do. It runs the formulas the driver would, in
// a graph of its own holding every file's, at each dial's two ends against all of them at rest. A
// morph's size comes from reading its deltas (dson::morphSize). Off the owner's thread, on a copy of
// what the skims said: it touches no figure.

#include <string>
#include <unordered_map>
#include <vector>

#include <figure/dson/morphs.hpp>
#include <figure/morph/morph_set.hpp>

namespace fv::pose {

// One morph file as the audit sees it.
struct MorphFact
{
  std::string     file, stem;
  dson::MorphInfo info;
  // the furthest its deltas move a vertex at a weight of 1, 0 without any (dson::morphSize)
  double size = 0;
};

// Channel name -> what it can do, for dials a panel can show: by id for a scene's dial and by file stem
// for the listing's, as a figure names them. A file that couldn't be told has none.
using DialReach = std::unordered_map<std::string, morph::Reach>;

DialReach auditReach(const std::vector<MorphFact>& facts);

}  // namespace fv::pose
