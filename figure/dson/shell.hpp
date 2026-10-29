#pragma once

// DAZ Studio geometry shells (studio/node/shell): a node drawn as a copy of another node's mesh, its
// geometry in the file empty. The copy follows the source (its skin, its morphs), carries materials of
// its own, may leave some of the source's surfaces out (material_group_<name>_vis,
// facet_group_<name>_vis), and is usually pushed off the source's surface along its normals by a push
// modifier of its own ("Offset Distance (cm)"). Hair products layer cards this way: the source holds
// the strands' cards once and each shell draws them again a few millimetres out with other maps.

#include <optional>
#include <set>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/mesh/mesh_build.hpp>

namespace fv::dson {

struct Shell
{
  std::string           source;  // the scene node whose mesh it copies
  std::set<std::string> hiddenMaterials, hiddenFacets;  // the groups whose polygons it leaves out
  double                offset = 0;  // cm along the normals (its push modifier)
};

// The shell a scene node is, or none: from the scene node, its library definition `def` and the
// scene's modifiers (a push modifier there points at its library entry in `file` or elsewhere).
std::optional<Shell> parseShell(const Json& node, const Json* def, const Json* sceneModifiers, Store& store, const std::string& file);

// The source's mesh input made the shell's: pushed `offset` along the cage's area-weighted normals
// (the push modifier before smoothing), then the polygons of hidden groups left out, their UV seam
// overrides with them. `polyGroup`: each polygon's facet group, indices into `facetGroups`.
void shellInput(mesh::MeshInput& m, const Shell& shell, const std::vector<std::string>& materialGroups,
                const std::vector<std::string>& facetGroups, const std::vector<int32_t>& polyGroup);

}  // namespace fv::dson
