#pragma once

// DSON morphs: modifier_library entries with `morph.deltas.values = [[vertex, dx, dy, dz], ...]` in
// the base mesh's vertex numbering. The main figure's formulas run in full (pose/erc.hpp,
// pose/figure_driver.hpp); for worn items and props, controller dials with no deltas of their own are
// followed through their linear ERC formulas here, and other formulas are skipped. A port of
// src/lib/loaders/dson/morphs.ts.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/mesh/mesh_build.hpp>

namespace fv::dson {

// A built mesh a morph can target.
struct MeshRef
{
  // index into SceneGraph::meshes
  int                              mesh = -1;
  std::shared_ptr<mesh::BuiltMesh> geo;
  // "file#geometryId" of the geometry this mesh came from, lowercase
  std::string geometryKey;
  // scene node and geometry instance ids: what a scene's morph dials name as their parent
  std::vector<std::string> owners;
  // geometry shells drawn as copies of this mesh (shell.hpp): a morph of it moves them too
  struct Shell
  {
    int                              mesh = -1;
    std::shared_ptr<mesh::BuiltMesh> geo;
  };
  std::vector<Shell> shells;
};

using TargetPtr = std::shared_ptr<const mesh::RenderTarget>;

struct MorphPart
{
  // index into the meshes the morph was made for (MeshRef::mesh)
  int       mesh = -1;
  TargetPtr target;
  std::string label;
  bool        hasLabel = false;
  double      min = 0, max = 1;
};

struct MorphDef
{
  std::string            id;
  std::vector<MorphPart> parts;
};

// Morph name prefixes that are bone-driven correctives or pure controllers in Genesis 8/9.
bool hiddenMorph(const std::string& name);

// Group label from the DAZ folder layout, e.g. ".../Morphs/DAZ 3D/Base Pose Head/x.dsf" -> "Base Pose Head".
std::string groupFromPath(const std::string& file);

// What a morph file says of its dial. DAZ writes it first in the modifier, before the formulas and the
// deltas, so the few KB this takes are all that is read of a file: a figure has thousands.
struct MorphInfo
{
  std::string           id, label;
  // DAZ's parameter group ("/Full Body/People/Feminine"), its region ("Legs") and its presentation
  // ("Modifier/Shape")
  std::string           path, region, kind;
  std::optional<double> min, max;
  // the file marks it as not a dial: not visible, under /Hidden, or an alias of another channel
  bool helper = false;
  bool alias  = false;
};

// Reads it from the start of a morph file's text, which may stop anywhere after it. Empty without a
// modifier.
std::optional<MorphInfo> morphInfoFromText(std::string_view head);
// The same from a file; empty when it can't be read.
std::optional<MorphInfo> morphInfo(const std::filesystem::path& file);

// Converts every morph modifier in a document into targets. `prefer` (indices into `meshes`)
// narrows the target to one scene node's meshes when several copies of the same geometry are loaded.
std::vector<MorphDef> morphsFromDoc(const Doc& doc, const std::string& file, const std::vector<MeshRef>& meshes,
                                    const std::vector<size_t>* prefer = nullptr);

// Loads one morph .dsf and returns its targets. A controller dial is built from the morphs it drives.
std::vector<MorphPart> loadMorphFile(Store& store, const std::string& file, const std::vector<MeshRef>& meshes);

// A morph with vertex deltas that a dial drives, and the factor the dial's value is multiplied by.
struct DrivenMorph
{
  std::string file, id;
  double      weight = 1;
};

// Every delta-carrying morph that setting `file#id` to `weight` moves, including itself when it
// has deltas, following linear ERC links (out = in * k) up to a few levels deep. Weights of morphs
// reached by more than one path are summed.
std::vector<DrivenMorph> drivenMorphs(Store& store, const std::string& file, const std::string& id, double weight = 1, int depth = 0);

}  // namespace fv::dson
