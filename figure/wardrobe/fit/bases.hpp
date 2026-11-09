#pragma once

// The zeroed base mesh of a Genesis generation, loaded from the user's library when it is
// installed. The shape correction carries it through the same bone transfer as the garment and
// measures how far the target body lies from it. A missing base is not an error: the garment keeps
// the bone transfer alone. A port of src/lib/wardrobe/fit/bases.ts.

#include <memory>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace fv::wardrobe {

struct BaseCage
{
  std::vector<float>    positions;
  std::vector<uint32_t> tris;
  // each triangle's surface (an index into `surfaces`, the file's material groups: "Torso", "Teeth")
  std::vector<uint16_t>    triSurface;
  std::vector<std::string> surfaces;
  // skin influences per vertex; `bone` indexes `skeleton`
  mesh::SkinInput skin;
  // the base figure's bones (names, not ids), with their file centres
  std::vector<SkelBone> skeleton;
  // per vertex, its UV (u, v), where the reader keeps them (a Skyrim body's: a cut body is matched to
  // its whole body by them); empty otherwise
  std::vector<float> uvs;
};

// The installed base cage for `generation`, or null when none of its files are in the library.
std::shared_ptr<const BaseCage> loadBaseCage(dson::Store& store, Generation generation, const std::string& gender);
// A figure file's base cage, read once (null when it holds none): the reference body maps are made on.
std::shared_ptr<const BaseCage> loadBaseCageFile(dson::Store& store, const std::string& file);
// The files loadBaseCage tries, in order: `gender`'s ("female", "male") first.
std::vector<std::string> baseFiles(Generation generation, const std::string& gender);
// Forget what was read, a base found missing too: an install or an uninstall changed the library.
void clearBaseCages();

}  // namespace fv::wardrobe
