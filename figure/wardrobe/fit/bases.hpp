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
  // skin influences per vertex; `bone` indexes `skeleton`
  mesh::SkinInput skin;
  // the base figure's bones (names, not ids), with their file centres
  std::vector<SkelBone> skeleton;
};

// The installed base cage for `generation`, or null when none of its files are in the library.
std::shared_ptr<const BaseCage> loadBaseCage(dson::Store& store, Generation generation, const std::string& gender);

}  // namespace fv::wardrobe
