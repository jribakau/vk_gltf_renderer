#pragma once

// A body type's map held in memory, for the wears that read it (docs/wardrobe-plan.md, step 4a). Made
// through the maps' own cache on disk the first time; again when the library's base figures are read
// again (an install or an uninstall), since a map is of them.

#include <memory>
#include <string>

#include <figure/bodymap/clone.hpp>

namespace fv::garment {

// The map of `generation`'s figure onto the reference: by Genesis 9's clone morph of it, or, for Genesis
// 9, the reference onto itself. Null when there is none (Skyrim, or a base or clone that isn't
// installed).
std::shared_ptr<const bodymap::TypeMap> typeMap(dson::Store& store, wardrobe::Generation generation, const std::string& gender);

// The reference's own map: every vertex at itself, both ways, with no frame turns.
bodymap::TypeMap referenceMap(const bodymap::Body& reference);

}  // namespace fv::garment
