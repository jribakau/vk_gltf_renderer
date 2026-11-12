#pragma once

// DAZ HD morph detail files (.dhdm): a morph's edits at each subdivision level of the cage, which a
// morph file names in its `hd_url` beside its base-resolution deltas. How they are placed and applied
// is figure/subdiv/hd_edits.hpp.
//
// Little-endian, undocumented by DAZ (layout after https://github.com/edolstra/dhdm, src/dhdm.cc):
//   header   u32 0xd0d0d0d0, u32 levels, f32 1.0, u32 levels
//   a level  u32 cage faces, u32 level (1, 2, ...), u32 edits, u32 bytes; then, per edited face,
//            u32 face, u32 edits of it, and the edits: below level 4, f32 a, u8 index, u8 (level +
//            1) * 16, f32 b, f32 c; from level 4, f32 a, u8 0, u16 index, u8 (level + 1) * 16, f32
//            b, f32 c. Read as 16 bits (a one-byte index is the high byte), the index holds, from the
//            top, the descendant face (2 bits a level) and the corner (2 bits).

#include <stdexcept>
#include <string_view>

#include <figure/subdiv/hd_edits.hpp>

namespace fv::dson {

struct Dhdm
{
  uint32_t           faceCount = 0;  // the cage's faces, as the file states them
  subdiv::LevelEdits levels;
};

// Throws std::runtime_error on a file that isn't one, or ends early.
Dhdm parseDhdm(std::string_view bytes);

}  // namespace fv::dson
