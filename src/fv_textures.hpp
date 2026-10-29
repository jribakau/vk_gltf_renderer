/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// Where the host takes a DAZ texture from when it is not the cooked (BC7) set: the cooker, an image
// composed here, or the source as it is. Pure, so it is tested without a device or a texture server
// (tests/test_fv_textures.cpp).
//

#include <algorithm>
#include <cctype>
#include <string>

namespace fv::textures {

// Whether the renderer reads this image itself (else only the cooker can: .tif, .bmp, ...).
inline bool readable(const std::string& path)
{
  std::string ext = path.substr(path.find_last_of('.') + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "webp";
}

enum class Fallback
{
  eCook,     // cooked unflipped, and waited for
  eCompose,  // composed into a PNG here
  eSource,   // the source image as it is
  eNone,     // nothing shows the right thing
};

// Whether a request's channels are derived from its source: a roughness or a metal map into glTF's
// metallic-roughness, or a height map into a normal map ("mr", "mt", "bn"). The source as it is would
// show the wrong thing.
inline bool derivedChannels(const std::string& channels)
{
  return channels == "mr" || channels == "mt" || channels == "bn";
}

// `derived`: channels derived from the source (derivedChannels);
// `laid`: a cutout, makeup, an iris colour or layers over the source; `cooker`: a texture server is
// there. A cutout alone changes the picture too much to go without, so the cooker folds it in,
// unless `original` (the Original textures setting): then only a format the renderer cannot read is
// cooked, and the rest is composed here, uncompressed.
inline Fallback fallbackOf(const std::string& rel, bool derived, bool laid, bool cooker, bool original)
{
  const bool cookLaid = cooker && !original;
  if(cooker && (!readable(rel) || (cookLaid && !derived && laid)))
    return Fallback::eCook;
  if((derived || (!cookLaid && laid)) && readable(rel))
    return Fallback::eCompose;
  return derived ? Fallback::eNone : Fallback::eSource;
}

}  // namespace fv::textures
