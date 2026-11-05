/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// Where the host takes a DAZ texture from when it is not the cooked set: the cooker, an image composed
// here, one texel until its cook lands, or the source as it is. Pure but for trimFolder, so it is tested
// without a device or a texture server (tests/test_fv_textures.cpp).
//

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

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
  eFlat,     // one texel that leaves the material's factors as they are, until its cook lands
  eNone,     // nothing shows the right thing
};

// Whether a request's channels are derived from its source: a roughness or a metal map into glTF's
// metallic-roughness, a height map into a normal map, or a weight map's grey into every channel
// ("mr", "mt", "bn", "wa"). The source as it is would show the wrong thing.
inline bool derivedChannels(const std::string& channels)
{
  return channels == "mr" || channels == "mt" || channels == "bn" || channels == "wa";
}

// `derived`: channels derived from the source (derivedChannels);
// `laid`: a cutout, makeup, an iris colour or layers over the source; `cooker`: a texture server is
// there. A cutout alone changes the picture too much to go without, so the cooker folds it in,
// unless `original` (the Original textures setting): then only a format the renderer cannot read is
// cooked, and the rest is composed here, uncompressed. With the cooker a derived map stands in as one
// texel until its cook lands: composed here it was a full-size PNG written to disk and decoded again,
// for the seconds until then.
inline Fallback fallbackOf(const std::string& rel, bool derived, bool laid, bool cooker, bool original)
{
  const bool cookLaid = cooker && !original;
  if(cooker && (!readable(rel) || (cookLaid && !derived && laid)))
    return Fallback::eCook;
  if(cookLaid && derived)
    return Fallback::eFlat;
  if((derived || (!cookLaid && laid)) && readable(rel))
    return Fallback::eCompose;
  return derived ? Fallback::eNone : Fallback::eSource;
}

// The files directly in `dir` removed by their write time, oldest first, until the rest come to
// `capBytes` or less. A file whose name starts with `keep` stays, as does one that can't be removed (in
// use). Returns the bytes removed.
inline uint64_t trimFolder(const std::filesystem::path& dir, uint64_t capBytes, const std::wstring& keep = {})
{
  struct Entry
  {
    std::filesystem::file_time_type time;
    uint64_t                        bytes = 0;
    std::filesystem::path           path;
  };
  std::vector<Entry> entries;
  uint64_t           total = 0;
  std::error_code    ec;
  for(std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
  {
    std::error_code e;
    if(!it->is_regular_file(e) || (!keep.empty() && it->path().filename().wstring().rfind(keep, 0) == 0))
      continue;
    Entry entry{it->last_write_time(e), it->file_size(e), it->path()};
    if(e)
      continue;
    total += entry.bytes;
    entries.push_back(std::move(entry));
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.time < b.time; });
  uint64_t removed = 0;
  for(const Entry& entry : entries)
  {
    if(total <= capBytes)
      break;
    std::error_code e;
    if(std::filesystem::remove(entry.path, e))
    {
      total -= entry.bytes;
      removed += entry.bytes;
    }
  }
  return removed;
}

}  // namespace fv::textures
