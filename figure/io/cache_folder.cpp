#include <figure/io/cache_folder.hpp>

#include <algorithm>
#include <vector>

namespace fv::io {

void trimCacheFolder(const std::filesystem::path& dir, std::string_view extension, uintmax_t budget)
{
  std::error_code ec;
  struct Entry
  {
    std::filesystem::path           path;
    uintmax_t                       size;
    std::filesystem::file_time_type time;
  };
  std::vector<Entry> files;
  uintmax_t          total = 0;
  for(const auto& e : std::filesystem::directory_iterator(dir, ec))
  {
    if(e.path().extension() != extension)
      continue;
    std::error_code fe;
    const uintmax_t size = e.file_size(fe);
    const auto      time = e.last_write_time(fe);
    if(fe)
      continue;
    files.push_back({e.path(), size, time});
    total += size;
  }
  if(total <= budget)
    return;
  std::sort(files.begin(), files.end(), [](const Entry& a, const Entry& b) { return a.time < b.time; });
  for(const Entry& f : files)
  {
    if(total <= budget / 4 * 3)
      break;
    std::error_code re;
    if(std::filesystem::remove(f.path, re))
      total -= f.size;
  }
}

void touchCacheFile(const std::filesystem::path& file)
{
  std::error_code ec;
  std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(), ec);
}

}  // namespace fv::io
