#include <figure/dson/listing.hpp>

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <set>

namespace fv::dson {

namespace fs = std::filesystem;

namespace {

constexpr auto kKeep = std::chrono::minutes(2);

struct Kept
{
  std::chrono::steady_clock::time_point at;
  std::vector<std::string>              files;
};

std::mutex                    g_mutex;
std::map<std::string, Kept>   g_kept;

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

}  // namespace

std::vector<std::string> listFiles(const Library& library, const std::string& rel, const std::string& ext)
{
  std::string key = rel + "|" + ext;
  for(const fs::path& root : library.roots())
  {
    // not generic_string: a folder outside the system's code page throws there, and fails every load
    const std::u8string where = root.generic_u8string();
    key += "|" + std::string(where.begin(), where.end());
  }
  key = lower(key);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto                        it = g_kept.find(key);
    if(it != g_kept.end() && std::chrono::steady_clock::now() - it->second.at < kKeep)
      return it->second.files;
  }
  std::vector<std::string> out;
  std::set<std::string>    seen;
  const std::string        wanted = lower(ext);
  for(const fs::path& root : library.roots())
  {
    std::error_code ec;
    // library paths are UTF-8; a narrow path or string is the system's code page, and throws on a
    // name it can't hold (MantaMade's M7UP Top has one among its morphs, and would not load)
    const fs::path  dir = root / fs::path(std::u8string(reinterpret_cast<const char8_t*>(rel.data()), rel.size()));
    if(!fs::is_directory(dir, ec))
      continue;
    for(fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec))
    {
      std::error_code ec2;
      if(!it->is_regular_file(ec2))
        continue;
      const auto utf8 = [](const fs::path& p) {
        const std::u8string u = p.generic_u8string();
        return std::string(u.begin(), u.end());
      };
      if(lower(utf8(it->path().extension())) != wanted)
        continue;
      const std::string file = rel + "/" + utf8(fs::relative(it->path(), dir, ec2));
      if(seen.insert(lower(file)).second)
        out.push_back(file);
    }
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kept[key] = {std::chrono::steady_clock::now(), out};
  return out;
}

void clearListings()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kept.clear();
}

}  // namespace fv::dson
