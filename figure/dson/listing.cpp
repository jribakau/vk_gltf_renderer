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
    key += "|" + root.generic_string();
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
    const fs::path  dir = root / fs::path(rel);
    if(!fs::is_directory(dir, ec))
      continue;
    for(fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec))
    {
      std::error_code ec2;
      if(!it->is_regular_file(ec2))
        continue;
      if(lower(it->path().extension().generic_string()) != wanted)
        continue;
      const std::string file = rel + "/" + fs::relative(it->path(), dir, ec2).generic_string();
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
