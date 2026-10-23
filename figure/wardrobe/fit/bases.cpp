#include <figure/wardrobe/fit/bases.hpp>

#include <algorithm>
#include <map>
#include <mutex>
#include <regex>

namespace fv::wardrobe {

namespace {

std::vector<std::string> filesOf(Generation g)
{
  switch(g)
  {
    case Generation::eG9: return {"data/DAZ 3D/Genesis 9/Base/Genesis9.dsf"};
    case Generation::eG8: return {"data/DAZ 3D/Genesis 8/Female/Genesis8Female.dsf", "data/DAZ 3D/Genesis 8/Male/Genesis8Male.dsf"};
    case Generation::eG81:
      return {"data/DAZ 3D/Genesis 8/Female 8_1/Genesis8_1Female.dsf", "data/DAZ 3D/Genesis 8/Male 8_1/Genesis8_1Male.dsf"};
    case Generation::eG3: return {"data/DAZ 3D/Genesis 3/Female/Genesis3Female.dsf", "data/DAZ 3D/Genesis 3/Male/Genesis3Male.dsf"};
    case Generation::eG2: return {"data/DAZ 3D/Genesis 2/Female/Genesis2Female.dsf", "data/DAZ 3D/Genesis 2/Male/Genesis2Male.dsf"};
  }
  return {};
}

std::vector<std::string> order(Generation g, const std::string& gender)
{
  const auto files = filesOf(g);
  if(gender.empty())
    return files;
  const std::regex         want(gender == "female" ? "female" : "male(?!.*female)", std::regex::icase);
  std::vector<std::string> out;
  for(const std::string& f : files)
    if(std::regex_search(f, want))
      out.push_back(f);
  for(const std::string& f : files)
    if(!std::regex_search(f, want))
      out.push_back(f);
  return out;
}

std::shared_ptr<const BaseCage> build(const dson::Doc& doc)
{
  using dson::get;
  const dson::Json* geos = get(&doc.json, "geometry_library");
  if(!geos || !geos->is_array())
    return nullptr;
  const dson::Json* best      = nullptr;
  int               bestCount = 0;
  for(const dson::Json& g : *geos)
  {
    const int n = doc.rows(get(get(&g, "vertices"), "values"))->count;
    if(n > bestCount)
    {
      best      = &g;
      bestCount = n;
    }
  }
  if(!best || !bestCount)
    return nullptr;
  auto cage = std::make_shared<BaseCage>();
  {
    const auto verts = doc.rows(get(get(best, "vertices"), "values"));
    cage->positions.resize(size_t(verts->count) * 3);
    for(int v = 0; v < verts->count; v++)
      for(int c = 0; c < 3; c++)
        cage->positions[size_t(v) * 3 + size_t(c)] = float(verts->cell(v, c));
    const auto polys = doc.rows(get(get(best, "polylist"), "values"));
    for(int p = 0; p < polys->count; p++)
    {
      const int s = polys->start[p] + 2, e = polys->start[p + 1];
      if(e - s < 3)
        continue;
      const uint32_t a = uint32_t(polys->data[s]);
      for(int k = s + 2; k < e; k++)
        cage->tris.insert(cage->tris.end(), {a, uint32_t(polys->data[k - 1]), uint32_t(polys->data[k])});
    }
  }
  // the skeleton: bone nodes with their names, parents and centres
  std::map<std::string, int> byId;
  if(const dson::Json* nodes = get(&doc.json, "node_library"); nodes && nodes->is_array())
  {
    std::map<std::string, std::string> nameOf;
    for(const dson::Json& n : *nodes)
      if(dson::getString(&n, "type") == "bone")
        nameOf[dson::getString(&n, "id")] = dson::getString(&n, "name", dson::getString(&n, "id"));
    for(const dson::Json& n : *nodes)
    {
      if(dson::getString(&n, "type") != "bone")
        continue;
      const std::string parentRef = dson::getString(&n, "parent");
      const std::string pid       = parentRef.empty() ? "" : dson::decodeUri(parentRef.substr(parentRef.find_last_of('#') == std::string::npos ? 0 : parentRef.find_last_of('#') + 1));
      SkelBone          bone;
      bone.name = dson::getString(&n, "name", dson::getString(&n, "id"));
      if(auto it = nameOf.find(pid); it != nameOf.end())
        bone.parent = it->second;
      bone.center = dson::vec3Channel(get(&n, "center_point"));
      byId[dson::getString(&n, "id")] = int(cage->skeleton.size());
      cage->skeleton.push_back(std::move(bone));
    }
  }
  // skin influences per vertex
  const size_t                                count = cage->positions.size() / 3;
  std::vector<std::vector<std::pair<int, float>>> lists(count);
  if(const dson::Json* mods = get(&doc.json, "modifier_library"); mods && mods->is_array())
    for(const dson::Json& m : *mods)
    {
      const dson::Json* skin = get(&m, "skin");
      if(!skin || skin->is_null())
        continue;
      if(const dson::Json* joints = get(skin, "joints"); joints && joints->is_array())
        for(const dson::Json& j : *joints)
        {
          std::string ref = dson::getString(&j, "node");
          if(ref.empty())
            ref = dson::getString(&j, "id");
          const auto id   = dson::decodeUri(ref.substr(ref.find_last_of('#') == std::string::npos ? 0 : ref.find_last_of('#') + 1));
          const auto bone = byId.find(id);
          const dson::Json* weights = get(get(&j, "node_weights"), "values");
          if(bone == byId.end() || !weights)
            continue;
          const auto pairs = doc.rows(weights);
          for(int i = 0; i < pairs->count; i++)
          {
            const double v = pairs->cell(i, 0), w = pairs->cell(i, 1);
            if(v >= 0 && size_t(v) < count && w > 0)
              lists[size_t(v)].emplace_back(bone->second, float(w));
          }
        }
      break;  // the first skin modifier
    }
  cage->skin.start.assign(count + 1, 0);
  for(size_t v = 0; v < count; v++)
    cage->skin.start[v + 1] = cage->skin.start[v] + int32_t(lists[v].size());
  for(size_t v = 0; v < count; v++)
    for(const auto& [b, w] : lists[v])
    {
      cage->skin.bone.push_back(b);
      cage->skin.weight.push_back(w);
    }
  return cage;
}

}  // namespace

std::shared_ptr<const BaseCage> loadBaseCage(dson::Store& store, Generation generation, const std::string& gender)
{
  static std::mutex                                     mutex;
  static std::map<std::string, std::shared_ptr<const BaseCage>> cache;
  for(const std::string& file : order(generation, gender))
  {
    {
      std::lock_guard lock(mutex);
      if(auto it = cache.find(file); it != cache.end())
      {
        if(it->second)
          return it->second;
        continue;
      }
    }
    std::shared_ptr<const BaseCage> cage;
    try
    {
      if(auto doc = store.load(file))
        cage = build(*doc);
    }
    catch(const std::exception&)
    {
    }
    {
      std::lock_guard lock(mutex);
      cache[file] = cage;
    }
    if(cage)
      return cage;
  }
  return nullptr;
}

}  // namespace fv::wardrobe
