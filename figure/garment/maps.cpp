#include <figure/garment/maps.hpp>

#include <map>
#include <mutex>
#include <utility>

#include <figure/bodymap/cache.hpp>
#include <figure/wardrobe/fit/bases.hpp>

namespace fv::garment {

namespace {

struct Held
{
  // the bases the map was made of: read again, they are other objects, and the map is made again
  std::shared_ptr<const wardrobe::BaseCage> base, g9;
  std::shared_ptr<const bodymap::TypeMap>   map;
};

std::mutex                                                     mutex;
std::map<std::pair<wardrobe::Generation, std::string>, Held> held;

}  // namespace

std::shared_ptr<const bodymap::TypeMap> typeMap(dson::Store& store, wardrobe::Generation generation, const std::string& gender)
{
  if(bodymap::cloneFile(generation, gender).empty())
    return nullptr;
  const auto base = wardrobe::loadBaseCage(store, generation, gender);
  const auto g9   = wardrobe::loadBaseCage(store, wardrobe::Generation::eG9, "");
  if(!base || !g9)
    return nullptr;
  const auto key = std::make_pair(generation, gender);
  {
    std::lock_guard lock(mutex);
    const auto      it = held.find(key);
    if(it != held.end() && it->second.base == base && it->second.g9 == g9)
      return it->second.map;
  }
  // made outside the lock: two wears asking at once make it twice, and the second answer is kept
  const auto inputs = bodymap::cloneInputs(store, generation, gender);
  Held       made{base, g9, nullptr};
  if(inputs)
    made.map = std::make_shared<const bodymap::TypeMap>(bodymap::cachedCloneMap(*inputs));
  std::lock_guard lock(mutex);
  held[key] = made;
  return made.map;
}

}  // namespace fv::garment
