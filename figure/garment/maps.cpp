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

bodymap::TypeMap referenceMap(const bodymap::Body& reference)
{
  bodymap::TypeMap out{reference, reference, {}, {}};
  const size_t     n = reference.positions.size() / 3;
  bodymap::Projection same;
  same.at.verts.resize(n * 3);
  same.at.weights.assign(n * 3, 0.f);
  same.at.used.assign(n, 1);
  same.mapped.assign(n, 1);
  for(size_t v = 0; v < n; v++)
  {
    same.at.verts[v * 3] = same.at.verts[v * 3 + 1] = same.at.verts[v * 3 + 2] = uint32_t(v);
    same.at.weights[v * 3]                                                     = 1;
  }
  out.map.toReference   = same;
  out.map.fromReference = std::move(same);
  return out;
}

std::shared_ptr<const bodymap::TypeMap> typeMap(dson::Store& store, wardrobe::Generation generation, const std::string& gender)
{
  const bool reference = generation == wardrobe::Generation::eG9;
  if(!reference && bodymap::cloneFile(generation, gender).empty())
    return nullptr;
  const auto g9   = wardrobe::loadBaseCage(store, wardrobe::Generation::eG9, "");
  const auto base = reference ? g9 : wardrobe::loadBaseCage(store, generation, gender);
  if(!base || !g9)
    return nullptr;
  const auto key = std::make_pair(generation, reference ? std::string() : gender);
  {
    std::lock_guard lock(mutex);
    const auto      it = held.find(key);
    if(it != held.end() && it->second.base == base && it->second.g9 == g9)
      return it->second.map;
  }
  // made outside the lock: two wears asking at once make it twice, and the second answer is kept
  Held made{base, g9, nullptr};
  if(reference)
    made.map = std::make_shared<const bodymap::TypeMap>(referenceMap(bodymap::bodyOf(*g9)));
  else if(const auto inputs = bodymap::cloneInputs(store, generation, gender))
    made.map = std::make_shared<const bodymap::TypeMap>(bodymap::cachedCloneMap(*inputs));
  std::lock_guard lock(mutex);
  held[key] = made;
  return made.map;
}

std::shared_ptr<const bodymap::TypeMap> bodyMap(dson::Store& store, const std::shared_ptr<const wardrobe::BaseCage>& body, const std::string& gender)
{
  // the bodies' maps, the last few asked for: a library has a body or two (the game's, a body mod's)
  static std::vector<Held> bodies;
  constexpr size_t         kKept = 4;
  const auto               g9    = wardrobe::loadBaseCage(store, wardrobe::Generation::eG9, "");
  if(!body || !g9)
    return nullptr;
  {
    std::lock_guard lock(mutex);
    for(const Held& h : bodies)
      if(h.base == body && h.g9 == g9)
        return h.map;
  }
  bodymap::FitInputs in{body, g9, {}};
  const bool         male   = gender == "male";
  auto               shaped = bodymap::cloneReference(store, wardrobe::Generation::eG8, male ? "male" : "female");
  if(!shaped)
    shaped = bodymap::cloneReference(store, wardrobe::Generation::eG8, male ? "female" : "male");
  in.reference = shaped ? *shaped : bodymap::bodyOf(*g9);
  Held made{body, g9, std::make_shared<const bodymap::TypeMap>(bodymap::cachedFittedMap(in))};
  std::lock_guard lock(mutex);
  bodies.push_back(made);
  if(bodies.size() > kKept)
    bodies.erase(bodies.begin());
  return made.map;
}

}  // namespace fv::garment
