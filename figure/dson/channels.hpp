#pragma once

// A material's DSON channels: values and images by normalised name, as collectChannels
// (materials.hpp) merges them across the files that define and override a surface.

#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/materials/surface.hpp>

namespace fv::dson {

struct Channel
{
  Json                       value;  // null when unset
  std::optional<std::string> image;  // library path
  // every name it was stored under, so an override finds them
  std::vector<std::string> keys;
  // its normalised id, empty without one: another channel's label or name never takes it over
  std::string id;
};

// Channels by normalised name, in the order first stored (as a Map iterates).
class Channels
{
public:
  const Channel* get(const std::string& normName) const;
  // the first of `names` (not normalised) present
  const Channel* find(std::initializer_list<const char*> names) const;
  void           set(const std::string& key, std::shared_ptr<Channel> c);
  void           erase(const std::string& key);
  std::shared_ptr<Channel> shared(const std::string& key) const;
  const std::vector<std::string>& order() const { return m_order; }
  // the shader the layers name, the last one that names any
  const mat::ShaderRef& shader() const { return m_shader; }
  void                  setShader(mat::ShaderRef s) { m_shader = std::move(s); }

private:
  mat::ShaderRef                                            m_shader;
  std::vector<std::string>                                  m_order;
  std::unordered_map<std::string, std::shared_ptr<Channel>> m_map;
};

// Lower case, letters and digits only.
std::string norm(std::string_view s);

}  // namespace fv::dson
