#include "shell.hpp"

#include <cmath>
#include <map>

namespace fv::dson {

namespace {

bool hasExtra(const Json* n, std::string_view type)
{
  const Json* extra = get(n, "extra");
  if(!extra || !extra->is_array())
    return false;
  for(const Json& e : *extra)
    if(getString(&e, "type") == type)
      return true;
  return false;
}

// Each channel in a node's studio_node_channels (or a modifier's studio_modifier_channels)
template <typename F>
void eachChannel(const Json* n, std::string_view type, F&& f)
{
  const Json* extra = get(n, "extra");
  if(!extra || !extra->is_array())
    return;
  for(const Json& e : *extra)
    if(getString(&e, "type") == type)
      if(const Json* chans = get(&e, "channels"); chans && chans->is_array())
        for(const Json& c : *chans)
          if(const Json* ch = get(&c, "channel"))
            f(*ch);
}

std::optional<bool> boolValue(const Json& ch)
{
  for(const char* key : {"current_value", "value"})
    if(const Json* v = get(&ch, key); v && v->is_boolean())
      return v->get<bool>();
  return std::nullopt;
}

bool endsWith(const std::string& s, std::string_view suffix)
{
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

std::optional<Shell> parseShell(const Json& node, const Json* def, const Json* sceneModifiers, Store& store, const std::string& file)
{
  if(!hasExtra(&node, "studio/node/shell") && !hasExtra(def, "studio/node/shell"))
    return std::nullopt;
  Shell shell;
  // the definition's channels, then the scene's over them
  std::map<std::string, bool> visible;
  for(const Json* n : {def, &node})
    eachChannel(n, "studio_node_channels", [&](const Json& ch) {
      const std::string id = getString(&ch, "id");
      if(id == "Shell Node")
      {
        const std::string ref = getString(&ch, "node");
        if(const size_t hash = ref.rfind('#'); hash != std::string::npos)
          shell.source = decodeUri(ref.substr(hash + 1));
      }
      else if(endsWith(id, "_vis"))
        if(const auto v = boolValue(ch))
          visible[id] = *v;
    });
  if(shell.source.empty())
    return std::nullopt;
  for(const auto& [id, on] : visible)
  {
    if(on)
      continue;
    const std::string name = id.substr(0, id.size() - 4);
    if(name.rfind("material_group_", 0) == 0)
      shell.hiddenMaterials.insert(name.substr(15));
    else if(name.rfind("facet_group_", 0) == 0)
      shell.hiddenFacets.insert(name.substr(12));
  }

  // its push modifier: the scene's instance, over the library entry it points at
  const std::string id = getString(&node, "id");
  if(sceneModifiers && sceneModifiers->is_array())
    for(const Json& m : *sceneModifiers)
    {
      const std::string parent = getString(&m, "parent");
      const size_t      hash   = parent.rfind('#');
      if(hash == std::string::npos || decodeUri(parent.substr(hash + 1)) != id || !hasExtra(&m, "studio/modifier/push"))
        continue;
      std::optional<double> value;
      const auto            read = [&](const Json* n) {
        eachChannel(n, "studio_modifier_channels", [&](const Json& ch) {
          if(getString(&ch, "id") == "Value")
            value = channelValue(&ch);
        });
      };
      try
      {
        if(auto ref = parseRef(getString(&m, "url"), file); ref && !ref->id.empty())
          if(auto lib = store.resolve(*ref, "modifier_library"))
            read(lib->entry);
      }
      catch(const std::exception&)
      {
      }
      read(&m);
      if(value)
        shell.offset = *value;
    }
  return shell;
}

void shellInput(mesh::MeshInput& m, const Shell& shell, const std::vector<std::string>& materialGroups,
                const std::vector<std::string>& facetGroups, const std::vector<int32_t>& polyGroup)
{
  const size_t np = m.polyMaterial.size();
  if(shell.offset != 0 && !m.positions.empty())
  {
    // area-weighted normals: each polygon's corners' cross products about its first corner
    std::vector<double> normal(m.positions.size(), 0.0);
    const auto          p = [&](int32_t v, int c) { return double(m.positions[size_t(v) * 3 + c]); };
    for(size_t pi = 0; pi < np; pi++)
    {
      const int32_t s = m.polyStart[pi], e = m.polyStart[pi + 1];
      double        n[3] = {0, 0, 0};
      for(int32_t k = s + 1; k + 1 < e; k++)
      {
        const int32_t a = m.polyVerts[s], b = m.polyVerts[k], c = m.polyVerts[k + 1];
        const double  u[3] = {p(b, 0) - p(a, 0), p(b, 1) - p(a, 1), p(b, 2) - p(a, 2)};
        const double  w[3] = {p(c, 0) - p(a, 0), p(c, 1) - p(a, 1), p(c, 2) - p(a, 2)};
        n[0] += u[1] * w[2] - u[2] * w[1];
        n[1] += u[2] * w[0] - u[0] * w[2];
        n[2] += u[0] * w[1] - u[1] * w[0];
      }
      for(int32_t k = s; k < e; k++)
        for(int c = 0; c < 3; c++)
          normal[size_t(m.polyVerts[k]) * 3 + c] += n[c];
    }
    for(size_t v = 0; v * 3 < m.positions.size(); v++)
    {
      const double* n   = &normal[v * 3];
      const double  len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if(len > 0)
        for(int c = 0; c < 3; c++)
          m.positions[v * 3 + c] += float(shell.offset * n[c] / len);
    }
  }

  if(shell.hiddenMaterials.empty() && shell.hiddenFacets.empty())
    return;
  const auto hidden = [&](size_t pi) {
    const int32_t mat = m.polyMaterial[pi];
    if(mat >= 0 && size_t(mat) < materialGroups.size() && shell.hiddenMaterials.count(materialGroups[size_t(mat)]))
      return true;
    const int32_t g = pi < polyGroup.size() ? polyGroup[pi] : -1;
    return g >= 0 && size_t(g) < facetGroups.size() && shell.hiddenFacets.count(facetGroups[size_t(g)]) > 0;
  };
  std::vector<int32_t> newIndex(np, -1), start{0}, verts, material;
  for(size_t pi = 0; pi < np; pi++)
  {
    if(hidden(pi))
      continue;
    newIndex[pi] = int32_t(material.size());
    verts.insert(verts.end(), m.polyVerts.begin() + m.polyStart[pi], m.polyVerts.begin() + m.polyStart[pi + 1]);
    start.push_back(int32_t(verts.size()));
    material.push_back(m.polyMaterial[pi]);
  }
  std::vector<int32_t> overrides;
  for(size_t i = 0; i + 2 < m.overrides.size(); i += 3)
  {
    const int32_t pi = m.overrides[i];
    if(pi < 0 || size_t(pi) >= np || newIndex[size_t(pi)] < 0)
      continue;
    overrides.insert(overrides.end(), {newIndex[size_t(pi)], m.overrides[i + 1], m.overrides[i + 2]});
  }
  m.polyStart    = std::move(start);
  m.polyVerts    = std::move(verts);
  m.polyMaterial = std::move(material);
  m.overrides    = std::move(overrides);
}

}  // namespace fv::dson
