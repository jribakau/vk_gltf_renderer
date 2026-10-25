#include <figure/dson/figure_loader.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <figure/dson/materials.hpp>
#include <figure/io/timing.hpp>
#include <figure/mesh/normals.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/generation.hpp>

namespace fv::dson {

namespace {

// node[key] unless null or absent, else def[key] (JavaScript's `node.key ?? def.key`).
const Json* pick(const Json* node, const Json* def, std::string_view key)
{
  const Json* v = get(node, key);
  if(v && !v->is_null())
    return v;
  v = get(def, key);
  return v && !v->is_null() ? v : nullptr;
}

std::string str(const Json* v, std::string fallback = {})
{
  return v && v->is_string() ? v->get<std::string>() : fallback;
}

// The first of the members that is set (`a.x ?? a.y ?? ...`), as a string.
std::string firstString(std::initializer_list<std::pair<const Json*, const char*>> options, std::string fallback = {})
{
  for(const auto& [obj, key] : options)
  {
    const Json* v = get(obj, key);
    if(v && !v->is_null())
      return v->is_string() ? v->get<std::string>() : v->dump();
  }
  return fallback;
}

std::string lastSegment(const std::string& s, char sep)
{
  const size_t at = s.rfind(sep);
  return at == std::string::npos ? s : s.substr(at + 1);
}

std::string lowerCase(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

std::string replaceAll(std::string s, const std::string& from, const std::string& to)
{
  for(size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size()))
    s.replace(at, from.size(), to);
  return s;
}

// A DSON node's transform: rotation and scale about its centre point, in its orientation frame, then
// translation. Geometry is stored in rest-pose world space, so this is relative to the parent.
void nodeTransform(Node& group, const Json* node, const Json* def)
{
  const auto        t     = vec3Channel(pick(node, def, "translation"));
  const auto        r     = vec3Channel(pick(node, def, "rotation"));
  const auto        s     = vec3Channel(pick(node, def, "scale"), {1, 1, 1});
  const double      gs    = channelValue(pick(node, def, "general_scale"), 1);
  const auto        c     = vec3Channel(pick(node, def, "center_point"));
  const auto        o     = vec3Channel(pick(node, def, "orientation"));
  const std::string order = firstString({{def, "rotation_order"}, {node, "rotation_order"}}, "XYZ");
  const bool        xyz3  = order.size() == 3 && std::all_of(order.begin(), order.end(), [](char ch) { return ch == 'X' || ch == 'Y' || ch == 'Z'; });
  const Quat        orient = quatFromEuler(o[0] * kDeg, o[1] * kDeg, o[2] * kDeg, xyz3 ? order : "XYZ");
  const Quat        rot    = quatFromEuler(r[0] * kDeg, r[1] * kDeg, r[2] * kDeg, threeOrder(order).data());
  const Quat        q      = orient * rot * orient.inverse();
  group.quaternion         = q;
  group.scale              = {s[0] * gs, s[1] * gs, s[2] * gs};
  const Vec3 center{c[0], c[1], c[2]};
  group.position = Vec3{t[0], t[1], t[2]} + center - rotate(center * group.scale, q);
}

// DAZ's node Visible switch, off in the scene file: a hidden helper such as a strand hair's growth cap.
bool hiddenNode(const Json* node)
{
  const Json* extra = get(node, "extra");
  if(!extra || !extra->is_array())
    return false;
  for(const Json& e : *extra)
    if(const Json* chans = get(&e, "channels"); chans && chans->is_array())
      for(const Json& c : *chans)
      {
        const Json* inner = get(&c, "channel");
        const Json& ch    = inner && !inner->is_null() ? *inner : c;
        if(getString(&ch, "id") != "Visible")
          continue;
        const Json* v = get(&ch, "current_value");
        if(!v || v->is_null())
          v = get(&ch, "value");
        return v && v->is_boolean() && !v->get<bool>();
      }
  return false;
}

// Whether a node is a DAZ Studio instance (a drawn copy of another node), which isn't loaded.
bool isDazInstance(const Json* node)
{
  const Json* extra = get(node, "extra");
  if(!extra || !extra->is_array())
    return false;
  static const std::regex type(R"((^|/)instance(_group)?$)");
  for(const Json& e : *extra)
  {
    if(std::regex_search(getString(&e, "type"), type) || (e.is_object() && e.contains("instance_items")))
      return true;
    if(const Json* chans = get(&e, "channels"); chans && chans->is_array())
      for(const Json& c : *chans)
      {
        const Json* inner = get(&c, "channel");
        const Json& ch    = inner && !inner->is_null() ? *inner : c;
        if(getString(&ch, "id") == "Instance Target")
          return true;
      }
  }
  return false;
}

// A channel in a node's extra channel lists (DAZ Studio's own: Visible, Instance Target, ...).
const Json* extraChannel(const Json* node, std::string_view id)
{
  const Json* extra = get(node, "extra");
  if(!extra || !extra->is_array())
    return nullptr;
  for(const Json& e : *extra)
    if(const Json* chans = get(&e, "channels"); chans && chans->is_array())
      for(const Json& c : *chans)
      {
        const Json* inner = get(&c, "channel");
        const Json& ch    = inner && !inner->is_null() ? *inner : c;
        if(getString(&ch, "id") == id)
          return &ch;
      }
  return nullptr;
}

// A DAZ Studio node instance: another scene node drawn again where the instance stands (its
// transform takes the target's place), with that node's children when its Instance Mode is "Node
// and Children" (0) rather than "Single Node" (1).
struct NodeInstance
{
  std::string id;  // the instance's scene node
  int         group = -1;
  std::string target;  // the scene node it draws; empty when it names none
  bool        children = false;
  bool        visible  = true;
};

std::optional<NodeInstance> nodeInstance(const Json* node, const Json* def)
{
  const Json* target = extraChannel(node, "Instance Target");
  if(!target)
    target = extraChannel(def, "Instance Target");
  if(!target)
    return std::nullopt;
  const Json* mode = extraChannel(node, "Instance Mode");
  if(!mode)
    mode = extraChannel(def, "Instance Mode");
  const Json* value = pick(mode, mode, "current_value");
  if(!value)
    value = get(mode, "value");
  NodeInstance out;
  out.id       = getString(node, "id");
  out.target   = decodeUri(lastSegment(firstString({{target, "node"}, {target, "current_value"}}), '#'));
  out.children = value && value->is_number() && value->get<double>() == 0;
  out.visible  = !hiddenNode(node);
  return out;
}

struct UvSet
{
  std::vector<float>   uvs;        // two floats per UV index
  std::vector<int32_t> overrides;  // (polygon, vertex, uv index) triplets
};

UvSet parseUvSet(const Doc& doc, const Json& entry)
{
  UvSet       out;
  const Json* uvs    = get(&entry, "uvs");
  const auto  values = doc.rows(uvs ? get(uvs, "values") : nullptr);
  out.uvs.resize(size_t(values->count) * 2);
  for(int i = 0; i < values->count; i++)
  {
    out.uvs[i * 2]     = float(values->cell(i, 0));
    out.uvs[i * 2 + 1] = float(values->cell(i, 1));
  }
  const auto triplets = doc.rows(get(&entry, "polygon_vertex_indices"));
  out.overrides.resize(size_t(triplets->count) * 3);
  for(int i = 0; i < triplets->count; i++)
    for(int k = 0; k < 3; k++)
      out.overrides[i * 3 + k] = toInt32(triplets->cell(i, k));
  return out;
}

// The build input for a DSON geometry (src/lib/loaders/dson/geometry.ts buildGeometry).
mesh::MeshInput meshInput(const Doc& doc, const Json& geo, const UvSet* uvSet, int level, std::vector<std::string>& materialGroups)
{
  const Json*     vertices = get(&geo, "vertices");
  const Json*     polylist = get(&geo, "polylist");
  const auto      verts    = doc.rows(vertices ? get(vertices, "values") : nullptr);
  // polygon rows: [group, material, v0, v1, ...]
  const auto      polys = doc.rows(polylist ? get(polylist, "values") : nullptr);
  mesh::MeshInput m;
  materialGroups.clear();
  if(const Json* groups = get(&geo, "polygon_material_groups"); groups)
    if(const Json* values = get(groups, "values"); values && values->is_array())
      for(const Json& v : *values)
        materialGroups.push_back(v.is_string() ? v.get<std::string>() : v.dump());
  if(materialGroups.empty() && !(get(&geo, "polygon_material_groups") && get(get(&geo, "polygon_material_groups"), "values")))
    materialGroups.push_back("Default");
  const int nv = verts->count;
  m.positions.resize(size_t(nv) * 3);
  for(int v = 0; v < nv; v++)
    for(int c = 0; c < 3; c++)
      m.positions[v * 3 + c] = float(verts->cell(v, c));
  const int np = polys->count;
  m.polyStart.assign(size_t(np) + 1, 0);
  for(int pi = 0; pi < np; pi++)
    m.polyStart[pi + 1] = m.polyStart[pi] + polys->start[pi + 1] - polys->start[pi] - 2;
  m.polyVerts.resize(m.polyStart[np]);
  m.polyMaterial.resize(np);
  for(int pi = 0; pi < np; pi++)
  {
    m.polyMaterial[pi] = toInt32(polys->cell(pi, 1));
    for(int k = polys->start[pi] + 2, at = m.polyStart[pi]; k < polys->start[pi + 1]; k++, at++)
      m.polyVerts[at] = toInt32(polys->data[k]);
  }
  m.materialCount = int(materialGroups.size());
  if(uvSet)
  {
    m.uvs       = uvSet->uvs;
    m.overrides = uvSet->overrides;
  }
  m.level    = level;
  m.boundary = getString(&geo, "edge_interpolation_mode") == "edges_and_corners" ? subdiv::BoundaryInterpolation::eEdgesAndCorners :
                                                                                  subdiv::BoundaryInterpolation::eEdges;
  // strand hair: polylines in the same row layout, drawn as ribbons when there are no polygons
  if(!np)
  {
    const Json* polyline = get(&geo, "polyline_list");
    const auto  t        = doc.rows(polyline ? get(polyline, "values") : nullptr);
    if(t->count)
    {
      mesh::Lines lines;
      lines.start.assign(size_t(t->count) + 1, 0);
      for(int l = 0; l < t->count; l++)
        lines.start[l + 1] = lines.start[l] + std::max(0, t->start[l + 1] - t->start[l] - 2);
      lines.verts.resize(lines.start[t->count]);
      lines.material.resize(t->count);
      for(int l = 0; l < t->count; l++)
      {
        lines.material[l] = toInt32(t->cell(l, 1));
        for(int k = t->start[l] + 2, at = lines.start[l]; k < t->start[l + 1]; k++, at++)
          lines.verts[at] = toInt32(t->data[k]);
      }
      m.lines = std::move(lines);
    }
  }
  return m;
}

// A strand ribbon's root and tip width, cm (src/lib/loaders/dson/geometry.ts ribbonWidths).
std::array<double, 2> ribbonWidths(std::optional<double> start, std::optional<double> end, double spacing)
{
  const double pi      = 3.14159265358979323846;
  const double authored = start && *start > 0 ? (*start / 10) * (pi / 2) : 0;
  const double taper    = start && *start > 0 && end && *end >= 0 ? std::min(1.0, *end / *start) : 0.5;
  const double root     = authored ? authored : spacing;
  return {root, root * taper};
}

// Opens the zero-width ribbons to `widths[material]` (root, tip in cm), then welds normals across them.
void widenRibbons(mesh::BuiltMesh& m, const std::vector<std::array<double, 2>>& widths)
{
  if(!m.ribbon)
    return;
  const auto& r = *m.ribbon;
  for(size_t i = 0; i < r.along.size(); i++)
  {
    const int  mat = r.material[i];
    const auto w   = mat >= 0 && mat < int(widths.size()) ? widths[mat] : (widths.empty() ? std::array<double, 2>{0.1, 0.05} : widths[0]);
    // The first 6% of the strand grows from nothing, so the root is not a full-width sheet and the
    // body can stay opaque (no any-hit on every card). along is 0 at the root.
    constexpr double kRoot = 0.06;
    double           gate  = 1;
    if(r.along[i] < kRoot)
    {
      const double t = r.along[i] / kRoot;
      gate           = t * t * (3.0 - 2.0 * t);
    }
    const double half = (w[0] + (w[1] - w[0]) * r.along[i]) / 2 * gate;
    for(int k = 0; k < 3; k++)
      m.position[i * 3 + k] = float(m.position[i * 3 + k] + r.side[i * 3 + k] * half);
  }
  mesh::weldNormals(m.position, m.renderIndex, m.firstSplit, m.renderTris, m.renderVertexCount, m.normal);
}

using Centers = std::map<std::string, std::array<double, 3>>;

// Joint centres for an item's skeleton: in a scene file the centres it recorded win, and the
// figure's fill in the rest (src/lib/loaders/dson/figure.ts fitCenters).
Centers fitCenters(const SceneGraph& scene, const Json& nodeLibrary, const Centers* own, const Rig& figure, bool preferFigure)
{
  Centers out = own ? *own : Centers{};
  if(!nodeLibrary.is_array())
    return out;
  std::vector<std::string> figureNames;
  for(const auto& [id, node] : figure.bones)
    figureNames.push_back(scene.nodes[size_t(node)].name);
  for(const Json& n : nodeLibrary)
  {
    const std::string id = getString(&n, "id");
    if(getString(&n, "type") != "bone" || (!preferFigure && out.count(id)))
      continue;
    int b = figure.find(scene, id);
    if(b < 0)
      if(const std::string name = getString(&n, "name"); !name.empty())
        b = figure.find(scene, name);
    if(b < 0)
      if(const std::string name = getString(&n, "name"); !name.empty())
        if(const auto alias = wardrobe::matchBone(name, figureNames))
          b = figure.find(scene, *alias);
    if(b >= 0 && scene.nodes[b].joint)
      out[id] = scene.nodes[b].joint->center;
  }
  return out;
}

std::string missingMessage(const std::string& fileName, const std::vector<std::string>& missing)
{
  std::string name = fileName;
  for(const char* ext : {".duf", ".dsf", ".DUF", ".DSF"})
    if(name.size() > 4 && name.compare(name.size() - 4, 4, ext) == 0)
      name.resize(name.size() - 4);
  std::string sample = missing[0] + (missing.size() > 1 ? ", " + missing[1] : "");
  if(missing.size() > 2)
    sample += " and " + std::to_string(missing.size() - 2) + " more";
  return name + " needs " + std::to_string(missing.size()) + " library file" + (missing.size() == 1 ? "" : "s")
         + " that aren't installed in the assets folder or the DAZ library: " + sample + ".";
}

}  // namespace

LoadResult loadDson(Store& store, const std::string& rel, const LoadOptions& options)
{
  const std::string file = rel.substr(std::min(rel.find_first_not_of('/'), rel.size()));
  auto              docPtr = store.load(file);
  const Json&       doc    = docPtr->json;
  LoadResult        out;
  auto              warn = [&](const std::string& msg) { out.warnings.push_back(msg); };

  std::unordered_map<std::string, std::string> images;
  collectImages(doc, images);

  SceneGraph& scene = out.scene;
  {
    Node        root;
    const Json* info = get(&doc, "asset_info");
    std::string id   = getString(info, "id");
    root.name        = !id.empty() ? replaceAll(lastSegment(id, '/'), "%20", " ") : lastSegment(file, '/');
    if(root.name.empty())
      root.name = "DAZ";
    out.root = scene.add(std::move(root));
    out.name = scene.nodes[out.root].name;
  }

  // ---- scene nodes (or a synthetic one for a bare .dsf)
  Json        synthetic = Json::array();
  const Json* sceneJson = get(&doc, "scene");
  const Json* nodesJson = get(sceneJson, "nodes");
  if(!nodesJson || !nodesJson->is_array() || nodesJson->empty())
  {
    const Json* lib    = get(&doc, "node_library");
    const Json* figure = nullptr;
    if(lib && lib->is_array())
    {
      for(const Json& n : *lib)
        if(getString(&n, "type") == "figure")
        {
          figure = &n;
          break;
        }
      if(!figure && !lib->empty())
        figure = &(*lib)[0];
    }
    const Json* geos = get(&doc, "geometry_library");
    if(!geos || !geos->is_array() || geos->empty())
      throw std::runtime_error("This file has no scene and no geometry. Open a character or figure .duf instead.");
    Json node            = Json::object();
    node["id"]           = figure ? getString(figure, "id", "node") : "node";
    node["url"]          = figure ? "#" + getString(figure, "id") : "";
    if(figure && get(figure, "label"))
      node["label"] = *get(figure, "label");
    node["geometries"] = Json::array();
    for(const Json& g : *geos)
      node["geometries"].push_back({{"id", getString(&g, "id")}, {"url", "#" + getString(&g, "id")}});
    synthetic.push_back(std::move(node));
    nodesJson = &synthetic;
  }
  const Json& nodes = *nodesJson;

  // scene materials by the geometry instance (or library geometry) they apply to, in file order
  std::vector<const Json*>                                   sceneMaterials;
  std::unordered_map<std::string, std::vector<int>>          sceneMaterialsById;
  if(const Json* mats = get(sceneJson, "materials"); mats && mats->is_array())
    for(const Json& m : *mats)
    {
      auto ref = parseRef(getString(&m, "geometry"), file);
      sceneMaterialsById[ref ? ref->id : std::string("\x01")].push_back(int(sceneMaterials.size()));
      sceneMaterials.push_back(&m);
    }
  // a library geometry id that is also another instance's id names that instance, not this one
  // (props whose geometries are all called "geometry", where the first copy keeps the bare id)
  std::unordered_set<std::string> instanceIds;
  for(const Json& n : nodes)
    if(const Json* geos = get(&n, "geometries"); geos && geos->is_array())
      for(const Json& g : *geos)
        instanceIds.insert(getString(&g, "id"));
  auto sceneMaterialsFor = [&](const std::string& instanceId, const std::string& geometryId) {
    std::vector<int> list;
    if(auto it = sceneMaterialsById.find(instanceId); it != sceneMaterialsById.end())
      list = it->second;
    if(instanceId != geometryId && !instanceIds.count(geometryId))
      if(auto it = sceneMaterialsById.find(geometryId); it != sceneMaterialsById.end())
        list.insert(list.end(), it->second.begin(), it->second.end());
    std::sort(list.begin(), list.end());
    std::vector<const Json*> res;
    for(int i : list)
      res.push_back(sceneMaterials[i]);
    return res;
  };

  // A character's scene file records each bone's final joint centre (after its shape morphs moved
  // it) in `preview`. Collected per figure scene node, so posing pivots where the body is.
  std::unordered_map<std::string, Centers> previewCenters;
  {
    std::unordered_map<std::string, const Json*> byId;
    for(const Json& sn : nodes)
      byId[getString(&sn, "id")] = &sn;
    auto parentOf = [&](const Json* sn) -> const Json* {
      const Json* p = get(sn, "parent");
      if(!p || !p->is_string())
        return nullptr;
      auto it = byId.find(decodeUri(lastSegment(p->get<std::string>(), '#')));
      return it == byId.end() ? nullptr : it->second;
    };
    auto isBonePreview = [](const Json* sn) { return getString(get(sn, "preview"), "type") == "bone"; };
    for(const Json& sn : nodes)
    {
      const Json* c = get(get(&sn, "preview"), "center_point");
      if(!isBonePreview(&sn) || !c || !c->is_array() || c->size() != 3)
        continue;
      const Json* fig = parentOf(&sn);
      for(int i = 0; i < 64 && fig && isBonePreview(fig); i++)
        fig = parentOf(fig);
      auto boneRef = parseRef(getString(&sn, "url"), file);
      if(!fig || !boneRef || boneRef->id.empty())
        continue;
      previewCenters[getString(fig, "id")][boneRef->id] = {(*c)[0].get<double>(), (*c)[1].get<double>(), (*c)[2].get<double>()};
    }
  }

  // Morph dials the scene sets, and the node or geometry instance each one sits on
  if(const Json* mods = get(sceneJson, "modifiers"); mods && mods->is_array())
    for(const Json& m : *mods)
    {
      const Json* url = get(&m, "url");
      if(!url || !url->is_string() || channelValue(get(&m, "channel")) == 0)
        continue;
      auto ref = parseRef(url->get<std::string>(), file);
      if(!ref || ref->id.empty())
        continue;
      auto owner = parseRef(getString(&m, "parent"), file);
      out.morphs.dials.push_back({ref->file, ref->id, owner ? owner->id : std::string(), channelValue(get(&m, "channel"))});
    }

  // ---- node definitions, then groups and rigs in file order (parents before children)
  std::vector<std::string> skipped;
  auto                     skip = [&](const Json& node, const std::string& why, bool missingFile) {
    // missing files are reported once, together, after every node has been tried
    if(missingFile)
      return;
    const std::string reason = firstString({{&node, "label"}, {&node, "id"}}) + ": " + why;
    skipped.push_back(reason);
    warn("Skipped " + reason);
  };

  struct MeshJob
  {
    const Json* node;
    int         group;
    std::string type;
    int         rig;   // index into out.rigs, -1 without
    int         at;    // where the meshes go among the group's children: after its rig, before child nodes
    int         item;  // index into out.items, -1 without
    std::string defFile;  // the library file of its node definition
  };
  std::vector<MeshJob>                       jobs;
  std::unordered_map<std::string, int>       itemOf;  // scene node or geometry instance id -> item
  std::unordered_map<std::string, std::pair<std::string, std::string>> boneNodes;  // scene bone node -> (figure node, bone id)
  std::unordered_map<std::string, int>       rigOf;  // figure scene node -> rig
  std::unordered_map<std::string, int>       instanceGroups;
  std::string                                mainNode;
  std::vector<std::string>                   dazInstances;  // instance groups, which aren't drawn
  std::vector<NodeInstance>                  nodeInstances;
  std::unordered_map<std::string, std::vector<std::string>> childrenOf;  // scene node -> its child nodes with groups
  std::unordered_map<std::string, std::vector<int>>         meshNodesOf;  // scene node -> its mesh nodes
  auto newItem = [&](const Json& node) {
    Item item;
    item.key   = options.attach ? options.attach->key + (out.items.empty() ? std::string() : "." + std::to_string(out.items.size()))
                                : getString(&node, "id");
    item.label = firstString({{&node, "label"}, {&node, "name"}, {&node, "id"}});
    out.items.push_back(std::move(item));
    return int(out.items.size()) - 1;
  };

  for(const Json& node : nodes)
  {
    std::optional<Store::Resolved> resolved;
    try
    {
      auto nodeRef = parseRef(getString(&node, "url"), file);
      if(nodeRef && !nodeRef->id.empty())
        resolved = store.resolve(*nodeRef, "node_library");
    }
    catch(const MissingFileError&)
    {
      skip(node, "", true);
      continue;
    }
    catch(const std::exception& e)
    {
      skip(node, e.what(), false);
      continue;
    }
    const Json*       def       = resolved ? resolved->entry : nullptr;
    const std::string type      = firstString({{def, "type"}, {&node, "type"}});
    const std::string id        = getString(&node, "id");
    const std::string parentRef = getString(&node, "parent");
    const std::string parentId  = parentRef.empty() ? "" : decodeUri(lastSegment(parentRef, '#'));
    if(type == "bone")
    {
      std::string figure;
      if(rigOf.count(parentId))
        figure = parentId;
      else if(auto it = boneNodes.find(parentId); it != boneNodes.end())
        figure = it->second.first;
      auto boneRef = parseRef(getString(&node, "url"), file);
      if(!figure.empty() && boneRef && !boneRef->id.empty())
        boneNodes[id] = {figure, boneRef->id};
      if(auto it = itemOf.find(parentId); it != itemOf.end())
        itemOf[id] = it->second;
    }
    if(type == "bone" || type == "camera" || type == "light")
      continue;

    Node groupNode;
    groupNode.name     = firstString({{&node, "label"}, {def, "label"}, {&node, "name"}, {&node, "id"}});
    groupNode.dsonNode = firstString({{&node, "name"}, {&node, "id"}});
    nodeTransform(groupNode, &node, def);

    // where it hangs, and the worn item it is part of
    int  item       = -1;
    if(auto it = itemOf.find(parentId); it != itemOf.end())
      item = it->second;
    int  parentBone = -1;
    auto onBone     = boneNodes.find(parentId);
    if(onBone != boneNodes.end())
      if(auto r = rigOf.find(onBone->second.first); r != rigOf.end())
        parentBone = out.rigs[r->second].byId(onBone->second.second);
    int group = -1;
    if(options.attach && item < 0)
    {
      // put on the figure later, by the wardrobe: under this scene's root until then
      static const std::regex selection(R"(^name://@selection(?:/([^:]*))?:?$)");
      std::smatch             sm;
      group = scene.add(std::move(groupNode), out.root);
      item  = newItem(node);
      out.items[item].objects.emplace_back(group, std::regex_match(parentRef, sm, selection) && sm[1].matched ? decodeUri(sm[1].str()) : std::string());
    }
    else if(parentBone >= 0)
    {
      // a group under the bone that cancels its rest offset, so nodes in figure space can hang from it
      Node mount;
      mount.name = scene.nodes[parentBone].name + " attachment";
      if(const auto& j = scene.nodes[parentBone].joint)
        mount.position = {-j->center[0], -j->center[1], -j->center[2]};
      const int mountId = scene.add(std::move(mount), parentBone);
      group             = scene.add(std::move(groupNode), mountId);
      if(item < 0 && onBone->second.first == mainNode)
      {
        item = newItem(node);
        out.items[item].objects.emplace_back(mountId, onBone->second.second);
      }
    }
    else
    {
      auto parentGroup = instanceGroups.find(parentId);
      group            = scene.add(std::move(groupNode), parentGroup != instanceGroups.end() ? parentGroup->second : out.root);
      if(item < 0 && !mainNode.empty() && parentId == mainNode)
      {
        item = newItem(node);
        out.items[item].objects.emplace_back(group, "");
      }
    }
    if(item >= 0)
    {
      Item& it = out.items[item];
      it.nodeNames.insert(firstString({{&node, "name"}, {&node, "id"}}));
      const Json* geos = get(&node, "geometries");
      if(resolved && geos && geos->is_array() && !geos->empty())
      {
        std::string src = resolved->file + "#" + getString(def, "id");
        std::transform(src.begin(), src.end(), src.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        it.sources.insert(src);
      }
      itemOf[id] = item;
      if(geos && geos->is_array())
        for(const Json& g : *geos)
          if(const std::string gid = getString(&g, "id"); !gid.empty())
            itemOf[gid] = item;
    }
    instanceGroups[id] = group;
    childrenOf[parentId].push_back(id);
    const Json* geos   = get(&node, "geometries");
    if(!geos || !geos->is_array() || geos->empty())
    {
      if(auto ni = nodeInstance(&node, def))
      {
        ni->group = group;
        nodeInstances.push_back(std::move(*ni));
      }
      else if(isDazInstance(&node) || isDazInstance(def))
        dazInstances.push_back(scene.nodes[group].name);
      continue;
    }

    int rig = -1;
    if(resolved && type == "figure")
    {
      // the first figure's folder holds the morphs of the figure that is posed
      if(out.morphs.morphDir.empty())
      {
        const size_t slash = resolved->file.rfind('/');
        out.morphs.morphDir = (slash == std::string::npos ? std::string() : resolved->file.substr(0, slash)) + "/Morphs";
      }
      const Json* lib     = get(&resolved->doc->json, "node_library");
      const Json  empty   = Json::array();
      const Json& library = lib ? *lib : empty;
      const Centers* own  = nullptr;
      if(auto it = previewCenters.find(id); it != previewCenters.end())
        own = &it->second;
      // an item binds at the figure's joint centres, which a character's shape has moved
      const int     mainRig = !mainNode.empty() && rigOf.count(mainNode) ? rigOf[mainNode] : -1;
      const Rig*    figureRig = options.attach ? options.attach->rig : (mainRig >= 0 ? &out.rigs[mainRig] : nullptr);
      const SceneGraph& figureScene = options.attach ? *options.attach->scene : scene;
      Centers       fitted;
      const Centers* centers = own;
      if(item >= 0 && figureRig)
      {
        // attaching, the figure's own centres win, so the item binds where a character's shape moved the joints
        fitted  = fitCenters(figureScene, library, own, *figureRig, options.attach.has_value());
        centers = &fitted;
      }
      if(auto r = buildRig(scene, library, centers))
      {
        scene.attach(r->root, group);
        out.rigs.push_back(std::move(*r));
        rig        = int(out.rigs.size()) - 1;
        rigOf[id]  = rig;
        if(item >= 0)
          out.items[item].rigs.push_back(rig);
        else if(mainNode.empty() && !options.attach)
        {
          mainNode      = id;
          out.mainRig   = rig;
          out.mainGroup = group;
        }
      }
    }
    jobs.push_back({&node, group, type, rig, int(scene.nodes[group].children.size()), item, resolved ? resolved->file : std::string()});
  }
  // the first three names and how many more
  auto someOf = [](const std::vector<std::string>& list) {
    std::string names;
    for(size_t i = 0; i < std::min<size_t>(3, list.size()); i++)
      names += (i ? ", " : "") + list[i];
    if(list.size() > 3)
      names += " and " + std::to_string(list.size() - 3) + " more";
    return names;
  };
  if(!dazInstances.empty())
    warn("DAZ Studio instance groups aren't drawn yet, so these are missing: " + someOf(dazInstances));

  // ---- meshes, one per geometry of each node, in file order
  std::unordered_map<std::string, int> materialCache;
  int                                  built = 0;
  for(const MeshJob& job : jobs)
  {
    std::vector<int>  made;
    std::vector<Mesh>    pending;
    std::vector<MeshRef> pendingRefs;
    std::vector<ItemPart> pendingParts;
    try
    {
      const Json* geos = get(job.node, "geometries");
      for(const Json& g : *geos)
      {
        auto gref = parseRef(getString(&g, "url"), file);
        if(!gref)
          continue;
        // a mesh that costs something is timed: its file, its build, its dForce data, its skin
        timing::Scope tm("mesh " + getString(&g, "id"), 10);
        auto          gres = [&] {
          timing::Scope tr("resolve geometry", 10);
          return store.resolve(*gref, "geometry_library");
        }();
        if(!gres)
        {
          warn("Missing geometry " + getString(&g, "url"));
          continue;
        }
        const Json& geo = *gres->entry;
        collectImages(gres->doc->json, images);
        const std::string gid  = getString(&g, "id");
        const auto        mine = sceneMaterialsFor(gid, getString(&geo, "id"));

        // UV set: a scene material may switch it (character presets do), otherwise the default
        std::string uvUri;
        for(const Json* m : mine)
          if(const Json* u = get(m, "uv_set"); u && u->is_string())
          {
            uvUri = u->get<std::string>();
            break;
          }
        if(uvUri.empty())
          uvUri = getString(&geo, "default_uv_set");
        std::optional<UvSet> uvSet;
        if(!uvUri.empty())
        {
          std::optional<Store::Resolved> ures;
          try
          {
            if(auto uref = parseRef(uvUri, gres->file))
              ures = store.resolve(*uref, "uv_set_library");
          }
          catch(const std::exception&)
          {
          }
          if(ures)
            uvSet = parseUvSet(*ures->doc, *ures->entry);
          else
            warn("Missing UV set " + uvUri);
        }
        // DSON marks SubD meshes; others (strand hair cards, props modelled as polygons) stay as they are
        const int level = getString(&geo, "type") == "subdivision_surface" ? std::clamp(options.subdivision, 0, 3) : 0;
        Mesh      meshOut;
        {
          timing::Scope tb("mesh::build", 10);
          meshOut.geo = std::make_shared<mesh::BuiltMesh>(mesh::build(meshInput(*gres->doc, geo, uvSet ? &*uvSet : nullptr, level, meshOut.materialGroups)));
        }

        // materials, one per polygon material group
        const Json* libMaterials = get(&gres->doc->json, "material_library");
        std::map<std::string, Channels> groupChannels;
        for(const std::string& name : meshOut.materialGroups)
        {
          const Json* lib = nullptr;
          if(libMaterials && libMaterials->is_array())
          {
            for(const Json& m : *libMaterials)
              if(const Json* groups = get(&m, "groups"); groups && groups->is_array()
                 && std::any_of(groups->begin(), groups->end(), [&](const Json& x) { return x.is_string() && x.get<std::string>() == name; }))
              {
                lib = &m;
                break;
              }
            if(!lib)
              for(const Json& m : *libMaterials)
                if(getString(&m, "id") == name || getString(&m, "name") == name)
                {
                  lib = &m;
                  break;
                }
          }
          // the geometry's own material, then each scene material for this group over the library
          // material it points at, merged in that order
          std::vector<std::pair<const Json*, std::string>> layers{{lib, gres->file}};
          std::vector<std::shared_ptr<const Doc>>          keep;
          for(const Json* sm : mine)
          {
            const Json* groups = get(sm, "groups");
            if(!groups || !groups->is_array()
               || !std::any_of(groups->begin(), groups->end(), [&](const Json& x) { return x.is_string() && x.get<std::string>() == name; }))
              continue;
            auto ref = parseRef(getString(sm, "url"), file);
            if(ref && ref->file != gres->file)
            {
              try
              {
                if(auto r2 = store.resolve(*ref, "material_library"))
                {
                  collectImages(r2->doc->json, images);
                  keep.push_back(r2->doc);
                  layers.emplace_back(r2->entry, r2->file);
                }
              }
              catch(const std::exception&)
              {
              }
            }
            layers.emplace_back(sm, file);
          }
          Channels ch;
          for(const auto& [def, from] : layers)
            collectChannels(def, ch, images, from);
          groupChannels[name]   = ch;
          const std::string key = materialKey(name, ch);
          auto              hit = materialCache.find(key);
          if(hit == materialCache.end())
          {
            scene.materials.push_back(buildMaterial(name, ch));
            hit = materialCache.emplace(key, int(scene.materials.size()) - 1).first;
          }
          meshOut.materials.push_back(hit->second);
        }
        // dForce settings, where the scene (or a material) asks for a simulation
        {
          timing::Scope td("buildDforce", 10);
          meshOut.dforce = buildDforce(*docPtr, gres->doc.get(), getString(job.node, "id"), gid, *gres->doc, geo, groupChannels);
        }
        if(meshOut.geo->ribbon)
        {
          std::vector<std::array<double, 2>> widths;
          for(int mi : meshOut.materials)
          {
            const Channels* chans = scene.materials[mi].channels.get();
            auto            width = [&](const char* n) -> std::optional<double> {
              const Channel* c = chans ? chans->get(norm(n)) : nullptr;
              return c && c->value.is_number() ? std::optional<double>(c->value.get<double>()) : std::nullopt;
            };
            widths.push_back(ribbonWidths(width("Line Start Width"), width("Line End Width"), meshOut.geo->ribbon->spacing));
          }
          widenRibbons(*meshOut.geo, widths);
        }

        // skin binding lives in the geometry's file, keyed to the geometry id
        if(job.rig >= 0)
        {
          timing::Scope ts("skin", 10);
          const Rig&    rig = out.rigs[job.rig];
          const Json* mods = get(&gres->doc->json, "modifier_library");
          const Json* skinMod = nullptr;
          if(mods && mods->is_array())
            for(const Json& m : *mods)
            {
              const Json* skin = get(&m, "skin");
              if(!skin || skin->is_null())
                continue;
              auto ref = parseRef(getString(skin, "geometry"), gres->file);
              if(!ref || ref->id.empty() || ref->id == getString(&geo, "id"))
              {
                skinMod = skin;
                break;
              }
            }
          if(skinMod)
            if(auto binding = skinBinding(*gres->doc, *skinMod, rig, meshOut.geo->dsonVertexCount))
            {
              meshOut.skin     = mesh::skin(*meshOut.geo, binding->input);
              meshOut.cageSkin = binding->input;
              meshOut.bones    = binding->order;
            }
        }

        meshOut.name = firstString({{&geo, "label"}, {&geo, "name"}}, gid);
        {
          MeshRef ref;
          ref.geo         = meshOut.geo;
          ref.geometryKey = lowerCase(gres->file + "#" + getString(&geo, "id"));
          ref.owners      = {getString(job.node, "id"), gid};
          pendingRefs.push_back(std::move(ref));
        }
        {
          ItemPart part;
          part.item     = job.item;
          part.conforms = job.rig >= 0 && meshOut.skinned();
          part.defFile  = job.defFile;
          if(job.item >= 0)
          {
            part.graft     = wardrobe::parseGraft(*gres->doc, geo);
            part.rigidity  = wardrobe::parseRigidity(*gres->doc, geo, meshOut.geo->dsonVertexCount);
            part.smoothing = parseSmoothing(*docPtr, gres->doc.get(), getString(job.node, "id"));
          }
          pendingParts.push_back(std::move(part));
        }
        pending.push_back(std::move(meshOut));
      }
      // a node whose build fails adds none of its meshes
      for(size_t pi = 0; pi < pending.size(); pi++)
      {
        Mesh& m = pending[pi];
        Node  meshNode;
        meshNode.name = m.name;
        meshNode.kind = NodeKind::eMesh;
        // the node's own mesh only: its children (a strand hair conforming to a hidden growth cap) stay
        meshNode.visible = !hiddenNode(job.node);
        meshNode.mesh    = int(scene.meshes.size());
        pendingRefs[pi].mesh = meshNode.mesh;
        pendingParts[pi].mesh = meshNode.mesh;
        if(job.item >= 0)
          out.parts.push_back(std::move(pendingParts[pi]));
        out.morphs.meshes.push_back(std::move(pendingRefs[pi]));
        scene.meshes.push_back(std::move(m));
        made.push_back(scene.add(std::move(meshNode)));
      }
    }
    catch(const MissingFileError&)
    {
      skip(*job.node, "", true);
    }
    catch(const std::exception& e)
    {
      skip(*job.node, e.what(), false);
    }
    // in file order, after the group's rig and before its child nodes
    for(size_t k = 0; k < made.size(); k++)
    {
      scene.attach(made[k], job.group, job.at + int(k));
      built++;
      if(job.item >= 0)
        out.items[job.item].meshes.push_back(scene.nodes[made[k]].mesh);
      if(scene.meshes[scene.nodes[made[k]].mesh].geo->fullIndex.empty())
        warn(scene.nodes[job.group].name + " has no polygons or strands, so it isn't drawn");
    }
    meshNodesOf[getString(job.node, "id")] = std::move(made);
  }

  // bind: as three's SkinnedMesh.bind(skeleton, mesh.matrixWorld) does with the scene at rest
  const auto worlds = scene.worlds();
  for(Node& n : scene.nodes)
  {
    if(n.kind != NodeKind::eMesh)
      continue;
    Mesh& m = scene.meshes[n.mesh];
    if(!m.skinned())
      continue;
    const int self = int(&n - scene.nodes.data());
    m.bindMatrix   = worlds[self];
    m.boneInverses.clear();
    for(int b : m.bones)
      m.boneInverses.push_back(inverse(worlds[b]));
  }

  // DAZ Studio node instances: what the target draws, drawn again under the instance's group. The
  // instance's transform takes the target's place, as DAZ Studio places it, and geometry is in rest
  // world space, so hanging the copies under the instance is all it takes. A copy shares its Mesh:
  // the glTF draws it from every node, and morphs and edits reach every copy.
  if(!nodeInstances.empty())
  {
    timing::Scope                                        ti("instances", 10);
    std::unordered_map<std::string, const NodeInstance*> instanceById;
    for(const NodeInstance& ni : nodeInstances)
      instanceById[ni.id] = &ni;
    // a drawn node copied under `parent`: a mesh node shares its Mesh, a group brings what hangs from it
    std::function<int(int, int, bool)> copy = [&](int src, int parent, bool visible) {
      Node n    = scene.nodes[src];
      n.visible = n.visible && visible;
      const std::vector<int> kids = n.children;
      const int              id   = scene.add(std::move(n), parent);
      for(int c : kids)
        copy(c, id, visible);
      return id;
    };
    std::unordered_map<std::string, std::vector<int>>   drawnOf;  // an instance -> the copies under its group
    std::unordered_map<std::string, std::string>        failed;   // an instance -> why it isn't drawn
    std::unordered_set<std::string>                     resolving;
    static const std::string                            kLoop = "they draw themselves";
    std::function<std::string(const std::string&)>      resolve;
    // scene node `x` drawn under `parent`: what it draws itself and, with `children`, its child nodes
    // (not the instances among them, which draw themselves); else why it can't be
    std::function<std::string(const std::string&, int, bool, bool, std::vector<int>&)> draw =
        [&](const std::string& x, int parent, bool children, bool visible, std::vector<int>& top) -> std::string {
      std::vector<int> source;
      if(instanceById.count(x))
      {
        if(std::string why = resolve(x); !why.empty())
          return why == kLoop ? why : "the instances they copy aren't drawn";
        source = drawnOf[x];
      }
      else if(auto it = meshNodesOf.find(x); it != meshNodesOf.end())
      {
        // a skinned node ignores its own transform, so a copy would stand where the figure does
        for(int n : it->second)
          if(scene.meshes[scene.nodes[n].mesh].skinned())
            return "they draw a posed figure";
        source = it->second;
      }
      else if(!instanceGroups.count(x))
        return "their target isn't in the scene";
      for(int n : source)
        top.push_back(copy(n, parent, visible));
      if(auto it = children ? childrenOf.find(x) : childrenOf.end(); it != childrenOf.end())
        for(const std::string& c : std::vector<std::string>(it->second))
        {
          if(instanceById.count(c))
            continue;
          Node g    = scene.nodes[instanceGroups.at(c)];
          g.visible = g.visible && visible;
          const int at = scene.add(std::move(g), parent);
          top.push_back(at);
          std::vector<int> under;
          draw(c, at, true, visible, under);  // a child that can't be drawn leaves an empty group
        }
      return {};
    };
    resolve = [&](const std::string& id) -> std::string {
      if(drawnOf.count(id))
        return {};
      if(auto it = failed.find(id); it != failed.end())
        return it->second;
      if(!resolving.insert(id).second)
        return kLoop;
      const NodeInstance& ni = *instanceById.at(id);
      std::vector<int>    top;
      std::string         why = ni.target.empty() ? "they name no target" : draw(ni.target, ni.group, ni.children, ni.visible, top);
      resolving.erase(id);
      if(why.empty())
        drawnOf[id] = std::move(top);
      else
        failed[id] = why;
      return why;
    };
    std::map<std::string, std::vector<std::string>> notDrawn;
    for(const NodeInstance& ni : nodeInstances)
      if(std::string why = resolve(ni.id); !why.empty())
        notDrawn[why].push_back(scene.nodes[ni.group].name);
    for(const auto& [why, names] : notDrawn)
      warn("Some DAZ Studio instances aren't drawn because " + why + ": " + someOf(names));
  }

  // what the morph engine works from
  out.morphs.mainNode = mainNode;
  out.morphs.itemOf.insert(itemOf.begin(), itemOf.end());
  if(!mainNode.empty())
  {
    out.morphs.mainOwners.insert(mainNode);
    for(size_t i = 0; i < out.morphs.meshes.size(); i++)
      if(out.morphs.meshes[i].owners[0] == mainNode)
      {
        out.morphs.mainMeshes.push_back(i);
        out.morphs.mainOwners.insert(out.morphs.meshes[i].owners.begin(), out.morphs.meshes[i].owners.end());
      }
    out.morphs.previewCenters = previewCenters.count(mainNode) > 0;
  }

  // what an item was authored for, for the wardrobe's fit
  {
    const std::string conform = wardrobe::conformTargetOf(doc);
    const auto        feet    = wardrobe::footChannels(doc);
    for(Item& item : out.items)
    {
      item.conformTarget = conform;
      item.footPose      = feet;
    }
  }
  out.missing = store.missing();
  if(!built)
  {
    if(!out.missing.empty())
      throw std::runtime_error(missingMessage(out.name, out.missing));
    std::string why = "No geometry could be built from this file.";
    for(size_t i = 0; i < std::min<size_t>(3, skipped.size()); i++)
      why += (i ? "; " : " ") + skipped[i];
    throw std::runtime_error(why);
  }
  if(!out.missing.empty())
    warn(missingMessage(out.name, out.missing));
  return out;
}

}  // namespace fv::dson
