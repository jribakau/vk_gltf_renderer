#include <figure/dson/dforce.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <regex>
#include <functional>

namespace fv::dson {

namespace {

const char* const kNodeType    = "studio/modifier/dynamic_simulation";
const char* const kSurfaceType = "studio/simulation_settings/dynamic_simulation";

// channel id -> current value (number, flag or text)
using Values = std::map<std::string, Json>;

struct Surface
{
  bool  visible       = true;
  float friction      = 0.4f;
  float layer         = 0;
  bool  collide       = true;
  bool  selfCollide   = true;
  float offset        = 0.2f;
  float strength      = 1;
  float stretch       = 0.8f;
  float shear         = 0.2f;
  float bend          = 0.5f;
  float buckling      = 0.05f;
  float bucklingRatio = 0.7f;
  float density       = 180;
  float contraction   = 1;
  float damping       = 0.1f;
};

// A modifier and the document it lives in (whose tables its rows are packed in).
struct Found
{
  const Json* json = nullptr;
  const Doc*  doc  = nullptr;
};

std::string parentId(const Json& mod)
{
  std::string p = getString(&mod, "parent");
  if(p.empty() || p[0] != '#')
    return p;
  std::string out;
  for(size_t i = 1; i < p.size(); i++)
  {
    if(p[i] == '%' && i + 2 < p.size() + 0 && std::isxdigit((unsigned char)p[i + 1]) && std::isxdigit((unsigned char)p[i + 2]))
    {
      out += char(std::stoi(p.substr(i + 1, 2), nullptr, 16));
      i += 2;
    }
    else
      out += p[i];
  }
  return out;
}

bool hasExtra(const Json& mod, const char* type)
{
  const Json* extra = get(&mod, "extra");
  if(!extra || !extra->is_array())
    return false;
  for(const Json& e : *extra)
    if(getString(&e, "type") == type)
      return true;
  return false;
}

// Channel id -> current value, from studio_modifier_channels and the modifier's own channel.
Values channelMap(const Json& mod)
{
  Values     out;
  const auto put = [&](const Json& ch) {
    const Json* c  = get(&ch, "channel");
    const Json& cc = c && c->is_object() ? *c : ch;
    const Json* id = get(&cc, "id");
    if(!id || !id->is_string())
      return;
    const Json* v = get(&cc, "current_value");
    if(!v || v->is_null())
      v = get(&cc, "value");
    if(v && (v->is_number() || v->is_boolean() || v->is_string()))
      out[id->get<std::string>()] = *v;
  };
  if(const Json* extra = get(&mod, "extra"); extra && extra->is_array())
    for(const Json& ex : *extra)
      if(const Json* chans = get(&ex, "channels"); chans && chans->is_array())
        for(const Json& c : *chans)
          put(c);
  if(get(&mod, "channel"))
    put(mod);
  return out;
}

std::optional<Found> lookup(const Doc& scene, const Doc* geo, const std::string& url)
{
  if(url.empty())
    return std::nullopt;
  const auto ref = parseRef(url, "");
  if(!ref || ref->id.empty())
    return std::nullopt;
  for(const Doc* doc : {&scene, geo})
  {
    if(!doc)
      continue;
    const Json* lib = get(&doc->json, "modifier_library");
    if(!lib || !lib->is_array())
      continue;
    for(const Json& m : *lib)
      if(getString(&m, "id") == ref->id)
        return Found{&m, doc};
  }
  return std::nullopt;
}

// The definition's channels, then the scene instance's over any it repeats.
Values mergedChannels(const Json& instance, const Doc& scene, const Doc* geo)
{
  Values out;
  if(const auto def = lookup(scene, geo, getString(&instance, "url")))
    out = channelMap(*def->json);
  for(auto& [k, v] : channelMap(instance))
    out[k] = v;
  return out;
}

float num(const Values& ch, const char* id, float fallback)
{
  const auto it = ch.find(id);
  return it != ch.end() && it->second.is_number() ? it->second.get<float>() : fallback;
}

bool flag(const Values& ch, const char* id, bool fallback)
{
  const auto it = ch.find(id);
  if(it == ch.end())
    return fallback;
  if(it->second.is_boolean())
    return it->second.get<bool>();
  if(it->second.is_number())
    return it->second.get<double>() != 0;
  return fallback;
}

Surface surfaceFrom(const Values& ch)
{
  const Surface d;
  Surface       s;
  s.visible       = flag(ch, "Visible in Simulation", d.visible);
  s.friction      = num(ch, "Friction", d.friction);
  s.layer         = num(ch, "Collision Layer", d.layer);
  s.collide       = flag(ch, "Collide", d.collide);
  s.selfCollide   = flag(ch, "Self Collide", d.selfCollide);
  s.offset        = num(ch, "Collision Offset", d.offset);
  s.strength      = num(ch, "Dynamics Strength", d.strength);
  s.stretch       = num(ch, "Stretch Stiffness", d.stretch);
  s.shear         = num(ch, "Shear Stiffness", d.shear);
  s.bend          = num(ch, "Bend Stiffness", d.bend);
  s.buckling      = num(ch, "Buckling Stiffness", d.buckling);
  s.bucklingRatio = num(ch, "Buckling Ratio", d.bucklingRatio);
  s.density       = num(ch, "Density", d.density);
  s.contraction   = num(ch, "Contraction-Expansion Ratio", d.contraction);
  s.damping       = num(ch, "Damping", d.damping);
  return s;
}

DforceKind kindOf(const Values& ch)
{
  const int t = int(num(ch, "Simulation Object Type", 1));
  if(t == 0)
    return DforceKind::eStatic;
  if(t == 2)
    return DforceKind::eAddon;
  return DforceKind::eDynamic;
}

// Pair rows [vertex, weight] on an extra whose id, name or type mentions an influence map.
std::vector<float> influenceMap(const Found& mod, int vertexCount)
{
  std::vector<float> out(size_t(vertexCount), 1.f);
  bool               any = false;
  static const std::regex re("influence", std::regex::icase);
  const auto take = [&](const Json* table) {
    const auto t = mod.doc->rows(table);
    if(!t || !t->count || t->start.size() < 2)
      return;
    if(t->start[1] - t->start[0] < 2)
      return;
    for(int i = 0; i < t->count; i++)
    {
      const double v = t->cell(i, 0), w = t->cell(i, 1);
      if(v >= 0 && v < vertexCount && std::isfinite(w))
      {
        out[size_t(v)] = float(w);
        any            = true;
      }
    }
  };
  const std::function<void(const Json&, const std::string&)> walk = [&](const Json& node, const std::string& label) {
    if(node.is_array())
    {
      for(const Json& x : node)
        walk(x, label);
      return;
    }
    if(!node.is_object())
      return;
    const std::string name = label + " " + getString(&node, "id") + " " + getString(&node, "name") + " " + getString(&node, "type");
    const bool        inf  = std::regex_search(name, re);
    if(inf)
      if(const Json* v = get(&node, "values"))
        take(v);
    for(auto it = node.begin(); it != node.end(); ++it)
    {
      if(it.key() == "values" && inf)
        continue;
      if(it->is_object() || it->is_array())
        walk(*it, name + " " + it.key());
    }
  };
  if(const Json* extra = get(mod.json, "extra"); extra && extra->is_array())
    for(const Json& ex : *extra)
      walk(ex, getString(&ex, "type"));
  return any ? out : std::vector<float>{};
}

struct NodeSim
{
  DforceKind         kind = DforceKind::eDynamic;
  bool               freeze = false, visible = true;
  std::vector<float> influence;
};

std::optional<NodeSim> nodeSim(const Doc& scene, const Doc* geo, const std::string& nodeId, int vertexCount)
{
  const Json* mods = get(get(&scene.json, "scene"), "modifiers");
  if(!mods || !mods->is_array())
    return std::nullopt;
  for(const Json& m : *mods)
  {
    if(!hasExtra(m, kNodeType) || parentId(m) != nodeId)
      continue;
    const Values ch  = mergedChannels(m, scene, geo);
    const auto   def = lookup(scene, geo, getString(&m, "url"));
    NodeSim      out;
    out.kind      = kindOf(ch);
    out.freeze    = flag(ch, "Freeze Simulation", false);
    out.visible   = flag(ch, "Visible in Simulation", true);
    out.influence = influenceMap(def ? *def : Found{&m, &scene}, vertexCount);
    return out;
  }
  return std::nullopt;
}

std::map<std::string, Surface> surfacesFor(const Doc& scene, const Doc* geo, const std::string& geomId)
{
  std::map<std::string, Surface> out;
  const Json*                    mods = get(get(&scene.json, "scene"), "modifiers");
  if(!mods || !mods->is_array())
    return out;
  for(const Json& m : *mods)
  {
    if(!hasExtra(m, kSurfaceType) || parentId(m) != geomId)
      continue;
    const Surface surface = surfaceFrom(mergedChannels(m, scene, geo));
    if(const Json* groups = get(&m, "groups"); groups && groups->is_array())
      for(const Json& g : *groups)
        if(g.is_string())
          out[g.get<std::string>()] = surface;
  }
  return out;
}

// Material channels of the same names, where a product stored them on the surface instead.
Surface applyMaterialOverride(const Surface& surface, const Channels* ch)
{
  if(!ch)
    return surface;
  const auto n = [&](const char* name, float fallback) {
    const Channel* c = ch->get(norm(name));
    return c && c->value.is_number() ? c->value.get<float>() : fallback;
  };
  const auto b = [&](const char* name, bool fallback) {
    const Channel* c = ch->get(norm(name));
    if(c && c->value.is_boolean())
      return c->value.get<bool>();
    if(c && c->value.is_number())
      return c->value.get<double>() != 0;
    return fallback;
  };
  Surface s;
  s.visible       = b("Visible in Simulation", surface.visible);
  s.friction      = n("Friction", surface.friction);
  s.layer         = n("Collision Layer", surface.layer);
  s.collide       = b("Collide", surface.collide);
  s.selfCollide   = b("Self Collide", surface.selfCollide);
  s.offset        = n("Collision Offset", surface.offset);
  s.strength      = n("Dynamics Strength", surface.strength);
  s.stretch       = n("Stretch Stiffness", surface.stretch);
  s.shear         = n("Shear Stiffness", surface.shear);
  s.bend          = n("Bend Stiffness", surface.bend);
  s.buckling      = n("Buckling Stiffness", surface.buckling);
  s.bucklingRatio = n("Buckling Ratio", surface.bucklingRatio);
  s.density       = n("Density", surface.density);
  s.contraction   = n("Contraction-Expansion Ratio", surface.contraction);
  s.damping       = n("Damping", surface.damping);
  return s;
}

}  // namespace

std::shared_ptr<DforceData> buildDforce(const Doc& scene, const Doc* geoDoc, const std::string& nodeId, const std::string& geomId,
                                        const Doc& geometryDoc, const Json& geometry, const std::map<std::string, Channels>& overrides)
{
  const Json* polylist = get(&geometry, "polylist");
  const Json* vertices = get(&geometry, "vertices");
  const auto  polys    = geometryDoc.rows(polylist ? get(polylist, "values") : nullptr);
  const auto  verts    = geometryDoc.rows(vertices ? get(vertices, "values") : nullptr);
  const int   nv       = verts ? verts->count : 0;
  std::vector<std::string> groups;
  if(const Json* g = get(get(&geometry, "polygon_material_groups"), "values"); g && g->is_array())
    for(const Json& v : *g)
      groups.push_back(v.is_string() ? v.get<std::string>() : v.dump());
  if(groups.empty())
    groups.push_back("Default");

  const auto node    = nodeSim(scene, geoDoc, nodeId, nv);
  const auto byGroup = surfacesFor(scene, geoDoc, geomId);
  if(!node && byGroup.empty())
    return nullptr;
  const int np = polys ? polys->count : 0;

  std::vector<Surface> perPoly;
  perPoly.reserve(size_t(np));
  for(int pi = 0; pi < np; pi++)
  {
    const double      mi   = polys->cell(pi, 1);
    const std::string name = mi >= 0 && size_t(mi) < groups.size() ? groups[size_t(mi)] : groups[0];
    const auto        it   = byGroup.find(name);
    const auto        ov   = overrides.find(name);
    perPoly.push_back(applyMaterialOverride(it != byGroup.end() ? it->second : Surface{}, ov != overrides.end() ? &ov->second : nullptr));
  }

  struct Sum
  {
    double strength = 0, stretch = 0, shear = 0, bend = 0, buckling = 0, bucklingRatio = 0, density = 0, contraction = 0, damping = 0,
           friction = 0, offset = 0, layer = 0, self = 0, visible = 0, collide = 0;
    int n = 0;
  };
  std::vector<Sum> sum(static_cast<size_t>(nv));
  for(int pi = 0; pi < np; pi++)
  {
    const Surface& s = perPoly[size_t(pi)];
    for(int k = polys->start[pi] + 2; k < polys->start[pi + 1]; k++)
    {
      const double d = polys->data[k];
      if(d < 0 || d >= nv)
        continue;
      Sum& a = sum[size_t(d)];
      a.n++;
      a.strength += s.strength;
      a.stretch += s.stretch;
      a.shear += s.shear;
      a.bend += s.bend;
      a.buckling += s.buckling;
      a.bucklingRatio += s.bucklingRatio;
      a.density += s.density;
      a.contraction += s.contraction;
      a.damping += s.damping;
      a.friction += s.friction;
      a.offset += s.offset;
      a.layer += s.layer;
      a.self += s.selfCollide ? 1 : 0;
      a.visible = std::max(a.visible, s.visible ? 1.0 : 0.0);
      a.collide = std::max(a.collide, s.collide ? 1.0 : 0.0);
    }
  }
  const Surface d;
  const auto    fill = [&](double Sum::*field, float fallback) {
    std::vector<float> a(static_cast<size_t>(nv));
    for(int v = 0; v < nv; v++)
      a[size_t(v)] = sum[size_t(v)].n ? float(sum[size_t(v)].*field / sum[size_t(v)].n) : fallback;
    return a;
  };

  auto out        = std::make_shared<DforceData>();
  out->kind       = node ? node->kind : DforceKind::eDynamic;
  out->freeze     = node ? node->freeze : false;
  out->visible    = node ? node->visible : true;
  if(node)
    out->influence = node->influence;
  out->strength = fill(&Sum::strength, d.strength);
  if(!out->influence.empty())
    for(int v = 0; v < nv; v++)
      out->strength[size_t(v)] *= out->influence[size_t(v)];
  out->stretch       = fill(&Sum::stretch, d.stretch);
  out->shear         = fill(&Sum::shear, d.shear);
  out->bend          = fill(&Sum::bend, d.bend);
  out->buckling      = fill(&Sum::buckling, d.buckling);
  out->bucklingRatio = fill(&Sum::bucklingRatio, d.bucklingRatio);
  out->density       = fill(&Sum::density, d.density);
  out->contraction   = fill(&Sum::contraction, d.contraction);
  out->damping       = fill(&Sum::damping, d.damping);
  out->friction      = fill(&Sum::friction, d.friction);
  out->offset        = fill(&Sum::offset, d.offset);
  const auto layers  = fill(&Sum::layer, d.layer);
  out->layer.resize(size_t(nv));
  out->selfCollide.resize(size_t(nv));
  out->surfaceVisible.resize(size_t(nv));
  out->surfaceCollide.resize(size_t(nv));
  for(int v = 0; v < nv; v++)
  {
    const Sum& a = sum[size_t(v)];
    out->layer[size_t(v)]          = int16_t(std::lround(layers[size_t(v)]));
    out->selfCollide[size_t(v)]    = a.n ? (a.self / a.n >= 0.5 ? 1 : 0) : 1;
    out->surfaceVisible[size_t(v)] = a.n ? (a.visible > 0 ? 1 : 0) : 1;
    out->surfaceCollide[size_t(v)] = a.n ? (a.collide > 0 ? 1 : 0) : 1;
  }
  // polygon loops for the constraints, without vertices the geometry doesn't have (a malformed
  // row would index past the per-vertex arrays) or loops left with fewer than two corners
  out->polyStart.assign(1, 0);
  out->polyVerts.clear();
  for(int pi = 0; pi < np; pi++)
  {
    const size_t at = out->polyVerts.size();
    for(int k = polys->start[pi] + 2; k < polys->start[pi + 1]; k++)
    {
      const double d = polys->data[k];
      if(d >= 0 && d < nv)
        out->polyVerts.push_back(int32_t(d));
    }
    if(out->polyVerts.size() - at < 2)
      out->polyVerts.resize(at);
    else
      out->polyStart.push_back(int32_t(out->polyVerts.size()));
  }
  return out;
}

}  // namespace fv::dson
