#include <figure/dson/dforce.hpp>

#include <figure/dson/modifiers.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <regex>
#include <functional>

namespace fv::dson {

namespace {

const char* const kNodeType    = "studio/modifier/dynamic_simulation";
const char* const kSurfaceType = "studio/simulation_settings/dynamic_simulation";

using Values = ChannelValues;

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

float num(const Values& ch, const char* id, float fallback)
{
  return float(numberOf(ch, id, fallback));
}

bool flag(const Values& ch, const char* id, bool fallback)
{
  return flagOf(ch, id, fallback);
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

// Pair rows [vertex, weight] on an extra whose id, name or type mentions an influence map. The map
// is sparse: a vertex it leaves out has weight 0 and follows the skin (a skirt's waistband is never
// listed). Without a map every vertex is free.
std::vector<float> influenceMap(const Found& mod, int vertexCount)
{
  std::vector<float> out(size_t(vertexCount), 0.f);
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
    if(!hasExtraType(m, kNodeType) || modifierParent(m) != nodeId)
      continue;
    const Values ch  = mergedModifierChannels(m, scene, geo);
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
    if(!hasExtraType(m, kSurfaceType) || modifierParent(m) != geomId)
      continue;
    const Surface surface = surfaceFrom(mergedModifierChannels(m, scene, geo));
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
  // a vertex no polygon has takes the defaults: a strand hair's million points (lines, no polygons)
  // go without a sum each
  std::vector<Sum> sum(np ? static_cast<size_t>(nv) : 0);
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
    if(sum.empty())
      return std::vector<float>(static_cast<size_t>(nv), fallback);
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
  out->selfCollide.resize(size_t(nv), 1);
  out->surfaceVisible.resize(size_t(nv), 1);
  out->surfaceCollide.resize(size_t(nv), 1);
  if(sum.empty())
    out->layer.assign(size_t(nv), int16_t(std::lround(float(d.layer))));
  for(int v = 0; v < int(sum.size()); v++)
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

std::shared_ptr<DforceData> defaultDforce(int vertexCount, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts, int layer)
{
  const Surface d;
  const size_t  nv  = size_t(std::max(0, vertexCount));
  auto          out = std::make_shared<DforceData>();
  out->kind         = DforceKind::eDynamic;
  out->strength.assign(nv, d.strength);
  out->stretch.assign(nv, d.stretch);
  out->shear.assign(nv, d.shear);
  out->bend.assign(nv, d.bend);
  out->buckling.assign(nv, d.buckling);
  out->bucklingRatio.assign(nv, d.bucklingRatio);
  out->density.assign(nv, d.density);
  out->contraction.assign(nv, d.contraction);
  out->damping.assign(nv, d.damping);
  out->friction.assign(nv, d.friction);
  out->offset.assign(nv, d.offset);
  out->layer.assign(nv, int16_t(layer));
  out->selfCollide.assign(nv, d.selfCollide ? 1 : 0);
  out->surfaceVisible.assign(nv, 1);
  out->surfaceCollide.assign(nv, d.collide ? 1 : 0);
  out->polyStart.assign(1, 0);
  for(size_t p = 0; p + 1 < polyStart.size(); p++)
  {
    const size_t at = out->polyVerts.size();
    for(int32_t k = polyStart[p]; k < polyStart[p + 1]; k++)
      if(polyVerts[size_t(k)] >= 0 && polyVerts[size_t(k)] < vertexCount)
        out->polyVerts.push_back(polyVerts[size_t(k)]);
    if(out->polyVerts.size() - at < 2)
      out->polyVerts.resize(at);
    else
      out->polyStart.push_back(int32_t(out->polyVerts.size()));
  }
  return out;
}

}  // namespace fv::dson
