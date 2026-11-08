// What a Figure wears: wearables are loaded and bound to the body on a worker (prepareWear), merged
// into the scene on the owning thread (commitWear), and taken off again (removeItem). A port of
// src/lib/wardrobe/Wardrobe.ts and the item-fitting half of src/lib/loaders/dson/figure.ts.

#include <figure/figure.hpp>

#include <algorithm>
#include <memory>
#include <cstdint>
#include <map>
#include <regex>
#include <utility>

#include <figure/physics/skin.hpp>
#include <figure/dson/materials.hpp>
#include <figure/dson/listing.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/nif/nif_wear.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/bind_cache.hpp>
#include <figure/wardrobe/collider.hpp>
#include <figure/wardrobe/fit/collide.hpp>
#include <figure/wardrobe/fit/fit.hpp>
#include <figure/wardrobe/fit/regions.hpp>
#include <figure/wardrobe/fit/heel.hpp>
#include <figure/wardrobe/follow.hpp>

namespace fv {

namespace {

std::string lowerCase(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

std::string stemOf(const std::string& file)
{
  const size_t slash = file.rfind('/');
  std::string  name  = slash == std::string::npos ? file : file.substr(slash + 1);
  if(name.size() >= 4 && lowerCase(name.substr(name.size() - 4)) == ".dsf")
    name.resize(name.size() - 4);
  return name;
}

std::string kindOf(const std::string& rel, const std::string& label, const std::vector<std::pair<int, std::string>>& objects)
{
  static const std::regex hair(R"(/hair/|\bhair\b|brow|lash)");
  static const std::regex anatomyDir(R"(/anatomy/)");
  static const std::regex anatomy(R"(/anatomy/|\beyes?\b|mouth|tear|genital|anatomical|teeth)");
  const std::string       path = lowerCase(rel + "/" + label);
  if(std::any_of(objects.begin(), objects.end(), [](const auto& o) { return !o.second.empty(); }))
    return "Prop";
  if(std::regex_search(path, hair) && !std::regex_search(path, anatomyDir))
    return "Hair";
  if(std::regex_search(path, anatomy))
    return "Anatomy";
  return "Clothing";
}

// How many of an item's vertices hang from a foot (their strongest bone's): which of an outfit's items
// is its shoe
size_t footVertices(const SceneGraph& scene, const std::vector<int>& meshes)
{
  size_t n = 0;
  for(int m : meshes)
    for(int16_t l : wardrobe::meshLimbs(scene, scene.meshes[size_t(m)]))
      n += l == wardrobe::limb::kLFoot || l == wardrobe::limb::kRFoot;
  return n;
}

// Each material as its meshes' contexts say, the first of them when they differ
void placeSurfaces(SceneGraph& scene, const std::map<int, mat::Context>& meshes)
{
  std::map<int, mat::Context> of;
  for(const auto& [mesh, context] : meshes)
    if(mesh >= 0 && mesh < int(scene.meshes.size()))
      for(int mi : scene.meshes[size_t(mesh)].materials)
        if(const auto [it, added] = of.emplace(mi, context); !added)
          it->second = std::min(it->second, context);
  for(const auto& [mi, context] : of)
  {
    if(mi < 0 || mi >= int(scene.materials.size()))
      continue;
    Material& m = scene.materials[size_t(mi)];
    m.context   = context;
    if(m.channels && mat::classify(m.name, context) != m.surface.kind)
      m.surface = dson::parseSurface(m.name, *m.channels, context);
    m.surface.context = context;
  }
}

// How near the body a strand point binds to the body under it, cm; further out it moves with the
// point above it on its strand. Hair lying on a shoulder or over the chest is within this; a strand
// hanging free past it follows where it last touched.
constexpr float kStrandReach = 3.f;

// An item morph whose deltas address vertices the item does not have belongs to another generation.
bool targetFits(const mesh::RenderTarget& t, int count)
{
  return std::all_of(t.cageIndices.begin(), t.cageIndices.end(), [&](uint32_t i) { return int(i) < count; });
}

// Whether an item morph that has no figure morph of the same name is worth a slider: fit adjustments
// and styles are; copies of figure morphs that aren't installed are not.
bool isAdjustment(const std::string& id)
{
  static const std::regex plumbing(R"((^|_)(bs|cbs|ctrl|jcm|pjcm|facs)_)", std::regex::icase);
  static const std::regex adjusts(R"(loosen|expand|tighten|shrink|adjust|collision|smooth|length|style|open|close|fit(?!ness))", std::regex::icase);
  return !std::regex_search(id, plumbing) || std::regex_search(id, adjusts);
}

// The item's own morph file for a channel, or null. DAZ's morph tools tag each file's modifier id with
// a hex suffix ("Mirel Head-0xa98947a" on the body, "-0xa989482" on the eyes); the copies share the name
// and the file stem.
const std::string* ownMorphFile(const std::map<std::string, std::string>& own, const std::string& channel)
{
  static const std::regex hex(R"(-0x[0-9a-f]+$)", std::regex::icase);
  const std::string       lower = lowerCase(channel);
  auto                    it    = own.find(lower);
  // only a tagged name pays for the regex: a driver asks for each dial it moves that the body has no target for
  if(it == own.end() && lower.find("-0x") != std::string::npos)
    it = own.find(std::regex_replace(lower, hex, ""));
  return it == own.end() ? nullptr : &it->second;
}

// "body_bs_LoosenNeck" -> "Loosen Neck"; the morph's own label replaces it once loaded.
std::string prettyMorphName(const std::string& id)
{
  static const std::regex prefix(R"(^(body|head|facs)_(bs|ctrl)_)", std::regex::icase);
  static const std::regex underscores("_+");
  static const std::regex camel("([a-z])([A-Z])");
  std::string             s = std::regex_replace(id, prefix, "");
  s                         = std::regex_replace(s, underscores, " ");
  s                         = std::regex_replace(s, camel, "$1 $2");
  const size_t a            = s.find_first_not_of(' ');
  const size_t b            = s.find_last_not_of(' ');
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// `parts` are the load's, with their meshes `meshBase` before the item's own (0 for the scene file's
// items, the merge offset for a wear); `index` is the item's place in that load.
WornItem record(const dson::Item& it, int index, const std::vector<dson::ItemPart>& parts, int meshBase, const std::string& rel, int id)
{
  WornItem w;
  w.id        = id;
  w.rel       = rel;
  w.key       = it.key;
  w.label     = it.label;
  w.objects   = it.objects;
  w.rigs      = it.rigs;
  w.meshes    = it.meshes;
  w.nodeNames = it.nodeNames;
  w.sources   = it.sources;
  w.kind      = kindOf(rel, it.label, it.objects);
  // what the collider keeps out of the body: a conforming mesh with the file's smoothing modifier, or
  // clothing with the modifier's defaults (DAZ Studio adds one to clothing as it is fitted)
  for(const dson::ItemPart& part : parts)
  {
    if(part.item != index || !part.conforms)
      continue;
    if(part.smoothing)
      w.collision[part.mesh + meshBase] = *part.smoothing;
    else if(w.kind == "Clothing")
      w.collision[part.mesh + meshBase] = dson::Smoothing{};
  }
  return w;
}

}  // namespace

struct Figure::PreparedWear
{
  dson::LoadResult loaded;
  // per item of `loaded`: how it was refitted when it was made for another generation
  std::vector<std::optional<wardrobe::FitDetail>> fits;
  // per item of `loaded`: known to be made for this figure's rig (wardrobe::madeForFigure)
  std::vector<bool> ownRig;
  // per entry of loaded.parts: the surface binding of a conforming part
  std::vector<std::shared_ptr<const wardrobe::SurfaceBinding>> bindings;
  std::string                                                  key;
  // per entry of loaded.parts: the body targets prepareWear was given, projected onto the part
  std::vector<std::vector<std::pair<dson::TargetPtr, dson::TargetPtr>>> preprojected;
  // the wearable's own morph files, per Morphs folder: stem (lowercase) -> file
  std::map<std::string, std::map<std::string, std::string>> ownMorphs;
};

void Figure::indexItems()
{
  items.clear();
  for(size_t i = 0; i < loaded.items.size(); i++)
    items.push_back(record(loaded.items[i], int(i), loaded.parts, 0, "", m_nextItem++));
}

const SceneGraph& Figure::preparedScene(const PreparedWear& wear)
{
  return wear.loaded.scene;
}

SceneGraph& Figure::preparedScene(PreparedWear& wear)
{
  return wear.loaded.scene;
}

const WornItem* Figure::item(int id) const
{
  for(const WornItem& w : items)
    if(w.id == id)
      return &w;
  return nullptr;
}

std::shared_ptr<const SceneGraph> Figure::wearBase() const
{
  timing::Scope t("wearBase copy");
  auto          out = std::make_shared<SceneGraph>();
  out->nodes        = loaded.scene.nodes;
  for(Node& n : out->nodes)
    if(n.joint)
      n.joint = std::make_shared<JointData>(*n.joint);
  // of the meshes a wear reads the body: the others' skins are as large as what has been worn
  out->meshes.resize(loaded.scene.meshes.size());
  if(!loaded.morphs.mainMeshes.empty())
  {
    const size_t body = size_t(loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh);
    out->meshes[body] = loaded.scene.meshes[body];
  }
  return out;
}

std::vector<dson::TargetPtr> Figure::activeBodyTargets() const
{
  std::vector<dson::TargetPtr> out;
  if(!morphs || loaded.morphs.mainMeshes.empty())
    return out;
  for(const auto& [target, weight] : morphs->activeTargets(loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh))
    out.push_back(target);
  return out;
}

// The wearable's own morph files in defFile's Morphs folder: stem (lowercase) -> file, listed once
// per folder into `ownMorphs`.
const std::map<std::string, std::string>& ownMorphsOf(const dson::Store& store, const std::string& defFile,
                                                      std::map<std::string, std::map<std::string, std::string>>& ownMorphs)
{
  const size_t      slash = defFile.rfind('/');
  const std::string dir   = (slash == std::string::npos ? std::string() : defFile.substr(0, slash)) + "/Morphs";
  auto              it    = ownMorphs.find(dir);
  if(it == ownMorphs.end())
  {
    std::map<std::string, std::string> found;
    for(const std::string& f : dson::listFiles(store.library(), dir, ".dsf"))
      found[lowerCase(stemOf(f))] = f;
    it = ownMorphs.emplace(dir, std::move(found)).first;
  }
  return it->second;
}

Figure::WearInputs Figure::wearInputs() const
{
  if(loaded.mainRig < 0 || loaded.morphs.mainMeshes.empty())
    throw std::runtime_error("Load a figure first: clothing and hair go on a figure with a skeleton.");
  WearInputs in;
  in.scene       = wearBase();
  in.rig         = std::make_shared<dson::Rig>(loaded.rigs[size_t(loaded.mainRig)]);
  in.bodyMesh    = loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh;
  in.morphDir    = loaded.morphs.morphDir;
  in.bodyTargets = activeBodyTargets();
  return in;
}

namespace {
void releaseMesh(Mesh& mesh)
{
  mesh.geo.reset();
  mesh.skin     = {};
  mesh.cageSkin = {};
  mesh.dforce.reset();
}
}  // namespace

WearQuality oneStepDown(const WearQuality& quality)
{
  // a hair gives its second pieces first: no strand is lost
  const bool pieces = quality.strandPieces > 1;
  return {std::max(quality.subdivision - 1, 0), pieces ? quality.strandKeep : std::max(quality.strandKeep / 2, kLeastStrandKeep), 1};
}

WearQuality oneStepUp(const WearQuality& quality, int subdivision)
{
  const bool strands = quality.strandKeep < 1;
  return {std::min(quality.subdivision + 1, std::max(subdivision, quality.subdivision)), std::min(quality.strandKeep * 2, 1.f),
          strands ? quality.strandPieces : 2};
}

WearQuality qualityOf(const SceneGraph& scene, const std::vector<int>& meshes)
{
  WearQuality out{0, 0, 0};
  bool        strands = false;
  for(int i : meshes)
    if(i >= 0 && i < int(scene.meshes.size()) && scene.meshes[size_t(i)].geo && scene.meshes[size_t(i)].geo->source)
    {
      const mesh::MeshInput& source = *scene.meshes[size_t(i)].geo->source;
      if(source.lines)
      {
        out.strandKeep   = std::max(out.strandKeep, source.lines->keep);
        out.strandPieces = std::max(out.strandPieces, source.lines->pieces);
        strands          = true;
      }
      else
        out.subdivision = std::max(out.subdivision, source.level);
    }
  if(!strands)
  {
    out.strandKeep   = 1;
    out.strandPieces = 2;
  }
  return out;
}

float shareAt(const Mesh& mesh, const WearQuality& quality)
{
  if(!mesh.geo || !mesh.geo->source)
    return 1;
  const mesh::MeshInput& source = *mesh.geo->source;
  if(source.lines)
    return source.lines->keep > 0 && source.lines->pieces > 0
               ? quality.strandKeep / source.lines->keep * float(quality.strandPieces) / float(source.lines->pieces)
               : 1;
  // a level is four faces for each
  return mesh.geo->subdivisionSurface ? float(std::pow(4.0, quality.subdivision - source.level)) : 1;
}

std::shared_ptr<RemadeItem> Figure::remakeInputs(int id, const WearQuality& quality) const
{
  const auto worn = std::find_if(items.begin(), items.end(), [&](const WornItem& w) { return w.id == id; });
  if(worn == items.end())
    return nullptr;
  auto out     = std::make_shared<RemadeItem>();
  out->item    = id;
  out->quality = quality;
  for(int i : worn->meshes)
  {
    if(i < 0 || i >= int(loaded.scene.meshes.size()))
      continue;
    const Mesh& m = loaded.scene.meshes[size_t(i)];
    if(!m.geo || !m.geo->source || shareAt(m, quality) == 1)
      continue;
    RemadeItem::Part part;
    part.mesh = i;
    part.from = m.geo;
    part.cage = m.geo->cagePositions;
    part.hidden = m.geo->hiddenPolys;
    if(!m.geo->ribbon)
      part.cageSkin = m.cageSkin;
    out->parts.push_back(std::move(part));
  }
  if(out->parts.empty())
    return nullptr;
  out->scene.materials = loaded.scene.materials;
  out->scene.meshes.resize(loaded.scene.meshes.size());
  for(const RemadeItem::Part& part : out->parts)
    out->scene.meshes[size_t(part.mesh)].materials = loaded.scene.meshes[size_t(part.mesh)].materials;
  return out;
}

void Figure::buildRemade(RemadeItem& item)
{
  for(RemadeItem::Part& part : item.parts)
  {
    timing::Scope t("remade", 10);
    part.made = mesh::remade(*part.from, part.cage, item.quality.subdivision, item.quality.strandKeep, item.quality.strandPieces);
    part.made->hiddenPolys = part.hidden;
    if(!part.cageSkin.start.empty())
      part.skin = mesh::skin(*part.made, part.cageSkin);
    item.scene.meshes[size_t(part.mesh)].geo = part.made;
  }
}

bool Figure::commitRemade(RemadeItem& item, int subdivision)
{
  const auto worn = std::find_if(items.begin(), items.end(), [&](const WornItem& w) { return w.id == item.item; });
  if(worn == items.end())
    return false;
  for(const RemadeItem::Part& part : item.parts)
    if(!part.made || part.mesh >= int(loaded.scene.meshes.size()) || loaded.scene.meshes[size_t(part.mesh)].geo != part.from
       || part.from->cagePositions != part.cage)
      return false;
  for(RemadeItem::Part& part : item.parts)
  {
    Mesh& m                = loaded.scene.meshes[size_t(part.mesh)];
    part.made->hiddenPolys = m.geo->hiddenPolys;
    m.geo                  = part.made;
    m.skin                 = std::move(part.skin);
    if(morphs)
      morphs->meshRemade(part.mesh);
  }
  worn->reduced = wearShortfall(loaded.scene, worn->meshes, qualityOf(loaded.scene, worn->meshes), subdivision);
  return true;
}

bool wearCanShrink(const SceneGraph& scene, const WearQuality& quality)
{
  for(const Mesh& m : scene.meshes)
    if(m.geo && (m.geo->ribbon ? quality.strandPieces > 1 || quality.strandKeep > kLeastStrandKeep
                               : m.geo->subdivisionSurface && quality.subdivision > 0))
      return true;
  return false;
}

std::string wearShortfall(const SceneGraph& scene, const std::vector<int>& meshes, const WearQuality& quality, int subdivision)
{
  bool strands = false, smooth = false;
  for(int i : meshes)
    if(i >= 0 && i < int(scene.meshes.size()) && scene.meshes[size_t(i)].geo)
    {
      const mesh::BuiltMesh& geo = *scene.meshes[size_t(i)].geo;
      strands                    = strands || geo.ribbon;
      smooth                     = smooth || (!geo.ribbon && geo.subdivisionSurface);
    }
  std::string out;
  if(strands && quality.strandKeep < 1)
    out = quality.strandKeep > kLeastStrandKeep ? "half its strands" : "a quarter of its strands";
  else if(strands && quality.strandPieces < 2)
    out = "its strands in half the triangles";
  if(smooth && quality.subdivision < subdivision)
    out += (out.empty() ? "Subdivision " : " and Subdivision ") + std::to_string(quality.subdivision);
  return out;
}

std::shared_ptr<Figure::PreparedWear> Figure::prepareWear(const std::string& rel, const WearQuality& quality, const WearInputs& inputs,
                                                          std::optional<bool> strandsFlipped)
{
  // what it reads of a parsed file stays while it runs
  const dson::Store::Hold hold(*m_store);
  auto out = std::make_shared<PreparedWear>();
  // a key of its own until commitWear gives it its place among the worn items
  static std::atomic<int> s_prepares{0};
  out->key = "p" + std::to_string(++s_prepares);
  dson::LoadOptions options;
  options.subdivision = quality.subdivision;
  options.strandKeep   = quality.strandKeep;
  options.strandPieces = quality.strandPieces;
  const SceneGraph& scene = *inputs.scene;
  options.attach      = dson::AttachTarget{&scene, inputs.rig.get(), out->key};
  const std::vector<dson::TargetPtr>& bodyTargets = inputs.bodyTargets;
  const std::string file = rel.substr(std::min(rel.find_first_not_of('/'), rel.size()));
  timing::Scope     t("prepareWear " + file);
  {
    timing::Scope tl("loadDson");
    // a Skyrim outfit's mesh: one the app installed by its library path, else from wherever it is,
    // its textures linked into the first content folder
    if(nif::isNif(rel))
    {
      const auto&           roots = m_store->library().roots();
      std::filesystem::path path(std::u8string(reinterpret_cast<const char8_t*>(rel.data()), rel.size()));
      if(!path.is_absolute())
      {
        const auto found = m_store->library().find(file);
        if(!found)
          throw std::runtime_error(file + " isn't in the content folders: was its mod uninstalled?");
        path = *found;
      }
      out->loaded = nif::loadWearable(path, options, {roots.empty() ? std::filesystem::path() : roots.front()});
    }
    else
      out->loaded = dson::loadDson(*m_store, file, options);
  }
  if(out->loaded.items.empty())
    throw std::runtime_error("Nothing in this file can be worn.");

  // an item made for another generation is refitted first: its cage moves onto this figure's skeleton
  // and shape, and the bindings below are made against the cage it ends with
  const int bodyMesh = inputs.bodyMesh;
  {
    timing::Scope      tf("fitItem");
    const std::string& morphDir = inputs.morphDir;
    wardrobe::FitFigure figure{scene, *inputs.rig, bodyMesh, morphDir.substr(0, morphDir.rfind("/Morphs"))};
    for(const dson::Item& it : out->loaded.items)
    {
      out->fits.push_back(wardrobe::fitItem(*m_store, out->loaded, it, rel, figure));
      out->ownRig.push_back(wardrobe::madeForFigure(out->loaded, it, rel, figure));
    }
  }
  // A strand hair's directions, measured beside the binding: their weld runs on one thread while the
  // bind keeps the pool busy. Both only read the item's scene. The future waits in its destructor,
  // so an error below cannot leave it reading a scene that is gone.
  std::future<void> strands;
  if(strandsFlipped)
    strands = std::async(std::launch::async, [&scene = std::as_const(out->loaded.scene), flipped = *strandsFlipped] {
      try
      {
        build::measureStrands(scene, flipped);
      }
      catch(const std::exception&)
      {
        // measured again after the presets, where an error is reported
      }
    });
  // every conforming part is bound to the body's unmorphed cage, in parallel. A skinned part binds
  // limb to limb (its left-thigh vertices to the left thigh): a legging's inner seam is as close to
  // the other leg, and a body morph that moves one leg would then drag the other's cloth
  auto          tb        = std::make_unique<timing::Scope>("bindToSurface");
  const auto&   body      = *scene.meshes[size_t(bodyMesh)].geo;
  const auto    bodyLimbs = [&] {
    timing::Scope tl("body limbs", 5);
    return wardrobe::meshLimbs(scene, scene.meshes[size_t(bodyMesh)]);
  }();
  out->bindings.resize(out->loaded.parts.size());
  std::vector<std::future<void>> jobs;
  auto                           tItems = std::make_unique<timing::Scope>("item limbs", 5);  // the filters, built here before each job
  for(size_t i = 0; i < out->loaded.parts.size(); i++)
  {
    const dson::ItemPart& part = out->loaded.parts[i];
    if(!part.conforms)
      continue;
    const Mesh&            itemMesh = out->loaded.scene.meshes[size_t(part.mesh)];
    const mesh::BuiltMesh& item     = *itemMesh.geo;
    auto filter = std::make_shared<std::optional<wardrobe::BindFilter>>(wardrobe::limbFilter(bodyLimbs, wardrobe::meshLimbs(out->loaded.scene, itemMesh)));
    jobs.push_back(jobs::pool().submit([&body, &item, &part, filter, slot = &out->bindings[i]] {
      const wardrobe::BindFilter* f = *filter ? &**filter : nullptr;
      // strand hair binds along each strand (see bindStrands); everything else point by point
      if(!item.cageLineStart.empty())
      {
        *slot = std::make_shared<wardrobe::SurfaceBinding>(
            wardrobe::cachedBindStrands(body.cagePositions, body.dsonTris, item.cagePositions, item.cageLineStart, item.cageLineVerts, kStrandReach, f));
        return;
      }
      // a graft's seam pairs number the body it was made for: on another mesh they'd weld it to strangers
      const std::span<const int32_t> pairs =
          part.graft && wardrobe::graftFits(*part.graft, body) ? std::span<const int32_t>(part.graft->pairs) : std::span<const int32_t>();
      *slot = std::make_shared<wardrobe::SurfaceBinding>(wardrobe::cachedBindToSurface(body.cagePositions, body.dsonTris, item.cagePositions, pairs, f));
    }));
  }
  tItems.reset();
  // every job ends before any error is rethrown: they write into `out`
  for(auto& j : jobs)
    j.wait();
  for(auto& j : jobs)
    j.get();
  tb.reset();
  // the wearable's own morph folders, listed here rather than at commit
  {
    timing::Scope tl("list own morphs", 1);
    for(const dson::ItemPart& part : out->loaded.parts)
      ownMorphsOf(*m_store, part.defFile, out->ownMorphs);
  }
  // the body's dialled-in targets projected onto every conforming part, each on a pool thread:
  // what the followers would otherwise project at commit, on the owning thread
  out->preprojected.resize(out->loaded.parts.size());
  if(!bodyTargets.empty())
  {
    timing::Scope                  tp("project followers");
    std::vector<std::future<void>> projections;
    for(size_t i = 0; i < out->loaded.parts.size(); i++)
    {
      const dson::ItemPart& part = out->loaded.parts[i];
      if(!part.conforms || !out->bindings[i])
        continue;
      auto& slot = out->preprojected[i];
      slot.resize(bodyTargets.size());
      const auto item = out->loaded.scene.meshes[size_t(part.mesh)].geo;
      for(size_t k = 0; k < bodyTargets.size(); k++)
        projections.push_back(jobs::pool().submit([&body, item, &part, binding = out->bindings[i], target = bodyTargets[k], cell = &slot[k]] {
          std::vector<float> scratch;
          *cell = {target, wardrobe::projectTarget(*binding, part.rigidity ? &*part.rigidity : nullptr, body, item->dsonVertexCount, *target, scratch)};
        }));
    }
    for(auto& j : projections)
      j.wait();
    for(auto& j : projections)
      j.get();
  }
  if(strands.valid())
  {
    timing::Scope ts("strands beside the bind (wait)", 1);
    strands.get();
  }
  return out;
}

void Figure::placeLoadedSurfaces(const std::string& rel)
{
  // a file that is a wearable or a prop has no body in it, though its own figure is the main one
  std::string type;
  try
  {
    if(const auto doc = m_store->load(rel))
      type = doc->json.value("asset_info", dson::Json::object()).value("type", "");
  }
  catch(const std::exception&)
  {
  }
  const std::string file     = kindOf(rel, "", {});
  const bool        hairFile = file == "Hair";
  const bool        bodyFile = file == "Anatomy" || (loaded.mainRig >= 0 && type != "wearable" && type != "prop" && !hairFile);
  std::map<int, mat::Context> meshes;
  // in a body's file, what is skinned and no item is part of the figure (Genesis 9's mouth, its eyes)
  for(size_t m = 0; m < loaded.scene.meshes.size(); m++)
    meshes[int(m)] = hairFile                                                                   ? mat::Context::eHair
                     : bodyFile && loaded.scene.meshes[m].skinned() ? mat::Context::eBody
                                                                                                       : mat::Context::eItem;
  if(bodyFile)
    for(size_t i : loaded.morphs.mainMeshes)
      meshes[loaded.morphs.meshes[i].mesh] = mat::Context::eBody;
  for(const dson::Item& it : loaded.items)
  {
    const std::string kind = kindOf(rel, it.label, it.objects);
    // a second figure in the file is a body as well: rigged and following nothing
    const mat::Context context = kind == "Hair" ? mat::Context::eHair
                                 : kind == "Anatomy" || (bodyFile && !it.rigs.empty() && it.conformTarget.empty()) ? mat::Context::eBody
                                                                                                                   : mat::Context::eItem;
    for(int m : it.meshes)
      if(const auto at = meshes.find(m); at != meshes.end())
        at->second = std::min(at->second, context);
  }
  placeSurfaces(loaded.scene, meshes);
}

std::vector<int> Figure::itemsReplacedBy(const PreparedWear& wear, const std::vector<int>& ids, bool only) const
{
  const dson::LoadResult& w = wear.loaded;
  std::set<std::string>   sources;
  for(size_t k = 0; k < w.items.size(); k++)
    if(!only || (k < ids.size() && ids[k] > 0))
      sources.insert(w.items[k].sources.begin(), w.items[k].sources.end());
  std::vector<int> old;
  for(const WornItem& i : items)
    if(std::any_of(i.sources.begin(), i.sources.end(), [&](const std::string& s) { return sources.count(s) > 0; }))
      old.push_back(i.id);
  return old;
}

WearResult Figure::commitWear(std::shared_ptr<PreparedWear> wear, const std::string& rel, const std::vector<int>& ids,
                              const std::vector<std::string>& keys, bool only, bool posesFeet)
{
  timing::Scope     t("commitWear " + rel);
  PreparedWear&     p   = *wear;
  dson::LoadResult& w   = p.loaded;
  SceneGraph&       dst = loaded.scene;
  WearResult        result;
  result.warnings = w.warnings;
  // what can fail for want of memory fails here, with the figure as it was: the merge below only moves
  {
    size_t mounts = 0;
    for(const dson::Item& it : w.items)
      mounts += it.objects.size();
    dst.nodes.reserve(dst.nodes.size() + w.scene.nodes.size() + mounts);
    dst.meshes.reserve(dst.meshes.size() + w.scene.meshes.size());
    dst.materials.reserve(dst.materials.size() + w.scene.materials.size());
    loaded.rigs.reserve(loaded.rigs.size() + w.rigs.size());
    loaded.items.reserve(loaded.items.size() + w.items.size());
    loaded.morphs.meshes.reserve(loaded.morphs.meshes.size() + w.morphs.meshes.size());
    items.reserve(items.size() + w.items.size());
  }
  // the same garment again replaces the worn one: the newer copy carries the file's fit settings
  for(int id : itemsReplacedBy(p, ids, only))
  {
    removeItem(id);
    result.replaced.push_back(id);
  }

  // its key now, in the order wears are put on: item k of the load is key, key.1, key.2 ... as the
  // loader names them. One coming back has the key it had, and later wears stay clear of it.
  for(const std::string& k : keys)
    if(k.size() > 1)
      m_wears = std::max(m_wears.load(), std::atoi(k.c_str() + 1));
  p.key = "w" + std::to_string(++m_wears);
  for(size_t k = 0; k < w.items.size(); k++)
  {
    const bool had = k < keys.size() && !keys[k].empty()
                     && std::none_of(items.begin(), items.end(), [&](const WornItem& i) { return i.key == keys[k]; });
    w.items[k].key = had ? keys[k] : p.key + (k ? "." + std::to_string(k) : std::string());
  }

  // ---- merge the scene: nodes, meshes, materials and skeletons, with their indices moved on
  const int nodeBase = int(dst.nodes.size()), meshBase = int(dst.meshes.size()), materialBase = int(dst.materials.size()),
            rigBase = int(loaded.rigs.size()), itemBase = int(loaded.items.size());
  for(fv::Node& n : w.scene.nodes)
  {
    if(n.parent >= 0)
      n.parent += nodeBase;
    for(int& c : n.children)
      c += nodeBase;
    if(n.mesh >= 0)
      n.mesh += meshBase;
    dst.nodes.push_back(std::move(n));
  }
  for(fv::Mesh& m : w.scene.meshes)
  {
    for(int& mi : m.materials)
      mi += materialBase;
    for(int& b : m.bones)
      b += nodeBase;
    dst.meshes.push_back(std::move(m));
  }
  for(fv::Material& m : w.scene.materials)
    dst.materials.push_back(std::move(m));
  for(dson::Rig& r : w.rigs)
  {
    r.root += nodeBase;
    for(auto& [id, node] : r.bones)
      node += nodeBase;
    loaded.rigs.push_back(std::move(r));
  }
  // the wearable's own root is dropped: its objects hang from the figure
  dst.nodes[size_t(nodeBase + w.root)].removed = true;
  const dson::Rig& figureRig = loaded.rigs[size_t(loaded.mainRig)];
  const int        body      = loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh;

  std::vector<int> added;
  // whatever fails from here on takes it off again, as the user would
  struct TakeOff
  {
    Figure&                 figure;
    const std::vector<int>& added;
    int                     nodeBase, meshBase;
    bool                    done = false;
    ~TakeOff()
    {
      if(done)
        return;
      for(int id : added)
        try
        {
          figure.removeItem(id);
        }
        catch(const std::exception&)
        {
        }
      // and what hadn't become an item yet
      SceneGraph& scene = figure.loaded.scene;
      for(size_t n = size_t(nodeBase); n < scene.nodes.size(); n++)
        scene.nodes[n].removed = true;
      for(size_t m = size_t(meshBase); m < scene.meshes.size(); m++)
        releaseMesh(scene.meshes[m]);
    }
  } takeOff{*this, added, nodeBase, meshBase};
  for(size_t k = 0; k < w.items.size(); k++)
  {
    dson::Item it = w.items[k];
    for(auto& [node, bone] : it.objects)
    {
      node += nodeBase;
      int parent = loaded.mainGroup;
      if(!bone.empty())
      {
        const int b = figureRig.find(dst, bone);
        if(b < 0)
          result.missingBones.push_back(bone);
        else
        {
          // a group under the bone that cancels its rest offset, so nodes in figure space can hang from it
          fv::Node mount;
          mount.name = dst.nodes[size_t(b)].name + " attachment";
          if(const auto& j = dst.nodes[size_t(b)].joint)
            mount.position = {-j->center[0], -j->center[1], -j->center[2]};
          parent = dst.add(std::move(mount), b);
        }
      }
      dst.attach(node, parent);
      if(parent != loaded.mainGroup)
        node = parent;
    }
    for(int& r : it.rigs)
      r += rigBase;
    for(int& m : it.meshes)
      m += meshBase;
    loaded.items.push_back(it);
    // an id it had, unless an item worn now has it; a new one after every id given out
    int id = k < ids.size() && ids[k] > 0 && std::none_of(items.begin(), items.end(), [&](const WornItem& i) { return i.id == ids[k]; }) ? ids[k] : 0;
    if(id)
      m_nextItem = std::max(m_nextItem, id + 1);
    else
      id = m_nextItem++;
    WornItem worn = record(it, int(k), w.parts, meshBase, rel, id);
    added.push_back(worn.id);
    items.push_back(std::move(worn));
  }
  // the wearable's morph meshes and the items they belong to, for later loads of item morph files
  std::vector<dson::MeshRef> itemRefs;
  for(dson::MeshRef ref : w.morphs.meshes)
  {
    ref.mesh += meshBase;
    for(dson::MeshRef::Shell& s : ref.shells)
      s.mesh += meshBase;
    itemRefs.push_back(ref);
    loaded.morphs.meshes.push_back(ref);
  }
  for(auto& [owner, idx] : w.morphs.itemOf)
    loaded.morphs.itemOf[owner] = idx + itemBase;
  const auto itemOfMesh = [&](int mesh) -> WornItem* {
    for(size_t k = 0; k < added.size(); k++)
      for(WornItem& i : items)
        if(i.id == added[k] && std::find(i.meshes.begin(), i.meshes.end(), mesh) != i.meshes.end())
          return &i;
    return nullptr;
  };
  const auto refOf = [&](int mesh) {
    for(const dson::MeshRef& r : itemRefs)
      if(r.mesh == mesh)
        return r;
    return dson::MeshRef{};
  };

  // ---- skeletons follow the figure's bones; the match says whether an item of no known generation was
  // made for this figure. One known to share its rig has nothing to prove
  int matched = 0, total = 0;
  for(size_t k = 0; k < added.size(); k++)
    if(const WornItem* worn = item(added[k]))
    {
      const auto& fit = k < p.fits.size() ? p.fits[k] : std::optional<wardrobe::FitDetail>{};
      pose::PoseRig::FollowPlan plan;
      if(fit)
      {
        plan.follow  = &fit->plan.follow;
        plan.inherit = &fit->plan.inherit;
        result.notes.push_back(fit->note);
      }
      for(int r : worn->rigs)
        if(pose)
        {
          const auto [m, t] = pose->addFollowers(loaded.rigs[size_t(r)], plan);
          if(k < p.ownRig.size() && p.ownRig[k])
            continue;
          matched += m;
          total += t;
        }
    }
  result.fit = total ? double(matched) / double(total) : 1.0;

  // A heel's file poses the figure's feet, as DAZ Studio applies its scene animations to the figure
  // it goes on, made for this figure or refitted to it: bone rotations, or the figure's foot dials (a
  // G9 heel may only set body_ctrl_lFootHeeledShoe, whose formulas turn the bones). A heel rests
  // shaped for that pose, around a flat foot with its heel below the floor (every heel in the
  // library, G8 and G9, 3 to 13 cm), so without it the shoe is bent and the foot flat. All of a
  // file's items carry its pose; it is set once, and the item with the most of its vertices on the
  // feet keeps what it set (an outfit's shoes, not its dress), so the feet go back when that one
  // comes off. A dial the figure lacks (a G9 heel on G8) is left out.
  if(pose && posesFeet && !w.items.empty())
  {
    std::vector<std::string> names;
    for(const auto& [id, node] : loaded.rigs[size_t(loaded.mainRig)].bones)
      names.push_back(dst.nodes[size_t(node)].name);
    const std::vector<wardrobe::FootChannel> feet  = wardrobe::retargetFootPose(w.items[0].footPose, names);
    WornItem*                                owner = nullptr;
    size_t                                   most  = 0;
    if(!feet.empty())
      for(int id : added)
        for(WornItem& i : items)
          if(i.id == id)
            if(const size_t on = added.size() > 1 ? footVertices(dst, i.meshes) : 0; !owner || on > most)
            {
              owner = &i;
              most  = on;
            }
    std::vector<pose::ChannelValue>             set;
    std::vector<std::pair<std::string, double>> dials;
    for(const wardrobe::FootChannel& c : feet)
      if(!owner)
        break;
      else if(!c.dial.empty())
      {
        if(morphs->channel(c.dial))
        {
          dials.emplace_back(c.dial, c.value);
          owner->footSet.push_back({"", "", c.value, morphs->get(c.dial), c.dial});
        }
      }
      else if(const auto before = pose->posed(c.bone, "rotation", c.axis))
      {
        set.push_back({c.bone, pose::ChannelKind::eRotation, c.axis, c.value});
        owner->footSet.push_back({c.bone, c.axis, c.value, *before});
      }
    if(!dials.empty())
      morphs->setMany(dials);
    if(!set.empty())
      pose->setMany(set);
    // what the rig and the dials kept after their limits, so undoing compares like with like
    if(owner)
      for(WornItem::FootSet& f : owner->footSet)
        if(!f.dial.empty())
          f.value = morphs->get(f.dial);
        else if(const auto now = pose->posed(f.bone, "rotation", f.axis))
          f.value = *now;
  }

  // ---- the wearable's own morph dials, then the items' fitting to the body's shape
  const std::set<std::string> before = [&] {
    std::set<std::string> names;
    for(const morph::Channel& c : morphs->channels())
      names.insert(c.name);
    return names;
  }();
  {
    timing::Scope                       td("other dials");
    std::vector<const dson::SceneDial*> dials;
    for(const dson::SceneDial& d : w.morphs.dials)
      dials.push_back(&d);
    dson::MorphContext ctx = w.morphs;
    ctx.meshes             = itemRefs;
    // itemOf indexes the wearable's own items
    applyOtherDials(ctx, w.items, dials, {});
  }
  // the wearable's own morph files, listed by prepareWear (or here, for a folder it did not meet)
  const auto listOwn = [&](const std::string& defFile) -> const std::map<std::string, std::string>& {
    return ownMorphsOf(*m_store, defFile, p.ownMorphs);
  };
  const auto&   bodyGeoPtr = dst.meshes[size_t(body)].geo;
  timing::Scope tp("followers and own morphs");
  for(size_t i = 0; i < w.parts.size(); i++)
  {
    const dson::ItemPart& part = w.parts[i];
    const int             mesh = part.mesh + meshBase;
    WornItem*             worn = itemOfMesh(mesh);
    if(!worn)
      continue;
    const auto& own    = listOwn(part.defFile);
    // a count, not the mesh: it may be made again (remade) while these live, and the old one should go
    const int   itemVertices = dst.meshes[size_t(mesh)].geo->dsonVertexCount;
    const dson::MeshRef ref = refOf(mesh);
    if(part.conforms && p.bindings[i])
    {
      std::erase_if(worn->follows, [&](const WornItem::Follow& had) { return had.mesh == mesh; });
      if(dst.meshes[size_t(mesh)].geo->cageLineStart.empty())
        worn->follows.push_back({mesh, p.bindings[i], part.rigidity ? std::make_shared<const wardrobe::Rigidity>(*part.rigidity) : nullptr});
      morph::Follower f;
      f.mesh   = mesh;
      f.source = body;
      // the scratch is the calling thread's: the morph set projects several targets side by side
      f.project = [binding = p.bindings[i], rigidity = part.rigidity, bodyGeo = bodyGeoPtr, itemVertices](const mesh::RenderTarget& t) -> dson::TargetPtr {
        thread_local std::vector<float> scratch;
        return wardrobe::projectTarget(*binding, rigidity ? &*rigidity : nullptr, *bodyGeo, itemVertices, t, scratch);
      };
      const auto files = std::make_shared<const std::map<std::string, std::string>>(own);
      f.label          = worn->label;
      f.hasOwn         = [files](const std::string& channel) { return ownMorphFile(*files, channel) != nullptr; };
      f.own            = [store = m_store, files, ref, itemVertices](const std::string& channel) -> std::shared_future<std::vector<dson::TargetPtr>> {
        const std::string* file = ownMorphFile(*files, channel);
        if(!file)
          return {};
        return jobs::pool()
            .submit([store, file = *file, ref, itemVertices]() -> std::vector<dson::TargetPtr> {
              std::vector<dson::TargetPtr> out;
              try
              {
                for(const dson::MorphPart& part : dson::loadMorphFile(*store, file, {ref}))
                  if(part.mesh == ref.mesh && part.target && targetFits(*part.target, itemVertices))
                    out.push_back(part.target);
              }
              catch(const std::exception&)
              {
              }
              return out;
            })
            .share();
      };
      if(i < p.preprojected.size())
        f.preprojected = std::move(p.preprojected[i]);
      {
        timing::Scope ta("addFollower", 5);
        morphs->addFollower(std::move(f));
      }
      if(part.graft && !wardrobe::graftFits(*part.graft, *bodyGeoPtr))
      {
        const std::string why = worn->label + " is a geograft made for another figure's mesh: here it can't hide the skin under it or join its seam to the body.";
        if(std::find(result.warnings.begin(), result.warnings.end(), why) == result.warnings.end())
          result.warnings.push_back(why);
      }
      else if(part.graft && !part.graft->hidden.empty())
      {
        auto& hidden = dst.meshes[size_t(body)].geo->hiddenPolys[worn->key];
        hidden.insert(hidden.end(), part.graft->hidden.begin(), part.graft->hidden.end());
        worn->hiddenPolys.insert(worn->hiddenPolys.end(), part.graft->hidden.begin(), part.graft->hidden.end());
      }
    }
    // morphs that aren't copies of a figure morph: fit adjustments ("Loosen Neck") and styles
    timing::Scope tAdj("adjustment dials", 5);
    for(const auto& [stem, file] : own)
    {
      const std::string id = stemOf(file);
      if(morphs->has(id) || dson::hiddenMorph(id) || !isAdjustment(id))
        continue;
      const std::string name = id + "@" + worn->key;
      if(morphs->has(name))
        continue;
      morph::ChannelOptions opts;
      opts.group = worn->label;
      opts.label = prettyMorphName(id);
      morphs->addLazy(
          name,
          [store = m_store, file, ref]() -> morph::Finish {
            auto parts = dson::loadMorphFile(*store, file, {ref});
            return [parts = std::move(parts)]() mutable {
              morph::LazyResult r;
              r.parts = std::move(parts);
              return r;
            };
          },
          opts);
      worn->channels.push_back(name);
    }
  }
  // dials the wearable brought, besides its adjustments: each leaves with the item whose meshes it
  // moves (the first item when it moves none yet, a file not read)
  if(!added.empty())
  {
    std::vector<WornItem*> wornNow;
    for(int id : added)
      for(WornItem& i : items)
        if(i.id == id)
          wornNow.push_back(&i);
    for(const morph::Channel& c : morphs->channels())
    {
      if(before.count(c.name) || wornNow.empty())
        continue;
      const std::set<int> moved = morphs->meshesOf(c.name);
      WornItem*           owner = wornNow.front();
      for(WornItem* w : wornNow)
        if(std::any_of(w->meshes.begin(), w->meshes.end(), [&](int m) { return moved.count(m) > 0; }))
        {
          owner = w;
          break;
        }
      if(std::find(owner->channels.begin(), owner->channels.end(), c.name) == owner->channels.end())
        owner->channels.push_back(c.name);
    }
  }
  {
    timing::Scope tf("flush");
    morphs->flush();
    morphs->sort();
    morphs->markLoaded(before);
  }
  // what it is on says what its surfaces' names mean: a graft is part of the body it goes on
  {
    std::map<int, mat::Context> meshes;
    for(int id : added)
      for(const WornItem& i : items)
        if(i.id == id)
          for(int m : i.meshes)
            meshes[m] = i.kind == "Hair"                           ? mat::Context::eHair
                        : i.kind == "Anatomy" || !i.hiddenPolys.empty() ? mat::Context::eBody
                                                                        : mat::Context::eItem;
    placeSurfaces(dst, meshes);
  }
  result.items = added;
  takeOff.done = true;
  // what the parsed files only this store holds
  m_store->trim();
  return result;
}

bool Figure::setItemKeepFull(int id, bool on)
{
  const auto it = std::find_if(items.begin(), items.end(), [&](const WornItem& i) { return i.id == id; });
  if(it == items.end())
    return false;
  it->keepFull = on;
  for(int m : it->meshes)
    if(m >= 0 && m < int(loaded.scene.meshes.size()))
      for(int mi : loaded.scene.meshes[size_t(m)].materials)
        if(mi >= 0 && mi < int(loaded.scene.materials.size()))
          loaded.scene.materials[size_t(mi)].keepFull = on;
  return true;
}

bool Figure::removeItem(int id)
{
  const auto it = std::find_if(items.begin(), items.end(), [&](const WornItem& i) { return i.id == id; });
  if(it == items.end())
    return false;
  WornItem    gone = std::move(*it);
  items.erase(it);
  SceneGraph& scene = loaded.scene;
  for(const auto& [node, bone] : gone.objects)
    scene.nodes[size_t(node)].removed = true;
  if(pose)
  {
    for(int r : gone.rigs)
      pose->removeFollowers(loaded.rigs[size_t(r)]);
    // the feet go back where they were before the item posed them, unless they were moved since
    std::vector<pose::ChannelValue>             back;
    std::vector<std::pair<std::string, double>> dialsBack;
    for(const WornItem::FootSet& f : gone.footSet)
      if(!f.dial.empty())
      {
        if(morphs->channel(f.dial) && morphs->get(f.dial) == f.value)
          dialsBack.emplace_back(f.dial, f.before);
      }
      else if(const auto now = pose->posed(f.bone, "rotation", f.axis); now && *now == f.value)
        back.push_back({f.bone, pose::ChannelKind::eRotation, f.axis, f.before});
    if(!dialsBack.empty())
      morphs->setMany(dialsBack);
    if(!back.empty())
      pose->setMany(back);
  }
  const std::set<int> goneMeshes(gone.meshes.begin(), gone.meshes.end());
  morphs->removeChannels(gone.channels);
  morphs->removeMeshes(goneMeshes);
  if(!loaded.morphs.mainMeshes.empty())
  {
    auto& hidden = scene.meshes[size_t(loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh)].geo->hiddenPolys;
    hidden.erase(gone.key);
    hidden.erase(gone.key + "#tight");
  }
  // its morph meshes and owner ids leave the load's morph context; the main meshes' indices follow
  dson::MorphContext&        ctx = loaded.morphs;
  std::vector<dson::MeshRef> kept;
  std::vector<size_t>        newIndex(ctx.meshes.size(), SIZE_MAX);
  for(size_t i = 0; i < ctx.meshes.size(); i++)
  {
    if(goneMeshes.count(ctx.meshes[i].mesh))
    {
      for(const std::string& owner : ctx.meshes[i].owners)
        ctx.itemOf.erase(owner);
      continue;
    }
    newIndex[i] = kept.size();
    kept.push_back(std::move(ctx.meshes[i]));
  }
  ctx.meshes = std::move(kept);
  std::vector<size_t> mains;
  for(size_t i : ctx.mainMeshes)
    if(i < newIndex.size() && newIndex[i] != SIZE_MAX)
      mains.push_back(newIndex[i]);
  ctx.mainMeshes = std::move(mains);
  // what it drew with goes, unless something still shown draws it; the slots stay, as the indices of
  // what was worn after it count on them
  const std::vector<bool> live = liveMeshes(scene);
  for(int m : gone.meshes)
    if(m >= 0 && m < int(live.size()) && !live[size_t(m)])
      releaseMesh(scene.meshes[size_t(m)]);
  m_store->trim();
  return true;
}

std::vector<bool> walkThroughMaterials(const SceneGraph& scene, const std::vector<WornItem>& items)
{
  std::vector<bool> passes(scene.meshes.size(), false);
  for(const WornItem& item : items)
    if(item.kind != "Anatomy" && item.hiddenPolys.empty())
      for(int m : item.meshes)
        if(m >= 0 && m < int(passes.size()))
          passes[size_t(m)] = true;
  // a taken-off item's meshes bound nothing, though no worn item holds them any more
  const std::vector<bool> live = liveMeshes(scene);
  std::vector<bool>       through(scene.materials.size(), true), bounds(scene.materials.size(), false);
  for(size_t m = 0; m < scene.meshes.size(); m++)
    if(!passes[m] && live[m])
      for(int mi : scene.meshes[m].materials)
        if(mi >= 0 && mi < int(bounds.size()))
          bounds[size_t(mi)] = true;
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
  {
    const mat::Kind kind = scene.materials[mi].surface.kind;
    through[mi] = kind == mat::Kind::eLashes || kind == mat::Kind::eHair || kind == mat::Kind::eCornea || !bounds[mi];
  }
  return through;
}

std::shared_ptr<const dson::DforceData> clothSettings(const mesh::BuiltMesh& geo, const std::shared_ptr<const dson::DforceData>& file,
                                                      std::optional<bool> on, float strength, int layer, float volume)
{
  const bool fileSimulates = file && file->simulated();
  const bool simulate      = on ? *on : fileSimulates;
  // strand hair: the file's settings, the strands falling or not
  if(file && file->hasStrands())
  {
    if(simulate == fileSimulates && strength == 1.f && volume == file->strandVolume)
      return file;
    auto data          = std::make_shared<dson::DforceData>(*file);
    data->strandsOn    = simulate;
    data->strandVolume = volume;
    data->freeze    = false;
    data->visible   = true;
    if(data->kind == dson::DforceKind::eStatic)
      data->kind = dson::DforceKind::eDynamic;
    const float s = std::clamp(strength, 0.f, 1.f);
    for(float& w : data->strandStrength)
      w *= s;
    return data;
  }
  if(!simulate)
  {
    if(!fileSimulates)
      return file;
    // the file's dynamic surface, held: it conforms and the cloth over it collides with it
    auto held    = std::make_shared<dson::DforceData>(*file);
    held->freeze = true;
    return held;
  }
  // as the file has it, untouched
  if(fileSimulates && !on && strength == 1.f)
    return file;
  std::shared_ptr<dson::DforceData> data;
  if(file && file->hasPolygons())
    data = std::make_shared<dson::DforceData>(*file);
  else if(geo.cagePolyStart.size() > 1)
    data = dson::defaultDforce(geo.dsonVertexCount, geo.cagePolyStart, geo.cagePolyVerts, layer);
  else
    return file;
  data->freeze  = false;
  data->visible = true;
  if(data->kind == dson::DforceKind::eStatic)
    data->kind = dson::DforceKind::eDynamic;
  const float s = std::clamp(strength, 0.f, kClothLoosest);
  for(float& w : data->strength)
    w = s <= 1.f ? w * s : w > 0 ? w + (1.f - w) * (s - 1.f) : 0.f;
  return data;
}

float clothLoosest(const std::shared_ptr<const dson::DforceData>& file)
{
  if(file && std::any_of(file->strength.begin(), file->strength.end(), [](float w) { return w > 0 && w < 1; }))
    return kClothLoosest;
  return 1.f;
}

std::vector<int> Figure::setItemCloth(int id, std::optional<bool> on, float strength, float volume)
{
  const auto it = std::find_if(items.begin(), items.end(), [&](WornItem& i) { return i.id == id; });
  if(it == items.end())
    return {};
  it->cloth.on       = on;
  it->cloth.strength = std::clamp(strength, 0.f, kClothLoosest);
  it->cloth.volume   = std::clamp(volume, 0.f, 1.f);
  // an item given settings here joins the top collision layer the figure's garments use, so it
  // keeps the side of each it started on (a belt over the skirt, a bra under the shirt) rather than
  // being pushed under a garment on a higher layer
  int layer = 0;
  for(const Mesh& mesh : loaded.scene.meshes)
    if(mesh.dforce && mesh.dforce->hasPolygons())
      for(int16_t l : mesh.dforce->layer)
        layer = std::max<int>(layer, l);
  // on a hair, the switch is for its strands: the cap they grow from stays on the head
  const auto strandsOf = [&](int m) {
    const auto saved = it->fileDforce.find(m);
    const auto& file = saved != it->fileDforce.end() ? saved->second : loaded.scene.meshes[size_t(m)].dforce;
    return file && file->hasStrands();
  };
  const bool hair = std::any_of(it->meshes.begin(), it->meshes.end(), [&](int m) { return m >= 0 && size_t(m) < loaded.scene.meshes.size() && strandsOf(m); });
  std::vector<int> changed;
  for(int m : it->meshes)
  {
    if(m < 0 || size_t(m) >= loaded.scene.meshes.size())
      continue;
    Mesh& mesh = loaded.scene.meshes[size_t(m)];
    if(!mesh.geo || (hair && !strandsOf(m)))
      continue;
    // the file's own settings, kept the first time the user changes this item
    if(!it->fileDforce.count(m))
      it->fileDforce[m] = mesh.dforce;
    auto next = clothSettings(*mesh.geo, it->fileDforce[m], on, it->cloth.strength, layer, it->cloth.volume);
    if(next != mesh.dforce)
    {
      mesh.dforce = std::move(next);
      changed.push_back(m);
    }
  }
  return changed;
}

bool Figure::setItemTight(int id, float amount)
{
  const auto it = std::find_if(items.begin(), items.end(), [&](WornItem& i) { return i.id == id; });
  if(it == items.end())
    return false;
  it->tight = std::clamp(amount, 0.f, 1.f);
  return true;
}

namespace {

// Skin under it: how close the fabric lies as made, and how far from its hems the skin has to be for a
// pose not to bare it.
constexpr float kCoverReach    = 2.f;
constexpr float kCoverHemClear = 3.f;

// Whether nothing shows through any surface of the mesh.
bool opaqueMesh(const SceneGraph& scene, const Mesh& mesh)
{
  for(const int m : mesh.materials)
  {
    if(m < 0 || size_t(m) >= scene.materials.size())
      continue;
    const Material&      mat = scene.materials[size_t(m)];
    const mat::Surface&  s   = mat.surface;
    const float          o   = mat.edits.opacity.value_or(s.opacity.value);
    if(o < 1 || !s.opacity.map.empty() || (s.refraction && (s.refraction->weight > 0 || !s.refraction->map.empty())))
      return false;
    if(mat.edits.alphaMode && *mat.edits.alphaMode != "opaque")
      return false;
  }
  return true;
}

}  // namespace

bool Figure::refreshTightHidden(bool collisionOn)
{
  if(loaded.morphs.mainMeshes.empty())
    return false;
  SceneGraph& scene = loaded.scene;
  const int   body  = loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh;
  auto&       geo   = *scene.meshes[size_t(body)].geo;
  bool        changed = false;
  for(WornItem& w : items)
  {
    const std::string key  = w.key + "#tight";
    // read every time: fabric made see-through later gives the skin back
    bool opaque = true;
    for(const auto& [mesh, smoothing] : w.collision)
      opaque = opaque && opaqueMesh(scene, scene.meshes[size_t(mesh)]);
    const bool want = collisionOn && w.visible && w.tight > 0 && !w.collision.empty() && opaque && !clothOf(w).on;
    if(!want)
    {
      if(w.tightChecked)
      {
        w.tightChecked = false;
        w.tightPolys.clear();
        changed = geo.hiddenPolys.erase(key) > 0 || changed;
      }
      continue;
    }
    if(w.tightChecked)
      continue;
    w.tightChecked = true;
    timing::Scope t("tight: covered skin", 5);
    const auto worlds = scene.worlds();
    const auto nodeOf = physics::nodeOfMesh(scene);
    const auto inRoot = [&](int mesh, const std::vector<float>& cage) {
      const Mat4 to = inverse(worlds[size_t(loaded.root)]) * (nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : Mat4{});
      return physics::transformPoints(to, cage);
    };
    const auto bodyCage = inRoot(body, morphs->morphedCage(body));
    for(const auto& [mesh, smoothing] : w.collision)
    {
      const Mesh& m = scene.meshes[size_t(mesh)];
      if(!m.geo)
        continue;
      const auto own  = morphs->cpuTarget(wardrobe::collideChannelOf(mesh), mesh);
      const auto cage = inRoot(mesh, morphs->morphedCage(mesh, own.get()));
      const auto polys = wardrobe::coveredPolys(bodyCage, geo.cagePolyStart, geo.cagePolyVerts, cage, m.geo->dsonTris, kCoverReach, kCoverHemClear);
      w.tightPolys.insert(w.tightPolys.end(), polys.begin(), polys.end());
    }
    if(!w.tightPolys.empty())
    {
      geo.hiddenPolys[key] = w.tightPolys;
      changed = true;
    }
  }
  return changed;
}

bool Figure::fallsByItself(const std::vector<int>& ids) const
{
  for(const WornItem& w : items)
    if(std::find(ids.begin(), ids.end(), w.id) != ids.end())
      for(int m : w.meshes)
        if(m >= 0 && size_t(m) < loaded.scene.meshes.size() && loaded.scene.meshes[size_t(m)].dforce && loaded.scene.meshes[size_t(m)].dforce->simulatesStrands())
          return true;
  return false;
}

ClothState Figure::clothOf(const WornItem& item) const
{
  ClothState s;
  s.strength = item.cloth.strength;
  s.volume   = item.cloth.volume;
  for(int m : item.meshes)
  {
    if(m < 0 || size_t(m) >= loaded.scene.meshes.size())
      continue;
    const Mesh& mesh = loaded.scene.meshes[size_t(m)];
    const auto  file = item.fileDforce.count(m) ? item.fileDforce.at(m) : mesh.dforce;
    if(!mesh.geo || (mesh.geo->cagePolyStart.size() < 2 && !(file && file->hasStrands())))
      continue;
    s.able = true;
    if(file && (file->hasPolygons() || file->hasStrands()))
      s.native = true;
    s.loosest = std::max(s.loosest, clothLoosest(file));
    if(file && file->hasStrands())
      s.strands = true;
    if(mesh.dforce && mesh.dforce->simulated())
      s.on = true;
  }
  return s;
}

bool Figure::setItemVisible(int id, bool visible)
{
  const auto it = std::find_if(items.begin(), items.end(), [&](WornItem& i) { return i.id == id; });
  if(it == items.end())
    return false;
  it->visible = visible;
  for(const auto& [node, bone] : it->objects)
    loaded.scene.nodes[size_t(node)].visible = visible;
  // a hidden geograft gives the body its own surface back
  if(!it->hiddenPolys.empty() && !loaded.morphs.mainMeshes.empty())
  {
    auto& hidden = loaded.scene.meshes[size_t(loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh)].geo->hiddenPolys;
    if(visible)
      hidden[it->key] = it->hiddenPolys;
    else
      hidden.erase(it->key);
  }
  return true;
}

}  // namespace fv
