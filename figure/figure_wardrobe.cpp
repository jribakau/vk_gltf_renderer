// What a Figure wears: wearables are loaded and bound to the body on a worker (prepareWear), merged
// into the scene on the owning thread (commitWear), and taken off again (removeItem). A port of
// src/lib/wardrobe/Wardrobe.ts and the item-fitting half of src/lib/loaders/dson/figure.ts.

#include <figure/figure.hpp>

#include <algorithm>
#include <map>
#include <regex>

#include <figure/dson/listing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/fit.hpp>
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

// An item morph whose deltas address vertices the item does not have belongs to another generation.
bool targetFits(const mesh::RenderTarget& t, int count)
{
  const auto& idx = t.cageIndices.empty() ? t.indices : t.cageIndices;
  return std::all_of(idx.begin(), idx.end(), [&](uint32_t i) { return int(i) < count; });
}

// Whether an item morph that has no figure morph of the same name is worth a slider: fit adjustments
// and styles are; copies of figure morphs that aren't installed are not.
bool isAdjustment(const std::string& id)
{
  static const std::regex plumbing(R"((^|_)(bs|cbs|ctrl|jcm|pjcm|facs)_)", std::regex::icase);
  static const std::regex adjusts(R"(loosen|expand|tighten|shrink|adjust|collision|smooth|length|style|open|close|fit(?!ness))", std::regex::icase);
  return !std::regex_search(id, plumbing) || std::regex_search(id, adjusts);
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

WornItem record(const dson::Item& it, const std::string& rel, int id)
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
  return w;
}

}  // namespace

struct Figure::PreparedWear
{
  dson::LoadResult loaded;
  // per item of `loaded`: how it was refitted when it was made for another generation
  std::vector<std::optional<wardrobe::FitDetail>> fits;
  // per entry of loaded.parts: the surface binding of a conforming part
  std::vector<std::shared_ptr<const wardrobe::SurfaceBinding>> bindings;
  std::string                                                  key;
};

void Figure::indexItems()
{
  items.clear();
  for(const dson::Item& it : loaded.items)
    items.push_back(record(it, "", m_nextItem++));
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
  auto out = std::make_shared<SceneGraph>(loaded.scene);
  for(Node& n : out->nodes)
    if(n.joint)
      n.joint = std::make_shared<JointData>(*n.joint);
  return out;
}

std::shared_ptr<Figure::PreparedWear> Figure::prepareWear(const std::string& rel, int subdivision, std::shared_ptr<const SceneGraph> base)
{
  if(loaded.mainRig < 0 || loaded.morphs.mainMeshes.empty())
    throw std::runtime_error("Load a figure first: clothing and hair go on a figure with a skeleton.");
  auto out = std::make_shared<PreparedWear>();
  out->key = "w" + std::to_string(++m_wears);
  dson::LoadOptions options;
  options.subdivision = subdivision;
  // the figure as it was when the wear started, or as it is when the caller owns it now
  const SceneGraph& scene = base ? *base : loaded.scene;
  options.attach      = dson::AttachTarget{&scene, &loaded.rigs[size_t(loaded.mainRig)], out->key};
  const std::string file = rel.substr(std::min(rel.find_first_not_of('/'), rel.size()));
  out->loaded            = dson::loadDson(*m_store, file, options);
  if(out->loaded.items.empty())
    throw std::runtime_error("Nothing in this file can be worn.");

  // an item made for another generation is refitted first: its cage moves onto this figure's skeleton
  // and shape, and the bindings below are made against the cage it ends with
  const int bodyMesh = loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh;
  {
    const std::string  morphDir = loaded.morphs.morphDir;
    wardrobe::FitFigure figure{scene, loaded.rigs[size_t(loaded.mainRig)], bodyMesh, morphDir.substr(0, morphDir.rfind("/Morphs"))};
    for(const dson::Item& it : out->loaded.items)
      out->fits.push_back(wardrobe::fitItem(*m_store, out->loaded, it, rel, figure));
  }
  // every conforming part is bound to the body's unmorphed cage, in parallel
  const auto& body = *scene.meshes[size_t(bodyMesh)].geo;
  out->bindings.resize(out->loaded.parts.size());
  std::vector<std::future<void>> jobs;
  for(size_t i = 0; i < out->loaded.parts.size(); i++)
  {
    const dson::ItemPart& part = out->loaded.parts[i];
    if(!part.conforms)
      continue;
    const mesh::BuiltMesh& item = *out->loaded.scene.meshes[size_t(part.mesh)].geo;
    jobs.push_back(jobs::pool().submit([&body, &item, &part, slot = &out->bindings[i]] {
      const std::span<const int32_t> pairs = part.graft ? std::span<const int32_t>(part.graft->pairs) : std::span<const int32_t>();
      *slot = std::make_shared<wardrobe::SurfaceBinding>(wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions, pairs));
    }));
  }
  // every job ends before any error is rethrown: they write into `out`
  for(auto& j : jobs)
    j.wait();
  for(auto& j : jobs)
    j.get();
  return out;
}

WearResult Figure::commitWear(std::shared_ptr<PreparedWear> wear, const std::string& rel)
{
  PreparedWear&     p   = *wear;
  dson::LoadResult& w   = p.loaded;
  SceneGraph&       dst = loaded.scene;
  WearResult        result;
  result.warnings = w.warnings;

  // the same garment again replaces the worn one: the newer copy carries the file's fit settings
  std::set<std::string> sources;
  for(const dson::Item& it : w.items)
    sources.insert(it.sources.begin(), it.sources.end());
  std::vector<int> old;
  for(const WornItem& i : items)
    if(std::any_of(i.sources.begin(), i.sources.end(), [&](const std::string& s) { return sources.count(s) > 0; }))
      old.push_back(i.id);
  for(int id : old)
  {
    removeItem(id);
    result.replaced.push_back(id);
  }

  // ---- merge the scene: nodes, meshes, materials and skeletons, with their indices moved on
  const int nodeBase = int(dst.nodes.size()), meshBase = int(dst.meshes.size()), materialBase = int(dst.materials.size()),
            rigBase = int(loaded.rigs.size()), itemBase = int(loaded.items.size());
  for(fv::Node n : w.scene.nodes)
  {
    if(n.parent >= 0)
      n.parent += nodeBase;
    for(int& c : n.children)
      c += nodeBase;
    if(n.mesh >= 0)
      n.mesh += meshBase;
    dst.nodes.push_back(std::move(n));
  }
  for(fv::Mesh m : w.scene.meshes)
  {
    for(int& mi : m.materials)
      mi += materialBase;
    for(int& b : m.bones)
      b += nodeBase;
    dst.meshes.push_back(std::move(m));
  }
  for(fv::Material& m : w.scene.materials)
    dst.materials.push_back(std::move(m));
  for(dson::Rig r : w.rigs)
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
  for(dson::Item it : w.items)
  {
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
    WornItem worn = record(it, rel, m_nextItem++);
    added.push_back(worn.id);
    items.push_back(std::move(worn));
  }
  // the wearable's morph meshes and the items they belong to, for later loads of item morph files
  std::vector<dson::MeshRef> itemRefs;
  for(dson::MeshRef ref : w.morphs.meshes)
  {
    ref.mesh += meshBase;
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

  // ---- skeletons follow the figure's bones; the match is how well it was made for this figure
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
          matched += m;
          total += t;
        }
    }
  result.fit = total ? double(matched) / double(total) : 1.0;

  // A heel's file poses the figure's feet, as DAZ Studio applies its scene animations to the figure
  // it goes on. Only a refitted item takes it: the refit moves the shoe's bones onto the figure's
  // joints, flattening a heel whose rest is its pointed shape, and the pose points it again.
  if(pose)
    for(size_t k = 0; k < added.size() && k < p.fits.size(); k++)
    {
      if(!p.fits[k])
        continue;
      WornItem* worn = nullptr;
      for(WornItem& i : items)
        if(i.id == added[k])
          worn = &i;
      if(!worn)
        continue;
      std::vector<std::string> names;
      for(const auto& [id, node] : loaded.rigs[size_t(loaded.mainRig)].bones)
        names.push_back(dst.nodes[size_t(node)].name);
      std::vector<pose::ChannelValue> set;
      for(const wardrobe::FootChannel& c : wardrobe::retargetFootPose(w.items[k].footPose, names))
        if(const auto before = pose->posed(c.bone, "rotation", c.axis))
        {
          set.push_back({c.bone, pose::ChannelKind::eRotation, c.axis, c.value});
          worn->footSet.push_back({c.bone, c.axis, c.value, *before});
        }
      if(set.empty())
        continue;
      pose->setMany(set);
      // what the rig kept after its limits, so undoing compares like with like
      for(WornItem::FootSet& f : worn->footSet)
        if(const auto now = pose->posed(f.bone, "rotation", f.axis))
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
    std::vector<const dson::SceneDial*> dials;
    for(const dson::SceneDial& d : w.morphs.dials)
      dials.push_back(&d);
    dson::MorphContext ctx = w.morphs;
    ctx.meshes             = itemRefs;
    // itemOf indexes the wearable's own items
    applyOtherDials(ctx, w.items, dials, {});
  }
  std::map<std::string, std::map<std::string, std::string>> ownMorphs;  // dir -> stem (lowercase) -> file
  const auto listOwn = [&](const std::string& defFile) -> const std::map<std::string, std::string>& {
    const size_t      slash = defFile.rfind('/');
    const std::string dir   = (slash == std::string::npos ? std::string() : defFile.substr(0, slash)) + "/Morphs";
    auto              it    = ownMorphs.find(dir);
    if(it == ownMorphs.end())
    {
      std::map<std::string, std::string> found;
      if(!defFile.empty())
        for(const std::string& f : dson::listFiles(m_store->library(), dir, ".dsf"))
          found[lowerCase(stemOf(f))] = f;
      it = ownMorphs.emplace(dir, std::move(found)).first;
    }
    return it->second;
  };
  const auto& bodyGeoPtr = dst.meshes[size_t(body)].geo;
  for(size_t i = 0; i < w.parts.size(); i++)
  {
    const dson::ItemPart& part = w.parts[i];
    const int             mesh = part.mesh + meshBase;
    WornItem*             worn = itemOfMesh(mesh);
    if(!worn)
      continue;
    const auto& own    = listOwn(part.defFile);
    const auto  itemGeo = dst.meshes[size_t(mesh)].geo;
    const dson::MeshRef ref = refOf(mesh);
    if(part.conforms && p.bindings[i])
    {
      morph::Follower f;
      f.mesh   = mesh;
      f.source = body;
      f.project = [binding = p.bindings[i], rigidity = part.rigidity, bodyGeo = bodyGeoPtr, itemGeo,
                   scratch = std::make_shared<std::vector<float>>()](const mesh::RenderTarget& t) -> dson::TargetPtr {
        return wardrobe::projectTarget(*binding, rigidity ? &*rigidity : nullptr, *bodyGeo, *itemGeo, t, *scratch);
      };
      // DAZ's morph tools tag each file's modifier id with a hex suffix ("Mirel Head-0xa98947a" on the
      // body, "-0xa989482" on the eyes); the copies share the name and the file stem
      f.own = [store = m_store, own, ref, itemGeo](const std::string& channel) -> std::shared_future<std::vector<dson::TargetPtr>> {
        static const std::regex hex(R"(-0x[0-9a-f]+$)", std::regex::icase);
        auto                    it = own.find(lowerCase(channel));
        if(it == own.end())
          it = own.find(lowerCase(std::regex_replace(channel, hex, "")));
        if(it == own.end())
          return {};
        return jobs::pool()
            .submit([store, file = it->second, ref, itemGeo]() -> std::vector<dson::TargetPtr> {
              std::vector<dson::TargetPtr> out;
              try
              {
                for(const dson::MorphPart& part : dson::loadMorphFile(*store, file, {ref}))
                  if(part.mesh == ref.mesh && part.target && targetFits(*part.target, itemGeo->dsonVertexCount))
                    out.push_back(part.target);
              }
              catch(const std::exception&)
              {
              }
              return out;
            })
            .share();
      };
      morphs->addFollower(std::move(f));
      if(part.graft && !part.graft->hidden.empty())
      {
        auto& hidden = dst.meshes[size_t(body)].geo->hiddenPolys[worn->key];
        hidden.insert(hidden.end(), part.graft->hidden.begin(), part.graft->hidden.end());
        worn->hiddenPolys.insert(worn->hiddenPolys.end(), part.graft->hidden.begin(), part.graft->hidden.end());
      }
    }
    // morphs that aren't copies of a figure morph: fit adjustments ("Loosen Neck") and styles
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
  // dials the wearable brought, besides its adjustments: they leave with it
  if(!added.empty())
    if(WornItem* first = &*std::find_if(items.begin(), items.end(), [&](const WornItem& i) { return i.id == added[0]; }))
      for(const morph::Channel& c : morphs->channels())
        if(!before.count(c.name) && std::find(first->channels.begin(), first->channels.end(), c.name) == first->channels.end())
          first->channels.push_back(c.name);
  morphs->flush();
  morphs->sort();
  result.items = added;
  return result;
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
    std::vector<pose::ChannelValue> back;
    for(const WornItem::FootSet& f : gone.footSet)
      if(const auto now = pose->posed(f.bone, "rotation", f.axis); now && *now == f.value)
        back.push_back({f.bone, pose::ChannelKind::eRotation, f.axis, f.before});
    if(!back.empty())
      pose->setMany(back);
  }
  morphs->removeChannels(gone.channels);
  morphs->removeMeshes(std::set<int>(gone.meshes.begin(), gone.meshes.end()));
  if(!loaded.morphs.mainMeshes.empty())
    scene.meshes[size_t(loaded.morphs.meshes[loaded.morphs.mainMeshes[0]].mesh)].geo->hiddenPolys.erase(gone.key);
  return true;
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
