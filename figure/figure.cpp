#include <figure/figure.hpp>

#include <algorithm>
#include <map>
#include <regex>
#include <set>
#include <unordered_map>

#include <figure/dson/listing.hpp>
#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/morph/shape_sliders.hpp>
#include <figure/pose/retarget.hpp>

namespace fv {

using dson::Json;

namespace {

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

std::string stemOf(const std::string& file)
{
  const size_t slash = file.rfind('/');
  std::string  name  = slash == std::string::npos ? file : file.substr(slash + 1);
  if(name.size() >= 4 && lower(name.substr(name.size() - 4)) == ".dsf")
    name.resize(name.size() - 4);
  return name;
}

// Where the figure's modifier files come from: the store, and the Morphs folder's index.
class StoreSource : public pose::FigureSource
{
public:
  StoreSource(std::shared_ptr<dson::Store> store, std::vector<dson::MeshRef> meshes, std::vector<size_t> main)
      : m_store(std::move(store))
      , m_meshes(std::move(meshes))
      , m_main(std::move(main))
  {
  }
  std::shared_ptr<const dson::Doc> doc(const std::string& file) override { return m_store->load(file); }
  std::vector<dson::MorphDef> targets(const dson::Doc& doc, const std::string& file) override
  {
    return dson::morphsFromDoc(doc, file, m_meshes, &m_main);
  }
  std::optional<std::string> find(const std::string& id) override
  {
    auto it = index.find(lower(id));
    return it == index.end() ? std::nullopt : std::optional<std::string>(it->second);
  }
  // modifier id (lowercase) -> file
  std::unordered_map<std::string, std::string> index;

private:
  std::shared_ptr<dson::Store> m_store;
  std::vector<dson::MeshRef>   m_meshes;
  std::vector<size_t>          m_main;
};

}  // namespace

std::unique_ptr<Figure> Figure::load(std::shared_ptr<dson::Store> store, const std::string& rel, const dson::LoadOptions& options, const Progress& progress)
{
  timing::Scope           t("Figure::load " + rel);
  std::unique_ptr<Figure> f(new Figure());
  f->m_store = store;
  f->m_cageMorphs = options.cageMorphs;
  {
    timing::Scope tl("loadDson");
    f->loaded = dson::loadDson(*store, rel, options);
    f->placeLoadedSurfaces(rel);
  }
  f->setup(progress);
  f->indexItems();
  if(auto doc = store->load(rel.substr(std::min(rel.find_first_not_of('/'), rel.size()))))
    f->addons = dson::postLoadAddons(doc->json);
  return f;
}

void Figure::setup(const Progress& progress)
{
  auto say = [&](const std::string& s) {
    if(progress)
      progress(s);
  };
  dson::MorphContext& ctx = loaded.morphs;
  dson::Store&        store = *m_store;
  morphs = std::make_unique<morph::MorphSet>([this](int i) { return loaded.scene.meshes[size_t(i)].geo.get(); });
  // the character's shape goes onto the cages alone, until the host says which meshes it refines
  if(m_cageMorphs)
    morphs->cageOnly = [this](int i) {
      const mesh::BuiltMesh* g = loaded.scene.meshes[size_t(i)].geo.get();
      return g && mesh::drawnFromCage(*g);
    };
  auto source = std::make_shared<StoreSource>(m_store, ctx.meshes, ctx.mainMeshes);
  m_source    = source;
  // The main figure's dials run through its formulas: a character preset's controller drives shape
  // morphs, an expression drives face bones, a body shape moves the joints.
  if(!ctx.mainMeshes.empty())
  {
    driver         = std::make_unique<pose::FigureDriver>(*morphs, source);
    morphs->driver = driver.get();
  }

  // every morph installed for the main figure, parsed on first use; listed first, so the morphs
  // the scene's formulas name can be found
  if(!ctx.morphDir.empty())
  {
    say("Indexing morphs");
    timing::Scope t("index morphs");
    const std::vector<std::string> morphFiles = dson::listFiles(store.library(), ctx.morphDir, ".dsf");
    // what each file says of its dial, on the pool: a load doesn't wait for thousands of little reads
    m_morphInfoJob = jobs::pool().submit([st = m_store, files = morphFiles]() {
      timing::Scope                                 t("morph details", 1);
      std::vector<std::optional<dson::MorphInfo>>   read(files.size());
      jobs::parallelFor(files.size(), 32, [&](size_t begin, size_t end) {
        for(size_t i = begin; i < end; i++)
          if(const auto path = st->library().find(files[i]))
            read[i] = dson::morphInfo(*path);
      });
      MorphNotes notes;
      for(size_t i = 0; i < files.size(); i++)
        if(read[i])
        {
          notes.facts.push_back({files[i], stemOf(files[i]), *read[i]});
          read[i]->formulas.reset();
          // a scene's dial goes by the id in the file, the listing's by the file's name
          if(!read[i]->alias && !read[i]->id.empty())
            notes.info.emplace(read[i]->id, *read[i]);
          notes.info[stemOf(files[i])] = *read[i];
          notes.files.add(files[i], stemOf(files[i]), *read[i]);
        }
      return notes;
    });
    for(const std::string& f : morphFiles)
    {
      const std::string stem = stemOf(f);
      source->index[lower(stem)] = f;
      static const std::regex correctives(R"(/Base Correctives/|/BaseJointCorrectives\.dsf$)", std::regex::icase);
      if(std::regex_search(f, correctives))
        m_correctiveFiles.push_back(f);
      if(dson::hiddenMorph(stem))
        continue;
      morph::ChannelOptions opts;
      opts.group = dson::groupFromPath(f);
      if(driver)
        morphs->addLazy(stem, driver->loader(f, stem), opts);
      else
        morphs->addLazy(
            stem,
            [st = m_store, meshes = ctx.meshes, f]() -> morph::Finish {
              auto parts = dson::loadMorphFile(*st, f, meshes);
              return [parts = std::move(parts)]() mutable {
                morph::LazyResult r;
                r.parts = std::move(parts);
                return r;
              };
            },
            opts);
    }
  }

  // the dials the scene sets: the main figure's run through its formulas
  std::vector<const dson::SceneDial*> mainDials, otherDials;
  for(const dson::SceneDial& d : ctx.dials)
    (driver && ctx.mainOwners.count(d.owner) ? mainDials : otherDials).push_back(&d);
  auto prefetch = [&](const std::vector<const dson::SceneDial*>& dials) {
    std::set<std::string> seen;
    for(const dson::SceneDial* d : dials)
      if(seen.insert(d->file).second)
        jobs::pool().submit([st = m_store, file = d->file] {
          try
          {
            st->load(file);
          }
          catch(const std::exception&)
          {
          }
        });
  };
  if(!mainDials.empty())
  {
    say("Resolving character morphs");
    timing::Scope t("character dials");
    prefetch(mainDials);
    std::vector<std::pair<std::string, double>> values;
    // what the file itself sets and isn't installed: the figure comes out without it
    std::vector<std::string> notInstalled;
    for(const dson::SceneDial* d : mainDials)
    {
      if(!m_store->library().find(d->file) && std::find(notInstalled.begin(), notInstalled.end(), d->file) == notInstalled.end())
        notInstalled.push_back(d->file);
      if(!morphs->has(d->id))
      {
        morph::ChannelOptions opts;
        opts.group = dson::groupFromPath(d->file);
        morphs->addLazy(d->id, driver->loader(d->file, d->id), opts);
      }
      values.emplace_back(d->id, d->value);
    }
    if(!notInstalled.empty())
      loaded.warnings.push_back(dson::missingMessage(loaded.name, notInstalled));
    // each dial's file loads, its formulas name more, and so on until nothing new is needed
    morphs->setMany(values);
    morphs->settled();
  }

  // other figures' and props' dials: their controllers are followed through linear formulas
  {
    timing::Scope t("other dials");
    applyOtherDials(ctx, loaded.items, otherDials, progress);
  }
  morphs->flush();
  morphs->sort();
  morphs->markLoaded();

  // posing for the main figure; other figures (conforming clothing) follow its bones by name
  if(loaded.mainRig >= 0)
  {
    timing::Scope                 t("pose rig");
    std::vector<const dson::Rig*> followers;
    for(size_t i = 0; i < loaded.rigs.size(); i++)
      if(int(i) != loaded.mainRig)
        followers.push_back(&loaded.rigs[i]);
    pose = std::make_unique<pose::PoseRig>(loaded.scene, loaded.rigs[size_t(loaded.mainRig)], followers);
    if(driver)
    {
      driver->rig       = pose.get();
      pose->afterPose   = [this] {
        driver->update();
        loadCorrectivesOnce();
      };
      driver->onDriven = [this] { loadCorrectivesOnce(); };
      std::vector<const dson::MeshRef*> bodies;
      for(const size_t i : ctx.mainMeshes)
        bodies.push_back(&ctx.meshes[i]);
      morph::addShapeSliders(*morphs, *driver, loaded.scene, bodies);
      // a character's scene file stores its joints where its shape put them; a bare figure's are the base's
      if(!ctx.mainNode.empty() && ctx.previewCenters)
        driver->captureBaseline();
      driver->update();
      morphs->settled();
    }
    // joints the shape moved: the skins bind at the new centres, as the load leaves them
    if(pose->takeRestMoved())
      pose->rebind();
    pose->takeMoved();
  }
}

void Figure::applyOtherDials(const dson::MorphContext& ctx, const std::vector<dson::Item>& items, const std::vector<const dson::SceneDial*>& dials, const Progress& progress)
{
  if(dials.empty())
    return;
  auto          say   = [&](const std::string& s) {
    if(progress)
      progress(s);
  };
  dson::Store& store = *m_store;
    say("Resolving character morphs");
    struct Total
    {
      std::string owner, file, id;
      double      value = 0;
    };
    std::vector<Total>            totals;
    std::map<std::string, size_t> totalAt;
    for(const dson::SceneDial* d : dials)
    {
      std::vector<dson::DrivenMorph> driven;
      try
      {
        driven = dson::drivenMorphs(store, d->file, d->id, d->value);
      }
      catch(const std::exception& e)
      {
        loaded.warnings.push_back("Morph " + d->id + ": " + e.what());
      }
      for(const dson::DrivenMorph& m : driven)
      {
        const std::string key = lower(d->owner + "|" + m.file + "#" + m.id);
        if(auto it = totalAt.find(key); it != totalAt.end())
          totals[it->second].value += m.weight;
        else
        {
          totalAt[key] = totals.size();
          totals.push_back({d->owner, m.file, m.id, m.weight});
        }
      }
    }
    struct FileWants
    {
      std::string                                owner, file;
      std::vector<std::pair<std::string, double>> wanted;
    };
    std::vector<FileWants>        files;
    std::map<std::string, size_t> fileAt;
    for(const Total& t : totals)
    {
      if(t.value == 0)
        continue;
      const std::string key = t.owner + "|" + t.file;
      auto              it  = fileAt.find(key);
      if(it == fileAt.end())
      {
        fileAt[key] = files.size();
        files.push_back({t.owner, t.file, {}});
        it = fileAt.find(key);
      }
      files[it->second].wanted.emplace_back(t.id, t.value);
    }
    for(const FileWants& f : files)
      jobs::pool().submit([st = m_store, file = f.file] {
        try
        {
          st->load(file);
        }
        catch(const std::exception&)
        {
        }
      });
    // one slider per morph, unless owners set it to different values: those get one each
    std::map<std::string, double> sliderValues;
    int                           mi = 0;
    for(const FileWants& f : files)
    {
      say("Applying character morphs " + std::to_string(++mi) + "/" + std::to_string(files.size()));
      try
      {
        auto                mdoc = store.load(f.file);
        std::vector<size_t> own;
        for(size_t i = 0; i < ctx.meshes.size(); i++)
        {
          const auto& owners = ctx.meshes[i].owners;
          if(std::find(owners.begin(), owners.end(), f.owner) != owners.end())
            own.push_back(i);
        }
        const auto itemIt = ctx.itemOf.find(f.owner);
        const dson::Item* item = itemIt == ctx.itemOf.end() ? nullptr : &items[size_t(itemIt->second)];
        for(const dson::MorphDef& m : dson::morphsFromDoc(*mdoc, f.file, ctx.meshes, own.empty() ? nullptr : &own))
        {
          auto w = std::find_if(f.wanted.begin(), f.wanted.end(), [&](const auto& x) { return x.first == m.id; });
          if(w == f.wanted.end())
            continue;
          auto              prev = sliderValues.find(m.id);
          const std::string name = prev == sliderValues.end() || prev->second == w->second ? m.id : m.id + " (" + f.owner + ")";
          sliderValues[name]     = w->second;
          for(const dson::MorphPart& part : m.parts)
          {
            morph::ChannelOptions opts;
            opts.value = w->second;
            opts.min   = part.min;
            opts.max   = part.max;
            if(name == m.id || item)
            {
              if(part.hasLabel)
                opts.label = part.label;
            }
            else
              opts.label = (part.hasLabel ? part.label : m.id) + " (" + f.owner + ")";
            opts.group  = item ? item->label : dson::groupFromPath(f.file);
            // an item's copy of a figure corrective the figure doesn't have: set by the file, not a dial
            opts.hidden = item && name != m.id && dson::hiddenMorph(m.id);
            morphs->addTarget(name, part.mesh, part.target, opts);
          }
        }
      }
      catch(const dson::MissingFileError&)
      {
      }
      catch(const std::exception& e)
      {
        loaded.warnings.push_back("Morph file " + f.file.substr(f.file.rfind('/') + 1) + ": " + e.what());
      }
    }
  }

void Figure::loadCorrectivesOnce()
{
  if(m_correctivesStarted || m_correctiveFiles.empty() || !driver)
    return;
  m_correctivesStarted = true;
  for(const std::string& file : m_correctiveFiles)
    m_correctiveJobs.push_back(jobs::pool().submit([source = m_source, file]() -> CorrectiveFile {
      CorrectiveFile out;
      out.file = file;
      try
      {
        out.doc  = source->doc(file);
        out.defs = source->targets(*out.doc, file);
      }
      catch(const std::exception&)
      {
        out.doc.reset();
      }
      return out;
    }));
}

std::string Figure::busy() const
{
  return m_correctiveJobs.empty() ? std::string() : "Loading joint correctives";
}

void Figure::finishCorrectives()
{
  std::vector<CorrectiveFile> files;
  for(auto& job : m_correctiveJobs)
    files.push_back(job.get());
  m_correctiveJobs.clear();
  std::optional<timing::Scope> step;
  step.emplace("correctives: targets", 1);
  for(CorrectiveFile& f : files)
  {
    if(!f.doc)
      continue;
    driver->graph.addDoc(f.doc->json, f.file);
    for(dson::MorphDef& m : f.defs)
    {
      if(morphs->has(m.id))
        continue;
      for(const dson::MorphPart& part : m.parts)
      {
        morph::ChannelOptions opts;
        if(part.hasLabel)
          opts.label = part.label;
        opts.group  = "Joint correctives";
        opts.hidden = true;
        morphs->addTarget(m.id, part.mesh, part.target, opts);
      }
    }
  }
  step.emplace("correctives: driver", 1);
  driver->update();
}

void Figure::settle()
{
  // a scene's dials want their controllers before it is shown
  if(m_morphInfoJob.valid())
    m_morphInfoJob.wait();
  for(;;)
  {
    {
      timing::Scope t("settle: dials", 1);
      do
        morphs->settled(true);
      while(noteMorphs());
    }
    if(m_correctiveJobs.empty())
      return;
    {
      timing::Scope t("settle: corrective files", 1);
      for(auto& job : m_correctiveJobs)
        job.wait();
    }
    timing::Scope t("settle: correctives added", 1);
    finishCorrectives();
  }
}

Figure::Changes Figure::drain()
{
  Changes c;
  if(pose)
  {
    c.nodes = pose->takeMoved();
    if(pose->takeRestMoved())
      c.rebound = pose->rebind();
  }
  c.meshes = morphs->takeChanges();
  return c;
}

bool Figure::noteMorphs()
{
  bool arrived = false;
  if(m_morphInfoJob.valid() && m_morphInfoJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    m_morphNotes  = m_morphInfoJob.get();
    m_morphsNoted = -1;
    arrived       = true;
    m_morphReachJob = jobs::pool().submit([st = m_store, facts = std::move(m_morphNotes.facts)]() mutable {
      timing::Scope t("morph reach", 1);
      // a few threads of the pool: a dial's file, which someone is waiting for, queues behind them
      jobs::parallelFor(
          facts.size(), 8,
          [&](size_t begin, size_t end) {
            for(size_t i = begin; i < end; i++)
              if(facts[i].info.hasMorph)
                if(const auto path = st->library().find(facts[i].file))
                  facts[i].size = dson::morphSize(*path);
          },
          3);
      return pose::auditReach(facts);
    });
  }
  if(m_morphReachJob.valid() && m_morphReachJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    m_morphReach  = m_morphReachJob.get();
    m_morphsNoted = -1;
  }
  bool again = false;
  if(!m_morphNotes.info.empty() && morphs->listVersion() != m_morphsNoted)
  {
    for(const morph::Channel& ch : morphs->channels())
    {
      if(const auto reach = m_morphReach.find(ch.name); reach != m_morphReach.end())
        morphs->setReach(ch.name, reach->second);
      const auto it = m_morphNotes.info.find(ch.name);
      if(it == m_morphNotes.info.end())
        continue;
      dson::MorphInfo info = it->second;
      // the listing's dial for a file whose own id the scene set: one slider, the one that holds the value
      if(!info.alias && info.id != ch.name && morphs->has(info.id))
        info.helper = true;
      if(ch.path != info.path || ch.kind != info.kind || ch.helper != info.helper || ch.rest != info.rest)
        again = morphs->describe(ch.name, info) || again;
    }
    m_morphsNoted = morphs->listVersion();
    // one that came or heard of its rest may stand away from it, which no change of value says
    m_readersSeen = -1;
  }
  // once the channels have heard of their rests, as formulas will read them now
  if(arrived && driver)
    driver->noteFiles(std::move(m_morphNotes.files));
  // a dial that has moved brings the files that read it, which nothing else would load
  if(driver && morphs->version() != m_readersSeen)
  {
    m_readersSeen = morphs->version();
    again         = driver->loadReaders() || again;
  }
  return again;
}

Figure::Changes Figure::pump()
{
  morphs->pump();
  noteMorphs();
  if(!m_correctiveJobs.empty()
     && std::all_of(m_correctiveJobs.begin(), m_correctiveJobs.end(),
                    [](auto& j) { return j.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }))
    finishCorrectives();
  return drain();
}

std::vector<std::string> Figure::takeWarnings()
{
  std::vector<std::string> out = std::exchange(loaded.warnings, {});
  for(std::string& e : morphs->takeErrors())
    out.push_back(std::move(e));
  return out;
}

namespace {

// The last key's value, or none when the animation has no key.
const dson::Json* lastKeyJson(const dson::Json& animation)
{
  const dson::Json* keys = dson::get(&animation, "keys");
  if(!keys || !keys->is_array() || keys->empty())
    return nullptr;
  const dson::Json& key = keys->back();
  if(!key.is_array() || key.size() < 2)
    return nullptr;
  return &key[1];
}

}  // namespace

PosePresetResult applyPosePreset(pose::PoseRig* pose, morph::MorphSet* morphs, const dson::Json& doc, const pose::ChannelOf& channelOf)
{
  PosePresetResult result;
  if(!pose)
    return result;
  const dson::Json* animations = dson::get(dson::get(&doc, "scene"), "animations");
  if(!animations || !animations->is_array())
    return result;

  using Kind = pose::PresetTarget::Kind;
  const pose::PresetResolver                  resolver(pose, morphs, channelOf);
  std::vector<pose::ChannelValue>             values;
  std::vector<std::pair<std::string, double>> morphValues;
  for(const dson::Json& a : *animations)
  {
    const dson::Json* raw = lastKeyJson(a);
    if(!raw || !(raw->is_number() || raw->is_boolean()))
      continue;
    const pose::PresetTarget t = resolver.resolve(dson::getString(&a, "url"));
    if(t.kind == Kind::eUnreadable)
      result.skipped.push_back(t.name);
    else if(t.kind == Kind::eVisibility)
      result.visibility.emplace_back(t.name, raw->is_boolean() ? raw->get<bool>() : raw->get<double>() != 0);
    else if(raw->is_number())
    {
      const double value = raw->get<double>();
      if(t.kind == Kind::eBone)
        values.push_back({t.bone, t.channel, t.axis, value});
      else if(t.kind == Kind::eDial)
        morphValues.emplace_back(t.name, value);
      else if(t.kind == Kind::eMissing && value != 0)
        result.skipped.push_back(t.name);
    }
  }
  pose->setMany(values);
  if(morphs && !morphValues.empty())
    morphs->setMany(morphValues);
  result.applied = int(values.size() + morphValues.size());
  return result;
}

WornItem* Figure::itemNamed(const std::string& node)
{
  for(WornItem& i : items)
    if(i.nodeNames.count(node) || i.label == node)
      return &i;
  return nullptr;
}

pose::ChannelOf Figure::wornDials()
{
  return [this](const std::string& node, const std::string& id) -> std::string {
    // a worn item's dial: "Loosen Neck" is stored as "<id>@<item key>"; on a named node, that item's
    if(!node.empty())
    {
      if(node == loaded.morphs.mainNode)
        return morphs->channel(id) ? id : std::string();
      const WornItem* i = itemNamed(node);
      return i && morphs->channel(id + "@" + i->key) ? id + "@" + i->key : std::string();
    }
    for(const WornItem& i : items)
      if(morphs->channel(id + "@" + i.key))
        return id + "@" + i.key;
    return {};
  };
}

pose::PoseClip Figure::loadPoseClip(const dson::Json& doc, const std::string& rel, std::string* retargetedFrom)
{
  pose::PoseClip own = pose::loadPoseClip(pose.get(), morphs.get(), doc, wornDials());
  if(!pose || pose::namesRigBones(doc, *pose))
    return own;
  const std::string source = pose::sourceFigureFile(doc, rel);
  if(source.empty())
    return own;
  // the skeleton the file was made for, as its base figure has it
  std::shared_ptr<const dson::Doc> file;
  try
  {
    file = m_store->load(source);
  }
  catch(const std::exception&)
  {
    return own;
  }
  const auto library = file ? file->json.find("node_library") : dson::Json::const_iterator{};
  if(!file || library == file->json.end())
    return own;
  SceneGraph                     scene;
  const std::optional<dson::Rig> rig = dson::buildRig(scene, *library, nullptr);
  if(!rig)
    return own;
  const pose::PoseRig from(scene, *rig, {});
  pose::PoseClip      clip = pose::retargetClip(doc, from, *pose);
  if(clip.channels.empty())
    return own;
  // the file's dials this figure has too go along
  for(pose::PoseClip::Channel& c : own.channels)
    if(c.target.kind == pose::PresetTarget::Kind::eDial)
      clip.channels.push_back(std::move(c));
  // skipped: what it names that is neither this figure's nor a bone of the skeleton it was made for
  // ("lThighBend rotation/x" is retargeted, a dial this figure lacks isn't)
  for(std::string& s : own.skipped)
  {
    const size_t      space = s.rfind(' ');
    const std::string bone  = space == std::string::npos ? std::string() : s.substr(0, space);
    if(bone.empty() || (bone != "figure" && !from.get(bone)))
      clip.skipped.push_back(std::move(s));
  }
  clip.duration = std::max(clip.duration, own.duration);
  if(retargetedFrom)
    *retargetedFrom = source;
  return clip;
}

PosePresetResult Figure::applyPosePreset(const dson::Json& doc)
{
  PosePresetResult result = fv::applyPosePreset(pose.get(), morphs.get(), doc, wornDials());
  // nodes shown or hidden: the worn items they are (a geograft's "Apply" and "Remove" presets)
  for(const auto& [node, visible] : result.visibility)
  {
    WornItem* i = itemNamed(node);
    if(!i)
      continue;
    if(i->visible != visible && setItemVisible(i->id, visible))
      result.itemsShown.push_back(i->id);
    result.applied++;
  }
  return result;
}

}  // namespace fv
