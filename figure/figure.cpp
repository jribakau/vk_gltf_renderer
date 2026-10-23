#include <figure/figure.hpp>

#include <algorithm>
#include <map>
#include <regex>
#include <set>
#include <unordered_map>

#include <figure/dson/listing.hpp>
#include <figure/jobs/pool.hpp>

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
  std::unique_ptr<Figure> f(new Figure());
  f->m_store = store;
  f->loaded  = dson::loadDson(*store, rel, options);
  f->setup(progress);
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
    for(const std::string& f : dson::listFiles(store.library(), ctx.morphDir, ".dsf"))
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
    prefetch(mainDials);
    std::vector<std::pair<std::string, double>> values;
    for(const dson::SceneDial* d : mainDials)
    {
      if(!morphs->has(d->id))
      {
        morph::ChannelOptions opts;
        opts.group = dson::groupFromPath(d->file);
        morphs->addLazy(d->id, driver->loader(d->file, d->id), opts);
      }
      values.emplace_back(d->id, d->value);
    }
    // each dial's file loads, its formulas name more, and so on until nothing new is needed
    morphs->setMany(values);
    morphs->settled();
  }

  // other figures' and props' dials: their controllers are followed through linear formulas, each
  // chain summed per owner (six copies of a box each open their own lid)
  if(!otherDials.empty())
  {
    say("Resolving character morphs");
    struct Total
    {
      std::string owner, file, id;
      double      value = 0;
    };
    std::vector<Total>            totals;
    std::map<std::string, size_t> totalAt;
    for(const dson::SceneDial* d : otherDials)
    {
      std::vector<dson::DrivenMorph> driven;
      try
      {
        driven = dson::drivenMorphs(store, d->file, d->id, d->value);
      }
      catch(const dson::MissingFileError&)
      {
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
        const dson::Item* item = itemIt == ctx.itemOf.end() ? nullptr : &loaded.items[size_t(itemIt->second)];
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
  morphs->flush();
  morphs->sort();

  // posing for the main figure; other figures (conforming clothing) follow its bones by name
  if(loaded.mainRig >= 0)
  {
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
  driver->update();
}

void Figure::settle()
{
  for(;;)
  {
    morphs->settled();
    if(m_correctiveJobs.empty())
      return;
    for(auto& job : m_correctiveJobs)
      job.wait();
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

Figure::Changes Figure::pump()
{
  morphs->pump();
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

}  // namespace fv
