// VaM Timeline clips on a figure (docs/physics-plan.md step 16, figure/physics/vam/): load {as: "vam",
// path, prop?, person?, clip?, scene?} reads a VaM scene, its .json or a .var package's, on a worker,
// and plays one Person's clip (named, else its longest) on the figure named or selected as its animation (FigureAnimation), in
// Physics mode, which it turns on. The Anim tab plays, seeks and stops it as it does a DAZ
// animation; the animation topic's clip carries `vam` ({path, scene, person, people, clip, clips:
// [{name, duration}]}), with which the tab opens another clip of the same scene.

#include "fv_host_impl.hpp"

#include <figure/physics/vam/timeline.hpp>
#include <figure/physics/vam/var_package.hpp>

namespace {

namespace vam = fv::physics::vam;

// a scene file of ~18 MB is a large one; a package's scene comes out of its zip whole
constexpr size_t kMaxScene = size_t(512) << 20;

std::string readWhole(const std::filesystem::path& path)
{
  std::ifstream in(path, std::ios::binary);
  if(!in)
    throw std::runtime_error("can't open " + utf8Of(path.filename()));
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  if(size < 0 || size_t(size) > kMaxScene)
    throw std::runtime_error(utf8Of(path.filename()) + " is too large to be a VaM scene");
  std::string text(size_t(size), '\0');
  in.seekg(0);
  in.read(text.data(), std::streamsize(text.size()));
  return text;
}

// a clip to start on where none is named: the longest, the first of those (a scene's short clips
// are mostly poses it passes through)
size_t firstClip(const vam::Person& p)
{
  size_t best = 0;
  for(size_t i = 1; i < p.clips.size(); i++)
    if(p.clips[i].length > p.clips[best].length)
      best = i;
  return best;
}

}  // namespace

Json FvHost::Impl::loadVam(const Json& args)
{
  const std::string path      = args.at("path").get<std::string>();
  const std::string target    = args.contains("prop") && args["prop"].is_string() ? args["prop"].get<std::string>() : "";
  const std::string person    = args.value("person", ""), clipName = args.value("clip", ""), sceneName = args.value("scene", "");
  const uint64_t    jobId      = ++jobs;
  const uint64_t    generation = loadGeneration;
  const std::string label      = "Loading " + utf8Of(pathOf(path).filename());
  job(jobId, {{"label", label}, {"state", "running"}});
  loaders.spawn([this, path, target, person, clipName, sceneName, generation, jobId, label] {
    const auto failed = [this, jobId, label](const std::string& problem) {
      job(jobId, {{"label", label}, {"state", "failed"}, {"error", problem}});
      notice(problem, "error");
    };
    try
    {
      const std::filesystem::path file = pathOf(path);
      std::string                 ext  = file.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
      std::string text, entry;
      if(ext == ".var")
      {
        const std::vector<std::string> scenes = vam::varScenes(file);
        if(scenes.empty())
          throw std::runtime_error(utf8Of(file.filename()) + " is a VaM package with no scene in it.");
        entry = scenes.front();
        if(!sceneName.empty())
        {
          const auto it = std::find(scenes.begin(), scenes.end(), sceneName);
          if(it == scenes.end())
            throw std::runtime_error(utf8Of(file.filename()) + " has no scene " + sceneName + ".");
          entry = *it;
        }
        text = vam::zipRead(file, entry, kMaxScene);
      }
      else
        text = readWhole(file);
      auto scene = std::make_shared<vam::Scene>(vam::readScene(text));
      post([this, scene, path, entry, target, person, clipName, generation, jobId, label, failed] {
        if(generation != loadGeneration)
        {
          job(jobId, {{"label", label}, {"state", "done"}});
          return;
        }
        try
        {
          const auto prop = figureProp(target.empty() ? Json::object() : Json{{"prop", target}});
          if(!prop || !prop->figure->fig->pose)
            throw std::runtime_error("A VaM clip plays on a DAZ figure. Select one first, then open it.");
          Loaded& f = *prop->figure;
          if(!f.physics || !f.physics->valid())
            throw std::runtime_error(prop->name + " has no body bones for Physics mode, which a VaM clip plays through.");
          // the Person named, else the first with clips
          const vam::Person* who = nullptr;
          Json               people = Json::array();
          for(const vam::Person& p : scene->people)
          {
            if(p.clips.empty())
              continue;
            people.push_back(p.id);
            if(!who && (person.empty() || p.id == person))
              who = &p;
          }
          if(!who)
            throw std::runtime_error(person.empty() ? "This VaM scene has no Person with Timeline clips." : "This VaM scene has no Person " + person + " with clips.");
          size_t index = firstClip(*who);
          if(!clipName.empty())
          {
            index = who->clips.size();
            for(size_t i = 0; i < who->clips.size() && index == who->clips.size(); i++)
              if(who->clips[i].name == clipName)
                index = i;
            if(index == who->clips.size())
              throw std::runtime_error(who->id + " has no clip " + clipName + ".");
          }
          const vam::Clip& clip = who->clips[index];
          // it plays through Physics mode
          if(!f.physics->settings().on)
          {
            fv::physics::PhysicsSettings s = f.physics->settings();
            s.on                           = true;
            f.physics->set(s);
            publishPhysics(f);
          }
          auto player = std::make_shared<vam::Player>(*who, index);
          // what holds, and what our figure has nothing for
          player->at(0, *f.physics, *f.fig->pose, f.fig->loaded.scene);
          const size_t                   held      = player->targets().size();
          const std::vector<std::string> unmatched = player->unmatched();
          fv::pose::PoseClip             keys;
          keys.duration = clip.length;
          Json source   = {{"path", path}, {"person", who->id}, {"people", people}, {"clip", clip.name}};
          if(!entry.empty())
            source["scene"] = entry;
          holdAnimation(*prop, std::move(keys), clip.name, true, std::move(source), player);
          f.animation->name = clip.name;
          publishAnimation(true);
          char length[32];
          std::snprintf(length, sizeof(length), "%.1f s", clip.length);
          std::string missing;
          for(const std::string& u : unmatched)
            missing += (missing.empty() ? "" : ", ") + u;
          notice("Playing " + clip.name + " of " + who->id + " on " + prop->name + " (" + length + ") in Physics mode, " + std::to_string(held)
                     + " of its controllers holding the body. The Anim tab has its controls and the scene's other clips."
                     + (missing.empty() ? "" : " Nothing on " + prop->name + " for " + missing + "."),
                 "info");
          job(jobId, {{"label", label}, {"state", "done"}});
        }
        catch(const std::exception& e)
        {
          failed(problemOf(e));
        }
      });
    }
    catch(const std::exception& e)
    {
      failed(problemOf(e));
    }
  });
  return {{"job", jobId}};
}

void FvHost::Impl::driveVam(Loaded& f)
{
  FigureAnimation& a = *f.animation;
  if(!f.fig->pose || !f.physics)
    return;
  // out of Physics mode nothing holds; back in, it comes in again
  if(!f.physics->settings().on)
  {
    a.vamBegins = true;
    return;
  }
  // a clip that begins comes in from where the body is
  if(a.vamBegins)
  {
    a.vam->begin(*f.physics, *f.fig->pose, f.fig->loaded.scene, a.time);
    a.vamBegins = false;
  }
  f.physics->drive(a.vam->at(a.time, *f.physics, *f.fig->pose, f.fig->loaded.scene));
}
