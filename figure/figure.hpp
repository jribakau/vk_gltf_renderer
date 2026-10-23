#pragma once

// A loaded DAZ scene as the renderer works with it: the scene graph the loader built, its morphs
// and the formulas that drive them, and the pose rig. Ports the morph and posing half of
// src/lib/loaders/dson/figure.ts, which wired these together as the last stages of a load.
//
// Threads: one thread uses a Figure at a time (the load worker while it loads, then the renderer's
// application thread). Morph files are read on the thread pool; pump() takes what arrived.

#include <functional>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <figure/dson/figure_loader.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/figure_driver.hpp>
#include <figure/pose/pose_rig.hpp>

namespace fv {

class Figure
{
public:
  using Progress = std::function<void(const std::string&)>;

  // Loads `rel` and sets its morphs, formulas and pose up. Blocks until the scene's own dials have
  // been applied, with every morph file they need read.
  static std::unique_ptr<Figure> load(std::shared_ptr<dson::Store> store, const std::string& rel, const dson::LoadOptions& options,
                                      const Progress& progress = {});

  dson::LoadResult                    loaded;
  std::unique_ptr<morph::MorphSet>    morphs;
  std::unique_ptr<pose::FigureDriver> driver;
  // null without a skeleton
  std::unique_ptr<pose::PoseRig> pose;

  // What changed since last asked: bone nodes to re-pose, meshes whose vertices moved, and meshes
  // whose skins were bound again at new joint centres.
  struct Changes
  {
    std::set<int>                  nodes;
    std::vector<morph::MeshChange> meshes;
    std::vector<int>               rebound;
    bool                           empty() const { return nodes.empty() && meshes.empty() && rebound.empty(); }
  };
  Changes drain();

  // Takes lazy loads that arrived and applies them; returns what changed. Call every frame.
  Changes pump();

  // Waits for every load in flight, joint correctives included.
  void settle();

  // What the figure is waiting on, for a progress line; empty when idle.
  std::string busy() const;

  // Loads that failed, worth telling the user about.
  std::vector<std::string> takeWarnings();

  // Starts loading the joint correctives, once (they join on the first pose change).
  void loadCorrectivesOnce();

private:
  Figure() = default;
  void setup(const Progress& progress);
  void finishCorrectives();

  struct CorrectiveFile
  {
    std::string                      file;
    std::shared_ptr<const dson::Doc> doc;
    std::vector<dson::MorphDef>      defs;
  };

  std::shared_ptr<dson::Store>                  m_store;
  std::shared_ptr<pose::FigureSource>           m_source;
  std::vector<std::string>                      m_correctiveFiles;
  std::vector<std::future<CorrectiveFile>>      m_correctiveJobs;
  bool                                          m_correctivesStarted = false;
};

}  // namespace fv
