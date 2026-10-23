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
#include <atomic>
#include <string>
#include <vector>

#include <figure/dson/addons.hpp>
#include <figure/dson/figure_loader.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/figure_driver.hpp>
#include <figure/pose/pose_rig.hpp>

namespace fv {

struct PosePresetResult
{
  int                      applied = 0;
  // Addresses this figure can't apply: dials it doesn't have, other figures' bones.
  std::vector<std::string> skipped;
};

// Applies a DAZ pose preset (.duf `scene.animations`) to a rig and its morphs: the last key of each
// animation, `name://@selection` urls and hierarchical `#bone?rotation/x/value` urls. Figure::applyPosePreset
// is this, on the figure's own pose and morphs.
PosePresetResult applyPosePreset(pose::PoseRig* pose, morph::MorphSet* morphs, const dson::Json& doc);

// What the figure wears: one wearable's nodes, which go on and come off together. Items that came
// with the scene file are listed too, with no library path.
struct WornItem
{
  int         id = 0;
  std::string rel;  // library path it was worn from; empty for the scene file's own
  std::string key, label;
  std::string kind;  // Clothing, Hair, Anatomy, Prop
  bool        visible = true;
  // the nodes it hangs from the figure by (scene node, the bone it is mounted on or ""), and what is under it
  std::vector<std::pair<int, std::string>> objects;
  std::vector<int>                         rigs;    // into LoadResult::rigs
  std::vector<int>                         meshes;  // into SceneGraph::meshes
  std::set<std::string>                    nodeNames, sources;
  // the dials it brought: fit adjustments such as "Loosen Neck"
  std::vector<std::string> channels;
  // body polygons a geograft hides while it is worn and shown
  std::vector<int32_t> hiddenPolys;
  // foot channels it set on the figure (a heel poses the feet), and the values they had before
  struct FootSet
  {
    std::string bone, axis;
    double      value = 0, before = 0;
  };
  std::vector<FootSet> footSet;
};

struct WearResult
{
  std::vector<int> items;     // ids of what went on
  std::vector<int> replaced;  // ids taken off because these are the same garment
  // share of the items' bones that found a figure bone: low means it was made for another figure
  double                   fit = 1;
  std::vector<std::string> missingBones, warnings;
  // how items made for another generation were refitted ("Genesis 8 outfit on Genesis 9, ...")
  std::vector<std::string> notes;
};

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

  // the add-ons the scene file asks to be worn after the load (Genesis 9 eyes, mouth, lashes)
  std::vector<dson::PostLoadAddon> addons;

  // ---- wardrobe (figure_wardrobe.cpp)
  std::vector<WornItem> items;

  // A wearable loaded and bound to the body. Reads the figure (its skeleton, its body's cage) and
  // changes nothing, so it can run on a worker while the figure is in use; commitWear puts it on.
  struct PreparedWear;
  // The wearable's own scene, for resolving its textures before it is merged.
  static const SceneGraph& preparedScene(const PreparedWear& wear);
  static SceneGraph&       preparedScene(PreparedWear& wear);
  std::shared_ptr<PreparedWear> prepareWear(const std::string& rel, int subdivision);
  // Merges a prepared wearable into the scene: nodes, skeleton, materials, morph followers. On the
  // figure's owning thread. The renderer's model has to be built again afterwards.
  WearResult commitWear(std::shared_ptr<PreparedWear> wear, const std::string& rel);
  // Takes an item off: its nodes leave the scene (the next build leaves them out), its skeleton
  // stops following, its meshes stop following the body. False for an unknown id.
  bool removeItem(int id);
  bool setItemVisible(int id, bool visible);
  const WornItem* item(int id) const;
  dson::Store&    store() const { return *m_store; }
  // Lists the items the scene file came with (Figure::load does).
  void indexItems();

  // Applies a DAZ pose preset to this figure's bones and morph dials.
  PosePresetResult applyPosePreset(const dson::Json& doc);

private:
  Figure() = default;
  void setup(const Progress& progress);
  void applyOtherDials(const dson::MorphContext& ctx, const std::vector<dson::Item>& items, const std::vector<const dson::SceneDial*>& dials, const Progress& progress);
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
  std::atomic<int>                              m_wears{0};
  int                                           m_nextItem = 1;
};

}  // namespace fv
