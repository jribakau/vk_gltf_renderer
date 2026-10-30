#pragma once

// A loaded DAZ scene as the renderer works with it: the scene graph the loader built, its morphs
// and the formulas that drive them, and the pose rig. Ports the morph and posing half of
// src/lib/loaders/dson/figure.ts, which wired these together as the last stages of a load.
//
// Threads: one thread uses a Figure at a time (the load worker while it loads, then the renderer's
// application thread). Morph files are read on the thread pool; pump() takes what arrived.

#include <functional>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <atomic>
#include <string>
#include <unordered_map>
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
  // Nodes, by DAZ name, the preset shows or hides (`extra/studio_node_channels/channels/Visible`).
  std::vector<std::pair<std::string, bool>> visibility;
  // Worn items Figure::applyPosePreset showed or hid for them (ids): the scene needs building again.
  std::vector<int> itemsShown;
};

// Applies a DAZ pose, shape or property preset (.duf `scene.animations`) to a rig and its morphs:
// the last key of each animation, `name://@selection` urls, `name://<node>#<modifier>` urls (a dial
// of a worn item, a geograft's say) and hierarchical `#bone?rotation/x/value` urls. Node visibility
// is reported, not applied. Figure::applyPosePreset is this, on the figure's own pose and morphs.
// `channelOf` finds the dial a modifier id means on a node (empty: the selection) when the figure
// has none of that name: a worn item's.
PosePresetResult applyPosePreset(pose::PoseRig* pose, morph::MorphSet* morphs, const dson::Json& doc,
                                 const std::function<std::string(const std::string& node, const std::string& modifier)>& channelOf = {});

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
  // the conforming meshes the collider keeps out of the body, with each one's smoothing settings
  // (the file's modifier, or defaults for clothing): empty for hair, anatomy and props without one
  std::map<int, dson::Smoothing> collision;
  // how far the collider pulls it onto the body, 0 (as it was made) to 1 (skin tight)
  float tight = 0;
  // the body polygons it covers, not drawn while it is tight (Figure::refreshTightHidden); whether
  // that has been worked out for it
  std::vector<int32_t> tightPolys;
  bool                 tightChecked = false;
  // dForce on this item as the user set it (Figure::setItemCloth): whether its meshes simulate
  // (unset: as the file says, a Dynamic Surface does and anything else conforms) and a dynamics
  // strength that scales the file's weights, or is every vertex's for an item the file gave none
  struct Cloth
  {
    std::optional<bool> on;
    float               strength = 1.f;
  };
  Cloth cloth;
  // each mesh's settings as its file had them (null: none), kept once the user first changes them
  std::map<int, std::shared_ptr<const dson::DforceData>> fileDforce;
  // foot channels it set on the figure (a heel poses the feet), and the values they had before
  struct FootSet
  {
    std::string bone, axis;
    double      value = 0, before = 0;
  };
  std::vector<FootSet> footSet;
};

// The settings a worn mesh simulates with, as the user's choice makes them: `on` unset keeps what
// the file says (a Dynamic Surface drapes, anything else conforms), true drapes the mesh (at the
// defaults of DAZ Studio's "Add dForce Modifier" when the file gives none, over the cage's polygon
// loops), false leaves it conforming (a file's dynamic surface held frozen, so it still collides).
// `strength` scales the weights; `layer` is the collision layer settings made here get. Null when
// the mesh has no polygons (strand hair) or is left as the file has it with none.
std::shared_ptr<const dson::DforceData> clothSettings(const mesh::BuiltMesh& geo, const std::shared_ptr<const dson::DforceData>& file,
                                                      std::optional<bool> on, float strength, int layer = 0);

// Per scene material: whether light scattering under the skin passes through it rather than
// leaving the skin there (the renderer's "fvWalkThrough"). Lashes, hair, brows and the eyes' wet
// films are never the skin's boundary, and neither is anything worn but anatomy: clothing, hair and
// props reach under the skin where they intersect it. The body, its geografts (whatever kind their
// path gives them) and anatomy add-ons (Genesis 9's eyes and mouth close the head) bound it. A
// material that a bounding mesh uses too bounds it.
std::vector<bool> walkThroughMaterials(const SceneGraph& scene, const std::vector<WornItem>& items);

// How a worn item stands with dForce, for the panels.
struct ClothState
{
  bool  able     = false;  // it has polygons to simulate
  bool  native   = false;  // its file carries simulation settings
  bool  on       = false;  // a mesh of it drapes now
  float strength = 1.f;
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

  // A wearable loaded and bound to the body; commitWear puts it on. It reads the figure's skeleton,
  // joint centres and body cage and changes nothing. On a worker while the figure is in use, pass
  // `base`, a wearBase() taken on the figure's thread: formulas move joint centres as dials change,
  // so the live scene graph is not safe to read from another thread.
  struct PreparedWear;
  // The scene graph a wear reads, copied: joints copied, built meshes shared (only commitWear,
  // on the owning thread, changes a cage). Owning thread.
  std::shared_ptr<const SceneGraph> wearBase() const;
  // The body's morph targets dialled in now: given to prepareWear(), it projects them onto the
  // wearable on its thread, so commitWear() finds the followers' shapes ready. Owning thread.
  std::vector<dson::TargetPtr> activeBodyTargets() const;
  // Starts reading a wearable's library files on the pool, for a prepareWear() that comes later.
  // Any thread.
  void prefetchWear(const std::string& rel) const { dson::prefetchScene(m_store, rel); }
  // The wearable's own scene, for resolving its textures before it is merged.
  static const SceneGraph& preparedScene(const PreparedWear& wear);
  static SceneGraph&       preparedScene(PreparedWear& wear);
  // Everything a prepareWear() reads of the figure, copied on its owning thread, so prepares can run
  // while the owner commits other wears (a commit grows the rigs and meshes a live read would use).
  struct WearInputs
  {
    std::shared_ptr<const SceneGraph> scene;  // wearBase()
    std::shared_ptr<const dson::Rig>  rig;    // the main rig
    int                               bodyMesh = -1;
    std::string                       morphDir;
    std::vector<dson::TargetPtr>      bodyTargets;  // activeBodyTargets()
  };
  // Owning thread. Throws when there is no figure with a skeleton to wear things on.
  WearInputs wearInputs() const;
  // Loads, fits and binds a wearable against `inputs`; reads nothing else of the figure, so any
  // thread may run several at once. Its key ("wN", which names its adjustment channels) is drawn
  // when it is committed, so keys follow the order wears are put on, not the order they finish.
  // `strandsFlipped`, when given, measures a strand hair's directions (build::measureStrands with
  // that flag) beside the binding, so the host's measure after the material presets finds them done.
  std::shared_ptr<PreparedWear> prepareWear(const std::string& rel, int subdivision, const WearInputs& inputs,
                                            std::optional<bool> strandsFlipped = std::nullopt);
  // The same against the figure as it is now, on its owning thread.
  std::shared_ptr<PreparedWear> prepareWear(const std::string& rel, int subdivision) { return prepareWear(rel, subdivision, wearInputs()); }
  // Reads each surface again knowing what its mesh belongs to (mat::Context), once its load is in
  void placeLoadedSurfaces(const std::string& rel);
  // Merges a prepared wearable into the scene: nodes, skeleton, materials, morph followers. On the
  // figure's owning thread. The renderer's model has to be built again afterwards. `ids`: the ids its
  // items take, in the file's order, when they come back (an undo, a scene file); the rest get new ones.
  // `keys` the same for their keys, so what names their dials ("Loosen Neck@w3": a scene file, the
  // history) still does. `only`: just the items with an id are coming back, so the file's other
  // items, worn or not, aren't replaced; the caller takes the new copies of those off again.
  WearResult commitWear(std::shared_ptr<PreparedWear> wear, const std::string& rel, const std::vector<int>& ids = {},
                        const std::vector<std::string>& keys = {}, bool only = false);
  // Takes an item off: its nodes leave the scene (the next build leaves them out), its skeleton
  // stops following, its meshes stop following the body. False for an unknown id.
  bool removeItem(int id);
  bool setItemVisible(int id, bool visible);
  // dForce on one item: see clothSettings. The meshes it changed are returned (the simulation and
  // the collider are to be told); empty for an unknown id or an item with nothing to simulate.
  std::vector<int> setItemCloth(int id, std::optional<bool> on, float strength);
  ClothState       clothOf(const WornItem& item) const;
  // How skin tight one item is, 0 to 1: the collider's next pass applies it. False for an unknown id.
  bool setItemTight(int id, float amount);
  // Brings what tight items hide of the body up to date: the skin fully under opaque fabric stops being
  // drawn, so nothing of it can show through at a sharp joint. Worked out once per item, at rest, and
  // not for fabric that lets anything through or that dForce moves. True when the body's drawn
  // triangles changed, which takes a composition.
  bool refreshTightHidden(bool collisionOn);
  const WornItem*  item(int id) const;
  dson::Store&    store() const { return *m_store; }
  // Lists the items the scene file came with (Figure::load does).
  void indexItems();

  // Applies a DAZ pose, shape or property preset to this figure's bones, morph dials and the
  // visibility of what it wears.
  PosePresetResult applyPosePreset(const dson::Json& doc);

private:
  Figure() = default;
  void setup(const Progress& progress);
  void applyOtherDials(const dson::MorphContext& ctx, const std::vector<dson::Item>& items, const std::vector<const dson::SceneDial*>& dials, const Progress& progress);
  void finishCorrectives();
  // Tells the channels what their files say of them, when that has been read or new channels came,
  // and loads the controllers of dials that moved. True when it started a load.
  bool noteMorphs();

  struct MorphNotes
  {
    std::unordered_map<std::string, dson::MorphInfo> info;
    // a dial's id or name -> the lazy controllers that follow it (dson::MorphInfo::follows)
    std::unordered_map<std::string, std::vector<std::string>> followers;
  };

  struct CorrectiveFile
  {
    std::string                      file;
    std::shared_ptr<const dson::Doc> doc;
    std::vector<dson::MorphDef>      defs;
  };

  std::shared_ptr<dson::Store>                  m_store;
  bool                                          m_cageMorphs = false;  // dson::LoadOptions::cageMorphs
  std::shared_ptr<pose::FigureSource>           m_source;
  std::vector<std::string>                      m_correctiveFiles;
  std::vector<std::future<CorrectiveFile>>      m_correctiveJobs;
  bool                                          m_correctivesStarted = false;
  std::future<MorphNotes>                       m_morphInfoJob;
  MorphNotes                                    m_morphNotes;
  int                                           m_morphsNoted = -1;
  int                                           m_followersSeen = -1;
  std::atomic<int>                              m_wears{0};
  int                                           m_nextItem = 1;
};

}  // namespace fv
