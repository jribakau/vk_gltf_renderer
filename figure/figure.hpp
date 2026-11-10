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
#include <span>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include <figure/dson/addons.hpp>
#include <figure/dson/figure_loader.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/physics/skin.hpp>
#include <figure/pose/figure_driver.hpp>
#include <figure/pose/morph_reach.hpp>
#include <figure/pose/pose_clip.hpp>
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
// the last key of each animation, its url resolved as pose::PresetResolver does (pose_clip.hpp).
// Node visibility is reported, not applied. Figure::applyPosePreset is this, on the figure's own pose
// and morphs. `channelOf` finds the dial a modifier id means on a node when the figure has none of
// that name: a worn item's.
PosePresetResult applyPosePreset(pose::PoseRig* pose, morph::MorphSet* morphs, const dson::Json& doc, const pose::ChannelOf& channelOf = {});

// What the figure wears: one wearable's nodes, which go on and come off together. Items that came
// with the scene file are listed too, with no library path.
// How much of its geometry a wearable is built with: all of it, until it doesn't fit GPU memory.
struct WearQuality
{
  int   subdivision  = 0;  // dson::LoadOptions::subdivision
  float strandKeep   = 1;  // dson::LoadOptions::strandKeep
  int   strandPieces = 2;  // dson::LoadOptions::strandPieces
};
// Fewer than this and a hair's strands, as wide as they would have to be, read as cards
constexpr float kLeastStrandKeep = 0.25f;
// One step down: a subdivision level less, and a hair's segments in one piece (the same shape in half
// the triangles, traced a little slower), then half the strands; as far as each goes.
WearQuality oneStepDown(const WearQuality& quality);
// One step up, back the way down came: as far as every level there is at `subdivision`, all the
// strands, then two pieces a segment.
WearQuality oneStepUp(const WearQuality& quality, int subdivision);
// What these meshes are built at now: the finest level of those subdivided, the largest share of strands
WearQuality qualityOf(const SceneGraph& scene, const std::vector<int>& meshes);
// The share of what a mesh draws now that it would draw at `quality`; 1 for what has no such choice.
float shareAt(const Mesh& mesh, const WearQuality& quality);
// Whether a step down from `quality` would make any of these meshes smaller.
bool wearCanShrink(const SceneGraph& scene, const WearQuality& quality);
// What `meshes` came in without at `quality`, where everything is at `subdivision`: "half its
// strands and Subdivision 1". Empty when they have all of it.
std::string wearShortfall(const SceneGraph& scene, const std::vector<int>& meshes, const WearQuality& quality, int subdivision);

// A worn item's meshes made again at another quality over the cages they have (Figure::remakeInputs). Only
// what is drawn changes: dials, the pose, a drape and bindings live on the cage.
struct RemadeItem
{
  int         item = 0;
  WearQuality quality;
  struct Part
  {
    int                                         mesh = -1;
    std::shared_ptr<mesh::BuiltMesh>            from, made;
    std::vector<float>                          cage;
    std::map<std::string, std::vector<int32_t>> hidden;
    mesh::SkinInput                             cageSkin;  // empty: the mesh keeps no drawn skin
    mesh::SkinOutput                            skin;
  };
  std::vector<Part> parts;
  // the figure's materials and these meshes alone: a hair's strands are measured here before it goes on
  SceneGraph scene;
};

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
  // what it is drawn without to fit GPU memory (wearShortfall); empty: all of it
  std::string reduced;
  // short of GPU memory, its textures come down only when nothing else can
  bool keepFull = false;
  // Made for another generation and refitted: by which path, "old" or "common" (wardrobe::FitPath); ""
  // for one made for this figure, which isn't fitted. `fitAble`: the common layer can fit it, so there
  // is a choice. `commonFit`: what its wear asked for (WearInputs::commonFit).
  std::string fitted;
  bool        fitAble = false, commonFit = false;
  // Its file has no simulation settings and none to be had (a Skyrim outfit): draped, its cloth on the
  // skin stays held and its trim is sewn to the cloth under it (clothPins)
  bool pinBySkin = false;
  // body polygons a geograft hides while it is worn and shown
  std::vector<int32_t> hiddenPolys;
  // the conforming meshes the collider keeps out of the body, with each one's smoothing settings
  // (the file's modifier, or defaults for clothing): empty for hair, anatomy and props without one
  std::map<int, dson::Smoothing> collision;
  // each conforming mesh's binding to the body's cage and its rigid groups, as its morphs follow the
  // body's: what moves the body after them moves it too (soft flesh, physics::followFlesh). Not strand
  // hair, which hangs from where it touches
  struct Follow
  {
    int                                             mesh = -1;  // into SceneGraph::meshes
    std::shared_ptr<const wardrobe::SurfaceBinding> binding;
    std::shared_ptr<const wardrobe::Rigidity>       rigidity;  // null: none
  };
  std::vector<Follow> follows;
  // how far the collider pulls it onto the body, 0 (as it was made) to 1 (skin tight)
  float tight = 0;
  // the body polygons it covers, not drawn while it is tight (Figure::refreshTightHidden); whether
  // that has been worked out for it
  std::vector<int32_t> tightPolys;
  bool                 tightChecked = false;
  int                  tightDials   = -1;  // MorphSet::dialVersion when the covered skin was measured
  // dForce on this item as the user set it (Figure::setItemCloth): whether its meshes simulate
  // (unset: as the file says, a Dynamic Surface does and anything else conforms) and a dynamics
  // strength: see clothSettings
  struct Cloth
  {
    std::optional<bool> on;
    float               strength = 1.f;
    // strand hair's body: see physics::hair::StrandSet::volume
    float               volume = dson::kStrandVolume;
  };
  Cloth cloth;
  // each mesh's settings as its file had them (null: none), kept once the user first changes them
  std::map<int, std::shared_ptr<const dson::DforceData>> fileDforce;
  // foot channels it set on the figure (a heel poses the feet; of a file's items, the one most on the
  // feet holds them), and the values they had before: a bone's rotation, or the figure's `dial`
  struct FootSet
  {
    std::string bone, axis;
    double      value = 0, before = 0;
    std::string dial;
  };
  std::vector<FootSet> footSet;
};

// The settings a worn mesh simulates with, as the user's choice makes them: `on` unset keeps what
// the file says (a Dynamic Surface drapes, anything else conforms), true drapes the mesh (at the
// defaults of DAZ Studio's "Add dForce Modifier" when the file gives none, over the cage's polygon
// loops), false leaves it conforming (a file's dynamic surface held frozen, so it still collides).
// `strength` is 1 for the file's weights; below that it scales them towards following the body, and
// above it lifts them towards fully cloth, there at kClothLoosest, a vertex the file holds at 0 stays
// held. `layer` is the collision layer settings made here get. Null when the mesh has no polygons
// (strand hair) or is left as the file has it with none.
//
// Settings made here, for a file with none, go over the cage's polygons with its triangles paired back
// into quads (garment::pairQuads), so a mesh of triangles shears as cloth does. `free`, when it has a
// value per cage vertex, is their weights: 0 held to the skin, 1 cloth (clothPins). `stitches` are
// couples of cage vertices kept at their distance apart, a polygon of two each (ClothPins::stitches).
constexpr float kClothLoosest = 2.f;
std::shared_ptr<const dson::DforceData> clothSettings(const mesh::BuiltMesh& geo, const std::shared_ptr<const dson::DforceData>& file,
                                                      std::optional<bool> on, float strength, int layer = 0, float volume = dson::kStrandVolume,
                                                      std::span<const float> free = {}, std::span<const int32_t> stitches = {});
// The strength past which those settings no longer change: only weights between none and full can loosen.
float clothLoosest(const std::shared_ptr<const dson::DforceData>& file);
// A pin map for a mesh whose file has none, by how far off the skin each cage vertex lies where `at`
// binds it to the body's cage: 0 (held) within physics::cloth::kOnSkin cm, 1 (cloth) from kOnSkinFade
// further, as the live view holds what lies on the skin.
std::vector<float> skinFree(std::span<const float> item, std::span<const float> body, const wardrobe::SurfaceBinding& at);
// That, and the mesh's hard pieces (garment::hardPieces, by its cage and `skin`), which are joined to
// nothing and as cloth would fall from the cloth they sit on. Trim (a piece up to 8 cm) within 3 cm of
// the mesh's own cloth is sewn to it (garment::stitchPieces) and is as free as the cloth it is sewn to;
// where that is held, so is it, with no stitches. A piece that finds no cloth, and a plate, is held: it
// stays where the outfit's bones carry it.
struct ClothPins
{
  std::vector<float>   free;      // per cage vertex: 0 held, 1 cloth
  std::vector<int32_t> stitches;  // (trim vertex, cloth vertex), two entries a stitch
  size_t               sewn = 0, held = 0;  // pieces sewn to free cloth, and pieces held
};
ClothPins clothPins(const mesh::BuiltMesh& geo, const mesh::SkinInput& skin, std::span<const float> body, const wardrobe::SurfaceBinding& at);

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
  bool  able     = false;  // it has polygons or strands to simulate
  bool  native   = false;  // its file carries simulation settings
  bool  on       = false;  // a mesh of it drapes now
  float strength = 1.f;
  float loosest  = 1.f;  // see clothLoosest
  bool  strands  = false;  // strand hair, whose `volume` means something
  float volume   = dson::kStrandVolume;
};

struct WearResult
{
  std::vector<int> items;     // ids of what went on
  std::vector<int> replaced;  // ids taken off because these are the same garment
  // share of the bones that found a figure bone, of the items not known to share the figure's rig: low
  // means it was made for another figure
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

  // What changed since last asked: bone nodes to re-pose, meshes whose vertices moved, meshes whose
  // skins were bound again at new joint centres, and nodes carried by what they ride on (rigid follow
  // nodes: moved to feed, but no pose of their own, so nothing that answers a pose need look again).
  struct Changes
  {
    std::set<int>                  nodes;
    std::vector<morph::MeshChange> meshes;
    std::vector<int>               rebound;
    std::set<int>                  carried;
    bool empty() const { return nodes.empty() && meshes.empty() && rebound.empty() && carried.empty(); }
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
  // A worn item's own joint correctives (`files`, its Morphs folder's), for the ones whose names the
  // figure has none of: MUO Sweater's forearm twists on Genesis 9. Read after the figure's own, so a
  // name both have stays the figure's (whose follower brings the item's copy along); each adds its
  // formulas to the driver and its target as a hidden channel on the item's mesh.
  void addItemCorrectives(const std::vector<std::string>& files, const dson::MeshRef& ref);

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
  void prefetchWear(const std::string& rel) const
  {
    // a Skyrim mesh (.nif) isn't a library file: prepareWear reads it
    if(rel.size() < 4 || (rel.compare(rel.size() - 4, 4, ".nif") != 0 && rel.compare(rel.size() - 4, 4, ".NIF") != 0))
      dson::prefetchScene(m_store, rel);
  }
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
    // not the figure's, the caller's choice for this wear: an item of another generation is fitted
    // through the common layer where that can fit it (wardrobe::FitPath::eCommon)
    bool commonFit = false;
  };
  // Owning thread. Throws when there is no figure with a skeleton to wear things on.
  WearInputs wearInputs() const;
  // Loads, fits and binds a wearable against `inputs`; reads nothing else of the figure, so any
  // thread may run several at once. Its key ("wN", which names its adjustment channels) is drawn
  // when it is committed, so keys follow the order wears are put on, not the order they finish.
  // `strandsFlipped`, when given, measures a strand hair's directions (build::measureStrands with
  // that flag) beside the binding, so the host's measure after the material presets finds them done.
  std::shared_ptr<PreparedWear> prepareWear(const std::string& rel, const WearQuality& quality, const WearInputs& inputs,
                                            std::optional<bool> strandsFlipped = std::nullopt);
  // The same against the figure as it is now, on its owning thread.
  std::shared_ptr<PreparedWear> prepareWear(const std::string& rel, const WearQuality& quality) { return prepareWear(rel, quality, wearInputs()); }
  std::shared_ptr<PreparedWear> prepareWear(const std::string& rel, int subdivision) { return prepareWear(rel, WearQuality{subdivision}); }
  // Reads each surface again knowing what its mesh belongs to (mat::Context), once its load is in
  void placeLoadedSurfaces(const std::string& rel);
  // Merges a prepared wearable into the scene: nodes, skeleton, materials, morph followers. On the
  // figure's owning thread. The renderer's model has to be built again afterwards. `ids`: the ids its
  // items take, in the file's order, when they come back (an undo, a scene file); the rest get new ones.
  // `keys` the same for their keys, so what names their dials ("Loosen Neck@w3": a scene file, the
  // history) still does. `only`: just the items with an id are coming back, so the file's other
  // items, worn or not, aren't replaced; the caller takes the new copies of those off again.
  // `posesFeet`: the file's foot pose goes on the figure (a heel's); a scene file coming back has the
  // feet as they were saved, which that pose would turn a second time.
  WearResult commitWear(std::shared_ptr<PreparedWear> wear, const std::string& rel, const std::vector<int>& ids = {},
                        const std::vector<std::string>& keys = {}, bool only = false, bool posesFeet = true);
  // The worn items commitWear would replace, given the same `ids` and `only`: the same garment again.
  std::vector<int> itemsReplacedBy(const PreparedWear& wear, const std::vector<int>& ids, bool only) const;
  // What building the item at `quality` reads, copied; null when it is gone or none of it would change. Owning thread.
  std::shared_ptr<RemadeItem> remakeInputs(int id, const WearQuality& quality) const;
  // Any thread
  static void buildRemade(RemadeItem& item);
  // Swaps them in and says what the item is drawn without, where everything is at `subdivision`. False, and
  // nothing changed, when the item went or a cage moved meanwhile. Owning thread.
  bool commitRemade(RemadeItem& item, int subdivision);
  // Takes an item off: its nodes leave the scene (the next build leaves them out), its skeleton
  // stops following, its meshes stop following the body. False for an unknown id.
  bool removeItem(int id);
  bool setItemVisible(int id, bool visible);
  // Marks an item Keep full, or not: its surfaces say so (Material::keepFull). False for an unknown id.
  bool setItemKeepFull(int id, bool on);
  // dForce on one item: see clothSettings. The meshes it changed are returned (the simulation and
  // the collider are to be told); empty for an unknown id or an item with nothing to simulate.
  std::vector<int> setItemCloth(int id, std::optional<bool> on, float strength, float volume = dson::kStrandVolume);
  ClothState       clothOf(const WornItem& item) const;
  // Whether one of these items is strand hair falling by its own settings, as nobody has said
  // otherwise (dson::strandsUnsettled), so the figure's dForce is to be on for it.
  bool             fallsByItself(const std::vector<int>& items) const;
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
  // The conforming items the scene file came with follow the body as a worn one does (Figure::load
  // does, after indexItems): each mesh bound to the body's cage, the body's morphs projected onto it
  // where the scene set no copy of its own, its own copies and correctives, its fit adjustments as
  // dials, its binding kept for the flesh.
  void followSceneItems();

  // Applies a DAZ pose, shape or property preset to this figure's bones, morph dials and the
  // visibility of what it wears.
  PosePresetResult applyPosePreset(const dson::Json& doc);
  // A pose file's keys over time, resolved against this figure's bones, its dials and its worn items' dials.
  // A file made for another Genesis (its bones named so; `rel`, its path, says which when the names
  // don't) is retargeted from that figure's skeleton (pose/retarget.hpp); `retargetedFrom` then names
  // the skeleton's file.
  pose::PoseClip loadPoseClip(const dson::Json& doc, const std::string& rel = {}, std::string* retargetedFrom = nullptr);

private:
  // a worn item by one of its nodes' DAZ names, or its label
  WornItem* itemNamed(const std::string& node);
  // the dial a preset's modifier means on a worn item (pose::ChannelOf)
  pose::ChannelOf wornDials();
  Figure() = default;
  void setup(const Progress& progress);
  // A conforming part's mesh (`mesh`, into the scene) follows the body by `binding`: the body's morphs
  // projected onto it (`preprojected`: those made ahead), its own copies of them from `own` where it
  // has one (none when `refit`, made for another generation), its own joint correctives, a graft's
  // hidden skin (`carriedHides` for a graft made for another mesh: the body's polygons the common fit
  // found under it, wardrobe::FitDetail::hides); what can't be done goes to `warnings`.
  void followBody(WornItem& worn, const dson::ItemPart& part, int mesh, std::shared_ptr<const wardrobe::SurfaceBinding> binding,
                  const std::map<std::string, std::string>& own, const dson::MeshRef& ref, bool refit,
                  std::vector<std::pair<dson::TargetPtr, dson::TargetPtr>> preprojected, std::vector<std::string>& warnings,
                  const std::vector<int32_t>* carriedHides = nullptr);
  // Its own morphs in `own` that aren't copies of a figure morph, fit adjustments ("Loosen Neck") and
  // styles, as dials of its own ("Loosen Neck@<key>").
  void addAdjustments(WornItem& worn, const std::map<std::string, std::string>& own, const dson::MeshRef& ref);
  void applyOtherDials(const dson::MorphContext& ctx, const std::vector<dson::Item>& items, const std::vector<const dson::SceneDial*>& dials, const Progress& progress);
  void finishCorrectives();
  // Tells the channels what their files say of them, when that has been read or new channels came,
  // and loads the files that read dials that moved. True when that left a load or the driver to run.
  bool noteMorphs();
  // Moves the rigid follow nodes (dson::RigidFollow) with the patches they follow when a pose, a shape
  // or a drape moved those, and adds the nodes it moved to `c.carried`.
  void followRigid(Changes& c);
  // per entry of loaded.follows (empty: to be made again): its reference vertices at rest and as the
  // morphs last had them, and their skin
  struct FollowState
  {
    std::vector<float> rest, morphed;
    physics::CageSkin  skin;
    bool               skinned = false;
  };
  std::vector<FollowState> m_followState;

  struct MorphNotes
  {
    std::unordered_map<std::string, dson::MorphInfo> info;
    // for the driver, which takes it on arrival
    pose::MorphFiles                                 files;
    // for the reach audit, which starts on arrival; the only copy of the formulas
    std::vector<pose::MorphFact>                     facts;
  };

  struct CorrectiveFile
  {
    std::string                      file;
    std::shared_ptr<const dson::Doc> doc;
    std::vector<dson::MorphDef>      defs;
    bool                             item = false;  // a worn item's (addItemCorrectives)
  };
  // starts reading the items' corrective files waiting, but the ones the figure has a channel for
  void startItemCorrectives();
  // a morph of that name is the figure's: a channel, or a file in its Morphs folders
  bool figureHas(const std::string& name) const;

  std::shared_ptr<dson::Store>                  m_store;
  bool                                          m_cageMorphs = false;  // dson::LoadOptions::cageMorphs
  std::shared_ptr<pose::FigureSource>           m_source;
  std::vector<std::string>                      m_correctiveFiles;
  std::vector<std::future<CorrectiveFile>>      m_correctiveJobs;
  bool                                          m_correctivesStarted = false;
  // worn items' corrective files waiting for the figure's own to be read, and the channels they made
  std::vector<std::pair<std::string, dson::MeshRef>> m_itemCorrectiveFiles;
  std::set<std::string>                         m_itemCorrectives;
  std::future<MorphNotes>                       m_morphInfoJob;
  MorphNotes                                    m_morphNotes;
  // what each dial can do: it reads every morph's deltas, so nothing waits for it, settle() included
  std::future<pose::DialReach>                  m_morphReachJob;
  pose::DialReach                               m_morphReach;
  int                                           m_morphsNoted = -1;
  int                                           m_readersSeen = -1;
  std::atomic<int>                              m_wears{0};
  int                                           m_nextItem = 1;
};

}  // namespace fv
