#pragma once

// dForce on a wearable: the per-vertex cloth settings a simulation needs. The modifier is usually
// added by the .duf, not the geometry: a scene modifier of type studio/modifier/dynamic_simulation
// on the node, and one studio/simulation_settings/dynamic_simulation per surface (its `groups` name
// the material groups). The numbers live on those modifiers' studio_modifier_channels. Some
// products also write the same names onto the material; those override the modifier when present.
// A port of src/lib/loaders/dson/dforce.ts. The settle itself is dforce/solver.hpp.

#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <figure/dson/channels.hpp>
#include <figure/dson/store.hpp>
#include <figure/mesh/mesh_build.hpp>

namespace fv::dson {

enum class DforceKind
{
  eStatic,
  eDynamic,
  eAddon,
};

// How much body strand hair keeps until someone says otherwise: see dforce::StrandSet::volume
constexpr float kStrandVolume = 0.5f;

// Per-vertex cloth settings, one value per cage vertex.
struct DforceData
{
  DforceKind kind = DforceKind::eDynamic;
  // Node "Freeze Simulation": a dynamic surface held as a collider
  bool freeze = false;
  // Node "Visible in Simulation"
  bool visible = true;
  // 0 sticks to the skinned vertex, 1 is free (a vertex the file's sparse map leaves out is 0);
  // empty means every vertex is free
  std::vector<float> influence;
  // Dynamics Strength x influence, already multiplied
  std::vector<float> strength, stretch, shear, bend, buckling, bucklingRatio, density, contraction, damping, friction, offset;
  std::vector<int16_t> layer;
  std::vector<uint8_t> selfCollide;
  // 0: the vertex's surfaces are hidden from the simulation
  std::vector<uint8_t> surfaceVisible;
  // 0: the vertex's surfaces simulate but neither push nor are pushed ("Collide" off)
  std::vector<uint8_t> surfaceCollide;
  // polygon loops in cage vertex indices, for stretch edges and quad shear: polygon i is
  // polyVerts[polyStart[i] .. polyStart[i + 1])
  std::vector<int32_t> polyStart{0}, polyVerts;

  // Strand hair: polylines, root first, with their surface's settings per strand. Strand i is
  // strandVerts[strandStart[i] .. strandStart[i + 1]).
  std::vector<int32_t> strandStart{0}, strandVerts;
  std::vector<float>   strandStrength, strandOffset, strandFriction, strandShapeRoot, strandShapeTip;
  std::vector<uint8_t> strandCollide;
  // The strands fall under gravity. Off, they keep their authored shape: a styled hair is what most
  // files are, and DAZ Studio simulates none until asked. On from the file only where that shape can't
  // be what was meant, long strands that don't hang (strandsUnsettled), and by the item's Cloth switch.
  bool strandsOn = false;
  float strandVolume = kStrandVolume;

  // Whether the settings cover any polygons, for the cloth solver's constraints.
  bool hasPolygons() const { return polyStart.size() > 1; }
  bool hasStrands() const { return strandStart.size() > 1; }
  bool simulatesStrands() const { return hasStrands() && strandsOn && visible && !freeze; }
  // A mesh the settle should drape, rather than only collide with.
  bool simulated() const { return (hasPolygons() || simulatesStrands()) && visible && !freeze && (kind == DforceKind::eDynamic || kind == DforceKind::eAddon); }
};

// Per-vertex settings for one geometry instance: `scene` is the .duf, `geoDoc` the geometry's own
// file (null when none), `nodeId` the scene node and `geomId` the geometry instance in it, and
// `geometry` the geometry entry. `overrides` are the merged material channels of each material
// group. Null when the node is not a simulation object.
std::shared_ptr<DforceData> buildDforce(const Doc& scene, const Doc* geoDoc, const std::string& nodeId, const std::string& geomId,
                                        const Doc& geometryDoc, const Json& geometry,
                                        const std::map<std::string, Channels>& overrides);

// Whether strand hair was left for a simulation to bring down, not styled: its strands are long
// (a median of 30 cm or more) and fewer than half of those over 15 cm run downward from root to tip.
// In this library, on 2026-10-31, that is Toyen's dForce Long Hair alone, which stands straight out.
// Short hair and updos don't hang either, and are styled that way.
bool strandsUnsettled(std::span<const float> positions, std::span<const int32_t> start, std::span<const int32_t> verts);

// The pre-render (PR) hairs each material group of a strand hair grows, from its surfaces'
// simulation settings ("PR Hairs (Per Guide)", "PR Hair Distribution Radius", "PR Hair Tip
// Separation", bias, gain, seed). Empty unless one of the node's dynamic_generate_hair modifiers has
// "Generate PR Hairs" on.
//
// A surface in "Target Surfaces" mode grows by density instead, at its PR stage or else its PS one
// ("Generate PS Hairs", the hairs DAZ would simulate; a scene without a simulation draws them as they
// grow). Clumping, random angles, parting and Surface Lattice Smooth are not grown. A surface in the
// other mode grows per guide at the PR stage only: its PS hairs, a multiple DAZ adds, are not grown.
std::map<std::string, mesh::Growth> prHairsFor(const Doc& scene, const Doc* geoDoc, const std::string& nodeId, const std::string& geomId);

// What the hairs in `grow` grow from when any has a density: the mesh on the node `node` hangs on, in
// the .duf `scene` from `file`. Null when none has one, or that mesh is missing or has no polygons.
std::shared_ptr<const mesh::GrowthSurface> growthSurfaceFor(Store& store, const Doc& scene, const std::string& file, const Json& node,
                                                            const std::map<std::string, mesh::Growth>& grow);

// Settings for a mesh whose file carries none, as DAZ Studio's "Add dForce Modifier: Dynamic
// Surface" gives it: a dynamic surface at the default values on every vertex (the collision layer
// as given; DAZ's default is 0), over the cage's polygon loops (polygon i is
// polyVerts[polyStart[i] .. polyStart[i + 1])).
std::shared_ptr<DforceData> defaultDforce(int vertexCount, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts, int layer = 0);

}  // namespace fv::dson
