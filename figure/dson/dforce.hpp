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

namespace fv::dson {

enum class DforceKind
{
  eStatic,
  eDynamic,
  eAddon,
};

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

  // Whether the settings cover any polygons. Strand hair carries a simulation modifier on polylines
  // alone; the settle has no constraints for those, so it neither drapes nor collides with them.
  bool hasPolygons() const { return polyStart.size() > 1; }
  // A mesh the settle should drape, rather than only collide with.
  bool simulated() const { return hasPolygons() && visible && !freeze && (kind == DforceKind::eDynamic || kind == DforceKind::eAddon); }
};

// Per-vertex settings for one geometry instance: `scene` is the .duf, `geoDoc` the geometry's own
// file (null when none), `nodeId` the scene node and `geomId` the geometry instance in it, and
// `geometry` the geometry entry. `overrides` are the merged material channels of each material
// group. Null when the node is not a simulation object.
std::shared_ptr<DforceData> buildDforce(const Doc& scene, const Doc* geoDoc, const std::string& nodeId, const std::string& geomId,
                                        const Doc& geometryDoc, const Json& geometry,
                                        const std::map<std::string, Channels>& overrides);

// Settings for a mesh whose file carries none, as DAZ Studio's "Add dForce Modifier: Dynamic
// Surface" gives it: a dynamic surface at the default values on every vertex (the collision layer
// as given; DAZ's default is 0), over the cage's polygon loops (polygon i is
// polyVerts[polyStart[i] .. polyStart[i + 1])).
std::shared_ptr<DforceData> defaultDforce(int vertexCount, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts, int layer = 0);

}  // namespace fv::dson
