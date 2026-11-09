#pragma once

// Body maps (docs/wardrobe-plan.md, step 3; "The map"): every vertex of a body type's base on one
// reference body, Genesis 9's base, and every reference vertex on it. For a Genesis type the reference
// is first shaped as that type by the clone morph Genesis 9 ships for it (CloneGenesis8Female and the
// rest), which lays it within millimetres of the type's own surface, so a vertex's map is its closest
// point there: no solve (3a). Only outer skin is addressed, a vertex matches only within its regions
// (from its own skin weights, rule 3), and only a triangle facing its way; one with nothing within
// reach is left unmapped.
//
// The reference is posed into the type's rest first, at its limb joints (intoRest). Each source
// vertex also has the turn from its frame to the unmorphed Genesis 9 base's where it lands
// (frames.hpp), and a map is kept on disk under its inputs (cache.hpp).
//
// Not here yet: the Skyrim types' fitted maps (3b).

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <string>
#include <vector>

#include <figure/bodymap/regions.hpp>
#include <figure/math/xform.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/generation.hpp>

namespace fv::dson {
class Store;
}
namespace fv::wardrobe {
struct BaseCage;
}

namespace fv::bodymap {

// A body as a map reads it.
struct Body
{
  std::vector<float>      positions;  // its cage, cm
  std::vector<uint32_t>   tris;
  std::vector<uint8_t>    outer;    // per triangle: outer skin, which a map addresses
  std::vector<RegionMask> regions;  // per vertex
};

// A base cage as a map reads it: regions from its skin, outer skin by surface.
Body bodyOf(const wardrobe::BaseCage& cage);

// Per vertex, the unit normal of the outer skin round it (its outer triangles' summed by area); zero
// for a vertex on none.
std::vector<Vec3> outerNormals(const Body& body);
// Per vertex, its share of the outer skin's area: a third of each outer triangle round it.
std::vector<double> vertexAreas(const Body& body);
// The turn about the origin that best takes each of `a` onto its `b`, by least squares (Horn's
// quaternion).
Quat bestTurn(const std::vector<Vec3>& a, const std::vector<Vec3>& b);

// Where a body cut to what shows (an outfit's carried body) ends: the edges no other triangle shares.
class CutEdges
{
public:
  explicit CutEdges(const Body& body);
  // Whether a point of triangle `tri` at barycentrics `w` lies on the cut: on such an edge, or at a
  // corner on one.
  bool on(size_t tri, const double w[3]) const;

private:
  const Body*                             m_body;
  std::set<std::pair<uint32_t, uint32_t>> m_edges;
  std::vector<uint8_t>                    m_corner;
};

struct ProjectOptions
{
  double maxAngle = 60;  // degrees a triangle's facing may lie from the vertex's normal
  double reach    = 5;   // cm: a vertex with nothing nearer is left unmapped
  // A vertex whose closest point is on `onto`'s cut lies past what that body has, and is left unmapped.
  bool refuseCut = false;
  // A vertex may match in the regions next to its own too (withNeighbours): for bodies whose rigs weight
  // differently (a carried Skyrim body's torso is all its chest, Genesis 9's chest and breasts).
  bool widen = false;
};

// Every vertex of `from` on its outer skin, at the closest point of `onto`'s outer skin within its
// regions (a vertex with none matches anywhere) and facing its way.
struct Projection
{
  wardrobe::SurfaceBinding at;      // three `onto` vertices and their weights, per `from` vertex
  std::vector<uint8_t>     mapped;  // per `from` vertex: whether it has a place
};
Projection project(const Body& from, const Body& onto, const ProjectOptions& options = {});

// A body type's map: its vertices on the reference, and the reference's on it; and per source vertex
// the turn from its frame to the unmorphed Genesis 9 base's where it lands (frameTurns), which the
// type's map maker fills (mapBodies alone leaves it empty).
struct BodyMap
{
  Projection        toReference, fromReference;
  std::vector<Quat> frameTurn;
};
BodyMap mapBodies(const Body& source, const Body& reference, const ProjectOptions& options = {});

// How good a map is (the step's gate): the round trip source -> reference -> source at each source
// vertex, in cm, by where it is (armpit, crotch, gluteal fold, under-bust, fingers, the rest, all);
// how far the source lies from where it lands on the reference; the shares mapped each way; each
// source triangle's stretch (the larger singular value of its map onto the reference); and the share
// of its triangles the map doesn't turn over.
struct MapQuality
{
  struct Spread
  {
    size_t n = 0;
    double median = 0, p95 = 0, max = 0;
  };
  std::map<std::string, Spread> roundTrip, distance;
  // the source's outer vertices left unmapped, and those landing over 1 cm off, counted by their regions
  std::map<std::string, size_t> unmappedBy, farBy;
  // the source's triangles the map turns over, by their first corner's regions
  std::map<std::string, size_t> foldedBy;
  // the source's triangles stretched past half again their size one way, by their first corner's regions
  std::map<std::string, size_t> stretchedBy;
  std::vector<uint32_t>         foldedTris;
  double                        mapped = 0, inverseCovers = 0, stretchP95 = 0, stretchMax = 0, foldFree = 0;
};
MapQuality mapQuality(const Body& source, const Body& reference, const BodyMap& map);

// The clone morph Genesis 9 ships for a generation's figure ("data/DAZ 3D/Genesis 9/Base/Morphs/Daz
// 3D/Clones/CloneGenesis8Female.dsf"), `gender` "female" or "male" ("" reads as female); "" for
// Genesis 9 and Skyrim, which have none.
std::string cloneFile(wardrobe::Generation generation, const std::string& gender);
// Genesis 9's base shaped as that type by its clone morph, or none when either isn't installed.
std::optional<Body> cloneReference(dson::Store& store, wardrobe::Generation generation, const std::string& gender);

// A part turned into the source's rest: its name ("shin", "index"), side, and how far it turned from
// the reference's rest, in degrees.
struct LimbTurn
{
  std::string joint;
  char        side    = 0;
  double      degrees = 0;
};
// The reference's cage (`shaped`: its positions shaped as the source's by the clone) turned part by
// part into the source's rest, then skinned by its own weights, without correctives. A clone reshapes
// Genesis 9 but keeps its stance, so a clone's legs, and Genesis 3's arms, lie elsewhere than the
// source's. The parts are the upper arm, forearm and hand, each finger whole, the thigh, shin and foot,
// and each toe whole, on each side. A part turns about its joint, found on each body alike, from its own
// weights, as the middle of the band where its region meets the one above (the knee where thigh meets
// shin), so where each rig puts its joint centres doesn't matter. A limb part is first pointed at the
// band where it meets the next, as the source's is; then each part, parents first, takes the turn that
// best brings its own vertices onto the source's skin in its region (closest points, Horn's fit, a few
// rounds), the parts below it turning with it.
std::vector<float> intoRest(const wardrobe::BaseCage& reference, const std::vector<float>& shaped, const Body& source, std::vector<LimbTurn>* turns = nullptr);

// What a type's clone map is made from: the type's base, Genesis 9's, and Genesis 9's shaped as the
// type by its clone (unposed). None when one of them isn't installed.
struct CloneInputs
{
  std::shared_ptr<const wardrobe::BaseCage> base, g9;
  Body                                      reference;
};
std::optional<CloneInputs> cloneInputs(dson::Store& store, wardrobe::Generation generation, const std::string& gender);

// A type's map: its body; the reference as laid onto it (posed into its rest, and for a fitted type
// deformed onto it, fit.hpp); the map both ways with the frame turns; and the limbs' turns into its rest.
struct TypeMap
{
  Body                  source, reference;
  BodyMap               map;
  std::vector<LimbTurn> turns;
};
TypeMap makeCloneMap(const CloneInputs& inputs);
// cloneInputs, then makeCloneMap through the cache (cachedCloneMap); none when cloneInputs has none.
std::optional<TypeMap> cloneMap(dson::Store& store, wardrobe::Generation generation, const std::string& gender);

}  // namespace fv::bodymap
