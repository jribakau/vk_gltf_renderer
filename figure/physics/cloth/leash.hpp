#pragma once

// Live cloth tied to the body (physics-plan 13c). Games keep simulated clothing out of the body with a
// backstop each vertex may not go behind, about a reference that moves with the body (Unreal's Backstop,
// NvCloth's separation constraints), beside a Max Distance from it that an artist paints. There the
// reference is the garment's own skin, and a DAZ dress's weights don't follow a thigh swinging into it.
// Here each vertex near the body is tied to the face of the body it lies on (a Ride, physics/cloth/ride.hpp:
// where on its nearest face, and how far off it), again after every step, so the tie goes with the cloth
// as it slides over the body (a skirt over a striding thigh). The backstop is the body's own faces: the
// face the vertex is over, of its tie's and the two rings about it, which it is held out of by its gap
// while it stays within its reach of where it was tied. Further off, it has swung away, and the collisions
// alone meet it, as before. (A sphere under the face, as Unreal's, sized by how round the body is there,
// fell away from the faces beside it along a thigh: of what the hold kept outside its sphere, 84% was in
// the body within 3 cm of its face.)
//
// Which side of the body a vertex belongs on is never taken from where it is: the body's outside is its
// faces' winding, and a vertex under its face a little way in (the wardrobe's pushOut rule, the one the
// drapes are judged by) is in the body and is held back out over it. The sided collisions keep such a
// vertex inside, as one that started a step inside a thigh (a late step carried the dress there by its own
// skin) stayed in for good.
//
// No Max Distance: nothing paints how far each part may go, and one guessed from how near the body a vertex
// lay held a veil on a figure turned over within 2.2 cm of the body, where it should fall away. The hold
// only ever pushes out.
//
// The simulation holds each tied vertex so every substep (both solvers), a late step's carry takes it with
// its face rather than by the garment's own skin, and the drawn drape is held so too between steps.
// Measured on 10TEST's walk (fv_parity's "walk", the Witch Dress on the CPU island, settled with about 40
// vertices in the body): steps every 90 ms, as at a 4K viewport, 63% of each carried, 103 vertices inside
// after each on average untied and 73 tied; steps that keep up (every 1/30 s), 61 and 32. What is left
// tied lies in the creases under the bust and at the armpits, as settled, and at a late step's shin: hem
// that hung further off than any tie, which the leg swings into while the step is carried.

#include <cstdint>
#include <vector>

#include <figure/physics/cloth/ride.hpp>
#include <figure/physics/cloth/surface.hpp>

namespace fv::physics::cloth {

// A vertex within kLeashClose cm of the body is held out of it while within kLeashReach cm of where it was
// tied; one further off, kLeashSlope cm further for each cm, so from 3 cm off within 10: it meets the face
// it lay over from further away.
constexpr float kLeashReach = 2.f;
constexpr float kLeashClose = 1.f;
constexpr float kLeashSlope = 4.f;
// A vertex under its face by up to this much is in the body and is held back out; deeper, its nearest face
// is as likely the far side of a limb (the wardrobe's pushOut and the solvers' recovery trust 3 cm).
constexpr float kLeashInside = 3.f;
// The most a hold moves a vertex in one go, cm, as the solvers' collisions push (their kCap).
constexpr float kLeashPush = 2.f;

// A drape's leashes: its ride on the body (bindRide's: where each vertex lies on its face, from where the
// reach is counted; one in the body, there moved out over its face by its gap), and per riding vertex how
// far from there it is held, and how far out of the body's faces (< 0: not held, beside a crease or deep
// in), cm.
struct Leashes
{
  Ride               ride;
  std::vector<float> reach, gap;

  bool empty() const { return ride.empty(); }
};

// The leashes of a drape where it lies on `ride`: `cloth` is the surface it is (its gaps, by the ride's
// vertex; null: a light copy's cloth, the model's least gap), `body` the body under it (its faces' gaps).
Leashes leash(Ride ride, const Surface* cloth, const Surface& body);
// The same with no reach: each vertex held out of the body however far it has gone from where it was
// bound, while it is over its face or one two sides about it. What Live draws when it meets the body as
// its capsule floor (physics/cloth/proxy.hpp): the drawn cloth held out of the faces the simulation left.
Leashes leashAll(Ride ride, const Surface* cloth, const Surface& body);

// The faces across each side of each face of `tris` (three a face, -1 at a border), as cloth::Model's
// triNext has them for its faces.
std::vector<int> facesAcross(const std::vector<uint32_t>& tris);

// p (xyz) held out of the body at `bodyAt` (its faces `tris`, `across` theirs): the face it is over, of
// `face` and the two rings about it, which becomes `face`; under it by less than kLeashInside, or nearer it
// than `gap`, it is moved out along the face's normal to the gap, by at most `most`. False when it is
// over none of them, or didn't move.
bool holdOut(const std::vector<float>& bodyAt, const std::vector<uint32_t>& tris, const std::vector<int>& across, int& face, float gap, float* p,
             float most = kLeashPush);

// The leashes' vertices `points` (xyz per riding vertex, in the ride's order) held so, each while within
// its reach of where its face has it, the body at `bodyAt` (every vertex of it) with its faces `tris` and
// `across` theirs, each all the way out at once: what is drawn, which nothing reads back. With `moved`, 1
// for each that moved. How many moved.
size_t holdLeashes(const Leashes& l, const std::vector<float>& bodyAt, const std::vector<uint32_t>& tris, const std::vector<int>& across,
                   std::vector<float>& points, std::vector<uint8_t>* moved = nullptr);

// p (xyz) carried with a face from its frame `from` to its frame `to`, as if it were part of it.
void carryOne(const RideFrame& from, const RideFrame& to, float* p);

// A leash in a simulation's index space (cloth::Model's): what a backend holds every substep. The
// backend finds its face among its own from the corners.
struct Tie
{
  int   vertex;     // the cloth's
  int   corner[3];  // its face's, the body's
  float bary[3];
  float local[3];
  float reach, gap;
  // how much a late step's carry takes it with its face, the rest by its own skin: the ride's weight
  float carry;
};

// The leashes as ties: the drape's vertices from `clothBase` in the model, the body's from `bodyBase`.
std::vector<Tie> tiesOf(const Leashes& l, int clothBase, int bodyBase);

// The cloth of `slots` (simulated surfaces of `surfaces`, at `positions`, as a simulation has them) tied to
// the body at `bodySlot` where it lies now, numbered as cloth::Model numbers the surfaces. `rides`, slot by
// slot: the last ties', each vertex found again on its face there (bindRide's hint) unless `fresh` or
// empty; given back as these ties'.
std::vector<Tie> tieTo(const std::vector<Surface>& surfaces, size_t bodySlot, const std::vector<size_t>& slots, const std::vector<std::vector<float>>& positions,
                       std::vector<Ride>& rides, bool fresh);

// What is held to its own skin, per vertex (1): no strength, or left out of the simulation, as the model
// has it. What a ride and a leash leave out.
std::vector<uint8_t> heldOf(const Surface& cloth);

}  // namespace fv::physics::cloth
