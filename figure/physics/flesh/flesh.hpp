#pragma once

// Soft flesh (docs/physics-plan.md step 8): the chest, the glutes, the belly and the thighs trail
// the body's motion and settle back. Each lobe's cage vertices are particles held to where the skin
// puts them; what moves is only how far they stray from it, so the rest state is exactly the skinned
// shape, DAZ's own, and nothing is ever baked.
//
// The solve is in that offset, d = x - s (the particle against its skinned target), in the body
// mesh's posed space (DSON centimetres):
//  - the skin's motion is felt as the change in its velocity from one frame to the next, the inertia
//    a lobe answers; no gravity, since the skinned shape already hangs as it does at rest. A change
//    faster than a body can make (a pose preset, a jump of the clip) is felt only up to 5 g, so it
//    sways rather than kicks;
//  - each particle is sprung to its target, as soft as its lobe and its place in it: firm at the
//    lobe's rim (held there), loosest at its core;
//  - each cage edge keeps its two ends' offsets together, which keeps a lobe smooth and in one piece;
//  - each lobe keeps its volume against the skin (physics/volume.hpp), split at the midline, so one
//    breast pressed doesn't swell the other;
//  - an offset is at most 3 cm, and at most 0.5 cm into the body;
//  - what it meets while it moves (step 11, touch.hpp: props, people, its own limbs) is a plane per
//    particle for the frame, which it stays in front of. A particle pressed may go 3 cm into the
//    body, and its lobe bulges by the contact's share of the volume pushed in (target -(1 - beta) V
//    rather than 0), as the still contact pass's dent does.
// Damping acts on the velocity relative to the skin, so a lobe moving with its skin is neither damped
// nor slowed. Substeps are 1/480 s whatever the frame rate, the remainder carried.
//
// Once the skin has been still for a second the damping ramps up, and what presses the flesh lets go
// over a quarter of a second: by then the still pass's dent has faded in under it (the skin sinks
// into it, and a pressed particle keeps its place as it does). After three it is asleep: the offsets
// are 0 and nothing is written until the skin moves again.
//
// One thread (the figure's owner) calls everything.

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <functional>

#include <figure/physics/skin.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/physics/collide/triangle_set.hpp>
#include <figure/physics/rigid/world.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics {

// The lobes' parts, as the settings and the panel name them.
enum class FleshPart
{
  eChest,
  eGlutes,
  eBelly,
  eThighs,
};
inline constexpr int kFleshParts = 4;
const char*          fleshPartName(FleshPart part);

struct FleshSettings
{
  bool on = false;
  // per part, how soft: 0 holds it to the skin, 1 is loosest (its spring from 8 Hz down to 1.5)
  std::array<double, kFleshParts> softness{0.7, 0.6, 0.3, 0.4};
  // how soon it settles (the damping ratio)
  double damping = 0.3;

  bool operator==(const FleshSettings&) const = default;
};

// The lobes of a body, found once from its rest cage, its skin and its bones.
struct FleshModel
{
  std::vector<uint32_t> vertex;  // per particle, its cage vertex
  std::vector<float>    share;   // per particle, its place in its lobe: 1 the core, toward 0 the rim
  std::vector<double>   weight;  // per particle, its inverse mass: heavier where the cage is coarser
  std::vector<uint8_t>  lobe;    // per particle
  // cage edges between two particles; an edge to a vertex that isn't one holds that end at 0
  std::vector<std::array<uint32_t, 2>> edges;
  std::vector<uint32_t>                held;  // per edge to a held vertex, its particle
  struct Lobe
  {
    std::string           name;
    FleshPart             part = FleshPart::eChest;
    std::vector<uint32_t> particles;
  };
  std::vector<Lobe> lobes;
  // the cage vertices next to the particles that aren't any, and the triangles that touch a particle
  // (over the particles, then those): the skinned surface's areas and normals come from them
  std::vector<uint32_t> support;
  std::vector<uint32_t> tris;
  size_t                cageCount = 0;
};

// What the builder reads, in the cage's rest space (cm, y up, the front +z).
struct FleshInput
{
  std::span<const float>    cage;  // xyz per cage vertex
  std::span<const uint32_t> tris;
  std::span<const int32_t>  polyStart, polyVerts;  // the polygon loops, for the edges; empty: the triangles'
  const mesh::SkinInput*    skin = nullptr;        // per cage vertex
  // the skin's bones in skin-index order: name and rest centre
  std::vector<std::pair<std::string, std::array<double, 3>>> bones;
};
FleshModel buildFlesh(const FleshInput& in);

// A plane a particle stays in front of for a frame: through `point` along `normal` (unit, out of what
// it meets), less `allow` (cm) it may be in; in the targets' space.
struct FleshContact
{
  uint32_t              particle = 0;
  std::array<double, 3> point{}, normal{};
  double                allow = 0;
};

// The solve on a model, as the tests drive it.
class FleshSolver
{
public:
  explicit FleshSolver(FleshModel model);

  const FleshModel& model() const { return m_model; }
  void              set(const FleshSettings& settings) { m_settings = settings; }
  const FleshSettings& settings() const { return m_settings; }

  // A frame: the skinned targets as the frame ends (xyz per particle, then per support vertex),
  // `seconds` after the last. The first frame, or one whose targets jumped further than a body moves,
  // starts at rest; the frame after takes the skin's velocity as it finds it. `reshaped`: the skin
  // moved because its shape did (a dial), which moves the lobes with it and isn't felt.
  void step(std::span<const float> targets, double seconds, bool reshaped = false);
  // What the particles meet over the next step (it forgets them after), and the share of the volume
  // they push in that bulges around the dent.
  void setContacts(std::vector<FleshContact> contacts, double bulge);
  // Per particle: pressed in the last step (its contact moved it, or its skin is in what it meets).
  std::span<const uint8_t> pressed() const { return m_pressed; }
  // Starts again at rest on the next step.
  void reset() { m_started = false; }
  bool asleep() const { return m_asleep; }
  // Per particle, xyz: where it is against its target (cm). 0 asleep.
  std::span<const double> offsets() const { return m_d; }
  // The largest offset (cm).
  double largest() const;
  // Per particle, xyz: the skinned surface's area times its normal (cm^2), as the last frame made it.
  std::span<const double> areaNormals() const { return m_an; }
  // ... and its unit normal
  std::span<const double> normals() const { return m_normal; }

  // the substep, fixed (s)
  static constexpr double kSubstep = 1.0 / 480;

private:
  void   surface(std::span<const float> targets);
  void   substep(double h, double calm, double hold);
  double omega(size_t lobe) const;

  FleshModel          m_model;
  FleshSettings       m_settings;
  std::vector<double> m_d, m_e;  // offsets and their velocity, xyz per particle
  std::vector<float>  m_last;    // the last frame's targets
  std::vector<double> m_u;       // the skin's velocity over the last frame, xyz per particle
  std::vector<double> m_an;      // per particle, xyz: the skinned surface's area times its normal
  std::vector<double> m_normal;  // per particle, xyz: its unit normal
  std::vector<FleshContact> m_contacts;
  double                    m_bulge = 0.7;
  std::vector<uint8_t>      m_pressed;
  double              m_carry   = 0;
  double              m_still   = 0;  // how long the skin has been still (s)
  bool                m_started = false;
  bool                m_knowsU  = false;  // m_u is the skin's: a frame since the start
  bool                m_asleep  = true;
};

// A figure's soft flesh: the solver on its body, driven from the figure's pose.
// What the flesh meets, in the figure root's space (centimetres): touch.hpp.
struct FleshColliders
{
  // props
  std::vector<ColliderRef> sets;
  struct Limb
  {
    Capsule now;
    // this body's share of an overlap: another person's softness against its own (overlapShare)
    double share = 1;
    // the figure's own: its part of the body (contact.hpp's `part`) and its capsule at rest, in the
    // body cage's space; -1 for another person's
    int     part = -1;
    Capsule rest;
  };
  std::vector<Limb> limbs;
  // the share of the volume pushed in that bulges around the dent (ContactSettings::bulge)
  double bulge = 0.7;
};

class FleshTouch;

class Flesh
{
public:
  // `bodyMesh` is the figure's body; `root` its root node, the space what it meets is in.
  Flesh(const SceneGraph& scene, const morph::MorphSet& morphs, int bodyMesh, int root = -1);
  ~Flesh();

  // What it meets while it moves, in the root's space (touch.hpp); asked each frame it is awake.
  std::function<FleshColliders()> colliders;

  // Whether the body has lobes to move.
  bool                 valid() const { return m_solver && !m_solver->model().vertex.empty(); }
  int                  bodyMesh() const { return m_bodyMesh; }
  const FleshSettings& settings() const { return m_settings; }
  // Off, the offsets go to 0 at the next step.
  void set(const FleshSettings& settings);

  // A frame, once the pose is on the graph: `seconds` simulated. True when the displacement changed.
  bool step(double seconds);
  bool awake() const { return m_settings.on && m_solver && !m_solver->asleep(); }
  // Per cage vertex of the body, xyz: where the flesh is against its skin (cm, the mesh's posed space);
  // all 0 asleep or off.
  std::span<const float> displacement() const { return m_displacement; }
  // Bumps when the settings or awake() change, for a status topic.
  int version() const { return m_version; }

  const FleshSolver* solver() const { return m_solver.get(); }

private:
  const SceneGraph&            m_scene;
  const morph::MorphSet&       m_morphs;
  int                          m_bodyMesh, m_root;
  std::unique_ptr<FleshTouch>  m_touch;
  std::vector<float>           m_targets;  // last frame's
  FleshSettings                m_settings;
  std::unique_ptr<FleshSolver> m_solver;
  physics::CageSkin             m_skin;
  std::vector<float>           m_displacement;
  std::vector<float>           m_scratch;
  std::vector<float>           m_rest;  // the lobes' cage as the morphs had it last frame
  bool                         m_written  = false;  // the displacement isn't all 0
  bool                         m_wasAwake = false;
  int                          m_version  = 0;
};

}  // namespace fv::physics
