#pragma once

// A cloth island (docs/physics-plan.md step 12): a position-based settle of cloth surfaces, on the
// particle core (physics/particles). One island holds every dynamic mesh and every collider of a job:
// collision layers are sided constraints, not separate solves. dForce's ClothSim, moved here.
//
// The cloth is simulated in frames of `frameDt`, each cut into substeps with one constraint sweep
// per substep (small steps converge where many sweeps over a large step do not). Masses come from
// each surface's density and the area around the vertex, so the stiffness numbers are compliances in
// physical units and a heavy fabric hangs differently from a light one. Colliders and pinned vertices
// sweep from the memorized pose to the target pose along their bones (a limb turns instead of
// shrinking along a chord), and the side a vertex is on is taken from where it was, so a limb
// sweeping through cloth pushes it out the side it started on. Pinning does not scale with the
// step count: strength 0 is infinite mass, and a partial strength is a tether to the skinned
// position solved with the edges every substep, as compliant as the strength makes it (a skirt's
// band fades from pinned to free over a few rows without tearing).
//
// A sweep moves a correction one edge along, so what holds a garment up (a strap on a shoulder, a
// band on the hips) wouldn't reach the cloth hanging from it within a substep, and the cloth would
// stretch until it slid off. Reach links carry it: one-sided constraints between vertices some
// centimetres apart over the surface, no longer than the way between them along the mesh's threads,
// as compliant as the softest edge on that way. Only stretching meets them, so cloth folds and gathers as it did. Only the
// stretch sweeps take them, not the sweep after a collision pass: drawn straight, a link would pull
// what contact has just pushed out back through a limb.
//
// Because the side is kept, a vertex that starts inside a lower layer (a fitted shape overlapping
// the body) would stay inside; those are moved out along the nearest face before the first step.
//
// `Island` is the persistent state: `settle()` runs the pose transition on it and then holds the
// pose until the cloth is still, and a live simulation gives it the pose the figure has now every
// frame and steps it once, so the cloth follows a drag.

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include <figure/physics/cloth/model.hpp>
#include <figure/physics/sweep.hpp>

namespace fv::physics::cloth {

// How a cloth island steps and settles. dForce's SettleOptions, moved here unchanged.
struct Options
{
  // Air resistance, DAZ's 0.15
  float air = 0.15f;
  // One frame of simulated time, seconds
  float frameDt = 1.f / 30.f;
  // Substeps per frame; 0 picks them from the motion (a collider moves at most half a gap per substep)
  int substeps = 0;
  // Constraint sweeps per substep
  int iterations = 1;
  // Further sweeps per substep over the stretch edges alone: one sweep leaves a hanging cloth's
  // rows stretched where they carry what hangs below (twice their length under a real skirt's
  // band), and the stretch edges are a third of the constraints, so these converge it at a third
  // of a sweep's cost each. Five bring that skirt's worst edge to 1.3x and its 99th centile to 1.07.
  int stretchSweeps = 5;
  // Long one-sided links along the cloth (see the top of this file); off only to compare against.
  bool reachLinks = true;
  // Seconds the pose transition takes; 0 picks them from how far the colliders move
  float transition = 0;
  // Seconds the cloth may take to come to rest once the pose is reached. It rarely does: a vertex caught
  // in a fold keeps moving. On the Kai Dress 1 s lands within 0.6 cm mean of 3 s, the hem under a
  // centimetre short, and from there it only creeps.
  float maxHold = 1.f;
  // Every frame, a free vertex touching nothing that lies behind the nearest face of a lower layer
  // within reach is put back outside it: cloth pulled through a limb by its edges between two
  // collision passes, or carried through by the sweep, would otherwise stay inside for good
  bool recoverLost = true;
  // Asked once a frame: true ends the settle where it is, as an edit has made it out of date.
  std::function<bool()> cancelled;
  // The largest move of a vertex over a frame, in cm, under which the cloth counts as still
  float restMove = 0.01f;
};

// The simulation state: rows built once from the rest shapes, then advanced a frame at a time.
class Island
{
public:
  Island(std::vector<Surface> surfaces, const Options& opts = {});
  ~Island();
  Island(const Island&)            = delete;
  Island& operator=(const Island&) = delete;

  // Advances one frame: the colliders and pinned vertices sweep from pose fraction `t0` to `t1`
  // (0 the pose they set out from, 1 the target) while the cloth integrates. `extraDamping` is a
  // further velocity decay per second, for bringing a held pose to rest quickly. The frame is
  // `seconds` long, cut into the same substeps; 0: Options::frameDt.
  void frame(float t0, float t1, float extraDamping = 0, float seconds = 0);
  // Live: a new target for a skinned surface's bones (its skin matrices in the mesh's space, as
  // `Surface::bones`). The pose its kinematic vertices have reached becomes where they set out
  // from, so the next frame(0, 1) sweeps them to the new target. Surfaces not given keep theirs. False
  // when this surface isn't skinned with as many bones, and nothing changed.
  bool setTarget(size_t surface, const std::vector<Mat4>& bones);
  // Live: a skinned surface's rest cage moved in its mesh's space (`Surface::restLocal`) by what the
  // pose drives, its correctives, or by a dent: what its kinematic vertices are skinned from from the
  // next frame. The rows keep the lengths they were built with. False when this surface isn't skinned
  // or the size differs, and nothing changed.
  bool setRest(size_t surface, std::vector<float> restLocal);
  // Which way is down and how hard, cm/s^2 in the job's space.
  void setGravity(const std::array<float, 3>& gravity);
  // The pose transition over its frames, then the hold until the cloth is still (or `maxHold`).
  void settle();
  // How far the fastest dynamic vertex moved over the last frame, cm.
  float lastMove() const;
  // Vertices the last frame found lost inside a lower layer and put back (Options::recoverLost).
  int recovered() const;
  // Substeps a frame takes and the seconds the transition takes, as the options or the motion set them.
  int   substepsPerFrame() const;
  float transitionSeconds() const;
  // Positions for every surface, in the same order and space as `rest` / `posed`.
  std::vector<std::vector<float>> positions() const;
  const Model&                    model() const;

private:
  struct State;
  std::unique_ptr<State> m_state;
};

// Simulated positions for every surface after the transition and the hold, in the same order and space
// as `rest` / `posed`.
std::vector<std::vector<float>> settle(const std::vector<Surface>& surfaces, const Options& opts = {});

}  // namespace fv::physics::cloth
