#pragma once

// dForce's cloth solver, which moved onto the physics core (docs/physics-plan.md step 12): the cloth
// is a particle island (figure/physics/cloth/island.hpp, where the solver is described) built from
// surfaces (figure/physics/cloth/surface.hpp). What stays here is the seam the dForce driver runs a
// simulation through, the CPU island or the renderer's GPU solver, and the old names, until dForce is
// retired (step 15).

#include <array>
#include <cstddef>
#include <vector>

#include <figure/physics/cloth/island.hpp>

namespace fv::dforce {

using SettleMesh    = physics::cloth::Surface;
using SettleOptions = physics::cloth::Options;
using ClothModel    = physics::cloth::Model;
using Edge          = physics::cloth::Distance;
using EdgeKind      = physics::cloth::DistanceKind;
using Tri           = physics::cloth::Face;
using physics::BonePose;
using physics::poseOf;
using physics::sweepAt;
using physics::cloth::holdsLength;
using physics::cloth::kBodyLayer;
using physics::cloth::kPlainLayer;
using physics::cloth::separateStart;
using physics::cloth::settle;

inline void skinMesh(const SettleMesh& m, const std::vector<Mat4>& bones, float* out)
{
  physics::cloth::skin(m, bones, out);
}

// A simulation, however it runs: the CPU island below, or a GPU one the renderer provides.
class ClothBackend
{
public:
  virtual ~ClothBackend() = default;
  // `seconds` of simulated time; 0: SettleOptions::frameDt
  virtual void  frame(float t0, float t1, float extraDamping = 0, float seconds = 0) = 0;
  virtual void  setTarget(size_t mesh, const std::vector<Mat4>& bones) = 0;
  // Which way is down and how hard, cm/s^2 in the job's space: the figure was turned.
  virtual void  setGravity(const std::array<float, 3>& gravity) = 0;
  virtual void  settle() = 0;
  // A settle in slices, for a simulation on the thread that draws: frames until `seconds` are used, at
  // least one. True once it is done.
  virtual bool  settleSome(double /*seconds*/)
  {
    settle();
    return true;
  }
  // It stopped and what it has isn't to be used (a position that isn't a number).
  virtual bool  failed() const { return false; }
  virtual float lastMove() const = 0;
  virtual std::vector<std::vector<float>> positions() const = 0;
  // True when it must be driven on the thread that made it (a GPU queue), never on the pool.
  virtual bool ownerThreadOnly() const { return false; }
  // Live: one frame. After a setTarget the pose sweeps from the one it reached to the new target
  // (0 to 1); otherwise it holds where the last sweep ended (1), as a settle's hold frames do. A
  // sweep from 0 on every frame would set the colliders and the pinned cloth out from the pose
  // they memorized (the rest pose, after a settle) each frame and drag them through the cloth to
  // the target again.
  void step(float extraDamping = 0, float seconds = 0)
  {
    frame(m_targetPending ? 0.f : 1.f, 1, extraDamping, seconds);
    m_targetPending = false;
  }

protected:
  // For setTarget: the next step sweeps.
  void targetSet() { m_targetPending = true; }

private:
  bool m_targetPending = false;
};

// The CPU simulation: the cloth island behind the seam.
class ClothSim final : public ClothBackend
{
public:
  ClothSim(std::vector<SettleMesh> meshes, const SettleOptions& opts = {});
  ~ClothSim() override;
  ClothSim(const ClothSim&)            = delete;
  ClothSim& operator=(const ClothSim&) = delete;

  // As physics::cloth::Island's.
  void  frame(float t0, float t1, float extraDamping = 0, float seconds = 0) override;
  void  setTarget(size_t mesh, const std::vector<Mat4>& bones) override;
  void  setGravity(const std::array<float, 3>& gravity) override;
  void  settle() override;
  float lastMove() const override;
  int   recovered() const;
  int   substepsPerFrame() const;
  float transitionSeconds() const;
  std::vector<std::vector<float>> positions() const override;
  const ClothModel&               model() const;

private:
  physics::cloth::Island m_island;
};

}  // namespace fv::dforce
