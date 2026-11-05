#include <figure/dforce/solver.hpp>

namespace fv::dforce {

ClothSim::ClothSim(std::vector<SettleMesh> meshes, const SettleOptions& opts)
    : m_island(std::move(meshes), opts)
{
}

ClothSim::~ClothSim() = default;

void ClothSim::frame(float t0, float t1, float extraDamping, float seconds)
{
  m_island.frame(t0, t1, extraDamping, seconds);
}

void ClothSim::setTarget(size_t mesh, const std::vector<Mat4>& bones)
{
  if(m_island.setTarget(mesh, bones))
    targetSet();
}

void ClothSim::setGravity(const std::array<float, 3>& gravity)
{
  m_island.setGravity(gravity);
}

void ClothSim::settle()
{
  m_island.settle();
}

float ClothSim::lastMove() const
{
  return m_island.lastMove();
}

int ClothSim::recovered() const
{
  return m_island.recovered();
}

int ClothSim::substepsPerFrame() const
{
  return m_island.substepsPerFrame();
}

float ClothSim::transitionSeconds() const
{
  return m_island.transitionSeconds();
}

std::vector<std::vector<float>> ClothSim::positions() const
{
  return m_island.positions();
}

const ClothModel& ClothSim::model() const
{
  return m_island.model();
}

}  // namespace fv::dforce
