#include <figure/physics/cloth/backend.hpp>

namespace fv::physics::cloth {

CpuBackend::CpuBackend(std::vector<Surface> meshes, const Options& opts)
    : m_island(std::move(meshes), opts)
{
}

CpuBackend::~CpuBackend() = default;

void CpuBackend::frame(float t0, float t1, float extraDamping, float seconds)
{
  m_island.frame(t0, t1, extraDamping, seconds);
}

void CpuBackend::setTarget(size_t mesh, const std::vector<Mat4>& bones)
{
  if(m_island.setTarget(mesh, bones))
    targetSet();
}

bool CpuBackend::setRest(size_t mesh, std::vector<float> restLocal)
{
  return m_island.setRest(mesh, std::move(restLocal));
}

bool CpuBackend::carry(float t)
{
  m_island.carry(t);
  return true;
}

size_t CpuBackend::tie(std::vector<Tie> ties)
{
  return m_island.setTies(std::move(ties));
}

void CpuBackend::setGravity(const std::array<float, 3>& gravity)
{
  m_island.setGravity(gravity);
}

void CpuBackend::settle()
{
  m_island.settle();
}

float CpuBackend::lastMove() const
{
  return m_island.lastMove();
}

int CpuBackend::recovered() const
{
  return m_island.recovered();
}

int CpuBackend::substepsPerFrame() const
{
  return m_island.substepsPerFrame();
}

float CpuBackend::transitionSeconds() const
{
  return m_island.transitionSeconds();
}

std::vector<std::vector<float>> CpuBackend::positions() const
{
  return m_island.positions();
}

const Model& CpuBackend::model() const
{
  return m_island.model();
}

}  // namespace fv::physics::cloth
