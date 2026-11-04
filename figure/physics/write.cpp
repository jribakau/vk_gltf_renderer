#include <figure/physics/write.hpp>

#include <cmath>
#include <memory>

namespace fv::physics {

bool ChannelWriter::write(morph::MorphSet& morphs, const mesh::BuiltMesh& geo, std::span<const float> delta) const
{
  const size_t     n = geo.cagePositions.size() / 3;
  mesh::CageDeltas cage;
  for(size_t v = 0; v < n && v * 3 + 2 < delta.size(); v++)
  {
    const float x = delta[v * 3], y = delta[v * 3 + 1], z = delta[v * 3 + 2];
    if(std::fabs(x) + std::fabs(y) + std::fabs(z) <= kMinMove)
      continue;
    cage.verts.push_back(int32_t(v));
    cage.deltas.insert(cage.deltas.end(), {x, y, z});
  }
  const dson::TargetPtr had = morphs.cpuTarget(name, mesh);
  const bool            on  = morphs.has(name) && morphs.get(name) == 1.0;
  if(!had && cage.verts.empty())
    return false;
  // the same as the channel holds, within what isn't worth a morph: nothing to write
  if(had && on)
  {
    std::vector<float> old(n * 3, 0.f);
    for(size_t k = 0; k < had->cageIndices.size(); k++)
      if(had->cageIndices[k] < n)
        for(int a = 0; a < 3; a++)
          old[had->cageIndices[k] * 3 + a] = had->cageDeltas[k * 3 + a];
    std::vector<float> now(n * 3, 0.f);
    for(size_t k = 0; k < cage.verts.size(); k++)
      for(int a = 0; a < 3; a++)
        now[size_t(cage.verts[k]) * 3 + a] = cage.deltas[k * 3 + a];
    bool same = true;
    for(size_t v = 0; v < n && same; v++)
      same = std::fabs(now[v * 3] - old[v * 3]) + std::fabs(now[v * 3 + 1] - old[v * 3 + 1]) + std::fabs(now[v * 3 + 2] - old[v * 3 + 2]) <= kMinMove;
    if(same)
      return false;
  }
  auto target = std::make_shared<const mesh::RenderTarget>(cage.verts.empty() ? mesh::RenderTarget{} : mesh::target(geo, cage));
  morph::ChannelOptions opts;
  opts.hidden = true;
  opts.label  = label;
  opts.value  = 1.0;
  opts.min    = 0.0;
  opts.max    = 1.0;
  morphs.replaceCpuTarget(name, mesh, std::move(target), opts);
  // a channel turned off earlier keeps its weight: back to 1
  if(morphs.get(name) != 1.0)
  {
    morphs.set(name, 1.0);
    morphs.flush();
  }
  return true;
}

bool ChannelWriter::off(morph::MorphSet& morphs) const
{
  if(!morphs.has(name) || morphs.get(name) == 0.0)
    return false;
  morphs.set(name, 0.0);
  return true;
}

}  // namespace fv::physics
