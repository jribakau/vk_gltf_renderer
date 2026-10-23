#include <figure/mesh/mesh_build.hpp>

#include <unordered_set>

namespace fv::mesh {

Drawn drawnTriangles(const BuiltMesh& mesh)
{
  if(mesh.hiddenPolys.empty())
    return {mesh.fullIndex, mesh.groups};
  std::unordered_set<int32_t> hidden;
  for(const auto& [owner, polys] : mesh.hiddenPolys)
    hidden.insert(polys.begin(), polys.end());
  Drawn out;
  out.index.reserve(mesh.fullIndex.size());
  for(const Group& g : mesh.groups)
  {
    Group kept = g;
    kept.start = uint32_t(out.index.size());
    for(uint32_t t = g.start; t < g.start + g.count; t += 3)
    {
      const size_t tri = t / 3;
      if(tri < mesh.triPoly.size() && hidden.count(int32_t(mesh.triPoly[tri])))
        continue;
      out.index.insert(out.index.end(), mesh.fullIndex.begin() + t, mesh.fullIndex.begin() + t + 3);
    }
    kept.count = uint32_t(out.index.size()) - kept.start;
    out.groups.push_back(kept);
  }
  return out;
}

}  // namespace fv::mesh
