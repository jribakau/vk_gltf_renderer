#include <figure/scene/graph.hpp>

#include <algorithm>
#include <cmath>

namespace fv {

int SceneGraph::add(Node node, int parent, int index)
{
  const int id = int(nodes.size());
  node.parent  = -1;
  node.children.clear();
  nodes.push_back(std::move(node));
  if(parent >= 0)
    attach(id, parent, index);
  return id;
}

void SceneGraph::attach(int node, int parent, int index)
{
  Node& n = nodes[node];
  if(n.parent >= 0)
  {
    auto& siblings = nodes[n.parent].children;
    siblings.erase(std::remove(siblings.begin(), siblings.end(), node), siblings.end());
  }
  n.parent = parent;
  if(parent < 0)
    return;
  auto& children = nodes[parent].children;
  if(index < 0 || index >= int(children.size()))
    children.push_back(node);
  else
    children.insert(children.begin() + index, node);
}

std::vector<int> SceneGraph::roots() const
{
  std::vector<int> out;
  for(int i = 0; i < int(nodes.size()); i++)
    if(nodes[i].parent < 0)
      out.push_back(i);
  return out;
}

Mat4 SceneGraph::local(int node) const
{
  const Node& n = nodes[node];
  return compose(n.position, n.quaternion, n.scale);
}

Mat4 SceneGraph::world(int node) const
{
  const Node& n = nodes[node];
  return n.parent < 0 ? local(node) : world(n.parent) * local(node);
}

std::vector<Mat4> SceneGraph::worlds() const
{
  std::vector<Mat4> out(nodes.size());
  std::vector<int>  stack = roots();
  std::reverse(stack.begin(), stack.end());
  while(!stack.empty())
  {
    const int i = stack.back();
    stack.pop_back();
    const Node& n = nodes[i];
    out[i]        = n.parent < 0 ? local(i) : out[n.parent] * local(i);
    for(auto it = n.children.rbegin(); it != n.children.rend(); ++it)
      stack.push_back(*it);
  }
  return out;
}

void measureUvScales(SceneGraph& scene)
{
  std::vector<double> area(scene.materials.size(), 0.0), uvArea(scene.materials.size(), 0.0);
  for(const Mesh& m : scene.meshes)
  {
    if(!m.geo || m.geo->uv.empty())
      continue;
    const mesh::BuiltMesh& g = *m.geo;
    for(const mesh::Group& group : g.groups)
    {
      const int mi = group.materialIndex >= 0 && group.materialIndex < int(m.materials.size()) ? m.materials[size_t(group.materialIndex)] : -1;
      if(mi < 0 || mi >= int(scene.materials.size()) || scene.materials[size_t(mi)].metresPerUv > 0)
        continue;
      for(uint32_t t = group.start; t + 2 < group.start + group.count && t + 2 < g.fullIndex.size(); t += 3)
      {
        const uint32_t a = g.fullIndex[t], b = g.fullIndex[t + 1], c = g.fullIndex[t + 2];
        const auto     p = [&](uint32_t v, int k) { return double(g.position[size_t(v) * 3 + size_t(k)]); };
        const auto     q = [&](uint32_t v, int k) { return double(g.uv[size_t(v) * 2 + size_t(k)]); };
        const double   e1[3] = {p(b, 0) - p(a, 0), p(b, 1) - p(a, 1), p(b, 2) - p(a, 2)};
        const double   e2[3] = {p(c, 0) - p(a, 0), p(c, 1) - p(a, 1), p(c, 2) - p(a, 2)};
        const double   n[3]  = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        area[size_t(mi)] += 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        uvArea[size_t(mi)] += 0.5 * std::abs((q(b, 0) - q(a, 0)) * (q(c, 1) - q(a, 1)) - (q(c, 0) - q(a, 0)) * (q(b, 1) - q(a, 1)));
      }
    }
  }
  for(size_t mi = 0; mi < scene.materials.size(); mi++)
  {
    if(scene.materials[mi].metresPerUv > 0 || !(area[mi] > 0) || !(uvArea[mi] > 0))
      continue;
    const double metres = 0.01 * std::sqrt(area[mi] / uvArea[mi]);
    // two significant figures, so a texture asked for before and after a small change is one
    const double step              = std::pow(10.0, std::floor(std::log10(metres)) - 1);
    scene.materials[mi].metresPerUv = float(std::round(metres / step) * step);
  }
}

bool takenOff(const SceneGraph& scene, int node)
{
  for(; node >= 0 && node < int(scene.nodes.size()); node = scene.nodes[size_t(node)].parent)
    if(scene.nodes[size_t(node)].removed)
      return true;
  return false;
}

std::vector<bool> liveMeshes(const SceneGraph& scene)
{
  std::vector<int> users(scene.meshes.size(), 0), live(scene.meshes.size(), 0);
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    const int mesh = scene.nodes[i].mesh;
    if(mesh < 0 || mesh >= int(scene.meshes.size()))
      continue;
    users[size_t(mesh)]++;
    live[size_t(mesh)] += takenOff(scene, int(i)) ? 0 : 1;
  }
  std::vector<bool> out(scene.meshes.size());
  for(size_t m = 0; m < out.size(); m++)
    out[m] = users[m] == 0 || live[m] > 0;
  return out;
}

}  // namespace fv
