#include <figure/scene/graph.hpp>

#include <algorithm>

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

}  // namespace fv
