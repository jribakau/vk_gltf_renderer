#ifndef FV_LIGHT_TREE_H
#define FV_LIGHT_TREE_H

// The emitters' light tree (Conty Estevez and Kulla 2018, "Importance Sampling of Many Lights with
// Adaptive Tree Splitting"; PBRT-v4's BVHLightSampler). Each emitting surface is a leaf
// (EmissiveGroup), and the host builds a binary tree over them by place. Its cut, at most
// EMISSIVE_GROUP_MAX nodes, competes with the environment and the punctual lights in the light pick
// (lightPickSet) as single lights; a light sample that picks a cut node then walks down to a leaf,
// each step choosing a child by what it gives the shading point, and a BSDF ray that hits an emitter
// recovers that walk's probability by climbing from the leaf back to the cut. A scene with no more
// emitters than the cut has leaves alone and no walk.
//
// The node array is GltfScene::emissiveGroups, its cut first. Slang and C++
// (tests/test_light_tree.cpp), so the tests run the arithmetic the shader runs; needs
// light_pick.h.slang and EmissiveGroup (gltf_scene_io.h.slang) before it.

// What a node gives a point, as a punctual light's estimate does: its triangles' weight over the
// distance to its bounds' centre, no nearer than its radius, times the cosine at which the nearest of
// its normals can face the point (PBRT-v4's LightBounds::Importance): the angle from the cone's axis to
// the point, less the cone's spread and the angle its bounds subtend. 0 where none of its triangles
// can face the point, as a one-sided triangle emits to its front alone (a light sample there would be
// discarded); a point within its bounds is not judged. With `cull`, 0 for a node whose bounds lie
// wholly below the plane of `normal`.
float emissiveNodeEstimate(EmissiveGroup g, float3 pos, float3 normal, bool cull)
{
  const float3 toCenter = g.center - pos;
  if(cull && dot(normal, toCenter) <= -g.radius)
    return 0.0F;
  const float d2       = dot(toCenter, toCenter);
  const float r2       = g.radius * g.radius;
  const float estimate = g.weight / max(d2, max(r2, 1e-8F));
  if(g.cosSpread <= -1.0F || d2 <= r2)
    return estimate;
  // the angle from the axis to the point, past the spread
  const float cosW = dot(g.axis, -toCenter) / sqrt(d2);
  if(cosW >= g.cosSpread)
    return estimate;
  const float sinW = sqrt(max(1.0F - cosW * cosW, 0.0F));
  const float sinO = sqrt(max(1.0F - g.cosSpread * g.cosSpread, 0.0F));
  const float cosX = cosW * g.cosSpread + sinW * sinO;
  const float sinX = sinW * g.cosSpread - cosW * sinO;
  // less the angle the bounds subtend
  const float sin2B = r2 / d2;
  const float cosB  = sqrt(1.0F - sin2B);
  if(cosX >= cosB)
    return estimate;
  return estimate * max(cosX * cosB + sinX * sqrt(sin2B), 0.0F);
}

// The probability of the step from an inner node to its child of estimate `child`, its sibling's
// being `sibling`: by the estimates, with the light pick's defensive share between the two
// (lightPickProbability), 0 for a child that cannot light the point.
float emissiveChildProbability(float child, float sibling)
{
  const int candidates = (child > 0.0F ? 1 : 0) + (sibling > 0.0F ? 1 : 0);
  return lightPickProbability(child, child + sibling, candidates);
}

struct EmissiveTreeWalk
{
  int   leaf;  // -1: no leaf below can light the point
  float prob;  // the probability of the walk from where it started
};

// From node `node` down to a leaf at the uniform `u` in [0, 1), reused at each step.
EmissiveTreeWalk emissiveTreeDescend(EmissiveGroup* nodes, int node, float3 pos, float3 normal, bool cull, float u)
{
  EmissiveTreeWalk walk;
  walk.leaf = node;
  walk.prob = 1.0F;
  while(nodes[walk.leaf].left >= 0)
  {
    const int   left  = nodes[walk.leaf].left;
    const int   right = nodes[walk.leaf].right;
    const float eLeft = emissiveNodeEstimate(nodes[left], pos, normal, cull);
    const float eRight = emissiveNodeEstimate(nodes[right], pos, normal, cull);
    const float pLeft = emissiveChildProbability(eLeft, eRight);
    if(!(eLeft + eRight > 0.0F))
    {
      walk.leaf = -1;
      walk.prob = 0.0F;
      return walk;
    }
    if(u < pLeft)
    {
      walk.leaf = left;
      walk.prob *= pLeft;
      u = u / pLeft;
    }
    else
    {
      walk.leaf = right;
      walk.prob *= 1.0F - pLeft;
      u = (u - pLeft) / (1.0F - pLeft);
    }
    u = min(u, 0.99999994F);
  }
  return walk;
}

// The probability emissiveTreeDescend reaches `leaf` from the cut node above it, and that node
// (EmissiveTreeWalk::leaf holds it here).
EmissiveTreeWalk emissiveTreeClimb(EmissiveGroup* nodes, int leaf, float3 pos, float3 normal, bool cull)
{
  EmissiveTreeWalk walk;
  walk.leaf = leaf;
  walk.prob = 1.0F;
  while(nodes[walk.leaf].parent >= 0)
  {
    const int   parent  = nodes[walk.leaf].parent;
    const int   sibling = nodes[parent].left == walk.leaf ? nodes[parent].right : nodes[parent].left;
    const float e       = emissiveNodeEstimate(nodes[walk.leaf], pos, normal, cull);
    walk.prob *= emissiveChildProbability(e, emissiveNodeEstimate(nodes[sibling], pos, normal, cull));
    walk.leaf = parent;
  }
  return walk;
}

struct EmissiveTreeStep
{
  int   leaf;
  float u;  // what is left of the uniform, stretched back over [0, 1), for the leaf's triangle
};

// From node `node` down to a leaf by the nodes' weights alone, at the uniform `u` in [0, 1): with the
// cut picked by weight too, every triangle by its own weight over all of them, the one table's
// distribution (the pick past LIGHT_PICK_MAX_LIGHTS punctual lights, and without ePtLightTree).
EmissiveTreeStep emissiveTreeDescendByWeight(EmissiveGroup* nodes, int node, float u)
{
  EmissiveTreeStep step;
  step.leaf = node;
  step.u    = u;
  while(nodes[step.leaf].left >= 0)
  {
    const float wLeft = nodes[nodes[step.leaf].left].weight;
    const float total = wLeft + nodes[nodes[step.leaf].right].weight;
    const float pLeft = total > 0.0F ? wLeft / total : 0.5F;
    if(step.u < pLeft)
    {
      step.leaf = nodes[step.leaf].left;
      step.u    = step.u / pLeft;
    }
    else
    {
      step.leaf = nodes[step.leaf].right;
      step.u    = (step.u - pLeft) / (1.0F - pLeft);
    }
    step.u = min(step.u, 0.99999994F);
  }
  return step;
}

#endif
