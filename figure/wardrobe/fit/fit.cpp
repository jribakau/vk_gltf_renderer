#include <figure/wardrobe/fit/fit.hpp>

#include <algorithm>

#include <figure/mesh/normals.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/bases.hpp>
#include <figure/wardrobe/fit/collide.hpp>
#include <figure/wardrobe/fit/regions.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace fv::wardrobe {

namespace {

// Bones of the rigs the item lists, by name, with the parent that is a bone.
std::vector<BoneRef> boneRefs(const SceneGraph& scene, const dson::LoadResult& w, const dson::Item& item)
{
  std::vector<BoneRef> refs;
  for(int r : item.rigs)
    for(const auto& [id, node] : w.rigs[size_t(r)].bones)
    {
      BoneRef     ref;
      const Node& n = scene.nodes[size_t(node)];
      ref.name      = n.name;
      if(n.parent >= 0 && scene.nodes[size_t(n.parent)].kind == NodeKind::eBone)
        ref.parent = scene.nodes[size_t(n.parent)].name;
      refs.push_back(std::move(ref));
    }
  return refs;
}

// Bones at the centres their files give: the space the unmorphed meshes are in.
std::vector<SkelBone> skeletonOf(const SceneGraph& scene, const std::vector<const dson::Rig*>& rigs)
{
  std::vector<SkelBone> out;
  for(const dson::Rig* rig : rigs)
    for(const auto& [id, node] : rig->bones)
    {
      const Node& n = scene.nodes[size_t(node)];
      if(!n.joint)
        continue;
      const auto& c = n.joint->centerAuthored != std::array<double, 3>{} ? n.joint->centerAuthored : n.joint->center;
      SkelBone    bone;
      bone.name = n.name;
      if(n.parent >= 0 && scene.nodes[size_t(n.parent)].kind == NodeKind::eBone)
        bone.parent = scene.nodes[size_t(n.parent)].name;
      bone.center = {c[0], c[1], c[2]};
      out.push_back(std::move(bone));
    }
  return out;
}

std::vector<int16_t> limbsOf(const SceneGraph& scene, const Mesh& mesh)
{
  if(mesh.cageSkin.start.empty())
    return std::vector<int16_t>(size_t(mesh.geo->dsonVertexCount), 0);
  std::vector<int16_t> boneLimb;
  for(int b : mesh.bones)
    boneLimb.push_back(limbOfBone(scene.nodes[size_t(b)].name));
  return vertexLimbs(size_t(mesh.geo->dsonVertexCount), &mesh.cageSkin, boneLimb);
}

// Replaces a mesh's cage and moves the drawn vertices with it (morphs have not been applied to a
// wearable yet, so there is nothing to carry along).
void writeCage(mesh::BuiltMesh& geo, const std::vector<float>& positions)
{
  if(positions.size() != geo.cagePositions.size())
    return;
  const std::vector<float> delta = mesh::shiftCage(geo, positions);
  for(size_t i = 0; i < geo.position.size() && i < delta.size(); i++)
    geo.position[i] += delta[i];
  mesh::weldNormals(geo.position, geo.renderIndex, geo.firstSplit, geo.renderTris, geo.renderVertexCount, geo.normal);
}

// Pushes what ended just under the skin back out, smoothing the push over the garment.
std::vector<float> collide(const std::vector<float>& positions, std::span<const uint32_t> tris, const Surface& body)
{
  const SurfaceBinding binding = bindToSurface(body.positions, body.tris, positions);
  return smoothCorrection(positions, pushOut(positions, body, binding, 0.15f), tris);
}

}  // namespace

std::optional<std::pair<Generation, Generation>> generationsOf(const dson::LoadResult& wearable, const dson::Item& item, const std::string& rel,
                                                               const FitFigure& figure)
{
  std::vector<std::string> figureNames, itemNames;
  for(const auto& [id, node] : figure.rig.bones)
    figureNames.push_back(figure.scene.nodes[size_t(node)].name);
  for(const BoneRef& b : boneRefs(wearable.scene, wearable, item))
    itemNames.push_back(b.name);
  const auto source = identifyGeneration(rel, itemNames, item.conformTarget);
  const auto target = identifyGeneration(figure.folder, figureNames, {});
  if(source && target && !sameRig(*source, *target))
    return std::make_pair(*source, *target);
  return std::nullopt;
}

std::optional<FitDetail> fitItem(dson::Store& store, dson::LoadResult& w, const dson::Item& item, const std::string& rel, const FitFigure& figure)
{
  const auto gens = generationsOf(w, item, rel, figure);
  if(!gens)
    return std::nullopt;
  const auto [source, target] = *gens;
  std::vector<std::string> figureNames;
  for(const auto& [id, node] : figure.rig.bones)
    figureNames.push_back(figure.scene.nodes[size_t(node)].name);
  const std::vector<BoneRef> itemBones = boneRefs(w.scene, w, item);
  FitDetail                  detail;
  detail.source = source;
  detail.target = target;
  detail.plan   = planLeaders(itemBones, figureNames);
  const std::vector<SkelBone> figureSkeleton = skeletonOf(figure.scene, {&figure.rig});
  std::vector<const dson::Rig*> itemRigs;
  for(int r : item.rigs)
    itemRigs.push_back(&w.rigs[size_t(r)]);
  const std::map<std::string, Rigid> rigid = rigidPlan(skeletonOf(w.scene, itemRigs), figureSkeleton);

  const Mesh&            bodyMesh = figure.scene.meshes[size_t(figure.mesh)];
  const mesh::BuiltMesh& body     = *bodyMesh.geo;
  const Surface          bodySurface{body.cagePositions, body.dsonTris};
  std::vector<std::string> boneNames;
  for(const BoneRef& b : itemBones)
    boneNames.push_back(b.name);
  const FitRegion region = regionOf(rel + "/" + item.label, boneNames);

  // the source base, carried through the same bone transfer, is what the garment was shaped around
  const auto cage = loadBaseCage(store, source, genderHint(rel));
  std::vector<float>       baseMoved;
  std::vector<int16_t>     baseLimbs;
  std::optional<Surface>   base;
  if(cage)
  {
    const auto plan = rigidPlan(cage->skeleton, figureSkeleton);
    std::vector<const Rigid*> perBone;
    std::vector<int16_t>      boneLimb;
    for(const SkelBone& b : cage->skeleton)
    {
      const auto it = plan.find(b.name);
      perBone.push_back(it == plan.end() ? nullptr : &it->second);
      boneLimb.push_back(limbOfBone(b.name));
    }
    baseMoved = rigidPositions(cage->positions, cage->skin, perBone);
    baseLimbs = vertexLimbs(baseMoved.size() / 3, &cage->skin, boneLimb);
    base      = Surface{baseMoved, cage->tris};
  }
  const std::vector<int16_t> bodyLimbs = limbsOf(figure.scene, bodyMesh);
  const bool footwear = region == FitRegion::eShoe || region == FitRegion::eBoot || region == FitRegion::eBootThigh;
  // the base-to-target search does not depend on the garment; footwear never asks for it
  std::optional<SurfaceBinding> pair;
  if(!footwear && base && cage && !sameTopology(source, target))
  {
    const auto filter = limbFilter(bodyLimbs, baseLimbs);
    pair              = bindToSurface(bodySurface.positions, bodySurface.tris, base->positions, {}, filter ? &*filter : nullptr);
  }
  // footwear skips the shape correction, so the foot's proportions come from a box fit
  std::vector<std::pair<int16_t, LimbBox>> feet;
  if(footwear && base)
    for(const auto& [side, l] : {std::pair{'l', limb::kLFoot}, std::pair{'r', limb::kRFoot}})
    {
      const SkelBone* foot = findSlot(figureSkeleton, side, "foot");
      const SkelBone* toes = findSlot(figureSkeleton, side, "toes");
      if(!foot || !toes)
        continue;
      const V3 dir{toes->center[0] - foot->center[0], toes->center[1] - foot->center[1], toes->center[2] - foot->center[2]};
      const int16_t limbId = l;
      if(auto box = limbBox(dir, foot->center, base->positions, [&](size_t v) { return baseLimbs[v] == limbId; }, body.cagePositions,
                            [&](size_t v) { return bodyLimbs[v] == limbId; }))
        feet.emplace_back(limbId, *box);
    }

  for(int meshIdx : item.meshes)
  {
    Mesh& m = w.scene.meshes[size_t(meshIdx)];
    if(m.cageSkin.start.empty() || !m.geo)
      continue;
    const mesh::SkinInput& skin = m.cageSkin;
    std::vector<const Rigid*> perBone;
    for(int b : m.bones)
    {
      const auto it = rigid.find(w.scene.nodes[size_t(b)].name);
      perBone.push_back(it == rigid.end() ? nullptr : &it->second);
    }
    std::vector<float> positions = rigidPositions(m.geo->cagePositions, skin, perBone);
    for(const auto& [l, box] : feet)
    {
      std::vector<bool> on;
      for(int b : m.bones)
        on.push_back(limbOfBone(w.scene.nodes[size_t(b)].name) == l);
      positions = applyLimbBox(box, positions, [&](size_t v) {
        double wsum = 0;
        for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
          if(on[size_t(skin.bone[size_t(i)])])
            wsum += skin.weight[size_t(i)];
        return std::min(1.0, wsum);
      });
    }
    // footwear keeps the bone transfer and the foot box: a heel's rest is often its posed shape (the
    // foot already pointed), which a correction measured against the flat base would bend
    if(!footwear)
    {
      if(base)
      {
        const std::vector<int16_t> itemLimbs = limbsOf(w.scene, m);
        const auto baseFilter = limbFilter(bodyLimbs, baseLimbs);
        const auto itemFilter = limbFilter(baseLimbs, itemLimbs);
        ShapeInputs in;
        in.sameTopology = sameTopology(source, target);
        in.baseFilter   = baseFilter ? &*baseFilter : nullptr;
        in.itemFilter   = itemFilter ? &*itemFilter : nullptr;
        in.baseToTarget = pair ? &*pair : nullptr;
        positions       = shapeCorrection(positions, m.geo->dsonTris, *base, bodySurface, in);
      }
      // (a dynamic surface owns its collision; dForce is not ported yet, so every surface is pushed out)
      positions = collide(positions, m.geo->dsonTris, bodySurface);
    }
    writeCage(*m.geo, positions);
  }
  detail.shaped = base.has_value();
  detail.note   = std::string(label(source)) + " outfit on " + label(target) + ", "
                + (base ? "refitted to its skeleton and shape" :
                          std::string("refitted to its skeleton (the ") + label(source) + " base figure isn't installed, so its shape wasn't matched)")
                + ".";
  return detail;
}

}  // namespace fv::wardrobe
