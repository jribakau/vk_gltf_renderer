#include <figure/wardrobe/fit/fit.hpp>

#include <algorithm>

#include <figure/garment/convert.hpp>
#include <figure/garment/decode.hpp>
#include <figure/garment/graft.hpp>
#include <figure/garment/maps.hpp>
#include <figure/garment/pieces.hpp>
#include <figure/mesh/normals.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/bind_cache.hpp>
#include <figure/wardrobe/fit/bases.hpp>
#include <figure/wardrobe/fit/collide.hpp>
#include <figure/wardrobe/fit/regions.hpp>
#include <figure/wardrobe/fit/transfer.hpp>
#include <figure/wardrobe/follow.hpp>

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

// The generations the item and the figure can be told to be of.
std::pair<std::optional<Generation>, std::optional<Generation>> identified(const dson::LoadResult& wearable, const dson::Item& item, const std::string& rel,
                                                                           const FitFigure& figure)
{
  std::vector<std::string> figureNames, itemNames;
  for(const auto& [id, node] : figure.rig.bones)
    figureNames.push_back(figure.scene.nodes[size_t(node)].name);
  for(const BoneRef& b : boneRefs(wearable.scene, wearable, item))
    itemNames.push_back(b.name);
  return {identifyGeneration(rel, itemNames, item.conformTarget, item.madeFor), identifyGeneration(figure.folder, figureNames, {})};
}

// The geograft a mesh of the wearable is, or null.
const Graft* graftOf(const dson::LoadResult& w, int mesh)
{
  for(const dson::ItemPart& part : w.parts)
    if(part.mesh == mesh && part.graft)
      return &*part.graft;
  return nullptr;
}

// Whose base an item's geograft was made on, where it says by its vertex count and the path doesn't
// ("Genesis 8 Male Genitalia For G8F" is the woman's): the gender whose base has that count.
std::optional<std::string> graftGender(dson::Store& store, const dson::LoadResult& w, const dson::Item& item, Generation source)
{
  for(int m : item.meshes)
    if(const Graft* g = graftOf(w, m); g && g->vertexCount >= 0)
      for(const char* gender : {"female", "male"})
        if(const auto base = loadBaseCage(store, source, gender); base && base->positions.size() == size_t(g->vertexCount) * 3)
          return std::string(gender);
  return std::nullopt;
}

}  // namespace

std::optional<std::pair<Generation, Generation>> generationsOf(const dson::LoadResult& wearable, const dson::Item& item, const std::string& rel,
                                                               const FitFigure& figure)
{
  const auto [source, target] = identified(wearable, item, rel, figure);
  if(source && target && !sameRig(*source, *target))
    return std::make_pair(*source, *target);
  return std::nullopt;
}

bool madeForFigure(const dson::LoadResult& wearable, const dson::Item& item, const std::string& rel, const FitFigure& figure)
{
  const auto [source, target] = identified(wearable, item, rel, figure);
  return source && target && sameRig(*source, *target);
}

std::vector<std::optional<V3>> mountsFrom(const dson::Item& item, const std::vector<SkelBone>* base)
{
  std::vector<std::optional<V3>> out;
  for(const auto& [node, bone] : item.objects)
  {
    std::optional<V3> from;
    if(!bone.empty() && base)
      for(const SkelBone& b : *base)
        if(b.name == bone)
        {
          from = b.center;
          break;
        }
    out.push_back(from);
  }
  return out;
}

std::shared_ptr<const BaseCage> wholeBase(const dson::LoadResult& w)
{
  auto out = std::make_shared<BaseCage>(*w.sourceWhole);
  if(w.sourceWholePreset.size() == out->positions.size())
    out->positions = w.sourceWholePreset;
  for(float& p : out->positions)
    p = float(double(p) * w.sourceScale);
  for(SkelBone& b : out->skeleton)
    for(double& c : b.center)
      c *= w.sourceScale;
  return out;
}

std::optional<FitDetail> fitItem(dson::Store& store, dson::LoadResult& w, const dson::Item& item, const std::string& rel, const FitFigure& figure,
                                 FitPath path)
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
  // the source base, carried through the same bone transfer, is what the garment was shaped around:
  // the body that came with it (a Skyrim outfit's), else its generation's from the library. The common
  // fit takes a Skyrim outfit's whole body where one is on this machine: the body it carries is that
  // body cut to what shows, and a piece may carry none
  const bool        viaWhole     = path == FitPath::eCommon && w.sourceWhole;
  const std::string sourceGender = graftGender(store, w, item, source).value_or(genderHint(rel));
  const auto        cage         = viaWhole ? wholeBase(w) : w.sourceBase ? w.sourceBase : loadBaseCage(store, source, sourceGender);
  std::vector<const dson::Rig*> itemRigs;
  for(int r : item.rigs)
    itemRigs.push_back(&w.rigs[size_t(r)]);
  // a pruned item's limbs turn by the base's joints it lacks
  const std::map<std::string, Rigid> rigid = rigidPlan(skeletonOf(w.scene, itemRigs), figureSkeleton, cage ? &cage->skeleton : nullptr);

  const Mesh&            bodyMesh = figure.scene.meshes[size_t(figure.mesh)];
  const mesh::BuiltMesh& body     = *bodyMesh.geo;
  const Surface          bodySurface{body.cagePositions, body.dsonTris};
  std::vector<std::string> boneNames;
  for(const BoneRef& b : itemBones)
    boneNames.push_back(b.name);
  const FitRegion region = item.region ? *item.region : regionOf(rel + "/" + item.label, boneNames);

  // a prop hung from a bone keeps its place on the joint: it was placed around the source's. With no
  // skinned mesh there is no skeleton or shape to refit, nor the base's search to make for one
  detail.mountFrom = mountsFrom(item, cage ? &cage->skeleton : nullptr);
  const bool skinned = std::any_of(item.meshes.begin(), item.meshes.end(), [&](int m) {
    const Mesh& mesh = w.scene.meshes[size_t(m)];
    return mesh.geo && !mesh.cageSkin.start.empty();
  });
  if(!skinned)
  {
    const bool onBones = std::any_of(item.objects.begin(), item.objects.end(), [](const auto& o) { return !o.second.empty(); });
    const bool carried = std::any_of(detail.mountFrom.begin(), detail.mountFrom.end(), [](const auto& f) { return f.has_value(); });
    detail.note        = std::string(label(source)) + " prop on " + label(target)
                + (!onBones  ? std::string(", placed as it was made.")
                   : carried ? std::string(", moved with the joint it hangs from.")
                             : std::string(": the ") + label(source) + " base figure isn't installed, so it hangs where it was made for that figure.");
    return detail;
  }
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
  const std::vector<int16_t> bodyLimbs = meshLimbs(figure.scene, bodyMesh);
  const bool boot     = region == FitRegion::eBoot || region == FitRegion::eBootThigh;
  const bool footwear = region == FitRegion::eShoe || boot;
  // the base-to-target search does not depend on the garment; a shoe never asks for it
  std::optional<SurfaceBinding> pair;
  if(region != FitRegion::eShoe && base && cage && !sameTopology(source, target))
  {
    const auto filter = limbFilter(bodyLimbs, baseLimbs);
    pair              = cachedBindToSurface(bodySurface.positions, bodySurface.tris, base->positions, {}, filter ? &*filter : nullptr);
  }
  // footwear's foot skips the shape correction, so its proportions come from a box fit
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

  // the common layer: the maps of the type the item was made for and of the figure's own, and the
  // figure's skin as its map's body has it
  std::shared_ptr<const bodymap::TypeMap> typeMap, targetMap;
  bodymap::Body                           onto;
  garment::Target                         decodeOnto;
  std::string                             notLayered;  // why the layer, asked for, didn't fit it
  const std::string targetGender = genderHint(figure.folder);
  const auto        mapped       = [](Generation g, const std::string& gender) { return g == Generation::eG9 || !bodymap::cloneFile(g, gender).empty(); };
  // a shoe keeps the bone transfer and the foot's box alone; a boot's shaft is cloth round the leg
  const bool shoe   = region == FitRegion::eShoe;
  detail.footwear   = shoe;
  // a Skyrim outfit goes through the map of its whole body, where one is on this machine
  const bool layerAble =
      !shoe && mapped(target, targetGender) && (w.sourceWhole ? true : cage && !w.sourceBase && mapped(source, sourceGender));
  // without one it has no map to be fitted through, and takes the body's weights where the direct fit lands it
  const bool weightsOnly = source == Generation::eSkyrim && !shoe;
  detail.commonAble      = layerAble || weightsOnly;
  if(path == FitPath::eCommon && !detail.commonAble)
    notLayered = shoe                            ? "a shoe keeps the fit of its foot"
                 : source == Generation::eSkyrim ? "the body this Skyrim outfit was made on isn't installed"
                 : !cage                         ? std::string("the ") + label(source) + " base figure isn't installed"
                                                 : std::string("there is no body map of ") + label(mapped(source, sourceGender) ? target : source);
  if(path == FitPath::eCommon && layerAble)
  {
    if(viaWhole)
    {
      // its body's map, read at the outfit's preset and size: the map is by vertex, whatever the shape
      if(const auto whole = garment::bodyMap(store, w.sourceWhole, w.sourceGender))
      {
        auto own              = std::make_shared<bodymap::TypeMap>(*whole);
        own->source.positions = cage->positions;
        typeMap               = std::move(own);
      }
    }
    else
      typeMap = garment::typeMap(store, source, sourceGender);
    targetMap = garment::typeMap(store, target, targetGender);
    if(typeMap && typeMap->source.positions.size() != cage->positions.size())
      typeMap = nullptr;
    if(targetMap && targetMap->source.positions.size() != body.cagePositions.size())
      targetMap = nullptr;
    if(!typeMap || !targetMap)
    {
      notLayered = viaWhole && !typeMap ? std::string("Genesis 9's base isn't installed to map its body by")
                                        : std::string("Genesis 9's base, or its clone morph of ") + label(!typeMap ? source : target) + ", isn't installed";
      typeMap    = nullptr;
    }
    else
    {
      onto.positions = body.cagePositions;
      onto.tris      = targetMap->source.tris;
      onto.outer     = targetMap->source.outer;
      if(target == Generation::eG9)
        decodeOnto.reference = &onto;
      else
      {
        // the reference laid on the figure's type, the figure's own skin, and its frames' turns
        decodeOnto.reference = &targetMap->reference;
        decodeOnto.body      = &onto;
        decodeOnto.frameTurn = &targetMap->map.frameTurn;
      }
    }
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
    // a boot's share on the foot's bones, for its skin (FitDetail::onFoot)
    if(boot)
    {
      std::vector<bool> foot;
      for(int b : m.bones)
      {
        const int16_t l = limbOfBone(w.scene.nodes[size_t(b)].name);
        foot.push_back(l == limb::kLFoot || l == limb::kRFoot);
      }
      std::vector<float> share(positions.size() / 3, 0.f);
      for(size_t v = 0; v < share.size(); v++)
      {
        for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
          if(foot[size_t(skin.bone[size_t(i)])])
            share[v] += skin.weight[size_t(i)];
        share[v] = std::min(1.f, share[v]);
      }
      detail.onFoot[meshIdx] = std::move(share);
    }
    // how much of each vertex rides on a foot
    std::vector<double> onFoot(positions.size() / 3, 0.0);
    for(const auto& [l, box] : feet)
    {
      std::vector<bool> on;
      for(int b : m.bones)
        on.push_back(limbOfBone(w.scene.nodes[size_t(b)].name) == l);
      const auto share = [&](size_t v) {
        double wsum = 0;
        for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
          if(on[size_t(skin.bone[size_t(i)])])
            wsum += skin.weight[size_t(i)];
        return std::min(1.0, wsum);
      };
      for(size_t v = 0; v < onFoot.size(); v++)
        onFoot[v] = std::max(onFoot[v], share(v));
      positions = applyLimbBox(box, positions, share);
    }
    // footwear's foot keeps the bone transfer and the foot box: a heel rests shaped for the foot pose
    // its file sets (its heel below the floor), not around the flat base a correction measures
    // against, which would bend it. A boot's shaft takes the correction, fading out onto the foot. It
    // is all still pushed out of the foot: a sole stays, lying off the foot's faces rather than behind them
    bool             layered = false;
    garment::Decoded decoded;
    if(typeMap && m.geo->cageLineStart.empty())
    {
      // the bones' tears mended first; not a boot's, whose foot has its box
      if(!boot)
        positions = garment::mend(m.geo->cagePositions, positions, m.geo->dsonTris);
      // bound to the base as it was made (a vertex's limb is the same moved or not)
      const auto               filter    = limbFilter(baseLimbs, meshLimbs(w.scene, m));
      const garment::Addresses addresses = garment::convert(m.geo->cagePositions, *typeMap, filter ? &*filter : nullptr);
      decoded                            = garment::decode(addresses, decodeOnto);
      // a geograft's seam is welded to its own figure's skin: here it lies on the skin at its place, not
      // at the height its address carries (the reference laid on the source lies a millimetre or two off)
      if(const Graft* g = graftOf(w, meshIdx))
        for(size_t i = 0; i + 1 < g->pairs.size(); i += 2)
        {
          const size_t v = size_t(g->pairs[i]);
          if(v >= decoded.hold.size() || decoded.hold[v] <= 0)
            continue;
          float p[3] = {0, 0, 0}, sum = 0;
          for(size_t j = 0; j < 3; j++)
          {
            const size_t c  = decoded.at.verts[v * 3 + j];
            const float  wj = decoded.at.weights[v * 3 + j];
            if(wj <= 0 || c * 3 + 2 >= body.cagePositions.size())
              continue;
            for(size_t k = 0; k < 3; k++)
              p[k] += wj * body.cagePositions[c * 3 + k];
            sum += wj;
          }
          if(sum > 0)
            for(size_t k = 0; k < 3; k++)
              decoded.positions[v * 3 + k] = p[k] / sum;
        }
      if(std::any_of(decoded.hold.begin(), decoded.hold.end(), [](float h) { return h > 0; }))
      {
        // a boot's shaft takes it, fading out onto the foot, as it takes the old shape correction
        const std::vector<float> settled = garment::settle(positions, decoded.positions, decoded.hold, m.geo->dsonTris);
        positions                        = boot ? fadeByFoot(positions, settled, onFoot) : settled;
        layered       = true;
        detail.common = true;
      }
    }
    if(region != FitRegion::eShoe && !layered)
    {
      if(base)
      {
        const std::vector<int16_t> itemLimbs = meshLimbs(w.scene, m);
        const auto baseFilter = limbFilter(bodyLimbs, baseLimbs);
        const auto itemFilter = limbFilter(baseLimbs, itemLimbs);
        ShapeInputs in;
        in.sameTopology = sameTopology(source, target);
        in.baseFilter   = baseFilter ? &*baseFilter : nullptr;
        in.itemFilter   = itemFilter ? &*itemFilter : nullptr;
        in.baseToTarget = pair ? &*pair : nullptr;
        in.lineStart    = m.geo->cageLineStart;
        in.lineVerts    = m.geo->cageLineVerts;
        const std::vector<float> corrected = shapeCorrection(positions, m.geo->dsonTris, *base, bodySurface, in);
        positions = boot ? fadeByFoot(positions, corrected, onFoot) : corrected;
      }
    }
    // a dynamic surface owns its collision: dForce drapes it over the body, so it is not pushed out here.
    // Nor a geograft the layer fitted: its seam lies on the skin and its inside under it, as on its own
    // figure, and the skin it covers is hidden (hides)
    const bool graftLaid = layered && graftOf(w, meshIdx);
    if(!(m.dforce && m.dforce->simulated()) && !graftLaid)
      positions = pushOutOfBody(positions, m.geo->dsonTris, bodySurface, m.geo->cageLineStart, m.geo->cageLineVerts, footwear ? 3.f : kSunkDepth);
    // with the common fit, a hard piece (a button, a buckle, a plate on one bone) is put back in the
    // shape it was made in, where the fit and the push left it. Not a shoe's, fitted as a whole
    // already, nor cloth dForce drapes
    if(path == FitPath::eCommon && !shoe && m.geo->cageLineStart.empty() && !(m.dforce && m.dforce->simulated()))
    {
      garment::Pieces pieces = garment::hardPieces(m.geo->cagePositions, m.geo->dsonTris, skin);
      if(pieces.count > 0)
      {
        positions = garment::keepPieces(m.geo->cagePositions, positions, pieces);
        // those a pose carries whole, for its skin
        for(int32_t& p : pieces.of)
          if(p >= 0 && !pieces.hard[size_t(p)])
            p = -1;
        detail.pieces[meshIdx] = std::move(pieces.of);
      }
    }
    // a vertex its address holds whole, left where the address put it, keeps that place on the body as
    // its binding (kPlacedWithin: the settle or the push moved the others, which bind by where they are)
    if(layered)
    {
      for(size_t v = 0; v < decoded.hold.size(); v++)
      {
        double off2 = 0;
        for(size_t k = 0; k < 3; k++)
          off2 += double(positions[v * 3 + k] - decoded.positions[v * 3 + k]) * (positions[v * 3 + k] - decoded.positions[v * 3 + k]);
        if(decoded.hold[v] < 1 || off2 > kPlacedWithin * kPlacedWithin)
          decoded.at.weights[v * 3] = decoded.at.weights[v * 3 + 1] = decoded.at.weights[v * 3 + 2] = 0;
      }
      detail.placed[meshIdx] = std::move(decoded.at);
      // a geograft made for another mesh hides the figure's skin under it, found through the maps
      if(const Graft* g = graftOf(w, meshIdx);
         g && !g->hidden.empty() && !graftFits(*g, body) && cage && (g->polyCount < 0 || g->polyCount == cage->polyCount))
        if(auto hides = garment::hiddenThrough(*cage, g->hidden, *typeMap, *targetMap, body.cagePolyStart, body.cagePolyVerts); !hides.empty())
          detail.hides[meshIdx] = std::move(hides);
    }
    writeCage(*m.geo, positions);
  }
  detail.shaped      = base.has_value();
  detail.bodyWeights = path == FitPath::eCommon && (detail.common || weightsOnly);
  detail.note        = std::string(label(source)) + " outfit on " + label(target) + ", "
                + (detail.common && viaWhole        ? std::string("fitted through the common layer by the map of the ") + w.sourceGender + " body it was made on"
                   : detail.common                  ? std::string("fitted through the common layer")
                   : base                           ? std::string("refitted to its skeleton and shape")
                   : source == Generation::eSkyrim ? std::string("refitted to its skeleton (no body came with it, so its shape wasn't matched)")
                                                    : std::string("refitted to its skeleton (the ") + label(source)
                                                          + " base figure isn't installed, so its shape wasn't matched)")
                + (path != FitPath::eCommon || detail.common ? std::string()
                   : weightsOnly                             ? std::string(", its cloth on the skin taking the body's weights")
                                                             : ", the old way: " + (notLayered.empty() ? std::string("none of its cloth lies near the skin") : notLayered))
                + (detail.hides.empty() ? std::string() : std::string(", hiding the skin under it as it does on its own figure"))
                + ".";
  return detail;
}

}  // namespace fv::wardrobe
