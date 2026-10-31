#include <figure/morph/shape_sliders.hpp>

#include <algorithm>
#include <cmath>

#include <figure/wardrobe/fit/bones.hpp>

namespace fv::morph {

using dson::Json;

namespace {

constexpr double kSliderMin = -1, kSliderMax = 1;

// a bone's axis needs a length; moves smaller than this are rounding
constexpr double kShortestBone = 1e-3;
constexpr double kSmallestMove = 1e-5;

ChannelOptions sliderOptions(const std::string& label)
{
  ChannelOptions opts;
  opts.label = label;
  opts.group = "Shape";
  opts.min   = kSliderMin;
  opts.max   = kSliderMax;
  return opts;
}

dson::MorphInfo sliderInfo(const std::string& region)
{
  dson::MorphInfo info;
  info.region = region;
  info.kind   = kShapeKind;
  return info;
}

// The skin's bones as a recipe reads them: from the file's centres, where a character's shape hasn't
// moved them, so that a slider is the same whichever shape it's added to.
std::vector<ShapeBone> shapeBones(const SceneGraph& scene, const Mesh& mesh)
{
  std::vector<ShapeBone> bones;
  bones.reserve(mesh.bones.size());
  for(const int id : mesh.bones)
  {
    const Node& node = scene.nodes[size_t(id)];
    ShapeBone   bone;
    bone.name = node.name;
    if(node.joint)
    {
      bone.center = node.joint->centerAuthored;
      bone.end    = node.joint->endPoint;
    }
    bones.push_back(std::move(bone));
  }
  return bones;
}

}  // namespace

const std::vector<GirthRecipe>& girthRecipes()
{
  static const std::vector<GirthRecipe> recipes{
      {"fv_shape_waist", "Waist", "Waist", {"spine1", "spine2"}, 0.22f, 1.f},
      {"fv_shape_thighs", "Thighs", "Legs", {"thigh"}, 0.22f, 0.25f},
  };
  return recipes;
}

const std::vector<DialRecipe>& dialRecipes()
{
  static const std::vector<DialRecipe> recipes{
      {"fv_shape_height", "Height", "Actor", "body_bs_ProportionHeight", 1},
  };
  return recipes;
}

mesh::CageDeltas girthDeltas(std::span<const float> cage, const mesh::SkinInput& skin, const std::vector<ShapeBone>& bones, const GirthRecipe& recipe)
{
  struct Axis
  {
    bool                  on = false;
    std::array<double, 3> center{}, dir{};
  };
  std::vector<Axis> axes(bones.size());
  for(size_t b = 0; b < bones.size(); b++)
  {
    const wardrobe::ParsedBone parsed = wardrobe::parseBone(bones[b].name);
    if(std::find(recipe.slots.begin(), recipe.slots.end(), parsed.slot) == recipe.slots.end())
      continue;
    std::array<double, 3> d{bones[b].end[0] - bones[b].center[0], bones[b].end[1] - bones[b].center[1], bones[b].end[2] - bones[b].center[2]};
    const double          length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if(length < kShortestBone)
      continue;
    axes[b] = {true, bones[b].center, {d[0] / length, d[1] / length, d[2] / length}};
  }

  mesh::CageDeltas out;
  const size_t     count = std::min(cage.size() / 3, skin.start.empty() ? size_t(0) : skin.start.size() - 1);
  for(size_t v = 0; v < count; v++)
  {
    std::array<double, 3> move{};
    for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
    {
      const size_t b = size_t(skin.bone[size_t(i)]);
      if(b >= axes.size() || !axes[b].on)
        continue;
      const Axis&           axis = axes[b];
      std::array<double, 3> offset{double(cage[v * 3]) - axis.center[0], double(cage[v * 3 + 1]) - axis.center[1], double(cage[v * 3 + 2]) - axis.center[2]};
      const double          along = offset[0] * axis.dir[0] + offset[1] * axis.dir[1] + offset[2] * axis.dir[2];
      for(int c = 0; c < 3; c++)
        offset[size_t(c)] -= along * axis.dir[size_t(c)];
      const double k = double(skin.weight[size_t(i)]) * recipe.cap;
      // off to one side of the midline: pushing back across the axis goes toward the other limb
      const double sideways = offset[0] * axis.center[0] < 0 ? recipe.inward : 1.f;
      move[0] += k * sideways * offset[0];
      move[1] += k * offset[1];
      move[2] += k * offset[2];
    }
    if(std::abs(move[0]) < kSmallestMove && std::abs(move[1]) < kSmallestMove && std::abs(move[2]) < kSmallestMove)
      continue;
    out.verts.push_back(int32_t(v));
    out.deltas.push_back(float(move[0]));
    out.deltas.push_back(float(move[1]));
    out.deltas.push_back(float(move[2]));
  }
  return out;
}

void addShapeSliders(MorphSet& morphs, pose::FigureDriver& driver, const SceneGraph& scene, const std::vector<const dson::MeshRef*>& bodies)
{
  for(const GirthRecipe& recipe : girthRecipes())
  {
    bool added = false;
    for(const dson::MeshRef* body : bodies)
    {
      const Mesh& mesh = scene.meshes[size_t(body->mesh)];
      if(!mesh.skinned() || !body->geo || mesh.cageSkin.start.size() != body->geo->cagePositions.size() / 3 + 1)
        continue;
      const mesh::CageDeltas deltas = girthDeltas(body->geo->cagePositions, mesh.cageSkin, shapeBones(scene, mesh), recipe);
      if(deltas.verts.empty())
        continue;
      const ChannelOptions opts = sliderOptions(recipe.label);
      morphs.addTarget(recipe.channel, body->mesh, std::make_shared<mesh::RenderTarget>(mesh::target(*body->geo, deltas)), opts);
      // the mesh's shells take it too, remade on their own polygons
      for(const dson::MeshRef::Shell& s : body->shells)
        morphs.addTarget(recipe.channel, s.mesh, std::make_shared<mesh::RenderTarget>(mesh::target(*s.geo, deltas)), opts);
      added = true;
    }
    if(added)
      morphs.describe(recipe.channel, sliderInfo(recipe.region));
  }

  Json modifiers = Json::array();
  for(const DialRecipe& recipe : dialRecipes())
  {
    if(!morphs.has(recipe.dial))
      continue;
    // nothing to read: its formula is what it does, and the channel stays when its load finds no deltas
    morphs.addLazy(
        recipe.channel,
        []() -> Finish {
          return [] {
            LazyResult result;
            result.keep = true;
            return result;
          };
        },
        sliderOptions(recipe.label));
    morphs.describe(recipe.channel, sliderInfo(recipe.region));
    modifiers.push_back({{"id", recipe.channel},
                         {"channel", {{"value", 0}}},
                         {"formulas", Json::array({{{"output", "#" + recipe.dial + "?value"},
                                                    {"operations", Json::array({{{"op", "push"}, {"url", "#" + recipe.channel + "?value"}},
                                                                                {{"op", "push"}, {"val", recipe.scale}},
                                                                                {{"op", "mult"}}})}}})}});
  }
  if(!modifiers.empty())
    driver.graph.addDoc(Json{{"modifier_library", std::move(modifiers)}}, "fv:shape-sliders");
}

}  // namespace fv::morph
