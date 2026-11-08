#include <figure/materials/procedural.hpp>

#include <algorithm>
#include <cmath>

#include <figure/math/noise.hpp>
#include <figure/wardrobe/fit/bones.hpp>

namespace fv::mat {

namespace {

// Where people sweat first, heaviest first (the regional maps of Taylor and Machado-Moreira 2013,
// read for what shows): the forehead, the spine and the underarms, then the chest, the upper lip, the
// chin, the neck and the hollows above the collarbones. Set by eye against the bones of Genesis 8 and 9, whose eyes sit about 0.03 below the
// forehead's middle and whose spine runs about 0.06 in front of the back. Z is the figure's front.
constexpr FieldPatch kSweatPatches[] = {
    {"eye", {0, 0.030, 0.010}, {0.045, 0.024, 0.060}, 1.00f},        // the forehead, out to the temples
    {"eye", {0, -0.020, 0.020}, {0.008, 0.015, 0.020}, 0.55f},       // the nose
    {"eye", {0, -0.036, 0.012}, {0.014, 0.006, 0.030}, 0.75f},       // the upper lip
    {"eye", {0, -0.055, 0.008}, {0.020, 0.012, 0.030}, 0.50f},       // the chin
    {"neck1", {0, 0, 0}, {0.040, 0.035, 0.040}, 0.55f},              // the neck
    {"collar", {0.025, 0.012, 0.035}, {0.018, 0.012, 0.020}, 0.50f}, // the hollows above the collarbones
    {"spine4", {0, 0.010, 0.060}, {0.032, 0.070, 0.050}, 0.70f},     // the chest, along the breastbone
    {"spine3", {0, 0, -0.060}, {0.028, 0.120, 0.040}, 0.80f},        // the spine
    {"spine1", {0, 0, -0.060}, {0.055, 0.040, 0.040}, 0.65f},        // the small of the back
    {"spine2", {0, 0, 0.050}, {0.022, 0.060, 0.040}, 0.45f},         // the belly's middle
    {"upperarm", {0, -0.040, 0}, {0.030, 0.032, 0.040}, 0.85f},      // the underarms
    {"forearm", {0, 0, 0.012}, {0.020, 0.020, 0.018}, 0.30f},        // the inner elbows
    {"hand", {0, -0.015, 0}, {0.025, 0.030, 0.025}, 0.35f},          // the palms
    {"shin", {0, 0, -0.022}, {0.022, 0.022, 0.018}, 0.35f},          // behind the knees
};

// Sweat is mostly a film: the skin's own surface made smoother (its lobes to about 0.2) and a little
// darker and richer, as a wet surface is (Lekner and Dick 1988), with beads on it. The film is even
// where it lies: noise in it gave patches a few centimetres across, each darker than the skin beside
// it, which read as blotches, not wet skin. A film as a coat over the skin did the same, the coat
// taking its share off the skin under it. The beads are clear domes of water, a coat at about water's
// 2% head on (half the coat's fixed 4%); tinted, or with the skin's lobes off under them, they were
// dark spots where they caught no light. Their own field follows the body more closely and clusters.
// The numbers are set by eye.
constexpr ProceduralDef kDefs[] = {
    {.key           = "sweat",
     .channel       = 0,
     .dropChannel   = 1,
     .amount        = &Edits::sweat,
     .patches       = kSweatPatches,
     .film          = {.floor = 0.14f, .blotch = 0.05f, .blotchDepth = 0},
     .drops         = {.floor = 0.04f, .blotch = 0.06f, .blotchDepth = 0.6f},
     .unknownBody   = 0.6f,
     .tint          = {0.9f, 0.88f, 0.87f},
     .filmRoughness = 0.22f,
     .coat          = 0.5f,
     .dropRoughness = 0.04f,
     .tileMetres    = kSweatTileMetres,
     .relief        = kSweatRelief,
     .heightMap     = kSweatHeightMap,
     .coverMap      = kSweatCoverMap},
};

using noise::fade;
using noise::noise3;

// The patch's share at a point: its whole weight out to half its radius, none past one and a half. A
// bell curve (exp of minus the distance squared) left most of a forehead at a third of its weight, so
// on Leyla 9 the face averaged 0.3 everywhere and nowhere was wetter than anywhere else.
double patchShare(const FieldPatch& p, const Vec3& centre, double side, double height, const Vec3& point)
{
  const double dx = (point.x - centre.x) / height - p.offset[0] * side;
  const double dy = (point.y - centre.y) / height - p.offset[1];
  const double dz = (point.z - centre.z) / height - p.offset[2];
  const double d  = std::sqrt(dx * dx / (p.radius[0] * p.radius[0]) + dy * dy / (p.radius[1] * p.radius[1]) + dz * dz / (p.radius[2] * p.radius[2]));
  return double(p.weight) * (1 - fade(std::clamp(d - 0.5, 0.0, 1.0)));
}

}  // namespace

BodyLandmarks bodyLandmarks(std::span<const std::pair<std::string, Vec3>> bones)
{
  BodyLandmarks out;
  for(const auto& [name, at] : bones)
  {
    const wardrobe::ParsedBone parsed = wardrobe::parseBone(name);
    // a limb's twist bone sits midway along it; the bend bone is its joint
    if(parsed.twist)
      continue;
    std::string key = parsed.side ? std::string(1, parsed.side) + ":" + parsed.slot : parsed.slot;
    out.at.try_emplace(std::move(key), at);
  }
  // both eyes as one place between them
  const auto left = out.at.find("l:eye"), right = out.at.find("r:eye");
  const auto footL = out.at.find("l:foot"), footR = out.at.find("r:foot");
  if(left == out.at.end() || right == out.at.end() || (footL == out.at.end() && footR == out.at.end()))
    return {};
  const Vec3 eye{(left->second.x + right->second.x) * 0.5, (left->second.y + right->second.y) * 0.5, (left->second.z + right->second.z) * 0.5};
  out.at["eye"] = eye;
  const double foot = footL == out.at.end() ? footR->second.y : footR == out.at.end() ? footL->second.y : std::min(footL->second.y, footR->second.y);
  // ankles to eyes is about 90% of a figure's height
  out.height = (eye.y - foot) * 1.12;
  if(out.height <= 0)
    return {};
  return out;
}

std::span<const ProceduralDef> proceduralDefs()
{
  return kDefs;
}

bool takesProcedural(const ProceduralDef&, const Surface& s)
{
  return s.kind == Kind::eSkin;
}

const ProceduralDef* proceduralOn(const Surface& s, const Edits& e)
{
  for(const ProceduralDef& def : kDefs)
    if((e.*def.amount).value_or(0.f) > 0 && takesProcedural(def, s))
      return &def;
  return nullptr;
}

float proceduralField(const ProceduralDef& def, const FieldShape& shape, const BodyLandmarks& body, const Vec3& point)
{
  double field;
  double scale = body.height;
  if(body.height <= 0)
  {
    field = def.unknownBody;
    // no size to go by: a figure's usual, in DSON's centimetres
    scale = 170;
  }
  else
  {
    // together, as chances are: each covers a share of what the others left
    double dry = 1 - double(shape.floor);
    for(const FieldPatch& p : def.patches)
    {
      if(const auto it = body.at.find(p.slot); it != body.at.end())
        dry *= 1 - patchShare(p, it->second, 1, body.height, point);
      else
        for(const auto& [prefix, side] : {std::pair{"l:", 1.0}, std::pair{"r:", -1.0}})
          if(const auto sided = body.at.find(std::string(prefix) + p.slot); sided != body.at.end())
            dry *= 1 - patchShare(p, sided->second, side, body.height, point);
    }
    field = 1 - dry;
  }
  if(shape.blotchDepth <= 0)
    return float(std::clamp(field, 0.0, 1.0));
  const double n = noise3(point.x / (shape.blotch * scale), point.y / (shape.blotch * scale), point.z / (shape.blotch * scale));
  return float(std::clamp(field * (1 - double(shape.blotchDepth) * n), 0.0, 1.0));
}

std::vector<uint32_t> proceduralMasks(std::span<const float> position, const Mat4& toWorld, const BodyLandmarks& body)
{
  std::vector<uint32_t> out(position.size() / 3, 0);
  for(size_t v = 0; v < out.size(); v++)
  {
    const Vec3 world = toWorld.transformPoint(Vec3{position[v * 3], position[v * 3 + 1], position[v * 3 + 2]});
    uint32_t   packed = 0;
    for(const ProceduralDef& def : kDefs)
    {
      packed |= uint32_t(std::lround(proceduralField(def, def.film, body, world) * 255.f)) << (def.channel * 8);
      packed |= uint32_t(std::lround(proceduralField(def, def.drops, body, world) * 255.f)) << (def.dropChannel * 8);
    }
    out[v] = packed;
  }
  return out;
}

}  // namespace fv::mat
