#include <figure/bodymap/regions.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>

namespace fv::bodymap {

namespace {

// the bits a right-hand region is moved up by
constexpr int kRightShift = 25;
static_assert(int(rig::Region::ePinkyToe) < kRightShift && int(rig::Region::ePinkyToe) + kRightShift < 64);

}  // namespace

RegionMask regionBit(rig::Region region, char side)
{
  if(region == rig::Region::eNone)
    return 0;
  return RegionMask(1) << (int(region) + (side == 'r' ? kRightShift : 0));
}

std::vector<RegionMask> vertexRegions(const mesh::SkinInput& skin, const std::vector<wardrobe::SkelBone>& bones, size_t vertices, float border)
{
  // each bone's region: its joint's on its side, else its nearest parent's that has one
  std::unordered_map<std::string, size_t> byName;
  for(size_t b = 0; b < bones.size(); b++)
    byName.emplace(bones[b].name, b);
  std::vector<RegionMask> boneRegion(bones.size(), 0);
  for(size_t b = 0; b < bones.size(); b++)
  {
    size_t at = b;
    for(int depth = 0; depth < 64; depth++)
    {
      const rig::Name name = rig::parse(bones[at].name);
      if(name.joint && name.joint->region != rig::Region::eNone)
      {
        boneRegion[b] = regionBit(name.joint->region, name.joint->sided ? name.side : 0);
        break;
      }
      const auto parent = bones[at].parent ? byName.find(*bones[at].parent) : byName.end();
      if(parent == byName.end())
        break;
      at = parent->second;
    }
  }
  std::vector<RegionMask> out(vertices, 0);
  if(skin.start.size() != vertices + 1)
    return out;
  for(size_t v = 0; v < vertices; v++)
  {
    float most = 0;
    int   best = -1;
    for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
      if(skin.weight[size_t(i)] > most)
        most = skin.weight[size_t(i)], best = skin.bone[size_t(i)];
    for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
    {
      const int b = skin.bone[size_t(i)];
      if(b >= 0 && size_t(b) < bones.size() && (b == best || skin.weight[size_t(i)] >= border))
        out[v] |= boneRegion[size_t(b)];
    }
  }
  return out;
}

namespace {

// the first joint of a region in the spec (parents come first)
const rig::Joint* firstJointOf(rig::Region region)
{
  for(const rig::Joint& j : rig::joints())
    if(j.region == region)
      return &j;
  return nullptr;
}

}  // namespace

rig::Region regionAbove(rig::Region region)
{
  const rig::Joint* j = firstJointOf(region);
  for(int guard = 0; j && guard < 64; guard++)
  {
    j = j->parent.empty() ? nullptr : rig::joint(j->parent);
    if(j && j->region != rig::Region::eNone && j->region != region)
      return j->region;
  }
  return rig::Region::eNone;
}

std::vector<rig::Region> regionsBelow(rig::Region region)
{
  const rig::Joint* j = firstJointOf(region);
  if(!j || regionAbove(region) == rig::Region::eNone)
    return {};
  for(const std::string_view aim : j->aims)
    if(const rig::Joint* a = aim.empty() ? nullptr : rig::joint(aim); a && a->region != rig::Region::eNone && a->region != region)
      return {a->region};
  std::vector<rig::Region> out;
  for(int r = 1; r <= int(rig::Region::ePinkyToe); r++)
    if(regionAbove(rig::Region(r)) == region)
      out.push_back(rig::Region(r));
  return out;
}

bool regionSided(rig::Region region)
{
  const rig::Joint* j = firstJointOf(region);
  return j && j->sided;
}

std::string regionNames(RegionMask mask)
{
  static const char* const names[] = {"none", "pelvis", "abdomen", "chest", "breast", "neck", "head", "collar", "upperarm", "forearm", "hand", "thumb", "index",
                                      "mid", "ring", "pinky", "thigh", "shin", "foot", "bigtoe", "indextoe", "midtoe", "ringtoe", "pinkytoe"};
  std::string out;
  for(int bit = 0; bit < 64; bit++)
    if(mask & (RegionMask(1) << bit))
    {
      const int region = bit % kRightShift;
      if(!out.empty())
        out += ' ';
      if(bit >= kRightShift)
        out += "r ";
      out += region < int(std::size(names)) ? names[region] : "?";
    }
  return out.empty() ? "none" : out;
}

bool outerSkin(std::string_view surface)
{
  std::string s;
  for(const char c : surface)
    if(std::isalnum(static_cast<unsigned char>(c)))
      s += char(std::tolower(static_cast<unsigned char>(c)));
  for(const std::string_view inner : {"pupil", "iris", "sclera", "cornea", "moisture", "reflection", "lacrimal", "lash", "teeth", "tooth", "tongue", "mouth", "gum", "cavity", "tear"})
    if(s.find(inner) != std::string::npos)
      return false;
  return true;
}

}  // namespace fv::bodymap
