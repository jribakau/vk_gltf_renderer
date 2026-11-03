#include <figure/pose/pose_clip.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <regex>

namespace fv::pose {

namespace {

std::string withoutValue(std::string prop)
{
  const std::string suffix = "/value";
  if(prop.size() >= suffix.size() && prop.compare(prop.size() - suffix.size(), suffix.size(), suffix) == 0)
    prop.resize(prop.size() - suffix.size());
  return prop;
}

// A key's interpolation, its third element: a name, or an array that starts with one.
bool holds(const dson::Json& key)
{
  if(key.size() < 3)
    return false;
  const dson::Json& how = key[2].is_array() && !key[2].empty() ? key[2][0] : key[2];
  if(!how.is_string())
    return false;
  std::string name = how.get<std::string>();
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return name == "constant" || name == "step";
}

}  // namespace

PresetResolver::PresetResolver(const PoseRig* pose, const morph::MorphSet* morphs, ChannelOf channelOf)
    : m_pose(pose)
    , m_morphs(morphs)
    , m_channelOf(std::move(channelOf))
{
  if(m_pose)
    for(const PoseBone& pb : m_pose->bones())
      if(pb.group == "Figure")
      {
        m_root = &pb;
        break;
      }
}

PresetTarget PresetResolver::resolve(const std::string& url) const
{
  static const std::regex selection(R"(^name://@selection(?:/([^:#]*))?(?:#([^:]*))?:\?(.*)$)");
  // a node by its DAZ name: "name://Advanced Natural Anatomy Layer_1006#PG_ANAL%20Anus%20Down:?value/value"
  static const std::regex named(R"(^name://([^@/#:][^/#:]*)(?:/([^:#]*))?(?:#([^:]*))?:\?(.*)$)");
  static const std::regex hierarchical(R"(^[^:]*:[^#]*#([^?]+)\?(.*)$)");
  static const std::regex channel(R"(^(rotation|translation|scale)/(x|y|z|general)$)");
  static const std::string visibleProp = "extra/studio_node_channels/channels/Visible";

  using Kind = PresetTarget::Kind;
  PresetTarget out;
  std::smatch  m;
  std::string  nodeName, boneName, modifier, prop;
  if(std::regex_match(url, m, selection))
  {
    boneName = m[1].matched ? dson::decodeUri(m[1].str()) : "";
    modifier = m[2].matched ? dson::decodeUri(m[2].str()) : "";
    prop     = m[3].str();
  }
  else if(std::regex_match(url, m, named))
  {
    nodeName = dson::decodeUri(m[1].str());
    boneName = m[2].matched ? dson::decodeUri(m[2].str()) : "";
    modifier = m[3].matched ? dson::decodeUri(m[3].str()) : "";
    prop     = m[4].str();
  }
  else if(std::regex_match(url, m, hierarchical))
  {
    boneName = dson::decodeUri(m[1].str());
    prop     = m[2].str();
  }
  else
  {
    out.kind = Kind::eUnreadable;
    out.name = url;
    return out;
  }
  prop = withoutValue(prop);
  if(prop == visibleProp && modifier.empty())
  {
    // shown or hidden: the node under the selection, or the one named
    out.name = !boneName.empty() ? boneName : nodeName;
    out.kind = out.name.empty() ? Kind::eIgnored : Kind::eVisibility;
    return out;
  }
  if(!modifier.empty())
  {
    if(prop != "value" || !boneName.empty())
      return out;
    // the figure's dial of that name (unless the url names another node), or the one a worn item made for it
    std::string           name = modifier;
    const morph::Channel* ch   = m_morphs && nodeName.empty() ? m_morphs->channel(name) : nullptr;
    if(!ch && m_morphs && m_channelOf)
    {
      name = m_channelOf(nodeName, modifier);
      ch   = name.empty() ? nullptr : m_morphs->channel(name);
    }
    out.kind = ch ? Kind::eDial : Kind::eMissing;
    out.name = ch ? name : nodeName.empty() ? modifier : nodeName + " " + modifier;
    return out;
  }
  // a bone of another node: its skeleton follows the figure's
  if(!nodeName.empty() || !m_pose)
    return out;
  const PoseBone* pb = boneName.empty() ? m_root : m_pose->get(boneName);
  std::smatch     pm;
  if(!std::regex_match(prop, pm, channel))
    return out;
  if(!pb)
  {
    out.kind = Kind::eMissing;
    out.name = (boneName.empty() ? "figure" : boneName) + " " + prop;
    return out;
  }
  ChannelKind kind;
  if(!parseKind(pm[1].str(), kind))
    return out;
  for(const Axis& axis : pb->axes)
    if(axis.kind == kind && axis.axis == pm[2].str())
    {
      out.kind    = Kind::eBone;
      out.bone    = pb->name;
      out.channel = kind;
      out.axis    = pm[2].str();
      return out;
    }
  return out;
}

bool isAnimated(const dson::Json& doc)
{
  const dson::Json* animations = dson::get(dson::get(&doc, "scene"), "animations");
  if(!animations || !animations->is_array())
    return false;
  for(const dson::Json& a : *animations)
  {
    const dson::Json* keys = dson::get(&a, "keys");
    if(!keys || !keys->is_array() || keys->size() < 2)
      continue;
    const dson::Json* first = nullptr;
    for(const dson::Json& key : *keys)
    {
      if(!key.is_array() || key.size() < 2 || !key[0].is_number() || !key[1].is_number())
        continue;
      if(!first)
        first = &key;
      else if(key[0].get<double>() != (*first)[0].get<double>())
        return true;
    }
  }
  return false;
}

PoseClip loadPoseClip(const PoseRig* pose, const morph::MorphSet* morphs, const dson::Json& doc, const ChannelOf& channelOf)
{
  PoseClip clip;
  const dson::Json* animations = dson::get(dson::get(&doc, "scene"), "animations");
  if(!pose || !animations || !animations->is_array())
    return clip;
  const PresetResolver resolver(pose, morphs, channelOf);
  for(const dson::Json& a : *animations)
  {
    const dson::Json* keys = dson::get(&a, "keys");
    if(!keys || !keys->is_array())
      continue;
    PoseClip::Channel c;
    for(const dson::Json& key : *keys)
      if(key.is_array() && key.size() >= 2 && key[0].is_number() && key[1].is_number())
      {
        c.times.push_back(key[0].get<float>());
        c.values.push_back(key[1].get<double>());
        c.held.push_back(holds(key));
      }
    if(c.times.empty())
      continue;
    using Kind = PresetTarget::Kind;
    c.target = resolver.resolve(dson::getString(&a, "url"));
    if(c.target.kind == Kind::eUnreadable
       || (c.target.kind == Kind::eMissing && std::any_of(c.values.begin(), c.values.end(), [](double v) { return v != 0; })))
      clip.skipped.push_back(c.target.name);
    if(c.target.kind != Kind::eBone && c.target.kind != Kind::eDial)
      continue;
    // keys in time order, as written they should be already
    if(!std::is_sorted(c.times.begin(), c.times.end()))
    {
      std::vector<size_t> order(c.times.size());
      std::iota(order.begin(), order.end(), size_t(0));
      std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return c.times[x] < c.times[y]; });
      PoseClip::Channel sorted;
      sorted.target = c.target;
      for(size_t i : order)
      {
        sorted.times.push_back(c.times[i]);
        sorted.values.push_back(c.values[i]);
        sorted.held.push_back(c.held[i]);
      }
      c = std::move(sorted);
    }
    clip.duration = std::max(clip.duration, c.times.back());
    clip.channels.push_back(std::move(c));
  }
  return clip;
}

double sample(const PoseClip::Channel& c, float time)
{
  if(c.times.empty())
    return 0;
  if(time <= c.times.front())
    return c.values.front();
  if(time >= c.times.back())
    return c.values.back();
  // the key after `time` and the one at or before it
  const size_t next = size_t(std::upper_bound(c.times.begin(), c.times.end(), time) - c.times.begin());
  const size_t at   = next - 1;
  const float  span = c.times[next] - c.times[at];
  if(c.held[at] || span <= 0)
    return c.values[at];
  const double f = double(time - c.times[at]) / double(span);
  return c.values[at] + (c.values[next] - c.values[at]) * f;
}

}  // namespace fv::pose
