#pragma once

// A DAZ pose file's animation: `scene.animations` whose keys run over time, as DAZ Studio saves an
// animated pose preset (a key a frame, [seconds, value]). Each channel is resolved once, to a bone
// channel of the figure's PoseRig or one of its dials, by the rule a pose preset is applied with
// (PresetResolver, which fv::applyPosePreset in figure.hpp uses too), and sampled at a time: linear
// between keys, DSON's default; a key marked "constant" holds until the next; before the first key
// and after the last the nearest one holds, as DAZ Studio does.
//
// Many pose presets carry two or more keys and are still poses: a rest-to-pose ramp saved for dForce
// to drape along. isAnimated says only that keys run over time, not that a file is meant to play.

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_rig.hpp>

namespace fv::pose {

// The dial a modifier id means on a node (empty: the selection) when the figure has none of that
// name: a worn item's. Empty for none.
using ChannelOf = std::function<std::string(const std::string& node, const std::string& modifier)>;

// What a preset animation's url names on a figure.
struct PresetTarget
{
  enum class Kind
  {
    eIgnored,     // nothing of this figure's to report: another node's bone, an axis the bone lacks, a dial's other property
    eUnreadable,  // none of the url forms: reported as `name`, the url
    eMissing,     // a bone or dial the figure doesn't have: reported as `name` when its value isn't 0
    eBone,        // `bone`'s `channel` / `axis`
    eDial,        // the dial `name`
    eVisibility,  // the node `name` shown or hidden
  };
  Kind        kind = Kind::eIgnored;
  std::string bone;
  ChannelKind channel = ChannelKind::eRotation;
  std::string axis;
  std::string name;
};

// Resolves preset urls against one figure's rig and dials: `name://@selection` urls,
// `name://<node>#<modifier>` urls (a worn item's dial, a geograft's say) and hierarchical
// `#bone?rotation/x/value` urls. A url without a bone names the figure's root.
class PresetResolver
{
public:
  PresetResolver(const PoseRig* pose, const morph::MorphSet* morphs, ChannelOf channelOf = {});
  PresetTarget resolve(const std::string& url) const;

private:
  const PoseRig*          m_pose   = nullptr;
  const morph::MorphSet*  m_morphs = nullptr;
  ChannelOf               m_channelOf;
  const PoseBone*         m_root = nullptr;
};

// Whether some channel of the file has keys at two different times.
bool isAnimated(const dson::Json& doc);

struct PoseClip
{
  struct Channel
  {
    PresetTarget        target;  // a bone channel or a dial
    std::vector<float>  times;   // seconds, in order
    std::vector<double> values;
    std::vector<bool>   held;    // the key holds its value until the next ("constant")
  };
  std::vector<Channel>     channels;
  float                    duration = 0;  // the last key's time
  std::vector<std::string> skipped;       // what the figure has nothing for, as applyPosePreset reports it
};

// The file's numeric channels that reach this figure's bones and dials.
PoseClip loadPoseClip(const PoseRig* pose, const morph::MorphSet* morphs, const dson::Json& doc, const ChannelOf& channelOf = {});

// One channel's value at `time` (seconds).
double sample(const PoseClip::Channel& channel, float time);

}  // namespace fv::pose
