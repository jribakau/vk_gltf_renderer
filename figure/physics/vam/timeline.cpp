#include <figure/physics/vam/timeline.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace fv::physics::vam {

namespace {

using Json = nlohmann::json;

// Timeline's null keyframe: no key on that side
constexpr int kNull = -1;

// VaM writes its numbers as text, in the invariant culture
double number(const Json& j, double fallback)
{
  if(j.is_number())
    return j.get<double>();
  if(!j.is_string())
    return fallback;
  const std::string& s = j.get_ref<const std::string&>();
  double             v = fallback;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  return ec == std::errc() ? v : fallback;
}

double number(const Json& o, const char* key, double fallback)
{
  const auto it = o.find(key);
  return it == o.end() ? fallback : number(*it, fallback);
}

bool flag(const Json& o, const char* key, bool fallback)
{
  const auto it = o.find(key);
  if(it == o.end())
    return fallback;
  if(it->is_boolean())
    return it->get<bool>();
  if(it->is_number())
    return it->get<double>() != 0;
  if(it->is_string())
  {
    const std::string& s = it->get_ref<const std::string&>();
    return s == "1" || s == "true" || s == "True";
  }
  return fallback;
}

std::string text(const Json& o, const char* key)
{
  const auto it = o.find(key);
  return it != o.end() && it->is_string() ? it->get<std::string>() : std::string();
}

std::optional<Vec3> vec(const Json& o, const char* key)
{
  const auto it = o.find(key);
  if(it == o.end() || !it->is_object())
    return std::nullopt;
  return Vec3{number(*it, "x", 0), number(*it, "y", 0), number(*it, "z", 0)};
}

int hexDigit(char c)
{
  if(c >= '0' && c <= '9')
    return c - '0';
  if(c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  if(c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  throw std::runtime_error(std::string("a Timeline key has '") + c + "' where a hex digit goes");
}

uint8_t hexByte(const std::string& s, size_t at)
{
  if(at + 2 > s.size())
    throw std::runtime_error("a Timeline key ends early: " + s);
  return uint8_t(hexDigit(s[at]) * 16 + hexDigit(s[at + 1]));
}

// four bytes as BitConverter writes a float: little-endian
float hexFloat(const std::string& s, size_t at)
{
  uint8_t b[4];
  for(int i = 0; i < 4; i++)
    b[i] = hexByte(s, at + size_t(i) * 2);
  const uint32_t u = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
  float          f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// DecodeKeyframe
Key decodeKey(const std::string& s, float lastV, int lastC, int version)
{
  if(s.empty())
    throw std::runtime_error("a Timeline key is empty");
  Key k;
  if(version <= 230)
  {
    // the first character says how many bytes each part takes: 25 per time byte, 5 per value byte, and
    // whether a curve type follows
    const char c     = s[0];
    int        index = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'z' ? c - 'a' + 10 : c - 'A' + 36;
    const int  tBytes = index / 25;
    index %= 25;
    const int  vBytes = index / 5;
    const bool hasC   = index % 5 != 0;
    if(tBytes != 4 || (vBytes != 0 && vBytes != 4))
      throw std::runtime_error("a Timeline key in an encoding this doesn't read: " + s);
    k.time  = hexFloat(s, 1);
    k.value = vBytes ? hexFloat(s, 1 + 8) : lastV;
    k.curve = hasC ? hexByte(s, 1 + size_t(tBytes + vBytes) * 2) : lastC;
    return k;
  }
  const int  bits     = s[0] - 'A';
  const bool hasValue = bits & 1, hasCurve = bits & 2;
  if(bits < 0 || bits > 3)
    throw std::runtime_error("a Timeline key starts with '" + s.substr(0, 1) + "'");
  k.time  = hexFloat(s, 1);
  k.value = hasValue ? hexFloat(s, 9) : lastV;
  k.curve = hasCurve ? hexByte(s, hasValue ? 17 : 9) : lastC;
  return k;
}

// BezierAnimationCurve.KeyframeBinarySearch's neighbours, as Evaluate uses them: the key at or before
// `time`, the first for a time before it
size_t segment(const std::vector<Key>& keys, float time)
{
  const auto it = std::upper_bound(keys.begin(), keys.end(), time, [](float t, const Key& k) { return t < k.time; });
  return it == keys.begin() ? 0 : size_t(it - keys.begin()) - 1;
}

// BezierAnimationCurve.ComputeValue
float valueBetween(const Key& current, const Key* next, float time)
{
  if(!next)
    return current.value;
  const float span = next->time - current.time;
  const float t    = span > 0 ? (time - current.time) / span : 0;
  switch(current.curve)
  {
    case eConstant:
      return current.value;
    case eLinear:
    case eFlatLinear:
      return current.value + (next->value - current.value) * t;
    default:
      break;
  }
  const float mt = 1 - t;
  return current.value * mt * mt * mt + 3 * current.out * mt * mt * t + 3 * next->in * mt * t * t + next->value * t * t * t;
}

bool isNull(const Key& k)
{
  return k.curve == kNull;
}

// ComputeKey, SmoothLocalInterpolation and LinearInterpolation
void computeKey(std::vector<Key>& keys, size_t key, bool loop)
{
  const size_t n = keys.size();
  Key          previous{0, 0, kNull, 0, 0};
  float        previousTime = 0;
  if(key >= 1)
  {
    previous     = keys[key - 1];
    previousTime = previous.time;
  }
  else if(loop)
  {
    previous     = keys[n - 2];
    previousTime = previous.time - keys[n - 1].time;
  }
  Key   current = keys[key];
  Key   next{0, 0, kNull, 0, 0};
  float nextTime = keys[n - 1].time;
  if(key + 1 < n)
  {
    next     = keys[key + 1];
    nextTime = next.time;
  }
  else if(loop)
  {
    next     = keys[1];
    nextTime = current.time + next.time;
  }

  int curve = current.curve;
  if(curve == eCopyPrevious && !isNull(previous))
  {
    current.value = previous.value;
    curve         = previous.curve == eCopyPrevious ? eSmoothLocal : previous.curve;
  }
  switch(curve)
  {
    case eLinear:
      current.in  = isNull(previous) ? current.value : current.value - (current.value - previous.value) / 3;
      current.out = isNull(next) ? current.value : current.value + (next.value - current.value) / 3;
      break;
    case eSmoothLocal:
    case eSmoothGlobal:
      if(!isNull(next) && !isNull(previous))
      {
        // Timeline's "no overshoot": the longer side's handle weighs more
        const float inHandle  = (current.value - previous.value) / 3;
        const float outHandle = (next.value - current.value) / 3;
        const float both      = nextTime - previousTime;
        const float inRatio   = both > 0 ? (current.time - previousTime) / both : 0.5f;
        const float outRatio  = both > 0 ? (nextTime - current.time) / both : 0.5f;
        const float weighted  = inHandle * inRatio + outHandle * outRatio;
        if(inRatio > outRatio)
        {
          current.in  = current.value - weighted;
          current.out = current.value + weighted * (outRatio / inRatio);
        }
        else
        {
          current.in  = current.value - (outRatio > 0 ? weighted * (inRatio / outRatio) : 0);
          current.out = current.value + weighted;
        }
      }
      else if(isNull(previous))
      {
        current.in  = current.value;
        current.out = current.value + (next.value - current.value) / 3;
      }
      else
      {
        current.in  = current.value - (current.value - previous.value) / 3;
        current.out = current.value;
      }
      break;
    case eLinearFlat:
      current.in  = isNull(previous) ? current.value : current.value - (current.value - previous.value) / 3;
      current.out = current.value;
      break;
    case eFlat:
    case eFlatLong:
    case eFlatLinear:
      current.in = current.out = current.value;
      break;
    case eBounce:
      if(!isNull(previous) && !isNull(next))
      {
        current.in  = current.value - (current.value - next.value) / 1.4f;
        current.out = current.value + (previous.value - current.value) / 1.8f;
      }
      else
        current.in = current.out = current.value;
      break;
    case eLeaveAsIs:
      break;
    default:
      // Timeline leaves a key of a type it doesn't know as it was
      return;
  }
  keys[key] = current;
}

// by id, the first of each (a Timeline storable is megabytes: not copied)
std::map<std::string, const Json*> storablesOf(const Json& list)
{
  std::map<std::string, const Json*> out;
  if(list.is_array())
    for(const Json& s : list)
      if(s.is_object())
        if(const std::string id = text(s, "id"); !id.empty())
          out.emplace(id, &s);
  return out;
}

bool isController(const std::string& id)
{
  return id == "control" || (id.size() > 7 && id.compare(id.size() - 7, 7, "Control") == 0);
}

Pose poseOf(const Json& storables)
{
  Pose p;
  for(const auto& [id, at] : storablesOf(storables))
  {
    const Json& s = *at;
    if(id == "control")
    {
      p.rootPosition = vec(s, "position");
      p.rootRotation = vec(s, "rotation");
    }
    if(isController(id))
    {
      Controller c;
      if(const std::string v = text(s, "positionState"); !v.empty())
        c.positionState = v;
      if(const std::string v = text(s, "rotationState"); !v.empty())
        c.rotationState = v;
      c.mass                 = number(s, "mass", c.mass);
      c.holdPositionSpring   = number(s, "holdPositionSpring", c.holdPositionSpring);
      c.holdPositionDamper   = number(s, "holdPositionDamper", c.holdPositionDamper);
      c.holdPositionMaxForce = number(s, "holdPositionMaxForce", c.holdPositionMaxForce);
      c.holdRotationSpring   = number(s, "holdRotationSpring", c.holdRotationSpring);
      c.holdRotationDamper   = number(s, "holdRotationDamper", c.holdRotationDamper);
      c.holdRotationMaxForce = number(s, "holdRotationMaxForce", c.holdRotationMaxForce);
      c.jointDriveSpring     = number(s, "jointDriveSpring", c.jointDriveSpring);
      c.jointDriveDamper     = number(s, "jointDriveDamper", c.jointDriveDamper);
      c.jointDriveMaxForce   = number(s, "jointDriveMaxForce", c.jointDriveMaxForce);
      c.position             = vec(s, "localPosition");
      c.rotation             = vec(s, "localRotation");
      p.controllers[id]      = c;
    }
    else if(const auto at = vec(s, "position"))
      p.bones[id] = *at;
  }
  return p;
}

Track trackOf(const Json& c, int version, float length, bool loop, bool continuous)
{
  Track t;
  t.controller = text(c, "Controller");
  t.position   = flag(c, "TargetsPosition", true) && flag(c, "ControlPosition", true);
  t.rotation   = flag(c, "TargetsRotation", true) && flag(c, "ControlRotation", true);
  const char* names[7] = {"X", "Y", "Z", "RotX", "RotY", "RotZ", "RotW"};
  Curve*      curves[7] = {&t.xyz[0], &t.xyz[1], &t.xyz[2], &t.rot[0], &t.rot[1], &t.rot[2], &t.rot[3]};
  for(int i = 0; i < 7; i++)
    if(const auto it = c.find(names[i]); it != c.end())
      curves[i]->keys = decodeKeys(*it, version);
  // as TransformAnimationTarget.AddEdgeFramesIfMissing: each three with its first curve's last type
  for(int g = 0; g < 2; g++)
  {
    Curve* const* group = g == 0 ? curves : curves + 3;
    const int     count = g == 0 ? 3 : 4;
    const int     type  = group[0]->keys.empty() ? int(eSmoothLocal) : group[0]->keys.back().curve;
    for(int i = 0; i < count; i++)
      group[i]->addEdges(length, type);
  }
  // AtomAnimationClip.Rebuild: a loop ends where it starts, then the rotation's signs are made
  // continuous from its own last key, then the control points
  if(loop)
    for(Curve* curve : curves)
      if(curve->keys.size() >= 2)
      {
        Key first  = curve->keys.front();
        first.time = curve->keys.back().time;
        curve->keys.back() = first;
      }
  const size_t n = t.rot[0].keys.size();
  if(continuous && n >= 2 && t.rot[1].keys.size() == n && t.rot[2].keys.size() == n && t.rot[3].keys.size() == n)
  {
    float last[4] = {t.rot[0].keys[n - 1].value, t.rot[1].keys[n - 1].value, t.rot[2].keys[n - 1].value, t.rot[3].keys[n - 1].value};
    for(size_t k = 0; k < n; k++)
    {
      float cur[4] = {t.rot[0].keys[k].value, t.rot[1].keys[k].value, t.rot[2].keys[k].value, t.rot[3].keys[k].value};
      if(cur[0] * last[0] + cur[1] * last[1] + cur[2] * last[2] + cur[3] * last[3] < 0)
        for(float& v : cur)
          v = -v;
      for(int a = 0; a < 4; a++)
      {
        t.rot[size_t(a)].keys[k].value = cur[a];
        last[a]                        = cur[a];
      }
    }
  }
  for(Curve* curve : curves)
    curve->compute(loop);
  return t;
}

}  // namespace

void Curve::addEdges(float length, int curveType)
{
  switch(keys.size())
  {
    case 0:
      keys.push_back({0, 0, curveType, 0, 0});
      keys.push_back({length, 0, curveType, 0, 0});
      return;
    case 1:
    {
      keys[0].time = 0;
      Key last     = keys[0];
      last.time    = length;
      keys.push_back(last);
      return;
    }
    default:
      break;
  }
  if(keys.front().time > 0)
  {
    if(keys.size() > 2)
      keys.insert(keys.begin(), Key{0, keys.front().value, curveType, 0, 0});
    else
      keys.front().time = 0;
  }
  if(keys.back().time < length)
  {
    if(keys.size() > 2)
      keys.push_back({length, keys.back().value, curveType, 0, 0});
    else
      keys.back().time = length;
  }
}

void Curve::compute(bool loop)
{
  const size_t n = keys.size();
  if(n <= 2)
  {
    for(Key& k : keys)
      if(k.curve != eLeaveAsIs)
        k.in = k.out = k.value;
    return;
  }
  for(size_t k = 0; k < n; k++)
    computeKey(keys, k, loop);
}

float Curve::at(float time) const
{
  if(keys.empty())
    return 0;
  if(keys.size() == 1)
    return keys[0].value;
  const size_t k = segment(keys, time);
  return valueBetween(keys[k], k + 1 < keys.size() ? &keys[k + 1] : nullptr, time);
}

Vec3 Track::positionAt(float time) const
{
  return {xyz[0].at(time), xyz[1].at(time), xyz[2].at(time)};
}

Quat Track::rotationAt(float time) const
{
  return Quat{rot[0].at(time), rot[1].at(time), rot[2].at(time), rot[3].at(time)}.normalized();
}

bool Controller::holdsPosition() const
{
  return positionState == "On" || positionState == "Hold" || positionState == "Lock" || positionState == "Comply";
}

bool Controller::holdsRotation() const
{
  return rotationState == "On" || rotationState == "Hold" || rotationState == "Lock" || rotationState == "Comply";
}

std::vector<Key> decodeKeys(const Json& keys, int version)
{
  std::vector<Key> out;
  if(!keys.is_array())
  {
    if(keys.is_null())
      return out;
    // the semicolon-separated text and the object of arrays that Timeline wrote before 2020
    throw std::runtime_error("a Timeline curve in an encoding older than this reads");
  }
  float lastT = -1, lastV = 0;
  int   lastC = eSmoothLocal;
  for(const Json& j : keys)
  {
    Key k;
    if(version >= 230 && j.is_string() && !j.get_ref<const std::string&>().empty())
      k = decodeKey(j.get_ref<const std::string&>(), lastV, lastC, version);
    else if(j.is_object())
    {
      k.time  = float(number(j, "t", 0));
      k.value = float(number(j, "v", lastV));
      k.curve = j.contains("c") ? int(number(j, "c", lastC)) : lastC;
      // tangents went with the Bezier curves (version 230): such a key is smoothed again
      if(version < 230 && j.contains("ti") && k.curve == eLeaveAsIs)
        k.curve = eSmoothLocal;
      k.in  = float(number(j, "i", k.value));
      k.out = float(number(j, "o", k.value));
    }
    else
      throw std::runtime_error("a Timeline key this doesn't read: " + j.dump());
    if(!std::isfinite(k.time) || !std::isfinite(k.value) || k.time < 0)
      continue;
    if(std::abs(k.time - lastT) <= 1.1920929e-7f)
      continue;
    lastT = k.time;
    lastV = k.value;
    lastC = k.curve;
    // AddKey: in time order (a key at a time already there is dropped)
    if(!out.empty() && k.time < out.back().time)
    {
      const auto at = std::lower_bound(out.begin(), out.end(), k.time, [](const Key& a, float t) { return a.time < t; });
      if(at != out.end() && std::abs(at->time - k.time) < 1e-4f)
        continue;
      out.insert(at, k);
    }
    else
      out.push_back(k);
  }
  return out;
}

Scene readScene(const Json& doc)
{
  if(!doc.is_object() || !doc.contains("atoms") || !doc["atoms"].is_array())
    throw std::runtime_error("this isn't a VaM scene (it has no atoms)");
  Scene scene;
  for(const Json& atom : doc["atoms"])
  {
    if(!atom.is_object() || text(atom, "type") != "Person")
      continue;
    Person person;
    person.id                  = text(atom, "id");
    static const Json none     = Json::array();
    const auto        listed   = atom.find("storables");
    const Json&       storables = listed != atom.end() && listed->is_array() ? *listed : none;
    person.scene               = poseOf(storables);
    const Json* timeline       = nullptr;
    for(const Json& s : storables)
      if(s.is_object() && text(s, "id").find("VamTimeline.AtomPlugin") != std::string::npos && s.contains("Animation"))
        timeline = &s["Animation"];
    if(timeline && timeline->is_object())
    {
      const int version = int(number(*timeline, "SerializeVersion", 0));
      for(const Json& c : timeline->value("Clips", Json::array()))
      {
        if(!c.is_object())
          continue;
        Clip clip;
        clip.name             = text(c, "AnimationName");
        clip.length           = std::max(0.f, float(number(c, "AnimationLength", 0)));
        clip.loop             = flag(c, "Loop", true);
        const bool continuous = flag(c, "EnsureQuaternionContinuity", true);
        for(const Json& t : c.value("Controllers", Json::array()))
          // a track of another atom (a light, a camera, a station) names it; the Person's own don't
          if(t.is_object() && (text(t, "Atom").empty() || text(t, "Atom") == person.id) && !text(t, "Controller").empty())
            clip.tracks.push_back(trackOf(t, version, clip.length, clip.loop, continuous));
        if(const auto it = c.find("Pose"); it != c.end() && it->is_object() && it->contains("storables"))
          clip.pose = poseOf((*it)["storables"]);
        person.clips.push_back(std::move(clip));
      }
    }
    scene.people.push_back(std::move(person));
  }
  return scene;
}

Scene readScene(const std::string& text)
{
  Json doc = Json::parse(text, nullptr, false);
  if(doc.is_discarded())
    throw std::runtime_error("this isn't a VaM scene (it isn't JSON)");
  return readScene(doc);
}

}  // namespace fv::physics::vam
