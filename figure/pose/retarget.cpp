#include <figure/pose/retarget.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <set>

#include <figure/wardrobe/fit/bones.hpp>

namespace fv::pose {

namespace {

// The bones a pose file's animations name, by their names as the urls give them.
std::set<std::string> namedBones(const dson::Json& doc)
{
  static const std::regex selection(R"(^name://@selection/([^:#]+):\?(rotation|translation)/)");
  static const std::regex hierarchical(R"(^[^:]*:[^#]*#([^?]+)\?(rotation|translation)/)");
  std::set<std::string> out;
  const auto            scene = doc.find("scene");
  if(scene == doc.end() || !scene->contains("animations"))
    return out;
  for(const dson::Json& a : (*scene)["animations"])
  {
    const std::string url = a.value("url", "");
    std::smatch       m;
    if(std::regex_search(url, m, selection) || std::regex_search(url, m, hierarchical))
      out.insert(dson::decodeUri(m[1].str()));
  }
  return out;
}

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

Vec3 v3(const std::array<double, 3>& a)
{
  return {a[0], a[1], a[2]};
}

Vec3 cross(const Vec3& a, const Vec3& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// A bone's local turn as its channels give it: orient · R · orient⁻¹
Quat localTurn(const PoseBone& pb, const std::array<double, 3>& degrees)
{
  return pb.orient * quatFromEuler(degrees[0] * kDeg, degrees[1] * kDeg, degrees[2] * kDeg, pb.order) * pb.orient.inverse();
}

// A rig's bones as limbs: each posable bone's parent bone, and per (slot, side) the limb's bones from
// its bend down its twists
struct Limbs
{
  const PoseRig&                                      rig;
  const SceneGraph&                                   scene;
  std::vector<int>                                    parent;  // per rig bone, -1 for the root
  std::vector<std::pair<std::string, char>>           slotOf;  // per rig bone
  std::map<std::pair<std::string, char>, std::vector<int>> limb;

  Limbs(const PoseRig& r, const SceneGraph& s)
      : rig(r)
      , scene(s)
  {
    const auto& bones = rig.bones();
    std::map<int, int> ofNode;
    for(size_t i = 0; i < bones.size(); i++)
      ofNode[bones[i].bone] = int(i);
    parent.assign(bones.size(), -1);
    for(size_t i = 0; i < bones.size(); i++)
    {
      for(int n = scene.nodes[size_t(bones[i].bone)].parent; n >= 0; n = scene.nodes[size_t(n)].parent)
        if(const auto it = ofNode.find(n); it != ofNode.end())
        {
          parent[i] = it->second;
          break;
        }
      const wardrobe::ParsedBone p = wardrobe::parseBone(bones[i].name);
      slotOf.emplace_back(p.slot, p.side);
    }
    // a limb's bones in order from the top, down its own chain: a twist bone the next limb hangs from
    // (Genesis 8's), not one at its side that formulas turn (Genesis 9's), which is left to them
    std::vector<char> onChain(bones.size(), 0);
    for(size_t c = 0; c < bones.size(); c++)
      if(const int a = parent[c]; a >= 0 && slotOf[size_t(a)] != slotOf[c])
        // c starts a limb below a: a is on its limb's chain, and so is all of that above it
        for(int x = a; x >= 0 && slotOf[size_t(x)] == slotOf[size_t(a)]; x = parent[size_t(x)])
          onChain[size_t(x)] = 1;
    std::map<std::pair<std::string, char>, std::vector<int>> all;
    for(size_t i = 0; i < bones.size(); i++)
      all[slotOf[i]].push_back(int(i));
    for(auto& [key, list] : all)
    {
      std::sort(list.begin(), list.end(), [&](int a, int b) { return depth(a) < depth(b); });
      std::vector<int>& kept = limb[key];
      for(size_t k = 0; k < list.size(); k++)
        if(k == 0 || onChain[size_t(list[k])])
          kept.push_back(list[k]);
    }
  }
  int depth(int b) const
  {
    int d = 0;
    for(int p = parent[size_t(b)]; p >= 0; p = parent[size_t(p)])
      d++;
    return d;
  }
  // which way a limb points at rest: from its top bone's joint to its last bone's end
  Vec3 direction(const std::vector<int>& bones) const
  {
    const auto& top  = scene.nodes[size_t(rig.bones()[size_t(bones.front())].bone)].joint;
    const auto& last = scene.nodes[size_t(rig.bones()[size_t(bones.back())].bone)].joint;
    return normalize(v3(last->endPoint) - v3(top->center));
  }
  // the top bone's twist axis (0 x, 1 y, 2 z): the first of its DAZ order, the last of PoseBone::order
  // (three's), whose turn its Euler makes first
  int twistAxis(const std::vector<int>& bones) const
  {
    const PoseBone& pb = rig.bones()[size_t(bones.front())];
    return pb.order.empty() ? 0 : pb.order.back() - 'X';
  }
  // a hinge's axis at rest, in the body's space: the one axis but its twist its top bone bends on (an
  // elbow); none for a limb that bends more ways
  std::optional<Vec3> hinge(const std::vector<int>& bones) const
  {
    const PoseBone& pb    = rig.bones()[size_t(bones.front())];
    const int       twist = twistAxis(bones);
    int             axis  = -1;
    for(int a = 0; a < 3; a++)
      if(a != twist && pb.axis(ChannelKind::eRotation, a))
      {
        if(axis >= 0)
          return std::nullopt;
        axis = a;
      }
    if(axis < 0)
      return std::nullopt;
    return rotate({axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, axis == 2 ? 1.0 : 0.0}, pb.orient);
  }
  // the limbs whose top bone hangs from one of these
  std::vector<const std::vector<int>*> below(const std::vector<int>& bones) const
  {
    std::vector<const std::vector<int>*> out;
    for(const auto& [key, list] : limb)
      if(const int p = parent[size_t(list.front())]; p >= 0 && std::find(bones.begin(), bones.end(), p) != bones.end())
        out.push_back(&list);
    return out;
  }
  // a bone's own direction at rest, from its joint to its end
  Vec3 own(int b) const
  {
    const auto& j = scene.nodes[size_t(rig.bones()[size_t(b)].bone)].joint;
    return normalize(v3(j->endPoint) - v3(j->center));
  }
};

// `v` without what lies along `about` (unit), unit; zero when nothing is left
Vec3 across(const Vec3& v, const Vec3& about)
{
  return normalize(v - about * dot(v, about));
}

// The channels for local turn `goal` (`want`: its Euler, degrees, as rotationTotals gives it) on a bone
// that lacks some of the axes: those at 0, and the ones it has fitted so it points where `goal` would
// point it (`dir`, its own direction at rest), with a little of its roll: a few damped Gauss-Newton steps.
std::array<double, 3> fitAxes(const PoseBone& pb, const Quat& goal, const Vec3& dir, std::array<double, 3> want)
{
  std::vector<int> free;
  bool             lacks = false;
  for(int a = 0; a < 3; a++)
    if(pb.axis(ChannelKind::eRotation, a))
      free.push_back(a);
    else
    {
      lacks |= std::abs(want[size_t(a)]) > 0.01;
      want[size_t(a)] = 0;
    }
  if(!lacks || free.empty())
    return want;
  const Vec3 side       = normalize(cross(dir, std::abs(dir.y) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0}));
  const Vec3 aim        = rotate(dir, goal), sideAim = rotate(side, goal);
  const auto residual   = [&](const std::array<double, 3>& v, std::array<double, 6>& r) {
    const Quat q = localTurn(pb, v);
    const Vec3 a = rotate(dir, q) - aim, b = (rotate(side, q) - sideAim) * 0.2;
    r            = {a.x, a.y, a.z, b.x, b.y, b.z};
  };
  const size_t n = free.size();
  for(int iter = 0; iter < 12; iter++)
  {
    std::array<double, 6> r0;
    residual(want, r0);
    // the slope of the residual by each axis it has (per degree)
    std::vector<std::array<double, 6>> J(n);
    for(size_t k = 0; k < n; k++)
    {
      std::array<double, 3> v = want;
      v[size_t(free[k])] += 0.05;
      residual(v, J[k]);
      for(int i = 0; i < 6; i++)
        J[k][size_t(i)] = (J[k][size_t(i)] - r0[size_t(i)]) / 0.05;
    }
    // (JᵀJ + λ) δ = -Jᵀr, at most three unknowns
    double A[3][4] = {};
    for(size_t k = 0; k < n; k++)
    {
      for(size_t l = 0; l < n; l++)
        for(int i = 0; i < 6; i++)
          A[k][l] += J[k][size_t(i)] * J[l][size_t(i)];
      A[k][k] += 1e-6;
      for(int i = 0; i < 6; i++)
        A[k][3] -= J[k][size_t(i)] * r0[size_t(i)];
    }
    for(size_t k = 0; k < n; k++)
    {
      size_t pivot = k;
      for(size_t l = k + 1; l < n; l++)
        if(std::abs(A[l][k]) > std::abs(A[pivot][k]))
          pivot = l;
      std::swap(A[k], A[pivot]);
      if(std::abs(A[k][k]) < 1e-12)
        return want;
      for(size_t l = 0; l < n; l++)
        if(l != k)
        {
          const double f = A[l][k] / A[k][k];
          for(size_t c = k; c < 4; c++)
            A[l][c] -= f * A[k][c];
        }
    }
    double most = 0;
    for(size_t k = 0; k < n; k++)
    {
      const double step = std::clamp(A[k][3] / A[k][k], -30.0, 30.0);
      want[size_t(free[k])] += step;
      most = std::max(most, std::abs(step));
    }
    if(most < 1e-4)
      break;
  }
  return want;
}

}  // namespace

std::string sourceFigureFile(const dson::Json& doc, const std::string& rel)
{
  const std::set<std::string> bones = namedBones(doc);
  const auto                  has   = [&](std::initializer_list<const char*> names) {
    return std::any_of(names.begin(), names.end(), [&](const char* n) { return bones.count(n) > 0; });
  };
  const std::string path = lower(rel);
  const auto        in   = [&](std::initializer_list<const char*> words) {
    return std::any_of(words.begin(), words.end(), [&](const char* w) { return path.find(w) != std::string::npos; });
  };
  if(has({"l_thigh", "r_thigh", "l_upperarm", "r_upperarm", "l_forearm", "l_shin"}))
    return "data/DAZ 3D/Genesis 9/Base/Genesis9.dsf";
  if(has({"lThighBend", "rThighBend", "lShldrBend", "rShldrBend", "abdomenLower", "chestLower"}))
  {
    if(in({"genesis 3 male", "michael 7"}))
      return "data/DAZ 3D/Genesis 3/Male/Genesis3Male.dsf";
    if(in({"genesis 3 female", "victoria 7"}))
      return "data/DAZ 3D/Genesis 3/Female/Genesis3Female.dsf";
    if(in({"genesis 8 male", "genesis 8.1 male", "michael 8", "michael8"}))
      return "data/DAZ 3D/Genesis 8/Male/Genesis8Male.dsf";
    if(in({"genesis 8.1 female", "8_1"}))
      return "data/DAZ 3D/Genesis 8/Female 8_1/Genesis8_1Female.dsf";
    return "data/DAZ 3D/Genesis 8/Female/Genesis8Female.dsf";
  }
  if(has({"lThigh", "rThigh", "lShldr", "rShldr", "abdomen"}))
    return in({"genesis 2 male", "michael 6"}) ? "data/DAZ 3D/Genesis 2/Male/Genesis2Male.dsf" : "data/DAZ 3D/Genesis 2/Female/Genesis2Female.dsf";
  return {};
}

std::string sourceFigureName(const std::string& file)
{
  static const std::regex stem(R"(Genesis(\d)(?:_(\d))?(Female|Male)?\.dsf$)");
  std::smatch             m;
  if(!std::regex_search(file, m, stem))
    return "figure";
  return "Genesis " + m[1].str() + (m[2].matched ? "." + m[2].str() : "") + (m[3].matched ? " " + m[3].str() : "");
}

bool namesRigBones(const dson::Json& doc, const PoseRig& rig)
{
  const std::set<std::string> bones = namedBones(doc);
  size_t                      found = 0;
  for(const std::string& b : bones)
    found += rig.get(b) ? 1 : 0;
  return bones.empty() || found * 2 >= bones.size();
}

PoseClip retargetClip(const dson::Json& doc, const PoseRig& from, const PoseRig& to)
{
  PoseClip out;
  // the file on the skeleton it was made for
  const PoseClip source = loadPoseClip(&from, nullptr, doc);
  const auto&    sb     = from.bones();
  const auto&    tb     = to.bones();
  const Limbs    src(from, from.scene()), dst(to, to.scene());
  // each source bone's rotation channels and its hip move, by axis
  std::vector<std::array<const PoseClip::Channel*, 3>> turns(sb.size()), moves(sb.size());
  std::set<float>                                       timeSet;
  for(const PoseClip::Channel& c : source.channels)
  {
    if(c.target.kind != PresetTarget::Kind::eBone)
      continue;
    const int b = from.indexOf(c.target.bone);
    const int a = axisIndex(c.target.axis);
    if(b < 0 || a < 0 || a > 2)
      continue;
    if(c.target.channel == ChannelKind::eRotation)
      turns[size_t(b)][size_t(a)] = &c;
    else if(c.target.channel == ChannelKind::eTranslation)
      moves[size_t(b)][size_t(a)] = &c;
    else
      continue;
    timeSet.insert(c.times.begin(), c.times.end());
  }
  if(timeSet.empty())
    return out;
  const std::vector<float> times(timeSet.begin(), timeSet.end());

  // the figure's limbs, each with its counterpart on the source and how its rest pose differs
  struct Match
  {
    std::vector<int> bones;      // the figure's, top first
    int              last = -1;  // the source limb's last bone: its turn is the limb's
    Quat             rest;       // from the figure's rest direction to the source's
    int              twist = 0;  // the figure's top bone's twist axis
    Vec3             dir;        // the figure's top bone's own direction at rest
  };
  std::vector<Match>                matches;
  std::vector<int>                  matchOf(tb.size(), -1);
  std::vector<char>                 moved(sb.size(), 0);
  for(size_t b = 0; b < sb.size(); b++)
    moved[b] = turns[b][0] || turns[b][1] || turns[b][2];
  // the figure's node (a url with no bone poses it) is each's own, whatever its name
  const auto rootOf = [](const Limbs& l) -> const std::pair<std::string, char>* {
    for(size_t b = 0; b < l.parent.size(); b++)
      if(l.parent[b] < 0)
        return &l.slotOf[b];
    return nullptr;
  };
  const auto* srcRoot = rootOf(src);
  const auto* dstRoot = rootOf(dst);
  const auto  counterpart = [&](const std::pair<std::string, char>& key) {
    if(dstRoot && srcRoot && key == *dstRoot)
      return src.limb.find(*srcRoot);
    return src.limb.find(key);
  };
  for(const auto& [key, bones] : dst.limb)
  {
    const auto it = counterpart(key);
    if(it == src.limb.end() || key.first.empty())
      continue;
    // only what the file moves: the limb, or a bone above it the figure has nothing for (the
    // metatarsals above the toes), up to the counterpart of the figure's limb above
    bool moves = std::any_of(it->second.begin(), it->second.end(), [&](int b) { return moved[size_t(b)]; });
    if(!moves)
    {
      // the nearest of the figure's limbs above with a counterpart
      auto upper = src.limb.end();
      for(int above = dst.parent[size_t(bones.front())]; above >= 0 && upper == src.limb.end(); above = dst.parent[size_t(above)])
        upper = counterpart(dst.slotOf[size_t(above)]);
      for(int b = src.parent[size_t(it->second.front())]; b >= 0 && !moves; b = src.parent[size_t(b)])
      {
        if(upper != src.limb.end() && std::find(upper->second.begin(), upper->second.end(), b) != upper->second.end())
          break;
        moves = moved[size_t(b)];
      }
    }
    if(!moves)
      continue;
    Match m;
    m.bones           = bones;
    m.last            = it->second.back();
    const Vec3   dt   = dst.direction(bones), ds = src.direction(it->second);
    const double cosA = dot(dt, ds);
    m.rest            = cosA > std::cos(0.5 * kDeg) ? Quat{} : quatFromUnitVectors(dt, ds);
    // a hinge below it (an elbow) bends about an axis set a little otherwise on each: the limb rolls
    // about its own direction so the two hinges lie alike, or the hinge couldn't reach the source's bend
    for(const std::vector<int>* child : dst.below(bones))
    {
      const auto                                    hinge = dst.hinge(*child);
      const auto                                    other = counterpart(dst.slotOf[size_t(child->front())]);
      const std::optional<Vec3>                     theirs = other != src.limb.end() ? src.hinge(other->second) : std::nullopt;
      if(!hinge || !theirs)
        continue;
      const Vec3   a = across(rotate(*hinge, m.rest), ds), b = across(*theirs, ds);
      const double s = dot(cross(a, b), ds), c = dot(a, b);
      if(dot(a, a) > 0.5 && dot(b, b) > 0.5)
      {
        const double angle = std::atan2(s, c), h = angle / 2;
        m.rest             = (Quat{ds.x * std::sin(h), ds.y * std::sin(h), ds.z * std::sin(h), std::cos(h)} * m.rest).normalized();
      }
      break;
    }
    m.twist           = dst.twistAxis(bones);
    m.dir             = dst.own(bones.front());
    for(int b : bones)
      matchOf[size_t(b)] = int(matches.size());
    matches.push_back(std::move(m));
  }

  // per figure bone, its channels over the keys
  std::vector<std::array<std::vector<double>, 3>> values(tb.size());
  std::vector<std::array<double, 3>>              nearOf(tb.size(), std::array<double, 3>{});
  std::vector<Quat>                               sWorld(sb.size()), tWorld(tb.size()), tLocal(tb.size());
  std::vector<char>                               sDone(sb.size()), tDone(tb.size());
  for(const float t : times)
  {
    std::fill(sDone.begin(), sDone.end(), 0);
    std::fill(tDone.begin(), tDone.end(), 0);
    // every source bone's turn in the body's space, parents first
    const std::function<Quat(int)> sw = [&](int b) -> Quat {
      if(sDone[size_t(b)])
        return sWorld[size_t(b)];
      std::array<double, 3> deg{};
      // as the source figure would show them: within its limits, as setMany keeps them
      for(int a = 0; a < 3; a++)
      {
        const Axis* axis = sb[size_t(b)].axis(ChannelKind::eRotation, a);
        double      v    = turns[size_t(b)][size_t(a)] ? sample(*turns[size_t(b)][size_t(a)], t) : sb[size_t(b)].posed(ChannelKind::eRotation, a);
        if(axis && axis->clamped)
          v = std::clamp(v, axis->min, axis->max);
        deg[size_t(a)] = v;
      }
      const int p             = src.parent[size_t(b)];
      sWorld[size_t(b)]       = (p >= 0 ? sw(p) : Quat{}) * localTurn(sb[size_t(b)], deg);
      sDone[size_t(b)]        = 1;
      return sWorld[size_t(b)];
    };
    // the figure's: a matched limb as the source's, its rest pose turned onto the source's; the rest
    // as they are at rest
    const std::function<Quat(int)> tw = [&](int b) -> Quat {
      if(tDone[size_t(b)])
        return tWorld[size_t(b)];
      const int  p     = dst.parent[size_t(b)];
      const Quat above = p >= 0 ? tw(p) : Quat{};
      const int  mi    = matchOf[size_t(b)];
      if(mi < 0)
      {
        tLocal[size_t(b)] = Quat{};
        tWorld[size_t(b)] = above;
      }
      else if(const Match& m = matches[size_t(mi)]; b == m.bones.front())
      {
        // the limb's whole turn below its parent: on its top bone, or split with its twist bone as
        // DAZ's Euler splits it: the top bone's turn but its twist, then the twist alone
        // A bone lacking an axis takes what points it best, and what lies below makes up the rest.
        const Quat            whole = (above.inverse() * sw(m.last) * m.rest).normalized();
        std::array<double, 3> turn  = to.rotationTotals(tb[size_t(b)], whole, nearOf[size_t(b)]);
        if(m.bones.size() > 1)
        {
          turn[size_t(m.twist)]      = 0;
          tLocal[size_t(b)]          = localTurn(tb[size_t(b)], fitAxes(tb[size_t(b)], whole, m.dir, turn));
          tLocal[size_t(m.bones[1])] = (tLocal[size_t(b)].inverse() * whole).normalized();
        }
        else
          tLocal[size_t(b)] = localTurn(tb[size_t(b)], fitAxes(tb[size_t(b)], whole, m.dir, turn));
        tWorld[size_t(b)] = above * tLocal[size_t(b)];
      }
      else
      {
        // a twist bone below the top: the twist split off it, or none further down
        if(b != m.bones[1])
          tLocal[size_t(b)] = Quat{};
        tWorld[size_t(b)] = above * tLocal[size_t(b)];
      }
      tDone[size_t(b)] = 1;
      return tWorld[size_t(b)];
    };
    for(size_t b = 0; b < tb.size(); b++)
    {
      tw(int(b));
      if(matchOf[b] < 0)
        continue;
      // each key near the last, so a channel never jumps a turn
      nearOf[b] = to.rotationTotals(tb[b], tLocal[b], nearOf[b]);
      for(int a = 0; a < 3; a++)
        values[b][size_t(a)].push_back(nearOf[b][size_t(a)]);
    }
  }

  const char* names[] = {"x", "y", "z"};
  const auto  channel = [&](const PoseBone& pb, ChannelKind kind, int a, std::vector<double> v) {
    if(!pb.axis(kind, a))
      return;
    PoseClip::Channel ch;
    ch.target = {PresetTarget::Kind::eBone, pb.name, kind, names[a], pb.name};
    ch.times  = times;
    ch.values = std::move(v);
    ch.held.assign(times.size(), false);
    out.channels.push_back(std::move(ch));
  };
  for(size_t b = 0; b < tb.size(); b++)
    if(matchOf[b] >= 0)
      for(int a = 0; a < 3; a++)
        channel(tb[b], ChannelKind::eRotation, a, std::move(values[b][size_t(a)]));
  // the hip's move, as far for the figure's height as for the source's
  for(size_t b = 0; b < sb.size(); b++)
  {
    if(!moves[b][0] && !moves[b][1] && !moves[b][2])
      continue;
    const auto& slot = src.slotOf[b];
    auto        it   = dst.limb.find(slot);
    if(srcRoot && dstRoot && slot == *srcRoot)
      it = dst.limb.find(*dstRoot);
    if(it == dst.limb.end())
      continue;
    const PoseBone& target = tb[size_t(it->second.front())];
    const double    hs     = from.scene().nodes[size_t(sb[b].bone)].joint->center[1];
    const double    ht     = to.scene().nodes[size_t(target.bone)].joint->center[1];
    const double    scale  = hs > 1 && ht > 1 ? ht / hs : 1;
    for(int a = 0; a < 3; a++)
    {
      std::vector<double> v;
      for(const float t : times)
        v.push_back(moves[b][size_t(a)] ? sample(*moves[b][size_t(a)], t) * scale : 0.0);
      if(moves[b][size_t(a)])
        channel(target, ChannelKind::eTranslation, a, std::move(v));
    }
  }
  out.duration = times.back();
  out.skipped  = source.skipped;
  return out;
}

}  // namespace fv::pose
