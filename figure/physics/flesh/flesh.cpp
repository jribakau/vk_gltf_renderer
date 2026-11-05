#include <figure/physics/flesh/flesh.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <unordered_map>

#include <figure/materials/procedural.hpp>
#include <figure/physics/collide/cage_graph.hpp>
#include <figure/physics/contact/contact.hpp>
#include <figure/physics/flesh/touch.hpp>
#include <figure/wardrobe/fit/bones.hpp>

namespace fv::physics {

namespace {

// the spring at softness 0+ and 1 (Hz), as the chest's soft bones have it
constexpr double kFirm = 8, kLoose = 1.5;
// how much stiffer an edge keeps two offsets together than a lobe's core is held: about 2.5 cm of
// smoothing on a 1 cm cage
constexpr double kEdgeStiffness = 6;
// the most a frame's change of the skin's velocity is felt (cm/s^2): 5 g, about what a landing jump
// gives the trunk; more is a jump of the pose, which only moves the lobe
constexpr double kMostAcceleration = 4900;
// a frame whose targets moved further than this (cm) starts again at rest
constexpr double kJump = 30;
constexpr double kMaxFrame = 1.0 / 20, kMaxSpeed = 400;
// the limits: how far from the skin (cm), and how far into the body; pressed by what it meets, as far
// as the still contact pass dents
constexpr double kReach = 3, kBackStop = 0.5, kPressDepth = 3;
// the skin still for `kRelease` seconds, what presses the flesh lets go over `kLetGo`
constexpr double kRelease = 1, kLetGo = 0.25;
// in a contact by less than this (cm) is on it: the targets are floats
constexpr double kTouch = 1e-5;
// the skin still for `kFree` seconds, the damping ramps in over `kCalming` to `kCalmRate`; at `kHold`
// it sleeps whatever (physics-plan: "Sleep, one rule", the cloth and flesh constants)
constexpr double kFree = 1, kCalming = 1, kHold = 3, kCalmRate = 6;
// asleep when every offset is under this (cm) and every speed under that (cm/s)
constexpr double kRestOffset = 0.01, kRestSpeed = 0.3;
// a part of a lobe softer than this is not a particle (its rim)
constexpr double kLeastShare = 0.04;

double smooth(double t)
{
  t = std::clamp(t, 0.0, 1.0);
  return t * t * (3 - 2 * t);
}

// A soft patch around a landmark, as procedural layers' patches are made: offset and radii in
// fractions of the figure's height, x mirrored on the right; full inside 0.4 of the radii, none at 1.
struct Patch
{
  const char*           slot;  // sided slots without their "l:"/"r:"
  bool                  sided;
  std::array<double, 3> offset, radius;
  int                   facing;  // 1 the front, -1 the back, 0 all round
};

double patchShare(const Patch& p, const Vec3& centre, double side, double height, const double* at)
{
  const double dx = (at[0] - centre.x) / height - p.offset[0] * side;
  const double dy = (at[1] - centre.y) / height - p.offset[1];
  const double dz = (at[2] - centre.z) / height - p.offset[2];
  const double d  = std::sqrt(dx * dx / (p.radius[0] * p.radius[0]) + dy * dy / (p.radius[1] * p.radius[1]) + dz * dz / (p.radius[2] * p.radius[2]));
  return 1 - smooth((d - 0.4) / 0.6);
}

// How much a vertex faces one way along z: none at the flank, all by 60 degrees round (as the shape
// sliders' facingShare).
double facingShare(double facing)
{
  return smooth(facing / 0.5);
}

struct LobeDef
{
  const char* name;
  FleshPart   part;
  int         side;  // 1 the figure's left (+x), -1 its right, 0 both
  // the pectoral's skin weight, or a patch
  bool  pectoral;
  Patch patch;
};

// The lobes. The breasts are the pectoral bones' own weights, as DAZ painted them; the others are
// patches around the hip joints and the belly's bone, as a woman's or a man's figure has them.
const LobeDef kLobes[] = {
    {"Left breast", FleshPart::eChest, 1, true, {}},
    {"Right breast", FleshPart::eChest, -1, true, {}},
    {"Left glute", FleshPart::eGlutes, 1, false, {"thigh", true, {-0.012, -0.018, -0.05}, {0.045, 0.05, 0.05}, -1}},
    {"Right glute", FleshPart::eGlutes, -1, false, {"thigh", true, {-0.012, -0.018, -0.05}, {0.045, 0.05, 0.05}, -1}},
    {"Belly", FleshPart::eBelly, 0, false, {"spine1", false, {0, -0.008, 0.055}, {0.06, 0.045, 0.05}, 1}},
    {"Left thigh", FleshPart::eThighs, 1, false, {"thigh", true, {0, -0.11, -0.005}, {0.05, 0.085, 0.05}, 0}},
    {"Right thigh", FleshPart::eThighs, -1, false, {"thigh", true, {0, -0.11, -0.005}, {0.05, 0.085, 0.05}, 0}},
};

void add3(double* a, const double* b, double s)
{
  a[0] += b[0] * s, a[1] += b[1] * s, a[2] += b[2] * s;
}
double dot3(const double* a, const double* b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

}  // namespace

const char* fleshPartName(FleshPart part)
{
  switch(part)
  {
    case FleshPart::eChest:
      return "chest";
    case FleshPart::eGlutes:
      return "glutes";
    case FleshPart::eBelly:
      return "belly";
    case FleshPart::eThighs:
      return "thighs";
  }
  return "";
}

FleshModel buildFlesh(const FleshInput& in)
{
  FleshModel   out;
  const size_t n = in.cage.size() / 3;
  out.cageCount  = n;
  if(!n)
    return out;
  const CageGraph graph(in.cage, in.polyStart, in.polyVerts, in.tris);

  // the landmarks, and each skin bone's side if it is a pectoral
  std::vector<std::pair<std::string, Vec3>> bones;
  std::vector<int>                          pectoral(in.bones.size(), 0);
  for(size_t b = 0; b < in.bones.size(); b++)
  {
    const auto& [name, c] = in.bones[b];
    bones.emplace_back(name, Vec3{c[0], c[1], c[2]});
    const wardrobe::ParsedBone parsed = wardrobe::parseBone(name);
    if(parsed.slot == "pectoral")
      pectoral[b] = parsed.side == 'l' ? 1 : parsed.side == 'r' ? -1 : 0;
  }
  const mat::BodyLandmarks body = mat::bodyLandmarks(bones);
  const bool               skin = in.skin && in.skin->start.size() == n + 1;

  // each vertex goes to the lobe it has most of
  std::vector<float>   share(n, 0.f);
  std::vector<uint8_t> lobeOf(n, 0);
  for(size_t v = 0; v < n; v++)
  {
    const double p[3]   = {in.cage[v * 3], in.cage[v * 3 + 1], in.cage[v * 3 + 2]};
    const double facing = graph.normal(v)[2];
    for(size_t l = 0; l < std::size(kLobes); l++)
    {
      const LobeDef& def = kLobes[l];
      double         s   = 0;
      if(def.pectoral)
      {
        if(!skin)
          continue;
        double w = 0;
        for(int32_t k = in.skin->start[v]; k < in.skin->start[v + 1]; k++)
          if(const int32_t b = in.skin->bone[size_t(k)]; b >= 0 && size_t(b) < pectoral.size() && pectoral[size_t(b)] == def.side)
            w += in.skin->weight[size_t(k)];
        s = smooth(w / 0.7);
      }
      else
      {
        if(body.height <= 0)
          continue;
        const std::string slot = def.patch.sided ? std::string(def.side > 0 ? "l:" : "r:") + def.patch.slot : def.patch.slot;
        const auto        it   = body.at.find(slot);
        if(it == body.at.end())
          continue;
        // split at the midline
        if(def.side != 0 && p[0] * def.side < 0)
          continue;
        s = patchShare(def.patch, it->second, def.side != 0 ? def.side : 1, body.height, p);
        if(def.patch.facing)
          s *= facingShare(facing * def.patch.facing);
      }
      if(s > share[v])
      {
        share[v]  = float(s);
        lobeOf[v] = uint8_t(l);
      }
    }
  }

  std::vector<int32_t> particleOf(n, -1);
  for(size_t v = 0; v < n; v++)
    if(share[v] > kLeastShare)
    {
      particleOf[v] = int32_t(out.vertex.size());
      out.vertex.push_back(uint32_t(v));
      out.share.push_back(share[v]);
      out.lobe.push_back(lobeOf[v]);
    }
  if(out.vertex.empty())
    return out;
  // the lobes that have particles, renumbered
  std::vector<int> lobeIndex(std::size(kLobes), -1);
  for(size_t p = 0; p < out.vertex.size(); p++)
  {
    int& li = lobeIndex[out.lobe[p]];
    if(li < 0)
    {
      li = int(out.lobes.size());
      out.lobes.push_back({kLobes[out.lobe[p]].name, kLobes[out.lobe[p]].part, {}});
    }
    out.lobes[size_t(li)].particles.push_back(uint32_t(p));
    out.lobe[p] = uint8_t(li);
  }
  // heavier where the cage is coarser, so a lobe answers alike however it is meshed
  double mean = 0;
  for(uint32_t v : out.vertex)
    mean += graph.area(v);
  mean /= double(out.vertex.size());
  for(uint32_t v : out.vertex)
    out.weight.push_back(graph.area(v) > 0 ? std::clamp(mean / graph.area(v), 0.25, 4.0) : 1.0);

  for(size_t p = 0; p < out.vertex.size(); p++)
    for(const uint32_t u : graph.next(out.vertex[p]))
    {
      if(const int32_t q = particleOf[u]; q >= 0)
      {
        if(size_t(q) > p)
          out.edges.push_back({uint32_t(p), uint32_t(q)});
      }
      else
        out.held.push_back(uint32_t(p));
    }
  // the triangles around the particles, over the particles and the vertices next to them (support)
  std::unordered_map<uint32_t, uint32_t> localOf;
  for(size_t p = 0; p < out.vertex.size(); p++)
    localOf.emplace(out.vertex[p], uint32_t(p));
  for(size_t t = 0; t + 2 < in.tris.size(); t += 3)
  {
    const uint32_t c[3] = {in.tris[t], in.tris[t + 1], in.tris[t + 2]};
    if(c[0] >= n || c[1] >= n || c[2] >= n || (particleOf[c[0]] < 0 && particleOf[c[1]] < 0 && particleOf[c[2]] < 0))
      continue;
    for(uint32_t v : c)
    {
      auto [it, added] = localOf.emplace(v, uint32_t(out.vertex.size() + out.support.size()));
      if(added)
        out.support.push_back(v);
      out.tris.push_back(it->second);
    }
  }
  return out;
}

FleshSolver::FleshSolver(FleshModel model)
    : m_model(std::move(model))
{
  const size_t n = m_model.vertex.size();
  m_d.assign(n * 3, 0.0);
  m_e.assign(n * 3, 0.0);
  m_u.assign(n * 3, 0.0);
  m_an.assign(n * 3, 0.0);
  m_normal.assign(n * 3, 0.0);
  m_pressed.assign(n, 0);
}

void FleshSolver::setContacts(std::vector<FleshContact> contacts, double bulge)
{
  m_contacts = std::move(contacts);
  m_bulge    = std::clamp(bulge, 0.0, 1.0);
}

double FleshSolver::omega(size_t lobe) const
{
  const double s = std::clamp(m_settings.softness[size_t(m_model.lobes[lobe].part)], 0.0, 1.0);
  return 2 * std::numbers::pi * (kFirm + (kLoose - kFirm) * s);
}

double FleshSolver::largest() const
{
  double most = 0;
  for(size_t i = 0; i + 2 < m_d.size(); i += 3)
    most = std::max(most, dot3(&m_d[i], &m_d[i]));
  return std::sqrt(most);
}

void FleshSolver::surface(std::span<const float> targets)
{
  std::fill(m_an.begin(), m_an.end(), 0.0);
  const size_t n = m_model.vertex.size();
  const auto&  t = m_model.tris;
  for(size_t k = 0; k + 2 < t.size(); k += 3)
  {
    const float* a  = &targets[t[k] * 3];
    const float* b  = &targets[t[k + 1] * 3];
    const float* c  = &targets[t[k + 2] * 3];
    const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
    const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
    // the cross product is twice the face's area along its normal: a sixth of it to each corner
    const double x[3] = {(u[1] * v[2] - u[2] * v[1]) / 6, (u[2] * v[0] - u[0] * v[2]) / 6, (u[0] * v[1] - u[1] * v[0]) / 6};
    for(int j = 0; j < 3; j++)
      if(t[k + size_t(j)] < n)
        add3(&m_an[t[k + size_t(j)] * 3], x, 1);
  }
  for(size_t i = 0; i < n; i++)
  {
    const double l = std::sqrt(dot3(&m_an[i * 3], &m_an[i * 3]));
    for(int a = 0; a < 3; a++)
      m_normal[i * 3 + size_t(a)] = l > 0 ? m_an[i * 3 + size_t(a)] / l : (a == 1 ? 1 : 0);
  }
}

void FleshSolver::step(std::span<const float> targets, double seconds, bool reshaped)
{
  // the contacts are this step's
  struct Forget
  {
    std::vector<FleshContact>& contacts;
    ~Forget() { contacts.clear(); }
  } forget{m_contacts};
  const size_t n = m_model.vertex.size();
  if(!n || targets.size() != (n + m_model.support.size()) * 3)
    return;
  const auto atRest = [&] {
    std::fill(m_d.begin(), m_d.end(), 0.0);
    std::fill(m_e.begin(), m_e.end(), 0.0);
    std::fill(m_u.begin(), m_u.end(), 0.0);
    std::fill(m_pressed.begin(), m_pressed.end(), uint8_t(0));
    m_last.assign(targets.begin(), targets.end());
    m_carry  = 0;
    m_asleep = true;
    m_knowsU = false;
  };
  if(!m_started)
  {
    atRest();
    m_started = true;
    m_still   = 0;
    return;
  }
  if(seconds <= 0)
  {
    // no time: where the skin went is a re-placement
    m_last.assign(targets.begin(), targets.end());
    return;
  }
  bool moved = false;
  if(reshaped)
  {
    // a change of shape moves the lobes with their skin and isn't felt; the skin's velocity is
    // taken to go on as it was. A pressed particle keeps its place: its skin moving under it is the
    // still pass's dent fading in, which takes the press over from it
    if(m_last.size() == targets.size())
      for(size_t i = 0; i < n; i++)
        if(m_pressed[i])
          for(size_t a = 0; a < 3; a++)
            m_d[i * 3 + a] -= double(targets[i * 3 + a]) - m_last[i * 3 + a];
    m_last.assign(targets.begin(), targets.end());
    m_still += seconds;
    if(m_asleep)
      return;
  }
  else
  {
    double most = 0;
    for(size_t i = 0; i < n * 3; i += 3)
    {
      const double dx = double(targets[i]) - m_last[i], dy = double(targets[i + 1]) - m_last[i + 1], dz = double(targets[i + 2]) - m_last[i + 2];
      most = std::max(most, dx * dx + dy * dy + dz * dz);
    }
    most = std::sqrt(most);
    if(most > kJump)
    {
      atRest();
      m_still = 0;
      return;
    }
    moved   = most > 1e-5;
    m_still = moved ? 0 : m_still + seconds;
    if(m_asleep && !moved)
    {
      // still and asleep: the skin's velocity is known, 0
      m_last.assign(targets.begin(), targets.end());
      m_knowsU = true;
      return;
    }
    m_asleep = false;
    // what the flesh feels: the change of the skin's velocity since the last frame, up to what a body
    // can do (none the frame after a start, which only learns the velocity)
    const double felt = kMostAcceleration * std::min(seconds, kMaxFrame);
    for(size_t i = 0; i < n * 3; i += 3)
    {
      double du[3];
      for(int a = 0; a < 3; a++)
      {
        const double u = (double(targets[i + size_t(a)]) - m_last[i + size_t(a)]) / seconds;
        du[a]          = u - m_u[i + size_t(a)];
        m_u[i + size_t(a)] = u;
      }
      const double l = std::sqrt(dot3(du, du));
      if(m_knowsU)
        add3(&m_e[i], du, l > felt ? -felt / l : -1.0);
    }
    m_knowsU = true;
    m_last.assign(targets.begin(), targets.end());
  }
  surface(targets);
  // a frame past the most simulated (a hitch) only moves the lobes with their skin, which in offsets is nothing
  m_carry += std::min(seconds, kMaxFrame);
  const int steps = int(m_carry / kSubstep);
  m_carry -= steps * kSubstep;
  const double calm = smooth((m_still - kFree) / kCalming);
  const double hold = 1 - smooth((m_still - kRelease) / kLetGo);
  for(int k = 0; k < steps; k++)
    substep(kSubstep, calm, hold);

  bool still = !moved;
  for(size_t i = 0; still && i < n * 3; i += 3)
    still = dot3(&m_d[i], &m_d[i]) < kRestOffset * kRestOffset && dot3(&m_e[i], &m_e[i]) < kRestSpeed * kRestSpeed;
  if(still || m_still >= kHold)
  {
    std::fill(m_d.begin(), m_d.end(), 0.0);
    std::fill(m_e.begin(), m_e.end(), 0.0);
    std::fill(m_u.begin(), m_u.end(), 0.0);
    std::fill(m_pressed.begin(), m_pressed.end(), uint8_t(0));
    m_asleep = true;
  }
}

void FleshSolver::substep(double h, double calm, double hold)
{
  const size_t n  = m_model.vertex.size();
  const size_t nl = m_model.lobes.size();
  const double z  = std::clamp(m_settings.damping, 0.02, 1.0);
  // per lobe: its spring, its damping over the substep, and whether it is held
  std::vector<double> w2h2(nl), damp(nl);
  std::vector<char>   held(nl);
  for(size_t l = 0; l < nl; l++)
  {
    const double w = omega(l);
    w2h2[l]        = w * w * h * h;
    damp[l]        = std::exp(-(2 * z * w + calm * kCalmRate) * h);
    held[l]        = m_settings.softness[size_t(m_model.lobes[l].part)] <= 0;
  }
  const auto free = [&](size_t i) { return !held[m_model.lobe[i]]; };
  std::vector<double> before(m_d);
  for(size_t i = 0; i < n; i++)
  {
    double* e = &m_e[i * 3];
    double* d = &m_d[i * 3];
    if(!free(i))
    {
      std::fill(d, d + 3, 0.0);
      std::fill(e, e + 3, 0.0);
      continue;
    }
    const double k = damp[m_model.lobe[i]];
    e[0] *= k, e[1] *= k, e[2] *= k;
    if(const double s = std::sqrt(dot3(e, e)); s > kMaxSpeed)
      for(int a = 0; a < 3; a++)
        e[a] *= kMaxSpeed / s;
    add3(d, e, h);
    // held to its target, as soft as its place in its lobe: one XPBD step toward 0 at compliance
    // share / (w omega^2), its inverse mass cancelling
    d[0] /= 1 + w2h2[m_model.lobe[i]] / double(m_model.share[i]);
    d[1] /= 1 + w2h2[m_model.lobe[i]] / double(m_model.share[i]);
    d[2] /= 1 + w2h2[m_model.lobe[i]] / double(m_model.share[i]);
  }
  // the edges keep their ends' offsets together; one to a held vertex holds its end toward 0
  for(const auto& [p, q] : m_model.edges)
  {
    const double wp = free(p) ? m_model.weight[p] : 0, wq = free(q) ? m_model.weight[q] : 0;
    if(wp + wq <= 0)
      continue;
    const double a = 1 / (kEdgeStiffness * std::max(w2h2[m_model.lobe[p]], w2h2[m_model.lobe[q]]));
    const double s = 1 / (wp + wq + a);
    double       c[3];
    for(int k = 0; k < 3; k++)
      c[k] = m_d[p * 3 + size_t(k)] - m_d[q * 3 + size_t(k)];
    add3(&m_d[p * 3], c, -wp * s);
    add3(&m_d[q * 3], c, wq * s);
  }
  for(const uint32_t p : m_model.held)
  {
    if(!free(p))
      continue;
    const double wp = m_model.weight[p];
    const double a  = 1 / (kEdgeStiffness * w2h2[m_model.lobe[p]]);
    const double s  = wp / (wp + a);
    for(int k = 0; k < 3; k++)
      m_d[p * 3 + size_t(k)] -= m_d[p * 3 + size_t(k)] * s;
  }
  // each lobe keeps its volume against the skin: sum A n . d = 0, or where it is pressed in, the
  // contact's share of what that pushed in comes back around it: sum A n . d = -(1 - beta) V
  for(const FleshModel::Lobe& lobe : m_model.lobes)
  {
    double c = 0, den = 0, dent = 0;
    for(const uint32_t i : lobe.particles)
      if(free(i))
      {
        const double an = dot3(&m_an[i * 3], &m_d[i * 3]);
        c += an;
        den += m_model.weight[i] * dot3(&m_an[i * 3], &m_an[i * 3]);
        if(m_pressed[i])
          dent += std::max(0.0, -an);
      }
    if(den <= 0)
      continue;
    c += (1 - m_bulge) * dent;
    for(const uint32_t i : lobe.particles)
      if(free(i))
        add3(&m_d[i * 3], &m_an[i * 3], -m_model.weight[i] * c / den);
  }
  // what it meets: in front of each plane, less what it may be in, and once the skin has been still
  // a while (`hold` toward 0) as deep as its skin is
  std::vector<uint8_t> pressed(n, 0);
  for(const FleshContact& k : m_contacts)
  {
    const size_t i = k.particle;
    if(i >= n || !free(i))
      continue;
    const float* s      = &m_last[i * 3];
    double*      d      = &m_d[i * 3];
    const double rel[3] = {s[0] - k.point[0], s[1] - k.point[1], s[2] - k.point[2]};
    const double skin   = -dot3(rel, k.normal.data());
    const double allow  = k.allow + (1 - hold) * std::max(0.0, skin);
    const double gap    = dot3(rel, k.normal.data()) + dot3(d, k.normal.data()) + allow;
    if(gap < -kTouch)
      add3(d, k.normal.data(), -gap);
    if(gap < -kTouch || skin > allow + kTouch)
      pressed[i] = 1;
  }
  // the limits, last: no further than its reach, nor further into the body than the back stop (or
  // than a press may take it)
  for(size_t i = 0; i < n; i++)
  {
    double*       d    = &m_d[i * 3];
    const double* nn   = &m_normal[i * 3];
    const double  stop = pressed[i] ? kPressDepth : kBackStop;
    if(const double into = dot3(d, nn); into < -stop)
      add3(d, nn, -stop - into);
    if(const double l = std::sqrt(dot3(d, d)); l > kReach)
      for(int a = 0; a < 3; a++)
        d[a] *= kReach / l;
  }
  for(size_t i = 0; i < n * 3; i++)
    m_e[i] = (m_d[i] - before[i]) / h;
  m_pressed.swap(pressed);
}

Flesh::Flesh(const SceneGraph& scene, const morph::MorphSet& morphs, int bodyMesh, int root)
    : m_scene(scene)
    , m_morphs(morphs)
    , m_bodyMesh(bodyMesh)
    , m_root(root)
{
  if(bodyMesh < 0 || size_t(bodyMesh) >= scene.meshes.size())
    return;
  const Mesh& body = scene.meshes[size_t(bodyMesh)];
  if(!body.geo || body.geo->cagePositions.empty() || !body.skinned() || body.cageSkin.start.size() != body.geo->cagePositions.size() / 3 + 1)
    return;
  FleshInput in;
  in.cage      = body.geo->cagePositions;
  in.tris      = body.geo->dsonTris;
  in.polyStart = body.geo->cagePolyStart;
  in.polyVerts = body.geo->cagePolyVerts;
  in.skin      = &body.cageSkin;
  for(const int id : body.bones)
  {
    const Node& node = scene.nodes[size_t(id)];
    in.bones.emplace_back(node.name, node.joint ? node.joint->centerAuthored : std::array<double, 3>{});
  }
  m_skin   = dforce::cageSkin(body.cageSkin);
  m_solver = std::make_unique<FleshSolver>(buildFlesh(in));
  m_displacement.assign(body.geo->cagePositions.size(), 0.f);
}

Flesh::~Flesh() = default;

void Flesh::set(const FleshSettings& settings)
{
  if(settings == m_settings)
    return;
  if(settings.on && !m_settings.on && m_solver)
    m_solver->reset();
  m_settings = settings;
  if(m_solver)
    m_solver->set(settings);
  m_version++;
}

bool Flesh::step(double seconds)
{
  const auto wake = [&] {
    if(awake() != m_wasAwake)
    {
      m_wasAwake = awake();
      m_version++;
    }
  };
  const auto clear = [&] {
    if(!m_written)
      return false;
    std::fill(m_displacement.begin(), m_displacement.end(), 0.f);
    m_written = false;
    return true;
  };
  if(!m_solver || !valid() || !m_settings.on)
  {
    wake();
    return clear();
  }
  const Mesh& body = m_scene.meshes[size_t(m_bodyMesh)];
  const FleshModel& model = m_solver->model();
  if(!body.geo || body.geo->cagePositions.size() != model.cageCount * 3)
  {
    wake();
    return clear();
  }
  // the lobes' targets: their cage as the morphs have it, skinned as the GPU skins it
  const std::span<const float> cage = m_morphs.morphedCageView(m_bodyMesh, m_scratch);
  const auto                   worlds = m_scene.worlds();
  const auto                   nodeOf = dforce::nodeOfMesh(m_scene);
  std::vector<uint32_t>        list(model.vertex);
  list.insert(list.end(), model.support.begin(), model.support.end());
  const std::vector<float> targets = dforce::skinLocal(cage, m_skin, dforce::currentSkin(m_scene, worlds, nodeOf, m_bodyMesh), list);
  // a dial that moved the lobes' cage (a breast's size) reshapes them: no motion to feel
  std::vector<float> rest;
  rest.reserve(list.size() * 3);
  for(const uint32_t v : list)
    rest.insert(rest.end(), {cage[v * 3], cage[v * 3 + 1], cage[v * 3 + 2]});
  const bool reshaped = !m_rest.empty() && rest != m_rest;
  m_rest              = std::move(rest);
  // what it meets while it moves, in the targets' space: the mesh's posed space
  if(colliders && (!m_solver->asleep() || targets != m_targets))
  {
    const FleshColliders with = colliders();
    if(!with.sets.empty() || !with.limbs.empty())
    {
      if(!m_touch)
      {
        // each particle's part of the body and how far it is from the part's edge, as self-contact has them
        const std::vector<uint8_t> parts = bodyParts(m_scene, body);
        const PartEdges            edges = partEdges(cage, body.geo->cagePolyStart, body.geo->cagePolyVerts, body.geo->dsonTris, parts);
        std::vector<uint8_t>       part;
        std::vector<float>         fromEdge, at;
        for(const uint32_t v : model.vertex)
        {
          part.push_back(v < parts.size() ? parts[v] : uint8_t(0));
          fromEdge.push_back(v < edges.fromEdge.size() ? edges.fromEdge[v] : 0.f);
          at.insert(at.end(), {cage[v * 3], cage[v * 3 + 1], cage[v * 3 + 2]});
        }
        m_touch = std::make_unique<FleshTouch>(std::move(part), std::move(fromEdge), edges.joined, std::move(at));
      }
      const int  node   = nodeOf[size_t(m_bodyMesh)];
      const Mat4 toRoot = (m_root >= 0 ? inverse(worlds[size_t(m_root)]) : Mat4{}) * (node >= 0 ? worlds[size_t(node)] : Mat4{});
      m_solver->setContacts(m_touch->find(targets, m_solver->offsets(), with, inverse(toRoot)), with.bulge);
    }
  }
  m_targets = targets;
  m_solver->step(targets, seconds, reshaped);
  wake();
  if(m_solver->asleep())
    return clear();
  const std::span<const double> d = m_solver->offsets();
  for(size_t p = 0; p < model.vertex.size(); p++)
    for(int a = 0; a < 3; a++)
      m_displacement[model.vertex[p] * 3 + size_t(a)] = float(d[p * 3 + size_t(a)]);
  m_written = true;
  return true;
}

}  // namespace fv::physics
