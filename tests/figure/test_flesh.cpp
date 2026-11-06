// Soft flesh (docs/physics-plan.md steps 8, 9 and 11): the lobes found on a body, their solve against
// the skin, the skin the targets come from against the GPU's, what is worn over it following it, and
// what it meets while it moves.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <figure/physics/skin.hpp>
#include <figure/physics/flesh/flesh.hpp>
#include <figure/physics/contact/contact.hpp>
#include <figure/physics/flesh/follow.hpp>
#include <figure/physics/flesh/touch.hpp>

#include "../common/skin_reference.hpp"

using namespace fv::physics;

namespace {

// A chest as a 21 x 13 grid of 1 cm quads in the XY plane, facing +z: the left breast painted on its
// pectoral at x = +5, the right at x = -5, 3.5 cm across, the rest on the chest's bone.
struct Chest
{
  int                   nx = 21, ny = 13;
  std::vector<float>    cage;
  std::vector<uint32_t> tris;
  std::vector<int32_t>  polyStart, polyVerts;
  fv::mesh::SkinInput   skin;
  FleshInput            in;

  // `dome`: each breast raised to that height at its middle, as a breast is (cm)
  explicit Chest(float dome = 0)
  {
    skin.start.push_back(0);
    for(int y = 0; y < ny; y++)
      for(int x = 0; x < nx; x++)
      {
        const float px = float(x - nx / 2), py = float(y - ny / 2);
        const auto  paint = [&](float cx) {
          const float r = std::hypot(px - cx, py) / 3.5f;
          return r < 1 ? 1 - r * r : 0.f;
        };
        const float l = paint(5), r = paint(-5);
        cage.insert(cage.end(), {px, py, dome * (l + r)});
        skin.bone.insert(skin.bone.end(), {0, 1, 2});
        skin.weight.insert(skin.weight.end(), {1 - l - r, l, r});
        skin.start.push_back(int32_t(skin.bone.size()));
      }
    for(int y = 0; y + 1 < ny; y++)
      for(int x = 0; x + 1 < nx; x++)
      {
        const uint32_t a = uint32_t(y * nx + x), b = a + 1, c = a + uint32_t(nx), d = c + 1;
        tris.insert(tris.end(), {a, b, d, a, d, c});
        polyStart.push_back(int32_t(polyVerts.size()));
        polyVerts.insert(polyVerts.end(), {int32_t(a), int32_t(b), int32_t(d), int32_t(c)});
      }
    polyStart.push_back(int32_t(polyVerts.size()));
    in.cage      = cage;
    in.tris      = tris;
    in.polyStart = polyStart;
    in.polyVerts = polyVerts;
    in.skin      = &skin;
    in.bones     = {{"spine3", {0, 0, -10}}, {"l_pectoral", {5, 0, -3}}, {"r_pectoral", {-5, 0, -3}}};
  }
};

// The model's targets (particles, then support vertices) from the cage moved by `shift` where `moves`
// says (all of it by default).
std::vector<float> targetsOf(const Chest& chest, const FleshModel& m, const glm::dvec3& shift, bool (*moves)(float x) = nullptr)
{
  std::vector<float> out;
  const auto         add = [&](uint32_t v) {
    const float* p    = &chest.cage[v * 3];
    const bool   take = !moves || moves(p[0]);
    for(int a = 0; a < 3; a++)
      out.push_back(float(p[a] + (take ? shift[a] : 0.0)));
  };
  for(uint32_t v : m.vertex)
    add(v);
  for(uint32_t v : m.support)
    add(v);
  return out;
}

FleshSettings on()
{
  FleshSettings s;
  s.on = true;
  return s;
}

const FleshModel::Lobe* lobeNamed(const FleshModel& m, const char* name)
{
  for(const FleshModel::Lobe& l : m.lobes)
    if(l.name == name)
      return &l;
  return nullptr;
}

double largestIn(const FleshSolver& solver, const FleshModel::Lobe& lobe)
{
  double most = 0;
  for(uint32_t p : lobe.particles)
  {
    const double* d = &solver.offsets()[p * 3];
    most            = std::max(most, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
  }
  return most;
}

}  // namespace

// Each breast is its pectoral's weights, split at the midline: two lobes, nothing joining them
TEST(Soft, eachBreastIsItsPectoralsWeights)
{
  Chest            chest;
  const FleshModel m = buildFlesh(chest.in);
  ASSERT_EQ(m.lobes.size(), 2u);
  const FleshModel::Lobe* left  = lobeNamed(m, "Left breast");
  const FleshModel::Lobe* right = lobeNamed(m, "Right breast");
  ASSERT_TRUE(left && right);
  for(uint32_t p : left->particles)
    EXPECT_GT(chest.cage[m.vertex[p] * 3], 0.f);
  for(uint32_t p : right->particles)
    EXPECT_LT(chest.cage[m.vertex[p] * 3], 0.f);
  for(const auto& [p, q] : m.edges)
    EXPECT_EQ(m.lobe[p], m.lobe[q]);
  // the core is all soft
  float most = 0;
  for(float s : m.share)
    most = std::max(most, s);
  EXPECT_FLOAT_EQ(most, 1.f);
  EXPECT_FALSE(m.held.empty());
  EXPECT_FALSE(m.support.empty());
}

TEST(Soft, aStillFigureDoesNotMove)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  const std::vector<float> still = targetsOf(chest, solver.model(), {0, 0, 0});
  for(int f = 0; f < 120; f++)
  {
    solver.step(still, 1.0 / 60);
    ASSERT_TRUE(solver.asleep()) << f;
    ASSERT_EQ(solver.largest(), 0.0);
  }
}

// A chest moving up and stopping: the breasts trail it, then overshoot past where it stopped, and
// come back to exactly the skinned shape, asleep
TEST(Soft, aStoppingChestOvershootsThenDecaysToTheSkinnedShape)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  const FleshModel& m = solver.model();
  double            y = 0;
  for(int f = 0; f < 30; f++)  // 0.5 s at 60 cm/s
  {
    y += 1;
    solver.step(targetsOf(chest, m, {0, y, 0}), 1.0 / 60);
  }
  // stopped: the breasts swing on up past the skin, then back
  double highest = 0, lowest = 0;
  int    asleepAt = -1;
  for(int f = 0; f < 300; f++)
  {
    solver.step(targetsOf(chest, m, {0, y, 0}), 1.0 / 60);
    double most = 0;
    for(const FleshModel::Lobe& l : m.lobes)
      for(uint32_t p : l.particles)
        most = std::abs(solver.offsets()[p * 3 + 1]) > std::abs(most) ? solver.offsets()[p * 3 + 1] : most;
    highest = std::max(highest, most);
    lowest  = std::min(lowest, most);
    if(solver.asleep() && asleepAt < 0)
      asleepAt = f;
  }
  EXPECT_GT(highest, 0.2) << "the overshoot";
  EXPECT_LT(lowest, -0.02) << "and the swing back";
  ASSERT_GE(asleepAt, 0);
  EXPECT_LE(asleepAt, 3 * 60 + 2) << "asleep within the hold";
  EXPECT_EQ(solver.largest(), 0.0);
}

// A lobe carried along with its skin at a steady 600 cm/s is on it: no motion of its own
TEST(Soft, aLobeMovingWithItsSkinAt600CmPerSecondStaysOnIt)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  double x = 0;
  for(int f = 0; f < 90; f++)
  {
    solver.step(targetsOf(chest, solver.model(), {x, 0, 0}), 1.0 / 60);
    x += 10;
    ASSERT_LT(solver.largest(), 0.01) << f;
  }
}

// Frames of 16 to 35 ms, the skin at a steady speed: nothing moves of its own, whatever the frames
TEST(Soft, aConstantVelocityTargetUnderJitteredFramesLeavesNoDisplacement)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  const double frames[] = {0.016, 0.035, 0.021, 0.017, 0.033, 0.028, 0.016, 0.019, 0.034, 0.025};
  double       t        = 0;
  for(int f = 0; f < 120; f++)
  {
    const double dt = frames[f % std::size(frames)];
    t += dt;
    solver.step(targetsOf(chest, solver.model(), {0, 200 * t, 50 * t}), dt);
    ASSERT_LT(solver.largest(), 0.01) << f;
  }
}

// A hitch (a quarter of a second in one frame) at a steady speed moves the lobe with its skin, no kick
TEST(Soft, aHitchMovesTheLobeWithoutAKick)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  double t = 0;
  for(int f = 0; f < 60; f++)
  {
    const double dt = f == 30 ? 0.25 : 1.0 / 60;
    t += dt;
    solver.step(targetsOf(chest, solver.model(), {0, 80 * t, 0}), dt);
    ASSERT_LT(solver.largest(), 0.01) << f;
  }
}

// While it swings each lobe keeps its volume against the skin: what moves out is matched by what moves in
TEST(Soft, volumeWithinFivePercent)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  const FleshModel& m = solver.model();
  // pushed back into the chest by a sudden start toward +z
  double z = 0;
  solver.step(targetsOf(chest, m, {0, 0, z}), 1.0 / 60);
  solver.step(targetsOf(chest, m, {0, 0, z}), 1.0 / 60);
  for(int f = 0; f < 40; f++)
  {
    z += f < 3 ? 2.0 : 0.0;
    solver.step(targetsOf(chest, m, {0, 0, z}), 1.0 / 60);
    for(const FleshModel::Lobe& l : m.lobes)
    {
      double net = 0, moved = 0;
      for(uint32_t p : l.particles)
      {
        const double* an = &solver.areaNormals()[p * 3];
        const double* d  = &solver.offsets()[p * 3];
        net += an[0] * d[0] + an[1] * d[1] + an[2] * d[2];
        moved += std::sqrt(an[0] * an[0] + an[1] * an[1] + an[2] * an[2]) * std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      }
      if(moved > 1e-6)
        EXPECT_LE(std::abs(net), 0.05 * moved) << l.name << " frame " << f;
    }
  }
}

// One breast shaken, the other still: they are lobes of their own, so it doesn't move or swell
TEST(Soft, oneBreastMovingLeavesTheOtherStill)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  const FleshModel&       m     = solver.model();
  const FleshModel::Lobe& left  = *lobeNamed(m, "Left breast");
  const FleshModel::Lobe& right = *lobeNamed(m, "Right breast");
  const auto              leftHalf = [](float x) { return x > 0.f; };
  double                  y        = 0;
  double                  swung    = 0;
  for(int f = 0; f < 60; f++)
  {
    y = f < 10 ? y + 1.5 : y;
    solver.step(targetsOf(chest, m, {0, y, 0}, leftHalf), 1.0 / 60);
    swung = std::max(swung, largestIn(solver, left));
    ASSERT_EQ(largestIn(solver, right), 0.0) << f;
  }
  EXPECT_GT(swung, 0.1);
}

// However hard the chest is driven, the flesh it leaves behind stays out of the body: at most 0.5 cm
// in along the skin's normal. A steady 4 g sideways sinks a breast's leading side as its other swells
// (it keeps its volume, so it can't sink all over).
TEST(Soft, theBackStopKeepsSkinOutOfTheBody)
{
  Chest       chest(2.5f);
  FleshSolver solver(buildFlesh(chest.in));
  FleshSettings s = on();
  s.softness      = {1, 1, 1, 1};
  s.damping       = 0.05;
  solver.set(s);
  const FleshModel& m = solver.model();
  double            deepest = 0;
  solver.step(targetsOf(chest, m, {0, 0, 0}), 1.0 / 60);
  solver.step(targetsOf(chest, m, {0, 0, 0}), 1.0 / 60);
  for(int f = 1; f <= 15; f++)
  {
    const double t = f / 60.0;
    solver.step(targetsOf(chest, m, {0.5 * 4000 * t * t, 0, 0}), 1.0 / 60);
    for(size_t p = 0; p < m.vertex.size(); p++)
    {
      const double* n    = &solver.normals()[p * 3];
      const double* d    = &solver.offsets()[p * 3];
      const double  into = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
      deepest            = std::min(deepest, into);
      ASSERT_GE(into, -0.5 - 1e-9) << f;
    }
  }
  EXPECT_LT(deepest, -0.45) << "it reached the stop";
}

// Off a lobe (softness 0) holds to its skin while the others move
TEST(Soft, aPartAtNoSoftnessHoldsToItsSkin)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  FleshSettings s = on();
  s.softness[size_t(FleshPart::eChest)] = 0;
  solver.set(s);
  double y = 0;
  for(int f = 0; f < 40; f++)
  {
    y += f < 10 ? 1 : 0;
    solver.step(targetsOf(chest, solver.model(), {0, y, 0}), 1.0 / 60);
    ASSERT_EQ(solver.largest(), 0.0) << f;
  }
}

// The lobes' targets are skinned as the GPU skins the cage: eight influences, and the dual
// quaternion share, within a fiftieth of a millimetre (floats against doubles)
TEST(Soft, subsetSkinMatchesTheGpuSkinReference)
{
  // three joints turned and moved, a vertex on six of their influences
  std::vector<glm::mat4> joints;
  for(int j = 0; j < 6; j++)
  {
    glm::mat4 m = glm::translate(glm::mat4(1.f), glm::vec3(float(j) * 3.f, 10.f - float(j), float(j % 2) * 4.f));
    m           = glm::rotate(m, glm::radians(20.f * float(j + 1)), glm::normalize(glm::vec3(1.f, float(j), 2.f)));
    joints.push_back(m);
  }
  fv::physics::SkinPose pose;
  pose.bind = pose.bindInverse = fv::Mat4{};
  for(const glm::mat4& j : joints)
  {
    fv::Mat4 m;
    for(int c = 0; c < 4; c++)
      for(int r = 0; r < 4; r++)
        m.e[size_t(c * 4 + r)] = j[c][r];
    pose.bones.push_back(m);
  }
  fv::physics::CageSkin skin;
  const int            n = 40;
  std::vector<float>   rest;
  for(int v = 0; v < n; v++)
  {
    rest.insert(rest.end(), {float(v % 7) * 5.f - 15.f, float(v) * 2.f, float(v % 3) * 4.f - 4.f});
    float sum = 0;
    float w[8];
    for(int k = 0; k < 8; k++)
    {
      w[k] = k < 6 ? float((v + k * 3) % 5 + 1) : 0.f;
      sum += w[k];
    }
    for(int k = 0; k < 8; k++)
    {
      skin.index8.push_back(uint16_t(k < 6 ? (v + k) % 6 : 0));
      skin.weight8.push_back(w[k] / sum);
    }
    skin.dualQuatShare.push_back(float(v % 4) / 3.f);
  }
  skin.index.assign(size_t(n) * 4, 0);
  skin.weight.assign(size_t(n) * 4, 0.f);
  std::vector<uint32_t> some;
  for(uint32_t v = 1; v < uint32_t(n); v += 3)
    some.push_back(v);
  const std::vector<float> cpu = fv::physics::skinLocal(rest, skin, pose, some);

  std::vector<glm::mat3>         normalMats;
  std::vector<rs::JointDualQuat> dqs;
  for(const glm::mat4& j : joints)
  {
    normalMats.push_back(glm::mat3(1.f));
    dqs.push_back(rs::jointDualQuat(j));
  }
  for(size_t k = 0; k < some.size(); k++)
  {
    const uint32_t   v = some[k];
    const glm::ivec4 ij[2] = {{skin.index8[v * 8], skin.index8[v * 8 + 1], skin.index8[v * 8 + 2], skin.index8[v * 8 + 3]},
                              {skin.index8[v * 8 + 4], skin.index8[v * 8 + 5], skin.index8[v * 8 + 6], skin.index8[v * 8 + 7]}};
    const glm::vec4  iw[2] = {{skin.weight8[v * 8], skin.weight8[v * 8 + 1], skin.weight8[v * 8 + 2], skin.weight8[v * 8 + 3]},
                              {skin.weight8[v * 8 + 4], skin.weight8[v * 8 + 5], skin.weight8[v * 8 + 6], skin.weight8[v * 8 + 7]}};
    const auto       gpu = rs::skinVertex({rest[v * 3], rest[v * 3 + 1], rest[v * 3 + 2]}, nullptr, nullptr, ij, iw, {joints, normalMats, dqs}, skin.dualQuatShare[v]);
    for(int a = 0; a < 3; a++)
      EXPECT_NEAR(cpu[k * 3 + size_t(a)], gpu.position[a], 2e-3) << "vertex " << v << " axis " << a;
  }
}

// A dial that reshapes the lobes (a breast's size) moves them with their skin: no jiggle as it moves
// or when it stops
TEST(Soft, aShapeDialMovesTheLobesWithoutAJiggle)
{
  Chest       chest;
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  solver.step(targetsOf(chest, solver.model(), {0, 0, 0}), 1.0 / 60);
  solver.step(targetsOf(chest, solver.model(), {0, 0, 0}), 1.0 / 60);
  double z = 0;
  for(int f = 0; f < 60; f++)
  {
    z += f < 20 ? 0.5 : 0;
    solver.step(targetsOf(chest, solver.model(), {0, 0, z}), 1.0 / 60, f < 20);
    ASSERT_EQ(solver.largest(), 0.0) << f;
    ASSERT_TRUE(solver.asleep()) << f;
  }
}

//--------------------------------------------------------------------------------------------------
// What the flesh meets while it moves (step 11, touch.hpp).

namespace {

constexpr double kPi = 3.14159265358979323846;

// The chest's particles as touch.hpp sees them: all on the torso, `fromEdge` from its edge, and the
// torso joined to the parts in `joined`
FleshTouch touchOf(const Chest& chest, const FleshModel& m, float fromEdge = 10, uint32_t joined = 0)
{
  std::vector<uint8_t> parts(m.vertex.size(), part::kTorso);
  std::vector<float>   edge(m.vertex.size(), fromEdge), rest;
  for(uint32_t v : m.vertex)
    rest.insert(rest.end(), {chest.cage[v * 3], chest.cage[v * 3 + 1], chest.cage[v * 3 + 2]});
  std::vector<uint32_t> joinedTo(part::kRFoot + 1, 0);
  joinedTo[part::kTorso] = joined;
  return FleshTouch(parts, edge, joinedTo, rest);
}

// A hand lying across the left breast's top along y (the dome `dome` high at x = 5), `into` it, moved
// with the chest by `shift`; at rest it hung at the figure's side
FleshColliders handOn(const glm::dvec3& shift, double into, double dome, int part = part::kLHand)
{
  FleshColliders       c;
  FleshColliders::Limb l;
  const double         r = 1.5, z = dome + r - into;
  l.now  = {{5 + shift.x, -4 + shift.y, z + shift.z}, {5 + shift.x, 4 + shift.y, z + shift.z}, r};
  l.part = part;
  l.rest = {{30, -4, 0}, {30, 4, 0}, r};
  c.limbs.push_back(l);
  return c;
}

// A frame pressed by `with`: its contacts where the targets are, then the step
void pressedStep(FleshSolver& solver, FleshTouch& touch, const std::vector<float>& targets, const FleshColliders& with, bool reshaped = false)
{
  solver.setContacts(touch.find(targets, solver.offsets(), with, fv::Mat4{}), with.bulge);
  solver.step(targets, 1.0 / 60, reshaped);
}

// Where each particle is: its target and its offset
std::vector<double> positions(const FleshSolver& solver, const std::vector<float>& targets)
{
  std::vector<double> out(solver.offsets().begin(), solver.offsets().end());
  for(size_t i = 0; i < out.size(); i++)
    out[i] += targets[i];
  return out;
}

double largestMove(const std::vector<double>& a, const std::vector<double>& b)
{
  double most = 0;
  for(size_t i = 0; i + 2 < a.size(); i += 3)
    most = std::max(most, std::hypot(a[i] - b[i], a[i + 1] - b[i + 1], a[i + 2] - b[i + 2]));
  return most;
}

// How far a point is from a capsule's axis
double fromAxis(const Capsule& c, const double* p)
{
  const glm::dvec3 a{c.a.x, c.a.y, c.a.z}, b{c.b.x, c.b.y, c.b.z}, x{p[0], p[1], p[2]};
  const double     t = std::clamp(glm::dot(x - a, b - a) / glm::dot(b - a, b - a), 0.0, 1.0);
  return glm::length(x - (a + (b - a) * t));
}

}  // namespace

// A hand resting on the left breast as the chest bobs (2 cm at 2 Hz) dents it: the flesh stays out of
// the hand and goes a centimetre in under it, while the right breast moves as it would alone.
TEST(Soft, aHandOnTheChestDentsItWhileItMoves)
{
  Chest       chest(2.5f);
  FleshSolver pressed(buildFlesh(chest.in)), alone(buildFlesh(chest.in));
  pressed.set(on());
  alone.set(on());
  const FleshModel&       m     = pressed.model();
  FleshTouch              touch = touchOf(chest, m);
  const FleshModel::Lobe* left  = lobeNamed(m, "Left breast");
  const FleshModel::Lobe* right = lobeNamed(m, "Right breast");
  ASSERT_TRUE(left && right);
  double deepest = 0;
  for(int f = 0; f < 90; f++)
  {
    const glm::dvec3     shift{0, 2 * std::sin(2 * kPi * 2 * f / 60.0), 0};
    const auto           targets = targetsOf(chest, m, shift);
    const FleshColliders hand    = handOn(shift, 1, 2.5);
    pressedStep(pressed, touch, targets, hand);
    alone.step(targets, 1.0 / 60);
    if(f < 2)
      continue;
    const std::vector<double> at = positions(pressed, targets);
    for(uint32_t p : left->particles)
    {
      ASSERT_GE(fromAxis(hand.limbs[0].now, &at[p * 3]), 1.5 - 0.01) << "frame " << f << ", particle " << p;
      deepest = std::max(deepest, double(targets[p * 3 + 2]) - at[p * 3 + 2]);
    }
    for(uint32_t p : right->particles)
      for(int a = 0; a < 3; a++)
        ASSERT_NEAR(pressed.offsets()[p * 3 + a], alone.offsets()[p * 3 + a], 1e-9) << "frame " << f;
  }
  EXPECT_GT(deepest, 0.9);
}

// A limb in the flesh at rest (a capsule that fits the arm loosely) is allowed as deep as it was: it
// makes no dent, only holds back what sways deeper into it, a tenth of a millimetre where the same hand
// pressing in afresh dents it a centimetre.
TEST(Soft, aLimbInTheFleshAtRestIsAllowedThatDeep)
{
  Chest       chest(2.5f);
  FleshSolver pressed(buildFlesh(chest.in)), alone(buildFlesh(chest.in));
  pressed.set(on());
  alone.set(on());
  FleshTouch touch = touchOf(chest, pressed.model());
  double     most  = 0;
  for(int f = 0; f < 60; f++)
  {
    const glm::dvec3 shift{0, 2 * std::sin(2 * kPi * 2 * f / 60.0), 0};
    const auto       targets = targetsOf(chest, pressed.model(), shift);
    FleshColliders   hand    = handOn(shift, 1, 2.5);
    hand.limbs[0].rest       = handOn({0, 0, 0}, 1, 2.5).limbs[0].now;
    pressedStep(pressed, touch, targets, hand);
    alone.step(targets, 1.0 / 60);
    for(size_t i = 0; i < alone.offsets().size(); i++)
      most = std::max(most, std::fabs(pressed.offsets()[i] - alone.offsets()[i]));
  }
  EXPECT_LT(most, 0.01);
}

// A particle never meets its own part's limb, and a part joined to its own only 5 cm or more from its
// part's edge along the skin.
TEST(Soft, aParticleMeetsNeitherItsOwnPartNorAJoinedOneByTheJoint)
{
  Chest                     chest(2.5f);
  const FleshModel          m       = buildFlesh(chest.in);
  const auto                targets = targetsOf(chest, m, {0, 0, 0});
  const std::vector<double> still(m.vertex.size() * 3, 0.0);
  const auto                found = [&](float fromEdge, uint32_t joined, int part) {
    FleshTouch touch = touchOf(chest, m, fromEdge, joined);
    return touch.find(targets, still, handOn({0, 0, 0}, 1, 2.5, part), fv::Mat4{}).size();
  };
  EXPECT_GT(found(10, 0, part::kLHand), 0u);
  EXPECT_EQ(found(10, 0, part::kTorso), 0u);
  const uint32_t arm = 1u << part::kLUpperArm;
  EXPECT_EQ(found(4, arm, part::kLUpperArm), 0u);
  EXPECT_GT(found(6, arm, part::kLUpperArm), 0u);
}

// Pressed in, the lobe bulges by the contact's share of what the press pushed in: sum A n . d comes to
// -(1 - beta) of the volume pushed in.
TEST(Soft, aPressBulgesItsLobeByTheContactsShare)
{
  for(const double beta : {0.0, 0.7})
  {
    Chest       chest(2.5f);
    FleshSolver solver(buildFlesh(chest.in));
    solver.set(on());
    const FleshModel& m     = solver.model();
    FleshTouch        touch = touchOf(chest, m);
    // drifting at a steady speed, so the flesh is awake and feels nothing but the hand
    for(int f = 0; f < 60; f++)
    {
      const glm::dvec3 shift{0, 0.02 * f, 0};
      FleshColliders   hand = handOn(shift, 1, 2.5);
      hand.bulge            = beta;
      pressedStep(solver, touch, targetsOf(chest, m, shift), hand);
    }
    double sum = 0, dent = 0;
    for(uint32_t p : lobeNamed(m, "Left breast")->particles)
    {
      const double* an = &solver.areaNormals()[p * 3];
      const double* d  = &solver.offsets()[p * 3];
      const double  v  = an[0] * d[0] + an[1] * d[1] + an[2] * d[2];
      sum += v;
      if(solver.pressed()[p])
        dent += std::max(0.0, -v);
    }
    ASSERT_GT(dent, 0.5) << "beta " << beta;
    EXPECT_NEAR(sum, -(1 - beta) * dent, 0.1 * dent) << "beta " << beta;
  }
}

// Once the chest stops, the still pass's dent fades in under the press (its skin sinks to where the
// flesh holds it, over a quarter of a second): no vertex moves more than half a millimetre a frame,
// and the flesh comes to rest on its skin.
TEST(Soft, theHandoverMovesNoVertexMoreThanHalfAMillimetreAFrame)
{
  Chest       chest(2.5f);
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  const FleshModel& m     = solver.model();
  FleshTouch        touch = touchOf(chest, m);
  glm::dvec3        shift{0, 0, 0};
  for(int f = 0; f < 30; f++)
  {
    shift.y = 0.05 * f;
    pressedStep(solver, touch, targetsOf(chest, m, shift), handOn(shift, 1, 2.5));
  }
  // stopped
  std::vector<float> base = targetsOf(chest, m, shift);
  for(int f = 0; f < 12; f++)
    pressedStep(solver, touch, base, handOn(shift, 1, 2.5));
  // the dent the still pass would write: each pressed particle's skin where the flesh holds it
  std::vector<double> was     = positions(solver, base);
  std::vector<float>  dented  = base;
  int                 handing = 0;
  for(size_t p = 0; p < m.vertex.size(); p++)
    if(solver.pressed()[p])
    {
      handing++;
      for(int a = 0; a < 3; a++)
        dented[p * 3 + a] = float(was[p * 3 + a]);
    }
  ASSERT_GT(handing, 5);
  double most = 0;
  for(int f = 1; f <= 15; f++)
  {
    std::vector<float> targets(base.size());
    for(size_t i = 0; i < base.size(); i++)
      targets[i] = base[i] + (dented[i] - base[i]) * float(f / 15.0);
    pressedStep(solver, touch, targets, handOn(shift, 1, 2.5), true);
    const std::vector<double> now = positions(solver, targets);
    most                           = std::max(most, largestMove(was, now));
    was                            = now;
  }
  for(int f = 0; f < 240 && !solver.asleep(); f++)
  {
    pressedStep(solver, touch, dented, handOn(shift, 1, 2.5));
    const std::vector<double> now = positions(solver, dented);
    most                          = std::max(most, largestMove(was, now));
    was                           = now;
  }
  EXPECT_LT(most, 0.05);
  EXPECT_TRUE(solver.asleep());
  EXPECT_EQ(solver.largest(), 0.0);
}

// With no dent to hand over to, a still press lets go a second after the chest stops, smoothly (a
// centimetre over a quarter of a second), and the flesh sleeps on its skin.
TEST(Soft, aStillPressLetsGoWithoutAJump)
{
  Chest       chest(2.5f);
  FleshSolver solver(buildFlesh(chest.in));
  solver.set(on());
  const FleshModel& m     = solver.model();
  FleshTouch        touch = touchOf(chest, m);
  glm::dvec3        shift{0, 0, 0};
  for(int f = 0; f < 30; f++)
  {
    shift.y = 0.05 * f;
    pressedStep(solver, touch, targetsOf(chest, m, shift), handOn(shift, 1, 2.5));
  }
  const std::vector<float> still = targetsOf(chest, m, shift);
  std::vector<double>      was   = positions(solver, still);
  double                   most  = 0;
  int                      f     = 0;
  for(; f < 240 && !solver.asleep(); f++)
  {
    pressedStep(solver, touch, still, handOn(shift, 1, 2.5));
    const std::vector<double> now = positions(solver, still);
    most                          = std::max(most, largestMove(was, now));
    was                           = now;
  }
  EXPECT_LT(most, 0.15);
  EXPECT_TRUE(solver.asleep());
  EXPECT_GT(f, 55) << "it let go before a second";
  EXPECT_LT(f, 3 * 60 + 2);
}

// A top bound to the chest (each of its vertices to three body vertices and their weights, as a
// conforming item is) moves as the chest's flesh does: the weighted sum of the three, a rigid group (a
// button) as one by its reference vertices' mean, and turned into the top's own space where that isn't
// the body's
TEST(Soft, aBoundTopMovesWithTheChest)
{
  // three top vertices: on body vertex 0; halfway between 1 and 2; and a button on vertex 3
  fv::wardrobe::SurfaceBinding top;
  top.verts   = {0, 1, 2, 1, 2, 0, 3, 0, 1};
  top.weights = {1, 0, 0, 0.5f, 0.5f, 0, 1, 0, 0};
  top.used    = {1, 1, 1, 1, 0};
  const std::vector<float> body = {0, 0, 1, 0, 2, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0};
  std::vector<float>       out;
  ASSERT_TRUE(followFlesh(top, nullptr, body, nullptr, out));
  EXPECT_EQ(out, (std::vector<float>{0, 0, 1, 0, 3, 0, 1, 0, 0}));

  // the button (vertex 2) moves with its reference, vertex 0, all the way
  fv::wardrobe::Rigidity button;
  button.weight = {0, 0, 1};
  button.groups.push_back({.refs = {0}, .mask = {2}});
  ASSERT_TRUE(followFlesh(top, &button, body, nullptr, out));
  EXPECT_EQ(out, (std::vector<float>{0, 0, 1, 0, 3, 0, 0, 0, 1}));

  // the top's space turned a quarter about y from the body's: +z there is +x here
  fv::Mat4 turn;
  turn.e = {0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 7, 8, 9, 1};
  ASSERT_TRUE(followFlesh(top, nullptr, body, &turn, out));
  EXPECT_NEAR(out[0], 1, 1e-6);
  EXPECT_NEAR(out[2], 0, 1e-6);
  EXPECT_NEAR(out[4], 3, 1e-6) << "a turn about y leaves y; the move isn't added";

  // per drawn vertex of a top drawn as it is: a seam's two copies of a cage vertex alike
  std::vector<float> drawn;
  drawnOffsets(std::vector<float>{0, 0, 1, 0, 3, 0}, std::vector<uint32_t>{0, 1, 1, 0}, drawn);
  EXPECT_EQ(drawn, (std::vector<float>{0, 0, 1, 0, 3, 0, 0, 3, 0, 0, 0, 1}));
}

// An item bound away from the lobes (shoes) never follows them, and one whose body points are still
// stays: no offsets at all
TEST(Soft, anUnboundItemStays)
{
  fv::wardrobe::SurfaceBinding shoe;
  shoe.verts   = {4, 4, 4};
  shoe.weights = {1, 0, 0};
  shoe.used    = {0, 0, 0, 0, 1};
  const std::vector<uint32_t> lobes = {0, 1, 2, 3};
  EXPECT_FALSE(bindsTo(shoe, lobes));
  std::vector<float> body(15, 0.f), out;
  body[0] = 1;
  EXPECT_FALSE(followFlesh(shoe, nullptr, body, nullptr, out));
  EXPECT_EQ(out, std::vector<float>(3, 0.f));

  fv::wardrobe::SurfaceBinding top;
  top.verts   = {0, 1, 2};
  top.weights = {0.2f, 0.3f, 0.5f};
  top.used    = {1, 1, 1, 0, 0};
  EXPECT_TRUE(bindsTo(top, lobes));
  EXPECT_FALSE(followFlesh(top, nullptr, std::vector<float>(15, 0.f), nullptr, out)) << "asleep: the body's offsets all 0";
  EXPECT_EQ(out, std::vector<float>(3, 0.f));
}
