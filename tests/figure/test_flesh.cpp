// Soft flesh (docs/physics-plan.md step 8): the lobes found on a body, their solve against the skin,
// and the skin the targets come from against the GPU's.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <figure/dforce/skin.hpp>
#include <figure/physics/flesh/flesh.hpp>

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
  fv::dforce::SkinPose pose;
  pose.bind = pose.bindInverse = fv::Mat4{};
  for(const glm::mat4& j : joints)
  {
    fv::Mat4 m;
    for(int c = 0; c < 4; c++)
      for(int r = 0; r < 4; r++)
        m.e[size_t(c * 4 + r)] = j[c][r];
    pose.bones.push_back(m);
  }
  fv::dforce::CageSkin skin;
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
  const std::vector<float> cpu = fv::dforce::skinLocal(rest, skin, pose, some);

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
