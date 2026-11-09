#include <figure/physics/skin.hpp>

#include <algorithm>
#include <array>
#include <cmath>

#include <figure/jobs/pool.hpp>

namespace fv::physics {

namespace {

// Column-major 3x3 inverse applied to `offset`; false when there is none.
bool invertBlend(const std::array<double, 3>& offset, const std::array<double, 9>& m, std::array<double, 3>& out)
{
  const double c00 = m[0], c10 = m[1], c20 = m[2];
  const double c01 = m[3], c11 = m[4], c21 = m[5];
  const double c02 = m[6], c12 = m[7], c22 = m[8];
  const double det = c00 * (c11 * c22 - c21 * c12) - c01 * (c10 * c22 - c20 * c12) + c02 * (c10 * c21 - c20 * c11);
  if(std::fabs(det) < 1e-6)
    return false;
  const double inv = 1 / det;
  const double i00 = (c11 * c22 - c21 * c12) * inv;
  const double i10 = (c20 * c12 - c10 * c22) * inv;
  const double i20 = (c10 * c21 - c20 * c11) * inv;
  const double i01 = (c21 * c02 - c01 * c22) * inv;
  const double i11 = (c00 * c22 - c20 * c02) * inv;
  const double i21 = (c20 * c01 - c00 * c21) * inv;
  const double i02 = (c01 * c12 - c11 * c02) * inv;
  const double i12 = (c10 * c02 - c00 * c12) * inv;
  const double i22 = (c00 * c11 - c10 * c01) * inv;
  const auto [x, y, z] = offset;
  out                  = {i00 * x + i01 * y + i02 * z, i10 * x + i11 * y + i12 * z, i20 * x + i21 * y + i22 * z};
  return true;
}

using detail::JointDq;

using V3 = std::array<double, 3>;
double dot3(const V3& a, const V3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
V3 cross3(const V3& a, const V3& b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

JointDq jointDq(const Mat4& m)
{
  const auto& e = m.e;
  const V3    a[3] = {{e[0], e[1], e[2]}, {e[4], e[5], e[6]}, {e[8], e[9], e[10]}};
  const V3    t{e[12], e[13], e[14]};
  V3          r[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const double l0 = std::sqrt(dot3(a[0], a[0]));
  if(l0 > 1e-12)
  {
    const V3     r0{a[0][0] / l0, a[0][1] / l0, a[0][2] / l0};
    const double p  = dot3(r0, a[1]);
    const V3     c1{a[1][0] - p * r0[0], a[1][1] - p * r0[1], a[1][2] - p * r0[2]};
    const double l1 = std::sqrt(dot3(c1, c1));
    if(l1 > 1e-12)
    {
      r[0] = r0;
      r[1] = {c1[0] / l1, c1[1] / l1, c1[2] / l1};
      r[2] = cross3(r[0], r[1]);
    }
  }
  JointDq out;
  // the rotation [r0 r1 r2] as a quaternion (R(i, j) is r[j][i])
  const double tr = r[0][0] + r[1][1] + r[2][2];
  auto&        q  = out.q;
  if(tr > 0)
  {
    const double s = std::sqrt(tr + 1) * 2;
    q              = {(r[1][2] - r[2][1]) / s, (r[2][0] - r[0][2]) / s, (r[0][1] - r[1][0]) / s, s / 4};
  }
  else if(r[0][0] > r[1][1] && r[0][0] > r[2][2])
  {
    const double s = std::sqrt(1 + r[0][0] - r[1][1] - r[2][2]) * 2;
    q              = {s / 4, (r[1][0] + r[0][1]) / s, (r[2][0] + r[0][2]) / s, (r[1][2] - r[2][1]) / s};
  }
  else if(r[1][1] > r[2][2])
  {
    const double s = std::sqrt(1 + r[1][1] - r[0][0] - r[2][2]) * 2;
    q              = {(r[1][0] + r[0][1]) / s, s / 4, (r[2][1] + r[1][2]) / s, (r[2][0] - r[0][2]) / s};
  }
  else
  {
    const double s = std::sqrt(1 + r[2][2] - r[0][0] - r[1][1]) * 2;
    q              = {(r[2][0] + r[0][2]) / s, (r[2][1] + r[1][2]) / s, s / 4, (r[0][1] - r[1][0]) / s};
  }
  const double ql = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for(double& c : q)
    c /= ql;
  // the stretch R^T A, and d = (t, 0) q / 2
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      out.k[size_t(i * 3 + j)] = dot3(r[i], a[j]);
  const V3 qv{q[0], q[1], q[2]};
  const V3 tq = cross3(t, qv);
  out.d       = {(q[3] * t[0] + tq[0]) / 2, (q[3] * t[1] + tq[1]) / 2, (q[3] * t[2] + tq[2]) / 2, -dot3(t, qv) / 2};
  return out;
}

}  // namespace

SkinBlend::SkinBlend(std::span<const Mat4> bones, const SkinView& skin)
    : m_bones(bones)
    , m_skin(skin)
{
  if(skin.dualQuats())
  {
    m_dqs.reserve(bones.size());
    for(const Mat4& m : bones)
      m_dqs.push_back(jointDq(m));
  }
}

// the vertex's skin as one matrix: the weighted sum of its skin matrices (eight with DAZ's skinning),
// mixed with its dual quaternion skin by its share; the weight total says whether it has any
double SkinBlend::vertex(size_t v, std::array<double, 16>& out) const
{
  const auto&         S    = m_bones;
  const auto&         dqs  = m_dqs;
  const SkinView&     skin = m_skin;
  out.fill(0);
  const bool   eight = !skin.index8.empty();
  const size_t per   = eight ? 8 : 4;
  const auto&  index = eight ? skin.index8 : skin.index;
  const auto&  wt    = eight ? skin.weight8 : skin.weight;
  const double share = dqs.size() != S.size() ? 0.0 : v < skin.dualQuatShare.size() ? skin.dualQuatShare[v] : skin.dualQuat;
  // A vertex all dual quaternion (most of a figure) takes nothing of the linear blend's 3x4 (it is
  // multiplied by 0 below), so that is summed only where it counts; its bottom row always is
  const auto linear = [&](bool whole) {
    double sum = 0;
    for(size_t j = 0; j < per; j++)
    {
      const double w = wt[v * per + j];
      const size_t k = index[v * per + j];
      if(w == 0 || k >= S.size())
        continue;
      sum += w;
      if(whole)
        for(size_t c = 0; c < 16; c++)
          out[c] += S[k].e[c] * w;
      else
        for(size_t c = 3; c < 16; c += 4)
          out[c] += S[k].e[c] * w;
    }
    return sum;
  };
  const bool   allDq = share == 1.0;
  const double wsum  = linear(!allDq);
  if(wsum <= 0 || share <= 0)
    return wsum;
  std::array<double, 4> b0{}, b1{}, first{};
  std::array<double, 9> kb{};
  bool                  met = false;
  for(size_t j = 0; j < per; j++)
  {
    const double w = wt[v * per + j];
    const size_t k = index[v * per + j];
    if(w == 0 || k >= S.size())
      continue;
    const JointDq& dq = dqs[k];
    if(!met)
      first = dq.q;
    met             = true;
    const double sw = dq.q[0] * first[0] + dq.q[1] * first[1] + dq.q[2] * first[2] + dq.q[3] * first[3] < 0 ? -w : w;
    for(int c = 0; c < 4; c++)
    {
      b0[size_t(c)] += sw * dq.q[size_t(c)];
      b1[size_t(c)] += sw * dq.d[size_t(c)];
    }
    for(int c = 0; c < 9; c++)
      kb[size_t(c)] += w * dq.k[size_t(c)];
  }
  const double len = std::sqrt(b0[0] * b0[0] + b0[1] * b0[1] + b0[2] * b0[2] + b0[3] * b0[3]);
  if(len <= 0)
  {
    // no rotation to blend: the linear blend alone, whole
    if(allDq)
    {
      out.fill(0);
      linear(true);
    }
    return wsum;
  }
  const double x = b0[0] / len, y = b0[1] / len, z = b0[2] / len, w = b0[3] / len;
  const V3     qv{x, y, z}, dv{b1[0] / len, b1[1] / len, b1[2] / len};
  const double dw = b1[3] / len;
  // R(q), then the vertex's affine dual quaternion skin R Kb + t, mixed into the linear blend
  const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
                       2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                       2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
  const V3     qd = cross3(qv, dv);
  const V3     t{2 * (w * dv[0] - dw * x + qd[0]), 2 * (w * dv[1] - dw * y + qd[1]), 2 * (w * dv[2] - dw * z + qd[2])};
  for(int row = 0; row < 3; row++)
  {
    for(int col = 0; col < 3; col++)
    {
      double a = 0;
      for(int i = 0; i < 3; i++)
        a += R[row * 3 + i] * kb[size_t(i * 3 + col)];
      out[size_t(col * 4 + row)] = out[size_t(col * 4 + row)] * (1 - share) + a * share;
    }
    out[size_t(12 + row)] = out[size_t(12 + row)] * (1 - share) + t[size_t(row)] * share;
  }
  return wsum;
}

std::vector<int> nodeOfMesh(const SceneGraph& scene)
{
  std::vector<int> nodeOf(scene.meshes.size(), -1);
  // the first node: a mesh drawn again by a DAZ Studio instance has the copy's node added after it
  for(size_t n = 0; n < scene.nodes.size(); n++)
    if(scene.nodes[n].kind == NodeKind::eMesh && scene.nodes[n].mesh >= 0 && nodeOf[size_t(scene.nodes[n].mesh)] < 0)
      nodeOf[size_t(scene.nodes[n].mesh)] = int(n);
  return nodeOf;
}

SkinPose currentSkin(const SceneGraph& scene, const std::vector<Mat4>& worlds, const std::vector<int>& nodeOf, int mesh)
{
  const Mesh& m = scene.meshes[size_t(mesh)];
  SkinPose    pose;
  pose.bind        = m.bindMatrix;
  pose.bindInverse = inverse(nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : m.bindMatrix);
  pose.bones.reserve(m.bones.size());
  for(size_t k = 0; k < m.bones.size(); k++)
    pose.bones.push_back(worlds[size_t(m.bones[k])] * (k < m.boneInverses.size() ? m.boneInverses[k] : Mat4{}));
  return pose;
}

std::vector<float> transformPoints(const Mat4& m, const std::vector<float>& src)
{
  std::vector<float> out(src.size());
  const auto&        e = m.e;
  for(size_t i = 0; i + 2 < src.size(); i += 3)
  {
    const double x = src[i], y = src[i + 1], z = src[i + 2];
    out[i]     = float(e[0] * x + e[4] * y + e[8] * z + e[12]);
    out[i + 1] = float(e[1] * x + e[5] * y + e[9] * z + e[13]);
    out[i + 2] = float(e[2] * x + e[6] * y + e[10] * z + e[14]);
  }
  return out;
}

std::vector<Mat4> skinMatrices(const SkinPose& pose)
{
  std::vector<Mat4> out;
  out.reserve(pose.bones.size());
  for(const Mat4& bone : pose.bones)
    out.push_back(pose.bindInverse * (bone * pose.bind));
  return out;
}

CageSkin cageSkin(const mesh::SkinInput& in)
{
  const size_t n = in.start.empty() ? 0 : in.start.size() - 1;
  CageSkin     out;
  out.index.assign(n * 4, 0);
  out.weight.assign(n * 4, 0.f);
  std::vector<std::pair<int, float>> pick;
  for(size_t v = 0; v < n; v++)
  {
    pick.clear();
    for(int k = in.start[v]; k < in.start[v + 1]; k++)
      if(in.weight[size_t(k)] > 0)
        pick.emplace_back(in.bone[size_t(k)], in.weight[size_t(k)]);
    std::stable_sort(pick.begin(), pick.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    const size_t m = std::min<size_t>(4, pick.size());
    float        sum = 0;
    for(size_t k = 0; k < m; k++)
      sum += pick[k].second;
    if(sum <= 0)
    {
      out.weight[v * 4] = 1;
      continue;
    }
    for(size_t k = 0; k < m; k++)
    {
      out.index[v * 4 + k]  = uint16_t(pick[k].first);
      out.weight[v * 4 + k] = pick[k].second / sum;
    }
  }
  out.dualQuat      = in.dualQuat;
  out.dualQuatShare = in.dualQuatMap.size() == n ? in.dualQuatMap : std::vector<float>{};
  // the eight strongest, where some vertex has more than four
  bool more = false;
  for(size_t v = 0; v < n && !more; v++)
    more = in.start[v + 1] - in.start[v] > 4;
  if(!more)
    return out;
  out.index8.assign(n * 8, 0);
  out.weight8.assign(n * 8, 0.f);
  for(size_t v = 0; v < n; v++)
  {
    pick.clear();
    for(int k = in.start[v]; k < in.start[v + 1]; k++)
      if(in.weight[size_t(k)] > 0)
        pick.emplace_back(in.bone[size_t(k)], in.weight[size_t(k)]);
    std::stable_sort(pick.begin(), pick.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    const size_t m   = std::min<size_t>(8, pick.size());
    float        sum = 0;
    for(size_t k = 0; k < m; k++)
      sum += pick[k].second;
    if(sum <= 0)
    {
      out.weight8[v * 8] = 1;
      continue;
    }
    for(size_t k = 0; k < m; k++)
    {
      out.index8[v * 8 + k]  = uint16_t(pick[k].first);
      out.weight8[v * 8 + k] = pick[k].second / sum;
    }
  }
  return out;
}

namespace {

// Rest vertex v skinned into `out`, or left as it is where nothing weighs it.
void skinVertex(const SkinBlend& blend, std::span<const float> rest, size_t v, float* out)
{
  std::array<double, 16> m;
  const double           x = rest[v * 3], y = rest[v * 3 + 1], z = rest[v * 3 + 2];
  if(blend.vertex(v, m) <= 0)
  {
    out[0] = rest[v * 3];
    out[1] = rest[v * 3 + 1];
    out[2] = rest[v * 3 + 2];
    return;
  }
  out[0] = float(m[0] * x + m[4] * y + m[8] * z + m[12]);
  out[1] = float(m[1] * x + m[5] * y + m[9] * z + m[13]);
  out[2] = float(m[2] * x + m[6] * y + m[10] * z + m[14]);
}

}  // namespace

// Each vertex on its own, on the pool: the collider skins the body and every garment it pushes as a
// pass begins, on the thread that owns the figure (wardrobe plan 0m)
std::vector<float> skinLocal(const std::vector<float>& rest, const CageSkin& skin, const SkinPose& pose)
{
  const size_t       n = rest.size() / 3;
  const auto         S = skinMatrices(pose);
  const SkinBlend    blend(S, SkinView::of(skin));
  std::vector<float> out(rest.size());
  jobs::parallelFor(n, 4096, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
      skinVertex(blend, rest, v, &out[v * 3]);
  });
  return out;
}

std::vector<float> skinLocal(std::span<const float> rest, const CageSkin& skin, const SkinPose& pose, std::span<const uint32_t> vertices)
{
  const auto         S = skinMatrices(pose);
  const SkinBlend    blend(S, SkinView::of(skin));
  std::vector<float> out(vertices.size() * 3);
  jobs::parallelFor(vertices.size(), 4096, [&](size_t begin, size_t end) {
    for(size_t k = begin; k < end; k++)
      if(vertices[k] * 3 + 2 < rest.size())
        skinVertex(blend, rest, vertices[k], &out[k * 3]);
  });
  return out;
}

namespace {

// Vertex v's delta whose skin puts it at `at` (mesh-local), into `out`: false where its blended 3x3 is
// singular (the delta stays 0).
bool deltaOf(const SkinBlend& blend, size_t v, std::span<const float> rest, const float* at, float* out)
{
  std::array<double, 16> m;
  const double           wsum = blend.vertex(v, m);
  const double           x = rest[v * 3], y = rest[v * 3 + 1], z = rest[v * 3 + 2];
  double                 sx, sy, sz;
  std::array<double, 9>  m3{1, 0, 0, 0, 1, 0, 0, 0, 1};
  if(wsum <= 0)
  {
    sx = x;
    sy = y;
    sz = z;
  }
  else
  {
    sx = m[0] * x + m[4] * y + m[8] * z + m[12];
    sy = m[1] * x + m[5] * y + m[9] * z + m[13];
    sz = m[2] * x + m[6] * y + m[10] * z + m[14];
    m3 = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]};
  }
  std::array<double, 3> d;
  if(!invertBlend({at[0] - sx, at[1] - sy, at[2] - sz}, m3, d))
    return false;
  out[0] = float(d[0]);
  out[1] = float(d[1]);
  out[2] = float(d[2]);
  return true;
}

}  // namespace

std::vector<float> drapeDelta(const std::vector<float>& rest, const std::vector<float>& simulated, const CageSkin& skin, const SkinPose& pose)
{
  const size_t       n   = rest.size() / 3;
  const auto         S = skinMatrices(pose);
  const SkinBlend    blend(S, SkinView::of(skin));
  std::vector<float> out(rest.size(), 0.f);
  // each vertex on its own, on the pool: a light copy's whole dress (the Kai Dress's 42,687 vertices)
  // took 16.6 ms of a live step on one thread with the riding
  jobs::parallelFor(n, 2048, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
      deltaOf(blend, v, rest, &simulated[v * 3], &out[v * 3]);
  });
  return out;
}

std::vector<float> drapeDelta(std::span<const float> rest, std::span<const float> simulated, const CageSkin& skin, const SkinPose& pose,
                              std::span<const uint32_t> vertices)
{
  const size_t       n = rest.size() / 3;
  const auto         S = skinMatrices(pose);
  const SkinBlend    blend(S, SkinView::of(skin));
  std::vector<float> out(vertices.size() * 3, 0.f);
  if(simulated.size() != vertices.size() * 3)
    return out;
  jobs::parallelFor(vertices.size(), 2048, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
      if(vertices[i] < n)
        deltaOf(blend, vertices[i], rest, &simulated[i * 3], &out[i * 3]);
  });
  return out;
}

}  // namespace fv::physics
