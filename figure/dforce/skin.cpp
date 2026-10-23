#include <figure/dforce/skin.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace fv::dforce {

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

// bindInverse * bone * bind, per bone
std::vector<Mat4> skinMatrices(const SkinPose& pose)
{
  std::vector<Mat4> out;
  out.reserve(pose.bones.size());
  for(const Mat4& bone : pose.bones)
    out.push_back(pose.bindInverse * (bone * pose.bind));
  return out;
}

// the weighted sum of the vertex's four skin matrices; the weight total says whether it has any
double blend(const std::vector<Mat4>& S, const Top4& skin, size_t v, std::array<double, 16>& out)
{
  out.fill(0);
  double wsum = 0;
  for(size_t j = 0; j < 4; j++)
  {
    const double w = skin.weight[v * 4 + j];
    const size_t k = skin.index[v * 4 + j];
    if(w == 0 || k >= S.size())
      continue;
    wsum += w;
    for(size_t c = 0; c < 16; c++)
      out[c] += S[k].e[c] * w;
  }
  return wsum;
}

}  // namespace

Top4 top4(const mesh::SkinInput& in)
{
  const size_t n = in.start.empty() ? 0 : in.start.size() - 1;
  Top4         out;
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
  return out;
}

std::vector<float> skinLocal(const std::vector<float>& rest, const Top4& skin, const SkinPose& pose)
{
  const size_t       n = rest.size() / 3;
  const auto         S = skinMatrices(pose);
  std::vector<float> out(rest.size());
  std::array<double, 16> m;
  for(size_t v = 0; v < n; v++)
  {
    const double x = rest[v * 3], y = rest[v * 3 + 1], z = rest[v * 3 + 2];
    if(blend(S, skin, v, m) <= 0)
    {
      out[v * 3] = rest[v * 3];
      out[v * 3 + 1] = rest[v * 3 + 1];
      out[v * 3 + 2] = rest[v * 3 + 2];
      continue;
    }
    out[v * 3]     = float(m[0] * x + m[4] * y + m[8] * z + m[12]);
    out[v * 3 + 1] = float(m[1] * x + m[5] * y + m[9] * z + m[13]);
    out[v * 3 + 2] = float(m[2] * x + m[6] * y + m[10] * z + m[14]);
  }
  return out;
}

std::vector<float> drapeDelta(const std::vector<float>& rest, const std::vector<float>& simulated, const Top4& skin, const SkinPose& pose)
{
  const size_t       n = rest.size() / 3;
  const auto         S = skinMatrices(pose);
  std::vector<float> out(rest.size(), 0.f);
  std::array<double, 16> m;
  for(size_t v = 0; v < n; v++)
  {
    const double wsum = blend(S, skin, v, m);
    const double x = rest[v * 3], y = rest[v * 3 + 1], z = rest[v * 3 + 2];
    double       sx, sy, sz;
    std::array<double, 9> m3{1, 0, 0, 0, 1, 0, 0, 0, 1};
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
    if(!invertBlend({simulated[v * 3] - sx, simulated[v * 3 + 1] - sy, simulated[v * 3 + 2] - sz}, m3, d))
      continue;
    out[v * 3]     = float(d[0]);
    out[v * 3 + 1] = float(d[1]);
    out[v * 3 + 2] = float(d[2]);
  }
  return out;
}

}  // namespace fv::dforce
