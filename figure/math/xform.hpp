#pragma once

// Transforms as three.js computes them: quaternions from Euler angles in any order, composition of
// position, rotation and scale into a column-major matrix, and matrix products, all in double. The
// TypeScript viewer built its figures with three.js, and the native port follows its formulas
// exactly, so the two agree to rounding (test/parity).

#include <array>
#include <cmath>
#include <string_view>

namespace fv {

struct Vec3
{
  double x = 0, y = 0, z = 0;

  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};

struct Quat
{
  double x = 0, y = 0, z = 0, w = 1;

  // a * b (three's multiplyQuaternions)
  Quat operator*(const Quat& b) const
  {
    const Quat& a = *this;
    return {a.x * b.w + a.w * b.x + a.y * b.z - a.z * b.y, a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z,
            a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
  }
  // three's invert(): the conjugate (a unit quaternion's inverse)
  Quat inverse() const { return {-x, -y, -z, w}; }
  Quat normalized() const
  {
    double l = std::sqrt(x * x + y * y + z * z + w * w);
    if(l == 0)
      return {0, 0, 0, 1};
    l = 1 / l;
    return {x * l, y * l, z * l, w * l};
  }
};

// Euler angles (radians) to a quaternion as three's Quaternion.setFromEuler does; `order` is a three
// order string ("XYZ", "ZYX", ...), the axes applied in that order to the vector's frame.
inline Quat quatFromEuler(double ex, double ey, double ez, std::string_view order)
{
  const double c1 = std::cos(ex / 2), c2 = std::cos(ey / 2), c3 = std::cos(ez / 2);
  const double s1 = std::sin(ex / 2), s2 = std::sin(ey / 2), s3 = std::sin(ez / 2);
  if(order == "YXZ")
    return {s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3, c1 * c2 * c3 + s1 * s2 * s3};
  if(order == "ZXY")
    return {s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3};
  if(order == "ZYX")
    return {s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3, c1 * c2 * c3 + s1 * s2 * s3};
  if(order == "YZX")
    return {s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3};
  if(order == "XZY")
    return {s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 + s1 * s2 * s3};
  // XYZ
  return {s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3};
}

// v rotated by q (three's Vector3.applyQuaternion).
inline Vec3 rotate(const Vec3& v, const Quat& q)
{
  const double tx = 2 * (q.y * v.z - q.z * v.y), ty = 2 * (q.z * v.x - q.x * v.z), tz = 2 * (q.x * v.y - q.y * v.x);
  return {v.x + q.w * tx + q.y * tz - q.z * ty, v.y + q.w * ty + q.z * tx - q.x * tz, v.z + q.w * tz + q.x * ty - q.y * tx};
}

// Column-major 4x4, as three's Matrix4.elements and glTF's matrices.
struct Mat4
{
  std::array<double, 16> e{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  // this * b (three's multiplyMatrices(this, b))
  Mat4 operator*(const Mat4& b) const
  {
    Mat4 out;
    for(int c = 0; c < 4; c++)
      for(int r = 0; r < 4; r++)
      {
        double s = 0;
        for(int k = 0; k < 4; k++)
          s += e[k * 4 + r] * b.e[c * 4 + k];
        out.e[c * 4 + r] = s;
      }
    return out;
  }

  Vec3 transformPoint(const Vec3& v) const
  {
    const double w = 1 / (e[3] * v.x + e[7] * v.y + e[11] * v.z + e[15]);
    return {(e[0] * v.x + e[4] * v.y + e[8] * v.z + e[12]) * w, (e[1] * v.x + e[5] * v.y + e[9] * v.z + e[13]) * w,
            (e[2] * v.x + e[6] * v.y + e[10] * v.z + e[14]) * w};
  }
};

// three's Matrix4.compose.
inline Mat4 compose(const Vec3& p, const Quat& q, const Vec3& s)
{
  const double x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
  const double xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
  const double yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
  const double wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
  Mat4         m;
  m.e = {(1 - (yy + zz)) * s.x, (xy + wz) * s.x, (xz - wy) * s.x, 0, (xy - wz) * s.y, (1 - (xx + zz)) * s.y, (yz + wx) * s.y, 0,
         (xz + wy) * s.z,       (yz - wx) * s.z, (1 - (xx + yy)) * s.z, 0, p.x, p.y, p.z, 1};
  return m;
}

// three's Matrix4.invert (gl-matrix's formula; a singular matrix gives all zeros, as three's does).
inline Mat4 inverse(const Mat4& m)
{
  const auto&  te  = m.e;
  const double n11 = te[0], n21 = te[1], n31 = te[2], n41 = te[3];
  const double n12 = te[4], n22 = te[5], n32 = te[6], n42 = te[7];
  const double n13 = te[8], n23 = te[9], n33 = te[10], n43 = te[11];
  const double n14 = te[12], n24 = te[13], n34 = te[14], n44 = te[15];
  const double t1 = n11 * n22 - n21 * n12, t2 = n11 * n32 - n31 * n12, t3 = n11 * n42 - n41 * n12;
  const double t4 = n21 * n32 - n31 * n22, t5 = n21 * n42 - n41 * n22, t6 = n31 * n42 - n41 * n32;
  const double t7 = n13 * n24 - n23 * n14, t8 = n13 * n34 - n33 * n14, t9 = n13 * n44 - n43 * n14;
  const double t10 = n23 * n34 - n33 * n24, t11 = n23 * n44 - n43 * n24, t12 = n33 * n44 - n43 * n34;
  const double det = t1 * t12 - t2 * t11 + t3 * t10 + t4 * t9 - t5 * t8 + t6 * t7;
  Mat4         out;
  if(det == 0)
  {
    out.e.fill(0);
    return out;
  }
  const double d = 1 / det;
  out.e = {(n22 * t12 - n32 * t11 + n42 * t10) * d, (n31 * t11 - n21 * t12 - n41 * t10) * d, (n24 * t6 - n34 * t5 + n44 * t4) * d,
           (n33 * t5 - n23 * t6 - n43 * t4) * d,    (n32 * t9 - n12 * t12 - n42 * t8) * d,  (n11 * t12 - n31 * t9 + n41 * t8) * d,
           (n34 * t3 - n14 * t6 - n44 * t2) * d,    (n13 * t6 - n33 * t3 + n43 * t2) * d,   (n12 * t11 - n22 * t9 + n42 * t7) * d,
           (n21 * t9 - n11 * t11 - n41 * t7) * d,   (n14 * t5 - n24 * t3 + n44 * t1) * d,  (n23 * t3 - n13 * t5 - n43 * t1) * d,
           (n22 * t8 - n12 * t10 - n32 * t7) * d,   (n11 * t10 - n21 * t8 + n31 * t7) * d, (n24 * t2 - n14 * t4 - n34 * t1) * d,
           (n13 * t4 - n23 * t2 + n33 * t1) * d};
  return out;
}

constexpr double kDeg = 3.14159265358979323846 / 180.0;

// DAZ rotation orders name the axes outermost first; three's Euler order is the reverse ("XYZ" -> "ZYX").
inline std::array<char, 4> threeOrder(std::string_view dazOrder)
{
  const bool valid = dazOrder.size() == 3
                     && ((dazOrder == "XYZ") || (dazOrder == "XZY") || (dazOrder == "YXZ") || (dazOrder == "YZX") || (dazOrder == "ZXY") || (dazOrder == "ZYX"));
  const std::string_view o = valid ? dazOrder : "XYZ";
  return {o[2], o[1], o[0], 0};
}

}  // namespace fv
