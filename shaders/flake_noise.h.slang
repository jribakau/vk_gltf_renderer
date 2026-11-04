#ifndef FLAKE_NOISE_H
#define FLAKE_NOISE_H

// Iray Uber's metallic flakes (FV_materials_flakes), as MDL's base.mdl makes them for the Uber: its
// flake_noise_texture and flake_noise_bump_texture at noise_type 0 (NVIDIA MDL SDK, BSD 3-clause).
// The UV point over the flake size is pushed by twice a smooth noise's gradient, so the cells come
// out ragged, then floored to a cell. Each cell is one flake: base.mdl's lattice noise at that cell
// gives its weight, pow(value, 1 / density), and its tilt, the noise's gradient against the normal
// scaled by |grad.z| + 1 / strength. The lattice's bytes are a hash here rather than base.mdl's five
// permutation tables: the same spread of values, with no table for the shader to index.
// Compiled as C++ too (tests/test_flake_noise.cpp).

// Wellons' lowbias32
uint flakeHash(uint x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

uint flakeCellHash(int3 c)
{
  return flakeHash(uint(c.x) + flakeHash(uint(c.y) + flakeHash(uint(c.z))));
}

// A lattice point's byte, 0..255, where base.mdl looks one up in its tables
float flakeLattice(int x, int y, int z)
{
  return float(flakeCellHash(int3(x, y, z)) & 255U);
}

struct FlakeNoise
{
  float3 grad;
  float  val;  // 0..1
};

// base.mdl's mi_noise(float3): a quadratic B-spline through the bytes of the lattice points from
// floor(p) on, over 256, and its gradient
FlakeNoise flakeSmoothNoise(float3 p)
{
  const float3 cell = floor(p);
  const int    ix = int(cell.x), iy = int(cell.y), iz = int(cell.z);
  const float  fx = p.x - cell.x, fy = p.y - cell.y, fz = p.z - cell.z;
  // the weights along x carry the 1/256
  const float  du2    = fx * (1.0F / 256.0F);
  const float  du2tmp = fx * du2;
  const float  du1    = (1.0F / 256.0F) - du2 - du2;
  const float  du0    = du2 - (1.0F / 256.0F);
  const float  u2     = du2tmp * 0.5F;
  const float  u1     = du2 + (0.5F / 256.0F) - du2tmp;
  const float  u0     = u2 - du2 + (0.5F / 256.0F);
  const float  du5 = fy, du4 = 1.0F - fy - fy, du3 = fy - 1.0F;
  const float  u5 = fy * fy * 0.5F, u4 = fy + 0.5F - fy * fy, u3 = u5 - fy + 0.5F;
  const float3 dw = float3(fz - 1.0F, 1.0F - fz - fz, fz);
  const float3 w  = float3(fz * fz * 0.5F - fz + 0.5F, fz + 0.5F - fz * fz, fz * fz * 0.5F);

  FlakeNoise r;
  r.grad = float3(0.0F);
  r.val  = 0.0F;
  for(int i = 0; i < 3; i++)
  {
    const int   z   = iz + i;
    const float n01 = flakeLattice(ix, iy, z), n02 = flakeLattice(ix + 1, iy, z), n03 = flakeLattice(ix + 2, iy, z);
    const float n11 = flakeLattice(ix, iy + 1, z), n12 = flakeLattice(ix + 1, iy + 1, z), n13 = flakeLattice(ix + 2, iy + 1, z);
    const float n21 = flakeLattice(ix, iy + 2, z), n22 = flakeLattice(ix + 1, iy + 2, z), n23 = flakeLattice(ix + 2, iy + 2, z);
    const float fxdz = u3 * (du0 * n01 + du1 * n02 + du2 * n03) + u4 * (du0 * n11 + du1 * n12 + du2 * n13)
                       + u5 * (du0 * n21 + du1 * n22 + du2 * n23);
    const float dyz0 = u0 * n01 + u1 * n02 + u2 * n03;
    const float dyz1 = u0 * n11 + u1 * n12 + u2 * n13;
    const float dyz2 = u0 * n21 + u1 * n22 + u2 * n23;
    const float dz   = u3 * dyz0 + u4 * dyz1 + u5 * dyz2;
    r.grad.x += w[i] * fxdz;
    r.grad.y += w[i] * (du3 * dyz0 + du4 * dyz1 + du5 * dyz2);
    r.val += w[i] * dz;
    r.grad.z += dw[i] * dz;
  }
  return r;
}

// base.mdl's mi_noise(int3): the same spline at the middle of the cell's lattice points, c - 1 to
// c + 1, which is flakeSmoothNoise(c - 0.5)
FlakeNoise flakeCellNoise(int3 c)
{
  float n[3][3][3];  // [z][y][x], from c - 1
  for(int z = 0; z < 3; z++)
    for(int y = 0; y < 3; y++)
      for(int x = 0; x < 3; x++)
        n[z][y][x] = flakeLattice(c.x + x - 1, c.y + y - 1, c.z + z - 1);
  const float corner = 0.0078125F / 256.0F, edge = 0.046875F / 256.0F, centre = 0.28125F / 256.0F;

  FlakeNoise r;
  r.grad.x = corner * (n[0][0][2] - n[0][0][0] + n[0][2][2] - n[0][2][0] + n[2][0][2] - n[2][0][0] + n[2][2][2] - n[2][2][0])
             + edge * (n[0][1][2] - n[0][1][0] + n[1][0][2] - n[1][0][0] + n[1][2][2] - n[1][2][0] + n[2][1][2] - n[2][1][0])
             + centre * (n[1][1][2] - n[1][1][0]);
  // each row along x, weighted 1, 6, 1
  float row[3][3];
  for(int z = 0; z < 3; z++)
    for(int y = 0; y < 3; y++)
      row[z][y] = n[z][y][0] + 6.0F * n[z][y][1] + n[z][y][2];
  r.grad.y = corner * (row[0][2] - row[0][0] + row[2][2] - row[2][0]) + edge * (row[1][2] - row[1][0]);
  const float dz0 = (0.001953125F / 256.0F) * (row[0][0] + row[0][2]) + (0.01171875F / 256.0F) * (n[0][1][0] + n[0][1][2])
                    + (0.0703125F / 256.0F) * n[0][1][1];
  const float dz2 = (0.001953125F / 256.0F) * (row[2][0] + row[2][2]) + (0.01171875F / 256.0F) * (n[2][1][0] + n[2][1][2])
                    + (0.0703125F / 256.0F) * n[2][1][1];
  r.grad.z = 4.0F * (dz2 - dz0);
  r.val    = dz0 + dz2 + (0.01171875F / 256.0F) * (row[1][0] + row[1][2]) + (0.0703125F / 256.0F) * (n[1][1][0] + n[1][1][2])
          + (0.421875F / 256.0F) * n[1][1][1];
  return r;
}

struct Flake
{
  int3   cell;
  float  reflectivity;  // the layer's weight in this flake, before the material's own
  float3 grad;          // its tilt
};

// The flake under a UV point, `size` UV units across
Flake flakeAt(float2 uv, float size, float density)
{
  float3 pos = float3(uv.x, uv.y, 0.0F) / size;
  pos += flakeSmoothNoise(pos).grad * 2.0F;
  Flake f;
  f.cell               = int3(floor(pos));
  const FlakeNoise n   = flakeCellNoise(f.cell);
  f.grad               = n.grad;
  f.reflectivity       = pow(max(n.val, 0.0F), 1.0F / max(density, 1e-4F));
  return f;
}

// Its normal, as base.mdl's apply_bump makes it from the UV tangents `T`, `B` and the normal `N`
float3 flakeNormal(Flake f, float3 T, float3 B, float3 N, float strength)
{
  return normalize(T * f.grad.x + B * f.grad.y + N * (abs(f.grad.z) + 1.0F / max(strength, 1e-4F)));
}

// Whether the flake is there at `weight`, its layer's weight times its reflectivity. Iray lays the
// flake over the base by that weight; here a flake is all mirror or not there, by a draw that stays
// with it, so a sparkle holds still from frame to frame and under DLSS rather than averaging away.
// Over many flakes the share is the same.
bool flakeShows(Flake f, float weight)
{
  return float(flakeHash(flakeCellHash(f.cell) ^ 0x9e3779b9U) >> 8) * (1.0F / 16777216.0F) < weight;
}

#endif
