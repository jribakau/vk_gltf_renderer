/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Adobe's lookup tables in a buffer (GltfScene::openPbrLuts), read through its texture hooks with the
// interpolation its array mode has. As `static const` arrays in the shader, every read made the driver
// copy them, and the closure cost 5x a sample's trace: a large table indexed at run time doesn't belong
// in a hot loop, as Sobol's direction table found. A polyglot: the C++ side tests it against the arrays.
//
// The buffer holds floats, in Adobe's table order (OpenPBR_LutId_*): a table's cells flattened with the
// last index fastest, and the sheen's three coefficients together.

#ifndef FV_OPENPBR_LUTS_H_SLANG
#define FV_OPENPBR_LUTS_H_SLANG

#define FV_OPENPBR_LUT_SIZE 32  // OpenPBR_EnergyTableSize, OpenPBR_LTCTableSize
#define FV_OPENPBR_LUT_CELLS_2D (FV_OPENPBR_LUT_SIZE * FV_OPENPBR_LUT_SIZE)
#define FV_OPENPBR_LUT_CELLS_3D (FV_OPENPBR_LUT_SIZE * FV_OPENPBR_LUT_SIZE * FV_OPENPBR_LUT_SIZE)
#define FV_OPENPBR_LUT_METAL_AVERAGE 6  // OpenPBR_LutId_IdealMetalAverageEnergyComplement: 1D, kept as thin 2D
#define FV_OPENPBR_LUT_SHEEN 7          // OpenPBR_LutId_LTC
#define FV_OPENPBR_LUT_FLOATS (2 * FV_OPENPBR_LUT_CELLS_3D + 4 * FV_OPENPBR_LUT_CELLS_2D + FV_OPENPBR_LUT_SIZE + 3 * FV_OPENPBR_LUT_CELLS_2D)

// Where a table starts
int fvOpenPbrLutStart(int id)
{
  if(id == 0)
    return 0;
  if(id == 1)
    return FV_OPENPBR_LUT_CELLS_3D;
  if(id == 2)
    return FV_OPENPBR_LUT_CELLS_3D + FV_OPENPBR_LUT_CELLS_2D;
  if(id == 3)
    return FV_OPENPBR_LUT_CELLS_3D + 2 * FV_OPENPBR_LUT_CELLS_2D;
  if(id == 4)
    return 2 * FV_OPENPBR_LUT_CELLS_3D + 2 * FV_OPENPBR_LUT_CELLS_2D;
  if(id == 5)
    return 2 * FV_OPENPBR_LUT_CELLS_3D + 3 * FV_OPENPBR_LUT_CELLS_2D;
  if(id == FV_OPENPBR_LUT_METAL_AVERAGE)
    return 2 * FV_OPENPBR_LUT_CELLS_3D + 4 * FV_OPENPBR_LUT_CELLS_2D;
  return 2 * FV_OPENPBR_LUT_CELLS_3D + 4 * FV_OPENPBR_LUT_CELLS_2D + FV_OPENPBR_LUT_SIZE;
}

// The two cells around a texture coordinate of Adobe's hooks, which are at the cells' centres
struct FvOpenPbrLutAxis
{
  int   i0;
  int   i1;
  float t;
};

FvOpenPbrLutAxis fvOpenPbrLutAxis(float u)
{
  const float      cell = clamp(u * float(FV_OPENPBR_LUT_SIZE) - 0.5F, 0.0F, float(FV_OPENPBR_LUT_SIZE - 1));
  FvOpenPbrLutAxis axis;
  axis.i0 = int(floor(cell));
  axis.i1 = min(axis.i0 + 1, FV_OPENPBR_LUT_SIZE - 1);
  axis.t  = cell - float(axis.i0);
  return axis;
}

float fvOpenPbrLutMix(float a, float b, float t)
{
  return a + (b - a) * t;
}

// OPENPBR_SAMPLE_2D_TEXTURE: a 2D table's first index is uv.y, its last uv.x
float4 fvOpenPbrLut2D(float* luts, int id, float2 uv)
{
  const int              start = fvOpenPbrLutStart(id);
  const FvOpenPbrLutAxis x     = fvOpenPbrLutAxis(uv.x);
  if(id == FV_OPENPBR_LUT_METAL_AVERAGE)
    return float4(fvOpenPbrLutMix(luts[start + x.i0], luts[start + x.i1], x.t), 0.0F, 0.0F, 0.0F);

  const FvOpenPbrLutAxis y   = fvOpenPbrLutAxis(uv.y);
  const int              i00 = y.i0 * FV_OPENPBR_LUT_SIZE + x.i0;
  const int              i01 = y.i0 * FV_OPENPBR_LUT_SIZE + x.i1;
  const int              i10 = y.i1 * FV_OPENPBR_LUT_SIZE + x.i0;
  const int              i11 = y.i1 * FV_OPENPBR_LUT_SIZE + x.i1;
  if(id == FV_OPENPBR_LUT_SHEEN)
  {
    float4 out_ = float4(0.0F);
    for(int c = 0; c < 3; c++)
    {
      const float near_ = fvOpenPbrLutMix(luts[start + 3 * i00 + c], luts[start + 3 * i01 + c], x.t);
      const float far_  = fvOpenPbrLutMix(luts[start + 3 * i10 + c], luts[start + 3 * i11 + c], x.t);
      out_[c]           = fvOpenPbrLutMix(near_, far_, y.t);
    }
    return out_;
  }
  const float near_ = fvOpenPbrLutMix(luts[start + i00], luts[start + i01], x.t);
  const float far_  = fvOpenPbrLutMix(luts[start + i10], luts[start + i11], x.t);
  return float4(fvOpenPbrLutMix(near_, far_, y.t), 0.0F, 0.0F, 0.0F);
}

// OPENPBR_SAMPLE_3D_TEXTURE: a 3D table's first index is uvw.z, its last uvw.x
float4 fvOpenPbrLut3D(float* luts, int id, float3 uvw)
{
  const int              start = fvOpenPbrLutStart(id);
  const FvOpenPbrLutAxis x     = fvOpenPbrLutAxis(uvw.x);
  const FvOpenPbrLutAxis y     = fvOpenPbrLutAxis(uvw.y);
  const FvOpenPbrLutAxis z     = fvOpenPbrLutAxis(uvw.z);
  const int              z0    = start + z.i0 * FV_OPENPBR_LUT_CELLS_2D;
  const int              z1    = start + z.i1 * FV_OPENPBR_LUT_CELLS_2D;
  const float near_ = fvOpenPbrLutMix(fvOpenPbrLutMix(luts[z0 + y.i0 * FV_OPENPBR_LUT_SIZE + x.i0], luts[z0 + y.i0 * FV_OPENPBR_LUT_SIZE + x.i1], x.t),
                                      fvOpenPbrLutMix(luts[z0 + y.i1 * FV_OPENPBR_LUT_SIZE + x.i0], luts[z0 + y.i1 * FV_OPENPBR_LUT_SIZE + x.i1], x.t), y.t);
  const float far_  = fvOpenPbrLutMix(fvOpenPbrLutMix(luts[z1 + y.i0 * FV_OPENPBR_LUT_SIZE + x.i0], luts[z1 + y.i0 * FV_OPENPBR_LUT_SIZE + x.i1], x.t),
                                      fvOpenPbrLutMix(luts[z1 + y.i1 * FV_OPENPBR_LUT_SIZE + x.i0], luts[z1 + y.i1 * FV_OPENPBR_LUT_SIZE + x.i1], x.t), y.t);
  return float4(fvOpenPbrLutMix(near_, far_, z.t), 0.0F, 0.0F, 0.0F);
}

#endif  // FV_OPENPBR_LUTS_H_SLANG
