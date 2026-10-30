/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// GltfScene::openPbrLuts's contents, from Adobe's arrays (fv_openpbr_luts.h.slang has the layout).
// C++ only: include it where the arrays are visible.

#pragma once

// `table` holds FV_OPENPBR_LUT_FLOATS
inline void fvOpenPbrFillLuts(float* table)
{
  const auto copy = [&](int id, const OpenPBR_EnergyTableElement* cells, int count) {
    for(int i = 0; i < count; i++)
      table[fvOpenPbrLutStart(id) + i] = float(cells[i]) / 65535.0f;
  };
  copy(0, OpenPBR_IdealDielectricEnergyComplement_Array, FV_OPENPBR_LUT_CELLS_3D);
  copy(1, OpenPBR_IdealDielectricAverageEnergyComplement_Array, FV_OPENPBR_LUT_CELLS_2D);
  copy(2, OpenPBR_IdealDielectricReflectionRatio_Array, FV_OPENPBR_LUT_CELLS_2D);
  copy(3, OpenPBR_OpaqueDielectricEnergyComplement_Array, FV_OPENPBR_LUT_CELLS_3D);
  copy(4, OpenPBR_OpaqueDielectricAverageEnergyComplement_Array, FV_OPENPBR_LUT_CELLS_2D);
  copy(5, OpenPBR_IdealMetalEnergyComplement_Array, FV_OPENPBR_LUT_CELLS_2D);
  copy(FV_OPENPBR_LUT_METAL_AVERAGE, OpenPBR_IdealMetalAverageEnergyComplement_Array, FV_OPENPBR_LUT_SIZE);
  const int sheen = fvOpenPbrLutStart(FV_OPENPBR_LUT_SHEEN);
  for(int i = 0; i < FV_OPENPBR_LUT_CELLS_2D; i++)
    for(int c = 0; c < 3; c++)
      table[sheen + 3 * i + c] = OpenPBR_LTC_Array[i][c];
}
