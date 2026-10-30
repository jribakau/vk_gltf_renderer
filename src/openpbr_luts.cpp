/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openpbr_luts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

using namespace glm;
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;

// Adobe's arrays alone, without its closure: what its interop layer would define for them
#define OPENPBR_USE_TEXTURE_LUTS 0
#define OPENPBR_ENERGY_TABLES_USE_UINT16 1
#define OPENPBR_UINT16 std::uint16_t
#define OPENPBR_CONSTEXPR_GLOBAL static const
#define OPENPBR_MAYBE_CONSTEXPR_GLOBAL static const
#include "../shaders/openpbr/adobe/impl/data/openpbr_energy_arrays.h"
#include "../shaders/openpbr/adobe/impl/data/openpbr_ltc_array.h"

// Its own copy of the polyglot: the unit tests link the closure's, which defines the same functions
namespace {
#include "../shaders/openpbr/fv_openpbr_luts.h.slang"
#include "../shaders/openpbr/fv_openpbr_lut_table.hpp"
}  // namespace

std::vector<float> openPbrLutTable()
{
  std::vector<float> table(FV_OPENPBR_LUT_FLOATS);
  fvOpenPbrFillLuts(table.data());
  return table;
}
