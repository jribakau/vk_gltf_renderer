/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

// The OpenPBR closure's lookup tables, as GltfScene::openPbrLuts holds them (shaders/openpbr/fv_openpbr_luts.h.slang)
std::vector<float> openPbrLutTable();
