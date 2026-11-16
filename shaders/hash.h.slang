/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Wellons' lowbias32: a well-mixed 32-bit integer hash, for the shaders' noises and seeds (flakes,
// rain, haze, Sobol's per-decision seeds). Plain integer code, so the headers the unit tests compile as
// C++ take it too (a `uint` type in scope).
#ifndef HASH_H_SLANG
#define HASH_H_SLANG

inline uint lowbias32(uint x)
{
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

#endif  // HASH_H_SLANG
