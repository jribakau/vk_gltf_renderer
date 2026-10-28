/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SOBOL_H
#define SOBOL_H

// Owen-scrambled Sobol points for the accumulation's fixed sampling decisions (Burley 2020,
// "Practical Hash-based Owen Scrambling", JCGT 9(4)): each decision -- the pixel's jitter, the
// lens, a vertex's light sample, its BSDF sample -- draws one 4D point of a Sobol sequence at the
// pixel's sample index, with its own hashed shuffle of the index and scramble of each dimension.
// Padding decisions this way needs only the first four dimensions, and keeps every one of them a
// (0,2)-sequence in its first two dimensions: n = 2^k samples stratify each pair of them.
//
// A C++/Slang polyglot (tests/test_sobol.cpp): `reversebits` is the shader intrinsic.

// The first four dimensions' direction numbers (Joe and Kuo, new-joe-kuo-6.21201), most
// significant bit first: bit b of the index toggles SOBOL_DIRECTIONS[d][b].
static const uint SOBOL_DIRECTIONS[4][32] = {
    {0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u, 0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
     0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u, 0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
     0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u, 0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
     0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u, 0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u},
    {0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u, 0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
     0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u, 0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
     0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u, 0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
     0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u, 0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu},
    {0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u, 0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u,
     0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u, 0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u,
     0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u, 0x6868e800u, 0x9c9c5c00u, 0xeeee8e00u, 0x5555c500u,
     0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u, 0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u},
    {0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u, 0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u,
     0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u, 0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u,
     0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u, 0xa0858800u, 0x914e5400u, 0xdbe79e00u, 0x25db6d00u,
     0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u, 0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u}};

// Dimension `dim` (0..3) of Sobol point `index`, as 32 bits of a fraction. The directions above are
// made as the bits are walked, by each dimension's recurrence (its primitive polynomial: x + 1,
// x^2 + x + 1, x^3 + x + 1), in registers rather than from the table indexed at run time: with the
// table, Sobol made a still frame several times dearer than PCG (2026-10-28, 71 ms with PCG on the RT
// pipeline). The table stays for tests/test_sobol.cpp, which checks the two agree.
// The first dimension's directions are the bits themselves, reversed.
uint sobolBits(uint index, uint dim)
{
  if(dim == 0u)
    return reversebits(index);
  uint x = 0u;
  uint a = 0x80000000u;                                  // the direction of the bit walked
  uint b = 0xc0000000u;                                  // the next one
  uint c = (dim == 1u) ? 0xa0000000u : (dim == 2u) ? 0x60000000u : 0x20000000u;  // the one after
  while(index != 0u)
  {
    if((index & 1u) != 0u)
      x ^= a;
    index >>= 1u;
    uint next;
    if(dim == 1u)
      next = c ^ (c >> 1u);  // v[k] = v[k-1] ^ v[k-1] >> 1
    else if(dim == 2u)
      next = c ^ b ^ (b >> 2u);  // v[k] = v[k-1] ^ v[k-2] ^ v[k-2] >> 2
    else
      next = b ^ a ^ (a >> 3u);  // v[k] = v[k-2] ^ v[k-3] ^ v[k-3] >> 3
    a = b;
    b = c;
    c = next;
  }
  return x;
}

// Laine and Karras' permutation, Burley's improved hash: a random Owen scramble of the bits from the
// least significant up
uint laineKarrasPermutation(uint x, uint seed)
{
  x ^= x * 0x3d20adeau;
  x += seed;
  x *= (seed >> 16u) | 1u;
  x ^= x * 0x05526c56u;
  x ^= x * 0x53a22864u;
  return x;
}

// An Owen scramble from the most significant bit down: the permutation on the reversed bits
uint nestedUniformScramble(uint x, uint seed)
{
  return reversebits(laineKarrasPermutation(reversebits(x), seed));
}

// A well-mixed 32-bit hash (lowbias32), and Boost's hash_combine, for the per-decision seeds
uint sobolHash(uint x)
{
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

uint sobolHashCombine(uint seed, uint v)
{
  return seed ^ (v + 0x9e3779b9u + (seed << 6u) + (seed >> 2u));
}

// Point `index` of the Owen-scrambled 4D Sobol sequence that `seed` names, each coordinate in
// [0, 1): its top 24 bits, so a float never rounds it up to 1
float4 sobolOwen4(uint index, uint seed)
{
  const uint shuffled = nestedUniformScramble(index, seed);
  float4     u;
  for(uint d = 0u; d < 4u; d++)
    u[d] = float(nestedUniformScramble(sobolBits(shuffled, d), sobolHash(sobolHashCombine(seed, d))) >> 8u) * (1.0F / 16777216.0F);
  return u;
}

#endif  // SOBOL_H
