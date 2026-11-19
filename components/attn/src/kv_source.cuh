// s0ma change 146: the attention kernels read their selected entries straight from the caches.
//
// Before, a gather wrote each row's 128 window entries and k compressed entries as BF16 [rows, width + k, 512] and
// the attention kernels read that copy. ds41_kv_load8 makes the eight BF16 values a kernel thread loads (entry slot,
// columns 8c..8c+7) from the FP8 window ring or the FP4 compressed cache with the gather's own operations, in its
// order: value = e4m3(byte) x 2^(scale-127) for the window, e2m1(nibble) x e4m3(scale) for the compressed entries, each
// one __fmul_rn and one __float2bfloat16_rn. The values are the gather's, bit for bit. Which slots are valid still
// comes from the local index the gather wrote (ds41_attn_gather_index writes only that).
#pragma once
#include "ds41/attention.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

__device__ __forceinline__ float ds41_kv_ex8(uint8_t c) { return __uint_as_float(unsigned(c) << 23); }
__device__ __forceinline__ float ds41_kv_exf8(uint8_t c) { return __half2float(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)); }
__device__ __forceinline__ float ds41_kv_exf4(uint8_t c) { return __half2float(__nv_cvt_fp4_to_halfraw(c, __NV_E2M1)); }
__device__ __forceinline__ uint32_t ds41_kv_pack(float lo, float hi) {
  return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) | (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}
// row: the query row within the launch (the gather's blockIdx.y); slot: its entry, valid (local index >= 0);
// c: which 8 of the 512 columns
__device__ __forceinline__ uint4 ds41_kv_load8(const ds41_attn_kv_source &s, int row, int slot, int c) {
  const int64_t pos = (s.dynamic_start ? *s.dynamic_start : s.start) + row;
  float v[8];
  if (slot < s.width) {   // the window ring: 512 e4m3 bytes, then 16 E8M0 scales (one a 32 columns)
    const int64_t p = s.prefill ? max(int64_t(0), pos - 127) + slot : pos - 127 + slot;
    const uint8_t *r = (const uint8_t *)s.swa + (p % s.capacity) * 528;
    const uint2 raw = *(const uint2 *)(r + c * 8);
    const float scale = ds41_kv_ex8(r[512 + c / 4]);
    const uint8_t *b = (const uint8_t *)&raw;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __fmul_rn(ds41_kv_exf8(b[i]), scale);
  } else {                // the compressed cache: 256 bytes of e2m1 pairs (low nibble first), then 32 e4m3 scales
    const int j = s.top_idx[int64_t(row) * s.k + slot - s.width];
    const uint8_t *r = (const uint8_t *)s.global_kv + int64_t(j) * 288;
    const uint32_t raw = *(const uint32_t *)(r + c * 4);
    const float scale = ds41_kv_exf8(r[256 + c / 2]);
    const uint8_t *b = (const uint8_t *)&raw;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __fmul_rn(ds41_kv_exf4((b[i / 2] >> (4 * (i & 1))) & 15), scale);
  }
  return make_uint4(ds41_kv_pack(v[0], v[1]), ds41_kv_pack(v[2], v[3]), ds41_kv_pack(v[4], v[5]), ds41_kv_pack(v[6], v[7]));
}
