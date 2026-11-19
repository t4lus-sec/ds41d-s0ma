// s0ma change 145: ds41_dense_fp8_rows (the Engram projection, K 6144, many rows a launch) against ds41_dense_fp8's
// native variants on each 8-row slice, on random data at the Engram shape (N 25,600 = 5 x 5,120, K 6,144):
//   every output bit for bit equal to variant 2 (what prefill's Engram rows use) and to variant 4 (the K 6144 default),
//   at 1, 7, 8, 9, 63, 64, 65, 100, 255 and 256 rows; and the time of 256 rows: 32 launches of 8 against one.
//
//   nvcc -O2 -std=c++17 -arch=sm_120a -Icomponents/dense/include components/dense/tests/fp8_rows_check.cu \
//        components/dense/src/dense.cu -o /tmp/fp8_rows_check && /tmp/fp8_rows_check
#include "ds41/dense.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { int e_ = int(x); if (e_) { std::printf("error %d at %s:%d\n", e_, __FILE__, __LINE__); std::exit(2); } } while (0)
using bf16 = __nv_bfloat16;

int main() {
  const int N = 25600, K = 6144, NG = K / 32, MAXM = 256;
  std::mt19937 rng(145);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::uniform_int_distribution<int> byte(0, 255), scale(118, 132), mag(-6, 6);
  // activations: rows of different magnitudes, a few outliers (the per-32 quantization sees all of it)
  std::vector<bf16> x(size_t(MAXM) * K);
  for (int r = 0; r < MAXM; ++r) {
    const float m = std::ldexp(1.f, mag(rng));
    for (int k = 0; k < K; ++k) {
      float v = normal(rng) * m;
      if (byte(rng) == 0) v *= 40.f;
      x[size_t(r) * K + k] = __float2bfloat16_rn(v);
    }
  }
  // weights: any finite e4m3 byte (0x7f and 0xff are NaN), scales around 2^0
  std::vector<uint8_t> w(size_t(N) * K), ws(size_t(N / 32) * NG);
  for (auto &b : w) { int v; do v = byte(rng); while ((v & 0x7f) == 0x7f); b = uint8_t(v); }
  for (auto &s : ws) s = uint8_t(scale(rng));

  bf16 *dx, *dy, *dref; uint8_t *daq, *das, *dw, *dws;
  CK(cudaMalloc(&dx, x.size() * 2)); CK(cudaMalloc(&daq, size_t(MAXM) * K)); CK(cudaMalloc(&das, size_t(MAXM) * NG));
  CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dws, ws.size()));
  CK(cudaMalloc(&dy, size_t(MAXM) * N * 2)); CK(cudaMalloc(&dref, size_t(MAXM) * N * 2));
  CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws, ws.data(), ws.size(), cudaMemcpyHostToDevice));
  CK(ds41_dense_quant(dx, daq, das, MAXM, K, nullptr));

  std::vector<uint16_t> got(size_t(MAXM) * N), want(size_t(MAXM) * N);
  int failed = 0;
  for (int variant : {2, 4}) {
    for (int M : {1, 7, 8, 9, 63, 64, 65, 100, 255, 256}) {
      CK(cudaMemset(dref, 0xff, size_t(M) * N * 2)); CK(cudaMemset(dy, 0xff, size_t(M) * N * 2));
      for (int r0 = 0; r0 < M; r0 += 8)
        CK(ds41_dense_fp8(daq + size_t(r0) * K, das + size_t(r0) * NG, dw, dws, dref + size_t(r0) * N,
                          std::min(8, M - r0), N, K, variant, nullptr));
      CK(ds41_dense_fp8_rows(daq, das, dw, dws, dy, M, N, K, nullptr));
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(want.data(), dref, size_t(M) * N * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(got.data(), dy, size_t(M) * N * 2, cudaMemcpyDeviceToHost));
      size_t diff = 0, first = 0;
      for (size_t i = 0; i < size_t(M) * N; ++i) if (got[i] != want[i]) { if (!diff) first = i; ++diff; }
      if (diff) {
        ++failed;
        std::printf("  variant %d, %3d rows: %zu of %zu outputs differ (first: row %zu col %zu, %04x against %04x)\n",
                    variant, M, diff, size_t(M) * N, first / N, first % N, got[first], want[first]);
      } else std::printf("  variant %d, %3d rows: bit for bit (%zu outputs)\n", variant, M, size_t(M) * N);
    }
  }
  // time: 256 rows, as prefill groups them now (32 launches of 8, variant 2) and with this change (one launch)
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float old_ms = 0, new_ms = 0;
  for (int rep = 0; rep < 6; ++rep) {
    CK(cudaEventRecord(a));
    for (int r0 = 0; r0 < MAXM; r0 += 8)
      CK(ds41_dense_fp8(daq + size_t(r0) * K, das + size_t(r0) * NG, dw, dws, dref + size_t(r0) * N, 8, N, K, 2, nullptr));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); if (rep) old_ms += ms / 5;
    CK(cudaEventRecord(a));
    CK(ds41_dense_fp8_rows(daq, das, dw, dws, dy, MAXM, N, K, nullptr));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b)); if (rep) new_ms += ms / 5;
  }
  std::printf("  256 rows: %.3f ms as 32 launches of 8 rows, %.3f ms as one launch (%.1fx)\n", old_ms, new_ms,
              new_ms > 0 ? old_ms / new_ms : 0.f);
  if (failed) { std::printf("FAIL: %d cases differ\n", failed); return 1; }
  std::printf("ok: ds41_dense_fp8_rows is bit for bit the 8-row launches\n");
  return 0;
}
