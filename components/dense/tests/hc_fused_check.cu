// s0ma change 128: the fused hyper-connection site (DS41_HC_FUSED=1) against the launches it replaces, on random data
// at the decode shapes (K 5120, 1..8 rows):
//   ds41_dense_hc_pre_norm: bit for bit hcpre + the decode RMSNorm (a copy of ds41e::hotpath::rms below) + the FP32 copy
//   ds41_dense_hc_fused_mixes: pre, post and comb within 1e-4 of ds41_dense_hc_mixes (only the order of its 20,480-term
//     sums differs); every row bit for bit the same alone as among the others; its count left at zero
//
//   nvcc -O2 -std=c++17 -arch=sm_120a -Icomponents/dense/include components/dense/tests/hc_fused_check.cu \
//        components/dense/src/ops.cu -lcublas -o /tmp/hc_fused_check && /tmp/hc_fused_check
#include "ds41/dense.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { int e_ = int(x); if (e_) { std::printf("error %d at %s:%d\n", e_, __FILE__, __LINE__); std::exit(2); } } while (0)
using bf16 = __nv_bfloat16;

// ds41e::hotpath::rms (engine/src/hotpath/kernels.cu), launched as rms_decode launches it
__global__ void ref_rms(const bf16 *x, const bf16 *w, bf16 *y, int k) {
  __shared__ float sh[512];
  int r = blockIdx.x, t = threadIdx.x;
  float v[4] = {0, 0, 0, 0};
  for (int i = t * 4; i < k; i += blockDim.x * 4)
    for (int j = 0; j < 4; ++j) if (i + j < k) { float z = __bfloat162float(x[r * k + i + j]); v[j] = __fadd_rn(v[j], __fmul_rn(z, z)); }
  float sum = __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[1]), v[2]), v[3]);
  sh[t] = sum; __syncthreads();
  for (int d = blockDim.x / 2; d >= 32; d >>= 1) { if (t < d) { sum = __fadd_rn(sum, sh[t + d]); sh[t] = sum; } __syncthreads(); }
  if (t < 32) {
    for (int d = 16; d; d >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, d));
    if (!t) sh[0] = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.f / k), 1e-20f));
  }
  __syncthreads();
  for (int i = t; i < k; i += blockDim.x)
    y[r * k + i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[r * k + i]), sh[0]), __bfloat162float(w[i])));
}
__global__ void ref_cast(const bf16 *x, float *y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __bfloat162float(x[i]); }

template <class T> T *upload(const std::vector<T> &v) {
  T *d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T> std::vector<T> download(const T *d, size_t n) {
  std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}

int main() {
  constexpr int K = 5120, K4 = 4 * K;
  std::mt19937 rng(128);
  std::normal_distribution<float> nd(0.f, 1.f);
  cublasHandle_t blas; if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) { std::printf("cublasCreate failed\n"); return 2; }
  int fail = 0;
  // the checkpoint's scales: fn ~ 1e-2, residual streams of a few units, base and scale of order one
  std::vector<float> fn(size_t(24) * K4), sc(3), base(24);
  for (auto &v : fn) v = 0.01f * nd(rng);
  for (auto &v : sc) v = 0.5f + 0.1f * nd(rng);
  for (auto &v : base) v = 0.3f * nd(rng);
  std::vector<bf16> wn(K);
  for (auto &v : wn) v = __float2bfloat16(1.f + 0.1f * nd(rng));
  float *dfn = upload(fn), *dsc = upload(sc), *dbase = upload(base);
  bf16 *dwn = upload(wn);
  const size_t pf = ds41_dense_hc_fused_floats(K);
  float *partial; unsigned *count;
  CK(cudaMalloc(&partial, pf * 4)); CK(cudaMalloc(&count, 4)); CK(cudaMemset(count, 0, 4));
  for (int M = 1; M <= 8; ++M) {
    std::vector<bf16> h(size_t(M) * K4);
    for (auto &v : h) v = __float2bfloat16(2.f * nd(rng));
    std::vector<float> incoming(size_t(M) * 4);
    for (auto &v : incoming) v = 0.25f + 0.05f * nd(rng);
    bf16 *dh = upload(h); float *dinc = upload(incoming);
    float *rpre, *rpost, *rcomb, *rx, *rmix, *rrstd, *fpre, *fpost, *fcomb, *yf, *rf;
    bf16 *rpre_x, *ry, *y;
    CK(cudaMalloc(&rpre, M * 16)); CK(cudaMalloc(&rpost, M * 16)); CK(cudaMalloc(&rcomb, M * 64));
    CK(cudaMalloc(&rx, size_t(M) * K4 * 4)); CK(cudaMalloc(&rmix, M * 96)); CK(cudaMalloc(&rrstd, M * 4));
    CK(cudaMalloc(&fpre, M * 16)); CK(cudaMalloc(&fpost, M * 16)); CK(cudaMalloc(&fcomb, M * 64));
    CK(cudaMalloc(&rpre_x, size_t(M) * K * 2)); CK(cudaMalloc(&ry, size_t(M) * K * 2)); CK(cudaMalloc(&y, size_t(M) * K * 2));
    CK(cudaMalloc(&yf, size_t(M) * K * 4)); CK(cudaMalloc(&rf, size_t(M) * K * 4));
    // the reference: the engine's per-row path (M=1 calls) and hcpre + RMSNorm + cast
    for (int r = 0; r < M; ++r)
      CK(ds41_dense_hc_mixes(blas, dh + size_t(r) * K4, dfn, dsc, dbase, rpre + r * 4, rpost + r * 4, rcomb + r * 16,
                             rx + size_t(r) * K4, rmix + r * 24, rrstd + r, 1, K, 1e-20f, 1e-6f, 0));
    CK(ds41_dense_hc_pre(dh, dinc, rpre_x, M, K, 0));
    ref_rms<<<M, 512>>>(rpre_x, dwn, ry, K);
    ref_cast<<<(M * K + 255) / 256, 256>>>(ry, rf, M * K);
    // the fused launches, twice (the count must come back to zero)
    for (int rep = 0; rep < 2; ++rep) {
      CK(ds41_dense_hc_fused_mixes(dh, dfn, dsc, dbase, fpre, fpost, fcomb, partial, count, M, K, 1e-20f, 1e-6f, 0));
      CK(ds41_dense_hc_pre_norm(dh, dinc, dwn, y, yf, M, K, 0));
    }
    CK(cudaDeviceSynchronize());
    // 1. hcpre + norm + cast: bit for bit
    auto a = download(ry, size_t(M) * K), b = download(y, size_t(M) * K);
    auto af = download(rf, size_t(M) * K), bf = download(yf, size_t(M) * K);
    const bool same_x = std::memcmp(a.data(), b.data(), a.size() * 2) == 0 && std::memcmp(af.data(), bf.data(), af.size() * 4) == 0;
    // 2. the coefficients against the reference
    auto p0 = download(rpre, M * 4), p1 = download(fpre, M * 4), q0 = download(rpost, M * 4), q1 = download(fpost, M * 4);
    auto c0 = download(rcomb, M * 16), c1 = download(fcomb, M * 16);
    double worst = 0; size_t equal = 0, total = 0;
    auto cmp = [&](const std::vector<float> &u, const std::vector<float> &v) {
      for (size_t i = 0; i < u.size(); ++i) { worst = std::max(worst, double(std::fabs(u[i] - v[i]))); equal += u[i] == v[i]; ++total; }
    };
    cmp(p0, p1); cmp(q0, q1); cmp(c0, c1);
    // 3. every row alone: the same bits as among the others
    bool invariant = true;
    for (int r = 0; r < M && M > 1; ++r) {
      CK(ds41_dense_hc_fused_mixes(dh + size_t(r) * K4, dfn, dsc, dbase, fpre, fpost, fcomb, partial, count, 1, K, 1e-20f, 1e-6f, 0));
      CK(cudaDeviceSynchronize());
      auto s0 = download(fpre, 4), s1 = download(fpost, 4), s2 = download(fcomb, 16);
      invariant = invariant && std::memcmp(s0.data(), p1.data() + r * 4, 16) == 0 && std::memcmp(s1.data(), q1.data() + r * 4, 16) == 0 &&
                  std::memcmp(s2.data(), c1.data() + r * 16, 64) == 0;
    }
    unsigned left = 1; CK(cudaMemcpy(&left, count, 4, cudaMemcpyDeviceToHost));
    const bool ok = same_x && worst < 1e-4 && invariant && left == 0;
    std::printf("rows %d: hcpre+norm+cast %s; pre/post/comb max |diff| %.2e (%zu of %zu bit for bit); rows alone %s; count %u  %s\n",
                M, same_x ? "bit for bit" : "DIFFERS", worst, equal, total, invariant ? "the same" : "DIFFER", left, ok ? "ok" : "FAIL");
    fail += !ok;
    for (void *p : {(void *)dh, (void *)dinc, (void *)rpre, (void *)rpost, (void *)rcomb, (void *)rx, (void *)rmix, (void *)rrstd,
                    (void *)fpre, (void *)fpost, (void *)fcomb, (void *)rpre_x, (void *)ry, (void *)y, (void *)yf, (void *)rf})
      cudaFree(p);
  }
  std::printf(fail ? "%d FAILED\n" : "all passed\n", fail);
  return fail ? 1 : 0;
}
