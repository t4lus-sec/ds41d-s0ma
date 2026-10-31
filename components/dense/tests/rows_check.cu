// s0ma leg 26: the batched-row paths against the per-row ones they replace, bit for bit, on random data at the
// verify step's shapes (DS41_ROWS_BATCHED=1 is exact only if every row's bits are the same).
//
//   nvcc -O2 -std=c++17 -arch=sm_120a -Icomponents/dense/include components/dense/tests/rows_check.cu \
//        components/dense/src/ops.cu -lcublas -o /tmp/rows_check && /tmp/rows_check
#include "ds41/dense.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { int e_ = int(x); if (e_) { std::printf("error %d at %s:%d\n", e_, __FILE__, __LINE__); std::exit(2); } } while (0)

template <class T> T *upload(const std::vector<T> &v) {
  T *d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T> std::vector<T> download(const T *d, size_t n) {
  std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
static bool same(const void *a, const void *b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; }

int main() {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  cublasHandle_t blas; if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) { std::printf("cublasCreate failed\n"); return 2; }
  int fail = 0;
  // the per-row cuBLAS weights of a verify step: the router gate (F32 384x5120), the hyper-connection mixes
  // (F32 24x20480), a BF16 row weight (512x5120) for completeness
  struct Shape { int N, K, f32; const char *what; } shapes[] = {
      {384, 5120, 1, "router gate F32 384x5120"}, {24, 20480, 1, "hyper-connection mixes F32 24x20480"},
      {512, 5120, 0, "BF16 512x5120"}};
  for (auto sh : shapes) {
    const size_t el = sh.f32 ? 4 : 2;
    for (int rows = 2; rows <= 6; ++rows) {
      std::vector<float> wf(size_t(sh.N) * sh.K), xf(size_t(rows) * sh.K);
      for (auto &v : wf) v = nd(rng) * 0.02f;
      for (auto &v : xf) v = nd(rng);
      std::vector<__nv_bfloat16> wb(wf.size()), xb(xf.size());
      for (size_t i = 0; i < wf.size(); ++i) wb[i] = __float2bfloat16(wf[i]);
      for (size_t i = 0; i < xf.size(); ++i) xb[i] = __float2bfloat16(xf[i]);
      void *w = sh.f32 ? (void *)upload(wf) : (void *)upload(wb);
      void *x = sh.f32 ? (void *)upload(xf) : (void *)upload(xb);
      void *y1, *y2; CK(cudaMalloc(&y1, rows * sh.N * el)); CK(cudaMalloc(&y2, rows * sh.N * el));
      for (int r = 0; r < rows; ++r)   // what the engine runs today: one M=1 call a row
        CK(ds41_dense_cublas(blas, (char *)x + r * sh.K * el, w, (char *)y1 + r * sh.N * el, 1, sh.N, sh.K, sh.f32, 1, 0));
      CK(ds41_dense_cublas_rows(blas, x, w, y2, rows, sh.N, sh.K, sh.f32, 0));
      CK(cudaDeviceSynchronize());
      auto a = download((char *)y1, rows * sh.N * el), b = download((char *)y2, rows * sh.N * el);
      bool ok = same(a.data(), b.data(), a.size());
      fail += !ok;
      std::printf("%s cuBLAS rows: %s, %d rows: %s\n", ok ? "ok  " : "FAIL", sh.what, rows,
                  ok ? "bit-identical to one call a row" : "DIFFERS from one call a row");
      cudaFree(w); cudaFree(x); cudaFree(y1); cudaFree(y2);
    }
  }
  // the hyper-connection mixes, whole: promote + stats, the mixes, Sinkhorn
  const int H = 5120, HC = 4;
  for (int rows = 2; rows <= 6; ++rows) {
    std::vector<__nv_bfloat16> h(size_t(rows) * HC * H);
    for (auto &v : h) v = __float2bfloat16(nd(rng));
    std::vector<float> fn(size_t(24) * HC * H), sc(3), base(24);
    for (auto &v : fn) v = nd(rng) * 0.01f;
    for (auto &v : sc) v = 0.5f + 0.1f * nd(rng);
    for (auto &v : base) v = 0.1f * nd(rng);
    auto *dh = upload(h); auto *dfn = upload(fn); auto *ds = upload(sc); auto *db = upload(base);
    float *out[2][6];
    const size_t sizes[6] = {size_t(rows) * 4, size_t(rows) * 4, size_t(rows) * 16, size_t(rows) * HC * H, size_t(rows) * 24, size_t(rows)};
    for (int v = 0; v < 2; ++v) for (int i = 0; i < 6; ++i) CK(cudaMalloc(&out[v][i], sizes[i] * 4));
    for (int r = 0; r < rows; ++r)
      CK(ds41_dense_hc_mixes(blas, dh + size_t(r) * HC * H, dfn, ds, db, out[0][0] + r * 4, out[0][1] + r * 4,
                             out[0][2] + r * 16, out[0][3] + size_t(r) * HC * H, out[0][4] + r * 24, out[0][5] + r,
                             1, H, 1e-20f, 1e-6f, 0));
    CK(ds41_dense_hc_mixes_rows(blas, dh, dfn, ds, db, out[1][0], out[1][1], out[1][2], out[1][3], out[1][4], out[1][5],
                                rows, H, 1e-20f, 1e-6f, 0));
    CK(cudaDeviceSynchronize());
    bool ok = true;
    const char *names[6] = {"pre", "post", "comb", "promoted", "mixes", "rstd"};
    for (int i = 0; i < 6; ++i) {
      auto a = download(out[0][i], sizes[i]), b = download(out[1][i], sizes[i]);
      if (!same(a.data(), b.data(), a.size() * 4)) { ok = false; std::printf("     %s differs\n", names[i]); }
    }
    fail += !ok;
    std::printf("%s hyper-connection mixes, %d rows: %s\n", ok ? "ok  " : "FAIL", rows,
                ok ? "every output bit-identical to one call a row" : "DIFFERS");
    for (int v = 0; v < 2; ++v) for (int i = 0; i < 6; ++i) cudaFree(out[v][i]);
    cudaFree(dh); cudaFree(dfn); cudaFree(ds); cudaFree(db);
  }
  std::printf(fail ? "%d FAILED: DS41_ROWS_BATCHED=1 would change tokens; leave it off\n" : "all passed: DS41_ROWS_BATCHED=1 keeps every row's bits\n", fail);
  return fail ? 1 : 0;
}
