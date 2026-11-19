// s0ma change 146: the attention kernels reading the caches themselves (ds41_attn_gather_index + ds41_attn_sparse_fused /
// ds41_attn_sparse_split_fused) against the gather and the kernels on its BF16 copy (ds41_attn_gather(_prefill_at) +
// ds41_attn_sparse / ds41_attn_sparse_split), on random caches at prefill's shapes:
//   the local index equal, every output bit for bit equal, for a cold chunk (the trailing-mask window, width 128 and a
//   short prompt's 50), a warm one (width 128), with 512 compressed entries a row (some masked: -1, out of range, or
//   not yet complete at the row's position) and with none; 1 and 256 rows; and the time of a 256-row chunk.
//
//   nvcc -O3 -std=c++17 -arch=sm_120a -ftz=false -DENABLE_BF16 -Icomponents/attn/include -I$TL/src \
//        -I$TL/3rdparty/cutlass/include components/attn/tests/fused_gather_check.cu components/attn/src/state.cu \
//        components/attn/src/attention.cu components/attn/src/split.cu -o /tmp/fused_gather_check && /tmp/fused_gather_check
//   (TL: the tilelang package ds41d's build uses, build/deps/tilelang)
#include "ds41/attention.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { int e_ = int(x); if (e_) { std::printf("error %d at %s:%d\n", e_, __FILE__, __LINE__); std::exit(2); } } while (0)

static std::mt19937 rng(146);
static int uni(int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); }
static uint8_t fp8_byte() { int v; do v = uni(0, 255); while ((v & 0x7f) == 0x7f); return uint8_t(v); }   // e4m3, no NaN

int main() {
  const int ROWS = 256, K = 512, CAP = 4096 + 128, GCAP = 3000, RATIO = 2;
  // the window ring: 512 e4m3 bytes and 16 E8M0 scales an entry; the compressed cache: 256 bytes of e2m1 pairs and 32
  // e4m3 scales (positive, finite)
  std::vector<uint8_t> swa(size_t(CAP) * 528), glob(size_t(GCAP) * 288);
  for (size_t e = 0; e < size_t(CAP); ++e) {
    for (int d = 0; d < 512; ++d) swa[e * 528 + d] = fp8_byte();
    for (int s = 0; s < 16; ++s) swa[e * 528 + 512 + s] = uint8_t(uni(112, 132));
  }
  for (size_t e = 0; e < size_t(GCAP); ++e) {
    for (int d = 0; d < 256; ++d) glob[e * 288 + d] = uint8_t(uni(0, 255));
    for (int s = 0; s < 32; ++s) { int v; do v = uni(0x08, 0x6f); while ((v & 0x7f) == 0x7f); glob[e * 288 + 256 + s] = uint8_t(v); }
  }
  std::vector<int32_t> ids(size_t(ROWS) * K);
  for (auto &j : ids) { int r = uni(0, 19); j = r == 0 ? -1 : r == 1 ? GCAP + uni(0, 9) : uni(0, GCAP - 1); }
  std::vector<uint16_t> q(size_t(ROWS) * 64 * 512);
  for (auto &v : q) { float f = std::normal_distribution<float>(0.f, 1.f)(rng); uint32_t b; std::memcpy(&b, &f, 4); v = uint16_t(b >> 16); }
  std::vector<float> sink(64);
  for (auto &v : sink) v = std::normal_distribution<float>(0.f, 1.f)(rng);

  void *dswa, *dglob, *dq, *dgathered, *do_ref, *do_new; int32_t *dids, *dloc_ref, *dloc_new; float *dsink, *dscores; int64_t *dstart;
  const size_t gathered = size_t(ROWS) * (128 + K) * 512 * 2, out = size_t(ROWS) * 64 * 512 * 2;
  CK(cudaMalloc(&dswa, swa.size())); CK(cudaMalloc(&dglob, glob.size())); CK(cudaMalloc(&dids, ids.size() * 4));
  CK(cudaMalloc(&dq, q.size() * 2)); CK(cudaMalloc(&dsink, 64 * 4)); CK(cudaMalloc(&dgathered, gathered));
  CK(cudaMalloc(&do_ref, out)); CK(cudaMalloc(&do_new, out)); CK(cudaMalloc(&dstart, 8));
  CK(cudaMalloc(&dloc_ref, size_t(ROWS) * (128 + K) * 4)); CK(cudaMalloc(&dloc_new, size_t(ROWS) * (128 + K) * 4));
  CK(cudaMalloc(&dscores, size_t(ROWS) * 64 * (128 + K) * 4));
  CK(cudaMemcpy(dswa, swa.data(), swa.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dglob, glob.data(), glob.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dsink, sink.data(), 64 * 4, cudaMemcpyHostToDevice));

  struct Case { const char *name; bool cold, split; int width, k, rows; int64_t start; };
  const Case cases[] = {
    {"cold chunk, 128 + 512, 256 rows      ", true, false, 128, 512, 256, 3000},
    {"cold chunk, 128 + 512, 1 row         ", true, false, 128, 512, 1, 3000},
    {"cold short prompt, 50 + 512, 256 rows", true, false, 50, 512, 256, 0},
    {"cold, window only (128 + 0), 256 rows", true, false, 128, 0, 256, 2000},
    {"warm chunk, split, 128 + 512, 256 rows", false, true, 128, 512, 256, 3500},
    {"warm chunk, split, 128 + 512, 1 row  ", false, true, 128, 512, 1, 3500},
    {"warm chunk, one kernel, 128 + 512    ", false, false, 128, 512, 256, 3500},
    {"warm, window only, 256 rows          ", false, false, 128, 0, 256, 3500},
  };
  std::vector<uint16_t> a(out / 2), b(out / 2);
  std::vector<int32_t> la(size_t(ROWS) * (128 + K)), lb(la.size());
  int failed = 0;
  for (const auto &c : cases) {
    const int count = c.width + c.k;
    ds41_attn_kv_source s{dswa, CAP, c.k ? dglob : nullptr, GCAP, c.k ? dids : nullptr, c.k, c.cold ? 0 : c.start,
                          c.cold ? dstart : nullptr, RATIO, c.width, c.cold ? 1 : 0};
    CK(cudaMemcpy(dstart, &c.start, 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(do_ref, 0xff, out)); CK(cudaMemset(do_new, 0xff, out));
    CK(cudaMemset(dloc_ref, 0x7f, la.size() * 4)); CK(cudaMemset(dloc_new, 0x7f, la.size() * 4));
    if (c.cold) CK(ds41_attn_gather_prefill_at(dswa, CAP, s.global_kv, GCAP, s.top_idx, c.k, dstart, c.rows, RATIO, c.width,
                                               dgathered, dloc_ref, nullptr));
    else CK(ds41_attn_gather(dswa, CAP, s.global_kv, GCAP, s.top_idx, c.k, c.start, c.rows, RATIO, dgathered, dloc_ref, nullptr));
    if (c.split) CK(ds41_attn_sparse_split(dq, dgathered, dsink, dloc_ref, do_ref, dscores, c.rows, count, 16, nullptr));
    else CK(ds41_attn_sparse(dq, dgathered, dsink, dloc_ref, do_ref, c.rows, count, 16, nullptr));
    CK(ds41_attn_gather_index(&s, c.rows, dloc_new, nullptr));
    if (c.split) CK(ds41_attn_sparse_split_fused(dq, &s, dsink, dloc_new, do_new, dscores, c.rows, count, 16, nullptr));
    else CK(ds41_attn_sparse_fused(dq, &s, dsink, dloc_new, do_new, c.rows, count, 16, nullptr));
    CK(cudaDeviceSynchronize());
    const size_t n_out = size_t(c.rows) * 64 * 512, n_loc = size_t(c.rows) * count;
    CK(cudaMemcpy(a.data(), do_ref, n_out * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), do_new, n_out * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(la.data(), dloc_ref, n_loc * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(lb.data(), dloc_new, n_loc * 4, cudaMemcpyDeviceToHost));
    size_t diff = 0, ldiff = 0, masked = 0, first = 0;
    for (size_t i = 0; i < n_out; ++i) if (a[i] != b[i]) { if (!diff) first = i; ++diff; }
    for (size_t i = 0; i < n_loc; ++i) { ldiff += la[i] != lb[i]; masked += la[i] < 0; }
    if (diff || ldiff) {
      ++failed;
      std::printf("  %s: %zu of %zu outputs differ (first %zu: %04x against %04x), %zu index entries differ\n", c.name, diff,
                  n_out, first, b[first], a[first], ldiff);
    } else std::printf("  %s: bit for bit (%zu outputs; %zu of %zu entries masked)\n", c.name, n_out, masked, n_loc);
  }
  // time: a 256-row chunk with 512 compressed entries a row, both forms of the cold path and the warm split
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int split = 0; split < 2; ++split) {
    const int64_t start = 3500; CK(cudaMemcpy(dstart, &start, 8, cudaMemcpyHostToDevice));
    ds41_attn_kv_source s{dswa, CAP, dglob, GCAP, dids, K, split ? start : 0, split ? nullptr : dstart, RATIO, 128, split ? 0 : 1};
    float old_ms = 0, new_ms = 0, ms;
    for (int rep = 0; rep < 6; ++rep) {
      CK(cudaEventRecord(e0));
      if (split) { CK(ds41_attn_gather(dswa, CAP, dglob, GCAP, dids, K, start, ROWS, RATIO, dgathered, dloc_ref, nullptr));
                   CK(ds41_attn_sparse_split(dq, dgathered, dsink, dloc_ref, do_ref, dscores, ROWS, 128 + K, 16, nullptr)); }
      else { CK(ds41_attn_gather_prefill_at(dswa, CAP, dglob, GCAP, dids, K, dstart, ROWS, RATIO, 128, dgathered, dloc_ref, nullptr));
             CK(ds41_attn_sparse(dq, dgathered, dsink, dloc_ref, do_ref, ROWS, 128 + K, 16, nullptr)); }
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1)); if (rep) old_ms += ms / 5;
      CK(cudaEventRecord(e0));
      CK(ds41_attn_gather_index(&s, ROWS, dloc_new, nullptr));
      if (split) CK(ds41_attn_sparse_split_fused(dq, &s, dsink, dloc_new, do_new, dscores, ROWS, 128 + K, 16, nullptr));
      else CK(ds41_attn_sparse_fused(dq, &s, dsink, dloc_new, do_new, ROWS, 128 + K, 16, nullptr));
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1)); if (rep) new_ms += ms / 5;
    }
    std::printf("  256 rows, %s: %.3f ms gathered, %.3f ms from the caches\n", split ? "warm (split)" : "cold", old_ms, new_ms);
  }
  if (failed) { std::printf("FAIL: %d cases differ\n", failed); return 1; }
  std::printf("ok: the fused attention is bit for bit the gathered one\n");
  return 0;
}
