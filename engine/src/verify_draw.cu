// s0ma: speculative acceptance on the GPU (DS41_SPARK_GPU_VERIFY=1, with DS41_SPARK_GPU_DRAW=1).
//
// The host path (speculation.cpp) builds p for every verify row in fp64 (probabilities_rows: first-maximum
// argmax, exp((double(x[i]) - x[b]) / t), a sequential normalizer, p[i] /= z), copies p and q into flat arrays and
// runs ds41_spark_verify_v2: draft k is accepted when u(k, ACCEPT) < min(1, p_k[d_k] / q_k[d_k]); the first
// rejection draws from max(0, p_k - q_k) with u(k, RESIDUAL); if every draft is accepted, the last row is drawn
// with u(count, TARGET). A draw is the inverse CDF: the first i whose running sum exceeds u * sum, the last
// positive id if none. u is spark_uniform_v2(seed, step, slot, purpose).
//
// This does the same arithmetic on the device from the target's logits (still on the card) and the drafter's q
// (left there by draft_draw.cu), with the same uniforms. Only the order of the sums differs (blocks, then a fixed
// order of blocks), so a result can differ from the host's only where a comparison falls within fp64 rounding
// (~1e-16) of its threshold. Anything the host would reject (nonfinite logits, a draft impossible under q, an
// empty residual) comes back as an error and the caller reruns the step's acceptance on the host, which reports it.
#include "ds41/spark_rng.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int G = 128, T = 256, R = 6;   // blocks over the vocabulary, threads a block, rows at most
constexpr uint32_t TARGET = 0, ACCEPT = 2, RESIDUAL = 3;   // ds41_spark_purpose

struct Out {
  int32_t tokens[R];
  int32_t emitted, accepted, rejected, error;   // error: 1 nonfinite logits, 2 draft impossible, 3 empty support
};

struct Scratch {
  float part_v[R][G];
  int part_i[R][G];
  double zpart[R][G];
  double dpart[G];
  int bad, row, residual;
  double u;
  Out out;
};

struct Drafts {
  const double *q[R - 1];   // q_k on the device, [n] f64 each
  int32_t d[R - 1];
};

__device__ void first_max(float &v, int &i, float v2, int i2) {
  if (i2 >= 0 && (i < 0 || v2 > v || (v2 == v && i2 < i))) { v = v2; i = i2; }
}

__device__ double block_sum(double x, double *sh) {   // a fixed tree: the same order every run
  const int t = threadIdx.x;
  sh[t] = x;
  __syncthreads();
  for (int d = T / 2; d; d >>= 1) {
    if (t < d) sh[t] += sh[t + d];
    __syncthreads();
  }
  const double r = sh[0];
  __syncthreads();
  return r;
}

__device__ void range_of(int n, int b, int &lo, int &hi) {
  const int chunk = (n + G - 1) / G;
  lo = min(n, b * chunk);
  hi = min(n, lo + chunk);
}

__device__ double total_of(const double *part) {
  double z = 0;
  for (int b = 0; b < G; ++b) z += part[b];
  return z;
}

// grid (G, rows): each block's first maximum of its slice of row y
__global__ void k_argmax(const float *x, int n, Scratch *s) {
  __shared__ float fv[T];
  __shared__ int fi[T];
  const int r = blockIdx.y;
  const float *row = x + size_t(r) * n;
  int lo, hi;
  range_of(n, blockIdx.x, lo, hi);
  float v = -INFINITY;
  int idx = -1;
  for (int i = lo + threadIdx.x; i < hi; i += T) {
    const float f = row[i];
    if (!isfinite(f)) s->bad = 1;
    first_max(v, idx, f, i);
  }
  fv[threadIdx.x] = v; fi[threadIdx.x] = idx;
  __syncthreads();
  for (int d = T / 2; d; d >>= 1) {
    if (threadIdx.x < d) {
      float a = fv[threadIdx.x]; int ai = fi[threadIdx.x];
      first_max(a, ai, fv[threadIdx.x + d], fi[threadIdx.x + d]);
      fv[threadIdx.x] = a; fi[threadIdx.x] = ai;
    }
    __syncthreads();
  }
  if (!threadIdx.x) { s->part_v[r][blockIdx.x] = fv[0]; s->part_i[r][blockIdx.x] = fi[0]; }
}

// p[r][i] = exp((double(x[r][i]) - x[r][b]) / t); each block's sum to zpart[r]
__global__ void k_exp(const float *x, int n, double t, double *p, Scratch *s) {
  __shared__ double sh[T];
  __shared__ int top_i;
  const int r = blockIdx.y;
  const float *row = x + size_t(r) * n;
  double *pr = p + size_t(r) * n;
  if (!threadIdx.x) {
    float v = -INFINITY;
    int idx = -1;
    for (int b = 0; b < G; ++b) first_max(v, idx, s->part_v[r][b], s->part_i[r][b]);
    top_i = idx < 0 ? 0 : idx;
  }
  __syncthreads();
  const double top = double(row[top_i]);
  int lo, hi;
  range_of(n, blockIdx.x, lo, hi);
  double z = 0;
  for (int i = lo + threadIdx.x; i < hi; i += T) {
    const double e = exp(__ddiv_rn(__dsub_rn(double(row[i]), top), t));
    pr[i] = e;
    z += e;
  }
  z = block_sum(z, sh);
  if (!threadIdx.x) s->zpart[r][blockIdx.x] = z;
}

// p[r][i] /= z[r]
__global__ void k_normalize(double *p, int n, Scratch *s) {
  __shared__ double z;
  const int r = blockIdx.y;
  double *pr = p + size_t(r) * n;
  if (!threadIdx.x) z = total_of(s->zpart[r]);
  __syncthreads();
  int lo, hi;
  range_of(n, blockIdx.x, lo, hi);
  for (int i = lo + threadIdx.x; i < hi; i += T) pr[i] = __ddiv_rn(pr[i], z);
}

// one thread: the accept tests in order; then which row is drawn from, with which uniform
__global__ void k_decide(const double *p, int n, int count, Drafts dr, uint64_t seed, uint64_t step, Scratch *s) {
  Out &o = s->out;
  o.emitted = o.accepted = o.rejected = o.error = 0;
  s->row = -1;
  if (s->bad) { o.error = 1; return; }
  for (int k = 0; k < count; ++k) {
    const int d = dr.d[k];
    const double qd = dr.q[k][d], pd = p[size_t(k) * n + d];
    if (!(qd > 0)) { o.error = 2; return; }
    const double u = spark_uniform_v2(seed, step, uint32_t(k), ACCEPT);
    if (u < fmin(1.0, __ddiv_rn(pd, qd))) { o.tokens[o.emitted++] = d; ++o.accepted; continue; }
    o.rejected = 1;
    s->row = k; s->residual = 1;
    s->u = spark_uniform_v2(seed, step, uint32_t(k), RESIDUAL);
    return;
  }
  s->row = count; s->residual = 0;
  s->u = spark_uniform_v2(seed, step, uint32_t(count), TARGET);
}

__device__ inline double weight(const double *pr, const double *qr, int i) {
  return qr ? fmax(0.0, __dsub_rn(pr[i], qr[i])) : pr[i];
}

__device__ const double *draw_q(const Drafts &dr, const Scratch *s) {
  return s->residual ? dr.q[s->row] : nullptr;
}

// grid G: each block's sum of the drawn row's weights (p, or max(0, p - q) after a rejection)
__global__ void k_draw_part(const double *p, int n, Drafts dr, Scratch *s) {
  __shared__ double sh[T];
  if (s->row < 0) return;   // an error: nothing to draw
  const double *pr = p + size_t(s->row) * n, *qr = draw_q(dr, s);
  int lo, hi;
  range_of(n, blockIdx.x, lo, hi);
  double z = 0;
  for (int i = lo + threadIdx.x; i < hi; i += T) z += weight(pr, qr, i);
  z = block_sum(z, sh);
  if (!threadIdx.x) s->dpart[blockIdx.x] = z;
}

// one block: the inverse-CDF draw over the drawn row (the same two-level search as draft_draw.cu)
__global__ void k_draw(const double *p, int n, Drafts dr, Scratch *s) {
  __shared__ double sub[T + 1];
  __shared__ int block, found, last, stop;
  __shared__ double base, target;
  const int t = threadIdx.x;
  if (s->row < 0) return;
  const double *pr = p + size_t(s->row) * n, *qr = draw_q(dr, s);
  if (!t) {
    const double total = total_of(s->dpart);
    stop = 0;
    if (!(total > 0)) { s->out.error = 3; stop = 1; }
    target = s->u * total;
    double c = 0;
    block = -1;
    last = -1;
    for (int j = 0; j < G; ++j) {
      if (s->dpart[j] > 0) last = j;
      if (block < 0 && target < c + s->dpart[j]) { block = j; base = c; }
      c += s->dpart[j];
    }
    found = INT32_MAX;
  }
  __syncthreads();
  if (stop) return;
  if (block < 0) {   // u * sum fell past every running sum (rounding): the host's last-positive fallback
    if (!t) {
      int lo, hi, id = -1;
      if (last >= 0) {
        range_of(n, last, lo, hi);
        for (int i = hi - 1; i >= lo; --i) if (weight(pr, qr, i) > 0) { id = i; break; }
      }
      if (id < 0) s->out.error = 3;
      else s->out.tokens[s->out.emitted++] = id;
    }
    return;
  }
  int lo, hi;
  range_of(n, block, lo, hi);
  const int len = hi - lo, chunk = (len + T - 1) / T;
  const int a = min(hi, lo + t * chunk), e = min(hi, a + chunk);
  double z = 0;
  for (int i = a; i < e; ++i) z += weight(pr, qr, i);
  sub[t] = z;
  __syncthreads();
  if (!t) {
    double c = base;
    for (int j = 0; j < T; ++j) { const double v = sub[j]; sub[j] = c; c += v; }
    sub[T] = c;
  }
  __syncthreads();
  if (target >= sub[t] && (t == T - 1 || target < sub[t + 1]) && a < e) {
    double c = sub[t];
    for (int i = a; i < e; ++i) {
      c += weight(pr, qr, i);
      if (target < c) { atomicMin(&found, i); break; }
    }
  }
  __syncthreads();
  if (!t) {
    int id = found;
    if (id == INT32_MAX) {   // inside the block by its sum, past every element by theirs: the last positive
      id = -1;
      for (int i = hi - 1; i >= lo; --i) if (weight(pr, qr, i) > 0) { id = i; break; }
    }
    if (id < 0) s->out.error = 3;
    else s->out.tokens[s->out.emitted++] = id;
  }
}
}  // namespace

extern "C" size_t ds41_verify_draw_scratch_bytes() { return sizeof(Scratch); }
extern "C" size_t ds41_verify_draw_result_bytes() { return sizeof(Out); }

// logits [count + 1, n] f32 on the device; q[k] [n] f64 on the device; d[k] the drafted ids; p [count + 1, n] f64
// device scratch; scratch from ds41_verify_draw_scratch_bytes(). *result points at the device copy of
// {int32 tokens[6], emitted, accepted, rejected, error} once the stream reaches it. temperature > 0 only (the
// caller floors it at 1e-5 as probabilities() does).
extern "C" int ds41_verify_draw_gpu(const float *logits, int n, int count, const double *const *q, const int32_t *d,
                                    double temperature, uint64_t seed, uint64_t step, double *p, void *scratch,
                                    void **result, cudaStream_t stream) {
  if (!logits || n < G || count < 0 || count > R - 1 || !p || !scratch || !result || !(temperature > 0) ||
      (count && (!q || !d)))
    return cudaErrorInvalidValue;
  Drafts dr{};
  for (int k = 0; k < count; ++k) {
    if (!q[k] || d[k] < 0 || d[k] >= n) return cudaErrorInvalidValue;
    dr.q[k] = q[k];
    dr.d[k] = d[k];
  }
  auto *s = static_cast<Scratch *>(scratch);
  *result = &s->out;
  const int rows = count + 1;
  cudaMemsetAsync(&s->bad, 0, sizeof(int), stream);
  k_argmax<<<dim3(G, rows), T, 0, stream>>>(logits, n, s);
  k_exp<<<dim3(G, rows), T, 0, stream>>>(logits, n, temperature, p, s);
  k_normalize<<<dim3(G, rows), T, 0, stream>>>(p, n, s);
  k_decide<<<1, 1, 0, stream>>>(p, n, count, dr, seed, step, s);
  k_draw_part<<<G, T, 0, stream>>>(p, n, dr, s);
  k_draw<<<1, T, 0, stream>>>(p, n, dr, s);
  return cudaGetLastError();
}
