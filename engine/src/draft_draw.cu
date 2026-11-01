// s0ma: a draft token drawn on the GPU (DS41_SPARK_GPU_DRAW=1).
//
// The host path (speculation.cpp) copies each drafted position's 129,280 logits to the host, builds q in fp64
// (probabilities(): first-maximum argmax, exp((double(x[i]) - x[b]) / t), a sequential normalizer, p[i] /= z) and
// draws from it (ds41_spark_sample_v2: u = spark_uniform_v2(seed, step, slot, purpose), the first i whose running
// sum exceeds u * sum(q), the last positive id if none). This does the same arithmetic on the device, in fp64, from
// the logits the Markov head just wrote, and hands back the token, max q (the stop rule's confidence) and, when
// sampling, q itself for the verifier. Only the order of the sums differs (blocks, then a fixed order of blocks), so
// a draw can differ from the host's only where u * sum(q) falls within fp64 rounding (~1e-16) of a running sum.
//
// Greedy (t = 0): the token is the first maximum, and the confidence is the largest probability at t = 1 (1 / z),
// as the host computes for its stop rule.
#include "ds41/spark_rng.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int G = 128, T = 256;   // blocks over the vocabulary, threads a block

struct Scratch {
  float part_v[G];
  int part_i[G];
  double zpart[G], qpart[G];
  int bad;
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

__global__ void k_argmax(const float *x, int n, Scratch *s) {
  __shared__ float fv[T];
  __shared__ int fi[T];
  int lo, hi;
  range_of(n, blockIdx.x, lo, hi);
  float v = -INFINITY;
  int idx = -1;
  for (int i = lo + threadIdx.x; i < hi; i += T) {
    const float f = x[i];
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
  if (!threadIdx.x) { s->part_v[blockIdx.x] = fv[0]; s->part_i[blockIdx.x] = fi[0]; }
}

__device__ int global_argmax(const Scratch *s) {   // every block reduces the partials the same way
  float v = -INFINITY;
  int idx = -1;
  for (int b = 0; b < G; ++b) first_max(v, idx, s->part_v[b], s->part_i[b]);
  return idx;
}

// e[i] = exp((double(x[i]) - x[b]) / t), written to q when sampling; each block's sum to zpart
__global__ void k_exp(const float *x, int n, double t, double *q, Scratch *s) {
  __shared__ double sh[T];
  __shared__ int top_i;
  if (!threadIdx.x) top_i = global_argmax(s);
  __syncthreads();
  const double top = double(x[top_i]);
  int lo, hi;
  range_of(n, blockIdx.x, lo, hi);
  double z = 0;
  for (int i = lo + threadIdx.x; i < hi; i += T) {
    const double e = exp(__dsub_rn(double(x[i]), top) / t);
    if (q) q[i] = e;
    z += e;
  }
  z = block_sum(z, sh);
  if (!threadIdx.x) s->zpart[blockIdx.x] = z;
}

__device__ double total_of(const double *part) {
  double z = 0;
  for (int b = 0; b < G; ++b) z += part[b];
  return z;
}

// q[i] /= z; each block's sum of q to qpart
__global__ void k_normalize(double *q, int n, Scratch *s) {
  __shared__ double sh[T];
  __shared__ double z;
  if (!threadIdx.x) z = total_of(s->zpart);
  __syncthreads();
  int lo, hi;
  range_of(n, blockIdx.x, lo, hi);
  double sum = 0;
  for (int i = lo + threadIdx.x; i < hi; i += T) {
    const double p = q[i] / z;
    q[i] = p;
    sum += p;
  }
  sum = block_sum(sum, sh);
  if (!threadIdx.x) s->qpart[blockIdx.x] = sum;
}

struct Result {
  int32_t token, argmax;
  double confidence, z;
};

// one block: the confidence, then the inverse-CDF draw (sampling) or the first maximum (greedy)
__global__ void k_finish(const double *q, int n, int greedy, uint64_t seed, uint64_t step, uint32_t slot,
                         uint32_t purpose, Scratch *s, Result *out) {
  __shared__ double sub[T + 1];
  __shared__ int block, found, last;
  __shared__ double base, target;
  const int t = threadIdx.x;
  if (!t) {
    const int b = global_argmax(s);
    const double z = total_of(s->zpart);
    out->argmax = b;
    out->z = z;
    if (s->bad) { out->token = -2; out->confidence = NAN; }
    else if (greedy) { out->token = b; out->confidence = 1.0 / z; }
    else {
      out->confidence = q[b];
      const double total = total_of(s->qpart);
      target = spark_uniform_v2(seed, step, slot, purpose) * total;
      double c = 0;
      block = -1;
      last = -1;
      for (int j = 0; j < G; ++j) {
        if (s->qpart[j] > 0) last = j;
        if (block < 0 && target < c + s->qpart[j]) { block = j; base = c; }
        c += s->qpart[j];
      }
    }
    found = INT32_MAX;
  }
  __syncthreads();
  if (greedy || s->bad) return;
  if (block < 0) {   // u * sum fell past every running sum (rounding): the host's last-positive fallback
    if (!t) {
      int lo, hi, id = -1;
      if (last >= 0) {
        range_of(n, last, lo, hi);
        for (int i = hi - 1; i >= lo; --i) if (q[i] > 0) { id = i; break; }
      }
      out->token = id;
    }
    return;
  }
  int lo, hi;
  range_of(n, block, lo, hi);
  const int len = hi - lo, chunk = (len + T - 1) / T;
  const int a = min(hi, lo + t * chunk), e = min(hi, a + chunk);
  double z = 0;
  for (int i = a; i < e; ++i) z += q[i];
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
      c += q[i];
      if (target < c) { atomicMin(&found, i); break; }
    }
  }
  __syncthreads();
  if (!t) {
    int id = found;
    if (id == INT32_MAX) {   // inside the block by its sum, past every element by theirs: the last positive
      id = -1;
      for (int i = hi - 1; i >= lo; --i) if (q[i] > 0) { id = i; break; }
    }
    out->token = id;
  }
}
}  // namespace

extern "C" size_t ds41_draft_draw_scratch_bytes() { return sizeof(Scratch) + sizeof(Result) + 64; }

// logits [n] f32 on the device; q [n] f64 on the device (written when temperature > 0); scratch from
// ds41_draft_draw_scratch_bytes(); result: {int32 token, int32 argmax, f64 confidence, f64 z} at scratch's end.
extern "C" int ds41_draft_draw_gpu(const float *logits, int n, double temperature, uint64_t seed, uint64_t step,
                                   uint32_t slot, uint32_t purpose, double *q, void *scratch, void **result,
                                   cudaStream_t stream) {
  if (!logits || n < G || !scratch || !result || !(temperature >= 0) || (temperature > 0 && !q))
    return cudaErrorInvalidValue;
  auto *s = static_cast<Scratch *>(scratch);
  auto *r = reinterpret_cast<Result *>(static_cast<char *>(scratch) + ((sizeof(Scratch) + 15) / 16) * 16);
  *result = r;
  cudaMemsetAsync(&s->bad, 0, sizeof(int), stream);
  const bool greedy = temperature == 0;
  k_argmax<<<G, T, 0, stream>>>(logits, n, s);
  k_exp<<<G, T, 0, stream>>>(logits, n, greedy ? 1.0 : temperature, greedy ? nullptr : q, s);
  if (!greedy) k_normalize<<<G, T, 0, stream>>>(q, n, s);
  k_finish<<<1, T, 0, stream>>>(q, n, greedy, seed, step, slot, purpose, s, r);
  return cudaGetLastError();
}
