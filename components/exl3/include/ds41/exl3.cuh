#pragma once
#include <ds41/kernels.cuh>

namespace ds41 {
// K=2..5 (exl3_set_bits; s0ma changes 12 and 15), mul1 EXL3. DeviceExpert weights are [K/16,N/16,16*bits] uint16 (48 for K3, 32 for K2);
// each scale pointer holds suh[K] followed immediately by svh[N], both FP16.
// GemmJob::x is BF16 [tokens,K], output BF16 [tokens,N]. x_scale is ignored.
// Device-resident jobs/table are reread on every invocation and graph replay.
// Dimensions must be positive multiples of 128; k <= 8192 (s0ma change 34; grouped_exl3_var: k <= 5120). No scratch.
// Decode supports k=128,256,1280,2304,4096,5120,8192 (k=8192: jobs of at most 4 rows); prefill supports any such multiple.
// Required DS41 dimensions: gate/up k=5120,n=2304; down k=2304,n=5120.
// Trellis addresses need 4-byte alignment, BF16/FP16 addresses 2-byte alignment.
// Decode tokens: 1..8. Prefill tokens: 1..max_tokens, max_tokens <= 256.
cudaError_t grouped_exl3(const DeviceExpert* table, const GemmJob* jobs, int count,
                         Projection projection, int n, int k, cudaStream_t stream);
cudaError_t grouped_exl3_prefill(const DeviceExpert* table, const GemmJob* jobs, int count,
                         Projection projection, int n, int k, int max_tokens, cudaStream_t stream);
// The bank's bits per weight, 3 (default) or 2: set once, before any launch or graph capture; every launch after
// it decodes that width. Not thread-safe against concurrent launches (the engine sets it at load).
cudaError_t exl3_set_bits(int bits);
int exl3_bits();
// s0ma change 12: the same launches at an explicit width (2 or 3), for a bank whose layers differ in width.
cudaError_t grouped_exl3_bits(const DeviceExpert* table, const GemmJob* jobs, int count,
                              Projection projection, int n, int k, int bits, cudaStream_t stream);
cudaError_t grouped_exl3_prefill_bits(const DeviceExpert* table, const GemmJob* jobs, int count,
                              Projection projection, int n, int k, int max_tokens, int bits, cudaStream_t stream);
// s0ma change 15: widths 2..5 (K4: 32 words a tile; K5: 40, as exllamav3's dq_dispatch<5>). grouped_exl3_var launches
// jobs of mixed widths at once: CTA y decodes at bits[jobs[y].expert] (a device array indexed by expert id).
// s0ma change 34 (N1): one dense tensor, table[job.expert] (its gate and gate_scale views), the job by value so a
// captured graph holds it. Decode: job.tokens 1..8 (1..4 for k=8192), k in 128, 256, 1280, 2304, 5120, 8192;
// prefill: 1..256 rows, any k multiple of 128 up to 8192. bits 2..5. x_stride/out_stride: the rows' strides in
// elements, 0 for k and n (a slice of wider rows otherwise; decode also takes k=4096, 0xSero's wo_a groups).
// Decode also runs `groups` tensors at consecutive table entries in one launch: group g is entry job.expert + g, reads
// job.x + g*group_x and writes job.output + g*group_out (elements); 0xSero's wo_a slices.
// s0ma change 37: with scratch (partial: exl3_dense_splitk_floats() floats; counts: groups*n/128 zeroed slots, left
// zeroed), a tensor of few column tiles runs with its K split across CTAs, partials added in a fixed order.
cudaError_t exl3_dense(const DeviceExpert* table, GemmJob job, int n, int k, int bits, cudaStream_t stream,
                       int x_stride = 0, int out_stride = 0, int groups = 1, int group_x = 0, int group_out = 0,
                       float* partial = nullptr, size_t partial_floats = 0, unsigned* counts = nullptr,
                       size_t count_slots = 0);
size_t exl3_dense_splitk_floats(int n, int k, int groups);
// s0ma change 40: the routed experts' grouped launch with K split across CTAs (partials summed in a fixed order; the
// same products in another order). bits: a device width per expert id, or nullptr and `uniform`. Scratch as above:
// exl3_var_splitk_floats() floats and count*n/128 zeroed slots, left zeroed. Without enough, the unsplit launch.
size_t exl3_var_splitk_floats(int n, int k, int count);
cudaError_t grouped_exl3_splitk(const DeviceExpert* table, const GemmJob* jobs, int count, Projection projection, int n,
                                int k, const uint8_t* bits, int uniform, float* partial, size_t partial_floats,
                                unsigned* counts, size_t count_slots, cudaStream_t stream);
// s0ma change 37: prefill as exllamav3 does it for many rows: W [k, n] FP16 decoded from table[index] (row = input
// feature), the rows rotated into FP16 [rows, k], the caller's FP16 GEMM into FP32 [rows, n], then rotated and scaled
// into BF16 (out_stride 0: n).
cudaError_t exl3_dense_reconstruct(const DeviceExpert* table, int index, int n, int k, int bits, half* w, cudaStream_t stream);
cudaError_t exl3_dense_rotate_in(const DeviceExpert* table, int index, const __nv_bfloat16* x, int rows, int k,
                                 int x_stride, half* a, cudaStream_t stream);
cudaError_t exl3_dense_rotate_out(const DeviceExpert* table, int index, const float* c, int rows, int n, int k,
                                  __nv_bfloat16* out, int out_stride, cudaStream_t stream);
// s0ma change 68: an expert's prefill with its rows rotated once. exl3_prefill_rotate writes job[0]'s rows (BF16,
// row stride k) rotated with projection p0's input scales into a0 and, if a1, with p1's into a1: FP16 [rows, k], as
// grouped_exl3_prefill rotates them. exl3_prefill_rotated is grouped_exl3_prefill_bits for count 1 and max_tokens rows
// with A read from those rows: the same results bit for bit. rows 1..256; a 16-byte aligned.
cudaError_t exl3_prefill_rotate(const DeviceExpert* table, const GemmJob* job, Projection p0, Projection p1, int k,
                                int rows, half* a0, half* a1, cudaStream_t stream);
cudaError_t exl3_prefill_rotated(const DeviceExpert* table, const GemmJob* job, Projection p, int n, int k, int rows,
                                 int bits, const half* a, cudaStream_t stream);
cudaError_t exl3_dense_prefill(const DeviceExpert* table, GemmJob job, int n, int k, int bits, cudaStream_t stream,
                               int x_stride = 0, int out_stride = 0);
// s0ma change 43: exl3_dense's decode (no split, no groups) writing FP32 rows: job.output points to floats and
// out_stride counts floats. The values are those the BF16 launch rounds. Decode rows: 1..8 (1..4 for k = 8192).
cudaError_t exl3_dense_f32(const DeviceExpert* table, GemmJob job, int n, int k, int bits, cudaStream_t stream,
                           int x_stride = 0, int out_stride = 0);
cudaError_t grouped_exl3_var(const DeviceExpert* table, const GemmJob* jobs, int count,
                             Projection projection, int n, int k, const uint8_t* bits, cudaStream_t stream);
}
