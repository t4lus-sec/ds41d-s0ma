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
cudaError_t exl3_dense(const DeviceExpert* table, GemmJob job, int n, int k, int bits, cudaStream_t stream,
                       int x_stride = 0, int out_stride = 0, int groups = 1, int group_x = 0, int group_out = 0);
cudaError_t exl3_dense_prefill(const DeviceExpert* table, GemmJob job, int n, int k, int bits, cudaStream_t stream,
                               int x_stride = 0, int out_stride = 0);
cudaError_t grouped_exl3_var(const DeviceExpert* table, const GemmJob* jobs, int count,
                             Projection projection, int n, int k, const uint8_t* bits, cudaStream_t stream);
}
