#pragma once
#include <cuda_runtime.h>
namespace ds41e::prefill {
void select_routes(const float*,const float*,int*,float*,cudaStream_t);
void gather_routes(const unsigned char*,const unsigned char*,const float*,const int*,
                   unsigned char*,unsigned char*,float*,int,cudaStream_t);
// EXL3 experts: BF16 rows (x [rows,5120] BF16, row = slot/6), no FP8 quantization.
void gather_routes_bf16(const void*,const float*,const int*,void*,float*,int,cudaStream_t);
void scatter_routes(const void*,const int*,void*,int,cudaStream_t);
// Batched prefill (--prefill-chunk): row-count-unbounded variants with the M=1 reduction geometry.
int rms_rows(const void* x,const void* w,void* y,int M,int K,float eps,cudaStream_t);
int hc_mixes_rows(void* blas,const void* h,const float* fn,const float* scale,const float* base,float* pre,float* post,
    float* comb,float* promoted,float* mixes,float* rstd,int M,int K,float neps,float heps,cudaStream_t);
void select_routes_rows(const float* biased,const float* unbiased,int* ids,float* weights,int rows,cudaStream_t);
// s0ma change 94: ds41_dense_engram_gate for count (<= 8) prompt rows of h ([rows][4][K]) in one launch, in place;
// kv holds each row's projection ([count][5][K]: 4 keys, the value), as the per-row calls' bit for bit
int engram_gate_rows(void* h,const int* rows,int count,const void* kv,const void* qw,const void* kw,int K,float eps,
    cudaStream_t);
int wo_a_rows(void* blas,const void* x,const void* w,void* y,int M,int N,int K,int G,cudaStream_t);
}
