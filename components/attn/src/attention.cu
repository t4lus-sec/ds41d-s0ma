#include <cstdlib>
#include "ds41/attention.h"
#include "attention_kernel.cuh"
extern "C" int ds41_attn_sparse(const void*q,const void*kv,const float*sink,const int*idx,void*out,int rows,int k,int heads,void*st){
 if(!q||!kv||!sink||!idx||!out||rows<1||k<1||k>640||(heads!=4&&heads!=8&&heads!=16))return cudaErrorInvalidValue;
 cudaError_t e=cudaFuncSetAttribute(ds41_sparse_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,86016);if(e!=cudaSuccess)return e;
 // s0ma change 75: row-major grid (a row's head groups adjacent) unless DS41_ATTN_ROWMAJOR=0
 static const int swap=[]{const char*v=getenv("DS41_ATTN_ROWMAJOR");return (v&&v[0]=='0')?0:1;}();
 const int y=swap&&rows<=65535;
 if(y)ds41_sparse_kernel<<<dim3(64/heads,rows),256,86016,(cudaStream_t)st>>>(sink,(const bfloat16_t*)kv,(bfloat16_t*)out,(const bfloat16_t*)q,idx,heads,rows,k,k,1);
 else ds41_sparse_kernel<<<dim3(rows,64/heads),256,86016,(cudaStream_t)st>>>(sink,(const bfloat16_t*)kv,(bfloat16_t*)out,(const bfloat16_t*)q,idx,heads,rows,k,k,0);
 return cudaGetLastError();
}
