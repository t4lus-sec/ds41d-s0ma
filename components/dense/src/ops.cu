#include <cstdlib>
#include <cstdint>
// s0ma change 32 (N4): a kernel launched behind a programmatic graph edge (DS41_PDL=1) starts before its
// predecessor has finished and waits here until that predecessor's writes are visible; otherwise a no-op.
#ifndef DS41_PDL_WAIT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define DS41_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#else
#define DS41_PDL_WAIT() do {} while (0)
#endif
#endif
#include "ds41/dense.h"
#include <cuda_bf16.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
namespace {
using bf16=__nv_bfloat16;
// Mirror PyTorch Reduce.cuh's contiguous FP32 mean: four independent accumulators,
// sequential combine, block halving, then descending warp shuffles. M controls its launch geometry.
__device__ __forceinline__ float blocksum(float sum,float* sh,int width){int tid=threadIdx.x;sh[tid]=sum;__syncthreads();
 for(int d=width/2;d>=32;d>>=1){if(tid%width<d){sum=__fadd_rn(sum,sh[tid+d]);sh[tid]=sum;}__syncthreads();}
 for(int d=16;d;d>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));if(tid%width==0)sh[tid]=sum;__syncthreads();
 for(int d=(blockDim.x/width)/2;d;d>>=1){if(tid%(width)==0&&tid/width<d){sum=__fadd_rn(sum,sh[tid+d*width]);sh[tid]=sum;}__syncthreads();}
 return sh[0];
}
int geometry(int M,int K,int& width){int height=1;while(height*2<=M&&height*2<=16)height*=2;int n=1;while(n*2<=K/4&&n*2<=512)n*=2;width=n<512/height?n:512/height;return K/width>=height*16?width*height:width;}
__global__ void rms(const bf16*x,const bf16*w,bf16*y,int M,int K,float eps,int width){DS41_PDL_WAIT();
 __shared__ float sums[512];int tid=threadIdx.x,row=blockIdx.x;if(row>=M)return;
 float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4){
  #pragma unroll
  for(int j=0;j<4;++j)if(i+j<K){float z=__bfloat162float(x[row*K+i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 }
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sums,width);
 if(!tid)sums[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));__syncthreads();
 for(int i=tid;i<K;i+=blockDim.x)y[row*K+i]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[row*K+i]),sums[0]),__bfloat162float(w[i])));
}
// Fused RMS -> explicit BF16 rounding -> optional FP8 quant or RoPE. The reduction
// geometry and every arithmetic boundary remain identical to the standalone ops.
__global__ void rmsrope(const bf16*x,const bf16*w,bf16*y,const float*f,int M,int K,int RD,float eps,int width,int inverse){DS41_PDL_WAIT();
 __shared__ float sums[512];int tid=threadIdx.x,row=blockIdx.x;float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4){
  #pragma unroll
  for(int j=0;j<4;++j)if(i+j<K){float z=__bfloat162float(x[row*K+i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 }
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sums,width);
 if(!tid)sums[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));__syncthreads();
 for(int k=tid*2;k<K;k+=blockDim.x*2){
  bf16 aa=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[row*K+k]),sums[0]),__bfloat162float(w[k])));
  bf16 bb=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[row*K+k+1]),sums[0]),__bfloat162float(w[k+1])));
  if(k>=K-RD){int pair=k-(K-RD);float a=__bfloat162float(aa),b=__bfloat162float(bb),c=f[row*RD+pair],si=f[row*RD+pair+1];if(inverse)si=-si;aa=__float2bfloat16_rn(__fmaf_rn(a,c,-__fmul_rn(b,si)));bb=__float2bfloat16_rn(__fmaf_rn(b,c,__fmul_rn(a,si)));}
  y[row*K+k]=aa;y[row*K+k+1]=bb;
 }
}
__global__ void hcpre(const bf16*h,const float*p,bf16*y,int M,int K){DS41_PDL_WAIT();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M*K)return;int row=i/K,k=i%K;float v[4];
 for(int j=0;j<4;++j)v[j]=__fmul_rn(p[row*4+j],__bfloat162float(h[(row*4+j)*K+k]));
 y[i]=__float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3]));
}
__global__ void hcpost(const bf16*y,const bf16*h,const float*p,const float*c,bf16*out,int M,int K){DS41_PDL_WAIT();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M*4*K)return;int row=i/(4*K),j=(i/K)%4,k=i%K;float v[4];
 for(int a=0;a<4;++a)v[a]=__fmul_rn(c[row*16+a*4+j],__bfloat162float(h[(row*4+a)*K+k]));
 float sum=__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3]);out[i]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(p[row*4+j],__bfloat162float(y[row*K+k])),sum));
}
__device__ float sum4(float v){v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,2));return __fadd_rn(v,__shfl_xor_sync(0xffffffff,v,1));}
__device__ float colsum4(float v){v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,8));return __fadd_rn(v,__shfl_xor_sync(0xffffffff,v,4));}
__global__ void sinkhorn(const float*m,const float*s,const float*b,float*pre,float*post,float*comb,int M,float eps,const float* rstd=nullptr){DS41_PDL_WAIT();
 int row=blockIdx.x,lane=threadIdx.x;if(row>=M)return;
 float z=lane<24?m[row*24+lane]:0.f;if(rstd)z=__fmul_rn(z,rstd[row]);
 if(lane<4){pre[row*4+lane]=__fadd_rn(1.f/(1.f+expf(-__fmaf_rn(z,s[0],b[lane]))),eps);float p=m[row*24+4+lane];if(rstd)p=__fmul_rn(p,rstd[row]);post[row*4+lane]=__fmul_rn(2.f,1.f/(1.f+expf(-__fmaf_rn(p,s[1],b[4+lane]))));}
 float q=lane<16?m[row*24+8+lane]:0.f;if(rstd)q=__fmul_rn(q,rstd[row]);float v=__fmaf_rn(q,s[2],b[8+(lane&15)]);
 float mx=fmaxf(v,__shfl_xor_sync(0xffffffff,v,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
 v=expf(v-mx);v=__fadd_rn(v/sum4(v),eps);v=v/__fadd_rn(colsum4(v),eps);
 for(int it=1;it<20;++it){v=v/__fadd_rn(sum4(v),eps);v=v/__fadd_rn(colsum4(v),eps);}
 if(lane<16)comb[row*16+lane]=v;
}
template<bool BF>
__global__ void f32linear(const float*x,const void*w,float*y,int M,int N,int K){DS41_PDL_WAIT();
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*4+warp;float acc[8]={0,0,0,0,0,0,0,0};
 if(row>=N)return;
 for(int k=lane;k<K;k+=32){float weight;if constexpr(BF)weight=__bfloat162float(((const bf16*)w)[(size_t)row*K+k]);else weight=((const float*)w)[(size_t)row*K+k];
  for(int m=0;m<M;++m)acc[m]=__fmaf_rn(weight,x[m*K+k],acc[m]);}
 for(int m=0;m<M;++m){for(int d=16;d;d>>=1)acc[m]=__fadd_rn(acc[m],__shfl_down_sync(0xffffffff,acc[m],d));if(!lane)y[m*N+row]=acc[m];}
}
__global__ void router(const float*logits,const float*bias,float*u,float*b,int count,int N){DS41_PDL_WAIT();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float z=logits[i],v=sqrtf(z>20.f?z:log1pf(expf(z)));u[i]=v;b[i]=__fadd_rn(v,bias[i%N]);}
__global__ void rope(bf16*x,const float*freq,int rows,int heads,int dim,int rd,int inverse){DS41_PDL_WAIT();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads*rd/2)return;int pair=i%(rd/2),head=(i/(rd/2))%heads,row=i/(heads*(rd/2));int off=(row*heads+head)*dim+dim-rd+pair*2;
 float a=__bfloat162float(x[off]),b=__bfloat162float(x[off+1]),c=freq[row*rd+pair*2],s=freq[row*rd+pair*2+1];if(inverse)s=-s;
 // PyTorch complex64 contracts the first product for real and second for imag.
 x[off]=__float2bfloat16_rn(__fmaf_rn(a,c,-__fmul_rn(b,s)));x[off+1]=__float2bfloat16_rn(__fmaf_rn(b,c,__fmul_rn(a,s)));
}
__global__ void engram(const bf16*h,const bf16*key,const bf16*value,const bf16*qw,const bf16*kw,const uint8_t*mask,bf16*out,int M,int K,float eps){DS41_PDL_WAIT();
 __shared__ float sh[3][512];int row=blockIdx.x,tid=threadIdx.x;float v[3][4]={{0,0,0,0},{0,0,0,0},{0,0,0,0}};
 for(int i=tid*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){int k=i+j;float a=__bfloat162float(h[row*K+k]),b=__bfloat162float(key[row*K+k]);float w=__fmul_rn(__bfloat162float(qw[(row%4)*K+k]),__bfloat162float(kw[(row%4)*K+k]));
  v[0][j]=__fadd_rn(v[0][j],__fmul_rn(a,a));v[1][j]=__fadd_rn(v[1][j],__fmul_rn(b,b));v[2][j]=__fadd_rn(v[2][j],__fmul_rn(__fmul_rn(a,w),b));}
 float sum[3];for(int r=0;r<3;++r){sum[r]=__fadd_rn(__fadd_rn(__fadd_rn(v[r][0],v[r][1]),v[r][2]),v[r][3]);sh[r][tid]=sum[r];}__syncthreads();
 for(int d=blockDim.x/2;d>=32;d>>=1){if(tid<d)for(int r=0;r<3;++r){sum[r]=__fadd_rn(sum[r],sh[r][tid+d]);sh[r][tid]=sum[r];}__syncthreads();}
 if(tid<32){for(int d=16;d;d>>=1)for(int r=0;r<3;++r)sum[r]=__fadd_rn(sum[r],__shfl_down_sync(0xffffffff,sum[r],d));
  if(!tid){float rstd=__fmul_rn(rsqrtf(__fadd_rn(__fmul_rn(sum[0],1.f/K),eps)),rsqrtf(__fadd_rn(__fmul_rn(sum[1],1.f/K),eps)));float dot=__fmul_rn(__fmul_rn(sum[2],rstd),1.f/sqrtf(float(K)));float gate=1.f/(1.f+expf(-copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot)));sh[0][0]=(mask&&!mask[row/4])?0.f:gate;}}
 __syncthreads();for(int k=tid;k<K;k+=blockDim.x)out[row*K+k]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(h[row*K+k]),__fmul_rn(sh[0][0],__bfloat162float(value[(row/4)*K+k]))));
}
__device__ __forceinline__ float mixvalue(const bf16*x,const float*pre,int row,int k,int K){if(!pre)return __bfloat162float(x[row*K+k]);float v[4];for(int j=0;j<4;++j)v[j]=__fmul_rn(pre[row*4+j],__bfloat162float(x[(row*4+j)*K+k]));return __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3])));}
__global__ void normquant(const bf16*x,const float*pre,const bf16*w,bf16*out,uint8_t*a,uint8_t*s,int M,int K,float eps,int width){DS41_PDL_WAIT();
 extern __shared__ bf16 normalized[];__shared__ float sh[512];int row=blockIdx.x,tid=threadIdx.x;float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){float z=mixvalue(x,pre,row,i+j,K);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sh,width);
 if(!tid)sh[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));__syncthreads();
 for(int i=tid;i<K;i+=blockDim.x){bf16 z=__float2bfloat16_rn(__fmul_rn(__fmul_rn(mixvalue(x,pre,row,i,K),sh[0]),__bfloat162float(w[i])));normalized[i]=z;if(out)out[row*K+i]=z;}__syncthreads();
 int lane=tid&31,warp=tid>>5;for(int g=warp;g<K/32;g+=blockDim.x/32){float z=__bfloat162float(normalized[g*32+lane]),mx=fabsf(z);for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));uint32_t bits=__float_as_uint(__fmul_rn(fmaxf(mx,1e-4f),1.f/448.f));unsigned exponent=(bits>>23)+((bits&0x7fffff)!=0);float inv=__uint_as_float((254-exponent)<<23);a[row*K+g*32+lane]=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,__fmul_rn(z,inv))),__NV_SATFINITE,__NV_E4M3);if(!lane)s[row*(K/32)+g]=exponent;}
}
__global__ void promote_stats(const bf16*x,float*y,float*rstd,int M,int K,float eps,int width){DS41_PDL_WAIT();__shared__ float sh[512];int row=blockIdx.x,tid=threadIdx.x;float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){float a=__bfloat162float(x[row*K+i+j]);y[row*K+i+j]=a;v[j]=__fadd_rn(v[j],__fmul_rn(a,a));}
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sh,width);if(!tid)rstd[row]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));
}
__global__ void promote(const bf16*x,float*y,int n){DS41_PDL_WAIT();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__bfloat162float(x[i]);}
__global__ void shared_act(const bf16*g,const bf16*u,bf16*z,uint8_t*a,uint8_t*s,int groups){DS41_PDL_WAIT();int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;if(group>=groups)return;int i=group*32+lane;
 float gate=fminf(__bfloat162float(g[i]),10.f),up=fminf(10.f,fmaxf(-10.f,__bfloat162float(u[i])));float value=__fmul_rn(gate/(1.f+expf(-gate)),up);bf16 zz=__float2bfloat16_rn(value);if(z)z[i]=zz;
 value=__bfloat162float(zz);float mx=fabsf(value);for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));uint32_t bits=__float_as_uint(__fmul_rn(fmaxf(mx,1e-4f),1.f/448.f));unsigned exponent=(bits>>23)+((bits&0x7fffff)!=0);float inv=__uint_as_float((254-exponent)<<23);a[i]=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,__fmul_rn(value,inv))),__NV_SATFINITE,__NV_E4M3);if(!lane)s[group]=exponent;
}
__global__ void routerweights(const float*s,const int32_t*ids,float*w,int M,int N,int topk,float scale){DS41_PDL_WAIT();int row=blockIdx.x,lane=threadIdx.x;float v[6]={0,0,0,0,0,0};if(lane>=topk)return;for(int i=0;i<topk;++i)v[i]=s[row*N+ids[row*topk+i]];float sum=topk==6?__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[4]),v[2]),__fadd_rn(__fadd_rn(v[1],v[5]),v[3])):__fadd_rn(__fadd_rn(v[0],v[2]),v[1]);w[row*topk+lane]=__fmul_rn(v[lane]/__fadd_rn(sum,1e-20f),scale);}
// s0ma change 128: a hyper-connection site's coefficients in one launch. Each of S CTAs takes 512 of the row's 4*K
// promoted inputs and, for every row, its share of the 24 mixes (fn . h) and of sum(h*h): each lane over its 16
// inputs in order, then a butterfly. The last CTA to arrive adds the S shares in CTA order, takes rstd from the sum
// and runs the Sinkhorn exactly as sinkhorn() does with those mixes and rstd. One row's arithmetic never depends on
// how many rows the launch has. Against ds41_dense_hc_mixes only the order of the 20,480-term sums changes (cuBLAS's
// GEMV and promote_stats' tree): a rounding-order change. No promoted copy and no host-side per-row calls.
constexpr int HCF_CH=512,HCF_OUT=25;
__global__ __launch_bounds__(256) void hc_fused_mixes(const bf16*h,const float*fn,const float*s,const float*b,float*pre,
    float*post,float*comb,float*partial,unsigned*count,int M,int K4,float neps,float heps){DS41_PDL_WAIT();
 __shared__ float hs[8][HCF_CH];
 __shared__ float mix[8][HCF_OUT];
 __shared__ int last;
 const int S=gridDim.x,c0=blockIdx.x*HCF_CH,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 for(int i=tid;i<M*HCF_CH;i+=blockDim.x){const int m=i/HCF_CH,k=i%HCF_CH;hs[m][k]=__bfloat162float(h[(size_t)m*K4+c0+k]);}
 __syncthreads();
 for(int o=warp;o<HCF_OUT;o+=8){
  float acc[8]={0,0,0,0,0,0,0,0};
  if(o<24){const float*f=fn+(size_t)o*K4+c0;
   for(int k=lane;k<HCF_CH;k+=32){const float wv=f[k];
    #pragma unroll
    for(int m=0;m<8;++m)if(m<M)acc[m]=__fmaf_rn(wv,hs[m][k],acc[m]);}
  }else{
   for(int k=lane;k<HCF_CH;k+=32){
    #pragma unroll
    for(int m=0;m<8;++m)if(m<M){const float a=hs[m][k];acc[m]=__fmaf_rn(a,a,acc[m]);}}
  }
  #pragma unroll
  for(int m=0;m<8;++m)if(m<M){float v=acc[m];for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,d));
   if(!lane)partial[((size_t)blockIdx.x*HCF_OUT+o)*8+m]=v;}
 }
 __threadfence();__syncthreads();
 if(!tid)last=atomicAdd(count,1u)==unsigned(S-1);
 __syncthreads();
 if(!last)return;
 __threadfence();
 // s0ma change 134: the shares loaded eight at a time, then added in CTA order as before (the same sum, bit for bit)
 for(int i=tid;i<M*HCF_OUT;i+=blockDim.x){const int m=i/HCF_OUT,o=i%HCF_OUT;float v=0.f;int q=0;
  for(;q+8<=S;q+=8){float w[8];
   #pragma unroll
   for(int j=0;j<8;++j)w[j]=__ldcg(partial+((size_t)(q+j)*HCF_OUT+o)*8+m);
   #pragma unroll
   for(int j=0;j<8;++j)v=__fadd_rn(v,w[j]);}
  for(;q<S;++q)v=__fadd_rn(v,__ldcg(partial+((size_t)q*HCF_OUT+o)*8+m));mix[m][o]=v;}
 __syncthreads();
 if(warp<M){const int row=warp;const float rs=rsqrtf(__fadd_rn(__fmul_rn(mix[row][24],1.f/K4),neps));
  // sinkhorn() line for line, its m[] the mixes above and its rstd rs
  float z=lane<24?mix[row][lane]:0.f;z=__fmul_rn(z,rs);
  if(lane<4){pre[row*4+lane]=__fadd_rn(1.f/(1.f+expf(-__fmaf_rn(z,s[0],b[lane]))),heps);float p=mix[row][4+lane];p=__fmul_rn(p,rs);post[row*4+lane]=__fmul_rn(2.f,1.f/(1.f+expf(-__fmaf_rn(p,s[1],b[4+lane]))));}
  float q=lane<16?mix[row][8+lane]:0.f;q=__fmul_rn(q,rs);float v=__fmaf_rn(q,s[2],b[8+(lane&15)]);
  float mx=fmaxf(v,__shfl_xor_sync(0xffffffff,v,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
  v=expf(v-mx);v=__fadd_rn(v/sum4(v),heps);v=v/__fadd_rn(colsum4(v),heps);
  for(int it=1;it<20;++it){v=v/__fadd_rn(sum4(v),heps);v=v/__fadd_rn(colsum4(v),heps);}
  if(lane<16)comb[row*16+lane]=v;
 }
 if(!tid)*count=0u;
}
// s0ma change 128: hcpre, then the decode RMSNorm (ds41e::hotpath::rms, eps 1e-20, its launch geometry), then optionally
// the FP32 copy (bf16_float): one block a row, the mixed row kept in shared memory. The same operations in the same
// order as the three launches, so the same bits.
__global__ void hc_pre_norm(const bf16*h,const float*p,const bf16*w,bf16*y,float*yf,int M,int K){DS41_PDL_WAIT();
 extern __shared__ bf16 mixed[];__shared__ float sh[512];
 const int r=blockIdx.x,t=threadIdx.x;if(r>=M)return;
 for(int k=t;k<K;k+=blockDim.x){float v[4];for(int j=0;j<4;++j)v[j]=__fmul_rn(p[r*4+j],__bfloat162float(h[((size_t)r*4+j)*K+k]));
  mixed[k]=__float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3]));}
 __syncthreads();
 float v[4]={0,0,0,0};
 for(int i=t*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){float z=__bfloat162float(mixed[i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 float sum=__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]);
 sh[t]=sum;__syncthreads();
 for(int d=blockDim.x/2;d>=32;d>>=1){if(t<d){sum=__fadd_rn(sum,sh[t+d]);sh[t]=sum;}__syncthreads();}
 if(t<32){for(int d=16;d;d>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));if(!t)sh[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),1e-20f));}
 __syncthreads();
 for(int i=t;i<K;i+=blockDim.x){const bf16 o=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(mixed[i]),sh[0]),__bfloat162float(w[i])));
  y[(size_t)r*K+i]=o;if(yf)yf[(size_t)r*K+i]=__bfloat162float(o);}
}
// s0ma change 139: the same two kernels for latency (DS41_HC_FAST=1), the same bits.
// hc_fused_mixes made each warp's 3-4 outputs one after another, the 16 weights a lane four loads at a time: 12-16 DRAM
// round trips in a row (~0.7 us each) for 2 MB a site. Here a CTA makes 5 outputs, one a warp (grid S x 5), and each
// lane loads its 16 weights at once, before the grid dependency wait (the weights are never written, so with DS41_PDL
// the loads overlap the kernel before). Each (chunk, output) share is the same sum of the same products in the same
// order, by the same lanes, then the same butterfly; the last CTA loads the S shares at once and adds them in CTA order.
constexpr int HCF2_OPC=5,HCF2_SMAX=64;
__global__ __launch_bounds__(32*HCF2_OPC) void hc_fused_mixes2(const bf16*h,const float*fn,const float*s,const float*b,
    float*pre,float*post,float*comb,float*partial,unsigned*count,int M,int K4,float neps,float heps){
 __shared__ float hs[8][HCF_CH];
 __shared__ float mix[8][HCF_OUT];
 __shared__ int last;
 const int S=gridDim.x,c0=blockIdx.x*HCF_CH,tid=threadIdx.x,lane=tid&31,warp=tid>>5,o=blockIdx.y*HCF2_OPC+warp;
 float wv[16];
 if(o<24){const float*f=fn+(size_t)o*K4+c0;
  #pragma unroll
  for(int i=0;i<16;++i)wv[i]=__ldg(f+lane+32*i);}
 DS41_PDL_WAIT();
 for(int i=tid;i<M*(HCF_CH/8);i+=blockDim.x){const int m=i/(HCF_CH/8),k=(i%(HCF_CH/8))*8;
  const uint4 u=*reinterpret_cast<const uint4*>(h+(size_t)m*K4+c0+k);const bf16*e=reinterpret_cast<const bf16*>(&u);
  #pragma unroll
  for(int j=0;j<8;++j)hs[m][k+j]=__bfloat162float(e[j]);}
 __syncthreads();
 if(o<HCF_OUT){
  float acc[8]={0,0,0,0,0,0,0,0};
  if(o<24){
   #pragma unroll
   for(int i=0;i<16;++i){const int k=lane+32*i;
    #pragma unroll
    for(int m=0;m<8;++m)if(m<M)acc[m]=__fmaf_rn(wv[i],hs[m][k],acc[m]);}
  }else{
   #pragma unroll
   for(int i=0;i<16;++i){const int k=lane+32*i;
    #pragma unroll
    for(int m=0;m<8;++m)if(m<M){const float a=hs[m][k];acc[m]=__fmaf_rn(a,a,acc[m]);}}
  }
  #pragma unroll
  for(int m=0;m<8;++m)if(m<M){float v=acc[m];for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,d));
   if(!lane)partial[((size_t)blockIdx.x*HCF_OUT+o)*8+m]=v;}
 }
 __threadfence();__syncthreads();
 if(!tid)last=atomicAdd(count,1u)==unsigned(S*gridDim.y-1);
 __syncthreads();
 if(!last)return;
 __threadfence();
 for(int i=tid;i<M*HCF_OUT;i+=blockDim.x){const int m=i/HCF_OUT,o2=i%HCF_OUT;float v=0.f;
  if(S<=HCF2_SMAX){float w[HCF2_SMAX];
   #pragma unroll
   for(int j=0;j<HCF2_SMAX;++j)if(j<S)w[j]=__ldcg(partial+((size_t)j*HCF_OUT+o2)*8+m);
   #pragma unroll
   for(int j=0;j<HCF2_SMAX;++j)if(j<S)v=__fadd_rn(v,w[j]);
  }else for(int q=0;q<S;++q)v=__fadd_rn(v,__ldcg(partial+((size_t)q*HCF_OUT+o2)*8+m));
  mix[m][o2]=v;}
 __syncthreads();
 for(int row=warp;row<M;row+=HCF2_OPC){const float rs=rsqrtf(__fadd_rn(__fmul_rn(mix[row][24],1.f/K4),neps));
  // sinkhorn() line for line, as in hc_fused_mixes
  float z=lane<24?mix[row][lane]:0.f;z=__fmul_rn(z,rs);
  if(lane<4){pre[row*4+lane]=__fadd_rn(1.f/(1.f+expf(-__fmaf_rn(z,s[0],b[lane]))),heps);float p=mix[row][4+lane];p=__fmul_rn(p,rs);post[row*4+lane]=__fmul_rn(2.f,1.f/(1.f+expf(-__fmaf_rn(p,s[1],b[4+lane]))));}
  float q=lane<16?mix[row][8+lane]:0.f;q=__fmul_rn(q,rs);float v=__fmaf_rn(q,s[2],b[8+(lane&15)]);
  float mx=fmaxf(v,__shfl_xor_sync(0xffffffff,v,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
  v=expf(v-mx);v=__fadd_rn(v/sum4(v),heps);v=v/__fadd_rn(colsum4(v),heps);
  for(int it=1;it<20;++it){v=v/__fadd_rn(sum4(v),heps);v=v/__fadd_rn(colsum4(v),heps);}
  if(lane<16)comb[row*16+lane]=v;
 }
 if(!tid)*count=0u;
}
// s0ma change 139: hc_pre_norm with its loads issued together: the norm weight (never written) read before the grid
// dependency wait, the four streams eight columns at a time. The same operations on the same values in the same order.
constexpr int HCPN_PRE=16;
__global__ void hc_pre_norm2(const bf16*h,const float*p,const bf16*w,bf16*y,float*yf,int M,int K){
 extern __shared__ bf16 mixed[];__shared__ float sh[512];
 const int r=blockIdx.x,t=threadIdx.x,B=blockDim.x,nI=(K+B-1)/B;
 float wr[HCPN_PRE];
 if(nI<=HCPN_PRE){
  #pragma unroll
  for(int u=0;u<HCPN_PRE;++u){const int i=t+u*B;wr[u]=u<nI&&i<K?__bfloat162float(w[i]):0.f;}}
 DS41_PDL_WAIT();
 if(r>=M)return;
 float pr[4];
 #pragma unroll
 for(int j=0;j<4;++j)pr[j]=p[r*4+j];
 for(int k0=t;k0<K;k0+=B*8){float hv[8][4];
  #pragma unroll
  for(int u=0;u<8;++u){const int k=k0+u*B;
   #pragma unroll
   for(int j=0;j<4;++j)hv[u][j]=k<K?__bfloat162float(h[((size_t)r*4+j)*K+k]):0.f;}
  #pragma unroll
  for(int u=0;u<8;++u){const int k=k0+u*B;if(k<K){float v[4];
   #pragma unroll
   for(int j=0;j<4;++j)v[j]=__fmul_rn(pr[j],hv[u][j]);
   mixed[k]=__float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3]));}}
 }
 __syncthreads();
 float v[4]={0,0,0,0};
 for(int i=t*4;i<K;i+=B*4)for(int j=0;j<4;++j)if(i+j<K){float z=__bfloat162float(mixed[i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 float sum=__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]);
 sh[t]=sum;__syncthreads();
 for(int d=B/2;d>=32;d>>=1){if(t<d){sum=__fadd_rn(sum,sh[t+d]);sh[t]=sum;}__syncthreads();}
 if(t<32){for(int d=16;d;d>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));if(!t)sh[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),1e-20f));}
 __syncthreads();
 if(nI<=HCPN_PRE){
  #pragma unroll
  for(int u=0;u<HCPN_PRE;++u){const int i=t+u*B;if(u<nI&&i<K){
   const bf16 o=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(mixed[i]),sh[0]),wr[u]));
   y[(size_t)r*K+i]=o;if(yf)yf[(size_t)r*K+i]=__bfloat162float(o);}}
 }else for(int i=t;i<K;i+=B){const bf16 o=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(mixed[i]),sh[0]),__bfloat162float(w[i])));
  y[(size_t)r*K+i]=o;if(yf)yf[(size_t)r*K+i]=__bfloat162float(o);}
}
__global__ void hcmean(const bf16*h,bf16*y,int M,int K){DS41_PDL_WAIT();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M*K)return;int row=i/K,k=i%K;float sum=0.f;for(int j=0;j<4;++j)sum=__fadd_rn(sum,__bfloat162float(h[(row*4+j)*K+k]));y[i]=__float2bfloat16_rn(__fmul_rn(sum,.25f));}
}
// s0ma change 39: cublasSetStream resets the handle's workspace to cuBLAS's default pool, which allocates with
// cudaMallocAsync inside a stream capture (memory allocation nodes in the graph). Set the stream only when it changes,
// so a workspace the engine gave the handle (DS41_PDL) stays in use; with none set, nothing changes.
static inline cublasStatus_t ds41_same_stream(cublasHandle_t h,cudaStream_t st){
    cudaStream_t cur=nullptr;
    if(cublasGetStream(h,&cur)==CUBLAS_STATUS_SUCCESS&&cur==st)return CUBLAS_STATUS_SUCCESS;
    return cublasSetStream(h,st);
}
extern "C" int ds41_dense_rms(const void*x,const void*w,void*y,int M,int K,float eps,cudaStream_t st){if(!x||!w||!y||M<1||M>8||K<128||K%4)return cudaErrorInvalidValue;
 int width,threads=geometry(M,K,width);rms<<<M,threads,0,st>>>((const bf16*)x,(const bf16*)w,(bf16*)y,M,K,eps,width);return cudaGetLastError();}
extern "C" int ds41_dense_hc_pre(const void*h,const float*p,void*y,int M,int K,cudaStream_t st){if(!h||!p||!y||M<1||K<1)return cudaErrorInvalidValue;hcpre<<<(M*K+255)/256,256,0,st>>>((const bf16*)h,p,(bf16*)y,M,K);return cudaGetLastError();}
extern "C" int ds41_dense_hc_post(const void*y,const void*h,const float*p,const float*c,void*out,int M,int K,cudaStream_t st){if(!y||!h||!p||!c||!out||M<1||K<1)return cudaErrorInvalidValue;hcpost<<<(M*4*K+255)/256,256,0,st>>>((const bf16*)y,(const bf16*)h,p,c,(bf16*)out,M,K);return cudaGetLastError();}
extern "C" int ds41_dense_hc_sinkhorn(const float*m,const float*s,const float*b,float*p,float*post,float*c,int M,float eps,cudaStream_t st){if(!m||!s||!b||!p||!post||!c||M<1)return cudaErrorInvalidValue;sinkhorn<<<M,32,0,st>>>(m,s,b,p,post,c,M,eps);return cudaGetLastError();}
extern "C" int ds41_dense_f32_bf16(const float*x,const void*w,float*y,int M,int N,int K,cudaStream_t st){if(!x||!w||!y||M<1||M>8||N<1||K<1)return cudaErrorInvalidValue;f32linear<true><<<(N+3)/4,128,0,st>>>(x,w,y,M,N,K);return cudaGetLastError();}
extern "C" int ds41_dense_f32(const float*x,const float*w,float*y,int M,int N,int K,cudaStream_t st){if(!x||!w||!y||M<1||M>8||N<1||K<1)return cudaErrorInvalidValue;f32linear<false><<<(N+3)/4,128,0,st>>>(x,w,y,M,N,K);return cudaGetLastError();}
extern "C" int ds41_dense_router_scores(const float*l,const float*bias,float*u,float*b,int M,int N,cudaStream_t st){if(!l||!bias||!u||!b||M<1||N<1)return cudaErrorInvalidValue;router<<<(M*N+255)/256,256,0,st>>>(l,bias,u,b,M*N,N);return cudaGetLastError();}
extern "C" int ds41_dense_rope(void*x,const float*f,int R,int H,int D,int RD,int inv,cudaStream_t st){if(!x||!f||R<1||H<1||RD<2||RD%2||D<RD)return cudaErrorInvalidValue;rope<<<(R*H*RD/2+255)/256,256,0,st>>>((bf16*)x,f,R,H,D,RD,inv);return cudaGetLastError();}
extern "C" int ds41_dense_engram_gate(const void*h,const void*k,const void*v,const void*qw,const void*kw,const uint8_t*mask,void*out,int M,int K,float eps,cudaStream_t st){if(!h||!k||!v||!qw||!kw||!out||M<1||M>8||K<128||K%4)return cudaErrorInvalidValue;int height=1;while(height*2<=M*4&&height*2<=16)height*=2;int width=512/height;engram<<<M*4,width,0,st>>>((const bf16*)h,(const bf16*)k,(const bf16*)v,(const bf16*)qw,(const bf16*)kw,mask,(bf16*)out,M,K,eps);return cudaGetLastError();}
extern "C" int ds41_dense_cublas(void* raw,const void*x,const void*w,void*y,int M,int N,int K,int f32,int groups,cudaStream_t st){if(!raw||!x||!w||!y||M<1||N<1||K<1||groups<1)return cudaErrorInvalidValue;
 auto h=(cublasHandle_t)raw;cublasStatus_t status=ds41_same_stream(h,st);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
 // Set the math mode explicitly: TF32 is never enabled for these reference exceptions.
 status=cublasSetMathMode(h,CUBLAS_DEFAULT_MATH);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
 float alpha=1.f,beta=0.f;auto dt=f32?CUDA_R_32F:CUDA_R_16BF;
 if(groups==1)status=cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,dt,K,x,dt,K,&beta,y,dt,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
 else status=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,dt,K,(long long)N*K,x,dt,K,(long long)M*K,&beta,y,dt,N,(long long)M*N,groups,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
 return status==CUBLAS_STATUS_SUCCESS?0:10000+status;
}
// s0ma leg 26: `rows` M=1 problems in one launch, each computed as ds41_dense_cublas(M=1) computes it (the same
// weight for every row: stride 0). Exact only if cuBLAS's batched kernel reduces each row as its M=1 call does: the
// engine uses it under DS41_ROWS_BATCHED=1, and the rows check and the benchmark's token comparison decide.
extern "C" int ds41_dense_cublas_rows(void* raw,const void*x,const void*w,void*y,int rows,int N,int K,int f32,cudaStream_t st){
 if(rows==1)return ds41_dense_cublas(raw,x,w,y,1,N,K,f32,1,st);
 if(!raw||!x||!w||!y||rows<1||N<1||K<1)return cudaErrorInvalidValue;
 auto h=(cublasHandle_t)raw;cublasStatus_t status=ds41_same_stream(h,st);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
 status=cublasSetMathMode(h,CUBLAS_DEFAULT_MATH);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
 float alpha=1.f,beta=0.f;auto dt=f32?CUDA_R_32F:CUDA_R_16BF;
 status=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,1,K,&alpha,w,dt,K,0,x,dt,K,(long long)K,&beta,y,dt,N,(long long)N,rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
 return status==CUBLAS_STATUS_SUCCESS?0:10000+status;
}
// ds41_dense_hc_mixes for M rows with every row's math as at M=1: one block a row in the M=1 geometry for the
// promotion and the Sinkhorn (exact by construction); the mixes GEMV as one M=1 call a row, or batched when
// batched_gemv says the batched call was found bit-identical for this shape and row count (the engine checks at load).
extern "C" int ds41_dense_hc_mixes_rows(void* handle,const void*h,const float*fn,const float*s,const float*b,float*pre,float*post,float*comb,float*x,float*mixes,float*rstd,int M,int K,float neps,float heps,int batched_gemv,cudaStream_t st){
 if(!handle||!h||!fn||!s||!b||!pre||!post||!comb||!x||!mixes||!rstd||M<1||M>8||K<32||K%32)return cudaErrorInvalidValue;
 int width,threads=geometry(1,K*4,width);promote_stats<<<M,threads,0,st>>>((const bf16*)h,x,rstd,M,K*4,neps,width);int rc=cudaGetLastError();if(rc)return rc;
 if(batched_gemv)rc=ds41_dense_cublas_rows(handle,x,fn,mixes,M,24,K*4,1,st);
 else for(int r=0;r<M&&!rc;++r)rc=ds41_dense_cublas(handle,x+(size_t)r*K*4,fn,mixes+r*24,1,24,K*4,1,1,st);
 if(rc)return rc;
 sinkhorn<<<M,32,0,st>>>(mixes,s,b,pre,post,comb,M,heps,rstd);return cudaGetLastError();
}
extern "C" int ds41_dense_norm_quant(const void*x,const float*pre,const void*w,void*out,void*a,void*s,int M,int K,float eps,cudaStream_t st){if(!x||!w||!a||!s||M<1||M>8||K<128||K%32||K>20480)return cudaErrorInvalidValue;int width,threads=geometry(M,K,width);
 normquant<<<M,threads,K*sizeof(bf16),st>>>((const bf16*)x,pre,(const bf16*)w,(bf16*)out,(uint8_t*)a,(uint8_t*)s,M,K,eps,width);return cudaGetLastError();}
extern "C" int ds41_dense_hc_mixes(void* handle,const void*h,const float*fn,const float*s,const float*b,float*pre,float*post,float*comb,float*x,float*mixes,float*rstd,int M,int K,float neps,float heps,cudaStream_t st){if(!handle||!h||!fn||!s||!b||!pre||!post||!comb||!x||!mixes||!rstd||M<1||M>8||K<32||K%32)return cudaErrorInvalidValue;int width,threads=geometry(M,K*4,width);promote_stats<<<M,threads,0,st>>>((const bf16*)h,x,rstd,M,K*4,neps,width);int rc=cudaGetLastError();if(rc)return rc;rc=ds41_dense_cublas(handle,x,fn,mixes,M,24,K*4,1,1,st);if(rc)return rc;sinkhorn<<<M,32,0,st>>>(mixes,s,b,pre,post,comb,M,heps,rstd);return cudaGetLastError();}
// s0ma change 128: see hc_fused_mixes and hc_pre_norm above
extern "C" size_t ds41_dense_hc_fused_floats(int K){return K>0&&(4*K)%HCF_CH==0?size_t(4*K/HCF_CH)*HCF_OUT*8:0;}
// s0ma change 139: variant 0 the change-128 kernels, 1 the change-139 ones (the same bits); the plain entries take
// DS41_HC_FAST (1: variant 1)
static int hc_fast(){static const int v=[]{const char*e=std::getenv("DS41_HC_FAST");return e&&e[0]=='1'?1:0;}();return v;}
extern "C" int ds41_dense_hc_fused_mixes_v(const void*h,const float*fn,const float*s,const float*b,float*pre,float*post,
    float*comb,float*partial,unsigned*count,int M,int K,float neps,float heps,int variant,cudaStream_t st){
 if(!h||!fn||!s||!b||!pre||!post||!comb||!partial||!count||M<1||M>8||K<1||(4*K)%HCF_CH)return cudaErrorInvalidValue;
 if(variant==1&&!(reinterpret_cast<uintptr_t>(h)&15)&&4*K/HCF_CH<=65535)
  hc_fused_mixes2<<<dim3(4*K/HCF_CH,(HCF_OUT+HCF2_OPC-1)/HCF2_OPC),32*HCF2_OPC,0,st>>>((const bf16*)h,fn,s,b,pre,post,comb,partial,count,M,4*K,neps,heps);
 else hc_fused_mixes<<<4*K/HCF_CH,256,0,st>>>((const bf16*)h,fn,s,b,pre,post,comb,partial,count,M,4*K,neps,heps);
 return cudaGetLastError();}
extern "C" int ds41_dense_hc_fused_mixes(const void*h,const float*fn,const float*s,const float*b,float*pre,float*post,
    float*comb,float*partial,unsigned*count,int M,int K,float neps,float heps,cudaStream_t st){
 return ds41_dense_hc_fused_mixes_v(h,fn,s,b,pre,post,comb,partial,count,M,K,neps,heps,hc_fast(),st);}
extern "C" int ds41_dense_hc_pre_norm_v(const void*h,const float*p,const void*w,void*y,float*yf,int M,int K,int variant,cudaStream_t st){
 if(!h||!p||!w||!y||M<1||K<128||K%4||K>16384)return cudaErrorInvalidValue;
 int threads=1;while(threads*2<=K/4&&threads*2<=512)threads*=2;   // as ds41e::hotpath::rms_decode
 if(variant==1)hc_pre_norm2<<<M,threads,K*sizeof(bf16),st>>>((const bf16*)h,p,(const bf16*)w,(bf16*)y,yf,M,K);
 else hc_pre_norm<<<M,threads,K*sizeof(bf16),st>>>((const bf16*)h,p,(const bf16*)w,(bf16*)y,yf,M,K);
 return cudaGetLastError();}
extern "C" int ds41_dense_hc_pre_norm(const void*h,const float*p,const void*w,void*y,float*yf,int M,int K,cudaStream_t st){
 return ds41_dense_hc_pre_norm_v(h,p,w,y,yf,M,K,hc_fast(),st);}
extern "C" int ds41_dense_shared_act(const void*g,const void*u,void*z,void*a,void*s,int M,int K,cudaStream_t st){if(!g||!u||!a||!s||M<1||K<32||K%32)return cudaErrorInvalidValue;shared_act<<<(M*K+127)/128,128,0,st>>>((const bf16*)g,(const bf16*)u,(bf16*)z,(uint8_t*)a,(uint8_t*)s,M*K/32);return cudaGetLastError();}
extern "C" int ds41_dense_promote(const void*x,float*y,int n,cudaStream_t st){if(!x||!y||n<1)return cudaErrorInvalidValue;promote<<<(n+255)/256,256,0,st>>>((const bf16*)x,y,n);return cudaGetLastError();}
extern "C" int ds41_dense_wo_a(void*raw,const void*x,const void*w,void*y,int M,int N,int K,int G,cudaStream_t st){if(!raw||!x||!w||!y||M<1||M>8||N<1||K<1||G<1)return cudaErrorInvalidValue;auto h=(cublasHandle_t)raw;auto status=ds41_same_stream(h,st);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;status=cublasSetMathMode(h,CUBLAS_DEFAULT_MATH);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;float alpha=1.f,beta=0.f;
 status=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,CUDA_R_16BF,K,(long long)N*K,x,CUDA_R_16BF,G*K,K,&beta,y,CUDA_R_16BF,G*N,N,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);return status==CUBLAS_STATUS_SUCCESS?0:10000+status;}
extern "C" int ds41_dense_router_weights(const float*s,const int32_t*ids,float*w,int M,int N,int topk,float scale,cudaStream_t st){if(!s||!ids||!w||M<1||N<topk||(topk!=3&&topk!=6))return cudaErrorInvalidValue;routerweights<<<M,32,0,st>>>(s,ids,w,M,N,topk,scale);return cudaGetLastError();}
extern "C" int ds41_dense_hc_mean(const void*h,void*y,int M,int K,cudaStream_t st){if(!h||!y||M<1||K<1)return cudaErrorInvalidValue;hcmean<<<(M*K+255)/256,256,0,st>>>((const bf16*)h,(bf16*)y,M,K);return cudaGetLastError();}

extern "C" int ds41_dense_rms_rope(const void*x,const void*w,void*y,const float*f,int M,int K,int RD,float eps,int inverse,cudaStream_t st){
 if(!x||!w||!y||!f||M<1||M>8||K<128||K%4||RD<2||RD%2||RD>K)return cudaErrorInvalidValue;
 int width,threads=geometry(M,K,width);rmsrope<<<M,threads,0,st>>>((const bf16*)x,(const bf16*)w,(bf16*)y,f,M,K,RD,eps,width,inverse);return cudaGetLastError();
}
