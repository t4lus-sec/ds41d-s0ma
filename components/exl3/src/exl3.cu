// s0ma change 32 (N4): a kernel launched behind a programmatic graph edge (DS41_PDL=1) starts before its
// predecessor has finished and waits here until that predecessor's writes are visible; otherwise a no-op.
#ifndef DS41_PDL_WAIT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define DS41_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#else
#define DS41_PDL_WAIT() do {} while (0)
#endif
#endif
#include <ds41/exl3.cuh>
#include "device.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
namespace ds41 {
namespace exl3_detail {
// Fixed 8-row tile, 128 columns, four contiguous K partitions. M never changes
// geometry, K order, rounding, or reduction order. No inter-CTA communication.
// s0ma change 34 (N1): ROWS is the most rows a job may hold (8; 4 for k=8192, whose 8-row input would not fit the
// 99 KB of shared memory a block may have). Rows past a job's count are never read, so ROWS changes nothing else.
template<int K,int BITS,int ROWS=8>
// s0ma change 34: xs and os are the input's and output's row strides in elements (K and N unless a slice of wider
// rows, as 0xSero's wo_a groups: 4096 of each 32,768-wide row in, 1024 of each 8192-wide row out)
__device__ __forceinline__ void decode_body(const DeviceExpert* table,const GemmJob job,int projection,int N,int xs=K,int os=-1) {
    if(os<0)os=N;
    if(job.tokens<1||job.tokens>ROWS){asm volatile("trap;");return;}
    DeviceExpert e=table[job.expert];wait_ready(e);
    const uint32_t* B=weights(e,projection);
    const half* su=scales(e,projection); const half* sv=su+K;
    extern __shared__ __align__(16) unsigned char storage[];
    half* A=reinterpret_cast<half*>(storage);
    float* red=reinterpret_cast<float*>(A+(ROWS*K<2048?2048:ROWS*K));
    int warp=threadIdx.x/32,lane=threadIdx.x%32,wn=warp%4,wk=warp/4;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x);
    for(int idx=warp;idx<job.tokens*(K/128);idx+=16) {
        int r=idx/(K/128),c=(idx%(K/128))*128;
        rotate_input(x+size_t(r)*xs+c,su+c,A+r*K+c,lane);
    }
    __syncthreads();
    constexpr int SLICES=K/16,CHUNK=(SLICES+3)/4,PF=4,WNT=2;
    const int start=wk*CHUNK, len=min(CHUNK,SLICES-start);
    const int r=lane/4; bool active=r<job.tokens;
    const uint32_t* A2=reinterpret_cast<const uint32_t*>(A);
    TileW<BITS> pf[PF][WNT];
    auto load=[&](int i,int t){
        size_t off=(size_t(start+i)*(N/16)+blockIdx.x*8+wn*2+t)*(8*BITS);
        return load_tile<BITS>(B+off,lane);
    };
    #pragma unroll
    for(int d=0;d<PF;++d)if(d<len){
        #pragma unroll
        for(int t=0;t<WNT;++t)pf[d][t]=load(d,t);
    }
    float acc[WNT][2][4]={};
    for(int base=0;base<len;base+=PF){
        #pragma unroll
        for(int d=0;d<PF;++d){
            int i=base+d;if(i>=len)break;
            TileW<BITS> bw[WNT];
            #pragma unroll
            for(int t=0;t<WNT;++t){bw[t]=pf[d][t];if(i+PF<len)pf[d][t]=load(i+PF,t);}
            int col=(start+i)*8+(lane&3);
            uint32_t a[4]={active?A2[r*(K/2)+col]:0u,0u,active?A2[r*(K/2)+col+4]:0u,0u};
            #pragma unroll
            for(int t=0;t<WNT;++t){Pair f0,f1;dqk<BITS>(bw[t],f0,f1,lane);mma(acc[t][0],a,f0);mma(acc[t][1],a,f1);}
        }
    }
    if(active){
        #pragma unroll
        for(int t=0;t<WNT;++t){
            #pragma unroll
            for(int f=0;f<2;++f){
                int c=wn*32+t*16+f*8+2*(lane&3);
                red[(wk*8+r)*128+c]=acc[t][f][0];red[(wk*8+r)*128+c+1]=acc[t][f][1];
            }
        }
    }
    __syncthreads();
    // Reuse the input buffer for the reduced tile. Each element has one writer.
    float* out=reinterpret_cast<float*>(A);
    for(int i=threadIdx.x;i<job.tokens*128;i+=512){
        float sum=0;
        #pragma unroll
        for(int w=0;w<4;++w)sum+=red[w*8*128+i];
        out[i]=sum;
    }
    __syncthreads();
    if(warp<job.tokens)rotate_output(out+warp*128,sv+blockIdx.x*128,job.output+size_t(warp)*os+blockIdx.x*128,lane);
}
template<int K,int BITS,int ROWS=8>
__global__ __launch_bounds__(512,1) void decode(const DeviceExpert* table,const GemmJob* jobs,int projection,int N) {DS41_PDL_WAIT();
    decode_body<K,BITS,ROWS>(table,jobs[blockIdx.y],projection,N);
}
// s0ma change 34 (N1): one job passed by value (a dense tensor: no device job array, and a captured graph keeps it)
template<int K,int BITS,int ROWS=8>
// blockIdx.y = g of a set of tensors at consecutive table entries, reading x + g*gx and writing output + g*go
__global__ __launch_bounds__(512,1) void decode_one(const DeviceExpert* table,const GemmJob job,int N,int xs,int os,int gx,int go) {DS41_PDL_WAIT();
    GemmJob mine=job;
    mine.expert=job.expert+blockIdx.y;
    mine.x=job.x+size_t(blockIdx.y)*gx*2;
    mine.output=job.output+size_t(blockIdx.y)*go;
    decode_body<K,BITS,ROWS>(table,mine,0,N,xs,os);
}
// s0ma change 15: one launch for jobs of several widths. Each CTA reads its job's expert's width from `bits` and
// runs that width's decode, so the work per CTA is exactly decode<K,BITS>'s (block-uniform branch, same shared memory).
template<int K>
__global__ __launch_bounds__(512,1) void decode_var(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,const uint8_t* bits) {DS41_PDL_WAIT();
    switch(bits[jobs[blockIdx.y].expert]){
        case 2: decode_body<K,2>(table,jobs[blockIdx.y],projection,N); break;
        case 3: decode_body<K,3>(table,jobs[blockIdx.y],projection,N); break;
        case 4: decode_body<K,4>(table,jobs[blockIdx.y],projection,N); break;
        case 5: decode_body<K,5>(table,jobs[blockIdx.y],projection,N); break;
        default: asm volatile("trap;");
    }
}
// s0ma: the same tile computed by a cluster of four CTAs, one per K partition (blockIdx.z), four warps each. Every
// warp does exactly what warp (wn, wk) of decode() does: the same rotated inputs (each CTA rotates only the 128-wide
// blocks its partition reads), the same loads, dequantization and MMAs in the same order. The partials meet in CTA 0's
// shared memory through the cluster and are summed as decode() sums them: 0 + p0 + p1 + p2 + p3. Bit-identical, with
// 4x the CTAs and a fifth of the shared memory, so the small decode grids fill the card.
template<int K>
struct SplitGeom {
    static constexpr int SLICES=K/16, CHUNK=(SLICES+3)/4;
    static constexpr int W=((CHUNK*16+127)/128+1)*128;   // the widest run of 128-blocks a partition touches
};
template<int K,int BITS>
__global__ void __cluster_dims__(1,1,4) __launch_bounds__(128) decode_split(const DeviceExpert* table,const GemmJob* jobs,int projection,int N) {DS41_PDL_WAIT();
    namespace cg=cooperative_groups;
    cg::cluster_group cluster=cg::this_cluster();
    using G=SplitGeom<K>;
    GemmJob job=jobs[blockIdx.y];
    if(job.tokens<1||job.tokens>8){asm volatile("trap;");return;}
    DeviceExpert e=table[job.expert];wait_ready(e);
    const uint32_t* B=weights(e,projection);
    const half* su=scales(e,projection); const half* sv=su+K;
    __shared__ __align__(16) half A[8*G::W];
    __shared__ __align__(16) float red[8*128];
    const int warp=threadIdx.x/32,lane=threadIdx.x%32,wn=warp,wk=blockIdx.z;
    constexpr int SLICES=G::SLICES,CHUNK=G::CHUNK,PF=16,WNT=2;   // s0ma change 6: 16 loads in flight a warp (was 4)
    const int start=wk*CHUNK, len=max(0,min(CHUNK,SLICES-start));
    const int lo=(start*16)/128, hi=len?((start+len)*16+127)/128:lo, nb=hi-lo;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x);
    for(int idx=warp;idx<job.tokens*nb;idx+=4) {
        int r=idx/nb,b=lo+idx%nb;
        rotate_input(x+r*K+b*128,su+b*128,A+r*G::W+(b-lo)*128,lane);
    }
    __syncthreads();
    const int r=lane/4; bool active=r<job.tokens;
    const uint32_t* A2=reinterpret_cast<const uint32_t*>(A);
    TileW<BITS> pf[PF][WNT];
    auto load=[&](int i,int t){
        size_t off=(size_t(start+i)*(N/16)+blockIdx.x*8+wn*2+t)*(8*BITS);
        return load_tile<BITS>(B+off,lane);
    };
    #pragma unroll
    for(int d=0;d<PF;++d)if(d<len){
        #pragma unroll
        for(int t=0;t<WNT;++t)pf[d][t]=load(d,t);
    }
    float acc[WNT][2][4]={};
    for(int base=0;base<len;base+=PF){
        #pragma unroll
        for(int d=0;d<PF;++d){
            int i=base+d;if(i>=len)break;
            TileW<BITS> bw[WNT];
            #pragma unroll
            for(int t=0;t<WNT;++t){bw[t]=pf[d][t];if(i+PF<len)pf[d][t]=load(i+PF,t);}
            int col=(start+i)*8+(lane&3)-lo*64;
            uint32_t a[4]={active?A2[r*(G::W/2)+col]:0u,0u,active?A2[r*(G::W/2)+col+4]:0u,0u};
            #pragma unroll
            for(int t=0;t<WNT;++t){Pair f0,f1;dqk<BITS>(bw[t],f0,f1,lane);mma(acc[t][0],a,f0);mma(acc[t][1],a,f1);}
        }
    }
    if(active){
        #pragma unroll
        for(int t=0;t<WNT;++t){
            #pragma unroll
            for(int f=0;f<2;++f){
                int c=wn*32+t*16+f*8+2*(lane&3);
                red[r*128+c]=acc[t][f][0];red[r*128+c+1]=acc[t][f][1];
            }
        }
    }
    cluster.sync();
    if(wk==0){
        const float* part[4];
        #pragma unroll
        for(int w=0;w<4;++w)part[w]=cluster.map_shared_rank(red,w);
        float* out=reinterpret_cast<float*>(A);   // this CTA's A is no longer read
        for(int i=threadIdx.x;i<job.tokens*128;i+=128){
            float sum=0;
            #pragma unroll
            for(int w=0;w<4;++w)sum+=part[w][i];
            out[i]=sum;
        }
    }
    cluster.sync();   // the other CTAs' shared memory stays valid until CTA 0 has read it
    if(wk==0){
        const float* out=reinterpret_cast<const float*>(A);
        for(int t=warp;t<job.tokens;t+=4)rotate_output(out+t*128,sv+blockIdx.x*128,job.output+t*N+blockIdx.x*128,lane);
    }
}
// Prefill: 64 rows x 128 cols. Eight warps own 16 cols each, reuse trellis
// fragments across four m16 tiles. K128 staging avoids an M*K scratch buffer.
// The 136-half stride distributes eight MMA rows over all shared banks.
template<int BITS>
__device__ __forceinline__ void prefill_body(const DeviceExpert* table,const GemmJob job,int projection,int N,int K,int max_tokens,int xs=-1,int os=-1) {
    if(xs<0)xs=K;
    if(os<0)os=N;
    if(job.tokens<1||job.tokens>max_tokens){asm volatile("trap;");return;}
    int m0=blockIdx.z*64;if(m0>=job.tokens)return;
    DeviceExpert e=table[job.expert];wait_ready(e);
    const uint32_t* B=weights(e,projection);
    const half* su=scales(e,projection);const half* sv=su+K;
    __shared__ __align__(16) half A[64*136];
    __shared__ float out[64*128];
    int warp=threadIdx.x/32,lane=threadIdx.x%32,r=lane/4;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x);
    float acc[4][2][4]={};
    for(int k0=0;k0<K;k0+=128){
        for(int row=warp;row<64;row+=8)if(m0+row<job.tokens)
            rotate_input(x+size_t(m0+row)*xs+k0,su+k0,A+row*136,lane);
        __syncthreads();
        #pragma unroll
        for(int ki=0;ki<8;++ki){
            const uint32_t* b=B+(size_t((k0/16)+ki)*(N/16)+blockIdx.x*8+warp)*(8*BITS);
            TileW<BITS> word=load_tile<BITS>(b,lane);
            Pair f0,f1;dqk<BITS>(word,f0,f1,lane);
            #pragma unroll
            for(int mi=0;mi<4;++mi){
                uint32_t a[4]={};
                #pragma unroll
                for(int hr=0;hr<2;++hr){int row=mi*16+r+hr*8;
                    if(m0+row<job.tokens){
                        const uint32_t* p=reinterpret_cast<const uint32_t*>(A+row*136)+ki*8+(lane&3);
                        a[hr]=p[0];a[hr+2]=p[4];
                    }
                }
                mma(acc[mi][0],a,f0);mma(acc[mi][1],a,f1);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int mi=0;mi<4;++mi){
        #pragma unroll
        for(int hr=0;hr<2;++hr){
            int row=mi*16+r+hr*8;
            #pragma unroll
            for(int f=0;f<2;++f){int col=warp*16+f*8+(lane&3)*2;
                out[row*128+col]=acc[mi][f][hr*2];out[row*128+col+1]=acc[mi][f][hr*2+1];
            }
        }
    }
    __syncthreads();
    for(int row=warp;row<64;row+=8)if(m0+row<job.tokens)
        rotate_output(out+row*128,sv+blockIdx.x*128,job.output+size_t(m0+row)*os+blockIdx.x*128,lane);
}
template<int BITS>
__global__ __launch_bounds__(256,1) void prefill(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,int K,int max_tokens) {DS41_PDL_WAIT();
    prefill_body<BITS>(table,jobs[blockIdx.y],projection,N,K,max_tokens);
}
// s0ma change 34 (N1): one dense job by value, up to 256 rows
template<int BITS>
__global__ __launch_bounds__(256,1) void prefill_one(const DeviceExpert* table,const GemmJob job,int N,int K,int xs,int os) {DS41_PDL_WAIT();
    prefill_body<BITS>(table,job,0,N,K,256,xs,os);
}
}
namespace {
// The bank's width, set once from its records before any launch (exl3_set_bits); K3 unless set.
int g_bits=3;
bool valid(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k){   // s0ma change 34: k to 8192
    return count>=0&&count<=65535&&int(p)>=0&&int(p)<=2&&n>0&&n%128==0&&k>0&&k%128==0&&k<=8192&&(count==0||(t&&j));
}
// s0ma: the cluster kernel only with DS41_EXL3_SPLIT=1. Measured slower in place (K2, 1.8 rows: 5.36 + 2.76 ms a step
// against decode()'s 4.73 + 1.95), so decode() stays the default; any launch error of the cluster kernel falls back.
bool split_on(){static const bool on=[]{const char*v=std::getenv("DS41_EXL3_SPLIT");return v&&v[0]=='1';}();return on;}
bool split_failed=false;
template<int K,int BITS> cudaError_t launch_decode(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,cudaStream_t s){
    if constexpr(K>5120){   // s0ma change 34: jobs of at most 4 rows (the caller splits larger ones)
        constexpr int bytes=4*K*2+4*8*128*4;
        auto err=cudaFuncSetAttribute(exl3_detail::decode<K,BITS,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
        if(err!=cudaSuccess)return err;
        exl3_detail::decode<K,BITS,4><<<dim3(n/128,count),512,bytes,s>>>(t,j,int(p),n);
        return cudaGetLastError();
    }
    if(split_on()&&!split_failed&&count<=65535){
        exl3_detail::decode_split<K,BITS><<<dim3(n/128,count,4),128,0,s>>>(t,j,int(p),n);
        cudaError_t e=cudaGetLastError();
        if(e==cudaSuccess)return e;
        split_failed=true;   // configuration errors are not sticky: fall back for good
    }
    constexpr int bytes=(8*K*2<4096?4096:8*K*2)+4*8*128*4;
    // Function attributes are host metadata operations, with no stream work,
    // allocation or synchronization; legal during capture as well as replay.
    auto err=cudaFuncSetAttribute(exl3_detail::decode<K,BITS>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess)return err;
    exl3_detail::decode<K,BITS><<<dim3(n/128,count),512,bytes,s>>>(t,j,int(p),n);
    return cudaGetLastError();
}
}
cudaError_t grouped_exl3(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,cudaStream_t s){
    return grouped_exl3_bits(t,j,count,p,n,k,g_bits,s);
}
template<int BITS> cudaError_t launch_k(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,cudaStream_t s){
    if(k==5120)return launch_decode<5120,BITS>(t,j,count,p,n,s);
    if(k==2304)return launch_decode<2304,BITS>(t,j,count,p,n,s);
    if(k==1280)return launch_decode<1280,BITS>(t,j,count,p,n,s);   // s0ma change 34: Sero's dense inputs
    if(k==4096)return launch_decode<4096,BITS>(t,j,count,p,n,s);
    if(k==8192)return launch_decode<8192,BITS>(t,j,count,p,n,s);
    if(k==256)return launch_decode<256,BITS>(t,j,count,p,n,s);
    if(k==128)return launch_decode<128,BITS>(t,j,count,p,n,s);
    return cudaErrorNotSupported;
}
template<int K> cudaError_t launch_var(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,const uint8_t* bits,cudaStream_t s){
    constexpr int bytes=(8*K*2<4096?4096:8*K*2)+4*8*128*4;
    auto err=cudaFuncSetAttribute(exl3_detail::decode_var<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess)return err;
    exl3_detail::decode_var<K><<<dim3(n/128,count),512,bytes,s>>>(t,j,int(p),n,bits);
    return cudaGetLastError();
}
cudaError_t grouped_exl3_var(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,const uint8_t* bits,cudaStream_t s){
    if(!valid(t,j,count,p,n,k) || !bits || k>5120)return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    if(k==5120)return launch_var<5120>(t,j,count,p,n,bits,s);
    if(k==2304)return launch_var<2304>(t,j,count,p,n,bits,s);
    if(k==256)return launch_var<256>(t,j,count,p,n,bits,s);
    if(k==128)return launch_var<128>(t,j,count,p,n,bits,s);
    return cudaErrorNotSupported;
}
cudaError_t grouped_exl3_bits(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,int bits,cudaStream_t s){
    if(!valid(t,j,count,p,n,k) || bits<2 || bits>5)return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    if(bits==4)return launch_k<4>(t,j,count,p,n,k,s);
    if(bits==5)return launch_k<5>(t,j,count,p,n,k,s);
    if(bits==2)return launch_k<2>(t,j,count,p,n,k,s);
    return launch_k<3>(t,j,count,p,n,k,s);
}
cudaError_t grouped_exl3_prefill(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,int max_tokens,cudaStream_t s){
    return grouped_exl3_prefill_bits(t,j,count,p,n,k,max_tokens,g_bits,s);
}
cudaError_t grouped_exl3_prefill_bits(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,int max_tokens,int bits,cudaStream_t s){
    if(!valid(t,j,count,p,n,k)||max_tokens<1||max_tokens>256||bits<2||bits>5)return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    if(bits==4){exl3_detail::prefill<4><<<dim3(n/128,count,(max_tokens+63)/64),256,0,s>>>(t,j,int(p),n,k,max_tokens);return cudaGetLastError();}
    if(bits==5){exl3_detail::prefill<5><<<dim3(n/128,count,(max_tokens+63)/64),256,0,s>>>(t,j,int(p),n,k,max_tokens);return cudaGetLastError();}
    if(bits==2)exl3_detail::prefill<2><<<dim3(n/128,count,(max_tokens+63)/64),256,0,s>>>(t,j,int(p),n,k,max_tokens);
    else exl3_detail::prefill<3><<<dim3(n/128,count,(max_tokens+63)/64),256,0,s>>>(t,j,int(p),n,k,max_tokens);
    return cudaGetLastError();
}
// s0ma change 34 (N1): a dense tensor (table[job.expert]'s gate views) on one job passed by value. Decode: 1..8 rows
// (1..4 for k=8192); prefill: 1..256 rows.
template<int K,int BITS> cudaError_t launch_one(const DeviceExpert* t,const GemmJob& j,int n,int xs,int os,int groups,int gx,int go,cudaStream_t s){
    constexpr int ROWS=K>5120?4:8;
    constexpr int bytes=(ROWS*K*2<4096?4096:ROWS*K*2)+4*8*128*4;
    if(j.tokens<1||j.tokens>ROWS)return cudaErrorInvalidValue;
    auto err=cudaFuncSetAttribute(exl3_detail::decode_one<K,BITS,ROWS>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess)return err;
    exl3_detail::decode_one<K,BITS,ROWS><<<dim3(n/128,groups),512,bytes,s>>>(t,j,n,xs,os,gx,go);
    return cudaGetLastError();
}
template<int BITS> cudaError_t launch_one_k(const DeviceExpert* t,const GemmJob& j,int n,int k,int xs,int os,int groups,int gx,int go,cudaStream_t s){
    switch(k){
        case 128: return launch_one<128,BITS>(t,j,n,xs,os,groups,gx,go,s);
        case 256: return launch_one<256,BITS>(t,j,n,xs,os,groups,gx,go,s);
        case 1280: return launch_one<1280,BITS>(t,j,n,xs,os,groups,gx,go,s);
        case 2304: return launch_one<2304,BITS>(t,j,n,xs,os,groups,gx,go,s);
        case 4096: return launch_one<4096,BITS>(t,j,n,xs,os,groups,gx,go,s);
        case 5120: return launch_one<5120,BITS>(t,j,n,xs,os,groups,gx,go,s);
        case 8192: return launch_one<8192,BITS>(t,j,n,xs,os,groups,gx,go,s);
        default: return cudaErrorNotSupported;
    }
}
cudaError_t exl3_dense(const DeviceExpert* t,GemmJob j,int n,int k,int bits,cudaStream_t s,int x_stride,int out_stride,
                       int groups,int group_x,int group_out){
    const int xs=x_stride?x_stride:k,os=out_stride?out_stride:n;
    if(!t||n<=0||n%128||k<=0||k%128||k>8192||xs<k||os<n||groups<1||groups>65535||(groups>1&&(group_x<k||group_out<n)))
        return cudaErrorInvalidValue;
    switch(bits){
        case 2: return launch_one_k<2>(t,j,n,k,xs,os,groups,group_x,group_out,s);
        case 3: return launch_one_k<3>(t,j,n,k,xs,os,groups,group_x,group_out,s);
        case 4: return launch_one_k<4>(t,j,n,k,xs,os,groups,group_x,group_out,s);
        case 5: return launch_one_k<5>(t,j,n,k,xs,os,groups,group_x,group_out,s);
        default: return cudaErrorInvalidValue;
    }
}
cudaError_t exl3_dense_prefill(const DeviceExpert* t,GemmJob j,int n,int k,int bits,cudaStream_t s,int x_stride,int out_stride){
    const int xs=x_stride?x_stride:k,os=out_stride?out_stride:n;
    if(!t||n<=0||n%128||k<=0||k%128||k>8192||xs<k||os<n||j.tokens<1||j.tokens>256)return cudaErrorInvalidValue;
    const dim3 grid(n/128,1,(j.tokens+63)/64);
    switch(bits){
        case 2: exl3_detail::prefill_one<2><<<grid,256,0,s>>>(t,j,n,k,xs,os); break;
        case 3: exl3_detail::prefill_one<3><<<grid,256,0,s>>>(t,j,n,k,xs,os); break;
        case 4: exl3_detail::prefill_one<4><<<grid,256,0,s>>>(t,j,n,k,xs,os); break;
        case 5: exl3_detail::prefill_one<5><<<grid,256,0,s>>>(t,j,n,k,xs,os); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
cudaError_t exl3_set_bits(int bits){
    if(bits<2||bits>5)return cudaErrorInvalidValue;
    g_bits=bits;return cudaSuccess;
}
int exl3_bits(){return g_bits;}
}
