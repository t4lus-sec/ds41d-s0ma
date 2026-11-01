#include <ds41/exl3.cuh>
#include "device.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
namespace ds41 {
namespace exl3_detail {
// Fixed 8-row tile, 128 columns, four contiguous K partitions. M never changes
// geometry, K order, rounding, or reduction order. No inter-CTA communication.
template<int K,int BITS>
__global__ __launch_bounds__(512,1) void decode(const DeviceExpert* table,const GemmJob* jobs,int projection,int N) {
    GemmJob job=jobs[blockIdx.y];
    if(job.tokens<1||job.tokens>8){asm volatile("trap;");return;}
    DeviceExpert e=table[job.expert];wait_ready(e);
    const uint32_t* B=weights(e,projection);
    const half* su=scales(e,projection); const half* sv=su+K;
    extern __shared__ __align__(16) unsigned char storage[];
    half* A=reinterpret_cast<half*>(storage);
    float* red=reinterpret_cast<float*>(A+(8*K<2048?2048:8*K));
    int warp=threadIdx.x/32,lane=threadIdx.x%32,wn=warp%4,wk=warp/4;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x);
    for(int idx=warp;idx<job.tokens*(K/128);idx+=16) {
        int r=idx/(K/128),c=(idx%(K/128))*128;
        rotate_input(x+r*K+c,su+c,A+r*K+c,lane);
    }
    __syncthreads();
    constexpr int SLICES=K/16,CHUNK=(SLICES+3)/4,PF=4,WNT=2;
    const int start=wk*CHUNK, len=min(CHUNK,SLICES-start);
    const int r=lane/4; bool active=r<job.tokens;
    const uint32_t* A2=reinterpret_cast<const uint32_t*>(A);
    uint32_t pf[PF][WNT];
    auto load=[&](int i,int t){
        size_t off=(size_t(start+i)*(N/16)+blockIdx.x*8+wn*2+t)*(8*BITS)+lane;
        return lane<8*BITS?__ldcs(B+off):0u;
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
            uint32_t bw[WNT];
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
    if(warp<job.tokens)rotate_output(out+warp*128,sv+blockIdx.x*128,job.output+warp*N+blockIdx.x*128,lane);
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
__global__ void __cluster_dims__(1,1,4) __launch_bounds__(128) decode_split(const DeviceExpert* table,const GemmJob* jobs,int projection,int N) {
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
    uint32_t pf[PF][WNT];
    auto load=[&](int i,int t){
        size_t off=(size_t(start+i)*(N/16)+blockIdx.x*8+wn*2+t)*(8*BITS)+lane;
        return lane<8*BITS?__ldcs(B+off):0u;
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
            uint32_t bw[WNT];
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
__global__ __launch_bounds__(256,1) void prefill(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,int K,int max_tokens) {
    GemmJob job=jobs[blockIdx.y];
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
            rotate_input(x+(m0+row)*K+k0,su+k0,A+row*136,lane);
        __syncthreads();
        #pragma unroll
        for(int ki=0;ki<8;++ki){
            const uint32_t* b=B+(size_t((k0/16)+ki)*(N/16)+blockIdx.x*8+warp)*(8*BITS);
            uint32_t word=lane<8*BITS?__ldcs(b+lane):0;
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
        rotate_output(out+row*128,sv+blockIdx.x*128,job.output+(m0+row)*N+blockIdx.x*128,lane);
}
}
namespace {
// The bank's width, set once from its records before any launch (exl3_set_bits); K3 unless set.
int g_bits=3;
bool valid(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k){
    return count>=0&&count<=65535&&int(p)>=0&&int(p)<=2&&n>0&&n%128==0&&k>0&&k%128==0&&k<=5120&&(count==0||(t&&j));
}
// s0ma: DS41_EXL3_SPLIT=0 keeps the one-CTA-per-tile kernel; any launch error of the cluster kernel falls back to it.
bool split_on(){static const bool on=[]{const char*v=std::getenv("DS41_EXL3_SPLIT");return !(v&&v[0]=='0');}();return on;}
bool split_failed=false;
template<int K,int BITS> cudaError_t launch_decode(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,cudaStream_t s){
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
    if(!valid(t,j,count,p,n,k))return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    if(g_bits==2){
        if(k==5120)return launch_decode<5120,2>(t,j,count,p,n,s);
        if(k==2304)return launch_decode<2304,2>(t,j,count,p,n,s);
        if(k==256)return launch_decode<256,2>(t,j,count,p,n,s);
        if(k==128)return launch_decode<128,2>(t,j,count,p,n,s);
        return cudaErrorNotSupported;
    }
    if(k==5120)return launch_decode<5120,3>(t,j,count,p,n,s);
    if(k==2304)return launch_decode<2304,3>(t,j,count,p,n,s);
    if(k==256)return launch_decode<256,3>(t,j,count,p,n,s);
    if(k==128)return launch_decode<128,3>(t,j,count,p,n,s);
    return cudaErrorNotSupported;
}
cudaError_t grouped_exl3_prefill(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,int max_tokens,cudaStream_t s){
    if(!valid(t,j,count,p,n,k)||max_tokens<1||max_tokens>256)return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    if(g_bits==2)exl3_detail::prefill<2><<<dim3(n/128,count,(max_tokens+63)/64),256,0,s>>>(t,j,int(p),n,k,max_tokens);
    else exl3_detail::prefill<3><<<dim3(n/128,count,(max_tokens+63)/64),256,0,s>>>(t,j,int(p),n,k,max_tokens);
    return cudaGetLastError();
}
cudaError_t exl3_set_bits(int bits){
    if(bits!=2&&bits!=3)return cudaErrorInvalidValue;
    g_bits=bits;return cudaSuccess;
}
int exl3_bits(){return g_bits;}
}
