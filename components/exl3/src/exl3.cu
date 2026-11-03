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
// s0ma change 43: OutT float keeps the FP32 result (job.output then points to floats; os in floats)
template<int K,int BITS,int ROWS=8,typename OutT=__nv_bfloat16>
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
    if(warp<job.tokens)rotate_output(out+warp*128,sv+blockIdx.x*128,reinterpret_cast<OutT*>(job.output)+size_t(warp)*os+blockIdx.x*128,lane);
}
template<int K,int BITS,int ROWS=8>
__global__ __launch_bounds__(512,1) void decode(const DeviceExpert* table,const GemmJob* jobs,int projection,int N) {DS41_PDL_WAIT();
    if(jobs[blockIdx.y].tokens<=0)return;   // s0ma change 61: an empty job (a padded device-built job list) does nothing
    decode_body<K,BITS,ROWS>(table,jobs[blockIdx.y],projection,N);
}
// s0ma change 34 (N1): one job passed by value (a dense tensor: no device job array, and a captured graph keeps it)
template<int K,int BITS,int ROWS=8,typename OutT=__nv_bfloat16>
// blockIdx.y = g of a set of tensors at consecutive table entries, reading x + g*gx and writing output + g*go
__global__ __launch_bounds__(512,1) void decode_one(const DeviceExpert* table,const GemmJob job,int N,int xs,int os,int gx,int go) {DS41_PDL_WAIT();
    GemmJob mine=job;
    mine.expert=job.expert+blockIdx.y;
    mine.x=job.x+size_t(blockIdx.y)*gx*2;
    mine.output=reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<OutT*>(job.output)+size_t(blockIdx.y)*go);
    decode_body<K,BITS,ROWS,OutT>(table,mine,0,N,xs,os);
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
    if(job.tokens==0)return;   // s0ma change 61: an empty job; every CTA of the cluster has the same job
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
// s0ma change 68: prefill with each job's rows rotated once a projection (prefill_body rotates them again in every
// one of the N/128 column blocks, eight dependent 16-bit loads and a shuffle butterfly a row and k-block, latency the
// CTA's eight warps cannot hide). rotate_job writes the FP16 rows prefill_body would put in shared memory (the same
// rotate_input on the same values), up to two projections from one read of the input (gate and up). prefill_rot then
// copies them in with 16-byte cp.async, double-buffered, and loads the next k-block's trellis words during this one.
// The mma sequence, k order and epilogue are prefill_body's: the same bits out.
__global__ __launch_bounds__(256) void rotate_job(const DeviceExpert* table,const GemmJob* jobs,int p0,int p1,int K,
                                                  half* a0,half* a1) {DS41_PDL_WAIT();
    const GemmJob job=jobs[0];
    const int kb=K/128,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp>=job.tokens*kb)return;
    DeviceExpert e=table[job.expert];wait_ready(e);
    const int r=warp/kb,c=(warp%kb)*128;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x)+size_t(r)*K+c;
    rotate_input(x,scales(e,p0)+c,a0+size_t(r)*K+c,lane);
    if(a1)rotate_input(x,scales(e,p1)+c,a1+size_t(r)*K+c,lane);
}
// s0ma change 76: rotate_job for many jobs at once (blockIdx.y the job): job j0[y]'s rows rotated with p0's scales into
// j0[y].x_scale and, with j1, with p1's into j1[y].x_scale (the same x). Each row as rotate_job does it.
__global__ __launch_bounds__(256) void rotate_group(const DeviceExpert* table,const GemmJob* j0,const GemmJob* j1,int p0,
                                                    int p1,int K) {DS41_PDL_WAIT();
    const GemmJob a=j0[blockIdx.y];
    const int kb=K/128,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp>=a.tokens*kb)return;
    DeviceExpert e=table[a.expert];wait_ready(e);
    const int r=warp/kb,c=(warp%kb)*128;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(a.x)+size_t(r)*K+c;
    rotate_input(x,scales(e,p0)+c,const_cast<half*>(reinterpret_cast<const half*>(a.x_scale))+size_t(r)*K+c,lane);
    if(j1){const GemmJob b=j1[blockIdx.y];
        rotate_input(x,scales(e,p1)+c,const_cast<half*>(reinterpret_cast<const half*>(b.x_scale))+size_t(r)*K+c,lane);}
}
__device__ __forceinline__ void cp_async16(void* dst,const void* src) {
    const unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(src));
}
template<int BITS>
__global__ __launch_bounds__(256,1) void prefill_rot(const DeviceExpert* table,const GemmJob* jobs,int projection,
                                                               int N,int K,const half* rotated) {DS41_PDL_WAIT();
    const GemmJob job=jobs[blockIdx.y];
    const int m0=blockIdx.z*64;if(m0>=job.tokens)return;
    // s0ma change 76: a grouped launch (many jobs) passes no `rotated`: each job's rows were rotated into job.x_scale
    if(!rotated)rotated=reinterpret_cast<const half*>(job.x_scale);
    DeviceExpert e=table[job.expert];wait_ready(e);
    const uint32_t* B=weights(e,projection);
    const half* sv=scales(e,projection)+K;
    // two 64x136 FP16 stages (34,816 B); after the k loop the same bytes hold the 64x128 FP32 results (32,768 B)
    __shared__ __align__(16) unsigned char smem[2*64*136*2];
    half* As=reinterpret_cast<half*>(smem);
    float* out=reinterpret_cast<float*>(smem);
    const int warp=threadIdx.x/32,lane=threadIdx.x%32,r=lane/4,rows=min(64,job.tokens-m0);
    auto stage=[&](int k0,int st){
        half* dst=As+st*64*136;
        for(int i=threadIdx.x;i<rows*16;i+=256){const int row=i>>4,c=(i&15)*8;
            cp_async16(dst+row*136+c,rotated+size_t(m0+row)*K+k0+c);}
        asm volatile("cp.async.commit_group;");
    };
    auto words=[&](int k0,TileW<BITS> (&w)[8]){
        #pragma unroll
        for(int ki=0;ki<8;++ki)w[ki]=load_tile<BITS>(B+(size_t((k0/16)+ki)*(N/16)+blockIdx.x*8+warp)*(8*BITS),lane);
    };
    float acc[4][2][4]={};
    TileW<BITS> cur[8],next[8];
    stage(0,0);words(0,cur);
    for(int k0=0,st=0;k0<K;k0+=128,st^=1){
        const bool more=k0+128<K;
        if(more){stage(k0+128,st^1);asm volatile("cp.async.wait_group 1;":::"memory");}
        else asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        if(more)words(k0+128,next);
        const half* A=As+st*64*136;
        #pragma unroll
        for(int ki=0;ki<8;++ki){
            Pair f0,f1;dqk<BITS>(cur[ki],f0,f1,lane);
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
        if(more){
            #pragma unroll
            for(int ki=0;ki<8;++ki)cur[ki]=next[ki];
        }
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
        rotate_output(out+row*128,sv+blockIdx.x*128,job.output+size_t(m0+row)*N+blockIdx.x*128,lane);
}
// s0ma change 34 (N1): one dense job by value, up to 256 rows
template<int BITS>
__global__ __launch_bounds__(256,1) void prefill_one(const DeviceExpert* table,const GemmJob job,int N,int K,int xs,int os) {DS41_PDL_WAIT();
    prefill_body<BITS>(table,job,0,N,K,256,xs,os);
}
// s0ma change 37: dense decode split over K across CTAs. 0xSero's dense tensors are narrow (wkv: 4 column tiles of
// 128, wq_a 10, shared w1/w3 18, wo_b 40) so one CTA a tile left most of the 188 SMs idle (N1's first bench:
// 63.2 ms a verify step against FP8's 56.7). Here blockIdx.z takes K range [z*KC, z*KC + KC): each CTA rotates and
// multiplies its slice of the input as decode_body does, writes its 128 columns' FP32 partial sums to scratch, and
// the last CTA of a tile to arrive (an atomic count) adds the partials in z order 0..S-1, so the result does not
// depend on arrival order, rotates the sum and writes BF16. It then resets the count for the next launch.
// The split-K work of one CTA: e's K range [z*KC, z*KC + KC) of 128 columns (blockIdx.x) for job's rows, x and out
// already at the job's own columns, its partials at partial (S x 8 x N floats), its tiles' counts at counts. Shared
// buffers come from the kernel (one set whatever BITS a launch decodes).
template<int KC,int BITS>
__device__ __forceinline__ void splitk_body(const DeviceExpert& e,const GemmJob& job,int projection,int N,int K,int xs,
                                            int os,const __nv_bfloat16* x,__nv_bfloat16* out_rows,float* partial,
                                            unsigned* counts,half* A,float* red,int* last) {
    constexpr int ROWS=8;
    const int S=K/KC, z=blockIdx.z;
    const uint32_t* B=weights(e,projection);
    const half* su=scales(e,projection); const half* sv=su+K;
    const int warp=threadIdx.x/32,lane=threadIdx.x%32,wn=warp%4,wk=warp/4;
    x+=size_t(z)*KC;
    for(int idx=warp;idx<job.tokens*(KC/128);idx+=16) {
        int r=idx/(KC/128),c=(idx%(KC/128))*128;
        rotate_input(x+size_t(r)*xs+c,su+size_t(z)*KC+c,A+r*KC+c,lane);
    }
    __syncthreads();
    constexpr int SLICES=KC/16,CHUNK=(SLICES+3)/4,PF=4,WNT=2;
    const int start=wk*CHUNK, len=min(CHUNK,SLICES-start), slice0=z*SLICES;
    const int r=lane/4; const bool active=r<job.tokens;
    const uint32_t* A2=reinterpret_cast<const uint32_t*>(A);
    TileW<BITS> pf[PF][WNT];
    auto load=[&](int i,int t){
        size_t off=(size_t(slice0+start+i)*(N/16)+blockIdx.x*8+wn*2+t)*(8*BITS);
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
            uint32_t a[4]={active?A2[r*(KC/2)+col]:0u,0u,active?A2[r*(KC/2)+col+4]:0u,0u};
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
    float* mine=partial+size_t(z)*ROWS*N;   // this CTA's partial: [z][row][N], its 128 columns
    for(int i=threadIdx.x;i<job.tokens*128;i+=512){
        float sum=0;
        #pragma unroll
        for(int w=0;w<4;++w)sum+=red[w*8*128+i];
        mine[size_t(i/128)*N+blockIdx.x*128+i%128]=sum;
    }
    __threadfence();
    __syncthreads();
    if(!threadIdx.x) *last=atomicAdd(counts+blockIdx.x,1u)==unsigned(S-1);
    __syncthreads();
    if(!*last)return;
    __threadfence();
    float* sum_out=red;   // the sum in z order, then the rotation, as decode_body's epilogue
    for(int i=threadIdx.x;i<job.tokens*128;i+=512){
        const size_t at=size_t(i/128)*N+blockIdx.x*128+i%128;
        float sum=0;
        for(int s=0;s<S;++s)sum+=__ldcg(partial+size_t(s)*ROWS*N+at);
        sum_out[i]=sum;
    }
    __syncthreads();
    if(warp<job.tokens)rotate_output(sum_out+warp*128,sv+blockIdx.x*128,out_rows+size_t(warp)*os+blockIdx.x*128,lane);
    if(!threadIdx.x) counts[blockIdx.x]=0u;
}
// s0ma change 37: dense decode split over K across CTAs. 0xSero's dense tensors are narrow (wkv: 4 column tiles of
// 128, wq_a 10, shared w1/w3 18, wo_b 40) so one CTA a tile left most of the 188 SMs idle. blockIdx.z takes K range
// [z*KC, z*KC + KC); the last CTA of a tile to arrive (an atomic count) adds the partials in z order 0..S-1, so the
// result does not depend on arrival order, rotates the sum and writes BF16, then resets the count.
template<int KC,int BITS>
__global__ __launch_bounds__(512,1) void decode_splitk(const DeviceExpert* table,const GemmJob job,int N,int K,int xs,int os,
                                                      int gx,int go,float* partial,unsigned* counts) {DS41_PDL_WAIT();
    constexpr int ROWS=8;
    __shared__ __align__(16) half A[ROWS*KC<2048?2048:ROWS*KC];
    __shared__ __align__(16) float red[4*8*128];
    __shared__ int last;
    const int g=blockIdx.y, S=K/KC;
    if(job.tokens<1||job.tokens>ROWS){asm volatile("trap;");return;}
    const DeviceExpert e=table[job.expert+g];
    splitk_body<KC,BITS>(e,job,0,N,K,xs,os,reinterpret_cast<const __nv_bfloat16*>(job.x)+size_t(g)*gx,
                         job.output+size_t(g)*go,partial+size_t(g)*S*ROWS*N,counts+size_t(g)*(N/128),A,red,&last);
}
// s0ma change 40: the routed experts' grouped launch split over K the same way (DS41_EXL3_SPLITK=1): job y's expert
// at its own width (bits[expert], or `uniform`), its own partials and counts. A layer's handful of experts gives
// gate/up 18 column tiles each: 6 experts filled 108 of 188 SMs.
template<int KC>
__global__ __launch_bounds__(512,1) void decode_var_splitk(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,
                                                          int K,const uint8_t* bits,int uniform,float* partial,unsigned* counts) {DS41_PDL_WAIT();
    constexpr int ROWS=8;
    __shared__ __align__(16) half A[ROWS*KC<2048?2048:ROWS*KC];
    __shared__ __align__(16) float red[4*8*128];
    __shared__ int last;
    const GemmJob job=jobs[blockIdx.y];
    if(job.tokens<1||job.tokens>ROWS){asm volatile("trap;");return;}
    const DeviceExpert e=table[job.expert];wait_ready(e);
    const int S=K/KC, b=bits?bits[job.expert]:uniform;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x);
    float* part=partial+size_t(blockIdx.y)*S*ROWS*N;
    unsigned* cnt=counts+size_t(blockIdx.y)*(N/128);
    switch(b){
        case 2: splitk_body<KC,2>(e,job,projection,N,K,K,N,x,job.output,part,cnt,A,red,&last); break;
        case 3: splitk_body<KC,3>(e,job,projection,N,K,K,N,x,job.output,part,cnt,A,red,&last); break;
        case 4: splitk_body<KC,4>(e,job,projection,N,K,K,N,x,job.output,part,cnt,A,red,&last); break;
        case 5: splitk_body<KC,5>(e,job,projection,N,K,K,N,x,job.output,part,cnt,A,red,&last); break;
        default: asm volatile("trap;");
    }
}
// s0ma change 37: prefill of a dense tensor as exllamav3 does for many rows: the weights decoded once into FP16
// [K, N] (row k = input feature), the rows rotated into FP16, one FP16 GEMM with FP32 sums (the caller's cuBLAS), and
// the FP32 result rotated and scaled into BF16. The same products as prefill_body, summed in cuBLAS's order.
// s0ma change 42: a block decodes one 16-row strip of 8 tiles (128 columns) into shared memory and writes it out in
// 16-byte rows, coalesced (change 37 wrote each weight on its own: 2-byte stores 16 rows apart)
template<int BITS>
__global__ __launch_bounds__(256) void reconstruct(const DeviceExpert* table,int index,int N,int K,half* W,
                                                  int projection=0) {DS41_PDL_WAIT();   // s0ma change 82: any projection
    __shared__ __align__(16) half strip[16][136];
    const int lane=threadIdx.x%32, warp=threadIdx.x/32, kt=blockIdx.y, nt=blockIdx.x*8+warp;
    const uint32_t* B=weights(table[index],projection)+(size_t(kt)*(N/16)+nt)*(8*BITS);
    TileW<BITS> w=load_tile<BITS>(B,lane);
    Pair f[2];
    dqk<BITS>(w,f[0],f[1],lane);
    #pragma unroll
    for(int h=0;h<2;++h){
        const half2 lo=*reinterpret_cast<const half2*>(&f[h].v[0]), hi=*reinterpret_cast<const half2*>(&f[h].v[1]);
        const int n=warp*16+h*8+lane/4, k=(lane%4)*2;   // the m16n8k16 B fragment: jj 0,1 at k, k+1; 2,3 at k+8, k+9
        strip[k][n]=__low2half(lo); strip[k+1][n]=__high2half(lo);
        strip[k+8][n]=__low2half(hi); strip[k+9][n]=__high2half(hi);
    }
    __syncthreads();
    const int row=threadIdx.x/16, col=(threadIdx.x%16)*8;
    *reinterpret_cast<uint4*>(W+(size_t(kt)*16+row)*N+size_t(blockIdx.x)*128+col)=*reinterpret_cast<const uint4*>(&strip[row][col]);
}
__global__ __launch_bounds__(256) void rotate_rows_in(const DeviceExpert* table,int index,const __nv_bfloat16* x,int rows,
                                                      int K,int xs,half* A) {DS41_PDL_WAIT();
    const int lane=threadIdx.x%32, item=blockIdx.x*8+threadIdx.x/32, blocks=K/128;
    if(item>=rows*blocks)return;
    const int r=item/blocks, c=(item%blocks)*128;
    rotate_input(x+size_t(r)*xs+c,scales(table[index],0)+c,A+size_t(r)*K+c,lane);
}
__global__ __launch_bounds__(256) void rotate_rows_out(const DeviceExpert* table,int index,const float* C,int rows,int N,
                                                       int K,__nv_bfloat16* out,int os,int projection=0) {DS41_PDL_WAIT();
    const int lane=threadIdx.x%32, item=blockIdx.x*8+threadIdx.x/32, blocks=N/128;
    if(item>=rows*blocks)return;
    const int r=item/blocks, c=(item%blocks)*128;
    rotate_output(C+size_t(r)*N+c,scales(table[index],projection)+K+c,out+size_t(r)*os+c,lane);
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
template<int K,int BITS,typename OutT=__nv_bfloat16> cudaError_t launch_one(const DeviceExpert* t,const GemmJob& j,int n,int xs,int os,int groups,int gx,int go,cudaStream_t s){
    constexpr int ROWS=K>5120?4:8;
    constexpr int bytes=(ROWS*K*2<4096?4096:ROWS*K*2)+4*8*128*4;
    if(j.tokens<1||j.tokens>ROWS)return cudaErrorInvalidValue;
    auto err=cudaFuncSetAttribute(exl3_detail::decode_one<K,BITS,ROWS,OutT>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess)return err;
    exl3_detail::decode_one<K,BITS,ROWS,OutT><<<dim3(n/128,groups),512,bytes,s>>>(t,j,n,xs,os,gx,go);
    return cudaGetLastError();
}
template<int BITS,typename OutT=__nv_bfloat16> cudaError_t launch_one_k(const DeviceExpert* t,const GemmJob& j,int n,int k,int xs,int os,int groups,int gx,int go,cudaStream_t s){
    switch(k){
        case 128: return launch_one<128,BITS,OutT>(t,j,n,xs,os,groups,gx,go,s);
        case 256: return launch_one<256,BITS,OutT>(t,j,n,xs,os,groups,gx,go,s);
        case 1280: return launch_one<1280,BITS,OutT>(t,j,n,xs,os,groups,gx,go,s);
        case 2304: return launch_one<2304,BITS,OutT>(t,j,n,xs,os,groups,gx,go,s);
        case 4096: return launch_one<4096,BITS,OutT>(t,j,n,xs,os,groups,gx,go,s);
        case 5120: return launch_one<5120,BITS,OutT>(t,j,n,xs,os,groups,gx,go,s);
        case 8192: return launch_one<8192,BITS,OutT>(t,j,n,xs,os,groups,gx,go,s);
        default: return cudaErrorNotSupported;
    }
}
// s0ma change 37: K split across CTAs when the tensor has few column tiles: the largest chunk KC of 1024, 512, 256,
// 128 that divides k and gives at least 120 CTAs (else 128), when the caller gave scratch for the partial sums
template<int KC> cudaError_t launch_splitk(const DeviceExpert* t,const GemmJob& j,int n,int k,int bits,int xs,int os,
                                           int groups,int gx,int go,float* partial,unsigned* counts,cudaStream_t s){
    const dim3 grid(n/128,groups,k/KC);
    switch(bits){
        case 2: exl3_detail::decode_splitk<KC,2><<<grid,512,0,s>>>(t,j,n,k,xs,os,gx,go,partial,counts); break;
        case 3: exl3_detail::decode_splitk<KC,3><<<grid,512,0,s>>>(t,j,n,k,xs,os,gx,go,partial,counts); break;
        case 4: exl3_detail::decode_splitk<KC,4><<<grid,512,0,s>>>(t,j,n,k,xs,os,gx,go,partial,counts); break;
        case 5: exl3_detail::decode_splitk<KC,5><<<grid,512,0,s>>>(t,j,n,k,xs,os,gx,go,partial,counts); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
int splitk_chunk(int n,int k,int groups){
    const long tiles=long(groups)*(n/128);
    if(tiles>=120)return 0;
    for(int kc:{1024,512,256,128})if(k%kc==0&&k/kc>=2&&tiles*(k/kc)>=120)return kc;
    return k%128==0&&k/128>=2?128:0;
}
size_t exl3_dense_splitk_floats(int n,int k,int groups){
    const int kc=splitk_chunk(n,k,groups);
    return kc?size_t(groups)*(k/kc)*8*n:0;
}
cudaError_t exl3_dense(const DeviceExpert* t,GemmJob j,int n,int k,int bits,cudaStream_t s,int x_stride,int out_stride,
                       int groups,int group_x,int group_out,float* partial,size_t partial_floats,unsigned* counts,
                       size_t count_slots){
    const int xs=x_stride?x_stride:k,os=out_stride?out_stride:n;
    if(!t||n<=0||n%128||k<=0||k%128||k>8192||xs<k||os<n||groups<1||groups>65535||(groups>1&&(group_x<k||group_out<n)))
        return cudaErrorInvalidValue;
    if(const int kc=splitk_chunk(n,k,groups);kc&&partial&&counts&&j.tokens>=1&&j.tokens<=8&&
       partial_floats>=exl3_dense_splitk_floats(n,k,groups)&&count_slots>=size_t(groups)*(n/128)){
        switch(kc){
            case 1024: return launch_splitk<1024>(t,j,n,k,bits,xs,os,groups,group_x,group_out,partial,counts,s);
            case 512: return launch_splitk<512>(t,j,n,k,bits,xs,os,groups,group_x,group_out,partial,counts,s);
            case 256: return launch_splitk<256>(t,j,n,k,bits,xs,os,groups,group_x,group_out,partial,counts,s);
            default: return launch_splitk<128>(t,j,n,k,bits,xs,os,groups,group_x,group_out,partial,counts,s);
        }
    }
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
// s0ma change 43: decode of one dense tensor into FP32 rows (Sero's K4 head: the logits), the unsplit kernel
cudaError_t exl3_dense_f32(const DeviceExpert* t,GemmJob j,int n,int k,int bits,cudaStream_t s,int x_stride,int out_stride){
    const int xs=x_stride?x_stride:k,os=out_stride?out_stride:n;
    if(!t||n<=0||n%128||k<=0||k%128||k>8192||xs<k||os<n||j.tokens<1)return cudaErrorInvalidValue;
    switch(bits){
        case 2: return launch_one_k<2,float>(t,j,n,k,xs,os,1,0,0,s);
        case 3: return launch_one_k<3,float>(t,j,n,k,xs,os,1,0,0,s);
        case 4: return launch_one_k<4,float>(t,j,n,k,xs,os,1,0,0,s);
        case 5: return launch_one_k<5,float>(t,j,n,k,xs,os,1,0,0,s);
        default: return cudaErrorInvalidValue;
    }
}
// s0ma change 68: prefill with the rows rotated once (exl3.cuh)
cudaError_t exl3_prefill_rotate(const DeviceExpert* t,const GemmJob* job,Projection p0,Projection p1,int k,int rows,
                                half* a0,half* a1,cudaStream_t s){
    if(!t||!job||!a0||rows<1||rows>256||k<=0||k%128)return cudaErrorInvalidValue;
    exl3_detail::rotate_job<<<(rows*(k/128)+7)/8,256,0,s>>>(t,job,int(p0),int(p1),k,a0,a1);
    return cudaGetLastError();
}
cudaError_t exl3_prefill_rotated(const DeviceExpert* t,const GemmJob* job,Projection p,int n,int k,int rows,int bits,
                                 const half* a,cudaStream_t s){
    if(!t||!job||!a||rows<1||rows>256||n<=0||n%128||k<=0||k%128)return cudaErrorInvalidValue;
    const dim3 grid(n/128,1,(rows+63)/64);
    switch(bits){
        case 2: exl3_detail::prefill_rot<2><<<grid,256,0,s>>>(t,job,int(p),n,k,a); break;
        case 3: exl3_detail::prefill_rot<3><<<grid,256,0,s>>>(t,job,int(p),n,k,a); break;
        case 4: exl3_detail::prefill_rot<4><<<grid,256,0,s>>>(t,job,int(p),n,k,a); break;
        case 5: exl3_detail::prefill_rot<5><<<grid,256,0,s>>>(t,job,int(p),n,k,a); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
// s0ma change 76: the same for `count` jobs in one launch each (exl3.cuh)
cudaError_t exl3_prefill_rotate_group(const DeviceExpert* t,const GemmJob* j0,const GemmJob* j1,int count,Projection p0,
                                      Projection p1,int k,int max_rows,cudaStream_t s){
    if(!t||!j0||count<1||count>65535||max_rows<1||max_rows>256||k<=0||k%128)return cudaErrorInvalidValue;
    exl3_detail::rotate_group<<<dim3((max_rows*(k/128)+7)/8,count),256,0,s>>>(t,j0,j1,int(p0),int(p1),k);
    return cudaGetLastError();
}
cudaError_t exl3_prefill_rotated_group(const DeviceExpert* t,const GemmJob* jobs,int count,Projection p,int n,int k,
                                       int max_rows,int bits,cudaStream_t s){
    if(!t||!jobs||count<1||count>65535||max_rows<1||max_rows>256||n<=0||n%128||k<=0||k%128)return cudaErrorInvalidValue;
    const dim3 grid(n/128,count,(max_rows+63)/64);
    switch(bits){
        case 2: exl3_detail::prefill_rot<2><<<grid,256,0,s>>>(t,jobs,int(p),n,k,nullptr); break;
        case 3: exl3_detail::prefill_rot<3><<<grid,256,0,s>>>(t,jobs,int(p),n,k,nullptr); break;
        case 4: exl3_detail::prefill_rot<4><<<grid,256,0,s>>>(t,jobs,int(p),n,k,nullptr); break;
        case 5: exl3_detail::prefill_rot<5><<<grid,256,0,s>>>(t,jobs,int(p),n,k,nullptr); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
// s0ma change 37: prefill pieces for the caller's FP16 GEMM
cudaError_t exl3_dense_reconstruct(const DeviceExpert* t,int index,int n,int k,int bits,half* w,cudaStream_t s){
    if(!t||!w||n<=0||n%128||k<=0||k%128)return cudaErrorInvalidValue;
    const dim3 blocks(n/128,k/16);
    switch(bits){
        case 2: exl3_detail::reconstruct<2><<<blocks,256,0,s>>>(t,index,n,k,w); break;
        case 3: exl3_detail::reconstruct<3><<<blocks,256,0,s>>>(t,index,n,k,w); break;
        case 4: exl3_detail::reconstruct<4><<<blocks,256,0,s>>>(t,index,n,k,w); break;
        case 5: exl3_detail::reconstruct<5><<<blocks,256,0,s>>>(t,index,n,k,w); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
// s0ma change 82: the same two for one projection of a routed expert (its prefill through a cuBLAS FP16 GEMM)
cudaError_t exl3_expert_reconstruct(const DeviceExpert* t,ExpertId id,Projection p,int n,int k,int bits,half* w,cudaStream_t s){
    if(!t||!w||n<=0||n%128||k<=0||k%128||int(p)<0||int(p)>2)return cudaErrorInvalidValue;
    const dim3 blocks(n/128,k/16);
    const int index=int(id),pr=int(p);
    switch(bits){
        case 2: exl3_detail::reconstruct<2><<<blocks,256,0,s>>>(t,index,n,k,w,pr); break;
        case 3: exl3_detail::reconstruct<3><<<blocks,256,0,s>>>(t,index,n,k,w,pr); break;
        case 4: exl3_detail::reconstruct<4><<<blocks,256,0,s>>>(t,index,n,k,w,pr); break;
        case 5: exl3_detail::reconstruct<5><<<blocks,256,0,s>>>(t,index,n,k,w,pr); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
cudaError_t exl3_expert_rotate_out(const DeviceExpert* t,ExpertId id,Projection p,const float* c,int rows,int n,int k,
                                   __nv_bfloat16* out,int out_stride,cudaStream_t s){
    if(!t||!c||!out||rows<1||n<=0||n%128||int(p)<0||int(p)>2)return cudaErrorInvalidValue;
    exl3_detail::rotate_rows_out<<<(rows*(n/128)+7)/8,256,0,s>>>(t,int(id),c,rows,n,k,out,out_stride?out_stride:n,int(p));
    return cudaGetLastError();
}
cudaError_t exl3_dense_rotate_in(const DeviceExpert* t,int index,const __nv_bfloat16* x,int rows,int k,int x_stride,
                                 half* a,cudaStream_t s){
    if(!t||!x||!a||rows<1||k<=0||k%128)return cudaErrorInvalidValue;
    exl3_detail::rotate_rows_in<<<(rows*(k/128)+7)/8,256,0,s>>>(t,index,x,rows,k,x_stride?x_stride:k,a);
    return cudaGetLastError();
}
cudaError_t exl3_dense_rotate_out(const DeviceExpert* t,int index,const float* c,int rows,int n,int k,
                                  __nv_bfloat16* out,int out_stride,cudaStream_t s){
    if(!t||!c||!out||rows<1||n<=0||n%128)return cudaErrorInvalidValue;
    exl3_detail::rotate_rows_out<<<(rows*(n/128)+7)/8,256,0,s>>>(t,index,c,rows,n,k,out,out_stride?out_stride:n);
    return cudaGetLastError();
}
// s0ma change 40: the routed experts' grouped launch split over K (k 5120: chunks of 1280 or 2560; k 2304: 768 or
// 1152): the fewest splits giving at least 564 CTAs (three per SM), the most if none does; none above 564 already
int splitk_var_chunk(int n,int k,int count){
    const long tiles=long(count)*(n/128);
    if(tiles>=564)return 0;
    const int* opts=nullptr;int m=0;
    static const int k5120[]={2560,1280},k2304[]={1152,768};
    if(k==5120){opts=k5120;m=2;}else if(k==2304){opts=k2304;m=2;}else return 0;
    for(int i=0;i<m;++i)if(tiles*(k/opts[i])>=564)return opts[i];
    return opts[m-1];
}
size_t exl3_var_splitk_floats(int n,int k,int count){
    const int kc=splitk_var_chunk(n,k,count);
    return kc?size_t(count)*(k/kc)*8*n:0;
}
cudaError_t grouped_exl3_splitk(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,
                                const uint8_t* bits,int uniform,float* partial,size_t partial_floats,unsigned* counts,
                                size_t count_slots,cudaStream_t s){
    if(!valid(t,j,count,p,n,k)||(!bits&&(uniform<2||uniform>5)))return cudaErrorInvalidValue;
    if(!count)return cudaSuccess;
    const int kc=splitk_var_chunk(n,k,count);
    if(!kc||!partial||!counts||partial_floats<exl3_var_splitk_floats(n,k,count)||count_slots<size_t(count)*(n/128))
        return bits?grouped_exl3_var(t,j,count,p,n,k,bits,s):grouped_exl3_bits(t,j,count,p,n,k,uniform,s);
    const dim3 grid(n/128,count,k/kc);
    switch(kc){
        case 2560: exl3_detail::decode_var_splitk<2560><<<grid,512,0,s>>>(t,j,int(p),n,k,bits,uniform,partial,counts); break;
        case 1280: exl3_detail::decode_var_splitk<1280><<<grid,512,0,s>>>(t,j,int(p),n,k,bits,uniform,partial,counts); break;
        case 1152: exl3_detail::decode_var_splitk<1152><<<grid,512,0,s>>>(t,j,int(p),n,k,bits,uniform,partial,counts); break;
        case 768: exl3_detail::decode_var_splitk<768><<<grid,512,0,s>>>(t,j,int(p),n,k,bits,uniform,partial,counts); break;
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
