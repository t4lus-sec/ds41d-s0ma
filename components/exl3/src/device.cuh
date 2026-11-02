#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace ds41::exl3_detail {
// Trellis extraction and mul1 codebook adapted from exllamav3 v1.6.3 (MIT).
// Copyright (c) 2025 Turboderp. See vendor/EXLLAMAV3_LICENSE.
struct Pair { uint32_t v[2]; };
__device__ __forceinline__ uint32_t pair(float x, float y) {
    half2 v=__floats2half2_rn(x,y); return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t decode_pair(uint32_t x, uint32_t y) {
    x *= 0x83DCD12Du; y *= 0x83DCD12Du;
    uint32_t sx=__dp4a(x,0x01010101u,0x6400u), sy=__dp4a(y,0x01010101u,0x6400u);
    half2 f=__halves2half2(__ushort_as_half(sx),__ushort_as_half(sy));
    half2 v=__hfma2(f,__half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931))); return *reinterpret_cast<uint32_t*>(&v);
}
// One lane's eight weights of a 16x16 tile (BITS bits each, 8*BITS words per tile, word `lane` held by lane
// `lane`): exllamav3's dq8<BITS> with the tile's words exchanged by shuffles. BITS=3 is the K3 kernel's dq3 exactly.
template<int BITS>
__device__ __forceinline__ void dqk(uint32_t word, Pair& f0, Pair& f1, int lane) {
    constexpr int W=8*BITS;
    const int b1=(lane*8+257)*BITS, b2=b1+7*BITS;
    const int i0=(b1-16)/32, i2=(b2-1)/32, s=(i2+1)*32-b2;
    uint32_t a=__shfl_sync(0xffffffff,word,i0%W), b=__shfl_sync(0xffffffff,word,i2%W);
    uint64_t joined=(uint64_t(a)<<32)|b;
    uint32_t w7=joined>>s, w3=joined>>(s+4*BITS);
    f0.v[0]=decode_pair((w3>>(3*BITS))&65535,(w3>>(2*BITS))&65535);
    f0.v[1]=decode_pair((w3>>BITS)&65535,w3&65535);
    f1.v[0]=decode_pair((w7>>(3*BITS))&65535,(w7>>(2*BITS))&65535);
    f1.v[1]=decode_pair((w7>>BITS)&65535,w7&65535);
}
// s0ma change 15: a tile's words as the lanes hold them. A tile has 8*BITS words: up to K4 one a lane (word `lane`);
// at K5 (40 words) lanes 0..7 hold a second, word 32+lane, in hi.
template<int BITS> struct TileW { uint32_t lo = 0, hi = 0; };
template<int BITS>
__device__ __forceinline__ TileW<BITS> load_tile(const uint32_t* tile, int lane) {
    static_assert(BITS >= 2 && BITS <= 5, "EXL3 widths 2..5");
    TileW<BITS> w;
    constexpr int W = 8 * BITS;
    w.lo = lane < (W < 32 ? W : 32) ? __ldcs(tile + lane) : 0u;
    if constexpr (W > 32) w.hi = lane < W - 32 ? __ldcs(tile + 32 + lane) : 0u;
    return w;
}
// Word `idx` of the tile from whichever lane holds it (every lane takes part in both shuffles).
template<int BITS>
__device__ __forceinline__ uint32_t tile_word(const TileW<BITS>& w, int idx) {
    uint32_t a = __shfl_sync(0xffffffff, w.lo, idx & 31);
    if constexpr (8 * BITS <= 32) return a;
    else { uint32_t b = __shfl_sync(0xffffffff, w.hi, idx & 31); return idx < 32 ? a : b; }
}
// A lane's eight weights from a tile held as TileW: up to K4 exactly dqk (one 64-bit window of two words); at K5 the
// eight windows span more than two words, so as exllamav3's dq_dispatch<5> (two dq4 of four windows, each within two
// words): w0..w3 at shifts s2+3*BITS .. s2 of the 64-bit pair, decoded as (w0,w1),(w2,w3).
template<int BITS>
__device__ __forceinline__ void dqk(const TileW<BITS>& w, Pair& f0, Pair& f1, int lane) {
    if constexpr (8 * BITS <= 32) { dqk<BITS>(w.lo, f0, f1, lane); return; }
    else {
        constexpr int W = 8 * BITS;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int b0 = (lane * 8 + 4 * h + 257) * BITS - 16, b2 = b0 + 3 * BITS + 16;
            const int i0 = b0 / 32, i2 = (b2 - 1) / 32, s2 = (i2 + 1) * 32 - b2;
            const uint32_t a = tile_word<BITS>(w, i0 % W), b = tile_word<BITS>(w, i2 % W);
            const uint32_t x = uint32_t(((uint64_t(a) << 32) | b) >> s2);
            Pair &f = h ? f1 : f0;
            f.v[0] = decode_pair((x >> (3 * BITS)) & 65535, (x >> (2 * BITS)) & 65535);
            f.v[1] = decode_pair((x >> BITS) & 65535, x & 65535);
        }
    }
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], Pair b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b.v[0]),"r"(b.v[1]));
}
__device__ __forceinline__ void had128(float& a,float& b,float& c,float& d,int lane) {
    float s0=a+b,d0=a-b,s1=c+d,d1=c-d;
    a=s0+s1; b=d0+d1; c=s0-s1; d=d0-d1;
    #pragma unroll
    for(int mask=1;mask<32;mask*=2) {
        float pa=__shfl_xor_sync(0xffffffff,a,mask),pb=__shfl_xor_sync(0xffffffff,b,mask);
        float pc=__shfl_xor_sync(0xffffffff,c,mask),pd=__shfl_xor_sync(0xffffffff,d,mask);
        float sign=lane&mask?-1.f:1.f;
        a=sign*a+pa; b=sign*b+pb; c=sign*c+pc; d=sign*d+pd;
    }
    a*=0.088388347648f;b*=0.088388347648f;c*=0.088388347648f;d*=0.088388347648f;
}
__device__ __forceinline__ void wait_ready(const DeviceExpert& e) {
    if(!e.ready)return;
    uint32_t value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(e.ready):"memory");
        if(value!=e.generation)__nanosleep(64);
    }while(value!=e.generation);
}
__device__ __forceinline__ const uint32_t* weights(const DeviceExpert& e,int p) {
    return reinterpret_cast<const uint32_t*>(p==0?e.gate:p==1?e.up:e.down);
}
__device__ __forceinline__ const half* scales(const DeviceExpert& e,int p) {
    return reinterpret_cast<const half*>(p==0?e.gate_scale:p==1?e.up_scale:e.down_scale);
}
__device__ __forceinline__ void rotate_input(const __nv_bfloat16* x,const half* su,half* out,int lane) {
    int c=lane*4;
    float a=__bfloat162float(x[c])*__half2float(su[c]);
    float b=__bfloat162float(x[c+1])*__half2float(su[c+1]);
    float d=__bfloat162float(x[c+2])*__half2float(su[c+2]);
    float e=__bfloat162float(x[c+3])*__half2float(su[c+3]);
    had128(a,b,d,e,lane);
    reinterpret_cast<uint2*>(out)[lane]=make_uint2(pair(a,b),pair(d,e));
}
__device__ __forceinline__ void rotate_output(const float* x,const half* sv,__nv_bfloat16* out,int lane) {
    float4 v=reinterpret_cast<const float4*>(x)[lane];
    float a=v.x,b=v.y,c=v.z,d=v.w;
    had128(a,b,c,d,lane);
    out[lane*4]=__float2bfloat16_rn(a*__half2float(sv[lane*4]));
    out[lane*4+1]=__float2bfloat16_rn(b*__half2float(sv[lane*4+1]));
    out[lane*4+2]=__float2bfloat16_rn(c*__half2float(sv[lane*4+2]));
    out[lane*4+3]=__float2bfloat16_rn(d*__half2float(sv[lane*4+3]));
}
// s0ma change 43: the same rotation and scale, the FP32 result kept (a head's logits; its BF16 rounding is this rounded)
__device__ __forceinline__ void rotate_output(const float* x,const half* sv,float* out,int lane) {
    float4 v=reinterpret_cast<const float4*>(x)[lane];
    float a=v.x,b=v.y,c=v.z,d=v.w;
    had128(a,b,c,d,lane);
    out[lane*4]=a*__half2float(sv[lane*4]);
    out[lane*4+1]=b*__half2float(sv[lane*4+1]);
    out[lane*4+2]=c*__half2float(sv[lane*4+2]);
    out[lane*4+3]=d*__half2float(sv[lane*4+3]);
}
}
