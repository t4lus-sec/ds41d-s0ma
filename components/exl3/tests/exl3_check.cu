// EXL3 K2/K3 grouped kernels against a plain reference, on random trellis words, scales and inputs.
//
//   nvcc -O2 -std=c++17 -arch=sm_120a -Icomponents/exl3/include -Icomponents/moe/include -Icomponents/exl3/src \
//        components/exl3/tests/exl3_check.cu components/exl3/src/exl3.cu -o /tmp/exl3_check && /tmp/exl3_check
//
// The reference decodes every weight on its own from the tile's bit stream (16-bit windows ending at bit
// (t + 257) * bits, most significant bit first, as exllamav3's dq<bits>), with the weight's place in the tile
// given by the m16n8k16 B fragment, and multiplies in FP32: Hadamard-128 rotations, suh/svh scales, FP16 input
// rounding, BF16 output rounding, as the kernel's documented math. A wrong decode or tile order gives errors of
// order 1; the kernels' own error is BF16 rounding. Also checked: a row's result is bit-identical whatever the
// row count (decode), and the 3-bit path is what it was.
#include <ds41/exl3.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

__device__ float decode_weight(const uint32_t* tile, int t, int bits) {
    const int W = 8 * bits, e = (t + 257) * bits;
    uint32_t v = 0;
    for (int i = 0; i < 16; ++i) {
        int p = e - 16 + i;
        uint32_t word = tile[(p / 32) % W];
        v = (v << 1) | ((word >> (31 - p % 32)) & 1u);
    }
    uint32_t x = v * 0x83DCD12Du;
    uint32_t s = __dp4a(x, 0x01010101u, 0x6400u);
    half f = __ushort_as_half((unsigned short)(s & 0xffff));
    half r = __hfma(f, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
    return __half2float(r);
}

// W[k][n] from the trellis [k/16][n/16][8*bits words]
__global__ void reference_weights(const uint32_t* trellis, float* W, int K, int N, int bits) {
    int k = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int kt = k / 16, nt = n / 16, kk = k % 16, nn = n % 16;
    int f = nn / 8, lane = (nn % 8) * 4 + (kk % 8) / 2, jj = (kk % 2) + (kk >= 8 ? 2 : 0);
    int t = lane * 8 + f * 4 + jj;
    const uint32_t* tile = trellis + (size_t(kt) * (N / 16) + nt) * (8 * bits);
    W[size_t(k) * N + n] = decode_weight(tile, t, bits);
}

static void hadamard128(float* v) {           // Sylvester order, scaled by 1/sqrt(128)
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) { float a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
    for (int i = 0; i < 128; ++i) v[i] *= 0.088388347648f;
}

struct Case { int bits, K, N, tokens; bool prefill; ds41::Projection p; };

static double run_case(const Case& c, std::mt19937& rng, bool* invariant) {
    const int K = c.K, N = c.N, M = c.tokens, bits = c.bits;
    const size_t words = size_t(K / 16) * (N / 16) * 8 * bits;
    std::vector<uint32_t> tr(words);
    for (auto& w : tr) w = rng();
    std::vector<half> sc(K + N);
    std::uniform_real_distribution<float> mag(0.5f, 1.5f);
    for (auto& s : sc) s = __float2half((rng() & 1 ? 1.f : -1.f) * mag(rng) * 0.05f);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> x(size_t(M) * K);
    for (auto& v : x) v = __float2bfloat16(nd(rng));

    uint32_t* dtr; half* dsc; __nv_bfloat16 *dx, *dout; float* dW;
    CK(cudaMalloc(&dtr, words * 4)); CK(cudaMalloc(&dsc, (K + N) * 2));
    CK(cudaMalloc(&dx, size_t(M) * K * 2)); CK(cudaMalloc(&dout, size_t(M) * N * 2));
    CK(cudaMalloc(&dW, size_t(K) * N * 4));
    CK(cudaMemcpy(dtr, tr.data(), words * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc, sc.data(), (K + N) * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), size_t(M) * K * 2, cudaMemcpyHostToDevice));

    ds41::DeviceExpert e{};
    const uint8_t* w = reinterpret_cast<const uint8_t*>(dtr);
    const uint8_t* s = reinterpret_cast<const uint8_t*>(dsc);
    e.gate = e.up = e.down = w; e.gate_scale = e.up_scale = e.down_scale = s;
    ds41::DeviceExpert* dt; ds41::GemmJob* dj;
    CK(cudaMalloc(&dt, sizeof e)); CK(cudaMemcpy(dt, &e, sizeof e, cudaMemcpyHostToDevice));
    ds41::GemmJob job{0, M, reinterpret_cast<const uint8_t*>(dx), nullptr, dout};
    CK(cudaMalloc(&dj, sizeof job)); CK(cudaMemcpy(dj, &job, sizeof job, cudaMemcpyHostToDevice));
    CK(ds41::exl3_set_bits(bits));
    if (c.prefill) CK(ds41::grouped_exl3_prefill(dt, dj, 1, c.p, N, K, M, 0));
    else CK(ds41::grouped_exl3(dt, dj, 1, c.p, N, K, 0));
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> out(size_t(M) * N);
    CK(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost));

    if (invariant && !c.prefill && M > 1) {          // row 0 alone must give the same bits
        ds41::GemmJob one{0, 1, reinterpret_cast<const uint8_t*>(dx), nullptr, dout};
        CK(cudaMemcpy(dj, &one, sizeof one, cudaMemcpyHostToDevice));
        CK(ds41::grouped_exl3(dt, dj, 1, c.p, N, K, 0));
        CK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> row(N);
        CK(cudaMemcpy(row.data(), dout, N * 2, cudaMemcpyDeviceToHost));
        for (int n = 0; n < N; ++n)
            if (__bfloat16_as_ushort(row[n]) != __bfloat16_as_ushort(out[n])) { *invariant = false; break; }
    }

    reference_weights<<<dim3((N + 127) / 128, K), 128>>>(dtr, dW, K, N, bits);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> W(size_t(K) * N);
    CK(cudaMemcpy(W.data(), dW, W.size() * 4, cudaMemcpyDeviceToHost));

    double err2 = 0, ref2 = 0;
    std::vector<float> xr(K), acc(N);
    for (int m = 0; m < M; ++m) {
        for (int k0 = 0; k0 < K; k0 += 128) {
            for (int i = 0; i < 128; ++i)
                xr[k0 + i] = __bfloat162float(x[size_t(m) * K + k0 + i]) * __half2float(sc[k0 + i]);
            hadamard128(&xr[k0]);
            for (int i = 0; i < 128; ++i) xr[k0 + i] = __half2float(__float2half_rn(xr[k0 + i]));
        }
        for (int n = 0; n < N; ++n) acc[n] = 0;
        for (int k = 0; k < K; ++k) {
            const float a = xr[k], *wr = &W[size_t(k) * N];
            for (int n = 0; n < N; ++n) acc[n] += a * wr[n];
        }
        for (int n0 = 0; n0 < N; n0 += 128) {
            hadamard128(&acc[n0]);
            for (int i = 0; i < 128; ++i) {
                double r = acc[n0 + i] * __half2float(sc[K + n0 + i]);
                double g = __bfloat162float(out[size_t(m) * N + n0 + i]);
                err2 += (g - r) * (g - r); ref2 += r * r;
            }
        }
    }
    cudaFree(dtr); cudaFree(dsc); cudaFree(dx); cudaFree(dout); cudaFree(dW); cudaFree(dt); cudaFree(dj);
    return std::sqrt(err2 / (ref2 > 0 ? ref2 : 1));
}

int main() {
    std::mt19937 rng(17);
    const Case cases[] = {
        {3, 5120, 2304, 1, false, ds41::Projection::Gate}, {3, 2304, 5120, 5, false, ds41::Projection::Down},
        {3, 5120, 2304, 37, true, ds41::Projection::Up},
        {2, 5120, 2304, 1, false, ds41::Projection::Gate}, {2, 5120, 2304, 8, false, ds41::Projection::Up},
        {2, 2304, 5120, 3, false, ds41::Projection::Down}, {2, 5120, 2304, 64, true, ds41::Projection::Gate},
        {2, 2304, 5120, 200, true, ds41::Projection::Down},
    };
    int fail = 0;
    for (const auto& c : cases) {
        bool inv = true;
        double e = run_case(c, rng, &inv);
        bool ok = e < 1e-2 && inv;
        fail += !ok;
        std::printf("%s K%d %s k=%d n=%d rows=%d: relative error %.2e%s\n", ok ? "ok  " : "FAIL", c.bits,
                    c.prefill ? "prefill" : "decode ", c.K, c.N, c.tokens, e, inv ? "" : "; row 0 differs by row count");
    }
    std::printf(fail ? "%d FAILED\n" : "all passed\n", fail);
    return fail ? 1 : 0;
}
