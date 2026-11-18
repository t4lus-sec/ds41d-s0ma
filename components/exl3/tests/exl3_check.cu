// s0ma change 32 (N4): a kernel launched behind a programmatic graph edge (DS41_PDL=1) starts before its
// predecessor has finished and waits here until that predecessor's writes are visible; otherwise a no-op.
#ifndef DS41_PDL_WAIT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define DS41_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#else
#define DS41_PDL_WAIT() do {} while (0)
#endif
#endif
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
#include <cstring>
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
__global__ void reference_weights(const uint32_t* trellis, float* W, int K, int N, int bits) {DS41_PDL_WAIT();
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

// s0ma change 37: the prefill route's GEMM, done plainly here (the engine uses cuBLAS): C[m][n] = sum_k A[m][k] W[k][n]
__global__ void plain_gemm(const half* A, const half* W, float* C, int M, int N, int K) {DS41_PDL_WAIT();
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= N) return;
    float acc = 0;
    for (int k = 0; k < K; ++k) acc += __half2float(A[size_t(m) * K + k]) * __half2float(W[size_t(k) * N + n]);
    C[size_t(m) * N + n] = acc;
}
// a split launch with its own scratch (partials, zeroed counts), twice: the counts must come back to zero
static void dense_split(const ds41::DeviceExpert* t, ds41::GemmJob j, int N, int K, int bits, int xs, int os, int groups,
                        int gx, int go, bool* invariant) {
    const size_t pf = ds41::exl3_dense_splitk_floats(N, K, groups), slots = size_t(groups) * (N / 128);
    float* partial = nullptr; unsigned* counts = nullptr;
    if (pf) { CK(cudaMalloc(&partial, pf * 4)); CK(cudaMalloc(&counts, slots * 4)); CK(cudaMemset(counts, 0, slots * 4)); }
    for (int rep = 0; rep < 2; ++rep) {
        CK(ds41::exl3_dense(t, j, N, K, bits, 0, xs, os, groups, gx, go, partial, pf, counts, slots));
        CK(cudaDeviceSynchronize());
    }
    if (counts) {
        std::vector<unsigned> h(slots);
        CK(cudaMemcpy(h.data(), counts, slots * 4, cudaMemcpyDeviceToHost));
        for (unsigned v : h) if (v) { *invariant = false; break; }
        cudaFree(partial); cudaFree(counts);
    }
}
// s0ma change 43: the FP32-output decode (Sero's K4 head): its values rounded to BF16 are the BF16 launch's bit for bit,
// and row 0 alone is row 0 among the others, bit for bit
static bool f32_case(int bits, int K, int N, int M, std::mt19937& rng) {
    const size_t words = size_t(K / 16) * (N / 16) * 8 * bits;
    std::vector<uint32_t> tr(words);
    for (auto& w : tr) w = rng();
    std::vector<half> sc(K + N);
    std::uniform_real_distribution<float> mag(0.5f, 1.5f);
    for (auto& v : sc) v = __float2half((rng() & 1 ? 1.f : -1.f) * mag(rng) * 0.05f);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> x(size_t(M) * K);
    for (auto& v : x) v = __float2bfloat16(nd(rng));
    uint32_t* dtr; half* dsc; __nv_bfloat16 *dx, *db; float *df, *d1;
    CK(cudaMalloc(&dtr, words * 4)); CK(cudaMalloc(&dsc, (K + N) * 2)); CK(cudaMalloc(&dx, size_t(M) * K * 2));
    CK(cudaMalloc(&db, size_t(M) * N * 2)); CK(cudaMalloc(&df, size_t(M) * N * 4)); CK(cudaMalloc(&d1, size_t(N) * 4));
    CK(cudaMemcpy(dtr, tr.data(), words * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc, sc.data(), (K + N) * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), size_t(M) * K * 2, cudaMemcpyHostToDevice));
    ds41::DeviceExpert e{};
    e.gate = e.up = e.down = reinterpret_cast<const uint8_t*>(dtr);
    e.gate_scale = e.up_scale = e.down_scale = reinterpret_cast<const uint8_t*>(dsc);
    ds41::DeviceExpert* dt;
    CK(cudaMalloc(&dt, sizeof e)); CK(cudaMemcpy(dt, &e, sizeof e, cudaMemcpyHostToDevice));
    const uint8_t* xin = reinterpret_cast<const uint8_t*>(dx);
    CK(ds41::exl3_dense(dt, ds41::GemmJob{0, M, xin, nullptr, db}, N, K, bits, 0));     // no scratch: the unsplit launch
    CK(ds41::exl3_dense_f32(dt, ds41::GemmJob{0, M, xin, nullptr, reinterpret_cast<__nv_bfloat16*>(df)}, N, K, bits, 0));
    CK(ds41::exl3_dense_f32(dt, ds41::GemmJob{0, 1, xin, nullptr, reinterpret_cast<__nv_bfloat16*>(d1)}, N, K, bits, 0));
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> hb(size_t(M) * N); std::vector<float> hf(size_t(M) * N), h1(N);
    CK(cudaMemcpy(hb.data(), db, hb.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hf.data(), df, hf.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h1.data(), d1, h1.size() * 4, cudaMemcpyDeviceToHost));
    size_t round = 0, row = 0;
    for (size_t i = 0; i < hb.size(); ++i)
        round += __bfloat16_as_ushort(__float2bfloat16_rn(hf[i])) != __bfloat16_as_ushort(hb[i]);
    for (int n = 0; n < N; ++n) row += std::memcmp(&h1[n], &hf[n], 4) != 0;
    cudaFree(dtr); cudaFree(dsc); cudaFree(dx); cudaFree(db); cudaFree(df); cudaFree(d1); cudaFree(dt);
    const bool ok = !round && !row;
    std::printf("%s K%d dense f32   k=%d n=%d rows=%d: %zu of %zu values round differently from the BF16 launch; row 0 alone "
                "differs in %zu\n", ok ? "ok  " : "FAIL", bits, K, N, M, round, hb.size(), row);
    return ok;
}
// s0ma change 122: an FNV-1a digest of every output, printed at the end: runs with DS41_EXL3_PF=4 and =8 must print the
// same one (the deeper prefetch changes when weights are loaded, never what is computed)
static uint64_t g_digest = 1469598103934665603ull;
static void digest(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { g_digest ^= b[i]; g_digest *= 1099511628211ull; }
}
struct Case { int bits, K, N, tokens; bool prefill; ds41::Projection p; bool var = false; bool dense = false; int stride = 1; bool dq = false; bool vsplit = false; };

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
    if (c.dense && c.stride > 1) {   // s0ma change 34: rows a slice of wider ones (wo_a's groups), in and out
        const int S = c.stride;
        __nv_bfloat16 *wx, *wout;
        CK(cudaMalloc(&wx, size_t(M) * K * S * 2)); CK(cudaMalloc(&wout, size_t(M) * N * S * 2));
        CK(cudaMemset(wx, 0x7f, size_t(M) * K * S * 2));
        CK(cudaMemcpy2D(wx + K, size_t(K) * S * 2, dx, size_t(K) * 2, size_t(K) * 2, M, cudaMemcpyDeviceToDevice));
        ds41::GemmJob sj{0, M, reinterpret_cast<const uint8_t*>(wx + K), nullptr, wout + N};
        if (c.prefill) CK(ds41::exl3_dense_prefill(dt, sj, N, K, bits, 0, K * S, N * S));
        else {   // two tensors at consecutive table entries in one launch; group 1 is the one checked
            ds41::DeviceExpert two[2] = {e, e}, *dt2;
            CK(cudaMalloc(&dt2, sizeof two)); CK(cudaMemcpy(dt2, two, sizeof two, cudaMemcpyHostToDevice));
            ds41::GemmJob gj{0, M, reinterpret_cast<const uint8_t*>(wx), nullptr, wout};
            dense_split(dt2, gj, N, K, bits, K * S, N * S, 2, K, N, invariant);   // s0ma change 37: split when narrow
            CK(cudaDeviceSynchronize()); cudaFree(dt2);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy2D(dout, size_t(N) * 2, wout + N, size_t(N) * S * 2, size_t(N) * 2, M, cudaMemcpyDeviceToDevice));
        cudaFree(wx); cudaFree(wout);
    }
    else if (c.vsplit) {   // s0ma change 40: the experts' grouped launch split over K (twice: counts back to zero)
        const size_t pf = ds41::exl3_var_splitk_floats(N, K, 1), slots = N / 128;
        float* partial; unsigned* counts;
        CK(cudaMalloc(&partial, (pf ? pf : 1) * 4)); CK(cudaMalloc(&counts, slots * 4)); CK(cudaMemset(counts, 0, slots * 4));
        uint8_t* dbits; const uint8_t b8 = uint8_t(bits);
        CK(cudaMalloc(&dbits, 1)); CK(cudaMemcpy(dbits, &b8, 1, cudaMemcpyHostToDevice));
        for (int rep = 0; rep < 2; ++rep) {
            CK(ds41::grouped_exl3_splitk(dt, dj, 1, c.p, N, K, rep ? nullptr : dbits, bits, partial, pf, counts, slots, 0));
            CK(cudaDeviceSynchronize());
        }
        std::vector<unsigned> h(slots);
        CK(cudaMemcpy(h.data(), counts, slots * 4, cudaMemcpyDeviceToHost));
        for (unsigned v : h) if (v && invariant) *invariant = false;
        cudaFree(partial); cudaFree(counts); cudaFree(dbits);
    }
    else if (c.dq) {   // s0ma change 37: prefill as decoded weights, rotated rows, a GEMM, rotated output
        half *W, *A; float* C;
        CK(cudaMalloc(&W, size_t(K) * N * 2)); CK(cudaMalloc(&A, size_t(M) * K * 2)); CK(cudaMalloc(&C, size_t(M) * N * 4));
        CK(ds41::exl3_dense_reconstruct(dt, 0, N, K, bits, W, 0));
        CK(ds41::exl3_dense_rotate_in(dt, 0, dx, M, K, 0, A, 0));
        plain_gemm<<<dim3((N + 127) / 128, M), 128>>>(A, W, C, M, N, K);
        CK(cudaGetLastError());
        CK(ds41::exl3_dense_rotate_out(dt, 0, C, M, N, K, dout, 0, 0));
        CK(cudaDeviceSynchronize());
        cudaFree(W); cudaFree(A); cudaFree(C);
    }
    else if (c.dense) {   // s0ma change 34: the by-value dense launches (gate views), at an explicit width
        CK(ds41::exl3_set_bits(bits == 2 ? 3 : 2));   // the global width must not matter
        if (c.prefill) CK(ds41::exl3_dense_prefill(dt, job, N, K, bits, 0));
        else {
            dense_split(dt, job, N, K, bits, 0, 0, 1, 0, 0, invariant);   // s0ma change 37: split when narrow
            // s0ma change 127: the weights and scales brought into L2 (both policies; a 15-byte tail skipped) on the
            // same stream, then the launch again: the same bits
            std::vector<uint16_t> before(size_t(M) * N), after(before.size());
            CK(cudaMemcpy(before.data(), dout, before.size() * 2, cudaMemcpyDeviceToHost));
            ds41::L2Range r[2] = {{dtr, words * 4}, {dsc, uint64_t(K + N) * 2 + 15}};
            ds41::L2Range* dr;
            CK(cudaMalloc(&dr, sizeof r)); CK(cudaMemcpy(dr, r, sizeof r, cudaMemcpyHostToDevice));
            for (int policy = 1; policy <= 2; ++policy) {
                r[1].bytes = uint64_t(K + N) * 2 - (policy == 2 ? 15 : 0);
                CK(cudaMemcpy(dr, r, sizeof r, cudaMemcpyHostToDevice));
                CK(ds41::exl3_l2_prefetch(dr, 2, policy, 8, 0));
                CK(cudaMemset(dout, 0, before.size() * 2));
                dense_split(dt, job, N, K, bits, 0, 0, 1, 0, 0, invariant);
                CK(cudaMemcpy(after.data(), dout, after.size() * 2, cudaMemcpyDeviceToHost));
                if (after != before) { std::printf("  differs after an L2 prefetch (policy %d)\n", policy); *invariant = false; }
            }
            cudaFree(dr);
        }
        CK(ds41::exl3_set_bits(bits));
    }
    else if (c.prefill) {
        CK(ds41::grouped_exl3_prefill(dt, dj, 1, c.p, N, K, M, 0));
        // s0ma change 68: the rows rotated once, then the pipelined kernel: the same bits (a1, the second output)
        half *ra0, *ra1; __nv_bfloat16* ro; ds41::GemmJob* drj;
        CK(cudaMalloc(&ra0, size_t(M) * K * 2)); CK(cudaMalloc(&ra1, size_t(M) * K * 2));
        CK(cudaMalloc(&ro, size_t(M) * N * 2)); CK(cudaMemset(ro, 0, size_t(M) * N * 2));
        ds41::GemmJob rj{0, M, reinterpret_cast<const uint8_t*>(dx), nullptr, ro};
        CK(cudaMalloc(&drj, sizeof rj)); CK(cudaMemcpy(drj, &rj, sizeof rj, cudaMemcpyHostToDevice));
        CK(ds41::exl3_prefill_rotate(dt, drj, c.p, c.p, K, M, ra0, ra1, 0));
        CK(ds41::exl3_prefill_rotated(dt, drj, c.p, N, K, M, bits, ra1, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> a(size_t(M) * N), b(size_t(M) * N);
        CK(cudaMemcpy(a.data(), dout, a.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), ro, b.size() * 2, cudaMemcpyDeviceToHost));
        if (a != b) { std::printf("  rotate-once prefill differs from grouped_exl3_prefill\n"); if (invariant) *invariant = false; }
        // s0ma change 76: two jobs in one grouped launch (all M rows, and the first half), each against the above
        {
            const int M2 = M > 1 ? M / 2 : 1;
            half *g0, *g1; __nv_bfloat16 *o0, *o1; ds41::GemmJob* dg;
            CK(cudaMalloc(&g0, size_t(M) * K * 2)); CK(cudaMalloc(&g1, size_t(M2) * K * 2));
            CK(cudaMalloc(&o0, size_t(M) * N * 2)); CK(cudaMalloc(&o1, size_t(M2) * N * 2));
            ds41::GemmJob gj[2] = {{0, M, reinterpret_cast<const uint8_t*>(dx), reinterpret_cast<const uint8_t*>(g0), o0},
                                   {0, M2, reinterpret_cast<const uint8_t*>(dx), reinterpret_cast<const uint8_t*>(g1), o1}};
            CK(cudaMalloc(&dg, sizeof gj)); CK(cudaMemcpy(dg, gj, sizeof gj, cudaMemcpyHostToDevice));
            CK(ds41::exl3_prefill_rotate_group(dt, dg, nullptr, 2, c.p, c.p, K, M, 0));
            CK(ds41::exl3_prefill_rotated_group(dt, dg, 2, c.p, N, K, M, bits, 0));
            CK(cudaDeviceSynchronize());
            std::vector<uint16_t> g(size_t(M) * N), h(size_t(M2) * N);
            CK(cudaMemcpy(g.data(), o0, g.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h.data(), o1, h.size() * 2, cudaMemcpyDeviceToHost));
            if (g != a || !std::equal(h.begin(), h.end(), a.begin())) {
                std::printf("  grouped rotate-once prefill differs\n"); if (invariant) *invariant = false; }
            cudaFree(g0); cudaFree(g1); cudaFree(o0); cudaFree(o1); cudaFree(dg);
        }
        // s0ma change 82: the expert's projection decoded once into FP16, a GEMM on the rotated rows, the rotated output:
        // the same products as prefill_rot summed in another order (relative error against it, not bits)
        {
            half* W; float* C; __nv_bfloat16* eo;
            CK(cudaMalloc(&W, size_t(K) * N * 2)); CK(cudaMalloc(&C, size_t(M) * N * 4)); CK(cudaMalloc(&eo, size_t(M) * N * 2));
            // only projection c.p holds this case's weights and scales: a wrong projection index reads noise
            uint32_t* jw; half* js;
            CK(cudaMalloc(&jw, words * 4)); CK(cudaMalloc(&js, (K + N) * 2));
            std::vector<uint32_t> jt(words); for (auto& v : jt) v = rng();
            std::vector<half> jsc(K + N); for (auto& v : jsc) v = __float2half(0.3f);
            CK(cudaMemcpy(jw, jt.data(), words * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(js, jsc.data(), (K + N) * 2, cudaMemcpyHostToDevice));
            ds41::DeviceExpert ej{};
            const uint8_t *jwb = reinterpret_cast<const uint8_t*>(jw), *jsb = reinterpret_cast<const uint8_t*>(js);
            ej.gate = c.p == ds41::Projection::Gate ? w : jwb; ej.gate_scale = c.p == ds41::Projection::Gate ? s : jsb;
            ej.up = c.p == ds41::Projection::Up ? w : jwb; ej.up_scale = c.p == ds41::Projection::Up ? s : jsb;
            ej.down = c.p == ds41::Projection::Down ? w : jwb; ej.down_scale = c.p == ds41::Projection::Down ? s : jsb;
            ds41::DeviceExpert* dt2;
            CK(cudaMalloc(&dt2, sizeof ej)); CK(cudaMemcpy(dt2, &ej, sizeof ej, cudaMemcpyHostToDevice));
            CK(ds41::exl3_expert_reconstruct(dt2, ds41::ExpertId(0), c.p, N, K, bits, W, 0));
            plain_gemm<<<dim3((N + 127) / 128, M), 128>>>(ra1, W, C, M, N, K);
            CK(cudaGetLastError());
            CK(ds41::exl3_expert_rotate_out(dt2, ds41::ExpertId(0), c.p, C, M, N, K, eo, 0, 0));
            CK(cudaDeviceSynchronize());
            cudaFree(jw); cudaFree(js); cudaFree(dt2);
            CK(cudaDeviceSynchronize());
            std::vector<__nv_bfloat16> g(size_t(M) * N), r(size_t(M) * N);
            CK(cudaMemcpy(g.data(), eo, g.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(r.data(), ro, r.size() * 2, cudaMemcpyDeviceToHost));
            double e2 = 0, r2 = 0;
            for (size_t i = 0; i < g.size(); ++i) {
                const double a_ = __bfloat162float(g[i]), b_ = __bfloat162float(r[i]);
                e2 += (a_ - b_) * (a_ - b_); r2 += b_ * b_;
            }
            const double rel = std::sqrt(e2 / (r2 > 0 ? r2 : 1));
            if (!(rel < 2e-3)) { std::printf("  expert GEMM prefill differs from prefill_rot: %.2e\n", rel); if (invariant) *invariant = false; }
            cudaFree(W); cudaFree(C); cudaFree(eo);
        }
        cudaFree(ra0); cudaFree(ra1); cudaFree(ro); cudaFree(drj);
    }
    else if (c.var) {   // s0ma change 15: the mixed-width launch, its expert's width from a device table
        uint8_t* dbits; const uint8_t b8 = uint8_t(bits);
        CK(cudaMalloc(&dbits, 1)); CK(cudaMemcpy(dbits, &b8, 1, cudaMemcpyHostToDevice));
        CK(ds41::exl3_set_bits(bits == 2 ? 3 : 2));   // the global width must not matter
        CK(ds41::grouped_exl3_var(dt, dj, 1, c.p, N, K, dbits, 0));
        CK(cudaDeviceSynchronize());
        // s0ma change 122: gate and up in one launch give grouped_exl3_var's bits (here both read the same weights)
        if (K == 5120 && c.p != ds41::Projection::Down) {
            __nv_bfloat16 *og, *ou; ds41::GemmJob *dg, *du;
            CK(cudaMalloc(&og, size_t(M) * N * 2)); CK(cudaMalloc(&ou, size_t(M) * N * 2));
            ds41::GemmJob gj{0, M, reinterpret_cast<const uint8_t*>(dx), nullptr, og}, uj{0, M, reinterpret_cast<const uint8_t*>(dx), nullptr, ou};
            CK(cudaMalloc(&dg, sizeof gj)); CK(cudaMalloc(&du, sizeof uj));
            CK(cudaMemcpy(dg, &gj, sizeof gj, cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, &uj, sizeof uj, cudaMemcpyHostToDevice));
            CK(ds41::grouped_exl3_var_gu(dt, dg, du, 1, N, K, dbits, 0));
            CK(cudaDeviceSynchronize());
            std::vector<uint16_t> a(size_t(M) * N), g(a.size()), u(a.size());
            CK(cudaMemcpy(a.data(), dout, a.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(g.data(), og, g.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(u.data(), ou, u.size() * 2, cudaMemcpyDeviceToHost));
            if (g != a || u != a) { std::printf("  gate+up launch differs from grouped_exl3_var\n"); if (invariant) *invariant = false; }
            cudaFree(og); cudaFree(ou); cudaFree(dg); cudaFree(du);
        }
        cudaFree(dbits);
        CK(ds41::exl3_set_bits(bits));
    }
    else CK(ds41::grouped_exl3(dt, dj, 1, c.p, N, K, 0));
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> out(size_t(M) * N);
    CK(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost));
    digest(out.data(), out.size() * 2);

    // row 0 alone must give the same bits, on the same path (s0ma change 40: a split launch sums K in its own order,
    // so it is compared with itself; a strided group launch may split differently from one tensor: not compared)
    if (invariant && !c.prefill && M > 1 && !(c.dense && c.stride > 1)) {
        ds41::GemmJob one{0, 1, reinterpret_cast<const uint8_t*>(dx), nullptr, dout};
        CK(cudaMemcpy(dj, &one, sizeof one, cudaMemcpyHostToDevice));
        if (c.dense) dense_split(dt, one, N, K, bits, 0, 0, 1, 0, 0, invariant);
        else if (c.vsplit) {
            const size_t pf = ds41::exl3_var_splitk_floats(N, K, 1), slots = N / 128;
            float* partial; unsigned* counts;
            CK(cudaMalloc(&partial, (pf ? pf : 1) * 4)); CK(cudaMalloc(&counts, slots * 4)); CK(cudaMemset(counts, 0, slots * 4));
            CK(ds41::grouped_exl3_splitk(dt, dj, 1, c.p, N, K, nullptr, bits, partial, pf, counts, slots, 0));
            CK(cudaDeviceSynchronize()); cudaFree(partial); cudaFree(counts);
        }
        else CK(ds41::grouped_exl3(dt, dj, 1, c.p, N, K, 0));
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
        // s0ma change 15: K4 and K5 (0xSero's 2.7 bpw bank), decode and prefill, and the mixed-width launch
        {4, 5120, 2304, 1, false, ds41::Projection::Gate}, {4, 2304, 5120, 6, false, ds41::Projection::Down},
        {4, 5120, 2304, 70, true, ds41::Projection::Up},
        {5, 5120, 2304, 2, false, ds41::Projection::Gate}, {5, 2304, 5120, 7, false, ds41::Projection::Down},
        {5, 2304, 5120, 65, true, ds41::Projection::Down},
        {2, 5120, 2304, 3, false, ds41::Projection::Gate, true}, {3, 2304, 5120, 2, false, ds41::Projection::Down, true},
        {4, 5120, 2304, 5, false, ds41::Projection::Up, true}, {5, 2304, 5120, 8, false, ds41::Projection::Down, true},
        // s0ma change 34 (N1): 0xSero's dense shapes: wq_b's k=1280, wo_b's k=8192 (decode: at most 4 rows a job)
        {4, 1280, 4096, 1, false, ds41::Projection::Gate}, {4, 1280, 4096, 6, false, ds41::Projection::Gate},
        {4, 1280, 4096, 256, true, ds41::Projection::Gate},
        {4, 8192, 5120, 1, false, ds41::Projection::Gate}, {5, 8192, 5120, 4, false, ds41::Projection::Gate},
        {4, 8192, 5120, 130, true, ds41::Projection::Gate}, {4, 5120, 1280, 6, false, ds41::Projection::Gate},
        {4, 5120, 1280, 6, false, ds41::Projection::Gate, false, true}, {5, 8192, 5120, 4, false, ds41::Projection::Gate, false, true},
        {4, 1280, 4096, 3, false, ds41::Projection::Gate, false, true}, {4, 5120, 2304, 256, true, ds41::Projection::Gate, false, true},
        {5, 8192, 5120, 97, true, ds41::Projection::Gate, false, true},
        {5, 4096, 1024, 6, false, ds41::Projection::Gate, false, true, 8}, {5, 4096, 1024, 200, true, ds41::Projection::Gate, false, true, 8},
        {5, 4096, 1024, 1, false, ds41::Projection::Gate, false, true},
        // s0ma change 37: 0xSero's narrow tensors split over K (wkv, wq_a, shared w1 and w2), and the prefill route
        {4, 5120, 512, 6, false, ds41::Projection::Gate, false, true}, {4, 5120, 1280, 1, false, ds41::Projection::Gate, false, true},
        {4, 5120, 2304, 8, false, ds41::Projection::Gate, false, true}, {4, 2304, 5120, 3, false, ds41::Projection::Gate, false, true},
        {3, 5120, 512, 2, false, ds41::Projection::Gate, false, true},
        {4, 5120, 1280, 256, true, ds41::Projection::Gate, false, false, 1, true}, {5, 4096, 1024, 100, true, ds41::Projection::Gate, false, false, 1, true},
        {4, 8192, 5120, 33, true, ds41::Projection::Gate, false, false, 1, true}, {2, 2304, 5120, 9, true, ds41::Projection::Gate, false, false, 1, true},
        // s0ma change 40: the routed experts' grouped launch split over K, each width, gate/up and down
        {2, 5120, 2304, 3, false, ds41::Projection::Gate, false, false, 1, false, true},
        {3, 5120, 2304, 6, false, ds41::Projection::Up, false, false, 1, false, true},
        {4, 2304, 5120, 2, false, ds41::Projection::Down, false, false, 1, false, true},
        {5, 2304, 5120, 8, false, ds41::Projection::Down, false, false, 1, false, true},
    };
    int fail = 0;
    for (const auto& c : cases) {
        bool inv = true;
        double e = run_case(c, rng, &inv);
        bool ok = e < 1e-2 && inv;
        fail += !ok;
        std::printf("%s K%d %s k=%d n=%d rows=%d: relative error %.2e%s\n", ok ? "ok  " : "FAIL", c.bits,
                    c.vsplit ? "experts split-K" : c.dq ? "dense dequant+gemm" : c.dense ? (c.prefill ? "dense prefill" : "dense decode ") : c.prefill ? "prefill" : c.var ? "mixed  " : "decode ", c.K, c.N, c.tokens, e, inv ? "" : "; row 0 differs by row count");
    }
    // s0ma change 43: Sero's K4 head (k 5120, n 129280) in FP32, at a smaller n; K5 and the k = 8192 four-row case too
    fail += !f32_case(4, 5120, 2048, 6, rng);
    fail += !f32_case(4, 5120, 1280, 1, rng);
    fail += !f32_case(5, 8192, 1024, 4, rng);
    std::printf("output digest %016llx (DS41_EXL3_PF=%s, DS41_EXL3_DENSE_PF=%s)\n", (unsigned long long)g_digest,
                std::getenv("DS41_EXL3_PF") ? std::getenv("DS41_EXL3_PF") : "4",
                std::getenv("DS41_EXL3_DENSE_PF") ? std::getenv("DS41_EXL3_DENSE_PF") : "4");
    std::printf(fail ? "%d FAILED\n" : "all passed\n", fail);
    return fail ? 1 : 0;
}
