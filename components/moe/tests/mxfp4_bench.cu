// s0ma: native MXFP4 experts against EXL3 experts in decode (step 1 of the native-MXFP4 plan). A layer's routed
// experts as the streamed decode path issues them: gate+up, then down; 6, 13 and 18 experts of 1 or 2 rows each.
//   native   the checkpoint's own MXFP4 experts (e2m1 + E8M0 a 32), grouped_mxfp4_native_parallel: gate, up and down
//            launches as ds41d runs a native bank (block-scaled FP4 tensor-core MMA, FP8 activations)
//   K2..K5   EXL3 experts all of one width, grouped_exl3_var_gu and grouped_exl3_var as Sero's bank runs them
// For each: cold (after an L2 flush, as decode meets an expert), warm (back to back), the floor (a plain read of as many
// bytes) and the time an expert. The question it answers: is a native expert faster than a K4 or K5 one, and by how
// much (the plan assumed ~88% of the floor for native against EXL3's ~75%).
//
//   nvcc -O2 -std=c++17 -arch=sm_120a -Icomponents/exl3/include -Icomponents/moe/include -Icomponents/exl3/src \
//        components/moe/tests/mxfp4_bench.cu components/exl3/src/exl3.cu components/moe/src/mxfp4.cu \
//        -o /tmp/mxfp4_bench && /tmp/mxfp4_bench
#include <ds41/exl3.cuh>
#include <ds41/kernels.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

__global__ void flush_l2(uint4* p, size_t n, unsigned v) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        p[i] = make_uint4(v, v + 1, v + 2, v + 3);
}
__global__ void stream_read(const uint4* p, size_t n, unsigned* sink) {
    unsigned acc = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) *sink = acc;
}
static uint4* g_flush; static size_t g_flush_n; static unsigned g_flush_v = 1;
static void flush() { flush_l2<<<1024, 256>>>(g_flush, g_flush_n, g_flush_v++); CK(cudaGetLastError()); }
template <class F> static void time_it(F launch, double& cold, double& warm, int reps = 40) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch(); CK(cudaDeviceSynchronize());
    std::vector<float> each;
    for (int i = 0; i < reps; ++i) {
        flush();
        CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); each.push_back(ms);
    }
    std::sort(each.begin(), each.end());
    cold = 0; for (int i = reps / 10; i < reps - reps / 10; ++i) cold += each[i]; cold = cold * 1000.0 / (reps - 2 * (reps / 10));
    CK(cudaEventRecord(a)); for (int i = 0; i < reps; ++i) launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); warm = ms * 1000.0 / reps;
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}
static uint8_t* upload(const std::vector<uint8_t>& v) { uint8_t* p; CK(cudaMalloc(&p, v.size())); CK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice)); return p; }

int main() {
    setenv("DS41_EXL3_PF", "8", 0); setenv("DS41_EXL3_GU", "1", 0); setenv("DS41_EXL3_OCC2", "1", 0);   // run.py's defaults
    int dev = 0, sms = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    std::printf("%d SMs; DS41_EXL3_PF=%s DS41_EXL3_GU=%s DS41_EXL3_OCC2=%s\n", sms, std::getenv("DS41_EXL3_PF"),
                std::getenv("DS41_EXL3_GU"), std::getenv("DS41_EXL3_OCC2"));
    g_flush_n = (size_t(256) << 20) / 16; CK(cudaMalloc(&g_flush, g_flush_n * 16));
    uint4* rd; const size_t rd_bytes = size_t(448) << 20; CK(cudaMalloc(&rd, rd_bytes)); CK(cudaMemset(rd, 1, rd_bytes));
    unsigned* sink; CK(cudaMalloc(&sink, 4));
    std::mt19937 rng(1119);
    const int E = 18, I = 2304, H = 5120;

    // activations: 2 rows of the layer input (H) and of the swiglu output (I), as BF16 (EXL3) and E4M3 + E8M0 (native)
    std::vector<__nv_bfloat16> xh(size_t(2) * H), xi(size_t(2) * I);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = __float2bfloat16_rn(nd(rng)); for (auto& v : xi) v = __float2bfloat16_rn(nd(rng));
    __nv_bfloat16 *dxh, *dxi; CK(cudaMalloc(&dxh, xh.size() * 2)); CK(cudaMalloc(&dxi, xi.size() * 2));
    CK(cudaMemcpy(dxh, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dxi, xi.data(), xi.size() * 2, cudaMemcpyHostToDevice));
    uint8_t *qh, *sh, *qi, *si; CK(cudaMalloc(&qh, 2 * H)); CK(cudaMalloc(&sh, 2 * H / 32)); CK(cudaMalloc(&qi, 2 * I)); CK(cudaMalloc(&si, 2 * I / 32));
    CK(ds41::quantize_mxfp8(dxh, qh, sh, 2, H, 0)); CK(ds41::quantize_mxfp8(dxi, qi, si, 2, I, 0));
    __nv_bfloat16 *og, *ou, *od; CK(cudaMalloc(&og, size_t(E) * 2 * I * 2)); CK(cudaMalloc(&ou, size_t(E) * 2 * I * 2)); CK(cudaMalloc(&od, size_t(E) * 2 * H * 2));

    // native experts: gate/up [I][H/2] + [I][H/32], down [H][I/2] + [H][I/32]
    std::uniform_int_distribution<int> byte(0, 255), e8(120, 130);
    auto bytes = [&](size_t n, bool scale) { std::vector<uint8_t> v(n); for (auto& b : v) b = uint8_t(scale ? e8(rng) : byte(rng)); return v; };
    std::vector<ds41::DeviceExpert> ntab(E);
    for (auto& e : ntab) {
        e.gate = upload(bytes(size_t(I) * H / 2, false)); e.gate_scale = upload(bytes(size_t(I) * H / 32, true));
        e.up = upload(bytes(size_t(I) * H / 2, false)); e.up_scale = upload(bytes(size_t(I) * H / 32, true));
        e.down = upload(bytes(size_t(H) * I / 2, false)); e.down_scale = upload(bytes(size_t(H) * I / 32, true));
    }
    const size_t n_gu = 2 * (size_t(I) * H / 2 + size_t(I) * H / 32), n_down = size_t(H) * I / 2 + size_t(H) * I / 32;
    ds41::DeviceExpert* dntab; CK(cudaMalloc(&dntab, E * sizeof(ntab[0]))); CK(cudaMemcpy(dntab, ntab.data(), E * sizeof(ntab[0]), cudaMemcpyHostToDevice));

    // EXL3 experts of each width: trellis words k/16 * n/16 * 8 * bits, scales k + n halves
    auto trellis = [&](int n, int k, int bits, size_t& b) {
        const size_t words = size_t(k / 16) * (n / 16) * 8 * bits;
        std::vector<uint8_t> tr(words * 4); for (auto& x : tr) x = uint8_t(byte(rng));
        std::vector<half> sc(k + n); for (auto& v : sc) v = __float2half((byte(rng) & 1 ? 1.f : -1.f) * 0.05f);
        uint8_t* s; CK(cudaMalloc(&s, sc.size() * 2)); CK(cudaMemcpy(s, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
        b = words * 4 + sc.size() * 2;
        return std::make_pair((const uint8_t*)upload(tr), (const uint8_t*)s);
    };
    struct Bank { int bits; ds41::DeviceExpert* table; uint8_t* widths; size_t gu, down; };
    std::vector<Bank> banks;
    for (int bits = 2; bits <= 5; ++bits) {
        std::vector<ds41::DeviceExpert> t(E); size_t gu = 0, dn = 0;
        for (auto& e : t) {
            size_t b1, b2, b3;
            auto g = trellis(I, H, bits, b1), u = trellis(I, H, bits, b2), d = trellis(H, I, bits, b3);
            e.gate = g.first; e.gate_scale = g.second; e.up = u.first; e.up_scale = u.second; e.down = d.first; e.down_scale = d.second;
            gu = b1 + b2; dn = b3;
        }
        Bank bk{bits, nullptr, nullptr, gu, dn};
        CK(cudaMalloc(&bk.table, E * sizeof(t[0]))); CK(cudaMemcpy(bk.table, t.data(), E * sizeof(t[0]), cudaMemcpyHostToDevice));
        bk.widths = upload(std::vector<uint8_t>(E, uint8_t(bits)));
        banks.push_back(bk);
    }
    std::printf("\nper expert: native %.2f MB (gate+up %.2f, down %.2f)", (n_gu + n_down) / 1e6, n_gu / 1e6, n_down / 1e6);
    for (auto& b : banks) std::printf("; K%d %.2f MB", b.bits, (b.gu + b.down) / 1e6);
    std::printf("\n\n%-30s %4s %8s %9s %9s %9s %9s %10s\n", "launch", "jobs", "MB", "cold us", "floor us", "warm us",
                "of floor", "us/expert");
    for (int count : {6, 13, 18}) for (int rows : {1, 2}) {
        std::vector<ds41::GemmJob> ng(count), nu(count), ndn(count), xg(count), xu(count), xd(count);
        for (int e = 0; e < count; ++e) {
            ng[e] = {ds41::ExpertId(e), rows, qh, sh, og + size_t(e) * 2 * I};
            nu[e] = {ds41::ExpertId(e), rows, qh, sh, ou + size_t(e) * 2 * I};
            ndn[e] = {ds41::ExpertId(e), rows, qi, si, od + size_t(e) * 2 * H};
            xg[e] = {ds41::ExpertId(e), rows, reinterpret_cast<const uint8_t*>(dxh), nullptr, og + size_t(e) * 2 * I};
            xu[e] = {ds41::ExpertId(e), rows, reinterpret_cast<const uint8_t*>(dxh), nullptr, ou + size_t(e) * 2 * I};
            xd[e] = {ds41::ExpertId(e), rows, reinterpret_cast<const uint8_t*>(dxi), nullptr, od + size_t(e) * 2 * H};
        }
        auto dev = [&](const std::vector<ds41::GemmJob>& j) { ds41::GemmJob* p; CK(cudaMalloc(&p, j.size() * sizeof(j[0])));
            CK(cudaMemcpy(p, j.data(), j.size() * sizeof(j[0]), cudaMemcpyHostToDevice)); return p; };
        ds41::GemmJob *dng = dev(ng), *dnu = dev(nu), *dnd = dev(ndn), *dxg = dev(xg), *dxu = dev(xu), *dxd = dev(xd);
        auto row = [&](const char* what, size_t per_expert, auto launch) {
            const size_t total = per_expert * count;
            double cold, warm, fc, fw;
            time_it(launch, cold, warm);
            time_it([&] { stream_read<<<sms * 4, 256>>>(rd, std::min(total, rd_bytes) / 16, sink); CK(cudaGetLastError()); }, fc, fw);
            char name[64]; std::snprintf(name, sizeof name, "%s, %d row%s", what, rows, rows > 1 ? "s" : "");
            std::printf("%-30s %4d %8.2f %9.1f %9.1f %9.1f %8.0f%% %10.2f\n", name, count, total / 1e6, cold, fc, warm,
                        100.0 * fc / cold, cold / count);
        };
        row("native gate+up", n_gu, [&] {
            CK(ds41::grouped_mxfp4_native_parallel(dntab, dng, count, ds41::Projection::Gate, I, H, 0));
            CK(ds41::grouped_mxfp4_native_parallel(dntab, dnu, count, ds41::Projection::Up, I, H, 0)); });
        for (auto& b : banks) {
            char what[32]; std::snprintf(what, sizeof what, "K%d gate+up", b.bits);
            row(what, b.gu, [&] { CK(ds41::grouped_exl3_var_gu(b.table, dxg, dxu, count, I, H, b.widths, 0, rows)); });
        }
        row("native down", n_down, [&] { CK(ds41::grouped_mxfp4_native_parallel(dntab, dnd, count, ds41::Projection::Down, H, I, 0)); });
        for (auto& b : banks) {
            char what[32]; std::snprintf(what, sizeof what, "K%d down", b.bits);
            row(what, b.down, [&] { CK(ds41::grouped_exl3_var(b.table, dxd, count, ds41::Projection::Down, H, I, b.widths, 0, rows)); });
        }
        std::printf("\n");
        cudaFree(dng); cudaFree(dnu); cudaFree(dnd); cudaFree(dxg); cudaFree(dxu); cudaFree(dxd);
    }
    std::printf("done\n");
    return 0;
}
