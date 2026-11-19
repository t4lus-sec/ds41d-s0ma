// s0ma change 130: the decode step's EXL3 launches timed one by one, as the engine makes them (0xSero's dense
// attention and shared-expert tensors through exl3_dense with split-K scratch; a layer's routed experts through
// grouped_exl3_var_gu and grouped_exl3_var), on random weights, at 1..4 rows. For each:
//   cold  the launch alone after the L2 was flushed (a 256 MiB write), as decode meets these weights
//   warm  the same launch repeated back to back, its weights then in L2
//   floor a plain streaming read of as many bytes from a flushed L2 (what the memory can do for that size)
// cold near floor: memory-bound, nothing to gain; cold far above floor and warm close to cold: the kernel's own time
// (launch, waves, rotations, the split-K tail), what a kernel change can win. Times in microseconds (CUDA events).
//
//   nvcc -O2 -std=c++17 -arch=sm_120a -Icomponents/exl3/include -Icomponents/moe/include -Icomponents/exl3/src \
//        components/exl3/tests/exl3_bench.cu components/exl3/src/exl3.cu -o /tmp/exl3_bench && /tmp/exl3_bench
#include <ds41/exl3.cuh>
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
    if (acc == 0x9e3779b9u) *sink = acc;   // never in practice; keeps the loads
}

struct Tensor { uint32_t* trellis; half* scales; size_t bytes; };
static Tensor make_tensor(int n, int k, int bits, std::mt19937& rng) {
    Tensor t;
    const size_t words = size_t(k / 16) * (n / 16) * 8 * bits;
    std::vector<uint32_t> tr(words);
    for (auto& w : tr) w = rng();
    std::vector<half> sc(k + n);
    for (auto& v : sc) v = __float2half((rng() & 1 ? 1.f : -1.f) * 0.05f);
    CK(cudaMalloc(&t.trellis, words * 4)); CK(cudaMalloc(&t.scales, (k + n) * 2));
    CK(cudaMemcpy(t.trellis, tr.data(), words * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(t.scales, sc.data(), (k + n) * 2, cudaMemcpyHostToDevice));
    t.bytes = words * 4 + size_t(k + n) * 2;
    return t;
}

static uint4* g_flush; static size_t g_flush_n; static unsigned g_flush_v = 1;
static void flush() { flush_l2<<<1024, 256>>>(g_flush, g_flush_n, g_flush_v++); CK(cudaGetLastError()); }

// cold: mean over reps of one launch after a flush; warm: mean of reps launches back to back
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

int main() {
    // the engine's published settings (run.py), unless the caller set them
    setenv("DS41_EXL3_PF", "8", 0); setenv("DS41_EXL3_GU", "1", 0);
    int dev = 0, sms = 0, l2 = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    std::printf("%d SMs, L2 %.0f MiB; DS41_EXL3_PF=%s DS41_EXL3_DENSE_PF=%s DS41_EXL3_OCC2=%s DS41_EXL3_OCC2_PF=%s DS41_EXL3_OCC2_EXPERT_PF=%s\n", sms, l2 / 1048576.0,
                std::getenv("DS41_EXL3_PF"), std::getenv("DS41_EXL3_DENSE_PF") ? std::getenv("DS41_EXL3_DENSE_PF") : "4",
                std::getenv("DS41_EXL3_OCC2") ? std::getenv("DS41_EXL3_OCC2") : "0",
                std::getenv("DS41_EXL3_OCC2_PF") ? std::getenv("DS41_EXL3_OCC2_PF") : "2",
                std::getenv("DS41_EXL3_OCC2_EXPERT_PF") ? std::getenv("DS41_EXL3_OCC2_EXPERT_PF") : "4");
    g_flush_n = (size_t(256) << 20) / 16; CK(cudaMalloc(&g_flush, g_flush_n * 16));
    std::mt19937 rng(130);
    // a read buffer for the floor, and inputs/outputs wide enough for every launch
    // s0ma change 134: room for the 18-expert floor (it read 64 MiB). Change 135: on its own line; on the line
    // above it commented out the allocation, and the floor read through an unset pointer
    uint4* rd; const size_t rd_bytes = size_t(192) << 20;
    CK(cudaMalloc(&rd, rd_bytes)); CK(cudaMemset(rd, 1, rd_bytes));
    unsigned* sink; CK(cudaMalloc(&sink, 4));
    __nv_bfloat16 *x, *y; CK(cudaMalloc(&x, size_t(8) * 32768 * 2)); CK(cudaMalloc(&y, size_t(8) * 32768 * 2));
    CK(cudaMemset(x, 0, size_t(8) * 32768 * 2));
    float* partial; unsigned* counts; const size_t pf = size_t(64) << 20; CK(cudaMalloc(&partial, pf * 4));
    CK(cudaMalloc(&counts, 1 << 16)); CK(cudaMemset(counts, 0, 1 << 16));

    // 0xSero's dense tensors (n, k, bits; wo_a: 8 slices at consecutive entries)
    struct Dense { const char* name; int n, k, bits, groups; };
    const Dense dense[] = {{"wq_a 1280x5120 K4", 1280, 5120, 4, 1}, {"wkv 512x5120 K4", 512, 5120, 4, 1},
                           {"wq_b 32768x1280 K4", 32768, 1280, 4, 1}, {"wo_a 8x1024x4096 K5", 1024, 4096, 5, 8},
                           {"wo_b 5120x8192 K4", 5120, 8192, 4, 1}, {"shared w1 2304x5120 K4", 2304, 5120, 4, 1}, {"shared w3 2304x5120 K4", 2304, 5120, 4, 1},
                           {"shared w2 5120x2304 K4", 5120, 2304, 4, 1}};
    std::printf("\n%-26s %4s %8s %9s %9s %9s %9s %9s\n", "launch", "rows", "MB", "cold us", "floor us", "warm us", "cold GB/s",
                "floor GB/s");
    double sum_cold[5] = {}, sum_floor[5] = {};
    for (const auto& d : dense) {
        std::vector<ds41::DeviceExpert> tab(d.groups);
        size_t bytes = 0;
        for (int g = 0; g < d.groups; ++g) {
            Tensor t = make_tensor(d.n, d.k, d.bits, rng);
            tab[g].gate = reinterpret_cast<const uint8_t*>(t.trellis); tab[g].gate_scale = reinterpret_cast<const uint8_t*>(t.scales);
            bytes += t.bytes;
        }
        ds41::DeviceExpert* dt; CK(cudaMalloc(&dt, tab.size() * sizeof(tab[0])));
        CK(cudaMemcpy(dt, tab.data(), tab.size() * sizeof(tab[0]), cudaMemcpyHostToDevice));
        for (int m = 1; m <= 4; ++m) {
            ds41::GemmJob job{0, m, reinterpret_cast<const uint8_t*>(x), nullptr, y};
            const bool woa = d.groups > 1;
            const size_t slots = size_t(d.groups) * (d.n / 128);
            auto launch = [&] {
                CK(ds41::exl3_dense(dt, job, d.n, d.k, d.bits, 0, woa ? 32768 : 0, woa ? 8192 : 0, d.groups, woa ? 4096 : 0,
                                    woa ? 1024 : 0, partial, pf, counts, slots));
            };
            double cold, warm, floor_c, floor_w;
            time_it(launch, cold, warm);
            const size_t n16 = std::min(bytes, rd_bytes) / 16;
            time_it([&] { stream_read<<<sms * 4, 256>>>(rd, n16, sink); CK(cudaGetLastError()); }, floor_c, floor_w);
            std::printf("%-26s %4d %8.2f %9.1f %9.1f %9.1f %9.0f %9.0f\n", d.name, m, bytes / 1e6, cold, floor_c, warm,
                        bytes / cold / 1e3, bytes / floor_c / 1e3);
            sum_cold[m] += cold; sum_floor[m] += floor_c;
        }
    }
    for (int m = 1; m <= 4; ++m)
        std::printf("a layer's dense launches at %d rows: %.1f us cold against a floor of %.1f (x40 layers: %.2f ms against %.2f)\n",
                    m, sum_cold[m], sum_floor[m], 40e-3 * sum_cold[m], 40e-3 * sum_floor[m]);

    // a layer's routed experts: count experts of mixed widths (K2/K3/K4 roughly as the 2.7 bpw bank's demands), one or
    // two rows each, gate+up in one launch then down, as the streamed decode path issues a ready batch
    std::printf("\n%-26s %4s %8s %9s %9s %9s %9s %9s\n", "experts (gate+up, down)", "jobs", "MB", "cold us", "floor us",
                "warm us", "cold GB/s", "floor GB/s");
    const int widths[] = {3, 2, 3, 4, 3, 2, 3, 3, 4, 2, 3, 4, 2, 3, 3, 2, 4, 3};
    std::vector<ds41::DeviceExpert> etab(18);
    std::vector<uint8_t> ebits(18);
    size_t ebytes[18][2];
    for (int e = 0; e < 18; ++e) {
        Tensor g = make_tensor(2304, 5120, widths[e], rng), u = make_tensor(2304, 5120, widths[e], rng),
               dn = make_tensor(5120, 2304, widths[e], rng);
        etab[e].gate = reinterpret_cast<const uint8_t*>(g.trellis); etab[e].gate_scale = reinterpret_cast<const uint8_t*>(g.scales);
        etab[e].up = reinterpret_cast<const uint8_t*>(u.trellis); etab[e].up_scale = reinterpret_cast<const uint8_t*>(u.scales);
        etab[e].down = reinterpret_cast<const uint8_t*>(dn.trellis); etab[e].down_scale = reinterpret_cast<const uint8_t*>(dn.scales);
        ebits[e] = uint8_t(widths[e]);
        ebytes[e][0] = g.bytes + u.bytes; ebytes[e][1] = dn.bytes;
    }
    ds41::DeviceExpert* det; uint8_t* dbits;
    CK(cudaMalloc(&det, etab.size() * sizeof(etab[0]))); CK(cudaMemcpy(det, etab.data(), etab.size() * sizeof(etab[0]), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dbits, 18)); CK(cudaMemcpy(dbits, ebits.data(), 18, cudaMemcpyHostToDevice));
    __nv_bfloat16 *eg, *eu, *ed; CK(cudaMalloc(&eg, size_t(18) * 2 * 2304 * 2)); CK(cudaMalloc(&eu, size_t(18) * 2 * 2304 * 2));
    CK(cudaMalloc(&ed, size_t(18) * 2 * 5120 * 2));
    for (int count : {6, 13, 18}) for (int rows : {1, 2}) {
        std::vector<ds41::GemmJob> gj(count), uj(count), dj(count);
        size_t gu_bytes = 0, d_bytes = 0;
        for (int e = 0; e < count; ++e) {
            gj[e] = {ds41::ExpertId(e), rows, reinterpret_cast<const uint8_t*>(x), nullptr, eg + size_t(e) * 2 * 2304};
            uj[e] = {ds41::ExpertId(e), rows, reinterpret_cast<const uint8_t*>(x), nullptr, eu + size_t(e) * 2 * 2304};
            dj[e] = {ds41::ExpertId(e), rows, reinterpret_cast<const uint8_t*>(x), nullptr, ed + size_t(e) * 2 * 5120};
            gu_bytes += ebytes[e][0]; d_bytes += ebytes[e][1];
        }
        ds41::GemmJob *dg, *du, *dd;
        CK(cudaMalloc(&dg, count * sizeof(gj[0]))); CK(cudaMalloc(&du, count * sizeof(gj[0]))); CK(cudaMalloc(&dd, count * sizeof(gj[0])));
        CK(cudaMemcpy(dg, gj.data(), count * sizeof(gj[0]), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(du, uj.data(), count * sizeof(gj[0]), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dd, dj.data(), count * sizeof(gj[0]), cudaMemcpyHostToDevice));
        const std::pair<const char*, size_t> parts[] = {{"gate+up", gu_bytes}, {"down", d_bytes}};
        for (int p = 0; p < 2; ++p) {
            auto launch = [&] {
                if (p == 0) CK(ds41::grouped_exl3_var_gu(det, dg, du, count, 2304, 5120, dbits, 0, rows));
                else CK(ds41::grouped_exl3_var(det, dd, count, ds41::Projection::Down, 5120, 2304, dbits, 0, rows));
            };
            double cold, warm, floor_c, floor_w;
            time_it(launch, cold, warm);
            const size_t n16 = std::min(parts[p].second, rd_bytes) / 16;
            time_it([&] { stream_read<<<sms * 4, 256>>>(rd, n16, sink); CK(cudaGetLastError()); }, floor_c, floor_w);
            char name[64]; std::snprintf(name, sizeof name, "%s, %d row%s each", parts[p].first, rows, rows > 1 ? "s" : "");
            std::printf("%-26s %4d %8.2f %9.1f %9.1f %9.1f %9.0f %9.0f\n", name, count, parts[p].second / 1e6, cold, floor_c, warm,
                        parts[p].second / cold / 1e3, parts[p].second / floor_c / 1e3);
        }
        cudaFree(dg); cudaFree(du); cudaFree(dd);
    }
    // the launch itself: an empty kernel's cold and warm time
    double ec, ew;
    time_it([&] { stream_read<<<1, 32>>>(rd, 0, sink); CK(cudaGetLastError()); }, ec, ew);
    std::printf("\nan empty launch: %.1f us alone, %.1f us back to back\n", ec, ew);
    std::printf("done\n");
    return 0;
}
