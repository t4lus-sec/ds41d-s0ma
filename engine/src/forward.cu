// s0ma change 32 (N4): a kernel launched behind a programmatic graph edge (DS41_PDL=1) starts before its
// predecessor has finished and waits here until that predecessor's writes are visible; otherwise a no-op.
#ifndef DS41_PDL_WAIT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define DS41_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#else
#define DS41_PDL_WAIT() do {} while (0)
#endif
#endif
#include "ds41/attention.h"
#include "ds41/blas_runtime.hpp"
#include "ds41/hotpath/kernels.cuh"
#include "ds41/hotpath/route_plan.hpp"
#include "ds41/hotpath/fused.hpp"
#include "ds41/hotpath/gate.hpp"
#include "ds41/hotpath/moe_stats.hpp"
#include "ds41/hotpath/readiness.cuh"
#include <array>
#include "ds41/dense.h"
#include "ds41/engine.hpp"
#include "ds41/engram_async.hpp"
#include "ds41/exl3.cuh"
#include "ds41/kernels.cuh"
#include "ds41/native_store.hpp"
#include "ds41/offload.hpp"
#include "ds41/ops.cuh"
#include "ds41/prefill/plan.hpp"
#include "ds41/prefill/kernels.cuh"
#include "ds41/prefill/stager.hpp"
#include "ds41/vision.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/stat.h>
#include <cublas_v2.h>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <random>
namespace ds41e {
namespace {
// Zero-copy byte copy on the SMs. The GPU has one copy engine per direction, so small per-layer
// metadata uploads and score/logit readbacks issued as cudaMemcpyAsync queue FIFO behind 18.8 MB
// expert uploads (H2D) and RAM demotions (D2H). Mapped pinned memory avoids both engines.
// Batched zero-copy: block i copies segment i; the segment table itself lives in mapped pinned memory.
struct ZcSeg { const unsigned char *src; unsigned char *dst; unsigned long long n; };
__global__ void zc_multi(const ZcSeg *__restrict__ segs, int count) {DS41_PDL_WAIT();
  for (int s = blockIdx.x; s < count; s += gridDim.x) {
    const ZcSeg g = segs[s];
    for (size_t i = size_t(threadIdx.x) * 16; i < g.n; i += size_t(blockDim.x) * 16) {
      if (i + 16 <= g.n && !(reinterpret_cast<uintptr_t>(g.src + i) & 15) && !(reinterpret_cast<uintptr_t>(g.dst + i) & 15))
        *reinterpret_cast<uint4 *>(g.dst + i) = *reinterpret_cast<const uint4 *>(g.src + i);
      else
        for (size_t j = i; j < g.n && j < i + 16; ++j) g.dst[j] = g.src[j];
    }
  }
}
__global__ void zc_copy(const unsigned char *__restrict__ src, unsigned char *__restrict__ dst, size_t n) {DS41_PDL_WAIT();
  const size_t stride = size_t(gridDim.x) * blockDim.x * 16;
  for (size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16; i < n; i += stride) {
    if (i + 16 <= n && !(reinterpret_cast<uintptr_t>(src + i) & 15) && !(reinterpret_cast<uintptr_t>(dst + i) & 15))
      *reinterpret_cast<uint4 *>(dst + i) = *reinterpret_cast<const uint4 *>(src + i);
    else
      for (size_t j = i; j < n && j < i + 16; ++j) dst[j] = src[j];
  }
}
constexpr int H = 5120, I = 2304, V = 129280, HC = 4, RING = 134;
void ck(int status, const char *op) {
  if (status)
    throw std::runtime_error(std::string(op) + ": " +
                             (status < 10000
                                  ? cudaGetErrorString(cudaError_t(status))
                                  : std::to_string(status)));
}
#define CK(op) ck(int(op), #op)
double clock_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
void vram_checkpoint(const char *label) {
  auto f=reinterpret_cast<void(*)(const char*)>(dlsym(RTLD_DEFAULT,"ds41_vram_checkpoint"));
  if (f) f(label);
}
struct Device {
  void *p = nullptr, *mapped_host = nullptr;
  size_t bytes = 0;
  Device() = default;
  explicit Device(size_t n) { resize(n); }
  ~Device() { release(); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&b) noexcept : p(b.p), mapped_host(b.mapped_host), bytes(b.bytes) {
    b.p = b.mapped_host = nullptr; b.bytes = 0;
  }
  Device &operator=(Device &&b) noexcept {
    if (this != &b) {
      release(); p=b.p; mapped_host=b.mapped_host; bytes=b.bytes;
      b.p=b.mapped_host=nullptr; b.bytes=0;
    }
    return *this;
  }
  void release() noexcept {
    if (mapped_host) cudaFreeHost(mapped_host);
    else if (p) cudaFree(p);
    p=mapped_host=nullptr; bytes=0;
  }
  void resize(size_t n) {
    release();
    if (n) { CK(cudaMalloc(&p,n)); bytes=n; }
  }
  void resize_mapped(size_t n) {
    release();
    if (n) {
      CK(cudaHostAlloc(&mapped_host,n,cudaHostAllocMapped));
      auto rc=cudaHostGetDevicePointer(&p,mapped_host,0);
      if (rc!=cudaSuccess) { release(); CK(rc); }
      bytes=n;
    }
  }
  template <class T = void> T *as(size_t offset = 0) const {
    return reinterpret_cast<T *>(static_cast<char *>(p) + offset);
  }
};
struct Weight {
  Device data;
  std::vector<uint64_t> shape;
  std::string dtype;
};
struct Global {
  Device kv, key, raw_kv, raw_score;
  int ratio, capacity;
  Global(int max, int r)
      : kv(size_t(max / r) * 288), key(size_t(max / r) * 68),
        raw_kv(8 * 512 * 4), raw_score(8 * 512 * 4), ratio(r),
        capacity(max / r) {}
};
struct Phase {
  cudaEvent_t start = nullptr, end = nullptr;
  explicit Phase() {
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&end));
  }
  ~Phase() {
    if (start)
      cudaEventDestroy(start);
    if (end)
      cudaEventDestroy(end);
  }
};
// s0ma change 32 (N4): programmatic edges between ds41d's own kernels in each captured graph (pdl_graph.cpp)
extern "C" int ds41_pdl_mode();
extern "C" int ds41_pdl_rewrite(cudaGraph_t graph, int mode, size_t *edges, const char **step);
extern "C" const char *ds41_pdl_node_name(cudaGraphNode_t node);
extern "C" const char *ds41_pdl_node_types(cudaGraph_t graph);
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ~Graph() {
    if (exec)
      cudaGraphExecDestroy(exec);
    if (graph)
      cudaGraphDestroy(graph);
  }
};
Store verified_store(const Config &c) {
  std::cerr << "native runtime " << native_blas_runtime().dump() << '\n';
  Store source(c.weights);
  source.verify_receipt(c.receipt);
  std::map<std::string, std::string> role_paths = {
      {"weight_index", c.weights + "/model.safetensors.index.json"},
      {"config", c.weights + "/config.json"},
      {"inference_config", c.weights + "/inference/config.json"},
      {"tokenizer", c.weights + "/tokenizer.json"},
      {"official_model", c.weights + "/inference/model.py"},
      {"official_kernel", c.weights + "/inference/kernel.py"},
      {"official_engram", c.weights + "/inference/engram.py"},
      {"official_generate", c.weights + "/inference/generate.py"},
      {"engram_export", c.engram},
      {"rope_export", c.rope},
      {"rope_metadata", c.rope + ".json"},
      {"expert_manifest", c.experts}};
  if (!c.engram_library.empty())
    role_paths.emplace("engram_library", c.engram_library);
  auto auxiliary = authenticate_files(
      c.auxiliary.empty() ? DS41_DEFAULT_AUXILIARY : c.auxiliary,
      DS41_AUX_SHA256, role_paths);
  if (auxiliary.at("weight_root") != c.weights ||
      auxiliary.at("revision") != "1bae0e1088993b9b82de08f59de5be5a34e42170")
    throw std::runtime_error("auxiliary checkpoint identity mismatch");
  auto engram = read_json(c.engram);
  for (const auto &entry : {std::pair<const char *, const char *>{
                                "inference/engram.py", "official_engram"},
                            {"inference/config.json", "inference_config"},
                            {"tokenizer.json", "tokenizer"}})
    if (engram.at("provenance").at(entry.first) !=
        auxiliary.at("files").at(entry.second).at("sha256"))
      throw std::runtime_error("Engram export provenance mismatch");
  auto rope = read_json(c.rope + ".json");
  auto rows = rope.at("rows").get<int64_t>();
  if (rope.at("schema") != "ds41-rope-v1" || rows < c.max_context ||
      rows > 1048576 || rope.at("dtype") != "F32" ||
      rope.at("layout") != "[2,rows,32,2] pure-SWA then compressed" ||
      rope.at("source_sha256") !=
          auxiliary.at("files").at("official_model").at("sha256") ||
      rope.at("sha256") !=
          auxiliary.at("files").at("rope_export").at("sha256") ||
      std::filesystem::file_size(c.rope) != uint64_t(rows) * 2 * 64 * 4)
    throw std::runtime_error("RoPE export provenance/geometry mismatch");
  if (!c.allow_unqualified)
    throw std::runtime_error(
        "M1 native forward is unqualified; full-model gold gate pending (use "
        "--allow-unqualified for development only)");
  return source;
}
class Native final : public Forward, public hotpath::GateEvidence {
  Config config;
  Store source;
  EngramHash hash;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  std::map<std::string, Weight> weights;
  // s0ma change 34 (N1): 0xSero's dense EXL3 tensors in place of native FP8 ones (DS41_SERO_DENSE=1|DIR). Each is one
  // DeviceExpert entry of dense_table (its gate views: trellis, then suh and svh); its FP8 weight and scale are freed.
  struct DenseExl3 { int index = 0, k = 0, n = 0, bits = 0; };
  std::map<std::string, DenseExl3> dense_exl3;      // the tensor's name without ".weight"
  std::vector<Device> dense_records;
  Device dense_table;
  double dense_extra_gib = 0;                         // VRAM the swap gave back, for expert slots
  std::set<std::string> dense_woa;                    // "layers.L.attn" whose wo_a runs as Sero's eight slices
  // s0ma change 47 (H1): the step's tier snapshot, each row's private cost so far, the cut counters
  std::vector<uint8_t> h1_snap;
  std::array<double, 8> h1_spent{};
  int h1_last_m = 0;
  int64_t h1_cuts = 0, h1_rows_cut = 0;
  // s0ma change 37: decode's split-K scratch, one set a verify lane (partial sums; counts, zeroed and left zeroed),
  // and prefill's: one layer's tensors decoded to FP16 (filled on a layer's first use, reused by its other chunks),
  // the rows rotated into FP16 and the GEMM's FP32 result, 256 rows at a time
  Device dense_split_partial[2], dense_split_counts[2];
  size_t dense_split_floats = 0;
  static constexpr size_t dense_split_slots = 4096;
  Device dense_cache, dense_rows_a, dense_rows_c;
  std::string dense_cache_layer;
  size_t dense_cache_used = 0;
  std::map<std::string, size_t> dense_cache_at;
  bool dense_since_quant = false;                     // an EXL3 projection ran since the last FP8 activation quant
  std::unique_ptr<ds41::OffloadStore> experts;
  // --exl3-experts: `experts` holds EXL3 K3 target experts (layers 0..39) and this store the native
  // MXFP4 MTP experts, all pinned in VRAM. Without it, `experts` serves every logical ID.
  std::unique_ptr<ds41::OffloadStore> mtp_experts;
  bool exl3 = false;
  int exl3_bits_loaded = 3;
  // s0ma change 12: each target layer's EXL3 width (2 or 3), from its records' size. A bank whose layers differ
  // ("experts": "mixed") gets one store slot class per width; exl3_bits_loaded is then 0.
  std::array<int, 40> layer_bits{};   // 0: the layer mixes widths (s0ma change 15)
  std::vector<uint8_t> width_of;      // each target expert's width
  Device expert_width_dev;            // the same on the device, for grouped_exl3_var
  bool exl3_mixed = false;
  uint64_t expert_payload_bytes = 0;   // the target store's VRAM slots in bytes (all classes)
  std::vector<int> class_widths;       // s0ma change 18: the width of each slot class of a mixed bank
  std::string mixed_k3_layers;         // for the log and stats: "9,12,17,..."
  int expert_bits(int layer) const { return layer >= 0 && layer < 40 && layer_bits[layer] ? layer_bits[layer] : exl3_bits_loaded; }
  size_t expert_slot_bytes = 18808832;
  ds41::OffloadStore &spark_store() { return mtp_experts ? *mtp_experts : *experts; }
  // Target-layer expert GEMM. EXL3 jobs read BF16 rows; native jobs MXFP8 rows and scales.
  // s0ma change 40: DS41_EXL3_SPLITK=1 splits each grouped launch over K when its experts give few CTAs (a scratch
  // set a verify lane; up to 64 jobs a launch)
  static bool exl3_splitk_on() {
    static const bool on = [] { const char *v = std::getenv("DS41_EXL3_SPLITK"); return v && std::string(v) == "1"; }();
    return on;
  }
  Device splitk_partial[2], splitk_counts[2];
  static constexpr int splitk_jobs = 64;
  void splitk_scratch(int lane) {
    if (!exl3 || !exl3_splitk_on() || splitk_partial[lane].bytes) return;
    const size_t floats = std::max(ds41::exl3_var_splitk_floats(2304, 5120, 1) * splitk_jobs,
                                   ds41::exl3_var_splitk_floats(5120, 2304, 1) * splitk_jobs);
    splitk_partial[lane].resize(floats * 4);
    splitk_counts[lane].resize(size_t(splitk_jobs) * 40 * 4);
    CK(cudaMemset(splitk_counts[lane].p, 0, splitk_counts[lane].bytes));
  }
  cudaError_t expert_gemm(const ds41::GemmJob *jobs, int count, ds41::Projection p,
                          int n, int k, cudaStream_t s, int layer) {
    if (exl3 && exl3_splitk_on() && count <= splitk_jobs) {
      const int lane = lane_b_active ? 1 : 0;
      if (splitk_partial[lane].bytes)
        return ds41::grouped_exl3_splitk(experts->table(), jobs, count, p, n, k,
                                         expert_bits(layer) ? nullptr : expert_width_dev.as<uint8_t>(), expert_bits(layer),
                                         splitk_partial[lane].as<float>(), splitk_partial[lane].bytes / 4,
                                         splitk_counts[lane].as<unsigned>(), splitk_counts[lane].bytes / 4, s);
    }
    if (exl3 && !expert_bits(layer))   // s0ma change 15: a layer whose experts differ in width
      return ds41::grouped_exl3_var(experts->table(), jobs, count, p, n, k, expert_width_dev.as<uint8_t>(), s);
    return exl3 ? ds41::grouped_exl3_bits(experts->table(), jobs, count, p, n, k, expert_bits(layer), s)
                : ds41::grouped_mxfp4_native_parallel(experts->table(), jobs, count, p, n, k, s);
  }
  std::unique_ptr<hotpath::FusedOps> fused;
  std::unique_ptr<ds41::DirectReader> reader;
  std::unique_ptr<ds41::HostBuffer> pages, host;
  std::unique_ptr<EngramAsync> async_engram;
  std::array<std::unique_ptr<EngramAsync::Ticket>, 2> engram_tickets;
  Json initial_engram_stats, host_cache_policy;
  Json prefill_summary=Json::object();
  std::map<int, std::unique_ptr<Global>> globals;
  std::vector<Device> windows;
  std::vector<std::unique_ptr<Phase>> phases;
  std::map<std::string, std::unique_ptr<Graph>> graphs;
  uint64_t graph_replays = 0, initial_replays = 0;
  uint64_t graph_captures = 0, initial_captures = 0;
  uint64_t pdl_edges = 0, pdl_graphs = 0;   // s0ma change 32
  // s0ma change 36: with DS41_PDL, each cuBLAS handle gets a fixed 32 MiB workspace (cuBLAS's own default size on this
  // card). Without one, cuBLAS allocates its workspace stream-ordered while a graph is captured: the graph then holds
  // memory allocation nodes, and such a graph can be neither cloned nor have edges removed (the two refusals seen).
  Device pdl_blas_workspace, pdl_lane_workspace;
  void pdl_workspace(cublasHandle_t handle, cudaStream_t on, Device &ws) {
    if (ds41_pdl_mode() <= 0) return;
    // s0ma change 39: the handle's stream first: setting a stream resets the workspace (the wrappers now set it only
    // when it changes, and the handle stays on this stream)
    if (cublasSetStream(handle, on) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSetStream (DS41_PDL)");
    ws.resize(size_t(32) << 20);
    if (cublasSetWorkspace(handle, ws.p, ws.bytes) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSetWorkspace (DS41_PDL)");
  }
  bool pdl_off = false;
  std::string pdl_error, pdl_alloc_note;   // s0ma change 38
  uint64_t pdl_skipped = 0;
  uint64_t graph_evictions = 0, initial_evictions = 0;
  std::array<std::array<int,40>,6> attention_scan_bucket{};
  bool graph_ready = false;
  Device h{HC * H * 2}, next_h{HC * H * 2}, x{H * 2}, y{H * 2}, f32{HC * H * 4};
  Device pre{4 * 4}, att_pre{4 * 4}, ffn_pre{4 * 4}, post{4 * 4}, comb{16 * 4},
      mixes{24 * 4}, rstd{4};
  Device qa{1280 * 2}, q{64 * 512 * 2}, kv{512 * 2}, latent{512 * 2},
      ik{128 * 2}, iq{32 * 128 * 2}, iw{32 * 2};
  Device packed{528}, gather{640 * 512 * 2}, local_idx{640 * 4},
      o{64 * 512 * 2}, wo{8192 * 2}, attn_scores{64 * 640 * 4};
  Device aq{32768}, as{1024}, gate{6 * I * 2}, up{6 * I * 2}, act{6 * I * 2},
      eq{6 * I}, es{6 * I / 32}, down{6 * H * 2}, shared{H * 2};
  Device sg{I * 2}, su{I * 2}, sa{I * 2}, jobs{6 * sizeof(ds41::GemmJob)},
      up_jobs{6 * sizeof(ds41::GemmJob)}, down_jobs{6 * sizeof(ds41::GemmJob)},
      ids_device{6 * 4}, route{6 * 4};
  Device router{384 * 4}, unbiased{384 * 4}, biased{384 * 4}, logits{V * 4};
  Device scores, top{512 * 4}, visible{4}, position_device{8}, blocks,
      block_visible{4}, chosen_blocks{2048 * 4}, candidate_ids{16384 * 4},
      candidate_top{512 * 4}, top_scratch;
  Device rows_raw{24 * 256}, scales_raw{24 * 8}, rows_bf{24 * 256 * 2},
      engram_kv{(HC + 1) * H * 2}, rows_interleaved{24 * 264};
  Device braw_kv{6*512*4}, braw_score{6*512*4}, batch_tokens{6*4};
  Device route_source{36*4}, route_slots{36*4}, bx{36*H}, bxs{36*H/32}, broute{36*4};
  Device bxh; // EXL3: 36 gathered BF16 rows
  Device compact_down{36*H*2}, sparse_scores{6*64*640*4};
  Device work_jobs{36*sizeof(ds41::GemmJob)}, work_up_jobs{36*sizeof(ds41::GemmJob)},
      work_down_jobs{36*sizeof(ds41::GemmJob)}, work_rows{36*4};
  std::array<std::vector<ds41::ExpertId>,40> previous_routes;
  std::vector<ds41::OffloadStore::PrefetchTicket> prefetch_tickets;
  // --prefetch-lookahead scratch. lookahead_sync->start: route scores on host; ->end: predicted
  // scores for lookahead_layer on host.
  Device lookahead_router{8 * 384 * 4}, lookahead_unbiased{8 * 384 * 4}, lookahead_biased{8 * 384 * 4};
  // s0ma change 18 (DS41_LOOKAHEAD_HC=1): the lookahead's router input built with the target layer's own
  // hyper-connection mix and ffn_norm from the current streams (six rows; scratch apart from the step's own).
  // s0ma change 21: DS41_LOOKAHEAD_HC=L applies it to target layers L and up ("1": all; 0 or unset: none). Sero's
  // audit: recall up in layers 7-39 (15: 37 -> 84%, 19: 46 -> 87%), down in 1-6 (2: 43 -> 26%), where the attention
  // still moves the streams most.
  const int lookahead_hc = [] { const char *v = std::getenv("DS41_LOOKAHEAD_HC"); return v && *v ? std::max(0, std::atoi(v)) : 0; }();
  Device la_pre{6 * 4 * 4}, la_post{6 * 4 * 4}, la_comb{6 * 16 * 4}, la_mixes{6 * 24 * 4}, la_rstd{6 * 4},
      la_promoted{6 * HC * H * 4}, la_x{6 * H * 2}, la_in{6 * H * 4};
  int lookahead_m = 1, lookahead_n = 4;
  std::unique_ptr<ds41::HostBuffer> spark_host;
  const bool prefill_chunk_fit = [] { const char *v = std::getenv("DS41_PREFILL_CHUNK_FIT"); return v && std::string(v) == "1"; }();
  // s0ma leg 26: a verify step's per-row cuBLAS rows, hyper-connection mixes and route weights in one launch each,
  // every row computed as on its own (DS41_ROWS_BATCHED=1; tokens checked identical by the lab's benchmark).
  // s0ma change 10: "1" (or "all") batches all three; a comma list picks parts: dense (cuBLAS rows), hc (the mixes'
  // promotion and Sinkhorn), route (route weights), to find which one changes tokens.
  const int rows_parts = [] {
    const char *v = std::getenv("DS41_ROWS_BATCHED");
    if (!v || !*v || std::string(v) == "0") return 0;
    if (std::string(v) == "1" || std::string(v) == "all") return 7;
    int parts = 0; std::string s(v), part;
    for (size_t i = 0; i <= s.size(); ++i) {
      if (i < s.size() && s[i] != ',') { part += s[i]; continue; }
      if (part == "dense") parts |= 1; else if (part == "hc") parts |= 2; else if (part == "route") parts |= 4;
      else throw std::invalid_argument("DS41_ROWS_BATCHED: 1, all, or a comma list of dense, hc, route");
      part.clear();
    }
    return parts;
  }();
  const bool rows_batched = rows_parts != 0;
  const bool rows_dense = rows_parts & 1, rows_hc = rows_parts & 2, rows_route = rows_parts & 4;
  // s0ma change 10: DS41_ROWS_VERIFY=1 recomputes every batched cuBLAS call one row at a time outside graph capture
  // and logs the shapes where the bits differ ("rows verify: ..." on stderr). A diagnostic: it syncs every call.
  const bool rows_verify = [] { const char *v = std::getenv("DS41_ROWS_VERIFY"); return v && std::string(v) == "1"; }();
  // s0ma change 29: the verify's logits also kept on the device, [6, V] f32, for acceptance there
  // (DS41_SPARK_GPU_VERIFY=1; layer-major verify only); vlogits_rows is 0 when the last verify left none
  const bool keep_device_logits = [] { const char *v = std::getenv("DS41_SPARK_GPU_VERIFY"); return v && (std::string(v) == "1" || std::string(v) == "check"); }();
  Device vlogits;
  int vlogits_rows = 0;
  Device rows_verify_buf;
  uint64_t rows_verify_calls = 0, rows_verify_bad = 0;
  std::set<std::tuple<int, int, int, int>> rows_verify_seen;
  static bool aligned256(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 255) == 0; }
  void rows_verify_call(const void *in, const void *weight, const void *out, int n, int k, bool f32, int m) {
    cudaStreamCaptureStatus cs;
    CK(cudaStreamIsCapturing(stream, &cs));
    if (cs != cudaStreamCaptureStatusNone) return;
    const size_t el = f32 ? 4 : 2, bytes = size_t(m) * n * el;
    if (rows_verify_buf.bytes < bytes) rows_verify_buf.resize(bytes);
    for (int r = 0; r < m; ++r)
      CK(ds41_dense_cublas(blas, static_cast<const char *>(in) + size_t(r) * k * el, weight,
                           rows_verify_buf.as<char>(size_t(r) * n * el), 1, n, k, f32, 1, stream));
    std::vector<char> a(bytes), b(bytes);
    CK(cudaMemcpyAsync(a.data(), out, bytes, cudaMemcpyDeviceToHost, stream));
    CK(cudaMemcpyAsync(b.data(), rows_verify_buf.p, bytes, cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    ++rows_verify_calls;
    if (std::memcmp(a.data(), b.data(), bytes)) {
      ++rows_verify_bad;
      if (rows_verify_seen.insert({n, k, int(f32), m}).second)
        std::cerr << "rows verify: DIFFERS " << n << "x" << k << (f32 ? " F32" : " BF16") << " rows " << m
                  << " (in " << (aligned256(in) ? "" : "un") << "aligned, out " << (aligned256(out) ? "" : "un")
                  << "aligned)\n";
    }
    if (rows_verify_calls % 5000 == 0)
      std::cerr << "rows verify: " << rows_verify_calls << " batched calls checked, " << rows_verify_bad << " differ\n";
  }
  // (N, K, F32?, rows) for which one batched call gave the same bits as one M=1 call a row, tested at load on every
  // F32/BF16 matrix shape of the checkpoint (cuBLAS picks its kernel by shape and batch count, not by the data).
  std::set<std::tuple<int, int, int, int>> rows_exact;
  bool rows_ok(int n, int k, bool f32, int rows) const { return rows_exact.count({n, k, int(f32), rows}) > 0; }
  void check_rows_batched() {
    std::map<std::tuple<int, int, int>, const void *> shapes;
    for (auto &[name, wt] : weights)
      if ((wt.dtype == "F32" || wt.dtype == "BF16") && wt.shape.size() == 2 && !wt.data.mapped_host && wt.data.p &&
          wt.shape[0] * wt.shape[1] <= (64ull << 20))
        shapes.emplace(std::make_tuple(int(wt.shape[0]), int(wt.shape[1]), int(wt.dtype == "F32")), wt.data.p);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int tested = 0;
    std::vector<std::string> inexact;
    for (auto &[shape, ptr] : shapes) {
      auto [n, k, f32] = shape;
      const size_t el = f32 ? 4 : 2;
      Device x(size_t(8) * k * el), y1(size_t(8) * n * el), y2(size_t(8) * n * el);
      std::vector<char> host(size_t(8) * k * el);
      for (size_t i = 0; i < size_t(8) * k; ++i) {
        float v = nd(rng);
        if (f32) std::memcpy(host.data() + i * 4, &v, 4);
        else { Bf16 b = __float2bfloat16(v); std::memcpy(host.data() + i * 2, &b, 2); }
      }
      CK(cudaMemcpy(x.p, host.data(), host.size(), cudaMemcpyHostToDevice));
      for (int rows = 2; rows <= 8; ++rows) {
        // s0ma change 8: compare both paths eagerly AND inside a stream capture, as the hot path runs them (eager
        // before its graphs are ready, captured after). cuBLAS may pick other kernels while capturing, so a shape is
        // batched only if one call equals one call a row in both modes.
        bool same_both = true;
        for (int captured = 0; captured <= 1 && same_both; ++captured) {
          auto enqueue = [&](bool batched) {
            if (batched) CK(ds41_dense_cublas_rows(blas, x.p, ptr, y2.p, rows, n, k, f32, stream));
            else for (int r = 0; r < rows; ++r)
              CK(ds41_dense_cublas(blas, x.as<char>(r * k * el), ptr, y1.as<char>(r * n * el), 1, n, k, f32, 1, stream));
          };
          for (int batched = 0; batched <= 1; ++batched) {
            if (!captured) { enqueue(batched); continue; }
            cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
            CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try { enqueue(batched); } catch (...) { cudaStreamEndCapture(stream, &g); if (g) cudaGraphDestroy(g); throw; }
            CK(cudaStreamEndCapture(stream, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            CK(cudaGraphLaunch(ge, stream));
            CK(cudaStreamSynchronize(stream));
            cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
          }
          CK(cudaStreamSynchronize(stream));
          std::vector<char> a(size_t(rows) * n * el), b(a.size());
          CK(cudaMemcpy(a.data(), y1.p, a.size(), cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(b.data(), y2.p, b.size(), cudaMemcpyDeviceToHost));
          same_both = std::memcmp(a.data(), b.data(), a.size()) == 0;
          CK(cudaMemsetAsync(y1.p, 0, a.size(), stream)); CK(cudaMemsetAsync(y2.p, 0xff, b.size(), stream));
        }
        ++tested;
        if (same_both) rows_exact.insert({n, k, f32, rows});
        else inexact.push_back(std::to_string(n) + "x" + std::to_string(k) + (f32 ? " F32" : " BF16") + " rows " +
                               std::to_string(rows));
      }
    }
    std::cerr << "rows batched (DS41_ROWS_BATCHED=1): " << rows_exact.size() << " of " << tested
              << " (shape, rows) pairs bit-identical in one call; one call a row for the rest";
    for (size_t i = 0; i < inexact.size() && i < 12; ++i) std::cerr << (i ? ", " : ": ") << inexact[i];
    std::cerr << "\n";
  }
  std::unique_ptr<ds41::HostBuffer> lookahead_host;
  std::unique_ptr<Phase> lookahead_sync;
  int lookahead_layer = -1, lookahead_k = 6;
  // s0ma change 11: DS41_LOOKAHEAD_NVME_DEPTH=D (2..3; default 0 = off) also scores layer l+D's router on layer l's
  // input and hands its top DS41_LOOKAHEAD_K a row to the store's NVMe read-ahead (DS41_MOE_SPEC_NVME), so an NVMe
  // read gets D layers of lead instead of one. A transfer hint like the lookahead: routing and outputs are unchanged.
  int lookahead_far_depth = 0, lookahead_far_layer = -1;
  std::unique_ptr<ds41::HostBuffer> lookahead_far_host;
  // DS41_LOOKAHEAD_AUDIT=path: per decode layer (m==1), each routed expert's tier at demand
  // (0 VRAM, 1 RAM, 2 NVMe) and its rank in the router lookahead made 1..3 layers earlier.
  // DS41_LOOKAHEAD_DRYRUN=1 scores without prefetching. Diagnostics only.
  std::ofstream lookahead_audit;
  bool lookahead_dryrun = false;
  int64_t audit_pos = 0;
  std::array<std::array<std::vector<int>, 3>, 40> audit_pred;
  // --verify-lanes (batch_lanes): lane B's spare scratch, stream, cuBLAS handle and step state are
  // swapped into the members while its work is enqueued; route_events[lane] marks score copies.
  bool lane_b_active = false;
  cudaStream_t lane_stream = nullptr;
  cublasHandle_t lane_blas = nullptr;
  std::vector<std::pair<Device *, Device>> lane_slots;
  int lane_candidate_count = 0;
  std::array<std::array<int,40>,6> lane_scan_bucket{};
  std::array<std::unique_ptr<EngramAsync::Ticket>, 2> lane_engram_tickets;
  std::array<cudaEvent_t, 2> route_events{};
  std::array<cudaEvent_t, 40> lane_att_events{};
  cudaEvent_t lane_entry = nullptr, lane_exit = nullptr;
  using FusedHead = int (*)(const float *, const void *, float *, int, int, int, cudaStream_t);
  void *fused_head_handle = nullptr;
  FusedHead fused_head = nullptr;
  // Target logits for m FP32 rows from the native BF16 head (--fused-head only).
  void fused_head_logits(const void *in, void *out, int m) {
    CK(fused_head(static_cast<const float *>(in), w("head.weight").data.p, static_cast<float *>(out), m, V, H, stream));
  }
  std::map<ds41::ExpertId, int> prefill_warm_counts;
  std::vector<std::future<void>> retirement_futures;
  size_t retirement_observed=0;
  uint64_t stream_callbacks=0, stream_hit_batches=0;
  Device batch_frequencies{12*64*4}, compressed_frequencies{6*64*4},
      lengths1{6*4}, lengths2{6*4}, packed_keys{6*68}, packed_globals{6*288};
  bool hotpath_capturing=false;
  std::vector<std::unique_ptr<Phase>> expert_events;
  size_t expert_event_index=0;
  std::unique_ptr<hotpath::ConsumerReadiness> consumer_readiness;
  double host_plan_ms=0, callback_enqueue_ms=0;
  Json profile_steps=Json::array(), step_demands=Json::array();
  Device gate_slab;
  std::vector<uint8_t> gate_host;
  std::vector<std::vector<uint8_t>> gate_rows;
  Device rope, woa_scratch;
  // Image prompts: the vision tower (built at the first image) and the IMAGE_START, IMAGE_NEW_LINE and
  // IMAGE_END embeddings, in that order.
  std::unique_ptr<VisionEncoder> vision;
  Device image_delimiters;
  Device capture_slab{6 * 3 * H * 2};
  int capture_slot = 0, capture_rows = 0, capture_committed = 0;
  int64_t capture_begin = 0;
  ds41_target_capture_sink capture_sink = nullptr;
  void *capture_user = nullptr;
  std::string callback_error;
  Device spark_aq, spark_as, spark_gate, spark_up, spark_act, spark_eq,
      spark_es, spark_jobs, spark_up_jobs, spark_down_jobs;
  int64_t committed = 0;
  int pending = 0;
  ds41e::prefill::Prefix prefix;
  std::vector<int> pending_ids;
  int shared_topk = 0, candidate_count = 0, eos_id = 1;
  ds41::Stats initial_stats{};
  mutable ds41::Stats last_expert_stats{};
  mutable bool expert_stats_available=false;
  double io_ms = 0, route_ms = 0, engram_ms = 0, total_ms = 0;
  size_t phase_index = 0;
  size_t upload_cursor = 1 << 20;
  bool in_step = false;
  bool poisoned = false;
  struct PrefillData;
  std::unique_ptr<PrefillData> pref;
  std::unique_ptr<ds41e::prefill::Stager> prefill_stage;
  // s0ma change 27: prefill lanes. A prompt's experts already in VRAM run on side streams, each with its own scratch,
  // beside the staged sweep on the main stream (DS41_PREFILL_LANES=N: N-1 side lanes; 1 = published, everything on
  // one stream). Each expert's math is unchanged and its rows land in their own slots of `routed`.
  struct PrefillLane {
    Device bxh, bw, gate, up, act, down;
    cudaStream_t s = nullptr;
    cudaEvent_t done = nullptr;
  };
  std::vector<std::unique_ptr<PrefillLane>> prefill_side;
  int prefill_lanes() {
    static const int n = [] {
      const char *v = std::getenv("DS41_PREFILL_LANES");
      const int x = v && *v ? std::atoi(v) : 1;
      if (x < 1 || x > 8) throw std::invalid_argument("DS41_PREFILL_LANES must be 1..8");
      return x;
    }();
    return n;
  }
  PrefillLane &prefill_lane(size_t i) {   // side lane i (0-based), created on first use
    while (prefill_side.size() <= i) {
      auto L = std::make_unique<PrefillLane>();
      L->bxh.resize(size_t(256) * H * 2); L->bw.resize(256 * 4); L->gate.resize(size_t(256) * I * 2);
      L->up.resize(size_t(256) * I * 2); L->act.resize(size_t(256) * I * 2); L->down.resize(size_t(256) * H * 2);
      CK(cudaStreamCreateWithFlags(&L->s, cudaStreamNonBlocking));
      CK(cudaEventCreateWithFlags(&L->done, cudaEventDisableTiming));
      prefill_side.push_back(std::move(L));
    }
    return *prefill_side[i];
  }
  int prefill_variant(cudaStream_t s) {   // 0: the main stream's scratch; i + 1: side lane i
    for (size_t i = 0; i < prefill_side.size(); ++i) if (prefill_side[i]->s == s) return int(i) + 1;
    return 0;
  }
  ds41e::prefill::TransferStats prefill_transfers;
  size_t prefill_vram_hits=0;
  std::vector<ds41::Record> prefill_records;
  std::map<std::string, double> measured;

#include "prefill/native.inc"
public:
  const std::vector<std::vector<uint8_t>> &hotpath_gate_rows() const override {
    return gate_rows;
  }
  explicit Native(const Config &c)
      : config(c), source(verified_store(c)), hash(c.engram) {
    // Every payload read, CUDA allocation, and host pin is after the full
    // receipt. This first engine is an N candidate until full fidelity gates
    // establish B0.
#if DS41_MOE_SOURCE_API<4
    if(c.dynamic_host || c.async_retire || c.reset_cache_history || c.host_recent_window!=-1 || c.host_decay_interval!=-1)
      throw std::invalid_argument("dynamic host/policy/history/async retirement require a jointly pinned MoE API4 build");
#else
    if(c.async_retire) retirement_futures.reserve(240);
#endif
    exl3 = !c.exl3_experts.empty();
    if (c.expert_slots < 6 || c.expert_slots > (exl3 ? 12000 : 5000))
      throw std::invalid_argument("expert slots must be 6..5000 (6..12000 with --exl3-experts)");
    if (exl3) bxh.resize(36 * H * 2);
    eos_id =
        read_json(c.weights + "/config.json").at("eos_token_id").get<int>();
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (auto &e : route_events) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    vram_checkpoint("native_members_before_cublas");
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasCreate");
    pdl_workspace(blas, stream, pdl_blas_workspace);   // s0ma change 36
    vram_checkpoint("native_after_cublas");
    if (c.cuda_stack_bytes) {
      if (c.cuda_stack_bytes > 65536) throw std::invalid_argument("CUDA stack request exceeds bounded candidate limit");
      CK(cudaDeviceSetLimit(cudaLimitStackSize, c.cuda_stack_bytes));
    }
    vram_checkpoint("native_after_stack_limit");
    if(c.hotpath_gate) {
      gate_slab.resize(6*hotpath::GateLayout::row_bytes);
      gate_host.resize(6*hotpath::GateLayout::row_bytes);
    }
    // Six-row verify arena. The same addresses serve every width and graph.
    for (Device *d : {&h,&next_h,&x,&y,&f32,&pre,&att_pre,&ffn_pre,&post,&comb,
         &mixes,&rstd,&qa,&q,&kv,&latent,&ik,&iq,&iw,&packed,&gather,&local_idx,
         &o,&wo,&aq,&as,&gate,&up,&act,&eq,&es,&down,&shared,&sg,&su,&sa,
         &jobs,&up_jobs,&down_jobs,&ids_device,&route,&router,&unbiased,&biased,
         &logits,&top,&visible,&block_visible,&chosen_blocks,&candidate_ids,
         &candidate_top,&rows_raw,&scales_raw,&rows_bf,&engram_kv,&rows_interleaved})
      d->resize(d->bytes * 6);
    if (keep_device_logits) vlogits.resize(size_t(6) * V * 4);
    host = std::make_unique<ds41::HostBuffer>(16 << 20, true);
    pages = std::make_unique<ds41::HostBuffer>(1 << 20, true);
    if (!c.fused_head_library.empty()) {
      // Spark drafts with its compact BF16 head over this same native matrix (head_kind 1).
      fused_head_handle = dlopen(c.fused_head_library.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!fused_head_handle) throw std::runtime_error(std::string("fused head: ") + dlerror());
      fused_head = reinterpret_cast<FusedHead>(dlsym(fused_head_handle, "ds41_fuse_head_decode"));
      if (!fused_head) throw std::runtime_error("fused head: missing ds41_fuse_head_decode");
    }
    if (c.prefetch_lookahead) {
      if (!c.streamed_experts)
        throw std::invalid_argument("--prefetch-lookahead requires --streamed-experts");
      if (const char *k = std::getenv("DS41_LOOKAHEAD_K")) lookahead_k = std::stoi(k);
      if (lookahead_k < 1 || lookahead_k > 32)
        throw std::invalid_argument("DS41_LOOKAHEAD_K must be 1..32");
      lookahead_host = std::make_unique<ds41::HostBuffer>(8 * 384 * 4, true);
      if (const char *d = std::getenv("DS41_LOOKAHEAD_NVME_DEPTH")) lookahead_far_depth = std::stoi(d);
      if (lookahead_far_depth && (lookahead_far_depth < 2 || lookahead_far_depth > 3))
        throw std::invalid_argument("DS41_LOOKAHEAD_NVME_DEPTH must be 0, 2 or 3");
      if (lookahead_far_depth) lookahead_far_host = std::make_unique<ds41::HostBuffer>(8 * 384 * 4, true);
      if (const char *n = std::getenv("DS41_LOOKAHEAD_N")) lookahead_n = std::stoi(n);
      if (lookahead_n < 1 || lookahead_n > 48) throw std::invalid_argument("DS41_LOOKAHEAD_N must be 1..48");
      if (const char *path = std::getenv("DS41_LOOKAHEAD_AUDIT")) {
        lookahead_audit.open(path);
        if (!lookahead_audit) throw std::runtime_error("lookahead audit open");
      }
      if (const char *dry = std::getenv("DS41_LOOKAHEAD_DRYRUN")) lookahead_dryrun = std::string(dry) == "1";
      lookahead_sync = std::make_unique<Phase>();
    }
    reader = std::make_unique<ds41::DirectReader>(64);
    load_weights();
    load_sero_dense();
    splitk_scratch(0);   // s0ma change 40
    if (fused_head && !dense_of("head")) {   // s0ma change 43: unless Sero's head replaced it
      const auto &head = w("head.weight");
      if (head.dtype != "BF16" || head.shape != std::vector<uint64_t>({uint64_t(V), uint64_t(H)}))
        throw std::runtime_error("--fused-head requires the native BF16 [vocab, hidden] head");
    }
    vram_checkpoint("native_after_weights");
    load_rope();
    vram_checkpoint("native_after_rope");
    if(!c.fuse_library.empty()) fused=std::make_unique<hotpath::FusedOps>(c.fuse_library);
    if (!c.engram_library.empty())
      async_engram = std::make_unique<EngramAsync>(c.engram_library, c);
    for (int l = 0; l < 40; ++l)
      windows.emplace_back(RING * 528);
    for (int l : {2, 8, 14, 20})
      globals.emplace(
          l, std::make_unique<Global>(config.max_context, l == 20 ? 1 : 2));
    scores.resize(size_t(std::max(16384, config.max_context + 8)) * 2 * 6);
    blocks.resize(size_t((config.max_context + 7) / 8) * 2 * 6);
    top_scratch.resize(
        ds41_attn_topk_scratch(6, std::max(16384, config.max_context + 8)));
    vram_checkpoint("native_after_kv");
    auto records = ds41::load_records(c.experts);
    if (records.size() != ds41::expert_count)
      throw std::runtime_error(
          "expert manifest must include all 15744 logical ids");
    // The manifest is not trusted to select different weights: every tensor
    // view must land on the indexed native tensor in this verified checkpoint.
    validate_records(records);
    if (exl3) {
      std::vector<ds41::Record> mtp;
      for (auto &r : records) if (r.id >= 15360) mtp.push_back(r);
      records = ds41::load_records(c.exl3_experts);
      validate_k3_records(records, c.exl3_receipts);
      expert_slot_bytes = 0;
      for (auto &r : records) expert_slot_bytes = std::max(expert_slot_bytes, (size_t(r.bytes) + 4095) / 4096 * 4096);
      make_mtp_store(std::move(mtp));
      vram_checkpoint("native_after_mtp_store");
    }
    if(c.prefill)prefill_records=records;
    ds41::OffloadStore::Config ec;
    ec.slots = c.expert_slots;
    ec.host_budget_bytes = c.host_bytes;
    ec.staging_slots = 8;
    ec.io_depth = 64;
#if DS41_MOE_SOURCE_API>=4
    ec.dynamic_host=c.dynamic_host;
    if(c.host_recent_window < -1 || c.host_decay_interval < -1)
      throw std::invalid_argument("invalid host cache policy");
    if(c.host_recent_window>=0) ec.host_recent_window=unsigned(c.host_recent_window);
    if(c.host_decay_interval>=0) ec.host_decay_interval=unsigned(c.host_decay_interval);
    host_cache_policy={{"recent_window",ec.host_recent_window},{"decay_interval",ec.host_decay_interval}};
#endif
    expert_payload_bytes = uint64_t(c.expert_slots) * expert_slot_bytes;
    if (exl3_mixed) mixed_store_classes(ec, c);
    experts = std::make_unique<ds41::OffloadStore>(std::move(records), ec);
    vram_checkpoint("native_after_store");
    if (!c.placement.empty()) {
      auto placement = ds41::load_placement(c.placement);
      if (mtp_experts) {
        // The MTP experts are already pinned in their own store.
        auto mtp = [](ds41::ExpertId id) { return id >= 15360; };
        auto &v = placement.vram_pins, &hr = placement.host_residents;
        v.erase(std::remove_if(v.begin(), v.end(), mtp), v.end());
        hr.erase(std::remove_if(hr.begin(), hr.end(), mtp), hr.end());
        for (int l = 40; l < 43; ++l)
          placement.dynamic_slots_per_layer[l] = placement.host_slots_per_layer[l] = 0;
      }
      if (exl3_mixed) {
        // s0ma change 16: raising a small width class (paid in a wider class's bytes) can leave fewer slots than the
        // placement's pins + budgets. The budgets are soft per-layer targets: trim the largest by one until they fit.
        size_t planned = placement.vram_pins.size();
        for (auto n : placement.dynamic_slots_per_layer) planned += n;
        auto &d = placement.dynamic_slots_per_layer;
        size_t trimmed = 0;
        while (planned > ec.slots) {
          auto big = std::max_element(d.begin(), d.begin() + 40);
          if (*big == 0) break;
          --*big; --planned; ++trimmed;
        }
        if (trimmed) std::cerr << "mixed EXL3 store: placement budgets trimmed by " << trimmed << " slots to the store's " << ec.slots << "\n";
        // s0ma change 23: and grown, in proportion to each layer's budget, when the store has more (DS41_EXPERT_EXTRA_GIB)
        if (planned < ec.slots) {
          std::array<size_t, 43> pinsl{};
          for (auto id : placement.vram_pins) ++pinsl[id < 15360 ? id / 384 : 40 + (id - 15360) / 128];
          size_t room = ec.slots - planned, total = 0, grown = 0;
          for (int l = 0; l < 40; ++l) total += d[l];
          for (int l = 0; l < 40 && total; ++l) {
            const size_t add = std::min(room * d[l] / total, 384 - pinsl[l] - d[l]);
            d[l] += add; grown += add;
          }
          for (int l = 0; grown < room && l < 40; ++l) if (d[l] + pinsl[l] < 384) { ++d[l]; ++grown; }
          std::cerr << "mixed EXL3 store: placement budgets grown by " << grown << " slots to the store's " << ec.slots << "\n";
        }
      }
      experts->place(placement);
    }
#if DS41_MOE_SOURCE_API>=4
    if(c.reset_cache_history) experts->reset_history();
#endif
    for (int j = 0; j < 200; ++j)
      phases.push_back(std::make_unique<Phase>());
    if(c.hotpath_profile) {
      for(int i=0;i<1440;++i) expert_events.push_back(std::make_unique<Phase>());
      consumer_readiness=std::make_unique<hotpath::ConsumerReadiness>();
    }
    reset();
    if (rows_dense) check_rows_batched();
    std::cerr << "ds41d native M1 loaded; UNQUALIFIED; expert slots="
              << c.expert_slots << (exl3_mixed ? " (EXL3 mixed target experts: " + mixed_k3_layers + "; native MTP store)"
                                    : exl3 ? (exl3_bits_loaded == 2 ? " (EXL3 K2 target experts, native MTP store)"
                                                   : " (EXL3 K3 target experts, native MTP store)") : "") << "\n";
  }
  ~Native() override {
    if (stream)
      cudaStreamSynchronize(stream);
    if (lane_stream)
      cudaStreamSynchronize(lane_stream);
    for (auto &ticket : engram_tickets)
      ticket.reset();
    for (auto &ticket : lane_engram_tickets)
      ticket.reset();
    for(auto &f:retirement_futures) if(f.valid()) {
      try { f.get(); } catch(...) { /* verify already poisons/reports errors */ }
    }
    retirement_futures.clear();
    async_engram.reset();
    graphs.clear(); // Captured consumers must die before the store/table.
    fused.reset();
    experts.reset();
    mtp_experts.reset();
    reader.reset();
    pages.reset();
    host.reset();
    if (blas)
      cublasDestroy(blas);
    if (lane_blas)
      cublasDestroy(lane_blas);
    lane_slots.clear();
    for (auto e : lane_att_events) if (e) cudaEventDestroy(e);
    for (auto e : route_events) if (e) cudaEventDestroy(e);
    if (lane_entry) cudaEventDestroy(lane_entry);
    if (lane_exit) cudaEventDestroy(lane_exit);
    if (lane_stream)
      cudaStreamDestroy(lane_stream);
    if (stream)
      cudaStreamDestroy(stream);
  }
  int vocab() const override { return V; }
  int eos() const override { return eos_id; }
  ds41_target_weights_v2 shared_weights() const override {
    return {2,
            sizeof(ds41_target_weights_v2),
            V,
            H,
            H,
            H,
            weights.at("embed.weight").data.p,
            weights.at("head.weight").data.as<float>(),
            stream,
            blas,
            rope.as<float>(),
            config.max_context};
  }
  ds41_target_services_v1 services() override {
    return {1, this,
            [](void *user, int stage, const void *input, const int32_t *ids,
               const float *weights, void *output, int rows, void *s) -> int {
              auto *self = static_cast<Native *>(user);
              return self->service_call([&] {
                self->routed_spark(stage, input, ids, weights, output, rows, s);
              });
            },
            [](void *user, int64_t position, void *s) -> int {
              auto *self = static_cast<Native *>(user);
              return self->service_call([&] {
                if (s != self->stream)
                  throw std::invalid_argument(
                      "target rollback stream mismatch");
                self->rollback(position);
              });
            },
            [](void *user) -> const char * {
              return static_cast<Native *>(user)->callback_error.c_str();
            },
            // s0ma change 43: the drafter's logits through Sero's head when it replaced the native one
            dense_of("head") ? +[](void *user, const void *x, float *out, int rows, void *s) -> int {
              auto *self = static_cast<Native *>(user);
              return self->service_call([&] { self->sero_head(x, out, rows, static_cast<cudaStream_t>(s)); });
            } : nullptr};
  }
  ds41_target_tensor_v1
  resident_tensor(const std::string &name) const override {
    ds41_target_tensor_v1 view{};
    view.abi_version = 1;
    view.struct_bytes = sizeof(view);
    if (name == "rope") {
      view.dtype = DS41_TARGET_F32;
      view.rows = config.max_context;
      view.columns = 64;
      view.data = rope.p;
      return view;
    }
    auto found = weights.find(name);
    if (found == weights.end())
      return {};
    const auto &t = found->second;
    if (t.shape.empty() || t.shape.size() > 2)
      throw std::runtime_error("resident tensor rank unsupported: " + name);
    view.rows = int64_t(t.shape[0]);
    view.columns = t.shape.size() == 2 ? int64_t(t.shape[1]) : 1;
    view.dtype =
        t.dtype == "BF16"                                  ? DS41_TARGET_BF16
        : t.dtype == "F32"                                 ? DS41_TARGET_F32
        : (t.dtype == "F8_E4M3" || t.dtype == "F8_E4M3FN") ? DS41_TARGET_FP8
        : (t.dtype == "U8" || t.dtype == "F8_E8M0" || t.dtype == "F8_E8M0FNU")
            ? DS41_TARGET_U8
            : -1;
    if (view.dtype < 0)
      throw std::runtime_error("resident tensor dtype unsupported: " + name);
    view.data = t.data.p;
    return view;
  }
  std::vector<std::string>
  resident_names(const std::string &prefix) const override {
    std::vector<std::string> names;
    for (const auto &t : weights)
      if (t.first.compare(0, prefix.size(), prefix) == 0)
        names.push_back(t.first);
    if (std::string("rope").compare(0, prefix.size(), prefix) == 0)
      names.push_back("rope");
    return names;
  }
  ds41_target_capture_v1 captures() const override {
    return {1,
            capture_rows,
            3 * H,
            capture_committed,
            capture_begin,
            size_t(3 * H * 2),
            capture_slab.p,
            stream};
  }
  void set_capture_sink(ds41_target_capture_sink sink, void *user) override {
    capture_sink = sink;
    capture_user = user;
  }
  void rollback(int64_t position) override {
    if (!pending) {
      if (position != committed)
        throw std::invalid_argument("rollback beyond pending verify is unsafe");
      return;
    }
    if (position < committed || position > committed + pending)
      throw std::invalid_argument("rollback outside pending verify");
    commit(size_t(position - committed));
  }
  uint64_t prefetch_experts(const std::vector<uint32_t> &ids, PrefetchIntent intent) override {
    for(auto id:ids) if(id>=ds41::expert_count) throw std::invalid_argument("prefetch logical expert id");
    if(mtp_experts && std::any_of(ids.begin(),ids.end(),[](uint32_t id){return id>=15360;})) {
      std::vector<uint32_t> target;
      for(auto id:ids) if(id<15360) target.push_back(id);
      return target.empty()?0:prefetch_experts(target,intent);
    }
    try {
      for(auto it=prefetch_tickets.begin();it!=prefetch_tickets.end();) {
        if(it->completion.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
          it->completion.get(); it=prefetch_tickets.erase(it);
        } else ++it;
      }
      auto priority=intent==PrefetchIntent::NextLayer?ds41::OffloadStore::PrefetchPriority::NextLayer:
          intent==PrefetchIntent::Draft?ds41::OffloadStore::PrefetchPriority::Draft:
          ds41::OffloadStore::PrefetchPriority::Session;
      auto ticket=experts->prefetch(ids,priority);
      auto id=ticket.id; prefetch_tickets.push_back(std::move(ticket)); return id;
    } catch(...) { poisoned=true; throw; }
  }
  // RAM-resident subset only: a wrong guess costs one host-to-device copy, never an NVMe read.
  void prefetch_ram_experts(const std::vector<ds41::ExpertId> &ids) {
    try {
      for(auto it=prefetch_tickets.begin();it!=prefetch_tickets.end();) {
        if(it->completion.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
          it->completion.get(); it=prefetch_tickets.erase(it);
        } else ++it;
      }
      prefetch_tickets.push_back(experts->prefetch_ram(ids,ds41::OffloadStore::PrefetchPriority::NextLayer));
    } catch(...) { poisoned=true; throw; }
  }
  void cancel_prefetch(uint64_t id) override {
    for(const auto &ticket:prefetch_tickets) if(ticket.id==id) experts->cancel(ticket);
  }
  void begin_request() override {
    if (poisoned)
      throw std::runtime_error("native forward/store poisoned; restart ds41d "
                               "before further model work");
    if (pending)
      throw std::logic_error("begin request during active verify");
    // Lookahead prefetch is per token. Settle it so served append prefill, which
    // refuses outstanding prefetch, stays available for the next prompt.
    if (lookahead_sync) {
      for (auto &ticket : prefetch_tickets) {
        experts->cancel(ticket);
        ticket.completion.wait();
      }
      prefetch_tickets.clear();
    }
    io_ms = route_ms = engram_ms = total_ms = 0;
    measured.clear();prefill_summary=Json::object();
    host_plan_ms=callback_enqueue_ms=0; profile_steps=Json::array();
    stream_callbacks=stream_hit_batches=0;
    initial_stats = sample_expert_stats();
    if (async_engram)
      initial_engram_stats = async_engram->stats();
    initial_replays = graph_replays;
    initial_captures = graph_captures;
    initial_evictions = graph_evictions;
  }
  struct NativeSnapshot : Snapshot {
    std::vector<Device> windows;
    std::map<int, std::pair<Device, Device>> raw;
    std::vector<int> prefix_ids;
  };
  std::shared_ptr<Snapshot> snapshot() override {
    if (poisoned || pending)
      throw std::logic_error("snapshot requires a committed, healthy state");
    auto s = std::make_shared<NativeSnapshot>();
    s->position = committed;
    for (auto &w : windows) {
      s->windows.emplace_back(w.bytes);
      CK(cudaMemcpyAsync(s->windows.back().p, w.p, w.bytes, cudaMemcpyDeviceToDevice, stream));
    }
    for (auto &[l, g] : globals) {
      auto &r = s->raw[l];
      r.first.resize(g->raw_kv.bytes); r.second.resize(g->raw_score.bytes);
      CK(cudaMemcpyAsync(r.first.p, g->raw_kv.p, g->raw_kv.bytes, cudaMemcpyDeviceToDevice, stream));
      CK(cudaMemcpyAsync(r.second.p, g->raw_score.p, g->raw_score.bytes, cudaMemcpyDeviceToDevice, stream));
    }
    s->prefix_ids = prefix.ids();
    CK(cudaStreamSynchronize(stream));
    return s;
  }
  void restore(const Snapshot &base) override {
    auto *s = dynamic_cast<const NativeSnapshot *>(&base);
    if (!s || s->windows.size() != windows.size() || s->position > committed ||
        int64_t(s->prefix_ids.size()) != s->position)
      throw std::invalid_argument("restore: incompatible snapshot");
    begin_request();
    for (size_t i = 0; i < windows.size(); ++i)
      CK(cudaMemcpyAsync(windows[i].p, s->windows[i].p, windows[i].bytes, cudaMemcpyDeviceToDevice, stream));
    for (auto &[l, g] : globals) {
      auto &r = s->raw.at(l);
      CK(cudaMemcpyAsync(g->raw_kv.p, r.first.p, g->raw_kv.bytes, cudaMemcpyDeviceToDevice, stream));
      CK(cudaMemcpyAsync(g->raw_score.p, r.second.p, g->raw_score.bytes, cudaMemcpyDeviceToDevice, stream));
    }
    committed = s->position;
    gate_rows.clear();
    for (auto &ticket : prefetch_tickets) experts->cancel(ticket);
    prefetch_tickets.clear();
    for (auto &routes : previous_routes) routes.clear();
    hash.truncate(size_t(s->position));
    prefix.reset(); prefix.commit(s->prefix_ids, s->prefix_ids.size()); pending_ids.clear();
    capture_rows = capture_committed = 0; capture_begin = 0;
    CK(cudaStreamSynchronize(stream));
  }
  void reset() override {
    begin_request();
    committed = 0;
    gate_rows.clear();
    for(auto &ticket:prefetch_tickets) experts->cancel(ticket);
    prefetch_tickets.clear();
    for(auto &routes:previous_routes) routes.clear();
    hash.reset();
    prefix.reset();pending_ids.clear();
    capture_rows = capture_committed = 0;
    capture_begin = 0;
    for (auto &w : windows)
      CK(cudaMemsetAsync(w.p, 0, w.bytes, stream));
    // Position-addressed global bytes need no clearing. Visibility masks and
    // completed compression groups prevent any future record from being read.
    CK(cudaStreamSynchronize(stream));
  }
  std::vector<std::vector<float>>
  verify(const std::vector<int> &input) override {
    if (poisoned)
      throw std::runtime_error("native forward/store poisoned; restart ds41d");
    if (pending || input.empty() || input.size() > 6 ||
        committed + int64_t(input.size()) > config.max_context)
      throw std::invalid_argument("verify state/width/context");
    for (int t : input)
      if (t < 0 || t >= V)
        throw std::invalid_argument("input id");
    pending = int(input.size());
    pending_ids = input;
    capture_begin = committed;
    capture_rows = pending;
    capture_committed = 0;
    std::vector<std::vector<float>> out;
    vlogits_rows = 0;
    const double verify_begin=config.hotpath_profile?clock_ms():0;
    try {
      check_moe_retirement(false);
      if (config.batched_verify)
        out = config.verify_lanes && input.size() >= 2 ? batch_lanes(input, committed) : batch(input, committed);
      else for (size_t i = 0; i < input.size(); ++i) {
        capture_slot = int(i);
        out.push_back(one(input[i], committed + i));
      }
      if (config.batched_verify && vlogits.p) vlogits_rows = int(out.size());
      if (out.size() < input.size()) {   // s0ma change 47 (H1): rows were cut; only the kept ones can be committed
        pending = int(out.size());
        pending_ids.resize(out.size());
        capture_rows = pending;
        // the acceptance kernel reads a bonus row after the kept ones: the last kept row stands in (its token is dropped)
        if (vlogits_rows)
          CK(cudaMemcpyAsync(static_cast<float *>(vlogits.p) + out.size() * size_t(V),
                             static_cast<float *>(vlogits.p) + (out.size() - 1) * size_t(V), size_t(V) * 4,
                             cudaMemcpyDeviceToDevice, stream));
      }
      const double retire_begin=config.hotpath_profile?clock_ms():0;
      check_moe_retirement(true);
      if(config.hotpath_profile && config.batched_verify && !profile_steps.empty()) {
        auto &step=profile_steps.back();
        step["verify_wall_ms"]=clock_ms()-verify_begin;
        step["retirement_registration_host_ms"]=clock_ms()-retire_begin;
      }
    } catch (...) {
      poisoned = true;
      throw;
    }
    return out;
  }
  const float *device_logits(int &rows) const override {
    rows = vlogits_rows;
    return vlogits_rows ? static_cast<const float *>(vlogits.p) : nullptr;
  }
  void commit(size_t accepted) override {
    if (poisoned)
      throw std::runtime_error("native forward/store poisoned; restart ds41d");
    if (!pending || accepted > size_t(pending))
      throw std::invalid_argument("accepted input prefix");
    prefix.commit(pending_ids,accepted);pending_ids.clear();
    committed += accepted;
    capture_committed = int(accepted);
    pending = 0;
    if (accepted && capture_sink) {
      try {
        auto view = captures();
        int rc = capture_sink(capture_user, &view);
        if (rc)
          throw std::runtime_error("DSpark capture sink failed: " +
                                   std::to_string(rc));
      } catch (...) {
        poisoned = true;
        throw;
      }
    }
  }
  DemandCounters demand_counters() const override {
    const auto s = sample_expert_stats();
    if(s.host_hits < initial_stats.host_hits ||
       s.nvme_misses < initial_stats.nvme_misses)
      throw std::runtime_error("target tier counter moved backwards");
    return {double(s.host_hits - initial_stats.host_hits),
            double(s.nvme_misses - initial_stats.nvme_misses)};
  }
  Json counters() const override {
    ds41::Stats s{};bool stale=false;std::string stats_error;
    try { s=sample_expert_stats(); }
    catch(const std::exception &e) {
      if(!expert_stats_available) throw;
      s=last_expert_stats;stale=true;stats_error=e.what();
    }
    Json j = {{"engine_mode", config.batched_verify ? "native-layer-major-verify-UNQUALIFIED" : "native-sequential-M1-UNQUALIFIED"},
              {"committed", committed},
              {"cache_epoch", prefix.epoch()},
              {"cache_committed_tokens", prefix.size()},
              {"expert_vram_hits", s.hits - initial_stats.hits},
              {"expert_host_hits", s.host_hits - initial_stats.host_hits},
              {"expert_nvme_misses", s.nvme_misses - initial_stats.nvme_misses},
              {"expert_h2d_bytes", s.h2d_bytes - initial_stats.h2d_bytes},
              {"expert_nvme_bytes", s.nvme_bytes - initial_stats.nvme_bytes},
              {"expert_spec_reads", s.spec_reads - initial_stats.spec_reads},     // s0ma change 8
              {"expert_spec_hits", s.spec_hits - initial_stats.spec_hits},
              {"expert_spec_waits", s.spec_waits - initial_stats.spec_waits},
              {"expert_spec_dropped", s.spec_dropped - initial_stats.spec_dropped},
              {"need_host_ms", io_ms},
              {"h1_cut_steps", h1_cuts}, {"h1_rows_cut", h1_rows_cut},   // s0ma change 47
              {"route_wait_ms", route_ms},
              {"engram_read_ms", engram_ms},
              {"forward_ms", total_ms},
              {"stream_phase_ms_including_stalls", measured}};
    for (size_t k = 0; k < class_widths.size() && k < 8; ++k) {   // s0ma change 18: demands by width class
      const std::string c = "expert_class_K" + std::to_string(class_widths[k]) + "_";
      j[c + "requests"] = s.class_requests[k] - initial_stats.class_requests[k];
      j[c + "vram_hits"] = s.class_hits[k] - initial_stats.class_hits[k];
      j[c + "host_hits"] = s.class_host_hits[k] - initial_stats.class_host_hits[k];
      j[c + "nvme_misses"] = s.class_nvme_misses[k] - initial_stats.class_nvme_misses[k];
    }
    {
      // DSpark drafter (MTP layers 40..42) share of the store traffic.
      uint64_t req=0,hit=0,host=0,nvme=0;
      for(int l=40;l<43;++l){
        const auto &a=s.layers[l], &b=initial_stats.layers[l];
        req+=a.requests-b.requests; hit+=a.hits-b.hits; host+=a.host_hits-b.host_hits; nvme+=a.nvme_misses-b.nvme_misses;
      }
      j["mtp_expert_requests"]=req; j["mtp_expert_vram_hits"]=hit;
      j["mtp_expert_host_hits"]=host; j["mtp_expert_nvme_misses"]=nvme;
    }
    j["fuse_library"]=config.fuse_library;
    j["moe_source_api"]=DS41_MOE_SOURCE_API;
    j["moe_stats_snapshot"]=stale?"last successful snapshot after error":"fresh";
    if(stale) j["moe_stats_error"]=stats_error;
    j["cpu_only_experts"]=0; // CPU backend is not integrated in this adapter.
    j["expert_kernel"]=exl3?"grouped_exl3 (target), grouped_mxfp4_native_parallel (MTP)":"grouped_mxfp4_native_parallel";
    j["target_expert_format"]=exl3_mixed?"EXL3 mixed mul1 ("+mixed_k3_layers+")":exl3?(exl3_bits_loaded==2?"EXL3 K2 mul1":"EXL3 K3 mul1"):"MXFP4";
    if(mtp_experts) j["mtp_expert_store"]="native MXFP4, 384 VRAM pins";
    j["xk_enabled"]=false; // Cache-only adapter never constructs DecodeBackend/XK.
    j["moe_demand_grouping"]="sorted unique logical-ID union per layer/verify";
    j["dynamic_host"]=config.dynamic_host;j["async_retire"]=config.async_retire;
    j["host_cache_policy"]=host_cache_policy;
    j["cache_history_initialization"]=config.reset_cache_history?
        "reset_history after placement; contents and pins retained":"placement seed histories retained";
    j["moe_delta"]=hotpath::moe_delta(initial_stats,s);
    j["hotpath_profile"]=config.hotpath_profile;
    j["step_profile"]=profile_steps;
    j["host_route_plan_ms"]=host_plan_ms;
    j["callback_enqueue_ms"]=callback_enqueue_ms;
    Json layer_stats=Json::array();
    for(int l=0;l<43;++l) {
      const auto &a=s.layers[l], &b=initial_stats.layers[l];
      layer_stats.push_back({{"layer",l},{"requests",a.requests-b.requests},
          {"vram_hits",a.hits-b.hits},{"host_hits",a.host_hits-b.host_hits},
          {"nvme_misses",a.nvme_misses-b.nvme_misses},{"stall_ns",a.stall_ns-b.stall_ns},
          {"h2d_bytes",a.h2d_bytes-b.h2d_bytes},{"nvme_bytes",a.nvme_bytes-b.nvme_bytes}});
    }
    j["expert_layers"]=layer_stats;
    j["streamed_experts"]=config.streamed_experts;
    j["stream_callbacks"]=stream_callbacks;
    j["stream_hit_batches"]=stream_hit_batches;
    j["prefetch_previous_routes"]=config.prefetch_routes;
    j["graphs_requested"] = config.graphs;
    j["prefill"] = prefill_summary;
    if (!config.hotpath_profile)
      j["prefill"].erase("layers");
    j["auxiliary_receipt_sha256"] = DS41_AUX_SHA256;
    j["mtp_loaded"] = config.load_mtp;
    uint64_t resident_bytes = 0, mtp_bytes = 0;
    for (const auto &item : weights) {
      resident_bytes += item.second.data.bytes;
      if (item.first.compare(0, 4, "mtp.") == 0)
        mtp_bytes += item.second.data.bytes;
    }
    j["resident_weight_bytes"] = resident_bytes;
    if (config.report_vram) j["vram_inventory"] = vram_inventory();
    j["mtp_resident_bytes"] = mtp_bytes;
    j["graph_count"] = graphs.size();
    j["graph_captures"] = graph_captures - initial_captures;
    j["graph_evictions"] = graph_evictions - initial_evictions;
    j["graph_replays"] = graph_replays - initial_replays;
    if (ds41_pdl_mode() > 0) {   // s0ma change 32
      j["pdl_mode"] = ds41_pdl_mode();
      j["pdl_graphs"] = pdl_graphs;
      j["pdl_edges"] = pdl_edges;
      j["pdl_refused"] = pdl_off;
      if (!pdl_error.empty()) j["pdl_error"] = pdl_error;
      j["pdl_skipped"] = pdl_skipped;   // s0ma change 38: graphs with allocation nodes, left as captured
      if (!pdl_alloc_note.empty()) j["pdl_alloc_note"] = pdl_alloc_note;
    }
    if (async_engram) {
      auto stats = async_engram->stats();
      for (const char *key : {"submitted_rows", "cache_hits", "unique_misses",
                              "disk_pages", "disk_bytes", "batches"})
        stats[key] = stats[key].get<uint64_t>() -
                     initial_engram_stats.at(key).get<uint64_t>();
      for (const char *key : {"io_ms", "total_ms"})
        stats[key] = stats[key].get<double>() -
                     initial_engram_stats.at(key).get<double>();
      j["engram_store"] = std::move(stats);
    }
    return j;
  }

  Json vram_inventory() const {
    Json j={{"schema","ds41-vram-census-v1"},{"max_context",config.max_context},
       {"host_embedding",config.host_embedding},{"reuse_woa",config.reuse_woa}};
    j["weights"]=Json::array();
    uint64_t weight_gpu=0,weight_host=0,arena=0,kv_bytes=0;
    for (const auto &entry:weights) {
      const auto &w=entry.second;
      bool mapped=w.data.mapped_host!=nullptr;
      j["weights"].push_back({{"name",entry.first},{"bytes",w.data.bytes},
          {"dtype",w.dtype},{"location",mapped?"mapped_pinned_host":"device"}});
      (mapped?weight_host:weight_gpu)+=w.data.bytes;
    }
    j["arenas"]=Json::object();
    for (auto entry : std::initializer_list<std::pair<const char*,const Device*>>{{"h",&h},{"next_h",&next_h},{"x",&x},{"y",&y},{"f32",&f32},{"pre",&pre},{"att_pre",&att_pre},{"ffn_pre",&ffn_pre},{"post",&post},{"comb",&comb},{"mixes",&mixes},{"rstd",&rstd},{"qa",&qa},{"q",&q},{"kv",&kv},{"latent",&latent},{"ik",&ik},{"iq",&iq},{"iw",&iw},{"packed",&packed},{"gather",&gather},{"local_idx",&local_idx},{"o",&o},{"wo",&wo},{"attn_scores",&attn_scores},{"aq",&aq},{"as",&as},{"gate",&gate},{"up",&up},{"act",&act},{"eq",&eq},{"es",&es},{"down",&down},{"shared",&shared},{"sg",&sg},{"su",&su},{"sa",&sa},{"jobs",&jobs},{"up_jobs",&up_jobs},{"down_jobs",&down_jobs},{"ids_device",&ids_device},{"route",&route},{"router",&router},{"unbiased",&unbiased},{"biased",&biased},{"logits",&logits},{"scores",&scores},{"top",&top},{"visible",&visible},{"position_device",&position_device},{"blocks",&blocks},{"block_visible",&block_visible},{"chosen_blocks",&chosen_blocks},{"candidate_ids",&candidate_ids},{"candidate_top",&candidate_top},{"top_scratch",&top_scratch},{"rows_raw",&rows_raw},{"scales_raw",&scales_raw},{"rows_bf",&rows_bf},{"engram_kv",&engram_kv},{"rows_interleaved",&rows_interleaved},{"braw_kv",&braw_kv},{"braw_score",&braw_score},{"batch_tokens",&batch_tokens},{"route_source",&route_source},{"route_slots",&route_slots},{"bx",&bx},{"bxs",&bxs},{"broute",&broute},{"compact_down",&compact_down},{"sparse_scores",&sparse_scores},{"work_jobs",&work_jobs},{"work_up_jobs",&work_up_jobs},{"work_down_jobs",&work_down_jobs},{"work_rows",&work_rows},{"batch_frequencies",&batch_frequencies},{"compressed_frequencies",&compressed_frequencies},{"lengths1",&lengths1},{"lengths2",&lengths2},{"packed_keys",&packed_keys},{"packed_globals",&packed_globals},{"gate_slab",&gate_slab},{"rope",&rope},{"woa_scratch",&woa_scratch},{"capture_slab",&capture_slab},{"spark_aq",&spark_aq},{"spark_as",&spark_as},{"spark_gate",&spark_gate},{"spark_up",&spark_up},{"spark_act",&spark_act},{"spark_eq",&spark_eq},{"spark_es",&spark_es},{"spark_jobs",&spark_jobs},{"spark_up_jobs",&spark_up_jobs},{"spark_down_jobs",&spark_down_jobs}}) {
      j["arenas"][entry.first]=entry.second->bytes; arena+=entry.second->bytes;
    }
    j["attention"]=Json::object();
    uint64_t window=0;
    for (const auto &w:windows) window+=w.bytes;
    j["attention"]["swa_windows"]=window; kv_bytes+=window;
    for (const auto &entry:globals) {
      const auto &g=*entry.second; uint64_t n=g.kv.bytes+g.key.bytes+g.raw_kv.bytes+g.raw_score.bytes;
      j["attention"][std::to_string(entry.first)]={{"kv",g.kv.bytes},{"indexer_key",g.key.bytes},
          {"raw_kv",g.raw_kv.bytes},{"raw_score",g.raw_score.bytes}};
      kv_bytes+=n;
    }
    j["weight_device_bytes"]=weight_gpu; j["weight_mapped_host_bytes"]=weight_host;
    j["arena_device_bytes"]=arena; j["attention_device_bytes"]=kv_bytes;
    uint64_t prefill_bytes=0;
    j["prefill_arenas"]=Json::object();
    if (pref) for (auto entry : std::initializer_list<std::pair<const char*,const Device*>>{{"pref.hidden",&pref->hidden},{"pref.previous_pre",&pref->previous_pre},{"pref.ffn_input",&pref->ffn_input},{"pref.ffn_post",&pref->ffn_post},{"pref.ffn_comb",&pref->ffn_comb},{"pref.window",&pref->window},{"pref.selected",&pref->selected},{"pref.candidates",&pref->candidates},{"pref.ids",&pref->ids},{"pref.weights",&pref->weights},{"pref.quant",&pref->quant},{"pref.scales",&pref->scales},{"pref.routed",&pref->routed},{"pref.capture",&pref->capture},{"pref.bx",&pref->bx},{"pref.bs",&pref->bs},{"pref.bw",&pref->bw},{"pref.gate",&pref->gate},{"pref.up",&pref->up},{"pref.act",&pref->act},{"pref.eq",&pref->eq},{"pref.es",&pref->es},{"pref.down",&pref->down}}) {
      j["prefill_arenas"][entry.first]=entry.second->bytes;prefill_bytes+=entry.second->bytes;
    }
    if (prefill_batch) for (auto entry : std::initializer_list<std::pair<const char*,const Device*>>{{"prefill_batch.h",&prefill_batch->h},{"prefill_batch.next",&prefill_batch->next},{"prefill_batch.x",&prefill_batch->x},{"prefill_batch.y",&prefill_batch->y},{"prefill_batch.pre",&prefill_batch->pre},{"prefill_batch.att_pre",&prefill_batch->att_pre},{"prefill_batch.ffn_pre",&prefill_batch->ffn_pre},{"prefill_batch.post",&prefill_batch->post},{"prefill_batch.comb",&prefill_batch->comb},{"prefill_batch.f32",&prefill_batch->f32},{"prefill_batch.mixes",&prefill_batch->mixes},{"prefill_batch.rstd",&prefill_batch->rstd},{"prefill_batch.qa",&prefill_batch->qa},{"prefill_batch.q",&prefill_batch->q},{"prefill_batch.kv",&prefill_batch->kv},{"prefill_batch.o",&prefill_batch->o},{"prefill_batch.wo",&prefill_batch->wo},{"prefill_batch.aq",&prefill_batch->aq},{"prefill_batch.as",&prefill_batch->as},{"prefill_batch.router",&prefill_batch->router},{"prefill_batch.unbiased",&prefill_batch->unbiased},{"prefill_batch.biased",&prefill_batch->biased},{"prefill_batch.sg",&prefill_batch->sg},{"prefill_batch.su",&prefill_batch->su},{"prefill_batch.sa",&prefill_batch->sa},{"prefill_batch.shared",&prefill_batch->shared}}) {
      j["prefill_arenas"][entry.first]=entry.second->bytes;prefill_bytes+=entry.second->bytes;
    }
    uint64_t prefill_stage_bytes=prefill_stage?prefill_stage->device_bytes():0;
    j["prefill_arena_device_bytes"]=prefill_bytes;
    j["prefill_staging_device_bytes"]=prefill_stage_bytes;
    j["target_device_requested_bytes"]=weight_gpu+arena+kv_bytes+prefill_bytes+prefill_stage_bytes;
    j["expert_slot_bytes"]=expert_slot_bytes; j["expert_slots"]=config.expert_slots;
    j["expert_payload_device_bytes"]=expert_payload_bytes+(mtp_experts?uint64_t(384)*18808832:0);
    j["moe_staging_pinned_host_bytes"]=uint64_t(8)*expert_slot_bytes+(mtp_experts?uint64_t(8)*18808832:0);
    j["engine_staging_pinned_host_bytes"]=(16+1)*(1<<20);
    j["graphs"]=graphs.size();
    size_t stack=0;CK(cudaDeviceGetLimit(&stack,cudaLimitStackSize));
    j["cuda_stack_bytes_per_thread"]=stack;
    size_t free=0,total=0; CK(cudaMemGetInfo(&free,&total));
    j["cuda_device_total_bytes"]=total;j["cuda_device_free_bytes"]=free;
    return j;
  }

private:
  template <class F> int service_call(F operation) noexcept {
    try {
      if (poisoned)
        throw std::runtime_error("native forward/store poisoned");
      operation();
      callback_error.clear();
      return 0;
    } catch (const std::exception &e) {
      callback_error = e.what();
    } catch (...) {
      callback_error = "unknown target callback failure";
    }
    poisoned = true;
    return -1;
  }
  void routed_spark(int stage, const void *input, const int32_t *ids,
                    const float *route_weights, void *output, int rows,
                    void *supplied_stream) {
    if (supplied_stream != stream || stage < 0 || stage >= 3 || rows < 1 ||
        rows > 8 || !input || !ids || !route_weights || !output)
      throw std::invalid_argument("DSpark routed adapter arguments/stream");
    if (pending)
      throw std::logic_error(
          "DSpark callback during target verify transaction");
    constexpr int max_rows = 8, max_jobs = 3 * max_rows;
    if (!spark_aq.p) {
      spark_aq.resize(max_rows * H);
      spark_as.resize(max_rows * H / 32);
      spark_gate.resize(max_jobs * I * 2);
      spark_up.resize(max_jobs * I * 2);
      spark_act.resize(max_jobs * I * 2);
      spark_eq.resize(max_jobs * I);
      spark_es.resize(max_jobs * I / 32);
      spark_jobs.resize(max_jobs * sizeof(ds41::GemmJob));
      spark_up_jobs.resize(max_jobs * sizeof(ds41::GemmJob));
      spark_down_jobs.resize(max_jobs * sizeof(ds41::GemmJob));
    }
    // One deliberate route boundary, then unique logical IDs under one lease.
    auto selected = static_cast<int32_t *>(host->data());
    double t = clock_ms();
    CK(cudaMemcpyAsync(selected, ids, rows * 3 * sizeof(int32_t),
                       cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    route_ms += clock_ms() - t;
    std::vector<ds41::ExpertId> logical(rows * 3), need;
    for (int r = 0; r < rows; ++r) {
      std::set<int32_t> unique_row;
      for (int k = 0; k < 3; ++k) {
        int id = selected[r * 3 + k];
        if (id < 0 || id >= 128 || !unique_row.insert(id).second)
          throw std::invalid_argument("invalid/duplicate DSpark route");
        logical[r * 3 + k] = ds41::expert_id(40 + stage, id);
      }
    }
    need = logical;
    std::sort(need.begin(), need.end());
    need.erase(std::unique(need.begin(), need.end()), need.end());
    if (!mtp_experts && need.size() > config.expert_slots)
      throw std::runtime_error(
          "DSpark unique demand exceeds expert slot budget");
    t = clock_ms();
    auto lease = spark_store().need_now(need, stream);
    io_ms += clock_ms() - t;
    try {
      int count = rows * 3;
      std::vector<ds41::GemmJob> job(count);
      for (int j = 0; j < count; ++j)
        job[j] = {logical[j], 1, spark_aq.as<uint8_t>((j / 3) * H),
                  spark_as.as<uint8_t>((j / 3) * H / 32),
                  spark_gate.as<Bf16>(j * I * 2)};
      // Job descriptors go through a pinned mapped buffer and a zero-copy kernel: a pageable H2D would
      // synchronize the stream before each copy. The buffer is rewritten only after the next route
      // readback, i.e. after these descriptors were consumed in stream order.
      constexpr size_t slab = size_t(max_jobs) * sizeof(ds41::GemmJob);
      if (!spark_host) spark_host = std::make_unique<ds41::HostBuffer>(3 * slab, true);
      auto *hj = static_cast<ds41::GemmJob *>(spark_host->data());
      auto *dj = static_cast<char *>(spark_host->device_data());
      std::copy(job.begin(), job.end(), hj);
      for (int j = 0; j < count; ++j) {
        hj[max_jobs + j] = job[j];
        hj[max_jobs + j].output = spark_up.as<Bf16>(j * I * 2);
        hj[2 * max_jobs + j] = {logical[j], 1, spark_eq.as<uint8_t>(j * I),
                                spark_es.as<uint8_t>(j * I / 32),
                                static_cast<Bf16 *>(output) + j * H};
      }
      zc(spark_jobs.p, dj, count * sizeof(ds41::GemmJob));
      zc(spark_up_jobs.p, dj + slab, count * sizeof(ds41::GemmJob));
      zc(spark_down_jobs.p, dj + 2 * slab, count * sizeof(ds41::GemmJob));
      CK(ds41::quantize_mxfp8(static_cast<const Bf16 *>(input),
                              spark_aq.as<uint8_t>(), spark_as.as<uint8_t>(),
                              rows, H, stream));
      CK(ds41::grouped_mxfp4_native_parallel(
          spark_store().table(), spark_jobs.as<ds41::GemmJob>(), count,
          ds41::Projection::Gate, I, H, stream));
      CK(ds41::grouped_mxfp4_native_parallel(
          spark_store().table(), spark_up_jobs.as<ds41::GemmJob>(), count,
          ds41::Projection::Up, I, H, stream));
      CK(ds41::swiglu(spark_gate.as<Bf16>(), spark_up.as<Bf16>(), route_weights,
                      spark_act.as<Bf16>(), count, I, 10, stream));
      CK(ds41::quantize_mxfp8(spark_act.as<Bf16>(), spark_eq.as<uint8_t>(),
                              spark_es.as<uint8_t>(), count, I, stream));
      CK(ds41::grouped_mxfp4_native_parallel(
          spark_store().table(), spark_down_jobs.as<ds41::GemmJob>(), count,
          ds41::Projection::Down, H, I, stream));
      // The store may recycle slots only after the down consumers complete.
      spark_store().retire(std::move(lease), stream);
    } catch (...) {
      cudaStreamSynchronize(stream);
      if (lease.owner)
        spark_store().retire(std::move(lease), stream);
      throw;
    }
  }
  template <class F> void graph(const std::string &key, F body) {
    if (!config.graphs || !graph_ready || !config.trace_dir.empty()) {
      body();
      return;
    }
    auto it = graphs.find(key);
    if (it != graphs.end()) {
      CK(cudaGraphLaunch(it->second->exec, stream));
      ++graph_replays;
      return;
    }
    auto recorded = std::make_unique<Graph>();
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      body();
    } catch (...) {
      cudaStreamEndCapture(stream, &recorded->graph);
      throw;
    }
    CK(cudaStreamEndCapture(stream, &recorded->graph));
    if (ds41_pdl_mode() < 0) throw std::invalid_argument("DS41_PDL takes 0, 1 or 2");
    if (const int mode = ds41_pdl_mode(); mode > 0 && !pdl_off) {
      // s0ma change 32: the captured graph's edges rewritten in place (cudaGraphClone is refused for captured graphs
      // here: "operation not supported"). A refusal stops the engine (change 35): a bench with DS41_PDL set would
      // otherwise run to its end measuring nothing new.
      size_t edges = 0;
      const char *step = "";
      cudaGraphInstantiateParams ip{};
      const int rewrite = ds41_pdl_rewrite(recorded->graph, mode, &edges, &step);
      if (rewrite == -100) {   // s0ma change 38: allocation nodes: this graph runs as captured, the others go on
        ++pdl_skipped;
        if (pdl_alloc_note.empty()) {
          pdl_alloc_note = std::string(key) + ": " + step;
          std::cerr << "DS41_PDL: graph " << pdl_alloc_note << " (memory allocation nodes; left as captured)\n";
        }
      }
      cudaError_t rc = rewrite == -100 ? cudaSuccess : cudaError_t(rewrite);
      if (rc == cudaSuccess && rewrite != -100) {
        step = "instantiating";
        rc = cudaGraphInstantiateWithParams(&recorded->exec, recorded->graph, &ip);
      }
      if (rc != cudaSuccess) {
        std::ostringstream why;
        why << step << ": " << cudaGetErrorString(rc) << "; nodes " << ds41_pdl_node_types(recorded->graph);
        if (ip.result_out != cudaGraphInstantiateSuccess) why << " (instantiate result " << int(ip.result_out) << ")";
        if (ip.errNode_out) why << " at kernel " << ds41_pdl_node_name(ip.errNode_out);
        pdl_error = why.str();
        pdl_off = true;
        cudaGetLastError();
        std::cerr << "DS41_PDL: graph " << key << " refused programmatic edges (" << pdl_error
                  << "); stopping (unset DS41_PDL to run without it)\n" << std::flush;
        throw std::runtime_error("DS41_PDL refused by the driver: " + pdl_error);
      }
      if (rewrite != -100) {
        pdl_edges += edges;
        ++pdl_graphs;
      }
    }
    if (!recorded->exec) CK(cudaGraphInstantiate(&recorded->exec, recorded->graph, 0));
    CK(cudaGraphLaunch(recorded->exec, stream));
    ++graph_replays;
    graphs.emplace(key, std::move(recorded));
    ++graph_captures;
  }
  void dump(int64_t pos, int layer, const char *tag, const Device &data,
            size_t elements) {
    if (config.trace_dir.empty())
      return;
    auto dir = std::filesystem::path(config.trace_dir) /
               ("pos" + std::to_string(pos)) /
               ("layer" + std::to_string(layer));
    std::filesystem::create_directories(dir);
    std::vector<uint16_t> raw(elements);
    CK(cudaMemcpyAsync(raw.data(), data.p, elements * 2, cudaMemcpyDeviceToHost,
                       stream));
    CK(cudaStreamSynchronize(stream));
    std::ofstream f(dir / (std::string(tag) + ".bf16"), std::ios::binary);
    f.write(reinterpret_cast<const char *>(raw.data()), raw.size() * 2);
    if (!f)
      throw std::runtime_error("failed diagnostic activation dump");
  }
  Weight &w(const std::string &n) {
    auto it = weights.find(n);
    if (it == weights.end())
      throw std::runtime_error("missing resident tensor: " + n);
    return it->second;
  }
  void upload(Device &d, const void *p, size_t n) {
    if (n > d.bytes)
      throw std::runtime_error("device upload overflow");
    if (in_step) {
      size_t aligned = (n + 63) & ~size_t(63);
      if (upload_cursor + aligned > host->size())
        throw std::runtime_error("per-step pinned metadata capacity");
      void *owned = static_cast<char *>(host->data()) + upload_cursor;
      std::memcpy(owned, p, n);
      p = owned;
      upload_cursor += aligned;
    }
    if (in_step && zc_batching) zc_pending.push_back({static_cast<const unsigned char *>(host_alias(p)), static_cast<unsigned char *>(d.p), n});
    else if (in_step) zc(d.p, host_alias(p), n);
    else CK(cudaMemcpyAsync(d.p, p, n, cudaMemcpyHostToDevice, stream));
  }
  // Group consecutive in-step uploads into one zero-copy launch (zc_begin ... zc_end). Only for sites
  // whose consumers are enqueued after zc_end().
  bool zc_batching = false;
  std::vector<ZcSeg> zc_pending;
  void zc_begin() { zc_batching = in_step; zc_pending.clear(); }
  void zc_end() {
    zc_batching = false;
    if (zc_pending.empty()) return;
    if (zc_pending.size() == 1) { zc(zc_pending[0].dst, zc_pending[0].src, zc_pending[0].n); zc_pending.clear(); return; }
    const size_t bytes = zc_pending.size() * sizeof(ZcSeg), aligned = (bytes + 63) & ~size_t(63);
    if (upload_cursor + aligned > host->size()) throw std::runtime_error("per-step pinned metadata capacity");
    void *table = static_cast<char *>(host->data()) + upload_cursor;
    std::memcpy(table, zc_pending.data(), bytes);
    upload_cursor += aligned;
    zc_multi<<<int(zc_pending.size()), 128, 0, stream>>>(static_cast<const ZcSeg *>(host_alias(table)), int(zc_pending.size()));
    CK(cudaGetLastError());
    zc_pending.clear();
  }
  // Device alias of an address inside the mapped pinned step buffer.
  void *host_alias(const void *p) const {
    return static_cast<char *>(host->device_data()) + (static_cast<const char *>(p) - static_cast<const char *>(host->data()));
  }
  void zc(void *dst, const void *src, size_t n) {
    if (!n) return;
    const int threads = 256;
    const int blocks = int(std::min<size_t>((n + 16 * threads - 1) / (16 * threads), 512));
    zc_copy<<<blocks, threads, 0, stream>>>(static_cast<const unsigned char *>(src), static_cast<unsigned char *>(dst), n);
    CK(cudaGetLastError());
  }
  template <class F> void phase(const std::string &name, F call) {
    if (phase_index >= phases.size())
      throw std::logic_error("phase capacity");
    auto &p = *phases[phase_index++];
    CK(cudaEventRecord(p.start, stream));
    call();
    CK(cudaEventRecord(p.end, stream));
    phase_names.push_back(name);
  }
  std::vector<std::string> phase_names;
  void load_weights() {
    size_t sum = 0;
    for (auto item : source.tensors()) {
      const auto &t = item.second;
      auto n = t.name;
      if (n.find(".experts.") != std::string::npos ||
          n.find(".engram.embed.") != std::string::npos ||
          (n.compare(0, 4, "mtp.") == 0 && !config.load_mtp) ||
          n.compare(0, 7, "vision.") == 0 || n.compare(0, 8, "aligner.") == 0 ||
          n.compare(0, 6, "image_") == 0)
        continue;
      bool promote =
          (n == "head.weight" && config.fused_head_library.empty()) ||
          n.find(".ffn.gate.weight") != std::string::npos ||
          n.find(".attn.compressor.wgate.weight") != std::string::npos ||
          n == "mtp.2.markov_head.head.weight" ||
          n == "mtp.2.confidence_head.proj.weight";
      if (n.find(".attn.compressor.wkv.weight") != std::string::npos &&
          n.compare(0, 10, "layers.20.") != 0)
        promote = true;
      auto bytes = source.read(t);
      Weight weight;
      weight.shape = t.shape;
      weight.dtype = t.dtype;
      if (n == "embed.weight" && config.host_embedding) {
        // Native BF16 bytes; GPU gather and Spark keep the same tensor view.
        weight.data.resize_mapped(t.bytes);
        std::memcpy(weight.data.mapped_host,bytes.data(),bytes.size());
      } else {
        weight.data.resize(t.bytes);
        upload(weight.data, bytes.data(), bytes.size());
      }
      CK(cudaStreamSynchronize(stream));
      if (promote && t.dtype == "BF16") {
        Device converted(t.bytes * 2);
        bf16_float(weight.data.p, converted.as<float>(), int(t.bytes / 2),
                   stream);
        weight.data = std::move(converted);
        weight.dtype = "F32";
      }
      sum += weight.data.bytes;
      weights.emplace(n, std::move(weight));
    }
    for (int l = 0; l < (config.load_mtp ? 43 : 40); ++l) {
      std::string p = (l < 40 ? "layers." + std::to_string(l)
                              : "mtp." + std::to_string(l - 40)) +
                      ".attn.wo_a.";
      auto &a = w(p + "weight");
      auto &s = w(p + "scale");
      if (a.dtype != "F8_E4M3" && a.dtype != "F8_E4M3FN")
        throw std::runtime_error(
            "wo_a must retain native FP8 source before exact BF16 dequant");
      if (config.reuse_woa && l < 40) {
        // Keep the authenticated FP8 source and original scales. Dequantize
        // one layer into a stream-ordered shared BF16 buffer before cuBLAS.
        if (!woa_scratch.bytes) woa_scratch.resize(a.shape[0]*a.shape[1]*2);
        if (woa_scratch.bytes != a.shape[0]*a.shape[1]*2)
          throw std::runtime_error("shared wo_a shape mismatch");
        continue;
      }
      Device dequant(a.shape[0] * a.shape[1] * 2);
      CK(ds41_dense_dequant(a.data.p, s.data.p, dequant.p, int(a.shape[0]),
                            int(a.shape[1]), stream));
      a.data = std::move(dequant);
      a.dtype = "BF16";
    }
    CK(cudaStreamSynchronize(stream));
    std::cerr << "resident source bytes " << sum
              << " (wo_a BF16 promotion additional)\n";
  }
  void load_rope() {
    auto meta = read_json(config.rope + ".json");
    if (meta.at("schema") != "ds41-rope-v1" ||
        meta.at("rows").get<int>() < config.max_context)
      throw std::runtime_error("RoPE metadata/context mismatch");
    size_t file_rows = meta.at("rows").get<size_t>();
    std::ifstream f(config.rope, std::ios::binary);
    if (!f)
      throw std::runtime_error("missing RoPE table");
    rope.resize(size_t(config.max_context) * 64 * 4 * 2);
    std::vector<float> part(size_t(config.max_context) * 64);
    for (int i = 0; i < 2; ++i) {
      f.seekg(i * file_rows * 64 * 4);
      f.read(reinterpret_cast<char *>(part.data()), part.size() * 4);
      if (!f)
        throw std::runtime_error("short RoPE table");
      CK(cudaMemcpyAsync(rope.as<char>(i * part.size() * 4), part.data(),
                         part.size() * 4, cudaMemcpyHostToDevice, stream));
      CK(cudaStreamSynchronize(stream));
    }
  }
  const float *freqs(int l, int64_t pos) const {
    return rope.as<float>((size_t(l >= 2) * config.max_context + pos) * 64 * 4);
  }
  const void *woa_weight(const std::string &prefix) {
    auto &weight=w(prefix+".weight");
    if (!config.reuse_woa || prefix.compare(0,7,"layers.") != 0)
      return weight.data.p;
    // This call is captured inside the layer output graph, so every replay
    // refreshes scratch on the same stream before its unchanged cuBLAS call.
    CK(ds41_dense_dequant(weight.data.p,w(prefix+".scale").data.p,
        woa_scratch.p,int(weight.shape[0]),int(weight.shape[1]),stream));
    return woa_scratch.p;
  }
  // s0ma change 34 (N1): Sero's dense EXL3 tensors. DS41_SERO_DENSE=1 reads <the Sero bank>/dense (ds41.sh sero dense),
  // or a directory. Taken: every mul1 tensor whose native counterpart is an FP8 [n, k] weight with a k the decode
  // kernels have (128, 256, 1280, 2304, 5120, 8192), wo_a as its eight 4096 -> 1024 slices, and not the head (its
  // logits would come out in BF16). DS41_SERO_DENSE_ONLY=a,b keeps only names containing one of those strings.
  void load_sero_dense() {
    const char *v = std::getenv("DS41_SERO_DENSE");
    if (!v || !*v || std::string(v) == "0") return;
    if (config.exl3_experts.empty()) throw std::invalid_argument("DS41_SERO_DENSE needs Sero's bank (--exl3-experts)");
    const std::filesystem::path dir = std::string(v) == "1"
        ? std::filesystem::path(config.exl3_experts).parent_path() / "dense" : std::filesystem::path(v);
    if (!std::filesystem::exists(dir / "dense.json"))
      throw std::runtime_error("DS41_SERO_DENSE: no " + (dir / "dense.json").string() + " (ds41.sh sero dense)");
    auto meta = read_json((dir / "dense.json").string());
    const auto file = dir / meta.at("file").get<std::string>();
    if (std::filesystem::file_size(file) != meta.at("bytes").get<uint64_t>())
      throw std::runtime_error("DS41_SERO_DENSE: " + file.string() + " is not the size dense.json states");
    std::vector<std::string> only;
    if (const char *o = std::getenv("DS41_SERO_DENSE_ONLY"); o && *o) {
      std::stringstream ss(o);
      for (std::string item; std::getline(ss, item, ',');) if (!item.empty()) only.push_back(item);
    }
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("DS41_SERO_DENSE: cannot open " + file.string());
    std::vector<ds41::DeviceExpert> table;
    size_t freed = 0, used = 0, skipped = 0;
    std::map<std::string, int> kinds;
    std::map<std::string, std::vector<const Json *>> woa_slices;   // "layers.L.attn.wo_a" -> its slices
    auto load_record = [&](const std::string &name, const Json &t) {
      const int k = t.at("k").get<int>(), n = t.at("n").get<int>(), bits = t.at("bits").get<int>();
      const size_t off = t.at("offset").get<uint64_t>(), bytes = t.at("bytes").get<uint64_t>(),
                   tb = t.at("trellis_bytes").get<uint64_t>();
      if (tb != size_t(k / 16) * (n / 16) * 16 * bits * 2 || bytes != tb + 2 * size_t(k) + 2 * size_t(n))
        throw std::runtime_error("DS41_SERO_DENSE: " + name + ": record sizes disagree with k, n and bits");
      std::vector<char> host(bytes);
      in.seekg(std::streamoff(off));
      if (!in.read(host.data(), std::streamsize(bytes)))
        throw std::runtime_error("DS41_SERO_DENSE: short read of " + name);
      Device record(bytes);
      CK(cudaMemcpy(record.p, host.data(), bytes, cudaMemcpyHostToDevice));
      ds41::DeviceExpert e{};
      e.gate = static_cast<const uint8_t *>(record.p);
      e.gate_scale = static_cast<const uint8_t *>(record.p) + tb;
      dense_exl3[name] = {int(table.size()), k, n, bits};
      table.push_back(e);
      dense_records.push_back(std::move(record));
      used += bytes;
      std::string kind = name;
      if (kind.compare(0, 7, "layers.") == 0) kind = kind.substr(kind.find('.', 7) + 1);
      if (const size_t sl = kind.find(".slice."); sl != std::string::npos) kind = kind.substr(0, sl) + " slices";
      ++kinds[kind + " K" + std::to_string(bits)];
    };
    for (const auto &t : meta.at("tensors")) {
      std::string name = t.at("name").get<std::string>();
      if (name.compare(0, 6, "model.") == 0) name = name.substr(6);
      const int k = t.at("k").get<int>(), n = t.at("n").get<int>(), bits = t.at("bits").get<int>();
      auto it = weights.find(name + ".weight");
      const bool fp8 = it != weights.end() && (it->second.dtype == "F8_E4M3" || it->second.dtype == "F8_E4M3FN");
      const bool shape = fp8 && it->second.shape == std::vector<uint64_t>({uint64_t(n), uint64_t(k)});
      const bool kernel = (k == 128 || k == 256 || k == 1280 || k == 2304 || k == 5120 || k == 8192) && n % 128 == 0 &&
                          bits >= 2 && bits <= 5;
      bool wanted = only.empty();
      for (const auto &o : only) wanted = wanted || name.find(o) != std::string::npos;
      // wo_a's eight groups come as eight slices (4096 -> 1024 each, K5): taken as a set, below
      if (const size_t at = name.find(".attn.wo_a.slice."); at != std::string::npos) {
        if (t.at("codebook").get<std::string>() == "mul1" && k == 4096 && n == 1024 && bits >= 2 && bits <= 5 && wanted)
          woa_slices[name.substr(0, at + 10)].push_back(&t);
        else ++skipped;
        continue;
      }
      // s0ma change 43: Sero's K4 head in place of the native BF16 one (DS41_SERO_HEAD=1): the target's logits and the
      // drafter's (through the services' head callback) both come from it, in FP32
      if (name == "head") {
        const char *hv = std::getenv("DS41_SERO_HEAD");
        auto hit = weights.find("head.weight");
        if (hv && std::string(hv) == "1" && t.at("codebook").get<std::string>() == "mul1" && k == H && n == V &&
            bits >= 2 && bits <= 5 && hit != weights.end() && hit->second.dtype == "BF16" &&
            hit->second.shape == std::vector<uint64_t>({uint64_t(V), uint64_t(H)})) {
          load_record("head", t);
          freed += hit->second.data.bytes;
          hit->second.data.release();
          hit->second.dtype = "EXL3";
        } else ++skipped;
        continue;
      }
      if (t.at("codebook").get<std::string>() != "mul1" || !shape || !kernel || !wanted ||
          weights.count(name + ".scale") == 0) {
        ++skipped;
        continue;
      }
      load_record(name, t);
      auto &weight = it->second, &scale = w(name + ".scale");
      freed += weight.data.bytes + scale.data.bytes;
      weight.data.release();
      scale.data.release();
      weight.dtype = "EXL3";   // dtype filters (FP8 scratch sizing, the rows check) now pass over it
    }
    // wo_a: a layer's eight slices replace its [8192, 4096] weight (BF16 after the load's promotion, or FP8 with
    // --reuse-woa), group g taking o's columns g*4096.. and writing wo's columns g*1024.. (ds41d's grouped wo_a)
    for (auto &[woa, slices] : woa_slices) {
      auto it = weights.find(woa + ".weight");
      std::map<int, const Json *> by_group;
      for (const Json *t : slices) {
        const std::string name = (*t).at("name").get<std::string>();
        by_group[std::stoi(name.substr(name.rfind('.') + 1))] = t;
      }
      if (it == weights.end() || it->second.shape != std::vector<uint64_t>({8192, 4096}) || by_group.size() != 8 ||
          by_group.begin()->first != 0 || by_group.rbegin()->first != 7) {
        skipped += slices.size();
        continue;
      }
      bool same_bits = true;
      for (auto &[g, t] : by_group) same_bits = same_bits && (*t).at("bits") == (*by_group.begin()->second).at("bits");
      if (!same_bits) { skipped += slices.size(); continue; }   // one launch runs them all at one width
      for (auto &[g, t] : by_group) load_record(woa + ".slice." + std::to_string(g), *t);   // consecutive entries
      freed += it->second.data.bytes;
      it->second.data.release();
      it->second.dtype = "EXL3";
      if (auto sc = weights.find(woa + ".scale"); sc != weights.end()) { freed += sc->second.data.bytes; sc->second.data.release(); }
      dense_woa.insert(woa.substr(0, woa.size() - 5));   // "layers.L.attn"
    }
    if (table.empty()) {
      std::cerr << "DS41_SERO_DENSE: none of the " << meta.at("tensors").size() << " tensors matched; native FP8 kept\n";
      return;
    }
    dense_table.resize(table.size() * sizeof(ds41::DeviceExpert));
    CK(cudaMemcpy(dense_table.p, table.data(), table.size() * sizeof(ds41::DeviceExpert), cudaMemcpyHostToDevice));
    // s0ma change 37: the split-K and prefill scratch, sized from the tensors taken
    std::map<std::string, size_t> layer_bytes;
    size_t widest_k = 0, widest_n = 0;
    for (const auto &[name, d] : dense_exl3) {
      if (name == "head") continue;   // s0ma change 43: decode only (sero_head): no split-K or prefill scratch
      const bool slice = name.find(".wo_a.slice.") != std::string::npos;
      dense_split_floats = std::max(dense_split_floats, ds41::exl3_dense_splitk_floats(d.n, d.k, slice ? 8 : 1));
      layer_bytes[name.substr(0, name.find('.', 7))] += (size_t(d.k) * d.n * 2 + 255) / 256 * 256;
      widest_k = std::max<size_t>(widest_k, d.k);
      widest_n = std::max<size_t>(widest_n, d.n);
    }
    size_t cache = 0;
    for (const auto &[layer, b] : layer_bytes) cache = std::max(cache, b);
    dense_split_partial[0].resize(std::max<size_t>(dense_split_floats, 1) * 4);
    dense_split_counts[0].resize(dense_split_slots * 4);
    CK(cudaMemset(dense_split_counts[0].p, 0, dense_split_slots * 4));
    dense_cache.resize(cache);
    dense_rows_a.resize(256 * widest_k * 2);
    dense_rows_c.resize(256 * widest_n * 4);
    used += dense_split_partial[0].bytes + dense_split_counts[0].bytes + dense_cache.bytes + dense_rows_a.bytes +
            dense_rows_c.bytes;
    dense_extra_gib = freed > used ? double(freed - used) / double(size_t(1) << 30) : 0;
    std::cerr << "Sero dense EXL3 (" << meta.at("source").get<std::string>() << "): " << table.size() << " tensors, "
              << used / 1e9 << " GB in place of " << freed / 1e9 << " GB of FP8 (" << skipped << " left native):";
    for (const auto &[kind, count] : kinds) std::cerr << " " << kind << " x" << count;
    std::cerr << "; " << dense_extra_gib << " GiB more expert slots\n";
  }
  const DenseExl3 *dense_of(const std::string &prefix) const {
    if (dense_exl3.empty()) return nullptr;
    auto it = dense_exl3.find(prefix);
    return it == dense_exl3.end() ? nullptr : &it->second;
  }
  // s0ma change 43: FP32 logits [m, V] from the final norm's BF16 rows [m, H] through Sero's K4 head, 8 rows a launch;
  // false (nothing enqueued) when the native head is in use
  bool sero_head(const void *in, void *out, int m, cudaStream_t s) {
    const DenseExl3 *d = dense_of("head");
    if (!d) return false;
    const auto *table = static_cast<const ds41::DeviceExpert *>(dense_table.p);
    for (int r = 0; r < m; r += 8) {
      ds41::GemmJob job{ds41::ExpertId(d->index), std::min(8, m - r), static_cast<const uint8_t *>(in) + size_t(r) * H * 2,
                        nullptr, reinterpret_cast<__nv_bfloat16 *>(static_cast<float *>(out) + size_t(r) * V)};
      CK(ds41::exl3_dense_f32(table, job, V, H, d->bits, s));
    }
    return true;
  }
  // y[m, n] = x[m, k] W^T through Sero's EXL3 tensor: the decode kernel up to 8 rows a launch (4 for k = 8192), the
  // prefill kernel above that, 256 rows a launch. Inputs and outputs BF16, as the FP8 path's.
  bool dense_linear(const std::string &prefix, const void *in, void *out, int m, int xs = 0, int os = 0) {
    const DenseExl3 *d = dense_of(prefix);
    if (!d) return false;
    if (!xs) xs = d->k;
    if (!os) os = d->n;
    const auto *table = static_cast<const ds41::DeviceExpert *>(dense_table.p);
    dense_since_quant = true;
    if (m > 8) { dense_prefill(prefix, *d, in, out, m, xs, os); return true; }
    // s0ma change 37: K split across CTAs for the narrow tensors (8 rows a launch then, whatever k)
    const bool split = ds41::exl3_dense_splitk_floats(d->n, d->k, 1) > 0;
    const int cap = split || d->k <= 5120 ? 8 : 4, lane = lane_b_active ? 1 : 0;
    for (int r = 0; r < m; r += cap) {
      ds41::GemmJob job{ds41::ExpertId(d->index), std::min(cap, m - r),
                        static_cast<const uint8_t *>(in) + size_t(r) * xs * 2, nullptr,
                        static_cast<__nv_bfloat16 *>(out) + size_t(r) * os};
      CK(ds41::exl3_dense(table, job, d->n, d->k, d->bits, stream, xs, os, 1, 0, 0,
                          dense_split_partial[lane].as<float>(), dense_split_partial[lane].bytes / 4,
                          dense_split_counts[lane].as<unsigned>(), dense_split_counts[lane].bytes / 4));
    }
    return true;
  }
  // s0ma change 37: prefill rows through Sero's tensor as exllamav3 runs many rows: its weights decoded to FP16 once
  // a layer, each 256 rows rotated into FP16, one cuBLAS FP16 GEMM with FP32 sums, the result rotated into BF16
  void dense_prefill(const std::string &prefix, const DenseExl3 &d, const void *in, void *out, int m, int xs, int os) {
    const auto *table = static_cast<const ds41::DeviceExpert *>(dense_table.p);
    const std::string layer = prefix.substr(0, prefix.find('.', 7));
    if (layer != dense_cache_layer) { dense_cache_layer = layer; dense_cache_at.clear(); dense_cache_used = 0; }
    auto at = dense_cache_at.find(prefix);
    if (at == dense_cache_at.end()) {
      const size_t bytes = (size_t(d.k) * d.n * 2 + 255) / 256 * 256;
      if (dense_cache_used + bytes > dense_cache.bytes) { dense_cache_at.clear(); dense_cache_used = 0; }
      CK(ds41::exl3_dense_reconstruct(table, d.index, d.n, d.k, d.bits,
                                      reinterpret_cast<half *>(dense_cache.as<char>(dense_cache_used)), stream));
      at = dense_cache_at.emplace(prefix, dense_cache_used).first;
      dense_cache_used += bytes;
    }
    const half *W = reinterpret_cast<const half *>(dense_cache.as<char>(at->second));
    cudaStream_t cur = nullptr;   // s0ma change 39: setting the same stream again would drop the workspace
    if (cublasGetStream(blas, &cur) != CUBLAS_STATUS_SUCCESS || cur != stream)
      if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSetStream");
    const float one = 1, zero = 0;
    for (int r = 0; r < m; r += 256) {
      const int rows = std::min(256, m - r);
      CK(ds41::exl3_dense_rotate_in(table, d.index, static_cast<const __nv_bfloat16 *>(in) + size_t(r) * xs, rows, d.k,
                                    xs, dense_rows_a.as<half>(), stream));
      if (cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, d.n, rows, d.k, &one, W, CUDA_R_16F, d.n,
                       dense_rows_a.p, CUDA_R_16F, d.k, &zero, dense_rows_c.p, CUDA_R_32F, d.n,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx (Sero dense prefill): " + prefix);
      CK(ds41::exl3_dense_rotate_out(table, d.index, dense_rows_c.as<float>(), rows, d.n, d.k,
                                     static_cast<__nv_bfloat16 *>(out) + size_t(r) * os, os, stream));
    }
  }
  // wo[m, 8192] = wo_a(o[m, 32768]) through Sero's eight slices: false when this layer keeps the native wo_a
  bool dense_wo_a(const std::string &attn, const void *o, void *wo, int m) {
    if (dense_woa.empty() || !dense_woa.count(attn)) return false;
    if (m <= 8) {   // decode: the eight slices (consecutive table entries) in one launch
      const DenseExl3 *d = dense_of(attn + ".wo_a.slice.0");
      ds41::GemmJob job{ds41::ExpertId(d->index), m, static_cast<const uint8_t *>(o), nullptr,
                        static_cast<__nv_bfloat16 *>(wo)};
      const int lane = lane_b_active ? 1 : 0;
      CK(ds41::exl3_dense(static_cast<const ds41::DeviceExpert *>(dense_table.p), job, 1024, 4096, d->bits, stream,
                          32768, 8192, 8, 4096, 1024, dense_split_partial[lane].as<float>(),
                          dense_split_partial[lane].bytes / 4, dense_split_counts[lane].as<unsigned>(),
                          dense_split_counts[lane].bytes / 4));
      dense_since_quant = true;
      return true;
    }
    for (int g = 0; g < 8; ++g)
      dense_linear(attn + ".wo_a.slice." + std::to_string(g), static_cast<const __nv_bfloat16 *>(o) + size_t(g) * 4096,
                   static_cast<__nv_bfloat16 *>(wo) + size_t(g) * 1024, m, 32768, 8192);
    return true;
  }
  void linear(const std::string &prefix, const void *in, void *out, int n,
              int k) {
    if (dense_linear(prefix, in, out, 1)) return;   // s0ma change 34
    auto &weight = w(prefix + ".weight");
    if (weight.shape != std::vector<uint64_t>({uint64_t(n), uint64_t(k)}))
      throw std::runtime_error("linear shape: " + prefix);
    if (weight.dtype == "F8_E4M3" || weight.dtype == "F8_E4M3FN") {
      CK(ds41_dense_quant(in, aq.p, as.p, 1, k, stream));
      CK(ds41_dense_fp8(aq.p, as.p, weight.data.p, w(prefix + ".scale").data.p,
                        out, 1, n, k, 2, stream));
    } else {
      CK(ds41_dense_cublas(blas, in, weight.data.p, out, 1, n, k,
                           weight.dtype == "F32", 1, stream));
    }
  }
  void norm(const std::string &name, const void *in, void *out, int k) {
    CK(ds41_dense_rms(in, w(name + ".weight").data.p, out, 1, k, 1e-20f,
                      stream));
  }
  void hc(const std::string &prefix, const std::string &kind,
          Device &target_pre) {
    CK(ds41_dense_hc_mixes(
        blas, h.p, w(prefix + ".hc_" + kind + "_fn").data.as<float>(),
        w(prefix + ".hc_" + kind + "_scale").data.as<float>(),
        w(prefix + ".hc_" + kind + "_base").data.as<float>(),
        target_pre.as<float>(), post.as<float>(), comb.as<float>(),
        f32.as<float>(), mixes.as<float>(), rstd.as<float>(), 1, H, 1e-20f,
        1e-6f, stream));
  }
  void apply_rope(Device &d, int heads, int dim, int l, int64_t pos,
                  bool inverse = false) {
    CK(ds41_dense_rope(d.p, freqs(l, pos), 1, heads, dim, 64, inverse, stream));
  }
  void engram(int layer, const std::vector<int64_t> &rowids, int m = 1) {
    double t = clock_ms();
    std::string prefix = "layers." + std::to_string(layer) + ".engram";
    if (async_engram) {
      const void *raw = engram_tickets[layer == 1 ? 0 : 1]->wait(rowids.size());
      upload(rows_interleaved, raw, rowids.size() * 264);
      async_engram->dequant(rows_interleaved.p, rows_bf.p, int(rowids.size()), stream);
      engram_ms += clock_ms() - t;
    } else {
      const auto &values = source.tensor(prefix + ".embed.weight");
      const auto &scales = source.tensor(prefix + ".embed.scale");
      std::map<std::pair<std::string, uint64_t>, size_t> page_index;
      std::vector<ds41::Record> requests;
      std::vector<void *> destinations;
      auto fetch = [&](const Tensor &tensor, int64_t row, size_t rowbytes) {
        if (row < 0 || uint64_t(row) >= tensor.shape[0])
          throw std::runtime_error("Engram row OOB");
        uint64_t address = tensor.offset + uint64_t(row) * rowbytes;
        for (uint64_t p = address & ~4095ull; p < address + rowbytes;
             p += 4096) {
          auto key = std::make_pair(tensor.path, p);
          if (page_index.count(key))
            continue;
          size_t index = page_index.size();
          page_index[key] = index;
          ds41::Record rec;
          rec.id = uint32_t(index);
          rec.bytes = 4096;
          ds41::Extent e{tensor.path,
                         p,
                         4096,
                         0,
                         std::min<uint64_t>(4096, tensor.file_bytes - p),
                         tensor.file_bytes};
          rec.extents.push_back(e);
          requests.push_back(std::move(rec));
        }
      };
      for (auto row : rowids) {
        fetch(values, row, 256);
        fetch(scales, row, 8);
      }
      if (requests.size() * 4096 > pages->size())
        throw std::runtime_error("Engram staging capacity");
      std::vector<const ds41::Record *> ptrs;
      for (size_t i = 0; i < requests.size(); ++i) {
        ptrs.push_back(&requests[i]);
        destinations.push_back(static_cast<char *>(pages->data()) + i * 4096);
      }
      reader->read(ptrs, destinations);
      auto gather_row = [&](const Tensor &tensor, int64_t row, size_t stride,
                            uint8_t *out) {
        uint64_t a = tensor.offset + uint64_t(row) * stride;
        for (size_t i = 0; i < stride; ++i) {
          uint64_t offset = a + i;
          auto ix = page_index.at({tensor.path, offset & ~4095ull});
          out[i] = static_cast<uint8_t *>(
              pages->data())[ix * 4096 + (offset & 4095)];
        }
      };
      auto raw = static_cast<uint8_t *>(host->data());
      for (size_t i = 0; i < rowids.size(); ++i) {
        gather_row(values, rowids[i], 256, raw + i * 256);
        gather_row(scales, rowids[i], 8, raw + rowids.size() * 256 + i * 8);
      }
      upload(rows_raw, raw, rowids.size() * 256);
      upload(scales_raw, raw + rowids.size() * 256, rowids.size() * 8);
      engram_ms += clock_ms() - t;
      engram_dequant(rows_raw.as<uint8_t>(), scales_raw.as<uint8_t>(),
                     rows_bf.p, int(rowids.size()), stream);
    }
    blinear(prefix + ".wkv", rows_bf.p, engram_kv.p, (HC + 1) * H, 6144, m);
    for (int r = 0; r < m; ++r)
    CK(ds41_dense_engram_gate(h.as<char>(r*HC*H*2), engram_kv.as<char>(r*(HC+1)*H*2), engram_kv.as<char>((r*(HC+1)+HC)*H*2),
                              w(prefix + ".q_weight").data.p,
                              w(prefix + ".k_weight").data.p, nullptr, next_h.as<char>(r*HC*H*2),
                              1, H, 1e-20f, stream));
    std::swap(h, next_h);
  }
  void attention(int l, int64_t pos) {
    std::string p = "layers." + std::to_string(l) + ".attn";
    apply_rope(q, 64, 512, l, pos);
    apply_rope(kv, 1, 512, l, pos);
    CK(ds41_attn_pack(kv.p, packed.p, 1, DS41_SWA_FP8, stream));
    CK(ds41_attn_ring_write(packed.p, windows[l].p, 528, pos, 1, RING, stream));
    int owner = l < 2 ? -1 : l < 8 ? 2 : l < 14 ? 8 : l < 20 ? 14 : 20;
    int ratio = l < 20 ? 2 : 1;
    int n = owner < 0 ? 0 : int((pos + 1) / ratio);
    bool owns = globals.count(l);
    bool index = l == 2 || l == 8 || l == 14 || l == 20 || l == 24 || l == 28 ||
                 l == 32 || l == 36;
    if (owns) {
      auto &g = *globals.at(l);
      bool complete = ratio == 1 || (pos % 2 == 1);
      if (ratio == 1)
        linear(p + ".compressor.wkv", x.p, latent.p, 512, H);
      else {
        bf16_float(x.p, f32.as<float>(), H, stream);
        auto off = (pos % 8) * 512 * 4;
        linear(p + ".compressor.wkv", f32.p, g.raw_kv.as<char>(off), 512, H);
        linear(p + ".compressor.wgate", f32.p, g.raw_score.as<char>(off), 512,
               H);
        if (complete)
          CK(ds41_attn_pool2(g.raw_kv.as<float>(), g.raw_score.as<float>(),
                             latent.p, pos, 1, 8, stream));
      }
      if (complete) {
        norm(p + ".compressor.norm", latent.p, latent.p, 512);
        linear(p + ".indexer.wk", latent.p, ik.p, 128, 512);
        norm(p + ".indexer.k_norm", ik.p, ik.p, 128);
        apply_rope(ik, 1, 128, l, pos + 1 - ratio);
        CK(ds41_attn_pack(ik.p, g.key.as<char>(size_t(n - 1) * 68), 1,
                          DS41_INDEX_FP4, stream));
        apply_rope(latent, 1, 512, l, pos + 1 - ratio);
        CK(ds41_attn_pack(latent.p, g.kv.as<char>(size_t(n - 1) * 288), 1,
                          DS41_GLOBAL_FP4, stream));
      }
    }
    int k = std::min(512, n);
    if (index && n) {
      auto &g = *globals.at(owner);
      linear(p + ".indexer.wq_b", qa.p, iq.p, 4096, 1280);
      apply_rope(iq, 32, 128, l, pos);
      CK(ds41_attn_roundtrip(iq.p, 32, 128, DS41_INDEX_FP4, stream));
      linear(p + ".indexer.weights_proj", x.p, iw.p, 32, H);
      scale_bf16(iw.p, 32, 1.f / 64, stream);
      upload(visible, &n, 4);
      int nscan = l > 20 ? candidate_count : n;
      const int32_t *keys = l > 20 ? candidate_ids.as<int32_t>() : nullptr;
      CK(ds41_attn_index_scores(iq.p, g.key.p, iw.p, keys,
                                visible.as<int32_t>(), scores.p, 1, nscan,
                                stream));
      if (l == 20) {
        int nb = (n + 7) / 8, kb = std::min(2048, nb);
        CK(ds41_attn_block_scores(scores.p, visible.as<int32_t>(), blocks.p, 1,
                                  n, stream));
        upload(block_visible, &nb, 4);
        CK(ds41_attn_topk(blocks.p, block_visible.as<int32_t>(),
                          chosen_blocks.as<int32_t>(), 1, nb, kb, top_scratch.p,
                          stream));
        CK(ds41_attn_expand_blocks(chosen_blocks.as<int32_t>(),
                                   candidate_ids.as<int32_t>(), 1, kb, n,
                                   stream));
        candidate_count = kb * 8;
      }
      if (l > 20) {
        int scan_visible = nscan;
        upload(block_visible, &scan_visible, 4);
        CK(ds41_attn_topk(scores.p, block_visible.as<int32_t>(),
                          candidate_top.as<int32_t>(), 1, nscan, k,
                          top_scratch.p, stream));
        CK(ds41_attn_remap_topk(candidate_top.as<int32_t>(),
                                candidate_ids.as<int32_t>(), top.as<int32_t>(),
                                1, k, nscan, stream));
      } else
        CK(ds41_attn_topk(scores.p, visible.as<int32_t>(), top.as<int32_t>(), 1,
                          n, k, top_scratch.p, stream));
      shared_topk = k;
    } else if (index)
      shared_topk = 0;
    if (owner < 0)
      k = 0;
    else if (shared_topk != k)
      throw std::runtime_error("shared index width mismatch");
    Global *g = owner < 0 ? nullptr : globals.at(owner).get();
    if (pos == 0) {
      upload(position_device, &pos, 8);
      CK(ds41_attn_gather_prefill_at(
          windows[l].p, RING, g ? g->kv.p : nullptr, g ? g->capacity : 0,
          k ? top.as<int32_t>() : nullptr, k, position_device.as<int64_t>(), 1,
          ratio, 1, gather.p, local_idx.as<int32_t>(), stream));
    } else
      CK(ds41_attn_gather(windows[l].p, RING, g ? g->kv.p : nullptr,
                          g ? g->capacity : 0, k ? top.as<int32_t>() : nullptr,
                          k, pos, 1, ratio, gather.p, local_idx.as<int32_t>(),
                          stream));
    int columns = (pos == 0 ? 1 : 128) + k;
    if (owner >= 0 && pos > 0)
      CK(ds41_attn_sparse_split(
          q.p, gather.p, w(p + ".attn_sink").data.as<float>(),
          local_idx.as<int32_t>(), o.p, attn_scores.as<float>(), 1, columns, 16,
          stream));
    else
      CK(ds41_attn_sparse(q.p, gather.p, w(p + ".attn_sink").data.as<float>(),
                          local_idx.as<int32_t>(), o.p, 1, columns, 16,
                          stream));
    apply_rope(o, 64, 512, l, pos, true);
    CK(ds41_attn_roundtrip(o.p, 1, 32768, DS41_SWA_FP8, stream));
    graph(p + ".output", [&] {
      if (!dense_wo_a(p, o.p, wo.p, 1))   // s0ma change 34: Sero's wo_a slices when loaded
        CK(ds41_dense_cublas(blas, o.p, woa_weight(p + ".wo_a"), wo.p, 1,
                             1024, 4096, 0, 8, stream));
      linear(p + ".wo_b", wo.p, y.p, H, 8192);
    });
  }
  // image: an image-span row, which selects experts with the vision bias (official Gate.bias_vl).
  void router_scores(int l, bool image = false) {
    std::string p = "layers." + std::to_string(l) + ".ffn";
    bf16_float(x.p, f32.as<float>(), H, stream);
    linear(p + ".gate", f32.p, router.p, 384, H);
    CK(ds41_dense_router_scores(
        router.as<float>(), w(p + (image ? ".gate.bias_vl" : ".gate.bias")).data.as<float>(),
        unbiased.as<float>(), biased.as<float>(), 1, 384, stream));
  }
  void ensure_vision() {
    if (vision) return;
    image_delimiters.resize(3 * H * 2);
    int i = 0;
    for (const char *name : {"image_start", "image_newline", "image_end"}) {
      const auto &t = source.tensor(name);
      if (t.dtype != "BF16" || t.shape != std::vector<uint64_t>({uint64_t(H)}))
        throw std::runtime_error(std::string("image delimiter dtype/shape: ") + name);
      auto bytes = source.read(t);
      CK(cudaMemcpyAsync(image_delimiters.as<char>(size_t(i++) * H * 2), bytes.data(), bytes.size(),
                         cudaMemcpyHostToDevice, stream));
      CK(cudaStreamSynchronize(stream));
    }
    vision = std::make_unique<VisionEncoder>(source, stream);
    std::cerr << "ds41d vision tower loaded\n";
  }
  void moe(int l) {
    std::string p = "layers." + std::to_string(l) + ".ffn";
    double t = clock_ms();
    auto score_host = static_cast<float *>(host->data());
    CK(cudaMemcpyAsync(score_host, biased.p, 384 * 4, cudaMemcpyDeviceToHost,
                       stream));
    CK(cudaStreamSynchronize(stream));
    route_ms += clock_ms() - t;
    check_moe_retirement(false);
    std::vector<int> selected(384);
    std::iota(selected.begin(), selected.end(), 0);
    std::partial_sort(selected.begin(), selected.begin() + 6, selected.end(),
                      [&](int a, int b) {
                        return score_host[a] == score_host[b]
                                   ? a < b
                                   : score_host[a] > score_host[b];
                      });
    selected.resize(6);
    upload(ids_device, selected.data(), 6 * 4);
    std::vector<ds41::ExpertId> need;
    for (int e : selected)
      need.push_back(ds41::expert_id(l, e));
    t = clock_ms();
    auto lease = experts->need_now(need, stream);
    io_ms += clock_ms() - t;
    try {
      std::vector<ds41::GemmJob> job(6);
      // EXL3 experts read the BF16 FFN input and BF16 activations directly.
      for (int j = 0; j < 6; ++j)
        job[j] = {need[j], 1, exl3 ? x.as<uint8_t>() : aq.as<uint8_t>(),
                  exl3 ? nullptr : as.as<uint8_t>(), gate.as<Bf16>(j * I * 2)};
      upload(jobs, job.data(), job.size() * sizeof(ds41::GemmJob));
      for (int j = 0; j < 6; ++j)
        job[j].output = up.as<Bf16>(j * I * 2);
      upload(up_jobs, job.data(), job.size() * sizeof(ds41::GemmJob));
      for (int j = 0; j < 6; ++j)
        job[j] = {need[j], 1, exl3 ? act.as<uint8_t>(j * I * 2) : eq.as<uint8_t>(j * I),
                  exl3 ? nullptr : es.as<uint8_t>(j * I / 32), down.as<Bf16>(j * H * 2)};
      upload(down_jobs, job.data(), job.size() * sizeof(ds41::GemmJob));
      graph(p + ".experts", [&] {
        route_weights(unbiased.as<float>(), ids_device.as<int>(),
                      route.as<float>(), 6, stream);
        if (!exl3)
          CK(ds41::quantize_mxfp8(x.as<Bf16>(), aq.as<uint8_t>(),
                                  as.as<uint8_t>(), 1, H, stream));
        CK(expert_gemm(jobs.as<ds41::GemmJob>(), 6, ds41::Projection::Gate, I, H, stream, l));
        CK(expert_gemm(up_jobs.as<ds41::GemmJob>(), 6, ds41::Projection::Up, I, H, stream, l));
        CK(ds41::swiglu(gate.as<Bf16>(), up.as<Bf16>(), route.as<float>(),
                        act.as<Bf16>(), 6, I, 10, stream));
        if (!exl3)
          CK(ds41::quantize_mxfp8(act.as<Bf16>(), eq.as<uint8_t>(),
                                  es.as<uint8_t>(), 6, I, stream));
        CK(expert_gemm(down_jobs.as<ds41::GemmJob>(), 6, ds41::Projection::Down, H, I, stream, l));
        linear(p + ".shared_experts.w1", x.p, sg.p, I, H);
        linear(p + ".shared_experts.w3", x.p, su.p, I, H);
        CK(ds41::swiglu(sg.as<Bf16>(), su.as<Bf16>(), nullptr, sa.as<Bf16>(), 1,
                        I, 10, stream));
        linear(p + ".shared_experts.w2", sa.p, shared.p, H, I);
        CK(ds41::reduce_experts(down.as<Bf16>(), ids_device.as<int32_t>(),
                                shared.as<Bf16>(), y.as<Bf16>(), 1, 6, H,
                                stream));
      });
      retire_experts(lease);
    } catch (...) {
      CK(cudaStreamSynchronize(stream));
      if (lease.owner)
        experts->retire(std::move(lease), stream);
      throw;
    }
  }
  std::vector<float> one(int token, int64_t pos) {
    double start = clock_ms();
    phase_index = 0;
    phase_names.clear();
    upload_cursor = 1 << 20;
    in_step = true;
    graph_ready = pos > 0;
    auto rowids = hash.rows(token, pos);
    if (async_engram) {
      // The previous one() synchronized this stream before its tickets can
      // be recycled. Keep both current tickets through all H2D consumers,
      // including exception cleanup in the Native destructor.
      engram_tickets[0] = async_engram->submit(1, rowids[0]);
      engram_tickets[1] = async_engram->submit(14, rowids[1]);
    }
    expand_embedding(w("embed.weight").data.as<Bf16>(size_t(token) * H * 2),
                     h.p, pre.as<float>(), H, stream);
    for (int l = 0; l < 40; ++l) {
      std::string p = "layers." + std::to_string(l);
      if (l == 1 || l == 14)
        engram(l, rowids[l == 1 ? 0 : 1]);
      dump(pos, l, "input", h, HC * H);
      if (l >= 37)
        capture_mean(
            h.p,
            capture_slab.as<char>((size_t(capture_slot) * 3 + l - 37) * H * 2),
            H, stream);
      phase("attention", [&] {
        graph(p + ".query", [&] {
          hc(p, "attn", att_pre);
          CK(ds41_dense_hc_pre(h.p, pre.as<float>(), x.p, 1, H, stream));
          norm(p + ".attn_norm", x.p, x.p, H);
          std::string a = p + ".attn";
          linear(a + ".wq_a", x.p, qa.p, 1280, H);
          norm(a + ".q_norm", qa.p, qa.p, 1280);
          linear(a + ".wq_b", qa.p, q.p, 32768, 1280);
          linear(a + ".wkv", x.p, kv.p, 512, H);
          norm(a + ".kv_norm", kv.p, kv.p, 512);
        });
        dump(pos, l, "attn_input", x, H);
        attention(l, pos);
        dump(pos, l, "attn_output", y, H);
        CK(ds41_dense_hc_post(y.p, h.p, post.as<float>(), comb.as<float>(),
                              next_h.p, 1, H, stream));
        std::swap(h, next_h);
      });
      phase("moe", [&] {
        graph(p + ".route", [&] {
          hc(p, "ffn", ffn_pre);
          CK(ds41_dense_hc_pre(h.p, att_pre.as<float>(), x.p, 1, H, stream));
          norm(p + ".ffn_norm", x.p, x.p, H);
          router_scores(l);
        });
        dump(pos, l, "ffn_input", x, H);
        moe(l);
        dump(pos, l, "ffn_output", y, H);
        CK(ds41_dense_hc_post(y.p, h.p, post.as<float>(), comb.as<float>(),
                              next_h.p, 1, H, stream));
        std::swap(h, next_h);
        std::swap(pre, ffn_pre);
      });
      dump(pos, l, "output", h, HC * H);
    }
    phase("head", [&] {
      CK(ds41_dense_hc_pre(h.p, pre.as<float>(), x.p, 1, H, stream));
      norm("norm", x.p, x.p, H);
      bf16_float(x.p, f32.as<float>(), H, stream);
      if (sero_head(x.p, logits.p, 1, stream)) {}   // s0ma change 43
      else if (fused_head) fused_head_logits(f32.p, logits.p, 1);
      else linear("head", f32.p, logits.p, V, H);
    });
    auto output = static_cast<float *>(host->data());
    CK(cudaMemcpyAsync(output, logits.p, V * 4, cudaMemcpyDeviceToHost,
                       stream));
    CK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < phase_index; ++i) {
      float elapsed;
      CK(cudaEventElapsedTime(&elapsed, phases[i]->start, phases[i]->end));
      measured[phase_names[i]] += elapsed;
    }
    total_ms += clock_ms() - start;
    in_step = false;
    return {output, output + V};
  }
  #include "hotpath/moe_adapter.inc"
  #include "hotpath/batch.inc"
  #include "hotpath/streamed.inc"
  // EXL3 K3 bank (k3/repack_bank.py): bank-LL.bin holds layer LL's 384 records of 13,316,096 bytes in
  // expert order; each record is w1, w3, w2 trellis [k/16,n/16,48] int16, each followed by suh[k],svh[n]
  // FP16. The layout and file coverage are checked exactly. Each receipt holds the bank's SHA-256 and the
  // size and mtime it had when that digest was last recomputed from disk (k3/verify_bank.sh); a bank
  // rewritten since then is refused. The digest itself is not recomputed at startup.
  // The same layout at K2 (scripts/k2_bank.py): trellis [k/16,n/16,32], records of 8,892,416 bytes. The width
  // comes from the records' size, is the same for every record, and selects the kernels (exl3_set_bits).
  // s0ma change 12: a mixed bank's store gets one slot class per width (K2 records 8,892,416 bytes, K3 13,316,096),
  // so a K2 expert never takes a K3-sized slot. VRAM: each class has the slots the placement gives its layers (pins
  // plus dynamic budget, which a mixed placement states in its layers' own sizes; they must add up to
  // --expert-slots). RAM: the tier's bytes split between the classes as the placement's per-layer host capacities
  // are (its seeds), in each class's slot size.
  static uint64_t exl3_rec_of(uint64_t bits) { return (3 * (320 * 144 * 16 * bits * 2 + 2 * (5120 + 2304)) + 4095) / 4096 * 4096; }
  // s0ma changes 12 and 15: a bank of several widths gets one store slot class per record size, so an expert only
  // takes a slot of its own size. VRAM: the placement's pinned experts each in their class; each layer's dynamic
  // budget split between the classes in proportion to that layer's unpinned experts of each (largest remainder), and
  // pins plus budgets must add up to --expert-slots. RAM: the tier's bytes split as the placement's seeds' bytes are.
  void mixed_store_classes(ds41::OffloadStore::Config &ec, const Config &c) {
    if (c.placement.empty()) throw std::invalid_argument("a mixed EXL3 bank needs its placement (--placement)");
    const auto pl = ds41::load_placement(c.placement);
    std::vector<int> widths;
    for (int b = 2; b <= 5; ++b) if (std::count(width_of.begin(), width_of.end(), uint8_t(b))) widths.push_back(b);
    const size_t n = widths.size();
    class_widths = widths;
    std::array<int, 6> cls{};
    cls.fill(-1);
    for (size_t k = 0; k < n; ++k) cls[widths[k]] = int(k);
    ec.expert_class.assign(ds41::expert_count, 0);
    for (ds41::ExpertId id = 0; id < 15360; ++id) ec.expert_class[id] = uint8_t(cls[width_of[id]]);
    std::vector<size_t> bytes(n), vram(n, 0), seeds(n, 0);
    std::vector<double> host_w(n, 0);
    for (size_t k = 0; k < n; ++k) bytes[k] = exl3_rec_of(widths[k]);
    std::vector<bool> pinned(15360);
    std::vector<size_t> pins(n, 0), unpinned(n, 0), layer_max(n, 0);
    for (auto id : pl.vram_pins) if (id < 15360) { pinned[id] = true; ++vram[ec.expert_class[id]]; ++pins[ec.expert_class[id]]; }
    for (int l = 0; l < 40; ++l) {
      std::vector<size_t> free_k(n, 0);
      size_t free_all = 0;
      for (int e = 0; e < 384; ++e) if (!pinned[l * 384 + e]) { ++free_k[ec.expert_class[l * 384 + e]]; ++free_all; }
      for (size_t k = 0; k < n; ++k) { unpinned[k] += free_k[k]; layer_max[k] = std::max(layer_max[k], free_k[k]); }
      const size_t d = pl.dynamic_slots_per_layer[l];
      std::vector<size_t> part(n);
      std::vector<std::pair<double, size_t>> rest;
      size_t given = 0;
      for (size_t k = 0; k < n; ++k) {
        const double exact = free_all ? double(d) * free_k[k] / free_all : 0;
        part[k] = size_t(exact); given += part[k]; rest.push_back({exact - part[k], k});
      }
      std::sort(rest.begin(), rest.end(), [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
      for (size_t j = 0; given < d && j < rest.size(); ++j, ++given) ++part[rest[j].second];
      for (size_t k = 0; k < n; ++k) vram[k] += part[k];
    }
    for (auto id : pl.host_residents) if (id < 15360) { ++seeds[ec.expert_class[id]]; host_w[ec.expert_class[id]] += bytes[ec.expert_class[id]]; }
    size_t total = 0;
    for (auto v : vram) total += v;
    if (total != size_t(c.expert_slots))
      throw std::invalid_argument("a mixed bank's placement must state every VRAM slot: its pins and budgets give " +
                                  std::to_string(total) + ", --expert-slots is " + std::to_string(c.expert_slots));
    // A width class that is small by count (0xSero's bank: 35 K5 experts) gets almost no dynamic slots from the
    // count split, but one layer's need set (up to 6 rows x 6 experts; the previous layer's lease may still be held)
    // plus the lookahead and the demotion reserve must fit in its unpinned slots, or acquire throws ("exceeds
    // unleased slot capacity of its size class"). Raise such a class to that floor, paid in bytes by the class with
    // the most dynamic bytes, so the VRAM spent on experts does not grow.
    // s0ma change 19: DS41_CLASS_WEIGHTS="K3:1.3,K2:0.9" re-splits the dynamic VRAM between the width classes in bytes:
    // class k gets the share w_k x (its unpinned experts) x (its record size) of the same dynamic bytes (w = 1 for every
    // class gives back the count split above). Sero's first bench: per unpinned expert, K3 is demanded ~1.3x as often as
    // K2 and takes 60% of the NVMe misses.
    // s0ma change 23: DS41_EXPERT_EXTRA_GIB=N adds N GiB of dynamic expert slots beyond the placement's VRAM (ds41d
    // leaves ~5 GiB free beside a mixed bank; prefill segments use free VRAM too), split between the classes as below.
    double extra_gib = 0;
    if (const char *x = std::getenv("DS41_EXPERT_EXTRA_GIB"); x && *x) {
      char *end = nullptr;
      extra_gib = std::strtod(x, &end);
      if (*end || !(extra_gib >= 0) || extra_gib > 64) throw std::invalid_argument("DS41_EXPERT_EXTRA_GIB: 0..64 GiB");
    }
    extra_gib += dense_extra_gib;   // s0ma change 34: the VRAM Sero's dense tensors gave back
    const char *cw_env = std::getenv("DS41_CLASS_WEIGHTS");
    if ((cw_env && *cw_env) || extra_gib > 0) {
      const char *cw = cw_env && *cw_env ? cw_env : "";
      std::vector<double> wk(n, 1.0);
      std::string spec(cw), item;
      for (size_t i = 0; i < spec.size() + (spec.empty() ? 0 : 1); ++i) {
        if (i < spec.size() && spec[i] != ',') { item += spec[i]; continue; }
        if (item.size() < 4 || item[0] != 'K' || item[2] != ':') throw std::invalid_argument("DS41_CLASS_WEIGHTS: K<w>:<weight>, e.g. K3:1.3");
        const int b = item[1] - '0';
        const double v = std::stod(item.substr(3));
        if (b < 2 || b > 5 || !(v > 0)) throw std::invalid_argument("DS41_CLASS_WEIGHTS: K2..K5 with positive weights");
        if (cls[b] >= 0 && size_t(cls[b]) < n && widths[cls[b]] == b) wk[cls[b]] = v;
        item.clear();
      }
      double dyn_bytes = 0, denom = 0;
      for (size_t k = 0; k < n; ++k) { dyn_bytes += double(vram[k] - pins[k]) * bytes[k]; denom += wk[k] * unpinned[k] * bytes[k]; }
      dyn_bytes += extra_gib * double(1ull << 30);
      std::ostringstream note;
      note << "mixed EXL3 store: DS41_CLASS_WEIGHTS=" << (*cw ? cw : "1") << ", DS41_EXPERT_EXTRA_GIB=" << extra_gib
           << ": dynamic slots";
      for (size_t k = 0; k < n; ++k) {
        const size_t dyn = denom > 0 ? std::min(unpinned[k], size_t(dyn_bytes * wk[k] * unpinned[k] / denom)) : 0;
        note << (k ? ", K" : " K") << widths[k] << " " << (vram[k] - pins[k]) << " -> " << dyn;
        vram[k] = pins[k] + dyn;
      }
      std::cerr << note.str() << "\n";
    }
    std::string floor_note;
    for (size_t k = 0; k < n; ++k) {
      // one layer's need set twice (the previous layer's lease may be held), plus the lookahead and this class's share
      // of the demotion reserve (change 16 splits it by class)
      const size_t floor_k = std::min(unpinned[k], 2 * std::min<size_t>(36, layer_max[k]) + 4);
      const size_t dyn = vram[k] - pins[k];
      if (dyn >= floor_k) continue;
      size_t donor = n;
      for (size_t j = 0; j < n; ++j)
        if (j != k && (donor == n || (vram[j] - pins[j]) * bytes[j] > (vram[donor] - pins[donor]) * bytes[donor])) donor = j;
      const size_t add = floor_k - dyn, take = (add * bytes[k] + bytes[donor] - 1) / bytes[donor];
      if (donor == n || vram[donor] - pins[donor] < take + 64) continue;
      vram[k] += add; vram[donor] -= take;
      floor_note += " K" + std::to_string(widths[k]) + " +" + std::to_string(add) + " (from K" + std::to_string(widths[donor]) +
                    " -" + std::to_string(take) + ")";
    }
    ec.slots = 0;
    for (auto v : vram) ec.slots += v;
    if (!floor_note.empty()) std::cerr << "mixed EXL3 store: small width classes raised to one layer's need set:" << floor_note << "\n";
    double hw = 0;
    for (auto w : host_w) hw += w;
    if (hw <= 0) for (size_t k = 0; k < n; ++k) { host_w[k] = double(std::count(width_of.begin(), width_of.end(), uint8_t(widths[k]))) * bytes[k]; hw += host_w[k]; }
    ec.class_bytes = bytes;
    ec.class_slots = vram;
    ec.class_host_slots.assign(n, 0);
    expert_payload_bytes = 0;
    std::ostringstream log;
    log << "mixed EXL3 store (" << mixed_k3_layers << "): VRAM";
    for (size_t k = 0; k < n; ++k) {
      ec.class_host_slots[k] = size_t(double(c.host_bytes) * host_w[k] / hw) / bytes[k];
      if (ec.class_host_slots[k] < seeds[k])
        throw std::invalid_argument("the RAM tier's K" + std::to_string(widths[k]) + " slots (" +
                                    std::to_string(ec.class_host_slots[k]) + ") cannot hold the placement's " +
                                    std::to_string(seeds[k]) + " seeds");
      expert_payload_bytes += uint64_t(vram[k]) * bytes[k];
      log << (k ? " + " : " ") << vram[k] << " K" << widths[k];
    }
    log << " slots (" << expert_payload_bytes / double(1ull << 30) << " GiB), RAM tier";
    for (size_t k = 0; k < n; ++k) log << (k ? " + " : " ") << ec.class_host_slots[k] << " K" << widths[k];
    std::cerr << log.str() << " slots\n";
  }
  void validate_k3_records(const std::vector<ds41::Record> &records, const std::string &receipts) {
    if (records.empty()) throw std::runtime_error("EXL3 manifest is empty");
    constexpr uint64_t SB = 2 * (5120 + 2304);
    auto rec_of = [](uint64_t bits) { return exl3_rec_of(bits); };
    if (rec_of(3) != 13316096) throw std::logic_error("K3 record size");
    // s0ma changes 12 and 15: the width is per expert, from its record's size (2..5 bits); a layer whose experts all
    // share one width launches at that width, a layer that mixes them through grouped_exl3_var.
    width_of.assign(15360, 0);
    for (auto &r : records) {
      if (r.id >= 15360) throw std::runtime_error("EXL3 manifest duplicate/id");
      int b = 0;
      for (int k = 2; k <= 5; ++k) if (r.bytes == rec_of(k)) b = k;
      if (!b) throw std::runtime_error("EXL3 record size is not K2..K5: expert " + std::to_string(r.id));
      width_of[r.id] = uint8_t(b);
    }
    std::array<size_t, 6> hist{};
    for (int l = 0; l < 40; ++l) {
      int lb = width_of[l * 384];
      for (int e = 0; e < 384; ++e) { ++hist[width_of[l * 384 + e]]; if (width_of[l * 384 + e] != lb) lb = 0; }
      layer_bits[l] = lb;
    }
    int widths = 0;
    for (int b = 2; b <= 5; ++b) widths += hist[b] > 0;
    exl3_mixed = widths > 1;
    exl3_bits_loaded = exl3_mixed ? 0 : width_of[0];
    if (!exl3_mixed && ds41::exl3_set_bits(exl3_bits_loaded) != cudaSuccess) throw std::runtime_error("EXL3 width not supported");
    if (std::all_of(layer_bits.begin(), layer_bits.end(), [](int b) { return b == 2 || b == 3; })) {
      for (int l = 0; l < 40; ++l)
        if (layer_bits[l] == 3) mixed_k3_layers += (mixed_k3_layers.empty() ? "" : ",") + std::to_string(l);
      mixed_k3_layers = "K3 in layers " + mixed_k3_layers + ", K2 elsewhere";
    } else {
      for (int b = 2; b <= 5; ++b) if (hist[b]) mixed_k3_layers += (mixed_k3_layers.empty() ? "" : ", ") + std::string("K") + std::to_string(b) + " " + std::to_string(hist[b]);
      mixed_k3_layers += " experts";
    }
    expert_width_dev.resize(15360);
    CK(cudaMemcpy(expert_width_dev.p, width_of.data(), 15360, cudaMemcpyHostToDevice));
    auto views_of = [&](uint64_t bits) {
      const uint64_t TB = 320 * 144 * 16 * bits * 2;
      return std::array<ds41::TensorView, 6>{ds41::TensorView{0, TB}, ds41::TensorView{TB, SB}, ds41::TensorView{TB + SB, TB},
          ds41::TensorView{2 * TB + SB, SB}, ds41::TensorView{2 * TB + 2 * SB, TB}, ds41::TensorView{3 * TB + 2 * SB, SB}};
    };
    if (records.size() != 15360)
      throw std::runtime_error("EXL3 manifest must hold exactly the 15360 target-layer experts");
    if (receipts.empty())
      throw std::invalid_argument("--exl3-experts requires --exl3-receipts");
    auto receipt = read_json(receipts);
    auto same = [](ds41::TensorView a, ds41::TensorView b) { return a.offset == b.offset && a.bytes == b.bytes; };
    std::vector<bool> seen(15360);
    std::map<unsigned, std::string> bank;
    std::map<unsigned, uint64_t> file_bytes;
    std::map<unsigned, std::vector<std::pair<uint64_t, uint64_t>>> spans;
    for (auto &r : records) {
      if (r.id >= 15360 || seen[r.id]) throw std::runtime_error("EXL3 manifest duplicate/id");
      seen[r.id] = true;
      unsigned layer = r.id / 384;
      const uint64_t REC = rec_of(width_of[r.id]);
      const auto v = views_of(width_of[r.id]);
      const auto &gate = v[0], &gate_scale = v[1], &up = v[2], &up_scale = v[3], &down = v[4], &down_scale = v[5];
      if (r.bytes != REC || !same(r.gate, gate) || !same(r.gate_scale, gate_scale) || !same(r.up, up) ||
          !same(r.up_scale, up_scale) || !same(r.down, down) || !same(r.down_scale, down_scale))
        throw std::runtime_error("EXL3 record layout mismatch: expert " + std::to_string(r.id));
      if (r.extents.size() != 1) throw std::runtime_error("EXL3 record must be one bank extent");
      const auto &e = r.extents[0];
      // the record lies whole, 4 KiB aligned, inside its layer's bank file (one record a 4 KiB-aligned run; K2/K3 banks
      // place expert e at e * REC, a mixed-width bank packs them in expert order)
      if (e.dst_offset != 0 || e.bytes != REC || (e.required_bytes && e.required_bytes != REC) ||
          e.file_offset % 4096 || e.file_offset + REC > e.expected_file_bytes)
        throw std::runtime_error("EXL3 bank extent mismatch: expert " + std::to_string(r.id));
      auto &path = bank[layer];
      if (path.empty()) { path = e.path; file_bytes[layer] = e.expected_file_bytes; }
      else if (path != e.path || file_bytes[layer] != e.expected_file_bytes) throw std::runtime_error("EXL3 layer spans several bank files");
      spans[layer].push_back({e.file_offset, e.file_offset + REC});
    }
    for (auto &[layer, v] : spans) {
      std::sort(v.begin(), v.end());
      for (size_t j = 1; j < v.size(); ++j)
        if (v[j].first < v[j - 1].second) throw std::runtime_error("EXL3 records overlap in layer " + std::to_string(layer));
    }
    for (auto &[layer, path] : bank) {
      const uint64_t FB = file_bytes[layer];
      char name[32];
      std::snprintf(name, sizeof name, "bank-%02u.bin", layer);
      auto file = std::filesystem::path(path);
      const auto key = std::to_string(layer);
      if (file.filename() != name || !receipt.contains(key) ||
          receipt[key].at("file").get<std::string>() != name ||
          receipt[key].at("bytes").get<uint64_t>() != FB ||
          receipt[key].at("sha256").get<std::string>().size() != 64 || !receipt[key].contains("mtime_ns"))
        throw std::runtime_error("EXL3 bank has no matching verified receipt: " + path);
      struct stat st {};
      if (::stat(path.c_str(), &st) || uint64_t(st.st_size) != FB ||
          uint64_t(st.st_mtim.tv_sec) * 1000000000ull + uint64_t(st.st_mtim.tv_nsec) !=
              receipt[key].at("mtime_ns").get<uint64_t>())
        throw std::runtime_error("EXL3 bank changed since its receipt was verified: " + path);
    }
  }
  // The DSpark drafter's 384 native MXFP4 experts, permanently resident in VRAM. The DS41_MOE_* policy
  // and GDS switches configure the target store, so this fixed store is built without them.
  void make_mtp_store(std::vector<ds41::Record> mtp) {
    if (mtp.size() != 384) throw std::runtime_error("native manifest must hold the 384 MTP experts");
    ds41::Placement place;
    for (auto &r : mtp) place.vram_pins.push_back(r.id);
    struct Saved { std::vector<std::pair<std::string, std::string>> v;
      ~Saved() { for (auto &[k, x] : v) setenv(k.c_str(), x.c_str(), 1); } } saved;
    for (const char *name : {"DS41_MOE_DEMOTE", "DS41_MOE_DEMOTE_EAGER", "DS41_MOE_DRIFT_0_50",
                             "DS41_MOE_GDS", "DS41_MOE_POLICY_AUDIT"})
      if (const char *value = std::getenv(name)) { saved.v.emplace_back(name, value); unsetenv(name); }
    ds41::OffloadStore::Config mc;
    mc.slots = 384;
    mc.staging_slots = 8;
    mc.io_depth = 64;
    mtp_experts = std::make_unique<ds41::OffloadStore>(std::move(mtp), mc);
    mtp_experts->place(place);
  }
  void validate_records(const std::vector<ds41::Record> &records) {
    std::set<unsigned> seen;
    for (auto &r : records) {
      std::vector<std::pair<uint64_t, uint64_t>> spans;
      for (const auto &extent : r.extents) {
        uint64_t required =
            extent.required_bytes ? extent.required_bytes : extent.bytes;
        if (required > extent.bytes ||
            extent.file_offset > extent.expected_file_bytes ||
            required > extent.expected_file_bytes - extent.file_offset)
          throw std::runtime_error(
              "invalid required expert span/file coverage");
        spans.emplace_back(extent.dst_offset, extent.bytes);
      }
      validate_spans(r.bytes, std::move(spans));
      std::vector<bool> used(r.extents.size());
      if (r.id >= ds41::expert_count || !seen.insert(r.id).second)
        throw std::runtime_error("expert manifest duplicate/id");
      unsigned layer = r.id < 15360 ? r.id / 384 : 40 + (r.id - 15360) / 128;
      unsigned expert = r.id < 15360 ? r.id % 384 : (r.id - 15360) % 128;
      std::string p = (layer < 40 ? "layers." + std::to_string(layer)
                                  : "mtp." + std::to_string(layer - 40)) +
                      ".ffn.experts." + std::to_string(expert);
      std::vector<std::pair<std::string, ds41::TensorView>> views = {
          {"w1.weight", r.gate}, {"w1.scale", r.gate_scale},
          {"w3.weight", r.up},   {"w3.scale", r.up_scale},
          {"w2.weight", r.down}, {"w2.scale", r.down_scale}};
      for (auto item : views) {
        auto &t = source.tensor(p + "." + item.first);
        if (item.second.bytes != t.bytes)
          throw std::runtime_error("expert view byte mismatch");
        bool match = false;
        for (size_t index = 0; index < r.extents.size(); ++index) {
          const auto &e = r.extents[index];
          uint64_t coverage = e.required_bytes ? e.required_bytes : e.bytes;
          if (item.second.offset >= e.dst_offset &&
              item.second.offset - e.dst_offset <= coverage &&
              item.second.bytes <=
                  coverage - (item.second.offset - e.dst_offset) &&
              e.path == t.path && e.expected_file_bytes == t.file_bytes &&
              e.file_offset + item.second.offset - e.dst_offset == t.offset)
            match = used[index] = true;
        }
        if (!match)
          throw std::runtime_error("expert view does not match verified native "
                                   "tensor/required extent coverage: " +
                                   p);
      }
      if (std::find(used.begin(), used.end(), false) != used.end())
        throw std::runtime_error("extraneous expert read span");
    }
  }
};
} // namespace
std::unique_ptr<Forward> make_native(const Config &c) {
  return std::make_unique<Native>(c);
}
} // namespace ds41e
