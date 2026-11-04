// Included inside Native. The decode/verify implementation remains independent.
// s0ma change 68: the routed experts' EXL3 prefill with each projection's rows rotated once (exl3_prefill_rotate,
// exl3_prefill_rotated: the same bits as grouped_exl3_prefill_bits). DS41_PREFILL_EXL3_ROTATED=0 keeps the old kernel.
// s0ma change 69: staged (RAM or drive) experts spread over the prefill lanes; DS41_PREFILL_MISS_LANES=0 keeps them on
// the main stream, one after another.
static bool prefill_miss_lanes() {
  static const bool on = [] { const char *v = std::getenv("DS41_PREFILL_MISS_LANES"); return !v || std::string(v) != "0"; }();
  return on;
}
// s0ma change 76: residents' expert batches grouped into waves of up to DS41_PREFILL_WAVE_ROWS rows (default 1024;
// 0: one expert at a time as before): one gather, one rotation, one launch per projection and width, one SwiGLU and
// one scatter a wave instead of eight launches an expert batch. Each batch's math is the per-expert path's.
static int prefill_wave_rows() {
  static const int n = [] {
    const char *v = std::getenv("DS41_PREFILL_WAVE_ROWS");
    const int x = v && *v ? std::atoi(v) : 1024;
    if (x != 0 && (x < 256 || x > 8192)) throw std::invalid_argument("DS41_PREFILL_WAVE_ROWS must be 0 or 256..8192");
    return x;
  }();
  return n;
}
// s0ma change 77: DS41_PREFILL_COPIES_FIRST=0 enqueues the residents before the staged copies start (as before)
static bool prefill_copies_first() {
  static const bool on = [] { const char *v = std::getenv("DS41_PREFILL_COPIES_FIRST"); return v && std::string(v) == "1"; }();   // change 78: off unless =1
  return on;
}
// s0ma change 82: a routed expert's prefill as three cuBLAS FP16 GEMMs (FP32 sums) on its weights decoded once into
// FP16, for an expert with at least DS41_PREFILL_EXL3_GEMM_MIN rows (32) in the segment; fewer rows keep prefill_rot,
// which decodes the weights again for every 64 rows inside the MMA loop. The same products as prefill_rot, summed in
// cuBLAS's order: a rounding-order change. DS41_PREFILL_EXL3_GEMM=1 turns it on.
static bool prefill_exl3_gemm() {
  static const bool on = [] { const char *v = std::getenv("DS41_PREFILL_EXL3_GEMM"); return v && std::string(v) == "1"; }();
  return on;
}
static int prefill_exl3_gemm_min() {
  static const int n = [] {
    const char *v = std::getenv("DS41_PREFILL_EXL3_GEMM_MIN");
    const int x = v && *v ? std::atoi(v) : 32;
    if (x < 1 || x > 1000000) throw std::invalid_argument("DS41_PREFILL_EXL3_GEMM_MIN must be 1..1000000");
    return x;
  }();
  return n;
}
static bool prefill_exl3_rotated() {
  static const bool on = [] { const char *v = std::getenv("DS41_PREFILL_EXL3_ROTATED"); return !v || std::string(v) != "0"; }();
  return on;
}
struct PrefillData {
  size_t rows;
  int64_t begin, end;
  bool cold;
  int candidate_width;
  int64_t candidate_begin;
  int dense_batch=1;
  bool warm=true;
  ds41e::prefill::Span query{0,0};
  Device hidden, previous_pre, ffn_input, ffn_post, ffn_comb;
  Device window, selected, candidates, ids, weights, quant, scales, routed, capture;
  Device bx{256*H}, bs{256*H/32}, bw{256*4};
  Device bxh; // EXL3: 256 gathered BF16 rows
  Device rg, ru, rd; // s0ma change 68: the rows rotated once for gate, up (256 x H FP16) and down (256 x I)
  // s0ma change 76: a lane's wave scratch (variant v: 0 the main stream, else side lane v-1), freed with the prompt
  struct Wave { Device x, bw, gate, up, act, down, rg, ru, rd; int rows = 0; };
  std::vector<Wave> waves;
  bool waves_off = false;
  // s0ma change 82: per variant, an expert's three projections decoded to FP16 [k, n], the GEMM's FP32 rows, and
  // cuBLAS's workspace; freed with the prompt
  struct Gemm { Device wg, wu, wd, c, ws, entry; };   // entry: the expert's table entry, copied to the device
  std::vector<Gemm> gemm;
  bool gemm_ready = false, gemm_tried = false;
  std::atomic<size_t> gemm_experts{0}, gemm_rows{0};   // (two enqueuing threads)
  Device gate{256*I*2}, up{256*I*2}, act{256*I*2},
      eq{256*I}, es{256*I/32}, down{256*H*2};
  std::array<std::vector<int64_t>,2> rowids;
  // 1 for rows inside an image span: no Engram contribution, n-grams stop at them, vision routing bias.
  std::vector<uint8_t> image;
  // s0ma change 55: both point into pinned buffers the engine keeps between prompts (prefill_pinned)
  ds41::HostBuffer* route_host=nullptr;
  ds41::HostBuffer* descriptors=nullptr;
  // s0ma change 86: the layer's job descriptors and route slots, and the waves', copied into VRAM (see pdesc)
  Device desc_dev, wave_dev;
  // s0ma change 89: per variant (0 the main stream, else side lane v-1), a VRAM expert table holding the entries of the
  // staged (and change-77 arena) experts that variant runs, each copied in just before its kernels
  std::vector<Device> lane_table;
  Json layers=Json::array();
  std::array<std::unique_ptr<EngramAsync::Ticket>,2> engram;
  std::array<const char*,2> engram_raw{};
  double engram_barrier_ms=0;
  static size_t required_bytes(size_t n,int64_t b,size_t tail) {
    const int64_t end=b+int64_t(n);
    const int candidate_width=std::min<int64_t>(2048,(end+7)/8)*8;
    const int64_t candidate_begin=ds41e::prefill::plan(b,n,tail)[20].query.begin;
    return n*(size_t(H)*(4*2+2+6*2+1)+H/32+512*4+6*8+24*4)
        +(n+128)*528+size_t(end-candidate_begin)*candidate_width*4
        +std::min<size_t>(128,n)*3*H*2+(256<<20);
  }
  PrefillData(size_t n,int64_t b,bool c,size_t tail,size_t budget):rows(n),begin(b),end(b+n),cold(c),
      candidate_width(std::min<int64_t>(2048,(end+7)/8)*8),
      candidate_begin(ds41e::prefill::plan(b,n,tail)[20].query.begin) {
    // Largest arrays are explicitly bounded by new rows, never prior context.
    const size_t need=required_bytes(n,b,tail);
    if(need>budget)throw std::invalid_argument("prefill activation budget exceeded: need "+
        std::to_string(need)+" bytes; declare sufficient reserved scratch without evicting decode cache");
    size_t free,total;CK(cudaMemGetInfo(&free,&total));
    if(need>free || free-need<(64<<20))
      throw std::runtime_error("prefill reserved scratch does not fit beside the live decode cache");
    hidden.resize(n*HC*H*2);previous_pre.resize(n*4*4);
    ffn_input.resize(n*H*2);ffn_post.resize(n*4*4);ffn_comb.resize(n*16*4);
    window.resize((n+128)*528);selected.resize(n*512*4);
    candidates.resize(size_t(end-candidate_begin)*candidate_width*4);
    ids.resize(n*6*4);weights.resize(n*6*4);
    quant.resize(n*H);scales.resize(n*H/32);routed.resize(n*6*H*2);
    capture.resize(std::min<size_t>(128,n)*3*H*2);
  }
};

// v0.4 atomically retains only resident IDs, without admission or policy
// credit. The v0.3 fallback is kept for its frozen historical gate bundle.
template<class Store> typename Store::Lease pborrow(Store& store,const std::vector<ds41::ExpertId>& ids) {
  if constexpr(Store::api_version>=4)return store.borrow_vram(ids,stream);
  else {
    std::vector<ds41::ExpertId> hits;
    for(auto id:ids)if(store.residency(id)>=0)hits.push_back(id);
    return hits.empty()?typename Store::Lease{}:store.need_now(hits,stream);
  }
}
template<class Store> std::vector<uint8_t> p_residency(Store& store) {
  if constexpr(Store::api_version>=4)return store.residency_snapshot();
  else return {};
}
// s0ma change 76: group, when given, takes a lane's whole share of the resident experts at once (grouped launches)
using PGroup=std::function<void(const std::vector<ds41::ExpertId>&,const ds41::DeviceExpert*,cudaStream_t)>;
// s0ma change 77: a layer's RAM experts copied while its attention runs. After each chunk's routing (cproject), the
// chunk's route ids come back to pinned memory under an event; a worker thread waits for them in order and copies
// each newly routed expert that is in the RAM tier and was not in VRAM at the layer's start into a per-prompt VRAM
// arena (DS41_PREFILL_PREFETCH_MIB, default 768; 0 turns it off), on its own copy stream, while the GPU works on
// the next chunks. pstream then serves those experts from the arena (the consumer waits on the copy's event)
// instead of the stager. The bytes are the RAM tier's own: what is computed does not change.
struct PrefillPrefetch {
  struct Job { int layer; cudaEvent_t ready; const int32_t *ids; int rows; };
  ds41::HostBuffer *table = nullptr, *ids = nullptr;   // pinned buffers the engine keeps between prompts
  Device arena;
  size_t used = 0, cap = 0;
  cudaStream_t copy = nullptr;
  // two pools, one a thread: chunk events (the main thread) and copy events (the worker), reused a layer later
  std::vector<cudaEvent_t> chunk_events, copy_events;
  size_t next_chunk = 0, next_copy = 0;
  std::thread worker;
  std::mutex m;
  std::condition_variable cv, idle;
  std::deque<Job> q;
  bool stop = false, busy = false;
  int layer = -1;
  std::vector<uint8_t> resident;            // VRAM membership at the layer's start
  std::unordered_map<ds41::ExpertId, cudaEvent_t> landed;
  std::unordered_set<ds41::ExpertId> seen;
  std::deque<ds41::OffloadStore::HostLease> leases;   // (a deque: a chunk keeps a pointer to its own)
  int device = 0;
  uint64_t copied = 0, bytes = 0, served = 0;
  static cudaEvent_t take(std::vector<cudaEvent_t> &pool, size_t &next) {
    if (next == pool.size()) { cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); pool.push_back(e); }
    return pool[next++];
  }
  ~PrefillPrefetch() {
    { std::lock_guard<std::mutex> g(m); stop = true; }
    cv.notify_all();
    if (worker.joinable()) worker.join();
    if (copy) { cudaStreamSynchronize(copy); cudaStreamDestroy(copy); }
    for (auto e : chunk_events) cudaEventDestroy(e);
    for (auto e : copy_events) cudaEventDestroy(e);
  }
};
std::unique_ptr<ds41::HostBuffer> prefill_prefetch_table, prefill_prefetch_ids;
std::unique_ptr<PrefillPrefetch> prefetch;
int prefetch_layer=-1;   // the layer pmoe is sweeping (pstream reads it)
bool prefill_waves_ready=false;   // s0ma change 77: every lane's wave scratch allocated for this sweep
Json prefill_summary_prefetch;
std::unordered_map<ds41::ExpertId, const ds41::Record *> prefetch_records;
static size_t prefill_prefetch_bytes() {
  static const size_t b = [] {
    const char *v = std::getenv("DS41_PREFILL_PREFETCH_MIB");
    const long x = v && *v ? std::atol(v) : 0;   // change 78: off unless asked (measured no gain)
    if (x < 0 || x > 16384) throw std::invalid_argument("DS41_PREFILL_PREFETCH_MIB must be 0..16384");
    return size_t(x) << 20;
  }();
  return b;
}
// at a prompt's start: the arena (smaller if VRAM is short; none if it cannot be had) and the worker
void prefetch_begin(size_t rows) {
  prefetch.reset();
  size_t want = prefill_prefetch_bytes();
  if (!want || !exl3 || prefill_records.empty()) return;
  if (prefetch_records.empty()) for (const auto &r : prefill_records) prefetch_records[r.id] = &r;
  auto P = std::make_unique<PrefillPrefetch>();
  // leave room for what the prompt still allocates (the stager's ring on a first prompt, the lanes' wave scratch)
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));
  const size_t margin = size_t(1) << 30;
  want = std::min(want, free_b > margin ? free_b - margin : 0);
  for (; want >= (size_t(128) << 20); want /= 2) {
    try { P->arena.resize(want); P->cap = want; break; }
    catch (const std::exception &) { (void)cudaGetLastError(); }   // a failed cudaMalloc leaves its error behind
  }
  if (!P->cap) { std::cerr << "prefill prefetch off for this prompt (no VRAM for its arena)\n"; return; }
  P->table = prefill_pinned(prefill_prefetch_table, ds41::expert_count * sizeof(ds41::DeviceExpert));
  P->ids = prefill_pinned(prefill_prefetch_ids, rows * 6 * 4);
  CK(cudaStreamCreateWithFlags(&P->copy, cudaStreamNonBlocking));
  CK(cudaGetDevice(&P->device));
  auto *self = P.get();
  P->worker = std::thread([this, self] { prefetch_loop(*self); });
  prefetch = std::move(P);
}
void prefetch_loop(PrefillPrefetch &P) {
  try { CK(cudaSetDevice(P.device)); } catch (...) { return; }
  for (;;) {
    PrefillPrefetch::Job j;
    {
      std::unique_lock<std::mutex> g(P.m);
      P.cv.wait(g, [&] { return P.stop || !P.q.empty(); });
      if (P.stop) return;
      j = P.q.front(); P.q.pop_front(); P.busy = true;
    }
    try {
      CK(cudaEventSynchronize(j.ready));
      std::vector<ds41::ExpertId> want;   // this chunk's newly routed experts, not in VRAM at the layer's start
      for (int k = 0; k < j.rows * 6; ++k) {
        const auto id = ds41::expert_id(unsigned(j.layer), uint32_t(j.ids[k]));
        if (!P.seen.insert(id).second) continue;
        if (id < P.resident.size() && (P.resident[id] & 1)) continue;
        if (id < P.resident.size() && !(P.resident[id] & 2)) continue;   // not in the RAM tier: the stager reads it
        if (prefetch_records.count(id)) want.push_back(id);
      }
      if (want.empty()) goto next;
      {
        // one store round trip a chunk; kept before any copy reads it (released after the layer's sweep)
        const ds41::OffloadStore::HostLease *lease = nullptr;
        { std::lock_guard<std::mutex> g(P.m); P.leases.push_back(experts->borrow_host(want)); lease = &P.leases.back(); }
        for (const auto &v : lease->views) {
          if (!v.data) continue;
          const auto &r = *prefetch_records.at(v.id);
          const size_t need = (r.bytes + 4095) & ~size_t(4095);
          if (P.used + need > P.cap) break;   // the arena is full for this layer
          auto *base = P.arena.as<uint8_t>(P.used);
          P.used += need;
          static_cast<ds41::DeviceExpert *>(P.table->data())[v.id] = {base + r.gate.offset, base + r.gate_scale.offset,
              base + r.up.offset, base + r.up_scale.offset, base + r.down.offset, base + r.down_scale.offset, nullptr, 0, -1};
          CK(cudaMemcpyAsync(base, v.data, r.bytes, cudaMemcpyHostToDevice, P.copy));
          cudaEvent_t done = PrefillPrefetch::take(P.copy_events, P.next_copy);
          CK(cudaEventRecord(done, P.copy));
          std::lock_guard<std::mutex> g(P.m);
          P.landed[v.id] = done;
          ++P.copied; P.bytes += r.bytes;
        }
      }
      next:;
    } catch (const std::exception &e) {
      std::cerr << "prefill prefetch: " << e.what() << "\n";
    }
    { std::lock_guard<std::mutex> g(P.m); P.busy = false; }
    P.idle.notify_all();
  }
}
// cproject, after a chunk's routes: hand its ids to the worker (layers that pmoe will sweep)
void prefetch_chunk(int layer, size_t slot, int rows) {
  if (!prefetch || rows < 1) return;
  auto &P = *prefetch;
  if (P.layer != layer) {   // a new layer: the previous one's arena was released by pmoe
    P.layer = layer;
    P.resident = experts->residency_snapshot();
  }
  auto *host = static_cast<int32_t *>(P.ids->data()) + slot * 6;
  CK(cudaMemcpyAsync(host, pref->ids.as<char>(slot * 6 * 4), size_t(rows) * 6 * 4, cudaMemcpyDeviceToHost, stream));
  cudaEvent_t e = PrefillPrefetch::take(P.chunk_events, P.next_chunk);
  CK(cudaEventRecord(e, stream));
  { std::lock_guard<std::mutex> g(P.m); P.q.push_back({layer, e, host, rows}); }
  P.cv.notify_one();
}
// pmoe, before its sweep: what the worker has copied for this layer (it finishes the chunks already handed over)
std::unordered_map<ds41::ExpertId, cudaEvent_t> prefetch_take(int layer) {
  if (!prefetch || prefetch->layer != layer) return {};
  auto &P = *prefetch;
  std::unique_lock<std::mutex> g(P.m);
  P.idle.wait(g, [&] { return P.q.empty() && !P.busy; });
  return P.landed;
}
// pmoe, after its sweep (the stream and the lanes synchronized): the arena is free for the next layer
void prefetch_release() {
  if (!prefetch) return;
  auto &P = *prefetch;
  std::unique_lock<std::mutex> g(P.m);
  P.idle.wait(g, [&] { return P.q.empty() && !P.busy; });
  CK(cudaStreamSynchronize(P.copy));
  P.landed.clear(); P.seen.clear(); P.leases.clear(); P.used = 0; P.next_chunk = P.next_copy = 0; P.layer = -1;
}
// s0ma change 84: the sweep's start (pmoe's timing event; the stager's copy timeline is measured from it), how many
// staged (RAM or drive) experts each lane was given (0 the main stream, i + 1 side lane i), and the staged lanes in use
cudaEvent_t prefill_sweep_origin=nullptr;
std::vector<size_t> prefill_lane_staged;
int staged_lanes_used=0;
template<class F> void pstream(std::vector<ds41::ExpertId> ids,F consume,const PGroup &group={}) {
  prefill_transfers={};prefill_vram_hits=0;staged_lanes_used=0;
  const int nsides=exl3?prefill_lanes()-1:0;
  prefill_lane_staged.assign(size_t(nsides)+1,0);
  int staged_lanes=0;   // s0ma change 84: side lanes 0..staged_lanes-1 take the staged experts only
  std::vector<ds41::ExpertId> main_share;   // s0ma change 84: the main stream's residents (with staged lanes)
  int side_pending=0;bool spread=false;
  std::vector<size_t> lane_load;   // s0ma change 69: experts given to each lane this layer (0: the main stream)
  std::vector<std::vector<ds41::ExpertId>> share;   // s0ma change 77: each side lane's residents
  std::function<void()> residents;
  std::vector<std::pair<ds41::ExpertId,cudaEvent_t>> early;   // s0ma change 77: misses already in the arena
  std::function<void(cudaStream_t)> early_serve;
  std::mutex lane_mutex;
  auto lease=pborrow(*experts,ids);
  std::set<ds41::ExpertId> hits;std::vector<ds41::ExpertId> misses;
  try {
    for(const auto &a:lease.allocations) {
      if(!a.hit)throw std::runtime_error("resident-only prefill borrow admitted a miss");
      hits.insert(a.id);
    }
    prefill_vram_hits=hits.size();
    for(auto id:ids)if(!hits.count(id))misses.push_back(id);
    // s0ma change 77: misses copied while the layer's attention ran come from the arena
    if(prefetch && !misses.empty() && prefetch_layer>=0 && !(vram_admit_budget>vram_admit_used)) {
      auto landed=prefetch_take(prefetch_layer);
      std::vector<ds41::ExpertId> rest;
      for(auto id:misses){auto it=landed.find(id);if(it==landed.end())rest.push_back(id);else early.push_back({id,it->second});}
      misses.swap(rest);
    }
    const int sides=exl3?prefill_lanes()-1:0;
    // s0ma change 69: staged experts on every lane too (with the side lanes in use, any expert of the layer)
    spread=sides>0 && prefill_miss_lanes() && !misses.empty();
    // s0ma change 84: DS41_PREFILL_STAGED_LANES=K: the staged experts on K side lanes of the greatest stream priority
    // (created so in prefill_lane), the residents on the main stream and the other side lanes. A staged expert's
    // kernels then get SMs ahead of the residents' waves, so its ring slot frees soon after its copy lands and the
    // copies are not held behind kernels queued on the same lane. Which lane runs an expert changes nothing computed.
    staged_lanes=spread?std::min(prefill_staged_lanes(),sides):0;
    staged_lanes_used=staged_lanes;
    if(!early.empty()) {   // s0ma change 77: each waits for its copy, then runs as a miss would (least loaded lane)
      const auto *ptable=static_cast<const ds41::DeviceExpert*>(prefetch->table->device_data());
      prefetch->served+=early.size();
      auto serve_early=[&,ptable](cudaStream_t fallback){
        for(auto &[id,ev]:early){
          cudaStream_t s=fallback;
          // (s0ma change 84: never the staged lanes 1..staged_lanes, which take the staged experts only)
          if(!lane_load.empty()){size_t best=0;for(size_t i=1;i<lane_load.size();++i)
              if(!(i<=size_t(staged_lanes)) && lane_load[i]<lane_load[best])best=i;
            ++lane_load[best];s=best?prefill_lane(best-1).s:stream;}
          CK(cudaStreamWaitEvent(s,ev,0));
          consume(id,ptable,s);
        }
      };
      early_serve=serve_early;
    }
    // s0ma change 77: the residents' work is enqueued by `residents` (below, after the staged copies have started)
    if(sides>0 && (!hits.empty() || spread || !early.empty())) {
      // s0ma change 27: resident experts on the side lanes, round robin, beside the staged sweep below; joined (and
      // their lease retired) before this layer's epilogue reads `routed`
      cudaEvent_t start=prefill_lane(size_t(sides-1)).done;   // any lane's event: recorded, then waited at once
      CK(cudaEventRecord(start,stream));
      for(int i=0;i<sides;++i)CK(cudaStreamWaitEvent(prefill_lane(size_t(i)).s,start,0));
      size_t k=0;
      lane_load.assign(size_t(sides)+1,0);
      share.assign(static_cast<size_t>(sides),{});
      if(staged_lanes>0) {   // s0ma change 84: round robin over the main stream and side lanes staged_lanes..sides-1
        const size_t lanes=size_t(sides-staged_lanes)+1;
        for(auto id:hits){const size_t at=k%lanes;
          if(!at){main_share.push_back(id);++lane_load[0];}
          else{share[size_t(staged_lanes)+at-1].push_back(id);++lane_load[size_t(staged_lanes)+at];}
          ++k;}
      } else
      for(auto id:hits){const size_t at=k%size_t(sides);share[at].push_back(id);++lane_load[at+1];++k;}
      residents=[&,sides]{
        if(!main_share.empty()) {   // s0ma change 84
          if(group)group(main_share,experts->table(),stream);
          else for(auto id:main_share)consume(id,experts->table(),stream);
        }
        for(int i=0;i<sides;++i){
          const std::vector<ds41::ExpertId> &mine=share[size_t(i)];
          if(mine.empty())continue;
          if(group)group(mine,experts->table(),prefill_lane(size_t(i)).s);   // s0ma change 76
          else for(auto id:mine)consume(id,experts->table(),prefill_lane(size_t(i)).s);
        }
        for(int i=0;i<sides;++i){auto &L=prefill_lane(size_t(i));CK(cudaEventRecord(L.done,L.s));}
      };
      side_pending=sides;
    } else residents=[&]{
      if(group && !hits.empty()) {
        std::vector<ds41::ExpertId> all(hits.begin(),hits.end());
        group(all,experts->table(),stream);
      } else for(auto id:hits)consume(id,experts->table(),stream);

      if(lease.serial)experts->retire(std::move(lease),stream);
    };
  }catch(...){cudaStreamSynchronize(stream);for(auto &L:prefill_side)cudaStreamSynchronize(L->s);side_pending=0;
    if(lease.serial&&lease.owner)experts->retire(std::move(lease),stream);throw;}
  auto run_residents=[&]{
    try{
      // s0ma change 77: the arena's experts first, on this thread, before the staged copies' thread can enqueue on
      // the same lanes (their per-expert scratch is the lane's)
      if(early_serve){auto f=std::move(early_serve);early_serve=nullptr;f(stream);}
      residents();}
    catch(...){cudaStreamSynchronize(stream);for(auto &L:prefill_side)cudaStreamSynchronize(L->s);side_pending=0;
      if(lease.serial&&lease.owner)experts->retire(std::move(lease),stream);throw;}
  };
  auto join=[&]{   // s0ma change 27: the main stream waits for the side lanes, then the resident lease ends
    if(!side_pending)return;
    for(int i=0;i<side_pending;++i)CK(cudaStreamWaitEvent(stream,prefill_side[size_t(i)]->done,0));
    side_pending=0;
    if(lease.serial)experts->retire(std::move(lease),stream);
  };
  // s0ma change 69: a staged expert, once landed, goes to the lane with the fewest experts so far (the main stream
  // first, while the side lanes run the residents). Each lane has its own scratch; each expert writes only its own
  // rows of `routed`, so which lane runs it changes nothing in the result.
  std::function<cudaStream_t(ds41::ExpertId)> pick;
  if(spread&&staged_lanes>0)pick=[&](ds41::ExpertId)->cudaStream_t{   // s0ma change 84: the staged lanes, in turn
    std::lock_guard<std::mutex> g(lane_mutex);
    size_t best=1;for(size_t i=2;i<=size_t(staged_lanes);++i)if(prefill_lane_staged[i]<prefill_lane_staged[best])best=i;
    ++prefill_lane_staged[best];++lane_load[best];return prefill_lane(best-1).s;
  };
  else if(spread)pick=[&](ds41::ExpertId)->cudaStream_t{
    std::lock_guard<std::mutex> g(lane_mutex);   // s0ma change 77: the arena's experts take lanes too
    size_t best=0;for(size_t i=1;i<lane_load.size();++i)if(lane_load[i]<lane_load[best])best=i;
    ++lane_load[best];++prefill_lane_staged[best];return best?prefill_lane(best-1).s:stream;
  };
  else prefill_lane_staged[0]=misses.size();   // (the early arena's experts are not counted)
  if(!misses.empty()) {
    if(!prefill_stage)build_prefill_stage("the first prompt's sweep");   // (s0ma change 87: normally built at load)
    auto lookup=config.prefill_host_view;
    auto lease=std::make_shared<ds41::OffloadStore::HostLease>(experts->borrow_host(misses));
    if(!prompt_bits.empty())for(const auto &v:lease->views)if(!v.data&&v.id<15360)prompt_bits[v.id]|=1;   // s0ma change 56
    if(!lookup)lookup=[lease](uint32_t id){
      for(const auto &v:lease->views)if(v.id==id&&v.data)
        return ds41e::prefill::HostView{v.data,v.record.bytes,lease};
      return ds41e::prefill::HostView{};
    };
    // s0ma change 57: of this layer's drive reads, those the prompt's last 256 rows routed (most routed first) go into
    // the RAM tier as they pass, within this layer's share of what is left of the prompt's budget
    ds41::OffloadStore::Admission adm;
    std::unordered_map<ds41::ExpertId,void*> keep_at;
    if(admit_budget>admit_used&&pref&&pref->warm&&!prompt_tail_count.empty()&&!ids.empty()&&ids.front()<15360) {
      std::vector<std::pair<int,ds41::ExpertId>> rank;
      for(const auto &v:lease->views)if(!v.data&&v.id<15360&&prompt_tail_count[v.id])rank.push_back({-int(prompt_tail_count[v.id]),v.id});
      if(!rank.empty()) {
        std::sort(rank.begin(),rank.end());
        std::vector<ds41::ExpertId> want;std::vector<uint32_t> credit;
        for(auto [c,id]:rank){want.push_back(id);credit.push_back(uint32_t(-c));}
        const unsigned l=ds41::expert_layer(ids.front());
        adm=experts->admit_begin(want,credit,(admit_budget-admit_used)/std::max(1u,40u-l));
        for(size_t i=0;i<adm.ids.size();++i)keep_at[adm.ids[i]]=adm.dst[i];
      }
    }
    ds41e::prefill::Stager::Keep keep;
    if(!keep_at.empty())keep=[&keep_at](ds41::ExpertId id)->void*{auto it=keep_at.find(id);return it==keep_at.end()?nullptr:it->second;};
    // s0ma change 60: the same ranking into VRAM, from RAM or the drive, within this layer's share of the VRAM budget
    ds41::OffloadStore::Admission vadm;
    std::unordered_map<ds41::ExpertId,void*> vkeep_at;
    if(vram_admit_budget>vram_admit_used&&pref&&pref->warm&&!prompt_tail_count.empty()&&!ids.empty()&&ids.front()<15360) {
      std::vector<std::pair<int,ds41::ExpertId>> rank;
      for(auto id:misses)if(id<15360&&prompt_tail_count[id])rank.push_back({-int(prompt_tail_count[id]),id});
      if(!rank.empty()) {
        std::sort(rank.begin(),rank.end());
        std::vector<ds41::ExpertId> want;std::vector<uint32_t> credit;
        for(auto [c,id]:rank){want.push_back(id);credit.push_back(uint32_t(-c));}
        const unsigned l=ds41::expert_layer(ids.front());
        vadm=experts->admit_vram_begin(want,credit,(vram_admit_budget-vram_admit_used)/std::max(1u,40u-l));
        for(size_t i=0;i<vadm.ids.size();++i)vkeep_at[vadm.ids[i]]=vadm.dst[i];
      }
    }
    ds41e::prefill::Stager::Keep keep_vram;
    if(!vkeep_at.empty())keep_vram=[&vkeep_at](ds41::ExpertId id)->void*{auto it=vkeep_at.find(id);return it==vkeep_at.end()?nullptr:it->second;};
    // s0ma change 77: the staged copies start first (on their own thread) and the residents are enqueued meanwhile:
    // a layer's first RAM copy no longer waits for ~1,000 resident launches. Only when staged experts go to the
    // lanes too (the residents and the staged experts then share no scratch: waves have their own).
    auto staged=[&]{return prefill_stage->stream(misses,stream,consume,lookup,[&]{p_engram_barrier();},keep,keep_vram,pick,
                                                 prefill_sweep_origin);};
    // only with every lane's wave scratch in hand: then the residents never use a lane's per-expert scratch, which
    // the staged experts' thread uses meanwhile
    if(spread && prefill_copies_first() && group && prefill_waves_ready) {
      int device=0;CK(cudaGetDevice(&device));
      if(early_serve){auto f=std::move(early_serve);early_serve=nullptr;f(stream);}   // before the copies' thread
      auto copies=std::async(std::launch::async,[&,device]{CK(cudaSetDevice(device));return staged();});
      try{run_residents();}catch(...){try{copies.get();}catch(...){}throw;}
      try{prefill_transfers=copies.get();}
      catch(...){for(auto &L:prefill_side)cudaStreamSynchronize(L->s);join();throw;}
    } else {
      run_residents();
      try{prefill_transfers=staged();}
      catch(...){for(auto &L:prefill_side)cudaStreamSynchronize(L->s);join();throw;}
    }
    if(spread)for(auto &L:prefill_side)CK(cudaEventRecord(L->done,L->s));   // change 69: after the staged work too
    if(vadm.token) {
      if(prefill_transfers.kept_vram_ram+prefill_transfers.kept_vram_disk!=vadm.ids.size())
        throw std::logic_error("prompt VRAM admission: an admitted expert was not copied");
      vram_admit_used+=vadm.bytes;
      experts->admit_vram_end(std::move(vadm));
    }
    if(adm.token) {
      if(prefill_transfers.kept!=adm.ids.size())
        throw std::logic_error("prompt admission: an admitted expert's bytes were not copied");
      admit_used+=adm.bytes;admit_experts+=adm.ids.size();admit_bytes+=adm.bytes;admit_copy_ms+=prefill_transfers.keep_ms;
      experts->admit_end(std::move(adm));
    }
  }
  else run_residents();
  join();
}
void p_engram_barrier() {
  if(!async_engram)return;
  double start=clock_ms();
  for(int j=0;j<2;++j)if(!pref->engram_raw[j])
    pref->engram_raw[j]=static_cast<const char*>(pref->engram[j]->wait(pref->rows*24));
  pref->engram_barrier_ms+=clock_ms()-start;
}
// Image rows among [first, first+rows) select experts with the vision bias: recompute their biased scores.
void pimage_router(int l,int64_t first,int rows,const float* router,float* unbiased,float* biased) {
  auto &pf=*pref;const size_t base=size_t(first-pf.begin);
  for(int r=0;r<rows;) {
    if(!pf.image[base+r]){++r;continue;}
    int e=r;while(e<rows&&pf.image[base+e])++e;
    CK(ds41_dense_router_scores(router+size_t(r)*384,w("layers."+std::to_string(l)+".ffn.gate.bias_vl").data.as<float>(),
        unbiased+size_t(r)*384,biased+size_t(r)*384,e-r,384,stream));
    r=e;
  }
}
void pcopy(void* dst,const void* src,size_t bytes) {
  CK(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));
}
// One row retains the accepted dense M1 geometry. Cold attention alone changes
// to the official full-N layout. Full-M cold cuBLAS equivalence is not claimed.
void pattention(int l,int64_t pos,bool output_projection=true) {
  auto &pf=*pref;
  std::string p="layers."+std::to_string(l)+".attn";
  apply_rope(q,64,512,l,pos);apply_rope(kv,1,512,l,pos);
  CK(ds41_attn_pack(kv.p,packed.p,1,DS41_SWA_FP8,stream));
  CK(ds41_attn_ring_write(packed.p,pf.window.p,528,pos,1,int(pf.rows+128),stream));
  int owner=l<2?-1:l<8?2:l<14?8:l<20?14:20;
  int ratio=l<20?2:1;
  int visible_n=owner<0?0:int((pos+1)/ratio);
  int n=owner<0?0:(pf.cold?int(pf.end/ratio):visible_n);
  int k=std::min(512,n);
  bool index=l==2||l==8||l==14||l==20||l==24||l==28||l==32||l==36;
  if(globals.count(l)) {
    auto &g=*globals.at(l);bool complete=ratio==1||pos%2==1;
    if(ratio==1)linear(p+".compressor.wkv",x.p,latent.p,512,H);
    else {
      bf16_float(x.p,f32.as<float>(),H,stream);auto off=(pos%8)*512*4;
      linear(p+".compressor.wkv",f32.p,g.raw_kv.as<char>(off),512,H);
      linear(p+".compressor.wgate",f32.p,g.raw_score.as<char>(off),512,H);
      if(complete)CK(ds41_attn_pool2(g.raw_kv.as<float>(),g.raw_score.as<float>(),latent.p,pos,1,8,stream));
    }
    if(complete) {
      norm(p+".compressor.norm",latent.p,latent.p,512);
      linear(p+".indexer.wk",latent.p,ik.p,128,512);
      norm(p+".indexer.k_norm",ik.p,ik.p,128);
      apply_rope(ik,1,128,l,pos+1-ratio);
      CK(ds41_attn_pack(ik.p,g.key.as<char>(size_t(visible_n-1)*68),1,DS41_INDEX_FP4,stream));
      apply_rope(latent,1,512,l,pos+1-ratio);
      CK(ds41_attn_pack(latent.p,g.kv.as<char>(size_t(visible_n-1)*288),1,DS41_GLOBAL_FP4,stream));
    }
  }
  if(!pf.query.contains(pos))return;
  size_t row=size_t(pos-pf.begin);
  int scan=l>20?pf.candidate_width:n;
  if(index&&n) {
    auto &g=*globals.at(owner);
    if(l>20)pcopy(candidate_ids.p,pf.candidates.as<char>(size_t(pos-pf.candidate_begin)*pf.candidate_width*4),pf.candidate_width*4);
    linear(p+".indexer.wq_b",qa.p,iq.p,4096,1280);apply_rope(iq,32,128,l,pos);
    CK(ds41_attn_roundtrip(iq.p,32,128,DS41_INDEX_FP4,stream));
    linear(p+".indexer.weights_proj",x.p,iw.p,32,H);scale_bf16(iw.p,32,1.f/64,stream);
    upload(visible,&visible_n,4);
    CK(ds41_attn_index_scores(iq.p,g.key.p,iw.p,l>20?candidate_ids.as<int32_t>():nullptr,
        visible.as<int32_t>(),scores.p,1,scan,stream));
    if(l==20) {
      int nb=(n+7)/8,kb=std::min(2048,nb),vb=(visible_n+7)/8;
      CK(ds41_attn_block_scores(scores.p,visible.as<int32_t>(),blocks.p,1,n,stream));
      upload(block_visible,&vb,4);
      CK(ds41_attn_topk(blocks.p,block_visible.as<int32_t>(),chosen_blocks.as<int32_t>(),1,nb,kb,top_scratch.p,stream));
      CK(ds41_attn_expand_blocks(chosen_blocks.as<int32_t>(),candidate_ids.as<int32_t>(),1,kb,n,stream));
      if(pos>=pf.candidate_begin) {
        auto* dst=pf.candidates.as<char>(size_t(pos-pf.candidate_begin)*pf.candidate_width*4);
        CK(cudaMemsetAsync(dst,0xff,pf.candidate_width*4,stream));
        pcopy(dst,candidate_ids.p,kb*8*4);
      }
    }
    if(l>20) {
      upload(block_visible,&scan,4);
      CK(ds41_attn_topk(scores.p,block_visible.as<int32_t>(),candidate_top.as<int32_t>(),1,scan,k,top_scratch.p,stream));
      CK(ds41_attn_remap_topk(candidate_top.as<int32_t>(),candidate_ids.as<int32_t>(),top.as<int32_t>(),1,k,scan,stream));
    }else CK(ds41_attn_topk(scores.p,visible.as<int32_t>(),top.as<int32_t>(),1,n,k,top_scratch.p,stream));
    pcopy(pf.selected.as<char>(row*512*4),top.p,k*4);
  }else if(index&&k) {
    CK(cudaMemsetAsync(top.p,0xff,k*4,stream));pcopy(pf.selected.as<char>(row*512*4),top.p,k*4);
  }else if(!index&&k)pcopy(top.p,pf.selected.as<char>(row*512*4),k*4);
  auto*g=owner<0?nullptr:globals.at(owner).get();
  int width=pf.cold?int(std::min<int64_t>(128,pf.end)):128;
  if(pf.cold) {
    upload(position_device,&pos,8);
    CK(ds41_attn_gather_prefill_at(pf.window.p,int(pf.rows+128),g?g->kv.p:nullptr,g?g->capacity:0,
        k?top.as<int32_t>():nullptr,k,position_device.as<int64_t>(),1,ratio,width,gather.p,local_idx.as<int32_t>(),stream));
  }else CK(ds41_attn_gather(pf.window.p,int(pf.rows+128),g?g->kv.p:nullptr,g?g->capacity:0,
      k?top.as<int32_t>():nullptr,k,pos,1,ratio,gather.p,local_idx.as<int32_t>(),stream));
  // Both frozen sparse implementations use the same arithmetic; keep decode
  // split dispatch for append and cold single-pass per its component contract.
  if(!pf.cold&&owner>=0)CK(ds41_attn_sparse_split(q.p,gather.p,w(p+".attn_sink").data.as<float>(),
      local_idx.as<int32_t>(),o.p,attn_scores.as<float>(),1,width+k,16,stream));
  else CK(ds41_attn_sparse(q.p,gather.p,w(p+".attn_sink").data.as<float>(),local_idx.as<int32_t>(),o.p,1,width+k,16,stream));
  apply_rope(o,64,512,l,pos,true);CK(ds41_attn_roundtrip(o.p,1,32768,DS41_SWA_FP8,stream));
  if(!output_projection)return;
  if(!dense_wo_a(p,o.p,wo.p,1))   // s0ma change 34: Sero's wo_a slices when loaded
    CK(ds41_dense_cublas(blas,o.p,woa_weight(p+".wo_a"),wo.p,1,1024,4096,0,8,stream));
  linear(p+".wo_b",wo.p,y.p,H,8192);
}
// s0ma change 55: the prompt's pinned host scratch, kept and grown instead of allocated per prompt and per layer. Each
// HostBuffer is a 2 MiB-aligned MADV_HUGEPAGE mapping, faulted in and cudaHostRegister'ed: 41 of them a prompt,
// each a huge-page fault that compacts memory under pressure (a 48 GiB tier: 5,906 compaction stalls, 99% failed).
// Callers reach these only after the stream has finished with the previous contents.
std::unique_ptr<ds41::HostBuffer> prefill_route_buf,prefill_desc_buf,prefill_wave_buf;   // (wave: change 76)
// s0ma change 26: DS41_PREFILL_STAGE_SLOTS=N (2..64; published 8) deepens the prompt sweep's staging ring. Sero's
// 1K+ prompts read the drive at 2.8-3.8 GB/s of its 7.2 over their sweeps: each slot's next read waits for the
// GEMM behind it. A slot is the largest record in VRAM and pinned RAM (~22 MB with K5 records).
size_t prefill_stage_slots() const {
  size_t n=8;
  if(const char* v=std::getenv("DS41_PREFILL_STAGE_SLOTS");v&&*v){n=size_t(std::strtoul(v,nullptr,10));
    if(n<2||n>64)throw std::invalid_argument("DS41_PREFILL_STAGE_SLOTS must be 2..64");}
  return n;
}
size_t prefill_stage_bytes() const {   // its VRAM slots (and as much pinned RAM): n x the largest record, 4 KiB rounded
  size_t r=0;for(const auto &x:prefill_records)r=std::max(r,(size_t(x.bytes)+4095)&~size_t(4095));
  return prefill_stage_slots()*r;
}
// s0ma change 87: the staging ring (VRAM slots, a pinned staging buffer, the mapped table) was built at the first
// prompt's first sweep, inside layer 0's timeline: ~0.6 s before that layer's first copy (2026-11-03: first_copy_ms
// 577.7 at layer 0, under 1.3 at every other layer; the pinned huge-page buffer faults and compacts under memory
// pressure). Now built at the end of load when free VRAM covers it plus 256 MiB (DS41_PREFILL_STAGER_AT_LOAD=0 keeps it
// lazy); the same ring
// and the same VRAM, which the first prompt took from the same margin. The log line says where and how long.
void build_prefill_stage(const char *when) {
  const double t=clock_ms();
  prefill_stage=std::make_unique<ds41e::prefill::Stager>(prefill_records,prefill_stage_slots());
  std::cerr<<"prefill stager: built at "<<when<<" in "<<clock_ms()-t<<" ms ("<<prefill_stage_slots()<<" slots, "
           <<prefill_stage->device_bytes()/1048576.0<<" MiB of VRAM and as much pinned RAM)\n";
}
void prefill_stage_at_load() {
  static const bool on=[]{const char *v=std::getenv("DS41_PREFILL_STAGER_AT_LOAD");return !(v&&v[0]=='0');}();
  if(!on||!exl3||prefill_records.empty()||prefill_stage)return;   // (EXL3 banks; MXFP4 keeps the lazy build)
  try{   // anything that fails here (a bad DS41_PREFILL_STAGE_SLOTS too) is left for the first prompt, as before
    size_t free=0,total=0;CK(cudaMemGetInfo(&free,&total));
    const size_t need=prefill_stage_bytes(),margin=size_t(256)<<20;   // (change 88: 512 -> 256, decode's own floor)
    if(free<need+margin){
      std::cerr<<"prefill stager: left for the first prompt ("<<free/1048576<<" MiB free at load, "<<(need+margin)/1048576
               <<" MiB wanted)\n";
      return;
    }
    build_prefill_stage("load");
  }catch(const std::exception &e){prefill_stage.reset();(void)cudaGetLastError();
    std::cerr<<"prefill stager: not built at load ("<<e.what()<<"); the first prompt builds it\n";}
}
ds41::HostBuffer* prefill_pinned(std::unique_ptr<ds41::HostBuffer>& b,size_t n) {
  if(!b||b->size()<n)b=std::make_unique<ds41::HostBuffer>(std::max(n+n/2,size_t(2)<<20),true);
  return b.get();
}
// s0ma change 86: the sweep's kernels read their job descriptors and route slots from VRAM, not from the pinned host
// buffer the host writes them into. Every warp of the gather and scatter kernels reads its row's slot, and kernel
// reads of mapped host memory are uncached PCIe reads: ~15 million a layer for an 8K segment (gather and scatter, 160
// warps per routed row), sharing the link with the RAM tier's expert copies. ds41-probe.sh contend (2026-11-03):
// 13 MiB copies 57.3 GB/s alone, 30.9 beside kernels reading 64 B from host memory, the 28-35 GB/s the sweep's copies
// ran at. Now each layer's descriptors (and each wave's) cross once, by a zero-copy kernel on the stream that uses
// them (not a cudaMemcpyAsync, which would queue behind the expert copies on the copy engine). The same values in
// the same order: exact. DS41_PREFILL_DEVICE_DESC=0 reads them from host memory as before.
static bool prefill_device_desc() {
  static const bool on=[]{const char *v=std::getenv("DS41_PREFILL_DEVICE_DESC");return !(v&&v[0]=='0');}();
  return on;
}
// n bytes from a mapped host address (its device alias) into VRAM, on s
static void pdesc(void *dst,const void *src_alias,size_t n,cudaStream_t s) {
  if(!n)return;
  const int threads=256;
  const int blocks=int(std::min<size_t>((n+16*threads-1)/(16*threads),512));
  zc_copy<<<blocks,threads,0,s>>>(static_cast<const unsigned char*>(src_alias),static_cast<unsigned char*>(dst),n);
  CK(cudaGetLastError());
}
Json pmoe(int l,ds41e::prefill::Span span) {
  auto &pf=*pref;size_t base=size_t(span.begin-pf.begin),count=size_t(span.size());
  auto* ids=static_cast<int32_t*>(pf.route_host->data());
  double host_start=clock_ms();
  CK(cudaMemcpyAsync(ids,pf.ids.as<char>(base*6*4),count*6*4,cudaMemcpyDeviceToHost,stream));
  CK(cudaStreamSynchronize(stream));
  double boundary=clock_ms()-host_start;
  if(!prompt_bits.empty())   // s0ma change 59: every expert the prompt routed
    for(size_t r=0;r<count;++r)for(int j=0;j<6;++j)prompt_bits[ds41::expert_id(l,ids[r*6+j])]|=4;
  if(!prompt_bits.empty()&&pf.warm)   // s0ma change 56: the prompt's last 256 rows (change 57: and how often)
    for(size_t r=0;r<count;++r)if(base+r+256>=pf.rows)for(int j=0;j<6;++j){
      const auto e=ds41::expert_id(l,ids[r*6+j]);prompt_bits[e]|=2;
      if(!prompt_tail_count.empty()&&prompt_tail_count[e]<65535)++prompt_tail_count[e];}
  if(config.prefill_warm_rows && pf.warm && pf.rows>=config.prefill_warm_rows && base+count+config.prefill_warm_rows>pf.rows)
    for(size_t r=count-std::min(count,base+count+config.prefill_warm_rows-pf.rows);r<count;++r)
      for(int j=0;j<6;++j)++prefill_warm_counts[ds41::expert_id(l,ids[r*6+j])];
  auto groups=ds41e::prefill::group(ids,count);
  size_t chunks=0;
  Json distribution={{"1",0},{"2",0},{"3-8",0},{"9-32",0},{"33-256",0},{"257+",0}};
  for(auto &g:groups){auto n=g.slots.size();chunks+=(n+255)/256;
    const char* bucket=n==1?"1":n==2?"2":n<=8?"3-8":n<=32?"9-32":n<=256?"33-256":"257+";
    distribution[bucket]=distribution[bucket].get<size_t>()+1;
  }
  const size_t variants=exl3?size_t(prefill_lanes()):1;   // s0ma change 27: each lane's jobs point at its own scratch
  size_t descriptor_bytes=(variants*chunks*3*sizeof(ds41::GemmJob)+63)&~size_t(63);
  pf.descriptors=prefill_pinned(prefill_desc_buf,descriptor_bytes+count*6*4);
  auto* cpu_jobs=static_cast<ds41::GemmJob*>(pf.descriptors->data());
  auto* gpu_jobs=static_cast<ds41::GemmJob*>(pf.descriptors->device_data());
  auto* cpu_slots=reinterpret_cast<int32_t*>(static_cast<char*>(pf.descriptors->data())+descriptor_bytes);
  auto* gpu_slots=reinterpret_cast<int32_t*>(static_cast<char*>(pf.descriptors->device_data())+descriptor_bytes);
  struct Batch {size_t job,offset;int rows;};
  std::map<ds41::ExpertId,std::vector<Batch>> batches;
  std::vector<ds41::ExpertId> needed;
  size_t job=0,offset=0;
  if(exl3 && !pf.bxh.p) pf.bxh.resize(size_t(256)*H*2);
  if(exl3 && (prefill_exl3_rotated()||prefill_exl3_gemm()) && !pf.rg.p){   // s0ma change 68 (82: the GEMM path too)
    pf.rg.resize(size_t(256)*H*2);pf.ru.resize(size_t(256)*H*2);pf.rd.resize(size_t(256)*I*2);
  }
  for(auto &g:groups) {
    auto id=ds41::expert_id(l,g.expert);needed.push_back(id);
    for(size_t first=0;first<g.slots.size();first+=256) {
      int m=int(std::min<size_t>(256,g.slots.size()-first));
      std::copy_n(g.slots.data()+first,m,cpu_slots+offset);
      // EXL3: gathered BF16 rows in, BF16 activations (act) into the down projection.
      const uint8_t *x=exl3?pf.bxh.as<uint8_t>():pf.bx.as<uint8_t>(),*xs=exl3?nullptr:pf.bs.as<uint8_t>();
      cpu_jobs[job+0]={id,m,x,xs,pf.gate.as<Bf16>()};
      cpu_jobs[job+1]={id,m,x,xs,pf.up.as<Bf16>()};
      cpu_jobs[job+2]=exl3?ds41::GemmJob{id,m,pf.act.as<uint8_t>(),nullptr,pf.down.as<Bf16>()}:
          ds41::GemmJob{id,m,pf.eq.as<uint8_t>(),pf.es.as<uint8_t>(),pf.down.as<Bf16>()};
      for(size_t v=1;v<variants;++v){   // s0ma change 27: the same jobs on side lane v-1's scratch
        auto &L=prefill_lane(v-1);auto *J=cpu_jobs+v*chunks*3+job;
        J[0]={id,m,L.bxh.as<uint8_t>(),nullptr,L.gate.as<Bf16>()};
        J[1]={id,m,L.bxh.as<uint8_t>(),nullptr,L.up.as<Bf16>()};
        J[2]={id,m,L.act.as<uint8_t>(),nullptr,L.down.as<Bf16>()};
      }
      batches[id].push_back({job,offset,m});job+=3;offset+=m;
    }
  }
  if(prefill_device_desc()) {   // s0ma change 86: the descriptors into VRAM, ahead of every lane (they wait on stream)
    const size_t n=descriptor_bytes+count*6*4;
    // a failed allocation keeps this layer on the mapped descriptors (as before), never fails the prompt
    if(pf.desc_dev.bytes<n)try{pf.desc_dev.resize(std::max(n+n/2,size_t(1)<<20));}
      catch(const std::exception &e){(void)cudaGetLastError();std::cerr<<"prefill: descriptors stay in host memory ("<<e.what()<<")\n";}
    if(pf.desc_dev.bytes>=n) {
      pdesc(pf.desc_dev.p,pf.descriptors->device_data(),n,stream);
      gpu_jobs=pf.desc_dev.as<ds41::GemmJob>();
      gpu_slots=reinterpret_cast<int32_t*>(pf.desc_dev.as<char>(descriptor_bytes));
    }
  }
  if(!exl3) CK(ds41::quantize_mxfp8(pf.ffn_input.as<Bf16>(base*H*2),pf.quant.as<uint8_t>(),
      pf.scales.as<uint8_t>(),int(count),H,stream));
  auto resident_before=experts->stats();

  std::vector<std::unique_ptr<Phase>> compute;
  for(size_t i=0;i<needed.size();++i)compute.push_back(std::make_unique<Phase>());
  std::atomic<size_t> timers_used{0};   // s0ma change 76: a wave records one timer for several experts
  std::vector<int> timer_lane(needed.size(),0);   // s0ma change 84: the variant (lane) each timer ran on
  // s0ma change 76: the residents' waves. Descriptors (jobs and slots) in a pinned arena the device reads, written
  // once a layer (the previous layer's sweep has finished: pmoe synchronizes the stream before it returns)
  const int wave_rows=exl3?prefill_wave_rows():0;
  const size_t wave_desc_jobs=(chunks*3*sizeof(ds41::GemmJob)+63)&~size_t(63);
  // (s0ma change 86: each wave's slots start on a 16-byte boundary, so its copy runs in 16-byte loads: chunks * 16
  // bytes of slack, one wave per chunk at most)
  const size_t wave_desc_bytes=wave_desc_jobs+count*6*4+chunks*16+64;
  ds41::HostBuffer *wave_desc=wave_rows?prefill_pinned(prefill_wave_buf,wave_desc_bytes):nullptr;
  if(wave_desc&&prefill_device_desc()&&pf.wave_dev.bytes<wave_desc_bytes)   // s0ma change 86 (as desc_dev on failure)
    try{pf.wave_dev.resize(std::max(wave_desc_bytes+wave_desc_bytes/2,size_t(1)<<20));}
    catch(const std::exception &e){(void)cudaGetLastError();std::cerr<<"prefill: wave descriptors stay in host memory ("<<e.what()<<")\n";}
  size_t wave_jobs_used=0,wave_slots_used=0;
  if(wave_rows&&pf.waves.size()<variants)pf.waves.resize(variants);
  // s0ma change 77: every lane's wave scratch before the sweep (change 77's copies-first needs all of it)
  prefill_waves_ready=false;
  if(wave_rows&&!pf.waves_off) {
    for(auto &W:pf.waves) if(!W.rows) {
      try {
        const size_t r=size_t(wave_rows);
        W.x.resize(r*H*2);W.bw.resize(r*4);W.gate.resize(r*I*2);W.up.resize(r*I*2);W.act.resize(r*I*2);
        W.down.resize(r*H*2);W.rg.resize(r*H*2);W.ru.resize(r*H*2);W.rd.resize(r*I*2);W.rows=wave_rows;
      } catch(const std::exception &e) {
        std::cerr<<"prefill waves off for this prompt (scratch: "<<e.what()<<")\n";
        pf.waves_off=true;for(auto &X:pf.waves)X=PrefillData::Wave{};break;
      }
    }
    prefill_waves_ready=!pf.waves_off;
  }
  // s0ma change 82: the GEMM path's scratch and handles, once a prompt (off for this prompt if they do not fit)
  if(exl3&&prefill_exl3_gemm()&&!pf.gemm_tried) {
    pf.gemm_tried=true;
    try {
      for(size_t v=0;v<variants;++v) {
        cudaStream_t sv=v?prefill_lane(v-1).s:stream;
        if(prefill_blas.size()<=v) {
          cublasHandle_t h=nullptr;
          if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublasCreate");
          if(cublasSetStream(h,sv)!=CUBLAS_STATUS_SUCCESS){cublasDestroy(h);throw std::runtime_error("cublasSetStream");}
          prefill_blas.push_back(h);prefill_blas_m.push_back(std::make_unique<std::mutex>());
        }
      }
      pf.gemm.resize(variants);
      for(size_t v=0;v<variants;++v) {
        auto &G=pf.gemm[v];
        G.wg.resize(size_t(H)*I*2);G.wu.resize(size_t(H)*I*2);G.wd.resize(size_t(I)*H*2);
        G.c.resize(size_t(256)*H*4);G.ws.resize(size_t(16)<<20);G.entry.resize(sizeof(ds41::DeviceExpert));
        if(cublasSetWorkspace(prefill_blas[v],G.ws.p,G.ws.bytes)!=CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("cublasSetWorkspace");
      }
      pf.gemm_ready=true;
    } catch(const std::exception &e) {
      (void)cudaGetLastError();pf.gemm.clear();pf.gemm_ready=false;
      std::cerr<<"prefill expert GEMMs off for this prompt ("<<e.what()<<")\n";
    }
  }
  auto gemm_rows_of=[&](ds41::ExpertId id){int n=0;for(const auto &b:batches.at(id))n+=b.rows;return n;};
  Phase sweep;CK(cudaEventRecord(sweep.start,stream));
  host_start=clock_ms();std::atomic<size_t> dispatched{0};   // (atomic: change 77's two enqueuing threads)
  // the per-expert path below, for one expert; the wave path for several (falls back to the per-expert path)
  std::function<void(ds41::ExpertId,const ds41::DeviceExpert*,cudaStream_t)> one;
  PGroup group;
  if(wave_rows) group=[&](const std::vector<ds41::ExpertId> &list,const ds41::DeviceExpert* table,cudaStream_t s){
    const int v=prefill_variant(s);
    auto &W=pf.waves.at(size_t(v));
    if(!pf.waves_off && !W.rows) {
      try {
        const size_t r=size_t(wave_rows);
        W.x.resize(r*H*2);W.bw.resize(r*4);W.gate.resize(r*I*2);W.up.resize(r*I*2);W.act.resize(r*I*2);
        W.down.resize(r*H*2);W.rg.resize(r*H*2);W.ru.resize(r*H*2);W.rd.resize(r*I*2);W.rows=wave_rows;
      } catch(const std::exception &e) {
        std::cerr<<"prefill waves off for this prompt (scratch: "<<e.what()<<")\n";
        pf.waves_off=true;W=PrefillData::Wave{};
      }
    }
    if(pf.waves_off) { for(auto id:list)one(id,table,s); return; }
    std::vector<ds41::ExpertId> rest;   // s0ma change 82: the GEMM path's experts one by one, the rest in waves
    // (not with change 77's copies-first: the Stager's thread then enqueues on the same lanes, whose per-expert
    // scratch only the waves keep apart from it)
    const bool split=pf.gemm_ready&&!prefill_copies_first();
    if(split) {
      for(auto id:list) if(gemm_rows_of(id)>=prefill_exl3_gemm_min()) one(id,table,s); else rest.push_back(id);
      if(rest.empty()) return;
    }
    const auto &todo=split?rest:list;
    // every batch of these experts, by width (a width's jobs contiguous within a wave)
    struct Item {int bits;ds41::ExpertId id;const Batch *b;};
    std::vector<Item> items;
    for(auto id:todo){const int bits=width_of.empty()?expert_bits(int(id/384)):width_of[id];
      for(const auto &b:batches.at(id))items.push_back({bits,id,&b});}
    std::stable_sort(items.begin(),items.end(),[](const Item &a,const Item &b){return a.bits<b.bits;});
    const size_t timer_at=timers_used++;auto &timer=*compute.at(timer_at);timer_lane.at(timer_at)=v;
    CK(cudaEventRecord(timer.start,s));
    auto *cpu_j=static_cast<ds41::GemmJob*>(wave_desc->data());
    auto *gpu_j=static_cast<ds41::GemmJob*>(wave_desc->device_data());
    auto *cpu_sl=reinterpret_cast<int32_t*>(static_cast<char*>(wave_desc->data())+wave_desc_jobs);
    auto *gpu_sl=reinterpret_cast<int32_t*>(static_cast<char*>(wave_desc->device_data())+wave_desc_jobs);
    const bool wave_vram=prefill_device_desc()&&pf.wave_dev.bytes>=wave_desc_bytes;   // s0ma change 86: each wave's descriptors into VRAM first
    auto *dev_j=wave_vram?pf.wave_dev.as<ds41::GemmJob>():gpu_j;
    auto *dev_sl=wave_vram?reinterpret_cast<int32_t*>(pf.wave_dev.as<char>(wave_desc_jobs)):gpu_sl;
    for(size_t i=0;i<items.size();) {
      // a wave: consecutive items while their rows fit
      size_t e=i;int rows=0;
      while(e<items.size()&&rows+items[e].b->rows<=W.rows){rows+=items[e].b->rows;++e;}
      const size_t n=e-i,j0=wave_jobs_used,sl=wave_slots_used;
      if(j0+3*n>chunks*3||sl+size_t(rows)>count*6+chunks*4)throw std::logic_error("prefill wave arena");
      ds41::GemmJob *G=cpu_j+j0,*U=G+n,*D=U+n;
      int r0=0,maxr=0;
      for(size_t k=0;k<n;++k){const auto &it=items[i+k];const int m=it.b->rows;maxr=std::max(maxr,m);
        G[k]={it.id,m,W.x.as<uint8_t>(size_t(r0)*H*2),W.rg.as<uint8_t>(size_t(r0)*H*2),W.gate.as<Bf16>(size_t(r0)*I*2)};
        U[k]={it.id,m,W.x.as<uint8_t>(size_t(r0)*H*2),W.ru.as<uint8_t>(size_t(r0)*H*2),W.up.as<Bf16>(size_t(r0)*I*2)};
        D[k]={it.id,m,W.act.as<uint8_t>(size_t(r0)*I*2),W.rd.as<uint8_t>(size_t(r0)*I*2),W.down.as<Bf16>(size_t(r0)*H*2)};
        std::copy_n(cpu_slots+it.b->offset,m,cpu_sl+sl+size_t(r0));r0+=m;}
      wave_jobs_used+=3*n;wave_slots_used=(wave_slots_used+size_t(rows)+3)&~size_t(3);   // (change 86: 16-byte aligned)
      if(wave_vram){pdesc(dev_j+j0,gpu_j+j0,3*n*sizeof(ds41::GemmJob),s);pdesc(dev_sl+sl,gpu_sl+sl,size_t(rows)*4,s);}
      const ds41::GemmJob *g=dev_j+j0,*u=g+n,*d=u+n;
      ds41e::prefill::gather_routes_bf16(pf.ffn_input.as<char>(base*H*2),pf.weights.as<float>(base*6*4),dev_sl+sl,
          W.x.p,W.bw.as<float>(),rows,s);
      CK(ds41::exl3_prefill_rotate_group(table,g,u,int(n),ds41::Projection::Gate,ds41::Projection::Up,H,maxr,s));
      for(size_t a=0;a<n;){size_t b=a;int mr=0;while(b<n&&items[i+b].bits==items[i+a].bits){mr=std::max(mr,items[i+b].b->rows);++b;}
        CK(ds41::exl3_prefill_rotated_group(table,g+a,int(b-a),ds41::Projection::Gate,I,H,mr,items[i+a].bits,s));
        CK(ds41::exl3_prefill_rotated_group(table,u+a,int(b-a),ds41::Projection::Up,I,H,mr,items[i+a].bits,s));a=b;}
      CK(ds41::swiglu(W.gate.as<Bf16>(),W.up.as<Bf16>(),W.bw.as<float>(),W.act.as<Bf16>(),rows,I,10,s));
      CK(ds41::exl3_prefill_rotate_group(table,d,nullptr,int(n),ds41::Projection::Down,ds41::Projection::Down,I,maxr,s));
      for(size_t a=0;a<n;){size_t b=a;int mr=0;while(b<n&&items[i+b].bits==items[i+a].bits){mr=std::max(mr,items[i+b].b->rows);++b;}
        CK(ds41::exl3_prefill_rotated_group(table,d+a,int(b-a),ds41::Projection::Down,H,I,mr,items[i+a].bits,s));a=b;}
      ds41e::prefill::scatter_routes(W.down.p,dev_sl+sl,pf.routed.as<char>(base*6*H*2),rows,s);
      CK(cudaGetLastError());
      i=e;
    }
    CK(cudaEventRecord(timer.end,s));
    dispatched+=todo.size();
  };
  // s0ma change 89: VRAM tables for the staged experts' entries, before the sweep (one() may run on the stager's thread)
  if(prefill_device_desc()&&pf.lane_table.size()<variants)pf.lane_table.resize(variants);
  if(prefill_device_desc())for(size_t v=0;v<variants;++v)if(!pf.lane_table[v].bytes)
    try{pf.lane_table[v].resize(ds41::expert_count*sizeof(ds41::DeviceExpert));}
    catch(const std::exception&){(void)cudaGetLastError();}   // that variant keeps reading the mapped table
  one=[&](ds41::ExpertId id,const ds41::DeviceExpert* mapped_table,cudaStream_t s){
    // s0ma change 89: a staged expert's table entry is host-mapped (the stager's, or the change-77 arena's), and every
    // warp of every block of its kernels reads it: ~3,600 uncached PCIe reads an expert, ~20 million an 8K prompt,
    // sharing the link with the copies (change 86's finding, the descriptors' smaller cousin). Its entry goes into this
    // variant's VRAM table first, on the stream that runs its kernels; the same 64 bytes: exact.
    const ds41::DeviceExpert* table=mapped_table;
    if(mapped_table!=experts->table()&&prefill_device_desc()) {
      const size_t v=size_t(prefill_variant(s));
      if(v<pf.lane_table.size()&&pf.lane_table[v].bytes) {
        auto *dev=pf.lane_table[v].as<ds41::DeviceExpert>();
        pdesc(dev+id,mapped_table+id,sizeof(ds41::DeviceExpert),s);
        table=dev;
      }
    }
    const size_t timer_at=timers_used++;auto &timer=*compute.at(timer_at);timer_lane.at(timer_at)=prefill_variant(s);
    ++dispatched;CK(cudaEventRecord(timer.start,s));
    // s0ma change 82: this expert through FP16 GEMMs on its weights decoded once (enough rows, scratch ready)
    const bool via_gemm=exl3&&pf.gemm_ready&&gemm_rows_of(id)>=prefill_exl3_gemm_min();
    std::unique_lock<std::mutex> blas_lock;
    PrefillData::Gemm *G=nullptr;cublasHandle_t gh=nullptr;
    if(via_gemm) {
      const int v=prefill_variant(s);
      blas_lock=std::unique_lock<std::mutex>(*prefill_blas_m.at(size_t(v)));
      G=&pf.gemm.at(size_t(v));gh=prefill_blas.at(size_t(v));
      cudaStream_t hs=nullptr;
      if(cublasGetStream(gh,&hs)!=CUBLAS_STATUS_SUCCESS||hs!=s)throw std::logic_error("prefill GEMM handle on another stream");
      const int bits=width_of.empty()?expert_bits(int(id/384)):width_of[id];
      // the entry into device memory first: a staged expert's table is host-mapped, and these kernels read the
      // entry in every block (prefill_rot reads it once a block too, but has far fewer blocks)
      CK(cudaMemcpyAsync(G->entry.p,table+id,sizeof(ds41::DeviceExpert),cudaMemcpyDefault,s));
      const auto *one_entry=G->entry.as<ds41::DeviceExpert>();
      CK(ds41::exl3_expert_reconstruct(one_entry,ds41::ExpertId(0),ds41::Projection::Gate,I,H,bits,G->wg.as<half>(),s));
      CK(ds41::exl3_expert_reconstruct(one_entry,ds41::ExpertId(0),ds41::Projection::Up,I,H,bits,G->wu.as<half>(),s));
      CK(ds41::exl3_expert_reconstruct(one_entry,ds41::ExpertId(0),ds41::Projection::Down,H,I,bits,G->wd.as<half>(),s));
      ++pf.gemm_experts;
    }
    auto gemm=[&](const Device &W,const void *A,int n,int k,int rows){   // C [rows, n] = A [rows, k] W [k, n]
      const float one_=1,zero_=0;
      if(cublasGemmEx(gh,CUBLAS_OP_N,CUBLAS_OP_N,n,rows,k,&one_,W.p,CUDA_R_16F,n,A,CUDA_R_16F,k,&zero_,G->c.p,
                      CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx (prefill expert)");
    };
    for(const auto &b:batches.at(id)) {
      if(exl3) {
        const int v=prefill_variant(s);   // s0ma change 27: 0 = the main stream's scratch, else side lane v-1's
        PrefillLane *L=v?prefill_side[size_t(v-1)].get():nullptr;
        void *bxh=L?L->bxh.p:pf.bxh.p,*down=L?L->down.p:pf.down.p;
        float *bw=L?L->bw.as<float>():pf.bw.as<float>();
        Bf16 *gate=L?L->gate.as<Bf16>():pf.gate.as<Bf16>(),*up=L?L->up.as<Bf16>():pf.up.as<Bf16>(),
            *act=L?L->act.as<Bf16>():pf.act.as<Bf16>();
        const ds41::GemmJob *J=gpu_jobs+size_t(v)*chunks*3+b.job;
        ds41e::prefill::gather_routes_bf16(pf.ffn_input.as<char>(base*H*2),pf.weights.as<float>(base*6*4),
            gpu_slots+b.offset,bxh,bw,b.rows,s);
        const int bits=width_of.empty()?expert_bits(int(id/384)):width_of[id];   // s0ma changes 12, 15: this expert's width
        if(via_gemm) {   // s0ma change 82
          half *rg=L?L->rg.as<half>():pf.rg.as<half>(),*ru=L?L->ru.as<half>():pf.ru.as<half>(),
              *rd=L?L->rd.as<half>():pf.rd.as<half>();
          CK(ds41::exl3_prefill_rotate(table,J,ds41::Projection::Gate,ds41::Projection::Up,H,b.rows,rg,ru,s));
          gemm(G->wg,rg,I,H,b.rows);
          const auto *one_entry=G->entry.as<ds41::DeviceExpert>();
          CK(ds41::exl3_expert_rotate_out(one_entry,ds41::ExpertId(0),ds41::Projection::Gate,G->c.as<float>(),b.rows,I,H,gate,I,s));
          gemm(G->wu,ru,I,H,b.rows);
          CK(ds41::exl3_expert_rotate_out(one_entry,ds41::ExpertId(0),ds41::Projection::Up,G->c.as<float>(),b.rows,I,H,up,I,s));
          CK(ds41::swiglu(gate,up,bw,act,b.rows,I,10,s));
          CK(ds41::exl3_prefill_rotate(table,J+2,ds41::Projection::Down,ds41::Projection::Down,I,b.rows,rd,nullptr,s));
          gemm(G->wd,rd,H,I,b.rows);
          CK(ds41::exl3_expert_rotate_out(one_entry,ds41::ExpertId(0),ds41::Projection::Down,G->c.as<float>(),b.rows,H,I,
                                          static_cast<Bf16*>(down),H,s));
          pf.gemm_rows+=size_t(b.rows);
        } else if(prefill_exl3_rotated()) {   // s0ma change 68: each projection's rows rotated once, the same bits out
          half *rg=L?L->rg.as<half>():pf.rg.as<half>(),*ru=L?L->ru.as<half>():pf.ru.as<half>(),
              *rd=L?L->rd.as<half>():pf.rd.as<half>();
          CK(ds41::exl3_prefill_rotate(table,J,ds41::Projection::Gate,ds41::Projection::Up,H,b.rows,rg,ru,s));
          CK(ds41::exl3_prefill_rotated(table,J,ds41::Projection::Gate,I,H,b.rows,bits,rg,s));
          CK(ds41::exl3_prefill_rotated(table,J+1,ds41::Projection::Up,I,H,b.rows,bits,ru,s));
          CK(ds41::swiglu(gate,up,bw,act,b.rows,I,10,s));
          CK(ds41::exl3_prefill_rotate(table,J+2,ds41::Projection::Down,ds41::Projection::Down,I,b.rows,rd,nullptr,s));
          CK(ds41::exl3_prefill_rotated(table,J+2,ds41::Projection::Down,H,I,b.rows,bits,rd,s));
        } else {
        CK(ds41::grouped_exl3_prefill_bits(table,J,1,ds41::Projection::Gate,I,H,b.rows,bits,s));
        CK(ds41::grouped_exl3_prefill_bits(table,J+1,1,ds41::Projection::Up,I,H,b.rows,bits,s));
        CK(ds41::swiglu(gate,up,bw,act,b.rows,I,10,s));
        CK(ds41::grouped_exl3_prefill_bits(table,J+2,1,ds41::Projection::Down,H,I,b.rows,bits,s));
        }
        ds41e::prefill::scatter_routes(down,gpu_slots+b.offset,pf.routed.as<char>(base*6*H*2),b.rows,s);
        CK(cudaGetLastError());
        continue;
      }
      ds41e::prefill::gather_routes(pf.quant.as<uint8_t>(),pf.scales.as<uint8_t>(),
          pf.weights.as<float>(base*6*4),gpu_slots+b.offset,pf.bx.as<uint8_t>(),
          pf.bs.as<uint8_t>(),pf.bw.as<float>(),b.rows,s);
      CK(ds41::grouped_mxfp4_prefill(table,gpu_jobs+b.job,1,ds41::Projection::Gate,I,H,b.rows,s));
      CK(ds41::grouped_mxfp4_prefill(table,gpu_jobs+b.job+1,1,ds41::Projection::Up,I,H,b.rows,s));
      CK(ds41::swiglu(pf.gate.as<Bf16>(),pf.up.as<Bf16>(),pf.bw.as<float>(),pf.act.as<Bf16>(),b.rows,I,10,s));
      CK(ds41::quantize_mxfp8(pf.act.as<Bf16>(),pf.eq.as<uint8_t>(),pf.es.as<uint8_t>(),b.rows,I,s));
      CK(ds41::grouped_mxfp4_prefill(table,gpu_jobs+b.job+2,1,ds41::Projection::Down,H,I,b.rows,s));
      ds41e::prefill::scatter_routes(pf.down.p,gpu_slots+b.offset,pf.routed.as<char>(base*6*H*2),b.rows,s);
      CK(cudaGetLastError());
    }
    CK(cudaEventRecord(timer.end,s));
  };
  prefetch_layer=l;   // s0ma change 77
  prefill_sweep_origin=sweep.start;   // s0ma change 84
  try{pstream(needed,[&](auto id,const ds41::DeviceExpert* table,auto s){one(id,table,s);},group);}
  catch(...){prefetch_layer=-1;prefill_sweep_origin=nullptr;throw;}
  prefetch_layer=-1;prefill_sweep_origin=nullptr;
  if(dispatched!=needed.size())throw std::runtime_error("prefill expert dispatch coverage");
  double schedule=clock_ms()-host_start;
  CK(cudaEventRecord(sweep.end,stream));CK(cudaStreamSynchronize(stream));
  prefetch_release();   // s0ma change 77: every consumer of the arena has finished (the lanes joined the stream)
  float sweep_ms;CK(cudaEventElapsedTime(&sweep_ms,sweep.start,sweep.end));
  double compute_ms=0;
  // s0ma change 84: per lane (0 the main stream), its expert kernels' time and when its last expert kernel ended, ms
  // after the sweep's start (from the expert timers, all on their lanes and complete: the lanes joined the stream)
  std::vector<double> lane_kernel(variants,0.),lane_last(variants,-1.);
  for(size_t i=0,n=timers_used.load();i<n;++i){float ms;CK(cudaEventElapsedTime(&ms,compute[i]->start,compute[i]->end));compute_ms+=ms;
    const size_t v=size_t(timer_lane[i]);
    if(v<variants){lane_kernel[v]+=ms;float t;CK(cudaEventElapsedTime(&t,sweep.start,compute[i]->end));lane_last[v]=std::max(lane_last[v],double(t));}}
  if(prefill_stage)prefill_stage->timeline(prefill_transfers);   // s0ma change 84: the copies, now that all is complete
  Json lane_end=Json::array(),lane_staged=Json::array(),lane_kernel_ms=Json::array();
  double last_end=0,first_end=1e300;bool any_end=false;
  for(size_t v=0;v<variants;++v){const double t=std::max(0.,lane_last[v]);lane_end.push_back(t);
    if(lane_last[v]>=0){last_end=std::max(last_end,t);first_end=std::min(first_end,t);any_end=true;}}
  if(!any_end)first_end=0;
  for(auto n:prefill_lane_staged)lane_staged.push_back(n);
  for(auto t:lane_kernel)lane_kernel_ms.push_back(t);
  auto resident_after=experts->stats();auto stage_after=prefill_transfers;
  if(resident_after.evictions!=resident_before.evictions&&!vram_admit_budget)   // s0ma change 60: admissions evict
    throw std::runtime_error("decode cache changed during prefill; competing forecasts must be quiesced");
  return {{"tokens_per_expert_histogram",distribution},{"unique_experts",needed.size()},{"expert_chunks",chunks},{"route_boundary_ms",boundary},
      {"expert_schedule_host_ms",schedule},{"expert_sweep_stream_ms",sweep_ms},
      {"expert_compute_stream_ms",compute_ms},{"expert_stream_gap_ms",std::max(0.,double(sweep_ms)-compute_ms)},
      {"h2d_bytes",stage_after.h2d_bytes},
      {"nvme_bytes",stage_after.nvme_bytes},
      {"decode_cache_evictions",resident_after.evictions-resident_before.evictions},
      {"decode_vram_hits",resident_after.hits-resident_before.hits},
      {"borrowed_vram_experts",prefill_vram_hits},{"host_tier_hits",stage_after.host_hits},{"nvme_expert_misses",stage_after.nvme_misses},
      {"h2d_stream_ms",stage_after.h2d_stream_ms},
      {"ram_service_including_backpressure_ms",stage_after.ram_service_host_ms},
      {"nvme_service_including_backpressure_ms",stage_after.nvme_service_host_ms},
      {"staging_mode","isolated-vram-ram-nvme-v0.3"},
      {"gemm_experts_cumulative",pf.gemm_experts.load()},{"gemm_rows_cumulative",pf.gemm_rows.load()},   // s0ma change 82
      // s0ma change 84: the copies on the GPU's clock (see TransferStats) and the lanes
      {"ram_copies",stage_after.ram_copies},{"ram_copy_bytes",stage_after.ram_copy_bytes},
      {"ram_copy_union_ms",stage_after.ram_copy_union_ms},{"ram_copy_sum_ms",stage_after.ram_copy_sum_ms},
      {"ram_copy_first_ms",stage_after.ram_copy_first_ms},{"ram_copy_last_ms",stage_after.ram_copy_last_ms},
      {"after_last_ram_copy_ms",stage_after.ram_copies?std::max(0.,double(sweep_ms)-stage_after.ram_copy_last_ms):0.},
      {"ram_slot_wait_ms",stage_after.ram_slot_wait_ms},{"ram_issue_gap_ms",stage_after.ram_issue_gap_ms},
      {"ring_wait_host_ms",stage_after.ring_wait_host_ms},{"ring_blocked",stage_after.ring_blocked},
      {"ram_ring_slots",stage_after.ram_slots},{"copy_streams",stage_after.copy_streams},
      {"ring_device_wait",stage_after.device_wait},
      {"disk_copies",stage_after.disk_copies},{"disk_copy_union_ms",stage_after.disk_copy_union_ms},
      {"copy_union_ms",stage_after.copy_union_ms},
      {"staged_lanes",staged_lanes_used},{"lane_end_ms",lane_end},{"lane_staged",lane_staged},{"lane_kernel_ms",lane_kernel_ms},
      {"lanes_last_end_ms",last_end},{"lanes_first_end_ms",first_end}};
}
#include "batch.inc"
#include "chunk.inc"

// s0ma change 67: the Engram projection for up to 8 prompt rows in one FP8 call. pengram ran it one row at a time:
// a full pass over the 157 MB wkv matrix for each row of the prompt, at layers 1 and 14 (~16,000 passes for an 8K
// prompt). The n-gram rows do not depend on the hidden state, so all of a group's rows are copied, dequantized and
// projected together; each row's gate then runs as before. Exact: the FP8 kernel keeps every k32 dot of each row and
// adds them in ascending K whatever the row count (its other rows are other MMA columns), and the input quantization
// is per 32 elements of a row. DS41_PREFILL_ENGRAM_BATCH=0 keeps the row-at-a-time path.
static bool prefill_engram_batch() {
  static const bool on = [] { const char *v = std::getenv("DS41_PREFILL_ENGRAM_BATCH"); return !v || std::string(v) != "0"; }();
  return on;
}
void pengram_rows(int layer,const char* raw,const int* rows,int count,Device& chunk_h) {
  auto p="layers."+std::to_string(layer)+".engram";
  auto &weight=w(p+".wkv.weight");
  const bool fp8=weight.dtype=="F8_E4M3"||weight.dtype=="F8_E4M3FN";
  if(!fp8||dense_of(p+".wkv")||weight.shape!=std::vector<uint64_t>({uint64_t((HC+1)*H),6144ull})) {
    for(int i=0;i<count;++i) {   // not the native FP8 projection: the row-at-a-time path
      char* row_h=chunk_h.as<char>(size_t(rows[i])*HC*H*2);
      pcopy(h.p,row_h,HC*H*2);pengram(layer,raw+size_t(rows[i])*24*264);pcopy(row_h,h.p,HC*H*2);
    }
    return;
  }
  constexpr size_t R=24*264;
  for(int i=0;i<count;++i)
    CK(cudaMemcpyAsync(pe_raw.as<char>(size_t(i)*R),raw+size_t(rows[i])*R,R,cudaMemcpyHostToDevice,stream));
  async_engram->dequant(pe_raw.p,pe_bf.p,24*count,stream);
  CK(ds41_dense_quant(pe_bf.p,pe_aq.p,pe_as.p,count,6144,stream));
  CK(ds41_dense_fp8(pe_aq.p,pe_as.p,weight.data.p,w(p+".wkv.scale").data.p,pe_kv.p,count,(HC+1)*H,6144,2,stream));
  for(int i=0;i<count;++i) {
    char* row_h=chunk_h.as<char>(size_t(rows[i])*HC*H*2);
    const char* kv=pe_kv.as<char>(size_t(i)*(HC+1)*H*2);
    pcopy(h.p,row_h,HC*H*2);
    CK(ds41_dense_engram_gate(h.p,kv,kv+size_t(HC)*H*2,w(p+".q_weight").data.p,w(p+".k_weight").data.p,nullptr,
                              next_h.p,1,H,1e-20f,stream));
    std::swap(h,next_h);
    pcopy(row_h,h.p,HC*H*2);
  }
}
void pengram(int layer,const void* raw) {
  auto p="layers."+std::to_string(layer)+".engram";
  CK(cudaMemcpyAsync(rows_interleaved.p,raw,24*264,cudaMemcpyHostToDevice,stream));
  async_engram->dequant(rows_interleaved.p,rows_bf.p,24,stream);
  linear(p+".wkv",rows_bf.p,engram_kv.p,(HC+1)*H,6144);
  CK(ds41_dense_engram_gate(h.p,engram_kv.p,engram_kv.as<char>(HC*H*2),
      w(p+".q_weight").data.p,w(p+".k_weight").data.p,nullptr,next_h.p,1,H,1e-20f,stream));
  std::swap(h,next_h);
}
public:
std::vector<int> cached_prefix() const override {return prefix.ids();}
std::map<std::string,std::vector<uint8_t>> live_state() override {
  if(pending||poisoned)throw std::logic_error("state inspection requires healthy committed boundary");
  std::map<std::string,std::vector<uint8_t>> out;
  auto copy=[&](const std::string& key,const void* pointer,size_t bytes){
    auto &v=out[key];v.resize(bytes);if(bytes)CK(cudaMemcpyAsync(v.data(),pointer,bytes,cudaMemcpyDeviceToHost,stream));
  };
  int64_t first=std::max<int64_t>(0,committed-128);
  for(int l=0;l<40;++l)for(int64_t p=first;p<committed;++p)
    copy("swa/"+std::to_string(l)+"/"+std::to_string(p),windows[l].as<char>(size_t(p%RING)*528),528);
  for(auto &item:globals) {
    auto &g=*item.second;auto key="global/"+std::to_string(item.first);
    size_t rows=size_t(committed/g.ratio);copy(key+"/kv",g.kv.p,rows*288);copy(key+"/index",g.key.p,rows*68);
    if(g.ratio==2&&committed%2) {
      size_t offset=size_t((committed-1)%8)*512*4;
      copy(key+"/raw-kv-tail",g.raw_kv.as<char>(offset),512*4);
      copy(key+"/raw-score-tail",g.raw_score.as<char>(offset),512*4);
    }
  }
  CK(cudaStreamSynchronize(stream));
  const auto &history=hash.history();auto &bytes=out["engram/token-history"];
  bytes.resize(size_t(committed)*sizeof(int64_t));
  if(!bytes.empty())std::memcpy(bytes.data(),history.data(),bytes.size());
  auto &ids=prefix.ids();auto &tokens=out["session/committed-input-ids"];
  tokens.resize(ids.size()*sizeof(int));if(!tokens.empty())std::memcpy(tokens.data(),ids.data(),tokens.size());
  return out;
}
// s0ma change 82: the GEMM path's scratch, every variant: three projections in FP16, 256 rows of FP32, a workspace
size_t prefill_gemm_bytes() const {
  if(!exl3||!prefill_exl3_gemm())return 0;
  return size_t(prefill_lanes())*(size_t(3)*H*I*2+size_t(256)*H*4+(size_t(16)<<20)+(size_t(4)<<20));
}
bool supports_prefill(PrefillMode mode,size_t rows) const override {
  if(!config.prefill||config.prefetch_routes||!prefetch_tickets.empty()||
      poisoned||pending||!rows||rows>size_t(config.max_context-committed)||
      (mode!=PrefillMode::Cold&&mode!=PrefillMode::Append)||
      (mode==PrefillMode::Cold?committed!=0:committed==0))return false;
  size_t need=PrefillData::required_bytes(rows,committed,1);
  if(need>config.prefill_budget_bytes)return false;
  need+=prefill_gemm_bytes();   // s0ma change 82: its scratch beside the segment's (free VRAM, not the budget)
  // Opt-in (DS41_PREFILL_CHUNK_FIT=1): a segment must also fit the chunked prefill scratch, so long
  // prompts split into fast chunked segments instead of one oversized M1-path prefill. 256 MiB slack
  // plus a 128 MiB margin keep the later allocations (taken in the same order) from racing.
  if(prefill_chunk_fit&&config.prefill_chunk_rows) {
    const int64_t end=committed+int64_t(rows);
    const int width=int(std::min<int64_t>(2048,(end+7)/8)*8);
    for(auto b:PrefillChunk::sizes(int(config.prefill_chunk_rows),end,width,prefill_fp8_elements()))need+=b;
    need+=size_t(256)<<20;
  }
  size_t free,total;CK(cudaMemGetInfo(&free,&total));
  return need<=free&&free-need>=(prefill_chunk_fit?(size_t(128)<<20):(size_t(64)<<20));
}
// s0ma change 51 (V6): the prompt borrows back the lendable expert chunks its first segment's scratch needs (as
// supports_prefill counts it, with the chunk scratch and 256 MiB for the buffers a first prompt keeps), or all of
// them when that cannot fit; the first verify after the prompt takes them back for experts, keeping lend_margin free.
size_t prefill_lend_need(size_t rows) const {
  size_t need=PrefillData::required_bytes(rows,committed,1)+prefill_gemm_bytes();
  if(config.prefill_chunk_rows) {
    const int64_t end=committed+int64_t(rows);
    const int width=int(std::min<int64_t>(2048,(end+7)/8)*8);
    for(auto b:PrefillChunk::sizes(int(config.prefill_chunk_rows),end,width,prefill_fp8_elements()))need+=b;
  }
  return need+(size_t(256)<<20)+(size_t(256)<<20)+(size_t(128)<<20);
}
void prefill_lend(size_t need) {
  if(!lend_chunks_total||lent_chunks>=lend_chunks_total)return;
  size_t free,total;CK(cudaMemGetInfo(&free,&total));
  if(free>=need)return;
  const size_t want=(need-free+lend_chunk_bytes-1)/lend_chunk_bytes;
  lent_chunks+=experts->lend(std::min(want,lend_chunks_total-lent_chunks));
}
// s0ma change 54: the RAM tier's contents when a prompt starts, one "tiers:" line a prompt. An expert in both VRAM and
// RAM is a duplicate: its RAM slot holds nothing the cache does not already have.
void tier_census() {
  if(poisoned||!experts)return;
  const auto bits=experts->residency_snapshot();
  const auto st=experts->stats();
  size_t vram=0,ram=0,both=0;double ram_b=0,both_b=0;
  for(size_t id=0;id<bits.size()&&id<15360;++id){
    const double b=width_of.empty()?double(expert_slot_bytes):double(exl3_rec_of(width_of[id]));
    if(bits[id]&1)++vram;
    if(bits[id]&2){++ram;ram_b+=b;}
    if((bits[id]&3)==3){++both;both_b+=b;}
  }
  std::cerr<<"tiers: VRAM "<<vram<<" experts, RAM "<<ram<<" ("<<ram_b/double(1ull<<30)<<" GiB), in both "<<both<<" ("
           <<both_b/double(1ull<<30)<<" GiB); demotions "<<st.demotions<<", victims already in RAM "<<st.demote_resident
           <<", RAM evictions "<<st.ram.evictions<<"; direct evictions "<<st.direct_evictions<<" ("<<st.rescued
           <<" copied back to RAM); only on the drive "<<(15360-(vram+ram-both))<<"\n";
  // s0ma change 65: how experts came to be only on the drive, and the store worker's own time a load
  std::cerr<<"store: lost by direct eviction "<<st.lost_direct<<", by demotion "<<st.lost_topup<<", by RAM eviction "
           <<st.lost_ram<<", by lend "<<st.lost_lend<<"; worker ms a load: acquire "
           <<(st.loads?st.acquire_ns/1e6/double(st.loads):0.0)<<", top-ups "<<(st.loads?st.topup_ns/1e6/double(st.loads):0.0)
           <<", RAM admissions "<<(st.loads?st.admit_ns/1e6/double(st.loads):0.0)<<" ("<<st.loads<<" loads)\n";
}
// s0ma change 56: decode's NVMe misses after a prompt against what that prompt read from the SSD and what its last
// 256 rows routed. A miss counted under "prompt read" is one the prompt's sweep had already brought through host
// memory; under "last 256 rows" one its tail routed. An upper bound on what admitting them into RAM could save.
void overlap_drain(bool step) {
  if(!overlap_on||!experts||prompt_bits.empty())return;
  const bool first=steps_after_prompt<32;
  for(auto id:experts->take_nvme_misses()) {
    if(id>=15360)continue;
    const uint8_t b=prompt_bits[id];
    for(auto *o:{&req_all,&after_all}){++o->n;o->disk+=b&1;o->tail+=(b>>1)&1;o->both+=(b&3)==3;}
    if(first)for(auto *o:{&req_first,&after_first}){++o->n;o->disk+=b&1;o->tail+=(b>>1)&1;o->both+=(b&3)==3;}
  }
  for(auto id:experts->take_ram_hits()) {   // s0ma change 59
    if(id>=15360)continue;
    const uint8_t b=prompt_bits[id];
    for(auto *o:{&ram_req_all,&ram_all}){++o->n;o->disk+=(b>>2)&1;o->tail+=(b>>1)&1;}
    if(first)for(auto *o:{&ram_req_first,&ram_first}){++o->n;o->disk+=(b>>2)&1;o->tail+=(b>>1)&1;}
  }
  if(step)++steps_after_prompt;
}
void overlap_report() {
  if(!overlap_on||prompt_bits.empty())return;
  overlap_drain(false);
  size_t disk=0,tail=0,both=0;double disk_b=0,both_b=0;
  for(size_t id=0;id<prompt_bits.size();++id){
    const double b=width_of.empty()?double(expert_slot_bytes):double(exl3_rec_of(width_of[id]));
    if(prompt_bits[id]&1){++disk;disk_b+=b;}
    if(prompt_bits[id]&2)++tail;
    if((prompt_bits[id]&3)==3){++both;both_b+=b;}
  }
  ++overlap_requests;
  auto line=[](const char* name,const Overlap& a,const Overlap& f){
    std::cerr<<name<<": decode NVMe misses "<<a.n<<" (first 32 steps "<<f.n<<"); the prompt had read from the SSD "<<a.disk
             <<" ("<<f.disk<<"); its last 256 rows routed "<<a.tail<<" ("<<f.tail<<"); both "<<a.both<<" ("<<f.both<<")";
  };
  line("after prompt",req_all,req_first);
  std::cerr<<std::fixed<<std::setprecision(1)<<". The prompt: "<<disk<<" experts from the SSD ("<<disk_b/double(1ull<<30)<<" GiB), its last 256 rows "
           <<tail<<" routed, "<<both<<" of them from the SSD ("<<both_b/double(1ull<<30)<<" GiB)\n"<<std::defaultfloat<<std::setprecision(6);
  line("after prompts, all",after_all,after_first);
  std::cerr<<" ("<<overlap_requests<<" requests)\n";
  auto ram_line=[](const char* name,const Overlap& a,const Overlap& f){
    std::cerr<<name<<": decode VRAM misses served from RAM "<<a.n<<" (first 32 steps "<<f.n<<"); the prompt routed "<<a.disk
             <<" ("<<f.disk<<"); its last 256 rows routed "<<a.tail<<" ("<<f.tail<<")";
  };
  ram_line("after prompt, RAM",ram_req_all,ram_req_first);
  std::cerr<<std::fixed<<std::setprecision(1)<<". When the prompt ended, "<<tail_not_vram<<" experts its last 256 rows routed were not in VRAM ("
           <<tail_not_vram_b/double(1ull<<30)<<" GiB)\n"<<std::defaultfloat<<std::setprecision(6);
  ram_line("after prompts, all, RAM",ram_all,ram_first);
  std::cerr<<"\n";
  ram_req_all=ram_req_first=Overlap{};
  if(vram_admit_budget){const auto v=experts->vram_admit_stats();
    std::cerr<<std::fixed<<std::setprecision(1)<<"prompt VRAM admission: "<<v.experts<<" experts ("<<v.bytes/double(1ull<<30)
             <<" GiB) copied into VRAM slots from the prompts' staging, "<<v.victims<<" residents evicted ("<<v.victims_dropped
             <<" of them not in RAM; "<<vram_admit_budget/double(1ull<<30)<<" GiB a prompt at most)\n"<<std::defaultfloat<<std::setprecision(6);}
  if(admit_budget)std::cerr<<std::fixed<<std::setprecision(1)<<"prompt admission: "<<admit_experts<<" experts ("
                           <<admit_bytes/double(1ull<<30)<<" GiB) copied into RAM from the prompts' drive reads, "
                           <<admit_copy_ms<<" ms on the reader thread to queue their copies ("<<admit_budget/double(1ull<<30)
                           <<" GiB a prompt at most)\n"<<std::defaultfloat<<std::setprecision(6);
  req_all=req_first=Overlap{};
}
void begin_prompt(int64_t rows,bool append) override {
  (void)append;
  tier_census();
  if(overlap_on&&experts) {   // s0ma change 56: close the previous request's count, open this prompt's
    overlap_report();
    if(prompt_bits.empty())experts->log_nvme_misses(true);
    prompt_bits.assign(15360,0);steps_after_prompt=0;
    if(admit_budget||vram_admit_budget)prompt_tail_count.assign(15360,0);
    admit_used=0;vram_admit_used=0;
  }
  if(!lend_chunks_total||poisoned)return;
  lend_prompt=true;++lend_prompts;lend_retry=0;
  const double t0=clock_ms();
  const int64_t fresh=std::max<int64_t>(1,std::min<int64_t>(rows-committed,config.max_context-committed));
  // s0ma change 53: DS41_PREFILL_LEND_ALL=1 lends every chunk to every prompt (the prompt sees today's free VRAM)
  static const bool lend_all=[]{const char* v=std::getenv("DS41_PREFILL_LEND_ALL");return v&&std::string(v)=="1";}();
  prefill_lend(lend_all?~size_t(0)>>1:prefill_lend_need(size_t(fresh)));
  const double t=clock_ms()-t0;lend_begin_ms+=t;lend_begin_max_ms=std::max(lend_begin_max_ms,t);
}
void end_prompt(bool) override {
  lend_prompt=false;lend_retry=0;
  if(overlap_on&&experts&&!prompt_bits.empty()) {   // s0ma change 59: what a VRAM admission of the tail would bring in
    const auto bits=experts->residency_snapshot();
    tail_not_vram=0;tail_not_vram_b=0;
    for(size_t id=0;id<15360&&id<bits.size();++id)if((prompt_bits[id]&2)&&!(bits[id]&1)){
      ++tail_not_vram;tail_not_vram_b+=width_of.empty()?double(expert_slot_bytes):double(exl3_rec_of(width_of[id]));}
  }
}
void lend_reclaim() {
  const double t0=clock_ms();
  const size_t got=experts->reclaim(lend_margin);
  lend_reclaim_ms+=clock_ms()-t0;
  lent_chunks-=std::min(got,lent_chunks);
  if(!lent_chunks) {
    const auto x=experts->lend_info();
    std::cerr<<"prefill lend: "<<lend_prompts<<" prompts, chunks lent "<<x.lends<<", reclaimed "<<x.reclaims<<" of "<<x.chunks
             <<" x "<<x.chunk_bytes/double(1<<20)<<" MiB; experts moved "<<x.moved<<", dropped "<<x.dropped
             <<"; busy "<<x.busy<<", failed "<<x.failed
             <<"; begin_prompt "<<lend_begin_ms/double(lend_prompts)<<" ms a prompt (max "<<lend_begin_max_ms<<"; settle "
             <<x.settle_ms<<", sync "<<x.sync_ms<<", moves "<<x.move_ms<<", cudaFree "<<x.free_ms<<" ms in all), reclaim "
             <<lend_reclaim_ms<<" ms in all (cudaMalloc "<<x.malloc_ms<<")\n";
  }
}
PrefillResult prefill(const std::vector<int>& input,const PrefillOptions& options) override {
  if(!config.prefill)throw std::logic_error("native prefill requires Config::prefill opt-in");
  if(poisoned)throw std::runtime_error("native forward/store poisoned; restart ds41d");
  if((options.mode!=PrefillMode::Cold&&options.mode!=PrefillMode::Append)||pending||input.empty()||input.size()>size_t(config.max_context-committed)||
      options.logits_tail<1||options.logits_tail>input.size()||options.chunk_rows<1||options.chunk_rows>4096||options.dense_batch_rows<1||options.dense_batch_rows>8||
      (options.mode==PrefillMode::Cold&&committed!=0)||
      (options.mode==PrefillMode::Append&&committed==0))
    throw std::invalid_argument("prefill mode/state/context/chunk/tail");
  for(int id:input)if(id<0||id>=V)throw std::invalid_argument("prefill token id");
  // s0ma change 90: the profiler range around the N-th prefill (closed however this call leaves)
  struct ProfileRange {
    bool open=false;size_t rows=0;long n=0;
    ~ProfileRange(){if(!open)return;(void)cudaDeviceSynchronize();(void)cudaProfilerStop();
      std::cerr<<"profiler range: closed after prefill "<<n<<" (rows "<<rows<<")\n";}
  } profile_range;
  if(profile_prefill>0&&++prefill_calls==profile_prefill) {
    CK(cudaDeviceSynchronize());CK(cudaProfilerStart());
    profile_range.open=true;profile_range.rows=input.size();profile_range.n=prefill_calls;profile_prefill=-1;
    std::cerr<<"profiler range: open at prefill "<<prefill_calls<<" (rows "<<input.size()<<", cached "<<committed<<")\n";
  }
  double begin_ms=clock_ms();
  // s0ma change 53: where a prompt's time goes outside its layers (one "prefill phases:" line a prompt)
  std::vector<std::pair<const char*,double>> phase_marks;
  auto mark=[&](const char* name){phase_marks.push_back({name,clock_ms()});};
  const void* initial_h=h.p;
  prefill_warm_counts.clear();
  std::vector<uint8_t> residency_before;
  try{residency_before=p_residency(*experts);}catch(...){poisoned=true;throw;}
  mark("residency");
  auto plan=ds41e::prefill::plan(committed,int64_t(input.size()),int64_t(options.logits_tail));
  // s0ma change 51: an image's encoder and rows need VRAM begin_prompt did not count: every lendable chunk
  if(options.images&&!options.images->empty())prefill_lend(~size_t(0)>>1);
  // Image rows of this segment. Each image the segment touches is encoded first, before the prefill arena
  // takes free VRAM; only its aligner rows (10 MB at most) stay allocated until the prefill returns.
  struct Rows {void* p=nullptr;~Rows(){if(p)cudaFree(p);}};
  std::vector<std::unique_ptr<Rows>> encoded;
  std::vector<const void*> image_embed(input.size(),nullptr);
  std::vector<uint8_t> image_rows(input.size(),0);
  double vision_ms=0;size_t images_encoded=0;
  if(options.images)for(const auto &img:*options.images) {
    const int64_t b=std::max<int64_t>(img.start,committed),e=std::min<int64_t>(img.end(),committed+int64_t(input.size()));
    if(b>=e)continue;
    ensure_vision();
    bool slots=false;
    for(int64_t pos=b;pos<e&&!slots;++pos)slots=img.kind(pos,nullptr)==1;
    const char* rows=nullptr;
    if(slots) {
      encoded.push_back(std::make_unique<Rows>());
      encoded.back()->p=vision->encode(img);rows=static_cast<const char*>(encoded.back()->p);
      vision_ms+=vision->last_ms();++images_encoded;
    }
    for(int64_t pos=b;pos<e;++pos) {
      const size_t r=size_t(pos-committed);int row=0;const int kind=img.kind(pos,&row);
      if(input[r]!=ImageSpan::token_id)throw std::invalid_argument("image span position without the image token id");
      image_rows[r]=1;
      image_embed[r]=kind==1?static_cast<const void*>(rows+size_t(row)*H*2)
          :image_delimiters.as<char>(size_t(kind==0?0:kind==2?1:2)*H*2);
    }
  }
  // Preflight allocations before any live state mutation. No eviction or cache
  // rebuild is used to make room for scratch. A failed allocation is retryable.
  mark("images");
  prefetch.reset();   // s0ma change 77 (a new prompt: a new arena, below)
  pref=std::make_unique<PrefillData>(input.size(),committed,options.mode==PrefillMode::Cold,
      options.logits_tail,config.prefill_budget_bytes);
  auto &pf=*pref;pf.dense_batch=int(options.dense_batch_rows);pf.warm=options.warm;pf.image=std::move(image_rows);
  pf.route_host=prefill_pinned(prefill_route_buf,input.size()*6*4);
  prefill_chunk.reset();
  if(config.prefill_chunk_rows) {
    // Optional scratch: allocated only beside the live decode cache, otherwise the M1 path runs.
    try {
      const int R=int(config.prefill_chunk_rows);const size_t fp8=prefill_fp8_elements();
      auto sizes=PrefillChunk::sizes(R,pf.end,pf.candidate_width,fp8);
      size_t need=0;for(auto b:sizes)need+=b;
      size_t free,total;CK(cudaMemGetInfo(&free,&total));
      if(need+(size_t(256)<<20)<=free)prefill_chunk=std::make_unique<PrefillChunk>(R,pf.end,pf.candidate_width,fp8);
    }catch(const std::exception&){prefill_chunk.reset();cudaGetLastError();}
  }
  if(prefill_chunk) {   // s0ma change 77: after the prompt's own scratch; a failure only turns it off
    try{prefetch_begin(input.size());}
    catch(const std::exception &e){(void)cudaGetLastError();prefetch.reset();std::cerr<<"prefill prefetch off: "<<e.what()<<"\n";}
  }
  mark("alloc");
  PrefillResult result;result.begin_position=committed;result.consumed_rows=input.size();
  try {
    in_step=true;graph_ready=false;
    for(size_t r=0;r<input.size();++r) {
      auto ids=hash.rows(input[r],committed+r,!pf.image[r]);
      for(int j=0;j<2;++j)pf.rowids[j].insert(pf.rowids[j].end(),ids[j].begin(),ids[j].end());
    }
    // Both whole-suffix tickets deduplicate globally. They start before layer0
    // and stay alive through their last H2D. Appends hash/submit only new IDs.
    if(async_engram)for(int j=0;j<2;++j)pf.engram[j]=async_engram->submit(j==0?1:14,pf.rowids[j]);
    mark("hash");
    // Start I/O before enqueueing prompt-sized embedding work as well.
    for(size_t r=0;r<input.size();++r) {
      expand_embedding(pf.image[r]?image_embed[r]:w("embed.weight").data.as<Bf16>(size_t(input[r])*H*2),h.p,pre.as<float>(),H,stream);
      pcopy(pf.hidden.as<char>(r*HC*H*2),h.p,HC*H*2);
      pcopy(pf.previous_pre.as<char>(r*4*4),pre.p,4*4);
    }
    CK(cudaStreamSynchronize(stream));
    for(const auto &work:plan) {
      int l=work.layer;auto p="layers."+std::to_string(l);pf.query=work.query;
      upload_cursor=1<<20;
      CK(cudaMemsetAsync(pf.window.p,0,pf.window.bytes,stream));
      if(!pf.cold) {
        int64_t old=std::max<int64_t>(0,pf.begin-128);
        upload(position_device,&old,8);
        // packed scratch is one row, so copy the ring's retained tail in two
        // device spans with absolute modulo addressing and no host roundtrip.
        for(int64_t pos=old;pos<pf.begin;++pos)
          pcopy(pf.window.as<char>(size_t(pos%int64_t(pf.rows+128))*528),windows[l].as<char>(size_t(pos%RING)*528),528);

      }
      Phase project;CK(cudaEventRecord(project.start,stream));
      auto projection=work.global.size()?work.global:work.window;
      for(int64_t first=projection.begin;first<projection.end;first+=int64_t(options.chunk_rows)) {
        int64_t end=std::min(projection.end,first+int64_t(options.chunk_rows));
        upload_cursor=1<<20;
        const char* raw=nullptr;
        if((l==1||l==14)&&async_engram) {
          int j=l==1?0:1;
          if(!pf.engram_raw[j])pf.engram_raw[j]=static_cast<const char*>(pf.engram[j]->wait(pf.rows*24));
          raw=pf.engram_raw[j]+size_t(first-pf.begin)*24*264;
        }
        if(prefill_chunk) {
          const int64_t R=prefill_chunk->R;
          for(int64_t pos=first;pos<end;pos+=R)
            cproject(work,pos,int(std::min<int64_t>(R,end-pos)),raw?raw+size_t(pos-first)*24*264:nullptr);
        } else if(pf.dense_batch>1) {
          if(!prefill_batch)prefill_batch=std::make_unique<PrefillBatch>();
          for(int64_t pos=first;pos<end;pos+=pf.dense_batch)
            pproject_batch(work,pos,int(std::min<int64_t>(pf.dense_batch,end-pos)),
                raw?raw+size_t(pos-first)*24*264:nullptr);
        } else for(int64_t pos=first;pos<end;++pos) {
          size_t r=size_t(pos-pf.begin);
          pcopy(h.p,pf.hidden.as<char>(r*HC*H*2),HC*H*2);
          pcopy(pre.p,pf.previous_pre.as<char>(r*4*4),4*4);
          if((l==1||l==14)&&!pf.image[r]) {
            if(raw)pengram(l,raw+size_t(pos-first)*24*264);
            else {auto &all=pf.rowids[l==1?0:1];engram(l,{all.begin()+r*24,all.begin()+(r+1)*24});}
          }
          if(work.capture.contains(pos))capture_mean(h.p,pf.capture.as<char>(size_t(pos-work.capture.begin)*3*H*2+(l-37)*H*2),H,stream);
          hc(p,"attn",att_pre);
          CK(ds41_dense_hc_pre(h.p,pre.as<float>(),x.p,1,H,stream));norm(p+".attn_norm",x.p,x.p,H);
          if(work.query.contains(pos)) {
            linear(p+".attn.wq_a",x.p,qa.p,1280,H);norm(p+".attn.q_norm",qa.p,qa.p,1280);
            linear(p+".attn.wq_b",qa.p,q.p,32768,1280);
          }
          linear(p+".attn.wkv",x.p,kv.p,512,H);norm(p+".attn.kv_norm",kv.p,kv.p,512);
          pattention(l,pos);
          if(!work.query.contains(pos))continue;
          CK(ds41_dense_hc_post(y.p,h.p,post.as<float>(),comb.as<float>(),next_h.p,1,H,stream));std::swap(h,next_h);
          pcopy(pf.hidden.as<char>(r*HC*H*2),h.p,HC*H*2);
          hc(p,"ffn",ffn_pre);
          CK(ds41_dense_hc_pre(h.p,att_pre.as<float>(),x.p,1,H,stream));norm(p+".ffn_norm",x.p,x.p,H);
          router_scores(l,pf.image[r]!=0);
          ds41e::prefill::select_routes(biased.as<float>(),unbiased.as<float>(),pf.ids.as<int>(r*6*4),pf.weights.as<float>(r*6*4),stream);
          pcopy(pf.ffn_input.as<char>(r*H*2),x.p,H*2);
          pcopy(pf.ffn_post.as<char>(r*4*4),post.p,4*4);pcopy(pf.ffn_comb.as<char>(r*16*4),comb.p,16*4);
          pcopy(pf.previous_pre.as<char>(r*4*4),ffn_pre.p,4*4);
        }
        CK(cudaStreamSynchronize(stream)); // metadata and Engram ticket lifetime
      }
      CK(cudaEventRecord(project.end,stream));CK(cudaStreamSynchronize(stream));
      float project_ms;CK(cudaEventElapsedTime(&project_ms,project.start,project.end));
      Json timings=pmoe(l,work.query);
      Phase epilogue;CK(cudaEventRecord(epilogue.start,stream));
      if(prefill_chunk) {
        const int64_t R=prefill_chunk->R;
        for(int64_t pos=work.query.begin;pos<work.query.end;pos+=R)
          cshared(l,pos,int(std::min<int64_t>(R,work.query.end-pos)));
      } else if(pf.dense_batch>1) {
        for(int64_t pos=work.query.begin;pos<work.query.end;pos+=pf.dense_batch)
          pshared_batch(l,pos,int(std::min<int64_t>(pf.dense_batch,work.query.end-pos)));
      } else for(int64_t pos=work.query.begin;pos<work.query.end;++pos) {
        size_t r=size_t(pos-pf.begin);
        pcopy(x.p,pf.ffn_input.as<char>(r*H*2),H*2);
        linear(p+".ffn.shared_experts.w1",x.p,sg.p,I,H);linear(p+".ffn.shared_experts.w3",x.p,su.p,I,H);
        CK(ds41::swiglu(sg.as<Bf16>(),su.as<Bf16>(),nullptr,sa.as<Bf16>(),1,I,10,stream));
        linear(p+".ffn.shared_experts.w2",sa.p,shared.p,H,I);
        CK(ds41::reduce_experts(pf.routed.as<Bf16>(r*6*H*2),pf.ids.as<int32_t>(r*6*4),shared.as<Bf16>(),y.as<Bf16>(),1,6,H,stream));
        CK(ds41_dense_hc_post(y.p,pf.hidden.as<char>(r*HC*H*2),pf.ffn_post.as<float>(r*4*4),pf.ffn_comb.as<float>(r*16*4),next_h.p,1,H,stream));
        pcopy(pf.hidden.as<char>(r*HC*H*2),next_h.p,HC*H*2);
      }
      int64_t tail=std::max<int64_t>(0,pf.end-128);
      for(int64_t pos=tail;pos<pf.end;++pos)
        pcopy(windows[l].as<char>(size_t(pos%RING)*528),pf.window.as<char>(size_t(pos%int64_t(pf.rows+128))*528),528);
      CK(cudaEventRecord(epilogue.end,stream));CK(cudaStreamSynchronize(stream));
      float epilogue_ms;CK(cudaEventElapsedTime(&epilogue_ms,epilogue.start,epilogue.end));
      timings["engram_prestage_barrier_ms_cumulative"]=pf.engram_barrier_ms;
      timings["layer"]=l;timings["query_rows"]=work.query.size();timings["project_route_stream_ms"]=project_ms;
      timings["shared_reduce_hc_stream_ms"]=epilogue_ms;pf.layers.push_back(timings);
      std::cerr<<"prefill layer="<<l<<" rows="<<work.query.size()<<" project_ms="<<project_ms
          <<" expert_ms="<<timings["expert_sweep_stream_ms"]<<" epilogue_ms="<<epilogue_ms;
      {   // s0ma change 84: the RAM copies' timeline and the lanes' ends (ms from the sweep's start)
        const double u=timings["ram_copy_union_ms"].get<double>();
        std::cerr<<" copies="<<timings["ram_copies"].get<uint64_t>()<<" copy_busy_ms="<<u<<" copy_gbps="
                 <<(u>0?double(timings["ram_copy_bytes"].get<uint64_t>())/u/1e6:0.)
                 <<" first_copy_ms="<<timings["ram_copy_first_ms"].get<double>()<<" last_copy_ms="<<timings["ram_copy_last_ms"].get<double>()
                 <<" slot_wait_ms="<<timings["ram_slot_wait_ms"].get<double>()<<" issue_gap_ms="<<timings["ram_issue_gap_ms"].get<double>()
                 <<" ring_wait_host_ms="<<timings["ring_wait_host_ms"].get<double>()<<" lanes_end_ms=";
        const auto &ends=timings["lane_end_ms"];const auto &st=timings["lane_staged"];
        for(size_t v=0;v<ends.size();++v)std::cerr<<(v?"/":"")<<int(ends[v].get<double>()+0.5);
        std::cerr<<" lanes_staged=";
        for(size_t v=0;v<st.size();++v)std::cerr<<(v?"/":"")<<st[v].get<size_t>();
      }
      std::cerr<<"\n";
    }
    mark("layers");
    for(size_t r=input.size()-options.logits_tail;r<input.size();++r) {
      pcopy(h.p,pf.hidden.as<char>(r*HC*H*2),HC*H*2);pcopy(pre.p,pf.previous_pre.as<char>(r*4*4),4*4);
      CK(ds41_dense_hc_pre(h.p,pre.as<float>(),x.p,1,H,stream));norm("norm",x.p,x.p,H);
      bf16_float(x.p,f32.as<float>(),H,stream);if(sero_head(x.p,logits.p,1,stream)){}else if(fused_head)fused_head_logits(f32.p,logits.p,1);else linear("head",f32.p,logits.p,V,H);
      result.logits.emplace_back(V);
      CK(cudaMemcpyAsync(result.logits.back().data(),logits.p,V*4,cudaMemcpyDeviceToHost,stream));
      CK(cudaStreamSynchronize(stream));
    }
    mark("head");
    // s0ma change 57: a prompt that admitted experts into RAM changed RAM membership on purpose; VRAM's must still hold
    if(!residency_before.empty()&&vram_admit_used) {
      // s0ma change 60: a prompt that admitted experts into VRAM changed both tiers on purpose
    } else if(!residency_before.empty()&&admit_used) {
      const auto after=p_residency(*experts);
      for(size_t i=0;i<after.size()&&i<residency_before.size();++i)
        if((after[i]&1)!=(residency_before[i]&1))throw std::runtime_error("prefill changed decode VRAM membership");
    } else if(!residency_before.empty()&&p_residency(*experts)!=residency_before)
      throw std::runtime_error("prefill changed decode tier membership; quiesce competing admissions");
    prefix.commit(input,input.size());
    committed=pf.end;
    mark("check");
    size_t capture_count=std::min<size_t>(128,pf.rows);
    for(size_t first=0;first<capture_count;first+=6) {
      capture_rows=int(std::min<size_t>(6,capture_count-first));capture_committed=capture_rows;
      capture_begin=pf.end-capture_count+first;
      pcopy(capture_slab.p,pf.capture.as<char>(first*3*H*2),capture_rows*3*H*2);
      if(capture_sink){auto view=captures();int rc=capture_sink(capture_user,&view);if(rc)throw std::runtime_error("prefill capture sink failed");}
      CK(cudaStreamSynchronize(stream));
    }
    // Decode graphs capture these scratch addresses. The clipped cone can run
    // an odd number of attention rows, unlike the full40-layer token schedule.
    if(h.p!=initial_h)std::swap(h,next_h);
    result.layers=pf.layers;in_step=false;
    const size_t chunk_rows=prefill_chunk?size_t(prefill_chunk->R):0;
    mark("captures");
    if(prefetch){prefill_summary_prefetch={{"copied",prefetch->copied},{"bytes",prefetch->bytes},{"served",prefetch->served},
        {"arena_bytes",prefetch->cap}};std::cerr<<"prefill prefetch: "<<prefetch->served<<" of "<<prefetch->copied
        <<" experts copied during attention served ("<<prefetch->bytes/1048576<<" MiB, arena "<<(prefetch->cap>>20)<<" MiB)\n";}
    prefetch.reset();   // s0ma change 77: the arena and its worker belong to the prompt
    prefill_chunk.reset();pref.reset();result.elapsed_ms=clock_ms()-begin_ms;total_ms+=result.elapsed_ms;
    mark("free");
    {
      size_t free=0,total=0;CK(cudaMemGetInfo(&free,&total));
      std::cerr<<"prefill phases: rows "<<input.size()<<", free "<<(free>>20)<<" MiB, lent "<<lent_chunks<<" of "
               <<lend_chunks_total<<" chunks, ms:";
      double at=begin_ms;
      for(auto &[name,t]:phase_marks){std::cerr<<" "<<name<<" "<<std::fixed<<std::setprecision(1)<<t-at;at=t;}
      std::cerr<<std::defaultfloat<<std::setprecision(6)<<"\n";
    }
    prefill_summary={{"mode",options.mode==PrefillMode::Cold?"cold-full-N-M1-dense-candidate":"append-decode-layout"},
        {"rows",input.size()},{"chunk_rows",options.chunk_rows},{"dense_batch_rows",options.dense_batch_rows},{"batched_chunk_rows",chunk_rows},{"cache_reused_tokens",result.begin_position},
        {"elapsed_ms",result.elapsed_ms},{"cache_retention_checked",!residency_before.empty()},{"layers",result.layers},
        {"images_encoded",images_encoded},{"vision_ms",vision_ms}};
    if(!prefill_warm_counts.empty()) {
      // Residency only (after the tier-membership check): warm RAM with the prompt tail's experts.
      std::vector<std::pair<int,ds41::ExpertId>> order;
      for(auto [id,count]:prefill_warm_counts)order.push_back({-count,id});
      std::sort(order.begin(),order.end());
      std::vector<ds41::ExpertId> ids;for(auto [count,id]:order)ids.push_back(id);
      double warm_start=clock_ms();
      size_t read=experts->warm_host(ids,size_t(config.prefill_warm_gib*double(size_t(1)<<30)));
      prefill_summary["warm_host"]={{"candidates",ids.size()},{"read",read},{"ms",clock_ms()-warm_start}};
      prefill_warm_counts.clear();
    }
    // s0ma change 48: the first prompt's peak (its activations and chunk scratch are still held here), and the
    // buffers it leaves allocated for decode (the stager ring, lanes, the dense FP16 cache) show as the next line
    static bool ledger_first=true;
    if(ledger_first){ledger_first=false;CK(cudaStreamSynchronize(stream));vram_checkpoint("first_prefill_peak");}
    return result;
  }catch(...){cudaStreamSynchronize(stream);prefill_chunk.reset();poisoned=true;in_step=false;throw;}
}
private:
