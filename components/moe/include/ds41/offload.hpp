#pragma once
#include <cstdint>
#include "ds41/store.hpp"
#include <cuda_runtime.h>
#include <future>
#include <atomic>

namespace ds41 {
class OffloadStore {
public:
    static constexpr unsigned api_version=4;
    struct Config {
        size_t slots = 64, staging_slots = 8;
        size_t prefetch_batch = 1; // demand can overtake queued speculative chunks
        size_t host_budget_bytes = 0;
        unsigned io_depth = 64;
        bool ready_flags = false; // default event ordering, same device table ABI
        unsigned recent_window = 6, decay_interval = 32;
        size_t io_chunk_bytes = 0, h2d_chunk_bytes = 0; // 0: complete native extent/record
        bool distinct_lfu = true;
        bool dynamic_host = false; // placement host IDs become evictable seeds
        unsigned host_recent_window = 6, host_decay_interval = 32;
        // Explicit idle-window refills only; conservative vs measured 14.17 GB/s.
        double refill_gbps = 8.0;
        // s0ma change 12: slot classes for records of several sizes (a mixed EXL3 bank, e.g. K2 and K3 layers).
        // Empty: one class, every slot sized for the largest record (as before). Otherwise class c's slots are
        // class_bytes[c] each (a multiple of 4096): class_slots[c] in VRAM (summing to slots), class_host_slots[c]
        // in the RAM tier (their bytes within host_budget_bytes); layer l's records use class layer_class[l], or
        // (s0ma change 15) expert id's use expert_class[id] when that is given (a bank whose layers mix widths).
        std::vector<size_t> class_bytes, class_slots, class_host_slots;
        std::array<uint8_t,43> layer_class{};
        std::vector<uint8_t> expert_class;
        // s0ma change 51 (V6): the last lend_slots VRAM slots of class lend_class live in their own allocations,
        // lend_chunk_slots slots each, which lend() frees for prompt processing and reclaim() allocates again (empty).
        // They are lent while place() runs, so no pin lands in them. GDS reads skip them (they are not registered).
        int lend_class = -1;
        size_t lend_slots = 0, lend_chunk_slots = 0;
    };
    struct LendInfo {
        size_t chunks = 0, lent = 0, chunk_bytes = 0;
        uint64_t lends = 0, reclaims = 0, moved = 0, dropped = 0, busy = 0, failed = 0;
        // s0ma change 52: where lend() and reclaim() spend their time, summed (ms)
        double settle_ms = 0, sync_ms = 0, move_ms = 0, free_ms = 0, malloc_ms = 0;
    };
    struct CachePolicy {
        bool decay256=false, drift_0_50=false, fast_minima=false;
    };
    // Before place(), only. Existing Config and api_version=4 are unchanged.
    void cache_policy(CachePolicy options);
    struct Lease {
        uint64_t serial = 0;
        std::vector<SlotPool::Allocation> allocations;
        uintptr_t owner = 0;
    };
    // Callbacks run on the store worker and may only enqueue CUDA work. Use
    // logical expert IDs to select output buffers; never reduce in callback order.
    using ReadyCallback = std::function<void(ExpertId,cudaStream_t)>;
    using ReadyBatchCallback = std::function<void(const std::vector<ExpertId>&,cudaStream_t)>;
    enum class PrefetchPriority : unsigned { Session=0, Draft=1, NextLayer=2 };
    struct PrefetchResult { uint64_t requested=0, submitted=0, cancelled=0; };
    struct PrefetchTicket {
        uint64_t id=0;
        std::shared_future<PrefetchResult> completion;
        std::shared_ptr<std::atomic<bool>> cancelled;
    };
    struct HostView {
        ExpertId id=0;
        Record record; // original read plan and six native tensor offsets
        const uint8_t* data=nullptr; // null means not host-resident
        const uint8_t* device_data=nullptr; // explicit mapped CUDA alias
    };
    struct HostLease {
        std::vector<HostView> views;
    private:
        // Shared slot tokens prevent overwrite until the last lease copy dies.
        std::shared_ptr<const HostBuffer> backing_;
        std::vector<std::shared_ptr<const void>> retention_;
        friend class OffloadStore;
    };
    OffloadStore(std::vector<Record> records, Config config);
    ~OffloadStore();
    OffloadStore(const OffloadStore&) = delete;
    OffloadStore& operator=(const OffloadStore&) = delete;
    Lease need_now(std::vector<ExpertId> ids, cudaStream_t consumer);
    Lease need_streamed(std::vector<ExpertId> ids,cudaStream_t consumer,ReadyCallback ready);
    // Coalesces resident hits into one callback; cold completions are singletons.
    // drift=false: a demand outside the width-one token-major traversal (e.g. a pipelined verify's
    // second lane); it updates LFU/tiers but not the drift controller.
    Lease need_streamed_batches(std::vector<ExpertId> ids,cudaStream_t consumer,ReadyBatchCallback ready,bool drift=true);
    // Fixed-capacity prefill ring. Each callback must enqueue all consumers of
    // that expert. Scratch/output storage remains caller-owned until stream end.
    void stream_prefill(std::vector<ExpertId> ids,cudaStream_t consumer,ReadyCallback ready,size_t window=0);
    std::future<void> prefetch(std::vector<ExpertId> ids);
    PrefetchTicket prefetch(std::vector<ExpertId> ids,PrefetchPriority priority);
    PrefetchTicket prefetch_ram(std::vector<ExpertId> ids,PrefetchPriority priority=PrefetchPriority::NextLayer);
    // s0ma change 141: prefetch_ram that drops the VRAM-resident ids too, then uploads at most max_uploads of the rest in
    // the order given (best first): the cap counts uploads, not candidates
    PrefetchTicket prefetch_ram_capped(std::vector<ExpertId> ids,size_t max_uploads,PrefetchPriority priority=PrefetchPriority::NextLayer);
    // s0ma change 11: hand predicted experts to the NVMe read-ahead (DS41_MOE_SPEC_NVME); only those on NVMe alone are
    // read, into its pinned slots, never into VRAM or the RAM tier. A no-op when the read-ahead is off.
    void read_ahead(std::vector<ExpertId> ids);
    // Caller grants an actual link-idle window. One record per admission,
    // no H2D, no policy credit; skip if copies/demand or the deadline intervene.
    PrefetchTicket refill_host(std::vector<ExpertId> ids,uint64_t idle_ns,size_t byte_budget);
    // Bulk NVMe->RAM admission for a caller-granted idle point (e.g. after prompt prefill): records
    // already in VRAM or RAM are skipped; the rest are admitted in order until the byte budget, a
    // blocked admission or a slot reused within the batch, then read in parallel. No H2D, no policy
    // credit. Runs as one demand-priority worker task; returns the number of records read.
    size_t warm_host(std::vector<ExpertId> ids,size_t byte_budget);
    void cancel(const PrefetchTicket& ticket) noexcept;
    void retire(Lease lease, cudaStream_t consumer);
    std::future<void> retire_async(Lease lease,cudaStream_t consumer);
    void pin_vram(std::vector<ExpertId> ids);
    void pin_host(std::vector<ExpertId> ids);
    // No tier allocation, IO, DMA, VRAM admission, policy updates or eviction.
    // Keep the lease until CPU work and every asynchronous reader have finished.
    HostLease borrow_host(std::vector<ExpertId> ids);
    Lease borrow_vram(std::vector<ExpertId> ids,cudaStream_t consumer);
    void reset_history(); // preserve cache contents, clear conversation LFU history
    // s0ma change 51: free up to `chunks` lendable chunks (their valuable experts move into the class's lowest-ranked
    // slots first); returns how many were freed. A chunk with a live lease is left (counted as busy).
    size_t lend(size_t chunks);
    // Allocate lent chunks again, empty, while cudaMemGetInfo leaves at least margin_bytes free; returns how many.
    size_t reclaim(size_t margin_bytes);
    LendInfo lend_info();
    // s0ma change 56: a log of decode's NVMe misses (demand loads only), for measurements; off until switched on.
    // take_nvme_misses() returns the ids logged since the last call. Thread-safe; no store round trip.
    void log_nvme_misses(bool on);
    // s0ma change 62: decode's RAM hits and NVMe misses so far (demand loads), read without a store round trip
    std::pair<uint64_t, uint64_t> demand_totals() const;
    // s0ma change 63: how many of the placement's vram_preload were loaded, and how many loads had to evict (it stops there)
    std::pair<size_t, size_t> preload_result() const;
    // s0ma change 57: RAM-tier slots for experts a prompt is streaming from the drive anyway. admit_begin takes ids in
    // priority order (each with an LFU credit), skips those already in VRAM or RAM, and reserves slots (the least
    // valued RAM residents are evicted) until byte_budget or a blocked admission; dst[i] is where ids[i]'s record
    // bytes go. They stay held, and must be written, before admit_end publishes them.
    struct Admission { uint64_t token = 0; std::vector<ExpertId> ids; std::vector<void*> dst; size_t bytes = 0; };
    Admission admit_begin(const std::vector<ExpertId>& ids, const std::vector<uint32_t>& credit, size_t byte_budget);
    void admit_end(Admission admission);
    // s0ma change 60: the same into VRAM. admit_vram_begin takes slots for ids not in VRAM (the least valued unleased
    // residents of each one's class are evicted; one without a RAM copy is dropped), publishes their table entries
    // and holds them; dst[i] is ids[i]'s slot, to be filled on the GPU. admit_vram_end, once those copies have
    // completed, marks the slots ready and releases them.
    Admission admit_vram_begin(const std::vector<ExpertId>& ids, const std::vector<uint32_t>& credit, size_t byte_budget);
    void admit_vram_end(Admission admission);
    struct VramAdmitStats { uint64_t experts = 0, bytes = 0, victims = 0, victims_dropped = 0; };
    VramAdmitStats vram_admit_stats();
    std::vector<ExpertId> take_nvme_misses();
    // s0ma change 59: and of decode's VRAM misses served from the RAM tier (same switch)
    std::vector<ExpertId> take_ram_hits();
    void report_error(std::exception_ptr error) noexcept;
    void check_error();
    // bit 0: VRAM, bit 1: RAM. Advisory only; admission rechecks on the worker.
    std::vector<uint8_t> residency_snapshot();
    void place(const Placement& placement); // startup, before graph/consumer submission
    const DeviceExpert* table() const;
    int residency(ExpertId id);
    // s0ma change 142: inside a ready-batch callback only (the store worker): the event that completes a split upload
    // of id still in flight, else nullptr. A consumer waits on it before reading the expert's down projection.
    cudaEvent_t rest_event(ExpertId id);
    Stats stats();
private:
    PrefetchTicket prefetch_impl(std::vector<ExpertId> ids,PrefetchPriority priority,bool ram_only,size_t max_uploads=SIZE_MAX);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ds41
