#pragma once
#include "ds41/prefill/tiers.hpp"
#include "ds41/store.hpp"
#include <cuda_runtime.h>
namespace ds41e::prefill {
struct TransferStats {
  uint64_t host_hits=0,nvme_misses=0,h2d_bytes=0,nvme_bytes=0;
  double h2d_stream_ms=0,host_service_ms=0;
  double ram_service_host_ms=0,nvme_service_host_ms=0;
  uint64_t kept=0; double keep_ms=0;   // s0ma change 57
  uint64_t kept_vram_ram=0,kept_vram_disk=0;   // s0ma change 60
  // s0ma change 84: the copies on the GPU's clock, ms from the sweep's origin event (stream()'s `origin`). A copy's
  // interval is its begin..end events on its copy stream. union: the time at least one copy ran (the link busy, two
  // copy streams not counted twice, unlike h2d_stream_ms); first/last: the first copy's start, the last one's end.
  // Between first and last, each stretch with no RAM copy running is put down to the copy that starts next: until its
  // ring slot's previous expert had finished its kernels (slot_wait: the ring waits on kernels), and after that (issue
  // gap: the slot was free but the copy had not been issued or started). ring_wait_host: the RAM thread's time blocked
  // on a slot's previous kernels (ring_blocked of its waits blocked); 0 with DS41_PREFILL_RING_DEVICE_WAIT=1.
  uint64_t ram_copies=0,ram_copy_bytes=0,ram_slots=0,copy_streams=0,ring_blocked=0,disk_copies=0,device_wait=0;
  double ram_copy_sum_ms=0,ram_copy_union_ms=0,ram_copy_first_ms=0,ram_copy_last_ms=0;
  double ram_slot_wait_ms=0,ram_issue_gap_ms=0,ring_wait_host_ms=0;
  double disk_copy_union_ms=0,copy_union_ms=0;
};
// Fixed isolated VRAM and pinned disk ring. It never admits to the decode pool.
// At most one stream() call at a time. Callbacks only enqueue GPU operations,
// never call this stager or the originating store. They are serialized so one
// consumer scratch slab can safely be reused between complete expert jobs.
// s0ma change 69: pick(id), called under that serialization when the record has landed, may name another consumer
// stream for it (a prefill lane with its own scratch); the slot's reuse then waits on that stream.
class Stager {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  using Ready=std::function<void(ds41::ExpertId,const ds41::DeviceExpert*,cudaStream_t)>;
  Stager(std::vector<ds41::Record>,size_t slots=8);
  ~Stager();
  size_t device_bytes() const noexcept; // immutable payload allocation census
  // s0ma change 57: keep(id) may return a host destination for a record read from the drive; its bytes are copied
  // there from the pinned ring before the ring slot is reused (the decode store's RAM tier, see admit_begin).
  using Keep=std::function<void*(ds41::ExpertId)>;
  TransferStats stream(const std::vector<ds41::ExpertId>&,cudaStream_t,
      const Ready&,const HostLookup& = {},const std::function<void()>& before_disk = {},const Keep& keep = {},
      const Keep& keep_vram = {},   // s0ma change 60: a device destination (the decode store's VRAM slot), copied D2D
      const std::function<cudaStream_t(ds41::ExpertId)>& pick = {},   // s0ma change 69: each record's consumer
      cudaEvent_t origin = nullptr);   // s0ma change 84: a timing event recorded before the call (the sweep's start)
  // s0ma change 84: the last stream() call's copies on the GPU's clock (the TransferStats timeline fields), measured
  // from its origin. Call after synchronizing the sweep's consumers; measurement only (an error leaves zeros).
  void timeline(TransferStats&) noexcept;
};
}
