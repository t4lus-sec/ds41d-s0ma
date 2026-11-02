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
};
// Fixed isolated VRAM and pinned disk ring. It never admits to the decode pool.
// At most one stream() call at a time. Callbacks only enqueue GPU operations,
// never call this stager or the originating store. They are serialized so one
// consumer scratch slab can safely be reused between complete expert jobs.
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
      const Keep& keep_vram = {});   // s0ma change 60: a device destination (the decode store's VRAM slot), copied D2D
};
}
