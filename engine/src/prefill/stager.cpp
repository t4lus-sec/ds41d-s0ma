#include "ds41/prefill/stager.hpp"
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>
namespace ds41e::prefill {
namespace {
// s0ma change 84: DS41_PREFILL_RING_DEVICE_WAIT=1: a ring slot's next upload waits on the GPU for the kernels of the
// slot's previous expert (cudaStreamWaitEvent on its copy stream) instead of the uploading thread blocking on them
// (cudaEventSynchronize) and only then enqueueing the copy. The same copies into the same slots in the same order on
// the same streams, each expert's kernels on the lane pick() names: only who waits changes, so nothing computed
// does. The host then enqueues a layer's copies and staged kernels at once, and a freed slot's copy starts without
// a host wake-up and enqueue in between. Mind DS41_PREFILL_COPIES_FIRST=1 without DS41_PREFILL_STAGED_LANES: residents
// enqueued on a lane after a staged expert then queue behind that expert's copy (correct, but slower).
bool ring_device_wait(){static const bool on=[]{const char*v=std::getenv("DS41_PREFILL_RING_DEVICE_WAIT");return v&&v[0]=='1';}();return on;}
struct RingWait {double ms=0;uint64_t blocked=0;};
void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
double ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Event {
 cudaEvent_t e=nullptr;
 Event(){ck(cudaEventCreate(&e));}
 ~Event(){if(e)cudaEventDestroy(e);}
 Event(const Event&)=delete;Event& operator=(const Event&)=delete;
};
}
struct Stager::Impl {
 size_t slots,record_bytes=0;
 int device=0;
 std::map<ds41::ExpertId,ds41::Record> records;
 ds41::DirectReader reader{64};
 std::unique_ptr<ds41::HostBuffer> staging,table;
 void* weights=nullptr;
 cudaStream_t ram=nullptr,disk=nullptr,ram2=nullptr;   // ram2: s0ma change 77, a second copy stream
 std::vector<std::unique_ptr<Event>> consumed,dma_done;
 std::vector<unsigned char> used;
 std::mutex enqueue;
 std::exception_ptr poison;
 // s0ma change 84: each upload's begin/end/fin timing events, pooled (a sweep's are complete before the next sweep: its
 // caller synchronizes, and stream() drains the ring first), and what timeline() needs of the last sweep
 std::vector<std::unique_ptr<Event>> pool_begin,pool_end,pool_fin;
 std::unique_ptr<Event> own_origin;
 struct Last {bool valid=false;size_t ram=0,disk=0,ram_slots=0;uint64_t ram_bytes=0;std::vector<long> prev;cudaEvent_t origin=nullptr;} last;
 Impl(std::vector<ds41::Record> input,size_t n):slots(n),used(n,false) {
  if(n<2||input.empty())throw std::invalid_argument("isolated staging needs at least two slots");
  for(auto &r:input){if(!r.bytes||r.bytes>SIZE_MAX-4095)throw std::invalid_argument("stager record size");
   for(auto v:{r.gate,r.gate_scale,r.up,r.up_scale,r.down,r.down_scale})if(v.offset>r.bytes||v.bytes>r.bytes-v.offset)throw std::invalid_argument("stager view bounds");
   if(!records.emplace(r.id,r).second||r.id>=ds41::expert_count)throw std::invalid_argument("stager expert ID");record_bytes=std::max(record_bytes,(r.bytes+4095)&~size_t(4095));}
  try {
   ck(cudaGetDevice(&device));ck(cudaMalloc(&weights,slots*record_bytes));
   ck(cudaStreamCreateWithFlags(&ram,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&disk,cudaStreamNonBlocking));
   ck(cudaStreamCreateWithFlags(&ram2,cudaStreamNonBlocking));
   staging=std::make_unique<ds41::HostBuffer>(slots*record_bytes,true);
   table=std::make_unique<ds41::HostBuffer>(ds41::expert_count*sizeof(ds41::DeviceExpert),true);
   std::fill_n(static_cast<ds41::DeviceExpert*>(table->data()),ds41::expert_count,ds41::DeviceExpert{});
   for(size_t i=0;i<slots;++i){consumed.push_back(std::make_unique<Event>());dma_done.push_back(std::make_unique<Event>());}
  }catch(...){clear();throw;}
 }
 void clear()noexcept {
  if(ram)cudaStreamSynchronize(ram);if(ram2)cudaStreamSynchronize(ram2);if(disk)cudaStreamSynchronize(disk);
  // Last consumers must finish before weight slots and mapped descriptors die.
  for(size_t i=0;i<consumed.size();++i)if(used[i])cudaEventSynchronize(consumed[i]->e);
  consumed.clear();dma_done.clear();table.reset();staging.reset();
  if(weights)cudaFree(weights);weights=nullptr;
  if(ram)cudaStreamDestroy(ram);if(ram2)cudaStreamDestroy(ram2);if(disk)cudaStreamDestroy(disk);ram=disk=ram2=nullptr;
 }
 ~Impl(){clear();}
 void upload(ds41::ExpertId id,const void* data,size_t slot,cudaStream_t copy,
             cudaStream_t consumer,const Ready& ready,Event& begin,Event& end,
             const std::function<cudaStream_t(ds41::ExpertId)>& pick,Event& fin,RingWait& wait) {
  if(used[slot]) {   // s0ma change 84: on the GPU (the copy stream waits) or, as before, on this thread
   if(ring_device_wait())ck(cudaStreamWaitEvent(copy,consumed[slot]->e,0));
   else {const auto q=cudaEventQuery(consumed[slot]->e);
    if(q==cudaErrorNotReady){const double t=ms();ck(cudaEventSynchronize(consumed[slot]->e));wait.ms+=ms()-t;++wait.blocked;}
    else ck(q);}
  }
  auto &r=records.at(id);auto* base=static_cast<uint8_t*>(weights)+slot*record_bytes;
  // Each logical ID appears once in a sweep; this mapped descriptor stays
  // stable until all callbacks in that sweep finish. No pageable H2D metadata.
  auto* views=static_cast<ds41::DeviceExpert*>(table->data());
  views[id]={base+r.gate.offset,base+r.gate_scale.offset,base+r.up.offset,base+r.up_scale.offset,
      base+r.down.offset,base+r.down_scale.offset,nullptr,0,int32_t(slot)};
  ck(cudaEventRecord(begin.e,copy));
  ck(cudaMemcpyAsync(base,data,r.bytes,cudaMemcpyHostToDevice,copy));
  ck(cudaEventRecord(end.e,copy));
  {std::lock_guard<std::mutex> lock(enqueue);
   if(pick)if(cudaStream_t s=pick(id))consumer=s;   // s0ma change 69
   ck(cudaStreamWaitEvent(consumer,end.e,0));
   ready(id,static_cast<ds41::DeviceExpert*>(table->device_data()),consumer);
   ck(cudaEventRecord(fin.e,consumer));   // s0ma change 84: when the slot is free again, on the GPU's clock
   ck(cudaEventRecord(consumed[slot]->e,consumer));used[slot]=true;
  }
 }
};
Stager::Stager(std::vector<ds41::Record> records,size_t slots):impl(new Impl(std::move(records),slots)){}
Stager::~Stager()=default;
size_t Stager::device_bytes() const noexcept{return impl->slots*impl->record_bytes;}
TransferStats Stager::stream(const std::vector<ds41::ExpertId>& ids,cudaStream_t consumer,
    const Ready& ready,const HostLookup& host,const std::function<void()>& before_disk,const Keep& keep,const Keep& keep_vram,
    const std::function<cudaStream_t(ds41::ExpertId)>& pick,cudaEvent_t origin) {
 auto &p=*impl;if(p.poison)std::rethrow_exception(p.poison);
 if(!consumer||!ready)throw std::invalid_argument("stager consumer/callback");
 // A later sweep may reuse a logical table entry in a different slot.
 // Drain the preceding sweep before republishing any mapped descriptor.
 for(size_t i=0;i<p.slots;++i)if(p.used[i])ck(cudaEventSynchronize(p.consumed[i]->e));
 double start=ms();TransferStats stats;
 std::vector<std::pair<ds41::ExpertId,HostView>> ram;
 std::vector<const ds41::Record*> disk;
 std::set<ds41::ExpertId> seen;
 for(auto id:ids){
  if(!seen.insert(id).second)throw std::invalid_argument("stager duplicate expert");
  auto &r=p.records.at(id);auto view=host?host(id):HostView{};
  if(view.data){if(view.bytes<r.bytes||!view.keepalive)throw std::invalid_argument("unsafe prefill host view");
   cudaPointerAttributes attributes{};ck(cudaPointerGetAttributes(&attributes,view.data));
   if(attributes.type!=cudaMemoryTypeHost)throw std::invalid_argument("prefill host view must be pinned host memory");
   ram.emplace_back(id,std::move(view));++stats.host_hits;}
  else {disk.push_back(&r);++stats.nvme_misses;for(auto &e:r.extents)stats.nvme_bytes+=e.bytes;}
  stats.h2d_bytes+=r.bytes;
 }
 p.last.valid=false;   // s0ma change 84
 if(ids.empty())return stats;
 size_t ram_slots=ram.empty()?0:disk.empty()?p.slots:std::max<size_t>(1,p.slots/2);
 size_t disk_slots=p.slots-ram_slots;
 // s0ma change 84: the events from the pool (created once, grown to the largest sweep); the timeline's origin (the
 // caller's event, else one recorded here on the consumer before any upload), and for each upload (RAM first, then the
 // drive's) the upload whose slot it reuses in this sweep
 while(p.pool_begin.size()<ids.size()){p.pool_begin.push_back(std::make_unique<Event>());p.pool_end.push_back(std::make_unique<Event>());
  p.pool_fin.push_back(std::make_unique<Event>());}
 auto &begins=p.pool_begin,&ends=p.pool_end,&fins=p.pool_fin;
 if(!origin){if(!p.own_origin)p.own_origin=std::make_unique<Event>();ck(cudaEventRecord(p.own_origin->e,consumer));origin=p.own_origin->e;}
 std::vector<long> prev(ids.size(),-1);
 std::vector<long> last_in(p.slots,-1);
 RingWait ram_wait,disk_wait;
 std::future<void> ram_job;
 try {
  if(!ram.empty())ram_job=std::async(std::launch::async,[&]{
   double ram_start=ms();ck(cudaSetDevice(p.device));
   // s0ma change 77: RAM uploads on two copy streams (the card has more than one copy engine), by slot parity, so a
   // slot's uploads and its keep copies stay on one stream in order; one stream if the slots are odd or with
   // DS41_PREFILL_COPY_STREAMS=1
   static const int streams=[]{const char*v=std::getenv("DS41_PREFILL_COPY_STREAMS");return v&&v[0]=='1'?1:2;}();
   const bool two=streams==2&&ram_slots%2==0;
   stats.copy_streams=two?2:1;
   for(size_t i=0;i<ram.size();++i){
    cudaStream_t cs=two&&((i%ram_slots)&1)?p.ram2:p.ram;
    prev[i]=i>=ram_slots?long(i-ram_slots):-1;
    p.upload(ram[i].first,ram[i].second.data,i%ram_slots,cs,consumer,ready,*begins[i],*ends[i],pick,*fins[i],ram_wait);
    // s0ma change 60: into the decode store's VRAM slot too, D2D on the same stream, before the slot's next upload
    if(keep_vram)if(void* d=keep_vram(ram[i].first)){
     ck(cudaMemcpyAsync(d,static_cast<const uint8_t*>(p.weights)+(i%ram_slots)*p.record_bytes,p.records.at(ram[i].first).bytes,
                        cudaMemcpyDeviceToDevice,cs));++stats.kept_vram_ram;}
   }
   ck(cudaStreamSynchronize(p.ram)); // borrowed host leases end only after DMA
   ck(cudaStreamSynchronize(p.ram2));
   stats.ram_service_host_ms=ms()-ram_start;
  });
  if(!disk.empty()) {
   if(before_disk)before_disk();
   double io_start=ms();
   std::vector<void*> ring;
   for(size_t i=0;i<disk_slots;++i)ring.push_back(static_cast<char*>(p.staging->data())+i*p.record_bytes);
   p.reader.stream(disk,ring,[&](size_t i,size_t buffer){
    prev[ram.size()+i]=last_in[ram_slots+buffer];last_in[ram_slots+buffer]=long(ram.size()+i);
    p.upload(disk[i]->id,ring[buffer],ram_slots+buffer,p.disk,consumer,ready,*begins[ram.size()+i],*ends[ram.size()+i],pick,
             *fins[ram.size()+i],disk_wait);
    ck(cudaEventRecord(p.dma_done[buffer]->e,p.disk));
    // s0ma change 57: a kept record goes to its RAM-tier slot. Change 58: by a D2H DMA from the VRAM slot it was just
    // uploaded to, on the same stream (after the upload, before the slot's next upload), instead of a CPU memcpy on
    // this reading thread (~1 ms a record, 7 s a bench). The stream is synchronized before stream() returns.
    if(keep)if(void* d=keep(disk[i]->id)){
     double t=ms();
     ck(cudaMemcpyAsync(d,static_cast<const uint8_t*>(p.weights)+(ram_slots+buffer)*p.record_bytes,disk[i]->bytes,
                        cudaMemcpyDeviceToHost,p.disk));
     stats.keep_ms+=ms()-t;++stats.kept;}
    if(keep_vram)if(void* d=keep_vram(disk[i]->id)){   // s0ma change 60
     ck(cudaMemcpyAsync(d,static_cast<const uint8_t*>(p.weights)+(ram_slots+buffer)*p.record_bytes,disk[i]->bytes,
                        cudaMemcpyDeviceToDevice,p.disk));++stats.kept_vram_disk;}
   },[&](size_t buffer){ck(cudaEventSynchronize(p.dma_done[buffer]->e));});
   ck(cudaStreamSynchronize(p.disk));
   stats.nvme_service_host_ms=ms()-io_start;
  }
  if(ram_job.valid())ram_job.get();
  for(size_t i=0;i<ids.size();++i){float elapsed;ck(cudaEventElapsedTime(&elapsed,begins[i]->e,ends[i]->e));stats.h2d_stream_ms+=elapsed;}
  // s0ma change 84: what the timeline needs; timeline() computes it once the caller has synchronized
  p.last.valid=true;p.last.ram=ram.size();p.last.disk=disk.size();p.last.ram_slots=ram.empty()?0:ram_slots;
  p.last.ram_bytes=0;for(const auto &r:ram)p.last.ram_bytes+=p.records.at(r.first).bytes;
  p.last.prev=std::move(prev);p.last.origin=origin;
  stats.device_wait=ring_device_wait()?1:0;
  stats.ring_wait_host_ms=ram_wait.ms;stats.ring_blocked=ram_wait.blocked;
  stats.host_service_ms=ms()-start;
  return stats;
 }catch(...){
  p.poison=std::current_exception();
  if(ram_job.valid())try{ram_job.get();}catch(...){}
  cudaStreamSynchronize(p.ram);cudaStreamSynchronize(p.ram2);cudaStreamSynchronize(p.disk);cudaStreamSynchronize(consumer);
  std::rethrow_exception(p.poison);
 }
}
// s0ma change 84: the last sweep's copies on the GPU's clock, ms from its origin, into stats. Call it after the caller
// has synchronized the sweep's consumers (every upload's begin/end and every fin then complete). Measurement only: an
// error leaves the fields at zero and never poisons the stager.
void Stager::timeline(TransferStats& stats) noexcept {
 auto &p=*impl;
 if(!p.last.valid)return;
 p.last.valid=false;
 try {
  const auto &L=p.last;
  ck(cudaEventSynchronize(L.origin));
  auto at=[&](cudaEvent_t e){float t;ck(cudaEventElapsedTime(&t,L.origin,e));return double(t);};
  struct Copy {double s,e,free;};
  auto collect=[&](size_t first,size_t n){
   std::vector<Copy> c;c.reserve(n);
   for(size_t i=first;i<first+n;++i){
    double f=-1e300;
    if(L.prev[i]>=0){const auto e=p.pool_fin[size_t(L.prev[i])]->e;ck(cudaEventSynchronize(e));f=at(e);}
    c.push_back({at(p.pool_begin[i]->e),at(p.pool_end[i]->e),f});
   }
   std::sort(c.begin(),c.end(),[](const Copy&a,const Copy&b){return a.s<b.s;});
   return c;
  };
  auto union_of=[](const std::vector<Copy>& c){double u=0,lo=0,hi=-1e300;
   for(const auto &x:c){if(x.s>hi){if(hi>lo)u+=hi-lo;lo=x.s;hi=x.e;}else hi=std::max(hi,x.e);}
   if(hi>lo)u+=hi-lo;
   return u;};
  const auto rc=collect(0,L.ram),dc=collect(L.ram,L.disk);
  TransferStats t;
  t.ram_copies=rc.size();t.disk_copies=dc.size();t.ram_slots=L.ram_slots;t.ram_copy_bytes=L.ram_bytes;
  t.ram_copy_union_ms=union_of(rc);t.disk_copy_union_ms=union_of(dc);
  {std::vector<Copy> all(rc);all.insert(all.end(),dc.begin(),dc.end());
   std::sort(all.begin(),all.end(),[](const Copy&a,const Copy&b){return a.s<b.s;});t.copy_union_ms=union_of(all);}
  if(!rc.empty()){
   t.ram_copy_first_ms=rc.front().s;double busy=rc.front().e;
   for(const auto &x:rc){
    t.ram_copy_sum_ms+=x.e-x.s;t.ram_copy_last_ms=std::max(t.ram_copy_last_ms,x.e);
    if(x.s>busy){const double freed=std::min(std::max(x.free,busy),x.s);   // idle [busy, x.s): slot wait, then issue gap
     t.ram_slot_wait_ms+=freed-busy;t.ram_issue_gap_ms+=x.s-freed;}
    busy=std::max(busy,x.e);
   }
  }
  stats.ram_copies=t.ram_copies;stats.disk_copies=t.disk_copies;stats.ram_slots=t.ram_slots;stats.ram_copy_bytes=t.ram_copy_bytes;
  stats.ram_copy_union_ms=t.ram_copy_union_ms;stats.disk_copy_union_ms=t.disk_copy_union_ms;stats.copy_union_ms=t.copy_union_ms;
  stats.ram_copy_first_ms=t.ram_copy_first_ms;stats.ram_copy_last_ms=t.ram_copy_last_ms;stats.ram_copy_sum_ms=t.ram_copy_sum_ms;
  stats.ram_slot_wait_ms=t.ram_slot_wait_ms;stats.ram_issue_gap_ms=t.ram_issue_gap_ms;
 }catch(...){cudaGetLastError();}
}

}
