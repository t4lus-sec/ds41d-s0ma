// v0.2 callbacks execute on the store worker. All per-batch scratch is owned by
// Native and reused only through its ordered consumer stream. Callbacks never
// reenter the store or wait for GPU work.
// (s0ma change 142: rest_event is the one store call they make, a worker-local read with no lock or queue.)
void bmoe_streamed(const std::string &p,int m,const hotpath::RoutePlan &plan,
                  const std::vector<ds41::GemmJob> &gj,
                  const std::vector<ds41::GemmJob> &uj,
                  const std::vector<ds41::GemmJob> &dj) {
  // s0ma change 62: the shared expert was queued by bmoe before its route wait
  const int layer=int(plan.ids.front()/384);
  const bool readiness=consumer_readiness && !lane_b_active;
  if(readiness) consumer_readiness->shared_done(layer,stream);
  double t=clock_ms();
  // Lane B demands skip the width-one drift contract (like DSpark's); LFU/tiers still see them.
  auto lease=experts->need_streamed_batches(plan.ids,stream,
      [&](const std::vector<ds41::ExpertId> &ready,cudaStream_t consumer){
    double enqueue_start=clock_ms();
    if(consumer!=stream || ready.empty()) throw std::logic_error("stream callback contract");
    if(readiness) consumer_readiness->arrived(layer,stream);
    std::array<ds41::GemmJob,36> gates{},ups{},downs{};
    std::array<int,36> rows{}; int count=0,n=int(ready.size());
    for(int i=0;i<n;++i) {
      auto it=std::lower_bound(plan.ids.begin(),plan.ids.end(),ready[i]);
      if(it==plan.ids.end() || *it!=ready[i]) throw std::logic_error("foreign ready expert");
      size_t ix=size_t(it-plan.ids.begin());
      gates[i]=gj[ix]; ups[i]=uj[ix]; downs[i]=dj[ix];
      const auto &e=plan.experts[ix];
      for(int j=0;j<e.count;++j) rows[count++]=e.begin+j;
    }
    zc_begin();
    upload(work_jobs,gates.data(),n*sizeof(ds41::GemmJob));
    upload(work_up_jobs,ups.data(),n*sizeof(ds41::GemmJob));
    upload(work_down_jobs,downs.data(),n*sizeof(ds41::GemmJob));
    upload(work_rows,rows.data(),count*4);
    zc_end();
    ++stream_callbacks;
    if(n>1) ++stream_hit_batches;
    // Node launch geometry depends only on counts. Actual logical IDs, token
    // membership and physical slots are device metadata refreshed before replay.
    Phase *event=nullptr;
    if(config.hotpath_profile && !lane_b_active) {
      if(expert_event_index==expert_events.size()) throw std::logic_error("expert profile event capacity");
      if(route_gap_first[layer]==-1) route_gap_first[layer]=int(expert_event_index);   // s0ma change 83: the gap closes
      event=expert_events[expert_event_index++].get();
      CK(cudaEventRecord(event->start,stream));
    }
    // The device job descriptors carry every layer/expert/output address.
    // Reuse this body across layers; dimensions and scratch addresses are fixed.
    // s0ma change 142: an expert uploaded from RAM in two copies (DS41_MOE_SPLIT_UPLOAD=1) reached this callback when
    // its gate and up landed; its down projection waits for the rest (rest_event: worker-local, takes no store lock)
    std::array<cudaEvent_t,36> rest{}; int rests=0;
    for(int i=0;i<n;++i) if(cudaEvent_t e=experts->rest_event(ready[i])) rest[rests++]=e;
    auto gate_up=[&]{
      CK(expert_gemm_gate_up(work_jobs.as<ds41::GemmJob>(),work_up_jobs.as<ds41::GemmJob>(),n,stream,layer,m));   // s0ma changes 122, 136
      if(exl3) hotpath::activate_routes_bf16(gate.p,up.p,broute.as<float>(),work_rows.as<int>(),
          act.p,count,stream);
      else hotpath::activate_routes(gate.p,up.p,broute.as<float>(),work_rows.as<int>(),
          eq.as<uint8_t>(),es.as<uint8_t>(),count,stream);
    };
    auto down_proj=[&]{
      CK(expert_gemm(work_down_jobs.as<ds41::GemmJob>(),n,ds41::Projection::Down,H,I,stream,layer,m));   // s0ma change 136: m rows
    };
    if(!rests) bgraph("experts.stream."+std::to_string(n)+"."+std::to_string(count),m,[&]{ gate_up(); down_proj(); });
    else {
      bgraph("experts.stream.gu."+std::to_string(n)+"."+std::to_string(count),m,gate_up);
      for(int i=0;i<rests;++i) CK(cudaStreamWaitEvent(stream,rest[i],0));
      bgraph("experts.stream.down."+std::to_string(n),m,down_proj);
      ++split_callbacks;
    }
    if(event) {
      CK(cudaEventRecord(event->end,stream));
      consumer_readiness->expert_done(layer,event->end);
    }
    callback_enqueue_ms+=clock_ms()-enqueue_start;
  },!lane_b_active);
  io_ms+=clock_ms()-t;
  try {
    // Logical IDs and scatter maps are refreshed before each ordered launch.
    bgraph("experts.reduce",m,[&]{
      hotpath::scatter_routes(compact_down.p,route_slots.as<int>(),down.p,plan.count,stream);
      CK(ds41::reduce_experts(down.as<Bf16>(),ids_device.as<int32_t>(),
          shared.as<Bf16>(),y.as<Bf16>(),m,6,H,stream));
    });
    retire_experts(lease);
  } catch(...) {
    CK(cudaStreamSynchronize(stream));
    if(lease.owner) experts->retire(std::move(lease),stream);
    throw;
  }
  previous_routes[layer]=plan.ids;
  if(lookahead_layer==layer+1 && !lane_b_active) {
    lookahead_layer=-1;
    CK(cudaEventSynchronize(lookahead_sync->end));
    const auto *scores=static_cast<const float *>(lookahead_host->data());
    auto rank=[](const float *score,int k){
      std::array<int,384> order{};
      std::iota(order.begin(),order.end(),0);
      std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](int a,int b){
        return score[a]==score[b]?a<b:score[a]>score[b];
      });
      return std::vector<int>(order.begin(),order.begin()+k);
    };
    if(lookahead_audit.is_open())
      for(int d=1;d<=3 && layer+d<40;++d) audit_pred[layer+d][d-1]=rank(scores+(d-1)*384,32);
    if(!lookahead_dryrun) {
      // Every row's top-k, best score first across rows; at most lookahead_n candidates. The store
      // drops VRAM-resident and NVMe-only ones and uploads the RAM-resident rest behind this demand.
      std::vector<std::pair<float,int>> cand;
      for(int r=0;r<lookahead_m;++r)
        for(int e:rank(scores+r*384,lookahead_k)) cand.push_back({scores[r*384+e],e});
      std::stable_sort(cand.begin(),cand.end(),[](const auto &a,const auto &b){return a.first>b.first;});
      std::vector<ds41::ExpertId> predicted; std::set<int> seen;
      // s0ma change 141: with DS41_LOOKAHEAD_RAM_FIRST=1 every distinct candidate goes, best first, and the store uploads
      // at most lookahead_n of those not already in VRAM. Before, the cap took the best lookahead_n candidates, and at
      // full residency (~93% of demands from VRAM) those were nearly always resident already, so little was uploaded
      for(auto &c:cand) if(seen.insert(c.second).second && (lookahead_ram_first || int(predicted.size())<lookahead_n))
        predicted.push_back(ds41::ExpertId((layer+1)*384+c.second));
      if(lookahead_ram_first) prefetch_ram_experts(predicted,size_t(lookahead_n));
      else prefetch_ram_experts(predicted);
    }
    if(lookahead_far_layer>=0) {
      // s0ma change 11: layer far's predicted experts, NVMe-only ones read ahead into pinned slots
      const auto *far=static_cast<const float *>(lookahead_far_host->data());
      std::vector<ds41::ExpertId> ids; std::set<int> seen;
      for(int r=0;r<lookahead_m;++r)
        for(int e:rank(far+r*384,lookahead_k)) if(seen.insert(e).second) ids.push_back(ds41::ExpertId(lookahead_far_layer*384+e));
      lookahead_far_layer=-1;
      if(!lookahead_dryrun) experts->read_ahead(std::move(ids));
    }
  }
  // Predictions affect transfers only. They never choose target routing or
  // admission. Queued prefetch is bounded by the store's one-expert chunk.
  if(config.prefetch_routes && layer+1<40 && !previous_routes[layer+1].empty()) {
    prefetch_experts(previous_routes[layer+1],PrefetchIntent::NextLayer);
  }
}
