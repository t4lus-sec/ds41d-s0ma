#include <cstdlib>
#include <cstring>
#include <cmath>
#include "ds41/engine.hpp"
#include "ds41/spark.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#ifdef DS41_SPARK_POLICY_V2
#include "ds41/spark_policy.h"
#endif

extern "C" size_t ds41_draft_draw_scratch_bytes();
extern "C" int ds41_draft_draw_gpu(const float *logits, int n, double temperature, uint64_t seed, uint64_t step,
                                   uint32_t slot, uint32_t purpose, double *q, void *scratch, void **result,
                                   cudaStream_t stream);
extern "C" size_t ds41_verify_draw_scratch_bytes();
extern "C" size_t ds41_verify_draw_result_bytes();
extern "C" int ds41_verify_draw_gpu(const float *logits, int n, int count, const double *const *q, const int32_t *d,
                                    double temperature, uint64_t seed, uint64_t step, double *p, void *scratch,
                                    void **result, cudaStream_t stream);
namespace ds41e {
#ifdef DS41_SPARK_NATIVE
namespace {
class SparkTarget final : public Forward, public DraftProvider {
  std::unique_ptr<Forward> target_;
  std::unique_ptr<ds41_spark, decltype(&ds41_spark_destroy)> spark_{
      nullptr, ds41_spark_destroy};
  ds41_target_weights_v2 weights_{};
  ds41_target_services_v1 services_{};
  ds41_spark_policy_config config_{};
  ds41_spark_policy policy_{};
  ds41_target_capture_sink extra_sink_ = nullptr;
  void *extra_user_ = nullptr;
  int fixed_width_;
  bool policy_v2_ = false;
#ifdef DS41_SPARK_POLICY_V2
  ds41_spark_policy_config_v2 policy2_config_{};
  ds41_spark_policy_state_v2 policy2_state_{};
  ds41_spark_policy_features_v2 policy2_features_{};
  ds41_spark_policy_decision_v2 policy2_decision_{};
  bool policy2_pending_ = false, q0_ready_ = false;
  std::vector<float> q0_;
  std::vector<int> width_history_;
  uint64_t policy2_attempts_ = 0;
  static double clock_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void policy_checked(int rc, const char *operation) {
    if (rc) {
      poisoned_ = true;
      throw std::runtime_error(std::string("Spark policy-v2 ") + operation +
                               " failed: " + std::to_string(rc));
    }
  }
  void reset_policy2() {
    if (!policy_v2_) return;
    policy_checked(ds41_spark_policy_reset_v2(&policy2_state_, &policy2_config_),
                   "reset");
    policy2_pending_ = q0_ready_ = false;
    q0_.clear(); width_history_.clear(); policy2_attempts_ = 0;
  }
#endif
  size_t bindings_ = 0;
  bool poisoned_ = false;
  bool prompt_active_ = false, cold_seed_pending_ = false;
  bool incomplete_prompt_ = false;
  int64_t prompt_rows_ = 0, captures_seen_ = 0, tail_rows_ = 0;
  // Opt-in (DS41_SPARK_APPEND_RESTORE=1): prefill appends longer than 128 rows and segmented cold
  // prompts. The drafter is restored from the append's last 128 captures (frozen append-tail ABI).
  bool append_restore_ = false, append_pending_ = false;
  uint64_t append_restores_ = 0;
  float last_confidence_ = std::numeric_limits<float>::quiet_NaN();
  // s0ma GPU draws: q [V] f64 and the kernels' scratch on the device, pinned copies for the host
  double *draw_q_dev_ = nullptr, *draw_q_host_ = nullptr;
  void *draw_scratch_ = nullptr, *draw_result_host_ = nullptr;
  // s0ma change 29 (DS41_SPARK_GPU_VERIFY=1): q stays on the device, one slot a draft position ([5, V] f64), and the
  // acceptance runs there (verify_draw.cu): p [6, V] f64, its scratch, a pinned copy of the result
  // DS41_SPARK_GPU_VERIFY=check also reruns every step's acceptance on the host from the same logits and q,
  // keeps the host's result and counts the steps whose results differ (spark_gpu_verify_mismatches)
  static int gpu_verify_mode() {
    static const int mode = [] {
      const char *v = std::getenv("DS41_SPARK_GPU_VERIFY");
      return !v ? 0 : std::string(v) == "1" ? 1 : std::string(v) == "check" ? 2 : 0;
    }();
    return mode;
  }
  static bool gpu_verify_on() { return gpu_verify_mode() != 0; }
  double *verify_p_dev_ = nullptr;
  void *verify_scratch_ = nullptr, *verify_result_host_ = nullptr;
  uint64_t gpu_verify_steps_ = 0, gpu_verify_fallbacks_ = 0, gpu_verify_mismatches_ = 0;
  std::unique_ptr<float, decltype(&cudaFreeHost)> confidence_host_{[] { void *p = nullptr; if (cudaMallocHost(&p, sizeof(float)) != cudaSuccess) throw std::runtime_error("confidence host buffer"); return static_cast<float *>(p); }(), cudaFreeHost};
  int64_t tail_limit_ = 0, append_old_ = 0;
  std::string seed_mode_ = "unseeded";
  std::unique_ptr<void, decltype(&cudaFree)> prompt_tail_{nullptr, cudaFree};
  std::string callback_error_;

  void check() const {
    if (poisoned_)
      throw std::runtime_error("native Spark adapter poisoned; restart ds41d");
  }
  void checked(int rc) {
    if (rc) {
      poisoned_ = true;
      throw std::runtime_error(std::string("native Spark adapter: ") +
                               ds41_spark_error());
    }
  }
  void cuda_checked(cudaError_t rc) {
    if (rc != cudaSuccess) {
      poisoned_ = true;
      throw std::runtime_error(std::string("native Spark CUDA: ") +
                               cudaGetErrorString(rc));
    }
  }
  void bind(const char *name, const ds41_target_tensor_v1 &view) {
    if (view.abi_version != 1 || view.struct_bytes != sizeof(view) ||
        !view.data)
      throw std::runtime_error("invalid borrowed MTP tensor view");
    checked(ds41_spark_bind(spark_.get(), name,
                            {view.data, view.dtype, view.rows, view.columns}));
    ++bindings_;
  }
  void accept(const ds41_target_capture_v1 &view) {
    if (!cold_seed_pending_ && !append_pending_) {
      checked(ds41_spark_ingest(spark_.get(), view.hidden_bf16,
                               view.begin_position, view.committed_rows));
      return;
    }
    const int64_t limit = tail_limit_ ? tail_limit_ : prompt_rows_;
    if (view.begin_position != captures_seen_ ||
        view.committed_rows > limit - captures_seen_)
      throw std::runtime_error("noncontiguous cold prompt captures");
    const int64_t tail_begin = std::max<int64_t>(0, limit - 128);
    const int64_t end = captures_seen_ + view.committed_rows;
    const int64_t begin = std::max(captures_seen_, tail_begin);
    if (begin < end) {
      auto *dst = static_cast<char *>(prompt_tail_.get()) +
                  (begin - tail_begin) * 30720;
      auto *src = static_cast<const char *>(view.hidden_bf16) +
                  (begin - captures_seen_) * 30720;
      cuda_checked(cudaMemcpyAsync(dst, src, (end - begin) * 30720,
                                   cudaMemcpyDeviceToDevice,
                                   static_cast<cudaStream_t>(weights_.cuda_stream)));
      tail_rows_ += end - begin;
    }
    captures_seen_ = end;
  }
  static int bind_view(void *user, const char *name,
                       const ds41_target_tensor_v1 *view) noexcept {
    auto &self = *static_cast<SparkTarget *>(user);
    try {
      self.bind(name, *view);
      return 0;
    } catch (const std::exception &e) {
      self.callback_error_ = e.what();
      self.poisoned_ = true;
      return -1;
    }
  }
  static int ingest(void *user, const ds41_target_capture_v1 *view) noexcept {
    auto &self = *static_cast<SparkTarget *>(user);
    try {
      self.check();
      if (!view || view->abi_version != 1 || !view->hidden_bf16 ||
          view->columns != 15360 || view->begin_position < 0 ||
          view->committed_rows < 1 || view->committed_rows > view->rows ||
          view->rows > 6 || view->row_stride_bytes != 30720 ||
          view->cuda_stream != self.weights_.cuda_stream)
        throw std::runtime_error("invalid accepted target captures");
      self.accept(*view);
      if (self.extra_sink_ && self.extra_sink_(self.extra_user_, view))
        throw std::runtime_error("external capture sink failed");
      return 0;
    } catch (const std::exception &e) {
      self.callback_error_ = e.what();
      self.poisoned_ = true;
      return -1;
    }
  }

public:
  SparkTarget(std::unique_ptr<Forward> target, int width, bool policy_v2)
      : target_(std::move(target)), fixed_width_(width), policy_v2_(policy_v2) {
#ifndef DS41_SPARK_POLICY_V2
    if (policy_v2_)
      throw std::invalid_argument("rebuild with DS41_SPARK_POLICY_V2 for this option");
#else
    if (policy_v2_) {
      ds41_spark_policy_default_v2(&policy2_config_);
      reset_policy2();
    }
#endif
    if (width < 0 || width > 6 || !target_)
      throw std::invalid_argument("Spark fixed width must be 0..6");
    if (ds41_target_query(target_.get(), DS41_TARGET_WEIGHTS, 2, &weights_,
                          sizeof(weights_)) ||
        ds41_target_query(target_.get(), DS41_TARGET_SERVICES, 1, &services_,
                          sizeof(services_)))
      throw std::runtime_error("native target weights-v2/services unavailable");
    ds41_spark_config config{};
    config.abi_version = DS41_SPARK_ABI;
    config.stream = weights_.cuda_stream;
    config.blas_handle = weights_.blas_handle;
    config.user = services_.user;
    config.experts = services_.experts;
    config.rollback = services_.rollback;
    config.head = services_.head;   // s0ma change 43
    ds41_spark *raw = nullptr;
    checked(ds41_spark_create(&config, &raw));
    spark_.reset(raw);
    int count =
        ds41_target_visit_tensors(target_.get(), "mtp.", 1, bind_view, this);
    if (count || bindings_ == 0)
      throw std::runtime_error("native Spark needs --load-mtp bindings: " +
                               callback_error_);
    for (const char *name : {"embed.weight", "head.weight", "rope"}) {
      // s0ma change 43: the target serves the draft logits through its own head (Sero's K4): no BF16 head to bind
      if (services_.head && std::string(name) == "head.weight") continue;
      ds41_target_tensor_v1 view{};
      if (ds41_target_tensor(target_.get(), name, 1, &view, sizeof(view)))
        throw std::runtime_error(std::string("missing shared native tensor ") +
                                 name);
      if (std::string(name) == "head.weight" && view.dtype == DS41_TARGET_BF16) {
        // Native BF16 target head (--fused-head): the drafter uses the compact BF16 head over the
        // same matrix. Draft distributions only; the target verifier keeps its own head.
        bind("draft.head.weight", view);
        continue;
      }
      bind(name, view);
    }
    ds41_spark_policy_default(&config_);
    // Owner exclusive resident benchmark: context128/synthetic captures,
    // Markov/confidence + Torch FP32 softmax + GPU v2 draw/feedback included.
    // Target work and draft offload are excluded. Actual completed engine
    // observations replace these priors; the host fallback has its own cost.
    const double resident_step_ms[] = {0,       3.76635, 4.26110,
                                       4.85622, 5.27362, 5.83464};
    std::copy(resident_step_ms, resident_step_ms + 6, config_.draft_ms);
    ds41_spark_policy_reset(&policy_, &config_);
    void *tail = nullptr;
    cuda_checked(cudaMalloc(&tail, 128 * 30720));
    prompt_tail_.reset(tail);
    target_->set_capture_sink(ingest, this);
    if (const char *v = std::getenv("DS41_SPARK_APPEND_RESTORE")) append_restore_ = std::string(v) == "1";
  }
  ~SparkTarget() override {
    try {
      target_->set_capture_sink(nullptr, nullptr);
    } catch (...) {
    }
    cudaStreamSynchronize(static_cast<cudaStream_t>(weights_.cuda_stream));
  }
  float last_confidence() const override { return last_confidence_; }
  int vocab() const override { return target_->vocab(); }
  int eos() const override { return target_->eos(); }
  void begin_request() override {
    check();
#ifdef DS41_SPARK_POLICY_V2
    // Server attempt nonces restart on each request, including prefix reuse.
    if (policy2_pending_) throw std::logic_error("request during policy transaction");
    reset_policy2();
#endif
    target_->begin_request();
  }
  void reset() override {
    check();
    target_->reset();
    checked(ds41_spark_reset(spark_.get()));
    ds41_spark_policy_reset(&policy_, &config_);
#ifdef DS41_SPARK_POLICY_V2
    reset_policy2();
#endif
    prompt_active_ = cold_seed_pending_ = incomplete_prompt_ = false;
    prompt_rows_ = captures_seen_ = tail_rows_ = 0;
    append_pending_ = false; tail_limit_ = append_old_ = 0;
    seed_mode_ = "unseeded";
  }
  void begin_prompt(int64_t rows, bool append) override {
    check();
    const int64_t committed = ds41_spark_committed(spark_.get());
    if (prompt_active_ || incomplete_prompt_ || rows < 1 ||
        (append ? committed < 1 || committed > rows : committed != 0))
      throw std::logic_error("invalid Spark prompt lifecycle");
    target_->begin_prompt(rows, append);
    prompt_active_ = true;
    cold_seed_pending_ = !append;
    prompt_rows_ = rows;
    captures_seen_ = tail_rows_ = 0;
    append_pending_ = false; tail_limit_ = append_old_ = 0;
  }
  void end_prompt(bool complete) override {
    check();
    if (!prompt_active_)
      throw std::logic_error("Spark prompt was not started");
    target_->end_prompt(complete);
    if (!complete) {
      incomplete_prompt_ = true;
      seed_mode_ = "cancelled-incomplete";
    } else if (cold_seed_pending_) {
      if (captures_seen_ != prompt_rows_ ||
          tail_rows_ != std::min<int64_t>(128, prompt_rows_))
        throw std::logic_error("missing cold prompt captures");
      // Projection RMS uses the full prompt M, independently of the target's
      // scheduling width. The bounded buffer retains only the last128 rows.
      checked(ds41_spark_seed(spark_.get(), prompt_tail_.get(), prompt_rows_));
      seed_mode_ = "full-M-cold";
    } else {
      if (ds41_spark_committed(spark_.get()) != prompt_rows_)
        throw std::logic_error("Spark append length differs from target prompt");
      if (seed_mode_ != "segmented-cold+append-restore")
        seed_mode_ = append_restores_ ? "cached-append-restore" : "cached-M1-append";
    }
    prompt_active_ = cold_seed_pending_ = false;
  }
  std::vector<std::vector<float>> verify(const std::vector<int> &ids) override {
    check();
    if (incomplete_prompt_)
      throw std::logic_error("reset required after incomplete Spark prompt");
    return target_->verify(ids);
  }
  bool supports_prefill(PrefillMode mode, size_t rows) const override {
    // The frozen owner ABI has no cached-M1 absolute-tail restoration hook.
    // A short append contains every new capture; a long append uses verify.
    if (append_restore_ && prompt_active_)
      return target_->supports_prefill(mode, rows);
    // Without the restore, a cold prefill must cover the whole prompt.
    return (mode == PrefillMode::Cold ? int64_t(rows) == prompt_rows_ : rows <= 128) &&
           target_->supports_prefill(mode, rows);
  }
  PrefillResult prefill(const std::vector<int> &ids,
                       const PrefillOptions &options) override {
    check();
    if (!prompt_active_ || incomplete_prompt_ ||
        !supports_prefill(options.mode, ids.size()))
      throw std::logic_error("unsupported Spark prefill lifecycle");
    const int64_t n = int64_t(ids.size());
    bool segment_seed = false, restore = false;
    if (options.mode == PrefillMode::Cold) {
      if (!cold_seed_pending_ || n > prompt_rows_ || (n != prompt_rows_ && !append_restore_) ||
          captures_seen_ != 0 || tail_rows_ != 0)
        throw std::logic_error("invalid full-prompt capture-tail descriptor");
      // Only this explicit full-prompt entrypoint may start the accepted tail
      // beyond zero. The ordinary verify sink still requires every input row.
      // A first segment of a longer prompt seeds the drafter from its own tail.
      segment_seed = n != prompt_rows_;
      tail_limit_ = segment_seed ? n : 0;
      captures_seen_ = std::max<int64_t>(0, n - 128);
    } else {
      const int64_t old = ds41_spark_committed(spark_.get());
      if (cold_seed_pending_ || old + n > prompt_rows_ ||
          (old + n != prompt_rows_ && !append_restore_))
        throw std::logic_error("invalid cached append capture descriptor");
      if (n > 128) {
        restore = true;
        append_pending_ = true; append_old_ = old; tail_limit_ = old + n;
        captures_seen_ = old + n - 128; tail_rows_ = 0;
      }
    }
    try {
      auto result = target_->prefill(ids, options);
      auto stream = weights_.cuda_stream;
      if (segment_seed) {
        if (captures_seen_ != n || tail_rows_ != std::min<int64_t>(128, n))
          throw std::logic_error("missing segmented cold prompt captures");
        checked(ds41_spark_seed(spark_.get(), prompt_tail_.get(), n));
        cold_seed_pending_ = false; tail_limit_ = 0; captures_seen_ = tail_rows_ = 0;
        seed_mode_ = "segmented-cold+append-restore";
      }
      if (restore) {
        if (captures_seen_ != append_old_ + n || tail_rows_ != 128)
          throw std::logic_error("missing append tail captures");
        ds41_spark_append_tail_v1 d{};
        d.abi_version = 1; d.struct_bytes = sizeof(d);
        d.previous_committed = append_old_; d.committed_end = append_old_ + n;
        d.capture_begin = append_old_ + n - 128; d.rows = 128; d.columns = 15360;
        d.row_stride_bytes = 30720; d.captures_bf16 = prompt_tail_.get(); d.cuda_stream = stream;
        checked(ds41_spark_restore_append_tail_m1(spark_.get(), &d));
        append_pending_ = false; tail_limit_ = append_old_ = 0; captures_seen_ = tail_rows_ = 0;
        ++append_restores_;
      }
      return result;
    } catch (...) {
      poisoned_ = true;
      throw;
    }
  }
  std::vector<int> cached_prefix() const override {
    return target_->cached_prefix();
  }
  std::map<std::string, std::vector<uint8_t>> live_state() override {
    return target_->live_state();
  }
  void commit(size_t count) override {
    check();
    target_->commit(count);
  }
  void rollback(int64_t position) override {
    check();
    checked(ds41_spark_rollback(spark_.get(), position));
#ifdef DS41_SPARK_POLICY_V2
    q0_ready_ = false;
#endif
  }
  uint64_t prefetch_experts(const std::vector<uint32_t> &ids,
                            PrefetchIntent intent) override {
    check();
    return target_->prefetch_experts(ids, intent);
  }
  void cancel_prefetch(uint64_t ticket) override {
    check();
    target_->cancel_prefetch(ticket);
  }
  ds41_target_weights_v2 shared_weights() const override {
    return target_->shared_weights();
  }
  ds41_target_services_v1 services() override { return target_->services(); }
  ds41_target_tensor_v1
  resident_tensor(const std::string &name) const override {
    return target_->resident_tensor(name);
  }
  std::vector<std::string>
  resident_names(const std::string &name) const override {
    return target_->resident_names(name);
  }
  ds41_target_capture_v1 captures() const override {
    return target_->captures();
  }
  void set_capture_sink(ds41_target_capture_sink sink, void *user) override {
    extra_sink_ = sink;
    extra_user_ = user;
  }
  DraftProvider *drafter() override { return this; }
  DemandCounters demand_counters() const override {
    return target_->demand_counters();
  }
  Json counters() const override {
    auto out = target_->counters();
    out["spark_native_candidate"] = true;
    out["spark_committed_inputs"] = ds41_spark_committed(spark_.get());
    out["spark_fixed_width"] = fixed_width_;
    out["spark_sampling_path"] = gpu_verify_on() ? "device-fp64-v2-UNQUALIFIED" : "host-fp64-v2-UNQUALIFIED";
    if (gpu_verify_on()) {   // s0ma change 29
      out["spark_gpu_verify_steps"] = gpu_verify_steps_;
      out["spark_gpu_verify_fallbacks"] = gpu_verify_fallbacks_;
    }
    if (gpu_verify_mode() == 2) out["spark_gpu_verify_mismatches"] = gpu_verify_mismatches_;
    out["spark_seed_mode"] = seed_mode_;
    out["spark_append_restores"] = append_restores_;
    out["spark_draft_prior_scope"] =
        "owner-resident-context128-fp32-softmax-gpu-v2-excludes-target-offload";
    out["spark_policy_draft_ms"] =
        std::vector<double>(config_.draft_ms, config_.draft_ms + 6);
#ifdef DS41_SPARK_POLICY_V2
    if (policy_v2_)
      out["spark_policy_v2"] = {
          {"enabled", true}, {"performance_default", false},
          {"mode", DS41_SPARK_MODE_UNKNOWN}, {"generation", policy2_state_.generation},
          {"completed", policy2_state_.completed}, {"attempts", policy2_attempts_},
          {"width_history", width_history_},
          {"history_truncated", policy2_attempts_ > width_history_.size()},
          {"verify_ms_priors", std::vector<double>(policy2_config_.verify_ms, policy2_config_.verify_ms + 6)},
          {"draft_ms_priors", std::vector<double>(policy2_config_.draft_ms, policy2_config_.draft_ms + 6)},
          {"other_ms_priors", std::vector<double>(policy2_config_.other_ms, policy2_config_.other_ms + 6)}};
#endif
    return out;
  }
  int prepare(const DraftContext &context) override {
    check();
    if (context.committed_inputs != ds41_spark_committed(spark_.get()))
      throw std::logic_error("target and Spark committed inputs differ");
#ifdef DS41_SPARK_POLICY_V2
    if (policy_v2_ && !fixed_width_) {
      if (policy2_pending_) throw std::logic_error("overlapping policy attempts");
      policy2_features_ = {};
      auto &f = policy2_features_;
      f.size = sizeof(f); f.version = DS41_SPARK_POLICY_VERSION;
      f.nonce = context.nonce; f.committed_inputs = context.committed_inputs;
      f.known_anchor = context.anchor;
      f.remaining_tokens = f.max_width = context.maximum_width;
      if (ds41_spark_committed(spark_.get()) + 5 > weights_.rope_rows)
        f.max_width = 1;
      f.qualified_mask = 63;
      f.mode = DS41_SPARK_MODE_UNKNOWN; f.token_kind = DS41_SPARK_KIND_UNKNOWN;
      f.temperature = context.temperature; f.top_p = context.top_p;
      policy_checked(ds41_spark_policy_decide_v2(&policy2_state_, &policy2_config_,
                       &f, &policy2_decision_), "preflight");
      policy2_pending_ = true; q0_ready_ = false;
      ++policy2_attempts_;
      if (width_history_.size() < 4096) width_history_.push_back(policy2_decision_.width);
      return policy2_decision_.width;
    }
#endif
    return choose_width(context.maximum_width);
  }
  int choose_width(int maximum) override {
    check();
    // The drafter always evaluates five positions, even for a short verify.
    if (ds41_spark_committed(spark_.get()) + 5 > weights_.rope_rows)
      return 1;
    return fixed_width_ ? std::min(maximum, fixed_width_)
                        : ds41_spark_policy_choose_ex(&policy_, &config_, 16, 3,
                                                      maximum);
  }
  void begin(int anchor) override {
    check();
    if (prompt_active_ || incomplete_prompt_)
      throw std::logic_error("cannot draft before Spark prompt completion");
    const float *u = nullptr;
    const void *hd = nullptr;
#ifdef DS41_SPARK_POLICY_V2
    if (policy2_pending_ && anchor != policy2_features_.known_anchor)
      throw std::logic_error("policy anchor changed before forward");
    const double started = clock_ms();
#endif
    checked(ds41_spark_forward(spark_.get(), anchor, &u, &hd));
#ifdef DS41_SPARK_POLICY_V2
    if (policy2_pending_) {
      const float *data = nullptr, *confidence = nullptr;
      checked(ds41_spark_markov(spark_.get(), 0, anchor, &data, &confidence));
      q0_.resize(vocab()); float raw = 0;
      auto stream = static_cast<cudaStream_t>(weights_.cuda_stream);
      cuda_checked(cudaMemcpyAsync(q0_.data(), data, q0_.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
      cuda_checked(cudaMemcpyAsync(&raw, confidence, sizeof(raw),
                                   cudaMemcpyDeviceToHost, stream));
      cuda_checked(cudaStreamSynchronize(stream));
      policy2_features_.confidence_available = 1;
      policy2_features_.confidence_raw = raw;
      policy2_features_.paid_draft_ms = clock_ms() - started;
      q0_ready_ = true;
    }
#endif
  }
  int finalize_width(int preflight) override {
#ifdef DS41_SPARK_POLICY_V2
    if (policy2_pending_) {
      if (!q0_ready_) throw std::logic_error("final policy admission before q0");
      policy_checked(ds41_spark_policy_decide_v2(&policy2_state_, &policy2_config_,
                       &policy2_features_, &policy2_decision_), "finalize");
      if (policy2_attempts_ <= width_history_.size())
        width_history_.back() = policy2_decision_.width;
      return policy2_decision_.width;
    }
#endif
    return preflight;
  }
  bool draw_gpu(int k, int previous, double temperature, double top_p, uint64_t seed, uint64_t step,
                uint32_t slot, uint32_t purpose, DraftDraw &out) override {
    static const bool on = [] { const char *v = std::getenv("DS41_SPARK_GPU_DRAW"); return v && v[0] == '1'; }();
    if (!on || top_p != 1.0 || !(temperature >= 0)) return false;
#ifdef DS41_SPARK_POLICY_V2
    if (policy2_pending_) return false;   // q0 was already copied for the policy; keep its host path
#endif
    check();
    const size_t n = size_t(vocab());
    if (k < 0 || k > 4) return false;
    if (!draw_q_dev_) {
      cuda_checked(cudaMalloc(reinterpret_cast<void **>(&draw_q_dev_), (gpu_verify_on() ? 5 : 1) * n * sizeof(double)));
      cuda_checked(cudaMalloc(&draw_scratch_, ds41_draft_draw_scratch_bytes()));
      cuda_checked(cudaMallocHost(reinterpret_cast<void **>(&draw_q_host_), n * sizeof(double)));
      cuda_checked(cudaMallocHost(&draw_result_host_, 64));
    }
    const float *data = nullptr, *confidence = nullptr;
    checked(ds41_spark_markov(spark_.get(), k, previous, &data, &confidence));
    auto stream = static_cast<cudaStream_t>(weights_.cuda_stream);
    // probabilities() floors a positive temperature at 1e-5; zero is greedy
    const double t = temperature > 0 ? std::max(temperature, 1e-5) : 0.0;
    void *result = nullptr;
    double *q_dev = draw_q_dev_ + (gpu_verify_on() ? size_t(k) * n : 0);
    cuda_checked(cudaError_t(ds41_draft_draw_gpu(data, int(n), t, seed, step, slot, purpose, q_dev,
                                                 draw_scratch_, &result, stream)));
    cuda_checked(cudaMemcpyAsync(draw_result_host_, result, 24, cudaMemcpyDeviceToHost, stream));
    cuda_checked(cudaMemcpyAsync(confidence_host_.get(), confidence, sizeof(float), cudaMemcpyDeviceToHost, stream));
    if (t > 0 && !gpu_verify_on())
      cuda_checked(cudaMemcpyAsync(draw_q_host_, draw_q_dev_, n * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_checked(cudaStreamSynchronize(stream));
    const auto *r = static_cast<const char *>(draw_result_host_);
    int32_t token;
    double c;
    std::memcpy(&token, r, 4);
    std::memcpy(&c, r + 8, 8);
    if (token < 0 || size_t(token) >= n || !std::isfinite(c))
      throw std::runtime_error(token == -2 ? "draft logits not finite" : "GPU draft draw found no token");
    last_confidence_ = *confidence_host_;
    out.token = token;
    out.confidence = c;
    out.head = last_confidence_;
    out.q = t > 0 && !gpu_verify_on() ? draw_q_host_ : nullptr;
    out.q_device = t > 0 && gpu_verify_on() ? q_dev : nullptr;
    return true;
  }
  // s0ma change 29: speculative acceptance on the device; false (and the host path) whenever it cannot or the
  // device reports what the host would reject, so the host's own checks report it
  bool verify_gpu(const std::vector<int> &proposals, const std::vector<const double *> &q, double temperature,
                  uint64_t seed, uint64_t step, std::vector<int> &emitted, size_t &accepted,
                  bool &rejected) override {
    if (!gpu_verify_on() || !(temperature > 0) || proposals.size() > 5 || q.size() != proposals.size()) return false;
    check();
    int rows = 0;
    const float *logits = target_->device_logits(rows);
    const size_t n = size_t(vocab());
    if (!logits || rows != int(proposals.size()) + 1) { ++gpu_verify_fallbacks_; return false; }
    if (!verify_p_dev_) {
      cuda_checked(cudaMalloc(reinterpret_cast<void **>(&verify_p_dev_), 6 * n * sizeof(double)));
      cuda_checked(cudaMalloc(&verify_scratch_, ds41_verify_draw_scratch_bytes()));
      cuda_checked(cudaMallocHost(&verify_result_host_, ds41_verify_draw_result_bytes()));
    }
    std::vector<int32_t> d(proposals.begin(), proposals.end());
    auto stream = static_cast<cudaStream_t>(weights_.cuda_stream);
    void *result = nullptr;
    cuda_checked(cudaError_t(ds41_verify_draw_gpu(logits, int(n), int(d.size()), q.data(), d.data(), temperature,
                                                  seed, step, verify_p_dev_, verify_scratch_, &result, stream)));
    cuda_checked(cudaMemcpyAsync(verify_result_host_, result, ds41_verify_draw_result_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
    cuda_checked(cudaStreamSynchronize(stream));
    int32_t r[10];   // tokens[6], emitted, accepted, rejected, error
    std::memcpy(r, verify_result_host_, sizeof(r));
    if (r[9] || r[6] < 1 || r[6] > int32_t(proposals.size()) + 1 || r[7] < 0 || r[7] > r[6]) {
      ++gpu_verify_fallbacks_;
      return false;
    }
    for (int i = 0; i < r[6]; ++i)
      if (r[i] < 0 || size_t(r[i]) >= n) { ++gpu_verify_fallbacks_; return false; }
    emitted.assign(r, r + r[6]);
    accepted = size_t(r[7]);
    rejected = r[8] != 0;
    ++gpu_verify_steps_;
    if (gpu_verify_mode() == 2) {   // the host's acceptance on the same inputs; its result is the one used
      std::vector<std::vector<float>> x(static_cast<size_t>(rows), std::vector<float>(n));
      for (int i = 0; i < rows; ++i)
        cuda_checked(cudaMemcpyAsync(x[i].data(), logits + size_t(i) * n, n * 4, cudaMemcpyDeviceToHost, stream));
      std::vector<Distribution> qh(q.size());
      for (size_t k = 0; k < q.size(); ++k) fetch_q(q[k], qh[k]);
      cuda_checked(cudaStreamSynchronize(stream));
      const auto host = reject_sample(proposals, qh, probabilities_rows(x, temperature, 1.0), Rng{seed}, step,
                                      temperature);
      if (host.emitted != emitted || host.accepted != accepted || host.rejected != rejected) {
        ++gpu_verify_mismatches_;
        emitted = host.emitted;
        accepted = host.accepted;
        rejected = host.rejected;
      }
    }
    return true;
  }
  void fetch_q(const double *q_device, std::vector<double> &out) override {
    const size_t n = size_t(vocab());
    out.resize(n);
    auto stream = static_cast<cudaStream_t>(weights_.cuda_stream);
    cuda_checked(cudaMemcpyAsync(out.data(), q_device, n * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_checked(cudaStreamSynchronize(stream));
  }
  const float *device_logits(int &rows) const override { return target_->device_logits(rows); }
  std::vector<float> logits(int k, int previous) override {
    check();
#ifdef DS41_SPARK_POLICY_V2
    if (policy2_pending_ && k == 0) {
      if (!q0_ready_ || previous != policy2_features_.known_anchor)
        throw std::logic_error("first proposal differs from admitted q0");
      return q0_;
    }
#endif
    const float *data = nullptr, *confidence = nullptr;
    checked(ds41_spark_markov(spark_.get(), k, previous, &data, &confidence));
    std::vector<float> out(vocab());
    auto stream = static_cast<cudaStream_t>(weights_.cuda_stream);
    cuda_checked(cudaMemcpyAsync(out.data(), data, out.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    cuda_checked(cudaMemcpyAsync(confidence_host_.get(), confidence, sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    cuda_checked(cudaStreamSynchronize(stream));
    last_confidence_ = *confidence_host_;
    return out;
  }
  void complete(const DraftObservation &o, bool censored) override {
#ifdef DS41_SPARK_POLICY_V2
    if (policy2_pending_) {
      if (o.nonce != policy2_features_.nonce)
        throw std::logic_error("policy completion nonce mismatch");
      ds41_spark_policy_feedback_v2 f{};
      f.size = sizeof(f); f.version = DS41_SPARK_POLICY_VERSION;
      f.width = o.width; f.accepted_drafts = int(o.accepted_drafts);
      f.tested_drafts = int(o.tested_drafts); f.timing_valid = 1;
      f.verify_ms = o.verify_ms; f.draft_ms = o.draft_ms; f.other_ms = o.other_ms;
      f.host_misses = o.host_misses; f.nvme_misses = o.nvme_misses;
      policy_checked(ds41_spark_policy_observe_v2(&policy2_state_, &policy2_config_,
                       &policy2_decision_, &f), "observe");
      policy2_pending_ = false;
      return;
    }
#endif
    if (!censored) observe(o);
  }
  void abort(uint64_t nonce) override {
#ifdef DS41_SPARK_POLICY_V2
    if (policy2_pending_) {
      if (nonce != policy2_features_.nonce)
        throw std::logic_error("policy abort nonce mismatch");
      policy_checked(ds41_spark_policy_abort_v2(&policy2_state_, nonce), "abort");
      if (policy2_attempts_ <= width_history_.size()) width_history_.back() *= -1;
      policy2_pending_ = q0_ready_ = false;
    }
#else
    (void)nonce;
#endif
  }
  void observe(const DraftObservation &observation) override {
    check();
    const int index = observation.width - 1;
    const bool first = policy_.steps[index] == 0;
    checked(ds41_spark_policy_observe(
        &policy_, &config_, observation.width, int(observation.accepted_drafts),
        observation.verify_ms, observation.host_misses,
        observation.nvme_misses));
    // The transaction is complete before measured cost can affect admission.
    const double alpha = first ? 1.0 : config_.ewma;
    config_.draft_ms[index] +=
        alpha * (observation.draft_ms - config_.draft_ms[index]);
  }
};
} // namespace
std::unique_ptr<Forward> make_spark_target(std::unique_ptr<Forward> target,
                                           int width, bool policy_v2) {
  return std::make_unique<SparkTarget>(std::move(target), width, policy_v2);
}
#else
std::unique_ptr<Forward> make_spark_target(std::unique_ptr<Forward>, int, bool) {
  throw std::runtime_error("unsupported: rebuild with DS41_SPARK_NATIVE=ON for "
                           "the gated drafter candidate");
}
#endif
} // namespace ds41e
