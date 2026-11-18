#include <cstdlib>
#include "ds41/engine.hpp"
#include "ds41/native_store.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <optional>
#include <sys/time.h>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <poll.h>
#include <random>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
namespace ds41e {
namespace {
volatile std::sig_atomic_t stopping = 0;
void stop_signal(int) { stopping = 1; }
double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
int integer(const Json &j, const char *key, int def, int lo, int hi) {
  if (!j.contains(key))
    return def;
  auto v = j.at(key);
  if (!v.is_number_integer())
    throw std::invalid_argument(std::string(key) + " must be an integer");
  auto n = v.get<int64_t>();
  if (n < lo || n > hi)
    throw std::invalid_argument(std::string(key) + " out of bounds");
  return int(n);
}
std::vector<int> token_ids(const Json &j, int vocab) {
  if (!j.is_array() || j.empty() || j.size() > 1048576)
    throw std::invalid_argument("token_ids must be nonempty (max 1M)");
  std::vector<int> ids;
  for (auto id : j) {
    if (!id.is_number_integer())
      throw std::invalid_argument("token id must be an integer");
    auto i = id.get<int64_t>();
    if (i < 0 || i >= vocab)
      throw std::invalid_argument("token id outside vocabulary");
    ids.push_back(int(i));
  }
  return ids;
}
std::vector<uint8_t> base64_decode(const std::string &text) {
  static const auto table = [] {
    std::array<int16_t, 256> t;
    t.fill(-1);
    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[uint8_t(alphabet[i])] = int16_t(i);
    return t;
  }();
  std::vector<uint8_t> out;
  out.reserve(text.size() / 4 * 3);
  uint32_t buffer = 0;
  int bits = 0;
  size_t padding = 0;
  for (char ch : text) {
    if (ch == '=') { ++padding; continue; }
    const int v = table[uint8_t(ch)];
    if (v < 0 || padding) throw std::invalid_argument("image pixels must be base64");
    buffer = (buffer << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(uint8_t(buffer >> bits));
    }
  }
  if (padding > 2 || (text.size() % 4))
    throw std::invalid_argument("image pixels must be padded base64");
  return out;
}
// The prompt's images: geometry and resized RGB pixels from the API (see ImageSpan), checked against the
// token ids. Spans are sorted, disjoint, inside the prompt and hold the image token id at every position.
std::vector<ImageSpan> image_spans(const Json &op, const std::vector<int> &ids) {
  std::vector<ImageSpan> spans;
  if (!op.contains("images") || op.at("images").is_null())
    return spans;
  const auto &list = op.at("images");
  if (!list.is_array() || list.size() > 64)
    throw std::invalid_argument("images must be an array of at most 64 entries");
  int64_t previous_end = 0;
  for (const auto &j : list) {
    if (!j.is_object())
      throw std::invalid_argument("image entries must be objects");
    for (const char *key : {"start", "n_vit_h", "n_vit_w", "n_llm_h", "n_llm_w", "pixels"})
      if (!j.contains(key))
        throw std::invalid_argument(std::string("image entry needs ") + key);
    ImageSpan img;
    if (!j.at("start").is_number_integer())
      throw std::invalid_argument("image start must be an integer");
    img.start = j.at("start").get<int64_t>();
    img.n_vit_h = integer(j, "n_vit_h", 0, 1, 1 << 14);
    img.n_vit_w = integer(j, "n_vit_w", 0, 1, 1 << 14);
    img.n_llm_h = integer(j, "n_llm_h", 0, 1, ImageSpan::max_tokens);
    img.n_llm_w = integer(j, "n_llm_w", 0, 1, ImageSpan::max_tokens);
    if (img.n_llm_h != (img.n_vit_h + 2) / 3 || img.n_llm_w != (img.n_vit_w + 2) / 3 ||
        img.tokens() > ImageSpan::max_tokens)
      throw std::invalid_argument("image grid does not match the vision geometry");
    // Bounds before end(): start + tokens() must not overflow, so start is checked against the prompt first.
    const int64_t prompt = int64_t(ids.size());
    if (img.start < previous_end || img.start > prompt || img.tokens() > prompt - img.start)
      throw std::invalid_argument("image spans must be sorted, disjoint and inside the prompt");
    if (!j.at("pixels").is_string())
      throw std::invalid_argument("image pixels must be a base64 string");
    img.pixels = base64_decode(j.at("pixels").get<std::string>());
    if (img.pixels.size() != size_t(img.n_vit_h) * 14 * img.n_vit_w * 14 * 3)
      throw std::invalid_argument("image pixels must be n_vit_h*14 rows of n_vit_w*14 RGB pixels");
    for (int64_t pos = img.start; pos < img.end(); ++pos)
      if (ids[pos] != ImageSpan::token_id)
        throw std::invalid_argument("image span positions must hold the image token id");
    const std::string geometry = std::to_string(img.n_vit_h) + "x" + std::to_string(img.n_vit_w) + ":";
    img.digest = sha256_bytes(geometry.data(), geometry.size()) + sha256_bytes(img.pixels.data(), img.pixels.size());
    previous_end = img.end();
    spans.push_back(std::move(img));
  }
  return spans;
}
// Prefix-cache identity per position: the token id, or for image positions the image digest and offset, so
// two different images with the same geometry (identical token ids) never share cached state.
std::vector<uint64_t> position_keys(const std::vector<int> &ids, const std::vector<ImageSpan> &images) {
  std::vector<uint64_t> keys(ids.begin(), ids.end());
  for (const auto &img : images) {
    const uint64_t base = std::stoull(img.digest.substr(0, 15), nullptr, 16) ^
                          std::stoull(img.digest.substr(64, 15), nullptr, 16);
    for (int64_t pos = img.start; pos < img.end(); ++pos)
      keys[pos] = (uint64_t(1) << 63) | ((base + uint64_t(pos - img.start) * 0x9E3779B97F4A7C15ull) >> 1);
  }
  return keys;
}
bool send_json(int fd, const Json &event,
               const std::function<bool()> &cancelled = {}) {
  auto raw = event.dump() + "\n";
  size_t pos = 0;
  const bool terminal = event.value("type", std::string()) == "finish";
  double deadline = now() + 2.0;
  while (pos < raw.size()) {
    if (!terminal && (stopping || (cancelled && cancelled()))) {
      if (!pos)
        return false;
      deadline = std::min(deadline, now() + .25);
    }
    if (now() >= deadline) {
      // Never append a finish to an incomplete JSON frame. A stalled peer
      // becomes a transport error, and cannot retain the model lock forever.
      shutdown(fd, SHUT_RDWR);
      return false;
    }
    pollfd p{fd, POLLOUT, 0};
    int ready = poll(&p, 1, 100);
    if (ready < 0 && errno == EINTR)
      continue;
    if (ready < 0 || p.revents & (POLLERR | POLLHUP | POLLNVAL))
      return false;
    if (!ready)
      continue;
    auto n = send(fd, raw.data() + pos, raw.size() - pos, MSG_NOSIGNAL);
    if (n < 0 && (errno == EINTR || errno == EAGAIN))
      continue;
    if (n <= 0)
      return false;
    pos += n;
  }
  return true;
}
double log_normalizer(const std::vector<float> &logits) {
  double mx = logits.at(argmax(logits)), sum = 0;
  for (float value : logits)
    sum += std::exp(double(value) - mx);
  return mx + std::log(sum);
}
std::string logits_digest(const std::vector<float> &logits) {
  static_assert(sizeof(float) == 4, "FP32 logit fingerprint required");
  const uint32_t one = 1;
  if (*reinterpret_cast<const uint8_t *>(&one) == 1)
    return sha256_bytes(logits.data(), logits.size() * sizeof(float));
  std::vector<uint8_t> bytes(logits.size() * 4);
  for (size_t i = 0; i < logits.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &logits[i], 4);
    for (int k = 0; k < 4; ++k)
      bytes[4 * i + k] = uint8_t(bits >> (8 * k));
  }
  return sha256_bytes(bytes.data(), bytes.size());
}
std::vector<int> verify_widths(const Json &op) {
  if (!op.contains("verify_width"))
    return {1};
  auto widths = op.at("verify_width");
  if (!widths.is_array())
    widths = Json::array({widths});
  if (widths.empty() || widths.size() > 64)
    throw std::invalid_argument("verify_width pattern must have 1..64 entries");
  std::vector<int> out;
  for (const auto &width : widths) {
    if (!width.is_number_integer() || width.get<int64_t>() < 1 ||
        width.get<int64_t>() > 6)
      throw std::invalid_argument("verify_width entries must be integers 1..6");
    out.push_back(width.get<int>());
  }
  return out;
}
// Read no more than one bounded frame at a time, preserving pipelined controls.
bool read_line(int fd, std::string &pending, std::string &line) {
  for (;;) {
    auto end = pending.find('\n');
    if (end != std::string::npos) {
      line = pending.substr(0, end);
      pending.erase(0, end + 1);
      return true;
    }
    if (stopping)
      return false;
    if (pending.size() > (256u << 20))
      throw std::invalid_argument("request frame exceeds 256 MiB");
    pollfd p{fd, POLLIN, 0};
    int r = poll(&p, 1, 100);
    if (r < 0 && errno == EINTR)
      continue;
    if (r < 0 || p.revents & (POLLERR | POLLNVAL))
      return false;
    if (!r)
      continue;
    char buf[16384];
    auto n = recv(fd, buf, sizeof(buf), 0);
    if (n < 0 && (errno == EINTR || errno == EAGAIN))
      continue;
    if (n <= 0)
      return false;
    pending.append(buf, n);
  }
}
bool disconnected(int fd) {
  char c;
  int n = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  return n == 0 ||
         (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
}
struct Live {
  std::atomic<bool> cancel{false};
};
class Service {
  std::unique_ptr<Forward> model;
  std::mutex serial, registry_mutex;
  std::map<std::string, std::shared_ptr<Live>> live;
  int context;
  bool prefix_cache, prefix_any = false;
  size_t prefill_dense_batch = 1;
  std::atomic<uint64_t> sequence{0};
  // One live native state. Only consumed inputs are recorded here; a sampled
  // correction/bonus that has not been fed back is never reusable KV.
  struct Prefix {
    std::string conversation;
    std::vector<int> inputs;
    std::vector<uint64_t> keys; // position_keys of inputs: what reuse compares
    std::vector<float> logits;
    std::shared_ptr<Forward::Snapshot> boundary;
    size_t boundary_len = 0;
    std::vector<float> boundary_logits;
    bool same_conversation(const std::string &id, bool any) const {
      return any || (!id.empty() && id == conversation);
    }
    bool matches(const std::string &id, const std::vector<uint64_t> &prompt, bool any) const {
      return same_conversation(id, any) && !keys.empty() &&
             keys.size() <= prompt.size() &&
             std::equal(keys.begin(), keys.end(), prompt.begin());
    }
  } prefix;
  // s0ma change 72: suffix drafting. The committed tokens of the live conversation, indexed by their last 4, 8, 16
  // and 32 tokens (each n-gram's latest end). Before a step, the context's last tokens (the anchor included) are
  // looked up, longest n first; on a match the tokens that followed the earlier occurrence are proposed. Whether a
  // step copies or asks the drafter depends on the copies' running acceptance by match length against the
  // drafter's running accepted drafts a step. Either way the target verifies: the output distribution is exact.
  struct SuffixIndex {
    static constexpr int kN[4] = {32, 16, 8, 4};
    std::vector<int> tokens;
    std::array<std::unordered_map<uint64_t, uint32_t>, 4> last;   // n-gram hash -> end (exclusive) of its latest
    std::array<std::unordered_map<uint64_t, uint32_t>, 4> before;   // and of the one before (runs of one token)
    static uint64_t hash(const int *p, int n) {
      uint64_t h = 0x9E3779B97F4A7C15ull ^ uint64_t(n);
      for (int i = 0; i < n; ++i) { h ^= uint64_t(uint32_t(p[i])) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); h *= 0xBF58476D1CE4E5B9ull; }
      return h;
    }
    void clear() { tokens.clear(); for (auto &m : last) m.clear(); for (auto &m : before) m.clear(); }
    void push(int t) {
      tokens.push_back(t);
      const size_t e = tokens.size();
      for (int i = 0; i < 4; ++i)
        if (e >= size_t(kN[i])) {
          auto &slot = last[size_t(i)][hash(tokens.data() + e - kN[i], kN[i])];
          if (slot) before[size_t(i)][hash(tokens.data() + e - kN[i], kN[i])] = slot;
          slot = uint32_t(e);
        }
    }
    // the indexed tokens become `inputs` (appending when they extend what is indexed, else from scratch)
    void sync(const std::vector<int> &inputs) {
      if (inputs.size() < tokens.size() || !std::equal(tokens.begin(), tokens.end(), inputs.begin())) clear();
      for (size_t i = tokens.size(); i < inputs.size(); ++i) push(inputs[i]);
    }
    // the continuation of the latest earlier occurrence of (tokens + anchor)'s last n tokens, longest n first;
    // returns the n that matched (0: none) and up to `want` tokens in out
    int lookup(int anchor, int want, std::vector<int> &out) const {
      out.clear();
      const size_t e = tokens.size() + 1;
      std::vector<int> tail;
      for (int i = 0; i < 4; ++i) {
        const int n = kN[i];
        if (e < size_t(n) + 1) continue;
        tail.assign(tokens.end() - (n - 1), tokens.end());
        tail.push_back(anchor);
        const uint64_t h = hash(tail.data(), n);
        auto it = last[size_t(i)].find(h);
        if (it == last[size_t(i)].end()) continue;
        size_t end = it->second;   // tokens[end - n, end) == tail, if not a hash collision
        if (end >= tokens.size()) {   // the latest ends at the context's end (a run of one token): the one before
          auto b = before[size_t(i)].find(h);
          if (b == before[size_t(i)].end()) continue;
          end = b->second;
        }
        if (end >= tokens.size() || !std::equal(tail.begin(), tail.end(), tokens.begin() + (end - n))) continue;
        for (size_t j = end; j < tokens.size() && int(out.size()) < want; ++j) out.push_back(tokens[j]);
        return n;
      }
      return 0;
    }
  } suffix;
  // per match length: the copies' acceptance a draft (EWMA, priors below); the drafter's accepted drafts a step
  struct SuffixPolicy {
    double copy[4] = {0.85, 0.75, 0.6, 0.45};
    double spark = 1.8;
    static int bucket(int n) { return n >= 32 ? 0 : n >= 16 ? 1 : n >= 8 ? 2 : 3; }
    double expected(int n, int drafts) const {
      const double a = copy[bucket(n)];
      double e = 0, r = 1;
      for (int k = 0; k < drafts; ++k) { r *= a; e += r; }
      return e;
    }
    void observe_copy(int n, size_t accepted, size_t tested) {
      if (!tested) return;
      double &a = copy[bucket(n)];
      const double w = 1 - std::pow(0.97, double(tested));
      a += w * (double(accepted) / double(tested) - a);
    }
    void observe_spark(size_t accepted) { spark += 0.03 * (double(accepted) - spark); }
    uint64_t declined = 0;   // every 32nd declined match is copied anyway, so a bucket's estimate can recover
  } suffix_policy;
  // s0ma change 73: several conversations. ds41d keeps one live sequence; a request from another conversation (a
  // worker call between two of s0ma's main turns) used to make the next main turn read its whole context again.
  // Now, at the end of every complete prompt, the sequence's whole state goes to host memory
  // (Forward::save_conversation); a request whose prompt extends a saved prompt, and that the live sequence does not
  // serve, loads it and prefills only what is new. The prompt's end is the point the next turn shares even when the
  // earlier answer is encoded differently in the history. Least recently used first out, within
  // DS41_CONVERSATIONS_GIB (default 1 GiB of host memory; 0 turns it off).
  struct Saved {
    std::string conversation;
    std::vector<int> inputs;
    std::vector<uint64_t> keys;
    std::vector<float> logits;
    std::shared_ptr<Forward::Snapshot> state;
    size_t bytes = 0;
    int priority = 0;   // s0ma change 109: the best (lowest) submit priority that saved or used it
    std::chrono::steady_clock::time_point used = std::chrono::steady_clock::now();
  };
  std::list<Saved> saved;
  size_t saved_bytes = 0;
  uint64_t conversation_saves = 0, conversation_loads = 0, conversation_evictions = 0;
  static size_t conversations_budget() {
    static const size_t b = [] {
      const char *v = std::getenv("DS41_CONVERSATIONS_GIB");
      const double g = v && *v ? std::strtod(v, nullptr) : 1.0;
      if (!std::isfinite(g) || g < 0 || g > 64) throw std::invalid_argument("DS41_CONVERSATIONS_GIB must be 0..64");
      return size_t(g * double(size_t(1) << 30));
    }();
    return b;
  }
  // s0ma change 109: an entry that a newer prompt extends goes only within its own conversation (or when either has
  // no id). Lookups still take any conversation's prompt (prefix_any); this only keeps a side call or a subagent whose
  // prompt extends the main turn's from deleting the main turn's save.
  bool supersedes(const std::string &newer, const std::string &older) const {
    return newer == older || (prefix_any && (newer.empty() || older.empty()));
  }
  // s0ma change 109: RAM eviction. Entries unused for DS41_CONVERSATIONS_KEEP_S (default 1800) go first, least
  // recently used first; then the worst priority (the largest number: s0ma's workers) before a better one, least
  // recently used first among equals. The newest entry never goes.
  static std::chrono::seconds conversations_keep() {
    static const auto v = [] {
      const char *e = std::getenv("DS41_CONVERSATIONS_KEEP_S");
      const long x = e && *e ? std::atol(e) : 1800;
      return std::chrono::seconds(std::max(0L, x));
    }();
    return v;
  }
  std::list<Saved>::iterator eviction_victim() {
    const auto stale = std::chrono::steady_clock::now() - conversations_keep();
    auto victim = saved.end();
    for (auto it = std::next(saved.begin()); it != saved.end(); ++it) {
      if (victim == saved.end()) { victim = it; continue; }
      const bool a = it->used < stale, b = victim->used < stale;
      // later in the list is less recently used, so ties go to the later one
      if (a != b ? a : it->priority >= victim->priority) victim = it;
    }
    return victim;
  }
  void save_live(int priority = 0) {
    const size_t budget = conversations_budget();
    if (!budget || prefix.inputs.empty()) return;
    for (auto it = saved.begin(); it != saved.end(); ++it)   // this very prompt is saved already (a retry)
      if ((prefix_any || it->conversation == prefix.conversation) && it->keys == prefix.keys) {
        it->priority = std::min(it->priority, priority);
        it->used = std::chrono::steady_clock::now();
        saved.splice(saved.begin(), saved, it);
        return;
      }
    std::shared_ptr<Forward::Snapshot> state;
    try { state = model->save_conversation(); } catch (const std::exception &e) {
      std::cerr << "conversation save failed: " << e.what() << "\n";
      return;
    }
    if (!state) return;
    Saved e{prefix.conversation, prefix.inputs, prefix.keys, prefix.logits, state, 0, priority};
    if (disk_on()) disk_enqueue(e);   // s0ma change 81 (also when RAM cannot hold it)
    if (remember(std::move(e))) ++conversation_saves;
  }
  // the RAM list's own bookkeeping: an entry in at the front, the prompts it extends out, the oldest out past the budget
  bool remember(Saved e) {
    const size_t budget = conversations_budget();
    e.bytes = e.state->host_bytes + e.inputs.size() * 4 + e.keys.size() * 8 + e.logits.size() * 4;
    if (!budget || e.bytes > budget) return false;
    // an earlier prompt of the same conversation that this one extends is no longer needed
    for (auto it = saved.begin(); it != saved.end();) {
      if (supersedes(e.conversation, it->conversation) && it->keys.size() <= e.keys.size() &&
          std::equal(it->keys.begin(), it->keys.end(), e.keys.begin())) {
        saved_bytes -= it->bytes;
        it = saved.erase(it);
      } else ++it;
    }
    saved_bytes += e.bytes;
    saved.push_front(std::move(e));
    while (saved_bytes > budget && saved.size() > 1) {
      auto victim = eviction_victim();
      saved_bytes -= victim->bytes;
      saved.erase(victim);
      ++conversation_evictions;
    }
    return true;
  }
  // s0ma change 81: the saved conversations on disk too, so a restarted ds41d reads none of a known prompt again.
  // Every RAM save is also written, in the background, to DS41_CONVERSATIONS_DIR (one file a prompt: identity tag,
  // conversation, ids, keys, the prompt's last logits, then the model's own bytes via Forward::write_conversation),
  // written to a temporary name and renamed. A file is used only when its tag equals DS41_CONVERSATIONS_TAG, which
  // run.py builds from everything that shapes the state (the build's files, the bank, the settings), so a rebuilt
  // or reconfigured engine never loads another's state. At start the directory is indexed (headers only); a request
  // that no RAM entry serves as far loads the longest matching file, then keeps it in RAM. A file a newer one extends
  // is deleted; past DS41_CONVERSATIONS_DISK_GIB the least recently used go first. Off unless the tag and a budget are
  // set (run.py sets them for serving; the lab sets the budget to 0).
  struct DiskEntry {
    std::string path, conversation;
    std::vector<uint64_t> keys;   // empty for another tag's file (counted for the budget only)
    uint64_t bytes = 0;
    int64_t used = 0;             // ns: the file's mtime, refreshed on each load
    bool ours = false;
  };
  struct DiskJob { Saved e; std::chrono::steady_clock::time_point at; };
  std::mutex disk_m;
  std::condition_variable disk_cv;
  std::list<DiskEntry> disk;
  std::deque<DiskJob> disk_jobs;
  bool disk_stop = false;
  std::thread disk_writer;
  std::string disk_dir, disk_tag;
  uint64_t disk_budget = 0, disk_writes = 0, disk_loads = 0, disk_deletes = 0, disk_failures = 0, disk_skipped = 0;
  double disk_write_ms = 0, disk_read_ms = 0;
  bool disk_on() const { return disk_budget && !disk_tag.empty(); }
  std::atomic<int> active{0};   // requests holding the engine
  size_t disk_queued_bytes = 0;
  // a 64-bit digest of a byte stream, eight bytes at a time (any chunking gives the same value)
  struct Digest {
    uint64_t h = 0x9E3779B97F4A7C15ull, total = 0;
    uint8_t carry[8];
    int n = 0;
    void word(uint64_t w) { h = (h ^ w) * 0xBF58476D1CE4E5B9ull; h ^= h >> 31; }
    void add(const void *p, size_t len) {
      auto *b = static_cast<const uint8_t *>(p);
      total += len;
      while (len && n) { carry[n++] = *b++; --len; if (n == 8) { uint64_t w; std::memcpy(&w, carry, 8); word(w); n = 0; } }
      for (; len >= 8; b += 8, len -= 8) { uint64_t w; std::memcpy(&w, b, 8); word(w); }
      while (len--) carry[n++] = *b++;
    }
    uint64_t done() {
      uint64_t w = 0;
      std::memcpy(&w, carry, size_t(n));
      word(w ^ (total << 3));
      return h;
    }
  };
  // the writer's sink: every byte digested; every 32 MiB it yields to a request holding the engine, unless the job
  // has waited 60 s (so a busy s0ma still gets its files)
  struct WriteBuf : std::streambuf {
    std::streambuf *out;
    Digest d;
    std::atomic<int> &active;
    std::chrono::steady_clock::time_point due;
    size_t since = 0;
    WriteBuf(std::streambuf *o, std::atomic<int> &a, std::chrono::steady_clock::time_point t) : out(o), active(a), due(t) {}
    std::streamsize xsputn(const char *p, std::streamsize n) override {
      d.add(p, size_t(n));
      since += size_t(n);
      if (since >= (size_t(32) << 20)) {
        since = 0;
        while (active.load() > 0 && std::chrono::steady_clock::now() < due)
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      return out->sputn(p, n);
    }
    int_type overflow(int_type c) override {
      if (traits_type::eq_int_type(c, traits_type::eof())) return traits_type::not_eof(c);
      const char ch = traits_type::to_char_type(c);
      return xsputn(&ch, 1) == 1 ? c : traits_type::eof();
    }
  };
  static void put(std::ostream &out, const void *p, size_t n) { out.write(static_cast<const char *>(p), std::streamsize(n)); }
  static void put64(std::ostream &out, uint64_t v) { put(out, &v, 8); }
  static uint64_t get64(std::istream &in) {
    uint64_t v = 0; in.read(reinterpret_cast<char *>(&v), 8);
    if (!in) throw std::runtime_error("conversation file: truncated");
    return v;
  }
  static std::string get_string(std::istream &in, uint64_t most) {
    const uint64_t n = get64(in);
    if (n > most) throw std::runtime_error("conversation file: header");
    std::string v(n, '\0');
    in.read(v.data(), std::streamsize(n));
    if (!in) throw std::runtime_error("conversation file: truncated");
    return v;
  }
  static int64_t mtime_ns(const std::string &path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) ? 0 : int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
  }
  // the header up to the keys: (tag, conversation, keys); false when not a conversation file
  static bool read_header(std::istream &in, std::string &tag, std::string &conversation, std::vector<int> &inputs,
                          std::vector<uint64_t> &keys) {
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "DS41CV01", 8)) return false;
    tag = get_string(in, 4096);
    conversation = get_string(in, 4096);
    const uint64_t n = get64(in);
    if (n == 0 || n > (uint64_t(1) << 24)) throw std::runtime_error("conversation file: length");
    inputs.resize(n);
    keys.resize(n);
    in.read(reinterpret_cast<char *>(inputs.data()), std::streamsize(n * 4));
    in.read(reinterpret_cast<char *>(keys.data()), std::streamsize(n * 8));
    if (!in) throw std::runtime_error("conversation file: truncated");
    return true;
  }
  void disk_open() {   // never throws: on any trouble the disk cache stays off
    try { disk_open_or_throw(); } catch (const std::exception &e) {
      std::cerr << "conversations on disk: off (" << e.what() << ")\n";
      disk_budget = 0; disk_tag.clear(); disk.clear(); disk_bytes_all = 0;
    }
  }
  void disk_open_or_throw() {
    const char *dir = std::getenv("DS41_CONVERSATIONS_DIR"), *tag = std::getenv("DS41_CONVERSATIONS_TAG"),
               *gib = std::getenv("DS41_CONVERSATIONS_DISK_GIB");
    const double g = gib && *gib ? std::strtod(gib, nullptr) : 0.0;
    if (!std::isfinite(g) || g < 0 || g > 4096) throw std::invalid_argument("DS41_CONVERSATIONS_DISK_GIB must be 0..4096");
    if (!dir || !*dir || !tag || !*tag || !(g > 0) || !prefix_cache) return;
    disk_dir = dir;
    disk_tag = tag;
    disk_budget = uint64_t(g * double(uint64_t(1) << 30));
    std::error_code ec;
    std::filesystem::create_directories(disk_dir, ec);
    ::chmod(disk_dir.c_str(), 0700);   // the files hold s0ma's token ids
    size_t ours = 0;
    for (auto &f : std::filesystem::directory_iterator(disk_dir, ec)) {
      const std::string path = f.path().string();
      if (path.size() > 13 && path.compare(path.size() - 13, 13, ".ds41conv.tmp") == 0) {   // a write cut short
        std::filesystem::remove(f.path(), ec);
        continue;
      }
      if (f.path().extension() != ".ds41conv") continue;
      DiskEntry d;
      d.path = path;
      d.bytes = uint64_t(f.file_size(ec));
      d.used = mtime_ns(path);
      try {
        std::ifstream in(path, std::ios::binary);
        std::string t;
        std::vector<int> inputs;
        if (read_header(in, t, d.conversation, inputs, d.keys) && t == disk_tag) { d.ours = true; ++ours; }
        else d.keys.clear();
      } catch (const std::exception &) { d.keys.clear(); }
      disk.push_back(std::move(d));
    }
    for (auto &d : disk) disk_bytes_all += d.bytes;
    std::cerr << "conversations on disk (" << disk_dir << "): " << ours << " for this engine of " << disk.size()
              << ", " << double(disk_bytes_all) / double(uint64_t(1) << 30) << " GiB of "
              << double(disk_budget) / double(uint64_t(1) << 30) << "\n";
    disk_writer = std::thread([this] { disk_loop(); });
  }
  uint64_t disk_bytes_all = 0;
  // s0ma change 109: conversations ended (conversation_end) while their file may be in the writer's hands: such a file
  // is deleted once written, not indexed
  struct Ended {
    std::string id;
    bool prefix = false;
    bool hit(const std::string &c) const { return prefix ? c.compare(0, id.size(), id) == 0 : c == id; }
  };
  std::vector<Ended> ended;
  // queued states are held beside the RAM list (each about its conversation's KV): at most a budget's worth, and a
  // queued prompt that this one extends is not written at all
  void disk_enqueue(const Saved &e) {
    std::lock_guard<std::mutex> l(disk_m);
    for (auto it = disk_jobs.begin(); it != disk_jobs.end();) {
      const auto &k = it->e.keys;
      if (supersedes(e.conversation, it->e.conversation) && k.size() <= e.keys.size() &&
          std::equal(k.begin(), k.end(), e.keys.begin())) {
        disk_queued_bytes -= it->e.state->host_bytes;
        it = disk_jobs.erase(it);
      } else ++it;
    }
    ended.erase(std::remove_if(ended.begin(), ended.end(), [&](const Ended &x) { return x.hit(e.conversation); }),
                ended.end());   // the conversation goes on (a new one under an ended id)
    const size_t limit = std::max(conversations_budget(), size_t(1) << 30);
    if (disk_jobs.size() >= 4 || disk_queued_bytes + e.state->host_bytes > limit) { ++disk_skipped; return; }
    disk_queued_bytes += e.state->host_bytes;
    disk_jobs.push_back({e, std::chrono::steady_clock::now()});
    disk_cv.notify_one();
  }
  static std::string file_name(const Saved &e) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a over the conversation and the keys
    auto mix = [&](const void *p, size_t n) { auto *b = static_cast<const uint8_t *>(p); for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; } };
    mix(e.conversation.data(), e.conversation.size());
    mix(e.keys.data(), e.keys.size() * 8);
    char name[48];
    std::snprintf(name, sizeof name, "%016llx-%zu.ds41conv", static_cast<unsigned long long>(h), e.keys.size());
    return name;
  }
  void disk_loop() {
    for (;;) {
      DiskJob job;
      {
        std::unique_lock<std::mutex> l(disk_m);
        disk_cv.wait(l, [&] { return disk_stop || !disk_jobs.empty(); });
        if (disk_jobs.empty()) return;
        job = std::move(disk_jobs.front());
        disk_jobs.pop_front();
        disk_queued_bytes -= job.e.state->host_bytes;
      }
      // not while a request runs (its decode reads Engram rows from the drive), unless the job has waited 60 s
      const auto due = job.at + std::chrono::seconds(60);
      while (active.load() > 0 && std::chrono::steady_clock::now() < due && !disk_stop)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      const auto t0 = std::chrono::steady_clock::now();
      const Saved &e = job.e;
      const std::string path = disk_dir + "/" + file_name(e), tmp = path + ".tmp";   // (*.ds41conv.tmp)
      bool ok = false;
      try {
        {
          const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
          if (fd < 0) throw std::runtime_error("open " + tmp);
          ::close(fd);
          std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
          WriteBuf sink(file.rdbuf(), active, job.at + std::chrono::seconds(60));
          std::ostream out(&sink);
          out.write("DS41CV01", 8);
          put64(out, disk_tag.size()); put(out, disk_tag.data(), disk_tag.size());
          put64(out, e.conversation.size()); put(out, e.conversation.data(), e.conversation.size());
          put64(out, e.inputs.size());
          put(out, e.inputs.data(), e.inputs.size() * 4);
          put(out, e.keys.data(), e.keys.size() * 8);
          put64(out, e.logits.size()); put(out, e.logits.data(), e.logits.size() * 4);
          if (!model->write_conversation(*e.state, out)) throw std::runtime_error("this model cannot write its state");
          // the digest of every byte before it, then the end marker (both outside the digest)
          const uint64_t digest = sink.d.done();
          file.write(reinterpret_cast<const char *>(&digest), 8);
          file.write("DS41END1", 8);
          file.flush();
          if (!out || !file) throw std::runtime_error("write " + tmp + " (disk full?)");
        }
        {   // on the drive before the name is: a crash leaves the old file or the new one, never half of it
          const int fd = ::open(tmp.c_str(), O_RDONLY | O_CLOEXEC);
          if (fd < 0 || ::fdatasync(fd)) { if (fd >= 0) ::close(fd); throw std::runtime_error("fdatasync " + tmp); }
          ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // out of the page cache: s0ma's next turn needs RAM more
          ::close(fd);
        }
        std::filesystem::rename(tmp, path);
        ok = true;
      } catch (const std::exception &x) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        std::cerr << "conversation write failed: " << x.what() << "\n";
      }
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::lock_guard<std::mutex> l(disk_m);
      disk_write_ms += ms;
      if (!ok) { ++disk_failures; continue; }
      ++disk_writes;
      std::error_code ec;
      if (std::any_of(ended.begin(), ended.end(), [&](const Ended &x) { return x.hit(e.conversation); })) {
        std::filesystem::remove(path, ec);   // s0ma change 109: its conversation ended while it was written
        ++disk_deletes;
        for (auto it = disk.begin(); it != disk.end(); ++it)
          if (it->path == path) { disk_bytes_all -= it->bytes; disk.erase(it); break; }
        continue;
      }
      // the files this one extends (and an older copy of itself) go
      for (auto it = disk.begin(); it != disk.end();) {
        const bool extended = it->ours && supersedes(e.conversation, it->conversation) &&
            it->keys.size() <= e.keys.size() && std::equal(it->keys.begin(), it->keys.end(), e.keys.begin());
        if (extended || it->path == path) {
          if (it->path != path) { std::filesystem::remove(it->path, ec); ++disk_deletes; }
          disk_bytes_all -= it->bytes;
          it = disk.erase(it);
        } else ++it;
      }
      DiskEntry d{path, e.conversation, e.keys, uint64_t(std::filesystem::file_size(path, ec)), mtime_ns(path), true};
      disk_bytes_all += d.bytes;
      disk.push_back(std::move(d));
      while (disk_bytes_all > disk_budget && disk.size() > 1) {   // least recently used first, never the newest
        auto oldest = std::min_element(disk.begin(), std::prev(disk.end()),
                                       [](const DiskEntry &a, const DiskEntry &b) { return a.used < b.used; });
        std::filesystem::remove(oldest->path, ec);
        ++disk_deletes;
        disk_bytes_all -= oldest->bytes;
        disk.erase(oldest);
      }
    }
  }
  // the longest file of this engine whose keys begin this request's and are longer than `beyond`, read in full
  std::optional<Saved> disk_read(const std::string &conversation, const std::vector<uint64_t> &keys, size_t beyond) {
    std::string path;
    {
      std::lock_guard<std::mutex> l(disk_m);
      const DiskEntry *best = nullptr;
      for (auto &d : disk)
        if (d.ours && (prefix_any || (!conversation.empty() && d.conversation == conversation)) &&
            d.keys.size() > beyond && d.keys.size() <= keys.size() &&
            std::equal(d.keys.begin(), d.keys.end(), keys.begin()) && (!best || d.keys.size() > best->keys.size()))
          best = &d;
      if (!best) return std::nullopt;
      path = best->path;
    }
    const auto t0 = std::chrono::steady_clock::now();
    try {
      {   // the digest first: every byte but the last 16 (the digest and the end marker)
        std::ifstream whole(path, std::ios::binary);
        whole.seekg(0, std::ios::end);
        const int64_t size = int64_t(whole.tellg());
        if (size < 64) throw std::runtime_error("conversation file: too short");
        whole.seekg(0);
        Digest d;
        std::vector<char> buf(size_t(8) << 20);
        for (int64_t left = size - 16; left > 0;) {
          const std::streamsize n = std::streamsize(std::min<int64_t>(left, int64_t(buf.size())));
          whole.read(buf.data(), n);
          if (!whole) throw std::runtime_error("conversation file: truncated");
          d.add(buf.data(), size_t(n));
          left -= n;
        }
        uint64_t stored = 0;
        char end[8];
        whole.read(reinterpret_cast<char *>(&stored), 8);
        whole.read(end, 8);
        if (!whole || std::memcmp(end, "DS41END1", 8)) throw std::runtime_error("conversation file: end marker");
        if (stored != d.done()) throw std::runtime_error("conversation file: digest (damaged)");
      }
      std::ifstream in(path, std::ios::binary);
      Saved e;
      std::string tag;
      if (!read_header(in, tag, e.conversation, e.inputs, e.keys) || tag != disk_tag)
        throw std::runtime_error("conversation file: header");
      const uint64_t v = get64(in);
      if (v != uint64_t(model->vocab())) throw std::runtime_error("conversation file: logits");
      e.logits.resize(v);
      in.read(reinterpret_cast<char *>(e.logits.data()), std::streamsize(v * 4));
      if (!in) throw std::runtime_error("conversation file: truncated");
      e.state = model->read_conversation(in);
      if (!e.state || e.state->position != int64_t(e.inputs.size())) throw std::runtime_error("conversation file: state");
      for (size_t i = 0; i < e.inputs.size(); ++i)   // the keys are the ids but at image positions (their digests)
        if (e.inputs[i] < 0 || e.inputs[i] >= model->vocab()) throw std::runtime_error("conversation file: ids");
      uint64_t digest = 0;
      char end[8];
      in.read(reinterpret_cast<char *>(&digest), 8);
      in.read(end, 8);
      if (!in || std::memcmp(end, "DS41END1", 8)) throw std::runtime_error("conversation file: end marker");
      std::lock_guard<std::mutex> l(disk_m);
      disk_read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ++disk_loads;
      ::utimensat(AT_FDCWD, path.c_str(), nullptr, 0);   // least recently used is by mtime, across restarts too
      for (auto &d : disk) if (d.path == path) d.used = mtime_ns(path);
      return e;
    } catch (const std::exception &x) {
      std::cerr << "conversation file unusable (" << path << "): " << x.what() << "\n";
      std::lock_guard<std::mutex> l(disk_m);
      ++disk_failures;
      std::error_code ec;
      std::filesystem::remove(path, ec);   // damaged or unusable: never tried again
      for (auto it = disk.begin(); it != disk.end(); ++it)
        if (it->path == path) { disk_bytes_all -= it->bytes; disk.erase(it); break; }
      return std::nullopt;
    }
  }
  // the saved prompt that is the longest prefix of this request's (its conversation, or any with prefix_any)
  std::list<Saved>::iterator find_saved(const std::string &conversation, const std::vector<uint64_t> &keys) {
    auto best = saved.end();
    for (auto it = saved.begin(); it != saved.end(); ++it)
      if ((prefix_any || (!conversation.empty() && it->conversation == conversation)) && !it->keys.empty() &&
          it->keys.size() <= keys.size() && std::equal(it->keys.begin(), it->keys.end(), keys.begin()) &&
          (best == saved.end() || it->keys.size() > best->keys.size()))
        best = it;
    return best;
  }
  static bool suffix_on() {
    static const bool on = [] { const char *v = std::getenv("DS41_SUFFIX_DRAFT"); return !v || std::string(v) != "0"; }();
    return on;
  }

public:
  std::atomic<int> connections{0};
  Service(std::unique_ptr<Forward> m, int c, bool reuse, bool any = false, size_t dense_batch = 1)
      : model(std::move(m)), context(c), prefix_cache(reuse), prefix_any(any),
        prefill_dense_batch(dense_batch) {
    disk_open();   // s0ma change 81
    parked.resize(size_t(std::max(1, model->live_slots()) - 1));   // s0ma change 91
    scheduler = std::thread([this] { schedule_loop(); });
  }
  ~Service() {
    { std::lock_guard<std::mutex> l(sched_m); sched_stop = true; }   // s0ma change 91: jobs end first
    sched_cv.notify_all();
    if (scheduler.joinable()) scheduler.join();
    { std::lock_guard<std::mutex> l(disk_m); disk_stop = true; }
    disk_cv.notify_all();
    if (disk_writer.joinable()) disk_writer.join();   // pending writes finish first
  }
  // s0ma change 91: several requests at once. One scheduler thread owns the model; a connection thread parses its
  // request into a Job, queues it and waits until the scheduler has finished it. The scheduler starts (prefills)
  // queued jobs while live slots are free, best priority first (the submit op's `priority`, lowest first: s0ma's main
  // turn asks for a lower number than its workers), and between starts runs decode steps: one job of the best
  // priority alone takes today's single-stream step (drafting, exact); several take one multi-conversation step
  // (each one row, verify_multi). Jobs of a worse priority wait while better ones run, unless DS41_BATCH_LOWER=1.
  // With DS41_LIVE_SLOTS=1 one request runs at a time, as before.
  struct Job {
    int fd = -1;
    std::string id, verb, conversation, draft_mode;
    Json op;
    std::vector<int> ids;
    std::vector<uint64_t> keys;
    std::vector<ImageSpan> images;
    Sampling sampling;
    int priority = 0;
    uint64_t arrival = 0, last_step = 0;
    std::shared_ptr<Live> state;
    bool registered = false, finished = false, done = false, started = false;
    std::mutex m;
    std::condition_variable cv;
    int slot = -1, prompt = 0, completed = 0;
    double t_start = 0, first = 0, last = 0, prefill_end = 0;
    std::vector<float> logits;
    size_t reused = 0, consumed_prompt = 0;
    bool loaded = false, from_file = false;
    Json prefill_counters;
    Rng rng{0};
    std::unique_ptr<AttemptCounter> attempts;
    DraftProvider *drafter = nullptr;
    bool copying = false;
    SuffixIndex suffix;
    std::vector<int> copy;
    uint64_t copy_steps = 0, copy_drafted = 0, copy_accepted = 0;
    std::string reason = "length";
    uint64_t cycles = 0, drafted = 0, accepted = 0, verified_accepted = 0, multi_steps = 0;
    double draft_ms = 0, verify_ms = 0, draft_forward_ms = 0, draft_draw_ms = 0;
    int anchor = -1;
    uint64_t epoch = 0;    // the model's counter epoch at its prefill's end
    uint64_t multi_drafted = 0, multi_accepted = 0;   // s0ma change 93: drafts in multi steps
    double queued = 0;      // s0ma change 96: when it was queued (a probe's gather window)
    int probe = -1;         // a probe (max_tokens 1, prefilled in scratch, no live slot): -1 not decided yet
    double draft_p = -1;   // its first draft's acceptance in multi steps (EWMA; -1: none yet)
    size_t combined = 0;   // s0ma change 91: the prompts its prefill shared (0: alone)
  };
  // the model's slots 1..n-1 (slot 0 is `prefix`, slot0_job and slot0_used)
  struct Parked { Prefix prefix; Job *job = nullptr; uint64_t used = 0; };
  std::vector<Parked> parked;
  Job *slot0_job = nullptr;
  uint64_t slot0_used = 0, use_clock = 0, arrivals = 0;
  uint64_t model_epoch = 0;   // bumped by every begin_request/reset/restore/load (they restart the model's counters)
  std::mutex sched_m;
  std::condition_variable sched_cv;
  std::deque<std::shared_ptr<Job>> queue;
  std::vector<std::shared_ptr<Job>> running;
  std::thread scheduler;
  bool sched_stop = false;
  std::vector<std::function<void()>> controls;   // s0ma change 109: run by the scheduler between steps
  // s0ma change 109: conversation_end {conversation_id | conversation_prefix} drops that conversation's saved states
  // (RAM, the disk writer's queue, its files) and puts its idle live slot first in line for reuse; conversations lists
  // the saved ones (ids, lengths, tiers: never tokens). Both run on the scheduler thread, between steps.
  Json control(const std::string &verb, const Json &op) {
    std::string id;
    bool by_prefix = false;
    if (verb == "conversation_end") {
      const bool exact = op.contains("conversation_id") && op.at("conversation_id").is_string(),
                 pre = op.contains("conversation_prefix") && op.at("conversation_prefix").is_string();
      if (exact == pre) return {{"ok", false}, {"error", "conversation_end takes conversation_id or conversation_prefix"}};
      id = op.at(exact ? "conversation_id" : "conversation_prefix").get<std::string>();
      by_prefix = pre;
      if (id.empty() || id.size() > 4096 || (by_prefix && id.size() < 4))
        return {{"ok", false}, {"error", "conversation id: 1..4096 bytes (a prefix at least 4)"}};
    }
    auto done = std::make_shared<std::promise<Json>>();
    auto result = done->get_future();
    {
      std::lock_guard<std::mutex> l(sched_m);
      if (sched_stop) return {{"ok", false}, {"error", "stopping"}};
      controls.push_back([this, done, verb, id, by_prefix] {
        try {
          done->set_value(verb == "conversations" ? list_conversations() : end_conversation(id, by_prefix));
        } catch (const std::exception &e) {
          done->set_value(Json{{"ok", false}, {"error", e.what()}});
        }
      });
    }
    sched_cv.notify_all();
    if (result.wait_for(std::chrono::minutes(10)) != std::future_status::ready)
      return {{"ok", false}, {"error", "timed out waiting for the scheduler"}};
    return result.get();
  }
  Json end_conversation(const std::string &id, bool by_prefix) {
    const Ended match{id, by_prefix};
    size_t ram = 0, disk_files = 0, queued = 0;
    uint64_t bytes = 0;
    for (auto it = saved.begin(); it != saved.end();)
      if (match.hit(it->conversation)) {
        saved_bytes -= it->bytes;
        bytes += it->bytes;
        ++ram;
        it = saved.erase(it);
      } else ++it;
    size_t slots = 0;
    for (auto &P : parked)
      if (!P.job && !P.prefix.inputs.empty() && match.hit(P.prefix.conversation)) { P.used = 0; ++slots; }
    if (disk_on()) {
      std::lock_guard<std::mutex> l(disk_m);
      for (auto it = disk_jobs.begin(); it != disk_jobs.end();)
        if (match.hit(it->e.conversation)) {
          disk_queued_bytes -= it->e.state->host_bytes;
          ++queued;
          it = disk_jobs.erase(it);
        } else ++it;
      std::error_code ec;
      for (auto it = disk.begin(); it != disk.end();)
        if (it->ours && match.hit(it->conversation)) {
          std::filesystem::remove(it->path, ec);
          ++disk_deletes;
          ++disk_files;
          bytes += it->bytes;
          disk_bytes_all -= it->bytes;
          it = disk.erase(it);
        } else ++it;
      ended.push_back(match);
      if (ended.size() > 256) ended.erase(ended.begin());
    }
    ++conversations_ended;
    return {{"ok", true}, {"dropped_ram", ram}, {"dropped_disk", disk_files}, {"dropped_queued", queued},
            {"idle_slots", slots}, {"bytes", bytes}};
  }
  Json list_conversations() {
    const auto t = std::chrono::steady_clock::now();
    Json ram = Json::array(), files = Json::array();
    for (auto &e : saved)
      ram.push_back({{"conversation_id", e.conversation}, {"tokens", e.keys.size()}, {"bytes", e.bytes},
                     {"priority", e.priority},
                     {"idle_s", std::chrono::duration<double>(t - e.used).count()}});
    if (disk_on()) {
      std::lock_guard<std::mutex> l(disk_m);
      for (auto &d : disk)
        if (d.ours) files.push_back({{"conversation_id", d.conversation}, {"tokens", d.keys.size()}, {"bytes", d.bytes}});
    }
    return {{"ok", true}, {"ram", ram}, {"disk", files}, {"ram_bytes", saved_bytes},
            {"ram_budget", conversations_budget()}};
  }
  uint64_t conversations_ended = 0;
  static bool batch_lower() {
    static const bool v = [] { const char *e = std::getenv("DS41_BATCH_LOWER"); return e && e[0] == '1'; }();
    return v;
  }
  static bool multi_off() {   // DS41_BATCH_MULTI=0: several jobs take turns at single steps instead
    static const bool v = [] { const char *e = std::getenv("DS41_BATCH_MULTI"); return e && e[0] == '0'; }();
    return v;
  }
  bool cancelled(Job &J) { return stopping || (J.state && J.state->cancel.load()) || disconnected(J.fd); }
  Prefix &entry(int k) { return k == 0 ? prefix : parked[size_t(k - 1)].prefix; }
  Job *&holder(int k) { return k == 0 ? slot0_job : parked[size_t(k - 1)].job; }
  uint64_t &used(int k) { return k == 0 ? slot0_used : parked[size_t(k - 1)].used; }
  // a conversation the slot-0 content can move to slot k only if it fits the parked arenas
  bool parkable() { return model->committed_position() <= model->slot_context(); }
  // exchange slot 0 and slot k, model and bookkeeping
  void bring(int k) {
    if (!slot0_job && prefix.inputs.empty()) {   // a dropped, failed or scored conversation: not one to park
      ++model_epoch;
      model->reset();
    }
    model->swap_slot(k);
    auto &P = parked[size_t(k - 1)];
    std::swap(prefix, P.prefix);
    std::swap(slot0_job, P.job);
    std::swap(slot0_used, P.used);
    if (slot0_job) slot0_job->slot = 0;
    if (P.job) P.job->slot = k;
  }
  // slot 0's content can be replaced: an idle conversation there is already in RAM (saved at its prompt's end). It is
  // not saved again: with its answer it would extend that save, which remember() would then drop, and the next turn
  // (or the same prompt again) could no longer load it (change 91's first bench: 0 loaded, an 8K prompt prefilled again)
  void drop_slot0() {
    if (slot0_job) throw std::logic_error("drop_slot0: a running job");
    prefix = {};
  }
  void finish(Job &J, std::string reason, Json error = nullptr, Json stats = Json::object()) {
    if (J.finished) return;
    J.finished = true;
    Json usage = {{"prompt_tokens", J.prompt}, {"completion_tokens", J.completed}};
    for (const auto &field : {
             std::pair<const char *, const char *>{"reused_prompt_tokens", "cached_prompt_tokens"},
             {"prefill_ms", "prefill_ms"}, {"generation_ms", "decode_ms"},
             {"drafted", "draft_tokens"}, {"accepted", "accepted_draft_tokens"},
             {"prompt_cache", "prompt_cache"}})   // s0ma change 109
      if (stats.contains(field.first)) usage[field.second] = stats.at(field.first);
    send_json(J.fd, {{"type", "finish"}, {"finish_reason", reason}, {"usage", usage}, {"error", error},
                     {"stats", stats}});
  }
  void release(Job &J) {   // the connection thread may go on (and close)
    { std::lock_guard<std::mutex> l(J.m); J.done = true; }
    J.cv.notify_all();
  }
  void handle(int fd) {
    auto job = std::make_shared<Job>();
    Job &J = *job;
    J.fd = fd;
    std::string pending, line;
    try {
      while (read_line(fd, pending, line)) {
        auto op = Json::parse(line);
        auto verb = op.value("op", std::string());
        if (verb == "cancel") {
          bool found = false;
          {
            std::lock_guard<std::mutex> lock(registry_mutex);
            auto it = live.find(op.value("request_id", std::string()));
            if (it != live.end()) {
              it->second->cancel = true;
              found = true;
            }
          }
          if (!send_json(fd, {{"ok", found}}))
            break;
          continue;
        }
        if (verb == "conversation_end" || verb == "conversations") {   // s0ma change 109
          if (!send_json(fd, control(verb, op))) break;
          continue;
        }
        if (verb == "train_capture" && !train_capture_dir())   // s0ma change 99: lab only
          throw std::invalid_argument("train_capture needs DS41_TRAIN_CAPTURE_DIR (ds41.sh lab drafter capture)");
        if (verb != "submit" && verb != "score" && verb != "train_capture")
          throw std::invalid_argument("unknown op");
        J.verb = verb;
        J.op = op;
        J.ids = token_ids(op.at("token_ids"), model->vocab());
        J.prompt = int(J.ids.size());
        if (J.prompt > context)
          throw std::invalid_argument("prompt exceeds max_context");
        J.images = image_spans(op, J.ids);
        J.keys = position_keys(J.ids, J.images);
        if (!J.images.empty() && (verb == "score" || verb == "train_capture"))
          throw std::invalid_argument("unsupported: score requests with images");
        J.draft_mode = op.value("draft", std::string("auto"));
        if (J.draft_mode != "off" && J.draft_mode != "auto" && J.draft_mode != "on")
          throw std::invalid_argument(
              "unsupported: draft mode must be off/auto/on");
        if (J.draft_mode == "on" && !model->drafter())
          throw std::invalid_argument(
              "unsupported: native drafting is not enabled in this build");
        if (op.contains("conversation_id") &&
            !op.at("conversation_id").is_null()) {
          J.conversation = op.at("conversation_id").get<std::string>();
          if (J.conversation.size() > 4096)
            throw std::invalid_argument("conversation_id exceeds 4096 bytes");
        }
        if (op.contains("priority") && !op.at("priority").is_null()) {
          if (!op.at("priority").is_number_integer()) throw std::invalid_argument("priority must be an integer");
          J.priority = int(std::clamp<int64_t>(op.at("priority").get<int64_t>(), -1000000000, 1000000000));
        }
        Json s = op.value("sampling", Json::object());
        if (!s.is_object())
          throw std::invalid_argument("sampling must be an object");
        Sampling &sampling = J.sampling;
        sampling.temperature = s.value("temperature", 1.0);
        sampling.top_p = s.value("top_p", 1.0);
        if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
            !std::isfinite(sampling.top_p) || sampling.top_p <= 0 ||
            sampling.top_p > 1)
          throw std::invalid_argument("invalid temperature/top_p");
        sampling.max_tokens = integer(s, "max_tokens", 4096, 0, 1048576);
        sampling.logprobs =
            integer(s, "logprobs", 0, 0, std::min(1024, model->vocab()));
        if (s.contains("seed") && !s.at("seed").is_null()) {
          if (!s.at("seed").is_number_integer() ||
              (!s.at("seed").is_number_unsigned() &&
               s.at("seed").get<int64_t>() < 0))
            throw std::invalid_argument("seed must be a uint64");
          sampling.seed = s.at("seed").get<uint64_t>();
        } else {
          std::random_device random;
          sampling.seed = (uint64_t(random()) << 32) ^ random();
        }
        if (s.contains("stop_token_ids")) {
          if (!s.at("stop_token_ids").is_array())
            throw std::invalid_argument("stop_token_ids must be an array");
          for (auto x : s.at("stop_token_ids")) {
            if (!x.is_number_integer() || x.get<int64_t>() < 0 ||
                x.get<int64_t>() >= model->vocab())
              throw std::invalid_argument("invalid stop token");
            sampling.stop.push_back(x.get<int>());
          }
        }
        if (op.contains("request_id") && !op.at("request_id").is_null())
          J.id = op.at("request_id").get<std::string>();
        if (J.id.empty())
          J.id = "ds41-" + std::to_string(++sequence);
        if (J.id.size() > 256)
          throw std::invalid_argument("request_id too long");
        J.state = std::make_shared<Live>();
        {
          std::lock_guard<std::mutex> lock(registry_mutex);
          if (live.size() >= 8 + probes_max())   // s0ma change 96: room for a probe group beside them
            throw std::runtime_error("engine queue full");
          if (!live.emplace(J.id, J.state).second) {
            J.state.reset();
            throw std::invalid_argument("request_id already live");
          }
          J.registered = true;
        }
        {
          std::lock_guard<std::mutex> l(sched_m);
          J.arrival = ++arrivals;
          J.queued = now();
          queue.push_back(job);
        }
        sched_cv.notify_all();
        std::unique_lock<std::mutex> l(J.m);
        J.cv.wait(l, [&] { return J.done; });
        break;
      }
    } catch (const std::exception &e) {
      finish(J, "error", e.what());   // only requests the scheduler never took get here
    }
    if (J.registered) {
      std::lock_guard<std::mutex> lock(registry_mutex);
      live.erase(J.id);
    }
    close(fd);
    --connections;
  }
  void schedule_loop() {
    std::vector<std::function<void()>> todo;
    for (;;) {
      {
        std::unique_lock<std::mutex> l(sched_m);
        sched_cv.wait_for(l, std::chrono::milliseconds(10),
                          [&] { return sched_stop || !queue.empty() || !running.empty() || !controls.empty(); });
        todo.swap(controls);
      }
      for (auto &c : todo) c();   // s0ma change 109: between steps, on the thread that owns the caches
      todo.clear();
      {
        std::lock_guard<std::mutex> l(sched_m);
        if (sched_stop && queue.empty() && running.empty() && controls.empty()) return;
        if (queue.empty() && running.empty()) continue;
      }
      try {
        tick();
      } catch (const std::exception &e) {   // never terminate: the running jobs end with the error
        std::cerr << "scheduler: " << e.what() << "\n";
        auto all = running;
        for (auto &j : all) { fail(*j, e.what()); retire(j); }
      }
    }
  }
  // a job that failed: its slot's state is not reusable (the model poisons itself where it must)
  void fail(Job &J, const std::string &what) {
    if (J.slot >= 0) {
      if (holder(J.slot) == &J) holder(J.slot) = nullptr;
      entry(J.slot) = {};
    }
    // s0ma change 98: the engine's own error message (never request content) goes to the log as well
    std::cerr << "request failed (" << J.verb << ", priority " << J.priority << ", slot " << J.slot
              << (J.probe == 1 ? ", probe" : "") << "): " << what.substr(0, 400) << "\n";
    finish(J, "error", what);
    if (J.started) --active;
    J.started = false;
  }
  void retire(const std::shared_ptr<Job> &job) {
    running.erase(std::remove(running.begin(), running.end(), job), running.end());
    release(*job);
  }
  void tick() {
    std::vector<std::shared_ptr<Job>> waiting;
    {
      std::lock_guard<std::mutex> l(sched_m);
      waiting.assign(queue.begin(), queue.end());
    }
    // queued jobs whose client left or cancelled finish without touching the model
    for (auto &j : waiting)
      if (cancelled(*j)) {
        finish(*j, "cancelled");
        { std::lock_guard<std::mutex> l(sched_m); queue.erase(std::find(queue.begin(), queue.end(), j)); }
        release(*j);
      }
    {
      std::lock_guard<std::mutex> l(sched_m);
      waiting.assign(queue.begin(), queue.end());
    }
    std::stable_sort(waiting.begin(), waiting.end(), [](const auto &a, const auto &b) {
      return a->priority != b->priority ? a->priority < b->priority : a->arrival < b->arrival;
    });
    const int slots = model->live_slots();
    int best = std::numeric_limits<int>::max();
    for (auto &j : running) best = std::min(best, j->priority);
    // s0ma change 96: the best waiting priority's probes, together (and with a waiting prompt of theirs that fits)
    if (!waiting.empty() && !(probe_hold && !running.empty())) {
      const int cls = waiting.front()->priority;
      // s0ma change 97: DS41_PROBE_DEFER=1: probes wait while a conversation of their priority or better decodes
      if ((running.empty() || batch_lower() || cls <= best) && !(probe_defer() && !running.empty() && best <= cls)) {
        std::vector<std::shared_ptr<Job>> probes;
        size_t rows = 0;
        bool other = false;   // a request of that priority that is not a probe waits too (it is not held back)
        double oldest = std::numeric_limits<double>::max(), newest = 0;
        for (auto &j : waiting) {
          if (j->priority != cls) break;
          if (!probe_job(*j)) { other = true; continue; }
          if (probes.size() >= probes_max() || (!probes.empty() && rows + j->ids.size() > combine_rows())) continue;
          rows += j->ids.size();
          oldest = std::min(oldest, j->queued);
          newest = std::max(newest, j->queued);
          probes.push_back(j);
        }
        if (!probes.empty() && !other && probes.size() < probes_max()) {
          // the window: DS41_PROBE_GATHER_MS without a new probe, at most DS41_PROBE_GATHER_MAX_MS after the first
          const double t = now();
          const double left = std::min(probe_gather_ms() - (t - newest) * 1000, probe_gather_max_ms() - (t - oldest) * 1000);
          if (left > 0) {
            bool idle = true;   // nothing of this priority or better runs: wait out the window here
            for (auto &j : running) idle = idle && j->priority > cls;
            if (idle) {
              std::this_thread::sleep_for(std::chrono::microseconds(int64_t(left * 1000) + 1));
              return;
            }
            probes.clear();   // the running ones step meanwhile
          }
        }
        if (!probes.empty()) {
          std::vector<std::shared_ptr<Job>> starters;
          if (combine_on() && model->supports_prefill_multi() && !(slot0_job && !parkable())) {
            int room = slots - int(running.size());
            for (auto &j : waiting) {
              if (j->priority != cls) break;
              if (probe_job(*j)) continue;
              if (probes.size() + starters.size() >= size_t(Forward::kMaxGroups)) break;
              if (room <= 0 || alone(*j) || j->sampling.max_tokens == 0 ||
                  j->ids.size() + 1 > size_t(model->slot_context()) || rows + j->ids.size() > combine_rows()) break;
              rows += j->ids.size();
              starters.push_back(j);
              --room;
            }
          }
          ++probe_batches;
          start_many(starters, probes);
          probe_hold = true;
          // nothing runs yet: the requests that could not join (no room to combine, too long, a score) start now,
          // in their order, rather than behind the next probes
          if (!running.empty()) goto decode;
          {
            std::lock_guard<std::mutex> l(sched_m);
            waiting.assign(queue.begin(), queue.end());
          }
          std::stable_sort(waiting.begin(), waiting.end(), [](const auto &a, const auto &b) {
            return a->priority != b->priority ? a->priority < b->priority : a->arrival < b->arrival;
          });
        }
      }
    }
    // s0ma change 91: two or more prompts that may start now prefill together (one priority; none longer than a
    // parked slot holds; not while slot 0 holds a running conversation too long to park)
    if (combine_on() && model->supports_prefill_multi() && !(slot0_job && !parkable())) {
      std::vector<std::shared_ptr<Job>> starters;
      int room = slots - int(running.size());
      for (auto &j : waiting) {
        if (room <= 0) break;
        if (!running.empty() && !batch_lower() && j->priority > best) break;
        if (!starters.empty() && !batch_lower() && j->priority > starters.front()->priority) break;
        if (probe_job(*j)) continue;   // s0ma change 96: probes wait for their own group
        if (alone(*j) || j->sampling.max_tokens == 0 ||
            j->ids.size() + 1 > size_t(model->slot_context())) break;
        starters.push_back(j);
        --room;
      }
      if (starters.size() >= 2 && start_many(starters)) {
        for (auto &j : running) best = std::min(best, j->priority);
        goto decode;
      }
    }
    for (auto &j : waiting) {
      if (int(running.size()) >= slots) break;
      if (!running.empty() && !batch_lower() && j->priority > best) break;
      if (probe_job(*j)) continue;   // s0ma change 96: probes wait for their own group
      if (alone(*j) && !running.empty()) break;   // a score request resets slot 0: it runs alone
      { std::lock_guard<std::mutex> l(sched_m); queue.erase(std::find(queue.begin(), queue.end(), j)); }
      if (j->verb == "score") {
        run_score(*j);
        release(*j);
        return;
      }
      if (j->verb == "train_capture") {
        run_train_capture(*j);
        release(*j);
        return;
      }
      bool started = false;
      try {
        started = start(*j);
      } catch (const std::exception &e) {
        fail(*j, e.what());
        release(*j);
        continue;
      }
      if (!started) {   // no slot can take it yet: back to the queue, in its place
        std::lock_guard<std::mutex> l(sched_m);
        queue.push_front(j);
        break;
      }
      if (j->finished) { release(*j); continue; }
      running.push_back(j);
      best = std::min(best, j->priority);
    }
  decode:
    if (running.empty()) return;
    {   // running jobs whose client left or cancelled end now, stepping or waiting (their state is consistent)
      auto all = running;
      for (auto &j : all)
        if (cancelled(*j)) { finish_job(*j, "cancelled"); retire(j); }
      if (running.empty()) return;
      best = std::numeric_limits<int>::max();
      for (auto &j : running) best = std::min(best, j->priority);
    }
    std::vector<std::shared_ptr<Job>> turn;
    for (auto &j : running)
      if (batch_lower() || j->priority == best) turn.push_back(j);
    std::stable_sort(turn.begin(), turn.end(), [](const auto &a, const auto &b) { return a->last_step < b->last_step; });
    if (turn.size() > 6) turn.resize(6);
    const bool multi = turn.size() > 1 && !multi_off() && model->supports_verify_multi();
    probe_hold = false;
    if (multi) step_multi(turn);
    else step_single(turn.front());
  }
  // s0ma change 93: which conversations of a multi step draft one token. A step of s conversations and r rows costs
  // M[s] + B[r] - B[s] ms (M: a step of s conversations one row each; B: one conversation's r rows) plus a draft's
  // own cost; a draft yields its acceptance p (each job's, EWMA; its single-stream acceptance before any). The plan
  // takes the drafts (best p first) that raise expected tokens a millisecond. Measured step times replace the model
  // once a (s, r) has 4 of them; every 32nd step tries a (s, r) not measured yet. DS41_MULTI_DRAFT=0: none.
  struct DraftPlan {
    std::vector<double> M{0, 17.2, 24.1, 28.6, 36.7, 45.0, 54.0}, B{0, 17.2, 22.1, 27.7, 32.4, 42.7, 53.7};
    double draft_ms = 1.2;
    std::map<std::pair<size_t, size_t>, std::pair<double, int>> seen;   // (s, r) -> EWMA ms, count
    uint64_t steps = 0;
    bool on = true;
    DraftPlan() {
      const char *v = std::getenv("DS41_MULTI_DRAFT");
      on = !(v && v[0] == '0');
      auto table = [](const char *name, std::vector<double> &t) {
        const char *e = std::getenv(name);
        if (!e || !*e) return;
        std::vector<double> x{0};
        for (const char *p = e; *p;) {
          char *end = nullptr;
          const double d = std::strtod(p, &end);
          if (end == p || !(d > 0)) throw std::invalid_argument(std::string(name) + ": six positive ms, comma separated");
          x.push_back(d);
          p = *end == ',' ? end + 1 : end;
        }
        if (x.size() != 7) throw std::invalid_argument(std::string(name) + ": six positive ms, comma separated");
        t = x;
      };
      table("DS41_MULTI_STEP_MS", M);
      table("DS41_MULTI_ROWS_MS", B);
    }
    double cost(size_t s, size_t r) const {
      auto it = seen.find({s, r});
      if (it != seen.end() && it->second.second >= 4) return it->second.first;
      return M[std::min<size_t>(s, 6)] + B[std::min<size_t>(r, 6)] - B[std::min<size_t>(s, 6)] + double(r - s) * draft_ms;
    }
    static double p_of(const Job &j) {
      if (j.draft_p >= 0) return j.draft_p;
      return j.drafted ? std::min(0.95, double(j.accepted) / double(j.drafted)) : 0.7;
    }
    template <class Ok>
    std::vector<size_t> choose(const std::vector<std::shared_ptr<Job>> &members, Ok ok) {
      std::vector<size_t> order;
      const size_t s = members.size();
      if (!on) return order;
      for (size_t g = 0; g < s; ++g) if (ok(*members[g])) order.push_back(g);
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return p_of(*members[a]) > p_of(*members[b]); });
      const size_t most = std::min(order.size(), size_t(6) - std::min<size_t>(s, 6));
      size_t best = 0;
      double rate = double(s) / cost(s, s), tokens = double(s);
      for (size_t d = 1; d <= most; ++d) {
        tokens += p_of(*members[order[d - 1]]);
        const double r = tokens / cost(s, s + d);
        if (r > rate) { rate = r; best = d; }
      }
      if (++steps % 32 == 0)   // explore: the smallest draft count whose (s, r) is not measured yet
        for (size_t d = 0; d <= most; ++d) {
          auto it = seen.find({s, s + d});
          if (it == seen.end() || it->second.second < 4) { best = d; break; }
        }
      order.resize(best);
      return order;
    }
    void observe(size_t s, size_t r, double ms) {
      auto &e = seen[{s, r}];
      e.first = e.second ? e.first + 0.2 * (ms - e.first) : ms;
      ++e.second;
    }
    void accepted(Job &j, size_t a) { j.draft_p = j.draft_p < 0 ? double(a) : j.draft_p + 0.1 * (double(a) - j.draft_p); }
  } draft_plan;
  // s0ma change 96: probes. A request for one token (max_tokens 1, no images, a prompt up to the model's scratch
  // context) needs no live slot: its token comes from its prefill's last logits. Probes waiting at the best priority
  // prefill together in scratch (Forward::kScratch groups, up to DS41_PROBES_MAX and DS41_PREFILL_COMBINE_ROWS rows),
  // with a waiting prompt of theirs when one fits, and answer and end there: never parked, decoded or saved.
  // DS41_PROBE_GATHER_MS: when the engine is idle (or runs only worse priorities), probes wait until none has arrived
  // for that long (5 ms; at most DS41_PROBE_GATHER_MAX_MS, 4x it, after the first) for the others of their group: s0ma's
  // six reach ds41d a few ms apart, each tokenized in turn by the adapter and the API. A running better-or-equal priority takes a step between two probe prefills.
  // DS41_PROBES=0 (or DS41_PROBE_SAVE=1, which keeps their saves): one-token requests take live slots as before.
  static size_t probes_max() {
    static const size_t v = [] {
      const char *e = std::getenv("DS41_PROBES_MAX");
      const long x = e && *e ? std::atol(e) : 16;
      return size_t(std::clamp(x, 1L, long(Forward::kMaxGroups)));
    }();
    return v;
  }
  static double probe_gather_ms() {
    static const double v = [] {
      const char *e = std::getenv("DS41_PROBE_GATHER_MS");
      const double x = e && *e ? std::atof(e) : 5.0;
      return std::clamp(x, 0.0, 50.0);
    }();
    return v;
  }
  static double probe_gather_max_ms() {
    static const double v = [] {
      const char *e = std::getenv("DS41_PROBE_GATHER_MAX_MS");
      const double x = e && *e ? std::atof(e) : 4 * probe_gather_ms();
      return std::clamp(x, probe_gather_ms(), 200.0);
    }();
    return v;
  }
  static bool probe_defer() {
    static const bool v = [] { const char *e = std::getenv("DS41_PROBE_DEFER"); return e && e[0] == '1'; }();
    return v;
  }
  static bool probe_save() {
    static const bool v = [] { const char *e = std::getenv("DS41_PROBE_SAVE"); return e && e[0] == '1'; }();
    return v;
  }
  // decided once, when the scheduler first sees it: not a probe when an idle slot or a saved conversation already
  // holds part of its prompt (it continues that conversation: its one-by-one path reuses it, as before)
  bool probe_job(Job &J) {
    if (J.probe >= 0) return J.probe == 1;
    J.probe = 0;
    if (probe_save() || !model->supports_scratch_prefill() || J.verb != "submit" || J.sampling.max_tokens != 1 ||
        !J.images.empty() || J.ids.empty() || int64_t(J.ids.size()) > model->scratch_context())
      return false;
    if (prefix_cache) {
      size_t served = 0;
      for (int k = 0; k < model->live_slots(); ++k)
        if (!holder(k)) served = std::max(served, served_by(entry(k), J));
      auto best = find_saved(J.conversation, J.keys);
      if (best != saved.end()) served = std::max(served, best->keys.size());
      if (!served && disk_on()) {   // a file of a saved conversation (matched as disk_read does; not read here)
        std::lock_guard<std::mutex> l(disk_m);
        for (auto &d : disk)
          if (d.ours && (prefix_any || (!J.conversation.empty() && d.conversation == J.conversation)) &&
              !d.keys.empty() && d.keys.size() <= J.keys.size() && std::equal(d.keys.begin(), d.keys.end(), J.keys.begin()))
            served = std::max(served, d.keys.size());
      }
      if (served) return false;
    }
    J.probe = 1;
    return true;
  }
  bool probe_hold = false;   // a probe prefill ran between two steps of running jobs: they step first
  uint64_t probe_batches = 0;
  static bool combine_on() {   // DS41_PREFILL_COMBINE=0: prompts prefill one by one
    static const bool v = [] { const char *e = std::getenv("DS41_PREFILL_COMBINE"); return !(e && e[0] == '0'); }();
    return v;
  }
  static size_t combine_rows() {   // a prompt remainder up to this many rows joins a combined prefill
    static const size_t v = [] {
      const char *e = std::getenv("DS41_PREFILL_COMBINE_ROWS");
      const long x = e && *e ? std::atol(e) : 16384;
      return size_t(std::max(1L, x));
    }();
    return v;
  }
  // s0ma change 91: several jobs start at once. Each takes its slot and its cached prefix in turn (slot 0 each time);
  // the remainders without images prefill together (Forward::prefill_multi); the others, and all of them if the
  // model declines, take the single path; then each concludes in slot 0
  bool start_many(const std::vector<std::shared_ptr<Job>> &js, const std::vector<std::shared_ptr<Job>> &probes = {}) {
    std::vector<std::shared_ptr<Job>> claimed;
    bool any = false;
    const double t0 = now();
    for (auto &j : probes) {   // s0ma change 96: out of the queue; no slot
      { std::lock_guard<std::mutex> l(sched_m); queue.erase(std::find(queue.begin(), queue.end(), j)); }
      j->t_start = t0;
    }
    for (auto &j : js) {
      { std::lock_guard<std::mutex> l(sched_m); queue.erase(std::find(queue.begin(), queue.end(), j)); }
      bool ok = false;
      try {
        ok = claim(*j);
        if (ok) any = true;
        if (ok && (j->finished || !prepare(*j))) { release(*j); continue; }
      } catch (const std::exception &e) {
        any = true;
        fail(*j, e.what());
        release(*j);
        continue;
      }
      if (!ok) {
        std::lock_guard<std::mutex> l(sched_m);
        queue.push_front(j);
        break;
      }
      claimed.push_back(j);
    }
    std::vector<Forward::PrefillGroup> groups;
    std::vector<std::shared_ptr<Job>> grouped, single;
    for (auto &j : claimed) {
      const size_t rest = j->ids.size() - entry(j->slot).inputs.size();
      const size_t cap = j->slot ? size_t(model->slot_context()) : size_t(context);
      if (rest && rest <= combine_rows() && j->ids.size() <= cap && j->images.empty() && !cancelled(*j)) {
        Forward::PrefillGroup g;
        g.slot = j->slot;
        g.ids.assign(j->ids.begin() + std::ptrdiff_t(entry(j->slot).inputs.size()), j->ids.end());
        g.options.mode = entry(j->slot).inputs.empty() ? PrefillMode::Cold : PrefillMode::Append;
        g.options.dense_batch_rows = prefill_dense_batch;
        groups.push_back(std::move(g));
        grouped.push_back(j);
      } else {
        single.push_back(j);
      }
    }
    std::vector<std::shared_ptr<Job>> probing;   // s0ma change 96: scratch groups, after the slots' groups
    for (auto &j : probes) {
      if (cancelled(*j)) { finish(*j, "cancelled"); release(*j); continue; }
      Forward::PrefillGroup g;
      g.slot = Forward::kScratch;
      g.ids = j->ids;
      g.options.mode = PrefillMode::Cold;
      g.options.dense_batch_rows = prefill_dense_batch;
      g.options.warm = false;   // its experts are not the running conversations'
      groups.push_back(std::move(g));
      probing.push_back(j);
    }
    if (groups.size() >= 2 || !probing.empty()) {
      try {
        auto results = model->prefill_multi(groups);
        for (size_t p = 0; p < probing.size(); ++p) {
          Job &J = *probing[p];
          J.logits = std::move(results[grouped.size() + p].logits.back());
          J.combined = groups.size();
        }
        for (size_t g = 0; g < grouped.size(); ++g) {
          Job &J = *grouped[g];
          auto &P = entry(J.slot);
          const size_t from = P.inputs.size();
          P.inputs.insert(P.inputs.end(), J.ids.begin() + std::ptrdiff_t(from), J.ids.end());
          P.keys.insert(P.keys.end(), J.keys.begin() + std::ptrdiff_t(from), J.keys.end());
          P.logits = results[g].logits.back();
          J.logits = std::move(results[g].logits.back());
          J.consumed_prompt = J.ids.size() - J.reused;
          J.combined = grouped.size();
        }
      } catch (const PrefillDeclined &e) {   // nothing changed: one by one
        std::cerr << "combined prefill declined (" << e.what() << "): one by one\n";
        single.insert(single.end(), grouped.begin(), grouped.end());
        grouped.clear();
        // s0ma change 97: the probes again on their own, in halves until they fit (one that does not fit alone goes
        // back to the queue, in its place, as an ordinary request)
        std::vector<std::shared_ptr<Job>> back, done;
        if (!probing.empty()) probe_halves(probing, done, back, groups.size() > probing.size());
        if (!back.empty()) {
          std::lock_guard<std::mutex> l(sched_m);
          for (auto it = back.rbegin(); it != back.rend(); ++it) { (*it)->probe = 0; queue.push_front(*it); }
        }
        probing = std::move(done);
      } catch (const std::exception &e) {
        for (auto &j : grouped) { fail(*j, e.what()); release(*j); }
        for (auto &j : probing) { fail(*j, e.what()); release(*j); }
        grouped.clear();
        probing.clear();
      }
    } else {
      single.insert(single.end(), grouped.begin(), grouped.end());
      grouped.clear();
    }
    auto settle = [&](const std::shared_ptr<Job> &j, bool run_segments) {
      Job &J = *j;
      try {
        if (J.slot != 0) {
          if (slot0_job && !parkable()) throw std::runtime_error("slot 0 holds a conversation too long to park");
          if (!slot0_job && !prefix.inputs.empty() && !parkable()) drop_slot0();
          bring(J.slot);
        }
        if (run_segments) segments(J);
        conclude(J);
      } catch (const std::exception &e) {
        fail(J, e.what());
      }
      if (J.finished) release(J);
      else running.push_back(j);
    };
    for (auto &j : probing) answer_probe(*j);
    for (auto &j : grouped) settle(j, false);
    for (auto &j : single) settle(j, true);
    return any || !probes.empty();
  }
  // s0ma change 97: probes alone in one prefill; declined (the scratch does not fit), each half on its own. `whole`:
  // try them all first (false when they were just declined together). Answered ones go to done (logits set), probes
  // that do not fit even alone to back, and failed ones end here.
  void probe_halves(const std::vector<std::shared_ptr<Job>> &ps, std::vector<std::shared_ptr<Job>> &done,
                    std::vector<std::shared_ptr<Job>> &back, bool whole) {
    if (ps.empty()) return;
    if (whole) {
      std::vector<Forward::PrefillGroup> gs;
      for (auto &j : ps) {
        Forward::PrefillGroup g;
        g.slot = Forward::kScratch;
        g.ids = j->ids;
        g.options.mode = PrefillMode::Cold;
        g.options.dense_batch_rows = prefill_dense_batch;
        g.options.warm = false;
        gs.push_back(std::move(g));
      }
      try {
        auto results = model->prefill_multi(gs);
        for (size_t p = 0; p < ps.size(); ++p) {
          ps[p]->logits = std::move(results[p].logits.back());
          ps[p]->combined = ps.size();
          done.push_back(ps[p]);
        }
        return;
      } catch (const PrefillDeclined &e) {
        std::cerr << "probe prefill of " << ps.size() << " declined (" << e.what() << ")\n";
      } catch (const std::exception &e) {
        for (auto &j : ps) { fail(*j, e.what()); release(*j); }
        return;
      }
    }
    if (ps.size() == 1) { back.push_back(ps[0]); return; }
    const auto mid = ps.begin() + std::ptrdiff_t(ps.size() / 2);
    probe_halves({ps.begin(), mid}, done, back, true);
    probe_halves({mid, ps.end()}, done, back, true);
  }
  // s0ma change 96: a probe's one token from its prefill's logits, as conclude() draws an anchor; then it ends
  void answer_probe(Job &J) {
    J.prefill_end = now();
    J.reused = 0;
    J.consumed_prompt = J.ids.size();
    J.epoch = ~uint64_t(0);   // its counters are shared with whatever else ran: reported as such
    J.started = true;
    ++active;
    try {
      J.prefill_counters = model->counters();
      J.rng = Rng{J.sampling.seed};
      J.attempts = std::make_unique<AttemptCounter>();
      J.drafter = nullptr;
      if (cancelled(J)) {
        finish_job(J, "cancelled");
      } else {
        const uint64_t nonce = J.attempts->take();
        ++J.cycles;
        J.anchor = J.sampling.temperature == 0
                       ? argmax(J.logits)
                       : sample(probabilities(J.logits, J.sampling.temperature, J.sampling.top_p),
                                J.rng, nonce, 0, Purpose::Target);
        emit(J, J.anchor, J.logits);
        finish_job(J, J.reason);
      }
    } catch (const std::exception &e) {
      fail(J, e.what());
    } catch (...) {
      fail(J, "unknown error answering a probe");
    }
    release(J);
  }
  bool emit(Job &J, int token, const std::vector<float> &row) {
    if (cancelled(J)) {
      J.reason = "cancelled";
      return false;
    }
    if (token == model->eos() ||
        std::find(J.sampling.stop.begin(), J.sampling.stop.end(), token) != J.sampling.stop.end()) {
      J.reason = "stop";
      return false;
    }
    Json event = {{"type", "token"}, {"token_id", token}};
    if (J.sampling.logprobs) {
      const double normalizer = log_normalizer(row);
      event["logsumexp"] = normalizer;
      event["logits_sha256"] = logits_digest(row);
      event["logprob"] = double(row[token]) - normalizer;
      event["top_logprobs"] = Json::array();
      for (int t : top_order(row, J.sampling.logprobs))
        event["top_logprobs"].push_back({{"token_id", t}, {"logprob", double(row[t]) - normalizer}});
    }
    if (!send_json(J.fd, event, [&] { return cancelled(J); })) {
      J.reason = "cancelled";
      return false;
    }
    J.last = now();
    if (!J.first) J.first = J.last;
    ++J.completed;
    return true;
  }
  // what of J's prompt a slot's conversation serves (its whole prompt, or its boundary snapshot)
  size_t served_by(const Prefix &P, const Job &J) const {
    if (P.matches(J.conversation, J.keys, prefix_any)) return P.inputs.size();
    if (P.boundary && P.boundary_len && P.same_conversation(J.conversation, prefix_any) &&
        P.boundary_len <= J.ids.size() && P.boundary_len <= P.inputs.size() &&
        std::equal(P.keys.begin(), P.keys.begin() + P.boundary_len, J.keys.begin()))
      return P.boundary_len;
    return 0;
  }
  // the slot a starting job takes: the idle slot whose conversation serves the most of its prompt, else a fresh slot 0
  // (the job's conversation is then in slot 0)
  bool claim(Job &J) {
    if (J.sampling.max_tokens == 0) {   // as before: nothing touched
      finish(J, "length");
      return true;
    }
    const int slots = model->live_slots();
    int k = -1;
    size_t served = 0;
    if (prefix_cache)
      for (int s = 0; s < slots; ++s) {
        if (holder(s)) continue;
        const size_t n = served_by(entry(s), J);
        if (n > served) { served = n; k = s; }
      }
    const bool movable = slot0_job ? parkable() : prefix.inputs.empty() || parkable();
    if (k > 0) {
      if (!movable) {
        if (slot0_job) return false;
        drop_slot0();
      }
      bring(k);
    } else if (k < 0 && (slot0_job || !prefix.inputs.empty())) {
      int v = -1;
      for (int s = 1; s < slots; ++s) {
        if (holder(s)) continue;
        if (entry(s).inputs.empty()) { v = s; break; }
        if (v < 0 || used(s) < used(v)) v = s;
      }
      if (v > 0 && movable) {
        const bool occupied = !entry(v).inputs.empty();
        bring(v);
        if (occupied) drop_slot0();
      } else {
        if (slot0_job) return false;
        drop_slot0();
      }
    }
    if (slot0_job) throw std::logic_error("start: slot 0 is still taken");
    slot0_job = &J;
    J.slot = 0;
    J.started = true;
    ++active;
    slot0_used = ++use_clock;
    J.last_step = use_clock;
    return true;
  }
  bool start(Job &J) {
    if (!claim(J)) return false;
    if (!J.finished) prefill(J);
    return true;
  }
  // today's prompt path (reuse, rewind, a saved conversation, else cold; then the prefill and the first token)
  void prefill(Job &J) {
    if (!prepare(J)) return;
    segments(J);
    conclude(J);
  }
  // the job in slot 0: what of its prompt the slot (or a saved conversation) already holds; false when it has ended
  bool prepare(Job &J) {
    const auto &ids = J.ids;
    const auto &keys = J.keys;
    const auto &sampling = J.sampling;
    const std::string &conversation = J.conversation;
    double started = now();
    J.t_start = started;
    std::vector<float> &logits = J.logits;
    if (sampling.max_tokens == 0) {
      finish_job(J, "length");
      return false;
    }
    size_t reused = prefix_cache && prefix.matches(conversation, keys, prefix_any)
                        ? prefix.inputs.size()
                        : 0;
    bool rewound = false;
    if (!reused && prefix_cache && prefix.boundary && prefix.boundary_len &&
        prefix.same_conversation(conversation, prefix_any) &&
        prefix.boundary_len <= ids.size() && prefix.boundary_len <= prefix.inputs.size() &&
        std::equal(prefix.keys.begin(), prefix.keys.begin() + prefix.boundary_len, keys.begin())) {
      try {
        ++model_epoch;
        model->restore(*prefix.boundary);
        prefix.inputs.resize(prefix.boundary_len);
        prefix.keys.resize(prefix.boundary_len);
        prefix.logits = prefix.boundary_logits;
        reused = prefix.boundary_len;
        rewound = true;
      } catch (const std::exception &) {
        reused = 0;
      }
    }
    bool loaded = false, from_file = false;
    if (!reused && prefix_cache && (!saved.empty() || disk_on())) {   // s0ma change 73 (81: or from disk)
      auto best = find_saved(conversation, keys);
      std::optional<Saved> file;
      if (disk_on()) file = disk_read(conversation, keys, best != saved.end() ? best->keys.size() : 0);
      const Saved *use = file ? &*file : best != saved.end() ? &*best : nullptr;
      if (use) {
        try {
          ++model_epoch;
          model->load_conversation(*use->state);
          prefix = {};
          prefix.conversation = use->conversation;
          prefix.inputs = use->inputs;
          prefix.keys = use->keys;
          prefix.logits = use->logits;
          reused = prefix.inputs.size();
          rewound = loaded = true;
          ++conversation_loads;
          if (file) { from_file = true; file->priority = J.priority; remember(std::move(*file)); }
          else {
            best->priority = std::min(best->priority, J.priority);   // s0ma change 109
            best->used = std::chrono::steady_clock::now();
            saved.splice(saved.begin(), saved, best);
          }
        } catch (const std::exception &e) {
          std::cerr << "conversation load failed: " << e.what() << "\n";
          reused = 0;
        }
      }
    }
    if (reused) {
      // a loaded conversation starts a request too (the drafter's policy-v2 nonces restart with the server's)
      if (!rewound || loaded) { ++model_epoch; model->begin_request(); }
      logits = prefix.logits;
    } else {
      prefix = {};
      ++model_epoch;
      model->reset();
    }
    prefix.conversation = conversation;
    J.reused = reused;
    J.loaded = loaded;
    J.from_file = from_file;
    return true;
  }
  // the rest of the prompt: prefill segments, then single rows
  void segments(Job &J) {
    const auto &ids = J.ids;
    const auto &keys = J.keys;
    const auto &images = J.images;
    const int prompt = J.prompt;
    const size_t reused = J.reused;
    auto cancelled = [&] { return this->cancelled(J); };
    std::vector<float> &logits = J.logits;
    model->begin_prompt(prompt, reused != 0);
    auto consume = [&](int token) {
      auto next = model->verify({token}).at(0);
      model->commit(1);
      prefix.keys.push_back(keys[prefix.inputs.size()]);
      prefix.inputs.push_back(token);
      prefix.logits = next;
      logits = std::move(next);
    };
    // Prefill the suffix in the largest segments the model accepts (one activation arena must fit
    // beside the live decode cache): the first in the prompt's mode, the rest as appends. Prompts or
    // remainders the model cannot prefill use the sequential consume path.
    size_t next = reused;
    while (next < ids.size() && !cancelled()) {
      const PrefillMode mode = next ? PrefillMode::Append : PrefillMode::Cold;
      size_t rows = ids.size() - next;
      if (!model->supports_prefill(mode, rows)) {
        size_t lo = 0, hi = rows;  // largest supported multiple of 256 below rows
        while (hi - lo > 256) {
          size_t mid = (lo + (hi - lo) / 2) / 256 * 256;
          if (mid <= lo) mid = lo + 256;
          if (model->supports_prefill(mode, mid)) lo = mid; else hi = mid;
        }
        rows = lo;
      }
      if (!rows)
        break;
      PrefillOptions options;
      options.mode = mode;
      options.dense_batch_rows = prefill_dense_batch;
      options.warm = next + rows == ids.size();
      options.images = images.empty() ? nullptr : &images;
      if (next + rows < ids.size()) {   // s0ma change 119: the later segments' tokens, for their Engram rows
        options.lookahead = ids.data() + next + rows;
        options.lookahead_count = ids.size() - (next + rows);
      }
      std::vector<int> segment(ids.begin() + next, ids.begin() + next + rows);
      auto result = model->prefill(segment, options);
      if (result.begin_position != int64_t(next) ||
          result.consumed_rows != rows || result.logits.size() != 1 ||
          result.logits.back().size() != size_t(model->vocab()))
        throw std::runtime_error("prefill returned inconsistent committed rows");
      prefix.inputs.insert(prefix.inputs.end(), segment.begin(), segment.end());
      prefix.keys.insert(prefix.keys.end(), keys.begin() + next, keys.begin() + next + rows);
      prefix.logits = result.logits.back();
      logits = std::move(result.logits.back());
      next += rows;
    }
    // The sequential path has no image embeddings: an image the prefill could not take is an error.
    for (const auto &img : images)
      if (img.end() > int64_t(next) && !cancelled())
        throw std::runtime_error("image prompt needs a prefill segment, and it did not fit beside "
                                 "the expert cache");
    for (size_t i = next; i < ids.size(); ++i) {
      if (cancelled())
        break;
      consume(ids[i]);
    }
    J.consumed_prompt = prefix.inputs.size() - reused;
    model->end_prompt(prefix.inputs.size() == ids.size());
  }
  // the job in slot 0 after its prompt: the boundary snapshot, the saved conversation, the first token
  void conclude(Job &J) {
    const auto &ids = J.ids;
    const auto &sampling = J.sampling;
    auto cancelled = [&] { return this->cancelled(J); };
    std::vector<float> &logits = J.logits;
    const bool prompt_complete = prefix.inputs.size() == ids.size();
    if (!prompt_complete) {
      prefix = {};
    } else if (prefix_cache) {
      try {
        prefix.boundary = model->snapshot();
        prefix.boundary_len = prefix.boundary ? prefix.inputs.size() : 0;
        prefix.boundary_logits = prefix.boundary ? prefix.logits : std::vector<float>{};
      } catch (const std::exception &) {
        prefix.boundary.reset(); prefix.boundary_len = 0;
      }
      // s0ma change 73 (96: not a one-token answer's, unless DS41_PROBE_SAVE=1: its prompt is not extended)
      if (!cancelled() && (sampling.max_tokens > 1 || probe_save())) save_live(J.priority);
    }
    J.prefill_end = now();
    J.prefill_counters = model->counters();
    J.epoch = model_epoch;
    J.rng = Rng{sampling.seed};
    J.attempts = std::make_unique<AttemptCounter>();
    J.drafter = J.draft_mode == "off" ? nullptr : model->drafter();
    J.copying = J.drafter && suffix_on();   // s0ma change 72
    if (J.copying) J.suffix.sync(prefix.inputs);
    if (cancelled()) {
      finish_job(J, "cancelled");
      return;
    }
    // Initial target anchor has its own nonce, including greedy/stops.
    const uint64_t nonce = J.attempts->take();
    ++J.cycles;
    J.anchor = sampling.temperature == 0
                   ? argmax(logits)
                   : sample(probabilities(logits, sampling.temperature, sampling.top_p),
                            J.rng, nonce, 0, Purpose::Target);
    if (!emit(J, J.anchor, logits) || J.completed >= sampling.max_tokens ||
        prefix.inputs.size() >= size_t(context))
      finish_job(J, J.reason);
  }
  // one single-stream speculative step of the job in slot 0 (today's decode loop body)
  void decode_step(Job &J) {
    const Sampling &sampling = J.sampling;
    DraftProvider *drafter = J.drafter;
    const int maximum = std::min({6, sampling.max_tokens - J.completed,
                                  context - int(prefix.inputs.size())});
    // Burn the nonce before width selection, including failed attempts.
    const uint64_t nonce = J.attempts->take();
    ++J.cycles;
    int match = 0;
    auto &copy = J.copy;
    const int copy_max = J.copying ? std::min(maximum, drafter->max_width()) : 0;
    if (J.copying && copy_max > 1) {   // s0ma change 72: copy when the copies are expected to beat the drafter
      match = J.suffix.lookup(J.anchor, copy_max - 1, copy);
      if (!match || copy.empty() ||
          (suffix_policy.expected(match, int(copy.size())) < suffix_policy.spark &&
           ++suffix_policy.declined % 32 != 0)) {
        copy.clear();
        match = 0;
      }
    }
    SpeculativeStep step(*model, drafter, J.anchor,
                         int64_t(prefix.inputs.size()), sampling,
                         uint64_t(J.prompt + J.completed), nonce, maximum,
                         match ? &copy : nullptr);   // (copy holds at most copy_max - 1 tokens)
    const auto &result = step.result();
    J.drafted += step.inputs().size() - 1;
    J.verified_accepted += result.accepted;
    J.draft_ms += step.observation().draft_ms;
    J.verify_ms += step.observation().verify_ms;
    J.draft_forward_ms += step.observation().draft_forward_ms;
    J.draft_draw_ms += step.observation().draft_draw_ms;
    size_t delivered = 0;
    bool active = true;
    for (size_t k = 0; k < result.emitted.size(); ++k) {
      if (!emit(J, result.emitted[k], step.rows()[k])) {
        active = false;
        break;
      }
      ++delivered;
    }
    if (cancelled(J) && J.completed < sampling.max_tokens) {
      J.reason = "cancelled";
      active = false;
    }
    const bool terminal = !active || J.completed == sampling.max_tokens;
    const size_t consumed = step.finish(delivered, terminal, J.reason == "cancelled");
    J.accepted += std::min(delivered, result.accepted);
    if (J.copying) {   // s0ma change 72
      const size_t tested = std::min(delivered, result.accepted) +
                            (result.rejected && delivered == result.emitted.size() ? 1 : 0);
      if (step.copied()) {
        ++J.copy_steps; J.copy_drafted += step.inputs().size() - 1; J.copy_accepted += std::min(delivered, result.accepted);
        suffix_policy.observe_copy(match, std::min(delivered, result.accepted), tested);
      } else if (step.inputs().size() > 1) {
        suffix_policy.observe_spark(std::min(delivered, result.accepted));
      }
    }
    prefix.inputs.insert(prefix.inputs.end(), step.inputs().begin(), step.inputs().begin() + consumed);
    if (J.copying)
      for (size_t i = 0; i < consumed; ++i) J.suffix.push(step.inputs()[i]);
    prefix.keys.insert(prefix.keys.end(), step.inputs().begin(), step.inputs().begin() + consumed);
    prefix.logits = step.rows()[consumed - 1];
    J.anchor = result.emitted.back();
    if (!active || J.completed >= sampling.max_tokens || prefix.inputs.size() >= size_t(context))
      finish_job(J, J.reason);
  }
  void J0_step(const std::shared_ptr<Job> &job) {   // the job in slot 0, one single step
    Job &J = *job;
    try {
      J.last_step = slot0_used = ++use_clock;
      decode_step(J);
    } catch (const std::exception &e) {
      fail(J, e.what());
    }
    if (J.finished) retire(job);
  }
  void step_single(const std::shared_ptr<Job> &job) {
    Job &J = *job;
    try {
      if (J.slot != 0) {
        if (slot0_job && !parkable()) {   // a running conversation too long to park holds slot 0: step in place
          step_multi({job}, true);
          return;
        }
        if (!slot0_job && !prefix.inputs.empty() && !parkable()) drop_slot0();
        bring(J.slot);
      }
      J.last_step = slot0_used = ++use_clock;
      decode_step(J);
    } catch (const std::exception &e) {
      fail(J, e.what());
    }
    if (J.finished) retire(job);
  }
  // one multi-conversation step: each job's anchor as one row, in its own slot
  void step_multi(const std::vector<std::shared_ptr<Job>> &turn, bool alone = false) {
    std::vector<Forward::VerifyGroup> groups;
    std::vector<std::shared_ptr<Job>> members;
    for (auto &j : turn) {
      const size_t cap = j->slot == 0 ? size_t(context) : size_t(model->slot_context());
      if (entry(j->slot).inputs.size() + 1 > cap) continue;
      groups.push_back({j->slot, {j->anchor}});
      members.push_back(j);
    }
    if (members.empty() || (members.size() < 2 && !alone)) {   // the rest need slot 0
      auto pick = members.empty() ? turn.front() : *std::find_if(turn.begin(), turn.end(), [&](auto &j) {
        return std::find(members.begin(), members.end(), j) == members.end(); });
      if (members.empty() && alone) {
        // slot 0 holds a running conversation too long to park: step that one (it frees slot 0 when it ends)
        auto holder0 = std::find_if(running.begin(), running.end(), [&](auto &j) { return j.get() == slot0_job; });
        if (holder0 != running.end()) { auto h0 = *holder0; J0_step(h0); }
        return;
      }
      step_single(pick);
      return;
    }
    // s0ma change 93: which conversations draft one token this step (DraftPlan: expected tokens a millisecond)
    const size_t n = members.size();
    std::vector<int> drafts(n, -1);
    std::vector<std::vector<double>> qs(n);
    std::vector<uint64_t> nonces(n);
    for (size_t g = 0; g < n; ++g) nonces[g] = members[g]->attempts->take();
    std::vector<size_t> want = draft_plan.choose(members, [&](const Job &j) {
      const size_t cap = j.slot == 0 ? size_t(context) : size_t(model->slot_context());
      return j.drafter && j.draft_mode != "off" && j.sampling.max_tokens - j.completed >= 2 &&
             entry(j.slot).inputs.size() + 2 <= std::min(cap, size_t(context));
    });
    const double t0 = now();
    std::vector<std::vector<std::vector<float>>> rows;
    std::vector<Verification> result(n);
    try {
      double draft_ms = 0;
      for (size_t g : want) {
        Job &J = *members[g];
        const double d0 = now();
        int token = -1;
        if (J.drafter->draft_one(J.slot, J.anchor, J.sampling.temperature, J.sampling.top_p, J.rng.seed, nonces[g],
                                 token, qs[g])) {
          drafts[g] = token;
          groups[g].tokens.push_back(token);
        }
        J.draft_ms += (now() - d0) * 1000;
        draft_ms += (now() - d0) * 1000;
      }
      const double v0 = now();
      rows = model->verify_multi(groups);
      const double ms = (now() - v0) * 1000;
      for (size_t g = 0; g < n; ++g) {
        Job &J = *members[g];
        ++J.cycles; ++J.multi_steps;
        J.verify_ms += ms / double(n);
        const bool d = drafts[g] >= 0;
        std::vector<int> proposals;
        std::vector<Distribution> q;
        if (d) {
          proposals.push_back(drafts[g]);
          if (J.sampling.temperature != 0) q.push_back(std::move(qs[g]));
        }
        result[g] = reject_sample(proposals, q, probabilities_rows(rows[g], J.sampling.temperature, J.sampling.top_p),
                                  J.rng, nonces[g], J.sampling.temperature);
        if (result[g].emitted.empty()) throw std::runtime_error("multi step: no token");
      }
    } catch (const std::exception &e) {
      for (auto &j : members) { fail(*j, e.what()); retire(j); }
      return;
    }
    // the tokens go out first (a stop or a cancel ends a job's delivered prefix), then every group commits what it
    // delivered (1 + accepted drafts among them, as SpeculativeStep::finish), before any other model call
    std::vector<size_t> delivered(n, 0), commit(n, 1);
    std::vector<uint8_t> active(n, 1);
    for (size_t g = 0; g < n; ++g) {
      Job &J = *members[g];
      for (size_t k = 0; k < result[g].emitted.size(); ++k) {
        if (!emit(J, result[g].emitted[k], rows[g][k])) { active[g] = 0; break; }
        ++delivered[g];
      }
      if (cancelled(J) && J.completed < J.sampling.max_tokens) { J.reason = "cancelled"; active[g] = 0; }
      commit[g] = 1 + std::min(delivered[g], result[g].accepted);
    }
    try {
      for (size_t g = 0; g < n; ++g) model->commit_group(g, commit[g]);
    } catch (const std::exception &e) {
      for (auto &j : members) { fail(*j, e.what()); retire(j); }
      return;
    }
    size_t rows_used = 0, tokens_out = 0;
    for (size_t g = 0; g < n; ++g) { rows_used += groups[g].tokens.size(); tokens_out += delivered[g]; }
    draft_plan.observe(n, rows_used, (now() - t0) * 1000);
    for (size_t g = 0; g < n; ++g) {
      Job &J = *members[g];
      try {
        auto &P = entry(J.slot);
        J.last_step = used(J.slot) = ++use_clock;
        const auto &inputs = groups[g].tokens;
        P.inputs.insert(P.inputs.end(), inputs.begin(), inputs.begin() + std::ptrdiff_t(commit[g]));
        P.keys.insert(P.keys.end(), inputs.begin(), inputs.begin() + std::ptrdiff_t(commit[g]));
        P.logits = rows[g][commit[g] - 1];
        if (J.copying) for (size_t i = 0; i < commit[g]; ++i) J.suffix.push(inputs[i]);
        if (drafts[g] >= 0) {
          ++J.drafted; ++J.multi_drafted;
          J.verified_accepted += result[g].accepted;
          const size_t ok = std::min(delivered[g], result[g].accepted);
          J.accepted += ok; J.multi_accepted += ok;
          if (delivered[g] == result[g].emitted.size()) draft_plan.accepted(J, result[g].accepted);
        }
        J.anchor = result[g].emitted.back();
        if (!active[g] || J.completed >= J.sampling.max_tokens || P.inputs.size() >= size_t(context))
          finish_job(J, J.reason);
      } catch (const std::exception &e) {
        fail(J, e.what());
      }
      if (J.finished) retire(members[g]);
    }
  }
  void finish_job(Job &J, const std::string &reason) {
    J.reason = reason;
    auto stats = model->counters();
    stats["seed"] = J.sampling.seed;
    stats["sampling_profile"] = "spark-host-fp64-temperature-floor-1e-5";
    stats["rng_protocol"] = "spark-v2-monotone-attempt-slot-53bit";
    stats["score_profile"] = "fp64-logsumexp-from-fp32-logits";
    stats["prefill_ms"] = ((J.prefill_end ? J.prefill_end : now()) - J.t_start) * 1000;
    stats["prefill_mode"] = "verify";
    stats["reused_prompt_tokens"] = J.reused;
    stats["prefix_cache_requested"] = prefix_cache;
    stats["prefill_input_tokens"] = J.consumed_prompt;
    const bool same = J.epoch == model_epoch;   // no other request restarted the counters since this one's prefill
    const Json &pc = J.prefill_counters.is_object() ? J.prefill_counters : stats;
    stats["prefill_tier_counters"] = tier_counters(Json::object(), pc, false);
    stats["decode_tier_counters"] = tier_counters(pc, stats, same);
    if (!same) stats["tier_counters_shared"] = true;
    stats["ttft_ms"] = J.first ? (J.first - J.t_start) * 1000 : 0;
    stats["decode_ms"] = J.first ? (J.last - J.first) * 1000 : 0;
    // The public usage rate divides all completion tokens by this span,
    // including the initial anchor. The steady-state metric below instead
    // divides subsequent output intervals by first-to-last elapsed time.
    stats["generation_ms"] = J.first ? (J.last - J.prefill_end) * 1000 : 0;
    stats["first_token_time"] = J.first;
    stats["last_token_time"] = J.last;
    stats["t_first_token"] = J.first;
    stats["t_last_token"] = J.last;
    stats["cycles"] = J.cycles;
    stats["draft_mode"] = J.drafter ? "native-candidate" : "off";
    stats["draft_ms"] = J.draft_ms;
    stats["target_verify_ms"] = J.verify_ms;
    stats["draft_forward_ms"] = J.draft_forward_ms;   // s0ma: the drafter's forward (begin)
    stats["draft_draw_ms"] = J.draft_draw_ms;         // s0ma: Markov heads, q and draws per position
    stats["draft_gpu_draw"] = [] { const char *v = std::getenv("DS41_SPARK_GPU_DRAW"); return v && v[0] == '1'; }();
    stats["decode_tps"] = J.completed > 1 && J.last > J.first ? (J.completed - 1) / (J.last - J.first) : 0;
    stats["drafted"] = J.drafted;
    stats["accepted"] = J.accepted;
    stats["verified_accepted"] = J.verified_accepted;
    stats["acceptance"] = J.drafted ? double(J.accepted) / double(J.drafted) : 0.0;
    stats["conversation_loaded"] = J.loaded;     // s0ma change 73
    stats["conversation_from_disk"] = J.from_file;   // s0ma change 81
    // s0ma change 109: where the reused part of the prompt came from
    stats["prompt_cache"] = !J.reused ? "none" : J.from_file ? "disk" : J.loaded ? "ram" : "live";
    stats["priority"] = J.priority;              // s0ma change 91
    stats["multi_steps"] = J.multi_steps;
    stats["combined_prefill"] = J.combined;
    stats["multi_drafted"] = J.multi_drafted;   // s0ma change 93
    stats["multi_accepted"] = J.multi_accepted;
    stats["live_slot"] = J.slot;
    stats["probe"] = J.probe == 1;   // s0ma change 96
    if (disk_on()) {
      std::lock_guard<std::mutex> l(disk_m);
      size_t ours = 0;
      for (auto &d : disk) ours += d.ours;
      stats["conversations_on_disk"] = ours;
      stats["conversations_disk_bytes"] = disk_bytes_all;
      stats["conversation_disk_writes"] = disk_writes;
      stats["conversation_disk_loads"] = disk_loads;
      stats["conversation_disk_deletes"] = disk_deletes;
      stats["conversation_disk_failures"] = disk_failures;
      stats["conversation_disk_skipped"] = disk_skipped;
      stats["conversation_disk_write_ms"] = disk_write_ms;
      stats["conversation_disk_read_ms"] = disk_read_ms;
    }
    stats["conversations_saved"] = saved.size();
    stats["conversations_saved_bytes"] = saved_bytes;
    stats["conversation_saves"] = conversation_saves;
    stats["conversation_loads"] = conversation_loads;
    stats["conversation_evictions"] = conversation_evictions;
    stats["conversations_ended"] = conversations_ended;
    stats["copy_steps"] = J.copy_steps;          // s0ma change 72: suffix drafting
    stats["copy_drafted"] = J.copy_drafted;
    stats["copy_accepted"] = J.copy_accepted;
    finish(J, reason, nullptr, stats);
    if (J.slot >= 0 && holder(J.slot) == &J) holder(J.slot) = nullptr;
    if (J.started) --active;
    J.started = false;
  }
  // a score request (today's path): it resets slot 0, so it runs only when no job is running
  // s0ma change 99: drafter-training captures (lab only): DS41_TRAIN_CAPTURE_DIR, an existing directory
  static const std::string &train_capture_dir_path() {
    static const std::string v = [] { const char *e = std::getenv("DS41_TRAIN_CAPTURE_DIR"); return e ? std::string(e) : std::string(); }();
    return v;
  }
  static bool train_capture_dir() { return !train_capture_dir_path().empty(); }
  static bool alone(const Job &j) { return j.verb == "score" || j.verb == "train_capture"; }
  void run_train_capture(Job &J) {
    ++active;
    struct Done { std::atomic<int> &a; ~Done() { --a; } } done{active};
    try {
      const std::string name = J.op.value("name", std::string());
      if (name.empty() || name.size() > 64 ||
          !std::all_of(name.begin(), name.end(), [](char c) { return std::isalnum((unsigned char)c) || c == '-' || c == '_'; }))
        throw std::invalid_argument("train_capture: name must be 1..64 of [A-Za-z0-9_-]");
      const int topk = integer(J.op, "topk", 32, 1, 64);
      prefix = {};
      ++model_epoch;
      PrefillOptions options;
      options.dense_batch_rows = prefill_dense_batch;
      Json r = model->train_capture(J.ids, options, train_capture_dir_path() + "/" + name, topk);
      prefix = {};
      ++model_epoch;
      finish(J, "stop", nullptr, r);
    } catch (const std::exception &e) {
      prefix = {};
      ++model_epoch;
      try { model->reset(); } catch (...) {}
      fail(J, e.what());
    }
  }
  void run_score(Job &J) {
    auto cancelled = [&] { return this->cancelled(J); };
    const auto &ids = J.ids;
    const int prompt = J.prompt;
    const Json &op = J.op;
    ++active;
    struct Done { std::atomic<int> &a; ~Done() { --a; } } done{active};
    try {
      prefix = {};
      ++model_epoch;
      model->reset();
      int from = integer(op, "score_from", 0, 0, prompt - 1),
          top = integer(op, "top", 64, 1, std::min(1024, model->vocab()));
      auto mode = op.value("mode", std::string("verify"));
      if (mode != "verify" && mode != "prefill")
        throw std::invalid_argument("unknown score mode");
      auto widths = verify_widths(op);
      if (mode != "verify")
        throw std::invalid_argument(
            "unsupported: cold prefill score is not implemented");
      if (op.value("return_routes", false))
        throw std::invalid_argument(
            "unsupported: route events are not implemented");
      size_t cycle = 0;
      for (int begin = 0; begin < prompt && !cancelled();) {
        int count =
            std::min(widths[cycle++ % widths.size()], prompt - begin);
        std::vector<int> chunk(ids.begin() + begin,
                               ids.begin() + begin + count);
        auto rows = model->verify(chunk);
        if (rows.size() != size_t(count))
          throw std::runtime_error(
              "target verify returned wrong row count");
        model->commit(count);
        bool sent = true;
        for (int row = 0; row < count && !cancelled(); ++row) {
          int i = begin + row;
          if (i < from)
            continue;
          auto &l = rows[row];
          double normalizer = log_normalizer(l);
          auto top_ids = top_order(l, top);
          Json logp = Json::array();
          for (int t : top_ids)
            logp.push_back(double(l[t]) - normalizer);
          Json event = {{"type", "score"},
                        {"position", i},
                        {"top1", argmax(l)},
                        {"top_id", top_ids},
                        {"top_logprob", logp},
                        {"logsumexp", normalizer},
                        {"logits_sha256", logits_digest(l)}};
          if (op.value("return_logits", false))
            event["logits"] = l;
          if (i + 1 < prompt) {
            event["next_token_id"] = ids[i + 1];
            event["next_logprob"] = double(l[ids[i + 1]]) - normalizer;
          }
          if (!send_json(J.fd, event, cancelled)) {
            sent = false;
            break;
          }
        }
        begin += count;
        if (!sent)
          break;
      }
      auto stats = model->counters();
      stats["prefill_mode"] = "verify";
      stats["verify_width"] = widths;
      finish(J, cancelled() ? "cancelled" : "stop", nullptr, stats);
    } catch (const std::exception &e) {
      prefix = {};
      finish(J, "error", e.what());
    }
  }
  template <class T>
  static std::vector<int> top_order(const std::vector<T> &p, int n) {
    std::vector<int> ids(p.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(
        ids.begin(), ids.begin() + n, ids.end(),
        [&](int a, int b) { return p[a] == p[b] ? a < b : p[a] > p[b]; });
    ids.resize(n);
    return ids;
  }
  // s0ma change 91: strict=false when the model's counters were restarted (another request's begin_request or
  // reset) since `before`: a counter that went back is left out instead of failing the request
  static Json tier_counters(const Json &before, const Json &after, bool strict = true) {
    Json out = Json::object();
    for (const char *key :
         {"expert_vram_hits", "expert_host_hits", "expert_nvme_misses",
          "expert_h2d_bytes", "expert_nvme_bytes", "mtp_expert_requests",
          "mtp_expert_vram_hits", "mtp_expert_host_hits", "mtp_expert_nvme_misses",
          "expert_spec_reads", "expert_spec_hits", "expert_spec_waits", "expert_spec_dropped"}) {
      if (!after.contains(key))
        continue;
      uint64_t a = after.at(key).get<uint64_t>(),
               b = before.value(key, uint64_t(0));
      if (a < b) {
        if (!strict) continue;
        throw std::runtime_error("request tier counter moved backwards");
      }
      out[key] = a - b;
    }
    for (auto it = after.begin(); it != after.end(); ++it)   // s0ma change 18: demands by width class
      if (it.key().rfind("expert_class_", 0) == 0 && it->is_number_unsigned()) {
        const uint64_t a = it->get<uint64_t>(), b = before.value(it.key(), uint64_t(0));
        if (a < b) { if (!strict) continue; throw std::runtime_error("request tier counter moved backwards"); }
        out[it.key()] = a - b;
      }
    for (const char *key :
         {"need_host_ms", "route_wait_ms", "engram_read_ms", "forward_ms"})
      if (after.contains(key))
        out[key] = after.at(key).get<double>() - before.value(key, 0.0);
    return out;
  }
};
class Replay final : public Forward, public DraftProvider {
  std::vector<std::vector<float>> rows;
  size_t position = 0, pending = 0;
  int eos_;
  int delay;
  size_t calls = 0;
  // s0ma change 91 (protocol tests): "live_slots": n parks n-1 more positions; verify_multi steps them together
  std::vector<size_t> parked_pos;
  int64_t slot_ctx = 0;
  std::vector<Forward::VerifyGroup> multi_groups;
  std::vector<uint8_t> multi_done;
  size_t multi_calls = 0, swaps = 0;
  bool multi_pending = false, decline_multi = false;
  size_t multi_prefills = 0;

public:
  explicit Replay(const std::string &path) {
    auto j = read_json(path);
    if (j.at("schema") != "ds41-replay-test-v1")
      throw std::invalid_argument("explicit test replay schema required");
    rows = j.at("logits").get<std::vector<std::vector<float>>>();
    if (rows.empty() || rows[0].empty())
      throw std::invalid_argument("empty replay");
    for (auto r : rows)
      if (r.size() != rows[0].size())
        throw std::invalid_argument("replay shape");
    eos_ = j.value("eos", -1);
    delay = j.value("delay_ms", 0);
    parked_pos.assign(size_t(std::max(1, j.value("live_slots", 1)) - 1), 0);
    slot_ctx = j.value("slot_context", 1 << 20);
    decline_multi = j.value("decline_multi", false);
    spark = j.value("spark", false);
    has_drafter = j.value("drafter", false);
    probes = j.value("probes", false);   // (the protocol tests of the one-by-one path keep it)
    conversations = j.value("conversations", false);
    probe_ctx = j.value("probe_context", int64_t(1) << 20);
    probe_fit = j.value("probe_fit", size_t(1) << 20);
  }
  int live_slots() const override { return 1 + int(parked_pos.size()); }
  int64_t slot_context() const override { return parked_pos.empty() ? 0 : slot_ctx; }
  int64_t committed_position() const override { return int64_t(position); }
  void swap_slot(int k) override {
    if (pending || k < 1 || k > int(parked_pos.size())) throw std::logic_error("replay swap_slot");
    // as Native and the Spark bridge refuse
    if (incomplete) throw std::logic_error("swap_slot during a prompt");
    if (int64_t(position) > slot_ctx || int64_t(parked_pos[size_t(k - 1)]) > slot_ctx)
      throw std::logic_error("swap_slot: a conversation longer than DS41_SLOT_CONTEXT");
    std::swap(position, parked_pos[size_t(k - 1)]);
    ++swaps;
  }
  // "spark": true models the drafter bridge's prompt lifecycle: an incomplete prompt blocks swaps until a reset
  bool spark = false, incomplete = false;
  // s0ma change 93 (protocol tests): "drafter": true gives a perfect drafter (the next row's argmax), widths up to 3
  bool has_drafter = false;
  size_t draft_calls = 0;
  DraftProvider *drafter() override { return has_drafter ? this : nullptr; }
  int choose_width(int maximum) override { return std::min(3, maximum); }
  void begin(int) override {}
  std::vector<float> logits(int k, int) override { return rows[(position + size_t(k)) % rows.size()]; }
  void observe(const DraftObservation &) override {}
  void rollback(int64_t) override {}
  bool draft_one(int slot, int, double temperature, double top_p, uint64_t seed, uint64_t step, int &token,
                 std::vector<double> &q) override {
    if (slot < 0 || slot > int(parked_pos.size())) return false;
    const auto &r = rows[(slot ? parked_pos[size_t(slot - 1)] : position) % rows.size()];
    if (temperature == 0) { token = argmax(r); q.clear(); }
    else { q = probabilities(r, temperature, top_p); token = sample(q, Rng{seed}, step, 0, Purpose::Draft); }
    ++draft_calls;
    return true;
  }
  void end_prompt(bool complete) override { if (spark && !complete) incomplete = true; }
  // (s0ma change 109) "conversations": true saves and loads conversations (the position; 1 KiB each)
  bool conversations = false;
  std::shared_ptr<Snapshot> save_conversation() override {
    if (!conversations || pending) return nullptr;
    auto s = std::make_shared<Snapshot>();
    s->position = int64_t(position);
    s->host_bytes = 1024;
    return s;
  }
  void load_conversation(const Snapshot &s) override {
    if (!conversations || pending) throw std::logic_error("replay load_conversation");
    position = size_t(s.position);
  }
  bool supports_verify_multi() const override { return !parked_pos.empty(); }
  std::vector<std::vector<std::vector<float>>> verify_multi(const std::vector<VerifyGroup> &groups) override {
    if (pending || groups.empty()) throw std::logic_error("replay verify_multi");
    if (incomplete) throw std::logic_error("verify_multi during a prompt");
    size_t total = 0;
    std::vector<std::vector<std::vector<float>>> out;
    for (auto &g : groups) {
      if (g.slot < 0 || g.slot > int(parked_pos.size()) || g.tokens.empty()) throw std::logic_error("replay group");
      const size_t at = g.slot ? parked_pos[size_t(g.slot - 1)] : position;
      out.emplace_back();
      for (size_t i = 0; i < g.tokens.size(); ++i) out.back().push_back(rows[(at + i) % rows.size()]);
      total += g.tokens.size();
    }
    if (total > 6) throw std::logic_error("replay verify_multi rows");
    if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    multi_groups = groups;
    multi_done.assign(groups.size(), 0);
    pending = total;
    multi_pending = true;
    ++calls; ++multi_calls;
    return out;
  }
  // (s0ma change 91) the prompts' positions advance together; each gets the row of its last position
  bool supports_prefill_multi() const override { return !parked_pos.empty(); }
  // (s0ma change 96) "probes": true takes scratch groups; a probe's row is that of its prompt's last position, as the
  // one-by-one path gives it
  bool probes = false;
  int64_t probe_ctx = 1 << 20;
  size_t probe_prefills = 0, probe_groups = 0, largest_prefill = 0, probe_fit = 1 << 20;
  bool supports_scratch_prefill() const override { return probes; }
  // (s0ma change 99) a training capture's files: position i's next-token row is rows[i % rows] (as verify gives it),
  // its top-k as log-probabilities, and zero drafter rows
  Json train_capture(const std::vector<int> &ids, const PrefillOptions &, const std::string &out, int topk) override {
    if (pending || ids.empty() || topk < 1 || topk > int(rows.front().size())) throw std::invalid_argument("replay train_capture");
    reset();
    const size_t n = ids.size(), K = size_t(topk), V = rows.front().size();
    std::vector<int32_t> top(n * K);
    std::vector<float> lp(n * K), rest(n);
    for (size_t i = 0; i < n; ++i) {
      const auto &l = rows[i % rows.size()];
      double m = -INFINITY, z = 0;
      for (float v : l) m = std::max(m, double(v));
      for (float v : l) z += std::exp(double(v) - m);
      const double lse = m + std::log(z);
      std::vector<int> o(V);
      std::iota(o.begin(), o.end(), 0);
      std::partial_sort(o.begin(), o.begin() + long(K), o.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
      double mass = 0;
      for (size_t k = 0; k < K; ++k) {
        top[i * K + k] = o[k];
        lp[i * K + k] = float(double(l[o[k]]) - lse);
        mass += std::exp(double(l[o[k]]) - lse);
      }
      rest[i] = float(std::log(std::max(1.0 - mass, 1e-30)));
    }
    FILE *f = std::fopen((out + ".top").c_str(), "wb");
    if (!f) throw std::runtime_error("replay train_capture: cannot write");
    std::fwrite(top.data(), 4, top.size(), f); std::fwrite(lp.data(), 4, lp.size(), f); std::fwrite(rest.data(), 4, n, f);
    std::fclose(f);
    std::vector<uint16_t> kv(n * 3 * 512, 0);
    f = std::fopen((out + ".kv").c_str(), "wb");
    if (!f) throw std::runtime_error("replay train_capture: cannot write");
    std::fwrite(kv.data(), 2, kv.size(), f);
    std::fclose(f);
    reset();
    return {{"rows", n}, {"topk", topk}};
  }
  int64_t scratch_context() const override { return probe_ctx; }
  std::vector<PrefillResult> prefill_multi(const std::vector<PrefillGroup> &groups,
                                           const std::function<void(size_t, bool)> &around) override {
    size_t scratch = 0;
    for (auto &g : groups) scratch += g.slot == kScratch;
    if (pending || groups.empty() || (groups.size() < 2 && !scratch) || groups.size() > size_t(kMaxGroups) ||
        (scratch && !probes) || (scratch < groups.size() && parked_pos.empty() && groups.size() - scratch > 1))
      throw std::logic_error("replay prefill_multi");
    if (incomplete && scratch < groups.size()) throw std::logic_error("prefill_multi during a prompt");
    if (decline_multi) throw PrefillDeclined("replay test: declined");
    if (scratch > probe_fit) throw PrefillDeclined("replay test: the probes' scratch does not fit");
    std::vector<PrefillResult> out;
    for (size_t g = 0; g < groups.size(); ++g) {
      if (groups[g].slot == kScratch) {
        if (groups[g].options.mode != PrefillMode::Cold || groups[g].ids.empty() ||
            int64_t(groups[g].ids.size()) > probe_ctx) throw std::logic_error("replay probe group");
        PrefillResult r;
        r.begin_position = 0;
        r.consumed_rows = groups[g].ids.size();
        r.logits.push_back(rows[(groups[g].ids.size() - 1) % rows.size()]);
        out.push_back(std::move(r));
        continue;
      }
      auto &at = groups[g].slot ? parked_pos.at(size_t(groups[g].slot - 1)) : position;
      if ((groups[g].options.mode == PrefillMode::Cold) != (at == 0)) throw std::logic_error("replay prefill_multi mode");
      PrefillResult r;
      r.begin_position = int64_t(at);
      r.consumed_rows = groups[g].ids.size();
      at += groups[g].ids.size();
      r.logits.push_back(rows[(at - 1) % rows.size()]);
      if (around) { around(g, true); around(g, false); }
      out.push_back(std::move(r));
    }
    if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    if (groups.size() > scratch || groups.size() > 1) ++multi_prefills;
    if (scratch) { ++probe_prefills; probe_groups += scratch; }
    largest_prefill = std::max(largest_prefill, groups.size());
    return out;
  }
  void commit_group(size_t g, size_t n) override {
    if (g >= multi_groups.size() || multi_done[g] || n > multi_groups[g].tokens.size())
      throw std::logic_error("replay commit_group");
    multi_done[g] = 1;
    const int k = multi_groups[g].slot;
    (k ? parked_pos[size_t(k - 1)] : position) += n;
    if (std::all_of(multi_done.begin(), multi_done.end(), [](uint8_t d) { return d != 0; })) { pending = 0; multi_pending = false; }
  }
  int vocab() const override { return int(rows[0].size()); }
  int eos() const override { return eos_; }
  void begin_request() override { calls = 0; }
  void reset() override {
    begin_request();
    position = 0;
    pending = 0;
    incomplete = false;
  }
  std::vector<std::vector<float>> verify(const std::vector<int> &ids) override {
    if (pending || ids.empty())
      throw std::logic_error("active/empty transaction");
    pending = ids.size();
    ++calls;
    if (delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    std::vector<std::vector<float>> out;
    for (size_t i = 0; i < ids.size(); ++i)
      out.push_back(rows[(position + i) % rows.size()]);
    return out;
  }
  void commit(size_t n) override {
    if (multi_pending) throw std::logic_error("replay commit during a multi step");
    if (!pending || n > pending)
      throw std::logic_error("invalid commit");
    position += n;
    pending = 0;
  }
  Json counters() const override {
    return {{"engine_mode", "REPLAY_TEST_ONLY"},
            {"committed", position},
            {"verify_calls", calls},
            {"expert_vram_hits", calls},   // like Native's counters: since the last begin_request
            {"multi_calls", multi_calls},
            {"multi_prefills", multi_prefills},
            {"probe_prefills", probe_prefills},
            {"probe_groups", probe_groups},
            {"largest_prefill", largest_prefill},
            {"multi_drafts", draft_calls},
            {"slot_swaps", swaps}};
  }
};
} // namespace
std::unique_ptr<Forward> make_replay(const std::string &path) {
  return std::make_unique<Replay>(path);
}
void serve(const std::string &path, std::unique_ptr<Forward> model,
           int max_context, bool prefix_cache, bool prefix_any_conversation,
           size_t prefill_dense_batch) {
  if (path.size() >= sizeof(sockaddr_un::sun_path))
    throw std::invalid_argument("Unix socket path too long");
  int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listener < 0)
    throw std::runtime_error("socket failed");
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
  // Refuse to replace any preexisting socket/file: another process may own it.
  if (bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))) {
    close(listener);
    throw std::runtime_error(
        "socket bind failed (remove only your own stale socket): " +
        std::string(strerror(errno)));
  }
  chmod(path.c_str(), 0600);
  if (listen(listener, 64)) {
    close(listener);
    unlink(path.c_str());
    throw std::runtime_error("listen failed");
  }
  stopping = 0;
  std::signal(SIGTERM, stop_signal);
  std::signal(SIGINT, stop_signal);
  std::signal(SIGPIPE, SIG_IGN);
  Service service(std::move(model), max_context, prefix_cache, prefix_any_conversation,
                  prefill_dense_batch);
  struct Worker {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<Worker> workers;
  std::cerr << "ds41d listening " << path << "\n";
  while (!stopping) {
    for (auto i = workers.begin(); i != workers.end();)
      if (i->done->load()) {
        i->thread.join();
        i = workers.erase(i);
      } else
        ++i;
    pollfd p{listener, POLLIN, 0};
    if (poll(&p, 1, 100) <= 0)
      continue;
    int client =
        accept4(listener, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
    if (client < 0)
      continue;
    if (service.connections >= 64) {
      close(client);
      continue;
    }
    ++service.connections;
    auto done = std::make_shared<std::atomic<bool>>(false);
    workers.push_back({std::thread([&service, client, done] {
                         service.handle(client);
                         *done = true;
                       }),
                       done});
  }
  close(listener);
  for (auto &w : workers)
    w.thread.join();
  unlink(path.c_str());
}
} // namespace ds41e
