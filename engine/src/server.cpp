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
  void save_live() {
    const size_t budget = conversations_budget();
    if (!budget || prefix.inputs.empty()) return;
    for (auto it = saved.begin(); it != saved.end(); ++it)   // this very prompt is saved already (a retry)
      if ((prefix_any || it->conversation == prefix.conversation) && it->keys == prefix.keys) {
        saved.splice(saved.begin(), saved, it);
        return;
      }
    std::shared_ptr<Forward::Snapshot> state;
    try { state = model->save_conversation(); } catch (const std::exception &e) {
      std::cerr << "conversation save failed: " << e.what() << "\n";
      return;
    }
    if (!state) return;
    Saved e{prefix.conversation, prefix.inputs, prefix.keys, prefix.logits, state, 0};
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
      if ((prefix_any || it->conversation == e.conversation) && it->keys.size() <= e.keys.size() &&
          std::equal(it->keys.begin(), it->keys.end(), e.keys.begin())) {
        saved_bytes -= it->bytes;
        it = saved.erase(it);
      } else ++it;
    }
    saved_bytes += e.bytes;
    saved.push_front(std::move(e));
    while (saved_bytes > budget && saved.size() > 1) {
      saved_bytes -= saved.back().bytes;
      saved.pop_back();
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
  // queued states are held beside the RAM list (each about its conversation's KV): at most a budget's worth, and a
  // queued prompt that this one extends is not written at all
  void disk_enqueue(const Saved &e) {
    std::lock_guard<std::mutex> l(disk_m);
    for (auto it = disk_jobs.begin(); it != disk_jobs.end();) {
      const auto &k = it->e.keys;
      if ((prefix_any || it->e.conversation == e.conversation) && k.size() <= e.keys.size() &&
          std::equal(k.begin(), k.end(), e.keys.begin())) {
        disk_queued_bytes -= it->e.state->host_bytes;
        it = disk_jobs.erase(it);
      } else ++it;
    }
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
      // the files this one extends (and an older copy of itself) go
      for (auto it = disk.begin(); it != disk.end();) {
        const bool extended = it->ours && (prefix_any || it->conversation == e.conversation) &&
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
  // slot 0's content can be replaced: an idle conversation there goes to RAM (it was saved at its prompt's end)
  void drop_slot0() {
    if (slot0_job) throw std::logic_error("drop_slot0: a running job");
    if (!prefix.inputs.empty() && !prefix.boundary) save_live();
    prefix = {};
  }
  void finish(Job &J, std::string reason, Json error = nullptr, Json stats = Json::object()) {
    if (J.finished) return;
    J.finished = true;
    Json usage = {{"prompt_tokens", J.prompt}, {"completion_tokens", J.completed}};
    for (const auto &field : {
             std::pair<const char *, const char *>{"reused_prompt_tokens", "cached_prompt_tokens"},
             {"prefill_ms", "prefill_ms"}, {"generation_ms", "decode_ms"},
             {"drafted", "draft_tokens"}, {"accepted", "accepted_draft_tokens"}})
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
        if (verb != "submit" && verb != "score")
          throw std::invalid_argument("unknown op");
        J.verb = verb;
        J.op = op;
        J.ids = token_ids(op.at("token_ids"), model->vocab());
        J.prompt = int(J.ids.size());
        if (J.prompt > context)
          throw std::invalid_argument("prompt exceeds max_context");
        J.images = image_spans(op, J.ids);
        J.keys = position_keys(J.ids, J.images);
        if (!J.images.empty() && verb == "score")
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
          if (live.size() >= 8)
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
    for (;;) {
      {
        std::unique_lock<std::mutex> l(sched_m);
        sched_cv.wait_for(l, std::chrono::milliseconds(10),
                          [&] { return sched_stop || !queue.empty() || !running.empty(); });
        if (sched_stop && queue.empty() && running.empty()) return;
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
    // s0ma change 91: two or more prompts that may start now prefill together (one priority; none longer than a
    // parked slot holds; not while slot 0 holds a running conversation too long to park)
    if (combine_on() && model->supports_prefill_multi() && !(slot0_job && !parkable())) {
      std::vector<std::shared_ptr<Job>> starters;
      int room = slots - int(running.size());
      for (auto &j : waiting) {
        if (room <= 0) break;
        if (!running.empty() && !batch_lower() && j->priority > best) break;
        if (!starters.empty() && !batch_lower() && j->priority > starters.front()->priority) break;
        if (j->verb == "score" || j->sampling.max_tokens == 0 ||
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
      if (j->verb == "score" && !running.empty()) break;   // a score request resets slot 0: it runs alone
      { std::lock_guard<std::mutex> l(sched_m); queue.erase(std::find(queue.begin(), queue.end(), j)); }
      if (j->verb == "score") {
        run_score(*j);
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
    if (multi) step_multi(turn);
    else step_single(turn.front());
  }
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
  bool start_many(const std::vector<std::shared_ptr<Job>> &js) {
    std::vector<std::shared_ptr<Job>> claimed;
    bool any = false;
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
    if (groups.size() >= 2) {
      try {
        auto results = model->prefill_multi(groups);
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
      } catch (const std::exception &e) {
        for (auto &j : grouped) { fail(*j, e.what()); release(*j); }
        grouped.clear();
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
    for (auto &j : grouped) settle(j, false);
    for (auto &j : single) settle(j, true);
    return any;
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
        auto &P = entry(s);
        size_t n = 0;
        if (P.matches(J.conversation, J.keys, prefix_any)) n = P.inputs.size();
        else if (P.boundary && P.boundary_len && P.same_conversation(J.conversation, prefix_any) &&
                 P.boundary_len <= J.ids.size() && P.boundary_len <= P.inputs.size() &&
                 std::equal(P.keys.begin(), P.keys.begin() + P.boundary_len, J.keys.begin()))
          n = P.boundary_len;
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
          if (file) { from_file = true; remember(std::move(*file)); }
          else saved.splice(saved.begin(), saved, best);
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
      if (!cancelled()) save_live();   // s0ma change 73
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
    std::vector<int> tokens(members.size(), -1);
    std::vector<std::vector<std::vector<float>>> rows;
    try {
      const double t0 = now();
      rows = model->verify_multi(groups);
      const double ms = (now() - t0) * 1000;
      for (size_t g = 0; g < members.size(); ++g) {
        Job &J = *members[g];
        const uint64_t nonce = J.attempts->take();
        ++J.cycles; ++J.multi_steps;
        J.verify_ms += ms / double(members.size());
        Verification v = reject_sample({}, {}, probabilities_rows(rows[g], J.sampling.temperature, J.sampling.top_p),
                                       J.rng, nonce, J.sampling.temperature);
        if (v.emitted.empty()) throw std::runtime_error("multi step: no token");
        tokens[g] = v.emitted.front();
      }
      for (size_t g = 0; g < members.size(); ++g) model->commit_group(g, 1);
    } catch (const std::exception &e) {
      for (auto &j : members) { fail(*j, e.what()); retire(j); }
      return;
    }
    for (size_t g = 0; g < members.size(); ++g) {
      Job &J = *members[g];
      try {
      auto &P = entry(J.slot);
      J.last_step = used(J.slot) = ++use_clock;
      P.inputs.push_back(J.anchor);
      P.keys.push_back(uint64_t(J.anchor));
      P.logits = rows[g][0];
      if (J.copying) J.suffix.push(J.anchor);
      bool active = emit(J, tokens[g], rows[g][0]);
      J.anchor = tokens[g];
      if (cancelled(J) && J.completed < J.sampling.max_tokens) { J.reason = "cancelled"; active = false; }
      if (!active || J.completed >= J.sampling.max_tokens || P.inputs.size() >= size_t(context))
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
    stats["priority"] = J.priority;              // s0ma change 91
    stats["multi_steps"] = J.multi_steps;
    stats["combined_prefill"] = J.combined;
    stats["live_slot"] = J.slot;
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
    stats["copy_steps"] = J.copy_steps;          // s0ma change 72: suffix drafting
    stats["copy_drafted"] = J.copy_drafted;
    stats["copy_accepted"] = J.copy_accepted;
    finish(J, reason, nullptr, stats);
    if (J.slot >= 0 && holder(J.slot) == &J) holder(J.slot) = nullptr;
    if (J.started) --active;
    J.started = false;
  }
  // a score request (today's path): it resets slot 0, so it runs only when no job is running
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
class Replay final : public Forward {
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
  void end_prompt(bool complete) override { if (spark && !complete) incomplete = true; }
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
  std::vector<PrefillResult> prefill_multi(const std::vector<PrefillGroup> &groups,
                                           const std::function<void(size_t, bool)> &around) override {
    if (pending || groups.size() < 2) throw std::logic_error("replay prefill_multi");
    if (decline_multi) throw PrefillDeclined("replay test: declined");
    std::vector<PrefillResult> out;
    for (size_t g = 0; g < groups.size(); ++g) {
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
    ++multi_prefills;
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
