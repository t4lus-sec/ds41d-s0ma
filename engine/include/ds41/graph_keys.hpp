#pragma once
// s0ma change 100: selective CUDA graphs (DS41_GRAPHS=fixed).
//
// ds41d's hot path wraps groups of launches in graph(key, body). Capturing every key (--graphs) cost more than replay
// saved on this box (change 45): attention's keys carry the scan bucket and the position mod 8, the experts' keys the
// count of ready experts, so new shapes kept being captured. The fixed kinds below depend only on the layer and the
// row count: their kernel arguments are weights, fixed scratch and m, so each is captured once per (lane, rows, layer)
// and replayed for the life of the process. Attention and the expert batches stay eager.
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ds41e {

// 0: --graphs decides (every key, as before); 1: fixed keys only (graphs on without --graphs); 2: every key
inline int graphs_mode() {
  static const int mode = [] {
    const char *v = std::getenv("DS41_GRAPHS");
    const std::string s = v ? v : "";
    if (s.empty() || s == "0") return 0;
    if (s == "fixed") return 1;
    if (s == "all") return 2;
    return -1;
  }();
  if (mode < 0) throw std::invalid_argument("DS41_GRAPHS takes fixed or all (unset: --graphs decides)");
  return mode;
}

inline bool ends_with(const std::string &s, const char *tail) {
  const std::string t(tail);
  return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

// A key whose body is the same launches with the same arguments at every step of a given row count: the layer's
// query projections (HC mix, norms, wq_a/wq_b/wkv), the attention output projection, the FFN's HC mix, norm and router,
// the router lookahead, the shared expert, the routed experts' scatter and reduce (6 routes a row, fixed), the head
inline bool fixed_graph_key(const std::string &key) {
  if (key.find(".attention.") != std::string::npos || key.find("experts.stream.") != std::string::npos) return false;
  return ends_with(key, ".query") || ends_with(key, ".output") || ends_with(key, ".route") ||
         ends_with(key, ".lookahead") || ends_with(key, ".shared") || ends_with(key, "experts.reduce") ||
         key == "head" || ends_with(key, ".head");
}

}  // namespace ds41e
