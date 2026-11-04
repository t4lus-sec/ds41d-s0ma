// s0ma change 100: which graph keys DS41_GRAPHS=fixed captures (the keys ds41d's hot path builds)
#include "ds41/graph_keys.hpp"
#include <cassert>
#include <string>

int main() {
  using ds41e::fixed_graph_key;
  for (const char *lane : {"", "laneB."})
    for (int m = 1; m <= 6; ++m)
      for (int l = 0; l < 40; ++l) {
        const std::string b = std::string(lane) + "batch." + std::to_string(m) + ".";
        const std::string p = "layers." + std::to_string(l);
        // fixed: per (lane, rows, layer)
        assert(fixed_graph_key(b + p + ".query"));
        assert(fixed_graph_key(b + p + ".route"));
        assert(fixed_graph_key(b + p + ".attn.output"));
        assert(fixed_graph_key(b + p + ".ffn.shared"));
        assert(fixed_graph_key(b + p + ".ffn.lookahead"));
        // shape-dependent: eager
        assert(!fixed_graph_key(b + p + ".attention.4096." + std::to_string(l % 8)));
        assert(!fixed_graph_key(b + p + ".attention.512.0"));
        assert(!fixed_graph_key(b + p + ".ffn.experts." + std::to_string(6 * m)));
        assert(!fixed_graph_key(b + "experts.stream." + std::to_string(l % 6 + 1) + "." + std::to_string(m * 3)));
        if (!l) {
          assert(fixed_graph_key(b + "head"));
          assert(fixed_graph_key(b + "experts.reduce"));
        }
      }
  // the single-row path's keys (one())
  assert(fixed_graph_key("layers.3.query") && fixed_graph_key("layers.3.route"));
  assert(fixed_graph_key("layers.3.attn.output"));
  assert(!fixed_graph_key("layers.3.ffn.experts"));
  return 0;
}
