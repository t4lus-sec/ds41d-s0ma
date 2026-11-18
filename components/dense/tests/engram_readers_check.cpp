// s0ma change 118: the Engram store's parallel readers and merged reads against the single reader, page by page.
// CPU only (no GPU: pin_memory 0). Builds a small stand-in checkpoint (two layers' FP8 rows and E8M0 scales in one
// safetensors shard, its index, a hash file and an integrity receipt) in DIR, then reads the same rows with
// DS41_ENGRAM_READERS / DS41_ENGRAM_RUN_PAGES set to 1/1, 4/16 and 3/2, with and without the row cache, and checks
// every row's 264 bytes against the file.
//   g++ -std=c++17 -O2 -Iinclude -Iengram/vendor -I<cuda include> tests/engram_readers_check.cpp engram/src/store.cpp
//       -L<cudart dir> -l:libcudart.so.13 -lpthread -o engram_readers_check && ./engram_readers_check DIR
#include "ds41/engram.h"
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using Json = nlohmann::json;

static uint8_t byte_at(int tensor, uint64_t i) { return uint8_t((i * 2654435761u + tensor * 97u + (i >> 9)) & 0xff); }

int main(int argc, char **argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s DIR\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  // 24 small primes a layer (row counts differ between the layers)
  std::vector<int64_t> pr;
  for (int64_t n = 3001; pr.size() < 48; n += 2) {
    bool p = true;
    for (int64_t d = 3; d * d <= n; d += 2) if (n % d == 0) { p = false; break; }
    if (p) pr.push_back(n);
  }
  int64_t rows[2] = {0, 0};
  Json layers = Json::array();
  for (int l = 0; l < 2; ++l) {
    std::vector<int64_t> primes(pr.begin() + l * 24, pr.begin() + l * 24 + 24), offsets;
    int64_t off = 0;
    for (auto p : primes) { offsets.push_back(off); off += p; }
    rows[l] = off;
    layers.push_back({{"layer", l ? 14 : 1}, {"multipliers", {3, 5, 7, 11}}, {"primes", primes}, {"offsets", offsets}});
  }
  std::vector<int64_t> map(129280);
  for (size_t i = 0; i < map.size(); ++i) map[i] = int64_t(i % 99092);
  Json hash = {{"schema", "ds41-engram-hash-v1"}, {"token_map", map}, {"pad_id", 0}, {"compressed_vocab_size", 99092},
               {"layers", layers}};
  // the shard: four tensors, data in order
  const std::string shard = "model-00001-of-00001.safetensors";
  const char *names[4] = {"model.layers.1.engram.embed.weight", "model.layers.1.engram.embed.weight_scale_inv",
                          "model.layers.14.engram.embed.weight", "model.layers.14.engram.embed.weight_scale_inv"};
  uint64_t sizes[4] = {uint64_t(rows[0]) * 256, uint64_t(rows[0]) * 8, uint64_t(rows[1]) * 256, uint64_t(rows[1]) * 8};
  Json header = Json::object();
  uint64_t at = 0;
  for (int t = 0; t < 4; ++t) {
    header[names[t]] = {{"dtype", t % 2 ? "F8_E8M0" : "F8_E4M3"},
                        {"shape", {rows[t / 2], t % 2 ? 8 : 256}}, {"data_offsets", {at, at + sizes[t]}}};
    at += sizes[t];
  }
  std::string h = header.dump();
  while ((8 + h.size()) % 8) h += ' ';
  {
    std::ofstream f(dir + "/" + shard, std::ios::binary);
    const uint64_t n = h.size();
    f.write(reinterpret_cast<const char *>(&n), 8);
    f.write(h.data(), std::streamsize(h.size()));
    for (int t = 0; t < 4; ++t) {
      std::vector<char> buf(sizes[t]);
      for (uint64_t i = 0; i < sizes[t]; ++i) buf[i] = char(byte_at(t, i));
      f.write(buf.data(), std::streamsize(buf.size()));
    }
  }
  struct stat st{};
  stat((dir + "/" + shard).c_str(), &st);
  Json index = {{"weight_map", Json::object()}};
  for (auto n : names) index["weight_map"][n] = shard;
  Json receipt = {{"complete", true}, {"root", dir},
                  {"files", {{{"path", shard}, {"ok", true}, {"size", uint64_t(st.st_size)},
                              {"mtime_ns", uint64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec}}}}};
  std::ofstream(dir + "/index.json") << index.dump();
  std::ofstream(dir + "/hash.json") << hash.dump();
  std::ofstream(dir + "/receipt.json") << receipt.dump();

  // the rows to read: random, with duplicates, more than one batch (4096 rows) a ticket
  std::mt19937_64 rng(7);
  std::vector<int64_t> want[2];
  for (int l = 0; l < 2; ++l) {
    for (int i = 0; i < 30000; ++i) want[l].push_back(int64_t(rng() % uint64_t(rows[l])));
    for (int i = 0; i < 2000; ++i) want[l].push_back(want[l][rng() % want[l].size()]);   // repeats
    for (int64_t r = 0; r < 3000; ++r) want[l].push_back(r);                              // a dense run
    want[l].push_back(rows[l] - 1);                                                       // the last row
  }
  struct Case { const char *readers, *run; uint64_t cache; };
  const Case cases[] = {{"1", "1", 0}, {"4", "16", 0}, {"3", "2", 0}, {"4", "16", 100000}, {"4", "256", 0}};
  int failures = 0;
  for (auto c : cases) {
    setenv("DS41_ENGRAM_READERS", c.readers, 1);
    setenv("DS41_ENGRAM_RUN_PAGES", c.run, 1);
    const std::string root = dir, ix = dir + "/index.json", hs = dir + "/hash.json", rc = dir + "/receipt.json";
    ds41_engram_config cfg{};
    cfg.struct_bytes = sizeof cfg; cfg.abi_version = 1;
    cfg.weight_root = root.c_str(); cfg.index_json = ix.c_str(); cfg.hash_json = hs.c_str(); cfg.integrity_json = rc.c_str();
    cfg.cache_rows = c.cache; cfg.queue_depth = 32; cfg.max_batch_rows = 4096; cfg.pin_memory = 0;
    ds41_engram_store *s = nullptr;
    if (ds41_engram_open(&cfg, &s)) { std::printf("open: %s\n", ds41_engram_error()); return 1; }
    for (int pass = 0; pass < (c.cache ? 2 : 1); ++pass)
      for (int l = 0; l < 2; ++l) {
        ds41_engram_ticket *t = nullptr;
        if (ds41_engram_submit(s, l ? 14 : 1, want[l].data(), want[l].size(), &t) || ds41_engram_wait(t)) {
          std::printf("read: %s\n", ds41_engram_error()); return 1;
        }
        const void *data; size_t n;
        ds41_engram_data(t, &data, &n);
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) {
          const auto *row = static_cast<const uint8_t *>(data) + i * 264;
          const uint64_t r = uint64_t(want[l][i]);
          for (int b = 0; b < 256; ++b) bad += row[b] != byte_at(l * 2, r * 256 + b);
          for (int b = 0; b < 8; ++b) bad += row[256 + b] != byte_at(l * 2 + 1, r * 8 + b);
        }
        ds41_engram_ticket_free(t);
        if (bad) { ++failures; std::printf("FAIL readers %s run %s cache %llu pass %d layer %d: %zu bytes differ\n", c.readers,
                                           c.run, (unsigned long long)c.cache, pass, l ? 14 : 1, bad); }
      }
    ds41_engram_stats sx{};
    ds41_engram_get_stats(s, &sx);
    std::printf("readers %s, run pages %s, cache %llu: %llu rows, %llu cache hits, %llu pages in %llu batches\n", c.readers,
                c.run, (unsigned long long)c.cache, (unsigned long long)sx.submitted_rows,
                (unsigned long long)sx.cache_hits, (unsigned long long)sx.disk_pages, (unsigned long long)sx.batches);
    ds41_engram_close(s);
  }
  std::printf(failures ? "engram readers: %d FAILED\n" : "engram readers: all rows match the file (%d failures)\n", failures);
  return failures ? 1 : 0;
}
