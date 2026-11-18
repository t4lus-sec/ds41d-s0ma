# ds41d for s0ma: DeepSeek-V4.1-Flash-1017 at ~80 tok/s (~100 on code) on one RTX PRO 6000

**This is a fork of [ds41d](https://github.com/SSHdotCodes/deepseek-v4.1-flash-pro6000)** (SSHdotCodes, MIT),
pinned at upstream commit `5b0ce26`, with 123 changes on top. It serves the s0ma agent harness. Upstream's README
follows [below](#upstream-readme), unchanged.

The model quant is **[0xSero's EXL3 2.7 bpw quant of DeepSeek-V4.1-Flash-1017](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw)**.
This fork adds the engine support to run it: per-expert widths from K2 to K5, his dense tensors, and a tiered store
sized for his bank. All credit for the quant, and its quality numbers below, goes to 0xSero.

## Results

Same method as upstream's: `scripts/benchmark.py` with its six fixed prompts (three short chats, a 2.3K-token
follow-up, two 8K-token agent turns) plus a warm-up. Temperature 1.0, top-p 1.0, thinking `high`. Mean of seeds 21
and 22.

**The box:**
- RTX PRO 6000 Blackwell **Max-Q** (300 W), PCIe 5.0 x16. Measured host to GPU: 57 GB/s.
- 128 GB of DDR5-5600 (2 × 64 GB).
- Driver 615.71.09, CUDA 13.4.

| build | expert weights | decode | first token |
|---|---|---:|---|
| upstream `5b0ce26` (upstream's numbers) | native MXFP4 | 31.6 tok/s | 8K cold: 13.3 s |
| upstream `5b0ce26` (upstream's numbers) | public K3 EXL3 | 49.1 tok/s | 8K cold: 10.7 s |
| this fork, change 17, 64 GB RAM | 0xSero 2.7 bpw | 33.3 tok/s | mean 8.4 s |
| this fork, change 25, 64 GB RAM | 0xSero 2.7 bpw | 38.1 tok/s | mean 7.9 s |
| this fork, change 69, 128 GB RAM | 0xSero 2.7 bpw | 81.3 tok/s | 8K cold: 5.5 s; mean 2.7 s |
| this fork, change 97 | 0xSero 2.7 bpw | ~80 tok/s | 8K cold: ~3.3 s; mean 1.11 s |
| this fork, change 111 (3 interleaved rounds) | 0xSero 2.7 bpw | 79.2 tok/s | mean 1.13 s |
| the same, with 2 GiB held for an embedding model on the GPU | 0xSero 2.7 bpw | 78.0 tok/s | mean 1.20 s |
| **this fork, change 123** (3 interleaved rounds), 2.4 GiB held for an embedding model | **0xSero 2.7 bpw** | **79.3 tok/s** | **mean 1.10 s** |

**How to read the table:**
- **Decode** includes thinking tokens.
- **Code runs faster than the table:** about 100 tok/s, up to ~110, observed in s0ma's daily coding sessions. That
  figure is from use, not from a fixed benchmark. Code repeats more of its context, so more drafts are accepted, and
  suffix drafting copies more of them.
- **What the change 123 figure is made of:** a verify step takes 26.2 ms and drafting 2.3 ms. 74% of draft
  tokens are accepted. Against change 122 with its two decode switches off, the same rounds measured 77.6 tok/s and
  26.9 ms a verify step, with identical tokens.
- **Long prompts** (changes 117 to 120, `lab long` at a 1M-token context): first token for a cold 128K-token prompt
  went from 75.4 s to 48.7 s, 64K from 34.1 to 21.1 s, 32K from 15.7 to 10.6 s.
- **Mean first token** averages all six prompts. Upstream reports only cold prompts.
- **The whole bank lives in VRAM and RAM.** At 128 GB, 0xSero's experts fit in VRAM plus the pinned RAM tier, so
  decode reads no experts from the NVMe drive. Engram's n-gram rows are still read from it.
- **Token-exact checks:** changes that claim exactness were checked against the build before them. The same seeds
  must give the same tokens (`7/7` per seed).

**Concurrency.** Changes 91 and 93 run several conversations at once in live slots. A profile of one
conversation's steps measured the cost of extra rows: dense work and attention batch well, going from 6.6 to 9.3 ms
of attention between one and six rows. Past four rows, though, steps wait on RAM-to-GPU expert copies, because
more rows reach colder experts. Batching conversations together is therefore estimated at about 1.2 to 1.3× total
throughput at best on this card. That estimate has not been measured.

## Quality

This fork runs 0xSero's tensors as published:
- **his routed experts:** 2.70 bpw. Each expert has its own width: 6,278 at K2, 7,447 at K3, 1,600 at K4 and 35
  at K5.
- **his dense tensors:** attention, shared experts and the head at K4 or K5.

Everything else is either exact in distribution or changes rounding order only:
- speculative decoding with rejection sampling;
- copying drafts from the conversation;
- caching and scheduling;
- expert placement.

The drafter's own experts are quantized (K2). That changes only how many drafts are accepted, not the output
distribution.

**0xSero's measurements** against native 1017, on sudoingX's 199-conversation panel (disjoint from his 1.2M-token
calibration set), from his model card:

| | |
|---|---|
| mean KL | **0.0304 nats** (95% CI 0.0268–0.0343) |
| top-1 agreement | 94.95% |
| panel perplexity | 1.6010 native → 1.6231 |

We have not re-measured KL on this fork. The engine's `score` verb returns teacher-forced log-probabilities, so it
can be checked.

## What the fork changes

The full list is in `git log 5b0ce26..`: one commit per change, each with what it does and what it measured.

**Expert banks:**
- EXL3 K2 and mixed K2/K3 banks.
- 0xSero's bank, with a width per expert (`grouped_exl3_var`), slot classes per width in VRAM and RAM, and class
  weights.
- His dense tensors and K4 head, split over K in decode and run through FP16 GEMMs in prefill.
- The VRAM those free goes to more expert slots.

**Decode:**
- The drafter's routed experts from an EXL3 bank.
- Draft draws and acceptance on the GPU.
- Drafts copied from the conversation's own tokens (suffix drafting).
- Row batching.
- Selective CUDA graphs with PDL.
- EXL3 decode kernels with deeper weight prefetch, and a layer's gate and up projections in one launch.
- GPU route planning.
- Top-p on the host without a vocabulary sort.
- Full residency at load whenever the RAM tier can hold the bank.

**Prefill:**
- Staged experts spread over four lanes, in grouped launches.
- Engram reads batched and deeper.
- One-row-per-warp routing.
- The prefill reserve lent to expert slots during decode, and expert slots carved for long prompts' scratch.
- Engram reads merged into larger runs over several readers, and read ahead for a prompt's later segments.
- A prompt's drive reads admitted into the RAM tier as they pass.

**Serving:**
- Every finished prompt's state saved in RAM and on disk, so a returning conversation prefills only its new
  tokens.
- Live slots with a priority scheduler for several conversations.
- Prompt caches per conversation for subagents, ended with `POST /v1/conversations/end`.
- One-token probes prefilled together.
- A VRAM reserve for another process on the card (`vram_reserve_gib`).

**API:**
- Special strings in request text stay text, so a `</think>` in a document can't end the model's reasoning.
- A loop guard ends repeating reasoning, and repeating passages are cut out of the history.
- 120 s keep-alive.

**Tooling:**
- A lab for A/B benches.
- A drafter trainer that trains on s0ma's own requests, against the served target.

## Running it

s0ma builds and runs this fork with its `ds41.sh` script. The script applies these changes onto upstream `5b0ce26`,
fetches the banks and manages the service.

Without it, the pieces are:
1. `scripts/sero_bank.py fetch models/sero-bank` (about 186 GB) and `scripts/sero_bank.py dense models/sero-bank`.
2. `"experts": "sero"` in `config.local.json`.
3. Upstream's `scripts/build.sh` and `scripts/run.py start`.

The drafter's EXL3 bank is prepared by `ds41.sh`, and the steps are not yet documented outside it.

## Credits

- **SSHdotCodes**, for ds41d, which this fork builds on.
- **[0xSero](https://huggingface.co/0xSero)**, for the EXL3 2.7 bpw quant this fork serves, and its quality
  measurements.
- **sudoingX**, for the evaluation panel behind those measurements.
- **turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3)** (MIT), for the EXL3 format, which the
  fork's K2–K5 kernels decode.
- **DeepSeek**, for the model.

The fork's changes are MIT, as is upstream. No weights are included.

---

<a id="upstream-readme"></a>

# DeepSeek-V4.1-Flash-1017 on one RTX PRO 6000 (upstream README)

**31.6 tok/s decode at native precision on one 96 GB card with 128 GB of RAM.** The model is DeepSeek's [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): a 510 GB checkpoint with native MXFP4 experts and FP8 dense weights. It runs here unconverted and unquantized, with:

- a 1,048,576-token context;
- thinking with `low` / `high` / `max` reasoning effort;
- tool calls;
- images, through the checkpoint's own vision tower;
- speculative decoding with the checkpoint's own MTP layers;
- OpenAI Chat Completions, Anthropic Messages and OpenAI Responses APIs.

`ds41d` is a purpose-built engine. 296 GB of the checkpoint are routed experts, three times the card's memory. Each MoE layer's 384 experts are split three ways: the most used stay in VRAM, the next tier in pinned host RAM, and the rest are read from NVMe on demand, straight into VRAM with GPUDirect Storage.

| temperature 1.0, top-p 1.0, thinking `high` | |
|---|---:|
| decode, mean of 3 short chats, a 2.3K-token multi-turn follow-up and two 8K-token agent turns | **31.6 tok/s** |
| decode per prompt | 27–38 tok/s |
| time to first token, cold 8K / 32K-token prompt | 13.3 s / 33.4 s |
| time to first token, follow-up turn on an 8K-token conversation | 0.5 s |
| against DeepSeek's reference implementation, 16-token greedy continuation | 16/16 tokens agree, mean KL 0.015 |

Decode rates include thinking tokens and exclude prompt processing. They vary by prompt with how many experts each token needs from RAM and NVMe and how many drafted tokens are accepted (82–92%). A fresh clone of this repo, built with `scripts/build.sh`, returned the same tokens as the measured service on all 14 benchmark requests, at the same speed (31.61 against 31.67 tok/s), and its binaries have the same code. [Benchmarks](docs/benchmarks.md).

## Faster with 3-bit experts

`"experts": "k3"` runs the 40 target layers' routed experts from public 3-bit EXL3 quants instead of the native MXFP4. Each expert shrinks from 18.8 to 13.3 MB, so more of them fit in VRAM and RAM and fewer come from NVMe. Everything else, the drafter included, stays native.

| same prompts and method, fresh clone | native | `"experts": "k3"` |
|---|---:|---:|
| decode, mean of the 6 benchmark prompts, seeds 21 and 22 | 31.65 tok/s | **49.13 tok/s** |
| time to first token, cold 8K / 32K-token prompt | 13.3 s / 33.4 s | **10.7 s / 28.4 s** |

Perplexity on the model's own outputs is 2.1% higher. GSM8K, HumanEval and tool-call checks showed no loss, and on code the model thinks about 15% longer, so an answer still takes about a quarter less decode time. The bank takes 191 GiB more disk: `scripts/k3_bank.py fetch models/k3-bank` downloads it from the pinned quants and checks every layer's SHA-256. Method and results: [docs/exl3.md](docs/exl3.md).

## What you need

- **GPU:** one RTX PRO 6000 Blackwell 96 GB, otherwise idle. The engine uses about 93 GB of VRAM.
- **RAM:** 128 GB at least. The service pins 104 GiB of host memory for experts: by default your total RAM minus 16 GiB, at most 104 GiB. The expert placement seeds 96 GiB of experts into RAM, so the tier cannot be smaller; 96 GiB measured 3.3% slower than 104 GiB.
- **Disk:** 520 GB on a fast NVMe drive for the checkpoint, read during decode (an expert is 18.8 MB). Measured on one PCIe 5.0 drive with ext4. Another 10 GB for the Python environments and the build.
- **Software:**
  - Linux with cgroup v2 and `systemctl --user`. The service runs with `MemorySwapMax=0`.
  - NVIDIA driver 615 (measured with 615.71.09).
  - **CUDA Toolkit 13.4.0** exactly. The engine checks that it runs with that release's `libcudart.so.13.4.92` and with the cuBLAS 13.1.1.3 pair the build downloads.
  - GCC 13 or newer (measured with 15.2), CMake 3.24 or newer, OpenSSL headers. On Ubuntu: `sudo apt install build-essential cmake libssl-dev`.
  - [uv](https://docs.astral.sh/uv/) or Python 3.12 with `venv`.
- **Recommended:** GPUDirect Storage P2P, so NVMe reads skip host RAM. It takes one modprobe file and a reboot ([docs/gds.md](docs/gds.md)). Without it, reads go through a host bounce buffer: 30.8 instead of 31.6 tok/s, with the same tokens.

## Install and run

```bash
git clone https://github.com/SSHdotCodes/deepseek-v4.1-flash-pro6000.git
cd deepseek-v4.1-flash-pro6000

# .venv (API, downloads, benchmarks) and .venv-assets (PyTorch 2.14, to export the RoPE and Engram tables).
scripts/setup.sh

# 510 GB at the pinned revision into models/; checks every file's size, and small files' hashes.
.venv/bin/python scripts/download.py

# Pinned cuBLAS and TileLang wheels (SHA-256 checked), the components, the engine assets (hashes all 510 GB
# against the Hub's LFS hashes once, about 6 minutes), ds41d and its tests. Output: build/dist.
scripts/build.sh

# Preflight (idle GPU, RAM, build), then a systemd user service with no swap. Loads in about a minute.
python3 scripts/run.py start
```

The server is at **http://127.0.0.1:18087/v1**, model ID **`deepseek-v4.1-flash-1017`**, loopback only. Other actions:

- `python3 scripts/run.py logs`: show the logs.
- `python3 scripts/run.py stop`: unload the model.
- `python3 scripts/run.py command`: print the exact engine and API command lines.

To change the checkpoint location, port, unit name, RAM tier or GDS mode, copy `config.example.json` to `config.local.json` ([operations](docs/operations.md)).

```bash
curl http://127.0.0.1:18087/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash-1017", "messages": [{"role": "user", "content": "Explain an LRU cache."}],
  "max_tokens": 4096, "temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high"}'
```

Port 18087 serves:

- OpenAI chat completions, with tools, streaming and reasoning split out (`reasoning_content`);
- Anthropic Messages, at `/v1/messages` and `/v1/messages/count_tokens`, with thinking blocks and tool use;
- OpenAI Responses, at `/v1/responses`.

Each API takes images in its own format: chat `image_url`, Responses `input_image`, Anthropic `image` blocks with a base64 or URL source. Send them as `data:` URLs or base64, or as public http(s) URLs, which the server fetches. It refuses local paths and private addresses. An image costs up to 1,024 prompt tokens (184 for 512 x 512, 496 for 1024 x 768), and a request may carry up to 16 ([images](docs/engine.md#images)).

Defaults:

- Thinking is on, at `high`. `reasoning_effort` takes `low`, `high` or `max` (the encoding's 50, 75 and 100); `"thinking": false` answers directly. Anthropic `thinking.budget_tokens` maps to the same three levels.
- Sampling is DeepSeek's recommendation for this model: temperature 1.0, top-p 1.0.
- A request may ask for up to 131,072 output tokens.
- One request runs at a time; up to 16 wait in a queue. A request that extends the latest conversation prefills only its new tokens.

## Reproduce the measurements

```bash
python3 scripts/run.py stop
python3 scripts/benchmark.py --seeds 21 22 --out results/bench.json     # about 6 minutes
python3 scripts/run.py start
.venv/bin/python scripts/api_check.py --out results/api-check.json     # API checks and prompt timings
.venv/bin/python scripts/vision_e2e.py --out results/vision-e2e.json    # image answers, prefix reuse
```

`benchmark.py` starts its own engine with the service's exact command and runs the six fixed prompts in `benchmarks/cases`, plus a warm-up, once per seed. It records time to first token, decode rate and a SHA-256 of every token sequence. Sampling is seeded and exact, so on the same GPU, driver and build the tokens should equal [ours](benchmarks/results-20261018.json); the script reports how many match. `api_check.py` checks every API surface (chat, streaming, reasoning effort, tool calls and round trips, Anthropic messages and tools, `count_tokens`, Responses) and times cold 2K, 8K and 32K-token prompts and a follow-up turn. `vision_e2e.py` asks about generated images and the checkpoint's example photo and checks prefix reuse with images ([results](benchmarks/vision-20261018)). GitHub CI runs the repository tests, the API tests and script checks; CUDA and model checks need the GPU.

## How it works

- **Three expert tiers.** Every routed expert is 18.8 MB of MXFP4 weights and scales, 15,744 of them across 40 MoE layers and the 3 MTP layers.
  - VRAM holds 4,452 of them (78 GiB). 3,240 are pinned by a placement ranked by routing frequency; the rest are a per-layer LFU cache.
  - The RAM tier holds about 5,900 more in 104 GiB of pinned memory, also LFU, and exclusive of VRAM: an expert evicted from VRAM moves to RAM.
  - The remaining third is read from NVMe when routed, in 2 MiB chunks, straight into its VRAM slot through cuFile.
- **Speculative decoding (DSpark).** The checkpoint's three MTP layers draft up to four tokens, and the target verifies them in one pass that loads each layer's experts once for all rows. Before each draft token the engine estimates its acceptance from the drafter's confidence and stops when another row would not pay for itself. Rejection sampling keeps the output distribution the target's own.
- **Prefetching.** While layer *l* runs, layer *l+1*'s router is scored on its input and the predicted RAM-resident experts start copying to VRAM. After a long prompt, the experts it used are read from NVMe into RAM for the answer.
- **Prompts.** Prefill runs 256 rows per launch in segments sized to the VRAM the expert cache leaves free, and a follow-up turn reuses the latest conversation's state.

Decode is bound by expert transfers, not compute. Profiling during development (raw data not included) found the GPU busy for about 22 ms of an 80 ms verify step. Each layer costs about 0.16 ms plus 0.17 ms per expert from RAM and 1.19 ms per expert from NVMe. More RAM, a second NVMe drive or faster DRAM would help most. [Engine details](docs/engine.md), [asset pipeline](docs/assets.md).

## Credits and licenses

Thanks to **DeepSeek** for DeepSeek-V4.1-Flash-1017, its encoding and its reference implementation, and to the **TileLang** and **NVIDIA CUTLASS / cuBLAS / cuFile** teams.

Repo code is MIT. The model, its prompt encoding and inference code are MIT; no weights are included in this repository. The benchmark prompts come from this project, WildChat-1M (ODC-BY 1.0) and betterwright-agent-traces (Apache-2.0). See [THIRD_PARTY.md](THIRD_PARTY.md).
