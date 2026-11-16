#!/usr/bin/env python3
"""Loopback DeepSeek-V4.1-Flash-1017 server: ds41d plus the OpenAI/Anthropic API, the measured profile, no swap."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30
# VRAM expert-cache layout: which experts are pinned in VRAM or RAM, per layer, including the MTP layer the
# drafter uses (docs/engine.md#expert-residency).
PLACEMENT = ROOT / 'profiles/p80-mtp48.placement'
# 4,452 VRAM expert slots of 18.8 MB (77.99 GiB). With the KV cache, prefill scratch and dense weights this
# leaves about 3 GiB of the 96 GB card free.
EXPERT_SLOTS = 4452
# "experts": "k3" (docs/exl3.md): the routed experts of the 40 target layers run from the EXL3 3-bit bank that
# scripts/k3_bank.py builds (13.3 MB per expert). The same expert VRAM then holds 5,745 target slots (same pins, more
# dynamic slots) plus a separate store with the drafter's 384 native experts.
K3_PLACEMENT = ROOT / 'profiles/k3-p80.placement'
K3_EXPERT_SLOTS = 5745
# "experts": "k2" (s0ma's leg 12, scripts/k2_bank.py): the same at two bits (8.9 MB per expert): the same expert VRAM
# holds 8,602 target slots (k2-p80: the same pins, the dynamic budgets scaled again).
K2_PLACEMENT = ROOT / 'profiles/k2-p80.placement'
K2_EXPERT_SLOTS = 8602
# K2's drafting, measured on the 64 GB box (s0ma, 2026-10-31; claude/ds41-serving.md): with the 2-bit bank a verify
# row is cheap enough that longer drafts pay. STOP_Q 0.65 -> 0.2 and width 4 -> 6 took decode 54.7 -> 58.4 tok/s
# (0.1: 56.1, 0.05: 55.4). Native and K3 keep their measured (published) drafting.
K2_SPARK_WIDTH = '6'
# "experts": "mixed" (s0ma change 12, scripts/mixed_bank.py): K3 in some layers, K2 in the rest, from the two banks.
# The bank directory holds its manifest, receipts and placement; the placement states every VRAM slot (each K3
# layer's capacity restated in 13.3 MB slots), and ds41d gives each width its own slot class. Drafting as K2.
MIXED_PLACEMENT = None   # the bank's mixed-p80.placement unless set (the lab sets a trimmed copy)
# "experts": "sero" (s0ma change 15, scripts/sero_bank.py): 0xSero's 2.7 bpw quant's routed experts (a width each,
# K2..K5), the quality baseline. Its bank directory has the same files as a mixed bank and runs the same way.
K2_ENV = {'DS41_SPARK_STOP_Q': '0.2',
          # s0ma change 14: a verify step's dense rows in one cuBLAS call where the load check found it bit-identical
          # (+1.7 tok/s on K2, 58.3 -> 60.1). At run time ~2 in 5,000 such calls round differently (a K3 run's
          # verifier: 32x5120 BF16, 2 rows), so K3 keeps it off to stay token-exact with its published run; the lossy
          # banks take the speed. DS41_ROWS_BATCHED=0 turns it off.
          'DS41_ROWS_BATCHED': '1'}
# s0ma change 20: 0xSero's bank splits its width classes' dynamic VRAM with K3 weighted 1.3 (change 19): 34.13 -> 35.04
# tok/s, NVMe misses 21,915 -> 21,337 (K3:1.6 34.94). A lab --env DS41_CLASS_WEIGHTS=... still wins.
# s0ma change 24: and drafts ending at confidence 0.35 (K2's 0.2 makes rows Sero pays NVMe reads for): 34.33 -> 36.01
# tok/s, NVMe misses 20,332 -> 17,506, acceptance 0.660 -> 0.723 (width 4 at 0.35: 35.73).
SERO_ENV = {'DS41_CLASS_WEIGHTS': 'K3:1.3', 'DS41_SPARK_STOP_Q': '0.35'}
# s0ma change 25: 3.5 GiB more expert slots at max_context 262,144 (2.4 at 1M: the arenas hold ~1.1 GB more). Measured:
# 36.01 -> 38.08 tok/s, NVMe misses 17,506 -> 15,947, first tokens unchanged (8K 15.0-15.7 s); 4.5 GiB pushed the 8K
# prompts' prefill into two segments (+7-12 s each).
SERO_MAX_CONTEXT = 262144
SERO_EXTRA_GIB, SERO_EXTRA_GIB_1M = '3.5', '2.4'
# The measured host RAM tier. More RAM lowers NVMe reads (96 -> 104 GiB: +3.3% decode). s0ma change 63: up to 112, so
# Sero's full residency (every unpinned expert in VRAM or RAM, ~90-100 GiB of tier) fits a 128 GB box with room to spare.
MAX_HOST_GIB = 112
# The placement seeds 5,480 experts into the RAM tier (95.99 GiB), so the tier needs at least 96 GiB.
MIN_HOST_GIB = 96
# RAM left to the OS, the API and page cache next to the pinned tier.
RESERVE_GIB = 16

# Every flag is part of the measured profile; see docs/engine.md for what each one does.
ENGINE_FLAGS = [
    # ds41d keeps a guard from its first native-forward milestone; the full-model checks it waited for have
    # passed since (docs/engine.md#quality).
    '--allow-unqualified',
    # decode and verify steps are replayed as CUDA graphs
    '--graphs',
    # routed experts run from the VRAM expert cache, every row of a verify step batched per layer
    '--streamed-experts',
    # the RAM tier is a cache too (LFU, recent window 2, decay every 256 steps), exclusive of VRAM
    '--dynamic-host', '--host-recent', '2', '--host-decay', '256', '--reset-cache-history',
    # a request that extends any earlier conversation prefills only its new tokens
    '--prefix-cache', '--prefix-any-conversation',
    # batched prefill: 256 rows per dense/attention launch, 8 FP8 projection rows per launch
    '--prefill', '--prefill-chunk', '256', '--prefill-dense-batch', '8',
    # after a prompt of 256+ rows, read the experts its last 256 rows used from NVMe into RAM (up to 16 GiB)
    '--prefill-warm-rows', '256', '--prefill-warm-gib', '16',
    # token embedding in host RAM and a 1 KiB CUDA stack: more VRAM for expert slots
    '--host-embedding', '--cuda-stack-bytes', '1024',
    # attention wo_a stays in its native FP8 (dequantized in registers, FP32 FMA) instead of a BF16 copy
    '--compact-woa',
    # DSpark speculative decoding with the checkpoint's own MTP layer: up to 4 drafted tokens per step,
    # verified by the target with exact rejection sampling
    '--spark', '--spark-width', '4',
    # s0ma: retire each layer's expert lease without a blocking round trip to the store thread (measured exact)
    '--async-retire',
    # score layer l+1's router on layer l's input and prefetch the predicted RAM-resident experts into VRAM
    # (a transfer hint: routing and outputs are unchanged)
    '--prefetch-lookahead',
]

ENGINE_ENV = {
    # dense projections on the reference (cuBLAS-compatible) path
    'DS41_DENSE_REFERENCE': '1',
    # NVMe misses are read in 2 MiB O_DIRECT chunks, each uploaded as soon as it lands
    'DS41_MOE_CHUNK_UPLOAD': '1', 'DS41_MOE_IO_CHUNK_KB': '2048',
    # expert-cache policy: LFU counts decay every 256 steps, drift-aware admission for width-1 decode,
    # faster eviction minima
    'DS41_MOE_DECAY256': '1', 'DS41_MOE_DRIFT_0_50': '1', 'DS41_MOE_TOKEN_MAJOR_WIDTH1': '1',
    'DS41_MOE_FAST_MINIMA': '1',
    # an expert evicted from VRAM moves to the RAM tier instead of being dropped
    'DS41_MOE_DEMOTE': '1',
    # s0ma change 28: a prompt's resident experts run on 3 side streams beside the staged sweep (EXL3 banks only;
    # exact). Sero: first token 7.90 -> 6.60 s mean, 8K prompts 15.0-15.8 -> 12.5-13.1 s, decode unchanged (37.54)
    'DS41_PREFILL_LANES': '4',
    # s0ma change 30: speculative acceptance on the GPU (change 29; same uniforms, fp64, only the sums' order
    # differs). Sero: 37.44 -> 38.42 tok/s, acceptance unchanged (0.715). =check reruns each step on the host; =0 off
    'DS41_SPARK_GPU_VERIFY': '1',
    # DSpark adaptive draft length. Before drafting token k the engine estimates its acceptance from the
    # drafter's top probability and confidence head (a logistic model), and stops when the expected gain no
    # longer pays for another verify row. The decision uses only what is known before token k is drawn, so
    # sampling stays exact.
    'DS41_SPARK_STOP_Q': '0.65', 'DS41_SPARK_STOP_MODEL': '-2.049,2.987,-0.488,0.85,-0.063',
    'DS41_SPARK_STOP_SCALE': '1,1.05,1.42,1.79,2.16',
    # lookahead prefetch: top 4 predicted experts per row, for the first 2 rows of a step
    'DS41_LOOKAHEAD_K': '4', 'DS41_LOOKAHEAD_N': '2',
    # after a long appended turn the drafter restarts from the append's last 128 rows
    'DS41_SPARK_APPEND_RESTORE': '1',
    # s0ma: each drafted position's q, confidence and draw computed on the GPU (measured exact, 14/14)
    'DS41_SPARK_GPU_DRAW': '1',
    # long prompts split into prefill segments that fit the VRAM the expert cache leaves free
    'DS41_PREFILL_CHUNK_FIT': '1',
    # s0ma change 85: change 84's staged lanes. A prompt's experts copied from RAM run on 2 side lanes of the greatest
    # stream priority, so their ring slots free soon after the copies land (exact; Sero 2026-11-03, one seed, tokens
    # 7/7 identical): sweep 8K 2946 -> 2601 ms, 2.3K 2185 -> 1939 ms; prefill -7 to -8% at every size; ring slot waits
    # 821 -> 20 ms. =0 off
    'DS41_PREFILL_STAGED_LANES': '2',
    # s0ma change 85: change 83's routes picked and planned on the GPU (exact, tokens 7/7 identical): the GPU's wait
    # for the host's routes 1.18 -> 0.67 ms a step, decode 78.3 -> 78.8 tok/s. =0 off, =check compares with the host
    'DS41_GPU_ROUTE': '1',
    # s0ma change 91: 4 conversations live in VRAM (3 parked, each up to DS41_SLOT_CONTEXT = min(max_context, 262,144)
    # tokens; ~235 MiB each), served at once by the engine's scheduler: one request of the best priority alone runs
    # the single-stream step, several of the same priority share multi-conversation steps. =1: one at a time
    'DS41_LIVE_SLOTS': '4',
}
# GPUDirect Storage: NVMe misses during decode are read straight into their VRAM slot (docs/gds.md).
GDS_ENV = {'DS41_MOE_GDS': '1', 'CUFILE_USE_PCIP2PDMA': 'true'}


def configuration():
    cfg = json.loads((ROOT / 'config.example.json').read_text())
    local = ROOT / 'config.local.json'
    mine = json.loads(local.read_text()) if local.exists() else {}
    cfg.update(mine)
    if cfg.get('experts') == 'sero' and 'max_context' not in mine:
        # s0ma change 25: s0ma's window (its adapter caps requests at 262,144); the VRAM the 1M arenas held goes to
        # expert slots (SERO_ENV's DS41_EXPERT_EXTRA_GIB)
        cfg['max_context'] = SERO_MAX_CONTEXT
    for key in ('weights', 'assets', 'build'):
        cfg[key] = str((ROOT / Path(cfg[key]).expanduser()).resolve())
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', cfg['unit']):
        raise ValueError('unit must contain only letters, digits, underscores or hyphens')
    if not isinstance(cfg['port'], int) or not 1024 <= cfg['port'] <= 65535:
        raise ValueError('port must be an unprivileged port number')
    if cfg['host_gib'] != 'auto' and not (isinstance(cfg['host_gib'], int) and MIN_HOST_GIB <= cfg['host_gib']):
        raise ValueError(f'host_gib must be "auto" or an integer of at least {MIN_HOST_GIB}')
    if cfg['gds'] not in ('auto', True, False):
        raise ValueError('gds must be "auto", true or false')
    if not (isinstance(cfg['max_context'], int) and 4096 <= cfg['max_context'] <= 1048576):
        raise ValueError('max_context must be 4096..1048576')
    if cfg['experts'] not in ('native', 'k3', 'k2', 'mixed', 'sero'):
        raise ValueError('experts must be "native", "k3", "k2", "mixed" or "sero"')
    cfg['k3_bank'] = str((ROOT / Path(cfg['k3_bank']).expanduser()).resolve())
    cfg['k2_bank'] = str((ROOT / Path(cfg.get('k2_bank', 'models/k2-bank')).expanduser()).resolve())
    cfg['mixed_bank'] = str((ROOT / Path(cfg.get('mixed_bank', 'models/mixed-bank')).expanduser()).resolve())
    cfg['sero_bank'] = str((ROOT / Path(cfg.get('sero_bank', 'models/sero-bank')).expanduser()).resolve())
    return cfg


def bank_dir(cfg):
    """the mixed-width bank's directory: the sero bank for "sero", the assembled mixed bank for "mixed\""""
    return Path(cfg['sero_bank'] if cfg['experts'] == 'sero' else cfg['mixed_bank'])


def mixed_placement(cfg):
    return Path(MIXED_PLACEMENT) if MIXED_PLACEMENT else bank_dir(cfg) / 'mixed-p80.placement'


def placement_slots(path):
    """the target VRAM slots a placement states: pins and dynamic budgets of layers 0..39"""
    n = 0
    for x in Path(path).read_text().splitlines():
        if x.startswith('vram ') and int(x.split()[1]) < 15360:
            n += 1
        elif x.startswith('budget ') and int(x.split()[1]) < 40:
            n += int(x.split()[2])
    return n


def meminfo():
    return {l.split(':')[0]: int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()}


def host_gib(cfg):
    if cfg['host_gib'] != 'auto':
        return cfg['host_gib']
    return min(MAX_HOST_GIB, math.floor(meminfo()['MemTotal'] / GIB) - RESERVE_GIB)


def cuda_home():
    return Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.4'))


def cuda_lib():
    return cuda_home() / 'targets/x86_64-linux/lib'


def p2p_enabled():
    """The nvidia module options that gds/enable-p2p.sh sets (static BAR1, no write-combined IO mappings)."""
    try:
        dwords = Path('/sys/module/nvidia/parameters/NVreg_RegistryDwords').read_text()
    except OSError:
        return False
    return 'RMForceStaticBar1=1' in dwords and (cuda_lib() / 'libcufile.so.0').exists()


def gds(cfg):
    return p2p_enabled() if cfg['gds'] == 'auto' else cfg['gds']


def mount_needs_shim(path):
    """cuFile requires data=ordered to be visible on ext4. The kernel prints data= only when it was passed at
    mount time, so an ext4 mount in the default (ordered) mode looks unsupported without the shim."""
    best, options = '', ''
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, _, right = line.partition(' - ')
        point = left.split()[4].replace('\\040', ' ')
        if (str(path) + '/').startswith(point.rstrip('/') + '/') and len(point) >= len(best):
            fstype, _, super_options = right.split()[:3]
            best, options = point, (fstype, left.split()[5] + ',' + super_options)
    return bool(options) and options[0] == 'ext4' and 'data=' not in options[1]


def runtime_dir(cfg):
    return Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')) / cfg['unit']


def cufile_config(cfg):
    """Render gds/cufile.json with a log directory; returns its path."""
    state = runtime_dir(cfg)
    state.mkdir(parents=True, exist_ok=True)
    out = state / 'cufile.json'
    out.write_text((ROOT / 'gds/cufile.json').read_text().replace('@CUFILE_LOG_DIR@', str(state)))
    return out


def command(cfg, socket):
    dist, assets = Path(cfg['build']) / 'dist', Path(cfg['assets'])
    k3 = cfg['experts'] == 'k3'
    k2 = cfg['experts'] == 'k2'
    mixed = cfg['experts'] in ('mixed', 'sero')
    cmd = [str(dist / 'ds41d'), 'serve', '--socket', str(socket),
           '--weights', cfg['weights'], '--receipt', str(assets / 'full-integrity.json'),
           '--auxiliary', str(assets / 'auxiliary.json'), '--engram', str(assets / 'engram.json'),
           '--rope', str(assets / 'rope.bin'), '--experts', str(assets / 'records.tsv'),
           '--engram-lib', str(dist / 'libds41_engram.so'), '--fused-head', str(dist / 'libds41_fuse_head.so'),
           '--placement', str(K3_PLACEMENT if k3 else K2_PLACEMENT if k2 else mixed_placement(cfg) if mixed else PLACEMENT),
           '--expert-slots', str(K3_EXPERT_SLOTS if k3 else K2_EXPERT_SLOTS if k2 else
                                 placement_slots(mixed_placement(cfg)) if mixed else EXPERT_SLOTS),
           '--host-gib', str(host_gib(cfg)), '--max-context', str(cfg['max_context'])] + ENGINE_FLAGS
    if k3:
        # the native manifest still supplies the drafter's experts; target layers come from the bank
        bank = Path(cfg['k3_bank'])
        cmd += ['--exl3-experts', str(bank / 'records-k3.tsv'), '--exl3-receipts', str(bank / 'receipts.json')]
    if k2 or mixed:
        # the first --spark-width is the profile's; one appended later (a lab A/B) still wins
        cmd[cmd.index('--spark-width') + 1] = K2_SPARK_WIDTH
    if k2:
        bank = Path(cfg['k2_bank'])
        cmd += ['--exl3-experts', str(bank / 'records-k2.tsv'), '--exl3-receipts', str(bank / 'receipts.json')]
    if mixed:
        bank = bank_dir(cfg)
        cmd += ['--exl3-experts', str(bank / 'records-mixed.tsv'), '--exl3-receipts', str(bank / 'receipts.json')]
    if cfg['experts'] == 'sero':
        # s0ma change 44: no RAM warm-up after a prompt. On a 48 GiB tier it read ~1,000 experts (~12 GiB) from the
        # drive before every 256+ row prompt's first token, 1.8 s each: first token 7.21 -> 6.37 s mean, decode
        # unchanged (38.86 / 39.14 tok/s), 41 GB less read from the drive in a bench
        for flag in ('--prefill-warm-rows', '--prefill-warm-gib'):
            if flag in cmd:
                at = cmd.index(flag)
                del cmd[at:at + 2]
        # s0ma change 45: no CUDA graphs. ds41d captures a graph for nearly every new shape (rows, layer, scan bucket,
        # position mod 8) and here the GPU mostly waits on the host and the drive, so capturing costs more than
        # replaying saves. Two runs each (Sero, no warm-up): decode 38.32 / 39.19 -> 39.68 / 39.69 tok/s, verify
        # 58.0 / 56.9 -> 55.2 ms a step, first token 6.43 / 6.37 -> 5.59 s mean (an 8K prompt 15.1 -> 10.2 s)
        if '--graphs' in cmd:
            cmd.remove('--graphs')
    return cmd


def parked_gib(cfg, env):
    """s0ma change 91: the VRAM of the parked live slots (DS41_LIVE_SLOTS - 1 of them, each DS41_SLOT_CONTEXT tokens:
    356 B a token at layer 20, 178 B at layers 2, 8, 14, 2.8 MB of SWA rings)."""
    try:
        slots = max(1, min(8, int(env.get('DS41_LIVE_SLOTS', '1'))))
        cap = int(env.get('DS41_SLOT_CONTEXT') or min(cfg['max_context'], 262144))
    except ValueError:
        return 0.0
    per = cap // 2 * 356 * 3 + cap * 356 + 40 * 134 * 528 + 4 * 2 * 8 * 512 * 4
    return (slots - 1) * per / 2**30


def patch_gib(env):
    """s0ma change 99: the device memory a trained drafter patch takes (DS41_SPARK_PATCH=DIR, its patch.safetensors),
    taken from the expert slots like the parked slots' arenas, plus 0.1 GiB of slack."""
    d = env.get('DS41_SPARK_PATCH', '')
    if not d or d == '0':
        return 0.0
    try:
        return Path(d, 'patch.safetensors').stat().st_size / 2**30 + 0.1
    except OSError:
        return 0.0


def config_reserve_gib(cfg):
    """s0ma change 112: config.local.json's vram_reserve_gib (ds41.sh serve --embedder-gpu writes the embedder's
    measured VRAM there), 0 when absent."""
    try:
        return max(0.0, float(cfg.get('vram_reserve_gib') or 0))
    except (TypeError, ValueError):
        raise SystemExit('vram_reserve_gib in config.local.json must be a number of GiB')


def reserve_gib(env):
    """s0ma change 111: VRAM another process holds beside ds41d (s0ma's embedder on the GPU: DS41_VRAM_RESERVE_GIB), or
    the lab's stand-in for it (DS41_VRAM_BALLAST_GIB, which ds41d allocates itself), taken from the expert slots so
    prompts keep the free VRAM they had."""
    total = 0.0
    for k in ('DS41_VRAM_RESERVE_GIB', 'DS41_VRAM_BALLAST_GIB'):
        try:
            total += max(0.0, float(env.get(k) or 0))
        except ValueError:
            raise SystemExit(f'{k} must be a number of GiB')
    return total


def graphs_gib(env):
    """s0ma change 103: the device memory DS41_GRAPHS=fixed|all takes (the cuBLAS handles' fixed 32 MiB workspace and
    ~400 instantiated graphs: 176 MiB less free at the first prefill, measured 2026-11-04), taken from the expert slots
    so the prompts' scratch keeps what it had (without it an 8K prompt was prefilled in two pieces, +1.2 s)"""
    return 0.25 if env.get('DS41_GRAPHS', '') in ('fixed', 'all') else 0.0


# s0ma change 105: the drafter's starved experts (global ids; change 80's audit of the weights)
SPARK_STARVED = '15361,15370,15372,15378,15385,15400,15404,15451,15454,15463,15476,15487,15488,15490,15496,15497,15501,15502,15503,15504,15505,15521,15528,15536,15547,15555,15557,15560,15572,15574,15575,15576,15580,15581,15587,15592,15593,15594,15605,15607,15616,15618,15619,15620,15623,15624,15634,15638,15639,15641,15642,15644,15645,15646,15647,15648,15649,15650,15652,15653,15657,15658,15666,15671,15672,15673,15676,15678,15679,15680,15681,15684,15686,15687,15688,15689,15691,15692,15702,15705,15706,15708,15711,15715,15716,15720,15721,15722,15723,15724,15729,15731,15733,15737,15739,15740,15741,15742'


def cold_gib(env):
    """s0ma change 105: with DS41_SPARK_COLD the first prefill had 54 MiB less free (2,226 against 2,280 MiB) and the
    7,999-row prompt was prefilled in two pieces (+0.8 s); 0.1 GiB of the expert slots' extra goes back to it"""
    return 0.1 if env.get('DS41_SPARK_COLD') else 0.0


def environment(cfg, render=True):
    dist = Path(cfg['build']) / 'dist'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS41_', 'CUFILE_'))}
    env.pop('LD_PRELOAD', None)
    env.update(ENGINE_ENV)
    if config_reserve_gib(cfg):   # s0ma change 112: serving beside the embedder on the GPU; a lab --env still wins
        env.setdefault('DS41_VRAM_RESERVE_GIB', f'{config_reserve_gib(cfg):.2f}')
    if cfg.get('experts') in ('k2', 'mixed', 'sero'):
        env.update(K2_ENV)
    if cfg.get('experts') == 'sero':
        env.update(SERO_ENV)
        # s0ma change 48: Sero's own dense tensors (change 34: 38.2 -> 40.1 tok/s) and K4 head (change 43: 0.99 GB
        # to expert slots, drafting 3.19 -> 2.64 ms a step) whenever the bank has them (ds41.sh sero dense); a lab
        # --env DS41_SERO_DENSE=0 or DS41_SERO_HEAD=0 (in ENGINE_ENV, applied first) still wins
        if (bank_dir(cfg) / 'dense' / 'dense.json').exists():
            env.setdefault('DS41_SERO_DENSE', '1')
            env.setdefault('DS41_SERO_HEAD', '1')
        # s0ma change 50: the drafter's routed experts from its EXL3 bank (change 49, ds41.sh mtp quantize) whenever one
        # is complete, K2 before K3. Drafts are verified, so only acceptance can move; the freed VRAM goes to target
        # slots. Seed 21 at a 46 GiB tier: 37.29 -> 38.65 tok/s with K3 (drafter 6,892 -> 4,882 MiB, NVMe misses -6.1%,
        # acceptance 0.727 -> 0.735), 40.59 with K2 (3,262 MiB, misses -10.6%, acceptance 0.757). A lab
        # --env DS41_MTP_EXL3=0 (native MXFP4 drafter) or =DIR still wins
        for k in (2, 3):
            mtp = ROOT / 'models' / f'mtp-k{k}'
            if (mtp / 'records-mtp.tsv').exists() and (mtp / 'receipts.json').exists():
                env.setdefault('DS41_MTP_EXL3', str(mtp))
                break
        # s0ma change 106: CUDA graphs for the fixed groups (change 100) with programmatic dependent launch. Measured
        # 2026-11-04, Sero, two seeds: verify 26.85 -> 26.3-26.4 ms a step in all four graph runs, 79.1-79.6 tok/s
        # against 77.8-80.2 plain; first token as plain once change 103 kept the prompts' VRAM (1.116 s). lab
        # concurrent and lab probes with both on: streams together give the same greedy tokens as alone, probes alone vs
        # together identical, nothing failed. A lab --env DS41_GRAPHS=0 / DS41_PDL=0 turns either off
        env.setdefault('DS41_GRAPHS', 'fixed')
        env.setdefault('DS41_PDL', '1')
        # s0ma change 105: the drafter's 98 starved experts (change 80's weight audit) read in place from host memory
        # instead of pinned in VRAM (change 81); the freed ~0.87 GiB goes to target slots. Measured 2026-11-04
        # (route counts): 0 of 28,800 drafter routes reach them. Sero, two seeds: 80.93 tok/s against 77.8-80.2 plain,
        # verify 26.85 -> 26.4 ms a step, acceptance 0.755. Needs the EXL3 drafter bank. A lab --env DS41_SPARK_COLD=0
        # turns it off
        if env.get('DS41_MTP_EXL3', '0') not in ('', '0'):
            env.setdefault('DS41_SPARK_COLD', SPARK_STARVED)
        if env.get('DS41_SPARK_COLD') == '0':   # s0ma change 107: a lab --env DS41_SPARK_COLD=0 turns it off
            env.pop('DS41_SPARK_COLD')
        # s0ma change 52 (V6): 2 GiB more K3 slots in 4 chunks that each prompt borrows back for its scratch (change 51).
        # Seed 22, 46 GiB tier: 40.43 -> 41.88 tok/s, tokens 7/7 identical, verify 57.4 -> 55.3 ms a step; first
        # token 5.65 -> 6.13 s mean (all of it inside the engine's prefill; lending itself 3-10 ms). A lab
        # --env DS41_PREFILL_LEND_GIB=0 turns it off
        env.setdefault('DS41_PREFILL_LEND_GIB', '2')
        # s0ma change 58: up to 8 GiB a prompt of the experts its last 256 rows routed and its sweep read from the drive,
        # into the RAM tier as they pass (change 57). Seed 22, 48 GiB tier: 41.2-42.8 -> 45.07 tok/s (three runs of
        # the build before), NVMe misses 7,842-7,879 -> 6,527, verify 54.1 -> 51.0 ms a step, tokens 7/7 identical.
        # A lab --env DS41_PROMPT_ADMIT_GIB=0 turns it off
        env.setdefault('DS41_PROMPT_ADMIT_GIB', '8')
        # change 25: a lab --env DS41_EXPERT_EXTRA_GIB=... (in ENGINE_ENV) wins
        # s0ma change 91: the parked conversations' arenas come out of these slots, so prefill keeps the free VRAM it
        # had (1 GiB less free pushed 8K prompts into two segments, change 25)
        extra = float(SERO_EXTRA_GIB if cfg['max_context'] <= SERO_MAX_CONTEXT else SERO_EXTRA_GIB_1M)
        extra = extra - parked_gib(cfg, env) - patch_gib(env) - graphs_gib(env) - cold_gib(env) - reserve_gib(env)
        # s0ma change 115: below zero (the embedder's reserve at a 1M context) ds41d takes the rest from the slots Sero's
        # dense set and the drafter gave back; clamped to zero it came out of free VRAM, and at 1M there is none: OOM
        if extra < 0:
            print(f'run.py: {-extra:.2f} GiB of the reserve is beyond run.py\'s extra expert slots; ds41d takes it from '
                  'the slots its dense set and drafter give back', file=sys.stderr)
        env.setdefault('DS41_EXPERT_EXTRA_GIB', f'{extra:.2f}')
    # s0ma change 79: the conversations kept in RAM (change 73) sized to the window: a saved conversation holds ~0.9 KB a
    # token (the four global layers' compressed KV and indexer keys up to its frontier, 890 B, and its ids) and 3 MB
    # of rings, so s0ma's main turn and its workers at twice max_context in all need 2 KiB a token of it, plus 1 GiB.
    # 1 GiB up to 262,144 tokens (as before); 3 GiB at 1,048,576. A lab --env DS41_CONVERSATIONS_GIB=... still wins
    env.setdefault('DS41_CONVERSATIONS_GIB',
                   str(max(1, -(-2 * cfg['max_context'] * 1024 // 2**30) + 1) if cfg['max_context'] > 262144 else 1))
    # the pinned cuBLAS pair (dist/blas) before CUDA 13.4's own libraries (cudart, cuFile)
    env['LD_LIBRARY_PATH'] = f'{dist}:{dist}/blas:{cuda_lib()}'
    # s0ma change 81: the conversations saved to disk (ds41d's server: a restarted engine reads none of a known prompt
    # again), keyed by a tag of everything that shapes a conversation's state; a lab --env DS41_CONVERSATIONS_* wins
    # (the lab turns the disk off: its benchmark prompts must be read cold every run)
    env.setdefault('DS41_CONVERSATIONS_DIR', str(ROOT / 'conversations'))
    env.setdefault('DS41_CONVERSATIONS_DISK_GIB', '32')
    if gds(cfg):
        env.update(GDS_ENV)
        env['CUFILE_ENV_PATH_JSON'] = str(cufile_config(cfg) if render else runtime_dir(cfg) / 'cufile.json')
        if mount_needs_shim(Path(cfg['weights'])):
            env['LD_PRELOAD'] = str(dist / 'mountinfo_shim.so')
    env['DS41_CONVERSATIONS_TAG'] = conversations_tag(cfg, env)
    return env


def conversations_tag(cfg, env):
    """s0ma change 81: a digest of what shapes a saved conversation's bytes: the build's files (size and mtime: a
    rebuild changes it), the engine's arguments but the socket and the RAM tier, each file or bank they name, and
    every DS41_* setting but the disk cache's own. A file of another tag is never loaded."""
    import hashlib

    def stamp(value):
        q = Path(value)
        try:
            if q.is_dir():
                r = q / 'receipts.json'
                q = r if r.exists() else q
            st = q.stat()
            return f'{q}:{st.st_size}:{st.st_mtime_ns}'
        except (OSError, ValueError):
            return None

    parts = []
    dist = Path(cfg['build']) / 'dist'
    if dist.is_dir():
        parts += [stamp(f) for f in sorted(dist.rglob('*')) if f.is_file()]
    cmd, skip, i = command(cfg, 'socket'), {'--socket', '--host-gib'}, 0
    while i < len(cmd):
        if cmd[i] in skip:
            i += 2
            continue
        parts.append(cmd[i])
        if str(cmd[i]).startswith('/'):
            parts.append(stamp(cmd[i]))
        i += 1
    for k in sorted(env):
        if k.startswith('DS41_') and not k.startswith('DS41_CONVERSATIONS'):
            parts.append(f'{k}={env[k]}')
            if str(env[k]).startswith('/'):
                parts.append(stamp(env[k]))
    return hashlib.sha256('\n'.join(p for p in parts if p).encode()).hexdigest()


def live_slots(cfg):
    try:
        return max(1, min(8, int(environment(cfg, render=False).get('DS41_LIVE_SLOTS', '1'))))
    except (ValueError, OSError):
        return 1


def probes(cfg):
    """s0ma change 96: one-token requests the API admits beside the live slots (ds41d's probe groups); 0: off."""
    try:
        env = environment(cfg, render=False)
        if str(env.get('DS41_PROBES', '1')).startswith('0') or str(env.get('DS41_PROBE_SAVE', '0')).startswith('1'):
            return 0
        return max(1, min(32, int(env.get('DS41_PROBES_MAX', '16'))))
    except (ValueError, OSError):
        return 0


def api_command(cfg, socket):
    return [str(ROOT / '.venv/bin/python'), '-m', 'ds41_api.server', '--backend', 'socket', '--socket', str(socket),
            '--tokenizer', str(Path(cfg['weights']) / 'tokenizer.json'), '--model', cfg['model_id'],
            '--host', '127.0.0.1', '--port', str(cfg['port']),
            # s0ma change 91: as many requests reach the engine as it keeps live slots (it queues the rest itself)
            '--max-concurrent', str(live_slots(cfg)),
            '--max-probes', str(probes(cfg)),   # s0ma change 96
            '--max-queue', str(cfg['max_queue']), '--max-tokens-limit', str(cfg['max_tokens_limit'])]


def preflight(cfg):
    if sys.platform != 'linux':
        raise RuntimeError('This profile requires Linux, NVIDIA CUDA and cgroup v2')
    dist = Path(cfg['build']) / 'dist'
    if not (dist / 'ds41d').exists():
        raise RuntimeError('Build first: scripts/build.sh')
    if not (ROOT / '.venv/bin/python').exists():
        raise RuntimeError('Create the API environment first: scripts/setup.sh')
    if cfg['experts'] == 'k3' and not (Path(cfg['k3_bank']) / 'records-k3.tsv').exists():
        raise RuntimeError(f'No 3-bit expert bank at {cfg["k3_bank"]}: run scripts/k3_bank.py fetch {cfg["k3_bank"]}')
    if cfg['experts'] == 'k2' and not K2_PLACEMENT.exists():
        raise RuntimeError(f'No {K2_PLACEMENT.name}: run scripts/k2_bank.py placement')
    if cfg['experts'] == 'sero' and not (bank_dir(cfg) / 'records-mixed.tsv').exists():
        raise RuntimeError(f'No sero bank at {cfg["sero_bank"]}: run scripts/sero_bank.py fetch {cfg["sero_bank"]}')
    if cfg['experts'] == 'mixed' and not (Path(cfg['mixed_bank']) / 'records-mixed.tsv').exists():
        raise RuntimeError(f'No mixed bank at {cfg["mixed_bank"]}: run scripts/mixed_bank.py build {cfg["mixed_bank"]} '
                           '--k2 K2_BANK --k3 K3_BANK')
    if cfg['experts'] == 'k2' and not (Path(cfg['k2_bank']) / 'records-k2.tsv').exists():
        raise RuntimeError(f'No 2-bit expert bank at {cfg["k2_bank"]}: run scripts/k2_bank.py fetch {cfg["k2_bank"]}')
    aux = json.loads((Path(cfg['assets']) / 'auxiliary.json').read_text())
    if aux['weight_root'] != cfg['weights']:
        raise RuntimeError(f'The assets were prepared for {aux["weight_root"]}; run scripts/build.sh again')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1 or 'RTX PRO 6000 Blackwell' not in gpu[0] or int(gpu[0].split(',')[1]) < 97000:
        raise RuntimeError('This measured profile targets exactly one 96 GB RTX PRO 6000 Blackwell')
    # s0ma change 112: a process ds41d serves beside (s0ma's embedder: config vram_reserve_gib) may hold its reserve
    allow = 2048 + int(1024 * config_reserve_gib(cfg))
    if int(gpu[0].split(',')[2]) > allow:
        raise RuntimeError(f'The GPU is in use beyond {allow} MiB; stop other GPU processes (the profile needs nearly '
                           'all 96 GB)')
    tier, mem = host_gib(cfg), meminfo()
    if tier < MIN_HOST_GIB:
        raise RuntimeError(f'The expert placement needs a RAM tier of at least {MIN_HOST_GIB} GiB '
                           f'(this machine allows {tier}): 128 GB of RAM is the minimum')
    if mem['MemAvailable'] < (tier + 4) * GIB:
        raise RuntimeError(f'Need {tier + 4} GiB available RAM for a {tier} GiB pinned expert tier; '
                           f'{mem["MemAvailable"] / GIB:.1f} GiB is available')
    if not Path('/sys/fs/cgroup/cgroup.controllers').exists():
        raise RuntimeError('cgroup v2 is required to enforce MemorySwapMax=0')
    mode = ('GDS P2P' + (' + mountinfo shim' if mount_needs_shim(Path(cfg['weights'])) else '')) if gds(cfg) \
        else 'host bounce (GDS off; see docs/gds.md)'
    print(f'Preflight passed: idle RTX PRO 6000, {tier} GiB RAM tier, NVMe reads: {mode}.', flush=True)


def evict_page_cache(path):
    """Drop the checkpoint's clean page cache before pinning the RAM tier, so pinning does not start with
    reclaim pressure (systemd-oomd acts on sustained pressure)."""
    for root, _, files in os.walk(path):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def serve(cfg):
    cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    if (cgroup / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Refusing to pin experts without effective memory.swap.max=0')
    socket = runtime_dir(cfg) / 'engine.sock'
    socket.parent.mkdir(parents=True, exist_ok=True)
    socket.unlink(missing_ok=True)
    evict_page_cache(cfg['weights'])
    if cfg['experts'] == 'k3':
        evict_page_cache(cfg['k3_bank'])
    if cfg['experts'] == 'k2':
        evict_page_cache(cfg['k2_bank'])
    argv = command(cfg, socket)
    print(shlex.join(argv), flush=True)
    engine = subprocess.Popen(argv, env=environment(cfg))
    api = None

    def stop(*_):
        for p in (api, engine):
            if p and p.poll() is None:
                p.terminate()
        for p in (api, engine):
            if p:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = time.time() + 900
    while not socket.exists():
        if engine.poll() is not None:
            raise SystemExit(f'ds41d exited while loading (rc={engine.returncode})')
        if time.time() > deadline:
            stop()
        time.sleep(1)
    argv = api_command(cfg, socket)
    print(shlex.join(argv), flush=True)
    api = subprocess.Popen(argv, cwd=ROOT / 'api')
    while True:
        for name, p in (('ds41d', engine), ('api', api)):
            if p.poll() is not None:
                print(f'{name} exited rc={p.returncode}; stopping the service', flush=True)
                stop()
        time.sleep(2)


def wait_ready(cfg):
    url = f'http://127.0.0.1:{cfg["port"]}/health'
    for _ in range(1200):
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', cfg['unit']], check=False).returncode:
            raise RuntimeError('Server exited; run scripts/run.py logs')
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print(f'Ready: http://127.0.0.1:{cfg["port"]}/v1  model {cfg["model_id"]}', flush=True)
                    return
        except (OSError, TimeoutError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError('Readiness timed out; inspect logs (the service is still managed by systemd)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'stop', 'status', 'logs', 'command', 'preflight', 'serve', 'paths'])
    args = p.parse_args()
    cfg = configuration()
    if args.action == 'command':
        env = environment(cfg, render=False)
        socket = runtime_dir(cfg) / 'engine.sock'
        keys = sorted(k for k in env if k.startswith(('DS41_', 'CUFILE_'))) + ['LD_LIBRARY_PATH'] + \
            (['LD_PRELOAD'] if 'LD_PRELOAD' in env else [])
        print(' '.join(f'{k}={shlex.quote(env[k])}' for k in keys))
        print(shlex.join(command(cfg, socket)))
        print(shlex.join(api_command(cfg, socket)))
    elif args.action == 'paths':
        # for scripts/build.sh: the configured checkpoint, assets and build directories
        print('\n'.join((cfg['weights'], cfg['assets'], cfg['build'])))
    elif args.action == 'preflight':
        preflight(cfg)
    elif args.action == 'start':
        preflight(cfg)
        # The pinned tier plus about 2 GiB of process memory; page cache fills the rest up to the limit.
        limit = host_gib(cfg) + 12
        subprocess.run(['systemd-run', '--user', '--collect', '--unit', cfg['unit'],
                        f'--property=MemoryMax={limit}G', '--property=MemorySwapMax=0',
                        '--property=KillMode=control-group', '--property=TimeoutStopSec=60',
                        f'--setenv=CUDA_HOME={cuda_home()}',
                        '--', sys.executable, str(Path(__file__).resolve()), 'serve'], check=True)
        wait_ready(cfg)
    elif args.action == 'serve':
        serve(cfg)
    elif args.action == 'logs':
        subprocess.run(['journalctl', '--user', '-u', cfg['unit'], '-n', '100', '--no-pager'], check=True)
    else:
        subprocess.run(['systemctl', '--user', args.action, cfg['unit']], check=False)


if __name__ == '__main__':
    main()
