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
# The measured host RAM tier. More RAM lowers NVMe reads (96 -> 104 GiB: +3.3% decode); larger tiers are untested.
MAX_HOST_GIB = 104
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
    return cmd


def environment(cfg, render=True):
    dist = Path(cfg['build']) / 'dist'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS41_', 'CUFILE_'))}
    env.pop('LD_PRELOAD', None)
    env.update(ENGINE_ENV)
    if cfg.get('experts') in ('k2', 'mixed', 'sero'):
        env.update(K2_ENV)
    if cfg.get('experts') == 'sero':
        env.update(SERO_ENV)
        # change 25: a lab --env DS41_EXPERT_EXTRA_GIB=... (in ENGINE_ENV) wins
        env.setdefault('DS41_EXPERT_EXTRA_GIB', SERO_EXTRA_GIB if cfg['max_context'] <= SERO_MAX_CONTEXT else SERO_EXTRA_GIB_1M)
    # the pinned cuBLAS pair (dist/blas) before CUDA 13.4's own libraries (cudart, cuFile)
    env['LD_LIBRARY_PATH'] = f'{dist}:{dist}/blas:{cuda_lib()}'
    if gds(cfg):
        env.update(GDS_ENV)
        env['CUFILE_ENV_PATH_JSON'] = str(cufile_config(cfg) if render else runtime_dir(cfg) / 'cufile.json')
        if mount_needs_shim(Path(cfg['weights'])):
            env['LD_PRELOAD'] = str(dist / 'mountinfo_shim.so')
    return env


def api_command(cfg, socket):
    return [str(ROOT / '.venv/bin/python'), '-m', 'ds41_api.server', '--backend', 'socket', '--socket', str(socket),
            '--tokenizer', str(Path(cfg['weights']) / 'tokenizer.json'), '--model', cfg['model_id'],
            '--host', '127.0.0.1', '--port', str(cfg['port']), '--max-concurrent', '1',
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
    if int(gpu[0].split(',')[2]) > 2048:
        raise RuntimeError('The GPU is in use; stop other GPU processes (the profile needs nearly all 96 GB)')
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
