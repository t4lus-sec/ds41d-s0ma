#!/usr/bin/env python3
"""0xSero's EXL3 2.7 bpw quant of DeepSeek-V4.1-Flash-1017 as a ds41d expert bank for `"experts": "sero"` (s0ma
change 15): a known-good quality baseline (its card: mean KL 0.0304 nats against native 1017 on sudoingX's
199-conversation panel, top-1 agreement 94.95%).

Its routed experts have a width each (exl3_moe_plan.json: 6,278 K2, 7,447 K3, 1,600 K4, 35 K5; mul1 codebook). This
reads each expert's nine tensors (w1/w3/w2 trellis, suh, svh) with HTTP range reads into DIR/bank-LL.bin, one record
per expert in the K2/K3 banks' layout at its own width (8.9 / 13.3 / 17.7 / 22.2 MB), packed in expert order.
Only the routed experts are taken; ds41d keeps every other tensor (attention, shared experts, head, drafter) native,
where the quant has K4/K5 for them, so the bank is Sero's experts, not the whole quant.

  inspect [--source REPO[@REV]]     tensor names, dtypes and shapes of layer 0's first experts (no weights)
  fetch DIR [--source REPO[@REV]] [--layers 0-39] [--threads 16]
                                    ~186 GB; resumes (layers with a matching receipt are kept); then writes
                                    DIR/records-mixed.tsv, DIR/receipts.json, DIR/mixed-p80.placement, DIR/layers.json
  verify DIR                        recompute each bank file's SHA-256 and re-stamp its receipt

The revision defaults to the repo's current commit, which is then pinned in every receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_bank as kb  # noqa: E402
import k2_bank  # noqa: E402

REPO = '0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw'
ORDER = kb.ORDER
LAYOUTS = {b: k2_bank.layout(b) for b in (2, 3, 4, 5)}   # bits -> (views, record bytes)
K2_REC = LAYOUTS[2][1]
ALIASES = {'w1': ('w1', 'gate_proj', 'gate'), 'w3': ('w3', 'up_proj', 'up'), 'w2': ('w2', 'down_proj', 'down')}


def revision(repo):
    req = urllib.request.Request(f'https://huggingface.co/api/models/{repo}', headers=kb._auth())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())['sha']


def open_repo(source):
    name, _, rev = source.partition('@')
    rev = rev or revision(name)
    if len(rev) != 40:
        raise SystemExit(f'--source {source}: want REPO or REPO@<40-hex commit>')
    repo = kb.Repo(name, rev)
    plan = json.loads(repo._read('exl3_moe_plan.json'))
    if plan.get('codebook') != 'mul1':
        raise SystemExit(f'{repo.name}: codebook {plan.get("codebook")!r}; ds41d decodes mul1 only')
    widths = {int(l): [int(c) for c in v['k']] for l, v in plan['layers'].items()}
    if sorted(widths) != list(range(40)) or any(len(w) != 384 or set(w) - {2, 3, 4, 5} for w in widths.values()):
        raise SystemExit(f'{repo.name}: exl3_moe_plan.json is not 40 layers x 384 experts of widths 2..5')
    return repo, f'{name}@{rev}', widths


def prefix(repo, layer, e, w):
    """the tensor name prefix of expert e's projection w in this quant (exllamav3 keeps the checkpoint's names)"""
    for base in (f'layers.{layer}.ffn.experts.{e}', f'model.layers.{layer}.ffn.experts.{e}',
                 f'model.layers.{layer}.mlp.experts.{e}', f'layers.{layer}.mlp.experts.{e}'):
        for alias in ALIASES[w]:
            if f'{base}.{alias}.trellis' in repo.index:
                return f'{base}.{alias}'
    near = [k for k in repo.index if f'.{layer}.' in k and 'expert' in k][:12]
    raise SystemExit(f'{repo.name}: no trellis for layer {layer} expert {e} {w}; names near it: {near}')


def expert_record(repo, layer, e, bits):
    views, rec = LAYOUTS[bits]
    spans, record = [], bytearray(rec)
    for w, name, (k, n) in ORDER:
        p = prefix(repo, layer, e, w)
        if repo.meta(f'{p}.mcg') is not None:
            raise ValueError(f'{repo.name} {p}: the mcg codebook (ds41d decodes mul1 only)')
        want = {'trellis': ('I16', [k // 16, n // 16, 16 * bits]), 'suh': ('F16', [k]), 'svh': ('F16', [n])}
        dest = {'trellis': views[name][0], 'suh': views[name + '_scale'][0], 'svh': views[name + '_scale'][0] + 2 * k}
        for t, (dtype, shape) in want.items():
            found = repo.meta(f'{p}.{t}')
            if found is None:
                raise ValueError(f'{repo.name} {p}.{t} missing')
            shard, entry = found
            if entry['dtype'] != dtype or entry['shape'] != shape:
                raise ValueError(f'{repo.name} {p}.{t}: {entry["dtype"]} {entry["shape"]}, want {dtype} {shape} '
                                 f'(K{bits}, as exl3_moe_plan.json says)')
            a, b = entry['data_offsets']
            base = repo.header(shard)['__base__']
            spans.append((shard, base + a, base + b, dest[t]))
    spans.sort()
    i = 0
    while i < len(spans):  # one range read for tensors less than 1 MiB apart in the same shard
        j = i
        while j + 1 < len(spans) and spans[j + 1][0] == spans[i][0] and spans[j + 1][1] - spans[j][2] <= 1 << 20:
            j += 1
        raw = kb.get(repo.base + spans[i][0], spans[i][1], spans[j][2] - 1)
        for shard, a, b, d in spans[i:j + 1]:
            record[d:d + b - a] = raw[a - spans[i][1]:b - spans[i][1]]
        i = j + 1
    return bytes(record)


def offsets(widths):
    """each expert's offset in its layer's bank file (records packed in expert order; all are 4 KiB multiples)"""
    out, off = [], 0
    for b in widths:
        out.append(off)
        off += LAYOUTS[b][1]
    return out, off


def receipt(path, digest, src, widths):
    st = os.stat(path)
    return {'file': path.name, 'bytes': st.st_size, 'sha256': digest, 'mtime_ns': st.st_mtime_ns, 'source': src,
            'widths': ''.join(map(str, widths)), 'verified': time.strftime('%Y-%m-%dT%H:%M:%S%z')}


def current(out, receipts, layer, src=None, widths=None):
    r, path = receipts.get(str(layer)), out / f'bank-{layer:02d}.bin'
    if not r or not path.exists() or len(r.get('sha256', '')) != 64 or not r.get('widths'):
        return False
    if src is not None and r.get('source') != src:
        return False
    if widths is not None and r['widths'] != ''.join(map(str, widths)):
        return False
    st = os.stat(path)
    return st.st_size == offsets([int(c) for c in r['widths']])[1] and st.st_mtime_ns == r.get('mtime_ns')


def manifest(out, receipts):
    lines = ['# DS41MOE01 EXL3 mixed-width bank (scripts/sero_bank.py): 0xSero 2.7 bpw routed experts, mul1; '
             'gate/up/down -> trellis, *_scale -> suh then svh']
    for layer in range(40):
        widths = [int(c) for c in receipts[str(layer)]['widths']]
        offs, total = offsets(widths)
        path = (out / f'bank-{layer:02d}.bin').resolve()
        for e, (b, off) in enumerate(zip(widths, offs)):
            v, rec = LAYOUTS[b]
            eid = layer * 384 + e
            lines.append('R %d %d %d %d %d %d %d %d %d %d %d %d %d %d' % (
                eid, rec, *v['gate'], *v['gate_scale'], *v['up'], *v['up_scale'], *v['down'], *v['down_scale']))
            lines.append('E %d "%s" %d %d 0 %d %d' % (eid, path, off, rec, rec, total))
    (out / 'records-mixed.tsv').write_text('\n'.join(lines) + '\n')


def placement(out, receipts, root=Path(__file__).resolve().parents[1]):
    """k2-p80 in the same VRAM bytes for this bank: pins as k2-p80 (each at its own width); the bytes left go to the
    layers as k2-p80's dynamic budgets do, each layer's budget in slots of its unpinned experts' average size (ds41d
    splits it between the widths in proportion to those experts, so the bytes match)."""
    src = root / 'profiles/k2-p80.placement'
    if not src.exists():
        k2_bank.placement(root)
    lines = src.read_text().splitlines()
    width = {}
    for layer in range(40):
        for e, c in enumerate(receipts[str(layer)]['widths']):
            width[layer * 384 + e] = int(c)
    pins = [int(x.split()[1]) for x in lines if x.startswith('vram ') and int(x.split()[1]) < 15360]
    pinned = set(pins)
    budget = {int(x.split()[1]): int(x.split()[2]) for x in lines if x.startswith('budget ') and int(x.split()[1]) < 40}
    total = sum(budget.values()) + len(pins)
    vram = total * K2_REC                                   # k2-p80's expert VRAM, in bytes
    left = vram - sum(LAYOUTS[width[e]][1] for e in pins)
    new, slots = {}, len(pins)
    for layer, b in budget.items():
        free = [LAYOUTS[width[layer * 384 + e]][1] for e in range(384) if layer * 384 + e not in pinned]
        avg = sum(free) / len(free)
        new[layer] = min(len(free), int(0.995 * left * b / sum(budget.values()) / avg))
        slots += new[layer]
    out_lines = [f'# DS41PLACEMENT01 sero-p80 (scripts/sero_bank.py): k2-p80 in the same VRAM bytes ({vram / 2**30:.2f} '
                 'GiB) for 0xSero\'s 2.7 bpw experts;', f'# {slots:,} target slots in all.']
    for x in lines:
        if x.startswith('#'):
            continue
        if x.startswith('budget ') and int(x.split()[1]) < 40:
            x = f'budget {x.split()[1]} {new[int(x.split()[1])]}'
        out_lines.append(x)
    (out / 'mixed-p80.placement').write_text('\n'.join(out_lines) + '\n')
    return slots


def finish(out, receipts, src):
    manifest(out, receipts)
    slots = placement(out, receipts)
    hist = {b: sum(r['widths'].count(str(b)) for r in (receipts[str(l)] for l in range(40))) for b in (2, 3, 4, 5)}
    gb = sum(receipts[str(l)]['bytes'] for l in range(40)) / 1e9
    (out / 'layers.json').write_text(json.dumps({'source': src, 'widths': hist, 'expert_slots': slots,
                                                 'k3_layers': []}) + '\n')
    print(f'bank complete: {out.resolve()}/records-mixed.tsv; experts K2 {hist[2]:,}, K3 {hist[3]:,}, K4 {hist[4]:,}, '
          f'K5 {hist[5]:,}; {gb:.1f} GB; {slots:,} VRAM slots in k2-p80\'s bytes', flush=True)


def fetch(out, layers, source, threads):
    out.mkdir(parents=True, exist_ok=True)
    repo, src, widths = open_repo(source)
    receipts, t0, fetched = kb.load_receipts(out), time.time(), 0
    for layer in layers:
        if current(out, receipts, layer, src, widths[layer]):
            continue
        offs, total = offsets(widths[layer])
        path, tmp = out / f'bank-{layer:02d}.bin', out / f'bank-{layer:02d}.partial'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, total)

            def one(e):
                os.pwrite(fd, expert_record(repo, layer, e, widths[layer][e]), offs[e])
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(one, range(384)))
            os.fsync(fd)
        finally:
            os.close(fd)
        digest = kb.file_sha256(tmp)
        os.replace(tmp, path)
        receipts[str(layer)] = receipt(path, digest, src, widths[layer])
        kb.save_receipts(out, receipts)
        fetched += total
        rate = fetched / 1e6 / max(1e-9, time.time() - t0)
        mix = {b: widths[layer].count(b) for b in (2, 3, 4, 5) if widths[layer].count(b)}
        print(json.dumps({'layer': layer, 'GB': round(total / 1e9, 2), 'widths': mix, 'sha256': digest[:16],
                          'MB_s': round(rate, 1)}), flush=True)
    missing = [l for l in range(40) if not current(out, receipts, l, src, widths[l])]
    if missing:
        print(f'layers missing: {missing}', flush=True)
        return
    finish(out, receipts, src)


def verify(out):
    receipts = kb.load_receipts(out)

    def check(layer):
        path, r = out / f'bank-{layer:02d}.bin', receipts.get(str(layer))
        if not path.exists() or not r or not current(out, receipts, layer):
            return layer, None
        before = os.stat(path).st_mtime_ns
        digest = kb.file_sha256(path)
        return layer, digest if os.stat(path).st_mtime_ns == before else None
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(check, range(40)))
    bad = [layer for layer, d in results if d is None]
    if bad:
        raise SystemExit(f'verify failed for layers {bad}; run fetch to rebuild them')
    changed = [layer for layer, d in results if receipts[str(layer)].get('sha256') != d]
    if changed:
        raise SystemExit(f'layers {changed} differ from their receipts\' digests: run fetch to rebuild them')
    for layer, d in results:
        r = receipts[str(layer)]
        receipts[str(layer)] = receipt(out / f'bank-{layer:02d}.bin', d, r['source'], [int(c) for c in r['widths']])
    kb.save_receipts(out, receipts)
    finish(out, receipts, receipts['0']['source'])


def inspect(source):
    repo, src, widths = open_repo(source)
    print(f'{src}: plan widths layer 0: ' + ''.join(map(str, widths[0][:48])) + '...')
    for e in range(3):
        for w, _, _ in ORDER:
            p = prefix(repo, 0, e, w)
            for t in ('trellis', 'suh', 'svh', 'mul1', 'mcg'):
                found = repo.meta(f'{p}.{t}')
                if found:
                    print(f'  {p}.{t}: {found[1]["dtype"]} {found[1]["shape"]} ({found[0]})')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['inspect', 'fetch', 'verify'])
    p.add_argument('dir', type=Path, nargs='?')
    p.add_argument('--source', default=REPO, help='REPO or REPO@REVISION (default: the repo\'s current commit)')
    p.add_argument('--layers', default='0-39')
    p.add_argument('--threads', type=int, default=16)
    a = p.parse_args()
    if a.action == 'inspect':
        inspect(a.source)
        return
    if a.dir is None:
        raise SystemExit(f'{a.action} needs DIR')
    if a.action == 'verify':
        verify(a.dir)
        return
    lo, _, hi = a.layers.partition('-')
    fetch(a.dir, range(int(lo), int(hi or lo) + 1), a.source, a.threads)


if __name__ == '__main__':
    main()
