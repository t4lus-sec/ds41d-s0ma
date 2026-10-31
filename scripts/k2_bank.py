#!/usr/bin/env python3
"""Build or verify an EXL3 2-bit expert bank for `"experts": "k2"`: the K2 routed-expert tensors (mul1 codebook) of
target layers 0..39 from public EXL3 quants, read with HTTP range reads straight into DIR/bank-LL.bin, in the K3
bank's layout at two bits (384 records of 8,892,416 bytes a layer, 3.4 GB; 127 GiB in all).

  fetch DIR --source REPO@REV[:LAYERS] [--source ...] [--layers 0-39] [--threads 16]
               one or more sources (LAYERS like 0-17,23-39; a source without LAYERS covers the rest); each layer's
               tensors must be K2 with the mul1 codebook. There is no pinned digest (unlike the K3 bank): each
               finished layer's SHA-256 is computed from disk into its receipt, and the receipt names its source.
               Resumes: finished layers with a matching receipt are kept.
  verify DIR   recompute every bank file's SHA-256 and re-stamp its receipt (after copying or moving the bank).
  placement    write profiles/k2-p80.placement (from k3-p80; scripts/run.py needs it for "experts": "k2").

Find sources with ds41-kscan.py (which public quants hold K2 experts, by layer). Uses k3_bank.py's reader."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_bank as kb  # noqa: E402

BITS = 2
H, I = kb.H, kb.I
ORDER = kb.ORDER


def layout(bits):
    off, views = 0, {}
    for w, name, (k, n) in ORDER:
        tb = (k // 16) * (n // 16) * 16 * bits * 2
        views[name] = (off, tb)
        views[name + '_scale'] = (off + tb, 2 * (k + n))
        off += tb + 2 * (k + n)
    return views, (off + 4095) // 4096 * 4096


VIEWS, REC = layout(BITS)
assert REC == 8892416 and layout(3)[1] == kb.REC


def parse_sources(specs):
    """[(repo, rev, set of layers or None)]: the first source naming a layer serves it; one without layers the rest."""
    out = []
    for spec in specs:
        name, _, layers = spec.partition(':')
        repo, _, rev = name.partition('@')
        if not repo or not rev:
            raise SystemExit(f'--source {spec}: want REPO@REVISION[:LAYERS]')
        sel = None
        if layers:
            sel = set()
            for part in layers.split(','):
                lo, _, hi = part.partition('-')
                sel.update(range(int(lo), int(hi or lo) + 1))
        out.append((repo, rev, sel))
    return out


def source_for(sources, layer):
    for repo, rev, sel in sources:
        if sel is not None and layer in sel:
            return repo, rev
    for repo, rev, sel in sources:
        if sel is None:
            return repo, rev
    raise SystemExit(f'no --source covers layer {layer}')


def expert_record(repo, layer, e):
    p = f'layers.{layer}.ffn.experts.{e}'
    spans, record = [], bytearray(REC)
    for w, name, (k, n) in ORDER:
        if repo.meta(f'{p}.{w}.mul1') is None or repo.meta(f'{p}.{w}.mcg') is not None:
            raise ValueError(f'{repo.name} {p}.{w}: not the mul1 codebook (ds41d decodes mul1 only)')
        want = {'trellis': ('I16', [k // 16, n // 16, 16 * BITS]), 'suh': ('F16', [k]), 'svh': ('F16', [n])}
        dest = {'trellis': VIEWS[name][0], 'suh': VIEWS[name + '_scale'][0], 'svh': VIEWS[name + '_scale'][0] + 2 * k}
        for t, (dtype, shape) in want.items():
            found = repo.meta(f'{p}.{w}.{t}')
            if found is None:
                raise ValueError(f'{repo.name} {p}.{w}.{t} missing')
            shard, entry = found
            if entry['dtype'] != dtype or entry['shape'] != shape:
                raise ValueError(f'{repo.name} {p}.{w}.{t}: {entry["dtype"]} {entry["shape"]}, want {dtype} {shape} '
                                 f'(K{BITS})')
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


def receipt(path, digest, src):
    st = os.stat(path)
    return {'file': path.name, 'bytes': st.st_size, 'sha256': digest, 'mtime_ns': st.st_mtime_ns,
            'source': src, 'bits': BITS, 'verified': time.strftime('%Y-%m-%dT%H:%M:%S%z')}


def current(out, receipts, layer, src=None):
    r, path = receipts.get(str(layer)), out / f'bank-{layer:02d}.bin'
    if not r or not path.exists() or r.get('bits') != BITS or len(r.get('sha256', '')) != 64:
        return False
    if src is not None and r.get('source') != src:
        return False
    st = os.stat(path)
    return st.st_size == 384 * REC and st.st_mtime_ns == r.get('mtime_ns')


def manifest(out):
    lines = [f'# DS41MOE01 EXL3 K{BITS} mul1 bank (scripts/k2_bank.py): gate/up/down -> trellis, *_scale -> suh then svh']
    v = VIEWS
    for layer in range(40):
        path = (out / f'bank-{layer:02d}.bin').resolve()
        for e in range(384):
            eid = layer * 384 + e
            lines.append('R %d %d %d %d %d %d %d %d %d %d %d %d %d %d' % (
                eid, REC, *v['gate'], *v['gate_scale'], *v['up'], *v['up_scale'], *v['down'], *v['down_scale']))
            lines.append('E %d "%s" %d %d 0 %d %d' % (eid, path, e * REC, REC, REC, 384 * REC))
    (out / 'records-k2.tsv').write_text('\n'.join(lines) + '\n')


def fetch(out, layers, sources, threads):
    out.mkdir(parents=True, exist_ok=True)
    receipts, repos, t0, fetched = kb.load_receipts(out), {}, time.time(), 0
    for layer in layers:
        repo_name, rev = source_for(sources, layer)
        src = f'{repo_name}@{rev}'
        if current(out, receipts, layer, src):
            continue
        repo = repos.get(src) or repos.setdefault(src, kb.Repo(repo_name, rev))
        path, tmp = out / f'bank-{layer:02d}.bin', out / f'bank-{layer:02d}.partial'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, 384 * REC)

            def one(e):
                os.pwrite(fd, expert_record(repo, layer, e), e * REC)
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(one, range(384)))
            os.fsync(fd)
        finally:
            os.close(fd)
        digest = kb.file_sha256(tmp)
        os.replace(tmp, path)
        receipts[str(layer)] = receipt(path, digest, src)
        kb.save_receipts(out, receipts)
        fetched += 384 * REC
        rate = fetched / 1e6 / max(1e-9, time.time() - t0)
        print(json.dumps({'layer': layer, 'source': repo.name, 'sha256': digest[:16], 'MB_s': round(rate, 1)}),
              flush=True)
    if all(current(out, receipts, layer) for layer in range(40)):
        manifest(out)
        print(f'bank complete: {out.resolve()}/records-k2.tsv', flush=True)
    else:
        print(f'layers missing: {[l for l in range(40) if not current(out, receipts, l)]}', flush=True)


def verify(out):
    receipts = kb.load_receipts(out)

    def check(layer):
        path = out / f'bank-{layer:02d}.bin'
        r = receipts.get(str(layer))
        if not path.exists() or not r or os.stat(path).st_size != 384 * REC:
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
        receipts[str(layer)] = receipt(out / f'bank-{layer:02d}.bin', d, receipts[str(layer)].get('source', '?'))
    kb.save_receipts(out, receipts)
    manifest(out)
    print(f'verified 40 layers; receipts and records-k2.tsv rewritten in {out.resolve()}', flush=True)


def placement(root=Path(__file__).resolve().parents[1], slots=8602):
    """profiles/k2-p80.placement from k3-p80: the same pins and seeds, the target layers' dynamic budgets scaled to
    fill `slots` (8,602 two-bit slots in the K3 profile's expert VRAM)."""
    src = (root / 'profiles/k3-p80.placement').read_text().splitlines()
    pins = sum(1 for x in src if x.startswith('vram '))
    tgt = [(i, int(x.split()[1]), int(x.split()[2])) for i, x in enumerate(src)
           if x.startswith('budget ') and int(x.split()[1]) < 40]
    tot, want = sum(b for _, _, b in tgt), slots - pins
    scaled = [b * want / tot for _, _, b in tgt]
    ints = [int(x) for x in scaled]
    for j in sorted(range(len(ints)), key=lambda j: scaled[j] - ints[j], reverse=True)[:want - sum(ints)]:
        ints[j] += 1
    out = list(src)
    for (i, ly, _), b in zip(tgt, ints):
        out[i] = f'budget {ly} {b}'
    out = [x for x in out if not x.startswith('#')]
    head = ['# DS41PLACEMENT01 k2-p80 (scripts/k2_bank.py placement): k3-p80 for "experts": "k2" (EXL3 2-bit target',
            f'# experts, 8.9 MB each). Same vram pins and host seeds; the target layers\' dynamic VRAM budgets scaled from '
            f'{tot:,}', f'# to {want:,} slots ({slots:,} in all); the MTP layers\' budgets stay 0 (the drafter\'s 384 native '
            'experts are pinned in their own store).']
    dst = root / 'profiles/k2-p80.placement'
    dst.write_text('\n'.join(head + out) + '\n')
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['fetch', 'verify', 'placement'])
    p.add_argument('dir', type=Path, nargs='?')
    p.add_argument('--source', action='append', default=[], help='REPO@REVISION[:LAYERS]; repeatable')
    p.add_argument('--layers', default='0-39')
    p.add_argument('--threads', type=int, default=16)
    a = p.parse_args()
    if a.action == 'placement':
        print(f'wrote {placement()}')
        return
    if a.dir is None:
        raise SystemExit(f'{a.action} needs DIR')
    if a.action == 'verify':
        verify(a.dir)
        return
    if not a.source:
        raise SystemExit('fetch needs --source REPO@REVISION[:LAYERS] (ds41-kscan.py lists the K2 quants)')
    lo, _, hi = a.layers.partition('-')
    layers = range(int(lo), int(hi or lo) + 1)
    if not layers or layers.start < 0 or layers.stop > 40:
        raise SystemExit('--layers must lie within 0-39')
    fetch(a.dir, layers, parse_sources(a.source), a.threads)


if __name__ == '__main__':
    sys.exit(main())
